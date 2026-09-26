#pragma once

// Cost-aware draft length (LLM-740).
//
// The EMA controller sizes a draft from acceptance alone. That is the right
// question only if a drafted token is free. It is not: the drafter pays for it
// (MTP runs one head pass per token; a block drafter like DSpark pays roughly
// the same for any length) and the target pays to verify it. What we want is
// the most TOKENS PER SECOND, so pick
//
//     n* = argmax_n  (1 + E[accepted | n]) / (t_draft(n) + t_verify(n + 1))
//
// The target always yields one token per round, hence the 1 +.
//
// Everything on the right is measured online, per implementation:
//   - acceptance per draft POSITION, as a conditional rate q_k = P(k accepted
//     | k-1 accepted and k drafted). A round that drafted n and accepted a
//     tells us positions 1..a passed and a+1 failed (if a < n); positions past
//     a+1 were never tested, so they are not counted at all. That is how the
//     censoring the EMA controller works around is handled here: an untested
//     position is unknown, not a failure.
//   - time per round, as a straight line in the width: t(n) = a + b*n, fitted
//     by least squares with the same forgetting as acceptance, so it follows the
//     context as attention gets slower. Only the fitted line is used, never a
//     per-width table: a table entry goes stale as soon as the controller stops
//     visiting that width, and relative-to-mean tables drift with the choice
//     itself (the first draft of this file did that; the unit test caught it).
//   - every explore_every rounds the controller drafts one longer or one shorter
//     than its best, so the line always has more than one width to fit.
//
// An unseen position borrows the previous position's rate: optimistic, so the
// controller tries the longer draft; the measurement then corrects it. With too
// little spread to fit a slope, the slope is 0 (also optimistic toward longer).
//
// ponytail: one timing model per implementation, not per sequence. With
// --parallel 1 (our server) that is exact; with several slots decoding in one
// batch the verify time is shared and this model will blur. Split per batch
// size if multi-slot serving ever matters.

#include <algorithm>
#include <cstdint>
#include <vector>

// exponentially weighted least-squares line y = a + b*x
struct common_spec_line {
    double w = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;

    void add(double x, double y, double decay) {
        w   = w   * decay + 1;
        sx  = sx  * decay + x;
        sy  = sy  * decay + y;
        sxx = sxx * decay + x * x;
        sxy = sxy * decay + x * y;
    }
    double at(double x) const {
        if (w <= 0) {
            return 0;
        }
        const double mx = sx / w, my = sy / w;
        const double var = sxx / w - mx * mx;
        const double b   = var > 0.05 ? std::max(0.0, (sxy / w - mx * my) / var) : 0.0;
        return std::max(0.0, my + b * (x - mx));
    }
};

struct common_spec_cost {
    // one observation carries weight 1; older ones decay by this per round
    static constexpr double  decay         = 0.97;  // ~33-round memory
    static constexpr double  prior_n       = 2.0;   // pseudo-observations behind each prior
    static constexpr double  q1_prior      = 0.7;   // first-position acceptance before any data
    static constexpr int32_t explore_every = 8;     // one probe at best+-1 per this many rounds

    int32_t n_max = 0;

    // acceptance, per position k = 1..n_max (index k-1)
    std::vector<double> pass;
    std::vector<double> fail;

    common_spec_line draft_line;    // t_draft(n drafted)
    common_spec_line verify_line;   // t_verify(m = n drafted + 1)
    int32_t          n_rounds = 0;

    void reset(int32_t n_max_) {
        n_max = std::max(1, n_max_);
        pass.assign(n_max, 0.0);
        fail.assign(n_max, 0.0);
        draft_line  = {};
        verify_line = {};
        n_rounds    = 0;
    }

    // conditional acceptance at position k (1-based)
    double q(int32_t k) const {
        double prior = q1_prior;
        double qk    = q1_prior;
        for (int32_t i = 1; i <= k; ++i) {
            const double s = pass[i - 1];
            const double f = fail[i - 1];
            qk    = (s + prior_n * prior) / (s + f + prior_n);
            prior = qk;
        }
        return qk;
    }

    double t_draft(int32_t n)  const { return draft_line.at(n); }
    double t_verify(int32_t m) const { return verify_line.at(m); }

    // expected tokens per second of drafting n
    double rate(int32_t n) const {
        double e = 0.0, p = 1.0;
        for (int32_t k = 1; k <= n; ++k) {
            p *= q(k);
            e += p;
        }
        const double t = t_draft(n) + t_verify(n + 1);
        return t > 0.0 ? (1.0 + e) / t : 0.0;
    }

    // best draft length in [lo, hi]; before any timing exists, the longest
    // (optimistic, same as the EMA controller's cold start leaning upward)
    int32_t best(int32_t lo, int32_t hi) const {
        hi = std::min(hi, n_max);
        lo = std::max(1, std::min(lo, hi));
        if (verify_line.w <= 0) {
            return hi;
        }
        int32_t n_best = lo;
        double  r_best = -1.0;
        for (int32_t n = lo; n <= hi; ++n) {
            const double r = rate(n);
            if (r > r_best) {
                r_best = r;
                n_best = n;
            }
        }
        return n_best;
    }

    // the length to draft this round: best, or a +-1 probe every explore_every rounds
    int32_t choose(int32_t lo, int32_t hi) {
        const int32_t n = best(lo, hi);
        if (++n_rounds % explore_every != 0) {
            return n;
        }
        hi = std::min(hi, n_max);
        lo = std::max(1, std::min(lo, hi));
        const int32_t step = (n_rounds / explore_every) % 2 ? 1 : -1;
        return std::max(lo, std::min(hi, n + step));
    }

    // one verification round: n_drafted tokens proposed, n_accepted kept
    void observe(int32_t n_drafted, int32_t n_accepted, double t_draft_s, double t_verify_s) {
        if (n_drafted < 1 || n_max < 1) {
            return;
        }
        n_drafted  = std::min(n_drafted, n_max);
        n_accepted = std::max(0, std::min(n_accepted, n_drafted));

        for (int32_t i = 0; i < n_max; ++i) {
            pass[i] *= decay;
            fail[i] *= decay;
        }
        for (int32_t k = 1; k <= n_accepted; ++k) {
            pass[k - 1] += 1.0;
        }
        if (n_accepted < n_drafted) {
            fail[n_accepted] += 1.0;   // position n_accepted+1 was tested and failed
        }

        if (t_draft_s > 0.0 && t_verify_s > 0.0) {
            draft_line.add(n_drafted, t_draft_s, decay);
            verify_line.add(n_drafted + 1, t_verify_s, decay);
        }
    }
};
