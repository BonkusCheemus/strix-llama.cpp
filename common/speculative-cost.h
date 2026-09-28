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
//   - time per round, per WIDTH: one running average for each draft length n
//     (t_draft) and each verify width m (t_verify). Verify time is not a line in
//     the width on every backend: on HIP with turbo KV, widths 1-2 run the VEC
//     FA kernel and widths 3+ convert the whole KV cache to f16 first, a step
//     that grows with depth. A width never measured borrows its nearest measured
//     neighbour (flat, so an untried longer draft looks no dearer: optimistic,
//     it gets tried, the measurement corrects it).
//   - every explore_every rounds the controller drafts one longer or one shorter
//     than its best; every 4th such probe goes to the width measured longest ago
//     instead, so entries left behind by a context change get refreshed.
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

// running average of a time per integer width, with nearest-neighbour fill
struct common_spec_width_times {
    static constexpr double alpha = 0.3;   // weight of the newest observation

    std::vector<double>  t;       // average seconds per width (index = width)
    std::vector<int64_t> seen_at; // round of the last observation, -1 = never

    void reset(int32_t w_max) {
        t.assign(w_max + 1, 0.0);
        seen_at.assign(w_max + 1, -1);
    }
    bool any() const {
        return std::any_of(seen_at.begin(), seen_at.end(), [](int64_t r) { return r >= 0; });
    }
    void add(int32_t w, double y, int64_t round) {
        if (w < 0 || w >= (int32_t) t.size()) {
            return;
        }
        t[w]       = seen_at[w] < 0 ? y : t[w] + alpha * (y - t[w]);
        seen_at[w] = round;
    }
    double at(int32_t w) const {
        if (w < 0 || w >= (int32_t) t.size()) {
            return 0.0;
        }
        if (seen_at[w] >= 0) {
            return t[w];
        }
        int32_t lo = w - 1, hi = w + 1;
        while (lo >= 0 && seen_at[lo] < 0) --lo;
        while (hi < (int32_t) t.size() && seen_at[hi] < 0) ++hi;
        const bool has_lo = lo >= 0, has_hi = hi < (int32_t) t.size();
        if (has_lo && has_hi) {
            return t[lo] + (t[hi] - t[lo]) * (w - lo) / (double) (hi - lo);
        }
        return has_lo ? t[lo] : (has_hi ? t[hi] : 0.0);
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

    common_spec_width_times draft_t;   // t_draft(n drafted)
    common_spec_width_times verify_t;  // t_verify(m = n drafted + 1)
    int32_t                 n_rounds = 0;

    void reset(int32_t n_max_) {
        n_max = std::max(1, n_max_);
        pass.assign(n_max, 0.0);
        fail.assign(n_max, 0.0);
        draft_t.reset(n_max);
        verify_t.reset(n_max + 1);
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

    double t_draft(int32_t n)  const { return draft_t.at(n); }
    double t_verify(int32_t m) const { return verify_t.at(m); }

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
        if (!verify_t.any()) {
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

    // the length to draft this round: best, a +-1 probe every explore_every
    // rounds, and every 4th probe the width measured longest ago
    int32_t choose(int32_t lo, int32_t hi) {
        const int32_t n = best(lo, hi);
        if (++n_rounds % explore_every != 0) {
            return n;
        }
        hi = std::min(hi, n_max);
        lo = std::max(1, std::min(lo, hi));
        const int32_t probe = n_rounds / explore_every;
        if (probe % 4 == 0) {
            int32_t n_old = lo;
            for (int32_t k = lo; k <= hi; ++k) {
                if (verify_t.seen_at[k + 1] < verify_t.seen_at[n_old + 1]) {
                    n_old = k;
                }
            }
            return n_old;
        }
        const int32_t step = probe % 2 ? 1 : -1;
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
            draft_t.add(n_drafted, t_draft_s, n_rounds);
            verify_t.add(n_drafted + 1, t_verify_s, n_rounds);
        }
    }
};
