// LLM-740: the cost-aware draft controller finds the draft length a known
// world rewards most, from the same noisy observations the server gives it.
// Build: g++ -std=c++17 -O2 -I common tests/test-speculative-cost.cpp -o /tmp/tsc && /tmp/tsc

#include "speculative-cost.h"

#include <cassert>
#include <cstdio>
#include <random>

struct world {
    std::vector<double> q;              // true conditional acceptance per position
    double d0, d1, v0, v1;              // t_draft = d0 + d1*n, t_verify = v0 + v1*m (seconds)
    double noise;                       // relative timing noise

    double rate(int n) const {
        double e = 0, p = 1;
        for (int k = 1; k <= n; ++k) { p *= q[k - 1]; e += p; }
        return (1 + e) / (d0 + d1 * n + v0 + v1 * (n + 1));
    }
    int best(int hi) const {
        int nb = 1;
        for (int n = 2; n <= hi; ++n) if (rate(n) > rate(nb)) nb = n;
        return nb;
    }
};

// run the controller for `rounds`; return the true rate it achieved over the last
// 200 rounds (probe rounds included, as the server pays for them) / the optimum
static double run(const world & w, int n_max, int rounds, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    common_spec_cost c;
    c.reset(n_max);
    double got = 0;
    for (int r = 0; r < rounds; ++r) {
        const int n = c.choose(1, n_max);
        if (r >= rounds - 200) got += w.rate(n);
        int a = 0;
        while (a < n && u(rng) < w.q[a]) ++a;
        const double jit = 1 + w.noise * (2 * u(rng) - 1);
        c.observe(n, a, (w.d0 + w.d1 * n) * jit, (w.v0 + w.v1 * (n + 1)) * jit);
    }
    return got / 200 / w.rate(w.best(n_max));
}

int main() {
    // MTP-like: each drafted token costs a head pass; acceptance collapses after
    // two positions (1337Hero's measured 0.83/0.38/0.20 shape). Short drafts win.
    world mtp{{0.83, 0.38, 0.20, 0.15, 0.1, 0.1, 0.1, 0.1}, 0.0005, 0.004, 0.060, 0.002, 0.05};
    // DSpark-like: one block pass for any length; acceptance decays slowly.
    world dsp{{0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55}, 0.012, 0.0002, 0.060, 0.002, 0.05};
    // verify gets expensive per token (long context): drafting long must lose.
    world dear{{0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3}, 0.001, 0.003, 0.050, 0.020, 0.05};

    const world * ws[] = {&mtp, &dsp, &dear};
    const char * names[] = {"mtp", "dspark", "dear-verify"};
    for (int i = 0; i < 3; ++i) {
        double worst = 1;
        for (unsigned s = 1; s <= 20; ++s) worst = std::min(worst, run(*ws[i], 8, 400, s));
        std::printf("%-12s optimum n=%d  worst achieved rate / optimum over 20 seeds = %.3f\n",
                    names[i], ws[i]->best(8), worst);
        assert(worst >= 0.95);
    }

    // censoring: rounds that accept everything must never LOWER the estimate
    // of a position they did not test
    common_spec_cost c;
    c.reset(8);
    for (int r = 0; r < 50; ++r) c.observe(2, 2, 0.001, 0.05);
    assert(c.q(3) >= common_spec_cost::q1_prior - 1e-9);
    assert(c.q(1) > 0.95 && c.q(2) > 0.95);

    // n_min binds, n_cfg caps
    assert(c.best(3, 8) >= 3);
    assert(c.best(1, 2) <= 2);

    std::printf("ok\n");
    return 0;
}
