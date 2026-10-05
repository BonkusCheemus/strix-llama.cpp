#pragma once

#include <algorithm>
#include <cstdint>

// Credit-bucket adaptive draft-depth controller.
//
// Per-sequence state: n_cur (current draft depth) plus n_bucket (accumulated
// surplus). A full accept (n_accepted >= n_cur) deposits
// max(1, n_accepted - 1); a partial accept withdraws (n_cur - n_accepted).
// The depth climbs one step when the bucket hits its per-depth cap and drops
// one step when it empties, with the over/underflow carried into the new
// level -- at most one step per update. n_cur stays within [floor, n_max].
//
// Per-depth sizing (upstream speculative-adaptive.h semantics): the cap at
// depth d is drop_pressure(d) + climb_budget(d), evaluated at the CURRENT
// depth, never global from n_max. Climbing lands at surplus + D(new depth);
// dropping lands at deficit - D(old depth) + cap(new depth).
//
// Generalized beyond the MTP-only design: this eats DSpark verify results
// (n_accepted per verify round). Feed isolation needs no type check: each
// speculative impl owns its own per-seq bucket vector and
// common_speculative_accept() updates only impl_last -- the impl that produced
// the draft -- so the DSpark impl's buckets only ever see DSpark rounds,
// never ngram-mod rounds (measured no-op/weak-signal, excluded by design).
struct common_spec_bucket {
    static constexpr int32_t floor_default = 3;

    int32_t n_cur    = 0; // current draft depth
    int32_t n_bucket = 0; // accumulated surplus
    int32_t n_max    = 0; // ceiling for n_cur
    int32_t n_floor  = floor_default;

    static int32_t drop_pressure(int32_t d) { return std::max(60, 10 * d); }
    static int32_t climb_budget (int32_t d) { return 20 + 6 * (d - 1); }
    static int32_t bucket_cap   (int32_t d) { return drop_pressure(d) + climb_budget(d); }

    int32_t cap() const { return bucket_cap(n_cur); }

    // n_start == 0 selects the cold start: n_max - 3, bounded by floor.
    void reset(int32_t n_max_new, int32_t floor_new, int32_t n_start = 0) {
        n_max   = std::max(1, n_max_new);
        n_floor = std::max(1, std::min(floor_new, n_max));
        n_cur   = (n_start > 0) ? n_start : (n_max - 3);
        n_cur   = std::min(n_max, std::max(n_floor, n_cur));
        n_bucket = drop_pressure(n_cur);
    }

    void update(int32_t n_accepted) {
        const int32_t delta = (n_accepted >= n_cur)
            ? std::max(1, n_accepted - 1)
            : (n_accepted - n_cur);
        n_bucket += delta;
        const int32_t c_old = bucket_cap(n_cur);
        if (n_bucket >= c_old) {
            if (n_cur < n_max) {
                const int32_t surplus = n_bucket - c_old;
                ++n_cur;
                // surplus carries into the new level, bounded at the new depth; one step max
                n_bucket = std::min(bucket_cap(n_cur), surplus + drop_pressure(n_cur));
            } else {
                n_bucket = c_old; // at ceiling, park at cap
            }
        } else if (n_bucket <= 0) {
            if (n_cur > n_floor) {
                const int32_t deficit = n_bucket;
                const int32_t d_old   = n_cur;
                --n_cur;
                // deficit carries into the new level, bounded at the new depth; one step max
                const int32_t c_new = bucket_cap(n_cur);
                n_bucket = std::max(0, std::min(c_new, deficit - drop_pressure(d_old) + c_new));
            } else {
                n_bucket = 0; // at floor, park at 0
            }
        }
    }
};
