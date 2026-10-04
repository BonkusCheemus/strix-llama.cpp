/*
 * Test: GGML_OP_GATED_DELTA_NET emit_mode==1 (replay ingredients) vs emit_mode==0 (full snapshots).
 *
 * Proves the core DRC-phase-1 claim: replaying the small per-token (k, v, g, beta)
 * ingredients captured by emit_mode==1 through a fresh K==1, emit_mode==0 call
 * reconstructs the exact same recurrent state as a from-scratch full run would
 * have produced at that position -- without needing q or a full S_v x S_v state
 * snapshot for every retained step.
 *
 * Scenario: N=6 total tokens, a checkpoint taken after c=2 tokens, and a "draft"
 * of r-c=3 more tokens (positions 2,3,4) that need to be replayed on top of the
 * checkpoint to reach the state after r=5 tokens. This mirrors the real usage:
 * checkpoint = last accepted position, replay = tokens between the checkpoint
 * and the new rollback point.
 *
 * kda=false and n_seqs=1, H_k==H_v (no GQA broadcast) for this first proof --
 * those are straightforward generalizations of the same ingredient encoding,
 * not additional risk to the core claim.
 *
 * This is the only test that exercises emit_mode==1 end-to-end through a real
 * replay reconstruction; test-backend-ops.cpp's GATED_DELTA_NET cases only
 * sanity-check that emit_mode==1 builds/runs/produces finite output.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>

struct ggml_backend_ptr {
    ggml_backend_t ptr = nullptr;
    ~ggml_backend_ptr() { if (ptr) ggml_backend_free(ptr); }
    ggml_backend_t get() { return ptr; }
};

struct ggml_backend_buffer_ptr {
    ggml_backend_buffer_t ptr = nullptr;
    ~ggml_backend_buffer_ptr() { if (ptr) ggml_backend_buffer_free(ptr); }
    ggml_backend_buffer_t get() { return ptr; }
};

static float max_abs(const float * a, const float * b, size_t n) {
    float m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, fabsf(a[i] - b[i]));
    return m;
}

static uint64_t g_rng = 0x1234567890abcdefULL;
static float rnd(float lo, float hi) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = (float) ((g_rng >> 33) & 0xffffff) / (float) 0x1000000;
    return lo + u * (hi - lo);
}

// Runs one ggml_gated_delta_net call on `backend` with the given host-side inputs
// (each sized for n_tokens tokens, H heads, S_v head width, n_seqs==1, no GQA) and
// returns the full output buffer (attn scores followed by K snapshot/ingredient rows).
static std::vector<float> run_gdn(
        ggml_backend_t backend,
        const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
        const std::vector<float> & g, const std::vector<float> & beta, const std::vector<float> & state,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t K, int32_t emit_mode,
        float out_sentinel = 0.0f) {
    ggml_init_params iparams = { /*.mem_size=*/ 1024 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * tq    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1);
    ggml_tensor * tk    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1);
    ggml_tensor * tv    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1);
    ggml_tensor * tg    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H, n_tokens, 1); // kda=false: scalar gate
    ggml_tensor * tbeta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H, n_tokens, 1);
    ggml_tensor * tstate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, 1);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tbeta, tstate, K, emit_mode);
    ggml_build_forward_expand(gf, out);

    // allocate AFTER the graph (including its output tensor) is fully built, so `out` gets
    // real backing memory too -- allocating before this point leaves out->data == NULL.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_ptr buf;
    buf.ptr = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    ggml_backend_tensor_set(tq,     q.data(),     0, q.size()     * sizeof(float));
    ggml_backend_tensor_set(tk,     k.data(),     0, k.size()     * sizeof(float));
    ggml_backend_tensor_set(tv,     v.data(),     0, v.size()     * sizeof(float));
    ggml_backend_tensor_set(tg,     g.data(),     0, g.size()     * sizeof(float));
    ggml_backend_tensor_set(tbeta,  beta.data(),  0, beta.size()  * sizeof(float));
    ggml_backend_tensor_set(tstate, state.data(), 0, state.size() * sizeof(float));

    // OPT-57 slot-layout probe: prefill the WHOLE output with a sentinel so a slot that
    // the op never wrote is distinguishable from a slot it wrote with a plausible value.
    // Written AFTER the inputs and BEFORE compute, so the op overwrites only what it
    // actually produces. Default 0.0f leaves the pre-existing behaviour untouched.
    if (out_sentinel != 0.0f) {
        std::vector<float> fill(ggml_nelements(out), out_sentinel);
        ggml_backend_tensor_set(out, fill.data(), 0, fill.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: gated_delta_net graph compute failed\n");
        ggml_free(ctx);
        return {};
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_free(ctx);
    return result;
}

int main() {
    const int64_t S_v = 16;
    const int64_t H   = 4;
    const int64_t N   = 6; // total tokens
    const int64_t c   = 2; // checkpoint position (tokens accepted so far)
    const int64_t r   = 5; // rollback target (ground truth: state after r tokens)

    ggml_backend_ptr backend;
    backend.ptr = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend.get()) {
        fprintf(stderr, "FAIL: could not init CPU backend\n");
        return 1;
    }

    // full-sequence host buffers, shared across all calls below (slices taken as needed)
    std::vector<float> q_full(S_v * H * N), k_full(S_v * H * N), v_full(S_v * H * N);
    std::vector<float> g_full(H * N), beta_full(H * N);
    for (auto & x : q_full) x = rnd(-1.0f, 1.0f);
    for (auto & x : k_full) x = rnd(-1.0f, 1.0f);
    for (auto & x : v_full) x = rnd(-0.3f, 5.0f);
    for (auto & x : g_full) x = rnd(-5.0f, -1e-4f);   // gate is exp(g), g < 0 like real usage
    for (auto & x : beta_full) x = rnd(0.0f, 1.0f);
    std::vector<float> s0(S_v * S_v * H);
    for (auto & x : s0) x = rnd(-0.1f, 0.1f);

    auto slice = [&](const std::vector<float> & full, int64_t row_width, int64_t t0, int64_t t1) {
        return std::vector<float>(full.begin() + t0 * row_width, full.begin() + t1 * row_width);
    };

    // --- 1. ground truth: full run over the first r tokens, K=1, emit_mode=0 ---
    std::vector<float> out_truth = run_gdn(backend.get(),
        slice(q_full, S_v * H, 0, r), slice(k_full, S_v * H, 0, r), slice(v_full, S_v * H, 0, r),
        slice(g_full, H, 0, r), slice(beta_full, H, 0, r), s0,
        S_v, H, /*n_tokens=*/r, /*K=*/1, /*emit_mode=*/0);
    const int64_t attn_r = S_v * H * r;
    std::vector<float> state_truth(out_truth.begin() + attn_r, out_truth.begin() + attn_r + S_v * S_v * H);

    // --- 2. checkpoint: full run over the first c tokens, K=1, emit_mode=0 ---
    std::vector<float> out_ckpt = run_gdn(backend.get(),
        slice(q_full, S_v * H, 0, c), slice(k_full, S_v * H, 0, c), slice(v_full, S_v * H, 0, c),
        slice(g_full, H, 0, c), slice(beta_full, H, 0, c), s0,
        S_v, H, /*n_tokens=*/c, /*K=*/1, /*emit_mode=*/0);
    const int64_t attn_c = S_v * H * c;
    std::vector<float> state_ckpt(out_ckpt.begin() + attn_c, out_ckpt.begin() + attn_c + S_v * S_v * H);

    // --- 3. ingredients: full run over all N tokens, K=N, emit_mode=1 (captures every position) ---
    std::vector<float> out_ingr = run_gdn(backend.get(),
        q_full, k_full, v_full, g_full, beta_full, s0,
        S_v, H, /*n_tokens=*/N, /*K=*/N, /*emit_mode=*/1);
    const int64_t attn_N = S_v * H * N;
    const int64_t snap_size = 4 * S_v * H; // ingredient rows per slot (k,v,g,beta x S_v x H)

    // emit_mode=1's trailing final-state block should match a from-scratch full-N-token run,
    // independent of the replay mechanism below (this checks the Phase 2 op-contract extension).
    std::vector<float> out_truth_N = run_gdn(backend.get(),
        q_full, k_full, v_full, g_full, beta_full, s0,
        S_v, H, /*n_tokens=*/N, /*K=*/1, /*emit_mode=*/0);
    std::vector<float> state_truth_N(out_truth_N.begin() + attn_N, out_truth_N.begin() + attn_N + S_v * S_v * H);
    const int64_t final_state_offset = attn_N + N * snap_size;
    std::vector<float> final_state_ingr(out_ingr.begin() + final_state_offset, out_ingr.begin() + final_state_offset + S_v * S_v * H);
    const float max_diff_final = max_abs(final_state_ingr.data(), state_truth_N.data(), final_state_ingr.size());
    printf("[gdn-ingredient-replay] emit_mode=1 trailing final-state vs from-scratch N-token run: max_abs=%.3e\n", max_diff_final);
    if (max_diff_final > 1e-4f) {
        fprintf(stderr, "FAIL: emit_mode=1's trailing final-state block does not match a from-scratch run (max_abs=%.3e)\n", max_diff_final);
        return 1;
    }

    // extract ingredients for tokens [c, r) in forward order, building a synthetic replay batch
    const int64_t n_replay = r - c;
    std::vector<float> q_replay(S_v * H * n_replay, 0.0f); // dummy: q doesn't affect state
    std::vector<float> k_replay(S_v * H * n_replay);
    std::vector<float> v_replay(S_v * H * n_replay);
    std::vector<float> g_replay(H * n_replay);
    std::vector<float> beta_replay(H * n_replay);

    for (int64_t rt = 0; rt < n_replay; rt++) {
        const int64_t t    = c + rt;   // original token index
        const int64_t slot = t;        // emit_mode=1 chronological order; K=N here so slot == t directly
        for (int64_t h = 0; h < H; h++) {
            const float * ingr = &out_ingr[attn_N + slot * snap_size + h * 4 * S_v];
            std::copy(ingr,               ingr + S_v, &k_replay[rt * S_v * H + h * S_v]);
            std::copy(ingr + S_v,         ingr + 2*S_v, &v_replay[rt * S_v * H + h * S_v]);
            g_replay[rt * H + h]    = ingr[2 * S_v]; // broadcast scalar: any of the S_v copies works
            beta_replay[rt * H + h] = ingr[3 * S_v];
        }
    }

    // --- 4. replay: checkpoint state + extracted ingredients, K=1, emit_mode=0 ---
    std::vector<float> out_replay = run_gdn(backend.get(),
        q_replay, k_replay, v_replay, g_replay, beta_replay, state_ckpt,
        S_v, H, /*n_tokens=*/n_replay, /*K=*/1, /*emit_mode=*/0);
    const int64_t attn_replay = S_v * H * n_replay;
    std::vector<float> state_replay(out_replay.begin() + attn_replay, out_replay.begin() + attn_replay + S_v * S_v * H);

    const float max_diff = max_abs(state_replay.data(), state_truth.data(), state_replay.size());
    printf("[gdn-ingredient-replay] N=%lld c=%lld r=%lld S_v=%lld H=%lld: state_replay vs state_truth max_abs=%.3e\n",
           (long long) N, (long long) c, (long long) r, (long long) S_v, (long long) H, max_diff);

    if (max_diff > 1e-4f) {
        fprintf(stderr, "FAIL: ingredient replay did not reconstruct the ground-truth state (max_abs=%.3e)\n", max_diff);
        return 1;
    }

    printf("OK: ingredient replay reconstructs the ground-truth full-snapshot state\n");

    // ---------------------------------------------------------------------
    // OPT-57 slot-layout probe (REVIEW67). Question: with N < K, WHICH ring
    // slots does the op actually write, and whose ingredients do they hold?
    //
    // This exists because the emit formula and the scatter disagree:
    //   ops.cpp:11172 / gated_delta_net.cu:216  target_slot = t - (n_tokens - K)
    //   delta-net-base.cpp:727-737             copies the FIRST n_written rows
    // and ops.cpp:11082-11083 documents a third answer ("only slots 0..n_tokens-1").
    // The arithmetic alone cannot settle which is right, so this measures it:
    // prefill the output with a sentinel, run emit_mode=1, then for each slot
    // report whether it was written at all and which token's k it holds.
    //
    // Nothing here asserts a conclusion -- it prints what the buffer contains.
    // ---------------------------------------------------------------------
    {
        const int64_t P_S_v = 16;
        const int64_t P_H   = 4;
        const int64_t P_N   = 3;      // N < K, the short-batch case
        const int64_t P_K   = 8;
        const float   SENT  = -987654.0f;

        ggml_backend_ptr pbe;
        pbe.ptr = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!pbe.get()) {
            fprintf(stderr, "FAIL: slot-probe could not init CPU backend\n");
            return 1;
        }

        // Every ingredient component is encoded so a written slot can be attributed on
        // k, v, g AND beta, and the 4-row layout (k, v, g, beta) can be asserted. The
        // previous revision of this probe matched on k only, which cannot distinguish a
        // correctly-packed block from one whose rows are mis-ordered.
        //
        // Bases are far apart so a mis-packed block cannot accidentally match the wrong
        // component:  k ~ 1e3,  v ~ 2e3,  g ~ 3e3,  beta ~ 4e3, then +t and +s.
        std::vector<float> pq(P_S_v * P_H * P_N), pk(P_S_v * P_H * P_N), pv(P_S_v * P_H * P_N);
        std::vector<float> pg(P_H * P_N), pbeta(P_H * P_N), ps0(P_S_v * P_S_v * P_H);
        for (int64_t t = 0; t < P_N; ++t) {
            for (int64_t h = 0; h < P_H; ++h) {
                for (int64_t s = 0; s < P_S_v; ++s) {
                    const size_t i = s + P_S_v * (h + P_H * t);
                    pk[i] = 1000.0f + 10.0f * (float) t + (float) s;
                    pv[i] = 2000.0f + 10.0f * (float) t + (float) s;
                    pq[i] = rnd(-1.0f, 1.0f);
                }
                pg[t * P_H + h] = 3000.0f + 10.0f * (float) t + (float) h;
                pbeta[t * P_H + h] = 4000.0f + 10.0f * (float) t + (float) h;
            }
        }
        for (auto & x : ps0) x = rnd(-0.1f, 0.1f);

        const std::vector<float> pout = run_gdn(pbe.get(), pq, pk, pv, pg, pbeta, ps0,
                                                P_S_v, P_H, /*n_tokens=*/P_N, /*K=*/P_K,
                                                /*emit_mode=*/1, /*out_sentinel=*/SENT);

        const int64_t attn_elems  = P_S_v * P_H * P_N;
        const int64_t slot_stride = 4 * P_S_v * P_H;      // snap_rows_per_head(4) * S_v * H
        printf("\n[slot-probe] N=%lld K=%lld S_v=%lld H=%lld sentinel=%.0f\n",
               (long long) P_N, (long long) P_K, (long long) P_S_v, (long long) P_H, SENT);

        // ASASSERT: this block is now a regression, not an observation log. The previous
        // revision only printed what the buffer held, so any change in slot placement
        // would still have exited 0. Written-slot identity, untouched-slot identity and
        // the all-4-rows-exact layout are each asserted and each fail the test.
        std::vector<int64_t> written_slots, intact_slots;
        bool layout_all_exact = true;

        for (int64_t slot = 0; slot < P_K; ++slot) {
            int64_t heads_written = 0;
            for (int64_t h = 0; h < P_H; ++h) {
                const size_t base = (size_t) (attn_elems + slot * slot_stride + h * (4 * P_S_v));
                if (base + 4 * P_S_v > pout.size()) { continue; }

                bool untouched = true;
                for (int64_t j = 0; j < 4 * P_S_v; ++j) {
                    if (pout[base + j] != SENT) { untouched = false; break; }
                }
                if (untouched) { continue; }   // slot/head not written by this call
                ++heads_written;

                // Attribute the block on ALL FOUR components and assert the 4-row layout.
                //
                // CORRECTION: the previous revision of this matcher expected EVERY
                // component to vary with s. That is wrong for g and beta. ops.cpp:11179-11194
                // emits them BROADCAST -- `ingr_o[2*S_v + i] = g_d[0]` and
                // `ingr_o[3*S_v + i] = beta_val` for every i in S_v -- so the expected value
                // is base + 10*t + h, constant across s. Expecting base + 10*t + s produced
                // errors of exactly max(h, S_v-1-h), i.e. 15,14,13,12 for heads 0..3, which
                // is what the previous run printed. That was a defect in this PROBE, not in
                // the emit code: k, v, g and beta are all read at the same t (ops.cpp:11120-11125)
                // and target_slot depends only on t, so a slot's block is internally
                // consistent by construction.
                struct { const char * name; float base; int64_t row; bool broadcast; } comp[4] = {
                    {"k",    1000.0f, 0, false},
                    {"v",    2000.0f, 1, false},
                    {"g",    3000.0f, 2, true },
                    {"beta", 4000.0f, 3, true },
                };
                int64_t t_of[4] = { -1, -1, -1, -1 };
                float   e_of[4] = { 0, 0, 0, 0 };
                for (int c = 0; c < 4; ++c) {
                    for (int64_t t = 0; t < P_N; ++t) {
                        float e = 0.0f;
                        for (int64_t s = 0; s < P_S_v; ++s) {
                            const float want = comp[c].broadcast
                                ? comp[c].base + 10.0f * (float) t + (float) h
                                : comp[c].base + 10.0f * (float) t + (float) s;
                            e = std::max(e, std::fabs(pout[base + comp[c].row * P_S_v + s] - want));
                        }
                        if (t_of[c] < 0 || e < e_of[c]) { e_of[c] = e; t_of[c] = t; }
                    }
                }
                const bool all_agree = (t_of[0] == t_of[1] && t_of[1] == t_of[2] && t_of[2] == t_of[3]);
                const bool all_clean = (e_of[0] == 0.0f && e_of[1] == 0.0f &&
                                        e_of[2] == 0.0f && e_of[3] == 0.0f);
                printf("[slot-probe] slot %lld head %lld: WRITTEN  k->t%lld v->t%lld g->t%lld beta->t%lld"
                       "  layout=%s  exact=%s\n",
                       (long long) slot, (long long) h,
                       (long long) t_of[0], (long long) t_of[1],
                       (long long) t_of[2], (long long) t_of[3],
                       all_agree ? "all 4 rows agree on token t" : "g/beta DISAGREE with k/v on token",
                       all_clean ? "yes" : "no");
                if (!all_agree || !all_clean) {
                    printf("[slot-probe]   DIAG err k=%.3e v=%.3e g=%.3e beta=%.3e\n",
                           e_of[0], e_of[1], e_of[2], e_of[3]);
                    layout_all_exact = false;
                }
            }
            if (heads_written > 0) { written_slots.push_back(slot); }
        }

        for (int64_t slot = 0; slot < P_K; ++slot) {
            bool any = false;
            for (int64_t h = 0; h < P_H && !any; ++h) {
                const size_t base = (size_t) (attn_elems + slot * slot_stride + h * (4 * P_S_v));
                for (int64_t j = 0; j < 4 * P_S_v; ++j) {
                    if (pout[base + j] != SENT) { any = true; break; }
                }
            }
            if (!any) { intact_slots.push_back(slot); }
        }

        // Expected slot map, from ops.cpp:11172 / gated_delta_net.cu:216:
        //   target_slot = t - (n_tokens - K)
        // With P_N=3, P_K=8 that puts token t at slot t+5, i.e. slots 5,6,7 -- NOT 0,1,2.
        // This is the disagreement with delta-net-base.cpp:727-737, which copies the FIRST
        // n_written rows (ring slots 0,1,2). Asserting 5,6,7 pins the producer's behaviour
        // so a future change to either side breaks this test rather than printing a new
        // observation and exiting 0.
        std::vector<int64_t> expect_written, expect_intact;
        for (int64_t t = 0; t < P_N; ++t) { expect_written.push_back(t - (P_N - P_K)); }
        for (int64_t s = 0; s < P_K; ++s) {
            if (std::find(expect_written.begin(), expect_written.end(), s) == expect_written.end()) {
                expect_intact.push_back(s);
            }
        }

        const bool slots_ok = (written_slots == expect_written) && (intact_slots == expect_intact);
        printf("[slot-probe] ASSERT written=%lld (expect %lld) intact=%lld (expect %lld)"
               "  all-4-rows-exact=%s  slotmap=%s\n",
               (long long) written_slots.size(), (long long) expect_written.size(),
               (long long) intact_slots.size(), (long long) expect_intact.size(),
               layout_all_exact ? "yes" : "NO",
               slots_ok ? "matches t-(n_tokens-K)" : "MISMATCH");
        if (!slots_ok || !layout_all_exact) {
            fprintf(stderr, "FAIL: slot layout. expected written slots [");
            for (size_t i = 0; i < expect_written.size(); ++i) fprintf(stderr, "%lld ", (long long) expect_written[i]);
            fprintf(stderr, "] got [");
            for (size_t i = 0; i < written_slots.size(); ++i) fprintf(stderr, "%lld ", (long long) written_slots[i]);
            fprintf(stderr, "]; expected intact [");
            for (size_t i = 0; i < expect_intact.size(); ++i) fprintf(stderr, "%lld ", (long long) expect_intact[i]);
            fprintf(stderr, "] got [");
            for (size_t i = 0; i < intact_slots.size(); ++i) fprintf(stderr, "%lld ", (long long) intact_slots[i]);
            fprintf(stderr, "]; all-4-rows-exact=%s\n", layout_all_exact ? "yes" : "no");
            return 1;
        }
        printf("[slot-probe] OK: %lld slots written at t-(n_tokens-K), %lld sentinel-intact,"
               " all 4 component rows exact at every written slot/head\n",
               (long long) written_slots.size(), (long long) intact_slots.size());
    }

    // ---------------------------------------------------------------------
    // OPT-57 accepted-1 replay from a VALID short span (REVIEW67 / 714d93c7).
    //
    // Producer: N=3, K=8, emit_mode=1. The op writes slots 5,6,7 -- target_slot =
    // t-(n_tokens-K) -- holding tokens t=0,1,2, which are the last three COMPLETED
    // decodes. Those three real ingredient blocks are extracted into packed views and
    // replayed through a fresh K=1, emit_mode=0 call from the SAME initial state.
    // Ground truth: one from-scratch K=1, emit_mode=0 run over all 3 tokens.
    //
    // Values are physically valid here -- g finite and negative, beta in [0,1] -- unlike
    // the slot-layout probe above, whose far-apart magic bases (1e3..4e3) exist only to
    // attribute a slot to a token. exp(3000) is not a gate any real decode produces and
    // is useless as replay truth.
    //
    // The sentinel is retained deliberately: extraction asserts the three source slots
    // were actually written, so an unwritten slot can never be read in as an ingredient.
    // No ring slot is hand-invented and no uninitialised data is compared.
    // ---------------------------------------------------------------------
    {
        const int64_t A_S_v = 16;
        const int64_t A_H   = 4;
        const int64_t A_N   = 3;   // short span: the last 3 completed decodes
        const int64_t A_K   = 8;   // ring is larger than the span -> producer writes 5,6,7
        const float   A_SENT = -987654.0f;

        ggml_backend_ptr abe;
        abe.ptr = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!abe.get()) {
            fprintf(stderr, "FAIL: accepted-1 replay could not init CPU backend\n");
            return 1;
        }

        std::vector<float> aq(A_S_v * A_H * A_N), ak(A_S_v * A_H * A_N), av(A_S_v * A_H * A_N);
        std::vector<float> ag(A_H * A_N), ab(A_H * A_N), as0(A_S_v * A_S_v * A_H);
        for (auto & x : aq) x = rnd(-1.0f, 1.0f);
        for (auto & x : ak) x = rnd(-1.0f, 1.0f);
        for (auto & x : av) x = rnd(-0.3f, 5.0f);
        for (auto & x : ag) x = rnd(-5.0f, -1e-4f);   // finite negative gate, as real usage
        for (auto & x : ab) x = rnd(0.0f, 1.0f);     // beta bounded to [0,1]
        for (auto & x : as0) x = rnd(-0.1f, 0.1f);

        // CORRECTION (kurumi 3cbd46f6, acceptance WITHDRAWN and re-issued):
        // the previous revision of this block called itself "accepted-1" but called
        // truth AND replay with n_tokens = A_N = 3 and extracted all three slots. That
        // proves a FULL 3-token reconstruction, not the requested retained-1 / rejected-2.
        // A label is not a test. What is now asserted is the narrow claim:
        //   producer  N=3, K=8  ->  slots 5,6,7
        //   accepted  R=1       ->  token t=0, which lives in slot 5
        //   truth  = 1 token sliced from the ORIGINAL input arrays (t=0)
        //   replay = 1 token extracted from REAL produced slot 5 only
        // Slots 6 and 7 (the rejected tokens) are deliberately never read, and every
        // offset below is computed from A_R=1, NOT from the producer's 3.
        const int64_t A_R        = 1;                     // ACCEPTED tokens, explicit
        const int64_t a_attn     = A_S_v * A_H * A_N;     // producer score block (3 tokens)
        const int64_t t_attn     = A_S_v * A_H * A_R;     // score block for the 1-token calls
        const int64_t a_slot_stride = 4 * A_S_v * A_H;
        const int64_t a_first_slot = A_K - A_N;           // 5: where t=0 lands
        const int64_t state_elems  = A_S_v * A_S_v * A_H;

        // ground truth: the SAME 1 token taken from the original inputs, from-scratch
        std::vector<float> tq(aq.begin(), aq.begin() + A_S_v * A_H * A_R);
        std::vector<float> tk(ak.begin(), ak.begin() + A_S_v * A_H * A_R);
        std::vector<float> tv(av.begin(), av.begin() + A_S_v * A_H * A_R);
        std::vector<float> tg(ag.begin(), ag.begin() + A_H * A_R);
        std::vector<float> tb(ab.begin(), ab.begin() + A_H * A_R);
        std::vector<float> a_truth = run_gdn(abe.get(), tq, tk, tv, tg, tb, as0,
                                             A_S_v, A_H, /*n_tokens=*/A_R, /*K=*/1, /*emit_mode=*/0);

        // producer: UNCHANGED -- N=3, K=8, emit_mode=1 -> ingredients land in slots 5,6,7
        std::vector<float> a_out = run_gdn(abe.get(), aq, ak, av, ag, ab, as0,
                                           A_S_v, A_H, /*n_tokens=*/A_N, /*K=*/A_K,
                                           /*emit_mode=*/1, /*out_sentinel=*/A_SENT);

        // extract ONLY the ACCEPTED token, from the ONE real produced slot it lives in
        std::vector<float> rq(A_S_v * A_H * A_R, 0.0f);   // q does not affect the state
        std::vector<float> rk(A_S_v * A_H * A_R), rv(A_S_v * A_H * A_R);
        std::vector<float> rg(A_H * A_R), rb(A_H * A_R);
        int64_t replay_count = 0;
        for (int64_t rt = 0; rt < A_R; ++rt) {
            const int64_t slot = a_first_slot + rt;      // slot 5 only
            for (int64_t h = 0; h < A_H; ++h) {
                const size_t base = (size_t) (a_attn + slot * a_slot_stride + h * (4 * A_S_v));
                // The negative control that keeps this honest: an unwritten slot must
                // never be readable as an ingredient.
                for (int64_t j = 0; j < 4 * A_S_v; ++j) {
                    if (a_out[base + j] == A_SENT) {
                        fprintf(stderr, "FAIL: accepted replay would read an UNWRITTEN slot %lld head %lld\n",
                                (long long) slot, (long long) h);
                        return 1;
                    }
                }
                const float * in = &a_out[base];
                std::copy(in,         in + A_S_v,   &rk[rt * A_S_v * A_H + h * A_S_v]);
                std::copy(in + A_S_v, in + 2*A_S_v, &rv[rt * A_S_v * A_H + h * A_S_v]);
                rg[rt * A_H + h] = in[2 * A_S_v];   // broadcast scalar row
                rb[rt * A_H + h] = in[3 * A_S_v];
            }
            ++replay_count;
        }
        if (replay_count != A_R) {
            fprintf(stderr, "FAIL: expected replay_count == %lld, extracted %lld\n",
                    (long long) A_R, (long long) replay_count);
            return 1;
        }

        // replay: same initial state, extracted ingredients, 1 token, K=1, emit_mode=0
        std::vector<float> a_rep = run_gdn(abe.get(), rq, rk, rv, rg, rb, as0,
                                           A_S_v, A_H, /*n_tokens=*/A_R, /*K=*/1, /*emit_mode=*/0);

        // Length and finiteness BEFORE any max_abs. A NaN compares false against every
        // threshold, so max_abs alone can never make a broken state look clean; a run
        // that goes quiet here would otherwise report 0.000e+00 for the wrong reason.
        if ((int64_t) a_truth.size() != t_attn + state_elems) {
            fprintf(stderr, "FAIL: truth buffer is %zu elems, expected %lld (1-token score block + state)\n",
                    a_truth.size(), (long long) (t_attn + state_elems));
            return 1;
        }
        if ((int64_t) a_rep.size() != t_attn + state_elems) {
            fprintf(stderr, "FAIL: replay buffer is %zu elems, expected %lld (1-token score block + state)\n",
                    a_rep.size(), (long long) (t_attn + state_elems));
            return 1;
        }
        const float * truth_state = a_truth.data() + t_attn;
        const float * rep_state   = a_rep.data()   + t_attn;
        for (int64_t i = 0; i < state_elems; ++i) {
            if (!std::isfinite(truth_state[i]) || !std::isfinite(rep_state[i])) {
                fprintf(stderr, "FAIL: non-finite state entry at %lld (truth=%.6g replay=%.6g)\n",
                        (long long) i, truth_state[i], rep_state[i]);
                return 1;
            }
        }

        const float md = max_abs(rep_state, truth_state, (size_t) state_elems);
        printf("[accepted replay] producer N=%lld K=%lld (slots %lld..%lld) | ACCEPTED R=%lld"
               " (slot %lld only, t=0) | g<0 finite beta in[0,1]\n",
               (long long) A_N, (long long) A_K,
               (long long) a_first_slot, (long long) (a_first_slot + A_N - 1),
               (long long) A_R, (long long) a_first_slot);
        printf("[accepted replay] state_replay vs state_truth max_abs=%.3e over %lld state entries"
               " (replay_count=%lld, buffers %zu elems, all finite)\n",
               md, (long long) state_elems, (long long) replay_count, a_rep.size());
        if (md > 1e-4f) {
            fprintf(stderr, "FAIL: replay of the accepted token from REAL produced slot %lld"
                    " did not reconstruct ground truth (max_abs=%.3e)\n",
                    (long long) a_first_slot, md);
            return 1;
        }
        printf("OK: 1 accepted token replayed from a real produced slot reconstructs ground truth;"
               " %lld producer tokens were never read\n", (long long) (A_N - A_R));
    }

    return 0;
}
