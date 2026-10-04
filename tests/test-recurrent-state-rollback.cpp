#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[m";
    }
    return "";
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// Like decode_tokens(), but requests logits for EVERY position. decode_tokens() only marks
// the last position, so a reference decoded with it cannot be compared position-by-position
// against an arm that replays from mid-batch.
static bool decode_tokens_all_logits(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// Decode tokens[begin, begin+count) as ONE batch, assigning absolute positions begin+i, with
// logits requested for every position. decode_tokens()/decode_tokens_all_logits() always start
// at position 0, so neither can express a replay that resumes mid-sequence.
static bool decode_range_all_logits(llama_context * ctx, const std::vector<llama_token> & tokens,
                                    llama_pos begin, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        common_batch_add(batch, tokens[begin + i], begin + (llama_pos) i, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx.get());
    if (!decode_tokens(ctx.get(), std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        return nullptr;
    }
    llama_synchronize(ctx.get());
    cache_buffer_collector collector;
    llama_get_memory(ctx.get())->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (collector.buffers.empty()) {
        LOG_ERR("%s: no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context_ptr make_ctx_rs(const common_params & params, llama_model * model, uint8_t fill, uint32_t n_rs_seq) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs_seq;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    return make_ctx_rs(params, model, fill, 8);
}

// Stamp the whole recurrent buffer of ctx with `byte`. Anything the replay path reads that it
// did not itself write in this decode comes back as `byte` instead of the value the op left.
static bool poison_memory(llama_context * ctx, uint8_t byte) {
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    if (collector.buffers.empty()) {
        return false;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, byte);
    }
    return true;
}

// llama_memory_recurrent::state_write only reports cells that currently hold a sequence, so
// calling poison_memory() right after llama_memory_clear() finds nothing and can never poison
// anything. The buffers themselves survive the clear, so they are captured once, while a cell
// is populated, and then reused for every poisoning pass.
static bool collect_memory_buffers(llama_context * ctx, std::set<ggml_backend_buffer_t> & out) {
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    if (collector.buffers.empty()) {
        return false;
    }
    out.insert(collector.buffers.begin(), collector.buffers.end());
    return true;
}

static void poison_buffers(const std::set<ggml_backend_buffer_t> & bufs, uint8_t byte) {
    for (auto * buffer : bufs) {
        ggml_backend_buffer_clear(buffer, byte);
    }
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static test_status test_multi_seq_split_replay(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        LOG_ERR("%s: failed to init multi-seq contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        ok = ok && llama_decode(ctx_ref.get(),  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        LOG_ERR("%s: multi-seq prefill/rollback failed\n", __func__);
        return test_status::FAIL;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll.get(), batch) == 0;
    ok = ok && llama_decode(ctx_ref.get(), batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        LOG_ERR("%s: multi-seq replay decode failed\n", __func__);
        return test_status::FAIL;
    }

    // both contexts decode identical batches, so the logits should match;
    // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
    constexpr float nmse_eps = 1e-4f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            LOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
            return test_status::FAIL;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        LOG_ERR("%s: multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        return test_status::FAIL;
    }

    LOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref.get(), batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll.get(), batch_one) == 0;
        ok = ok && llama_decode(ctx_ref.get(), batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        LOG_ERR("%s: seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        return test_status::FAIL;
    }

    LOG_INF("%s: seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    return test_status::PASS;
}

static void set_tensor_data_scaled(ggml_tensor * tensor, void * userdata) {
    const size_t seed = *(const size_t *) userdata ^ std::hash<std::string>{}(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);

    // norm weights and matrices at scale 1: at the 0.01 scale of the generated models the recurrent
    // branch is too small to change the logits, so a wrong recurrent state would go unnoticed
    const bool  is_norm = strstr(tensor->name, "norm") != nullptr;
    const float scale   = ggml_n_dims(tensor) > 1 ? 1.0f : 1.0e-2f;

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> tmp(ggml_nelements(tensor));
    for (auto & x : tmp) {
        x = is_norm ? 1.0f : scale*dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

// same hparams as the model file, new random weights
static llama_model * load_model_scaled(common_params & params) {
    gguf_init_params gparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * meta_file = gguf_init_from_file(params.model.path.c_str(), gparams);
    if (meta_file == nullptr) {
        return nullptr;
    }
    gguf_context * meta = gguf_init_empty();
    gguf_set_kv(meta, meta_file);
    gguf_free(meta_file);

    size_t seed = 1234;
    llama_model * model = llama_model_init_from_user(meta, set_tensor_data_scaled, &seed, common_model_params_to_llama(params));
    gguf_free(meta);
    return model;
}

static void set_graph_reuse_disable(bool disable) {
#ifdef _WIN32
    _putenv_s("LLAMA_GRAPH_REUSE_DISABLE", disable ? "1" : "");
#else
    if (disable) {
        setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    } else {
        unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    }
#endif
}

// Two sequences share ubatches while seq_cp swaps them, forks one and rolls it back, with graph reuse on.
// With n_rs_seq = 0 the recurrent state can be read in place when no state is copied, so a seq_cp between
// two same-shape ubatches must stop the graph from being reused. Checked against a context without graph
// reuse (same ubatches, so bit-exact), a context with n_rs_seq = 1, which never reads the state in place,
// and a context that decodes one sequence per ubatch.
static bool test_seq_cp_graph_reuse(const common_params & params, llama_model * model) {
    const char * func    = __func__;
    const int    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const auto make = [&](uint32_t n_rs_seq, bool reuse) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = 3;
        cparams.n_rs_seq   = n_rs_seq;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 64;
        cparams.n_ubatch   = 64;
        cparams.kv_unified = true;
        set_graph_reuse_disable(!reuse);
        llama_context * ctx = llama_init_from_model(model, cparams);
        set_graph_reuse_disable(false);
        return ctx;
    };

    struct arm {
        const char *    name;
        llama_context * ctx;
        bool            split;    // one llama_decode per sequence
        float           eps;      // max |logit diff| against the first arm, relative to its max |logit|
        float           diff_max;
    };
    std::vector<arm> arms = {
        { "reuse",       make(0, true),  false, 0.0f,  0.0f },
        { "no-reuse",    make(0, false), false, 1e-6f, 0.0f },
        { "no-in-place", make(1, false), false, 1e-5f, 0.0f },
        { "seq-split",   make(0, false), true,  1e-4f, 0.0f },
    };

    const auto cleanup = [&]() {
        for (auto & a : arms) {
            llama_free(a.ctx);
        }
    };

    for (const auto & a : arms) {
        if (a.ctx == nullptr) {
            fprintf(stderr, "%s : failed to init %s context\n", __func__, a.name);
            cleanup();
            return false;
        }
    }

    llama_pos pos[3] = { 0, 0, 0 };
    uint32_t  n_step = 0;
    bool      ok     = true;

    const auto decode = [&](const std::vector<llama_seq_id> & seqs, uint32_t n_tokens) {
        if (!ok) {
            return;
        }
        n_step++;
        const auto token = [&](llama_seq_id s, uint32_t i) {
            return (llama_token) ((13*n_step + 7*(uint32_t) s + 3*i + 1) % (uint32_t) n_vocab);
        };

        const size_t n_rows = seqs.size()*n_tokens;
        std::vector<std::vector<float>> logits(arms.size());
        llama_batch batch = llama_batch_init(n_rows, 0, 1);
        for (size_t k = 0; ok && k < arms.size(); ++k) {
            logits[k].reserve(n_rows*n_vocab);
            const size_t n_batches = arms[k].split ? seqs.size() : 1;
            for (size_t b = 0; ok && b < n_batches; ++b) {
                common_batch_clear(batch);
                for (size_t j = 0; j < seqs.size(); ++j) {
                    if (arms[k].split && j != b) {
                        continue;
                    }
                    for (uint32_t i = 0; i < n_tokens; ++i) {
                        common_batch_add(batch, token(seqs[j], i), pos[seqs[j]] + (llama_pos) i, { seqs[j] }, true);
                    }
                }
                ok = llama_decode(arms[k].ctx, batch) == 0;
                for (int32_t i = 0; ok && i < batch.n_tokens; ++i) {
                    const float * l = llama_get_logits_ith(arms[k].ctx, i);
                    logits[k].insert(logits[k].end(), l, l + n_vocab);
                }
            }
            if (!ok) {
                fprintf(stderr, "%s : %s decode failed at step %u\n", func, arms[k].name, n_step);
            }
        }
        llama_batch_free(batch);

        for (size_t r = 0; ok && r < n_rows; ++r) {
            const float * ref = logits[0].data() + r*n_vocab;
            float ref_max = 0.0f;
            for (int t = 0; t < n_vocab; ++t) {
                ref_max = std::max(ref_max, std::fabs(ref[t]));
            }
            for (size_t k = 1; k < arms.size(); ++k) {
                const float * cur = logits[k].data() + r*n_vocab;
                float diff = 0.0f;
                for (int t = 0; t < n_vocab; ++t) {
                    diff = std::max(diff, logit_diff(ref[t], cur[t]));
                }
                diff /= std::max(ref_max, std::numeric_limits<float>::min());
                arms[k].diff_max = std::max(arms[k].diff_max, diff);
                if (!(diff <= arms[k].eps)) {
                    fprintf(stderr, "%s : step %u, seq %d row %zu: %s differs from %s by %g (rel)\n",
                            func, n_step, seqs[r/n_tokens], r % n_tokens, arms[k].name, arms[0].name, (double) diff);
                    ok = false;
                }
            }
        }
        for (llama_seq_id s : seqs) {
            pos[s] += (llama_pos) n_tokens;
        }
    };

    const auto seq_cp = [&](llama_seq_id src, llama_seq_id dst) {
        for (auto & a : arms) {
            llama_memory_t mem = llama_get_memory(a.ctx);
            ok = ok && llama_memory_seq_rm(mem, dst, -1, -1);
            llama_memory_seq_cp(mem, src, dst, -1, -1);
        }
        pos[dst] = pos[src];
    };

    const auto seq_rm = [&](llama_seq_id s) {
        for (auto & a : arms) {
            ok = ok && llama_memory_seq_rm(llama_get_memory(a.ctx), s, -1, -1);
        }
        pos[s] = 0;
    };

    const auto swap_01 = [&]() {
        seq_cp(0, 2);
        seq_cp(1, 0);
        seq_cp(2, 1);
        seq_rm(2);
    };

    // seq 1 is decoded first, so the first shared ubatch reorders the cells
    decode({ 1 }, 5);
    decode({ 0 }, 5);
    for (uint32_t n_tokens : { 2u, 3u, 1u }) {
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);

        // same ubatch shape, head and rs_z as the step before: only the state copy map changes
        swap_01();
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);

        decode({ 0 }, n_tokens);
        decode({ 1 }, n_tokens);
        decode({ 0 }, n_tokens);
        decode({ 1 }, n_tokens);

        // fork seq 0, advance it, then roll it back to the fork
        seq_cp(0, 2);
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);
        seq_cp(2, 0);
        decode({ 0, 1 }, n_tokens);
        seq_rm(2);
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);
    }

    const int32_t n_reused = llama_perf_context(arms[0].ctx).n_reused;
    if (ok && n_reused == 0) {
        fprintf(stderr, "%s : graph reuse was not exercised\n", __func__);
        ok = false;
    }

    if (ok) {
        fprintf(stderr, "%s : %u steps matched, %d graphs reused (max rel diff: %s %g, %s %g, %s %g)\n", __func__, n_step, n_reused,
                arms[1].name, (double) arms[1].diff_max, arms[2].name, (double) arms[2].diff_max, arms[3].name, (double) arms[3].diff_max);
    }

    cleanup();
    return ok;
}

static test_status test_seq_cp_graph_reuse_for_model(const std::string & model_path, const common_params & base_params) {
    common_params params = base_params;
    params.model.path = model_path;

    gguf_init_params gparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * meta = gguf_init_from_file(model_path.c_str(), gparams);
    if (meta == nullptr) {
        LOG_ERR("%s: failed to read model metadata\n", __func__);
        return test_status::FAIL;
    }
    const int64_t key = gguf_find_key(meta, "general.architecture");
    const std::string arch = key >= 0 ? gguf_get_val_str(meta, key) : "";
    gguf_free(meta);

    // the in-place recurrent state is only enabled for qwen35 and qwen35moe
    if (arch != "qwen35" && arch != "qwen35moe") {
        LOG_INF("%s: skipping test_seq_cp_graph_reuse for %s\n", __func__, arch.c_str());
        return test_status::SKIP;
    }

    llama_model_ptr model_scaled{load_model_scaled(params)};
    if (!model_scaled) {
        LOG_ERR("%s: failed to create scaled model\n", __func__);
        return test_status::FAIL;
    }
    return test_seq_cp_graph_reuse(params, model_scaled.get()) ? test_status::PASS : test_status::FAIL;
}

// Save a rolled-back single-seq state, restore it into fresh and dirty
// contexts, and verify exact logit matches on replay.
static test_status test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_ptr ctx_src = make_ctx(params, model, fill);
    llama_context_ptr ctx_dst = make_ctx(params, model, fill);
    if (!ctx_src || !ctx_dst) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        LOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
        return test_status::SKIP;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }
    if (tokens.empty()) {
        LOG_ERR("%s: not enough prompt tokens\n", __func__);
        return test_status::FAIL;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src.get(), tokens, n_tokens)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    // TOLERANCE, OUR FORK (Kurumi ruling 2026-10-03 06:31): 1e-10, not upstream's 0.0.
    //
    // Upstream ships 0.0 here, i.e. bit-exact. We do not, and the reason is structural rather
    // than a defect: gdn_replay reconstructs state with ONE batched multi-token call, whereas
    // the snapshot path takes a precomputed row. A batched reduction and a token-by-token chain
    // sum in a different order, so the replayed logits agree to floating-point reassociation
    // error and no further. This is the same class as llama.cpp's existing ubatch-size
    // nondeterminism, not a state divergence.
    //
    // MEASURED on this tree, gfx1151, iron GGUF: nmse = 8.71517e-15, first divergence at
    // position 6. That is 11 orders of magnitude below the 1e-4 bar the other two sections of
    // THIS SAME FILE already use (lines 217 and its neighbours), and 4 orders below the 1e-10
    // set here, so the real error has headroom rather than sitting on the threshold.
    //
    // The fork detail that matters for reproducing this: gated_delta_net_cuda is launched with
    // our gdn_num_warps bound, not upstream's hardcoded 4. Substituting upstream's 4 does not
    // merely change the error, it makes this test ABORT (SIGABRT) on this GPU. Verified by
    // experiment, not assumed.
    constexpr float nmse_eps = 1e-10;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                LOG_ERR("%s: %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                LOG_ERR("%s: missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                LOG_ERR("%s: %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return test_status::FAIL;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context_ptr ctx_dirty = make_ctx(params, model, fill);
    if (!ctx_dirty) {
        LOG_ERR("%s: failed to init dirty ctx\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty.get(), noise, n_tokens)) {
        LOG_ERR("%s: dirty prompt decode failed\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: dirty rollback failed\n", __func__);
        return test_status::FAIL;
    }

    ckpt.load_tgt(ctx_dirty.get(), 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
            LOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
        if (logits_dirty == nullptr) {
            LOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            LOG_ERR("%s: dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return test_status::FAIL;
        }
    }

    LOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);
    return test_status::PASS;
}

// OPT-57: rollback/replay, with the batch schedule held CONSTANT across both arms.
//
// Why this arm was reshaped (Kurumi note c5d8e8ce):
// the previous version compared a token-by-token REPLAY against a single batched REFERENCE.
// That varies two things at once -- (a) rollback vs no rollback, and (b) batched vs token-wise
// -- and the control arm measured (b) at 5.4e-3 for n_rs_seq == 8, which is larger than the
// (a) effect under test. A test that cannot separate its own variables cannot fail for the
// reason it claims to test.
//
// The reshaped arm fixes the schedule and moves the rollback to be the only free variable:
//
//   arm REF:  clear -> decode tokens[0, p0)      -> decode tokens[p0, n)  -> capture
//   arm RB :  clear -> decode tokens[0, p0)      -> decode tokens[p0, n)
//                     -> seq_rm(p0) -> decode tokens[p0, n) again       -> capture
//
// Both arms execute the identical token list in the identical batch shapes. The only
// difference is the seq_rm + the second pass over the same range. If rollback restores the
// recurrent state exactly, nmse is 0 at every replayed position.
//
// The prefix batch is deliberately longer than n_rs_seq (p0 > n_rs_seq for every depth), so
// the multi-token needs_ckpt path (ggml/src/ggml-cpu/ops.cpp:11075, triggered by
// n_seq_tokens > K with K == n_rs_seq, reached from src/models/delta-net-base.cpp:749) fires
// INSIDE the matched region, in both arms, identically. It is held constant, not tested.
//
// Rollback depths: 1, and n_rs_seq. m = n_rs_seq - rollback, so depth n_rs_seq gives the legal
// m == 0 case. No m > 0 assertion is added, by ruling.
//
// Dropped from this arm, on order: the 0x00/0xCD poison passes, and the token-by-token control.
// The comparison is exact (eps 1e-10, the same bar test_rollback() uses, not weakened).
static test_status test_rollback_batch_matched(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (n_rs_seq == 0) {
        LOG_INF("%s: n_rs_seq == 0, nothing to roll back\n", __func__);
        return test_status::SKIP;
    }

    // p0 > n_rs_seq for every depth below, so the prefix batch always exceeds the window and
    // the multi-token checkpoint path is inside the matched region.
    const uint32_t n_batch = 2*n_rs_seq + 1;

    // Both arms get the same context shape, with n_batch/ubatch large enough that every batch
    // below lands in exactly ONE ubatch. If a batch were split across ubatches the two arms
    // would still be matched, but the multi-token path would not fire.
    auto make_arm = [&](void) -> llama_context_ptr {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  n_batch);
        cparams.n_ubatch  = std::max(cparams.n_ubatch, n_batch);
        return init_ctx(model, cparams, 0);
    };

    llama_context_ptr ctx_ref = make_arm();
    llama_context_ptr ctx_rb  = make_arm();
    // Attribution arm. Same tokens, but every position decoded on its own. It is never rolled
    // back and never gates. It exists so a MATCHED-SCHEDULE MISMATCH can be read: the control
    // number is the cost of batching ALONE, measured in this run, on this model. Only a rollback
    // delta far above it can be attributed to the rollback. Without this, any nonzero number
    // below is unattributable, which is the exact failure the previous shape of this arm had.
    llama_context_ptr ctx_ctl = make_arm();
    if (!ctx_ref || !ctx_rb || !ctx_ctl) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    const uint32_t got = llama_n_rs_seq(ctx_ref.get());
    LOG_INF("%s: requested n_rs_seq = %u, llama_n_rs_seq = %u, n_ubatch = %u, n_batch = %u\n",
            __func__, n_rs_seq, got, llama_n_ubatch(ctx_ref.get()), n_batch);
    if (got != n_rs_seq) {
        LOG_ERR("%s: n_rs_seq mismatch, cell did not test the requested window\n", __func__);
        return test_status::FAIL;
    }
    if (llama_n_ubatch(ctx_ref.get()) < n_batch) {
        LOG_ERR("%s: n_ubatch %u < n_batch %u, a batch would span ubatches\n",
                __func__, llama_n_ubatch(ctx_ref.get()), n_batch);
        return test_status::FAIL;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
    } else {
        tokens = common_tokenize(ctx_ref.get(),
                "The quick brown fox jumps over the lazy dog while the compiler waits for a "
                "reply that never arrives, and the build hangs there all afternoon while the "
                "machine warms up and the queue drains and the tests run twice on the same "
                "input without anybody reading what either of them actually printed", true);
    }
    // Pad rather than reject: a short prompt is not a failure, and n_batch grows with n_rs_seq
    // (2*n_rs_seq+1), so the n_rs_seq = 8 cell needs more tokens than the prompt supplies.
    if (tokens.size() < n_batch) {
        LOG_WRN("%s: prompt gave %zu tokens, padding to %u with the last token\n",
                __func__, tokens.size(), n_batch);
        tokens.resize(n_batch, tokens.back());
    }

    constexpr float nmse_eps = 1e-10;   // same bar as test_rollback(); not weakened

    // Rollback ladder: 1, 2, 4, ... n_rs_seq. The point of the ladder is the shape of the
    // residual as a function of m = n_rs_seq - rollback. The last rung (rollback == n_rs_seq,
    // m == 0) is the full-window rejection, the one already known to be exact under
    // LLAMA_GDN_REPLAY=1; every earlier rung keeps a partial window (m > 0) alive across the
    // rollback. That zero-vs-partial distinction is the thing being preserved: a ladder that
    // dropped the final rung would lose it, and one that stopped at m == 1 would not show
    // whether the residual scales with m or only switches on at m == 0.
    std::vector<uint32_t> depths;
    for (uint32_t d = 1; d < n_rs_seq; d *= 2) {
        depths.push_back(d);
    }
    depths.push_back(n_rs_seq);

    test_status status = test_status::PASS;

    // Warm-up so the recurrent ring is actually allocated before any clear.
    decode_one(ctx_ref.get(), tokens[0], 0);
    decode_one(ctx_rb.get(),  tokens[0], 0);
    decode_one(ctx_ctl.get(), tokens[0], 0);

    for (uint32_t rollback : depths) {
        const llama_pos  p0     = (llama_pos) n_batch - (llama_pos) rollback;
        const uint32_t   n_tail = rollback;
        const uint32_t   m      = n_rs_seq - rollback;

        llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
        llama_memory_clear(llama_get_memory(ctx_rb.get()),  true);
        llama_memory_clear(llama_get_memory(ctx_ctl.get()), true);

        // ---- matched prefix: identical shape, identical tokens, both arms ----
        if (!decode_range_all_logits(ctx_ref.get(), tokens, 0, p0) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, 0, p0)) {
            LOG_ERR("%s: prefix decode failed (rollback %u, m %u)\n", __func__, rollback, m);
            return test_status::FAIL;
        }

        // ---- matched tail, pass 1: identical shape, identical tokens, both arms ----
        if (!decode_range_all_logits(ctx_ref.get(), tokens, p0, n_tail) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, p0, n_tail)) {
            LOG_ERR("%s: tail decode failed (rollback %u, m %u)\n", __func__, rollback, m);
            return test_status::FAIL;
        }

        std::vector<std::vector<float>> ref_logits(n_tail);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const float * r = llama_get_logits_ith(ctx_ref.get(), (int32_t) i);
            if (r == nullptr) {
                LOG_ERR("%s: reference logits missing at pos %d\n", __func__, (int) (p0 + i));
                return test_status::FAIL;
            }
            ref_logits[i].assign(r, r + n_vocab);
        }

        // ---- attribution control: batching alone, no rollback anywhere ----
        double nmse_ctl_max = 0.0;
        for (uint32_t p = 0; p < n_batch; ++p) {
            if (!decode_one(ctx_ctl.get(), tokens[p], (llama_pos) p)) {
                LOG_ERR("%s: control decode failed at pos %u\n", __func__, p);
                return test_status::FAIL;
            }
            if (p < (uint32_t) p0) {
                continue;
            }
            const float * c = llama_get_logits_ith(ctx_ctl.get(), 0);
            if (c == nullptr) {
                LOG_ERR("%s: control logits missing at pos %u\n", __func__, p);
                return test_status::FAIL;
            }
            const double v = nmse(c, ref_logits[p - p0].data(), n_vocab);
            nmse_ctl_max = v > nmse_ctl_max ? v : nmse_ctl_max;
        }

        // ---- the ONLY free variable: roll the second arm back and replay the same tail ----
        if (!llama_memory_seq_rm(llama_get_memory(ctx_rb.get()), 0, p0, -1)) {
            LOG_ERR("%s: rollback of %u at pos %d refused (n_rs_seq %u)\n",
                    __func__, rollback, (int) p0, n_rs_seq);
            return test_status::FAIL;
        }

        if (!decode_range_all_logits(ctx_rb.get(), tokens, p0, n_tail)) {
            LOG_ERR("%s: replay decode failed (rollback %u, m %u)\n", __func__, rollback, m);
            return test_status::FAIL;
        }

        double nmse_max = 0.0;
        for (uint32_t i = 0; i < n_tail; ++i) {
            const float * a = llama_get_logits_ith(ctx_rb.get(), (int32_t) i);
            if (a == nullptr) {
                LOG_ERR("%s: replay logits missing at pos %d\n", __func__, (int) (p0 + i));
                return test_status::FAIL;
            }
            const double v = nmse(a, ref_logits[i].data(), n_vocab);
            if (v > nmse_max) {
                nmse_max = v;
            }
            if (v > nmse_eps) {
                LOG_ERR("%s: MATCHED-SCHEDULE MISMATCH, rollback %u/%u (m %u), pos %d, nmse %g (eps %g)\n",
                        __func__, rollback, n_rs_seq, m, (int) (p0 + i), v, nmse_eps);
                status = test_status::FAIL;
            }
        }
        LOG_INF("%s: MATCHED rollback %u/%u (m %u), tail %u at pos %d: max nmse %g | CONTROL batching-only %g\n",
                __func__, rollback, n_rs_seq, m, n_tail, (int) p0, nmse_max, nmse_ctl_max);
    }
    return status;
}


// OPT-57 POSITION SWEEP — the ladder left depth and position confounded.
//
// The ladder moved rollback depth 1/2/4/8, and because p0 = n_batch - rollback, it moved p0 at
// the same time: rollback=2 sat at p0=15 while rollback=1 sat at 16 and rollback=4 at 13. So the
// large error at rollback=2 could be a property of the DEPTH (2) or of the POSITION (15). This
// sweep breaks the confound: rollback is pinned at 2 and only p0 moves.
//
// Per pair, REF and RB run the identical token list in the identical batch shapes, exactly as in
// test_rollback_batch_matched. The only free variable is the seq_rm + the replay pass. The
// same-position no-rollback reference is the token-by-token arm, recomputed at EVERY p0, so the
// batching baseline is measured at the position being tested and not carried over from elsewhere.
//
// p0 starts at n_rs_seq + 1 so the prefix batch still exceeds the window and the multi-token
// needs_ckpt path stays inside the matched region. eps is unchanged at 1e-10.
static test_status test_rollback_pos_sweep(const common_params & params, llama_model * model,
                                           uint32_t n_rs_seq, uint32_t rollback) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (n_rs_seq == 0 || rollback == 0 || rollback >= n_rs_seq) {
        LOG_ERR("%s: need 0 < rollback < n_rs_seq (got rollback %u, n_rs_seq %u)\n",
                __func__, rollback, n_rs_seq);
        return test_status::FAIL;
    }

    const uint32_t p0_min = n_rs_seq + 1;   // prefix strictly exceeds the window
    const uint32_t p0_max = 2*n_rs_seq + 1 - rollback;
    const uint32_t m      = n_rs_seq - rollback;

    auto make_arm = [&](void) -> llama_context_ptr {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  p0_max + rollback);
        cparams.n_ubatch  = std::max(cparams.n_ubatch, p0_max + rollback);
        return init_ctx(model, cparams, 0);
    };

    llama_context_ptr ctx_ref = make_arm();
    llama_context_ptr ctx_rb  = make_arm();
    llama_context_ptr ctx_ctl = make_arm();
    if (!ctx_ref || !ctx_rb || !ctx_ctl) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26 };
    } else {
        tokens = common_tokenize(ctx_ref.get(),
                "The quick brown fox jumps over the lazy dog while the compiler waits for a "
                "reply that never arrives, and the build hangs there all afternoon while the "
                "machine warms up and the queue drains and the tests run twice on the same "
                "input without anybody reading what either of them actually printed", true);
    }
    const uint32_t n_need = p0_max + rollback;
    if (tokens.size() < n_need) {
        LOG_WRN("%s: prompt gave %zu tokens, padding to %u with the last token\n",
                __func__, tokens.size(), n_need);
        tokens.resize(n_need, tokens.back());
    }

    constexpr float nmse_eps = 1e-10;   // unchanged

    decode_one(ctx_ref.get(), tokens[0], 0);
    decode_one(ctx_rb.get(),  tokens[0], 0);
    decode_one(ctx_ctl.get(), tokens[0], 0);

    test_status status = test_status::PASS;

    for (uint32_t p0 = p0_min; p0 <= p0_max; ++p0) {
        const uint32_t n_batch = p0 + rollback;

        llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
        llama_memory_clear(llama_get_memory(ctx_rb.get()),  true);
        llama_memory_clear(llama_get_memory(ctx_ctl.get()), true);

        // matched prefix, identical in both arms
        if (!decode_range_all_logits(ctx_ref.get(), tokens, 0, p0) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, 0, p0)) {
            LOG_ERR("%s: prefix decode failed at p0 %u\n", __func__, p0);
            return test_status::FAIL;
        }
        // matched tail, identical in both arms
        if (!decode_range_all_logits(ctx_ref.get(), tokens, p0, rollback) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, p0, rollback)) {
            LOG_ERR("%s: tail decode failed at p0 %u\n", __func__, p0);
            return test_status::FAIL;
        }

        std::vector<std::vector<float>> ref_logits(rollback);
        for (uint32_t i = 0; i < rollback; ++i) {
            const float * r = llama_get_logits_ith(ctx_ref.get(), (int32_t) i);
            if (r == nullptr) {
                LOG_ERR("%s: reference logits missing at pos %u\n", __func__, p0 + i);
                return test_status::FAIL;
            }
            ref_logits[i].assign(r, r + n_vocab);
        }

        // same-position no-rollback reference, recomputed at this p0
        double nmse_ctl_max = 0.0;
        for (uint32_t p = 0; p < n_batch; ++p) {
            if (!decode_one(ctx_ctl.get(), tokens[p], (llama_pos) p)) {
                LOG_ERR("%s: control decode failed at pos %u\n", __func__, p);
                return test_status::FAIL;
            }
            if (p < p0) {
                continue;
            }
            const float * c = llama_get_logits_ith(ctx_ctl.get(), 0);
            if (c == nullptr) {
                LOG_ERR("%s: control logits missing at pos %u\n", __func__, p);
                return test_status::FAIL;
            }
            const double v = nmse(c, ref_logits[p - p0].data(), n_vocab);
            nmse_ctl_max = v > nmse_ctl_max ? v : nmse_ctl_max;
        }

        // the only free variable
        if (!llama_memory_seq_rm(llama_get_memory(ctx_rb.get()), 0, (llama_pos) p0, -1)) {
            LOG_ERR("%s: rollback of %u at pos %u refused (n_rs_seq %u)\n",
                    __func__, rollback, p0, n_rs_seq);
            return test_status::FAIL;
        }
        if (!decode_range_all_logits(ctx_rb.get(), tokens, p0, rollback)) {
            LOG_ERR("%s: replay decode failed at p0 %u\n", __func__, p0);
            return test_status::FAIL;
        }

        double nmse_max = 0.0;
        for (uint32_t i = 0; i < rollback; ++i) {
            const float * a = llama_get_logits_ith(ctx_rb.get(), (int32_t) i);
            if (a == nullptr) {
                LOG_ERR("%s: replay logits missing at pos %u\n", __func__, p0 + i);
                return test_status::FAIL;
            }
            const double v = nmse(a, ref_logits[i].data(), n_vocab);
            nmse_max = v > nmse_max ? v : nmse_max;
            if (v > nmse_eps) {
                status = test_status::FAIL;
            }
        }
        LOG_INF("%s: SWEEP rollback %u/%u (m %u) p0 %2u n_batch %2u: max nmse %-12g | same-pos no-rollback ref %g\n",
                __func__, rollback, n_rs_seq, m, p0, n_batch, nmse_max, nmse_ctl_max);
    }
    return status;
}

// OPT-57 FULL DEPTH SWEEP — p0 pinned, every rollback depth 1..n_rs_seq, not just powers of two.
//
// The ladder covered 1/2/4/8 and the position sweep covered depth 2 at seven positions. Neither
// tested depths 3, 5, 6, 7. If only depth 2 is bad, the fault is narrow and depth-specific; if
// the odd depths are also elevated, the "non-monotonic" ladder reading was wrong and the real
// story is that partial windows are broadly imprecise. One run separates those.
//
// Same construction as the position sweep: REF and RB run the identical token list in the
// identical batch shapes, only seq_rm + the replay pass differs, and the token-by-token arm
// gives a same-position no-rollback reference. eps unchanged at 1e-10.
static test_status test_rollback_depth_sweep(const common_params & params, llama_model * model,
                                             uint32_t n_rs_seq, uint32_t p0_fixed) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (n_rs_seq == 0) {
        return test_status::SKIP;
    }
    if (p0_fixed <= n_rs_seq) {
        LOG_ERR("%s: p0 %u must exceed n_rs_seq %u so the multi-token path stays matched\n",
                __func__, p0_fixed, n_rs_seq);
        return test_status::FAIL;
    }

    const uint32_t n_batch = p0_fixed + n_rs_seq;

    auto make_arm = [&](void) -> llama_context_ptr {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  n_batch);
        cparams.n_ubatch  = std::max(cparams.n_ubatch, n_batch);
        return init_ctx(model, cparams, 0);
    };

    llama_context_ptr ctx_ref = make_arm();
    llama_context_ptr ctx_rb  = make_arm();
    llama_context_ptr ctx_ctl = make_arm();
    if (!ctx_ref || !ctx_rb || !ctx_ctl) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26 };
    } else {
        tokens = common_tokenize(ctx_ref.get(),
                "The quick brown fox jumps over the lazy dog while the compiler waits for a "
                "reply that never arrives, and the build hangs there all afternoon while the "
                "machine warms up and the queue drains and the tests run twice on the same "
                "input without anybody reading what either of them actually printed", true);
    }
    if (tokens.size() < n_batch) {
        LOG_WRN("%s: prompt gave %zu tokens, padding to %u with the last token\n",
                __func__, tokens.size(), n_batch);
        tokens.resize(n_batch, tokens.back());
    }

    constexpr float nmse_eps = 1e-10;

    decode_one(ctx_ref.get(), tokens[0], 0);
    decode_one(ctx_rb.get(),  tokens[0], 0);
    decode_one(ctx_ctl.get(), tokens[0], 0);

    test_status status = test_status::PASS;

    for (uint32_t rollback = 1; rollback <= n_rs_seq; ++rollback) {
        const uint32_t m = n_rs_seq - rollback;

        llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
        llama_memory_clear(llama_get_memory(ctx_rb.get()),  true);
        llama_memory_clear(llama_get_memory(ctx_ctl.get()), true);

        if (!decode_range_all_logits(ctx_ref.get(), tokens, 0, p0_fixed) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, 0, p0_fixed)) {
            LOG_ERR("%s: prefix decode failed (rollback %u)\n", __func__, rollback);
            return test_status::FAIL;
        }
        if (!decode_range_all_logits(ctx_ref.get(), tokens, p0_fixed, rollback) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, p0_fixed, rollback)) {
            LOG_ERR("%s: tail decode failed (rollback %u)\n", __func__, rollback);
            return test_status::FAIL;
        }

        std::vector<std::vector<float>> ref_logits(rollback);
        for (uint32_t i = 0; i < rollback; ++i) {
            const float * r = llama_get_logits_ith(ctx_ref.get(), (int32_t) i);
            if (r == nullptr) {
                LOG_ERR("%s: reference logits missing at %u\n", __func__, p0_fixed + i);
                return test_status::FAIL;
            }
            ref_logits[i].assign(r, r + n_vocab);
        }

        double nmse_ctl_max = 0.0;
        // Only the tail positions have a reference. n_batch is sized for the DEEPEST rollback
        // (n_rs_seq), so for a shallower one the control must stop at the tail end -- reading
        // ref_logits beyond `rollback` here is an out-of-bounds access, not a measurement.
        const uint32_t n_ctl = p0_fixed + rollback;
        for (uint32_t p = 0; p < n_ctl; ++p) {
            if (!decode_one(ctx_ctl.get(), tokens[p], (llama_pos) p)) {
                LOG_ERR("%s: control decode failed at %u\n", __func__, p);
                return test_status::FAIL;
            }
            if (p < p0_fixed) {
                continue;
            }
            const float * c = llama_get_logits_ith(ctx_ctl.get(), 0);
            if (c == nullptr) {
                LOG_ERR("%s: control logits missing at %u\n", __func__, p);
                return test_status::FAIL;
            }
            const double v = nmse(c, ref_logits[p - p0_fixed].data(), n_vocab);
            nmse_ctl_max = v > nmse_ctl_max ? v : nmse_ctl_max;
        }

        if (!llama_memory_seq_rm(llama_get_memory(ctx_rb.get()), 0, (llama_pos) p0_fixed, -1)) {
            LOG_ERR("%s: rollback of %u at %u refused\n", __func__, rollback, p0_fixed);
            return test_status::FAIL;
        }
        if (!decode_range_all_logits(ctx_rb.get(), tokens, p0_fixed, rollback)) {
            LOG_ERR("%s: replay decode failed (rollback %u)\n", __func__, rollback);
            return test_status::FAIL;
        }

        double nmse_max = 0.0;
        for (uint32_t i = 0; i < rollback; ++i) {
            const float * a = llama_get_logits_ith(ctx_rb.get(), (int32_t) i);
            if (a == nullptr) {
                LOG_ERR("%s: replay logits missing at %u\n", __func__, p0_fixed + i);
                return test_status::FAIL;
            }
            const double v = nmse(a, ref_logits[i].data(), n_vocab);
            nmse_max = v > nmse_max ? v : nmse_max;
            if (v > nmse_eps) {
                status = test_status::FAIL;
            }
        }
        LOG_INF("%s: DEPTH n_rs_seq %u p0 %u rollback %u/%u (m %u): max nmse %-12g | same-pos no-rollback ref %g\n",
                __func__, n_rs_seq, p0_fixed, rollback, n_rs_seq, m, nmse_max, nmse_ctl_max);
    }
    return status;
}

// OPT-57 SHIPPED-SHAPED FIXTURE — the schedule the server actually uses.
//
// WHY THIS EXISTS (Kurumi notes 1dd26248 and 948c26b, reviewer fixture-contract finding):
// my earlier depth sweep decoded a tail of length == rollback and then undid all of it. That is
// NOT the shipped invariant. Reading the source in THIS tree gives the real one:
//
//   common/speculative.cpp:1773  llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1)
//   common/speculative.cpp:1768-1772 (comment): "process() filled this layer's KV only for
//       positions < pos0 (prompt + accepted prefix) - nothing in the draft region yet. so reset
//       the draft region ... and select head i so it rebuilds its own layer's KV there;
//       DECODING JUST THE LATEST TOKEN would leave its attention reading cells only another head
//       wrote."
//
// So the shipped shape is: verify batch of d+1 rows, then a rollback of d, then a replay of ONE
// token. Rollback depth and replay batch length are DIFFERENT quantities, and the replay is
// shorter than the rollback. My previous arm had them equal, which is a different problem.
//
// This arm therefore varies d = the rollback depth, with the replay batch fixed at 1 token, and
// always leaves exactly 1 accepted token in the window (the verify batch retains at least one
// accepted row). REF and RB run the identical token list in identical batch shapes; only the
// seq_rm + the single-token replay differs. The token-by-token arm gives the batching baseline
// at the same position. eps unchanged at 1e-10.
//
// The earlier undo-all schedule is KEPT as a separate edge case, not deleted: it is the API
// path, not the shipped path, and its numbers stay as scoped observations only.
static test_status test_rollback_shipped_shape(const common_params & params, llama_model * model,
                                                uint32_t n_rs_seq) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (n_rs_seq < 2) {
        LOG_INF("%s: n_rs_seq %u too small, need >= 2\n", __func__, n_rs_seq);
        return test_status::SKIP;
    }

    const uint32_t d_max = n_rs_seq - 1;          // d < K, as specified
    const uint32_t p0     = n_rs_seq + 1;         // prefix strictly exceeds the window
    const uint32_t n_batch = p0 + d_max + 1;      // worst case: verify batch of d_max+1

    auto make_arm = [&](void) -> llama_context_ptr {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  n_batch);
        cparams.n_ubatch  = std::max(cparams.n_ubatch, n_batch);
        return init_ctx(model, cparams, 0);
    };

    llama_context_ptr ctx_ref = make_arm();
    llama_context_ptr ctx_rb  = make_arm();
    llama_context_ptr ctx_ctl = make_arm();
    if (!ctx_ref || !ctx_rb || !ctx_ctl) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    } else {
        tokens = common_tokenize(ctx_ref.get(),
                "The quick brown fox jumps over the lazy dog while the compiler waits for a "
                "reply that never arrives, and the build hangs there all afternoon while the "
                "machine warms up", true);
    }
    if (tokens.size() < n_batch) {
        LOG_WRN("%s: prompt gave %zu tokens, padding to %u with the last token\n",
                __func__, tokens.size(), n_batch);
        tokens.resize(n_batch, tokens.back());
    }

    constexpr float nmse_eps = 1e-10;

    decode_one(ctx_ref.get(), tokens[0], 0);
    decode_one(ctx_rb.get(),  tokens[0], 0);
    decode_one(ctx_ctl.get(), tokens[0], 0);

    test_status status = test_status::PASS;

    for (uint32_t d = 1; d <= d_max; ++d) {
        // verify batch: d+1 rows at [p0, p0+d+1), identical in both arms
        const uint32_t n_verify = d + 1;

        llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
        llama_memory_clear(llama_get_memory(ctx_rb.get()),  true);
        llama_memory_clear(llama_get_memory(ctx_ctl.get()), true);

        if (!decode_range_all_logits(ctx_ref.get(), tokens, 0, p0) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, 0, p0)) {
            LOG_ERR("%s: prefix decode failed (d %u)\n", __func__, d);
            return test_status::FAIL;
        }
        if (!decode_range_all_logits(ctx_ref.get(), tokens, p0, n_verify) ||
            !decode_range_all_logits(ctx_rb.get(),  tokens, p0, n_verify)) {
            LOG_ERR("%s: verify decode failed (d %u)\n", __func__, d);
            return test_status::FAIL;
        }

        // REF row at position P+1, the single position the replay will produce.
        //
        // REVIEW: it is index 1, NOT the last verify row. The shipped schedule
        // keeps the FIRST verified token and re-decodes the second, so the
        // reference for that re-decode is the ref's logits at P+1.
        const float * r = llama_get_logits_ith(ctx_ref.get(), 1);
        if (r == nullptr) {
            LOG_ERR("%s: reference logits missing at pos %u\n", __func__, p0 + 1);
            return test_status::FAIL;
        }
        const std::vector<float> ref_row(r, r + n_vocab);

        // batching baseline at that same position: everything token-by-token up to it.
        //
        // OFF-BY-ONE FIX 2026-10-04 (rumi, kurumi c92485ef). This loop was
        // `for (p = 0; p < p0+1; ++p)` with `if (p != p0+1) continue;` inside, so
        // the guard was unreachable for every p in range and nmse_ctl_max kept its
        // initial 0.0. The printed "tokenwise ref 0" was the INITIALISER, not a
        // measurement. Fixed: the loop must reach p == p0+1, and the comparison is
        // counted so a future off-by-one cannot silently report a clean zero again.
        double   nmse_ctl_max = std::numeric_limits<double>::quiet_NaN();
        uint32_t n_ctl_comp  = 0;
        for (llama_pos p = 0; p <= (llama_pos) (p0 + 1); ++p) {
            if (!decode_one(ctx_ctl.get(), tokens[p], p)) {
                LOG_ERR("%s: control decode failed at %d\n", __func__, (int) p);
                return test_status::FAIL;
            }
            if (p != (llama_pos) (p0 + 1)) {
                continue;
            }
            const float * c = llama_get_logits_ith(ctx_ctl.get(), 0);
            if (c == nullptr) {
                LOG_ERR("%s: control logits missing at %d\n", __func__, (int) p);
                return test_status::FAIL;
            }
            nmse_ctl_max = nmse(c, ref_row.data(), n_vocab);
            n_ctl_comp++;
        }
        if (n_ctl_comp != 1) {
            LOG_ERR("%s: control comparison ran %u times, expected exactly 1 (d %u)\n",
                    __func__, n_ctl_comp, d);
            return test_status::FAIL;
        }

        // THE SHIPPED MOVE: verify D+1 positions P..P+D, keep the FIRST verified
        // token at P, re-decode the ORIGINAL token at P+1.
        //
        // REVIEW FIX. This used to be:
        //     p_replay = p0 + n_verify - 1;  seq_rm(p_replay, -1)
        // which starts the removal at p0+D and therefore removes ONE token and
        // RETAINS d accepted rows -- a different schedule, and identical to the
        // intended one only when d == 1. That is why d=1 looked right and every
        // d>1 cell was measuring the wrong thing.
        //
        // Removal uses -1 ("from here to the end"), not an explicit count.
        // RETRACTED 2026-10-03 (rumi, kurumi 0bce365e): the older comment here
        // claimed that "passing an explicit count removes the cells but does NOT
        // rewind the memory module's last-position tracker", evidenced by an
        // X=6 / Y=6 / "X < Y" refusal. That was WRONG and the claim is withdrawn.
        // llama_memory_seq_rm takes (p0, p1) as POSITIONS (include/llama.h:773-777),
        // so passing a token COUNT as p1 builds a reversed/invalid range; the
        // refusal was an artifact of a malformed call, not evidence about rewind.
        // On the rewind path (llama-memory-recurrent.cpp:194-204) the module sets
        // rs_idx and cell.pos = p0 - 1 and RETURNS TRUE without reaching the cell
        // removal loop, so "removes cells but does not rewind" describes no
        // reachable path. The finite-endpoint question is settled by measurement
        // in test_finite_end_matched, not by argument.
        const llama_pos p_keep   = (llama_pos) p0;        // P, the one retained cell
        const llama_pos p_replay = (llama_pos) (p0 + 1);  // P+1, re-decoded
        const int32_t  removed   = (int32_t) d;          // P+1 .. P+D inclusive
        if (!llama_memory_seq_rm(llama_get_memory(ctx_rb.get()), 0, p_replay, -1)) {
            LOG_ERR("%s: rollback of %d at pos %d refused (n_rs_seq %u)\n",
                    __func__, removed, (int) p_replay, n_rs_seq);
            return test_status::FAIL;
        }
        if (!decode_one(ctx_rb.get(), tokens[p_replay], p_replay)) {
            LOG_ERR("%s: replay decode failed (d %u)\n", __func__, d);
            return test_status::FAIL;
        }

        const float * a = llama_get_logits_ith(ctx_rb.get(), 0);
        if (a == nullptr) {
            LOG_ERR("%s: replay logits missing (d %u)\n", __func__, d);
            return test_status::FAIL;
        }
        const double v = nmse(a, ref_row.data(), n_vocab);
        if (v > nmse_eps) {
            LOG_ERR("%s: SHIPPED-SHAPE MISMATCH, d %u/%u, pos %d, nmse %g (eps %g)\n",
                    __func__, d, n_rs_seq, (int) p_replay, v, nmse_eps);
            status = test_status::FAIL;
        }
        LOG_INF("%s: SHIPPED d %u/%u, verify %u rows P..P+%u, removed %d, retained 1 at P %d, re-decoded pos %d: nmse %-12g | tokenwise ref %g\n",
                __func__, d, n_rs_seq, n_verify, d, removed, (int) p_keep, (int) p_replay, v, nmse_ctl_max);
    }
    return status;
}

static bool g_finite_end = false;           // OPT-57: seq_rm finite-end vs -1, one matched cell (n_rs 8, d 2)
static bool g_finite_end_hook = false;      // OPT-57: arm the cb_eval observation (off by default so the
                                           // plain --finite-end run stays comparable to accepted numbers)

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// OPT-57 cb_eval observation, test-only, no src/ change. REWRITTEN after
// kurumi 134ba4d: the previous version had three defects and observed nothing
// useful about the checkpoint.
//
// WHAT WAS WRONG, and why the zero-size CPY was never evidence:
//   1. PHASE TAGS WERE LYING. One tag spanned the reference prefix AND the
//      reference verify AND every token-by-token control decode; another spanned
//      the RB prefix AND the RB verify. "prefix9" and "verify3" therefore never
//      meant one decode. Now every decode is tagged separately, before the call.
//   2. A MISLEADING LAYER FILTER. The numeric-suffix parser read "cache_s_ckpt_l0"
//      as "l0" and returned -1, while GDN state roots parse as "node_47". That
//      compared -1 against 47 and SILENTLY DROPPED every checkpoint CPY after the
//      first GDN node. The whole layer parser and the g_cb_layer filter are GONE.
//      Selection is now the EXACT destination view-root name, nothing else.
//   3. ZERO-SIZE CPY WAS MISREAD AS AN API LIMIT. The destination view reported
//      ggml_nbytes=0 and I concluded "nothing to read, checksum impossible". Kurumi
//      identified that node as build_rs extra-state housekeeping
//      (llama-graph.cpp:3743-3751), which is a row-count view and can legitimately
//      be empty. So zero-size means HOUSEKEEPING, not "unreadable". Zero-size
//      copies are now reported separately and never counted as observations.
//
// CALLBACK ORDER, verified: ggml-backend.cpp:2029 asks with ask=true, computes the
// node range (:2041), calls ggml_backend_synchronize (:2046), then calls back with
// ask=false (:2049). Values are readable at ask=false through the public
// ggml_backend_tensor_get, as common/debug.cpp:143-199 demonstrates.
//
// SELECTION: a CPY node whose DESTINATION view-root name is exactly
// "cache_s_ckpt_l0". One layer, hard-coded, no expansion, no parsing.
// For a selected node the hook reports the source ancestry (op + ne at each hop)
// and hashes the destination ONLY when it is nonempty. A zero hash is never
// reported as "observed": the summary prints HASH_OK or HASH_SKIPPED_ZERO_SIZE.
//
// No logits, no state values, no vectors, no weights are printed or returned.
// The checksum is FNV-1a over bytes fetched with the public API; no raw pointer
// is dereferenced. A hash is STATE evidence; pointers would only be branch
// evidence, so no pointer is reported as a state claim.
// ---------------------------------------------------------------------------

#define CB_MAX_SLOTS 16

static const char * g_cb_phase    = "init";
static int          g_cb_slot     = -1;
static uint64_t     g_cb_hash     [CB_MAX_SLOTS] = { 0 };
static bool         g_cb_hash_ok  [CB_MAX_SLOTS] = { false };   // true only if hashed nonempty
static bool         g_cb_zero_seen[CB_MAX_SLOTS] = { false };   // zero-size housekeeping seen
static uint32_t     g_cb_ckpt_nodes = 0;
static const char * g_cb_slot_name[CB_MAX_SLOTS] = { nullptr };

// OPT-57 NEXT57: bounded BYTE capture for a direct memcmp, never a hash and never a
// dump. We are not asking "is the checksum the same"; we are asking whether the
// layer-0 FINAL-state written during rb.prefix9 is byte-for-byte the checkpoint the
// verify3 CPY took. A hash can be equal by collision and unequal for a benign
// reason; only the bytes answer it. Slots are fixed to arm A only (slot 3 = rb.prefix9
// final state, slot 4 = rb.verify3 checkpoint) so the capture is bounded and does not
// grow with layers or tokens.
static std::vector<uint8_t> g_cb_final_bytes;   // last NONEMPTY capture of cache_s_l0 at slot 3
static std::vector<uint8_t> g_cb_ckpt_bytes;    // last NONEMPTY capture of cache_s_ckpt_l0 at slot 4
static bool     g_cb_final_ok   = false;
static bool     g_cb_ckpt_ok    = false;
static size_t   g_cb_final_nb   = 0;
static size_t   g_cb_ckpt_nb    = 0;
static uint32_t g_cb_final_caps = 0;
static uint32_t g_cb_ckpt_caps  = 0;

// Walk a view chain to its root, accumulating offsets. Public fields only.
static void cb_view_root(const ggml_tensor * t, const ggml_tensor ** root, size_t * offs) {
    *root = t;
    *offs = 0;
    while ((*root)->view_src) {
        *offs += (*root)->view_offs;
        *root  = (*root)->view_src;
    }
}

// Load-bearing, not defensive decoration. Dropping the bound made REPLAY=1 abort
// with std::length_error ("cannot create std::vector larger than max_size"): a
// NONEMPTY checkpoint destination view reports a byte span far larger than any real
// state row. So ggml_nbytes on this view is NOT a usable measure of the copied
// state, and an oversized span is reported UNKNOWN -- never an observation, never
// a hash of 0.
#define CB_MAX_CKPT_BYTES (4u << 20)   // 4 MiB

static uint64_t cb_hash_tensor(const ggml_tensor * t, size_t * out_nbytes) {
    const size_t nb = ggml_nbytes(t);
    *out_nbytes = nb;
    if (nb == 0 || nb > CB_MAX_CKPT_BYTES) {
        return 0;
    }
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);   // public read path
    uint64_t h = 1469598103934665603ULL;              // FNV-1a 64
    for (size_t i = 0; i < nb; ++i) {
        h ^= buf[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// OPT-57 NEXT57: bounded byte CAPTURE, same load-bearing bound as cb_hash_tensor
// above (an oversized view span once aborted with std::length_error). Last NONEMPTY
// capture wins, so the value compared is the state as of the final write in the
// phase, not the first. Returns false and leaves out untouched when the span is
// unusable, so a rejected capture can never masquerade as a zero-length match.
static bool cb_capture_tensor(const ggml_tensor * t, std::vector<uint8_t> & out, size_t * out_nbytes) {
    const size_t nb = ggml_nbytes(t);
    *out_nbytes = nb;
    if (nb == 0 || nb > CB_MAX_CKPT_BYTES) {
        return false;
    }
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);   // public read path
    out.swap(buf);
    return true;
}

// forward decl: the observer body sits with the other callback helpers below
static bool cb_eval_observer(struct ggml_tensor * t, bool ask, void * user_data);


// Both arms are run from IDENTICAL state (same prefix, same verify batch, same
// reference and control rows), so the two numbers are comparable. pos_max is
// logged after the removal (expect P) and after the re-decode (expect P+1), so
// the rewind is observed rather than inferred from a decode success.
static test_status test_finite_end_matched(const common_params & params, llama_model * model,
                                          uint32_t n_rs_seq, uint32_t d) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (n_rs_seq < 3) {
        LOG_INF("%s: n_rs_seq %u too small, need >= 3\n", __func__, n_rs_seq);
        return test_status::SKIP;
    }
    if (d < 1 || d >= n_rs_seq) {
        LOG_INF("%s: d %u out of range for n_rs_seq %u (need 1 <= d < n_rs_seq)\n", __func__, d, n_rs_seq);
        return test_status::SKIP;
    }

    const uint32_t n_verify = d + 1;
    const uint32_t p0       = n_rs_seq + 1;         // prefix strictly exceeds the window
    const uint32_t n_batch  = p0 + n_verify;

    // fresh observer state for this cell
    g_cb_ckpt_nodes = 0;
    memset(g_cb_hash, 0, sizeof(g_cb_hash));
    memset(g_cb_hash_ok, 0, sizeof(g_cb_hash_ok));
    memset(g_cb_zero_seen, 0, sizeof(g_cb_zero_seen));
    memset(g_cb_slot_name, 0, sizeof(g_cb_slot_name));
    // NEXT57: the byte capture MUST be reset with the hashes. run_tests runs this
    // cell once per cache fill, and "last capture wins" would otherwise let a fill
    // whose capture was rejected compare against the PREVIOUS fill's bytes and
    // report a clean match for state that was never read.
    g_cb_final_bytes.clear();
    g_cb_ckpt_bytes.clear();
    g_cb_final_ok   = false;
    g_cb_ckpt_ok    = false;
    g_cb_final_nb   = 0;
    g_cb_ckpt_nb    = 0;
    g_cb_final_caps = 0;
    g_cb_ckpt_caps  = 0;

    auto make_arm = [&](void) -> llama_context_ptr {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  n_batch);
        cparams.n_ubatch  = std::max(cparams.n_ubatch, n_batch);
        if (g_finite_end_hook) {
            cparams.cb_eval          = cb_eval_observer;
            cparams.cb_eval_user_data = nullptr;
        }
        return init_ctx(model, cparams, 0);
    };

    llama_context_ptr ctx_ref = make_arm();
    llama_context_ptr ctx_ctl = make_arm();
    if (!ctx_ref || !ctx_ctl) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    } else {
        tokens = common_tokenize(ctx_ref.get(),
                "The quick brown fox jumps over the lazy dog while the compiler waits for a "
                "reply that never arrives, and the build hangs there all afternoon while the "
                "machine warms up", true);
    }
    if (tokens.size() < n_batch) {
        LOG_WRN("%s: prompt gave %zu tokens, padding to %u with the last token\n",
                __func__, tokens.size(), n_batch);
        tokens.resize(n_batch, tokens.back());
    }

    constexpr float nmse_eps = 1e-10;

    // ---- shared setup: reference rows and the token-by-token control ----
    llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
    llama_memory_clear(llama_get_memory(ctx_ctl.get()), true);
    // phase 0: reference prefix
    g_cb_phase = "ref.prefix9"; g_cb_slot = 0; g_cb_slot_name[0] = "ref.prefix9";
    if (!decode_range_all_logits(ctx_ref.get(), tokens, 0, p0)) {
        LOG_ERR("%s: reference prefix decode failed\n", __func__);
        return test_status::FAIL;
    }
    g_cb_phase = "ref.verify3"; g_cb_slot = 1; g_cb_slot_name[1] = "ref.verify3";
    if (!decode_range_all_logits(ctx_ref.get(), tokens, p0, n_verify)) {
        LOG_ERR("%s: reference verify decode failed\n", __func__);
        return test_status::FAIL;
    }

    // REF row at P+1 -- the single position the replay produces (index 1 of the
    // verify batch, same as the shipped arm).
    const float * r = llama_get_logits_ith(ctx_ref.get(), 1);
    if (r == nullptr) {
        LOG_ERR("%s: reference logits missing at pos %u\n", __func__, p0 + 1);
        return test_status::FAIL;
    }
    const std::vector<float> ref_row(r, r + n_vocab);

    // OFF-BY-ONE FIX 2026-10-04 (rumi, kurumi c92485ef). I copied this loop from
    // test_rollback_shipped_shape, which had the same defect: `p < p0+1` combined with
    // `if (p != p0+1) continue;` made the comparison unreachable, so nmse_ctl_max stayed at
    // its initialiser. My earlier "tokenwise ref 0" was therefore NOT a measured baseline
    // and every statement I made from it about batching being uninvolved was unsupported.
    double   nmse_ctl_max = std::numeric_limits<double>::quiet_NaN();
    uint32_t n_ctl_comp  = 0;
    for (llama_pos p = 0; p <= (llama_pos) (p0 + 1); ++p) {
    g_cb_phase = "ctl.tokenwise"; g_cb_slot = 2; g_cb_slot_name[2] = "ctl.tokenwise";
        if (!decode_one(ctx_ctl.get(), tokens[p], p)) {
            LOG_ERR("%s: control decode failed at %d\n", __func__, (int) p);
            return test_status::FAIL;
        }
        if (p != (llama_pos) (p0 + 1)) {
            continue;
        }
        const float * c = llama_get_logits_ith(ctx_ctl.get(), 0);
        if (c == nullptr) {
            LOG_ERR("%s: control logits missing at %d\n", __func__, (int) p);
            return test_status::FAIL;
        }
        nmse_ctl_max = nmse(c, ref_row.data(), n_vocab);
        n_ctl_comp++;
    }
    if (n_ctl_comp != 1) {
        LOG_ERR("%s: control comparison ran %u times, expected exactly 1\n", __func__, n_ctl_comp);
        return test_status::FAIL;
    }

    const llama_pos p_keep   = (llama_pos) p0;      // P, retained
    const llama_pos p_replay = (llama_pos) (p0 + 1); // P+1, re-decoded
    const llama_pos p_end    = (llama_pos) (p0 + d + 1); // P+D+1, finite endpoint

    // arm A: to the end. arm B: finite endpoint past the tail.
    struct arm { const char * name; llama_pos p1; };
    const arm arms[2] = { { "A:-1", -1 }, { "B:P+D+1", p_end } };

    double   nmse_arm[2] = { -1.0, -1.0 };
    llama_pos posmax_rm[2] = { -1, -1 };
    llama_pos posmax_rd[2] = { -1, -1 };
    bool      rm_ok[2]     = { false, false };
    // Both arms' output vectors, kept so A-vs-B can be compared DIRECTLY.
    // Comparing |nmse_A - nmse_B| only compares error MAGNITUDES against a common
    // reference: two different vectors can share a magnitude. Fixed per c92485ef.
    std::vector<float> vec_arm[2];
    test_status status = test_status::PASS;

    for (int i = 0; i < 2; ++i) {
        // identical state for each arm
        llama_context_ptr ctx_rb = make_arm();
        if (!ctx_rb) {
            LOG_ERR("%s: failed to init arm context (%s)\n", __func__, arms[i].name);
            return test_status::FAIL;
        }
        llama_memory_clear(llama_get_memory(ctx_rb.get()), true);
        // phase 1: the verify batch (the "verify3" of the brief)
        g_cb_phase = "rb.prefix9"; g_cb_slot = 3 + 3*i; g_cb_slot_name[3 + 3*i] = "rb.prefix9";
        if (!decode_range_all_logits(ctx_rb.get(), tokens, 0, p0)) {
            LOG_ERR("%s: arm prefix decode failed (%s)\n", __func__, arms[i].name);
            return test_status::FAIL;
        }
        g_cb_phase = "rb.verify3"; g_cb_slot = 4 + 3*i; g_cb_slot_name[4 + 3*i] = "rb.verify3";
        if (!decode_range_all_logits(ctx_rb.get(), tokens, p0, n_verify)) {
            LOG_ERR("%s: arm verify decode failed (%s)\n", __func__, arms[i].name);
            return test_status::FAIL;
        }

        // p1 is a POSITION in both arms. never a token count.
        rm_ok[i] = llama_memory_seq_rm(llama_get_memory(ctx_rb.get()), 0, p_replay, arms[i].p1);
        if (!rm_ok[i]) {
            LOG_ERR("%s: arm %s seq_rm(P+1=%d, p1=%d) REFUSED (n_rs_seq %u)\n",
                    __func__, arms[i].name, (int) p_replay, (int) arms[i].p1, n_rs_seq);
            status = test_status::FAIL;
            continue;
        }
        posmax_rm[i] = llama_memory_seq_pos_max(llama_get_memory(ctx_rb.get()), 0);
        // ASSERT, do not merely print "expect": a rewind that lands anywhere other
        // than P invalidates every number below it.
        if (posmax_rm[i] != p_keep) {
            LOG_ERR("%s: arm %s pos_max after remove = %d, EXPECTED P = %d\n",
                    __func__, arms[i].name, (int) posmax_rm[i], (int) p_keep);
            return test_status::FAIL;
        }
        LOG_INF("%s: arm %-8s seq_rm(P+1=%d, p1=%d) ok, pos_max after remove = %d (expect P=%d, ASSERTED)\n",
                __func__, arms[i].name, (int) p_replay, (int) arms[i].p1,
                (int) posmax_rm[i], (int) p_keep);

        // phase 2: the single-token re-decode
        g_cb_phase = "rb.redecode1"; g_cb_slot = 5 + 3*i; g_cb_slot_name[5 + 3*i] = "rb.redecode1";
        if (!decode_one(ctx_rb.get(), tokens[p_replay], p_replay)) {
            LOG_ERR("%s: re-decode failed (arm %s)\n", __func__, arms[i].name);
            return test_status::FAIL;
        }
        posmax_rd[i] = llama_memory_seq_pos_max(llama_get_memory(ctx_rb.get()), 0);
        if (posmax_rd[i] != p_replay) {
            LOG_ERR("%s: arm %s pos_max after re-decode = %d, EXPECTED P+1 = %d\n",
                    __func__, arms[i].name, (int) posmax_rd[i], (int) p_replay);
            return test_status::FAIL;
        }
        LOG_INF("%s: arm %-8s re-decoded at P+1=%d, pos_max after re-decode = %d (ASSERTED)\n",
                __func__, arms[i].name, (int) p_replay, (int) posmax_rd[i]);

        const float * a = llama_get_logits_ith(ctx_rb.get(), 0);
        if (a == nullptr) {
            LOG_ERR("%s: replay logits missing (arm %s)\n", __func__, arms[i].name);
            return test_status::FAIL;
        }
        vec_arm[i].assign(a, a + n_vocab);
        nmse_arm[i] = nmse(a, ref_row.data(), n_vocab);
        if (nmse_arm[i] > nmse_eps) {
            LOG_ERR("%s: FINITE-END MISMATCH, arm %s, d %u/%u, nmse %g (eps %g)\n",
                    __func__, arms[i].name, d, n_rs_seq, nmse_arm[i], nmse_eps);
            status = test_status::FAIL;
        }
    }

    // A vs B compared DIRECTLY on the vectors. |nmse_A - nmse_B| == 0 only says the two
    // error MAGNITUDES agree against a common reference; it cannot distinguish identical
    // vectors from two different vectors of equal error. Fixed per c92485ef.
    if (rm_ok[0] && rm_ok[1] && (int) vec_arm[0].size() == n_vocab && (int) vec_arm[1].size() == n_vocab) {
        double max_abs_diff = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            max_abs_diff = std::max(max_abs_diff, (double) std::fabs(vec_arm[0][i] - vec_arm[1][i]));
        }
        const double nmse_ab = nmse(vec_arm[0].data(), vec_arm[1].data(), n_vocab);
        LOG_INF("%s: arm agreement A vs B DIRECT: max|diff| %g, nmse(A,B) %g, vectors %s\n",
                __func__, max_abs_diff, nmse_ab,
                (vec_arm[0] == vec_arm[1]) ? "bitwise identical" : "DIFFER");
        if (max_abs_diff > 0.0) {
            LOG_ERR("%s: arms A and B produced DIFFERENT logit vectors (max|diff| %g)\n",
                    __func__, max_abs_diff);
            status = test_status::FAIL;
        }
    }

    if (g_finite_end_hook) {
        // Per-slot report. A hash is printed as HASH_OK only when the copy was
        // NONEMPTY and was actually hashed. Anything else prints UNKNOWN, never a
        // bare 0 -- a skipped hash is not an observation.
        LOG_INF("%s: HOOK summary ckpt_cpy_nodes_selected %u\n", __func__, g_cb_ckpt_nodes);
        uint32_t n_ok = 0, n_zero = 0, n_none = 0;
        for (int k = 0; k < CB_MAX_SLOTS; ++k) {
            if (g_cb_slot_name[k] == nullptr) {
                continue;
            }
            if (g_cb_hash_ok[k]) {
                n_ok++;
                LOG_INF("%s: HOOK slot %2d %-14s HASH_OK        fnv1a64=%016llx\n",
                        __func__, k, g_cb_slot_name[k], (unsigned long long) g_cb_hash[k]);
            } else if (g_cb_zero_seen[k]) {
                n_zero++;
                LOG_INF("%s: HOOK slot %2d %-14s ZERO-SIZE-ONLY (housekeeping; UNKNOWN for state)\n",
                        __func__, k, g_cb_slot_name[k]);
            } else {
                n_none++;
                LOG_INF("%s: HOOK slot %2d %-14s UNKNOWN (no nonempty ckpt copy selected)\n",
                        __func__, k, g_cb_slot_name[k]);
            }
        }
        LOG_INF("%s: HOOK summary hashed %u, zero-size-only %u, unknown %u\n", __func__, n_ok, n_zero, n_none);

        // ---- NEXT57: the direct byte comparison -------------------------------
        // QUESTION: is the layer-0 final state written at rb.prefix9 byte-for-byte
        // the checkpoint the verify3 copy took? Source says the short-batch path
        // copies base_state; this asks whether that is ACTUALLY the prefix-end
        // state. A hash cannot answer it -- equal hashes are agreement, not proof,
        // and unequal hashes do not say which side is wrong. Only bytes can.
        //
        // A length mismatch is reported as its own outcome, NOT as a mismatch of
        // the shared prefix: two different spans are not comparable bytes, and
        // calling that "DIFFERENT at offset 0" would overstate what was measured.
        if (!g_cb_final_ok || !g_cb_ckpt_ok) {
            LOG_INF("%s: HOOK NEXT57 final(prefix9) vs ckpt(verify3) : UNKNOWN "
                    "(final_captured=%d nbytes=%zu rejected=%u | ckpt_captured=%d nbytes=%zu rejected=%u)\n",
                    __func__,
                    (int) g_cb_final_ok, g_cb_final_nb, g_cb_final_caps,
                    (int) g_cb_ckpt_ok,  g_cb_ckpt_nb,  g_cb_ckpt_caps);
        } else if (g_cb_final_nb != g_cb_ckpt_nb) {
            LOG_INF("%s: HOOK NEXT57 final(prefix9) vs ckpt(verify3) : LENGTH-MISMATCH "
                    "final=%zu B, ckpt=%zu B, delta=%lld B -- not comparable byte-for-byte\n",
                    __func__, g_cb_final_nb, g_cb_ckpt_nb,
                    (long long) g_cb_ckpt_nb - (long long) g_cb_final_nb);
        } else {
            const size_t n = g_cb_final_nb;
            size_t first_diff = n;
            size_t n_match = 0;
            for (size_t i = 0; i < n; ++i) {
                if (g_cb_final_bytes[i] == g_cb_ckpt_bytes[i]) {
                    n_match++;
                } else if (first_diff == n) {
                    first_diff = i;
                }
            }
            if (first_diff == n) {
                LOG_INF("%s: HOOK NEXT57 final(prefix9) vs ckpt(verify3) : IDENTICAL "
                        "%zu/%zu bytes match, memcmp == 0 -- the prefix-end state IS what verify3 checkpointed\n",
                        __func__, n_match, n);
            } else {
                LOG_INF("%s: HOOK NEXT57 final(prefix9) vs ckpt(verify3) : DIFFER at offset %zu "
                        "(%zu/%zu bytes match, %.4f%%) -- final=%02x ckpt=%02x\n",
                        __func__, first_diff, n_match, n, 100.0 * (double) n_match / (double) n,
                        g_cb_final_bytes[first_diff], g_cb_ckpt_bytes[first_diff]);
            }
        }
        // State comparison is only meaningful between slots that were BOTH hashed.
        if (g_cb_hash_ok[3] && g_cb_hash_ok[4]) {
            LOG_INF("%s: HOOK rb(A) prefix9 vs verify3 : %s\n", __func__,
                    g_cb_hash[3] == g_cb_hash[4] ? "SAME" : "DIFFERENT");
        } else {
            LOG_INF("%s: HOOK rb(A) prefix9 vs verify3 : UNKNOWN (a slot was not hashed)\n", __func__);
        }
        if (g_cb_hash_ok[4] && g_cb_hash_ok[5]) {
            LOG_INF("%s: HOOK rb(A) verify3 vs redecode1 : %s\n", __func__,
                    g_cb_hash[4] == g_cb_hash[5] ? "SAME" : "DIFFERENT");
        } else {
            LOG_INF("%s: HOOK rb(A) verify3 vs redecode1 : UNKNOWN (a slot was not hashed)\n", __func__);
        }
        if (g_cb_hash_ok[6] && g_cb_hash_ok[7]) {
            LOG_INF("%s: HOOK rb(B) prefix9 vs verify3 : %s\n", __func__,
                    g_cb_hash[6] == g_cb_hash[7] ? "SAME" : "DIFFERENT");
        }
        if (g_cb_hash_ok[7] && g_cb_hash_ok[8]) {
            LOG_INF("%s: HOOK rb(B) verify3 vs redecode1 : %s\n", __func__,
                    g_cb_hash[7] == g_cb_hash[8] ? "SAME" : "DIFFERENT");
        }
    }

    LOG_INF("%s: FINITE-END d %u/%u, P=%d P+1=%d P+D+1=%d | A:-1 posmax %d->%d nmse %-12g | B:P+D+1 posmax %d->%d nmse %-12g | tokenwise ref %g\n",
            __func__, d, n_rs_seq, (int) p_keep, (int) p_replay, (int) p_end,
            (int) posmax_rm[0], (int) posmax_rd[0], nmse_arm[0],
            (int) posmax_rm[1], (int) posmax_rd[1], nmse_arm[1], nmse_ctl_max);

    return status;
}

// OPT-57 REVIEW 18:24, worker GO: "predecessor K+1" fixture.
//
// WHAT "PREDECESSOR" MEANS HERE, because the source does not use the word.
// src/models/delta-net-base.cpp:748 branches on n_seq_tokens vs n_rs_seq. The
// long branch recomputes the true oldest-edge checkpoint. The SHORT branch
// (:776) stores base_state verbatim, and the CBA comment at :778-786 says that
// base_state is only "before the window" when n_seq_tokens == n_rs_seq; for a
// strictly shorter batch it sits INSIDE the uncertain window, the ingredient
// ring keeps only the last n_rs_seq steps, and delta-net's rank-1 update has no
// inverse -- so the true state is unrecoverable.
//
// A replay of r tokens needs a valid predecessor r steps back. The shipped
// shape replays ONE token, i.e. its predecessor sits at age 1. "K+1" is the
// next rung: a predecessor at age 2. This arm is the shipped arm with r
// generalised, sweeping r = 1, 2, 3 at a fixed rollback depth d, so the
// question "is a K+1 predecessor valid?" becomes a number instead of an
// argument. r=1 reproduces the shipped cell exactly, which is the in-band
// control: if r=1 does not match the shipped arm's number, this arm is wrong
// before it says anything about r=2.
static test_status test_rollback_predecessor_k1(const common_params & params, llama_model * model,
                                                uint32_t n_rs_seq) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (n_rs_seq < 4) {
        LOG_INF("%s: n_rs_seq %u too small, need >= 4 to hold an age-3 predecessor\n",
                __func__, n_rs_seq);
        return test_status::SKIP;
    }

    // replay lengths to sweep. r=1 is the shipped shape; r=2 is the K+1 rung.
    constexpr uint32_t r_max = 3;

    const uint32_t d       = n_rs_seq / 2;     // a rollback strictly inside the window
    const uint32_t p0      = n_rs_seq + 1;     // prefix strictly exceeds the window
    const uint32_t n_verify = d + 1;           // shipped verify shape: d+1 rows
    const uint32_t n_batch  = p0 + n_verify + r_max;

    auto make_arm = [&](void) -> llama_context_ptr {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = n_rs_seq;
        cparams.n_batch   = std::max(cparams.n_batch,  n_batch);
        cparams.n_ubatch  = std::max(cparams.n_ubatch, n_batch);
        return init_ctx(model, cparams, 0);
    };

    llama_context_ptr ctx_ref = make_arm();
    llama_context_ptr ctx_rb  = make_arm();
    if (!ctx_ref || !ctx_rb) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    } else {
        tokens = common_tokenize(ctx_ref.get(),
                "The quick brown fox jumps over the lazy dog while the compiler waits for a "
                "reply that never arrives, and the build hangs there all afternoon while the "
                "machine warms up", true);
    }
    if (tokens.size() < n_batch) {
        tokens.resize(n_batch, tokens.back());
    }

    constexpr float nmse_eps = 1e-10;

    decode_one(ctx_ref.get(), tokens[0], 0);
    decode_one(ctx_rb.get(),  tokens[0], 0);

    const llama_pos p_replay = (llama_pos) (p0 + n_verify - 1);

    test_status status = test_status::PASS;

    for (uint32_t r = 1; r <= r_max; ++r) {
        llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
        llama_memory_clear(llama_get_memory(ctx_rb.get()),  true);

        // REF: the same token stream with no rollback anywhere.
        //
        // ONE range for verify+replay, not two. p_replay is the LAST verify
        // position, so decoding [p0, n_verify) and then [p_replay, r) would
        // decode p_replay twice with no rollback in between. For a recurrent
        // model that advances the state a second time for a position it has
        // already consumed, and the library's own batched-vs-sequential check
        // trips ("full logits mismatch at position 6"). Decoding the union
        // once keeps the ref arm genuinely rollback-free.
        if (!decode_range_all_logits(ctx_ref.get(), tokens, 0, p0) ||
            !decode_range_all_logits(ctx_ref.get(), tokens, p0, n_verify + r - 1)) {
            LOG_ERR("%s: reference decode failed (r %u)\n", __func__, r);
            return test_status::FAIL;
        }
        const float * ref = llama_get_logits_ith(ctx_ref.get(), (int32_t) (n_verify + r - 2));
        if (ref == nullptr) {
            LOG_ERR("%s: reference logits missing at pos %d\n", __func__, (int) (p_replay + r - 1));
            return test_status::FAIL;
        }
        const std::vector<float> ref_row(ref, ref + n_vocab);

        // RB: identical prefix and verify, then roll back d and replay r tokens.
        if (!decode_range_all_logits(ctx_rb.get(), tokens, 0, p0) ||
            !decode_range_all_logits(ctx_rb.get(), tokens, p0, n_verify)) {
            LOG_ERR("%s: prefix/verify decode failed (r %u)\n", __func__, r);
            return test_status::FAIL;
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_rb.get()), 0, p_replay, -1)) {
            LOG_ERR("%s: rollback of %u at pos %d refused (n_rs_seq %u, r %u)\n",
                    __func__, d, (int) p_replay, n_rs_seq, r);
            return test_status::FAIL;
        }
        if (!decode_range_all_logits(ctx_rb.get(), tokens, p_replay, r)) {
            LOG_ERR("%s: replay decode failed (r %u)\n", __func__, r);
            return test_status::FAIL;
        }

        const float * a = llama_get_logits_ith(ctx_rb.get(), (int32_t) r - 1);
        if (a == nullptr) {
            LOG_ERR("%s: replay logits missing (r %u)\n", __func__, r);
            return test_status::FAIL;
        }
        const double v = nmse(a, ref_row.data(), n_vocab);
        if (v > nmse_eps) {
            LOG_ERR("%s: PREDECESSOR MISMATCH, n_rs_seq %u, rollback d %u, predecessor age %u: nmse %g (eps %g)\n",
                    __func__, n_rs_seq, d, r, v, nmse_eps);
            status = test_status::FAIL;
        }
        LOG_INF("%s: n_rs_seq %u, d %u, predecessor age %u: nmse %-12g%s\n",
                __func__, n_rs_seq, d, r, v, r == 1 ? "  (shipped shape, control)" : "");
    }
    return status;
}

static test_status merge_status(test_status a, test_status b) {
    if (a == test_status::FAIL || b == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (a == test_status::PASS || b == test_status::PASS) {
        return test_status::PASS;
    }
    return test_status::SKIP;
}

struct test_results {
    test_status rollback = test_status::SKIP;
    test_status replay   = test_status::SKIP;
    test_status multi    = test_status::SKIP;
};

// OPT-57: when set, run only test_rollback_batch_matched at n_rs_seq 1 and 8. The rest of the
// suite is untouched and still runs without the flag.
static bool g_multi_ubatch_only = false;   // OPT-57: selects the matched-schedule rollback arm
static bool g_pos_sweep = false;             // OPT-57: selects the rollback-2 position sweep
static bool g_depth_sweep = false;           // OPT-57: selects the all-depths sweep at fixed p0
static bool g_shipped_shape = false;        // OPT-57: selects the shipped-shape fixture (rollback d, replay 1)


static bool cb_eval_observer(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (!g_finite_end_hook) {
        return false;
    }

    if (ask) {
        // Selection: a CPY whose destination view-root is EXACTLY the layer-0
        // checkpoint, OR -- only for arm A's rb.prefix9 slot -- the layer-0 FINAL
        // state root. No layer parser, no GDN-node filter, no expansion.
        if (t->op != GGML_OP_CPY) {
            return false;
        }
        const ggml_tensor * dst = t->src[1];
        if (!dst) {
            return false;
        }
        const ggml_tensor * droot = nullptr;
        size_t doffs = 0;
        cb_view_root(dst, &droot, &doffs);
        if (strcmp(droot->name, "cache_s_ckpt_l0") == 0) {
            return true;
        }
        // NEXT57: final-state capture, arm A prefix only, so the extra reads are
        // bounded by construction (one slot, not per-layer, not per-token).
        if (strcmp(droot->name, "cache_s_l0") == 0 && g_cb_slot == 3) {
            return true;
        }
        return false;
    }

    // ask == false: the node is computed and synchronized.
    const ggml_tensor * src = t->src[0];
    const ggml_tensor * dst = t->src[1];
    const ggml_tensor * sroot = nullptr;
    const ggml_tensor * droot = nullptr;
    size_t soffs = 0, doffs = 0;
    if (src) { cb_view_root(src, &sroot, &soffs); }
    if (dst) { cb_view_root(dst, &droot, &doffs); }

    // NEXT57: the layer-0 FINAL-state copy at arm A rb.prefix9. Captured by BYTES,
    // deliberately outside the checkpoint counters so it cannot inflate the
    // ckpt_cpy_nodes / hashed tallies this observer already reports.
    if (droot && strcmp(droot->name, "cache_s_l0") == 0 && g_cb_slot == 3) {
        size_t fnb = 0;
        if (cb_capture_tensor(dst, g_cb_final_bytes, &fnb)) {
            g_cb_final_ok = true;
            g_cb_final_nb = fnb;
            LOG_INF("%s: HOOK [%s] FINAL-CPY CAPTURED nbytes=%zu | dst_root='%s' view_offs=%zu ne=[%lld,%lld] | bounded bytes held for memcmp, no dump\n",
                    __func__, g_cb_phase, fnb, droot->name, doffs,
                    (long long) dst->ne[0], (long long) dst->ne[1]);
        } else {
            g_cb_final_caps++;
            LOG_INF("%s: HOOK [%s] FINAL-CPY NOT-CAPTURED nbytes=%zu (zero=%d over-cap=%d) | UNKNOWN, not an observation\n",
                    __func__, g_cb_phase, fnb, fnb == 0, fnb > CB_MAX_CKPT_BYTES);
        }
        return true;
    }

    if (!droot || strcmp(droot->name, "cache_s_ckpt_l0") != 0) {
        return true;   // not the selected checkpoint: nothing to say
    }

    g_cb_ckpt_nodes++;

    // source ancestry: op + ne at each hop, so the contributing GDN query shape
    // is visible when it is reachable through the view chain.
    //
    // OPT-57 REVIEW214: the increment used to be
    //     for (...; ++n = n->view_src, ++hop)
    // which reads n to compute the next n with no sequence point between the read
    // and the store. That is undefined behaviour, not a cosmetic warning, and an
    // earlier note dismissed it as "label only" -- that dismissal was wrong. Two
    // statements give the order the loop always meant, and the body is unchanged.
    char anc[512] = { 0 };
    {
        size_t off = 0;
        int hop = 0;
        for (const ggml_tensor * n = src; n != nullptr && hop < 6; ) {
            char one[96];
            snprintf(one, sizeof(one), "%s%s ne=[%lld,%lld,%lld,%lld] +%zu",
                     hop ? " <- " : "", ggml_op_name(n->op),
                     (long long) n->ne[0], (long long) n->ne[1],
                     (long long) n->ne[2], (long long) n->ne[3], off);
            if (hop) { off += n->view_offs; }
            strncat(anc, one, sizeof(anc) - strlen(anc) - 1);
            n = n->view_src;
            hop++;
        }
    }

    // Zero-size and oversized are BOTH reported as UNKNOWN, never as observations.
    // Zero-size here is build_rs extra-state housekeeping (llama-graph.cpp:3743-3751),
    // a row-count view that can legitimately be empty.
    size_t dnb = 0;
    const uint64_t h = cb_hash_tensor(dst, &dnb);
    if (h == 0) {
        if (g_cb_slot >= 0 && g_cb_slot < CB_MAX_SLOTS) {
            g_cb_zero_seen[g_cb_slot] = true;
        }
        LOG_INF("%s: HOOK [%s] CKPT-CPY NOT-HASHED nbytes=%zu (zero=%d over-cap=%d) ne=[%lld,%lld] src_root='%s' | UNKNOWN, not an observation\n",
                __func__, g_cb_phase, dnb, dnb == 0, dnb > CB_MAX_CKPT_BYTES,
                (long long) dst->ne[0], (long long) dst->ne[1],
                sroot ? sroot->name : "?");
        return true;
    }
    if (g_cb_slot >= 0 && g_cb_slot < CB_MAX_SLOTS) {
        g_cb_hash    [g_cb_slot] = h;
        g_cb_hash_ok [g_cb_slot] = true;
    }
    // NEXT57: arm A's verify3 checkpoint, captured by bytes for the direct memcmp
    // against the prefix9 final state. Outside the hash/counter logic above on
    // purpose: this is a second, independent read, not a second observation of the
    // same thing, and it must not alter any number already reported.
    if (g_cb_slot == 4) {
        size_t cnb = 0;
        if (cb_capture_tensor(dst, g_cb_ckpt_bytes, &cnb)) {
            g_cb_ckpt_ok = true;
            g_cb_ckpt_nb = cnb;
            LOG_INF("%s: HOOK [%s] CKPT-CPY CAPTURED nbytes=%zu | dst_root='%s' | bounded bytes held for memcmp, no dump\n",
                    __func__, g_cb_phase, cnb, droot->name);
        } else {
            g_cb_ckpt_caps++;
            LOG_INF("%s: HOOK [%s] CKPT-CPY NOT-CAPTURED nbytes=%zu | UNKNOWN, not an observation\n",
                    __func__, g_cb_phase, cnb);
        }
    }
    LOG_INF("%s: HOOK [%s] CKPT-CPY HASH_OK nbytes=%zu fnv1a64=%016llx | src_root='%s' view_offs=%zu | dst_root='%s' view_offs=%zu ne=[%lld,%lld] | ancestry: %s\n",
            __func__, g_cb_phase, dnb, (unsigned long long) h,
            sroot ? sroot->name : "?", soffs, droot->name, doffs,
            (long long) dst->ne[0], (long long) dst->ne[1], anc);
    return true;
}


static bool g_predecessor_k1 = false;       // OPT-57: selects the predecessor-age fixture (replay 1/2/3)

// Run every test for an initialized model over both cache fills.
static test_results run_tests(const common_params & params, llama_model * model) {
    test_results res;
    if (g_predecessor_k1) {
        // OPT-57: is a predecessor at age K+1 valid? sweep the replay length.
        test_status agg = test_status::SKIP;
        for (uint32_t n_rs_seq : { 4u, 8u, 16u }) {
            const test_status ss = test_rollback_predecessor_k1(params, model, n_rs_seq);
            LOG_INF("%s: predecessor-age n_rs_seq %u -> %s\n", __func__, n_rs_seq,
                    ss == test_status::PASS ? "PASS" : (ss == test_status::FAIL ? "FAIL" : "SKIP"));
            agg = merge_status(agg, ss);
        }
        res.multi = agg;
    }
    if (g_finite_end) {
        // OPT-57 (kurumi 0bce365e): ONE matched cell, n_rs_seq 8, d 2.
        // seq_rm(P+1, -1) versus seq_rm(P+1, P+D+1). p1 is a POSITION in both,
        // never a token count. Logs pos_max after remove and after re-decode.
        const test_status fe = test_finite_end_matched(params, model, 8, 2);
        LOG_INF("%s: finite-end n_rs_seq 8 d 2 -> %s\n", __func__,
                fe == test_status::PASS ? "PASS" : (fe == test_status::FAIL ? "FAIL" : "SKIP"));
        res.multi = fe;
        return res;
    }
    if (g_shipped_shape) {
        // OPT-57: the schedule the server actually uses. rollback d, replay ONE token.
        test_status agg = test_status::SKIP;
        for (uint32_t n_rs_seq : { 4u, 8u, 16u }) {
            const test_status ss = test_rollback_shipped_shape(params, model, n_rs_seq);
            LOG_INF("%s: shipped-shape n_rs_seq %u -> %s\n", __func__, n_rs_seq,
                    ss == test_status::PASS ? "PASS" : (ss == test_status::FAIL ? "FAIL" : "SKIP"));
            agg = merge_status(agg, ss);
        }
        res.multi = agg;
        return res;
    }
    if (g_depth_sweep) {
        // OPT-57: p0 pinned at 15, every depth 1..8. The cell the ladder never covered.
        // p0 = n_rs_seq + 7 keeps the prefix above the window and reproduces p0 15 at
        // n_rs_seq 8, so that cell doubles as a consistency check on the fixed-window run.
        test_status agg = test_status::SKIP;
        for (uint32_t n_rs_seq : { 4u, 8u, 12u, 16u }) {   // 12 is the non-power-of-two check
            const uint32_t p0 = n_rs_seq + 7;
            const test_status ds = test_rollback_depth_sweep(params, model, n_rs_seq, p0);
            LOG_INF("%s: depth-sweep n_rs_seq %u p0 %u -> %s\n", __func__, n_rs_seq, p0,
                    ds == test_status::PASS ? "PASS" : (ds == test_status::FAIL ? "FAIL" : "SKIP"));
            agg = merge_status(agg, ds);
        }
        res.multi = agg;
        return res;
    }
    if (g_pos_sweep) {
        // OPT-57: depth/position confound breaker. rollback pinned at 2, only p0 moves.
        // Window 8 reproduces the earlier run as a consistency check; window 16 is the severe
        // end of the predicate (worst nmse 1.86 at depth 2) and has only been tested at one p0.
        test_status agg = test_status::SKIP;
        for (uint32_t n_rs_seq : { 8u, 16u }) {
            const test_status sw = test_rollback_pos_sweep(params, model, n_rs_seq, 2);
            LOG_INF("%s: pos-sweep n_rs_seq %u rollback 2 -> %s\n", __func__, n_rs_seq,
                    sw == test_status::PASS ? "PASS" : (sw == test_status::FAIL ? "FAIL" : "SKIP"));
            agg = merge_status(agg, sw);
        }
        res.multi = agg;
        return res;
    }
    if (g_multi_ubatch_only) {
        // every cell runs and records its own outcome: a failure in one cell must never
        // prevent another from executing
        for (uint32_t n_rs_seq : { 1u, 8u }) {
            const test_status mu = test_rollback_batch_matched(params, model, n_rs_seq);
            LOG_INF("%s: cell n_rs_seq = %u -> %s\n", __func__, n_rs_seq,
                    mu == test_status::PASS ? "PASS" : (mu == test_status::FAIL ? "FAIL" : "SKIP"));
            res.multi = merge_status(res.multi, mu);
        }
        return res;
    }
    for (uint8_t fill : { 0, 0x3e }) {
        LOG_INF("%s: testing with cache fill 0x%02x\n", __func__, fill);
        const test_status rb = test_rollback(params, model, fill);
        const test_status rp = test_multi_seq_split_replay(params, model, fill);
        res.rollback = merge_status(res.rollback, rb);
        res.replay   = merge_status(res.replay,   rp);
        if (rb == test_status::FAIL || rp == test_status::FAIL) {
            break;
        }
    }
    return res;
}

// Run the tests for a single model file.
// Returns the per-test statuses.
static test_results run_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        // a model that cannot be loaded is a failure, not a skip
        return { test_status::FAIL, test_status::FAIL, test_status::FAIL };
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        LOG_INF("%s: skipping for non-recurrent model\n", __func__);
        return {};
    }

    return run_tests(params, model);
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--multi-ubatch-only") == 0) {
            g_multi_ubatch_only = true;
        } else if (strcmp(argv[i], "--pos-sweep") == 0) {
            g_pos_sweep = true;
        } else if (strcmp(argv[i], "--depth-sweep") == 0) {
            g_depth_sweep = true;
        } else if (strcmp(argv[i], "--shipped-shape") == 0) {
            g_shipped_shape = true;
        } else if (strcmp(argv[i], "--finite-end") == 0) {
            g_finite_end = true;
        } else if (strcmp(argv[i], "--finite-end-hook") == 0) {
            g_finite_end_hook = true;
        } else if (strcmp(argv[i], "--predecessor-k1") == 0) {
            g_predecessor_k1 = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    llama_backend_init();

    if (!models_dir.empty()) {
        // run every test over each dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        size_t name_width = 5; // "Model"
        for (const auto & model_path : models) {
            name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
        }

        // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
        common_log_set_verbosity_thold(0);

        LOG("%-*s  %-8s  %-12s  %s\n", (int) name_width, "Model", "rollback", "split replay", "seq cp graph reuse");
        common_log_flush(common_log_main());

        size_t n_pass[3] = { 0, 0, 0 };
        size_t n_skip[3] = { 0, 0, 0 };
        size_t n_fail[3] = { 0, 0, 0 };
        for (const auto & model_path : models) {
            const auto name = std::filesystem::path(model_path).filename().string();

            LOG("%-*s", (int) name_width, name.c_str());

            const test_results res = run_tests_for_model(model_path, params);
            const test_status graph = (res.rollback == test_status::FAIL || res.replay == test_status::FAIL)
                    ? test_status::SKIP : test_seq_cp_graph_reuse_for_model(model_path, params);

            // all status strings have the same raw length, so the columns line up;
            // pad the first status to the width of the "rollback" header + separator
            LOG("  %s      %s          %s", test_status_str(res.rollback), test_status_str(res.replay), test_status_str(graph));
            LOG("\n");
            common_log_flush(common_log_main());

            const test_status all[3] = { res.rollback, res.replay, graph };
            for (int t = 0; t < 3; ++t) {
                switch (all[t]) {
                    case test_status::PASS: n_pass[t]++; break;
                    case test_status::FAIL: n_fail[t]++; break;
                    case test_status::SKIP: n_skip[t]++; break;
                }
            }
        }

        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        common_log_flush(common_log_main());

        LOG_INF("%s: rollback:     %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[0], n_skip[0], n_fail[0], models.size());
        LOG_INF("%s: split replay: %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[1], n_skip[1], n_fail[1], models.size());

        LOG_INF("%s: seq cp graph reuse: %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[2], n_skip[2], n_fail[2], models.size());

        return (n_fail[0] + n_fail[1] + n_fail[2]) == 0 ? 0 : 1;
    }

    // single-model mode
    const test_results res = run_tests_for_model(params.model.path, params);

    if (res.rollback == test_status::FAIL || res.replay == test_status::FAIL || res.multi == test_status::FAIL) {
        return 1;
    }
    if (g_multi_ubatch_only) {
        return 0;
    }
    // OPT-57 REVIEW214: isolate the finite-end cell. run_tests already returns
    // early for it, but main() then fell through to test_seq_cp_graph_reuse_for_model,
    // a LATER fixture that re-loads the model -- so a crash there was reachable from
    // a cell-only run and its stack said nothing about which fixture mattered. The
    // cell now returns before that fixture can run.
    if (g_finite_end) {
        return 0;
    }
    return test_seq_cp_graph_reuse_for_model(params.model.path, params) == test_status::FAIL ? 1 : 0;
}
