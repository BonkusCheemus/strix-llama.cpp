// Prefill flash attention on RDNA3 WMMA for D=256 with an even GQA ratio (Qwen3.x full-attention layers:
// 24 Q heads / 4 KV heads). Adapted from gufo's WmmaCausalAttention (gufo-org/gufo,
// src/models/qwen/hip/kernels/attention_wmma.hip, MIT, Copyright (c) the gufo authors):
//   * 32 queries x 2 heads = 64 rows per block against 16 keys, so each block re-reads K/V for more rows;
//   * V reaches LDS transposed with one key per lane (conflict-free LDS writes);
//   * the next K/V tile is loaded into registers while the current tile is used.
// (Prefetching the mask values the same way was measured slower: 52.7 vs 46.3 ms at kv 45056, nb 512.)
// (Decoding turbo4 K/V inside the loads instead of converting to f16 first was also slower: 60-61 vs 46 ms.)
// Changes for ggml: Q/K/V/mask/dst through ggml strides, the KQ mask instead of an implicit causal mask
// (so multi-sequence and kv-unified batches are correct), scale as a parameter, no gate/LSE outputs.
// K/V must already be f16 (launch converts quantized K/V like the MMA path does).
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-wmma-gqa.cuh"

#if defined(RDNA3)
typedef _Float16 wg_v16h __attribute__((ext_vector_type(16)));
typedef float    wg_v8f  __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ wg_v8f wg_wmma(const wg_v16h a, const wg_v16h b, const wg_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// 16 contiguous halves; every fragment base is 16-byte aligned by construction.
static __device__ __forceinline__ wg_v16h wg_load_frag(const half * p) {
    union { wg_v16h f; uint4 u[2]; } cvt;
    cvt.u[0] = *reinterpret_cast<const uint4 *>(p);
    cvt.u[1] = *reinterpret_cast<const uint4 *>(p + 8);
    return cvt.f;
}
#endif // defined(RDNA3)

static constexpr int WG_D     = 256;  // head size
static constexpr int WG_HEADS = 2;    // Q heads per block, must divide the GQA ratio
static constexpr int WG_QROWS = 32;   // queries per block
static constexpr int WG_KEYS  = 16;   // keys per tile

__launch_bounds__(256, 2)
static __global__ void flash_attn_wmma_gqa_d256(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, float * __restrict__ dst, const float scale,
        const int n_tokens, const int n_kv, const int n_head, const int gqa,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int64_t nb31, const int64_t nb33, const int ne33) {
#if defined(RDNA3)
    constexpr int kRowBlocks     = (WG_QROWS / 16) * WG_HEADS;  // 4
    constexpr int kSTiles        = kRowBlocks;                  // one 16-key block per tile
    constexpr int kKSteps        = WG_D / 16;                   // 16
    constexpr int kKStepsPerWave = kKSteps / (8 / kSTiles);     // 8: two waves split each S tile's k range
    constexpr int kRows          = kRowBlocks * 16;             // 64
    constexpr int kSoftmaxLanes  = 256 / kRows;                 // 4
    constexpr int kPerLane       = WG_KEYS / kSoftmaxLanes;     // 4
    constexpr int kKStride       = WG_D + 8;                    // K row padding: spread fragment reads over banks
    constexpr int kVtStride      = WG_KEYS + 8;
    constexpr int kVRegs         = (WG_KEYS * WG_D) / (256 * 8);   // 2 uint4 of V per thread per tile
    constexpr int kKRegs         = (WG_KEYS * (WG_D / 8)) / 256;   // 2 uint4 of K per thread per tile
    constexpr int kKvLdsHalves   = WG_KEYS * kKStride > WG_D * kVtStride ? WG_KEYS * kKStride : WG_D * kVtStride;
    static_assert(kKStepsPerWave * 8 / kSTiles == kKSteps, "k split");

    const int tid     = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane    = threadIdx.x;
    const int wave    = threadIdx.y;
    const int sub     = lane & 15;
    const int half_id = lane >> 4;

    const int query_start      = blockIdx.x * WG_QROWS;
    const int first_query_head = blockIdx.y * WG_HEADS;
    const int seq              = blockIdx.z;
    const int kv_head          = first_query_head / gqa;

    const auto rb_head   = [&](int rb) { return first_query_head + rb % WG_HEADS; };
    const auto rb_offset = [&](int rb) { return (rb / WG_HEADS) * 16; };

    __shared__ half  kv_lds[kKvLdsHalves];
    __shared__ float s_lds[2][kSTiles][16][17];
    __shared__ half  p_lds[kRows][WG_KEYS + 8];
    __shared__ float row_sum[kRows];
    __shared__ float row_scale[kRows];

    const int s_tile = wave % kSTiles;  // S tile = row block (one key block)
    const int s_kh   = wave / kSTiles;  // which half of the head dimension

    wg_v16h q_frag[kKStepsPerWave];
    {
        const int  t    = query_start + rb_offset(s_tile) + sub;
        const bool live = t < n_tokens;
        const float * q_row = (const float *) (Q + seq*nb03 + int64_t(t)*nb01 + int64_t(rb_head(s_tile))*nb02);
#pragma unroll
        for (int ks = 0; ks < kKStepsPerWave; ++ks) {
            const int d0 = (s_kh*kKStepsPerWave + ks) * 16;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float4 f = live ? reinterpret_cast<const float4 *>(q_row + d0)[v] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                q_frag[ks][v*4 + 0] = (_Float16) (f.x * scale);
                q_frag[ks][v*4 + 1] = (_Float16) (f.y * scale);
                q_frag[ks][v*4 + 2] = (_Float16) (f.z * scale);
                q_frag[ks][v*4 + 3] = (_Float16) (f.w * scale);
            }
        }
    }

    wg_v8f o_acc[kRowBlocks][2] = {};  // this wave owns dim tiles `wave` and `wave + 8`
    float running_max = -INFINITY;     // per softmax row, held by its kSoftmaxLanes owner lanes
    float running_sum = 0.0f;

    // Softmax row owned by this thread and its mask row.
    const int sm_rg = tid / kSoftmaxLanes;
    const int sm_seg = tid % kSoftmaxLanes;
    const int sm_t   = query_start + rb_offset(sm_rg / 16) + sm_rg % 16;
    const half * mask_row = sm_t < n_tokens ?
        (const half *) (mask + (seq % ne33)*nb33 + int64_t(sm_t)*nb31) : nullptr;

    const char * k_base = K + seq*nb13 + kv_head*nb12;
    const int    v_key   = lane % WG_KEYS;
    const int    v_slice = (tid / WG_KEYS) * (kVRegs * 8);
    const char * v_base  = V + seq*nb23 + kv_head*nb22 + v_slice*sizeof(half);

    const auto load_k = [&](int key_start, uint4 * out) {
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            const int idx = tid + n*256;
            const int key = key_start + idx / (WG_D/8);
            const int d8  = (idx % (WG_D/8)) * 8;
            out[n] = key < n_kv ? *reinterpret_cast<const uint4 *>(k_base + int64_t(key)*nb11 + d8*sizeof(half))
                                : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    const auto load_v = [&](int key_start, uint4 * out) {
        const int key = key_start + v_key;
#pragma unroll
        for (int j = 0; j < kVRegs; ++j) {
            out[j] = key < n_kv ? *reinterpret_cast<const uint4 *>(v_base + int64_t(key)*nb21 + j*8*sizeof(half))
                                : make_uint4(0u, 0u, 0u, 0u);
        }
    };

    uint4 k_cur[kKRegs], v_cur[kVRegs];
    uint4 k_pre[kKRegs] = {}, v_pre[kVRegs] = {};
    load_k(0, k_cur);
    load_v(0, v_cur);

    for (int key_start = 0; key_start < n_kv; key_start += WG_KEYS) {
        // stage K from the registers the previous iteration prefetched
        __syncthreads();
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            const int idx = tid + n*256;
            *reinterpret_cast<uint4 *>(&kv_lds[(idx / (WG_D/8))*kKStride + (idx % (WG_D/8))*8]) = k_cur[n];
        }
        __syncthreads();

        // prefetch the next tile; everything below covers its latency
        const int next_start = key_start + WG_KEYS;
        if (next_start < n_kv) {
            load_k(next_start, k_pre);
            load_v(next_start, v_pre);
        }

        // S = Q K^T
        {
            wg_v8f s_acc = {};
#pragma unroll
            for (int ks = 0; ks < kKStepsPerWave; ++ks) {
                const int d0 = (s_kh*kKStepsPerWave + ks) * 16;
                s_acc = wg_wmma(q_frag[ks], wg_load_frag(&kv_lds[sub*kKStride + d0]), s_acc);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s_lds[s_kh][s_tile][2*i + half_id][sub] = s_acc[i];
            }
        }
        __syncthreads();

        // online softmax with the KQ mask: kSoftmaxLanes threads per row
        {
            const int rb  = sm_rg / 16;
            const int row = sm_rg % 16;
            float part_max = -INFINITY;
            float vals[kPerLane];
#pragma unroll
            for (int m = 0; m < kPerLane; ++m) {
                const int col = sm_seg*kPerLane + m;
                const int key = key_start + col;
                const float mv = mask_row && key < n_kv ? __half2float(mask_row[key]) : -INFINITY;
                vals[m] = mv == -INFINITY ? -INFINITY : s_lds[0][rb][row][col] + s_lds[1][rb][row][col] + mv;
                part_max = fmaxf(part_max, vals[m]);
            }
#pragma unroll
            for (int off = 1; off < kSoftmaxLanes; off <<= 1) {
                part_max = fmaxf(part_max, __shfl_xor(part_max, off, WARP_SIZE));
            }
            const float next_max    = fmaxf(running_max, part_max);
            const float prior_scale = isfinite(running_max) ? __expf(running_max - next_max) : 0.0f;
            float part_sum = 0.0f;
#pragma unroll
            for (int m = 0; m < kPerLane; ++m) {
                const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0f;
                part_sum += w;
                p_lds[sm_rg][sm_seg*kPerLane + m] = __float2half(w);
            }
#pragma unroll
            for (int off = 1; off < kSoftmaxLanes; off <<= 1) {
                part_sum += __shfl_xor(part_sum, off, WARP_SIZE);
            }
            running_max = next_max;
            running_sum = running_sum*prior_scale + part_sum;
            if (sm_seg == 0) {
                row_scale[sm_rg] = prior_scale;
            }
        }
        __syncthreads();

#pragma unroll
        for (int rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float s = row_scale[rb*16 + 2*i + half_id];
                o_acc[rb][0][i] *= s;
                o_acc[rb][1][i] *= s;
            }
        }

        // stage V transposed: lane L owns key L % 16, so consecutive lanes write consecutive columns of V^T
        // (no barrier needed: the softmax barrier already separates the S-phase reads of K)
#pragma unroll
        for (int j = 0; j < kVRegs; ++j) {
            const half * packed = reinterpret_cast<const half *>(&v_cur[j]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                kv_lds[(v_slice + j*8 + i)*kVtStride + v_key] = packed[i];
            }
        }
        __syncthreads();

        // O += P V
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const wg_v16h v_frag = wg_load_frag(&kv_lds[((wave + t*8)*16 + sub)*kVtStride]);
#pragma unroll
            for (int rb = 0; rb < kRowBlocks; ++rb) {
                o_acc[rb][t] = wg_wmma(wg_load_frag(&p_lds[rb*16 + sub][0]), v_frag, o_acc[rb][t]);
            }
        }

#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            k_cur[n] = k_pre[n];
        }
#pragma unroll
        for (int n = 0; n < kVRegs; ++n) {
            v_cur[n] = v_pre[n];
        }
    }
    if (sm_seg == 0) {
        row_sum[sm_rg] = running_sum;
    }
    __syncthreads();

    // dst layout: [D, n_head, n_tokens, n_seq] f32
#pragma unroll
    for (int rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = 2*i + half_id;
                const int tok = query_start + rb_offset(rb) + row;
                if (tok >= n_tokens) {
                    continue;
                }
                const float den = row_sum[rb*16 + row];
                dst[((int64_t(seq)*n_tokens + tok)*n_head + rb_head(rb))*WG_D + (wave + t*8)*16 + sub] =
                    den > 0.0f ? o_acc[rb][t][i] / den : 0.0f;
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst, scale, n_tokens, n_kv, n_head, gqa,
        nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33, ne33);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

bool ggml_cuda_flash_attn_ext_wmma_gqa_supported(const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    static const bool enabled = [] {
        const char * e = getenv("GGML_FA_WMMA_GQA");
        return e && atoi(e) != 0;
    }();
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    return enabled && GGML_CUDA_CC_IS_RDNA3(cc) &&
        Q->ne[0] == WG_D && K->ne[0] == WG_D && V->ne[0] == WG_D &&
        Q->ne[1] >= 64 &&                                     // prefill batches; decode/verify keep tile/vec
        K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0 && (Q->ne[2] / K->ne[2]) % WG_HEADS == 0 &&
        Q->ne[3] == K->ne[3] && Q->ne[3] == V->ne[3] &&
        mask != nullptr && mask->ne[2] == 1 &&
        dst->src[4] == nullptr && dst->src[5] == nullptr &&  // no sinks, no selected keys
        max_bias == 0.0f && logit_softcap == 0.0f &&
        Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 &&
        (K->type == GGML_TYPE_F16 || ggml_get_to_fp16_cuda(K->type)) &&
        (V->type == GGML_TYPE_F16 || ggml_get_to_fp16_cuda(V->type));
#else
    GGML_UNUSED(dst);
    return false;
#endif // GGML_USE_HIP
}

// K or V to f16 in the buffer the allocator reserved for the MMA path (same scheme as launch_fattn).
// ponytail: duplicates launch_fattn's conversion block; fold both into one helper if a third caller appears.
static const char * wg_to_f16(const ggml_tensor * T, half * T_f16, int64_t nb[4], cudaStream_t stream) {
    const char * data = (const char *) T->data;
    nb[1] = T->nb[1]; nb[2] = T->nb[2]; nb[3] = T->nb[3];
    if (T->type == GGML_TYPE_F16) {
        return data;
    }
    const size_t bs = ggml_blck_size(T->type);
    const size_t ts = ggml_type_size(T->type);
    if (ggml_is_contiguously_allocated(T)) {
        ggml_get_to_fp16_cuda(T->type)(data, T_f16, ggml_nelements(T), stream);
        nb[1] = nb[1]*bs*sizeof(half)/ts;
        nb[2] = nb[2]*bs*sizeof(half)/ts;
        nb[3] = nb[3]*bs*sizeof(half)/ts;
    } else {
        GGML_ASSERT(T->nb[0] == ts);
        ggml_get_to_fp16_nc_cuda(T->type)(data, T_f16, T->ne[0], T->ne[1], T->ne[2], T->ne[3],
            T->nb[1]/ts, T->nb[2]/ts, T->nb[3]/ts, stream);
        nb[1] = T->ne[0]*sizeof(half);
        nb[2] = T->ne[1]*nb[1];
        nb[3] = T->ne[2]*nb[2];
    }
    return (const char *) T_f16;
}

void ggml_cuda_flash_attn_ext_wmma_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    int64_t nbK[4], nbV[4];
    const char * K_data = wg_to_f16(K, (half *) f16_extra.K, nbK, stream);
    const char * V_data;
    if (V_is_K_view && V->type != GGML_TYPE_F16) {
        V_data = K_data;
        nbV[1] = nbK[1]; nbV[2] = nbK[2]; nbV[3] = nbK[3];
    } else {
        V_data = wg_to_f16(V, (half *) f16_extra.V, nbV, stream);
    }
    GGML_ASSERT(nbK[1] % 16 == 0 && nbK[2] % 16 == 0 && nbV[1] % 16 == 0 && nbV[2] % 16 == 0);

    const dim3 grid((Q->ne[1] + WG_QROWS - 1) / WG_QROWS, Q->ne[2] / WG_HEADS, Q->ne[3]);
    const dim3 block(WARP_SIZE, 8, 1);
    flash_attn_wmma_gqa_d256<<<grid, block, 0, stream>>>(
        (const char *) Q->data, K_data, V_data, (const char *) mask->data, (float *) dst->data, scale,
        Q->ne[1], K->ne[1], Q->ne[2], Q->ne[2] / K->ne[2],
        Q->nb[1], Q->nb[2], Q->nb[3], nbK[1], nbK[2], nbK[3], nbV[1], nbV[2], nbV[3],
        mask->nb[1], mask->nb[3], mask->ne[3]);
    CUDA_CHECK(cudaGetLastError());
}
