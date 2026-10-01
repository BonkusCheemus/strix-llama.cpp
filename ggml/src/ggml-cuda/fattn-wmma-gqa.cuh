#pragma once

#include "common.cuh"

// RDNA3 WMMA prefill attention, D=256, even GQA ratio (see fattn-wmma-gqa.cu). Opt-in: GGML_FA_WMMA_GQA=1.
bool ggml_cuda_flash_attn_ext_wmma_gqa_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_wmma_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
