#if !defined(GGML_FA_TYPES_COMP)
#define GGML_FA_TYPES_COMP

// FaTypeK / FaTypeV spec constant values. These mirror enum ggml_type so the
// host can pass the type directly. Keep in sync with ggml.h.
#define FA_TYPE_F32   0u
#define FA_TYPE_F16   1u
#define FA_TYPE_Q4_0  2u
#define FA_TYPE_Q4_1  3u
#define FA_TYPE_Q5_0  6u
#define FA_TYPE_Q5_1  7u
#define FA_TYPE_Q8_0  8u
#define FA_TYPE_IQ4_NL 20u
#define FA_TYPE_BF16 30u

#endif // !defined(GGML_FA_TYPES_COMP)

// TurboQuant / dynamic KV types (ggml ids 41-47)
#define FA_TYPE_Q1_0 41u
#define FA_TYPE_Q2_0 42u
#define FA_TYPE_TURBO2_0 43u
#define FA_TYPE_TURBO3_0 44u
#define FA_TYPE_TQ3_1S 45u
#define FA_TYPE_TQ4_1S 46u
#define FA_TYPE_TURBO4_0 47u
