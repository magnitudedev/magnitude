// ABI bindings for entries which read attention. Cache publication does not include these.
#define ATTENTION_INPUTS()                                                                      \
    attention::Inputs {                                                                         \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_QUERY), SEISMIC_PTR(SEISMIC_BUFFER_GATE), \
            SEISMIC_PTR(SEISMIC_BUFFER_KEY), SEISMIC_PTR(SEISMIC_BUFFER_VALUE),                 \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM)),            \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM)),              \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_NORM)),            \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS)),       \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES)),             \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE)),                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRESH)),                   \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS)),            \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES)),   \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_AMPLITUDES)),    \
            element::word_f32(SEISMIC_PARAM_EPSILON),                                           \
            element::word_f32(SEISMIC_PARAM_SCALE),                                             \
            SEISMIC_PARAM_GATE_FUNCTION != 0                                                    \
    }
#define ATTENTION_QUERY_AT(row, query_head)                                              \
    (static_cast<attention::u64>(row) * SEISMIC_QUERY_STRIDE_0 +                         \
     static_cast<attention::u64>(query_head) * SEISMIC_QUERY_STRIDE_1)
#define ATTENTION_GATE_AT(row, query_head)                                               \
    (static_cast<attention::u64>(row) * SEISMIC_GATE_STRIDE_0 +                          \
     static_cast<attention::u64>(query_head) * SEISMIC_GATE_STRIDE_1)
#define ATTENTION_VISIBLE(in, row, span, bound)                                          \
    ((in).visible[static_cast<attention::u64>(row) * SEISMIC_VISIBLE_STRIDE_0 +          \
                  static_cast<attention::u64>(span) * SEISMIC_VISIBLE_STRIDE_1 +         \
                  static_cast<attention::u64>(bound) * SEISMIC_VISIBLE_STRIDE_2])
#define ATTENTION_FRESH(in, row, bound)                                                  \
    ((in).fresh[static_cast<attention::u64>(row) * SEISMIC_FRESH_STRIDE_0 +              \
                static_cast<attention::u64>(bound) * SEISMIC_FRESH_STRIDE_1])

#define ATTENTION_QUERY_NORM_STRIDE SEISMIC_QUERY_NORM_STRIDE_1
