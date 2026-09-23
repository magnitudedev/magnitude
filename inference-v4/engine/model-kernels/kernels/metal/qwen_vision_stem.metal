#define VISION_DENSE_LOAD(NAME, PREFIX) \
inline float NAME(device const uchar *base, ulong logical) { \
  /* Vision bootstrap favors correctness; packed projector weights remain explicit unsupported bindings. */ \
  if (PREFIX##_KIND == 0) return *reinterpret_cast<device const float *>(base + logical * PREFIX##_PACKET_SIZE); \
  if (PREFIX##_KIND == 1) return float(*reinterpret_cast<device const half *>(base + logical * PREFIX##_PACKET_SIZE)); \
  if (PREFIX##_KIND == 2) return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * PREFIX##_PACKET_SIZE)) << 16); \
  return 0.0f; \
}

#if defined(SEISMIC_TEMPORAL_WEIGHT_0_REPRESENTATION_F32)
#define TEMPORAL_WEIGHT_0_KIND 0
#elif defined(SEISMIC_TEMPORAL_WEIGHT_0_REPRESENTATION_F16)
#define TEMPORAL_WEIGHT_0_KIND 1
#elif defined(SEISMIC_TEMPORAL_WEIGHT_0_REPRESENTATION_BF16)
#define TEMPORAL_WEIGHT_0_KIND 2
#else
#error "native vision stem requires dense first temporal patch weights"
#endif
#if defined(SEISMIC_TEMPORAL_WEIGHT_1_REPRESENTATION_F32)
#define TEMPORAL_WEIGHT_1_KIND 0
#elif defined(SEISMIC_TEMPORAL_WEIGHT_1_REPRESENTATION_F16)
#define TEMPORAL_WEIGHT_1_KIND 1
#elif defined(SEISMIC_TEMPORAL_WEIGHT_1_REPRESENTATION_BF16)
#define TEMPORAL_WEIGHT_1_KIND 2
#else
#error "native vision stem requires dense second temporal patch weights"
#endif
#if defined(SEISMIC_BIAS_REPRESENTATION_F32)
#define BIAS_KIND 0
#elif defined(SEISMIC_BIAS_REPRESENTATION_F16)
#define BIAS_KIND 1
#elif defined(SEISMIC_BIAS_REPRESENTATION_BF16)
#define BIAS_KIND 2
#else
#error "native vision stem requires dense bias"
#endif
#if defined(SEISMIC_TABLE_REPRESENTATION_F32)
#define TABLE_KIND 0
#elif defined(SEISMIC_TABLE_REPRESENTATION_F16)
#define TABLE_KIND 1
#elif defined(SEISMIC_TABLE_REPRESENTATION_BF16)
#define TABLE_KIND 2
#else
#error "native vision stem requires a dense position table"
#endif
#define TEMPORAL_WEIGHT_0_PACKET_SIZE SEISMIC_TEMPORAL_WEIGHT_0_PACKET_SIZE
#define TEMPORAL_WEIGHT_1_PACKET_SIZE SEISMIC_TEMPORAL_WEIGHT_1_PACKET_SIZE
#define BIAS_PACKET_SIZE SEISMIC_BIAS_PACKET_SIZE
#define TABLE_PACKET_SIZE SEISMIC_TABLE_PACKET_SIZE
VISION_DENSE_LOAD(load_temporal_weight_0, TEMPORAL_WEIGHT_0)
VISION_DENSE_LOAD(load_temporal_weight_1, TEMPORAL_WEIGHT_1)
VISION_DENSE_LOAD(load_bias, BIAS)
VISION_DENSE_LOAD(load_table, TABLE)

inline void store_activation(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE) = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE) = half(value);
#else
    uint bits = as_type<uint>(value);
    *reinterpret_cast<device ushort *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE) = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
#endif
}

kernel void qwen_vision_stem(
    device const float *pixels [[buffer(SEISMIC_BUFFER_PIXELS)]],
    device const uchar *temporal_weight_0 [[buffer(SEISMIC_BUFFER_TEMPORAL_WEIGHT_0)]],
    device const uchar *temporal_weight_1 [[buffer(SEISMIC_BUFFER_TEMPORAL_WEIGHT_1)]],
    device const uchar *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *indices [[buffer(SEISMIC_BUFFER_INDICES)]],
    device const float *coefficients [[buffer(SEISMIC_BUFFER_COEFFICIENTS)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_H) return;
    ulong row = index / SEISMIC_DIM_H;
    ulong output = index % SEISMIC_DIM_H;
    float projected = load_bias(bias, output * SEISMIC_BIAS_STRIDE_0);
    for (ulong y = 0; y < SEISMIC_DIM_P; ++y)
        for (ulong x = 0; x < SEISMIC_DIM_P; ++x)
            for (ulong channel = 0; channel < SEISMIC_DIM_C; ++channel) {
                ulong pixel_base = row * SEISMIC_PIXELS_STRIDE_0
                    + channel * SEISMIC_PIXELS_STRIDE_1
                    + y * SEISMIC_PIXELS_STRIDE_3 + x * SEISMIC_PIXELS_STRIDE_4;
                ulong weight_0_index = y * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_0
                    + x * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_1
                    + channel * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_2
                    + output * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_3;
                ulong weight_1_index = y * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_0
                    + x * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_1
                    + channel * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_2
                    + output * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_3;
                projected = metal::fma(
                    pixels[pixel_base],
                    load_temporal_weight_0(temporal_weight_0, weight_0_index), projected);
                projected = metal::fma(
                    pixels[pixel_base + SEISMIC_PIXELS_STRIDE_2],
                    load_temporal_weight_1(temporal_weight_1, weight_1_index), projected);
            }
    for (ulong corner = 0; corner < 4; ++corner) {
        int table_row = indices[row * SEISMIC_INDICES_STRIDE_0 + corner * SEISMIC_INDICES_STRIDE_1];
        float coefficient = coefficients[row * SEISMIC_COEFFICIENTS_STRIDE_0
            + corner * SEISMIC_COEFFICIENTS_STRIDE_1];
        ulong table_index = output * SEISMIC_TABLE_STRIDE_0 + ulong(table_row) * SEISMIC_TABLE_STRIDE_1;
        projected = metal::fma(load_table(table, table_index), coefficient, projected);
    }
    ulong result_index = row * SEISMIC_RESULT_0_STRIDE_0 + output * SEISMIC_RESULT_0_STRIDE_1;
    store_activation(result, result_index, projected);
}
