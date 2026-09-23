#define DEFINE_VISION_DENSE(NAME, PREFIX) \
inline float NAME(device const uchar *base, ulong logical) { \
  if (PREFIX##_KIND == 0) return *reinterpret_cast<device const float *>(base + logical * PREFIX##_PACKET_SIZE); \
  if (PREFIX##_KIND == 1) return float(*reinterpret_cast<device const half *>(base + logical * PREFIX##_PACKET_SIZE)); \
  return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * PREFIX##_PACKET_SIZE)) << 16); \
}

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#define A_KIND 0
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
#define A_KIND 1
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
#define A_KIND 2
#else
#error "vision merger requires dense activations"
#endif
#define A_PACKET_SIZE SEISMIC_ELEMENT_A_PACKET_SIZE
#define DECLARE_DENSE_KIND(PREFIX) /* marker */
#if defined(SEISMIC_NORM_WEIGHT_REPRESENTATION_F32)
#define NW_KIND 0
#elif defined(SEISMIC_NORM_WEIGHT_REPRESENTATION_F16)
#define NW_KIND 1
#elif defined(SEISMIC_NORM_WEIGHT_REPRESENTATION_BF16)
#define NW_KIND 2
#else
#error "vision merger requires dense norm weights"
#endif
#if defined(SEISMIC_NORM_BIAS_REPRESENTATION_F32)
#define NB_KIND 0
#elif defined(SEISMIC_NORM_BIAS_REPRESENTATION_F16)
#define NB_KIND 1
#elif defined(SEISMIC_NORM_BIAS_REPRESENTATION_BF16)
#define NB_KIND 2
#else
#error "vision merger requires dense norm bias"
#endif
#if defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F32)
#define UW_KIND 0
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F16)
#define UW_KIND 1
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_BF16)
#define UW_KIND 2
#else
#error "vision merger currently requires dense resident up weights"
#endif
#if defined(SEISMIC_UP_BIAS_REPRESENTATION_F32)
#define UB_KIND 0
#elif defined(SEISMIC_UP_BIAS_REPRESENTATION_F16)
#define UB_KIND 1
#else
#define UB_KIND 2
#endif
#if defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_F32)
#define DW_KIND 0
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_F16)
#define DW_KIND 1
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_BF16)
#define DW_KIND 2
#else
#error "vision merger currently requires dense resident down weights"
#endif
#if defined(SEISMIC_DOWN_BIAS_REPRESENTATION_F32)
#define DB_KIND 0
#elif defined(SEISMIC_DOWN_BIAS_REPRESENTATION_F16)
#define DB_KIND 1
#else
#define DB_KIND 2
#endif
#define NW_PACKET_SIZE SEISMIC_NORM_WEIGHT_PACKET_SIZE
#define NB_PACKET_SIZE SEISMIC_NORM_BIAS_PACKET_SIZE
#define UW_PACKET_SIZE SEISMIC_UP_WEIGHT_PACKET_SIZE
#define UB_PACKET_SIZE SEISMIC_UP_BIAS_PACKET_SIZE
#define DW_PACKET_SIZE SEISMIC_DOWN_WEIGHT_PACKET_SIZE
#define DB_PACKET_SIZE SEISMIC_DOWN_BIAS_PACKET_SIZE
DEFINE_VISION_DENSE(load_a, A)
DEFINE_VISION_DENSE(load_nw, NW)
DEFINE_VISION_DENSE(load_nb, NB)
DEFINE_VISION_DENSE(load_uw, UW)
DEFINE_VISION_DENSE(load_ub, UB)
DEFINE_VISION_DENSE(load_dw, DW)
DEFINE_VISION_DENSE(load_db, DB)

inline void store_a(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE) = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE) = half(value);
#else
    uint bits = as_type<uint>(value);
    *reinterpret_cast<device ushort *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE) = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
#endif
}

inline float merger_normalized(device const uchar *hidden, device const uchar *nw,
    device const uchar *nb, ulong token, ulong column, float epsilon) {
    float sum = 0.0f, squares = 0.0f;
    for (ulong i = 0; i < SEISMIC_DIM_H; ++i) {
        float value = load_a(hidden, token * SEISMIC_HIDDEN_STRIDE_0 + i * SEISMIC_HIDDEN_STRIDE_1);
        sum += value;
        squares += value * value;
    }
    float mean = sum / float(SEISMIC_DIM_H);
    float variance = squares / float(SEISMIC_DIM_H) - mean * mean;
    float value = load_a(hidden, token * SEISMIC_HIDDEN_STRIDE_0 + column * SEISMIC_HIDDEN_STRIDE_1);
    return (value - mean) * metal::rsqrt(variance + epsilon)
        * load_nw(nw, column * SEISMIC_NORM_WEIGHT_STRIDE_0)
        + load_nb(nb, column * SEISMIC_NORM_BIAS_STRIDE_0);
}

kernel void qwen_vision_merger(
    device const uchar *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *norm_weight [[buffer(SEISMIC_BUFFER_NORM_WEIGHT)]],
    device const uchar *norm_bias [[buffer(SEISMIC_BUFFER_NORM_BIAS)]],
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],
    device const uchar *up_bias [[buffer(SEISMIC_BUFFER_UP_BIAS)]],
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],
    device const uchar *down_bias [[buffer(SEISMIC_BUFFER_DOWN_BIAS)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_D) return;
    ulong row = index / SEISMIC_DIM_D;
    ulong output = index % SEISMIC_DIM_D;
    float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    float value = load_db(down_bias, output * SEISMIC_DOWN_BIAS_STRIDE_0);
    for (ulong feature = 0; feature < SEISMIC_DIM_G * SEISMIC_DIM_H; ++feature) {
        float up = load_ub(up_bias, feature * SEISMIC_UP_BIAS_STRIDE_0);
        for (ulong source = 0; source < SEISMIC_DIM_G * SEISMIC_DIM_H; ++source) {
            ulong token = row * SEISMIC_DIM_G + source / SEISMIC_DIM_H;
            ulong column = source % SEISMIC_DIM_H;
            up = metal::fma(merger_normalized(hidden, norm_weight, norm_bias, token, column, epsilon),
                load_uw(up_weight, source * SEISMIC_UP_WEIGHT_STRIDE_0 + feature * SEISMIC_UP_WEIGHT_STRIDE_1), up);
        }
        float activated = 0.5f * up * (1.0f + metal::erf(up * 0.7071067811865475f));
        value = metal::fma(activated,
            load_dw(down_weight, feature * SEISMIC_DOWN_WEIGHT_STRIDE_0 + output * SEISMIC_DOWN_WEIGHT_STRIDE_1), value);
    }
    store_a(result, row * SEISMIC_RESULT_0_STRIDE_0 + output * SEISMIC_RESULT_0_STRIDE_1, value);
}
