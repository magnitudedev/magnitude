#ifndef SEISMIC_MATH_ADAPTER_H
#define SEISMIC_MATH_ADAPTER_H
/* Only round-to-nearest values are part of Seismic's primitive contract;
   host fenv flags and errno are not observable on this backend. */
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef float float_t;
typedef double double_t;
#define FLT_EVAL_METHOD 0
#define DBL_EPSILON 0x1p-52
#define LDBL_MANT_DIG 53
#define WANT_ROUNDING 1
#define TOINT_INTRINSICS 0
#define M_PI_2 0x1.921fb54442d18p+0
#define predict_false(x) __builtin_expect(!!(x),0)
#define FORCE_EVAL(x) ((void)0)
#define GET_FLOAT_WORD(i,x) ((i)=asuint(x))
#define hidden
#define init_jk seismic_math_init_jk
#define ipio2 seismic_math_ipio2
#define PIo2 seismic_math_PIo2
#define sinf seismic_sin
#define cosf seismic_cos
#define logf seismic_log
#define scalbn seismic_math_scalbn
#define floor seismic_math_floor
#define __sindf seismic_math_sindf
#define __cosdf seismic_math_cosdf
#define __rem_pio2f seismic_math_rem_pio2f
#define __rem_pio2_large seismic_math_rem_pio2_large
#define __logf_data seismic_math_logf_data
static inline uint32_t asuint(float x){union {float f;uint32_t u;} bits={x};return bits.u;}
static inline float asfloat(uint32_t x){union {uint32_t u;float f;} bits={x};return bits.f;}
static inline float eval_as_float(double x){return (float)x;}
static inline float __math_divzerof(uint32_t sign){return asfloat(0x7f800000u|(sign<<31));}
static inline float __math_invalidf(float x){return (x-x)/0.0f;}
float __sindf(double);
float __cosdf(double);
int __rem_pio2f(float,double*);
int __rem_pio2_large(double*,double*,int,int,int);
double scalbn(double,int);
double floor(double);
#endif
