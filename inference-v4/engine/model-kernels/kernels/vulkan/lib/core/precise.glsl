// Transcendentals with a bounded error independent of the argument. Vulkan
// allows `exp` an error of 3 + 2|x| ulp, which grows past the activation
// roundings the portable bodies publish (rotary frequencies at |x| ~ 9, GELU
// arguments). These use Cody-Waite reduction by ln 2 (fused products) and a
// degree-8 Taylor polynomial on |r| <= ln(2) / 2 (truncation below 2^-27),
// so the result is within about 1 ulp. Results below the normal range flush
// to zero (`ldexp` is defined for exponents in [-126, 128] only).
//
// This file is independent of any entry ABI.

float precise_exp(float x) {
    if (x > 88.72283935546875)
        return uintBitsToFloat(0x7f800000u);
    if (x < -87.33654022216797)
        return 0.0;
    const float k = roundEven(x * 1.4426950408889634);
    float r = seismic_fma_rn(-k, 0.693145751953125, x);
    r = seismic_fma_rn(-k, 1.4286068203094172e-06, r);
    float p = 2.48015873015873e-05;
    p = seismic_fma_rn(p, r, 1.984126984126984e-04);
    p = seismic_fma_rn(p, r, 1.388888888888889e-03);
    p = seismic_fma_rn(p, r, 8.333333333333333e-03);
    p = seismic_fma_rn(p, r, 4.166666666666666e-02);
    p = seismic_fma_rn(p, r, 1.666666666666667e-01);
    p = seismic_fma_rn(p, r, 0.5);
    p = seismic_fma_rn(p, r, 1.0);
    p = seismic_fma_rn(p, r, 1.0);
    return ldexp(p, int(k));
}
