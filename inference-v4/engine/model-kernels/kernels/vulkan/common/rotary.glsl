// Rotary-embedding angle evaluation shared by the decoder attention
// (`common/attention.glsl`) and the vision block (`common/vision.glsl`).
//
// This file is independent of any entry ABI.

// sin and cos of an F32 rotary angle (|angle| up to the context length). The
// angle is reduced to [-pi, pi] by a three-part 2*pi (Cody-Waite, fused
// products) and evaluated with the hardware functions, which are accurate on
// that interval.
float rotary_sincos(float angle, out float cosine) {
    const float turns = roundEven(angle * 0.15915494309189535);
    float reduced = seismic_fma_rn(-turns, 6.28125, angle);
    reduced = seismic_fma_rn(-turns, 0.0019354820251464844, reduced);
    reduced = seismic_fma_rn(-turns, -1.7484555314695172e-07, reduced);
    cosine = cos(reduced);
    return sin(reduced);
}
