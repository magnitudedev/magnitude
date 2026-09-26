//! Scalar functions of the portable bodies, evaluated with the platform's
//! correctly rounded or faithfully rounded `f32` library. They are scalar
//! calls, so every tier produces the same bits.

#[inline(always)]
pub fn exp(x: f32) -> f32 {
    x.exp()
}

#[inline(always)]
pub fn exp2(x: f32) -> f32 {
    x.exp2()
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x * sigmoid(x)`, as `x / (1 + exp(-x))`.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `log(1 + exp(x))`, as `max(x, 0) + log(1 + exp(-|x|))`.
#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// `tanh`-form GELU.
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

/// `sin` and `cos` of `angle`.
#[inline(always)]
pub fn sin_cos(angle: f32) -> (f32, f32) {
    angle.sin_cos()
}
