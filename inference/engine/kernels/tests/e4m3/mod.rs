//! The reference value of an E4M3 byte code (OCP FP8 E4M3,
//! `torch.float8_e4m3fn`), which the E4M3 tests decode against apart from
//! the registry's own decode.

/// `S.EEEE.MMM`: (1 + M/8) 2^(E - 7), or M 2^-9 where E is 0; no
/// infinities, the sign kept on zero; both NaN codes S.1111.111 the
/// registry's one NaN (`f32::NAN`, 0x7fc00000), whichever their sign.
pub fn value(code: u8) -> f32 {
    let (exponent, mantissa) = (i32::from(code >> 3 & 15), f32::from(code & 7));
    if code & 0x7f == 0x7f {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    if code & 0x80 != 0 { -magnitude } else { magnitude }
}
