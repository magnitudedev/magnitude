// An independent host model of the trained block roundings
// (`rounding.seismic`, DeepSeek's reference `block_round`): each run's scale
// from its largest magnitude, the F32 quotient x / s clamped, then the
// nearest value of the element's enumerated codes (ties to the code of even
// encoding), times s. Shared by the rounding and indexer tests.

#![allow(dead_code)]

/// A rounding's four entry scalars (`block`, `element`, `scale`,
/// `amax_floor`).
#[derive(Clone, Copy, Debug)]
pub struct Rounding {
    pub block: i32,
    pub element: i32,
    pub scale: i32,
    pub amax_floor: f32,
}

/// No rounding.
pub const NONE: Rounding = Rounding { block: 0, element: 0, scale: 0, amax_floor: 0.0 };
/// The window rows': E4M3 per 32, UE8M0 scales, floor 1e-4.
pub const WINDOW: Rounding = Rounding { block: 32, element: 0, scale: 0, amax_floor: 1e-4 };
/// The indexer's queries and keys: E2M1 per 32, UE8M0 scales, floor 6 2^-126.
pub const INDEX: Rounding = Rounding { block: 32, element: 1, scale: 0, amax_floor: 6.0 * f32::MIN_POSITIVE };
/// The compressed rows: E2M1 per 16, E4M3 scales, floor 6 2^-9.
pub const COMPRESSED: Rounding = Rounding { block: 16, element: 1, scale: 1, amax_floor: 6.0 / 512.0 };

/// The non-negative codes of a float format in encoding order (the index of
/// a code is its encoding without the sign): `mantissa` bits, exponent bias
/// `bias`, the largest code `max`. Encoding 0 and the subnormals first.
fn codes(mantissa: u32, bias: i32, max: f64) -> Vec<f64> {
    let mut codes = Vec::new();
    for encoding in 0u32.. {
        let (exponent, fraction) = (encoding >> mantissa, encoding & ((1 << mantissa) - 1));
        let significand = f64::from(fraction) / f64::from(1u32 << mantissa);
        let value = if exponent == 0 {
            significand * 2f64.powi(1 - bias)
        } else {
            (1.0 + significand) * 2f64.powi(exponent as i32 - bias)
        };
        if value > max {
            break;
        }
        codes.push(value);
    }
    codes
}

fn e4m3_codes() -> Vec<f64> {
    codes(3, 7, 448.0)
}

fn e2m1_codes() -> Vec<f64> {
    codes(1, 1, 6.0)
}

/// `v` to its nearest code, ties to the even encoding, with `v`'s sign; `v`
/// within the codes' range.
fn nearest(codes: &[f64], v: f32) -> f32 {
    let magnitude = f64::from(v.abs());
    let mut best = 0;
    for (index, code) in codes.iter().enumerate() {
        let (distance, best_distance) = ((code - magnitude).abs(), (codes[best] - magnitude).abs());
        if distance < best_distance || (distance == best_distance && index % 2 == 0) {
            best = index;
        }
    }
    let code = codes[best] as f32;
    if v.is_sign_negative() { -code } else { code }
}

/// The least power of two at or above `v` (> 0).
fn power_of_two_ceil(v: f32) -> f32 {
    let mut power = 2f64.powi(-149);
    while power < f64::from(v) {
        power *= 2.0;
    }
    power as f32
}

/// `row` rounded run by run, as the reference computes it in F32.
pub fn round(rounding: Rounding, row: &[f32]) -> Vec<f32> {
    if rounding.block == 0 {
        return row.to_vec();
    }
    // The E4M3 scale grid is unsaturated, as the reference's `round_codes`
    // is: past 448 it continues with E4M3's step per binade.
    let e4m3 = codes(3, 7, 2f64.powi(20));
    let (codes, max) = if rounding.element == 0 { (e4m3_codes(), 448.0f32) } else { (e2m1_codes(), 6.0f32) };
    row.chunks_exact(rounding.block as usize)
        .flat_map(|run| {
            let amax = run.iter().fold(0.0f32, |a, x| a.max(x.abs())).max(rounding.amax_floor);
            let s = if rounding.scale == 0 {
                power_of_two_ceil(amax * (1.0 / max))
            } else {
                nearest(&e4m3, amax / max)
            };
            let codes = &codes;
            run.iter().map(move |x| nearest(codes, (x / s).clamp(-max, max)) * s)
        })
        .collect()
}
