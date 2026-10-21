//! The NVFP4 operand recipe as the kernel tests' host references take it
//! (docs/precision.md, "NVFP4 operands"; `nvfp4_operands.rs` checks the
//! portable body against an independent search over both codes).

/// Rows of `k` as NVFP4 operands: per row the least power of two g at or
/// above amax / 2688; per 16 values the UE4M3 scale s of amax / (6 g); each
/// value g s E2M1(x / (g s)), every step rounded to nearest even.
pub fn nvfp4_rows(x: &[f32], k: usize) -> Vec<f32> {
    let e2m1 = |v: f32| {
        let a = v.abs();
        let m: f32 = match a {
            a if a <= 0.25 => 0.0,
            a if a < 0.75 => 0.5,
            a if a <= 1.25 => 1.0,
            a if a < 1.75 => 1.5,
            a if a <= 2.5 => 2.0,
            a if a < 3.5 => 3.0,
            a if a <= 5.0 => 4.0,
            _ => 6.0,
        };
        if v < 0.0 { -m } else { m }
    };
    let ue4m3 = |v: f32| {
        let exponent = ((v.to_bits() >> 23) & 255).max(121);
        let quantum = f32::from_bits((exponent - 3) << 23);
        ((v / quantum).round_ties_even() * quantum).min(448.0)
    };
    x.chunks(k)
        .flat_map(|row| {
            let amax = row.iter().fold(0f32, |held, v| held.max(v.abs()));
            let bits = (amax / 2688.0).to_bits();
            let g = f32::from_bits(
                ((bits >> 23) + u32::from(bits & 0x7f_ffff != 0)).clamp(1, 254) << 23,
            );
            row.chunks(16).flat_map(move |block| {
                let maximum = block.iter().fold(0f32, |held, v| held.max(v.abs()));
                let d = g * ue4m3(maximum / (6.0 * g));
                block
                    .iter()
                    .map(move |v| if d == 0.0 { 0.0 } else { e2m1(v / d) * d })
            })
        })
        .collect()
}

/// `x` (rows of `k`) as a product over weights of operand recipe `nvfp4`
/// takes it.
pub fn operand_rows(x: &[f32], k: usize, nvfp4: bool) -> Vec<f32> {
    if nvfp4 { nvfp4_rows(x, k) } else { x.to_vec() }
}
