//! Numerical probes run at open (§16.1).
//!
//! Multiply-add, on devices without `VK_KHR_shader_fma`: GLSL `fma` compiles
//! to `GLSL.std.450 Fma`, which the precision rules only require to behave
//! like a multiply followed by an add; RADV splits it when a `NoContraction`
//! product of the same operands sits next to it. Where the device has
//! `VK_KHR_shader_fma`, formation binds `fma` to `OpFmaKHR` instead and no
//! probe runs. Elsewhere (NVIDIA 580) `fma` is accepted only if it is fused
//! in exactly that pattern, and the `NoContraction` expression beside it is
//! rounded twice.
//!
//! Rounding, on devices whose modules do not declare `RoundingModeRTE 32`
//! (NVIDIA proprietary, whose compiler traps on fp32 `OpFDiv` under it): fp32
//! add, multiply and signed/unsigned integer conversion must round to nearest
//! even on witnesses where round-to-nearest-even differs from every other
//! rounding (toward zero, up, down, ties away).

use crate::device::{OpenError, Rounded};
use crate::direct::{DirectBatch, DirectLaunch};
use crate::formation::{DirectModule, Kernel};
use crate::Device;

const PRELUDE: &str = "#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(push_constant, scalar) uniform arguments_t { uint64_t arguments; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer words_t { uint64_t w[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer floats_t { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer bits_t { uint v[]; };
";

/// Form `probe` of `body` with `threads` invocations in one group, run it
/// over `input` words, and read `outputs` words written after them.
fn run(device: &Device, body: &str, threads: u32, input: &[u32], outputs: usize) -> Result<Vec<u32>, OpenError> {
    let failed = |error: &dyn std::fmt::Display| OpenError::Creation(format!("a numerical probe failed: {error}"));
    let source = format!("{PRELUDE}{body}\nvoid main() {{\n    SEISMIC_KERNEL();\n}}\n");
    let module = DirectModule::form(
        device,
        &source,
        &[Kernel {
            name: "probe",
            threads: [threads, 1, 1],
            constants: &[],
        }],
        |_| None,
        |_, _| {},
    )
    .map_err(|error| failed(&format!("{error:?}")))?;
    let bytes = ((input.len() + outputs) * 4) as u64;
    let values = device.allocate(bytes, 16).map_err(|error| failed(&error))?;
    let input_bytes = input.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    device.write(&values, 0, &input_bytes).map_err(|error| failed(&error))?;
    let mut batch = DirectBatch::new(device).map_err(|error| failed(&error))?;
    batch
        .launch(&DirectLaunch {
            module: &module,
            function: 0,
            buffers: &[],
            words: &[],
            scalar_results: (&values, 0),
            groups: [1, 1, 1],
        })
        .map_err(|error| failed(&error))?;
    batch
        .commit()
        .and_then(|submission| submission.finish())
        .map_err(|error| failed(&error))?;
    let mut results = vec![0u8; outputs * 4];
    device
        .read(&values, input_bytes.len() as u64, &mut results)
        .map_err(|error| failed(&error))?;
    Ok(results
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
        .collect())
}

/// a = b = 1 + 2^-12, c = -(1 + 2^-11): a*b = 1 + 2^-11 + 2^-24 exactly, so
/// the fused result is 2^-24 while a rounded product gives 0.
const MULTIPLY_ADD_WITNESS: [u32; 3] = [0x3f80_0800, 0x3f80_0800, 0xbf80_1000];
const FUSED: u32 = 0x3380_0000;
const ROUNDED_TWICE: u32 = 0;

/// `fma` beside `a * b + c` of the same operands; the seal pass makes the
/// latter `NoContraction` (the GLSL `precise` of the failing pattern).
const MULTIPLY_ADD: &str = "void probe() {
    floats_t values = floats_t(words_t(arguments).w[0]);
    const float a = values.v[0], b = values.v[1], c = values.v[2];
    values.v[3] = fma(a, b, c);
    values.v[4] = a * b + c;
}";

pub(crate) fn multiply_add(device: &Device) -> Result<(), OpenError> {
    let results = run(device, MULTIPLY_ADD, 1, &MULTIPLY_ADD_WITNESS, 2)?;
    let (fma, separate) = (results[0], results[1]);
    if fma == FUSED && separate == ROUNDED_TWICE {
        Ok(())
    } else {
        Err(OpenError::MultiplyAdd { fma, separate })
    }
}

/// (a, b, i): `a + b` and `a * b` as fp32 bits, `i` converted as a signed
/// and as an unsigned integer. Each non-exact result has a nearest-even
/// answer that another rounding would not give: sums and products just above
/// a half ulp (vs toward zero), exact ties to an odd and an even neighbor
/// (vs ties away), both signs (vs up and down), and integers of 25 to 32
/// significant bits.
const ROUNDING_WITNESSES: [[u32; 3]; 6] = [
    // 1 + 2^-24(1 + 2^-23); 2^24 + 1 (tie, even below).
    [0x3f80_0000, 0x3380_0001, 0x0100_0001],
    // (1 + 2^-23) + 2^-24 (tie, even above); 2^24 + 3 (tie, even above).
    [0x3f80_0001, 0x3380_0000, 0x0100_0003],
    // (1 + 2^-12 + 2^-23)(1 + 2^-12): above a half ulp; 2^31 - 1.
    [0x3f80_0801, 0x3f80_0800, 0x7fff_ffff],
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 (tie, even below); -(2^24 + 1).
    [0x3f80_0800, 0x3f80_0800, 0xfeff_ffff],
    // -1 - 2^-24(1 + 2^-23): the negative of the first sum; 2^24 - 1 exact.
    [0xbf80_0000, 0xb380_0001, 0x00ff_ffff],
    // -(1 + 2^-12 + 2^-23)(1 + 2^-12); -(2^31 - 1), and 2^31 + 1 unsigned.
    [0xbf80_0801, 0x3f80_0800, 0x8000_0001],
];

const ROUNDING: &str = "void probe() {
    bits_t values = bits_t(words_t(arguments).w[0]);
    const uint i = gl_LocalInvocationID.x;
    if (i >= 6u)
        return;
    const float a = uintBitsToFloat(values.v[3u * i]), b = uintBitsToFloat(values.v[3u * i + 1u]);
    const uint n = values.v[3u * i + 2u];
    const uint result = 18u + 4u * i;
    values.v[result] = floatBitsToUint(a + b);
    values.v[result + 1u] = floatBitsToUint(a * b);
    values.v[result + 2u] = floatBitsToUint(float(int(n)));
    values.v[result + 3u] = floatBitsToUint(float(n));
}";

/// The host's IEEE round-to-nearest-even results of one witness, in the
/// kernel's output order.
fn nearest_even([a, b, n]: [u32; 3]) -> [(Rounded, u32); 4] {
    let (a, b) = (f32::from_bits(a), f32::from_bits(b));
    [
        (Rounded::Add, (a + b).to_bits()),
        (Rounded::Multiply, (a * b).to_bits()),
        (Rounded::SignedConversion, (n as i32 as f32).to_bits()),
        (Rounded::UnsignedConversion, (n as f32).to_bits()),
    ]
}

pub(crate) fn rounding(device: &Device) -> Result<(), OpenError> {
    let input = ROUNDING_WITNESSES.iter().flatten().copied().collect::<Vec<_>>();
    let results = run(device, ROUNDING, 32, &input, 4 * ROUNDING_WITNESSES.len())?;
    for (witness, actual) in ROUNDING_WITNESSES.iter().zip(results.chunks_exact(4)) {
        for ((operation, expected), actual) in nearest_even(*witness).into_iter().zip(actual) {
            if expected != *actual {
                return Err(OpenError::Rounding {
                    operation,
                    operands: *witness,
                    expected,
                    actual: *actual,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every non-exact witness result differs from rounding toward zero.
    #[test]
    fn witnesses_distinguish_nearest_even_from_truncation() {
        let truncated = |exact: f64| {
            let nearest = exact as f32;
            if f64::from(nearest).abs() > exact.abs() {
                f32::from_bits(nearest.to_bits() - 1)
            } else {
                nearest
            }
        };
        let mut distinguishing = 0;
        for witness in ROUNDING_WITNESSES {
            let (a, b) = (f64::from(f32::from_bits(witness[0])), f64::from(f32::from_bits(witness[1])));
            let n = witness[2];
            for (exact, (_, expected)) in [a + b, a * b, f64::from(n as i32), f64::from(n)]
                .into_iter()
                .zip(nearest_even(witness))
            {
                if truncated(exact).to_bits() != expected {
                    distinguishing += 1;
                }
            }
        }
        assert!(distinguishing >= 10, "{distinguishing} results tell nearest-even from truncation");
    }
}
