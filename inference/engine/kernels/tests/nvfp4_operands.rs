//! The NVFP4 operand recipe (docs/precision.md, "NVFP4 operands"): the
//! portable `operand_rows`, whose quotients round in F32, against a host
//! reference that rounds each exact quotient by search over every value of
//! the two codes, and the registry's recipe of each representation.

use seismic_lang::{
    checked::check_source,
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

/// Every E2M1 magnitude, by code.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Every finite UE4M3 value, by code (exponent bias 7, three mantissa bits).
fn ue4m3_values() -> Vec<f32> {
    (0u32..127)
        .map(|code| {
            let (exponent, mantissa) = (code >> 3, code & 7);
            if exponent == 0 {
                mantissa as f32 * 2f32.powi(-9)
            } else {
                (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
            }
        })
        .collect()
}

/// The quotient `n / d` rounded to the nearest of `values` (ascending, by
/// code), ties to the even code; beyond the largest, the largest. F64 holds
/// every quotient of two F32 values closely enough that a distance tie is
/// exact: a midpoint of either grid has at most five significant bits.
fn nearest_even(values: &[f32], n: f32, d: f32) -> f32 {
    let v = f64::from(n) / f64::from(d);
    let mut best = 0;
    for (code, value) in values.iter().enumerate() {
        let distance = (f64::from(*value) - v).abs();
        let held = (f64::from(values[best]) - v).abs();
        if distance < held || (distance == held && code % 2 == 0) {
            best = code;
        }
    }
    values[best]
}

/// The recipe of one row, by search.
fn reference_row(row: &[f32]) -> Vec<f32> {
    let scales = ue4m3_values();
    let amax = row.iter().fold(0f32, |held, value| held.max(value.abs()));
    let mut g = 2f32.powi(-126);
    while 2688.0 * g < amax {
        g *= 2.0;
    }
    row.chunks(16)
        .flat_map(|block| {
            let block_max = block.iter().fold(0f32, |held, value| held.max(value.abs()));
            let d = g * nearest_even(&scales, block_max, 6.0 * g);
            block
                .iter()
                .map(move |value| {
                    let m = if d == 0.0 {
                        6.0
                    } else {
                        nearest_even(&E2M1, value.abs(), d)
                    };
                    m.copysign(*value) * d
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `operand_rows` over `rows` with recipe `recipe`, in F32.
fn portable(rows: &[Vec<f32>], recipe: i32) -> Vec<Vec<f32>> {
    let module = check_source(seismic_std::sources()).unwrap();
    let bindings = ElementBindings::new()
        .bind("T", registry::dense(DType::F32))
        .bind("V", registry::dense(DType::F32));
    let logical = module
        .entry(module.entry_named("operand_rows").unwrap(), &bindings)
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let width = rows[0].len();
    let x = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![rows.len(), width],
        rows.iter().flatten().map(|value| f64::from(*value)).collect(),
    ));
    let y = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![rows.len(), width],
        vec![0.0; rows.len() * width],
    ));
    let outcome = interpreter
        .run_bounded(
            &[
                Arg::Tensor(x),
                Arg::Scalar(ReferenceScalar::I32(recipe)),
                Arg::Tensor(y),
            ],
            u64::MAX,
        )
        .unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable operand_rows failed: {failure}");
    }
    let output = outcome
        .inputs()
        .find(|input| input.ordinal() == 2)
        .expect("y is written");
    let reader = output.tensor();
    (0..rows.len())
        .map(|row| {
            (0..width)
                .map(|k| reader.read(row * width + k).unwrap() as f32)
                .collect()
        })
        .collect()
}

/// Rows across binades, signs, ties, quotients beside ties and an all-zero
/// block, with a trailing partial block.
fn rows() -> Vec<Vec<f32>> {
    let mut state = 0x9e37_79b9u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let width = 16 * 5 + 7;
    (0..7)
        .map(|row| {
            (0..width)
                .map(|k| {
                    let unit = next() as f32 / u32::MAX as f32 * 2.0 - 1.0;
                    let binade = 2f32.powi((next() % 24) as i32 - 12 + 6 * row);
                    match k / 16 {
                        2 => 0.0,
                        // Exact ties of the E2M1 grid under the block's scale.
                        3 if row == 0 => [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0][k % 8],
                        // One ulp beside ties of the E2M1 grid under a
                        // block scale of exactly 3 (amax 18 - ulp: g is
                        // 2^-7 and s 384), where the F32 quotient is
                        // nearest the tie.
                        _ if row == 6 => {
                            let tie = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0][k % 8] * 3.0;
                            f32::from_bits(if k % 2 == 0 { tie.to_bits() + 1 } else { tie.to_bits() - 1 })
                        }
                        _ => unit * binade,
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn nvfp4_operand_rows_round_by_the_recipe() {
    let rows = rows();
    let quantized = portable(&rows, 1);
    for (row, (input, output)) in rows.iter().zip(&quantized).enumerate() {
        let expected = reference_row(input);
        for (k, (got, want)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "row {row} column {k}: x {} gave {got}, the recipe {want}",
                input[k]
            );
        }
    }
}

#[test]
fn rounded_operand_rows_are_the_rows() {
    let rows = rows();
    assert_eq!(portable(&rows, 0), rows);
}

/// Only NVFP4 weights take NVFP4 operands, and each NVFP4 storage has one
/// element of each recipe (W4A16 and W4A4).
#[test]
fn only_nvfp4_weights_take_nvfp4_operands() {
    for info in registry::representations() {
        let nvfp4 = info.representation == "nvfp4g16" || info.representation == "nvfp4_e2m1_block16";
        if info.operands == registry::OperandRecipe::Nvfp4Rows {
            assert!(nvfp4, "{}", info.name);
        }
        let recipes = registry::representations()
            .iter()
            .filter(|other| other.representation == info.representation && other.layout == info.layout)
            .map(|other| other.operands)
            .collect::<Vec<_>>();
        let expected: &[_] = if nvfp4 {
            &[registry::OperandRecipe::Rounded, registry::OperandRecipe::Nvfp4Rows]
        } else {
            &[registry::OperandRecipe::Rounded]
        };
        assert_eq!(recipes, expected, "{}", info.name);
    }
}
