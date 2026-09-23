//! The CPU native float conformance rows (design A8 §2.2.2, G-A8-new-7).
//!
//! `profile.rs` is the only code owner of the rows and records each row's
//! basis; A8 §2.2.2's CPU row (which contract §2.12.2 cites) is transcribed
//! here and compared with the stated table row for row.
use crate::open::open_host;
use seismic_ir::kernel::ops::SubgroupCombine;
use seismic_ir::target::{NativeFloatBehavior, NativeFloatOp};
use seismic_lang::types::DType;

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

/// A8 §2.2.2's CPU row, transcribed. A row change is an edit to that table
/// first, then to `profile.rs` and this copy.
fn design_rows() -> Vec<((NativeFloatOp, DType), NativeFloatBehavior)> {
    use NativeFloatBehavior::{Deviating, Exact};
    use NativeFloatOp as Op;
    let mut rows = Vec::new();
    for dtype in FLOATS {
        // Add, Sub, Mul, Div, Fma, Sqrt, Min, Max, Compare: Exact. F32: Contract
        // (IEEE instructions under the strict floating environment); F16/BF16:
        // Proof (f32 then RNE; f64 round-to-odd for Fma).
        for op in [
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Fma,
            Op::Sqrt,
            Op::Min,
            Op::Max,
            Op::Compare,
        ] {
            rows.push(((op, dtype), Exact));
        }
        // Every conversion: Exact.
        for from in FLOATS.into_iter().filter(|from| *from != dtype) {
            rows.push(((Op::ConvertFrom(from), dtype), Exact));
        }
        rows.push(((Op::ConvertToInteger, dtype), Exact));
        // Rsqrt, Exp, Log, Sin, Cos: Deviating (libm imports).
        for op in [Op::Rsqrt, Op::Exp, Op::Log, Op::Sin, Op::Cos] {
            rows.push(((op, dtype), Deviating));
        }
    }
    // F32 Rem: Contract (host `fmodf`, exact by C99 F.10.7.1).
    rows.push(((Op::Rem, DType::F32), Exact));
    rows
}

/// Every key the conformance vocabulary can name, so that a stated row the
/// design lacks (a subgroup or fragment row, for example) is found as well as
/// a missing one.
fn every_key() -> Vec<(NativeFloatOp, DType)> {
    use NativeFloatOp as Op;
    let mut ops = vec![
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Rem,
        Op::Min,
        Op::Max,
        Op::Fma,
        Op::Sqrt,
        Op::Rsqrt,
        Op::Exp,
        Op::Log,
        Op::Sin,
        Op::Cos,
        Op::Compare,
        Op::ConvertToInteger,
    ];
    ops.extend(DType::ALL.map(Op::ConvertFrom));
    ops.extend([SubgroupCombine::Add, SubgroupCombine::Max, SubgroupCombine::Min].map(Op::SubgroupFold));
    ops.extend(DType::ALL.map(Op::FragmentMultiplyAccumulate));
    ops.into_iter()
        .flat_map(|op| DType::ALL.map(move |dtype| (op, dtype)))
        .collect()
}

#[test]
fn stated_rows_equal_the_design_table() {
    let expected = design_rows();
    for (index, (key, _)) in expected.iter().enumerate() {
        assert!(
            expected[..index].iter().all(|(earlier, _)| earlier != key),
            "the transcription states {key:?} twice"
        );
    }
    let opened = open_host().expect("CPU host");
    let table = opened.device.native_float();
    let mismatches: Vec<String> = every_key()
        .into_iter()
        .filter_map(|(op, dtype)| {
            let design = expected
                .iter()
                .find(|(key, _)| *key == (op, dtype))
                .map(|(_, behavior)| *behavior);
            let stated = table.behavior(op, dtype);
            (stated != design)
                .then(|| format!("{op:?} {dtype:?}: profile.rs states {stated:?}, design {design:?}"))
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} row(s) differ from A8 §2.2.2's CPU row:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// The CPU target has no subgroup: one owner of that fact (C4a-1 (a)), and no
/// fragment format can exist without one.
#[test]
fn the_cpu_target_has_no_subgroup() {
    let opened = open_host().expect("CPU host");
    assert_eq!(opened.device.limits().subgroup_width, None);
    assert!(opened.device.fragment_formats().is_empty());
}
