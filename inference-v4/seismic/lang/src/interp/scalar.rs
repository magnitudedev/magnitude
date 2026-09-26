//! Interpreter consumption of language-owned scalar recipes. Values retain their
//! typed payload bits; this module only performs the checked source promotions.
use crate::intrinsics::MathOp;
use crate::reference_math::{
    evaluate as evaluate_recipe, scalar_recipe, ReferenceScalar, ScalarFailure, ScalarOp,
};
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::DType;

fn evaluate(op: ScalarOp, args: &[ReferenceScalar]) -> Result<ReferenceScalar, ScalarFailure> {
    let types: Vec<_> = args.iter().map(|v| v.dtype()).collect();
    evaluate_recipe(&scalar_recipe(op, &types), args)
}
pub(super) fn binary(
    op: BinaryOp,
    a: ReferenceScalar,
    b: ReferenceScalar,
    hint: Option<DType>,
) -> Result<ReferenceScalar, ScalarFailure> {
    let dtype = DType::promote(a.dtype(), b.dtype())
        .or_else(|| hint.filter(|d| d.is_numeric()))
        .unwrap_or(a.dtype());
    let a = cast(dtype, a);
    let b = if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        b
    } else {
        cast(dtype, b)
    };
    evaluate(ScalarOp::Binary(op), &[a, b])
}
pub(super) fn unary(op: UnaryOp, a: ReferenceScalar) -> Result<ReferenceScalar, ScalarFailure> {
    evaluate(ScalarOp::Unary(op), &[a])
}
pub(super) fn cast(to: DType, a: ReferenceScalar) -> ReferenceScalar {
    evaluate(ScalarOp::Cast(to), &[a]).expect("numeric conversion has no source failure")
}
pub(super) fn math(op: MathOp, args: &[ReferenceScalar]) -> Result<ReferenceScalar, ScalarFailure> {
    assert_eq!(
        args.len(),
        op.arity(),
        "checked scalar recipe arity differs"
    );
    let dtype = args[1..].iter().fold(args[0].dtype(), |d, a| {
        DType::promote(d, a.dtype()).unwrap_or(d)
    });
    let args: Vec<_> = args.iter().map(|&a| cast(dtype, a)).collect();
    evaluate(ScalarOp::Math(op), &args)
}
