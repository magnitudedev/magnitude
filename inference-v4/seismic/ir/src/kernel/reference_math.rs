//! Instantiation of the language-owned terminal scalar graph. Algorithms and
//! exceptional cases live only in seismic-lang; this module maps its vocabulary.
use super::internals::{PortableBuilder, PortableValue};
use super::ops::{self, ValueType};
use crate::target::PhysicalDialect;
use seismic_lang::intrinsics::MathOp;
use seismic_lang::reference_math as reference;
use seismic_lang::types::DType;

pub use reference::VERSION;
pub fn digest() -> [u8; 32] {
    reference::digest()
}
fn dtype(ty: ValueType) -> DType {
    match ty {
        ValueType::Scalar(dtype) => dtype,
        ValueType::Bool => DType::Bool,
        _ => panic!("scalar recipe operand has non-scalar type"),
    }
}
fn value_type(dtype: DType) -> ValueType {
    if dtype == DType::Bool {
        ValueType::Bool
    } else {
        ValueType::Scalar(dtype)
    }
}

// Internal expansion is never exposed without either consuming its failure
// effects in source continuation or establishing that the recipe is total.
struct ExpandedScalar {
    value: PortableValue,
    failures: Vec<(PortableValue, reference::ScalarFailure)>,
}

pub(crate) fn continue_scalar<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    op: reference::ScalarOp,
    inputs: &[PortableValue],
    alive: PortableValue,
    mut destination: impl FnMut(
        reference::ScalarFailure,
    ) -> (super::internals::PortableTensor, PortableValue),
) -> (PortableValue, PortableValue) {
    let result = expand_scalar(builder, op, inputs);
    let partial = !result.failures.is_empty();
    let mut successful = alive;
    for (failed, cause) in result.failures {
        let (status, index) = destination(cause);
        // Every cause uses incoming liveness. One cause stopping this participant
        // must not manufacture another cause or a failure on an inactive peer.
        let stopped = builder.not(alive);
        let passed = builder.not(failed);
        let ignored_or_passed = builder.logic(ops::LogicOp::Or, stopped, passed);
        builder.record_source_check(&status, index, ignored_or_passed);
        successful = builder.logic(ops::LogicOp::And, successful, passed);
    }
    let value = if partial {
        builder.branch(
            successful,
            |_| vec![result.value],
            |builder| {
                let zero = reference::ReferenceScalar::from_bits(dtype(result.value.ty), 0);
                vec![builder.constant(ops::ConstantValue::from_scalar(zero), result.value.ty)]
            },
        )[0]
    } else {
        result.value
    };
    (value, successful)
}

fn expand_scalar<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    op: reference::ScalarOp,
    inputs: &[PortableValue],
) -> ExpandedScalar {
    let types: Vec<_> = inputs.iter().map(|v| dtype(v.ty)).collect();
    let recipe = reference::scalar_recipe(op, &types);
    expand_recipe(builder, &recipe, inputs)
}
fn expand_recipe<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    recipe: &reference::ReferenceRecipe,
    inputs: &[PortableValue],
) -> ExpandedScalar {
    let mut values = Vec::with_capacity(recipe.nodes().len());
    for node in recipe.nodes() {
        let get = |v: reference::ReferenceValue| values[v.ordinal()];
        let value = match *node {
            reference::ReferenceNode::Input { operand, dtype } => {
                let input = inputs[operand as usize];
                assert_eq!(input.ty, value_type(dtype), "scalar recipe operand type");
                input
            }
            reference::ReferenceNode::Constant(value) => builder.constant(
                ops::ConstantValue::from_scalar(value),
                value_type(value.dtype()),
            ),
            reference::ReferenceNode::Word { op, a, b } => match op {
                reference::WordOp::Add | reference::WordOp::Sub => builder.binary_terminal(
                    if op == reference::WordOp::Add {
                        ops::BinaryOp::Add
                    } else {
                        ops::BinaryOp::Sub
                    },
                    get(a),
                    get(b),
                ),
                _ => builder.bit_terminal(
                    match op {
                        reference::WordOp::And => ops::BitOp::And,
                        reference::WordOp::Or => ops::BitOp::Or,
                        reference::WordOp::Xor => ops::BitOp::Xor,
                        reference::WordOp::Shl => ops::BitOp::Shl,
                        reference::WordOp::Shr => ops::BitOp::Shr,
                        _ => unreachable!(),
                    },
                    get(a),
                    get(b),
                ),
            },
            reference::ReferenceNode::Compare { op, a, b } => builder.cmp_terminal(
                match op {
                    reference::CmpOp::Eq => ops::CmpOp::Eq,
                    reference::CmpOp::Ne => ops::CmpOp::Ne,
                    reference::CmpOp::Lt => ops::CmpOp::Lt,
                    reference::CmpOp::Le => ops::CmpOp::Le,
                    reference::CmpOp::Gt => ops::CmpOp::Gt,
                    reference::CmpOp::Ge => ops::CmpOp::Ge,
                },
                get(a),
                get(b),
            ),
            reference::ReferenceNode::And { a, b } => {
                builder.logic(ops::LogicOp::And, get(a), get(b))
            }
            reference::ReferenceNode::Not { value } => builder.not(get(value)),
            reference::ReferenceNode::Select { condition, yes, no } => {
                builder.select(get(condition), get(yes), get(no))
            }
            reference::ReferenceNode::Bits { value } => builder.scalar_bits(get(value)),
            reference::ReferenceNode::FromBits { value, dtype } => {
                builder.scalar_from_bits(get(value), dtype)
            }
        };
        values.push(value);
    }
    ExpandedScalar {
        value: values[recipe.output().ordinal()],
        failures: recipe
            .failures()
            .iter()
            .map(|&(p, f)| (values[p.ordinal()], f))
            .collect(),
    }
}
pub(crate) fn expand_total<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    op: reference::ScalarOp,
    inputs: &[PortableValue],
) -> PortableValue {
    let types: Vec<_> = inputs.iter().map(|v| dtype(v.ty)).collect();
    let recipe = reference::scalar_recipe(op, &types);
    expand_total_recipe(builder, &recipe, inputs)
}
pub(super) fn expand_total_recipe<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    recipe: &reference::ReferenceRecipe,
    inputs: &[PortableValue],
) -> PortableValue {
    assert!(
        recipe.failures().is_empty(),
        "partial scalar operation requires a source failure continuation"
    );
    expand_recipe(builder, recipe, inputs).value
}
pub(crate) fn expand<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    op: MathOp,
    input: PortableValue,
) -> PortableValue {
    expand_total(builder, reference::ScalarOp::Math(op), &[input])
}
