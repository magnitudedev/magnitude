//! Instantiation of the language-owned reference-math recipe into typed
//! kernel IR. Algorithms and coefficients live only in `seismic-lang`.

use super::internals::{PortableBuilder, PortableValue};
use super::ops::{self, ValueType};
use crate::target::KernelDialect;
use seismic_lang::intrinsics::MathOp;
use seismic_lang::reference_math as reference;
use seismic_lang::types::DType;

pub use reference::VERSION;

pub fn digest() -> [u8; 32] {
    reference::digest()
}

fn value_type(ty: reference::ValueType) -> ValueType {
    match ty {
        reference::ValueType::Scalar(dtype) => ValueType::Scalar(dtype),
        reference::ValueType::Bool => ValueType::Bool,
    }
}

pub(crate) fn expand<B: KernelDialect>(
    builder: &mut PortableBuilder<'_, B>,
    op: MathOp,
    input: PortableValue,
) -> PortableValue {
    let reference_op = reference::ReferenceMathOp::try_from(op)
        .unwrap_or_else(|()| panic!("non-reference math operation reached exact recipe lowering"));
    let ValueType::Scalar(dtype @ (DType::F16 | DType::BF16 | DType::F32)) = input.ty else {
        panic!("reference math requires a floating scalar")
    };
    let recipe = reference::recipe(reference_op, dtype);
    let mut values = Vec::with_capacity(recipe.nodes().len());
    for node in recipe.nodes() {
        let get = |value: reference::ReferenceValue| values[value.ordinal()];
        let value = match node {
            reference::ReferenceNode::Input { .. } => input,
            reference::ReferenceNode::Constant { value, ty } => {
                let value = match *value {
                    reference::ConstantValue::F32(bits) => {
                        ops::ConstantValue::F32(f32::from_bits(bits))
                    }
                    reference::ConstantValue::U32(value) => ops::ConstantValue::U32(value),
                    reference::ConstantValue::I32(value) => ops::ConstantValue::I32(value),
                };
                builder.constant(value, value_type(*ty))
            }
            reference::ReferenceNode::Binary { op, a, b } => builder.binary(
                match op {
                    reference::BinaryOp::Add => ops::BinaryOp::Add,
                    reference::BinaryOp::Sub => ops::BinaryOp::Sub,
                    reference::BinaryOp::Mul => ops::BinaryOp::Mul,
                    reference::BinaryOp::Div => ops::BinaryOp::Div,
                },
                get(*a),
                get(*b),
            ),
            reference::ReferenceNode::Bit { op, a, b } => builder.bit(
                match op {
                    reference::BitOp::And => ops::BitOp::And,
                    reference::BitOp::Or => ops::BitOp::Or,
                    reference::BitOp::Shl => ops::BitOp::Shl,
                    reference::BitOp::Shr => ops::BitOp::Shr,
                },
                get(*a),
                get(*b),
            ),
            reference::ReferenceNode::Compare { op, a, b } => builder.cmp(
                match op {
                    reference::CmpOp::Eq => ops::CmpOp::Eq,
                    reference::CmpOp::Ne => ops::CmpOp::Ne,
                    reference::CmpOp::Lt => ops::CmpOp::Lt,
                    reference::CmpOp::Le => ops::CmpOp::Le,
                    reference::CmpOp::Gt => ops::CmpOp::Gt,
                    reference::CmpOp::Ge => ops::CmpOp::Ge,
                },
                get(*a),
                get(*b),
            ),
            reference::ReferenceNode::And { a, b } => {
                builder.logic(ops::LogicOp::And, get(*a), get(*b))
            }
            reference::ReferenceNode::Not { value } => builder.not(get(*value)),
            reference::ReferenceNode::Select { condition, yes, no } => {
                builder.select(get(*condition), get(*yes), get(*no))
            }
            reference::ReferenceNode::Unary { op, value } => builder.unary(
                match op {
                    reference::UnaryOp::Neg => ops::UnaryOp::Neg,
                    reference::UnaryOp::Abs => ops::UnaryOp::Abs,
                },
                get(*value),
            ),
            reference::ReferenceNode::Cast { value, to } => {
                builder.cast(get(*value), value_type(*to))
            }
            reference::ReferenceNode::Bitcast { value, to } => {
                builder.bitcast(get(*value), value_type(*to))
            }
        };
        values.push(value);
    }
    values[recipe.output().ordinal()]
}
