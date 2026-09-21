//! Canonical language definition of unary reference math.
//!
//! The recipes are derived from the f32 Sun/FreeBSD algorithms carried by
//! libm 0.2.16, with a pure-f32/u32 Payne-Hanek reducer for the full finite
//! f32 trigonometric domain. A recipe is an ordered, closed graph of primitive
//! operations. The language interpreter evaluates this graph and the compiler
//! instantiates the same graph into kernel IR; neither owns a second formula.
//! `Exact` therefore means zero deviation from this versioned recipe, not a
//! claim that every transcendental is the correctly-rounded real function.
//!
//! The exp/log and primary sin/cos polynomials originate in FreeBSD msun:
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunPro, a Sun Microsystems, Inc. business. Permission to use,
//! copy, modify, and distribute this software is freely granted, provided
//! that this notice is preserved.

use crate::intrinsics::MathOp;
use crate::types::DType;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::sync::OnceLock;

pub const VERSION: &str = "seismic-reference-math-f32-v1";

pub fn digest() -> [u8; 32] {
    static DIGEST: OnceLock<[u8; 32]> = OnceLock::new();
    *DIGEST.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(VERSION.as_bytes());
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for op in [
                ReferenceMathOp::Exp,
                ReferenceMathOp::Rsqrt,
                ReferenceMathOp::Sqrt,
                ReferenceMathOp::Log,
                ReferenceMathOp::Sin,
                ReferenceMathOp::Cos,
                ReferenceMathOp::Abs,
            ] {
                encode_recipe(&mut hash, &recipe(op, dtype));
            }
        }
        hash.finalize().into()
    })
}

fn encode_recipe(hash: &mut Sha256, recipe: &ReferenceRecipe) {
    fn dtype_tag(dtype: DType) -> u8 {
        match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            DType::I32 => 3,
            DType::U32 => 4,
            DType::Bool => 5,
        }
    }
    fn ty(hash: &mut Sha256, ty: ValueType) {
        match ty {
            ValueType::Scalar(dtype) => hash.update([0, dtype_tag(dtype)]),
            ValueType::Bool => hash.update([1, 0]),
        }
    }
    fn value(hash: &mut Sha256, value: ReferenceValue) {
        hash.update(value.ordinal.to_le_bytes());
        ty(hash, value.ty);
    }
    fn binary_tag(op: BinaryOp) -> u8 {
        match op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
        }
    }
    fn bit_tag(op: BitOp) -> u8 {
        match op {
            BitOp::And => 0,
            BitOp::Or => 1,
            BitOp::Shl => 2,
            BitOp::Shr => 3,
        }
    }
    fn cmp_tag(op: CmpOp) -> u8 {
        match op {
            CmpOp::Eq => 0,
            CmpOp::Ne => 1,
            CmpOp::Lt => 2,
            CmpOp::Le => 3,
            CmpOp::Gt => 4,
            CmpOp::Ge => 5,
        }
    }
    fn unary_tag(op: UnaryOp) -> u8 {
        match op {
            UnaryOp::Neg => 0,
            UnaryOp::Abs => 1,
        }
    }

    hash.update((recipe.nodes.len() as u64).to_le_bytes());
    for node in recipe.nodes.iter() {
        match *node {
            ReferenceNode::Input { dtype } => hash.update([0, dtype_tag(dtype)]),
            ReferenceNode::Constant {
                value: constant,
                ty: value_ty,
            } => {
                hash.update([1]);
                ty(hash, value_ty);
                match constant {
                    ConstantValue::F32(bits) => {
                        hash.update([0]);
                        hash.update(bits.to_le_bytes());
                    }
                    ConstantValue::U32(value) => {
                        hash.update([1]);
                        hash.update(value.to_le_bytes());
                    }
                    ConstantValue::I32(value) => {
                        hash.update([2]);
                        hash.update(value.to_le_bytes());
                    }
                }
            }
            ReferenceNode::Binary { op, a, b } => {
                hash.update([2, binary_tag(op)]);
                value(hash, a);
                value(hash, b);
            }
            ReferenceNode::Bit { op, a, b } => {
                hash.update([3, bit_tag(op)]);
                value(hash, a);
                value(hash, b);
            }
            ReferenceNode::Compare { op, a, b } => {
                hash.update([4, cmp_tag(op)]);
                value(hash, a);
                value(hash, b);
            }
            ReferenceNode::And { a, b } => {
                hash.update([5]);
                value(hash, a);
                value(hash, b);
            }
            ReferenceNode::Not { value: operand } => {
                hash.update([6]);
                value(hash, operand);
            }
            ReferenceNode::Select { condition, yes, no } => {
                hash.update([7]);
                value(hash, condition);
                value(hash, yes);
                value(hash, no);
            }
            ReferenceNode::Unary { op, value: operand } => {
                hash.update([8, unary_tag(op)]);
                value(hash, operand);
            }
            ReferenceNode::Cast { value: operand, to } => {
                hash.update([9]);
                value(hash, operand);
                ty(hash, to);
            }
            ReferenceNode::Bitcast { value: operand, to } => {
                hash.update([10]);
                value(hash, operand);
                ty(hash, to);
            }
        }
    }
    value(hash, recipe.output);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceMathOp {
    Exp,
    Rsqrt,
    Sqrt,
    Log,
    Sin,
    Cos,
    Abs,
}

impl TryFrom<MathOp> for ReferenceMathOp {
    type Error = ();
    fn try_from(value: MathOp) -> Result<Self, Self::Error> {
        Ok(match value {
            MathOp::Exp => Self::Exp,
            MathOp::Rsqrt => Self::Rsqrt,
            MathOp::Sqrt => Self::Sqrt,
            MathOp::Log => Self::Log,
            MathOp::Sin => Self::Sin,
            MathOp::Cos => Self::Cos,
            MathOp::Abs => Self::Abs,
            // `exp_fast` is explicitly approximate. Fma/Min/Max are direct
            // multi-operand primitives and never enter this unary recipe.
            MathOp::ExpFast | MathOp::Fma | MathOp::Max | MathOp::Min => return Err(()),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Scalar(DType),
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceValue {
    ordinal: u32,
    ty: ValueType,
}
impl ReferenceValue {
    pub fn ordinal(self) -> usize {
        self.ordinal as usize
    }
    pub fn ty(self) -> ValueType {
        self.ty
    }
}
type V = ReferenceValue;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstantValue {
    F32(u32),
    U32(u32),
    I32(i32),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitOp {
    And,
    Or,
    Shl,
    Shr,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReferenceNode {
    Input {
        dtype: DType,
    },
    Constant {
        value: ConstantValue,
        ty: ValueType,
    },
    Binary {
        op: BinaryOp,
        a: ReferenceValue,
        b: ReferenceValue,
    },
    Bit {
        op: BitOp,
        a: ReferenceValue,
        b: ReferenceValue,
    },
    Compare {
        op: CmpOp,
        a: ReferenceValue,
        b: ReferenceValue,
    },
    And {
        a: ReferenceValue,
        b: ReferenceValue,
    },
    Not {
        value: ReferenceValue,
    },
    Select {
        condition: ReferenceValue,
        yes: ReferenceValue,
        no: ReferenceValue,
    },
    Unary {
        op: UnaryOp,
        value: ReferenceValue,
    },
    Cast {
        value: ReferenceValue,
        to: ValueType,
    },
    Bitcast {
        value: ReferenceValue,
        to: ValueType,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRecipe {
    nodes: Box<[ReferenceNode]>,
    output: V,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReferenceScalar {
    F16(u16),
    BF16(u16),
    F32(u32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl ReferenceScalar {
    pub fn dtype(self) -> DType {
        match self {
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
            Self::Bool(_) => DType::Bool,
        }
    }
}

/// Evaluates the same ordered primitive graph consumed by compiler lowering.
/// Every floating primitive is a distinct f32 expression and therefore
/// rounds before its result is stored in the next recipe slot.
pub fn evaluate(recipe: &ReferenceRecipe, input: ReferenceScalar) -> ReferenceScalar {
    let _strict_float = StrictFloatEnvironment::enter();
    let mut values: Vec<ReferenceScalar> = Vec::with_capacity(recipe.nodes.len());
    for node in recipe.nodes.iter() {
        let get = |value: ReferenceValue| values[value.ordinal()];
        let value = match *node {
            ReferenceNode::Input { dtype } => {
                assert_eq!(input.dtype(), dtype, "reference recipe input dtype differs");
                input
            }
            ReferenceNode::Constant { value, .. } => match value {
                ConstantValue::F32(bits) => ReferenceScalar::F32(bits),
                ConstantValue::U32(value) => ReferenceScalar::U32(value),
                ConstantValue::I32(value) => ReferenceScalar::I32(value),
            },
            ReferenceNode::Binary { op, a, b } => eval_binary(op, get(a), get(b)),
            ReferenceNode::Bit { op, a, b } => eval_bit(op, get(a), get(b)),
            ReferenceNode::Compare { op, a, b } => eval_compare(op, get(a), get(b)),
            ReferenceNode::And { a, b } => match (get(a), get(b)) {
                (ReferenceScalar::Bool(a), ReferenceScalar::Bool(b)) => {
                    ReferenceScalar::Bool(a && b)
                }
                _ => unreachable!("closed reference recipe logic types differ"),
            },
            ReferenceNode::Not { value } => match get(value) {
                ReferenceScalar::Bool(value) => ReferenceScalar::Bool(!value),
                _ => unreachable!("closed reference recipe not operand is not bool"),
            },
            ReferenceNode::Select { condition, yes, no } => match get(condition) {
                ReferenceScalar::Bool(true) => get(yes),
                ReferenceScalar::Bool(false) => get(no),
                _ => unreachable!("closed reference recipe select condition is not bool"),
            },
            ReferenceNode::Unary { op, value } => eval_unary(op, get(value)),
            ReferenceNode::Cast { value, to } => eval_cast(get(value), to),
            ReferenceNode::Bitcast { value, to } => eval_bitcast(get(value), to),
        };
        values.push(value);
    }
    values[recipe.output.ordinal()]
}

struct StrictFloatEnvironment {
    saved: u64,
}
impl StrictFloatEnvironment {
    #[cfg(target_arch = "x86_64")]
    fn enter() -> Self {
        let saved = unsafe { core::arch::x86_64::_mm_getcsr() };
        let strict = saved & !((1 << 6) | (3 << 13) | (1 << 15));
        unsafe { core::arch::x86_64::_mm_setcsr(strict) };
        Self {
            saved: u64::from(saved),
        }
    }
    #[cfg(target_arch = "aarch64")]
    fn enter() -> Self {
        let saved: u64;
        unsafe { core::arch::asm!("mrs {saved}, fpcr", saved = out(reg) saved) };
        let strict = saved & !((1 << 19) | (3 << 22) | (1 << 24) | (1 << 25));
        unsafe { core::arch::asm!("msr fpcr, {strict}", strict = in(reg) strict) };
        Self { saved }
    }
    #[cfg(target_arch = "wasm32")]
    fn enter() -> Self {
        // WebAssembly scalar floating operations have fixed IEEE rounding
        // and gradual-underflow semantics; there is no mutable FP mode.
        Self { saved: 0 }
    }
}
impl Drop for StrictFloatEnvironment {
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_setcsr(self.saved as u32)
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("msr fpcr, {saved}", saved = in(reg) self.saved)
        };
        #[cfg(target_arch = "wasm32")]
        let _ = self.saved;
    }
}

fn eval_binary(op: BinaryOp, a: ReferenceScalar, b: ReferenceScalar) -> ReferenceScalar {
    match (a, b) {
        (ReferenceScalar::F32(a), ReferenceScalar::F32(b)) => {
            let (a, b) = (f32::from_bits(a), f32::from_bits(b));
            ReferenceScalar::F32(
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                }
                .to_bits(),
            )
        }
        (ReferenceScalar::U32(a), ReferenceScalar::U32(b)) => ReferenceScalar::U32(match op {
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Sub => a.wrapping_sub(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            BinaryOp::Div => unreachable!("reference recipe integer division is unconstructible"),
        }),
        (ReferenceScalar::I32(a), ReferenceScalar::I32(b)) => ReferenceScalar::I32(match op {
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Sub => a.wrapping_sub(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            BinaryOp::Div => unreachable!("reference recipe integer division is unconstructible"),
        }),
        _ => unreachable!("closed reference recipe binary types differ"),
    }
}
fn eval_bit(op: BitOp, a: ReferenceScalar, b: ReferenceScalar) -> ReferenceScalar {
    let apply = |a: u32, b: u32| match op {
        BitOp::And => a & b,
        BitOp::Or => a | b,
        BitOp::Shl => a
            .checked_shl(b)
            .expect("reference recipe shift is out of range"),
        BitOp::Shr => a
            .checked_shr(b)
            .expect("reference recipe shift is out of range"),
    };
    match (a, b) {
        (ReferenceScalar::U32(a), ReferenceScalar::U32(b)) => ReferenceScalar::U32(apply(a, b)),
        (ReferenceScalar::I32(a), ReferenceScalar::I32(b)) => {
            ReferenceScalar::I32(apply(a as u32, b as u32) as i32)
        }
        _ => unreachable!("closed reference recipe bit types differ"),
    }
}
fn eval_compare(op: CmpOp, a: ReferenceScalar, b: ReferenceScalar) -> ReferenceScalar {
    let result = match (a, b) {
        (ReferenceScalar::F32(a), ReferenceScalar::F32(b)) => {
            compare(op, f32::from_bits(a), f32::from_bits(b))
        }
        (ReferenceScalar::U32(a), ReferenceScalar::U32(b)) => compare(op, a, b),
        (ReferenceScalar::I32(a), ReferenceScalar::I32(b)) => compare(op, a, b),
        _ => unreachable!("closed reference recipe comparison types differ"),
    };
    ReferenceScalar::Bool(result)
}
fn compare<T: PartialEq + PartialOrd>(op: CmpOp, a: T, b: T) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}
fn eval_unary(op: UnaryOp, value: ReferenceScalar) -> ReferenceScalar {
    match value {
        ReferenceScalar::F32(bits) => {
            let value = f32::from_bits(bits);
            ReferenceScalar::F32(
                match op {
                    UnaryOp::Neg => -value,
                    UnaryOp::Abs => value.abs(),
                }
                .to_bits(),
            )
        }
        ReferenceScalar::I32(value) => ReferenceScalar::I32(match op {
            UnaryOp::Neg => value.wrapping_neg(),
            UnaryOp::Abs => value.wrapping_abs(),
        }),
        _ => unreachable!("closed reference recipe unary type"),
    }
}
fn eval_cast(value: ReferenceScalar, to: ValueType) -> ReferenceScalar {
    use crate::registry::{bf16_round, f16_bits, f16_to_f32};
    match (value, to) {
        (value, ValueType::Scalar(dtype)) if value.dtype() == dtype => value,
        (ReferenceScalar::F16(bits), ValueType::Scalar(DType::F32)) => {
            ReferenceScalar::F32(f16_to_f32(bits).to_bits())
        }
        (ReferenceScalar::BF16(bits), ValueType::Scalar(DType::F32)) => {
            ReferenceScalar::F32(u32::from(bits) << 16)
        }
        (ReferenceScalar::F32(bits), ValueType::Scalar(DType::F16)) => {
            ReferenceScalar::F16(f16_bits(f32::from_bits(bits)))
        }
        (ReferenceScalar::F32(bits), ValueType::Scalar(DType::BF16)) => {
            ReferenceScalar::BF16((bf16_round(f32::from_bits(bits)).to_bits() >> 16) as u16)
        }
        (ReferenceScalar::F32(bits), ValueType::Scalar(DType::I32)) => {
            ReferenceScalar::I32(f32::from_bits(bits) as i32)
        }
        (ReferenceScalar::F32(bits), ValueType::Scalar(DType::U32)) => {
            ReferenceScalar::U32(f32::from_bits(bits) as u32)
        }
        (ReferenceScalar::I32(value), ValueType::Scalar(DType::F32)) => {
            ReferenceScalar::F32((value as f32).to_bits())
        }
        (ReferenceScalar::U32(value), ValueType::Scalar(DType::F32)) => {
            ReferenceScalar::F32((value as f32).to_bits())
        }
        _ => unreachable!("closed reference recipe cast pair"),
    }
}
fn eval_bitcast(value: ReferenceScalar, to: ValueType) -> ReferenceScalar {
    match (value, to) {
        (ReferenceScalar::F32(bits), ValueType::Scalar(DType::U32)) => ReferenceScalar::U32(bits),
        (ReferenceScalar::U32(bits), ValueType::Scalar(DType::F32)) => ReferenceScalar::F32(bits),
        _ => unreachable!("closed reference recipe bitcast pair"),
    }
}
impl ReferenceRecipe {
    pub fn nodes(&self) -> &[ReferenceNode] {
        &self.nodes
    }
    pub fn output(&self) -> ReferenceValue {
        self.output
    }
}

const F32_TY: ValueType = ValueType::Scalar(DType::F32);
const U32_TY: ValueType = ValueType::Scalar(DType::U32);
const I32_TY: ValueType = ValueType::Scalar(DType::I32);

struct RecipeBuilder {
    nodes: Vec<ReferenceNode>,
}
impl RecipeBuilder {
    fn push(&mut self, ty: ValueType, node: ReferenceNode) -> V {
        let ordinal = u32::try_from(self.nodes.len()).expect("reference recipe exceeds u32 nodes");
        self.nodes.push(node);
        V { ordinal, ty }
    }
}
struct Recipe {
    builder: RefCell<RecipeBuilder>,
}

impl Recipe {
    fn new(dtype: DType) -> (Self, V) {
        let mut builder = RecipeBuilder { nodes: Vec::new() };
        let input = builder.push(ValueType::Scalar(dtype), ReferenceNode::Input { dtype });
        (
            Self {
                builder: RefCell::new(builder),
            },
            input,
        )
    }
    fn push(&self, ty: ValueType, node: ReferenceNode) -> V {
        self.builder.borrow_mut().push(ty, node)
    }
    fn f(&self, bits: u32) -> V {
        self.push(
            F32_TY,
            ReferenceNode::Constant {
                value: ConstantValue::F32(bits),
                ty: F32_TY,
            },
        )
    }
    fn u(&self, value: u32) -> V {
        self.push(
            U32_TY,
            ReferenceNode::Constant {
                value: ConstantValue::U32(value),
                ty: U32_TY,
            },
        )
    }
    fn i(&self, value: i32) -> V {
        self.push(
            I32_TY,
            ReferenceNode::Constant {
                value: ConstantValue::I32(value),
                ty: I32_TY,
            },
        )
    }
    fn bin(&self, op: BinaryOp, a: V, b: V) -> V {
        assert_eq!(a.ty, b.ty);
        if op == BinaryOp::Div {
            assert_eq!(
                a.ty, F32_TY,
                "reference recipe integer division is forbidden"
            );
        }
        self.push(a.ty, ReferenceNode::Binary { op, a, b })
    }
    fn bit(&self, op: BitOp, a: V, b: V) -> V {
        assert_eq!(a.ty, b.ty);
        assert!(matches!(a.ty, ValueType::Scalar(DType::U32 | DType::I32)));
        self.push(a.ty, ReferenceNode::Bit { op, a, b })
    }
    fn cmp(&self, op: CmpOp, a: V, b: V) -> V {
        assert_eq!(a.ty, b.ty);
        self.push(ValueType::Bool, ReferenceNode::Compare { op, a, b })
    }
    fn and(&self, a: V, b: V) -> V {
        assert_eq!(a.ty, ValueType::Bool);
        assert_eq!(b.ty, ValueType::Bool);
        self.push(ValueType::Bool, ReferenceNode::And { a, b })
    }
    fn select(&self, condition: V, yes: V, no: V) -> V {
        assert_eq!(condition.ty, ValueType::Bool);
        assert_eq!(yes.ty, no.ty);
        self.push(yes.ty, ReferenceNode::Select { condition, yes, no })
    }
    fn neg(&self, value: V) -> V {
        self.push(
            value.ty,
            ReferenceNode::Unary {
                op: UnaryOp::Neg,
                value,
            },
        )
    }
    fn fbits(&self, value: V) -> V {
        assert_eq!(value.ty, F32_TY);
        self.push(U32_TY, ReferenceNode::Bitcast { value, to: U32_TY })
    }
    fn from_bits(&self, value: V) -> V {
        assert_eq!(value.ty, U32_TY);
        self.push(F32_TY, ReferenceNode::Bitcast { value, to: F32_TY })
    }
    fn cast(&self, value: V, to: ValueType) -> V {
        self.push(to, ReferenceNode::Cast { value, to })
    }
    fn unary(&self, op: UnaryOp, value: V) -> V {
        self.push(value.ty, ReferenceNode::Unary { op, value })
    }
    fn not(&self, value: V) -> V {
        assert_eq!(value.ty, ValueType::Bool);
        self.push(ValueType::Bool, ReferenceNode::Not { value })
    }
}

pub fn recipe(op: ReferenceMathOp, dtype: DType) -> ReferenceRecipe {
    assert!(dtype.is_float(), "reference math requires a floating dtype");
    let (b, input) = Recipe::new(dtype);
    let b = &b;
    let original = input.ty;
    let x = match original {
        ValueType::Scalar(DType::F32) => input,
        ValueType::Scalar(DType::F16 | DType::BF16) => b.cast(input, F32_TY),
        _ => unreachable!("dtype float check and closed value types agree"),
    };
    let result = match op {
        ReferenceMathOp::Exp => exp(b, x),
        ReferenceMathOp::Log => log(b, x),
        ReferenceMathOp::Sin => trig(b, x, false),
        ReferenceMathOp::Cos => trig(b, x, true),
        ReferenceMathOp::Sqrt => sqrt(b, x),
        ReferenceMathOp::Rsqrt => {
            let root = sqrt(b, x);
            let one = b.f(0x3f80_0000);
            b.bin(BinaryOp::Div, one, root)
        }
        ReferenceMathOp::Abs => b.unary(UnaryOp::Abs, x),
    };
    let output = if original == F32_TY {
        result
    } else {
        b.cast(result, original)
    };
    let nodes = std::mem::take(&mut b.builder.borrow_mut().nodes).into_boxed_slice();
    ReferenceRecipe { nodes, output }
}

// Sun/FreeBSD expf, expressed with explicit f32 operations and a bit-built
// scale so the sequence is identical on every backend.
fn exp(b: &Recipe, x: V) -> V {
    let bits = b.fbits(x);
    let abs_bits = b.bit(BitOp::And, bits, b.u(0x7fff_ffff));
    let negative = b.cmp(CmpOp::Ne, b.bit(BitOp::And, bits, b.u(0x8000_0000)), b.u(0));
    let nan = b.cmp(CmpOp::Gt, abs_bits, b.u(0x7f80_0000));
    let overflow = b.and(
        b.not(negative),
        b.cmp(CmpOp::Ge, abs_bits, b.u(0x42b1_7218)),
    );
    let hard_underflow = b.and(negative, b.cmp(CmpOp::Ge, abs_bits, b.u(0x42cf_f1b5)));

    // Keep the float-to-int conversion inside its defined interval even
    // though the final special-case select returns the original infinity/NaN.
    let safe_x = b.select(
        b.cmp(CmpOp::Gt, x, b.f(0x42b0_0000)),
        b.f(0x42b0_0000),
        b.select(b.cmp(CmpOp::Lt, x, b.f(0xc2d0_0000)), b.f(0xc2d0_0000), x),
    );
    let large_reduction = b.cmp(CmpOp::Gt, abs_bits, b.u(0x3f85_1592));
    let needs_reduction = b.cmp(CmpOp::Gt, abs_bits, b.u(0x3eb1_7218));
    let half = b.select(negative, b.f(0xbf00_0000), b.f(0x3f00_0000));
    let rounded = b.bin(
        BinaryOp::Add,
        b.bin(BinaryOp::Mul, b.f(0x3fb8_aa3b), safe_x),
        half,
    );
    let k_large = b.cast(rounded, I32_TY);
    let k_small = b.select(negative, b.i(-1), b.i(1));
    let k = b.select(
        needs_reduction,
        b.select(large_reduction, k_large, k_small),
        b.i(0),
    );
    let kf = b.cast(k, F32_TY);
    let hi = b.bin(
        BinaryOp::Sub,
        safe_x,
        b.bin(BinaryOp::Mul, kf, b.f(0x3f31_7200)),
    );
    let lo = b.bin(BinaryOp::Mul, kf, b.f(0x35bf_be8e));
    let reduced = b.bin(BinaryOp::Sub, hi, lo);
    let xx = b.bin(BinaryOp::Mul, reduced, reduced);
    let poly = b.bin(
        BinaryOp::Add,
        b.f(0x3e2a_aa8f),
        b.bin(BinaryOp::Mul, xx, b.f(0xbb35_5215)),
    );
    let c = b.bin(BinaryOp::Sub, reduced, b.bin(BinaryOp::Mul, xx, poly));
    let correction = b.bin(
        BinaryOp::Div,
        b.bin(BinaryOp::Mul, reduced, c),
        b.bin(BinaryOp::Sub, b.f(0x4000_0000), c),
    );
    let y = b.bin(
        BinaryOp::Add,
        b.f(0x3f80_0000),
        b.bin(BinaryOp::Add, b.bin(BinaryOp::Sub, correction, lo), hi),
    );
    let scaled = scalbn(b, y, k);
    let tiny = b.cmp(CmpOp::Le, abs_bits, b.u(0x3900_0000));
    let ordinary = b.select(tiny, b.bin(BinaryOp::Add, b.f(0x3f80_0000), x), scaled);
    let infinity = b.f(0x7f80_0000);
    let ordinary = b.select(overflow, infinity, ordinary);
    let ordinary = b.select(hard_underflow, b.f(0), ordinary);
    b.select(nan, x, ordinary)
}

fn scalbn(b: &Recipe, y: V, k: V) -> V {
    let normal = b.cmp(CmpOp::Ge, k, b.i(-126));
    let normal_exp = b.select(normal, b.bin(BinaryOp::Add, k, b.i(127)), b.i(0));
    let normal_bits = b.bit(BitOp::Shl, b.cast(normal_exp, U32_TY), b.u(23));
    let sub_shift = b.bin(BinaryOp::Add, k, b.i(149));
    let sub_shift = b.select(
        b.cmp(CmpOp::Lt, sub_shift, b.i(0)),
        b.i(0),
        b.select(b.cmp(CmpOp::Gt, sub_shift, b.i(31)), b.i(31), sub_shift),
    );
    let sub_bits = b.bit(BitOp::Shl, b.u(1), b.cast(sub_shift, U32_TY));
    let scale = b.from_bits(b.select(normal, normal_bits, sub_bits));
    b.bin(BinaryOp::Mul, y, scale)
}

// Sun/FreeBSD logf with exact bit normalization of subnormals.
fn log(b: &Recipe, x: V) -> V {
    let original_bits = b.fbits(x);
    let abs_bits = b.bit(BitOp::And, original_bits, b.u(0x7fff_ffff));
    let sign = b.cmp(
        CmpOp::Ne,
        b.bit(BitOp::And, original_bits, b.u(0x8000_0000)),
        b.u(0),
    );
    let zero = b.cmp(CmpOp::Eq, abs_bits, b.u(0));
    let special = b.cmp(CmpOp::Ge, abs_bits, b.u(0x7f80_0000));
    let subnormal = b.and(
        b.cmp(CmpOp::Ne, abs_bits, b.u(0)),
        b.cmp(CmpOp::Lt, abs_bits, b.u(0x0080_0000)),
    );
    let scaled = b.bin(BinaryOp::Mul, x, b.f(0x4c00_0000));
    let work = b.select(subnormal, scaled, x);
    let mut ix = b.fbits(work);
    ix = b.bin(BinaryOp::Add, ix, b.u(0x004a_fb0d));
    let exponent = b.bit(BitOp::Shr, ix, b.u(23));
    let mut k = b.bin(BinaryOp::Sub, b.cast(exponent, I32_TY), b.i(127));
    k = b.select(subnormal, b.bin(BinaryOp::Sub, k, b.i(25)), k);
    ix = b.bin(
        BinaryOp::Add,
        b.bit(BitOp::And, ix, b.u(0x007f_ffff)),
        b.u(0x3f35_04f3),
    );
    let normalized = b.from_bits(ix);
    let one = b.f(0x3f80_0000);
    let fv = b.bin(BinaryOp::Sub, normalized, one);
    let s = b.bin(
        BinaryOp::Div,
        fv,
        b.bin(BinaryOp::Add, b.f(0x4000_0000), fv),
    );
    let z = b.bin(BinaryOp::Mul, s, s);
    let w = b.bin(BinaryOp::Mul, z, z);
    let t1 = b.bin(
        BinaryOp::Mul,
        w,
        b.bin(
            BinaryOp::Add,
            b.f(0x3ecc_ce13),
            b.bin(BinaryOp::Mul, w, b.f(0x3e78_9e26)),
        ),
    );
    let t2 = b.bin(
        BinaryOp::Mul,
        z,
        b.bin(
            BinaryOp::Add,
            b.f(0x3f2a_aaaa),
            b.bin(BinaryOp::Mul, w, b.f(0x3e91_e9ee)),
        ),
    );
    let r = b.bin(BinaryOp::Add, t2, t1);
    let hfsq = b.bin(
        BinaryOp::Mul,
        b.f(0x3f00_0000),
        b.bin(BinaryOp::Mul, fv, fv),
    );
    let dk = b.cast(k, F32_TY);
    let result = b.bin(
        BinaryOp::Add,
        b.bin(
            BinaryOp::Sub,
            b.bin(
                BinaryOp::Add,
                b.bin(BinaryOp::Mul, s, b.bin(BinaryOp::Add, hfsq, r)),
                b.bin(BinaryOp::Mul, dk, b.f(0x3717_f7d1)),
            ),
            hfsq,
        ),
        b.bin(
            BinaryOp::Add,
            fv,
            b.bin(BinaryOp::Mul, dk, b.f(0x3f31_7180)),
        ),
    );
    let nan = b.f(0x7fc0_0000);
    let negative_or_nan = b.select(sign, nan, x);
    let result = b.select(special, negative_or_nan, result);
    let result = b.select(sign, nan, result);
    b.select(zero, b.f(0xff80_0000), result)
}

// Deterministic f32 Newton sequence with bit normalization for subnormals.
fn sqrt(b: &Recipe, x: V) -> V {
    let bits = b.fbits(x);
    let abs_bits = b.bit(BitOp::And, bits, b.u(0x7fff_ffff));
    let sign = b.cmp(CmpOp::Ne, b.bit(BitOp::And, bits, b.u(0x8000_0000)), b.u(0));
    let zero = b.cmp(CmpOp::Eq, abs_bits, b.u(0));
    let nan = b.cmp(CmpOp::Gt, abs_bits, b.u(0x7f80_0000));
    let infinity = b.cmp(CmpOp::Eq, abs_bits, b.u(0x7f80_0000));
    let subnormal = b.and(
        b.cmp(CmpOp::Ne, abs_bits, b.u(0)),
        b.cmp(CmpOp::Lt, abs_bits, b.u(0x0080_0000)),
    );
    let work = b.select(subnormal, b.bin(BinaryOp::Mul, x, b.f(0x4b80_0000)), x);
    let guess_bits = b.bin(
        BinaryOp::Add,
        b.bit(BitOp::Shr, b.fbits(work), b.u(1)),
        b.u(0x1fc0_0000),
    );
    let mut guess = b.from_bits(guess_bits);
    for _ in 0..7 {
        guess = b.bin(
            BinaryOp::Mul,
            b.f(0x3f00_0000),
            b.bin(BinaryOp::Add, guess, b.bin(BinaryOp::Div, work, guess)),
        );
    }
    guess = b.select(
        subnormal,
        b.bin(BinaryOp::Mul, guess, b.f(0x3980_0000)),
        guess,
    );
    let invalid = b.and(sign, b.not(zero));
    let result = b.select(invalid, b.f(0x7fc0_0000), guess);
    let result = b.select(infinity, x, result);
    let result = b.select(nan, x, result);
    b.select(zero, x, result)
}

// floor((2/pi) * 2^192), little-endian base-2^12 digits.
const TWO_OVER_PI_12: [u32; 16] = [
    65, 1081, 2364, 2393, 2914, 3085, 1245, 3923, 2001, 629, 2556, 338, 3652, 1764, 2435, 2607,
];

fn trig(b: &Recipe, x: V, cosine: bool) -> V {
    let bits = b.fbits(x);
    let abs_bits = b.bit(BitOp::And, bits, b.u(0x7fff_ffff));
    let sign = b.cmp(CmpOp::Ne, b.bit(BitOp::And, bits, b.u(0x8000_0000)), b.u(0));
    let finite = b.cmp(CmpOp::Lt, abs_bits, b.u(0x7f80_0000));
    let small = b.cmp(CmpOp::Le, abs_bits, b.u(0x3f49_0fdb));

    let mantissa = b.bit(
        BitOp::Or,
        b.bit(BitOp::And, abs_bits, b.u(0x007f_ffff)),
        b.u(0x0080_0000),
    );
    let m0 = b.bit(BitOp::And, mantissa, b.u(0xfff));
    let m1 = b.bit(BitOp::And, b.bit(BitOp::Shr, mantissa, b.u(12)), b.u(0xfff));
    let mut digits = Vec::with_capacity(21);
    let mut carry = b.u(0);
    for digit in 0..=16 {
        let c0 = b.u(TWO_OVER_PI_12.get(digit).copied().unwrap_or(0));
        let c1 = b.u(digit
            .checked_sub(1)
            .and_then(|index| TWO_OVER_PI_12.get(index).copied())
            .unwrap_or(0));
        let sum = b.bin(
            BinaryOp::Add,
            b.bin(
                BinaryOp::Add,
                b.bin(BinaryOp::Mul, m0, c0),
                b.bin(BinaryOp::Mul, m1, c1),
            ),
            carry,
        );
        digits.push(b.bit(BitOp::And, sum, b.u(0xfff)));
        carry = b.bit(BitOp::Shr, sum, b.u(12));
    }
    digits.push(carry);
    while digits.len() < 21 {
        digits.push(b.u(0));
    }

    let exponent = b.bit(BitOp::And, b.bit(BitOp::Shr, abs_bits, b.u(23)), b.u(0xff));
    // Fraction window begins at S-25, where S=215-E=342-exponent.
    let fraction_start = clamp_window_start(b, b.bin(BinaryOp::Sub, b.u(317), exponent));
    let fraction = extract_window(b, &digits, fraction_start, 25);
    let rounds_up = b.cmp(CmpOp::Ge, fraction, b.u(1 << 24));
    let quotient_start = clamp_window_start(b, b.bin(BinaryOp::Sub, b.u(342), exponent));
    let quotient = extract_window(b, &digits, quotient_start, 2);
    let rounded = b.bin(BinaryOp::Add, quotient, b.select(rounds_up, b.u(1), b.u(0)));
    let quadrant = b.bit(BitOp::And, rounded, b.u(3));
    let signed_quadrant = b.select(
        sign,
        b.bit(BitOp::And, b.bin(BinaryOp::Sub, b.u(0), quadrant), b.u(3)),
        quadrant,
    );
    let fraction_f = b.bin(
        BinaryOp::Mul,
        b.cast(fraction, F32_TY),
        b.f(0x3300_0000), // 2^-25
    );
    let signed_fraction = b.select(
        rounds_up,
        b.bin(BinaryOp::Sub, fraction_f, b.f(0x3f80_0000)),
        fraction_f,
    );
    let signed_fraction = b.select(sign, b.neg(signed_fraction), signed_fraction);
    let reduced = b.bin(
        BinaryOp::Add,
        b.bin(BinaryOp::Mul, signed_fraction, b.f(0x3fc9_0000)),
        b.bin(BinaryOp::Mul, signed_fraction, b.f(0x39fd_aa22)),
    );
    let s = sin_kernel(b, reduced);
    let c = cos_kernel(b, reduced);
    let q0 = b.cmp(CmpOp::Eq, signed_quadrant, b.u(0));
    let q1 = b.cmp(CmpOp::Eq, signed_quadrant, b.u(1));
    let q2 = b.cmp(CmpOp::Eq, signed_quadrant, b.u(2));
    let general = if cosine {
        b.select(q0, c, b.select(q1, b.neg(s), b.select(q2, b.neg(c), s)))
    } else {
        b.select(q0, s, b.select(q1, c, b.select(q2, b.neg(s), b.neg(c))))
    };
    let direct = if cosine {
        cos_kernel(b, x)
    } else {
        sin_kernel(b, x)
    };
    let result = b.select(small, direct, general);
    b.select(finite, result, b.bin(BinaryOp::Sub, x, x))
}

fn clamp_window_start(b: &Recipe, start: V) -> V {
    // The selected three-digit window covers bit starts through 216. Inputs
    // below the direct-kernel threshold do not consume Payne-Hanek's result,
    // but every ordinary SSA operation must still have defined shift counts.
    b.select(b.cmp(CmpOp::Gt, start, b.u(216)), b.u(216), start)
}

fn extract_window(b: &Recipe, digits: &[V], start: V, width: u32) -> V {
    let mut selected = digits[0];
    let mut next = digits[1];
    let mut after = digits[2];
    let mut base = b.u(0);
    for index in 1..=18usize {
        let threshold = b.u((index as u32) * 12);
        let take = b.cmp(CmpOp::Ge, start, threshold);
        selected = b.select(take, digits[index], selected);
        next = b.select(take, digits[index + 1], next);
        after = b.select(take, digits[index + 2], after);
        base = b.select(take, threshold, base);
    }
    let offset = b.bin(BinaryOp::Sub, start, base);
    let low = b.bit(BitOp::Or, selected, b.bit(BitOp::Shl, next, b.u(12)));
    let low = b.bit(BitOp::Shr, low, offset);
    let high_shift = b.bin(BinaryOp::Sub, b.u(24), offset);
    let high = b.bit(BitOp::Shl, after, high_shift);
    b.bit(
        BitOp::And,
        b.bit(BitOp::Or, low, high),
        b.u((1u32 << width) - 1),
    )
}

fn sin_kernel(b: &Recipe, x: V) -> V {
    let z = b.bin(BinaryOp::Mul, x, x);
    let w = b.bin(BinaryOp::Mul, z, z);
    let r = b.bin(
        BinaryOp::Add,
        b.f(0xb950_07cf),
        b.bin(BinaryOp::Mul, z, b.f(0x3636_6c3c)),
    );
    let sx = b.bin(BinaryOp::Mul, z, x);
    b.bin(
        BinaryOp::Add,
        b.bin(
            BinaryOp::Add,
            x,
            b.bin(
                BinaryOp::Mul,
                sx,
                b.bin(
                    BinaryOp::Add,
                    b.f(0xbe2a_aaab),
                    b.bin(BinaryOp::Mul, z, b.f(0x3c08_8884)),
                ),
            ),
        ),
        b.bin(BinaryOp::Mul, b.bin(BinaryOp::Mul, sx, w), r),
    )
}

fn cos_kernel(b: &Recipe, x: V) -> V {
    let z = b.bin(BinaryOp::Mul, x, x);
    let w = b.bin(BinaryOp::Mul, z, z);
    let r = b.bin(
        BinaryOp::Add,
        b.f(0xbab6_043f),
        b.bin(BinaryOp::Mul, z, b.f(0x37cc_9a17)),
    );
    b.bin(
        BinaryOp::Add,
        b.bin(
            BinaryOp::Add,
            b.f(0x3f80_0000),
            b.bin(BinaryOp::Mul, z, b.f(0xbf00_0000)),
        ),
        b.bin(
            BinaryOp::Add,
            b.bin(BinaryOp::Mul, w, b.f(0x3d2a_aa9f)),
            b.bin(BinaryOp::Mul, b.bin(BinaryOp::Mul, w, z), r),
        ),
    )
}
