//! Language-owned scalar semantics as total, typed-bit recipes.
//!
//! The terminal vocabulary contains only unsigned word arithmetic, comparisons,
//! Boolean selection and bit-preserving transport. Both interpretation and kernel
//! construction consume this same graph. Transcendentals preserve the ordered
//! Sun/FreeBSD f32 recipes; their scalar steps expand through the same owner.
//!
//! The exp/log and primary sin/cos polynomials originate in FreeBSD msun:
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunPro, a Sun Microsystems, Inc. business. Permission to use,
//! copy, modify, and distribute this software is freely granted, provided
//! that this notice is preserved.

mod primitive;

use crate::intrinsics::MathOp;
use crate::syntax::ast;
use crate::types::DType;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub const VERSION: &str = "seismic-scalar-reference-bits-v2";

/// Operation identities describe source semantics, never physical approximations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarOp {
    Binary(ast::BinaryOp),
    Unary(ast::UnaryOp),
    Math(MathOp),
    Cast(DType),
}

/// The scalar meaning of an ordinary checked primitive. Structural primitives
/// have no scalar recipe; their owning source constructor handles them.
pub fn scalar_operation(primitive: &crate::intrinsics::PrimitiveId) -> Option<ScalarOp> {
    use crate::intrinsics::PrimitiveId;
    Some(match primitive {
        PrimitiveId::Unary(op) => ScalarOp::Unary(*op),
        PrimitiveId::Binary(op) => ScalarOp::Binary(*op),
        PrimitiveId::Math(op) => ScalarOp::Math(*op),
        PrimitiveId::Cast(dtype) => ScalarOp::Cast(*dtype),
        _ => return None,
    })
}

/// Payload bits are authoritative, including NaN payloads and signed zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    pub fn bits(self) -> u32 {
        match self {
            Self::F16(x) | Self::BF16(x) => u32::from(x),
            Self::F32(x) | Self::U32(x) => x,
            Self::I32(x) => x as u32,
            Self::Bool(x) => u32::from(x),
        }
    }
    pub fn from_bits(dtype: DType, bits: u32) -> Self {
        match dtype {
            DType::F16 => Self::F16(bits as u16),
            DType::BF16 => Self::BF16(bits as u16),
            DType::F32 => Self::F32(bits),
            DType::I32 => Self::I32(bits as i32),
            DType::U32 => Self::U32(bits),
            DType::Bool => Self::Bool(bits != 0),
        }
    }
    /// Diagnostic projection only. Semantic operations consume `bits` instead.
    pub fn to_f64(self) -> f64 {
        match self {
            Self::F16(x) => crate::registry::f16_to_f32(x) as f64,
            Self::BF16(x) => f32::from_bits(u32::from(x) << 16) as f64,
            Self::F32(x) => f32::from_bits(x) as f64,
            Self::I32(x) => f64::from(x),
            Self::U32(x) => f64::from(x),
            Self::Bool(x) => f64::from(u8::from(x)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceValue {
    ordinal: u32,
    ty: DType,
}
impl ReferenceValue {
    pub fn ordinal(self) -> usize {
        self.ordinal as usize
    }
    pub fn ty(self) -> DType {
        self.ty
    }
}
type V = ReferenceValue;

// These are recipe-construction operations. They do not survive as terminals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BitOp {
    And,
    Or,
    Shl,
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WordOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
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

/// There is deliberately no floating terminal, division terminal or source-op
/// callback: expanding a recipe cannot recursively request another scalar recipe.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceNode {
    Input { operand: u32, dtype: DType },
    Constant(ReferenceScalar),
    Word { op: WordOp, a: V, b: V },
    Compare { op: CmpOp, a: V, b: V },
    And { a: V, b: V },
    Not { value: V },
    Select { condition: V, yes: V, no: V },
    Bits { value: V },
    FromBits { value: V, dtype: DType },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarFailure {
    IntegerDivisionByZero,
    SignedDivisionOverflow,
    ShiftCount,
}
impl ScalarFailure {
    pub fn message(self) -> &'static str {
        match self {
            Self::IntegerDivisionByZero => "integer division by zero",
            Self::SignedDivisionOverflow => "signed integer division overflow",
            Self::ShiftCount => "integer shift count must be in 0..32",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceRecipe {
    nodes: Box<[ReferenceNode]>,
    output: V,
    failures: Box<[(V, ScalarFailure)]>,
}
impl ReferenceRecipe {
    pub fn nodes(&self) -> &[ReferenceNode] {
        &self.nodes
    }
    pub fn output(&self) -> V {
        self.output
    }
    /// Failure predicates are evaluated at the owning source operation, after
    /// total eager evaluation of its bit recipe and before publishing its result.
    pub fn failures(&self) -> &[(V, ScalarFailure)] {
        &self.failures
    }
}

fn terminal(node: &ReferenceNode, get: impl Fn(V) -> ReferenceScalar) -> ReferenceScalar {
    let word = |v| {
        let ReferenceScalar::U32(x) = get(v) else {
            panic!("non-word terminal")
        };
        x
    };
    let boolean = |v| {
        let ReferenceScalar::Bool(x) = get(v) else {
            panic!("non-Boolean terminal")
        };
        x
    };
    match *node {
        ReferenceNode::Input { .. } => unreachable!("input resolved by recipe invocation"),
        ReferenceNode::Constant(x) => x,
        ReferenceNode::Word { op, a, b } => {
            let (a, b) = (word(a), word(b));
            ReferenceScalar::U32(match op {
                WordOp::Add => a.wrapping_add(b),
                WordOp::Sub => a.wrapping_sub(b),
                WordOp::And => a & b,
                WordOp::Or => a | b,
                WordOp::Xor => a ^ b,
                WordOp::Shl => a.checked_shl(b).expect("constructed word shift"),
                WordOp::Shr => a.checked_shr(b).expect("constructed word shift"),
            })
        }
        ReferenceNode::Compare { op, a, b } => {
            let (a, b) = (word(a), word(b));
            ReferenceScalar::Bool(match op {
                CmpOp::Eq => a == b,
                CmpOp::Ne => a != b,
                CmpOp::Lt => a < b,
                CmpOp::Le => a <= b,
                CmpOp::Gt => a > b,
                CmpOp::Ge => a >= b,
            })
        }
        ReferenceNode::And { a, b } => ReferenceScalar::Bool(boolean(a) && boolean(b)),
        ReferenceNode::Not { value } => ReferenceScalar::Bool(!boolean(value)),
        ReferenceNode::Select { condition, yes, no } => {
            if boolean(condition) {
                get(yes)
            } else {
                get(no)
            }
        }
        ReferenceNode::Bits { value } => ReferenceScalar::U32(get(value).bits()),
        ReferenceNode::FromBits { value, dtype } => ReferenceScalar::from_bits(dtype, word(value)),
    }
}
pub fn evaluate(
    recipe: &ReferenceRecipe,
    inputs: &[ReferenceScalar],
) -> Result<ReferenceScalar, ScalarFailure> {
    assert_eq!(
        inputs.len(),
        recipe
            .nodes
            .iter()
            .filter(|node| matches!(node, ReferenceNode::Input { .. }))
            .count(),
        "reference operand arity"
    );
    let mut values = Vec::with_capacity(recipe.nodes.len());
    for node in &recipe.nodes {
        values.push(match *node {
            ReferenceNode::Input { operand, dtype } => {
                let value = inputs[operand as usize];
                assert_eq!(value.dtype(), dtype, "reference operand type");
                value
            }
            _ => terminal(node, |v| values[v.ordinal()]),
        });
    }
    for &(predicate, failure) in &recipe.failures {
        if values[predicate.ordinal()] == ReferenceScalar::Bool(true) {
            return Err(failure);
        }
    }
    Ok(values[recipe.output.ordinal()])
}
const F32_TY: DType = DType::F32;
const U32_TY: DType = DType::U32;
const I32_TY: DType = DType::I32;
#[derive(Default)]
struct RecipeBuilder {
    nodes: Vec<ReferenceNode>,
    interned: HashMap<ReferenceNode, V>,
    failures: Vec<(V, ScalarFailure)>,
}
impl RecipeBuilder {
    fn constant(&self, v: V) -> Option<ReferenceScalar> {
        if let ReferenceNode::Constant(x) = self.nodes[v.ordinal()] {
            Some(x)
        } else {
            None
        }
    }
    fn push(&mut self, ty: DType, mut node: ReferenceNode) -> V {
        // Constant propagation is performed by the same terminal evaluator; it
        // never introduces another host implementation of source arithmetic.
        let operands: Vec<V> = match node {
            ReferenceNode::Input { .. } | ReferenceNode::Constant(_) => vec![],
            ReferenceNode::Word { a, b, .. }
            | ReferenceNode::Compare { a, b, .. }
            | ReferenceNode::And { a, b } => vec![a, b],
            ReferenceNode::Not { value }
            | ReferenceNode::Bits { value }
            | ReferenceNode::FromBits { value, .. } => vec![value],
            ReferenceNode::Select { condition, yes, no } => {
                if yes == no {
                    return yes;
                }
                if let Some(ReferenceScalar::Bool(condition)) = self.constant(condition) {
                    return if condition { yes } else { no };
                }
                vec![condition, yes, no]
            }
        };
        if !operands.is_empty() && operands.iter().all(|&v| self.constant(v).is_some()) {
            node = ReferenceNode::Constant(terminal(&node, |v| self.constant(v).unwrap()));
        }
        if let Some(&v) = self.interned.get(&node) {
            assert_eq!(v.ty, ty);
            return v;
        }
        let v = V {
            ordinal: self.nodes.len().try_into().expect("recipe node capacity"),
            ty,
        };
        self.nodes.push(node.clone());
        self.interned.insert(node, v);
        v
    }
}
#[derive(Default)]
struct Recipe {
    builder: RefCell<RecipeBuilder>,
}
impl Recipe {
    fn input(&self, operand: u32, dtype: DType) -> V {
        self.push(dtype, ReferenceNode::Input { operand, dtype })
    }
    fn push(&self, ty: DType, node: ReferenceNode) -> V {
        self.builder.borrow_mut().push(ty, node)
    }
    fn constant(&self, v: ReferenceScalar) -> V {
        self.push(v.dtype(), ReferenceNode::Constant(v))
    }
    fn f(&self, x: u32) -> V {
        self.constant(ReferenceScalar::F32(x))
    }
    fn u(&self, x: u32) -> V {
        self.constant(ReferenceScalar::U32(x))
    }
    fn i(&self, x: i32) -> V {
        self.constant(ReferenceScalar::I32(x))
    }
    fn boolean(&self, x: bool) -> V {
        self.constant(ReferenceScalar::Bool(x))
    }
    fn word(&self, op: WordOp, a: V, b: V) -> V {
        assert_eq!(a.ty, U32_TY);
        assert_eq!(b.ty, U32_TY);
        self.push(U32_TY, ReferenceNode::Word { op, a, b })
    }
    fn word_cmp(&self, op: CmpOp, a: V, b: V) -> V {
        assert_eq!(a.ty, U32_TY);
        assert_eq!(b.ty, U32_TY);
        self.push(DType::Bool, ReferenceNode::Compare { op, a, b })
    }
    fn bits(&self, v: V) -> V {
        if v.ty == U32_TY {
            v
        } else {
            self.push(U32_TY, ReferenceNode::Bits { value: v })
        }
    }
    fn typed(&self, v: V, dtype: DType) -> V {
        assert_eq!(v.ty, U32_TY);
        if dtype == DType::U32 {
            v
        } else {
            self.push(dtype, ReferenceNode::FromBits { value: v, dtype })
        }
    }
    fn and(&self, a: V, b: V) -> V {
        self.push(DType::Bool, ReferenceNode::And { a, b })
    }
    fn or(&self, a: V, b: V) -> V {
        self.not(self.and(self.not(a), self.not(b)))
    }
    fn not(&self, value: V) -> V {
        self.push(DType::Bool, ReferenceNode::Not { value })
    }
    fn select(&self, condition: V, yes: V, no: V) -> V {
        assert_eq!(condition.ty, DType::Bool);
        assert_eq!(yes.ty, no.ty);
        self.push(yes.ty, ReferenceNode::Select { condition, yes, no })
    }
    fn fail_if(&self, p: V, f: ScalarFailure) {
        self.builder.borrow_mut().failures.push((p, f));
    }
    fn finish(self, output: V) -> ReferenceRecipe {
        let b = self.builder.into_inner();
        ReferenceRecipe {
            nodes: b.nodes.into_boxed_slice(),
            output,
            failures: b.failures.into_boxed_slice(),
        }
    }
    fn bin(&self, op: BinaryOp, a: V, b: V) -> V {
        primitive::binary(
            self,
            match op {
                BinaryOp::Add => ast::BinaryOp::Add,
                BinaryOp::Sub => ast::BinaryOp::Sub,
                BinaryOp::Mul => ast::BinaryOp::Mul,
                BinaryOp::Div => ast::BinaryOp::Div,
            },
            a,
            b,
        )
    }
    fn bit(&self, op: BitOp, a: V, b: V) -> V {
        assert_eq!(a.ty, b.ty);
        let x = self.word(
            match op {
                BitOp::And => WordOp::And,
                BitOp::Or => WordOp::Or,
                BitOp::Shl => WordOp::Shl,
                BitOp::Shr => WordOp::Shr,
            },
            self.bits(a),
            self.bits(b),
        );
        self.typed(x, a.ty)
    }
    fn cmp(&self, op: CmpOp, a: V, b: V) -> V {
        primitive::compare(self, op, a, b)
    }
    fn neg(&self, value: V) -> V {
        primitive::negate(self, value)
    }
    fn cast(&self, value: V, to: DType) -> V {
        primitive::cast(self, value, to)
    }
    fn fbits(&self, value: V) -> V {
        assert_eq!(value.ty, F32_TY);
        self.bits(value)
    }
    fn from_bits(&self, value: V) -> V {
        self.typed(value, DType::F32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RecipeKey {
    Scalar(ScalarOp, Vec<DType>),
    Code(crate::registry::CodeInterpretation, u32),
    FloatCode(crate::registry::FloatCodeFormat),
}
fn cached_recipe(key: RecipeKey, build: impl FnOnce() -> ReferenceRecipe) -> Arc<ReferenceRecipe> {
    static RECIPES: OnceLock<Mutex<HashMap<RecipeKey, Arc<ReferenceRecipe>>>> = OnceLock::new();
    let cache = RECIPES.get_or_init(Mutex::default);
    if let Some(recipe) = cache.lock().unwrap().get(&key).cloned() {
        return recipe;
    }
    let recipe = Arc::new(build());
    cache.lock().unwrap().entry(key).or_insert(recipe).clone()
}

pub fn scalar_recipe(op: ScalarOp, operands: &[DType]) -> Arc<ReferenceRecipe> {
    cached_recipe(RecipeKey::Scalar(op, operands.to_vec()), || {
        let b = Recipe::default();
        let inputs: Vec<_> = operands
            .iter()
            .enumerate()
            .map(|(i, &d)| b.input(i as u32, d))
            .collect();
        let output = primitive::operation(&b, op, &inputs);
        b.finish(output)
    })
}

/// Interpret the bits of one registry code. This is the same terminal recipe
/// consumed by packed reference reads and kernel construction.
pub fn code_recipe(
    interpretation: &crate::registry::CodeInterpretation,
    bits: u32,
) -> Arc<ReferenceRecipe> {
    use crate::registry::CodeInterpretation;
    assert!((1..=32).contains(&bits), "code width is invalid");
    cached_recipe(RecipeKey::Code(interpretation.clone(), bits), || {
        let b = Recipe::default();
        let input = b.input(0, DType::U32);
        let mask = u32::MAX >> (32 - bits);
        let raw = b.word(WordOp::And, input, b.u(mask));
        let word = match interpretation {
            CodeInterpretation::Unsigned => raw,
            CodeInterpretation::TwosComplement => {
                let sign = b.u(1 << (bits - 1));
                b.word(WordOp::Sub, b.word(WordOp::Xor, raw, sign), sign)
            }
            CodeInterpretation::Offset(offset) => b.word(WordOp::Sub, raw, b.u(*offset as u32)),
            CodeInterpretation::Table(table) => {
                assert_eq!(
                    table.len() as u64,
                    1u64 << bits,
                    "code table must cover its codes"
                );
                let mut value = b.u(table[0] as u32);
                for (i, &entry) in table.iter().enumerate().skip(1) {
                    value = b.select(
                        b.word_cmp(CmpOp::Eq, raw, b.u(i as u32)),
                        b.u(entry as u32),
                        value,
                    );
                }
                value
            }
        };
        let out = b.typed(word, DType::I32);
        b.finish(out)
    })
}

/// Floating code formats are finite registry data. Expand their exact payload
/// table into ordinary word comparisons and selections before kernel closure.
pub fn float_code_recipe(format: crate::registry::FloatCodeFormat) -> Arc<ReferenceRecipe> {
    cached_recipe(RecipeKey::FloatCode(format), || {
        let b = Recipe::default();
        let input = b.input(0, DType::U32);
        let count = 1u32 << format.bits();
        let raw = b.word(WordOp::And, input, b.u(count - 1));
        let mut out = b.f(format.decode(0).to_bits());
        for code in 1..count {
            out = b.select(
                b.word_cmp(CmpOp::Eq, raw, b.u(code)),
                b.f(format.decode(code).to_bits()),
                out,
            );
        }
        b.finish(out)
    })
}

fn transcendental(b: &Recipe, op: MathOp, input: V) -> V {
    let original = input.ty;
    let x = b.cast(input, F32_TY);
    let result = match op {
        MathOp::Exp => exp(b, x),
        MathOp::Log => log(b, x),
        MathOp::Sin => trig(b, x, false),
        MathOp::Cos => trig(b, x, true),
        MathOp::Sqrt => sqrt(b, x),
        MathOp::Rsqrt => b.bin(BinaryOp::Div, b.f(0x3f80_0000), sqrt(b, x)),
        _ => unreachable!("non-transcendental operation"),
    };
    b.cast(result, original)
}

/// Quantization starts with the lexer's parsed F64 bits, never with F32.
pub fn float_literal(dtype: DType, value: f64) -> ReferenceScalar {
    primitive::literal(dtype, value.to_bits())
}
pub fn integer_literal(dtype: DType, value: i128) -> ReferenceScalar {
    primitive::integer_literal(dtype, value)
}

pub fn digest() -> [u8; 32] {
    // The semantic revision covers the terminal constructors and finite scalar
    // algorithms, as well as the ordered transcendental coefficients below.
    static DIGEST: OnceLock<[u8; 32]> = OnceLock::new();
    *DIGEST.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(VERSION.as_bytes());
        hash.update(include_bytes!("reference_math/primitive.rs"));
        hash.update(include_bytes!("reference_math.rs"));
        hash.finalize().into()
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use ast::BinaryOp as B;

    fn apply(op: ScalarOp, args: &[ReferenceScalar]) -> Result<ReferenceScalar, ScalarFailure> {
        let types: Vec<_> = args.iter().map(|x| x.dtype()).collect();
        evaluate(&scalar_recipe(op, &types), args)
    }
    fn canonical(bits: u32) -> u32 {
        if bits & 0x7fff_ffff > 0x7f80_0000 {
            0x7fc0_0000
        } else {
            bits
        }
    }
    #[test]
    fn literal_quantization_rounds_once_from_parsed_f64() {
        assert_eq!(
            float_literal(DType::F16, 1.0004882812500002),
            ReferenceScalar::F16(0x3c01)
        );
        assert_eq!(
            float_literal(DType::F16, -0.0),
            ReferenceScalar::F16(0x8000)
        );
        assert_eq!(
            float_literal(DType::F32, f64::MAX),
            ReferenceScalar::F32(0x7f80_0000)
        );
        assert_eq!(
            float_literal(DType::F32, f64::from_bits(1)),
            ReferenceScalar::F32(0)
        );
        assert_eq!(
            integer_literal(DType::F32, i128::from(i64::MIN)),
            ReferenceScalar::F32(0xdf00_0000)
        );
    }
    #[test]
    fn narrow_fma_rounds_the_exact_sum_once() {
        assert_eq!(
            apply(
                ScalarOp::Math(MathOp::Fma),
                &[
                    ReferenceScalar::BF16(0x3fc0),
                    ReferenceScalar::BF16(0x3f81),
                    ReferenceScalar::BF16(0xb280)
                ]
            ),
            Ok(ReferenceScalar::BF16(0x3fc1))
        );
    }
    #[test]
    fn exceptional_values_and_remainder_have_one_contract() {
        let f = |v: f32| ReferenceScalar::F32(v.to_bits());
        assert_eq!(apply(ScalarOp::Binary(B::Rem), &[f(7.), f(4.)]), Ok(f(3.)));
        assert_eq!(
            apply(ScalarOp::Binary(B::Rem), &[f(-8.), f(4.)]),
            Ok(f(-0.))
        );
        assert_eq!(
            apply(ScalarOp::Binary(B::Mul), &[f(0.), f(f32::INFINITY)]),
            Ok(ReferenceScalar::F32(0x7fc0_0000))
        );
        assert_eq!(
            apply(
                ScalarOp::Binary(B::Add),
                &[ReferenceScalar::F32(0xff80_0001), f(1.)]
            ),
            Ok(ReferenceScalar::F32(0x7fc0_0000))
        );
        assert_eq!(
            apply(ScalarOp::Math(MathOp::Min), &[f(0.), f(-0.)]),
            Ok(f(-0.))
        );
        assert_eq!(
            apply(ScalarOp::Math(MathOp::Max), &[f(-0.), f(0.)]),
            Ok(f(0.))
        );
        assert_eq!(
            apply(
                ScalarOp::Math(MathOp::Min),
                &[ReferenceScalar::F32(0xff80_0001), f(1.)]
            ),
            Ok(f(1.))
        );
    }
    #[test]
    fn source_integer_failures_are_outputs_of_total_recipes() {
        use ReferenceScalar::{I32, U32};
        for (a, c, failure) in [
            (i32::MIN, -1, ScalarFailure::SignedDivisionOverflow),
            (1, 0, ScalarFailure::IntegerDivisionByZero),
        ] {
            assert_eq!(
                apply(ScalarOp::Binary(B::Div), &[I32(a), I32(c)]),
                Err(failure)
            );
        }
        for (a, c) in [
            (i32::MIN, 3),
            (-1, i32::MIN),
            (i32::MIN, i32::MIN),
            (7, -3),
            (-7, -3),
        ] {
            assert_eq!(
                apply(ScalarOp::Binary(B::Div), &[I32(a), I32(c)]),
                Ok(I32(a.div_euclid(c)))
            );
            assert_eq!(
                apply(ScalarOp::Binary(B::Rem), &[I32(a), I32(c)]),
                Ok(I32(a.rem_euclid(c)))
            );
        }
        assert_eq!(
            apply(ScalarOp::Binary(B::Shl), &[U32(1), U32(32)]),
            Err(ScalarFailure::ShiftCount)
        );
        assert_eq!(
            apply(ScalarOp::Binary(B::Shr), &[I32(-2), I32(1)]),
            Ok(I32(-1))
        );
        assert_eq!(
            apply(ScalarOp::Binary(B::Mul), &[U32(u32::MAX), U32(u32::MAX)]),
            Ok(U32(1))
        );
    }
    #[test]
    fn narrow_transport_abs_neg_and_identity_preserve_payloads_exhaustively() {
        for dtype in [DType::F16, DType::BF16] {
            let abs = scalar_recipe(ScalarOp::Math(MathOp::Abs), &[dtype]);
            let neg = scalar_recipe(ScalarOp::Unary(ast::UnaryOp::Neg), &[dtype]);
            let identity = scalar_recipe(ScalarOp::Cast(dtype), &[dtype]);
            for bits in 0..=u16::MAX {
                let value = ReferenceScalar::from_bits(dtype, u32::from(bits));
                assert_eq!(
                    evaluate(&abs, &[value]).unwrap().bits(),
                    u32::from(bits & 0x7fff)
                );
                assert_eq!(
                    evaluate(&neg, &[value]).unwrap().bits(),
                    u32::from(bits ^ 0x8000)
                );
                assert_eq!(evaluate(&identity, &[value]).unwrap(), value);
            }
        }
    }
    #[test]
    fn finite_f32_recipes_match_independent_ieee_samples() {
        let ops = [B::Add, B::Sub, B::Mul, B::Div, B::Rem];
        let recipes: Vec<_> = ops
            .iter()
            .map(|&op| scalar_recipe(ScalarOp::Binary(op), &[DType::F32; 2]))
            .collect();
        let fma = scalar_recipe(ScalarOp::Math(MathOp::Fma), &[DType::F32; 3]);
        let mut state = 0x83ac_51a7u32;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let edge = [
            0,
            0x8000_0000,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
        ];
        for i in 0..300 {
            let a = if i < 100 { edge[i / 10] } else { random() };
            let c = if i < 100 { edge[i % 10] } else { random() };
            let z = random();
            let (x, y, w) = (f32::from_bits(a), f32::from_bits(c), f32::from_bits(z));
            for (op, recipe) in ops.iter().zip(&recipes) {
                let expected = match op {
                    B::Add => x + y,
                    B::Sub => x - y,
                    B::Mul => x * y,
                    B::Div => x / y,
                    B::Rem => x % y,
                    _ => unreachable!(),
                };
                let result =
                    evaluate(recipe, &[ReferenceScalar::F32(a), ReferenceScalar::F32(c)]).unwrap();
                assert_eq!(
                    result.bits(),
                    canonical(expected.to_bits()),
                    "{op:?}({a:08x},{c:08x})"
                );
            }
            let result = evaluate(
                &fma,
                &[
                    ReferenceScalar::F32(a),
                    ReferenceScalar::F32(c),
                    ReferenceScalar::F32(z),
                ],
            )
            .unwrap();
            assert_eq!(
                result.bits(),
                canonical(x.mul_add(y, w).to_bits()),
                "fma({a:08x},{c:08x},{z:08x})"
            );
        }
    }
    #[test]
    fn saturating_casts_and_narrow_cross_casts_are_explicit() {
        for value in [
            f32::NEG_INFINITY,
            -4294967296.,
            -2147483648.,
            -1.9,
            -0.,
            0.,
            1.9,
            2147483648.,
            4294967296.,
            f32::INFINITY,
            f32::NAN,
        ] {
            let input = ReferenceScalar::F32(value.to_bits());
            assert_eq!(
                apply(ScalarOp::Cast(DType::I32), &[input]),
                Ok(ReferenceScalar::I32(value as i32))
            );
            assert_eq!(
                apply(ScalarOp::Cast(DType::U32), &[input]),
                Ok(ReferenceScalar::U32(value as u32))
            );
        }
        assert_eq!(
            apply(ScalarOp::Cast(DType::BF16), &[ReferenceScalar::F16(0xfc01)]),
            Ok(ReferenceScalar::BF16(0x7fc0))
        );
        assert_eq!(
            apply(ScalarOp::Cast(DType::F16), &[ReferenceScalar::BF16(0x8000)]),
            Ok(ReferenceScalar::F16(0x8000))
        );
    }
    #[test]
    fn transcendental_recipes_have_only_total_word_terminals() {
        for op in [
            MathOp::Exp,
            MathOp::Log,
            MathOp::Sin,
            MathOp::Cos,
            MathOp::Sqrt,
            MathOp::Rsqrt,
        ] {
            let recipe = scalar_recipe(ScalarOp::Math(op), &[DType::F32]);
            for bits in [
                0,
                0x8000_0000,
                1,
                0x3f80_0000,
                0xbf80_0000,
                0x7f7f_ffff,
                0x7f80_0000,
                0xff80_0000,
                0x7f80_0001,
            ] {
                assert!(
                    evaluate(&recipe, &[ReferenceScalar::F32(bits)]).is_ok(),
                    "{op:?}({bits:08x})"
                );
            }
        }
        assert_eq!(
            evaluate(
                &scalar_recipe(ScalarOp::Math(MathOp::Exp), &[DType::F32]),
                &[ReferenceScalar::F32(0)]
            )
            .unwrap(),
            ReferenceScalar::F32(0x3f80_0000)
        );
        assert_eq!(
            evaluate(
                &scalar_recipe(ScalarOp::Math(MathOp::Log), &[DType::F32]),
                &[ReferenceScalar::F32(0x3f80_0000)]
            )
            .unwrap(),
            ReferenceScalar::F32(0)
        );
    }
    #[test]
    fn division_extrema_exercise_the_full_denominator_bound() {
        for (dtype, least, largest, infinity) in [
            (DType::F32, 1, 0x7f7f_ffff, 0x7f80_0000),
            (DType::BF16, 1, 0x7f7f, 0x7f80),
            (DType::F16, 1, 0x7bff, 0x7c00),
        ] {
            let least = ReferenceScalar::from_bits(dtype, least);
            let largest = ReferenceScalar::from_bits(dtype, largest);
            assert_eq!(
                apply(ScalarOp::Binary(B::Div), &[least, largest])
                    .unwrap()
                    .bits(),
                0
            );
            assert_eq!(
                apply(ScalarOp::Binary(B::Div), &[largest, least])
                    .unwrap()
                    .bits(),
                infinity
            );
        }
        assert_eq!(
            apply(
                ScalarOp::Binary(B::Div),
                &[
                    ReferenceScalar::U32(u32::MAX),
                    ReferenceScalar::U32(0x8000_0001)
                ]
            ),
            Ok(ReferenceScalar::U32(1))
        );
        assert_eq!(
            apply(
                ScalarOp::Binary(B::Rem),
                &[
                    ReferenceScalar::U32(u32::MAX),
                    ReferenceScalar::U32(0x8000_0001)
                ]
            ),
            Ok(ReferenceScalar::U32(0x7fff_fffe))
        );
    }
    #[test]
    fn code_recipes_preserve_signed_codes_and_floating_specials() {
        use crate::registry::{CodeInterpretation, FloatCodeFormat};
        let signed = code_recipe(&CodeInterpretation::TwosComplement, 8);
        for raw in 0u32..256 {
            assert_eq!(
                evaluate(&signed, &[ReferenceScalar::U32(raw)]),
                Ok(ReferenceScalar::I32(raw as u8 as i8 as i32))
            );
        }
        let shifted = code_recipe(&CodeInterpretation::Offset(32), 6);
        assert_eq!(
            evaluate(&shifted, &[ReferenceScalar::U32(0)]),
            Ok(ReferenceScalar::I32(-32))
        );
        assert_eq!(
            evaluate(&shifted, &[ReferenceScalar::U32(63)]),
            Ok(ReferenceScalar::I32(31))
        );
        for (format, raw, expected) in [
            (FloatCodeFormat::E2M1, 0, 0),
            (FloatCodeFormat::E2M1, 8, 0x8000_0000),
            (FloatCodeFormat::E2M1, 1, 0x3f00_0000),
            (FloatCodeFormat::E2M1, 7, 0x40c0_0000),
            (FloatCodeFormat::E4M3, 1, 0x3b00_0000),
            (FloatCodeFormat::E4M3, 0x7f, 0x7fc0_0000),
            (FloatCodeFormat::E4M3, 0xff, 0xffc0_0000),
            (FloatCodeFormat::UE4M3, 0x80, 0),
        ] {
            assert_eq!(
                evaluate(&float_code_recipe(format), &[ReferenceScalar::U32(raw)]),
                Ok(ReferenceScalar::F32(expected))
            );
        }
    }
    #[test]
    #[should_panic(expected = "reference operand arity")]
    fn malformed_recipe_invocation_is_a_caller_defect() {
        let recipe = scalar_recipe(ScalarOp::Unary(ast::UnaryOp::Neg), &[DType::F32]);
        let _ = evaluate(&recipe, &[ReferenceScalar::F32(0), ReferenceScalar::F32(0)]);
    }
}
