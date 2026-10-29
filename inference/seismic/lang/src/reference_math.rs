//! Language-owned scalar semantics as total, typed-bit recipes.
//!
//! The terminal vocabulary contains only unsigned word arithmetic, comparisons,
//! Boolean selection, constant-table lookup and bit-preserving transport. Both
//! interpretation and kernel construction consume this same graph. `exp` and
//! `log` preserve the ordered Sun/FreeBSD f32 recipes; their scalar steps
//! expand through the same owner.
//! `sqrt` is correctly rounded, and `sin`/`cos` evaluate the FreeBSD kernels in
//! 64-bit fixed point with one final rounding. Every math recipe returns the
//! canonical NaN for a NaN operand. `exp`, `log`, `sin` and `cos` are within one
//! ulp; `sqrt` is exact.
//!
//! The exp/log recipes and the sin/cos kernel coefficients originate in FreeBSD msun:
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunPro, a Sun Microsystems, Inc. business. Permission to use,
//! copy, modify, and distribute this software is freely granted, provided
//! that this notice is preserved.

pub mod conversion;
mod primitive;

use crate::intrinsics::MathOp;
use crate::syntax::ast;
use crate::types::DType;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub const VERSION: &str = "seismic-scalar-reference-bits-v5";

/// Operation identities describe source semantics, never physical approximations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ScalarOp {
    Binary(ast::BinaryOp),
    Unary(ast::UnaryOp),
    Math(MathOp),
    Cast(DType),
    /// Exact payload transport between F32 and U32.
    Bitcast(DType),
    /// A signed 64-bit two's-complement integer, given as the U32 words
    /// `(low, high)`, rounded once to nearest-even in the float dtype. It equals
    /// `integer_to_float` for every value in `[-2^63, 2^63)`.
    IntegerToFloat(DType),
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
        PrimitiveId::Bitcast(dtype) => ScalarOp::Bitcast(*dtype),
        _ => return None,
    })
}

/// Payload bits are authoritative, including NaN payloads and signed zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReferenceScalar {
    F16(u16),
    BF16(u16),
    F32(u32),
    I32(i32),
    U32(u32),
    U64(u64),
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
            Self::U64(_) => DType::U64,
            Self::Bool(_) => DType::Bool,
        }
    }
    /// The payload, zero-extended. Only a U64 payload exceeds 32 bits.
    pub fn bits(self) -> u64 {
        match self {
            Self::F16(x) | Self::BF16(x) => u64::from(x),
            Self::F32(x) | Self::U32(x) => u64::from(x),
            Self::I32(x) => u64::from(x as u32),
            Self::U64(x) => x,
            Self::Bool(x) => u64::from(x),
        }
    }
    /// The payload of a dtype; bits above its width are discarded.
    pub fn from_bits(dtype: DType, bits: u64) -> Self {
        match dtype {
            DType::F16 => Self::F16(bits as u16),
            DType::BF16 => Self::BF16(bits as u16),
            DType::F32 => Self::F32(bits as u32),
            DType::I32 => Self::I32(bits as i32),
            DType::U32 => Self::U32(bits as u32),
            DType::U64 => Self::U64(bits),
            DType::Bool => Self::Bool(bits != 0),
        }
    }
    /// Diagnostic projection only. Semantic operations consume `bits` instead.
    pub fn to_f64(self) -> f64 {
        match self {
            Self::F16(x) => conversion::exact_f64(DType::F16, u32::from(x)),
            Self::BF16(x) => conversion::exact_f64(DType::BF16, u32::from(x)),
            Self::F32(x) => f32::from_bits(x) as f64,
            Self::I32(x) => f64::from(x),
            Self::U32(x) => f64::from(x),
            Self::U64(x) => x as f64,
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
    Shl,
    Shr,
}

/// Word operations act on two words of one width, U32 or U64. A shift count
/// is a word of the shifted width and is below that width.
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
    /// The payload word of a value: U64 for a U64 value, else U32.
    Bits { value: V },
    FromBits { value: V, dtype: DType },
    /// A word of the other width: zero extension from U32 to U64, or the low
    /// word of a U64.
    Resize { value: V, dtype: DType },
    /// The U32 word of entry `index` of a constant table, an `i32` in two's
    /// complement (`LookupTable::entry`: the index modulo the table's length).
    Lookup {
        table: crate::registry::LookupTable,
        index: V,
    },
}
impl ReferenceNode {
    fn operands(&self) -> Vec<V> {
        match *self {
            Self::Input { .. } | Self::Constant(_) => vec![],
            Self::Word { a, b, .. } | Self::Compare { a, b, .. } | Self::And { a, b } => {
                vec![a, b]
            }
            Self::Not { value }
            | Self::Bits { value }
            | Self::FromBits { value, .. }
            | Self::Resize { value, .. } => vec![value],
            Self::Select { condition, yes, no } => vec![condition, yes, no],
            Self::Lookup { index, .. } => vec![index],
        }
    }
    fn ty(&self) -> DType {
        match *self {
            Self::Input { dtype, .. }
            | Self::FromBits { dtype, .. }
            | Self::Resize { dtype, .. } => dtype,
            Self::Constant(value) => value.dtype(),
            Self::Word { a, .. } => a.ty,
            Self::Bits { value } => word_type(value.ty),
            Self::Compare { .. } | Self::And { .. } | Self::Not { .. } => DType::Bool,
            Self::Select { yes, .. } => yes.ty,
            Self::Lookup { .. } => DType::U32,
        }
    }
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
            Self::ShiftCount => "integer shift count must be below the operand width",
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
    /// The recipe of a use whose failures are proved unreachable: the same output
    /// and operands, without the failure predicates and the nodes only they read.
    pub fn without_failures(&self) -> Self {
        let mut live = vec![false; self.nodes.len()];
        live[self.output.ordinal()] = true;
        for (ordinal, node) in self.nodes.iter().enumerate().rev() {
            if matches!(node, ReferenceNode::Input { .. }) {
                live[ordinal] = true;
            }
            if live[ordinal] {
                for operand in node.operands() {
                    live[operand.ordinal()] = true;
                }
            }
        }
        let mut renumbered: Vec<Option<V>> = vec![None; self.nodes.len()];
        let mut nodes = Vec::new();
        for (ordinal, node) in self.nodes.iter().enumerate() {
            if !live[ordinal] {
                continue;
            }
            let at = |v: V| renumbered[v.ordinal()].expect("operand precedes its use");
            let node = match *node {
                ReferenceNode::Input { .. } | ReferenceNode::Constant(_) => node.clone(),
                ReferenceNode::Word { op, a, b } => ReferenceNode::Word {
                    op,
                    a: at(a),
                    b: at(b),
                },
                ReferenceNode::Compare { op, a, b } => ReferenceNode::Compare {
                    op,
                    a: at(a),
                    b: at(b),
                },
                ReferenceNode::And { a, b } => ReferenceNode::And { a: at(a), b: at(b) },
                ReferenceNode::Not { value } => ReferenceNode::Not { value: at(value) },
                ReferenceNode::Select { condition, yes, no } => ReferenceNode::Select {
                    condition: at(condition),
                    yes: at(yes),
                    no: at(no),
                },
                ReferenceNode::Bits { value } => ReferenceNode::Bits { value: at(value) },
                ReferenceNode::FromBits { value, dtype } => ReferenceNode::FromBits {
                    value: at(value),
                    dtype,
                },
                ReferenceNode::Resize { value, dtype } => ReferenceNode::Resize {
                    value: at(value),
                    dtype,
                },
                ReferenceNode::Lookup { table, index } => ReferenceNode::Lookup {
                    table,
                    index: at(index),
                },
            };
            renumbered[ordinal] = Some(V {
                ordinal: nodes.len() as u32,
                ty: node.ty(),
            });
            nodes.push(node);
        }
        Self {
            nodes: nodes.into_boxed_slice(),
            output: renumbered[self.output.ordinal()].unwrap(),
            failures: Box::new([]),
        }
    }
}

/// The word type carrying the payload of a dtype.
pub fn word_type(dtype: DType) -> DType {
    if dtype == DType::U64 {
        DType::U64
    } else {
        DType::U32
    }
}

/// The bits of `dtype`'s word. An integer shift count is read as that word
/// (a negative count is a huge one) and must be below this width
/// ([`ScalarFailure::ShiftCount`]).
pub fn word_bits(dtype: DType) -> u32 {
    if word_type(dtype) == DType::U64 {
        64
    } else {
        32
    }
}
fn terminal(node: &ReferenceNode, get: impl Fn(V) -> ReferenceScalar) -> ReferenceScalar {
    // A word of either width, with the width of its operands.
    let words = |a: V, b: V| match (get(a), get(b)) {
        (ReferenceScalar::U32(a), ReferenceScalar::U32(b)) => (u64::from(a), u64::from(b), 32),
        (ReferenceScalar::U64(a), ReferenceScalar::U64(b)) => (a, b, 64),
        _ => panic!("non-word terminal"),
    };
    let word = |v| match get(v) {
        ReferenceScalar::U32(x) => u64::from(x),
        ReferenceScalar::U64(x) => x,
        _ => panic!("non-word terminal"),
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
            let (a, b, width) = words(a, b);
            let mask = u64::MAX >> (64 - width);
            let shift = |shifted: fn(u64, u32) -> Option<u64>| {
                assert!(b < width, "constructed word shift");
                shifted(a, b as u32).unwrap()
            };
            let value = match op {
                WordOp::Add => a.wrapping_add(b),
                WordOp::Sub => a.wrapping_sub(b),
                WordOp::And => a & b,
                WordOp::Or => a | b,
                WordOp::Xor => a ^ b,
                WordOp::Shl => shift(u64::checked_shl),
                WordOp::Shr => shift(u64::checked_shr),
            } & mask;
            ReferenceScalar::from_bits(if width == 64 { DType::U64 } else { DType::U32 }, value)
        }
        ReferenceNode::Compare { op, a, b } => {
            let (a, b, _) = words(a, b);
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
        ReferenceNode::Bits { value } => {
            let value = get(value);
            ReferenceScalar::from_bits(word_type(value.dtype()), value.bits())
        }
        ReferenceNode::FromBits { value, dtype } => ReferenceScalar::from_bits(dtype, word(value)),
        ReferenceNode::Resize { value, dtype } => ReferenceScalar::from_bits(dtype, word(value)),
        ReferenceNode::Lookup { table, index } => {
            let ReferenceScalar::U32(index) = get(index) else {
                panic!("a lookup index is a U32 word")
            };
            ReferenceScalar::U32(table.entry(index) as u32)
        }
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
const U64_TY: DType = DType::U64;
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
        if let ReferenceNode::Select { condition, yes, no } = node {
            if yes == no {
                return yes;
            }
            if let Some(ReferenceScalar::Bool(condition)) = self.constant(condition) {
                return if condition { yes } else { no };
            }
        }
        let operands = node.operands();
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
    fn u64(&self, x: u64) -> V {
        self.constant(ReferenceScalar::U64(x))
    }
    fn i(&self, x: i32) -> V {
        self.constant(ReferenceScalar::I32(x))
    }
    fn boolean(&self, x: bool) -> V {
        self.constant(ReferenceScalar::Bool(x))
    }
    fn word(&self, op: WordOp, a: V, b: V) -> V {
        assert!(a.ty == U32_TY || a.ty == U64_TY);
        assert_eq!(b.ty, a.ty);
        self.push(a.ty, ReferenceNode::Word { op, a, b })
    }
    fn word_cmp(&self, op: CmpOp, a: V, b: V) -> V {
        assert!(a.ty == U32_TY || a.ty == U64_TY);
        assert_eq!(b.ty, a.ty);
        self.push(DType::Bool, ReferenceNode::Compare { op, a, b })
    }
    fn bits(&self, v: V) -> V {
        if v.ty == U32_TY || v.ty == U64_TY {
            v
        } else {
            self.push(U32_TY, ReferenceNode::Bits { value: v })
        }
    }
    fn typed(&self, v: V, dtype: DType) -> V {
        assert_eq!(v.ty, word_type(dtype));
        if dtype == v.ty {
            v
        } else {
            self.push(dtype, ReferenceNode::FromBits { value: v, dtype })
        }
    }
    fn resize(&self, v: V, dtype: DType) -> V {
        assert!(v.ty == U32_TY || v.ty == U64_TY);
        assert!(dtype == U32_TY || dtype == U64_TY);
        if v.ty == dtype {
            v
        } else {
            self.push(dtype, ReferenceNode::Resize { value: v, dtype })
        }
    }
    /// The 32-bit limbs of a U64 word, low first.
    fn limbs(&self, v: V) -> [V; 2] {
        assert_eq!(v.ty, U64_TY);
        let high = self.word(WordOp::Shr, v, self.u64(32));
        [self.resize(v, U32_TY), self.resize(high, U32_TY)]
    }
    /// The U64 word of two 32-bit limbs, low first.
    fn joined(&self, limbs: [V; 2]) -> V {
        let high = self.word(WordOp::Shl, self.resize(limbs[1], U64_TY), self.u64(32));
        self.word(WordOp::Or, self.resize(limbs[0], U64_TY), high)
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
    fn lookup(&self, table: crate::registry::LookupTable, index: V) -> V {
        assert_eq!(index.ty, U32_TY);
        self.push(U32_TY, ReferenceNode::Lookup { table, index })
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

/// The widest code whose recipe's values `cached_code` tables; a wider
/// code's recipe is evaluated per read.
const TABLED_BITS: u32 = 16;

/// The value at `raw` of the code recipe of `key` (`recipe`: total, of one
/// `U32` input whose low `bits` bits it reads), for a code of at most
/// `TABLED_BITS` bits read from the recipe's values at every code, evaluated
/// once per process.
fn cached_code(
    key: RecipeKey,
    bits: u32,
    raw: u32,
    recipe: impl FnOnce() -> Arc<ReferenceRecipe>,
) -> ReferenceScalar {
    let value = |recipe: &ReferenceRecipe, raw| {
        evaluate(recipe, &[ReferenceScalar::U32(raw)]).expect("a code recipe is total")
    };
    if bits > TABLED_BITS {
        return value(&recipe(), raw);
    }
    type Tables = HashMap<RecipeKey, Arc<[ReferenceScalar]>>;
    static TABLES: OnceLock<RwLock<Tables>> = OnceLock::new();
    let tables = TABLES.get_or_init(Default::default);
    let known = tables.read().unwrap().get(&key).cloned();
    let table = known.unwrap_or_else(|| {
        let recipe = recipe();
        let table: Arc<[ReferenceScalar]> = (0..1u32 << bits).map(|raw| value(&recipe, raw)).collect();
        tables.write().unwrap().entry(key).or_insert(table).clone()
    });
    table[(raw & (u32::MAX >> (32 - bits))) as usize]
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
            CodeInterpretation::Odd => {
                b.word(WordOp::Add, b.word(WordOp::Shl, raw, b.u(1)), b.u(1))
            }
            CodeInterpretation::SignedOdd => {
                let low = b.word(WordOp::And, raw, b.u(mask >> 1));
                let odd = b.word(WordOp::Add, b.word(WordOp::Shl, low, b.u(1)), b.u(1));
                b.select(
                    b.word_cmp(CmpOp::Eq, b.word(WordOp::Shr, raw, b.u(bits - 1)), b.u(0)),
                    odd,
                    b.word(WordOp::Sub, b.u(0), odd),
                )
            }
            CodeInterpretation::Table(codebook) => {
                let table = codebook.values;
                assert_eq!(
                    table.len() as u64,
                    1u64 << bits,
                    "code table must cover its codes"
                );
                let mut value = b.u(i32::from(table[0]) as u32);
                for (i, &entry) in table.iter().enumerate().skip(1) {
                    value = b.select(
                        b.word_cmp(CmpOp::Eq, raw, b.u(i as u32)),
                        b.u(i32::from(entry) as u32),
                        value,
                    );
                }
                value
            }
            CodeInterpretation::Grid(_) => unreachable!(
                "a vector code depends on its value's lane: `DecodeStep::InterpretGridCode`"
            ),
        };
        let out = b.typed(word, DType::I32);
        b.finish(out)
    })
}

/// `code_recipe(interpretation, bits)` at `raw` (`cached_code`).
pub fn code_value(
    interpretation: &crate::registry::CodeInterpretation,
    bits: u32,
    raw: u32,
) -> ReferenceScalar {
    cached_code(RecipeKey::Code(interpretation.clone(), bits), bits, raw, || {
        code_recipe(interpretation, bits)
    })
}

/// `float_code_recipe(format)` at `raw` (`cached_code`).
pub fn float_code_value(format: crate::registry::FloatCodeFormat, raw: u32) -> ReferenceScalar {
    cached_code(RecipeKey::FloatCode(format), format.bits(), raw, || {
        float_code_recipe(format)
    })
}

/// The code of a vector-coded value (`GridCodes::code`) from its entry
/// (operand 0) and its lane (operand 1, below `GRID_VALUES`): word operations
/// around one lookup of the grid's lane values (`LookupTable::Grid`).
pub fn grid_code_recipe(grid: &'static crate::registry::GridCodes) -> ReferenceRecipe {
    use crate::registry::{GridSigns, LookupTable};
    let b = Recipe::default();
    let entry = b.input(0, DType::U32);
    let lane = b.input(1, DType::U32);
    let lane_shift = grid.lanes().trailing_zeros();
    assert_eq!(
        1 << lane_shift,
        grid.lanes(),
        "grid lanes are a power of two"
    );
    // The index this lane reads: index `lane / lanes` of the entry.
    let mut shift = b.u(0);
    for which in 1..grid.indices() {
        shift = b.select(
            b.word_cmp(
                CmpOp::Eq,
                b.word(WordOp::Shr, lane, b.u(lane_shift)),
                b.u(which),
            ),
            b.u(which * grid.index_bits()),
            shift,
        );
    }
    // Lane `lane % lanes` of that point, at `index * lanes + lane % lanes`
    // of the table: the lookup keeps the index's own bits, the table holding
    // `2^index_bits` points.
    let point = b.word(
        WordOp::Shl,
        b.word(WordOp::Shr, entry, shift),
        b.u(lane_shift),
    );
    let value = b.lookup(
        LookupTable::Grid(grid),
        b.word(
            WordOp::Or,
            point,
            b.word(WordOp::And, lane, b.u(grid.lanes() - 1)),
        ),
    );
    // A magnitude signed by its lane's sign (`GridSigns::lanes`: eight sign
    // bits, or seven and their even-parity eighth), or a signed value.
    let explicit = || {
        b.word(
            WordOp::And,
            b.word(WordOp::Shr, entry, b.u(grid.sign_bit())),
            b.u((1 << grid.signs.bits()) - 1),
        )
    };
    let lane_signs = match grid.signs {
        GridSigns::None => None,
        GridSigns::Explicit8 => Some(explicit()),
        GridSigns::ParityPlus7 => {
            let explicit = explicit();
            let mut parity = explicit;
            for fold in [4, 2, 1] {
                parity = b.word(WordOp::Xor, parity, b.word(WordOp::Shr, parity, b.u(fold)));
            }
            Some(b.word(
                WordOp::Or,
                explicit,
                b.word(
                    WordOp::Shl,
                    b.word(WordOp::And, parity, b.u(1)),
                    b.u(grid.signs.bits()),
                ),
            ))
        }
    };
    let value = match lane_signs {
        None => value,
        Some(signs) => {
            let negative = b.word(WordOp::And, b.word(WordOp::Shr, signs, lane), b.u(1));
            b.select(
                b.word_cmp(CmpOp::Eq, negative, b.u(1)),
                b.word(WordOp::Sub, b.u(0), value),
                value,
            )
        }
    };
    // A delta's code: `2^e value + 1`, `- 1` with the delta bit set.
    let value = match grid.delta {
        Some(exponent) => {
            let scaled = b.word(WordOp::Shl, value, b.u(exponent));
            let set = b.word(
                WordOp::And,
                b.word(WordOp::Shr, entry, b.u(grid.delta_bit())),
                b.u(1),
            );
            b.select(
                b.word_cmp(CmpOp::Eq, set, b.u(1)),
                b.word(WordOp::Sub, scaled, b.u(1)),
                b.word(WordOp::Add, scaled, b.u(1)),
            )
        }
        None => value,
    };
    let out = b.typed(value, DType::I32);
    b.finish(out)
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
        MathOp::Exp2 => exp2(b, x),
        MathOp::Log => log(b, x),
        MathOp::Sin => primitive::sine_or_cosine(b, x, false),
        MathOp::Cos => primitive::sine_or_cosine(b, x, true),
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
/// The meaning of a quantity cast to a float dtype: the exact integer rounded
/// once to nearest-even, overflowing to a signed infinity.
pub fn integer_to_float(dtype: DType, value: &num_bigint::BigInt) -> ReferenceScalar {
    assert!(
        dtype.is_float(),
        "integer_to_float target must be a float dtype"
    );
    primitive::integer_to_float(dtype, value)
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

// Base-two exponential has its own range reduction. Its source recipe does
// not first round the entire x*ln(2), which would lose the exact integer scale.
// The bounded fraction uses the existing F32 exponential recipe; power-of-two
// scaling and exceptional values are constructed from owned payload operations.
fn exp2(b: &Recipe, x: V) -> V {
    let bits = b.fbits(x);
    let magnitude = b.bit(BitOp::And, bits, b.u(0x7fffffff));
    let nan = b.cmp(CmpOp::Gt, magnitude, b.u(0x7f800000));
    let overflow = b.cmp(CmpOp::Ge, x, b.f(128f32.to_bits()));
    let underflow = b.cmp(CmpOp::Le, x, b.f((-150f32).to_bits()));
    let safe = b.select(b.or(nan, b.or(overflow, underflow)), b.f(0), x);
    let rounded = primitive::operation(b, ScalarOp::Math(MathOp::Rint), &[safe]);
    let k = b.cast(rounded, I32_TY);
    let fraction = b.bin(BinaryOp::Sub, safe, rounded);
    let hi = b.bin(BinaryOp::Mul, fraction, b.f(0x3f317200));
    let lo = b.bin(BinaryOp::Mul, fraction, b.f(0x35bfbe8e));
    let y = exp(b, b.bin(BinaryOp::Add, hi, lo));
    let deep = b.cmp(CmpOp::Lt, k, b.i(-126));
    let top = b.cmp(CmpOp::Eq, k, b.i(128));
    let scale_k = b.select(top, b.i(127), k);
    let field = b.bin(
        BinaryOp::Add,
        scale_k,
        b.select(deep, b.i(127 + 24), b.i(127)),
    );
    let scale = b.from_bits(b.bit(BitOp::Shl, b.cast(field, U32_TY), b.u(23)));
    let result = b.bin(BinaryOp::Mul, y, scale);
    let result = b.select(deep, b.bin(BinaryOp::Mul, result, b.f(0x33800000)), result);
    let result = b.select(
        top,
        b.bin(BinaryOp::Mul, result, b.f(2f32.to_bits())),
        result,
    );
    let result = b.select(
        overflow,
        b.f(0x7f800000),
        b.select(underflow, b.f(0), result),
    );
    b.select(nan, b.f(0x7fc00000), result)
}

// Sun/FreeBSD e_expf.c, expressed with explicit f32 operations and bit-built
// scales so the sequence is identical on every backend. Every branch is
// evaluated eagerly; its float-to-int conversion saturates outside the branch.
fn exp(b: &Recipe, x: V) -> V {
    let bits = b.fbits(x);
    let abs_bits = b.bit(BitOp::And, bits, b.u(0x7fff_ffff));
    let negative = b.cmp(CmpOp::Ne, b.bit(BitOp::And, bits, b.u(0x8000_0000)), b.u(0));
    let nan = b.cmp(CmpOp::Gt, abs_bits, b.u(0x7f80_0000));
    // x > o_threshold, including +inf; x < u_threshold, including -inf.
    let overflow = b.and(
        b.not(negative),
        b.cmp(CmpOp::Ge, abs_bits, b.u(0x42b1_7218)),
    );
    let underflow = b.and(negative, b.cmp(CmpOp::Gt, abs_bits, b.u(0x42cf_f1b5)));
    let reduces = b.cmp(CmpOp::Gt, abs_bits, b.u(0x3eb1_7218));
    let medium = b.cmp(CmpOp::Lt, abs_bits, b.u(0x3f85_1592));
    let tiny = b.cmp(CmpOp::Lt, abs_bits, b.u(0x3900_0000));

    // 0.5 ln2 < |x| < 1.5 ln2: hi = x - ln2HI[xsb], lo = ln2LO[xsb], k = 1-2xsb.
    let medium_hi = b.bin(
        BinaryOp::Sub,
        x,
        b.select(negative, b.f(0xbf31_7200), b.f(0x3f31_7200)),
    );
    let medium_lo = b.select(negative, b.f(0xb5bf_be8e), b.f(0x35bf_be8e));
    let medium_k = b.select(negative, b.i(-1), b.i(1));
    // Otherwise k = (int)(invln2*x + halF[xsb]), hi = x - k*ln2HI, lo = k*ln2LO.
    let half = b.select(negative, b.f(0xbf00_0000), b.f(0x3f00_0000));
    let large_k = b.cast(
        b.bin(
            BinaryOp::Add,
            b.bin(BinaryOp::Mul, b.f(0x3fb8_aa3b), x),
            half,
        ),
        I32_TY,
    );
    let large_kf = b.cast(large_k, F32_TY);
    let large_hi = b.bin(
        BinaryOp::Sub,
        x,
        b.bin(BinaryOp::Mul, large_kf, b.f(0x3f31_7200)),
    );
    let large_lo = b.bin(BinaryOp::Mul, large_kf, b.f(0x35bf_be8e));
    let hi = b.select(medium, medium_hi, large_hi);
    let lo = b.select(medium, medium_lo, large_lo);
    let k = b.select(reduces, b.select(medium, medium_k, large_k), b.i(0));
    let r = b.select(reduces, b.bin(BinaryOp::Sub, hi, lo), x);

    let one = b.f(0x3f80_0000);
    let two = b.f(0x4000_0000);
    let t = b.bin(BinaryOp::Mul, r, r);
    let c = b.bin(
        BinaryOp::Sub,
        r,
        b.bin(
            BinaryOp::Mul,
            t,
            b.bin(
                BinaryOp::Add,
                b.f(0x3e2a_aa8f),
                b.bin(BinaryOp::Mul, t, b.f(0xbb35_5215)),
            ),
        ),
    );
    let rc = b.bin(BinaryOp::Mul, r, c);
    // k == 0: one-((x*c)/(c-2.0)-x)
    let unreduced = b.bin(
        BinaryOp::Sub,
        one,
        b.bin(
            BinaryOp::Sub,
            b.bin(BinaryOp::Div, rc, b.bin(BinaryOp::Sub, c, two)),
            r,
        ),
    );
    // Otherwise y = one-((lo-(x*c)/(2.0-c))-hi), scaled by 2^k.
    let y = b.bin(
        BinaryOp::Sub,
        one,
        b.bin(
            BinaryOp::Sub,
            b.bin(
                BinaryOp::Sub,
                lo,
                b.bin(BinaryOp::Div, rc, b.bin(BinaryOp::Sub, two, c)),
            ),
            hi,
        ),
    );
    // k >= -125: y*twopk, and k == 128: y*2.0F*0x1p127F.
    // k < -125: y*twopk*twom100 with twopk = 2^(k+100).
    let deep = b.cmp(CmpOp::Lt, k, b.i(-125));
    let field = b.bin(BinaryOp::Add, k, b.select(deep, b.i(0x7f + 100), b.i(0x7f)));
    let twopk = b.from_bits(b.bit(BitOp::Shl, b.cast(field, U32_TY), b.u(23)));
    let scaled = b.bin(BinaryOp::Mul, y, twopk);
    let scaled = b.select(deep, b.bin(BinaryOp::Mul, scaled, b.f(0x0d80_0000)), scaled);
    let scaled = b.select(
        b.cmp(CmpOp::Eq, k, b.i(128)),
        b.bin(
            BinaryOp::Mul,
            b.bin(BinaryOp::Mul, y, two),
            b.f(0x7f00_0000),
        ),
        scaled,
    );
    let ordinary = b.select(b.cmp(CmpOp::Eq, k, b.i(0)), unreduced, scaled);
    let ordinary = b.select(tiny, b.bin(BinaryOp::Add, one, x), ordinary);
    let ordinary = b.select(overflow, b.f(0x7f80_0000), ordinary);
    let ordinary = b.select(underflow, b.f(0), ordinary);
    b.select(nan, b.f(0x7fc0_0000), ordinary)
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
    let invalid = b.or(sign, b.cmp(CmpOp::Gt, abs_bits, b.u(0x7f80_0000)));
    let result = b.select(special, x, result);
    let result = b.select(invalid, b.f(0x7fc0_0000), result);
    b.select(zero, b.f(0xff80_0000), result)
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
    fn grid_code_recipe_is_the_grid_code_of_every_entry_index_and_lane() {
        for repr in crate::repr::REPRS {
            let crate::registry::CodeInterpretation::Grid(grid) = repr.code else {
                continue;
            };
            let recipe = grid_code_recipe(grid);
            // Every index at every index position, with sign, delta and
            // carried bits drawn from a fixed sequence.
            let mut state = 0x9e37_79b9u32;
            for index in 0..grid.points().len() as u32 {
                for position in 0..grid.indices() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let mask = u32::MAX >> (32 - grid.entry_bits());
                    let entry = (state
                        & mask
                        & !(((1 << grid.index_bits()) - 1) << (position * grid.index_bits())))
                        | (index << (position * grid.index_bits()));
                    for lane in 0..crate::registry::GRID_VALUES {
                        let actual = evaluate(
                            &recipe,
                            &[ReferenceScalar::U32(entry), ReferenceScalar::U32(lane)],
                        )
                        .unwrap();
                        assert_eq!(
                            actual,
                            ReferenceScalar::I32(grid.code(entry, lane)),
                            "`{}` entry {entry:#x} lane {lane}",
                            repr.name
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn base_two_exponential_keeps_integer_scale_and_exceptional_values() {
        let compute = |bits| {
            apply(ScalarOp::Math(MathOp::Exp2), &[ReferenceScalar::F32(bits)])
                .unwrap()
                .bits() as u32
        };
        for exponent in -149i32..=127 {
            let expected = if exponent < -126 {
                1u32 << (exponent + 149)
            } else {
                ((exponent + 127) as u32) << 23
            };
            assert_eq!(compute((exponent as f32).to_bits()), expected);
        }
        for (input, output) in [
            (-150f32, 0u32),
            (128f32, 0x7f800000),
            (f32::INFINITY, 0x7f800000),
            (f32::NEG_INFINITY, 0),
        ] {
            assert_eq!(compute(input.to_bits()), output);
        }
        assert_eq!(compute(0x7f800001), 0x7fc00000);
        // Both signs of zero map to exactly one.
        assert_eq!(compute(0), 1f32.to_bits());
        assert_eq!(compute(0x80000000), 1f32.to_bits());
    }

    #[test]
    fn integral_rounding_and_payload_reinterpretation_are_distinct() {
        let round = |x: f32| {
            apply(
                ScalarOp::Math(MathOp::Rint),
                &[ReferenceScalar::F32(x.to_bits())],
            )
            .unwrap()
            .bits() as u32
        };
        for (x, y) in [
            (0.5, 0.0),
            (1.5, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (-0.5, -0.0),
            (-1.5, -2.0),
            (-2.5, -2.0),
        ] {
            assert_eq!(round(x), (y as f32).to_bits());
        }
        assert_eq!(round(f32::from_bits(1)), 0);
        assert_eq!(round(f32::from_bits(0x80000001)), 0x80000000);
        assert_eq!(round(f32::INFINITY), f32::INFINITY.to_bits());
        assert_eq!(round(f32::NEG_INFINITY), f32::NEG_INFINITY.to_bits());
        assert_eq!(round(f32::from_bits(0x7f800001)), 0x7fc00000);
        // Reinterpretation transports payloads, including signaling NaNs; a
        // numeric cast or arithmetic operation would destroy this distinction.
        for bits in [0, 0x80000000, 1, 0x7f800001, 0x7fc01234, 0xffffffff] {
            let value =
                apply(ScalarOp::Bitcast(DType::F32), &[ReferenceScalar::U32(bits)]).unwrap();
            assert_eq!(value.bits() as u32, bits);
            assert_eq!(
                apply(ScalarOp::Bitcast(DType::U32), &[value]).unwrap(),
                ReferenceScalar::U32(bits)
            );
        }
        for dtype in [DType::F16, DType::BF16] {
            for (x, expected) in [(0.5f32, 0.0f32), (1.5, 2.0), (-0.5, -0.0)] {
                let input =
                    apply(ScalarOp::Cast(dtype), &[ReferenceScalar::F32(x.to_bits())]).unwrap();
                let result = apply(ScalarOp::Math(MathOp::Rint), &[input]).unwrap();
                let expected = apply(
                    ScalarOp::Cast(dtype),
                    &[ReferenceScalar::F32(expected.to_bits())],
                )
                .unwrap();
                assert_eq!(result, expected);
            }
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
    fn u64_recipes_are_modular_words() {
        use ReferenceScalar::{U64, Bool, F32, I32, U32};
        let random = std::cell::RefCell::new(xorshift(0x5eed_6400));
        let random = || random.borrow_mut()();
        let word = || {
            // Mix full words with small and boundary values.
            let x = u64::from(random()) << 32 | u64::from(random());
            match random() % 8 {
                0 => 0,
                1 => u64::MAX,
                2 => x >> (random() % 64),
                3 => u64::from(random()),
                _ => x,
            }
        };
        let binary = |op, a, c| apply(ScalarOp::Binary(op), &[U64(a), U64(c)]);
        for _ in 0..600 {
            let (a, c) = (word(), word());
            assert_eq!(binary(B::Add, a, c), Ok(U64(a.wrapping_add(c))));
            assert_eq!(binary(B::Sub, a, c), Ok(U64(a.wrapping_sub(c))));
            assert_eq!(binary(B::Mul, a, c), Ok(U64(a.wrapping_mul(c))));
            assert_eq!(binary(B::BitAnd, a, c), Ok(U64(a & c)));
            assert_eq!(binary(B::BitOr, a, c), Ok(U64(a | c)));
            assert_eq!(binary(B::BitXor, a, c), Ok(U64(a ^ c)));
            assert_eq!(binary(B::Lt, a, c), Ok(Bool(a < c)));
            assert_eq!(binary(B::Eq, a, a), Ok(Bool(true)));
            if c == 0 {
                assert_eq!(
                    binary(B::Div, a, c),
                    Err(ScalarFailure::IntegerDivisionByZero)
                );
            } else {
                assert_eq!(binary(B::Div, a, c), Ok(U64(a / c)));
                assert_eq!(binary(B::Rem, a, c), Ok(U64(a % c)));
            }
            let count = c % 80;
            let shifted = |shift: fn(u64, u32) -> Option<u64>| {
                shift(a, count as u32).ok_or(ScalarFailure::ShiftCount).map(U64)
            };
            assert_eq!(binary(B::Shl, a, count), shifted(u64::checked_shl));
            assert_eq!(binary(B::Shr, a, count), shifted(u64::checked_shr));
            assert_eq!(
                apply(ScalarOp::Binary(B::Shr), &[U64(a), U32(count as u32)]),
                shifted(u64::checked_shr)
            );
            assert_eq!(
                apply(ScalarOp::Math(MathOp::Max), &[U64(a), U64(c)]),
                Ok(U64(a.max(c)))
            );
            assert_eq!(
                apply(ScalarOp::Unary(ast::UnaryOp::BitNot), &[U64(a)]),
                Ok(U64(!a))
            );
            assert_eq!(
                apply(ScalarOp::Unary(ast::UnaryOp::Neg), &[U64(a)]),
                Ok(U64(a.wrapping_neg()))
            );
            // Modular integer conversion in both directions.
            let low = a as u32;
            assert_eq!(apply(ScalarOp::Cast(DType::U32), &[U64(a)]), Ok(U32(low)));
            assert_eq!(
                apply(ScalarOp::Cast(DType::I32), &[U64(a)]),
                Ok(I32(low as i32))
            );
            assert_eq!(
                apply(ScalarOp::Cast(DType::U64), &[I32(low as i32)]),
                Ok(U64(low as i32 as u64))
            );
            assert_eq!(
                apply(ScalarOp::Cast(DType::U64), &[U32(low)]),
                Ok(U64(u64::from(low)))
            );
            assert_eq!(apply(ScalarOp::Cast(DType::Bool), &[U64(a)]), Ok(Bool(a != 0)));
            // Floating conversion rounds once and saturates.
            assert_eq!(
                apply(ScalarOp::Cast(DType::F32), &[U64(a)]),
                Ok(F32((a as f32).to_bits()))
            );
            let x = f32::from_bits(random());
            let expected = if x.is_nan() { 0 } else { x as u64 };
            assert_eq!(
                apply(ScalarOp::Cast(DType::U64), &[F32(x.to_bits())]),
                Ok(U64(expected)),
                "u64({x:e})"
            );
        }
        for x in [0.0f32, 1.0, 1.5, 4294967296.0, 1.8446743e19, f32::INFINITY, -3.0] {
            assert_eq!(
                apply(ScalarOp::Cast(DType::U64), &[F32(x.to_bits())]),
                Ok(U64(x as u64))
            );
        }
        assert_eq!(apply(ScalarOp::Cast(DType::U64), &[Bool(true)]), Ok(U64(1)));
        assert_eq!(integer_literal(DType::U64, i128::from(u64::MAX)), U64(u64::MAX));
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
                let value = ReferenceScalar::from_bits(dtype, u64::from(bits));
                assert_eq!(
                    evaluate(&abs, &[value]).unwrap().bits() as u32,
                    u32::from(bits & 0x7fff)
                );
                assert_eq!(
                    evaluate(&neg, &[value]).unwrap().bits() as u32,
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
                    result.bits() as u32,
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
                result.bits() as u32,
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

    fn xorshift(mut state: u32) -> impl FnMut() -> u32 {
        move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        }
    }
    fn unary(op: MathOp, dtype: DType) -> impl Fn(u32) -> u32 {
        let recipe = scalar_recipe(ScalarOp::Math(op), &[dtype]);
        move |bits| {
            evaluate(&recipe, &[ReferenceScalar::from_bits(dtype, bits.into())])
                .unwrap()
                .bits() as u32
        }
    }
    /// Distance in representable F32 values; +0 and -0 are one value.
    fn ulps(a: u32, c: u32) -> u64 {
        let key = |x: u32| {
            let magnitude = i64::from(x & 0x7fff_ffff);
            if x & 0x8000_0000 != 0 {
                -magnitude
            } else {
                magnitude
            }
        };
        key(a).abs_diff(key(c))
    }
    fn assert_within_one_ulp(name: &str, f: &impl Fn(u32) -> u32, g: fn(f64) -> f64, x: u32) {
        let expected = (g(f64::from(f32::from_bits(x))) as f32).to_bits();
        let result = f(x);
        assert!(
            ulps(result, expected) <= 1,
            "{name}({x:08x}) = {result:08x}, reference {expected:08x}"
        );
    }

    #[test]
    fn sqrt_is_correctly_rounded() {
        for (dtype, nan) in [(DType::F16, 0x7e00), (DType::BF16, 0x7fc0)] {
            let sqrt = unary(MathOp::Sqrt, dtype);
            for bits in 0..=u32::from(u16::MAX) {
                let x = conversion::exact_f64(dtype, bits);
                let expected = if x.is_nan() || x < 0.0 {
                    nan
                } else {
                    float_literal(dtype, x.sqrt()).bits() as u32
                };
                assert_eq!(sqrt(bits), expected, "sqrt({dtype:?} {bits:04x})");
            }
        }
        // The first F16 case the Newton sequence rounded the wrong way.
        assert_eq!(unary(MathOp::Sqrt, DType::F16)(0x0bff), 0x23ff);
        let sqrt = unary(MathOp::Sqrt, DType::F32);
        let mut random = xorshift(0x5a17_c3e9);
        let edges = [
            0,
            0x8000_0000,
            1,
            2,
            3,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
        ];
        let binade_edges = (1..255u32).flat_map(|field| [field << 23, (field << 23) - 1]);
        let samples = edges
            .into_iter()
            .chain([
                0x7f7f_ffff,
                0x7f80_0000,
                0xff80_0000,
                0xbf80_0000,
                0x7f80_0001,
            ])
            .chain(binade_edges)
            .chain((0..20_000).map(|_| random()));
        for bits in samples {
            let expected = canonical(f32::from_bits(bits).sqrt().to_bits());
            assert_eq!(sqrt(bits), expected, "sqrt({bits:08x})");
        }
    }

    #[test]
    fn rsqrt_is_one_over_sqrt_in_the_operand_dtype() {
        for (dtype, one) in [
            (DType::F32, 0x3f80_0000u32),
            (DType::F16, 0x3c00),
            (DType::BF16, 0x3f80),
        ] {
            let rsqrt = unary(MathOp::Rsqrt, dtype);
            let sqrt = unary(MathOp::Sqrt, dtype);
            let divide = scalar_recipe(ScalarOp::Binary(B::Div), &[dtype; 2]);
            let mut random = xorshift(0x0c0f_fee5);
            for _ in 0..300 {
                let bits = random()
                    & if dtype == DType::F32 {
                        u32::MAX
                    } else {
                        0xffff
                    };
                let expected = evaluate(
                    &divide,
                    &[
                        ReferenceScalar::from_bits(dtype, one.into()),
                        ReferenceScalar::from_bits(dtype, sqrt(bits).into()),
                    ],
                )
                .unwrap()
                .bits() as u32;
                assert_eq!(rsqrt(bits), expected, "rsqrt({dtype:?} {bits:08x})");
            }
        }
    }

    #[test]
    fn exp_has_no_plateau_below_the_overflow_threshold() {
        let exp = unary(MathOp::Exp, DType::F32);
        assert!(ulps(exp(88.5f32.to_bits()), 0x7f4c_dcc4) <= 1);
        assert_eq!(exp(0x42b1_7218), 0x7f80_0000);
        assert_ne!(exp(0x42b1_7217), 0x7f80_0000);
        for bits in (0x42b0_0000..=0x42b1_7217).step_by(47).chain([0x42b1_7217]) {
            assert_within_one_ulp("exp", &exp, f64::exp, bits);
        }
        // Every reduction branch and the gradual-underflow scale.
        let mut random = xorshift(0x1357_9bdf);
        for _ in 0..4_000 {
            let bits = random();
            if f32::from_bits(bits).abs() < 104.0 {
                assert_within_one_ulp("exp", &exp, f64::exp, bits);
            }
        }
        for x in [
            -103.97f32, -103.9, -100.0, -87.5, -1.5, -0.5, 0.25, 0.4, 0.6, 1.1, 60.0,
        ] {
            assert_within_one_ulp("exp", &exp, f64::exp, x.to_bits());
        }
        assert_eq!(exp(0), 0x3f80_0000);
        assert_eq!(exp(0x8000_0000), 0x3f80_0000);
        assert_eq!(exp(0x7f80_0000), 0x7f80_0000);
        assert_eq!(exp(0xff80_0000), 0);
    }

    #[test]
    fn sine_and_cosine_reduce_in_fixed_point() {
        let sin = unary(MathOp::Sin, DType::F32);
        let cos = unary(MathOp::Cos, DType::F32);
        // f32(3 * f32(pi)): the two-term reduction returned -0.
        assert_eq!(sin(0x4116_cbe4), 0xb2cc_de2e);
        // 16367173 * 2^72 is the F32 closest to a multiple of pi/2.
        let hardest = (16_367_173.0f32 * 2f32.powi(72)).to_bits();
        let mut samples = vec![hardest, hardest ^ 0x8000_0000, 0x3f49_0fda, 0x3f49_0fdb];
        let mut random = xorshift(0x2468_ace1);
        samples.extend((0..3_000).map(|_| random()));
        // Neighbourhoods of k * pi/2.
        samples.extend((1..=1u32 << 20).step_by(2_111).flat_map(|k| {
            let near = (f64::from(k) * std::f64::consts::FRAC_PI_2) as f32;
            (0..3).map(move |d| near.to_bits() - 1 + d)
        }));
        for bits in samples {
            if !f32::from_bits(bits).is_finite() {
                continue;
            }
            assert_within_one_ulp("sin", &sin, f64::sin, bits);
            assert_within_one_ulp("cos", &cos, f64::cos, bits);
        }
        assert_eq!(sin(0), 0);
        assert_eq!(sin(0x8000_0000), 0x8000_0000);
        assert_eq!(sin(0x8000_0001), 0x8000_0001);
        assert_eq!(cos(0x8000_0000), 0x3f80_0000);
        assert_eq!(cos(1), 0x3f80_0000);
        for bits in [0x7f80_0000, 0xff80_0000, 0x7fc0_1234] {
            assert_eq!(sin(bits), 0x7fc0_0000);
            assert_eq!(cos(bits), 0x7fc0_0000);
        }
    }

    /// Every 13th F16 encoding: all binades, signs and specials. The complete
    /// sweep (0 failures) takes minutes on an unoptimized build.
    #[test]
    fn narrow_transcendentals_are_within_one_ulp() {
        for op in [MathOp::Exp, MathOp::Log, MathOp::Sin, MathOp::Cos] {
            let reference: fn(f64) -> f64 = match op {
                MathOp::Exp => f64::exp,
                MathOp::Log => f64::ln,
                MathOp::Sin => f64::sin,
                _ => f64::cos,
            };
            let f = unary(op, DType::F16);
            for bits in (0..=u32::from(u16::MAX)).step_by(13) {
                let x = conversion::exact_f64(DType::F16, bits);
                let result = f(bits);
                let expected = float_literal(DType::F16, reference(x)).bits() as u32;
                if expected & 0x7c00 == 0x7c00 && expected & 0x3ff != 0 {
                    assert_eq!(result, 0x7e00, "{op:?}(f16 {bits:04x})");
                } else {
                    // F16 has the same sign-magnitude order, 16 bits lower.
                    assert!(
                        ulps(result << 16, expected << 16) >> 16 <= 1,
                        "{op:?}(f16 {bits:04x}) = {result:04x}, reference {expected:04x}"
                    );
                }
            }
        }
    }

    #[test]
    fn math_recipes_return_the_canonical_nan() {
        for (dtype, payloads, nan) in [
            (
                DType::F32,
                [0x7fc0_1234, 0xffc0_0001, 0x7f80_0001],
                0x7fc0_0000,
            ),
            (DType::F16, [0x7e12, 0xfe01, 0x7c01], 0x7e00),
            (DType::BF16, [0x7fc1, 0xffc0, 0x7f81], 0x7fc0),
        ] {
            for op in MathOp::ALL {
                if matches!(op, MathOp::Abs) {
                    continue;
                }
                let recipe = scalar_recipe(ScalarOp::Math(op), &vec![dtype; op.arity()]);
                for payload in payloads {
                    let args = vec![ReferenceScalar::from_bits(dtype, payload); op.arity()];
                    assert_eq!(
                        evaluate(&recipe, &args).unwrap().bits() as u32,
                        nan,
                        "{op:?}({dtype:?} {payload:08x})"
                    );
                }
            }
        }
    }

    #[test]
    fn quantity_to_float_rounds_the_exact_integer_once() {
        use num_bigint::BigInt;
        let convert = |dtype: DType, value: BigInt| integer_to_float(dtype, &value).bits() as u32;
        assert_eq!(convert(DType::F32, BigInt::from(16_777_217)), 0x4b80_0000);
        assert_eq!(convert(DType::F16, BigInt::from(65_520)), 0x7c00);
        assert_eq!(convert(DType::F16, BigInt::from(65_519)), 0x7bff);
        let power = |exponent: u32| BigInt::from(1) << exponent;
        assert_eq!(convert(DType::F32, power(63) + 1), 0x5f00_0000);
        assert_eq!(
            convert(DType::F32, BigInt::from(65_536).pow(4)),
            0x5f80_0000
        );
        assert_eq!(convert(DType::F32, power(100_000)), 0x7f80_0000);
        assert_eq!(convert(DType::BF16, -power(100_000)), 0xff80);
        assert_eq!(convert(DType::F32, BigInt::from(0)), 0);
        let mut random = xorshift(0x7777_1111);
        for _ in 0..2_000 {
            let value = (i64::from(random()) << 32 | i64::from(random())) >> (random() % 64);
            assert_eq!(
                convert(DType::F32, BigInt::from(value)),
                (value as f32).to_bits()
            );
            let narrow = value >> 11;
            for dtype in [DType::F16, DType::BF16] {
                assert_eq!(
                    convert(dtype, BigInt::from(narrow)),
                    float_literal(dtype, narrow as f64).bits() as u32,
                    "{dtype:?}({narrow})"
                );
            }
        }
    }

    #[test]
    fn word_integer_to_float_equals_the_exact_conversion_across_i64() {
        use num_bigint::BigInt;
        let mut values = vec![
            0,
            1,
            -1,
            i64::MIN,
            i64::MIN + 1,
            i64::MAX,
            i64::MAX - 1,
            16_777_217,
            -16_777_217,
            65_519,
            65_520,
            -65_520,
            (1 << 53) + 1,
            1 << 32,
            (1 << 32) - 1,
            -(1 << 32),
            u32::MAX as i64,
        ];
        for exponent in 0..63 {
            let power = 1i64 << exponent;
            values.extend([power, -power, power - 1, 1 - power, power + 1, -power - 1]);
            // Nearest-even ties at every binary precision of the three dtypes.
            for fraction in [8, 11, 24] {
                if exponent > fraction {
                    let half = 1i64 << (exponent - fraction - 1);
                    values.extend([power + half, -(power + half), power + 3 * half]);
                }
            }
        }
        let mut random = xorshift(0x1f2e_3d4c);
        for _ in 0..4_000 {
            let value = (i64::from(random()) << 32 | i64::from(random())) >> (random() % 64);
            values.push(value);
        }
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let recipe = scalar_recipe(ScalarOp::IntegerToFloat(dtype), &[DType::U32; 2]);
            for &value in &values {
                let words = [
                    ReferenceScalar::U32(value as u32),
                    ReferenceScalar::U32((value >> 32) as u32),
                ];
                assert_eq!(
                    evaluate(&recipe, &words),
                    Ok(integer_to_float(dtype, &BigInt::from(value))),
                    "{dtype:?}({value})"
                );
            }
        }
    }

    #[test]
    fn failure_free_projection_keeps_values_and_drops_failure_nodes() {
        use ReferenceScalar::I32;
        let full = scalar_recipe(ScalarOp::Binary(B::Div), &[DType::I32; 2]);
        let projected = full.without_failures();
        assert!(projected.failures().is_empty());
        assert!(projected.nodes().len() < full.nodes().len());
        for (a, c) in [(7, 2), (-7, 2), (i32::MIN, 3), (5, -1)] {
            assert_eq!(
                evaluate(&projected, &[I32(a), I32(c)]),
                evaluate(&full, &[I32(a), I32(c)])
            );
        }
        let add = scalar_recipe(ScalarOp::Binary(B::Add), &[DType::F32; 2]);
        let args = [
            ReferenceScalar::F32(0x3f80_0000),
            ReferenceScalar::F32(0x4000_0000),
        ];
        assert_eq!(
            evaluate(&add.without_failures(), &args),
            evaluate(&add, &args)
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
                    .bits() as u32,
                0
            );
            assert_eq!(
                apply(ScalarOp::Binary(B::Div), &[largest, least])
                    .unwrap()
                    .bits() as u32,
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
