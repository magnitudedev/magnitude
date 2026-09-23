//! Shared projection of actual closed scalar definitions, memory, and control.
//! Terms are derived from executable operations, never source identity assertions.
use super::*;
use seismic_ir::kernel::ops::{self, Op, ValueType};
use seismic_ir::repr::ScalarKind;
use seismic_ir::schedule::{ParametricSchedule, ScheduleStep};
use seismic_ir::storage::{GlobalAllocationTopology, ViewBase};
use seismic_lang::entry::SemanticProgram;
use seismic_lang::reference_math::{self as reference, ReferenceNode, ReferenceScalar, WordOp};
use std::collections::HashMap;
mod packed;
mod repeat;

pub(super) type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Term(pub(super) usize);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Node {
    /// Unique unavailable actual definition. It never establishes equality,
    /// including equality with itself at an analysis demand boundary.
    Opaque(usize),
    Iteration(u32),
    Header { depth:u32, ordinal:u32, kind:ScalarKind },
    Fold { start:Term, end:Term, initial:Vec<Term>, next:Vec<Term>, output:u32 },
    Input(seismic_lang::expr::SymbolId, ScalarKind),
    Constant(ReferenceScalar),
    Natural(u64),
    Word(WordOp, Term, Term),
    Compare(reference::CmpOp, Term, Term),
    And(Term, Term),
    Not(Term),
    Select(Term, Term, Term),
    Bits(Term),
    FromBits(Term, DType),
    NaturalFromWord(Term, DType),
    WordFromNatural(Term, DType),
    NaturalAdd(Term, Term),
    NaturalMul(Term, Term),
    NaturalDiv(Term, u32),
    NaturalRem(Term, u32),
    PackedBits { root: ViewBase, bit: Term, width: u32 },
    Read(Place),
}

/// An address in the actual closed storage owner. Selected allocation
/// instances will be resolved through its region product, never a new alias ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Place {
    pub(super) root: ViewBase,
    pub(super) byte: Term,
    pub(super) dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Effect {
    Write(Place, Term),
    Failure(Term, SourceFailure),
}

#[derive(Clone)]
pub(super) enum MemoryWrite {
    Point(Place, Term),
    Conditional(Term, Box<MemoryWrite>),
    Fill {
        root: ViewBase,
        start: u64,
        bytes: u64,
        pattern: seismic_ir::schedule::FillValue,
    },
}
impl MemoryWrite {
    pub(super) fn root(&self) -> ViewBase {
        match self {
            Self::Point(place, _) => place.root,
            Self::Conditional(_, write) => write.root(),
            Self::Fill { root, .. } => *root,
        }
    }
}

/// Lexical assumptions derived by traversing actual control. This is not a
/// source fact import: only the active physical arm contributes its predicate.
#[derive(Clone, Default)]
struct PathDomain {
    predicates: Vec<(Term, bool)>,
}
impl PathDomain {
    fn enter_arm(&mut self, condition: Term, taken: bool) {
        self.predicates.push((condition, taken));
    }
    fn establishes(&self, terms: &mut Terms, predicate: Term) -> bool {
        if terms.contains_opaque(predicate) {
            return false;
        }
        let predicate = terms.under(predicate, &self.predicates);
        matches!(terms.nodes[predicate.0], Node::Constant(ReferenceScalar::Bool(true)))
    }
}

#[derive(Clone, Default)]
pub(super) struct State {
    path: PathDomain,
    pub(super) slots: HashMap<seismic_lang::expr::SymbolId, Term>,
    pub(super) writes: Vec<MemoryWrite>,
    pub(super) effects: Vec<Effect>,
}

#[derive(Default)]
pub(super) struct Terms {
    pub(super) nodes: Vec<Node>,
    pub(super) interned: HashMap<Node, Term>,
}

impl Terms {
    pub(super) fn opaque(&mut self) -> Term {
        let ordinal = self.nodes.len();
        self.node(Node::Opaque(ordinal))
    }
    pub(super) fn contains_opaque(&self, term: Term) -> bool {
        let mut pending = vec![term];
        let mut seen = std::collections::HashSet::new();
        while let Some(term) = pending.pop() {
            if !seen.insert(term) { continue; }
            match &self.nodes[term.0] {
                Node::Opaque(_) => return true,
                Node::Input(..) | Node::Constant(_) | Node::Natural(_) | Node::Iteration(_) | Node::Header{..} => {},
                Node::Fold{start,end,initial,next,..} => { pending.extend([*start,*end]); pending.extend(initial); pending.extend(next); },
                Node::Word(_, a, b) | Node::Compare(_, a, b) | Node::And(a, b)
                | Node::NaturalAdd(a, b) | Node::NaturalMul(a, b) => pending.extend([*a, *b]),
                Node::Select(condition, yes, no) => pending.extend([*condition, *yes, *no]),
                Node::Not(value) | Node::Bits(value) | Node::FromBits(value, _)
                | Node::NaturalFromWord(value, _) | Node::WordFromNatural(value, _)
                | Node::NaturalDiv(value, _) | Node::NaturalRem(value, _) => pending.push(*value),
                Node::PackedBits { bit, .. } => pending.push(*bit),
                Node::Read(place) => pending.push(place.byte),
            }
        }
        false
    }
    /// Restrict a term to the successful continuation of preceding source
    /// failures. Undefined results on a failed path are never observations.
    pub(super) fn under(&mut self, term: Term, assumptions: &[(Term, bool)]) -> Term {
        let mut projected: Vec<Term> = Vec::with_capacity(term.0 + 1);
        for ordinal in 0..=term.0 {
            let current = Term(ordinal);
            let get = |term: Term| projected[term.0];
            let known = assumptions.iter().find_map(|(term, value)| {
                if *term == current {
                    Some(*value)
                } else if matches!(self.nodes[term.0], Node::Not(inner) if inner == current) {
                    Some(!value)
                } else {
                    None
                }
            });
            let value = if let Some(value) = known {
                self.boolean(value)
            } else {
                match self.nodes[ordinal].clone() {
                    Node::Word(op, a, b) => self.node(Node::Word(op, get(a), get(b))),
                    Node::Compare(op, a, b) => self.compare(op, get(a), get(b)),
                    Node::And(a, b) => self.and(get(a), get(b)),
                    Node::Not(a) => self.not(get(a)),
                    Node::Select(c, a, b) => self.select(get(c), get(a), get(b)),
                    Node::Bits(a) => self.node(Node::Bits(get(a))),
                    Node::FromBits(a, dtype) => self.node(Node::FromBits(get(a), dtype)),
                    Node::NaturalFromWord(a, dtype) => {
                        self.node(Node::NaturalFromWord(get(a), dtype))
                    }
                    Node::WordFromNatural(a,dtype) => self.word_from_natural(get(a),dtype),
                    Node::NaturalAdd(a, b) => self.natural_binary(false, get(a), get(b)),
                    Node::NaturalMul(a, b) => self.natural_binary(true, get(a), get(b)),
                    Node::NaturalDiv(value, divisor) => self.natural_div_rem(false,get(value),divisor),
                    Node::NaturalRem(value, divisor) => self.natural_div_rem(true,get(value),divisor),
                    Node::PackedBits{root,bit,width} => self.node(Node::PackedBits{root,bit:get(bit),width}),
                    Node::Fold{start,end,initial,next,output} => self.node(Node::Fold{start:get(start),end:get(end),initial:initial.into_iter().map(get).collect(),next:next.into_iter().map(get).collect(),output}),
                    Node::Read(mut place) => {
                        place.byte = get(place.byte);
                        self.node(Node::Read(place))
                    }
                    _ => current,
                }
            };
            projected.push(value);
        }
        projected[term.0]
    }
    pub(super) fn node(&mut self, node: Node) -> Term {
        if let Some(term) = self.interned.get(&node) {
            return *term;
        }
        let term = Term(self.nodes.len());
        self.nodes.push(node.clone());
        self.interned.insert(node, term);
        term
    }
    pub(super) fn scalar(&mut self, value: ReferenceScalar) -> Term {
        self.node(Node::Constant(value))
    }
    pub(super) fn boolean(&mut self, value: bool) -> Term {
        self.scalar(ReferenceScalar::Bool(value))
    }
    pub(super) fn select(&mut self, condition: Term, yes: Term, no: Term) -> Term {
        if yes == no {
            return yes;
        }
        match (&self.nodes[yes.0], &self.nodes[no.0]) {
            (
                Node::Constant(ReferenceScalar::Bool(true)),
                Node::Constant(ReferenceScalar::Bool(false)),
            ) => return condition,
            (
                Node::Constant(ReferenceScalar::Bool(false)),
                Node::Constant(ReferenceScalar::Bool(true)),
            ) => return self.not(condition),
            _ => {}
        }
        match self.nodes[condition.0] {
            Node::Constant(ReferenceScalar::Bool(true)) => yes,
            Node::Constant(ReferenceScalar::Bool(false)) => no,
            _ => self.node(Node::Select(condition, yes, no)),
        }
    }
    pub(super) fn and(&mut self, a: Term, b: Term) -> Term {
        if a == b {
            return a;
        }
        match (&self.nodes[a.0], &self.nodes[b.0]) {
            (Node::Constant(ReferenceScalar::Bool(false)), _)
            | (_, Node::Constant(ReferenceScalar::Bool(false))) => self.boolean(false),
            (Node::Constant(ReferenceScalar::Bool(true)), _) => b,
            (_, Node::Constant(ReferenceScalar::Bool(true))) => a,
            _ => self.node(Node::And(a, b)),
        }
    }
    pub(super) fn not(&mut self, value: Term) -> Term {
        match self.nodes[value.0] {
            Node::Constant(ReferenceScalar::Bool(value)) => self.boolean(!value),
            Node::Not(value) => value,
            _ => self.node(Node::Not(value)),
        }
    }
    pub(super) fn compare(&mut self, op: reference::CmpOp, a: Term, b: Term) -> Term {
        // Natural geometry is separate from source word arithmetic. Only
        // exact constants/identity simplify here; scalar recipes stay intact.
        let constant = match (&self.nodes[a.0], &self.nodes[b.0]) {
            (Node::Natural(a), Node::Natural(b)) => Some(a.cmp(b)),
            (Node::Constant(ReferenceScalar::U32(a)), Node::Constant(ReferenceScalar::U32(b))) => {
                Some(a.cmp(b))
            }
            (
                Node::Constant(ReferenceScalar::Bool(a)),
                Node::Constant(ReferenceScalar::Bool(b)),
            ) => Some(a.cmp(b)),
            _ if a == b && !self.contains_opaque(a) => Some(std::cmp::Ordering::Equal),
            _ => None,
        };
        if let Some(ordering) = constant {
            use reference::CmpOp::*;
            return self.boolean(match op {
                Eq => ordering.is_eq(),
                Ne => !ordering.is_eq(),
                Lt => ordering.is_lt(),
                Le => !ordering.is_gt(),
                Gt => ordering.is_gt(),
                Ge => !ordering.is_lt(),
            });
        }
        if let Node::Select(condition, yes, no) = self.nodes[a.0] {
            if matches!(self.nodes[yes.0], Node::Constant(_))
                && matches!(self.nodes[no.0], Node::Constant(_))
            {
                let yes = self.compare(op, yes, b);
                let no = self.compare(op, no, b);
                return self.select(condition, yes, no);
            }
        }
        self.node(Node::Compare(op, a, b))
    }
    pub(super) fn natural(&mut self, value: Term, dtype: DType) -> Result<Term> {
        Ok(match self.nodes[value.0] {
            Node::Constant(ReferenceScalar::I32(value)) if value >= 0 => {
                self.node(Node::Natural(value as u64))
            }
            Node::Constant(ReferenceScalar::U32(value)) => {
                self.node(Node::Natural(u64::from(value)))
            }
            Node::Natural(_)
            | Node::Iteration(_)
            | Node::Input(_, ScalarKind::Nat64)
            | Node::NaturalFromWord(..)
            | Node::NaturalAdd(..)
            | Node::NaturalMul(..) => value,
            _ if matches!(dtype, DType::I32 | DType::U32) => {
                self.node(Node::NaturalFromWord(value, dtype))
            }
            _ => return Err("index relation requires an actual natural or word value"),
        })
    }
    pub(super) fn word_from_natural(&mut self, value:Term, dtype:DType) -> Term {
        if let Node::Natural(value) = self.nodes[value.0] {
            return self.scalar(ReferenceScalar::from_bits(dtype,value as u32));
        }
        self.node(Node::WordFromNatural(value,dtype))
    }
    pub(super) fn natural_binary(&mut self, multiply: bool, a: Term, b: Term) -> Term {
        let (left, right) = (&self.nodes[a.0], &self.nodes[b.0]);
        match (left, right) {
            (Node::Natural(a), Node::Natural(b)) => {
                if let Some(value) = if multiply {
                    a.checked_mul(*b)
                } else {
                    a.checked_add(*b)
                } {
                    return self.node(Node::Natural(value));
                }
            }
            (Node::Natural(0), _) => return if multiply { a } else { b },
            (_, Node::Natural(0)) => return if multiply { b } else { a },
            (Node::Natural(1), _) if multiply => return b,
            (_, Node::Natural(1)) if multiply => return a,
            _ => {}
        }
        self.node(if multiply {
            Node::NaturalMul(a, b)
        } else {
            Node::NaturalAdd(a, b)
        })
    }
    pub(super) fn bits(&mut self, value: Term, dtype: DType) -> Term {
        match dtype {
            DType::U32 => value,
            DType::Bool => {
                let one = self.scalar(ReferenceScalar::U32(1));
                let zero = self.scalar(ReferenceScalar::U32(0));
                self.select(value, one, zero)
            }
            _ => self.node(Node::Bits(value)),
        }
    }
    pub(super) fn from_bits(&mut self, value: Term, dtype: DType) -> Term {
        match dtype {
            DType::U32 => value,
            DType::Bool => {
                let zero = self.scalar(ReferenceScalar::U32(0));
                self.compare(reference::CmpOp::Ne, value, zero)
            }
            _ => self.node(Node::FromBits(value, dtype)),
        }
    }
    pub(super) fn recipe(
        &mut self,
        recipe: &reference::ReferenceRecipe,
        inputs: &[Term],
    ) -> (Term, Vec<(Term, reference::ScalarFailure)>) {
        let mut values = Vec::with_capacity(recipe.nodes().len());
        for node in recipe.nodes() {
            let get = |value: reference::ReferenceValue| values[value.ordinal()];
            let value = match *node {
                ReferenceNode::Input { operand, .. } => inputs[operand as usize],
                ReferenceNode::Constant(value) => self.scalar(value),
                ReferenceNode::Word { op, a, b } => self.node(Node::Word(op, get(a), get(b))),
                ReferenceNode::Compare { op, a, b } => self.compare(op, get(a), get(b)),
                ReferenceNode::And { a, b } => self.and(get(a), get(b)),
                ReferenceNode::Not { value } => self.not(get(value)),
                ReferenceNode::Select { condition, yes, no } => {
                    self.select(get(condition), get(yes), get(no))
                }
                ReferenceNode::Bits { value } => self.bits(get(value), value.ty()),
                ReferenceNode::FromBits { value, dtype } => self.from_bits(get(value), dtype),
            };
            values.push(value);
        }
        (
            values[recipe.output().ordinal()],
            recipe
                .failures()
                .iter()
                .map(|(value, cause)| (values[value.ordinal()], *cause))
                .collect(),
        )
    }
}

pub(super) struct Analysis<'a> {
    pub(super) terms: Terms,
    pub(super) expressions: &'a ExprArena,
    pub(super) inputs: HashMap<seismic_lang::expr::SymbolId, Term>,
    pub(super) storage: &'a GlobalAllocationTopology,
    pub(super) external: std::collections::HashSet<ViewBase>,
    pub(super) loop_depth: u32,
}
impl Analysis<'_> {
    /// Discharge access geometry from the actual coordinate and view terms.
    /// Unknown entailment is unfinished safety, including for discarded reads.
    pub(super) fn coordinates_in_bounds(&mut self, view: AnyBufferView, indices: &[Term], state: &State) -> Result<()> {
        let extents = self.storage.view(view).extents.clone();
        if indices.len() != extents.len() {
            return Err("actual memory index rank differs");
        }
        for (&index, extent) in indices.iter().zip(extents) {
            let extent = self.expression(extent.into(), &state.slots)?;
            let within = self.terms.compare(reference::CmpOp::Lt, index, extent);
            if !state.path.establishes(&mut self.terms, within) {
                return Err("actual memory coordinate safety is not established");
            }
        }
        Ok(())
    }

    pub(super) fn place(&mut self, view: AnyBufferView, indices: &[Term], state: &State) -> Result<Place> {
        self.coordinates_in_bounds(view, indices, state)?;
        let layout = self.storage.view(view);
        if !matches!(layout.base, ViewBase::Allocation(_)) {
            return Err("selected allocation-instance relation is unfinished");
        }
        if !matches!(layout.mapping, seismic_ir::storage::ViewMapping::Direct | seismic_ir::storage::ViewMapping::WholeAllocation) {
            return Err("transformed place relation is unfinished");
        }
        let RepresentationKind::Dense(dtype) =
            registry::representation_info(view.representation()).kind
        else {
            return Err("packed storage relation is unfinished");
        };
        if indices.len() != layout.strides.len() {
            return Err("actual memory index rank differs");
        }
        let offset = layout.offset;
        let strides = layout.strides.clone();
        let root = layout.base;
        let mut byte = self.expression(offset.into(), &state.slots)?;
        let width = self.terms.node(Node::Natural(u64::from(dtype.bytes())));
        for (&index, stride) in indices.iter().zip(strides) {
            let stride = self.expression(stride.into(), &state.slots)?;
            let element = self.terms.natural_binary(true, index, stride);
            let displacement = self.terms.natural_binary(true, element, width);
            byte = self.terms.natural_binary(false, byte, displacement);
        }
        Ok(Place { root, byte, dtype })
    }
    pub(super) fn read(&mut self, state: &State, place: Place) -> Result<Term> {
        let mut value = self.terms.node(Node::Read(place.clone()));
        for written in &state.writes {
            if written.root() != place.root {
                let (ViewBase::Allocation(a),ViewBase::Allocation(b)) = (written.root(),place.root) else { return Err("selected allocation alias relation is unfinished"); };
                if self.storage.allocations_may_overlap(a,b) {
                    return Err("cross-root mutable alias relation is unfinished");
                }
                continue;
            }
            value = self.read_after_write(&place, value, written)?;
        }
        Ok(value)
    }
    pub(super) fn read_after_write(
        &mut self,
        place: &Place,
        mut value: Term,
        written: &MemoryWrite,
    ) -> Result<Term> {
        match written {
            MemoryWrite::Conditional(condition, write) => {
                let replacement = self.read_after_write(place, value, write)?;
                value = self.terms.select(*condition, replacement, value);
            }
            MemoryWrite::Point(written, replacement) => {
                if written.dtype != place.dtype {
                    return Err("overlapping typed storage relation is unfinished");
                }
                let same = self
                    .terms
                    .compare(reference::CmpOp::Eq, place.byte, written.byte);
                value = self.terms.select(same, *replacement, value);
            }
            MemoryWrite::Fill {
                start,
                bytes,
                pattern,
                ..
            } => {
                let Node::Natural(address) = self.terms.nodes[place.byte.0] else {
                    return Err("symbolic filled-region read relation is unfinished");
                };
                let end = start
                    .checked_add(*bytes)
                    .ok_or("fill byte extent overflow")?;
                let read_end = address
                    .checked_add(u64::from(place.dtype.bytes()))
                    .ok_or("read byte extent overflow")?;
                if read_end <= *start || address >= end {
                    return Ok(value);
                }
                if address < *start || read_end > end {
                    return Err("partial filled scalar relation is unfinished");
                }
                let mut bits = [0; 4];
                for (index, byte) in bits[..place.dtype.bytes() as usize].iter_mut().enumerate() {
                    *byte = pattern.pattern()
                        [((address - start) as usize + index) % pattern.pattern().len()];
                }
                value = self.terms.scalar(ReferenceScalar::from_bits(
                    place.dtype,
                    u32::from_le_bytes(bits),
                ));
            }
        }
        Ok(value)
    }
    pub(super) fn write(&mut self, state: &mut State, place: Place, value: Term) {
        if self.external.contains(&place.root) {
            state.effects.push(Effect::Write(place.clone(), value));
        }
        state.writes.push(MemoryWrite::Point(place, value));
    }
    pub(super) fn failure(&mut self, state: &mut State, failed: Term, cause: SourceFailure) {
        if !matches!(
            self.terms.nodes[failed.0],
            Node::Constant(ReferenceScalar::Bool(false))
        ) {
            state.effects.push(Effect::Failure(failed, cause));
        }
    }
    pub(super) fn expression(
        &mut self,
        expression: seismic_lang::expr::AnyExpr,
        slots: &HashMap<seismic_lang::expr::SymbolId, Term>,
    ) -> Result<Term> {
        use seismic_lang::expr::NodeView;
        Ok(match self.expressions.view(expression) {
            NodeView::NatConst(value) => self.terms.node(Node::Natural(value)),
            NodeView::IntConst(value) if value >= 0 => self.terms.node(Node::Natural(value as u64)),
            NodeView::BoolConst(value) => self.terms.boolean(value),
            NodeView::ScalarConst { dtype, bits } => {
                self.terms.scalar(ReferenceScalar::from_bits(dtype, bits))
            }
            NodeView::Symbol(symbol) => *slots
                .get(&symbol)
                .or_else(|| self.inputs.get(&symbol))
                .ok_or("expression has an unbound actual operand")?,
            NodeView::Binary {
                op: seismic_lang::expr::BinaryOp::Add,
                lhs,
                rhs,
            }
            | NodeView::Binary {
                op: seismic_lang::expr::BinaryOp::Mul,
                lhs,
                rhs,
            } => {
                let multiply = matches!(
                    self.expressions.view(expression),
                    NodeView::Binary {
                        op: seismic_lang::expr::BinaryOp::Mul,
                        ..
                    }
                );
                let a = self.expression(lhs, slots)?;
                let b = self.expression(rhs, slots)?;
                self.terms.natural_binary(multiply, a, b)
            }
            NodeView::Select {
                cond,
                then,
                otherwise,
            } => {
                let condition = self.expression(cond.into(), slots)?;
                let yes = self.expression(then, slots)?;
                let no = self.expression(otherwise, slots)?;
                self.terms.select(condition, yes, no)
            }
            NodeView::Cmp { op, lhs, rhs } => {
                let a = self.expression(lhs, slots)?;
                let b = self.expression(rhs, slots)?;
                let op = match op {
                    seismic_lang::expr::CmpOp::Eq => reference::CmpOp::Eq,
                    seismic_lang::expr::CmpOp::Ne => reference::CmpOp::Ne,
                    seismic_lang::expr::CmpOp::Lt => reference::CmpOp::Lt,
                    seismic_lang::expr::CmpOp::Le => reference::CmpOp::Le,
                    seismic_lang::expr::CmpOp::Gt => reference::CmpOp::Gt,
                    seismic_lang::expr::CmpOp::Ge => reference::CmpOp::Ge,
                };
                self.terms.compare(op, a, b)
            }
            _ => return Err("quantity expression relation is unfinished"),
        })
    }

    pub(super) fn physical_block<B: seismic_target::TargetFamily>(
        &mut self,
        kernel: &seismic_ir::kernel::Kernel<B>,
        block: seismic_ir::kernel::BlockId,
        values: &mut HashMap<ops::ErasedValue, Term>,
        state: &mut State,
    ) -> Result<Vec<Term>> {
        for operation in &kernel.block(block).ops {
            if let Op::Yield { values: yields } = operation {
                return yields.iter().map(|value| values.get(value).copied().ok_or("physical operand unavailable")).collect();
            }
            let assignment = (|| -> Result<Option<(ops::ErasedValue, Term)>> {
                let get = |value: &ops::ErasedValue| values.get(value).copied().ok_or("physical operand unavailable");
                Ok(match operation {

                Op::Constant { out, value } => Some((
                    *out,
                    match value {
                        ops::ConstantValue::F32(value) => {
                            self.terms.scalar(ReferenceScalar::F32(value.to_bits()))
                        }
                        ops::ConstantValue::F16(value) => {
                            self.terms.scalar(ReferenceScalar::F16(*value))
                        }
                        ops::ConstantValue::BF16(value) => {
                            self.terms.scalar(ReferenceScalar::BF16(*value))
                        }
                        ops::ConstantValue::U32(value) => {
                            self.terms.scalar(ReferenceScalar::U32(*value))
                        }
                        ops::ConstantValue::I32(value) => {
                            self.terms.scalar(ReferenceScalar::I32(*value))
                        }
                        ops::ConstantValue::Bool(value) => self.terms.boolean(*value),
                        ops::ConstantValue::Index(value) => self.terms.node(Node::Natural(*value)),
                    },
                )),
                Op::ScalarArg { out, index } => {
                    let (symbol, _) = kernel.interface().scalar_args[*index as usize];
                    Some((
                        *out,
                        *state
                            .slots
                            .get(&symbol)
                            .or_else(|| self.inputs.get(&symbol))
                            .ok_or("physical scalar input unavailable")?,
                    ))
                }
                Op::NatArg { out, index } => Some((
                    *out,
                    self.expression(
                        kernel.interface().nat_args[*index as usize].into(),
                        &state.slots,
                    )?,
                )),
                Op::Extent { out, place, axis } => {
                    let ops::PlaceRef::Global { slot } = place else {
                        return Err("local extent relation is unfinished");
                    };
                    let view = kernel.interface().bindings[slot.ordinal() as usize].view;
                    Some((
                        *out,
                        self.expression(
                            self.storage.view(view).extents[*axis as usize].into(),
                            &state.slots,
                        )?,
                    ))
                }
                Op::Binary { op, out, a, b }
                    if kernel.value_type(*out) == ValueType::Scalar(DType::U32) =>
                {
                    let operation = match op {
                        ops::BinaryOp::Add => WordOp::Add,
                        ops::BinaryOp::Sub => WordOp::Sub,
                        _ => {
                            return Err(
                                "physical word operation has no terminal reference relation",
                            )
                        }
                    };
                    Some((
                        *out,
                        self.terms.node(Node::Word(operation, get(a)?, get(b)?)),
                    ))
                }
                Op::Bit { op, out, a, b }
                    if kernel.value_type(*out) == ValueType::Scalar(DType::U32) =>
                {
                    let operation = match op {
                        ops::BitOp::And => WordOp::And,
                        ops::BitOp::Or => WordOp::Or,
                        ops::BitOp::Xor => WordOp::Xor,
                        ops::BitOp::Shl => WordOp::Shl,
                        ops::BitOp::Shr => WordOp::Shr,
                    };
                    Some((
                        *out,
                        self.terms.node(Node::Word(operation, get(a)?, get(b)?)),
                    ))
                }
                Op::Cmp { op, out, a, b }
                    if matches!(
                        kernel.value_type(*a),
                        ValueType::Scalar(DType::U32) | ValueType::Index | ValueType::Bool
                    ) =>
                {
                    let operation = match op {
                        ops::CmpOp::Eq => reference::CmpOp::Eq,
                        ops::CmpOp::Ne => reference::CmpOp::Ne,
                        ops::CmpOp::Lt => reference::CmpOp::Lt,
                        ops::CmpOp::Le => reference::CmpOp::Le,
                        ops::CmpOp::Gt => reference::CmpOp::Gt,
                        ops::CmpOp::Ge => reference::CmpOp::Ge,
                    };
                    Some((*out, self.terms.compare(operation, get(a)?, get(b)?)))
                }
                Op::ScalarBits { out, a } => Some((
                    *out,
                    self.terms.bits(get(a)?, dtype(kernel.value_type(*a))?),
                )),
                Op::ScalarFromBits { out, a } => Some((
                    *out,
                    self.terms
                        .from_bits(get(a)?, dtype(kernel.value_type(*out))?),
                )),
                Op::Bitcast { out, a, to } => {
                    let value = self.terms.bits(get(a)?, dtype(kernel.value_type(*a))?);
                    Some((*out, self.terms.from_bits(value, dtype(*to)?)))
                }
                Op::Cast {
                    out,
                    a,
                    to: ValueType::Index,
                } => Some((
                    *out,
                    self.terms.natural(get(a)?, dtype(kernel.value_type(*a))?)?,
                )),
                Op::Cast { out, a, to: ValueType::Scalar(dtype @ (DType::I32 | DType::U32)) }
                    if kernel.value_type(*a) == ValueType::Index => {
                    Some((*out,self.terms.word_from_natural(get(a)?,*dtype)))
                }
                Op::Read {
                    out, place, index, ..
                } => {
                    let ops::PlaceRef::Global { slot } = place else {
                        return Err("local place relation is unfinished");
                    };
                    let view = kernel.interface().bindings[slot.ordinal() as usize].view;
                    let coordinates = index.iter().map(get).collect::<Result<Vec<_>>>()?;
                    let place = self.place(view, &coordinates, state)?;
                    Some((*out, self.read(state, place)?))
                }
                Op::ReadPlaneField { out, place, plane, field, index } => {
                    let ops::PlaceRef::Global {slot} = place else { return Err("local packed field relation is unfinished"); };
                    let view = kernel.interface().bindings[slot.ordinal() as usize].view;
                    let coordinates = index.iter().map(get).collect::<Result<Vec<_>>>()?;
                    Some((*out,self.plane_field(view,&coordinates,*plane,*field,state)?))
                }
                Op::Write {
                    place,
                    index,
                    value,
                    ..
                } => {
                    let ops::PlaceRef::Global { slot } = place else {
                        return Err("local place relation is unfinished");
                    };
                    let view = kernel.interface().bindings[slot.ordinal() as usize].view;
                    let coordinates = index.iter().map(get).collect::<Result<Vec<_>>>()?;
                    let place = self.place(view, &coordinates, state)?;
                    self.write(state, place, get(value)?);
                    None
                }
                Op::Atomic {
                    op,
                    place,
                    index,
                    value,
                    ..
                } => {
                    let ops::PlaceRef::Global { slot } = place else {
                        return Err("local atomic relation is unfinished");
                    };
                    let view = kernel.interface().bindings[slot.ordinal() as usize].view;
                    let coordinates = index.iter().map(get).collect::<Result<Vec<_>>>()?;
                    let place = self.place(view, &coordinates, state)?;
                    if place.dtype != DType::U32 {
                        return Err("non-word atomic relation is unfinished");
                    }
                    let replacement = get(value)?;
                    let previous = self.read(state, place.clone())?;
                    let value = match op {
                        seismic_lang::intrinsics::AtomicOp::Max
                            if matches!(
                                self.terms.nodes[previous.0],
                                Node::Constant(ReferenceScalar::U32(0))
                            ) =>
                        {
                            replacement
                        }
                        seismic_lang::intrinsics::AtomicOp::Max
                        | seismic_lang::intrinsics::AtomicOp::Min => {
                            let condition = self.terms.compare(
                                if *op == seismic_lang::intrinsics::AtomicOp::Max {
                                    reference::CmpOp::Ge
                                } else {
                                    reference::CmpOp::Le
                                },
                                previous,
                                replacement,
                            );
                            self.terms.select(condition, previous, replacement)
                        }
                        _ => return Err("atomic operation relation is unfinished"),
                    };
                    self.write(state, place, value);
                    None
                }
                Op::Select { out, cond, a, b } => {
                    Some((*out, self.terms.select(get(cond)?, get(a)?, get(b)?)))
                }
                Op::Logic { op, out, a, b } => {
                    let value = match op {
                        ops::LogicOp::And => self.terms.and(get(a)?, get(b)?),
                        ops::LogicOp::Or => {
                            let truth = self.terms.boolean(true);
                            self.terms.select(get(a)?, truth, get(b)?)
                        }
                    };
                    Some((*out, value))
                }
                Op::Not { out, a } => Some((*out, self.terms.not(get(a)?))),
                Op::StoreSlot { slot, value, .. } => {
                    state.slots.insert(
                        kernel.interface().result_slots[*slot as usize].symbol(),
                        get(value)?,
                    );
                    None
                }
                Op::Repeat { start, end, binder, carries_in, carry_params, body, outs } => {
                    let start = get(start)?;
                    let end = get(end)?;
                    let initial = carries_in.iter().map(get).collect::<Result<Vec<_>>>()?;
                    let results = self.scalar_kernel_repeat(kernel, start, end, *binder, initial, carry_params, *body, values, state)?;
                    if results.len() != outs.len() { return Err("repeat result arity differs"); }
                    for (out, result) in outs.iter().zip(results) { values.insert(*out, result); }
                    None
                }
                Op::Branch {
                    cond,
                    then,
                    otherwise,
                    outs,
                } => {
                    let condition = get(cond)?;
                    let mut yes = state.clone();
                    let mut no = state.clone();
                    yes.path.enter_arm(condition, true);
                    no.path.enter_arm(condition, false);
                    let yes_values =
                        self.physical_block(kernel, *then, &mut values.clone(), &mut yes)?;
                    let no_values =
                        self.physical_block(kernel, *otherwise, &mut values.clone(), &mut no)?;
                    if yes.effects.len() != state.effects.len()
                        || no.effects.len() != state.effects.len()
                    {
                        return Err("conditional observable effect relation is unfinished");
                    }
                    let prior = state.writes.len();
                    state.writes.extend(
                        yes.writes
                            .into_iter()
                            .skip(prior)
                            .map(|write| MemoryWrite::Conditional(condition, Box::new(write))),
                    );
                    let opposite = self.terms.not(condition);
                    state.writes.extend(
                        no.writes
                            .into_iter()
                            .skip(prior)
                            .map(|write| MemoryWrite::Conditional(opposite, Box::new(write))),
                    );
                    for (symbol, yes) in yes.slots {
                        let no = *no
                            .slots
                            .get(&symbol)
                            .ok_or("branch slot lacks a complete definition")?;
                        state
                            .slots
                            .insert(symbol, self.terms.select(condition, yes, no));
                    }
                    if outs.len() != yes_values.len() || outs.len() != no_values.len() {
                        return Err("branch result arity differs");
                    }
                    for ((out, yes), no) in outs.iter().zip(yes_values).zip(no_values) {
                        values.insert(*out, self.terms.select(condition, yes, no));
                    }
                    None
                }
                Op::Yield { .. } => unreachable!("handled before scalar projection"),
                _ => {
                    return Err(
                        "physical state, control or native operation relation is unfinished",
                    )
                }
                })
            })();
            let assignment = match assignment {
                Ok(value) => value,
                Err(_) if matches!(operation,
                    Op::Math { .. } | Op::Fma { .. } | Op::VectorFma { .. }
                    | Op::Bitcast { .. } | Op::ScalarBits { .. } | Op::ScalarFromBits { .. }
                    | Op::VectorSplat { .. } | Op::VectorFromLanes { .. }
                    | Op::Cmp { .. } | Op::Select { .. } | Op::Logic { .. } | Op::Not { .. }
                ) => {
                    // These closed register operations are total and have no
                    // access, participation, or failure effect. An unsupported
                    // numerical result may be opaque; memory/control operations
                    // and potentially partial integer operations must still fail
                    // projection even when none of their outputs is demanded.
                    for out in operation.defined_values() {
                        values.insert(*out, self.terms.opaque());
                    }
                    None
                }
                Err(reason) => return Err(reason),
            };
            if let Some((out, value)) = assignment {
                values.insert(out, value);
            }
        }
        Ok(Vec::new())
    }

    pub(super) fn schedule<B: seismic_target::TargetFamily>(
        &mut self,
        schedule: &ParametricSchedule<B>,
        kernels: &seismic_ir::kernel::KernelArena<B>,
        steps: &[ScheduleStep],
        state: &mut State,
    ) -> Result<()> {
        for step in steps {
            match step {
                ScheduleStep::Repeat { symbol, start, end, body, carries, .. } => {
                    let start = self.expression((*start).into(), &state.slots)?;
                    let end = self.expression((*end).into(), &state.slots)?;
                    self.scalar_schedule_repeat(schedule, kernels, *symbol, start, end, body, carries, state)?;
                }
                ScheduleStep::Imported { body, .. } => {
                    self.schedule(schedule, kernels, body, state)?
                }
                ScheduleStep::Launch(id) => {
                    let launch = schedule.launch(*id);
                    for expression in launch.grid.iter().chain(&launch.workgroup) {
                        if !matches!(
                            self.expressions.view((*expression).into()),
                            seismic_lang::expr::NodeView::NatConst(1)
                        ) {
                            return Err("participant-map relation is unfinished");
                        }
                    }
                    let kernel = kernels.kernel(launch.kernel);
                    self.physical_block(kernel, kernel.root(), &mut HashMap::new(), state)?;
                }
                ScheduleStep::ScalarMove(move_) => {
                    let value = *state
                        .slots
                        .get(&move_.from.symbol())
                        .ok_or("scalar move reads unavailable physical value")?;
                    state.slots.insert(move_.to.symbol(), value);
                }
                // Publication retains a descriptor; it neither reads nor writes
                // tensor elements. Tensor-result relations are checked by their
                // result owner, not this scalar memory-effect analysis.
                ScheduleStep::PublishTensor { .. } => {}
                ScheduleStep::BindArgumentTensor { .. } => return Err("tensor argument descriptor relation is unfinished"),
                ScheduleStep::BeginAllocationInstance { source: view, .. } => {
                    if self.loop_depth > 0 {
                        return Err("repeated allocation-instance relation is unfinished");
                    }
                    let root = self.storage.view(*view).base;
                    if self.external.contains(&root) {
                        return Err("external allocation redefinition differs from source");
                    }
                    state.writes.retain(|write| write.root() != root);
                }
                ScheduleStep::ScalarRead(read) => {
                    let indices = read
                        .index
                        .iter()
                        .map(|index| self.expression((*index).into(), &state.slots))
                        .collect::<Result<Vec<_>>>()?;
                    let place = self.place(read.source, &indices, state)?;
                    let value = self.read(state, place)?;
                    state.slots.insert(read.to.symbol(), value);
                }
                ScheduleStep::Fill(fill) => {
                    let view = self.storage.view(fill.destination);
                    if self.external.contains(&view.base) {
                        return Err("observable filled-region relation is unfinished");
                    }
                    let start = self.expression(view.offset.into(), &state.slots)?;
                    let bytes = self.expression(fill.bytes.into(), &state.slots)?;
                    let (Node::Natural(start), Node::Natural(bytes)) =
                        (&self.terms.nodes[start.0], &self.terms.nodes[bytes.0])
                    else {
                        return Err("symbolic filled-region relation is unfinished");
                    };
                    state.writes.push(MemoryWrite::Fill {
                        root: view.base,
                        start: *start,
                        bytes: *bytes,
                        pattern: fill.value,
                    });
                }
                ScheduleStep::Check(check) => {
                    let value = *state
                        .slots
                        .get(&check.condition.symbol())
                        .ok_or("source check has no actual published condition")?;
                    let failed = match check.expectation {
                        seismic_ir::schedule::ScalarCheckExpectation::BoolTrue => {
                            self.terms.not(value)
                        }
                        seismic_ir::schedule::ScalarCheckExpectation::U32Zero => {
                            let zero = self.terms.scalar(ReferenceScalar::U32(0));
                            self.terms.compare(reference::CmpOp::Ne, value, zero)
                        }
                    };
                    self.failure(state, failed, check.site.failure.clone());
                }
                _ => return Err("ordered state or schedule control relation is unfinished"),
            }
        }
        Ok(())
    }

}

pub(super) fn dtype(ty: ValueType) -> Result<DType> {
    match ty {
        ValueType::Scalar(dtype) => Ok(dtype),
        ValueType::Bool => Ok(DType::Bool),
        _ => Err("terminal scalar relation has non-scalar type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrelated_native_scalar_definition_does_not_hide_a_published_literal() {
        use crate::realization::demand_driven_tests::FakeTarget;
        use seismic_ir::construction::{AllocationPlan, Construction};
        let mut arena = ExprArena::default();
        let mut construction = Construction::<FakeTarget>::new(&mut arena, vec![], false, 0);
        let unknown_slot = construction.schedule_state().slot_any(&mut arena, ScalarKind::Scalar(DType::F32));
        let literal_slot = construction.schedule_state().slot_any(&mut arena, ScalarKind::Scalar(DType::U32));
        let vectors = seismic_ir::target::VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let unknown_output = kernel.result_slot(unknown_slot);
        let literal_output = kernel.result_slot(literal_slot);
        let input = kernel.constant(ops::ConstantValue::F32(1.0), ValueType::Scalar(DType::F32));
        let unavailable = kernel.math_approximate(seismic_lang::intrinsics::MathOp::Exp, input);
        kernel.store_slot(unknown_output, unavailable);
        let literal = kernel.constant(ops::ConstantValue::U32(3), ValueType::Scalar(DType::U32));
        kernel.store_slot(literal_output, literal);
        kernel.close();
        let token = construction.schedule(&mut arena, 0).close();
        let executable = construction.close(token).normalize_launches(&mut arena, 1024, 64).unwrap().analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct()).finish();
        let mut analysis = Analysis { terms: Terms::default(), expressions: &arena, inputs: HashMap::new(), storage: executable.storage(), external: Default::default(), loop_depth:0 };
        let (_, kernel) = executable.kernels().kernels().next().unwrap();
        let mut state = State::default();
        analysis.physical_block(kernel, kernel.root(), &mut HashMap::new(), &mut state).unwrap();
        assert!(analysis.terms.contains_opaque(state.slots[&unknown_slot.symbol()]));
        let literal = state.slots[&literal_slot.symbol()];
        assert!(!analysis.terms.contains_opaque(literal));
        assert!(matches!(analysis.terms.nodes[literal.0], Node::Constant(ReferenceScalar::U32(3))));
    }

    #[test]
    fn discarded_invalid_read_still_requires_actual_access_safety() {
        use crate::realization::demand_driven_tests::FakeTarget;
        use seismic_ir::construction::{AllocationPlan, Construction};
        use seismic_ir::storage::GlobalBufferKind;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<FakeTarget>::new(&mut arena, vec![], false, 0);
        let representation = registry::dense(DType::F32);
        let extent = arena.nat(1);
        let bytes = arena.nat(4);
        let zero = arena.nat(0);
        let allocation = construction.storage_mut().allocate(GlobalBufferKind::Arena, bytes, 4);
        let ordinal = construction.storage_mut().dense_view(&mut arena, allocation, representation, zero, vec![extent]);
        let view = construction.view(ordinal, representation);
        let vectors = seismic_ir::target::VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let place = kernel.arg_view(view, false);
        let index = kernel.index_constant(1);
        // The unused output must not erase the out-of-range access.
        let _discarded = kernel.read(place, &[index]);
        let kernel = kernel.close();
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.launch_sequential(kernel);
        let token = schedule.close();
        let executable = construction.close(token).normalize_launches(&mut arena, 1024, 64).unwrap().analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct()).finish();
        let mut analysis = Analysis { terms: Terms::default(), expressions: &arena, inputs: HashMap::new(), storage: executable.storage(), external: Default::default(), loop_depth: 0 };
        let (_, kernel) = executable.kernels().kernels().next().unwrap();
        assert_eq!(analysis.physical_block(kernel, kernel.root(), &mut HashMap::new(), &mut State::default()).unwrap_err(), "actual memory coordinate safety is not established");
    }

    #[test]
    fn actual_branch_facts_are_lexical_and_cannot_establish_opaque_bounds() {
        let mut arena = ExprArena::default();
        let symbol = arena.schedule_slot(0, seismic_lang::expr::SymbolSort::Nat);
        let mut terms = Terms::default();
        let index = terms.node(Node::Input(symbol, ScalarKind::Nat64));
        let extent = terms.node(Node::Natural(8));
        let within = terms.compare(reference::CmpOp::Lt, index, extent);
        let before = State::default();
        let mut yes = before.clone();
        let mut no = before.clone();
        yes.path.enter_arm(within, true);
        no.path.enter_arm(within, false);
        assert!(yes.path.establishes(&mut terms, within));
        assert!(!no.path.establishes(&mut terms, within));
        assert!(!before.path.establishes(&mut terms, within), "an arm premise does not escape its join");
        let unknown = terms.opaque();
        let unknown_bound = terms.compare(reference::CmpOp::Lt, unknown, extent);
        yes.path.enter_arm(unknown_bound, true);
        assert!(!yes.path.establishes(&mut terms, unknown_bound));
    }

    #[test]
    fn opaque_dependencies_remain_pending_without_poisoning_independent_terms() {
        let mut terms = Terms::default();
        let unavailable = terms.opaque();
        let another = terms.opaque();
        assert_ne!(unavailable, another);
        let one = terms.node(Node::Natural(1));
        let dependent = terms.natural_binary(false, unavailable, one);
        assert!(terms.contains_opaque(unavailable));
        assert!(terms.contains_opaque(dependent));
        let self_comparison = terms.compare(reference::CmpOp::Eq, unavailable, unavailable);
        assert!(terms.contains_opaque(self_comparison), "opaque identity is not a value proof");
        let mut arena = ExprArena::default();
        let condition_symbol = arena.schedule_slot(0, seismic_lang::expr::SymbolSort::Scalar(DType::Bool));
        let condition = terms.node(Node::Input(condition_symbol, ScalarKind::Scalar(DType::Bool)));
        let guarded = terms.select(condition, dependent, one);
        assert!(terms.contains_opaque(guarded));
        let known = terms.compare(reference::CmpOp::Le, one, one);
        assert!(!terms.contains_opaque(known));
        assert!(matches!(terms.nodes[known.0], Node::Constant(ReferenceScalar::Bool(true))));
    }
}
