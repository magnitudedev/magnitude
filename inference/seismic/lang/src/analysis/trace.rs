//! Symbolic execution of an entry's portable reference program.
//!
//! The trace follows the semantic program the interpreter runs, but with
//! symbolic values: every scalar is a [`Term`] over entry variables and loop
//! coordinates, and every tensor is a storage holding the writes made to it.
//! Loops are not unrolled: a write inside a loop is kept with its coordinate
//! free, and a later read solves the write's index for that coordinate. A
//! loop-carried value is resolved when it is an accumulation (`acc + g`,
//! `max(acc, g)`); any other carried dependence is left unresolved, and every
//! loop-carried floating dependence is recorded as a reduction. A branch whose
//! condition the entry's domain decides is followed only on the arm that
//! runs.
//!
//! A write of a floating value that is affine in the conversion of a stored
//! integer element, `slope * float(code) + intercept` with both resolved, is
//! recorded as a decode of that element.

use super::domain::Domain;
use super::term::{Binary, Fold, Node, Term, Unary, Variable};
use crate::entry::{
    LogicalEntryView, LoopKind, ParameterKind, RegionKind, ScalarRef, SemanticFunction,
    SemanticNodeView, SemanticType, SliceAxis, TensorStorage, ViewTransform,
};
use crate::expr::{AnyExpr, BinaryOp as ExprBinary, CmpOp, LoopBinderId, NaryOp, NodeView, SymbolKind, UnaryOp as ExprUnary};
use crate::ids::{FamilyId, RegionId, SemanticValueId};
use crate::intrinsics::{AtomicOp, MathOp, PrimitiveId, ReduceOp};
use crate::registry::{self, RepresentationKind};
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::DType;
use std::collections::HashMap;
use std::sync::Arc;

/// A symbolic value of the program.
#[derive(Clone, Debug)]
pub(crate) enum Value {
    Scalar(Term),
    Tensor(Tensor),
    Computed(Arc<Computed>),
    Tuple(Vec<Value>),
    Range(Term, Term),
    /// A value the trace does not model (an opaque intrinsic value).
    Unknown(Vec<Variable>),
}

/// A view of one storage.
#[derive(Clone, Debug)]
pub(crate) struct Tensor {
    pub(crate) storage: usize,
    /// View steps from the storage to this view, outermost last.
    steps: Vec<Step>,
    pub(crate) shape: Vec<Term>,
}

#[derive(Clone, Debug)]
enum Step {
    Slice(Vec<Axis>),
    Transpose(Vec<u32>),
    Reshape { from: Vec<Term>, to: Vec<Term> },
}

#[derive(Clone, Debug)]
enum Axis {
    Point(Term),
    Range(Term),
    Full,
}

/// A tensor computed elementwise or by a reduction, read lazily.
#[derive(Debug)]
pub(crate) enum Computed {
    Elementwise {
        op: PrimitiveId,
        inputs: Vec<Value>,
        dtype: DType,
        input_types: Vec<DType>,
    },
    Reduce {
        op: ReduceOp,
        axis: u32,
        input: Value,
        extent: Term,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum Root {
    /// An entry (or isolated function) tensor parameter: unwritten elements
    /// are its stored values.
    Parameter {
        name: String,
        representation: crate::ids::RepresentationId,
    },
    Fill(Term),
    Undefined,
    /// A copy or exact conversion of another tensor's value.
    Copy(Box<Value>),
}

#[derive(Clone, Debug)]
pub(crate) struct Storage {
    pub(crate) root: Root,
    pub(crate) entries: Vec<Entry>,
}

#[derive(Clone, Debug)]
pub(crate) enum Entry {
    Write(Record),
    /// Reads below this entry are the state carried into the iteration of
    /// loop placeholder `id`.
    Carried(u32),
}

#[derive(Clone, Debug)]
pub(crate) struct Record {
    pub(crate) index: Vec<Term>,
    pub(crate) value: Term,
    /// Conditions under which the write happened.
    pub(crate) guard: Vec<Term>,
    /// Closed loops whose coordinates remain free in `index` and `value`.
    pub(crate) generated: Vec<Coordinate>,
}

/// An open or closed loop coordinate.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Coordinate {
    pub(crate) binder: u32,
    pub(crate) start: Term,
    pub(crate) end: Term,
}

/// An accumulation of a floating value across loop iterations.
#[derive(Clone, Debug)]
pub(crate) struct ReductionRecord {
    /// The number of accumulation steps into one accumulator.
    pub(crate) extent: Term,
    pub(crate) accumulator: DType,
    /// Open loops whose coordinates remain free in `extent`.
    pub(crate) coordinates: Vec<Coordinate>,
    /// The storage accumulated into, for merging nested and sequential
    /// accumulations of the same state.
    pub(crate) storage: Option<usize>,
}

/// The conversion to floating of an integer computed from one stored integer
/// element of an entry parameter.
#[derive(Clone, Debug)]
struct Conversion {
    /// The converted value.
    term: Term,
    /// The integer converted.
    code: Term,
    /// The inclusive range of the integer converted.
    range: (f64, f64),
    parameter: String,
    /// The stored element's index.
    index: Vec<Term>,
}

/// A written value `slope * float(code) + intercept`.
#[derive(Clone, Debug)]
pub(crate) struct DecodeRecord {
    pub(crate) parameter: String,
    /// The index of the stored element the code is computed from.
    pub(crate) index: Vec<Term>,
    pub(crate) code: Term,
    pub(crate) range: (f64, f64),
    pub(crate) slope: Term,
    pub(crate) intercept: Term,
    /// Loops whose coordinates remain free in the terms, innermost first.
    pub(crate) coordinates: Vec<Coordinate>,
    conversion: Term,
}

/// The inclusive range of an integer word of `dtype`.
fn word_range(dtype: DType) -> Option<(f64, f64)> {
    match dtype {
        DType::I32 => Some((f64::from(i32::MIN), f64::from(i32::MAX))),
        DType::U32 => Some((0.0, f64::from(u32::MAX))),
        DType::U64 => Some((0.0, u64::MAX as f64)),
        DType::Bool => Some((0.0, 1.0)),
        DType::F32 | DType::BF16 | DType::F16 => None,
    }
}

struct Frame {
    values: HashMap<SemanticValueId, Value>,
    binders: HashMap<LoopBinderId, Term>,
}

/// One pass over an ordered loop body records reads of state written in the
/// same body, so the body can be traced again with that state carried.
struct LoopScope {
    /// Storage entry counts when the iteration started.
    marks: HashMap<usize, usize>,
    /// Storages read below their mark.
    read_before: Vec<usize>,
}

pub(crate) struct Trace<'a> {
    entry: LogicalEntryView<'a>,
    domain: Domain,
    storages: Vec<Storage>,
    next_binder: u32,
    open: Vec<Coordinate>,
    guards: Vec<Term>,
    scopes: Vec<LoopScope>,
    conversions: Vec<Conversion>,
    /// Integer entry parameters with a stored value converted to floating.
    pub(crate) converted: Vec<String>,
    pub(crate) reductions: Vec<ReductionRecord>,
    pub(crate) decodes: Vec<DecodeRecord>,
}

impl<'a> Trace<'a> {
    pub(crate) fn new(entry: LogicalEntryView<'a>) -> Self {
        let mut trace = Trace {
            entry,
            // Stated below: the domain's predicate is a term of this trace.
            domain: Domain::new(&Term::constant(1.0), []),
            storages: Vec::new(),
            next_binder: 0,
            open: Vec::new(),
            guards: Vec::new(),
            scopes: Vec::new(),
            conversions: Vec::new(),
            converted: Vec::new(),
            reductions: Vec::new(),
            decodes: Vec::new(),
        };
        let predicate = trace.static_term(AnyExpr::Bool(entry.domain().predicate().node()));
        let dimensions = entry
            .schema()
            .dimensions()
            .iter()
            .map(|dimension| Term::variable(Variable::Dimension(dimension.name.clone())));
        trace.domain = Domain::new(&predicate, dimensions);
        trace
    }

    fn fresh(&mut self) -> u32 {
        self.next_binder += 1;
        self.next_binder
    }

    fn reference(&self, family: FamilyId) -> &'a SemanticFunction {
        let program = self.entry.program();
        program.function(program.family(family).reference().function())
    }

    /// Trace the entry's root function.
    pub(crate) fn run_entry(&mut self) {
        let schema = self.entry.schema();
        let program = self.entry.program();
        let root = program.function(program.family(program.root()).reference().function());
        let mut arguments = HashMap::new();
        for parameter in schema.parameters() {
            let value = match &parameter.kind {
                ParameterKind::Tensor {
                    representation,
                    axes,
                    ..
                } => {
                    let shape = axes
                        .iter()
                        .map(|axis| self.static_term(AnyExpr::Nat(*axis)))
                        .collect();
                    self.new_storage(
                        Root::Parameter {
                            name: parameter.name.clone(),
                            representation: *representation,
                        },
                        shape,
                    )
                }
                ParameterKind::Scalar { .. } | ParameterKind::Index { .. } => {
                    Value::Scalar(Term::variable(Variable::Parameter(parameter.name.clone())))
                }
                ParameterKind::Range { .. } => Value::Range(
                    Term::variable(Variable::Parameter(format!("{}.start", parameter.name))),
                    Term::variable(Variable::Parameter(format!("{}.end", parameter.name))),
                ),
            };
            arguments.insert(parameter.value, value);
        }
        let mut frame = Frame {
            values: HashMap::new(),
            binders: HashMap::new(),
        };
        for parameter in root.parameters() {
            let value = match arguments.remove(&parameter.value) {
                Some(value) => value,
                None => Value::Scalar(Term::variable(Variable::Dimension(parameter.name.clone()))),
            };
            frame.values.insert(parameter.value, value);
        }
        self.region(root, root.root(), &mut frame);
    }

    /// Trace `function` with `arguments` bound to its parameters in order.
    fn call(&mut self, function: &'a SemanticFunction, arguments: Vec<Value>) -> Vec<Value> {
        let mut frame = Frame {
            values: HashMap::new(),
            binders: HashMap::new(),
        };
        for (parameter, value) in function.parameters().iter().zip(arguments) {
            frame.values.insert(parameter.value, value);
        }
        self.region(function, function.root(), &mut frame);
        function
            .results()
            .iter()
            .map(|result| frame.values[result].clone())
            .collect()
    }

    /// A term for an entry axis extent.
    pub(crate) fn nat_term(&mut self, axis: crate::expr::NatExpr) -> Term {
        self.static_term(AnyExpr::Nat(axis))
    }

    /// A term for an arena expression that reads only invocation symbols.
    fn static_term(&mut self, expression: AnyExpr) -> Term {
        let frame = Frame {
            values: HashMap::new(),
            binders: HashMap::new(),
        };
        self.expr_term(&frame, expression)
    }

    fn expr_term(&mut self, frame: &Frame, expression: AnyExpr) -> Term {
        let arena = self.entry.arena();
        let schema = self.entry.schema();
        match arena.view(expression) {
            NodeView::NatConst(value) => Term::constant(value as f64),
            NodeView::IntConst(value) => Term::constant(value as f64),
            NodeView::BoolConst(value) => Term::constant(f64::from(u8::from(value))),
            NodeView::ScalarConst { dtype, bits } => Term::constant(
                crate::reference_math::ReferenceScalar::from_bits(dtype, bits).to_f64(),
            ),
            NodeView::Symbol(symbol) => match arena.symbol_kind(symbol) {
                SymbolKind::CallDimension(_) => schema
                    .dimensions()
                    .iter()
                    .find(|dimension| dimension.symbol == symbol)
                    .map(|dimension| Term::variable(Variable::Dimension(dimension.name.clone())))
                    .unwrap_or_else(|| Term::opaque(Vec::new())),
                SymbolKind::CallScalar(argument) => {
                    let parameter = schema.parameter(argument.parameter);
                    let name = match argument.component {
                        crate::expr::ScalarComponent::Value => parameter.name.clone(),
                        crate::expr::ScalarComponent::RangeStart => {
                            format!("{}.start", parameter.name)
                        }
                        crate::expr::ScalarComponent::RangeEnd => format!("{}.end", parameter.name),
                    };
                    Term::variable(Variable::Parameter(name))
                }
                SymbolKind::RuntimeValue(value) => match frame.values.get(&value) {
                    Some(Value::Scalar(term)) => term.clone(),
                    _ => Term::opaque(Vec::new()),
                },
                SymbolKind::LoopBinder(binder) => frame
                    .binders
                    .get(&binder)
                    .cloned()
                    .unwrap_or_else(|| Term::opaque(Vec::new())),
                _ => Term::opaque(Vec::new()),
            },
            NodeView::Unary { op, operand } => {
                let operand = self.expr_term(frame, operand);
                match op {
                    ExprUnary::Not => Term::unary(Unary::Not, operand),
                    ExprUnary::NatFromInt | ExprUnary::IntFromNat | ExprUnary::IntFromScalar => {
                        operand
                    }
                    ExprUnary::ScalarIntegerDefined => Term::constant(1.0),
                }
            }
            NodeView::Binary { op, lhs, rhs } => {
                let (a, b) = (self.expr_term(frame, lhs), self.expr_term(frame, rhs));
                match op {
                    ExprBinary::Add => a.add(b),
                    ExprBinary::Sub => a.sub(b),
                    ExprBinary::Mul => a.mul(b),
                    ExprBinary::Div => Term::binary(Binary::Quotient, a, b),
                    ExprBinary::CeilDiv => Term::binary(
                        Binary::Quotient,
                        a.add(b.clone()).sub(Term::constant(1.0)),
                        b,
                    ),
                    ExprBinary::Rem => Term::binary(Binary::Remainder, a, b),
                    ExprBinary::Min => Term::binary(Binary::Min, a, b),
                    ExprBinary::Max => Term::binary(Binary::Max, a, b),
                    ExprBinary::AlignUp => Term::binary(
                        Binary::Quotient,
                        a.add(b.clone()).sub(Term::constant(1.0)),
                        b.clone(),
                    )
                    .mul(b),
                    ExprBinary::And => Term::binary(Binary::And, a, b),
                    ExprBinary::Or => Term::binary(Binary::Or, a, b),
                    ExprBinary::Implies => Term::binary(Binary::Or, Term::unary(Unary::Not, a), b),
                    ExprBinary::Iff => Term::binary(Binary::Eq, a, b),
                }
            }
            NodeView::Nary { op, operands } => {
                let operands = operands
                    .iter()
                    .map(|operand| self.expr_term(frame, *operand))
                    .collect::<Vec<_>>();
                let (identity, combine) = match op {
                    NaryOp::All => (1.0, Binary::And),
                    NaryOp::Any => (0.0, Binary::Or),
                    NaryOp::Product => (1.0, Binary::Mul),
                    NaryOp::DurationAdd => return Term::opaque(Vec::new()),
                };
                operands
                    .into_iter()
                    .fold(Term::constant(identity), |acc, item| Term::binary(combine, acc, item))
            }
            NodeView::Select {
                cond,
                then,
                otherwise,
            } => {
                let condition = self.expr_term(frame, AnyExpr::Bool(cond));
                let then = self.expr_term(frame, then);
                let otherwise = self.expr_term(frame, otherwise);
                Term::select(condition, then, otherwise)
            }
            NodeView::Cmp { op, lhs, rhs } => {
                let (a, b) = (self.expr_term(frame, lhs), self.expr_term(frame, rhs));
                let op = match op {
                    CmpOp::Eq => Binary::Eq,
                    CmpOp::Ne => Binary::Ne,
                    CmpOp::Lt => Binary::Lt,
                    CmpOp::Le => Binary::Le,
                    CmpOp::Gt => Binary::Gt,
                    CmpOp::Ge => Binary::Ge,
                };
                Term::binary(op, a, b)
            }
            NodeView::In { operand, values } => {
                let operand = self.expr_term(frame, operand);
                values.iter().fold(Term::constant(0.0), |acc, value| {
                    Term::binary(
                        Binary::Or,
                        acc,
                        Term::binary(Binary::Eq, operand.clone(), Term::constant(*value as f64)),
                    )
                })
            }
            NodeView::Fold {
                op,
                binder,
                start,
                extent,
                body,
            } => {
                let coordinate = self.fresh();
                let start = self.expr_term(frame, AnyExpr::Nat(start));
                let extent = self.expr_term(frame, AnyExpr::Nat(extent));
                let inner = Frame {
                    values: frame.values.clone(),
                    binders: {
                        let mut binders = frame.binders.clone();
                        binders.insert(binder, Term::variable(Variable::Binder(coordinate)));
                        binders
                    },
                };
                let body = self.expr_term(&inner, AnyExpr::Nat(body));
                let op = match op {
                    crate::expr::FoldOp::Sum => Fold::Sum,
                    crate::expr::FoldOp::Max => Fold::Max,
                    crate::expr::FoldOp::Product => return Term::opaque(body.variables()),
                };
                Term::fold(op, coordinate, start.clone(), start.add(extent), body)
            }
            NodeView::ScalarInteger { .. }
            | NodeView::Duration(_)
            | NodeView::DurationScale { .. } => Term::opaque(Vec::new()),
        }
    }

    fn scalar_ref(&mut self, frame: &Frame, value: &ScalarRef) -> Term {
        match value {
            ScalarRef::Static(expression) => self.expr_term(frame, AnyExpr::Nat(*expression)),
            ScalarRef::Value(id) => scalar(&frame.values[id]),
        }
    }

    fn shape_of(&mut self, frame: &Frame, ty: &SemanticType) -> Vec<Term> {
        match ty {
            SemanticType::Tensor(tensor) => tensor
                .axes
                .iter()
                .map(|axis| self.expr_term(frame, AnyExpr::Nat(*axis)))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn region(&mut self, function: &'a SemanticFunction, region: RegionId, frame: &mut Frame) {
        for (_, node) in function.nodes(region) {
            crate::cancellation::check();
            self.node(function, node.view(), frame);
        }
    }

    fn node(&mut self, function: &'a SemanticFunction, view: SemanticNodeView<'_>, frame: &mut Frame) {
        match view {
            SemanticNodeView::Primitive {
                primitive,
                inputs,
                output,
            } => {
                let value = self.primitive(function, primitive, inputs, output, frame);
                frame.values.insert(output, value);
            }
            SemanticNodeView::Elementwise {
                primitive,
                inputs,
                output,
            } => {
                let dtype = match &function.value(output).ty {
                    SemanticType::Tensor(tensor) => registry::representation_info(tensor.representation).decoded,
                    _ => DType::F32,
                };
                let input_types = inputs
                    .iter()
                    .map(|input| value_dtype(&function.value(*input).ty))
                    .collect();
                let computed = Computed::Elementwise {
                    op: primitive.clone(),
                    inputs: inputs.iter().map(|input| frame.values[input].clone()).collect(),
                    dtype,
                    input_types,
                };
                frame.values.insert(output, Value::Computed(Arc::new(computed)));
            }
            SemanticNodeView::Reduce {
                op,
                axis,
                input,
                output,
                ..
            } => {
                let input_value = frame.values[&input].clone();
                let input_shape = self.shape_of(frame, &function.value(input).ty);
                let extent = input_shape
                    .get(axis as usize)
                    .cloned()
                    .unwrap_or_else(|| Term::opaque(Vec::new()));
                let accumulator = value_dtype(&function.value(input).ty);
                if op == ReduceOp::Sum && accumulator.is_float() {
                    self.record_reduction(extent.clone(), DType::F32, None);
                }
                let computed = Computed::Reduce {
                    op,
                    axis,
                    input: input_value,
                    extent,
                };
                let value = if matches!(function.value(output).ty, SemanticType::Tensor(_)) {
                    Value::Computed(Arc::new(computed))
                } else {
                    Value::Scalar(self.read_computed(&computed, &[]))
                };
                frame.values.insert(output, value);
            }
            SemanticNodeView::Call {
                family,
                inputs,
                outputs,
            } => {
                let callee = self.reference(family);
                let arguments = inputs.iter().map(|input| frame.values[input].clone()).collect();
                let results = self.call(callee, arguments);
                for (output, value) in outputs.iter().zip(results) {
                    frame.values.insert(*output, value);
                }
            }
            SemanticNodeView::Alloc { output, .. } => {
                let shape = self.shape_of(frame, &function.value(output).ty);
                let value = self.new_storage(Root::Undefined, shape);
                frame.values.insert(output, value);
            }
            SemanticNodeView::Fill { value, like, output } => {
                let shape = self.shape_of(frame, &function.value(output).ty);
                let _ = like;
                let value = self.new_storage(Root::Fill(Term::constant(value.value())), shape);
                frame.values.insert(output, value);
            }
            SemanticNodeView::Copy { input, output }
            | SemanticNodeView::RepresentationConvert { input, output, .. } => {
                let shape = self.shape_of(frame, &function.value(output).ty);
                let source = frame.values[&input].clone();
                let value = self.new_storage(Root::Copy(Box::new(source)), shape);
                frame.values.insert(output, value);
            }
            SemanticNodeView::View {
                base,
                extents,
                transform,
                output,
            } => {
                let value = self.view(function, frame, base, extents, transform, output);
                frame.values.insert(output, value);
            }
            SemanticNodeView::ElementRead {
                place,
                indices,
                output,
            } => {
                let index = indices.iter().map(|index| scalar(&frame.values[index])).collect::<Vec<_>>();
                let place = frame.values[&place].clone();
                let value = self.read(&place, &index);
                frame.values.insert(output, Value::Scalar(value));
            }
            SemanticNodeView::ElementWrite {
                place,
                indices,
                value,
                output,
            } => {
                let index = indices.iter().map(|index| scalar(&frame.values[index])).collect::<Vec<_>>();
                let place_value = frame.values[&place].clone();
                let value = scalar(&frame.values[&value]);
                if let Value::Tensor(tensor) = &place_value {
                    self.write(tensor, index, value);
                }
                frame.values.insert(output, place_value);
            }
            SemanticNodeView::Store {
                destination,
                value,
                output,
            } => {
                let destination_value = frame.values[&destination].clone();
                let source = frame.values[&value].clone();
                if let Value::Tensor(tensor) = &destination_value {
                    self.store(tensor, &source);
                }
                let base = match &function.value(destination).ty {
                    SemanticType::Tensor(tensor) => match tensor.storage {
                        TensorStorage::View { base, .. } => frame.values.get(&base).cloned(),
                        _ => None,
                    },
                    _ => None,
                };
                frame.values.insert(output, base.unwrap_or(destination_value));
            }
            SemanticNodeView::Atomic {
                op,
                place,
                arguments,
                output,
            } => {
                let place_value = frame.values[&place].clone();
                if let (Value::Tensor(tensor), Some((value, indices))) =
                    (&place_value, arguments.split_last())
                {
                    let index = indices.iter().map(|index| scalar(&frame.values[index])).collect::<Vec<_>>();
                    let value = scalar(&frame.values[value]);
                    let current = self.read(&place_value, &index);
                    let next = match op {
                        AtomicOp::Add => current.add(value),
                        AtomicOp::Max => Term::binary(Binary::Max, current, value),
                        AtomicOp::Min => Term::binary(Binary::Min, current, value),
                    };
                    self.write(tensor, index, next);
                }
                frame.values.insert(output, place_value);
            }
            SemanticNodeView::If {
                condition,
                captures,
                outputs,
                then,
                otherwise,
            } => {
                let condition = scalar(&frame.values[&condition]);
                let captured = captures.iter().map(|capture| frame.values[capture].clone()).collect::<Vec<_>>();
                if let Some(taken) = self.domain.decide(&condition) {
                    let region = if taken { then } else { otherwise };
                    let results = self.branch(function, region, &captured, frame, None);
                    for (output, value) in outputs.iter().zip(results) {
                        frame.values.insert(*output, value);
                    }
                    return;
                }
                let then_results =
                    self.branch(function, then, &captured, frame, Some(condition.clone()));
                let else_results = self.branch(
                    function,
                    otherwise,
                    &captured,
                    frame,
                    Some(Term::unary(Unary::Not, condition.clone())),
                );
                for ((output, then), otherwise) in outputs.iter().zip(then_results).zip(else_results) {
                    frame.values.insert(*output, merge(&condition, then, otherwise));
                }
            }
            SemanticNodeView::Loop {
                kind,
                start,
                end,
                captures,
                outputs,
                body,
                carries,
            } => {
                let start = scalar(&frame.values[&start]);
                let end = scalar(&frame.values[&end]);
                self.lower_loop(function, frame, kind, start, end, captures, outputs, body, carries);
            }
            SemanticNodeView::Check { .. } => {}
            SemanticNodeView::TuplePack { inputs, output } => {
                let items = inputs.iter().map(|input| frame.values[input].clone()).collect();
                frame.values.insert(output, Value::Tuple(items));
            }
            SemanticNodeView::TupleGet { tuple, index, output } => {
                let item = match &frame.values[&tuple] {
                    Value::Tuple(items) => items[index as usize].clone(),
                    other => Value::Unknown(value_variables(other)),
                };
                frame.values.insert(output, item);
            }
            SemanticNodeView::Extent { tensor, axis, output } => {
                let extent = match &frame.values[&tensor] {
                    Value::Tensor(found) if (axis as usize) < found.shape.len() => {
                        found.shape[axis as usize].clone()
                    }
                    _ => {
                        let shape = self.shape_of(frame, &function.value(tensor).ty);
                        shape.get(axis as usize).cloned().unwrap_or_else(|| Term::opaque(Vec::new()))
                    }
                };
                frame.values.insert(output, Value::Scalar(extent));
            }
            SemanticNodeView::Intrinsic { inputs, output, .. } => {
                let dependencies = inputs
                    .iter()
                    .flat_map(|input| value_variables(&frame.values[input]))
                    .collect();
                frame.values.insert(output, Value::Unknown(dependencies));
            }
        }
    }

    fn branch(
        &mut self,
        function: &'a SemanticFunction,
        region: RegionId,
        captured: &[Value],
        frame: &Frame,
        guard: Option<Term>,
    ) -> Vec<Value> {
        let mut child = Frame {
            values: frame.values.clone(),
            binders: frame.binders.clone(),
        };
        for (parameter, value) in function.region(region).parameters().iter().zip(captured) {
            child.values.insert(*parameter, value.clone());
        }
        let guards = self.guards.len();
        self.guards.extend(guard);
        self.region(function, region, &mut child);
        self.guards.truncate(guards);
        function
            .region(region)
            .results()
            .iter()
            .map(|result| child.values[result].clone())
            .collect()
    }

    fn new_storage(&mut self, root: Root, shape: Vec<Term>) -> Value {
        self.storages.push(Storage {
            root,
            entries: Vec::new(),
        });
        Value::Tensor(Tensor {
            storage: self.storages.len() - 1,
            steps: Vec::new(),
            shape,
        })
    }

    fn view(
        &mut self,
        function: &'a SemanticFunction,
        frame: &Frame,
        base: SemanticValueId,
        extents: &[SemanticValueId],
        transform: &ViewTransform,
        output: SemanticValueId,
    ) -> Value {
        let Value::Tensor(base_tensor) = frame.values[&base].clone() else {
            return Value::Unknown(value_variables(&frame.values[&base]));
        };
        let shape = self.shape_of(frame, &function.value(output).ty);
        let step = match transform {
            ViewTransform::Identity => {
                return Value::Tensor(Tensor { shape, ..base_tensor });
            }
            ViewTransform::Plane { .. } => {
                let dependencies = match &self.storages[base_tensor.storage].root {
                    Root::Parameter { name, .. } => vec![Variable::Parameter(name.clone())],
                    _ => Vec::new(),
                };
                let value = self.new_storage(
                    Root::Copy(Box::new(Value::Unknown(dependencies))),
                    shape,
                );
                return value;
            }
            ViewTransform::Slice { axes } => Step::Slice(
                axes.iter()
                    .map(|axis| match axis {
                        SliceAxis::Point { value, .. } => Axis::Point(self.scalar_ref(frame, value)),
                        SliceAxis::Range { start, .. } => Axis::Range(
                            start
                                .as_ref()
                                .map(|start| self.scalar_ref(frame, start))
                                .unwrap_or_else(|| Term::constant(0.0)),
                        ),
                        SliceAxis::Full => Axis::Full,
                    })
                    .collect(),
            ),
            ViewTransform::Transpose { permutation } => Step::Transpose(permutation.clone()),
            ViewTransform::Reshape { .. } => Step::Reshape {
                from: base_tensor.shape.clone(),
                to: extents.iter().map(|extent| scalar(&frame.values[extent])).collect(),
            },
        };
        let mut steps = base_tensor.steps.clone();
        steps.push(step);
        Value::Tensor(Tensor {
            storage: base_tensor.storage,
            steps,
            shape,
        })
    }

    /// The storage index of view element `index`.
    fn storage_index(tensor: &Tensor, index: &[Term]) -> Vec<Term> {
        let mut index = index.to_vec();
        for step in tensor.steps.iter().rev() {
            index = match step {
                Step::Slice(axes) => {
                    let mut out = Vec::new();
                    let mut remaining = index.into_iter();
                    for axis in axes {
                        match axis {
                            Axis::Point(value) => out.push(value.clone()),
                            Axis::Range(start) => {
                                let coordinate = remaining.next().unwrap_or_else(|| Term::constant(0.0));
                                out.push(start.clone().add(coordinate));
                            }
                            Axis::Full => {
                                out.push(remaining.next().unwrap_or_else(|| Term::constant(0.0)))
                            }
                        }
                    }
                    out.extend(remaining);
                    out
                }
                Step::Transpose(permutation) => {
                    let mut out = vec![Term::constant(0.0); permutation.len()];
                    for (new_axis, old_axis) in permutation.iter().enumerate() {
                        out[*old_axis as usize] = index[new_axis].clone();
                    }
                    out
                }
                Step::Reshape { from, to } => {
                    let mut flat = Term::constant(0.0);
                    for (coordinate, extent) in index.iter().zip(to) {
                        flat = flat.mul(extent.clone()).add(coordinate.clone());
                    }
                    let mut out = vec![Term::constant(0.0); from.len()];
                    let mut remainder = flat;
                    for axis in (0..from.len()).rev() {
                        if axis == 0 {
                            out[axis] = remainder.clone();
                        } else {
                            out[axis] = Term::binary(Binary::Remainder, remainder.clone(), from[axis].clone());
                            remainder = Term::binary(Binary::Quotient, remainder, from[axis].clone());
                        }
                    }
                    out
                }
            };
        }
        index
    }

    /// Read element `index` of a tensor value.
    pub(crate) fn read(&mut self, value: &Value, index: &[Term]) -> Term {
        match value {
            Value::Tensor(tensor) => {
                let index = Self::storage_index(tensor, index);
                let upto = self.storages[tensor.storage].entries.len();
                self.lookup(tensor.storage, &index, upto)
            }
            Value::Computed(computed) => self.read_computed(computed, index),
            Value::Scalar(term) => term.clone(),
            other => Term::opaque(value_variables(other)),
        }
    }

    fn read_computed(&mut self, computed: &Computed, index: &[Term]) -> Term {
        match computed {
            Computed::Elementwise {
                op,
                inputs,
                dtype,
                input_types,
            } => {
                let operands = inputs.iter().map(|input| self.read(input, index)).collect::<Vec<_>>();
                self.scalar_op(op, operands, input_types, *dtype)
            }
            Computed::Reduce {
                op,
                axis,
                input,
                extent,
            } => {
                let binder = self.fresh();
                let mut inner = index.to_vec();
                inner.insert((*axis as usize).min(inner.len()), Term::variable(Variable::Binder(binder)));
                let body = self.read(input, &inner);
                let fold = match op {
                    ReduceOp::Sum => Fold::Sum,
                    ReduceOp::Max => Fold::Max,
                    ReduceOp::Min => Fold::Min,
                    ReduceOp::Argmax => return Term::opaque(body.variables()),
                };
                Term::fold(fold, binder, Term::constant(0.0), extent.clone(), body)
            }
        }
    }

    fn lookup(&mut self, storage: usize, index: &[Term], upto: usize) -> Term {
        crate::cancellation::check();
        for position in (0..upto).rev() {
            let record = match &self.storages[storage].entries[position] {
                Entry::Carried(id) => {
                    let id = *id;
                    return Term::placeholder(id, index.to_vec());
                }
                Entry::Write(record) => record.clone(),
            };
            for scope in &mut self.scopes {
                if scope.marks.get(&storage).is_some_and(|mark| position < *mark)
                    && !scope.read_before.contains(&storage)
                {
                    scope.read_before.push(storage);
                }
            }
            match matches(&record, index) {
                Matched::No => continue,
                Matched::Yes(value, condition) => {
                    if condition.is_empty() {
                        return value;
                    }
                    let rest = self.lookup(storage, index, position);
                    let condition = condition
                        .into_iter()
                        .reduce(|a, b| Term::binary(Binary::And, a, b))
                        .unwrap_or_else(|| Term::constant(1.0));
                    return Term::select(condition, value, rest);
                }
                Matched::Unknown => {
                    let mut dependencies = record.value.variables();
                    for term in index.iter().chain(&record.index) {
                        dependencies.extend(term.variables());
                    }
                    // Whether the write replaced the element is not
                    // resolved. An entry parameter keeps what it held before
                    // the write as the read's other outcome.
                    if matches!(self.storages[storage].root, Root::Parameter { .. }) {
                        let earlier = self.lookup(storage, index, position);
                        return Term::overwritten(dependencies, earlier);
                    }
                    return Term::opaque(dependencies);
                }
            }
        }
        for scope in &mut self.scopes {
            if scope.marks.contains_key(&storage) && !scope.read_before.contains(&storage) {
                scope.read_before.push(storage);
            }
        }
        match self.storages[storage].root.clone() {
            Root::Parameter { name, .. } => Term::variable(Variable::Element {
                parameter: name,
                index: index.to_vec(),
            }),
            Root::Fill(value) => value,
            Root::Undefined => Term::opaque(Vec::new()),
            Root::Copy(source) => self.read(&source, index),
        }
    }

    /// A component read of a packed tensor value.
    fn read_component(
        &mut self,
        value: &Value,
        component: registry::QuantizedComponent,
        index: &[Term],
    ) -> Term {
        if let Value::Tensor(tensor) = value {
            let storage = &self.storages[tensor.storage];
            if let (Root::Parameter { name, .. }, true) = (&storage.root, storage.entries.is_empty()) {
                return Term::variable(Variable::Component {
                    parameter: name.clone(),
                    component,
                    index: Self::storage_index(tensor, index),
                });
            }
        }
        Term::opaque(value_variables(value))
    }

    fn write(&mut self, tensor: &Tensor, index: Vec<Term>, value: Term) {
        let index = Self::storage_index(tensor, &index);
        self.note_decodes(&value, &[]);
        self.storages[tensor.storage].entries.push(Entry::Write(Record {
            index,
            value,
            guard: self.guards.clone(),
            generated: Vec::new(),
        }));
    }

    /// Slice assignment: every element of the destination view takes the
    /// corresponding element of `source`.
    fn store(&mut self, destination: &Tensor, source: &Value) {
        let coordinates = destination
            .shape
            .iter()
            .map(|extent| Coordinate {
                binder: self.fresh(),
                start: Term::constant(0.0),
                end: extent.clone(),
            })
            .collect::<Vec<_>>();
        let index = coordinates
            .iter()
            .map(|coordinate| Term::variable(Variable::Binder(coordinate.binder)))
            .collect::<Vec<_>>();
        let value = self.read(source, &index);
        self.note_decodes(&value, &coordinates);
        let storage_index = Self::storage_index(destination, &index);
        self.storages[destination.storage].entries.push(Entry::Write(Record {
            index: storage_index,
            value,
            guard: self.guards.clone(),
            generated: coordinates,
        }));
    }

    /// Record the decodes that `value`, about to be written for every
    /// assignment of the `generated` coordinates, makes of stored integer
    /// elements. Parameter contents are taken where no unresolved earlier
    /// write of this call replaced them.
    fn note_decodes(&mut self, value: &Term, generated: &[Coordinate]) {
        if self.conversions.is_empty() {
            return;
        }
        let value = value.unshadowed();
        for position in 0..self.conversions.len() {
            let conversion = &self.conversions[position];
            if !value.mentions(&conversion.term)
                || self.decodes.iter().any(|decode| decode.conversion == conversion.term)
            {
                continue;
            }
            let Some((slope, intercept)) = value.affine_in(&conversion.term) else {
                continue;
            };
            if !slope.is_resolved() || !intercept.is_resolved() {
                continue;
            }
            let conversion = conversion.clone();
            // Innermost first; a closing loop moves its coordinate last.
            let coordinates = generated.iter().rev().chain(self.open.iter().rev()).cloned().collect();
            self.decodes.push(DecodeRecord {
                parameter: conversion.parameter,
                index: conversion.index,
                code: conversion.code,
                range: conversion.range,
                slope,
                intercept,
                coordinates,
                conversion: conversion.term,
            });
        }
    }

    fn record_reduction(&mut self, extent: Term, accumulator: DType, storage: Option<usize>) {
        let guard = self
            .guards
            .iter()
            .cloned()
            .reduce(|a, b| Term::binary(Binary::And, a, b));
        let extent = match guard {
            Some(guard) => Term::select(guard, extent, Term::constant(0.0)),
            None => extent,
        };
        self.reductions.push(ReductionRecord {
            extent,
            accumulator,
            coordinates: self.open.clone(),
            storage,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_loop(
        &mut self,
        function: &'a SemanticFunction,
        frame: &mut Frame,
        kind: LoopKind,
        start: Term,
        end: Term,
        captures: &[SemanticValueId],
        outputs: &[SemanticValueId],
        body: RegionId,
        carries: &[crate::entry::Carry],
    ) {
        let RegionKind::LoopBody {
            expression_binder,
            binder_value,
            ..
        } = function.region(body).kind().clone()
        else {
            return;
        };
        let binder = self.fresh();
        let coordinate = Term::variable(Variable::Binder(binder));
        let parameters = function.region(body).parameters().to_vec();
        let captured = captures.iter().map(|capture| frame.values[capture].clone()).collect::<Vec<_>>();
        let marks: HashMap<usize, usize> = self
            .storages
            .iter()
            .enumerate()
            .map(|(storage, contents)| (storage, contents.entries.len()))
            .collect();
        let reductions_before = self.reductions.len();
        let decodes_before = self.decodes.len();
        let conversions_before = self.conversions.len();
        let snapshot = self.storages.iter().map(|storage| storage.entries.len()).collect::<Vec<_>>();
        let storages_before = self.storages.len();

        // Scalar carries enter as placeholders.
        let scalar_carries = carries
            .iter()
            .filter(|carry| !matches!(function.value(carry.parameter).ty, SemanticType::Tensor(_)))
            .map(|carry| (carry.parameter, self.fresh()))
            .collect::<Vec<_>>();
        let run = |this: &mut Self, carried_storages: &[(usize, u32)]| -> Frame {
            let mut child = Frame {
                values: frame.values.clone(),
                binders: frame.binders.clone(),
            };
            child.binders.insert(expression_binder, coordinate.clone());
            child.values.insert(binder_value, Value::Scalar(coordinate.clone()));
            for (parameter, value) in parameters[1..].iter().zip(&captured) {
                let value = match scalar_carries.iter().find(|(carried, _)| carried == parameter) {
                    Some((_, id)) => Value::Scalar(Term::placeholder(*id, Vec::new())),
                    None => value.clone(),
                };
                child.values.insert(*parameter, value);
            }
            for (storage, id) in carried_storages {
                this.storages[*storage].entries.push(Entry::Carried(*id));
            }
            this.open.push(Coordinate {
                binder,
                start: start.clone(),
                end: end.clone(),
            });
            this.scopes.push(LoopScope {
                marks: this
                    .storages
                    .iter()
                    .enumerate()
                    .map(|(storage, contents)| (storage, contents.entries.len()))
                    .collect(),
                read_before: Vec::new(),
            });
            this.region(function, body, &mut child);
            child
        };
        let restore = |this: &mut Self| {
            this.storages.truncate(storages_before);
            for (storage, length) in snapshot.iter().enumerate() {
                this.storages[storage].entries.truncate(*length);
            }
            this.reductions.truncate(reductions_before);
            this.decodes.truncate(decodes_before);
            this.conversions.truncate(conversions_before);
            this.open.pop();
            this.scopes.pop();
        };

        let mut child = run(self, &[]);
        let scope = self.scopes.last().map(|scope| scope.read_before.clone()).unwrap_or_default();
        let written = (0..storages_before)
            .filter(|storage| self.storages[*storage].entries.len() > marks[storage])
            .collect::<Vec<_>>();
        let carried = if kind == LoopKind::Ordered {
            written
                .iter()
                .copied()
                .filter(|storage| scope.contains(storage))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let carried = carried.into_iter().map(|storage| (storage, self.fresh())).collect::<Vec<_>>();
        if !carried.is_empty() {
            restore(self);
            child = run(self, &carried);
        }
        self.open.pop();
        self.scopes.pop();

        // Close the body's writes over the loop coordinate.
        let empty = Term::binary(Binary::Ge, start.clone(), end.clone());
        let last = end.clone().sub(Term::constant(1.0));
        for storage in 0..self.storages.len() {
            let from = if storage < storages_before { marks[&storage] } else { 0 };
            let carried_id = carried.iter().find(|(found, _)| *found == storage).map(|(_, id)| *id);
            let mut entries = self.storages[storage].entries.split_off(from);
            if carried_id.is_some() {
                entries.retain(|entry| !matches!(entry, Entry::Carried(_)));
            }
            let mut closed = Vec::new();
            for entry in entries {
                let Entry::Write(mut record) = entry else {
                    closed.push(entry);
                    continue;
                };
                let carries_state = carried_id.is_some_and(|id| record.value.carries(id));
                if carries_state {
                    let id = carried_id.expect("carried storage");
                    let before = self.lookup(storage, &record.index, from);
                    record.value = self.accumulate(
                        &record.value,
                        id,
                        &record.index,
                        before,
                        binder,
                        &start,
                        &end,
                    );
                    closed.push(Entry::Write(record));
                    continue;
                }
                if storage >= storages_before {
                    // Storage created inside the body: one per iteration.
                    if record.index.iter().any(|term| term.mentions(&coordinate))
                        || record.value.mentions(&coordinate)
                    {
                        record.generated.push(Coordinate {
                            binder,
                            start: start.clone(),
                            end: end.clone(),
                        });
                    }
                    closed.push(Entry::Write(record));
                    continue;
                }
                if record.index.iter().any(|term| term.mentions(&coordinate)) {
                    record.generated.push(Coordinate {
                        binder,
                        start: start.clone(),
                        end: end.clone(),
                    });
                } else {
                    // Every iteration writes the same element: the last
                    // iteration's write survives a nonempty loop.
                    record.value = record.value.bind(binder, &last);
                    record.guard.push(Term::unary(Unary::Not, empty.clone()));
                }
                closed.push(Entry::Write(record));
            }
            self.storages[storage].entries.extend(closed);
        }

        // Merge accumulations into carried state made by inner loops.
        for (storage, _) in &carried {
            let inner = self.reductions[reductions_before..]
                .iter()
                .filter(|reduction| reduction.storage == Some(*storage))
                .map(|reduction| reduction.extent.clone())
                .collect::<Vec<_>>();
            let accumulator = self.reductions[reductions_before..]
                .iter()
                .find(|reduction| reduction.storage == Some(*storage))
                .map(|reduction| reduction.accumulator);
            let mut index = reductions_before;
            while index < self.reductions.len() {
                if self.reductions[index].storage == Some(*storage) {
                    self.reductions.remove(index);
                } else {
                    index += 1;
                }
            }
            let per_iteration = if inner.is_empty() {
                Term::constant(1.0)
            } else {
                inner.into_iter().reduce(Term::add).expect("nonempty")
            };
            let extent = Term::fold(Fold::Sum, binder, start.clone(), end.clone(), per_iteration);
            self.record_reduction(extent, accumulator.unwrap_or(DType::F32), Some(*storage));
        }
        for reduction in &mut self.reductions[reductions_before..] {
            reduction.coordinates.retain(|open| open.binder != binder);
        }
        for decode in &mut self.decodes[decodes_before..] {
            decode.coordinates.retain(|open| open.binder != binder);
            decode.coordinates.push(Coordinate {
                binder,
                start: start.clone(),
                end: end.clone(),
            });
        }

        // Carried scalars.
        for (output, carry) in outputs.iter().zip(carries) {
            let output = *output;
            let initial = frame.values[&carry.initial].clone();
            let value = match scalar_carries.iter().find(|(parameter, _)| *parameter == carry.parameter) {
                Some((_, id)) => {
                    let yielded = scalar(&child.values[&carry.yielded]);
                    let initial = scalar(&initial);
                    let dtype = value_dtype(&function.value(carry.parameter).ty);
                    let resolved = if yielded.carries(*id) {
                        let resolved = self.accumulate_scalar(&yielded, *id, initial.clone(), binder, &start, &end);
                        if dtype.is_float() {
                            self.record_reduction(end.clone().sub(start.clone()), dtype, None);
                        }
                        resolved
                    } else {
                        Term::select(empty.clone(), initial, yielded.bind(binder, &last))
                    };
                    Value::Scalar(resolved)
                }
                None => child.values.get(&carry.yielded).cloned().unwrap_or(initial),
            };
            frame.values.insert(output, value);
        }
    }

    /// The value after the loop of a scalar carried with placeholder `id`
    /// whose next value is `yielded`.
    fn accumulate_scalar(
        &mut self,
        yielded: &Term,
        id: u32,
        initial: Term,
        binder: u32,
        start: &Term,
        end: &Term,
    ) -> Term {
        let placeholder = Term::placeholder(id, Vec::new());
        self.close_accumulation(yielded, &placeholder, initial, binder, start, end)
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &mut self,
        value: &Term,
        id: u32,
        index: &[Term],
        before: Term,
        binder: u32,
        start: &Term,
        end: &Term,
    ) -> Term {
        let placeholder = Term::placeholder(id, index.to_vec());
        self.close_accumulation(value, &placeholder, before, binder, start, end)
    }

    /// Resolve `value`, a function of `placeholder`, as the result of
    /// iterating `binder` over `start..end` from `initial`.
    fn close_accumulation(
        &mut self,
        value: &Term,
        placeholder: &Term,
        initial: Term,
        binder: u32,
        start: &Term,
        end: &Term,
    ) -> Term {
        let others = |term: &Term| {
            term.any(&|sub| matches!(sub.node(), Node::Placeholder(..)) && sub != placeholder)
        };
        // acc + g
        if let Some(affine) = value.affine() {
            if affine.coefficient(placeholder) == 1.0 {
                let increment = affine.without(placeholder);
                if !increment.mentions(placeholder) && !others(&increment) {
                    return initial.add(Term::fold(Fold::Sum, binder, start.clone(), end.clone(), increment));
                }
            }
        }
        // max(acc, g) / min(acc, g)
        if let Node::Binary(op @ (Binary::Max | Binary::Min), a, b) = value.node() {
            let increment = if a == placeholder { Some(b) } else if b == placeholder { Some(a) } else { None };
            if let Some(increment) = increment {
                if !increment.mentions(placeholder) && !others(increment) {
                    let fold = if *op == Binary::Max { Fold::Max } else { Fold::Min };
                    return Term::binary(
                        *op,
                        initial,
                        Term::fold(fold, binder, start.clone(), end.clone(), increment.clone()),
                    );
                }
            }
        }
        let mut dependencies = initial.variables();
        dependencies.extend(value.variables());
        dependencies.retain(|variable| !matches!(variable, Variable::Binder(found) if *found == binder));
        Term::opaque(dependencies)
    }

    fn primitive(
        &mut self,
        function: &'a SemanticFunction,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
        frame: &Frame,
    ) -> Value {
        let output_dtype = value_dtype(&function.value(output).ty);
        match primitive {
            PrimitiveId::Constant(constant) => Value::Scalar(Term::constant(constant.to_f64())),
            PrimitiveId::Symbolic(expression) => {
                Value::Scalar(self.expr_term(frame, AnyExpr::Int(*expression)))
            }
            PrimitiveId::RangeMake => Value::Range(
                scalar(&frame.values[&inputs[0]]),
                scalar(&frame.values[&inputs[1]]),
            ),
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => match &frame.values[&inputs[0]] {
                Value::Range(start, end) => Value::Scalar(if matches!(primitive, PrimitiveId::RangeStart) {
                    start.clone()
                } else {
                    end.clone()
                }),
                other => Value::Scalar(Term::opaque(value_variables(other))),
            },
            PrimitiveId::QuantizedComponentRead { component, .. } => {
                let tensor = frame.values[&inputs[0]].clone();
                let index = inputs[1..].iter().map(|input| scalar(&frame.values[input])).collect::<Vec<_>>();
                Value::Scalar(self.read_component(&tensor, *component, &index))
            }
            PrimitiveId::Cast(dtype) => {
                let input = scalar(&frame.values[&inputs[0]]);
                let from = value_dtype(&function.value(inputs[0]).ty);
                Value::Scalar(self.cast(input, from, *dtype))
            }
            _ => {
                let operands = inputs.iter().map(|input| scalar(&frame.values[input])).collect::<Vec<_>>();
                let input_types = inputs
                    .iter()
                    .map(|input| value_dtype(&function.value(*input).ty))
                    .collect::<Vec<_>>();
                Value::Scalar(self.scalar_op(primitive, operands, &input_types, output_dtype))
            }
        }
    }

    /// The dtype of the integer elements entry parameter `name` stores.
    fn integer_elements(&self, name: &str) -> Option<DType> {
        self.storages.iter().find_map(|storage| match &storage.root {
            Root::Parameter {
                name: found,
                representation,
            } if found == name => match registry::representation_info(*representation).kind {
                RepresentationKind::Dense(dtype) if dtype.is_int() => Some(dtype),
                _ => None,
            },
            _ => None,
        })
    }

    /// Record the conversion of integer `input` (of dtype `from`) to floating
    /// `to`: the integer parameters whose stored values it converts, and the
    /// conversion itself when its value is computed from exactly one stored
    /// integer element and nothing else stored.
    fn note_conversion(&mut self, input: &Term, from: DType, to: DType) {
        let code = input.unshadowed();
        let reads = code.value_reads();
        // An unresolved value may be computed from anything it reads.
        let sources = reads.clone().unwrap_or_else(|| code.variables());
        for source in &sources {
            if let Variable::Element { parameter, .. } = source {
                if self.integer_elements(parameter).is_some() && !self.converted.contains(parameter) {
                    self.converted.push(parameter.clone());
                }
            }
        }
        let Some([Variable::Element { parameter, index }]) = reads.as_deref() else {
            return;
        };
        if self.integer_elements(parameter).is_none() {
            return;
        }
        let term = Term::round(to, code.clone());
        if self.conversions.iter().any(|known| known.term == term) {
            return;
        }
        let stored = |variable: &Variable| match variable {
            Variable::Element { parameter, .. } => {
                self.integer_elements(parameter).and_then(word_range)
            }
            _ => None,
        };
        let word = word_range(from).expect("an integer word has a range");
        let range = match code.bounds(&stored) {
            Some((low, high)) => (low.max(word.0), high.min(word.1)),
            None => word,
        };
        self.conversions.push(Conversion {
            term,
            code,
            range,
            parameter: parameter.clone(),
            index: index.clone(),
        });
    }

    fn cast(&mut self, input: Term, from: DType, to: DType) -> Term {
        if from.is_float() && to.is_float() {
            return if matches!(to, DType::F32) { input } else { Term::round(to, input) };
        }
        if !from.is_float() && to.is_float() {
            self.note_conversion(&input, from, to);
            return Term::round(to, input);
        }
        if from.is_float() && !to.is_float() {
            return Term::truncate(input);
        }
        input
    }

    fn scalar_op(&mut self, primitive: &PrimitiveId, operands: Vec<Term>, input_types: &[DType], output: DType) -> Term {
        let integer = input_types.first().is_some_and(|dtype| !dtype.is_float());
        match primitive {
            PrimitiveId::Unary(op) => {
                let a = operands[0].clone();
                match op {
                    UnaryOp::Neg => Term::unary(Unary::Neg, a),
                    UnaryOp::Not => Term::unary(Unary::Not, a),
                    UnaryOp::BitNot => Term::opaque(a.variables()),
                }
            }
            PrimitiveId::Binary(op) => {
                let (a, b) = (operands[0].clone(), operands[1].clone());
                let op = match op {
                    BinaryOp::Add => Binary::Add,
                    BinaryOp::Sub => Binary::Sub,
                    BinaryOp::Mul => Binary::Mul,
                    BinaryOp::Div if integer => Binary::Quotient,
                    BinaryOp::Div => Binary::Div,
                    BinaryOp::Rem => Binary::Remainder,
                    BinaryOp::Shl => Binary::Shl,
                    BinaryOp::Shr => Binary::Shr,
                    BinaryOp::BitAnd => Binary::BitAnd,
                    BinaryOp::BitOr => Binary::BitOr,
                    BinaryOp::BitXor => Binary::BitXor,
                    BinaryOp::Eq => Binary::Eq,
                    BinaryOp::Ne => Binary::Ne,
                    BinaryOp::Lt => Binary::Lt,
                    BinaryOp::Le => Binary::Le,
                    BinaryOp::Gt => Binary::Gt,
                    BinaryOp::Ge => Binary::Ge,
                    BinaryOp::And => Binary::And,
                    BinaryOp::Or => Binary::Or,
                };
                let value = Term::binary(op, a, b);
                if output.is_float() && !matches!(output, DType::F32) {
                    Term::round(output, value)
                } else {
                    value
                }
            }
            PrimitiveId::Math(op) => {
                let a = operands[0].clone();
                match op {
                    MathOp::Fma => a.mul(operands[1].clone()).add(operands[2].clone()),
                    MathOp::Rint => Term::unary(Unary::Rint, a),
                    MathOp::Exp => Term::unary(Unary::Exp, a),
                    MathOp::Exp2 => Term::unary(Unary::Exp2, a),
                    MathOp::Rsqrt => Term::unary(Unary::Rsqrt, a),
                    MathOp::Sqrt => Term::unary(Unary::Sqrt, a),
                    MathOp::Log => Term::unary(Unary::Log, a),
                    MathOp::Sin => Term::unary(Unary::Sin, a),
                    MathOp::Cos => Term::unary(Unary::Cos, a),
                    MathOp::Abs => Term::unary(Unary::Abs, a),
                    MathOp::Max => Term::binary(Binary::Max, a, operands[1].clone()),
                    MathOp::Min => Term::binary(Binary::Min, a, operands[1].clone()),
                }
            }
            PrimitiveId::Select => Term::select(operands[0].clone(), operands[1].clone(), operands[2].clone()),
            PrimitiveId::Cast(dtype) => {
                let from = input_types.first().copied().unwrap_or(DType::F32);
                self.cast(operands[0].clone(), from, *dtype)
            }
            _ => Term::opaque(operands.iter().flat_map(Term::variables).collect()),
        }
    }
}

/// How a write record matches a read of `index`.
enum Matched {
    No,
    /// The value, under these extra conditions.
    Yes(Term, Vec<Term>),
    Unknown,
}

fn matches(record: &Record, index: &[Term]) -> Matched {
    if record.index.len() != index.len() {
        return Matched::Unknown;
    }
    // Solve the generated coordinates.
    let mut solution: Vec<(u32, Term)> = Vec::new();
    for coordinate in &record.generated {
        let variable = Term::variable(Variable::Binder(coordinate.binder));
        let mut solved = None;
        for (written, read) in record.index.iter().zip(index) {
            let Some(affine) = written.affine() else { continue };
            let coefficient = affine.coefficient(&variable);
            if coefficient == 0.0 {
                continue;
            }
            let rest = affine.without(&variable);
            let others = record
                .generated
                .iter()
                .any(|other| other.binder != coordinate.binder && rest.mentions(&Term::variable(Variable::Binder(other.binder))));
            if others || coefficient.abs() != 1.0 {
                continue;
            }
            let value = read.clone().sub(rest).mul(Term::constant(coefficient));
            solved = Some(value);
            break;
        }
        match solved {
            Some(value) => solution.push((coordinate.binder, value)),
            None => return Matched::Unknown,
        }
    }
    let substitute = |term: &Term| {
        solution
            .iter()
            .fold(term.clone(), |term, (binder, value)| term.bind(*binder, value))
    };
    // Guards use the same generated coordinates as the write's value and
    // indices. Close them under the solved read coordinates as well.
    let mut condition = record.guard.iter().map(substitute).collect::<Vec<_>>();
    for (written, read) in record.index.iter().zip(index) {
        let written = substitute(written);
        match written.compare(read) {
            Some(true) => {}
            Some(false) => return Matched::No,
            None => condition.push(Term::binary(Binary::Eq, written, read.clone())),
        }
    }
    for coordinate in &record.generated {
        let value = solution
            .iter()
            .find(|(binder, _)| *binder == coordinate.binder)
            .map(|(_, value)| value.clone())
            .expect("solved coordinate");
        let start = substitute(&coordinate.start);
        let end = substitute(&coordinate.end);
        if value.compare(&start) != Some(true) || start.compare(&end).is_none() {
            condition.push(Term::binary(Binary::Ge, value.clone(), start));
        }
        condition.push(Term::binary(Binary::Lt, value, end));
    }
    let condition = condition
        .into_iter()
        .filter(|term| term.as_constant() != Some(1.0))
        .collect::<Vec<_>>();
    if condition.iter().any(|term| term.as_constant() == Some(0.0)) {
        return Matched::No;
    }
    Matched::Yes(substitute(&record.value), condition)
}

fn merge(condition: &Term, then: Value, otherwise: Value) -> Value {
    match (then, otherwise) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(Term::select(condition.clone(), a, b)),
        (Value::Tensor(a), Value::Tensor(b)) if a.storage == b.storage => Value::Tensor(a),
        (Value::Tuple(a), Value::Tuple(b)) if a.len() == b.len() => Value::Tuple(
            a.into_iter()
                .zip(b)
                .map(|(a, b)| merge(condition, a, b))
                .collect(),
        ),
        (Value::Range(a0, a1), Value::Range(b0, b1)) => Value::Range(
            Term::select(condition.clone(), a0, b0),
            Term::select(condition.clone(), a1, b1),
        ),
        (a, b) => {
            let mut dependencies = value_variables(&a);
            dependencies.extend(value_variables(&b));
            dependencies.extend(condition.variables());
            Value::Unknown(dependencies)
        }
    }
}

pub(crate) fn scalar(value: &Value) -> Term {
    match value {
        Value::Scalar(term) => term.clone(),
        other => Term::opaque(value_variables(other)),
    }
}

pub(crate) fn value_variables(value: &Value) -> Vec<Variable> {
    match value {
        Value::Scalar(term) => term.variables(),
        Value::Tuple(items) => items.iter().flat_map(value_variables).collect(),
        Value::Range(a, b) => {
            let mut variables = a.variables();
            variables.extend(b.variables());
            variables
        }
        Value::Unknown(dependencies) => dependencies.clone(),
        Value::Tensor(_) | Value::Computed(_) => Vec::new(),
    }
}

pub(crate) fn value_dtype(ty: &SemanticType) -> DType {
    match ty {
        SemanticType::Scalar(dtype) => *dtype,
        SemanticType::Tensor(tensor) => registry::representation_info(tensor.representation).decoded,
        _ => DType::I32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_loop_writes_preserve_the_old_value_at_unwritten_indices() {
        let entry = super::super::tests::entry(
            "fn probe[N](x: &tensor[N] f32, out: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        if x[i] > 0.0:\n            out[i] = x[i] * x[i]\n",
        );
        let mut trace = Trace::new(entry.as_view());
        trace.run_entry();
        let storage = trace.storages.iter().position(|storage| {
            matches!(&storage.root, Root::Parameter { name, .. } if name == "out")
        }).unwrap();
        let term = trace.lookup(storage, &[Term::constant(2.0)], trace.storages[storage].entries.len());
        for input in [-3.0, 0.0, 3.0] {
            let actual = term.evaluate(&|variable| match variable {
                Variable::Dimension(name) if name == "N" => Some(4.0),
                Variable::Element { parameter, index } if index == &[Term::constant(2.0)] => {
                    match parameter.as_str() { "x" => Some(input), "out" => Some(7.0), _ => None }
                }
                _ => None,
            });
            assert_eq!(actual, Ok(if input > 0.0 { input * input } else { 7.0 }), "{term}");
        }
    }

    #[test]
    fn conditional_initialization_does_not_create_a_value_on_the_other_path() {
        let entry = super::super::tests::entry("fn probe(x: f32) -> f32:\n    return x\n");
        let mut trace = Trace::new(entry.as_view());
        let value = trace.new_storage(Root::Undefined, vec![Term::constant(1.0)]);
        let guard = Variable::Parameter("guard".into());
        trace.guards.push(Term::variable(guard.clone()));
        let Value::Tensor(tensor) = &value else { panic!("expected tensor") };
        trace.write(tensor, vec![Term::constant(0.0)], Term::constant(5.0));
        trace.guards.pop();
        let term = trace.read(&value, &[Term::constant(0.0)]);
        assert_eq!(term.evaluate(&|variable| (variable == &guard).then_some(1.0)), Ok(5.0));
        assert_eq!(term.evaluate(&|variable| (variable == &guard).then_some(0.0)), Err(super::super::TermError::Unresolved));
    }
}
