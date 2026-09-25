//! Source-order definite initialization. This is a mandatory checking rule,
//! run before a checked definition is published. Regions describe actual
//! logical coordinates; no runtime storage or compiler-side certificate exists.
use super::{ir, prove, xfer, CheckedOutcome, Checker};
use crate::checked::DiagnosticRule;
use crate::expr::{AnyExpr, ExprArena, IntExpr, SymbolId};
pub(super) use crate::initialization::InitializationContract as Contract;
use crate::initialization::RegionMapping;
use crate::initialization::{
    Bound, Condition, Exit, InitializationView, ParameterAccess, ParameterPart, ParameterPath,
    Path, Region, RegionOps, Requirement, VisitSeparation,
};
use crate::intrinsics::{AtomicOp, IndexSlot, PrimitiveId};
use crate::reference_math::ReferenceScalar;
use crate::span::Span;
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::{DType, ValueType};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, PartialEq)]
struct Place {
    root: usize,
    view: InitializationView,
}
impl std::ops::Deref for Place {
    type Target = InitializationView;
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}
impl std::ops::DerefMut for Place {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.view
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
struct Scalar {
    integer: Option<IntExpr>,
    condition: Option<Condition>,
    range: Option<(IntExpr, IntExpr)>,
}
#[derive(Clone, Debug)]
enum Target {
    Binding(super::ownership::LocalPlace),
    Tensor(Place),
    Tuple(Vec<Target>),
}
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Scalar(Scalar),
    Tensor(Place),
    Tuple(Vec<Value>),
    Void,
}
impl Value {
    fn at(&self, path: &[usize]) -> &Self {
        match path.split_first() {
            None => self,
            Some((index, rest)) => match self {
                Self::Tuple(parts) => parts[*index].at(rest),
                _ => panic!("checked value projection is not a tuple"),
            },
        }
    }
    fn at_mut(&mut self, path: &[usize]) -> &mut Self {
        match path.split_first() {
            None => self,
            Some((index, rest)) => match self {
                Self::Tuple(parts) => parts[*index].at_mut(rest),
                _ => panic!("checked value projection is not a tuple"),
            },
        }
    }
}
#[derive(Clone, Debug)]
struct Root {
    name: String,
    axes: Vec<IntExpr>,
    parameter: Option<ParameterPath>,
    written: Region,
    incoming: Region,
}
#[derive(Clone)]
struct World {
    values: HashMap<ir::LocalId, Value>,
    roots: HashMap<usize, Root>,
    path: Path,
    facts: prove::Facts,
    requirements: Vec<Requirement>,
    deferred: Vec<Read>,
    accesses: Vec<Access>,
    defer_depth: usize,
}
#[derive(Clone)]
struct Read {
    root: usize,
    region: Region,
    available: Region,
    path: Path,
    facts: prove::Facts,
    span: Span,
}
#[derive(Clone)]
struct Access {
    root: usize,
    region: Region,
    path: Path,
    span: Span,
    write: bool,
    atomic: Option<AtomicOp>,
}
impl Access {
    /// Two accesses the separation proof cannot tell apart.
    fn same_footprint(&self, other: &Access) -> bool {
        self.root == other.root
            && self.write == other.write
            && self.atomic == other.atomic
            && self.path == other.path
            && self.region == other.region
    }
}
impl World {
    /// Record a may-access once per distinct footprint, so independence
    /// compares each distinct pair once.
    fn record_access(&mut self, access: Access) {
        if !self
            .accesses
            .iter()
            .any(|recorded| recorded.same_footprint(&access))
        {
            self.accesses.push(access);
        }
    }
}
/// The first pair of accesses whose distinct visits may overlap.
struct VisitOverlap {
    root: usize,
    left: Span,
    right: Span,
}
/// Check the body `block` followed by its one exit `result` (L5), and build
/// the definition's initialization contract.
pub(super) fn check(
    checker: &mut Checker<'_>,
    block: &mut ir::Block,
    result: &mut [ir::Expr],
    facts: prove::Facts,
) -> Contract {
    let mut owner = Initialization {
        checker,
        next_root: 0,
        next_version: 0,
        binders: vec![],
        symbols: vec![],
        record_loops: true,
    };
    let mut initial = World {
        values: HashMap::new(),
        roots: HashMap::new(),
        path: vec![],
        facts,
        requirements: vec![],
        deferred: vec![],
        accesses: vec![],
        defer_depth: 0,
    };
    let params = owner.checker.sig.params.clone();
    for (ordinal, parameter) in params.iter().enumerate() {
        let local = ir::LocalId::new(ordinal as u32);
        let value = owner.parameter(
            &mut initial,
            &parameter.ty,
            ParameterPath::root(ordinal),
            local,
        );
        initial.values.insert(local, value);
    }
    let mut worlds = owner.block(vec![initial], block);
    for world in &mut worlds {
        for expression in result.iter_mut() {
            let value = owner.expression(world, expression);
            owner.consume(world, &value, expression.span);
        }
    }
    let mut contract = Contract {
        symbols: owner.symbols.clone(),
        ..Contract::empty()
    };
    for world in worlds {
        contract.requirements.extend(world.requirements);
        contract
            .accesses
            .extend(world.accesses.iter().filter_map(|access| {
                world
                    .roots
                    .get(&access.root)?
                    .parameter
                    .clone()
                    .map(|parameter| ParameterAccess {
                        parameter,
                        region: access.region.clone(),
                        path: access.path.clone(),
                        write: access.write,
                        atomic: access.atomic,
                    })
            }));
        contract.exits.push(Exit {
            path: world.path,
            written: world
                .roots
                .into_values()
                .filter_map(|root| root.parameter.map(|p| (p, root.written)))
                .collect(),
        });
    }
    owner.close_contract(contract)
}

struct Initialization<'a, 'env> {
    checker: &'a mut Checker<'env>,
    next_root: usize,
    next_version: u64,
    binders: Vec<SymbolId>,
    symbols: Vec<(SymbolId, ParameterPart)>,
    record_loops: bool,
}
impl Initialization<'_, '_> {
    fn fresh_condition(&mut self) -> Condition {
        let id = self.next_version;
        self.next_version += 1;
        Condition::Version(id, self.binders.clone())
    }
    fn fresh_place(
        &mut self,
        world: &mut World,
        axes: &[IntExpr],
        parameter: Option<ParameterPath>,
        initialized: bool,
        name: String,
    ) -> Place {
        let root = self.next_root;
        self.next_root += 1;
        world.roots.insert(
            root,
            Root {
                name,
                axes: axes.to_vec(),
                parameter,
                written: if initialized {
                    Region::Full
                } else {
                    Region::Empty
                },
                incoming: Region::Empty,
            },
        );
        Place {
            root,
            view: self.root_view(axes),
        }
    }
    fn result(&mut self, world: &mut World, ty: &ValueType, initialized: bool) -> Value {
        match ty {
            ValueType::Tensor(t) => Value::Tensor(self.fresh_place(
                world,
                &t.axes,
                None,
                initialized,
                "allocation".into(),
            )),
            ValueType::Tuple(items) => Value::Tuple(
                items
                    .iter()
                    .map(|ty| self.result(world, ty, initialized))
                    .collect(),
            ),
            ValueType::Void => Value::Void,
            ValueType::Scalar(DType::Bool) => Value::Scalar(Scalar {
                condition: Some(self.fresh_condition()),
                ..Default::default()
            }),
            ValueType::Scalar(DType::U32) | ValueType::Index { .. } => Value::Scalar(Scalar {
                integer: Some(self.fresh_integer().1),
                ..Default::default()
            }),
            _ => Value::Scalar(Scalar::default()),
        }
    }
    fn parameter(
        &mut self,
        world: &mut World,
        ty: &ValueType,
        path: ParameterPath,
        local: ir::LocalId,
    ) -> Value {
        match ty {
            ValueType::Tuple(parts) => Value::Tuple(
                parts
                    .iter()
                    .enumerate()
                    .map(|(index, ty)| self.parameter(world, ty, path.child(index), local))
                    .collect(),
            ),
            ValueType::Tensor(t) => Value::Tensor(self.fresh_place(
                world,
                &t.axes,
                Some(path.clone()),
                false,
                self.checker.sig.params[path.parameter].name.clone(),
            )),
            ValueType::Index { .. } | ValueType::Scalar(DType::U32) => {
                let (symbol, value) = if let Some(s) = self.checker.locals[local.index()].symbol {
                    (s, self.arena().int_symbol(s))
                } else {
                    self.fresh_integer()
                };
                self.symbols.push((symbol, ParameterPart::Integer(path)));
                Value::Scalar(Scalar {
                    integer: Some(value),
                    ..Default::default()
                })
            }
            ValueType::Range { bound } => {
                let (s, start) = self.fresh_integer();
                let (e, end) = self.fresh_integer();
                self.symbols.extend([
                    (s, ParameterPart::Start(path.clone())),
                    (e, ParameterPart::End(path)),
                ]);
                let zero = self.arena().int(0);
                world.facts.set_range(s, zero, *bound);
                world.facts.set_range(e, start, *bound);
                Value::Scalar(Scalar {
                    range: Some((start, end)),
                    ..Default::default()
                })
            }
            ValueType::Scalar(DType::Bool) => Value::Scalar(Scalar {
                condition: Some(Condition::Parameter(path)),
                ..Default::default()
            }),
            _ => self.result(world, ty, true),
        }
    }
    fn region(&mut self, place: &Place) -> Region {
        self.view_region(&place.view)
    }
    fn select(
        &mut self,
        place: &Place,
        selections: &[(Option<IntExpr>, Option<IntExpr>, bool)],
    ) -> Place {
        Place {
            root: place.root,
            view: self.select_view(&place.view, selections),
        }
    }
    fn reshape(&mut self, place: &Place, axes: &[IntExpr]) -> Place {
        Place {
            root: place.root,
            view: self.reshape_view(&place.view, axes),
        }
    }
    fn write_region(&mut self, world: &mut World, root: usize, region: Region) {
        let Some(data) = world.roots.get(&root).cloned() else {
            return;
        };
        let written = self.normalize(data.written.union(region), &world.facts);
        let written = if self.covered(
            &written,
            &Region::Full,
            &world.path,
            &world.facts,
            &data.axes,
        ) {
            Region::Full
        } else {
            written
        };
        world.roots.get_mut(&root).unwrap().written = written;
    }
    fn require(&mut self, world: &mut World, root: usize, region: Region, span: Span) {
        let data = world.roots[&root].clone();
        let axes = &data.axes;
        if self.covered(
            &Region::Empty,
            &Region::Full,
            &world.path,
            &world.facts,
            axes,
        ) {
            return;
        }
        let available = data.written.clone().union(data.incoming.clone());
        if self.covered(&available, &region, &world.path, &world.facts, axes) {
            // Reads supplied by an abstract incoming parameter remain call
            // requirements. Invariant inference may provisionally supply that
            // incoming set, but cannot turn it into a produced write.
            if let Some(parameter) = data.parameter.clone() {
                if !self.covered(&data.written, &region, &world.path, &world.facts, axes) {
                    world.requirements.push(Requirement {
                        parameter,
                        region: region.clone(),
                        path: world.path.clone(),
                        span,
                    });
                }
            }
            return;
        }
        if world.defer_depth > 0 {
            world.deferred.push(Read {
                root,
                region,
                available,
                path: world.path.clone(),
                facts: world.facts.clone(),
                span,
            });
            return;
        }
        if let Some(parameter) = data.parameter {
            world.requirements.push(Requirement {
                parameter,
                region: region.clone(),
                path: world.path.clone(),
                span,
            });
            world.roots.get_mut(&root).unwrap().incoming = data.incoming.union(region);
            return;
        }
        let available = self.normalize(available, &world.facts);
        let region = self.normalize(region, &world.facts);
        let demonstrated = matches!(available, Region::Empty)
            && match region {
                Region::Full => true,
                Region::Linear(a, b) => self.lt(&world.facts, a, b),
                Region::Image { domain, .. } => domain
                    .iter()
                    .all(|bound| self.lt(&world.facts, bound.start, bound.end)),
                _ => false,
            };
        let message = if demonstrated {
            format!(
                "read before initialization of `{}`: this accessed region has not been written",
                data.name
            )
        } else {
            format!(
                "cannot establish initialization of the accessed region of `{}` before this read",
                data.name
            )
        };
        self.checker
            .error(DiagnosticRule::Initialization, span, message);
    }
    fn access(
        &mut self,
        world: &mut World,
        root: usize,
        region: Region,
        span: Span,
        write: bool,
        atomic: Option<AtomicOp>,
    ) {
        let path = world.path.clone();
        world.record_access(Access {
            root,
            region,
            path,
            span,
            write,
            atomic,
        });
    }
    fn read(&mut self, world: &mut World, root: usize, region: Region, span: Span) {
        self.access(world, root, region.clone(), span, false, None);
        self.require(world, root, region, span);
    }
    fn consume(&mut self, world: &mut World, value: &Value, span: Span) {
        match value {
            Value::Tensor(place) => {
                let region = self.region(place);
                self.read(world, place.root, region, span);
            }
            Value::Tuple(items) => {
                for item in items {
                    self.consume(world, item, span)
                }
            }
            _ => {}
        }
    }

    fn integer(&mut self, value: &Value) -> IntExpr {
        match value {
            Value::Scalar(Scalar {
                integer: Some(v), ..
            }) => *v,
            _ => self.fresh_integer().1,
        }
    }
    fn boolean(&mut self, value: &Value) -> Condition {
        match value {
            Value::Scalar(Scalar {
                condition: Some(v), ..
            }) => v.clone(),
            _ => self.fresh_condition(),
        }
    }
    /// Mutable because a call keeps only the alternatives whose contracts
    /// apply at it.
    fn expression(&mut self, world: &mut World, expression: &mut ir::Expr) -> Value {
        match &mut expression.kind {
            ir::ExprKind::Local(local) => world
                .values
                .get(local)
                .cloned()
                .expect("checked local has a source-order value binding"),
            ir::ExprKind::Literal(value) => {
                let mut scalar = Scalar::default();
                match value {
                    ReferenceScalar::I32(v) => {
                        scalar.integer = Some(self.arena().int(i64::from(*v)))
                    }
                    ReferenceScalar::U32(v) => {
                        scalar.integer = Some(self.arena().int(i64::from(*v)))
                    }
                    ReferenceScalar::Bool(v) => scalar.condition = Some(Condition::Constant(*v)),
                    _ => {}
                }
                Value::Scalar(scalar)
            }
            ir::ExprKind::Dimension(_) => Value::Scalar(Scalar {
                integer: expression.sym,
                ..Default::default()
            }),
            ir::ExprKind::Primitive { id, operands, .. } => {
                let values = operands
                    .iter_mut()
                    .map(|e| self.expression(world, e))
                    .collect::<Vec<_>>();
                match id {
                    PrimitiveId::TuplePack => Value::Tuple(values),
                    PrimitiveId::TupleGet(index) => match values.into_iter().next() {
                        Some(Value::Tuple(items)) => items[*index as usize].clone(),
                        _ => self.result(world, &expression.ty, true),
                    },
                    PrimitiveId::TensorAlloc => self.result(world, &expression.ty, false),
                    PrimitiveId::Fill(_) => self.result(world, &expression.ty, true),
                    PrimitiveId::Extent { axis } => {
                        let extent = match values.first() {
                            Some(Value::Tensor(place)) => place.axes.get(*axis as usize).copied(),
                            _ => None,
                        };
                        Value::Scalar(Scalar {
                            integer: extent,
                            ..Default::default()
                        })
                    }
                    PrimitiveId::SliceView { indices } => {
                        let Some(Value::Tensor(place)) = values.first() else {
                            return self.result(world, &expression.ty, true);
                        };
                        let mut rest = values[1..].iter();
                        let mut selected = vec![];
                        for index in indices {
                            match index {
                                IndexSlot::Point { .. } => selected.push((
                                    Some(self.integer(rest.next().unwrap())),
                                    None,
                                    true,
                                )),
                                IndexSlot::Range { start, end, .. } => selected.push((
                                    start.then(|| self.integer(rest.next().unwrap())),
                                    end.then(|| self.integer(rest.next().unwrap())),
                                    false,
                                )),
                                IndexSlot::Full => selected.push((None, None, false)),
                            }
                        }
                        Value::Tensor(self.select(place, &selected))
                    }
                    PrimitiveId::Transpose => match values.first() {
                        Some(Value::Tensor(place)) => {
                            let mut place = place.clone();
                            place.axes.reverse();
                            place.coordinates.reverse();
                            Value::Tensor(place)
                        }
                        _ => self.result(world, &expression.ty, true),
                    },
                    PrimitiveId::Reshape => match (values.first(), expression.ty.shaped()) {
                        (Some(Value::Tensor(place)), Some(shape)) => {
                            Value::Tensor(self.reshape(place, &shape.axes))
                        }
                        _ => self.result(world, &expression.ty, true),
                    },
                    PrimitiveId::ElementRead { .. } => {
                        if let Some(Value::Tensor(place)) = values.first() {
                            let selected = values[1..]
                                .iter()
                                .map(|v| (Some(self.integer(v)), None, true))
                                .collect::<Vec<_>>();
                            let place = self.select(place, &selected);
                            let region = self.region(&place);
                            self.read(world, place.root, region, expression.span);
                        }
                        self.result(world, &expression.ty, true)
                    }
                    PrimitiveId::RangeMake => Value::Scalar(Scalar {
                        range: Some((self.integer(&values[0]), self.integer(&values[1]))),
                        ..Default::default()
                    }),
                    PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                        let integer = match values.first() {
                            Some(Value::Scalar(Scalar {
                                range: Some((start, end)),
                                ..
                            })) => Some(if matches!(id, PrimitiveId::RangeStart) {
                                *start
                            } else {
                                *end
                            }),
                            _ => expression.sym,
                        };
                        Value::Scalar(Scalar {
                            integer,
                            ..Default::default()
                        })
                    }
                    PrimitiveId::Unary(op) => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        if matches!(expression.ty, ValueType::Tensor(_)) {
                            return self.result(world, &expression.ty, true);
                        }
                        let mut scalar = Scalar::default();
                        match op {
                            UnaryOp::Not => {
                                scalar.condition =
                                    Some(Condition::Not(Box::new(self.boolean(&values[0]))))
                            }
                            UnaryOp::Neg | UnaryOp::BitNot => {
                                scalar.integer = expression.sym;
                            }
                        }
                        Value::Scalar(scalar)
                    }
                    PrimitiveId::Binary(op) => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        if matches!(expression.ty, ValueType::Tensor(_)) {
                            return self.result(world, &expression.ty, true);
                        }
                        let mut scalar = Scalar::default();
                        let unchanged_symbols =
                            operands.iter().zip(&values).all(|(source, value)| {
                                match (source.sym, value) {
                                    (
                                        Some(expected),
                                        Value::Scalar(Scalar {
                                            integer: Some(actual),
                                            ..
                                        }),
                                    ) => self.same(expected, *actual),
                                    _ => false,
                                }
                            });
                        if matches!(op, BinaryOp::And | BinaryOp::Or) {
                            let a = Box::new(self.boolean(&values[0]));
                            let b = Box::new(self.boolean(&values[1]));
                            scalar.condition = Some(if *op == BinaryOp::And {
                                Condition::And(a, b)
                            } else {
                                Condition::Or(a, b)
                            });
                        } else {
                            let a = self.integer(&values[0]);
                            let b = self.integer(&values[1]);
                            match op {
                                BinaryOp::Add
                                | BinaryOp::Sub
                                | BinaryOp::Mul
                                | BinaryOp::Div
                                | BinaryOp::Rem => {
                                    scalar.integer =
                                        unchanged_symbols.then_some(expression.sym).flatten()
                                }
                                BinaryOp::Eq
                                | BinaryOp::Ne
                                | BinaryOp::Lt
                                | BinaryOp::Le
                                | BinaryOp::Gt
                                | BinaryOp::Ge => {
                                    scalar.condition = Some(
                                        if values.iter().all(|v| {
                                            matches!(
                                                v,
                                                Value::Scalar(Scalar {
                                                    integer: Some(_),
                                                    ..
                                                })
                                            )
                                        }) {
                                            Condition::Compare(*op, a, b)
                                        } else {
                                            self.fresh_condition()
                                        },
                                    )
                                }
                                _ => {}
                            }
                        }
                        Value::Scalar(scalar)
                    }
                    PrimitiveId::Constant(_) | PrimitiveId::Symbolic(_) => Value::Scalar(Scalar {
                        integer: expression.sym,
                        ..Default::default()
                    }),
                    PrimitiveId::Cast(_) => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        let mut result = self.result(world, &expression.ty, true);
                        if let Value::Scalar(s) = &mut result {
                            if expression.sym.is_some() {
                                s.integer = expression.sym;
                            }
                        }
                        result
                    }
                    PrimitiveId::Copy
                    | PrimitiveId::RepresentationConvert(_)
                    | PrimitiveId::Math(_)
                    | PrimitiveId::Select
                    | PrimitiveId::Reduce { .. } => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        self.result(world, &expression.ty, true)
                    }
                    PrimitiveId::Atomic { .. } => {
                        unreachable!("checked source atomics use their place-bearing expression")
                    }
                }
            }
            ir::ExprKind::Atomic {
                op,
                place,
                indices,
                value,
                ..
            } => {
                let place = Self::local_place(world, place);
                let indices = self.selection(world, indices);
                let value = self.expression(world, value);
                self.consume(world, &value, expression.span);
                if let Some(place) = place {
                    let place = self.select(&place, &indices);
                    let region = self.region(&place);
                    self.require(world, place.root, region.clone(), expression.span);
                    self.access(
                        world,
                        place.root,
                        region.clone(),
                        expression.span,
                        true,
                        Some(*op),
                    );
                    self.write_region(world, place.root, region);
                }
                Value::Void
            }
            ir::ExprKind::PlaneView { base, .. } => self.expression(world, base),
            ir::ExprKind::Intrinsic { overload, args } => {
                let values = args
                    .iter_mut()
                    .map(|e| self.expression(world, e))
                    .collect::<Vec<_>>();
                // Entry construction resolves one row of the overload, so an
                // operand is read when some row reads it and written when some
                // row writes it.
                let signatures = overload
                    .rows
                    .iter()
                    .map(|row| crate::registry::intrinsic_signature(*row))
                    .collect::<Vec<_>>();
                for (ordinal, value) in values.iter().enumerate() {
                    let read = signatures.iter().any(|signature| {
                        matches!(
                            signature.arguments[ordinal].category,
                            crate::registry::OperandCategory::Readable { .. }
                                | crate::registry::OperandCategory::Writable { .. }
                        )
                    });
                    if read {
                        self.consume(world, value, expression.span);
                    }
                }
                let written = signatures
                    .iter()
                    .flat_map(|signature| signature.effects.writes.iter().copied())
                    .collect::<BTreeSet<_>>();
                for ordinal in written {
                    if let Value::Tensor(place) = &values[ordinal as usize] {
                        let region = self.region(place);
                        self.access(world, place.root, region, expression.span, true, None);
                    }
                }
                // Writable intrinsic operands retain their incoming state unless
                // the intrinsic explicitly describes a complete destination.
                self.result(world, &expression.ty, true)
            }
            // L31: the quantity is the word's value.
            ir::ExprKind::IndexPosition(word) => match self.expression(world, word) {
                Value::Scalar(Scalar {
                    integer: Some(integer),
                    ..
                }) => Value::Scalar(Scalar {
                    integer: Some(integer),
                    ..Default::default()
                }),
                _ => Value::Scalar(Scalar {
                    integer: expression.sym,
                    ..Default::default()
                }),
            },
            ir::ExprKind::Call { call, args } => {
                for (_, value) in &mut call.seeds {
                    self.expression(world, value);
                }
                let values = args
                    .iter_mut()
                    .map(|e| self.expression(world, e))
                    .collect::<Vec<_>>();
                self.call(world, call, &values, expression.span);
                self.result(world, &expression.ty, true)
            }
        }
    }
    fn bind(&mut self, world: &mut World, pattern: &ir::Pattern, value: Value) {
        match (pattern, value) {
            (ir::Pattern::Local(local), value) => {
                if let Value::Tensor(place) = &value {
                    if world.roots[&place.root].parameter.is_none() {
                        world.roots.get_mut(&place.root).unwrap().name =
                            self.checker.locals[local.index()].name.clone();
                    }
                }
                world.values.insert(*local, value);
            }
            (ir::Pattern::Tuple(patterns), Value::Tuple(values)) => {
                for (p, v) in patterns.iter().zip(values) {
                    self.bind(world, p, v)
                }
            }
            _ => {}
        }
    }
    /// The tensor a local place currently holds.
    fn local_place(world: &World, place: &super::ownership::LocalPlace) -> Option<Place> {
        match world
            .values
            .get(&place.local)
            .map(|value| value.at(&place.path))
        {
            Some(Value::Tensor(place)) => Some(place.clone()),
            _ => None,
        }
    }
    /// The per-axis selection of checked indices, evaluated in order.
    fn selection(
        &mut self,
        world: &mut World,
        indices: &mut [ir::Index],
    ) -> Vec<(Option<IntExpr>, Option<IntExpr>, bool)> {
        let mut selected = vec![];
        for index in indices {
            match index {
                ir::Index::Point { value, .. } => {
                    let value = self.expression(world, value);
                    selected.push((Some(self.integer(&value)), None, true));
                }
                ir::Index::Range { start, end, .. } => {
                    let start = start.as_mut().map(|e| {
                        let v = self.expression(world, e);
                        self.integer(&v)
                    });
                    let end = end.as_mut().map(|e| {
                        let v = self.expression(world, e);
                        self.integer(&v)
                    });
                    selected.push((start, end, false));
                }
                ir::Index::Full => selected.push((None, None, false)),
            }
        }
        selected
    }
    fn target(&mut self, world: &mut World, place: &mut ir::Place) -> Target {
        match place {
            ir::Place::Tuple(places) => {
                Target::Tuple(places.iter_mut().map(|p| self.target(world, p)).collect())
            }
            ir::Place::Local(local) => Target::Binding(local.clone()),
            ir::Place::Element { root, indices } => {
                let place = Self::local_place(world, root)
                    .expect("checked element target has a tensor binding");
                let selected = self.selection(world, indices);
                Target::Tensor(self.select(&place, &selected))
            }
        }
    }
    fn assign(&mut self, world: &mut World, target: Target, value: Value, span: Span) {
        match target {
            Target::Tuple(targets) => {
                let Value::Tuple(values) = value else {
                    panic!("checked tuple assignment has a tuple value");
                };
                for (target, value) in targets.into_iter().zip(values) {
                    self.assign(world, target, value, span);
                }
            }
            Target::Binding(local) => {
                *world
                    .values
                    .get_mut(&local.local)
                    .expect("checked binding exists")
                    .at_mut(&local.path) = value;
            }
            Target::Tensor(place) => {
                self.consume(world, &value, span);
                let region = self.region(&place);
                self.access(world, place.root, region.clone(), span, true, None);
                self.write_region(world, place.root, region);
            }
        }
    }

    fn import_integer(&mut self, transfer: &mut Transfer<'_>, value: IntExpr) -> IntExpr {
        let source = &transfer.source.arena;
        let symbols = &mut transfer.symbols;
        xfer::transfer_int(source, value, self.arena(), &mut |symbol, arena| {
            AnyExpr::Int(*symbols.entry(symbol).or_insert_with(|| {
                let s = arena.proof_variable(crate::expr::SymbolSort::Int);
                arena.int_symbol(s)
            }))
        })
    }
    fn import_path(&mut self, transfer: &mut Transfer<'_>, path: &Path) -> Path {
        CallMapping {
            owner: self,
            transfer,
        }
        .path(path)
    }
    fn import_region(&mut self, transfer: &mut Transfer<'_>, region: &Region) -> Region {
        CallMapping {
            owner: self,
            transfer,
        }
        .region(region)
    }
    fn map_region(&mut self, region: Region, place: &Place) -> Region {
        self.map_view_region(region, &place.view)
    }
    fn call(&mut self, world: &mut World, call: &mut ir::Call, arguments: &[Value], span: Span) {
        let env = self.checker.env;
        let reference = env.resolved.families[call.family.index()].contract;
        let Some(Some(source)) = env.checked.get(reference.index()) else {
            unreachable!("a callee is checked before its caller (L20)")
        };
        let arguments = ordered_arguments(&call.arg_order, arguments);
        let Some(contract) = available_contract(source) else {
            // Cascade suppression: the callee's own diagnostics explain the
            // call. No requirement is recorded; every `&mut` argument is
            // written in full.
            for (parameter, argument) in source.signature.params.iter().zip(&arguments) {
                self.write_exclusive(world, &parameter.ownership, argument);
            }
            return;
        };
        let site = self.call_site(world, source, contract, &call.dimensions, &arguments, span);
        if self.record_loops {
            let mut applicable = Vec::with_capacity(call.candidates.len());
            for alternative in &call.candidates {
                applicable.push(
                    alternative.definition == reference
                        || self.alternative_applicable(
                            world,
                            &site,
                            alternative,
                            &call.dimensions,
                            &arguments,
                            span,
                        ),
                );
            }
            let mut applicable = applicable.into_iter();
            call.candidates.retain(|_| applicable.next().unwrap());
        }
        for access in site.accesses {
            world.record_access(access);
        }
        for (root, path, region) in site.requirements {
            let mut checked = world.clone();
            let before = checked.requirements.len();
            let deferred = checked.deferred.len();
            if !path
                .iter()
                .all(|(c, v)| self.assume(&mut checked.path, &mut checked.facts, c.clone(), *v))
            {
                continue;
            }
            self.require(&mut checked, root, region.clone(), span);
            world
                .requirements
                .extend(checked.requirements.into_iter().skip(before));
            world
                .deferred
                .extend(checked.deferred.into_iter().skip(deferred));
            if world.roots[&root].parameter.is_some() {
                let incoming = world.roots[&root]
                    .incoming
                    .clone()
                    .union(Region::Guard(path, Box::new(region)));
                world.roots.get_mut(&root).unwrap().incoming = incoming;
            }
        }
        for (root, region) in site.writes {
            self.write_region(world, root, region);
        }
    }
    /// Instantiate one member's checked contract at this call, in the
    /// caller's roots, coordinates and conditions. `dimensions` are the
    /// family's dimensions at the call, which every member shares by ordinal;
    /// `arguments` are in the family's parameter order.
    fn call_site(
        &mut self,
        world: &World,
        source: &CheckedOutcome,
        contract: &Contract,
        dimensions: &[IntExpr],
        arguments: &[Value],
        span: Span,
    ) -> CallSite {
        let mut transfer = Transfer {
            source,
            arguments,
            symbols: HashMap::new(),
        };
        for (dimension, &value) in source.signature.dimensions.iter().zip(dimensions) {
            transfer.symbols.insert(dimension.symbol, value);
        }
        for (symbol, part) in &contract.symbols {
            let value = match part {
                ParameterPart::Integer(p) => self.integer(argument(arguments, p)),
                ParameterPart::Start(p) | ParameterPart::End(p) => match argument(arguments, p) {
                    Value::Scalar(Scalar {
                        range: Some((a, b)),
                        ..
                    }) => {
                        if matches!(part, ParameterPart::Start(_)) {
                            *a
                        } else {
                            *b
                        }
                    }
                    _ => self.fresh_integer().1,
                },
            };
            transfer.symbols.insert(*symbol, value);
        }
        let mut site = CallSite {
            accesses: vec![],
            requirements: vec![],
            writes: vec![],
        };
        for access in &contract.accesses {
            let Value::Tensor(place) = argument(&arguments, &access.parameter) else {
                continue;
            };
            let mut path = world.path.clone();
            let mut facts = world.facts.clone();
            let callee_path = self.import_path(&mut transfer, &access.path);
            if !callee_path.iter().all(|(condition, truth)| {
                self.assume(&mut path, &mut facts, condition.clone(), *truth)
            }) {
                continue;
            }
            let region = self.import_region(&mut transfer, &access.region);
            let region = self.map_region(region, place);
            site.accesses.push(Access {
                root: place.root,
                region,
                path,
                span,
                write: access.write,
                atomic: access.atomic,
            });
        }
        for requirement in &contract.requirements {
            let Value::Tensor(place) = argument(&arguments, &requirement.parameter) else {
                continue;
            };
            let path = self.import_path(&mut transfer, &requirement.path);
            let region = self.import_region(&mut transfer, &requirement.region);
            let region = self.map_region(region, place);
            site.requirements.push((place.root, path, region));
        }
        for exit in &contract.exits {
            let path = self.import_path(&mut transfer, &exit.path);
            for (parameter, written) in &exit.written {
                let Value::Tensor(place) = argument(&arguments, parameter) else {
                    continue;
                };
                let region = self.import_region(&mut transfer, written);
                let region = self.map_region(region, place);
                site.writes
                    .push((place.root, Region::Guard(path.clone(), Box::new(region))));
            }
        }
        site
    }
    fn alternative_applicable(
        &mut self,
        world: &World,
        reference: &CallSite,
        alternative: &ir::Candidate,
        dimensions: &[IntExpr],
        arguments: &[Value],
        span: Span,
    ) -> bool {
        let env = self.checker.env;
        let Some(Some(source)) = env.checked.get(alternative.definition.index()) else {
            unreachable!("a family member is checked before its callers (L20)")
        };
        let Some(contract) = available_contract(source) else {
            return false;
        };
        let site = self.call_site(world, source, contract, dimensions, arguments, span);
        self.applicable_at(world, reference, &site)
    }
    /// Whether an alternative's contract applies where the reference's does:
    /// it requires nothing beyond the reference's requirements and what is
    /// already written, and its guaranteed writes include the reference's.
    fn applicable_at(
        &mut self,
        world: &World,
        reference: &CallSite,
        alternative: &CallSite,
    ) -> bool {
        for (root, path, region) in &alternative.requirements {
            let data = &world.roots[root];
            let available = reference
                .requirements
                .iter()
                .filter(|(required, _, _)| required == root)
                .fold(
                    data.written.clone().union(data.incoming.clone()),
                    |available, (_, path, region)| {
                        available.union(Region::Guard(path.clone(), Box::new(region.clone())))
                    },
                );
            let required = Region::Guard(path.clone(), Box::new(region.clone()));
            if !self.covered(&available, &required, &world.path, &world.facts, &data.axes) {
                return false;
            }
        }
        for (root, region) in &reference.writes {
            let data = &world.roots[root];
            let available = alternative
                .writes
                .iter()
                .filter(|(written, _)| written == root)
                .fold(data.written.clone(), |available, (_, region)| {
                    available.union(region.clone())
                });
            if !self.covered(&available, region, &world.path, &world.facts, &data.axes) {
                return false;
            }
        }
        true
    }
    /// A call whose contract is unavailable writes every exclusive borrow.
    fn write_exclusive(&mut self, world: &mut World, ownership: &ir::Ownership, value: &Value) {
        match (ownership, value) {
            (ir::Ownership::Tuple(parts), Value::Tuple(values)) => {
                for (ownership, value) in parts.iter().zip(values) {
                    self.write_exclusive(world, ownership, value);
                }
            }
            (ir::Ownership::Exclusive, Value::Tensor(place)) => {
                let region = self.region(place);
                self.write_region(world, place.root, region);
            }
            _ => {}
        }
    }

    fn block(&mut self, mut worlds: Vec<World>, block: &mut ir::Block) -> Vec<World> {
        for statement in &mut block.statements {
            let mut next = vec![];
            for mut world in worlds {
                match statement {
                    ir::Stmt::Let { pattern, value } => {
                        let value = self.expression(&mut world, value);
                        self.bind(&mut world, pattern, value);
                        next.push(world);
                    }
                    ir::Stmt::Assign { place, value, .. } => {
                        // Checked addresses and right-hand values precede the
                        // store; element address expressions may themselves read.
                        let target = self.target(&mut world, place);
                        let span = value.span;
                        let value = self.expression(&mut world, value);
                        self.assign(&mut world, target, value, span);
                        next.push(world);
                    }
                    ir::Stmt::Evaluate(expression) => {
                        self.expression(&mut world, expression);
                        next.push(world);
                    }
                    ir::Stmt::If {
                        condition,
                        then_body,
                        else_body,
                        join_symbols,
                        ..
                    } => {
                        let value = self.expression(&mut world, condition);
                        let condition = self.boolean(&value);
                        let mut arms = [vec![], vec![]];
                        for (arm, truth, body) in
                            [(0, true, &mut *then_body), (1, false, &mut *else_body)]
                        {
                            let mut branch = world.clone();
                            if !self.assume(
                                &mut branch.path,
                                &mut branch.facts,
                                condition.clone(),
                                truth,
                            ) {
                                continue;
                            }
                            arms[arm] = self.block(vec![branch], body);
                        }
                        let [then_worlds, else_worlds] = arms;
                        next.extend(self.join(
                            &world,
                            &condition,
                            then_worlds,
                            else_worlds,
                            join_symbols,
                        ));
                    }
                    ir::Stmt::Loop {
                        kind,
                        binder,
                        start,
                        end,
                        body,
                        initialization,
                        separation,
                        ..
                    } => next.extend(self.loop_body(
                        world,
                        *kind,
                        *binder,
                        start,
                        end,
                        body,
                        initialization,
                        separation,
                    )),
                }
            }
            worlds = next;
        }
        worlds
    }
    /// L30: the worlds after a two-way control join on `condition`, from the
    /// `entry` world both sides started from: `then_worlds` continue where it
    /// held and `else_worlds` where it did not. One world on each side merges
    /// into one world. Arms that bind a tensor local to different places stay
    /// separate worlds.
    fn join(
        &mut self,
        entry: &World,
        condition: &Condition,
        then_worlds: Vec<World>,
        else_worlds: Vec<World>,
        join_symbols: &[(ir::LocalId, SymbolId)],
    ) -> Vec<World> {
        if let ([then], [els]) = (then_worlds.as_slice(), else_worlds.as_slice()) {
            if let Some(joined) = self.join_worlds(entry, condition, then, els, join_symbols) {
                return vec![joined];
            }
        }
        then_worlds.into_iter().chain(else_worlds).collect()
    }
    /// One world for both arms: unchanged values stay, a changed integer
    /// becomes its join symbol with the bounds both arms prove, initialized
    /// regions are guarded by the arm that wrote them, and the records each
    /// arm added are kept with their own paths. Path facts learned inside an
    /// arm do not survive. `None` when the arms bind a tensor local to
    /// different places.
    fn join_worlds(
        &mut self,
        entry: &World,
        condition: &Condition,
        then: &World,
        els: &World,
        join_symbols: &[(ir::LocalId, SymbolId)],
    ) -> Option<World> {
        let scope = self.scope_symbols(entry);
        let mut joined = World {
            values: HashMap::new(),
            roots: HashMap::new(),
            path: entry.path.clone(),
            facts: entry.facts.clone(),
            requirements: entry.requirements.clone(),
            deferred: entry.deferred.clone(),
            accesses: entry.accesses.clone(),
            defer_depth: entry.defer_depth,
        };
        for local in entry.values.keys() {
            let symbol = join_symbols
                .iter()
                .find(|(changed, _)| changed == local)
                .map(|(_, symbol)| *symbol);
            let value = self.join_value(
                &mut joined.facts,
                [then, els],
                &then.values[local],
                &els.values[local],
                symbol,
                &scope,
            )?;
            joined.values.insert(*local, value);
        }
        for (id, root) in &then.roots {
            let root = match els.roots.get(id) {
                Some(other) => Root {
                    written: Region::branch(condition, root.written.clone(), other.written.clone()),
                    incoming: Region::branch(
                        condition,
                        root.incoming.clone(),
                        other.incoming.clone(),
                    ),
                    ..root.clone()
                },
                None => root.clone(),
            };
            joined.roots.insert(*id, root);
        }
        for (id, root) in &els.roots {
            joined.roots.entry(*id).or_insert_with(|| root.clone());
        }
        for arm in [then, els] {
            joined
                .requirements
                .extend_from_slice(&arm.requirements[entry.requirements.len()..]);
            joined
                .deferred
                .extend_from_slice(&arm.deferred[entry.deferred.len()..]);
            for access in &arm.accesses[entry.accesses.len()..] {
                joined.record_access(access.clone());
            }
        }
        Some(joined)
    }
    /// The value of one local after a join. `symbol` is the checker's join
    /// symbol of a changed integer local.
    fn join_value(
        &mut self,
        facts: &mut prove::Facts,
        arms: [&World; 2],
        then_value: &Value,
        else_value: &Value,
        symbol: Option<SymbolId>,
        scope: &BTreeSet<SymbolId>,
    ) -> Option<Value> {
        if then_value == else_value {
            return Some(then_value.clone());
        }
        match (then_value, else_value) {
            (Value::Scalar(a), Value::Scalar(b)) => {
                let integer = match (a.integer, b.integer, symbol) {
                    (Some(x), Some(y), _) if x == y => Some(x),
                    (Some(x), Some(y), symbol) => {
                        Some(self.join_integer(facts, arms, [x, y], symbol, scope))
                    }
                    (_, _, Some(symbol)) => Some(self.arena().int_symbol(symbol)),
                    (_, _, None) => None,
                };
                let condition = match (&a.condition, &b.condition) {
                    (Some(x), Some(y)) if x == y => Some(x.clone()),
                    (Some(_), Some(_)) => Some(self.fresh_condition()),
                    _ => None,
                };
                let range = match (a.range, b.range) {
                    (Some(x), Some(y)) if x == y => Some(x),
                    (Some((then_start, then_end)), Some((else_start, else_end))) => Some((
                        self.join_integer(facts, arms, [then_start, else_start], None, scope),
                        self.join_integer(facts, arms, [then_end, else_end], None, scope),
                    )),
                    _ => None,
                };
                Some(Value::Scalar(Scalar {
                    integer,
                    condition,
                    range,
                }))
            }
            (Value::Tuple(a), Value::Tuple(b)) => a
                .iter()
                .zip(b)
                .map(|(x, y)| self.join_value(facts, arms, x, y, None, scope))
                .collect::<Option<Vec<_>>>()
                .map(Value::Tuple),
            (Value::Tensor(_), Value::Tensor(_)) => None,
            _ => panic!("checked join arms bind one value kind"),
        }
    }
    /// A joined integer: `symbol` (or a fresh one) bounded by exactly the
    /// bounds both arm values prove (`Facts::join_bounds`).
    fn join_integer(
        &mut self,
        facts: &mut prove::Facts,
        arms: [&World; 2],
        values: [IntExpr; 2],
        symbol: Option<SymbolId>,
        scope: &BTreeSet<SymbolId>,
    ) -> IntExpr {
        let (symbol, value) = match symbol {
            Some(symbol) => (symbol, self.arena().int_symbol(symbol)),
            None => self.fresh_integer(),
        };
        facts.join_bounds(
            self.arena(),
            symbol,
            [(&arms[0].facts, values[0]), (&arms[1].facts, values[1])],
            &|candidate| scope.contains(&candidate),
        );
        value
    }
}

struct Transfer<'a> {
    source: &'a CheckedOutcome,
    arguments: &'a [Value],
    symbols: HashMap<SymbolId, IntExpr>,
}

/// One candidate's contract instantiated at one call: its accesses, its
/// requirements `(root, path, region)` and its guaranteed writes
/// `(root, region)`, all in the caller.
struct CallSite {
    accesses: Vec<Access>,
    requirements: Vec<(usize, Path, Region)>,
    writes: Vec<(usize, Region)>,
}

/// A callee's initialization contract. It is unavailable when the callee's
/// own check produced diagnostics; a call site then records nothing against
/// it (cascade suppression).
fn available_contract(source: &CheckedOutcome) -> Option<&Contract> {
    source
        .diagnostics
        .is_empty()
        .then_some(&source.initialization)
}

/// Call argument values in the family's parameter order (L3).
fn ordered_arguments(order: &[usize], arguments: &[Value]) -> Vec<Value> {
    order.iter().map(|&i| arguments[i].clone()).collect()
}

struct CallMapping<'borrow, 'checker, 'env, 'source> {
    owner: &'borrow mut Initialization<'checker, 'env>,
    transfer: &'borrow mut Transfer<'source>,
}
impl RegionMapping for CallMapping<'_, '_, '_, '_> {
    fn integer(&mut self, value: IntExpr) -> IntExpr {
        self.owner.import_integer(self.transfer, value)
    }
    fn binder(&mut self, symbol: SymbolId) -> SymbolId {
        let value = if let Some(value) = self.transfer.symbols.get(&symbol) {
            *value
        } else {
            let (_, value) = self.owner.fresh_integer();
            self.transfer.symbols.insert(symbol, value);
            value
        };
        match self.owner.arena_ref().view(AnyExpr::Int(value)) {
            crate::expr::NodeView::Symbol(s) => s,
            _ => unreachable!("bound coordinate maps to a bound coordinate"),
        }
    }
    fn parameter(&mut self, path: &ParameterPath) -> ParameterPath {
        path.clone()
    }
    fn predicate(&mut self, path: &ParameterPath) -> Condition {
        self.owner.boolean(argument(self.transfer.arguments, path))
    }
}

impl Initialization<'_, '_> {
    fn condition_symbols(&self, condition: &Condition, symbols: &mut BTreeSet<SymbolId>) {
        match condition {
            Condition::Version(_, binders) | Condition::Actual(_, binders) => {
                symbols.extend(binders.iter().copied());
            }
            Condition::Compare(_, left, right) => {
                symbols.extend(prove::symbols(self.arena_ref(), *left));
                symbols.extend(prove::symbols(self.arena_ref(), *right));
            }
            Condition::Not(inner) => self.condition_symbols(inner, symbols),
            Condition::And(left, right) | Condition::Or(left, right) => {
                self.condition_symbols(left, symbols);
                self.condition_symbols(right, symbols);
            }
            Condition::Constant(_) | Condition::Parameter(_) => {}
        }
    }
    fn region_symbols(&self, region: &Region, symbols: &mut BTreeSet<SymbolId>) {
        match region {
            Region::Linear(start, end) => {
                symbols.extend(prove::symbols(self.arena_ref(), *start));
                symbols.extend(prove::symbols(self.arena_ref(), *end));
            }
            Region::Image {
                domain,
                coordinates,
            } => {
                for coordinate in coordinates {
                    symbols.extend(prove::symbols(self.arena_ref(), *coordinate));
                }
                self.domain_symbols(domain, symbols);
            }
            Region::LinearImage { domain, address } => {
                symbols.extend(prove::symbols(self.arena_ref(), *address));
                self.domain_symbols(domain, symbols);
            }
            Region::Union(parts) | Region::Intersection(parts) => {
                for part in parts {
                    self.region_symbols(part, symbols);
                }
            }
            Region::Bind(bound, inner) => {
                symbols.insert(bound.symbol);
                symbols.extend(prove::symbols(self.arena_ref(), bound.start));
                symbols.extend(prove::symbols(self.arena_ref(), bound.end));
                self.region_symbols(inner, symbols);
            }
            Region::Guard(path, inner) => {
                for (condition, _) in path {
                    self.condition_symbols(condition, symbols);
                }
                self.region_symbols(inner, symbols);
            }
            Region::Empty | Region::Full => {}
        }
    }
    fn domain_symbols(&self, domain: &[Bound], symbols: &mut BTreeSet<SymbolId>) {
        for bound in domain {
            symbols.insert(bound.symbol);
            symbols.extend(prove::symbols(self.arena_ref(), bound.start));
            symbols.extend(prove::symbols(self.arena_ref(), bound.end));
        }
    }
    /// Symbols shared by distinct visits of a loop over `[start, end)`.
    fn captured_symbols(&self, entry: &World, start: IntExpr, end: IntExpr) -> BTreeSet<SymbolId> {
        let mut symbols = self.scope_symbols(entry);
        symbols.extend(prove::symbols(self.arena_ref(), start));
        symbols.extend(prove::symbols(self.arena_ref(), end));
        symbols
    }
    /// Symbols that denote the same value everywhere after `entry`: shape
    /// symbols, enclosing loop binders, the values of its locals and its
    /// path conditions.
    fn scope_symbols(&self, entry: &World) -> BTreeSet<SymbolId> {
        fn value_symbols(value: &Value, arena: &ExprArena, symbols: &mut BTreeSet<SymbolId>) {
            match value {
                Value::Scalar(scalar) => {
                    if let Some(integer) = scalar.integer {
                        symbols.extend(prove::symbols(arena, integer));
                    }
                    if let Some((start, end)) = scalar.range {
                        symbols.extend(prove::symbols(arena, start));
                        symbols.extend(prove::symbols(arena, end));
                    }
                }
                Value::Tuple(parts) => {
                    for part in parts {
                        value_symbols(part, arena, symbols);
                    }
                }
                Value::Tensor(_) | Value::Void => {}
            }
        }
        let mut symbols = self
            .dimension_symbols()
            .into_iter()
            .collect::<BTreeSet<_>>();
        symbols.extend(self.binders.iter().copied());
        for value in entry.values.values() {
            value_symbols(value, self.arena_ref(), &mut symbols);
        }
        for (condition, _) in &entry.path {
            self.condition_symbols(condition, &mut symbols);
        }
        symbols
    }
    fn independent_path(&self, path: &Path, symbol: SymbolId) -> Path {
        path.iter()
            .filter(|(c, _)| !self.condition_mentions(c, symbol))
            .cloned()
            .collect()
    }
    fn compatible_paths(a: &Path, b: &Path) -> bool {
        !a.iter().any(|(condition, value)| {
            b.iter()
                .any(|(other, other_value)| condition == other && value != other_value)
        })
    }
    fn distinct_visit_accesses_separate(
        &mut self,
        left: &Access,
        right: &Access,
        symbol: SymbolId,
        start: IntExpr,
        end: IntExpr,
        facts: &prove::Facts,
        captured: &BTreeSet<SymbolId>,
        extents: &[IntExpr],
    ) -> bool {
        let (other_symbol, other) = self.fresh_integer();
        let mut symbols = BTreeSet::new();
        self.region_symbols(&right.region, &mut symbols);
        for (condition, _) in &right.path {
            self.condition_symbols(condition, &mut symbols);
        }
        let mut rename = HashMap::from([(symbol, other)]);
        for local in symbols {
            if local != symbol && !captured.contains(&local) {
                rename.insert(local, self.fresh_integer().1);
            }
        }
        let other_region = self.rename_region(&right.region, &rename);
        let other_path = right
            .path
            .iter()
            .map(|(condition, truth)| (self.rename_condition(condition, &rename), *truth))
            .collect::<Vec<_>>();
        let mut facts = facts.clone();
        let one = self.arena().int(1);
        let upper = self.arena().int_sub(end, one);
        facts.set_range(other_symbol, start, upper);
        let current = self.arena().int_symbol(symbol);
        for (a, b) in [(current, other), (other, current)] {
            let mut orientation = facts.clone();
            let mut path = vec![];
            if !left
                .path
                .iter()
                .chain(&other_path)
                .all(|(condition, truth)| {
                    self.assume(&mut path, &mut orientation, condition.clone(), *truth)
                })
                || !self.assume(
                    &mut path,
                    &mut orientation,
                    Condition::Compare(BinaryOp::Lt, a, b),
                    true,
                )
            {
                continue;
            }
            if !self.separated_regions(
                left.region.clone(),
                other_region.clone(),
                &orientation,
                extents,
            ) {
                return false;
            }
        }
        true
    }
    /// The proof that distinct visits of a loop over `symbol in [start, end)`
    /// touch separated elements of storage live at loop entry whenever one of
    /// them writes, including accesses imported from callee contracts. Two
    /// `parallel for` atomic accesses with one operation commute and need no
    /// separation (L12); an ordered loop has no atomic access at all (L25 I3).
    /// `Err` names the first pair that may overlap.
    fn visits_separated(
        &mut self,
        kind: ir::LoopKind,
        entry: &World,
        outcomes: &[World],
        access_start: usize,
        symbol: SymbolId,
        start: IntExpr,
        end: IntExpr,
        facts: &prove::Facts,
    ) -> Result<(), VisitOverlap> {
        let captured = self.captured_symbols(entry, start, end);
        let accesses = outcomes
            .iter()
            .flat_map(|outcome| outcome.accesses.iter().skip(access_start))
            .filter(|access| entry.roots.contains_key(&access.root))
            .cloned()
            .collect::<Vec<_>>();
        for (left_index, left) in accesses.iter().enumerate() {
            for right in accesses.iter().skip(left_index) {
                let commuting = match kind {
                    ir::LoopKind::Independent => {
                        left.atomic.is_some() && left.atomic == right.atomic
                    }
                    ir::LoopKind::Ordered => false,
                };
                let atomic = left.atomic.is_some() || right.atomic.is_some();
                if (!left.write && !right.write) || commuting {
                    continue;
                }
                // Distinct roots are distinct storage: parameters never alias.
                if left.root != right.root {
                    continue;
                }
                let overlap = VisitOverlap {
                    root: left.root,
                    left: left.span,
                    right: right.span,
                };
                if kind == ir::LoopKind::Ordered && atomic {
                    return Err(overlap);
                }
                let extents = entry.roots[&left.root].axes.clone();
                if !self.distinct_visit_accesses_separate(
                    left, right, symbol, start, end, facts, &captured, &extents,
                ) {
                    return Err(overlap);
                }
            }
        }
        Ok(())
    }
    fn rename_region(&mut self, region: &Region, map: &HashMap<SymbolId, IntExpr>) -> Region {
        match region {
            Region::Empty => Region::Empty,
            Region::Full => Region::Full,
            Region::Linear(a, b) => {
                Region::Linear(self.substitute(*a, map), self.substitute(*b, map))
            }
            Region::Image {
                domain,
                coordinates,
            } => Region::Image {
                domain: self.rename_domain(domain, map),
                coordinates: coordinates
                    .iter()
                    .map(|c| self.substitute(*c, map))
                    .collect(),
            },
            Region::LinearImage { domain, address } => Region::LinearImage {
                domain: self.rename_domain(domain, map),
                address: self.substitute(*address, map),
            },
            Region::Union(parts) => {
                Region::Union(parts.iter().map(|p| self.rename_region(p, map)).collect())
            }
            Region::Intersection(parts) => {
                Region::Intersection(parts.iter().map(|p| self.rename_region(p, map)).collect())
            }
            Region::Bind(bound, inner) => Region::Bind(
                Bound {
                    symbol: self.renamed_bound(bound.symbol, map),
                    start: self.substitute(bound.start, &map),
                    end: self.substitute(bound.end, &map),
                },
                Box::new(self.rename_region(inner, map)),
            ),
            Region::Guard(path, inner) => Region::Guard(
                path.iter()
                    .map(|(c, v)| (self.rename_condition(c, &map), *v))
                    .collect(),
                Box::new(self.rename_region(inner, map)),
            ),
        }
    }
    fn rename_domain(&mut self, domain: &[Bound], map: &HashMap<SymbolId, IntExpr>) -> Vec<Bound> {
        domain
            .iter()
            .map(|d| Bound {
                symbol: self.renamed_bound(d.symbol, map),
                start: self.substitute(d.start, map),
                end: self.substitute(d.end, map),
            })
            .collect()
    }
    fn renamed_bound(&self, symbol: SymbolId, map: &HashMap<SymbolId, IntExpr>) -> SymbolId {
        let Some(value) = map.get(&symbol) else {
            return symbol;
        };
        match self.arena_ref().view(AnyExpr::Int(*value)) {
            crate::expr::NodeView::Symbol(renamed) => renamed,
            _ => unreachable!("bound coordinate maps to a bound coordinate"),
        }
    }
    fn rename_condition(&mut self, c: &Condition, map: &HashMap<SymbolId, IntExpr>) -> Condition {
        match c {
            Condition::Version(version, binders) => Condition::Version(
                *version,
                binders
                    .iter()
                    .flat_map(|symbol| {
                        map.get(symbol)
                            .map(|value| prove::symbols(self.arena_ref(), *value))
                            .unwrap_or_else(|| vec![*symbol])
                    })
                    .collect(),
            ),
            Condition::Actual(value, binders) => Condition::Actual(
                *value,
                binders
                    .iter()
                    .flat_map(|symbol| {
                        map.get(symbol)
                            .map(|value| prove::symbols(self.arena_ref(), *value))
                            .unwrap_or_else(|| vec![*symbol])
                    })
                    .collect(),
            ),
            Condition::Compare(op, a, b) => {
                Condition::Compare(*op, self.substitute(*a, map), self.substitute(*b, map))
            }
            Condition::Not(c) => Condition::Not(Box::new(self.rename_condition(c, map))),
            Condition::And(a, b) => Condition::And(
                Box::new(self.rename_condition(a, map)),
                Box::new(self.rename_condition(b, map)),
            ),
            Condition::Or(a, b) => Condition::Or(
                Box::new(self.rename_condition(a, map)),
                Box::new(self.rename_condition(b, map)),
            ),
            other => other.clone(),
        }
    }
    fn record_loop_initialization(
        &mut self,
        metadata: &mut crate::initialization::LoopInitialization,
        entry: &World,
        symbol: SymbolId,
        binder: ir::LocalId,
        path: &Path,
        guaranteed: &HashMap<usize, Region>,
        invariants: &[(super::ownership::LocalPlace, Region)],
    ) {
        fn leaves(value: &Value, path: ParameterPath, result: &mut Vec<(ParameterPath, Value)>) {
            if let Value::Tuple(parts) = value {
                for (index, value) in parts.iter().enumerate() {
                    leaves(value, path.child(index), result);
                }
            } else {
                result.push((path, value.clone()));
            }
        }
        let offset = self.checker.sig.params.len();
        let mut captures = Vec::new();
        for (local, value) in &entry.values {
            leaves(
                value,
                ParameterPath::root(offset + local.index()),
                &mut captures,
            );
        }
        captures.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (symbol, part) in &self.symbols {
            if !metadata
                .transfer
                .symbols
                .iter()
                .any(|(prior, _)| prior == symbol)
            {
                metadata.transfer.symbols.push((*symbol, part.clone()));
            }
        }
        metadata.binder = ParameterPath::root(offset + binder.index());
        if !metadata
            .transfer
            .symbols
            .iter()
            .any(|(prior, _)| *prior == symbol)
        {
            metadata
                .transfer
                .symbols
                .push((symbol, ParameterPart::Integer(metadata.binder.clone())));
        }
        for (path, value) in &captures {
            if let Value::Scalar(Scalar {
                integer: Some(value),
                ..
            }) = value
            {
                if let crate::expr::NodeView::Symbol(symbol) =
                    self.arena_ref().view(AnyExpr::Int(*value))
                {
                    if !metadata
                        .transfer
                        .symbols
                        .iter()
                        .any(|(prior, _)| *prior == symbol)
                    {
                        metadata
                            .transfer
                            .symbols
                            .push((symbol, ParameterPart::Integer(path.clone())));
                    }
                }
            }
        }
        let mut allowed = self.dimension_symbols();
        allowed.extend(metadata.transfer.symbols.iter().map(|(symbol, _)| *symbol));
        allowed.extend(self.binders.iter().copied());
        let mut written = Vec::new();
        for (capture, value) in &captures {
            if let Value::Tensor(place) = value {
                let writes = guaranteed
                    .get(&place.root)
                    .cloned()
                    .unwrap_or(Region::Empty);
                let logical = self.project_view_region(writes, &place.view, path, &entry.facts);
                written.push((capture.clone(), logical));
            }
        }
        metadata.transfer.exits.push(Exit {
            path: path.clone(),
            written,
        });
        metadata.transfer = self.close_transfer(metadata.transfer.clone(), &allowed);
        for (local, invariant) in invariants {
            let mut parameter = ParameterPath::root(offset + local.local.index());
            for index in &local.path {
                parameter = parameter.child(*index);
            }
            let region = self.boundary_region(invariant.clone(), &allowed, false);
            if let Some((_, prior)) = metadata.carried.iter_mut().find(|(p, _)| *p == parameter) {
                *prior = prior.clone().intersection(region);
            } else {
                metadata.carried.push((parameter, region));
            }
        }
    }

    fn carried_region(&mut self, world: &World, place: &Place) -> Region {
        let root = &world.roots[&place.root];
        self.project_view_region(
            root.written.clone().union(root.incoming.clone()),
            &place.view,
            &world.path,
            &world.facts,
        )
    }
    fn carries_region(&mut self, world: &World, place: &Place, region: &Region) -> bool {
        let root = &world.roots[&place.root];
        let available = root.written.clone().union(root.incoming.clone());
        let required = self.map_region(region.clone(), place);
        self.covered(&available, &required, &world.path, &world.facts, &root.axes)
    }
    fn loop_body(
        &mut self,
        mut entry: World,
        kind: ir::LoopKind,
        binder: ir::LocalId,
        start: &mut ir::Expr,
        end: &mut ir::Expr,
        body: &mut ir::Block,
        metadata: &mut crate::initialization::LoopInitialization,
        separation: &mut VisitSeparation,
    ) -> Vec<World> {
        let loop_span = start.span;
        let start_value = self.expression(&mut entry, start);
        let start = self.integer(&start_value);
        let end_value = self.expression(&mut entry, end);
        let end = self.integer(&end_value);
        let nonempty = Condition::Compare(BinaryOp::Lt, start, end);
        let mut empty = entry.clone();
        let empty = self
            .assume(&mut empty.path, &mut empty.facts, nonempty.clone(), false)
            .then_some(empty);
        let mut iteration = entry.clone();
        if !self.assume(
            &mut iteration.path,
            &mut iteration.facts,
            nonempty.clone(),
            true,
        ) {
            return empty.into_iter().collect();
        }
        let symbol = self.checker.locals[binder.index()]
            .symbol
            .expect("checked loop binder has its symbolic identity");
        let current = self.arena().int_symbol(symbol);
        let one = self.arena().int(1);
        let upper = self.arena().int_sub(end, one);
        iteration.facts.set_range(symbol, start, upper);
        let access_facts = iteration.facts.clone();
        iteration.values.insert(
            binder,
            Value::Scalar(Scalar {
                integer: Some(current),
                ..Default::default()
            }),
        );
        // Carry state is one inductive value contract. Derive a region
        // invariant, verify the body once under it, and preserve that same
        // root/initialization meaning at every next visit.
        self.binders.push(symbol);
        let width = self.arena().int_sub(end, start);
        let single = self.same(width, one);
        let mut reassigned = Vec::new();
        assigned_bindings(body, &mut reassigned);
        let carries = reassigned
            .iter()
            .filter_map(|local| {
                match iteration
                    .values
                    .get(&local.local)
                    .map(|value| value.at(&local.path))
                {
                    Some(Value::Tensor(place)) => Some((local.clone(), place.clone())),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        if !single {
            for local in &reassigned {
                if let Some(value) = iteration.values.get_mut(&local.local) {
                    let value = value.at_mut(&local.path);
                    if matches!(value, Value::Scalar(_)) {
                        *value = Value::Scalar(Scalar {
                            condition: Some(self.fresh_condition()),
                            ..Default::default()
                        });
                    }
                }
            }
        }
        iteration.defer_depth += 1;
        let deferred_start = iteration.deferred.len();
        let access_start = iteration.accesses.len();
        let mut invariants = Vec::new();
        let outcomes = if carries.is_empty() || single {
            self.block(vec![iteration], body)
        } else {
            let mut allowed = self.dimension_symbols();
            allowed.extend(self.symbols.iter().map(|(symbol, _)| *symbol));
            allowed.extend(self.binders.iter().copied().filter(|s| *s != symbol));
            for value in entry.values.values() {
                if let Value::Scalar(Scalar {
                    integer: Some(value),
                    ..
                }) = value
                {
                    allowed.extend(prove::symbols(self.arena_ref(), *value));
                }
            }
            let mut trial = iteration.clone();
            // A formal input's unknown incoming set may be constrained by
            // this carry contract. This provisional assumption is discharged
            // against the real initial value before verifying the body.
            for (_, place) in &carries {
                if trial.roots[&place.root].parameter.is_some() {
                    trial.roots.get_mut(&place.root).unwrap().incoming = Region::Full;
                }
            }
            let diagnostics = self.checker.diagnostics.len();
            let record_loops = std::mem::replace(&mut self.record_loops, false);
            let preliminary = self.block(vec![trial.clone()], body);
            self.record_loops = record_loops;
            self.checker.diagnostics.truncate(diagnostics);
            for (local, place) in &carries {
                let initial_root = &iteration.roots[&place.root];
                let mut initial = initial_root
                    .written
                    .clone()
                    .union(initial_root.incoming.clone());
                if let Some(parameter) = &initial_root.parameter {
                    for outcome in &preliminary {
                        for required in outcome
                            .requirements
                            .iter()
                            .filter(|r| &r.parameter == parameter)
                        {
                            let region = Region::Guard(
                                required.path.clone(),
                                Box::new(required.region.clone()),
                            );
                            let region =
                                Region::Bind(Bound { symbol, start, end }, Box::new(region));
                            let region = self.normalize(region, &iteration.facts);
                            initial = initial.union(self.boundary_region(region, &allowed, true));
                        }
                    }
                }
                let mut invariant = self.project_view_region(
                    initial,
                    &place.view,
                    &iteration.path,
                    &iteration.facts,
                );
                for outcome in &preliminary {
                    let Some(Value::Tensor(next)) = outcome
                        .values
                        .get(&local.local)
                        .map(|value| value.at(&local.path))
                    else {
                        unreachable!("tensor carry retains checked type");
                    };
                    if !self.carries_region(outcome, next, &invariant) {
                        invariant = invariant.intersection(self.carried_region(outcome, next));
                    }
                }
                invariant = self.normalize(invariant, &entry.facts);
                invariant = self.boundary_region(invariant, &allowed, false);
                let required = self.map_region(invariant.clone(), place);
                self.require(&mut iteration, place.root, required, loop_span);
                let name = iteration.roots[&place.root].name.clone();
                let abstract_place =
                    self.fresh_place(&mut iteration, &place.axes, None, false, name);
                // The abstract carry owns exactly this root's logical
                // elements. Full is therefore a complete-root fact here;
                // retaining it lets a later bijective transpose or reshape
                // project full initialization through its actual view.
                let written = if matches!(invariant, Region::Full) {
                    Region::Full
                } else {
                    self.map_region(invariant.clone(), &abstract_place)
                };
                iteration
                    .roots
                    .get_mut(&abstract_place.root)
                    .unwrap()
                    .written = written;
                *iteration
                    .values
                    .get_mut(&local.local)
                    .expect("carried local exists")
                    .at_mut(&local.path) = Value::Tensor(abstract_place);
                invariants.push((local.clone(), invariant));
            }
            let mut verified = self.block(vec![iteration], body);
            for outcome in &mut verified {
                for (local, invariant) in &invariants {
                    let Some(Value::Tensor(next)) = outcome
                        .values
                        .get(&local.local)
                        .map(|value| value.at(&local.path))
                        .cloned()
                    else {
                        unreachable!("tensor carry retains checked type");
                    };
                    let actual = self.carried_region(outcome, &next);
                    if !self.carries_region(outcome, &next, invariant) {
                        self.checker.error(
                            DiagnosticRule::Initialization,
                            loop_span,
                            "cannot establish an inductive initialized region for this tensor carry",
                        );
                    }
                    // Loop-private coordinates cannot escape as the final
                    // value's initialized set. Keep the verified invariant
                    // plus any independently established exit region.
                    let actual = self.boundary_region(actual, &allowed, false);
                    if outcome.roots[&next.root].parameter.is_none()
                        && !matches!(outcome.roots[&next.root].written, Region::Full)
                    {
                        outcome.roots.get_mut(&next.root).unwrap().written =
                            self.map_region(actual.union(invariant.clone()), &next);
                    }
                }
            }
            verified
        };
        self.binders.pop();
        match kind {
            ir::LoopKind::Ordered if self.record_loops => {
                // L25: (I2) no local live at loop entry is rebound, and
                // (I1)/(I3) distinct visits touch separated storage.
                let separated = reassigned
                    .iter()
                    .all(|local| !entry.values.contains_key(&local.local))
                    && self
                        .visits_separated(
                            kind,
                            &entry,
                            &outcomes,
                            access_start,
                            symbol,
                            start,
                            end,
                            &access_facts,
                        )
                        .is_ok();
                separation.record(separated);
            }
            ir::LoopKind::Ordered => {}
            ir::LoopKind::Independent => {
                if let Err(overlap) = self.visits_separated(
                    kind,
                    &entry,
                    &outcomes,
                    access_start,
                    symbol,
                    start,
                    end,
                    &access_facts,
                ) {
                    let root = &entry.roots[&overlap.root].name;
                    self.checker.error(
                        DiagnosticRule::Independence,
                        overlap.right,
                        format!(
                            "parallel for cannot establish independent visits: ordinary accesses to `{root}` at source offsets {} and {} may overlap across distinct visits",
                            overlap.left.start, overlap.right.start,
                        ),
                    );
                }
            }
        }
        let mut exits = vec![];
        for outcome in &outcomes {
            let path = self.independent_path(&outcome.path, symbol);
            let matching = outcomes
                .iter()
                .filter(|other| {
                    Self::compatible_paths(&path, &self.independent_path(&other.path, symbol))
                })
                .collect::<Vec<_>>();
            let mut guaranteed = HashMap::new();
            for root in entry.roots.keys() {
                let writes = matching.iter().fold(Region::Full, |state, other| {
                    state.intersection(
                        other
                            .roots
                            .get(root)
                            .map_or(Region::Empty, |r| r.written.clone()),
                    )
                });
                let writes = self.normalize(writes, &entry.facts);
                guaranteed.insert(*root, writes);
            }
            if self.record_loops {
                self.record_loop_initialization(
                    metadata,
                    &entry,
                    symbol,
                    binder,
                    &path,
                    &guaranteed,
                    &invariants,
                );
            }
            let mut exit = outcome.clone();
            exit.accesses = entry.accesses.clone();
            for access in outcome.accesses.iter().skip(access_start) {
                if !entry.roots.contains_key(&access.root) {
                    continue;
                }
                let region = Region::Bind(
                    Bound { symbol, start, end },
                    Box::new(access.region.clone()),
                );
                let region = self.normalize(region, &entry.facts);
                let path = self.independent_path(&access.path, symbol);
                exit.record_access(Access {
                    root: access.root,
                    region,
                    path,
                    span: access.span,
                    write: access.write,
                    atomic: access.atomic,
                });
            }
            exit.defer_depth = entry.defer_depth;
            exit.deferred = entry.deferred.clone();
            exit.facts = entry.facts.clone();
            exit.path = entry.path.clone();
            if !path
                .iter()
                .all(|(c, v)| self.assume(&mut exit.path, &mut exit.facts, c.clone(), *v))
            {
                continue;
            }
            if !self.assume(&mut exit.path, &mut exit.facts, nonempty.clone(), true) {
                continue;
            }
            // Reads are checked against their own source-position state and,
            // only for ordered loops, completed earlier iterations.
            for read in outcome.deferred.iter().skip(deferred_start) {
                let mut available = read.available.clone();
                if kind == ir::LoopKind::Ordered {
                    if let Some(writes) = guaranteed.get(&read.root) {
                        let (prior_symbol, prior) = self.fresh_integer();
                        let writes = self.rename_region(writes, &HashMap::from([(symbol, prior)]));
                        let prefix = Region::Bind(
                            Bound {
                                symbol: prior_symbol,
                                start,
                                end: current,
                            },
                            Box::new(writes),
                        );
                        available = available.union(self.normalize(prefix, &read.facts));
                    }
                }
                let axes = &outcome.roots[&read.root].axes;
                if self.covered(&available, &read.region, &read.path, &read.facts, axes) {
                    continue;
                }
                let region =
                    Region::Bind(Bound { symbol, start, end }, Box::new(read.region.clone()));
                let region = self.normalize(region, &exit.facts);
                let path = self.independent_path(&read.path, symbol);
                if entry.defer_depth > 0 {
                    exit.deferred.push(Read {
                        root: read.root,
                        region,
                        available: Region::Empty,
                        path,
                        facts: exit.facts.clone(),
                        span: read.span,
                    });
                } else if let Some(root) = exit.roots.get(&read.root).cloned() {
                    if let Some(parameter) = root.parameter {
                        exit.requirements.push(Requirement {
                            parameter,
                            region: region.clone(),
                            path: path.clone(),
                            span: read.span,
                        });
                        exit.roots.get_mut(&read.root).unwrap().incoming =
                            root.incoming.union(Region::Guard(path, Box::new(region)));
                    } else {
                        // Recheck with the saved source-position state, never
                        // with writes encountered later while deriving the body.
                        let mut at_read = exit.clone();
                        at_read.path = read.path.clone();
                        at_read.facts = read.facts.clone();
                        at_read.roots.get_mut(&read.root).unwrap().written = available;
                        at_read.roots.get_mut(&read.root).unwrap().incoming = Region::Empty;
                        self.require(&mut at_read, read.root, read.region.clone(), read.span);
                    }
                }
            }
            for (root, before) in &entry.roots {
                let written = guaranteed.get(root).cloned().unwrap_or(Region::Empty);
                let completed = Region::Bind(Bound { symbol, start, end }, Box::new(written));
                exit.roots.get_mut(root).unwrap().written = before.written.clone();
                self.write_region(&mut exit, *root, completed);
            }
            exit.values
                .retain(|local, _| entry.values.contains_key(local));
            exits.push(exit);
        }
        // Loop exits join like `if` arms, on whether the loop ran at all.
        self.join(&entry, &nonempty, exits, empty.into_iter().collect(), &[])
    }
}

fn assigned_bindings(block: &ir::Block, result: &mut Vec<super::ownership::LocalPlace>) {
    fn target_bindings(place: &ir::Place, result: &mut Vec<super::ownership::LocalPlace>) {
        match place {
            ir::Place::Local(local) => {
                if !result.contains(local) {
                    result.push(local.clone());
                }
            }
            ir::Place::Tuple(parts) => {
                for part in parts {
                    target_bindings(part, result);
                }
            }
            ir::Place::Element { .. } => {}
        }
    }
    for statement in &block.statements {
        match statement {
            ir::Stmt::Assign { place: target, .. } => target_bindings(target, result),
            ir::Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                assigned_bindings(then_body, result);
                assigned_bindings(else_body, result);
            }
            ir::Stmt::Loop { body, .. } => assigned_bindings(body, result),
            _ => {}
        }
    }
}

impl RegionOps for Initialization<'_, '_> {
    fn arena(&mut self) -> &mut ExprArena {
        &mut self.checker.arena
    }
    fn arena_ref(&self) -> &ExprArena {
        &self.checker.arena
    }
    fn fresh_variable(&mut self) -> SymbolId {
        self.checker
            .arena
            .proof_variable(crate::expr::SymbolSort::Int)
    }
}

fn argument<'a>(arguments: &'a [Value], path: &ParameterPath) -> &'a Value {
    let mut value = &arguments[path.parameter];
    for &field in &path.fields {
        let Value::Tuple(parts) = value else {
            panic!("checked parameter path selects tuple");
        };
        value = &parts[field];
    }
    value
}

impl Initialization<'_, '_> {
    /// The symbols of the definition's dimensions.
    fn dimension_symbols(&self) -> Vec<SymbolId> {
        self.checker
            .sig
            .dimensions
            .iter()
            .map(|dimension| dimension.symbol)
            .collect()
    }
    fn close_contract(&mut self, contract: Contract) -> Contract {
        let mut allowed = self.dimension_symbols();
        allowed.extend(contract.symbols.iter().map(|(symbol, _)| *symbol));
        self.close_transfer(contract, &allowed)
    }
}
