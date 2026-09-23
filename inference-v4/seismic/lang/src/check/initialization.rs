//! Source-order definite initialization. This is a mandatory checking rule,
//! run before a checked definition is published. Regions describe actual
//! logical coordinates; no runtime storage or compiler-side certificate exists.
use super::{ir, prove, xfer, CheckedOutcome, Checker};
use crate::expr::{AnyExpr, ExprArena, IntExpr, SymbolId};
pub(super) use crate::initialization::InitializationContract as Contract;
use crate::initialization::RegionMapping;
use crate::initialization::{
    Bound, Condition, Exit, InitializationView, ParameterAccess, ParameterPart, ParameterPath,
    Path, Region, RegionOps, Requirement,
};
use crate::intrinsics::{IndexSlot, PrimitiveId};
use crate::reference_math::ReferenceScalar;
use crate::span::{Diagnostic, Span};
use crate::syntax::ast::{AssignOp, BinaryOp, UnaryOp};
use crate::types::{DType, ValueType};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
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
#[derive(Clone, Debug, Default)]
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
#[derive(Clone, Debug)]
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
    elements: IntExpr,
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
    returned: bool,
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
    atomic: bool,
}
pub(super) fn check(
    checker: &mut Checker<'_>,
    block: &mut ir::Block,
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
        returned: false,
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
    let worlds = owner.block(vec![initial], block);
    let mut result = Contract {
        symbols: owner.symbols.clone(),
        ..Contract::empty()
    };
    for world in worlds {
        result.requirements.extend(world.requirements);
        result
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
        result.exits.push(Exit {
            path: world.path,
            written: world
                .roots
                .into_values()
                .filter_map(|root| root.parameter.map(|p| (p, root.written)))
                .collect(),
        });
    }
    owner.close_contract(result)
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
    fn product(&mut self, axes: &[IntExpr]) -> IntExpr {
        axes.iter()
            .fold(self.arena().int(1), |p, a| self.arena().int_mul(p, *a))
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
        let elements = self.product(axes);
        world.roots.insert(
            root,
            Root {
                name,
                elements,
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
    fn root_domain(&mut self, root: &Root) -> Region {
        let zero = self.arena().int(0);
        Region::Interval(zero, root.elements)
    }
    fn write_region(&mut self, world: &mut World, root: usize, region: Region) {
        let Some(data) = world.roots.get(&root).cloned() else {
            return;
        };
        let domain = self.root_domain(&data);
        let written = self.normalize(data.written.union(region), &world.facts);
        let written = if self.covered(&written, &domain, &world.path, &world.facts) {
            Region::Full
        } else {
            written
        };
        world.roots.get_mut(&root).unwrap().written = written;
    }
    fn require(&mut self, world: &mut World, root: usize, region: Region, span: Span) {
        let data = world.roots[&root].clone();
        let domain = self.root_domain(&data);
        if self.covered(&Region::Empty, &domain, &world.path, &world.facts) {
            return;
        }
        let available = data.written.clone().union(data.incoming.clone());
        if self.covered(&available, &region, &world.path, &world.facts) {
            // Reads supplied by an abstract incoming parameter remain call
            // requirements. Invariant inference may provisionally supply that
            // incoming set, but cannot turn it into a produced write.
            if let Some(parameter) = data.parameter.clone() {
                if !self.covered(&data.written, &region, &world.path, &world.facts) {
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
                Region::Interval(a, b) => self.lt(&world.facts, a, b),
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
            .diagnostics
            .push(Diagnostic::new(span, message));
    }
    fn access(
        &mut self,
        world: &mut World,
        root: usize,
        region: Region,
        span: Span,
        write: bool,
        atomic: bool,
    ) {
        world.accesses.push(Access {
            root,
            region,
            path: world.path.clone(),
            span,
            write,
            atomic,
        });
    }
    fn read(&mut self, world: &mut World, root: usize, region: Region, span: Span) {
        self.access(world, root, region.clone(), span, false, false);
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
    fn expression(&mut self, world: &mut World, expression: &ir::Expr) -> Value {
        match &expression.kind {
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
            ir::ExprKind::Primitive { id, operands } => {
                let values = operands
                    .iter()
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
                                IndexSlot::Point => selected.push((
                                    Some(self.integer(rest.next().unwrap())),
                                    None,
                                    true,
                                )),
                                IndexSlot::Range { start, end } => selected.push((
                                    start.then(|| self.integer(rest.next().unwrap())),
                                    end.then(|| self.integer(rest.next().unwrap())),
                                    false,
                                )),
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
                    PrimitiveId::Materialize
                    | PrimitiveId::Clone
                    | PrimitiveId::Load
                    | PrimitiveId::RepresentationConvert(_)
                    | PrimitiveId::Decode
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
                place,
                indices,
                value,
                ..
            } => {
                let place = self.expression(world, place);
                let indices = indices
                    .iter()
                    .map(|e| {
                        let v = self.expression(world, e);
                        (Some(self.integer(&v)), None, true)
                    })
                    .collect::<Vec<_>>();
                let value = self.expression(world, value);
                self.consume(world, &value, expression.span);
                if let Value::Tensor(place) = place {
                    let place = self.select(&place, &indices);
                    let region = self.region(&place);
                    self.require(world, place.root, region.clone(), expression.span);
                    self.access(
                        world,
                        place.root,
                        region.clone(),
                        expression.span,
                        true,
                        true,
                    );
                    self.write_region(world, place.root, region);
                }
                Value::Void
            }
            ir::ExprKind::PlaneView { base, .. } => self.expression(world, base),
            ir::ExprKind::Intrinsic { id, args } => {
                let values = args
                    .iter()
                    .map(|e| self.expression(world, e))
                    .collect::<Vec<_>>();
                let signature = crate::registry::intrinsic_signature(*id);
                for (argument, value) in signature.arguments.iter().zip(&values) {
                    if matches!(
                        argument.category,
                        crate::registry::OperandCategory::Readable { .. }
                            | crate::registry::OperandCategory::Writable { .. }
                    ) {
                        self.consume(world, value, expression.span);
                    }
                }
                for ordinal in signature.effects.writes.iter().copied() {
                    if let Some(Value::Tensor(place)) = values.get(ordinal as usize) {
                        let region = self.region(place);
                        self.access(world, place.root, region, expression.span, true, false);
                    }
                }
                // Writable intrinsic operands retain their incoming state unless
                // the intrinsic explicitly describes a complete destination.
                self.result(world, &expression.ty, true)
            }
            ir::ExprKind::Call { call, args } => {
                for (_, value) in &call.explicit_shapes {
                    self.expression(world, value);
                }
                let values = args
                    .iter()
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
    fn place(&mut self, world: &mut World, place: &ir::Place) -> Option<Place> {
        match place {
            ir::Place::Local(local) => match world
                .values
                .get(&local.local)
                .map(|value| value.at(&local.path))
            {
                Some(Value::Tensor(place)) => Some(place.clone()),
                _ => None,
            },
            ir::Place::Element { root, indices } => {
                let Some(Value::Tensor(place)) = world
                    .values
                    .get(&root.local)
                    .map(|value| value.at(&root.path).clone())
                else {
                    return None;
                };
                let mut selected = vec![];
                for index in indices {
                    match index {
                        ir::Index::Point { value, .. } => {
                            let value = self.expression(world, value);
                            selected.push((Some(self.integer(&value)), None, true));
                        }
                        ir::Index::Range { start, end, .. } => {
                            let start = start.as_ref().map(|e| {
                                let v = self.expression(world, e);
                                self.integer(&v)
                            });
                            let end = end.as_ref().map(|e| {
                                let v = self.expression(world, e);
                                self.integer(&v)
                            });
                            selected.push((start, end, false));
                        }
                    }
                }
                Some(self.select(&place, &selected))
            }
            ir::Place::Tuple(_) => None,
        }
    }
    fn target(&mut self, world: &mut World, place: &ir::Place, op: AssignOp) -> Target {
        match place {
            ir::Place::Tuple(places) => {
                Target::Tuple(places.iter().map(|p| self.target(world, p, op)).collect())
            }
            ir::Place::Local(local) => {
                if op != AssignOp::Assign {
                    if let Some(place) = self.place(world, place) {
                        return Target::Tensor(place);
                    }
                }
                Target::Binding(local.clone())
            }
            ir::Place::Element { .. } => Target::Tensor(
                self.place(world, place)
                    .expect("checked element target has a tensor binding"),
            ),
        }
    }
    fn assign(
        &mut self,
        world: &mut World,
        target: Target,
        op: AssignOp,
        value: Value,
        span: Span,
    ) {
        match target {
            Target::Tuple(targets) => {
                let Value::Tuple(values) = value else {
                    panic!("checked tuple assignment has a tuple value");
                };
                for (target, value) in targets.into_iter().zip(values) {
                    self.assign(world, target, op, value, span);
                }
            }
            Target::Binding(local) => {
                let value = if op == AssignOp::Assign {
                    value
                } else {
                    // A scalar compound update defines a new value version.
                    Value::Scalar(Scalar {
                        condition: Some(self.fresh_condition()),
                        ..Default::default()
                    })
                };
                *world
                    .values
                    .get_mut(&local.local)
                    .expect("checked binding exists")
                    .at_mut(&local.path) = value;
            }
            Target::Tensor(place) => {
                self.consume(world, &value, span);
                let region = self.region(&place);
                if op != AssignOp::Assign {
                    self.read(world, place.root, region.clone(), span);
                }
                self.access(world, place.root, region.clone(), span, true, false);
                self.write_region(world, place.root, region);
            }
        }
    }

    fn import_integer(&mut self, transfer: &mut Transfer<'_>, value: IntExpr) -> IntExpr {
        let source = &transfer.source.arena;
        let symbols = &mut transfer.symbols;
        xfer::transfer_int(source, value, self.arena(), &mut |symbol, arena| {
            AnyExpr::Int(*symbols.entry(symbol).or_insert_with(|| {
                let s = arena.loop_binder().1;
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
    fn call(&mut self, world: &mut World, call: &ir::Call, arguments: &[Value], span: Span) {
        let env = self.checker.env;
        let contract = env.resolved.families[call.family.index()].contract;
        let Some(candidate) = call.candidates.iter().find(|c| c.definition == contract) else {
            self.checker.error(
                span,
                "cannot establish initialization: the fixed reference call contract is unavailable",
            );
            return;
        };
        let Some(Some(source)) = env.checked.get(contract.index()) else {
            self.checker.error(
                span,
                "recursive call cycle prevents construction of its initialization contract",
            );
            return;
        };
        let arguments = candidate
            .arg_order
            .iter()
            .map(|&i| arguments[i].clone())
            .collect::<Vec<_>>();
        let mut transfer = Transfer {
            source,
            arguments: &arguments,
            symbols: HashMap::new(),
        };
        for (&symbol, &value) in source
            .signature
            .shape_symbols
            .iter()
            .zip(&candidate.shape_args)
        {
            transfer.symbols.insert(symbol, value);
        }
        for (symbol, part) in &source.initialization.symbols {
            let value = match part {
                ParameterPart::Integer(p) => self.integer(argument(&arguments, p)),
                ParameterPart::Start(p) | ParameterPart::End(p) => match argument(&arguments, p) {
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
        for access in &source.initialization.accesses {
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
            world.accesses.push(Access {
                root: place.root,
                region,
                path,
                span,
                write: access.write,
                atomic: access.atomic,
            });
        }
        for requirement in &source.initialization.requirements {
            let Some(Value::Tensor(place)) = Some(argument(&arguments, &requirement.parameter))
            else {
                continue;
            };
            let path = self.import_path(&mut transfer, &requirement.path);
            let region = self.import_region(&mut transfer, &requirement.region);
            let region = self.map_region(region, place);
            let mut checked = world.clone();
            let before = checked.requirements.len();
            let deferred = checked.deferred.len();
            if !path
                .iter()
                .all(|(c, v)| self.assume(&mut checked.path, &mut checked.facts, c.clone(), *v))
            {
                continue;
            }
            self.require(&mut checked, place.root, region.clone(), span);
            world
                .requirements
                .extend(checked.requirements.into_iter().skip(before));
            world
                .deferred
                .extend(checked.deferred.into_iter().skip(deferred));
            if world.roots[&place.root].parameter.is_some() {
                let incoming = world.roots[&place.root]
                    .incoming
                    .clone()
                    .union(Region::Guard(path, Box::new(region)));
                world.roots.get_mut(&place.root).unwrap().incoming = incoming;
            }
        }
        for exit in &source.initialization.exits {
            let path = self.import_path(&mut transfer, &exit.path);
            for (parameter, written) in &exit.written {
                let Some(Value::Tensor(place)) = Some(argument(&arguments, parameter)) else {
                    continue;
                };
                let region = self.import_region(&mut transfer, written);
                let region = self.map_region(region, place);
                self.write_region(
                    world,
                    place.root,
                    Region::Guard(path.clone(), Box::new(region)),
                );
            }
        }
    }

    fn block(&mut self, mut worlds: Vec<World>, block: &mut ir::Block) -> Vec<World> {
        let mut returned = vec![];
        for statement in &mut block.statements {
            let mut next = vec![];
            for mut world in worlds {
                if world.returned {
                    returned.push(world);
                    continue;
                }
                match statement {
                    ir::Stmt::Let { pattern, value, .. } => {
                        let value = self.expression(&mut world, value);
                        self.bind(&mut world, pattern, value);
                        next.push(world);
                    }
                    ir::Stmt::Assign {
                        place, op, value, ..
                    } => {
                        // Checked addresses and right-hand values precede the
                        // store; element address expressions may themselves read.
                        let target = self.target(&mut world, place, *op);
                        let span = value.span;
                        let value = self.expression(&mut world, value);
                        self.assign(&mut world, target, *op, value, span);
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
                        ..
                    } => {
                        let value = self.expression(&mut world, condition);
                        let condition = self.boolean(&value);
                        for (truth, body) in [(true, &mut *then_body), (false, &mut *else_body)] {
                            let mut branch = world.clone();
                            if !self.assume(
                                &mut branch.path,
                                &mut branch.facts,
                                condition.clone(),
                                truth,
                            ) {
                                continue;
                            }
                            let outcomes = self.block(vec![branch], body);
                            for outcome in outcomes {
                                if outcome.returned {
                                    returned.push(outcome);
                                } else {
                                    next.push(outcome);
                                }
                            }
                        }
                    }
                    ir::Stmt::Loop {
                        kind,
                        binder,
                        start,
                        end,
                        body,
                        initialization,
                        ..
                    } => next.extend(self.loop_body(
                        world,
                        *kind,
                        *binder,
                        start,
                        end,
                        body,
                        initialization,
                    )),
                }
            }
            worlds = next;
        }
        if let ir::Terminator::Return(values) = &block.terminator {
            for world in &mut worlds {
                for expression in values {
                    let value = self.expression(world, expression);
                    self.consume(world, &value, expression.span);
                }
                world.returned = true;
            }
        }
        returned.extend(worlds);
        returned
    }
}

struct Transfer<'a> {
    source: &'a CheckedOutcome,
    arguments: &'a [Value],
    symbols: HashMap<SymbolId, IntExpr>,
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
            Region::Interval(start, end) => {
                symbols.extend(prove::symbols(self.arena_ref(), *start));
                symbols.extend(prove::symbols(self.arena_ref(), *end));
            }
            Region::Image { domain, address } => {
                symbols.extend(prove::symbols(self.arena_ref(), *address));
                for bound in domain {
                    symbols.insert(bound.symbol);
                    symbols.extend(prove::symbols(self.arena_ref(), bound.start));
                    symbols.extend(prove::symbols(self.arena_ref(), bound.end));
                }
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
    fn captured_symbols(&self, entry: &World, start: IntExpr, end: IntExpr) -> BTreeSet<SymbolId> {
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
        let mut symbols = self.checker.sig.shape_symbols.iter().copied().collect::<BTreeSet<_>>();
        symbols.extend(prove::symbols(self.arena_ref(), start));
        symbols.extend(prove::symbols(self.arena_ref(), end));
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
    fn may_share_root(&self, world: &World, left: usize, right: usize) -> bool {
        if left == right {
            return true;
        }
        let Some(a) = world
            .roots
            .get(&left)
            .and_then(|root| root.parameter.as_ref())
        else {
            return false;
        };
        let Some(b) = world
            .roots
            .get(&right)
            .and_then(|root| root.parameter.as_ref())
        else {
            return false;
        };
        self.checker.sig.aliases.iter().any(|&(x, y)| {
            (x == a.parameter && y == b.parameter) || (y == a.parameter && x == b.parameter)
        })
    }
    fn separated_regions(&mut self, left: Region, right: Region, facts: &prove::Facts) -> bool {
        let left = self.normalize(left, facts);
        let right = self.normalize(right, facts);
        match (left, right) {
            (Region::Empty, _) | (_, Region::Empty) => true,
            (Region::Union(parts), right) => parts
                .into_iter()
                .all(|part| self.separated_regions(part, right.clone(), facts)),
            (left, Region::Union(parts)) => parts
                .into_iter()
                .all(|part| self.separated_regions(left.clone(), part, facts)),
            (Region::Intersection(parts), right) => parts
                .into_iter()
                .any(|part| self.separated_regions(part, right.clone(), facts)),
            (left, Region::Intersection(parts)) => parts
                .into_iter()
                .any(|part| self.separated_regions(left.clone(), part, facts)),
            (Region::Guard(_, inner), right) => self.separated_regions(*inner, right, facts),
            (left, Region::Guard(_, inner)) => self.separated_regions(left, *inner, facts),
            (Region::Interval(a, b), Region::Interval(c, d)) => {
                self.le(facts, b, c) || self.le(facts, d, a)
            }
            // Unsupported images cannot be called independent. Normalization
            // above handles contiguous views and exact point selections.
            _ => false,
        }
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
            if !self.separated_regions(left.region.clone(), other_region.clone(), &orientation) {
                return false;
            }
        }
        true
    }
    fn independent_accesses(
        &mut self,
        entry: &World,
        outcomes: &[World],
        access_start: usize,
        symbol: SymbolId,
        start: IntExpr,
        end: IntExpr,
        facts: &prove::Facts,
    ) {
        let captured = self.captured_symbols(entry, start, end);
        let accesses = outcomes
            .iter()
            .flat_map(|outcome| outcome.accesses.iter().skip(access_start))
            .filter(|access| entry.roots.contains_key(&access.root))
            .cloned()
            .collect::<Vec<_>>();
        for (left_index, left) in accesses.iter().enumerate() {
            for right in accesses.iter().skip(left_index) {
                if (!left.write && !right.write) || (left.atomic && right.atomic) {
                    continue;
                }
                if !self.may_share_root(entry, left.root, right.root) {
                    continue;
                }
                // May-alias parameters have no checked relative base offset.
                // Their individual logical coordinates cannot establish
                // physical disjointness, even when the numbers differ.
                let mut left_region = left.clone();
                let mut right_region = right.clone();
                if left.root != right.root {
                    left_region.region = Region::Full;
                    right_region.region = Region::Full;
                }
                if !self.distinct_visit_accesses_separate(
                    &left_region,
                    &right_region,
                    symbol,
                    start,
                    end,
                    facts,
                    &captured,
                ) {
                    let root = &entry.roots[&left.root].name;
                    self.checker.error(
                        right.span,
                        format!(
                            "parallel for cannot establish independent visits: ordinary accesses to `{root}` at source offsets {} and {} may overlap across distinct visits",
                            left.span.start, right.span.start,
                        ),
                    );
                    return;
                }
            }
        }
    }
    fn rename_region(&mut self, region: &Region, map: &HashMap<SymbolId, IntExpr>) -> Region {
        match region {
            Region::Empty => Region::Empty,
            Region::Full => Region::Full,
            Region::Interval(a, b) => {
                Region::Interval(self.substitute(*a, &map), self.substitute(*b, &map))
            }
            Region::Image { domain, address } => Region::Image {
                domain: domain
                    .iter()
                    .map(|d| Bound {
                        symbol: self.renamed_bound(d.symbol, map),
                        start: self.substitute(d.start, &map),
                        end: self.substitute(d.end, &map),
                    })
                    .collect(),
                address: self.substitute(*address, &map),
            },
            Region::Union(parts) => Region::Union(
                parts
                    .iter()
                    .map(|p| self.rename_region(p, map))
                    .collect(),
            ),
            Region::Intersection(parts) => Region::Intersection(
                parts
                    .iter()
                    .map(|p| self.rename_region(p, map))
                    .collect(),
            ),
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
        let mut allowed = self.checker.sig.shape_symbols.clone();
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
        self.covered(&available, &required, &world.path, &world.facts)
    }
    fn loop_body(
        &mut self,
        mut entry: World,
        kind: ir::LoopKind,
        binder: ir::LocalId,
        start: &ir::Expr,
        end: &ir::Expr,
        body: &mut ir::Block,
        metadata: &mut crate::initialization::LoopInitialization,
    ) -> Vec<World> {
        let loop_span = start.span;
        let start_value = self.expression(&mut entry, start);
        let start = self.integer(&start_value);
        let end_value = self.expression(&mut entry, end);
        let end = self.integer(&end_value);
        let nonempty = Condition::Compare(BinaryOp::Lt, start, end);
        let mut result = vec![];
        let mut empty = entry.clone();
        if self.assume(&mut empty.path, &mut empty.facts, nonempty.clone(), false) {
            result.push(empty);
        }
        let mut iteration = entry.clone();
        if !self.assume(
            &mut iteration.path,
            &mut iteration.facts,
            nonempty.clone(),
            true,
        ) {
            return result;
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
            let mut allowed = self.checker.sig.shape_symbols.clone();
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
                iteration.roots.get_mut(&abstract_place.root).unwrap().written = written;
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
                        self.checker.error(loop_span,"cannot establish an inductive initialized region for this tensor carry");
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
        if kind == ir::LoopKind::Independent {
            self.independent_accesses(
                &entry,
                &outcomes,
                access_start,
                symbol,
                start,
                end,
                &access_facts,
            );
        }
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
                exit.accesses.push(Access {
                    root: access.root,
                    region: self.normalize(region, &entry.facts),
                    path: self.independent_path(&access.path, symbol),
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
                if self.covered(&available, &read.region, &read.path, &read.facts) {
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
            result.push(exit);
        }
        result
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
    fn close_contract(&mut self, contract: Contract) -> Contract {
        let mut allowed = self.checker.sig.shape_symbols.clone();
        allowed.extend(contract.symbols.iter().map(|(symbol, _)| *symbol));
        self.close_transfer(contract, &allowed)
    }
}
