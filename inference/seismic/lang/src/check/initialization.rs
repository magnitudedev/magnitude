//! Source-order definite initialization. This is a mandatory checking rule,
//! run before a checked definition is published. Regions describe actual
//! logical coordinates; no runtime storage or compiler-side certificate exists.
use super::{ir, prove, xfer, CheckedOutcome, Checker};
use crate::checked::DiagnosticRule;
use crate::expr::{
    AnyExpr, BoolExpr, ErasedScalarExpr, ExprArena, IntExpr, ScalarOperand, SymbolId, SymbolSort,
};
pub(super) use crate::initialization::InitializationContract as Contract;
use crate::initialization::RegionMapping;
use crate::initialization::{
    successful_paths, Bound, Condition, Exit, InitializationView, ParameterAccess, ParameterPart,
    ParameterPath, Path, Region, RegionOps, Requirement, VisitSeparation,
};
use crate::intrinsics::{AtomicOp, IndexSlot, PrimitiveId};
use crate::reference_math::ReferenceScalar;
use crate::span::Span;
use crate::syntax::ast::{BinaryOp, UnaryOp};
use crate::types::{DType, ValueType};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Optional region simplification and ordered-loop optimization use established
/// arithmetic. Required initialization and parallel separation retain full proof.
struct EstablishedRegionOps<'a, R: RegionOps>(&'a mut R);

impl<R: RegionOps> RegionOps for EstablishedRegionOps<'_, R> {
    fn arena(&mut self) -> &mut ExprArena {
        self.0.arena()
    }
    fn arena_ref(&self) -> &ExprArena {
        self.0.arena_ref()
    }
    fn fresh_variable(&mut self) -> SymbolId {
        self.0.fresh_variable()
    }
    fn le(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        let difference = self.arena().int_sub(b, a);
        prove::established_nonnegative(self.arena_ref(), facts, difference)
    }
    fn lt(&mut self, facts: &prove::Facts, a: IntExpr, b: IntExpr) -> bool {
        let difference = self.arena().int_sub(b, a);
        let one = self.arena().int(1);
        let difference = self.arena().int_sub(difference, one);
        prove::established_nonnegative(self.arena_ref(), facts, difference)
    }
}

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
#[derive(Clone, Debug, PartialEq)]
enum Scalar {
    Quantity(IntExpr),
    Word {
        dtype: DType,
        value: IntExpr,
        binding: Option<SymbolId>,
    },
    Boolean {
        value: ErasedScalarExpr,
        condition: Condition,
    },
    Floating(ErasedScalarExpr),
    Range {
        start: IntExpr,
        end: IntExpr,
    },
}
impl Scalar {
    fn integer(&self) -> Option<IntExpr> {
        match self {
            Self::Quantity(v) | Self::Word { value: v, .. } => Some(*v),
            _ => None,
        }
    }
    fn typed(&self) -> Option<ErasedScalarExpr> {
        match self {
            Self::Boolean { value, .. } | Self::Floating(value) => Some(*value),
            _ => None,
        }
    }
    fn condition(&self) -> Option<&Condition> {
        match self {
            Self::Boolean { condition, .. } => Some(condition),
            _ => None,
        }
    }
    fn range(&self) -> Option<(IntExpr, IntExpr)> {
        match self {
            Self::Range { start, end } => Some((*start, *end)),
            _ => None,
        }
    }
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
    Opaque,
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
    /// Exact may-access identity; diagnostic placement does not change an
    /// independence obligation. Borrowed keys retain the actual arena symbols.
    fn footprint(&self) -> (usize, &Region, &Path, bool, Option<AtomicOp>) {
        (self.root, &self.region, &self.path, self.write, self.atomic)
    }
    /// Two accesses the separation proof cannot tell apart.
    fn same_footprint(&self, other: &Access) -> bool {
        self.footprint() == other.footprint()
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
    for mut world in worlds {
        // Exit transfer drops this world's facts. Preserve their write meaning
        // before closing the contract over public parameter coordinates.
        let roots = world
            .roots
            .iter()
            .filter_map(|(id, root)| root.parameter.as_ref().map(|_| *id))
            .collect::<Vec<_>>();
        for (root, written) in owner.materialized_writes(&world, &roots) {
            world.roots.get_mut(&root).unwrap().written = written;
        }
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
            ValueType::Scalar(DType::Bool) => {
                let value = self.fresh_scalar(DType::Bool);
                let condition = self.fresh_condition();
                Value::Scalar(Scalar::Boolean { value, condition })
            }
            ValueType::Integer
            | ValueType::Scalar(DType::I32 | DType::U32)
            | ValueType::Index { .. } => {
                let (symbol, integer) = self.fresh_integer();
                world.facts.assume_type(self.arena(), symbol, ty);
                self.integer_value(ty, integer)
            }
            ValueType::Scalar(dtype) => Value::Scalar(Scalar::Floating(self.fresh_scalar(*dtype))),
            ValueType::Range { bound } => {
                let (s, start) = self.fresh_integer();
                let (e, end) = self.fresh_integer();
                let zero = self.arena().int(0);
                world.facts.set_range(s, zero, *bound);
                world.facts.set_range(e, start, *bound);
                Value::Scalar(Scalar::Range { start, end })
            }
            _ => Value::Opaque,
        }
    }
    fn fresh_scalar(&mut self, dtype: DType) -> ErasedScalarExpr {
        let symbol = self.arena().checker_variable(SymbolSort::Scalar(dtype));
        self.arena().scalar_symbol_erased(symbol)
    }
    fn integer_value(&self, ty: &ValueType, value: IntExpr) -> Value {
        Value::Scalar(match ty {
            ValueType::Scalar(dtype @ (DType::I32 | DType::U32)) => Scalar::Word {
                dtype: *dtype,
                value,
                binding: None,
            },
            _ => Scalar::Quantity(value),
        })
    }
    fn boolean_value(&mut self, condition: Condition) -> Value {
        let value = self.fresh_scalar(DType::Bool);
        Value::Scalar(Scalar::Boolean { value, condition })
    }
    fn literal_value(&mut self, literal: ReferenceScalar) -> Value {
        match literal {
            ReferenceScalar::I32(v) => {
                let value = self.arena().int(i64::from(v));
                self.integer_value(&ValueType::Scalar(DType::I32), value)
            }
            ReferenceScalar::U32(v) => {
                let value = self.arena().int(i64::from(v));
                self.integer_value(&ValueType::Scalar(DType::U32), value)
            }
            ReferenceScalar::Bool(v) => {
                let value = self.arena().scalar_const_erased(literal);
                Value::Scalar(Scalar::Boolean {
                    value,
                    condition: Condition::Constant(v),
                })
            }
            _ => Value::Scalar(Scalar::Floating(self.arena().scalar_const_erased(literal))),
        }
    }
    /// An opaque producer owns one value at this source position. Repeated
    /// uses retain that identity; another producer never inherits it.
    fn source_result(
        &mut self,
        world: &mut World,
        ty: &ValueType,
        source: &ir::SourceValue,
    ) -> Value {
        match (ty, source) {
            (ValueType::Tuple(types), ir::SourceValue::Tuple(sources)) => {
                assert_eq!(types.len(), sources.len(), "checked source result product");
                Value::Tuple(
                    types
                        .iter()
                        .zip(sources)
                        .map(|(ty, source)| self.source_result(world, ty, source))
                        .collect(),
                )
            }
            (
                ValueType::Scalar(dtype @ (DType::I32 | DType::U32)),
                ir::SourceValue::Scalar(source),
            ) => {
                let binding = source
                    .binding()
                    .expect("opaque word result owns its publication");
                world.facts.assume_type(self.arena(), binding, ty);
                let value = self.arena().int_symbol(binding);
                Value::Scalar(Scalar::Word {
                    dtype: *dtype,
                    value,
                    binding: Some(binding),
                })
            }
            (ValueType::Integer | ValueType::Index { .. }, ir::SourceValue::Scalar(source)) => {
                let value = if let Some(binding) = source.binding() {
                    world.facts.assume_type(self.arena(), binding, ty);
                    self.arena().int_symbol(binding)
                } else {
                    source
                        .integer()
                        .expect("quantity result owns its exact value")
                };
                Value::Scalar(Scalar::Quantity(value))
            }
            (ValueType::Scalar(DType::Bool), ir::SourceValue::Scalar(source)) => {
                let value = source
                    .scalar()
                    .expect("Boolean result owns an actual typed leaf");
                let condition = self.fresh_condition();
                Value::Scalar(Scalar::Boolean { value, condition })
            }
            (ValueType::Scalar(_), ir::SourceValue::Scalar(source)) => {
                Value::Scalar(Scalar::Floating(
                    source
                        .scalar()
                        .expect("floating result owns an actual typed leaf"),
                ))
            }
            (ValueType::Range { bound }, ir::SourceValue::Range { start, end }) => {
                let start_symbol = start.binding().expect("range result start publication");
                let end_symbol = end.binding().expect("range result end publication");
                let start = self.arena().int_symbol(start_symbol);
                let end = self.arena().int_symbol(end_symbol);
                let zero = self.arena().int(0);
                world.facts.set_range(start_symbol, zero, *bound);
                world.facts.set_range(end_symbol, start, *bound);
                Value::Scalar(Scalar::Range { start, end })
            }
            (_, ir::SourceValue::Untracked) => self.result(world, ty, true),
            _ => panic!("opaque result preserves its checked source product"),
        }
    }
    fn expression_result(&mut self, world: &mut World, expression: &ir::Expr) -> Value {
        self.source_result(world, &expression.ty, &expression.value)
    }

    fn scalar_operation(
        &mut self,
        world: &mut World,
        result_type: &ValueType,
        source: &ir::SourceValue,
        id: &PrimitiveId,
        operands: &[ir::Expr],
        values: &[Value],
    ) -> Value {
        let actual = operands
            .iter()
            .zip(values)
            .map(|(operand, value)| super::scalar_quantity::Operand {
                ty: &operand.ty,
                value: match value {
                    Value::Scalar(scalar) => scalar.integer(),
                    _ => None,
                },
                scalar: match value {
                    Value::Scalar(scalar) => scalar.typed(),
                    _ => None,
                },
                literal: match &operand.kind {
                    ir::ExprKind::Literal(value) => Some(*value),
                    _ => None,
                },
            })
            .collect::<Vec<_>>();
        let integer = super::scalar_quantity::exact(self.arena(), id, result_type, &actual);
        match integer {
            Some(integer) => {
                world.facts.assume_scalar_defined(self.arena(), integer);
                if matches!(
                    self.arena_ref().view(integer.into()),
                    crate::expr::NodeView::ScalarInteger { .. }
                ) {
                    let defined = self.arena().scalar_integer_defined(integer);
                    let binders = self.arena_ref().free_symbols(defined.into());
                    assert!(
                        self.assume(
                            &mut world.path,
                            &mut world.facts,
                            Condition::Actual(defined, binders),
                            true
                        ),
                        "a reached scalar operation has a successful continuation"
                    );
                }
                self.integer_value(result_type, integer)
            }
            None => self.source_result(world, result_type, source),
        }
    }

    fn quantity_word(&mut self, ty: &ValueType, quantity: IntExpr) -> Value {
        let ValueType::Scalar(dtype @ (DType::I32 | DType::U32)) = ty else {
            return self.integer_value(ty, quantity);
        };
        let value = self.arena().int_to_word(quantity, *dtype);
        self.integer_value(ty, value)
    }
    fn parameter_source(&self, local: ir::LocalId, fields: &[usize]) -> &ir::SourceValue {
        let mut source = &self.checker.locals[local.index()].value;
        for field in fields {
            let ir::SourceValue::Tuple(parts) = source else {
                panic!("checked parameter tuple path")
            };
            source = &parts[*field];
        }
        source
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
            ValueType::Integer
            | ValueType::Index { .. }
            | ValueType::Scalar(DType::I32 | DType::U32) => {
                let (symbol, value) =
                    if let Some(s) = self.parameter_source(local, &path.fields).binding() {
                        (s, self.arena().int_symbol(s))
                    } else {
                        self.fresh_integer()
                    };
                self.symbols.push((symbol, ParameterPart::Integer(path)));
                self.integer_value(ty, value)
            }
            ValueType::Range { bound } => {
                let bindings = match self.parameter_source(local, &path.fields) {
                    ir::SourceValue::Range { start, end } => (start.binding(), end.binding()),
                    _ => (None, None),
                };
                let (s, start) = bindings
                    .0
                    .map(|s| (s, self.arena().int_symbol(s)))
                    .unwrap_or_else(|| self.fresh_integer());
                let (e, end) = bindings
                    .1
                    .map(|e| (e, self.arena().int_symbol(e)))
                    .unwrap_or_else(|| self.fresh_integer());
                self.symbols.extend([
                    (s, ParameterPart::Start(path.clone())),
                    (e, ParameterPart::End(path)),
                ]);
                let zero = self.arena().int(0);
                world.facts.set_range(s, zero, *bound);
                world.facts.set_range(e, start, *bound);
                Value::Scalar(Scalar::Range { start, end })
            }
            ValueType::Scalar(dtype) => {
                let symbol = self
                    .parameter_source(local, &path.fields)
                    .binding()
                    .unwrap_or_else(|| self.arena().checker_variable(SymbolSort::Scalar(*dtype)));
                let value = self.arena().scalar_symbol_erased(symbol);
                self.symbols
                    .push((symbol, ParameterPart::Scalar(path.clone())));
                Value::Scalar(if *dtype == DType::Bool {
                    Scalar::Boolean {
                        value,
                        condition: Condition::Parameter(path),
                    }
                } else {
                    Scalar::Floating(value)
                })
            }
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
        if !world.roots.contains_key(&root) {
            return;
        }
        // Capture only this effect while its source-position facts are live.
        // Earlier effects already own their reached meaning and guards.
        let region = EstablishedRegionOps(self).normalize(region, &world.facts);
        // Outside loops no later exit materializes this root, so a write of
        // every element is recorded as the whole root now, exactly as a loop
        // exit or join would promote it.
        let whole = world.defer_depth == 0
            && !matches!(world.roots[&root].written, Region::Full)
            && EstablishedRegionOps(self).covered(
                &region,
                &Region::Full,
                &world.path,
                &world.facts,
                &world.roots[&root].axes,
            );
        let data = world.roots.get_mut(&root).unwrap();
        let written = std::mem::replace(&mut data.written, Region::Empty);
        data.written = if whole {
            Region::Full
        } else {
            written.union(region)
        };
    }
    /// Simplify accumulated writes before their fact world or coordinate scope
    /// is projected away. Whole-root promotion is optional; required reads
    /// retain their full authoritative coverage proof.
    fn materialized_written(&mut self, world: &World, root: usize) -> Region {
        self.materialized_writes(world, &[root])
            .remove(&root)
            .unwrap()
    }
    /// Materialize one world's separate roots before its private facts escape.
    /// Root geometry owns promotion; only the normalization scopes are shared.
    fn materialized_writes(&mut self, world: &World, roots: &[usize]) -> HashMap<usize, Region> {
        let regions = roots
            .iter()
            .map(|root| world.roots[root].written.clone())
            .collect();
        let regions = EstablishedRegionOps(self).normalize_many(regions, &world.facts);
        roots
            .iter()
            .copied()
            .zip(regions)
            .map(|(root, written)| {
                let written = if EstablishedRegionOps(self).covered(
                    &written,
                    &Region::Full,
                    &world.path,
                    &world.facts,
                    &world.roots[&root].axes,
                ) {
                    Region::Full
                } else {
                    written
                };
                (root, written)
            })
            .collect()
    }
    fn require(&mut self, world: &mut World, root: usize, region: Region, span: Span) {
        let data = world.roots[&root].clone();
        let axes = &data.axes;
        let zero = self.arena().int(0);
        if axes
            .iter()
            .any(|axis| self.optional_le(&world.facts, *axis, zero))
        {
            return;
        }
        let available = data.written.clone().union(data.incoming.clone());
        // Without an incoming assumption, a loop read can retain its exact
        // source-position state until the loop's authoritative resolution. An
        // unsuccessful optional elimination is not a failed initialization
        // proof. Abstract incoming sets (including trial carry invariants)
        // still follow the full requirement re-export path below.
        if world.defer_depth > 0 && matches!(data.incoming, Region::Empty) {
            if EstablishedRegionOps(self).covered(
                &available,
                &region,
                &world.path,
                &world.facts,
                axes,
            ) || EstablishedRegionOps(self).covered(
                &Region::Empty,
                &Region::Full,
                &world.path,
                &world.facts,
                axes,
            ) {
                return;
            }
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
        // Outside loops, an input read is exported as a call requirement
        // unless this body's own writes cover it or it is vacuous; whether
        // earlier requirements already assumed it changes neither outcome.
        if world.defer_depth == 0 {
            if let Some(parameter) = data.parameter.clone() {
                if self.covered(&data.written, &region, &world.path, &world.facts, axes)
                    || self.covered(
                        &Region::Empty,
                        &Region::Full,
                        &world.path,
                        &world.facts,
                        axes,
                    )
                {
                    return;
                }
                world.requirements.push(Requirement {
                    parameter,
                    region: region.clone(),
                    path: world.path.clone(),
                    span,
                });
                world.roots.get_mut(&root).unwrap().incoming = data.incoming.union(region);
                return;
            }
        }
        if self.covered(&available, &region, &world.path, &world.facts, axes) {
            // Reads supplied by an abstract incoming parameter remain call
            // requirements. Invariant inference may provisionally supply that
            // incoming set, but cannot turn it into a produced write.
            if let Some(parameter) = data.parameter.clone() {
                if !self.covered(&data.written, &region, &world.path, &world.facts, axes) {
                    if self.covered(
                        &Region::Empty,
                        &Region::Full,
                        &world.path,
                        &world.facts,
                        axes,
                    ) {
                        return;
                    }
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
        // Required vacuity remains authoritative before diagnostics or effects.
        if self.covered(
            &Region::Empty,
            &Region::Full,
            &world.path,
            &world.facts,
            axes,
        ) {
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
        let mut regions = self
            .normalize_many(vec![available, region], &world.facts)
            .into_iter();
        let available = regions.next().unwrap();
        let region = regions.next().unwrap();
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
            Value::Scalar(Scalar::Word {
                binding: Some(binding),
                ..
            }) => self.arena().int_symbol(*binding),
            Value::Scalar(scalar) => scalar.integer().unwrap_or_else(|| self.fresh_integer().1),
            _ => self.fresh_integer().1,
        }
    }
    fn boolean(&mut self, value: &Value) -> Condition {
        match value {
            Value::Scalar(scalar) => scalar
                .condition()
                .cloned()
                .unwrap_or_else(|| self.fresh_condition()),
            _ => self.fresh_condition(),
        }
    }
    /// Mutable because a call keeps only the alternatives whose contracts
    /// apply at it.
    fn expression(&mut self, world: &mut World, expression: &mut ir::Expr) -> Value {
        let mut value = self.expression_value(world, expression);
        if let Value::Scalar(scalar) = &mut value {
            if let Some(actual) = scalar.integer() {
                if let Some(binding) = expression.binding() {
                    assert!(
                        world.facts.define_scalar(self.arena_ref(), binding, actual),
                        "reached source occurrence is immutable and acyclic"
                    );
                    world
                        .facts
                        .assume_type(self.arena(), binding, &expression.ty);
                    if let Scalar::Word {
                        binding: actual_binding,
                        ..
                    } = scalar
                    {
                        *actual_binding = Some(binding);
                    }
                }
            }
            let produces_scalar = matches!(
                &expression.kind,
                ir::ExprKind::Primitive {
                    id: PrimitiveId::Unary(_)
                        | PrimitiveId::Binary(_)
                        | PrimitiveId::Cast(_)
                        | PrimitiveId::Math(_)
                        | PrimitiveId::Bitcast(_),
                    ..
                }
            );
            if let Some(actual) = expression.scalar().filter(|_| produces_scalar) {
                match scalar {
                    Scalar::Boolean { value, .. } | Scalar::Floating(value) => *value = actual,
                    _ => {}
                }
            }
        }
        value
    }
    fn expression_value(&mut self, world: &mut World, expression: &mut ir::Expr) -> Value {
        match &mut expression.kind {
            ir::ExprKind::Local(local) => world
                .values
                .get(local)
                .cloned()
                .expect("checked local has a source-order value binding"),
            ir::ExprKind::Literal(value) => self.literal_value(*value),
            ir::ExprKind::Dimension(_) => Value::Scalar(Scalar::Quantity(
                expression
                    .integer()
                    .expect("checked dimension has a quantity"),
            )),
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
                        match extent {
                            Some(extent) => self.quantity_word(&expression.ty, extent),
                            None => self.expression_result(world, expression),
                        }
                    }
                    PrimitiveId::SliceView { indices } => {
                        let Some(Value::Tensor(place)) = values.first() else {
                            return self.result(world, &expression.ty, true);
                        };
                        let mut rest = values[1..].iter();
                        let mut selected = vec![];
                        for (axis, index) in indices.iter().enumerate() {
                            let (selection, checks) = match index {
                                IndexSlot::Point { check } => (
                                    (Some(self.integer(rest.next().unwrap())), None, true),
                                    (*check, false, false),
                                ),
                                IndexSlot::Range {
                                    start,
                                    end,
                                    check_start,
                                    check_order,
                                    check_end,
                                    ..
                                } => (
                                    (
                                        start.then(|| self.integer(rest.next().unwrap())),
                                        end.then(|| self.integer(rest.next().unwrap())),
                                        false,
                                    ),
                                    (*check_start, *check_order, *check_end),
                                ),
                                IndexSlot::Full => ((None, None, false), (false, false, false)),
                            };
                            self.assume_access_bounds(world, place.axes[axis], &selection, checks);
                            selected.push(selection);
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
                    PrimitiveId::ElementRead { checks }
                    | PrimitiveId::QuantizedComponentRead { checks, .. } => {
                        if let Some(Value::Tensor(place)) = values.first() {
                            let selected = values[1..]
                                .iter()
                                .map(|v| (Some(self.integer(v)), None, true))
                                .collect::<Vec<_>>();
                            for (axis, selection) in selected.iter().enumerate() {
                                self.assume_access_bounds(
                                    world,
                                    place.axes[axis],
                                    selection,
                                    (checks[axis], false, false),
                                );
                            }
                            let place = self.select(place, &selected);
                            let region = self.region(&place);
                            self.read(world, place.root, region, expression.span);
                        }
                        self.expression_result(world, expression)
                    }
                    PrimitiveId::RangeMake => Value::Scalar(Scalar::Range {
                        start: self.integer(&values[0]),
                        end: self.integer(&values[1]),
                    }),
                    PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                        let integer = match values.first() {
                            Some(Value::Scalar(Scalar::Range { start, end })) => {
                                Some(if matches!(id, PrimitiveId::RangeStart) {
                                    *start
                                } else {
                                    *end
                                })
                            }
                            _ => None,
                        };
                        match integer {
                            Some(integer) => self.quantity_word(&expression.ty, integer),
                            None => self.expression_result(world, expression),
                        }
                    }
                    PrimitiveId::Unary(op) => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        if *op == UnaryOp::Not && !matches!(expression.ty, ValueType::Tensor(_)) {
                            {
                                let condition = Condition::Not(Box::new(self.boolean(&values[0])));
                                self.boolean_value(condition)
                            }
                        } else {
                            self.scalar_operation(
                                world,
                                &expression.ty,
                                &expression.value,
                                id,
                                operands,
                                &values,
                            )
                        }
                    }
                    PrimitiveId::Binary(op) => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        if matches!(expression.ty, ValueType::Tensor(_)) {
                            return self.result(world, &expression.ty, true);
                        }
                        if matches!(op, BinaryOp::And | BinaryOp::Or) {
                            let a = Box::new(self.boolean(&values[0]));
                            let b = Box::new(self.boolean(&values[1]));
                            self.boolean_value(if *op == BinaryOp::And {
                                Condition::And(a, b)
                            } else {
                                Condition::Or(a, b)
                            })
                        } else if matches!(
                            op,
                            BinaryOp::Eq
                                | BinaryOp::Ne
                                | BinaryOp::Lt
                                | BinaryOp::Le
                                | BinaryOp::Gt
                                | BinaryOp::Ge
                        ) {
                            let condition = match (&values[0], &values[1]) {
                                (Value::Scalar(a), Value::Scalar(b)) => {
                                    match (a.integer(), b.integer()) {
                                        (Some(_), Some(_)) => Condition::Compare(
                                            *op,
                                            self.integer(&values[0]),
                                            self.integer(&values[1]),
                                        ),
                                        _ => self.fresh_condition(),
                                    }
                                }
                                _ => self.fresh_condition(),
                            };
                            self.boolean_value(condition)
                        } else {
                            self.scalar_operation(
                                world,
                                &expression.ty,
                                &expression.value,
                                id,
                                operands,
                                &values,
                            )
                        }
                    }
                    PrimitiveId::Constant(value) => self.literal_value(*value),
                    PrimitiveId::Symbolic(value) => self.integer_value(&expression.ty, *value),
                    PrimitiveId::Cast(_) | PrimitiveId::Math(_) | PrimitiveId::Bitcast(_) => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        self.scalar_operation(
                            world,
                            &expression.ty,
                            &expression.value,
                            id,
                            operands,
                            &values,
                        )
                    }
                    PrimitiveId::Copy => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        if matches!(values.first(), Some(Value::Scalar(_))) {
                            values[0].clone()
                        } else {
                            self.expression_result(world, expression)
                        }
                    }
                    PrimitiveId::Select => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        if !matches!(
                            expression.ty,
                            ValueType::Scalar(_) | ValueType::Integer | ValueType::Index { .. }
                        ) {
                            return self.expression_result(world, expression);
                        }
                        match (&values[0], &values[1], &values[2]) {
                            (
                                Value::Scalar(Scalar::Boolean {
                                    condition: Condition::Constant(selected),
                                    ..
                                }),
                                a,
                                b,
                            ) => {
                                if *selected {
                                    a.clone()
                                } else {
                                    b.clone()
                                }
                            }
                            (_, a, b) if matches!(a, Value::Scalar(_)) && a == b => a.clone(),
                            _ => self.scalar_operation(
                                world,
                                &expression.ty,
                                &expression.value,
                                id,
                                operands,
                                &values,
                            ),
                        }
                    }
                    PrimitiveId::RepresentationConvert(_) | PrimitiveId::Reduce { .. } => {
                        values
                            .iter()
                            .for_each(|v| self.consume(world, v, expression.span));
                        self.expression_result(world, expression)
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
                let indices = self.selection(
                    world,
                    indices,
                    place.as_ref().map(|place| place.axes.as_slice()),
                );
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
                self.expression_result(world, expression)
            }
            // L31: the quantity is the word's value.
            ir::ExprKind::IndexPosition(word) => {
                let actual = self.expression(world, word);
                match actual {
                    Value::Scalar(scalar) => match scalar.integer() {
                        Some(integer) => {
                            let quantity = word
                                .binding()
                                .map(|binding| self.arena().int_symbol(binding))
                                .unwrap_or(integer);
                            Value::Scalar(Scalar::Quantity(quantity))
                        }
                        None => self.expression_result(world, expression),
                    },
                    _ => self.expression_result(world, expression),
                }
            }
            ir::ExprKind::Call { call, args } => {
                for (_, value) in &mut call.seeds {
                    self.expression(world, value);
                }
                let values = args
                    .iter_mut()
                    .map(|e| self.expression(world, e))
                    .collect::<Vec<_>>();
                self.call(world, call, &values, expression.span);
                self.expression_result(world, expression)
            }
        }
    }
    /// A frontend occurrence/version and this traversal's reached value name
    /// the same runtime quantity. The equality is world-local, not a fact
    /// imported from a later checker state or a new arithmetic assumption.
    fn bind_actual_value(
        &mut self,
        facts: &mut prove::Facts,
        source: &ir::SourceValue,
        actual: &mut Value,
    ) {
        match (source, actual) {
            (ir::SourceValue::Scalar(source), Value::Scalar(actual)) => {
                if let (Some(binding), Some(meaning)) = (source.binding(), actual.integer()) {
                    assert!(
                        facts.define_scalar(self.arena_ref(), binding, meaning),
                        "reached source occurrence is immutable and acyclic"
                    );
                    match actual {
                        Scalar::Word {
                            dtype,
                            binding: actual_binding,
                            ..
                        } => {
                            facts.assume_type(self.arena(), binding, &ValueType::Scalar(*dtype));
                            *actual_binding = Some(binding);
                        }
                        Scalar::Quantity(_) => {}
                        _ => unreachable!("integer meaning belongs to word or quantity"),
                    }
                }
                // Actual typed scalar leaves retain copy/capture identity. A
                // version's publication is an alias, not another opaque read.
            }
            (ir::SourceValue::Tuple(sources), Value::Tuple(actuals)) => {
                assert_eq!(sources.len(), actuals.len(), "checked source product shape");
                for (source, actual) in sources.iter().zip(actuals) {
                    self.bind_actual_value(facts, source, actual);
                }
            }
            (
                ir::SourceValue::Range {
                    start: source_start,
                    end: source_end,
                },
                Value::Scalar(Scalar::Range { start, end }),
            ) => {
                for (source, meaning) in [(source_start, *start), (source_end, *end)] {
                    if let Some(binding) = source.binding() {
                        assert!(
                            facts.define_scalar(self.arena_ref(), binding, meaning),
                            "reached source occurrence is immutable and acyclic"
                        );
                    }
                }
            }
            (ir::SourceValue::Untracked, _) => {}
            _ => panic!("checked source value and reached value have one product shape"),
        }
    }
    fn bind_value_bindings(
        &mut self,
        world: &mut World,
        versions: &[(ir::LocalId, ir::SourceValue)],
    ) {
        for (local, source) in versions {
            if let Some(actual) = world.values.get_mut(local) {
                self.bind_actual_value(&mut world.facts, source, actual);
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
    /// The successful continuation of an emitted access check owns these
    /// bounds. They describe actual view axes and reached source values, not
    /// a waiver of initialization or an assumed static-width equality.
    fn assume_access_bounds(
        &mut self,
        world: &mut World,
        extent: IntExpr,
        selection: &(Option<IntExpr>, Option<IntExpr>, bool),
        checks: (bool, bool, bool),
    ) {
        let zero = self.arena().int(0);
        let start = selection.0.unwrap_or(zero);
        if selection.2 {
            if checks.0 {
                world.facts.assume_nonnegative(self.arena(), start);
                let end = self.arena().int_sub(extent, start);
                let one = self.arena().int(1);
                let last = self.arena().int_sub(end, one);
                world.facts.assume_nonnegative(self.arena(), last);
            }
        } else {
            let end = selection.1.unwrap_or(extent);
            if checks.0 {
                world.facts.assume_nonnegative(self.arena(), start);
            }
            if checks.1 {
                let width = self.arena().int_sub(end, start);
                world.facts.assume_nonnegative(self.arena(), width);
            }
            if checks.2 {
                let tail = self.arena().int_sub(extent, end);
                world.facts.assume_nonnegative(self.arena(), tail);
            }
        }
    }
    /// Target/atomic checks execute after each selected index, before the next
    /// index or RHS. Primitive accesses instead evaluate all operands first.
    fn selection(
        &mut self,
        world: &mut World,
        indices: &mut [ir::Index],
        axes: Option<&[IntExpr]>,
    ) -> Vec<(Option<IntExpr>, Option<IntExpr>, bool)> {
        let mut selected = vec![];
        for (axis, index) in indices.iter_mut().enumerate() {
            let (selection, checks) = match index {
                ir::Index::Point {
                    value,
                    runtime_check,
                } => {
                    let value = self.expression(world, value);
                    (
                        (Some(self.integer(&value)), None, true),
                        (*runtime_check, false, false),
                    )
                }
                ir::Index::Range {
                    start,
                    end,
                    check_start,
                    check_order,
                    check_end,
                    ..
                } => {
                    let start = start.as_mut().map(|e| {
                        let v = self.expression(world, e);
                        self.integer(&v)
                    });
                    let end = end.as_mut().map(|e| {
                        let v = self.expression(world, e);
                        self.integer(&v)
                    });
                    (
                        (start, end, false),
                        (*check_start, *check_order, *check_end),
                    )
                }
                ir::Index::Full => ((None, None, false), (false, false, false)),
            };
            if let Some(axes) = axes {
                self.assume_access_bounds(world, axes[axis], &selection, checks);
            }
            selected.push(selection);
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
                let selected = self.selection(world, indices, Some(&place.axes));
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
            *symbols.entry(symbol).or_insert_with(|| {
                let sort = source.symbol_sort(symbol);
                let s = arena.checker_variable(sort);
                match sort {
                    SymbolSort::Int => AnyExpr::Int(arena.int_symbol(s)),
                    SymbolSort::Nat => AnyExpr::Nat(arena.nat_symbol(s)),
                    SymbolSort::Scalar(_) => AnyExpr::Scalar(arena.scalar_symbol_erased(s)),
                }
            })
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
        // Repeated requirements of one site are one obligation.
        let mut requirements = Vec::<(usize, Path, Region)>::with_capacity(site.requirements.len());
        for requirement in site.requirements {
            if !requirements.contains(&requirement) {
                requirements.push(requirement);
            }
        }
        for (root, path, region) in requirements {
            // A requirement reads only its own root under the call's path;
            // check it in that slice of the world and keep its new effects.
            let mut checked = World {
                values: HashMap::new(),
                roots: HashMap::from([(root, world.roots[&root].clone())]),
                path: world.path.clone(),
                facts: world.facts.clone(),
                requirements: vec![],
                deferred: vec![],
                accesses: vec![],
                defer_depth: world.defer_depth,
            };
            if !path
                .iter()
                .all(|(c, v)| self.assume(&mut checked.path, &mut checked.facts, c.clone(), *v))
            {
                continue;
            }
            self.require(&mut checked, root, region.clone(), span);
            world.requirements.extend(checked.requirements);
            world.deferred.extend(checked.deferred);
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
            // A call that writes every element leaves the whole root written.
            let whole = world
                .roots
                .get(&root)
                .is_some_and(|data| !matches!(data.written, Region::Full))
                && EstablishedRegionOps(self).covered(
                    &world.roots[&root].written,
                    &Region::Full,
                    &world.path,
                    &world.facts,
                    &world.roots[&root].axes,
                );
            if whole {
                world.roots.get_mut(&root).unwrap().written = Region::Full;
            }
        }
        let returned = successful_paths(&site.exits);
        if !self.assume(&mut world.path, &mut world.facts, returned.clone(), true) {
            world.path.push((returned, true));
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
            transfer
                .symbols
                .insert(dimension.symbol, AnyExpr::Int(value));
        }
        for (symbol, part) in &contract.symbols {
            let value = match part {
                ParameterPart::Integer(p) => AnyExpr::Int(self.integer(argument(arguments, p))),
                ParameterPart::Scalar(p) => match argument(arguments, p) {
                    Value::Scalar(scalar) => AnyExpr::Scalar(
                        scalar
                            .typed()
                            .expect("checked actual scalar parameter has its typed leaf"),
                    ),
                    _ => panic!("checked scalar parameter is not a scalar"),
                },
                ParameterPart::Start(p) | ParameterPart::End(p) => {
                    let integer = match argument(arguments, p) {
                        Value::Scalar(Scalar::Range { start, end }) => {
                            if matches!(part, ParameterPart::Start(_)) {
                                *start
                            } else {
                                *end
                            }
                        }
                        _ => self.fresh_integer().1,
                    };
                    AnyExpr::Int(integer)
                }
            };
            transfer.symbols.insert(*symbol, value);
        }
        let mut site = CallSite {
            accesses: vec![],
            requirements: vec![],
            writes: vec![],
            exits: vec![],
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
            site.exits.push(path.clone());
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
                    ir::Stmt::Let {
                        pattern,
                        value,
                        value_bindings,
                    } => {
                        let value = self.expression(&mut world, value);
                        self.bind(&mut world, pattern, value);
                        self.bind_value_bindings(&mut world, value_bindings);
                        next.push(world);
                    }
                    ir::Stmt::Assign {
                        place,
                        value,
                        value_bindings,
                        ..
                    } => {
                        // Checked addresses and right-hand values precede the
                        // store; element address expressions may themselves read.
                        let target = self.target(&mut world, place);
                        let span = value.span;
                        let value = self.expression(&mut world, value);
                        self.assign(&mut world, target, value, span);
                        self.bind_value_bindings(&mut world, value_bindings);
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
                        capture_bindings,
                        join_bindings,
                        ..
                    } => {
                        self.bind_value_bindings(&mut world, capture_bindings);
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
                            join_bindings,
                        ));
                    }
                    ir::Stmt::Loop {
                        kind,
                        binder,
                        start,
                        end,
                        body,
                        value_versions,
                        initialization,
                        separation,
                    } => {
                        let outcomes = self.loop_body(
                            world,
                            *kind,
                            *binder,
                            start,
                            end,
                            body,
                            value_versions,
                            initialization,
                            separation,
                        );
                        let exit_bindings = value_versions
                            .iter()
                            .map(|(local, _, exit)| (*local, exit.clone()))
                            .collect::<Vec<_>>();
                        for mut outcome in outcomes {
                            self.bind_value_bindings(&mut outcome, &exit_bindings);
                            next.push(outcome);
                        }
                    }
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
        join_bindings: &[(ir::LocalId, ir::SourceValue)],
    ) -> Vec<World> {
        if let ([then], [els]) = (then_worlds.as_slice(), else_worlds.as_slice()) {
            if let Some(mut joined) = self.join_worlds(entry, condition, then, els, join_bindings) {
                self.bind_value_bindings(&mut joined, join_bindings);
                return vec![joined];
            }
        }
        then_worlds
            .into_iter()
            .chain(else_worlds)
            .map(|mut world| {
                // Locals declared inside an arm end with it, as in one
                // merged world.
                world
                    .values
                    .retain(|local, _| entry.values.contains_key(local));
                self.bind_value_bindings(&mut world, join_bindings);
                world
            })
            .collect()
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
        join_bindings: &[(ir::LocalId, ir::SourceValue)],
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
            let symbol = join_bindings
                .iter()
                .find(|(changed, _)| changed == local)
                .map(|(_, value)| value);
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
        // A root neither arm wrote still holds the entry's region; it is
        // materialized once, in the entry world both arms extend.
        let unwritten = entry
            .roots
            .iter()
            .filter(|(id, before)| {
                [then, els].iter().all(|arm| {
                    arm.roots
                        .get(id)
                        .is_some_and(|after| after.written == before.written)
                })
            })
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let common = self.materialized_writes(entry, &unwritten);
        let then_roots = then
            .roots
            .keys()
            .copied()
            .filter(|id| !common.contains_key(id))
            .collect::<Vec<_>>();
        let else_roots = els
            .roots
            .keys()
            .copied()
            .filter(|id| !common.contains_key(id))
            .collect::<Vec<_>>();
        let mut then_written = self.materialized_writes(then, &then_roots);
        let mut else_written = self.materialized_writes(els, &else_roots);
        then_written.extend(common.clone());
        else_written.extend(common);
        for (id, root) in &then.roots {
            let root = match els.roots.get(id) {
                Some(other) => Root {
                    // Element overwrites cannot revoke writes established at
                    // branch entry. Arm normalization may represent those
                    // regions differently, so retain their original authority
                    // explicitly rather than relying on syntactic intersection.
                    written: entry
                        .roots
                        .get(id)
                        .map_or(Region::Empty, |before| before.written.clone())
                        .union(Region::branch(
                            condition,
                            then_written[id].clone(),
                            else_written[id].clone(),
                        )),
                    incoming: Region::branch(
                        condition,
                        root.incoming.clone(),
                        other.incoming.clone(),
                    ),
                    ..root.clone()
                },
                None => Root {
                    written: then_written[id].clone(),
                    ..root.clone()
                },
            };
            joined.roots.insert(*id, root);
        }
        for (id, root) in &els.roots {
            if !joined.roots.contains_key(id) {
                joined.roots.insert(
                    *id,
                    Root {
                        written: else_written[id].clone(),
                        ..root.clone()
                    },
                );
            }
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
    /// Join one actual source product, retaining each leaf's publication and
    /// the bounds established in both arms.
    fn join_value(
        &mut self,
        facts: &mut prove::Facts,
        arms: [&World; 2],
        then_value: &Value,
        else_value: &Value,
        source: Option<&ir::SourceValue>,
        scope: &BTreeSet<SymbolId>,
    ) -> Option<Value> {
        if then_value == else_value {
            return Some(then_value.clone());
        }
        let symbol = source
            .and_then(ir::SourceValue::binding)
            .filter(|symbol| self.arena_ref().symbol_sort(*symbol) == SymbolSort::Int);
        match (then_value, else_value) {
            (Value::Scalar(a), Value::Scalar(b)) => Some(Value::Scalar(match (a, b) {
                (Scalar::Quantity(a), Scalar::Quantity(b)) => {
                    Scalar::Quantity(self.join_integer(facts, arms, [*a, *b], symbol, scope))
                }
                (
                    Scalar::Word {
                        dtype, value: a, ..
                    },
                    Scalar::Word {
                        dtype: other,
                        value: b,
                        ..
                    },
                ) if dtype == other => Scalar::Word {
                    dtype: *dtype,
                    value: self.join_integer(facts, arms, [*a, *b], symbol, scope),
                    binding: symbol,
                },
                (Scalar::Boolean { condition: a, .. }, Scalar::Boolean { condition: b, .. }) => {
                    let condition = if a == b {
                        a.clone()
                    } else {
                        self.fresh_condition()
                    };
                    Scalar::Boolean {
                        value: source
                            .and_then(ir::SourceValue::scalar)
                            .unwrap_or_else(|| self.fresh_scalar(DType::Bool)),
                        condition,
                    }
                }
                (Scalar::Floating(a), Scalar::Floating(b))
                    if ScalarOperand::Scalar(*a).dtype(self.arena_ref())
                        == ScalarOperand::Scalar(*b).dtype(self.arena_ref()) =>
                {
                    Scalar::Floating(source.and_then(ir::SourceValue::scalar).unwrap_or_else(
                        || self.fresh_scalar(ScalarOperand::Scalar(*a).dtype(self.arena_ref())),
                    ))
                }
                (Scalar::Range { start: a, end: b }, Scalar::Range { start: c, end: d }) => {
                    let (start_binding, end_binding) = match source {
                        Some(ir::SourceValue::Range { start, end }) => {
                            (start.binding(), end.binding())
                        }
                        _ => (None, None),
                    };
                    Scalar::Range {
                        start: self.join_integer(facts, arms, [*a, *c], start_binding, scope),
                        end: self.join_integer(facts, arms, [*b, *d], end_binding, scope),
                    }
                }
                _ => panic!("checked join arms bind one scalar kind"),
            })),
            (Value::Tuple(a), Value::Tuple(b)) => a
                .iter()
                .zip(b)
                .enumerate()
                .map(|(index, (x, y))| {
                    let child = match source {
                        Some(ir::SourceValue::Tuple(parts)) => Some(&parts[index]),
                        _ => None,
                    };
                    self.join_value(facts, arms, x, y, child, scope)
                })
                .collect::<Option<Vec<_>>>()
                .map(Value::Tuple),
            (Value::Tensor(_), Value::Tensor(_)) => None,
            _ => panic!("checked join arms bind one value kind"),
        }
    }
    /// A joined integer: `symbol` (or a fresh one) bounded by the
    /// common bounds established for both arm values (`Facts::join_bounds`).
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
    symbols: HashMap<SymbolId, AnyExpr>,
}

/// One candidate's contract instantiated at one call: its accesses, its
/// requirements `(root, path, region)` and its guaranteed writes
/// `(root, region)`, all in the caller.
struct CallSite {
    accesses: Vec<Access>,
    requirements: Vec<(usize, Path, Region)>,
    writes: Vec<(usize, Region)>,
    exits: Vec<Path>,
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
    fn boolean(&mut self, value: BoolExpr) -> (BoolExpr, Vec<SymbolId>) {
        let source = &self.transfer.source.arena;
        let symbols = &mut self.transfer.symbols;
        let value = xfer::transfer_bool(source, value, self.owner.arena(), &mut |symbol, arena| {
            *symbols.entry(symbol).or_insert_with(|| {
                let sort = source.symbol_sort(symbol);
                let s = arena.checker_variable(sort);
                match sort {
                    SymbolSort::Int => AnyExpr::Int(arena.int_symbol(s)),
                    SymbolSort::Nat => AnyExpr::Nat(arena.nat_symbol(s)),
                    SymbolSort::Scalar(_) => AnyExpr::Scalar(arena.scalar_symbol_erased(s)),
                }
            })
        });
        (value, self.owner.arena_ref().free_symbols(value.into()))
    }
    fn integer(&mut self, value: IntExpr) -> IntExpr {
        self.owner.import_integer(self.transfer, value)
    }
    fn binder(&mut self, symbol: SymbolId) -> SymbolId {
        let value = if let Some(value) = self.transfer.symbols.get(&symbol) {
            *value
        } else {
            let (_, value) = self.owner.fresh_integer();
            let value = AnyExpr::Int(value);
            self.transfer.symbols.insert(symbol, value);
            value
        };
        match self.owner.arena_ref().view(value) {
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
        fn value_bindings(value: &Value, arena: &ExprArena, symbols: &mut BTreeSet<SymbolId>) {
            match value {
                Value::Scalar(scalar) => {
                    if let Some(integer) = scalar.integer() {
                        symbols.extend(prove::symbols(arena, integer));
                    }
                    if let Scalar::Word {
                        binding: Some(binding),
                        ..
                    } = scalar
                    {
                        symbols.insert(*binding);
                    }
                    if let Some(value) = scalar.typed() {
                        symbols.extend(arena.free_symbols(AnyExpr::Scalar(value)));
                    }
                    if let Some((start, end)) = scalar.range() {
                        symbols.extend(prove::symbols(arena, start));
                        symbols.extend(prove::symbols(arena, end));
                    }
                }
                Value::Tuple(parts) => {
                    for part in parts {
                        value_bindings(part, arena, symbols);
                    }
                }
                Value::Tensor(_) | Value::Void | Value::Opaque => {}
            }
        }
        let mut symbols = self
            .dimension_symbols()
            .into_iter()
            .collect::<BTreeSet<_>>();
        symbols.extend(self.binders.iter().copied());
        for value in entry.values.values() {
            value_bindings(value, self.arena_ref(), &mut symbols);
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
    fn visit_regions_separate(
        &mut self,
        kind: ir::LoopKind,
        left: Region,
        right: Region,
        facts: &prove::Facts,
        extents: &[IntExpr],
    ) -> bool {
        match kind {
            ir::LoopKind::Ordered => {
                EstablishedRegionOps(self).separated_regions(left, right, facts, extents)
            }
            ir::LoopKind::Independent => self.separated_regions(left, right, facts, extents),
        }
    }
    fn distinct_visit_accesses_separate(
        &mut self,
        kind: ir::LoopKind,
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
        let mut rename = HashMap::from([(symbol, AnyExpr::Int(other))]);
        for local in symbols {
            if local != symbol && !captured.contains(&local) {
                let sort = self.arena_ref().symbol_sort(local);
                let fresh = self.arena().checker_variable(sort);
                let value = match sort {
                    SymbolSort::Int => AnyExpr::Int(self.arena().int_symbol(fresh)),
                    SymbolSort::Nat => AnyExpr::Nat(self.arena().nat_symbol(fresh)),
                    SymbolSort::Scalar(_) => {
                        AnyExpr::Scalar(self.arena().scalar_symbol_erased(fresh))
                    }
                };
                rename.insert(local, value);
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
            // Guarded accesses are subsets of these same footprints. If
            // loop geometry already separates their overapproximations,
            // their potentially complex path predicates cannot introduce
            // an overlap. Retain the path-aware proof when geometry alone
            // is insufficient, including accesses separated only by guards.
            let mut geometry = facts.clone();
            let mut geometry_path = vec![];
            if !self.assume(
                &mut geometry_path,
                &mut geometry,
                Condition::Compare(BinaryOp::Lt, a, b),
                true,
            ) {
                continue;
            }
            if self.visit_regions_separate(
                kind,
                left.region.clone(),
                other_region.clone(),
                &geometry,
                extents,
            ) {
                continue;
            }
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
            if !self.visit_regions_separate(
                kind,
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
        let mut seen = HashSet::new();
        let mut root_indices = HashMap::new();
        let mut roots: Vec<Vec<&Access>> = Vec::new();
        for access in outcomes
            .iter()
            .flat_map(|outcome| outcome.accesses.iter().skip(access_start))
            .filter(|access| entry.roots.contains_key(&access.root))
        {
            // Outcome worlds can retain the same prefix accesses. Prove one
            // obligation for each exact footprint, keeping its first span.
            if !seen.insert(access.footprint()) {
                continue;
            }
            let index = *root_indices.entry(access.root).or_insert_with(|| {
                roots.push(Vec::new());
                roots.len() - 1
            });
            roots[index].push(access);
        }
        // Distinct roots cannot alias. Preserve first-root encounter order,
        // and retain self-pairs so one footprint still needs visit separation.
        for accesses in roots {
            let extents = entry.roots[&accesses[0].root].axes.clone();
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
                    let overlap = VisitOverlap {
                        root: left.root,
                        left: left.span,
                        right: right.span,
                    };
                    if kind == ir::LoopKind::Ordered && atomic {
                        return Err(overlap);
                    }
                    if !self.distinct_visit_accesses_separate(
                        kind, left, right, symbol, start, end, facts, &captured, &extents,
                    ) {
                        return Err(overlap);
                    }
                }
            }
        }
        Ok(())
    }
    fn rename_integer(&mut self, value: IntExpr, map: &HashMap<SymbolId, AnyExpr>) -> IntExpr {
        let values = map.iter().map(|(s, v)| (*s, *v)).collect::<Vec<_>>();
        match self.arena().substitute_any(AnyExpr::Int(value), &values) {
            AnyExpr::Int(value) => value,
            _ => unreachable!("typed substitution preserves integer result"),
        }
    }
    fn rename_region(&mut self, region: &Region, map: &HashMap<SymbolId, AnyExpr>) -> Region {
        match region {
            Region::Empty => Region::Empty,
            Region::Full => Region::Full,
            Region::Linear(a, b) => {
                Region::Linear(self.rename_integer(*a, map), self.rename_integer(*b, map))
            }
            Region::Image {
                domain,
                coordinates,
            } => Region::Image {
                domain: self.rename_domain(domain, map),
                coordinates: coordinates
                    .iter()
                    .map(|c| self.rename_integer(*c, map))
                    .collect(),
            },
            Region::LinearImage { domain, address } => Region::LinearImage {
                domain: self.rename_domain(domain, map),
                address: self.rename_integer(*address, map),
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
                    start: self.rename_integer(bound.start, &map),
                    end: self.rename_integer(bound.end, &map),
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
    fn rename_domain(&mut self, domain: &[Bound], map: &HashMap<SymbolId, AnyExpr>) -> Vec<Bound> {
        domain
            .iter()
            .map(|d| Bound {
                symbol: self.renamed_bound(d.symbol, map),
                start: self.rename_integer(d.start, map),
                end: self.rename_integer(d.end, map),
            })
            .collect()
    }
    fn renamed_bound(&self, symbol: SymbolId, map: &HashMap<SymbolId, AnyExpr>) -> SymbolId {
        let Some(value) = map.get(&symbol) else {
            return symbol;
        };
        match self.arena_ref().view(*value) {
            crate::expr::NodeView::Symbol(renamed) => renamed,
            _ => unreachable!("bound coordinate maps to a bound coordinate"),
        }
    }
    fn rename_condition(&mut self, c: &Condition, map: &HashMap<SymbolId, AnyExpr>) -> Condition {
        match c {
            Condition::Version(version, binders) => Condition::Version(
                *version,
                binders
                    .iter()
                    .flat_map(|symbol| {
                        map.get(symbol)
                            .map(|value| self.arena_ref().free_symbols(*value))
                            .unwrap_or_else(|| vec![*symbol])
                    })
                    .collect(),
            ),
            Condition::Actual(value, binders) => {
                let replacements = map.iter().map(|(s, v)| (*s, *v)).collect::<Vec<_>>();
                let value = match self
                    .arena()
                    .substitute_any(AnyExpr::Bool(*value), &replacements)
                {
                    AnyExpr::Bool(value) => value,
                    _ => unreachable!("predicate substitution preserves Bool sort"),
                };
                let binders = binders
                    .iter()
                    .flat_map(|symbol| {
                        map.get(symbol)
                            .map(|value| self.arena_ref().free_symbols(*value))
                            .unwrap_or_else(|| vec![*symbol])
                    })
                    .collect();
                Condition::Actual(value, binders)
            }
            Condition::Compare(op, a, b) => Condition::Compare(
                *op,
                self.rename_integer(*a, map),
                self.rename_integer(*b, map),
            ),
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
            if let Value::Scalar(scalar) = value {
                let value = match scalar {
                    Scalar::Word {
                        binding: Some(binding),
                        ..
                    } => Some(AnyExpr::Int(self.arena().int_symbol(*binding))),
                    _ => scalar
                        .integer()
                        .map(AnyExpr::Int)
                        .or_else(|| scalar.typed().map(AnyExpr::Scalar)),
                };
                let Some(value) = value else {
                    continue;
                };
                if let crate::expr::NodeView::Symbol(symbol) = self.arena_ref().view(value) {
                    if !metadata
                        .transfer
                        .symbols
                        .iter()
                        .any(|(prior, _)| *prior == symbol)
                    {
                        metadata.transfer.symbols.push((
                            symbol,
                            if scalar.typed().is_some() {
                                ParameterPart::Scalar(path.clone())
                            } else {
                                ParameterPart::Integer(path.clone())
                            },
                        ));
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
                let logical = EstablishedRegionOps(self).project_view_region(
                    writes,
                    &place.view,
                    path,
                    &entry.facts,
                );
                written.push((capture.clone(), logical));
            }
        }
        let mut parameter_axes = self.parameter_axes();
        for (capture, value) in &captures {
            if let Value::Tensor(place) = value {
                parameter_axes.insert(capture.clone(), place.view.axes.clone());
            }
        }
        // Close only this outcome's exit, then conjoin it with the exits
        // already closed: reclosing a closed transfer would expand its
        // disjunctive success path into one copy of every write per path.
        let outcome = self.close_transfer(
            crate::initialization::InitializationContract {
                exits: vec![Exit {
                    path: path.clone(),
                    written,
                }],
                symbols: metadata.transfer.symbols.clone(),
                ..crate::initialization::InitializationContract::empty()
            },
            &allowed,
            &parameter_axes,
        );
        let prior = std::mem::replace(
            &mut metadata.transfer,
            crate::initialization::InitializationContract::empty(),
        );
        metadata.transfer = prior.conjoin_closed(outcome);
        for (local, invariant) in invariants {
            let mut parameter = ParameterPath::root(offset + local.local.index());
            for index in &local.path {
                parameter = parameter.child(*index);
            }
            let region = self.boundary_region(
                invariant.clone(),
                &allowed,
                false,
                parameter_axes.get(&parameter).map(Vec::as_slice),
            );
            if let Some((_, prior)) = metadata.carried.iter_mut().find(|(p, _)| *p == parameter) {
                *prior = prior.clone().intersection(region);
            } else {
                metadata.carried.push((parameter, region));
            }
        }
    }

    fn carried_region(&mut self, world: &World, place: &Place) -> Region {
        let root = &world.roots[&place.root];
        let written = self.materialized_written(world, place.root);
        EstablishedRegionOps(self).project_view_region(
            written.union(root.incoming.clone()),
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
        value_versions: &[(ir::LocalId, ir::SourceValue, ir::SourceValue)],
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
            .value
            .binding()
            .expect("checked loop binder has its symbolic identity");
        let current = self.arena().int_symbol(symbol);
        let one = self.arena().int(1);
        let upper = self.arena().int_sub(end, one);
        iteration.facts.set_range(symbol, start, upper);
        let access_facts = iteration.facts.clone();
        iteration
            .values
            .insert(binder, Value::Scalar(Scalar::Quantity(current)));
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
                let mut ty = &self.checker.locals[local.local.index()].ty;
                for index in &local.path {
                    let ValueType::Tuple(parts) = ty else {
                        unreachable!("checked tuple assignment path")
                    };
                    ty = &parts.as_slice()[*index];
                }
                let ty = ty.clone();
                if matches!(
                    iteration
                        .values
                        .get(&local.local)
                        .map(|value| value.at(&local.path)),
                    Some(Value::Scalar(_))
                ) {
                    let unknown = self.result(&mut iteration, &ty, true);
                    *iteration
                        .values
                        .get_mut(&local.local)
                        .unwrap()
                        .at_mut(&local.path) = unknown;
                }
            }
        }
        let header_bindings = value_versions
            .iter()
            .map(|(local, header, _)| (*local, header.clone()))
            .collect::<Vec<_>>();
        self.bind_value_bindings(&mut iteration, &header_bindings);
        iteration.defer_depth += 1;
        let deferred_start = iteration.deferred.len();
        let access_start = iteration.accesses.len();
        let mut invariants = Vec::new();
        let mut outcomes = if carries.is_empty() || single {
            self.block(vec![iteration], body)
        } else {
            let mut allowed = self.dimension_symbols();
            allowed.extend(self.symbols.iter().map(|(symbol, _)| *symbol));
            allowed.extend(self.binders.iter().copied().filter(|s| *s != symbol));
            for value in entry.values.values() {
                if let Value::Scalar(scalar) = value {
                    if let Some(value) = scalar.integer() {
                        allowed.extend(prove::symbols(self.arena_ref(), value));
                    }
                    if let Scalar::Word {
                        binding: Some(binding),
                        ..
                    } = scalar
                    {
                        allowed.push(*binding);
                    }
                    if let Some(value) = scalar.typed() {
                        allowed.extend(self.arena_ref().free_symbols(AnyExpr::Scalar(value)));
                    }
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
                let root_axes = initial_root.axes.clone();
                let mut initial = self
                    .materialized_written(&iteration, place.root)
                    .union(initial_root.incoming.clone());
                if let Some(parameter) = &initial_root.parameter {
                    let regions = preliminary
                        .iter()
                        .flat_map(|outcome| {
                            outcome
                                .requirements
                                .iter()
                                .filter(|r| &r.parameter == parameter)
                        })
                        .map(|required| {
                            let region = Region::Guard(
                                required.path.clone(),
                                Box::new(required.region.clone()),
                            );
                            Region::Bind(Bound { symbol, start, end }, Box::new(region))
                        })
                        .collect();
                    for region in self.normalize_many(regions, &iteration.facts) {
                        initial = initial.union(self.boundary_region(
                            region,
                            &allowed,
                            true,
                            Some(&root_axes),
                        ));
                    }
                }
                let mut invariant = EstablishedRegionOps(self).project_view_region(
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
                invariant =
                    self.boundary_region(invariant, &allowed, false, Some(&place.view.axes));
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
                    let actual =
                        self.boundary_region(actual, &allowed, false, Some(&next.view.axes));
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
        // Outcomes still own their arm and iteration facts here. Snapshot each
        // exported root once before matching paths and binding private axes.
        for outcome in &mut outcomes {
            // A root this visit never wrote keeps the entry's own region.
            let roots = entry
                .roots
                .iter()
                .filter(|(root, before)| {
                    outcome
                        .roots
                        .get(root)
                        .is_some_and(|after| after.written != before.written)
                })
                .map(|(root, _)| *root)
                .collect::<Vec<_>>();
            for (root, written) in self.materialized_writes(outcome, &roots) {
                outcome.roots.get_mut(&root).unwrap().written = written;
            }
        }
        // Roots no visit wrote: their completed loop adds nothing.
        let unwritten = entry
            .roots
            .iter()
            .filter(|(root, before)| {
                outcomes.iter().all(|outcome| {
                    outcome
                        .roots
                        .get(root)
                        .is_some_and(|after| after.written == before.written)
                })
            })
            .map(|(root, _)| *root)
            .collect::<HashSet<_>>();
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
            let exported_roots = entry.roots.keys().copied().collect::<Vec<_>>();
            let written_roots = exported_roots
                .iter()
                .copied()
                .filter(|root| !unwritten.contains(root))
                .collect::<Vec<_>>();
            let guaranteed_regions = written_roots
                .iter()
                .map(|root| {
                    matching.iter().fold(Region::Full, |state, other| {
                        state.intersection(
                            other
                                .roots
                                .get(root)
                                .map_or(Region::Empty, |r| r.written.clone()),
                        )
                    })
                })
                .collect();
            let mut guaranteed = written_roots
                .iter()
                .copied()
                .zip(self.normalize_many(guaranteed_regions, &entry.facts))
                .collect::<HashMap<_, _>>();
            for root in &unwritten {
                guaranteed.insert(*root, entry.roots[root].written.clone());
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
            // Incoming sets are input assumptions, not produced writes. Do
            // not carry an iteration-private guard into the next outer world.
            // Rebuild additions from the quantified requirements below.
            for (root, before) in &entry.roots {
                exit.roots.get_mut(root).unwrap().incoming = before.incoming.clone();
            }
            // Incoming/trial-contract reads may have been exported immediately
            // rather than deferred. Their private guards have the same loop
            // lifetime as recorded accesses and deferred read requirements.
            let requirements = outcome
                .requirements
                .iter()
                .skip(entry.requirements.len())
                .collect::<Vec<_>>();
            let accesses = outcome
                .accesses
                .iter()
                .skip(access_start)
                .filter(|access| entry.roots.contains_key(&access.root))
                .collect::<Vec<_>>();
            let quantified = |path: &Path, region: &Region| {
                Region::Bind(
                    Bound { symbol, start, end },
                    Box::new(Region::guarded(path.clone(), region.clone())),
                )
            };
            // One export boundary owns one normalization batch. Each effect
            // retains its result and metadata, while identical binder/guard
            // scopes reuse their own reached scalar and derivation state.
            let regions = requirements
                .iter()
                .map(|required| quantified(&required.path, &required.region))
                .chain(
                    accesses
                        .iter()
                        .map(|access| quantified(&access.path, &access.region)),
                )
                .collect();
            let mut regions = self.normalize_many(regions, &entry.facts).into_iter();
            exit.requirements = entry.requirements.clone();
            for required in requirements {
                let region = regions.next().expect("normalized requirement export");
                let path = self.independent_path(&required.path, symbol);
                for root in exit.roots.values_mut() {
                    if root.parameter.as_ref() == Some(&required.parameter) {
                        root.incoming = root
                            .incoming
                            .clone()
                            .union(Region::guarded(path.clone(), region.clone()));
                    }
                }
                exit.requirements.push(Requirement {
                    parameter: required.parameter.clone(),
                    region,
                    path,
                    span: required.span,
                });
            }
            exit.accesses = entry.accesses.clone();
            for access in accesses {
                let region = regions.next().expect("normalized access export");
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
            assert!(regions.next().is_none(), "extra normalized loop exports");
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
            let mut exported_reads = Vec::new();
            for read in outcome.deferred.iter().skip(deferred_start) {
                let mut available = read.available.clone();
                if kind == ir::LoopKind::Ordered {
                    if let Some(writes) = guaranteed.get(&read.root) {
                        let (prior_symbol, prior) = self.fresh_integer();
                        let writes = self
                            .rename_region(writes, &HashMap::from([(symbol, AnyExpr::Int(prior))]));
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
                // Resolve vacuity with the original private loop facts before
                // binding the region or projecting any path symbols away.
                // This is required at every nesting level, including reads
                // propagated to an enclosing loop or exported as input effects.
                if self.covered(&Region::Empty, &Region::Full, &read.path, &read.facts, axes) {
                    continue;
                }
                if entry.defer_depth == 0 {
                    let Some(root) = exit.roots.get(&read.root) else {
                        continue;
                    };
                    if root.parameter.is_none() {
                        // A local failure stays at its saved source position;
                        // it does not produce a quantified input effect.
                        let mut at_read = exit.clone();
                        at_read.path = read.path.clone();
                        at_read.facts = read.facts.clone();
                        at_read.roots.get_mut(&read.root).unwrap().written = available;
                        at_read.roots.get_mut(&read.root).unwrap().incoming = Region::Empty;
                        self.require(&mut at_read, read.root, read.region.clone(), read.span);
                        continue;
                    }
                }
                // A read's guard is stated over its own source-position
                // quantities. Reach it in that world, so a condition on an
                // iteration value names the binder it denotes before the
                // binder is quantified and independent conditions are kept.
                let path = read
                    .path
                    .iter()
                    .map(|(condition, truth)| {
                        (
                            self.reached_condition(condition.clone(), &read.facts),
                            *truth,
                        )
                    })
                    .collect::<Path>();
                exported_reads.push((read, path));
            }
            // Each read was resolved against its own saved world. Only the
            // surviving effects share the exit world at this export boundary.
            // Keep coordinate-dependent guards inside their quantified domain.
            let regions = exported_reads
                .iter()
                .map(|(read, path)| quantified(path, &read.region))
                .collect();
            let regions = self.normalize_many(regions, &exit.facts);
            for ((read, path), region) in exported_reads.into_iter().zip(regions) {
                let path = self.independent_path(&path, symbol);
                if entry.defer_depth > 0 {
                    exit.deferred.push(Read {
                        root: read.root,
                        region,
                        available: Region::Empty,
                        path,
                        facts: exit.facts.clone(),
                        span: read.span,
                    });
                } else {
                    let root = exit.roots[&read.root].clone();
                    let parameter = root.parameter.expect("classified loop input requirement");
                    exit.requirements.push(Requirement {
                        parameter,
                        region: region.clone(),
                        path: path.clone(),
                        span: read.span,
                    });
                    exit.roots.get_mut(&read.root).unwrap().incoming =
                        root.incoming.union(Region::Guard(path, Box::new(region)));
                }
            }
            let completed = written_roots
                .iter()
                .map(|root| {
                    let written = guaranteed.get(root).cloned().unwrap_or(Region::Empty);
                    Region::Bind(Bound { symbol, start, end }, Box::new(written))
                })
                .collect();
            // Produced writes use established-only simplification, independently
            // of mandatory read export; their earlier effects remain intact.
            let completed = EstablishedRegionOps(self).normalize_many(completed, &exit.facts);
            for (root, completed) in written_roots.into_iter().zip(completed) {
                let written = entry.roots[&root].written.clone().union(completed);
                // A completed write of every element is the whole root, so
                // later reads need not rediscover it through each geometry.
                let whole = !matches!(written, Region::Full)
                    && EstablishedRegionOps(self).covered(
                        &written,
                        &Region::Full,
                        &exit.path,
                        &exit.facts,
                        &exit.roots[&root].axes,
                    );
                exit.roots.get_mut(&root).unwrap().written =
                    if whole { Region::Full } else { written };
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
            .checker_variable(crate::expr::SymbolSort::Int)
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
        let parameter_axes = self.parameter_axes();
        self.close_transfer(contract, &allowed, &parameter_axes)
    }
    /// Contract coordinates belong to each logical parameter leaf. Selected
    /// and reshaped captures supply their own view axes at the loop boundary.
    fn parameter_axes(&self) -> HashMap<ParameterPath, Vec<IntExpr>> {
        fn leaves(
            ty: &ValueType,
            path: ParameterPath,
            result: &mut HashMap<ParameterPath, Vec<IntExpr>>,
        ) {
            match ty {
                ValueType::Tensor(tensor) => {
                    result.insert(path, tensor.axes.clone());
                }
                ValueType::Tuple(parts) => {
                    for (ordinal, part) in parts.iter().enumerate() {
                        leaves(part, path.child(ordinal), result);
                    }
                }
                _ => {}
            }
        }
        let mut result = HashMap::new();
        for (ordinal, parameter) in self.checker.sig.params.iter().enumerate() {
            leaves(&parameter.ty, ParameterPath::root(ordinal), &mut result);
        }
        result
    }
}
