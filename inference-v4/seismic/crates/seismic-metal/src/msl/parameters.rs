//! Typed launch operands shared by retained emission and model construction.
//! Values refer to the original enclosing solver parameters, not a canonical
//! mapping selected to make the existing printer accept a family.
use crate::terminal::{Expression as E, Site, Statement as S, Type as T};
use magnitude_solver::model::ModelBuilder;
use seismic_accounting::algebra::{Algebra, Error, Symbolic, Value};
use seismic_lang::{ast::{BinaryOp as B, UnaryOp}, sym::Sym};
use std::collections::BTreeMap;

#[derive(Clone)]
pub(crate) struct Axis {
    pub extent: Value,
    pub count: Value,
    pub step: Value,
    pub stride: Value,
    divisor: Value,
    count_divisor: Value,
}
#[derive(Clone)]
pub(crate) struct Launch {
    pub work_items: Value,
    pub parts: Value,
    pub axes: Vec<Axis>,
    bindings: Vec<Value>,
    definitions: Vec<(Value, Sym)>,
}
#[derive(Clone, Default)]
pub(crate) struct Operands {
    pub bindings: BTreeMap<String, Value>,
    /// Unconditional equations over registered native numeric operands.
    pub definitions: BTreeMap<String, Sym>,
    pub allocations: Vec<Allocation>,
    pub arrays: Vec<Array>,
}
pub(crate) fn symbol(value: Value) -> String { format!("seismic_family_number_{}__", value.id().0) }
fn original(value: Value) -> Sym {
    let (minimum, maximum) = value.bounds();
    if minimum == maximum { Sym::constant(minimum as i64) } else { Sym::param(&symbol(value)) }
}
impl Operands {
    pub(crate) fn expression(&mut self, name: String, value: Value) -> E {
        let (minimum, maximum) = value.bounds();
        if minimum == maximum {
            E::Integer(minimum as i64, T::U32)
        } else {
            let original = symbol(value);
            self.bindings.insert(original.clone(), value);
            if name != original {
                self.definitions.insert(name.clone(), Sym::param(&original));
            }
            self.bindings.insert(name.clone(), value);
            E::variable(name, T::U32)
        }
    }
    pub(crate) fn define(&mut self, value: Value, definition: Sym) -> Result<(), String> {
        if value.bounds().0 == value.bounds().1 { return Ok(()); }
        let name = symbol(value);
        self.expression(name.clone(), value);
        if definition == Sym::param(&name) { return Ok(()); }
        if self.definitions.get(&name).is_some_and(|previous| *previous != definition) {
            return Err("one original Metal numeric operand has conflicting definitions".into());
        }
        self.definitions.insert(name, definition);
        Ok(())
    }
    pub(crate) fn instantiate(&self, program: &crate::terminal::Program, values: &[i64]) -> Result<crate::terminal::Program, String> {
        let mut selected = program.clone();
        for (name, binding) in &self.bindings {
            let value = *values.get(binding.id().0).ok_or("missing original Metal launch parameter")?;
            let unsigned = u64::try_from(value).map_err(|_| "negative Metal launch parameter")?;
            if unsigned < binding.bounds().0 || unsigned > binding.bounds().1 || unsigned > u64::from(u32::MAX) {
                return Err("Metal launch parameter lies outside its retained coordinate domain".into());
            }
            let value = E::Integer(value, T::U32);
            for launch in &mut selected.launches {
                for site in launch { site.statement = crate::terminal::rewrite::statement(&site.statement, name, &value); }
            }
        }
        for allocation in &self.allocations {
            let selected_allocation = allocation.selected(values)?;
            if let Some(body) = selected.launches.get_mut(allocation.launch) {
                let mut retained = Vec::with_capacity(body.len());
                for mut site in body.drain(..) {
                    if let S::Array { name, elements, .. } = &mut site.statement {
                        if name == &allocation.symbol {
                            let Some(declaration) = &selected_allocation else { continue; };
                            let unit = seismic_realization::dispatch::GroupDispatch::new(1, crate::execution::SUBGROUP as u64, 1)?;
                            let layout = declaration.layout(&unit)?;
                            *elements = match declaration.placement {
                                seismic_realization::dispatch::TilePlacement::GroupShared => layout.shared_elements_per_item,
                                _ => layout.private_elements_per_lane,
                            };
                        }
                    }
                    retained.push(site);
                }
                *body = retained;
            }
        }
        Ok(selected)
    }
    pub(crate) fn source(&self, source: &str, values: &[i64]) -> Result<String, String> {
        let mut source = source.to_owned();
        for allocation in &self.allocations {
            let begin = allocation.boundary(false);
            let end = allocation.boundary(true);
            if let Some(start) = source.find(&begin) {
                let content = start + begin.len();
                let finish = source[content..].find(&end).map(|offset| content + offset).ok_or("unterminated retained backing declaration")?;
                let replacement = match allocation.selected(values)? {
                    Some(declaration) if declaration.placement == seismic_realization::dispatch::TilePlacement::GroupShared => {
                        format!("threadgroup {} {}[{} * {}];", T::from(declaration.dtype).metal(), declaration.symbol,
                            declaration.capacity.max(1), super::grouping_parameter(allocation.launch))
                    },
                    Some(_) => source[content..finish].to_string(),
                    None => String::new(),
                };
                source.replace_range(start..finish + end.len(), &replacement);
            } else if allocation.placement == seismic_realization::dispatch::TilePlacement::GroupShared {
                return Err("shared backing declaration has no retained activation boundary".into());
            }
        }
        for (name, binding) in &self.bindings {
            let value = *values.get(binding.id().0).ok_or("missing original Metal declaration parameter")?;
            let unsigned = u64::try_from(value).map_err(|_| "negative Metal declaration parameter")?;
            if unsigned < binding.bounds().0 || unsigned > binding.bounds().1 || unsigned > u64::from(u32::MAX) {
                return Err("Metal declaration parameter lies outside its retained domain".into());
            }
            source = source.replace(name, &E::Integer(value, T::U32).render());
        }
        Ok(source)
    }
    pub(crate) fn shared_bytes(&self, launch: usize, dispatch: &seismic_realization::dispatch::GroupDispatch, values: &[i64]) -> Result<Option<u64>, String> {
        let mut found = false;
        let mut bytes = 0u64;
        for allocation in self.allocations.iter().filter(|allocation| allocation.launch == launch) {
            found = true;
            if let Some(declaration) = allocation.selected(values)? {
                bytes = bytes.checked_add(declaration.layout(dispatch)?.shared_bytes_per_group)
                    .ok_or("selected Metal shared storage overflow")?;
            }
        }
        Ok(found.then_some(bytes))
    }
    pub(crate) fn tiles(&self, launch: usize, values: &[i64]) -> Result<Option<Vec<seismic_realization::dispatch::TileDeclaration>>, String> {
        let mut found = false;
        let mut identities = std::collections::HashSet::new();
        let mut declarations = Vec::new();
        for array in self.arrays.iter().filter(|array| array.launch == launch) {
            found = true;
            if let Some(declaration) = array.selected(values)? {
                if !identities.insert(array.id) { return Err("retained allocation has multiple selected declaration variants".into()); }
                declarations.push(declaration);
            }
        }
        Ok(found.then_some(declarations))
    }
}
impl Launch {
    /// Derive suffix strides from the same count variables used in dispatch.
    pub(crate) fn new(builder: &mut ModelBuilder, name: &str, work_items: Value, parts: Value,
        axes: impl IntoIterator<Item=(Value, Value, Value)>) -> Result<Self, Error> {
        let axes = axes.into_iter().collect::<Vec<_>>();
        let mut bindings = vec![work_items, parts];
        let mut definitions = Vec::new();
        let one = Symbolic::new(builder, name).constant(1)?;
        let mut stride = one;
        let mut result = Vec::new();
        for (extent, count, step) in axes.into_iter().rev() {
            if extent.bounds().1 > i32::MAX as u64 || count.bounds().1 > u32::MAX as u64
                || step.bounds().0 == 0 || step.bounds().1 > u32::MAX as u64 {
                return Err(Error::Unsupported("Metal parameterized mapping exceeds its index representation".into()));
            }
            let mut algebra = Symbolic::new(builder, name);
            let one = algebra.constant(1)?;
            let divisor = algebra.maximum(stride, one)?;
            let count_divisor = algebra.maximum(count, one)?;
            bindings.extend([extent, count, step, stride, divisor, count_divisor]);
            // Keep zero-safe maxima as their own original operands unless the
            // retained domain proves that the maximum is the input itself.
            if stride.bounds().0 >= 1 { definitions.push((divisor, original(stride))); }
            if count.bounds().0 >= 1 { definitions.push((count_divisor, original(count))); }
            result.push(Axis { extent, count, step, stride, divisor, count_divisor });
            let next = Symbolic::new(builder, name).product(stride, count)?;
            definitions.push((next, original(stride).mul(&original(count))));
            bindings.push(next);
            stride = next;
        }
        result.reverse();
        // Empty logical mappings use zero strides on every axis, even axes
        // after the zero extent. Match WorkMapping's concrete empty-domain ABI.
        if stride.bounds().0 == 0 {
            let live = builder.variable(format!("{name}.coordinates_live"), magnitude_solver::model::Domain::boolean());
            builder.guarded_constraint(vec![magnitude_solver::model::Literal::new(live, 1)],
                magnitude_solver::model::Constraint::LinearLe { terms: vec![magnitude_solver::model::LinearTerm::new(stride.id(), -1)], rhs: -1 });
            builder.guarded_constraint(vec![magnitude_solver::model::Literal::new(live, 0)],
                magnitude_solver::model::Constraint::LinearLe { terms: vec![magnitude_solver::model::LinearTerm::new(stride.id(), 1)], rhs: 0 });
            let live = Value::binding(live, &magnitude_solver::model::Domain::boolean())?;
            bindings.push(live);
            for axis in &mut result {
                let mut algebra = Symbolic::new(builder, name);
                let stride = algebra.product(axis.stride, live)?;
                definitions.push((stride, original(axis.stride).mul(&original(live))));
                axis.stride = stride;
                axis.divisor = algebra.maximum(axis.stride, one)?;
                bindings.extend([axis.stride, axis.divisor]);
                if axis.stride.bounds().0 >= 1 { definitions.push((axis.divisor, original(axis.stride))); }
            }
        }
        if work_items.bounds().1 > u32::MAX as u64 || parts.bounds().0 == 0 || parts.bounds().1 > u32::MAX as u64 {
            return Err(Error::Unsupported("Metal parameterized launch exceeds its slot representation".into()));
        }
        Ok(Self { work_items, parts, axes: result, bindings, definitions })
    }
    pub(crate) fn program(&self, launch: usize, grouping: E, coordinate_names: &[String],
        operands: &mut Operands) -> Result<Vec<Site>, String> {
        if coordinate_names.len() != self.axes.len() { return Err("retained launch coordinate arity mismatch".into()); }
        // A statically empty mapping emits zero coordinates and consumes no
        // suffix strides. Such unused suffix products can exceed u32 even
        // though the launch itself has zero work; do not turn them into native
        // operands merely to carry arithmetic metadata.
        if self.work_items.bounds().1 != 0 {
            for &binding in &self.bindings { operands.expression(symbol(binding), binding); }
            for (binding, definition) in &self.definitions { operands.define(*binding, definition.clone())?; }
        }
        let mut statements = Vec::new();
        let mut push = |statement| statements.push(Site { operation: None, statement });
        let value = |name: &str, ty| E::variable(name, ty);
        let prefix = format!("seismic_family_launch_{launch}");
        let work_items = operands.expression(format!("{prefix}_work__"), self.work_items);
        push(S::Let { name: "seismic_entry_base".into(), ty: T::U32,
            value: E::binary(B::Mul, value("tg_pos.x", T::U32), grouping, T::U32) });
        push(S::Let { name: "seismic_entry_slot".into(), ty: T::U32,
            value: E::binary(B::Add, value("seismic_entry_base", T::U32), value("sg_id", T::U32), T::U32) });
        push(S::Let { name: "seismic_entry_live".into(), ty: T::Bool,
            value: E::Helper(crate::support::Helper::WorkItemLive,
                vec![value("seismic_entry_slot", T::U32), work_items], T::Bool) });
        push(S::ReturnIf(E::Unary(UnaryOp::Not, Box::new(value("seismic_entry_live", T::Bool)), T::Bool)));
        let item = if self.parts.bounds() == (1, 1) { value("seismic_entry_slot", T::U32) } else {
            let parts = operands.expression(format!("{prefix}_parts__"), self.parts);
            push(S::Let { name: "part".into(), ty: T::I32,
                value: E::binary(B::Rem, value("seismic_entry_slot", T::U32), parts.clone(), T::U32).cast(T::I32) });
            E::binary(B::Div, value("seismic_entry_slot", T::U32), parts, T::U32)
        };
        push(S::Let { name: "item".into(), ty: T::U32, value: item });
        for (axis, (geometry, name)) in self.axes.iter().zip(coordinate_names).enumerate() {
            // An empty work domain exits at the padding guard before these
            // divisions; retaining a zero constant divisor would still be an
            // invalid target expression, so empty axes have coordinate zero.
            let coordinate = if self.work_items.bounds() == (0, 0) { E::Integer(0, T::I32) } else {
                let stride = operands.expression(format!("{prefix}_axis{axis}_stride__"), geometry.divisor);
                let count = operands.expression(format!("{prefix}_axis{axis}_count__"), geometry.count_divisor);
                let step = operands.expression(format!("{prefix}_axis{axis}_step__"), geometry.step);
                E::binary(B::Mul,
                    E::binary(B::Rem, E::binary(B::Div, value("item", T::U32), stride, T::U32), count, T::U32), step, T::U32).cast(T::I32)
            };
            push(S::Let { name: name.clone(), ty: T::I32, value: coordinate });
        }
        Ok(statements)
    }
}

/// One actual native backing array. The logical capacity and activation are
/// bound to the same source/layout decisions as its uses. Alternative arrays
/// may share a binding identity only through an explicit allocation selection.
#[derive(Clone)]
pub(crate) struct Allocation {
    pub launch: usize,
    pub symbol: String,
    pub dtype: seismic_lang::types::DType,
    pub placement: seismic_realization::dispatch::TilePlacement,
    pub capacity: Value,
    pub guards: Vec<magnitude_solver::model::Literal>,
}
impl Allocation {
    pub(crate) fn selected(&self, values: &[i64]) -> Result<Option<seismic_realization::dispatch::TileDeclaration>, String> {
        for guard in &self.guards {
            if *values.get(guard.variable.0).ok_or("missing allocation activation")? != guard.value { return Ok(None); }
        }
        let capacity = values.get(self.capacity.id().0).and_then(|&value| u64::try_from(value).ok()).ok_or("missing selected backing capacity")?;
        if capacity < self.capacity.bounds().0 || capacity > self.capacity.bounds().1 { return Err("selected backing capacity is outside its original domain".into()); }
        Ok(Some(seismic_realization::dispatch::TileDeclaration {symbol: self.symbol.clone(), dtype: self.dtype, capacity, placement: self.placement.clone()}))
    }
    pub(crate) fn boundary(&self, end: bool) -> String {
        format!("/*seismic_family_allocation_{}_{}_{}__*/", self.launch, self.symbol, if end {"end"} else {"begin"})
    }
}

/// One logical allocation request, distinct from a reusable native backing.
/// Preserve lexical request order so installation can compare the selected
/// terminal inventory directly with the selected memory plan.
#[derive(Clone)]
pub(crate) struct Array {
    pub launch: usize,
    pub id: crate::memory::AllocationId,
    pub symbol: String,
    pub dtype: seismic_lang::types::DType,
    pub placement: seismic_realization::dispatch::TilePlacement,
    pub capacity: Value,
    pub guards: Vec<magnitude_solver::model::Literal>,
}
impl Array {
    pub(crate) fn selected(&self, values: &[i64]) -> Result<Option<seismic_realization::dispatch::TileDeclaration>, String> {
        Allocation { launch: self.launch, symbol: self.symbol.clone(), dtype: self.dtype,
            placement: self.placement.clone(), capacity: self.capacity, guards: self.guards.clone() }.selected(values)
    }
}
