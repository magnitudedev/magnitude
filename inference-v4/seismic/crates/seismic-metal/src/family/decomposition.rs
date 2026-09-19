//! Original Metal decomposition parameters and their exact dispatch equations.
//! Numeric domains are retained as arithmetic, including the partition tail and
//! split merge launch; no completed executions are constructed by this module.
use crate::{
    execution::SUBGROUP,
    tuning::{Capacities, Decomposition, Form},
};
use magnitude_solver::model::{Constraint, Domain, LinearTerm, Literal, ModelBuilder, VarId};
use seismic_accounting::algebra::{Algebra, Error, Symbolic, Value};
use seismic_lang::{ir::StmtKind, lowered_ir::LoweredIr, sym::Sym, types::Ty};
use seismic_realization::dispatch::{GroupDispatch, WorkMapping, geometry};

pub struct Binding {
    pub partition: Value,
    pub split: Value,
    pub partitioned: VarId,
    pub split_active: VarId,
    pub phases: Vec<Phase>,
    source: LoweredIr,
    form: Form,
    decisions: Vec<(String, Value)>,
}
pub struct Phase {
    pub axes: Vec<Axis>,
    pub work_items: Value,
    pub main: Launch,
    pub merge: Option<Launch>,
    partition_axis: Option<Axis>,
}
pub struct Axis {
    pub extent: Value,
    pub step: Value,
    pub count: Value,
}
pub struct Launch {
    pub presence: Option<VarId>,
    pub work_items: Value,
    pub items_per_group: Value,
    pub groups: Value,
    pub threads_per_group: Value,
    pub dispatched_lanes: Value,
}
pub struct Selected {
    pub decomposition: Decomposition,
    pub mappings: Vec<WorkMapping>,
    pub groups: Vec<u64>,
    pub dispatches: Vec<GroupDispatch>,
    pub decisions: Vec<seismic_compiler::tuner::family::Decision>,
}
/// Numeric operands and one bounded occurrence union for the target printer.
/// WorkMapping here only supplies structural axis identities; original numeric
/// values below own every emitted coordinate and the selected native dispatch.
pub(crate) struct WorkTemplate {
    pub source: super::source::Template,
    pub split: Value,
    pub retained_split: Option<crate::execution::RetainedSplit>,
    pub phases: Vec<Mapping>,
    pub structure: Vec<WorkMapping>,
    pub config: crate::execution::Config,
    pub partition_parameters: Vec<(usize, seismic_lang::types::DType)>,
    pub partition_active: VarId,
}
#[derive(Clone)]
pub(crate) struct Mapping {
    pub indices: Vec<seismic_lang::ir::VarId>,
    pub axes: Vec<MappingAxis>,
    pub work_items: Value,
    pub work_parameter: String,
    pub parts: Value,
    pub merge_work: Option<Value>,
}
#[derive(Clone)]
pub(crate) struct MappingAxis {
    pub extent: Value,
    pub step: Value,
    pub count: Value,
    pub extent_parameter: String,
    pub step_parameter: String,
    pub count_parameter: String,
}

fn interval(lo: u64, hi: u64) -> Result<Domain, Error> {
    let lo = i64::try_from(lo).map_err(|_| unsupported())?;
    let hi = i64::try_from(hi).map_err(|_| unsupported())?;
    Domain::interval(lo, hi).map_err(|e| Error::Invalid(e.to_string()))
}
fn unsupported() -> Error {
    Error::Unsupported("Metal decomposition exceeds the shared integer domain".into())
}
fn positive(builder: &mut ModelBuilder, name: &str, lo: u64, hi: u64) -> Result<Value, Error> {
    Symbolic::new(builder, name).positive("value", interval(lo, hi)?)
}
fn constant(builder: &mut ModelBuilder, name: &str, value: u64) -> Result<Value, Error> {
    Symbolic::new(builder, name).constant(value)
}
fn set(
    builder: &mut ModelBuilder,
    guards: Vec<Literal>,
    value: Value,
    expected: u64,
) -> Result<(), Error> {
    builder.guarded_constraint(
        guards,
        Constraint::InDomain {
            variable: value.id(),
            domain: interval(expected, expected)?,
        },
    );
    Ok(())
}
fn active_above(
    builder: &mut ModelBuilder,
    name: &str,
    value: Value,
    threshold: u64,
) -> Result<VarId, Error> {
    let (lo, hi) = value.bounds();
    let active = builder.variable(
        name,
        if lo > threshold {
            Domain::singleton(1)
        } else if hi <= threshold {
            Domain::singleton(0)
        } else {
            Domain::boolean()
        },
    );
    if hi > threshold {
        builder.guarded_constraint(
            vec![Literal::new(active, 1)],
            Constraint::InDomain {
                variable: value.id(),
                domain: interval(lo.max(threshold + 1), hi)?,
            },
        );
    }
    if lo <= threshold {
        builder.guarded_constraint(
            vec![Literal::new(active, 0)],
            Constraint::InDomain {
                variable: value.id(),
                domain: interval(lo, hi.min(threshold))?,
            },
        );
    }
    Ok(active)
}
fn conditional_count(builder: &mut ModelBuilder, name: &str, value: Value, presence: Option<VarId>) -> Result<Value, Error> {
    let Some(presence) = presence else { return Ok(value); };
    let count = Symbolic::new(builder, name).variable("count", interval(0, value.bounds().1)?)?;
    set(builder, vec![Literal::new(presence, 0)], count, 0)?;
    builder.guarded_constraint(vec![Literal::new(presence, 1)], Constraint::Equal { left: count.id(), right: value.id() });
    Ok(count)
}
fn presence_expression(parameters: &mut std::collections::BTreeMap<String, Value>, presence: Option<VarId>) -> Result<Sym, Error> {
    match presence {
        None => Ok(Sym::constant(1)),
        Some(variable) => {
            let name = format!("seismic_presence_{}__", variable.0);
            parameters.insert(name.clone(), Value::binding(variable, &Domain::boolean())?);
            Ok(Sym::param(&name))
        }
    }
}
impl Binding {
    pub(crate) fn work_template(&self, builder: &mut ModelBuilder,
        source: &super::source::Template, capacities: &Capacities,
        limits: seismic_accounting::workload::DerivationLimits) -> Result<WorkTemplate, Error> {
        use seismic_lang::{ir::{Expr, ExprKind, VarKind}, sym::{Atom, Sym}, types::DType,
            widen::parameterized::{Family as Widening, Guard}};
        let mut source = source.clone();
        source.function = self.source.clone();
        let split_symbol = format!("seismic_split_{}__", self.split.id().0);
        source.parameters.insert(split_symbol.clone(), self.split);
        let mut partition_extents = std::collections::BTreeMap::new();
        let partition_parameters = if self.phases.iter().any(|phase| phase.partition_axis.is_some()) {
            let mut pieces = Vec::new();
            for (phase_index, phase) in self.phases.iter().enumerate() {
                let axis = phase.partition_axis.as_ref().ok_or_else(|| Error::Invalid("partition proof must cover every phase".into()))?;
                let width = axis.extent.bounds().1;
                let piece = positive(builder, &format!("metal.phase{phase_index}.partition_piece"), 1, width)?;
                set(builder, vec![Literal::new(self.partitioned, 0)], piece, width)?;
                builder.guarded_constraint(vec![Literal::new(self.partitioned, 1)], Constraint::Equal { left: piece.id(), right: self.partition.id() });
                let parameter = format!("seismic_partition_{}_piece__", piece.id().0);
                source.parameters.insert(parameter.clone(), piece);
                let denominator = Sym::param(&parameter);
                partition_extents.insert(axis.extent.id(), Sym::constant(width as i64)
                    .add(&denominator).sub(&Sym::constant(1)).quot(&denominator));
                pieces.push(Sym::param(&parameter));
            }
            let partition = seismic_lang::partition::parameterized::apply(&source.function, &pieces).map_err(Error::Invalid)?;
            source.function = partition.function;
            for (original, copy) in partition.aliases {
                if let Some(predicate) = source.predicates.iter().find(|predicate| predicate.variable == original).cloned() {
                    source.predicates.push(super::source::Predicate { variable: copy, presence: predicate.presence });
                }
            }
            partition.parameters
        } else { Vec::new() };
        let mut mappings = Vec::new();
        let mut structure = Vec::new();
        let mut ordinary = std::collections::BTreeMap::new();
        let mut split_selector = None;
        let one = constant(builder, "metal.retained.parts", 1)?;
        let mut generated = 0usize;
        for (phase_index, (phase, statement)) in self.phases.iter().zip(&mut source.function.body).enumerate() {
            let StmtKind::Parallel { vars: indices, body, .. } = &mut statement.kind else {
                return Err(Error::Invalid("retained mapping requires a typed work domain".into()));
            };
            if indices.len() != phase.axes.len() + usize::from(phase.partition_axis.is_some()) { return Err(Error::Invalid("retained mapping rank differs from source".into())); }
            let split_body = phase.merge.as_ref().map(|_| body.clone());
            let mut axes = Vec::new();
            let mut extents = Vec::new();
            for (&index, axis) in indices.iter().zip(phase.axes.iter().chain(phase.partition_axis.as_ref())) {
                let extent = axis.extent.bounds();
                let extent_parameter = format!("seismic_mapping_{}_extent__", axis.extent.id().0);
                let step_parameter = format!("seismic_mapping_{}_step__", axis.step.id().0);
                let count_parameter = format!("seismic_mapping_{}_count__", axis.count.id().0);
                source.parameters.insert(extent_parameter.clone(), axis.extent);
                source.parameters.insert(step_parameter.clone(), axis.step);
                source.parameters.insert(count_parameter.clone(), axis.count);
                if let Some(definition) = partition_extents.get(&axis.extent.id()) {
                    source.numeric_definitions.insert(extent_parameter.clone(), definition.clone());
                }
                let step = Sym::param(&step_parameter);
                source.numeric_definitions.insert(count_parameter.clone(), Sym::param(&extent_parameter)
                    .add(&step).sub(&Sym::constant(1)).quot(&step));
                axes.push(MappingAxis { extent: axis.extent, step: axis.step, count: axis.count, extent_parameter, step_parameter, count_parameter });
                extents.push(extent.1);
                if !matches!(source.function.vars[index].kind, VarKind::Index(_)) { return Err(Error::Invalid("retained mapping index has no symbolic identity".into())); }
            }
            for (&inner, axis) in indices.iter().zip(&axes).rev() {
                if axis.step.bounds() == (1,1) { continue; }
                let VarKind::Index(atom) = &source.function.vars[inner].kind else { unreachable!() };
                let atom = atom.clone();
                let base = Expr { kind: ExprKind::Var(inner), ty: Ty::Scalar(DType::I32), sym: Some(Sym::atom(atom.clone())), span: statement.span };
                let width = Sym::param(&axis.step_parameter);
                let extent = axis.extent.bounds().1;
                let maximum = usize::try_from(axis.step.bounds().1).map_err(|_| unsupported())?;
                let occurrences = body.len().checked_mul(maximum).and_then(|count| count.checked_mul(3)).ok_or_else(unsupported)?;
                generated = generated.checked_add(occurrences).ok_or_else(unsupported)?;
                if generated > limits.operations { return Err(Error::Unsupported(format!("retained mapping occurrence construction exceeds operation limit {}", limits.operations))); }
                let retained = Widening::new(body, inner, &atom, width, Sym::param(&axis.extent_parameter), &source.function.vars, &base).map_err(Error::Invalid)?;
                let denominator = axis.step;
                let (quotient, remainder) = {
                    let maximum = i64::try_from(extent).map_err(|_| unsupported())?;
                    let quotient = builder.variable(format!("metal.mapping{phase_index}.complete"), Domain::interval(0, maximum).map_err(|e| Error::Invalid(e.to_string()))?);
                    let remainder = builder.variable(format!("metal.mapping{phase_index}.tail"), Domain::interval(0, denominator.bounds().1 as i64 - 1).map_err(|e| Error::Invalid(e.to_string()))?);
                    builder.constraint(Constraint::Arithmetic(magnitude_solver::model::Arithmetic::DivRem {
                        numerator: axis.extent.id(), denominator: denominator.id(), quotient, remainder }));
                    (quotient, remainder)
                };
                let _ = quotient;
                let remainder_domain = Domain::interval(0, denominator.bounds().1 as i64 - 1).map_err(|e| Error::Invalid(e.to_string()))?;
                let remainder_value = Value::binding(remainder, &remainder_domain)?;
                let mut guard_variables = std::collections::BTreeMap::new();
                let mut predicates = Vec::new();
                let union = retained.union(maximum, &mut source.function.vars, &mut |guard, then, els, vars| {
                    let threshold = match guard { Guard::WidthAtLeast(width) => (false, width), Guard::HasTail => (true, 1) };
                    let presence = if let Some(&presence) = guard_variables.get(&threshold) { presence } else {
                        let (value, threshold) = match guard { Guard::WidthAtLeast(width) => (denominator, width - 1), Guard::HasTail => (remainder_value, 0) };
                        let presence = active_above(builder, &format!("metal.mapping{phase_index}.guard{}", guard_variables.len()), value, threshold).map_err(|e| e.to_string())?;
                        guard_variables.insert(match guard { Guard::WidthAtLeast(width) => (false,width), Guard::HasTail => (true,1) }, presence);
                        presence
                    };
                    let (body, predicate) = super::source::predicate_region(vars, presence, then, els);
                    predicates.push(predicate); Ok(body)
                }).map_err(Error::Invalid)?;
                source.predicates.extend(predicates);
                for (original, copy) in union.aliases {
                    if let Some(predicate) = source.predicates.iter().find(|predicate| predicate.variable == original).cloned() {
                        source.predicates.push(super::source::Predicate { variable: copy, presence: predicate.presence });
                    }
                }
                *body = union.body;
            }
            if let Some(split_body) = split_body {
                let copied = seismic_lang::widen::parameterized::copy_bindings(body, &mut source.function.vars);
                for &(original, copy) in &copied.aliases {
                    if let Some(predicate) = source.predicates.iter().find(|predicate| predicate.variable == original).cloned() {
                        source.predicates.push(super::source::Predicate { variable: copy, presence: predicate.presence });
                    }
                }
                let (selected, predicate) = super::source::predicate_region(&mut source.function.vars, self.split_active, Vec::new(), copied.body);
                split_selector = Some(crate::msl::variable_symbol(&source.function.vars[predicate.variable], predicate.variable));
                source.predicates.push(predicate);
                ordinary.insert(phase_index, crate::execution::RetainedOrdinary { body: selected, aliases: copied.aliases });
                *body = split_body;
            }
            let work_parameter = format!("seismic_mapping_{}_work__", phase.main.work_items.id().0);
            source.parameters.insert(work_parameter.clone(), phase.main.work_items);
            let logical_work = axes.iter().fold(Sym::constant(1), |work, axis| work.mul(&Sym::param(&axis.count_parameter)));
            let main_work = if phase.merge.is_some() { logical_work.mul(&Sym::param(&split_symbol)) } else { logical_work.clone() };
            let presence = presence_expression(&mut source.parameters, phase.main.presence)?;
            source.numeric_definitions.insert(work_parameter.clone(), main_work.mul(&presence));
            if let Some(merge) = &phase.merge {
                let parameter = format!("seismic_mapping_{}_work__", merge.work_items.id().0);
                source.parameters.insert(parameter.clone(), merge.work_items);
                let presence = presence_expression(&mut source.parameters, merge.presence)?;
                source.numeric_definitions.insert(parameter, logical_work.mul(&presence));
            }
            mappings.push(Mapping { indices: indices.clone(), axes, work_items: phase.main.work_items, work_parameter,
                parts: if phase.merge.is_some() { self.split } else { one }, merge_work: phase.merge.as_ref().map(|merge| merge.work_items) });
            structure.push(WorkMapping::new(&extents, &vec![1; extents.len()]).map_err(Error::Invalid)?);
        }
        let config = capacities.config(&Decomposition::default()).map_err(Error::Invalid)?;
        let retained_split = split_selector.map(|selector| crate::execution::RetainedSplit {
            parts_symbol: split_symbol, selector, maximum: self.split.bounds().1 as i64, ordinary,
        });
        Ok(WorkTemplate { source, split: self.split, retained_split, phases: mappings, structure, config, partition_parameters, partition_active: self.partitioned })
    }
    pub fn append(
        builder: &mut ModelBuilder,
        name: &str,
        source: &LoweredIr,
        form: &Form,
        capacities: &Capacities,
    ) -> Result<Self, Error> {
        Self::append_with_presence(builder, name, source, form, capacities, &[], &std::collections::BTreeMap::new(), &std::collections::BTreeSet::new())
    }
    pub(crate) fn append_retained(builder: &mut ModelBuilder, name: &str, source: &super::source::Template,
        form: &Form, capacities: &Capacities) -> Result<Self, Error> {
        let numeric = source.parameters.iter().map(|(name, value)| {
            let (minimum, maximum) = value.bounds();
            Ok((name.clone(), (i64::try_from(minimum).map_err(|_| unsupported())?, i64::try_from(maximum).map_err(|_| unsupported())?)))
        }).collect::<Result<std::collections::BTreeMap<_, _>, Error>>()?;
        let selectors = source.predicates.iter().map(|predicate| predicate.variable).collect();
        Self::append_with_presence(builder, name, &source.function, form, capacities, &source.phase_presence, &numeric, &selectors)
    }
    fn append_with_presence(builder: &mut ModelBuilder, name: &str, source: &LoweredIr,
        form: &Form, capacities: &Capacities, phase_presence: &[Option<VarId>], numeric: &std::collections::BTreeMap<String, (i64, i64)>, selectors: &std::collections::BTreeSet<seismic_lang::ir::VarId>) -> Result<Self, Error> {
        let source = seismic_realization::phases::work_domains_retained(source, numeric, selectors).map_err(Error::Invalid)?;
        if !phase_presence.is_empty() && phase_presence.len() != source.body.len() { return Err(Error::Invalid("retained source phase presence differs from its work domains".into())); }
        let fixed = match form {
            Form::Automatic => None,
            Form::Fixed(value) => Some(value),
        };
        let partition_widths = if seismic_lang::partition::pointwise(&source, 1).is_ok() {
            Some(
                source
                    .body
                    .iter()
                    .map(|phase| {
                        let StmtKind::Parallel { body, .. } = &phase.kind else {
                            unreachable!()
                        };
                        body.iter()
                            .filter_map(|statement| {
                                let StmtKind::Assign { target, .. } = &statement.kind else {
                                    return None;
                                };
                                let Ty::Tile(tile) = &target.ty else {
                                    return None;
                                };
                                (tile.shape.len() == 1)
                                    .then(|| tile.shape[0].as_constant())
                                    .flatten()
                            })
                            .min()
                            .and_then(|n| u64::try_from(n).ok())
                            .ok_or_else(|| {
                                Error::Invalid("pointwise phase has no positive tile extent".into())
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let maximum_partition = partition_widths
            .as_ref()
            .and_then(|widths| widths.iter().min())
            .copied()
            .unwrap_or(0);
        let (partition_lo, partition_hi) = if let Some(fixed) = fixed {
            let piece = fixed
                .tile_piece
                .map(u64::try_from)
                .transpose()
                .map_err(|_| Error::Invalid("negative fixed partition".into()))?
                .unwrap_or(0);
            if piece > maximum_partition {
                return Err(Error::Invalid(
                    "fixed partition is outside its proven source domain".into(),
                ));
            }
            (piece, piece)
        } else {
            (0, maximum_partition)
        };
        let partition = Symbolic::new(builder, &format!("{name}.partition"))
            .variable("width", interval(partition_lo, partition_hi)?)?;
        let partitioned = active_above(builder, &format!("{name}.partition.active"), partition, 0)?;
        let one = constant(builder, name, 1)?;
        let partition_divisor = Symbolic::new(builder, name).maximum(partition, one)?;
        let maximum_split = crate::execution::split_domain(&source).map_err(Error::Invalid)?;
        let (split_lo, split_hi) = if let Some(fixed) = fixed {
            let value = u64::try_from(fixed.split)
                .map_err(|_| Error::Invalid("negative fixed split".into()))?;
            if value == 0 || value > maximum_split {
                return Err(Error::Invalid(
                    "fixed split is outside its proven source domain".into(),
                ));
            }
            (value, value)
        } else {
            (1, maximum_split)
        };
        let split = positive(builder, &format!("{name}.split"), split_lo, split_hi)?;
        let split_active = active_above(builder, &format!("{name}.split.active"), split, 1)?;
        set(builder, vec![Literal::new(partitioned, 1)], split, 1)?;
        let split_phases = if split_hi > 1 {
            let split_source = seismic_lang::reduction::structured::materialize(&source)
                .map_err(Error::Invalid)?;
            seismic_lang::split::split_candidates(&split_source.body, &split_source.vars)
                .into_iter()
                .map(|candidate| candidate.stmt)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let maximum_groups = capacities.max_threads_per_threadgroup / SUBGROUP as u64;
        if maximum_groups == 0 {
            return Err(Error::Invalid(
                "Metal requires at least one complete SIMD group".into(),
            ));
        }
        let mut decisions = vec![
            (format!("{name}.partition"), partition),
            (format!("{name}.split"), split),
        ];
        let mut phases = Vec::new();
        for (phase_index, statement) in source.body.iter().enumerate() {
            let presence = phase_presence.get(phase_index).copied().flatten();
            let StmtKind::Parallel { extents, .. } = &statement.kind else {
                return Err(Error::Invalid(
                    "Metal work domains require parallel phases".into(),
                ));
            };
            let mut axes = Vec::new();
            let mut zero_extent = false;
            for (axis_index, extent) in extents.iter().enumerate() {
                let extent = extent
                    .as_constant()
                    .and_then(|n| u64::try_from(n).ok())
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "Metal mapping needs a retained symbolic extent binding".into(),
                        )
                    })?;
                if extent > i32::MAX as u64 {
                    return Err(Error::Invalid(
                        "Metal logical extent exceeds its index type".into(),
                    ));
                }
                zero_extent |= extent == 0;
                let label = format!("{name}.phase{phase_index}.axis{axis_index}");
                let (lo, hi) = if let Some(fixed) = fixed {
                    let step = if axis_index + 1 == extents.len() && fixed.tile_piece.is_none() {
                        fixed.per_item
                    } else {
                        1
                    };
                    let step = u64::try_from(step)
                        .map_err(|_| Error::Invalid("negative fixed mapping".into()))?;
                    if step == 0 {
                        return Err(Error::Invalid("zero fixed mapping step".into()));
                    }
                    (step, step)
                } else {
                    (1, extent.max(1))
                };
                let step = positive(builder, &label, lo, hi)?;
                if fixed.is_none() {
                    if let Some(presence) = presence { set(builder, vec![Literal::new(presence, 0)], step, 1)?; }
                }
                set(builder, vec![Literal::new(split_active, 1)], step, 1)?;
                let extent = constant(builder, &label, extent)?;
                let count = Symbolic::new(builder, &label).ceil_div(extent, step)?;
                axes.push(Axis {
                    extent,
                    step,
                    count,
                });
                decisions.push((label, step));
            }
            let partition_axis = if let Some(widths) = &partition_widths {
                let label = format!("{name}.phase{phase_index}.partition_axis");
                let width = constant(builder, &label, widths[phase_index])?;
                let partition_count =
                    Symbolic::new(builder, &label).ceil_div(width, partition_divisor)?;
                let extent = positive(builder, &label, 1, widths[phase_index])?;
                set(builder, vec![Literal::new(partitioned, 0)], extent, 1)?;
                builder.guarded_constraint(
                    vec![Literal::new(partitioned, 1)],
                    Constraint::Equal {
                        left: extent.id(),
                        right: partition_count.id(),
                    },
                );
                let fixed_step = fixed.map(|d| {
                    if d.tile_piece.is_some() {
                        d.per_item as u64
                    } else {
                        1
                    }
                });
                let step = positive(
                    builder,
                    &format!("{label}.step"),
                    fixed_step.unwrap_or(1),
                    fixed_step.unwrap_or(widths[phase_index]),
                )?;
                if fixed.is_none() {
                    if let Some(presence) = presence { set(builder, vec![Literal::new(presence, 0)], step, 1)?; }
                    builder.constraint(Constraint::LinearLe {
                        terms: vec![
                            LinearTerm::new(step.id(), 1),
                            LinearTerm::new(extent.id(), -1),
                        ],
                        rhs: 0,
                    });
                }
                set(builder, vec![Literal::new(partitioned, 0)], step, 1)?;
                // Fixed per_item applies to the innermost transformed axis.
                if fixed.is_some_and(|d| d.tile_piece.is_some()) {
                    if let Some(axis) = axes.last() {
                        set(builder, vec![], axis.step, 1)?;
                    }
                }
                let count = Symbolic::new(builder, &label).ceil_div(extent, step)?;
                decisions.push((label, step));
                Some(Axis {
                    extent,
                    step,
                    count,
                })
            } else {
                None
            };
            let mut work_items = constant(builder, name, u64::from(!zero_extent))?;
            for axis in axes.iter().chain(partition_axis.as_ref()) {
                work_items = Symbolic::new(builder, name).product(work_items, axis.count)?;
            }
            let split_phase = split_phases.contains(&phase_index);
            let main_work = if split_phase {
                Symbolic::new(builder, name).product(work_items, split)?
            } else {
                work_items
            };
            let main_work = conditional_count(builder, &format!("{name}.phase{phase_index}.active_work"), main_work, presence)?;
            let main = Launch::append(
                builder,
                &format!("{name}.phase{phase_index}.main"),
                main_work,
                presence,
                maximum_groups,
            )?;
            decisions.push((
                format!("{name}.phase{phase_index}.main.grouping"),
                main.items_per_group,
            ));
            let merge = if split_phase {
                let merge_presence = match presence {
                    None => split_active,
                    Some(presence) => {
                        let active = builder.variable(format!("{name}.phase{phase_index}.merge.active"), Domain::boolean());
                        builder.constraint(Constraint::BoolAnd { output: active, inputs: vec![presence, split_active] }); active
                    },
                };
                let merge_work = conditional_count(builder, &format!("{name}.phase{phase_index}.merge.active_work"), work_items, Some(merge_presence))?;
                let merge = Launch::append(
                    builder,
                    &format!("{name}.phase{phase_index}.merge"),
                    merge_work,
                    Some(merge_presence),
                    maximum_groups,
                )?;
                set(
                    builder,
                    vec![Literal::new(split_active, 0)],
                    merge.items_per_group,
                    1,
                )?;
                decisions.push((
                    format!("{name}.phase{phase_index}.merge.grouping"),
                    merge.items_per_group,
                ));
                Some(merge)
            } else {
                None
            };
            phases.push(Phase {
                axes,
                partition_axis,
                work_items,
                main,
                merge,
            });
        }
        Ok(Self {
            partition,
            split,
            partitioned,
            split_active,
            phases,
            source,
            form: form.clone(),
            decisions,
        })
    }
    pub fn fixed(&self) -> bool {
        self.decisions
            .iter()
            .all(|(_, value)| value.bounds().0 == value.bounds().1)
    }
    pub(crate) fn launches(&self) -> Vec<&Launch> {
        self.phases.iter().flat_map(|phase| std::iter::once(&phase.main)
            .chain(phase.merge.iter())).collect()
    }
    pub fn reconstruct(&self, values: &[i64]) -> Result<Selected, Error> {
        let partition = read(values, self.partition)?;
        let split = read(values, self.split)?;
        let decomposition = Decomposition {
            tile_piece: (partition != 0).then_some(partition as i64),
            split: split as i64,
            per_item: match &self.form {
                Form::Automatic => 1,
                Form::Fixed(value) => value.per_item,
            },
        };
        let mut mappings = Vec::new();
        let mut groups = Vec::new();
        let mut dispatches = Vec::new();
        for phase in &self.phases {
            let axes = phase
                .axes
                .iter()
                .chain(phase.partition_axis.as_ref())
                .collect::<Vec<_>>();
            let extents = axes
                .iter()
                .map(|axis| read(values, axis.extent))
                .collect::<Result<Vec<_>, _>>()?;
            if extents.len() != axes.len() {
                return Err(Error::Reconstruction(
                    "selected Metal mapping rank differs from its retained template".into(),
                ));
            }
            let steps = axes
                .iter()
                .map(|axis| read(values, axis.step))
                .collect::<Result<Vec<_>, _>>()?;
            let mapping = WorkMapping::new(&extents, &steps).map_err(Error::Reconstruction)?;
            for (axis, selected) in axes.iter().zip(mapping.axes()) {
                check(values, axis.extent, selected.logical_extent)?;
                check(values, axis.count, selected.extent)?;
            }
            check(values, phase.work_items, mapping.work_items())?;
            let main = phase.main.reconstruct(values)?;
            let present = active(values, phase.main.presence)?;
            check(
                values,
                phase.main.work_items,
                if !present { 0 } else { mapping
                    .work_items()
                    .checked_mul(if phase.merge.is_some() { split } else { 1 })
                    .ok_or_else(unsupported)? },
            )?;
            groups.push(main.items_per_group);
            dispatches.push(main);
            if let Some(merge) = &phase.merge {
                let merge_present = active(values, merge.presence)?;
                if merge_present != (present && split > 1) { return Err(Error::Reconstruction("merge presence differs from its original split phase".into())); }
                check(values, merge.work_items, if merge_present { mapping.work_items() } else { 0 })?;
                let dispatch = merge.reconstruct(values)?;
                groups.push(dispatch.items_per_group);
                dispatches.push(dispatch);
            }
            mappings.push(mapping);
        }
        let decisions = self
            .decisions
            .iter()
            .map(|(identity, value)| {
                Ok(seismic_compiler::tuner::family::Decision {
                    identity: identity.clone(),
                    value: read(values, *value)? as i64,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Selected {
            decomposition,
            mappings,
            groups,
            dispatches,
            decisions,
        })
    }
    /// Available only when the complete original decomposition domain is a
    /// singleton. This is never used to select a representative of a family.
    pub fn unique(&self) -> Result<Option<Selected>, Error> {
        if !self.fixed() { return Ok(None); }
        self.canonical_work()
    }
    /// Retain unresolved grouping while constructing the invariant work-item
    /// implementation. Grouping operands are introduced by the terminal printer;
    /// the returned group values only describe its canonical coordinate frame.
    pub(crate) fn canonical_work(&self) -> Result<Option<Selected>, Error> {
        if self.decisions.iter().filter(|(name, _)| !name.ends_with(".grouping"))
            .any(|(_, value)| value.bounds().0 != value.bounds().1) { return Ok(None); }
        let partition = self.partition.bounds().0;
        let split = self.split.bounds().0;
        let decomposition = Decomposition {
            tile_piece: (partition != 0).then_some(partition as i64),
            split: split as i64,
            per_item: match &self.form {
                Form::Automatic => 1,
                Form::Fixed(value) => value.per_item,
            },
        };
        let transformed = if let Some(piece) = decomposition.tile_piece {
            seismic_lang::partition::pointwise(&self.source, piece)
                .map_err(Error::Reconstruction)?
                .function
        } else {
            self.source.clone()
        };
        let mut mappings = Vec::new();
        let mut groups = Vec::new();
        for (phase, statement) in self.phases.iter().zip(&transformed.body) {
            let StmtKind::Parallel { extents, .. } = &statement.kind else {
                unreachable!()
            };
            let extents = extents
                .iter()
                .map(|extent| {
                    extent
                        .as_constant()
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or_else(|| {
                            Error::Invalid("unique Metal phase extent is not specialized".into())
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let steps = phase
                .axes
                .iter()
                .chain(phase.partition_axis.as_ref().filter(|_| partition != 0))
                .map(|axis| axis.step.bounds().0)
                .collect::<Vec<_>>();
            mappings.push(WorkMapping::new(&extents, &steps).map_err(Error::Reconstruction)?);
            groups.push(phase.main.items_per_group.bounds().0);
            if split > 1 {
                if let Some(merge) = &phase.merge {
                    groups.push(merge.items_per_group.bounds().0);
                }
            }
        }
        Ok(Some(Selected {
            decomposition,
            mappings,
            dispatches: self.phases.iter().flat_map(|phase| std::iter::once(&phase.main)
                .chain(phase.merge.as_ref().filter(|_| split > 1))).map(|launch|
                GroupDispatch::new(launch.work_items.bounds().0, SUBGROUP as u64, launch.items_per_group.bounds().0)
                    .map_err(Error::Reconstruction)).collect::<Result<Vec<_>, _>>()?,
            groups,
            decisions: self
                .decisions
                .iter()
                .map(
                    |(identity, value)| seismic_compiler::tuner::family::Decision {
                        identity: identity.clone(),
                        value: value.bounds().0 as i64,
                    },
                )
                .collect(),
        }))
    }
}
impl Launch {
    fn append(
        builder: &mut ModelBuilder,
        name: &str,
        work_items: Value,
        presence: Option<VarId>,
        maximum_groups: u64,
    ) -> Result<Self, Error> {
        let items_per_group = positive(builder, &format!("{name}.grouping"), 1, maximum_groups)?;
        if let Some(presence) = presence {
            builder.constraint(Constraint::InactiveValue { active: Literal::new(presence, 1), variable: items_per_group.id(), inactive: 1 });
        }
        let mut algebra = Symbolic::new(builder, name);
        let lanes = algebra.constant(SUBGROUP as u64)?;
        let geometry = geometry::dispatch(&mut algebra, work_items, lanes, items_per_group)?;
        builder.constraint(Constraint::LinearLe {
            terms: vec![LinearTerm::new(geometry.dispatched_lanes.id(), 1)],
            rhs: i128::from(u32::MAX) * i128::from(SUBGROUP),
        });
        Ok(Self {
            presence,
            work_items,
            items_per_group,
            groups: geometry.groups,
            threads_per_group: geometry.threads_per_group,
            dispatched_lanes: geometry.dispatched_lanes,
        })
    }
    pub fn reconstruct(&self, values: &[i64]) -> Result<GroupDispatch, Error> {
        let selected = GroupDispatch::new(
            read(values, self.work_items)?,
            SUBGROUP as u64,
            read(values, self.items_per_group)?,
        )
        .map_err(Error::Reconstruction)?;
        check(values, self.groups, selected.groups)?;
        check(values, self.threads_per_group, selected.threads_per_group)?;
        check(values, self.dispatched_lanes, selected.dispatched_lanes())?;
        Ok(selected)
    }
}
fn read(values: &[i64], value: Value) -> Result<u64, Error> {
    values
        .get(value.id().0)
        .and_then(|n| u64::try_from(*n).ok())
        .filter(|&selected| value.bounds().0 <= selected && selected <= value.bounds().1)
        .ok_or_else(|| Error::Reconstruction("missing or out-of-domain Metal geometry value".into()))
}
fn active(values: &[i64], presence: Option<VarId>) -> Result<bool, Error> {
    match presence.map(|presence| values.get(presence.0)) {
        None | Some(Some(1)) => Ok(true),
        Some(Some(0)) => Ok(false),
        _ => Err(Error::Reconstruction("missing or invalid Metal phase presence".into())),
    }
}
fn check(values: &[i64], value: Value, expected: u64) -> Result<(), Error> {
    if read(values, value)? != expected {
        return Err(Error::Reconstruction(
            "Metal geometry differs from its selected implementation".into(),
        ));
    }
    Ok(())
}
