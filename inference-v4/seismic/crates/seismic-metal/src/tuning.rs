//! Metal's implementation choices, shared by diagnostics and compiler selection.
use crate::{
    execution::{Config, Execution},
    model,
};
use seismic_accounting::{
    schedule,
    selection::{Choices, Domain, Objective},
    workload::{DerivationError, DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner::{self as compiler, Preparation};
use seismic_lang::lowered_ir::LoweredIr;
use std::sync::Arc;
mod bounds;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decomposition {
    pub per_item: i64,
    pub split: i64,
    pub tile_piece: Option<i64>,
}
impl Default for Decomposition {
    fn default() -> Self {
        Self {
            per_item: 1,
            split: 1,
            tile_piece: None,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Form {
    #[default]
    Automatic,
    /// Explicit diagnostic restriction, never the default production domain.
    Fixed(Decomposition),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capacities {
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
}
impl Capacities {
    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Self {
        Self {
            max_threads_per_threadgroup: device.max_threads_per_threadgroup,
            max_threadgroup_bytes: device.max_threadgroup_bytes,
        }
    }
    fn config(&self, form: &Decomposition) -> Result<Config, String> {
        Ok(Config {
            loads: seismic_realization::LoadStrategy::Materialize,
            sg_per_tg: 1,
            piece: None,
            per_item: form.per_item,
            split: form.split,
            tile_piece: form.tile_piece,
            max_threads_per_threadgroup: self
                .max_threads_per_threadgroup
                .try_into()
                .map_err(|_| "Metal thread capacity exceeds i64")?,
            max_threadgroup_bytes: self
                .max_threadgroup_bytes
                .try_into()
                .map_err(|_| "Metal shared capacity exceeds i64")?,
        })
    }
}
pub fn expand(
    function: &LoweredIr,
    form: &Form,
    capacities: &Capacities,
    path: &[usize],
) -> Result<Preparation<Execution>, String> {
    expand_retained(Arc::new(function.clone()), form, capacities, path)
}

fn expand_retained(
    function: Arc<LoweredIr>,
    form: &Form,
    capacities: &Capacities,
    path: &[usize],
) -> Result<Preparation<Execution>, String> {
    if let seismic_realization::phases::Applicability::Unresolved { reason } =
        seismic_realization::phases::assess(&function)?
    {
        return Ok(Preparation::Unresolved(format!("Metal phase realization: {reason}")));
    }
    let (decomposition, mappings, consumed) = match decomposition(&function, form, path)? {
        DecompositionExpansion::Choice { name, alternatives, dispatch } => {
            return Ok(Preparation::Choice {
                name,
                alternatives: Domain::new(DecompositionChoice {
                    publications: bounds::Publications::derive(&function),
                    dispatch,
                    function, form: form.clone(), capacities: capacities.clone(),
                    prefix: path.to_vec(), alternatives,
                })?,
            });
        }
        DecompositionExpansion::Selected {
            value,
            mappings,
            consumed,
        } => (value, mappings, consumed),
    };
    let path = &path[consumed..];
    match crate::choices::expand_with_mappings(
        &function,
        capacities.config(&decomposition)?,
        mappings.as_deref(),
        path,
    )? {
        crate::choices::Expansion::Choice(domain) => Ok(Preparation::Choice {
            name: format!("Metal {:?}", domain.decision()),
            alternatives: Domain::new(domain)?,
        }),
        crate::choices::Expansion::Infeasible {
            launch,
            required,
            available,
        } => Ok(Preparation::Infeasible(
            seismic_accounting::selection::CapacityViolation {
                resource: format!("Metal launch {launch} shared bytes per group"),
                required,
                available,
            },
        )),
        crate::choices::Expansion::Execution {
            execution,
            consumed,
        } => {
            let family = std::sync::Arc::new(crate::family::GroupFamily::derive(execution)?);
            let mut prepared = family.next(Vec::new())?;
            for &index in &path[consumed..] {
                let Preparation::Choice { alternatives, .. } = &prepared else {
                    return Err("unused Metal execution decisions".into());
                };
                prepared = alternatives
                    .owner::<crate::family::GroupingChoices>()
                    .expect("remaining Metal choices are retained launch groupings")
                    .refine(index)?;
            }
            Ok(prepared)
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conditions {
    pub target: String,
    pub capacities: Capacities,
    pub form: Form,
    pub hardware: model::Hardware,
}
pub struct Backend {
    conditions: Conditions,
}
impl Backend {
    pub fn with_conditions(conditions: Conditions) -> Result<Self, String> {
        conditions.capacities.config(&Decomposition::default())?;
        conditions.hardware.validate()?;
        if conditions.target.is_empty() {
            return Err("Metal target identity is empty".into());
        }
        Ok(Self { conditions })
    }
    #[cfg(target_os = "macos")]
    pub fn new(
        device: &crate::runtime::DeviceInfo,
        hardware: &model::Hardware,
        form: &Form,
    ) -> Result<Self, String> {
        Self::with_conditions(Conditions {
            target: device.name.clone(),
            capacities: Capacities::from_device(device),
            hardware: hardware.clone(),
            form: form.clone(),
        })
    }
}
impl compiler::Backend for Backend {
    type Execution = Execution;
    type Conditions = Conditions;
    fn name(&self) -> &'static str {
        "metal"
    }
    fn conditions(&self) -> Conditions {
        self.conditions.clone()
    }
    fn description(&self) -> compiler::Description {
        compiler::Description {
            target: self.conditions.target.clone(),
            contracts: self.conditions.hardware.identity.clone(),
            form: format!("Metal {:?}", self.conditions.form),
            objective: "conditional retained MSL completion; native mapping unqualified".into(),
            timebase: self.conditions.hardware.timebase.clone(),
        }
    }
    fn prepare(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<Preparation<Execution>, String> {
        expand(
            function,
            &self.conditions.form,
            &self.conditions.capacities,
            path,
        )
    }
    fn refine(
        &self,
        alternatives: &Domain,
        index: usize,
    ) -> Result<Option<Preparation<Execution>>, String> {
        if let Some(choice) = alternatives.owner::<DecompositionChoice>() {
            if choice.form != self.conditions.form || choice.capacities != self.conditions.capacities {
                return Err("retained Metal decomposition has different execution conditions".into());
            }
            return choice.refine(index).map(Some);
        }
        if let Some(choice) = alternatives.owner::<crate::choices::ExecutionChoice>() {
            let config = choice.config();
            if u64::try_from(config.max_threads_per_threadgroup).ok()
                != Some(self.conditions.capacities.max_threads_per_threadgroup)
                || u64::try_from(config.max_threadgroup_bytes).ok()
                    != Some(self.conditions.capacities.max_threadgroup_bytes)
            {
                return Err("retained Metal preparation has different device capacities".into());
            }
            return Ok(Some(match choice.refine(index)? {
                crate::choices::Expansion::Choice(choice) => Preparation::Choice {
                    name: format!("Metal {:?}", choice.decision()),
                    alternatives: Domain::new(choice)?,
                },
                crate::choices::Expansion::Infeasible {
                    launch,
                    required,
                    available,
                } => Preparation::Infeasible(seismic_accounting::selection::CapacityViolation {
                    resource: format!("Metal launch {launch} shared bytes per group"),
                    required,
                    available,
                }),
                crate::choices::Expansion::Execution { execution, .. } => {
                    std::sync::Arc::new(crate::family::GroupFamily::derive(execution)?)
                        .next(Vec::new())?
                }
            }));
        }
        let Some(choice) = alternatives.owner::<crate::family::GroupingChoices>() else {
            return Ok(None);
        };
        let config = &choice.family().execution().config;
        if u64::try_from(config.max_threads_per_threadgroup).ok()
            != Some(self.conditions.capacities.max_threads_per_threadgroup)
            || u64::try_from(config.max_threadgroup_bytes).ok()
                != Some(self.conditions.capacities.max_threadgroup_bytes)
        {
            return Err("retained Metal group family has different device capacities".into());
        }
        choice.refine(index).map(Some)
    }
    fn relax(
        &self,
        alternatives: &Domain,
        indices: std::ops::Range<usize>,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        if indices.is_empty() || indices.end > alternatives.len() {
            return Err("Metal relaxation needs a nonempty subdomain".into());
        }
        if let Some(choice) = alternatives.owner::<seismic_lang::lower::alternatives::LoweringChoice>() {
            let mut demand = schedule::Demand::new(self.conditions.hardware.timebase.clone(), self.conditions.hardware.resources.clone())?;
            bounds::Publications::derive(choice.prepared()).include(&mut demand, &self.conditions.hardware)?;
            return Ok(Some(demand));
        }
        if let Some(choice) = alternatives.owner::<DecompositionChoice>() {
            let mut demand = match choice.dispatch.demand(&choice.alternatives, indices,
                self.conditions.capacities.max_threads_per_threadgroup, &self.conditions.hardware)? {
                Some(demand) => demand,
                None => schedule::Demand::new(self.conditions.hardware.timebase.clone(), self.conditions.hardware.resources.clone())?,
            };
            choice.publications.include(&mut demand, &self.conditions.hardware)?;
            return Ok(Some(demand));
        }
        if let Some(choice) = alternatives.owner::<crate::family::GroupingChoices>() {
            return choice.relax(indices, &self.conditions.hardware, workload, limits);
        }
        if let Some(choice) = alternatives.owner::<crate::choices::ExecutionChoice>() {
            return choice.relax(&self.conditions.hardware, workload, limits);
        }
        Ok(None)
    }
    fn analyze(
        &self,
        execution: &Execution,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<schedule::evaluation::Model, DerivationError> {
        Ok(schedule::evaluation::Model::Structured {
            model: model::structured_execution(execution, &self.conditions.hardware, workload, limits)?,
            expansion_limit: limits.operations as u64,
        })
    }
    fn relax_execution(
        &self,
        execution: &Execution,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        let account = execution.relaxation(workload, limits)?;
        Ok(Some(model::relaxed_demand(&account, &self.conditions.hardware)?))
    }
    fn materialize(&self, execution: &Execution, _: &Objective) -> Result<Execution, String> {
        // Target statement order is fixed in this form. The solver only schedules
        // concurrent workgroups under the explicitly conditional machine model.
        Ok(execution.clone())
    }
    fn check_materialization(
        &self,
        source: &Execution,
        selected: &Execution,
        _: &Objective,
    ) -> Result<(), String> {
        if !source.same_implementation(selected) {
            return Err("Metal materialization changed the selected implementation".into());
        }
        Ok(())
    }
}

/// Decomposition still owns the complete lowered computation. Refining a
/// mapping or split must not replay portable lowering and its earlier choices.
/// Its typed numeric domain remains the exact legal domain, with no preferred
/// widths or implicit completion of unresolved decisions.
#[derive(Clone, PartialEq)]
pub struct DecompositionChoice {
    function: Arc<LoweredIr>,
    form: Form,
    capacities: Capacities,
    prefix: Vec<usize>,
    alternatives: Domain,
    publications: bounds::Publications,
    dispatch: bounds::Dispatch,
}
impl std::fmt::Debug for DecompositionChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecompositionChoice")
            .field("entry", &self.function.name)
            .field("prefix", &self.prefix)
            .field("alternatives", &self.alternatives)
            .finish()
    }
}
impl Choices for DecompositionChoice {
    type Alternative = String;
    fn len(&self) -> usize { self.alternatives.len() }
    fn get(&self, index: usize) -> Option<String> { self.alternatives.label(index) }
}
impl DecompositionChoice {
    pub fn alternatives(&self) -> &Domain { &self.alternatives }
    pub fn prepared(&self) -> &LoweredIr { &self.function }
    pub fn refine(&self, index: usize) -> Result<Preparation<Execution>, String> {
        if index >= self.alternatives.len() { return Err("Metal decomposition choice is outside its domain".into()); }
        let mut prefix = self.prefix.clone();
        prefix.push(index);
        expand_retained(self.function.clone(), &self.form, &self.capacities, &prefix)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingDecision {
    pub phase: usize,
    pub axis: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitDecision;
/// Optional independent tile partition, followed by dependent ownership domains
/// over the transformed axes. The unpartitioned source remains one alternative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionChoices {
    pub maximum: u64,
}
impl Choices for PartitionChoices {
    type Alternative = Option<i64>;
    fn len(&self) -> usize {
        self.maximum as usize + 1
    }
    fn get(&self, index: usize) -> Option<Self::Alternative> {
        if index == 0 {
            Some(None)
        } else if (index as u64) <= self.maximum {
            Some(Some(index as i64))
        } else {
            None
        }
    }
}
enum DecompositionExpansion {
    Choice {
        name: String,
        alternatives: Domain,
        dispatch: bounds::Dispatch,
    },
    Selected {
        value: Decomposition,
        mappings: Option<Vec<seismic_realization::dispatch::WorkMapping>>,
        consumed: usize,
    },
}
fn decomposition(
    function: &LoweredIr,
    form: &Form,
    path: &[usize],
) -> Result<DecompositionExpansion, String> {
    if let Form::Fixed(value) = form {
        return Ok(DecompositionExpansion::Selected {
            value: value.clone(),
            mappings: None,
            consumed: 0,
        });
    }
    let mut normalized = seismic_realization::phases::work_domains(function)?;
    let mut consumed = 0;
    let mut value = Decomposition::default();
    if seismic_lang::partition::pointwise(&normalized, 1).is_ok() {
        let widths = normalized
            .body
            .iter()
            .flat_map(|phase| match &phase.kind {
                seismic_lang::ir::StmtKind::Parallel { body, .. } => body.as_slice(),
                _ => &[],
            })
            .filter_map(|statement| {
                let seismic_lang::ir::StmtKind::Assign { target, .. } = &statement.kind else {
                    return None;
                };
                match &target.ty {
                    seismic_lang::types::Ty::Tile(t) if t.shape.len() == 1 => {
                        t.shape[0].as_constant().and_then(|n| u64::try_from(n).ok())
                    }
                    _ => None,
                }
            });
        let maximum = widths.min().ok_or("pointwise domain has no tile extent")?;
        let domain = PartitionChoices { maximum };
        let Some(&index) = path.get(consumed) else {
            return Ok(DecompositionExpansion::Choice {
                name: "Metal independent tile partition".into(),
                alternatives: Domain::new(domain)?,
                dispatch: bounds::Dispatch::derive(&normalized, &[], &[], bounds::Remaining::Partition)?,
            });
        };
        consumed += 1;
        value.tile_piece = domain.get(index).ok_or("Metal partition outside domain")?;
        if let Some(piece) = value.tile_piece {
            normalized = seismic_lang::partition::pointwise(&normalized, piece)?.function;
        }
    }
    let mut mappings = Vec::new();
    let mut scalar_mapping = true;
    for (phase, statement) in normalized.body.iter().enumerate() {
        let seismic_lang::ir::StmtKind::Parallel { extents, .. } = &statement.kind else {
            return Err("Metal decomposition requires normalized parallel regions".into());
        };
        let extents = extents
            .iter()
            .map(|e| {
                e.as_constant()
                    .and_then(|n| u64::try_from(n).ok())
                    .ok_or("Metal mapping needs a nonnegative specialized extent")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut steps = Vec::new();
        for (axis, &extent) in extents.iter().enumerate() {
            let domain = seismic_accounting::selection::IntegerRange::new(
                MappingDecision { phase, axis },
                1,
                extent.max(1),
            )?;
            let Some(&index) = path.get(consumed) else {
                return Ok(DecompositionExpansion::Choice {
                    name: format!("Metal phase {phase} axis {axis} coordinates per work item"),
                    alternatives: Domain::new(domain)?,
                    dispatch: bounds::Dispatch::derive(&normalized, &mappings, &steps,
                        bounds::Remaining::Mapping { phase, extent })?,
                });
            };
            consumed += 1;
            let step = domain.get(index).ok_or("Metal mapping outside domain")?;
            scalar_mapping &= step == 1;
            steps.push(step);
        }
        mappings.push(seismic_realization::dispatch::WorkMapping::new(
            &extents, &steps,
        )?);
    }
    if scalar_mapping && value.tile_piece.is_none() {
        let maximum = crate::execution::split_domain(&normalized)?;
        if maximum > 1 {
            let domain =
                seismic_accounting::selection::IntegerRange::new(SplitDecision, 1, maximum)?;
            let Some(&index) = path.get(consumed) else {
                return Ok(DecompositionExpansion::Choice {
                    name: "Metal reduction partitions".into(),
                    alternatives: Domain::new(domain)?,
                    dispatch: bounds::Dispatch::derive(&normalized, &mappings, &[],
                        bounds::Remaining::Split { phases: seismic_lang::split::split_candidates(&normalized.body, &normalized.vars)
                            .iter().map(|candidate| candidate.stmt).collect() })?,
                });
            };
            consumed += 1;
            value.split = domain
                .get(index)
                .ok_or("Metal reduction split outside domain")? as i64;
        }
    }
    Ok(DecompositionExpansion::Selected {
        value,
        mappings: Some(mappings),
        consumed,
    })
}
