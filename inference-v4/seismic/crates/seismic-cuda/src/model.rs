//! Resource derivation from the retained terminal PTX implementation.
//!
//! This is an explicitly hypothetical instruction-preserving machine model. PTX
//! is a virtual ISA: this module does not assert a native instruction mapping,
//! register allocation, cache behavior, warp reconvergence, or block placement.
//! Those premises are retained, never inferred from a device name or timings.
mod trace;

use crate::{execution::Execution, ptx};
use seismic_accounting::{
    schedule,
    workload::{DerivationError, DerivationLimit, DerivationLimits, ScalarWorkload},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    HypotheticalInstructionPreservingPtxV1,
}

/// Defines the cohort trace, rather than silently assuming a physical NVIDIA
/// reconvergence policy. Lanes at the lowest live PTX position issue together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CohortPolicy {
    LowestPosition,
}

/// Exact reduction of identical block residency vectors to a homogeneous pool.
/// Interval coloring partitions every feasible block schedule into unit_count*k
/// resident slots, then k slots per unit. Service resources are separate explicit
/// hypotheses; this does not pool or infer per-unit execution throughput.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    HomogeneousResidentSlots,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceScope {
    Device,
    Block,
    Warp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resource {
    pub name: String,
    pub scope: ResourceScope,
    pub capacity: u64,
    pub unit: schedule::CapacityUnit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Launch,
    BlockAdmission,
    WarpStart,
    WarpCompletion,
    BlockCompletion,
    Completion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requirement {
    Instruction(ptx::Primitive),
    WholeBody(ptx::Helper),
    Lifecycle(Lifecycle),
}

/// Quantities derived from the implementation and active-lane address geometry.
/// MemorySectors means address coverage, not physical cache misses or traffic.
/// VirtualRegisterBits is declaration storage, not native register allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    One,
    IssuedLanes,
    ActiveLanes,
    RequestedBytes,
    MemorySectors { bytes: u64 },
    BlockThreads,
    BlockWarps,
    VirtualRegisterBits,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Amount {
    pub quantity: Quantity,
    pub scale: u64,
}
impl Amount {
    pub const fn fixed(units: u64) -> Self {
        Self {
            quantity: Quantity::One,
            scale: units,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ticks {
    Fixed(u64),
    /// A declared service law in model ticks. No measured candidate score.
    Service {
        demand: Amount,
        per_tick: u64,
        base: u64,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub resource: usize,
    pub offset: u64,
    pub duration: Ticks,
    pub units: Amount,
}
/// Hardware timing and resource service for one retained PTX primitive.
/// Instruction expansion, control and dependencies are owned by TargetPlan;
/// this input cannot supply extra tasks or an alternative implementation graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimitiveTiming {
    pub primitive: ptx::Primitive,
    pub latency: Ticks,
    pub reservations: Vec<Reservation>,
}

/// Capacity held for a block's actual scheduled lifetime. The quantity's mapping
/// to physical resident units is a hypothesis, especially for virtual registers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residency {
    pub resource: usize,
    pub units: Amount,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitResidency {
    pub name: String,
    pub capacity: u64,
    pub units_per_block: Amount,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaHardware {
    pub identity: String,
    pub scope: Scope,
    pub timebase: schedule::Timebase,
    pub execution_units: usize,
    pub warp_width: u32,
    pub cohorts: CohortPolicy,
    /// The runtime allocator's guaranteed alignment for compiler-owned global
    /// buffers. External buffer alignment belongs to ScalarWorkload.
    pub internal_alignment: u64,
    pub resources: Vec<Resource>,
    /// Timing/service of actual terminal instructions. Bundled functions must
    /// first expand from their retained implementation; they have no price entry.
    /// Lifecycle nodes describe protocol ordering, excluding submission overhead.
    pub timings: Vec<PrimitiveTiming>,
    pub block_residency: Vec<Residency>,
    /// Every per-unit resident constraint. Blocks have one identical fixed vector
    /// in this form; arbitrary per-unit service sharing is not admitted here.
    pub per_unit_residency: Vec<UnitResidency>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Allocation {
    External(u64),
    Scratch,
    BufferTable,
    Scalars,
    Statuses,
    Parameter { lane: u64, parameter: usize },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub lane: u64,
    pub allocation: Allocation,
    pub offset: u64,
    pub bytes: u32,
    pub alignment: u64,
    pub write: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub requirement: Requirement,
    pub block: Option<u64>,
    pub warp: Option<u64>,
    pub position: Option<usize>,
    pub origin: Option<ptx::Origin>,
    pub issued_lanes: Vec<u32>,
    pub active_lanes: Vec<u32>,
    pub accesses: Vec<Access>,
    pub predecessors: Vec<usize>,
    pub start_predecessors: Vec<usize>,
}
pub struct DerivedCudaModel<'a> {
    implementation: &'a Execution,
    hardware: &'a CudaHardware,
    workload: &'a ScalarWorkload,
    pub model: schedule::Model,
    pub scope: Scope,
    pub placement: Placement,
    pub cohorts: CohortPolicy,
    pub events: Vec<Event>,
    // Event i maps directly to operation i; there is no authored expansion DAG.
    pub requirements: Vec<Requirement>,
    pub instructions: u64,
}

impl<'a> DerivedCudaModel<'a> {
    pub fn implementation(&self) -> &'a Execution {
        self.implementation
    }
    pub fn hardware(&self) -> &'a CudaHardware {
        self.hardware
    }
    pub fn workload(&self) -> &'a ScalarWorkload {
        self.workload
    }
}

pub fn requirements(execution: &Execution) -> Vec<Requirement> {
    target_requirements(execution.target_plan())
}
pub(crate) fn target_requirements(target: &ptx::TargetPlan) -> Vec<Requirement> {
    let mut required = Vec::new();
    for r in target
        .requirements()
        .map(|r| match r {
            ptx::Requirement::Instruction(p) => Requirement::Instruction(p),
            ptx::Requirement::WholeBody(h) => Requirement::WholeBody(h),
        })
        .chain(
            [
                Lifecycle::Launch,
                Lifecycle::BlockAdmission,
                Lifecycle::WarpStart,
                Lifecycle::WarpCompletion,
                Lifecycle::BlockCompletion,
                Lifecycle::Completion,
            ]
            .map(Requirement::Lifecycle),
        )
    {
        if !required.contains(&r) {
            required.push(r);
        }
    }
    required
}

pub fn derive_cuda<'a>(
    execution: &'a Execution,
    hardware: &'a CudaHardware,
    workload: &'a ScalarWorkload,
    placement: &Placement,
    limits: DerivationLimits,
) -> Result<DerivedCudaModel<'a>, DerivationError> {
    let required = requirements(execution);
    validate(execution, hardware, placement, &required, limits)?;
    let traced = trace::derive(execution, hardware, workload, limits)?;
    build(
        execution, hardware, workload, placement, traced, required, limits,
    )
}

/// Account the runtime's ordered launch sequence. Each phase completes before
/// the next begins; private ABI/scratch storage is phase-local. External values
/// (including invalidation by unknown stores) carry through canonical allocation
/// identity, so later phases never restart from the invocation's initial bytes.
pub fn derive_sequence(
    executions: &[Execution],
    hardware: &CudaHardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<schedule::Model, DerivationError> {
    let first = executions
        .first()
        .ok_or("CUDA analysis needs a nonempty phase sequence")?;
    if executions.iter().any(|e| {
        e.program().buffers != first.program().buffers
            || e.program().scalars != first.program().scalars
    }) {
        return Err("CUDA phase sequence has inconsistent invocation ABI".into());
    }
    let mut result = schedule::Model {
        relationship: seismic_accounting::authority::ModelRelationship::hypothetical_execution(),
        identity: format!("{}:ordered-ptx-sequence", hardware.identity),
        timebase: hardware.timebase.clone(),
        resources: Vec::new(),
        operations: Vec::new(),
        lifetimes: Vec::new(),
        static_orders: Vec::new(),
        unmapped: Vec::new(),
    };
    // Validate binding/input budgets before copying potentially large byte maps.
    let required = requirements(first);
    validate(
        first,
        hardware,
        &Placement::HomogeneousResidentSlots,
        &required,
        limits,
    )?;
    let known_bytes = workload.allocations.iter().try_fold(0usize, |n, a| {
        n.checked_add(a.known_bytes.len())
            .ok_or("CUDA known-value input size overflow")
    })?;
    if workload.allocations.len() > limits.operations
        || workload.buffers.len() > limits.operations
        || workload.scalars.len() > limits.operations
        || known_bytes > limits.operations
    {
        return Err(DerivationError::Exhausted(DerivationLimit::Operations(
            limits.operations,
        )));
    }
    let mut state = workload.clone();
    let mut instructions = 0u64;
    let mut previous_completion = None;
    for (phase, execution) in executions.iter().enumerate() {
        let remaining = DerivationLimits {
            instructions: limits.instructions.checked_sub(instructions).ok_or(
                DerivationError::Exhausted(DerivationLimit::Instructions(limits.instructions)),
            )?,
            operations: limits
                .operations
                .checked_sub(result.operations.len())
                .ok_or(DerivationError::Exhausted(DerivationLimit::Operations(
                    limits.operations,
                )))?,
        };
        let required = requirements(execution);
        validate(
            execution,
            hardware,
            &Placement::HomogeneousResidentSlots,
            &required,
            remaining,
        )?;
        let mut traced = trace::derive(execution, hardware, &state, remaining)?;
        instructions = instructions
            .checked_add(traced.instructions)
            .ok_or("CUDA sequence instruction count overflow")?;
        let mut external_values = std::mem::take(&mut traced.external_values);
        let mut part = build(
            execution,
            hardware,
            &state,
            &Placement::HomogeneousResidentSlots,
            traced,
            required,
            remaining,
        )?
        .model;
        for allocation in &mut state.allocations {
            allocation.known_bytes = external_values
                .remove(&allocation.id)
                .ok_or("CUDA phase lost an external allocation")?;
        }
        let resource_base = result.resources.len();
        let operation_base = result.operations.len();
        if resource_base
            .checked_add(part.resources.len())
            .is_none_or(|n| n > limits.operations)
        {
            return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                limits.operations,
            )));
        }
        for resource in &mut part.resources {
            resource.name = format!("phase{phase}:{}", resource.name);
        }
        for operation in &mut part.operations {
            operation.name = format!("phase{phase}:{}", operation.name);
            for p in operation
                .predecessors
                .iter_mut()
                .chain(&mut operation.start_predecessors)
            {
                *p += operation_base;
            }
            for reservation in &mut operation.reservations {
                reservation.resource += resource_base;
            }
        }
        // The trace owns a launch root and a completion join of every block end.
        // Phase ordering therefore also closes all resident-capacity lifetimes.
        if let Some(previous) = previous_completion {
            part.operations[0].predecessors.push(previous);
        }
        previous_completion = Some(operation_base + part.operations.len() - 1);
        for lifetime in &mut part.lifetimes {
            lifetime.resource += resource_base;
            lifetime.begin.operation += operation_base;
            lifetime.end.operation += operation_base;
        }
        debug_assert!(
            part.static_orders.is_empty(),
            "PTX instruction order is fixed"
        );
        result.resources.extend(part.resources);
        result.operations.extend(part.operations);
        result.lifetimes.extend(part.lifetimes);
    }
    result.lower_bound()?;
    Ok(result)
}

fn validate(
    execution: &Execution,
    hardware: &CudaHardware,
    placement: &Placement,
    required: &[Requirement],
    limits: DerivationLimits,
) -> Result<(), DerivationError> {
    let Placement::HomogeneousResidentSlots = placement;
    validate_target(execution.target_plan(), hardware, required, limits)
}
pub(crate) fn validate_target(
    target: &ptx::TargetPlan,
    hardware: &CudaHardware,
    required: &[Requirement],
    limits: DerivationLimits,
) -> Result<(), DerivationError> {
    target.validate()?;
    if hardware.identity.is_empty()
        || hardware.execution_units == 0
        || hardware.warp_width == 0
        || !hardware.internal_alignment.is_power_of_two()
        || hardware.timebase.seconds_numerator == 0
        || hardware.timebase.seconds_denominator == 0
    {
        return Err(
            "CUDA hardware needs identified positive geometry, alignment and timebase".into(),
        );
    }
    if limits.instructions == 0 {
        return Err(DerivationError::Exhausted(DerivationLimit::Instructions(
            limits.instructions,
        )));
    }
    if limits.operations == 0
        || hardware.resources.len() > limits.operations
        || hardware.timings.len() > limits.operations
        || hardware.per_unit_residency.len() > limits.operations
    {
        return Err(DerivationError::Exhausted(DerivationLimit::Operations(
            limits.operations,
        )));
    }
    let mut names = BTreeSet::new();
    for r in &hardware.resources {
        if r.name.is_empty() || !names.insert(&r.name) || r.capacity == 0 {
            return Err("CUDA resources need unique names and positive capacity".into());
        }
    }
    for (i, timing) in hardware.timings.iter().enumerate() {
        if hardware.timings[..i]
            .iter()
            .any(|t| t.primitive == timing.primitive)
            || timing.reservations.is_empty()
        {
            return Err(
                "CUDA primitive timing must be unique and carry bounded hardware service".into(),
            );
        }
        if timing.reservations.len() > limits.operations {
            return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                limits.operations,
            )));
        }
        validate_ticks(timing.latency)?;
        for reservation in &timing.reservations {
            if reservation.resource >= hardware.resources.len() || reservation.units.scale == 0 {
                return Err("invalid CUDA primitive hardware service".into());
            }
            validate_ticks(reservation.duration)?;
            validate_quantity(reservation.units.quantity)?;
        }
    }
    for requirement in required {
        match requirement {
            Requirement::Instruction(primitive)
                if !hardware.timings.iter().any(|t| &t.primitive == primitive) =>
            {
                return Err(format!("missing CUDA hardware timing for {primitive:?}").into());
            }
            Requirement::WholeBody(helper) => {
                return Err(format!(
                    "CUDA analysis needs the retained {helper:?} implementation expanded into PTX operations; a supplied whole-body cost is not an implementation"
                ).into());
            }
            _ => {}
        }
    }
    let mut residency = BTreeSet::new();
    for r in &hardware.block_residency {
        let resource = hardware
            .resources
            .get(r.resource)
            .ok_or("invalid CUDA residency resource")?;
        if r.units.scale == 0
            || !residency.insert(r.resource)
            || matches!(resource.unit, schedule::CapacityUnit::ServicePerTick(_))
            || matches!(resource.scope, ResourceScope::Warp)
        {
            return Err("invalid or duplicated CUDA block residency mapping".into());
        }
        if !matches!(
            r.units.quantity,
            Quantity::One
                | Quantity::BlockThreads
                | Quantity::BlockWarps
                | Quantity::VirtualRegisterBits
        ) {
            return Err("block residency must use block geometry".into());
        }
    }
    if hardware.per_unit_residency.is_empty() {
        return Err("homogeneous CUDA residency needs per-unit capacity constraints".into());
    }
    let mut names = BTreeSet::new();
    for r in &hardware.per_unit_residency {
        if r.name.is_empty()
            || !names.insert(&r.name)
            || r.capacity == 0
            || r.units_per_block.scale == 0
            || !matches!(
                r.units_per_block.quantity,
                Quantity::One
                    | Quantity::BlockThreads
                    | Quantity::BlockWarps
                    | Quantity::VirtualRegisterBits
            )
        {
            return Err("invalid homogeneous CUDA resident capacity".into());
        }
    }
    Ok(())
}
fn validate_quantity(q: Quantity) -> Result<(), String> {
    if matches!(q,Quantity::MemorySectors{bytes} if !bytes.is_power_of_two()) {
        return Err("memory coverage granularity must be a power of two".into());
    }
    Ok(())
}
fn validate_ticks(t: Ticks) -> Result<(), String> {
    if let Ticks::Service {
        demand, per_tick, ..
    } = t
    {
        if per_tick == 0 || demand.scale == 0 {
            return Err("service supply and demand scale must be positive".into());
        }
        validate_quantity(demand.quantity)?;
    }
    Ok(())
}

fn quantity(
    q: Quantity,
    event: &Event,
    execution: &Execution,
    hardware: &CudaHardware,
) -> Result<u64, String> {
    Ok(match q {
        Quantity::One => 1,
        Quantity::IssuedLanes => event.issued_lanes.len() as u64,
        Quantity::ActiveLanes => event.active_lanes.len() as u64,
        Quantity::RequestedBytes => event.accesses.iter().try_fold(0u64, |n, a| {
            n.checked_add(u64::from(a.bytes))
                .ok_or("requested byte overflow")
        })?,
        Quantity::MemorySectors { bytes } => {
            let mut coverage = BTreeSet::new();
            for access in &event.accesses {
                if access.alignment < bytes {
                    return Err("memory sector geometry needs known allocation alignment".into());
                }
                let end = access
                    .offset
                    .checked_add(u64::from(access.bytes))
                    .ok_or("memory range overflow")?;
                for sector in access.offset / bytes..end.div_ceil(bytes) {
                    coverage.insert((access.allocation.clone(), sector));
                }
            }
            coverage.len() as u64
        }
        Quantity::BlockThreads => {
            if event.block.is_none() {
                return Err("block demand outside block scope".into());
            }
            execution.dispatch().threads_per_group
        }
        Quantity::BlockWarps => {
            if event.block.is_none() {
                return Err("warp demand outside block scope".into());
            }
            execution
                .dispatch()
                .threads_per_group
                .div_ceil(u64::from(hardware.warp_width))
        }
        Quantity::VirtualRegisterBits => {
            if event.block.is_none() {
                return Err("register demand outside block scope".into());
            }
            let bits = virtual_register_bits(execution.target_plan())?;
            bits.checked_mul(execution.dispatch().threads_per_group)
                .ok_or("virtual register storage overflow")?
        }
    })
}
pub(crate) fn virtual_register_bits(target: &ptx::TargetPlan) -> Result<u64, String> {
    target.registers().iter().try_fold(0u64, |n, r| {
        n.checked_add(match r.class {
            ptx::RegisterClass::Bits64 => 64,
            ptx::RegisterClass::Predicate => 1,
            _ => 32,
        })
        .ok_or_else(|| "virtual register storage overflow".into())
    })
}
fn amount(a: Amount, e: &Event, x: &Execution, c: &CudaHardware) -> Result<u64, String> {
    amount_from(a, &|q| quantity(q, e, x, c))
}
fn amount_from(
    a: Amount,
    quantity: &impl Fn(Quantity) -> Result<u64, String>,
) -> Result<u64, String> {
    quantity(a.quantity)?
        .checked_mul(a.scale)
        .ok_or_else(|| "CUDA demand overflow".into())
}
fn ticks_from(
    t: Ticks,
    quantity: &impl Fn(Quantity) -> Result<u64, String>,
) -> Result<u64, String> {
    match t {
        Ticks::Fixed(t) => Ok(t),
        Ticks::Service {
            demand,
            per_tick,
            base,
        } => amount_from(demand, quantity)?
            .div_ceil(per_tick)
            .checked_add(base)
            .ok_or_else(|| "CUDA service time overflow".into()),
    }
}

/// One primitive's hardware service, shared by concrete trace accounting and
/// necessary-demand relaxation. Only the derived quantity environment differs.
pub(crate) fn primitive_service(
    timing: &PrimitiveTiming,
    quantity: impl Fn(Quantity) -> Result<u64, String>,
) -> Result<(u64, Vec<schedule::Reservation>), String> {
    let latency = ticks_from(timing.latency, &quantity)?;
    let mut reservations = Vec::new();
    for reservation in &timing.reservations {
        let units = amount_from(reservation.units, &quantity)?;
        let duration = ticks_from(reservation.duration, &quantity)?;
        if units > 0 && duration > 0 {
            reservations.push(schedule::Reservation {
                resource: reservation.resource,
                offset: reservation.offset,
                duration,
                units,
            });
        }
    }
    Ok((latency, reservations))
}

fn resource_instance(
    scope: ResourceScope,
    event: &Event,
    _placement: &Placement,
) -> Result<u64, String> {
    match scope {
        ResourceScope::Device => Ok(0),
        ResourceScope::Block => event
            .block
            .ok_or_else(|| "block resource outside block".into()),
        ResourceScope::Warp => event
            .warp
            .ok_or_else(|| "warp resource outside warp".into()),
    }
}

fn build<'a>(
    execution: &'a Execution,
    hardware: &'a CudaHardware,
    workload: &'a ScalarWorkload,
    placement: &Placement,
    traced: trace::Trace,
    required: Vec<Requirement>,
    limits: DerivationLimits,
) -> Result<DerivedCudaModel<'a>, DerivationError> {
    let mut model = schedule::Model {
        relationship: seismic_accounting::authority::ModelRelationship::hypothetical_execution(),
        identity: format!("{}:ptx-kernel", hardware.identity),
        timebase: hardware.timebase.clone(),
        resources: Vec::new(),
        operations: Vec::new(),
        lifetimes: Vec::new(),
        static_orders: Vec::new(),
        unmapped: Vec::new(),
    };
    let mut instances = BTreeMap::new();
    let mut admissions = BTreeMap::new();
    let geometry = Event {
        requirement: Requirement::Lifecycle(Lifecycle::BlockAdmission),
        block: Some(0),
        warp: None,
        position: None,
        origin: None,
        issued_lanes: Vec::new(),
        active_lanes: Vec::new(),
        accesses: Vec::new(),
        predecessors: Vec::new(),
        start_predecessors: Vec::new(),
    };
    let mut slots = u64::MAX;
    for r in &hardware.per_unit_residency {
        let demand = amount(r.units_per_block, &geometry, execution, hardware)?;
        if demand > 0 {
            slots = slots.min(r.capacity / demand);
        }
    }
    if slots == u64::MAX || slots == 0 {
        return Err("CUDA homogeneous residency has no finite feasible block capacity".into());
    }
    let capacity = slots
        .checked_mul(hardware.execution_units as u64)
        .ok_or("CUDA resident slot capacity overflow")?;
    model.resources.push(schedule::Resource {
        name: "cuda.homogeneous-resident-slots".into(),
        capacity,
        unit: schedule::CapacityUnit::Slots,
    });
    for (event_id, event) in traced.events.iter().enumerate() {
        if model.operations.len() >= limits.operations {
            return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                limits.operations,
            )));
        }
        let timing = match event.requirement {
            Requirement::Instruction(primitive) => Some(
                hardware
                    .timings
                    .iter()
                    .find(|t| t.primitive == primitive)
                    .ok_or("missing traced PTX timing")?,
            ),
            Requirement::Lifecycle(_) => None,
            Requirement::WholeBody(_) => {
                return Err("unexpanded PTX helper reached resource derivation".into());
            }
        };
        let (latency, service) = timing
            .map(|t| primitive_service(t, |q| quantity(q, event, execution, hardware)))
            .transpose()?
            .unwrap_or_default();
        let mut reservations = Vec::new();
        for reservation in service {
            let units = reservation.units;
            let duration = reservation.duration;
            if reservation
                .offset
                .checked_add(duration)
                .is_none_or(|end| end > latency)
            {
                return Err("CUDA hardware service extends beyond primitive completion".into());
            }
            let resource = instance(
                reservation.resource,
                event,
                hardware,
                placement,
                &mut model,
                &mut instances,
                limits,
            )?;
            if units > model.resources[resource].capacity {
                return Err("CUDA primitive demand exceeds scoped hardware capacity".into());
            }
            reservations.push(schedule::Reservation {
                resource,
                offset: reservation.offset,
                duration,
                units,
            });
        }
        let begin = model.operations.len();
        let completion = begin;
        model.operations.push(schedule::Operation {
            name: format!("ptx.event{event_id}"),
            predecessors: event.predecessors.clone(),
            start_predecessors: event.start_predecessors.clone(),
            latency,
            reservations,
        });
        if event.requirement == Requirement::Lifecycle(Lifecycle::BlockAdmission) {
            admissions.insert(event.block.unwrap(), begin);
        }
        if event.requirement == Requirement::Lifecycle(Lifecycle::BlockCompletion) {
            model.lifetimes.push(schedule::Lifetime {
                resource: 0,
                units: 1,
                begin: schedule::Event {
                    operation: admissions[&event.block.unwrap()],
                    point: schedule::Point::Start,
                },
                end: schedule::Event {
                    operation: completion,
                    point: schedule::Point::Completion,
                },
            });
            for residency in &hardware.block_residency {
                let units = amount(residency.units, event, execution, hardware)?;
                if units == 0 {
                    continue;
                }
                let resource = instance(
                    residency.resource,
                    event,
                    hardware,
                    placement,
                    &mut model,
                    &mut instances,
                    limits,
                )?;
                if units > model.resources[resource].capacity {
                    return Err("one CUDA block exceeds resident capacity".into());
                }
                model.lifetimes.push(schedule::Lifetime {
                    resource,
                    units,
                    begin: schedule::Event {
                        operation: admissions[&event.block.unwrap()],
                        point: schedule::Point::Start,
                    },
                    end: schedule::Event {
                        operation: completion,
                        point: schedule::Point::Completion,
                    },
                });
            }
        }
    }
    model.lower_bound()?;
    Ok(DerivedCudaModel {
        implementation: execution,
        hardware: hardware,
        workload,
        model,
        scope: hardware.scope,
        placement: placement.clone(),
        cohorts: hardware.cohorts,
        events: traced.events,
        requirements: required,
        instructions: traced.instructions,
    })
}
fn instance(
    resource: usize,
    event: &Event,
    hardware: &CudaHardware,
    placement: &Placement,
    model: &mut schedule::Model,
    instances: &mut BTreeMap<(usize, u64), usize>,
    limits: DerivationLimits,
) -> Result<usize, DerivationError> {
    let definition = &hardware.resources[resource];
    let i = resource_instance(definition.scope, event, placement)?;
    if let Some(&r) = instances.get(&(resource, i)) {
        return Ok(r);
    }
    if model.resources.len() >= limits.operations {
        return Err(DerivationError::Exhausted(DerivationLimit::Operations(
            limits.operations,
        )));
    }
    let r = model.resources.len();
    model.resources.push(schedule::Resource {
        name: format!("{}:{:?}:{i}", definition.name, definition.scope),
        capacity: definition.capacity,
        unit: definition.unit.clone(),
    });
    instances.insert((resource, i), r);
    Ok(r)
}
