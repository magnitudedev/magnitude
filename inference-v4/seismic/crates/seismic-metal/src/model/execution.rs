//! Conditional timing of the retained MSL statement implementation. This is a
//! pooled-device, source-order machine; it is not a claim about native scheduling.
use crate::terminal::{Expression, Primitive, Statement, Type};
use seismic_accounting::{
    authority::ModelRelationship,
    schedule::{self, Model, Operation, Reservation, Resource, Timebase},
    workload::{DerivationError, DerivationLimit, DerivationLimits, ScalarWorkload},
};
use seismic_lang::{
    abi::ScalarLayout,
    ast::{BinaryOp, UnaryOp},
};
use std::collections::{BTreeMap, BTreeSet};
use super::{access::AccessPattern, memory::Memory};
use schedule::structured::{Node as StructuredNode, Order as StructuredOrder, Structured};
use std::sync::Arc;
#[cfg(test)]
mod repetition_tests;
#[cfg(test)]
mod symbolic_tests;
mod ranges;
mod grouping;
mod family;
mod parameters;
pub(crate) use grouping::{derive as grouping_traces, Traces as GroupingTraces};
mod facts;
pub(super) mod affine;

#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Units {
    PerLane(u64),
    PerSubgroup(u64),
    /// Units per distinct aligned device-memory block touched by one operation.
    /// The resource names the modeled boundary; this does not assert DRAM traffic.
    PerTransaction { bytes: u64, units: u64 },
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Service {
    pub resource: usize,
    pub offset: u64,
    pub duration: u64,
    pub units: Units,
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Timing {
    pub primitive: Primitive,
    pub latency: u64,
    pub services: Vec<Service>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Hardware {
    pub identity: String,
    pub timebase: Timebase,
    pub resources: Vec<Resource>,
    /// Explicit pooled resident capacities; no device-name occupancy guesses.
    pub resident_groups: u64,
    pub resident_shared_bytes: u64,
    pub timings: Vec<Timing>,
}
impl Timing {
    /// Expand declared service using the same equations for concrete lane facts
    /// and unresolved lane/transaction counts. Address analysis owns transaction
    /// geometry; an unavailable count must remain an explicit analysis gap.
    pub fn account<A: seismic_accounting::algebra::Algebra>(
        &self,
        algebra: &mut A,
        lanes: A::Value,
        mut transactions: impl FnMut(&mut A, u64) -> Result<A::Value, A::Error>,
    ) -> Result<(A::Value, Vec<seismic_accounting::algebra::ResourceUse<A::Value>>), A::Error> {
        let latency = algebra.constant(self.latency)?;
        let mut uses = Vec::with_capacity(self.services.len());
        for service in &self.services {
            let units = match service.units {
                Units::PerLane(scale) => {
                    let scale = algebra.constant(scale)?;
                    algebra.product(lanes, scale)?
                }
                Units::PerSubgroup(units) => algebra.constant(units)?,
                Units::PerTransaction { bytes, units } => {
                    let count = transactions(algebra, bytes)?;
                    let scale = algebra.constant(units)?;
                    algebra.product(count, scale)?
                }
            };
            uses.push(seismic_accounting::algebra::ResourceUse {
                resource: service.resource, offset: service.offset,
                duration: algebra.constant(service.duration)?, units,
            });
        }
        Ok((latency, uses))
    }
}
impl Hardware {
    /// Expand declared primitive service for concrete operations. Missing
    /// mappings or access geometry never acquire invented costs.
    fn operation(&self, primitive: &Primitive, lanes: u64, access: Option<&AccessPattern>) -> Result<Option<Operation>, String> {
        let Some(timing) = self.timings.iter().find(|t| &t.primitive == primitive) else {
            return Ok(None);
        };
        let transactions = |bytes| access.and_then(|a| a.transactions(bytes));
        if timing.services.iter().any(|service| matches!(service.units, Units::PerTransaction { bytes, .. }
            if transactions(bytes).is_none())) { return Ok(None); }
        let mut algebra = seismic_accounting::algebra::Concrete::<String>::default();
        let (latency, uses) = timing.account(&mut algebra, lanes, |_, bytes| {
            transactions(bytes).ok_or_else(|| "unresolved Metal transaction geometry".into())
        })?;
        let reservations = uses.into_iter().map(|r| Reservation {
            resource: r.resource, offset: r.offset, duration: r.duration, units: r.units,
        }).collect();
        Ok(Some(Operation {
            name: format!("{primitive:?}"),
            predecessors: Vec::new(),
            start_predecessors: Vec::new(),
            latency,
            reservations,
        }))
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.identity.is_empty()
            || self.resident_groups == 0
            || self.timebase.seconds_numerator == 0
            || self.timebase.seconds_denominator == 0
        {
            return Err(
                "Metal model needs an identity, positive resident capacity and timebase".into(),
            );
        }
        let mut names = BTreeSet::new();
        for r in &self.resources {
            if r.capacity == 0 || r.name.is_empty() || !names.insert(&r.name) {
                return Err(
                    "Metal model resources require unique names and positive capacities".into(),
                );
            }
        }
        for (i, t) in self.timings.iter().enumerate() {
            if self.timings[..i].iter().any(|p| p.primitive == t.primitive) {
                return Err("duplicate Metal primitive timing".into());
            }
            for s in &t.services {
                let units = match s.units {
                    Units::PerLane(n) | Units::PerSubgroup(n) => n,
                    Units::PerTransaction { bytes, units } => {
                        if !bytes.is_power_of_two() || !matches!(t.primitive, Primitive::Read { space: crate::terminal::Space::Device, .. } | Primitive::Write { space: crate::terminal::Space::Device, .. } | Primitive::VectorRead { space: crate::terminal::Space::Device, .. }) {
                            return Err("transaction service needs scalar device memory and a power-of-two byte granularity".into());
                        }
                        units
                    },
                };
                if s.resource >= self.resources.len()
                    || s.duration == 0
                    || units == 0
                    || s.offset
                        .checked_add(s.duration)
                        .is_none_or(|end| end > t.latency)
                {
                    return Err("invalid Metal primitive service interval".into());
                }
            }
            if t.latency > 0 && t.services.is_empty() {
                return Err("positive Metal primitive timing must declare its service".into());
            }
        }
        Ok(())
    }
}
/// Dynamic terminal operations, grouped by their actual active lane count.
/// These are emitted-operation counts, not native instruction or transaction counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationCount {
    pub primitive: Primitive,
    pub lanes: u64,
    pub instances: u64,
    /// Active device-address union, when every address and its binding are known.
    pub access: Option<AccessPattern>,
}
#[derive(Clone, Debug)]
pub struct InvocationAccount {
    pub operations: Vec<OperationCount>,
    pub unmapped: Vec<String>,
    pub exhausted: Option<DerivationLimit>,
    pub visits: u64,
}
impl InvocationAccount {
    pub fn is_complete(&self) -> bool {
        self.exhausted.is_none() && self.unmapped.is_empty()
    }
    /// Relax ordering and residency from the counted execution using the same
    /// primitive mappings as schedule construction. Incomplete counts and
    /// missing mappings are not a complete execution-demand claim.
    pub fn demand(&self, hardware: &Hardware) -> Result<Option<schedule::Demand>, String> {
        hardware.validate()?;
        if !self.is_complete() {
            return Ok(None);
        }
        let mut demand =
            schedule::Demand::new(hardware.timebase.clone(), hardware.resources.clone())?;
        for term in &self.operations {
            let Some(operation) = hardware.operation(&term.primitive, term.lanes, term.access.as_ref())? else {
                return Ok(None);
            };
            demand.include(&operation, term.instances)?;
        }
        Ok(Some(demand))
    }
}

pub fn execution(
    execution: &crate::execution::Execution,
    hardware: &Hardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<Model, DerivationError> {
    hardware.validate()?;
    let mut model = Model {
        relationship: ModelRelationship::hypothetical_execution(),
        identity: format!(
            "{}:{}:{}",
            execution.function().name,
            workload.identity,
            hardware.identity
        ),
        timebase: hardware.timebase.clone(),
        resources: hardware.resources.clone(),
        operations: Vec::new(),
        lifetimes: Vec::new(),
        static_orders: Vec::new(),
        unmapped: Vec::new(),
    };
    let groups_resource = model.resources.len();
    model.resources.push(Resource {
        name: "Metal resident threadgroups".into(),
        capacity: hardware.resident_groups,
        unit: schedule::CapacityUnit::Slots,
    });
    let shared_resource = if hardware.resident_shared_bytes > 0 {
        let index = model.resources.len();
        model.resources.push(Resource {
            name: "Metal resident shared bytes".into(),
            capacity: hardware.resident_shared_bytes,
            unit: schedule::CapacityUnit::Bytes,
        });
        Some(index)
    } else {
        None
    };
    let mut state = Derivation::new(
        Sink::Schedule {
            hardware,
            model,
            groups_resource,
            shared_resource,
        },
        limits,
    );
    state.invocation(execution, workload)?;
    let Sink::Schedule { mut model, .. } = state.sink else {
        unreachable!()
    };
    model.unmapped.sort();
    model.unmapped.dedup();
    Ok(model)
}

/// Derive launch, group and subgroup structure directly from the same terminal
/// walk as the flat oracle. No flattened scheduling graph is constructed first.
pub fn structured_execution(
    execution: &crate::execution::Execution,
    hardware: &Hardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<Structured, DerivationError> {
    structured_execution_inner(execution, hardware, workload, limits, true, true)
}
fn structured_execution_inner(
    execution: &crate::execution::Execution,
    hardware: &Hardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
    compact: bool,
    groups: bool,
) -> Result<Structured, DerivationError> {
    hardware.validate()?;
    let mut resources = hardware.resources.clone();
    let groups_resource = resources.len();
    resources.push(Resource { name: "Metal resident threadgroups".into(), capacity: hardware.resident_groups, unit: schedule::CapacityUnit::Slots });
    let shared_resource = (hardware.resident_shared_bytes > 0).then(|| {
        let index = resources.len();
        resources.push(Resource { name: "Metal resident shared bytes".into(), capacity: hardware.resident_shared_bytes, unit: schedule::CapacityUnit::Bytes });
        index
    });
    let model = Structured { relationship: ModelRelationship::hypothetical_execution(),
        identity: format!("{}:{}:{}", execution.function().name, workload.identity, hardware.identity),
        timebase: hardware.timebase.clone(), resources,
        root: Arc::new(StructuredNode::Compose { order: StructuredOrder::Serial, children: vec![] }), unmapped: vec![] };
    let mut state = Derivation::new(Sink::Structured {
        hardware, model, groups_resource, shared_resource, current: vec![], frames: vec![], operations: 0, invariant_mapping_gap: false,
    }, limits);
    state.structured_loops = compact;
    state.structured_groups = groups;
    state.invocation(execution, workload)?;
    let used_repetition = state.used_repetition;
    let used_groups = state.used_group_repetition;
    let invariant_mapping_gap = state.sink.has_invariant_mapping_gap();
    let Sink::Structured { mut model, current, frames, .. } = state.sink else { unreachable!() };
    if !frames.is_empty() { return Err("unclosed structured invocation frame".into()); }
    model.root = Arc::new(StructuredNode::Compose { order: StructuredOrder::Serial, children: current });
    model.unmapped.sort(); model.unmapped.dedup();
    if !invariant_mapping_gap && used_groups && !model.unmapped.is_empty() {
        return structured_execution_inner(execution, hardware, workload, limits, compact, false);
    }
    if !invariant_mapping_gap && used_repetition && !model.unmapped.is_empty() {
        // Abstract data may lose facts needed by later control or transaction
        // geometry. Refine that analysis concretely under the same limits and
        // execution; neither change the implementation nor drop its coverage.
        return structured_execution_inner(execution, hardware, workload, limits, false, false);
    }
    Ok(model)
}

/// Walk exactly the same selected terminal execution as schedule derivation,
/// without retaining a node for every dynamic instruction or inventing timings.
/// A limit or unresolved value leaves an explicitly incomplete diagnostic account.
pub fn invocation_account(
    execution: &crate::execution::Execution,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
) -> Result<InvocationAccount, DerivationError> {
    let account = InvocationAccount {
        operations: Vec::new(),
        unmapped: Vec::new(),
        exhausted: None,
        visits: 0,
    };
    let mut state = Derivation::new(
        Sink::Count {
            account,
            indices: Default::default(),
        },
        limits,
    );
    let exhausted = match state.invocation(execution, workload) {
        Ok(()) => None,
        Err(DerivationError::Exhausted(limit)) => Some(limit),
        Err(error) => return Err(error),
    };
    let Sink::Count { mut account, .. } = state.sink else {
        unreachable!()
    };
    account.exhausted = exhausted;
    account.visits = state.visits.min(limits.instructions);
    account.unmapped.sort();
    account.unmapped.dedup();
    Ok(account)
}

enum Sink<'a> {
    Structured {
        hardware: &'a Hardware,
        model: Structured,
        groups_resource: usize,
        shared_resource: Option<usize>,
        current: Vec<Arc<StructuredNode>>,
        frames: Vec<Vec<Arc<StructuredNode>>>,
        operations: usize,
        // A missing primitive service or resident pool cannot be recovered by
        // subdividing the invocation's coordinate domain.
        invariant_mapping_gap: bool,
    },
    Schedule {
        hardware: &'a Hardware,
        model: Model,
        groups_resource: usize,
        shared_resource: Option<usize>,
    },
    Count {
        account: InvocationAccount,
        indices: std::collections::HashMap<(Primitive, u64, Option<AccessPattern>), usize>,
    },
}
struct Checkpoint {
    current: Vec<Arc<StructuredNode>>,
    frames: Vec<Vec<Arc<StructuredNode>>>,
    operations: usize,
    unmapped: usize,
    invariant_mapping_gap: bool,
}
impl Sink<'_> {
    fn checkpoint(&self) -> Option<Checkpoint> {
        match self {
            Self::Structured { current, frames, operations, model, invariant_mapping_gap, .. } =>
                Some(Checkpoint { current: current.clone(), frames: frames.clone(), operations: *operations,
                    unmapped: model.unmapped.len(), invariant_mapping_gap: *invariant_mapping_gap }),
            Self::Count { .. } | Self::Schedule { .. } => None,
        }
    }
    fn has_refinable_gap(&self, checkpoint: &Option<Checkpoint>) -> bool {
        match (self, checkpoint) {
            (Self::Structured { model, invariant_mapping_gap: false, .. }, Some(Checkpoint { unmapped, .. })) => model.unmapped.len() > *unmapped,
            _ => false,
        }
    }
    fn has_invariant_mapping_gap(&self) -> bool {
        matches!(self, Self::Structured { invariant_mapping_gap: true, .. })
    }
    fn restore(&mut self, checkpoint: Option<Checkpoint>) -> Result<(), String> {
        match (self, checkpoint) {
            (Self::Structured { current, frames, operations, model, invariant_mapping_gap, .. },
                Some(Checkpoint { current: old_current, frames: old_frames, operations: old_operations, unmapped, invariant_mapping_gap: old_gap })) => {
                    *current = old_current; *frames = old_frames; *operations = old_operations;
                    model.unmapped.truncate(unmapped); *invariant_mapping_gap = old_gap;
                }
            _ => return Err("missing dispatch refinement checkpoint".into()),
        }
        Ok(())
    }

    fn begin_structure(&mut self) {
        if let Self::Structured { current, frames, .. } = self { frames.push(std::mem::take(current)); }
    }
    fn end_structure(&mut self, order: StructuredOrder, reservations: Vec<(usize, u64)>, parallel_member: bool) -> Result<(), String> {
        if let Self::Structured { current, frames, .. } = self {
            let body = Arc::new(StructuredNode::Compose { order, children: std::mem::take(current) });
            *current = frames.pop().expect("structured traversal owns a matching frame");
            let node = if reservations.is_empty() { body } else { Arc::new(StructuredNode::Scope { reservations, body }) };
            if parallel_member { StructuredNode::append_parallel(current, node)?; } else { current.push(node); }
        }
        Ok(())
    }
    fn end_repetition(&mut self, count: u64) {
        if let Self::Structured { current, frames, .. } = self {
            let body = Arc::new(StructuredNode::Compose { order: StructuredOrder::Serial, children: std::mem::take(current) });
            *current = frames.pop().expect("structured loop owns a matching frame");
            current.push(Arc::new(StructuredNode::Repeat { order: StructuredOrder::Serial, count, body }));
        }
    }
    fn repeat_group(&mut self, count: u64) -> Result<(), String> {
        let Self::Structured { current, .. } = self else { return Err("group repetition requires structured accounting".into()); };
        let body = current.pop().ok_or("missing abstract group body")?;
        current.push(Arc::new(StructuredNode::Repeat { order: StructuredOrder::Parallel, count, body }));
        Ok(())
    }
    fn gap(&mut self, message: String) {
        match self {
            Self::Structured { model, .. } => model.unmapped.push(message),
            Self::Schedule { model, .. } => model.unmapped.push(message),
            Self::Count { account, .. } => account.unmapped.push(message),
        }
    }
    fn group_lifetime(&mut self, begin: usize, end: usize, bytes: u64, factor_adjacent: bool) -> Result<(), String> {
        if let Self::Structured { groups_resource, shared_resource, model, invariant_mapping_gap, .. } = self {
            let mut reservations = vec![(*groups_resource, 1)];
            if bytes > 0 {
                if let Some(resource) = shared_resource { reservations.push((*resource, bytes)); }
                else {
                    model.unmapped.push("shared allocation has no pooled resident capacity".into());
                    *invariant_mapping_gap = true;
                }
            }
            return self.end_structure(StructuredOrder::Serial, reservations, factor_adjacent);
        }
        if let Self::Schedule {
            model,
            groups_resource,
            shared_resource,
            ..
        } = self
        {
            let lifetime = |resource, units| schedule::Lifetime {
                resource,
                units,
                begin: schedule::Event {
                    operation: begin,
                    point: schedule::Point::Start,
                },
                end: schedule::Event {
                    operation: end,
                    point: schedule::Point::Completion,
                },
            };
            model.lifetimes.push(lifetime(*groups_resource, 1));
            if bytes > 0 {
                if let Some(resource) = shared_resource {
                    model.lifetimes.push(lifetime(*resource, bytes));
                } else {
                    model
                        .unmapped
                        .push("shared allocation has no pooled resident capacity".into());
                }
            }
        }
        Ok(())
    }
}
fn canonical_scalars(
    emitted: &crate::msl::Emitted,
    workload: &ScalarWorkload,
) -> Result<BTreeMap<String, Values>, String> {
    let layout = ScalarLayout::words(&emitted.scalars)?;
    layout.validate_bytes(&workload.scalars)?;
    let mut result = BTreeMap::new();
    for (i, p) in emitted.scalars.iter().enumerate() {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&workload.scalars[i * 8..i * 8 + 8]);
        result.insert(
            format!("sc.{}", p.name),
            [if workload.integer_domains.iter().any(|domain| domain.input == seismic_accounting::workload::IntegerInput::Scalar { slot: i }) { None } else { Some(u64::from_le_bytes(bytes)) }; 32],
        );
    }
    Ok(result)
}
fn validate_workload(emitted: &crate::msl::Emitted, w: &ScalarWorkload) -> Result<(), String> {
    w.validate()?;
    for domain in &w.integer_domains {
        if let seismic_accounting::workload::IntegerInput::Scalar { slot } = domain.input {
            let parameter = emitted.scalars.get(slot).ok_or("integer domain scalar missing")?;
            let compatible = match parameter.dtype {
                seismic_lang::types::DType::I32 => domain.bytes == 4 && domain.signed,
                seismic_lang::types::DType::U32 => domain.bytes == 4 && !domain.signed,
                _ => false,
            };
            if !compatible || parameter.index_bound.is_some_and(|bound| domain.range.min < 0 || domain.range.max >= i128::from(bound)) {
                return Err("integer domain differs from scalar ABI or index bounds".into());
            }
        }
    }
    if w.identity.is_empty() || w.buffers.len() != emitted.buffers.len() {
        return Err("Metal workload identity or buffer count differs from target ABI".into());
    }
    let mut allocations = BTreeMap::new();
    for a in &w.allocations {
        if a.alignment == 0
            || !a.alignment.is_power_of_two()
            || allocations.insert(a.id, a).is_some()
            || a.known_bytes.keys().any(|offset| *offset >= a.bytes)
        {
            return Err("invalid Metal workload allocation".into());
        }
    }
    for (b, spec) in w.buffers.iter().zip(&emitted.buffers) {
        let a = allocations
            .get(&b.allocation)
            .ok_or("unknown Metal allocation")?;
        if b.bytes < spec.bytes as u64
            || b.offset
                .checked_add(b.bytes)
                .is_none_or(|end| end > a.bytes)
            || a.alignment < spec.alignment as u64
            || b.offset % (spec.alignment as u64) != 0
        {
            return Err("Metal workload buffer violates target ABI".into());
        }
    }
    for &(a, b, exact) in &emitted.alias_pairs {
        let (left_bytes, right_bytes) = (
            emitted.buffers[a].bytes as u64,
            emitted.buffers[b].bytes as u64,
        );
        let (a, b) = (&w.buffers[a], &w.buffers[b]);
        if a.allocation == b.allocation
            && left_bytes != 0
            && right_bytes != 0
            && a.offset < b.offset + right_bytes
            && b.offset < a.offset + left_bytes
            && !(exact && a.offset == b.offset && left_bytes == right_bytes)
        {
            return Err("Metal workload alias violates source or partition admission".into());
        }
    }
    Ok(())
}
pub(super) type Values = [Option<u64>; 32];
#[derive(Clone)]
struct Facts { ranges: ranges::Ranges, affine: affine::Values }
impl Default for Facts {
    fn default() -> Self { Self { ranges: [None; 32], affine: std::array::from_fn(|_| None) } }
}

struct Derivation<'a> {
    sink: Sink<'a>,
    limits: DerivationLimits,
    visits: u64,
    structured_loops: bool,
    structured_groups: bool,
    used_group_repetition: bool,
    used_repetition: bool,
    last: Option<usize>,
    scope: String,
    env: BTreeMap<String, Values>,
    ranges: BTreeMap<String, ranges::Ranges>,
    affine: BTreeMap<String, affine::Values>,
    next_coordinate: u64,
    expression_depth: usize,
    facts: std::collections::HashMap<Expression, Facts>,
    /// Guarded facts of the current retained statement. These are premises,
    /// separate from the disposable cache of evaluated expression facts.
    assumptions: std::collections::HashMap<Expression, Facts>,
    returned_facts: Facts,
    memory: Memory,
    active: u32,
    alive: u32,
    returned: Values,
    helpers: std::collections::HashMap<(crate::support::Helper, Type), std::sync::Arc<HelperBody>>,
}
struct HelperBody {
    parameters: Vec<&'static str>,
    statements: Vec<crate::terminal::Site>,
}
impl<'a> Derivation<'a> {
    fn new(sink: Sink<'a>, limits: DerivationLimits) -> Self {
        Self {
            sink,
            limits,
            visits: 0,
            structured_loops: false,
            structured_groups: false,
            used_group_repetition: false,
            used_repetition: false,
            last: None,
            scope: String::new(),
            env: BTreeMap::new(),
            ranges: BTreeMap::new(),
            affine: BTreeMap::new(),
            next_coordinate: 1,
            expression_depth: 0,
            facts: Default::default(),
            assumptions: Default::default(),
            returned_facts: Default::default(),
            memory: Memory::default(),
            active: u32::MAX,
            alive: u32::MAX,
            returned: [None; 32],
            helpers: Default::default(),
        }
    }
    fn invocation(
        &mut self,
        execution: &crate::execution::Execution,
        workload: &ScalarWorkload,
    ) -> Result<(), DerivationError> {
        let emitted = crate::msl::prepare_execution(execution)?;
        validate_workload(&emitted, workload)?;
        if emitted.terminal.launches().len() != emitted.launches.len() {
            self.sink
                .gap("launch has no retained terminal implementation".into());
            return Ok(());
        }
        let scalar_values = canonical_scalars(&emitted, workload)?;
        self.memory.initialize(&emitted, workload)?;
        self.next_coordinate = workload.integer_domains.len() as u64 + 1;
        let mut predecessor = None;
        for (launch, (metadata, body)) in emitted
            .launches
            .iter()
            .zip(emitted.terminal.launches())
            .enumerate()
        {
            self.scope = format!("launch {launch}");
            self.last = predecessor;
            let start = self.issue(Primitive::Launch, 1)?;
            let dispatch = metadata
                .dispatch
                .as_ref()
                .ok_or("Metal target launch has no dispatch")?;
            self.memory.launch(launch, dispatch.work_items == 1);
            let mut groups = Vec::new();
            self.sink.begin_structure(); // parallel groups after launch submission
            // Unknown group coordinates cover the entire dispatch, not one
            // sampled group. Admission requires a fully mapped invariant walk;
            // unknown predicates/transactions trigger concrete refinement below.
            let full_groups = dispatch.work_items / dispatch.items_per_group;
            let mut pending_groups = if self.structured_groups && full_groups > 0 {
                let mut regions = Vec::new();
                if full_groups < dispatch.groups { regions.push((full_groups, dispatch.groups)); }
                regions.push((0, full_groups));
                regions
            } else { Vec::new() };
            let mut concrete_group = 0;
            loop {
                let region = if self.structured_groups && full_groups > 0 { pending_groups.pop() }
                    else if concrete_group < dispatch.groups { let group = concrete_group; concrete_group += 1; Some((group, group + 1)) }
                    else { None };
                let Some((group, end_group)) = region else { break; };
                let repeat_groups = end_group - group > 1;
                let checkpoint = if repeat_groups { self.sink.checkpoint() } else { None };
                self.used_group_repetition |= repeat_groups;
                self.sink.begin_structure(); // scoped group admission and body
                self.scope = format!("launch {launch} group {group}");
                self.last = Some(start);
                let begin = self.issue(Primitive::Group, 1)?;
                let mut subgroups = Vec::new();
                self.sink.begin_structure(); // parallel subgroup bodies
                // A singleton phase may publish retained controls. Interpret its
                // one active subgroup concretely, so speculative refinement never
                // changes the values seen by a later launch. Padding is bounded
                // by the device's maximum subgroups per threadgroup.
                let compact_subgroups = self.structured_groups && dispatch.work_items != 1;
                let mut pending_subgroups = if compact_subgroups { vec![(0, dispatch.items_per_group)] } else { Vec::new() };
                let mut concrete_subgroup = 0;
                loop {
                    let region = if compact_subgroups { pending_subgroups.pop() }
                        else if concrete_subgroup < dispatch.items_per_group { let subgroup = concrete_subgroup; concrete_subgroup += 1; Some((subgroup, subgroup + 1)) }
                        else { None };
                    let Some((subgroup, end_subgroup)) = region else { break; };
                    let repeat_subgroups = end_subgroup - subgroup > 1;
                    let subgroup_checkpoint = if repeat_subgroups { self.sink.checkpoint() } else { None };
                    self.used_group_repetition |= repeat_subgroups;
                    self.sink.begin_structure(); // source order within subgroup
                    self.scope = format!("launch {launch} group {group} subgroup {subgroup}");
                    self.last = Some(begin);
                    self.env = scalar_values.clone();
                    self.ranges.clear();
                    self.affine.clear();
                    for (index, domain) in workload.integer_domains.iter().enumerate() {
                        if let seismic_accounting::workload::IntegerInput::Scalar { slot } = domain.input {
                            let name = format!("sc.{}", emitted.scalars[slot].name);
                            let value = affine::Value::domain(index as u64 + 1, domain)?;
                            self.affine.insert(name.clone(), std::array::from_fn(|_| Some(value.clone())));
                            self.ranges.insert(name, [Some((domain.range.min, domain.range.max)); 32]);
                        }
                    }
                    self.memory.subgroup();
                    for slot in &execution.memory().launches()[launch].slots {
                        if slot.placement == seismic_realization::dispatch::TilePlacement::GroupShared {
                            let layout = slot.layout(dispatch)?;
                            self.memory.array(&slot.symbol, u64::from(slot.dtype.bytes()), layout.shared_elements_per_item.checked_mul(dispatch.items_per_group).ok_or("Metal shared array element overflow")?, crate::terminal::Space::Threadgroup)?;
                        }
                    }
                    self.active = u32::MAX;
                    self.alive = u32::MAX;
                    self.returned = [None; 32];
                    self.env
                        .insert("lane".into(), std::array::from_fn(|i| Some(i as u64)));
                    for b in emitted.buffers.iter().chain(&emitted.scratch_bindings) {
                        let name = if b.plane.is_empty() {
                            b.parameter.clone()
                        } else {
                            format!("{}_{}", b.parameter, b.plane)
                        };
                        self.env.insert(name, [None; 32]);
                    }
                    self.env.insert("tg_pos.x".into(), [if repeat_groups { None } else { Some(group) }; 32]);
                    if repeat_groups { self.affine.insert("tg_pos.x".into(), std::array::from_fn(|_| Some(affine::Value::coordinate(0, i128::from(group), i128::from(end_group - 1))))); self.ranges.insert("tg_pos.x".into(), [Some((i128::from(group), i128::from(end_group - 1))); 32]); }
                    self.env.insert("sg_id".into(), [if repeat_subgroups { None } else { Some(subgroup) }; 32]);
                    if repeat_subgroups {
                        let coordinate = self.next_coordinate;
                        self.next_coordinate = coordinate.checked_add(1).ok_or("subgroup coordinate overflow")?;
                        self.affine.insert("sg_id".into(), std::array::from_fn(|_| Some(affine::Value::coordinate(coordinate, i128::from(subgroup), i128::from(end_subgroup - 1)))));
                        self.ranges.insert("sg_id".into(), [Some((i128::from(subgroup), i128::from(end_subgroup - 1))); 32]);
                    }
                    if let Err(gap) = self.block(body, 0, body.len())? {
                        self.memory.unfinished_publication();
                        self.sink.gap(format!("{}: {gap}", self.scope));
                    }
                    self.sink.end_structure(StructuredOrder::Serial, vec![], !repeat_subgroups)?;
                    if repeat_subgroups && self.sink.has_refinable_gap(&subgroup_checkpoint) {
                        self.sink.restore(subgroup_checkpoint)?;
                        let middle = subgroup + (end_subgroup - subgroup) / 2;
                        pending_subgroups.push((middle, end_subgroup));
                        pending_subgroups.push((subgroup, middle));
                        continue;
                    }
                    if repeat_subgroups && matches!(self.sink, Sink::Structured { .. }) {
                        self.sink.repeat_group(end_subgroup - subgroup)?;
                    }
                    if matches!(self.sink, Sink::Schedule { .. }) {
                        subgroups.push(self.last.unwrap_or(begin));
                    }
                }
                self.sink.end_structure(StructuredOrder::Parallel, vec![], false)?;
                let end = self.join(subgroups)?;
                self.sink
                    .group_lifetime(begin, end, metadata.declared_threadgroup_bytes, !repeat_groups)?;
                if matches!(self.sink, Sink::Schedule { .. }) {
                    groups.push(end);
                }
                if repeat_groups && self.sink.has_refinable_gap(&checkpoint) {
                    self.sink.restore(checkpoint)?;
                    let middle = group + (end_group - group) / 2;
                    pending_groups.push((middle, end_group));
                    pending_groups.push((group, middle));
                    continue;
                }
                if repeat_groups && matches!(self.sink, Sink::Structured { .. }) {
                    self.sink.repeat_group(end_group - group)?;
                }
            }
            self.sink.end_structure(StructuredOrder::Parallel, vec![], false)?;
            predecessor = Some(self.join(groups)?);
        }
        Ok(())
    }
    fn issue(&mut self, primitive: Primitive, lanes: u64) -> Result<usize, DerivationError> {
        self.issue_access(primitive, lanes, None)
    }
    fn issue_access(&mut self, primitive: Primitive, lanes: u64, access: Option<AccessPattern>) -> Result<usize, DerivationError> {
        self.visits = self
            .visits
            .checked_add(1)
            .ok_or("Metal visit count overflow")?;
        if self.visits > self.limits.instructions {
            return Err(DerivationError::Exhausted(DerivationLimit::Instructions(
                self.limits.instructions,
            )));
        }
        let i = match &mut self.sink {
            Sink::Structured { hardware, model, current, operations, invariant_mapping_gap, .. } => {
                if *operations >= self.limits.operations { return Err(DerivationError::Exhausted(DerivationLimit::Operations(self.limits.operations))); }
                let operation = if let Some(operation) = hardware.operation(&primitive, lanes, access.as_ref())? { operation }
                else {
                    // An absent service is invariant under every dispatch and
                    // loop refinement. Unknown access geometry may still become
                    // exact on smaller coordinate regions.
                    *invariant_mapping_gap |= !hardware.timings.iter().any(|timing| timing.primitive == primitive);
                    model.unmapped.push(format!("Metal primitive or access geometry {primitive:?}"));
                    Operation { name: format!("{primitive:?}"), predecessors: vec![], start_predecessors: vec![], latency: 0, reservations: vec![] }
                };
                current.push(Arc::new(StructuredNode::Operation(operation)));
                *operations += 1;
                0
            }
            Sink::Count { account, indices } => {
                let key = (primitive, lanes, access);
                let index = if let Some(index) = indices.get(&key) {
                    *index
                } else {
                    if account.operations.len() >= self.limits.operations {
                        return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                            self.limits.operations,
                        )));
                    }
                    let index = account.operations.len();
                    account.operations.push(OperationCount {
                        primitive: key.0.clone(),
                        lanes,
                        instances: 0,
                        access: key.2.clone(),
                    });
                    indices.insert(key, index);
                    index
                };
                let count = &mut account.operations[index].instances;
                *count = count
                    .checked_add(1)
                    .ok_or("Metal operation count overflow")?;
                0
            }
            Sink::Schedule {
                hardware, model, ..
            } => {
                if model.operations.len() >= self.limits.operations {
                    return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                        self.limits.operations,
                    )));
                }
                let (latency, reservations) =
                    if let Some(operation) = hardware.operation(&primitive, lanes, access.as_ref())? {
                        (operation.latency, operation.reservations)
                    } else {
                        model
                            .unmapped
                            .push(format!("Metal primitive or access geometry {primitive:?}"));
                        (0, Vec::new())
                    };
                let i = model.operations.len();
                model.operations.push(Operation {
                    name: format!("{} instruction {i}: {primitive:?}", self.scope),
                    predecessors: self.last.into_iter().collect(),
                    start_predecessors: Vec::new(),
                    latency,
                    reservations,
                });
                i
            }
        };
        self.last = Some(i);
        Ok(i)
    }
    fn join(&mut self, predecessors: Vec<usize>) -> Result<usize, DerivationError> {
        let i = match &mut self.sink {
            Sink::Count { .. } | Sink::Structured { .. } => 0,
            Sink::Schedule { model, .. } => {
                if model.operations.len() >= self.limits.operations {
                    return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                        self.limits.operations,
                    )));
                }
                let i = model.operations.len();
                model.operations.push(Operation {
                    name: format!("completion {i}"),
                    predecessors,
                    start_predecessors: Vec::new(),
                    latency: 0,
                    reservations: Vec::new(),
                });
                i
            }
        };
        self.last = Some(i);
        Ok(i)
    }
    fn expr(&mut self, e: &Expression) -> Result<Result<Values, String>, DerivationError> {
        if self.expression_depth == 0 { self.facts.clear(); }
        self.expression_depth += 1;
        let result = self.expr_inner(e);
        self.expression_depth -= 1;
        result
    }
    fn expr_inner(&mut self, e: &Expression) -> Result<Result<Values, String>, DerivationError> {
        macro_rules! value {
            ($e:expr) => {
                match self.expr($e)? {
                    Ok(v) => v,
                    Err(g) => return Ok(Err(g)),
                }
            };
        }
        use Expression as E;
        let lanes = u64::from(self.active.count_ones());
        if lanes == 0 {
            return Ok(Ok([None; 32]));
        }
        let result = match e {
            E::Integer(n, _) => [Some(*n as u64); 32],
            E::Float(bits, t) => {
                [if *t == Type::F32 {
                    Some((f64::from_bits(*bits) as f32).to_bits() as u64)
                } else {
                    None
                }; 32]
            }
            E::Parameter { name, ty } => {
                self.issue(
                    Primitive::Read {
                        space: crate::terminal::Space::Constant,
                        ty: *ty,
                    },
                    lanes,
                )?;
                self.env
                    .get(name)
                    .copied()
                    .ok_or("unbound Metal scalar ABI field")?
            }
            E::VectorElement { name, component, .. } => self.env.get(&format!("{name}[{component}]")).copied().ok_or("unbound target vector component")?,
            E::Variable(name, _) => match self.env.get(name) {
                Some(v) => *v,
                None => return Ok(Err(format!("unresolved target value {name}"))),
            },
            E::Unmapped(_, _) => return Ok(Err("unmapped target expression".into())),
            E::Binary(op, a, b, _) => {
                let (a, b) = (value!(a), value!(b));
                self.issue(
                    Primitive::Binary {
                        operation: *op,
                        ty: a_type(e),
                    },
                    lanes,
                )?;
                std::array::from_fn(|i| {
                    a[i].zip(b[i])
                        .and_then(|(a, b)| integer_binary(*op, a, b, a_type(e)))
                })
            }
            E::Unary(op, a, ty) => {
                let a = value!(a);
                self.issue(
                    Primitive::Unary {
                        operation: *op,
                        ty: *ty,
                    },
                    lanes,
                )?;
                a.map(|a| {
                    a.and_then(|a| {
                        if matches!(ty, Type::F16 | Type::BF16 | Type::F32) {
                            None
                        } else {
                            match op {
                                UnaryOp::Not => Some(u64::from(a == 0)),
                                UnaryOp::BitNot => Some(!a),
                                UnaryOp::Neg => {
                                    if matches!(ty, Type::I32) {
                                        (a as i32).checked_neg().map(|n| n as u32 as u64)
                                    } else if matches!(ty, Type::I64) {
                                        (a as i64).checked_neg().map(|n| n as u64)
                                    } else {
                                        Some(a.wrapping_neg())
                                    }
                                }
                            }
                        }
                    })
                })
            }
            E::Cast(ty, a) => {
                let from = a.ty();
                let a = value!(a);
                self.issue(Primitive::Cast { from, to: *ty }, lanes)?;
                a.map(|v| v.and_then(|v| integer_cast(v, from, *ty)))
            }
            E::Bitcast(ty, a) => {
                let from = a.ty();
                let a = value!(a);
                self.issue(Primitive::Bitcast { from, to: *ty }, lanes)?;
                a
            }
            E::ShortCircuit { or, left, right } => {
                let left = value!(left);
                let outer = self.active;
                let yes = match self.predicate(left) {
                    Ok(v) => v,
                    Err(g) => return Ok(Err(g)),
                };
                self.issue(Primitive::Branch, lanes)?;
                self.active = if *or { outer & !yes } else { yes };
                let right = value!(right);
                self.active = outer;
                std::array::from_fn(|i| {
                    if (*or && yes & (1 << i) != 0) || (!*or && yes & (1 << i) == 0) {
                        Some(u64::from(*or))
                    } else {
                        right[i].map(|v| u64::from(v != 0))
                    }
                })
            }
            E::EagerSelect(c, a, b) => {
                let condition = value!(c);
                let yes = value!(a);
                let yes_facts = self.expression_facts(a);
                let no = value!(b);
                let no_facts = self.expression_facts(b);
                self.issue(Primitive::Select, lanes)?;
                self.facts.insert(e.clone(), facts::eager_selection(condition, &yes_facts, &no_facts, self.active));
                std::array::from_fn(|lane| match condition[lane] {
                    Some(0) => no[lane],
                    Some(_) => yes[lane],
                    None if yes[lane] == no[lane] => yes[lane],
                    None => None,
                })
            }
            E::Select(c, a, b) => {
                let c = value!(c);
                let outer = self.active;
                let mut yes = 0;
                let mut no = 0;
                for (i, c) in c.iter().enumerate() {
                    if outer & (1 << i) != 0 {
                        match c {
                            Some(0) => no |= 1 << i,
                            Some(_) => yes |= 1 << i,
                            None => return Ok(Err("unknown target select predicate".into())),
                        }
                    }
                }
                self.active = yes;
                let a_values = value!(a);
                let a_facts = self.expression_facts(a);
                self.active = no;
                let b_values = value!(b);
                let b_facts = self.expression_facts(b);
                self.active = outer;
                self.issue(Primitive::Select, lanes)?;
                self.facts.insert(e.clone(), Facts {
                    ranges: std::array::from_fn(|i| if yes & (1 << i) != 0 { a_facts.ranges[i] } else { b_facts.ranges[i] }),
                    affine: std::array::from_fn(|i| if yes & (1 << i) != 0 { a_facts.affine[i].clone() } else { b_facts.affine[i].clone() }),
                });
                std::array::from_fn(|i| if yes & (1 << i) != 0 { a_values[i] } else { b_values[i] })
            }
            E::Builtin(name, args, ty) => {
                let mut values = Vec::with_capacity(args.len());
                let mut facts = Vec::with_capacity(args.len());
                for a in args {
                    values.push(value!(a));
                    facts.push(self.expression_facts(a));
                }
                self.issue(
                    Primitive::Builtin {
                        name: name.clone(),
                        inputs: args.iter().map(E::ty).collect(),
                        result: *ty,
                    },
                    lanes,
                )?;
                self.facts.insert(e.clone(), facts::builtin(name, args, &facts, *ty, self.active));
                builtin_values(name, &values, *ty, self.active)
            }
            E::Helper(helper, args, ty) => {
                let element = if *helper == crate::support::Helper::Write {
                    args.get(3).map(E::ty).unwrap_or(*ty)
                } else {
                    *ty
                };
                let arguments = crate::terminal::helper::arguments(*helper, args, element);
                // A helper definition is immutable for its operation/element
                // type. Retain its typed body across dynamic invocations while
                // deriving every invocation's values, branches and operations.
                let body = if let Some(body) = self.helpers.get(&(*helper, element)) {
                    body.clone()
                } else {
                    let definition = crate::support::Definition::new(*helper);
                    let body = std::sync::Arc::new(HelperBody {
                        parameters: definition.parameters.iter().map(|(name, _)| *name).collect(),
                        statements: crate::terminal::helper::body(&definition, element)?,
                    });
                    self.helpers.insert((*helper, element), body.clone());
                    body
                };
                let values = arguments
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut bindings = BTreeMap::new();
                for (name, value) in body.parameters.iter().zip(values) {
                    match value {
                        Ok(v) => { bindings.insert((*name).to_string(), v); }
                        Err(g) => return Ok(Err(g)),
                    }
                }
                let pointers = body.parameters.iter().zip(args).filter_map(|(name, arg)| {
                    if let E::Variable(source, _) = arg { self.memory.pointers.get(source).cloned().map(|p| ((*name).to_string(), p)) } else { None }
                }).collect();
                let saved_pointers = std::mem::replace(&mut self.memory.pointers, pointers);
                let argument_ranges = body.parameters.iter().zip(&arguments).map(|(name, arg)| ((*name).to_string(), self.expression_ranges(arg))).collect();
                let argument_affine = body.parameters.iter().zip(&arguments).map(|(name, arg)| ((*name).to_string(), self.expression_affine(arg))).collect();
                let saved = std::mem::replace(&mut self.env, bindings);
                let saved_ranges = std::mem::replace(&mut self.ranges, argument_ranges);
                let saved_affine = std::mem::replace(&mut self.affine, argument_affine);
                let saved_facts = std::mem::take(&mut self.facts);
                let saved_assumptions = std::mem::take(&mut self.assumptions);
                let saved_returned_facts = std::mem::take(&mut self.returned_facts);
                let outer = (self.active, self.alive, self.returned);
                self.alive = self.active;
                self.returned = [None; 32];
                let result = self.block(&body.statements, 0, body.statements.len())?;
                let result = result.map_err(|reason| {
                    let lane = outer.0.trailing_zeros() as usize;
                    let bounds = body.parameters.iter().map(|name| ((*name).to_string(),
                        self.ranges.get(*name).and_then(|values| values.get(lane).copied()).flatten())).collect::<Vec<_>>();
                    format!("{helper:?} helper with argument bounds {bounds:?}: {reason}")
                });
                let values = self.returned;
                let returned_facts = std::mem::replace(&mut self.returned_facts, saved_returned_facts);
                self.facts = saved_facts;
                self.assumptions = saved_assumptions;
                self.facts.insert(e.clone(), returned_facts);
                self.env = saved;
                self.ranges = saved_ranges;
                self.affine = saved_affine;
                self.memory.pointers = saved_pointers;
                (self.active, self.alive, self.returned) = outer;
                return Ok(result.map(|_| values));
            }
            E::Read {
                name, index, space, ty,
            } => {
                let indices = value!(index);
                let access = if *space == crate::terminal::Space::Device { self.access(name, index, indices, ty.bytes(), ty.bytes()) } else { None };
                self.issue_access(
                    Primitive::Read {
                        space: *space,
                        ty: *ty,
                    },
                    lanes, access,
                )?;
                let affine_indices = self.expression_affine(index);
                let symbolic = self.memory.read_symbolic(name, indices, affine_indices, index.ty(), *ty, self.active,
                    &mut self.next_coordinate, self.limits.operations)?;
                self.facts.insert(e.clone(), Facts {
                    ranges: std::array::from_fn(|lane| symbolic[lane].as_ref().and_then(|v| v.bounds())),
                    affine: symbolic,
                });
                self.memory.read(name, indices, index.ty(), ty.bytes(), self.active)
            }
        };
        if matches!(e.ty(), Type::F16 | Type::BF16 | Type::F32) { return Ok(Ok(result)); }
        let mut facts = Facts { ranges: self.expression_ranges(e), affine: self.expression_affine(e) };
        for (lane, bits) in result.iter().enumerate() {
            if let Some(bits) = bits {
                if let Some(range) = ranges::exact(*bits, e.ty()) {
                    facts.ranges[lane] = Some(range);
                    facts.affine[lane] = Some(affine::Value::constant(range.0));
                }
            }
        }
        let intervals = facts.ranges;
        self.facts.insert(e.clone(), facts);
        Ok(Ok(std::array::from_fn(|i| result[i].or_else(|| intervals[i]
            .filter(|(lo, hi)| lo == hi).map(|(value, _)| value as u64)).map(|v| mask(v, e.ty())))))
    }
    fn assign(&mut self, name: &str, values: Values) {
        self.facts.clear();
        self.assumptions.clear();
        self.ranges.remove(name);
        self.affine.remove(name);
        let old = self.env.entry(name.into()).or_insert([None; 32]);
        for i in 0..32 {
            if self.active & (1 << i) != 0 {
                old[i] = values[i];
            }
        }
    }
    fn predicate(&self, values: Values) -> Result<u32, String> {
        let mut yes = 0;
        for (i, v) in values.iter().enumerate() {
            if self.active & (1 << i) != 0 {
                match v {
                    Some(0) => {}
                    Some(_) => yes |= 1 << i,
                    None => return Err("unknown target control predicate".into()),
                }
            }
        }
        Ok(yes)
    }
    fn block(
        &mut self,
        body: &[crate::terminal::Site],
        start: usize,
        end: usize,
    ) -> Result<Result<(), String>, DerivationError> {
        let mut at = start;
        while at < end && self.active != 0 {
            self.facts.clear();
            macro_rules! value {
                ($e:expr) => {
                    match self.expr($e)? {
                        Ok(v) => v,
                        Err(g) => return Ok(Err(g)),
                    }
                };
            }
            let lanes = u64::from(self.active.count_ones());
            match &body[at].statement {
                Statement::Let { name, value, .. } | Statement::Assign { name, value } => {
                    let v = value!(value);
                    let intervals = self.expression_ranges(value);
                    let affine = self.expression_affine(value);
                    self.assign(name, v);
                    self.affine.insert(name.clone(), std::array::from_fn(|i| if self.active & (1 << i) != 0 { affine[i].clone() } else { None }));
                    self.ranges.insert(name.clone(), std::array::from_fn(|i| if self.active & (1 << i) != 0 { intervals[i] } else { None }));
                }
                Statement::Evaluate(e) => {
                    value!(e);
                }
                Statement::ReturnIf(e) => {
                    let values = value!(e);
                    let yes = match self.predicate(values) {
                        Ok(v) => v,
                        Err(g) => return Ok(Err(g)),
                    };
                    self.issue(Primitive::Branch, lanes)?;
                    if yes != 0 {
                        self.issue(Primitive::Return, u64::from(yes.count_ones()))?;
                    }
                    self.alive &= !yes;
                    self.active &= !yes;
                }
                Statement::Return(e) => {
                    let values = if let Some(e) = e {
                        value!(e)
                    } else {
                        [None; 32]
                    };
                    let facts = e.as_ref().map(|e| Facts { ranges: self.expression_ranges(e), affine: self.expression_affine(e) }).unwrap_or_default();
                    self.issue(Primitive::Return, lanes)?;
                    for (i, v) in values.iter().enumerate() {
                        if self.active & (1 << i) != 0 {
                            self.returned[i] = *v;
                            self.returned_facts.ranges[i] = facts.ranges[i];
                            self.returned_facts.affine[i] = facts.affine[i].clone();
                        }
                    }
                    self.alive &= !self.active;
                    self.active = 0;
                }
                Statement::FailureStatus => {
                    return Ok(Err("workload can execute target validity failure".into()));
                }
                Statement::Scope => {
                    let (close, alternative) = matching_end(body, at, end)?;
                    if alternative.is_some() { return Err("else on target scope".into()); }
                    let previous_ranges = self.ranges.clone();
                    let previous_affine = self.affine.clone();
                    let mut declarations = Vec::new();
                    let mut position = at + 1;
                    while position < close {
                        match &body[position].statement {
                            Statement::Let { name, .. } | Statement::Array { name, .. }
                            | Statement::Pointer { name, .. } | Statement::Fragment { name, .. } => declarations.push((name.clone(), self.env.get(name).copied(), self.memory.pointers.get(name).cloned())),
                            Statement::VectorRead { name, components, .. } => {
                                for component in 0..*components {
                                    let name = format!("{name}[{component}]");
                                    declarations.push((name.clone(), self.env.get(&name).copied(), None));
                                }
                            }
                            Statement::For { .. } | Statement::If(_) | Statement::Scope => {
                                position = matching_end(body, position, close)?.0;
                            }
                            _ => {}
                        }
                        position += 1;
                    }
                    if let Err(gap) = self.block(body, at + 1, close)? { return Ok(Err(gap)); }
                    for (name, previous, pointer) in declarations {
                        if let Some(value) = previous_affine.get(&name) { self.affine.insert(name.clone(), value.clone()); } else { self.affine.remove(&name); }
                        if let Some(range) = previous_ranges.get(&name) { self.ranges.insert(name.clone(), *range); }
                        else { self.ranges.remove(&name); }
                        if let Some(pointer) = pointer { self.memory.pointers.insert(name.clone(), pointer); }
                        else { self.memory.pointers.remove(&name); }
                        if let Some(value) = previous { self.env.insert(name, value); }
                        else { self.env.remove(&name); }
                    }
                    at = close;
                }
                Statement::For {
                    name,
                    start,
                    end: upper,
                    step,
                } => {
                    let (close, alternative) = matching_end(body, at, end)?;
                    if alternative.is_some() {
                        return Err("else on target loop".into());
                    }
                    if *step <= 0 {
                        return Err("nonpositive target loop step".into());
                    }
                    let outer = self.active;
                    let mut n = value!(start);
                    self.assign(name, n);
                    if self.structured_loops {
                        let bounds = self.expression_ranges(upper);
                        let limit = bounds.iter().enumerate().filter(|(lane, _)| self.active & (1 << lane) != 0)
                            .try_fold(None, |previous, (_, range)| {
                                let (lo, hi) = (*range)?;
                                if lo != hi || previous.is_some_and(|v| v != lo) { None } else { Some(Some(lo)) }
                            }).flatten().and_then(|v| i64::try_from(v).ok());
                        if let Some(limit) = limit {
                            if !reads_name(upper, name) && invariant_loop_bound(upper, &body[at + 1..close]) {
                            if let Some(count) = repeated_iterations(name, &body[at + 1..close], n, self.active, limit, *step).filter(|&count| count > 1) {
                                // All iterations have the same active participants
                                // and operation structure. Abstract every assigned
                                // value before the one retained visit, so no initial
                                // accumulator can masquerade as a final value.
                                self.memory.forget_values();
                                for site in &body[at + 1..close] {
                                    if let Statement::Let { name, .. } | Statement::Assign { name, .. } = &site.statement {
                                        self.assign(name, [None; 32]);
                                    }
                                    if let Statement::Pointer { name, .. } = &site.statement { self.memory.pointers.remove(name); }
                                    if let Statement::VectorRead { name, components, .. } = &site.statement {
                                        for component in 0..*components { self.assign(&format!("{name}[{component}]"), [None; 32]); }
                                    }
                                }
                                self.assign(name, [None; 32]);
                                let coordinate = self.next_coordinate;
                                self.next_coordinate = coordinate.checked_add(1).ok_or("symbolic coordinate overflow")?;
                                let induction: affine::Values = std::array::from_fn(|lane| {
                                    let start = i128::from(n[lane]? as i32);
                                    affine::Value::coordinate(coordinate, 0, i128::from(count - 1))
                                        .scale(i128::from(*step))?.add(affine::Value::constant(start))
                                });
                                self.ranges.insert(name.clone(), std::array::from_fn(|lane| induction[lane].as_ref().and_then(|v| v.bounds())));
                                self.affine.insert(name.clone(), induction);
                                self.used_repetition = true;
                                self.sink.begin_structure();
                                value!(upper);
                                self.issue(Primitive::Binary { operation: BinaryOp::Lt, ty: Type::I32 }, lanes)?;
                                self.issue(Primitive::Branch, lanes)?;
                                if let Err(gap) = self.block(body, at + 1, close)? {
                                    self.sink.end_repetition(count);
                                    return Ok(Err(gap));
                                }
                                self.issue(Primitive::Binary { operation: BinaryOp::Add, ty: Type::I32 }, lanes)?;
                                self.sink.end_repetition(count);
                                value!(upper);
                                self.issue(Primitive::Binary { operation: BinaryOp::Lt, ty: Type::I32 }, lanes)?;
                                self.issue(Primitive::Branch, lanes)?;
                                for value in &mut n {
                                    *value = value.map(|v| ((v as i32 as i64) + count as i64 * *step) as u32 as u64);
                                }
                                self.assign(name, n);
                                at = close + 1;
                                continue;
                            }
                            }
                        }
                    }
                    loop {
                        let limit = value!(upper);
                        let mut live = 0;
                        for i in 0..32 {
                            if self.active & (1 << i) != 0 {
                                let (Some(n), Some(limit)) = (n[i], limit[i]) else {
                                    return Ok(Err("unknown target loop domain".into()));
                                };
                                if (n as i32) < (limit as i32) {
                                    live |= 1 << i;
                                }
                            }
                        }
                        self.issue(
                            Primitive::Binary {
                                operation: BinaryOp::Lt,
                                ty: Type::I32,
                            },
                            u64::from(self.active.count_ones()),
                        )?;
                        self.issue(Primitive::Branch, u64::from(self.active.count_ones()))?;
                        self.active = live;
                        if live == 0 {
                            break;
                        }
                        if let Err(g) = self.block(body, at + 1, close)? {
                            return Ok(Err(g));
                        }
                        if self.active == 0 {
                            break;
                        }
                        self.issue(
                            Primitive::Binary {
                                operation: BinaryOp::Add,
                                ty: Type::I32,
                            },
                            u64::from(self.active.count_ones()),
                        )?;
                        for i in 0..32 {
                            if self.active & (1 << i) != 0 {
                                n[i] = n[i]
                                    .and_then(|n| (n as i32).checked_add(*step as i32))
                                    .map(|n| n as u32 as u64);
                            }
                        }
                        self.assign(name, n);
                    }
                    self.active = outer & self.alive;
                    at = close;
                }
                Statement::If(condition) => {
                    let (close, alternative) = matching_end(body, at, end)?;
                    let condition = value!(condition);
                    let yes = match self.predicate(condition) {
                        Ok(v) => v,
                        Err(g) => return Ok(Err(g)),
                    };
                    let outer = self.active;
                    self.issue(Primitive::Branch, lanes)?;
                    self.active = yes;
                    if let Err(g) = self.block(body, at + 1, alternative.unwrap_or(close))? {
                        return Ok(Err(g));
                    }
                    self.active = (outer & !yes) & self.alive;
                    if let Some(other) = alternative {
                        if let Err(g) = self.block(body, other + 1, close)? {
                            return Ok(Err(g));
                        }
                    }
                    self.active = outer & self.alive;
                    at = close;
                }
                Statement::VectorRead { name, base, index, ty, components } => {
                    let indices = value!(index);
                    let byte_width = ty.bytes() * u64::from(*components);
                    // Indices remain scalar-element coordinates; only the
                    // requested interval grows to the vector's exact payload.
                    let access = self.access(base, index, indices, ty.bytes(), byte_width);
                    self.issue_access(Primitive::VectorRead { space: crate::terminal::Space::Device, ty: *ty, components: *components }, lanes, access)?;
                    for component in 0..*components {
                        let component_indices = indices.map(|v| v.and_then(|v| integer_binary(BinaryOp::Add, v, u64::from(component), index.ty())));
                        let values = self.memory.read(base, component_indices, index.ty(), ty.bytes(), self.active);
                        self.assign(&format!("{name}[{component}]"), values);
                    }
                }
                Statement::Pointer {
                    name, base,
                    index,
                    space,
                    ty,
                } => {
                    let indices = value!(index);
                    if let Some(pointer) = self.memory.pointers.get(base).map(|p| p.offset_symbolic(indices, index.ty(), ty.bytes(), self.expression_affine(index))) {
                        self.memory.pointers.insert(name.clone(), pointer);
                    } else { self.memory.pointers.remove(name); }
                    self.issue(
                        Primitive::Address {
                            space: *space,
                            ty: *ty,
                        },
                        lanes,
                    )?;
                    self.assign(name, [None; 32]);
                }
                Statement::Array { name, ty, elements } => {
                    self.memory.array(name, ty.bytes(), *elements, crate::terminal::Space::Private)?;
                }
                Statement::Fragment { .. } => {}
                Statement::MatrixLoad {
                    layout,
                    offset,
                    leading,
                    space,
                    transpose,
                    ..
                } => {
                    if self.active != u32::MAX {
                        return Ok(Err("matrix load requires all subgroup participants".into()));
                    }
                    value!(offset);
                    value!(leading);
                    self.issue(
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                        lanes,
                    )?;
                    self.issue(
                        Primitive::MatrixLoad {
                            layout: *layout,
                            space: *space,
                            transpose: *transpose,
                        },
                        lanes,
                    )?;
                }
                Statement::MatrixStore {
                    base, layout,
                    offset,
                    leading,
                    space,
                    ..
                } => {
                    if self.active != u32::MAX {
                        return Ok(Err("matrix store requires all subgroup participants".into()));
                    }
                    value!(offset);
                    value!(leading);
                    self.memory.invalidate(base, *space);
                    self.issue(
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                        lanes,
                    )?;
                    self.issue(
                        Primitive::MatrixStore {
                            layout: *layout,
                            space: *space,
                        },
                        lanes,
                    )?;
                }
                Statement::MatrixMultiplyAccumulate { layouts, .. } => {
                    if self.active != u32::MAX {
                        return Ok(Err(
                            "matrix arithmetic requires all subgroup participants".into()
                        ));
                    }
                    self.issue(
                        Primitive::MatrixMultiplyAccumulate { layouts: *layouts },
                        lanes,
                    )?;
                }
                Statement::Barrier => {
                    if self.active != u32::MAX {
                        return Err("partial target barrier participation".into());
                    }
                    self.issue(Primitive::Barrier, lanes)?;
                }
                Statement::End | Statement::Else => return Err("unmatched target scope".into()),
                Statement::Unmapped(_) => {
                    return Ok(Err(format!(
                        "unmapped target statement at {:?}",
                        body[at].operation
                    )));
                }
                Statement::Write {
                    name, index,
                    space,
                    ty,
                    value,
                    ..
                } => {
                    let indices = value!(index);
                    let access = if *space == crate::terminal::Space::Device { self.access(name, index, indices, ty.bytes(), ty.bytes()) } else { None };
                    let value = value.clone().cast(*ty);
                    let values = value!(&value);
                    self.issue_access(
                        Primitive::Write {
                            space: *space,
                            ty: *ty,
                        },
                        lanes, access,
                    )?;
                    let symbolic = self.expression_affine(&value);
                    self.memory.write(name, indices, index.ty(), *ty, values, symbolic, self.active, *space);
                    self.facts.clear();
                }
            }
            at += 1;
        }
        Ok(Ok(()))
    }
}

fn reads_name(expression: &Expression, name: &str) -> bool {
    match expression {
        Expression::Variable(source, _) | Expression::Parameter { name: source, .. } => source == name,
        Expression::Binary(_, a, b, _) | Expression::ShortCircuit { left: a, right: b, .. } => reads_name(a, name) || reads_name(b, name),
        Expression::Unary(_, a, _) | Expression::Cast(_, a) | Expression::Bitcast(_, a) | Expression::Read { index: a, .. } => reads_name(a, name),
        Expression::Select(c, a, b) | Expression::EagerSelect(c, a, b) => reads_name(c, name) || reads_name(a, name) || reads_name(b, name),
        Expression::Helper(_, args, _) | Expression::Builtin(_, args, _) => args.iter().any(|a| reads_name(a, name)),
        _ => false,
    }
}

fn invariant_loop_bound(expression: &Expression, body: &[crate::terminal::Site]) -> bool {
    match expression {
        Expression::Integer(..) => true,
        Expression::Variable(name, _) | Expression::Parameter { name, .. } => !body.iter().any(|site| match &site.statement {
            Statement::Let { name: target, .. } | Statement::Assign { name: target, .. }
            | Statement::For { name: target, .. } | Statement::Pointer { name: target, .. }
            | Statement::Array { name: target, .. } | Statement::VectorRead { name: target, .. } => target == name,
            _ => false,
        }),
        Expression::Binary(_, a, b, _) => invariant_loop_bound(a, body) && invariant_loop_bound(b, body),
        Expression::Cast(_, a) | Expression::Unary(_, a, _) => invariant_loop_bound(a, body),
        _ => false,
    }
}

/// Admit a repeated operation structure, not a guessed value trajectory. Every
/// expression is eagerly evaluated and the body cannot change participation,
/// pointer bindings or its own induction variable. Consequently abstracting
/// data can lose later facts but cannot invent extra dynamic operations here.
fn repeated_iterations(name: &str, body: &[crate::terminal::Site], starts: Values, active: u32, limit: i64, step: i64) -> Option<u64> {
    use Expression as E;
    use Statement as S;
    fn eager(e: &E) -> bool {
        match e {
            E::Integer(..) | E::Float(..) | E::Variable(..) | E::Parameter { .. } | E::VectorElement { .. } => true,
            E::Binary(_, a, b, _) => eager(a) && eager(b),
            E::Unary(_, a, _) | E::Cast(_, a) | E::Bitcast(_, a) | E::Read { index: a, .. } => eager(a),
            E::Builtin(_, args, _) => args.iter().all(eager),
            E::Helper(_, args, _) => args.iter().all(eager),
            E::Select(c, a, b) | E::EagerSelect(c, a, b) => eager(c) && eager(a) && eager(b),
            E::ShortCircuit { left, right, .. } => eager(left) && eager(right),
            E::Unmapped(..) => false,
        }
    }
    if step <= 0 || step > i64::from(i32::MAX) || i32::try_from(limit).is_err() || active == 0 { return None; }
    let mut iterations = None;
    for (lane, start) in starts.iter().enumerate() {
        if active & (1 << lane) == 0 { continue; }
        let start = (*start)? as i32 as i64;
        let count = if start >= limit { 0 } else { (limit - start + step - 1) / step };
        if start + count * step > i64::from(i32::MAX) { return None; }
        if iterations.is_some_and(|previous| previous != count as u64) { return None; }
        iterations = Some(count as u64);
    }
    let mut at = 0;
    while at < body.len() {
        let site = &body[at];
        let admitted = match &site.statement {
            S::Let { name: target, value, .. } | S::Assign { name: target, value } => target != name && eager(value),
            S::Evaluate(value) => eager(value),
            S::If(value) => eager(value) && !reads_name(value, name),
            S::Scope | S::End | S::Else => true,
            S::Pointer { name: target, index, .. } => target != name && eager(index),
            S::Write { index, value, .. } => eager(index) && eager(value),
            S::VectorRead { name: target, index, .. } => target != name && eager(index),
            S::Array { name: target, .. } | S::Fragment { name: target, .. } => target != name,
            S::MatrixLoad { offset, leading, .. } | S::MatrixStore { offset, leading, .. } => eager(offset) && eager(leading),
            S::MatrixMultiplyAccumulate { .. } | S::Barrier => true,
            S::For { name: inner, start, end, .. } if inner != name => {
                // The nested invocation independently establishes its trip
                // count and trace over the enclosing symbolic domain.
                eager(start) && eager(end)
            }
            _ => false,
        };
        if !admitted { return None; }
        at += 1;
    }
    iterations
}

/// Only builtins whose exact bit semantics are established here supply known
/// control/address values. Floating arithmetic remains unknown rather than
/// substituting host math for a device operation.
fn builtin_values(name: &str, args: &[Values], ty: Type, active: u32) -> Values {
    if name == "simd_shuffle" && args.len() == 2 {
        return std::array::from_fn(|i| {
            if active & (1 << i) == 0 { return None; }
            let lane = usize::try_from(args[1][i]?).ok()?;
            if lane >= 32 || active & (1 << lane) == 0 { return None; }
            args[0][lane]
        });
    }
    if matches!(ty, Type::F16 | Type::BF16 | Type::F32) { return [None; 32]; }
    let combine = |a, b, maximum| {
        let less = integer_binary(BinaryOp::Lt, a, b, ty)? != 0;
        Some(if less == maximum { b } else { a })
    };
    if matches!(name, "min" | "max") && args.len() == 2 {
        return std::array::from_fn(|i| combine(args[0][i]?, args[1][i]?, name == "max"));
    }
    if matches!(name, "simd_min" | "simd_max") && args.len() == 1 {
        let mut result = None;
        for i in 0..32 {
            if active & (1 << i) == 0 { continue; }
            let Some(value) = args[0][i] else { return [None; 32]; };
            result = match result { None => Some(value), Some(previous) => combine(previous, value, name == "simd_max") };
        }
        return [result; 32];
    }
    [None; 32]
}

fn matching_end(
    body: &[crate::terminal::Site],
    at: usize,
    end: usize,
) -> Result<(usize, Option<usize>), String> {
    let mut depth = 0;
    let mut alternative = None;
    for (i, s) in body.iter().enumerate().take(end).skip(at + 1) {
        match s.statement {
            Statement::If(_) | Statement::For { .. } | Statement::Scope => depth += 1,
            Statement::End if depth == 0 => return Ok((i, alternative)),
            Statement::End => depth -= 1,
            Statement::Else if depth == 0 => alternative = Some(i),
            _ => {}
        }
    }
    Err("unterminated typed target scope".into())
}
fn a_type(e: &Expression) -> Type {
    match e {
        Expression::Binary(_, a, _, _) => a.ty(),
        _ => e.ty(),
    }
}
fn mask(v: u64, t: Type) -> u64 {
    match t {
        Type::Bool => u64::from(v != 0),
        Type::I32 | Type::U32 | Type::F32 => v & u32::MAX as u64,
        Type::F16 | Type::BF16 => v & u16::MAX as u64,
        _ => v,
    }
}
fn integer_cast(v: u64, from: Type, to: Type) -> Option<u64> {
    if matches!(from, Type::F16 | Type::BF16 | Type::F32)
        || matches!(to, Type::F16 | Type::BF16 | Type::F32)
    {
        return if from == to { Some(v) } else { None };
    }
    let v = match from {
        Type::I32 => (v as i32 as i64) as u64,
        _ => v,
    };
    Some(mask(v, to))
}
fn integer_binary(op: BinaryOp, a: u64, b: u64, t: Type) -> Option<u64> {
    if matches!(t, Type::F16 | Type::BF16 | Type::F32) {
        return None;
    }
    use BinaryOp::*;
    let signed = matches!(t, Type::I32 | Type::I64);
    let ai = if t == Type::I32 {
        a as i32 as i64
    } else {
        a as i64
    };
    let bi = if t == Type::I32 {
        b as i32 as i64
    } else {
        b as i64
    };
    Some(match op {
        Add => {
            if signed {
                ai.checked_add(bi)
                    .filter(|v| t != Type::I32 || i32::try_from(*v).is_ok())? as u64
            } else {
                a.wrapping_add(b)
            }
        }
        Sub => {
            if signed {
                ai.checked_sub(bi)
                    .filter(|v| t != Type::I32 || i32::try_from(*v).is_ok())? as u64
            } else {
                a.wrapping_sub(b)
            }
        }
        Mul => {
            if signed {
                ai.checked_mul(bi)
                    .filter(|v| t != Type::I32 || i32::try_from(*v).is_ok())? as u64
            } else {
                a.wrapping_mul(b)
            }
        }
        Div if b != 0 => {
            if signed {
                ai.checked_div(bi)? as u64
            } else {
                a / b
            }
        }
        Rem if b != 0 => {
            if signed {
                ai.checked_rem(bi)? as u64
            } else {
                a % b
            }
        }
        Eq => u64::from(a == b),
        Ne => u64::from(a != b),
        Lt => u64::from(if signed { ai < bi } else { a < b }),
        Le => u64::from(if signed { ai <= bi } else { a <= b }),
        Gt => u64::from(if signed { ai > bi } else { a > b }),
        Ge => u64::from(if signed { ai >= bi } else { a >= b }),
        And => u64::from(a != 0 && b != 0),
        Or => u64::from(a != 0 || b != 0),
        BitAnd => a & b,
        BitOr => a | b,
        BitXor => a ^ b,
        Shl if b < if matches!(t, Type::I64 | Type::U64) {
            64
        } else {
            32
        } =>
        {
            a << b
        }
        Shr if b < if matches!(t, Type::I64 | Type::U64) {
            64
        } else {
            32
        } =>
        {
            if signed {
                (ai >> b) as u64
            } else {
                a >> b
            }
        }
        _ => return None,
    })
}


#[derive(Clone, Debug)]
pub struct Requirements {
    pub primitives: Vec<Primitive>,
    pub unmapped: Vec<String>,
}
/// Resource keys come from the same terminal nodes that render the selected MSL.
/// This inventories static mappings, not dynamic work or a per-kernel timing table.
pub fn requirements(execution: &crate::execution::Execution) -> Result<Requirements, String> {
    let emitted = crate::msl::prepare_execution(execution)?;
    let mut out = Requirements {
        primitives: vec![Primitive::Launch, Primitive::Group],
        unmapped: Vec::new(),
    };
    fn add(out: &mut Requirements, p: Primitive) {
        if !out.primitives.contains(&p) {
            out.primitives.push(p);
        }
    }
    fn expression(e: &Expression, out: &mut Requirements) -> Result<(), String> {
        use Expression as E;
        match e {
            E::Integer(..) | E::Float(..) | E::Variable(..) | E::VectorElement { .. } => {}
            E::Parameter { ty, .. } => add(
                out,
                Primitive::Read {
                    space: crate::terminal::Space::Constant,
                    ty: *ty,
                },
            ),
            E::Unmapped(..) => out.unmapped.push("terminal expression".into()),
            E::Binary(op, a, b, _) => {
                expression(a, out)?;
                expression(b, out)?;
                add(
                    out,
                    Primitive::Binary {
                        operation: *op,
                        ty: a.ty(),
                    },
                );
            }
            E::Unary(op, a, t) => {
                expression(a, out)?;
                add(
                    out,
                    Primitive::Unary {
                        operation: *op,
                        ty: *t,
                    },
                );
            }
            E::Cast(t, a) => {
                expression(a, out)?;
                add(
                    out,
                    Primitive::Cast {
                        from: a.ty(),
                        to: *t,
                    },
                );
            }
            E::Bitcast(t, a) => {
                expression(a, out)?;
                add(
                    out,
                    Primitive::Bitcast {
                        from: a.ty(),
                        to: *t,
                    },
                );
            }
            E::ShortCircuit { left, right, .. } => {
                expression(left, out)?;
                expression(right, out)?;
                add(out, Primitive::Branch);
            }
            E::Select(c, a, b) | E::EagerSelect(c, a, b) => {
                for e in [&**c, &**a, &**b] {
                    expression(e, out)?;
                }
                add(out, Primitive::Select);
            }
            E::Builtin(name, args, result) => {
                for a in args {
                    expression(a, out)?;
                }
                add(
                    out,
                    Primitive::Builtin {
                        name: name.clone(),
                        inputs: args.iter().map(E::ty).collect(),
                        result: *result,
                    },
                );
            }
            E::Helper(helper, args, result) => {
                let definition = crate::support::Definition::new(*helper);
                let element = if *helper == crate::support::Helper::Write {
                    args[3].ty()
                } else {
                    *result
                };
                for a in crate::terminal::helper::arguments(*helper, args, element) {
                    expression(&a, out)?;
                }
                statements(&crate::terminal::helper::body(&definition, element)?, out)?;
            }
            E::Read {
                index, space, ty, ..
            } => {
                expression(index, out)?;
                add(
                    out,
                    Primitive::Read {
                        space: *space,
                        ty: *ty,
                    },
                );
            }
        }
        Ok(())
    }
    fn statements(body: &[crate::terminal::Site], out: &mut Requirements) -> Result<(), String> {
        for site in body {
            match &site.statement {
                Statement::Let { value, .. }
                | Statement::Assign { value, .. }
                | Statement::Evaluate(value) => expression(value, out)?,
                Statement::Write {
                    index,
                    space,
                    ty,
                    value,
                    ..
                } => {
                    expression(index, out)?;
                    expression(&value.clone().cast(*ty), out)?;
                    add(
                        out,
                        Primitive::Write {
                            space: *space,
                            ty: *ty,
                        },
                    );
                }
                Statement::VectorRead { index, ty, components, .. } => {
                    expression(index, out)?;
                    add(out, Primitive::VectorRead { space: crate::terminal::Space::Device, ty: *ty, components: *components });
                }
                Statement::Pointer {
                    index, space, ty, ..
                } => {
                    expression(index, out)?;
                    add(
                        out,
                        Primitive::Address {
                            space: *space,
                            ty: *ty,
                        },
                    );
                }
                Statement::For { start, end, .. } => {
                    expression(start, out)?;
                    expression(end, out)?;
                    add(
                        out,
                        Primitive::Binary {
                            operation: BinaryOp::Lt,
                            ty: Type::I32,
                        },
                    );
                    add(
                        out,
                        Primitive::Binary {
                            operation: BinaryOp::Add,
                            ty: Type::I32,
                        },
                    );
                    add(out, Primitive::Branch);
                }
                Statement::If(e) | Statement::ReturnIf(e) => {
                    expression(e, out)?;
                    add(out, Primitive::Branch);
                    if matches!(site.statement, Statement::ReturnIf(_)) {
                        add(out, Primitive::Return);
                    }
                }
                Statement::Return(e) => {
                    if let Some(e) = e {
                        expression(e, out)?;
                    }
                    add(out, Primitive::Return);
                }
                Statement::MatrixLoad {
                    layout,
                    offset,
                    leading,
                    space,
                    transpose,
                    ..
                } => {
                    expression(offset, out)?;
                    expression(leading, out)?;
                    add(
                        out,
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                    );
                    add(
                        out,
                        Primitive::MatrixLoad {
                            layout: *layout,
                            space: *space,
                            transpose: *transpose,
                        },
                    );
                }
                Statement::MatrixStore {
                    layout,
                    offset,
                    leading,
                    space,
                    ..
                } => {
                    expression(offset, out)?;
                    expression(leading, out)?;
                    add(
                        out,
                        Primitive::Address {
                            space: *space,
                            ty: layout.dtype.into(),
                        },
                    );
                    add(
                        out,
                        Primitive::MatrixStore {
                            layout: *layout,
                            space: *space,
                        },
                    );
                }
                Statement::MatrixMultiplyAccumulate { layouts, .. } => add(
                    out,
                    Primitive::MatrixMultiplyAccumulate { layouts: *layouts },
                ),
                Statement::Barrier => add(out, Primitive::Barrier),
                Statement::Unmapped(_) => {
                    out.unmapped.push(format!("statement {:?}", site.operation))
                }
                Statement::Array { .. }
                | Statement::Fragment { .. }
                | Statement::Else
                | Statement::Scope
                | Statement::End
                | Statement::FailureStatus => {}
            }
        }
        Ok(())
    }
    for launch in emitted.terminal.launches() {
        statements(launch, &mut out)?;
    }
    out.unmapped.sort();
    out.unmapped.dedup();
    Ok(out)
}
