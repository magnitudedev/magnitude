use super::weights::activation_dtype;
use super::{weight_bytes_by_component, ModelLoadPlan};
use crate::{
    PreparedHeadGraphs, PreparedStateCopyGraphs, PreparedTargetGraphs, PreparedTargetReadoutGraphs,
    PreparedVisionGraphs,
};
use magnitude_model_contracts::ModelDefinition;
use magnitude_model_state::{
    BankCapacity, ComponentDescriptor, ComponentSpec, KvCodec, ModelStateLayout, StateStore,
};
use seismic::Device;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionCapacityPlan {
    pub live_sequences: usize,
    pub submitted_successors: usize,
    pub branch_checkpoints: usize,
    pub retained_prefixes: usize,
    pub retained_prefix_bytes: u64,
    pub retained_method_features: usize,
    pub retained_media_features: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    pub active_requests: usize,
    pub in_flight_requests: usize,
    /// Maximum method branch checkpoints that can be resident concurrently.
    pub branch_checkpoints: usize,
    pub max_batch_rows: usize,
    /// Maximum physical rows in one launch that require logits projection.
    pub max_projected_rows: usize,
    pub max_images_per_request: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    pub storage_bytes: u64,
    pub retention_bytes: u64,
    pub safety_reserve_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCapacityPlan {
    pub active: usize,
    pub in_flight: usize,
    pub branch_checkpoints: usize,
    /// Explicit branch checkpoints plus cross-request prefix entries.
    pub retained: usize,
    /// Cross-request entries within `retained`.
    pub retention_entries: usize,
    pub history_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBytes {
    pub target_weights: u64,
    pub head_weights: u64,
    pub vision_weights: u64,
    pub history: u64,
    pub recurrent_banks: u64,
    pub retained_features: u64,
    /// Fixed native argument/result buffers owned by every attested entry.
    pub prepared_programs: u64,
    pub scratch: u64,
    pub safety_reserve: u64,
}

/// A Seismic-derived physical family charge. The engine chooses concurrency;
/// Seismic supplies every byte and every tensor layout within each slot.
/// Each workspace slot holds `upload_regions` upload regions of
/// `upload_bytes`, allocated with the slot: one per graph run its lease
/// keeps in flight at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeGraphCharge {
    pub workspace_bytes: u64,
    pub output_bytes: u64,
    pub upload_bytes: u64,
    pub upload_regions: usize,
    pub workspace_slots: usize,
    pub output_slots: usize,
    pub committed_bytes: u64,
}

/// A family's per-slot storage: scratch, output arena and upload region
/// bytes, and the runs one workspace lease keeps in flight.
#[derive(Clone, Copy, Debug)]
struct FamilyFootprint {
    workspace_bytes: u64,
    output_bytes: u64,
    upload_bytes: u64,
    runs_in_flight: usize,
}

impl FamilyFootprint {
    /// A family whose lease runs one graph at a time to completion.
    fn serial(family: &seismic::NativeGraphFamily) -> Self {
        Self {
            workspace_bytes: family.workspace_bytes(),
            output_bytes: family.output_bytes(),
            upload_bytes: family.upload_bytes(),
            runs_in_flight: 1,
        }
    }
}

impl NativeGraphCharge {
    fn from_prepared(graphs: &PreparedTargetGraphs, in_flight: usize) -> Result<Self, String> {
        let output_slots = in_flight
            .checked_mul(2)
            .ok_or("target graph output slot count overflow")?;
        Self::from_footprint(
            FamilyFootprint {
                workspace_bytes: graphs.workspace_bytes(),
                output_bytes: graphs.output_bytes(),
                upload_bytes: graphs.family().upload_bytes(),
                runs_in_flight: graphs.runs_per_step(),
            },
            in_flight,
            output_slots,
        )
    }

    fn from_readout(
        graphs: &PreparedTargetReadoutGraphs,
        active: usize,
        in_flight: usize,
    ) -> Result<Self, String> {
        let output_slots = active
            .checked_add(in_flight)
            .ok_or("target readout output slot count overflow")?;
        Self::from_footprint(FamilyFootprint::serial(graphs.family()), in_flight, output_slots)
    }

    fn from_footprint(
        footprint: FamilyFootprint,
        workspace_slots: usize,
        output_slots: usize,
    ) -> Result<Self, String> {
        let count = |value: usize| u64::try_from(value).map_err(|_| "slot count exceeds u64");
        let per_slot = footprint
            .upload_bytes
            .checked_mul(count(footprint.runs_in_flight)?)
            .and_then(|uploads| uploads.checked_add(footprint.workspace_bytes))
            .ok_or("graph slot byte count overflows")?;
        let committed_bytes = per_slot
            .checked_mul(count(workspace_slots)?)
            .and_then(|bytes| {
                footprint
                    .output_bytes
                    .checked_mul(u64::try_from(output_slots).ok()?)
                    .and_then(|outputs| bytes.checked_add(outputs))
            })
            .ok_or("graph charge overflows")?;
        Ok(Self {
            workspace_bytes: footprint.workspace_bytes,
            output_bytes: footprint.output_bytes,
            upload_bytes: footprint.upload_bytes,
            upload_regions: footprint.runs_in_flight,
            workspace_slots,
            output_slots,
            committed_bytes,
        })
    }
}

/// Exact projection used to construct one numerical state arena. The planner
/// owns every capacity and byte fact; construction only materializes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateStorePlan {
    pub context_rows: usize,
    pub history_rows: usize,
    pub history_components: Vec<ComponentDescriptor>,
    pub recurrent_components: Vec<ComponentSpec>,
    pub bank_capacity: BankCapacity,
    pub history_row_bytes: u64,
    pub history_bytes: u64,
    pub recurrent_bank_bytes: u64,
    pub zero_seed_bytes: u64,
    pub recurrent_pool_bytes: u64,
}

impl StateStorePlan {
    pub fn allocate(&self, device: Rc<Device>) -> Result<Rc<StateStore>, String> {
        let store = StateStore::new(
            device,
            self.context_rows,
            self.history_rows,
            self.history_components.clone(),
            self.recurrent_components.clone(),
            self.bank_capacity,
        )
        .map_err(|error| error.to_string())?;
        let trace = store
            .allocation_trace()
            .map_err(|error| error.to_string())?;
        if trace.context_capacity != self.context_rows
            || trace.history_capacity != self.history_rows
            || trace.history_row_bytes != self.history_row_bytes
            || trace.history_bytes != self.history_bytes
            || trace.bank_capacity != self.bank_capacity
            || trace.recurrent_bank_bytes != self.recurrent_bank_bytes
            || trace.zero_seed_bytes != self.zero_seed_bytes
            || trace.recurrent_pool_bytes != self.recurrent_pool_bytes
        {
            return Err("state allocation differs from its resource-plan projection".into());
        }
        Ok(store)
    }
}

impl ResourceBytes {
    pub fn total(self) -> Result<u64, String> {
        [
            self.target_weights,
            self.head_weights,
            self.vision_weights,
            self.history,
            self.recurrent_banks,
            self.retained_features,
            self.prepared_programs,
            self.scratch,
            self.safety_reserve,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| "resource byte total overflow".into())
    }
}

/// Immutable allocation authority. Every physical constructor receives the
/// relevant projection of this value rather than recomputing capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourcePlan {
    pub(super) storage_bytes: u64,
    retention_budget_bytes: u64,
    capacity: StateCapacityPlan,
    target_state: StateStorePlan,
    head_state: Option<StateStorePlan>,
    retained_entry_bytes: u64,
    bytes: ResourceBytes,
    pub(super) qualification_peak_bytes: u64,
    target_graph: NativeGraphCharge,
    target_readout_graph: NativeGraphCharge,
    head_graph: Option<NativeGraphCharge>,
    vision_graph: Option<NativeGraphCharge>,
    state_graph: NativeGraphCharge,
    retention: RetentionCapacityPlan,
    steady_committed_bytes: u64,
    startup_peak_bytes: u64,
}

impl ResourcePlan {
    pub(crate) fn qualification_peak_bytes(&self) -> u64 {
        self.qualification_peak_bytes
    }
    pub fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }
    pub fn retention_budget_bytes(&self) -> u64 {
        self.retention_budget_bytes
    }
    pub fn capacity(&self) -> StateCapacityPlan {
        self.capacity
    }
    pub fn target_state(&self) -> &StateStorePlan {
        &self.target_state
    }
    pub fn head_state(&self) -> Option<&StateStorePlan> {
        self.head_state.as_ref()
    }
    pub fn retained_entry_bytes(&self) -> u64 {
        self.retained_entry_bytes
    }
    pub fn bytes(&self) -> ResourceBytes {
        self.bytes
    }

    pub(super) fn validate(mut self) -> Result<Self, String> {
        if self.bytes.scratch
            != self
                .target_graph
                .committed_bytes
                .checked_add(self.target_readout_graph.committed_bytes)
                .and_then(|bytes| {
                    bytes.checked_add(self.head_graph.map_or(0, |graph| graph.committed_bytes))
                })
                .and_then(|bytes| {
                    bytes.checked_add(self.vision_graph.map_or(0, |graph| graph.committed_bytes))
                })
                .and_then(|bytes| bytes.checked_add(self.state_graph.committed_bytes))
                .ok_or("pooled byte charge overflow")?
        {
            return Err(
                "resource plan pooled byte charge differs from admitted Seismic footprints".into(),
            );
        }
        if self.retention.live_sequences != self.capacity.active
            || self.retention.submitted_successors != self.capacity.in_flight
            || self.retention.branch_checkpoints != self.capacity.branch_checkpoints
            || self.retention.retained_prefixes != self.capacity.retention_entries
            || self.capacity.retained
                != self
                    .capacity
                    .branch_checkpoints
                    .checked_add(self.capacity.retention_entries)
                    .ok_or("retention slot count overflow")?
            || self.retention.retained_prefix_bytes != self.retention_budget_bytes
        {
            return Err("resource plan retention capacities disagree with service limits".into());
        }
        let planned_media_outputs = self.vision_graph.map_or(0, |graph| graph.output_slots);
        if self.retention.retained_media_features != planned_media_outputs {
            return Err("retained media capacity differs from planned output leases".into());
        }
        self.steady_committed_bytes = self.bytes.total()?;
        self.startup_peak_bytes = self
            .steady_committed_bytes
            .checked_add(self.qualification_peak_bytes)
            .ok_or("startup peak byte count overflow")?;
        if self.startup_peak_bytes > self.storage_bytes {
            return Err(format!(
                "resource plan startup peak {} exceeds {} bytes",
                self.startup_peak_bytes, self.storage_bytes
            ));
        }
        Ok(self)
    }

    pub fn target_graph(&self) -> NativeGraphCharge {
        self.target_graph
    }
    pub fn target_readout_graph(&self) -> NativeGraphCharge {
        self.target_readout_graph
    }
    pub fn head_graph(&self) -> Option<NativeGraphCharge> {
        self.head_graph
    }
    pub fn vision_graph(&self) -> Option<NativeGraphCharge> {
        self.vision_graph
    }
    pub fn state_graph(&self) -> NativeGraphCharge {
        self.state_graph
    }
    pub fn retention(&self) -> RetentionCapacityPlan {
        self.retention
    }
    pub fn steady_committed_bytes(&self) -> u64 {
        self.steady_committed_bytes
    }
    pub fn startup_peak_bytes(&self) -> u64 {
        self.startup_peak_bytes
    }

    pub fn allocate_target_state(&self, device: Rc<Device>) -> Result<Rc<StateStore>, String> {
        self.target_state.allocate(device)
    }

    pub fn allocate_head_state(
        &self,
        device: Rc<Device>,
    ) -> Result<Option<Rc<StateStore>>, String> {
        self.head_state
            .as_ref()
            .map(|plan| plan.allocate(device))
            .transpose()
    }

    pub fn memory_requirements(&self) -> crate::platform::MemoryRequirements {
        crate::platform::MemoryRequirements {
            immutable_weights: self.bytes.target_weights
                + self.bytes.head_weights
                + self.bytes.vision_weights,
            state: self.bytes.history + self.bytes.recurrent_banks + self.bytes.retained_features,
            scratch: self.bytes.scratch
                + self.bytes.prepared_programs
                + self.qualification_peak_bytes,
            safety_margin: self.bytes.safety_reserve,
        }
    }
}

pub struct ResourcePlanner;

/// State shape and retention capacity are known before Seismic constructs
/// graph ports. Graph scratch is admitted only after Seismic seals its checked
/// stage topology and reports the exact storage footprint.
#[derive(Clone, Debug, PartialEq)]
pub struct StateResourcePlan {
    definition: ModelDefinition,
    load: ModelLoadPlan,
    limits: ResourceLimits,
    budget: ResourceBudget,
    capacity: StateCapacityPlan,
    target_state: StateStorePlan,
    head_state: Option<StateStorePlan>,
    retained_entry_bytes: u64,
    history_bytes: u64,
    recurrent_banks_bytes: u64,
    retained_features_bytes: u64,
    retention: RetentionCapacityPlan,
}

impl StateResourcePlan {
    pub fn capacity(&self) -> StateCapacityPlan {
        self.capacity
    }

    pub fn target_state(&self) -> &StateStorePlan {
        &self.target_state
    }

    pub fn head_state(&self) -> Option<&StateStorePlan> {
        self.head_state.as_ref()
    }
}

impl ResourcePlanner {
    pub fn state_plan(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        limits: ResourceLimits,
        budget: ResourceBudget,
    ) -> Result<StateResourcePlan, String> {
        if limits.active_requests == 0
            || limits.in_flight_requests == 0
            || limits.max_batch_rows == 0
            || limits.max_projected_rows == 0
            || limits.max_projected_rows > limits.max_batch_rows
            || limits.max_images_per_request == 0
        {
            return Err("resource limits must be positive".into());
        }
        if limits.max_images_per_request > magnitude_artifacts::MAX_IMAGES_PER_REQUEST {
            return Err("planned image limit exceeds preprocessing capacity".into());
        }
        if budget.storage_bytes == 0
            || budget.safety_reserve_bytes >= budget.storage_bytes
            || budget.retention_bytes > budget.storage_bytes - budget.safety_reserve_bytes
        {
            return Err("invalid resource storage budget".into());
        }
        let layout = ModelStateLayout::derive(
            &definition.geometry,
            load.head
                .as_ref()
                .and_then(|_| definition.head.as_ref())
                .map_or(0, |head| head.depth()),
            codec,
        )?;
        let target_history_row_bytes = history_row_bytes(&layout.target_history)?;
        let head_history_row_bytes = history_row_bytes(&layout.head_history)?;
        let checkpoint_history_row_bytes = target_history_row_bytes
            .checked_add(head_history_row_bytes)
            .ok_or("history row byte count overflow")?;
        let recurrent_bank_bytes = layout
            .target_recurrent
            .iter()
            .try_fold(0u64, |total, component| {
                let values = component.shape.iter().try_fold(1u64, |count, extent| {
                    count.checked_mul(u64::try_from(*extent).ok()?)
                })?;
                total.checked_add(values.checked_mul(u64::from(component.dtype.bytes()))?)
            })
            .ok_or("recurrent bank byte count overflow")?;
        let context = usize::try_from(definition.geometry.context_limit)
            .map_err(|_| "context limit exceeds host domain")?;
        let checkpoint_state_bytes = checkpoint_history_row_bytes
            .checked_mul(definition.geometry.context_limit)
            .and_then(|bytes| bytes.checked_add(recurrent_bank_bytes))
            .ok_or("checkpoint state byte count overflow")?;
        let activation_bytes =
            u64::from(activation_dtype(definition.geometry.activation_dtype).bytes());
        let method_feature_bytes = if load.head.is_some() {
            definition
                .geometry
                .hidden
                .checked_mul(activation_bytes)
                .and_then(|row| row.checked_mul(definition.geometry.context_limit))
                .ok_or("retained method feature byte count overflow")?
        } else {
            0
        };
        let retained_entry_bytes = checkpoint_state_bytes
            .checked_add(method_feature_bytes)
            .ok_or("retained unit byte count overflow")?;
        let retention_entries = if retained_entry_bytes == 0 {
            0
        } else {
            usize::try_from(budget.retention_bytes / retained_entry_bytes)
                .map_err(|_| "retained checkpoint count exceeds host domain")?
        };
        let retained = limits
            .branch_checkpoints
            .checked_add(retention_entries)
            .ok_or("retained numerical capacity overflow")?;
        let capacity = StateCapacityPlan {
            active: limits.active_requests,
            in_flight: limits.in_flight_requests,
            branch_checkpoints: limits.branch_checkpoints,
            retained,
            retention_entries,
            history_rows: limits
                .active_requests
                .checked_add(limits.in_flight_requests)
                .and_then(|owners| owners.checked_add(retained))
                .and_then(|owners| owners.checked_mul(context))
                .and_then(|rows| rows.checked_add(limits.max_batch_rows))
                .ok_or("history row capacity overflow")?,
        };
        let bank_capacity = BankCapacity {
            active: capacity.active,
            in_flight: capacity.in_flight,
            retained: capacity.retained,
        };
        let bank_count = bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?;
        let recurrent_pool_bytes = recurrent_bank_bytes
            .checked_mul(u64::try_from(bank_count).map_err(|_| "bank capacity exceeds u64")?)
            .ok_or("recurrent bank allocation byte count overflow")?;
        let target_state = state_store_plan(
            context,
            capacity.history_rows,
            layout.target_history,
            layout.target_recurrent,
            bank_capacity,
        )?;
        let head_state = (!layout.head_history.is_empty())
            .then(|| {
                state_store_plan(
                    context,
                    capacity.history_rows,
                    layout.head_history,
                    Vec::new(),
                    bank_capacity,
                )
            })
            .transpose()?;
        let history = target_state
            .history_bytes
            .checked_add(head_state.as_ref().map_or(0, |state| state.history_bytes))
            .ok_or("history allocation byte count overflow")?;
        let planned_recurrent = target_state
            .recurrent_pool_bytes
            .checked_add(
                head_state
                    .as_ref()
                    .map_or(0, |state| state.recurrent_pool_bytes),
            )
            .ok_or("recurrent allocation byte count overflow")?;
        if planned_recurrent != recurrent_pool_bytes {
            return Err("state projections disagree on recurrent allocation bytes".into());
        }
        let retained_features = method_feature_bytes
            .checked_mul(u64::try_from(retained).map_err(|_| "retained count exceeds u64")?)
            .ok_or("retained feature allocation byte count overflow")?;
        let retention = RetentionCapacityPlan {
            live_sequences: capacity.active,
            submitted_successors: capacity.in_flight,
            branch_checkpoints: capacity.branch_checkpoints,
            retained_prefixes: capacity.retention_entries,
            retained_prefix_bytes: budget.retention_bytes,
            retained_method_features: if load.head.is_some() { retained } else { 0 },
            retained_media_features: if load.vision.is_some() {
                capacity
                    .active
                    .checked_add(capacity.in_flight)
                    .and_then(|owners| owners.checked_add(capacity.retained))
                    .and_then(|owners| owners.checked_mul(limits.max_images_per_request))
                    .ok_or("retained media feature capacity overflow")?
            } else {
                0
            },
        };
        Ok(StateResourcePlan {
            definition: definition.clone(),
            load: load.clone(),
            limits,
            budget,
            capacity,
            target_state,
            head_state,
            retained_entry_bytes,
            history_bytes: history,
            recurrent_banks_bytes: planned_recurrent,
            retained_features_bytes: retained_features,
            retention,
        })
    }

    pub fn plan_with_state(
        state: StateResourcePlan,
        target_graphs: &PreparedTargetGraphs,
        target_readout_graphs: &PreparedTargetReadoutGraphs,
        head_graphs: Option<&PreparedHeadGraphs>,
        vision_graphs: Option<&PreparedVisionGraphs>,
        state_graphs: &PreparedStateCopyGraphs,
    ) -> Result<ResourcePlan, String> {
        let definition = &state.definition;
        let load = &state.load;
        let limits = state.limits;
        let budget = state.budget;
        let qualification_peak = crate::AttestedPrograms::qualification_peak_bytes(load);
        let source_import_peak = load
            .target
            .iter()
            .chain(load.head.iter().flatten())
            .chain(load.vision.iter().flatten())
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        let qualification_peak_bytes = qualification_peak.max(source_import_peak);
        let target_graph =
            NativeGraphCharge::from_prepared(target_graphs, limits.in_flight_requests)?;
        let target_readout_graph = NativeGraphCharge::from_readout(
            target_readout_graphs,
            limits.active_requests,
            limits.in_flight_requests,
        )?;
        let head_graph = head_graphs
            .map(|graphs| {
                NativeGraphCharge::from_footprint(
                    FamilyFootprint::serial(graphs.family()),
                    limits.in_flight_requests,
                    limits
                        .active_requests
                        .checked_add(limits.in_flight_requests)
                        .ok_or("head graph output slot count overflow")?,
                )
            })
            .transpose()?;
        let vision_graph = vision_graphs
            .map(|graphs| {
                NativeGraphCharge::from_footprint(
                    FamilyFootprint::serial(graphs.family()),
                    limits.in_flight_requests,
                    state.retention.retained_media_features,
                )
            })
            .transpose()?;
        let state_graph = NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(state_graphs.family()),
            limits.in_flight_requests,
            limits.in_flight_requests,
        )?;
        let scratch = target_graph
            .committed_bytes
            .checked_add(target_readout_graph.committed_bytes)
            .and_then(|bytes| {
                bytes.checked_add(head_graph.map_or(0, |graph| graph.committed_bytes))
            })
            .and_then(|bytes| {
                bytes.checked_add(vision_graph.map_or(0, |graph| graph.committed_bytes))
            })
            .and_then(|bytes| bytes.checked_add(state_graph.committed_bytes))
            .ok_or("numerical pool byte count overflow")?;
        let prepared_programs = crate::AttestedPrograms::planned_invocation_workspace_bytes(
            &load
                .program_plan(definition)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let [target_weights, head_weights, vision_weights] = weight_bytes_by_component(load)?;
        let bytes = ResourceBytes {
            target_weights,
            head_weights,
            vision_weights,
            history: state.history_bytes,
            recurrent_banks: state.recurrent_banks_bytes,
            retained_features: state.retained_features_bytes,
            prepared_programs,
            scratch,
            safety_reserve: budget.safety_reserve_bytes,
        };
        let required = bytes.total()?;
        if std::env::var_os("MAGNITUDE_TRACE_RESOURCES").is_some() {
            eprintln!(
                "resource plan budget={} required={} qualification_peak={} weights=[{},{},{}] history={} recurrent_banks={} retained_features={} prepared_programs={} scratch={} safety_reserve={} graph=[target:{},readout:{},head:{},vision:{},state:{}]",
                budget.storage_bytes,
                required,
                qualification_peak_bytes,
                bytes.target_weights,
                bytes.head_weights,
                bytes.vision_weights,
                bytes.history,
                bytes.recurrent_banks,
                bytes.retained_features,
                bytes.prepared_programs,
                bytes.scratch,
                bytes.safety_reserve,
                target_graph.committed_bytes,
                target_readout_graph.committed_bytes,
                head_graph.map_or(0, |graph| graph.committed_bytes),
                vision_graph.map_or(0, |graph| graph.committed_bytes),
                state_graph.committed_bytes,
            );
        }
        if required > budget.storage_bytes {
            return Err(format!(
                "resource plan requires {required} bytes but the device budget is {}",
                budget.storage_bytes
            ));
        }
        ResourcePlan {
            storage_bytes: budget.storage_bytes,
            retention_budget_bytes: budget.retention_bytes,
            capacity: state.capacity,
            target_state: state.target_state,
            head_state: state.head_state,
            retained_entry_bytes: state.retained_entry_bytes,
            bytes,
            qualification_peak_bytes,
            target_graph,
            target_readout_graph,
            head_graph,
            vision_graph,
            state_graph,
            retention: state.retention,
            steady_committed_bytes: 0,
            startup_peak_bytes: 0,
        }
        .validate()
    }
}

fn state_store_plan(
    context_rows: usize,
    history_rows: usize,
    history_components: Vec<ComponentDescriptor>,
    recurrent_components: Vec<ComponentSpec>,
    bank_capacity: BankCapacity,
) -> Result<StateStorePlan, String> {
    let history_row_bytes = history_row_bytes(&history_components)?;
    let history_bytes = history_row_bytes
        .checked_mul(u64::try_from(history_rows).map_err(|_| "history rows exceed u64")?)
        .ok_or("history allocation byte count overflow")?;
    let recurrent_bank_bytes = recurrent_components
        .iter()
        .try_fold(0_u64, |total, component| {
            let values = component.shape.iter().try_fold(1_u64, |count, extent| {
                count.checked_mul(u64::try_from(*extent).ok()?)
            })?;
            total.checked_add(values.checked_mul(u64::from(component.dtype.bytes()))?)
        })
        .ok_or("recurrent bank byte count overflow")?;
    let recurrent_pool_bytes = recurrent_bank_bytes
        .checked_mul(
            u64::try_from(
                bank_capacity
                    .storage_total()
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|_| "bank capacity exceeds u64")?,
        )
        .ok_or("recurrent pool byte count overflow")?;
    Ok(StateStorePlan {
        context_rows,
        history_rows,
        history_components,
        recurrent_components,
        bank_capacity,
        history_row_bytes,
        history_bytes,
        recurrent_bank_bytes,
        zero_seed_bytes: recurrent_bank_bytes,
        recurrent_pool_bytes,
    })
}

fn history_row_bytes(
    components: &[magnitude_model_state::ComponentDescriptor],
) -> Result<u64, String> {
    components
        .iter()
        .flat_map(|component| component.planes())
        .try_fold(0u64, |total, plane| {
            total
                .checked_add(
                    u64::try_from(plane.row_bytes).map_err(|_| "history row bytes exceed u64")?,
                )
                .ok_or_else(|| "history row byte count overflow".into())
        })
}
