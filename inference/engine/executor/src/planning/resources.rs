use super::{source_import_peak_bytes, weight_bytes_by_component, ModelLoadPlan, PlannedMethod};
use crate::{
    PreparedHeadGraphs, PreparedStateCopyGraphs, PreparedTargetGraphs, PreparedTargetReadoutGraphs,
    PreparedVisionGraphs,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{
    BankCapacity, ComponentDescriptor, ComponentSpec, KvCodec, ModelStateLayout, StateStore,
};
use seismic::{DType, Device, Element, SlabLayout, SlabRegion};
use std::rc::Rc;

/// The service's bounds a load plans for. None is a request-count batch
/// width: a launch is bounded by its token budget (rows, projected rows and
/// request slots are compiled shape classes), and every per-request resource
/// beyond what one request needs grows elastically under heap claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Rows one launch admits: the larger step token budget.
    pub max_launch_rows: usize,
    /// Requests one launch serves: its largest request-slot class.
    pub max_launch_slots: usize,
    /// Maximum physical rows in one launch that require logits projection.
    pub max_projected_rows: usize,
    pub max_images_per_request: usize,
    /// Queue each continuable target step's successor before the step
    /// completes (cross-step pipelining): one more target launch in flight
    /// and one more successor bank per live request.
    pub lookahead: bool,
}

impl ResourceLimits {
    /// Target launches in flight at once: a step and, with lookahead, its
    /// queued successor. The owner submits one launch at a time.
    pub fn target_launches(&self) -> usize {
        1 + usize::from(self.lookahead)
    }

    /// The graph slots a load commits at startup: what one request needs to
    /// run. Workspace slots cover the launches in flight at once, whatever
    /// the request count. Outputs a request retains after its launch
    /// completes (readout features, head drafts, encoded images) start at one
    /// request's need and grow one slot at a time under a heap claim.
    pub fn startup_slots(&self) -> StartupSlots {
        let launches = self.target_launches();
        StartupSlots {
            target: GraphSlots {
                workspace: launches,
                output: 2 * launches,
            },
            readout: GraphSlots {
                workspace: launches,
                output: 1 + launches,
            },
            head: GraphSlots {
                workspace: 1,
                output: 2,
            },
            vision: GraphSlots {
                workspace: 1,
                output: self.max_images_per_request,
            },
            state: GraphSlots {
                workspace: 1,
                output: 1,
            },
        }
    }
}

/// One graph family's slot counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphSlots {
    pub workspace: usize,
    pub output: usize,
}

/// Every graph family's startup slots (see [`ResourceLimits::startup_slots`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartupSlots {
    pub target: GraphSlots,
    pub readout: GraphSlots,
    pub head: GraphSlots,
    pub vision: GraphSlots,
    pub state: GraphSlots,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceCapacity {
    /// Stable capacity of the selected device's physical allocation domain.
    pub domain_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCapacityPlan {
    /// Largest number of one-row requests the physical domain could hold.
    pub live_requests: usize,
    /// Explicit branch checkpoints resident at once.
    pub checkpoints: usize,
    /// Cross-request prefix entries retention may hold.
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
    /// Fixed native argument/result buffers owned by every attested entry.
    pub prepared_programs: u64,
    pub scratch: u64,
}

/// A Seismic-derived physical family charge at startup. The engine chooses
/// the slot counts; Seismic supplies every byte and every tensor layout
/// within each slot. Each workspace slot holds `upload_regions` upload
/// regions of `upload_bytes`, allocated with the slot: one per graph run its
/// lease keeps in flight at once. Output slots beyond `output_slots` are
/// elastic run-time claims of `output_bytes` each.
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
    pub(crate) fn from_checked(
        storage: seismic::NativeGraphStorageBytes,
        runs_in_flight: usize,
        slots: GraphSlots,
    ) -> Result<Self, String> {
        Self::from_footprint(
            FamilyFootprint {
                workspace_bytes: storage.workspace,
                output_bytes: storage.output,
                upload_bytes: storage.upload,
                runs_in_flight,
            },
            slots,
        )
    }

    fn from_prepared(graphs: &PreparedTargetGraphs, slots: GraphSlots) -> Result<Self, String> {
        Self::from_footprint(
            FamilyFootprint {
                workspace_bytes: graphs.workspace_bytes(),
                output_bytes: graphs.output_bytes(),
                upload_bytes: graphs.family().upload_bytes(),
                runs_in_flight: graphs.runs_per_step(),
            },
            slots,
        )
    }

    fn from_footprint(footprint: FamilyFootprint, slots: GraphSlots) -> Result<Self, String> {
        let GraphSlots {
            workspace: workspace_slots,
            output: output_slots,
        } = slots;
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
    pub fn max_visible_spans(&self) -> Result<usize, String> {
        if self.history_components.is_empty() {
            return Ok(1);
        }
        magnitude_state::max_visible_spans(self.context_rows, self.history_slab_rows()? as usize)
    }

    pub fn history_slab_rows(&self) -> Result<u32, String> {
        u32::try_from(magnitude_state::history_rows_per_slab(self.history_row_bytes)?)
            .map_err(|_| "history slab rows exceed u32".into())
    }

    pub fn bank_slab_banks(&self) -> Result<u32, String> {
        u32::try_from(magnitude_state::banks_per_slab(self.recurrent_bank_bytes)?)
            .map_err(|_| "bank slab count exceeds u32".into())
    }

    pub fn history_slab_layout(&self) -> Result<Option<SlabLayout>, String> {
        if self.history_components.is_empty() {
            return Ok(None);
        }
        let regions = self
            .history_components
            .iter()
            .flat_map(ComponentDescriptor::planes)
            .map(|plane| SlabRegion {
                element: slab_element(plane.dtype),
                row_shape: plane.row_extents.iter().map(|&size| size as u64).collect(),
            })
            .collect::<Vec<_>>();
        SlabLayout::for_regions(
            u64::from(self.history_slab_rows()?),
            self.history_rows as u64,
            &regions,
        )
        .map(Some)
        .map_err(|error| error.to_string())
    }

    pub fn bank_slab_layout(&self) -> Result<Option<SlabLayout>, String> {
        if self.recurrent_components.is_empty() {
            return Ok(None);
        }
        let regions = self
            .recurrent_components
            .iter()
            .map(|component| SlabRegion {
                element: slab_element(component.dtype),
                row_shape: component.shape.iter().map(|&size| size as u64).collect(),
            })
            .collect::<Vec<_>>();
        let logical_banks = self
            .bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?;
        SlabLayout::for_regions(
            u64::from(self.bank_slab_banks()?),
            logical_banks as u64,
            &regions,
        )
        .map(Some)
        .map_err(|error| error.to_string())
    }

    pub fn startup_history_bytes(&self) -> Result<u64, String> {
        self.history_slab_layout()?.map_or(Ok(0), |layout| {
            layout
                .address_table_bytes
                .checked_add(layout.slab_bytes)
                .ok_or_else(|| "initial history slab charge overflows".into())
        })
    }

    pub fn startup_bank_bytes(&self) -> Result<u64, String> {
        self.bank_slab_layout()?.map_or(Ok(0), |layout| {
            layout
                .address_table_bytes
                .checked_add(layout.slab_bytes)
                .ok_or_else(|| "initial bank slab charge overflows".into())
        })
    }

    pub fn history_bytes_at_depth(&self, depth: u64) -> Result<u64, String> {
        let Some(layout) = self.history_slab_layout()? else {
            return Ok(0);
        };
        let slabs = depth.max(1).div_ceil(u64::from(self.history_slab_rows()?));
        layout
            .slab_bytes
            .checked_mul(slabs)
            .and_then(|bytes| bytes.checked_add(layout.address_table_bytes))
            .ok_or_else(|| "history slab fit charge overflows".into())
    }

    pub fn bank_bytes_at_count(&self, banks: u64) -> Result<u64, String> {
        let Some(layout) = self.bank_slab_layout()? else {
            return Ok(0);
        };
        let slabs = banks.max(1).div_ceil(u64::from(self.bank_slab_banks()?));
        layout
            .slab_bytes
            .checked_mul(slabs)
            .and_then(|bytes| bytes.checked_add(layout.address_table_bytes))
            .ok_or_else(|| "bank slab fit charge overflows".into())
    }

    /// Address tables and the first history and bank slabs committed by
    /// `StateStore::new` for each present store.
    pub fn initial_committed_bytes(&self) -> Result<u64, String> {
        self.startup_history_bytes()?
            .checked_add(self.startup_bank_bytes()?)
            .ok_or_else(|| "initial state slab charge overflows".into())
    }

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
            || store.committed_bytes() != self.initial_committed_bytes()?
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
            self.prepared_programs,
            self.scratch,
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
    pub(super) domain_capacity_bytes: u64,
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
    steady_committed_bytes: u64,
    startup_peak_bytes: u64,
}

impl ResourcePlan {
    pub(crate) fn qualification_peak_bytes(&self) -> u64 {
        self.qualification_peak_bytes
    }
    pub fn domain_capacity_bytes(&self) -> u64 {
        self.domain_capacity_bytes
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
        self.steady_committed_bytes = self.bytes.total()?;
        self.startup_peak_bytes = self
            .steady_committed_bytes
            .checked_add(self.qualification_peak_bytes)
            .ok_or("startup peak byte count overflow")?;
        if self.startup_peak_bytes > self.domain_capacity_bytes {
            return Err(format!(
                "resource plan startup peak {} exceeds {} bytes",
                self.startup_peak_bytes, self.domain_capacity_bytes
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
}

pub struct ResourcePlanner;

/// State shape and retention capacity are known before Seismic constructs
/// graph ports. Graph scratch is admitted only after Seismic seals its checked
/// stage topology and reports the exact storage footprint.
#[derive(Clone, Debug, PartialEq)]
pub struct StateResourcePlan {
    definition: ModelDefinition,
    load: ModelLoadPlan,
    codec: KvCodec,
    limits: ResourceLimits,
    capacity_bytes: ResourceCapacity,
    capacity: StateCapacityPlan,
    target_state: StateStorePlan,
    head_state: Option<StateStorePlan>,
    retained_entry_bytes: u64,
    history_bytes: u64,
    recurrent_banks_bytes: u64,
}

impl StateResourcePlan {
    pub fn limits(&self) -> ResourceLimits {
        self.limits
    }

    pub fn capacity(&self) -> StateCapacityPlan {
        self.capacity
    }

    pub fn fit_state_bytes(&self, depth: u64, banks: u64) -> Result<u64, String> {
        let target_history = self.target_state.history_bytes_at_depth(depth)?;
        let head_history = self
            .head_state
            .as_ref()
            .map(|state| state.history_bytes_at_depth(depth))
            .transpose()?
            .unwrap_or(0);
        let target_banks = self.target_state.bank_bytes_at_count(banks)?;
        target_history
            .checked_add(head_history)
            .and_then(|bytes| bytes.checked_add(target_banks))
            .ok_or_else(|| "state fit charge overflows".into())
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
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
        capacity_bytes: ResourceCapacity,
    ) -> Result<StateResourcePlan, String> {
        if limits.max_launch_slots == 0
            || limits.max_launch_slots > limits.max_launch_rows
            || limits.max_launch_rows == 0
            || limits.max_projected_rows == 0
            || limits.max_projected_rows > limits.max_launch_rows
            || limits.max_images_per_request == 0
        {
            return Err("resource limits must be positive".into());
        }
        if limits.max_images_per_request > magnitude_artifacts::MAX_IMAGES_PER_REQUEST {
            return Err("planned image limit exceeds preprocessing capacity".into());
        }
        if capacity_bytes.domain_bytes == 0 {
            return Err("device domain has zero capacity".into());
        }
        let layout = ModelStateLayout::derive(
            &definition.geometry,
            load.head
                .as_ref()
                .and_then(|_| definition.head.as_ref())
                .map_or(0, |head| head.depth()),
            codec,
            method.draft_rows(),
        )?;
        let target_history_row_bytes = history_row_bytes(&layout.target_history)?;
        let head_history_row_bytes = history_row_bytes(&layout.head_history)?;
        let checkpoint_history_row_bytes = target_history_row_bytes
            .checked_add(head_history_row_bytes)
            .ok_or("history row byte count overflow")?;
        let recurrent_bank_bytes = recurrent_bank_bytes(&layout.target_recurrent)?;
        let context = usize::try_from(definition.geometry.context_limit)
            .map_err(|_| "context limit exceeds host domain")?;
        // Generation methods keep their carried feature rows on the host, so
        // a retained entry charges only numerical state. Entries on one path
        // share their history rows, so an entry's marginal state is its recurrent
        // bank; without recurrent state it is a full context of history.
        let retained_entry_bytes = if recurrent_bank_bytes != 0 {
            recurrent_bank_bytes
        } else {
            checkpoint_history_row_bytes
                .checked_mul(definition.geometry.context_limit)
                .ok_or("checkpoint state byte count overflow")?
        };
        // Reservations are index bounds sealed into graphs, not memory: every
        // bank and history row is committed on demand under a heap claim. A
        // live request holds its accepted bank and, in flight, a successor
        // (two with lookahead); a branch checkpoint or retention entry holds
        // one more. Retention is bounded by what the domain could back.
        let retention_entries = if retained_entry_bytes == 0 {
            0
        } else {
            usize::try_from(capacity_bytes.domain_bytes / retained_entry_bytes)
                .unwrap_or(usize::MAX)
        };
        let history_bound =
            usize::try_from(capacity_bytes.domain_bytes / checkpoint_history_row_bytes.max(1))
                .unwrap_or(usize::MAX);
        if checkpoint_history_row_bytes != 0 && history_bound < context {
            return Err("one context exceeds the device domain's history capacity".into());
        }
        // A request needs at least one history row and one recurrent bank.
        // This is an address-space ceiling from physical bytes, never an
        // admission policy; each actual row and bank is claimed on demand.
        let minimum_request_bytes = checkpoint_history_row_bytes
            .checked_add(recurrent_bank_bytes)
            .ok_or("minimum request byte count overflow")?
            .max(1);
        let possible_requests = usize::try_from(
            capacity_bytes.domain_bytes / minimum_request_bytes,
        )
        .unwrap_or(usize::MAX)
        .max(1);
        let bank_capacity = BankCapacity {
            active: possible_requests,
            in_flight: possible_requests
                .checked_mul(limits.target_launches())
                .ok_or("in-flight bank count overflow")?,
            retained: possible_requests
                .checked_add(retention_entries)
                .ok_or("retained numerical capacity overflow")?,
        };
        let capacity = StateCapacityPlan {
            live_requests: possible_requests,
            checkpoints: possible_requests,
            retention_entries,
            // The history reservation (graphs are sealed over it): a context
            // for every bank owner, but never more rows than the device's
            // stable domain capacity could ever back.
            history_rows: history_bound.max(context),
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
        // Seismic's fixed address tables and first history and bank slabs are
        // committed at store construction. Further slabs use heap claims.
        let history = target_state
            .startup_history_bytes()?
            .checked_add(
                head_state
                    .as_ref()
                    .map(|state| state.startup_history_bytes())
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or("startup history slab charge overflows")?;
        let planned_recurrent = target_state
            .startup_bank_bytes()?
            .checked_add(
                head_state
                    .as_ref()
                    .map(|state| state.startup_bank_bytes())
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or("startup bank slab charge overflows")?;
        Ok(StateResourcePlan {
            definition: definition.clone(),
            load: load.clone(),
            codec,
            limits,
            capacity_bytes,
            capacity,
            target_state,
            head_state,
            retained_entry_bytes,
            history_bytes: history,
            recurrent_banks_bytes: planned_recurrent,
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
        let capacity_bytes = state.capacity_bytes;
        let qualification_peak = crate::AttestedPrograms::qualification_peak_bytes(load)?;
        let source_import_peak = load
            .target
            .iter()
            .chain(load.head.iter().flatten())
            .chain(load.vision.iter().flatten())
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        let source_import_peak = source_import_peak_bytes(source_import_peak)?;
        let qualification_peak_bytes = qualification_peak.max(source_import_peak);
        let slots = limits.startup_slots();
        let target_graph = NativeGraphCharge::from_prepared(target_graphs, slots.target)?;
        let target_readout_graph = NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(target_readout_graphs.family()),
            slots.readout,
        )?;
        let head_graph = head_graphs
            .map(|graphs| {
                NativeGraphCharge::from_footprint(
                    FamilyFootprint::serial(graphs.family()),
                    slots.head,
                )
            })
            .transpose()?;
        let vision_graph = vision_graphs
            .map(|graphs| {
                NativeGraphCharge::from_footprint(
                    FamilyFootprint::serial(graphs.family()),
                    slots.vision,
                )
            })
            .transpose()?;
        let state_graph = NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(state_graphs.family()),
            slots.state,
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
                .program_plan(definition, state.codec)
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
            prepared_programs,
            scratch,
        };
        let required = bytes.total()?;
        if std::env::var_os("MAGNITUDE_TRACE_RESOURCES").is_some() {
            eprintln!(
                "resource plan domain_capacity={} required={} qualification_peak={} weights=[{},{},{}] history={} recurrent_banks={} prepared_programs={} scratch={} graph=[target:{},readout:{},head:{},vision:{},state:{}]",
                capacity_bytes.domain_bytes,
                required,
                qualification_peak_bytes,
                bytes.target_weights,
                bytes.head_weights,
                bytes.vision_weights,
                bytes.history,
                bytes.recurrent_banks,
                bytes.prepared_programs,
                bytes.scratch,
                target_graph.committed_bytes,
                target_readout_graph.committed_bytes,
                head_graph.map_or(0, |graph| graph.committed_bytes),
                vision_graph.map_or(0, |graph| graph.committed_bytes),
                state_graph.committed_bytes,
            );
        }
        if required > capacity_bytes.domain_bytes {
            return Err(format!(
                "resource plan requires {required} bytes but the device domain has {}",
                capacity_bytes.domain_bytes
            ));
        }
        ResourcePlan {
            domain_capacity_bytes: capacity_bytes.domain_bytes,
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
    let recurrent_bank_bytes = recurrent_bank_bytes(&recurrent_components)?;
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

fn slab_element(dtype: DType) -> Element {
    match dtype {
        DType::F32 => Element::f32(),
        DType::F16 => Element::f16(),
        DType::BF16 => Element::bf16(),
        DType::I32 => Element::i32(),
        DType::U32 => Element::u32(),
        DType::Bool => Element::bool(),
    }
}

#[cfg(test)]
mod slab_plan_tests {
    use super::*;
    use magnitude_state::LayerRef;
    use seismic::{BackendName, DeviceCatalog};

    #[test]
    fn startup_and_fit_charges_match_slab_storage() {
        let history = ComponentDescriptor::new(
            LayerRef::Target(0),
            KvCodec::Dense.spec(DType::BF16, 32, 32),
            1,
        )
        .unwrap();
        let plan = state_store_plan(
            600_000,
            1_200_000,
            vec![history],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 2,
                in_flight: 2,
                retained: 1,
            },
        )
        .unwrap();
        let history = plan.history_slab_layout().unwrap().unwrap();
        let banks = plan.bank_slab_layout().unwrap().unwrap();
        assert_eq!(plan.bank_slab_banks().unwrap(), 4);
        assert_eq!(
            plan.initial_committed_bytes().unwrap(),
            history.address_table_bytes
                + history.slab_bytes
                + banks.address_table_bytes
                + banks.slab_bytes
        );
        let device = Rc::new(
            DeviceCatalog::discover()
                .unwrap()
                .open_backend(BackendName::Cpu)
                .unwrap(),
        );
        let store = plan.allocate(device).unwrap();
        assert_eq!(store.committed_bytes(), plan.initial_committed_bytes().unwrap());

        let history_depth = u64::from(plan.history_slab_rows().unwrap()) + 1;
        assert_eq!(
            plan.history_bytes_at_depth(0).unwrap(),
            history.address_table_bytes + history.slab_bytes
        );
        assert_eq!(
            plan.history_bytes_at_depth(history_depth).unwrap(),
            history.address_table_bytes + 2 * history.slab_bytes
        );
        assert_eq!(
            plan.bank_bytes_at_count(5).unwrap(),
            banks.address_table_bytes + 2 * banks.slab_bytes
        );
    }
}

pub(super) fn history_row_bytes(
    components: &[magnitude_state::ComponentDescriptor],
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

pub(super) fn recurrent_bank_bytes(components: &[ComponentSpec]) -> Result<u64, String> {
    components.iter().try_fold(0u64, |total, component| {
        let bytes = u64::try_from(component.bytes()?)
            .map_err(|_| "recurrent component bytes exceed u64")?;
        total
            .checked_add(bytes)
            .ok_or_else(|| "recurrent bank byte count overflow".into())
    })
}
