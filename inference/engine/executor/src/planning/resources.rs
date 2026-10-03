use super::{source_import_peak_bytes, weight_bytes_by_component, ModelLoadPlan, PlannedMethod};
use crate::{
    PlanError, CapacityError, ResourceKind, PreparedDrafterGraphs, PreparedStateCopyGraphs,
    PreparedTargetGraphs, PreparedTargetReadoutGraphs, PreparedVisionGraphs,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{
    BankCapacity, ComponentDescriptor, ComponentSpec, HistoryDomainId, HistoryDomainKind,
    HistoryDomainLayout, HistoryDomainPlan, HistoryDomainTrace, KvCodec, LayerRef,
    ModelStateLayout, StateStore, StoreBindings,
};
use seismic::{DType, Device, Element, SlabLayout, SlabRegion};
use std::rc::Rc;

/// The service's bounds a load plans for. None is a request-count batch
/// width: a launch is bounded by its token budget (rows and request slots
/// are compiled shape classes), its vocabulary work by the selection bound,
/// and every per-request resource beyond what one request needs grows
/// elastically under heap claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Rows one launch admits: the larger step token budget.
    pub max_launch_rows: usize,
    /// Requests one launch serves: its largest request-slot class.
    pub max_launch_slots: usize,
    /// Target rows one launch selects a token for. The selection bound: every
    /// vocabulary-wide buffer of the target readout is sized by it.
    pub max_selected_rows: usize,
    /// Requests one drafter launch drafts for: the selection bound of the
    /// head and separate draft.
    pub max_drafting_slots: usize,
    /// Rows one launch may export full logits for. Zero for a served load,
    /// which prepares no logits class; diagnostics that read logits set it.
    pub exported_logits_rows: usize,
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
    /// run. Activations cover the launches in flight at once, whatever the
    /// request count; each holds its own upload regions, and every one binds
    /// the device's one workspace arena. The target's outputs are its
    /// residual pair: each block graph reads one and writes the other, and
    /// the device's submission order lets every launch reuse the same pair.
    /// Outputs a request retains after its launch completes (readout
    /// features, head drafts, encoded images) start at one request's need
    /// and grow one slot at a time under a heap claim.
    pub fn startup_slots(&self) -> StartupSlots {
        let launches = self.target_launches();
        StartupSlots {
            target: GraphSlots {
                activations: launches,
                output: 2,
            },
            readout: GraphSlots {
                activations: launches,
                output: 1 + launches,
            },
            // A prompt chunk's drafter entry rides with each target launch
            // in flight.
            head: GraphSlots {
                activations: launches,
                output: 1 + launches,
            },
            // Vision holds nothing until an image arrives: a text-only
            // session never uses it. Its workspace is the shared arena.
            vision: GraphSlots {
                activations: 0,
                output: 0,
            },
            state: GraphSlots {
                activations: 1,
                output: 1,
            },
        }
    }
}

/// One graph family's slot counts: concurrent activations, each with its
/// upload regions, and output arenas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphSlots {
    pub activations: usize,
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
    /// Preallocated device-local activation transport, never model weights.
    pub activation_transfer: u64,
}

/// A Seismic-derived physical family charge at startup. The engine chooses
/// the slot counts; Seismic supplies every byte and every tensor layout
/// within each slot. Each activation holds `upload_regions` upload regions
/// of `upload_bytes`, allocated with it: one per graph run its lease keeps
/// in flight at once. Output slots beyond `output_slots` are elastic
/// run-time claims of `output_bytes` each. `workspace_bytes` is not part of
/// `committed_bytes`: every family binds the device's one workspace arena,
/// which the plan charges once at the largest family's workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeGraphCharge {
    pub workspace_bytes: u64,
    pub output_bytes: u64,
    pub upload_bytes: u64,
    pub upload_regions: usize,
    pub activations: usize,
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
            activations,
            output: output_slots,
        } = slots;
        let count = |value: usize| u64::try_from(value).map_err(|_| "slot count exceeds u64");
        let per_activation = footprint
            .upload_bytes
            .checked_mul(count(footprint.runs_in_flight)?)
            .ok_or("graph activation byte count overflows")?;
        let committed_bytes = per_activation
            .checked_mul(count(activations)?)
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
            activations,
            output_slots,
            committed_bytes,
        })
    }
}

/// The device's one workspace arena: the largest workspace of `charges`.
pub(crate) fn arena_bytes(charges: &[&NativeGraphCharge]) -> u64 {
    charges
        .iter()
        .map(|charge| charge.workspace_bytes)
        .max()
        .unwrap_or(0)
}

/// Every graph byte a load commits: the workspace arena, and each family's
/// activations and outputs.
pub(crate) fn graph_scratch_bytes(charges: &[&NativeGraphCharge]) -> Result<u64, String> {
    charges
        .iter()
        .try_fold(arena_bytes(charges), |bytes, charge| {
            bytes.checked_add(charge.committed_bytes)
        })
        .ok_or_else(|| "graph scratch byte count overflows".into())
}

/// Exact projection used to construct one numerical state arena. The planner
/// owns every capacity and byte fact; construction only materializes it.
/// `history` is the store's history domains as `StateStore::new` takes them;
/// `domains` projects each stored domain, indexed by `HistoryDomainId`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateStorePlan {
    pub context_rows: usize,
    /// The largest advance (slot rows): Window domains' row limit.
    pub max_advance: usize,
    pub history: Vec<HistoryDomainPlan>,
    pub domains: Vec<HistoryStorePlan>,
    pub recurrent_components: Vec<ComponentSpec>,
    pub bank_capacity: BankCapacity,
    pub recurrent_bank_bytes: u64,
    pub zero_seed_bytes: u64,
    pub recurrent_pool_bytes: u64,
}

/// One stored history domain of a store plan: exactly what the store derives
/// for it (see `StateStore::allocation_trace`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStorePlan {
    pub kind: HistoryDomainKind,
    pub components: Vec<ComponentDescriptor>,
    /// Reserved row addresses, sealed into graphs.
    pub rows: usize,
    pub row_bytes: u64,
    pub slab_rows: u32,
    pub page_rows: u32,
    pub span_limit: usize,
}

/// The history an attention layer binds: its stored domain, and the
/// component whose regions it reads (its own, or a Shared domain's source).
#[derive(Clone, Copy, Debug)]
pub struct LayerHistory<'a> {
    pub domain: HistoryDomainId,
    pub store: &'a HistoryStorePlan,
    pub component: &'a ComponentDescriptor,
}

impl HistoryStorePlan {
    fn trace(&self) -> HistoryDomainTrace {
        HistoryDomainTrace {
            kind: self.kind,
            capacity: self.rows,
            row_bytes: self.row_bytes,
            bytes: self.row_bytes * self.rows as u64,
            slab_rows: self.slab_rows as usize,
            page_rows: self.page_rows as usize,
            span_limit: self.span_limit,
        }
    }

    pub fn slab_layout(&self) -> Result<SlabLayout, String> {
        let regions = self
            .components
            .iter()
            .flat_map(ComponentDescriptor::planes)
            .map(|plane| SlabRegion {
                element: slab_element(plane.dtype),
                row_shape: plane.row_extents.iter().map(|&size| size as u64).collect(),
            })
            .collect::<Vec<_>>();
        SlabLayout::for_regions(u64::from(self.slab_rows), self.rows as u64, &regions)
            .map_err(|error| error.to_string())
    }
}

impl StateStorePlan {
    /// The most spans any history of the store presents to one launch (a
    /// Shared reader included, see `magnitude_state::history_geometry`): the
    /// span class bound graphs are sealed to and launches are checked
    /// against (1 without history).
    pub fn span_limit(&self) -> usize {
        self.domains
            .iter()
            .map(|domain| domain.span_limit)
            .max()
            .unwrap_or(1)
    }

    /// The history `layer` binds, resolving a Shared layer to its source.
    /// `None` for a layer without history in this store.
    pub fn layer_history(&self, layer: LayerRef) -> Option<LayerHistory<'_>> {
        let owner = self
            .history
            .iter()
            .find_map(|plan| match &plan.layout {
                HistoryDomainLayout::Shared { source, layers } if layers.contains(&layer) => {
                    Some(*source)
                }
                _ => None,
            })
            .unwrap_or(layer);
        self.domains.iter().enumerate().find_map(|(index, store)| {
            store
                .components
                .iter()
                .find(|component| component.layer == owner)
                .map(|component| LayerHistory {
                    domain: HistoryDomainId(index),
                    store,
                    component,
                })
        })
    }

    /// The store's one history domain, for a store whose history the
    /// caller addresses as one domain (the draft head's).
    pub fn sole_history(&self) -> Result<&HistoryStorePlan, String> {
        match self.domains.as_slice() {
            [domain] => Ok(domain),
            domains => Err(format!(
                "store has {} history domains where one is required",
                domains.len()
            )),
        }
    }

    pub fn bank_slab_banks(&self) -> Result<u32, String> {
        u32::try_from(magnitude_state::banks_per_slab(self.recurrent_bank_bytes)?)
            .map_err(|_| "bank slab count exceeds u32".into())
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

    /// Every stored domain's address table and first slab.
    pub fn startup_history_bytes(&self) -> Result<u64, String> {
        self.domains.iter().try_fold(0u64, |total, domain| {
            let layout = domain.slab_layout()?;
            layout
                .address_table_bytes
                .checked_add(layout.slab_bytes)
                .and_then(|bytes| total.checked_add(bytes))
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

    /// One request's history at `depth` tokens: a Token domain holds `depth`
    /// rows, a Window domain its steady footprint of `n` plus one advance,
    /// independent of the depth; each rounded to whole slabs of its domain.
    pub fn history_bytes_at_depth(&self, depth: u64) -> Result<u64, String> {
        self.domains.iter().try_fold(0u64, |total, domain| {
            let layout = domain.slab_layout()?;
            let rows = match domain.kind {
                HistoryDomainKind::Token => depth,
                kind => u64::try_from(
                    kind.row_limit(
                        usize::try_from(depth).map_err(|_| "fit depth exceeds host range")?,
                        self.max_advance,
                    )
                    .map_err(|error| error.to_string())?,
                )
                .map_err(|_| "window rows exceed u64")?,
            };
            // Whole pages: a history's first page may be partial once a
            // window releases the rows before it, its last page always may.
            let page = u64::from(domain.page_rows);
            let front = u64::from(matches!(domain.kind, HistoryDomainKind::Window { .. }));
            let pages = rows.max(1).div_ceil(page) + front;
            let slabs = (pages * page).div_ceil(u64::from(domain.slab_rows));
            layout
                .slab_bytes
                .checked_mul(slabs)
                .and_then(|bytes| bytes.checked_add(layout.address_table_bytes))
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| "history slab fit charge overflows".into())
        })
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

    /// The planned store and its one binding right.
    pub fn allocate(&self, device: Rc<Device>) -> Result<StoreBindings, String> {
        let (store, bindings) = StateStore::new(
            device,
            self.context_rows,
            self.max_advance,
            self.history.clone(),
            self.recurrent_components.clone(),
            self.bank_capacity,
        )
        .map_err(|error| error.to_string())?;
        let trace = store
            .allocation_trace()
            .map_err(|error| error.to_string())?;
        if trace.context_capacity != self.context_rows
            || trace.history
                != self
                    .domains
                    .iter()
                    .map(HistoryStorePlan::trace)
                    .collect::<Vec<_>>()
            || trace.bank_capacity != self.bank_capacity
            || trace.recurrent_bank_bytes != self.recurrent_bank_bytes
            || trace.zero_seed_bytes != self.zero_seed_bytes
            || trace.recurrent_pool_bytes != self.recurrent_pool_bytes
            || store.committed_bytes() != self.initial_committed_bytes()?
        {
            return Err("state allocation differs from its resource-plan projection".into());
        }
        Ok(bindings)
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
            self.activation_transfer,
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

    /// Every family's charge.
    fn graph_charges(&self) -> Vec<&NativeGraphCharge> {
        [
            Some(&self.target_graph),
            Some(&self.target_readout_graph),
            self.head_graph.as_ref(),
            self.vision_graph.as_ref(),
            Some(&self.state_graph),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// The device's one workspace arena, which every family binds.
    pub fn arena_bytes(&self) -> u64 {
        arena_bytes(&self.graph_charges())
    }

    pub(super) fn validate(mut self) -> Result<Self, PlanError> {
        let planning = PlanError::ResourcePlanning;
        if self.bytes.scratch != graph_scratch_bytes(&self.graph_charges()).map_err(planning)? {
            return Err(planning(
                "resource plan pooled byte charge differs from admitted Seismic footprints".into(),
            ));
        }
        self.steady_committed_bytes = self.bytes.total().map_err(planning)?;
        self.startup_peak_bytes = self
            .steady_committed_bytes
            .checked_add(self.qualification_peak_bytes)
            .ok_or_else(|| planning("startup peak byte count overflow".into()))?;
        if self.startup_peak_bytes > self.domain_capacity_bytes {
            return Err(PlanError::Resource(CapacityError {
                resource: ResourceKind::DeviceMemory,
                required: self.startup_peak_bytes,
                available: self.domain_capacity_bytes,
            }));
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

    pub fn allocate_target_state(&self, device: Rc<Device>) -> Result<StoreBindings, String> {
        self.target_state.allocate(device)
    }

    pub fn allocate_head_state(
        &self,
        device: Rc<Device>,
    ) -> Result<Option<StoreBindings>, String> {
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
    target_layer_range: std::ops::Range<usize>,
    retained_entry_bytes: u64,
    history_bytes: u64,
    recurrent_banks_bytes: u64,
}

impl StateResourcePlan {
    pub(crate) fn matches_stage(
        &self,
        view: &magnitude_state::StageModelView<'_>,
        load: &ModelLoadPlan,
    ) -> bool {
        self.definition.decoder == *view.decoder()
            && self.target_layer_range == view.global_range()
            && self.load == *load
    }

    #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
    pub(crate) fn is_for_stage(
        &self,
        definition: &ModelDefinition,
        range: std::ops::Range<usize>,
        load: &ModelLoadPlan,
    ) -> bool {
        self.definition == *definition && self.target_layer_range == range && self.load == *load
    }

    /// Original decoder range; component offsets remain local to this plan.
    pub fn target_layer_range(&self) -> std::ops::Range<usize> {
        self.target_layer_range.clone()
    }

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

    /// The state slabs a load commits at startup, before any request.
    pub fn startup_state_bytes(&self) -> Result<u64, String> {
        self.history_bytes
            .checked_add(self.recurrent_banks_bytes)
            .ok_or_else(|| "startup state bytes overflow".into())
    }

    pub fn target_state(&self) -> &StateStorePlan {
        &self.target_state
    }

    pub fn head_state(&self) -> Option<&StateStorePlan> {
        self.head_state.as_ref()
    }
}

#[cfg(any(test, feature = "experimental-pipeline-cuda"))]
pub(crate) struct StageResourceCharges {
    pub target: NativeGraphCharge,
    pub readout: NativeGraphCharge,
    pub state: NativeGraphCharge,
    pub constant_bytes: u64,
    pub program_bytes: u64,
    pub activation_bytes: u64,
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
        Self::state_plan_with_stage(
            definition,
            load,
            method,
            codec,
            limits,
            capacity_bytes,
            None,
        )
    }

    /// Local storage geometry for an explicitly selected original block range.
    /// This neither chooses a partition nor authorizes aggregate-memory fit.
    #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
    pub fn stage_state_plan(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        limits: ResourceLimits,
        capacity_bytes: ResourceCapacity,
        range: std::ops::Range<usize>,
    ) -> Result<StateResourcePlan, String> {
        let view = magnitude_state::StageModelView::new(&definition.decoder, range)?;
        if load.head.is_some()
            || load.vision.is_some()
            || definition.draft.is_some()
            || limits.lookahead
        {
            return Err("pipeline state excludes drafting, vision and lookahead".into());
        }
        Self::state_plan_with_stage(
            definition,
            load,
            PlannedMethod::Plain,
            codec,
            limits,
            capacity_bytes,
            Some(&view),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn state_plan_with_stage(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
        capacity_bytes: ResourceCapacity,
        stage: Option<&magnitude_state::StageModelView<'_>>,
    ) -> Result<StateResourcePlan, String> {
        if limits.max_launch_slots == 0
            || limits.max_launch_slots > limits.max_launch_rows
            || limits.max_launch_rows == 0
            || limits.max_selected_rows == 0
            || limits.max_selected_rows > limits.max_launch_rows
            || limits.max_drafting_slots == 0
            || limits.max_drafting_slots > limits.max_launch_slots
            || limits.exported_logits_rows > limits.max_launch_rows
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
        let drafter = if load.head.is_some() {
            crate::operators::draft::drafter_blocks(definition)
        } else {
            Vec::new()
        };
        let layout = match stage {
            Some(view) => ModelStateLayout::derive_stage(view, codec, method.draft_rows())?,
            None => {
                ModelStateLayout::derive(&definition.decoder, &drafter, codec, method.draft_rows())?
            }
        };
        let history_domains = || layout.target_history.iter().chain(&layout.head_history);
        let checkpoint_history_row_bytes = history_row_bytes(&layout.target_history)?
            .checked_add(history_row_bytes(&layout.head_history)?)
            .ok_or("history row byte count overflow")?;
        let recurrent_bank_bytes = recurrent_bank_bytes(&layout.target_recurrent)?;
        let context = usize::try_from(definition.decoder.context_limit)
            .map_err(|_| "context limit exceeds host domain")?;
        let max_advance = limits.max_launch_rows;
        // A checkpoint's window rows: `n` per Window(n) domain.
        let checkpoint_window_bytes = history_domains()
            .filter(|domain| matches!(domain.kind(), HistoryDomainKind::Window { .. }))
            .try_fold(0u64, |total, domain| {
                let rows = domain
                    .kind()
                    .checkpoint_rows(context)
                    .map_err(|error| error.to_string())?;
                domain
                    .row_bytes()
                    .map_err(|error| error.to_string())?
                    .checked_mul(rows as u64)
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or_else(|| String::from("window checkpoint byte count overflow"))
            })?;
        // Generation methods keep their carried feature rows on the host, so
        // a retained entry charges only numerical state. Entries on one path
        // share their history rows, so an entry's marginal state is its recurrent
        // bank; without recurrent state it is a full context of history.
        let retained_entry_bytes = if recurrent_bank_bytes != 0 {
            recurrent_bank_bytes
        } else {
            checkpoint_history_row_bytes
                .checked_mul(definition.decoder.context_limit)
                .and_then(|bytes| bytes.checked_add(checkpoint_window_bytes))
                .ok_or("checkpoint state byte count overflow")?
        };
        // One row in every stored domain.
        let request_row_bytes = history_domains().try_fold(0u64, |total, domain| {
            total
                .checked_add(domain.row_bytes().map_err(|error| error.to_string())?)
                .ok_or_else(|| String::from("history row byte count overflow"))
        })?;
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
        let minimum_request_bytes = request_row_bytes
            .checked_add(recurrent_bank_bytes)
            .ok_or("minimum request byte count overflow")?
            .max(1);
        let possible_requests =
            usize::try_from(capacity_bytes.domain_bytes / minimum_request_bytes)
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
        let rows = HistoryRows {
            context,
            max_advance,
            token: capacity.history_rows,
            domain_bytes: capacity_bytes.domain_bytes,
        };
        let target_state = state_store_plan(
            rows,
            layout.target_history,
            layout.target_recurrent,
            bank_capacity,
        )?;
        let head_state = (!layout.head_history.is_empty())
            .then(|| state_store_plan(rows, layout.head_history, Vec::new(), bank_capacity))
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
            target_layer_range: stage.map_or(0..definition.decoder.blocks.len(), |view| {
                view.global_range()
            }),
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

    /// Admit an explicitly assigned stage against its own capacity. Original
    /// roles remain in the load/program topology; only assigned physical storage
    /// appears in this local plan. This neither searches nor pools capacity.
    #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
    pub(crate) fn stage_plan_with_state(
        state: StateResourcePlan,
        assignment: &crate::pipeline::StageAssignment,
        charges: StageResourceCharges,
    ) -> Result<ResourcePlan, PlanError> {
        Self::derive_stage_plan(state, assignment, charges)
            .map_err(PlanError::ResourcePlanning)?
            .validate()
    }

    #[cfg(any(test, feature = "experimental-pipeline-cuda"))]
    fn derive_stage_plan(
        state: StateResourcePlan,
        assignment: &crate::pipeline::StageAssignment,
        charges: StageResourceCharges,
    ) -> Result<ResourcePlan, String> {
        if !state.is_for_stage(
            assignment.definition(),
            assignment.view().global_range(),
            &state.load,
        ) {
            return Err("stage resource plan differs from original model projection".into());
        }
        let assigned = assignment.weights(&state.load).map_err(|e| e.to_string())?;
        let target_weights = assignment
            .resident_bytes(&state.load)
            .map_err(|e| e.to_string())?;
        let qualification_peak_bytes = source_import_peak_bytes(
            assigned
                .iter()
                .map(|weight| weight.source_bytes)
                .max()
                .unwrap_or(0),
        )?;
        let graph_charges = [&charges.target, &charges.readout, &charges.state];
        let bytes = ResourceBytes {
            target_weights,
            head_weights: 0,
            vision_weights: 0,
            history: state.history_bytes,
            recurrent_banks: state.recurrent_banks_bytes,
            prepared_programs: charges
                .program_bytes
                .checked_add(charges.constant_bytes)
                .ok_or("local prepared byte count overflows")?,
            scratch: graph_scratch_bytes(&graph_charges)?,
            activation_transfer: charges.activation_bytes,
        };
        Ok(ResourcePlan {
            domain_capacity_bytes: state.capacity_bytes.domain_bytes,
            capacity: state.capacity,
            target_state: state.target_state,
            head_state: None,
            retained_entry_bytes: state.retained_entry_bytes,
            bytes,
            qualification_peak_bytes,
            target_graph: charges.target,
            target_readout_graph: charges.readout,
            head_graph: None,
            vision_graph: None,
            state_graph: charges.state,
            steady_committed_bytes: 0,
            startup_peak_bytes: 0,
        })
    }

    pub fn plan_with_state(
        state: StateResourcePlan,
        target_graphs: &PreparedTargetGraphs,
        target_readout_graphs: &PreparedTargetReadoutGraphs,
        head_graphs: Option<&PreparedDrafterGraphs>,
        vision_graphs: Option<&PreparedVisionGraphs>,
        state_graphs: &PreparedStateCopyGraphs,
    ) -> Result<ResourcePlan, PlanError> {
        Self::derive_plan_with_state(
            state,
            target_graphs,
            target_readout_graphs,
            head_graphs,
            vision_graphs,
            state_graphs,
        )
        .map_err(PlanError::ResourcePlanning)?
        .validate()
    }

    fn derive_plan_with_state(
        state: StateResourcePlan,
        target_graphs: &PreparedTargetGraphs,
        target_readout_graphs: &PreparedTargetReadoutGraphs,
        head_graphs: Option<&PreparedDrafterGraphs>,
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
        let charges = [
            Some(&target_graph),
            Some(&target_readout_graph),
            head_graph.as_ref(),
            vision_graph.as_ref(),
            Some(&state_graph),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let scratch = graph_scratch_bytes(&charges)?;
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
            activation_transfer: 0,
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
            // Each family's placed workspace against its liveness floor: a
            // placement regression shows as placed above floor.
            let workspace = |family: &seismic::NativeGraphFamily| {
                format!(
                    "{}/{}",
                    family.workspace_bytes(),
                    family.workspace_floor_bytes()
                )
            };
            eprintln!(
                "resource plan arena={} workspace placed/floor=[target:{},readout:{},head:{},vision:{},state:{}]",
                arena_bytes(&charges),
                workspace(target_graphs.family()),
                workspace(target_readout_graphs.family()),
                head_graphs.map_or_else(|| "-".into(), |graphs| workspace(graphs.family())),
                vision_graphs.map_or_else(|| "-".into(), |graphs| workspace(graphs.family())),
                workspace(state_graphs.family()),
            );
        }
        Ok(ResourcePlan {
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
        })
    }
}

/// The row reservations of a store's history domains.
#[derive(Clone, Copy)]
struct HistoryRows {
    context: usize,
    max_advance: usize,
    /// Every Token domain's reservation.
    token: usize,
    /// A Window domain reserves what the device domain could back, and at
    /// least its row limit.
    domain_bytes: u64,
}

impl HistoryRows {
    fn rows(self, layout: &HistoryDomainLayout) -> Result<usize, String> {
        let kind = layout.kind();
        Ok(match kind {
            HistoryDomainKind::Token => self.token,
            HistoryDomainKind::Window { .. } => {
                let row_bytes = layout.row_bytes().map_err(|error| error.to_string())?;
                usize::try_from(self.domain_bytes / row_bytes.max(1))
                    .unwrap_or(usize::MAX)
                    .max(
                        kind.row_limit(self.context, self.max_advance)
                            .map_err(|error| error.to_string())?,
                    )
            }
            HistoryDomainKind::Shared { .. } => 0,
            HistoryDomainKind::Block { .. } => {
                return Err(format!("history domain {kind:?} is not supported"))
            }
        })
    }
}

fn state_store_plan(
    rows: HistoryRows,
    history: Vec<HistoryDomainLayout>,
    recurrent_components: Vec<ComponentSpec>,
    bank_capacity: BankCapacity,
) -> Result<StateStorePlan, String> {
    let layouts = history;
    // The rows the store reserves for each stored domain (whole pages), which
    // graphs are sealed over.
    let history = layouts
        .iter()
        .map(|layout| {
            let logical_rows = match layout {
                HistoryDomainLayout::Shared { .. } => 0,
                _ => layout
                    .geometry(&layouts, rows.context, rows.max_advance)
                    .map_err(|error| error.to_string())?
                    .reserved_rows(rows.rows(layout)?)
                    .ok_or("history rows overflow whole pages")?,
            };
            Ok(HistoryDomainPlan {
                logical_rows,
                layout: layout.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let domains = history
        .iter()
        .filter(|plan| !matches!(plan.layout, HistoryDomainLayout::Shared { .. }))
        .map(|plan| {
            let row_bytes = plan.layout.row_bytes().map_err(|error| error.to_string())?;
            let geometry = plan
                .layout
                .geometry(&layouts, rows.context, rows.max_advance)
                .map_err(|error| error.to_string())?;
            Ok(HistoryStorePlan {
                kind: plan.layout.kind(),
                components: plan.layout.components().to_vec(),
                rows: plan.logical_rows,
                row_bytes,
                slab_rows: u32::try_from(geometry.slab_rows)
                    .map_err(|_| "history slab rows exceed u32")?,
                page_rows: u32::try_from(geometry.page_rows)
                    .map_err(|_| "history page rows exceed u32")?,
                span_limit: geometry.span_limit,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
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
        context_rows: rows.context,
        max_advance: rows.max_advance,
        history,
        domains,
        recurrent_components,
        bank_capacity,
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
            HistoryRows {
                context: 600_000,
                max_advance: 512,
                token: 1_200_000,
                domain_bytes: 1 << 30,
            },
            vec![HistoryDomainLayout::Token {
                components: vec![history],
            }],
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
        let history = plan.domains[0].slab_layout().unwrap();
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
        assert_eq!(
            store.committed_bytes(),
            plan.initial_committed_bytes().unwrap()
        );

        let history_depth = u64::from(plan.domains[0].slab_rows) + 1;
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

/// History bytes per token of context: the rows of every Token domain.
/// Window domains hold a bounded number of rows whatever the context (see
/// `StateStorePlan::history_bytes_at_depth`).
pub(super) fn history_row_bytes(domains: &[HistoryDomainLayout]) -> Result<u64, String> {
    domains
        .iter()
        .filter(|domain| domain.kind() == HistoryDomainKind::Token)
        .try_fold(0u64, |total, domain| {
            total
                .checked_add(domain.row_bytes().map_err(|error| error.to_string())?)
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

#[cfg(test)]
mod stage_admission_tests {
    use super::*;
    #[test]
    fn explicit_stage_admission_counts_only_owned_storage_and_checks_local_peak() {
        let configuration = crate::assessment::fixtures::QWEN35_CONFIGURATIONS[0];
        let (definition, manifest) = crate::assessment::fixtures::declared_model(&configuration);
        let definition = Rc::new(definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            super::super::ComponentSelection {
                head: false,
                vision: false,
            },
            crate::resident_layout(crate::ExecutionPath::Native, seismic::BackendName::Cuda),
        )
        .unwrap();
        let model = crate::pipeline::PipelineModel::new(
            definition.clone(),
            vec![0..1, 1..definition.decoder.blocks.len()],
        )
        .unwrap();
        let limits = ResourceLimits {
            max_launch_rows: 2,
            max_launch_slots: 1,
            max_selected_rows: 1,
            max_drafting_slots: 1,
            exported_logits_rows: 1,
            max_images_per_request: 1,
            lookahead: false,
        };
        let graph = NativeGraphCharge::from_checked(
            seismic::NativeGraphStorageBytes {
                workspace: 100,
                output: 20,
                upload: 7,
            },
            1,
            GraphSlots {
                activations: 1,
                output: 2,
            },
        )
        .unwrap();
        let charges = || StageResourceCharges {
            target: graph,
            readout: graph,
            state: graph,
            constant_bytes: 11,
            program_bytes: 13,
            activation_bytes: 32,
        };
        for assignment in model.stages() {
            let state = ResourcePlanner::stage_state_plan(
                &definition,
                &load,
                KvCodec::Dense,
                limits,
                ResourceCapacity {
                    domain_bytes: 64 * 1024 * 1024 * 1024,
                },
                assignment.view().global_range(),
            )
            .unwrap();
            let plan =
                ResourcePlanner::stage_plan_with_state(state.clone(), &assignment, charges())
                    .unwrap();
            assert_eq!(
                plan.bytes.target_weights,
                assignment.resident_bytes(&load).unwrap()
            );
            assert_eq!(plan.bytes.prepared_programs, 24);
            assert_eq!(plan.bytes.activation_transfer, 32);
            assert_eq!(plan.bytes.scratch, 100 + 3 * graph.committed_bytes);
            assert_eq!(plan.steady_committed_bytes(), plan.bytes.total().unwrap());
            assert!(plan.startup_peak_bytes() > plan.steady_committed_bytes());
            let mut too_small = state;
            too_small.capacity_bytes.domain_bytes = plan.startup_peak_bytes() - 1;
            assert_eq!(
                ResourcePlanner::stage_plan_with_state(too_small, &assignment, charges())
                    .err()
                    .unwrap(),
                PlanError::Resource(CapacityError {
                    resource: ResourceKind::DeviceMemory,
                    required: plan.startup_peak_bytes(),
                    available: plan.startup_peak_bytes() - 1,
                })
            );
        }
    }
}
