use super::{
    GroupRows,
    block::{BlockGraphInputs, ClassNumber, certify_block},
    classes::{
        BlockListings, BlockServed, FamilyOutline, Holder, Passes, Topology,
        bank_layer_by_block, block_class_numbers, block_class_slice, block_listings,
        block_listings_and_controls, block_routed, block_sealed_stages, block_served,
        block_stages, block_upload_regions, chained_once_runs, class_topology, decode_launch,
        once_class, once_graphs, segment_classes, slot_scores,
    },
    entry::{EntryTokens, certify_entry, notice_bytes, notice_rows, target_view},
    heap::{
        GraphCounts, PlannedHeap, PlannedSealedGraphs, SealedGraphsHeap, block_binding_storage,
        block_graph_storage,
    },
    prepared::{BlockGraphShape, OnceGraphShape},
};
#[cfg(debug_assertions)]
use super::classes::{
    BlockGraphEnumeration, BlockOutputScales, LaunchRuns, StagedClass, UnreadScales,
    enumerate_block_graphs,
};
use crate::{
    ModelLoadPlan, PassLaunches, ResourceLimits, StateResourcePlan, TargetProgramPlan,
    operators::{
        BlockKeeps, paired_block,
        per_layer::graph::{
            CheckedPerLayerEntryEntries, per_layer_entry, per_layer_entry_class_slice, split_entry,
        },
        routed::split::{OnceStage, RunStage, Stage},
    },
    programs::{
        graph::{
            GraphError, RowForm,
            draft::Certified,
            sizing::{BlockDecodeCensus, SizingOutline, TargetFamilySizingCensus},
        },
        native_constants::{CheckedGraphFamilyResources, CheckedGraphResources, GraphConstant},
    },
};
use magnitude_family_contracts::{Decoder, WeightKind, WeightRole, WeightScope};
use seismic::{
    BackendName, GraphCensus, HostHeap, LaunchGraphs, LaunchedGraph, NativeGraphClassSlice,
    NativeGraphLayout,
};
use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Default)]
pub(super) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const K: u64 = 0x517cc1b727220a95;

    #[inline]
    fn mix(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        let mut h = self.hash;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let word = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
            self.mix(word);
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let word = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
            self.mix(u64::from(word));
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            let word = u16::from_ne_bytes(bytes[..2].try_into().unwrap());
            self.mix(u64::from(word));
            bytes = &bytes[2..];
        }
        if let Some(&byte) = bytes.first() {
            self.mix(u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

pub(super) type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
pub(super) type FxHashSet<T> = HashSet<T, FxBuildHasher>;

#[inline]
pub(super) fn fx_map<K, V>() -> FxHashMap<K, V> {
    HashMap::with_hasher(FxBuildHasher)
}

#[inline]
pub(super) fn fx_set<T>() -> FxHashSet<T> {
    HashSet::with_hasher(FxBuildHasher)
}

/// The same decoder class ladder as `PreparedTargetGraphs::prepare`. The
/// family takes independent maxima because its workspace, output and upload
/// holdings may peak in different members.
#[cfg(test)]
pub(crate) fn checked_target_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
) -> Result<CheckedGraphResources, GraphError> {
    certify_target_family::<NativeGraphLayout>(
        backend,
        load,
        geometry,
        state,
        plan,
        limits,
        PassLaunches::whole_chain(limits),
    )
    .map(|certificate| certificate.resources)
}

/// The target family a load of `plan` prepares, as its assessment charges
/// it ([`CheckedTargetFamily`]), from the one composition the sealer
/// follows.
pub(crate) fn checked_target_family(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
    pass: PassLaunches,
) -> Result<CheckedTargetFamily, GraphError> {
    let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
    let layouts = FamilyLayouts::<SizingOutline>::of_entries(backend, load, geometry, plan, &outline)?;
    let cache = Arc::new(Mutex::new(layouts));
    let cache_handle: Arc<dyn Any + Send + Sync> = cache.clone();
    let mut guard = cache.lock().expect("target family layout cache mutex poisoned");
    guard.certify_checked(outline, load, geometry, state, plan, limits, pass, cache_handle)
}

/// [`checked_target_family`] of `plan`'s family and of the same target's
/// without its drafter's writes (`certify_target_and_undrafted`).
pub(crate) fn checked_target_and_undrafted(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
    pass: PassLaunches,
) -> Result<[CheckedTargetFamily; 2], GraphError> {
    let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
    let layouts = FamilyLayouts::<SizingOutline>::of_entries(backend, load, geometry, plan, &outline)?;
    let cache = Arc::new(Mutex::new(layouts));
    let cache_handle: Arc<dyn Any + Send + Sync> = cache.clone();
    let mut guard = cache.lock().expect("target family layout cache mutex poisoned");
    guard.certify_checked_pair(outline, load, geometry, state, plan, limits, pass, cache_handle)
}

/// What the load's assessment charges of the target family, from the one
/// composition the sealer follows ([`checked_target_family`]).
pub(crate) struct CheckedTargetFamily {
    /// The family's storage.
    pub resources: CheckedGraphResources,
    /// The graphs a load of it seals and the host heap they hold.
    pub sealed: PlannedSealedGraphs,
    /// The upload regions of a workspace slot
    /// (`PreparedTargetGraphs::upload_regions`).
    pub upload_regions: Vec<u64>,
    /// What its graphs are made of (`TargetFamilyCertificate::graphs`).
    pub graphs: LaunchGraphs,
    /// The bytes the own buffers of a prompt group hold beside the floor's
    /// (`GroupRows::group_bytes`), granted when a group opens: at the places
    /// each pass size merges, up to the plan's chain places.
    pub group_bytes: crate::planning::ByPass<u64>,
    /// Accumulated launch census and deduplicated kernel set recorded by
    /// `SizingDraft` during assessment (empty when certified via `NativeGraphLayout`).
    pub census: TargetFamilySizingCensus,
    /// Shared `Mutex<FamilyLayouts<SizingOutline>>` cache so subsequent
    /// certifications of the same target (resident form, smaller scratch
    /// windows, different pass sizes) reuse all invariant entry, per-layer,
    /// and block-shape layouts.
    cache: Arc<dyn Any + Send + Sync>,
}

impl CheckedTargetFamily {
    fn clone_checked(&self) -> Self {
        Self {
            resources: CheckedGraphResources {
                storage: self.resources.storage,
                binding_constant_bytes: self.resources.binding_constant_bytes,
                graphs: self.resources.graphs.clone(),
                host_heap: self.resources.host_heap,
            },
            sealed: self.sealed,
            upload_regions: self.upload_regions.clone(),
            graphs: self.graphs.clone(),
            group_bytes: self.group_bytes.clone(),
            census: self.census.clone(),
            cache: Arc::clone(&self.cache),
        }
    }

    /// Opaque handle to the shared target-family layout cache, passed back to
    /// [`Self::certify_with_cache`] across `bank`, `resident`, `WindowedBankForm`,
    /// and `at_pass` certifications.
    pub(crate) fn cache_handle(&self) -> Arc<dyn Any + Send + Sync> {
        Arc::clone(&self.cache)
    }

    /// Certify a target family reusing an existing layout cache from another
    /// holding/pass (or plain/drafted variant) of the same target (falling back
    /// to a fresh cache if the scope differs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn certify_with_cache(
        cache: &Arc<dyn Any + Send + Sync>,
        backend: BackendName,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        pass: PassLaunches,
    ) -> Result<CheckedTargetFamily, GraphError> {
        let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
        let mutex = cache
            .as_ref()
            .downcast_ref::<Mutex<FamilyLayouts<SizingOutline>>>()
            .expect("invariant violated: target cache handle is not FamilyLayouts<SizingOutline>");
        let mut guard = mutex.lock().expect("target family layout cache mutex poisoned");
        if guard.matches_scope(backend, &outline, plan) {
            guard.ensure_entries(backend, load, geometry, &outline)?;
            guard.certify_checked(outline, load, geometry, state, plan, limits, pass, Arc::clone(cache))
        } else {
            drop(guard);
            let layouts = FamilyLayouts::<SizingOutline>::of_entries(backend, load, geometry, plan, &outline)?;
            let fresh = Arc::new(Mutex::new(layouts));
            let fresh_handle: Arc<dyn Any + Send + Sync> = fresh.clone();
            let mut guard = fresh.lock().expect("target family layout cache mutex poisoned");
            guard.certify_checked(outline, load, geometry, state, plan, limits, pass, fresh_handle)
        }
    }

    /// Certify a target family and its undrafted counterpart reusing an
    /// existing layout cache from another holding/pass or plain variant of the
    /// same target (falling back to a fresh cache if the scope differs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn certify_and_undrafted_with_cache(
        cache: &Arc<dyn Any + Send + Sync>,
        backend: BackendName,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        pass: PassLaunches,
    ) -> Result<[CheckedTargetFamily; 2], GraphError> {
        let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
        let mutex = cache
            .as_ref()
            .downcast_ref::<Mutex<FamilyLayouts<SizingOutline>>>()
            .expect("invariant violated: target cache handle is not FamilyLayouts<SizingOutline>");
        let mut guard = mutex.lock().expect("target family layout cache mutex poisoned");
        if guard.matches_scope(backend, &outline, plan) {
            guard.ensure_entries(backend, load, geometry, &outline)?;
            guard.certify_checked_pair(outline, load, geometry, state, plan, limits, pass, Arc::clone(cache))
        } else {
            drop(guard);
            let layouts = FamilyLayouts::<SizingOutline>::of_entries(backend, load, geometry, plan, &outline)?;
            let fresh = Arc::new(Mutex::new(layouts));
            let fresh_handle: Arc<dyn Any + Send + Sync> = fresh.clone();
            let mut guard = fresh.lock().expect("target family layout cache mutex poisoned");
            guard.certify_checked_pair(outline, load, geometry, state, plan, limits, pass, fresh_handle)
        }
    }
}

pub(super) struct TargetFamilyCertificate<L = NativeGraphLayout> {
    pub(super) outline: FamilyOutline,
    /// How each block is served (`block_served`).
    pub(super) served: Vec<BlockServed>,
    /// What its prompt groups' buffers span.
    pub(super) group: GroupRows,
    pub(super) resources: CheckedGraphResources,
    /// What the family's decode and largest launches replay
    /// (`seismic::NativeGraphFamily::graphs`), bounded from the layouts the
    /// plans seal into.
    pub(super) graphs: LaunchGraphs,
    /// The graphs a load seals into these layouts, and their host heap.
    pub(super) sealed: PlannedSealedGraphs,
    /// The upload regions of a workspace slot: one for each graph run that
    /// writes input one step queues on it before any completes, each
    /// holding that run's input until it completes. The embedding entry's,
    /// the per-layer entry's of a per-layer model, each the largest input
    /// its layouts place, the launch's controls, which its blocks bind
    /// (`LaunchControlLayout`), and each block's (`block_upload_regions`).
    pub(super) upload_regions: Vec<u64>,
    pub(super) entries: BTreeMap<EntryTokens, L>,
    /// The per-layer entry graph's layout, of a per-layer entry.
    pub(super) per_layer: Option<L>,
    /// Per block, its certified graphs by topology (`class_topology`).
    pub(super) blocks: Vec<BTreeMap<Topology, CertifiedBlockGraph<L>>>,
    /// Per tapped block, its untapped graphs' (`FamilyOutline::untapped`):
    /// its undrafted shape's, as a load without the drafter certifies them;
    /// none for a block that taps nothing.
    pub(super) untapped: Vec<BTreeMap<Topology, CertifiedBlockGraph<L>>>,
    /// The passes a group runs (`group_pass_rows`).
    pub(super) passes: Passes,
}

/// What every class of one block's row regime prepares to: one topology.
#[derive(Clone)]
pub(super) struct CertifiedBlockGraph<L = NativeGraphLayout> {
    /// The layout its plans seal into.
    pub(super) layout: L,
    /// The host heap of the vectors each of its prepared graphs owns
    /// (`owned_vectors_heap`). Every weight, constant and state port is a
    /// port of the graph, and the layout holds its classes to its ports,
    /// so the template's lengths are each class's.
    pub(super) owned: HostHeap,
}

pub(super) fn certify_target_family<L: Certified + Send + Sync>(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
    pass: PassLaunches,
) -> Result<TargetFamilyCertificate<L>, GraphError> {
    let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
    FamilyLayouts::of_entries(backend, load, geometry, plan, &outline)?
        .certify(outline, load, geometry, state, plan, limits, pass)
}

/// The family of `plan`, and the family of the same target without its
/// drafter's writes (`FamilyOutline::undrafted`), what a load of the target
/// alone seals: from one certification of the layouts they share, each
/// family's composition, bindings, upload regions, census and floor extents
/// its own. A plan that drafts with nothing has one family, twice.
pub(super) fn certify_target_and_undrafted<L: Certified + Send + Sync>(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &Decoder,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
    pass: PassLaunches,
) -> Result<[TargetFamilyCertificate<L>; 2], GraphError> {
    let outline = FamilyOutline::of(backend, load, geometry, state, plan, limits)?;
    let undrafted = outline.undrafted();
    let mut layouts = FamilyLayouts::of_entries(backend, load, geometry, plan, &outline)?;
    Ok([
        layouts.certify(outline, load, geometry, state, plan, limits, pass)?,
        layouts.certify(undrafted, load, geometry, state, plan, limits, pass)?,
    ])
}

/// What a certified graph adds to its family's resources: its storage and
/// constants as a template certified at its rows' classes, or as the one
/// class of a block's pass or fill.
#[derive(Clone)]
enum Included {
    Classes(Vec<GraphConstant>, Vec<u64>),
    Once(Vec<GraphConstant>),
}

/// One block shape's certified graphs by topology, with what each adds to
/// its family's resources.
type ShapeLayouts<L> = BTreeMap<Topology, (CertifiedBlockGraph<L>, Included)>;

/// Invariant scope over which cached entry and block layouts in
/// `FamilyLayouts` are valid.
#[derive(Clone, PartialEq, Eq)]
struct LayoutsScope {
    backend: BackendName,
    row_classes: Vec<u64>,
    max_slots: u64,
    max_segments: u64,
    has_per_layer: bool,
}

/// Entry-graph properties that depend on the load plan (`fronts`, token-row
/// copies, and streamed embedding table landing rows). When only this scope
/// changes (e.g., plain -> drafted or banked -> resident), `entries` is
/// re-certified while `per_layer`, `launches`, `groups`, and `once_layouts`
/// remain valid.
#[derive(Clone, PartialEq, Eq)]
struct EntryScope {
    copies_token_rows: bool,
    notifies_selection: bool,
    keeps_token_rows: bool,
    fronts: bool,
    target_table: Option<(Vec<u64>, seismic::Element, bool)>,
}

impl EntryScope {
    fn of(load: &ModelLoadPlan) -> Self {
        let target_table = load
            .weight_by_role(WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Embedding,
            })
            .map(|weight| {
                (
                    weight.shape.clone(),
                    weight.resident,
                    matches!(weight.holding, crate::planning::Holding::Streamed(_)),
                )
            });
        Self {
            copies_token_rows: load.copies_token_rows(),
            notifies_selection: load.notifies_selection(),
            keeps_token_rows: load.keeps_token_rows(),
            fronts: load.fronts(),
            target_table,
        }
    }
}

#[derive(Clone, PartialEq)]
struct LaunchKey {
    split: bool,
    resident: bool,
    shape: BlockGraphShape,
}

impl LaunchKey {
    fn fx_hash(&self) -> u64 {
        use std::hash::Hash;
        let mut h = FxHasher::default();
        self.split.hash(&mut h);
        self.resident.hash(&mut h);
        self.shape.fx_hash().hash(&mut h);
        h.finish()
    }
}

#[derive(Clone, PartialEq)]
struct GroupKey {
    passes: Passes,
    untapped: bool,
    split: bool,
    resident: bool,
    streams: bool,
    shape: BlockGraphShape,
}

impl GroupKey {
    fn fx_hash(&self) -> u64 {
        use std::hash::Hash;
        let mut h = FxHasher::default();
        self.passes.region.hash(&mut h);
        self.passes.launches.hash(&mut h);
        self.untapped.hash(&mut h);
        self.split.hash(&mut h);
        self.resident.hash(&mut h);
        self.streams.hash(&mut h);
        self.shape.fx_hash().hash(&mut h);
        h.finish()
    }
}

#[derive(Clone, PartialEq)]
struct OnceKey {
    stage: OnceStage,
    rows: u64,
    streams: bool,
    shape: OnceGraphShape,
}

impl OnceKey {
    fn fx_hash(&self) -> u64 {
        use std::hash::Hash;
        let mut h = FxHasher::default();
        std::mem::discriminant(&self.stage).hash(&mut h);
        self.rows.hash(&mut h);
        self.streams.hash(&mut h);
        self.shape.fx_hash().hash(&mut h);
        h.finish()
    }
}

/// The layouts a target's families seal into, each certified once: the
/// entries', and each block shape's (`BlockGraphShape`, which decides
/// every graph a block prepares, the sealer sharing one plan between the
/// blocks of a shape): the launches' own, and its group's at the passes
/// the group runs. The families of one target differ only in which shape
/// each block is of, so each reads its blocks' layouts here.
struct FamilyLayouts<L> {
    scope: LayoutsScope,
    entry_scope: EntryScope,
    entry_generation: u64,
    entries: BTreeMap<EntryTokens, L>,
    per_layer: Option<L>,
    empty_layouts: Arc<ShapeLayouts<L>>,
    launches: Vec<(LaunchKey, Arc<ShapeLayouts<L>>)>,
    launch_index: FxHashMap<u64, Vec<usize>>,
    /// A shape's group graphs at a group's passes: a block's own, its
    /// combines and its passes and fills; or, of a tapped block's untapped
    /// shape (`untapped == true`), its combines alone.
    groups: Vec<(GroupKey, Arc<ShapeLayouts<L>>)>,
    group_index: FxHashMap<u64, Vec<usize>>,
    /// Cached `OnceStage::Pass` and `OnceStage::Fill` layouts keyed by
    /// feed-forward-only [`OnceKey`], shared between `group_pass_rows` and
    /// group assembly across blocks with identical routed sublayers.
    once_layouts: Vec<(OnceKey, CertifiedBlockGraph<L>, Vec<GraphConstant>)>,
    once_index: FxHashMap<u64, Vec<usize>>,
    /// Cached deduplicated kernel sets keyed by
    /// `(entry_generation, sorted_launch_ids, sorted_group_ids)` so repeat
    /// `certify_checked` calls clone the pre-built `HashSet` with zero SipHash
    /// hashing of `KernelRequest`s.
    cached_kernels: FxHashMap<(u64, Vec<usize>, Vec<usize>), HashSet<seismic::KernelRequest>>,
}

/// Core outputs of certifying one family outline against `FamilyLayouts`,
/// before expanding per-block layout maps (only needed by `TargetFamilyCertificate`).
struct CertifiedCore<L> {
    outline: FamilyOutline,
    served: Vec<BlockServed>,
    group: GroupRows,
    resources: CheckedGraphResources,
    graphs: LaunchGraphs,
    sealed: PlannedSealedGraphs,
    upload_regions: Vec<u64>,
    passes: Passes,
    /// `(block_index, launch_id, group_id, launches, groups)` per distinct block shape.
    distinct_blocks: Vec<(usize, usize, Option<usize>, Arc<ShapeLayouts<L>>, Arc<ShapeLayouts<L>>)>,
    distinct_untapped: Vec<(usize, Arc<ShapeLayouts<L>>, Arc<ShapeLayouts<L>>)>,
    untapped_holders: Vec<Option<Holder>>,
}

/// Run `work` for each item in `items` across the persistent assessment pool,
/// returning results (or the first error in input order) in exact input order.
fn run_in_parallel<T: Send + Sync, R: Send, E: Send>(
    items: Vec<T>,
    work: impl Fn(&T) -> Result<R, E> + Sync,
) -> Result<Vec<R>, E> {
    crate::assessment::pool::par_for_results(&items, work)
}

/// Shared read-only inputs for certifying a single family outline against
/// `FamilyLayouts`.
struct CertifyContext<'a> {
    outline: &'a FamilyOutline,
    load: &'a ModelLoadPlan,
    geometry: &'a Decoder,
    state: &'a StateResourcePlan,
    plan: &'a TargetProgramPlan,
    regimes: &'a [Vec<u64>],
}

struct RunTask {
    slot_idx: usize,
    block_index: usize,
    of_untapped: bool,
    regime_idx: usize,
    listed: bool,
    scored: bool,
    stage: RunStage,
    largest: u64,
    is_attention: bool,
}

struct MissingGroup {
    index: usize,
    of_untapped: bool,
    split: bool,
    resident: bool,
    streams: bool,
    key_hash: u64,
    key: GroupKey,
}

type BlockMetaMap = FxHashMap<(usize, bool), (BlockGraphShape, bool, bool, bool)>;

struct ResolvedFamilyIds {
    distinct_blocks: Vec<usize>,
    distinct_untapped: Vec<usize>,
    untapped_holders: Vec<Option<Holder>>,
    block_launch_ids: FxHashMap<usize, usize>,
    untapped_launch_ids: FxHashMap<usize, usize>,
    block_group_ids: FxHashMap<usize, Option<usize>>,
    untapped_group_ids: FxHashMap<usize, Option<usize>>,
}

/// A block's pass and fill, of the one class of `rows` rows; a fill reads
/// no rows.
fn certify_once_block<L: Certified>(
    ctx: &CertifyContext<'_>,
    index: usize,
    stage: OnceStage,
    rows: u64,
) -> Result<(L, Vec<GraphConstant>, HostHeap), GraphError> {
    let slot = ctx.plan.blocks()[index];
    let (slice, classes) = match stage {
        OnceStage::Pass => (
            NativeGraphClassSlice::new().dimension("M", [rows]),
            &[(None, "M", ClassNumber::Rows)][..],
        ),
        OnceStage::Fill => (NativeGraphClassSlice::new(), &[][..]),
    };
    let certified = certify_block::<L>(
        ctx.outline.backend,
        slot,
        None,
        BlockGraphInputs::of_class(
            ctx.load,
            ctx.geometry,
            ctx.state,
            index,
            once_class(rows, stage),
            rows,
            ctx.outline.carries[index],
        ),
        classes,
        &[vec![slice]],
    )
    .map_err(|error| error.context(format!("target block {index}'s {stage:?}")))?;
    let [(layout, (constants, owned))] = <[_; 1]>::try_from(certified)
        .unwrap_or_else(|_| unreachable!("one slice set certifies to one layout"));
    Ok((layout, constants, owned))
}

fn certify_run_task<L: Certified>(
    ctx: &CertifyContext<'_>,
    task: &RunTask,
    prefill_tiles: &[Vec<u64>],
    pass_rows: u64,
) -> Result<(usize, Topology, CertifiedBlockGraph<L>, Included), GraphError> {
    let index = task.block_index;
    let slot = ctx.plan.blocks()[index];
    let paired = paired_block(&ctx.geometry.blocks[index]).map_err(|error| error.to_string())?;
    let regime_rows = &ctx.regimes[task.regime_idx];
    let listings: &[u64] = if task.listed { &prefill_tiles[task.slot_idx] } else { &[] };
    let classes = block_class_numbers(&paired, !listings.is_empty());
    let drafted = (
        (task.largest, 1, 1, listings.last().copied().unwrap_or(0), u64::from(task.scored)),
        Stage::Run(task.stage),
    );
    let slices = regime_rows
        .iter()
        .map(|&rows| {
            block_class_slice(&paired, rows, ctx.outline.max_slots, ctx.outline.max_segments, listings)
        })
        .collect::<Vec<_>>();
    let certified = certify_block::<L>(
        ctx.outline.backend,
        slot,
        ctx.outline
            .tap_width
            .map(|width| (width, ctx.outline.tapped[index].positions()))
            .filter(|(_, positions)| !task.of_untapped && positions.any()),
        BlockGraphInputs {
            residual: !task.of_untapped && ctx.outline.exports_residual(index),
            ..BlockGraphInputs::of_class(
                ctx.load,
                ctx.geometry,
                ctx.state,
                index,
                drafted,
                pass_rows,
                ctx.outline.carries[index],
            )
        },
        &classes,
        &[slices],
    )
    .map_err(|error| error.context(format!("target block {index}")))?;
    let [(layout, (constants, owned))] = <[_; 1]>::try_from(certified)
        .unwrap_or_else(|_| unreachable!("one slice set certifies to one layout"));
    Ok((
        task.slot_idx,
        class_topology(drafted),
        CertifiedBlockGraph { layout, owned },
        Included::Classes(constants, regime_rows.clone()),
    ))
}

fn collect_run_tasks(
    ctx: &CertifyContext<'_>,
    entries: &[(usize, bool, bool, bool)],
    group: bool,
) -> Result<(Vec<Vec<u64>>, Vec<RunTask>), GraphError> {
    let has_prefill = !crate::operators::attention::graph::decodes(ctx.outline.largest_rows);
    let mut prefill_tiles: Vec<Vec<u64>> = Vec::with_capacity(entries.len());
    let mut tasks: Vec<RunTask> = Vec::new();
    for (slot_idx, &(index, of_untapped, split, resident)) in entries.iter().enumerate() {
        let slot = ctx.plan.blocks()[index];
        let tiles = if has_prefill {
            block_listings(ctx.state, ctx.outline.backend, slot, index, ctx.outline.largest_rows)?.tiles
        } else {
            Vec::new()
        };
        let scores = slot_scores(slot);
        let is_attention = matches!(slot.mixer(), crate::MixerProgramSlot::Attention(_));
        for (regime_idx, regime_rows) in ctx.regimes.iter().enumerate() {
            let largest = *regime_rows.last().expect("a regime holds a row class");
            let is_decode = crate::operators::attention::graph::decodes(largest);
            let stages = RunStage::sealed(split, resident, largest);
            let has_stage = stages.iter().any(|&stage| (stage == RunStage::Combine) == group);
            if !has_stage {
                continue;
            }
            let topologies = [(false, false), (true, false)]
                .into_iter()
                .chain(scores.then_some((false, true)));
            for (listed, scored) in topologies {
                if listed && (is_decode || tiles.is_empty()) {
                    continue;
                }
                for &stage in stages.iter().filter(|&&stage| (stage == RunStage::Combine) == group) {
                    tasks.push(RunTask {
                        slot_idx,
                        block_index: index,
                        of_untapped,
                        regime_idx,
                        listed,
                        scored,
                        stage,
                        largest,
                        is_attention,
                    });
                }
            }
        }
        prefill_tiles.push(tiles);
    }
    // Schedule heaviest tasks first (Attention, listed, largest rows)
    // to minimize tail latency under work-stealing.
    tasks.sort_unstable_by_key(|t| (
        !t.is_attention,
        !t.listed,
        std::cmp::Reverse(t.largest),
        t.slot_idx,
    ));
    Ok((prefill_tiles, tasks))
}

impl<L: Certified + Send + Sync> FamilyLayouts<L> {
    /// The entry graphs' layouts, the same in every family of the target.
    fn of_entries(
        backend: BackendName,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        plan: &TargetProgramPlan,
        outline: &FamilyOutline,
    ) -> Result<Self, GraphError> {
        let FamilyOutline { ref row_classes, largest_rows, .. } = *outline;
        let entries = [EntryTokens::Uploaded, EntryTokens::Selected]
            .into_iter()
            .map(|source| Ok((source, certify_entry(backend, load, geometry, source, row_classes, largest_rows)?)))
            .collect::<Result<BTreeMap<_, _>, GraphError>>()?;
        let per_layer = match (plan.per_layer(), &geometry.entry.per_layer) {
            (None, None) => None,
            (Some(binding), Some(entry)) => {
                let checked = CheckedPerLayerEntryEntries::new(binding);
                let mut graph = L::new_draft(backend);
                let (topology, numbers) = split_entry(&binding, entry);
                let topology = crate::programs::graph::facts::constant_reals(topology);
                per_layer_entry(
                    &mut graph,
                    checked.entries(),
                    &topology,
                    &crate::programs::graph::facts::constant(numbers),
                    &target_view(load, WeightKind::PerLayerModelProjection)?,
                    &target_view(load, WeightKind::PerLayerProjectionNorm)?,
                    &seismic::Dim::from(largest_rows),
                )?;
                let layout = L::certify_one(
                    graph,
                    vec![per_layer_entry_class_slice(row_classes.iter().copied())],
                )
                .map_err(|error| error.context("per-layer entry graph"))?;
                Some(layout)
            }
            _ => return Err("the per-layer entry's binding and geometry disagree".into()),
        };
        let scope = LayoutsScope {
            backend,
            row_classes: outline.row_classes.clone(),
            max_slots: outline.max_slots,
            max_segments: outline.max_segments,
            has_per_layer: plan.per_layer().is_some(),
        };
        let entry_scope = EntryScope::of(load);
        Ok(Self {
            scope,
            entry_scope,
            entry_generation: 0,
            entries,
            per_layer,
            empty_layouts: Arc::new(ShapeLayouts::new()),
            launches: Vec::new(),
            launch_index: fx_map(),
            groups: Vec::new(),
            group_index: fx_map(),
            once_layouts: Vec::new(),
            once_index: fx_map(),
            cached_kernels: fx_map(),
        })
    }

    fn find_launch(&self, hash: u64, key: &LaunchKey) -> Option<usize> {
        self.launch_index
            .get(&hash)
            .and_then(|bucket| bucket.iter().copied().find(|&id| self.launches[id].0 == *key))
    }

    fn insert_launch(&mut self, hash: u64, key: LaunchKey, value: Arc<ShapeLayouts<L>>) -> usize {
        let id = self.launches.len();
        self.launches.push((key, value));
        self.launch_index.entry(hash).or_default().push(id);
        id
    }

    fn find_group(&self, hash: u64, key: &GroupKey) -> Option<usize> {
        self.group_index
            .get(&hash)
            .and_then(|bucket| bucket.iter().copied().find(|&id| self.groups[id].0 == *key))
    }

    fn insert_group(&mut self, hash: u64, key: GroupKey, value: Arc<ShapeLayouts<L>>) -> usize {
        let id = self.groups.len();
        self.groups.push((key, value));
        self.group_index.entry(hash).or_default().push(id);
        id
    }

    fn find_once(&self, hash: u64, key: &OnceKey) -> Option<usize> {
        self.once_index
            .get(&hash)
            .and_then(|bucket| bucket.iter().copied().find(|&id| self.once_layouts[id].0 == *key))
    }

    fn insert_once(
        &mut self,
        hash: u64,
        key: OnceKey,
        graph: CertifiedBlockGraph<L>,
        constants: Vec<GraphConstant>,
    ) -> usize {
        let id = self.once_layouts.len();
        self.once_layouts.push((key, graph, constants));
        self.once_index.entry(hash).or_default().push(id);
        id
    }

    fn matches_scope(&self, backend: BackendName, outline: &FamilyOutline, plan: &TargetProgramPlan) -> bool {
        self.scope.backend == backend
            && self.scope.row_classes == outline.row_classes
            && self.scope.max_slots == outline.max_slots
            && self.scope.max_segments == outline.max_segments
            && self.scope.has_per_layer == plan.per_layer().is_some()
    }

    fn ensure_entries(
        &mut self,
        backend: BackendName,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        outline: &FamilyOutline,
    ) -> Result<(), GraphError> {
        let entry_scope = EntryScope::of(load);
        if self.entry_scope != entry_scope {
            let FamilyOutline { ref row_classes, largest_rows, .. } = *outline;
            self.entries = [EntryTokens::Uploaded, EntryTokens::Selected]
                .into_iter()
                .map(|source| Ok((source, certify_entry(backend, load, geometry, source, row_classes, largest_rows)?)))
                .collect::<Result<BTreeMap<_, _>, GraphError>>()?;
            self.entry_scope = entry_scope;
            self.entry_generation += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn certify_checked_pair(
        &mut self,
        outline: FamilyOutline,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        pass: PassLaunches,
        cache: Arc<dyn Any + Send + Sync>,
    ) -> Result<[CheckedTargetFamily; 2], GraphError> {
        let is_drafted = outline.tap_width.is_some() || outline.residual.is_some();
        let undrafted = is_drafted.then(|| outline.undrafted());
        let target = self.certify_checked(
            outline,
            load,
            geometry,
            state,
            plan,
            limits,
            pass,
            Arc::clone(&cache),
        )?;
        let undrafted = match undrafted {
            Some(undrafted) => self.certify_checked(
                undrafted,
                load,
                geometry,
                state,
                plan,
                limits,
                pass,
                cache,
            )?,
            None => target.clone_checked(),
        };
        Ok([target, undrafted])
    }

    /// Certify `outline` directly into a [`CheckedTargetFamily`] without
    /// cloning per-block layout maps across identical-shape blocks.
    #[allow(clippy::too_many_arguments)]
    fn certify_checked(
        &mut self,
        outline: FamilyOutline,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        pass: PassLaunches,
        cache: Arc<dyn Any + Send + Sync>,
    ) -> Result<CheckedTargetFamily, GraphError> {
        let core = self.certify_core(outline, load, geometry, state, plan, limits, pass)?;
        let held_pass = crate::PassLaunches::new(
            std::num::NonZeroUsize::new(core.passes.launches()).expect("a pass covers a launch"),
        );
        let group_bytes = crate::planning::ByPass::of(held_pass, |pass| {
            core.group.group_bytes(&core.served, pass.get())
        })?;
        let mut census = TargetFamilySizingCensus::default();
        let mut launch_ids: Vec<usize> = Vec::with_capacity(core.distinct_blocks.len());
        let mut group_ids: Vec<usize> = Vec::with_capacity(core.distinct_blocks.len());
        for &(_, launch_id, group_id, _, _) in &core.distinct_blocks {
            launch_ids.push(launch_id);
            if let Some(gid) = group_id {
                group_ids.push(gid);
            }
        }
        launch_ids.sort_unstable();
        launch_ids.dedup();
        group_ids.sort_unstable();
        group_ids.dedup();
        let kernel_cache_key = (self.entry_generation, launch_ids, group_ids);
        if let Some(cached) = self.cached_kernels.get(&kernel_cache_key) {
            census.kernels = cached.clone();
        } else {
            let mut seen: FxHashSet<&seismic::KernelRequest> = fx_set();
            for layout in self.entries.values().chain(&self.per_layer) {
                if let Some((_, kernels, _)) = layout.sizing_census() {
                    seen.extend(kernels.iter());
                }
            }
            for &id in &kernel_cache_key.1 {
                for (graph, _) in self.launches[id].1.values() {
                    if let Some((_, kernels, _)) = graph.layout.sizing_census() {
                        seen.extend(kernels.iter());
                    }
                }
            }
            for &id in &kernel_cache_key.2 {
                for (graph, _) in self.groups[id].1.values() {
                    if let Some((_, kernels, _)) = graph.layout.sizing_census() {
                        seen.extend(kernels.iter());
                    }
                }
            }
            if !seen.is_empty() {
                let mut kernels = HashSet::with_capacity(seen.len());
                kernels.extend(seen.into_iter().cloned());
                census.kernels = kernels.clone();
                self.cached_kernels.insert(kernel_cache_key, kernels);
            }
        }
        if let Some(uploaded) = self.entries.get(&EntryTokens::Uploaded) {
            if let Some((calls, _, timelines)) = uploaded.sizing_census() {
                census.entry_calls.extend_from_slice(calls);
                census.entry_timelines += timelines;
            }
        }
        if let Some(per_layer) = &self.per_layer {
            if let Some((calls, _, timelines)) = per_layer.sizing_census() {
                census.entry_calls.extend_from_slice(calls);
                census.entry_timelines += timelines;
            }
        }
        let decode_form = RowForm::of(1);
        fn find_decode<'a, L>(
            launches: &'a ShapeLayouts<L>,
            groups: &'a ShapeLayouts<L>,
            decode_form: RowForm,
            stage: RunStage,
        ) -> Option<&'a CertifiedBlockGraph<L>> {
            let scored = (decode_form, false, true, Stage::Run(stage), 0);
            let plain = (decode_form, false, false, Stage::Run(stage), 0);
            launches
                .get(&scored)
                .or_else(|| groups.get(&scored))
                .or_else(|| launches.get(&plain))
                .or_else(|| groups.get(&plain))
                .map(|(graph, _)| graph)
        }
        let mut shape_index_by_first: FxHashMap<usize, usize> = fx_map();
        for (index, _, _, launches, groups) in &core.distinct_blocks {
            if let Some(whole) = find_decode(launches, groups, decode_form, RunStage::Whole) {
                if let Some((calls, _, timelines)) = whole.layout.sizing_census() {
                    let resident = find_decode(launches, groups, decode_form, RunStage::Resident).and_then(|res| {
                        res.layout
                            .sizing_census()
                            .map(|(c, _, t)| (c.to_vec(), t))
                    });
                    let shape_idx = census.block_shapes.len();
                    census.block_shapes.push(BlockDecodeCensus {
                        calls: calls.to_vec(),
                        timelines,
                        resident,
                    });
                    shape_index_by_first.insert(*index, shape_idx);
                }
            }
        }
        if !shape_index_by_first.is_empty() {
            census.block_map = (0..core.outline.shapes.len())
                .map(|index| shape_index_by_first[&core.outline.first_of_shape(index)])
                .collect();
        }
        Ok(CheckedTargetFamily {
            resources: core.resources,
            sealed: core.sealed,
            upload_regions: core.upload_regions,
            graphs: core.graphs,
            group_bytes,
            census,
            cache,
        })
    }

    /// The family of `outline`, its blocks' layouts certified where no
    /// family certified their shape's.
    #[allow(clippy::too_many_arguments)]
    fn certify(
        &mut self,
        outline: FamilyOutline,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        pass: PassLaunches,
    ) -> Result<TargetFamilyCertificate<L>, GraphError> {
        let core = self.certify_core(outline, load, geometry, state, plan, limits, pass)?;
        let block_count = core.outline.shapes.len();
        let mut blocks = vec![BTreeMap::new(); block_count];
        for (index, _, _, launches, groups) in &core.distinct_blocks {
            let map = launches
                .iter()
                .chain(groups.iter())
                .map(|(&topology, (graph, _))| (topology, graph.clone()))
                .collect::<BTreeMap<_, _>>();
            blocks[*index] = map;
        }
        for index in 0..block_count {
            let first = core.outline.first_of_shape(index);
            if first != index {
                blocks[index] = blocks[first].clone();
            }
        }
        let mut untapped_held = vec![BTreeMap::new(); block_count];
        for (index, launches, groups) in &core.distinct_untapped {
            untapped_held[*index] = launches
                .iter()
                .chain(groups.iter())
                .map(|(&topology, (graph, _))| (topology, graph.clone()))
                .collect::<BTreeMap<_, _>>();
        }
        let untapped = core
            .untapped_holders
            .iter()
            .map(|holder| match holder {
                None => BTreeMap::new(),
                Some(Holder::Block(block)) => blocks[*block].clone(),
                Some(Holder::Untapped(first)) => untapped_held[*first].clone(),
            })
            .collect::<Vec<_>>();
        Ok(TargetFamilyCertificate {
            outline: core.outline,
            served: core.served,
            group: core.group,
            resources: core.resources,
            graphs: core.graphs,
            sealed: core.sealed,
            upload_regions: core.upload_regions,
            entries: self.entries.clone(),
            per_layer: self.per_layer.clone(),
            blocks,
            untapped,
            passes: core.passes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn certify_core(
        &mut self,
        outline: FamilyOutline,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        pass: PassLaunches,
    ) -> Result<CertifiedCore<L>, GraphError> {
        let block_count = geometry.blocks.len();
        let distinct_blocks = (0..block_count)
            .filter(|&index| outline.first_of_shape(index) == index)
            .collect::<Vec<_>>();
        // Precompute row regimes once for the family (identical for all blocks).
        let regimes: Vec<Vec<u64>> = {
            let mut by_form: BTreeMap<RowForm, Vec<u64>> = BTreeMap::new();
            for &rows in &outline.row_classes {
                by_form.entry(RowForm::of(rows)).or_default().push(rows);
            }
            by_form.into_values().collect()
        };
        let chain = Passes {
            region: outline.largest_rows,
            launches: pass.get().min(limits.chain_capacity.get()),
        };
        let untapped_holders = outline.all_untapped_holders();
        let distinct_untapped = (0..block_count)
            .filter(|&index| untapped_holders[index] == Some(Holder::Untapped(index)))
            .collect::<Vec<_>>();
        let shaped = distinct_blocks
            .iter()
            .map(|&index| (index, false))
            .chain(distinct_untapped.iter().map(|&index| (index, true)))
            .collect::<Vec<_>>();
        let bank_layers = bank_layer_by_block(load, block_count);
        let (block_launch_ids, untapped_launch_ids, block_group_ids, untapped_group_ids, passes) = {
            let ctx = CertifyContext {
                outline: &outline,
                load,
                geometry,
                state,
                plan,
                regimes: &regimes,
            };
            let (block_meta, block_launch_ids, untapped_launch_ids) =
                self.certify_launches(&ctx, &shaped, &bank_layers, limits, chain)?;
            let passes = self.resolve_passes(
                &ctx,
                &distinct_blocks,
                &bank_layers,
                &block_launch_ids,
                &untapped_launch_ids,
                chain,
            )?;
            let (block_group_ids, untapped_group_ids) =
                self.certify_groups(&ctx, &shaped, block_meta, passes)?;
            (block_launch_ids, untapped_launch_ids, block_group_ids, untapped_group_ids, passes)
        };
        self.finish_core(
            outline,
            load,
            geometry,
            state,
            plan,
            limits,
            passes,
            ResolvedFamilyIds {
                distinct_blocks,
                distinct_untapped,
                untapped_holders,
                block_launch_ids,
                untapped_launch_ids,
                block_group_ids,
                untapped_group_ids,
            },
        )
    }

    fn certify_launches(
        &mut self,
        ctx: &CertifyContext<'_>,
        shaped: &[(usize, bool)],
        bank_layers: &[Option<usize>],
        limits: ResourceLimits,
        chain: Passes,
    ) -> Result<(BlockMetaMap, FxHashMap<usize, usize>, FxHashMap<usize, usize>), GraphError> {
        let bank_resident = ctx.load.expert_bank().is_some_and(|bank| bank.resident);
        let chain_merges = limits.chain_capacity.get() > 1;
        let has_prompts = !crate::operators::routed::fused_graph::decodes(ctx.outline.largest_rows);
        let mut block_meta: BlockMetaMap = fx_map();
        let mut block_launch_ids: FxHashMap<usize, usize> = fx_map();
        let mut untapped_launch_ids: FxHashMap<usize, usize> = fx_map();
        let mut pending_launch_lookups: Vec<(usize, bool, u64, LaunchKey)> = Vec::new();
        let mut missing_launches: Vec<(usize, bool, bool, bool, u64, LaunchKey)> = Vec::new();
        for &(index, of_untapped) in shaped {
            let shape = match of_untapped {
                true => ctx
                    .outline
                    .untapped(index)
                    .expect("a distinct untapped shape is a tapped block's"),
                false => ctx.outline.shapes[index].clone(),
            };
            let streams = bank_layers[index].is_some();
            let resident = streams && bank_resident;
            let split = has_prompts
                && (chain_merges || streams)
                && block_routed(ctx.geometry, index)?.is_some();
            let key = LaunchKey { split, resident, shape: shape.clone() };
            let key_hash = key.fx_hash();
            if let Some(id) = self.find_launch(key_hash, &key) {
                match of_untapped {
                    true => {
                        untapped_launch_ids.insert(index, id);
                    }
                    false => {
                        block_launch_ids.insert(index, id);
                    }
                }
            } else {
                if !missing_launches
                    .iter()
                    .any(|(_, _, _, _, pending_hash, pending)| *pending_hash == key_hash && *pending == key)
                {
                    missing_launches.push((index, of_untapped, split, resident, key_hash, key.clone()));
                }
                pending_launch_lookups.push((index, of_untapped, key_hash, key));
            }
            block_meta.insert((index, of_untapped), (shape, split, resident, streams));
        }
        if !missing_launches.is_empty() {
            let launch_entries: Vec<(usize, bool, bool, bool)> = missing_launches
                .iter()
                .map(|&(index, of_untapped, split, resident, _, _)| (index, of_untapped, split, resident))
                .collect();
            let (prefill_tiles, launch_tasks) = collect_run_tasks(ctx, &launch_entries, false)?;
            let mut launch_maps: Vec<ShapeLayouts<L>> = (0..missing_launches.len())
                .map(|_| ShapeLayouts::new())
                .collect();
            if !launch_tasks.is_empty() {
                let results = run_in_parallel(launch_tasks, |task| {
                    certify_run_task::<L>(ctx, task, &prefill_tiles, chain.rows())
                })?;
                for (slot_idx, topology, graph, included) in results {
                    launch_maps[slot_idx].insert(topology, (graph, included));
                }
            }
            for ((_, _, _, _, key_hash, key), graphs) in missing_launches.into_iter().zip(launch_maps) {
                self.insert_launch(key_hash, key, Arc::new(graphs));
            }
            for (index, of_untapped, key_hash, key) in pending_launch_lookups {
                let id = self
                    .find_launch(key_hash, &key)
                    .expect("launch key was certified or cached");
                match of_untapped {
                    true => {
                        untapped_launch_ids.insert(index, id);
                    }
                    false => {
                        block_launch_ids.insert(index, id);
                    }
                }
            }
        }
        Ok((block_meta, block_launch_ids, untapped_launch_ids))
    }

    fn resolve_passes(
        &mut self,
        ctx: &CertifyContext<'_>,
        distinct_blocks: &[usize],
        bank_layers: &[Option<usize>],
        block_launch_ids: &FxHashMap<usize, usize>,
        untapped_launch_ids: &FxHashMap<usize, usize>,
        chain: Passes,
    ) -> Result<Passes, GraphError> {
        let arena = block_launch_ids
            .values()
            .chain(untapped_launch_ids.values())
            .flat_map(|&id| self.launches[id].1.values())
            .map(|(graph, _)| graph.layout.storage_bytes().workspace)
            .chain(
                self.entries
                    .values()
                    .chain(&self.per_layer)
                    .map(|layout| layout.storage_bytes().workspace),
            )
            .max()
            .expect("a target family has entry graphs");
        if chain.launches <= 1 {
            return Ok(chain);
        }
        let mut routed_blocks: Vec<usize> = Vec::new();
        let mut routed_once_meta: FxHashMap<usize, (OnceGraphShape, bool)> = fx_map();
        for &index in distinct_blocks {
            if block_routed(ctx.geometry, index)?.is_none() {
                continue;
            }
            let once_shape = ctx.outline.shapes[index].once_shape();
            let streams = bank_layers[index].is_some();
            if !routed_once_meta
                .values()
                .any(|(s, st)| *st == streams && *s == once_shape)
            {
                routed_blocks.push(index);
                routed_once_meta.insert(index, (once_shape, streams));
            }
        }
        group_pass_rows(&routed_blocks, ctx.geometry, chain, arena, |index, rows| {
            let (ref once_shape, streams) = routed_once_meta[&index];
            let key = OnceKey {
                stage: OnceStage::Pass,
                rows,
                streams,
                shape: once_shape.clone(),
            };
            let key_hash = key.fx_hash();
            if let Some(id) = self.find_once(key_hash, &key) {
                return Ok(self.once_layouts[id].1.layout.storage_bytes().workspace);
            }
            let (layout, constants, owned) = certify_once_block::<L>(ctx, index, OnceStage::Pass, rows)?;
            let workspace = layout.storage_bytes().workspace;
            self.insert_once(key_hash, key, CertifiedBlockGraph { layout, owned }, constants);
            Ok(workspace)
        })
    }

    fn certify_groups(
        &mut self,
        ctx: &CertifyContext<'_>,
        shaped: &[(usize, bool)],
        mut block_meta: BlockMetaMap,
        passes: Passes,
    ) -> Result<(FxHashMap<usize, Option<usize>>, FxHashMap<usize, Option<usize>>), GraphError> {
        let mut block_group_ids: FxHashMap<usize, Option<usize>> = fx_map();
        let mut untapped_group_ids: FxHashMap<usize, Option<usize>> = fx_map();
        let mut pending_group_lookups: Vec<(usize, bool, u64, GroupKey)> = Vec::new();
        let mut missing_groups: Vec<MissingGroup> = Vec::new();
        for &(index, of_untapped) in shaped {
            let (shape, split, resident, streams) = block_meta
                .remove(&(index, of_untapped))
                .expect("shape metadata recorded in launch phase");
            // Non-split blocks have no `RunStage::Combine` and no `once_graphs`,
            // so their group layout map is always empty.
            if !split {
                match of_untapped {
                    true => {
                        untapped_group_ids.insert(index, None);
                    }
                    false => {
                        block_group_ids.insert(index, None);
                    }
                }
                continue;
            }
            let key = GroupKey {
                passes,
                untapped: of_untapped,
                split,
                resident,
                streams,
                shape,
            };
            let key_hash = key.fx_hash();
            if let Some(id) = self.find_group(key_hash, &key) {
                match of_untapped {
                    true => {
                        untapped_group_ids.insert(index, Some(id));
                    }
                    false => {
                        block_group_ids.insert(index, Some(id));
                    }
                }
            } else {
                if !missing_groups
                    .iter()
                    .any(|pending| pending.key_hash == key_hash && pending.key == key)
                {
                    missing_groups.push(MissingGroup {
                        index,
                        of_untapped,
                        split,
                        resident,
                        streams,
                        key_hash,
                        key: key.clone(),
                    });
                }
                pending_group_lookups.push((index, of_untapped, key_hash, key));
            }
        }
        if !missing_groups.is_empty() {
            self.certify_missing_groups(ctx, missing_groups, passes)?;
            for (index, of_untapped, key_hash, key) in pending_group_lookups {
                let id = self
                    .find_group(key_hash, &key)
                    .expect("group key was certified or cached");
                match of_untapped {
                    true => {
                        untapped_group_ids.insert(index, Some(id));
                    }
                    false => {
                        block_group_ids.insert(index, Some(id));
                    }
                }
            }
        }
        Ok((block_group_ids, untapped_group_ids))
    }

    fn certify_missing_groups(
        &mut self,
        ctx: &CertifyContext<'_>,
        missing_groups: Vec<MissingGroup>,
        passes: Passes,
    ) -> Result<(), GraphError> {
        let group_entries: Vec<(usize, bool, bool, bool)> = missing_groups
            .iter()
            .map(|entry| (entry.index, entry.of_untapped, entry.split, entry.resident))
            .collect();
        let (prefill_tiles, combine_tasks) = collect_run_tasks(ctx, &group_entries, true)?;
        let mut missing_once: Vec<(u64, OnceKey, usize)> = Vec::new();
        let mut shape_once_keys: Vec<Vec<(OnceStage, u64, u64, OnceKey)>> =
            Vec::with_capacity(missing_groups.len());
        for entry in &missing_groups {
            if entry.of_untapped {
                shape_once_keys.push(Vec::new());
                continue;
            }
            let once_shape = entry.key.shape.once_shape();
            let mut keys_for_shape = Vec::new();
            for (stage, rows) in once_graphs(entry.streams, entry.split, passes) {
                let once_key = OnceKey {
                    stage,
                    rows,
                    streams: entry.streams,
                    shape: once_shape.clone(),
                };
                let once_hash = once_key.fx_hash();
                if self.find_once(once_hash, &once_key).is_none()
                    && !missing_once
                        .iter()
                        .any(|(pending_hash, k, _)| *pending_hash == once_hash && *k == once_key)
                {
                    missing_once.push((once_hash, once_key.clone(), entry.index));
                }
                keys_for_shape.push((stage, rows, once_hash, once_key));
            }
            shape_once_keys.push(keys_for_shape);
        }
        enum GroupTask {
            Combine(RunTask),
            Once {
                once_idx: usize,
                block_index: usize,
                stage: OnceStage,
                rows: u64,
            },
        }
        enum GroupTaskResult<L> {
            Combine(usize, Topology, CertifiedBlockGraph<L>, Included),
            Once(usize, CertifiedBlockGraph<L>, Vec<GraphConstant>),
        }
        let mut group_tasks: Vec<GroupTask> =
            Vec::with_capacity(combine_tasks.len() + missing_once.len());
        for task in combine_tasks {
            group_tasks.push(GroupTask::Combine(task));
        }
        for (once_idx, (_, once_key, index)) in missing_once.iter().enumerate() {
            group_tasks.push(GroupTask::Once {
                once_idx,
                block_index: *index,
                stage: once_key.stage,
                rows: once_key.rows,
            });
        }
        let mut group_maps: Vec<ShapeLayouts<L>> = (0..missing_groups.len())
            .map(|_| ShapeLayouts::new())
            .collect();
        let mut certified_once: Vec<Option<(CertifiedBlockGraph<L>, Vec<GraphConstant>)>> =
            (0..missing_once.len()).map(|_| None).collect();
        if !group_tasks.is_empty() {
            let results = run_in_parallel(group_tasks, |task| match task {
                GroupTask::Combine(run_task) => {
                    let (slot_idx, topology, graph, included) =
                        certify_run_task::<L>(ctx, run_task, &prefill_tiles, passes.rows())?;
                    Ok::<_, GraphError>(GroupTaskResult::Combine(slot_idx, topology, graph, included))
                }
                GroupTask::Once { once_idx, block_index, stage, rows } => {
                    let (layout, constants, owned) =
                        certify_once_block::<L>(ctx, *block_index, *stage, *rows)?;
                    Ok::<_, GraphError>(GroupTaskResult::Once(
                        *once_idx,
                        CertifiedBlockGraph { layout, owned },
                        constants,
                    ))
                }
            })?;
            for res in results {
                match res {
                    GroupTaskResult::Combine(slot_idx, topology, graph, included) => {
                        group_maps[slot_idx].insert(topology, (graph, included));
                    }
                    GroupTaskResult::Once(once_idx, graph, constants) => {
                        certified_once[once_idx] = Some((graph, constants));
                    }
                }
            }
        }
        for ((once_hash, once_key, _), certified) in missing_once.into_iter().zip(certified_once) {
            let (graph, constants) = certified.expect("every missing once key was certified");
            self.insert_once(once_hash, once_key, graph, constants);
        }
        for (slot_idx, once_list) in shape_once_keys.into_iter().enumerate() {
            for (stage, rows, once_hash, once_key) in once_list {
                let id = self
                    .find_once(once_hash, &once_key)
                    .expect("once key was certified or cached");
                let (_, graph, constants) = &self.once_layouts[id];
                group_maps[slot_idx].insert(
                    class_topology(once_class(rows, stage)),
                    (graph.clone(), Included::Once(constants.clone())),
                );
            }
        }
        for (entry, graphs) in missing_groups.into_iter().zip(group_maps) {
            self.insert_group(entry.key_hash, entry.key, Arc::new(graphs));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_core(
        &self,
        outline: FamilyOutline,
        load: &ModelLoadPlan,
        geometry: &Decoder,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        passes: Passes,
        ids: ResolvedFamilyIds,
    ) -> Result<CertifiedCore<L>, GraphError> {
        let mut family = CheckedGraphFamilyResources::new();
        for layout in self.entries.values().chain(&self.per_layer) {
            family.include(layout.storage_bytes(), []);
        }
        let distinct_block_pairs: Vec<(usize, usize, Option<usize>, Arc<ShapeLayouts<L>>, Arc<ShapeLayouts<L>>)> =
            ids.distinct_blocks
                .iter()
                .map(|&index| {
                    let launch_id = ids.block_launch_ids[&index];
                    let group_id = ids.block_group_ids[&index];
                    let group_arc = match group_id {
                        Some(id) => Arc::clone(&self.groups[id].1),
                        None => Arc::clone(&self.empty_layouts),
                    };
                    (
                        index,
                        launch_id,
                        group_id,
                        Arc::clone(&self.launches[launch_id].1),
                        group_arc,
                    )
                })
                .collect();
        let distinct_untapped_pairs: Vec<(usize, Arc<ShapeLayouts<L>>, Arc<ShapeLayouts<L>>)> =
            ids.distinct_untapped
                .iter()
                .map(|&index| {
                    let launch_id = ids.untapped_launch_ids[&index];
                    let group_arc = match ids.untapped_group_ids[&index] {
                        Some(id) => Arc::clone(&self.groups[id].1),
                        None => Arc::clone(&self.empty_layouts),
                    };
                    (index, Arc::clone(&self.launches[launch_id].1), group_arc)
                })
                .collect();
        for (_, _, _, launches, groups) in &distinct_block_pairs {
            for (graph, included) in launches.values().chain(groups.values()) {
                let storage = graph.layout.storage_bytes();
                match included {
                    Included::Classes(constants, rows) => family.include_classes(storage, constants, rows)?,
                    Included::Once(constants) => family.include(storage, constants.iter().cloned()),
                }
            }
        }
        let untapped_rows = outline.untapped_rows;
        for (_, launches, groups) in &distinct_untapped_pairs {
            for (graph, included) in launches.values().chain(groups.values()) {
                let Included::Classes(constants, rows) = included else {
                    unreachable!("an untapped shape certifies only run graphs")
                };
                let rows = rows
                    .iter()
                    .copied()
                    .filter(|&rows| rows <= untapped_rows)
                    .collect::<Vec<_>>();
                if !rows.is_empty() {
                    family.include_classes(graph.layout.storage_bytes(), constants, &rows)?;
                }
            }
        }
        let served = block_served(load, geometry, limits, outline.chain_rows, passes)?;
        // Flat index of every distinct certified block graph by `(holder, topology)`.
        let mut by_holder: FxHashMap<(Holder, Topology), &CertifiedBlockGraph<L>> = fx_map();
        let mut holder_maps: FxHashMap<Holder, (&ShapeLayouts<L>, &ShapeLayouts<L>)> = fx_map();
        for (index, _, _, launches, groups) in &distinct_block_pairs {
            let holder = Holder::Block(*index);
            holder_maps.insert(holder, (launches.as_ref(), groups.as_ref()));
            for (&topology, (graph, _)) in launches.iter().chain(groups.iter()) {
                by_holder.insert((holder, topology), graph);
            }
        }
        for (index, launches, groups) in &distinct_untapped_pairs {
            let holder = Holder::Untapped(*index);
            holder_maps.insert(holder, (launches.as_ref(), groups.as_ref()));
            for (&topology, (graph, _)) in launches.iter().chain(groups.iter()) {
                by_holder.insert((holder, topology), graph);
            }
        }
        let constants = family.constant_heap()?;
        let composition = TargetFamilyComposition::compose(
            &outline,
            &served,
            passes,
            load,
            state,
            plan,
            limits,
            &self.entries,
            self.per_layer.as_ref(),
            &by_holder,
            &holder_maps,
            &ids.untapped_holders,
            constants,
        )?;
        #[cfg(debug_assertions)]
        verify_composition_against_enumeration(
            &composition,
            &outline,
            &served,
            passes,
            load,
            state,
            plan,
            limits,
            &self.entries,
            self.per_layer.as_ref(),
            &by_holder,
            &holder_maps,
            &ids.untapped_holders,
            constants,
        )?;
        // The floor group's buffers and the pass's and the selection notice,
        // which the family binds with its constants
        // (`PreparedTargetGraphs::binding_constant_bytes`).
        let mut resources = family.finish()?;
        let notice = notice_bytes(notice_rows(load, outline.largest_rows))?;
        let group_rows = GroupRows::of(load, geometry, plan, outline.tap_width, outline.largest_rows);
        let floor = group_rows.extents(&served, 1);
        let (group, pass_bytes) = (floor.bytes()?, floor.pass_bytes()?);
        resources.binding_constant_bytes = resources
            .binding_constant_bytes
            .checked_add(group)
            .and_then(|bytes| bytes.checked_add(pass_bytes))
            .and_then(|bytes| bytes.checked_add(notice))
            .ok_or("target binding charge overflows")?;
        Ok(CertifiedCore {
            outline,
            served,
            group: group_rows,
            resources,
            graphs: composition.graphs,
            sealed: composition.sealed,
            upload_regions: composition.upload_regions,
            passes,
            distinct_blocks: distinct_block_pairs,
            distinct_untapped: distinct_untapped_pairs,
            untapped_holders: ids.untapped_holders,
        })
    }
}

/// Single-owner closed-form composition of a target family's sealed-graph
/// host heap (`PlannedSealedGraphs`), launch-graph census (`LaunchGraphs`),
/// and workspace upload regions (`Vec<u64>`), shared by both load
/// reservation (`NativeGraphLayout`) and assessment (`SizingOutline`).
struct TargetFamilyComposition {
    sealed: PlannedSealedGraphs,
    graphs: LaunchGraphs,
    upload_regions: Vec<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Graphed {
    Entry(EntryTokens),
    PerLayer,
    Block(usize, Topology),
    Untapped(usize, Topology),
}

fn seal_many<L: Certified>(planned: &mut PlannedHeap, layout: &L, count: u64) -> Result<(), GraphError> {
    let n = usize::try_from(count).map_err(|_| GraphError::from("planned sealed graph count overflows"))?;
    planned.graphs = planned
        .graphs
        .checked_add(n)
        .ok_or("planned sealed graph count overflows")?;
    planned.plans = planned
        .plans
        .plus(
            layout
                .plan_heap()
                .times(count)
                .ok_or("planned sealed graph heap overflows")?,
        )
        .ok_or("planned sealed graph heap overflows")?;
    Ok(())
}

fn seal_block_many<L: Certified>(
    planned: &mut PlannedHeap,
    certified: &CertifiedBlockGraph<L>,
    count: u64,
) -> Result<(), GraphError> {
    seal_many(planned, &certified.layout, count)?;
    planned.owned = planned
        .owned
        .plus(
            certified
                .owned
                .times(count)
                .ok_or("planned prepared graph heap overflows")?,
        )
        .ok_or("planned prepared graph heap overflows")?;
    Ok(())
}

fn add_count(counts: &mut GraphCounts, stage: Stage, n: usize) {
    match stage {
        Stage::Once(OnceStage::Pass) => counts.passes += n,
        Stage::Run(_) | Stage::Once(OnceStage::Fill) => counts.runs += n,
    }
}

impl TargetFamilyComposition {
    #[allow(clippy::too_many_arguments)]
    fn compose<L: Certified>(
        outline: &FamilyOutline,
        served: &[BlockServed],
        passes: Passes,
        load: &ModelLoadPlan,
        state: &StateResourcePlan,
        plan: &TargetProgramPlan,
        limits: ResourceLimits,
        entries: &BTreeMap<EntryTokens, L>,
        per_layer: Option<&L>,
        by_holder: &FxHashMap<(Holder, Topology), &CertifiedBlockGraph<L>>,
        holder_maps: &FxHashMap<Holder, (&ShapeLayouts<L>, &ShapeLayouts<L>)>,
        untapped_holders: &[Option<Holder>],
        constants: HostHeap,
    ) -> Result<Self, GraphError> {
        let FamilyOutline {
            ref row_classes,
            max_slots,
            max_segments,
            ref keeps,
            untapped_rows,
            ..
        } = *outline;
        let blocks = plan.blocks();
        let block_count = blocks.len();
        let (prefill_listings, controls) = block_listings_and_controls(outline, state, plan)?;
        let empty_listings = BlockListings::default();

        // Determine per-block sharing under UnreadScales::Distinct (at_most) and
        // UnreadScales::Equal (at_least) in O(B) using `outline.first_of_shape`
        // and `untapped_holders` (avoiding O(B^2) `BlockGraphShape` comparisons).
        let units: Vec<bool> = (0..block_count)
            .map(|index| {
                !crate::native::output_tails(blocks[index])
                    .into_iter()
                    .any(crate::native::reads_layer_scale)
            })
            .collect();
        let mut seals_own_most = vec![false; block_count];
        let mut seals_own_least = vec![false; block_count];
        let mut has_unit_block = vec![false; block_count];
        for index in 0..block_count {
            let first = outline.first_of_shape(index);
            seals_own_least[index] = first == index;
            if !units[index] {
                seals_own_most[index] = true;
            } else {
                seals_own_most[index] = !has_unit_block[first];
                has_unit_block[first] = true;
            }
        }
        let mut seals_untapped_most = vec![false; block_count];
        let mut seals_untapped_least = vec![false; block_count];
        let mut seen_unit_untapped_block = vec![false; block_count];
        let mut seen_unit_untapped_own = vec![false; block_count];
        for index in 0..block_count {
            let Some(holder) = untapped_holders[index] else {
                continue;
            };
            seals_untapped_least[index] = holder == Holder::Untapped(index);
            if !units[index] {
                seals_untapped_most[index] = true;
            } else {
                let (shared_with_block, seen) = match holder {
                    Holder::Block(b) => (has_unit_block[b], &mut seen_unit_untapped_block[b]),
                    Holder::Untapped(u) => (false, &mut seen_unit_untapped_own[u]),
                };
                seals_untapped_most[index] = !shared_with_block && !*seen;
                *seen = true;
            }
        }

        let lookup = |holder: Holder, topology: Topology, index: usize, of_untapped: bool| {
            by_holder.get(&(holder, topology)).copied().ok_or_else(|| {
                GraphError::from(format!(
                    "target graph topology {topology:?} of block {index} (untapped: {of_untapped}) has no certified layout"
                ))
            })
        };

        // Intern table for `LaunchGraphs` layouts.
        let mut graphed_table: Vec<Graphed> = Vec::new();
        let mut graphed_ids: FxHashMap<Graphed, u16> = fx_map();
        let mut intern = |graphed: Graphed| -> u16 {
            *graphed_ids.entry(graphed).or_insert_with(|| {
                let id = u16::try_from(graphed_table.len()).expect("distinct graphed layouts exceed u16");
                graphed_table.push(graphed);
                id
            })
        };
        let mut entry_ids: Vec<u16> = Vec::with_capacity(entries.len() + usize::from(per_layer.is_some()));
        for &source in entries.keys() {
            entry_ids.push(intern(Graphed::Entry(source)));
        }
        if per_layer.is_some() {
            entry_ids.push(intern(Graphed::PerLayer));
        }

        // 1. Sealed & bound host heaps (closed-form O(B × R × T) arithmetic).
        let row_count = row_classes.len() as u64;
        let mut planned_most = PlannedHeap::default();
        let mut bound = HostHeap::default();
        for layout in entries.values().chain(per_layer) {
            seal_many(&mut planned_most, layout, row_count)?;
            bound = bound
                .plus(
                    layout
                        .bound_heap()
                        .times(row_count)
                        .ok_or("planned graph binding heap overflows")?,
                )
                .ok_or("planned graph binding heap overflows")?;
        }
        let mut planned_least = PlannedHeap {
            graphs: planned_most.graphs,
            plans: planned_most.plans,
            owned: planned_most.owned,
        };
        let mut sealed_most = GraphCounts::default();
        let mut sealed_least = GraphCounts::default();
        let mut block_bindings = vec![GraphCounts::default(); block_count];
        let mut once_staged: Vec<(u16, OnceStage, u64)> = Vec::new();

        let seg_count = segment_classes(max_segments).count() as u64;
        let scores: Vec<bool> = blocks.iter().map(|&slot| slot_scores(slot)).collect();

        for index in 0..block_count {
            let first = outline.first_of_shape(index);
            let block_holder = Holder::Block(first);
            let u_holder = untapped_holders[index];
            let block_keeps = keeps[index];
            let block_scores = scores[index];
            let history_segs = if block_keeps.history() { seg_count } else { 1 };
            let seal_own_m = seals_own_most[index];
            let seal_own_l = seals_own_least[index];
            let seal_u_m = seals_untapped_most[index];
            let seal_u_l = seals_untapped_least[index];

            for &rows in row_classes {
                let slot_count = if block_keeps.bank() { rows.min(max_slots) } else { 1 };
                let listings = if crate::operators::attention::graph::decodes(rows) {
                    &empty_listings
                } else {
                    &prefill_listings[index]
                };
                let listing_segs = if !block_keeps.history() || listings.tiles.is_empty() {
                    0
                } else if listings.every_segment_count {
                    seg_count
                } else {
                    segment_classes(max_segments)
                        .filter(|&s| listings.segments(s, max_segments) == s)
                        .count() as u64
                };
                let n_plain = history_segs * slot_count;
                let n_scored = if block_keeps.history() && block_scores {
                    history_segs * slot_count
                } else {
                    0
                };
                let n_listed = listing_segs * slot_count * (listings.tiles.len() as u64);

                let form = RowForm::of(rows);
                let stages = block_sealed_stages(served, index, rows);
                let has_untapped = rows <= untapped_rows && u_holder.is_some();

                for &stage in stages {
                    let run_stage = Stage::Run(stage);
                    for (listed, scored, count) in [
                        (false, false, n_plain),
                        (false, true, n_scored),
                        (true, false, n_listed),
                    ] {
                        if count == 0 {
                            continue;
                        }
                        let topology = (form, listed, scored, run_stage, 0);
                        let certified = lookup(block_holder, topology, index, false)?;
                        let count_usize = usize::try_from(count)
                            .map_err(|_| GraphError::from("planned graph count overflows"))?;
                        bound = bound
                            .plus(
                                certified
                                    .layout
                                    .bound_heap()
                                    .times(count)
                                    .ok_or("planned graph binding heap overflows")?,
                            )
                            .ok_or("planned graph binding heap overflows")?;
                        add_count(&mut block_bindings[index], run_stage, count_usize);
                        if seal_own_m {
                            seal_block_many(&mut planned_most, certified, count)?;
                            add_count(&mut sealed_most, run_stage, count_usize);
                        }
                        if seal_own_l {
                            seal_block_many(&mut planned_least, certified, count)?;
                            add_count(&mut sealed_least, run_stage, count_usize);
                        }
                        intern(Graphed::Block(first, topology));

                        if has_untapped {
                            let uh = u_holder.expect("checked is_some");
                            let u_certified = lookup(uh, topology, index, true)?;
                            bound = bound
                                .plus(
                                    u_certified
                                        .layout
                                        .bound_heap()
                                        .times(count)
                                        .ok_or("planned graph binding heap overflows")?,
                                )
                                .ok_or("planned graph binding heap overflows")?;
                            block_bindings[index].untapped += count_usize;
                            if seal_u_m {
                                seal_block_many(&mut planned_most, u_certified, count)?;
                                add_count(&mut sealed_most, run_stage, count_usize);
                            }
                            if seal_u_l {
                                seal_block_many(&mut planned_least, u_certified, count)?;
                                add_count(&mut sealed_least, run_stage, count_usize);
                            }
                            let u_graphed = match uh {
                                Holder::Block(block) => Graphed::Block(outline.first_of_shape(block), topology),
                                Holder::Untapped(first_u) => Graphed::Untapped(first_u, topology),
                            };
                            intern(u_graphed);
                        }
                    }
                }
            }

            for (once_stage, pass_rows) in once_graphs(served[index].streams, served[index].split.is_some(), passes) {
                let stage = Stage::Once(once_stage);
                let topology = class_topology(once_class(pass_rows, once_stage));
                let certified = lookup(block_holder, topology, index, false)?;
                bound = bound
                    .plus(certified.layout.bound_heap())
                    .ok_or("planned graph binding heap overflows")?;
                add_count(&mut block_bindings[index], stage, 1);
                if seal_own_m {
                    seal_block_many(&mut planned_most, certified, 1)?;
                    add_count(&mut sealed_most, stage, 1);
                }
                if seal_own_l {
                    seal_block_many(&mut planned_least, certified, 1)?;
                    add_count(&mut sealed_least, stage, 1);
                }
                let id = intern(Graphed::Block(first, topology));
                once_staged.push((id, once_stage, pass_rows));
            }
        }

        let bound = bound
            .plus(block_binding_storage(&block_bindings)?)
            .ok_or("planned graph binding heap overflows")?;
        let at_most = SealedGraphsHeap {
            graphs: planned_most.graphs,
            plans: planned_most.plans,
            storage: block_graph_storage(&block_bindings, sealed_most)?,
            owned: planned_most.owned,
        };
        let at_least = SealedGraphsHeap {
            graphs: planned_least.graphs,
            plans: planned_least.plans,
            storage: block_graph_storage(&block_bindings, sealed_least)?,
            owned: planned_least.owned,
        };
        let sealed = PlannedSealedGraphs {
            at_most,
            at_least,
            bound,
            constants,
        };

        // 2. Upload regions (cached per distinct (first, untapped_holder, is_split)).
        let upload = |layout: &L| layout.storage_bytes().upload;
        let largest = |layouts: &mut dyn Iterator<Item = &L>| {
            layouts.map(upload).max().filter(|&bytes| bytes > 0)
        };
        let mut upload_regions = Vec::new();
        if let Some(bytes) = largest(&mut entries.values()) {
            upload_regions.push(bytes);
        }
        if let Some(bytes) = largest(&mut per_layer.into_iter()) {
            upload_regions.push(bytes);
        }
        upload_regions.push(controls.region_bytes());
        let mut region_cache: FxHashMap<(usize, Option<Holder>, bool), Vec<u64>> = fx_map();
        for index in 0..block_count {
            let first = outline.first_of_shape(index);
            let u_holder = untapped_holders[index];
            let is_split = served[index].split.is_some();
            let regions = region_cache
                .entry((first, u_holder, is_split))
                .or_insert_with(|| {
                    let (b_launches, b_groups) = holder_maps[&Holder::Block(first)];
                    let u_maps = u_holder.map(|uh| holder_maps[&uh]);
                    block_upload_regions(
                        row_classes.iter().copied(),
                        |rows| block_stages(served, index, rows),
                        |rows, stage| {
                            let form = RowForm::of(rows);
                            let block_iter = b_launches.iter().chain(b_groups.iter());
                            let untapped_iter = u_maps
                                .filter(|_| rows <= untapped_rows)
                                .into_iter()
                                .flat_map(|(ul, ug)| ul.iter().chain(ug.iter()));
                            block_iter
                                .chain(untapped_iter)
                                .filter(|(&(f, _, _, of, _), _)| {
                                    f == form && matches!(of, Stage::Run(of) if of.place() == stage)
                                })
                                .map(|(_, (graph, _))| upload(&graph.layout))
                                .max()
                                .unwrap_or(0)
                        },
                    )
                });
            upload_regions.extend_from_slice(regions);
        }

        // 3. LaunchGraphs (`most`, `decode`, `chained`), grouped by distinct block
        // equivalence classes and evaluated over exact `LaunchClassSignature`s.
        let windows = load.expert_bank().map_or(1, crate::ExpertBankPlan::windows);
        let single_pass = passes.of_chain(1);

        struct BlockGroup<'a> {
            first: usize,
            u_holder: Option<Holder>,
            keeps: BlockKeeps,
            scores: bool,
            streams: bool,
            resident: bool,
            is_split: bool,
            listings: &'a BlockListings,
            count: u64,
            once_runs: Vec<(u16, u64)>,
        }

        let mut block_groups: Vec<BlockGroup<'_>> = Vec::new();
        for index in 0..block_count {
            let first = outline.first_of_shape(index);
            let u_holder = untapped_holders[index];
            let block_keeps = keeps[index];
            let block_scores = scores[index];
            let streams = served[index].streams;
            let resident = served[index].resident;
            let is_split = served[index].split.is_some();
            let listings = &prefill_listings[index];

            if let Some(group) = block_groups.iter_mut().find(|g| {
                g.first == first
                    && g.u_holder == u_holder
                    && g.keeps == block_keeps
                    && g.scores == block_scores
                    && g.streams == streams
                    && g.resident == resident
                    && g.is_split == is_split
                    && g.listings == listings
            }) {
                group.count += 1;
            } else {
                let once_runs = once_graphs(streams, is_split, passes)
                    .into_iter()
                    .map(|(once_stage, pass_rows)| {
                        let topology = class_topology(once_class(pass_rows, once_stage));
                        let id = graphed_ids[&Graphed::Block(first, topology)];
                        let repeat = if streams && (once_stage == OnceStage::Fill || pass_rows == single_pass) {
                            windows
                        } else {
                            1
                        };
                        (id, repeat)
                    })
                    .collect();
                block_groups.push(BlockGroup {
                    first,
                    u_holder,
                    keeps: block_keeps,
                    scores: block_scores,
                    streams,
                    resident,
                    is_split,
                    listings,
                    count: 1,
                    once_runs,
                });
            }
        }

        let mut seg_pairs: Vec<(bool, bool)> = Vec::with_capacity(3);
        for s in segment_classes(max_segments) {
            let pair = (s == 1, s == max_segments);
            if !seg_pairs.contains(&pair) {
                seg_pairs.push(pair);
            }
        }

        let k = graphed_table.len();
        let mut most = vec![0u64; k];
        let mut decode = vec![0u64; k];
        let mut runs = vec![0u64; k];
        let mut sigs: FxHashSet<(bool, bool, bool, u64, bool)> = fx_set();

        for &rows in row_classes {
            let is_decode_launch = decode_launch((rows, 1, 1, 0, 0), limits);
            let is_decode_rows = crate::operators::attention::graph::decodes(rows);
            let form = RowForm::of(rows);

            sigs.clear();
            for g in &block_groups {
                let max_slot = if g.keeps.bank() { rows.min(max_slots) } else { 1 };
                let slot_flags: &[bool] = if max_slot >= 2 { &[true, false] } else { &[true] };
                let listings = if is_decode_rows { &empty_listings } else { g.listings };
                if g.keeps.history() {
                    for &(first_seg, max_seg) in &seg_pairs {
                        let lists = !listings.tiles.is_empty() && (listings.every_segment_count || max_seg);
                        for &first_slot in slot_flags {
                            sigs.insert((first_seg, max_seg, first_slot, 0, false));
                            if g.scores {
                                sigs.insert((first_seg, max_seg, first_slot, 0, true));
                            }
                            if lists {
                                for &tile in &listings.tiles {
                                    sigs.insert((first_seg, max_seg, first_slot, tile, false));
                                }
                            }
                        }
                    }
                } else {
                    for &first_slot in slot_flags {
                        sigs.insert((true, max_segments == 1, first_slot, 0, false));
                    }
                }
            }

            for &(first_seg, max_seg, first_slot, listed, scored) in &sigs {
                runs.fill(0);
                for &id in &entry_ids {
                    runs[id as usize] += 1;
                }
                for g in &block_groups {
                    if RunStage::served(g.is_split, rows).contains(&RunStage::Front) {
                        for &(id, repeat) in &g.once_runs {
                            runs[id as usize] += repeat * g.count;
                        }
                    }
                    let max_slot = if g.keeps.bank() { rows.min(max_slots) } else { 1 };
                    if !first_slot && max_slot < 2 {
                        continue;
                    }
                    let listings = if is_decode_rows { &empty_listings } else { g.listings };
                    let matches = if g.keeps.history() {
                        if listed == 0 && !scored {
                            true
                        } else if listed == 0 && scored {
                            g.scores
                        } else if listed > 0 && !scored {
                            (listings.every_segment_count || max_seg) && listings.tiles.contains(&listed)
                        } else {
                            false
                        }
                    } else {
                        first_seg && listed == 0 && !scored
                    };
                    if !matches {
                        continue;
                    }
                    for &stage in RunStage::sealed(g.is_split, g.resident, rows) {
                        let topo = (form, listed > 0, scored, Stage::Run(stage), 0);
                        let id = graphed_ids[&Graphed::Block(g.first, topo)];
                        runs[id as usize] += g.count;
                        if rows <= untapped_rows {
                            if let Some(uh) = g.u_holder {
                                let u_graphed = match uh {
                                    Holder::Block(block) => Graphed::Block(outline.first_of_shape(block), topo),
                                    Holder::Untapped(first_u) => Graphed::Untapped(first_u, topo),
                                };
                                let u_id = graphed_ids[&u_graphed];
                                runs[u_id as usize] += g.count;
                            }
                        }
                    }
                }
                for i in 0..k {
                    let count = runs[i];
                    if count > most[i] {
                        most[i] = count;
                    }
                    if is_decode_launch && count > decode[i] {
                        decode[i] = count;
                    }
                }
            }
        }

        let overflow = || GraphError::from("a target family's graph census overflows");
        let recorded = |graphed: Graphed| -> Result<GraphCensus, GraphError> {
            let layout = match graphed {
                Graphed::Entry(source) => &entries[&source],
                Graphed::PerLayer => per_layer.ok_or("a per-layer run has no layout")?,
                Graphed::Block(index, topology) => {
                    &by_holder
                        .get(&(Holder::Block(index), topology))
                        .ok_or_else(|| {
                            format!("target graph topology {topology:?} of block {index} has no certified layout")
                        })?
                        .layout
                }
                Graphed::Untapped(index, topology) => {
                    &by_holder
                        .get(&(Holder::Untapped(index), topology))
                        .ok_or("an untapped block class has no certified layout")?
                        .layout
                }
            };
            Ok(layout.storage_bytes().recorded)
        };
        let chain = |launches: usize| {
            let further = chained_once_runs(passes, launches, windows);
            let mut runs = vec![0u64; graphed_table.len()];
            for &(id, once, rows) in &once_staged {
                if let Some(&(_, count)) = further.iter().find(|&&(key, _)| key == (once, rows)) {
                    runs[id as usize] += count;
                }
            }
            runs.iter().enumerate().filter(|&(_, &count)| count > 0).try_fold(
                GraphCensus::default(),
                |census, (id, &count)| {
                    census
                        .plus(recorded(graphed_table[id])?.times(count).ok_or_else(overflow)?)
                        .ok_or_else(overflow)
                },
            )
        };
        let chained = match outline.backend.graphs_update_in_flight() {
            true => Vec::new(),
            false => (2..=limits.chain_capacity.get())
                .map(chain)
                .collect::<Result<Vec<_>, _>>()?,
        };
        if decode.iter().all(|&count| count == 0) {
            return Err("a target family has no decode launch class".into());
        }
        let mut order: Vec<usize> = (0..graphed_table.len()).filter(|&i| most[i] > 0).collect();
        order.sort_unstable_by_key(|&i| graphed_table[i]);
        let graphs = order
            .into_iter()
            .map(|i| {
                Ok(LaunchedGraph {
                    graph: recorded(graphed_table[i])?,
                    pinned_runs: decode[i],
                    runs: most[i],
                })
            })
            .collect::<Result<Vec<_>, GraphError>>()?;
        let graphs = LaunchGraphs::of(graphs, chained).ok_or_else(overflow)?;
        Ok(Self {
            sealed,
            graphs,
            upload_regions,
        })
    }
}

/// Debug-only independent verification that `TargetFamilyComposition`'s
/// closed-form `(sealed, graphs, upload_regions)` matches the full
/// `enumerate_block_graphs` tuple enumeration bit-for-bit.
#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
fn verify_composition_against_enumeration<L: Certified>(
    composition: &TargetFamilyComposition,
    outline: &FamilyOutline,
    served: &[BlockServed],
    passes: Passes,
    load: &ModelLoadPlan,
    state: &StateResourcePlan,
    plan: &TargetProgramPlan,
    limits: ResourceLimits,
    entries: &BTreeMap<EntryTokens, L>,
    per_layer: Option<&L>,
    by_holder: &FxHashMap<(Holder, Topology), &CertifiedBlockGraph<L>>,
    holder_maps: &FxHashMap<Holder, (&ShapeLayouts<L>, &ShapeLayouts<L>)>,
    untapped_holders: &[Option<Holder>],
    constants: HostHeap,
) -> Result<(), GraphError> {
    let certified_of = |index: usize, class: StagedClass, of_untapped: bool| -> Result<&CertifiedBlockGraph<L>, GraphError> {
        let holder = match of_untapped {
            true => untapped_holders[index],
            false => Some(Holder::Block(outline.first_of_shape(index))),
        };
        holder
            .and_then(|holder| by_holder.get(&(holder, class_topology(class))).copied())
            .ok_or_else(|| {
                GraphError::from(format!(
                    "target graph class {class:?} of block {index} (untapped: {of_untapped}) has no certified layout"
                ))
            })
    };
    let enumerate = |unread| {
        enumerate_block_graphs(outline, served, passes, state, plan, BlockOutputScales::Planned(unread))
    };
    let enumeration = enumerate(UnreadScales::Distinct)?;
    let least = match enumeration.exact {
        true => None,
        false => Some(enumerate(UnreadScales::Equal)?),
    };
    let count = |enumeration: &BlockGraphEnumeration| {
        let mut planned = PlannedHeap::default();
        for _ in &outline.row_classes {
            for layout in entries.values().chain(per_layer) {
                planned.seal(layout)?;
            }
        }
        let mut sealed = GraphCounts::default();
        let sealing = enumeration
            .sealing
            .iter()
            .map(|&(index, class)| (index, class, false))
            .chain(
                enumeration
                    .untapped_sealing
                    .iter()
                    .map(|&(index, class)| (index, class, true)),
            );
        for (index, class, of_untapped) in sealing {
            planned.seal_block(certified_of(index, class, of_untapped)?)?;
            sealed.count(class.1);
        }
        Ok::<_, GraphError>((planned, sealed))
    };
    let (planned, sealed_blocks) = count(&enumeration)?;
    let mut bound = HostHeap::default();
    let mut binding = |layout: &L| {
        bound = bound
            .plus(layout.bound_heap())
            .ok_or("planned graph binding heap overflows")?;
        Ok::<_, GraphError>(())
    };
    for _ in &outline.row_classes {
        for layout in entries.values().chain(per_layer) {
            binding(layout)?;
        }
    }
    let mut block_bindings = vec![GraphCounts::default(); outline.shapes.len()];
    let bound_classes = enumeration
        .sealing
        .iter()
        .copied()
        .chain(enumeration.sharing.iter().map(|&(index, class, _)| (index, class)))
        .map(|(index, class)| (index, class, false))
        .chain(
            enumeration
                .untapped_sealing
                .iter()
                .copied()
                .chain(enumeration.untapped_sharing.iter().map(|&(index, class, _)| (index, class)))
                .map(|(index, class)| (index, class, true)),
        );
    for (index, class, of_untapped) in bound_classes {
        binding(&certified_of(index, class, of_untapped)?.layout)?;
        match of_untapped {
            true => block_bindings[index].untapped += 1,
            false => block_bindings[index].count(class.1),
        }
    }
    let bound = bound
        .plus(block_binding_storage(&block_bindings)?)
        .ok_or("planned graph binding heap overflows")?;
    let graphs_heap = |(planned, sealed_blocks): (PlannedHeap, GraphCounts)| {
        Ok::<_, GraphError>(SealedGraphsHeap {
            graphs: planned.graphs,
            plans: planned.plans,
            storage: block_graph_storage(&block_bindings, sealed_blocks)?,
            owned: planned.owned,
        })
    };
    let at_most = graphs_heap((planned, sealed_blocks))?;
    let at_least = match &least {
        None => at_most,
        Some(least) => graphs_heap(count(least)?)?,
    };
    let ref_sealed = PlannedSealedGraphs {
        at_most,
        at_least,
        bound,
        constants,
    };
    debug_assert_eq!(
        composition.sealed, ref_sealed,
        "TargetFamilyComposition sealed diverged from enumerate_block_graphs"
    );

    let upload = |layout: &L| layout.storage_bytes().upload;
    let largest = |layouts: &mut dyn Iterator<Item = &L>| {
        layouts.map(upload).max().filter(|&bytes| bytes > 0)
    };
    let ref_upload_regions: Vec<u64> = largest(&mut entries.values())
        .into_iter()
        .chain(largest(&mut per_layer.into_iter()))
        .chain([enumeration.controls.region_bytes()])
        .chain((0..outline.shapes.len()).flat_map(|index| {
            let first = outline.first_of_shape(index);
            let (b_launches, b_groups) = holder_maps[&Holder::Block(first)];
            let u_maps = untapped_holders[index].map(|uh| holder_maps[&uh]);
            block_upload_regions(
                outline.row_classes.iter().copied(),
                |rows| block_stages(served, index, rows),
                |rows, stage| {
                    let form = RowForm::of(rows);
                    let block_iter = b_launches.iter().chain(b_groups.iter());
                    let untapped_iter = u_maps
                        .filter(|_| rows <= outline.untapped_rows)
                        .into_iter()
                        .flat_map(|(ul, ug)| ul.iter().chain(ug.iter()));
                    block_iter
                        .chain(untapped_iter)
                        .filter(|(&(f, _, _, of, _), _)| {
                            f == form && matches!(of, Stage::Run(of) if of.place() == stage)
                        })
                        .map(|(_, (graph, _))| upload(&graph.layout))
                        .max()
                        .unwrap_or(0)
                },
            )
        }))
        .collect();
    debug_assert_eq!(
        composition.upload_regions, ref_upload_regions,
        "TargetFamilyComposition upload_regions diverged from enumerate_block_graphs"
    );

    let mut graphed_table: Vec<Graphed> = Vec::new();
    let mut graphed_ids: HashMap<Graphed, u16> = HashMap::new();
    let mut intern = |graphed: Graphed| -> u16 {
        *graphed_ids.entry(graphed).or_insert_with(|| {
            let id = u16::try_from(graphed_table.len()).expect("distinct graphed layouts exceed u16");
            graphed_table.push(graphed);
            id
        })
    };
    let mut entry_ids = Vec::new();
    for &source in entries.keys() {
        entry_ids.push(intern(Graphed::Entry(source)));
    }
    if per_layer.is_some() {
        entry_ids.push(intern(Graphed::PerLayer));
    }
    let mut once_staged: Vec<(u16, OnceStage, u64)> = Vec::new();
    let mut run_entries: Vec<(usize, StagedClass, u16)> = Vec::new();
    for (index, staged) in enumeration
        .sealing
        .iter()
        .copied()
        .chain(enumeration.sharing.iter().map(|&(index, class, _)| (index, class)))
    {
        let id = intern(Graphed::Block(outline.first_of_shape(index), class_topology(staged)));
        if let Stage::Once(once) = staged.1 {
            once_staged.push((id, once, staged.0.0));
        }
        run_entries.push((index, staged, id));
    }
    for (index, staged) in enumeration
        .untapped_sealing
        .iter()
        .copied()
        .chain(enumeration.untapped_sharing.iter().map(|&(index, class, _)| (index, class)))
    {
        let topology = class_topology(staged);
        let graphed = match untapped_holders[index].expect("an untapped graph is a tapped block's") {
            Holder::Block(block) => Graphed::Block(outline.first_of_shape(block), topology),
            Holder::Untapped(first) => Graphed::Untapped(first, topology),
        };
        let id = intern(graphed);
        run_entries.push((index, staged, id));
    }
    let launch_runs = LaunchRuns::of(served, run_entries);
    let windows = load.expert_bank().map_or(1, crate::ExpertBankPlan::windows);
    let k = graphed_table.len();
    let mut most = vec![0u64; k];
    let mut decode = vec![0u64; k];
    let mut runs = vec![0u64; k];
    for &class in &enumeration.launch_classes {
        runs.fill(0);
        for &id in &entry_ids {
            runs[id as usize] += 1;
        }
        for &id in launch_runs.of_class(class, passes, windows) {
            runs[id as usize] += 1;
        }
        let is_decode = decode_launch(class, limits);
        for i in 0..k {
            let count = runs[i];
            if count > most[i] {
                most[i] = count;
            }
            if is_decode && count > decode[i] {
                decode[i] = count;
            }
        }
    }
    let overflow = || GraphError::from("a target family's graph census overflows");
    let recorded = |graphed: Graphed| -> Result<GraphCensus, GraphError> {
        let layout = match graphed {
            Graphed::Entry(source) => &entries[&source],
            Graphed::PerLayer => per_layer.ok_or("a per-layer run has no layout")?,
            Graphed::Block(index, topology) => &by_holder[&(Holder::Block(index), topology)].layout,
            Graphed::Untapped(index, topology) => &by_holder[&(Holder::Untapped(index), topology)].layout,
        };
        Ok(layout.storage_bytes().recorded)
    };
    let chain = |launches: usize| {
        let further = chained_once_runs(passes, launches, windows);
        let mut runs = vec![0u64; graphed_table.len()];
        for &(id, once, rows) in &once_staged {
            if let Some(&(_, count)) = further.iter().find(|&&(key, _)| key == (once, rows)) {
                runs[id as usize] += count;
            }
        }
        runs.iter().enumerate().filter(|&(_, &count)| count > 0).try_fold(
            GraphCensus::default(),
            |census, (id, &count)| {
                census
                    .plus(recorded(graphed_table[id])?.times(count).ok_or_else(overflow)?)
                    .ok_or_else(overflow)
            },
        )
    };
    let chained = match outline.backend.graphs_update_in_flight() {
        true => Vec::new(),
        false => (2..=limits.chain_capacity.get())
            .map(chain)
            .collect::<Result<Vec<_>, _>>()?,
    };
    let mut order: Vec<usize> = (0..graphed_table.len()).filter(|&i| most[i] > 0).collect();
    order.sort_unstable_by_key(|&i| graphed_table[i]);
    let ref_graphs = order
        .into_iter()
        .map(|i| {
            Ok(LaunchedGraph {
                graph: recorded(graphed_table[i])?,
                pinned_runs: decode[i],
                runs: most[i],
            })
        })
        .collect::<Result<Vec<_>, GraphError>>()?;
    let ref_graphs = LaunchGraphs::of(ref_graphs, chained).ok_or_else(overflow)?;
    debug_assert_eq!(
        composition.graphs, ref_graphs,
        "TargetFamilyComposition graphs diverged from enumerate_block_graphs"
    );
    Ok(())
}

/// The passes a group runs: of the most whole launches, of at most
/// `chain`'s, whose pass needs no more workspace (`pass_workspace` of block
/// and rows) at any of the routed `blocks` than `arena`, the workspace the
/// launches' own graphs need. A merge set of more runs splits into passes
/// of these rows, so the pass's grouping and expert scratch share the
/// arena's bytes and add none. A pass needs more workspace with more rows;
/// one launch's pass is the least a group runs, as a launch of its own
/// would.
fn group_pass_rows(
    blocks: &[usize],
    geometry: &Decoder,
    chain: Passes,
    arena: u64,
    mut pass_workspace: impl FnMut(usize, u64) -> Result<u64, GraphError>,
) -> Result<Passes, GraphError> {
    let mut passes = chain;
    for &index in blocks {
        if block_routed(geometry, index)?.is_none() {
            continue;
        }
        // The most launches that fit: at least one, at most those found.
        let (mut fits, mut over) = (1, passes.launches + 1);
        while over - fits > 1 {
            let middle = (fits + over) / 2;
            match pass_workspace(index, passes.of_chain(middle))? <= arena {
                true => fits = middle,
                false => over = middle,
            }
        }
        passes.launches = fits;
    }
    Ok(passes)
}
