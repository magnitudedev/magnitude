//! Seismic graph slots for numerical execution and owned import sources.

mod graph;
pub(crate) use graph::{IdleSlots, Outputs, VisionGrowth};
pub use graph::{
    GraphOutputOwner, GraphOutputTensor, NativeGraphOutputLease, NativeGraphPool,
    NativeGraphWorkspaceLease, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
    VisionGraphPool,
};

use crate::Stored;
use crate::programs::graph::readout;
use crate::{
    ExecutionPlan, InvariantError, PreparedDrafterGraphs, PreparedStateCopyGraphs,
    PreparedTargetGraphs, PreparedTargetReadoutGraphs, PreparedVisionGraphs, ResourceDomainId,
    WeightPlan,
};
use magnitude_artifacts::FileSource;
use magnitude_batching::LaunchClass;
use seismic::{BackendName, Device, HostRegion, ReadOnlyMappedRegion, Tensor};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolClass {
    Target(LaunchClass),
    Head(LaunchClass),
    Vision { patch_rows: usize },
    Import { bytes: u64 },
}

/// One plan-backed source tensor owned through import completion. On Metal it
/// views an immutable artifact mapping; other backends own a staged upload.
pub struct ImportWorkspaceLease {
    domain: ResourceDomainId,
    class: PoolClass,
    bytes: u64,
    source: ImportSource,
}

/// The tensor one import entry reads a weight from: a view of the mapped
/// artifact, or an upload its importer fills.
pub(crate) enum ImportSource {
    Mapped(Tensor),
    Staged(Tensor),
}

impl ImportSource {
    /// Whether `weight`'s import reads `stored` in place on `device`: the
    /// device reads mapped pages ([`ImportWindow::serves`]), the weight is
    /// uploaded as stored, and its bytes in the file keep its source's
    /// alignment (a GGUF tensor starts at a multiple of its file's
    /// `general.alignment`, any power of two; a window maps whole pages, so
    /// its view of the bytes is aligned as their file offset is).
    pub(crate) fn reads_in_place(
        weight: &WeightPlan,
        stored: &Stored,
        device: &Device,
    ) -> Result<bool, AllocationError> {
        Ok(ImportWindow::serves(device)
            && !weight.host_prepared()
            && ReadOnlyMappedRegion::keeps_alignment(
                weight.source,
                &[source_count(weight)?],
                stored.file_range().1,
            )
            .map_err(AllocationError::Tensor)?)
    }

    /// The source of `weight`'s import on `device`, the one way every
    /// import takes it, a load's and a tuning unit's alike. A weight read
    /// in place ([`Self::reads_in_place`]) is a view of `window` where one
    /// is given: the artifact's own pages, which the process's footprint
    /// does not count. Any other is an upload of the weight's source bytes
    /// on the device, not yet written
    /// ([`crate::programs::native_import::fill_staged_source`] writes it).
    pub(crate) fn acquire(
        weight: &WeightPlan,
        stored: &Stored,
        window: Option<&ImportWindow>,
        device: &Device,
    ) -> Result<Self, AllocationError> {
        let count = source_count(weight)?;
        match window {
            Some(window) if Self::reads_in_place(weight, stored, device)? => {
                Ok(Self::Mapped(window.tensor(stored, weight, count)?))
            }
            // SAFETY: the importer fills this exact tensor before binding
            // it as a kernel source.
            _ => unsafe { Self::upload(weight, device) }.map(Self::Staged),
        }
    }

    /// An upload of `weight`'s source bytes on `device`, not yet written.
    ///
    /// # Safety
    /// The caller fills it
    /// ([`crate::programs::native_import::fill_staged_source`]) before
    /// binding it as a kernel source.
    pub(crate) unsafe fn upload(
        weight: &WeightPlan,
        device: &Device,
    ) -> Result<Tensor, AllocationError> {
        unsafe { Tensor::uninitialized(device, weight.upload, &[source_count(weight)?]) }
            .map_err(AllocationError::Tensor)
    }

    /// An upload of `weight`'s source bytes on `device`, filled by a read of
    /// `artifact` around the file cache: what an expert bank's stacks are
    /// imported from, a load's or a tuning unit's: none of their bytes stay
    /// in it.
    pub(crate) fn uncached(
        weight: &WeightPlan,
        artifact: &magnitude_artifacts::gguf::GgufArtifact,
        device: &Device,
    ) -> Result<Tensor, String> {
        let stored = Stored::from_gguf(artifact, &weight.descriptor).map_err(|error| error.to_string())?;
        // SAFETY: filled here, before any entry binds it.
        let mut upload = unsafe { Self::upload(weight, device) }.map_err(|error| error.to_string())?;
        crate::programs::native_import::fill_staged_source(
            &stored,
            weight,
            &mut upload,
            seismic_io::CachePolicy::Uncached,
        )
        .map_err(|error| error.to_string())?;
        Ok(upload)
    }

    pub(crate) fn tensor(&self) -> &Tensor {
        match self {
            Self::Mapped(tensor) | Self::Staged(tensor) => tensor,
        }
    }

    pub(crate) fn staged_mut(&mut self) -> Option<&mut Tensor> {
        match self {
            Self::Mapped(_) => None,
            Self::Staged(tensor) => Some(tensor),
        }
    }

    pub(crate) fn into_tensor(self) -> Tensor {
        match self {
            Self::Mapped(tensor) | Self::Staged(tensor) => tensor,
        }
    }
}

/// The elements of `weight`'s source (and of its upload).
fn source_count(weight: &WeightPlan) -> Result<u64, AllocationError> {
    crate::planning::elements(&weight.shape).ok_or_else(|| {
        AllocationError::Plan(InvariantError {
            context: "import source",
            detail: "weight element count overflows".into(),
        })
    })
}

/// One page-rounded artifact range charged once and shared by the source
/// tensor views of an ordered import batch.
pub(crate) struct ImportWindow {
    source: Arc<FileSource>,
    start: u64,
    end: u64,
    data_offset: u64,
    region: ReadOnlyMappedRegion,
}

impl ImportWindow {
    /// The window of exactly `stored`'s range in its artifact.
    pub(crate) fn of(device: &Device, stored: &Stored) -> Result<Self, AllocationError> {
        let (file, start, end) = stored_range(stored)?;
        Self::map(device, file, start, end)
    }
}

/// `stored`'s file and the range of it the tensor occupies.
fn stored_range(stored: &Stored) -> Result<(Arc<FileSource>, u64, u64), AllocationError> {
    let (file, offset, length) = stored.file_range();
    let end = offset.checked_add(length).ok_or_else(|| {
        AllocationError::Plan(InvariantError {
            context: "import window",
            detail: "source range overflows".into(),
        })
    })?;
    Ok((file.clone(), offset, end))
}

impl ImportWindow {
    /// Whether `device` reads an artifact's mapped pages in place: where it
    /// does, a weight uploaded as stored whose bytes there keep its source's
    /// alignment is imported from a window and nowhere else; any other is
    /// uploaded ([`ImportSource::reads_in_place`]).
    pub(crate) fn serves(device: &Device) -> bool {
        cfg!(unix) && device.backend() == BackendName::Metal
    }

    /// The page-rounded range `start..end` of `source`, mapped for `device`.
    #[cfg(unix)]
    pub(crate) fn map(
        device: &Device,
        source: Arc<FileSource>,
        start: u64,
        end: u64,
    ) -> Result<Self, AllocationError> {
        let invalid = |detail: &str| {
            AllocationError::Plan(InvariantError {
                context: "import window",
                detail: detail.into(),
            })
        };
        if device.backend() != BackendName::Metal || end <= start {
            return Err(invalid(
                "a mapped window requires Metal and a nonempty range",
            ));
        }
        let window = source
            .map_window(start, end - start)
            .map_err(|error| AllocationError::Mapping(error.to_string()))?;
        let pointer = std::ptr::NonNull::new(window.as_ref().as_ref().as_ptr() as *mut u8)
            .ok_or_else(|| invalid("mapped artifact window has no address"))?;
        // SAFETY: the window owns the immutable mapped range through every
        // device view and every in-flight submission using it.
        let host = unsafe { HostRegion::new(pointer, window.mapped_len(), window.clone()) };
        let region = ReadOnlyMappedRegion::new(device, host).map_err(AllocationError::Tensor)?;
        Ok(Self {
            source,
            start,
            end,
            data_offset: window.data_offset() as u64,
            region,
        })
    }

    #[cfg(not(unix))]
    pub(crate) fn map(
        _device: &Device,
        _source: Arc<FileSource>,
        _start: u64,
        _end: u64,
    ) -> Result<Self, AllocationError> {
        Err(AllocationError::Mapping(
            "mapped import windows require a Unix host".into(),
        ))
    }

    pub(crate) fn tensor(
        &self,
        stored: &Stored,
        weight: &WeightPlan,
        count: u64,
    ) -> Result<Tensor, AllocationError> {
        let (source, offset, length) = stored.file_range();
        let end = offset.checked_add(length).ok_or_else(|| {
            AllocationError::Plan(InvariantError {
                context: "import window",
                detail: "source range overflows".into(),
            })
        })?;
        if !Arc::ptr_eq(source, &self.source) || offset < self.start || end > self.end {
            return Err(AllocationError::Plan(InvariantError {
                context: "import window",
                detail: "source tensor lies outside its mapped window".into(),
            }));
        }
        self.region
            .tensor(
                weight.source,
                &[count],
                self.data_offset + offset - self.start,
            )
            .map_err(AllocationError::Tensor)
    }
}

impl ImportWorkspaceLease {
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }
    pub fn class(&self) -> PoolClass {
        self.class
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn source(&self) -> &Tensor {
        self.source.tensor()
    }
    pub fn staged_mut(&mut self) -> Option<&mut Tensor> {
        self.source.staged_mut()
    }
}

impl fmt::Debug for ImportWorkspaceLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImportWorkspaceLease")
            .field("domain", &self.domain)
            .field("class", &self.class())
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationError {
    /// A tensor, region or arena the device could not allocate, its
    /// capacity refusals among them.
    Tensor(seismic::TensorError),
    /// A graph slot the device could not form, its capacity refusals (a
    /// slot's tensors) among them.
    Workflow(seismic::WorkflowError),
    /// An artifact window the host could not map: never a capacity
    /// refusal.
    Mapping(String),
    /// A thread the load's storage work runs on that could not start.
    Thread(String),
    Plan(InvariantError),
}

impl fmt::Display for AllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => write!(formatter, "resource allocation: {error}"),
            Self::Workflow(error) => write!(formatter, "resource allocation: {error}"),
            Self::Mapping(error) => write!(formatter, "resource allocation: {error}"),
            Self::Thread(error) => write!(formatter, "resource allocation: {error}"),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AllocationError {}

impl From<AllocationError> for seismic::BuildError {
    /// An allocation the device refused for capacity is a refusal;
    /// anything else a failure.
    fn from(error: AllocationError) -> Self {
        match error {
            AllocationError::Tensor(error)
            | AllocationError::Workflow(seismic::WorkflowError::TensorView(error)) => error.into(),
            other => Self::Failed(other.to_string()),
        }
    }
}

/// A target readout pool ([`AllocatedResources::readout_pool`]).
#[derive(Clone, Copy)]
enum ReadoutPool {
    /// Any launch's: the floor's slots.
    Launch,
    /// A prompt chunk's (`readout::chunk_holds`): a group's places beyond
    /// the floor.
    Chunk,
}

/// A prompt-group class whose pools' slots [`AllocatedResources::grow_class`]
/// allocated, for [`AllocatedResources::hold_class`] to hold: only an
/// allocated class is held.
#[must_use]
pub(crate) struct GrownClass {
    places: std::num::NonZeroUsize,
    slots: crate::StartupSlots,
}

/// Numerical graph slots authorized by one ResourcePlan. Import sources are
/// allocated separately during component materialization.
pub struct AllocatedResources {
    domain: ResourceDomainId,
    /// The device's one graph workspace, which every pool's activations bind.
    arena: seismic::NativeExecutionArena,
    target_graph: NativeGraphPool,
    /// Readout slots any launch's readout runs in: the floor's, a step and
    /// its lookahead successor.
    target_readout_graph: NativeGraphPool,
    /// Readout slots a prompt chunk's readout runs in, sized by what it
    /// exports (`readout::chunk_holds`): a group's places beyond
    /// the floor.
    target_readout_chunk_graph: NativeGraphPool,
    head_graph: Option<NativeGraphPool>,
    vision_graph: Option<VisionGraphPool>,
    state_graph: NativeGraphPool,
    /// The places of the prompt-group class the pools hold: at least the
    /// floor's one.
    places: std::num::NonZeroUsize,
    /// The load's release thread, which the pools' idle slots release on.
    releases: crate::Releases,
    /// The idle slots on their way there, until a boundary finds them
    /// released ([`Self::released_slots`]).
    releasing: Option<SlotsReleasing>,
}

/// Idle graph slots handed to the release thread at `began`: how many, and
/// their release's outcome once it ends there, the bytes it released.
struct SlotsReleasing {
    slots: usize,
    began: std::time::Instant,
    done: crate::storage_thread::Done<u64>,
}

/// Idle graph slots released off the submitting thread
/// ([`AllocatedResources::released_slots`]): how many, the bytes the device
/// is no longer charged for them, and when they were handed over and
/// released.
#[derive(Clone, Copy, Debug)]
pub struct SlotsReleased {
    pub slots: usize,
    pub bytes: u64,
    pub began: std::time::Instant,
    pub released: std::time::Instant,
}

impl AllocatedResources {
    /// The workspace arena once, with the record tables slots formed over
    /// it, and the arenas these pools own while their slots are lent to
    /// launches or output views.
    pub fn committed_bytes(&self) -> Result<u64, &'static str> {
        [
            Some(&self.target_graph),
            Some(&self.target_readout_graph),
            Some(&self.target_readout_chunk_graph),
            self.head_graph.as_ref(),
            Some(&self.state_graph),
        ]
        .into_iter()
        .flatten()
        .try_fold(self.arena.charged_bytes(), |bytes, pool| {
            bytes
                .checked_add(pool.committed_bytes()?)
                .ok_or("graph pool charge overflows")
        })?
        .checked_add(
            self.vision_graph
                .as_ref()
                .map_or(Ok(0), VisionGraphPool::committed_bytes)?,
        )
        .ok_or("graph pool charge overflows")
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    pub fn target_graph(&self) -> &NativeGraphPool {
        &self.target_graph
    }

    /// The readout pool a launch projecting `projected` rows takes its
    /// activation and output slot from ([`Self::readout_pool`]).
    pub fn target_readout_graph(&self, projected: usize) -> &NativeGraphPool {
        self.readout(self.readout_pool(projected))
    }

    /// The one choice of a launch's readout pool, among those whose output
    /// class holds its `projected` rows, smallest first: the first with an
    /// activation and an output slot free; else the first with an
    /// activation free, whose retained outputs grow by a slot
    /// (`ExecutorDomain::provision_output`); else the launch pool, which
    /// holds any launch's.
    fn readout_pool(&self, projected: usize) -> ReadoutPool {
        let pools = readout::chunk_holds(projected as u64)
            .then_some(ReadoutPool::Chunk)
            .into_iter()
            .chain([ReadoutPool::Launch]);
        let first = |holds: fn(&NativeGraphPool) -> bool| {
            pools.clone().find(|&pool| holds(self.readout(pool)))
        };
        first(|pool| pool.available_workspace() > 0 && pool.available_output() > 0)
            .or_else(|| first(|pool| pool.available_workspace() > 0))
            .unwrap_or(ReadoutPool::Launch)
    }

    fn readout(&self, pool: ReadoutPool) -> &NativeGraphPool {
        match pool {
            ReadoutPool::Launch => &self.target_readout_graph,
            ReadoutPool::Chunk => &self.target_readout_chunk_graph,
        }
    }
    pub fn head_graph(&self) -> Option<&NativeGraphPool> {
        self.head_graph.as_ref()
    }
    pub fn vision_graph(&self) -> Option<&VisionGraphPool> {
        self.vision_graph.as_ref()
    }
    pub(crate) fn vision_graph_mut(&mut self) -> Option<&mut VisionGraphPool> {
        self.vision_graph.as_mut()
    }
    pub fn state_graph(&self) -> &NativeGraphPool {
        &self.state_graph
    }

    /// The pool whose output slots `lane`'s launches retain beyond their
    /// completion, for a launch projecting `projected` rows: target readout
    /// or head outputs. A target launch's is the readout pool it takes its
    /// slots from ([`Self::readout_pool`]). Encoded images hold slots of
    /// their own class ([`VisionGraphPool`]).
    pub(crate) fn retained_outputs_mut(
        &mut self,
        lane: crate::domain::ReservationLane,
        projected: usize,
    ) -> Option<&mut NativeGraphPool> {
        match lane {
            crate::domain::ReservationLane::Target => Some(match self.readout_pool(projected) {
                ReadoutPool::Launch => &mut self.target_readout_graph,
                ReadoutPool::Chunk => &mut self.target_readout_chunk_graph,
            }),
            crate::domain::ReservationLane::Head => self.head_graph.as_mut(),
            crate::domain::ReservationLane::Vision => None,
        }
    }

    /// The places of the prompt-group class held ([`Self::hold_class`]): a
    /// chain runs at most that many launches. The one record of it: the
    /// family's group buffers are held with it (`grant_group`).
    pub fn group_places(&self) -> std::num::NonZeroUsize {
        self.places
    }

    /// The bytes [`Self::grow_class`] of `slots` allocates.
    pub(crate) fn class_bytes(&mut self, slots: crate::StartupSlots) -> Result<u64, &'static str> {
        self.class_pools(slots).try_fold(0u64, |bytes, (pool, slots)| {
            bytes
                .checked_add(pool.hold_bytes(slots)?)
                .ok_or("a graph class's bytes overflow")
        })
    }

    /// What the pools' slots of a class whose pools hold `slots` add to the
    /// device's graph budget beyond those allocated
    /// ([`NativeGraphPool::hold_graphs`]).
    pub(crate) fn class_graphs(&mut self, slots: crate::StartupSlots) -> Result<seismic::GraphCensus, &'static str> {
        self.class_pools(slots).try_fold(seismic::GraphCensus::default(), |census, (pool, slots)| {
            pool.hold_graphs(slots)
                .and_then(|graphs| census.plus(graphs))
                .ok_or("a graph class's graphs overflow")
        })
    }

    /// Allocate the pools' slots of the class of `places` places, whose
    /// pools hold `slots` (`ResourceLimits::class_slots`), to be held by
    /// [`Self::hold_class`]. The caller holds a heap claim for
    /// [`Self::class_bytes`] across this call. On a failure the class held
    /// before stays, and what was allocated idles until released.
    pub(crate) fn grow_class(
        &mut self,
        places: std::num::NonZeroUsize,
        slots: crate::StartupSlots,
    ) -> Result<GrownClass, AllocationError> {
        for (pool, slots) in self.class_pools(slots) {
            pool.grow(slots)?;
        }
        Ok(GrownClass { places, slots })
    }

    /// Hold the class [`Self::grow_class`] allocated.
    pub(crate) fn hold_class(&mut self, grown: GrownClass) {
        for (pool, slots) in self.class_pools(grown.slots) {
            pool.hold(slots);
        }
        self.places = grown.places;
    }

    /// The pools a prompt group's class sizes, each with its slots of
    /// `slots`. The class's readout slots beyond the launch pool's floor run
    /// prompt chunks' readouts: they are the chunk pool's.
    fn class_pools(
        &mut self,
        slots: crate::StartupSlots,
    ) -> impl Iterator<Item = (&mut NativeGraphPool, crate::GraphSlots)> {
        let floor = self.target_readout_graph.floor();
        let chunks = crate::GraphSlots {
            activations: slots.readout.activations.saturating_sub(floor.activations),
            output: slots.readout.output.saturating_sub(floor.output),
        };
        [
            (Some(&mut self.target_graph), slots.target),
            (Some(&mut self.target_readout_chunk_graph), chunks),
            (self.head_graph.as_mut(), slots.head),
        ]
        .into_iter()
        .filter_map(|(pool, slots)| pool.map(|pool| (pool, slots)))
    }

    /// Release every pool's free slots beyond its held class, the held
    /// prompt-group class down to `class`'s places where given (the
    /// places of a class whose pools hold its slots: a settle's target,
    /// or Reclaim's floor of one place), and vision's free storage, on the
    /// load's release thread (`Releases`), `wake` completed once they are
    /// released: whether a release began, which a later boundary finishes
    /// ([`Self::released_slots`]). None begins while one runs
    /// ([`Self::releasing_slots`]).
    pub(crate) fn release_idle_slots(
        &mut self,
        class: Option<(std::num::NonZeroUsize, crate::StartupSlots)>,
        wake: crate::CompletionWake,
    ) -> bool {
        assert!(self.releasing.is_none(), "a graph slot release begins only once the one before it finished");
        if let Some((places, slots)) = class {
            assert!(places <= self.places, "a class is released to no more places than it holds");
            for (pool, slots) in self.class_pools(slots) {
                pool.keep(slots);
            }
            self.places = places;
        }
        let mut idle = IdleSlots::default();
        for pool in [
            Some(&mut self.target_graph),
            Some(&mut self.target_readout_graph),
            Some(&mut self.target_readout_chunk_graph),
            self.head_graph.as_mut(),
            Some(&mut self.state_graph),
        ]
        .into_iter()
        .flatten()
        {
            pool.release_idle(&mut idle);
        }
        if let Some(vision) = self.vision_graph.as_mut() {
            vision.release_idle(&mut idle);
        }
        let slots = idle.len();
        if slots == 0 {
            return false;
        }
        let began = std::time::Instant::now();
        let done = self.releases.run(wake, move || idle.release());
        self.releasing = Some(SlotsReleasing { slots, began, done });
        true
    }

    /// Whether idle slots are being released ([`Self::release_idle_slots`]).
    pub(crate) fn releasing_slots(&self) -> bool {
        self.releasing.is_some()
    }

    /// The idle slots' release, once the release thread has ended it; none
    /// while it runs or where none was begun. Its panic is the device's
    /// invariant violation, reported as one.
    pub(crate) fn released_slots(&mut self) -> Result<Option<SlotsReleased>, String> {
        let Some(release) = self.releasing.as_ref() else {
            return Ok(None);
        };
        let Some((outcome, released)) = release.done.lock().expect("graph slot release outcome").take() else {
            return Ok(None);
        };
        let SlotsReleasing { slots, began, .. } = self.releasing.take().expect("a release in hand");
        outcome.map(|bytes| Some(SlotsReleased { slots, bytes, began, released }))
    }

    /// The load's release thread (`Releases`), which the expert bank
    /// releases on too.
    pub fn releases(&self) -> crate::Releases {
        self.releases.clone()
    }
}

pub struct ResourceAllocator;

impl ResourceAllocator {
    /// Imports are serialized at startup or optional-component materialization.
    /// Each lease owns its source tensor until physical completion.
    pub(crate) fn import_workspace(
        execution: &ExecutionPlan,
        weight: &WeightPlan,
        stored: &Stored,
        window: Option<&ImportWindow>,
        device: &Device,
        domain: ResourceDomainId,
    ) -> Result<ImportWorkspaceLease, AllocationError> {
        let invalid = |detail: &str| {
            AllocationError::Plan(InvariantError {
                context: "import workspace",
                detail: detail.into(),
            })
        };
        if !execution.weights().any(|planned| planned == weight) {
            return Err(invalid("weight is absent from the execution plan"));
        }
        let count = crate::planning::elements(&weight.shape)
            .ok_or_else(|| invalid("weight element count overflows"))?;
        // A weight imported alone and read in place has a window of its own
        // range; its source view keeps the mapping.
        let own = match window {
            None if ImportSource::reads_in_place(weight, stored, device)? => {
                let (file, start, end) = stored_range(stored)?;
                Some(ImportWindow::map(device, file, start, end)?)
            }
            _ => None,
        };
        let source = ImportSource::acquire(weight, stored, window.or(own.as_ref()), device)?;
        let tensor = source.tensor();
        if !tensor.belongs_to(device)
            || tensor.element() != weight.upload
            || tensor.extents() != [count]
            || tensor.byte_len() != weight.source_bytes
            || matches!(&source, ImportSource::Staged(_))
                && tensor.storage_bytes() != weight.source_bytes
        {
            return Err(invalid(
                "source tensor differs from the admitted import contract",
            ));
        }
        Ok(ImportWorkspaceLease {
            domain,
            class: PoolClass::Import {
                bytes: weight.source_bytes,
            },
            bytes: weight.source_bytes,
            source,
        })
    }

    pub fn allocate(
        execution: &ExecutionPlan,
        device: &Device,
        domain: ResourceDomainId,
        target_graphs: &PreparedTargetGraphs,
        target_readout_graphs: &PreparedTargetReadoutGraphs,
        head_graphs: Option<&PreparedDrafterGraphs>,
        vision_graphs: Option<&PreparedVisionGraphs>,
        state_graphs: &PreparedStateCopyGraphs,
    ) -> Result<AllocatedResources, AllocationError> {
        if device.backend() != execution.device().backend()
            || device.info().name() != execution.device().name()
        {
            return Err(AllocationError::Plan(InvariantError {
                context: "resource allocator",
                detail: "opened device differs from the selected execution plan".into(),
            }));
        }
        let plan = execution.resources();
        // The admitted charge was derived from a serial family's footprint.
        let admitted = |family: &seismic::NativeGraphFamily, charge: &crate::NativeGraphCharge| {
            family.workspace_bytes() == charge.workspace_bytes
                && family.output_bytes() == charge.output_bytes
                && crate::planning::serial_upload_regions(family.upload_bytes())
                    == charge.upload_regions
        };
        let arena = device
            .execution_arena(plan.arena_bytes())
            .map_err(AllocationError::Tensor)?;
        // The target's charge is the prepared graphs' footprint
        // (`NativeGraphCharge::from_prepared`).
        let target_graph = NativeGraphPool::new(
            domain.clone(),
            target_graphs.family(),
            plan.target_graph(),
            &arena,
            Outputs::Device,
        )?;
        let readout_charge = plan.target_readout_graph();
        let chunk_charge = plan.target_readout_chunk_graph();
        if !admitted(target_readout_graphs.family(), readout_charge)
            || !admitted(target_readout_graphs.chunk_family(), chunk_charge)
        {
            return Err(AllocationError::Plan(InvariantError {
                context: "resource allocator",
                detail: "prepared target readout graph differs from admitted Seismic footprint"
                    .into(),
            }));
        }
        let target_readout_graph = NativeGraphPool::new(
            domain.clone(),
            target_readout_graphs.family(),
            readout_charge,
            &arena,
            Outputs::Host,
        )?;
        let target_readout_chunk_graph = NativeGraphPool::new(
            domain.clone(),
            target_readout_graphs.chunk_family(),
            chunk_charge,
            &arena,
            Outputs::Host,
        )?;
        let head_graph = match (head_graphs, plan.head_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => Some(
                NativeGraphPool::new(domain.clone(), graphs.family(), charge, &arena, Outputs::Host)?,
            ),
            (None, None) => None,
            _ => {
                return Err(AllocationError::Plan(InvariantError {
                    context: "resource allocator",
                    detail: "prepared head graph differs from admitted Seismic footprint".into(),
                }));
            }
        };
        let vision_graph = match (vision_graphs, plan.vision_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => {
                Some(VisionGraphPool::new(domain.clone(), graphs.class_families()))
            }
            (None, None) => None,
            _ => {
                return Err(AllocationError::Plan(InvariantError {
                    context: "resource allocator",
                    detail: "prepared vision graph differs from admitted Seismic footprint".into(),
                }));
            }
        };
        let state_charge = plan.state_graph();
        if !admitted(state_graphs.family(), state_charge) {
            return Err(AllocationError::Plan(InvariantError {
                context: "resource allocator",
                detail: "prepared state graph differs from admitted Seismic footprint".into(),
            }));
        }
        let state_graph =
            NativeGraphPool::new(domain.clone(), state_graphs.family(), state_charge, &arena, Outputs::Device)?;
        let allocated = AllocatedResources {
            domain: domain.clone(),
            arena,
            target_graph,
            target_readout_graph,
            target_readout_chunk_graph,
            head_graph,
            vision_graph,
            state_graph,
            places: std::num::NonZeroUsize::MIN,
            releases: crate::Releases::start().map_err(AllocationError::Thread)?,
            releasing: None,
        };
        Ok(allocated)
    }
}
