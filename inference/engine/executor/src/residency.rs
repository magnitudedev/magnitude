//! Authoritative device residency for family-neutral semantic weights.
//!
//! Artifact parsing supplies immutable stored slices. This module validates
//! their logical descriptor, imports them through the once-prepared execution
//! path, and owns the sole cache of resulting device tensors.

use crate::device_table::{
    DeviceTable, KeptRows, ResidentTable, StoredRows, TileRun, TileSource,
};
use crate::planning::ImportPlan;
use crate::programs::ProgramSubmission;
use crate::programs::native_import::NativeImportProgram;
use crate::stored_table::StoredTable;
use crate::{
    AllocationError, ExecutionPlan, ImportLaunchInputs, ImportProgram, InvariantError,
    ResidentWeightSlot, ResourceAllocator, ResourceDomainId, SubmitError, ValidatedImportLaunch,
    WeightPlan, WeightStorageIdentity,
};
use magnitude_artifacts::{
    Error as ArtifactError, FileSource,
    gguf::{Encoding, GgufArtifact},
};
use magnitude_family_contracts::{ScaleCodes, WeightDescriptor, WeightRole};
use seismic::{
    DType, Device, Element, NativeTensorBatch, Tensor, TraceDetail, TracedSubmission,
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub enum WeightImportError {
    Artifact(ArtifactError),
    /// A weight's file could not be opened for reads past the cache.
    Read(seismic_io::Error),
    Invalid(String),
    Device(String),
    /// An allocation in `contract` storage its store had no room for: it
    /// needed `required` bytes where `available` were left.
    Refused { contract: seismic::Contract, required: u64, available: u64 },
    Allocation(AllocationError),
    Attestation(crate::CatalogError),
    Submit(SubmitError),
    Completion(crate::DeviceError),
    Invariant(InvariantError),
}

impl fmt::Display for WeightImportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(formatter, "weight source: {error}"),
            Self::Read(error) => write!(formatter, "weight source: {error}"),
            Self::Invalid(message) => formatter.write_str(message),
            Self::Device(message) => write!(formatter, "weight device import: {message}"),
            Self::Refused { contract, required, available } => {
                write!(formatter, "{contract:?} storage requires {required} bytes; {available} bytes available")
            }
            Self::Allocation(error) => error.fmt(formatter),
            Self::Attestation(error) => error.fmt(formatter),
            Self::Submit(error) => error.fmt(formatter),
            Self::Completion(error) => error.fmt(formatter),
            Self::Invariant(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for WeightImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            Self::Read(error) => Some(error),
            Self::Allocation(error) => Some(error),
            Self::Attestation(error) => Some(error),
            Self::Submit(error) => Some(error),
            Self::Completion(error) => Some(error),
            Self::Invariant(error) => Some(error),
            Self::Invalid(_) | Self::Device(_) | Self::Refused { .. } => None,
        }
    }
}

impl From<seismic_io::Error> for WeightImportError {
    fn from(error: seismic_io::Error) -> Self {
        Self::Read(error)
    }
}

impl From<ArtifactError> for WeightImportError {
    fn from(error: ArtifactError) -> Self {
        Self::Artifact(error)
    }
}

impl From<String> for WeightImportError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

impl From<&str> for WeightImportError {
    fn from(message: &str) -> Self {
        Self::Invalid(message.to_owned())
    }
}

impl WeightImportError {
    /// A tensor allocation in `contract` storage that failed: refused for
    /// room where the device's ledger refused it (its capacity refusal),
    /// else the device's failure.
    pub(crate) fn allocating(contract: seismic::Contract, error: &seismic::TensorError) -> Self {
        match error {
            seismic::TensorError::Execution(execution) => match execution.capacity_refusal() {
                Some((required, available)) => Self::Refused { contract, required, available },
                None => Self::Device(error.to_string()),
            },
            error => Self::Device(error.to_string()),
        }
    }

    /// [`Self::allocating`] of a resource allocation in `contract` storage.
    pub(crate) fn of_allocation(contract: seismic::Contract, error: AllocationError) -> Self {
        match error {
            AllocationError::Tensor(error) => Self::allocating(contract, &error),
            error => Self::Allocation(error),
        }
    }
}

fn invalid(message: impl Into<String>) -> WeightImportError {
    WeightImportError::Invalid(message.into())
}

/// A validated dense byte range retained from an immutable artifact source.
#[derive(Clone, Debug)]
pub struct StoredTensor {
    pub source: Arc<FileSource>,
    pub offset: u64,
    pub nbytes: u64,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl StoredTensor {
    pub fn read(&self) -> Result<Vec<u8>, WeightImportError> {
        let length = usize::try_from(self.nbytes)
            .map_err(|_| invalid("tensor exceeds the host address range"))?;
        self.source.read(self.offset, length).map_err(Into::into)
    }
}

/// The byte range of a stored grid of block scales
/// (`ImportTransform::ScaleBlocks`), one code of `codes` a tile.
#[derive(Clone, Debug)]
pub struct StoredGrid {
    pub source: Arc<FileSource>,
    pub offset: u64,
    pub nbytes: u64,
    pub codes: ScaleCodes,
}

impl StoredGrid {
    pub fn read(&self) -> Result<Vec<u8>, WeightImportError> {
        let length = usize::try_from(self.nbytes)
            .map_err(|_| invalid("scale grid exceeds the host address range"))?;
        self.source.read(self.offset, length).map_err(Into::into)
    }
}

/// Storage representation of one semantic model weight.
#[derive(Clone, Debug)]
pub enum Stored {
    Dense(StoredTensor),
    /// Blocks of `encoding`, scaled by their own scales or, with a `grid`,
    /// by its block scales (`seismic::GgufScales`).
    GgmlBlocks {
        source: Arc<FileSource>,
        offset: u64,
        nbytes: u64,
        shape: Vec<u64>,
        encoding: Encoding,
        grid: Option<StoredGrid>,
    },
}

/// The immutable source tensor and the component identity from which it was
/// resolved. Only artifact lookup constructs this import input.
#[derive(Clone, Debug)]
pub struct ImportArtifactTensor {
    artifact: magnitude_artifacts::ArtifactIdentity,
    stored: Stored,
}

impl ImportArtifactTensor {
    pub fn from_gguf(
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
    ) -> Result<Self, WeightImportError> {
        Ok(Self {
            artifact: artifact.identity(),
            stored: Stored::from_gguf(artifact, descriptor)?,
        })
    }

    pub fn artifact(&self) -> magnitude_artifacts::ArtifactIdentity {
        self.artifact
    }
    pub fn stored(&self) -> &Stored {
        &self.stored
    }
}

impl Stored {
    pub(crate) fn file_range(&self) -> (&Arc<FileSource>, u64, u64) {
        match self {
            Self::Dense(tensor) => (&tensor.source, tensor.offset, tensor.nbytes),
            Self::GgmlBlocks {
                source,
                offset,
                nbytes,
                ..
            } => (source, *offset, *nbytes),
        }
    }

    pub fn shape(&self) -> &[u64] {
        match self {
            Self::Dense(tensor) => &tensor.shape,
            Self::GgmlBlocks { shape, .. } => shape,
        }
    }

    pub fn source_bytes(&self) -> u64 {
        match self {
            Self::Dense(tensor) => tensor.nbytes,
            Self::GgmlBlocks { nbytes, .. } => *nbytes,
        }
    }

    /// The grid of block scales the stored blocks are read under.
    pub fn grid(&self) -> Option<&StoredGrid> {
        match self {
            Self::Dense(_) => None,
            Self::GgmlBlocks { grid, .. } => grid.as_ref(),
        }
    }

    pub fn source_element(&self) -> Option<Element> {
        match self {
            Self::Dense(tensor) => Some(Element::dense(tensor.dtype)),
            Self::GgmlBlocks {
                encoding,
                shape,
                grid,
                ..
            } => crate::planning::scaled_source_element(
                *encoding,
                crate::import_transforms::scales_of(grid.as_ref().map(|grid| grid.codes)),
                shape.last().copied().unwrap_or(1),
            ),
        }
    }

    /// The first `rows` of the tensor's leading dimension (an expert stack's
    /// first `rows` experts), which lie first in the file: its bytes, and
    /// its scale grid's, in proportion. None where the tensor's bytes do not
    /// divide evenly into its leading rows.
    fn leading(&self, rows: u64) -> Option<Self> {
        let leading = *self.shape().first()?;
        let part = |nbytes: u64| crate::planning::leading_bytes(nbytes, leading, rows);
        match self {
            Self::Dense(tensor) => Some(Self::Dense(StoredTensor {
                nbytes: part(tensor.nbytes)?,
                shape: [&[rows][..], &tensor.shape[1..]].concat(),
                ..tensor.clone()
            })),
            Self::GgmlBlocks { source, offset, nbytes, shape, encoding, grid } => Some(Self::GgmlBlocks {
                source: source.clone(),
                offset: *offset,
                nbytes: part(*nbytes)?,
                shape: [&[rows][..], &shape[1..]].concat(),
                encoding: *encoding,
                grid: match grid {
                    Some(grid) => Some(StoredGrid { nbytes: part(grid.nbytes)?, ..grid.clone() }),
                    None => None,
                },
            }),
        }
    }

    /// The stored tensor `weight` imports: the artifact's tensor `weight`
    /// names ([`Self::from_gguf`]), of which, where `leading`, a plan of an
    /// expert stack's first experts (`WeightPlan::leading`), only those first
    /// rows. A dense weight's plan may differ from its stored rows by its
    /// transforms, which no prefix of the stored bytes stands for.
    pub fn of_plan(
        artifact: &GgufArtifact,
        weight: &crate::WeightPlan,
        leading: bool,
    ) -> Result<Self, WeightImportError> {
        let stored = Self::from_gguf(artifact, &weight.descriptor)?;
        if !leading {
            return Ok(stored);
        }
        let (Some(&rows), Some(&stored_rows)) = (weight.shape.first(), stored.shape().first()) else {
            return Err(invalid(format!("{:?} has no leading dimension", weight.descriptor.name)));
        };
        let name = &weight.descriptor.name;
        let leading = stored
            .leading(rows)
            .ok_or_else(|| invalid(format!("{rows} leading rows of {name:?} ({stored_rows} stored)")))?;
        match leading.shape()[1..] == weight.shape[1..] {
            true => Ok(leading),
            false => Err(invalid(format!(
                "the planned shape {:?} of {name:?} differs from its stored {:?}",
                weight.shape,
                stored.shape()
            ))),
        }
    }

    /// The stored tensor `descriptor` names, with its grid of block scales
    /// where it names one (bound and admitted with the weight,
    /// `Tensors::companion_scale`).
    pub fn from_gguf(
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
    ) -> Result<Self, WeightImportError> {
        let grid = crate::import_transforms::grid(descriptor)
            .map(|(name, _, codes)| {
                let grid = artifact.tensor(name)?;
                Ok::<_, WeightImportError>(StoredGrid {
                    source: grid.source,
                    offset: grid.offset,
                    nbytes: grid.nbytes,
                    codes,
                })
            })
            .transpose()?;
        let tensor = artifact.tensor(&descriptor.name)?;
        match tensor.encoding {
            Encoding::F32 | Encoding::F16 | Encoding::BF16 => {
                assert!(grid.is_none(), "an admitted dense weight {:?} has no scale grid", descriptor.name);
                Ok(Self::Dense(StoredTensor {
                source: tensor.source,
                offset: tensor.offset,
                nbytes: tensor.nbytes,
                dtype: match tensor.encoding {
                    Encoding::F32 => DType::F32,
                    Encoding::F16 => DType::F16,
                    Encoding::BF16 => DType::BF16,
                    _ => unreachable!("dense encoding was matched above"),
                },
                shape: tensor.shape,
                }))
            }
            encoding => Ok(Self::GgmlBlocks {
                source: tensor.source,
                offset: tensor.offset,
                nbytes: tensor.nbytes,
                shape: tensor.shape,
                encoding,
                grid,
            }),
        }
    }
}

#[derive(Clone)]
pub struct ResidentWeight {
    descriptor: WeightDescriptor,
    tensor: Tensor,
    /// The resident second-level scale (`WeightPlan::scale`): one F32 value
    /// per matrix, bound to its entries' accumulator-scale ports.
    scale: Option<Tensor>,
    /// The resident packing bound (`WeightPlan::bound`), which the import
    /// makes, bound to the packed projection entries' `<weight>_bound` ports.
    bound: Option<Tensor>,
    /// The weight as a gathered table. Of a streamed one `tensor` is the
    /// slots, and every reader of the weight shares the device table.
    table: ResidentTable,
}

impl ResidentWeight {
    /// Visit every tensor the weight holds: its own, its scale and bound,
    /// and its table's beside them.
    pub(crate) fn visit_storage(&self, visit: &mut dyn FnMut(&Tensor)) {
        visit(&self.tensor);
        if let Some(tensor) = &self.scale {
            visit(tensor);
        }
        if let Some(tensor) = &self.bound {
            visit(tensor);
        }
        self.table.visit_extra_storage(visit);
    }

    /// The weight as its row-gathering readers find it.
    pub fn table(&self) -> &ResidentTable {
        &self.table
    }

    pub fn belongs_to(&self, device: &Device) -> bool {
        self.tensor.belongs_to(device)
    }

    pub fn descriptor(&self) -> &WeightDescriptor {
        &self.descriptor
    }

    pub fn element(&self) -> Element {
        self.tensor.element()
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn scale(&self) -> Option<&Tensor> {
        self.scale.as_ref()
    }

    pub fn bound(&self) -> Option<&Tensor> {
        self.bound.as_ref()
    }

    /// The weight of a streamed table: the slots that hold some of its
    /// rows, and the device table they are.
    fn streamed(descriptor: WeightDescriptor, table: Rc<DeviceTable>) -> Self {
        Self {
            descriptor,
            tensor: table.slots_tensor().clone(),
            scale: None,
            bound: None,
            table: ResidentTable::streamed(table),
        }
    }

    /// The weight of a stored table: the slots that hold some of its rows as
    /// stored, and the stored table they are.
    fn stored(descriptor: WeightDescriptor, table: Arc<StoredTable>) -> Self {
        Self {
            descriptor,
            tensor: table.slots_tensor().clone(),
            scale: None,
            bound: None,
            table: ResidentTable::Stored(table),
        }
    }

    /// Every allocation of the weight: its representation (with the record
    /// table its storage holds, `ModelLoadPlan::weight_storage_bytes`), its
    /// scale, its packing bound and a streamed table's directory and
    /// staging.
    pub(crate) fn storage_bytes(&self) -> u64 {
        self.storage_bytes_staging(self.table.staging_bytes())
    }

    /// [`Self::storage_bytes`], its table's stagings counted as `stagings`
    /// bytes: every other allocation as charged now, a stored table's slots
    /// at the records the bank resized them to.
    fn storage_bytes_staging(&self, stagings: u64) -> u64 {
        self.tensor.charged_bytes()
            + self.scale.as_ref().map_or(0, Tensor::storage_bytes)
            + self.bound.as_ref().map_or(0, Tensor::storage_bytes)
            + self.table.held_bytes()
            + stagings
    }
}

/// Brings a streamed table's tiles from its artifact: read as stored and
/// converted by the entry that would import the whole weight, queued ahead
/// of the step that names them. The device converts them before it runs
/// that step's graphs, and the host does not wait: the conversion's
/// completion guards its staging's reuse.
struct TileImport {
    table: crate::host_tables::HostTable,
    import: crate::native::AttestedImport,
    device: Rc<Device>,
}

impl TileSource for TileImport {
    fn bring(&self, runs: &[TileRun<'_>], staging: &mut Tensor) -> Result<Option<seismic::NativeTensorBatchCompletion>, String> {
        let failed = |error: &dyn fmt::Display| format!("streamed table rows: {error}");
        let tiles = runs
            .iter()
            .flat_map(|run| run.tiles.iter().copied())
            .collect::<Vec<_>>();
        let tile = staging.extents()[0] / tiles.len() as u64;
        let tile_bytes = tile as usize * self.table.row_bytes();
        let mut bytes = vec![0; tiles.len() * tile_bytes];
        // Tiles that follow one another in the table are read together.
        let mut at = 0;
        while at < tiles.len() {
            let run = (at..tiles.len())
                .take_while(|next| tiles[*next] == tiles[at] + (*next - at) as u64 * tile)
                .count();
            self.table
                .gather_run(
                    tiles[at],
                    &mut bytes[at * tile_bytes..(at + run) * tile_bytes],
                )
                .map_err(|error| failed(&error))?;
            at += run;
        }
        staging
            .write_from_host(&bytes)
            .map_err(|error| failed(&error))?;
        // Every run's conversion is queued in one batch.
        let mut batch = NativeTensorBatch::new(&self.device);
        let mut staged = 0;
        for run in runs {
            let rows = run.tiles.len() as u64 * tile;
            let source = staging
                .slice_leading(staged, staged + rows)
                .map_err(|error| failed(&error))?;
            // A streamed table has no packing bound.
            crate::programs::native_import::enqueue_into(
                &self.import,
                &mut batch,
                &source,
                &run.slots,
                None,
            )
            .map_err(|error| failed(&error))?;
            staged += rows;
        }
        batch.submit().map(Some).map_err(|error| failed(&error))
    }

    fn rows(&self) -> Arc<dyn StoredRows> {
        Arc::new(self.table.clone())
    }
}

impl StoredRows for crate::host_tables::HostTable {
    fn read(&self, first: u64, out: &mut [u8]) -> Result<(), String> {
        self.gather_run(first, out)
            .map_err(|error| format!("streamed table rows: {error}"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ResidencyKey {
    artifact: magnitude_artifacts::ArtifactIdentity,
    name: String,
    transforms: Vec<magnitude_family_contracts::ImportTransform>,
    resident: Element,
}

enum ImportPreparation {
    Resident(ResidentWeight),
    Pending {
        key: ResidencyKey,
        descriptor: WeightDescriptor,
        scale: Option<Tensor>,
        launch: ValidatedImportLaunch,
        program: NativeImportProgram,
    },
}

struct OrderedImport {
    plan: WeightPlan,
    source: Arc<FileSource>,
    offset: u64,
    target: DType,
}

impl OrderedImport {
    fn of(
        artifact: &GgufArtifact,
        weights: &[WeightPlan],
        activation: DType,
    ) -> Result<Vec<Self>, WeightImportError> {
        let mut seen = HashSet::<WeightStorageIdentity>::new();
        let mut ordered = Vec::new();
        // A host-placed weight takes no import entry (`import_progressive`).
        // Nor is a streamed table imported: its rows reach the device as
        // steps name them (`stream_table`); nor a banked expert stack, whose
        // values are its bank's (`open_expert_bank`).
        for plan in weights
            .iter()
            .filter(|plan| !plan.placed_on_host() && plan.holding == crate::Holding::Whole)
        {
            if !seen.insert(plan.storage_identity()) {
                continue;
            }
            let tensor = ImportArtifactTensor::from_gguf(artifact, &plan.descriptor)?;
            let (source, offset, _) = tensor.stored().file_range();
            ordered.push(OrderedImport {
                plan: plan.clone(),
                source: source.clone(),
                offset,
                target: plan.resident.dtype().unwrap_or(activation),
            });
        }
        // A split component's tensors live in several files: order each
        // file's tensors contiguously, read in file order.
        ordered.sort_by(|left, right| {
            left.source
                .path()
                .cmp(right.source.path())
                .then(left.offset.cmp(&right.offset))
        });
        Ok(ordered)
    }

    /// What every import of one batch shares: its file.
    fn file(&self) -> *const FileSource {
        Arc::as_ptr(&self.source)
    }
}

/// Coarse host costs of the most recent batched component import. The caller
/// can use these alongside its total import duration to identify work outside
/// preparation (the sources' reads among it), submission, waiting, and
/// publication.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImportReport {
    pub batches: usize,
    pub weights: usize,
    pub preparing: Duration,
    pub submitting: Duration,
    pub waiting: Duration,
    pub publishing: Duration,
}

/// Observes a component import in resident bytes: completed, then total.
pub type ImportObserver = Box<dyn FnMut(u64, u64)>;

/// The resident bytes of the component import in progress and where they are
/// reported.
struct ImportProgress {
    completed: u64,
    total: u64,
    observer: ImportObserver,
}

/// The component whose import brought a weight into a residency cache: the
/// target's, at load, or an optional component's, by its loader
/// ([`ComponentLoader`]). A weight a later component finds in the cache
/// stays the importer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Importer {
    Target,
    Head,
    Vision,
}

/// A completed import in a residency cache: the weight, its importer, and
/// a streamed table's stagings as imported, at the floor's class; those it
/// holds beyond it for a group's class are the group's. Everything else the
/// weight holds is read as charged now (`ResidentWeight::storage_bytes`):
/// a stored table's slots are resized by the bank after import.
struct Imported {
    weight: ResidentWeight,
    importer: Importer,
    stagings: u64,
}

/// A residency cache's completed imports, each with its importer, so every
/// importer's bytes are its own imports' and never what is left of another's.
/// One cache, shared: the store imports into it, and a component it imported
/// reads its own bytes from it after the store is gone
/// ([`crate::ResidentTarget`]).
#[derive(Clone, Default)]
pub(crate) struct Imports(Rc<RefCell<HashMap<ResidencyKey, Imported>>>);

impl Imports {
    fn get(&self, key: &ResidencyKey) -> Option<ResidentWeight> {
        self.0.borrow().get(key).map(|imported| imported.weight.clone())
    }

    /// Cache `weight`, `importer`'s import: its bytes as imported.
    fn insert(&self, key: ResidencyKey, weight: ResidentWeight, importer: Importer) -> u64 {
        let bytes = weight.storage_bytes();
        let stagings = weight.table.staging_bytes();
        self.0.borrow_mut().insert(key, Imported { weight, importer, stagings });
        bytes
    }

    /// The bytes of `importer`'s imports ([`Imported`]). Tied roles share
    /// one key and one allocation; a failed component import may leave a
    /// partial set, which this still counts exactly.
    pub(crate) fn bytes(&self, importer: Importer) -> Result<u64, &'static str> {
        self.0
            .borrow()
            .values()
            .filter(|imported| imported.importer == importer)
            .try_fold(0u64, |bytes, imported| {
                bytes
                    .checked_add(imported.weight.storage_bytes_staging(imported.stagings))
                    .ok_or("resident cache charge overflows")
            })
    }

    /// Drop `importer`'s imports; every other importer's stay resident with
    /// their storage identity.
    fn release(&self, importer: Importer) {
        self.0.borrow_mut().retain(|_, imported| imported.importer != importer);
    }

    fn len(&self) -> usize {
        self.0.borrow().len()
    }

    /// Visit every tensor of the cached imports.
    fn visit_storage(&self, visit: &mut dyn FnMut(&Tensor)) {
        for imported in self.0.borrow().values() {
            imported.weight.visit_storage(visit);
        }
    }
}

/// Sole cache and importer for one planned device. Resident components are
/// assembled from this store before their numerical programs are bound.
pub struct ResidencyStore {
    device: Rc<Device>,
    programs: Rc<crate::native::AttestedPrograms>,
    execution: ExecutionPlan,
    domain: ResourceDomainId,
    resident: Imports,
    /// Whose import the store makes now: the target's until an optional
    /// component's loader takes it.
    importer: Importer,
    import_report: ImportReport,
    import_progress: Option<ImportProgress>,
    /// Where the rows steps name of streamed tables are kept, when they are.
    kept_table_rows: Option<std::path::PathBuf>,
    /// What the job stored of the bank's CPU sites.
    cpu_site: crate::assessment::StoredCpuSite,
    /// The load's release thread, which the expert bank releases on.
    releases: crate::Releases,
    /// The changes of the load's ledgers' scope, which the expert bank
    /// holds each memory change of with its record from its open
    /// (`memory::ledger::Changing`).
    changes: std::sync::Arc<crate::memory::ledger::Changes>,
    /// The claim to a linked host tier's load-time slots, which the bank
    /// allocates as each backs from its open: held exactly where the plan's
    /// bank has host slots ([`crate::memory::ledger::DeviceLedgers::host_backing`]),
    /// split from the target import's claim as the store is made for it.
    host_backing: Option<crate::memory::ledger::Claim>,
}

impl ResidencyStore {
    /// The store of a load's target import: `weights` is the import's claim
    /// on `ledgers`, which the tier of the plan's bank takes its backing
    /// from, so a bank opened by this store has its host tier's claim.
    pub fn for_target(
        device: Rc<Device>,
        programs: Rc<crate::native::AttestedPrograms>,
        execution: ExecutionPlan,
        domain: ResourceDomainId,
        releases: crate::Releases,
        ledgers: &crate::memory::ledger::DeviceLedgers,
        weights: &mut crate::memory::ledger::Held,
    ) -> Result<Self, WeightImportError> {
        let backing = ledgers.host_backing(weights, execution.load()).map_err(invalid)?;
        Self::make(device, programs, execution, domain, releases, ledgers.changes().clone(), backing)
    }

    /// The store of a component that opens no expert bank (the vision
    /// tower's): its plan holds none, so no bank opens through it, with or
    /// without a host tier's backing.
    pub fn for_component(
        device: Rc<Device>,
        programs: Rc<crate::native::AttestedPrograms>,
        execution: ExecutionPlan,
        domain: ResourceDomainId,
        releases: crate::Releases,
        changes: std::sync::Arc<crate::memory::ledger::Changes>,
    ) -> Result<Self, WeightImportError> {
        Self::make(device, programs, execution.without_expert_bank(), domain, releases, changes, None)
    }

    fn make(
        device: Rc<Device>,
        programs: Rc<crate::native::AttestedPrograms>,
        execution: ExecutionPlan,
        domain: ResourceDomainId,
        releases: crate::Releases,
        changes: std::sync::Arc<crate::memory::ledger::Changes>,
        host_backing: Option<crate::memory::ledger::Claim>,
    ) -> Result<Self, WeightImportError> {
        if !programs.belongs_to(&device)
            || execution.device().backend() != device.backend()
            || execution.device().name() != device.info().name()
        {
            return Err(invalid(
                "attested programs, selected plan, and residency device differ",
            ));
        }
        Ok(Self {
            device,
            programs,
            execution,
            domain,
            resident: Imports::default(),
            importer: Importer::Target,
            import_report: ImportReport::default(),
            import_progress: None,
            kept_table_rows: None,
            cpu_site: Default::default(),
            releases,
            changes,
            host_backing,
        })
    }

    /// Cache a completed import, counting it toward the import in progress.
    /// The weight is resident either way; a refusal stops the import after it.
    fn publish(
        &mut self,
        key: ResidencyKey,
        weight: ResidentWeight,
    ) -> Result<(), WeightImportError> {
        let bytes = self.resident.insert(key, weight, self.importer);
        self.imported(bytes)
    }

    /// Count `storage_bytes` toward the import in progress.
    fn imported(&mut self, storage_bytes: u64) -> Result<(), WeightImportError> {
        if let Some(progress) = &mut self.import_progress {
            // Planned and allocated storage agree; the bound keeps a
            // disagreement from reporting past completion.
            progress.completed = progress
                .completed
                .saturating_add(storage_bytes)
                .min(progress.total);
            (progress.observer)(progress.completed, progress.total);
        }
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn belongs_to(&self, device: &Device) -> bool {
        self.device.info().id() == device.info().id()
    }
    pub fn len(&self) -> usize {
        self.resident.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Visit every tensor of the completed resident imports.
    pub(crate) fn visit_storage(&self, visit: &mut dyn FnMut(&Tensor)) {
        self.resident.visit_storage(visit);
    }

    /// The cache this store imports into.
    pub(crate) fn imports(&self) -> Imports {
        self.resident.clone()
    }

    pub fn import_report(&self) -> ImportReport {
        self.import_report
    }

    /// Keep the rows steps name of this store's streamed tables in
    /// `directory`, a file per table, for the next time each is served.
    /// Open the bank's CPU sites by what the job stored.
    pub fn keep_cpu_site(&mut self, cpu_site: crate::assessment::StoredCpuSite) {
        self.cpu_site = cpu_site;
    }

    pub fn keep_table_rows(&mut self, directory: std::path::PathBuf) {
        self.kept_table_rows = Some(directory);
    }

    /// The resident weight of `descriptor` when its plan streams it
    /// (`WeightPlan::streamed`): a streamed table is not imported. Its
    /// device table is made and seeded the first time a reader asks for the
    /// weight, and shared by every reader after.
    fn streamed(
        &mut self,
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
    ) -> Result<Option<ResidentWeight>, WeightImportError> {
        let Some(plan) = self
            .execution
            .weights()
            .find(|weight| {
                weight.component.identity == artifact.identity()
                    && weight.descriptor == *descriptor
                    && weight.streamed().is_some()
            })
            .cloned()
        else {
            return Ok(None);
        };
        let key = ResidencyKey {
            artifact: artifact.identity(),
            name: descriptor.name.clone(),
            transforms: descriptor.transforms.clone(),
            resident: plan.resident,
        };
        if let Some(weight) = self.resident.get(&key) {
            return Ok(Some(weight));
        }
        let weight = match plan.stored_in_slots() {
            false => {
                ResidentWeight::streamed(descriptor.clone(), self.stream_table(artifact, &plan)?)
            }
            true => ResidentWeight::stored(descriptor.clone(), self.store_table(artifact, &plan)?),
        };
        // The plan charged the slots (their table too), the directory and
        // the staging.
        let planned = self.execution.load().weight_storage_bytes(&plan).map_err(invalid)?;
        if weight.storage_bytes() != planned {
            return Err(invalid(format!(
                "streamed table {:?} allocates {} bytes but its plan charged {planned}",
                descriptor.name,
                weight.storage_bytes()
            )));
        }
        self.publish(key, weight.clone())?;
        Ok(Some(weight))
    }

    /// The device table of the streamed gathered table `plan`, its slots
    /// filled with the rows worth holding before any step names one.
    fn stream_table(
        &self,
        artifact: &GgufArtifact,
        plan: &WeightPlan,
    ) -> Result<Rc<DeviceTable>, WeightImportError> {
        let streamed = plan
            .streamed()
            .ok_or_else(|| invalid("a weight held whole has no device table"))?;
        let crate::planning::TableFill::Imported(flights) = streamed.fill else {
            return Err(invalid("a stored table has no device table"));
        };
        let stored = Stored::from_gguf(artifact, &plan.descriptor)
            .map_err(|error| invalid(error.to_string()))?;
        let table = crate::host_tables::HostTable::open(&stored)
            .map_err(|error| invalid(error.to_string()))?;
        // The kept rows are the table's in this artifact file, by where it
        // lies there, in these slots.
        let kept = self.kept_table_rows.as_ref().map(|directory| {
            let (source, offset, length) = stored.file_range();
            KeptRows::of(
                directory,
                &format!(
                    "{}\n{offset}\n{length}\n{:?}\n{:?}",
                    source.path().display(),
                    plan.resident,
                    plan.shape
                ),
            )
        });
        let table = DeviceTable::new(
            &self.device,
            plan.resident,
            table.source(),
            streamed.table_rows,
            &plan.shape[1..],
            plan.shape[0] - streamed.landing_rows,
            streamed.landing_rows,
            streamed.staging_rows,
            flights,
            streamed.directory,
            Box::new(TileImport {
                import: self
                    .programs
                    .import_entry(plan)
                    .map_err(|error| invalid(error.to_string()))?,
                device: self.device.clone(),
                table,
            }),
            kept,
        )
        .map_err(invalid)?;
        table.seed().map_err(invalid)?;
        let table = Rc::new(table);
        crate::settle::owe(Rc::downgrade(&table));
        Ok(table)
    }

    /// The stored table of the streamed gathered table `plan`
    /// (`TableFill::Stored`), its slots holding the rows steps named when it
    /// was last served. Its rows are read from the artifact without the
    /// file cache, a range of each of its row tensors
    /// (`import_transforms::row_tensors`), each from the file that holds it.
    fn store_table(
        &self,
        artifact: &GgufArtifact,
        plan: &WeightPlan,
    ) -> Result<Arc<StoredTable>, WeightImportError> {
        let streamed = plan
            .streamed()
            .ok_or_else(|| invalid("a weight held whole has no stored table"))?;
        let names = crate::import_transforms::row_tensors(&plan.descriptor);
        let tensors = names.iter().map(|name| artifact.tensor(name)).collect::<Result<Vec<_>, _>>()?;
        let row_bytes = plan
            .upload
            .canonical_byte_len(&[1, plan.table_width()])
            .map_err(|error| invalid(error.to_string()))?;
        let bytes = crate::planning::stored_row_ranges(
            &names.iter().zip(&tensors).map(|(name, tensor)| (*name, tensor.nbytes)).collect::<Vec<_>>(),
            streamed.table_rows,
            row_bytes,
        )
        .map_err(|error| invalid(format!("stored table {:?}: {error}", plan.descriptor.name)))?;
        // Every range's file, offset and bytes, which the rows kept are of.
        let kept = self.kept_table_rows.as_ref().map(|directory| {
            let ranges = tensors
                .iter()
                .zip(&bytes)
                .map(|(tensor, bytes)| format!("{}\n{}\n{bytes}\n", tensor.source.path().display(), tensor.offset))
                .collect::<String>();
            KeptRows::of(directory, &format!("{ranges}{:?}\n{:?}", plan.resident, plan.shape))
        });
        let [records, record_rows, width] = plan.shape[..] else {
            return Err(invalid(format!("stored table {:?} is not records of rows", plan.descriptor.name)));
        };
        // A table the expert bank holds is reserved, resized as the bank's
        // one owner prices it (`ExpertBankPlan::tables`).
        let banked = self
            .execution
            .load()
            .expert_bank()
            .is_some_and(|bank| bank.tables.iter().any(|table| table.role == plan.role));
        let slots = match banked {
            true => crate::stored_table::StoredSlots::Reserved { records, record_rows },
            false => crate::stored_table::StoredSlots::Whole { records, record_rows },
        };
        // A step waits on the rows its lookups name: one head's rows of a
        // step, each its ranges in their files, are the table's urgent
        // request (the ranges of one file are its parts).
        let ranges = tensors
            .iter()
            .zip(&bytes)
            .map(|(tensor, &range)| {
                let parts = tensors
                    .iter()
                    .zip(&bytes)
                    .filter(|(other, _)| Arc::ptr_eq(&other.source, &tensor.source))
                    .map(|(_, bytes)| *bytes)
                    .collect();
                let urgent = seismic_io::Urgent { parts, records: magnitude_batching::MAX_CLASS_ROWS as u64 };
                Ok(crate::stored_table::RowRange {
                    source: seismic_io::Reader::of_urgent(&*tensor.source, seismic_io::CachePolicy::Uncached, &urgent)?,
                    first: tensor.offset,
                    bytes: usize::try_from(range).map_err(|_| invalid("a table row's range exceeds the host address range"))?,
                })
            })
            .collect::<Result<Vec<_>, WeightImportError>>()?;
        let table = StoredTable::new(
            &self.device,
            &plan.descriptor,
            plan.upload,
            plan.resident,
            streamed.table_rows,
            width,
            slots,
            streamed.directory,
            ranges,
            kept,
        )
        .map_err(invalid)?;
        table.seed().map_err(invalid)?;
        let table = Arc::new(table);
        crate::settle::owe_shared(Arc::downgrade(&table));
        Ok(table)
    }

    /// The load's expert bank (`ExpertBankPlan`), allocated over the files
    /// `artifact` stores its experts in, empty or, holding every expert,
    /// imported whole, and counted toward the import in progress; `None` of
    /// a load without one.
    pub(crate) fn open_expert_bank(
        &mut self,
        artifact: &GgufArtifact,
    ) -> Result<Option<Arc<crate::experts::ExpertBank>>, WeightImportError> {
        let Some(plan) = self.execution.load().expert_bank() else {
            return Ok(None);
        };
        // The bank allocates its plan's layout, which the plan charges.
        let sites = self.programs.expert_bank_sites(plan).map_err(invalid)?;
        // Each streamed layer's bank lines, which the bank completes with its
        // CPU pool's and landings' (`ExpertBank::lines`); the load's pricing
        // adds what the measurement job stored (`HostLines::of_bank`).
        let lines = crate::assessment::BankLines::of_bank(plan).map_err(|error| invalid(error.to_string()))?;
        let bank = crate::experts::ExpertBank::open(&self.device, plan, artifact, &sites, lines, &self.cpu_site, self.releases.clone(), (self.changes.clone(), self.host_backing.take()))?;
        // Each stack through the entry a resident load imports it with, from
        // a read around the file cache: no expert byte stays in it.
        if plan.whole() {
            bank.import_whole(&mut |weight, records| {
                let entry = self.programs.import_entry(weight).map_err(|error| error.to_string())?;
                let stored = Stored::from_gguf(artifact, &weight.descriptor).map_err(|error| error.to_string())?;
                let source = crate::resources::filled_import(weight, &stored, &self.device)?;
                crate::programs::native_import::import_into(&entry, &source, records, None)
                    .map_err(|error| error.to_string())
            })
            .map_err(invalid)?;
        }
        self.imported(plan.bytes().map_err(invalid)?)?;
        Ok(Some(Arc::new(bank)))
    }

    /// The imported physical view selected for an explicit semantic weight role.
    pub(crate) fn planned(&self, role: WeightRole) -> Option<&WeightPlan> {
        self.execution
            .load()
            .weights()
            .find(|weight| weight.role == role)
    }

    /// Imports a plane of a progressive weight (`ImportTransform::Progressive`):
    /// the host places it from the codes of the stored rows, and the device
    /// holds the placed bytes as they are.
    pub(crate) fn import_progressive(
        &mut self,
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
    ) -> Result<ResidentWeight, WeightImportError> {
        let planned = self
            .execution
            .weights()
            .find(|weight| {
                weight.component.identity == artifact.identity() && weight.descriptor == *descriptor
            })
            .cloned()
            .ok_or_else(|| {
                invalid(format!(
                    "weight {:?} is absent from the admitted WeightPlan",
                    descriptor.name
                ))
            })?;
        if !planned.placed_on_host() || planned.upload != planned.resident {
            return Err(invalid(format!(
                "weight {:?} is not a progressive plane",
                descriptor.name
            )));
        }
        let key = ResidencyKey {
            artifact: artifact.identity(),
            name: descriptor.name.clone(),
            transforms: descriptor.transforms.clone(),
            resident: planned.resident,
        };
        if let Some(weight) = self.resident.get(&key) {
            return Ok(weight);
        }
        let source = ImportArtifactTensor::from_gguf(artifact, descriptor)?;
        let stored = source.stored();
        let bytes = crate::programs::native_import::stored_source_bytes(stored)
            .map_err(|error| invalid(error.to_string()))?;
        let bytes = crate::import_transforms::upload_bytes(&planned, stored, bytes)
            .map_err(invalid)?;
        let tensor = Tensor::from_host(&self.device, planned.resident, &planned.shape, &bytes)
            .map_err(|error| WeightImportError::allocating(seismic::Contract::DeviceOnly, &error))?;
        let weight = ResidentWeight {
            descriptor: descriptor.clone(),
            tensor,
            scale: None,
            bound: None,
            table: ResidentTable::whole(),
        };
        self.publish(key, weight.clone())?;
        Ok(weight)
    }

    /// Imports a dense weight into the element its admitted plan chose.
    pub(crate) fn import_gguf_planned(
        &mut self,
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
    ) -> Result<ResidentWeight, WeightImportError> {
        let target = self
            .execution
            .weights()
            .find(|weight| {
                weight.component.identity == artifact.identity() && weight.descriptor == *descriptor
            })
            .and_then(|weight| weight.resident.dtype())
            .ok_or_else(|| {
                invalid(format!(
                    "weight {:?} has no dense element in the admitted WeightPlan",
                    descriptor.name
                ))
            })?;
        self.import_gguf(artifact, descriptor, target)
    }

    pub(crate) fn import_gguf(
        &mut self,
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
        target: DType,
    ) -> Result<ResidentWeight, WeightImportError> {
        if let Some(weight) = self.streamed(artifact, descriptor)? {
            return Ok(weight);
        }
        let (key, descriptor, scale, launch, mut program) =
            match self.prepare_gguf(artifact, descriptor, target)? {
                ImportPreparation::Resident(weight) => return Ok(weight),
                ImportPreparation::Pending {
                    key,
                    descriptor,
                    scale,
                    launch,
                    program,
                } => (key, descriptor, scale, launch, program),
            };
        let submission = program
            .submit(launch)
            .map_err(|(error, _launch)| WeightImportError::Submit(error))?;
        let completed = submission.finish().map_err(WeightImportError::Completion)?;
        let (_, destination) = completed.into_parts();
        let (tensor, bound) = destination.into_parts();
        let weight = ResidentWeight {
            descriptor,
            tensor,
            scale,
            bound,
            table: ResidentTable::whole(),
        };
        self.publish(key, weight.clone())?;
        Ok(weight)
    }

    fn prepare_gguf(
        &self,
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
        target: DType,
    ) -> Result<ImportPreparation, WeightImportError> {
        let planned = self
            .execution
            .weights()
            .find(|weight| {
                weight.component.identity == artifact.identity() && weight.descriptor == *descriptor
            })
            .cloned()
            .ok_or_else(|| {
                invalid(format!(
                    "weight {:?} is absent from the admitted WeightPlan",
                    descriptor.name
                ))
            })?;
        if planned
            .resident
            .dtype()
            .is_some_and(|dtype| dtype != target)
        {
            return Err(invalid(format!(
                "weight {:?} import differs from the admitted WeightPlan",
                descriptor.name
            )));
        }
        let source = ImportArtifactTensor::from_gguf(artifact, descriptor)?;
        let stored = source.stored();
        validate_request(descriptor, stored, target)?;
        let actual_source = stored
            .source_element()
            .ok_or_else(|| invalid("unsupported source representation"))?;
        let layout = crate::resident_layout(
            self.execution.policy().path(),
            self.execution.device().backend(),
        );
        // The plan dequantizes a packed weight on the host when its kernels
        // read dense weights only: it uploads F32 and becomes dense.
        let dequantized = planned.upload != planned.source;
        let (upload, actual_resident) = match stored {
            Stored::Dense(_) => (actual_source, Element::dense(target)),
            Stored::GgmlBlocks { .. } if dequantized => (Element::f32(), Element::dense(target)),
            Stored::GgmlBlocks { .. } => (
                actual_source,
                crate::planning::resident_storage(
                    actual_source,
                    crate::import_transforms::scaled_activations(descriptor),
                    target,
                    layout,
                )
                .ok_or_else(|| invalid("unsupported packed resident representation"))?,
            ),
        };
        // The importer uploads the transformed source: its logical shape in
        // the upload representation.
        let uploaded = upload
            .canonical_byte_len(&descriptor.shape)
            .map_err(|error| invalid(error.to_string()))?;
        if planned.source != actual_source
            || planned.upload != upload
            || planned.resident != actual_resident
            || planned.source_bytes != uploaded
        {
            return Err(invalid(format!(
                "weight {:?} representation differs from the admitted WeightPlan",
                descriptor.name
            )));
        }
        let key = ResidencyKey {
            artifact: artifact.identity(),
            name: descriptor.name.clone(),
            transforms: descriptor.transforms.clone(),
            resident: actual_resident,
        };
        if let Some(weight) = self.resident.get(&key) {
            if weight.descriptor() != descriptor {
                return Err(invalid(
                    "resident tensor has a conflicting logical descriptor",
                ));
            }
            return Ok(ImportPreparation::Resident(weight));
        }
        let tensor = if self.device.backend() == seismic::BackendName::Metal {
            // SAFETY: the attested import's only result is this tensor. Both
            // Metal import kernels write every physical byte, including the
            // packed layout's row and plane padding, before submission returns.
            unsafe { Tensor::uninitialized(&self.device, planned.resident, &planned.shape) }
        } else {
            Tensor::zeros(&self.device, planned.resident, &planned.shape)
        }
        .map_err(|error| WeightImportError::allocating(seismic::Contract::DeviceOnly, &error))?;
        let destination = ResidentWeightSlot::new(&planned, &self.device, tensor)
            .map_err(WeightImportError::Invariant)?;
        let workspace = ResourceAllocator::import_workspace(
            &self.execution,
            &planned,
            &self.device,
            self.domain.clone(),
        )
        .map_err(|error| WeightImportError::of_allocation(seismic::Contract::DeviceOnly, error))?;
        let launch = ValidatedImportLaunch::new(
            ImportLaunchInputs::new(planned.clone(), source, workspace, destination),
            &self.domain,
        )
        .map_err(|(_, error)| WeightImportError::Invariant(error))?;
        let program = self
            .programs
            .bind_import(&planned)
            .map_err(WeightImportError::Attestation)?;
        let scale = planned
            .scale
            .as_ref()
            .map(|scale| import_scale(&self.device, artifact, scale))
            .transpose()?;
        Ok(ImportPreparation::Pending {
            key,
            descriptor: descriptor.clone(),
            scale,
            launch,
            program,
        })
    }

    /// Preload `weights` from the component each is stored in: a target
    /// load's weights include a separate draft's fusion (stored in the
    /// draft), and a separate draft's include nothing of the projector.
    fn preload_components(
        &mut self,
        package: &magnitude_artifacts::Package,
        weights: &[WeightPlan],
        activation: DType,
    ) -> Result<(), WeightImportError> {
        use crate::ArtifactComponentKind;
        let mut target = Vec::new();
        let mut draft = Vec::new();
        for weight in weights {
            match weight.component.kind {
                ArtifactComponentKind::Target => target.push(weight.clone()),
                ArtifactComponentKind::Draft => draft.push(weight.clone()),
                ArtifactComponentKind::Projector => {
                    return Err(invalid("a target or draft load plans a projector weight"));
                }
            }
        }
        self.preload_component(package.target(), &target, activation)?;
        if !draft.is_empty() {
            let artifact = package
                .draft()
                .ok_or_else(|| invalid("draft weights planned without the draft component"))?;
            self.preload_component(artifact, &draft, activation)?;
        }
        Ok(())
    }

    /// Import one component in source-file order. Whole adjacent tensors
    /// share one ordered native submission, their staged sources together
    /// within the import bound. Semantic
    /// assembly below then reads the completed weights from the sole cache.
    fn preload_component(
        &mut self,
        artifact: &GgufArtifact,
        weights: &[WeightPlan],
        activation: DType,
    ) -> Result<(), WeightImportError> {
        if self.device.backend() != seismic::BackendName::Metal || weights.is_empty() {
            return Ok(());
        }
        let ordered = OrderedImport::of(artifact, weights, activation)?;
        // The same source bound drives admission and batching. Whole
        // streamed tables and banked expert stacks never enlarge this bound:
        // only their slots (or no upload at all) are source storage.
        let imports = ImportPlan::new(weights.iter());
        let mut report = ImportReport::default();
        // Measurement only: timed Metal encoders change the device path, so
        // these numbers attribute work but are not production load timings.
        let trace_batch = std::env::var("MAGNITUDE_TRACE_IMPORT_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let mut first = 0;
        while first < ordered.len() {
            let batch_len = imports
                .batch(ordered[first..].iter().map(|item| (item.file(), item.plan.source_bytes)))
                .expect("the remaining import sources are nonempty");
            let last = first + batch_len;
            report.batches += 1;
            let mut batch = NativeTensorBatch::new(&self.device);
            let mut pending = Vec::new();
            let mut traced_weights = Vec::new();
            let preparing = Instant::now();
            for item in &ordered[first..last] {
                match self.prepare_gguf(artifact, &item.plan.descriptor, item.target)? {
                    ImportPreparation::Resident(_) => {}
                    ImportPreparation::Pending {
                        key,
                        descriptor,
                        scale,
                        launch,
                        program,
                    } => {
                        let launch = program
                            .enqueue(&mut batch, launch)
                            .map_err(|(error, _)| WeightImportError::Submit(error))?;
                        if trace_batch == Some(report.batches - 1) {
                            traced_weights.push((
                                item.plan.descriptor.name.clone(),
                                item.plan.source.name().to_owned(),
                                item.plan.source_bytes,
                            ));
                        }
                        pending.push((key, descriptor, scale, launch));
                    }
                }
            }
            report.preparing += preparing.elapsed();
            if !pending.is_empty() {
                report.weights += pending.len();
                let trace = if trace_batch == Some(report.batches - 1) {
                    match self.device.trace_submissions(TraceDetail::Launches) {
                        Ok(trace) => Some(trace),
                        Err(error) => {
                            eprintln!("magnitude-engine: import trace unavailable: {error}");
                            None
                        }
                    }
                } else {
                    None
                };
                let submitting = Instant::now();
                let completion = batch.submit().map_err(|error| {
                    WeightImportError::Submit(SubmitError::Device(crate::DeviceError::Execution(
                        error.to_string(),
                    )))
                })?;
                report.submitting += submitting.elapsed();
                let waiting = Instant::now();
                completion.wait().map_err(|error| {
                    WeightImportError::Completion(crate::DeviceError::Execution(error.to_string()))
                })?;
                report.waiting += waiting.elapsed();
                if let Some(trace) = trace {
                    match trace.collect() {
                        Ok(submissions) => {
                            report_timed_import(report.batches - 1, &traced_weights, &submissions)
                        }
                        Err(error) => eprintln!("magnitude-engine: import trace failed: {error}"),
                    }
                }
                let publishing = Instant::now();
                for (key, descriptor, scale, launch) in pending {
                    let (_, workspace, destination) = launch.into_submission_parts();
                    drop(workspace);
                    let (tensor, bound) = destination.into_parts();
                    self.publish(
                        key,
                        ResidentWeight {
                            descriptor,
                            tensor,
                            scale,
                            bound,
                            table: ResidentTable::whole(),
                        },
                    )?;
                }
                report.publishing += publishing.elapsed();
            }
            first = last;
        }
        self.import_report = report;
        Ok(())
    }

    /// Import the target, reporting its resident bytes to `observer` from
    /// none to all of them.
    pub fn load_target(
        &mut self,
        definition: &magnitude_family_contracts::ModelDefinition,
        package: &magnitude_artifacts::Package,
        mut observer: ImportObserver,
    ) -> Result<crate::ResidentTarget, crate::ResidencyError> {
        crate::resident_weights::validate_definition_package(definition, package)?;
        let weights = self.execution.load().target().to_vec();
        let mut storage = HashSet::new();
        let bank = self
            .execution
            .load()
            .expert_bank()
            .map_or(Ok(0), crate::ExpertBankPlan::charged_beyond_records);
        let total = weights
            .iter()
            .filter(|weight| storage.insert(weight.storage_identity()))
            .map(WeightPlan::storage_bytes)
            .chain([bank])
            .sum::<Result<u64, String>>()
            .map_err(crate::ResidencyError::Invalid)?;
        observer(0, total);
        self.import_progress = Some(ImportProgress {
            completed: 0,
            total,
            observer,
        });
        let target = self
            .preload_components(
                package,
                &weights,
                crate::resident_weights::activation_dtype(definition.decoder.activation_dtype),
            )
            .map_err(crate::ResidencyError::from)
            .and_then(|()| crate::resident_weights::import_target(definition, package, self));
        let mut progress = self
            .import_progress
            .take()
            .expect("the target import owns its progress");
        if target.is_ok() && progress.completed < progress.total {
            (progress.observer)(progress.total, progress.total);
        }
        target
    }

    pub fn load_head(
        &mut self,
        definition: &magnitude_family_contracts::ModelDefinition,
        package: &magnitude_artifacts::Package,
    ) -> Result<Option<crate::ResidentHead>, crate::ResidencyError> {
        crate::resident_weights::validate_definition_package(definition, package)?;
        if let Some(weights) = self.execution.load().head().map(|weights| weights.to_vec()) {
            // A draft head's weights are the target's; a separate draft's
            // are its own component's, beside the target's tied ones.
            self.preload_components(
                package,
                &weights,
                crate::resident_weights::activation_dtype(definition.decoder.activation_dtype),
            )?;
        }
        crate::resident_weights::import_optional_head(definition, package, self)
    }

    pub fn load_vision(
        &mut self,
        definition: &magnitude_family_contracts::ModelDefinition,
        package: &magnitude_artifacts::Package,
    ) -> Result<Option<crate::ResidentVision>, crate::ResidencyError> {
        crate::resident_weights::validate_definition_package(definition, package)?;
        if let (Some(weights), Some(projector)) = (
            self.execution
                .load()
                .vision()
                .map(|weights| weights.to_vec()),
            package.projector(),
        ) {
            self.preload_component(projector, &weights, DType::F32)?;
        }
        crate::resident_weights::import_optional_vision(definition, package, self)
    }
}

/// A weight's resident second-level scale (`WeightPlan::scale`) on `device`:
/// its stored F32 values, one per matrix (a stored single value repeated per
/// matrix of a stack), uploaded as they are. A banked expert stack's is its
/// bank's.
pub(crate) fn import_scale(
    device: &Device,
    artifact: &GgufArtifact,
    scale: &crate::WeightScalePlan,
) -> Result<Tensor, WeightImportError> {
    let descriptor = WeightDescriptor::stored(scale.tensor.as_str(), [scale.stored]);
    let Stored::Dense(stored) = Stored::from_gguf(artifact, &descriptor)? else {
        return Err(invalid(format!(
            "second-level scale {:?} is not stored dense",
            scale.tensor
        )));
    };
    if stored.dtype != DType::F32 || stored.shape != [scale.stored] {
        return Err(invalid(format!(
            "second-level scale {:?} differs from the admitted WeightPlan",
            scale.tensor
        )));
    }
    let values = stored.read()?;
    let extent = usize::try_from(scale.extent)
        .map_err(|_| invalid("scale extent exceeds the host address range"))?;
    // One stored value serves every matrix; else one per matrix, of which a
    // plan of an expert stack's first experts (`WeightPlan::leading`) takes
    // the leading `extent`.
    let bytes = match scale.stored {
        1 => values.repeat(extent),
        _ => values[..extent * size_of::<f32>()].to_vec(),
    };
    Tensor::from_host(device, Element::f32(), &[scale.extent], &bytes)
        .map_err(|error| WeightImportError::allocating(seismic::Contract::DeviceOnly, &error))
}

/// Attribution only. Timed encoders add overhead, and one import entry has
/// one launch today. Keep a mismatch diagnostic rather than affecting load.
fn report_timed_import(
    batch: usize,
    weights: &[(String, String, u64)],
    submissions: &[TracedSubmission],
) {
    let launches = submissions
        .iter()
        .flat_map(|submission| &submission.launches)
        .collect::<Vec<_>>();
    if launches.len() != weights.len() {
        eprintln!(
            "magnitude-engine: timed import batch {batch}: {} launches for {} weights",
            launches.len(),
            weights.len()
        );
        return;
    }
    let mut by_format = BTreeMap::<String, (usize, u64, f64)>::new();
    let mut slowest = Vec::new();
    for (launch, (name, format, bytes)) in launches.into_iter().zip(weights) {
        let seconds = launch.device.map_or(0.0, |(start, end)| end - start);
        let entry = by_format.entry(format.clone()).or_default();
        entry.0 += 1;
        entry.1 += bytes;
        entry.2 += seconds;
        slowest.push((seconds, name, format, bytes));
    }
    eprintln!("magnitude-engine: timed import batch {batch} (measurement encoders)");
    for (format, (count, bytes, seconds)) in by_format {
        eprintln!(
            "magnitude-engine: import format={format} weights={count} source_bytes={bytes} device_ms={:.3}",
            seconds * 1000.0
        );
    }
    slowest.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (seconds, name, format, bytes) in slowest.into_iter().take(10) {
        eprintln!(
            "magnitude-engine: import slow_weight={name} format={format} source_bytes={bytes} device_ms={:.3}",
            seconds * 1000.0
        );
    }
}

type ComponentImport<T> = fn(
    &mut ResidencyStore,
    &magnitude_family_contracts::ModelDefinition,
    &magnitude_artifacts::Package,
) -> Result<Option<T>, crate::ResidencyError>;

/// A one-shot optional component authority. Numerical stages receive this
/// typed loader, never direct mutable access to the weight cache. Success or
/// failure is retained so concurrent request paths cannot trigger a second
/// import after the first attempt.
pub struct ComponentLoader<T: Clone> {
    store: RefCell<ResidencyStore>,
    definition: Rc<magnitude_family_contracts::ModelDefinition>,
    package: std::sync::Arc<magnitude_artifacts::Package>,
    import: ComponentImport<T>,
    cached: RefCell<Option<Result<T, Rc<crate::ResidencyError>>>>,
    /// The claims of the imports the component brought into its cache,
    /// released as they are ([`Self::unload`]).
    claims: RefCell<Vec<crate::memory::ledger::Claim>>,
}

impl<T: Clone> ComponentLoader<T> {
    /// Visit every tensor the component's resident imports hold.
    pub(crate) fn visit_storage(&self, visit: &mut dyn FnMut(&Tensor)) {
        self.store.borrow().visit_storage(visit);
    }

    /// The bytes of the imports the component brought into its cache, as
    /// imported: none of another component's it found there.
    pub fn own_bytes(&self) -> Result<u64, &'static str> {
        let store = self.store.borrow();
        store.resident.bytes(store.importer)
    }

    /// Drop the imports the component brought into its cache; another
    /// component's it found there stay resident with their storage identity.
    pub fn unload(&self) {
        if matches!(&*self.cached.borrow(), Some(Ok(_))) {
            *self.cached.borrow_mut() = None;
        }
        let store = self.store.borrow();
        // The import's free and its claims' release are one change.
        store.changes.clone().change(|_| {
            store.resident.release(store.importer);
            self.claims.borrow_mut().clear();
        });
    }

    fn new(
        mut store: ResidencyStore,
        definition: Rc<magnitude_family_contracts::ModelDefinition>,
        package: std::sync::Arc<magnitude_artifacts::Package>,
        import: ComponentImport<T>,
        importer: Importer,
    ) -> Self {
        store.importer = importer;
        Self {
            store: RefCell::new(store),
            definition,
            package,
            import,
            cached: RefCell::new(None),
            claims: RefCell::new(Vec::new()),
        }
    }

    /// Whether the component is imported ([`Self::import`]).
    pub fn imported(&self) -> bool {
        matches!(&*self.cached.borrow(), Some(Ok(_)))
    }

    /// Import the component under `held`, the claims of the bytes its
    /// import brings into its cache, kept until [`Self::unload`]. An attempt
    /// made before is returned as it ended, `held` released.
    pub fn import(&self, mut held: crate::memory::ledger::Held) -> Result<T, Rc<crate::ResidencyError>> {
        assert!(!self.imported(), "invariant violated: a component is imported once until unloaded");
        let imported = held.allocate(|| self.load());
        if imported.is_ok() {
            *self.claims.borrow_mut() = held.into_claims();
        }
        imported
    }

    pub fn load(&self) -> Result<T, Rc<crate::ResidencyError>> {
        if let Some(cached) = self.cached.borrow().as_ref() {
            return cached.clone();
        }
        let imported = (self.import)(
            &mut self.store.borrow_mut(),
            &self.definition,
            &self.package,
        )
        .and_then(|component| {
            component.ok_or_else(|| {
                crate::ResidencyError::Invalid(
                    "enabled component is absent from the package".into(),
                )
            })
        })
        .map_err(Rc::new);
        *self.cached.borrow_mut() = Some(imported.clone());
        imported
    }
}

impl ComponentLoader<crate::ResidentHead> {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        self.store
            .borrow()
            .programs
            .drafter_graphs()
            .ok_or("head graph family was not prepared")?
            .binding_constant_bytes()
    }

    pub fn head(
        store: ResidencyStore,
        definition: Rc<magnitude_family_contracts::ModelDefinition>,
        package: std::sync::Arc<magnitude_artifacts::Package>,
    ) -> Result<Self, WeightImportError> {
        if definition.head.is_none() && definition.draft.is_none() {
            return Err(invalid("the model definition enables no drafter"));
        }
        Ok(Self::new(
            store,
            definition,
            package,
            ResidencyStore::load_head,
            Importer::Head,
        ))
    }
}

impl ComponentLoader<crate::ResidentVision> {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        self.store
            .borrow()
            .programs
            .vision_graphs()
            .ok_or("vision graph family was not prepared")?
            .binding_constant_bytes()
    }

    pub fn vision(
        store: ResidencyStore,
        definition: Rc<magnitude_family_contracts::ModelDefinition>,
        package: std::sync::Arc<magnitude_artifacts::Package>,
    ) -> Result<Self, WeightImportError> {
        if definition.vision.is_none() {
            return Err(invalid(
                "vision component is not enabled by the model definition",
            ));
        }
        Ok(Self::new(
            store,
            definition,
            package,
            ResidencyStore::load_vision,
            Importer::Vision,
        ))
    }
}

fn validate_request(
    descriptor: &WeightDescriptor,
    stored: &Stored,
    target: DType,
) -> Result<usize, WeightImportError> {
    if !matches!(target, DType::F32 | DType::F16 | DType::BF16) {
        return Err(invalid("weight target must be F32, F16, or BF16"));
    }
    if descriptor
        .transformed_shape(stored.shape())
        .map_err(|error| invalid(error.to_string()))?
        != descriptor.shape
    {
        return Err(invalid(
            "stored weight does not transform to its logical descriptor",
        ));
    }
    // The stored tensor's own byte layout, before any transform.
    let count = element_count(stored.shape())?;
    match stored {
        Stored::Dense(stored) => {
            if validate_dense(stored)? != count {
                return Err(invalid(
                    "stored dense weight does not match its logical descriptor",
                ));
            }
            if !matches!(stored.dtype, DType::F32 | DType::F16 | DType::BF16) {
                return Err(invalid("stored dense weight must be F32, F16, or BF16"));
            }
        }
        Stored::GgmlBlocks {
            nbytes,
            shape,
            encoding,
            ..
        } => {
            if shape
                .last()
                .is_none_or(|extent| !extent.is_multiple_of(encoding.block_elements()))
            {
                return Err(invalid("invalid packed weight block geometry"));
            }
            let blocks = u64::try_from(count)
                .ok()
                .and_then(|count| count.checked_div(encoding.block_elements()))
                .ok_or_else(|| invalid("packed weight block count overflow"))?;
            let expected = blocks
                .checked_mul(encoding.block_bytes())
                .ok_or_else(|| invalid("packed weight byte count overflow"))?;
            if *nbytes != expected {
                return Err(invalid("packed weight byte count mismatch"));
            }
            if stored.source_element().is_none() {
                return Err(invalid(format!(
                    "GGUF encoding {encoding:?} has no qualified native import"
                )));
            }
        }
    }
    Ok(count)
}

fn element_count(shape: &[u64]) -> Result<usize, WeightImportError> {
    if shape.is_empty() {
        return Err(invalid("weight shape must not be empty"));
    }
    shape.iter().try_fold(1usize, |count, extent| {
        if *extent == 0 {
            return Err(invalid("weight dimensions must be positive"));
        }
        count
            .checked_mul(
                usize::try_from(*extent)
                    .map_err(|_| invalid("weight dimension exceeds host address range"))?,
            )
            .ok_or_else(|| invalid("weight element count overflow"))
    })
}

fn validate_dense(stored: &StoredTensor) -> Result<usize, WeightImportError> {
    let count = element_count(&stored.shape)?;
    let bytes = count
        .checked_mul(stored.dtype.bytes() as usize)
        .ok_or_else(|| invalid("stored dense byte count overflow"))?;
    if u64::try_from(bytes).ok() != Some(stored.nbytes) {
        return Err(invalid("stored dense byte count mismatch"));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An expert stack's first experts are a prefix of its bytes and of its
    /// scale grid's: `leading` reads them alone, and refuses what does not
    /// divide into its leading rows.
    #[test]
    fn a_stacks_leading_experts_are_a_prefix_of_its_bytes_and_its_grid() {
        let path = std::env::temp_dir().join(format!("magnitude-leading-stack-{}.bin", std::process::id()));
        // 8 experts of 6 bytes after 16 bytes of header, then 8 of 2 bytes.
        let bytes = [vec![0xee; 16], (0..48).collect::<Vec<u8>>(), (100..116).collect::<Vec<u8>>()].concat();
        std::fs::write(&path, &bytes).unwrap();
        let source = Arc::new(FileSource::open(&path).unwrap());
        let stack = Stored::GgmlBlocks {
            source: source.clone(),
            offset: 16,
            nbytes: 48,
            shape: vec![8, 3, 2],
            encoding: Encoding::Q4K,
            grid: Some(StoredGrid { source, offset: 64, nbytes: 16, codes: ScaleCodes::Ue8m0 }),
        };
        let Some(Stored::GgmlBlocks { offset, nbytes, shape, grid: Some(grid), .. }) = stack.leading(3) else {
            panic!("a stack of 8 experts has a leading 3");
        };
        assert_eq!((offset, nbytes, shape), (16, 18, vec![3, 3, 2]));
        assert_eq!((grid.offset, grid.nbytes), (64, 6));
        assert!(stack.leading(9).is_none(), "more experts than the stack holds");
        assert!(stack.leading(8).is_some());
        std::fs::remove_file(&path).unwrap();
    }

    /// A reserved stored table the bank shrinks after its import is read
    /// at what its slots are charged then: the importer's bytes fall with
    /// the slots' charge, never above it.
    #[test]
    fn an_imported_table_is_counted_at_its_resized_charge() {
        const ROWS: u64 = 4096;
        let path = std::env::temp_dir().join(format!("magnitude-imported-table-{}.bin", std::process::id()));
        std::fs::write(&path, vec![0u8; 100 + ROWS as usize * 90]).unwrap();
        let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let descriptor = WeightDescriptor::stored("ngram", [ROWS, 160]);
        let table = Arc::new(
            StoredTable::new(
                &device,
                &descriptor,
                Element::named("gguf_iq4_nl").unwrap(),
                Element::named("iq4g32s@rows16").unwrap(),
                ROWS,
                160,
                crate::stored_table::StoredSlots::Reserved { records: 4, record_rows: 256 },
                crate::planning::DirectoryForm::Hashed,
                vec![crate::stored_table::RowRange {
                    source: seismic_io::Reader::open(&path, seismic_io::CachePolicy::Uncached).unwrap(),
                    first: 100,
                    bytes: 90,
                }],
                None,
            )
            .unwrap(),
        );
        let key = ResidencyKey {
            artifact: magnitude_artifacts::ArtifactIdentity([0; 32]),
            name: descriptor.name.clone(),
            transforms: Vec::new(),
            resident: Element::named("iq4g32s@rows16").unwrap(),
        };
        let weight = ResidentWeight::stored(descriptor, table.clone());
        let imports = Imports::default();
        let imported = imports.insert(key, weight.clone(), Importer::Target);
        table.resize(2).unwrap().unwrap();
        assert!(weight.storage_bytes() < imported);
        assert_eq!(imports.bytes(Importer::Target), Ok(weight.storage_bytes()));
        drop((imports, weight, table));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checked_element_count_rejects_invalid_shapes() {
        assert_eq!(element_count(&[2, 3, 4]).unwrap(), 24);
        assert!(element_count(&[]).is_err());
        assert!(element_count(&[2, 0]).is_err());
        assert!(element_count(&[u64::MAX, 2]).is_err());
    }

    #[test]
    fn residency_keys_separate_artifact_components() {
        let target = ResidencyKey {
            artifact: magnitude_artifacts::ArtifactIdentity([1; 32]),
            name: "shared.weight".into(),
            transforms: Vec::new(),
            resident: Element::bf16(),
        };
        let projector = ResidencyKey {
            artifact: magnitude_artifacts::ArtifactIdentity([2; 32]),
            name: "shared.weight".into(),
            transforms: Vec::new(),
            resident: Element::bf16(),
        };
        assert_ne!(target, projector);
        // Two row ranges of one stored tensor are distinct resident weights.
        let first_rows = ResidencyKey {
            transforms: vec![magnitude_family_contracts::ImportTransform::Rows(
                magnitude_family_contracts::RowRange { start: 0, rows: 2 },
            )],
            ..target.clone()
        };
        assert_ne!(target, first_rows);
    }

    /// Each importer's bytes are its own imports', as imported, never what
    /// is left of another's: a head finding the target's weights in the
    /// cache it inherits counts none of them, and releasing its imports
    /// keeps every one of the target's.
    #[test]
    fn each_importers_bytes_are_its_own_imports() {
        let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let weight = |name: &str, elements: u64| {
            let key = ResidencyKey {
                artifact: magnitude_artifacts::ArtifactIdentity([0; 32]),
                name: name.into(),
                transforms: Vec::new(),
                resident: Element::f32(),
            };
            let weight = ResidentWeight {
                descriptor: WeightDescriptor { name: name.into(), shape: vec![elements], transforms: Vec::new() },
                tensor: Tensor::zeros(&device, Element::f32(), &[elements]).unwrap(),
                scale: None,
                bound: None,
                table: ResidentTable::Whole,
            };
            (key, weight)
        };
        let imports = Imports::default();
        let mut owned = [0u64; 2];
        for (name, elements, importer) in [("embedding", 64, Importer::Target), ("output", 32, Importer::Target), ("draft", 16, Importer::Head)] {
            let (key, weight) = weight(name, elements);
            let bytes = weight.tensor.charged_bytes();
            assert_eq!(imports.insert(key, weight, importer), bytes);
            owned[usize::from(importer == Importer::Head)] += bytes;
        }
        assert_eq!(imports.bytes(Importer::Target), Ok(owned[0]));
        assert_eq!(imports.bytes(Importer::Head), Ok(owned[1]));
        assert_eq!(imports.bytes(Importer::Vision), Ok(0));
        imports.release(Importer::Head);
        assert_eq!(imports.bytes(Importer::Head), Ok(0));
        assert_eq!(imports.bytes(Importer::Target), Ok(owned[0]));
        assert!(imports.get(&weight("embedding", 64).0).is_some(), "the target's imports stay resident");
    }

    /// An allocation its store has no room for is refused for room, in the
    /// storage it was made in, never a device failure: a CPU device whose
    /// store Seismic bounds to what it claims refuses a 1 MiB weight.
    #[test]
    fn an_allocation_without_room_is_refused_for_room() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).unwrap();
        let bounds = device.bounds();
        for view in device.ledgers().stores {
            bounds.bound(seismic::Ledger::Store(view.store), Some(view.claim)).unwrap();
        }
        let error = Tensor::zeros(&device, Element::bool(), &[1 << 20]).expect_err("a store with no room refuses");
        assert!(
            matches!(
                WeightImportError::allocating(seismic::Contract::DeviceOnly, &error),
                WeightImportError::Refused { contract: seismic::Contract::DeviceOnly, required, .. } if required >= 1 << 20
            ),
            "{error}"
        );
    }
}
