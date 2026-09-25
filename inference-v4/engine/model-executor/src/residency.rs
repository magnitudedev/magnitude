//! Authoritative device residency for family-neutral semantic weights.
//!
//! Artifact parsing supplies immutable stored slices. This module validates
//! their logical descriptor, imports them through the once-prepared execution
//! path, and owns the sole cache of resulting device tensors.

use crate::programs::native_import::NativeImportProgram;
use crate::programs::ProgramSubmission;
use crate::resources::ImportWindow;
use crate::{resident_element, source_element};
use crate::{
    AllocationError, ExecutionPlan, ImportLaunchInputs, ImportProgram, InvariantError,
    ResidentWeightSlot, ResourceAllocator, ResourceDomainId, SubmitError, ValidatedImportLaunch,
    WeightPlan, WeightStorageIdentity,
};
use magnitude_artifacts::{
    gguf::{Encoding, GgufArtifact},
    Error as ArtifactError, FileSource,
};
use magnitude_model_contracts::WeightDescriptor;
use seismic::{DType, Device, Element, NativeTensorBatch, Tensor, TraceDetail, TracedSubmission};
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
    Invalid(String),
    Device(String),
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
            Self::Invalid(message) => formatter.write_str(message),
            Self::Device(message) => write!(formatter, "weight device import: {message}"),
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
            Self::Allocation(error) => Some(error),
            Self::Attestation(error) => Some(error),
            Self::Submit(error) => Some(error),
            Self::Completion(error) => Some(error),
            Self::Invariant(error) => Some(error),
            Self::Invalid(_) | Self::Device(_) => None,
        }
    }
}

impl From<ArtifactError> for WeightImportError {
    fn from(error: ArtifactError) -> Self {
        Self::Artifact(error)
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

/// Storage representation of one semantic model weight.
#[derive(Clone, Debug)]
pub enum Stored {
    Dense(StoredTensor),
    GgmlBlocks {
        source: Arc<FileSource>,
        offset: u64,
        nbytes: u64,
        shape: Vec<u64>,
        encoding: Encoding,
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

    pub fn source_element(&self) -> Option<Element> {
        match self {
            Self::Dense(tensor) => Some(Element::dense(tensor.dtype)),
            Self::GgmlBlocks { encoding, .. } => source_element(*encoding),
        }
    }

    pub fn from_gguf(
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
    ) -> Result<Self, WeightImportError> {
        let tensor = artifact.tensor(&descriptor.name)?;
        match tensor.encoding {
            Encoding::F32 | Encoding::F16 | Encoding::BF16 => Ok(Self::Dense(StoredTensor {
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
            })),
            encoding => Ok(Self::GgmlBlocks {
                source: tensor.source,
                offset: tensor.offset,
                nbytes: tensor.nbytes,
                shape: tensor.shape,
                encoding,
            }),
        }
    }
}

#[derive(Clone)]
pub struct ResidentWeight {
    descriptor: WeightDescriptor,
    tensor: Tensor,
}

impl ResidentWeight {
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
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ResidencyKey {
    artifact: magnitude_artifacts::ArtifactIdentity,
    name: String,
    resident: Element,
}

enum ImportPreparation {
    Resident(ResidentWeight),
    Pending {
        key: ResidencyKey,
        descriptor: WeightDescriptor,
        launch: ValidatedImportLaunch,
        program: NativeImportProgram,
    },
}

struct OrderedImport {
    plan: WeightPlan,
    source: Arc<FileSource>,
    offset: u64,
    end: u64,
    target: DType,
}

/// Coarse host costs of the most recent mapped component import. The caller
/// can use these alongside its total import duration to identify work outside
/// mapping, preparation, submission, waiting, and publication.
#[derive(Clone, Copy, Debug, Default)]
pub struct MappedImportReport {
    pub windows: usize,
    pub weights: usize,
    pub mapping: Duration,
    pub preparing: Duration,
    pub submitting: Duration,
    pub waiting: Duration,
    pub publishing: Duration,
}

/// Sole cache and importer for one planned device. Resident components are
/// assembled from this store before their numerical programs are bound.
pub struct ResidencyStore {
    device: Rc<Device>,
    programs: Rc<crate::native::AttestedPrograms>,
    execution: ExecutionPlan,
    domain: ResourceDomainId,
    resident: HashMap<ResidencyKey, ResidentWeight>,
    mapped_import: MappedImportReport,
}

impl ResidencyStore {
    pub fn new(
        device: Rc<Device>,
        programs: Rc<crate::native::AttestedPrograms>,
        execution: ExecutionPlan,
        domain: ResourceDomainId,
    ) -> Result<Self, WeightImportError> {
        if !programs.belongs_to(&device)
            || execution.device().backend() != device.backend()
            || execution.device().name() != device.info().name
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
            resident: HashMap::new(),
            mapped_import: MappedImportReport::default(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn belongs_to(&self, device: &Device) -> bool {
        self.device.info().id == device.info().id
    }
    pub fn len(&self) -> usize {
        self.resident.len()
    }
    pub fn is_empty(&self) -> bool {
        self.resident.is_empty()
    }

    /// Charges of completed resident imports held by this cache. Tied roles
    /// share one key and one allocation; a failed component import may leave
    /// a partial set, which this observation still counts exactly.
    pub fn resident_bytes(&self) -> Result<u64, &'static str> {
        self.resident.values().try_fold(0u64, |bytes, weight| {
            bytes
                .checked_add(weight.tensor.storage_bytes())
                .ok_or("resident cache charge overflows")
        })
    }

    pub fn mapped_import_report(&self) -> MappedImportReport {
        self.mapped_import
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
                weight.component.identity == artifact.identity()
                    && weight.descriptor.name == descriptor.name
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
        let (key, descriptor, launch, mut program) =
            match self.prepare_gguf(artifact, descriptor, target, None)? {
                ImportPreparation::Resident(weight) => return Ok(weight),
                ImportPreparation::Pending {
                    key,
                    descriptor,
                    launch,
                    program,
                } => (key, descriptor, launch, program),
            };
        let submission = program
            .submit(launch)
            .map_err(|(error, _launch)| WeightImportError::Submit(error))?;
        let completed = submission.finish().map_err(WeightImportError::Completion)?;
        let (_, destination) = completed.into_parts();
        let weight = ResidentWeight {
            descriptor,
            tensor: destination.into_tensor(),
        };
        self.resident.insert(key, weight.clone());
        Ok(weight)
    }

    fn prepare_gguf(
        &self,
        artifact: &GgufArtifact,
        descriptor: &WeightDescriptor,
        target: DType,
        window: Option<&ImportWindow>,
    ) -> Result<ImportPreparation, WeightImportError> {
        let planned = self
            .execution
            .weights()
            .find(|weight| {
                weight.component.identity == artifact.identity()
                    && weight.descriptor.name == descriptor.name
            })
            .cloned()
            .ok_or_else(|| {
                invalid(format!(
                    "weight {:?} is absent from the admitted WeightPlan",
                    descriptor.name
                ))
            })?;
        if planned.descriptor != *descriptor
            || planned
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
        let actual_resident = match stored {
            Stored::Dense(_) => Element::dense(target),
            Stored::GgmlBlocks { encoding, .. } => resident_element(*encoding, target, layout)
                .ok_or_else(|| invalid("unsupported packed resident representation"))?,
        };
        if planned.source != actual_source
            || planned.resident != actual_resident
            || planned.source_bytes != stored.source_bytes()
        {
            return Err(invalid(format!(
                "weight {:?} representation differs from the admitted WeightPlan",
                descriptor.name
            )));
        }
        let key = ResidencyKey {
            artifact: artifact.identity(),
            name: descriptor.name.clone(),
            resident: actual_resident,
        };
        if let Some(weight) = self.resident.get(&key) {
            if weight.descriptor() != descriptor {
                return Err(invalid(
                    "resident tensor has a conflicting logical descriptor",
                ));
            }
            return Ok(ImportPreparation::Resident(weight.clone()));
        }
        let tensor = if self.device.backend() == seismic::BackendName::Metal {
            // SAFETY: the attested import's only result is this tensor. Both
            // Metal import kernels write every physical byte, including the
            // packed layout's row and plane padding, before submission returns.
            unsafe { Tensor::uninitialized(&self.device, planned.resident, &planned.shape) }
        } else {
            Tensor::zeros(&self.device, planned.resident, &planned.shape)
        }
        .map_err(|error| WeightImportError::Device(error.to_string()))?;
        let destination = ResidentWeightSlot::new(&planned, &self.device, tensor)
            .map_err(WeightImportError::Invariant)?;
        let workspace = ResourceAllocator::import_workspace(
            &self.execution,
            &planned,
            stored,
            window,
            &self.device,
            self.domain.clone(),
        )
        .map_err(WeightImportError::Allocation)?;
        let launch = ValidatedImportLaunch::new(
            ImportLaunchInputs::new(planned.clone(), source, workspace, destination),
            &self.domain,
        )
        .map_err(|(_, error)| WeightImportError::Invariant(error))?;
        let program = self
            .programs
            .bind_import(&planned)
            .map_err(WeightImportError::Attestation)?;
        Ok(ImportPreparation::Pending {
            key,
            descriptor: descriptor.clone(),
            launch,
            program,
        })
    }

    /// Import one component in source-file order. Whole adjacent tensors
    /// share a mapped window and one ordered native submission. Semantic
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
        let mut seen = HashSet::<WeightStorageIdentity>::new();
        let mut ordered = Vec::new();
        for plan in weights {
            if !seen.insert(plan.storage_identity()) {
                continue;
            }
            let tensor = ImportArtifactTensor::from_gguf(artifact, &plan.descriptor)?;
            let (source, offset, length) = tensor.stored().file_range();
            let end = offset
                .checked_add(length)
                .ok_or_else(|| invalid("source range overflows"))?;
            ordered.push(OrderedImport {
                plan: plan.clone(),
                source: source.clone(),
                offset,
                end,
                target: plan.resident.dtype().unwrap_or(activation),
            });
        }
        ordered.sort_by_key(|item| item.offset);
        let largest = ordered
            .iter()
            .map(|item| item.plan.source_bytes)
            .max()
            .unwrap_or(0);
        let mut report = MappedImportReport::default();
        // Measurement only: timed Metal encoders change the device path, so
        // these numbers attribute work but are not production load timings.
        let trace_window = std::env::var("MAGNITUDE_TRACE_IMPORT_WINDOW")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let mut first = 0;
        while first < ordered.len() {
            let source = ordered[first].source.clone();
            let start = ordered[first].offset;
            let mut end = ordered[first].end;
            let mut last = first + 1;
            while last < ordered.len()
                && Arc::ptr_eq(&source, &ordered[last].source)
                && ordered[last].end.saturating_sub(start) <= largest
            {
                end = end.max(ordered[last].end);
                last += 1;
            }
            let mapping = Instant::now();
            let window = ImportWindow::new(&self.execution, &self.device, source, start, end)
                .map_err(WeightImportError::Allocation)?;
            report.mapping += mapping.elapsed();
            report.windows += 1;
            let mut batch = NativeTensorBatch::new(&self.device);
            let mut pending = Vec::new();
            let mut traced_weights = Vec::new();
            let preparing = Instant::now();
            for item in &ordered[first..last] {
                match self.prepare_gguf(
                    artifact,
                    &item.plan.descriptor,
                    item.target,
                    Some(&window),
                )? {
                    ImportPreparation::Resident(_) => {}
                    ImportPreparation::Pending {
                        key,
                        descriptor,
                        launch,
                        program,
                    } => {
                        let launch = program
                            .enqueue(&mut batch, launch)
                            .map_err(|(error, _)| WeightImportError::Submit(error))?;
                        if trace_window == Some(report.windows - 1) {
                            traced_weights.push((
                                item.plan.descriptor.name.clone(),
                                item.plan.source.name().to_owned(),
                                item.plan.source_bytes,
                            ));
                        }
                        pending.push((key, descriptor, launch));
                    }
                }
            }
            report.preparing += preparing.elapsed();
            if !pending.is_empty() {
                report.weights += pending.len();
                let trace = if trace_window == Some(report.windows - 1) {
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
                            report_timed_import(report.windows - 1, &traced_weights, &submissions)
                        }
                        Err(error) => eprintln!("magnitude-engine: import trace failed: {error}"),
                    }
                }
                let publishing = Instant::now();
                for (key, descriptor, launch) in pending {
                    let (_, workspace, destination) = launch.into_submission_parts();
                    drop(workspace);
                    self.resident.insert(
                        key,
                        ResidentWeight {
                            descriptor,
                            tensor: destination.into_tensor(),
                        },
                    );
                }
                report.publishing += publishing.elapsed();
            }
            first = last;
        }
        self.mapped_import = report;
        Ok(())
    }

    pub fn load_target(
        &mut self,
        definition: &magnitude_model_contracts::ModelDefinition,
        package: &magnitude_artifacts::Package,
    ) -> Result<crate::ResidentTarget, crate::ResidencyError> {
        crate::resident_weights::validate_definition_package(definition, package)?;
        let weights = self.execution.load().target().to_vec();
        self.preload_component(
            package.target(),
            &weights,
            crate::resident_weights::activation_dtype(definition.geometry.activation_dtype),
        )?;
        crate::resident_weights::import_target(definition, package, self)
    }

    pub fn load_head(
        &mut self,
        definition: &magnitude_model_contracts::ModelDefinition,
        package: &magnitude_artifacts::Package,
    ) -> Result<Option<crate::ResidentHead>, crate::ResidencyError> {
        crate::resident_weights::validate_definition_package(definition, package)?;
        if let Some(weights) = self.execution.load().head().map(|weights| weights.to_vec()) {
            self.preload_component(
                package.target(),
                &weights,
                crate::resident_weights::activation_dtype(definition.geometry.activation_dtype),
            )?;
        }
        crate::resident_weights::import_optional_head(definition, package, self)
    }

    pub fn load_vision(
        &mut self,
        definition: &magnitude_model_contracts::ModelDefinition,
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

/// Attribution only. Timed encoders add overhead, and one import entry has
/// one launch today. Keep a mismatch diagnostic rather than affecting load.
fn report_timed_import(
    window: usize,
    weights: &[(String, String, u64)],
    submissions: &[TracedSubmission],
) {
    let launches = submissions
        .iter()
        .flat_map(|submission| &submission.launches)
        .collect::<Vec<_>>();
    if launches.len() != weights.len() {
        eprintln!(
            "magnitude-engine: timed import window {window}: {} launches for {} weights",
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
    eprintln!("magnitude-engine: timed import window {window} (measurement encoders)");
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
    &magnitude_model_contracts::ModelDefinition,
    &magnitude_artifacts::Package,
) -> Result<Option<T>, crate::ResidencyError>;

/// A one-shot optional component authority. Numerical stages receive this
/// typed loader, never direct mutable access to the weight cache. Success or
/// failure is retained so concurrent request paths cannot trigger a second
/// import after the first attempt.
pub struct ComponentLoader<T: Clone> {
    store: RefCell<ResidencyStore>,
    definition: Rc<magnitude_model_contracts::ModelDefinition>,
    package: std::sync::Arc<magnitude_artifacts::Package>,
    import: ComponentImport<T>,
    cached: RefCell<Option<Result<T, Rc<crate::ResidencyError>>>>,
}

impl<T: Clone> ComponentLoader<T> {
    pub fn resident_bytes(&self) -> Result<u64, &'static str> {
        self.store.borrow().resident_bytes()
    }

    fn new(
        store: ResidencyStore,
        definition: Rc<magnitude_model_contracts::ModelDefinition>,
        package: std::sync::Arc<magnitude_artifacts::Package>,
        import: ComponentImport<T>,
    ) -> Self {
        Self {
            store: RefCell::new(store),
            definition,
            package,
            import,
            cached: RefCell::new(None),
        }
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
    /// Drop only imports introduced by the head. Target weights sharing the
    /// cache stay resident and retain their original storage identity.
    pub fn unload(&self) {
        let loaded = matches!(&*self.cached.borrow(), Some(Ok(_)));
        if loaded {
            *self.cached.borrow_mut() = None;
        }
        let mut store = self.store.borrow_mut();
        let target = store
            .execution
            .load()
            .target()
            .iter()
            .map(|weight| ResidencyKey {
                artifact: weight.component.identity,
                name: weight.descriptor.name.clone(),
                resident: weight.resident,
            })
            .collect::<HashSet<_>>();
        store.resident.retain(|key, _| target.contains(key));
    }

    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        self.store
            .borrow()
            .programs
            .head_graphs()
            .ok_or("head graph family was not prepared")?
            .binding_constant_bytes()
    }

    pub fn head(
        store: ResidencyStore,
        definition: Rc<magnitude_model_contracts::ModelDefinition>,
        package: std::sync::Arc<magnitude_artifacts::Package>,
    ) -> Result<Self, WeightImportError> {
        if definition.head.is_none() {
            return Err(invalid(
                "head component is not enabled by the model definition",
            ));
        }
        Ok(Self::new(
            store,
            definition,
            package,
            ResidencyStore::load_head,
        ))
    }
}

impl ComponentLoader<crate::ResidentVision> {
    pub fn unload(&self) {
        let loaded = matches!(&*self.cached.borrow(), Some(Ok(_)));
        if loaded {
            *self.cached.borrow_mut() = None;
        }
        self.store.borrow_mut().resident.clear();
    }

    pub fn vision(
        store: ResidencyStore,
        definition: Rc<magnitude_model_contracts::ModelDefinition>,
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
    let count = element_count(&descriptor.shape)?;
    match stored {
        Stored::Dense(stored) => {
            if stored.shape != descriptor.shape || validate_dense(stored)? != count {
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
            if shape != &descriptor.shape || element_count(shape)? != count {
                return Err(invalid(
                    "stored packed weight does not match its logical descriptor",
                ));
            }
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
            if source_element(*encoding).is_none() {
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

    #[test]
    fn catalog_encoding_map_is_exact() {
        for encoding in [
            Encoding::Q8_0,
            Encoding::Q4K,
            Encoding::Q5K,
            Encoding::Q6K,
            Encoding::Iq4Xs,
        ] {
            assert!(source_element(encoding).is_some());
            for layout in seismic::Layout::ALL {
                assert!(resident_element(encoding, DType::BF16, layout).is_some());
            }
        }
        for encoding in [
            Encoding::Q4_0,
            Encoding::Q5_0,
            Encoding::Q5_1,
            Encoding::Q3K,
            Encoding::Iq4Nl,
            Encoding::Iq3S,
            Encoding::Mxfp4,
            Encoding::Nvfp4,
            Encoding::Q1_0,
            Encoding::I32,
        ] {
            assert!(source_element(encoding).is_none());
            assert!(resident_element(encoding, DType::BF16, seismic::Layout::Rows16).is_none());
        }
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
            resident: Element::bf16(),
        };
        let projector = ResidencyKey {
            artifact: magnitude_artifacts::ArtifactIdentity([2; 32]),
            name: "shared.weight".into(),
            resident: Element::bf16(),
        };
        assert_ne!(target, projector);
    }
}
