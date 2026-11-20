//! Safe, language-neutral calls to entries discovered after the host was compiled.
//! Signature trees are projections of checked source; execution uses the typed API's runtime.
use super::*;
use seismic_lang::checked::{CheckedModule, EntryInfo, SourceFile, SourceSet};
pub use seismic_lang::checked::{ElementSummary, SignatureType, TensorAccess};
use seismic_runtime::api::kernel as runtime;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex, OnceLock, Weak,
};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Error {
    pub kind: &'static str,
    pub message: String,
}
impl Error {
    pub fn new(kind: &'static str, message: impl ToString) -> Self {
        Self {
            kind,
            message: message.to_string(),
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for Error {}
impl From<CallError> for Error {
    fn from(e: CallError) -> Self {
        let kind = match &e {
            CallError::Invocation(_) => "InvocationError",
            CallError::Output(_) => "OutputError",
            CallError::Execution(_) | CallError::Unfenced { .. } => "ExecutionError",
            CallError::Workflow(_) => "WorkflowError",
        };
        Self::new(kind, e)
    }
}
impl From<TensorError> for Error {
    fn from(e: TensorError) -> Self {
        Self::new("TensorError", e)
    }
}
impl From<ExecutionError> for Error {
    fn from(e: ExecutionError) -> Self {
        Self::new("ExecutionError", e)
    }
}
impl From<runtime::PrepareError> for Error {
    fn from(e: runtime::PrepareError) -> Self {
        match e {
            runtime::PrepareError::Source(e) => Self::new("SourceError", e),
            runtime::PrepareError::Preparation(e) => Self::new("PreparationError", e),
        }
    }
}
impl From<seismic_lang::source::LoadError> for Error {
    fn from(e: seismic_lang::source::LoadError) -> Self {
        let kind = match &e {
            seismic_lang::source::LoadError::Io(_) => "OSError",
            seismic_lang::source::LoadError::Source(_) => "SourceError",
            _ => "ValueError",
        };
        Self::new(kind, e)
    }
}
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone)]
pub struct Module {
    checked: Arc<CheckedModule>,
}
impl Module {
    pub fn load(paths: &[PathBuf], include_std: bool) -> Result<Self, Error> {
        if paths.is_empty() {
            return Err(Error::new("ValueError", "load requires at least one path"));
        }
        if paths.len() == 1 && paths[0].extension().is_some_and(|s| s == "seismicbundle") {
            if !include_std {
                return Err(Error::new(
                    "ValueError",
                    "std cannot be disabled for a checked bundle",
                ));
            }
            let bytes = std::fs::read(&paths[0]).map_err(|e| Error::new("OSError", e))?;
            // A bundle file is external input: its sources are checked here.
            let checked = seismic_lang::bundle::check_bundle_sources(&bytes)
                .map_err(|e| Error::new("BundleError", e))?;
            return Ok(Self {
                checked: Arc::new(checked),
            });
        }
        let prelude = if include_std {
            seismic_std::sources()
        } else {
            SourceSet::default()
        };
        let checked = seismic_lang::source::load(paths, prelude)?.module;
        Ok(Self {
            checked: Arc::new(checked),
        })
    }
    pub fn source(
        text: &str,
        name: &str,
        include_std: bool,
        base: Option<&Path>,
    ) -> Result<Self, Error> {
        let mut sources = if include_std {
            seismic_std::sources()
        } else {
            SourceSet::default()
        };
        sources.push(SourceFile {
            path: name.into(),
            text: text.into(),
        });
        // Native includes of inline assets may resolve only inside `base`;
        // labels never implicitly grant filesystem access for them.
        let roots = match base {
            Some(base) => seismic_lang::source::source_roots(&[base.to_path_buf()])?,
            None => Vec::new(),
        };
        let (checked, _) = seismic_lang::source::check_with_assets(sources, base, &roots)?;
        Ok(Self {
            checked: Arc::new(checked),
        })
    }
    pub fn identity(&self) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(seismic_lang::bundle::encode_checked_bundle(&self.checked))
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        std::fs::write(
            path,
            seismic_lang::bundle::encode_checked_bundle(&self.checked),
        )
        .map_err(|e| Error::new("OSError", e))
    }
    pub fn functions(&self) -> Vec<Function> {
        (0..self.checked.entries().len())
            .map(|index| Function {
                module: self.clone(),
                index,
            })
            .collect()
    }
    pub fn function(&self, name: &str) -> Result<Function, Error> {
        let index = self
            .checked
            .entries()
            .iter()
            .position(|e| e.name == name)
            .ok_or_else(|| Error::new("KeyError", name))?;
        Ok(Function {
            module: self.clone(),
            index,
        })
    }
}

#[derive(Clone)]
pub struct Function {
    module: Module,
    index: usize,
}
impl Function {
    fn info(&self) -> &EntryInfo {
        &self.module.checked.entries()[self.index]
    }
    pub fn name(&self) -> &str {
        &self.info().name
    }
    pub fn parameters(&self) -> &[(String, SignatureType)] {
        &self.info().parameter_types
    }
    pub fn result_type(&self) -> &SignatureType {
        &self.info().result_type
    }
    pub fn elements(&self) -> &[String] {
        &self.info().element_parameters
    }
    pub fn numerical_subjects(&self) -> Vec<String> {
        seismic_compiler::numerics::subject_names(self.info())
    }
    fn validate_policy(&self, policy: &PrecisionPolicy) -> Result<(), Error> {
        seismic_compiler::numerics::validate_policy_subjects(self.info(), policy)
            .map_err(|e| Error::new("ValueError", e))
    }
    pub fn dimensions(&self) -> &[String] {
        &self.info().dimensions
    }
    fn bindings(
        &self,
        elements: &BTreeMap<String, Element>,
    ) -> Result<seismic_lang::entry::ElementBindings, Error> {
        if elements.keys().collect::<BTreeSet<_>>()
            != self.elements().iter().collect::<BTreeSet<_>>()
        {
            return Err(Error::new(
                "ValueError",
                format!(
                    "{} requires exactly these element bindings: {:?}",
                    self.name(),
                    self.elements()
                ),
            ));
        }
        Ok(elements
            .iter()
            .fold(seismic_lang::entry::ElementBindings::new(), |b, (n, e)| {
                b.bind(n, e.id())
            }))
    }
    /// The entry prepared on `charges`' device, its calls allocating
    /// through `charges`.
    pub fn prepare(
        &self,
        charges: &super::Charges,
        elements: BTreeMap<String, Element>,
        options: PreparationOptions,
    ) -> Result<Kernel, Error> {
        self.validate_policy(&options.precision)?;
        let start = Instant::now();
        let prepared = runtime::prepare(
            &self.module.checked,
            self.info().id,
            self.bindings(&elements)?,
            charges.inner(),
            options,
        )?;
        Ok(Kernel {
            function: self.clone(),
            charges: charges.share(),
            elements,
            inner: KernelKind::Ordinary(Arc::new(prepared)),
            preparation_seconds: start.elapsed().as_secs_f64(),
        })
    }
    /// Form the entry's native implementation for `device`'s backend under
    /// `specialization`. Only Metal and CUDA are reachable here: CPU native
    /// implementations are Rust compiled into a generated-binding build.
    pub fn prepare_native(
        &self,
        charges: &super::Charges,
        elements: BTreeMap<String, Element>,
        specialization: crate::NativeSpecialization,
    ) -> Result<Kernel, Error> {
        let start = Instant::now();
        let bindings = self.bindings(&elements)?;
        let prepared = runtime::prepare_native(
            &self.module.checked,
            self.info().id,
            bindings,
            charges.device().inner(),
            specialization,
            None,
        )?;
        Ok(Kernel {
            function: self.clone(),
            charges: charges.share(),
            elements,
            inner: KernelKind::Native(Arc::new(prepared)),
            preparation_seconds: start.elapsed().as_secs_f64(),
        })
    }
}

static STORAGES: OnceLock<Mutex<BTreeMap<u64, Weak<Storage>>>> = OnceLock::new();

fn storages() -> &'static Mutex<BTreeMap<u64, Weak<Storage>>> {
    STORAGES.get_or_init(Default::default)
}

struct Storage {
    allocation_id: u64,
    gate: Mutex<()>,
    handles: AtomicUsize,
}
impl Drop for Storage {
    fn drop(&mut self) {
        let mut map = lock(storages());
        if map
            .get(&self.allocation_id)
            .is_some_and(|entry| entry.strong_count() == 0)
        {
            map.remove(&self.allocation_id);
        }
    }
}
struct TensorHandle {
    value: Mutex<Option<super::Tensor>>,
    storage: Arc<Storage>,
    view: bool,
}
impl Drop for TensorHandle {
    fn drop(&mut self) {
        if self
            .value
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
        {
            self.storage.handles.fetch_sub(1, Ordering::SeqCst);
        }
    }
}
/// A dynamic host owner. Cloning aliases validity; views are separately tracked borrows.
#[derive(Clone)]
/// A tensor of a host binding: what it allocates is charged to the charges
/// it is made through.
pub struct Tensor(Arc<TensorHandle>);
impl Tensor {
    fn new(value: super::Tensor, view: bool) -> Self {
        let id = value.descriptor().allocation;
        let storage = {
            let mut map = lock(storages());
            let storage = map.get(&id).and_then(Weak::upgrade).unwrap_or_else(|| {
                let s = Arc::new(Storage {
                    allocation_id: id,
                    gate: Mutex::new(()),
                    handles: AtomicUsize::new(0),
                });
                map.insert(id, Arc::downgrade(&s));
                s
            });
            storage.handles.fetch_add(1, Ordering::SeqCst);
            storage
        };
        Self(Arc::new(TensorHandle {
            value: Mutex::new(Some(value)),
            storage,
            view,
        }))
    }
    fn raw(&self) -> Result<super::Tensor, Error> {
        lock(&self.0.value)
            .clone()
            .ok_or_else(|| Error::new("TensorError", "tensor was moved"))
    }
    pub fn from_host(
        charges: &super::Charges,
        element: Element,
        shape: &[u64],
        bytes: &[u8],
    ) -> Result<Self, Error> {
        Ok(Self::new(
            super::Tensor::from_host(charges, element, shape, bytes)?,
            false,
        ))
    }
    pub fn zeros(charges: &super::Charges, element: Element, shape: &[u64]) -> Result<Self, Error> {
        Ok(Self::new(
            super::Tensor::zeros(charges, element, shape)?,
            false,
        ))
    }
    pub fn shape(&self) -> Result<Vec<u64>, Error> {
        Ok(self.raw()?.extents().to_vec())
    }
    pub fn element(&self) -> Result<Element, Error> {
        Ok(self.raw()?.element())
    }
    pub fn device(&self) -> Result<Device, Error> {
        Ok(self.raw()?.device())
    }
    pub fn byte_len(&self) -> Result<u64, Error> {
        Ok(self.raw()?.byte_len())
    }
    pub fn read(&self) -> Result<Vec<u8>, Error> {
        let _g = lock(&self.0.storage.gate);
        Ok(self.raw()?.read_to_host()?)
    }
    pub fn write(&self, bytes: &[u8]) -> Result<(), Error> {
        let _g = lock(&self.0.storage.gate);
        Ok(self.raw()?.write_from_host(bytes)?)
    }
    /// A copy of it, allocated through `charges`.
    pub fn copy(&self, charges: &super::Charges) -> Result<Self, Error> {
        let _g = lock(&self.0.storage.gate);
        let t = self.raw()?;
        Self::from_host(charges, t.element(), t.extents(), &t.read_to_host()?)
    }
    pub fn copy_from(&self, source: &Self) -> Result<(), Error> {
        let mut storages = vec![self.0.storage.clone(), source.0.storage.clone()];
        storages.sort_by_key(|s| Arc::as_ptr(s) as usize);
        storages.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let _guards: Vec<_> = storages.iter().map(|s| lock(&s.gate)).collect();
        let mut target = self.raw()?;
        let source = source.raw()?;
        if target.extents() != source.extents()
            || target.element() != source.element()
            || !source.belongs_to(&target.device())
        {
            return Err(Error::new(
                "TensorError",
                "copy_from requires matching shape, element, and device",
            ));
        }
        target.write_from_host(&source.read_to_host()?)?;
        Ok(())
    }
    pub fn slice(&self, start: u64, end: u64) -> Result<Self, Error> {
        let _g = lock(&self.0.storage.gate);
        Ok(Self::new(self.raw()?.slice_leading(start, end)?, true))
    }
    pub fn reshape(&self, shape: &[u64]) -> Result<Self, Error> {
        let _g = lock(&self.0.storage.gate);
        let t = self.raw()?;
        let volume = |s: &[u64]| {
            if s.contains(&0) {
                Some(0u128)
            } else {
                s.iter().try_fold(1u128, |n, v| n.checked_mul(*v as u128))
            }
        };
        if volume(t.extents()).is_none() || volume(t.extents()) != volume(shape) {
            return Err(Error::new(
                "TensorError",
                "reshape must preserve logical element count",
            ));
        }
        Ok(Self::new(t.reshape(shape)?, true))
    }
    fn commit(&self) {
        if lock(&self.0.value).take().is_some() {
            self.0.storage.handles.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Scalars retain their declared storage bits at the dynamic boundary.
#[derive(Clone, Debug)]
pub enum Scalar {
    F32(u32),
    F16(u16),
    BF16(u16),
    I32(i32),
    U32(u32),
    U64(u64),
    Bool(bool),
    Index(seismic_lang::expr::BigUint),
    Range(seismic_lang::expr::BigUint, seismic_lang::expr::BigUint),
}
impl Scalar {
    fn encoded(&self) -> runtime::EncodedScalar {
        use runtime::EncodedScalar as E;
        match self.clone() {
            Self::F32(v) => E::from_f32(f32::from_bits(v)),
            Self::F16(v) => E::F16(v),
            Self::BF16(v) => E::BF16(v),
            Self::I32(v) => E::I32(v),
            Self::U32(v) => E::U32(v),
            Self::U64(v) => E::U64(v),
            Self::Bool(v) => E::Bool(v),
            Self::Index(v) => E::Index(v),
            Self::Range(start, end) => E::Range { start, end },
        }
    }
    fn matches(&self, ty: &SignatureType) -> bool {
        matches!(
            (self, ty),
            (Self::F32(_), SignatureType::Scalar(DType::F32))
                | (Self::F16(_), SignatureType::Scalar(DType::F16))
                | (Self::BF16(_), SignatureType::Scalar(DType::BF16))
                | (Self::I32(_), SignatureType::Scalar(DType::I32))
                | (Self::U32(_), SignatureType::Scalar(DType::U32))
                | (Self::U64(_), SignatureType::Scalar(DType::U64))
                | (Self::Bool(_), SignatureType::Scalar(DType::Bool))
                | (Self::Index(_), SignatureType::Index)
                | (Self::Range(..), SignatureType::Range)
        )
    }
}
#[derive(Clone)]
pub enum Value {
    Unit,
    Tuple(Vec<Value>),
    Tensor(Tensor),
    Move(Tensor),
    Scalar(Scalar),
}
enum KernelKind {
    Ordinary(Arc<runtime::PreparedAny>),
    Native(Arc<runtime::NativePreparedAny>),
}
pub struct Kernel {
    function: Function,
    /// What its calls allocate through.
    charges: super::Charges,
    elements: BTreeMap<String, Element>,
    inner: KernelKind,
    pub preparation_seconds: f64,
}
enum CallOutcome {
    Terminated(
        seismic_lang::failure::SourceTermination<Value, seismic_compiler::errors::CheckFailure>,
    ),
    ResourceLimit(Error),
}

fn is_allocation_capacity(e: &CallError) -> bool {
    matches!(
        e,
        CallError::Invocation(InvocationError::AllocationCapacity { .. })
            | CallError::Execution(ExecutionError::AllocationCapacity { .. })
    )
}

impl Kernel {
    pub fn function(&self) -> &Function {
        &self.function
    }
    pub fn device(&self) -> Device {
        self.charges.device()
    }
    pub fn is_native(&self) -> bool {
        matches!(self.inner, KernelKind::Native(_))
    }
    pub fn call(&self, args: &[Value]) -> Result<Value, Error> {
        match self.call_outcome_limited(args, None)? {
            CallOutcome::Terminated(seismic_lang::failure::SourceTermination::Returned(value)) => {
                Ok(value)
            }
            CallOutcome::Terminated(seismic_lang::failure::SourceTermination::Failed(failure)) => {
                Err(Error::from(ExecutionError::DataCheckFailed(failure)))
            }
            CallOutcome::ResourceLimit(err) => Err(err),
        }
    }
    fn call_outcome_limited(
        &self,
        args: &[Value],
        allocation_limit: Option<u64>,
    ) -> Result<CallOutcome, Error> {
        if args.len() != self.function.parameters().len() {
            return Err(Error::new("TypeError", "wrong argument count"));
        }
        let mut tensors = Vec::new();
        for v in args {
            collect_tensors(v, &mut tensors);
        }
        let mut storages: Vec<_> = tensors.iter().map(|t| t.0.storage.clone()).collect();
        storages.sort_by_key(|s| Arc::as_ptr(s) as usize);
        storages.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let _guards: Vec<_> = storages.iter().map(|s| lock(&s.gate)).collect();
        let mut encoded = runtime::EncodedArgs::new();
        let mut moves = Vec::new();
        for ((name, ty), value) in self.function.parameters().iter().zip(args) {
            encode(ty, value, &self.elements, &mut encoded, &mut moves).map_err(|mut e| {
                e.message = format!("{} argument {name}: {}", self.function.name(), e.message);
                e
            })?;
        }
        let result = match &self.inner {
            KernelKind::Ordinary(kernel) => {
                let mut draft = runtime::workflow(self.charges.inner())
                    .map_err(|e| Error::from(CallError::Workflow(e)))?;
                let mut workflow_args = runtime::EncodedWorkflowArgs::new();
                for v in args {
                    workflow_encode(v, &mut workflow_args)?;
                }
                runtime::enqueue(&mut draft, kernel, workflow_args)
                    .map_err(|e| Error::from(CallError::Workflow(e)))?;
                let bound = runtime::bind_workflow(draft)?
                    .with_allocation_limit(allocation_limit.unwrap_or(u64::MAX));
                let admitted = match runtime::admit_workflow(bound) {
                    Ok(admitted) => admitted,
                    Err(e) if is_allocation_capacity(&e) => {
                        return Ok(CallOutcome::ResourceLimit(Error::from(e)));
                    }
                    Err(e) => return Err(Error::from(e)),
                };
                for t in &moves {
                    t.commit();
                }
                let completion = match runtime::submit_workflow(admitted) {
                    Ok(completion) => completion,
                    Err(e) if is_allocation_capacity(&e) => {
                        return Ok(CallOutcome::ResourceLimit(Error::from(e)));
                    }
                    Err(e) => return Err(Error::from(e)),
                };
                match completion.outcome() {
                    Ok(seismic_lang::failure::SourceTermination::Returned(mut groups)) => {
                        assert_eq!(
                            groups.len(),
                            1,
                            "one dynamic invocation has one result group"
                        );
                        seismic_lang::failure::SourceTermination::Returned(groups.pop().unwrap())
                    }
                    Ok(seismic_lang::failure::SourceTermination::Failed(failure)) => {
                        seismic_lang::failure::SourceTermination::Failed(failure)
                    }
                    Err(e) if is_allocation_capacity(&e) => {
                        return Ok(CallOutcome::ResourceLimit(Error::from(e)));
                    }
                    Err(e) => return Err(Error::from(e)),
                }
            }
            KernelKind::Native(kernel) => {
                let output = match runtime::call_native_with_commit(
                    kernel,
                    self.charges.inner(),
                    encoded,
                    || {
                        for t in &moves {
                            t.commit();
                        }
                    },
                ) {
                    Ok(output) => output,
                    Err(e) if is_allocation_capacity(&e) => {
                        return Ok(CallOutcome::ResourceLimit(Error::from(e)));
                    }
                    Err(e) => return Err(Error::from(e)),
                };
                seismic_lang::failure::SourceTermination::Returned(output.into_values())
            }
        };
        match result {
            seismic_lang::failure::SourceTermination::Returned(values) => {
                decode(self.function.result_type(), &mut values.into_iter()).map(|value| {
                    CallOutcome::Terminated(seismic_lang::failure::SourceTermination::Returned(
                        value,
                    ))
                })
            }
            seismic_lang::failure::SourceTermination::Failed(failure) => Ok(
                CallOutcome::Terminated(seismic_lang::failure::SourceTermination::Failed(failure)),
            ),
        }
    }
}
fn collect_tensors<'a>(v: &'a Value, out: &mut Vec<&'a Tensor>) {
    match v {
        Value::Tensor(t) | Value::Move(t) => out.push(t),
        Value::Tuple(v) => {
            for x in v {
                collect_tensors(x, out)
            }
        }
        _ => {}
    }
}
fn encode(
    ty: &SignatureType,
    v: &Value,
    elements: &BTreeMap<String, Element>,
    out: &mut runtime::EncodedArgs,
    moves: &mut Vec<Tensor>,
) -> Result<(), Error> {
    let bad = || Error::new("TypeError", "value does not match the checked signature");
    match (ty, v) {
        (SignatureType::Unit, Value::Unit) => {}
        (SignatureType::Tuple(types), Value::Tuple(values)) if types.len() == values.len() => {
            for (t, v) in types.iter().zip(values) {
                encode(t, v, elements, out, moves)?;
            }
        }
        (
            SignatureType::Tensor {
                access,
                rank,
                element,
            },
            Value::Tensor(t) | Value::Move(t),
        ) => {
            let owned = *access == TensorAccess::Owned;
            if owned != matches!(v, Value::Move(_)) {
                return Err(Error::new(
                    "TypeError",
                    if owned {
                        "owned tensor requires move(tensor)"
                    } else {
                        "borrowed tensor cannot accept a move"
                    },
                ));
            }
            if owned
                && (t.0.view
                    || t.0.storage.handles.load(Ordering::SeqCst) != 1
                    || moves.iter().any(|m| Arc::ptr_eq(&m.0, &t.0)))
            {
                return Err(Error::new(
                    "InvocationError",
                    "owned tensor has live views or duplicate moves",
                ));
            }
            let raw = t.raw()?;
            let expected = match element {
                ElementSummary::Fixed(n) => Element::named(n).expect("checked element"),
                ElementSummary::Parameter(n) => *elements.get(n).expect("checked element binding"),
            };
            if raw.element() != expected || raw.extents().len() != *rank as usize {
                return Err(Error::new(
                    "InvocationError",
                    format!(
                        "expected rank {rank} {}, got {:?} {}",
                        expected.name(),
                        raw.extents(),
                        raw.element().name()
                    ),
                ));
            }
            out.push_tensor(raw.inner().clone());
            if owned {
                moves.push(t.clone());
            }
        }
        (ty, Value::Scalar(s)) if s.matches(ty) => out.push_scalar(s.encoded()),
        _ => return Err(bad()),
    }
    Ok(())
}
fn workflow_encode(v: &Value, out: &mut runtime::EncodedWorkflowArgs) -> Result<(), Error> {
    match v {
        Value::Unit => {}
        Value::Tuple(v) => {
            for x in v {
                workflow_encode(x, out)?
            }
        }
        Value::Tensor(t) | Value::Move(t) => out.push_external_tensor(t.raw()?.inner().clone()),
        Value::Scalar(s) => out.push_scalar(s.encoded()),
    }
    Ok(())
}
fn decode(
    ty: &SignatureType,
    values: &mut impl Iterator<Item = runtime::DecodedValue>,
) -> Result<Value, Error> {
    use runtime::DecodedValue as V;
    Ok(match ty {
        SignatureType::Unit => Value::Unit,
        SignatureType::Tuple(t) => Value::Tuple(
            t.iter()
                .map(|t| decode(t, values))
                .collect::<Result<_, _>>()?,
        ),
        SignatureType::Tensor { .. } => match values.next() {
            Some(V::Tensor(inner)) => Value::Tensor(Tensor::new(super::Tensor { inner }, false)),
            _ => return Err(Error::new("InternalError", "result contract mismatch")),
        },
        _ => {
            let Some(V::Scalar(v)) = values.next() else {
                return Err(Error::new(
                    "InternalError",
                    "scalar result contract mismatch",
                ));
            };
            Value::Scalar(match v.into_value() {
                ArgumentValue::F32(v) => Scalar::F32(v.to_bits()),
                ArgumentValue::F16(v) => Scalar::F16(v),
                ArgumentValue::BF16(v) => Scalar::BF16(v),
                ArgumentValue::I32(v) => Scalar::I32(v),
                ArgumentValue::U32(v) => Scalar::U32(v),
                ArgumentValue::U64(v) => Scalar::U64(v),
                ArgumentValue::Bool(v) => Scalar::Bool(v),
                ArgumentValue::Index(v) => Scalar::Index(v),
                ArgumentValue::Range { start, end } => Scalar::Range(start, end),
                ArgumentValue::Tensor(_) => {
                    return Err(Error::new(
                        "InternalError",
                        "scalar result contract mismatch",
                    ));
                }
            })
        }
    })
}

// Own the charges (and their device) before borrowing the preparation
// context. The generated cell drops the campaign first and never invents a
// static device lifetime. The campaign's trials, and the kernels it
// publishes, allocate through the owned charges.
type Campaign<'a> = runtime::FeedbackPreparation<'a>;
self_cell::self_cell! {
    struct OwnedCampaign {
        owner: super::Charges,
        #[not_covariant]
        dependent: Campaign,
    }
}
pub struct FeedbackSession {
    campaign: OwnedCampaign,
    function: Function,
    elements: BTreeMap<String, Element>,
}
impl Function {
    pub fn start_feedback(
        &self,
        charges: &super::Charges,
        elements: BTreeMap<String, Element>,
        options: PreparationOptions,
    ) -> Result<(FeedbackSession, Kernel), Error> {
        self.validate_policy(&options.precision)?;
        let EvaluationMethod::Feedback(feedback) = options.evaluation else {
            return Err(Error::new(
                "ValueError",
                "start_feedback requires Feedback options",
            ));
        };
        let bindings = self.bindings(&elements)?;
        let start = Instant::now();
        let mut initial = None;
        let campaign = OwnedCampaign::try_new(charges.share(), |owner| {
            let (session, kernel) = runtime::start_feedback(
                &self.module.checked,
                self.info().id,
                bindings,
                owner.inner(),
                options.precision,
                feedback,
            )?;
            initial = Some(kernel);
            Ok::<_, runtime::PrepareError>(session)
        })?;
        let kernel = Kernel {
            function: self.clone(),
            charges: charges.share(),
            elements: elements.clone(),
            inner: KernelKind::Ordinary(Arc::new(
                initial.expect("successful feedback publishes kernel"),
            )),
            preparation_seconds: start.elapsed().as_secs_f64(),
        };
        Ok((
            FeedbackSession {
                campaign,
                function: self.clone(),
                elements,
            },
            kernel,
        ))
    }
}
impl FeedbackSession {
    pub fn continue_for(&mut self, duration: std::time::Duration) -> Result<Kernel, Error> {
        let start = Instant::now();
        let inner = self
            .campaign
            .with_dependent_mut(|_, s| s.continue_for(duration))?;
        Ok(Kernel {
            function: self.function.clone(),
            charges: self.campaign.borrow_owner().share(),
            elements: self.elements.clone(),
            inner: KernelKind::Ordinary(Arc::new(inner)),
            preparation_seconds: start.elapsed().as_secs_f64(),
        })
    }
    pub fn report(&self) -> FeedbackReport {
        self.campaign.with_dependent(|_, s| s.report().clone())
    }
}

mod workflow;
pub use workflow::{Pending, Workflow, WorkflowValue};

#[derive(Clone, Copy)]
pub enum ScopeKind {
    Dimension,
    Scalar,
    RangeStart,
    RangeEnd,
}
#[derive(Clone)]
pub struct Scope {
    entry: seismic_lang::ids::StableEntryId,
    inner: InvocationScope,
}
impl Function {
    pub fn scope_parameters(&self) -> Vec<(String, SignatureType)> {
        use seismic_lang::checked::ParameterSummaryKind as P;
        self.info()
            .parameters
            .iter()
            .filter_map(|p| {
                let ty = match p.kind {
                    P::Scalar(d) => SignatureType::Scalar(d),
                    P::Index => SignatureType::Index,
                    P::Range => SignatureType::Range,
                    _ => return None,
                };
                Some((parameter_path(p), ty))
            })
            .collect()
    }
    pub fn scope(
        &self,
        constraints: Vec<(ScopeKind, String, Scalar, Scalar)>,
    ) -> Result<Scope, Error> {
        use seismic_compiler::feedback::InvocationParameter as P;
        use seismic_lang::checked::ParameterSummaryKind as K;
        use seismic_lang::expr::SymbolValue as S;
        let mut scope = InvocationScope::for_entry(self.info().stable);
        for (kind, name, lower, upper) in constraints {
            let (parameter, ty) = match kind {
                ScopeKind::Dimension => {
                    let i = self
                        .dimensions()
                        .iter()
                        .position(|n| n == &name)
                        .ok_or_else(|| {
                            Error::new("ValueError", format!("unknown dimension {name}"))
                        })?;
                    (P::Dimension(i as u32), SignatureType::Index)
                }
                _ => {
                    let (i, p) = self
                        .info()
                        .parameters
                        .iter()
                        .enumerate()
                        .find(|(_, p)| parameter_path(p) == name)
                        .ok_or_else(|| {
                            Error::new("ValueError", format!("unknown parameter {name}"))
                        })?;
                    match (kind, &p.kind) {
                        (ScopeKind::Scalar, K::Scalar(d)) => {
                            (P::Scalar(i as u32), SignatureType::Scalar(*d))
                        }
                        (ScopeKind::Scalar, K::Index) => {
                            (P::Scalar(i as u32), SignatureType::Index)
                        }
                        (ScopeKind::RangeStart, K::Range) => {
                            (P::RangeStart(i as u32), SignatureType::Index)
                        }
                        (ScopeKind::RangeEnd, K::Range) => {
                            (P::RangeEnd(i as u32), SignatureType::Index)
                        }
                        _ => return Err(Error::new("TypeError", "scope parameter kind mismatch")),
                    }
                }
            };
            if !lower.matches(&ty) || !upper.matches(&ty) {
                return Err(Error::new("TypeError", "scope endpoint type mismatch"));
            }
            if !lower.ordered_interval(&upper) {
                return Err(Error::new(
                    "ValueError",
                    "scope interval is inverted or nonfinite",
                ));
            }
            let symbol = |v: &Scalar| -> Result<S, Error> {
                if let Scalar::Index(n) = v {
                    return Ok(S::Nat(n.clone()));
                }
                Ok(v.symbol())
            };
            scope.constrain(parameter, symbol(&lower)?, symbol(&upper)?);
        }
        Ok(Scope {
            entry: self.info().stable,
            inner: scope,
        })
    }
    pub fn apply_scope(
        &self,
        options: &mut PreparationOptions,
        scope: &Scope,
    ) -> Result<(), Error> {
        if scope.entry != self.info().stable {
            return Err(Error::new("ValueError", "scope belongs to another entry"));
        }
        let EvaluationMethod::Feedback(feedback) = &mut options.evaluation else {
            return Err(Error::new("ValueError", "scope requires feedback"));
        };
        feedback.optimize_for = Some(scope.inner.clone());
        Ok(())
    }
}
fn parameter_path(p: &seismic_lang::checked::ParameterSummary) -> String {
    let mut name = p.name.clone();
    for i in &p.path {
        name.push_str(&format!("[{i}]"));
    }
    name
}
impl Scalar {
    fn ordered_interval(&self, upper: &Self) -> bool {
        match (self, upper) {
            (Self::Index(a), Self::Index(b)) => a <= b,
            (Self::I32(a), Self::I32(b)) => a <= b,
            (Self::U32(a), Self::U32(b)) => a <= b,
            (Self::U64(a), Self::U64(b)) => a <= b,
            (Self::Bool(a), Self::Bool(b)) => a <= b,
            (Self::F32(a), Self::F32(b)) => {
                let (a, b) = (f32::from_bits(*a), f32::from_bits(*b));
                a.is_finite() && b.is_finite() && a <= b
            }
            (Self::F16(a), Self::F16(b)) => {
                let (a, b) = (
                    seismic_lang::registry::f16_to_f32(*a),
                    seismic_lang::registry::f16_to_f32(*b),
                );
                a.is_finite() && b.is_finite() && a <= b
            }
            (Self::BF16(a), Self::BF16(b)) => {
                let (a, b) = (
                    f32::from_bits((*a as u32) << 16),
                    f32::from_bits((*b as u32) << 16),
                );
                a.is_finite() && b.is_finite() && a <= b
            }
            _ => false,
        }
    }
    fn symbol(&self) -> seismic_lang::expr::SymbolValue {
        use seismic_lang::expr::SymbolValue as S;
        match self.clone() {
            Self::F32(v) => S::F32(f32::from_bits(v)),
            Self::F16(v) => S::F16(v),
            Self::BF16(v) => S::BF16(v),
            Self::I32(v) => S::I32(v),
            Self::U32(v) => S::U32(v),
            Self::U64(v) => S::U64(v),
            Self::Bool(v) => S::Bool(v),
            Self::Index(v) => S::Nat(v),
            Self::Range(..) => unreachable!(),
        }
    }
}

mod observation;
pub use observation::CheckReport;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_rejects_inverted_u64_interval_above_f64_precision() {
        let module = Module::source(
            "fn id(x: u64) -> u64:\n    return x\n",
            "scope_u64.seismic",
            false,
            None,
        )
        .unwrap();
        let function = module.function("id").unwrap();
        let lo = (1u64 << 53) + 2;
        let hi = (1u64 << 53) + 1;
        let Err(err) = function.scope(vec![(
            ScopeKind::Scalar,
            "x".to_owned(),
            Scalar::U64(lo),
            Scalar::U64(hi),
        )]) else {
            panic!("inverted u64 interval above 2^53 must be rejected");
        };
        assert_eq!(err.kind, "ValueError");
        assert!(
            function
                .scope(vec![(
                    ScopeKind::Scalar,
                    "x".to_owned(),
                    Scalar::U64(hi),
                    Scalar::U64(lo),
                )])
                .is_ok(),
            "ordered u64 interval above 2^53 must be accepted"
        );
    }

    #[test]
    fn storage_drop_removes_entry_without_clobbering_reused_id() {
        let id = u64::MAX - 17;
        let first = {
            let mut map = lock(storages());
            let s = Arc::new(Storage {
                allocation_id: id,
                gate: Mutex::new(()),
                handles: AtomicUsize::new(1),
            });
            map.insert(id, Arc::downgrade(&s));
            s
        };
        assert!(lock(storages()).contains_key(&id));
        drop(first);
        assert!(
            !lock(storages()).contains_key(&id),
            "dropping the last Storage handle must remove its STORAGES entry"
        );

        // Simulate allocation_id reuse while an older Storage is about to drop.
        let older = Arc::new(Storage {
            allocation_id: id,
            gate: Mutex::new(()),
            handles: AtomicUsize::new(0),
        });
        let newer = {
            let mut map = lock(storages());
            let s = Arc::new(Storage {
                allocation_id: id,
                gate: Mutex::new(()),
                handles: AtomicUsize::new(1),
            });
            map.insert(id, Arc::downgrade(&s));
            s
        };
        drop(older);
        assert!(
            lock(storages())
                .get(&id)
                .and_then(Weak::upgrade)
                .is_some_and(|s| Arc::ptr_eq(&s, &newer)),
            "dropping an older Storage must not remove a live reused allocation_id entry"
        );
        drop(newer);
        assert!(!lock(storages()).contains_key(&id));
    }
}
