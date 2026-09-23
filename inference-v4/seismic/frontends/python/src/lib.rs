//! Python adaptation only. The public seismic crate owns every call contract.
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use seismic::dynamic as d;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

fn error(e: d::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err((e.kind.to_owned(), e.message))
}
fn element(name: &str) -> PyResult<seismic::Element> {
    seismic::Element::named(name)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown element {name}")))
}
fn catalog() -> PyResult<&'static Mutex<seismic::DeviceCatalog>> {
    static CATALOG: OnceLock<Mutex<seismic::DeviceCatalog>> = OnceLock::new();
    if let Some(c) = CATALOG.get() {
        return Ok(c);
    }
    let c =
        seismic::DeviceCatalog::discover().map_err(|e| error(d::Error::new("TargetError", e)))?;
    let _ = CATALOG.set(Mutex::new(c));
    Ok(CATALOG.get().expect("catalog initialized"))
}

#[pyclass(name = "Device", frozen)]
#[derive(Clone)]
struct Device {
    inner: seismic::Device,
}
#[pymethods]
impl Device {
    #[getter]
    fn name(&self) -> String {
        self.inner.info().name.clone()
    }
    #[getter]
    fn backend(&self) -> String {
        self.inner.backend().as_str().into()
    }
    #[getter]
    fn capabilities(&self) -> Vec<String> {
        self.inner.capabilities().to_vec()
    }
    fn memory_usage(&self) -> (u64, Option<u64>, u64) {
        let m = self.inner.memory_usage();
        (m.charged, m.limit, m.pool_charged)
    }
    fn set_memory_limit(&self, bytes: Option<u64>) -> PyResult<()> {
        self.inner
            .set_memory_limit(bytes)
            .map_err(|e| error(d::Error::new("TensorError", e)))
    }
    fn __repr__(&self) -> String {
        format!("Device({}, {:?})", self.backend(), self.name())
    }
}
#[pyfunction]
fn devices() -> PyResult<Vec<(String, String, String)>> {
    let c = catalog()?.lock().unwrap();
    Ok(c.topology()
        .devices()
        .iter()
        .map(|i| {
            (
                i.selector.to_string(),
                i.name.clone(),
                i.backend.as_str().to_owned(),
            )
        })
        .collect())
}
/// Opens an exact device selector (`host-cpu`, `metal:…`, `cuda:…`), or,
/// for low-level use, the first device of a backend name (`cpu`, `metal`).
#[pyfunction]
fn device(py: Python<'_>, selector: &str) -> PyResult<Device> {
    let selector = selector.to_owned();
    py.detach(move || {
        let c = catalog()?.lock().unwrap();
        let id = match selector.parse::<seismic::DeviceSelector>() {
            Ok(exact) => c
                .resolve(exact)
                .map_err(|e| error(d::Error::new("TargetError", e)))?,
            Err(_) => c
                .topology()
                .devices()
                .iter()
                .find(|i| i.backend.as_str() == selector)
                .map(|i| i.id)
                .ok_or_else(|| {
                    error(d::Error::new(
                        "TargetError",
                        format!("no device matches `{selector}`"),
                    ))
                })?,
        };
        Ok(Device {
            inner: c
                .open(id)
                .map_err(|e| error(d::Error::new("TargetError", e)))?,
        })
    })
}

#[pyclass(name = "Tensor", frozen)]
#[derive(Clone)]
struct Tensor {
    inner: d::Tensor,
}
#[pymethods]
impl Tensor {
    #[staticmethod]
    fn from_bytes(
        py: Python<'_>,
        device: &Device,
        name: &str,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    ) -> PyResult<Self> {
        let device = device.inner.clone();
        let element = element(name)?;
        py.detach(move || {
            d::Tensor::from_host(&device, element, &shape, &bytes)
                .map(|inner| Self { inner })
                .map_err(error)
        })
    }
    #[staticmethod]
    fn zeros(py: Python<'_>, device: &Device, name: &str, shape: Vec<u64>) -> PyResult<Self> {
        let device = device.inner.clone();
        let element = element(name)?;
        py.detach(move || {
            d::Tensor::zeros(&device, element, &shape)
                .map(|inner| Self { inner })
                .map_err(error)
        })
    }
    #[getter]
    fn shape(&self) -> PyResult<Vec<u64>> {
        self.inner.shape().map_err(error)
    }
    #[getter]
    fn element(&self) -> PyResult<String> {
        Ok(self.inner.element().map_err(error)?.name().into())
    }
    #[getter]
    fn device(&self) -> PyResult<Device> {
        Ok(Device {
            inner: self.inner.device().map_err(error)?,
        })
    }
    #[getter]
    fn nbytes(&self) -> PyResult<u64> {
        self.inner.byte_len().map_err(error)
    }
    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.inner.read()).map_err(error)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn write(&self, py: Python<'_>, bytes: Vec<u8>) -> PyResult<()> {
        py.detach(|| self.inner.write(&bytes)).map_err(error)
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| self.inner.copy())
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn copy_from(&self, py: Python<'_>, source: &Tensor) -> PyResult<()> {
        py.detach(|| self.inner.copy_from(&source.inner))
            .map_err(error)
    }
    fn same_device(&self, device: &Device) -> PyResult<bool> {
        Ok(self
            .inner
            .device()
            .map_err(error)?
            .same_device(&device.inner))
    }
    fn reshape(&self, py: Python<'_>, shape: Vec<u64>) -> PyResult<Self> {
        py.detach(|| self.inner.reshape(&shape))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn slice(&self, py: Python<'_>, start: u64, end: u64) -> PyResult<Self> {
        py.detach(|| self.inner.slice(start, end))
            .map(|inner| Self { inner })
            .map_err(error)
    }
}

#[pyclass(name = "Move", frozen)]
struct Move {
    tensor: d::Tensor,
}
#[pyfunction]
fn move_tensor(tensor: &Tensor) -> Move {
    Move {
        tensor: tensor.inner.clone(),
    }
}

#[pyclass(name = "Scalar", frozen)]
struct Scalar {
    inner: d::Scalar,
}
#[pymethods]
impl Scalar {
    #[new]
    #[pyo3(signature=(kind, word, end=0))]
    fn new(kind: &str, word: u64, end: u64) -> PyResult<Self> {
        let inner = match kind {
            "f32" => d::Scalar::F32(
                u32::try_from(word)
                    .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("f32 word"))?,
            ),
            "f16" => d::Scalar::F16(
                u16::try_from(word)
                    .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("f16 word"))?,
            ),
            "bf16" => d::Scalar::BF16(
                u16::try_from(word)
                    .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("bf16 word"))?,
            ),
            "i32" => d::Scalar::I32(
                u32::try_from(word)
                    .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("i32 word"))?
                    as i32,
            ),
            "u32" => d::Scalar::U32(
                u32::try_from(word)
                    .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("u32 word"))?,
            ),
            "bool" if word <= 1 => d::Scalar::Bool(word != 0),
            "index" => d::Scalar::Index(word),
            "range" => d::Scalar::Range(word, end),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "invalid scalar kind or word",
                ))
            }
        };
        Ok(Self { inner })
    }
}
fn value(obj: &Bound<'_, PyAny>) -> PyResult<d::Value> {
    if obj.is_none() {
        return Ok(d::Value::Unit);
    }
    if let Ok(t) = obj.extract::<PyRef<Tensor>>() {
        return Ok(d::Value::Tensor(t.inner.clone()));
    }
    if let Ok(t) = obj.extract::<PyRef<Move>>() {
        return Ok(d::Value::Move(t.tensor.clone()));
    }
    if let Ok(s) = obj.extract::<PyRef<Scalar>>() {
        return Ok(d::Value::Scalar(s.inner.clone()));
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        return Ok(d::Value::Tuple(
            t.iter().map(|x| value(&x)).collect::<PyResult<_>>()?,
        ));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a Seismic tensor, scalar, move, tuple, or None",
    ))
}
fn result(py: Python<'_>, v: d::Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        d::Value::Unit => py.None(),
        d::Value::Tuple(v) => PyTuple::new(
            py,
            v.into_iter()
                .map(|v| result(py, v))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any()
        .unbind(),
        d::Value::Tensor(inner) => Py::new(py, Tensor { inner })?.into_any(),
        d::Value::Scalar(s) => {
            let (k, a, b) = match s {
                d::Scalar::F32(v) => ("f32", v as u64, 0),
                d::Scalar::F16(v) => ("f16", v as u64, 0),
                d::Scalar::BF16(v) => ("bf16", v as u64, 0),
                d::Scalar::I32(v) => ("i32", v as u32 as u64, 0),
                d::Scalar::U32(v) => ("u32", v as u64, 0),
                d::Scalar::Bool(v) => ("bool", v as u64, 0),
                d::Scalar::Index(v) => ("index", v, 0),
                d::Scalar::Range(a, b) => ("range", a, b),
            };
            (k, a, b).into_pyobject(py)?.into_any().unbind()
        }
        d::Value::Move(_) => unreachable!("move is an input only"),
    })
}
fn signature_type(t: &d::SignatureType) -> serde_json::Value {
    use d::SignatureType as S;
    use serde_json::json;
    match t {
        S::Unit => json!({"kind":"unit"}),
        S::Tuple(v) => {
            json!({"kind":"tuple","items":v.iter().map(signature_type).collect::<Vec<_>>()})
        }
        S::Scalar(d) => json!({"kind":"scalar","dtype":d.name()}),
        S::Index => json!({"kind":"index"}),
        S::Range => json!({"kind":"range"}),
        S::Tensor {
            access,
            rank,
            element,
        } => {
            json!({"kind":"tensor","access":format!("{access:?}").to_lowercase(),"rank":rank,"element":match element {d::ElementSummary::Fixed(n)=>json!({"fixed":n}),d::ElementSummary::Parameter(n)=>json!({"parameter":n})}})
        }
    }
}
fn signature(f: &d::Function) -> String {
    serde_json::json!({"name":f.name(),"parameters":f.parameters().iter().map(|(n,t)|serde_json::json!({"name":n,"type":signature_type(t)})).collect::<Vec<_>>(),"result":signature_type(f.result_type()),"elements":f.elements(),"dimensions":f.dimensions(),"numerical_subjects":f.numerical_subjects(),"scope_parameters":f.scope_parameters().iter().map(|(n,t)|serde_json::json!({"name":n,"type":signature_type(t)})).collect::<Vec<_>>()}).to_string()
}

#[pyclass(name = "Module", frozen)]
struct Module {
    inner: d::Module,
}
#[pymethods]
impl Module {
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity()
    }
    fn names(&self) -> Vec<String> {
        self.inner
            .functions()
            .iter()
            .map(|f| f.name().into())
            .collect()
    }
    fn function(&self, name: &str) -> PyResult<Function> {
        Ok(Function {
            inner: self.inner.function(name).map_err(error)?,
        })
    }
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.inner.save(&path)).map_err(error)
    }
}
#[pyfunction]
fn load(py: Python<'_>, paths: Vec<PathBuf>, include_std: bool) -> PyResult<Module> {
    py.detach(|| d::Module::load(&paths, include_std))
        .map(|inner| Module { inner })
        .map_err(error)
}
#[pyfunction]
#[pyo3(signature=(text,name,include_std,base=None))]
fn load_source(
    py: Python<'_>,
    text: String,
    name: String,
    include_std: bool,
    base: Option<PathBuf>,
) -> PyResult<Module> {
    py.detach(|| d::Module::source(&text, &name, include_std, base.as_deref()))
        .map(|inner| Module { inner })
        .map_err(error)
}

#[pyclass(name = "Scope", frozen)]
struct Scope {
    inner: d::Scope,
}

#[pyclass(name = "Function", frozen)]
struct Function {
    inner: d::Function,
}
fn options(text: &str) -> PyResult<seismic::PreparationOptions> {
    use seismic::precision::*;
    let v: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let precision = match v["precision"]["kind"].as_str().unwrap_or("exact") {
        "exact" => PrecisionPolicy::Exact,
        "unconstrained" => PrecisionPolicy::Unconstrained,
        "bounded" => {
            let p = &v["precision"];
            let tolerance = |p: &serde_json::Value| -> PyResult<Tolerance> {
                let limit = |name: &str| {
                    Limit::new(p[name].as_f64().unwrap_or(0.0))
                        .map_err(pyo3::exceptions::PyValueError::new_err)
                };
                Ok(Tolerance {
                    absolute: limit("atol")?,
                    relative: limit("rtol")?,
                    relative_floor: limit("relative_floor")?,
                    ulps: p["ulps"].as_u64(),
                })
            };
            let outputs = p["outputs"]
                .as_object()
                .map(|v| {
                    v.iter()
                        .map(|(n, v)| Ok((n.clone(), tolerance(v)?)))
                        .collect::<PyResult<BTreeMap<_, _>>>()
                })
                .transpose()?
                .unwrap_or_default();
            PrecisionPolicy::Bounded {
                default: tolerance(p)?,
                outputs,
                inputs: BTreeMap::new(),
                specials: SpecialPolicy {
                    nan: p["preserve_nan"].as_bool().unwrap_or(true),
                    infinity: p["preserve_infinity"].as_bool().unwrap_or(true),
                    signed_zero: p["preserve_signed_zero"].as_bool().unwrap_or(true),
                    subnormal: p["preserve_subnormal"].as_bool().unwrap_or(true),
                },
            }
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err("unknown precision")),
    };
    let evaluation = if v["evaluation"]["kind"] == "feedback" {
        let e = &v["evaluation"];
        let secs = e["search_seconds"].as_f64().unwrap_or(0.0);
        let search_time = std::time::Duration::try_from_secs_f64(secs)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid search duration"))?;
        seismic::EvaluationMethod::Feedback(seismic::FeedbackOptions {
            search_time,
            seed: e["seed"].as_u64().unwrap_or(0),
            experiment_memory_bytes: e["experiment_memory_bytes"].as_u64().unwrap_or(268435456),
            reference_work_limit: e["reference_work_limit"].as_u64().unwrap_or(1000000),
            ..Default::default()
        })
    } else {
        seismic::EvaluationMethod::Analytical
    };
    Ok(seismic::PreparationOptions {
        precision,
        evaluation,
    })
}
#[pymethods]
impl Function {
    fn scope(
        &self,
        py: Python<'_>,
        constraints: Vec<(String, String, Py<Scalar>, Py<Scalar>)>,
    ) -> PyResult<Scope> {
        let constraints = constraints
            .into_iter()
            .map(|(kind, name, a, b)| {
                let kind = match kind.as_str() {
                    "dimensions" => d::ScopeKind::Dimension,
                    "scalars" => d::ScopeKind::Scalar,
                    "range_starts" => d::ScopeKind::RangeStart,
                    "range_ends" => d::ScopeKind::RangeEnd,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "unknown scope kind",
                        ))
                    }
                };
                Ok((
                    kind,
                    name,
                    a.borrow(py).inner.clone(),
                    b.borrow(py).inner.clone(),
                ))
            })
            .collect::<PyResult<_>>()?;
        Ok(Scope {
            inner: self.inner.scope(constraints).map_err(error)?,
        })
    }
    #[getter]
    fn signature(&self) -> String {
        signature(&self.inner)
    }
    fn start_feedback(
        &self,
        py: Python<'_>,
        device: &Device,
        elements: BTreeMap<String, String>,
        config: &str,
        scope: Option<&Scope>,
    ) -> PyResult<(FeedbackSession, Kernel)> {
        let elements = elements
            .into_iter()
            .map(|(n, e)| Ok((n, element(&e)?)))
            .collect::<PyResult<_>>()?;
        let mut options = options(config)?;
        if let Some(scope) = scope {
            self.inner
                .apply_scope(&mut options, &scope.inner)
                .map_err(error)?;
        }
        let device = device.inner.clone();
        let (session, kernel) = py
            .detach(|| self.inner.start_feedback(&device, elements, options))
            .map_err(error)?;
        let last_report = report_json(session.report());
        Ok((
            FeedbackSession {
                inner: Some(session),
                last_report,
            },
            Kernel {
                inner: Arc::new(kernel),
            },
        ))
    }
    fn prepare(
        &self,
        py: Python<'_>,
        device: &Device,
        elements: BTreeMap<String, String>,
        config: &str,
        native: bool,
        scope: Option<&Scope>,
    ) -> PyResult<Kernel> {
        let elements = elements
            .into_iter()
            .map(|(n, e)| Ok((n, element(&e)?)))
            .collect::<PyResult<_>>()?;
        let mut options = options(config)?;
        if let Some(scope) = scope {
            self.inner
                .apply_scope(&mut options, &scope.inner)
                .map_err(error)?;
        }
        let device = device.inner.clone();
        let inner = py
            .detach(|| {
                if native {
                    self.inner.prepare_native(&device, elements)
                } else {
                    self.inner.prepare(&device, elements, options)
                }
            })
            .map_err(error)?;
        Ok(Kernel {
            inner: Arc::new(inner),
        })
    }
}
#[pyclass(name = "FeedbackSession")]
struct FeedbackSession {
    inner: Option<d::FeedbackSession>,
    last_report: String,
}
fn report_json(r: seismic::FeedbackReport) -> String {
    serde_json::json!({"elapsed":r.elapsed.as_secs_f64(),"attempted_points":r.attempted_points,"measured_points":r.measured_points,"prepared_candidates":r.prepared_candidates,"confirmed_points":r.confirmed_points,"native_budget_exhausted":r.native_budget_exhausted,"navigation_resource_limited":r.navigation_resource_limited,"numerical_pending_attempts":r.numerical_pending_attempts,"timing_endpoint":r.timing_endpoint()}).to_string()
}
#[pymethods]
impl FeedbackSession {
    fn continue_for(&mut self, py: Python<'_>, seconds: f64) -> PyResult<Kernel> {
        let duration = std::time::Duration::try_from_secs_f64(seconds)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid search duration"))?;
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("feedback session is closed"))?;
        let kernel = py.detach(|| inner.continue_for(duration)).map_err(error)?;
        self.last_report = report_json(inner.report());
        Ok(Kernel {
            inner: Arc::new(kernel),
        })
    }
    #[getter]
    fn report(&self) -> String {
        self.last_report.clone()
    }
    fn close(&mut self) {
        self.inner.take();
    }
}
#[pyclass(name = "Kernel", frozen)]
struct Kernel {
    inner: Arc<d::Kernel>,
}
#[pymethods]
impl Kernel {
    fn check(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        config: &str,
        memory_bytes: u64,
        work_limit: u64,
    ) -> PyResult<(String, String, u64)> {
        let args = args
            .iter()
            .map(|a| value(&a))
            .collect::<PyResult<Vec<_>>>()?;
        let policy = options(config)?.precision;
        let report = py
            .detach(|| self.inner.check(&args, policy, memory_bytes, work_limit))
            .map_err(error)?;
        Ok((report.status.into(), report.diagnostic, report.work_units))
    }
    #[getter]
    fn preparation_seconds(&self) -> f64 {
        self.inner.preparation_seconds
    }
    fn call(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let args = args
            .iter()
            .map(|a| value(&a))
            .collect::<PyResult<Vec<_>>>()?;
        let v = py.detach(|| self.inner.call(&args)).map_err(error)?;
        result(py, v)
    }
}
#[pyfunction]
fn element_info(name: &str) -> PyResult<(String, Option<String>)> {
    let e = element(name)?;
    Ok((e.name().into(), e.dtype().map(|d| d.name().into())))
}
#[pyclass(name = "Pending", frozen)]
struct Pending {
    inner: d::Pending,
    moved: bool,
}
#[pymethods]
impl Pending {
    #[getter]
    fn signature(&self) -> String {
        signature_type(self.inner.signature()).to_string()
    }
    fn moved(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            moved: true,
        }
    }
    fn slice(&self, start: u64, end: u64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.slice(start, end).map_err(error)?,
            moved: false,
        })
    }
}
fn workflow_value(v: &Bound<'_, PyAny>) -> PyResult<d::WorkflowValue> {
    if v.is_none() {
        return Ok(d::WorkflowValue::Unit);
    }
    if let Ok(p) = v.extract::<PyRef<Pending>>() {
        return Ok(if p.moved {
            d::WorkflowValue::Move(p.inner.clone())
        } else {
            d::WorkflowValue::Pending(p.inner.clone())
        });
    }
    if let Ok(t) = v.cast::<PyTuple>() {
        return Ok(d::WorkflowValue::Tuple(
            t.iter()
                .map(|v| workflow_value(&v))
                .collect::<PyResult<_>>()?,
        ));
    }
    Ok(d::WorkflowValue::External(value(v)?))
}
fn workflow_result(py: Python<'_>, v: d::WorkflowValue) -> PyResult<Py<PyAny>> {
    match v {
        d::WorkflowValue::Unit => Ok(py.None()),
        d::WorkflowValue::Tuple(v) => Ok(PyTuple::new(
            py,
            v.into_iter()
                .map(|v| workflow_result(py, v))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any()
        .unbind()),
        d::WorkflowValue::Pending(inner) => Ok(Py::new(
            py,
            Pending {
                inner,
                moved: false,
            },
        )?
        .into_any()),
        d::WorkflowValue::External(v) => result(py, v),
        d::WorkflowValue::Move(_) => unreachable!(),
    }
}
#[pyclass(name = "Workflow")]
struct Workflow {
    inner: d::Workflow,
}
#[pymethods]
impl Workflow {
    #[new]
    fn new(device: &Device) -> Self {
        Self {
            inner: d::Workflow::new(&device.inner),
        }
    }
    fn close(&mut self) {
        self.inner.close();
    }
    fn enqueue(
        &mut self,
        py: Python<'_>,
        kernel: &Kernel,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        let args = args
            .iter()
            .map(|v| workflow_value(&v))
            .collect::<PyResult<_>>()?;
        workflow_result(
            py,
            self.inner
                .enqueue(kernel.inner.clone(), args)
                .map_err(error)?,
        )
    }
    fn run(&mut self, py: Python<'_>, outputs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let outputs = workflow_value(outputs)?;
        let result = py.detach(|| self.inner.run(outputs)).map_err(error)?;
        workflow_result(py, result)
    }
}
#[pyfunction]
fn compare(
    py: Python<'_>,
    actual: Vec<f64>,
    expected: Vec<f64>,
    dtype: &str,
    config: &str,
    equal_nan: bool,
    signed_zero: bool,
) -> PyResult<Vec<(bool, f64, f64, u64, bool)>> {
    let dtype = element(dtype)?.dtype().ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("comparison requires a dense element")
    })?;
    if actual.len() != expected.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "comparison length mismatch",
        ));
    }
    let mut policy = options(config)?.precision;
    if let seismic::precision::PrecisionPolicy::Bounded { specials, .. } = &mut policy {
        specials.signed_zero = signed_zero;
        specials.subnormal = false;
    }
    Ok(py.detach(move || {
        actual
            .into_iter()
            .zip(expected)
            .map(|(a, e)| {
                let c = seismic::testing::compare_element(&policy, "value", dtype, e, a);
                (
                    c.accepted && (equal_nan || !(a.is_nan() || e.is_nan())),
                    c.absolute_error,
                    c.relative_error,
                    c.ulps,
                    c.special_changed,
                )
            })
            .collect()
    }))
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "PanicException",
        m.py().get_type::<pyo3::panic::PanicException>(),
    )?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_class::<Scope>()?;
    m.add_class::<Workflow>()?;
    m.add_class::<Pending>()?;
    m.add_class::<FeedbackSession>()?;
    m.add_class::<Device>()?;
    m.add_class::<Tensor>()?;
    m.add_class::<Scalar>()?;
    m.add_class::<Move>()?;
    m.add_class::<Module>()?;
    m.add_class::<Function>()?;
    m.add_class::<Kernel>()?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(device, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_source, m)?)?;
    m.add_function(wrap_pyfunction!(move_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(element_info, m)?)?;
    m.add(
        "build_profile",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(())
}
