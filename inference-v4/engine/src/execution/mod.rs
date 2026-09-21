//! Invocation validation and submission over sealed prepared compositions.
//! Model policy chooses parameters; this owner validates their contracts and
//! executes prepared invocations. No compilation path exists here (package
//! E1): `PlanCompiler` is importable only by the preparation module.
use crate::preparation::PreparedComposition;
use seismic_lang::types::ExtentExpr;
use seismic_realization::failure::{ExecutionFailure, InvalidInvocation};
use seismic_runtime::invocation::{Bindings, PreparedInvocation};
use seismic_runtime::plan::{CompiledPlan, InvocationResults};
use seismic_runtime::submission::Submission;
use seismic_runtime::{Buffer, ExecutionObservation};
use std::collections::{BTreeMap, HashMap};

use crate::Error;

/// Why one prepared stage could not be invoked or executed. Classes follow
/// the engine rows of the closure ledger: an invocation outside the prepared
/// envelope and a contract rejection are `Invalid`; execution failures are
/// `Safety`/`External` inside [`ExecutionFailure`]; everything else is an
/// engine-side request diagnostic.
#[derive(Debug)]
pub enum StageError {
    /// The actual shapes select no prepared class; rejected before
    /// submission.
    OutsideEnvelope {
        entry: String,
        actual: Vec<(String, u64)>,
    },
    Invalid(InvalidInvocation),
    Execution(ExecutionFailure),
    Capacity { required: usize, available: usize },
    Request(String),
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutsideEnvelope { entry, actual } => write!(
                f,
                "forward geometry {actual:?} lies outside the prepared envelope of {entry}"
            ),
            Self::Invalid(failure) => write!(f, "{failure}"),
            Self::Execution(failure) => write!(f, "{failure}"),
            Self::Capacity {
                required,
                available,
            } => write!(
                f,
                "allocation requires {required} bytes; {available} charged bytes available"
            ),
            Self::Request(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for StageError {}

impl From<StageError> for Error {
    fn from(error: StageError) -> Self {
        match error {
            StageError::OutsideEnvelope { entry, actual } => Self::Request(format!(
                "forward geometry {actual:?} lies outside the prepared envelope of {entry}"
            )),
            StageError::Invalid(failure) => Self::Invocation(failure),
            StageError::Execution(failure) => Self::Execution(failure),
            StageError::Capacity {
                required,
                available,
            } => Self::Capacity {
                required,
                available,
            },
            StageError::Request(message) => Self::Request(message),
        }
    }
}

struct StageInvocation<'a> {
    composition: &'a PreparedComposition,
    scratch: HashMap<String, Buffer>,
    tensors: &'a HashMap<String, Buffer>,
    scalars: &'a HashMap<String, f64>,
    shapes: &'a BTreeMap<String, u64>,
}

impl Bindings for StageInvocation<'_> {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        if let Some(weight) = self.composition.weights().get(root) {
            return weight.plane(plane);
        }
        if !plane.is_empty() {
            return None;
        }
        self.tensors
            .get(root)
            .or_else(|| self.scratch.get(root))
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        self.composition
            .scalars()
            .get(name)
            .or_else(|| self.scalars.get(name))
            .copied()
    }
    fn shape(&self, name: &str) -> Option<u64> {
        self.shapes.get(name).copied()
    }
}

fn request(message: impl Into<String>) -> StageError {
    StageError::Request(message.into())
}

fn actual_extents(
    tensor: &seismic_lang::types::TensorType,
    shapes: &BTreeMap<String, u64>,
) -> Result<Vec<usize>, StageError> {
    for (name, &value) in shapes {
        if i64::try_from(value).is_err() {
            return Err(request(format!(
                "shape parameter {name} value {value} exceeds the 64-bit signed index limit"
            )));
        }
    }
    tensor
        .axes
        .iter()
        .map(|axis| match axis {
            ExtentExpr::Sym(extent) => extent
                .eval(&|parameter| {
                    shapes.get(parameter).and_then(|&v| i64::try_from(v).ok())
                })
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| request("unresolved intermediate tensor shape")),
            ExtentExpr::Static(extent) => usize::try_from(*extent)
                .map_err(|_| request("static intermediate extent exceeds usize")),
            ExtentExpr::Runtime(id) => Err(request(format!(
                "intermediate runtime extent {id:?} has no caller-provided value"
            ))),
        })
        .collect()
}

fn validate_controls(
    composition: &PreparedComposition,
    tensors: &HashMap<String, Buffer>,
) -> Result<(), StageError> {
    for (name, range) in composition.control_domains() {
        let dtype = composition.control_types()[name];
        let buffer = tensors
            .get(name)
            .ok_or_else(|| request("unbound integer control"))?;
        let mut bytes = vec![0; buffer.len()];
        buffer
            .read(&mut bytes)
            .map_err(|e| request(format!("reading integer control {name}: {e}")))?;
        if bytes.len() % 4 != 0 {
            return Err(request("integer control width is not a whole number of words"));
        }
        for word in bytes.chunks_exact(4) {
            let raw: [u8; 4] = word.try_into().expect("integer control width");
            let value = if dtype == seismic_lang::types::DType::I32 {
                i128::from(i32::from_le_bytes(raw))
            } else {
                i128::from(u32::from_le_bytes(raw))
            };
            if !range.contains(value) {
                return Err(request(
                    "invocation does not establish its integer input domain",
                ));
            }
        }
    }
    Ok(())
}

/// Validate one invocation against the composition's prepared envelope and
/// its sealed contract. Never compiles; class selection is dispatch among
/// already-prepared compiled plans.
pub fn invoke(
    composition: &PreparedComposition,
    shapes: &BTreeMap<String, u64>,
    tensors: &HashMap<String, Buffer>,
    scalars: &HashMap<String, f64>,
) -> Result<PreparedInvocation, StageError> {
    if tensors
        .values()
        .any(|buffer| !buffer.belongs_to(composition.device()))
    {
        return Err(request(
            "composition tensor belongs to another resource domain",
        ));
    }
    if tensors.len() != composition.external().len()
        || tensors.keys().any(|n| !composition.external().contains(n))
    {
        return Err(request(
            "composition external bindings differ from declared inputs/outputs",
        ));
    }
    if scalars.len() != composition.runtime_scalars().len()
        || scalars.keys().any(|n| !composition.runtime_scalars().contains(n))
    {
        return Err(request(
            "composition runtime scalar bindings differ from declared parameters",
        ));
    }
    validate_controls(composition, tensors)?;
    let class = composition
        .envelope()
        .select(shapes)
        .ok_or_else(|| StageError::OutsideEnvelope {
            entry: composition.entry().to_string(),
            actual: shapes.iter().map(|(n, &v)| (n.clone(), v)).collect(),
        })?;
    let plan: &CompiledPlan =
        composition.plan(class).ok_or_else(|| request("prepared class has no compiled plan"))?;
    let mut scratch = HashMap::new();
    for (name, tensor) in composition.intermediates() {
        let element = match &tensor.elem {
            seismic_lang::types::Elem::Param(parameter) => composition
                .envelope()
                .elements()
                .get(parameter.as_str())
                .ok_or_else(|| request("unbound intermediate dtype"))?,
            element => element,
        };
        let seismic_lang::types::Elem::Dtype(dtype) = element else {
            return Err(request("intermediate tensor must be dense"));
        };
        let bytes = actual_extents(tensor, shapes)?
            .into_iter()
            .try_fold(dtype.bytes() as usize, |n, d| {
                n.checked_mul(d)
                    .ok_or_else(|| request("intermediate allocation overflow"))
            })?;
        let buffer = composition
            .device()
            .buffer(bytes)
            .map_err(|error| match error {
                seismic_runtime::Error::Capacity {
                    required,
                    available,
                } => StageError::Capacity {
                    required,
                    available,
                },
                seismic_runtime::Error::External(failure) => {
                    StageError::Execution(ExecutionFailure::External(failure))
                }
                seismic_runtime::Error::LimitBelowCharges { limit, charged } => {
                    StageError::Request(format!(
                        "allocation limit {limit} cannot be below the {charged} retained charged bytes"
                    ))
                }
                seismic_runtime::Error::Range {
                    requested,
                    available,
                } => StageError::Request(format!(
                    "{requested} bytes requested of a {available}-byte bound range"
                )),
            })?;
        scratch.insert(name.clone(), buffer);
    }
    let invocation = StageInvocation {
        composition,
        scratch,
        tensors,
        scalars,
        shapes,
    };
    plan.prepare(&invocation).map_err(|failure| match failure {
        ExecutionFailure::Invocation(invalid) => StageError::Invalid(invalid),
        other => StageError::Execution(other),
    })
}

/// Invoke one compiled plan directly with caller-supplied bindings (the
/// weight-import path: model loading before residency).
pub fn invoke_plan(
    plan: &CompiledPlan,
    bindings: &dyn Bindings,
) -> Result<PreparedInvocation, StageError> {
    plan.prepare(bindings).map_err(|failure| match failure {
        ExecutionFailure::Invocation(invalid) => StageError::Invalid(invalid),
        other => StageError::Execution(other),
    })
}

fn sole(mut results: Vec<InvocationResults>) -> Result<InvocationResults, StageError> {
    if results.len() != 1 {
        return Err(request("submission lost its invocation"));
    }
    Ok(results.pop().expect("length checked"))
}

/// Execute one prepared invocation synchronously.
pub fn run(invocation: PreparedInvocation) -> Result<InvocationResults, StageError> {
    sole(
        Submission::single(invocation)
            .execute()
            .map_err(StageError::Execution)?,
    )
}

/// Execute one prepared invocation synchronously with observation.
pub fn run_observed(
    invocation: PreparedInvocation,
) -> Result<(InvocationResults, ExecutionObservation), StageError> {
    let mut observed = Submission::single(invocation)
        .execute_observed()
        .map_err(StageError::Execution)?;
    if observed.len() != 1 {
        return Err(request("submission lost its invocation"));
    }
    Ok(observed.pop().expect("length checked"))
}

/// Invoke and execute one stage in a single synchronous step.
pub fn execute(
    composition: &PreparedComposition,
    shapes: &BTreeMap<String, u64>,
    tensors: &HashMap<String, Buffer>,
    scalars: &HashMap<String, f64>,
) -> Result<InvocationResults, StageError> {
    run(invoke(composition, shapes, tensors, scalars)?)
}

/// Result-plane handles of one prepared invocation, captured before
/// execution. Buffer contents are valid after the invocation's synchronous
/// execution completes.
pub fn result_planes(invocation: &PreparedInvocation) -> InvocationResults {
    InvocationResults {
        planes: invocation.results().to_vec(),
        scalars: Vec::new(),
    }
}

/// The dense result buffer of one owned result plane path.
pub fn result_buffer(results: &InvocationResults, path: &[u32]) -> Result<Buffer, StageError> {
    results
        .planes
        .iter()
        .find(|result| result.path == path && result.plane.is_empty())
        .map(|result| result.buffer.clone())
        .ok_or_else(|| request(format!("owned result path {path:?} has no dense plane")))
}

/// A batch of prepared invocations executed in source order through one
/// synchronous submission.
#[derive(Default)]
pub struct StageBatch {
    submission: Submission,
}

impl StageBatch {
    pub fn add(&mut self, invocation: PreparedInvocation) {
        self.submission.append(Submission::single(invocation));
    }
    pub fn len(&self) -> usize {
        self.submission.len()
    }
    pub fn is_empty(&self) -> bool {
        self.submission.is_empty()
    }
    pub fn execute(self) -> Result<Vec<InvocationResults>, StageError> {
        self.submission.execute().map_err(StageError::Execution)
    }
    pub fn execute_observed(
        self,
    ) -> Result<Vec<(InvocationResults, ExecutionObservation)>, StageError> {
        self.submission
            .execute_observed()
            .map_err(StageError::Execution)
    }
}

/// Aggregate per-invocation observations into one batch observation.
pub fn batch_observation(
    observed: &[(InvocationResults, ExecutionObservation)],
) -> ExecutionObservation {
    ExecutionObservation {
        host_seconds: observed.iter().map(|(_, o)| o.host_seconds).sum(),
        launches: observed
            .iter()
            .flat_map(|(_, observation)| observation.launches.iter().cloned())
            .collect(),
    }
}
