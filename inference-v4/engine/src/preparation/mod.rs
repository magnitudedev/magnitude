//! Explicit model and input preparation (package E1).
//!
//! This module is the engine's only compiler owner above the runtime.
//! `PreparedComposition::prepare` compiles every capacity class of a declared
//! workload envelope eagerly through `PlanCompiler::compile_entry` and returns
//! only after every class is natively sealed. Execution modules consume
//! prepared types only; they cannot import or construct `PlanCompiler`.
//!
//! Class domains are disjoint and collectively cover the envelope; this is
//! validated by `WorkloadEnvelope::new`. `WorkloadEnvelope::select` is the
//! deterministic dispatch of one actual shape assignment to exactly one
//! already-prepared class; `None` rejects the invocation before submission and
//! never triggers compilation.

use crate::weights::residency::ResidentWeight;
use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
use seismic_lang::sir::Param;
use seismic_lang::types::{DType, Elem, ExtentExpr, TensorType, ValueType};
use seismic_runtime::plan::{CompiledPlan, PlanCompiler};
use seismic_runtime::Device;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::Error;

/// Preparation-phase inputs re-exported from their owner: the checked program
/// text and the compiler settings travel to preparation sessions; no other
/// engine module names the compiler's module.
pub use seismic_lang::sir::Program;
pub use seismic_runtime::plan::Settings;

/// One shape parameter's domain across the whole envelope, before it is
/// partitioned into classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeShape {
    Exact(u64),
    Bounded { min: u64, max: u64, expected: u64 },
}

impl EnvelopeShape {
    fn domain(self) -> (u64, u64) {
        match self {
            Self::Exact(value) => (value, value),
            Self::Bounded { min, max, .. } => (min, max),
        }
    }
}

/// Inclusive bounds every element of an integer control must satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerRange {
    pub min: i128,
    pub max: i128,
}
impl IntegerRange {
    pub fn contains(&self, value: i128) -> bool {
        self.min <= value && value <= self.max
    }
}

/// The general, model-agnostic workload envelope of one composition:
/// exact/bounded shape domains and an optional finite disjoint partition into
/// capacity classes. Every class is produced by the same
/// logical-to-native pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkloadEnvelope {
    shapes: BTreeMap<String, EnvelopeShape>,
    elements: BTreeMap<String, Elem>,
    classes: Vec<CapacityClass>,
}

/// One capacity class: a complete binding of every shape parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityClass {
    pub name: String,
    pub shapes: BTreeMap<String, ShapeBinding>,
}

impl CapacityClass {
    fn domain(&self) -> BTreeMap<String, (u64, u64)> {
        self.shapes
            .iter()
            .map(|(name, binding)| {
                let domain = match binding.domain() {
                    seismic_lang::logical::specialization::ShapeDomain::Exact(value) => {
                        (value, value)
                    }
                    seismic_lang::logical::specialization::ShapeDomain::Bounded { min, max } => {
                        (min, max)
                    }
                };
                (name.clone(), domain)
            })
            .collect()
    }
}

/// Why an envelope could not be constructed (a preparation-time diagnostic of
/// the caller's request).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvelopeError {
    /// Two classes admit the same shape assignment.
    Overlapping { left: String, right: String },
    /// Some shape assignment of the envelope belongs to no class.
    Uncovered { description: String },
    /// A class binds a parameter the envelope does not declare, or binds one
    /// outside the envelope domain.
    OutsideEnvelope { class: String, name: String },
    /// A bounded binding violates `min <= expected <= max`.
    InvalidBounds { name: String },
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overlapping { left, right } => {
                write!(f, "capacity classes `{left}` and `{right}` overlap")
            }
            Self::Uncovered { description } => {
                write!(f, "workload envelope region {description} belongs to no class")
            }
            Self::OutsideEnvelope { class, name } => write!(
                f,
                "class `{class}` binds shape parameter `{name}` outside the envelope"
            ),
            Self::InvalidBounds { name } => {
                write!(f, "shape parameter `{name}` violates min <= expected <= max")
            }
        }
    }
}
impl std::error::Error for EnvelopeError {}

type Box = BTreeMap<String, (u64, u64)>;

/// `whole` minus `part`, as the face slabs of `part`: for each axis in
/// order, the portion of the (successively narrowed) tube below or above
/// `part`'s interval on that axis is emitted — it lies outside `part` — and
/// the intersecting tube continues to the remaining axes, where it may still
/// escape `part`. A tube that ends inside `part`'s interval on every axis is
/// covered and dropped. A tube found disjoint on some axis never met `part`
/// and is kept whole.
fn subtract(whole: &Box, part: &Box) -> Vec<Box> {
    let mut pieces = Vec::new();
    let mut tube = whole.clone();
    for (name, &(low, high)) in part {
        let (tube_low, tube_high) = tube[name];
        if high < tube_low || low > tube_high {
            pieces.push(tube);
            return pieces;
        }
        if tube_low < low {
            let mut below = tube.clone();
            below.insert(name.clone(), (tube_low, low - 1));
            pieces.push(below);
        }
        if high < tube_high {
            let mut above = tube.clone();
            above.insert(name.clone(), (high + 1, tube_high));
            pieces.push(above);
        }
        tube.insert(name.clone(), (tube_low.max(low), tube_high.min(high)));
    }
    pieces
}

fn describe(region: &Box) -> String {
    region
        .iter()
        .map(|(name, &(low, high))| {
            if low == high {
                format!("{name}={low}")
            } else {
                format!("{name}={low}-{high}")
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

impl WorkloadEnvelope {
    /// The only constructor. With no classes, the envelope is one class: the
    /// whole declared domain. With classes, every class binds every envelope
    /// parameter inside its domain, class domains are pairwise disjoint, and
    /// they collectively cover the envelope.
    pub fn new(
        shapes: BTreeMap<String, EnvelopeShape>,
        elements: BTreeMap<String, Elem>,
        classes: Vec<CapacityClass>,
    ) -> Result<WorkloadEnvelope, EnvelopeError> {
        for (name, shape) in &shapes {
            if let EnvelopeShape::Bounded { min, max, expected } = *shape {
                if !(min <= expected && expected <= max) {
                    return Err(EnvelopeError::InvalidBounds { name: name.clone() });
                }
            }
        }
        let envelope: Box = shapes
            .iter()
            .map(|(name, shape)| (name.clone(), shape.domain()))
            .collect();
        let classes = if classes.is_empty() {
            vec![CapacityClass {
                name: "envelope".into(),
                shapes: shapes
                    .iter()
                    .map(|(name, shape)| {
                        let binding = match *shape {
                            EnvelopeShape::Exact(value) => ShapeBinding::Exact(value),
                            EnvelopeShape::Bounded { min, max, expected } => {
                                ShapeBinding::Bounded { min, max, expected }
                            }
                        };
                        (name.clone(), binding)
                    })
                    .collect(),
            }]
        } else {
            for class in &classes {
                for name in class.shapes.keys() {
                    match shapes.get(name) {
                        None => {
                            return Err(EnvelopeError::OutsideEnvelope {
                                class: class.name.clone(),
                                name: name.clone(),
                            })
                        }
                        Some(_) => {}
                    }
                }
                for (name, binding) in &class.shapes {
                    if let ShapeBinding::Bounded { min, max, expected } = *binding {
                        if !(min <= expected && expected <= max) {
                            return Err(EnvelopeError::InvalidBounds { name: name.clone() });
                        }
                    }
                    let (class_low, class_high) = match binding.domain() {
                        seismic_lang::logical::specialization::ShapeDomain::Exact(value) => {
                            (value, value)
                        }
                        seismic_lang::logical::specialization::ShapeDomain::Bounded { min, max } => {
                            (min, max)
                        }
                    };
                    let (low, high) = envelope[name];
                    if class_low < low || class_high > high {
                        return Err(EnvelopeError::OutsideEnvelope {
                            class: class.name.clone(),
                            name: name.clone(),
                        });
                    }
                }
                for name in shapes.keys() {
                    if !class.shapes.contains_key(name) {
                        return Err(EnvelopeError::Uncovered {
                            description: format!("parameter {name} in class `{}`", class.name),
                        });
                    }
                }
            }
            for (index, left) in classes.iter().enumerate() {
                for right in classes.iter().skip(index + 1) {
                    let overlapping = left.domain().iter().zip(right.domain().iter()).all(
                        |((_, (left_low, left_high)), (_, (right_low, right_high)))| {
                            left_low <= right_high && right_low <= left_high
                        },
                    );
                    if overlapping {
                        return Err(EnvelopeError::Overlapping {
                            left: left.name.clone(),
                            right: right.name.clone(),
                        });
                    }
                }
            }
            let domains: Vec<Box> = classes.iter().map(CapacityClass::domain).collect();
            let mut work = vec![(envelope.clone(), 0usize)];
            let mut uncovered: Vec<Box> = Vec::new();
            while let Some((fragment, index)) = work.pop() {
                match domains.get(index) {
                    None => uncovered.push(fragment),
                    Some(domain) => {
                        for piece in subtract(&fragment, domain) {
                            work.push((piece, index + 1));
                        }
                    }
                }
            }
            if let Some(region) = uncovered.first() {
                return Err(EnvelopeError::Uncovered {
                    description: describe(region),
                });
            }
            classes
        };
        Ok(WorkloadEnvelope {
            shapes,
            elements,
            classes,
        })
    }

    /// Geometric capacity-class partition: every bounded parameter is split
    /// at quadratically growing capacities (`1, 4, 16, ...` intersected with
    /// its domain), and the classes are the cartesian product of the
    /// per-parameter intervals. Small actuals select small-capacity classes,
    /// so capacity-sized internal storage stays proportional to the actual
    /// workload.
    pub fn geometric(
        shapes: BTreeMap<String, EnvelopeShape>,
        elements: BTreeMap<String, Elem>,
    ) -> Result<WorkloadEnvelope, EnvelopeError> {
        let mut parameters: Vec<Vec<(String, u64, u64, u64)>> = Vec::with_capacity(shapes.len());
        for (name, shape) in &shapes {
            let intervals = match *shape {
                EnvelopeShape::Exact(value) => vec![(value, value, value)],
                EnvelopeShape::Bounded { min, max, expected } => {
                    let mut capacities = (0..32)
                        .map(|power| 4u64.saturating_pow(power))
                        .take_while(|&capacity| capacity < max)
                        .filter(|&capacity| capacity >= min)
                        .collect::<Vec<_>>();
                    if capacities.last() != Some(&max) {
                        capacities.push(max);
                    }
                    let mut intervals = Vec::with_capacity(capacities.len());
                    let mut low = min;
                    for &high in &capacities {
                        intervals.push((low, high, expected.clamp(low, high)));
                        low = match high.checked_add(1) {
                            Some(next) => next,
                            None => break,
                        };
                    }
                    intervals
                }
            };
            parameters.push(
                intervals
                    .into_iter()
                    .map(|(low, high, expected)| (name.clone(), low, high, expected))
                    .collect(),
            );
        }
        let mut classes = vec![Vec::new()];
        for intervals in &parameters {
            let mut next = Vec::with_capacity(classes.len() * intervals.len());
            for prefix in &classes {
                for interval in intervals {
                    next.push(prefix.iter().chain([interval]).cloned().collect::<Vec<_>>());
                }
            }
            classes = next;
        }
        let classes = classes
            .into_iter()
            .map(|assignment| {
                let name = assignment
                    .iter()
                    .map(|(name, low, high, _)| {
                        if low == high {
                            format!("{name}={low}")
                        } else {
                            format!("{name}={low}-{high}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("/");
                CapacityClass {
                    name,
                    shapes: assignment
                        .into_iter()
                        .map(|(name, low, high, expected)| {
                            let binding = if low == high {
                                ShapeBinding::Exact(low)
                            } else {
                                ShapeBinding::Bounded {
                                    min: low,
                                    max: high,
                                    expected,
                                }
                            };
                            (name, binding)
                        })
                        .collect(),
                }
            })
            .collect();
        WorkloadEnvelope::new(shapes, elements, classes)
    }

    pub fn shapes(&self) -> &BTreeMap<String, EnvelopeShape> {
        &self.shapes
    }

    pub fn elements(&self) -> &BTreeMap<String, Elem> {
        &self.elements
    }

    pub fn classes(&self) -> &[CapacityClass] {
        &self.classes
    }

    /// Deterministic selection of the one class whose domain contains the
    /// actual shapes. A bounded binding requires the actual value; an exact
    /// binding is satisfied by its fixed value. `None` means the invocation
    /// lies outside the envelope and is rejected before submission.
    pub fn select(&self, actual: &BTreeMap<String, u64>) -> Option<usize> {
        self.classes.iter().position(|class| {
            class.shapes.iter().all(|(name, binding)| match *binding {
                ShapeBinding::Exact(value) => actual.get(name).is_none_or(|&v| v == value),
                ShapeBinding::Bounded { min, max, .. } => actual
                    .get(name)
                    .is_some_and(|&v| min <= v && v <= max),
            })
        })
    }
}

/// What preparation compiles: the composition contract (entry, envelope,
/// weights, externals, intermediates, fixed scalars). Only this module can
/// compile it.
pub struct CompositionSpec {
    pub entry: String,
    pub envelope: WorkloadEnvelope,
    pub weights: HashMap<String, ResidentWeight>,
    pub external: HashSet<String>,
    pub intermediates: HashSet<String>,
    pub scalars: HashMap<String, f64>,
}

/// A prepared composition: one sealed compiled plan per capacity class plus
/// the binding facts execution needs. Execution modules prepare and submit
/// invocations against it; they cannot compile.
pub struct PreparedComposition {
    device: seismic_runtime::Device,
    entry: String,
    envelope: WorkloadEnvelope,
    plans: Vec<CompiledPlan>,
    weights: HashMap<String, ResidentWeight>,
    external: HashSet<String>,
    intermediates: HashMap<String, TensorType>,
    runtime_scalars: HashSet<String>,
    scalars: HashMap<String, f64>,
    control_types: HashMap<String, DType>,
    control_domains: HashMap<String, IntegerRange>,
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::Request(error.to_string())
}

/// Entry extents resolved against one class binding: an exact parameter
/// contributes its value, a bounded parameter its capacity. Resident model
/// state must cover the whole class.
fn class_extents(
    tensor: &TensorType,
    class: &BTreeMap<String, ShapeBinding>,
) -> Result<Vec<u64>, String> {
    for (name, binding) in class {
        let capacity = binding.capacity();
        if i64::try_from(capacity).is_err() {
            return Err(format!(
                "shape parameter {name} capacity {capacity} exceeds the 64-bit signed index limit"
            ));
        }
    }
    tensor.axes.iter().map(|axis| match axis {
        ExtentExpr::Sym(extent) => extent
            .eval(&|parameter| {
                class.get(parameter).and_then(|binding| match *binding {
                    ShapeBinding::Exact(value) => i64::try_from(value).ok(),
                    ShapeBinding::Bounded { max, .. } => i64::try_from(max).ok(),
                })
            })
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| "unresolved entry tensor shape".to_string()),
        ExtentExpr::Static(extent) => Ok(*extent),
        ExtentExpr::Runtime(id) => Err(format!(
            "entry tensor runtime extent {id:?} has no caller-provided value"
        )),
    }).collect()
}

fn tensor<'a>(params: &'a [Param], name: &str) -> Option<&'a TensorType> {
    params.iter().find_map(|p| match &p.ty {
        ValueType::Tensor(t) if p.name == name => Some(t),
        _ => None,
    })
}

impl PreparedComposition {
    /// Prepare every class of the spec. Returns only after every class has
    /// been natively sealed; a failure retains nothing.
    pub fn prepare(
        compiler: &mut PlanCompiler<'_>,
        spec: CompositionSpec,
    ) -> Result<PreparedComposition, Error> {
        let CompositionSpec {
            entry,
            envelope,
            weights,
            external,
            intermediates,
            scalars,
        } = spec;
        let program: &Program = compiler.program();
        let family = program
            .resolve_family(&entry)
            .map_err(|e| invalid(format!("{entry}: {e}")))?;
        // Every implementation in a linked family shares one contract; the
        // first function body states the entry ABI.
        let params = &program
            .definition(
                *family
                    .bodies
                    .iter()
                    .chain(&family.lowerings)
                    .next()
                    .ok_or_else(|| invalid(format!("{entry}: composition has no implementation")))?,
            )
            .params;
        let mut elements = envelope
            .elements()
            .iter()
            .map(|(name, elem)| (name.clone(), elem.clone()))
            .collect::<HashMap<String, Elem>>();
        for (name, weight) in &weights {
            if !weight.belongs_to(compiler.device()) {
                return Err(invalid(format!(
                    "weight {name} belongs to another resource domain"
                )));
            }
            let ValueType::Tensor(t) = &params
                .iter()
                .find(|p| &p.name == name)
                .ok_or_else(|| invalid(format!("unknown weight {name}")))?
                .ty
            else {
                return Err(invalid(format!("weight {name} is not a tensor")));
            };
            for class in envelope.classes() {
                let expected = class_extents(t, &class.shapes).map_err(invalid)?;
                if expected != weight.descriptor().shape {
                    return Err(invalid(format!(
                        "weight {name} shape differs from composition class `{}`",
                        class.name
                    )));
                }
            }
            match &t.elem {
                Elem::Param(p) => {
                    if let Some(previous) = elements.insert(p.clone(), weight.element().clone()) {
                        if &previous != weight.element() {
                            return Err(invalid(format!(
                                "weight {name} disagrees on element parameter {p}"
                            )));
                        }
                    }
                }
                concrete if concrete != weight.element() => {
                    return Err(invalid(format!(
                        "weight {name} has wrong element type"
                    )));
                }
                _ => {}
            }
        }
        for name in &external {
            if weights.contains_key(name) || tensor(params, name).is_none() {
                return Err(invalid(format!("invalid external tensor {name}")));
            }
        }
        let mut intermediate_types = HashMap::new();
        for name in &intermediates {
            let t = tensor(params, name)
                .ok_or_else(|| invalid(format!("invalid intermediate tensor {name}")))?;
            if weights.contains_key(name) || external.contains(name) {
                return Err(invalid(format!("invalid intermediate tensor {name}")));
            }
            intermediate_types.insert(name.clone(), t.clone());
        }
        for Param { name, ty, .. } in params {
            if matches!(ty, ValueType::Tensor(_))
                && !weights.contains_key(name)
                && !external.contains(name)
                && !intermediates.contains(name)
            {
                return Err(invalid(format!(
                    "composition tensor {name} has no declared owner"
                )));
            }
        }
        for name in scalars.keys() {
            if !params
                .iter()
                .any(|p| &p.name == name && matches!(p.ty, ValueType::Scalar(_)))
            {
                return Err(invalid(format!("invalid scalar {name}")));
            }
        }
        let runtime_scalars = params
            .iter()
            .filter(|p| matches!(p.ty, ValueType::Scalar(_)) && !scalars.contains_key(&p.name))
            .map(|p| p.name.clone())
            .collect::<HashSet<_>>();
        let mut control_types = HashMap::new();
        for Param { name, ty, .. } in params {
            if !external.contains(name) {
                continue;
            }
            let ValueType::Tensor(t) = ty else {
                continue;
            };
            let element = match &t.elem {
                Elem::Param(parameter) => elements
                    .get(parameter.as_str())
                    .ok_or_else(|| invalid("unbound control dtype"))?,
                element => element,
            };
            let Elem::Dtype(dtype @ (DType::I32 | DType::U32)) = element else {
                continue;
            };
            control_types.insert(name.clone(), *dtype);
        }
        let mut plans = Vec::with_capacity(envelope.classes().len());
        for class in envelope.classes() {
            let domain = SpecializationDomain::new(
                compiler.program(),
                &entry,
                class.shapes.clone(),
                elements
                    .iter()
                    .map(|(name, elem)| (name.clone(), elem.clone()))
                    .collect(),
            )
            .map_err(|e| invalid(format!("{entry}: {e}")))?;
            plans.push(
                compiler
                    .compile_entry(&domain)
                    .map_err(|e| invalid(format!("{entry}/{}: {e}", class.name)))?,
            );
        }
        Ok(PreparedComposition {
            device: compiler.device().clone(),
            entry,
            envelope,
            plans,
            weights,
            external,
            intermediates: intermediate_types,
            runtime_scalars,
            scalars,
            control_types,
            control_domains: HashMap::new(),
        })
    }

    /// Admit only declared control inputs. Changing integer controls is
    /// guarded by `control_domain`; other external tensors are data.
    pub fn control_inputs(mut self, names: &[&str]) -> Result<Self, Error> {
        if names.iter().any(|name| !self.external.contains(*name)) {
            return Err(invalid("control input is not an external tensor"));
        }
        Ok(self)
    }

    /// Admit changing integer controls only under one checked finite domain,
    /// established on every invocation before any numerical work is bound.
    pub fn control_domain(
        mut self,
        name: &str,
        range: IntegerRange,
    ) -> Result<Self, Error> {
        if !self.control_types.contains_key(name) {
            return Err(invalid(
                "varying control domain requires an external integer tensor",
            ));
        }
        self.control_domains.insert(name.into(), range);
        Ok(self)
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub fn envelope(&self) -> &WorkloadEnvelope {
        &self.envelope
    }

    pub fn plan(&self, class: usize) -> Option<&CompiledPlan> {
        self.plans.get(class)
    }

    pub fn kernel_count(&self) -> usize {
        self.plans.iter().map(CompiledPlan::kernel_count).sum()
    }

    pub(crate) fn device(&self) -> &seismic_runtime::Device {
        &self.device
    }
    pub(crate) fn weights(&self) -> &HashMap<String, ResidentWeight> {
        &self.weights
    }
    pub(crate) fn external(&self) -> &HashSet<String> {
        &self.external
    }
    pub(crate) fn intermediates(&self) -> &HashMap<String, TensorType> {
        &self.intermediates
    }
    pub(crate) fn runtime_scalars(&self) -> &HashSet<String> {
        &self.runtime_scalars
    }
    pub(crate) fn scalars(&self) -> &HashMap<String, f64> {
        &self.scalars
    }
    pub(crate) fn control_types(&self) -> &HashMap<String, DType> {
        &self.control_types
    }
    pub(crate) fn control_domains(&self) -> &HashMap<String, IntegerRange> {
        &self.control_domains
    }
}


/// One preparation session: the engine's only handle to the compiler. Every
/// module above preparation (model construction, conditioned/vision input
/// preparation, weight import) compiles through this type; none can name
/// `PlanCompiler`.
pub struct PreparationSession<'a> {
    compiler: PlanCompiler<'a>,
}

impl<'a> PreparationSession<'a> {
    pub fn new(device: &'a Device, program: &'a Program, settings: Settings) -> Self {
        Self {
            compiler: PlanCompiler::new(device, program, settings),
        }
    }

    /// Prepare every class of one composition spec; returns only after every
    /// class is natively sealed.
    pub fn prepare(&mut self, spec: CompositionSpec) -> Result<PreparedComposition, Error> {
        PreparedComposition::prepare(&mut self.compiler, spec)
    }

    /// Compile one standalone entry under one complete specialization domain
    /// (the weight-import kernels). Eager: the returned plan is natively
    /// sealed.
    pub fn compile_entry(
        &mut self,
        entry: &str,
        shapes: BTreeMap<String, ShapeBinding>,
        elements: BTreeMap<String, Elem>,
    ) -> Result<CompiledPlan, Error> {
        let domain = SpecializationDomain::new(self.compiler.program(), entry, shapes, elements)
            .map_err(invalid)?;
        let started = std::time::Instant::now();
        let plan = self
            .compiler
            .compile_entry(&domain)
            .map_err(|e| invalid(format!("{entry}: {e}")))?;
        crate::telemetry::span_compile(
            entry,
            plan.kernel_count() as u64,
            plan.estimated_cost(),
            plan.optimal(),
            started.elapsed().as_secs_f64(),
        );
        Ok(plan)
    }
}

impl From<EnvelopeError> for Error {
    fn from(error: EnvelopeError) -> Self {
        Error::Request(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(x: (u64, u64), y: (u64, u64)) -> Box {
        BTreeMap::from([("x".to_string(), x), ("y".to_string(), y)])
    }

    fn class(name: &str, x: (u64, u64), y: (u64, u64)) -> CapacityClass {
        CapacityClass {
            name: name.into(),
            shapes: BTreeMap::from([
                (
                    "x".into(),
                    if x.0 == x.1 {
                        ShapeBinding::Exact(x.0)
                    } else {
                        ShapeBinding::Bounded {
                            min: x.0,
                            max: x.1,
                            expected: x.0,
                        }
                    },
                ),
                (
                    "y".into(),
                    if y.0 == y.1 {
                        ShapeBinding::Exact(y.0)
                    } else {
                        ShapeBinding::Bounded {
                            min: y.0,
                            max: y.1,
                            expected: y.0,
                        }
                    },
                ),
            ]),
        }
    }

    fn envelope(classes: Vec<CapacityClass>) -> Result<WorkloadEnvelope, EnvelopeError> {
        WorkloadEnvelope::new(
            BTreeMap::from([
                (
                    "x".into(),
                    EnvelopeShape::Bounded {
                        min: 0,
                        max: 10,
                        expected: 0,
                    },
                ),
                (
                    "y".into(),
                    EnvelopeShape::Bounded {
                        min: 0,
                        max: 10,
                        expected: 0,
                    },
                ),
            ]),
            BTreeMap::from([("A".into(), Elem::Dtype(DType::F32))]),
            classes,
        )
    }

    #[test]
    fn subtraction_keeps_the_intersecting_tube_for_later_axes() {
        // [0,10]^2 minus [0,7]^2 is the L-shape, not only the far corner:
        // the x-intersecting tube must survive to be split on y.
        let pieces = subtract(&region((0, 10), (0, 10)), &region((0, 7), (0, 7)));
        assert_eq!(pieces.len(), 2);
        assert!(pieces.contains(&region((8, 10), (0, 10))));
        assert!(pieces.contains(&region((0, 7), (8, 10))));
    }

    #[test]
    fn subtraction_drops_the_fully_covered_tube() {
        assert_eq!(subtract(&region((0, 10), (0, 10)), &region((0, 10), (0, 10))), Vec::<Box>::new());
        assert_eq!(subtract(&region((3, 5), (2, 9)), &region((0, 10), (0, 10))), Vec::<Box>::new());
    }

    #[test]
    fn diagonal_classes_leave_their_corners_uncovered() {
        // [0,7]^2 union [3,10]^2 leaves [8,10]x[0,2] and [0,2]x[8,10]
        // uncovered; the subtraction chain must certify exactly that.
        let l_shape = subtract(&region((0, 10), (0, 10)), &region((0, 7), (0, 7)));
        let remainder = l_shape
            .iter()
            .flat_map(|fragment| subtract(fragment, &region((3, 10), (3, 10))))
            .collect::<Vec<_>>();
        assert_eq!(remainder.len(), 2);
        assert!(remainder.contains(&region((8, 10), (0, 2))));
        assert!(remainder.contains(&region((0, 2), (8, 10))));
    }

    #[test]
    fn disjoint_classes_that_erase_one_axis_are_not_certified_as_covering() {
        // The strip class spans the whole x domain; only y in [0,4] is
        // covered, so the middle column of the upper band stays uncovered.
        let result = envelope(vec![
            class("bottom", (0, 10), (0, 4)),
            class("upper-left", (0, 4), (5, 10)),
            class("upper-right", (6, 10), (5, 10)),
        ]);
        assert!(matches!(
            result,
            Err(EnvelopeError::Uncovered { description }) if description.contains("x=5")
        ));
    }

    #[test]
    fn disjoint_covering_classes_are_accepted() {
        let result = envelope(vec![
            class("lower-left", (0, 4), (0, 4)),
            class("lower-right", (5, 10), (0, 4)),
            class("upper-left", (0, 4), (5, 10)),
            class("upper-right", (5, 10), (5, 10)),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn one_class_equal_to_the_envelope_terminates_with_no_fragments() {
        let result = envelope(vec![class("whole", (0, 10), (0, 10))]);
        assert!(result.is_ok());
        assert_eq!(result.expect("validated").classes().len(), 1);
    }

    #[test]
    fn geometric_partitions_are_disjoint_and_covering() {
        let envelope = WorkloadEnvelope::geometric(
            BTreeMap::from([
                (
                    "M".into(),
                    EnvelopeShape::Bounded {
                        min: 1,
                        max: 10,
                        expected: 2,
                    },
                ),
                (
                    "R".into(),
                    EnvelopeShape::Bounded {
                        min: 1,
                        max: 6,
                        expected: 1,
                    },
                ),
            ]),
            BTreeMap::from([("A".into(), Elem::Dtype(DType::F32))]),
        )
        .expect("geometric classes cover and are disjoint");
        for m in 1..=10u64 {
            for r in 1..=6u64 {
                let actual = BTreeMap::from([("M".to_string(), m), ("R".to_string(), r)]);
                assert!(envelope.select(&actual).is_some(), "M={m} R={r} selects a class");
                let matches = envelope
                    .classes()
                    .iter()
                    .filter(|class| {
                        class.shapes.iter().all(|(name, binding)| {
                            let value = if name == "M" { m } else { r };
                            match *binding {
                                ShapeBinding::Exact(exact) => exact == value,
                                ShapeBinding::Bounded { min, max, .. } => {
                                    min <= value && value <= max
                                }
                            }
                        })
                    })
                    .count();
                assert_eq!(matches, 1, "M={m} R={r} matches exactly one class");
            }
        }
    }
}
