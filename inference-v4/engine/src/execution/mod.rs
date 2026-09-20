//! Resident bindings for checked numerical compositions. Model policy chooses
//! parameters; this owner validates their contracts and retains scratch/code.
use crate::weights::residency::ResidentWeight;
use seismic_lang::{
    sir::Param,
    types::{DType, Elem, ExtentExpr, TensorType, ValueType},
};
use seismic_runtime::{
    plan::{Bindings, CompiledPlan, InvocationResults, PlanCompiler, StepObservation, Submission},
    Buffer, Device, Error,
};
use std::collections::{HashMap, HashSet};

pub struct Composition {
    device: Device,
    spec: CompositionSpec,
    control_types: HashMap<String, (DType, usize)>,
    control_domains: HashMap<String, IntegerRange>,
    plan: CompiledPlan,
    weights: HashMap<String, ResidentWeight>,
    scratch: HashMap<String, Buffer>,
    external: HashSet<String>,
    runtime_scalars: HashSet<String>,
    scalars: HashMap<String, f64>,
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
/// Entry parameters carry only semantic extents; a slice width has no caller value.
fn extents(tensor: &TensorType, shapes: &HashMap<String, i64>) -> Result<Vec<usize>, String> {
    tensor
        .axes
        .iter()
        .map(|axis| match axis {
            ExtentExpr::Sym(extent) => extent
                .eval(&|p| shapes.get(p).copied())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| "unresolved entry tensor shape".to_string()),
            ExtentExpr::Static(_) | ExtentExpr::Runtime(_) => {
                Err("entry tensor extent is not a shape parameter".into())
            }
        })
        .collect()
}
fn tensor<'a>(params: &'a [Param], name: &str) -> Option<&'a TensorType> {
    params.iter().find_map(|p| match &p.ty {
        ValueType::Tensor(t) if p.name == name => Some(t),
        _ => None,
    })
}
#[derive(Clone)]
pub struct CompositionSpec {
    pub entry: String,
    pub shapes: HashMap<String, i64>,
    pub elements: HashMap<String, Elem>,
    pub weights: HashMap<String, ResidentWeight>,
    pub external: HashSet<String>,
    pub intermediates: HashSet<String>,
    pub scalars: HashMap<String, f64>,
}
impl Composition {
    pub fn compile(compiler: &mut PlanCompiler<'_>, spec: CompositionSpec) -> Result<Self, Error> {
        let retained = spec.clone();
        let CompositionSpec {
            entry,
            shapes,
            mut elements,
            weights,
            external,
            intermediates,
            scalars,
        } = spec;
        let program = compiler.program();
        let family = program.resolve_family(&entry)?;
        // Every implementation in a linked family shares one contract; the first
        // function body states the entry ABI.
        let params = &program
            .definition(
                *family
                    .bodies
                    .iter()
                    .chain(&family.lowerings)
                    .next()
                    .ok_or("composition has no implementation")?,
            )
            .params;
        for (name, weight) in &weights {
            if !weight.belongs_to(compiler.device()) {
                return Err(format!("weight {name} belongs to another resource domain").into());
            }
            let ValueType::Tensor(t) = &params
                .iter()
                .find(|p| &p.name == name)
                .ok_or_else(|| format!("unknown weight {name}"))?
                .ty
            else {
                return Err(format!("weight {name} is not a tensor").into());
            };
            let expected = extents(t, &shapes)?
                .into_iter()
                .map(|v| v as u64)
                .collect::<Vec<_>>();
            if expected != weight.descriptor().shape {
                return Err(format!("weight {name} shape differs from composition").into());
            }
            match &t.elem {
                Elem::Param(p) => {
                    if let Some(previous) = elements.insert(p.clone(), weight.element().clone()) {
                        if &previous != weight.element() {
                            return Err(format!(
                                "weight {name} disagrees on element parameter {p}"
                            )
                            .into());
                        }
                    }
                }
                concrete if concrete != weight.element() => {
                    return Err(format!("weight {name} has wrong element type").into());
                }
                _ => {}
            }
        }
        for name in &external {
            if weights.contains_key(name) || tensor(params, name).is_none() {
                return Err(format!("invalid external tensor {name}").into());
            }
        }
        for name in &intermediates {
            if weights.contains_key(name)
                || external.contains(name)
                || tensor(params, name).is_none()
            {
                return Err(format!("invalid intermediate tensor {name}").into());
            }
        }
        for Param { name, ty, .. } in params {
            if matches!(ty, ValueType::Tensor(_))
                && !weights.contains_key(name)
                && !external.contains(name)
                && !intermediates.contains(name)
            {
                return Err(format!("composition tensor {name} has no declared owner").into());
            }
        }
        for name in scalars.keys() {
            if !params
                .iter()
                .any(|p| &p.name == name && matches!(p.ty, ValueType::Scalar(_)))
            {
                return Err(format!("invalid scalar {name}").into());
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
            let ValueType::Tensor(tensor) = ty else {
                continue;
            };
            let element = match &tensor.elem {
                Elem::Param(parameter) => elements
                    .get(parameter.as_str())
                    .ok_or("unbound control dtype")?,
                element => element,
            };
            let Elem::Dtype(dtype @ (DType::I32 | DType::U32)) = element else {
                continue;
            };
            let count = extents(tensor, &shapes)?
                .into_iter()
                .try_fold(1usize, |n, extent| n.checked_mul(extent))
                .ok_or("control tensor extent overflow")?;
            control_types.insert(name.clone(), (*dtype, count));
        }
        let mut scratch = HashMap::new();
        for Param { name, ty, .. } in params {
            if weights.contains_key(name) || external.contains(name) {
                continue;
            }
            if let ValueType::Tensor(t) = ty {
                let element = match &t.elem {
                    Elem::Param(p) => elements.get(p.as_str()).ok_or("unbound scratch dtype")?,
                    e => e,
                };
                let Elem::Dtype(dtype) = element else {
                    return Err("composition scratch must be dense".into());
                };
                let bytes = extents(t, &shapes)?
                    .into_iter()
                    .try_fold(dtype.bytes() as usize, |n, d| n.checked_mul(d))
                    .ok_or("scratch allocation overflow")?;
                scratch.insert(name.clone(), compiler.device().buffer(bytes)?);
            }
        }
        Ok(Self {
            device: compiler.device().clone(),
            spec: retained,
            control_types,
            control_domains: HashMap::new(),
            plan: compiler.compile_entry(&entry, &shapes, &elements)?,
            weights,
            scratch,
            external,
            runtime_scalars,
            scalars,
        })
    }
    /// Specialize logical dimensions while sharing resident model weights.
    pub(crate) fn with_dimensions(
        &self,
        compiler: &mut PlanCompiler<'_>,
        dimensions: &[(&str, usize)],
    ) -> Result<Self, Error> {
        let mut spec = self.spec.clone();
        for &(name, extent) in dimensions {
            if extent == 0 || !spec.shapes.contains_key(name) {
                return Err("specialization requires positive declared dimensions".into());
            }
            spec.shapes.insert(
                name.into(),
                i64::try_from(extent).map_err(|_| "specialized dimension overflow")?,
            );
        }
        let mut composition = Self::compile(compiler, spec)?;
        composition.control_domains = self.control_domains.clone();
        Ok(composition)
    }
    pub(crate) fn control_inputs(self, names: &[&str]) -> Result<Self, String> {
        if names.iter().any(|name| !self.external.contains(*name)) {
            return Err("control input is not an external tensor".into());
        }
        Ok(self)
    }
    /// Admit changing integer controls only under one checked finite domain,
    /// established on every invocation before any numerical work is bound.
    pub(crate) fn control_domain(
        mut self,
        name: &str,
        range: IntegerRange,
    ) -> Result<Self, String> {
        if !self.control_types.contains_key(name) {
            return Err("varying control domain requires an external integer tensor".into());
        }
        self.control_domains.insert(name.into(), range);
        Ok(self)
    }
    pub(crate) fn shares_compilation(&self, other: &Self) -> bool {
        self.plan.shares_compilation(&other.plan)
    }
    pub fn kernel_count(&self) -> usize {
        self.plan.kernel_count()
    }
    /// What selection decided, once this composition's kernel exists. Reading it
    /// never triggers selection or native compilation.
    pub fn selection(&self) -> Result<Option<seismic_runtime::Selection>, String> {
        if self.plan.kernel_count() == 0 {
            return Ok(None);
        }
        let kernel = self.plan.kernel()?;
        let kernel = kernel
            .try_borrow()
            .map_err(|_| "shared kernel is already executing")?;
        Ok(Some(kernel.selection().clone()))
    }
    pub fn execute(
        &mut self,
        tensors: &HashMap<String, Buffer>,
        scalars: &HashMap<String, f64>,
    ) -> Result<InvocationResults, String> {
        let mut submission = self.prepare(tensors, scalars)?;
        submission.execute_sequential()?;
        Ok(submission.results_for(0)?.clone())
    }
    pub fn execute_observed(
        &mut self,
        tensors: &HashMap<String, Buffer>,
        scalars: &HashMap<String, f64>,
    ) -> Result<Vec<StepObservation>, String> {
        self.invoke(tensors, scalars, true)
    }
    fn invoke(
        &mut self,
        tensors: &HashMap<String, Buffer>,
        scalars: &HashMap<String, f64>,
        observed: bool,
    ) -> Result<Vec<StepObservation>, String> {
        let mut submission = self.prepare(tensors, scalars)?;
        if observed {
            submission.execute_steps_observed()
        } else {
            submission.execute_sequential().map(|_| Vec::new())
        }
    }
    pub fn prepare(
        &self,
        tensors: &HashMap<String, Buffer>,
        scalars: &HashMap<String, f64>,
    ) -> Result<Submission, String> {
        if tensors
            .values()
            .any(|buffer| !buffer.belongs_to(&self.device))
        {
            return Err("composition tensor belongs to another resource domain".into());
        }
        if tensors.len() != self.external.len()
            || tensors.keys().any(|n| !self.external.contains(n))
        {
            return Err("composition external bindings differ from declared inputs/outputs".into());
        }
        if scalars.len() != self.runtime_scalars.len()
            || scalars.keys().any(|n| !self.runtime_scalars.contains(n))
        {
            return Err(
                "composition runtime scalar bindings differ from declared parameters".into(),
            );
        }
        let invocation = Invocation {
            weights: &self.weights,
            scratch: &self.scratch,
            tensors,
            fixed: &self.scalars,
            scalars,
        };
        for (name, range) in &self.control_domains {
            let (dtype, elements) = self.control_types[name];
            let buffer = tensors.get(name).ok_or("unbound integer control")?;
            let mut bytes = vec![
                0;
                elements
                    .checked_mul(dtype.bytes() as usize)
                    .ok_or("control extent overflow")?
            ];
            buffer.read(&mut bytes)?;
            for bytes in bytes.chunks_exact(4) {
                let raw: [u8; 4] = bytes.try_into().expect("integer control width");
                let value = if dtype == DType::I32 {
                    i128::from(i32::from_le_bytes(raw))
                } else {
                    i128::from(u32::from_le_bytes(raw))
                };
                if !range.contains(value) {
                    return Err("invocation does not establish its integer input domain".into());
                }
            }
        }
        self.plan.prepare(&invocation)
    }
}
struct Invocation<'a> {
    weights: &'a HashMap<String, ResidentWeight>,
    scratch: &'a HashMap<String, Buffer>,
    tensors: &'a HashMap<String, Buffer>,
    fixed: &'a HashMap<String, f64>,
    scalars: &'a HashMap<String, f64>,
}
impl Bindings for Invocation<'_> {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        if let Some(weight) = self.weights.get(root) {
            return weight.plane(plane);
        }
        if !plane.is_empty() {
            return None;
        }
        self.tensors.get(root).or_else(|| self.scratch.get(root))
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        self.fixed
            .get(name)
            .or_else(|| self.scalars.get(name))
            .copied()
    }
}
