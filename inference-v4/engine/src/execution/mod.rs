//! Resident bindings for checked numerical compositions. Model policy chooses
//! parameters; this owner validates their contracts and retains scratch/code.
use crate::weights::residency::ResidentWeight;
use seismic_lang::types::{DType, Elem, Ty};
use seismic_runtime::{
    plan::{Bindings, CompiledPlan, PlanCompiler, StepObservation, Submission},
    Buffer, Device, Error,
};
use std::collections::{HashMap, HashSet};

pub struct Composition {
    device: Device,
    spec: CompositionSpec,
    controls: HashSet<String>,
    control_types: HashMap<String, (DType, usize)>,
    control_domains: HashMap<String, seismic_runtime::tuner::IntegerRange>,
    plan: CompiledPlan,
    weights: HashMap<String, ResidentWeight>,
    scratch: HashMap<String, Buffer>,
    external: HashSet<String>,
    runtime_scalars: HashSet<String>,
    scalars: HashMap<String, f64>,
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
        let function = compiler
            .program()
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or("unknown composition")?;
        for (name, weight) in &weights {
            if !weight.belongs_to(compiler.device()) {
                return Err(format!("weight {name} belongs to another resource domain").into());
            }
            let (_, Ty::Tensor(t)) = function
                .params
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| format!("unknown weight {name}"))?
            else {
                return Err(format!("weight {name} is not a tensor").into());
            };
            let expected = t
                .shape
                .iter()
                .map(|s| {
                    s.eval(&|p| shapes.get(p).copied())
                        .and_then(|v| u64::try_from(v).ok())
                        .ok_or("unresolved weight shape")
                })
                .collect::<Result<Vec<_>, _>>()?;
            if expected != weight.descriptor().shape {
                return Err(format!("weight {name} shape differs from composition").into());
            }
            match &t.elem {
                Elem::Param(p) => {
                    if let Some(previous) = elements.insert(p.clone(), weight.element().clone()) {
                        if &previous != weight.element() {
                            return Err(format!(
                                "weight {name} disagrees on element parameter {p}"
                            ).into());
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
            if weights.contains_key(name)
                || !function
                    .params
                    .iter()
                    .any(|(n, t)| n == name && matches!(t, Ty::Tensor(_)))
            {
                return Err(format!("invalid external tensor {name}").into());
            }
        }
        for name in &intermediates {
            if weights.contains_key(name)
                || external.contains(name)
                || !function
                    .params
                    .iter()
                    .any(|(n, t)| n == name && matches!(t, Ty::Tensor(_)))
            {
                return Err(format!("invalid intermediate tensor {name}").into());
            }
        }
        for (name, ty) in &function.params {
            if matches!(ty, Ty::Tensor(_))
                && !weights.contains_key(name)
                && !external.contains(name)
                && !intermediates.contains(name)
            {
                return Err(format!("composition tensor {name} has no declared owner").into());
            }
        }
        for name in scalars.keys() {
            if !function
                .params
                .iter()
                .any(|(n, t)| n == name && matches!(t, Ty::Scalar(_)))
            {
                return Err(format!("invalid scalar {name}").into());
            }
        }
        let runtime_scalars = function
            .params
            .iter()
            .filter_map(|(name, ty)| {
                (matches!(ty, Ty::Scalar(_)) && !scalars.contains_key(name)).then_some(name.clone())
            })
            .collect::<HashSet<_>>();
        let ownership = seismic_lang::composition::Ownership {
            intermediates: intermediates.iter().cloned().collect(),
        };
        let mut control_types = HashMap::new();
        for (name, ty) in &function.params {
            if !external.contains(name) { continue; }
            let Ty::Tensor(tensor) = ty else { continue; };
            let element = match &tensor.elem {
                Elem::Param(parameter) => elements.get(parameter).ok_or("unbound control dtype")?,
                element => element,
            };
            let Elem::Dtype(dtype @ (DType::I32 | DType::U32)) = element else { continue; };
            let elements = tensor.shape.iter().try_fold(1usize, |n, shape| {
                shape.eval(&|name| shapes.get(name).copied()).and_then(|extent| usize::try_from(extent).ok())
                    .and_then(|extent| n.checked_mul(extent)).ok_or("control tensor extent overflow")
            })?;
            control_types.insert(name.clone(), (*dtype, elements));
        }
        let mut scratch = HashMap::new();
        for (name, ty) in &function.params {
            if weights.contains_key(name) || external.contains(name) {
                continue;
            }
            if let Ty::Tensor(t) = ty {
                let element = match &t.elem {
                    Elem::Param(p) => elements.get(p).ok_or("unbound scratch dtype")?,
                    e => e,
                };
                let Elem::Dtype(dtype) = element else {
                    return Err("composition scratch must be dense".into());
                };
                let bytes = t.shape.iter().try_fold(dtype.bytes() as usize, |n, s| {
                    s.eval(&|p| shapes.get(p).copied())
                        .and_then(|v| usize::try_from(v).ok())
                        .and_then(|d| n.checked_mul(d))
                        .ok_or("scratch allocation overflow or unresolved shape")
                })?;
                scratch.insert(name.clone(), compiler.device().buffer(bytes)?);
            }
        }
        Ok(Self {
            device: compiler.device().clone(),
            spec: retained,
            controls: HashSet::new(),
            control_types,
            control_domains: HashMap::new(),
            plan: compiler.compile_entry(&entry, &shapes, &elements, &ownership)?,
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
            spec.shapes.insert(name.into(), i64::try_from(extent)
                .map_err(|_| "specialized dimension overflow")?);
        }
        let mut composition = Self::compile(compiler, spec)?;
        composition.controls = self.controls.clone();
        composition.control_domains = self.control_domains.clone();
        Ok(composition)
    }
    pub(crate) fn control_inputs(mut self, names: &[&str]) -> Result<Self, String> {
        if names.iter().any(|name| !self.external.contains(*name)) { return Err("control input is not an external tensor".into()); }
        self.controls.extend(names.iter().map(|name| name.to_string()));
        Ok(self)
    }
    /// Admit changing integer controls only under one checked finite domain.
    /// Selection must account for the whole range before the code is reusable.
    pub(crate) fn control_domain(mut self, name: &str, range: seismic_runtime::tuner::IntegerRange) -> Result<Self, String> {
        if !self.control_types.contains_key(name) {
            return Err("varying control domain requires an external integer tensor".into());
        }
        self.controls.insert(name.into());
        self.control_domains.insert(name.into(), range);
        Ok(self)
    }
    pub(crate) fn shares_compilation(&self, other: &Self) -> bool { self.plan.shares_compilation(&other.plan) }
    pub fn kernel_count(&self) -> usize {
        self.plan.kernel_count()
    }
    pub fn execute(
        &mut self,
        tensors: &HashMap<String, Buffer>,
        scalars: &HashMap<String, f64>,
    ) -> Result<(), String> {
        self.invoke(tensors, scalars, false).map(|_| ())
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
        if tensors.values().any(|buffer| !buffer.belongs_to(&self.device)) {
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
            generalize_controls: self.plan.supports_integer_domains(),
            controls: &self.controls,
            control_types: &self.control_types,
            control_domains: &self.control_domains,
            weights: &self.weights,
            scratch: &self.scratch,
            tensors,
            fixed: &self.scalars,
            scalars,
        };
        if !invocation.generalize_controls {
            // Preserve input bounds on forms that specialize exact control
            // bytes. Their lack of domain analysis never removes a condition.
            for (name, range) in &self.control_domains {
                let (dtype, elements) = self.control_types[name];
                let buffer = tensors.get(name).ok_or("unbound integer control")?;
                let mut bytes = vec![0; elements.checked_mul(dtype.bytes() as usize).ok_or("control extent overflow")?];
                buffer.read(&mut bytes)?;
                for bytes in bytes.chunks_exact(4) {
                    let raw: [u8; 4] = bytes.try_into().expect("integer control width");
                    let value = if dtype == DType::I32 { i128::from(i32::from_le_bytes(raw)) }
                        else { i128::from(u32::from_le_bytes(raw)) };
                    if !range.contains(value) { return Err("invocation does not establish its integer input domain".into()); }
                }
            }
        }
        self.plan.prepare(&invocation)
    }
}
struct Invocation<'a> {
    generalize_controls: bool,
    controls: &'a HashSet<String>,
    control_types: &'a HashMap<String, (DType, usize)>,
    control_domains: &'a HashMap<String, seismic_runtime::tuner::IntegerRange>,
    weights: &'a HashMap<String, ResidentWeight>,
    scratch: &'a HashMap<String, Buffer>,
    tensors: &'a HashMap<String, Buffer>,
    fixed: &'a HashMap<String, f64>,
    scalars: &'a HashMap<String, f64>,
}
impl Bindings for Invocation<'_> {
    fn known_buffer(&self, root: &str, plane: &str) -> bool { plane.is_empty() && self.controls.contains(root) }
    fn buffer_domains(&self, root: &str, plane: &str) -> Vec<seismic_runtime::tuner::BufferIntegerDomain> {
        if !self.generalize_controls || !plane.is_empty() { return Vec::new(); }
        let Some(&range) = self.control_domains.get(root) else { return Vec::new(); };
        let (dtype, elements) = self.control_types[root];
        let width = dtype.bytes() as usize;
        (0..elements).map(|index| seismic_runtime::tuner::BufferIntegerDomain {
            offset: (index * width) as u64,
            bytes: width as u8,
            signed: dtype == DType::I32,
            range,
        }).collect()
    }
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
