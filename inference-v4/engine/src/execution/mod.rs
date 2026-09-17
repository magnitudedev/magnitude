//! Resident bindings for checked numerical compositions. Model policy chooses
//! parameters; this owner validates their contracts and retains scratch/code.
use crate::weights::residency::ResidentWeight;
use seismic_lang::{
    plan::plan_specialized,
    types::{Elem, Ty},
};
use seismic_runtime::{
    plan::{Bindings, CompiledPlan, PlanCompiler, StepObservation, Submission},
    Buffer,
};
use std::collections::{HashMap, HashSet};

pub struct Composition {
    plan: CompiledPlan,
    weights: HashMap<String, ResidentWeight>,
    scratch: HashMap<String, Buffer>,
    external: HashSet<String>,
    runtime_scalars: HashSet<String>,
    scalars: HashMap<String, f64>,
}
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
    pub fn compile(compiler: &mut PlanCompiler<'_>, spec: CompositionSpec) -> Result<Self, String> {
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
            let (_, Ty::Tensor(t)) = function
                .params
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| format!("unknown weight {name}"))?
            else {
                return Err(format!("weight {name} is not a tensor"));
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
                return Err(format!("weight {name} shape differs from composition"));
            }
            match &t.elem {
                Elem::Param(p) => {
                    if let Some(previous) = elements.insert(p.clone(), weight.element().clone()) {
                        if &previous != weight.element() {
                            return Err(format!(
                                "weight {name} disagrees on element parameter {p}"
                            ));
                        }
                    }
                }
                concrete if concrete != weight.element() => {
                    return Err(format!("weight {name} has wrong element type"))
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
                return Err(format!("invalid external tensor {name}"));
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
                return Err(format!("invalid intermediate tensor {name}"));
            }
        }
        for (name, ty) in &function.params {
            if matches!(ty, Ty::Tensor(_))
                && !weights.contains_key(name)
                && !external.contains(name)
                && !intermediates.contains(name)
            {
                return Err(format!("composition tensor {name} has no declared owner"));
            }
        }
        for name in scalars.keys() {
            if !function
                .params
                .iter()
                .any(|(n, t)| n == name && matches!(t, Ty::Scalar(_)))
            {
                return Err(format!("invalid scalar {name}"));
            }
        }
        let runtime_scalars = function
            .params
            .iter()
            .filter_map(|(name, ty)| {
                (matches!(ty, Ty::Scalar(_)) && !scalars.contains_key(name)).then_some(name.clone())
            })
            .collect::<HashSet<_>>();
        let plan = plan_specialized(compiler.program(), &entry, &shapes, &elements)?;
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
            plan: compiler.compile(&plan)?,
            weights,
            scratch,
            external,
            runtime_scalars,
            scalars,
        })
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
