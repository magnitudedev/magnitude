//! Executing a composition plan on Metal: every distinct (kernel, shapes) is lowered, emitted
//! and compiled once; a step of the composition is one command buffer holding every launch
//! of every kernel call, with tensor arguments bound as buffer offsets into the caller's
//! parameters.

use crate::{execution::Config, msl};
use crate::runtime::{Buffer, Device, Pipeline};
use seismic_lang::plan::{Plan, ScalarSource, Step};
use seismic_lang::program::Program;
use seismic_lang::types::DType;
use seismic_realization::storage::plane_byte_offset;
use std::collections::HashMap;

/// A buffer of a caller parameter: dense tensors have one part, packed tensors three
/// (words, scale, bias).
pub trait Bindings {
    fn buffer(&self, root: &str, part: &str) -> Option<&Buffer>;
    fn scalar(&self, name: &str) -> Option<f64>;
}

pub struct CompiledStep {
    pub pipeline: usize,
    /// Per buffer slot of the pipeline: (root, part, byte offset).
    pub slots: Vec<(String, String, usize)>,
    /// Scalar bytes builder: (name, dtype, source).
    pub scalars: Vec<(String, DType, ScalarSource)>,
    pub kernel: String,
}

pub struct CompiledPlan {
    pub pipelines: Vec<Pipeline>,
    pub steps: Vec<CompiledStep>,
}

pub fn compile_plan(device: &Device, program: &Program, plan: &Plan, cfg: Config) -> Result<CompiledPlan, String> {
    let mut pipelines: Vec<Pipeline> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut steps = Vec::new();
    for step in &plan.steps {
        let key = step_key(step);
        let pipeline = match index.get(&key) {
            Some(i) => *i,
            None => {
                let lowered = seismic_lang::lower::lower_specialized(program, &step.kernel, "metal", &step.shapes, &step.elements, &seismic_lang::lower::Options { piece: cfg.piece })?;
                let emitted = msl::emit_with(&lowered, cfg.clone())?;
                let p = device.compile(emitted)?;
                pipelines.push(p);
                index.insert(key, pipelines.len() - 1);
                pipelines.len() - 1
            }
        };
        let emitted = &pipelines[pipeline].emitted;
        let mut slots = Vec::new();
        for slot in &emitted.buffers {
            let binding = step.tensors.iter().find(|t| t.param == slot.parameter).ok_or_else(|| format!("kernel `{}` parameter `{}` is unbound in the plan", step.kernel, slot.parameter))?;
            let offset = plane_byte_offset(&binding.elem, &slot.plane, binding.elem_offset)?;
            slots.push((binding.root.clone(), slot.plane.clone(), offset));
        }
        let mut scalars = Vec::new();
        for parameter in &emitted.scalars {
            let (name,dtype)=(&parameter.name,&parameter.dtype);
            let (_, source) = step.scalars.iter().find(|(n, _)| n == name).ok_or_else(|| format!("kernel `{}` scalar `{name}` is unbound in the plan", step.kernel))?;
            scalars.push((name.clone(), *dtype, source.clone()));
        }
        steps.push(CompiledStep { pipeline, slots, scalars, kernel: step.kernel.clone() });
    }
    Ok(CompiledPlan { pipelines, steps })
}

fn step_key(step: &Step) -> String {
    let mut shapes: Vec<(&String, &i64)> = step.shapes.iter().collect();
    shapes.sort();
    let elems: Vec<String> = step.tensors.iter().map(|t| format!("{:?}", t.elem)).collect();
    format!("{}|{:?}|{}", step.kernel, shapes, elems.join(","))
}

impl Device {
    /// Encode every step of the plan into one command buffer and wait. Returns GPU seconds.
    pub fn run_plan(&self, plan: &CompiledPlan, bindings: &dyn Bindings) -> Result<f64, String> {
        self.run_plan_steps(plan, bindings, 0..plan.steps.len())
    }

    /// Encode a range of the plan's steps into one command buffer and wait.
    pub fn run_plan_steps(&self, plan: &CompiledPlan, bindings: &dyn Bindings, range: std::ops::Range<usize>) -> Result<f64, String> {
        self.run_plan_steps_repeated(plan, bindings, range, 1)
    }

    /// Encode a range of steps `repeat` times over in one command buffer and wait. Returns
    /// GPU seconds for the whole command buffer.
    pub fn run_plan_steps_repeated(&self, plan: &CompiledPlan, bindings: &dyn Bindings, range: std::ops::Range<usize>, repeat: usize) -> Result<f64, String> {
        let steps = plan.steps.get(range).ok_or("Metal plan step range is out of bounds")?;
        let prepared = steps.iter().map(|step| {
            let pipeline = plan.pipelines.get(step.pipeline).ok_or("invalid Metal plan pipeline index")?;
            if step.slots.len() != pipeline.emitted.buffers.len() { return Err("invalid plan binding count".to_string()); }
            let buffers = step.slots.iter().zip(&pipeline.emitted.buffers).map(|((root,part,offset),slot)| {
                let buffer = bindings.buffer(root,part).ok_or_else(||format!("tensor `{root}` ({part}) is unbound"))?;
                let end = offset.checked_add(slot.bytes).ok_or("Metal plan view overflow")?;
                buffer.view(*offset..end)
            }).collect::<Result<Vec<_>,String>>()?;
            let values = step.scalars.iter().map(|(_,_,source)| match source {
                ScalarSource::Literal(x)=>Ok(*x),
                ScalarSource::Param(p)=>bindings.scalar(p).ok_or_else(||format!("scalar `{p}` is unbound")),
            }).collect::<Result<Vec<_>,String>>()?;
            Ok((pipeline,buffers,pipeline.emitted.encode_scalars(&values)?))
        }).collect::<Result<Vec<_>,String>>()?;
        let invocations = prepared.iter().map(|(pipeline,buffers,scalars)| crate::runtime::Invocation {
            pipeline,buffers:buffers.iter().collect(),scalars:scalars.clone(),
        }).collect::<Vec<_>>();
        self.run_many(&invocations,repeat)
    }
}
