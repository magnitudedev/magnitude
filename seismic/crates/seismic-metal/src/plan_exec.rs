//! Executing a composition plan on Metal: every distinct (kernel, shapes) is lowered, emitted
//! and compiled once; a step of the composition is one command buffer holding every launch
//! of every kernel call, with tensor arguments bound as buffer offsets into the caller's
//! parameters.

use crate::msl::{self, Config};
use crate::runtime::{Buffer, Device, Pipeline};
use objc2_metal::{MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize};
use seismic_lang::plan::{Plan, ScalarSource, Step};
use seismic_lang::program::Program;
use seismic_lang::repr;
use seismic_lang::types::{DType, Elem};
use std::collections::HashMap;
use std::ptr::NonNull;

/// A buffer of a caller parameter: dense tensors have one part, packed tensors three
/// (words, scale, bias).
pub trait Bindings {
    fn buffer(&self, root: &str, part: &str) -> Option<&Buffer>;
    fn scalar(&self, name: &str) -> Option<f64>;
}

pub struct CompiledStep {
    pub pipeline: usize,
    /// Per buffer slot of the pipeline: (root, part, byte offset).
    pub slots: Vec<(String, &'static str, usize)>,
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
                let lowered = seismic_lang::lower::lower_with(program, &step.kernel, "metal", &step.shapes, &seismic_lang::lower::Options { piece: cfg.piece })?;
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
            let binding = step.tensors.iter().find(|t| t.param == slot.param).ok_or_else(|| format!("kernel `{}` parameter `{}` is unbound in the plan", step.kernel, slot.param))?;
            let offset = byte_offset(&binding.elem, slot.part, binding.elem_offset)?;
            slots.push((binding.root.clone(), slot.part, offset));
        }
        let mut scalars = Vec::new();
        for (name, dtype) in &emitted.scalars {
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

fn byte_offset(elem: &Elem, part: &str, elem_offset: i64) -> Result<usize, String> {
    let e = elem_offset as usize;
    match elem {
        Elem::Dtype(d) => Ok(e * d.bytes() as usize),
        Elem::Repr(r) => {
            let rep = repr::lookup(r).unwrap();
            let cpw = rep.codes_per_word() as usize;
            let group = rep.group as usize;
            match part {
                "words" => {
                    if e % cpw != 0 {
                        return Err(format!("view offset {e} is not word-aligned for `{r}`"));
                    }
                    Ok(e / cpw * 4)
                }
                _ => {
                    if e % group != 0 {
                        return Err(format!("view offset {e} is not group-aligned for `{r}`"));
                    }
                    Ok(e / group * rep.coefficient.bytes() as usize)
                }
            }
        }
        Elem::Param(p) => Err(format!("generic element `{p}` in a plan")),
    }
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
        let command = self.queue().commandBuffer().ok_or("could not create a command buffer")?;
        // One compute encoder for the whole range: Metal orders dispatches within an encoder
        // by their resource hazards, and an encoder costs far more than a dispatch.
        let encoder = command.computeCommandEncoder().ok_or("could not create a compute encoder")?;
        let steps = &plan.steps[range];
        let mut first_step = true;
        for step in steps.iter().cycle().take(steps.len() * repeat) {
            // Each step consumes what earlier steps wrote, so steps do not overlap.
            if !first_step {
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
            first_step = false;
            let pipeline = &plan.pipelines[step.pipeline];
            let mut scalar_bytes = Vec::new();
            for (name, dtype, source) in &step.scalars {
                let v = match source {
                    ScalarSource::Literal(x) => *x,
                    ScalarSource::Param(p) => bindings.scalar(p).ok_or_else(|| format!("scalar `{p}` is unbound"))?,
                };
                match dtype {
                    DType::I32 => scalar_bytes.extend_from_slice(&(v as i32).to_le_bytes()),
                    DType::U32 => scalar_bytes.extend_from_slice(&(v as u32).to_le_bytes()),
                    _ => scalar_bytes.extend_from_slice(&(v as f32).to_le_bytes()),
                }
                let _ = name;
            }
            for (launch, state) in pipeline.states() {
                // A launch that consumes an earlier one's stores must not overlap it.
                if launch.after_barrier {
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                }
                encoder.setComputePipelineState(state);
                for (i, (root, part, offset)) in step.slots.iter().enumerate() {
                    let b = bindings.buffer(root, part).ok_or_else(|| format!("tensor `{root}` ({part}) is unbound"))?;
                    unsafe { encoder.setBuffer_offset_atIndex(Some(b.raw()), *offset, i) };
                }
                for (i, b) in pipeline.scratch_buffers().iter().enumerate() {
                    unsafe { encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, step.slots.len() + i) };
                }
                let scalar_slot = step.slots.len() + pipeline.scratch_buffers().len();
                if !scalar_bytes.is_empty() {
                    unsafe { encoder.setBytes_length_atIndex(NonNull::new(scalar_bytes.as_ptr() as *mut _).unwrap(), scalar_bytes.len(), scalar_slot) };
                }
                let grid = MTLSize { width: launch.threadgroups as usize, height: 1, depth: 1 };
                let group = MTLSize { width: launch.threads_per_threadgroup as usize, height: 1, depth: 1 };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
            }
        }
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(e) = command.error() {
            return Err(format!("command buffer failed: {}", e.localizedDescription()));
        }
        Ok(command.GPUEndTime() - command.GPUStartTime())
    }
}
