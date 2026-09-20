//! Full-model, single-session measurement through the unified runtime pipeline.
//! The caller supplies only the search budget; this module cannot choose an
//! implementation. Selection estimates are reported as estimates, never as timings.
use super::{decoder::Decoder, loading::Model};
use seismic_runtime::{plan::Settings, Device};
use serde::Serialize;
use std::{path::Path, rc::Rc, time::Instant};

pub struct Baseline {
    decoder: Decoder,
    artifact: String,
    backend: String,
    context_capacity: usize,
    load_seconds: f64,
}
#[derive(Serialize)]
pub struct Sample {
    pub prefill_seconds: f64,
    pub decode_seconds: Vec<f64>,
    /// GPU command-buffer interval of the batched submission of each forward.
    pub prefill_device_seconds: Option<f64>,
    pub decode_device_seconds: Vec<Option<f64>>,
    /// Includes final-logit readback and commit. Warm samples must create no code.
    pub new_kernels: usize,
    pub prefill_logits: Vec<f32>,
    pub final_logits: Vec<f32>,
}
#[derive(Serialize)]
pub struct Component {
    pub stage: String,
    pub block: Option<usize>,
    pub entry: String,
    pub host_seconds: f64,
    pub device_seconds: Option<f64>,
    /// Native stage interval of every dispatch of this entry, in launch order.
    pub dispatches: Vec<Dispatch>,
}
#[derive(Serialize)]
pub struct Dispatch {
    pub launch: usize,
    pub kernel: String,
    pub threadgroups: u64,
    pub threads_per_threadgroup: u64,
    pub device_seconds: f64,
}
/// The resolved physical plan retained by one compiled entry.
#[derive(Serialize)]
pub struct EntrySelection {
    pub entry: String,
    pub assignment: Vec<Decision>,
    pub estimated_cost: i64,
    pub optimal: bool,
    pub resources: Vec<LaunchResource>,
    pub capability_fingerprint: String,
    pub numerical_evidence: String,
    pub numerical_evidence_identity: Option<String>,
    pub numerical_reasons: Vec<String>,
    /// Identity of the compiled specialization.
    pub shapes: Vec<(String, i64)>,
    pub elements: Vec<(String, String)>,
    pub compile_seconds: f64,
}
#[derive(Serialize)]
pub struct Decision {
    pub choice: u32,
    pub logical_alternative: u32,
    pub physical_alternative: u32,
}
#[derive(Serialize)]
pub struct LaunchResource {
    pub launch: u64,
    pub workgroups: [u64; 3],
    pub device_bytes: u64,
    pub workgroup_bytes: u64,
    pub private_bytes_per_participant: u64,
    pub bindings: u64,
    pub threads_per_group: u64,
}
impl From<seismic_runtime::Selection> for EntrySelection {
    fn from(s: seismic_runtime::Selection) -> Self {
        Self {
            assignment: s
                .assignment
                .selections()
                .iter()
                .map(|(choice, selection)| Decision {
                    choice: choice.0,
                    logical_alternative: selection.logical_alternative,
                    physical_alternative: selection.physical_alternative,
                })
                .collect(),
            estimated_cost: s.estimated_cost,
            optimal: s.optimal,
            resources: s
                .resources
                .into_iter()
                .map(|resource| LaunchResource {
                    launch: resource.launch.0,
                    workgroups: resource.workgroups,
                    device_bytes: resource.device_bytes,
                    workgroup_bytes: resource.workgroup_bytes,
                    private_bytes_per_participant: resource.private_bytes_per_participant,
                    bindings: resource.bindings,
                    threads_per_group: resource.threads_per_group,
                })
                .collect(),
            capability_fingerprint: s.capability_fingerprint,
            numerical_evidence: format!("{:?}", s.numerical_assessment.evidence),
            numerical_evidence_identity: s.numerical_evidence_identity,
            numerical_reasons: s.numerical_assessment.reasons,
            compile_seconds: s.compile.as_secs_f64(),
            entry: s.entry,
            shapes: s.shapes,
            elements: s.elements,
        }
    }
}
pub fn selections(decoder: &Decoder) -> Result<Vec<EntrySelection>, String> {
    Ok(decoder.selections()?.into_iter().map(Into::into).collect())
}
#[derive(Serialize)]
pub struct Report {
    pub artifact: String,
    pub backend: String,
    pub selections: Vec<EntrySelection>,
    pub context_capacity: usize,
    pub prompt: Vec<u32>,
    pub continuation: Vec<u32>,
    pub load_seconds: f64,
    pub cold: Sample,
    pub warm: Vec<Sample>,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    /// A separate instrumented forward, excluded from throughput samples.
    pub prefill_components: Vec<Component>,
    pub decode_components: Vec<Vec<Component>>,
}
impl Baseline {
    pub fn load(
        path: impl AsRef<Path>,
        device: Rc<Device>,
        settings: Settings,
        context_capacity: usize,
    ) -> Result<Self, String> {
        let start = Instant::now();
        let backend = device.backend().to_string();
        let model = Model::open(path)?;
        let artifact = model.description().artifact_identity.to_string();
        let decoder = model.load(device, settings, context_capacity, 1)?;
        Ok(Self {
            decoder,
            artifact,
            backend,
            context_capacity,
            load_seconds: start.elapsed().as_secs_f64(),
        })
    }
    fn sample(&mut self, prompt: &[u32], continuation: &[u32]) -> Result<Sample, String> {
        let mut state = self.decoder.state_store().create()?;
        let before = self.decoder.compiled_kernel_count();
        let start = Instant::now();
        let (advance, observation) = self
            .decoder
            .prefill_batched(&mut state, prompt)
            .map_err(|e| format!("prefill: {e}"))?;
        let prefill_device_seconds = observation.device_seconds;
        let prefill_logits = advance.logits().to_vec();
        advance.commit()?;
        let prefill_seconds = start.elapsed().as_secs_f64();
        let mut final_logits = prefill_logits.clone();
        let mut decode_seconds = Vec::new();
        let mut decode_device_seconds = Vec::new();
        for &token in continuation {
            let start = Instant::now();
            let (advance, observation) = self
                .decoder
                .propose_batched(&mut state, token)
                .map_err(|e| format!("decode: {e}"))?;
            final_logits = advance.logits().to_vec();
            advance.commit()?;
            decode_seconds.push(start.elapsed().as_secs_f64());
            decode_device_seconds.push(observation.device_seconds);
            if final_logits.iter().any(|v| !v.is_finite()) {
                return Err("non-finite decode logits".into());
            }
        }
        if prefill_logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite prefill logits".into());
        }
        Ok(Sample {
            prefill_seconds,
            decode_seconds,
            prefill_device_seconds,
            decode_device_seconds,
            new_kernels: self.decoder.compiled_kernel_count() - before,
            prefill_logits,
            final_logits,
        })
    }
    /// One cold trial, one excluded warm-up, then three measured trials. Forced
    /// continuations keep the workload identical even when engines' logits differ.
    pub fn measure(&mut self, prompt: &[u32], continuation: &[u32]) -> Result<Report, String> {
        if prompt.is_empty()
            || continuation.is_empty()
            || prompt
                .len()
                .checked_add(continuation.len())
                .is_none_or(|n| n > self.context_capacity)
        {
            return Err(
                "baseline requires nonempty prompt/continuation within context capacity".into(),
            );
        }
        if prompt
            .iter()
            .chain(continuation)
            .any(|&t| u64::from(t) >= self.decoder.geometry().vocabulary)
        {
            return Err("baseline token outside vocabulary".into());
        }
        let cold = self.sample(prompt, continuation)?;
        self.sample(prompt, continuation)?;
        let mut warm = Vec::new();
        for _ in 0..3 {
            let sample = self.sample(prompt, continuation)?;
            if sample.new_kernels != 0 {
                return Err("warm forward compiled new kernels; refusing to label compilation as steady-state throughput".into());
            }
            warm.push(sample);
        }
        let prefill_tokens_per_second = (prompt.len() * warm.len()) as f64
            / warm.iter().map(|s| s.prefill_seconds).sum::<f64>();
        let decode_tokens_per_second = (continuation.len() * warm.len()) as f64
            / warm.iter().flat_map(|s| &s.decode_seconds).sum::<f64>();
        let components = |steps: Vec<super::decoder::DecoderStepObservation>| {
            steps
                .into_iter()
                .map(|s| Component {
                    stage: s.stage,
                    block: s.block,
                    entry: s.step.entry,
                    host_seconds: s.step.execution.host_seconds,
                    device_seconds: s.step.execution.device_seconds,
                    dispatches: s
                        .step
                        .dispatches
                        .into_iter()
                        .map(|d| Dispatch {
                            launch: d.launch,
                            kernel: d.kernel,
                            threadgroups: d.threadgroups,
                            threads_per_threadgroup: d.threads_per_threadgroup,
                            device_seconds: d.device_seconds,
                        })
                        .collect(),
                })
                .collect::<Vec<_>>()
        };
        let mut state = self.decoder.state_store().create()?;
        let (advance, observations) = self.decoder.prefill_observed(&mut state, prompt)?;
        advance.commit()?;
        let prefill_components = components(observations);
        let mut decode_components = Vec::new();
        for &token in continuation.iter().take(4) {
            let (advance, observations) = self.decoder.propose_observed(&mut state, token)?;
            advance.commit()?;
            decode_components.push(components(observations));
        }
        let selections = selections(&self.decoder)?;
        Ok(Report {
            artifact: self.artifact.clone(),
            backend: self.backend.clone(),
            selections,
            context_capacity: self.context_capacity,
            prompt: prompt.to_vec(),
            continuation: continuation.to_vec(),
            load_seconds: self.load_seconds,
            cold,
            warm,
            prefill_tokens_per_second,
            decode_tokens_per_second,
            prefill_components,
            decode_components,
        })
    }
}
