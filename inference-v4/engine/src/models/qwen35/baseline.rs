//! Full-model, single-session measurement through the ordinary selected runtime.
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
/// What selection decided for one compiled entry, in units of `estimate_model`.
#[derive(Serialize)]
pub struct EntrySelection {
    pub entry: String,
    pub status: String,
    pub estimate: u64,
    pub seed_estimate: u64,
    pub lower_bound: u64,
    pub unresolved: Vec<String>,
    /// Identity of the compiled specialization.
    pub shapes: Vec<(String, i64)>,
    pub elements: Vec<(String, String)>,
    pub strategy: String,
    pub model_variables: usize,
    pub model_factors: usize,
    /// Wall seconds per phase. `solve` = family + backend hooks + export + seed + search;
    /// `compile` = instantiate + realize + emit + native compile.
    pub seconds: PhaseSeconds,
    pub solve_seconds: f64,
    pub compile_seconds: f64,
    pub exact_phase: SolverPhase,
    pub neighborhood_phase: Option<SolverPhase>,
    pub greedy_sweeps: u32,
    pub greedy_trials: u64,
    /// The selected and the seed witness: occurrence -> candidate, site -> value,
    /// sequence -> cover.
    pub witness: WitnessRecord,
    pub seed: WitnessRecord,
}
#[derive(Serialize)]
pub struct PhaseSeconds {
    pub family: f64,
    pub backend_hooks: f64,
    pub export: f64,
    pub seed: f64,
    pub search: f64,
    pub instantiate: f64,
    pub realize: f64,
    /// `None` where the backend does not separate emission from native compilation.
    pub emit: Option<f64>,
    pub native_compile: f64,
}
#[derive(Serialize)]
pub struct SolverPhase {
    pub seconds: f64,
    pub work: u64,
    pub nodes: u64,
}
#[derive(Serialize, PartialEq)]
pub struct WitnessRecord {
    pub choices: Vec<(u32, u32)>,
    pub sites: Vec<(u32, i64)>,
    pub covers: Vec<(u32, Vec<(u32, u32)>)>,
}
impl From<&seismic_lang::family::Witness> for WitnessRecord {
    fn from(w: &seismic_lang::family::Witness) -> Self {
        Self {
            choices: w.choices.iter().map(|(o, c)| (o.0, *c)).collect(),
            sites: w.sites.iter().map(|(s, v)| (s.0, *v)).collect(),
            covers: w.covers.iter().map(|(s, c)| (s.0, c.clone())).collect(),
        }
    }
}
impl From<seismic_runtime::Selection> for EntrySelection {
    fn from(s: seismic_runtime::Selection) -> Self {
        let phase = |p: seismic_runtime::Phase| SolverPhase { seconds: p.time.as_secs_f64(), work: p.work, nodes: p.nodes };
        let t = s.timings;
        let compile = t.instantiate + t.realize + s.emit.unwrap_or_default() + s.native_compile;
        Self {
            status: format!("{:?}", s.status),
            estimate: s.estimate,
            seed_estimate: s.seed_estimate,
            lower_bound: s.lower_bound,
            strategy: format!("{:?}", s.search.strategy),
            model_variables: s.search.variables,
            model_factors: s.search.factors,
            seconds: PhaseSeconds {
                family: t.family.as_secs_f64(),
                backend_hooks: t.backend_hooks.as_secs_f64(),
                export: t.export.as_secs_f64(),
                seed: t.seed.as_secs_f64(),
                search: t.search.as_secs_f64(),
                instantiate: t.instantiate.as_secs_f64(),
                realize: t.realize.as_secs_f64(),
                emit: s.emit.map(|d| d.as_secs_f64()),
                native_compile: s.native_compile.as_secs_f64(),
            },
            solve_seconds: t.solve().as_secs_f64(),
            compile_seconds: compile.as_secs_f64(),
            exact_phase: phase(s.search.exact),
            neighborhood_phase: s.search.neighborhood.map(phase),
            greedy_sweeps: s.search.greedy_sweeps,
            greedy_trials: s.search.greedy_trials,
            witness: (&s.witness).into(),
            seed: (&s.seed).into(),
            entry: s.entry,
            unresolved: s.unresolved,
            shapes: s.shapes,
            elements: s.elements,
        }
    }
}
/// The selection records of `decoder`, all under one estimate model.
pub fn selections(decoder: &Decoder) -> Result<(String, Vec<EntrySelection>), String> {
    let selections = decoder.selections()?;
    let estimate_model = selections.first().ok_or("measured decoder retained no selection")?.estimate_model.clone();
    if selections.iter().any(|s| s.estimate_model != estimate_model) {
        return Err("decoder entries were selected under different estimate models".into());
    }
    Ok((estimate_model, selections.into_iter().map(Into::into).collect()))
}
#[derive(Serialize)]
pub struct Report {
    pub artifact: String,
    pub backend: String,
    pub estimate_model: String,
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
        let (estimate_model, selections) = selections(&self.decoder)?;
        Ok(Report {
            artifact: self.artifact.clone(),
            backend: self.backend.clone(),
            estimate_model,
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
