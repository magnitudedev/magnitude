//! Full-model, single-session measurement through the ordinary selected runtime.
//! Hardware settings are supplied by the caller; this module cannot choose an
//! implementation or manufacture a hardware timing model.
use super::{decoder::Decoder, loading::Model};
use seismic_runtime::{plan::Settings, Device};
use serde::Serialize;
use std::{path::Path, rc::Rc, time::Instant};

pub struct Baseline {
    decoder: Decoder,
    artifact: String,
    backend: String,
    hardware: String,
    context_capacity: usize,
    load_seconds: f64,
}
#[derive(Serialize)]
pub struct Sample {
    pub prefill_seconds: f64,
    pub decode_seconds: Vec<f64>,
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
}
#[derive(Serialize)]
pub struct Report {
    pub artifact: String,
    pub backend: String,
    pub hardware: String,
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
        let hardware = match &settings.hardware {
            seismic_runtime::tuner::Hardware::Cpu(h) => h.identity.clone(),
            seismic_runtime::tuner::Hardware::Cuda(h) => h.identity.clone(),
            #[cfg(target_os = "macos")]
            seismic_runtime::tuner::Hardware::Metal(h) => h.identity.clone(),
        };
        let backend = device.backend().to_string();
        let model = Model::open(path)?;
        let artifact = model.description().artifact_identity.to_string();
        let decoder = model.load(device, settings, context_capacity, 1)?;
        Ok(Self {
            decoder,
            artifact,
            backend,
            hardware,
            context_capacity,
            load_seconds: start.elapsed().as_secs_f64(),
        })
    }
    fn sample(&mut self, prompt: &[u32], continuation: &[u32]) -> Result<Sample, String> {
        let mut state = self.decoder.state_store().create()?;
        let before = self.decoder.compiled_kernel_count();
        let start = Instant::now();
        let (advance, _) = self
            .decoder
            .prefill_batched(&mut state, prompt)
            .map_err(|e| format!("prefill: {e}"))?;
        let prefill_logits = advance.logits().to_vec();
        advance.commit()?;
        let prefill_seconds = start.elapsed().as_secs_f64();
        let mut final_logits = prefill_logits.clone();
        let mut decode_seconds = Vec::new();
        for &token in continuation {
            let start = Instant::now();
            let (advance, _) = self
                .decoder
                .propose_batched(&mut state, token)
                .map_err(|e| format!("decode: {e}"))?;
            final_logits = advance.logits().to_vec();
            advance.commit()?;
            decode_seconds.push(start.elapsed().as_secs_f64());
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
        Ok(Report {
            artifact: self.artifact.clone(),
            backend: self.backend.clone(),
            hardware: self.hardware.clone(),
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
