//! One long-context session through the ordinary selected runtime: a prose payload
//! prefilled in fixed chunks, then greedy decode. Mirrors the V3 session-bench prose
//! workload (`benchmark_fixtures/prose.py`, serving `--prefill-tokens`). Every forward
//! records whether it compiled anything; only forwards that compiled nothing are warm.
use super::{baseline::{selections, EntrySelection}, decoder::Decoder, loading::Model};
use crate::inputs::{ByteBpeTokenizer, SpecialTokens, TokenId};
use seismic_runtime::{plan::Settings, Device};
use serde::Serialize;
use std::{path::Path, rc::Rc, time::Instant};

/// Context positions left unused after the payload and the decoded tokens.
pub const MARGIN: usize = 16;

/// `prose.normalize`: UTF-8 without BOM, LF line ends, the body between the Gutenberg
/// START and END marker lines, stripped.
pub fn normalize(content: &[u8]) -> Result<String, String> {
    let text = std::str::from_utf8(content).map_err(|e| format!("fixture: {e}"))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text).replace("\r\n", "\n").replace('\r', "\n");
    let marker = |which: &str| {
        let prefix = format!("*** {which} OF THE PROJECT GUTENBERG EBOOK ");
        let mut offset = 0;
        text.split_inclusive('\n').find_map(|line| {
            let start = offset;
            offset += line.len();
            (line.starts_with(&prefix) && line.trim_end().ends_with("***") && line.trim_end().len() >= prefix.len() + 3).then_some((start, offset))
        })
    };
    match (marker("START"), marker("END")) {
        (Some((_, body)), Some((end, _))) if body < end => Ok(text[body..end].trim().to_string()),
        _ => Err("missing or unordered Gutenberg body markers".into()),
    }
}

#[derive(Serialize)]
pub struct Forward {
    pub rows: usize,
    /// Context position of the first row.
    pub position: usize,
    pub seconds: f64,
    pub device_seconds: Option<f64>,
    /// Kernels selected and compiled inside this forward; nonzero means cold.
    pub new_kernels: usize,
}
#[derive(Serialize)]
pub struct Rate {
    pub forwards: usize,
    pub tokens: usize,
    pub seconds: f64,
    pub tokens_per_second: f64,
}
#[derive(Serialize)]
pub struct Latency {
    pub steps: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
}
#[derive(Serialize)]
pub struct Report {
    pub artifact: String,
    pub backend: String,
    pub estimate_model: String,
    pub context_capacity: usize,
    pub fixture_text_sha256: String,
    pub payload_tokens: usize,
    pub prefill_chunk: usize,
    pub decode_steps: usize,
    pub load_seconds: f64,
    /// Payload start to the first generated token, compilation included.
    pub cold_seconds_to_first_token: f64,
    /// All chunks (compilation included) and the chunks that compiled nothing.
    pub prefill_cold: Rate,
    pub prefill_warm: Rate,
    /// Decode steps that compiled nothing; attention cost grows with history.
    pub decode_warm: Rate,
    pub decode_latency: Latency,
    pub decode_first_32: Latency,
    pub decode_last_32: Latency,
    pub solve_seconds_total: f64,
    pub compile_seconds_total: f64,
    /// One record per compiled `(entry, shapes, elements)` specialization.
    pub selections: Vec<EntrySelection>,
    pub prefill: Vec<Forward>,
    pub decode: Vec<Forward>,
    /// The first token follows the payload; each decode step appends one.
    pub tokens: Vec<u32>,
    pub text: String,
}

pub struct Session {
    decoder: Decoder,
    tokenizer: ByteBpeTokenizer,
    artifact: String,
    backend: String,
    context_capacity: usize,
    load_seconds: f64,
}

fn rate<'a>(forwards: impl Iterator<Item = &'a Forward>) -> Rate {
    let (mut count, mut tokens, mut seconds) = (0, 0, 0.0);
    for forward in forwards {
        count += 1;
        tokens += forward.rows;
        seconds += forward.seconds;
    }
    Rate { forwards: count, tokens, seconds, tokens_per_second: tokens as f64 / seconds }
}

/// Nearest-rank percentiles over warm steps.
fn latency(steps: &[&Forward]) -> Latency {
    let mut ms: Vec<f64> = steps.iter().map(|f| f.seconds * 1e3).collect();
    ms.sort_by(f64::total_cmp);
    let rank = |p: f64| ms.get(((p * ms.len() as f64).ceil() as usize).saturating_sub(1)).copied().unwrap_or(f64::NAN);
    Latency { steps: ms.len(), mean_ms: ms.iter().sum::<f64>() / ms.len() as f64, p50_ms: rank(0.5), p90_ms: rank(0.9) }
}

fn argmax(logits: &[f32]) -> Result<u32, String> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite logits".into());
    }
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).ok_or_else(|| "empty logits".into())
}

impl Session {
    pub fn load(path: impl AsRef<Path>, device: Rc<Device>, settings: Settings, context_capacity: usize) -> Result<Self, String> {
        let start = Instant::now();
        let backend = device.backend().to_string();
        let model = Model::open(path)?;
        let artifact = model.description().artifact_identity.to_string();
        let tokenizer = ByteBpeTokenizer::new(model.tokenizer_config()?)?;
        let decoder = model.load(device, settings, context_capacity, 1)?;
        Ok(Self { decoder, tokenizer, artifact, backend, context_capacity, load_seconds: start.elapsed().as_secs_f64() })
    }

    /// Prefill the first `context_capacity - decode - MARGIN` tokens of the normalized
    /// fixture in `chunk`-row forwards, then decode `decode` tokens greedily.
    pub fn measure(&mut self, fixture: &[u8], chunk: usize, decode: usize) -> Result<Report, String> {
        use sha2::{Digest, Sha256};
        let text = normalize(fixture)?;
        let fixture_text_sha256 = Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        let budget = self.context_capacity.checked_sub(decode + MARGIN).filter(|n| *n > 0 && chunk > 0 && decode > 0).ok_or("context must exceed decode + margin; chunk and decode must be positive")?;
        // A prefix long enough for the payload: prose is well under 8 bytes per token.
        let mut bytes = text.len().min(budget * 8);
        while !text.is_char_boundary(bytes) {
            bytes -= 1;
        }
        let payload: Vec<u32> = self.tokenizer.encode(&text[..bytes], SpecialTokens::Literal)?.into_iter().map(|t| t.0).take(budget).collect();
        if payload.len() != budget {
            return Err(format!("fixture yields {} tokens, fewer than the {budget}-token payload", payload.len()));
        }

        let mut state = self.decoder.state_store().create()?;
        let (mut prefill, mut steps) = (Vec::new(), Vec::new());
        let mut logits = Vec::new();
        let started = Instant::now();
        for rows in payload.chunks(chunk) {
            let (before, position, start) = (self.decoder.compiled_kernel_count(), state.position(), Instant::now());
            let (advance, observation) = self.decoder.prefill_batched(&mut state, rows).map_err(|e| format!("prefill at {position}: {e}"))?;
            logits = advance.logits().to_vec();
            advance.commit()?;
            prefill.push(Forward { rows: rows.len(), position, seconds: start.elapsed().as_secs_f64(), device_seconds: observation.device_seconds, new_kernels: self.decoder.compiled_kernel_count() - before });
            eprintln!("prefill {}/{} ({:.2} s)", position + rows.len(), payload.len(), prefill.last().map_or(0.0, |f| f.seconds));
        }
        let mut tokens = vec![argmax(&logits)?];
        let cold_seconds_to_first_token = started.elapsed().as_secs_f64();
        for _ in 0..decode {
            let (before, position, start) = (self.decoder.compiled_kernel_count(), state.position(), Instant::now());
            let token = *tokens.last().ok_or("no token to continue from")?;
            let (advance, observation) = self.decoder.propose_batched(&mut state, token).map_err(|e| format!("decode at {position}: {e}"))?;
            let next = argmax(advance.logits())?;
            advance.commit()?;
            steps.push(Forward { rows: 1, position, seconds: start.elapsed().as_secs_f64(), device_seconds: observation.device_seconds, new_kernels: self.decoder.compiled_kernel_count() - before });
            tokens.push(next);
        }

        fn warm(forwards: &[Forward]) -> Vec<&Forward> {
            forwards.iter().filter(|f| f.new_kernels == 0).collect()
        }
        let (warm_prefill, warm_steps) = (warm(&prefill), warm(&steps));
        if warm_prefill.is_empty() || warm_steps.is_empty() {
            return Err("no forward ran without compiling; refusing to report compilation as throughput".into());
        }
        let window = 32.min(warm_steps.len());
        let mut detokenizer = self.tokenizer.decoder(true);
        let mut decoded = String::new();
        for token in &tokens {
            decoded.push_str(&detokenizer.push(TokenId(*token))?);
        }
        decoded.push_str(&detokenizer.finish()?);
        let (estimate_model, selections) = selections(&self.decoder)?;
        Ok(Report {
            artifact: self.artifact.clone(),
            backend: self.backend.clone(),
            estimate_model,
            context_capacity: self.context_capacity,
            fixture_text_sha256,
            payload_tokens: payload.len(),
            prefill_chunk: chunk,
            decode_steps: decode,
            load_seconds: self.load_seconds,
            cold_seconds_to_first_token,
            prefill_cold: rate(prefill.iter()),
            prefill_warm: rate(warm_prefill.iter().copied()),
            decode_warm: rate(warm_steps.iter().copied()),
            decode_latency: latency(&warm_steps),
            decode_first_32: latency(&warm_steps[..window]),
            decode_last_32: latency(&warm_steps[warm_steps.len() - window..]),
            solve_seconds_total: selections.iter().map(|s| s.solve_seconds).sum(),
            compile_seconds_total: selections.iter().map(|s| s.compile_seconds).sum(),
            selections,
            prefill,
            decode: steps,
            tokens,
            text: decoded,
        })
    }
}
