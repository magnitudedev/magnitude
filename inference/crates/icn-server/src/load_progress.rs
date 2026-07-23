use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use icn_contracts::models::{
    ModelFileRole, ModelOfferingTarget, ModelPackage, ModelServingConfiguration,
};
use icn_engine::{ModelLoadObserver, ModelLoadPhase};
use icn_models::ModelCache;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const MAX_SIGNATURES: usize = 256;
const MAX_SAMPLES: usize = 12;
const MAX_ENVIRONMENTS: usize = 8;

#[derive(Clone, Debug)]
pub struct PhaseEstimate {
    pub phase: ModelLoadPhase,
    pub work: f64,
    pub predicted: Duration,
}

#[derive(Clone)]
pub struct LoadProgressTracker {
    inner: std::sync::Arc<Mutex<TrackerState>>,
}

struct TrackerState {
    estimates: Vec<PhaseEstimate>,
    active: Option<(usize, Instant)>,
    completed_prediction_ms: f64,
    observations: Vec<PhaseObservation>,
    last_fraction: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PhaseObservation {
    phase: ModelLoadPhase,
    work: f64,
    duration_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct TimingStore {
    #[serde(default)]
    signatures: BTreeMap<String, SignatureHistory>,
    #[serde(default)]
    environment: BTreeMap<String, BTreeMap<ModelLoadPhase, Vec<PhaseObservation>>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct SignatureHistory {
    #[serde(default)]
    phases: BTreeMap<ModelLoadPhase, Vec<f64>>,
    touched_at: u64,
}

impl LoadProgressTracker {
    pub fn new(estimates: Vec<PhaseEstimate>) -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(TrackerState {
                estimates,
                active: None,
                completed_prediction_ms: 0.0,
                observations: Vec::new(),
                last_fraction: 0.0,
            })),
        }
    }

    pub fn fraction(&self) -> f32 {
        let Ok(mut state) = self.inner.lock() else {
            return 0.0;
        };
        let total_ms = state
            .estimates
            .iter()
            .map(|estimate| estimate.predicted.as_secs_f64() * 1000.0)
            .sum::<f64>()
            .max(1.0);
        let active_ms = state.active.map_or(0.0, |(index, started)| {
            let predicted_ms = state.estimates[index]
                .predicted
                .as_secs_f64()
                .mul_add(1000.0, 0.0)
                .max(1.0);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let local = if elapsed_ms <= predicted_ms {
                0.98 * elapsed_ms / predicted_ms
            } else {
                0.98 + 0.019 * (1.0 - (-(elapsed_ms - predicted_ms) / predicted_ms).exp())
            };
            predicted_ms * local
        });
        let fraction =
            (0.99 * (state.completed_prediction_ms + active_ms) / total_ms).clamp(0.0, 0.99) as f32;
        state.last_fraction = state.last_fraction.max(fraction);
        state.last_fraction
    }

    fn observations(&self) -> Vec<PhaseObservation> {
        self.inner
            .lock()
            .map(|state| state.observations.clone())
            .unwrap_or_default()
    }
}

impl ModelLoadObserver for LoadProgressTracker {
    fn phase_started(&self, phase: ModelLoadPhase) {
        let Ok(mut state) = self.inner.lock() else {
            return;
        };
        let Some(index) = state
            .estimates
            .iter()
            .position(|estimate| estimate.phase == phase)
        else {
            return;
        };
        state.active = Some((index, Instant::now()));
        tracing::debug!(phase = ?phase, "model load phase started");
    }

    fn phase_completed(&self, phase: ModelLoadPhase) {
        let Ok(mut state) = self.inner.lock() else {
            return;
        };
        let Some((index, started)) = state.active.take() else {
            return;
        };
        if state.estimates[index].phase != phase {
            return;
        }
        let estimate = state.estimates[index].clone();
        state.completed_prediction_ms += estimate.predicted.as_secs_f64() * 1000.0;
        state.observations.push(PhaseObservation {
            phase,
            work: estimate.work,
            duration_ms: started.elapsed().as_secs_f64() * 1000.0,
        });
        tracing::debug!(phase = ?phase, "model load phase completed");
    }
}

pub struct LoadProgressEstimator {
    cache: ModelCache,
    native_build: String,
}

impl LoadProgressEstimator {
    pub fn new(cache: ModelCache, native_build: String) -> Self {
        Self {
            cache,
            native_build,
        }
    }

    pub fn signature(
        &self,
        configuration: &ModelServingConfiguration,
        acceleration: &str,
        timing_plan_identity: &str,
        phases: &[ModelLoadPhase],
        previously_loaded_in_process: bool,
    ) -> String {
        #[derive(Serialize)]
        struct Evidence<'a> {
            native_build: &'a str,
            acceleration: &'a str,
            timing_plan_identity: &'a str,
            profile: &'a icn_contracts::models::ServingProfile,
            files: Vec<(&'a str, &'a ModelFileRole, u64)>,
            phases: &'a [ModelLoadPhase],
            previously_loaded_in_process: bool,
        }
        let files = packages(configuration)
            .into_iter()
            .flat_map(|package| package.files.iter())
            .map(|file| (file.sha256.as_str(), &file.role, file.size_bytes))
            .collect();
        let bytes = serde_json::to_vec(&Evidence {
            native_build: &self.native_build,
            acceleration,
            timing_plan_identity,
            profile: &configuration.profile,
            files,
            phases,
            previously_loaded_in_process,
        })
        .expect("load timing evidence is serializable");
        format!("{:x}", Sha256::digest(bytes))
    }

    pub fn estimate(
        &self,
        signature: &str,
        configuration: &ModelServingConfiguration,
        acceleration: &str,
        phases: &[ModelLoadPhase],
    ) -> Vec<PhaseEstimate> {
        let store = self.read_store();
        let environment = store.environment.get(&self.environment_key(acceleration));
        let work = phase_work(configuration);
        phases
            .iter()
            .copied()
            .map(|phase| {
                let phase_work = work.get(&phase).copied().unwrap_or(1.0).max(1.0);
                let fallback = environment_prediction(environment, phase, phase_work)
                    .unwrap_or_else(|| bundled_prior(phase, phase_work));
                let exact = store
                    .signatures
                    .get(signature)
                    .and_then(|history| history.phases.get(&phase));
                let predicted_ms =
                    exact
                        .filter(|samples| !samples.is_empty())
                        .map_or(fallback, |samples| {
                            let weight = (samples.len() as f64 / 3.0).min(1.0);
                            p60(samples) * weight + fallback * (1.0 - weight)
                        });
                PhaseEstimate {
                    phase,
                    work: phase_work,
                    predicted: Duration::from_secs_f64((predicted_ms / 1000.0).max(0.001)),
                }
            })
            .collect()
    }

    pub fn record_success(
        &self,
        signature: &str,
        acceleration: &str,
        tracker: &LoadProgressTracker,
    ) {
        let observations = tracker.observations();
        if observations.is_empty() {
            return;
        }
        let mut store = self.read_store();
        let touched_at = unix_time_seconds();
        let environment_key = self.environment_key(acceleration);
        let history = store.signatures.entry(signature.to_owned()).or_default();
        history.touched_at = touched_at;
        for observation in &observations {
            push_bounded(
                history.phases.entry(observation.phase).or_default(),
                observation.duration_ms,
            );
            let environment = store
                .environment
                .entry(environment_key.clone())
                .or_default()
                .entry(observation.phase)
                .or_default();
            environment.push(observation.clone());
            if environment.len() > MAX_SIGNATURES * MAX_SAMPLES {
                environment.drain(..environment.len() - MAX_SIGNATURES * MAX_SAMPLES);
            }
        }
        while store.signatures.len() > MAX_SIGNATURES {
            let Some(oldest) = store
                .signatures
                .iter()
                .min_by_key(|(_, history)| history.touched_at)
                .map(|(signature, _)| signature.clone())
            else {
                break;
            };
            store.signatures.remove(&oldest);
        }
        while store.environment.len() > MAX_ENVIRONMENTS {
            let Some(stale) = store
                .environment
                .keys()
                .find(|candidate| *candidate != &environment_key)
                .cloned()
            else {
                break;
            };
            store.environment.remove(&stale);
        }
        self.cache.write_load_timing_store(&store);
    }

    fn read_store(&self) -> TimingStore {
        let Some(Value::Object(mut root)) = self.cache.read_load_timing_store::<Value>() else {
            return TimingStore::default();
        };
        TimingStore {
            signatures: recover_signatures(root.remove("signatures")),
            environment: recover_environments(root.remove("environment")),
        }
    }

    fn environment_key(&self, acceleration: &str) -> String {
        format!(
            "{:x}",
            Sha256::digest(format!("{}\0{acceleration}", self.native_build).as_bytes())
        )
    }
}

fn recover_signatures(value: Option<Value>) -> BTreeMap<String, SignatureHistory> {
    let Some(Value::Object(entries)) = value else {
        return BTreeMap::new();
    };
    entries
        .into_iter()
        .filter(|(signature, _)| valid_cache_identity(signature))
        .filter_map(|(signature, value)| {
            let Value::Object(mut history) = value else {
                return None;
            };
            let phases = recover_phase_durations(history.remove("phases"));
            (!phases.is_empty()).then(|| {
                let touched_at = history
                    .remove("touched_at")
                    .and_then(|value| serde_json::from_value(value).ok())
                    .unwrap_or_default();
                (signature, SignatureHistory { phases, touched_at })
            })
        })
        .take(MAX_SIGNATURES)
        .collect()
}

fn recover_phase_durations(value: Option<Value>) -> BTreeMap<ModelLoadPhase, Vec<f64>> {
    let Some(Value::Object(entries)) = value else {
        return BTreeMap::new();
    };
    entries
        .into_iter()
        .filter_map(|(phase, value)| {
            let phase = decode_phase(phase)?;
            let Value::Array(samples) = value else {
                return None;
            };
            let samples = retain_recent(samples, MAX_SAMPLES, |value| {
                serde_json::from_value::<f64>(value)
                    .ok()
                    .filter(|sample| sample.is_finite() && *sample > 0.0)
            });
            (!samples.is_empty()).then_some((phase, samples))
        })
        .collect()
}

fn recover_environments(
    value: Option<Value>,
) -> BTreeMap<String, BTreeMap<ModelLoadPhase, Vec<PhaseObservation>>> {
    let Some(Value::Object(environments)) = value else {
        return BTreeMap::new();
    };
    environments
        .into_iter()
        .filter(|(environment, _)| valid_cache_identity(environment))
        .filter_map(|(environment, value)| {
            let Value::Object(phases) = value else {
                return None;
            };
            let phases = phases
                .into_iter()
                .filter_map(|(phase, value)| {
                    let phase = decode_phase(phase)?;
                    let Value::Array(samples) = value else {
                        return None;
                    };
                    let samples = retain_recent(samples, MAX_SIGNATURES * MAX_SAMPLES, |value| {
                        serde_json::from_value::<PhaseObservation>(value)
                            .ok()
                            .filter(|sample| {
                                sample.phase == phase
                                    && sample.work.is_finite()
                                    && sample.work > 0.0
                                    && sample.duration_ms.is_finite()
                                    && sample.duration_ms > 0.0
                            })
                    });
                    (!samples.is_empty()).then_some((phase, samples))
                })
                .collect::<BTreeMap<_, _>>();
            (!phases.is_empty()).then_some((environment, phases))
        })
        .take(MAX_ENVIRONMENTS)
        .collect()
}

fn valid_cache_identity(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_phase(value: String) -> Option<ModelLoadPhase> {
    serde_json::from_value(Value::String(value)).ok()
}

fn retain_recent<T>(
    values: Vec<Value>,
    maximum: usize,
    decode: impl Fn(Value) -> Option<T>,
) -> Vec<T> {
    let mut recovered = values
        .into_iter()
        .rev()
        .filter_map(decode)
        .take(maximum)
        .collect::<Vec<_>>();
    recovered.reverse();
    recovered
}

fn packages(configuration: &ModelServingConfiguration) -> Vec<&ModelPackage> {
    match &configuration.target {
        ModelOfferingTarget::Package { package } => vec![package],
        ModelOfferingTarget::SpeculativeDecodingPair { target, draft, .. } => {
            vec![target, draft]
        }
    }
}

fn phase_work(configuration: &ModelServingConfiguration) -> BTreeMap<ModelLoadPhase, f64> {
    let mut result = BTreeMap::new();
    let (target, draft) = match &configuration.target {
        ModelOfferingTarget::Package { package } => (package, None),
        ModelOfferingTarget::SpeculativeDecodingPair { target, draft, .. } => (target, Some(draft)),
    };
    let target_bytes = target
        .files
        .iter()
        .filter(|file| file.role == ModelFileRole::Weights)
        .map(|file| file.size_bytes)
        .sum::<u64>();
    let embedded_mtp_bytes = target
        .files
        .iter()
        .filter(|file| file.role == ModelFileRole::Mtp)
        .map(|file| file.size_bytes)
        .sum::<u64>();
    let draft_bytes = draft.map_or(embedded_mtp_bytes, |package| {
        package
            .files
            .iter()
            .filter(|file| matches!(file.role, ModelFileRole::Weights | ModelFileRole::Mtp))
            .map(|file| file.size_bytes)
            .sum()
    });
    let projector_bytes = target
        .files
        .iter()
        .filter(|file| file.role == ModelFileRole::Projector)
        .map(|file| file.size_bytes)
        .sum::<u64>();
    let context_work = f64::from(configuration.profile.context_length)
        * f64::from(configuration.profile.parallel_sequences);
    result.insert(ModelLoadPhase::TargetModel, target_bytes as f64);
    result.insert(ModelLoadPhase::TargetContext, context_work);
    result.insert(ModelLoadPhase::DraftModel, draft_bytes as f64);
    result.insert(ModelLoadPhase::DraftContext, context_work);
    result.insert(ModelLoadPhase::Projector, projector_bytes as f64);
    result.insert(
        ModelLoadPhase::Runtime,
        f64::from(configuration.profile.parallel_sequences),
    );
    result.insert(ModelLoadPhase::Warmup, target_bytes as f64);
    result.insert(ModelLoadPhase::Finalize, 1.0);
    result
}

fn environment_prediction(
    environment: Option<&BTreeMap<ModelLoadPhase, Vec<PhaseObservation>>>,
    phase: ModelLoadPhase,
    work: f64,
) -> Option<f64> {
    let samples = environment?.get(&phase)?;
    let scaled = samples
        .iter()
        .rev()
        .take(MAX_SIGNATURES)
        .map(|sample| {
            let ratio = if matches!(phase, ModelLoadPhase::Runtime | ModelLoadPhase::Finalize) {
                1.0
            } else {
                (work / sample.work.max(1.0)).clamp(0.25, 4.0)
            };
            sample.duration_ms * ratio
        })
        .collect::<Vec<_>>();
    (!scaled.is_empty()).then(|| p60(&scaled))
}

fn bundled_prior(phase: ModelLoadPhase, work: f64) -> f64 {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    match phase {
        ModelLoadPhase::TargetModel | ModelLoadPhase::DraftModel => 350.0 + 160.0 * work / GIB,
        ModelLoadPhase::TargetContext | ModelLoadPhase::DraftContext => 200.0 + 0.025 * work,
        ModelLoadPhase::Projector => 200.0 + 160.0 * work / GIB,
        ModelLoadPhase::Runtime => 120.0 + 20.0 * work,
        ModelLoadPhase::Warmup => 450.0 + 25.0 * work / GIB,
        ModelLoadPhase::Finalize => 120.0,
    }
}

fn p60(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * 0.6).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len().saturating_sub(1));
    sorted[index]
}

fn push_bounded(values: &mut Vec<f64>, value: f64) {
    values.push(value);
    if values.len() > MAX_SAMPLES {
        values.drain(..values.len() - MAX_SAMPLES);
    }
}

fn unix_time_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_never_reaches_ready() {
        let tracker = LoadProgressTracker::new(vec![PhaseEstimate {
            phase: ModelLoadPhase::Finalize,
            work: 1.0,
            predicted: Duration::from_millis(1),
        }]);
        tracker.phase_started(ModelLoadPhase::Finalize);
        std::thread::sleep(Duration::from_millis(3));
        assert!(tracker.fraction() < 1.0);
        tracker.phase_completed(ModelLoadPhase::Finalize);
        assert_eq!(tracker.fraction(), 0.99);
    }

    #[test]
    fn exact_history_blends_in_over_three_samples() {
        let fallback = 1_000.0;
        let samples = vec![2_000.0, 2_000.0, 2_000.0];
        let weight = (samples.len() as f64 / 3.0).min(1.0);
        assert_eq!(p60(&samples) * weight + fallback * (1.0 - weight), 2_000.0);
    }
}
