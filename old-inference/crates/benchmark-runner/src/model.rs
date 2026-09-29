use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SUITE_VERSION: &str = "4";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetKind {
    Generic,
    Icn,
    LlamaCpp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetConfig {
    pub name: String,
    pub endpoint: String,
    pub model: String,
    pub kind: TargetKind,
    #[serde(skip)]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_id: Option<u32>,
    #[serde(default)]
    pub configuration: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRunConfig {
    pub root: PathBuf,
    pub profile: String,
    pub targets: Vec<TargetConfig>,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub controlled_host: bool,
    #[serde(default)]
    pub exclusive_device: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub id: String,
    pub description: String,
    pub cases: Vec<String>,
    pub repetitions: usize,
    pub warmups: usize,
    #[serde(default)]
    pub concurrency: Vec<usize>,
    #[serde(default)]
    pub closed_loop_multiplier: usize,
    pub request_timeout_seconds: u64,
    #[serde(default)]
    pub controlled: bool,
    #[serde(default)]
    pub require_exclusive_device: bool,
    #[serde(default)]
    pub max_repetitions: usize,
    #[serde(default)]
    pub confidence_half_width_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureCatalog {
    pub schema_version: u32,
    pub suite_version: String,
    pub prompt_short_tokens: usize,
    pub prompt_long_tokens: usize,
    pub output_short_tokens: u32,
    pub output_long_tokens: u32,
    pub carrier_block: String,
    pub answer_short: String,
    pub answer_long: String,
    pub tool_fixture: ToolFixture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolFixture {
    pub path: String,
    pub before: String,
    pub old_text: String,
    pub new_text: String,
    pub tool_name: String,
    pub tool_result: String,
    pub final_acknowledgement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Valid,
    Invalid,
    Unsupported,
    Error,
    Cancelled,
    Unstable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostEvidence {
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub controlled_host: bool,
    pub exclusive_device: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetProbe {
    pub health: Option<Value>,
    pub models: Option<Value>,
    pub properties: Option<Value>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryEvidence {
    pub source: Option<String>,
    pub baseline_bytes: Option<u64>,
    pub peak_bytes: Option<u64>,
    pub retained_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageObservation {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingObservation {
    pub cache_n: Option<u64>,
    pub prompt_n: Option<u64>,
    pub prompt_ms: Option<f64>,
    pub prompt_per_second: Option<f64>,
    pub predicted_n: Option<u64>,
    pub predicted_ms: Option<f64>,
    pub predicted_per_second: Option<f64>,
    pub sampler_ms: Option<f64>,
    pub parser_ms: Option<f64>,
    pub draft_n: Option<u64>,
    pub draft_n_accepted: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawStreamEvent {
    pub elapsed_ms: f64,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallObservation {
    pub index: u64,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestObservation {
    pub id: String,
    pub outcome: RunOutcome,
    pub status: Option<u16>,
    pub headers_ms: Option<f64>,
    pub first_event_ms: Option<f64>,
    pub first_semantic_ms: Option<f64>,
    pub completed_ms: f64,
    pub finish_reason: Option<String>,
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCallObservation>,
    pub usage: Option<UsageObservation>,
    pub timings: Option<TimingObservation>,
    pub output_sha256: String,
    pub raw_events: Vec<RawStreamEvent>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkSignature {
    pub request_id: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    pub output_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSample {
    pub repetition: usize,
    pub arm: String,
    pub outcome: RunOutcome,
    pub makespan_ms: f64,
    pub metrics: BTreeMap<String, f64>,
    pub requests: Vec<RequestObservation>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub samples: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub median_absolute_deviation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub id: String,
    pub title: String,
    pub question: String,
    pub outcome: RunOutcome,
    pub samples: Vec<ScenarioSample>,
    pub summaries: BTreeMap<String, MetricSummary>,
    pub work_signatures: Vec<WorkSignature>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetResult {
    pub target: TargetConfig,
    pub probe: TargetProbe,
    pub experiments: Vec<ExperimentResult>,
    pub memory: MemoryEvidence,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub schema_version: u32,
    pub suite_version: String,
    pub run_id: String,
    pub recorded_at_unix_ms: u128,
    pub profile: Profile,
    pub fixtures_sha256: String,
    pub host: HostEvidence,
    pub targets: Vec<TargetResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub experiment: String,
    pub metric: String,
    pub candidate_median: f64,
    pub reference_median: f64,
    pub candidate_reference_ratio: f64,
    pub confidence_low: Option<f64>,
    pub confidence_high: Option<f64>,
    pub stable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub schema_version: u32,
    pub run_id: String,
    pub candidate: String,
    pub reference: String,
    pub outcome: RunOutcome,
    pub semantic_match: bool,
    pub work_match: bool,
    pub metrics: Vec<MetricComparison>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    RunStarted {
        run_id: String,
    },
    TargetStarted {
        target: String,
    },
    ExperimentStarted {
        target: String,
        experiment: String,
    },
    SampleCompleted {
        target: String,
        experiment: String,
        arm: String,
        repetition: usize,
    },
    TargetCompleted {
        target: String,
    },
    RunCompleted {
        run_id: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum BenchmarkError {
    #[error("invalid benchmark configuration: {0}")]
    InvalidConfig(String),
    #[error("benchmark asset error: {0}")]
    Asset(String),
    #[error("endpoint error: {0}")]
    Endpoint(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}
