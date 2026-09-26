//! Behavioral reasoning-control discovery using the owned native template.
use magnitude_templates::{Request, Template};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

pub const DETECTOR: &str = "v4-reasoning-1";
pub const CONTROLS: &[&str] = &[
    "enable_thinking",
    "thinking",
    "thinking_mode",
    "reasoning_effort",
];
const EFFORTS: &[(&str, &[&str])] = &[
    ("minimal", &["minimal"]),
    ("low", &["low"]),
    ("medium", &["medium"]),
    ("high", &["high"]),
    ("xhigh", &["xhigh", "extra_high", "extra-high", "very_high"]),
    ("max", &["max"]),
];
type Controls = BTreeMap<String, Value>;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffortMapping {
    pub effort: String,
    pub controls: Controls,
    pub aliases: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortDomain {
    Closed,
    SharedFallback,
    OpenOrIgnored,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningProfile {
    pub detector: String,
    pub template_identity: String,
    pub option_context: String,
    pub default_effort: Option<String>,
    pub mappings: Vec<EffortMapping>,
    pub baseline_shapes: Vec<bool>,
    pub effort_domain: EffortDomain,
    pub supports_reasoning_output: Option<bool>,
    pub supports_preserve_reasoning: bool,
}
/// Normalize a caller's reasoning-effort spelling to the public vocabulary.
/// Native disabling spellings become `none`; extra-high spellings become `xhigh`.
pub fn normalize_effort(value: &str) -> Option<&'static str> {
    Some(match value {
        "none" | "off" | "no_think" | "disabled" => "none",
        "xhigh" | "extra_high" | "extra-high" | "very_high" => "xhigh",
        "minimal" => "minimal",
        "low" => "low",
        "medium" => "medium",
        "high" => "high",
        "max" => "max",
        "adaptive" => "adaptive",
        _ => return None,
    })
}

/// Ordinal rank of an enabled effort; `none` and named modes have no rank.
fn rank(effort: &str) -> Option<usize> {
    EFFORTS.iter().position(|(name, _)| *name == effort)
}

/// What the caller asked of reasoning, before it meets a model's profile.
/// Budgets are hard engine caps on reasoning tokens; they never derive from
/// an effort.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningIntent {
    /// The template's own default behavior.
    #[default]
    ModelDefault,
    Disabled,
    /// Reasoning on, at the model's enabled default.
    Enabled,
    /// A normalized effort (see [`normalize_effort`]).
    Effort { effort: String },
}

/// A request's reasoning after resolution against the model profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedReasoning {
    /// The normalized effort whose recipe renders the request; `None` keeps the
    /// template default without inspecting it.
    pub effort: Option<String>,
    /// Template arguments that realize the effort.
    pub controls: Controls,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReasoningError {
    /// The model cannot run the requested effort.
    UnsupportedEffort { effort: String, supported: Vec<String> },
    /// The model has no way to disable reasoning.
    CannotDisable { supported: Vec<String> },
    /// The model has no enabled reasoning behavior.
    NoEnabledEffort { supported: Vec<String> },
    /// A requested effort is outside the public vocabulary.
    UnknownEffort(String),
}

impl std::fmt::Display for ReasoningError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedEffort { effort, supported } => write!(
                formatter,
                "reasoning effort {effort} is unsupported for this model; supported values: {}",
                supported.join(", ")
            ),
            Self::CannotDisable { supported } => write!(
                formatter,
                "reasoning cannot be disabled for this model; supported values: {}",
                supported.join(", ")
            ),
            Self::NoEnabledEffort { supported } => write!(
                formatter,
                "reasoning cannot be enabled for this model; supported values: {}",
                supported.join(", ")
            ),
            Self::UnknownEffort(effort) => {
                write!(formatter, "unsupported reasoning_effort spelling: {effort}")
            }
        }
    }
}

impl std::error::Error for ReasoningError {}

impl ReasoningProfile {
    fn supported(&self) -> Vec<String> {
        self.mappings.iter().map(|m| m.effort.clone()).collect()
    }

    fn mapping(&self, effort: &str) -> Option<&EffortMapping> {
        self.mappings
            .iter()
            .find(|m| m.effort == effort || m.aliases.iter().any(|a| a == effort))
    }

    /// The enabled behavior used when a caller enables reasoning without
    /// naming a supported level: the default when it is enabled, otherwise
    /// the first enabled mapping.
    fn enabled_default(&self) -> Option<&EffortMapping> {
        self.default_effort
            .as_deref()
            .filter(|effort| *effort != "none")
            .and_then(|effort| self.mapping(effort))
            .or_else(|| self.mappings.iter().find(|m| m.effort != "none"))
    }

    /// Resolve an effort the model does not declare exactly: an ordinal rounds
    /// up to the nearest declared enabled level and clamps to the highest; a
    /// named mode falls back to the enabled default; `none` is an error.
    fn reconcile(&self, effort: &str) -> Result<&EffortMapping, ReasoningError> {
        if let Some(mapping) = self.mapping(effort) {
            return Ok(mapping);
        }
        if effort == "none" {
            return Err(ReasoningError::CannotDisable {
                supported: self.supported(),
            });
        }
        let ranked = || {
            self.mappings
                .iter()
                .filter_map(|m| rank(&m.effort).map(|r| (r, m)))
        };
        let rounded = rank(effort).and_then(|requested| {
            ranked()
                .filter(|(r, _)| *r >= requested)
                .min_by_key(|(r, _)| *r)
                .or_else(|| ranked().max_by_key(|(r, _)| *r))
                .map(|(_, m)| m)
        });
        rounded
            .or_else(|| self.enabled_default())
            .ok_or_else(|| ReasoningError::UnsupportedEffort {
                effort: effort.into(),
                supported: self.supported(),
            })
    }

    pub fn resolve(&self, intent: &ReasoningIntent) -> Result<ResolvedReasoning, ReasoningError> {
        let mapping = match intent {
            ReasoningIntent::ModelDefault => {
                return Ok(ResolvedReasoning {
                    effort: self.default_effort.clone(),
                    controls: Controls::new(),
                })
            }
            ReasoningIntent::Disabled => {
                self.mapping("none")
                    .ok_or_else(|| ReasoningError::CannotDisable {
                        supported: self.supported(),
                    })?
            }
            ReasoningIntent::Enabled => {
                self.enabled_default()
                    .ok_or_else(|| ReasoningError::NoEnabledEffort {
                        supported: self.supported(),
                    })?
            }
            ReasoningIntent::Effort { effort } => {
                let normalized = normalize_effort(effort)
                    .ok_or_else(|| ReasoningError::UnknownEffort(effort.clone()))?;
                self.reconcile(normalized)?
            }
        };
        Ok(ResolvedReasoning {
            effort: Some(mapping.effort.clone()),
            controls: mapping.controls.clone(),
        })
    }
    pub fn fingerprint(&self) -> String {
        Sha256::digest(serde_json::to_vec(self).expect("serializable reasoning profile"))
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// Preparation-owner cache. Profiles contain data only; native template handles
/// are never retained. Source/options/build identity come from Template::identity.
pub struct ProfileCache {
    entries: VecDeque<(Arc<ReasoningProfile>, usize)>,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    hits: u64,
}
impl ProfileCache {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_bytes,
            bytes: 0,
            hits: 0,
        }
    }
    pub fn cached_entries(&self) -> usize {
        self.entries.len()
    }
    pub fn cached_bytes(&self) -> usize {
        self.bytes
    }
    pub fn hits(&self) -> u64 {
        self.hits
    }
    pub fn inspect(
        &mut self,
        template: &Template,
        arguments: &Map<String, Value>,
    ) -> Result<Arc<ReasoningProfile>, String> {
        let context = option_context(arguments)?;
        if let Some(index) = self.entries.iter().position(|(p, _)| {
            p.template_identity == template.identity()
                && p.option_context == context
                && p.detector == DETECTOR
        }) {
            let entry = self.entries.remove(index).unwrap();
            let profile = entry.0.clone();
            self.entries.push_back(entry);
            self.hits += 1;
            return Ok(profile);
        }
        let profile = Arc::new(inspect_reasoning(template, arguments)?);
        let charge = serde_json::to_vec(profile.as_ref())
            .map_err(|e| e.to_string())?
            .len();
        if self.max_entries > 0 && charge <= self.max_bytes {
            while self.entries.len() >= self.max_entries || self.bytes > self.max_bytes - charge {
                self.bytes -= self.entries.pop_front().unwrap().1;
            }
            self.bytes += charge;
            self.entries.push_back((profile.clone(), charge));
        }
        Ok(profile)
    }
}
fn option_context(arguments: &Map<String, Value>) -> Result<String, String> {
    if CONTROLS.iter().any(|key| arguments.contains_key(*key)) {
        return Err("reasoning probe context must omit reasoning controls".into());
    }
    serde_json::to_string(&canonical(&Value::Object(arguments.clone()))).map_err(|e| e.to_string())
}
#[derive(Clone, PartialEq, Eq)]
struct Signature {
    prompt: String,
    prefix: String,
    parser: String,
    grammar: String,
    thinking: bool,
    start: String,
    ends: Vec<String>,
}
type Outcomes = Vec<Option<Signature>>;
type Observed = (EffortMapping, Outcomes);
#[derive(Clone, Copy)]
struct EffortPolicy {
    rejected: bool,
    fallback: bool,
}
fn comparable(base: &Outcomes, candidate: &Outcomes) -> bool {
    base.iter().any(Option::is_some)
        && base
            .iter()
            .zip(candidate)
            .all(|(a, b)| a.is_none() || b.is_some())
}
fn names(outcomes: &Outcomes, effort: &str) -> bool {
    let Some((_, spellings)) = EFFORTS.iter().find(|(name, _)| *name == effort) else {
        return false;
    };
    outcomes.iter().flatten().any(|s| {
        let text = format!("{}\n{}", s.prompt, s.prefix).to_lowercase();
        spellings.iter().any(|name| {
            text.split(|c: char| {
                !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
            })
            .any(|word| word == *name)
        })
    })
}
fn observed(effort: &str, controls: Controls, outcomes: Outcomes) -> Observed {
    (
        EffortMapping {
            effort: effort.into(),
            controls,
            aliases: vec![],
        },
        outcomes,
    )
}
fn control(name: &str, value: Value) -> Controls {
    BTreeMap::from([(name.into(), value)])
}
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), canonical(v)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        _ => value.clone(),
    }
}
struct Probe<'a> {
    template: &'a Template,
    arguments: &'a Map<String, Value>,
    shapes: Vec<Request>,
    cache: BTreeMap<String, Outcomes>,
    first_error: Option<String>,
}
impl Probe<'_> {
    fn render(&mut self, controls: &Controls) -> Outcomes {
        let key = serde_json::to_string(controls).unwrap();
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        let outcomes = self
            .shapes
            .iter()
            .map(|shape| {
                let mut request = shape.clone();
                request.template_arguments = self.arguments.clone();
                request.template_arguments.extend(controls.clone());
                match self.template.prepare(&request) {
                    Ok(plan) => {
                        let d = plan.description();
                        Some(Signature {
                            prompt: d.prompt.clone(),
                            prefix: d.generation_prefix.clone(),
                            parser: d.parser.clone(),
                            grammar: d.grammar.clone(),
                            thinking: d.supports_thinking,
                            start: d.thinking_start.clone(),
                            ends: d.thinking_ends.clone(),
                        })
                    }
                    Err(error) => {
                        self.first_error.get_or_insert_with(|| error.to_string());
                        None
                    }
                }
            })
            .collect::<Outcomes>();
        self.cache.insert(key, outcomes.clone());
        outcomes
    }
    fn effort(
        &mut self,
        effort: &str,
        spellings: &[&str],
        enabled: &Controls,
        base: &Outcomes,
        invalid: &Outcomes,
        policy: EffortPolicy,
    ) -> Result<Option<Observed>, String> {
        let mut selected: Option<Observed> = None;
        for spelling in spellings {
            let mut controls = enabled.clone();
            controls.insert("reasoning_effort".into(), json!(spelling));
            let outcomes = self.render(&controls);
            if !comparable(base, &outcomes)
                || (!policy.rejected
                    && (!policy.fallback || (outcomes == *invalid && !names(&outcomes, effort))))
            {
                continue;
            }
            if let Some((_, previous)) = &selected {
                if *previous != outcomes {
                    return Err(format!("reasoning aliases for {effort} render differently"));
                }
            } else {
                selected = Some(observed(effort, controls, outcomes));
            }
        }
        Ok(selected)
    }
}
fn shapes() -> Vec<Request> {
    let tool = json!({"type":"function","function":{"name":"weather","description":"Get the current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}});
    let question = json!({"role":"user","content":"What is the weather in Paris?"});
    let mut requests = vec![
        Request::new(
            vec![json!({"role":"user","content":"Explain why the sky appears blue."})],
            946684800,
        ),
        Request::new(vec![question.clone()], 946684800),
        Request::new(
            vec![
                question,
                json!({"role":"assistant","content":null,"reasoning_content":"I should check the weather tool.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]}),
                json!({"role":"tool","content":"18 C and clear","name":"weather","tool_call_id":"call_1"}),
            ],
            946684800,
        ),
    ];
    requests[1].tools = vec![tool.clone()];
    requests[2].tools = vec![tool];
    requests
}

/// A fixed probe set bounds preparation work. Omitted request controls do not
/// invoke discovery or insert the informational default into the actual prompt.
pub fn inspect_reasoning(
    template: &Template,
    arguments: &Map<String, Value>,
) -> Result<ReasoningProfile, String> {
    let option_context = option_context(arguments)?;
    let mut probe = Probe {
        template,
        arguments,
        shapes: shapes(),
        cache: BTreeMap::new(),
        first_error: None,
    };
    let baseline = probe.render(&Controls::new());
    if !baseline.iter().any(Option::is_some) {
        return Err(format!(
            "reasoning inspection rejected every conversation probe: {}",
            probe.first_error.unwrap_or_default()
        ));
    }
    let mut toggle = None;
    for (name, off, on) in [
        ("enable_thinking", json!(false), json!(true)),
        ("thinking", json!(false), json!(true)),
        ("thinking_mode", json!("chat"), json!("thinking")),
        ("thinking_mode", json!("disabled"), json!("enabled")),
    ] {
        let disabled = control(name, off);
        let enabled = control(name, on);
        let off = probe.render(&disabled);
        let on = probe.render(&enabled);
        if comparable(&baseline, &off) && comparable(&baseline, &on) && off != on {
            toggle = Some((disabled, enabled, off, on));
            break;
        }
    }
    let has_toggle = toggle.is_some();
    let (disabled, enabled, off, on) = toggle.unwrap_or_else(|| {
        (
            Controls::new(),
            Controls::new(),
            baseline.clone(),
            baseline.clone(),
        )
    });
    let adaptive_controls = control("thinking_mode", json!("adaptive"));
    let adaptive = probe.render(&adaptive_controls);
    let has_adaptive =
        has_toggle && comparable(&baseline, &adaptive) && adaptive != off && adaptive != on;
    let effort_base = probe.render(&enabled);
    // Unique invalid spellings distinguish arbitrary value echoes from a shared
    // fallback. These are probe values, not security tokens.
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = format!(
        "{:?}-{}",
        std::time::SystemTime::now(),
        NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let invalid = ["a", "b"].map(|suffix| {
        let mut c = enabled.clone();
        c.insert(
            "reasoning_effort".into(),
            json!(format!("magnitude-invalid-{nonce}-{suffix}")),
        );
        probe.render(&c)
    });
    let rejected = effort_base
        .iter()
        .zip(&invalid[0])
        .zip(&invalid[1])
        .all(|((base, a), b)| base.is_none() || (a.is_none() && b.is_none()));
    let fallback = comparable(&effort_base, &invalid[0])
        && comparable(&effort_base, &invalid[1])
        && invalid[0] == invalid[1];
    let effort_policy = EffortPolicy { rejected, fallback };
    let disabled_effort = if rejected || fallback {
        probe.effort(
            "none",
            &["none", "off", "no_think", "disabled"],
            &enabled,
            &effort_base,
            &invalid[0],
            effort_policy,
        )?
    } else {
        None
    };
    let mut options: Vec<Observed> = vec![];
    if rejected || fallback {
        for (effort, spellings) in EFFORTS {
            let Some(mut option) = probe.effort(
                effort,
                spellings,
                &enabled,
                &effort_base,
                &invalid[0],
                effort_policy,
            )?
            else {
                continue;
            };
            if let Some(index) = options.iter().position(|existing| existing.1 == option.1) {
                if names(&option.1, &options[index].0.effort) && !names(&option.1, effort) {
                    options[index].0.aliases.push((*effort).into());
                    continue;
                }
                let existing = options.remove(index);
                option.0.aliases = existing.0.aliases;
                option.0.aliases.push(existing.0.effort);
            }
            options.push(option);
        }
    }
    if options.is_empty() && has_toggle {
        options.push(observed("none", disabled, off));
        if has_adaptive {
            options.push(observed("adaptive", adaptive_controls, adaptive));
        }
        options.push(observed("high", enabled, on.clone()));
    } else if options.is_empty() {
        if let Some(disabled) = disabled_effort {
            if disabled.1 != effort_base {
                options.push(disabled);
                options.push(observed("high", enabled, effort_base));
            }
        }
    } else if has_toggle {
        options.insert(0, observed("none", disabled, off));
    } else if let Some(disabled) = disabled_effort {
        options.insert(0, disabled);
    }
    let preserve = template
        .capabilities()
        .map_err(|e| e.to_string())?
        .get("supports_preserve_reasoning")
        .copied()
        .unwrap_or(false);
    if options.is_empty() {
        let thinking = preserve
            || baseline.iter().chain(&on).flatten().any(|s| {
                s.thinking || s.prompt.contains("<think>") || s.prompt.contains("<reasoning>")
            });
        options.push(observed(
            if thinking { "high" } else { "none" },
            Controls::new(),
            baseline.clone(),
        ));
    }
    let defaults = options
        .iter()
        .filter(|(_, outcomes)| *outcomes == baseline)
        .map(|(m, _)| m.effort.clone())
        .collect::<Vec<_>>();
    Ok(ReasoningProfile {
        detector: DETECTOR.into(),
        template_identity: template.identity().into(),
        option_context,
        default_effort: if defaults.len() == 1 {
            Some(defaults[0].clone())
        } else {
            None
        },
        mappings: options.into_iter().map(|(m, _)| m).collect(),
        baseline_shapes: baseline.iter().map(Option::is_some).collect(),
        effort_domain: if rejected {
            EffortDomain::Closed
        } else if fallback {
            EffortDomain::SharedFallback
        } else {
            EffortDomain::OpenOrIgnored
        },
        supports_reasoning_output: baseline
            .iter()
            .chain(&on)
            .flatten()
            .any(|s| s.thinking || !s.start.is_empty() || !s.ends.is_empty())
            .then_some(true),
        supports_preserve_reasoning: preserve,
    })
}
