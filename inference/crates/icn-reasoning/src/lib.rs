//! Native chat-template and reasoning capability inspection.

use std::collections::BTreeMap;

use getrandom::fill;
use icn_contracts::{
    AutomaticReasoningBudget, CapabilityEvidence, EffectiveTemplateInputs, NativeReasoningControls,
    NormalizedReasoningEffort, ReasoningCapability, ReasoningControlDomain, ReasoningDelimiters,
    ReasoningEffortMapping, ReasoningProfile, ReasoningVisibility, TemplateCapabilities,
};
use llama_cpp_2::common_chat::{
    ChatContent, ChatMessage, ChatPrepareOptions, ChatTemplateKwarg, ChatTool, ChatToolCall,
    ChatToolChoice, CommonChatTemplates,
};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use sha2::{Digest, Sha256};

const DISABLED_EFFORT_SPELLINGS: &[&str] = &["none", "off", "no_think", "disabled"];

const EFFORT_DEFINITIONS: &[(&str, &[&str])] = &[
    ("minimal", &["minimal"]),
    ("low", &["low"]),
    ("medium", &["medium"]),
    ("high", &["high"]),
    ("xhigh", &["xhigh", "extra_high", "extra-high", "very_high"]),
    ("max", &["max"]),
];

/// Version of the complete template-inspection semantics stored in model inspection caches.
pub const TEMPLATE_INSPECTION_CACHE_IDENTITY: &str = "template-inspection-v2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateInspection {
    pub template_fingerprint: String,
    pub capabilities: TemplateCapabilities,
    pub reasoning: ReasoningCapability,
    pub profile: ReasoningProfile,
}

pub fn inspect_template_inputs_with_backend(
    backend: &LlamaBackend,
    inputs: &EffectiveTemplateInputs,
) -> Result<TemplateInspection, InspectionError> {
    let params = LlamaModelParams::default().with_no_alloc(true);
    let model =
        LlamaModel::load_from_file(backend, &inputs.model_path, &params).map_err(native_error)?;
    let templates = CommonChatTemplates::from_model(&model).map_err(native_error)?;
    inspect_templates(&templates)
}

#[derive(Debug, thiserror::Error)]
pub enum InspectionError {
    #[error("native chat-template inspection failed: {0}")]
    Native(String),
    #[error("operating-system randomness failed: {0}")]
    Random(String),
}

/// Inspect a raw template without loading model weights or constructing a context.
pub fn inspect_template(
    template: &str,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<TemplateInspection, InspectionError> {
    let templates =
        CommonChatTemplates::from_template(template, bos_token, eos_token).map_err(native_error)?;
    inspect_templates(&templates)
}

/// Inspect an already constructed native template handle.
pub fn inspect_templates(
    templates: &CommonChatTemplates,
) -> Result<TemplateInspection, InspectionError> {
    let source = templates.source(None).map_err(native_error)?;
    let tool_use_source = templates.source(Some("tool_use")).map_err(native_error)?;
    let mut fingerprint_material = source.as_bytes().to_vec();
    if !tool_use_source.is_empty() {
        fingerprint_material.push(0);
        fingerprint_material.extend_from_slice(tool_use_source.as_bytes());
    }
    let fingerprint = format!("sha256:{:x}", Sha256::digest(&fingerprint_material));
    let native = templates.capabilities().map_err(native_error)?;
    let capabilities = TemplateCapabilities {
        string_content: native.supports_string_content,
        typed_content: native.supports_typed_content,
        tools: native.supports_tools,
        tool_calls: native.supports_tool_calls,
        parallel_tool_calls: native.supports_parallel_tool_calls,
        system_role: native.supports_system_role,
        preserve_reasoning: native.supports_preserve_reasoning,
        object_arguments: native.supports_object_arguments,
        enable_thinking: native.supports_enable_thinking,
    };

    let shapes = probe_shapes();
    let profile = inspect_profile(templates, &shapes, &fingerprint, &capabilities).unwrap_or_else(
        |_| ReasoningProfile {
            default_effort: None,
            mappings: Vec::new(),
            template_fingerprint: fingerprint.clone(),
        },
    );
    let default_controls = profile
        .default_effort
        .as_ref()
        .and_then(|effort| profile.mapping(effort))
        .map_or_else(NativeReasoningControls::default, |mapping| {
            mapping.controls.clone()
        });
    let prepared = render_outcomes(templates, &shapes, &default_controls);
    let delimiters = prepared
        .iter()
        .find_map(|outcome| match outcome {
            RenderOutcome::Rendered(item) => {
                item.start
                    .as_ref()
                    .zip(item.end.as_ref())
                    .map(|(start, end)| ReasoningDelimiters::Known {
                        start: start.clone(),
                        end: end.clone(),
                    })
            }
            RenderOutcome::Rejected(_) => None,
        })
        .unwrap_or(ReasoningDelimiters::Unavailable);
    let levels = profile
        .mappings
        .iter()
        .map(|mapping| mapping.effort.0.clone())
        .collect::<Vec<_>>();
    let reasoning = ReasoningCapability::Supported {
        control: ReasoningControlDomain::Effort {
            levels,
            default: profile
                .default_effort
                .as_ref()
                .map(|effort| effort.0.clone()),
        },
        visibility: if capabilities.preserve_reasoning {
            ReasoningVisibility::Preserved
        } else {
            ReasoningVisibility::Hidden
        },
        delimiters,
        evidence: CapabilityEvidence::BoundedTemplateProbe {
            fingerprint: fingerprint.clone(),
        },
    };

    Ok(TemplateInspection {
        template_fingerprint: fingerprint,
        capabilities,
        reasoning,
        profile,
    })
}

fn inspect_profile(
    templates: &CommonChatTemplates,
    shapes: &[ProbeShape],
    fingerprint: &str,
    capabilities: &TemplateCapabilities,
) -> Result<ReasoningProfile, InspectionError> {
    let omitted = NativeReasoningControls::default();
    let baseline = render_outcomes(templates, shapes, &omitted);
    if !baseline
        .iter()
        .any(|outcome| matches!(outcome, RenderOutcome::Rendered(_)))
    {
        let reason = baseline
            .into_iter()
            .find_map(|outcome| match outcome {
                RenderOutcome::Rejected(reason) => Some(reason),
                RenderOutcome::Rendered(_) => None,
            })
            .unwrap_or_else(|| "template rejected every probe shape".to_owned());
        return Err(InspectionError::Native(reason));
    }

    let toggle_candidates = [
        (
            NativeReasoningControls {
                enable_thinking: Some(false),
                template_args: BTreeMap::new(),
            },
            NativeReasoningControls {
                enable_thinking: Some(true),
                template_args: BTreeMap::new(),
            },
        ),
        (
            kwarg_controls("thinking", false),
            kwarg_controls("thinking", true),
        ),
        (
            string_kwarg_controls("thinking_mode", "chat"),
            string_kwarg_controls("thinking_mode", "thinking"),
        ),
        (
            string_kwarg_controls("thinking_mode", "disabled"),
            string_kwarg_controls("thinking_mode", "enabled"),
        ),
    ];
    let toggle = toggle_candidates
        .into_iter()
        .find_map(|(disabled, enabled)| {
            let disabled_outcomes = render_outcomes(templates, shapes, &disabled);
            let enabled_outcomes = render_outcomes(templates, shapes, &enabled);
            (comparable(&baseline, &disabled_outcomes)
                && comparable(&baseline, &enabled_outcomes)
                && !equivalent(&disabled_outcomes, &enabled_outcomes))
            .then_some((disabled, enabled, disabled_outcomes, enabled_outcomes))
        });
    let (disabled_controls, enabled_controls, disabled, enabled) =
        toggle.clone().unwrap_or_else(|| {
            (
                omitted.clone(),
                omitted.clone(),
                baseline.clone(),
                baseline.clone(),
            )
        });

    let adaptive_controls = string_kwarg_controls("thinking_mode", "adaptive");
    let adaptive_outcomes = render_outcomes(templates, shapes, &adaptive_controls);
    let adaptive_toggle = toggle.as_ref().is_some_and(|_| {
        comparable(&baseline, &adaptive_outcomes)
            && !equivalent(&adaptive_outcomes, &disabled)
            && !equivalent(&adaptive_outcomes, &enabled)
    });

    let effort_baseline_controls = if toggle.is_some() {
        enabled_controls.clone()
    } else {
        omitted.clone()
    };
    let effort_baseline = render_outcomes(templates, shapes, &effort_baseline_controls);
    let invalid_a = random_invalid_effort()?;
    let invalid_b = random_invalid_effort()?;
    let invalid_a_outcomes = render_outcomes(
        templates,
        shapes,
        &effort_controls(&effort_baseline_controls, &invalid_a),
    );
    let invalid_b_outcomes = render_outcomes(
        templates,
        shapes,
        &effort_controls(&effort_baseline_controls, &invalid_b),
    );
    let bounded_domain =
        bounded_effort_domain(&effort_baseline, invalid_a_outcomes, invalid_b_outcomes);
    let (mut disabled_effort, effort_options) = if let Some(domain) = &bounded_domain {
        let disabled = probe_normalized_effort(
            templates,
            shapes,
            &effort_baseline_controls,
            &effort_baseline,
            domain,
            "none",
            DISABLED_EFFORT_SPELLINGS,
        )?;
        let efforts = collapse_equivalent_efforts(probe_effort_options(
            templates,
            shapes,
            &effort_baseline_controls,
            &effort_baseline,
            domain,
        )?);
        (disabled, efforts)
    } else {
        (None, Vec::new())
    };

    let mut options = effort_options;

    if options.is_empty() && adaptive_toggle {
        options.push(observed_mapping(
            "none",
            disabled_controls.clone(),
            disabled.clone(),
        ));
        options.push(observed_mapping(
            "adaptive",
            adaptive_controls,
            adaptive_outcomes,
        ));
        options.push(observed_mapping(
            "high",
            enabled_controls.clone(),
            enabled.clone(),
        ));
    } else if options.is_empty() && toggle.is_some() {
        options.push(observed_mapping(
            "none",
            disabled_controls,
            disabled.clone(),
        ));
        options.push(observed_mapping("high", enabled_controls, enabled.clone()));
    } else if options.is_empty()
        && disabled_effort
            .as_ref()
            .is_some_and(|option| !equivalent(&option.outcomes, &effort_baseline))
    {
        options.push(disabled_effort.take().expect("checked as present"));
        options.push(observed_mapping(
            "high",
            effort_baseline_controls,
            effort_baseline,
        ));
    } else if !options.is_empty() {
        if toggle.is_some() {
            options.insert(
                0,
                observed_mapping("none", disabled_controls, disabled.clone()),
            );
        } else if let Some(disabled_option) = disabled_effort.take() {
            options.insert(0, disabled_option);
        }
    }

    let observed_thinking = baseline
        .iter()
        .chain(enabled.iter())
        .any(|outcome| match outcome {
            RenderOutcome::Rendered(signature) => {
                signature.supports_thinking
                    || signature.prompt.contains("<think>")
                    || signature.prompt.contains("<reasoning>")
            }
            RenderOutcome::Rejected(_) => false,
        })
        || capabilities.preserve_reasoning;
    if options.is_empty() {
        options.push(if observed_thinking {
            observed_mapping("high", omitted.clone(), baseline.clone())
        } else {
            observed_mapping("none", omitted.clone(), baseline.clone())
        });
    }

    let default_effort = detect_default_effort(&baseline, &options);
    let mappings = options.into_iter().map(|option| option.mapping).collect();

    Ok(ReasoningProfile {
        default_effort,
        mappings,
        template_fingerprint: fingerprint.to_owned(),
    })
}

#[derive(Debug, Clone)]
enum BoundedEffortDomain {
    RejectsUnknown,
    SharedUnknownFallback(Vec<RenderOutcome>),
}

impl BoundedEffortDomain {
    fn accepts(
        &self,
        baseline: &[RenderOutcome],
        candidate: &[RenderOutcome],
        normalized: &str,
    ) -> bool {
        if !comparable(baseline, candidate) {
            return false;
        }
        match self {
            Self::RejectsUnknown => true,
            Self::SharedUnknownFallback(fallback) => {
                !equivalent(candidate, fallback) || render_names_effort(candidate, normalized)
            }
        }
    }
}

fn bounded_effort_domain(
    baseline: &[RenderOutcome],
    invalid_a: Vec<RenderOutcome>,
    invalid_b: Vec<RenderOutcome>,
) -> Option<BoundedEffortDomain> {
    if rejected_where_baseline_renders(baseline, &invalid_a)
        && rejected_where_baseline_renders(baseline, &invalid_b)
    {
        return Some(BoundedEffortDomain::RejectsUnknown);
    }
    if comparable(baseline, &invalid_a)
        && comparable(baseline, &invalid_b)
        && equivalent(&invalid_a, &invalid_b)
    {
        return Some(BoundedEffortDomain::SharedUnknownFallback(invalid_a));
    }
    None
}

#[derive(Debug, Clone)]
struct ObservedOption {
    mapping: ReasoningEffortMapping,
    outcomes: Vec<RenderOutcome>,
}

fn observed_mapping(
    effort: &str,
    controls: NativeReasoningControls,
    outcomes: Vec<RenderOutcome>,
) -> ObservedOption {
    ObservedOption {
        mapping: mapping(effort, controls),
        outcomes,
    }
}

fn probe_effort_options(
    templates: &CommonChatTemplates,
    shapes: &[ProbeShape],
    base_controls: &NativeReasoningControls,
    baseline: &[RenderOutcome],
    domain: &BoundedEffortDomain,
) -> Result<Vec<ObservedOption>, InspectionError> {
    EFFORT_DEFINITIONS
        .iter()
        .filter_map(|(normalized, native_values)| {
            probe_normalized_effort(
                templates,
                shapes,
                base_controls,
                baseline,
                domain,
                normalized,
                native_values,
            )
            .transpose()
        })
        .collect()
}

fn probe_normalized_effort(
    templates: &CommonChatTemplates,
    shapes: &[ProbeShape],
    base_controls: &NativeReasoningControls,
    baseline: &[RenderOutcome],
    domain: &BoundedEffortDomain,
    normalized: &str,
    native_values: &[&str],
) -> Result<Option<ObservedOption>, InspectionError> {
    let mut selected: Option<ObservedOption> = None;
    for native_value in native_values {
        let controls = effort_controls(base_controls, native_value);
        let outcomes = render_outcomes(templates, shapes, &controls);
        if !domain.accepts(baseline, &outcomes, normalized) {
            continue;
        }
        if let Some(existing) = &selected {
            if !equivalent(&existing.outcomes, &outcomes) {
                return Err(InspectionError::Native(format!(
                    "native spellings for normalized effort {normalized} render differently"
                )));
            }
        } else {
            selected = Some(observed_mapping(normalized, controls, outcomes));
        }
    }
    Ok(selected)
}

fn collapse_equivalent_efforts(options: Vec<ObservedOption>) -> Vec<ObservedOption> {
    let mut distinct: Vec<ObservedOption> = Vec::new();
    for option in options {
        if let Some(index) = distinct
            .iter()
            .position(|existing| equivalent(&existing.outcomes, &option.outcomes))
        {
            let existing_is_named =
                render_names_effort(&option.outcomes, distinct[index].mapping.effort.as_str());
            let option_is_named =
                render_names_effort(&option.outcomes, option.mapping.effort.as_str());
            if existing_is_named && !option_is_named {
                continue;
            }
            distinct.remove(index);
        }
        distinct.push(option);
    }
    distinct
}

fn render_names_effort(outcomes: &[RenderOutcome], effort: &str) -> bool {
    let Some((_, affiliations)) = EFFORT_DEFINITIONS
        .iter()
        .find(|(normalized, _)| *normalized == effort)
    else {
        return false;
    };
    outcomes.iter().any(|outcome| match outcome {
        RenderOutcome::Rendered(signature) => affiliations.iter().any(|affiliation| {
            contains_effort_name(&signature.prompt, affiliation)
                || contains_effort_name(&signature.generation_prompt, affiliation)
        }),
        RenderOutcome::Rejected(_) => false,
    })
}

fn contains_effort_name(text: &str, effort: &str) -> bool {
    let text = text.to_ascii_lowercase();
    text.match_indices(effort).any(|(start, matched)| {
        let end = start + matched.len();
        let bounded_before = text[..start]
            .chars()
            .next_back()
            .is_none_or(|character| !is_effort_name_character(character));
        let bounded_after = text[end..]
            .chars()
            .next()
            .is_none_or(|character| !is_effort_name_character(character));
        bounded_before && bounded_after
    })
}

fn is_effort_name_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
}

fn detect_default_effort(
    baseline: &[RenderOutcome],
    options: &[ObservedOption],
) -> Option<NormalizedReasoningEffort> {
    let matches = options
        .iter()
        .filter(|option| equivalent(baseline, &option.outcomes))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [option] => Some(option.mapping.effort.clone()),
        [] | [_, _, ..] => None,
    }
}

fn mapping(effort: &str, controls: NativeReasoningControls) -> ReasoningEffortMapping {
    ReasoningEffortMapping {
        effort: NormalizedReasoningEffort::parse(effort).expect("policy effort is normalized"),
        controls,
        automatic_budget: AutomaticReasoningBudget::Disabled,
    }
}

fn effort_controls(base: &NativeReasoningControls, effort: &str) -> NativeReasoningControls {
    let mut controls = base.clone();
    controls.template_args.insert(
        "reasoning_effort".to_owned(),
        serde_json::Value::String(effort.to_owned()),
    );
    controls
}

fn kwarg_controls(key: &str, value: bool) -> NativeReasoningControls {
    NativeReasoningControls {
        enable_thinking: None,
        template_args: BTreeMap::from([(key.to_owned(), serde_json::Value::Bool(value))]),
    }
}

fn string_kwarg_controls(key: &str, value: &str) -> NativeReasoningControls {
    NativeReasoningControls {
        enable_thinking: None,
        template_args: BTreeMap::from([(
            key.to_owned(),
            serde_json::Value::String(value.to_owned()),
        )]),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RenderSignature {
    prompt: String,
    generation_prompt: String,
    parser: String,
    supports_thinking: bool,
    start: Option<String>,
    end: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RenderOutcome {
    Rendered(RenderSignature),
    Rejected(String),
}

#[derive(Debug, Clone)]
struct ProbeShape {
    messages: Vec<ChatMessage>,
    tools: Vec<ChatTool>,
}

fn render_outcomes(
    templates: &CommonChatTemplates,
    shapes: &[ProbeShape],
    controls: &NativeReasoningControls,
) -> Vec<RenderOutcome> {
    shapes
        .iter()
        .map(|shape| {
            let template_kwargs = controls
                .template_args
                .iter()
                .map(|(key, value)| ChatTemplateKwarg {
                    key: key.clone(),
                    value_json: serde_json::to_string(value).expect("JSON values always serialize"),
                })
                .collect();
            match templates.prepare(&ChatPrepareOptions {
                messages: shape.messages.clone(),
                tools: shape.tools.clone(),
                tool_choice: ChatToolChoice::Auto,
                enable_thinking: controls.enable_thinking,
                template_kwargs,
                ..ChatPrepareOptions::default()
            }) {
                Ok(prepared) => RenderOutcome::Rendered(RenderSignature {
                    prompt: prepared.prompt().to_owned(),
                    generation_prompt: prepared.generation_prompt().to_owned(),
                    parser: prepared.parser_definition().to_owned(),
                    supports_thinking: prepared.supports_thinking(),
                    start: prepared.thinking_start_tag().map(str::to_owned),
                    end: prepared.thinking_end_tag().map(str::to_owned),
                }),
                Err(error) => RenderOutcome::Rejected(error.to_string()),
            }
        })
        .collect()
}

fn comparable(baseline: &[RenderOutcome], candidate: &[RenderOutcome]) -> bool {
    let mut rendered = false;
    for (baseline, candidate) in baseline.iter().zip(candidate) {
        if matches!(baseline, RenderOutcome::Rendered(_)) {
            rendered = true;
            if !matches!(candidate, RenderOutcome::Rendered(_)) {
                return false;
            }
        }
    }
    rendered
}

fn rejected_where_baseline_renders(
    baseline: &[RenderOutcome],
    candidate: &[RenderOutcome],
) -> bool {
    let mut rendered = false;
    for (baseline, candidate) in baseline.iter().zip(candidate) {
        if matches!(baseline, RenderOutcome::Rendered(_)) {
            rendered = true;
            if !matches!(candidate, RenderOutcome::Rejected(_)) {
                return false;
            }
        }
    }
    rendered
}

fn equivalent(left: &[RenderOutcome], right: &[RenderOutcome]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| match (left, right) {
            (RenderOutcome::Rendered(left), RenderOutcome::Rendered(right)) => left == right,
            (RenderOutcome::Rejected(_), RenderOutcome::Rejected(_)) => true,
            _ => false,
        })
}

fn random_invalid_effort() -> Result<String, InspectionError> {
    let mut bytes = [0_u8; 16];
    fill(&mut bytes).map_err(|error| InspectionError::Random(error.to_string()))?;
    let suffix = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("magnitude-invalid-{suffix}"))
}

fn probe_shapes() -> Vec<ProbeShape> {
    let tool = ChatTool {
        name: "weather".to_owned(),
        description: "Get the current weather".to_owned(),
        parameters_json:
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                .to_owned(),
    };
    let plain = vec![ChatMessage::user("Explain why the sky appears blue.")];
    let with_tools = vec![ChatMessage::user("What is the weather in Paris?")];
    let after_tool = vec![
        ChatMessage::user("What is the weather in Paris?"),
        ChatMessage {
            role: "assistant".to_owned(),
            content: None,
            tool_calls: vec![ChatToolCall {
                name: "weather".to_owned(),
                arguments: r#"{"city":"Paris"}"#.to_owned(),
                id: Some("call_1".to_owned()),
            }],
            reasoning_content: Some("I should check the weather tool.".to_owned()),
            tool_name: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "tool".to_owned(),
            content: Some(ChatContent::Text("18 C and clear".to_owned())),
            tool_calls: Vec::new(),
            reasoning_content: None,
            tool_name: Some("weather".to_owned()),
            tool_call_id: Some("call_1".to_owned()),
        },
    ];
    vec![
        ProbeShape {
            messages: plain,
            tools: Vec::new(),
        },
        ProbeShape {
            messages: with_tools,
            tools: vec![tool.clone()],
        },
        ProbeShape {
            messages: after_tool,
            tools: vec![tool],
        },
    ]
}

fn native_error(error: impl std::fmt::Display) -> InspectionError {
    InspectionError::Native(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"#;
    const TOGGLE: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}assistant:"#;
    const FIXED: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}<think>assistant:"#;
    const THINKING_BOOL: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if thinking %}<think>{% endif %}assistant:"#;
    const THINKING_MODE: &str = r#"{% set mode = thinking_mode|default('adaptive') %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}[{{ mode }}]assistant:"#;
    const EFFORT_TOGGLE: &str = r#"{% set effort = reasoning_effort|default('high') %}{% if effort not in ('none', 'high') %}{{ raise_exception('unsupported effort') }}{% endif %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if effort == 'high' %}<think>{% endif %}assistant:"#;
    const EFFORT_NONE_MATCHES_LOW: &str = r#"{% set effort = reasoning_effort|default('high') %}{% if effort not in ('none', 'low', 'high') %}{{ raise_exception('unsupported effort') }}{% endif %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if effort == 'high' %}[high]{% else %}[shared]{% endif %}assistant:"#;
    const CLOSED_EFFORT: &str = r#"{% set effort = reasoning_effort|default('high') %}{% if effort not in ('low', 'medium', 'high') %}{{ raise_exception('unsupported effort') }}{% endif %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if effort == 'low' %}[low]{% elif effort == 'medium' %}[medium]{% elif effort == 'high' %}[high]{% endif %}assistant:"#;
    const ONE_ENABLED_EFFORT_BEHAVIOR: &str = r#"{% set effort = reasoning_effort|default('high') %}{% if effort not in ('low', 'medium', 'high', 'xhigh', 'max') %}{{ raise_exception('unsupported effort') }}{% endif %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking is undefined or enable_thinking is true %}<think>{% endif %}assistant:"#;
    const SHARED_FALLBACK_EFFORT: &str = r#"{% set effort = reasoning_effort|default('high') %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if effort == 'low' %}[low]{% elif effort == 'high' %}[high]{% else %}[fallback]{% endif %}assistant:"#;
    const NAMED_SHARED_FALLBACK_EFFORT: &str = r#"{% set effort = 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking is undefined or enable_thinking %}[Reasoning Effort: {{ effort }}]<think>{% endif %}assistant:"#;
    const QWEN_3_8_EFFORT: &str = r#"{% if enable_thinking is undefined or enable_thinking is true %}{% set effort = reasoning_effort|default('xhigh') %}{% if effort == 'high' %}{% set effort = 'xhigh' %}{% endif %}{% if effort not in ('xhigh', 'medium', 'low') %}{{ raise_exception('unsupported effort') }}{% endif %}{% if effort == 'xhigh' %}[xhigh]{% elif effort == 'low' %}[low]{% endif %}{% endif %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"#;
    const REVERSE_EFFORT_ALIAS: &str = r#"{% set effort = reasoning_effort|default('high') %}{% if effort == 'xhigh' %}{% set effort = 'high' %}{% endif %}{% if effort not in ('low', 'medium', 'high') %}{{ raise_exception('unsupported effort') }}{% endif %}{% if effort == 'low' %}[low]{% elif effort == 'medium' %}[medium]{% elif effort == 'high' %}[high]{% endif %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"#;
    const UNNAMED_DEFAULT_EFFORT: &str = r#"{% if reasoning_effort is defined and reasoning_effort not in ('low', 'high') %}{{ raise_exception('unsupported effort') }}{% endif %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if reasoning_effort == 'low' %}[low]{% elif reasoning_effort == 'high' %}[high]{% else %}[unnamed]{% endif %}assistant:"#;
    const OPEN_EFFORT: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if reasoning_effort %}[{{ reasoning_effort }}]{% endif %}assistant:"#;

    fn efforts(template: &str) -> Vec<String> {
        inspect_template(template, None, None)
            .unwrap()
            .profile
            .mappings
            .into_iter()
            .map(|mapping| mapping.effort.0)
            .collect()
    }

    fn default_effort(template: &str) -> Option<String> {
        inspect_template(template, None, None)
            .unwrap()
            .profile
            .default_effort
            .map(|effort| effort.0)
    }

    #[test]
    fn plain_template_normalizes_to_none() {
        let result = inspect_template(BASIC, None, None).unwrap();
        assert_eq!(efforts(BASIC), ["none"]);
        assert!(matches!(
            result.reasoning,
            ReasoningCapability::Supported {
                control: ReasoningControlDomain::Effort { .. },
                ..
            }
        ));
    }

    #[test]
    fn native_toggle_normalizes_to_none_and_high() {
        let result = inspect_template(TOGGLE, None, None).unwrap();
        assert_eq!(efforts(TOGGLE), ["none", "high"]);
        assert_eq!(default_effort(TOGGLE).as_deref(), Some("high"));
        assert_eq!(
            result.profile.mappings[1].controls.enable_thinking,
            Some(true)
        );
    }

    #[test]
    fn fixed_reasoning_normalizes_to_high_only() {
        assert_eq!(efforts(FIXED), ["high"]);
    }

    #[test]
    fn alternate_boolean_normalizes_to_none_and_high() {
        assert_eq!(efforts(THINKING_BOOL), ["none", "high"]);
    }

    #[test]
    fn three_state_mode_preserves_adaptive() {
        assert_eq!(efforts(THINKING_MODE), ["none", "adaptive", "high"]);
    }

    #[test]
    fn effort_only_toggle_normalizes_to_none_and_high() {
        let result = inspect_template(EFFORT_TOGGLE, None, None).unwrap();
        assert_eq!(efforts(EFFORT_TOGGLE), ["none", "high"]);
        assert_eq!(default_effort(EFFORT_TOGGLE).as_deref(), Some("high"));
        assert!(
            result.profile.mappings[0]
                .controls
                .template_args
                .contains_key("reasoning_effort")
        );
    }

    #[test]
    fn verified_none_is_retained_when_it_renders_like_low() {
        assert_eq!(efforts(EFFORT_NONE_MATCHES_LOW), ["none", "low", "high"]);
        assert_eq!(default_effort(EFFORT_NONE_MATCHES_LOW).as_deref(), Some("high"));
    }

    #[test]
    fn bounded_probe_reports_only_a_closed_effort_domain() {
        assert_eq!(efforts(CLOSED_EFFORT), ["none", "low", "medium", "high"]);
        assert_eq!(
            default_effort(CLOSED_EFFORT).as_deref(),
            Some("high")
        );
    }

    #[test]
    fn accepted_baseline_equivalent_effort_is_the_default() {
        let result = inspect_template(QWEN_3_8_EFFORT, None, None).unwrap();
        assert_eq!(
            result
                .profile
                .mappings
                .iter()
                .map(|mapping| mapping.effort.as_str())
                .collect::<Vec<_>>(),
            ["none", "low", "medium", "xhigh"]
        );
        assert_eq!(default_effort(QWEN_3_8_EFFORT).as_deref(), Some("xhigh"));
        assert_eq!(
            result
                .profile
                .mapping(result.profile.default_effort.as_ref().unwrap())
                .unwrap()
                .controls,
            NativeReasoningControls {
                enable_thinking: Some(true),
                template_args: BTreeMap::from([(
                    "reasoning_effort".to_owned(),
                    serde_json::Value::String("xhigh".to_owned()),
                )]),
            }
        );
    }

    #[test]
    fn rendered_affiliation_selects_xhigh_over_its_high_alias() {
        let result = inspect_template(QWEN_3_8_EFFORT, None, None).unwrap();
        assert!(
            result
                .profile
                .mapping(&NormalizedReasoningEffort("high".to_owned()))
                .is_none()
        );
        assert!(
            result
                .profile
                .mapping(&NormalizedReasoningEffort("xhigh".to_owned()))
                .is_some()
        );
    }

    #[test]
    fn rendered_affiliation_selects_high_over_its_xhigh_alias() {
        let result = inspect_template(REVERSE_EFFORT_ALIAS, None, None).unwrap();
        assert_eq!(
            result
                .profile
                .mappings
                .iter()
                .map(|mapping| mapping.effort.as_str())
                .collect::<Vec<_>>(),
            ["low", "medium", "high"]
        );
        assert_eq!(default_effort(REVERSE_EFFORT_ALIAS).as_deref(), Some("high"));
    }

    #[test]
    fn rank_selects_the_representative_when_render_has_no_affiliated_name() {
        let result = inspect_template(ONE_ENABLED_EFFORT_BEHAVIOR, None, None).unwrap();
        assert_eq!(
            result
                .profile
                .mappings
                .iter()
                .map(|mapping| mapping.effort.as_str())
                .collect::<Vec<_>>(),
            ["none", "max"]
        );
        assert_eq!(
            default_effort(ONE_ENABLED_EFFORT_BEHAVIOR).as_deref(),
            Some("max")
        );
    }

    #[test]
    fn shared_unknown_fallback_does_not_hide_a_distinct_named_default() {
        let result = inspect_template(SHARED_FALLBACK_EFFORT, None, None).unwrap();
        assert_eq!(
            result
                .profile
                .mappings
                .iter()
                .map(|mapping| mapping.effort.as_str())
                .collect::<Vec<_>>(),
            ["none", "low", "high"]
        );
        assert_eq!(
            default_effort(SHARED_FALLBACK_EFFORT).as_deref(),
            Some("high")
        );
    }

    #[test]
    fn rendered_name_identifies_a_shared_unknown_fallback() {
        assert_eq!(
            efforts(NAMED_SHARED_FALLBACK_EFFORT),
            ["none", "high", "max"]
        );
        assert_eq!(
            default_effort(NAMED_SHARED_FALLBACK_EFFORT).as_deref(),
            Some("max")
        );
    }

    #[test]
    fn unnamed_default_preserves_model_default() {
        let result = inspect_template(UNNAMED_DEFAULT_EFFORT, None, None).unwrap();
        assert_eq!(result.profile.default_effort, None);
        assert_eq!(efforts(UNNAMED_DEFAULT_EFFORT), ["none", "low", "high"]);
    }

    #[test]
    fn bounded_probe_does_not_claim_an_open_pass_through_domain() {
        assert_eq!(efforts(OPEN_EFFORT), ["none", "high"]);
    }

    #[test]
    fn every_automatic_budget_is_disabled() {
        let result = inspect_template(CLOSED_EFFORT, None, None).unwrap();
        assert!(
            result.profile.mappings.iter().all(|mapping| matches!(
                mapping.automatic_budget,
                AutomaticReasoningBudget::Disabled
            ))
        );
    }

}
