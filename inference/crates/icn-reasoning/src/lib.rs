//! Model-free chat-template and reasoning capability inspection.

use std::collections::BTreeMap;

use getrandom::fill;
use icn_contracts::{
    AutomaticReasoningBudget, CapabilityEvidence, EffectiveTemplateInputs, NativeReasoningControls,
    NormalizedReasoningEffort, ReasoningCapability, ReasoningControlDomain, ReasoningDelimiters,
    ReasoningEffortMapping, ReasoningProfile, ReasoningVisibility, TemplateAssessment,
    TemplateAssessor, TemplateCapabilities,
};
use llama_cpp_2::common_chat::{
    ChatContent, ChatMessage, ChatPrepareOptions, ChatTemplateKwarg, ChatTool, ChatToolCall,
    ChatToolChoice, CommonChatTemplates,
};
use sha2::{Digest, Sha256};

const EFFORT_DEFINITIONS: &[(&str, &[&str])] = &[
    ("none", &["none", "off", "no_think"]),
    ("minimal", &["minimal"]),
    ("low", &["low"]),
    ("medium", &["medium"]),
    ("high", &["high"]),
    ("xhigh", &["xhigh", "extra_high", "extra-high", "very_high"]),
    ("max", &["max"]),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateInspection {
    pub template_fingerprint: String,
    pub capabilities: TemplateCapabilities,
    pub reasoning: ReasoningCapability,
    pub profile: ReasoningProfile,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NativeTemplateAssessor;

impl TemplateAssessor for NativeTemplateAssessor {
    fn cache_identity(&self) -> &str {
        concat!("icn-native-template-assessor:", env!("CARGO_PKG_VERSION"))
    }

    fn assess(&self, inputs: &EffectiveTemplateInputs) -> Result<TemplateAssessment, String> {
        inspect_template_inputs(inputs)
            .map(|inspection| TemplateAssessment {
                capabilities: inspection.capabilities,
                reasoning: inspection.reasoning,
                fingerprint: inspection.template_fingerprint,
            })
            .map_err(|error| error.to_string())
    }
}

pub fn inspect_template_inputs(
    inputs: &EffectiveTemplateInputs,
) -> Result<TemplateInspection, InspectionError> {
    let templates = CommonChatTemplates::from_metadata(
        inputs.default_template.as_deref(),
        inputs.tool_use_template.as_deref(),
        inputs.bos_token.as_deref(),
        inputs.eos_token.as_deref(),
    )
    .map_err(native_error)?;
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
    inspect_template_inputs(&EffectiveTemplateInputs {
        default_template: Some(template.to_owned()),
        tool_use_template: None,
        bos_token: bos_token.map(str::to_owned),
        eos_token: eos_token.map(str::to_owned),
    })
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
    let profile = inspect_profile(templates, &shapes, &fingerprint, &capabilities)?;
    let default_mapping = profile
        .mapping(&profile.default_effort)
        .expect("inspected profile contains its default");
    let prepared = render_outcomes(templates, &shapes, &default_mapping.controls);
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
            default: Some(profile.default_effort.0.clone()),
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

    if let Some(profile) = declared_profile(fingerprint) {
        for mapping in &profile.mappings {
            let outcomes = render_outcomes(templates, shapes, &mapping.controls);
            if !comparable(&baseline, &outcomes) {
                return Err(InspectionError::Native(format!(
                    "declared reasoning recipe {} is rejected by template {fingerprint}",
                    mapping.effort.as_str()
                )));
            }
        }
        return Ok(profile);
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
    let invalids_reject = rejected_where_baseline_renders(&effort_baseline, &invalid_a_outcomes)
        && rejected_where_baseline_renders(&effort_baseline, &invalid_b_outcomes);
    let invalids_share_fallback = comparable(&effort_baseline, &invalid_a_outcomes)
        && comparable(&effort_baseline, &invalid_b_outcomes)
        && equivalent(&invalid_a_outcomes, &invalid_b_outcomes);
    let open_pass_through = comparable(&effort_baseline, &invalid_a_outcomes)
        && comparable(&effort_baseline, &invalid_b_outcomes)
        && !equivalent(&invalid_a_outcomes, &invalid_b_outcomes);

    let mut mappings = Vec::new();
    if !open_pass_through && (invalids_reject || invalids_share_fallback) {
        for (normalized, native_values) in EFFORT_DEFINITIONS {
            let mut selected: Option<(NativeReasoningControls, Vec<RenderOutcome>)> = None;
            for native_value in *native_values {
                let controls = effort_controls(&effort_baseline_controls, native_value);
                let outcomes = render_outcomes(templates, shapes, &controls);
                if !comparable(&effort_baseline, &outcomes) {
                    continue;
                }
                if invalids_share_fallback && equivalent(&outcomes, &invalid_a_outcomes) {
                    continue;
                }
                if equivalent(&outcomes, &effort_baseline) {
                    continue;
                }
                if let Some((_, existing)) = &selected {
                    if !equivalent(existing, &outcomes) {
                        return Err(InspectionError::Native(format!(
                            "native aliases for normalized effort {normalized} render differently"
                        )));
                    }
                } else {
                    selected = Some((controls, outcomes));
                }
            }
            if let Some((controls, _)) = selected {
                mappings.push(mapping(normalized, controls));
            }
        }
    }

    if mappings.is_empty() && adaptive_toggle {
        mappings.push(mapping("none", disabled_controls.clone()));
        mappings.push(mapping("adaptive", adaptive_controls));
        mappings.push(mapping("high", enabled_controls.clone()));
    }

    let has_none = mappings
        .iter()
        .any(|mapping| mapping.effort.as_str() == "none");
    if mappings.is_empty() && toggle.is_some() {
        mappings.push(mapping("none", disabled_controls));
        mappings.push(mapping("high", enabled_controls));
    } else if toggle.is_some() && !has_none {
        mappings.insert(0, mapping("none", disabled_controls));
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
    if mappings.is_empty() {
        mappings.push(if observed_thinking {
            mapping("high", omitted.clone())
        } else {
            mapping("none", omitted.clone())
        });
    }

    let default_effort = mappings
        .iter()
        .find(|mapping| {
            let outcomes = render_outcomes(templates, shapes, &mapping.controls);
            comparable(&baseline, &outcomes) && equivalent(&baseline, &outcomes)
        })
        .or_else(|| {
            mappings
                .iter()
                .find(|mapping| mapping.effort.as_str() == "high")
        })
        .unwrap_or(&mappings[0])
        .effort
        .clone();

    Ok(ReasoningProfile {
        default_effort,
        mappings,
        template_fingerprint: fingerprint.to_owned(),
    })
}

fn declared_profile(fingerprint: &str) -> Option<ReasoningProfile> {
    let definitions: Vec<(&str, NativeReasoningControls)>;
    let default;
    match fingerprint {
        // zai-org/GLM-5.2, revision b4734de4facf877f85769a911abafc5283eab3d9.
        "sha256:172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679" => {
            definitions = vec![
                (
                    "none",
                    NativeReasoningControls {
                        enable_thinking: Some(false),
                        template_args: BTreeMap::new(),
                    },
                ),
                (
                    "high",
                    NativeReasoningControls {
                        enable_thinking: Some(true),
                        template_args: BTreeMap::from([(
                            "reasoning_effort".into(),
                            serde_json::Value::String("high".into()),
                        )]),
                    },
                ),
                (
                    "max",
                    NativeReasoningControls {
                        enable_thinking: Some(true),
                        template_args: BTreeMap::from([(
                            "reasoning_effort".into(),
                            serde_json::Value::String("max".into()),
                        )]),
                    },
                ),
            ];
            default = "max";
        }
        // openai/gpt-oss-120b, revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a.
        "sha256:a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146" => {
            definitions = vec![
                ("low", string_kwarg_controls("reasoning_effort", "low")),
                (
                    "medium",
                    string_kwarg_controls("reasoning_effort", "medium"),
                ),
                ("high", string_kwarg_controls("reasoning_effort", "high")),
            ];
            default = "medium";
        }
        // deepseek-ai/DeepSeek-V4-Pro Jinja, revision 83adbd0b1e5f49ced28cbb6cb3bcc89a7360ed3d.
        "sha256:31ae9909c6818e5a7fe82c538ea31cd330b25c06cf6b65e11f532a8d389e1cbc" => {
            definitions = vec![
                ("none", string_kwarg_controls("thinking_mode", "chat")),
                (
                    "high",
                    controls_with_strings(&[
                        ("thinking_mode", "thinking"),
                        ("reasoning_effort", "high"),
                    ]),
                ),
                (
                    "max",
                    controls_with_strings(&[
                        ("thinking_mode", "thinking"),
                        ("reasoning_effort", "max"),
                    ]),
                ),
            ];
            default = "high";
        }
        _ => return None,
    }
    Some(ReasoningProfile {
        default_effort: NormalizedReasoningEffort::parse(default)
            .expect("declared default is normalized"),
        mappings: definitions
            .iter()
            .map(|(effort, controls)| mapping(effort, controls.clone()))
            .collect(),
        template_fingerprint: fingerprint.to_owned(),
    })
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

fn controls_with_strings(values: &[(&str, &str)]) -> NativeReasoningControls {
    NativeReasoningControls {
        enable_thinking: None,
        template_args: values
            .iter()
            .map(|(key, value)| {
                (
                    (*key).to_owned(),
                    serde_json::Value::String((*value).to_owned()),
                )
            })
            .collect(),
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
    const THINKING_MODE: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}[{{ thinking_mode }}]assistant:"#;
    const CLOSED_EFFORT: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if reasoning_effort == 'low' %}[low]{% elif reasoning_effort == 'medium' %}[medium]{% elif reasoning_effort == 'high' %}[high]{% endif %}assistant:"#;
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
        assert_eq!(result.profile.default_effort.as_str(), "none");
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
    fn bounded_probe_reports_only_a_closed_effort_domain() {
        assert_eq!(efforts(CLOSED_EFFORT), ["none", "low", "medium", "high"]);
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

    #[test]
    fn declared_ambiguous_domains_have_exact_normalized_options() {
        let cases = [
            (
                "sha256:172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
                &["none", "high", "max"][..],
                "max",
            ),
            (
                "sha256:a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
                &["low", "medium", "high"][..],
                "medium",
            ),
            (
                "sha256:31ae9909c6818e5a7fe82c538ea31cd330b25c06cf6b65e11f532a8d389e1cbc",
                &["none", "high", "max"][..],
                "high",
            ),
        ];
        for (fingerprint, expected, default) in cases {
            let profile = declared_profile(fingerprint).unwrap();
            assert_eq!(
                profile
                    .mappings
                    .iter()
                    .map(|mapping| mapping.effort.as_str())
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(profile.default_effort.as_str(), default);
            assert!(profile.mappings.iter().all(|mapping| matches!(
                mapping.automatic_budget,
                AutomaticReasoningBudget::Disabled
            )));
        }
    }
}
