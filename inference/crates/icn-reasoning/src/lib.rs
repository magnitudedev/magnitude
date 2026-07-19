//! Model-free chat-template and reasoning capability inspection.

use getrandom::fill;
use icn_contracts::{
    CapabilityEvidence, ReasoningCapability, ReasoningControlDomain, ReasoningDelimiters,
    ReasoningVisibility, TemplateAssessment, TemplateAssessor, TemplateCapabilities,
};
use llama_cpp_2::common_chat::{
    ChatContent, ChatMessage, ChatPrepareOptions, ChatTemplateKwarg, ChatTool, ChatToolCall,
    ChatToolChoice, CommonChatTemplates,
};
use sha2::{Digest, Sha256};

const EFFORT_CANDIDATES: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateInspection {
    pub template_fingerprint: String,
    pub capabilities: TemplateCapabilities,
    pub reasoning: ReasoningCapability,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NativeTemplateAssessor;

impl TemplateAssessor for NativeTemplateAssessor {
    fn assess(
        &self,
        template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<TemplateAssessment, String> {
        inspect_template(template, bos_token, eos_token)
            .map(|inspection| TemplateAssessment {
                capabilities: inspection.capabilities,
                reasoning: inspection.reasoning,
                fingerprint: inspection.template_fingerprint,
            })
            .map_err(|error| error.to_string())
    }
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
    let fingerprint = format!("sha256:{:x}", Sha256::digest(source.as_bytes()));
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
    let enabled = render_signatures(templates, &shapes, true, None);
    let disabled = render_signatures(templates, &shapes, false, None);
    let native_thinking = capabilities.enable_thinking
        || enabled
            .as_ref()
            .is_ok_and(|signatures| signatures.iter().any(|item| item.supports_thinking));
    let toggle_changes_prompt = enabled
        .as_ref()
        .ok()
        .zip(disabled.as_ref().ok())
        .is_some_and(|(left, right)| left != right);

    let reasoning = if native_thinking || capabilities.preserve_reasoning || toggle_changes_prompt {
        let effort_levels = detect_effort_levels(templates, &shapes)?;
        let prepared = enabled.as_ref().ok().and_then(|items| items.first());
        let delimiters = prepared
            .and_then(|item| item.start.as_ref().zip(item.end.as_ref()))
            .map_or(ReasoningDelimiters::Unavailable, |(start, end)| {
                ReasoningDelimiters::Known {
                    start: start.clone(),
                    end: end.clone(),
                }
            });
        let visibility = if capabilities.preserve_reasoning {
            ReasoningVisibility::Preserved
        } else {
            ReasoningVisibility::Configurable
        };
        let control = if effort_levels.len() >= 2 {
            let default = infer_default_effort(templates, &shapes, &effort_levels);
            ReasoningControlDomain::Effort {
                levels: effort_levels,
                default,
            }
        } else {
            ReasoningControlDomain::Toggle { default: true }
        };
        ReasoningCapability::Supported {
            control,
            visibility,
            delimiters,
            evidence: CapabilityEvidence::NativeTemplate {
                fingerprint: fingerprint.clone(),
            },
        }
    } else {
        ReasoningCapability::Unsupported {
            evidence: CapabilityEvidence::NativeTemplate {
                fingerprint: fingerprint.clone(),
            },
        }
    };

    Ok(TemplateInspection {
        template_fingerprint: fingerprint,
        capabilities,
        reasoning,
    })
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

#[derive(Debug, Clone)]
struct ProbeShape {
    messages: Vec<ChatMessage>,
    tools: Vec<ChatTool>,
}

fn render_signatures(
    templates: &CommonChatTemplates,
    shapes: &[ProbeShape],
    enable_thinking: bool,
    effort: Option<&str>,
) -> Result<Vec<RenderSignature>, InspectionError> {
    shapes
        .iter()
        .map(|shape| {
            let template_kwargs = effort
                .map(|value| ChatTemplateKwarg {
                    key: "reasoning_effort".to_owned(),
                    value_json: serde_json::to_string(value)
                        .expect("string serialization cannot fail"),
                })
                .into_iter()
                .collect();
            let prepared = templates
                .prepare(&ChatPrepareOptions {
                    messages: shape.messages.clone(),
                    tools: shape.tools.clone(),
                    tool_choice: ChatToolChoice::Auto,
                    enable_thinking,
                    template_kwargs,
                    ..ChatPrepareOptions::default()
                })
                .map_err(native_error)?;
            Ok(RenderSignature {
                prompt: prepared.prompt().to_owned(),
                generation_prompt: prepared.generation_prompt().to_owned(),
                parser: prepared.parser_definition().to_owned(),
                supports_thinking: prepared.supports_thinking(),
                start: prepared.thinking_start_tag().map(str::to_owned),
                end: prepared.thinking_end_tag().map(str::to_owned),
            })
        })
        .collect()
}

fn detect_effort_levels(
    templates: &CommonChatTemplates,
    shapes: &[ProbeShape],
) -> Result<Vec<String>, InspectionError> {
    let baseline = match render_signatures(templates, shapes, true, None) {
        Ok(value) => value,
        Err(_) => return Ok(Vec::new()),
    };
    let mut accepted = Vec::new();
    for candidate in EFFORT_CANDIDATES {
        if render_signatures(templates, shapes, true, Some(candidate))
            .is_ok_and(|rendered| rendered != baseline)
        {
            accepted.push((*candidate).to_owned());
        }
    }
    if accepted.len() < 2 {
        return Ok(Vec::new());
    }

    // A closed effort domain rejects or ignores unrelated randomized values. If arbitrary values
    // affect output, the kwarg is pass-through and we cannot truthfully publish an exact domain.
    for invalid in [random_invalid_effort()?, random_invalid_effort()?] {
        if render_signatures(templates, shapes, true, Some(&invalid))
            .is_ok_and(|rendered| rendered != baseline)
        {
            return Ok(Vec::new());
        }
    }
    Ok(accepted)
}

fn infer_default_effort(
    templates: &CommonChatTemplates,
    shapes: &[ProbeShape],
    levels: &[String],
) -> Option<String> {
    let baseline = render_signatures(templates, shapes, true, None).ok()?;
    levels.iter().find_map(|level| {
        render_signatures(templates, shapes, true, Some(level))
            .ok()
            .filter(|rendered| rendered == &baseline)
            .map(|_| level.clone())
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
            reasoning_content: None,
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
    const CLOSED_EFFORT: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if reasoning_effort == 'low' %}[low]{% elif reasoning_effort == 'medium' %}[medium]{% elif reasoning_effort == 'high' %}[high]{% endif %}assistant:"#;
    const OPEN_EFFORT: &str = r#"{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if reasoning_effort %}[{{ reasoning_effort }}]{% endif %}assistant:"#;

    #[test]
    fn plain_template_is_not_promoted_to_reasoning() {
        let result = inspect_template(BASIC, None, None).unwrap();
        assert!(matches!(
            result.reasoning,
            ReasoningCapability::Unsupported { .. }
        ));
    }

    #[test]
    fn native_toggle_is_detected_without_loading_a_model() {
        let result = inspect_template(TOGGLE, None, None).unwrap();
        assert!(matches!(
            result.reasoning,
            ReasoningCapability::Supported {
                control: ReasoningControlDomain::Toggle { .. },
                ..
            }
        ));
    }

    #[test]
    fn bounded_probe_reports_only_a_closed_effort_domain() {
        let result = inspect_template(CLOSED_EFFORT, None, None).unwrap();
        let ReasoningCapability::Supported {
            control: ReasoningControlDomain::Effort { levels, .. },
            ..
        } = result.reasoning
        else {
            panic!("expected an exact effort domain")
        };
        assert_eq!(levels, ["low", "medium", "high"]);
    }

    #[test]
    fn bounded_probe_does_not_claim_an_open_pass_through_domain() {
        let result = inspect_template(OPEN_EFFORT, None, None).unwrap();
        assert!(matches!(
            result.reasoning,
            ReasoningCapability::Supported {
                control: ReasoningControlDomain::Toggle { .. },
                ..
            }
        ));
    }
}
