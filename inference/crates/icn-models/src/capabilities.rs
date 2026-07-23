use icn_contracts::models::{ModelCapabilities, ModelReasoningCapabilities};
use icn_contracts::{CapabilitySupport, InventoryProperties, ReasoningCapability};

pub(crate) fn model_capabilities(properties: &InventoryProperties) -> ModelCapabilities {
    let InventoryProperties::Inspected {
        modalities,
        tools,
        structured_output,
        reasoning,
        ..
    } = properties
    else {
        return ModelCapabilities {
            vision: false,
            tools: false,
            structured_output: false,
            reasoning: ModelReasoningCapabilities {
                supported: false,
                efforts: Vec::new(),
                default_effort: None,
            },
        };
    };
    let reasoning = match reasoning {
        ReasoningCapability::Unsupported { .. } => ModelReasoningCapabilities {
            supported: false,
            efforts: Vec::new(),
            default_effort: None,
        },
        ReasoningCapability::Supported { control, .. } => {
            let (efforts, requested_default) = match control {
                icn_contracts::ReasoningControlDomain::Toggle { default } => (
                    vec!["none".to_owned(), "high".to_owned()],
                    Some(if *default { "high" } else { "none" }.to_owned()),
                ),
                icn_contracts::ReasoningControlDomain::Effort { levels, default } => {
                    (levels.clone(), default.clone())
                }
                icn_contracts::ReasoningControlDomain::Budget { .. } => {
                    (vec!["high".to_owned()], Some("high".to_owned()))
                }
                icn_contracts::ReasoningControlDomain::EffortAndBudget {
                    levels,
                    default_effort,
                    ..
                } => (levels.clone(), default_effort.clone()),
            };
            let efforts = efforts.into_iter().fold(Vec::new(), |mut unique, effort| {
                if !unique.contains(&effort) {
                    unique.push(effort);
                }
                unique
            });
            let default_effort = requested_default
                .filter(|effort| efforts.contains(effort))
                .or_else(|| efforts.first().cloned());
            ModelReasoningCapabilities {
                supported: !efforts.is_empty(),
                efforts,
                default_effort,
            }
        }
    };
    ModelCapabilities {
        vision: modalities.iter().any(|modality| modality == "vision"),
        tools: matches!(tools, CapabilitySupport::Supported { .. }),
        structured_output: matches!(structured_output, CapabilitySupport::Supported { .. }),
        reasoning,
    }
}
