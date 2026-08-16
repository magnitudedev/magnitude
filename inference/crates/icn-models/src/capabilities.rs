use icn_contracts::models::{ModelCapabilities, ModelReasoningCapabilities};
use icn_contracts::{CapabilitySupport, InventoryProperties, ReasoningCapability};

fn supports_image_input(modalities: &[String]) -> bool {
    modalities.iter().any(|modality| modality == "image")
}

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
            match requested_default.filter(|effort| efforts.contains(effort)) {
                Some(default_effort) if !efforts.is_empty() => ModelReasoningCapabilities {
                    supported: true,
                    efforts,
                    default_effort: Some(default_effort),
                },
                _ => ModelReasoningCapabilities {
                    supported: false,
                    efforts: Vec::new(),
                    default_effort: None,
                },
            }
        }
    };
    ModelCapabilities {
        vision: supports_image_input(modalities),
        tools: matches!(tools, CapabilitySupport::Supported { .. }),
        structured_output: matches!(structured_output, CapabilitySupport::Supported { .. }),
        reasoning,
    }
}

#[cfg(test)]
mod tests {
    use super::supports_image_input;

    #[test]
    fn image_inventory_modality_maps_to_vision_capability() {
        assert!(supports_image_input(&[
            "text".to_owned(),
            "image".to_owned()
        ]));
        assert!(!supports_image_input(&["text".to_owned()]));
    }
}
