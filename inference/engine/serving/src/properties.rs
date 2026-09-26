//! Model properties from the model's host artifacts. Host-only: the model is
//! never leased or loaded (integration spec §8.5, §9.5).
use axum::Json;
use axum::extract::{Path, State};
use serde::Serialize;
use utoipa::ToSchema;

use crate::Serving;
use crate::error::{ApiError, ErrorResponse, ServingError};
use magnitude_engine::chat::ModelProperties;

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PropsResponse {
    pub build_info: String,
    pub model_path: String,
    pub model_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub general_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub general_architecture: Option<String>,
    pub default_generation_settings: DefaultGenerationSettings,
    pub modalities: Modalities,
    pub chat_template: String,
    pub template_fingerprint: String,
    pub template_capabilities: TemplateCapabilitiesResponse,
    pub reasoning: ReasoningProfileResponse,
    pub training_context_tokens: u32,
    /// Sliding-window attention span; 0 when every layer attends over the
    /// full context, as in every family the engine executes.
    pub sliding_window_tokens: i32,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ReasoningProfileResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub default_reasoning_effort: Option<String>,
    pub reasoning_efforts: Vec<String>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct DefaultGenerationSettings {
    pub n_ctx: u32,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Modalities {
    pub vision: bool,
    pub audio: bool,
    pub video: bool,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct TemplateCapabilitiesResponse {
    pub string_content: bool,
    pub typed_content: bool,
    pub tools: bool,
    pub tool_calls: bool,
    pub parallel_tool_calls: bool,
    pub system_role: bool,
    pub preserve_reasoning: bool,
    pub object_arguments: bool,
    pub enable_thinking: bool,
}

/// Context sizes in the published 32-bit shape.
fn tokens(count: u64) -> Result<u32, ServingError> {
    u32::try_from(count)
        .map_err(|_| ServingError::Internal(format!("context of {count} tokens exceeds u32")))
}

pub fn props_response(properties: ModelProperties) -> Result<PropsResponse, ServingError> {
    let capability = |name: &str| {
        properties
            .template_capabilities
            .get(name)
            .copied()
            .ok_or_else(|| {
                ServingError::Internal(format!("native template capabilities lack {name}"))
            })
    };
    let template_capabilities = TemplateCapabilitiesResponse {
        string_content: capability("supports_string_content")?,
        typed_content: capability("supports_typed_content")?,
        tools: properties.tools,
        tool_calls: capability("supports_tool_calls")?,
        parallel_tool_calls: capability("supports_parallel_tool_calls")?,
        system_role: capability("supports_system_role")?,
        preserve_reasoning: properties.reasoning.supports_preserve_reasoning,
        object_arguments: capability("supports_object_arguments")?,
        enable_thinking: properties
            .reasoning
            .mappings
            .iter()
            .any(|mapping| mapping.controls.contains_key("enable_thinking")),
    };
    Ok(PropsResponse {
        build_info: format!("magnitude-engine {}", env!("CARGO_PKG_VERSION")),
        model_path: properties.model_path.display().to_string(),
        model_size_bytes: properties.model_size_bytes,
        general_name: properties.name,
        general_architecture: properties.architecture,
        default_generation_settings: DefaultGenerationSettings {
            n_ctx: tokens(properties.context_tokens)?,
        },
        modalities: Modalities {
            vision: properties.vision,
            audio: false,
            video: false,
        },
        chat_template: properties.chat_template,
        template_fingerprint: properties.template_fingerprint,
        template_capabilities,
        reasoning: ReasoningProfileResponse {
            default_reasoning_effort: properties.reasoning.default_effort.clone(),
            reasoning_efforts: properties
                .reasoning
                .mappings
                .iter()
                .map(|mapping| mapping.effort.clone())
                .collect(),
        },
        training_context_tokens: tokens(properties.training_context_tokens)?,
        sliding_window_tokens: 0,
    })
}

#[utoipa::path(post, path = "/api/v1/models/{model_id}/properties", operation_id = "getModelProperties", tag = "models",
    params(("model_id" = String, Path, description = "Canonical model ID")),
    responses(
        (status = 200, description = "Model and template properties from its installed material", body = PropsResponse),
        (status = 400, description = "Invalid model identity", body = ErrorResponse),
        (status = 404, description = "Model is not installed", body = ErrorResponse),
        (status = 409, description = "Installed model cannot be resolved for serving", body = ErrorResponse),
        (status = 422, description = "Installed model failed integrity validation", body = ErrorResponse),
        (status = 500, description = "Properties unavailable", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "serving.model_properties", skip_all, fields(model.id = %model_id), err(Debug))]
pub async fn props(
    State(state): State<Serving>,
    Path(model_id): Path<String>,
) -> Result<Json<PropsResponse>, ApiError> {
    let host = state.models.host(&model_id).await.map_err(ApiError::from)?;
    let properties = tokio::task::spawn_blocking(move || host.properties())
        .await
        .map_err(|error| ApiError::server(format!("properties task failed: {error}")))?
        .map_err(ApiError::from)?;
    Ok(Json(props_response(properties).map_err(ApiError::from)?))
}
