use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub context_size: u32,
    pub gpu_layers: u32,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("model backend failed: {0}")]
    Backend(String),
    #[error("token callback failed: {0}")]
    Callback(String),
}

pub trait InferenceEngine {
    fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError>;
}

pub trait ChatInferenceEngine {
    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError>;
}

pub trait CompletionBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;
    fn complete(
        &self,
        request: ChatRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError>;
}
