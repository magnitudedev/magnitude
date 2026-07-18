//! llama.cpp adapter for ICN's backend-independent inference API.

use std::cmp::max;
use std::num::NonZeroU32;
use std::sync::Mutex;

use icn_core::{
    ChatInferenceEngine, ChatMessage, ChatRequest, CompletionBackend, FinishReason,
    GenerateRequest, Generation, InferenceEngine, InferenceError, ModelConfig,
};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

/// A loaded GGUF model. `model` must be dropped before `_backend`.
pub struct LlamaCppEngine {
    model: LlamaModel,
    _backend: LlamaBackend,
    context_size: NonZeroU32,
    chat_template: LlamaChatTemplate,
}

impl LlamaCppEngine {
    pub fn load(config: ModelConfig) -> Result<Self, InferenceError> {
        if !config.model_path.is_file() {
            return Err(InferenceError::InvalidConfig(format!(
                "GGUF model does not exist: {}",
                config.model_path.display()
            )));
        }
        let context_size = NonZeroU32::new(config.context_size).ok_or_else(|| {
            InferenceError::InvalidConfig("context_size must be greater than zero".into())
        })?;
        let backend = LlamaBackend::init().map_err(backend_error)?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .map_err(backend_error)?;
        let chat_template = model
            .chat_template(None)
            .or_else(|_| LlamaChatTemplate::new("chatml"))
            .map_err(backend_error)?;
        Ok(Self {
            model,
            _backend: backend,
            context_size,
            chat_template,
        })
    }

    pub fn format_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String, InferenceError> {
        let messages = messages
            .iter()
            .map(|message| LlamaChatMessage::new(message.role.clone(), message.content.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(backend_error)?;
        self.model
            .apply_chat_template(&self.chat_template, &messages, true)
            .map_err(backend_error)
    }
}

impl InferenceEngine for LlamaCppEngine {
    fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        if !request.temperature.is_finite() || request.temperature < 0.0 {
            return Err(InferenceError::InvalidConfig(
                "temperature must be finite and non-negative".into(),
            ));
        }
        if !request.top_p.is_finite() || !(0.0..=1.0).contains(&request.top_p) {
            return Err(InferenceError::InvalidConfig(
                "top_p must be finite and between 0 and 1".into(),
            ));
        }
        let tokens = self
            .model
            .str_to_token(&request.prompt, AddBos::Always)
            .map_err(backend_error)?;
        let prompt_tokens = tokens.len();
        let required_tokens = prompt_tokens
            .checked_add(request.max_new_tokens as usize)
            .ok_or_else(|| InferenceError::InvalidConfig("token count overflow".into()))?;
        if required_tokens > self.context_size.get() as usize {
            return Err(InferenceError::InvalidConfig(format!(
                "prompt ({prompt_tokens} tokens) + output ({} tokens) exceeds context size ({})",
                request.max_new_tokens, self.context_size
            )));
        }

        let context_params = LlamaContextParams::default().with_n_ctx(Some(self.context_size));
        let mut context = self
            .model
            .new_context(&self._backend, context_params)
            .map_err(backend_error)?;
        let mut batch = LlamaBatch::new(max(prompt_tokens, 1), 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(backend_error)?;
        context.decode(&mut batch).map_err(backend_error)?;
        let mut sampler = if request.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_p(request.top_p, 1),
                LlamaSampler::temp(request.temperature),
                LlamaSampler::dist(request.seed),
            ])
        };
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut generated_tokens = 0;
        let mut finish_reason = FinishReason::Length;
        let mut position = i32::try_from(prompt_tokens)
            .map_err(|error| InferenceError::InvalidConfig(error.to_string()))?;

        for token_index in 0..request.max_new_tokens {
            let token = sampler.sample(&context, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                finish_reason = FinishReason::Stop;
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(backend_error)?;
            output.push_str(&piece);
            generated_tokens += 1;
            on_token(&piece)?;
            if token_index + 1 < request.max_new_tokens {
                batch.clear();
                batch
                    .add(token, position, &[0], true)
                    .map_err(backend_error)?;
                context.decode(&mut batch).map_err(backend_error)?;
                position += 1;
            }
        }
        Ok(Generation {
            text: output,
            prompt_tokens,
            generated_tokens,
            finish_reason,
        })
    }
}

impl ChatInferenceEngine for LlamaCppEngine {
    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        let prompt = self.format_chat_prompt(messages)?;
        self.generate(
            &GenerateRequest {
                prompt,
                ..request.clone()
            },
            on_token,
        )
    }
}

pub struct LlamaCompletionBackend {
    model_id: String,
    engine: Mutex<LlamaCppEngine>,
}

impl LlamaCompletionBackend {
    pub fn new(model_id: impl Into<String>, engine: LlamaCppEngine) -> Self {
        Self {
            model_id: model_id.into(),
            engine: Mutex::new(engine),
        }
    }
}

impl CompletionBackend for LlamaCompletionBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(
        &self,
        request: ChatRequest,
        on_token: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| InferenceError::Backend("model executor lock was poisoned".into()))?;
        engine.generate_chat(
            &request.messages,
            &GenerateRequest {
                prompt: String::new(),
                max_new_tokens: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
                seed: request.seed,
            },
            on_token,
        )
    }
}

fn backend_error(error: impl std::fmt::Display) -> InferenceError {
    InferenceError::Backend(error.to_string())
}
