//! Production-owned primitive operations used by the external parity harness.
//!
//! The JSONL transport lives in `icn-parity-probe`. Operation ownership stays
//! here so parity exercises the same safe bindings and projections as ICN.

use std::fmt::{Display, Formatter};
use std::fs;
use std::num::{NonZeroI32, NonZeroU32};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use icn_contracts::{ExecutionConfig, FlashAttention, GpuLayers, ResolvedExecutionPlan};
use llama_cpp_2::common_chat::{
    ChatContent, ChatContinuation, ChatFormat, ChatGrammarTrigger, ChatMessage, ChatParserOptions,
    ChatPrepareOptions, ChatReasoningFormat, ChatSemanticDelta, ChatTemplateKwarg, ChatTool,
    ChatToolChoice, CommonChatTemplates, ParsedChatMessage, PreparedChat,
};
use llama_cpp_2::common_sampling::{
    CommonReasoningBudget, CommonSampler, CommonSamplerConfig, ReasoningBudgetLimit,
};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{FlashAttentionPolicy, LlamaContextParams};
use llama_cpp_2::context::session::LlamaStateSeqFlags;
use llama_cpp_2::llama_backend::{LlamaBackend, LlamaThreadPool, LlamaThreadPoolParams};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, RopeType, VocabType};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use llama_cpp_2::{
    LlamaBackendDeviceType, LogOptions, list_llama_ggml_backend_devices, send_logs_to_tracing,
};

const OPERATIONS: [&str; 22] = [
    "protocol.describe",
    "configuration.inspect",
    "model.metadata",
    "tokenizer.tokenize",
    "tokenizer.token-to-piece",
    "chat-template.render",
    "grammar.json-schema-to-grammar",
    "sampler.apply",
    "decode.execute-plan",
    "state.execute-script",
    "llama-bench.prompt-processing",
    "llama-bench.token-generation",
    "llama-bench.prompt-generation",
    "llama-bench.context-depth",
    "llama-batched-bench.sequence-throughput",
    "sampler.bench",
    "chat-template.bench",
    "chat-parser.inspect",
    "chat-parser.bench",
    "reasoning-budget.inspect",
    "decode.abort",
    "decode.abort-recovery",
];

/// A classified primitive-probe failure. The process adapter projects this to
/// the neutral JSONL error envelope without learning operation semantics.
#[derive(Debug)]
pub struct ProbeError {
    class: &'static str,
    code: &'static str,
    message: String,
}

impl ProbeError {
    fn invalid(code: &'static str, message: impl Display) -> Self {
        Self {
            class: "invalid-input",
            code,
            message: message.to_string(),
        }
    }

    fn runtime(code: &'static str, error: impl Display) -> Self {
        Self {
            class: "runtime-error",
            code,
            message: error.to_string(),
        }
    }

    /// Neutral outcome class used by the JSONL adapter.
    #[must_use]
    pub const fn class(&self) -> &'static str {
        self.class
    }

    /// Stable machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }
}

impl Display for ProbeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProbeError {}

/// Execute one neutral primitive operation through ICN-owned code.
pub fn execute(operation: &str, input: &Map<String, Value>) -> Result<Value, ProbeError> {
    match operation {
        "protocol.describe" => Ok(json!({
            "protocolVersion": 1,
            "transport": "jsonl-stdin-stdout",
            "operations": OPERATIONS,
        })),
        "configuration.inspect" => configuration_inspect(input),
        "model.metadata" => model_metadata(input),
        "tokenizer.tokenize" => tokenizer_tokenize(input),
        "tokenizer.token-to-piece" => tokenizer_token_to_piece(input),
        "chat-template.render" => chat_template_render(input),
        "grammar.json-schema-to-grammar" => json_schema_to_grammar(input),
        "sampler.apply" => sampler_apply(input),
        "sampler.bench" => sampler_bench(input),
        "chat-template.bench" => chat_template_bench(input),
        "chat-parser.inspect" => chat_parser_inspect(input),
        "chat-parser.bench" => chat_parser_bench(input),
        "reasoning-budget.inspect" => reasoning_budget_inspect(input),
        "decode.abort" => decode_abort(input),
        "decode.abort-recovery" => decode_abort_recovery(input),
        "decode.execute-plan" => decode_execute_plan(input),
        "state.execute-script" => state_execute_script(input),
        operation @ ("llama-bench.prompt-processing"
        | "llama-bench.token-generation"
        | "llama-bench.prompt-generation"
        | "llama-bench.context-depth") => llama_bench(operation, input),
        "llama-batched-bench.sequence-throughput" => llama_batched_bench(input),
        _ => Err(ProbeError::invalid(
            "unsupported-operation",
            format!("unsupported operation: {operation}"),
        )),
    }
}

fn decode_input<T: DeserializeOwned>(input: &Map<String, Value>) -> Result<T, ProbeError> {
    serde_json::from_value(Value::Object(input.clone()))
        .map_err(|error| ProbeError::invalid("invalid-case", error))
}

fn configuration_inspect(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: ConfigurationInspectInput = decode_input(input)?;
    validate_context_request(&input.context)?;
    let loaded = load_model(&input.model_path, false)?;
    let threads = nonzero_i32(input.context.threads, "threads")?;
    let batch_threads = nonzero_i32(input.context.batch_threads, "batch_threads")?;
    let execution = ExecutionConfig {
        offload_kqv: input.context.offload_kqv,
        threads: NonZeroU32::new(input.context.threads),
        threads_batch: NonZeroU32::new(input.context.batch_threads),
        ..ExecutionConfig::default()
    };
    let config = ResolvedExecutionPlan {
        model_path: input.model_path.clone(),
        context_size: input.context.context_tokens,
        batch_size: input.context.batch_tokens,
        ubatch_size: input.context.micro_batch_tokens,
        max_sequences: input.context.sequences,
        prefill_quantum: input.context.batch_tokens,
        radix_cache: Default::default(),
        execution,
        projector: None,
        mtp: icn_contracts::MtpConfig::default(),
    };
    let params = super::native_context_params(&config, threads, batch_threads);
    let context = loaded
        .model
        .new_context(&loaded.backend, params)
        .map_err(|error| ProbeError::runtime("context-create", error))?;

    Ok(json!({
        "effective": {
            "contextTokens": context.n_ctx(),
            "contextTokensPerSequence": context.n_ctx_seq(),
            "batchTokens": context.n_batch(),
            "microBatchTokens": context.n_ubatch(),
            "sequences": context.n_seq_max(),
            "recurrentSequences": context.n_rs_seq(),
            "threads": context.n_threads(),
            "batchThreads": context.n_threads_batch(),
        }
    }))
}

fn model_metadata(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: ModelMetadataInput = decode_input(input)?;
    let loaded = load_model(&input.model_path, false)?;
    let model = &loaded.model;

    Ok(json!({
        "sizeBytes": model.size(),
        "parameterCount": model.n_params(),
        "dimensions": {
            "contextTrain": model.n_ctx_train(),
            "embedding": model.n_embd(),
            "layers": model.n_layer(),
            "attentionHeads": model.n_head(),
            "kvAttentionHeads": model.n_head_kv(),
            "slidingWindow": model.n_swa(),
        },
        "architecture": {
            "ropeType": rope_type_name(model.rope_type()),
            "recurrent": model.is_recurrent(),
            "hybrid": model.is_hybrid(),
        },
        "vocabulary": {
            "type": vocab_type_name(model.vocab_type()),
            "tokenCount": model.n_vocab(),
            "addBos": model.should_add_bos(),
            "specialTokens": {
                "bos": model.token_bos().0,
                "eos": model.token_eos().0,
                "newline": model.token_nl().0,
            },
        },
        "metadataCount": model.meta_count(),
    }))
}

fn tokenizer_tokenize(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: TokenizeInput = decode_input(input)?;
    if !input.parse_special {
        return Err(ProbeError::invalid(
            "unsupported-parse-special",
            "the production safe tokenizer requires parseSpecial=true",
        ));
    }
    if input.text.len() > i32::MAX as usize {
        return Err(ProbeError::invalid(
            "text-too-large",
            "text byte length exceeds i32::MAX",
        ));
    }
    let loaded = load_model_with_options(&input.model_path, true, &input.load_options())?;
    let tokens = loaded
        .model
        .str_to_token(
            &input.text,
            if input.add_special {
                AddBos::Always
            } else {
                AddBos::Never
            },
        )
        .map_err(|error| ProbeError::runtime("tokenize", error))?;
    let mut output = json!({
        "inputBytes": input.text.as_bytes(),
        "tokens": tokens.iter().map(|token| token.0).collect::<Vec<_>>(),
        "addSpecial": input.add_special,
        "parseSpecial": true,
    });
    if input.include_pieces {
        let pieces = tokens
            .iter()
            .map(|token| {
                Ok(json!({
                    "token": token.0,
                    "bytes": token_piece_bytes(&loaded.model, *token, input.piece_special)?,
                }))
            })
            .collect::<Result<Vec<_>, ProbeError>>()?;
        output["pieces"] = json!(pieces);
        output["pieceSpecial"] = json!(input.piece_special);
    }
    Ok(output)
}

fn tokenizer_token_to_piece(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: TokenToPieceInput = decode_input(input)?;
    if !input.special {
        return Err(ProbeError::invalid(
            "unsupported-special-mode",
            "the production token-to-piece boundary requires special=true",
        ));
    }
    if input.tokens.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-tokens",
            "tokens must not be empty",
        ));
    }
    if !input.lstrip.is_null() {
        return Err(ProbeError::invalid(
            "unsupported-lstrip",
            "the production token-to-piece boundary requires lstrip=null (Rust None)",
        ));
    }

    let loaded = load_model(&input.model_path, true)?;
    validate_token_ids(input.tokens.iter().copied(), loaded.model.n_vocab())?;

    let mut concatenated_bytes = Vec::new();
    let pieces = input
        .tokens
        .iter()
        .copied()
        .map(|token| {
            let bytes = token_piece_bytes(&loaded.model, LlamaToken::new(token), true)?;
            concatenated_bytes.extend_from_slice(&bytes);
            Ok(bytes)
        })
        .collect::<Result<Vec<_>, ProbeError>>()?;

    Ok(json!({
        "pieces": pieces,
        "concatenatedBytes": concatenated_bytes,
    }))
}

fn reasoning_budget_inspect(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: ReasoningBudgetInspectInput = decode_input(input)?;
    if input.budget_tokens == 0 || input.budget_tokens > i32::MAX.cast_unsigned() {
        return Err(ProbeError::invalid(
            "invalid-reasoning-budget",
            "budgetTokens must be in [1, i32::MAX]",
        ));
    }
    if !input.controllable {
        return Err(ProbeError::invalid(
            "invalid-reasoning-budget",
            "the reasoning-budget inspection boundary requires controllable=true",
        ));
    }
    if input.start_tag.is_empty() || input.end_tag.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-reasoning-budget",
            "startTag and endTag must not be empty",
        ));
    }

    let loaded = load_model(&input.model_path, true)?;
    let start_tokens = tokenize_reasoning_text(&loaded.model, &input.start_tag, "startTag")?;
    let end_tokens = tokenize_reasoning_text(&loaded.model, &input.end_tag, "endTag")?;
    let forced_text = format!("{}{}", input.forced_message, input.end_tag);
    let forced_tokens = tokenize_reasoning_text(&loaded.model, &forced_text, "forced sequence")?;

    let mut sampler = CommonSampler::new(
        &loaded.model,
        &CommonSamplerConfig {
            reasoning_budget: Some(CommonReasoningBudget {
                limit: ReasoningBudgetLimit::Tokens(input.budget_tokens),
                start_tag: input.start_tag,
                end_tag: input.end_tag,
                forced_message: input.forced_message,
                controllable: true,
            }),
            ..CommonSamplerConfig::default()
        },
    )
    .map_err(|error| ProbeError::runtime("reasoning-budget-init", error))?;

    let force_before_start = sampler.force_reasoning_end();
    accept_generated_tokens(&mut sampler, &start_tokens)?;
    let last_token_after_start = sampler.last_token().map(|token| token.0);
    let force_after_start = sampler.force_reasoning_end();
    let force_while_forcing = sampler.force_reasoning_end();
    accept_generated_tokens(&mut sampler, &forced_tokens)?;
    let last_token_after_forced_sequence = sampler.last_token().map(|token| token.0);
    let force_after_completion = sampler.force_reasoning_end();

    if force_before_start || !force_after_start || force_while_forcing || force_after_completion {
        return Err(ProbeError::runtime(
            "unexpected-reasoning-budget-observation",
            "the public reasoning-budget force observations did not follow idle/counting/forcing/completed semantics",
        ));
    }

    Ok(json!({
        "tokenized": {
            "startTokens": token_ids(&start_tokens),
            "endTokens": token_ids(&end_tokens),
            "forcedTokens": token_ids(&forced_tokens),
        },
        "observations": {
            "forceBeforeStart": force_before_start,
            "lastTokenAfterStart": last_token_after_start,
            "forceAfterStart": force_after_start,
            "forceWhileForcing": force_while_forcing,
            "lastTokenAfterForcedSequence": last_token_after_forced_sequence,
            "forceAfterCompletion": force_after_completion,
        },
    }))
}

fn tokenize_reasoning_text(
    model: &LlamaModel,
    text: &str,
    field: &'static str,
) -> Result<Vec<LlamaToken>, ProbeError> {
    let tokens = model
        .str_to_token(text, AddBos::Never)
        .map_err(|error| ProbeError::runtime("reasoning-budget-tokenize", error))?;
    if tokens.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-reasoning-budget",
            format!("{field} must tokenize to at least one model token"),
        ));
    }
    Ok(tokens)
}

fn accept_generated_tokens(
    sampler: &mut CommonSampler<'_>,
    tokens: &[LlamaToken],
) -> Result<(), ProbeError> {
    for token in tokens {
        sampler
            .accept_generated(*token)
            .map_err(|error| ProbeError::runtime("reasoning-budget-accept", error))?;
    }
    Ok(())
}

fn token_ids(tokens: &[LlamaToken]) -> Vec<i32> {
    tokens.iter().map(|token| token.0).collect()
}

fn json_schema_to_grammar(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: JsonSchemaGrammarInput = decode_input(input)?;
    if input.force_gbnf {
        return Err(ProbeError::invalid(
            "unsupported-force-gbnf",
            "the production safe binding supports forceGbnf=false only",
        ));
    }
    let schema = serde_json::to_string(&input.schema)
        .map_err(|error| ProbeError::invalid("invalid-json-schema", error))?;
    let grammar = llama_cpp_2::json_schema_to_grammar(&schema)
        .map_err(|error| ProbeError::runtime("json-schema-to-grammar", error))?;
    Ok(json!({"grammar": grammar, "forceGbnf": false}))
}

fn decode_abort(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: DecodeAbortInput = decode_input(input)?;
    validate_abort_context(
        input.n_gpu_layers,
        input.use_mmap,
        input.use_mlock,
        input.context_tokens,
        input.offload_kqv,
        input.operation_offload,
        &input.flash_attention,
        input.threads,
        input.batch_threads,
    )?;

    let loaded = load_model_with_options(&input.model_path, false, &input.load_options())?;
    validate_token_ids(std::iter::once(input.token), loaded.model.n_vocab())?;
    let params = abort_context_params(input.context_tokens, input.threads, input.batch_threads);
    let mut context = loaded
        .model
        .new_context(&loaded.backend, params)
        .map_err(|error| ProbeError::runtime("context-create", error))?;

    execute_presignalled_abort(&mut context, input.token)?;
    Ok(json!({"class": "aborted"}))
}

fn decode_abort_recovery(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: DecodeAbortRecoveryInput = decode_input(input)?;
    validate_abort_context(
        input.n_gpu_layers,
        input.use_mmap,
        input.use_mlock,
        input.context_tokens,
        input.offload_kqv,
        input.operation_offload,
        &input.flash_attention,
        input.threads,
        input.batch_threads,
    )?;
    if input.logit_indices.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-logit-indices",
            "logitIndices must not be empty",
        ));
    }

    let loaded = load_model_with_options(&input.model_path, false, &input.load_options())?;
    validate_token_ids(std::iter::once(input.token), loaded.model.n_vocab())?;
    validate_logit_indices(&input.logit_indices, loaded.model.n_vocab())?;
    let params = abort_context_params(input.context_tokens, input.threads, input.batch_threads);
    let mut context = loaded
        .model
        .new_context(&loaded.backend, params)
        .map_err(|error| ProbeError::runtime("context-create", error))?;

    execute_presignalled_abort(&mut context, input.token)?;
    // execute_presignalled_abort synchronized every backend and removed its
    // callback. Complete the production recovery boundary by fully discarding
    // logical memory state without zero-filling the backing allocation.
    context.clear_memory(false);

    let mut recovery_batch = single_token_batch(input.token)?;
    context
        .decode(&mut recovery_batch)
        .map_err(|error| ProbeError::runtime("recovery-decode", error))?;
    context.synchronize();
    let logits = selected_logits(context.get_logits_ith(0), &input.logit_indices);

    Ok(json!({
        "abort": {"class": "aborted"},
        "recovery": {
            "status": {"code": 0, "class": "success"},
            "logits": logits,
        },
    }))
}

#[allow(clippy::too_many_arguments)]
fn validate_abort_context(
    n_gpu_layers: u32,
    use_mmap: bool,
    use_mlock: bool,
    context_tokens: u32,
    offload_kqv: bool,
    operation_offload: bool,
    flash_attention: &str,
    threads: u32,
    batch_threads: u32,
) -> Result<(), ProbeError> {
    if n_gpu_layers != 0 || !use_mmap || use_mlock {
        return Err(ProbeError::invalid(
            "invalid-model-configuration",
            "CPU cancellation probes require nGpuLayers=0, useMmap=true, and useMlock=false",
        ));
    }
    if context_tokens == 0 || context_tokens > i32::MAX.cast_unsigned() {
        return Err(ProbeError::invalid(
            "invalid-context",
            "contextTokens must be in [1, i32::MAX]",
        ));
    }
    validate_explicit_cpu_context(
        offload_kqv,
        operation_offload,
        flash_attention,
        threads,
        batch_threads,
    )
}

fn abort_context_params(
    context_tokens: u32,
    threads: u32,
    batch_threads: u32,
) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(Some(
            NonZeroU32::new(context_tokens).expect("validated context size is non-zero"),
        ))
        .with_n_seq_max(1)
        .with_n_threads(i32::try_from(threads).expect("validated threads fit i32"))
        .with_n_threads_batch(
            i32::try_from(batch_threads).expect("validated batch threads fit i32"),
        )
        .with_offload_kqv(false)
        .with_op_offload(false)
        .with_flash_attention(FlashAttentionPolicy::Disabled)
}

fn execute_presignalled_abort(
    context: &mut LlamaContext<'_>,
    token: i32,
) -> Result<(), ProbeError> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let handle = context.install_abort_callback_with_flag(cancelled);
    handle.cancel();
    let mut batch = single_token_batch(token)?;
    let result = context.decode(&mut batch);
    context.synchronize();
    context.clear_abort_callback();

    match result {
        Err(llama_cpp_2::DecodeError::Aborted) => Ok(()),
        Ok(()) => Err(ProbeError::runtime(
            "abort-not-observed",
            "the pre-signalled CPU decode completed instead of reporting Aborted",
        )),
        Err(error) => Err(ProbeError::runtime(
            "unexpected-abort-status",
            format!("the pre-signalled CPU decode returned {error:?} instead of Aborted"),
        )),
    }
}

fn single_token_batch(token: i32) -> Result<LlamaBatch<'static>, ProbeError> {
    let mut batch = LlamaBatch::new(1, 1);
    batch
        .add(LlamaToken::new(token), 0, &[0], true)
        .map_err(|error| ProbeError::runtime("batch-add", error))?;
    Ok(batch)
}

fn decode_execute_plan(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: DecodePlanInput = decode_input(input)?;
    validate_decode_plan(&input)?;
    validate_explicit_cpu_context(
        input.offload_kqv,
        input.operation_offload,
        &input.flash_attention,
        input.threads,
        input.batch_threads,
    )?;
    let loaded = load_model_with_options(&input.model_path, false, &input.load_options())?;
    validate_logit_indices(&input.logit_indices, loaded.model.n_vocab())?;
    validate_token_ids(
        input.batch.iter().map(|item| item.token),
        loaded.model.n_vocab(),
    )?;
    let sequence_count = sequence_count_for(
        input
            .batch
            .iter()
            .flat_map(|item| item.sequence_ids.iter().copied()),
    )?;
    let context_size =
        NonZeroU32::new(input.context_tokens).expect("validated context size is non-zero");
    let params = LlamaContextParams::default()
        .with_n_ctx(Some(context_size))
        .with_n_seq_max(sequence_count)
        .with_n_threads(i32::try_from(input.threads).expect("validated threads fit i32"))
        .with_n_threads_batch(
            i32::try_from(input.batch_threads).expect("validated batch threads fit i32"),
        )
        .with_offload_kqv(input.offload_kqv)
        .with_op_offload(input.operation_offload)
        .with_flash_attention(FlashAttentionPolicy::Disabled);
    let mut context = loaded
        .model
        .new_context(&loaded.backend, params)
        .map_err(|error| ProbeError::runtime("context-create", error))?;
    if input.batch.len() > context.n_batch() as usize {
        return Err(ProbeError::invalid(
            "batch-too-large",
            format!(
                "batch has {} tokens but the effective context permits {}",
                input.batch.len(),
                context.n_batch()
            ),
        ));
    }
    if input.batch.len() > context.n_ctx() as usize {
        return Err(ProbeError::invalid(
            "batch-too-large",
            "batch item indices must fit inside the effective context",
        ));
    }
    let max_sequences_per_token = input
        .batch
        .iter()
        .map(|item| item.sequence_ids.len())
        .max()
        .unwrap_or(1);
    let max_sequences_per_token = i32::try_from(max_sequences_per_token).map_err(|_| {
        ProbeError::invalid("too-many-sequences", "sequence ID list exceeds i32::MAX")
    })?;
    let mut batch = LlamaBatch::new(input.batch.len(), max_sequences_per_token);
    for item in &input.batch {
        batch
            .add(
                LlamaToken::new(item.token),
                item.position,
                &item.sequence_ids,
                item.request_logits,
            )
            .map_err(|error| ProbeError::runtime("batch-add", error))?;
    }
    context
        .decode(&mut batch)
        .map_err(|error| ProbeError::runtime("decode", error))?;
    context.synchronize();

    let logits = input
        .batch
        .iter()
        .enumerate()
        .filter(|(_, item)| item.request_logits)
        .map(|(batch_index, _)| {
            let batch_index_i32 =
                i32::try_from(batch_index).expect("validated batch indices fit into i32");
            json!({
                "batchIndex": batch_index,
                "values": selected_logits(
                    context.get_logits_ith(batch_index_i32),
                    &input.logit_indices,
                ),
            })
        })
        .collect::<Vec<_>>();
    let plan = input
        .batch
        .iter()
        .map(|item| {
            json!({
                "token": item.token,
                "position": item.position,
                "sequenceIds": item.sequence_ids,
                "requestLogits": item.request_logits,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "plan": plan,
        "status": {"code": 0, "class": "success"},
        "logits": logits,
    }))
}

fn state_execute_script(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: StateScriptInput = decode_input(input)?;
    validate_state_script(&input)?;
    validate_explicit_cpu_context(
        input.offload_kqv,
        input.operation_offload,
        &input.flash_attention,
        input.threads,
        input.batch_threads,
    )?;
    let loaded = load_model_with_options(&input.model_path, false, &input.load_options())?;
    validate_token_ids(
        input
            .prepare_tokens
            .iter()
            .copied()
            .chain(
                input
                    .operations
                    .iter()
                    .filter_map(|operation| match operation {
                        StateOperationInput::Decode { token, .. } => Some(*token),
                        StateOperationInput::Remove { .. } => None,
                    }),
            ),
        loaded.model.n_vocab(),
    )?;
    validate_logit_indices(&input.logit_indices, loaded.model.n_vocab())?;
    let context_tokens = derived_state_context_tokens(&input)?;
    let sequence_count = u32::try_from(input.sequence_id)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            ProbeError::invalid(
                "invalid-sequence-id",
                "sequenceId must be non-negative and below u32::MAX",
            )
        })?;
    let params = LlamaContextParams::default()
        .with_n_ctx(Some(
            NonZeroU32::new(context_tokens).expect("derived context size is non-zero"),
        ))
        .with_n_seq_max(sequence_count)
        .with_n_threads(i32::try_from(input.threads).expect("validated threads fit i32"))
        .with_n_threads_batch(
            i32::try_from(input.batch_threads).expect("validated batch threads fit i32"),
        )
        .with_offload_kqv(input.offload_kqv)
        .with_op_offload(input.operation_offload)
        .with_flash_attention(FlashAttentionPolicy::Disabled);
    let mut context = loaded
        .model
        .new_context(&loaded.backend, params)
        .map_err(|error| ProbeError::runtime("context-create", error))?;
    if input.prepare_tokens.len() > context.n_batch() as usize {
        return Err(ProbeError::invalid(
            "batch-too-large",
            "prepareTokens exceed the effective batch capacity",
        ));
    }
    if !input.prepare_tokens.is_empty() {
        let mut prepare = LlamaBatch::new(input.prepare_tokens.len(), 1);
        for (position, token) in input.prepare_tokens.iter().copied().enumerate() {
            prepare
                .add(
                    LlamaToken::new(token),
                    i32::try_from(position).expect("validated prepare position fits into i32"),
                    &[input.sequence_id],
                    position + 1 == input.prepare_tokens.len(),
                )
                .map_err(|error| ProbeError::runtime("batch-add", error))?;
        }
        context
            .decode(&mut prepare)
            .map_err(|error| ProbeError::runtime("prepare-decode", error))?;
        context.synchronize();
    }

    let mut operations = Vec::with_capacity(input.operations.len());
    let mut logits = Vec::new();
    for (index, operation) in input.operations.iter().enumerate() {
        match operation {
            StateOperationInput::Remove {
                position_start,
                position_end,
            } => {
                context.synchronize();
                let success = context
                    .clear_kv_cache_seq(
                        Some(input.sequence_id.cast_unsigned()),
                        Some(*position_start),
                        Some(*position_end),
                    )
                    .map_err(|error| ProbeError::runtime("state-remove", error))?;
                operations.push(json!({
                    "index": index,
                    "type": "remove",
                    "positionStart": position_start,
                    "positionEnd": position_end,
                    "success": success,
                }));
            }
            StateOperationInput::Decode {
                token,
                position,
                request_logits,
            } => {
                let mut batch = LlamaBatch::new(1, 1);
                batch
                    .add(
                        LlamaToken::new(*token),
                        *position,
                        &[input.sequence_id],
                        *request_logits,
                    )
                    .map_err(|error| ProbeError::runtime("batch-add", error))?;
                context
                    .decode(&mut batch)
                    .map_err(|error| ProbeError::runtime("decode", error))?;
                context.synchronize();
                operations.push(json!({
                    "index": index,
                    "type": "decode",
                    "token": token,
                    "position": position,
                    "requestLogits": request_logits,
                    "status": {"code": 0, "class": "success"},
                }));
                if *request_logits {
                    logits.push(json!({
                        "operationIndex": index,
                        "values": selected_logits(context.get_logits_ith(0), &input.logit_indices),
                    }));
                }
            }
        }
    }
    context.synchronize();
    let state = json!({
        "sequenceId": input.sequence_id,
        "positionMin": context.memory_seq_pos_min(input.sequence_id),
        "positionMax": context.kv_cache_seq_pos_max(input.sequence_id),
    });

    Ok(json!({
        "operations": operations,
        "state": state,
        "logits": logits,
    }))
}

fn llama_batched_bench(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: LlamaBatchedBenchInput = decode_input(input)?;
    input.validate()?;
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    let threads = nonzero_i32(input.engine_configuration.threads, "threads")?;
    let flash_attention = match input.engine_configuration.flash_attention.as_str() {
        "auto" => FlashAttention::Auto,
        "on" => FlashAttention::Enabled,
        "off" => FlashAttention::Disabled,
        value => {
            return Err(ProbeError::invalid(
                "invalid-engine-configuration",
                format!("unsupported flashAttention policy: {value}"),
            ));
        }
    };
    let flash_attention_value = match flash_attention {
        FlashAttention::Auto => -1,
        FlashAttention::Disabled => 0,
        FlashAttention::Enabled => 1,
    };
    let (gpu_layers, normalized_gpu_layers) = input.engine_configuration.gpu_layers.normalized()?;
    let batched_bench_gpu_layers = input
        .engine_configuration
        .gpu_layers
        .batched_bench_value()?;
    let execution = ExecutionConfig {
        gpu_layers,
        threads: NonZeroU32::new(input.engine_configuration.threads),
        threads_batch: NonZeroU32::new(input.engine_configuration.threads),
        flash_attention,
        kv_unified: input.kv_unified,
        ..ExecutionConfig::default()
    };
    let config = ResolvedExecutionPlan {
        model_path: input.model_path.clone(),
        context_size: input.context_tokens,
        batch_size: input.batch_tokens,
        ubatch_size: input.micro_batch_tokens,
        max_sequences: input.parallel_sequences,
        prefill_quantum: input.batch_tokens,
        radix_cache: Default::default(),
        execution,
        projector: None,
        mtp: icn_contracts::MtpConfig::default(),
    };
    if !config.model_path.is_file() {
        return Err(ProbeError::invalid(
            "invalid-model-path",
            format!(
                "modelPath is not a regular file: {}",
                config.model_path.display()
            ),
        ));
    }

    let backend =
        LlamaBackend::init().map_err(|error| ProbeError::runtime("backend-initialize", error))?;
    let model_params = super::native_model_params(&config.execution)
        .map_err(|error| ProbeError::invalid("invalid-engine-configuration", error))?;
    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .map_err(|error| ProbeError::runtime("model-load", error))?;
    // The official batched-bench leaves n_outputs_max at zero (n_batch) and uses the context's
    // default CPU backend pool. Preserve both details here; production server construction is
    // compared separately by the endpoint benchmark.
    let context_params =
        super::native_context_params(&config, threads, threads).with_n_outputs_max(None);
    let mut context = model
        .new_context(&backend, context_params)
        .map_err(|error| ProbeError::runtime("context-create", error))?;
    let vocabulary_size = model.n_vocab();
    if vocabulary_size <= 0 {
        return Err(ProbeError::runtime(
            "invalid-vocabulary",
            "loaded model reports no vocabulary tokens",
        ));
    }
    let (_, _, devices) = benchmark_hardware_identity()?;

    // Upstream performs one 16-token synchronized warmup before clearing logical memory for the
    // measured row.
    let mut warmup = LlamaBatch::new(16, 1);
    for position in 0..16_i32 {
        warmup
            .add(
                next_llama_bench_token(vocabulary_size),
                position,
                &[0],
                false,
            )
            .map_err(|error| ProbeError::runtime("warmup-batch", error))?;
    }
    context
        .decode(&mut warmup)
        .map_err(|error| ProbeError::runtime("warmup-decode", error))?;
    context.synchronize();
    context.clear_memory(false);

    // Upstream constructs the complete prompt batch before starting its prompt
    // timer, then presents n_batch-sized views to decode. Prepare equivalent
    // chunks here so Rust allocation and batch population are excluded too.
    let mut prompt_batches = prepare_batched_prompt(&input, vocabulary_size)?;
    let prompt_started = Instant::now();
    for batch in &mut prompt_batches {
        context
            .decode(batch)
            .map_err(|error| ProbeError::runtime("prompt-decode", error))?;
    }
    context.synchronize();
    let prompt_seconds = prompt_started.elapsed().as_secs_f64();

    if input.shared_prompt {
        for sequence_id in 1..input.parallel_sequences {
            context
                .copy_kv_cache_seq(
                    0,
                    i32::try_from(sequence_id).expect("validated sequence"),
                    None,
                    None,
                )
                .map_err(|error| ProbeError::runtime("shared-prompt-copy", error))?;
        }
        if !input.kv_unified {
            let mut dummy = LlamaBatch::new(1, 1);
            dummy
                .add(
                    next_llama_bench_token(vocabulary_size),
                    i32::try_from(input.prompt_tokens).expect("validated prompt length"),
                    &[0],
                    true,
                )
                .map_err(|error| ProbeError::runtime("shared-prompt-dummy-batch", error))?;
            context
                .decode(&mut dummy)
                .map_err(|error| ProbeError::runtime("shared-prompt-dummy-decode", error))?;
            context.synchronize();
            context
                .clear_kv_cache_seq(None, Some(input.prompt_tokens), None)
                .map_err(|error| ProbeError::runtime("shared-prompt-dummy-remove", error))?;
        }
    }

    // The C++ tool clears and reuses one preallocated batch for every
    // generation step. Allocate once before timing and preserve that contract.
    let mut generation_batch = LlamaBatch::new(
        usize::try_from(input.parallel_sequences).expect("validated sequence count"),
        1,
    );
    let generation_started = Instant::now();
    decode_batched_generation(&mut context, &input, vocabulary_size, &mut generation_batch)?;
    let generation_seconds = generation_started.elapsed().as_secs_f64();
    if prompt_seconds <= 0.0 || generation_seconds <= 0.0 {
        return Err(ProbeError::runtime(
            "invalid-duration",
            "batched benchmark measured a zero duration",
        ));
    }

    let prompt_work = if input.shared_prompt {
        input.prompt_tokens
    } else {
        input.prompt_tokens * input.parallel_sequences
    };
    let generation_work = input.generation_tokens_per_sequence * input.parallel_sequences;
    let measured_tokens = prompt_work + generation_work;
    let total_seconds = prompt_seconds + generation_seconds;
    let tokens_per_second = f64::from(measured_tokens) / total_seconds;
    let n_kv = input.expected_kv_tokens;
    let value = json!({
        "n_kv_max": context.n_ctx(),
        "n_batch": context.n_batch(),
        "n_ubatch": context.n_ubatch(),
        "flash_attn": flash_attention_value,
        "is_pp_shared": i32::from(input.shared_prompt),
        "n_gpu_layers": batched_bench_gpu_layers,
        "n_threads": context.n_threads(),
        "n_threads_batch": context.n_threads_batch(),
        "pp": input.prompt_tokens,
        "tg": input.generation_tokens_per_sequence,
        "pl": input.parallel_sequences,
        "n_kv": n_kv,
        "t_pp": prompt_seconds,
        "speed_pp": f64::from(prompt_work) / prompt_seconds,
        "t_tg": generation_seconds,
        "speed_tg": f64::from(generation_work) / generation_seconds,
        "t": total_seconds,
        "speed": tokens_per_second,
    });
    let mut effective_configuration = value
        .as_object()
        .expect("batched benchmark value is an object")
        .iter()
        .filter(|(name, _)| {
            matches!(
                name.as_str(),
                "n_kv_max"
                    | "n_batch"
                    | "n_ubatch"
                    | "flash_attn"
                    | "is_pp_shared"
                    | "n_gpu_layers"
                    | "n_threads"
                    | "n_threads_batch"
                    | "pp"
                    | "tg"
                    | "pl"
                    | "n_kv"
            )
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<Map<_, _>>();
    let output = Value::Object(effective_configuration.clone());
    effective_configuration.insert("n_gpu_layers".to_owned(), normalized_gpu_layers);
    effective_configuration.insert("kv_unified".to_owned(), json!(input.kv_unified));

    Ok(json!({
        "schemaVersion": 1,
        "output": output,
        "measurements": [
            {"name": "prompt_duration", "unit": "s", "samples": [prompt_seconds]},
            {"name": "generation_duration", "unit": "s", "samples": [generation_seconds]},
            {"name": "duration", "unit": "s", "samples": [total_seconds]},
            {"name": "tokens_per_second", "unit": "tokens/s", "samples": [tokens_per_second]},
        ],
        "effectiveConfiguration": effective_configuration,
        "devices": devices,
        "producerVersion": concat!("icn-engine/", env!("CARGO_PKG_VERSION")),
    }))
}

fn prepare_batched_prompt(
    input: &LlamaBatchedBenchInput,
    vocabulary_size: i32,
) -> Result<Vec<LlamaBatch<'static>>, ProbeError> {
    let capacity = usize::try_from(input.batch_tokens)
        .map_err(|error| ProbeError::invalid("invalid-batch", error))?;
    let prompt_work = input
        .prompt_tokens
        .checked_mul(if input.shared_prompt {
            1
        } else {
            input.parallel_sequences
        })
        .expect("validated prompt work");
    let chunk_count = prompt_work.div_ceil(input.batch_tokens);
    let mut batches =
        Vec::with_capacity(usize::try_from(chunk_count).expect("validated prompt chunk count"));
    let mut batch = LlamaBatch::new(capacity, 1);
    let prompt_sequences = if input.shared_prompt {
        1
    } else {
        input.parallel_sequences
    };
    for sequence_id in 0..prompt_sequences {
        for position in 0..input.prompt_tokens {
            if usize::try_from(batch.n_tokens()).expect("batch token count") == capacity {
                batches.push(batch);
                batch = LlamaBatch::new(capacity, 1);
            }
            batch
                .add(
                    next_llama_bench_token(vocabulary_size),
                    i32::try_from(position).expect("validated prompt position"),
                    &[i32::try_from(sequence_id).expect("validated sequence")],
                    position + 1 == input.prompt_tokens,
                )
                .map_err(|error| ProbeError::runtime("prompt-batch", error))?;
        }
    }
    if batch.n_tokens() > 0 {
        batches.push(batch);
    }
    Ok(batches)
}

fn decode_batched_generation(
    context: &mut LlamaContext<'_>,
    input: &LlamaBatchedBenchInput,
    vocabulary_size: i32,
    batch: &mut LlamaBatch<'_>,
) -> Result<(), ProbeError> {
    if input.separate_generation {
        for sequence_id in 0..input.parallel_sequences {
            for offset in 0..input.generation_tokens_per_sequence {
                batch.clear();
                batch
                    .add(
                        next_llama_bench_token(vocabulary_size),
                        i32::try_from(input.prompt_tokens + offset)
                            .expect("validated generation position"),
                        &[i32::try_from(sequence_id).expect("validated sequence")],
                        true,
                    )
                    .map_err(|error| ProbeError::runtime("generation-batch", error))?;
                context
                    .decode(batch)
                    .map_err(|error| ProbeError::runtime("generation-decode", error))?;
                context.synchronize();
            }
        }
    } else {
        for offset in 0..input.generation_tokens_per_sequence {
            batch.clear();
            for sequence_id in 0..input.parallel_sequences {
                batch
                    .add(
                        next_llama_bench_token(vocabulary_size),
                        i32::try_from(input.prompt_tokens + offset)
                            .expect("validated generation position"),
                        &[i32::try_from(sequence_id).expect("validated sequence")],
                        true,
                    )
                    .map_err(|error| ProbeError::runtime("generation-batch", error))?;
            }
            context
                .decode(batch)
                .map_err(|error| ProbeError::runtime("generation-decode", error))?;
            context.synchronize();
        }
    }
    Ok(())
}

fn llama_bench(operation: &str, input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: LlamaBenchInput = decode_input(input)?;
    validate_llama_bench_input(operation, &input)?;

    // llama-bench installs a null native logger unless --verbose is requested.
    // Matching that here keeps native logging and stderr I/O out of the timed
    // set_n_threads/decode path.
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    let threads = nonzero_i32(input.engine_configuration.threads, "threads")?;
    let flash_attention = match input.engine_configuration.flash_attention.as_str() {
        "auto" => FlashAttention::Auto,
        "on" => FlashAttention::Enabled,
        "off" => FlashAttention::Disabled,
        value => {
            return Err(ProbeError::invalid(
                "invalid-engine-configuration",
                format!("unsupported flashAttention policy: {value}"),
            ));
        }
    };
    let (gpu_layers, normalized_gpu_layers) = input.engine_configuration.gpu_layers.normalized()?;
    let execution = ExecutionConfig {
        gpu_layers,
        threads: NonZeroU32::new(input.engine_configuration.threads),
        threads_batch: NonZeroU32::new(input.engine_configuration.threads),
        flash_attention,
        ..ExecutionConfig::default()
    };
    let requested_context = input
        .prompt_tokens
        .checked_add(input.generation_tokens)
        .and_then(|value| value.checked_add(input.context_depth))
        .expect("validated benchmark context sum fits in u32");
    let config = ResolvedExecutionPlan {
        model_path: input.model_path.clone(),
        context_size: requested_context,
        batch_size: input.batch_tokens,
        ubatch_size: input.micro_batch_tokens,
        max_sequences: 1,
        prefill_quantum: input.batch_tokens,
        radix_cache: Default::default(),
        execution,
        projector: None,
        mtp: icn_contracts::MtpConfig::default(),
    };

    if !config.model_path.is_file() {
        return Err(ProbeError::invalid(
            "invalid-model-path",
            format!(
                "modelPath is not a regular file: {}",
                config.model_path.display()
            ),
        ));
    }
    let backend =
        LlamaBackend::init().map_err(|error| ProbeError::runtime("backend-initialize", error))?;
    let model_params = super::native_model_params(&config.execution)
        .map_err(|error| ProbeError::invalid("invalid-engine-configuration", error))?;
    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .map_err(|error| ProbeError::runtime("model-load", error))?;
    let context_params = super::native_context_params(&config, threads, threads);
    let mut context = model
        .new_context(&backend, context_params)
        .map_err(|error| ProbeError::runtime("context-create", error))?;
    let threadpool_params = LlamaThreadPoolParams::new(threads)
        .with_strict_cpu(input.engine_configuration.cpu_strict)
        .with_poll(input.engine_configuration.threadpool_poll);
    let mut threadpool = LlamaThreadPool::new(&backend, &threadpool_params)
        .map_err(|error| ProbeError::runtime("threadpool-create", error))?;
    let (actual_backends, device_identity, devices) = benchmark_hardware_identity()?;
    let mut context = context.attach_threadpool(&mut threadpool);

    // Match llama-bench's per-case lifecycle. The model, context, and persistent
    // threadpool are all constructed before either warmup or measurement.
    context.clear_memory(false);
    if input.warmup {
        if input.prompt_tokens > 0 {
            let _ = llama_bench_prompt(
                &mut context,
                input.prompt_tokens,
                input.batch_tokens,
                threads.get(),
            )?;
        }
        if input.generation_tokens > 0 {
            let _ = llama_bench_generate(&mut context, 1, threads.get())?;
        }
    }

    context.clear_memory(false);
    let (_depth_state, prepared_context_depth) = if input.context_depth > 0 {
        let observed = llama_bench_prompt(
            &mut context,
            input.context_depth,
            input.batch_tokens,
            threads.get(),
        )?;
        // llama-bench retains this snapshot for later repetitions. The parity
        // runner deliberately performs one repetition per process, but keeping
        // the same snapshot alive preserves the upstream preparation contract.
        (
            Some(
                context
                    .state_seq_get_data_ext_owned(0, LlamaStateSeqFlags::empty())
                    .map_err(|error| ProbeError::runtime("context-depth-state", error))?,
            ),
            observed,
        )
    } else {
        (None, LlamaBenchDecodeWork::default())
    };

    let mut measured_prompt = LlamaBenchDecodeWork::default();
    let mut measured_generation = LlamaBenchDecodeWork::default();
    let mut measurement_repetitions = 0_u32;
    let started = Instant::now();
    if input.prompt_tokens > 0 {
        measured_prompt = llama_bench_prompt(
            &mut context,
            input.prompt_tokens,
            input.batch_tokens,
            threads.get(),
        )?;
    }
    if input.generation_tokens > 0 {
        measured_generation =
            llama_bench_generate(&mut context, input.generation_tokens, threads.get())?;
    }
    measurement_repetitions += 1;
    let duration_ns = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
        ProbeError::runtime("duration-overflow", "measured duration exceeds u64::MAX ns")
    })?;
    if duration_ns == 0 {
        return Err(ProbeError::runtime(
            "invalid-duration",
            "measured duration was zero",
        ));
    }

    let measured_tokens = measured_prompt
        .tokens
        .checked_add(measured_generation.tokens)
        .expect("validated measured-token sum fits in u32");
    let tokens_per_second = f64::from(measured_tokens) * 1_000_000_000.0 / duration_ns as f64;
    let effective_configuration = json!({
        "threads": context.n_threads(),
        "batch_size": context.n_batch(),
        "ubatch_size": context.n_ubatch(),
        "requested_batch_size": input.batch_tokens,
        "requested_ubatch_size": input.micro_batch_tokens,
        "effective_n_ctx": context.n_ctx(),
        "n_gpu_layers": normalized_gpu_layers,
        "cpu_strict": threadpool_params.strict_cpu(),
        "threadpool_poll": threadpool_params.poll(),
        "kv_type_k": "f16",
        "kv_type_v": "f16",
        "split_mode": "layer",
        "main_gpu": 0,
        "offload_kqv": true,
        // An empty safe-binding device list is llama.cpp's automatic-device
        // policy. llama-bench serializes that same effective policy as
        // `auto`, and its default all-zero split vector as `0.00`.
        "devices": "auto",
        "tensor_split": "0.00",
        "use_mmap": true,
        "use_direct_io": false,
        "embeddings": false,
        "operation_offload": true,
        "no_host": false,
        "kv_unified": false,
        "flash_attention": input.engine_configuration.flash_attention,
        "n_prompt": measured_prompt.tokens,
        "n_gen": measured_generation.tokens,
        "context_depth": prepared_context_depth.tokens,
        "requested_n_ctx": requested_context,
        "swa_full": false,
        "memory_clear_data": false,
        "warmup": input.warmup,
        "threadpool_contract": "persistent-per-case",
        "actual_backends": actual_backends,
        "device_identity": device_identity,
    });
    let effective_configuration = effective_configuration
        .as_object()
        .expect("configuration literal is an object");

    Ok(json!({
        "schemaVersion": 1,
        "output": {
            "operation": operation,
            "measuredPromptTokens": measured_prompt.tokens,
            "measuredPromptDecodeCalls": measured_prompt.decode_calls,
            "measuredGenerationTokens": measured_generation.tokens,
            "measuredGenerationDecodeCalls": measured_generation.decode_calls,
            "preparedContextDepthTokens": prepared_context_depth.tokens,
            "preparedContextDepthDecodeCalls": prepared_context_depth.decode_calls,
            "measurementRepetitions": measurement_repetitions,
            "measuredTokens": measured_tokens,
            "tokenSchedule": input.token_schedule,
        },
        "measurements": [
            {"name": "duration", "unit": "ns", "samples": [duration_ns as f64]},
            {"name": "tokens_per_second", "unit": "tokens/s", "samples": [tokens_per_second]},
        ],
        // Work is intentionally omitted. The parity runner owns the canonical
        // invocation case and derives WorkDefinition without injected runtime
        // fields such as modelPath and engineConfiguration.
        "effectiveConfiguration": effective_configuration,
        "devices": devices,
        "producerVersion": concat!("icn-engine/", env!("CARGO_PKG_VERSION")),
    }))
}

fn llama_bench_prompt(
    context: &mut LlamaContext<'_>,
    n_prompt: u32,
    n_batch: u32,
    n_threads: i32,
) -> Result<LlamaBenchDecodeWork, ProbeError> {
    context.set_n_threads(n_threads, n_threads);
    let vocabulary_size = context.model.n_vocab();
    if vocabulary_size <= 0 {
        return Err(ProbeError::runtime(
            "invalid-vocabulary",
            "loaded model reports no vocabulary tokens",
        ));
    }
    let batch_capacity =
        usize::try_from(n_batch).map_err(|error| ProbeError::invalid("invalid-batch", error))?;
    let mut tokens = vec![LlamaToken::new(0); batch_capacity];
    let mut processed = 0_u32;
    let mut observed = LlamaBenchDecodeWork::default();
    while processed < n_prompt {
        let token_count = (n_prompt - processed).min(n_batch);
        tokens[0] = if processed == 0 && context.model.should_add_bos() {
            context.model.token_bos()
        } else {
            next_llama_bench_token(vocabulary_size)
        };
        for token in &mut tokens[1..usize::try_from(token_count).expect("token count fits usize")] {
            *token = next_llama_bench_token(vocabulary_size);
        }
        let mut batch = LlamaBatch::get_one(
            &tokens[..usize::try_from(token_count).expect("token count fits usize")],
        )
        .map_err(|error| ProbeError::runtime("batch-create", error))?;
        context
            .decode(&mut batch)
            .map_err(|error| ProbeError::runtime("prompt-decode", error))?;
        processed += token_count;
        observed.tokens += token_count;
        observed.decode_calls += 1;
    }
    context.synchronize();
    Ok(observed)
}

fn llama_bench_generate(
    context: &mut LlamaContext<'_>,
    n_gen: u32,
    n_threads: i32,
) -> Result<LlamaBenchDecodeWork, ProbeError> {
    context.set_n_threads(n_threads, n_threads);
    let vocabulary_size = context.model.n_vocab();
    if vocabulary_size <= 0 {
        return Err(ProbeError::runtime(
            "invalid-vocabulary",
            "loaded model reports no vocabulary tokens",
        ));
    }
    let mut token = if context.model.should_add_bos() {
        context.model.token_bos()
    } else {
        next_llama_bench_token(vocabulary_size)
    };
    let mut observed = LlamaBenchDecodeWork::default();
    for _ in 0..n_gen {
        let one = [token];
        let mut batch = LlamaBatch::get_one(&one)
            .map_err(|error| ProbeError::runtime("batch-create", error))?;
        context
            .decode(&mut batch)
            .map_err(|error| ProbeError::runtime("generation-decode", error))?;
        context.synchronize();
        observed.tokens += 1;
        observed.decode_calls += 1;
        // llama-bench advances std::rand even after the final decode.
        token = next_llama_bench_token(vocabulary_size);
    }
    Ok(observed)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LlamaBenchDecodeWork {
    tokens: u32,
    decode_calls: u32,
}

fn next_llama_bench_token(vocabulary_size: i32) -> LlamaToken {
    // SAFETY: C's rand takes no arguments, has no preconditions, and returns a
    // non-negative int. Using the process-default seed is part of the pinned
    // llama-bench workload; deliberately do not call srand here.
    let value = unsafe { c_rand() };
    LlamaToken::new(value % vocabulary_size)
}

unsafe extern "C" {
    #[link_name = "rand"]
    fn c_rand() -> std::ffi::c_int;
}

fn benchmark_hardware_identity() -> Result<(String, String, Vec<Value>), ProbeError> {
    let devices = list_llama_ggml_backend_devices();
    if devices.is_empty() {
        return Err(ProbeError::runtime(
            "device-inventory",
            "llama.cpp reported no backend devices",
        ));
    }
    let mut backends = Vec::<String>::new();
    let mut cpu_descriptions = Vec::<String>::new();
    let mut gpu_descriptions = Vec::<String>::new();
    let projected = devices
        .iter()
        .map(|device| {
            let backend = if device.backend.starts_with("RPC") {
                "RPC"
            } else {
                device.backend.as_str()
            };
            if backend != "CPU" && !backends.iter().any(|existing| existing == backend) {
                backends.push(backend.to_owned());
            }
            match device.device_type {
                LlamaBackendDeviceType::Cpu | LlamaBackendDeviceType::Accelerator => {
                    cpu_descriptions.push(device.description.clone());
                }
                LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu => {
                    gpu_descriptions.push(device.description.clone());
                }
                LlamaBackendDeviceType::Unknown => {}
            }
            json!({
                "backend": device.backend,
                "name": if device.description.is_empty() {
                    &device.name
                } else {
                    &device.description
                },
                "memory_bytes": device.memory_total,
            })
        })
        .collect::<Vec<_>>();
    let actual_backends = if backends.is_empty() {
        "CPU".to_owned()
    } else {
        backends.join(",")
    };
    let device_identity = if gpu_descriptions.is_empty() {
        cpu_descriptions.join(", ")
    } else {
        gpu_descriptions.join(", ")
    };
    if device_identity.is_empty() {
        return Err(ProbeError::runtime(
            "device-inventory",
            "llama.cpp backend devices have no CPU/GPU description",
        ));
    }
    Ok((actual_backends, device_identity, projected))
}

fn validate_llama_bench_input(operation: &str, input: &LlamaBenchInput) -> Result<(), ProbeError> {
    if input.effective_engine_configuration != "profile" {
        return Err(ProbeError::invalid(
            "invalid-engine-configuration",
            "effective_engine_configuration must be 'profile'",
        ));
    }
    if input.token_schedule != "pinned-llama-bench-c-rand-default" {
        return Err(ProbeError::invalid(
            "invalid-token-schedule",
            "token_schedule must be 'pinned-llama-bench-c-rand-default'",
        ));
    }
    if input.repetitions != 1 {
        return Err(ProbeError::invalid(
            "invalid-repetitions",
            "repetitions must be exactly one; the parity runner owns process-level repetition",
        ));
    }
    if input.batch_tokens == 0
        || input.batch_tokens > i32::MAX.cast_unsigned()
        || input.micro_batch_tokens == 0
        || input.micro_batch_tokens > input.batch_tokens
    {
        return Err(ProbeError::invalid(
            "invalid-batch",
            "batch_tokens must be in [1, i32::MAX] and micro_batch_tokens in [1, batch_tokens]",
        ));
    }
    if input.engine_configuration.backend.trim().is_empty() {
        return Err(ProbeError::invalid(
            "invalid-engine-configuration",
            "engineConfiguration.backend must not be empty",
        ));
    }
    nonzero_i32(input.engine_configuration.threads, "threads")?;
    let requested_context = input
        .prompt_tokens
        .checked_add(input.generation_tokens)
        .and_then(|value| value.checked_add(input.context_depth))
        .ok_or_else(|| ProbeError::invalid("invalid-context", "context token sum overflowed"))?;
    if requested_context == 0 || requested_context > i32::MAX.cast_unsigned() {
        return Err(ProbeError::invalid(
            "invalid-context",
            "prompt, generation, and depth must request a context in [1, i32::MAX]",
        ));
    }
    let shape_matches = match operation {
        "llama-bench.prompt-processing" => {
            input.prompt_tokens > 0 && input.generation_tokens == 0 && input.context_depth == 0
        }
        "llama-bench.token-generation" => {
            input.prompt_tokens == 0 && input.generation_tokens > 0 && input.context_depth == 0
        }
        "llama-bench.prompt-generation" => {
            input.prompt_tokens > 0 && input.generation_tokens > 0 && input.context_depth == 0
        }
        "llama-bench.context-depth" => {
            input.prompt_tokens == 0 && input.generation_tokens > 0 && input.context_depth > 0
        }
        _ => false,
    };
    if !shape_matches {
        return Err(ProbeError::invalid(
            "invalid-workload-shape",
            format!("input token counts do not match {operation}"),
        ));
    }
    Ok(())
}

fn nonzero_i32(value: u32, field: &'static str) -> Result<NonZeroI32, ProbeError> {
    let value = i32::try_from(value).map_err(|_| {
        ProbeError::invalid(
            "invalid-engine-configuration",
            format!("{field} exceeds i32::MAX"),
        )
    })?;
    NonZeroI32::new(value).ok_or_else(|| {
        ProbeError::invalid(
            "invalid-engine-configuration",
            format!("{field} must be non-zero"),
        )
    })
}

fn selected_logits(logits: &[f32], indices: &[i32]) -> Vec<Value> {
    indices
        .iter()
        .map(|index| {
            let value =
                logits[usize::try_from(*index).expect("validated logit index is non-negative")];
            let (value, class) = finite_number(value);
            json!({"index": index, "value": value, "class": class})
        })
        .collect()
}

fn chat_template_render(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: ChatTemplateInput = decode_input(input)?;
    let templates = CommonChatTemplates::from_template(
        &input.template,
        input.bos_token.as_deref(),
        input.eos_token.as_deref(),
    )
    .map_err(|error| ProbeError::runtime("chat-template-init", error))?;
    let capabilities = templates
        .capabilities()
        .map_err(|error| ProbeError::runtime("chat-template-capabilities", error))?;
    let prepared = templates
        .prepare(&ChatPrepareOptions {
            messages: input
                .messages
                .into_iter()
                .map(ChatMessageInput::into_native)
                .collect::<Result<Vec<_>, _>>()?,
            grammar: input.grammar,
            json_schema: input
                .json_schema
                .map(|schema| serde_json::to_string(&schema))
                .transpose()
                .map_err(|error| ProbeError::invalid("invalid-json-schema", error))?,
            add_generation_prompt: input.add_generation_prompt,
            continuation: parse_continuation(input.continue_final_message.as_ref())?,
            use_jinja: input.use_jinja,
            tools: input
                .tools
                .into_iter()
                .map(ChatToolInput::into_native)
                .collect::<Result<Vec<_>, _>>()?,
            tool_choice: parse_tool_choice(&input.tool_choice)?,
            parallel_tool_calls: Some(input.parallel_tool_calls),
            reasoning_format: parse_reasoning_format(&input.reasoning_format)?,
            enable_thinking: Some(input.enable_thinking),
            template_kwargs: input
                .chat_template_kwargs
                .into_iter()
                .map(|(key, value)| {
                    Ok(ChatTemplateKwarg {
                        key,
                        value_json: serde_json::to_string(&value).map_err(|error| {
                            ProbeError::invalid("invalid-template-kwarg", error)
                        })?,
                    })
                })
                .collect::<Result<Vec<_>, ProbeError>>()?,
            force_pure_content: input.force_pure_content,
        })
        .map_err(|error| ProbeError::runtime("chat-template-prepare", error))?;

    let triggers = prepared
        .grammar_triggers()
        .iter()
        .map(|trigger| match trigger {
            ChatGrammarTrigger::Token { value, token } => {
                json!({"type": "token", "value": value, "token": token})
            }
            ChatGrammarTrigger::Word(value) => {
                json!({"type": "word", "value": value, "token": -1})
            }
            ChatGrammarTrigger::Pattern(value) => {
                json!({"type": "pattern", "value": value, "token": -1})
            }
            ChatGrammarTrigger::PatternFull(value) => {
                json!({"type": "pattern-full", "value": value, "token": -1})
            }
        })
        .collect::<Vec<_>>();
    let spans = prepared
        .message_spans()
        .iter()
        .map(|span| json!({"role": span.role, "position": span.position, "length": span.length}))
        .collect::<Vec<_>>();

    Ok(json!({
        "source": templates
            .source(None)
            .map_err(|error| ProbeError::runtime("chat-template-source", error))?,
        "explicitTemplate": templates.was_explicit(),
        "capabilities": {
            "supports_object_arguments": capabilities.supports_object_arguments,
            "supports_parallel_tool_calls": capabilities.supports_parallel_tool_calls,
            "supports_preserve_reasoning": capabilities.supports_preserve_reasoning,
            "supports_string_content": capabilities.supports_string_content,
            "supports_system_role": capabilities.supports_system_role,
            "supports_tool_calls": capabilities.supports_tool_calls,
            "supports_tools": capabilities.supports_tools,
            "supports_typed_content": capabilities.supports_typed_content,
        },
        "format": chat_format_name(prepared.format()),
        "prompt": prepared.prompt(),
        "grammar": prepared.grammar(),
        "grammarLazy": prepared.grammar_lazy(),
        "generationPrompt": prepared.generation_prompt(),
        "supportsThinking": prepared.supports_thinking(),
        "thinkingStartTag": prepared.thinking_start_tag().unwrap_or_default(),
        "thinkingEndTag": prepared.thinking_end_tag().unwrap_or_default(),
        "grammarTriggers": triggers,
        "preservedTokens": prepared.preserved_tokens(),
        "additionalStops": prepared.additional_stops(),
        "parser": prepared.parser_definition(),
        "messageSpans": spans,
    }))
}

fn sampler_apply(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: SamplerInput = decode_input(input)?;
    if input.candidates.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-candidates",
            "candidates must not be empty",
        ));
    }
    let selection = input.selection.normalized_raw();
    let mut components = input
        .samplers
        .iter()
        .map(make_sampler)
        .collect::<Result<Vec<_>, _>>()?;
    match input.selection.kind.as_str() {
        "distribution" => components.push(LlamaSampler::dist(input.selection.seed)),
        "greedy" => components.push(LlamaSampler::greedy()),
        "none" => {}
        value => {
            return Err(ProbeError::invalid(
                "invalid-selection",
                format!("unsupported selection type: {value}"),
            ));
        }
    }
    let mut sampler = LlamaSampler::chain(components, true);
    for token in input.accepted_tokens {
        sampler
            .try_accept(LlamaToken::new(token))
            .map_err(|error| ProbeError::runtime("sampler-accept", error))?;
    }
    let mut candidates = LlamaTokenDataArray::new(
        input
            .candidates
            .into_iter()
            .map(|candidate| {
                LlamaTokenData::new(
                    LlamaToken::new(candidate.id),
                    candidate.logit,
                    candidate.probability,
                )
            })
            .collect(),
        false,
    );
    sampler.apply(&mut candidates);
    let selected_token = candidates.selected_token().map(|token| token.0);
    let output = candidates
        .data
        .iter()
        .map(|candidate| {
            let (logit, logit_class) = finite_number(candidate.logit());
            let (probability, probability_class) = finite_number(candidate.p());
            json!({
                "id": candidate.id().0,
                "logit": logit,
                "logitClass": logit_class,
                "probability": probability,
                "probabilityClass": probability_class,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "candidates": output,
        "sorted": candidates.sorted,
        "selectedIndex": candidates.selected,
        "selectedToken": selected_token,
        "selection": selection,
    }))
}

fn sampler_bench(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: SamplerBenchInput = decode_input(input)?;
    if input.candidate_count == 0 || input.candidate_count > i32::MAX as usize {
        return Err(ProbeError::invalid(
            "invalid-candidate-count",
            "candidate_count must fit the positive llama_token range",
        ));
    }
    if input.candidate_generator.kind != "seeded-uniform-logits" {
        return Err(ProbeError::invalid(
            "invalid-candidate-generator",
            "candidate_generator.kind must be seeded-uniform-logits",
        ));
    }
    if !input.candidate_generator.minimum.is_finite()
        || !input.candidate_generator.maximum.is_finite()
        || input.candidate_generator.minimum >= input.candidate_generator.maximum
    {
        return Err(ProbeError::invalid(
            "invalid-candidate-generator",
            "candidate generator bounds must be finite and increasing",
        ));
    }
    if input.sampler.kind != "top-k" {
        return Err(ProbeError::invalid(
            "invalid-sampler",
            "sampler.type must be top-k",
        ));
    }
    if input.sampler.k <= 0
        || usize::try_from(input.sampler.k)
            .ok()
            .is_none_or(|value| value > input.candidate_count)
    {
        return Err(ProbeError::invalid(
            "invalid-sampler",
            "sampler.k must be positive and no larger than candidate_count",
        ));
    }
    if input.iterations == 0 {
        return Err(ProbeError::invalid(
            "invalid-iterations",
            "iterations must be positive",
        ));
    }

    // Match the oracle's language-neutral Numerical Recipes LCG exactly. The
    // source vector, working allocation, and sampler construction all remain
    // outside warmup and measurement.
    let source = generate_uniform_candidates(
        input.candidate_count,
        input.candidate_generator.seed,
        input.candidate_generator.minimum,
        input.candidate_generator.maximum,
    );
    let mut result = LlamaTokenDataArray::new(
        vec![LlamaTokenData::new(LlamaToken::new(0), 0.0, 0.0); input.candidate_count],
        false,
    );
    let mut sampler = LlamaSampler::top_k(input.sampler.k);
    let mut executed_warmup_iterations = 0_u32;
    for _ in 0..input.warmup_iterations {
        sampler_bench_apply_once(&source, &mut result, &mut sampler)?;
        executed_warmup_iterations += 1;
    }

    let mut executed_measurement_iterations = 0_u32;
    let mut semantic_checksum = SEMANTIC_CHECKSUM_OFFSET;
    let started = Instant::now();
    for _ in 0..input.iterations {
        sampler_bench_apply_once(&source, &mut result, &mut sampler)?;
        let iteration_semantic = sampler_bench_semantic_checksum(&result);
        semantic_checksum = fold_iteration_semantic(semantic_checksum, iteration_semantic);
        executed_measurement_iterations += 1;
    }
    let duration_ns = positive_duration_ns(started)?;

    if result.selected.is_some() {
        return Err(ProbeError::runtime(
            "unexpected-selection",
            "top-k benchmark unexpectedly selected a candidate",
        ));
    }
    let output = json!({
        "resultCandidateCount": result.data.len(),
        "sorted": result.sorted,
        "resultTokenIds": result
            .data
            .iter()
            .map(|candidate| candidate.id().0)
            .collect::<Vec<_>>(),
        "executedWarmupIterations": executed_warmup_iterations,
        "executedMeasurementIterations": executed_measurement_iterations,
        "semanticChecksum": semantic_checksum,
    });
    Ok(timed_probe_evidence(output, duration_ns))
}

fn generate_uniform_candidates(
    count: usize,
    seed: u32,
    minimum: f32,
    maximum: f32,
) -> Vec<LlamaTokenData> {
    let mut state = seed;
    (0..count)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 * (1.0_f32 / 16_777_216.0_f32);
            let logit = minimum + (maximum - minimum) * unit;
            LlamaTokenData::new(
                LlamaToken::new(i32::try_from(index).expect("validated token index fits i32")),
                logit,
                0.0,
            )
        })
        .collect()
}

fn sampler_bench_apply_once(
    source: &[LlamaTokenData],
    result: &mut LlamaTokenDataArray,
    sampler: &mut LlamaSampler,
) -> Result<(), ProbeError> {
    if result.data.capacity() < source.len() {
        return Err(ProbeError::runtime(
            "invalid-candidate-buffer",
            "sampler reduced the working vector capacity",
        ));
    }

    // The safe sampler projection updates Vec's logical length to the native
    // result size. The oracle restores the same allocation's logical size with
    // an aggregate assignment before every call. All capacity slots were
    // initialized at construction, LlamaTokenData is Copy, and top-k does not
    // deallocate the caller-owned buffer, so restoring that length is sound and
    // keeps the timed boundary to the oracle's O(1) reset + copy/apply/reset.
    unsafe { result.data.set_len(source.len()) };
    result.data.copy_from_slice(source);
    result.selected = None;
    result.sorted = false;
    sampler.apply(result);
    sampler.reset();
    Ok(())
}

fn sampler_bench_semantic_checksum(result: &LlamaTokenDataArray) -> u64 {
    let mut value = result.data.len() as u64;
    value = value.wrapping_mul(SEMANTIC_CHECKSUM_PRIME) ^ u64::from(result.sorted);
    value = value.wrapping_mul(SEMANTIC_CHECKSUM_PRIME)
        ^ result.selected.map_or(u64::MAX, |index| index as u64);
    for candidate in &result.data {
        value = value.wrapping_mul(SEMANTIC_CHECKSUM_PRIME)
            ^ u64::from(candidate.id().0.cast_unsigned());
    }
    value
}

fn chat_template_bench(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: ChatTemplateBenchInput = decode_input(input)?;
    if input.iterations == 0 {
        return Err(ProbeError::invalid(
            "invalid-iterations",
            "iterations must be positive",
        ));
    }

    // Logical fixture names belong to runner-owned work evidence. Only these
    // resolved, digest-verified paths are used for I/O and never escape in the
    // operation evidence.
    let _ = (&input.template_fixture, &input.input_fixture);
    let template_source = fs::read_to_string(&input.fixture_paths.template)
        .map_err(|error| ProbeError::runtime("fixture-read", error))?;
    let request_source = fs::read_to_string(&input.fixture_paths.input)
        .map_err(|error| ProbeError::runtime("fixture-read", error))?;
    let request: Value = serde_json::from_str(&request_source)
        .map_err(|error| ProbeError::invalid("invalid-chat-fixture", error))?;
    let request = request.as_object().ok_or_else(|| {
        ProbeError::invalid(
            "invalid-chat-fixture",
            "chat input fixture must contain an object",
        )
    })?;
    let bos = chat_request_optional::<String>(request, "bos_token", String::new())?;
    let eos = chat_request_optional::<String>(request, "eos_token", String::new())?;
    let templates = CommonChatTemplates::from_template(
        &template_source,
        (!bos.is_empty()).then_some(bos.as_str()),
        (!eos.is_empty()).then_some(eos.as_str()),
    )
    .map_err(|error| ProbeError::runtime("chat-template-init", error))?;

    let mut warmup_semantic_sink = 0_u64;
    let mut executed_warmup_iterations = 0_u32;
    for _ in 0..input.warmup_iterations {
        let summary = chat_bench_prepare_once(&templates, request)?;
        warmup_semantic_sink ^= chat_summary_checksum(summary);
        executed_warmup_iterations += 1;
    }

    let mut measured_summary = None;
    let mut semantic_checksum = SEMANTIC_CHECKSUM_OFFSET;
    let mut executed_measurement_iterations = 0_u32;
    let started = Instant::now();
    for _ in 0..input.iterations {
        let summary = chat_bench_prepare_once(&templates, request)?;
        semantic_checksum =
            fold_iteration_semantic(semantic_checksum, chat_summary_checksum(summary));
        executed_measurement_iterations += 1;
        measured_summary = Some(summary);
    }
    let duration_ns = positive_duration_ns(started)?;
    std::hint::black_box(warmup_semantic_sink);
    std::hint::black_box(semantic_checksum);
    let summary = measured_summary.expect("positive iterations produce a summary");
    let semantic = chat_bench_prepare(&templates, request)?;
    if summarize_chat_preparation(&semantic) != summary {
        return Err(ProbeError::runtime(
            "unstable-chat-preparation",
            "untimed semantic preparation did not match the measured summary",
        ));
    }
    let grammar_triggers = project_chat_grammar_triggers(&semantic);
    let message_spans = project_chat_message_spans(&semantic);
    let output = json!({
        "format": chat_format_name(summary.format),
        "prompt": semantic.prompt(),
        "grammar": semantic.grammar(),
        "generationPrompt": semantic.generation_prompt(),
        "parser": semantic.parser_definition(),
        "grammarTriggers": grammar_triggers,
        "preservedTokens": semantic.preserved_tokens(),
        "additionalStops": semantic.additional_stops(),
        "messageSpans": message_spans,
        "promptBytes": summary.prompt_bytes,
        "grammarBytes": summary.grammar_bytes,
        "generationPromptBytes": summary.generation_prompt_bytes,
        "triggerCount": summary.trigger_count,
        "preservedTokenCount": summary.preserved_token_count,
        "stopCount": summary.stop_count,
        "parserBytes": summary.parser_bytes,
        "spanCount": summary.span_count,
        "executedWarmupIterations": executed_warmup_iterations,
        "executedMeasurementIterations": executed_measurement_iterations,
        "semanticChecksum": semantic_checksum,
    });
    Ok(timed_probe_evidence(output, duration_ns))
}

fn chat_bench_prepare_once(
    templates: &CommonChatTemplates,
    request: &Map<String, Value>,
) -> Result<ChatPreparationSummary, ProbeError> {
    let prepared = chat_bench_prepare(templates, request)?;
    Ok(summarize_chat_preparation(&prepared))
}

fn chat_bench_prepare(
    templates: &CommonChatTemplates,
    request: &Map<String, Value>,
) -> Result<PreparedChat, ProbeError> {
    // Conversion from the neutral JSON request belongs inside the benchmark,
    // exactly like the native oracle's chat_bench_inputs(request) call.
    let messages = chat_request_required::<Vec<ChatMessageInput>>(request, "messages")?
        .into_iter()
        .map(ChatMessageInput::into_native)
        .collect::<Result<Vec<_>, _>>()?;
    let grammar = chat_request_optional::<String>(request, "grammar", String::new())?;
    let json_schema = request
        .get("json_schema")
        .filter(|value| !value.is_null())
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| ProbeError::invalid("invalid-json-schema", error))?;
    let add_generation_prompt = chat_request_optional(request, "add_generation_prompt", true)?;
    let continuation = parse_continuation(request.get("continue_final_message"))?;
    let use_jinja = chat_request_optional(request, "use_jinja", true)?;
    let tools = chat_request_optional::<Vec<ChatToolInput>>(request, "tools", Vec::new())?
        .into_iter()
        .map(ChatToolInput::into_native)
        .collect::<Result<Vec<_>, _>>()?;
    let tool_choice = parse_tool_choice(&chat_request_optional::<String>(
        request,
        "tool_choice",
        default_auto(),
    )?)?;
    let parallel_tool_calls = chat_request_optional(request, "parallel_tool_calls", false)?;
    let reasoning_format = parse_reasoning_format(&chat_request_optional::<String>(
        request,
        "reasoning_format",
        default_none(),
    )?)?;
    let enable_thinking = chat_request_optional(request, "enable_thinking", true)?;
    let template_kwargs = match request.get("chat_template_kwargs") {
        None => Vec::new(),
        Some(Value::Object(values)) => values
            .iter()
            .map(|(key, value)| {
                Ok(ChatTemplateKwarg {
                    key: key.clone(),
                    value_json: serde_json::to_string(value)
                        .map_err(|error| ProbeError::invalid("invalid-template-kwarg", error))?,
                })
            })
            .collect::<Result<Vec<_>, ProbeError>>()?,
        Some(_) => {
            return Err(ProbeError::invalid(
                "invalid-chat-fixture",
                "chat_template_kwargs must be an object",
            ));
        }
    };
    let force_pure_content = chat_request_optional(request, "force_pure_content", false)?;

    templates
        .prepare(&ChatPrepareOptions {
            messages,
            grammar: (!grammar.is_empty()).then_some(grammar),
            json_schema,
            add_generation_prompt,
            continuation,
            use_jinja,
            tools,
            tool_choice,
            parallel_tool_calls: Some(parallel_tool_calls),
            reasoning_format,
            enable_thinking: Some(enable_thinking),
            template_kwargs,
            force_pure_content,
        })
        .map_err(|error| ProbeError::runtime("chat-template-prepare", error))
}

fn summarize_chat_preparation(prepared: &PreparedChat) -> ChatPreparationSummary {
    ChatPreparationSummary {
        format: prepared.format(),
        prompt_bytes: prepared.prompt().len(),
        grammar_bytes: prepared.grammar().len(),
        generation_prompt_bytes: prepared.generation_prompt().len(),
        trigger_count: prepared.grammar_triggers().len(),
        preserved_token_count: prepared.preserved_tokens().len(),
        stop_count: prepared.additional_stops().len(),
        parser_bytes: prepared.parser_definition().len(),
        span_count: prepared.message_spans().len(),
    }
}

fn project_chat_grammar_triggers(prepared: &PreparedChat) -> Vec<Value> {
    prepared
        .grammar_triggers()
        .iter()
        .map(|trigger| match trigger {
            ChatGrammarTrigger::Token { value, token } => {
                json!({"type": "token", "value": value, "token": token})
            }
            ChatGrammarTrigger::Word(value) => {
                json!({"type": "word", "value": value, "token": -1})
            }
            ChatGrammarTrigger::Pattern(value) => {
                json!({"type": "pattern", "value": value, "token": -1})
            }
            ChatGrammarTrigger::PatternFull(value) => {
                json!({"type": "pattern-full", "value": value, "token": -1})
            }
        })
        .collect()
}

fn project_chat_message_spans(prepared: &PreparedChat) -> Vec<Value> {
    prepared
        .message_spans()
        .iter()
        .map(|span| json!({"role": span.role, "position": span.position, "length": span.length}))
        .collect()
}

fn chat_request_required<T: DeserializeOwned>(
    request: &Map<String, Value>,
    field: &'static str,
) -> Result<T, ProbeError> {
    let value = request.get(field).ok_or_else(|| {
        ProbeError::invalid(
            "invalid-chat-fixture",
            format!("chat input fixture is missing {field}"),
        )
    })?;
    serde_json::from_value(value.clone())
        .map_err(|error| ProbeError::invalid("invalid-chat-fixture", error))
}

fn chat_request_optional<T: DeserializeOwned>(
    request: &Map<String, Value>,
    field: &'static str,
    default: T,
) -> Result<T, ProbeError> {
    request.get(field).map_or(Ok(default), |value| {
        serde_json::from_value(value.clone())
            .map_err(|error| ProbeError::invalid("invalid-chat-fixture", error))
    })
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ChatPreparationSummary {
    format: ChatFormat,
    prompt_bytes: usize,
    grammar_bytes: usize,
    generation_prompt_bytes: usize,
    trigger_count: usize,
    preserved_token_count: usize,
    stop_count: usize,
    parser_bytes: usize,
    span_count: usize,
}

fn chat_summary_checksum(summary: ChatPreparationSummary) -> u64 {
    let mut value: u64 = match summary.format {
        ChatFormat::ContentOnly => 0,
        ChatFormat::PegSimple => 1,
        ChatFormat::PegNative => 2,
        ChatFormat::PegGemma4 => 3,
    };
    for component in [
        summary.prompt_bytes,
        summary.grammar_bytes,
        summary.generation_prompt_bytes,
        summary.trigger_count,
        summary.preserved_token_count,
        summary.stop_count,
        summary.parser_bytes,
        summary.span_count,
    ] {
        value = value.wrapping_mul(SEMANTIC_CHECKSUM_PRIME) ^ component as u64;
    }
    value
}

const SEMANTIC_CHECKSUM_OFFSET: u64 = 1_469_598_103_934_665_603;
const SEMANTIC_CHECKSUM_PRIME: u64 = 1_099_511_628_211;

const fn fold_iteration_semantic(accumulator: u64, iteration_semantic: u64) -> u64 {
    (accumulator ^ iteration_semantic).wrapping_mul(SEMANTIC_CHECKSUM_PRIME)
}

fn positive_duration_ns(started: Instant) -> Result<u64, ProbeError> {
    let duration = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
        ProbeError::runtime("duration-overflow", "measured duration exceeds u64::MAX ns")
    })?;
    Ok(duration.max(1))
}

fn timed_probe_evidence(output: Value, duration_ns: u64) -> Value {
    json!({
        "schemaVersion": 1,
        "output": output,
        "measurements": [
            {"name": "duration", "unit": "ns", "samples": [duration_ns]},
        ],
        "effectiveConfiguration": {},
    })
}

fn chat_parser_inspect(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: ChatParserInspectInput = decode_input(input)?;
    validate_chat_parser_contract(
        &input.template_fixture,
        &input.input_fixture,
        &input.content_fixture,
        input.use_jinja,
        input.force_pure_content,
        input.parser_options,
    )?;
    let fixtures = load_parser_fixtures(&input.fixture_paths)?;
    let prepared = prepare_content_only_chat(&fixtures, input.use_jinja, input.force_pure_content)?;
    let parser_options = input.parser_options.into_native();
    let snapshot_parser = prepared
        .parser(parser_options)
        .map_err(|error| ProbeError::runtime("chat-parser-init", error))?;
    let snapshot = snapshot_parser
        .parse_final(&fixtures.input)
        .map_err(|error| ProbeError::runtime("chat-parser-final", error))?;
    let snapshot_evidence = parsed_chat_message_evidence(&snapshot);

    let mut all_finals_match = true;
    let mut partitions = Vec::with_capacity(fixtures.chunkings.len());
    for (partition_index, partition) in fixtures.chunkings.iter().enumerate() {
        let mut stream = prepared
            .stream_parser(parser_options)
            .map_err(|error| ProbeError::runtime("chat-stream-init", error))?;
        let mut pushes = Vec::with_capacity(partition.len());
        for (chunk_index, chunk) in partition.iter().enumerate() {
            let deltas = stream
                .push(chunk)
                .map_err(|error| ProbeError::runtime("chat-stream-push", error))?;
            pushes.push(json!({
                "index": chunk_index,
                "chunkByteLength": chunk.len(),
                "deltas": project_chat_semantic_deltas(&deltas),
            }));
        }
        let (final_message, finish_deltas) = stream
            .finish()
            .map_err(|error| ProbeError::runtime("chat-stream-finish", error))?;
        let matches_snapshot = final_message == snapshot;
        all_finals_match &= matches_snapshot;
        partitions.push(json!({
            "index": partition_index,
            "pushes": pushes,
            "finishDeltas": project_chat_semantic_deltas(&finish_deltas),
            "final": parsed_chat_message_evidence(&final_message),
            "matchesSnapshot": matches_snapshot,
        }));
    }

    Ok(json!({
        "prepared": prepared_content_only_evidence(input.use_jinja, input.force_pure_content),
        "snapshot": snapshot_evidence,
        "partitions": partitions,
        "allFinalsMatch": all_finals_match,
    }))
}

fn chat_parser_bench(input: &Map<String, Value>) -> Result<Value, ProbeError> {
    let input: ChatParserBenchInput = decode_input(input)?;
    validate_chat_parser_contract(
        &input.template_fixture,
        &input.input_fixture,
        &input.content_fixture,
        input.use_jinja,
        input.force_pure_content,
        input.parser_options,
    )?;
    if input.iterations == 0 {
        return Err(ProbeError::invalid(
            "invalid-iterations",
            "iterations must be positive",
        ));
    }

    // Fixture I/O, template construction, chat preparation, and reusable
    // parser construction are all deliberately outside warmup and timing.
    let fixtures = load_parser_fixtures(&input.fixture_paths)?;
    let prepared = prepare_content_only_chat(&fixtures, input.use_jinja, input.force_pure_content)?;
    let parser = prepared
        .parser(input.parser_options.into_native())
        .map_err(|error| ProbeError::runtime("chat-parser-init", error))?;
    let mut semantic_sink = 0_u64;
    for _ in 0..input.warmup_iterations {
        let parsed = parser
            .parse_final(&fixtures.input)
            .map_err(|error| ProbeError::runtime("chat-parser-final", error))?;
        semantic_sink ^= semantic_message_checksum(&parsed);
    }

    let mut measured = None;
    let started = Instant::now();
    for _ in 0..input.iterations {
        let parsed = parser
            .parse_final(&fixtures.input)
            .map_err(|error| ProbeError::runtime("chat-parser-final", error))?;
        semantic_sink ^= semantic_message_checksum(&parsed);
        measured = Some(parsed);
    }
    let duration_ns = positive_duration_ns(started)?;
    std::hint::black_box(semantic_sink);
    let message = measured.expect("positive iterations produce a parsed message");
    Ok(timed_probe_evidence(
        json!({
            "prepared": prepared_content_only_evidence(
                input.use_jinja,
                input.force_pure_content,
            ),
            "inputByteLength": fixtures.input.len(),
            "message": parsed_chat_message_evidence(&message),
        }),
        duration_ns,
    ))
}

fn validate_chat_parser_contract(
    template_fixture: &str,
    input_fixture: &str,
    content_fixture: &str,
    use_jinja: bool,
    force_pure_content: bool,
    parser_options: ChatParserOptionsInput,
) -> Result<(), ProbeError> {
    if template_fixture != "fixtures/templates/chatml-basic.jinja"
        || input_fixture != "fixtures/templates/chatml-basic-input.json"
        || content_fixture != "fixtures/parser/content-replay.json"
    {
        return Err(ProbeError::invalid(
            "invalid-parser-fixture",
            "chat parser probes require the pinned ChatML template, input, and content fixtures",
        ));
    }
    if use_jinja || !force_pure_content {
        return Err(ProbeError::invalid(
            "invalid-parser-boundary",
            "content-only parser probes require use_jinja=false and force_pure_content=true",
        ));
    }
    if parser_options != ChatParserOptionsInput::CONTENT_ONLY {
        return Err(ProbeError::invalid(
            "invalid-parser-options",
            "the pinned content-only parser options must all be false",
        ));
    }
    Ok(())
}

fn load_parser_fixtures(paths: &ChatParserFixturePaths) -> Result<ParserFixtureData, ProbeError> {
    let template_source = fs::read_to_string(&paths.template)
        .map_err(|error| ProbeError::runtime("fixture-read", error))?;
    let request_source = fs::read_to_string(&paths.input)
        .map_err(|error| ProbeError::runtime("fixture-read", error))?;
    let request: Value = serde_json::from_str(&request_source)
        .map_err(|error| ProbeError::invalid("invalid-chat-fixture", error))?;
    let request = request.as_object().cloned().ok_or_else(|| {
        ProbeError::invalid(
            "invalid-chat-fixture",
            "chat input fixture must contain an object",
        )
    })?;

    let content_source = fs::read_to_string(&paths.content)
        .map_err(|error| ProbeError::runtime("fixture-read", error))?;
    let content: Value = serde_json::from_str(&content_source)
        .map_err(|error| ProbeError::invalid("invalid-parser-fixture", error))?;
    let content = content.as_object().ok_or_else(|| {
        ProbeError::invalid(
            "invalid-parser-fixture",
            "content fixture must contain an object",
        )
    })?;
    let input = chat_request_required::<String>(content, "input")?;
    let chunkings = chat_request_required::<Vec<Vec<String>>>(content, "chunkings")?;
    if chunkings.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-parser-fixture",
            "content fixture must declare at least one chunk partition",
        ));
    }
    for partition in &chunkings {
        if partition.is_empty() {
            return Err(ProbeError::invalid(
                "invalid-parser-fixture",
                "chunk partitions must not be empty",
            ));
        }
        if partition.concat() != input {
            return Err(ProbeError::invalid(
                "invalid-parser-fixture",
                "each chunk partition must concatenate exactly to input",
            ));
        }
    }
    Ok(ParserFixtureData {
        template_source,
        request,
        input,
        chunkings,
    })
}

fn prepare_content_only_chat(
    fixtures: &ParserFixtureData,
    use_jinja: bool,
    force_pure_content: bool,
) -> Result<PreparedChat, ProbeError> {
    let bos = chat_request_optional::<String>(&fixtures.request, "bos_token", String::new())?;
    let eos = chat_request_optional::<String>(&fixtures.request, "eos_token", String::new())?;
    let templates = CommonChatTemplates::from_template(
        &fixtures.template_source,
        (!bos.is_empty()).then_some(bos.as_str()),
        (!eos.is_empty()).then_some(eos.as_str()),
    )
    .map_err(|error| ProbeError::runtime("chat-template-init", error))?;
    let mut request = fixtures.request.clone();
    request.insert("use_jinja".to_owned(), json!(use_jinja));
    request.insert("force_pure_content".to_owned(), json!(force_pure_content));
    let prepared = chat_bench_prepare(&templates, &request)?;
    if prepared.format() != ChatFormat::ContentOnly || !prepared.parser_definition().is_empty() {
        return Err(ProbeError::runtime(
            "invalid-parser-boundary",
            "pinned legacy ChatML preparation did not select the content-only parser boundary",
        ));
    }
    Ok(prepared)
}

fn prepared_content_only_evidence(use_jinja: bool, force_pure_content: bool) -> Value {
    json!({
        "format": "content-only",
        "useJinja": use_jinja,
        "forcePureContent": force_pure_content,
    })
}

fn parsed_chat_message_evidence(message: &ParsedChatMessage) -> Value {
    json!({
        "role": message.role,
        "content": message.content,
        "toolCalls": message
            .tool_calls
            .iter()
            .map(|call| json!({
                "name": call.name,
                "arguments": call.arguments,
                "id": normalized_optional_string(call.id.as_deref()),
            }))
            .collect::<Vec<_>>(),
        "reasoningContent": normalized_optional_string(message.reasoning_content.as_deref()),
        "toolName": normalized_optional_string(message.tool_name.as_deref()),
        "toolCallId": normalized_optional_string(message.tool_call_id.as_deref()),
    })
}

fn project_chat_semantic_deltas(deltas: &[ChatSemanticDelta]) -> Vec<Value> {
    deltas
        .iter()
        .map(|delta| match delta {
            ChatSemanticDelta::Reasoning(text) => {
                json!({"kind": "reasoning", "text": text})
            }
            ChatSemanticDelta::Content(text) => json!({"kind": "content", "text": text}),
            ChatSemanticDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } => json!({
                "kind": "tool-call",
                "index": index,
                "id": id,
                "name": name,
                "arguments": arguments,
            }),
        })
        .collect()
}

fn normalized_optional_string(value: Option<&str>) -> Value {
    value
        .filter(|value| !value.is_empty())
        .map_or(Value::Null, |value| json!(value))
}

fn semantic_message_checksum(message: &ParsedChatMessage) -> u64 {
    let mut result = SEMANTIC_CHECKSUM_OFFSET;
    for value in [
        message.role.as_str(),
        message.content.as_str(),
        message.reasoning_content.as_deref().unwrap_or_default(),
        message.tool_name.as_deref().unwrap_or_default(),
        message.tool_call_id.as_deref().unwrap_or_default(),
    ] {
        include_semantic_checksum_value(&mut result, value);
    }
    for call in &message.tool_calls {
        include_semantic_checksum_value(&mut result, call.id.as_deref().unwrap_or_default());
        include_semantic_checksum_value(&mut result, &call.name);
        include_semantic_checksum_value(&mut result, &call.arguments);
    }
    result ^ message.tool_calls.len() as u64
}

fn include_semantic_checksum_value(result: &mut u64, value: &str) {
    for byte in value.bytes() {
        *result = (*result ^ u64::from(byte)).wrapping_mul(SEMANTIC_CHECKSUM_PRIME);
    }
    *result = (*result ^ 255).wrapping_mul(SEMANTIC_CHECKSUM_PRIME);
}

fn make_sampler(input: &SamplerDefinition) -> Result<LlamaSampler, ProbeError> {
    Ok(match input.kind.as_str() {
        "top-k" => LlamaSampler::top_k(required_i32(input.k, "k")?),
        "top-p" => LlamaSampler::top_p(required_f32(input.p, "p")?, input.min_keep),
        "min-p" => LlamaSampler::min_p(required_f32(input.p, "p")?, input.min_keep),
        "typical" => LlamaSampler::typical(required_f32(input.p, "p")?, input.min_keep),
        "temperature" => LlamaSampler::temp(required_f32(input.temperature, "temperature")?),
        "temperature-ext" => LlamaSampler::temp_ext(
            required_f32(input.temperature, "temperature")?,
            required_f32(input.delta, "delta")?,
            required_f32(input.exponent, "exponent")?,
        ),
        "xtc" => LlamaSampler::xtc(
            required_f32(input.probability, "probability")?,
            required_f32(input.threshold, "threshold")?,
            input.min_keep,
            input.seed,
        ),
        "top-n-sigma" => LlamaSampler::top_n_sigma(required_f32(input.n, "n")?),
        "penalties" => LlamaSampler::penalties(
            required_i32(input.last_n, "lastN")?,
            required_f32(input.repeat, "repeat")?,
            required_f32(input.frequency, "frequency")?,
            required_f32(input.presence, "presence")?,
        ),
        value => {
            return Err(ProbeError::invalid(
                "invalid-sampler",
                format!("unsupported sampler type: {value}"),
            ));
        }
    })
}

fn required_f32(value: Option<f32>, field: &'static str) -> Result<f32, ProbeError> {
    value.ok_or_else(|| ProbeError::invalid("invalid-sampler", format!("missing field: {field}")))
}

fn required_i32(value: Option<i32>, field: &'static str) -> Result<i32, ProbeError> {
    value.ok_or_else(|| ProbeError::invalid("invalid-sampler", format!("missing field: {field}")))
}

fn finite_number(value: f32) -> (Value, &'static str) {
    if value.is_finite() {
        (json!(value), "finite")
    } else if value.is_nan() {
        (Value::Null, "nan")
    } else if value.is_sign_positive() {
        (Value::Null, "positive-infinity")
    } else {
        (Value::Null, "negative-infinity")
    }
}

fn chat_format_name(value: ChatFormat) -> &'static str {
    match value {
        ChatFormat::ContentOnly => "Content-only",
        ChatFormat::PegSimple => "peg-simple",
        ChatFormat::PegNative => "peg-native",
        ChatFormat::PegGemma4 => "peg-gemma4",
    }
}

fn parse_reasoning_format(value: &str) -> Result<ChatReasoningFormat, ProbeError> {
    match value {
        "none" => Ok(ChatReasoningFormat::None),
        "auto" => Ok(ChatReasoningFormat::Auto),
        "deepseek-legacy" => Ok(ChatReasoningFormat::DeepSeekLegacy),
        "deepseek" => Ok(ChatReasoningFormat::DeepSeek),
        value => Err(ProbeError::invalid(
            "invalid-reasoning-format",
            format!("unknown reasoning format: {value}"),
        )),
    }
}

fn parse_tool_choice(value: &str) -> Result<ChatToolChoice, ProbeError> {
    match value {
        "auto" => Ok(ChatToolChoice::Auto),
        "required" => Ok(ChatToolChoice::Required),
        "none" => Ok(ChatToolChoice::None),
        value => Err(ProbeError::invalid(
            "invalid-tool-choice",
            format!("unknown tool choice: {value}"),
        )),
    }
}

fn parse_continuation(value: Option<&Value>) -> Result<ChatContinuation, ProbeError> {
    match value {
        None | Some(Value::Bool(false)) | Some(Value::Null) => Ok(ChatContinuation::None),
        Some(Value::Bool(true)) => Ok(ChatContinuation::Auto),
        Some(Value::String(value)) => match value.as_str() {
            "none" => Ok(ChatContinuation::None),
            "auto" => Ok(ChatContinuation::Auto),
            "reasoning" => Ok(ChatContinuation::Reasoning),
            "content" => Ok(ChatContinuation::Content),
            value => Err(ProbeError::invalid(
                "invalid-continuation",
                format!("unknown continuation: {value}"),
            )),
        },
        Some(_) => Err(ProbeError::invalid(
            "invalid-continuation",
            "continueFinalMessage must be a boolean or string",
        )),
    }
}

struct LoadedModel {
    // Fields are declared in dependency order so the model is dropped before
    // the backend registration it relies on.
    model: LlamaModel,
    backend: LlamaBackend,
}

#[derive(Debug)]
struct ModelLoadOptions {
    n_gpu_layers: u32,
    use_mmap: bool,
    use_mlock: bool,
}

impl Default for ModelLoadOptions {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
        }
    }
}

fn load_model(path: &Path, vocab_only: bool) -> Result<LoadedModel, ProbeError> {
    load_model_with_options(path, vocab_only, &ModelLoadOptions::default())
}

fn load_model_with_options(
    path: &Path,
    vocab_only: bool,
    options: &ModelLoadOptions,
) -> Result<LoadedModel, ProbeError> {
    if !path.is_file() {
        return Err(ProbeError::invalid(
            "invalid-model-path",
            format!("modelPath is not a regular file: {}", path.display()),
        ));
    }
    let backend =
        LlamaBackend::init().map_err(|error| ProbeError::runtime("backend-initialize", error))?;
    let params = LlamaModelParams::default()
        .with_n_gpu_layers(options.n_gpu_layers)
        .with_vocab_only(vocab_only)
        .with_use_mmap(options.use_mmap)
        .with_use_mlock(options.use_mlock);
    let model = LlamaModel::load_from_file(&backend, path, &params)
        .map_err(|error| ProbeError::runtime("model-load", error))?;
    Ok(LoadedModel { model, backend })
}

fn token_piece_bytes(
    model: &LlamaModel,
    token: LlamaToken,
    special: bool,
) -> Result<Vec<u8>, ProbeError> {
    match model.token_to_piece_bytes(token, 8, special, None) {
        Ok(bytes) => Ok(bytes),
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(required)) => {
            let required = required.checked_neg().ok_or_else(|| {
                ProbeError::runtime("token-piece", "native piece length overflowed i32")
            })?;
            let required = usize::try_from(required).map_err(|error| {
                ProbeError::runtime("token-piece", format!("invalid piece length: {error}"))
            })?;
            model
                .token_to_piece_bytes(token, required, special, None)
                .map_err(|error| ProbeError::runtime("token-piece", error))
        }
        Err(error) => Err(ProbeError::runtime("token-piece", error)),
    }
}

fn rope_type_name(value: Option<RopeType>) -> &'static str {
    match value {
        None => "none",
        Some(RopeType::Norm) => "norm",
        Some(RopeType::NeoX) => "neox",
        Some(RopeType::MRope) => "mrope",
        Some(RopeType::Vision) => "vision",
    }
}

fn vocab_type_name(value: VocabType) -> &'static str {
    match value {
        VocabType::BPE => "bpe",
        VocabType::SPM => "spm",
    }
}

fn validate_context_request(input: &ContextRequest) -> Result<(), ProbeError> {
    if input.context_tokens == 0 || input.context_tokens > i32::MAX.cast_unsigned() {
        return Err(ProbeError::invalid(
            "invalid-context",
            "context_tokens must be between 1 and i32::MAX",
        ));
    }
    if input.batch_tokens == 0 || input.batch_tokens > i32::MAX.cast_unsigned() {
        return Err(ProbeError::invalid(
            "invalid-context",
            "batch_tokens must be between 1 and i32::MAX",
        ));
    }
    if input.micro_batch_tokens == 0 || input.micro_batch_tokens > input.batch_tokens {
        return Err(ProbeError::invalid(
            "invalid-context",
            "micro_batch_tokens must be between 1 and batch_tokens",
        ));
    }
    if input.sequences == 0 || input.sequences > i32::MAX.cast_unsigned() {
        return Err(ProbeError::invalid(
            "invalid-context",
            "sequences must be between 1 and i32::MAX",
        ));
    }
    nonzero_i32(input.threads, "threads")?;
    nonzero_i32(input.batch_threads, "batch_threads")?;
    if input.embeddings {
        return Err(ProbeError::invalid(
            "unsupported-embeddings",
            "production ResolvedExecutionPlan does not expose embeddings; embeddings must be false",
        ));
    }
    Ok(())
}

fn validate_explicit_cpu_context(
    offload_kqv: bool,
    operation_offload: bool,
    flash_attention: &str,
    threads: u32,
    batch_threads: u32,
) -> Result<(), ProbeError> {
    if offload_kqv || operation_offload || flash_attention != "off" {
        return Err(ProbeError::invalid(
            "invalid-context-configuration",
            "CPU-resident decode/state probes require offloadKqv=false, operationOffload=false, and flashAttention='off'",
        ));
    }
    nonzero_i32(threads, "threads")?;
    nonzero_i32(batch_threads, "batchThreads")?;
    Ok(())
}

fn validate_decode_plan(input: &DecodePlanInput) -> Result<(), ProbeError> {
    if input.context_tokens == 0 || input.context_tokens > i32::MAX.cast_unsigned() {
        return Err(ProbeError::invalid(
            "invalid-context",
            "context_tokens must be between 1 and i32::MAX",
        ));
    }
    if input.batch.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-batch",
            "batch must not be empty",
        ));
    }
    if input.batch.len() > i32::MAX as usize {
        return Err(ProbeError::invalid(
            "invalid-batch",
            "batch length exceeds i32::MAX",
        ));
    }
    if input.logit_indices.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-logit-indices",
            "logit_indices must not be empty",
        ));
    }
    for (index, item) in input.batch.iter().enumerate() {
        if item.position < 0 || item.position.cast_unsigned() >= input.context_tokens {
            return Err(ProbeError::invalid(
                "invalid-position",
                format!("batch item {index} position is outside the requested context"),
            ));
        }
        if item.sequence_ids.is_empty() {
            return Err(ProbeError::invalid(
                "invalid-sequence-id",
                format!("batch item {index} has no sequence IDs"),
            ));
        }
        if item.sequence_ids.iter().any(|sequence| *sequence < 0) {
            return Err(ProbeError::invalid(
                "invalid-sequence-id",
                format!("batch item {index} has a negative sequence ID"),
            ));
        }
        let mut unique = item.sequence_ids.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != item.sequence_ids.len() {
            return Err(ProbeError::invalid(
                "invalid-sequence-id",
                format!("batch item {index} contains duplicate sequence IDs"),
            ));
        }
    }
    Ok(())
}

fn validate_state_script(input: &StateScriptInput) -> Result<(), ProbeError> {
    if input.sequence_id < 0 {
        return Err(ProbeError::invalid(
            "invalid-sequence-id",
            "sequence_id must be non-negative",
        ));
    }
    if input.prepare_tokens.len() > i32::MAX as usize {
        return Err(ProbeError::invalid(
            "invalid-prepare",
            "prepare_tokens length exceeds i32::MAX",
        ));
    }
    if input.prepare_tokens.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-prepare",
            "prepare_tokens must not be empty",
        ));
    }
    if input.operations.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-script",
            "operations must not be empty",
        ));
    }
    if input.logit_indices.is_empty() {
        return Err(ProbeError::invalid(
            "invalid-logit-indices",
            "logit_indices must not be empty",
        ));
    }
    for (index, operation) in input.operations.iter().enumerate() {
        match operation {
            StateOperationInput::Remove {
                position_start,
                position_end,
            } if position_start >= position_end => {
                return Err(ProbeError::invalid(
                    "invalid-position-range",
                    format!("operation {index} must name a non-empty position range"),
                ));
            }
            StateOperationInput::Decode { position, .. } if *position < 0 => {
                return Err(ProbeError::invalid(
                    "invalid-position",
                    format!("operation {index} has a negative decode position"),
                ));
            }
            _ => {}
        }
    }
    derived_state_context_tokens(input)?;
    Ok(())
}

fn derived_state_context_tokens(input: &StateScriptInput) -> Result<u32, ProbeError> {
    let mut required = u64::try_from(input.prepare_tokens.len()).map_err(|error| {
        ProbeError::invalid(
            "invalid-prepare",
            format!("invalid prepare length: {error}"),
        )
    })?;
    for operation in &input.operations {
        let end = match operation {
            StateOperationInput::Remove { .. } => 0,
            StateOperationInput::Decode { position, .. } => {
                u64::try_from(*position)
                    .map_err(|_| ProbeError::invalid("invalid-position", "negative position"))?
                    + 1
            }
        };
        required = required.max(end);
    }
    required = required.max(32);
    if required > i32::MAX as u64 {
        return Err(ProbeError::invalid(
            "invalid-context",
            "derived context size exceeds i32::MAX",
        ));
    }
    Ok(u32::try_from(required).expect("validated context size fits into u32"))
}

fn sequence_count_for(sequence_ids: impl Iterator<Item = i32>) -> Result<u32, ProbeError> {
    let maximum = sequence_ids.max().unwrap_or(0);
    if maximum < 0 {
        return Err(ProbeError::invalid(
            "invalid-sequence-id",
            "sequence IDs must be non-negative",
        ));
    }
    maximum
        .cast_unsigned()
        .checked_add(1)
        .ok_or_else(|| ProbeError::invalid("invalid-sequence-id", "sequence ID is too large"))
}

fn validate_token_ids(
    tokens: impl Iterator<Item = i32>,
    vocabulary_size: i32,
) -> Result<(), ProbeError> {
    if vocabulary_size <= 0 {
        return Err(ProbeError::runtime(
            "invalid-vocabulary",
            "loaded model reports no vocabulary tokens",
        ));
    }
    for token in tokens {
        if !(0..vocabulary_size).contains(&token) {
            return Err(ProbeError::invalid(
                "invalid-token",
                format!("token {token} is outside [0, {vocabulary_size})"),
            ));
        }
    }
    Ok(())
}

fn validate_logit_indices(indices: &[i32], vocabulary_size: i32) -> Result<(), ProbeError> {
    for index in indices {
        if !(0..vocabulary_size).contains(index) {
            return Err(ProbeError::invalid(
                "invalid-logit-index",
                format!("logit index {index} is outside [0, {vocabulary_size})"),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ConfigurationInspectInput {
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    context: ContextRequest,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextRequest {
    context_tokens: u32,
    batch_tokens: u32,
    micro_batch_tokens: u32,
    sequences: u32,
    threads: u32,
    batch_threads: u32,
    embeddings: bool,
    offload_kqv: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ModelMetadataInput {
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    #[serde(default, rename = "stable_fields")]
    _stable_fields: Vec<String>,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TokenizeInput {
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    text: String,
    #[serde(default)]
    add_special: bool,
    #[serde(default = "yes")]
    parse_special: bool,
    #[serde(default = "yes")]
    include_pieces: bool,
    #[serde(default = "yes")]
    piece_special: bool,
    #[serde(default)]
    n_gpu_layers: u32,
    #[serde(default = "yes")]
    use_mmap: bool,
    #[serde(default)]
    use_mlock: bool,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

impl TokenizeInput {
    fn load_options(&self) -> ModelLoadOptions {
        ModelLoadOptions {
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TokenToPieceInput {
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    tokens: Vec<i32>,
    special: bool,
    lstrip: Value,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReasoningBudgetInspectInput {
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    budget_tokens: u32,
    start_tag: String,
    end_tag: String,
    forced_message: String,
    controllable: bool,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct JsonSchemaGrammarInput {
    schema: Value,
    #[serde(default)]
    force_gbnf: bool,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecodeAbortInput {
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    n_gpu_layers: u32,
    use_mmap: bool,
    use_mlock: bool,
    context_tokens: u32,
    threads: u32,
    batch_threads: u32,
    offload_kqv: bool,
    operation_offload: bool,
    flash_attention: String,
    token: i32,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

impl DecodeAbortInput {
    fn load_options(&self) -> ModelLoadOptions {
        ModelLoadOptions {
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecodeAbortRecoveryInput {
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    n_gpu_layers: u32,
    use_mmap: bool,
    use_mlock: bool,
    context_tokens: u32,
    threads: u32,
    batch_threads: u32,
    offload_kqv: bool,
    operation_offload: bool,
    flash_attention: String,
    token: i32,
    logit_indices: Vec<i32>,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

impl DecodeAbortRecoveryInput {
    fn load_options(&self) -> ModelLoadOptions {
        ModelLoadOptions {
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodePlanInput {
    #[serde(rename = "modelPath")]
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    context_tokens: u32,
    batch: Vec<DecodeBatchItem>,
    logit_indices: Vec<i32>,
    #[serde(rename = "nGpuLayers")]
    n_gpu_layers: u32,
    #[serde(rename = "useMmap")]
    use_mmap: bool,
    #[serde(rename = "useMlock")]
    use_mlock: bool,
    #[serde(rename = "offloadKqv")]
    offload_kqv: bool,
    #[serde(rename = "operationOffload")]
    operation_offload: bool,
    #[serde(rename = "flashAttention")]
    flash_attention: String,
    threads: u32,
    #[serde(rename = "batchThreads")]
    batch_threads: u32,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

impl DecodePlanInput {
    fn load_options(&self) -> ModelLoadOptions {
        ModelLoadOptions {
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodeBatchItem {
    token: i32,
    position: i32,
    sequence_ids: Vec<i32>,
    request_logits: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateScriptInput {
    #[serde(rename = "modelPath")]
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    prepare_tokens: Vec<i32>,
    sequence_id: i32,
    operations: Vec<StateOperationInput>,
    #[serde(default = "default_logit_indices")]
    logit_indices: Vec<i32>,
    #[serde(rename = "nGpuLayers")]
    n_gpu_layers: u32,
    #[serde(rename = "useMmap")]
    use_mmap: bool,
    #[serde(rename = "useMlock")]
    use_mlock: bool,
    #[serde(rename = "offloadKqv")]
    offload_kqv: bool,
    #[serde(rename = "operationOffload")]
    operation_offload: bool,
    #[serde(rename = "flashAttention")]
    flash_attention: String,
    threads: u32,
    #[serde(rename = "batchThreads")]
    batch_threads: u32,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

impl StateScriptInput {
    fn load_options(&self) -> ModelLoadOptions {
        ModelLoadOptions {
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
enum StateOperationInput {
    Remove {
        position_start: u32,
        position_end: u32,
    },
    Decode {
        token: i32,
        position: i32,
        request_logits: bool,
    },
}

fn default_logit_indices() -> Vec<i32> {
    vec![0, 1, 2, 3]
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaBatchedBenchInput {
    #[serde(rename = "modelPath")]
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    parallel_sequences: u32,
    shared_prompt: bool,
    prompt_tokens: u32,
    generation_tokens_per_sequence: u32,
    context_tokens: u32,
    batch_tokens: u32,
    micro_batch_tokens: u32,
    kv_unified: bool,
    expected_kv_tokens: u32,
    separate_generation: bool,
    repetitions: u32,
    effective_engine_configuration: String,
    #[serde(rename = "engineConfiguration")]
    engine_configuration: LlamaBenchEngineConfiguration,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
}

impl LlamaBatchedBenchInput {
    fn validate(&self) -> Result<(), ProbeError> {
        if self.effective_engine_configuration != "profile" {
            return Err(ProbeError::invalid(
                "invalid-engine-configuration",
                "effective_engine_configuration must be 'profile'",
            ));
        }
        if self.engine_configuration.backend.trim().is_empty() {
            return Err(ProbeError::invalid(
                "invalid-engine-configuration",
                "engineConfiguration.backend must not be empty",
            ));
        }
        nonzero_i32(self.engine_configuration.threads, "threads")?;
        if self.engine_configuration.cpu_strict || self.engine_configuration.threadpool_poll != 50 {
            return Err(ProbeError::invalid(
                "unsupported-threadpool-policy",
                "llama-batched-bench parity requires the upstream default CPU pool policy",
            ));
        }
        if self.repetitions != 1 {
            return Err(ProbeError::invalid(
                "invalid-repetitions",
                "repetitions must be exactly one; the parity runner owns process-level repetition",
            ));
        }
        if self.parallel_sequences == 0 || self.parallel_sequences > i32::MAX.cast_unsigned() {
            return Err(ProbeError::invalid(
                "invalid-sequence-count",
                "parallel_sequences must be in [1, i32::MAX]",
            ));
        }
        if self.prompt_tokens == 0 || self.generation_tokens_per_sequence == 0 {
            return Err(ProbeError::invalid(
                "invalid-workload-shape",
                "prompt and per-sequence generation lengths must be positive",
            ));
        }
        if self.batch_tokens == 0
            || self.batch_tokens > i32::MAX.cast_unsigned()
            || self.micro_batch_tokens == 0
            || self.micro_batch_tokens > self.batch_tokens
        {
            return Err(ProbeError::invalid(
                "invalid-batch",
                "batch_tokens must be in [1, i32::MAX] and micro_batch_tokens in [1, batch_tokens]",
            ));
        }
        let prompt_work =
            self.prompt_tokens
                .checked_mul(if self.shared_prompt && self.kv_unified {
                    1
                } else {
                    self.parallel_sequences
                });
        let generation_work = self
            .generation_tokens_per_sequence
            .checked_mul(self.parallel_sequences);
        let expected = prompt_work.and_then(|prompt| {
            generation_work.and_then(|generation| prompt.checked_add(generation))
        });
        if expected != Some(self.expected_kv_tokens) {
            return Err(ProbeError::invalid(
                "invalid-kv-work",
                "expected_kv_tokens does not match the declared workload",
            ));
        }
        if self.context_tokens < self.expected_kv_tokens {
            return Err(ProbeError::invalid(
                "invalid-context",
                "context_tokens is smaller than the declared KV work",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaBenchInput {
    #[serde(rename = "modelPath")]
    model_path: PathBuf,
    #[serde(default, rename = "model_id")]
    _case_model_id: Option<String>,
    #[serde(default, rename = "modelId")]
    _runtime_model_id: Option<String>,
    prompt_tokens: u32,
    generation_tokens: u32,
    context_depth: u32,
    batch_tokens: u32,
    micro_batch_tokens: u32,
    repetitions: u32,
    warmup: bool,
    effective_engine_configuration: String,
    token_schedule: String,
    #[serde(rename = "engineConfiguration")]
    engine_configuration: LlamaBenchEngineConfiguration,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LlamaBenchEngineConfiguration {
    backend: String,
    threads: u32,
    gpu_layers: LlamaBenchGpuLayers,
    flash_attention: String,
    cpu_strict: bool,
    threadpool_poll: u32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LlamaBenchGpuLayers {
    Name(String),
    Count(u32),
}

impl LlamaBenchGpuLayers {
    fn normalized(&self) -> Result<(GpuLayers, Value), ProbeError> {
        match self {
            Self::Name(value) if value == "all" => Ok((GpuLayers::All, json!("all"))),
            Self::Name(value) => Err(ProbeError::invalid(
                "invalid-engine-configuration",
                format!("unsupported gpuLayers policy: {value}"),
            )),
            Self::Count(value) => Ok((GpuLayers::Count(*value), json!(value))),
        }
    }

    fn batched_bench_value(&self) -> Result<Value, ProbeError> {
        match self {
            // common_params serializes LLAMA_MODEL_DEFAULT (the value used by
            // --n-gpu-layers all) as the upstream numeric sentinel -2.
            Self::Name(value) if value == "all" => Ok(json!(-2)),
            Self::Name(value) => Err(ProbeError::invalid(
                "invalid-engine-configuration",
                format!("unsupported gpuLayers policy: {value}"),
            )),
            Self::Count(value) => Ok(json!(value)),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ChatTemplateInput {
    template: String,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default = "yes")]
    add_generation_prompt: bool,
    messages: Vec<ChatMessageInput>,
    #[serde(default)]
    tools: Vec<ChatToolInput>,
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    json_schema: Option<Value>,
    #[serde(default)]
    continue_final_message: Option<Value>,
    #[serde(default = "yes")]
    use_jinja: bool,
    #[serde(default = "default_auto")]
    tool_choice: String,
    #[serde(default)]
    parallel_tool_calls: bool,
    #[serde(default = "default_none")]
    reasoning_format: String,
    #[serde(default = "yes")]
    enable_thinking: bool,
    #[serde(default)]
    force_pure_content: bool,
    #[serde(default)]
    chat_template_kwargs: Map<String, Value>,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatTemplateBenchInput {
    template_fixture: String,
    input_fixture: String,
    warmup_iterations: u32,
    iterations: u32,
    #[serde(rename = "fixturePaths")]
    fixture_paths: ChatTemplateBenchFixturePaths,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatTemplateBenchFixturePaths {
    #[serde(rename = "chatml-basic-template")]
    template: PathBuf,
    #[serde(rename = "chatml-basic-input")]
    input: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatParserInspectInput {
    template_fixture: String,
    input_fixture: String,
    content_fixture: String,
    use_jinja: bool,
    force_pure_content: bool,
    parser_options: ChatParserOptionsInput,
    #[serde(rename = "fixturePaths")]
    fixture_paths: ChatParserFixturePaths,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatParserBenchInput {
    template_fixture: String,
    input_fixture: String,
    content_fixture: String,
    use_jinja: bool,
    force_pure_content: bool,
    parser_options: ChatParserOptionsInput,
    warmup_iterations: u32,
    iterations: u32,
    #[serde(rename = "fixturePaths")]
    fixture_paths: ChatParserFixturePaths,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChatParserOptionsInput {
    reasoning_in_content: bool,
    parse_tool_calls: bool,
    is_continuation: bool,
    echo: bool,
    debug: bool,
}

impl ChatParserOptionsInput {
    const CONTENT_ONLY: Self = Self {
        reasoning_in_content: false,
        parse_tool_calls: false,
        is_continuation: false,
        echo: false,
        debug: false,
    };

    const fn into_native(self) -> ChatParserOptions {
        ChatParserOptions {
            reasoning_in_content: self.reasoning_in_content,
            parse_tool_calls: self.parse_tool_calls,
            is_continuation: self.is_continuation,
            echo: self.echo,
            debug: self.debug,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatParserFixturePaths {
    #[serde(rename = "chatml-basic-template")]
    template: PathBuf,
    #[serde(rename = "chatml-basic-input")]
    input: PathBuf,
    #[serde(rename = "content-replay")]
    content: PathBuf,
}

struct ParserFixtureData {
    template_source: String,
    request: Map<String, Value>,
    input: String,
    chunkings: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessageInput {
    role: String,
    content: Option<Value>,
    #[serde(default, rename = "tool_calls")]
    tool_calls: Vec<ChatToolCallInput>,
    #[serde(default, rename = "reasoning_content")]
    reasoning_content: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "tool_call_id")]
    tool_call_id: Option<String>,
}

impl ChatMessageInput {
    fn into_native(self) -> Result<ChatMessage, ProbeError> {
        let content = self
            .content
            .map(|value| match value {
                Value::String(value) => Ok(ChatContent::Text(value)),
                Value::Null => Ok(ChatContent::Text(String::new())),
                _ => Err(ProbeError::invalid(
                    "invalid-message-content",
                    "the parity chat-template operation currently accepts string content",
                )),
            })
            .transpose()?;
        Ok(ChatMessage {
            role: self.role,
            content,
            tool_calls: self
                .tool_calls
                .into_iter()
                .map(ChatToolCallInput::into_native)
                .collect::<Result<Vec<_>, _>>()?,
            reasoning_content: self.reasoning_content,
            tool_name: self.name,
            tool_call_id: self.tool_call_id,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatToolCallInput {
    #[serde(default)]
    id: Option<String>,
    function: ChatFunctionCallInput,
}

impl ChatToolCallInput {
    fn into_native(self) -> Result<llama_cpp_2::common_chat::ChatToolCall, ProbeError> {
        Ok(llama_cpp_2::common_chat::ChatToolCall {
            name: self.function.name,
            arguments: match self.function.arguments {
                Value::String(value) => value,
                value => serde_json::to_string(&value)
                    .map_err(|error| ProbeError::invalid("invalid-tool-call", error))?,
            },
            id: self.id,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatFunctionCallInput {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatToolInput {
    #[serde(rename = "type")]
    kind: String,
    function: ChatFunctionDefinitionInput,
}

impl ChatToolInput {
    fn into_native(self) -> Result<ChatTool, ProbeError> {
        if self.kind != "function" {
            return Err(ProbeError::invalid(
                "invalid-tool",
                format!("unsupported tool type: {}", self.kind),
            ));
        }
        Ok(ChatTool {
            name: self.function.name,
            description: self.function.description.unwrap_or_default(),
            parameters_json: serde_json::to_string(&self.function.parameters)
                .map_err(|error| ProbeError::invalid("invalid-tool", error))?,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatFunctionDefinitionInput {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SamplerInput {
    candidates: Vec<CandidateInput>,
    #[serde(default)]
    samplers: Vec<SamplerDefinition>,
    #[serde(default)]
    selection: SelectionInput,
    #[serde(default)]
    accepted_tokens: Vec<i32>,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplerBenchInput {
    candidate_count: usize,
    candidate_generator: SamplerBenchCandidateGenerator,
    sampler: SamplerBenchDefinition,
    warmup_iterations: u32,
    iterations: u32,
    #[serde(default, rename = "fixturePaths")]
    _fixture_paths: Map<String, Value>,
    #[serde(default, rename = "engineConfiguration")]
    _engine_configuration: Option<LlamaBenchEngineConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplerBenchCandidateGenerator {
    kind: String,
    seed: u32,
    minimum: f32,
    maximum: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplerBenchDefinition {
    #[serde(rename = "type")]
    kind: String,
    k: i32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateInput {
    id: i32,
    logit: f32,
    #[serde(default)]
    probability: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SamplerDefinition {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    k: Option<i32>,
    #[serde(default)]
    p: Option<f32>,
    #[serde(default = "one_usize")]
    min_keep: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    delta: Option<f32>,
    #[serde(default)]
    exponent: Option<f32>,
    #[serde(default)]
    probability: Option<f32>,
    #[serde(default)]
    threshold: Option<f32>,
    #[serde(default)]
    seed: u32,
    #[serde(default)]
    n: Option<f32>,
    #[serde(default)]
    last_n: Option<i32>,
    #[serde(default)]
    repeat: Option<f32>,
    #[serde(default)]
    frequency: Option<f32>,
    #[serde(default)]
    presence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SelectionInput {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    seed: u32,
    #[serde(flatten)]
    raw_extra: Map<String, Value>,
}

impl Default for SelectionInput {
    fn default() -> Self {
        Self {
            kind: "distribution".to_owned(),
            seed: 0,
            raw_extra: Map::new(),
        }
    }
}

impl SelectionInput {
    fn normalized_raw(&self) -> Value {
        let mut raw = self.raw_extra.clone();
        raw.insert("type".to_owned(), json!(self.kind));
        raw.insert("seed".to_owned(), json!(self.seed));
        Value::Object(raw)
    }
}

const fn yes() -> bool {
    true
}

fn default_auto() -> String {
    "auto".to_owned()
}

fn default_none() -> String {
    "none".to_owned()
}

const fn one_usize() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine_configuration() -> Value {
        json!({
            "backend": "metal",
            "threads": 4,
            "gpuLayers": "all",
            "flashAttention": "auto",
            "cpuStrict": false,
            "threadpoolPoll": 50
        })
    }

    #[test]
    fn sampler_projection_matches_the_pinned_top_k_vector() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "candidates": [
                {"id": 0, "logit": -std::f64::consts::LN_10, "probability": 0.1},
                {"id": 1, "logit": -1.6094379124341003, "probability": 0.2},
                {"id": 2, "logit": -1.2039728043259361, "probability": 0.3},
                {"id": 3, "logit": -0.916290731874155, "probability": 0.4}
            ],
            "samplers": [{"type": "top-k", "k": 3}],
            "selection": {"type": "distribution", "seed": 0}
        }))
        .unwrap();

        let output = execute("sampler.apply", &input).unwrap();

        assert_eq!(output["selectedIndex"], 1);
        assert_eq!(output["selectedToken"], 2);
        assert_eq!(output["candidates"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn sampler_bench_projects_the_deterministic_top_k_summary() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "candidate_count": 16,
            "candidate_generator": {
                "kind": "seeded-uniform-logits",
                "seed": 0,
                "minimum": -1.0,
                "maximum": 1.0
            },
            "sampler": {"type": "top-k", "k": 4},
            "warmup_iterations": 2,
            "iterations": 3,
            "engineConfiguration": test_engine_configuration()
        }))
        .unwrap();

        let evidence = execute("sampler.bench", &input).unwrap();

        assert_eq!(evidence["schemaVersion"], 1);
        assert_eq!(evidence["output"]["resultCandidateCount"], 4);
        assert_eq!(evidence["output"]["sorted"], true);
        assert_eq!(evidence["output"]["resultTokenIds"], json!([13, 2, 10, 11]));
        assert_eq!(evidence["output"]["executedWarmupIterations"], 2);
        assert_eq!(evidence["output"]["executedMeasurementIterations"], 3);
        assert_eq!(
            evidence["output"]["semanticChecksum"],
            json!(8_254_323_595_071_633_661_u64)
        );
        assert_eq!(evidence["measurements"][0]["name"], "duration");
        assert_eq!(
            evidence["measurements"][0]["samples"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn chat_template_bench_matches_the_bounded_oracle_summary() {
        let fixture_root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../parity/fixtures/templates");
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "template_fixture": "fixtures/templates/chatml-basic.jinja",
            "input_fixture": "fixtures/templates/chatml-basic-input.json",
            "warmup_iterations": 2,
            "iterations": 3,
            "fixturePaths": {
                "chatml-basic-template": fixture_root.join("chatml-basic.jinja"),
                "chatml-basic-input": fixture_root.join("chatml-basic-input.json")
            },
            "engineConfiguration": test_engine_configuration()
        }))
        .unwrap();

        let evidence = execute("chat-template.bench", &input).unwrap();

        assert_eq!(evidence["output"]["format"], "peg-native");
        assert_eq!(evidence["output"]["promptBytes"], 206);
        assert_eq!(evidence["output"]["generationPromptBytes"], 22);
        assert_eq!(evidence["output"]["parserBytes"], 351);
        assert_eq!(evidence["output"]["spanCount"], 4);
        assert_eq!(evidence["output"]["executedWarmupIterations"], 2);
        assert_eq!(evidence["output"]["executedMeasurementIterations"], 3);
        assert_eq!(
            evidence["output"]["semanticChecksum"],
            json!(11_466_969_006_787_227_218_u64)
        );
        assert_eq!(
            evidence["output"]["generationPrompt"],
            "<|im_start|>assistant\n"
        );
        assert!(
            evidence["output"]["prompt"]
                .as_str()
                .unwrap()
                .contains("Say hello to 世界.")
        );
        assert_eq!(
            evidence["output"]["messageSpans"].as_array().unwrap().len(),
            4
        );
    }

    fn parser_probe_input() -> Map<String, Value> {
        let fixture_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../parity/fixtures");
        serde_json::from_value(json!({
            "template_fixture": "fixtures/templates/chatml-basic.jinja",
            "input_fixture": "fixtures/templates/chatml-basic-input.json",
            "content_fixture": "fixtures/parser/content-replay.json",
            "use_jinja": false,
            "force_pure_content": true,
            "parser_options": {
                "reasoning_in_content": false,
                "parse_tool_calls": false,
                "is_continuation": false,
                "echo": false,
                "debug": false
            },
            "fixturePaths": {
                "chatml-basic-template": fixture_root.join("templates/chatml-basic.jinja"),
                "chatml-basic-input": fixture_root.join("templates/chatml-basic-input.json"),
                "content-replay": fixture_root.join("parser/content-replay.json")
            },
            "engineConfiguration": test_engine_configuration()
        }))
        .unwrap()
    }

    #[test]
    fn chat_parser_inspect_matches_every_declared_chunk_partition() {
        let evidence = execute("chat-parser.inspect", &parser_probe_input()).unwrap();

        assert_eq!(evidence["prepared"]["format"], "content-only");
        assert_eq!(evidence["snapshot"]["role"], "assistant");
        assert_eq!(
            evidence["snapshot"]["content"],
            "Plain text, then Unicode: 世界 — café — 🌍."
        );
        assert_eq!(evidence["partitions"].as_array().unwrap().len(), 3);
        assert_eq!(
            evidence["partitions"][1]["pushes"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
        assert_eq!(evidence["partitions"][1]["pushes"][2]["chunkByteLength"], 6);
        assert_eq!(evidence["allFinalsMatch"], true);
    }

    #[test]
    fn chat_parser_bench_emits_one_versioned_duration_sample() {
        let mut input = parser_probe_input();
        input.insert("warmup_iterations".to_owned(), json!(1));
        input.insert("iterations".to_owned(), json!(1));

        let evidence = execute("chat-parser.bench", &input).unwrap();

        assert_eq!(evidence["schemaVersion"], 1);
        assert_eq!(evidence["output"]["inputByteLength"], 52);
        assert_eq!(evidence["output"]["message"]["role"], "assistant");
        assert_eq!(evidence["measurements"][0]["name"], "duration");
        assert_eq!(
            evidence["measurements"][0]["samples"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(evidence.get("work").is_none());
    }

    #[test]
    fn protocol_advertises_only_implemented_operations() {
        let output = execute("protocol.describe", &Map::new()).unwrap();
        assert_eq!(
            output["operations"].as_array().unwrap().len(),
            OPERATIONS.len()
        );
        for operation in [
            "tokenizer.token-to-piece",
            "reasoning-budget.inspect",
            "decode.abort",
            "decode.abort-recovery",
        ] {
            assert!(
                output["operations"]
                    .as_array()
                    .unwrap()
                    .contains(&json!(operation))
            );
        }
        assert!(execute("not.an.operation", &Map::new()).is_err());
    }

    #[test]
    fn grammar_uses_the_production_non_forced_contract() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "boolean"}},
                "required": ["answer"],
                "additionalProperties": false
            },
            "forceGbnf": false
        }))
        .unwrap();

        let output = execute("grammar.json-schema-to-grammar", &input).unwrap();
        assert_eq!(output["forceGbnf"], false);
        assert!(!output["grammar"].as_str().unwrap().is_empty());
    }

    #[test]
    fn grammar_rejects_the_unavailable_forced_mode() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "schema": {"type": "boolean"},
            "forceGbnf": true
        }))
        .unwrap();

        let error = execute("grammar.json-schema-to-grammar", &input).unwrap_err();
        assert_eq!(error.code(), "unsupported-force-gbnf");
    }

    #[test]
    fn tokenizer_rejects_semantics_the_safe_api_cannot_express() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "text": "hello",
            "parseSpecial": false
        }))
        .unwrap();

        let error = execute("tokenizer.tokenize", &input).unwrap_err();
        assert_eq!(error.code(), "unsupported-parse-special");
    }

    #[test]
    fn token_to_piece_requires_the_production_special_and_lstrip_boundary() {
        let special_input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "tokens": [1],
            "special": false,
            "lstrip": null
        }))
        .unwrap();
        let error = execute("tokenizer.token-to-piece", &special_input).unwrap_err();
        assert_eq!(error.code(), "unsupported-special-mode");

        let lstrip_input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "tokens": [1],
            "special": true,
            "lstrip": 1
        }))
        .unwrap();
        let error = execute("tokenizer.token-to-piece", &lstrip_input).unwrap_err();
        assert_eq!(error.code(), "unsupported-lstrip");
    }

    #[test]
    fn reasoning_budget_requires_a_positive_controllable_budget_before_model_loading() {
        let zero_budget = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "budgetTokens": 0,
            "startTag": "<think>",
            "endTag": "</think>",
            "forcedMessage": "",
            "controllable": true
        }))
        .unwrap();
        let error = execute("reasoning-budget.inspect", &zero_budget).unwrap_err();
        assert_eq!(error.code(), "invalid-reasoning-budget");

        let uncontrollable = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "budgetTokens": 1,
            "startTag": "<think>",
            "endTag": "</think>",
            "forcedMessage": "",
            "controllable": false
        }))
        .unwrap();
        let error = execute("reasoning-budget.inspect", &uncontrollable).unwrap_err();
        assert_eq!(error.code(), "invalid-reasoning-budget");
    }

    #[test]
    fn abort_requires_an_explicit_cpu_context_before_model_loading() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "nGpuLayers": 1,
            "useMmap": true,
            "useMlock": false,
            "contextTokens": 32,
            "threads": 1,
            "batchThreads": 1,
            "offloadKqv": false,
            "operationOffload": false,
            "flashAttention": "off",
            "token": 1
        }))
        .unwrap();

        let error = execute("decode.abort", &input).unwrap_err();
        assert_eq!(error.code(), "invalid-model-configuration");
    }

    #[test]
    fn abort_recovery_requires_observed_logit_indices_before_model_loading() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "nGpuLayers": 0,
            "useMmap": true,
            "useMlock": false,
            "contextTokens": 32,
            "threads": 1,
            "batchThreads": 1,
            "offloadKqv": false,
            "operationOffload": false,
            "flashAttention": "off",
            "token": 1,
            "logitIndices": []
        }))
        .unwrap();

        let error = execute("decode.abort-recovery", &input).unwrap_err();
        assert_eq!(error.code(), "invalid-logit-indices");
    }

    #[test]
    fn decode_validates_positions_before_loading_a_model() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "model_id": "fixture-model",
            "modelId": "fixture-model",
            "engineConfiguration": test_engine_configuration(),
            "nGpuLayers": 0,
            "useMmap": true,
            "useMlock": false,
            "offloadKqv": false,
            "operationOffload": false,
            "flashAttention": "off",
            "threads": 1,
            "batchThreads": 1,
            "context_tokens": 32,
            "batch": [{
                "token": 1,
                "position": 32,
                "sequence_ids": [0],
                "request_logits": true
            }],
            "logit_indices": [0]
        }))
        .unwrap();

        let error = execute("decode.execute-plan", &input).unwrap_err();
        assert_eq!(error.code(), "invalid-position");
    }

    #[test]
    fn decode_rejects_hidden_context_offload_before_loading_a_model() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "model_id": "fixture-model",
            "modelId": "fixture-model",
            "engineConfiguration": test_engine_configuration(),
            "nGpuLayers": 0,
            "useMmap": true,
            "useMlock": false,
            "offloadKqv": true,
            "operationOffload": false,
            "flashAttention": "off",
            "threads": 1,
            "batchThreads": 1,
            "context_tokens": 32,
            "batch": [{
                "token": 1,
                "position": 0,
                "sequence_ids": [0],
                "request_logits": true
            }],
            "logit_indices": [0]
        }))
        .unwrap();

        let error = execute("decode.execute-plan", &input).unwrap_err();
        assert_eq!(error.code(), "invalid-context-configuration");
    }

    #[test]
    fn state_rejects_empty_removal_ranges_before_model_loading() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "model_id": "fixture-model",
            "modelId": "fixture-model",
            "engineConfiguration": test_engine_configuration(),
            "nGpuLayers": 0,
            "useMmap": true,
            "useMlock": false,
            "offloadKqv": false,
            "operationOffload": false,
            "flashAttention": "off",
            "threads": 1,
            "batchThreads": 1,
            "prepare_tokens": [1],
            "sequence_id": 0,
            "operations": [{
                "type": "remove",
                "position_start": 1,
                "position_end": 1
            }]
        }))
        .unwrap();

        let error = execute("state.execute-script", &input).unwrap_err();
        assert_eq!(error.code(), "invalid-position-range");
    }

    #[test]
    fn configuration_uses_the_strict_runner_envelope_and_rejects_embeddings() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "model_id": "fixture-model",
            "modelId": "fixture-model",
            "engineConfiguration": test_engine_configuration(),
            "context": {
                "context_tokens": 32,
                "batch_tokens": 16,
                "micro_batch_tokens": 16,
                "sequences": 1,
                "threads": 1,
                "batch_threads": 1,
                "embeddings": true,
                "offload_kqv": true
            }
        }))
        .unwrap();

        let error = execute("configuration.inspect", &input).unwrap_err();
        assert_eq!(error.code(), "unsupported-embeddings");
    }

    #[test]
    fn llama_bench_accepts_the_strict_runner_envelope() {
        let input = serde_json::from_value::<Map<String, Value>>(json!({
            "modelPath": "/does/not/exist.gguf",
            "model_id": "fixture-model",
            "modelId": "fixture-model",
            "prompt_tokens": 16,
            "generation_tokens": 0,
            "context_depth": 0,
            "batch_tokens": 16,
            "micro_batch_tokens": 16,
            "repetitions": 1,
            "warmup": true,
            "token_schedule": "pinned-llama-bench-c-rand-default",
            "effective_engine_configuration": "profile",
            "engineConfiguration": test_engine_configuration()
        }))
        .unwrap();
        let parsed: LlamaBenchInput = decode_input(&input).unwrap();

        validate_llama_bench_input("llama-bench.prompt-processing", &parsed).unwrap();
    }
}
