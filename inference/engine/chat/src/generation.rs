//! Numerical generation options for a prepared chat request, bounded by the
//! loaded model.
use super::{ByteBpeTokenizer, ChatError, GenerationControls, PreparedChat};
use magnitude_generation::{MethodChoice, Options, Sampling, Shaping};

/// Host-selected generation method policy. The concrete proposal width is
/// resolved per request because the qualified defaults differ for greedy and
/// sampled decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodPolicy {
    Plain,
    Mtp {
        greedy_proposals: u8,
        sampled_proposals: u8,
    },
    /// A separate draft (DFlash, DSpark) drafting `proposals` tokens per
    /// block, greedy and sampled alike.
    DFlash { proposals: u8 },
}

impl MethodPolicy {
    pub fn validate_method(self, identity: &str) -> Result<(), String> {
        match self {
            Self::Plain if identity == "plain" => Ok(()),
            Self::Plain => Err("plain policy requires the plain generation method".into()),
            Self::Mtp {
                greedy_proposals,
                sampled_proposals,
            } => {
                let capacity = identity
                    .strip_prefix("mtp:")
                    .and_then(|value| value.rsplit_once(':'))
                    .and_then(|(_, capacity)| capacity.parse::<u8>().ok())
                    .ok_or("MTP policy requires a prepared MTP generation method")?;
                if greedy_proposals == 0
                    || sampled_proposals == 0
                    || greedy_proposals > capacity
                    || sampled_proposals > capacity
                {
                    return Err("MTP proposal width exceeds the prepared head capacity".into());
                }
                Ok(())
            }
            Self::DFlash { proposals } => {
                let prepared = identity
                    .strip_prefix("dflash:")
                    .and_then(|value| value.rsplit_once(':'))
                    .and_then(|(_, proposals)| proposals.parse::<u8>().ok())
                    .ok_or("DFlash policy requires a prepared DFlash generation method")?;
                if proposals == 0 || proposals != prepared {
                    return Err("DFlash proposal width differs from the prepared draft's".into());
                }
                Ok(())
            }
        }
    }

    pub fn resolve(self, sampling: Sampling) -> MethodChoice {
        match self {
            Self::Plain => MethodChoice::Plain,
            Self::Mtp {
                greedy_proposals,
                sampled_proposals,
            } => MethodChoice::Mtp {
                proposals: match sampling {
                    Sampling::Greedy => greedy_proposals,
                    Sampling::Categorical => sampled_proposals,
                },
            },
            Self::DFlash { proposals } => MethodChoice::DFlash { proposals },
        }
    }
}

/// The loaded model's generation bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelLimits {
    /// Served context in tokens.
    pub context_tokens: usize,
    /// Output projection width (at least the tokenizer vocabulary).
    pub vocabulary: usize,
    /// Accepted-token publication bound per request.
    pub output_capacity: usize,
    pub forced_quantum: usize,
    pub method: MethodPolicy,
}

/// Options for generating from `chat` whose model input holds `prompt_tokens`
/// tokens (after media expansion). The prompt must leave at least one context
/// position for generation.
pub fn generation_options(
    chat: &PreparedChat,
    prompt_tokens: usize,
    controls: &GenerationControls,
    tokenizer: &ByteBpeTokenizer,
    limits: &ModelLimits,
) -> Result<Options, ChatError> {
    if prompt_tokens >= limits.context_tokens {
        return Err(ChatError::ContextLengthExceeded {
            prompt_tokens,
            context_tokens: limits.context_tokens,
        });
    }
    let invalid = |message: &str| ChatError::InvalidRequest(message.into());
    let sampling = &controls.sampling;
    let greedy = sampling.temperature == 0.0;
    let shaping = Shaping {
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        repetition_penalty: sampling.repetition_penalty,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
    }
    .validate()
    .map_err(invalid)?;
    let sampling = if greedy {
        Sampling::Greedy
    } else {
        Sampling::Categorical
    };
    let available = limits.context_tokens - prompt_tokens + 1;
    let reasoning_budget = controls
        .reasoning_budget
        .map(|tokens| chat.reasoning_budget(tokenizer, tokens))
        .transpose()?;
    Ok(Options {
        max_tokens: controls
            .max_output_tokens
            .map_or(available, |limit| (limit.get() as usize).min(available)),
        output_capacity: limits.output_capacity,
        context_limit: limits.context_tokens,
        vocabulary: limits.vocabulary,
        stop_tokens: tokenizer.stop_tokens().clone(),
        suppressed_tokens: tokenizer.suppressed_tokens().clone(),
        sampling,
        shaping,
        seed: controls.sampling.seed,
        forced_quantum: limits.forced_quantum,
        method: limits.method.resolve(sampling),
        end_of_generation: controls.end_of_generation,
        reasoning_budget,
    })
}
