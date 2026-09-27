//! Device-free adaptation of package-owned tokenizer and template payloads.
//! This layer never reparses the package's raw GGUF directory.
use super::{
    BpeConfig, Normalization, PieceEncoding, PieceKind, Split, SplitBehavior, TemplateBundle,
    TemplateVariant, TokenId, TokenizerError,
};
use magnitude_artifacts::{
    gguf::{Scalar, Value as GgufValue},
    TemplatePayload, TokenizerPayload,
};
use std::collections::{BTreeMap, BTreeSet};

fn strings(payload: &TokenizerPayload, key: &str) -> Result<Vec<String>, String> {
    match payload.value(key) {
        Some(GgufValue::Array(values)) => values
            .iter()
            .map(|v| match v {
                Scalar::String(s) => Ok(s.clone()),
                _ => Err(format!("{key} requires strings")),
            })
            .collect(),
        _ => Err(format!("missing string array {key}")),
    }
}

/// Adapt literal package template sources and tokenizer special-token metadata
/// into the generic rendering bundle.
pub fn gguf_templates(
    templates: &TemplatePayload,
    tokenizer: &TokenizerPayload,
) -> Result<TemplateBundle, String> {
    let mut variants = BTreeMap::new();
    for item in &templates.sources {
        variants.insert(
            item.name.clone(),
            TemplateVariant {
                name: item.name.clone(),
                source: item.source.clone(),
                provenance: item.provenance.clone(),
            },
        );
    }
    let pieces = match tokenizer.value("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(pieces)) => pieces,
        _ => return Err("missing string array tokenizer.ggml.tokens".into()),
    };
    if !pieces
        .iter()
        .all(|piece| matches!(piece, Scalar::String(_)))
    {
        return Err("tokenizer.ggml.tokens requires strings".into());
    }
    let mut tokens = BTreeMap::new();
    for (native, name) in [
        ("bos", "bos_token"),
        ("eos", "eos_token"),
        ("unknown", "unk_token"),
        ("padding", "pad_token"),
        ("separator", "sep_token"),
        ("cls", "cls_token"),
        ("mask", "mask_token"),
    ] {
        if let Some(value) = tokenizer.value(&format!("tokenizer.ggml.{native}_token_id")) {
            let id = value
                .unsigned()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("invalid special-token ID")?;
            let piece = match pieces.get(id) {
                Some(Scalar::String(piece)) => piece,
                Some(_) => return Err("tokenizer.ggml.tokens requires strings".into()),
                None => return Err("special-token ID outside vocabulary".into()),
            };
            tokens.insert(name.into(), piece.clone());
        }
    }
    TemplateBundle::new(variants.into_values().collect(), "default".into(), tokens)
}

/// Whether a container's tokenizer begins every sequence with its BOS.
#[derive(Clone, Copy)]
enum SequenceStart {
    /// `tokenizer.ggml.add_bos_token` decides; the profile's default applies
    /// when the key is absent.
    Declared { default: bool },
    /// The BOS is inserted whatever the container declares.
    Always,
}

/// One tokenizer scheme as the container names it. Each reproduces the
/// normalizer, pre-tokenizer and BPE options of the scheme's reference
/// `tokenizer.json`; absent-key defaults follow llama.cpp (18443257a30c,
/// `llama-vocab.cpp`), the GGUF reference reader.
struct Profile {
    normalization: Normalization,
    splits: &'static [(&'static str, SplitBehavior)],
    encoding: PieceEncoding,
    ignore_merges: bool,
    sequence_start: SequenceStart,
}

/// Contraction, letter, number, punctuation and whitespace pieces with
/// numbers split into groups of one to three digits (Llama 3).
const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

impl Profile {
    /// The profile the container's `tokenizer.ggml.model` and
    /// `tokenizer.ggml.pre` name. Profiles are container facts, never model
    /// family ownership.
    fn select(model: &str, pre: Option<&str>) -> Result<Self, TokenizerError> {
        use SplitBehavior::{Isolated, MergedWithNext};
        let byte_level = |splits, ignore_merges, add_bos| Self {
            normalization: Normalization::None,
            splits,
            encoding: PieceEncoding::ByteLevel,
            ignore_merges,
            sequence_start: SequenceStart::Declared { default: add_bos },
        };
        Ok(match (model, pre) {
            ("gpt2", Some("qwen35")) => Self {
                normalization: Normalization::Nfc,
                ..byte_level(
                    &[(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                        Isolated,
                    )],
                    false,
                    false,
                )
            },
            // llama.cpp's Llama 3 pre-tokenizer: its pieces, merges ignored for
            // a piece in the vocabulary, the BOS by default.
            ("gpt2", Some("lfm2" | "llama-bpe")) => byte_level(&[(LLAMA3, Isolated)], true, true),
            // Digit groups of one to three first, then Llama 3 pieces over
            // whole digit runs.
            ("gpt2", Some("minicpm5")) => byte_level(
                &[
                    (r"\p{N}{1,3}", Isolated),
                    (
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                        Isolated,
                    ),
                ],
                false,
                false,
            ),
            // GPT-4o pieces: case-split letter runs with trailing contractions.
            ("gpt2", Some("llama4")) => byte_level(
                &[(
                    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                    Isolated,
                )],
                true,
                false,
            ),
            // Tekken pieces: case-split letter runs and single digits.
            // llama.cpp reads this name with its Llama 3 expression instead.
            ("gpt2", Some("pixtral")) => byte_level(
                &[(
                    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                    Isolated,
                )],
                true,
                true,
            ),
            // SentencePiece-style BPE: spaces become U+2581 and merges run over
            // the whole normalized text. llama.cpp inserts the BOS even when
            // the container declares none.
            ("gemma4", None) => Self {
                normalization: Normalization::Metaspace,
                splits: &[],
                encoding: PieceEncoding::MetaspaceByteFallback,
                ignore_merges: false,
                sequence_start: SequenceStart::Always,
            },
            _ => {
                return Err(TokenizerError::UnsupportedProfile {
                    model: model.into(),
                    pre: pre.map(Into::into),
                })
            }
        })
    }
}

/// Tokenizer metadata the adaptation interprets. Any other key could change
/// encoding semantics, so it is rejected.
const INTERPRETED_KEYS: &[&str] = &[
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    // Piece scores rank SentencePiece unigram merges; BPE ranks by merges.
    "tokenizer.ggml.scores",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.seperator_token_id",
    "tokenizer.ggml.cls_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    // Separators join sequence pairs for ranking; generation encodes one
    // sequence.
    "tokenizer.ggml.add_sep_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.suppress_tokens",
];

/// Pieces that end a turn by the GGUF convention (llama.cpp 18443257a30c,
/// `llama-vocab.cpp` end-of-generation detection), besides the declared EOS,
/// EOT and EOM identities.
const TURN_ENDS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
    "[e~[",
];

/// The end-of-generation set: the declared identities plus the vocabulary's
/// turn-ending pieces. `<|end|>` only separates messages when the vocabulary
/// has harmony's `<|return|>` and `<|call|>` (or `<|calls|>` and
/// `<|flush|>`), and `</s>` is ordinary text when `<|tool_response>` ends
/// generation.
fn end_of_generation(
    pieces: &[String],
    declared: impl IntoIterator<Item = TokenId>,
) -> BTreeSet<TokenId> {
    let find = |text: &str| pieces.iter().position(|piece| piece == text);
    let mut stops: BTreeSet<TokenId> = declared.into_iter().collect();
    stops.extend(
        pieces
            .iter()
            .enumerate()
            .filter(|(_, piece)| TURN_ENDS.contains(&piece.as_str()))
            .map(|(id, _)| TokenId(id as u32)),
    );
    let stopping = |text: &str| find(text).is_some_and(|id| stops.contains(&TokenId(id as u32)));
    let mut ordinary = Vec::new();
    if stopping("<|end|>")
        && ((stopping("<|return|>") && stopping("<|call|>"))
            || (stopping("<|calls|>") && stopping("<|flush|>")))
    {
        ordinary.push("<|end|>");
    }
    if stopping("<|tool_response>") {
        ordinary.push("</s>");
    }
    for text in ordinary {
        if let Some(id) = find(text) {
            stops.remove(&TokenId(id as u32));
        }
    }
    stops
}

fn flag(payload: &TokenizerPayload, key: &str) -> Result<Option<bool>, TokenizerError> {
    match payload.value(key) {
        None => Ok(None),
        Some(GgufValue::Scalar(Scalar::Bool(value))) => Ok(Some(*value)),
        Some(_) => Err(TokenizerError::Invalid(format!("{key} requires a boolean"))),
    }
}

/// Select and adapt a supported BPE profile entirely from the package's
/// literal tokenizer payload.
pub fn gguf_byte_bpe(
    payload: &TokenizerPayload,
    artifact_identity: String,
) -> Result<BpeConfig, TokenizerError> {
    if let Some(entry) = payload
        .metadata
        .iter()
        .find(|entry| !INTERPRETED_KEYS.contains(&entry.name.as_str()))
    {
        return Err(TokenizerError::UnsupportedMetadata {
            key: entry.name.clone(),
        });
    }
    let text = |key: &str| match payload.value(key) {
        None => Ok(None),
        Some(value) => value
            .string()
            .map(Some)
            .ok_or_else(|| TokenizerError::Invalid(format!("{key} requires text"))),
    };
    let profile = Profile::select(
        text("tokenizer.ggml.model")?.ok_or("missing tokenizer.ggml.model")?,
        text("tokenizer.ggml.pre")?,
    )?;
    for key in [
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.add_space_prefix",
    ] {
        if flag(payload, key)? == Some(true) {
            return Err(TokenizerError::UnsupportedMetadata { key: key.into() });
        }
    }
    let pieces = strings(payload, "tokenizer.ggml.tokens")?;
    let kinds = match payload.value("tokenizer.ggml.token_type") {
        Some(GgufValue::Array(values)) => values
            .iter()
            .map(|v| {
                let kind = match v {
                    Scalar::Unsigned(n) => *n,
                    Scalar::Signed(n) => u64::try_from(*n).map_err(|_| "invalid piece kind")?,
                    _ => return Err("invalid piece kind"),
                };
                Ok(match kind {
                    1 => PieceKind::Normal,
                    2 => PieceKind::Unknown,
                    3 => PieceKind::Control,
                    4 => PieceKind::UserDefined,
                    5 => PieceKind::Unused,
                    6 => PieceKind::Byte,
                    _ => return Err("invalid piece kind"),
                })
            })
            .collect::<Result<Vec<_>, &str>>()?,
        _ => return Err("missing GGUF token kinds".into()),
    };
    // A merge names two pieces separated by the first space after its first
    // character, so the left piece may itself be a space.
    let merges = strings(payload, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|entry| {
            let split = entry
                .char_indices()
                .skip(1)
                .find(|(_, c)| *c == ' ')
                .map(|(index, _)| index)
                .ok_or("BPE merge must contain two pieces")?;
            Ok((entry[..split].into(), entry[split + 1..].into()))
        })
        .collect::<Result<Vec<_>, TokenizerError>>()?;
    let token = |key: &str| match payload.value(key) {
        None => Ok(None),
        Some(value) => value
            .unsigned()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| (n as usize) < pieces.len())
            .map(|n| Some(TokenId(n)))
            .ok_or_else(|| TokenizerError::Invalid(format!("{key} is not a vocabulary token"))),
    };
    let eos = token("tokenizer.ggml.eos_token_id")?.ok_or("missing EOS identity")?;
    let stop_tokens = end_of_generation(
        &pieces,
        [
            Some(eos),
            token("tokenizer.ggml.eot_token_id")?,
            token("tokenizer.ggml.eom_token_id")?,
        ]
        .into_iter()
        .flatten(),
    );
    let add_bos = match profile.sequence_start {
        SequenceStart::Declared { default } => {
            flag(payload, "tokenizer.ggml.add_bos_token")?.unwrap_or(default)
        }
        SequenceStart::Always => true,
    };
    let implicit_bos = if add_bos {
        Some(token("tokenizer.ggml.bos_token_id")?.ok_or("implicit BOS requires a BOS identity")?)
    } else {
        None
    };
    let suppressed_tokens = match payload.value("tokenizer.ggml.suppress_tokens") {
        None => BTreeSet::new(),
        Some(GgufValue::Array(values)) => values
            .iter()
            .map(|value| {
                match value {
                    Scalar::Signed(n) => u32::try_from(*n).ok(),
                    Scalar::Unsigned(n) => u32::try_from(*n).ok(),
                    _ => None,
                }
                .filter(|&n| (n as usize) < pieces.len())
                .map(TokenId)
                .ok_or("tokenizer.ggml.suppress_tokens requires vocabulary tokens")
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("tokenizer.ggml.suppress_tokens requires an array".into()),
    };
    Ok(BpeConfig {
        artifact_identity,
        pieces,
        kinds,
        merges,
        normalization: profile.normalization,
        splits: profile
            .splits
            .iter()
            .map(|&(pattern, behavior)| Split {
                pattern: pattern.into(),
                behavior,
            })
            .collect(),
        encoding: profile.encoding,
        ignore_merges: profile.ignore_merges,
        implicit_bos,
        stop_tokens,
        suppressed_tokens,
    })
}
