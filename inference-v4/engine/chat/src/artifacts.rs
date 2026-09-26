//! Device-free adaptation of package-owned tokenizer and template payloads.
//! This layer never reparses the package's raw GGUF directory.
use super::{BpeConfig, PieceKind, TemplateBundle, TemplateVariant, TokenId};
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
    let pieces = strings(tokenizer, "tokenizer.ggml.tokens")?;
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
            tokens.insert(
                name.into(),
                pieces
                    .get(id)
                    .ok_or("special-token ID outside vocabulary")?
                    .clone(),
            );
        }
    }
    TemplateBundle::new(variants.into_values().collect(), "default".into(), tokens)
}

/// Select and adapt a supported byte-BPE profile entirely from the package's
/// literal tokenizer payload. Profile names are container facts, not model
/// family ownership.
pub fn gguf_byte_bpe(
    payload: &TokenizerPayload,
    artifact_identity: String,
) -> Result<BpeConfig, String> {
    if payload
        .value("tokenizer.ggml.model")
        .and_then(GgufValue::string)
        != Some("gpt2")
        || payload
            .value("tokenizer.ggml.pre")
            .and_then(GgufValue::string)
            != Some("qwen35")
    {
        return Err("unsupported GGUF byte-BPE tokenizer profile".into());
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        if let Some(value) = payload.value(key) {
            if value != &GgufValue::Scalar(Scalar::Bool(false)) {
                return Err("implicit token insertion is unsupported".into());
            }
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
    let merges = strings(payload, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|entry| {
            let parts: Vec<_> = entry.split(' ').collect();
            if parts.len() != 2 {
                return Err("BPE merge must contain two pieces".to_string());
            }
            Ok((parts[0].into(), parts[1].into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let eos = payload
        .value("tokenizer.ggml.eos_token_id")
        .and_then(GgufValue::unsigned)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("missing or invalid EOS identity")?;
    let mut stop_tokens = BTreeSet::from([TokenId(eos)]);
    stop_tokens.extend(
        pieces
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.as_str(), "<|endoftext|>" | "<|im_end|>"))
            .map(|(i, _)| TokenId(i as u32)),
    );
    Ok(BpeConfig { artifact_identity,pieces,kinds,merges,
        pattern:r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".into(),
        normalize_nfc:true,stop_tokens })
}
