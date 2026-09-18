//! Device-free tokenizer and template interpretation. Reads are bounded and do not
//! import weights, start workers, or fetch missing files.
use super::{BpeConfig, PieceKind, TokenId};
use crate::{
    chat::{TemplateBundle, TemplateVariant},
    weights::gguf::{Directory, Scalar, Value as GgufValue},
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

const METADATA_LIMIT: u64 = 16 * 1024 * 1024;
const TOKENIZER_LIMIT: u64 = 256 * 1024 * 1024;

fn read(path: &Path, limit: u64) -> Result<String, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if file.metadata().map_err(|e| e.to_string())?.len() > limit {
        return Err("input metadata exceeds size limit".into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("input metadata exceeds size limit".into());
    }
    String::from_utf8(bytes).map_err(|e| e.to_string())
}
fn variant(name: &str, value: &Value, path: &Path) -> Result<TemplateVariant, String> {
    Ok(TemplateVariant {
        name: name.into(),
        source: value.as_str().ok_or("template source must be text")?.into(),
        provenance: path.display().to_string(),
    })
}

/// Later sources override earlier ones per variant: processor config, tokenizer
/// config, named Jinja files, then the explicit default Jinja file.
pub fn directory_templates(path: &Path) -> Result<TemplateBundle, String> {
    let mut variants = BTreeMap::new();
    let mut tokens = BTreeMap::new();
    for filename in ["processor_config.json", "tokenizer_config.json"] {
        let source = path.join(filename);
        if !source.try_exists().map_err(|e| e.to_string())? {
            continue;
        }
        let config: Value =
            serde_json::from_str(&read(&source, METADATA_LIMIT)?).map_err(|e| e.to_string())?;
        let config = config
            .as_object()
            .ok_or("template configuration must be an object")?;
        if let Some(value) = config.get("chat_template").filter(|v| !v.is_null()) {
            let declared = match value {
                Value::String(_) => vec![variant("default", value, &source)?],
                Value::Object(map) => map
                    .iter()
                    .map(|(name, value)| variant(name, value, &source))
                    .collect::<Result<Vec<_>, _>>()?,
                Value::Array(list) => list
                    .iter()
                    .map(|entry| {
                        variant(
                            entry
                                .get("name")
                                .and_then(Value::as_str)
                                .ok_or("template variant needs a name")?,
                            entry
                                .get("template")
                                .ok_or("template variant needs a source")?,
                            &source,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()?,
                _ => return Err("unsupported artifact chat_template representation".into()),
            };
            let mut names = BTreeSet::new();
            for item in declared {
                if !names.insert(item.name.clone()) {
                    return Err("duplicate named templates in configuration".into());
                }
                variants.insert(item.name.clone(), item);
            }
        }
        for (name, value) in config {
            if !name.ends_with("_token") || value.is_boolean() || value.is_null() {
                continue;
            }
            let text = if value.is_object() {
                value.get("content")
            } else {
                Some(value)
            }
            .and_then(Value::as_str)
            .ok_or("invalid artifact special token")?;
            tokens.insert(name.clone(), text.to_owned());
        }
        if let Some(extra) = config.get("extra_special_tokens").filter(|v| !v.is_null()) {
            match extra {
                Value::Array(list) if list.iter().all(Value::is_string) => {}
                Value::Object(map) => {
                    for (name, value) in map {
                        tokens.insert(
                            name.clone(),
                            value.as_str().ok_or("special token must be text")?.into(),
                        );
                    }
                }
                _ => return Err("extra special tokens must be a list or named mapping".into()),
            }
        }
    }
    let named = path.join("chat_templates");
    if named.try_exists().map_err(|e| e.to_string())? {
        let mut files = std::fs::read_dir(&named)
            .map_err(|e| e.to_string())?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        files.sort();
        for file in files
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "jinja"))
        {
            let name = file
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("template filename must be UTF-8")?;
            variants.insert(
                name.into(),
                TemplateVariant {
                    name: name.into(),
                    source: read(&file, METADATA_LIMIT)?,
                    provenance: file.display().to_string(),
                },
            );
        }
    }
    let default = path.join("chat_template.jinja");
    if default.try_exists().map_err(|e| e.to_string())? {
        variants.insert(
            "default".into(),
            TemplateVariant {
                name: "default".into(),
                source: read(&default, METADATA_LIMIT)?,
                provenance: default.display().to_string(),
            },
        );
    }
    TemplateBundle::new(variants.into_values().collect(), "default".into(), tokens)
}

fn strings(directory: &Directory, key: &str) -> Result<Vec<String>, String> {
    match directory.value(key) {
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
pub fn gguf_templates(directory: &Directory, provenance: &str) -> Result<TemplateBundle, String> {
    let mut variants = BTreeMap::new();
    for item in &directory.metadata {
        let name = if item.name == "tokenizer.chat_template" {
            "default"
        } else if let Some(name) = item.name.strip_prefix("tokenizer.chat_template.") {
            name
        } else {
            continue;
        };
        if name == "default"
            && item.name != "tokenizer.chat_template"
            && directory.value("tokenizer.chat_template").is_some()
        {
            continue;
        }
        variants.insert(
            name.to_string(),
            TemplateVariant {
                name: name.into(),
                source: item
                    .value
                    .string()
                    .ok_or("GGUF template must be text")?
                    .into(),
                provenance: format!("{provenance}#{}", item.name),
            },
        );
    }
    let pieces = strings(directory, "tokenizer.ggml.tokens")?;
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
        if let Some(value) = directory.value(&format!("tokenizer.ggml.{native}_token_id")) {
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

pub fn qwen35_gguf(directory: &Directory, artifact_identity: String) -> Result<BpeConfig, String> {
    if directory
        .value("tokenizer.ggml.model")
        .and_then(GgufValue::string)
        != Some("gpt2")
        || directory
            .value("tokenizer.ggml.pre")
            .and_then(GgufValue::string)
            != Some("qwen35")
    {
        return Err("Qwen3.5 tokenization requires its declared byte BPE metadata".into());
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        if let Some(value) = directory.value(key) {
            if value != &GgufValue::Scalar(Scalar::Bool(false)) {
                return Err("implicit token insertion is unsupported".into());
            }
        }
    }
    let pieces = strings(directory, "tokenizer.ggml.tokens")?;
    let kinds = match directory.value("tokenizer.ggml.token_type") {
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
    let merges = strings(directory, "tokenizer.ggml.merges")?
        .into_iter()
        .map(|entry| {
            let parts: Vec<_> = entry.split(' ').collect();
            if parts.len() != 2 {
                return Err("BPE merge must contain two pieces".to_string());
            }
            Ok((parts[0].into(), parts[1].into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let eos = directory
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

pub fn mlx_tokenizer(path: &Path, artifact_identity: String) -> Result<BpeConfig, String> {
    let data: Value = serde_json::from_str(&read(&path.join("tokenizer.json"), TOKENIZER_LIMIT)?)
        .map_err(|e| e.to_string())?;
    let model = &data["model"];
    for key in ["dropout", "continuing_subword_prefix", "end_of_word_suffix"] {
        if model.get(key).is_some_and(|v| !v.is_null()) {
            return Err(format!("unsupported BPE model option: {key}"));
        }
    }
    for key in ["ignore_merges", "fuse_unk"] {
        if model.get(key).is_some_and(|v| v != false) {
            return Err(format!("unsupported BPE model option: {key}"));
        }
    }
    if model["type"] != "BPE"
        || model.get("byte_fallback").is_some_and(|v| v != false)
        || model.get("unk_token").is_some_and(|v| !v.is_null())
    {
        return Err("MLX Qwen tokenizer requires byte BPE without unknown replacement".into());
    }
    let pre = data["pre_tokenizer"]["pretokenizers"]
        .as_array()
        .ok_or("missing pretokenizers")?;
    if pre.len() != 2
        || data["normalizer"] != serde_json::json!({"type":"NFC"})
        || pre[0]["type"] != "Split"
        || pre[0]["behavior"] != "Isolated"
        || pre[0]["invert"] != false
        || pre[1]
            != serde_json::json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
    {
        return Err("unsupported converted Qwen pre-tokenization".into());
    }
    let vocab = model["vocab"].as_object().ok_or("missing BPE vocabulary")?;
    let mut by_id = BTreeMap::new();
    for (piece, id) in vocab {
        let id = id
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("invalid vocabulary ID")?;
        if by_id
            .insert(id, (piece.clone(), PieceKind::Normal))
            .is_some()
        {
            return Err("duplicate vocabulary ID".into());
        }
    }
    let mut added_ids = BTreeSet::new();
    for token in data["added_tokens"]
        .as_array()
        .ok_or("missing added-token declarations")?
    {
        if ["single_word", "lstrip", "rstrip", "normalized"]
            .iter()
            .any(|k| token[*k] != false)
        {
            return Err("unsupported added-token matching policy".into());
        }
        let id = token["id"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("invalid added-token ID")?;
        if !added_ids.insert(id) {
            return Err("duplicate added-token ID".into());
        }
        let piece = token["content"]
            .as_str()
            .ok_or("invalid added-token content")?
            .to_string();
        if by_id
            .get(&id)
            .is_some_and(|(existing, _)| existing != &piece)
        {
            return Err("added token changes vocabulary identity".into());
        }
        let kind = if token["special"]
            .as_bool()
            .ok_or("invalid added-token special flag")?
        {
            PieceKind::Control
        } else {
            PieceKind::UserDefined
        };
        by_id.insert(id, (piece, kind));
    }
    if by_id.is_empty() || by_id.keys().enumerate().any(|(i, id)| i != *id as usize) {
        return Err("tokenizer IDs must be contiguous".into());
    }
    let (pieces, kinds): (Vec<_>, Vec<_>) = by_id.into_values().unzip();
    let merges: Vec<(String, String)> = serde_json::from_value(model["merges"].clone())
        .map_err(|e| format!("invalid merge pairs: {e}"))?;
    let stop_tokens = pieces
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.as_str(), "<|endoftext|>" | "<|im_end|>"))
        .map(|(i, _)| TokenId(i as u32))
        .collect();
    Ok(BpeConfig {
        artifact_identity,
        pieces,
        kinds,
        merges,
        pattern: pre[0]["pattern"]["Regex"]
            .as_str()
            .ok_or("missing split regex")?
            .into(),
        normalize_nfc: true,
        stop_tokens,
    })
}
