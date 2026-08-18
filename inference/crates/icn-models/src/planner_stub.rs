use std::collections::BTreeSet;
use std::ops::Range;

const MAGIC: &[u8; 4] = b"GGUF";
const MIN_VERSION: u32 = 2;
const MAX_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_TENSORS: u64 = 10_000_000;
const MAX_ARRAY_ELEMENTS: u64 = 10_000_000;
const MAX_DIMS: u32 = 8;
const MAX_STRING_BYTES: u64 = 128 * 1024 * 1024;

const TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
const TOKENIZER_TOKENS: &str = "tokenizer.ggml.tokens";

const REMOVED_METADATA: &[&str] = &[
    TOKENIZER_MODEL,
    TOKENIZER_TOKENS,
    "tokenizer.ggml.merges",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.precompiled_charsmap",
    "tokenizer.huggingface.json",
    "tokenizer.rwkv.world",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannerStubContext {
    architecture: String,
    vocabulary_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerStubComponent {
    Primary,
    Shard,
    Companion,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlannerStubError {
    #[error("planner source is not GGUF")]
    InvalidMagic,
    #[error("planner source uses unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("planner source is structurally invalid: {0}")]
    Invalid(&'static str),
    #[error("planner source contains non-UTF-8 metadata")]
    Utf8,
}

#[derive(Debug)]
struct ParsedHeader {
    version: u32,
    tensor_count: u64,
    entries: Vec<MetadataEntry>,
    tensor_directory: Range<usize>,
    architecture: Option<String>,
    token_count: Option<u32>,
    declared_vocabulary_size: Option<u32>,
    alignment: u32,
}

#[derive(Debug)]
struct MetadataEntry {
    key: String,
    bytes: Range<usize>,
}

#[derive(Debug)]
enum ValueSummary {
    U32(u32),
    String(String),
    Array { element_type: u32, count: u64 },
    Other,
}

pub fn planner_stub_context(source: &[u8]) -> Result<PlannerStubContext, PlannerStubError> {
    let parsed = parse_header(source)?;
    let architecture = parsed.architecture.ok_or(PlannerStubError::Invalid(
        "primary GGUF has no architecture",
    ))?;
    let vocabulary_size = match (parsed.token_count, parsed.declared_vocabulary_size) {
        (Some(tokens), Some(declared)) if tokens != declared => {
            return Err(PlannerStubError::Invalid(
                "token count differs from declared vocabulary size",
            ));
        }
        (Some(tokens), _) => tokens,
        (None, Some(declared)) => declared,
        (None, None) => {
            return Err(PlannerStubError::Invalid(
                "primary GGUF has no vocabulary cardinality",
            ));
        }
    };
    Ok(PlannerStubContext {
        architecture,
        vocabulary_size,
    })
}

pub fn compact_planner_stub(
    source: &[u8],
    context: &PlannerStubContext,
    component: PlannerStubComponent,
) -> Result<Vec<u8>, PlannerStubError> {
    let parsed = parse_header(source)?;
    if component != PlannerStubComponent::Companion {
        if let Some(architecture) = &parsed.architecture
            && architecture != &context.architecture
        {
            return Err(PlannerStubError::Invalid(
                "split GGUF architecture differs from its primary",
            ));
        }
        if let Some(tokens) = parsed.token_count
            && tokens != context.vocabulary_size
        {
            return Err(PlannerStubError::Invalid(
                "split GGUF token count differs from its primary",
            ));
        }
        if let Some(declared) = parsed.declared_vocabulary_size
            && declared != context.vocabulary_size
        {
            return Err(PlannerStubError::Invalid(
                "split GGUF vocabulary size differs from its primary",
            ));
        }
    }

    if component == PlannerStubComponent::Primary && parsed.architecture.is_none() {
        return Err(PlannerStubError::Invalid(
            "primary GGUF has no architecture",
        ));
    }
    let vocabulary_key = format!("{}.vocab_size", context.architecture);
    let kept = parsed
        .entries
        .iter()
        .filter(|entry| {
            component != PlannerStubComponent::Primary
                || (!removed_metadata(&entry.key) && entry.key != vocabulary_key)
        })
        .collect::<Vec<_>>();
    let uses_synthetic_vocabulary = component == PlannerStubComponent::Primary;
    let added = if uses_synthetic_vocabulary { 3_u64 } else { 0 };
    let metadata_count = u64::try_from(kept.len())
        .map_err(|_| PlannerStubError::Invalid("metadata count overflows u64"))?
        .checked_add(added)
        .ok_or(PlannerStubError::Invalid("metadata count overflow"))?;

    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&parsed.version.to_le_bytes());
    output.extend_from_slice(&parsed.tensor_count.to_le_bytes());
    output.extend_from_slice(&metadata_count.to_le_bytes());
    for entry in kept {
        output.extend_from_slice(&source[entry.bytes.clone()]);
    }
    if component == PlannerStubComponent::Primary {
        encode_string_entry(&mut output, TOKENIZER_MODEL, "llama");
        if uses_synthetic_vocabulary {
            encode_empty_string_array_entry(&mut output, TOKENIZER_TOKENS, context.vocabulary_size);
        }
        encode_u32_entry(&mut output, &vocabulary_key, context.vocabulary_size);
    }
    output.extend_from_slice(&source[parsed.tensor_directory]);
    let aligned = output
        .len()
        .checked_next_multiple_of(parsed.alignment as usize)
        .ok_or(PlannerStubError::Invalid("planner stub alignment overflow"))?;
    output.resize(aligned, 0);
    Ok(output)
}

fn removed_metadata(key: &str) -> bool {
    REMOVED_METADATA.contains(&key)
        || key == "tokenizer.chat_template"
        || key.starts_with("tokenizer.chat_template.")
}

fn encode_string_entry(output: &mut Vec<u8>, key: &str, value: &str) {
    encode_string(output, key);
    output.extend_from_slice(&8_u32.to_le_bytes());
    encode_string(output, value);
}

fn encode_u32_entry(output: &mut Vec<u8>, key: &str, value: u32) {
    encode_string(output, key);
    output.extend_from_slice(&4_u32.to_le_bytes());
    output.extend_from_slice(&value.to_le_bytes());
}

fn encode_empty_string_array_entry(output: &mut Vec<u8>, key: &str, count: u32) {
    encode_string(output, key);
    output.extend_from_slice(&9_u32.to_le_bytes());
    output.extend_from_slice(&8_u32.to_le_bytes());
    output.extend_from_slice(&u64::from(count).to_le_bytes());
    for _ in 0..count {
        output.extend_from_slice(&0_u64.to_le_bytes());
    }
}

fn encode_string(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn parse_header(source: &[u8]) -> Result<ParsedHeader, PlannerStubError> {
    let mut reader = Reader::new(source);
    if reader.bytes(4)? != MAGIC {
        return Err(PlannerStubError::InvalidMagic);
    }
    let version = reader.u32()?;
    if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
        return Err(PlannerStubError::UnsupportedVersion(version));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count > MAX_TENSORS {
        return Err(PlannerStubError::Invalid("tensor count exceeds bound"));
    }
    if metadata_count > MAX_METADATA_ENTRIES {
        return Err(PlannerStubError::Invalid("metadata count exceeds bound"));
    }

    let mut entries = Vec::with_capacity(
        usize::try_from(metadata_count)
            .map_err(|_| PlannerStubError::Invalid("metadata count overflows usize"))?,
    );
    let mut keys = BTreeSet::new();
    let mut architecture = None;
    let mut token_count = None;
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..metadata_count {
        let start = reader.position;
        let key = reader.string()?.to_owned();
        if !keys.insert(key.clone()) {
            return Err(PlannerStubError::Invalid("duplicate metadata key"));
        }
        let value_type = reader.u32()?;
        let summary = reader.value(value_type)?;
        let end = reader.position;
        match key.as_str() {
            "general.architecture" => match summary {
                ValueSummary::String(value) if !value.is_empty() => architecture = Some(value),
                _ => {
                    return Err(PlannerStubError::Invalid(
                        "architecture metadata is not a non-empty string",
                    ));
                }
            },
            "general.alignment" => match summary {
                ValueSummary::U32(value) => alignment = value,
                _ => {
                    return Err(PlannerStubError::Invalid(
                        "alignment metadata is not uint32",
                    ));
                }
            },
            TOKENIZER_TOKENS => match summary {
                ValueSummary::Array {
                    element_type: 8,
                    count,
                } => {
                    token_count = Some(u32::try_from(count).map_err(|_| {
                        PlannerStubError::Invalid("vocabulary size exceeds uint32")
                    })?);
                }
                _ => {
                    return Err(PlannerStubError::Invalid(
                        "tokenizer tokens are not a string array",
                    ));
                }
            },
            _ => {}
        }
        entries.push(MetadataEntry {
            key,
            bytes: start..end,
        });
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(PlannerStubError::Invalid("alignment is not a power of two"));
    }

    let declared_vocabulary_size = architecture.as_ref().and_then(|architecture| {
        let key = format!("{architecture}.vocab_size");
        entries.iter().position(|entry| entry.key == key)
    });
    let declared_vocabulary_size = declared_vocabulary_size
        .map(|index| {
            let mut value = Reader::new(&source[entries[index].bytes.clone()]);
            let _ = value.string()?;
            if value.u32()? != 4 {
                return Err(PlannerStubError::Invalid(
                    "vocabulary size metadata is not uint32",
                ));
            }
            value.u32()
        })
        .transpose()?;
    let tensor_directory_start = reader.position;
    for _ in 0..tensor_count {
        let _ = reader.string()?;
        let dimensions = reader.u32()?;
        if dimensions == 0 || dimensions > MAX_DIMS {
            return Err(PlannerStubError::Invalid(
                "tensor dimension count is invalid",
            ));
        }
        reader.skip(
            u64::from(dimensions)
                .checked_mul(8)
                .ok_or(PlannerStubError::Invalid("tensor dimensions overflow"))?,
        )?;
        let _ = reader.u32()?;
        let _ = reader.u64()?;
    }
    let tensor_directory = tensor_directory_start..reader.position;
    let aligned = reader
        .position
        .checked_next_multiple_of(alignment as usize)
        .ok_or(PlannerStubError::Invalid("source alignment overflow"))?;
    if aligned != source.len() {
        return Err(PlannerStubError::Invalid(
            "source is not an exact aligned GGUF header",
        ));
    }

    Ok(ParsedHeader {
        version,
        tensor_count,
        entries,
        tensor_directory,
        architecture,
        token_count,
        declared_vocabulary_size,
        alignment,
    })
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], PlannerStubError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(PlannerStubError::Invalid("source offset overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(PlannerStubError::Invalid("source ended unexpectedly"))?;
        self.position = end;
        Ok(value)
    }

    fn skip(&mut self, count: u64) -> Result<(), PlannerStubError> {
        let count = usize::try_from(count)
            .map_err(|_| PlannerStubError::Invalid("skip length overflows usize"))?;
        let _ = self.bytes(count)?;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, PlannerStubError> {
        Ok(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, PlannerStubError> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().map_err(
            |_| PlannerStubError::Invalid("invalid uint16"),
        )?))
    }

    fn u32(&mut self) -> Result<u32, PlannerStubError> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().map_err(
            |_| PlannerStubError::Invalid("invalid uint32"),
        )?))
    }

    fn u64(&mut self) -> Result<u64, PlannerStubError> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().map_err(
            |_| PlannerStubError::Invalid("invalid uint64"),
        )?))
    }

    fn string(&mut self) -> Result<&'a str, PlannerStubError> {
        let length = self.u64()?;
        if length > MAX_STRING_BYTES {
            return Err(PlannerStubError::Invalid("string exceeds bound"));
        }
        let length = usize::try_from(length)
            .map_err(|_| PlannerStubError::Invalid("string length overflows usize"))?;
        std::str::from_utf8(self.bytes(length)?).map_err(|_| PlannerStubError::Utf8)
    }

    fn value(&mut self, value_type: u32) -> Result<ValueSummary, PlannerStubError> {
        match value_type {
            0 | 1 | 7 => {
                let _ = self.u8()?;
                Ok(ValueSummary::Other)
            }
            2 | 3 => {
                let _ = self.u16()?;
                Ok(ValueSummary::Other)
            }
            4 => Ok(ValueSummary::U32(self.u32()?)),
            5 | 6 => {
                let _ = self.u32()?;
                Ok(ValueSummary::Other)
            }
            8 => Ok(ValueSummary::String(self.string()?.to_owned())),
            9 => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                if count > MAX_ARRAY_ELEMENTS {
                    return Err(PlannerStubError::Invalid("array exceeds bound"));
                }
                if element_type == 9 {
                    return Err(PlannerStubError::Invalid("nested arrays are unsupported"));
                }
                for _ in 0..count {
                    let _ = self.value(element_type)?;
                }
                Ok(ValueSummary::Array {
                    element_type,
                    count,
                })
            }
            10 | 11 | 12 => {
                let _ = self.u64()?;
                Ok(ValueSummary::Other)
            }
            _ => Err(PlannerStubError::Invalid(
                "unknown GGUF metadata value type",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_string(output: &mut Vec<u8>, key: &str, value: &str) {
        encode_string_entry(output, key, value);
    }

    fn entry_u32(output: &mut Vec<u8>, key: &str, value: u32) {
        encode_u32_entry(output, key, value);
    }

    fn entry_string_array(output: &mut Vec<u8>, key: &str, values: &[&str]) {
        encode_string(output, key);
        output.extend_from_slice(&9_u32.to_le_bytes());
        output.extend_from_slice(&8_u32.to_le_bytes());
        output.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            encode_string(output, value);
        }
    }

    fn entry_i32_array(output: &mut Vec<u8>, key: &str, values: &[i32]) {
        encode_string(output, key);
        output.extend_from_slice(&9_u32.to_le_bytes());
        output.extend_from_slice(&5_u32.to_le_bytes());
        output.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn entry_f32_array(output: &mut Vec<u8>, key: &str, values: &[f32]) {
        encode_string(output, key);
        output.extend_from_slice(&9_u32.to_le_bytes());
        output.extend_from_slice(&6_u32.to_le_bytes());
        output.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn tensor(output: &mut Vec<u8>, name: &str, dimensions: &[u64], kind: u32, offset: u64) {
        encode_string(output, name);
        output.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        for dimension in dimensions {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        output.extend_from_slice(&kind.to_le_bytes());
        output.extend_from_slice(&offset.to_le_bytes());
    }

    fn header(
        metadata: Vec<u8>,
        metadata_count: u64,
        tensors: Vec<u8>,
        tensor_count: u64,
    ) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(&3_u32.to_le_bytes());
        output.extend_from_slice(&tensor_count.to_le_bytes());
        output.extend_from_slice(&metadata_count.to_le_bytes());
        output.extend_from_slice(&metadata);
        output.extend_from_slice(&tensors);
        output.resize(output.len().next_multiple_of(32), 0);
        output
    }

    fn primary_header() -> Vec<u8> {
        let mut metadata = Vec::new();
        entry_string(&mut metadata, "general.architecture", "llama");
        entry_u32(&mut metadata, "general.alignment", 32);
        entry_u32(&mut metadata, "llama.context_length", 4096);
        entry_string(&mut metadata, TOKENIZER_MODEL, "gpt2");
        entry_string_array(&mut metadata, TOKENIZER_TOKENS, &["one", "two", "three"]);
        entry_string_array(&mut metadata, "tokenizer.ggml.merges", &["o n", "t w"]);
        entry_f32_array(&mut metadata, "tokenizer.ggml.scores", &[1.0, 2.0, 3.0]);
        entry_i32_array(&mut metadata, "tokenizer.ggml.token_type", &[1, 1, 3]);
        entry_i32_array(&mut metadata, "tokenizer.ggml.suppress_tokens", &[1, 2]);
        entry_string(&mut metadata, "tokenizer.chat_template", "large template");
        entry_string(&mut metadata, "vendor.future_metadata", "preserved");
        let mut tensors = Vec::new();
        tensor(&mut tensors, "token_embd.weight", &[4, 3], 0, 0);
        header(metadata, 11, tensors, 1)
    }

    #[test]
    fn compact_stub_is_deterministic_and_preserves_planning_state() {
        let source = primary_header();
        let context = planner_stub_context(&source).unwrap();
        assert_eq!(context.architecture, "llama");
        assert_eq!(context.vocabulary_size, 3);

        let first = compact_planner_stub(&source, &context, PlannerStubComponent::Primary).unwrap();
        let second =
            compact_planner_stub(&source, &context, PlannerStubComponent::Primary).unwrap();
        assert_eq!(first, second);
        assert!(first.len() < source.len());

        let source_parsed = parse_header(&source).unwrap();
        let compact = parse_header(&first).unwrap();
        assert_eq!(compact.architecture.as_deref(), Some("llama"));
        assert_eq!(compact.declared_vocabulary_size, Some(3));
        assert_eq!(compact.token_count, Some(3));
        assert_eq!(
            &source[source_parsed.tensor_directory],
            &first[compact.tensor_directory]
        );
        let keys = compact
            .entries
            .iter()
            .map(|entry| entry.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(keys.contains("tokenizer.ggml.suppress_tokens"));
        assert!(keys.contains(TOKENIZER_MODEL));
        assert!(keys.contains("vendor.future_metadata"));
        assert!(keys.contains(TOKENIZER_TOKENS));
        assert!(!keys.contains("tokenizer.ggml.merges"));
        assert!(!keys.contains("tokenizer.ggml.scores"));
        assert!(!keys.contains("tokenizer.ggml.token_type"));
        assert!(!keys.contains("tokenizer.chat_template"));
    }

    #[test]
    fn every_primary_uses_a_placeholder_vocabulary() {
        let source = primary_header();
        let parsed = parse_header(&source).unwrap();
        let suppress = parsed
            .entries
            .iter()
            .find(|entry| entry.key == "tokenizer.ggml.suppress_tokens")
            .unwrap();
        let suppress_bytes = suppress.bytes.len();
        let unaligned_end = parsed.tensor_directory.end - suppress_bytes;
        let mut source_without_suppress = source;
        source_without_suppress.drain(suppress.bytes.clone());
        source_without_suppress[16..24].copy_from_slice(&10_u64.to_le_bytes());
        source_without_suppress
            .truncate(unaligned_end.next_multiple_of(DEFAULT_ALIGNMENT as usize));
        let context = planner_stub_context(&source_without_suppress).unwrap();
        let stub = compact_planner_stub(
            &source_without_suppress,
            &context,
            PlannerStubComponent::Primary,
        )
        .unwrap();
        assert_eq!(parse_header(&stub).unwrap().token_count, Some(3));
    }

    #[test]
    fn split_shard_keeps_its_metadata_without_primary_overrides() {
        let context = planner_stub_context(&primary_header()).unwrap();
        let mut metadata = Vec::new();
        entry_u32(&mut metadata, "split.no", 1);
        entry_u32(&mut metadata, "split.count", 2);
        let mut tensors = Vec::new();
        tensor(&mut tensors, "blk.0.weight", &[2, 2], 0, 64);
        let source = header(metadata, 2, tensors, 1);

        let stub = compact_planner_stub(&source, &context, PlannerStubComponent::Shard).unwrap();
        let parsed = parse_header(&stub).unwrap();
        assert_eq!(parsed.architecture, None);
        assert_eq!(parsed.entries.len(), 2);
        assert!(
            !parsed
                .entries
                .iter()
                .any(|entry| entry.key == TOKENIZER_MODEL)
        );
        assert_eq!(parsed.declared_vocabulary_size, None);
    }

    #[test]
    fn companion_keeps_independent_architecture_and_vocabulary_metadata() {
        let context = planner_stub_context(&primary_header()).unwrap();
        let mut metadata = Vec::new();
        entry_string(&mut metadata, "general.architecture", "draft");
        entry_string(&mut metadata, TOKENIZER_MODEL, "gpt2");
        entry_string_array(&mut metadata, TOKENIZER_TOKENS, &["one", "two"]);
        entry_u32(&mut metadata, "draft.vocab_size", 2);
        let mut tensors = Vec::new();
        tensor(&mut tensors, "draft.weight", &[2, 2], 0, 0);
        let source = header(metadata, 4, tensors, 1);

        let stub =
            compact_planner_stub(&source, &context, PlannerStubComponent::Companion).unwrap();
        let parsed = parse_header(&stub).unwrap();
        assert_eq!(parsed.architecture.as_deref(), Some("draft"));
        assert_eq!(parsed.token_count, Some(2));
        assert_eq!(parsed.declared_vocabulary_size, Some(2));
    }

    #[test]
    fn rejects_inconsistent_declared_vocabulary_size() {
        let mut source = primary_header();
        let parsed = parse_header(&source).unwrap();
        let insertion = parsed.tensor_directory.start;
        let mut entry = Vec::new();
        entry_u32(&mut entry, "llama.vocab_size", 4);
        source.splice(insertion..insertion, entry);
        source[16..24].copy_from_slice(&12_u64.to_le_bytes());
        source.resize(source.len().next_multiple_of(32), 0);
        assert!(matches!(
            planner_stub_context(&source),
            Err(PlannerStubError::Invalid(
                "token count differs from declared vocabulary size"
            ))
        ));
    }
}
