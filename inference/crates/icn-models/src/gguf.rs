//! Bounded GGUF metadata inspection without loading tensor contents.

use llama_cpp_2::gguf::{FileType, FileTypeNameError, UnknownFileType};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const MAGIC: [u8; 4] = *b"GGUF";
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_TENSORS: u64 = 10_000_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_STRING_ARRAY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIMS: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufInspection {
    pub version: u32,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub quantization: Option<String>,
    pub quantization_name: Option<String>,
    pub parameter_count: Option<u64>,
    pub active_parameter_count: Option<u64>,
    pub training_context_length: Option<u32>,
    pub nextn_predict_layers: Option<u32>,
    pub tokenizer: Option<String>,
    pub chat_template: Option<String>,
    pub tool_use_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub base_models: Vec<String>,
    pub modalities: Vec<String>,
    pub tensor_count: u64,
    pub header_bytes: u64,
    pub fingerprint_material: Vec<u8>,
    pub execution_role: Option<GgufExecutionRole>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufExecutionRole {
    Draft,
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("failed to read GGUF: {0}")]
    Io(#[from] io::Error),
    #[error("file is not GGUF")]
    InvalidMagic,
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("GGUF header is structurally invalid: {0}")]
    Invalid(&'static str),
    #[error("GGUF metadata string is not UTF-8")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error(transparent)]
    FileTypeName(#[from] FileTypeNameError),
    #[error(transparent)]
    UnknownFileType(#[from] UnknownFileType),
}

#[derive(Debug, Clone)]
enum Value {
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    String(String),
    StringArray(Vec<String>),
    Other,
}

pub fn inspect(path: &Path) -> Result<GgufInspection, GgufError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = CheckedReader::new(BufReader::new(file), file_len);

    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = reader.u32()?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count > MAX_TENSORS {
        return Err(GgufError::Invalid("tensor count exceeds inspection bound"));
    }
    if metadata_count > MAX_METADATA_ENTRIES {
        return Err(GgufError::Invalid(
            "metadata count exceeds inspection bound",
        ));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        let value = reader.value(value_type)?;
        metadata.insert(key, value);
    }

    let mut derived_parameter_count = 0_u64;
    for _ in 0..tensor_count {
        let _name = reader.string()?;
        let dimensions = reader.u32()?;
        if dimensions == 0 || dimensions > MAX_DIMS {
            return Err(GgufError::Invalid("tensor dimension count is invalid"));
        }
        let mut elements = 1_u64;
        for _ in 0..dimensions {
            elements = elements
                .checked_mul(reader.u64()?)
                .ok_or(GgufError::Invalid("tensor element count overflow"))?;
        }
        derived_parameter_count = derived_parameter_count
            .checked_add(elements)
            .ok_or(GgufError::Invalid("model parameter count overflow"))?;
        let _tensor_type = reader.u32()?;
        let _offset = reader.u64()?;
    }

    let alignment = u64::from(u32_value(&metadata, "general.alignment").unwrap_or(32));
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::Invalid("GGUF alignment is invalid"));
    }
    let header_bytes = reader
        .position
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(GgufError::Invalid("GGUF header alignment overflow"))?;
    if header_bytes > file_len {
        return Err(GgufError::Invalid("GGUF header extends beyond end of file"));
    }

    let architecture = string_value(&metadata, "general.architecture");
    let execution_role = if architecture.as_deref() == Some("eagle3")
        || string_value(&metadata, "dflash.decoder_arch").is_some()
    {
        Some(GgufExecutionRole::Draft)
    } else {
        None
    };
    let training_context_length = architecture
        .as_ref()
        .and_then(|architecture| u32_value(&metadata, &format!("{architecture}.context_length")))
        .or_else(|| u32_value(&metadata, "llama.context_length"));
    let nextn_predict_layers = architecture.as_ref().and_then(|architecture| {
        u32_value(&metadata, &format!("{architecture}.nextn_predict_layers"))
    });
    // The expert activation ratio cannot be applied to the whole model: embeddings, attention,
    // shared experts, and output tensors remain active. Publish no active count unless GGUF grows
    // an authoritative aggregate field.
    let active_parameter_count = u64_value(&metadata, "general.active_parameter_count");
    let base_model_count = u32_value(&metadata, "general.base_model.count").unwrap_or(0);
    let base_models = (0..base_model_count)
        .filter_map(|index| string_value(&metadata, &format!("general.base_model.{index}.name")))
        .collect();
    let modalities = if metadata
        .keys()
        .any(|key| key.contains("vision") || key.contains("clip") || key.contains("projector"))
    {
        vec!["text".to_owned(), "image".to_owned()]
    } else {
        vec!["text".to_owned()]
    };
    let file_type = u32_value(&metadata, "general.file_type")
        .map(FileType::try_from)
        .transpose()?;
    let quantization = file_type
        .map(FileType::name)
        .transpose()?
        .map(str::to_owned);
    let quantization_name = file_type.map(friendly_quantization_name).map(str::to_owned);
    let parameter_count = u64_value(&metadata, "general.parameter_count")
        .or((derived_parameter_count > 0).then_some(derived_parameter_count));
    let tokens = string_array_value(&metadata, "tokenizer.ggml.tokens");
    let token_at = |key: &str| {
        u32_value(&metadata, key)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| tokens.and_then(|tokens| tokens.get(index)))
            .cloned()
    };

    let mut fingerprint_material = Vec::new();
    fingerprint_material.extend_from_slice(&version.to_le_bytes());
    fingerprint_material.extend_from_slice(&tensor_count.to_le_bytes());
    fingerprint_material.extend_from_slice(&metadata_count.to_le_bytes());
    if let Some(value) = architecture.as_ref() {
        fingerprint_material.extend_from_slice(value.as_bytes());
    }
    if let Some(value) = string_value(&metadata, "tokenizer.chat_template") {
        fingerprint_material.extend_from_slice(value.as_bytes());
    }

    Ok(GgufInspection {
        version,
        architecture,
        name: string_value(&metadata, "general.name"),
        quantization,
        quantization_name,
        parameter_count,
        active_parameter_count,
        training_context_length,
        nextn_predict_layers,
        tokenizer: string_value(&metadata, "tokenizer.ggml.model"),
        chat_template: string_value(&metadata, "tokenizer.chat_template"),
        tool_use_template: string_value(&metadata, "tokenizer.chat_template.tool_use"),
        bos_token: token_at("tokenizer.ggml.bos_token_id"),
        eos_token: token_at("tokenizer.ggml.eos_token_id"),
        base_models,
        modalities,
        tensor_count,
        header_bytes,
        fingerprint_material,
        execution_role,
    })
}

const fn friendly_quantization_name(file_type: FileType) -> &'static str {
    match file_type {
        FileType::AllF32 => "32-bit",
        FileType::MostlyF16 | FileType::MostlyBf16 => "16-bit",
        FileType::MostlyQ8_0 => "8-bit",
        FileType::MostlyQ6K => "6-bit",
        FileType::MostlyQ5_0
        | FileType::MostlyQ5_1
        | FileType::MostlyQ5KSmall
        | FileType::MostlyQ5KMedium => "5-bit",
        FileType::MostlyQ4_0
        | FileType::MostlyQ4_1
        | FileType::MostlyQ4KSmall
        | FileType::MostlyQ4KMedium
        | FileType::MostlyIq4Nl
        | FileType::MostlyIq4Xs
        | FileType::MostlyMxfp4Moe
        | FileType::MostlyNvfp4 => "4-bit",
        FileType::MostlyQ3KSmall
        | FileType::MostlyQ3KMedium
        | FileType::MostlyQ3KLarge
        | FileType::MostlyIq3Xs
        | FileType::MostlyIq3Xxs
        | FileType::MostlyIq3Small
        | FileType::MostlyIq3Medium => "3-bit",
        FileType::MostlyQ2K
        | FileType::MostlyIq2Xxs
        | FileType::MostlyIq2Xs
        | FileType::MostlyQ2KSmall
        | FileType::MostlyIq2Small
        | FileType::MostlyIq2Medium
        | FileType::MostlyTq2_0
        | FileType::MostlyQ2_0 => "2-bit",
        FileType::MostlyIq1Small
        | FileType::MostlyIq1Medium
        | FileType::MostlyTq1_0
        | FileType::MostlyQ1_0 => "1-bit",
        FileType::Guessed => "unknown",
    }
}

fn string_value(values: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    match values.get(key) {
        Some(Value::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn string_array_value<'a>(values: &'a BTreeMap<String, Value>, key: &str) -> Option<&'a [String]> {
    match values.get(key) {
        Some(Value::StringArray(values)) => Some(values),
        _ => None,
    }
}

fn u32_value(values: &BTreeMap<String, Value>, key: &str) -> Option<u32> {
    match values.get(key) {
        Some(Value::U32(value)) => Some(*value),
        Some(Value::I32(value)) => u32::try_from(*value).ok(),
        Some(Value::U64(value)) => u32::try_from(*value).ok(),
        Some(Value::I64(value)) => u32::try_from(*value).ok(),
        _ => None,
    }
}

fn u64_value(values: &BTreeMap<String, Value>, key: &str) -> Option<u64> {
    match values.get(key) {
        Some(Value::U32(value)) => Some(u64::from(*value)),
        Some(Value::I32(value)) => u64::try_from(*value).ok(),
        Some(Value::U64(value)) => Some(*value),
        Some(Value::I64(value)) => u64::try_from(*value).ok(),
        _ => None,
    }
}

struct CheckedReader<R> {
    inner: R,
    position: u64,
    length: u64,
}

impl<R: Read + Seek> CheckedReader<R> {
    fn new(inner: R, length: u64) -> Self {
        Self {
            inner,
            position: 0,
            length,
        }
    }

    fn read_exact(&mut self, buffer: &mut [u8]) -> Result<(), GgufError> {
        let amount =
            u64::try_from(buffer.len()).map_err(|_| GgufError::Invalid("read length overflow"))?;
        self.ensure_remaining(amount)?;
        self.inner.read_exact(buffer)?;
        self.position += amount;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        let mut bytes = [0; 1];
        self.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(i16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(i64::from_le_bytes(bytes))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let length = self.u64()?;
        if length > MAX_STRING_BYTES {
            return Err(GgufError::Invalid(
                "metadata string exceeds inspection bound",
            ));
        }
        let length = usize::try_from(length)
            .map_err(|_| GgufError::Invalid("metadata string length overflows usize"))?;
        let mut bytes = vec![0; length];
        self.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }

    fn value(&mut self, value_type: u32) -> Result<Value, GgufError> {
        match value_type {
            0 => Ok(Value::U32(u32::from(self.u8()?))),
            1 => Ok(Value::I32(i32::from(self.u8()? as i8))),
            2 => Ok(Value::U32(u32::from(self.u16()?))),
            3 => Ok(Value::I32(i32::from(self.i16()?))),
            4 => Ok(Value::U32(self.u32()?)),
            5 => Ok(Value::I32(self.i32()?)),
            6 => {
                self.skip(4)?;
                Ok(Value::Other)
            }
            7 => {
                let _ = self.u8()?;
                Ok(Value::Other)
            }
            8 => Ok(Value::String(self.string()?)),
            9 => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                if count > MAX_METADATA_ENTRIES {
                    return Err(GgufError::Invalid(
                        "metadata array exceeds inspection bound",
                    ));
                }
                if element_type == 8 {
                    let mut values = Vec::with_capacity(usize::try_from(count).map_err(|_| {
                        GgufError::Invalid("metadata array length overflows usize")
                    })?);
                    let mut total = 0_u64;
                    for _ in 0..count {
                        let value = self.string()?;
                        total = total
                            .checked_add(value.len() as u64)
                            .ok_or(GgufError::Invalid("metadata array size overflow"))?;
                        if total > MAX_STRING_ARRAY_BYTES {
                            return Err(GgufError::Invalid(
                                "metadata string array exceeds inspection bound",
                            ));
                        }
                        values.push(value);
                    }
                    return Ok(Value::StringArray(values));
                }
                for _ in 0..count {
                    let _ = self.value(element_type)?;
                }
                Ok(Value::Other)
            }
            10 => Ok(Value::U64(self.u64()?)),
            11 => Ok(Value::I64(self.i64()?)),
            12 => {
                self.skip(8)?;
                Ok(Value::Other)
            }
            _ => Err(GgufError::Invalid("unknown metadata value type")),
        }
    }

    fn skip(&mut self, amount: u64) -> Result<(), GgufError> {
        self.ensure_remaining(amount)?;
        let amount =
            i64::try_from(amount).map_err(|_| GgufError::Invalid("seek length overflows i64"))?;
        self.inner.seek(SeekFrom::Current(amount))?;
        self.position += amount as u64;
        Ok(())
    }

    fn ensure_remaining(&self, amount: u64) -> Result<(), GgufError> {
        if self
            .position
            .checked_add(amount)
            .is_none_or(|end| end > self.length)
        {
            return Err(GgufError::Invalid("header extends beyond end of file"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_gguf_without_panicking() {
        let path = std::env::temp_dir().join(format!(
            "icn-gguf-invalid-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"not a gguf").unwrap();
        let result = inspect(&path);
        std::fs::remove_file(path).unwrap();
        assert!(matches!(result, Err(GgufError::InvalidMagic)));
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn extracts_effective_template_variants_and_token_strings() {
        let path = std::env::temp_dir().join(format!(
            "icn-gguf-template-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&5_u64.to_le_bytes());

        push_string(&mut bytes, "tokenizer.chat_template");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "default-template");
        push_string(&mut bytes, "tokenizer.chat_template.tool_use");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "tool-template");
        push_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        push_string(&mut bytes, "<bos>");
        push_string(&mut bytes, "<eos>");
        push_string(&mut bytes, "tokenizer.ggml.bos_token_id");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        push_string(&mut bytes, "tokenizer.ggml.eos_token_id");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());

        let aligned = bytes.len().next_multiple_of(32);
        bytes.resize(aligned, 0);

        std::fs::write(&path, bytes).unwrap();
        let result = inspect(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(result.chat_template.as_deref(), Some("default-template"));
        assert_eq!(result.tool_use_template.as_deref(), Some("tool-template"));
        assert_eq!(result.bos_token.as_deref(), Some("<bos>"));
        assert_eq!(result.eos_token.as_deref(), Some("<eos>"));
    }
}
