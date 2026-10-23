//! Assessment fixture headers (`engine/fixtures/assessment/headers/`): one GGUF header, trimmed so
//! the whole set stays small, gzipped.
//!
//! Trimming keeps every metadata key, the vocabulary's length and token kinds, and the whole
//! tensor directory. It empties `tokenizer.ggml.merges` and blanks the text and score of every
//! normal token that no special-token key names. Assessment reads none of what is cut.

use std::collections::HashSet;
use std::io::{Cursor, Write};
use std::path::Path;

use flate2::{Compression, GzBuilder};
use magnitude_artifacts::gguf::{
    Array, Element, StoredHeader, StoredType, Strings, Value, ValueType,
};

/// The token kind of a normal token (`LLAMA_TOKEN_TYPE_NORMAL`).
const NORMAL_TOKEN: u64 = 1;

/// `header` (every byte of a GGUF through its aligned tensor-data offset) trimmed.
pub fn trim(header: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut stored = StoredHeader::read(&mut Cursor::new(header))?;
    let mut special = stored
        .metadata
        .iter()
        .filter(|entry| {
            let name = &entry.metadata.name;
            name.starts_with("tokenizer.ggml.") && name.ends_with("_token_id")
        })
        .filter_map(|entry| entry.metadata.value.unsigned())
        .collect::<HashSet<_>>();
    if let Some(Value::Array(ids)) = stored.value("tokenizer.ggml.eos_token_ids") {
        special.extend(ids.iter().filter_map(Element::unsigned));
    }
    let kinds = match stored.value("tokenizer.ggml.token_type") {
        Some(Value::Array(kinds)) => kinds.iter().map(Element::unsigned).collect(),
        _ => Vec::new(),
    };
    let normal = |index: usize| {
        kinds.get(index) == Some(&Some(NORMAL_TOKEN)) && !special.contains(&(index as u64))
    };
    for entry in &mut stored.metadata {
        let value = &mut entry.metadata.value;
        match (entry.metadata.name.as_str(), &*value) {
            ("tokenizer.ggml.merges", _) => {
                *value = Value::Array(Array::String(Strings::default()));
                entry.stored = StoredType::Array(ValueType::String);
            }
            ("tokenizer.ggml.tokens", Value::Array(Array::String(tokens))) => {
                let tokens = tokens
                    .iter()
                    .enumerate()
                    .map(|(index, token)| if normal(index) { "" } else { token })
                    .collect::<Strings>();
                *value = Value::Array(Array::String(tokens));
            }
            ("tokenizer.ggml.scores", Value::Array(Array::Float(scores))) => {
                let scores = scores
                    .iter()
                    .enumerate()
                    .map(|(index, score)| if normal(index) { 0.0 } else { *score })
                    .collect();
                *value = Value::Array(Array::Float(scores));
            }
            _ => {}
        }
    }
    Ok(stored.encode()?)
}

/// Writes `header` gzipped to `output`, named in the gzip header as `name`, with no timestamp.
pub fn write_gzip(output: &Path, name: &str, header: &[u8]) -> anyhow::Result<()> {
    let mut encoder = GzBuilder::new()
        .filename(name)
        .mtime(0)
        .write(std::fs::File::create(output)?, Compression::best());
    encoder.write_all(header)?;
    encoder.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    /// Every committed fixture header is trimmed already, so trimming it again changes no byte.
    #[test]
    fn trimming_a_fixture_header_again_changes_nothing() {
        let headers =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine/fixtures/assessment/headers");
        for entry in std::fs::read_dir(headers).expect("fixture headers") {
            let path = entry.expect("fixture header").path();
            let mut header = Vec::new();
            flate2::read::GzDecoder::new(std::fs::File::open(&path).expect("fixture header"))
                .read_to_end(&mut header)
                .expect("a gzipped header");
            assert!(
                trim(&header).expect("trimmed") == header,
                "{}",
                path.display()
            );
        }
    }
}
