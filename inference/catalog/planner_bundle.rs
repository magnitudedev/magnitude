use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::ops::Range;

use flate2::Compression;
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"MAGPLAN2";
const MAX_PLANNER_INPUT_BYTES: usize = 128 * 1024 * 1024;

pub fn encode(
    inputs: &BTreeMap<String, Vec<u8>>,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, String> {
    let mut encoded = Vec::new();
    let total = inputs.len();
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(
        &u32::try_from(inputs.len())
            .map_err(|_| "too many planner inputs".to_owned())?
            .to_le_bytes(),
    );
    for (index, (digest, input)) in inputs.iter().enumerate() {
        validate_digest(digest)?;
        if sha256(input) != *digest {
            return Err(format!(
                "planner input {digest} failed integrity validation"
            ));
        }
        let mut compressor = GzEncoder::new(Vec::new(), Compression::fast());
        compressor
            .write_all(input)
            .map_err(|error| error.to_string())?;
        let compressed = compressor.finish().map_err(|error| error.to_string())?;
        encoded.extend_from_slice(digest.as_bytes());
        encoded.extend_from_slice(
            &u64::try_from(input.len())
                .map_err(|_| "planner input is too large".to_owned())?
                .to_le_bytes(),
        );
        encoded.extend_from_slice(
            &u64::try_from(compressed.len())
                .map_err(|_| "compressed planner input is too large".to_owned())?
                .to_le_bytes(),
        );
        encoded.extend_from_slice(&compressed);
        progress(index + 1, total);
    }
    Ok(encoded)
}

#[derive(Debug)]
pub struct PlannerBundle<'a> {
    bytes: &'a [u8],
    entries: BTreeMap<String, Entry>,
}

#[derive(Clone, Debug)]
struct Entry {
    compressed: Range<usize>,
    uncompressed_len: usize,
}

impl<'a> PlannerBundle<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.get(..MAGIC.len()) != Some(MAGIC) {
            return Err("planner bundle has an invalid header".to_owned());
        }
        let mut cursor = MAGIC.len();
        let count = read_u32(bytes, &mut cursor)?;
        let mut entries = BTreeMap::new();
        for _ in 0..count {
            let digest = std::str::from_utf8(read_bytes(bytes, &mut cursor, 64)?)
                .map_err(|error| error.to_string())?
                .to_owned();
            validate_digest(&digest)?;
            let uncompressed_len = usize::try_from(read_u64(bytes, &mut cursor)?)
                .map_err(|_| "planner input is too large".to_owned())?;
            if uncompressed_len == 0 || uncompressed_len > MAX_PLANNER_INPUT_BYTES {
                return Err("planner input length is outside the supported bound".to_owned());
            }
            let compressed_len = usize::try_from(read_u64(bytes, &mut cursor)?)
                .map_err(|_| "compressed planner input is too large".to_owned())?;
            let start = cursor;
            read_bytes(bytes, &mut cursor, compressed_len)?;
            if entries
                .insert(
                    digest,
                    Entry {
                        compressed: start..cursor,
                        uncompressed_len,
                    },
                )
                .is_some()
            {
                return Err("planner bundle contains a duplicate input".to_owned());
            }
        }
        if cursor != bytes.len() {
            return Err("planner bundle contains trailing bytes".to_owned());
        }
        Ok(Self { bytes, entries })
    }

    pub fn contains(&self, digest: &str) -> bool {
        self.entries.contains_key(digest)
    }

    pub fn input(&self, digest: &str) -> Result<Vec<u8>, String> {
        let entry = self
            .entries
            .get(digest)
            .ok_or_else(|| format!("planner bundle is missing input {digest}"))?;
        let mut decoder = GzDecoder::new(&self.bytes[entry.compressed.clone()]);
        let mut input = Vec::with_capacity(entry.uncompressed_len);
        decoder
            .read_to_end(&mut input)
            .map_err(|error| error.to_string())?;
        if input.len() != entry.uncompressed_len || sha256(&input) != digest {
            return Err(format!(
                "planner input {digest} failed integrity validation"
            ));
        }
        Ok(input)
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_digest(digest: &str) -> Result<(), String> {
    if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!("invalid planner input digest {digest}"))
    }
}

fn read_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, length: usize) -> Result<&'a [u8], String> {
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| "planner bundle offset overflow".to_owned())?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| "planner bundle ended unexpectedly".to_owned())?;
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let value = read_bytes(bytes, cursor, 4)?
        .try_into()
        .map_err(|_| "invalid planner bundle integer".to_owned())?;
    Ok(u32::from_le_bytes(value))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let value = read_bytes(bytes, cursor, 8)?
        .try_into()
        .map_err(|_| "invalid planner bundle integer".to_owned())?;
    Ok(u64::from_le_bytes(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_round_trips_verified_inputs() {
        let input = b"compact planner input".to_vec();
        let digest = sha256(&input);
        let encoded = encode(
            &BTreeMap::from([(digest.clone(), input.clone())]),
            |_, _| {},
        )
        .unwrap();
        let bundle = PlannerBundle::parse(&encoded).unwrap();
        assert!(bundle.contains(&digest));
        assert_eq!(bundle.input(&digest).unwrap(), input);
    }

    #[test]
    fn old_bundle_format_is_not_accepted() {
        assert!(PlannerBundle::parse(b"MAGPLAN1\0\0\0\0").is_err());
    }

    #[test]
    fn invalid_declared_input_size_is_rejected_before_decompression() {
        let input = b"x".to_vec();
        let digest = sha256(&input);
        let mut encoded = encode(&BTreeMap::from([(digest, input)]), |_, _| {}).unwrap();
        let size_offset = MAGIC.len() + 4 + 64;
        encoded[size_offset..size_offset + 8]
            .copy_from_slice(&((MAX_PLANNER_INPUT_BYTES as u64) + 1).to_le_bytes());
        assert!(PlannerBundle::parse(&encoded).is_err());
    }
}
