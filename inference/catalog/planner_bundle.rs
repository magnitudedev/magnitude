use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::ops::Range;

use flate2::Compression;
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"MAGPLAN1";

pub fn encode(
    headers: &BTreeMap<String, Vec<u8>>,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, String> {
    let mut encoded = Vec::new();
    let total = headers.len();
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(
        &u32::try_from(headers.len())
            .map_err(|_| "too many planner headers".to_owned())?
            .to_le_bytes(),
    );
    for (index, (digest, header)) in headers.iter().enumerate() {
        validate_digest(digest)?;
        if sha256(header) != *digest {
            return Err(format!(
                "planner header {digest} failed integrity validation"
            ));
        }
        let mut compressor = GzEncoder::new(Vec::new(), Compression::fast());
        compressor
            .write_all(header)
            .map_err(|error| error.to_string())?;
        let compressed = compressor.finish().map_err(|error| error.to_string())?;
        encoded.extend_from_slice(digest.as_bytes());
        encoded.extend_from_slice(
            &u64::try_from(header.len())
                .map_err(|_| "planner header is too large".to_owned())?
                .to_le_bytes(),
        );
        encoded.extend_from_slice(
            &u64::try_from(compressed.len())
                .map_err(|_| "compressed planner header is too large".to_owned())?
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
                .map_err(|_| "planner header is too large".to_owned())?;
            let compressed_len = usize::try_from(read_u64(bytes, &mut cursor)?)
                .map_err(|_| "compressed planner header is too large".to_owned())?;
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
                return Err("planner bundle contains a duplicate header".to_owned());
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

    pub fn header(&self, digest: &str) -> Result<Vec<u8>, String> {
        let entry = self
            .entries
            .get(digest)
            .ok_or_else(|| format!("planner bundle is missing header {digest}"))?;
        let mut decoder = GzDecoder::new(&self.bytes[entry.compressed.clone()]);
        let mut header = Vec::with_capacity(entry.uncompressed_len);
        decoder
            .read_to_end(&mut header)
            .map_err(|error| error.to_string())?;
        if header.len() != entry.uncompressed_len || sha256(&header) != digest {
            return Err(format!(
                "planner header {digest} failed integrity validation"
            ));
        }
        Ok(header)
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_digest(digest: &str) -> Result<(), String> {
    if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!("invalid planner header digest {digest}"))
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
