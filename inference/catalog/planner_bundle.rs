//! The release header bundle: a manifest plus content-addressed inputs (exact GGUF headers).
//!
//! Inputs are split at content-defined boundaries and every distinct chunk is stored once,
//! compressed, with its SHA-256 digest. Variants of one model share their tokenizer and template
//! metadata byte for byte, so they share those chunks whatever their tensor directories.
//! [`PlannerBundle::verify`] checks every chunk once; an input is reassembled exactly and verified
//! against its own digest on every read.
//!
//! ```text
//! MAGIC | u64 manifest length | manifest
//!       | u32 chunk count  | { 32-byte digest | u64 length | u64 compressed length | gzip bytes }*
//!       | u32 input count  | { 64-byte hex digest | u64 length | u32 chunk count | u32 chunk* }*
//! ```

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::ops::Range;

use flate2::Compression;
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"MAGPLAN4";
const MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
const MAX_PLANNER_INPUT_BYTES: usize = 128 * 1024 * 1024;

/// Content-defined chunking: a boundary follows a byte where the gear hash's top bits match
/// `BOUNDARY_MASK` (about one boundary per 64 KiB), within `MIN_CHUNK..=MAX_CHUNK`.
const MIN_CHUNK: usize = 16 * 1024;
const MAX_CHUNK: usize = 256 * 1024;
const BOUNDARY_MASK: u64 = 0xffff << 48;
const GEAR: [u64; 256] = gear_table();

const fn gear_table() -> [u64; 256] {
    // splitmix64 from a fixed seed: the table is part of the format.
    let mut table = [0_u64; 256];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut index = 0;
    while index < 256 {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        table[index] = value ^ (value >> 31);
        index += 1;
    }
    table
}

fn chunks(bytes: &[u8]) -> Vec<Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut hash = 0_u64;
    for (index, byte) in bytes.iter().enumerate() {
        hash = (hash << 1).wrapping_add(GEAR[usize::from(*byte)]);
        let length = index + 1 - start;
        if (length >= MIN_CHUNK && hash & BOUNDARY_MASK == 0) || length == MAX_CHUNK {
            chunks.push(start..index + 1);
            start = index + 1;
            hash = 0;
        }
    }
    if start < bytes.len() {
        chunks.push(start..bytes.len());
    }
    chunks
}

pub fn encode(
    manifest: &[u8],
    inputs: &BTreeMap<String, Vec<u8>>,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, String> {
    if manifest.is_empty() || manifest.len() > MAX_MANIFEST_BYTES {
        return Err("planner bundle manifest length is outside the supported bound".to_owned());
    }
    let mut chunk_indices = BTreeMap::<[u8; 32], u32>::new();
    let mut stored_chunks = Vec::<([u8; 32], usize, Vec<u8>)>::new();
    let mut references = Vec::with_capacity(inputs.len());
    let total = inputs.len();
    for (index, (digest, input)) in inputs.iter().enumerate() {
        validate_digest(digest)?;
        if input.is_empty() || input.len() > MAX_PLANNER_INPUT_BYTES {
            return Err(format!(
                "planner input {digest} is outside the supported bound"
            ));
        }
        if sha256(input) != *digest {
            return Err(format!(
                "planner input {digest} failed integrity validation"
            ));
        }
        let mut chunk_references = Vec::new();
        for range in chunks(input) {
            let chunk = &input[range];
            let key: [u8; 32] = Sha256::digest(chunk).into();
            let chunk_index = match chunk_indices.get(&key) {
                Some(chunk_index) => *chunk_index,
                None => {
                    let chunk_index = u32::try_from(stored_chunks.len())
                        .map_err(|_| "too many planner chunks".to_owned())?;
                    let mut compressor = GzEncoder::new(Vec::new(), Compression::best());
                    compressor
                        .write_all(chunk)
                        .map_err(|error| error.to_string())?;
                    stored_chunks.push((
                        key,
                        chunk.len(),
                        compressor.finish().map_err(|error| error.to_string())?,
                    ));
                    chunk_indices.insert(key, chunk_index);
                    chunk_index
                }
            };
            chunk_references.push(chunk_index);
        }
        references.push((digest, input.len(), chunk_references));
        progress(index + 1, total);
    }
    let mut encoded = Vec::new();
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&u64_of(manifest.len()).to_le_bytes());
    encoded.extend_from_slice(manifest);
    encoded.extend_from_slice(&u32_of(stored_chunks.len())?.to_le_bytes());
    for (digest, length, compressed) in &stored_chunks {
        encoded.extend_from_slice(digest);
        encoded.extend_from_slice(&u64_of(*length).to_le_bytes());
        encoded.extend_from_slice(&u64_of(compressed.len()).to_le_bytes());
        encoded.extend_from_slice(compressed);
    }
    encoded.extend_from_slice(&u32_of(references.len())?.to_le_bytes());
    for (digest, length, chunk_references) in references {
        encoded.extend_from_slice(digest.as_bytes());
        encoded.extend_from_slice(&u64_of(length).to_le_bytes());
        encoded.extend_from_slice(&u32_of(chunk_references.len())?.to_le_bytes());
        for chunk in chunk_references {
            encoded.extend_from_slice(&chunk.to_le_bytes());
        }
    }
    Ok(encoded)
}

fn u64_of(value: usize) -> u64 {
    u64::try_from(value).expect("a length fits u64")
}

fn u32_of(value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| "planner bundle count exceeds u32".to_owned())
}

#[derive(Debug)]
pub struct PlannerBundle<'a> {
    bytes: &'a [u8],
    manifest: Range<usize>,
    chunks: Vec<Chunk>,
    inputs: BTreeMap<String, Input>,
}

#[derive(Clone, Debug)]
struct Chunk {
    digest: [u8; 32],
    compressed: Range<usize>,
    length: usize,
}

#[derive(Clone, Debug)]
struct Input {
    length: usize,
    chunks: Vec<usize>,
}

impl<'a> PlannerBundle<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.get(..MAGIC.len()) != Some(MAGIC) {
            return Err("planner bundle has an invalid header".to_owned());
        }
        let mut cursor = MAGIC.len();
        let manifest_len = read_length(bytes, &mut cursor)?;
        if manifest_len == 0 || manifest_len > MAX_MANIFEST_BYTES {
            return Err("planner bundle manifest length is outside the supported bound".to_owned());
        }
        let manifest_start = cursor;
        read_bytes(bytes, &mut cursor, manifest_len)?;
        let manifest = manifest_start..cursor;
        let chunk_count = read_u32(bytes, &mut cursor)?;
        let mut chunks = Vec::new();
        for _ in 0..chunk_count {
            let digest = read_bytes(bytes, &mut cursor, 32)?
                .try_into()
                .expect("the digest slice has 32 bytes");
            let length = read_length(bytes, &mut cursor)?;
            if length == 0 || length > MAX_CHUNK {
                return Err("planner chunk length is outside the supported bound".to_owned());
            }
            let compressed_len = read_length(bytes, &mut cursor)?;
            let start = cursor;
            read_bytes(bytes, &mut cursor, compressed_len)?;
            chunks.push(Chunk {
                digest,
                compressed: start..cursor,
                length,
            });
        }
        let input_count = read_u32(bytes, &mut cursor)?;
        let mut inputs = BTreeMap::new();
        for _ in 0..input_count {
            let digest = std::str::from_utf8(read_bytes(bytes, &mut cursor, 64)?)
                .map_err(|error| error.to_string())?
                .to_owned();
            validate_digest(&digest)?;
            let length = read_length(bytes, &mut cursor)?;
            if length == 0 || length > MAX_PLANNER_INPUT_BYTES {
                return Err("planner input length is outside the supported bound".to_owned());
            }
            let reference_count = read_u32(bytes, &mut cursor)?;
            let mut input_chunks = Vec::new();
            let mut assembled = 0_usize;
            for _ in 0..reference_count {
                let chunk = usize::try_from(read_u32(bytes, &mut cursor)?)
                    .map_err(|_| "planner chunk index overflows".to_owned())?;
                let chunk_length = chunks
                    .get(chunk)
                    .ok_or_else(|| "planner input references a missing chunk".to_owned())?
                    .length;
                assembled = assembled
                    .checked_add(chunk_length)
                    .filter(|assembled| *assembled <= length)
                    .ok_or_else(|| "planner input chunks exceed its length".to_owned())?;
                input_chunks.push(chunk);
            }
            if assembled != length {
                return Err("planner input chunks do not cover its length".to_owned());
            }
            if inputs
                .insert(
                    digest,
                    Input {
                        length,
                        chunks: input_chunks,
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
        Ok(Self {
            bytes,
            manifest,
            chunks,
            inputs,
        })
    }

    pub fn manifest(&self) -> &'a [u8] {
        &self.bytes[self.manifest.clone()]
    }

    pub fn digests(&self) -> impl Iterator<Item = &str> {
        self.inputs.keys().map(String::as_str)
    }

    /// The declared length of an input, without reassembling it.
    pub fn input_len(&self, digest: &str) -> Option<usize> {
        self.inputs.get(digest).map(|input| input.length)
    }

    /// Verify every stored chunk against its digest, decompressing each once.
    pub fn verify(&self) -> Result<(), String> {
        let mut chunk = Vec::with_capacity(MAX_CHUNK);
        for (index, stored) in self.chunks.iter().enumerate() {
            chunk.clear();
            self.decompress(stored, &mut chunk)
                .map_err(|error| format!("planner chunk {index}: {error}"))?;
            if Sha256::digest(&chunk).as_slice() != stored.digest {
                return Err(format!("planner chunk {index} failed integrity validation"));
            }
        }
        Ok(())
    }

    /// Reassemble an input and verify it against its digest.
    pub fn input(&self, digest: &str) -> Result<Vec<u8>, String> {
        let entry = self
            .inputs
            .get(digest)
            .ok_or_else(|| format!("planner bundle is missing input {digest}"))?;
        let mut input = Vec::with_capacity(entry.length);
        for chunk in &entry.chunks {
            self.decompress(&self.chunks[*chunk], &mut input)
                .map_err(|error| format!("planner input {digest}: {error}"))?;
        }
        if sha256(&input) != digest {
            return Err(format!(
                "planner input {digest} failed integrity validation"
            ));
        }
        Ok(input)
    }

    fn decompress(&self, chunk: &Chunk, output: &mut Vec<u8>) -> Result<(), String> {
        let start = output.len();
        GzDecoder::new(&self.bytes[chunk.compressed.clone()])
            .take(u64_of(chunk.length) + 1)
            .read_to_end(output)
            .map_err(|error| error.to_string())?;
        if output.len() - start != chunk.length {
            return Err("chunk length differs from its declaration".to_owned());
        }
        Ok(())
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

fn read_length(bytes: &[u8], cursor: &mut usize) -> Result<usize, String> {
    let value: [u8; 8] = read_bytes(bytes, cursor, 8)?
        .try_into()
        .map_err(|_| "invalid planner bundle integer".to_owned())?;
    usize::try_from(u64::from_le_bytes(value))
        .map_err(|_| "planner bundle length overflows".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(length: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 56) as u8
            })
            .collect()
    }

    fn bundle_of(inputs: &[Vec<u8>]) -> (Vec<u8>, Vec<String>) {
        let map = inputs
            .iter()
            .map(|input| (sha256(input), input.clone()))
            .collect::<BTreeMap<_, _>>();
        let digests = inputs.iter().map(|input| sha256(input)).collect();
        (encode(b"manifest", &map, |_, _| {}).unwrap(), digests)
    }

    #[test]
    fn bundle_round_trips_verified_inputs() {
        let inputs = vec![b"small header".to_vec(), pseudo_random(700_000, 1)];
        let (encoded, digests) = bundle_of(&inputs);
        let bundle = PlannerBundle::parse(&encoded).unwrap();
        assert_eq!(bundle.manifest(), b"manifest");
        bundle.verify().unwrap();
        for (input, digest) in inputs.iter().zip(&digests) {
            assert!(bundle.digests().any(|candidate| candidate == digest));
            assert_eq!(bundle.input_len(digest), Some(input.len()));
            assert_eq!(&bundle.input(digest).unwrap(), input);
        }
    }

    #[test]
    fn shared_content_is_stored_once() {
        // Two headers that differ only in a leading region share every later chunk.
        let shared = pseudo_random(2_000_000, 2);
        let first = [pseudo_random(10_000, 3), shared.clone()].concat();
        let second = [pseudo_random(30_000, 4), shared].concat();
        let (together, _) = bundle_of(&[first.clone(), second]);
        let (alone, _) = bundle_of(&[first]);
        assert!(together.len() < alone.len() + alone.len() / 10);
    }

    #[test]
    fn chunk_boundaries_respect_their_bounds() {
        let bytes = pseudo_random(3_000_000, 5);
        let ranges = chunks(&bytes);
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, bytes.len());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
            assert!((MIN_CHUNK..=MAX_CHUNK).contains(&pair[0].len()));
        }
    }

    #[test]
    fn corrupted_compressed_input_fails_integrity_validation() {
        let input = pseudo_random(100_000, 6);
        let (mut encoded, digests) = bundle_of(&[input]);
        // Flip a byte inside the first stored chunk's compressed payload.
        let first_chunk = MAGIC.len() + 8 + b"manifest".len() + 4 + 32 + 16 + 20;
        encoded[first_chunk] ^= 1;
        let bundle = PlannerBundle::parse(&encoded).unwrap();
        assert!(bundle.verify().is_err());
        assert!(bundle.input(&digests[0]).is_err());
    }

    #[test]
    fn old_bundle_format_is_not_accepted() {
        assert!(PlannerBundle::parse(b"MAGPLAN3\0\0\0\0").is_err());
    }

    #[test]
    fn invalid_declared_input_size_is_rejected_before_decompression() {
        let input = b"x".to_vec();
        let (mut encoded, _) = bundle_of(&[input]);
        let chunks = MAGIC.len() + 8 + b"manifest".len() + 4;
        let compressed_len = u64::from_le_bytes(
            encoded[chunks + 32 + 8..chunks + 32 + 16]
                .try_into()
                .unwrap(),
        );
        let size_offset = chunks + 32 + 16 + usize::try_from(compressed_len).unwrap() + 4 + 64;
        encoded[size_offset..size_offset + 8]
            .copy_from_slice(&((MAX_PLANNER_INPUT_BYTES as u64) + 1).to_le_bytes());
        assert!(PlannerBundle::parse(&encoded).is_err());
    }
}
