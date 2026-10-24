//! The GGUF header cache: every header fetched by range request, by its sha256, within a size
//! bound.
//!
//! Headers of one repository's files share their tokenizer, nearly all of each header's bytes. A
//! header is therefore stored as a recipe of segments: each large metadata entry ([`shared`]) is
//! a chunk file named by its own sha256 and stored once however many headers hold it, and the rest of the header is inline in the recipe. The cache is bounded by the bytes
//! of its recipes and the chunks they reference. Reading a header marks it used; once the cache
//! outgrows its capacity the least recently used headers are evicted, except those pinned by the
//! caller (installed packages and the release catalog's headers) and those used within a grace
//! period (an operation in flight). Chunks no recipe references any more go with them. Every read
//! verifies the assembled bytes against the digest, so an evicted, partially evicted or damaged
//! header is a miss that its caller re-fetches.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use magnitude_service_utils::file_cache::{read_bytes, write_bytes_atomic};
use sha2::{Digest, Sha256};

use magnitude_artifacts::gguf::{Entry, HEADER_LIMIT, Walk, walk};

/// A metadata entry at least this large (a tokenizer's token list, scores, types or merges) is
/// shared between headers instead of read or stored again.
const SHARED_ENTRY_BYTES: usize = 64 * 1024;

/// Whether `entry` is stored once for every header holding it, and read from another header
/// holding it instead of fetched again.
pub(crate) fn shared(entry: &Entry<'_>) -> bool {
    entry.range().len() >= SHARED_ENTRY_BYTES
}

/// The default bound on the header cache's bytes.
pub(crate) const HEADER_CAPACITY_BYTES: u64 = 512 * 1024 * 1024;
/// A header used this recently is never evicted.
pub(crate) const HEADER_EVICTION_GRACE: Duration = Duration::from_secs(10 * 60);

const RECIPE_MAGIC: &[u8; 4] = b"MGHR";
const RECIPE_VERSION: u32 = 1;
const INLINE: u8 = 0;
const CHUNK: u8 = 1;
const MAX_SEGMENTS: usize = 4096;

#[derive(Debug)]
struct SweepState {
    written_since_sweep: u64,
    swept: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct HeaderCache {
    recipes: PathBuf,
    chunks: PathBuf,
    lock: PathBuf,
    capacity: u64,
    grace: Duration,
    state: Arc<Mutex<SweepState>>,
}

#[derive(Debug, PartialEq, Eq)]
enum Segment {
    Inline(usize),
    Chunk { digest: String, length: usize },
}

impl HeaderCache {
    pub(crate) fn new(root: &Path, capacity: u64, grace: Duration) -> Self {
        Self {
            recipes: root.join("blobs/gguf-headers"),
            chunks: root.join("blobs/gguf-header-chunks"),
            lock: root.join(".locks/gguf-headers.lock"),
            capacity,
            grace,
            state: Arc::new(Mutex::new(SweepState {
                written_since_sweep: 0,
                swept: false,
            })),
        }
    }

    /// The header with this sha256, marked used, or `None` when it isn't (completely) stored.
    pub(crate) fn read(&self, digest: &str) -> Option<Vec<u8>> {
        if !valid_digest(digest) {
            return None;
        }
        let path = self.recipes.join(digest);
        let recipe = read_bytes(&path, HEADER_LIMIT)?;
        match self
            .assemble(&recipe)
            .filter(|header| hex_sha256(header) == digest)
        {
            Some(header) => {
                touch(&path);
                Some(header)
            }
            None => {
                self.discard(&path, &recipe);
                None
            }
        }
    }

    /// The bytes a recipe assembles, when it parses and every chunk it names is whole.
    fn assemble(&self, recipe: &[u8]) -> Option<Vec<u8>> {
        let (segments, mut inline) = parse_table(recipe)?;
        let mut header = Vec::new();
        for segment in segments {
            match segment {
                Segment::Inline(length) => {
                    let (bytes, rest) = inline.split_at_checked(length)?;
                    header.extend_from_slice(bytes);
                    inline = rest;
                }
                Segment::Chunk { digest, length } => {
                    let chunk = read_bytes(&self.chunks.join(&digest), length)?;
                    if chunk.len() != length {
                        return None;
                    }
                    header.extend_from_slice(&chunk);
                }
            }
            if header.len() > HEADER_LIMIT {
                return None;
            }
        }
        inline.is_empty().then_some(header)
    }

    /// Removes a recipe that doesn't assemble its header, and every chunk it names whose bytes
    /// aren't their name's, so the next write stores them again.
    fn discard(&self, path: &Path, recipe: &[u8]) {
        let _ = fs::remove_file(path);
        for segment in parse_table(recipe)
            .map(|(segments, _)| segments)
            .unwrap_or_default()
        {
            if let Segment::Chunk { digest, length } = segment {
                let chunk = self.chunks.join(&digest);
                if read_bytes(&chunk, length).is_none_or(|bytes| hex_sha256(&bytes) != digest) {
                    let _ = fs::remove_file(chunk);
                }
            }
        }
    }

    /// Whether the header with this sha256 is stored with every chunk it needs, marking it used
    /// for the read that follows; `read` still verifies it.
    pub(crate) fn contains(&self, digest: &str) -> bool {
        let path = self.recipes.join(digest);
        let stored = valid_digest(digest)
            && self
                .recipe_chunks(&path)
                .is_some_and(|chunks| chunks.iter().all(|chunk| self.chunks.join(chunk).is_file()));
        if stored {
            touch(&path);
        }
        stored
    }

    /// Stores `header`, returning its sha256, and evicts the least recently used headers not in
    /// `pinned` once the cache has outgrown its capacity.
    pub(crate) fn write(&self, header: &[u8], pinned: impl FnOnce() -> BTreeSet<String>) -> String {
        let digest = hex_sha256(header);
        if header.len() > HEADER_LIMIT {
            return digest;
        }
        let Ok(mut state) = self.state.lock() else {
            return digest;
        };
        let recipe_path = self.recipes.join(&digest);
        if !self.contains(&digest) {
            let mut segments = Vec::new();
            let mut inline = Vec::new();
            let mut written = 0_u64;
            let mut cursor = 0;
            if let Ok(Walk::Complete(layout)) = walk(header) {
                for entry in layout.entries.iter().filter(|entry| shared(entry)) {
                    let range = entry.range();
                    if range.start > cursor {
                        segments.push(Segment::Inline(range.start - cursor));
                        inline.extend_from_slice(&header[cursor..range.start]);
                    }
                    let bytes = &header[range];
                    let chunk = hex_sha256(bytes);
                    let path = self.chunks.join(&chunk);
                    if fs::metadata(&path)
                        .is_ok_and(|metadata| metadata.len() == bytes.len() as u64)
                    {
                        touch(&path);
                    } else {
                        write_bytes_atomic(&path, &self.lock, bytes, HEADER_LIMIT);
                        written += bytes.len() as u64;
                    }
                    segments.push(Segment::Chunk {
                        digest: chunk,
                        length: bytes.len(),
                    });
                    cursor = entry.end;
                }
            }
            if cursor < header.len() {
                segments.push(Segment::Inline(header.len() - cursor));
                inline.extend_from_slice(&header[cursor..]);
            }
            let recipe = encode_recipe(&segments, &inline);
            write_bytes_atomic(&recipe_path, &self.lock, &recipe, HEADER_LIMIT);
            written += recipe.len() as u64;
            state.written_since_sweep += written;
        }
        if !state.swept || state.written_since_sweep >= self.capacity / 8 {
            self.sweep(&pinned());
            state.swept = true;
            state.written_since_sweep = 0;
        }
        digest
    }

    /// Evicts least recently used headers until the cache fits its capacity, keeping pinned and
    /// recently used ones, then removes chunks no remaining header references.
    fn sweep(&self, pinned: &BTreeSet<String>) {
        let now = SystemTime::now();
        let recent = |modified: SystemTime| {
            now.duration_since(modified)
                .map_or(true, |age| age < self.grace)
        };
        let chunk_files = files(&self.chunks)
            .into_iter()
            .map(|(name, path, size, modified)| (name, (path, size, modified)))
            .collect::<BTreeMap<_, _>>();
        let mut recipes = Vec::new();
        for (name, path, size, modified) in files(&self.recipes) {
            match self.recipe_chunks(&path).filter(|_| valid_digest(&name)) {
                Some(chunks) => recipes.push((name, path, size, modified, chunks)),
                None if !recent(modified) => {
                    let _ = fs::remove_file(&path);
                }
                None => {}
            }
        }
        recipes.sort_by(|left, right| right.3.cmp(&left.3));
        let mut referenced = BTreeSet::new();
        let mut total = 0_u64;
        for (name, path, size, modified, chunks) in recipes {
            let added = size
                + chunks
                    .iter()
                    .filter(|chunk| !referenced.contains(*chunk))
                    .filter_map(|chunk| chunk_files.get(chunk).map(|(_, size, _)| *size))
                    .sum::<u64>();
            if pinned.contains(&name) || recent(modified) || total + added <= self.capacity {
                total += added;
                referenced.extend(chunks);
            } else {
                let _ = fs::remove_file(&path);
            }
        }
        for (name, (path, _, modified)) in chunk_files {
            if !referenced.contains(&name) && !recent(modified) {
                let _ = fs::remove_file(&path);
            }
        }
    }

    /// The chunks a stored recipe references, read from its segment table alone.
    fn recipe_chunks(&self, path: &Path) -> Option<Vec<String>> {
        let mut file = fs::File::open(path).ok()?;
        let mut preamble = [0_u8; 12];
        file.read_exact(&mut preamble).ok()?;
        let count = segment_count(&preamble)?;
        let mut bytes = preamble.to_vec();
        file.take((count * (1 + 8 + 32)) as u64)
            .read_to_end(&mut bytes)
            .ok()?;
        let (segments, _) = parse_table(&bytes)?;
        Some(
            segments
                .into_iter()
                .filter_map(|segment| match segment {
                    Segment::Chunk { digest, .. } => Some(digest),
                    Segment::Inline(_) => None,
                })
                .collect(),
        )
    }
}

/// The regular files of `directory`: name, path, size and modification time.
fn files(directory: &Path) -> Vec<(String, PathBuf, u64, SystemTime)> {
    let Ok(read) = fs::read_dir(directory) else {
        return Vec::new();
    };
    read.filter_map(Result::ok)
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            if !metadata.is_file() {
                return None;
            }
            Some((
                entry.file_name().into_string().ok()?,
                entry.path(),
                metadata.len(),
                metadata.modified().ok()?,
            ))
        })
        .collect()
}

/// Marks a file used. Failure only makes it look older.
fn touch(path: &Path) {
    if let Ok(file) = fs::File::options().write(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

/// `MGHR`, the format version, the segment count, each segment's kind, length and (for a
/// chunk) sha256, then the inline bytes in order.
fn encode_recipe(segments: &[Segment], inline: &[u8]) -> Vec<u8> {
    let mut bytes = RECIPE_MAGIC.to_vec();
    bytes.extend_from_slice(&RECIPE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(segments.len() as u32).to_le_bytes());
    for segment in segments {
        match segment {
            Segment::Inline(length) => {
                bytes.push(INLINE);
                bytes.extend_from_slice(&(*length as u64).to_le_bytes());
            }
            Segment::Chunk { digest, length } => {
                bytes.push(CHUNK);
                bytes.extend_from_slice(&(*length as u64).to_le_bytes());
                bytes.extend((0..digest.len() / 2).map(|index| {
                    u8::from_str_radix(&digest[2 * index..2 * index + 2], 16).unwrap_or(0)
                }));
            }
        }
    }
    bytes.extend_from_slice(inline);
    bytes
}

fn segment_count(preamble: &[u8]) -> Option<usize> {
    if preamble.get(..4)? != RECIPE_MAGIC
        || u32::from_le_bytes(preamble.get(4..8)?.try_into().ok()?) != RECIPE_VERSION
    {
        return None;
    }
    let count = u32::from_le_bytes(preamble.get(8..12)?.try_into().ok()?) as usize;
    (count <= MAX_SEGMENTS).then_some(count)
}

/// A recipe's segments and the rest of its bytes, the inline data.
fn parse_table(bytes: &[u8]) -> Option<(Vec<Segment>, &[u8])> {
    let count = segment_count(bytes)?;
    let mut offset = 12;
    let mut segments = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = *bytes.get(offset)?;
        let length = usize::try_from(u64::from_le_bytes(
            bytes.get(offset + 1..offset + 9)?.try_into().ok()?,
        ))
        .ok()
        .filter(|length| *length <= HEADER_LIMIT)?;
        offset += 9;
        segments.push(match kind {
            INLINE => Segment::Inline(length),
            CHUNK => {
                let digest = bytes
                    .get(offset..offset + 32)?
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect();
                offset += 32;
                Segment::Chunk { digest, length }
            }
            _ => return None,
        });
    }
    Some((segments, bytes.get(offset..)?))
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    /// A header named `name` with a token list of `tokens` entries (about 20 bytes each).
    fn header(name: &str, tokens: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut bytes, name);
        string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&(tokens as u64).to_le_bytes());
        for token in 0..tokens {
            string(&mut bytes, &format!("token-{token:06}"));
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes
    }

    fn stored_bytes(directory: &Path) -> u64 {
        ["blobs/gguf-headers", "blobs/gguf-header-chunks"]
            .iter()
            .flat_map(|relative| files(&directory.join(relative)))
            .map(|(_, _, size, _)| size)
            .sum()
    }

    fn age(cache: &HeaderCache, digest: &str, seconds: u64) {
        let file = fs::File::options()
            .write(true)
            .open(cache.recipes.join(digest))
            .unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(seconds))
            .unwrap();
    }

    #[test]
    fn headers_sharing_a_tokenizer_store_it_once() {
        let directory = tempfile::tempdir().unwrap();
        let cache = HeaderCache::new(
            directory.path(),
            HEADER_CAPACITY_BYTES,
            HEADER_EVICTION_GRACE,
        );
        let headers = (0..10)
            .map(|index| header(&format!("Model Q{index}"), 20_000))
            .collect::<Vec<_>>();
        let digests = headers
            .iter()
            .map(|header| cache.write(header, BTreeSet::new))
            .collect::<Vec<_>>();
        for (header, digest) in headers.iter().zip(&digests) {
            assert_eq!(cache.read(digest).as_ref(), Some(header));
            assert!(cache.contains(digest));
        }
        assert_eq!(
            files(&directory.path().join("blobs/gguf-header-chunks")).len(),
            1
        );
        let whole = headers.iter().map(Vec::len).sum::<usize>() as u64;
        let stored = stored_bytes(directory.path());
        assert!(
            stored < headers[0].len() as u64 + 10 * 1024,
            "{stored} bytes stored for {whole}"
        );
        let small = header("tiny", 4);
        let digest = cache.write(&small, BTreeSet::new);
        assert_eq!(
            cache.read(&digest),
            Some(small),
            "a header with no large entry is stored inline"
        );
    }

    #[test]
    fn eviction_keeps_the_cache_within_capacity_by_recency_and_spares_pinned_headers() {
        let directory = tempfile::tempdir().unwrap();
        let one = header("Model A", 20_000).len() as u64;
        let writer = HeaderCache::new(directory.path(), HEADER_CAPACITY_BYTES, Duration::ZERO);
        let cache = HeaderCache::new(directory.path(), 3 * one + one / 2, Duration::ZERO);
        let headers = (0..6)
            .map(|index| header(&format!("Model {index}"), 20_000 + index))
            .collect::<Vec<_>>();
        let digests = headers
            .iter()
            .enumerate()
            .map(|(index, header)| {
                let digest = writer.write(header, BTreeSet::new);
                age(&cache, &digest, 1000 - index as u64 * 100);
                digest
            })
            .collect::<Vec<_>>();
        // The oldest header is pinned and the second oldest was read last.
        assert!(cache.read(&digests[1]).is_some());
        let pinned = BTreeSet::from([digests[0].clone()]);
        cache.sweep(&pinned);
        assert!(stored_bytes(directory.path()) <= cache.capacity + one);
        assert!(
            cache.contains(&digests[0]),
            "a pinned header is never evicted"
        );
        assert!(cache.contains(&digests[1]), "a header read recently stays");
        assert!(cache.contains(&digests[5]), "the newest header stays");
        assert!(!cache.contains(&digests[2]) && !cache.contains(&digests[3]));
        assert!(
            cache.read(&digests[2]).is_none(),
            "an evicted header is a miss"
        );
        assert_eq!(
            files(&directory.path().join("blobs/gguf-header-chunks")).len(),
            files(&directory.path().join("blobs/gguf-headers")).len(),
            "chunks of evicted headers go with them"
        );
        let rewritten = cache.write(&headers[2], || pinned.clone());
        assert_eq!(
            cache.read(&rewritten).as_ref(),
            Some(&headers[2]),
            "an evicted header is stored again"
        );
    }

    #[test]
    fn a_damaged_or_foreign_entry_is_a_miss_and_is_swept() {
        let directory = tempfile::tempdir().unwrap();
        let cache = HeaderCache::new(directory.path(), HEADER_CAPACITY_BYTES, Duration::ZERO);
        let first = header("Model A", 20_000);
        let digest = cache.write(&first, BTreeSet::new);
        let chunk = files(&cache.chunks).pop().unwrap();
        fs::write(&chunk.1, b"damaged").unwrap();
        assert!(cache.read(&digest).is_none());
        assert!(!chunk.1.exists(), "a damaged chunk is removed");
        assert_eq!(cache.write(&first, BTreeSet::new), digest);
        assert_eq!(cache.read(&digest), Some(first.clone()), "and stored again");
        let legacy = cache.recipes.join("0".repeat(64));
        fs::write(&legacy, &first).unwrap();
        assert!(cache.read(&"0".repeat(64)).is_none());
        cache.sweep(&BTreeSet::new());
        assert!(!legacy.exists(), "a raw or unparsable entry is removed");
    }
}
