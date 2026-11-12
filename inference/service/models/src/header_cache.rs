//! The GGUF header cache: every header fetched by range request, by its sha256, within a size
//! bound.
//!
//! Headers of one repository's files share their tokenizer, nearly all of each header's bytes. A
//! header is therefore stored as a recipe of segments: the array payload of each large metadata
//! entry ([`shared`]) is a chunk file named by its own sha256 and stored once however many headers
//! hold it, and the rest of the header (each such entry's key, type and count among it) is inline
//! in the recipe, which carries the sha256 of its own bytes. A header's skeleton
//! ([`HeaderCache::read_skeleton`]), its directory without those payloads, is read from its recipe
//! alone. The cache is bounded by the bytes of its recipes and the chunks they reference. Reading a
//! header marks it used; once the cache outgrows its capacity the least recently used headers are
//! evicted, except those pinned by the caller (installed packages and the release catalog's
//! headers) and those used within a grace period (an operation in flight). Chunks no recipe
//! references any more go with them. Every read verifies the recipe against its own sha256, and a
//! whole header's read verifies the assembled bytes against the header's, so an evicted, partially
//! evicted or damaged header is a miss that its caller re-fetches.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use magnitude_service_utils::file_cache::{read_bytes, write_bytes_atomic, write_bytes_atomic_waiting};

use magnitude_artifacts::gguf::{Absence, Entry, HEADER_LIMIT, Skeleton, Walk, sha256, walk};

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

/// What another header's read can borrow for one large entry: headers holding it, or the claim to
/// fetch it.
pub(crate) enum Lending {
    Lent(Vec<Vec<u8>>),
    Claimed(Claim),
}

/// The one read fetching a large entry no stored header holds; dropping it, once that read has
/// stored its header, lets the reads waiting for the entry borrow it.
pub(crate) struct Claim {
    key: String,
    done: tokio::sync::watch::Sender<bool>,
    flights: Arc<Mutex<BTreeMap<String, tokio::sync::watch::Receiver<bool>>>>,
}

impl Drop for Claim {
    fn drop(&mut self) {
        if let Ok(mut flights) = self.flights.lock() {
            flights.remove(&self.key);
        }
        let _ = self.done.send(true);
    }
}

/// Headers listed, and lent, per large entry's fixed part: the most recently seen first.
const MAX_HEAD_REFERENCES: usize = 2;
const HEADS_INDEX_BYTES: usize = 4 * 1024;

const RECIPE_MAGIC: &[u8; 4] = b"MGHR";
const RECIPE_VERSION: u32 = 2;
/// Magic, version, the sha256 of the rest of the recipe and the segment count.
const RECIPE_PREAMBLE: usize = 4 + 4 + 32 + 4;
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
    heads: PathBuf,
    flights: Arc<Mutex<BTreeMap<String, tokio::sync::watch::Receiver<bool>>>>,
    lock: PathBuf,
    heads_lock: PathBuf,
    capacity: u64,
    grace: Duration,
    state: Arc<Mutex<SweepState>>,
}

#[derive(Debug, PartialEq, Eq)]
enum Segment {
    Inline(usize),
    Chunk { digest: [u8; 32], length: usize },
}

impl HeaderCache {
    pub(crate) fn new(root: &Path, capacity: u64, grace: Duration) -> Self {
        Self {
            recipes: root.join("blobs/gguf-headers"),
            chunks: root.join("blobs/gguf-header-chunks"),
            heads: root.join("blobs/gguf-header-heads"),
            flights: Arc::new(Mutex::new(BTreeMap::new())),
            lock: root.join(".locks/gguf-headers.lock"),
            heads_lock: root.join(".locks/gguf-header-heads.lock"),
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

    /// The skeleton of the header with this sha256 (every chunk's payload absent), marked used,
    /// or `None` when it isn't stored with every chunk it names. Only its recipe is read, verified
    /// by its own sha256; a chunk is only checked to be there, of its length.
    pub(crate) fn read_skeleton(&self, digest: &str) -> Option<Skeleton> {
        if !valid_digest(digest) {
            return None;
        }
        let path = self.recipes.join(digest);
        let recipe = read_bytes(&path, HEADER_LIMIT)?;
        match skeleton(&self.chunks, &recipe) {
            Some(skeleton) => {
                touch(&path);
                Some(skeleton)
            }
            None => {
                self.discard(&path, &recipe);
                None
            }
        }
    }

    /// The bytes a recipe assembles, when it parses and every chunk it names is whole.
    fn assemble(&self, recipe: &[u8]) -> Option<Vec<u8>> {
        let (segments, mut inline) = parse_recipe(recipe)?;
        let mut header = Vec::new();
        for segment in segments {
            match segment {
                Segment::Inline(length) => {
                    let (bytes, rest) = inline.split_at_checked(length)?;
                    header.extend_from_slice(bytes);
                    inline = rest;
                }
                Segment::Chunk { digest, length } => {
                    let chunk = read_bytes(&self.chunks.join(hex(&digest)), length)?;
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
                let chunk = self.chunks.join(hex(&digest));
                if read_bytes(&chunk, length).is_none_or(|bytes| sha256(&bytes) != digest) {
                    let _ = fs::remove_file(chunk);
                }
            }
        }
    }

    /// Discards the header with this sha256 (a payload of its skeleton failed to read): its recipe,
    /// and every chunk it names whose bytes aren't their name's, so the next write stores them
    /// again.
    pub(crate) fn discard_header(&self, digest: &str) {
        if !valid_digest(digest) {
            return;
        }
        let path = self.recipes.join(digest);
        if let Some(recipe) = read_bytes(&path, HEADER_LIMIT) {
            self.discard(&path, &recipe);
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
                self.index_entries(header, &layout.entries, &digest);
                for entry in layout.entries.iter().filter(|entry| shared(entry)) {
                    let range = entry.head().end..entry.end;
                    if range.start > cursor {
                        segments.push(Segment::Inline(range.start - cursor));
                        inline.extend_from_slice(&header[cursor..range.start]);
                    }
                    let bytes = &header[range];
                    let chunk = sha256(bytes);
                    let path = self.chunks.join(hex(&chunk));
                    if fs::metadata(&path)
                        .is_ok_and(|metadata| metadata.len() == bytes.len() as u64)
                    {
                        touch(&path);
                    } else {
                        write_bytes_atomic_waiting(&path, &self.lock, bytes, HEADER_LIMIT);
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
            write_bytes_atomic_waiting(&recipe_path, &self.lock, &recipe, HEADER_LIMIT);
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

    /// Headers holding a large entry whose fixed part (key, type and element count) is `head`, from
    /// any repository: references for reading another file's header without fetching that entry
    /// (see `remote_header`). Each is verified against its digest; an evicted one is skipped.
    pub(crate) fn sharing(&self, head: &[u8]) -> Vec<Vec<u8>> {
        let Some(list) = read_bytes(&self.heads.join(hex_sha256(head)), HEADS_INDEX_BYTES) else {
            return Vec::new();
        };
        String::from_utf8_lossy(&list)
            .lines()
            .filter_map(|digest| self.read(digest))
            .take(MAX_HEAD_REFERENCES)
            .collect()
    }

    /// Headers holding a large entry whose fixed part is `head`, waiting for a read already
    /// fetching that entry to store its header; or, when none holds it and none is fetching it,
    /// the claim to fetch it, which tells the waiters when dropped (after the header is stored).
    pub(crate) async fn lend_or_claim(&self, head: &[u8]) -> Lending {
        let key = hex_sha256(head);
        loop {
            let lent = self.sharing(head);
            if !lent.is_empty() {
                return Lending::Lent(lent);
            }
            let waiting = {
                let Ok(mut flights) = self.flights.lock() else {
                    return Lending::Lent(Vec::new());
                };
                match flights.get(&key) {
                    Some(flight) => flight.clone(),
                    None => {
                        let (done, flight) = tokio::sync::watch::channel(false);
                        flights.insert(key.clone(), flight);
                        return Lending::Claimed(Claim {
                            key,
                            done,
                            flights: Arc::clone(&self.flights),
                        });
                    }
                }
            };
            let mut waiting = waiting;
            // A claim dropped without storing its header (a failed read) ends the wait too; the
            // loop then finds nothing and claims the entry itself.
            let _ = waiting.wait_for(|done| *done).await;
        }
    }

    /// Records, for each large entry of `header`, that this header holds it.
    fn index_entries(&self, header: &[u8], entries: &[Entry<'_>], digest: &str) {
        for entry in entries.iter().filter(|entry| shared(entry)) {
            self.index_head(&header[entry.head().start..entry.head().end], digest);
        }
    }

    fn index_head(&self, head: &[u8], digest: &str) {
        let path = self.heads.join(hex_sha256(head));
        let list = read_bytes(&path, HEADS_INDEX_BYTES).unwrap_or_default();
        let listed = String::from_utf8_lossy(&list).into_owned();
        if listed.lines().any(|line| line == digest) {
            return;
        }
        let kept = std::iter::once(digest)
            .chain(listed.lines().filter(|line| valid_digest(line)))
            .take(MAX_HEAD_REFERENCES)
            .collect::<Vec<_>>()
            .join("\n");
        write_bytes_atomic(&path, &self.heads_lock, kept.as_bytes(), HEADS_INDEX_BYTES);
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
        let mut preamble = [0_u8; RECIPE_PREAMBLE];
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
                    Segment::Chunk { digest, .. } => Some(hex(&digest)),
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

/// `MGHR`, the format version, the sha256 of the rest, the segment count, each segment's kind,
/// length and (for a chunk) sha256, then the inline bytes in order.
fn encode_recipe(segments: &[Segment], inline: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
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
                bytes.extend_from_slice(digest);
            }
        }
    }
    bytes.extend_from_slice(inline);
    [
        RECIPE_MAGIC.as_slice(),
        &RECIPE_VERSION.to_le_bytes(),
        sha256(&bytes).as_slice(),
        &bytes,
    ]
    .concat()
}

fn segment_count(preamble: &[u8]) -> Option<usize> {
    if preamble.get(..4)? != RECIPE_MAGIC
        || u32::from_le_bytes(preamble.get(4..8)?.try_into().ok()?) != RECIPE_VERSION
    {
        return None;
    }
    let count = u32::from_le_bytes(preamble.get(40..44)?.try_into().ok()?) as usize;
    (count <= MAX_SEGMENTS).then_some(count)
}

/// A whole recipe's segments and inline data, when its bytes are the ones its sha256 names.
fn parse_recipe(bytes: &[u8]) -> Option<(Vec<Segment>, &[u8])> {
    let (digest, rest) = bytes.get(8..)?.split_at_checked(32)?;
    if sha256(rest).as_slice() != digest {
        return None;
    }
    parse_table(bytes)
}

/// The skeleton a whole recipe describes: its inline bytes, with each chunk an absent payload
/// that its chunk file holds; `None` when a chunk file is missing or not its length (its bytes are
/// checked against its sha256 when they are read).
fn skeleton(chunks: &Path, recipe: &[u8]) -> Option<Skeleton> {
    let (segments, inline) = parse_recipe(recipe)?;
    let mut absent = Vec::new();
    // Where the next segment starts in the whole header, and the inline bytes before it.
    let (mut offset, mut held) = (0_usize, 0_usize);
    for segment in segments {
        let length = match segment {
            Segment::Inline(length) => {
                held += length;
                length
            }
            Segment::Chunk { digest, length } => {
                let file = chunks.join(hex(&digest));
                if !fs::metadata(&file)
                    .is_ok_and(|chunk| chunk.is_file() && chunk.len() == length as u64)
                {
                    return None;
                }
                absent.push(Absence {
                    offset,
                    length,
                    digest,
                    file,
                });
                length
            }
        };
        offset = offset
            .checked_add(length)
            .filter(|end| *end <= HEADER_LIMIT)?;
    }
    (held == inline.len()).then(|| Skeleton {
        bytes: inline.to_vec(),
        absent,
    })
}

/// A recipe's segments and the rest of its bytes, the inline data, read from its segment table
/// alone (its sha256 unchecked).
fn parse_table(bytes: &[u8]) -> Option<(Vec<Segment>, &[u8])> {
    let count = segment_count(bytes)?;
    let mut offset = RECIPE_PREAMBLE;
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
                let digest = bytes.get(offset..offset + 32)?.try_into().ok()?;
                offset += 32;
                Segment::Chunk { digest, length }
            }
            _ => return None,
        });
    }
    Some((segments, bytes.get(offset..)?))
}

fn hex_sha256(bytes: &[u8]) -> String {
    hex(&sha256(bytes))
}

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
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

    /// A header named `name` with a token list of `tokens` entries (about 20 bytes each), a small
    /// array after it and one tensor.
    fn header(name: &str, tokens: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
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
        string(&mut bytes, "test.layers");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        for layer in [1_u32, 2, 3] {
            bytes.extend_from_slice(&layer.to_le_bytes());
        }
        string(&mut bytes, "token_embd.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&64_u64.to_le_bytes());
        bytes.extend_from_slice(&(tokens as u64).to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
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
    fn a_skeleton_reads_as_the_whole_header_its_large_payloads_read_apart_only_when_needed() {
        use magnitude_artifacts::{
            Error,
            gguf::{Element, Value, read_header, read_skeleton},
        };
        let directory = tempfile::tempdir().unwrap();
        let cache = HeaderCache::new(
            directory.path(),
            HEADER_CAPACITY_BYTES,
            HEADER_EVICTION_GRACE,
        );
        let header = header("Model", 20_000);
        let digest = cache.write(&header, BTreeSet::new);
        let skeleton = cache.read_skeleton(&digest).unwrap();
        assert_eq!(skeleton.absent.len(), 1);
        assert!(
            skeleton.bytes.len() < 1024,
            "{} bytes",
            skeleton.bytes.len()
        );
        let chunk = skeleton.absent[0].file.clone();
        let whole = read_header(header).unwrap();
        let skeletal = read_skeleton(skeleton).unwrap();
        let tokens = |directory: &magnitude_artifacts::gguf::Directory| match directory
            .value("tokenizer.ggml.tokens")
        {
            Some(Value::Array(tokens)) => tokens.clone(),
            other => panic!("{other:?}"),
        };
        // Its digest and length come from the recipe: no chunk is read.
        fs::rename(&chunk, chunk.with_extension("moved")).unwrap();
        let (apart, held) = (tokens(&skeletal), tokens(&whole));
        assert_eq!(apart.len(), held.len());
        assert_eq!(apart.digest().unwrap(), held.digest().unwrap());
        assert!(matches!(
            apart.element(7),
            Err(Error::PayloadRead { key, .. }) if key == "tokenizer.ggml.tokens"
        ));
        fs::rename(chunk.with_extension("moved"), &chunk).unwrap();
        assert_eq!(
            apart.element(19_999).unwrap(),
            Some(Element::String("token-019999"))
        );
        assert_eq!(apart.element(20_000).unwrap(), None);
        // The whole directory, its apart payload read from its chunk.
        assert_eq!(skeletal, whole);

        let recipe = cache.recipes.join(&digest);
        let mut damaged = fs::read(&recipe).unwrap();
        *damaged.last_mut().unwrap() ^= 1;
        fs::write(&recipe, damaged).unwrap();
        assert!(
            cache.read_skeleton(&digest).is_none(),
            "a damaged recipe is a miss"
        );
        assert!(!recipe.exists(), "and is removed");
    }

    #[test]
    fn a_verified_payload_is_shared_by_its_digest_only_while_an_array_holds_it() {
        use magnitude_artifacts::{
            Error,
            gguf::{Array, Element, Skeleton, Value, read_skeleton},
        };
        let directory = tempfile::tempdir().unwrap();
        let cache = HeaderCache::new(
            directory.path(),
            HEADER_CAPACITY_BYTES,
            HEADER_EVICTION_GRACE,
        );
        let tokens = |skeleton: Skeleton| -> Array {
            match read_skeleton(skeleton)
                .unwrap()
                .value("tokenizer.ggml.tokens")
            {
                Some(Value::Array(tokens)) => tokens.clone(),
                other => panic!("{other:?}"),
            }
        };
        // A tokenizer no other test's header has, so only this test holds its payload.
        let first = cache.write(&header("Model A", 20_001), BTreeSet::new);
        let second = cache.write(&header("Model B", 20_001), BTreeSet::new);
        let first = tokens(cache.read_skeleton(&first).unwrap());
        let skeleton = cache.read_skeleton(&second).unwrap();
        let chunk = skeleton.absent[0].file.clone();
        let shared = tokens(skeleton);
        let last = Some(Element::String("token-020000"));
        assert_eq!(first.element(20_000).unwrap(), last);
        fs::rename(&chunk, chunk.with_extension("moved")).unwrap();
        assert_eq!(
            shared.element(20_000).unwrap(),
            last,
            "another header's array reads the bytes the first verified"
        );
        drop((first, shared));
        // A chunk of its length with other bytes passes the skeleton's presence check, so
        // only a read that verifies it again can refuse it.
        let length = fs::metadata(chunk.with_extension("moved")).unwrap().len();
        fs::write(&chunk, vec![0_u8; usize::try_from(length).unwrap()]).unwrap();
        assert!(
            matches!(
                tokens(cache.read_skeleton(&second).unwrap()).element(0),
                Err(Error::DamagedPayload { .. })
            ),
            "once no array holds them, they are read and verified again"
        );
    }

    #[test]
    fn a_skeleton_whose_chunk_is_missing_or_not_its_length_is_a_miss() {
        let directory = tempfile::tempdir().unwrap();
        let cache = HeaderCache::new(
            directory.path(),
            HEADER_CAPACITY_BYTES,
            HEADER_EVICTION_GRACE,
        );
        let header = header("Model", 20_002);
        let digest = cache.write(&header, BTreeSet::new);
        let chunk = cache.read_skeleton(&digest).unwrap().absent[0].file.clone();
        let mut bytes = fs::read(&chunk).unwrap();
        bytes.pop();
        fs::write(&chunk, &bytes).unwrap();
        assert!(
            cache.read_skeleton(&digest).is_none(),
            "a chunk short of its length"
        );
        assert!(!cache.contains(&digest), "and the entry is discarded");
        assert_eq!(cache.write(&header, BTreeSet::new), digest);
        fs::remove_file(&chunk).unwrap();
        assert!(cache.read_skeleton(&digest).is_none(), "a missing chunk");
        assert!(!cache.contains(&digest));
    }

    #[test]
    fn a_recipe_of_the_previous_version_is_a_miss_and_is_stored_again() {
        let directory = tempfile::tempdir().unwrap();
        let cache = HeaderCache::new(directory.path(), HEADER_CAPACITY_BYTES, Duration::ZERO);
        let header = header("Model", 20_000);
        let digest = cache.write(&header, BTreeSet::new);
        let recipe = cache.recipes.join(&digest);
        let mut previous = fs::read(&recipe).unwrap();
        previous[4..8].copy_from_slice(&(RECIPE_VERSION - 1).to_le_bytes());
        fs::write(&recipe, previous).unwrap();
        assert!(!cache.contains(&digest));
        assert!(cache.read_skeleton(&digest).is_none());
        assert!(cache.read(&digest).is_none());
        assert_eq!(cache.write(&header, BTreeSet::new), digest);
        assert_eq!(cache.read(&digest), Some(header));
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
