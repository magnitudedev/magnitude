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
//! evicted, except those pinned (installed packages and the release catalog's headers) by any
//! cache of the store that lives, in whichever process and of whichever release, and those used
//! within a grace period (an operation in flight). Chunks no recipe
//! references any more go with them. Every read verifies the recipe against its own sha256, and a
//! whole header's read verifies the assembled bytes against the header's, so an evicted, partially
//! evicted or damaged header is a miss that its caller re-fetches.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use seismic_io::{Durability, Eviction, Examining, Layout, Store};

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

/// Where the cache's files lie under the store's root: each header's recipe by its sha256, each
/// shared payload's chunk by its own, and each large entry's fixed part's list of the headers
/// holding it.
const RECIPES: &str = "blobs/gguf-headers";
const CHUNKS: &str = "blobs/gguf-header-chunks";
const HEADS: &str = "blobs/gguf-header-heads";
/// Each live cache's pins, one file of digests per cache, under the cache's own lease
/// ([`HeaderCache::all_pinned`]).
const PINS: &str = "blobs/gguf-header-pins";
/// A pin file is read to this: a digest's line is 65 bytes.
const PINS_BYTES: u64 = 16 * 1024 * 1024;
/// The kernel cache under the model cache's root (`ModelCache::kernel_directory`): a store of
/// its own, which eviction leaves apart.
const KERNELS: &str = "kernels";

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
const HEADS_INDEX_BYTES: u64 = 4 * 1024;

const RECIPE_MAGIC: &[u8; 4] = b"MGHR";
const RECIPE_VERSION: u32 = 2;
/// Magic, version, the sha256 of the rest of the recipe and the segment count.
const RECIPE_PREAMBLE: usize = 4 + 4 + 32 + 4;
const INLINE: u8 = 0;
const CHUNK: u8 = 1;
const MAX_SEGMENTS: usize = 4096;

/// The header cache, in the model cache's store ([`seismic_io::Store`]), which every process
/// naming the cache root shares.
///
/// Recipes and chunks are content addresses: a recipe is the one encoding of its header, a chunk
/// its payload's bytes, so every writer of one writes what every other does, without a lock. A
/// header is read, whole or as its skeleton, under the store's lock held shared, so no eviction
/// removes a chunk between its recipe and its bytes; eviction holds it exclusively, from its
/// listing to its last removal, in whichever process it runs, so none evicts what another is
/// reading. What a skeleton's absent payloads are read from later is a chunk eviction may have
/// removed since: that read fails, and its caller discards the header and fetches it again. A
/// head's list of headers changes under its record's lock, from what it holds then, so no
/// header's reference is lost to another writer's.
#[derive(Clone, Debug)]
pub(crate) struct HeaderCache {
    store: Arc<Store>,
    flights: Arc<Mutex<BTreeMap<String, tokio::sync::watch::Receiver<bool>>>>,
    capacity: u64,
    grace: Duration,
    /// The bytes of headers this process stored since its cache last evicted: the store's own
    /// count ([`Store::written`]) holds every index of the root too.
    written: Arc<AtomicU64>,
    /// Whether this process's cache has evicted once.
    evicted: Arc<AtomicBool>,
    /// This cache's pins as the store holds them, from the first [`HeaderCache::pin`].
    pins: Arc<Mutex<Option<Pins>>>,
}

/// A cache's pins in the store: its lease among the caches that pin ([`Store::lease`]), whose
/// file names its digests, which it holds for as long as it lives, in whichever process: another's
/// eviction keeps the headers it names while the lease is held.
#[derive(Debug)]
struct Pins {
    lease: seismic_io::Lease,
    published: Option<BTreeSet<String>>,
}

#[derive(Debug, PartialEq, Eq)]
enum Segment {
    Inline(usize),
    Chunk { digest: [u8; 32], length: usize },
}

impl HeaderCache {
    /// The cache in `store`, bounded by `capacity` bytes of recipes and the chunks they reference,
    /// a header used within `grace` never evicted.
    pub(crate) fn new(store: Arc<Store>, capacity: u64, grace: Duration) -> Self {
        Self {
            store,
            flights: Arc::new(Mutex::new(BTreeMap::new())),
            capacity,
            grace,
            written: Arc::new(AtomicU64::new(0)),
            evicted: Arc::new(AtomicBool::new(false)),
            pins: Arc::new(Mutex::new(None)),
        }
    }

    fn recipe(digest: &str) -> PathBuf {
        Path::new(RECIPES).join(digest)
    }

    fn chunk(digest: &[u8; 32]) -> PathBuf {
        Path::new(CHUNKS).join(hex(digest))
    }

    /// The header with this sha256, marked used, or `None` when it isn't (completely) stored.
    pub(crate) fn read(&self, digest: &str) -> Option<Vec<u8>> {
        if !valid_digest(digest) {
            return None;
        }
        let recipe = {
            let shared = self.store.shared().ok()?;
            let recipe = shared.read(&Self::recipe(digest), HEADER_LIMIT as u64).ok().flatten()?;
            match self
                .assemble(&shared, &recipe)
                .filter(|header| hex_sha256(header) == digest)
            {
                Some(header) => return Some(header),
                None => recipe,
            }
        };
        self.discard(digest, &recipe);
        None
    }

    /// The skeleton of the header with this sha256 (every chunk's payload absent), marked used,
    /// or `None` when it isn't stored with every chunk it names. Only its recipe is read, verified
    /// by its own sha256; a chunk is only checked to be there, of its length.
    pub(crate) fn read_skeleton(&self, digest: &str) -> Option<Skeleton> {
        if !valid_digest(digest) {
            return None;
        }
        let recipe = {
            let shared = self.store.shared().ok()?;
            let recipe = shared.read(&Self::recipe(digest), HEADER_LIMIT as u64).ok().flatten()?;
            match skeleton(&self.store.root().join(CHUNKS), &recipe) {
                Some(skeleton) => return Some(skeleton),
                None => recipe,
            }
        };
        self.discard(digest, &recipe);
        None
    }

    /// The bytes a recipe assembles, when it parses and every chunk it names is whole, read under
    /// the store's lock held shared.
    fn assemble(&self, shared: &seismic_io::Shared<'_>, recipe: &[u8]) -> Option<Vec<u8>> {
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
                    let chunk = shared.read(&Self::chunk(&digest), length as u64).ok().flatten()?;
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

    /// Removes the recipe of `digest` that read as `recipe` and didn't assemble its header, and
    /// every chunk it names whose bytes aren't their name's, so the next write stores them again;
    /// under the store's lock held exclusively, so what it judges is what it removes: a recipe
    /// another writer stored again since is left.
    fn discard(&self, digest: &str, recipe: &[u8]) {
        let Ok(exclusive) = self.store.exclusive() else {
            return;
        };
        let path = Self::recipe(digest);
        if exclusive.examine(&path, HEADER_LIMIT as u64).ok().flatten().as_deref() == Some(recipe) {
            let _ = exclusive.remove(&path);
        }
        for segment in parse_table(recipe)
            .map(|(segments, _)| segments)
            .unwrap_or_default()
        {
            if let Segment::Chunk { digest, length } = segment {
                let chunk = Self::chunk(&digest);
                if exclusive
                    .examine(&chunk, length as u64)
                    .ok()
                    .flatten()
                    .is_none_or(|bytes| bytes.len() != length || sha256(&bytes) != digest)
                {
                    let _ = exclusive.remove(&chunk);
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
        if let Some(recipe) = self.store.read(&Self::recipe(digest), HEADER_LIMIT as u64).ok().flatten() {
            self.discard(digest, &recipe);
        }
    }

    /// Whether the header with this sha256 is stored with every chunk it needs, marking it used
    /// for the read that follows; `read` still verifies it.
    pub(crate) fn contains(&self, digest: &str) -> bool {
        if !valid_digest(digest) {
            return false;
        }
        let Ok(shared) = self.store.shared() else {
            return false;
        };
        let root = self.store.root();
        shared
            .read(&Self::recipe(digest), HEADER_LIMIT as u64)
            .ok()
            .flatten()
            .and_then(|recipe| recipe_chunks(&recipe))
            .is_some_and(|chunks| chunks.iter().all(|chunk| root.join(chunk).is_file()))
    }

    /// Stores `header`, returning its sha256, and evicts the least recently used headers no live
    /// cache pins once the cache has outgrown its capacity: at this process's first write, and
    /// every eighth of its capacity it writes after.
    pub(crate) fn write(&self, header: &[u8]) -> String {
        let digest = hex_sha256(header);
        if header.len() > HEADER_LIMIT {
            return digest;
        }
        if !self.contains(&digest) {
            let walked = walk(header);
            if let Ok(Walk::Complete(layout)) = &walked {
                self.index_entries(header, &layout.entries, &digest);
            }
            // The chunks and the recipe naming them are written under the store's lock held
            // shared, so no eviction falls between a chunk found there and the recipe naming it.
            let Ok(writing) = self.store.shared() else {
                return digest;
            };
            let mut segments = Vec::new();
            let mut inline = Vec::new();
            let mut written = 0_u64;
            let mut cursor = 0;
            if let Ok(Walk::Complete(layout)) = &walked {
                for entry in layout.entries.iter().filter(|entry| shared(entry)) {
                    let range = entry.head().end..entry.end;
                    if range.start > cursor {
                        segments.push(Segment::Inline(range.start - cursor));
                        inline.extend_from_slice(&header[cursor..range.start]);
                    }
                    let bytes = &header[range];
                    let chunk = sha256(bytes);
                    // Every read verifies a chunk against its name, so a torn one is a miss.
                    if writing.put_if_absent(&Self::chunk(&chunk), bytes, Durability::Volatile).unwrap_or(false) {
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
            // Replacing what is there, which did not hold the header (a recipe of another
            // version, or one whose chunks are gone): every writer writes this one encoding.
            let recipe = encode_recipe(&segments, &inline);
            if writing
                .publish(&Self::recipe(&digest), Durability::Volatile, |file| std::io::Write::write_all(file, &recipe))
                .is_ok()
            {
                written += recipe.len() as u64;
            }
            self.written.fetch_add(written, Ordering::Relaxed);
        }
        if !self.evicted.load(Ordering::Relaxed) || self.written.load(Ordering::Relaxed) >= self.capacity / 8 {
            self.evict();
        }
        digest
    }

    /// Headers holding a large entry whose fixed part (key, type and element count) is `head`, from
    /// any repository: references for reading another file's header without fetching that entry
    /// (see `remote_header`). Each is verified against its digest; an evicted one is skipped.
    pub(crate) fn sharing(&self, head: &[u8]) -> Vec<Vec<u8>> {
        self.sharing_by_key(&hex_sha256(head))
    }

    fn sharing_by_key(&self, key: &str) -> Vec<Vec<u8>> {
        let Some(list) = self
            .store
            .read(&Path::new(HEADS).join(key), HEADS_INDEX_BYTES)
            .ok()
            .flatten()
        else {
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
            let cache = self.clone();
            let lookup_key = key.clone();
            let lent = magnitude_service_utils::blocking::run(move || {
                cache.sharing_by_key(&lookup_key)
            })
            .await;
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

    /// Lists `digest` first among the headers holding the large entry whose fixed part is `head`,
    /// its record's lock held from the read to the write, so another writer's reference is never
    /// lost to this one.
    fn index_head(&self, head: &[u8], digest: &str) {
        let Ok(record) = self.store.lock(&Path::new(HEADS).join(hex_sha256(head))) else {
            return;
        };
        let list = record.read(HEADS_INDEX_BYTES).ok().flatten().unwrap_or_default();
        let listed = String::from_utf8_lossy(&list).into_owned();
        if listed.lines().any(|line| line == digest) {
            return;
        }
        let kept = std::iter::once(digest)
            .chain(listed.lines().filter(|line| valid_digest(line)))
            .take(MAX_HEAD_REFERENCES)
            .collect::<Vec<_>>()
            .join("\n");
        // A torn list holds lines that are no header's, which every reader skips.
        let _ = record.write(kept.as_bytes(), Durability::Volatile);
    }

    /// Evicts least recently used headers until the cache fits its capacity, keeping what every
    /// live cache pins ([`HeaderCache::all_pinned`]), recently used ones, and the chunks the
    /// headers kept reference ([`Headers`]). An eviction that finds the store held passes, and the
    /// next write evicts again.
    fn evict(&self) {
        let pinned = self.all_pinned();
        let layout = Headers { pinned: &pinned };
        match self.store.evict(self.capacity, self.grace, &layout) {
            Ok(true) => {
                self.evicted.store(true, Ordering::Relaxed);
                self.written.store(0, Ordering::Relaxed);
            }
            Ok(false) => {}
            Err(error) => tracing::warn!("header cache {}: eviction failed: {error}", self.store.root().display()),
        }
    }

    /// Pin `pinned` (the headers the installed packages and the release catalog need): published
    /// in the store as this cache's pins, under its lease, so every cache's eviction keeps them,
    /// in whichever process, while this cache lives. Its owner pins again whenever they change.
    pub(crate) fn pin(&self, pinned: BTreeSet<String>) {
        let mut own = self.pins.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if own.is_none() {
            match self.store.lease(Path::new(PINS)) {
                Ok(lease) => *own = Some(Pins { lease, published: None }),
                Err(error) => tracing::warn!("header cache {}: cannot hold its pins: {error}", self.store.root().display()),
            }
        }
        if let Some(pins) = own.as_mut().filter(|pins| pins.published.as_ref() != Some(&pinned)) {
            let lines = pinned.iter().map(|digest| format!("{digest}\n")).collect::<String>();
            match pins.lease.write(&self.store, lines.as_bytes(), Durability::Volatile) {
                Ok(()) => pins.published = Some(pinned),
                Err(error) => tracing::warn!("header cache {}: cannot publish its pins: {error}", self.store.root().display()),
            }
        }
    }

    /// What every live cache of the store pins, this one among them, in whichever process: each
    /// cache's published pins, which hold while its lease does; a gone cache's go
    /// ([`Store::leases`]).
    fn all_pinned(&self) -> BTreeSet<String> {
        let mut all = BTreeSet::new();
        let live = self.store.leases(Path::new(PINS)).unwrap_or_else(|error| {
            tracing::warn!("header cache {}: cannot list the pins: {error}", self.store.root().display());
            Vec::new()
        });
        for record in live {
            if let Ok(Some(bytes)) = self.store.read(&record, PINS_BYTES) {
                all.extend(String::from_utf8_lossy(&bytes).lines().filter(|line| valid_digest(line)).map(str::to_owned));
            }
        }
        all
    }
}

/// How the header cache's eviction places the files of the model cache's store: each recipe an
/// entry of its own, pinned where the caller pins it, referencing the chunks its table names,
/// which entries share; a file in the recipes that names no header, unreachable. A recipe whose
/// table does not parse (a damaged one, or one of another recipe version a build sharing the cache
/// writes) references nothing and is evicted as any other, least recently used first. The kernel
/// cache under the root is a store of its own, apart; everything else under the root (the head
/// lists, the indexes, the workspaces) is kept.
struct Headers<'a> {
    pinned: &'a BTreeSet<String>,
}

impl Layout for Headers<'_> {
    fn place(&self, path: &Path, _: &Examining<'_>) -> Eviction {
        if path == Path::new(KERNELS) {
            return Eviction::Apart;
        }
        if path == Path::new("blobs") || path == Path::new(RECIPES) || path == Path::new(CHUNKS) {
            // Walked, for what they hold.
            return Eviction::Alone;
        }
        let directory = path.parent().unwrap_or(Path::new(""));
        if directory == Path::new(CHUNKS) {
            return Eviction::Shared;
        }
        if directory != Path::new(RECIPES) {
            return Eviction::Kept;
        }
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        match (valid_digest(name), self.pinned.contains(name)) {
            (false, _) => Eviction::Unreachable,
            (true, true) => Eviction::Pinned,
            (true, false) => Eviction::Alone,
        }
    }

    /// A recipe's chunks; no other file references any (a kept file beside them, as the local
    /// models' configuration at the root, is never read).
    fn references(&self, path: &Path, examining: &Examining<'_>) -> Vec<PathBuf> {
        if path.parent() != Some(Path::new(RECIPES)) {
            return Vec::new();
        }
        examining
            .examine(path, HEADER_LIMIT as u64)
            .ok()
            .flatten()
            .and_then(|recipe| recipe_chunks(&recipe))
            .unwrap_or_default()
    }
}

/// The chunks a recipe references, under the store's root, read from its segment table alone.
fn recipe_chunks(recipe: &[u8]) -> Option<Vec<PathBuf>> {
    let (segments, _) = parse_table(recipe)?;
    Some(
        segments
            .into_iter()
            .filter_map(|segment| match segment {
                Segment::Chunk { digest, .. } => Some(HeaderCache::chunk(&digest)),
                Segment::Inline(_) => None,
            })
            .collect(),
    )
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
    use seismic_io::Privacy;
    use std::time::SystemTime;

    /// A header cache in a store of its own over `directory`, as another process opens it.
    fn cache(directory: &Path, capacity: u64, grace: Duration) -> HeaderCache {
        HeaderCache::new(Arc::new(Store::open(directory.to_path_buf(), Privacy::Owner).unwrap()), capacity, grace)
    }

    /// The regular files of `directory`: name, path, size and modification time.
    fn files(directory: &Path) -> Vec<(String, PathBuf, u64, SystemTime)> {
        let Ok(read) = fs::read_dir(directory) else {
            return Vec::new();
        };
        read.filter_map(Result::ok)
            .filter_map(|entry| {
                let metadata = entry.metadata().ok()?;
                metadata.is_file().then(|| {
                    (entry.file_name().into_string().unwrap(), entry.path(), metadata.len(), metadata.modified().unwrap())
                })
            })
            .collect()
    }

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
            .open(cache.store.root().join(RECIPES).join(digest))
            .unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(seconds))
            .unwrap();
    }

    #[test]
    fn headers_sharing_a_tokenizer_store_it_once() {
        let directory = tempfile::tempdir().unwrap();
        let cache = cache(directory.path(), HEADER_CAPACITY_BYTES, HEADER_EVICTION_GRACE);
        let headers = (0..10)
            .map(|index| header(&format!("Model Q{index}"), 20_000))
            .collect::<Vec<_>>();
        let digests = headers
            .iter()
            .map(|header| cache.write(header))
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
        let digest = cache.write(&small);
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
        let cache = cache(directory.path(), HEADER_CAPACITY_BYTES, HEADER_EVICTION_GRACE);
        let header = header("Model", 20_000);
        let digest = cache.write(&header);
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

        let recipe = directory.path().join(RECIPES).join(&digest);
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
        let cache = cache(directory.path(), HEADER_CAPACITY_BYTES, HEADER_EVICTION_GRACE);
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
        let first = cache.write(&header("Model A", 20_001));
        let second = cache.write(&header("Model B", 20_001));
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
        let cache = cache(directory.path(), HEADER_CAPACITY_BYTES, HEADER_EVICTION_GRACE);
        let header = header("Model", 20_002);
        let digest = cache.write(&header);
        let chunk = cache.read_skeleton(&digest).unwrap().absent[0].file.clone();
        let mut bytes = fs::read(&chunk).unwrap();
        bytes.pop();
        fs::write(&chunk, &bytes).unwrap();
        assert!(
            cache.read_skeleton(&digest).is_none(),
            "a chunk short of its length"
        );
        assert!(!cache.contains(&digest), "and the entry is discarded");
        assert_eq!(cache.write(&header), digest);
        fs::remove_file(&chunk).unwrap();
        assert!(cache.read_skeleton(&digest).is_none(), "a missing chunk");
        assert!(!cache.contains(&digest));
    }

    #[test]
    fn a_recipe_of_the_previous_version_is_a_miss_and_is_stored_again() {
        let directory = tempfile::tempdir().unwrap();
        let cache = cache(directory.path(), HEADER_CAPACITY_BYTES, Duration::ZERO);
        let header = header("Model", 20_000);
        let digest = cache.write(&header);
        let recipe = directory.path().join(RECIPES).join(&digest);
        let mut previous = fs::read(&recipe).unwrap();
        previous[4..8].copy_from_slice(&(RECIPE_VERSION - 1).to_le_bytes());
        fs::write(&recipe, previous).unwrap();
        assert!(!cache.contains(&digest));
        assert!(cache.read_skeleton(&digest).is_none());
        assert!(cache.read(&digest).is_none());
        assert_eq!(cache.write(&header), digest);
        assert_eq!(cache.read(&digest), Some(header));
    }

    /// A header another live cache pins (another process, of another catalog) stays through this
    /// cache's eviction, which pins nothing, from the moment that cache pins it, before it ever
    /// evicts; once that cache is gone, its pins go and so does the header.
    #[test]
    fn a_header_another_live_cache_pins_stays_until_that_cache_is_gone() {
        let directory = tempfile::tempdir().unwrap();
        let writer = cache(directory.path(), HEADER_CAPACITY_BYTES, Duration::ZERO);
        let digest = writer.write(&header("Model A", 20_000));
        drop(writer);
        let service = cache(directory.path(), 0, Duration::ZERO);
        service.pin(BTreeSet::from([digest.clone()]));
        let tool = cache(directory.path(), 0, Duration::ZERO);
        tool.evict();
        assert!(tool.contains(&digest), "another live cache's pin holds");
        drop(service);
        tool.evict();
        assert!(!tool.contains(&digest), "a gone cache's pins go with it");
        assert!(files(&directory.path().join(PINS)).is_empty(), "no live cache pins anything");
    }

    #[test]
    fn eviction_keeps_the_cache_within_capacity_by_recency_and_spares_pinned_headers() {
        let directory = tempfile::tempdir().unwrap();
        let one = header("Model A", 20_000).len() as u64;
        let writer = cache(directory.path(), HEADER_CAPACITY_BYTES, Duration::ZERO);
        let cache = cache(directory.path(), 3 * one + one / 2, Duration::ZERO);
        let headers = (0..6)
            .map(|index| header(&format!("Model {index}"), 20_000 + index))
            .collect::<Vec<_>>();
        let digests = headers
            .iter()
            .enumerate()
            .map(|(index, header)| {
                let digest = writer.write(header);
                age(&cache, &digest, 1000 - index as u64 * 100);
                digest
            })
            .collect::<Vec<_>>();
        // The oldest header is pinned and the second oldest was read last.
        assert!(cache.read(&digests[1]).is_some());
        cache.pin(BTreeSet::from([digests[0].clone()]));
        cache.evict();
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
        let rewritten = cache.write(&headers[2]);
        assert_eq!(
            cache.read(&rewritten).as_ref(),
            Some(&headers[2]),
            "an evicted header is stored again"
        );
    }

    #[test]
    fn a_damaged_or_foreign_entry_is_a_miss_and_is_swept() {
        let directory = tempfile::tempdir().unwrap();
        let cache = cache(directory.path(), HEADER_CAPACITY_BYTES, Duration::ZERO);
        let first = header("Model A", 20_000);
        let digest = cache.write(&first);
        let chunk = files(&directory.path().join(CHUNKS)).pop().unwrap();
        fs::write(&chunk.1, b"damaged").unwrap();
        assert!(cache.read(&digest).is_none());
        assert!(!chunk.1.exists(), "a damaged chunk is removed");
        assert_eq!(cache.write(&first), digest);
        assert_eq!(cache.read(&digest), Some(first.clone()), "and stored again");
        let legacy = directory.path().join(RECIPES).join("0".repeat(64));
        fs::write(&legacy, &first).unwrap();
        assert!(cache.read(&"0".repeat(64)).is_none());
        assert!(!legacy.exists(), "a raw or unparsable entry read is removed");
        let foreign = directory.path().join(RECIPES).join("not-a-digest");
        fs::write(&foreign, &first).unwrap();
        cache.evict();
        assert!(!foreign.exists(), "a file that names no header is swept");
    }

    /// Two processes listing headers of one large entry at once each keep the other's: the
    /// second's read waits, through the head's record lock, for the first's write.
    #[test]
    fn heads_listed_at_once_by_two_processes_keep_both() {
        let directory = tempfile::tempdir().unwrap();
        let first = cache(directory.path(), HEADER_CAPACITY_BYTES, HEADER_EVICTION_GRACE);
        let second = cache(directory.path(), HEADER_CAPACITY_BYTES, HEADER_EVICTION_GRACE);
        let head = b"tokenizer.ggml.tokens";
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        let record = Path::new(HEADS).join(hex_sha256(head));
        // The first has read the list (empty) and is about to write its own.
        let held = first.store.lock(&record).unwrap();
        assert_eq!(held.read(HEADS_INDEX_BYTES).unwrap(), None);
        std::thread::scope(|scope| {
            let listing = seismic_threads::spawn_scoped(scope, "listing", || second.index_head(head, &b)).unwrap();
            std::thread::sleep(Duration::from_millis(200));
            held.write(a.as_bytes(), Durability::Volatile).unwrap();
            drop(held);
            listing.join().unwrap();
        });
        let list = second.store.read(&record, HEADS_INDEX_BYTES).unwrap().unwrap();
        assert_eq!(String::from_utf8(list).unwrap(), format!("{b}\n{a}"), "the second lists itself before the first, which it read");
    }
}
