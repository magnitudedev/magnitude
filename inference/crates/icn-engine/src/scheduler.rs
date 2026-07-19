use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use icn_contracts::RadixCacheConfig;
use llama_cpp_2::context::kv_cache::{KvPageError, LlamaKvPageId};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{LlamaSequenceState, context::LlamaContext};
use sha2::{Digest, Sha256};

use crate::radix_cache::{PagedRadixCache, RadixNodeId};

#[derive(Clone, Debug)]
struct DiskKvPage {
    path: PathBuf,
    bytes: u64,
    sha256: [u8; 32],
}

#[derive(Clone, Debug)]
enum KvPageResidency {
    Device(LlamaKvPageId),
    Host(Arc<[u8]>),
    Disk(DiskKvPage),
}

#[derive(Debug)]
struct RadixTierState {
    host_limit: u64,
    disk_limit: u64,
    host_bytes: u64,
    disk_bytes: u64,
    disk_scope: Option<PathBuf>,
    next_file: u64,
}

impl RadixTierState {
    fn new(config: &RadixCacheConfig) -> Self {
        let disk_scope = (config.disk_bytes > 0).then(|| {
            let root = config.disk_path.clone().unwrap_or_else(std::env::temp_dir);
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            root.join(format!("magnitude-icn-kv-{}-{nonce}", std::process::id()))
        });
        Self {
            host_limit: config.host_bytes,
            disk_limit: config.disk_bytes,
            host_bytes: 0,
            disk_bytes: 0,
            disk_scope,
            next_file: 0,
        }
    }
}

impl Drop for RadixTierState {
    fn drop(&mut self) {
        if let Some(scope) = &self.disk_scope {
            let _ = fs::remove_dir_all(scope);
        }
    }
}

#[derive(Debug)]
pub(crate) enum RadixCacheError {
    Native(KvPageError),
    Io(io::Error),
    CorruptDiskPage(PathBuf),
    Capacity,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct RadixAttach {
    pub(crate) cached_tokens: usize,
    pub(crate) device_tokens: usize,
    pub(crate) host_tokens: usize,
    pub(crate) disk_tokens: usize,
    pub(crate) promotion_ms: f64,
}

impl fmt::Display for RadixCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(error) => error.fmt(formatter),
            Self::Io(error) => write!(formatter, "KV cache tier I/O failed: {error}"),
            Self::CorruptDiskPage(path) => {
                write!(
                    formatter,
                    "KV cache disk page is corrupt: {}",
                    path.display()
                )
            }
            Self::Capacity => formatter.write_str("KV cache has no evictable device capacity"),
        }
    }
}

impl From<KvPageError> for RadixCacheError {
    fn from(value: KvPageError) -> Self {
        Self::Native(value)
    }
}

impl From<io::Error> for RadixCacheError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PromptCheckpoint {
    pub(crate) target: LlamaSequenceState,
    pub(crate) draft: Option<LlamaSequenceState>,
    pub(crate) prefix: usize,
}

#[derive(Debug)]
pub(crate) struct SequenceCache {
    pub(crate) prompt: Vec<LlamaToken>,
    pub(crate) checkpoints: Vec<PromptCheckpoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkKind {
    Decode,
    Prefill { remaining: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkCandidate {
    pub(crate) sequence_id: i32,
    pub(crate) kind: WorkKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchWork {
    Decode { sequence_id: i32 },
    Prefill { sequence_id: i32, tokens: usize },
}

/// Magnitude-owned, policy-light batch assembly.
///
/// Every runnable decode token is placed before prompt work. Prompt work is then split into
/// rotating round-robin quanta until the logical batch is full. This keeps token generation
/// responsive without importing llama-server's slot or cache policies, while still filling large
/// batches when enough prompt work exists.
#[derive(Debug)]
pub(crate) struct BatchPlanner {
    cursor: usize,
    prefill_quantum: usize,
}

impl BatchPlanner {
    pub(crate) fn new(prefill_quantum: usize) -> Self {
        assert!(prefill_quantum > 0);
        Self {
            cursor: 0,
            prefill_quantum,
        }
    }

    pub(crate) fn plan(&mut self, candidates: &[WorkCandidate], capacity: usize) -> Vec<BatchWork> {
        if candidates.is_empty() || capacity == 0 {
            return Vec::new();
        }

        let mut available = capacity;
        let mut result = Vec::new();

        // Decode-first is explicit: at most one next-token decode per sequence per iteration.
        // Keep resident sequence order stable across iterations, as llama-server does. Besides
        // making completion order predictable, this lets backends reuse an identical generation
        // graph instead of cycling sequence permutations when every decode already fits.
        let mut decode_candidates = candidates
            .iter()
            .filter(|candidate| candidate.kind == WorkKind::Decode)
            .collect::<Vec<_>>();
        decode_candidates.sort_unstable_by_key(|candidate| candidate.sequence_id);
        for candidate in decode_candidates {
            if available == 0 {
                break;
            }
            result.push(BatchWork::Decode {
                sequence_id: candidate.sequence_id,
            });
            available -= 1;
        }

        let prefill_candidates = candidates
            .iter()
            .filter_map(|candidate| match candidate.kind {
                WorkKind::Decode => None,
                WorkKind::Prefill { remaining } => Some((candidate.sequence_id, remaining)),
            })
            .collect::<Vec<_>>();
        if prefill_candidates.is_empty() {
            return result;
        }

        let start = self.cursor % prefill_candidates.len();
        let mut prefill = (0..prefill_candidates.len())
            .map(|offset| prefill_candidates[(start + offset) % prefill_candidates.len()])
            .collect::<Vec<_>>();
        self.cursor = (start + 1) % prefill_candidates.len();

        while available > 0 {
            let mut progressed = false;
            for (sequence_id, remaining) in &mut prefill {
                if available == 0 {
                    break;
                }
                let tokens = (*remaining).min(self.prefill_quantum).min(available);
                if tokens == 0 {
                    continue;
                }
                result.push(BatchWork::Prefill {
                    sequence_id: *sequence_id,
                    tokens,
                });
                *remaining -= tokens;
                available -= tokens;
                progressed = true;
            }
            if !progressed {
                break;
            }
        }

        result
    }
}

#[derive(Debug)]
pub(crate) struct SequencePool {
    free: VecDeque<i32>,
    owned: BTreeSet<i32>,
    cached: BTreeMap<i32, SequenceCache>,
    radix: Option<PagedRadixCache<KvPageResidency>>,
    radix_tiers: Option<RadixTierState>,
    radix_locks: BTreeMap<i32, RadixNodeId>,
}

impl SequencePool {
    pub(crate) fn new(count: u32) -> Self {
        Self {
            free: (0..count)
                .map(|value| i32::try_from(value).expect("validated sequence count fits i32"))
                .collect(),
            owned: BTreeSet::new(),
            cached: BTreeMap::new(),
            radix: None,
            radix_tiers: None,
            radix_locks: BTreeMap::new(),
        }
    }

    pub(crate) fn enable_radix(&mut self, config: &RadixCacheConfig) {
        assert!(
            self.owned.is_empty(),
            "radix must be enabled before admission"
        );
        self.cached.clear();
        self.radix = Some(PagedRadixCache::new(config.page_tokens.get() as usize));
        self.radix_tiers = Some(RadixTierState::new(config));
    }

    pub(crate) fn radix_enabled(&self) -> bool {
        self.radix.is_some()
    }

    /// Return (logical cached tokens, tokens already resident in native device cells).
    pub(crate) fn cached_radix_prefix(&mut self, prompt: &[LlamaToken]) -> (usize, usize) {
        let Some(radix) = self.radix.as_mut() else {
            return (0, 0);
        };
        let path = radix.match_path(&prompt[..prompt.len().saturating_sub(1)]);
        let device = path
            .nodes
            .iter()
            .filter(|node| matches!(radix.page(**node), KvPageResidency::Device(_)))
            .count()
            * radix.page_size();
        (path.matched_tokens, device)
    }

    pub(crate) fn attach_radix_prefix(
        &mut self,
        context: &mut LlamaContext<'_>,
        sequence_id: i32,
        prompt: &[LlamaToken],
    ) -> Result<RadixAttach, RadixCacheError> {
        let Some(radix) = self.radix.as_mut() else {
            return Ok(RadixAttach::default());
        };
        let matchable = &prompt[..prompt.len().saturating_sub(1)];
        let matched = radix.match_path(matchable);
        radix.lock_path(matched.terminal);
        assert!(
            self.radix_locks
                .insert(sequence_id, matched.terminal)
                .is_none(),
            "sequence already held a radix lock"
        );

        let offloaded = matched
            .nodes
            .iter()
            .filter(|node| !matches!(radix.page(**node), KvPageResidency::Device(_)))
            .count();
        let required = offloaded.saturating_mul(radix.page_size());
        if !self.ensure_free_radix_cells(context, required) {
            self.unlock_radix_sequence(sequence_id);
            return Err(RadixCacheError::Capacity);
        }

        let page_size = self.radix.as_ref().expect("radix is enabled").page_size();
        let mut attached_tokens = 0;
        let mut attached_terminal = 0;
        let mut attached = RadixAttach::default();
        for node_id in matched.nodes {
            let source_tier = match self.radix.as_ref().expect("radix is enabled").page(node_id) {
                KvPageResidency::Device(_) => 0,
                KvPageResidency::Host(_) => 1,
                KvPageResidency::Disk(_) => 2,
            };
            let promotion_started = Instant::now();
            let page = match self.promote_radix_page(context, node_id) {
                Ok(page) => page,
                Err(_) => {
                    // A cache-tier failure must not fail inference. Keep the successfully attached
                    // prefix, atomically shorten its lock, discard the unreadable suffix, and let
                    // ordinary prefill reconstruct it.
                    self.unlock_radix_sequence(sequence_id);
                    if let Some(radix) = self.radix.as_mut() {
                        radix.lock_path(attached_terminal);
                    }
                    self.radix_locks.insert(sequence_id, attached_terminal);
                    let _ = self.remove_radix_subtree(context, node_id);
                    attached.cached_tokens = attached_tokens;
                    return Ok(attached);
                }
            };
            if source_tier != 0 {
                attached.promotion_ms += promotion_started.elapsed().as_secs_f64() * 1_000.0;
            }
            if let Err(error) = context.attach_kv_page(page, sequence_id) {
                let _ = context.clear_kv_cache_seq(Some(sequence_id as u32), None, None);
                self.unlock_radix_sequence(sequence_id);
                return Err(error.into());
            }
            attached_tokens += page_size;
            attached_terminal = node_id;
            match source_tier {
                0 => attached.device_tokens += page_size,
                1 => attached.host_tokens += page_size,
                2 => attached.disk_tokens += page_size,
                _ => unreachable!(),
            }
        }
        attached.cached_tokens = matched.matched_tokens;
        Ok(attached)
    }

    pub(crate) fn retain_radix_prefix(
        &mut self,
        context: &mut LlamaContext<'_>,
        sequence_id: i32,
        tokens: &[LlamaToken],
        resident_tokens: usize,
    ) {
        let Some(radix) = self.radix.as_mut() else {
            return;
        };
        let resident = resident_tokens.min(tokens.len());
        let _ = radix.insert_missing(&tokens[..resident], |start, end| {
            context
                .pin_kv_page(sequence_id, start as u32, end as u32)
                .map(KvPageResidency::Device)
        });
        if let Some(terminal) = self.radix_locks.remove(&sequence_id) {
            radix.unlock_path(terminal);
        }
    }

    /// Publish newly evaluated pages while the producing request stays active. Its old matched
    /// path lock is atomically replaced by a lock covering the now-longer materialized path.
    pub(crate) fn publish_active_radix_prefix(
        &mut self,
        context: &mut LlamaContext<'_>,
        sequence_id: i32,
        tokens: &[LlamaToken],
        resident_tokens: usize,
    ) -> bool {
        let Some(radix) = self.radix.as_mut() else {
            return false;
        };
        let resident = resident_tokens.min(tokens.len());
        let Ok(terminal) = radix.insert_missing(&tokens[..resident], |start, end| {
            context
                .pin_kv_page(sequence_id, start as u32, end as u32)
                .map(KvPageResidency::Device)
        }) else {
            return false;
        };
        if let Some(previous) = self.radix_locks.insert(sequence_id, terminal) {
            radix.unlock_path(previous);
        }
        radix.lock_path(terminal);
        true
    }

    pub(crate) fn ensure_free_radix_cells(
        &mut self,
        context: &mut LlamaContext<'_>,
        required: usize,
    ) -> bool {
        if self.radix.is_none() {
            return true;
        }
        while context.kv_page_stats().free_cells < required as u32 {
            if !self.demote_one_device_page(context) {
                return false;
            }
        }
        true
    }

    fn unlock_radix_sequence(&mut self, sequence_id: i32) {
        if let Some(terminal) = self.radix_locks.remove(&sequence_id)
            && let Some(radix) = self.radix.as_mut()
        {
            radix.unlock_path(terminal);
        }
    }

    fn promote_radix_page(
        &mut self,
        context: &mut LlamaContext<'_>,
        node_id: RadixNodeId,
    ) -> Result<LlamaKvPageId, RadixCacheError> {
        let residency = self
            .radix
            .as_ref()
            .expect("radix is enabled")
            .page(node_id)
            .clone();
        match residency {
            KvPageResidency::Device(page) => Ok(page),
            KvPageResidency::Host(blob) => {
                let page = context.import_kv_page(&blob)?;
                let bytes = blob.len() as u64;
                *self
                    .radix
                    .as_mut()
                    .expect("radix is enabled")
                    .page_mut(node_id) = KvPageResidency::Device(page);
                let tiers = self.radix_tiers.as_mut().expect("radix tiers exist");
                tiers.host_bytes = tiers.host_bytes.saturating_sub(bytes);
                Ok(page)
            }
            KvPageResidency::Disk(disk) => {
                let file = fs::File::open(&disk.path)?;
                let expected = usize::try_from(disk.bytes)
                    .map_err(|_| RadixCacheError::CorruptDiskPage(disk.path.clone()))?;
                if file.metadata()?.len() != disk.bytes {
                    return Err(RadixCacheError::CorruptDiskPage(disk.path));
                }
                let mut blob = Vec::with_capacity(expected);
                file.take(disk.bytes.saturating_add(1))
                    .read_to_end(&mut blob)?;
                if blob.len() as u64 != disk.bytes
                    || <[u8; 32]>::from(Sha256::digest(&blob)) != disk.sha256
                {
                    return Err(RadixCacheError::CorruptDiskPage(disk.path));
                }
                let page = context.import_kv_page(&blob)?;
                *self
                    .radix
                    .as_mut()
                    .expect("radix is enabled")
                    .page_mut(node_id) = KvPageResidency::Device(page);
                let tiers = self.radix_tiers.as_mut().expect("radix tiers exist");
                tiers.disk_bytes = tiers.disk_bytes.saturating_sub(disk.bytes);
                let _ = fs::remove_file(disk.path);
                Ok(page)
            }
        }
    }

    fn demote_one_device_page(&mut self, context: &mut LlamaContext<'_>) -> bool {
        let victim = self.radix.as_ref().and_then(|radix| {
            radix.resident_leaf(|page| matches!(page, KvPageResidency::Device(_)))
        });
        let Some(victim) = victim else {
            return false;
        };
        let page = match self.radix.as_ref().expect("radix exists").page(victim) {
            KvPageResidency::Device(page) => *page,
            _ => unreachable!("device resident selector returned another tier"),
        };
        let Ok(blob) = context.export_kv_page(page) else {
            return false;
        };
        let destination = self.store_lower_tier(context, blob);
        match destination {
            Ok(Some(destination)) => {
                if context.release_kv_page(page).is_err() {
                    self.discard_residency(context, destination);
                    return false;
                }
                *self.radix.as_mut().expect("radix exists").page_mut(victim) = destination;
                true
            }
            Ok(None) => self
                .remove_radix_subtree(context, victim)
                .is_some_and(|pages| !pages.is_empty()),
            Err(_) => self
                .remove_radix_subtree(context, victim)
                .is_some_and(|pages| !pages.is_empty()),
        }
    }

    fn store_lower_tier(
        &mut self,
        context: &mut LlamaContext<'_>,
        blob: Vec<u8>,
    ) -> Result<Option<KvPageResidency>, RadixCacheError> {
        let bytes = blob.len() as u64;
        let host_limit = self.radix_tiers.as_ref().expect("tiers exist").host_limit;
        if bytes <= host_limit {
            self.make_host_room(context, bytes)?;
            let tiers = self.radix_tiers.as_mut().expect("tiers exist");
            if tiers.host_bytes.saturating_add(bytes) <= tiers.host_limit {
                tiers.host_bytes = tiers.host_bytes.saturating_add(bytes);
                return Ok(Some(KvPageResidency::Host(Arc::from(blob))));
            }
        }
        self.write_disk_page(context, &blob)
            .map(|value| value.map(KvPageResidency::Disk))
    }

    fn make_host_room(
        &mut self,
        context: &mut LlamaContext<'_>,
        incoming: u64,
    ) -> Result<(), RadixCacheError> {
        loop {
            let tiers = self.radix_tiers.as_ref().expect("tiers exist");
            if tiers.host_bytes.saturating_add(incoming) <= tiers.host_limit {
                return Ok(());
            }
            let victim = self.radix.as_ref().and_then(|radix| {
                radix.resident_leaf(|page| matches!(page, KvPageResidency::Host(_)))
            });
            let Some(victim) = victim else {
                return Ok(());
            };
            let blob = match self.radix.as_ref().expect("radix exists").page(victim) {
                KvPageResidency::Host(blob) => Arc::clone(blob),
                _ => unreachable!("host selector returned another tier"),
            };
            if let Some(disk) = self.write_disk_page(context, &blob)? {
                let bytes = blob.len() as u64;
                *self.radix.as_mut().expect("radix exists").page_mut(victim) =
                    KvPageResidency::Disk(disk);
                let tiers = self.radix_tiers.as_mut().expect("tiers exist");
                tiers.host_bytes = tiers.host_bytes.saturating_sub(bytes);
            } else {
                let _ = self.remove_radix_subtree(context, victim);
            }
        }
    }

    fn write_disk_page(
        &mut self,
        context: &mut LlamaContext<'_>,
        blob: &[u8],
    ) -> Result<Option<DiskKvPage>, RadixCacheError> {
        let bytes = blob.len() as u64;
        let limit = self.radix_tiers.as_ref().expect("tiers exist").disk_limit;
        if bytes > limit {
            return Ok(None);
        }
        self.make_disk_room(context, bytes);
        let tiers = self.radix_tiers.as_mut().expect("tiers exist");
        if tiers.disk_bytes.saturating_add(bytes) > tiers.disk_limit {
            return Ok(None);
        }
        let scope = tiers
            .disk_scope
            .as_ref()
            .expect("positive disk limit has a scope");
        fs::create_dir_all(scope)?;
        let serial = tiers.next_file;
        tiers.next_file = tiers.next_file.saturating_add(1);
        let digest = <[u8; 32]>::from(Sha256::digest(blob));
        let final_path = scope.join(format!("{serial:016x}.kvp"));
        let temporary = scope.join(format!(".{serial:016x}.tmp"));
        let mut file = fs::File::create(&temporary)?;
        file.write_all(blob)?;
        file.sync_all()?;
        drop(file);
        if let Err(error) = fs::rename(&temporary, &final_path) {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
        if let Ok(directory) = fs::File::open(scope) {
            let _ = directory.sync_all();
        }
        tiers.disk_bytes = tiers.disk_bytes.saturating_add(bytes);
        Ok(Some(DiskKvPage {
            path: final_path,
            bytes,
            sha256: digest,
        }))
    }

    fn make_disk_room(&mut self, context: &mut LlamaContext<'_>, incoming: u64) {
        loop {
            let tiers = self.radix_tiers.as_ref().expect("tiers exist");
            if tiers.disk_bytes.saturating_add(incoming) <= tiers.disk_limit {
                break;
            }
            let victim = self.radix.as_ref().and_then(|radix| {
                radix.resident_leaf(|page| matches!(page, KvPageResidency::Disk(_)))
            });
            let Some(victim) = victim else {
                break;
            };
            if self.remove_radix_subtree(context, victim).is_none() {
                break;
            }
        }
    }

    fn remove_radix_subtree(
        &mut self,
        context: &mut LlamaContext<'_>,
        node_id: RadixNodeId,
    ) -> Option<Vec<KvPageResidency>> {
        let pages = self.radix.as_mut()?.remove_subtree(node_id)?;
        for page in pages.iter().cloned() {
            self.discard_residency(context, page);
        }
        Some(pages)
    }

    fn discard_residency(&mut self, context: &mut LlamaContext<'_>, residency: KvPageResidency) {
        let tiers = self.radix_tiers.as_mut().expect("tiers exist");
        match residency {
            KvPageResidency::Device(page) => {
                let _ = context.release_kv_page(page);
            }
            KvPageResidency::Host(blob) => {
                tiers.host_bytes = tiers.host_bytes.saturating_sub(blob.len() as u64);
            }
            KvPageResidency::Disk(disk) => {
                tiers.disk_bytes = tiers.disk_bytes.saturating_sub(disk.bytes);
                let _ = fs::remove_file(disk.path);
            }
        }
    }

    pub(crate) fn acquire(&mut self) -> Option<i32> {
        let sequence_id = self.free.pop_front()?;
        assert!(self.owned.insert(sequence_id), "sequence was already owned");
        Some(sequence_id)
    }

    pub(crate) fn acquire_matching(&mut self, prompt: &[LlamaToken]) -> Option<i32> {
        let best = self
            .free
            .iter()
            .enumerate()
            .max_by_key(|(_, sequence_id)| {
                self.cached.get(sequence_id).map_or(0, |cache| {
                    cache
                        .prompt
                        .iter()
                        .zip(prompt)
                        .take_while(|(left, right)| left == right)
                        .count()
                })
            })
            .map(|(index, _)| index)?;
        let sequence_id = self.free.remove(best)?;
        assert!(self.owned.insert(sequence_id), "sequence was already owned");
        Some(sequence_id)
    }

    pub(crate) fn release(&mut self, sequence_id: i32) {
        assert!(
            self.owned.remove(&sequence_id),
            "attempted to release an unowned sequence"
        );
        self.cached.remove(&sequence_id);
        assert!(
            self.radix_locks.remove(&sequence_id).is_none(),
            "radix path must be unlocked before sequence release"
        );
        self.free.push_front(sequence_id);
    }

    pub(crate) fn release_cached(&mut self, sequence_id: i32, cache: SequenceCache) {
        assert!(
            self.owned.remove(&sequence_id),
            "attempted to release an unowned sequence"
        );
        self.cached.insert(sequence_id, cache);
        self.free.push_front(sequence_id);
    }

    pub(crate) fn take_cache(&mut self, sequence_id: i32) -> Option<SequenceCache> {
        self.cached.remove(&sequence_id)
    }

    /// Remove a sequence from service after native cleanup failed. Reusing it could expose one
    /// request to another request's resident state, so capacity is deliberately reduced instead.
    pub(crate) fn quarantine(&mut self, sequence_id: i32) {
        assert!(
            self.owned.remove(&sequence_id),
            "attempted to quarantine an unowned sequence"
        );
        self.cached.remove(&sequence_id);
        if let Some(terminal) = self.radix_locks.remove(&sequence_id)
            && let Some(radix) = self.radix.as_mut()
        {
            radix.unlock_path(terminal);
        }
    }

    #[cfg(test)]
    pub(crate) fn owned(&self) -> &BTreeSet<i32> {
        &self.owned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_work_always_precedes_prefill() {
        let mut planner = BatchPlanner::new(2);
        let plan = planner.plan(
            &[
                WorkCandidate {
                    sequence_id: 0,
                    kind: WorkKind::Prefill { remaining: 10 },
                },
                WorkCandidate {
                    sequence_id: 1,
                    kind: WorkKind::Decode,
                },
                WorkCandidate {
                    sequence_id: 2,
                    kind: WorkKind::Prefill { remaining: 10 },
                },
            ],
            6,
        );
        assert_eq!(plan[0], BatchWork::Decode { sequence_id: 1 });
        assert_eq!(plan.iter().map(batch_size).sum::<usize>(), 6);
    }

    #[test]
    fn prompt_quanta_are_fair_and_fill_the_batch() {
        let mut planner = BatchPlanner::new(2);
        let candidates = [
            WorkCandidate {
                sequence_id: 0,
                kind: WorkKind::Prefill { remaining: 100 },
            },
            WorkCandidate {
                sequence_id: 1,
                kind: WorkKind::Prefill { remaining: 100 },
            },
        ];
        let first = planner.plan(&candidates, 6);
        let second = planner.plan(&candidates, 6);

        assert_eq!(first.iter().map(batch_size).sum::<usize>(), 6);
        assert_eq!(second.iter().map(batch_size).sum::<usize>(), 6);
        assert_eq!(
            first[0],
            BatchWork::Prefill {
                sequence_id: 0,
                tokens: 2
            }
        );
        assert_eq!(
            second[0],
            BatchWork::Prefill {
                sequence_id: 1,
                tokens: 2
            }
        );

        let allocated = |plan: &[BatchWork], sequence_id| {
            plan.iter()
                .filter_map(|work| match work {
                    BatchWork::Prefill {
                        sequence_id: id,
                        tokens,
                    } if *id == sequence_id => Some(*tokens),
                    _ => None,
                })
                .sum::<usize>()
        };
        assert!(allocated(&first, 0).abs_diff(allocated(&first, 1)) <= 2);
        assert!(allocated(&second, 0).abs_diff(allocated(&second, 1)) <= 2);
    }

    #[test]
    fn decode_order_is_stable_when_every_sequence_fits() {
        let mut planner = BatchPlanner::new(2);
        let candidates = [
            WorkCandidate {
                sequence_id: 2,
                kind: WorkKind::Decode,
            },
            WorkCandidate {
                sequence_id: 0,
                kind: WorkKind::Decode,
            },
            WorkCandidate {
                sequence_id: 1,
                kind: WorkKind::Decode,
            },
        ];
        let expected = vec![
            BatchWork::Decode { sequence_id: 0 },
            BatchWork::Decode { sequence_id: 1 },
            BatchWork::Decode { sequence_id: 2 },
        ];
        assert_eq!(planner.plan(&candidates, 3), expected);
        assert_eq!(planner.plan(&candidates, 3), expected);
    }

    #[test]
    fn sequence_ownership_is_isolated_and_reused_only_after_release() {
        let mut pool = SequencePool::new(2);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        assert_ne!(first, second);
        assert_eq!(pool.acquire(), None);
        assert_eq!(pool.owned(), &BTreeSet::from([first, second]));

        pool.release(first);
        assert_eq!(pool.acquire(), Some(first));
        assert_eq!(pool.owned(), &BTreeSet::from([first, second]));
    }

    #[test]
    fn failed_cleanup_quarantines_only_the_affected_sequence() {
        let mut pool = SequencePool::new(2);
        let cancelled = pool.acquire().unwrap();
        let survivor = pool.acquire().unwrap();
        pool.quarantine(cancelled);

        assert_eq!(pool.acquire(), None);
        assert_eq!(pool.owned(), &BTreeSet::from([survivor]));
        pool.release(survivor);
        assert_eq!(pool.acquire(), Some(survivor));
        assert_eq!(pool.acquire(), None);
    }

    #[test]
    fn retained_cache_returns_with_the_same_sequence() {
        let mut pool = SequencePool::new(2);
        let sequence = pool.acquire().unwrap();
        pool.release_cached(
            sequence,
            SequenceCache {
                prompt: vec![LlamaToken::new(7)],
                checkpoints: Vec::new(),
            },
        );
        assert_eq!(pool.acquire(), Some(sequence));
        let cache = pool.take_cache(sequence).unwrap();
        assert_eq!(cache.prompt, vec![LlamaToken::new(7)]);
        assert!(cache.checkpoints.is_empty());
    }

    #[test]
    fn matching_cache_is_selected_independently_of_free_order() {
        let mut pool = SequencePool::new(2);
        let first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        pool.release_cached(
            first,
            SequenceCache {
                prompt: vec![LlamaToken::new(1), LlamaToken::new(2)],
                checkpoints: Vec::new(),
            },
        );
        pool.release_cached(
            second,
            SequenceCache {
                prompt: vec![LlamaToken::new(7), LlamaToken::new(8)],
                checkpoints: Vec::new(),
            },
        );

        assert_eq!(
            pool.acquire_matching(&[LlamaToken::new(1), LlamaToken::new(9)]),
            Some(first)
        );
    }

    fn batch_size(work: &BatchWork) -> usize {
        match work {
            BatchWork::Decode { .. } => 1,
            BatchWork::Prefill { tokens, .. } => *tokens,
        }
    }
}
