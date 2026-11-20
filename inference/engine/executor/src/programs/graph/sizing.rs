//! Direct graph sizing from compiled `EntryContract`s without `.seismicbundle`
//! decoding, `NativeGraphMetadata`, `NativeGraphTraces`, or AST interpretation.
//!
//! `SizingDraft` implements `GraphDraft` so every family's topology builder runs
//! directly on it during assessment. Sealing/certifying over a regime's class
//! slices evaluates `E::contract_sizing_bound` and `E::contract_param_spec`,
//! resolves operand links via `seismic::sizing::resolve_operand_links`, places
//! storage via `seismic::sizing::plan_storage_with_links` (`R_b`), computes
//! `plan_heap`, `bound_heap`, and `recorded` (`GraphCensus`) using Seismic's
//! single-owner formulas, and simultaneously records the launch census
//! (`SizedCall`) and deduplicated kernel set (`KernelRequest`) for downstream
//! assessment.

use super::GraphError;
use super::draft::GraphDraft;
use crate::assessment::stage_timer;
use seismic::sizing::{
    argument_reference, bindings_heap, make_timeline_port, plan_heap_of, recorded_census_of,
    resolve_operand_links, CompiledStoragePlan, EncodedWorkflowArgs, OperandCarrySide,
    OperandCarrySpec, OperandNode, PendingWorkflowResults, PlanHeapNode, RepresentationId,
    ScratchNeed, StorageNode, StoragePort, TimelineNode, WorkflowResultRef,
    INPUT_NODE, OPERAND_ACTIVATION,
};
use seismic::contract::ScratchConfigs;
use seismic::{
    BackendName, ContractError, ContractNodeShape, ContractScratchMeta, Dim, Element,
    EntryContract, HostHeap, KernelDomainViolation, KernelRequest, LaunchEncoding,
    NativeGraphClassSlice, NativeGraphStorageBytes, NativePort, NativeTimelinePort, OperandCarry,
    TensorParamSpec, WorkflowTensor,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

static NEXT_SIZING_GRAPH: AtomicU64 = AtomicU64::new(1);

/// Fast non-cryptographic 64-bit FxHash for assessment-scoped memoization tables.
#[derive(Clone, Copy, Default)]
struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const K: u64 = 0x517cc1b727220a95;

    #[inline]
    fn mix(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // Avalanche so both low bits (H1) and high bits (H2) are well-distributed for SwissTable.
        let mut h = self.hash;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let word = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
            self.mix(word);
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let word = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
            self.mix(u64::from(word));
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            let word = u16::from_ne_bytes(bytes[..2].try_into().unwrap());
            self.mix(u64::from(word));
            bytes = &bytes[2..];
        }
        if let Some(&byte) = bytes.first() {
            self.mix(u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }
}

#[derive(Clone, Copy, Default)]
struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

/// Pass-through hasher for pre-hashed `u64` keys produced by `FxHasher::finish`.
#[derive(Default)]
struct IdentityHasher {
    hash: u64,
}

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("IdentityHasher only hashes u64 keys");
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }
}

#[derive(Clone, Copy, Default)]
struct IdentityBuildHasher;

impl BuildHasher for IdentityBuildHasher {
    type Hasher = IdentityHasher;

    #[inline]
    fn build_hasher(&self) -> IdentityHasher {
        IdentityHasher::default()
    }
}

enum CacheBucket<K, V> {
    One(K, V),
    Many(Vec<(K, V)>),
}

/// Upper bound on entries per thread-local `FxCache` table to bound steady-state
/// memory when caches are retained warm across `SizingScope` lifetimes.
const SCOPED_CACHE_MAX_ENTRIES: usize = 4096;

/// Pre-hashed, collision-safe map supporting zero-allocation borrowed lookups
/// against owned keys without `Borrow` lifetime constraints.
struct FxCache<K, V> {
    buckets: HashMap<u64, CacheBucket<K, V>, IdentityBuildHasher>,
}

impl<K, V> FxCache<K, V> {
    fn new() -> Self {
        Self {
            buckets: HashMap::with_hasher(IdentityBuildHasher),
        }
    }

    #[inline]
    fn get_with(&self, hash: u64, mut eq: impl FnMut(&K) -> bool) -> Option<&V> {
        match self.buckets.get(&hash)? {
            CacheBucket::One(k, v) => eq(k).then_some(v),
            CacheBucket::Many(entries) => entries.iter().find(|(k, _)| eq(k)).map(|(_, v)| v),
        }
    }

    fn insert_with(&mut self, hash: u64, key: K, value: V)
    where
        K: PartialEq,
    {
        if self.buckets.len() >= SCOPED_CACHE_MAX_ENTRIES {
            self.buckets.clear();
        }
        use std::collections::hash_map::Entry;
        match self.buckets.entry(hash) {
            Entry::Vacant(slot) => {
                slot.insert(CacheBucket::One(key, value));
            }
            Entry::Occupied(slot) => {
                let bucket = slot.into_mut();
                match bucket {
                    CacheBucket::One(k, v) if *k == key => {
                        *v = value;
                    }
                    CacheBucket::One(_, _) => {
                        let old = std::mem::replace(bucket, CacheBucket::Many(Vec::new()));
                        let CacheBucket::One(old_k, old_v) = old else {
                            unreachable!()
                        };
                        let CacheBucket::Many(vec) = bucket else {
                            unreachable!()
                        };
                        vec.push((old_k, old_v));
                        vec.push((key, value));
                    }
                    CacheBucket::Many(vec) => {
                        if let Some((_, v)) = vec.iter_mut().find(|(k, _)| *k == key) {
                            *v = value;
                        } else {
                            vec.push((key, value));
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn hash_str(h: &mut FxHasher, s: &str) {
    h.write_usize(s.len());
    h.write(s.as_bytes());
}

#[inline]
fn hash_elements(h: &mut FxHasher, els: &[(&str, Element)]) {
    h.write_usize(els.len());
    for (name, el) in els {
        hash_str(h, name);
        el.hash(h);
    }
}

#[inline]
fn hash_dims(h: &mut FxHasher, dims: &[(&str, u64)]) {
    h.write_usize(dims.len());
    for (name, v) in dims {
        hash_str(h, name);
        h.write_u64(*v);
    }
}

#[inline]
fn hash_dims_except(h: &mut FxHasher, dims: &[(&str, u64)], skip_slot: usize) {
    h.write_usize(dims.len());
    for (i, (name, v)) in dims.iter().enumerate() {
        hash_str(h, name);
        if i != skip_slot {
            h.write_u64(*v);
        }
    }
}

#[inline]
fn hash_u64_slice(h: &mut FxHasher, values: &[u64]) {
    h.write_usize(values.len());
    for &v in values {
        h.write_u64(v);
    }
}

/// Inline-small string avoiding heap allocation for short names (<= 22 bytes),
/// such as parameter, element, and dimension identifiers.
#[derive(Clone, Debug)]
enum CompactStr {
    Inline { len: u8, buf: [u8; 22] },
    Heap(Box<str>),
}

impl CompactStr {
    #[inline]
    fn new(s: &str) -> Self {
        let bytes = s.as_bytes();
        if bytes.len() <= 22 {
            let mut buf = [0u8; 22];
            buf[..bytes.len()].copy_from_slice(bytes);
            Self::Inline {
                len: bytes.len() as u8,
                buf,
            }
        } else {
            Self::Heap(Box::from(s))
        }
    }

    #[inline]
    fn as_str(&self) -> &str {
        match self {
            Self::Inline { len, buf } => {
                // SAFETY: `Self::Inline` is only constructed from a valid `&str` in `Self::new`.
                unsafe { std::str::from_utf8_unchecked(&buf[..usize::from(*len)]) }
            }
            Self::Heap(s) => s,
        }
    }
}

impl PartialEq for CompactStr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for CompactStr {}

impl Hash for CompactStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl std::fmt::Display for CompactStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[inline]
fn elements_eq(owned: &[(CompactStr, Element)], borrowed: &[(&str, Element)]) -> bool {
    owned.len() == borrowed.len()
        && owned
            .iter()
            .zip(borrowed)
            .all(|((on, oe), (bn, be))| on.as_str() == *bn && oe == be)
}

#[inline]
fn dims_eq(owned: &[(CompactStr, u64)], borrowed: &[(&str, u64)]) -> bool {
    owned.len() == borrowed.len()
        && owned
            .iter()
            .zip(borrowed)
            .all(|((on, ov), (bn, bv))| on.as_str() == *bn && ov == bv)
}

#[inline]
fn dims_eq_except(
    owned: &[(CompactStr, u64)],
    borrowed: &[(&str, u64)],
    skip_slot: usize,
) -> bool {
    owned.len() == borrowed.len()
        && owned
            .iter()
            .zip(borrowed)
            .enumerate()
            .all(|(i, ((on, ov), (bn, bv)))| on.as_str() == *bn && (i == skip_slot || ov == bv))
}

#[inline]
fn param_spec_prefix_hash(
    eval: ParamEvalFn,
    backend: BackendName,
    parameter: &str,
    els: &[(&str, Element)],
) -> FxHasher {
    let mut h = FxHasher::default();
    h.write_usize(eval as usize);
    backend.hash(&mut h);
    hash_str(&mut h, parameter);
    hash_elements(&mut h, els);
    h
}

#[inline]
fn sizing_point_prefix_hash(
    eval_point: SizingPointEvalFn,
    backend: BackendName,
    els: &[(&str, Element)],
) -> FxHasher {
    let mut h = FxHasher::default();
    h.write_usize(eval_point as usize);
    backend.hash(&mut h);
    hash_elements(&mut h, els);
    h
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScopedScratchConfigKey {
    eval_configs: usize,
    backend: BackendName,
    elements: Vec<(CompactStr, Element)>,
    static_values: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScopedSizingPointKey {
    eval_point: usize,
    backend: BackendName,
    elements: Vec<(CompactStr, Element)>,
    dims: Vec<(CompactStr, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScopedParamSpecKey {
    eval: usize,
    backend: BackendName,
    parameter: CompactStr,
    elements: Vec<(CompactStr, Element)>,
    dims: Vec<(CompactStr, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScopedParamReadDimsKey {
    eval: usize,
    backend: BackendName,
    parameter: CompactStr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScopedNodeCoverKey {
    eval_point: usize,
    backend: BackendName,
    elements: Vec<(CompactStr, Element)>,
    dims: Vec<(CompactStr, u64)>,
    varying_slot: usize,
    values: Vec<u64>,
}

struct ScopedScratchConfigs {
    depth: usize,
    map: FxCache<ScopedScratchConfigKey, Result<ScratchConfigs, ContractError>>,
    points: FxCache<ScopedSizingPointKey, Result<(Vec<u64>, Vec<u64>), ContractError>>,
    node_cover_points: FxCache<ScopedNodeCoverKey, Result<(Vec<u64>, Vec<u64>), ContractError>>,
    param_specs: FxCache<ScopedParamSpecKey, Result<TensorParamSpec, ContractError>>,
    param_read_dims: FxCache<ScopedParamReadDimsKey, Option<Arc<[&'static str]>>>,
}

type ScopedCell<'a> = Option<&'a RefCell<Option<ScopedScratchConfigs>>>;

thread_local! {
    static SCOPED_SCRATCH_CONFIGS: RefCell<Option<ScopedScratchConfigs>> = const { RefCell::new(None) };
}

/// Stack-lifetime RAII guard that activates thread-local memoization of
/// `EntryContract::contract_scratch_configs`, `EntryContract::contract_sizing_at_point`,
/// and `EntryContract::contract_param_spec` across `SizingDraft` instances.
///
/// Nesting-safe: all cached entries are pure functions of their keys (static contract
/// fn pointers, backend, element types, and dimension values), so dropping the outermost
/// `SizingScope` deactivates the scope (`depth = 0`) while retaining the thread-local
/// tables warm (bounded by `SCOPED_CACHE_MAX_ENTRIES`) instead of freeing and
/// reallocating them on every scope exit.
pub struct SizingScope {
    _private: (),
}

impl SizingScope {
    pub fn enter() -> Self {
        SCOPED_SCRATCH_CONFIGS.with(|cell| {
            let mut slot = cell.borrow_mut();
            match slot.as_mut() {
                Some(state) => state.depth += 1,
                None => {
                    *slot = Some(ScopedScratchConfigs {
                        depth: 1,
                        map: FxCache::new(),
                        points: FxCache::new(),
                        node_cover_points: FxCache::new(),
                        param_specs: FxCache::new(),
                        param_read_dims: FxCache::new(),
                    });
                }
            }
        });
        Self { _private: () }
    }
}

impl Drop for SizingScope {
    fn drop(&mut self) {
        SCOPED_SCRATCH_CONFIGS.with(|cell| {
            let mut slot = cell.borrow_mut();
            if let Some(state) = slot.as_mut() {
                debug_assert!(state.depth > 0);
                state.depth -= 1;
            }
        });
    }
}

fn eval_and_insert_param_spec(
    cell: &RefCell<Option<ScopedScratchConfigs>>,
    eval: ParamEvalFn,
    backend: BackendName,
    els: &[(&str, Element)],
    parameter: &str,
    dims: &[(&str, u64)],
) -> Result<TensorParamSpec, ContractError> {
    let eval_addr = eval as usize;
    let mut h = param_spec_prefix_hash(eval, backend, parameter, els);
    hash_dims(&mut h, dims);
    let hash = h.finish();
    let result = eval(backend, els, parameter, dims);
    let key = ScopedParamSpecKey {
        eval: eval_addr,
        backend,
        parameter: CompactStr::new(parameter),
        elements: els.iter().map(|(n, e)| (CompactStr::new(n), *e)).collect(),
        dims: dims.iter().map(|(n, v)| (CompactStr::new(n), *v)).collect(),
    };
    if let Some(state) = cell.borrow_mut().as_mut() {
        state.param_specs.insert_with(hash, key, result.clone());
    }
    result
}

fn eval_or_lookup_param_spec(
    scoped_cell: ScopedCell<'_>,
    eval: ParamEvalFn,
    backend: BackendName,
    els: &[(&str, Element)],
    parameter: &str,
    dims: &[(&str, u64)],
) -> Result<TensorParamSpec, ContractError> {
    let Some(cell) = scoped_cell else {
        return eval(backend, els, parameter, dims);
    };
    let eval_addr = eval as usize;
    let mut h = param_spec_prefix_hash(eval, backend, parameter, els);
    hash_dims(&mut h, dims);
    let hash = h.finish();
    let cached = {
        let guard = cell.borrow();
        guard.as_ref().and_then(|state| {
            state
                .param_specs
                .get_with(hash, |k| {
                    k.eval == eval_addr
                        && k.backend == backend
                        && k.parameter.as_str() == parameter
                        && dims_eq(&k.dims, dims)
                        && elements_eq(&k.elements, els)
                })
                .cloned()
        })
    };
    if let Some(result) = cached {
        return result;
    }
    let result = eval(backend, els, parameter, dims);
    let key = ScopedParamSpecKey {
        eval: eval_addr,
        backend,
        parameter: CompactStr::new(parameter),
        elements: els.iter().map(|(n, e)| (CompactStr::new(n), *e)).collect(),
        dims: dims.iter().map(|(n, v)| (CompactStr::new(n), *v)).collect(),
    };
    if let Some(state) = cell.borrow_mut().as_mut() {
        state.param_specs.insert_with(hash, key, result.clone());
    }
    result
}

/// Look up or evaluate `eval(backend, els, parameter, dims)` in `SCOPED_SCRATCH_CONFIGS`,
/// returning `spec.bytes` directly without cloning `TensorParamSpec` (`extents: Vec<u64>`).
#[allow(clippy::too_many_arguments)]
fn eval_or_lookup_param_bytes(
    scoped_cell: ScopedCell<'_>,
    eval: ParamEvalFn,
    backend: BackendName,
    prefix: FxHasher,
    els: &[(&str, Element)],
    parameter: &str,
    dims: &[(&str, u64)],
) -> Result<u64, ContractError> {
    let Some(cell) = scoped_cell else {
        return eval(backend, els, parameter, dims).map(|spec| spec.bytes);
    };
    let eval_addr = eval as usize;
    let mut h = prefix;
    hash_dims(&mut h, dims);
    let hash = h.finish();
    let cached = {
        let guard = cell.borrow();
        guard.as_ref().and_then(|state| {
            let entry = state.param_specs.get_with(hash, |k| {
                k.eval == eval_addr
                    && k.backend == backend
                    && k.parameter.as_str() == parameter
                    && dims_eq(&k.dims, dims)
                    && elements_eq(&k.elements, els)
            })?;
            Some(match entry {
                Ok(spec) => Ok(spec.bytes),
                Err(error) => Err(error.clone()),
            })
        })
    };
    if let Some(result) = cached {
        return result;
    }
    let result = eval(backend, els, parameter, dims);
    let bytes = match &result {
        Ok(spec) => Ok(spec.bytes),
        Err(error) => Err(error.clone()),
    };
    let key = ScopedParamSpecKey {
        eval: eval_addr,
        backend,
        parameter: CompactStr::new(parameter),
        elements: els.iter().map(|(n, e)| (CompactStr::new(n), *e)).collect(),
        dims: dims.iter().map(|(n, v)| (CompactStr::new(n), *v)).collect(),
    };
    if let Some(state) = cell.borrow_mut().as_mut() {
        state.param_specs.insert_with(hash, key, result);
    }
    bytes
}

/// Discover the exact set of dimension names read by `eval` for `(backend, parameter)`
/// by probing with an initially empty `dims` slice and adding each reported
/// `ContractError::MissingDimension(name)` from `find_dim`. Memoized per
/// `(eval, backend, parameter)` in `SizingScope`.
fn lookup_or_probe_param_read_dims(
    scoped_cell: ScopedCell<'_>,
    eval: ParamEvalFn,
    backend: BackendName,
    els: &[(&str, Element)],
    parameter: &str,
    find_dim: impl Fn(&str) -> Option<u64>,
) -> Option<Arc<[&'static str]>> {
    let eval_addr = eval as usize;
    let mut h = FxHasher::default();
    h.write_usize(eval_addr);
    backend.hash(&mut h);
    hash_str(&mut h, parameter);
    let hash = h.finish();
    if let Some(cell) = scoped_cell {
        let cached = {
            let guard = cell.borrow();
            guard.as_ref().and_then(|state| {
                state
                    .param_read_dims
                    .get_with(hash, |k| {
                        k.eval == eval_addr
                            && k.backend == backend
                            && k.parameter.as_str() == parameter
                    })
                    .cloned()
            })
        };
        if let Some(found) = cached {
            return found;
        }
    }
    let mut used: Vec<&'static str> = Vec::new();
    let mut probe_dims: Vec<(&str, u64)> = Vec::new();
    let mut probed_spec: Option<TensorParamSpec> = None;
    let discovered = loop {
        match eval(backend, els, parameter, &probe_dims) {
            Ok(spec) => {
                probed_spec = Some(spec);
                break Some(Arc::<[&'static str]>::from(used));
            }
            Err(ContractError::MissingDimension(missing)) => {
                if used.contains(&missing) {
                    break None;
                }
                let Some(val) = find_dim(missing) else {
                    break None;
                };
                used.push(missing);
                probe_dims.push((missing, val));
            }
            Err(_) => break None,
        }
    };
    if let Some(cell) = scoped_cell {
        let key = ScopedParamReadDimsKey {
            eval: eval_addr,
            backend,
            parameter: CompactStr::new(parameter),
        };
        let spec_entry = probed_spec.map(|spec| {
            let mut sh = param_spec_prefix_hash(eval, backend, parameter, els);
            hash_dims(&mut sh, &probe_dims);
            let spec_hash = sh.finish();
            let spec_key = ScopedParamSpecKey {
                eval: eval_addr,
                backend,
                parameter: CompactStr::new(parameter),
                elements: els.iter().map(|(n, e)| (CompactStr::new(n), *e)).collect(),
                dims: probe_dims
                    .iter()
                    .map(|(n, v)| (CompactStr::new(n), *v))
                    .collect(),
            };
            (spec_hash, spec_key, spec)
        });
        if let Some(state) = cell.borrow_mut().as_mut() {
            state
                .param_read_dims
                .insert_with(hash, key, discovered.clone());
            if let Some((spec_hash, spec_key, spec)) = spec_entry {
                state
                    .param_specs
                    .insert_with(spec_hash, spec_key, Ok(spec));
            }
        }
    }
    discovered
}

#[allow(clippy::too_many_arguments)]
fn eval_or_lookup_scratch_configs(
    entry_name: &'static str,
    scoped_cell: ScopedCell<'_>,
    eval_configs: ScratchConfigsEvalFn,
    backend: BackendName,
    els: &[(&str, Element)],
    statics: &[&'static str],
    dims: &[(String, u64)],
    point_dims: &[(&str, u64)],
) -> Result<ScratchConfigs, ContractError> {
    let start = stage_timer::is_recording().then(Instant::now);
    let record_miss = |start: Option<Instant>| {
        if let Some(start) = start {
            let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            stage_timer::record_stage(|t| {
                t.sd_scratch_configs_miss_ns += elapsed;
                t.sd_scratch_configs_misses += 1;
                let entry = t.sd_configs_by_entry.entry(entry_name).or_default();
                entry.0 += elapsed;
                entry.1 += 1;
            });
        }
    };
    let Some(cell) = scoped_cell else {
        let result = eval_configs(backend, els, point_dims);
        record_miss(start);
        return result;
    };
    let mut static_stack = [0u64; 16];
    let mut static_heap: Vec<u64> = Vec::new();
    let static_values: &[u64] = if statics.len() <= static_stack.len() {
        for (slot, &name) in static_stack.iter_mut().zip(statics) {
            match dims.iter().find(|(dim, _)| dim == name) {
                Some((_, v)) => *slot = *v,
                None => {
                    let result = eval_configs(backend, els, point_dims);
                    record_miss(start);
                    return result;
                }
            }
        }
        &static_stack[..statics.len()]
    } else {
        static_heap.reserve(statics.len());
        for &name in statics {
            match dims.iter().find(|(dim, _)| dim == name) {
                Some((_, v)) => static_heap.push(*v),
                None => {
                    let result = eval_configs(backend, els, point_dims);
                    record_miss(start);
                    return result;
                }
            }
        }
        &static_heap
    };
    let eval_addr = eval_configs as usize;
    let mut h = FxHasher::default();
    h.write_usize(eval_addr);
    backend.hash(&mut h);
    hash_elements(&mut h, els);
    h.write_usize(static_values.len());
    for &v in static_values {
        h.write_u64(v);
    }
    let hash = h.finish();
    let cached = {
        let guard = cell.borrow();
        guard.as_ref().and_then(|state| {
            state
                .map
                .get_with(hash, |k| {
                    k.eval_configs == eval_addr
                        && k.backend == backend
                        && k.static_values.as_slice() == static_values
                        && elements_eq(&k.elements, els)
                })
                .cloned()
        })
    };
    if let Some(result) = cached {
        if let Some(start) = start {
            let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            stage_timer::record_stage(|t| {
                t.sd_scratch_configs_hit_ns += elapsed;
                t.sd_scratch_configs_hits += 1;
            });
        }
        return result;
    }
    let result = eval_configs(backend, els, point_dims);
    let scoped_key = ScopedScratchConfigKey {
        eval_configs: eval_addr,
        backend,
        elements: els.iter().map(|(n, e)| (CompactStr::new(n), *e)).collect(),
        static_values: static_values.to_vec(),
    };
    if let Some(state) = cell.borrow_mut().as_mut() {
        state.map.insert_with(hash, scoped_key, result.clone());
    }
    record_miss(start);
    result
}

#[allow(clippy::too_many_arguments)]
fn eval_or_lookup_sizing_point(
    entry_name: &'static str,
    scoped_cell: ScopedCell<'_>,
    eval_point: SizingPointEvalFn,
    backend: BackendName,
    prefix: FxHasher,
    els: &[(&str, Element)],
    point_dims: &[(&str, u64)],
    configs: &ScratchConfigs,
    max_tensor_results: &mut [u64],
    max_scratch_bytes: &mut [u64],
    point_results: &mut Vec<u64>,
    point_scratch: &mut Vec<u64>,
) -> Result<(), ContractError> {
    let start = stage_timer::is_recording().then(Instant::now);
    let record_miss = |start: Option<Instant>| {
        if let Some(start) = start {
            let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            stage_timer::record_stage(|t| {
                t.sd_sizing_point_miss_ns += elapsed;
                t.sd_sizing_point_misses += 1;
                let entry = t.sd_point_by_entry.entry(entry_name).or_default();
                entry.0 += elapsed;
                entry.1 += 1;
            });
        }
    };
    let Some(cell) = scoped_cell else {
        let result = eval_point(
            backend,
            els,
            point_dims,
            configs,
            max_tensor_results,
            max_scratch_bytes,
        );
        record_miss(start);
        return result;
    };
    let eval_addr = eval_point as usize;
    let mut h = prefix;
    hash_dims(&mut h, point_dims);
    let hash = h.finish();
    let cached = {
        let guard = cell.borrow();
        guard.as_ref().and_then(|state| {
            let entry = state.points.get_with(hash, |k| {
                k.eval_point == eval_addr
                    && k.backend == backend
                    && dims_eq(&k.dims, point_dims)
                    && elements_eq(&k.elements, els)
            })?;
            match entry {
                Ok((results, scratch)) => {
                    for (dst, &src) in max_tensor_results.iter_mut().zip(results) {
                        *dst = (*dst).max(src);
                    }
                    for (dst, &src) in max_scratch_bytes.iter_mut().zip(scratch) {
                        *dst = (*dst).max(src);
                    }
                    Some(Ok(()))
                }
                Err(error) => Some(Err(error.clone())),
            }
        })
    };
    if let Some(result) = cached {
        if let Some(start) = start {
            let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            stage_timer::record_stage(|t| {
                t.sd_sizing_point_hit_ns += elapsed;
                t.sd_sizing_point_hits += 1;
            });
        }
        return result;
    }
    point_results.clear();
    point_results.resize(max_tensor_results.len(), 0);
    point_scratch.clear();
    point_scratch.resize(max_scratch_bytes.len(), 0);
    let eval_result = eval_point(
        backend,
        els,
        point_dims,
        configs,
        point_results,
        point_scratch,
    );
    let point_key = ScopedSizingPointKey {
        eval_point: eval_addr,
        backend,
        elements: els.iter().map(|(n, e)| (CompactStr::new(n), *e)).collect(),
        dims: point_dims
            .iter()
            .map(|(n, v)| (CompactStr::new(n), *v))
            .collect(),
    };
    let result = match eval_result {
        Ok(()) => {
            for (dst, &src) in max_tensor_results.iter_mut().zip(point_results.iter()) {
                *dst = (*dst).max(src);
            }
            for (dst, &src) in max_scratch_bytes.iter_mut().zip(point_scratch.iter()) {
                *dst = (*dst).max(src);
            }
            let stored = Ok((point_results.clone(), point_scratch.clone()));
            if let Some(state) = cell.borrow_mut().as_mut() {
                state.points.insert_with(hash, point_key, stored);
            }
            Ok(())
        }
        Err(error) => {
            if let Some(state) = cell.borrow_mut().as_mut() {
                state.points.insert_with(hash, point_key, Err(error.clone()));
            }
            Err(error)
        }
    };
    record_miss(start);
    result
}

/// Look up or compute the covered maximum `(max_tensor_results, max_scratch_bytes)` across
/// a single varying dimension (`point_dims[slot]`) over `values`.
///
/// Checks `node_cover_points` first (1 lookup across all `values`). On miss, checks `points`
/// for all `values` under a single `RefCell::borrow()`, evaluates any missed points outside
/// the borrow, and inserts all new `points` entries plus the `node_cover_points` entry under
/// a single `RefCell::borrow_mut()`.
#[allow(clippy::too_many_arguments)]
fn eval_or_lookup_node_cover_points(
    entry_name: &'static str,
    scoped_cell: ScopedCell<'_>,
    eval_point: SizingPointEvalFn,
    backend: BackendName,
    prefix: FxHasher,
    els: &[(&str, Element)],
    point_dims: &mut [(&str, u64)],
    slot: usize,
    values: &[u64],
    configs: &ScratchConfigs,
    max_tensor_results: &mut [u64],
    max_scratch_bytes: &mut [u64],
    point_results: &mut Vec<u64>,
    point_scratch: &mut Vec<u64>,
    missed_v: &mut Vec<(u64, u64)>,
) -> Result<(), ContractError> {
    let start = stage_timer::is_recording().then(Instant::now);
    let Some(cell) = scoped_cell else {
        for &v in values {
            point_dims[slot].1 = v;
            eval_point(
                backend,
                els,
                point_dims,
                configs,
                max_tensor_results,
                max_scratch_bytes,
            )?;
        }
        if let Some(start) = start {
            let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let n = values.len() as u64;
            stage_timer::record_stage(|t| {
                t.sd_sizing_point_miss_ns += elapsed;
                t.sd_sizing_point_misses += n;
                let entry = t.sd_point_by_entry.entry(entry_name).or_default();
                entry.0 += elapsed;
                entry.1 += n;
            });
        }
        return Ok(());
    };

    let eval_addr = eval_point as usize;
    let mut ch = prefix;
    ch.write_usize(slot);
    hash_dims_except(&mut ch, point_dims, slot);
    hash_u64_slice(&mut ch, values);
    let cover_hash = ch.finish();

    let mut hit_err: Option<ContractError> = None;
    let mut point_hits = 0u64;
    missed_v.clear();

    let cover_hit = {
        let guard = cell.borrow();
        if let Some(state) = guard.as_ref() {
            if let Some(entry) = state.node_cover_points.get_with(cover_hash, |k| {
                k.eval_point == eval_addr
                    && k.backend == backend
                    && k.varying_slot == slot
                    && k.values.as_slice() == values
                    && dims_eq_except(&k.dims, point_dims, slot)
                    && elements_eq(&k.elements, els)
            }) {
                match entry {
                    Ok((results, scratch)) => {
                        for (dst, &src) in max_tensor_results.iter_mut().zip(results) {
                            *dst = (*dst).max(src);
                        }
                        for (dst, &src) in max_scratch_bytes.iter_mut().zip(scratch) {
                            *dst = (*dst).max(src);
                        }
                        Some(Ok(()))
                    }
                    Err(error) => Some(Err(error.clone())),
                }
            } else {
                for &v in values {
                    if missed_v.iter().any(|&(mv, _)| mv == v) {
                        continue;
                    }
                    point_dims[slot].1 = v;
                    let mut sh = prefix;
                    hash_dims(&mut sh, point_dims);
                    let point_hash = sh.finish();
                    if let Some(entry) = state.points.get_with(point_hash, |k| {
                        k.eval_point == eval_addr
                            && k.backend == backend
                            && dims_eq(&k.dims, point_dims)
                            && elements_eq(&k.elements, els)
                    }) {
                        point_hits += 1;
                        match entry {
                            Ok((results, scratch)) => {
                                for (dst, &src) in max_tensor_results.iter_mut().zip(results) {
                                    *dst = (*dst).max(src);
                                }
                                for (dst, &src) in max_scratch_bytes.iter_mut().zip(scratch) {
                                    *dst = (*dst).max(src);
                                }
                            }
                            Err(error) => {
                                hit_err = Some(error.clone());
                                break;
                            }
                        }
                    } else {
                        missed_v.push((v, point_hash));
                    }
                }
                None
            }
        } else {
            None
        }
    };

    if let Some(result) = cover_hit {
        if let Some(start) = start {
            let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let n = values.len() as u64;
            stage_timer::record_stage(|t| {
                t.sd_sizing_point_hit_ns += elapsed;
                t.sd_sizing_point_hits += n;
            });
        }
        return result;
    }

    let mut computed: Vec<(u64, u64, Result<(Vec<u64>, Vec<u64>), ContractError>)> =
        Vec::with_capacity(missed_v.len());
    let mut point_misses = 0u64;
    for &(v, point_hash) in missed_v.iter() {
        point_dims[slot].1 = v;
        point_results.clear();
        point_results.resize(max_tensor_results.len(), 0);
        point_scratch.clear();
        point_scratch.resize(max_scratch_bytes.len(), 0);
        point_misses += 1;
        match eval_point(
            backend,
            els,
            point_dims,
            configs,
            point_results,
            point_scratch,
        ) {
            Ok(()) => {
                for (dst, &src) in max_tensor_results.iter_mut().zip(point_results.iter()) {
                    *dst = (*dst).max(src);
                }
                for (dst, &src) in max_scratch_bytes.iter_mut().zip(point_scratch.iter()) {
                    *dst = (*dst).max(src);
                }
                computed.push((
                    point_hash,
                    v,
                    Ok((point_results.clone(), point_scratch.clone())),
                ));
            }
            Err(error) => {
                hit_err = Some(error.clone());
                computed.push((point_hash, v, Err(error)));
                break;
            }
        }
    }

    let owned_elements: Vec<(CompactStr, Element)> =
        els.iter().map(|(n, e)| (CompactStr::new(n), *e)).collect();
    let mut base_dims: Vec<(CompactStr, u64)> = point_dims
        .iter()
        .map(|(n, v)| (CompactStr::new(n), *v))
        .collect();
    base_dims[slot].1 = 0;

    let cover_stored = match &hit_err {
        Some(error) => Err(error.clone()),
        None => Ok((max_tensor_results.to_vec(), max_scratch_bytes.to_vec())),
    };

    if let Some(state) = cell.borrow_mut().as_mut() {
        for (point_hash, v, res) in computed {
            let mut p_dims = base_dims.clone();
            p_dims[slot].1 = v;
            let point_key = ScopedSizingPointKey {
                eval_point: eval_addr,
                backend,
                elements: owned_elements.clone(),
                dims: p_dims,
            };
            state.points.insert_with(point_hash, point_key, res);
        }
        let cover_key = ScopedNodeCoverKey {
            eval_point: eval_addr,
            backend,
            elements: owned_elements,
            dims: base_dims,
            varying_slot: slot,
            values: values.to_vec(),
        };
        state
            .node_cover_points
            .insert_with(cover_hash, cover_key, cover_stored);
    }

    if let Some(start) = start {
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        stage_timer::record_stage(|t| {
            if point_misses > 0 {
                t.sd_sizing_point_miss_ns += elapsed;
                t.sd_sizing_point_misses += point_misses;
                let entry = t.sd_point_by_entry.entry(entry_name).or_default();
                entry.0 += elapsed;
                entry.1 += point_misses;
            } else {
                t.sd_sizing_point_hit_ns += elapsed;
                t.sd_sizing_point_hits += point_hits;
            }
        });
    }

    match hit_err {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

type ScratchConfigsEvalFn =
    fn(BackendName, &[(&str, Element)], &[(&str, u64)]) -> Result<ScratchConfigs, ContractError>;

type LaunchConfigsEvalFn =
    fn(BackendName, &[(&str, Element)], &[(&str, u64)]) -> Result<Vec<Vec<u64>>, ContractError>;

type LaunchMaskEvalFn =
    fn(BackendName, &[(&str, Element)], &[(&str, u64)], &[u64]) -> Result<u64, ContractError>;

type SizingPointEvalFn = fn(
    BackendName,
    &[(&str, Element)],
    &[(&str, u64)],
    &ScratchConfigs,
    &mut [u64],
    &mut [u64],
) -> Result<(), ContractError>;

type ScratchConfigKey<'a> = (usize, &'a [(String, Element)], &'a [(String, u64)]);
type LaunchConfigKey<'a> = (usize, &'a [(String, Element)], &'a [(String, u64)]);

pub(crate) type LaunchesEvalFn = fn(
    BackendName,
    &[(&str, Element)],
    &[(&str, u64)],
    &[(&str, u64, bool)],
) -> Result<u64, ContractError>;

fn eval_contract_launches<E: EntryContract>(
    backend: BackendName,
    elements: &[(&str, Element)],
    dims: &[(&str, u64)],
    words: &[(&str, u64, bool)],
) -> Result<u64, ContractError> {
    let params = E::contract_default_params(backend, elements, dims)?;
    E::contract_active_launches(backend, elements, dims, &params, words)
}

/// One external or uploaded port bound to a node call (weight, norm, scale,
/// state, host-uploaded table row, etc.), recorded by `SizingDraft` for L4's
/// byte/FLOP demand census.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SizedBoundPort {
    pub parameter_ordinal: usize,
    pub element: Element,
    pub extents: Vec<u64>,
    pub bytes: u64,
    pub is_weight: bool,
    pub is_upload: bool,
}

/// One enqueued node call recorded by `SizingDraft` for L4's launch & FLOP/byte
/// census (replacing `DecodeDemand`'s separate walk).
#[derive(Clone, Debug)]
pub(crate) struct SizedCall {
    pub entry: &'static str,
    pub backend: BackendName,
    pub elements: Vec<(String, Element)>,
    pub dimensions: Vec<(String, u64)>,
    pub bound_ports: Vec<SizedBoundPort>,
    pub timeline_before: usize,
    pub eval_launches: LaunchesEvalFn,
}

impl PartialEq for SizedCall {
    fn eq(&self, other: &Self) -> bool {
        self.entry == other.entry
            && self.backend == other.backend
            && self.elements == other.elements
            && self.dimensions == other.dimensions
            && self.bound_ports == other.bound_ports
            && self.timeline_before == other.timeline_before
    }
}

impl Eq for SizedCall {}

/// One distinct block shape's decode-regime calls (and resident decode calls,
/// when the block streams routed experts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlockDecodeCensus {
    pub calls: Vec<SizedCall>,
    pub timelines: usize,
    pub resident: Option<(Vec<SizedCall>, usize)>,
}

/// Accumulated launch census and deduplicated kernel set for a certified target
/// family, threaded through `CheckedTargetFamily` for L4.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TargetFamilySizingCensus {
    /// Distinct `(entry, backend, elements, statics)` kernels referenced by the family.
    pub kernels: HashSet<KernelRequest>,
    /// Ordered decode-regime calls for the target entry (+ per-layer entry, if any).
    pub entry_calls: Vec<SizedCall>,
    /// Timeline nodes in the entry graphs.
    pub entry_timelines: usize,
    /// Distinct block shapes' decode calls.
    pub block_shapes: Vec<BlockDecodeCensus>,
    /// Per block `0..blocks.len()`, index into `block_shapes`.
    pub block_map: Vec<usize>,
}

/// Accumulated launch census and deduplicated kernel set for a certified
/// readout family.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReadoutSizingCensus {
    pub kernels: HashSet<KernelRequest>,
    pub decode_calls: Vec<SizedCall>,
}

/// What a structural regime certifies to under direct contract sizing:
/// exact `NativeGraphStorageBytes`, `plan_heap`, and `bound_heap` matching
/// `NativeGraphLayout`, plus the accumulated launch census and kernel set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SizingOutline {
    storage: NativeGraphStorageBytes,
    plan_heap: HostHeap,
    bound_heap: HostHeap,
    launches: Arc<[SizedCall]>,
    kernels: Arc<[KernelRequest]>,
    timeline_count: usize,
}

impl SizingOutline {
    pub(crate) fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.storage
    }

    pub(crate) fn plan_heap(&self) -> HostHeap {
        self.plan_heap
    }

    pub(crate) fn bound_heap(&self) -> HostHeap {
        self.bound_heap
    }

    pub(crate) fn launches(&self) -> &[SizedCall] {
        &self.launches
    }

    pub(crate) fn kernels(&self) -> &[KernelRequest] {
        &self.kernels
    }

    pub(crate) fn timeline_count(&self) -> usize {
        self.timeline_count
    }
}

type ParamEvalFn =
    fn(BackendName, &[(&str, Element)], &str, &[(&str, u64)]) -> Result<TensorParamSpec, ContractError>;

enum PortSource {
    Direct {
        class_extent: Option<(usize, &'static str, Option<&'static str>)>,
    },
    Checked {
        entry_name: &'static str,
        class_scope: Option<&'static str>,
        parameter: CompactStr,
        elements: std::ops::Range<usize>,
        dimensions: std::ops::Range<usize>,
        eval: ParamEvalFn,
    },
}

struct SizingPort {
    element: Element,
    extents: Vec<u64>,
    byte_len: u64,
    local: bool,
    owned_input: bool,
    is_weight: bool,
    source: PortSource,
}

struct SizingNode {
    entry_name: &'static str,
    class_scope: Option<&'static str>,
    elements: Vec<(String, Element)>,
    dimensions: Vec<(String, u64)>,
    args: EncodedWorkflowArgs,
    activation: Option<RepresentationId>,
    shape: ContractNodeShape,
    scratch_meta: ContractScratchMeta,
    share: Option<(u32, WorkflowResultRef)>,
    timeline_before: usize,
    eval_configs: ScratchConfigsEvalFn,
    eval_point: SizingPointEvalFn,
    eval_launches: LaunchesEvalFn,
    eval_launch_configs: LaunchConfigsEvalFn,
    eval_launch_mask: LaunchMaskEvalFn,
}

/// Direct contract-backed `GraphDraft` used by assessment.
pub(crate) struct SizingDraft {
    backend: BackendName,
    identity: u64,
    class_scope: Option<&'static str>,
    ports: Vec<SizingPort>,
    checked_elements: Vec<(CompactStr, Element)>,
    checked_dimensions: Vec<(CompactStr, u64)>,
    nodes: Vec<SizingNode>,
    exports: Vec<WorkflowResultRef>,
    stores: Vec<(WorkflowResultRef, u32)>,
    carries: Vec<OperandCarrySpec>,
    timelines: Vec<(usize, TimelineNode)>,
    marks: Vec<usize>,
}

fn contract_error(
    entry: &'static str,
    backend: BackendName,
    statics: &[&'static str],
    dims: &[(&str, u64)],
    error: ContractError,
) -> GraphError {
    match error {
        ContractError::DomainViolation => GraphError::KernelDomain(KernelDomainViolation {
            entry,
            backend,
            statics: statics
                .iter()
                .filter_map(|&name| {
                    dims.iter()
                        .find(|(candidate, _)| *candidate == name)
                        .map(|(_, value)| (name.to_owned(), *value))
                })
                .collect(),
        }),
        ContractError::MissingConditionWord { word, index } => GraphError::Invalid(format!(
            "`{entry}`: launch condition reads `{word}[{index}]`, which was not supplied"
        )),
        other => GraphError::Invalid(format!("`{entry}`: {other}")),
    }
}

fn scratch_needs_from(meta: ContractScratchMeta) -> Vec<ScratchNeed> {
    meta.scratch
        .iter()
        .map(|s| ScratchNeed {
            bytes: 0,
            sync: s.sync,
            operand: s.operand,
            emission: s.emission,
        })
        .collect()
}

fn for_each_combination(
    axes: &[&[u64]],
    current: &mut Vec<u64>,
    f: &mut impl FnMut(&[u64]) -> Result<(), GraphError>,
) -> Result<(), GraphError> {
    if current.len() == axes.len() {
        return f(current);
    }
    let level = current.len();
    for &value in axes[level] {
        current.push(value);
        for_each_combination(axes, current, f)?;
        current.pop();
    }
    Ok(())
}

impl SizingDraft {
    pub(crate) fn new(backend: BackendName) -> Self {
        if stage_timer::is_recording() {
            stage_timer::record_stage(|t| t.sd_drafts_built += 1);
        }
        Self {
            backend,
            identity: NEXT_SIZING_GRAPH.fetch_add(1, Ordering::Relaxed),
            class_scope: None,
            ports: Vec::new(),
            checked_elements: Vec::new(),
            checked_dimensions: Vec::new(),
            nodes: Vec::new(),
            exports: Vec::new(),
            stores: Vec::new(),
            carries: Vec::new(),
            timelines: Vec::new(),
            marks: Vec::new(),
        }
    }

    fn push_port(&mut self, port: SizingPort) -> NativePort {
        let ordinal = u32::try_from(self.ports.len()).expect("native port ordinal exhausted");
        self.ports.push(port);
        NativePort::from_raw(self.identity, ordinal)
    }

    fn checked_port<E: EntryContract>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        local: bool,
        owned_input: bool,
    ) -> Result<NativePort, GraphError> {
        let mut stack_dims = [("", 0u64); 16];
        let mut heap_dims: Vec<(&str, u64)> = Vec::new();
        let dims: &[(&str, u64)] = if dimensions.len() <= stack_dims.len() {
            for (slot, (name, d)) in stack_dims.iter_mut().zip(dimensions) {
                *slot = (*name, d.clone().into().fixed());
            }
            &stack_dims[..dimensions.len()]
        } else {
            heap_dims.extend(
                dimensions
                    .iter()
                    .map(|(name, d)| (*name, d.clone().into().fixed())),
            );
            &heap_dims
        };

        let spec = stage_timer::time_stage(
            |t, ns| t.sd_port_param_spec_ns += ns,
            || {
                SCOPED_SCRATCH_CONFIGS.with(|cell| {
                    let eval: ParamEvalFn = E::contract_param_spec;
                    let eval_addr = eval as usize;
                    let backend = self.backend;
                    let mut read_h = FxHasher::default();
                    read_h.write_usize(eval_addr);
                    backend.hash(&mut read_h);
                    hash_str(&mut read_h, parameter);
                    let read_hash = read_h.finish();

                    let mut filtered_stack: [(&'static str, u64); 8] = [("", 0u64); 8];
                    let mut filtered_heap: Vec<(&'static str, u64)> = Vec::new();

                    enum FastPath {
                        Inactive,
                        Hit(Result<TensorParamSpec, ContractError>),
                        MissNeedProbe,
                        MissWithFiltered(usize),
                        MissFullDims,
                    }

                    let fast = {
                        let guard = cell.borrow();
                        match guard.as_ref().filter(|s| s.depth > 0) {
                            None => FastPath::Inactive,
                            Some(state) => match state.param_read_dims.get_with(read_hash, |k| {
                                k.eval == eval_addr
                                    && k.backend == backend
                                    && k.parameter.as_str() == parameter
                            }) {
                                None => FastPath::MissNeedProbe,
                                Some(Some(used)) => {
                                    let (filtered, tag): (&[(&'static str, u64)], usize) =
                                        if used.len() <= filtered_stack.len() {
                                            let mut n = 0;
                                            for &dim_name in used.iter() {
                                                if let Some((_, v)) =
                                                    dims.iter().find(|(name, _)| *name == dim_name)
                                                {
                                                    filtered_stack[n] = (dim_name, *v);
                                                    n += 1;
                                                }
                                            }
                                            (&filtered_stack[..n], n)
                                        } else {
                                            for &dim_name in used.iter() {
                                                if let Some((_, v)) =
                                                    dims.iter().find(|(name, _)| *name == dim_name)
                                                {
                                                    filtered_heap.push((dim_name, *v));
                                                }
                                            }
                                            (filtered_heap.as_slice(), usize::MAX)
                                        };
                                    let mut sh =
                                        param_spec_prefix_hash(eval, backend, parameter, elements);
                                    hash_dims(&mut sh, filtered);
                                    let spec_hash = sh.finish();
                                    if let Some(result) =
                                        state.param_specs.get_with(spec_hash, |k| {
                                            k.eval == eval_addr
                                                && k.backend == backend
                                                && k.parameter.as_str() == parameter
                                                && dims_eq(&k.dims, filtered)
                                                && elements_eq(&k.elements, elements)
                                        })
                                    {
                                        FastPath::Hit(result.clone())
                                    } else {
                                        FastPath::MissWithFiltered(tag)
                                    }
                                }
                                Some(None) => {
                                    let mut sh =
                                        param_spec_prefix_hash(eval, backend, parameter, elements);
                                    hash_dims(&mut sh, dims);
                                    let spec_hash = sh.finish();
                                    if let Some(result) =
                                        state.param_specs.get_with(spec_hash, |k| {
                                            k.eval == eval_addr
                                                && k.backend == backend
                                                && k.parameter.as_str() == parameter
                                                && dims_eq(&k.dims, dims)
                                                && elements_eq(&k.elements, elements)
                                        })
                                    {
                                        FastPath::Hit(result.clone())
                                    } else {
                                        FastPath::MissFullDims
                                    }
                                }
                            },
                        }
                    };

                    match fast {
                        FastPath::Inactive => eval(backend, elements, parameter, dims),
                        FastPath::Hit(result) => result,
                        FastPath::MissWithFiltered(tag) => {
                            let filtered: &[(&str, u64)] = if tag == usize::MAX {
                                filtered_heap.as_slice()
                            } else {
                                &filtered_stack[..tag]
                            };
                            eval_and_insert_param_spec(
                                cell, eval, backend, elements, parameter, filtered,
                            )
                        }
                        FastPath::MissFullDims => {
                            eval_and_insert_param_spec(cell, eval, backend, elements, parameter, dims)
                        }
                        FastPath::MissNeedProbe => {
                            if let Some(used) = lookup_or_probe_param_read_dims(
                                Some(cell),
                                eval,
                                backend,
                                elements,
                                parameter,
                                |missing| dims.iter().find(|(n, _)| *n == missing).map(|(_, v)| *v),
                            ) {
                                let filtered: &[(&str, u64)] =
                                    if used.len() <= filtered_stack.len() {
                                        let mut n = 0;
                                        for &dim_name in used.iter() {
                                            if let Some((_, v)) =
                                                dims.iter().find(|(name, _)| *name == dim_name)
                                            {
                                                filtered_stack[n] = (dim_name, *v);
                                                n += 1;
                                            }
                                        }
                                        &filtered_stack[..n]
                                    } else {
                                        for &dim_name in used.iter() {
                                            if let Some((_, v)) =
                                                dims.iter().find(|(name, _)| *name == dim_name)
                                            {
                                                filtered_heap.push((dim_name, *v));
                                            }
                                        }
                                        filtered_heap.as_slice()
                                    };
                                eval_or_lookup_param_spec(
                                    Some(cell),
                                    eval,
                                    backend,
                                    elements,
                                    parameter,
                                    filtered,
                                )
                            } else {
                                eval_or_lookup_param_spec(
                                    Some(cell),
                                    eval,
                                    backend,
                                    elements,
                                    parameter,
                                    dims,
                                )
                            }
                        }
                    }
                })
            },
        )
        .map_err(|error| contract_error(E::NAME, self.backend, &[], dims, error))?;

        let el_start = self.checked_elements.len();
        self.checked_elements
            .extend(elements.iter().map(|(name, el)| (CompactStr::new(name), *el)));
        let el_end = self.checked_elements.len();

        let dim_start = self.checked_dimensions.len();
        self.checked_dimensions
            .extend(dims.iter().map(|(name, v)| (CompactStr::new(name), *v)));
        let dim_end = self.checked_dimensions.len();

        Ok(self.push_port(SizingPort {
            element: spec.element,
            extents: spec.extents,
            byte_len: spec.bytes,
            local,
            owned_input,
            is_weight: false,
            source: PortSource::Checked {
                entry_name: E::NAME,
                class_scope: self.class_scope,
                parameter: CompactStr::new(parameter),
                elements: el_start..el_end,
                dimensions: dim_start..dim_end,
                eval: E::contract_param_spec,
            },
        }))
    }

    fn enqueue_node<E: EntryContract>(
        &mut self,
        elements: &[(&str, Element)],
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        stage_timer::time_stage(
            |t, ns| {
                t.sd_enqueue_ns += ns;
                t.sd_nodes_enqueued += 1;
            },
            || {
                let mut stack_dims = [("", 0u64); 16];
                let mut heap_dims: Vec<(&str, u64)> = Vec::new();
                let dims: &[(&str, u64)] = if dimensions.len() <= stack_dims.len() {
                    for (slot, (name, d)) in stack_dims.iter_mut().zip(dimensions) {
                        *slot = (*name, d.clone().into().fixed());
                    }
                    &stack_dims[..dimensions.len()]
                } else {
                    heap_dims.extend(
                        dimensions
                            .iter()
                            .map(|(name, d)| (*name, d.clone().into().fixed())),
                    );
                    &heap_dims
                };
                let shape = E::contract_node_shape(self.backend)
                    .map_err(|error| contract_error(E::NAME, self.backend, &[], dims, error))?;
                let scratch_meta = E::contract_scratch_meta(self.backend)
                    .map_err(|error| contract_error(E::NAME, self.backend, shape.statics, dims, error))?;
                let encoded = E::encode_workflow(args);
                let activation = elements
                    .iter()
                    .find(|(name, _)| *name == OPERAND_ACTIVATION)
                    .map(|(_, element)| element.id());
                let node_ordinal = u32::try_from(self.nodes.len()).expect("native graph node ordinal exhausted");
                let result_count = u32::try_from(shape.result_ranks.len()).expect("node result count fits u32");
                let pending = PendingWorkflowResults::new(self.identity, node_ordinal, result_count);
                let owned_elements = elements
                    .iter()
                    .map(|(name, el)| ((*name).to_owned(), *el))
                    .collect();
                let owned_dims = dims
                    .iter()
                    .map(|(name, v)| ((*name).to_owned(), *v))
                    .collect();
                self.nodes.push(SizingNode {
                    entry_name: E::NAME,
                    class_scope: self.class_scope,
                    elements: owned_elements,
                    dimensions: owned_dims,
                    args: encoded,
                    activation,
                    shape,
                    scratch_meta,
                    share: None,
                    timeline_before: self.timelines.len(),
                    eval_configs: E::contract_scratch_configs,
                    eval_point: E::contract_sizing_at_point,
                    eval_launches: eval_contract_launches::<E>,
                    eval_launch_configs: E::contract_launch_configs,
                    eval_launch_mask: E::contract_launch_mask,
                });
                Ok(E::decode_workflow(pending))
            },
        )
    }

    /// Certify this drafted topology over each slice set in `slice_sets`,
    /// returning one `SizingOutline` per slice set.
    pub(crate) fn certify_each(
        self,
        slice_sets: &[Vec<NativeGraphClassSlice>],
    ) -> Result<Vec<SizingOutline>, GraphError> {
        // 1 & 2. Resolve topology-level invariants once, plus plan heap, bound heap, and recorded GraphCensus.
        let (base_scratch, links, plan_heap, bound_heap, static_recorded) = stage_timer::time_stage(
            |t, ns| t.sd_operand_links_ns += ns,
            || -> Result<_, GraphError> {
                let base_scratch: Vec<Vec<ScratchNeed>> = self
                    .nodes
                    .iter()
                    .map(|node| scratch_needs_from(node.scratch_meta))
                    .collect();
                let operand_nodes: Vec<OperandNode<'_>> = self
                    .nodes
                    .iter()
                    .zip(&base_scratch)
                    .map(|(node, scratch)| OperandNode {
                        args: &node.args,
                        scratch,
                        activation: node.activation,
                    })
                    .collect();
                let links = resolve_operand_links(&operand_nodes, &self.carries)
                    .map_err(|error| GraphError::Invalid(error.to_string()))?;

                let port_ranks: Vec<usize> = self.ports.iter().map(|port| port.extents.len()).collect();
                let mut disjoint = 0usize;
                let mut heap_nodes = Vec::with_capacity(self.nodes.len());
                let mut census_nodes = if self.backend == BackendName::Cuda {
                    Vec::new()
                } else {
                    Vec::with_capacity(self.nodes.len())
                };
                let mut result_ranks: Vec<&[Option<usize>]> = Vec::with_capacity(self.nodes.len());
                for node in &self.nodes {
                    let external = |ordinal: usize| {
                        argument_reference(&node.args.arguments()[ordinal])
                            .filter(|reference| {
                                reference.node == INPUT_NODE && !self.ports[reference.result as usize].local
                            })
                            .map(|reference| reference.result)
                    };
                    for &[first, second] in node.shape.disjoint {
                        if let (Some(first), Some(second)) = (external(first), external(second)) {
                            if first != second {
                                disjoint += 1;
                            }
                        }
                    }
                    heap_nodes.push(PlanHeapNode {
                        buffers: node.shape.buffers,
                        arguments: node.shape.arguments,
                        words: node.shape.words,
                        launches: node.shape.launches,
                    });
                    if self.backend != BackendName::Cuda {
                        census_nodes.push((node.shape.arguments, node.shape.words, node.shape.launches));
                    }
                    result_ranks.push(node.shape.result_ranks);
                }
                let _ = disjoint;
                let plan_heap = plan_heap_of(
                    &port_ranks,
                    &heap_nodes,
                    &result_ranks,
                    self.exports.len(),
                    self.timelines.len(),
                    self.marks.len(),
                );
                let bound_heap = bindings_heap(self.ports.len(), self.timelines.len());
                let static_recorded = if self.backend == BackendName::Cuda {
                    None
                } else {
                    let timeline_nodes: Vec<TimelineNode> =
                        self.timelines.iter().map(|&(_, node)| node).collect();
                    Some(recorded_census_of(self.backend, &census_nodes, &timeline_nodes))
                };
                Ok((base_scratch, links, plan_heap, bound_heap, static_recorded))
            },
        )?;

        let timeline_count = self.timelines.len();
        // 3. Compile the storage placement topology once (shared across all slice sets).
        let compiled_plan = stage_timer::time_stage(
            |t, ns| t.sd_storage_compile_ns += ns,
            || {
                let template_ports: Vec<StoragePort> = self
                    .ports
                    .iter()
                    .map(|port| StoragePort {
                        byte_len: port.byte_len,
                        local: port.local,
                        owned_input: port.owned_input,
                        prewritten: false,
                    })
                    .collect();
                let template_results: Vec<Vec<Option<u64>>> = self
                    .nodes
                    .iter()
                    .map(|node| {
                        node.shape
                            .result_ranks
                            .iter()
                            .map(|rank| rank.map(|_| 1u64))
                            .collect()
                    })
                    .collect();
                let template_nodes: Vec<StorageNode<'_>> = self
                    .nodes
                    .iter()
                    .zip(&template_results)
                    .zip(&base_scratch)
                    .map(|((node, results), scratch)| StorageNode {
                        args: &node.args,
                        results: results.as_slice(),
                        scratch: scratch.as_slice(),
                        activation: node.activation,
                        share: node.share,
                    })
                    .collect();
                CompiledStoragePlan::of(
                    &template_ports,
                    &template_nodes,
                    &links,
                    &self.exports,
                    &self.stores,
                )
            },
        );

        // 4. For each slice set, cover buffer capacities across all class points and place storage.
        if stage_timer::is_recording() {
            stage_timer::record_stage(|t| t.sd_slice_sets += slice_sets.len() as u64);
        }
        let mut storage_per_slice = Vec::with_capacity(slice_sets.len());
        {
            let mut scratch_configs_cache: HashMap<ScratchConfigKey<'_>, ScratchConfigs, FxBuildHasher> =
                HashMap::with_hasher(FxBuildHasher);
            let mut launch_configs_cache: HashMap<LaunchConfigKey<'_>, Vec<Vec<u64>>, FxBuildHasher> =
                HashMap::with_hasher(FxBuildHasher);
            for slices in slice_sets {
                if slices.is_empty() {
                    return Err(GraphError::Invalid(
                        "graph regime has no admitted classes".to_owned(),
                    ));
                }
                let (port_bytes, result_bytes, mut scratch_needs) =
                    self.cover_capacities(slices, &base_scratch, &mut scratch_configs_cache)?;
                for (node_idx, needs) in scratch_needs.iter_mut().enumerate() {
                    links.charge_node_emissions(node_idx, needs);
                }
                let planned = stage_timer::time_stage(
                    |t, ns| t.sd_storage_eval_ns += ns,
                    || compiled_plan.storage_bytes(&port_bytes, &result_bytes, &scratch_needs),
                );
                let recorded = match static_recorded {
                    Some(recorded) => recorded,
                    None => {
                        let node_launches =
                            self.cuda_node_launches(slices, &mut launch_configs_cache)?;
                        seismic::cuda_recorded_census(
                            self.nodes.len(),
                            &node_launches,
                            &self.timelines,
                            &self.marks,
                        )
                    }
                };
                storage_per_slice.push(NativeGraphStorageBytes {
                    workspace: planned.workspace,
                    sync: planned.sync,
                    output: planned.output,
                    upload: planned.upload,
                    recorded,
                });
            }
        }
        if storage_per_slice.is_empty() {
            return Ok(Vec::new());
        }

        // 5. Record launch census and deduplicated kernel set for L4, consuming `self.nodes`
        //    so `elements` and `dimensions` move into `SizedCall` without cloning.
        let mut launches = Vec::with_capacity(self.nodes.len());
        let mut seen_kernel_sigs: FxCache<(&'static str, Vec<(String, Element)>, Vec<Option<u64>>), ()> =
            FxCache::new();
        let mut seen_kernels: HashSet<KernelRequest, FxBuildHasher> =
            HashSet::with_hasher(FxBuildHasher);
        let mut kernels = Vec::new();
        let mut static_values: Vec<Option<u64>> = Vec::new();
        for node in self.nodes {
            let mut bound_ports = Vec::new();
            for (ordinal, argument) in node.args.arguments().iter().enumerate() {
                if let Some(reference) = argument_reference(argument) {
                    if reference.node == INPUT_NODE {
                        let port = &self.ports[reference.result as usize];
                        if !port.local || port.owned_input {
                            bound_ports.push(SizedBoundPort {
                                parameter_ordinal: ordinal,
                                element: port.element,
                                extents: port.extents.clone(),
                                bytes: port.byte_len,
                                is_weight: port.is_weight,
                                is_upload: port.owned_input,
                            });
                        }
                    }
                }
            }

            static_values.clear();
            for &name in node.shape.statics {
                let value = node
                    .dimensions
                    .iter()
                    .find(|(dim, _)| dim == name)
                    .map(|(_, v)| *v);
                static_values.push(value);
            }
            let mut h = FxHasher::default();
            hash_str(&mut h, node.entry_name);
            h.write_usize(node.elements.len());
            for (name, el) in &node.elements {
                hash_str(&mut h, name);
                el.hash(&mut h);
            }
            static_values.hash(&mut h);
            let sig_hash = h.finish();
            let already_seen = seen_kernel_sigs
                .get_with(sig_hash, |k| {
                    k.0 == node.entry_name && k.2 == static_values && k.1 == node.elements
                })
                .is_some();
            if !already_seen {
                seen_kernel_sigs.insert_with(
                    sig_hash,
                    (node.entry_name, node.elements.clone(), static_values.clone()),
                    (),
                );
                let statics: BTreeMap<String, u64> = node
                    .shape
                    .statics
                    .iter()
                    .zip(&static_values)
                    .filter_map(|(&name, value)| value.map(|v| (name.to_owned(), v)))
                    .collect();
                let kernel = KernelRequest {
                    backend: self.backend,
                    entry: node.entry_name,
                    elements: node.elements.iter().cloned().collect(),
                    statics,
                };
                if seen_kernels.insert(kernel.clone()) {
                    kernels.push(kernel);
                }
            }

            launches.push(SizedCall {
                entry: node.entry_name,
                backend: self.backend,
                elements: node.elements,
                dimensions: node.dimensions,
                bound_ports,
                timeline_before: node.timeline_before,
                eval_launches: node.eval_launches,
            });
        }

        let launches: Arc<[SizedCall]> = Arc::from(launches);
        let kernels: Arc<[KernelRequest]> = Arc::from(kernels);
        let outlines = storage_per_slice
            .into_iter()
            .map(|storage| SizingOutline {
                storage,
                plan_heap,
                bound_heap,
                launches: Arc::clone(&launches),
                kernels: Arc::clone(&kernels),
                timeline_count,
            })
            .collect();
        Ok(outlines)
    }

    fn cuda_node_launches<'a>(
        &'a self,
        slices: &[NativeGraphClassSlice],
        launch_configs_cache: &mut HashMap<LaunchConfigKey<'a>, Vec<Vec<u64>>, FxBuildHasher>,
    ) -> Result<Vec<u64>, GraphError> {
        type NodeLaunchCacheKey<'k> =
            (usize, Option<&'static str>, &'k [(String, Element)], &'k [(String, u64)]);
        let mut node_cache: HashMap<NodeLaunchCacheKey<'a>, u64, FxBuildHasher> =
            HashMap::with_hasher(FxBuildHasher);
        let mut node_launches = Vec::with_capacity(self.nodes.len());
        let mut els: Vec<(&'a str, Element)> = Vec::new();
        let mut point_dims: Vec<(&'a str, u64)> = Vec::new();
        let mut varying_slots: Vec<usize> = Vec::new();
        let mut varying_values: Vec<&[u64]> = Vec::new();
        let mut axes: Vec<&[u64]> = Vec::new();
        let mut combo: Vec<u64> = Vec::new();
        let mut seen_single: HashSet<u64, FxBuildHasher> = HashSet::with_hasher(FxBuildHasher);
        let mut seen_multi: HashSet<Vec<u64>, FxBuildHasher> = HashSet::with_hasher(FxBuildHasher);
        let mut or_masks: Vec<u64> = Vec::new();

        for node in &self.nodes {
            let key: NodeLaunchCacheKey<'a> = (
                node.eval_launch_mask as usize,
                node.class_scope,
                node.elements.as_slice(),
                node.dimensions.as_slice(),
            );
            if let Some(&cached) = node_cache.get(&key) {
                node_launches.push(cached);
                continue;
            }

            els.clear();
            els.extend(node.elements.iter().map(|(name, el)| (name.as_str(), *el)));
            point_dims.clear();
            point_dims.extend(node.dimensions.iter().map(|(name, v)| (name.as_str(), *v)));

            let config_key: LaunchConfigKey<'a> = (
                node.eval_launch_configs as usize,
                node.elements.as_slice(),
                node.dimensions.as_slice(),
            );
            let configs = match launch_configs_cache.entry(config_key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let computed = (node.eval_launch_configs)(self.backend, &els, &point_dims)
                        .map_err(|error| {
                            contract_error(
                                node.entry_name,
                                self.backend,
                                node.shape.statics,
                                &point_dims,
                                error,
                            )
                        })?;
                    slot.insert(computed)
                }
            };

            varying_slots.clear();
            varying_values.clear();
            if !node.shape.launch_when_dims.is_empty() {
                for (i, (name, _)) in node.dimensions.iter().enumerate() {
                    let name_str = name.as_str();
                    if !node.shape.launch_when_dims.contains(&name_str) {
                        continue;
                    }
                    let Some(first_vals) =
                        slices[0].values(node.entry_name, node.class_scope, name_str)
                    else {
                        continue;
                    };
                    varying_slots.push(i);
                    varying_values.push(first_vals);
                    for slice in &slices[1..] {
                        let vals = slice
                            .values(node.entry_name, node.class_scope, name_str)
                            .ok_or_else(|| {
                                GraphError::Invalid(format!(
                                    "graph regime slice omits `{}` dimension `{name_str}`",
                                    node.entry_name
                                ))
                            })?;
                        varying_values.push(vals);
                    }
                }
            }

            or_masks.clear();
            or_masks.resize(configs.len(), 0u64);

            if varying_slots.is_empty() {
                for (or_mask, cfg) in or_masks.iter_mut().zip(configs.iter()) {
                    let mask = (node.eval_launch_mask)(self.backend, &els, &point_dims, cfg)
                        .map_err(|error| {
                            contract_error(
                                node.entry_name,
                                self.backend,
                                node.shape.statics,
                                &point_dims,
                                error,
                            )
                        })?;
                    *or_mask |= mask;
                }
            } else if varying_slots.len() == 1 {
                let slot = varying_slots[0];
                seen_single.clear();
                for &vals in &varying_values {
                    for &v in vals {
                        if !seen_single.insert(v) {
                            continue;
                        }
                        point_dims[slot].1 = v;
                        for (or_mask, cfg) in or_masks.iter_mut().zip(configs.iter()) {
                            let mask =
                                (node.eval_launch_mask)(self.backend, &els, &point_dims, cfg)
                                    .map_err(|error| {
                                        contract_error(
                                            node.entry_name,
                                            self.backend,
                                            node.shape.statics,
                                            &point_dims,
                                            error,
                                        )
                                    })?;
                            *or_mask |= mask;
                        }
                    }
                }
            } else {
                let n_slices = slices.len();
                let dedup = n_slices > 1;
                if dedup {
                    seen_multi.clear();
                }
                for s_idx in 0..n_slices {
                    axes.clear();
                    for v_idx in 0..varying_slots.len() {
                        axes.push(varying_values[v_idx * n_slices + s_idx]);
                    }
                    combo.clear();
                    for_each_combination(&axes, &mut combo, &mut |point| {
                        if dedup && !seen_multi.insert(point.to_vec()) {
                            return Ok(());
                        }
                        for (&slot, &v) in varying_slots.iter().zip(point) {
                            point_dims[slot].1 = v;
                        }
                        for (or_mask, cfg) in or_masks.iter_mut().zip(configs.iter()) {
                            let mask =
                                (node.eval_launch_mask)(self.backend, &els, &point_dims, cfg)
                                    .map_err(|error| {
                                        contract_error(
                                            node.entry_name,
                                            self.backend,
                                            node.shape.statics,
                                            &point_dims,
                                            error,
                                        )
                                    })?;
                            *or_mask |= mask;
                        }
                        Ok(())
                    })?;
                }
            }

            let max_launches = or_masks
                .iter()
                .map(|m| u64::from(m.count_ones()))
                .max()
                .unwrap_or(0);
            node_cache.insert(key, max_launches);
            node_launches.push(max_launches);
        }

        Ok(node_launches)
    }

    fn cover_capacities<'a>(
        &'a self,
        slices: &[NativeGraphClassSlice],
        base_scratch: &[Vec<ScratchNeed>],
        configs_cache: &mut HashMap<ScratchConfigKey<'a>, ScratchConfigs, FxBuildHasher>,
    ) -> Result<(Vec<u64>, Vec<Vec<Option<u64>>>, Vec<Vec<ScratchNeed>>), GraphError> {
        type CheckedPortKey<'k> = (
            usize,
            Option<&'static str>,
            &'k str,
            &'k [(CompactStr, Element)],
            &'k [(CompactStr, u64)],
        );
        SCOPED_SCRATCH_CONFIGS.with(|cell| {
            let scoped_cell: ScopedCell<'_> = cell
                .borrow()
                .as_ref()
                .is_some_and(|s| s.depth > 0)
                .then_some(cell);
            let mut varying: Vec<usize> = Vec::new();
            let mut varying_values: Vec<&[u64]> = Vec::new();
            let mut els: Vec<(&'a str, Element)> = Vec::new();
            let mut point_dims: Vec<(&'a str, u64)> = Vec::new();
            let mut axes: Vec<&[u64]> = Vec::new();
            let mut combo: Vec<u64> = Vec::new();
            let mut seen_single: Vec<u64> = Vec::new();
            let mut missed_v: Vec<(u64, u64)> = Vec::new();

            let port_bytes = stage_timer::time_stage(
                |t, ns| t.sd_cover_ports_ns += ns,
                || -> Result<Vec<u64>, GraphError> {
                    let mut port_cache: HashMap<CheckedPortKey<'a>, u64, FxBuildHasher> =
                        HashMap::with_hasher(FxBuildHasher);
                    let mut varying_slots: Vec<usize> = Vec::new();
                    let mut port_bytes = Vec::with_capacity(self.ports.len());
                    for port in &self.ports {
                        let max_bytes = match &port.source {
                            PortSource::Direct { class_extent: None } => port.byte_len,
                            PortSource::Direct {
                                class_extent: Some((axis, name, scope)),
                            } => {
                                fn lookup_direct<'s>(
                                    slice: &'s NativeGraphClassSlice,
                                    scope: Option<&str>,
                                    name: &str,
                                ) -> Option<&'s [u64]> {
                                    scope
                                        .and_then(|s| slice.class_values(Some(s), name))
                                        .or_else(|| slice.global(name))
                                }
                                let mut first_present = None;
                                for (s_idx, slice) in slices.iter().enumerate() {
                                    if let Some(values) = lookup_direct(slice, *scope, name) {
                                        first_present = Some((s_idx, values));
                                        break;
                                    }
                                }
                                match first_present {
                                    None => port.byte_len,
                                    Some((first_s_idx, _)) if first_s_idx > 0 => {
                                        return Err(GraphError::Invalid(format!(
                                            "graph regime slice omits port class dimension `{name}`"
                                        )));
                                    }
                                    Some((_, first_values)) => {
                                        let mut max_v: Option<u64> = None;
                                        for &v in first_values {
                                            max_v = Some(max_v.map_or(v, |prev| prev.max(v)));
                                        }
                                        for slice in &slices[1..] {
                                            let values =
                                                lookup_direct(slice, *scope, name).ok_or_else(
                                                    || {
                                                        GraphError::Invalid(format!(
                                                            "graph regime slice omits port class dimension `{name}`"
                                                        ))
                                                    },
                                                )?;
                                            for &v in values {
                                                max_v = Some(max_v.map_or(v, |prev| prev.max(v)));
                                            }
                                        }
                                        match max_v {
                                            Some(v) => {
                                                let mut extents = port.extents.clone();
                                                extents[*axis] = v;
                                                port.element
                                                    .canonical_byte_len(&extents)
                                                    .map_err(|error| {
                                                        GraphError::Invalid(error.to_string())
                                                    })?
                                            }
                                            None => 0,
                                        }
                                    }
                                }
                            }
                            PortSource::Checked {
                                entry_name,
                                class_scope,
                                parameter,
                                elements,
                                dimensions,
                                eval,
                            } => {
                                let port_elements = &self.checked_elements[elements.clone()];
                                let port_dimensions = &self.checked_dimensions[dimensions.clone()];
                                let port_key: CheckedPortKey<'a> = (
                                    *eval as usize,
                                    *class_scope,
                                    parameter.as_str(),
                                    port_elements,
                                    port_dimensions,
                                );
                                if let Some(&cached) = port_cache.get(&port_key) {
                                    cached
                                } else {
                                    let mut read_dims: Option<Option<Arc<[&'static str]>>> = None;
                                    varying.clear();
                                    varying_values.clear();
                                    for (i, (name, _)) in port_dimensions.iter().enumerate() {
                                        let name_str = name.as_str();
                                        let mut first_present = None;
                                        for (s_idx, slice) in slices.iter().enumerate() {
                                            if let Some(vals) =
                                                slice.values(entry_name, *class_scope, name_str)
                                            {
                                                first_present = Some((s_idx, vals));
                                                break;
                                            }
                                        }
                                        let Some((first_s_idx, first_vals)) = first_present else {
                                            continue;
                                        };
                                        if first_s_idx > 0 {
                                            return Err(GraphError::Invalid(format!(
                                                "graph regime slice omits `{entry_name}.{parameter}` dimension `{name_str}`"
                                            )));
                                        }
                                        let used = read_dims.get_or_insert_with(|| {
                                            els.clear();
                                            els.extend(
                                                port_elements
                                                    .iter()
                                                    .map(|(n, el)| (n.as_str(), *el)),
                                            );
                                            lookup_or_probe_param_read_dims(
                                                scoped_cell,
                                                *eval,
                                                self.backend,
                                                &els,
                                                parameter.as_str(),
                                                |missing| {
                                                    port_dimensions
                                                        .iter()
                                                        .find(|(n, _)| n.as_str() == missing)
                                                        .map(|(_, v)| *v)
                                                },
                                            )
                                        });
                                        let is_read = used
                                            .as_deref()
                                            .map_or(true, |u| u.iter().any(|&d| d == name_str));
                                        if is_read {
                                            varying.push(i);
                                            varying_values.push(first_vals);
                                            for slice in &slices[1..] {
                                                let vals = slice
                                                    .values(entry_name, *class_scope, name_str)
                                                    .ok_or_else(|| {
                                                        GraphError::Invalid(format!(
                                                            "graph regime slice omits `{entry_name}.{parameter}` dimension `{name_str}`"
                                                        ))
                                                    })?;
                                                varying_values.push(vals);
                                            }
                                        } else {
                                            for slice in &slices[1..] {
                                                if slice
                                                    .values(entry_name, *class_scope, name_str)
                                                    .is_none()
                                                {
                                                    return Err(GraphError::Invalid(format!(
                                                        "graph regime slice omits `{entry_name}.{parameter}` dimension `{name_str}`"
                                                    )));
                                                }
                                            }
                                        }
                                    }
                                    let best = if varying.is_empty() {
                                        port.byte_len
                                    } else {
                                        let read_dims = read_dims.flatten();
                                        point_dims.clear();
                                        varying_slots.clear();
                                        if let Some(used) = read_dims.as_deref() {
                                            for &dim_name in used {
                                                if let Some((_, v)) = port_dimensions
                                                    .iter()
                                                    .find(|(n, _)| n.as_str() == dim_name)
                                                {
                                                    point_dims.push((dim_name, *v));
                                                }
                                            }
                                            for &idx in &varying {
                                                let name = port_dimensions[idx].0.as_str();
                                                let slot = point_dims
                                                    .iter()
                                                    .position(|(n, _)| *n == name)
                                                    .expect("varying read dim present in point_dims");
                                                varying_slots.push(slot);
                                            }
                                        } else {
                                            point_dims.extend(
                                                port_dimensions
                                                    .iter()
                                                    .map(|(name, v)| (name.as_str(), *v)),
                                            );
                                            varying_slots.extend_from_slice(&varying);
                                        }
                                        let param_prefix = param_spec_prefix_hash(
                                            *eval,
                                            self.backend,
                                            parameter.as_str(),
                                            &els,
                                        );
                                        stage_timer::time_stage(
                                            |t, ns| t.sd_port_param_spec_ns += ns,
                                            || -> Result<u64, GraphError> {
                                                // Every Seismic tensor parameter's `contract_param_spec`
                                                // evaluates `canonical_bytes(el, &[axis_exprs...])`
                                                // where each `axis_expr` is monotonically non-decreasing
                                                // in every dynamic `u64` dimension, enforced at build
                                                // time by `seismic_build::render_contract::assert_axis_expr_non_decreasing`
                                                // (the only subtracted dimensions across all entries are
                                                // static 0/1 flags such as `VK`/`PR`, which never vary
                                                // across a regime). Thus the maximum byte capacity over
                                                // any Cartesian slice box is achieved at its upper corner
                                                // `(max(axis_0), ..., max(axis_k))`.
                                                if varying.len() == 1 {
                                                    let slot = varying_slots[0];
                                                    let max_v = varying_values
                                                        .iter()
                                                        .flat_map(|vals| vals.iter().copied())
                                                        .max();
                                                    let Some(v) = max_v else {
                                                        return Ok(0);
                                                    };
                                                    point_dims[slot].1 = v;
                                                    eval_or_lookup_param_bytes(
                                                        scoped_cell,
                                                        *eval,
                                                        self.backend,
                                                        param_prefix,
                                                        &els,
                                                        parameter.as_str(),
                                                        &point_dims,
                                                    )
                                                    .map_err(|error| {
                                                        contract_error(
                                                            entry_name,
                                                            self.backend,
                                                            &[],
                                                            &point_dims,
                                                            error,
                                                        )
                                                    })
                                                } else {
                                                    let mut best = 0u64;
                                                    let n_slices = slices.len();
                                                    let mut prev_corner: Vec<u64> = Vec::new();
                                                    for s_idx in 0..n_slices {
                                                        combo.clear();
                                                        let mut empty = false;
                                                        for v_idx in 0..varying.len() {
                                                            let axis = varying_values
                                                                [v_idx * n_slices + s_idx];
                                                            let Some(&max_v) = axis.iter().max()
                                                            else {
                                                                empty = true;
                                                                break;
                                                            };
                                                            combo.push(max_v);
                                                        }
                                                        if empty || combo == prev_corner {
                                                            continue;
                                                        }
                                                        prev_corner.clear();
                                                        prev_corner.extend_from_slice(&combo);
                                                        for (&slot, &v) in
                                                            varying_slots.iter().zip(&combo)
                                                        {
                                                            point_dims[slot].1 = v;
                                                        }
                                                        let bytes = eval_or_lookup_param_bytes(
                                                            scoped_cell,
                                                            *eval,
                                                            self.backend,
                                                            param_prefix,
                                                            &els,
                                                            parameter.as_str(),
                                                            &point_dims,
                                                        )
                                                        .map_err(|error| {
                                                            contract_error(
                                                                entry_name,
                                                                self.backend,
                                                                &[],
                                                                &point_dims,
                                                                error,
                                                            )
                                                        })?;
                                                        best = best.max(bytes);
                                                    }
                                                    Ok(best)
                                                }
                                            },
                                        )?
                                    };
                                    port_cache.insert(port_key, best);
                                    best
                                }
                            }
                        };
                        port_bytes.push(max_bytes);
                    }
                    Ok(port_bytes)
                },
            )?;

            // Deduplicate node sizing across identical (eval_point, class_scope, elements, dimensions).
            type NodeCacheKey<'k> =
                (usize, Option<&'static str>, &'k [(String, Element)], &'k [(String, u64)]);
            let mut cache: HashMap<
                NodeCacheKey<'a>,
                (Vec<Option<u64>>, Vec<ScratchNeed>),
                FxBuildHasher,
            > = HashMap::with_hasher(FxBuildHasher);
            let mut result_bytes = Vec::with_capacity(self.nodes.len());
            let mut scratch_needs = Vec::with_capacity(self.nodes.len());

            let mut max_tensor_results: Vec<u64> = Vec::new();
            let mut max_scratch_bytes: Vec<u64> = Vec::new();
            let mut point_results: Vec<u64> = Vec::new();
            let mut point_scratch: Vec<u64> = Vec::new();
            let mut slice_results: Vec<u64> = Vec::new();
            let mut slice_scratch: Vec<u64> = Vec::new();
            let mut mono_varying: Vec<usize> = Vec::new();
            let mut mono_values: Vec<&[u64]> = Vec::new();
            let mut prev_corner: Vec<u64> = Vec::new();

            for (node_idx, node) in self.nodes.iter().enumerate() {
                let key: NodeCacheKey<'a> = (
                    node.eval_point as usize,
                    node.class_scope,
                    node.elements.as_slice(),
                    node.dimensions.as_slice(),
                );
                if let Some((cached_results, cached_scratch)) = cache.get(&key) {
                    result_bytes.push(cached_results.clone());
                    scratch_needs.push(cached_scratch.clone());
                    continue;
                }

                if let Some(&name) = node.shape.statics.iter().find(|&&name| {
                    slices.iter().any(|slice| {
                        slice
                            .values(node.entry_name, node.class_scope, name)
                            .is_some()
                    })
                }) {
                    return Err(GraphError::Invalid(format!(
                        "graph regime classes vary `{}`'s static dimension `{name}`",
                        node.entry_name
                    )));
                }

                mono_varying.clear();
                mono_values.clear();
                varying.clear();
                varying_values.clear();
                for (i, (name, _)) in node.dimensions.iter().enumerate() {
                    let name_str = name.as_str();
                    let mut first_present = None;
                    for (s_idx, slice) in slices.iter().enumerate() {
                        if let Some(vals) =
                            slice.values(node.entry_name, node.class_scope, name_str)
                        {
                            first_present = Some((s_idx, vals));
                            break;
                        }
                    }
                    let Some((first_s_idx, first_vals)) = first_present else {
                        continue;
                    };
                    if first_s_idx > 0 {
                        return Err(GraphError::Invalid(format!(
                            "graph regime slice omits `{}` dimension `{name_str}`",
                            node.entry_name
                        )));
                    }
                    if node.shape.sizing_dims.contains(&name_str) {
                        let (target_dims, target_vals) =
                            if node.shape.monotonic_sizing_dims.contains(&name_str) {
                                (&mut mono_varying, &mut mono_values)
                            } else {
                                (&mut varying, &mut varying_values)
                            };
                        target_dims.push(i);
                        target_vals.push(first_vals);
                        for slice in &slices[1..] {
                            let vals = slice
                                .values(node.entry_name, node.class_scope, name_str)
                                .ok_or_else(|| {
                                    GraphError::Invalid(format!(
                                        "graph regime slice omits `{}` dimension `{name_str}`",
                                        node.entry_name
                                    ))
                                })?;
                            target_vals.push(vals);
                        }
                    } else {
                        for slice in &slices[1..] {
                            if slice
                                .values(node.entry_name, node.class_scope, name_str)
                                .is_none()
                            {
                                return Err(GraphError::Invalid(format!(
                                    "graph regime slice omits `{}` dimension `{name_str}`",
                                    node.entry_name
                                )));
                            }
                        }
                    }
                }

                els.clear();
                els.extend(node.elements.iter().map(|(name, el)| (name.as_str(), *el)));
                point_dims.clear();
                point_dims.extend(node.dimensions.iter().map(|(name, v)| (name.as_str(), *v)));
                let config_key: ScratchConfigKey<'a> = (
                    node.eval_configs as usize,
                    node.elements.as_slice(),
                    node.dimensions.as_slice(),
                );
                let configs = match configs_cache.entry(config_key) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        let computed = eval_or_lookup_scratch_configs(
                            node.entry_name,
                            scoped_cell,
                            node.eval_configs,
                            self.backend,
                            &els,
                            node.shape.statics,
                            &node.dimensions,
                            &point_dims,
                        )
                        .map_err(|error| {
                            contract_error(
                                node.entry_name,
                                self.backend,
                                node.shape.statics,
                                &point_dims,
                                error,
                            )
                        })?;
                        slot.insert(computed)
                    }
                };
                let point_prefix = sizing_point_prefix_hash(node.eval_point, self.backend, &els);
                let tensor_result_count = node
                    .shape
                    .result_ranks
                    .iter()
                    .filter(|r| r.is_some())
                    .count();
                max_tensor_results.clear();
                max_tensor_results.resize(tensor_result_count, 0);
                max_scratch_bytes.clear();
                max_scratch_bytes.resize(base_scratch[node_idx].len(), 0);
                let n_slices = slices.len();
                if varying.is_empty() && mono_varying.is_empty() {
                    eval_or_lookup_sizing_point(
                        node.entry_name,
                        scoped_cell,
                        node.eval_point,
                        self.backend,
                        point_prefix,
                        &els,
                        &point_dims,
                        configs,
                        &mut max_tensor_results,
                        &mut max_scratch_bytes,
                        &mut point_results,
                        &mut point_scratch,
                    )
                    .map_err(|error| {
                        contract_error(
                            node.entry_name,
                            self.backend,
                            node.shape.statics,
                            &point_dims,
                            error,
                        )
                    })?;
                } else if varying.is_empty() {
                    // Every varying sizing dimension is proven at build time to be
                    // monotonically non-decreasing in `contract_sizing_at_point`:
                    // the maximum over each slice box is achieved at its upper corner.
                    if mono_varying.len() == 1 {
                        let slot = mono_varying[0];
                        if let Some(max_v) =
                            mono_values.iter().flat_map(|vals| vals.iter().copied()).max()
                        {
                            point_dims[slot].1 = max_v;
                            eval_or_lookup_sizing_point(
                                node.entry_name,
                                scoped_cell,
                                node.eval_point,
                                self.backend,
                                point_prefix,
                                &els,
                                &point_dims,
                                configs,
                                &mut max_tensor_results,
                                &mut max_scratch_bytes,
                                &mut point_results,
                                &mut point_scratch,
                            )
                            .map_err(|error| {
                                contract_error(
                                    node.entry_name,
                                    self.backend,
                                    node.shape.statics,
                                    &point_dims,
                                    error,
                                )
                            })?;
                        }
                    } else {
                        prev_corner.clear();
                        for s_idx in 0..n_slices {
                            combo.clear();
                            let mut empty = false;
                            for m_idx in 0..mono_varying.len() {
                                let Some(&max_v) =
                                    mono_values[m_idx * n_slices + s_idx].iter().max()
                                else {
                                    empty = true;
                                    break;
                                };
                                combo.push(max_v);
                            }
                            if empty || combo == prev_corner {
                                continue;
                            }
                            prev_corner.clear();
                            prev_corner.extend_from_slice(&combo);
                            for (&idx, &v) in mono_varying.iter().zip(&combo) {
                                point_dims[idx].1 = v;
                            }
                            eval_or_lookup_sizing_point(
                                node.entry_name,
                                scoped_cell,
                                node.eval_point,
                                self.backend,
                                point_prefix,
                                &els,
                                &point_dims,
                                configs,
                                &mut max_tensor_results,
                                &mut max_scratch_bytes,
                                &mut point_results,
                                &mut point_scratch,
                            )
                            .map_err(|error| {
                                contract_error(
                                    node.entry_name,
                                    self.backend,
                                    node.shape.statics,
                                    &point_dims,
                                    error,
                                )
                            })?;
                        }
                    }
                } else if varying.len() == 1 && mono_varying.is_empty() {
                    let slot = varying[0];
                    let values: &[u64] = if varying_values.len() == 1 {
                        varying_values[0]
                    } else {
                        seen_single.clear();
                        for &vals in &varying_values {
                            for &v in vals {
                                if !seen_single.contains(&v) {
                                    seen_single.push(v);
                                }
                            }
                        }
                        &seen_single
                    };
                    eval_or_lookup_node_cover_points(
                        node.entry_name,
                        scoped_cell,
                        node.eval_point,
                        self.backend,
                        point_prefix,
                        &els,
                        &mut point_dims,
                        slot,
                        values,
                        configs,
                        &mut max_tensor_results,
                        &mut max_scratch_bytes,
                        &mut point_results,
                        &mut point_scratch,
                        &mut missed_v,
                    )
                    .map_err(|error| {
                        contract_error(
                            node.entry_name,
                            self.backend,
                            node.shape.statics,
                            &point_dims,
                            error,
                        )
                    })?;
                } else if varying.len() == 1 {
                    // Single non-monotonic varying dim plus one or more monotonic dims:
                    // collapse each monotonic dim to its per-slice maximum and cover
                    // only the non-monotonic axis via `eval_or_lookup_node_cover_points`.
                    let slot = varying[0];
                    if n_slices == 1 {
                        let nm_vals = varying_values[0];
                        if !nm_vals.is_empty() {
                            let mut empty = false;
                            for (m_idx, &mono_slot) in mono_varying.iter().enumerate() {
                                let Some(&max_v) = mono_values[m_idx].iter().max() else {
                                    empty = true;
                                    break;
                                };
                                point_dims[mono_slot].1 = max_v;
                            }
                            if !empty {
                                eval_or_lookup_node_cover_points(
                                    node.entry_name,
                                    scoped_cell,
                                    node.eval_point,
                                    self.backend,
                                    point_prefix,
                                    &els,
                                    &mut point_dims,
                                    slot,
                                    nm_vals,
                                    configs,
                                    &mut max_tensor_results,
                                    &mut max_scratch_bytes,
                                    &mut point_results,
                                    &mut point_scratch,
                                    &mut missed_v,
                                )
                                .map_err(|error| {
                                    contract_error(
                                        node.entry_name,
                                        self.backend,
                                        node.shape.statics,
                                        &point_dims,
                                        error,
                                    )
                                })?;
                            }
                        }
                    } else {
                        prev_corner.clear();
                        let mut prev_nm: Option<&[u64]> = None;
                        for s_idx in 0..n_slices {
                            let nm_vals = varying_values[s_idx];
                            if nm_vals.is_empty() {
                                continue;
                            }
                            combo.clear();
                            let mut empty = false;
                            for m_idx in 0..mono_varying.len() {
                                let Some(&max_v) =
                                    mono_values[m_idx * n_slices + s_idx].iter().max()
                                else {
                                    empty = true;
                                    break;
                                };
                                combo.push(max_v);
                            }
                            if empty || (combo == prev_corner && prev_nm == Some(nm_vals)) {
                                continue;
                            }
                            prev_corner.clear();
                            prev_corner.extend_from_slice(&combo);
                            prev_nm = Some(nm_vals);
                            for (&mono_slot, &v) in mono_varying.iter().zip(&combo) {
                                point_dims[mono_slot].1 = v;
                            }
                            slice_results.clear();
                            slice_results.resize(tensor_result_count, 0);
                            slice_scratch.clear();
                            slice_scratch.resize(base_scratch[node_idx].len(), 0);
                            eval_or_lookup_node_cover_points(
                                node.entry_name,
                                scoped_cell,
                                node.eval_point,
                                self.backend,
                                point_prefix,
                                &els,
                                &mut point_dims,
                                slot,
                                nm_vals,
                                configs,
                                &mut slice_results,
                                &mut slice_scratch,
                                &mut point_results,
                                &mut point_scratch,
                                &mut missed_v,
                            )
                            .map_err(|error| {
                                contract_error(
                                    node.entry_name,
                                    self.backend,
                                    node.shape.statics,
                                    &point_dims,
                                    error,
                                )
                            })?;
                            for (dst, &src) in max_tensor_results.iter_mut().zip(&slice_results) {
                                *dst = (*dst).max(src);
                            }
                            for (dst, &src) in max_scratch_bytes.iter_mut().zip(&slice_scratch) {
                                *dst = (*dst).max(src);
                            }
                        }
                    }
                } else {
                    for s_idx in 0..n_slices {
                        let mut empty = false;
                        for (m_idx, &mono_slot) in mono_varying.iter().enumerate() {
                            let Some(&max_v) = mono_values[m_idx * n_slices + s_idx].iter().max()
                            else {
                                empty = true;
                                break;
                            };
                            point_dims[mono_slot].1 = max_v;
                        }
                        if empty {
                            continue;
                        }
                        axes.clear();
                        for v_idx in 0..varying.len() {
                            axes.push(varying_values[v_idx * n_slices + s_idx]);
                        }
                        combo.clear();
                        for_each_combination(&axes, &mut combo, &mut |values| {
                            for (&idx, &v) in varying.iter().zip(values) {
                                point_dims[idx].1 = v;
                            }
                            eval_or_lookup_sizing_point(
                                node.entry_name,
                                scoped_cell,
                                node.eval_point,
                                self.backend,
                                point_prefix,
                                &els,
                                &point_dims,
                                configs,
                                &mut max_tensor_results,
                                &mut max_scratch_bytes,
                                &mut point_results,
                                &mut point_scratch,
                            )
                            .map_err(|error| {
                                contract_error(
                                    node.entry_name,
                                    self.backend,
                                    node.shape.statics,
                                    &point_dims,
                                    error,
                                )
                            })?;
                            Ok(())
                        })?;
                    }
                }
                let mut tensor_iter = max_tensor_results.iter().copied();
                let node_results: Vec<Option<u64>> = node
                    .shape
                    .result_ranks
                    .iter()
                    .map(|rank| {
                        rank.map(|_| {
                            tensor_iter
                                .next()
                                .expect("contract_sizing_at_point results match result_ranks")
                        })
                    })
                    .collect();
                let node_scratch: Vec<ScratchNeed> = base_scratch[node_idx]
                    .iter()
                    .zip(max_scratch_bytes.iter().copied())
                    .map(|(need, bytes)| ScratchNeed { bytes, ..*need })
                    .collect();

                cache.insert(key, (node_results.clone(), node_scratch.clone()));
                result_bytes.push(node_results);
                scratch_needs.push(node_scratch);
            }

            Ok((port_bytes, result_bytes, scratch_needs))
        })
    }
}

impl GraphDraft for SizingDraft {
    type Plan = SizingOutline;
    type Binding<'a, E: EntryContract + 'a> = &'a [(&'a str, Element)];
    type Ranged<'a, E: EntryContract + 'a> = &'a [(&'a str, Element)];
    type Forms = ();

    fn pick<'a, E: EntryContract + 'a>(
        ranged: Self::Ranged<'a, E>,
        _rows: &Dim,
    ) -> Self::Binding<'a, E> {
        ranged
    }

    fn first<'a, E: EntryContract + 'a>(ranged: Self::Ranged<'a, E>) -> Self::Binding<'a, E> {
        ranged
    }

    fn port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        let values: Vec<u64> = extents
            .iter()
            .map(|extent| Dim::from(extent.clone()).fixed())
            .collect();
        let byte_len = element
            .canonical_byte_len(&values)
            .map_err(|error| GraphError::Invalid(error.to_string()))?;
        Ok(self.push_port(SizingPort {
            element,
            extents: values,
            byte_len,
            local: false,
            owned_input: false,
            is_weight: false,
            source: PortSource::Direct { class_extent: None },
        }))
    }

    fn weight_port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        let values: Vec<u64> = extents
            .iter()
            .map(|extent| Dim::from(extent.clone()).fixed())
            .collect();
        let byte_len = element
            .canonical_byte_len(&values)
            .map_err(|error| GraphError::Invalid(error.to_string()))?;
        Ok(self.push_port(SizingPort {
            element,
            extents: values,
            byte_len,
            local: false,
            owned_input: false,
            is_weight: true,
            source: PortSource::Direct { class_extent: None },
        }))
    }

    fn set_class_scope(&mut self, scope: Option<&'static str>) {
        self.class_scope = scope;
    }

    fn port_with_class_extent<D: Clone>(
        &mut self,
        element: Element,
        extents: &[D],
        extent_axis: usize,
        class_dimension: &'static str,
    ) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        if extent_axis >= extents.len() {
            return Err(GraphError::Invalid(format!(
                "graph port extent axis {extent_axis} is absent"
            )));
        }
        let values: Vec<u64> = extents
            .iter()
            .map(|extent| Dim::from(extent.clone()).fixed())
            .collect();
        let byte_len = element
            .canonical_byte_len(&values)
            .map_err(|error| GraphError::Invalid(error.to_string()))?;
        Ok(self.push_port(SizingPort {
            element,
            extents: values,
            byte_len,
            local: false,
            owned_input: false,
            is_weight: false,
            source: PortSource::Direct {
                class_extent: Some((extent_axis, class_dimension, self.class_scope)),
            },
        }))
    }

    fn local_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.checked_port::<E>(entry, parameter, dimensions, true, false)
    }

    fn input_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.checked_port::<E>(entry, parameter, dimensions, true, true)
    }

    fn port_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.checked_port::<E>(entry, parameter, dimensions, false, false)
    }

    fn enqueue<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        self.enqueue_node::<E>(entry, dimensions, args)
    }

    fn enqueue_under<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
        _port: &NativePort,
        _index: u64,
        _encoded: LaunchEncoding,
    ) -> Result<E::WorkflowResults, GraphError> {
        self.enqueue_node::<E>(entry, dimensions, args)
    }

    fn export(&mut self, result: &WorkflowTensor) -> Result<(), GraphError> {
        let reference = result.raw_ref();
        if !self.exports.contains(&reference) {
            self.exports.push(reference);
        }
        Ok(())
    }

    fn share(&mut self, other: &WorkflowTensor, primary: &WorkflowTensor) -> Result<(), GraphError> {
        let other = other.raw_ref();
        let primary = primary.raw_ref();
        let node = self
            .nodes
            .get_mut(other.node as usize)
            .ok_or_else(|| GraphError::Invalid("shared result node out of bounds".to_owned()))?;
        node.share = Some((other.result, primary));
        Ok(())
    }

    fn store(&mut self, result: &WorkflowTensor, port: &NativePort) -> Result<(), GraphError> {
        self.stores.push((result.raw_ref(), port.ordinal()));
        Ok(())
    }

    fn operand_carry(
        &mut self,
        rows: u64,
        width: &Dim,
        activation: Element,
    ) -> Result<OperandCarry, GraphError> {
        let [(rows_repr, rows_extents), (quantized_repr, quantized_extents)] =
            seismic::sizing::operand_carry_ports(self.backend, Dim::from(rows), *width);
        let rows_port = self.port(Element::from_id(rows_repr), &rows_extents)?;
        let quantized_port = self.port(Element::from_id(quantized_repr), &quantized_extents)?;
        Ok(OperandCarry::from_ports(
            rows_port,
            quantized_port,
            activation,
            self.backend,
            [rows, width.fixed()],
        ))
    }

    fn carry_out(
        &mut self,
        result: &WorkflowTensor,
        carry: &OperandCarry,
        (): (),
    ) -> Result<(), GraphError> {
        self.carries.push(OperandCarrySpec {
            rows: carry.rows.ordinal(),
            quantized: carry.quantized.ordinal(),
            activation: carry.activation.id(),
            side: OperandCarrySide::Out {
                result: result.raw_ref(),
            },
        });
        Ok(())
    }

    fn carry_in(
        &mut self,
        hidden: &NativePort,
        carry: &OperandCarry,
        (): (),
        norm: &NativePort,
        epsilon: f32,
    ) -> Result<(), GraphError> {
        self.carries.push(OperandCarrySpec {
            rows: carry.rows.ordinal(),
            quantized: carry.quantized.ordinal(),
            activation: carry.activation.id(),
            side: OperandCarrySide::In {
                hidden: hidden.ordinal(),
                norm: norm.ordinal(),
                epsilon: epsilon.to_bits(),
            },
        });
        Ok(())
    }

    fn timeline<N: seismic::timeline::TimelineNodeKind>(
        &mut self,
        _node: N,
    ) -> NativeTimelinePort<N::Role> {
        let ordinal = u32::try_from(self.timelines.len()).expect("timeline ordinal exhausted");
        self.timelines.push((self.nodes.len(), N::NODE));
        make_timeline_port(self.identity, ordinal)
    }

    fn span_mark(&mut self) -> usize {
        self.marks.push(self.nodes.len());
        self.marks.len() - 1
    }

    fn seal(self) -> Result<Self::Plan, GraphError> {
        let [outline]: [SizingOutline; 1] = self
            .certify_each(&[vec![NativeGraphClassSlice::new()]])?
            .try_into()
            .unwrap_or_else(|_| unreachable!("one slice set certifies to one outline"));
        Ok(outline)
    }
}
