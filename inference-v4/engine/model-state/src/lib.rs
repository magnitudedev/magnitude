//! Family-neutral accepted sequence state and transactional history ownership.
//! Owned transactions can cross a submission boundary. Their callers reconcile
//! only after physical completion has been observed.
mod advance;
mod codec;
mod layout;

pub use advance::{
    CodecConversionStep, OwnedAdvanceBindings, OwnedAdvanceResolution, OwnedCodecAdvance,
    OwnedCodecBindings, OwnedCompaction, OwnedCompactionBindings, OwnedCompactionPreparation,
    OwnedRepairAdvance, OwnedStateAdvance,
};

pub use codec::{
    Codec, CodecIdentity, CodecSpec, ComponentDescriptor, KvCodec, LayerRef, LayoutError,
    PlaneDescriptor, PlaneName, VectorKind,
};
pub use layout::ModelStateLayout;

use seismic::{DType, Device, Element, Tensor};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    rc::Rc,
};

pub const MAX_VISIBLE_SEGMENTS: usize = 16;

/// Failures owned by the state store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Request(String),
    Tensor(seismic::TensorError),
    Layout(LayoutError),
    Capacity { required: u64, available_bytes: u64 },
    BanksExhausted { capacity: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(message) => f.write_str(message),
            Self::Tensor(error) => write!(f, "{error}"),
            Self::Layout(error) => write!(f, "{error}"),
            Self::Capacity {
                required,
                available_bytes,
            } => write!(
                f,
                "state capacity requires {required} bytes but {available_bytes} bytes are available"
            ),
            Self::BanksExhausted { capacity } => {
                write!(f, "all {capacity} recurrent banks have live claims")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Request(message.to_owned())
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Request(message)
    }
}

impl From<seismic::TensorError> for Error {
    fn from(error: seismic::TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<LayoutError> for Error {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

fn element(dtype: DType) -> Element {
    match dtype {
        DType::F32 => Element::f32(),
        DType::F16 => Element::f16(),
        DType::BF16 => Element::bf16(),
        DType::I32 => Element::i32(),
        DType::U32 => Element::u32(),
        DType::Bool => Element::bool(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}
impl ComponentSpec {
    fn bytes(&self) -> Result<usize, String> {
        if self.shape.is_empty() || self.shape.contains(&0) || !self.dtype.is_float() {
            return Err("state components require nonempty floating tensors".into());
        }
        self.shape
            .iter()
            .try_fold(self.dtype.bytes() as usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| "state component allocation overflow".into())
    }
}

/// One allocated attention-history plane. `base_row` is stable and currently
/// always zero because all components share the arena-global row domain.
#[derive(Clone)]
pub struct PlaneBuffer {
    pub plane_index: usize,
    pub component_index: usize,
    pub layer: LayerRef,
    pub vector: VectorKind,
    pub name: PlaneName,
    pub row_bytes: usize,
    pub base_row: usize,
    pub buffer: Tensor,
}

/// Free history rows as address-ordered, coalesced `(start, count)` holes.
///
/// Placement keeps every sequence's history in few segments however requests
/// interleave: a sequence grows in place into the hole that begins at its
/// history end, and a sequence that cannot grow in place (a fresh sequence, a
/// fork whose sibling took the rows, or a neighbour boundary) starts at the
/// middle of the largest hole, leaving the rows before it as growth room for
/// the history that ends there. Only a hole at row 0 has no such history and
/// is filled from its start.
struct Arena {
    free: Vec<(usize, usize)>,
}
impl Arena {
    fn available(&self) -> usize {
        self.free.iter().map(|(_, count)| count).sum()
    }

    /// Claim `count` rows, in logical order, for a sequence whose history ends
    /// at row `after`. The caller has checked `count <= self.available()`.
    fn claim(&mut self, after: Option<usize>, count: usize) -> Vec<(usize, usize)> {
        let mut claimed = Vec::new();
        let mut remaining = count;
        if let Some(end) = after {
            if let Some(hole) = self.free.iter().position(|(start, _)| *start == end) {
                let taken = self.free[hole].1.min(remaining);
                claimed.push(self.take(hole, 0, taken));
                remaining -= taken;
            }
        }
        while remaining > 0 {
            let hole = (0..self.free.len())
                .max_by_key(|&hole| (self.free[hole].1, std::cmp::Reverse(self.free[hole].0)))
                .expect("claimed rows never exceed the available rows");
            let (start, size) = self.free[hole];
            let taken = size.min(remaining);
            let offset = if start == 0 { 0 } else { (size - taken) / 2 };
            claimed.push(self.take(hole, offset, taken));
            remaining -= taken;
        }
        claimed
    }

    /// Remove `count` rows at `offset` within hole `hole`, keeping the free
    /// list address-ordered.
    fn take(&mut self, hole: usize, offset: usize, count: usize) -> (usize, usize) {
        let (start, size) = self.free[hole];
        let before = (offset > 0).then_some((start, offset));
        let after =
            (offset + count < size).then(|| (start + offset + count, size - offset - count));
        self.free
            .splice(hole..=hole, before.into_iter().chain(after));
        (start + offset, count)
    }
}
struct Extent {
    arena: Rc<RefCell<Arena>>,
    start: usize,
    count: usize,
}
type ExtentSet = Vec<Rc<Extent>>;
impl Drop for Extent {
    fn drop(&mut self) {
        if self.count == 0 {
            return;
        }
        let mut arena = self.arena.borrow_mut();
        arena.free.push((self.start, self.count));
        arena.free.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (start, count) in arena.free.drain(..) {
            if let Some((s, n)) = merged.last_mut() {
                if *s + *n == start {
                    *n += count;
                    continue;
                }
            }
            merged.push((start, count));
        }
        arena.free = merged;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankCapacity {
    pub active: usize,
    pub in_flight: usize,
    pub retained: usize,
}

impl BankCapacity {
    pub fn total(self) -> Result<usize, Error> {
        if self.active == 0 || self.in_flight == 0 {
            return Err(Error::Request(
                "bank pool requires positive active and in-flight capacity".into(),
            ));
        }
        self.active
            .checked_add(self.in_flight)
            .and_then(|value| value.checked_add(self.retained))
            .ok_or_else(|| Error::Request("bank pool capacity overflow".into()))
    }

    /// Writable banks plus one permanently pristine seed for new sequences.
    pub fn storage_total(self) -> Result<usize, Error> {
        self.total()?
            .checked_add(1)
            .ok_or_else(|| Error::Request("zero seed bank count overflow".into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateAllocationTrace {
    pub context_capacity: usize,
    pub history_capacity: usize,
    pub history_row_bytes: u64,
    pub history_bytes: u64,
    pub bank_capacity: BankCapacity,
    pub recurrent_bank_bytes: u64,
    pub zero_seed_bytes: u64,
    pub recurrent_pool_bytes: u64,
}

/// The permanently pristine bank every new sequence starts from. It is never
/// on the free list, so no advance can name it as its successor.
pub const ZERO_SEED_BANK: usize = 0;

struct BankPoolInner {
    capacity: usize,
    free: RefCell<Vec<usize>>,
}

/// A claim on one arena row of every recurrent component. Dropping the last
/// handle returns the row to the free list; the storage itself stays in the
/// store's arenas.
struct BankClaim {
    pool: Rc<BankPoolInner>,
    index: usize,
}

impl Drop for BankClaim {
    fn drop(&mut self) {
        if self.index != ZERO_SEED_BANK {
            self.pool.free.borrow_mut().push(self.index);
        }
    }
}

#[derive(Clone)]
struct BankHandle(Rc<BankClaim>);

impl BankHandle {
    fn index(&self) -> usize {
        self.0.index
    }
}

struct BankPool {
    inner: Rc<BankPoolInner>,
}

impl BankPool {
    fn new(capacity: BankCapacity) -> Result<(Self, BankHandle), Error> {
        let count = capacity.total()?;
        let storage_count = capacity.storage_total()?;
        let inner = Rc::new(BankPoolInner {
            capacity: count,
            free: RefCell::new((1..storage_count).rev().collect()),
        });
        let seed = BankHandle(Rc::new(BankClaim {
            pool: inner.clone(),
            index: ZERO_SEED_BANK,
        }));
        Ok((Self { inner }, seed))
    }

    fn acquire(&self) -> Result<BankHandle, Error> {
        let index = self
            .inner
            .free
            .borrow_mut()
            .pop()
            .ok_or(Error::BanksExhausted {
                capacity: self.inner.capacity,
            })?;
        Ok(BankHandle(Rc::new(BankClaim {
            pool: self.inner.clone(),
            index,
        })))
    }

    fn available(&self) -> usize {
        self.inner.free.borrow().len()
    }
}

/// One zero-initialized arena per recurrent component: `[banks, ..shape]`.
fn allocate_arenas(
    device: &Device,
    specs: &[ComponentSpec],
    banks: usize,
) -> Result<Vec<Tensor>, Error> {
    specs
        .iter()
        .map(|spec| {
            let extents = std::iter::once(banks)
                .chain(spec.shape.iter().copied())
                .map(|extent| {
                    u64::try_from(extent)
                        .map_err(|_| Error::Request("recurrent arena extent exceeds u64".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::zeros(device, element(spec.dtype), &extents).map_err(Error::from)
        })
        .collect()
}
/// History rows are a shared arena; recurrent components are arenas of banks,
/// and each accepted version is one immutable bank. A checkpoint retains both
/// without copying tensor contents.
pub struct StateStore {
    device: Rc<Device>,
    context_capacity: usize,
    history_capacity: usize,
    components: Vec<ComponentDescriptor>,
    total_history_row_bytes: u64,
    bank_capacity: BankCapacity,
    recurrent_bank_bytes: u64,
    component_specs: Vec<ComponentSpec>,
    recurrent: Vec<Tensor>,
    history: RefCell<Option<Vec<Tensor>>>,
    arena: Rc<RefCell<Arena>>,
    banks: BankPool,
    zero_seed: BankHandle,
    owners: Cell<usize>,
}
impl StateStore {
    pub fn new(
        device: Rc<Device>,
        context_capacity: usize,
        history_capacity: usize,
        components: Vec<ComponentDescriptor>,
        component_specs: Vec<ComponentSpec>,
        bank_capacity: BankCapacity,
    ) -> Result<Rc<Self>, Error> {
        if context_capacity == 0 || context_capacity > history_capacity {
            return Err(Error::Request(
                "history capacity must fit a positive sequence context".into(),
            ));
        }
        let total_history_row_bytes = validate_history_layout(&components, history_capacity)?;
        for spec in &component_specs {
            spec.bytes().map_err(Error::Request)?;
        }
        let recurrent_bank_bytes = component_specs.iter().try_fold(0_u64, |total, spec| {
            total
                .checked_add(
                    u64::try_from(spec.bytes().map_err(Error::Request)?)
                        .map_err(|_| Error::Request("recurrent bank bytes exceed u64".into()))?,
                )
                .ok_or_else(|| Error::Request("recurrent bank byte count overflow".into()))
        })?;
        let (banks, zero_seed) = BankPool::new(bank_capacity)?;
        let recurrent = allocate_arenas(&device, &component_specs, bank_capacity.storage_total()?)?;
        Ok(Rc::new(Self {
            device,
            context_capacity,
            history_capacity,
            components,
            total_history_row_bytes,
            bank_capacity,
            recurrent_bank_bytes,
            component_specs,
            recurrent,
            history: RefCell::new(None),
            arena: Rc::new(RefCell::new(Arena {
                free: vec![(0, history_capacity)],
            })),
            banks,
            zero_seed,
            owners: Cell::new(0),
        }))
    }
    pub fn component_specs(&self) -> &[ComponentSpec] {
        &self.component_specs
    }
    pub fn has_recurrent_components(&self) -> bool {
        !self.component_specs.is_empty()
    }
    /// One arena per recurrent component, `[banks, ..component shape]`. A
    /// bank index selects the same row of every arena. Kernels read a
    /// sequence's accepted bank and write only an advance's successor bank;
    /// they never write an accepted bank or [`ZERO_SEED_BANK`].
    pub fn recurrent_arenas(&self) -> &[Tensor] {
        &self.recurrent
    }
    /// Banks in each recurrent arena, including the zero seed.
    pub fn recurrent_bank_count(&self) -> Result<usize, Error> {
        self.bank_capacity.storage_total()
    }
    pub fn history_capacity(&self) -> usize {
        self.history_capacity
    }
    pub fn history_components(&self) -> &[ComponentDescriptor] {
        &self.components
    }
    pub fn total_history_row_bytes(&self) -> u64 {
        self.total_history_row_bytes
    }
    pub fn total_history_bytes(&self) -> u64 {
        self.total_history_row_bytes * self.history_capacity as u64
    }
    pub fn allocation_trace(&self) -> Result<StateAllocationTrace, Error> {
        let zero_seed_bytes = self.recurrent_bank_bytes;
        let recurrent_pool_bytes = self
            .recurrent_bank_bytes
            .checked_mul(
                u64::try_from(self.bank_capacity.storage_total()?)
                    .map_err(|_| Error::Request("bank capacity exceeds u64".into()))?,
            )
            .ok_or_else(|| Error::Request("recurrent pool byte count overflow".into()))?;
        Ok(StateAllocationTrace {
            context_capacity: self.context_capacity,
            history_capacity: self.history_capacity,
            history_row_bytes: self.total_history_row_bytes,
            history_bytes: self.total_history_bytes(),
            bank_capacity: self.bank_capacity,
            recurrent_bank_bytes: self.recurrent_bank_bytes,
            zero_seed_bytes,
            recurrent_pool_bytes,
        })
    }
    pub fn history_allocated(&self) -> bool {
        self.history.borrow().is_some()
    }
    pub fn history_planes(&self) -> Result<Vec<PlaneBuffer>, Error> {
        if self.history.borrow().is_none() {
            let buffers = self
                .components
                .iter()
                .flat_map(ComponentDescriptor::planes)
                .map(|plane| {
                    let mut extents = Vec::with_capacity(plane.row_extents.len() + 1);
                    extents.push(self.history_capacity as u64);
                    extents.extend(plane.row_extents.iter().map(|extent| *extent as u64));
                    Tensor::zeros(&self.device, element(plane.dtype), &extents).map_err(Error::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            *self.history.borrow_mut() = Some(buffers);
        }
        let history = self.history.borrow();
        let tensors = history.as_ref().unwrap();
        Ok(self
            .components
            .iter()
            .enumerate()
            .flat_map(|(component_index, component)| {
                component
                    .planes()
                    .iter()
                    .map(move |plane| (component_index, component.layer, plane))
            })
            .zip(tensors.iter())
            .enumerate()
            .map(
                |(plane_index, ((component_index, layer, plane), buffer))| PlaneBuffer {
                    plane_index,
                    component_index,
                    layer,
                    vector: plane.vector,
                    name: plane.name,
                    row_bytes: plane.row_bytes,
                    base_row: 0,
                    buffer: buffer.clone(),
                },
            )
            .collect())
    }
    pub fn occupied_rows(&self) -> usize {
        self.history_capacity - self.arena.borrow().available()
    }
    /// Recurrent bank bytes returned to the pool by closing the selected
    /// states. A bank is reclaimed only when every claim on it belongs to the
    /// selection: shared checkpoints, unselected descendants, and in-flight
    /// advances prevent reclaim, and the zero seed is never reclaimed.
    pub fn reclaimable(self: &Rc<Self>, states: &[&SequenceState]) -> Result<usize, Error> {
        if states.iter().any(|state| !Rc::ptr_eq(self, &state.store)) {
            return Err(Error::Request(
                "reclamation requires states from this store".into(),
            ));
        }
        let mut distinct: Vec<&SequenceState> = Vec::with_capacity(states.len());
        for state in states {
            if !distinct.iter().any(|seen| std::ptr::eq(*seen, *state)) {
                distinct.push(state);
            }
        }
        let mut banks = 0usize;
        let mut counted: Vec<*const BankClaim> = Vec::new();
        for state in &distinct {
            let claim = Rc::as_ptr(&state.bank.0);
            if state.bank.index() == ZERO_SEED_BANK || counted.contains(&claim) {
                continue;
            }
            counted.push(claim);
            let selected = distinct
                .iter()
                .filter(|other| Rc::ptr_eq(&other.bank.0, &state.bank.0))
                .count();
            if selected == Rc::strong_count(&state.bank.0) {
                banks += 1;
            }
        }
        let bytes = self
            .recurrent_bank_bytes
            .checked_mul(banks as u64)
            .ok_or_else(|| Error::Request("reclaimable state bytes overflow".into()))?;
        usize::try_from(bytes)
            .map_err(|_| Error::Request("reclaimable state bytes exceed host range".into()))
    }
    pub fn idle(&self) -> bool {
        self.owners.get() == 0
    }

    pub fn available_banks(&self) -> usize {
        self.banks.available()
    }
    pub fn available_rows(&self) -> usize {
        if self.components.is_empty() {
            return usize::MAX;
        }
        self.arena.borrow().available()
    }
    /// Drop the store's arena allocations when no sequence/checkpoint owns them.
    /// External completion/buffer pins may still retain physical storage.
    pub fn release_idle(&self) -> Result<usize, Error> {
        if !self.idle() {
            return Ok(0);
        }
        self.history.borrow_mut().take().map_or(Ok(0), |v| {
            usize::try_from(Tensor::reclaimable_bytes(v.iter())?)
                .map_err(|_| Error::Request("reclaimable history bytes exceed host range".into()))
        })
    }
    pub fn create(self: &Rc<Self>) -> Result<SequenceState, Error> {
        let bank = self.zero_seed.clone();
        self.owners.set(self.owners.get() + 1);
        Ok(SequenceState {
            store: self.clone(),
            position: 0,
            expected_end: 0,
            history_start: 0,
            retained_start: 0,
            extents: vec![],
            bank,
        })
    }
    /// Reserve `count` rows, in logical order, for a sequence whose history
    /// ends at row `after` (see [`Arena`] for placement).
    fn reserve(&self, after: Option<usize>, count: usize) -> Result<Vec<Rc<Extent>>, Error> {
        if self.components.is_empty() {
            return Ok(vec![]);
        }
        let mut arena = self.arena.borrow_mut();
        let available_rows = arena.available();
        if available_rows < count {
            let capacity = self.history_capacity as u64;
            let total_bytes = self.total_history_bytes();
            return Err(Error::Capacity {
                required: capacity_charge(count, total_bytes, capacity, true)?,
                available_bytes: capacity_charge(available_rows, total_bytes, capacity, false)?,
            });
        }
        Ok(arena
            .claim(after, count)
            .into_iter()
            .map(|(start, count)| {
                Rc::new(Extent {
                    arena: self.arena.clone(),
                    start,
                    count,
                })
            })
            .collect())
    }

    fn reserve_contiguous(&self, count: usize) -> Option<Rc<Extent>> {
        if count == 0 || self.components.is_empty() {
            return None;
        }
        let mut arena = self.arena.borrow_mut();
        let slot = arena.free.iter().position(|(_, size)| *size >= count)?;
        let (start, size) = arena.free[slot];
        if size == count {
            arena.free.remove(slot);
        } else {
            arena.free[slot] = (start + count, size - count);
        }
        Some(Rc::new(Extent {
            arena: self.arena.clone(),
            start,
            count,
        }))
    }
}

fn validate_history_layout(
    components: &[ComponentDescriptor],
    history_capacity: usize,
) -> Result<u64, LayoutError> {
    let mut layers = BTreeSet::new();
    let mut total_row_bytes = 0u64;
    for component in components {
        if !layers.insert(component.layer) {
            return Err(LayoutError::DuplicateLayer(component.layer));
        }
        for plane in component.planes() {
            let row_bytes = u64::try_from(plane.row_bytes)
                .map_err(|_| LayoutError::ArithmeticOverflow("plane row bytes"))?;
            row_bytes
                .checked_mul(history_capacity as u64)
                .ok_or(LayoutError::ArithmeticOverflow("history plane capacity"))?;
            total_row_bytes = total_row_bytes
                .checked_add(row_bytes)
                .ok_or(LayoutError::ArithmeticOverflow("history row bytes"))?;
        }
    }
    total_row_bytes
        .checked_mul(history_capacity as u64)
        .ok_or(LayoutError::ArithmeticOverflow("total history capacity"))?;
    Ok(total_row_bytes)
}

fn capacity_charge(
    rows: usize,
    total_bytes: u64,
    history_capacity: u64,
    round_up: bool,
) -> Result<u64, LayoutError> {
    let numerator = (rows as u64)
        .checked_mul(total_bytes)
        .ok_or(LayoutError::ArithmeticOverflow("state capacity charge"))?;
    Ok(if round_up {
        numerator.div_ceil(history_capacity)
    } else {
        numerator / history_capacity
    })
}
pub struct SequenceState {
    store: Rc<StateStore>,
    position: usize,
    expected_end: usize,
    history_start: usize,
    retained_start: usize,
    extents: Vec<Rc<Extent>>,
    bank: BankHandle,
}
impl Drop for SequenceState {
    fn drop(&mut self) {
        self.store.owners.set(self.store.owners.get() - 1);
    }
}
impl SequenceState {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        Rc::ptr_eq(&self.store, store)
    }

    pub fn expected_end(&self) -> usize {
        self.expected_end
    }
    /// The accepted recurrent bank: the row of every recurrent arena that
    /// holds this sequence's state.
    pub fn bank_index(&self) -> usize {
        self.bank.index()
    }
    pub fn anticipate(&mut self, position: usize) -> Result<(), String> {
        if position > self.store.context_capacity {
            return Err("anticipated position exceeds context capacity".into());
        }
        self.expected_end = self.expected_end.max(position);
        Ok(())
    }
    /// The arena row just past this sequence's last accepted row: where its
    /// next rows continue its history without a new segment.
    fn history_end(&self) -> Option<usize> {
        self.extents
            .last()
            .map(|extent| extent.start + extent.count)
    }
    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        let mut skip = self.history_start - self.retained_start;
        let mut spans: Vec<(usize, usize)> = vec![];
        for extent in &self.extents {
            let omitted = skip.min(extent.count);
            skip -= omitted;
            let (start, count) = (extent.start + omitted, extent.count - omitted);
            if count == 0 {
                continue;
            }
            if let Some((s, n)) = spans.last_mut() {
                if *s + *n == start {
                    *n += count;
                    continue;
                }
            }
            spans.push((start, count));
        }
        spans
    }
    pub fn trim_history(&mut self, before: usize) -> Result<(), String> {
        if before < self.history_start || before > self.position {
            return Err("history trim must lie within accepted logical positions".into());
        }
        let mut retained = self.retained_start;
        let mut count = 0;
        for extent in &self.extents {
            if retained + extent.count > before {
                break;
            }
            retained += extent.count;
            count += 1;
        }
        self.extents.drain(..count);
        self.retained_start = retained;
        self.history_start = before;
        Ok(())
    }

    pub fn compaction_needed(&self) -> bool {
        self.history_ranges().len() > MAX_VISIBLE_SEGMENTS
    }

    pub fn checkpoint(&self) -> StateCheckpoint {
        self.store.owners.set(self.store.owners.get() + 1);
        StateCheckpoint {
            store: self.store.clone(),
            position: self.position,
            history_start: self.history_start,
            retained_start: self.retained_start,
            extents: self.extents.clone(),
            bank: self.bank.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneCopy {
    pub plane_index: usize,
    pub from: Vec<usize>,
    pub to: Vec<usize>,
}

pub struct StateCheckpoint {
    store: Rc<StateStore>,
    position: usize,
    history_start: usize,
    retained_start: usize,
    extents: Vec<Rc<Extent>>,
    bank: BankHandle,
}
impl Drop for StateCheckpoint {
    fn drop(&mut self) {
        self.store.owners.set(self.store.owners.get() - 1);
    }
}
impl StateCheckpoint {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn bank_index(&self) -> usize {
        self.bank.index()
    }
    /// Physical resource bytes pinned by this checkpoint. History charges the
    /// complete allocated extents (including trimmed rows that cannot be
    /// released while the extent is shared); recurrent state charges the one
    /// bank retained by the checkpoint.
    pub fn retained_bytes(&self) -> Result<u64, String> {
        let history_rows = self.extents.iter().try_fold(0_u64, |total, extent| {
            total
                .checked_add(
                    u64::try_from(extent.count)
                        .map_err(|_| "checkpoint history rows exceed the byte domain")?,
                )
                .ok_or_else(|| "checkpoint history row count overflow".to_owned())
        })?;
        let history = history_rows
            .checked_mul(self.store.total_history_row_bytes)
            .ok_or("checkpoint history byte count overflow")?;
        history
            .checked_add(self.store.recurrent_bank_bytes)
            .ok_or_else(|| "checkpoint retained byte count overflow".to_owned())
    }
    pub fn fork(&self) -> SequenceState {
        self.store.owners.set(self.store.owners.get() + 1);
        SequenceState {
            store: self.store.clone(),
            position: self.position,
            expected_end: self.position,
            history_start: self.history_start,
            retained_start: self.retained_start,
            extents: self.extents.clone(),
            bank: self.bank.clone(),
        }
    }
}
fn install_commit(
    state: &mut SequenceState,
    extents: &mut Vec<Rc<Extent>>,
    following: &mut BankHandle,
    count: usize,
) {
    // V3 merges adjacent extents only when the accepted tail has no other
    // owners. A checkpoint's retained boundary must never grow.
    if let (Some(previous), Some(next)) = (state.extents.last_mut(), extents.first_mut()) {
        if Rc::strong_count(previous) == 1 && previous.start + previous.count == next.start {
            let previous = Rc::get_mut(previous).unwrap();
            let next = Rc::get_mut(next).unwrap();
            previous.count += next.count;
            next.count = 0;
            extents.remove(0);
        }
    }
    state.extents.append(extents);
    std::mem::swap(&mut state.bank, following);
    state.position += count;
}

fn split_extent_prefix(
    mut extents: ExtentSet,
    keep_rows: usize,
) -> Result<(ExtentSet, ExtentSet), String> {
    let total = extents.iter().map(|extent| extent.count).sum::<usize>();
    if keep_rows > total {
        return Err("accepted prefix exceeds reserved extent rows".into());
    }
    let mut remaining = keep_rows;
    let mut kept = Vec::new();
    let mut released = Vec::new();
    for mut extent in extents.drain(..) {
        if remaining == 0 {
            released.push(extent);
        } else if remaining >= extent.count {
            remaining -= extent.count;
            kept.push(extent);
        } else {
            if Rc::strong_count(&extent) != 1 {
                return Err("cannot split a reserved extent with outstanding claims".into());
            }
            let current = Rc::get_mut(&mut extent).unwrap();
            let arena = current.arena.clone();
            let start = current.start;
            let count = current.count;
            current.count = 0;
            kept.push(Rc::new(Extent {
                arena: arena.clone(),
                start,
                count: remaining,
            }));
            released.push(Rc::new(Extent {
                arena,
                start: start + remaining,
                count: count - remaining,
            }));
            remaining = 0;
        }
    }
    debug_assert_eq!(remaining, 0);
    Ok((kept, released))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog};

    fn cpu_device() -> Option<Rc<Device>> {
        DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
            .map(Rc::new)
    }

    fn bank_view(store: &StateStore, bank: usize) -> Tensor {
        store.recurrent_arenas()[0]
            .slice_leading(bank as u64, bank as u64 + 1)
            .unwrap()
    }

    fn bank_bytes(store: &StateStore, bank: usize) -> Vec<u8> {
        bank_view(store, bank).read_to_host().unwrap()
    }

    fn write_bank(store: &StateStore, bank: usize, bytes: &[u8]) {
        bank_view(store, bank).write_from_host(bytes).unwrap();
    }

    fn dense_component(width: usize) -> ComponentDescriptor {
        ComponentDescriptor::new(
            LayerRef::Target(0),
            CodecSpec::dense(DType::F32, width, width),
            1,
        )
        .unwrap()
    }

    #[test]
    fn component_specs_reject_non_float_empty_and_zero_shapes() {
        for spec in [
            ComponentSpec {
                shape: vec![],
                dtype: DType::F32,
            },
            ComponentSpec {
                shape: vec![4, 0],
                dtype: DType::F32,
            },
            ComponentSpec {
                shape: vec![4],
                dtype: DType::I32,
            },
        ] {
            assert_eq!(
                spec.bytes().unwrap_err(),
                "state components require nonempty floating tensors"
            );
        }
    }

    #[test]
    fn dropping_extents_coalesces_adjacent_arena_ranges() {
        let arena = Rc::new(RefCell::new(Arena { free: vec![] }));
        let left = Extent {
            arena: arena.clone(),
            start: 2,
            count: 3,
        };
        let right = Extent {
            arena: arena.clone(),
            start: 5,
            count: 4,
        };

        drop(right);
        drop(left);

        assert_eq!(arena.borrow().free, vec![(2, 7)]);
    }

    #[test]
    fn arena_grows_histories_in_place_and_starts_new_ones_mid_hole() {
        let mut arena = Arena {
            free: vec![(0, 100)],
        };
        // Row 0 has no preceding history: fill from its start.
        assert_eq!(arena.claim(None, 10), [(0, 10)]);
        // A new history leaves the rows after [0, 10) as that history's room.
        assert_eq!(arena.claim(None, 10), [(50, 10)]);
        assert_eq!(arena.free, [(10, 40), (60, 40)]);
        // Both grow in place, one row at a time, whatever the interleaving.
        for step in 0..5 {
            assert_eq!(arena.claim(Some(10 + step), 1), [(10 + step, 1)]);
            assert_eq!(arena.claim(Some(60 + step), 1), [(60 + step, 1)]);
        }
        // A third history starts mid-way through the largest hole (ties go to
        // the lower address); a history without room in place follows it.
        assert_eq!(arena.claim(None, 5), [(30, 5)]);
        assert_eq!(arena.claim(Some(35), 20), [(35, 15), (80, 5)]);
        assert_eq!(arena.free, [(15, 15), (65, 15), (85, 15)]);
        // A request larger than every hole takes whole holes, largest first.
        assert_eq!(arena.claim(Some(3), 40), [(15, 15), (65, 15), (87, 10)]);
        assert_eq!(arena.free, [(85, 2), (97, 3)]);
        assert_eq!(arena.available(), 5);
    }

    #[test]
    fn planes_allocate_lazily_as_one_stable_set() {
        let Some(device) = cpu_device() else {
            // A backend may be present yet fail Seismic's runtime calibration
            // under a noisy test host. Never substitute an accelerator here.
            return;
        };
        let store = StateStore::new(
            device,
            4,
            8,
            vec![dense_component(4)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        assert!(!store.history_allocated());
        assert_eq!(store.total_history_row_bytes(), 32);
        assert_eq!(store.total_history_bytes(), 256);

        let first = store.history_planes().unwrap();
        assert!(store.history_allocated());
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].plane_index, 0);
        assert_eq!(first[1].plane_index, 1);
        assert_eq!(first[0].vector, VectorKind::Key);
        assert_eq!(first[1].vector, VectorKind::Value);
        assert!(first.iter().all(|plane| {
            plane.component_index == 0
                && plane.layer == LayerRef::Target(0)
                && plane.name == PlaneName::Dense
                && plane.row_bytes == 16
                && plane.base_row == 0
                && plane.buffer.extents() == [8, 1, 4]
        }));

        let second = store.history_planes().unwrap();
        assert!(first
            .iter()
            .zip(second.iter())
            .all(|(left, right)| left.buffer.shares_allocation(&right.buffer)));
    }

    #[test]
    fn capacity_error_uses_all_plane_bytes() {
        let row_bytes = validate_history_layout(&[dense_component(4)], 8).unwrap();
        let total_bytes = row_bytes * 8;
        assert_eq!(capacity_charge(3, total_bytes, 8, true).unwrap(), 96);
        assert_eq!(capacity_charge(2, total_bytes, 8, false).unwrap(), 64);
    }

    #[test]
    fn store_rejects_total_capacity_overflow_before_allocation() {
        let component = dense_component(usize::MAX / 4);
        assert!(matches!(
            validate_history_layout(&[component], 2),
            Err(LayoutError::ArithmeticOverflow(_))
        ));
    }

    #[test]
    fn duplicate_layer_descriptors_are_rejected() {
        let component = dense_component(4);
        assert!(matches!(
            validate_history_layout(&[component.clone(), component], 2),
            Err(LayoutError::DuplicateLayer(LayerRef::Target(0)))
        ));
    }

    fn test_extent(arena: &Rc<RefCell<Arena>>, start: usize, count: usize) -> Rc<Extent> {
        Rc::new(Extent {
            arena: arena.clone(),
            start,
            count,
        })
    }

    fn extent_ranges(extents: &[Rc<Extent>]) -> Vec<(usize, usize)> {
        extents
            .iter()
            .map(|extent| (extent.start, extent.count))
            .collect()
    }

    #[test]
    fn prefix_split_handles_zero_interior_full_and_fragmented_extents() {
        for (accepted, expected_kept, expected_released) in [
            (0, vec![], vec![(0, 3), (8, 4)]),
            (5, vec![(0, 3), (8, 2)], vec![(10, 2)]),
            (7, vec![(0, 3), (8, 4)], vec![]),
        ] {
            let arena = Rc::new(RefCell::new(Arena { free: vec![] }));
            let extents = vec![test_extent(&arena, 0, 3), test_extent(&arena, 8, 4)];
            let (kept, released) = split_extent_prefix(extents, accepted).unwrap();
            assert_eq!(extent_ranges(&kept), expected_kept);
            assert_eq!(extent_ranges(&released), expected_released);
        }
    }

    #[test]
    fn rejected_tail_is_not_recycled_while_an_extent_claim_is_pinned() {
        let arena = Rc::new(RefCell::new(Arena { free: vec![] }));
        let extents = vec![test_extent(&arena, 0, 2), test_extent(&arena, 8, 3)];
        let tail_pin = extents[1].clone();
        let (kept, released) = split_extent_prefix(extents, 2).unwrap();
        drop(released);
        assert!(arena.borrow().free.is_empty());
        drop(tail_pin);
        assert_eq!(arena.borrow().free, [(8, 3)]);
        drop(kept);
        assert_eq!(arena.borrow().free, [(0, 2), (8, 3)]);
    }

    #[test]
    fn prefix_commit_repair_is_transactional_and_retains_both_banks() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![dense_component(1)],
            vec![ComponentSpec {
                shape: vec![1],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 2,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let state = store.create().unwrap();
        let original = state.bank_index();
        assert_eq!(original, ZERO_SEED_BANK);
        let checkpoint = state.checkpoint();

        let advance = OwnedStateAdvance::begin(state, 3).ok().unwrap();
        assert_eq!(advance.bindings().previous_bank, original);
        assert_ne!(advance.bindings().following_bank, original);
        assert!(advance.bindings().recurrent[0].shares_allocation(&store.recurrent_arenas()[0]));
        let OwnedAdvanceResolution::Repair(repair) = advance.commit(1).ok().unwrap() else {
            panic!("interior prefix must require repair");
        };
        assert_eq!(repair.rows(), 1);
        assert_eq!(store.occupied_rows(), 1);
        assert_eq!(repair.previous_bank(), original);
        assert_ne!(repair.following_bank(), original);
        let state = repair.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(state.bank_index(), original);
        assert_eq!(store.occupied_rows(), 0);

        let advance = OwnedStateAdvance::begin(state, 3).ok().unwrap();
        let OwnedAdvanceResolution::Repair(repair) = advance.commit(2).ok().unwrap() else {
            panic!("interior prefix must require repair");
        };
        let successor = repair.following_bank();
        let state = repair.commit();
        assert_eq!(state.position(), 2);
        assert_eq!(state.bank_index(), successor);
        assert_eq!(store.occupied_rows(), 2);
        assert_eq!(checkpoint.position(), 0);
        let checkpoint_fork = checkpoint.fork();
        assert_eq!(checkpoint_fork.position(), 0);
        assert_eq!(checkpoint_fork.bank_index(), original);
        assert!(checkpoint_fork.history_ranges().is_empty());
        drop(checkpoint_fork);

        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Aborted(state) = advance.commit(0).ok().unwrap() else {
            panic!("zero prefix must abort");
        };
        assert_eq!(state.position(), 2);
        assert_eq!(store.occupied_rows(), 2);
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(state.position(), 4);
        assert_eq!(store.occupied_rows(), 4);
    }

    #[test]
    fn checkpoint_reports_all_pinned_history_and_recurrent_storage() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![dense_component(1)],
            vec![ComponentSpec {
                shape: vec![1],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 3)
            .ok()
            .unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        let checkpoint = state.checkpoint();
        let recurrent = store.allocation_trace().unwrap().recurrent_bank_bytes;
        assert_eq!(recurrent, 4);
        assert_eq!(
            checkpoint.retained_bytes().unwrap(),
            3 * store.total_history_row_bytes() + recurrent
        );
    }

    #[test]
    fn attention_only_interior_prefix_commits_without_repair_when_fragmented() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 3,
                in_flight: 3,
                retained: 0,
            },
        )
        .unwrap();
        let states = (0..3)
            .map(|_| {
                let advance = OwnedStateAdvance::begin(store.create().unwrap(), 2)
                    .ok()
                    .unwrap();
                let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap()
                else {
                    panic!("full prefix must commit");
                };
                state
            })
            .collect::<Vec<_>>();
        let mut states = states.into_iter();
        let first = states.next().unwrap();
        let middle = states.next().unwrap();
        let last = states.next().unwrap();
        // Placement: first [0, 2), middle mid-hole at [4, 6), last [2, 4).
        let checkpoint = first.checkpoint();
        drop(last);
        // Two rows grow in place, the third starts the largest hole.
        let advance = OwnedStateAdvance::begin(first, 3).ok().unwrap();
        assert_eq!(advance.bindings().destinations, [2, 3, 6]);
        let OwnedAdvanceResolution::Committed(first) = advance.commit(2).ok().unwrap() else {
            panic!("attention prefix must commit without repair");
        };
        assert_eq!(first.position(), 4);
        assert_eq!(first.history_ranges(), [(0, 4)]);
        assert_eq!(checkpoint.position(), 2);
        assert_eq!(checkpoint.fork().history_ranges(), [(0, 2)]);
        assert_eq!(store.occupied_rows(), 6);
        assert_eq!(middle.position(), 2);
    }

    #[test]
    fn pooled_banks_are_reused_and_checkpoint_claims_force_cow() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![],
            vec![ComponentSpec {
                shape: vec![4],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        assert_eq!(store.available_banks(), 2);
        let state = store.create().unwrap();
        assert_eq!(store.available_banks(), 2);
        let zero_seed_bank = state.bank_index();

        let advance = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let following_bank = advance.bindings().following_bank;
        assert_ne!(following_bank, zero_seed_bank);
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(state.bank_index(), following_bank);
        assert_eq!(store.available_banks(), 1);
        let checkpoint = state.checkpoint();
        assert_eq!(checkpoint.bank_index(), following_bank);

        let advance = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let second_bank = advance.bindings().following_bank;
        assert_ne!(second_bank, following_bank);
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(store.available_banks(), 0);
        let Err((state, Error::BanksExhausted { capacity: 2 })) =
            OwnedStateAdvance::begin(state, 1)
        else {
            panic!("checkpoint must pin the old bank");
        };
        drop(checkpoint);
        assert_eq!(store.available_banks(), 1);
        let next = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        assert_eq!(next.bindings().following_bank, following_bank);
        let state = next.abort();
        assert_eq!(store.available_banks(), 1);
        let retry = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        assert_eq!(retry.bindings().following_bank, following_bank);
    }

    /// Every successor bank is disjoint from bank 0 and from every bank a live
    /// state, checkpoint, fork, or other in-flight advance can read, through
    /// forks, commits, aborts, and repairs.
    #[test]
    fn successor_banks_never_alias_a_readable_bank_or_the_zero_seed() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            16,
            16,
            vec![dense_component(1)],
            vec![
                ComponentSpec {
                    shape: vec![3, 8],
                    dtype: DType::BF16,
                },
                ComponentSpec {
                    shape: vec![2, 4, 4],
                    dtype: DType::F32,
                },
            ],
            BankCapacity {
                active: 3,
                in_flight: 3,
                retained: 2,
            },
        )
        .unwrap();
        assert_eq!(store.recurrent_arenas().len(), 2);
        assert_eq!(store.recurrent_arenas()[0].extents(), [9, 3, 8]);
        assert_eq!(store.recurrent_arenas()[1].extents(), [9, 2, 4, 4]);
        let readable = |states: &[&SequenceState], checkpoints: &[&StateCheckpoint]| {
            states
                .iter()
                .map(|state| state.bank_index())
                .chain(checkpoints.iter().map(|checkpoint| checkpoint.bank_index()))
                .collect::<Vec<_>>()
        };
        let parent = store.create().unwrap();
        let root = parent.checkpoint();
        let left = root.fork();
        let right = root.fork();
        let left = OwnedStateAdvance::begin(left, 2).ok().unwrap();
        let right = OwnedStateAdvance::begin(right, 1).ok().unwrap();
        let parent = OwnedStateAdvance::begin(parent, 3).ok().unwrap();
        let successors = [&left, &right, &parent].map(|advance| advance.bindings().following_bank);
        for (index, advance) in [&left, &right, &parent].into_iter().enumerate() {
            let bindings = advance.bindings();
            assert_eq!(bindings.previous_bank, ZERO_SEED_BANK);
            assert_ne!(bindings.following_bank, ZERO_SEED_BANK);
            assert!(!readable(&[], &[&root]).contains(&bindings.following_bank));
            assert!(successors
                .iter()
                .enumerate()
                .all(|(other, bank)| other == index || *bank != bindings.following_bank));
        }
        let OwnedAdvanceResolution::Committed(left) = left.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        let right = right.abort();
        let OwnedAdvanceResolution::Repair(repair) = parent.commit(2).ok().unwrap() else {
            panic!("interior recurrent prefix must repair");
        };
        let branch = left.checkpoint();
        let fork = branch.fork();
        assert_eq!(fork.bank_index(), left.bank_index());
        assert_ne!(repair.following_bank(), repair.previous_bank());
        assert!(!readable(&[&left, &right, &fork], &[&root, &branch])
            .contains(&repair.following_bank()));
        let parent = repair.commit();
        let advance = OwnedStateAdvance::begin(fork, 1).ok().unwrap();
        let following = advance.bindings().following_bank;
        assert_ne!(following, ZERO_SEED_BANK);
        assert!(!readable(&[&left, &right, &parent], &[&root, &branch]).contains(&following));
        drop(advance);
        drop((left, right, parent, root, branch));
        assert_eq!(store.available_banks(), 8);
    }

    #[test]
    fn fresh_sequences_share_a_pristine_seed_after_dirty_successor_reuse() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            8,
            8,
            vec![],
            vec![ComponentSpec {
                shape: vec![4],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let trace = store.allocation_trace().unwrap();
        assert_eq!(trace.recurrent_bank_bytes, 16);
        assert_eq!(trace.zero_seed_bytes, 16);
        assert_eq!(trace.recurrent_pool_bytes, 48);
        assert_eq!(store.available_banks(), 2);

        let first = store.create().unwrap();
        let seed_bank = first.bank_index();
        assert_eq!(seed_bank, ZERO_SEED_BANK);
        assert_eq!(bank_bytes(&store, seed_bank), vec![0; 16]);
        let advance = OwnedStateAdvance::begin(first, 1).ok().unwrap();
        let dirty = advance.bindings().following_bank;
        assert_ne!(dirty, seed_bank);
        write_bank(&store, dirty, &7.5f32.to_le_bytes().repeat(4));
        let OwnedAdvanceResolution::Committed(first) = advance.commit_all().ok().unwrap() else {
            panic!("full prefix must commit");
        };
        assert_eq!(first.bank_index(), dirty);
        assert_eq!(bank_bytes(&store, dirty), 7.5f32.to_le_bytes().repeat(4));
        drop(first);

        let second = store.create().unwrap();
        assert_eq!(second.bank_index(), seed_bank);
        assert_eq!(bank_bytes(&store, seed_bank), vec![0; 16]);
        assert_eq!(store.available_banks(), 2);
    }

    #[test]
    fn compaction_is_bit_exact_and_publishes_only_after_success() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            64,
            64,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let mut state = store.create().unwrap();
        state.extents = (0..17)
            .map(|row| test_extent(&store.arena, row * 2, 1))
            .collect();
        state.position = 17;
        store.arena.borrow_mut().free = vec![(40, 17)];

        let planes = store.history_planes().unwrap();
        for plane in &planes {
            let mut bytes = vec![0u8; plane.buffer.byte_len() as usize];
            for row in 0..64 {
                let start = row * plane.row_bytes;
                bytes[start..start + plane.row_bytes].fill(row as u8);
            }
            let mut buffer = plane.buffer.clone();
            buffer.write_from_host(&bytes).unwrap();
        }

        assert!(state.compaction_needed());
        let old_ranges = state.history_ranges();
        let OwnedCompactionPreparation::Ready(failed) =
            OwnedCompaction::prepare(state).ok().unwrap()
        else {
            panic!("contiguous destination must prepare compaction")
        };
        // A failed copy submission aborts its owned destination.
        let state = failed.abort();
        assert_eq!(state.history_ranges(), old_ranges);

        let OwnedCompactionPreparation::Ready(compaction) =
            OwnedCompaction::prepare(state).ok().unwrap()
        else {
            panic!("released destination must be reusable")
        };
        assert_eq!(compaction.copies().len(), 2);
        for copy in compaction.copies() {
            assert_eq!(copy.from, (0..17).map(|row| row * 2).collect::<Vec<_>>());
            assert_eq!(copy.to, (40..57).collect::<Vec<_>>());
        }
        let bindings = compaction.bindings();
        for copy in bindings.copies {
            let plane = &bindings.history[copy.plane_index];
            let mut bytes = plane.buffer.read_to_host().unwrap();
            for (&from, &to) in copy.from.iter().zip(&copy.to) {
                let source = bytes[from * plane.row_bytes..(from + 1) * plane.row_bytes].to_vec();
                bytes[to * plane.row_bytes..(to + 1) * plane.row_bytes].copy_from_slice(&source);
            }
            let mut buffer = plane.buffer.clone();
            buffer.write_from_host(&bytes).unwrap();
        }
        let state = compaction.commit();
        assert_eq!(state.history_ranges(), [(40, 17)]);
        assert!(!state.compaction_needed());
        for plane in store.history_planes().unwrap() {
            let bytes = plane.buffer.read_to_host().unwrap();
            for (logical, row) in (40..57).enumerate() {
                assert_eq!(
                    &bytes[row * plane.row_bytes..(row + 1) * plane.row_bytes],
                    vec![(logical * 2) as u8; plane.row_bytes]
                );
            }
        }
    }

    #[test]
    fn compaction_defers_when_no_contiguous_run_is_free() {
        let Some(device) = cpu_device() else {
            return;
        };
        let store = StateStore::new(
            device,
            64,
            64,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let mut state = store.create().unwrap();
        state.extents = (0..17)
            .map(|row| test_extent(&store.arena, row * 2, 1))
            .collect();
        state.position = 17;
        store.arena.borrow_mut().free = vec![(40, 8), (50, 9)];
        assert!(state.compaction_needed());
        let OwnedCompactionPreparation::Deferred {
            state,
            segments,
            visible_rows,
        } = OwnedCompaction::prepare(state).ok().unwrap()
        else {
            panic!("fragmented capacity must defer compaction");
        };
        assert_eq!((segments, visible_rows), (17, 17));
        assert_eq!(state.history_ranges().len(), 17);
    }

    /// Lock-step serving of `active` requests for `steps` steps: chunked
    /// prefills interleaved with speculative decode (1 + 0..=3 draft rows,
    /// a random accepted prefix), every advance of a step in flight at once,
    /// requests finishing and new ones admitted into their slots. The arena
    /// holds `contexts` full contexts plus one batch. Returns the largest
    /// segment count any accepted history reached and the number of decode
    /// steps the longest request ran.
    fn serve_interleaved(active: usize, contexts: usize, steps: usize) -> (usize, usize) {
        const CONTEXT: usize = 512;
        const BATCH_ROWS: usize = 64;
        let device = cpu_device().expect("the CPU backend is available");
        let store = StateStore::new(
            device,
            CONTEXT,
            contexts * CONTEXT + BATCH_ROWS,
            vec![dense_component(1)],
            vec![],
            BankCapacity {
                active,
                in_flight: active,
                retained: 0,
            },
        )
        .unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = |bound: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % bound as u64) as usize
        };
        struct Request {
            state: SequenceState,
            prompt: usize,
            length: usize,
            decode_steps: usize,
        }
        let mut slots: Vec<Option<Request>> = (0..active).map(|_| None).collect();
        let (mut max_segments, mut max_decode_steps) = (0, 0);
        for step in 0..steps {
            for slot in &mut slots {
                if slot.is_none() {
                    let prompt = 16 + random(160);
                    *slot = Some(Request {
                        state: store.create().unwrap(),
                        prompt,
                        length: prompt + 200 + random(CONTEXT - prompt - 200),
                        decode_steps: 0,
                    });
                }
            }
            // Every advance of the step reserves before any resolves; the
            // scheduling order rotates so no request always reserves first.
            let order = (0..active)
                .map(|index| (index + step) % active)
                .collect::<Vec<_>>();
            let mut advances = Vec::with_capacity(active);
            for &index in &order {
                let request = slots[index].take().unwrap();
                let position = request.state.position();
                let rows = if position < request.prompt {
                    (request.prompt - position).min(BATCH_ROWS / 2)
                } else {
                    (1 + random(4)).min(request.length - position)
                };
                let advance = match OwnedStateAdvance::begin(request.state, rows) {
                    Ok(advance) => advance,
                    Err((_, error)) => panic!("step {step}: {error}"),
                };
                advances.push((
                    index,
                    rows,
                    request.prompt,
                    request.length,
                    request.decode_steps,
                    advance,
                ));
            }
            advances.reverse();
            for (index, rows, prompt, length, decode_steps, advance) in advances {
                let decode = advance.position() >= prompt;
                let accepted = if decode { 1 + random(rows) } else { rows };
                let OwnedAdvanceResolution::Committed(state) =
                    advance.commit(accepted).ok().unwrap()
                else {
                    panic!("attention-only prefixes commit without repair");
                };
                let decode_steps = decode_steps + usize::from(decode);
                max_segments = max_segments.max(state.history_ranges().len());
                max_decode_steps = max_decode_steps.max(decode_steps);
                slots[index] = (state.position() < length).then_some(Request {
                    state,
                    prompt,
                    length,
                    decode_steps,
                });
            }
            let mut ranges = slots
                .iter()
                .flatten()
                .flat_map(|request| request.state.history_ranges())
                .collect::<Vec<_>>();
            ranges.sort_unstable();
            assert!(
                ranges
                    .windows(2)
                    .all(|pair| pair[0].0 + pair[0].1 <= pair[1].0),
                "live histories overlap"
            );
        }
        (max_segments, max_decode_steps)
    }

    #[test]
    fn interleaved_decode_keeps_every_history_in_few_segments() {
        // Arena sized as the resource planner sizes it: a context for every
        // active and every in-flight owner.
        let (segments, decode_steps) = serve_interleaved(8, 16, 800);
        assert!(
            decode_steps > 100,
            "requests ran {decode_steps} decode steps"
        );
        assert_eq!(segments, 1);
        // Tight arenas, down to exactly one context per request: requests
        // outgrow their room and move, but never fragment past two segments
        // (first-fit placement gave one segment per decode step).
        for (active, contexts) in [(4, 5), (8, 9), (8, 8), (16, 17)] {
            let (segments, decode_steps) = serve_interleaved(active, contexts, 3000);
            assert!(
                decode_steps > 100,
                "requests ran {decode_steps} decode steps"
            );
            assert!(
                segments <= 2,
                "{active} requests in {contexts} contexts reached {segments} segments"
            );
        }
    }
}
