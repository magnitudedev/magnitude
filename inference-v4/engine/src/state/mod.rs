//! Accepted sequence state and transactional history ownership, following V3.
//! Native calls are synchronous today: an advance becomes committable only after
//! its execution closure returns successful physical completion.
use seismic_lang::types::DType;
use seismic_runtime::{Buffer, Device, Error};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

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
struct Arena {
    free: Vec<(usize, usize)>,
}
struct Extent {
    arena: Rc<RefCell<Arena>>,
    start: usize,
    count: usize,
}
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
/// History rows are a shared arena; recurrent components are immutable accepted
/// versions. A checkpoint retains both without copying tensor contents.
pub struct StateStore {
    device: Rc<Device>,
    context_capacity: usize,
    history_capacity: usize,
    history_row_bytes: Vec<usize>,
    component_specs: Vec<ComponentSpec>,
    history: RefCell<Option<Vec<Buffer>>>,
    arena: Rc<RefCell<Arena>>,
    owners: Cell<usize>,
}
impl StateStore {
    pub fn new(
        device: Rc<Device>,
        context_capacity: usize,
        history_capacity: usize,
        history_row_bytes: Vec<usize>,
        component_specs: Vec<ComponentSpec>,
    ) -> Result<Rc<Self>, String> {
        if context_capacity == 0
            || context_capacity > history_capacity
            || history_row_bytes.contains(&0)
        {
            return Err("history capacity must fit a positive sequence context".into());
        }
        for bytes in &history_row_bytes {
            bytes
                .checked_mul(history_capacity)
                .ok_or("history allocation overflow")?;
        }
        for spec in &component_specs {
            spec.bytes()?;
        }
        Ok(Rc::new(Self {
            device,
            context_capacity,
            history_capacity,
            history_row_bytes,
            component_specs,
            history: RefCell::new(None),
            arena: Rc::new(RefCell::new(Arena {
                free: vec![(0, history_capacity)],
            })),
            owners: Cell::new(0),
        }))
    }
    pub fn component_specs(&self) -> &[ComponentSpec] {
        &self.component_specs
    }
    pub fn history(&self) -> Result<Vec<Buffer>, Error> {
        if self.history.borrow().is_none() {
            let buffers = self
                .history_row_bytes
                .iter()
                .map(|bytes| self.device.buffer(bytes * self.history_capacity))
                .collect::<Result<Vec<_>, _>>()?;
            *self.history.borrow_mut() = Some(buffers);
        }
        Ok(self.history.borrow().as_ref().unwrap().clone())
    }
    pub fn occupied_rows(&self) -> usize {
        self.history_capacity
            - self
                .arena
                .borrow()
                .free
                .iter()
                .map(|(_, n)| n)
                .sum::<usize>()
    }
    /// Recurrent component storage freed by closing the selected states. Shared
    /// checkpoints, unselected descendants, and execution pins prevent reclaim.
    pub fn reclaimable(self: &Rc<Self>, states: &[&SequenceState]) -> Result<usize, String> {
        if states.iter().any(|state| !Rc::ptr_eq(self, &state.store)) {
            return Err("reclamation requires states from this store".into());
        }
        Buffer::reclaimable_bytes(states.iter().flat_map(|state| state.values.iter()))
    }
    pub fn idle(&self) -> bool {
        self.owners.get() == 0
    }
    /// Drop the store's arena allocations when no sequence/checkpoint owns them.
    /// External completion/buffer pins may still retain physical storage.
    pub fn release_idle(&self) -> Result<usize, String> {
        if !self.idle() {
            return Ok(0);
        }
        self.history
            .borrow_mut()
            .take()
            .map_or(Ok(0), |v| Buffer::reclaimable_bytes(v.iter()))
    }
    fn allocate_values(&self, zero: bool) -> Result<Vec<Buffer>, Error> {
        self.component_specs
            .iter()
            .map(|spec| {
                let bytes = spec.bytes()?;
                let buffer = self.device.buffer(bytes)?;
                if zero {
                    buffer.write(&vec![0; bytes])?;
                }
                Ok(buffer)
            })
            .collect()
    }
    pub fn create(self: &Rc<Self>) -> Result<SequenceState, Error> {
        let values = self.allocate_values(true)?;
        self.owners.set(self.owners.get() + 1);
        Ok(SequenceState {
            store: self.clone(),
            position: 0,
            expected_end: 0,
            history_start: 0,
            retained_start: 0,
            extents: vec![],
            values,
        })
    }
    fn reserve(&self, count: usize) -> Result<Vec<Rc<Extent>>, String> {
        if self.history_row_bytes.is_empty() {
            return Ok(vec![]);
        }
        let mut arena = self.arena.borrow_mut();
        if arena.free.iter().map(|(_, n)| n).sum::<usize>() < count {
            return Err("shared history capacity exhausted".into());
        }
        let mut remaining = count;
        let mut reserved = Vec::new();
        let mut free = Vec::new();
        for &(start, size) in &arena.free {
            let taken = size.min(remaining);
            if taken > 0 {
                reserved.push(Rc::new(Extent {
                    arena: self.arena.clone(),
                    start,
                    count: taken,
                }));
                remaining -= taken;
            }
            if size > taken {
                free.push((start + taken, size - taken));
            }
        }
        arena.free = free;
        Ok(reserved)
    }
}
pub struct SequenceState {
    store: Rc<StateStore>,
    position: usize,
    expected_end: usize,
    history_start: usize,
    retained_start: usize,
    extents: Vec<Rc<Extent>>,
    values: Vec<Buffer>,
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
    pub fn values(&self) -> &[Buffer] {
        &self.values
    }
    pub fn anticipate(&mut self, position: usize) -> Result<(), String> {
        if position > self.store.context_capacity {
            return Err("anticipated position exceeds context capacity".into());
        }
        self.expected_end = self.expected_end.max(position);
        Ok(())
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
    /// The mutable borrow prevents a second advance, trimming or checkpointing
    /// while a proposal is unresolved. Dropping the advance aborts it.
    pub fn begin(&mut self, count: usize) -> Result<StateAdvance<'_>, Error> {
        if count == 0 || count > self.store.context_capacity - self.position {
            return Err("advance exceeds context capacity".into());
        }
        let extents = self.store.reserve(count)?;
        let following = self.store.allocate_values(false)?;
        Ok(StateAdvance {
            state: self,
            count,
            extents,
            following,
            attempted: false,
            completed: false,
        })
    }
    pub fn checkpoint(&self) -> StateCheckpoint {
        self.store.owners.set(self.store.owners.get() + 1);
        StateCheckpoint {
            store: self.store.clone(),
            position: self.position,
            history_start: self.history_start,
            retained_start: self.retained_start,
            extents: self.extents.clone(),
            values: self.values.clone(),
        }
    }
}
pub struct StateCheckpoint {
    store: Rc<StateStore>,
    position: usize,
    history_start: usize,
    retained_start: usize,
    extents: Vec<Rc<Extent>>,
    values: Vec<Buffer>,
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
    pub fn fork(&self) -> SequenceState {
        self.store.owners.set(self.store.owners.get() + 1);
        SequenceState {
            store: self.store.clone(),
            position: self.position,
            expected_end: self.position,
            history_start: self.history_start,
            retained_start: self.retained_start,
            extents: self.extents.clone(),
            values: self.values.clone(),
        }
    }
}
#[derive(Clone, Copy)]
pub struct AdvanceBindings<'a> {
    pub previous: &'a [Buffer],
    pub following: &'a [Buffer],
    pub history: &'a [Buffer],
    pub destinations: &'a [usize],
}
pub struct StateAdvance<'a> {
    state: &'a mut SequenceState,
    count: usize,
    extents: Vec<Rc<Extent>>,
    following: Vec<Buffer>,
    attempted: bool,
    completed: bool,
}
impl StateAdvance<'_> {
    pub fn position(&self) -> usize {
        self.state.position
    }
    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.state.history_ranges()
    }

    pub fn destinations(&self) -> Vec<usize> {
        self.extents
            .iter()
            .flat_map(|e| e.start..e.start + e.count)
            .collect()
    }
    /// All numerical calls inside the closure must complete before returning.
    /// Current native execution enforces that synchronous boundary on errors too.
    /// This API must not accept mere asynchronous submission as completion.
    pub fn execute(
        &mut self,
        run: impl FnOnce(AdvanceBindings<'_>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        Self::execute_batch(std::slice::from_mut(self), |bindings| run(bindings[0]))
    }

    /// Replace a preallocated proposal component with a compiler-owned result after the
    /// synchronous numerical execution that produced it. The accepted state adopts that
    /// allocation on commit; aborted advances drop it normally.
    pub fn replace_following(&mut self, index: usize, buffer: Buffer) -> Result<(), String> {
        if !self.completed {
            return Err("proposal results can be adopted only after successful completion".into());
        }
        let expected = self
            .following
            .get(index)
            .ok_or("proposal component index is out of bounds")?;
        if !buffer.belongs_to(&self.state.store.device) || buffer.len() != expected.len() {
            return Err("proposal result differs from its state component allocation".into());
        }
        self.following[index] = buffer;
        Ok(())
    }
    /// One synchronous completion covers every row's constituent work. No row
    /// can become committable if the shared execution fails or unwinds.
    pub fn execute_batch(
        advances: &mut [Self],
        run: impl FnOnce(&[AdvanceBindings<'_>]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if advances.is_empty() || advances.iter().any(|advance| advance.attempted) {
            return Err("batch is empty or an advance has already been executed".into());
        }
        for advance in advances.iter_mut() {
            advance.attempted = true;
        }
        let history = advances
            .iter()
            .map(|advance| advance.state.store.history())
            .collect::<Result<Vec<_>, _>>()?;
        let destinations = advances.iter().map(Self::destinations).collect::<Vec<_>>();
        let bindings = advances
            .iter()
            .enumerate()
            .map(|(i, advance)| AdvanceBindings {
                previous: &advance.state.values,
                following: &advance.following,
                history: &history[i],
                destinations: &destinations[i],
            })
            .collect::<Vec<_>>();
        run(&bindings)?;
        for advance in advances {
            advance.completed = true;
        }
        Ok(())
    }
    pub fn commit(mut self) -> Result<(), String> {
        if !self.completed {
            return Err("advance has no successful completion".into());
        }
        // V3 merges adjacent extents only when the accepted tail has no other
        // owners. A checkpoint's retained boundary must never grow.
        if let (Some(previous), Some(following)) =
            (self.state.extents.last_mut(), self.extents.first_mut())
        {
            if Rc::strong_count(previous) == 1 && previous.start + previous.count == following.start
            {
                let previous = Rc::get_mut(previous).unwrap();
                let following = Rc::get_mut(following).unwrap();
                previous.count += following.count;
                following.count = 0;
                self.extents.remove(0);
            }
        }
        self.state.extents.append(&mut self.extents);
        self.state.values = std::mem::take(&mut self.following);
        self.state.position += self.count;
        Ok(())
    }
    pub fn abort(self) {} // RAII releases only after synchronous execution returned.
}
