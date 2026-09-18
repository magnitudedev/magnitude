//! Logical model execution boundaries consumed by generation and service policy.
use crate::inputs::TokenId;
use crate::state::{SequenceState, StateCheckpoint, StateStore};
use seismic_runtime::Error;
use std::{cell::RefCell, rc::Rc};

/// A prepared numerical advance owns its tentative claims and submitted uses.
/// Dropping it aborts acceptance and must retain any still-running physical uses.
/// `selected` reports completion/selection errors; `commit` is atomic on failure.
/// Neither method may treat asynchronous submission as successful completion.
pub trait Advance {
    fn is_complete(&self) -> bool;
    fn selected(&mut self) -> Result<Option<TokenId>, String>;
    fn commit(&mut self) -> Result<(), String>;
}

struct Accepted<S> {
    state: SequenceState,
    semantics: S,
    pending: bool,
}

/// Execution-owner handle for numerical state and optional input continuation.
/// Cloning aliases the same accepted sequence; use a reconciled checkpoint to
/// create a fork. Unit semantics represents an ordinary text-only sequence.
#[derive(Clone)]
pub struct OwnedSequence<S = ()>(Rc<RefCell<Accepted<S>>>);
/// One request's numerical extent in a shared completed batch.
pub struct SequenceWork<'a, S = ()> {
    pub sequence: &'a OwnedSequence<S>,
    pub position: usize,
    pub count: usize,
}
impl OwnedSequence {
    pub fn new(state: SequenceState) -> Self {
        Self::with_semantics(state, ())
    }
    pub fn checkpoint(&self) -> Result<StateCheckpoint, String> {
        self.checkpoint_with_semantics().map(|(state, ())| state)
    }
    /// Exclusive recurrent allocation bytes freed by dropping this entire set
    /// of unique handles. Aliased owners and checkpoints keep their storage live.
    /// History remains charged to the store until its idle release.
    pub fn reclaimable(store: &Rc<StateStore>, sequences: &[&Self]) -> Result<usize, String> {
        let mut seen = std::collections::HashSet::new();
        let mut accepted = Vec::new();
        for sequence in sequences {
            if !seen.insert(Rc::as_ptr(&sequence.0)) {
                return Err("reclamation set contains a duplicate sequence".into());
            }
            let state = sequence.0.borrow();
            if state.pending || !state.state.belongs_to(store) {
                return Err("reclamation requires reconciled sequences from this store".into());
            }
            if Rc::strong_count(&sequence.0) == 1 {
                accepted.push(state);
            }
        }
        store.reclaimable(&accepted.iter().map(|a| &a.state).collect::<Vec<_>>())
    }
    /// Adapt an executor that completes synchronously. Execution commits only
    /// into a private fork; the returned row publishes it after generation has
    /// staged selection/grammar acceptance. The closure must finish all native
    /// uses before returning, including on failure. Asynchronous executors must
    /// use a completion-owning implementation of Advance instead.
    pub fn prepare_completed(
        &self,
        position: usize,
        count: usize,
        execute: impl FnOnce(&mut SequenceState) -> Result<Option<TokenId>, Error>,
    ) -> Result<Box<dyn Advance>, Error> {
        let mut rows = Self::prepare_completed_batch(
            &[SequenceWork {
                sequence: self,
                position,
                count,
            }],
            |states| Ok(vec![Ok(execute(states[0])?)]),
        )?;
        Ok(rows.remove(0))
    }
    pub fn prepare_completed_batch(
        work: &[SequenceWork<'_>],
        execute: impl FnOnce(
            &mut [&mut SequenceState],
        ) -> Result<Vec<Result<Option<TokenId>, String>>, Error>,
    ) -> Result<Vec<Box<dyn Advance>>, Error> {
        Self::prepare_completed_batch_with_semantics(work, |states| {
            execute(
                &mut states
                    .iter_mut()
                    .map(|(state, ())| &mut **state)
                    .collect::<Vec<_>>(),
            )
        })
    }
}
impl<S: Clone + 'static> OwnedSequence<S> {
    /// `S::clone` must snapshot mutable continuation. It may share immutable
    /// input plans and feature leases, but must not alias mutable accepted state.
    pub fn with_semantics(state: SequenceState, semantics: S) -> Self {
        Self(Rc::new(RefCell::new(Accepted {
            state,
            semantics,
            pending: false,
        })))
    }
    pub fn position(&self) -> usize {
        self.0.borrow().state.position()
    }
    pub fn pending(&self) -> bool {
        self.0.borrow().pending
    }
    /// Forkable numerical and semantic state captured at one accepted boundary.
    pub fn checkpoint_with_semantics(&self) -> Result<(StateCheckpoint, S), String> {
        let accepted = self.0.borrow();
        if accepted.pending {
            return Err("cannot checkpoint an unresolved sequence advance".into());
        }
        Ok((accepted.state.checkpoint(), accepted.semantics.clone()))
    }
    /// Acquire every tentative sequence together before synchronous execution.
    /// A shared failure or unwind publishes none of them. Completed selection
    /// failures are row-local and cannot commit; peers accept independently.
    /// The closure must finish every native use before returning or unwinding.
    pub fn prepare_completed_batch_with_semantics(
        work: &[SequenceWork<'_, S>],
        execute: impl FnOnce(
            &mut [(&mut SequenceState, &mut S)],
        ) -> Result<Vec<Result<Option<TokenId>, String>>, Error>,
    ) -> Result<Vec<Box<dyn Advance>>, Error> {
        if work.is_empty() {
            return Err("numerical batch must not be empty".into());
        }
        let mut seen = std::collections::HashSet::new();
        let mut ends = Vec::with_capacity(work.len());
        // Preflight the full batch before marking any accepted state pending.
        for item in work {
            if !seen.insert(Rc::as_ptr(&item.sequence.0)) {
                return Err("numerical batch contains aliased sequence owners".into());
            }
            let accepted = item.sequence.0.borrow();
            if accepted.pending || item.count == 0 || accepted.state.position() != item.position {
                return Err("sequence advance is pending, empty, or at a stale position".into());
            }
            ends.push(
                item.position
                    .checked_add(item.count)
                    .ok_or("sequence advance overflow")?,
            );
        }
        let mut rows = Vec::with_capacity(work.len());
        for item in work {
            let mut accepted = item.sequence.0.borrow_mut();
            let next = (
                accepted.state.checkpoint().fork(),
                accepted.semantics.clone(),
            );
            rows.push(CompletedAdvance {
                owner: item.sequence.0.clone(),
                next: Some(next),
                selected: Ok(None),
                resolved: false,
            });
            accepted.pending = true;
        }
        let selected = execute(
            &mut rows
                .iter_mut()
                .map(|row| {
                    let (state, semantics) = row.next.as_mut().unwrap();
                    (state, semantics)
                })
                .collect::<Vec<_>>(),
        )?;
        if selected.len() != rows.len() {
            return Err("executor returned the wrong number of selection outcomes".into());
        }
        for ((row, selected), end) in rows.iter_mut().zip(selected).zip(ends) {
            if row.next.as_ref().unwrap().0.position() != end {
                return Err("executor did not advance the declared sequence extent".into());
            }
            row.selected = selected;
        }
        Ok(rows
            .into_iter()
            .map(|row| Box::new(row) as Box<dyn Advance>)
            .collect())
    }
}

struct CompletedAdvance<S> {
    owner: Rc<RefCell<Accepted<S>>>,
    next: Option<(SequenceState, S)>,
    selected: Result<Option<TokenId>, String>,
    resolved: bool,
}
impl<S: 'static> Advance for CompletedAdvance<S> {
    fn is_complete(&self) -> bool {
        true
    }
    fn selected(&mut self) -> Result<Option<TokenId>, String> {
        if self.resolved {
            return Err("sequence advance already resolved".into());
        }
        self.selected.clone()
    }
    fn commit(&mut self) -> Result<(), String> {
        if self.resolved {
            return Err("sequence advance already resolved".into());
        }
        self.selected.as_ref().map_err(Clone::clone)?;
        let next = self
            .next
            .take()
            .ok_or("sequence advance has no successor")?;
        let mut accepted = self.owner.borrow_mut();
        accepted.state = next.0;
        accepted.semantics = next.1;
        accepted.pending = false;
        self.resolved = true;
        Ok(())
    }
}
impl<S> Drop for CompletedAdvance<S> {
    fn drop(&mut self) {
        if !self.resolved {
            self.owner.borrow_mut().pending = false;
        }
    }
}
