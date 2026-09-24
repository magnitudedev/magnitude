//! Owned tentative state, movable through an in-flight program submission.

use super::{
    append_ranges, install_commit, BankHandle, Claims, Codec, ComponentDescriptor, Error, KvCodec, LayerRef,
    PlaneBuffer, PlaneCopy, SequenceState, StateStore, Transaction, VectorKind,
    MAX_VISIBLE_SEGMENTS,
};
use seismic::Tensor;
use std::rc::Rc;

/// A read-only view whose tensors and row claims are owned by the transaction.
/// `recurrent` holds one arena per recurrent component; the advance reads
/// version (`previous_bank`, `previous_tape`) and writes only bank
/// `following_bank` of each. It publishes its recurrent state after `stop`
/// rows and records the tape of the rows after it.
#[derive(Clone, Copy)]
pub struct OwnedAdvanceBindings<'a> {
    pub recurrent: &'a [Tensor],
    pub previous_bank: usize,
    pub previous_tape: usize,
    pub following_bank: usize,
    pub stop: usize,
    pub history: &'a [PlaneBuffer],
    pub destinations: &'a [usize],
}

/// Tentative successor state. Moving this value into a launch and submission
/// preserves the source state, reserved extents, recurrent bank, and bindings
/// until physical completion and reconciliation.
pub struct OwnedStateAdvance {
    state: SequenceState,
    count: usize,
    /// The rows that always commit; the recurrent state is published after
    /// them and every later row is recorded on the tape.
    committed: usize,
    claims: Claims,
    following: BankHandle,
    history: Vec<PlaneBuffer>,
    recurrent: Rc<[Tensor]>,
    destinations: Vec<usize>,
    _transaction: Transaction,
}

/// Preparation keeps ownership of the source sequence in every outcome.
/// Deferral is a capacity result, not a partially reserved transaction.
pub enum OwnedCompactionPreparation {
    NotNeeded(SequenceState),
    Ready(OwnedCompaction),
    Deferred {
        state: SequenceState,
        segments: usize,
        visible_rows: usize,
    },
}

/// Tentative history copy with a reserved contiguous destination. Its source
/// extents remain accepted until the state program completes and commits it.
pub struct OwnedCompaction {
    state: SequenceState,
    destination: Claims,
    copies: Vec<PlaneCopy>,
    history: Vec<PlaneBuffer>,
    _transaction: Transaction,
}

/// Physical copy bindings pinned by the owned compaction until completion.
#[derive(Clone, Copy)]
pub struct OwnedCompactionBindings<'a> {
    pub history: &'a [PlaneBuffer],
    pub copies: &'a [PlaneCopy],
}

/// One semantic vector conversion. A codec can use several physical planes;
/// their indices are carried together rather than mistaken for one copy plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecConversionStep {
    pub layer: LayerRef,
    pub vector: VectorKind,
    pub source_codec: Codec,
    pub destination_codec: Codec,
    pub source_planes: Vec<usize>,
    pub destination_planes: Vec<usize>,
    pub from: Vec<usize>,
    pub to: Vec<usize>,
}

/// Physical buffers remain pinned by the owned transaction through finish.
/// Recurrent state moves from bank `source_bank` of the source arenas to bank
/// `destination_bank` of the destination arenas, a successor claimed for the
/// conversion (never the destination's zero seed).
pub struct OwnedCodecBindings<'a> {
    pub source_history: &'a [PlaneBuffer],
    pub destination_history: &'a [PlaneBuffer],
    pub source_recurrent: &'a [Tensor],
    pub source_bank: usize,
    pub destination_recurrent: &'a [Tensor],
    pub destination_bank: usize,
    pub conversions: &'a [CodecConversionStep],
}

/// A reserved destination and a complete, checked conversion program. Nothing
/// about the destination becomes accepted until a completed submission commits.
pub struct OwnedCodecAdvance {
    source: SequenceState,
    destination: SequenceState,
    claims: Claims,
    destination_bank: BankHandle,
    source_history: Vec<PlaneBuffer>,
    destination_history: Vec<PlaneBuffer>,
    source_recurrent: Rc<[Tensor]>,
    destination_recurrent: Rc<[Tensor]>,
    conversions: Vec<CodecConversionStep>,
    source_codec: KvCodec,
    destination_codec: KvCodec,
    rows: usize,
    _transaction: Transaction,
}

impl OwnedCodecAdvance {
    pub fn begin(
        source: SequenceState,
        destination: SequenceState,
    ) -> Result<Self, (SequenceState, SequenceState, Error)> {
        match Self::prepare(&source, &destination) {
            Ok((source_codec, destination_codec, from, rows)) => {
                let claims = match destination.store.reserve(None, rows) {
                    Ok(claims) => claims,
                    Err(error) => return Err((source, destination, error)),
                };
                let destination_bank = match destination.store.successor_bank() {
                    Ok(bank) => bank,
                    Err(error) => return Err((source, destination, error)),
                };
                let to = claims
                    .ranges()
                    .into_iter()
                    .flat_map(|(start, count)| start..start + count)
                    .collect::<Vec<_>>();
                let source_history = match source.store.history_planes() {
                    Ok(planes) => planes,
                    Err(error) => return Err((source, destination, error)),
                };
                let destination_history = match destination.store.history_planes() {
                    Ok(planes) => planes,
                    Err(error) => return Err((source, destination, error)),
                };
                let conversions = conversion_steps(&source, &destination, &from, &to);
                let transaction = destination.store.begin_transaction();
                Ok(Self {
                    source_recurrent: source.store.recurrent_arenas(),
                    destination_recurrent: destination.store.recurrent_arenas(),
                    source,
                    destination,
                    claims,
                    destination_bank,
                    source_history,
                    destination_history,
                    conversions,
                    source_codec,
                    destination_codec,
                    rows,
                    _transaction: transaction,
                })
            }
            Err(error) => Err((source, destination, error)),
        }
    }

    fn prepare(
        source: &SequenceState,
        destination: &SequenceState,
    ) -> Result<(KvCodec, KvCodec, Vec<usize>, usize), Error> {
        if Rc::ptr_eq(&source.store, &destination.store)
            || !Rc::ptr_eq(&source.store.device, &destination.store.device)
            || destination.position != 0
            || !destination.claims.is_empty()
            || destination.history_start != 0
            || source.store.component_specs != destination.store.component_specs
            || source.position > destination.store.context_capacity
            || source.expected_end > destination.store.context_capacity
        {
            return Err(Error::Request(
                "codec conversion requires a fresh compatible destination on the same device"
                    .into(),
            ));
        }
        let source_components = &source.store.components;
        let destination_components = &destination.store.components;
        if source_components.is_empty() || source_components.len() != destination_components.len() {
            return Err(Error::Request(
                "codec conversion requires matching nonempty history layouts".into(),
            ));
        }
        let source_codec = identify_codec(source_components)?;
        let destination_codec = identify_codec(destination_components)?;
        if source_codec == destination_codec {
            return Err(Error::Request(
                "codec conversion requires distinct codecs".into(),
            ));
        }
        for (before, after) in source_components.iter().zip(destination_components) {
            if before.layer != after.layer
                || before.codec.key_width != after.codec.key_width
                || before.codec.value_width != after.codec.value_width
            {
                return Err(Error::Request(
                    "codec conversion layers or vector widths differ".into(),
                ));
            }
        }
        let from = source
            .history_ranges()
            .into_iter()
            .flat_map(|(start, count)| start..start + count)
            .collect::<Vec<_>>();
        if from.is_empty() {
            return Err(Error::Request(
                "codec conversion requires visible history rows".into(),
            ));
        }
        let rows = from.len();
        if rows > destination.store.history_capacity {
            return Err(Error::Request(
                "codec conversion exceeds destination history capacity".into(),
            ));
        }
        Ok((source_codec, destination_codec, from, rows))
    }

    pub fn source_belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.source.belongs_to(store)
    }
    pub fn destination_belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.destination.belongs_to(store)
    }
    pub fn source_codec(&self) -> KvCodec {
        self.source_codec
    }
    pub fn destination_codec(&self) -> KvCodec {
        self.destination_codec
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn conversions(&self) -> &[CodecConversionStep] {
        &self.conversions
    }
    pub fn bindings(&self) -> OwnedCodecBindings<'_> {
        OwnedCodecBindings {
            source_history: &self.source_history,
            destination_history: &self.destination_history,
            source_recurrent: &self.source_recurrent,
            source_bank: self.source.bank.index(),
            destination_recurrent: &self.destination_recurrent,
            destination_bank: self.destination_bank.index(),
            conversions: &self.conversions,
        }
    }
    pub fn abort(self) -> (SequenceState, SequenceState) {
        (self.source, self.destination)
    }
    pub fn commit(self) -> SequenceState {
        let Self {
            source,
            mut destination,
            claims,
            mut destination_bank,
            ..
        } = self;
        destination.position = source.position;
        destination.expected_end = source.expected_end;
        destination.history_start = source.history_start;
        destination.claims.append(claims);
        std::mem::swap(&mut destination.bank, &mut destination_bank);
        destination.tape = source.tape;
        destination
    }
}

fn identify_codec(components: &[ComponentDescriptor]) -> Result<KvCodec, Error> {
    for candidate in [KvCodec::Dense, KvCodec::AffineK8V4, KvCodec::RotatedK4V4] {
        if components.iter().all(|component| {
            let dense_dtype = match component.codec.key {
                Codec::Dense { dtype } => dtype,
                _ => seismic::DType::F16,
            };
            let expected = candidate.spec(
                dense_dtype,
                component.codec.key_width,
                component.codec.value_width,
            );
            component.codec == expected
        }) {
            return Ok(candidate);
        }
    }
    Err(Error::Request(
        "history layout is not one named codec policy".into(),
    ))
}

fn conversion_steps(
    source: &SequenceState,
    destination: &SequenceState,
    from: &[usize],
    to: &[usize],
) -> Vec<CodecConversionStep> {
    let mut steps = Vec::with_capacity(source.store.components.len() * 2);
    let mut source_base = 0;
    let mut destination_base = 0;
    for (before, after) in source
        .store
        .components
        .iter()
        .zip(&destination.store.components)
    {
        for vector in [VectorKind::Key, VectorKind::Value] {
            let source_planes = before
                .planes()
                .iter()
                .enumerate()
                .filter_map(|(index, plane)| {
                    (plane.vector == vector).then_some(source_base + index)
                })
                .collect();
            let destination_planes = after
                .planes()
                .iter()
                .enumerate()
                .filter_map(|(index, plane)| {
                    (plane.vector == vector).then_some(destination_base + index)
                })
                .collect();
            steps.push(CodecConversionStep {
                layer: before.layer,
                vector,
                source_codec: if vector == VectorKind::Key {
                    before.codec.key
                } else {
                    before.codec.value
                },
                destination_codec: if vector == VectorKind::Key {
                    after.codec.key
                } else {
                    after.codec.value
                },
                source_planes,
                destination_planes,
                from: from.to_vec(),
                to: to.to_vec(),
            });
        }
        source_base += before.planes().len();
        destination_base += after.planes().len();
    }
    steps
}

impl OwnedCompaction {
    /// Join the longest suffix of a history's runs (at least two) holding at
    /// most `max_rows` rows into one contiguous run: the recent small runs
    /// decode leaves, moved with one bounded copy. Histories below the
    /// segment limit need nothing.
    pub fn prepare(
        state: SequenceState,
        max_rows: usize,
    ) -> Result<OwnedCompactionPreparation, (SequenceState, Error)> {
        let ranges = state.history_ranges();
        if ranges.len() < MAX_VISIBLE_SEGMENTS {
            return Ok(OwnedCompactionPreparation::NotNeeded(state));
        }
        let visible_rows = ranges.iter().map(|(_, count)| count).sum::<usize>();
        let mut moved = 0;
        let mut suffix = 0;
        for (_, count) in ranges.iter().rev() {
            if moved + count > max_rows {
                break;
            }
            moved += count;
            suffix += 1;
        }
        let destination = (suffix >= 2)
            .then(|| state.store.reserve_contiguous(moved))
            .flatten();
        let Some(destination) = destination else {
            return Ok(OwnedCompactionPreparation::Deferred {
                state,
                segments: ranges.len(),
                visible_rows,
            });
        };
        let history = match state.store.history_planes() {
            Ok(history) => history,
            Err(error) => return Err((state, error)),
        };
        let from = ranges[ranges.len() - suffix..]
            .iter()
            .flat_map(|(start, count)| *start..*start + *count)
            .collect::<Vec<_>>();
        let to = destination
            .ranges()
            .into_iter()
            .flat_map(|(start, count)| start..start + count)
            .collect::<Vec<_>>();
        let copies = state
            .store
            .components
            .iter()
            .flat_map(ComponentDescriptor::planes)
            .enumerate()
            .map(|(plane_index, _)| PlaneCopy {
                plane_index,
                from: from.clone(),
                to: to.clone(),
            })
            .collect();
        let transaction = state.store.begin_transaction();
        Ok(OwnedCompactionPreparation::Ready(Self {
            state,
            destination,
            copies,
            history,
            _transaction: transaction,
        }))
    }

    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.state.belongs_to(store)
    }
    pub fn rows(&self) -> usize {
        self.destination.rows()
    }
    pub fn copies(&self) -> &[PlaneCopy] {
        &self.copies
    }
    pub fn bindings(&self) -> OwnedCompactionBindings<'_> {
        OwnedCompactionBindings {
            history: &self.history,
            copies: &self.copies,
        }
    }

    /// Called only after the owning state submission has physically finished.
    pub fn commit(self) -> SequenceState {
        let Self {
            mut state,
            destination,
            ..
        } = self;
        let kept = state.claims.rows() - destination.rows();
        drop(state.claims.split_off(kept));
        state.claims.append(destination);
        state
    }

    pub fn abort(self) -> SequenceState {
        self.state
    }
}

impl OwnedStateAdvance {
    /// An advance whose rows all commit. Returns ownership of the unchanged
    /// source state if reservation fails.
    pub fn begin(state: SequenceState, count: usize) -> Result<Self, (SequenceState, Error)> {
        Self::begin_speculative(state, count, count)
    }

    /// An advance of which only the first `committed` rows always commit
    /// (a verification: the anchor, then proposals). Its recurrent state is
    /// published after the committed rows and the later rows are recorded
    /// on the successor bank's tape, so any accepted prefix of at least
    /// `committed` rows commits as a version of that one bank, with no
    /// second pass.
    pub fn begin_speculative(
        state: SequenceState,
        count: usize,
        committed: usize,
    ) -> Result<Self, (SequenceState, Error)> {
        if count == 0 || count > state.store.context_capacity.saturating_sub(state.position) {
            return Err((state, Error::from("advance exceeds context capacity")));
        }
        if committed == 0 || committed > count {
            return Err((
                state,
                Error::from("committed rows must be a nonempty prefix of the advance"),
            ));
        }
        let claims = match state.store.reserve(Some(&state.claims), count) {
            Ok(claims) => claims,
            Err(error) => return Err((state, error)),
        };
        let following = match state.store.successor_bank() {
            Ok(following) => following,
            Err(error) => return Err((state, error)),
        };
        let history = match state.store.history_planes() {
            Ok(history) => history,
            Err(error) => return Err((state, error)),
        };
        let destinations = claims
            .ranges()
            .into_iter()
            .flat_map(|(start, count)| start..start + count)
            .collect();
        let recurrent = state.store.recurrent_arenas();
        let transaction = state.store.begin_transaction();
        Ok(Self {
            state,
            count,
            committed,
            claims,
            following,
            history,
            recurrent,
            destinations,
            _transaction: transaction,
        })
    }

    pub fn position(&self) -> usize {
        self.state.position
    }

    pub fn rows(&self) -> usize {
        self.count
    }

    /// A launch must join every advance with the exact state arena selected
    /// by its executor, not merely an arena on the same device.
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.state.belongs_to(store)
    }

    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.state.history_ranges()
    }

    pub fn bindings(&self) -> OwnedAdvanceBindings<'_> {
        OwnedAdvanceBindings {
            recurrent: &self.recurrent,
            previous_bank: self.state.bank.index(),
            previous_tape: self.state.tape,
            following_bank: self.following.index(),
            stop: self.committed,
            history: &self.history,
            destinations: &self.destinations,
        }
    }

    /// Abort a submission that did not complete successfully and recover its
    /// unchanged accepted state. Dropping the transaction also releases every
    /// tentative claim, for terminal teardown that needs no source recovery.
    pub fn abort(self) -> SequenceState {
        self.state
    }

    pub fn commit_all(self) -> Result<OwnedAdvanceResolution, (SequenceState, Error)> {
        let rows = self.count;
        self.commit(rows)
    }

    /// Publish exactly the accepted physical prefix. Recurrent state commits
    /// as version (successor bank, accepted - committed): an accepted prefix
    /// shorter than the committed rows has no recurrent version and is
    /// refused. Stores without recurrent state commit any prefix.
    pub fn commit(self, accepted: usize) -> Result<OwnedAdvanceResolution, (SequenceState, Error)> {
        let Self {
            mut state,
            count,
            committed,
            claims,
            mut following,
            ..
        } = self;
        if accepted > count {
            return Err((
                state,
                Error::from("accepted prefix exceeds advance row count"),
            ));
        }
        if accepted == 0 {
            return Ok(OwnedAdvanceResolution::Aborted(state));
        }
        let tape = if state.store.has_recurrent_components() {
            let Some(tape) = accepted.checked_sub(committed) else {
                return Err((
                    state,
                    Error::from("accepted prefix ends before the published recurrent state"),
                ));
            };
            tape
        } else {
            0
        };
        let mut kept = claims;
        drop(kept.split_off(accepted));
        install_commit(&mut state, kept, &mut following, tape, accepted);
        Ok(OwnedAdvanceResolution::Committed(state))
    }
}

/// The tape rows of the version an advance of `count` rows publishes when
/// all of them commit.
fn published_tape(store: &StateStore, count: usize, committed: usize) -> usize {
    if store.has_recurrent_components() {
        count - committed
    } else {
        0
    }
}

impl OwnedStateAdvance {
    /// Tentative rows following this advance's rows, formed before this
    /// advance is reconciled; see [`OwnedSuccessorAdvance`].
    pub fn successor(&self, count: usize) -> Result<OwnedSuccessorAdvance, Error> {
        let mut ranges = self.state.history_ranges();
        append_ranges(&mut ranges, self.claims.ranges());
        OwnedSuccessorAdvance::reserve(
            &self.state.store,
            Predecessor {
                end: self.state.position + self.count,
                bank: self.following.index(),
                tape: published_tape(&self.state.store, self.count, self.committed),
                claims: &self.claims,
                ranges,
                history: &self.history,
                recurrent: &self.recurrent,
            },
            count,
        )
    }
}

/// What a successor needs of the tentative rows it follows.
struct Predecessor<'a> {
    /// The position after the predecessor's rows.
    end: usize,
    /// The version the predecessor publishes when all its rows commit.
    bank: usize,
    tape: usize,
    claims: &'a Claims,
    /// Visible history once the predecessor commits.
    ranges: Vec<(usize, usize)>,
    history: &'a [PlaneBuffer],
    recurrent: &'a Rc<[Tensor]>,
}

/// Tentative rows that follow an in-flight advance (or successor) before it
/// is reconciled, so the next step can be submitted while the previous one
/// executes. It reads the predecessor's published bank and history and
/// writes only its own rows and bank; its source state is the predecessor's
/// committed state, joined by [`OwnedSuccessorAdvance::attach`] once the
/// predecessor committed every row. Dropping it releases its rows and bank.
/// It holds a transaction, so the backing is never recommitted under it.
pub struct OwnedSuccessorAdvance {
    store: Rc<StateStore>,
    position: usize,
    previous_bank: usize,
    previous_tape: usize,
    ranges: Vec<(usize, usize)>,
    count: usize,
    claims: Claims,
    following: BankHandle,
    history: Vec<PlaneBuffer>,
    recurrent: Rc<[Tensor]>,
    destinations: Vec<usize>,
    transaction: Transaction,
}

impl OwnedSuccessorAdvance {
    fn reserve(
        store: &Rc<StateStore>,
        predecessor: Predecessor<'_>,
        count: usize,
    ) -> Result<Self, Error> {
        if count == 0 || count > store.context_capacity.saturating_sub(predecessor.end) {
            return Err(Error::from("successor advance exceeds context capacity"));
        }
        if predecessor.ranges.len() >= MAX_VISIBLE_SEGMENTS {
            return Err(Error::from("successor advance would exceed the segment limit"));
        }
        let claims = store.reserve(Some(predecessor.claims), count)?;
        let following = store.successor_bank()?;
        let destinations = claims
            .ranges()
            .into_iter()
            .flat_map(|(start, count)| start..start + count)
            .collect();
        Ok(Self {
            store: store.clone(),
            position: predecessor.end,
            previous_bank: predecessor.bank,
            previous_tape: predecessor.tape,
            ranges: predecessor.ranges,
            count,
            claims,
            following,
            history: predecessor.history.to_vec(),
            recurrent: predecessor.recurrent.clone(),
            destinations,
            transaction: store.begin_transaction(),
        })
    }

    /// Tentative rows following this successor's rows.
    pub fn successor(&self, count: usize) -> Result<OwnedSuccessorAdvance, Error> {
        let mut ranges = self.ranges.clone();
        append_ranges(&mut ranges, self.claims.ranges());
        Self::reserve(
            &self.store,
            Predecessor {
                end: self.position + self.count,
                bank: self.following.index(),
                tape: 0,
                claims: &self.claims,
                ranges,
                history: &self.history,
                recurrent: &self.recurrent,
            },
            count,
        )
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn rows(&self) -> usize {
        self.count
    }

    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        Rc::ptr_eq(&self.store, store)
    }

    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.ranges.clone()
    }

    pub fn bindings(&self) -> OwnedAdvanceBindings<'_> {
        OwnedAdvanceBindings {
            recurrent: &self.recurrent,
            previous_bank: self.previous_bank,
            previous_tape: self.previous_tape,
            following_bank: self.following.index(),
            stop: self.count,
            history: &self.history,
            destinations: &self.destinations,
        }
    }

    /// Join the predecessor's committed state: an ordinary advance over the
    /// rows and bank this successor reserved. `state` must be exactly the
    /// state the predecessor published (position, version and history).
    pub fn attach(self, state: SequenceState) -> Result<OwnedStateAdvance, (SequenceState, Error)> {
        if !state.belongs_to(&self.store)
            || state.position != self.position
            || state.bank.index() != self.previous_bank
            || state.tape != self.previous_tape
            || state.history_ranges() != self.ranges
        {
            return Err((
                state,
                Error::from("successor advance does not follow the committed state"),
            ));
        }
        let Self {
            count,
            claims,
            following,
            history,
            recurrent,
            destinations,
            transaction,
            ..
        } = self;
        Ok(OwnedStateAdvance {
            state,
            count,
            committed: count,
            claims,
            following,
            history,
            recurrent,
            destinations,
            _transaction: transaction,
        })
    }
}

/// The tentative rows of one launch slot: an advance of accepted state, or a
/// successor of an advance still in flight.
pub enum TentativeAdvance {
    Accepted(OwnedStateAdvance),
    Successor(OwnedSuccessorAdvance),
}

impl TentativeAdvance {
    pub fn position(&self) -> usize {
        match self {
            Self::Accepted(advance) => advance.position(),
            Self::Successor(advance) => advance.position(),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Accepted(advance) => advance.rows(),
            Self::Successor(advance) => advance.rows(),
        }
    }

    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        match self {
            Self::Accepted(advance) => advance.belongs_to(store),
            Self::Successor(advance) => advance.belongs_to(store),
        }
    }

    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Accepted(advance) => advance.history_ranges(),
            Self::Successor(advance) => advance.history_ranges(),
        }
    }

    pub fn bindings(&self) -> OwnedAdvanceBindings<'_> {
        match self {
            Self::Accepted(advance) => advance.bindings(),
            Self::Successor(advance) => advance.bindings(),
        }
    }

    pub fn successor(&self, count: usize) -> Result<OwnedSuccessorAdvance, Error> {
        match self {
            Self::Accepted(advance) => advance.successor(count),
            Self::Successor(advance) => advance.successor(count),
        }
    }
}

pub enum OwnedAdvanceResolution {
    Aborted(SequenceState),
    Committed(SequenceState),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BankCapacity, CodecSpec, ComponentDescriptor, ComponentSpec, LayerRef, StateStore,
    };
    use seismic::{BackendName, DType, DeviceCatalog};

    #[test]
    fn abort_recovers_source_and_releases_tentative_capacity() {
        let Some(device) = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
        else {
            return;
        };
        let store = StateStore::new(
            Rc::new(device),
            4,
            4,
            vec![
                ComponentDescriptor::new(LayerRef::Target(0), CodecSpec::dense(DType::F32, 1, 1), 1)
                    .unwrap(),
            ],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        assert_eq!(advance.bindings().destinations.len(), 2);
        assert_eq!(store.occupied_rows(), 2);
        let state = advance.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(store.occupied_rows(), 0);

        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!(state.position(), 2);
        assert_eq!(store.occupied_rows(), 2);
    }

    #[test]
    fn successors_follow_in_flight_advances_and_attach_to_their_commits() {
        let Some(device) = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
        else {
            return;
        };
        let store = StateStore::new(
            Rc::new(device),
            8,
            8,
            vec![
                ComponentDescriptor::new(LayerRef::Target(0), CodecSpec::dense(DType::F32, 1, 1), 1)
                    .unwrap(),
            ],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 3,
                retained: 0,
            },
        )
        .unwrap();
        let state = store.create().unwrap();
        let first = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let second = first.successor(1).unwrap();
        let third = second.successor(1).unwrap();
        assert_eq!((second.position(), third.position()), (2, 3));
        assert_eq!(second.bindings().previous_bank, first.bindings().following_bank);
        assert_eq!(third.bindings().previous_bank, second.bindings().following_bank);
        assert_eq!(second.history_ranges(), first.bindings().destinations.iter().map(|&row| (row, 1)).fold(
            Vec::new(),
            |mut ranges, range| {
                append_ranges(&mut ranges, [range]);
                ranges
            },
        ));
        // Rows follow their predecessors physically: one run.
        assert_eq!(second.bindings().destinations, [first.bindings().destinations[1] + 1]);
        assert_eq!(third.bindings().destinations, [second.bindings().destinations[0] + 1]);
        assert_eq!(store.occupied_rows(), 4);

        let OwnedAdvanceResolution::Committed(state) = first.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        // A successor attaches only to the state its predecessor published.
        let (state, _) = third.attach(state).err().unwrap();
        let second = second.attach(state).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = second.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!(state.position(), 3);
        assert_eq!(state.history_ranges().len(), 1);
        // The dropped third successor released its row and bank.
        assert_eq!(store.occupied_rows(), 3);
        let next = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let orphan = next.successor(1).unwrap();
        drop(orphan);
        assert_eq!(store.occupied_rows(), 4);
        let state = next.abort();
        assert_eq!(state.position(), 3);
        assert_eq!(store.occupied_rows(), 3);
    }

    #[test]
    fn codec_conversion_reserves_and_publishes_only_after_completion() {
        let Some(device) = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
        else {
            return;
        };
        let device = Rc::new(device);
        let source_store = StateStore::new(
            device.clone(),
            4,
            4,
            vec![
                ComponentDescriptor::new(LayerRef::Target(0), CodecSpec::dense(DType::F16, 32, 32), 1)
                    .unwrap(),
            ],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let destination_store = StateStore::new(
            device,
            4,
            4,
            vec![ComponentDescriptor::new(
                LayerRef::Target(0),
                KvCodec::AffineK8V4.spec(DType::F16, 32, 32),
                1,
            )
            .unwrap()],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let source = OwnedStateAdvance::begin(source_store.create().unwrap(), 2)
            .ok()
            .unwrap();
        let OwnedAdvanceResolution::Committed(source) = source.commit_all().ok().unwrap() else {
            panic!("source rows must commit");
        };
        let destination = destination_store.create().unwrap();
        let conversion = OwnedCodecAdvance::begin(source, destination).ok().unwrap();
        assert_eq!(conversion.rows(), 2);
        assert_eq!(conversion.source_codec(), KvCodec::Dense);
        assert_eq!(conversion.destination_codec(), KvCodec::AffineK8V4);
        assert_eq!(conversion.conversions().len(), 2);
        // Codes and coefficient planes per vector kind.
        assert_eq!(conversion.bindings().destination_history.len(), 4);
        assert_eq!(destination_store.occupied_rows(), 2);
        let (source, destination) = conversion.abort();
        assert_eq!(source.position(), 2);
        assert_eq!(destination.position(), 0);
        assert_eq!(destination_store.occupied_rows(), 0);
        let conversion = OwnedCodecAdvance::begin(source, destination).ok().unwrap();
        let destination = conversion.commit();
        assert_eq!(destination.position(), 2);
        assert_eq!(
            destination
                .history_ranges()
                .iter()
                .map(|(_, rows)| rows)
                .sum::<usize>(),
            2
        );
        assert_eq!(destination_store.occupied_rows(), 2);
    }

    fn recurrent_store(context: usize, in_flight: usize) -> Option<Rc<StateStore>> {
        let device = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())?;
        Some(
            StateStore::new(
                Rc::new(device),
                context,
                context,
                vec![ComponentDescriptor::new(
                    LayerRef::Target(0),
                    CodecSpec::dense(DType::F32, 1, 1),
                    1,
                )
                .unwrap()],
                vec![ComponentSpec {
                    shape: vec![1],
                    dtype: DType::F32,
                }],
                BankCapacity {
                    active: 2,
                    in_flight,
                    retained: 0,
                },
            )
            .unwrap(),
        )
    }

    /// A verification publishes its recurrent state after its committed rows
    /// and records the rest on the tape: any accepted prefix commits as
    /// version (successor bank, accepted - committed), read by the next
    /// advance, with no second pass.
    #[test]
    fn speculative_prefixes_commit_as_tape_versions() {
        let Some(store) = recurrent_store(8, 2) else {
            return;
        };
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin_speculative(state, 4, 1).ok().unwrap();
        let bindings = advance.bindings();
        assert_eq!((bindings.stop, bindings.previous_tape), (1, 0));
        let following = bindings.following_bank;
        let OwnedAdvanceResolution::Committed(state) = advance.commit(3).ok().unwrap() else {
            panic!("an accepted prefix past the committed rows commits");
        };
        assert_eq!((state.position(), state.bank_index(), state.tape_rows()), (3, following, 2));
        assert_eq!(store.occupied_rows(), 3);

        // The next advance reads the version; a checkpoint and its forks keep it.
        let checkpoint = state.checkpoint();
        assert_eq!(checkpoint.tape_rows(), 2);
        assert_eq!(checkpoint.fork().tape_rows(), 2);
        let advance = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let bindings = advance.bindings();
        assert_eq!(
            (bindings.previous_bank, bindings.previous_tape, bindings.stop),
            (following, 2, 1)
        );
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("a plain advance commits");
        };
        assert_eq!((state.position(), state.tape_rows()), (4, 0));

        // A prefix ending before the published state has no version.
        let advance = OwnedStateAdvance::begin_speculative(state, 3, 2).ok().unwrap();
        let (state, _) = advance.commit(1).err().unwrap();
        assert_eq!((state.position(), state.tape_rows()), (4, 0));
        assert_eq!(store.occupied_rows(), 4);
        let advance = OwnedStateAdvance::begin_speculative(state, 3, 2).ok().unwrap();
        let OwnedAdvanceResolution::Aborted(state) = advance.commit(0).ok().unwrap() else {
            panic!("an empty prefix aborts");
        };
        assert_eq!(state.position(), 4);
        assert!(OwnedStateAdvance::begin_speculative(state, 2, 0).is_err());
    }

    /// A successor of a verification follows the version the verification
    /// publishes when every row is accepted, and attaches only to it.
    #[test]
    fn successors_follow_the_tape_version_of_a_speculative_advance() {
        let Some(store) = recurrent_store(8, 3) else {
            return;
        };
        let first = OwnedStateAdvance::begin_speculative(store.create().unwrap(), 3, 1)
            .ok()
            .unwrap();
        let second = first.successor(1).unwrap();
        let bindings = second.bindings();
        assert_eq!(
            (bindings.previous_bank, bindings.previous_tape, bindings.stop),
            (first.bindings().following_bank, 2, 1)
        );
        let OwnedAdvanceResolution::Committed(state) = first.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!(state.tape_rows(), 2);
        let second = second.attach(state).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = second.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!((state.position(), state.tape_rows()), (4, 0));

        // A partially accepted verification publishes another version. A
        // launch commits the banks of both steps before either begins.
        store.provision(&[], 2).unwrap();
        let first = OwnedStateAdvance::begin_speculative(state, 3, 1).ok().unwrap();
        let second = first.successor(1).unwrap();
        let OwnedAdvanceResolution::Committed(state) = first.commit(2).ok().unwrap() else {
            panic!("an accepted prefix commits");
        };
        assert_eq!(state.tape_rows(), 1);
        assert!(second.attach(state).is_err());
    }
}
