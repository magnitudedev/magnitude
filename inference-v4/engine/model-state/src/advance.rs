//! Owned tentative state, movable through an in-flight program submission.

use super::{
    append_ranges, install_commit, BankHandle, Claims, Codec, ComponentDescriptor, Error, KvCodec, LayerRef,
    PlaneBuffer, PlaneCopy, SequenceState, StateStore, Transaction, VectorKind,
    MAX_VISIBLE_SEGMENTS,
};
use seismic::Tensor;
use std::rc::Rc;

/// A read-only view whose tensors and row claims are owned by the transaction.
/// `recurrent` holds one arena per recurrent component; the advance reads bank
/// `previous_bank` and writes only bank `following_bank` of each.
#[derive(Clone, Copy)]
pub struct OwnedAdvanceBindings<'a> {
    pub recurrent: &'a [Tensor],
    pub previous_bank: usize,
    pub following_bank: usize,
    pub history: &'a [PlaneBuffer],
    pub destinations: &'a [usize],
}

/// Tentative successor state. Moving this value into a launch and submission
/// preserves the source state, reserved extents, recurrent bank, and bindings
/// until physical completion and reconciliation.
pub struct OwnedStateAdvance {
    state: SequenceState,
    count: usize,
    claims: Claims,
    following: BankHandle,
    history: Vec<PlaneBuffer>,
    recurrent: Rc<[Tensor]>,
    destinations: Vec<usize>,
    transaction: Transaction,
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
    /// Returns ownership of the unchanged source state if reservation fails.
    pub fn begin(state: SequenceState, count: usize) -> Result<Self, (SequenceState, Error)> {
        if count == 0 || count > state.store.context_capacity.saturating_sub(state.position) {
            return Err((state, Error::from("advance exceeds context capacity")));
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
            claims,
            following,
            history,
            recurrent,
            destinations,
            transaction,
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
            following_bank: self.following.index(),
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

    /// Publish exactly the accepted physical prefix. A recurrent interior
    /// prefix yields owned repair work before any successor becomes visible.
    pub fn commit(self, accepted: usize) -> Result<OwnedAdvanceResolution, (SequenceState, Error)> {
        let Self {
            mut state,
            count,
            claims,
            mut following,
            history: _,
            recurrent,
            destinations: _,
            transaction,
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
        if accepted == count {
            install_commit(&mut state, claims, &mut following, count);
            return Ok(OwnedAdvanceResolution::Committed(state));
        }
        let mut kept = claims;
        drop(kept.split_off(accepted));
        if !state.store.has_recurrent_components() {
            install_commit(&mut state, kept, &mut following, accepted);
            return Ok(OwnedAdvanceResolution::Committed(state));
        }
        Ok(OwnedAdvanceResolution::Repair(OwnedRepairAdvance {
            state,
            accepted,
            claims: kept,
            following,
            recurrent,
            _transaction: transaction,
        }))
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
    /// The bank the predecessor publishes.
    bank: usize,
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
            following_bank: self.following.index(),
            history: &self.history,
            destinations: &self.destinations,
        }
    }

    /// Join the predecessor's committed state: an ordinary advance over the
    /// rows and bank this successor reserved. `state` must be exactly the
    /// state the predecessor published (position, bank and history).
    pub fn attach(self, state: SequenceState) -> Result<OwnedStateAdvance, (SequenceState, Error)> {
        if !state.belongs_to(&self.store)
            || state.position != self.position
            || state.bank.index() != self.previous_bank
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
            claims,
            following,
            history,
            recurrent,
            destinations,
            transaction,
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
    Repair(OwnedRepairAdvance),
}

/// Recompute the recurrent successor of an accepted interior prefix using an
/// ordinary state-program submission, then publish it atomically.
pub struct OwnedRepairAdvance {
    state: SequenceState,
    accepted: usize,
    claims: Claims,
    following: BankHandle,
    recurrent: Rc<[Tensor]>,
    _transaction: Transaction,
}

impl OwnedRepairAdvance {
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.state.belongs_to(store)
    }

    pub fn rows(&self) -> usize {
        self.accepted
    }

    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.state.history_ranges()
    }

    /// One arena per recurrent component; repair reads `previous_bank` and
    /// writes only `following_bank`.
    pub fn recurrent(&self) -> &[Tensor] {
        &self.recurrent
    }

    pub fn previous_bank(&self) -> usize {
        self.state.bank.index()
    }

    pub fn following_bank(&self) -> usize {
        self.following.index()
    }

    pub fn commit(self) -> SequenceState {
        let Self {
            mut state,
            accepted,
            claims,
            mut following,
            ..
        } = self;
        install_commit(&mut state, claims, &mut following, accepted);
        state
    }

    pub fn abort(self) -> SequenceState {
        self.state
    }
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
                ComponentDescriptor::new(LayerRef::Target(0), CodecSpec::dense(DType::F16, 8, 8), 1)
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
                KvCodec::AffineK8V4.spec(DType::F16, 8, 8),
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

    #[test]
    fn recurrent_interior_prefix_remains_unpublished_until_repair() {
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
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Repair(repair) = advance.commit(1).ok().unwrap() else {
            panic!("interior recurrent prefix must require repair");
        };
        assert_eq!(repair.rows(), 1);
        assert_eq!(store.occupied_rows(), 1);
        let state = repair.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(store.occupied_rows(), 0);
    }
}
