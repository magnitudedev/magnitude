//! Owned tentative state, movable through an in-flight program submission.

use super::{
    append_ranges, install_commit, ranges_from, BankHandle, Claims, Codec, ComponentDescriptor,
    ComponentRole, Error, HistoryDomainId, HistoryValue, KvCodec, LayerRef, LayoutError,
    PlaneBuffer, SequenceState, StateStore, StoreCopy, Tail, VectorKind,
};
use seismic::Tensor;
use std::{cell::Cell, rc::Rc};

/// A read-only view whose tensors and row claims are owned by the advance.
/// `recurrent` holds one arena per recurrent component; the advance reads
/// version (`previous_bank`, `previous_tape`) and writes only bank
/// `following_bank` of each. It publishes its recurrent state after `stop`
/// rows and records the tape of the rows after it. `history` holds the
/// planes of every stored domain ([`StateStore::history_planes`]);
/// `destinations[domain]` holds the advance's fresh rows in that domain, one
/// per advance row (for Block, one per block the advance completes, in
/// order).
#[derive(Clone, Copy)]
pub struct OwnedAdvanceBindings<'a> {
    pub recurrent: &'a [Tensor],
    pub previous_bank: usize,
    pub previous_tape: usize,
    pub following_bank: usize,
    pub stop: usize,
    pub history: &'a [PlaneBuffer],
    pub destinations: &'a [Vec<usize>],
}

/// Each domain's reserved rows, as row indices in logical order.
fn destinations(claims: &[Claims]) -> Vec<Vec<usize>> {
    claims
        .iter()
        .map(|claims| {
            claims
                .ranges()
                .into_iter()
                .flat_map(|(start, count)| start..start + count)
                .collect()
        })
        .collect()
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
    /// The longest prefix a commit may accept: the committed rows and the
    /// acceptable rows, never a front row.
    acceptable: usize,
    /// The advance's tentative rows, per stored domain.
    claims: Vec<Claims>,
    following: BankHandle,
    history: Vec<PlaneBuffer>,
    recurrent: Rc<[Tensor]>,
    destinations: Vec<Vec<usize>>,
}

/// Tentative copy of one domain's partial last page into a free page, for a
/// history that cannot grow in place because another history (a sibling
/// fork) continued that page. Its source rows remain accepted until the
/// copy ([`OwnedTailRelocation::copy`], run like any store copy) completes
/// and the relocation commits.
pub struct OwnedTailRelocation {
    state: SequenceState,
    domain: HistoryDomainId,
    destination: Claims,
    copy: StoreCopy,
}

/// One semantic vector conversion. A codec can use several physical planes;
/// their indices are carried together rather than mistaken for one copy plane.
/// Planes index the stores' [`StateStore::history_planes`]; rows are rows of
/// the layer's `domain`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecConversionStep {
    pub domain: HistoryDomainId,
    pub layer: LayerRef,
    pub vector: VectorKind,
    pub source_codec: Codec,
    pub destination_codec: Codec,
    pub source_planes: Vec<usize>,
    pub destination_planes: Vec<usize>,
    pub from: Vec<usize>,
    pub to: Vec<usize>,
}

/// Physical buffers remain pinned by the owned conversion through finish.
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
    claims: Vec<Claims>,
    destination_bank: BankHandle,
    source_history: Vec<PlaneBuffer>,
    destination_history: Vec<PlaneBuffer>,
    source_recurrent: Rc<[Tensor]>,
    destination_recurrent: Rc<[Tensor]>,
    conversions: Vec<CodecConversionStep>,
    source_codec: KvCodec,
    destination_codec: KvCodec,
    rows: usize,
}

impl OwnedCodecAdvance {
    pub fn begin(
        source: SequenceState,
        destination: SequenceState,
    ) -> Result<Self, (SequenceState, SequenceState, Error)> {
        match Self::prepare(&source, &destination) {
            Ok((source_codec, destination_codec, from)) => {
                let counts = from.iter().map(Vec::len).collect::<Vec<_>>();
                let claims = match destination
                    .snapshot
                    .store
                    .reserve_rows(&vec![Tail::NEW; counts.len()], &counts)
                {
                    Ok(claims) => claims,
                    Err(error) => return Err((source, destination, error)),
                };
                let destination_bank = match destination.snapshot.store.successor_bank() {
                    Ok(bank) => bank,
                    Err(error) => return Err((source, destination, error)),
                };
                let to = destinations(&claims);
                let rows = match counts
                    .iter()
                    .try_fold(0usize, |total, &count| total.checked_add(count))
                {
                    Some(rows) => rows,
                    None => {
                        return Err((
                            source,
                            destination,
                            LayoutError::ArithmeticOverflow("codec conversion row count").into(),
                        ));
                    }
                };
                let source_history = match source.snapshot.store.history_planes() {
                    Ok(planes) => planes,
                    Err(error) => return Err((source, destination, error)),
                };
                let destination_history = match destination.snapshot.store.history_planes() {
                    Ok(planes) => planes,
                    Err(error) => return Err((source, destination, error)),
                };
                let conversions = conversion_steps(&source, &destination, &from, &to);
                Ok(Self {
                    source_recurrent: source.snapshot.store.recurrent_arenas(),
                    destination_recurrent: destination.snapshot.store.recurrent_arenas(),
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
                })
            }
            Err(error) => Err((source, destination, error)),
        }
    }

    /// Check a conversion and return both codecs and, per stored domain, the
    /// source rows to convert.
    fn prepare(
        source: &SequenceState,
        destination: &SequenceState,
    ) -> Result<(KvCodec, KvCodec, Vec<Vec<usize>>), Error> {
        if Rc::ptr_eq(&source.snapshot.store, &destination.snapshot.store)
            || !source
                .snapshot
                .store
                .device
                .same_device(&destination.snapshot.store.device)
            || destination.snapshot.position != 0
            || destination
                .snapshot
                .domains
                .iter()
                .any(|domain| !domain.claims.is_empty())
            || destination.snapshot.history_start != 0
            || source.snapshot.store.component_specs != destination.snapshot.store.component_specs
            || source.snapshot.store.shared != destination.snapshot.store.shared
            || source.snapshot.position > destination.snapshot.store.context_capacity
            || source.expected_end > destination.snapshot.store.context_capacity
        {
            return Err(Error::Request(
                "codec conversion requires a fresh compatible destination on the same device"
                    .into(),
            ));
        }
        let source_domains = &source.snapshot.store.domains;
        let destination_domains = &destination.snapshot.store.domains;
        if source_domains.is_empty()
            || source_domains.len() != destination_domains.len()
            || source_domains
                .iter()
                .zip(destination_domains)
                .any(|(before, after)| {
                    before.kind != after.kind || before.components.len() != after.components.len()
                })
        {
            return Err(Error::Request(
                "codec conversion requires matching nonempty history layouts".into(),
            ));
        }
        let source_components = source_domains
            .iter()
            .flat_map(|domain| domain.components.iter().cloned())
            .collect::<Vec<_>>();
        let destination_components = destination_domains
            .iter()
            .flat_map(|domain| domain.components.iter().cloned())
            .collect::<Vec<_>>();
        let source_codec = identify_codec(&source_components)?;
        let destination_codec = identify_codec(&destination_components)?;
        if source_codec == destination_codec {
            return Err(Error::Request(
                "codec conversion requires distinct codecs".into(),
            ));
        }
        for (before, after) in source_components.iter().zip(&destination_components) {
            let widths = |component: &ComponentDescriptor| {
                component
                    .codec
                    .vectors()
                    .map(|(vector, _, width)| (vector, width))
                    .collect::<Vec<_>>()
            };
            if before.layer != after.layer || widths(before) != widths(after) {
                return Err(Error::Request(
                    "codec conversion layers or vector widths differ".into(),
                ));
            }
        }
        let from = source
            .domain_ranges()
            .into_iter()
            .map(|ranges| {
                ranges
                    .into_iter()
                    .flat_map(|(start, count)| start..start + count)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if from.iter().all(Vec::is_empty) {
            return Err(Error::Request(
                "codec conversion requires visible history rows".into(),
            ));
        }
        if from
            .iter()
            .zip(destination_domains)
            .any(|(rows, domain)| rows.len() > domain.capacity)
        {
            return Err(Error::Request(
                "codec conversion exceeds destination history capacity".into(),
            ));
        }
        Ok((source_codec, destination_codec, from))
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
            source_bank: self.source.bank_index(),
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
        destination.snapshot.position = source.snapshot.position;
        destination.expected_end = source.expected_end;
        destination.snapshot.history_start = source.snapshot.history_start;
        for ((dst, src), claims) in destination
            .snapshot
            .domains
            .iter_mut()
            .zip(&source.snapshot.domains)
            .zip(claims)
        {
            dst.start = src.start;
            dst.claims.append(claims);
        }
        std::mem::swap(&mut destination.snapshot.bank, &mut destination_bank);
        destination.snapshot.tape = source.snapshot.tape;
        destination
    }
}

fn identify_codec(components: &[ComponentDescriptor]) -> Result<KvCodec, Error> {
    for candidate in [KvCodec::Dense, KvCodec::AffineK8V4, KvCodec::RotatedK4V4] {
        // Produced rows and the model's own stores keep their codec whatever
        // the host policy; a host-policy history holds its own value vector
        // (stored keys are the model's own store, valueless rows produced).
        if components
            .iter()
            .filter(|component| component.role == ComponentRole::History)
            .all(|component| {
            let dense_dtype = match component.codec.key {
                Codec::Dense { dtype } => dtype,
                _ => seismic::DType::F16,
            };
            let HistoryValue::Own { width: values, .. } = component.codec.value else {
                panic!("a host-policy history component without its own values: {component:?}");
            };
            candidate.spec(dense_dtype, component.codec.key_width, values) == component.codec
        }) {
            return Ok(candidate);
        }
    }
    Err(Error::Request(
        "history layout is not one named codec policy".into(),
    ))
}

/// One step per (layer, vector) of every stored domain, carrying that
/// domain's rows. Plane indices number the planes of every domain in order.
fn conversion_steps(
    source: &SequenceState,
    destination: &SequenceState,
    from: &[Vec<usize>],
    to: &[Vec<usize>],
) -> Vec<CodecConversionStep> {
    let mut steps = Vec::new();
    let mut source_base = 0;
    let mut destination_base = 0;
    let pairs = source
        .snapshot
        .store
        .domains
        .iter()
        .zip(&destination.snapshot.store.domains)
        .enumerate()
        .flat_map(|(domain, (before, after))| {
            before
                .components
                .iter()
                .zip(&after.components)
                .map(move |pair| (domain, pair))
        });
    for (domain, (before, after)) in pairs {
        if from[domain].is_empty() {
            source_base += before.planes().len();
            destination_base += after.planes().len();
            continue;
        }
        // The vectors the rows store: equal on both sides (`prepare`).
        for (vector, source_codec, _) in before.codec.vectors() {
            let destination_codec = after
                .codec
                .codec(vector)
                .expect("codec conversion pairs layers storing the same vectors");
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
                domain: HistoryDomainId(domain),
                layer: before.layer,
                vector,
                source_codec,
                destination_codec,
                source_planes,
                destination_planes,
                from: from[domain].clone(),
                to: to[domain].clone(),
            });
        }
        source_base += before.planes().len();
        destination_base += after.planes().len();
    }
    steps
}

impl OwnedTailRelocation {
    /// Reserve a free page for a blocked last page of `domain` (see
    /// [`SequenceState::blocked_tails`]) and plan the copy of that page's
    /// rows of this history to its start. The caller has provisioned one
    /// free page (see [`SequenceState::relocation_demands`]); without one
    /// this is a capacity error and the state is unchanged.
    pub fn prepare(
        state: SequenceState,
        domain: HistoryDomainId,
    ) -> Result<Self, (SequenceState, Error)> {
        if !state.blocked_tails().contains(&domain) {
            return Err((
                state,
                Error::Request("only a blocked last page is relocated".into()),
            ));
        }
        let ranges = state.history_ranges(domain);
        let &(last, count) = ranges.last().expect("a blocked history has rows");
        let end = last + count;
        let page_rows = state.snapshot.store.history_page_rows(domain);
        // A page is filled in place from its first row, so the history's
        // rows of its last page are one run.
        let first = last.max((end - 1) / page_rows * page_rows);
        let moved = end - first;
        let Some(destination) = state.snapshot.store.reserve_page(domain, moved) else {
            let error = match (page_rows as u64)
                .checked_mul(state.snapshot.store.history_row_bytes(domain))
            {
                Some(required) => Error::Capacity {
                    required,
                    available_bytes: 0,
                },
                None => LayoutError::ArithmeticOverflow("relocation page bytes").into(),
            };
            return Err((state, error));
        };
        let to = destination
            .ranges()
            .into_iter()
            .flat_map(|(start, count)| start..start + count)
            .collect::<Vec<_>>();
        let copy = match state
            .snapshot
            .store
            .history_row_copy(domain, (first..end).collect(), to)
        {
            Ok(copy) => copy,
            Err(error) => return Err((state, error)),
        };
        Ok(Self {
            state,
            domain,
            destination,
            copy,
        })
    }

    /// The domain whose rows this relocation moves.
    pub fn domain(&self) -> HistoryDomainId {
        self.domain
    }
    /// The row copy to run before [`OwnedTailRelocation::commit`].
    pub fn copy(&self) -> &StoreCopy {
        &self.copy
    }

    /// Called only after the copy has physically finished.
    pub fn commit(self) -> SequenceState {
        let Self {
            mut state,
            domain,
            destination,
            ..
        } = self;
        let claims = &mut state.snapshot.domains[domain.0].claims;
        let kept = claims.rows() - destination.rows();
        drop(claims.split_off(kept));
        claims.append(destination);
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
        Self::begin_rows(state, count, 0, 0)
    }

    /// An advance of a pass's rows: its `committed` rows (at least one),
    /// which always commit, then `acceptable` rows that a commit may accept (a
    /// verification's proposals, a drafter's chained rows), then `front`
    /// rows that it never accepts (a two-row decode's front row). Its
    /// recurrent state is published after the committed rows and the later
    /// rows are recorded on the successor bank's tape, so any accepted
    /// prefix of at least the committed rows commits as a version of that
    /// one bank, with no second pass.
    pub fn begin_rows(
        state: SequenceState,
        committed: usize,
        acceptable: usize,
        front: usize,
    ) -> Result<Self, (SequenceState, Error)> {
        let Some(acceptable_end) = committed.checked_add(acceptable) else {
            return Err((
                state,
                LayoutError::ArithmeticOverflow("advance row count").into(),
            ));
        };
        let Some(count) = acceptable_end.checked_add(front) else {
            return Err((
                state,
                LayoutError::ArithmeticOverflow("advance row count").into(),
            ));
        };
        // Every pass commits a row: acceptable rows extend a commit of the
        // rows before them, and a front row runs beside the row it guesses
        // the successor of.
        if committed == 0 {
            return Err((state, Error::from("an advance commits rows")));
        }
        if count
            > state
                .snapshot
                .store
                .context_capacity
                .saturating_sub(state.snapshot.position)
        {
            return Err((state, Error::from("advance exceeds context capacity")));
        }
        let claims = match state
            .snapshot
            .store
            .reserve(&state.tails(), state.snapshot.position, count)
        {
            Ok(claims) => claims,
            Err(error) => return Err((state, error)),
        };
        let following = match state.snapshot.store.successor_bank() {
            Ok(following) => following,
            Err(error) => return Err((state, error)),
        };
        let history = match state.snapshot.store.history_planes() {
            Ok(history) => history,
            Err(error) => return Err((state, error)),
        };
        let destinations = destinations(&claims);
        let recurrent = state.snapshot.store.recurrent_arenas();
        Ok(Self {
            state,
            count,
            committed,
            acceptable: acceptable_end,
            claims,
            following,
            history,
            recurrent,
            destinations,
        })
    }

    pub fn position(&self) -> usize {
        self.state.position()
    }

    pub fn rows(&self) -> usize {
        self.count
    }

    /// A launch must join every advance with the exact state arena selected
    /// by its executor, not merely an arena on the same device.
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.state.belongs_to(store)
    }

    /// The accepted rows the advance reads in a domain, in logical order,
    /// from position [`OwnedStateAdvance::history_start`].
    pub fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        self.state.history_ranges(domain)
    }
    pub fn domain_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        self.state.domain_ranges()
    }
    pub fn history_start(&self, domain: HistoryDomainId) -> usize {
        self.state.history_start(domain)
    }
    /// The accepted rows of a domain at positions `from` and later.
    pub fn visible_ranges(&self, domain: HistoryDomainId, from: usize) -> Vec<(usize, usize)> {
        self.state.visible_ranges(domain, from)
    }
    pub fn span_count(&self) -> usize {
        self.state.span_count()
    }

    pub fn bindings(&self) -> OwnedAdvanceBindings<'_> {
        OwnedAdvanceBindings {
            recurrent: &self.recurrent,
            previous_bank: self.state.bank_index(),
            previous_tape: self.state.tape_rows(),
            following_bank: self.following.index(),
            stop: self.committed,
            history: &self.history,
            destinations: &self.destinations,
        }
    }

    /// Abort a submission that did not complete successfully and recover its
    /// unchanged accepted state. Dropping the advance also releases every
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
    /// refused. Stores without recurrent state commit any prefix. A front
    /// row is never accepted.
    pub fn commit(self, accepted: usize) -> Result<OwnedAdvanceResolution, (SequenceState, Error)> {
        let Self {
            mut state,
            count,
            committed,
            acceptable,
            mut claims,
            mut following,
            ..
        } = self;
        if accepted > count {
            return Err((
                state,
                Error::from("accepted prefix exceeds advance row count"),
            ));
        }
        if accepted > acceptable {
            return Err((state, Error::from("accepted prefix includes a front row")));
        }
        if accepted == 0 {
            return Ok(OwnedAdvanceResolution::Aborted(state));
        }
        let tape = if state.snapshot.store.has_recurrent_components() {
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
        keep_accepted(
            &state.snapshot.store,
            &mut claims,
            state.snapshot.position,
            accepted,
        );
        install_commit(&mut state, claims, &mut following, tape, accepted);
        Ok(OwnedAdvanceResolution::Committed(state))
    }

    /// The first chunk of a prompt group's chain behind it: `count` rows
    /// following this advance, which publishes its every row (a chain
    /// carries no speculation).
    pub fn chained(&self, count: usize) -> Result<OwnedChainedAdvance, Error> {
        chain_one(self.link(count)?)
    }

    fn link(&self, count: usize) -> Result<Link<'_>, Error> {
        if self.committed != self.count {
            return Err(Error::from("a chain follows an advance whose rows all commit"));
        }
        Ok(Link {
            predecessor: self.predecessor()?,
            count,
            reads: self.following.clone(),
            writes: None,
        })
    }
}

/// Keep, in each domain's `claims` of an advance of `accepted` rows from
/// `position`, the rows of that prefix: a Block domain the blocks it
/// completes.
fn keep_accepted(store: &StateStore, claims: &mut [Claims], position: usize, accepted: usize) {
    for (claims, domain) in claims.iter_mut().zip(&store.domains) {
        drop(claims.split_off(domain.kind.advance_rows(position, accepted)));
    }
}

impl OwnedStateAdvance {
    /// Tentative rows following this advance's rows, formed before this
    /// advance is reconciled, every one of which commits; see
    /// [`OwnedSuccessorAdvance`].
    pub fn successor(&self, count: usize) -> Result<OwnedSuccessorAdvance, Error> {
        follow_one(self.predecessor()?, count, 0, None)
    }

    /// What a successor follows: the state this advance publishes once its
    /// committed rows commit, its front rows' rows inherited. An advance
    /// with acceptable rows has none: what it commits is decided only when
    /// it is reconciled.
    fn predecessor(&self) -> Result<Predecessor<'_>, Error> {
        if self.acceptable != self.committed {
            return Err(Error::from("a successor follows an advance whose commit is decided by its reconciliation"));
        }
        let starts = self.state.history_starts();
        Ok(Predecessor::of(
            &self.state.snapshot.store,
            self.state.snapshot.position,
            self.committed,
            &self.claims,
            self.state.snapshot.history_start,
            &starts,
            self.state.domain_ranges(),
            self.following.index(),
            &self.history,
            &self.recurrent,
            self.state
                .snapshot
                .domains
                .iter()
                .map(|domain| domain.claims.id)
                .collect(),
        ))
    }
}

/// Each domain's history once a predecessor ending at `end` commits: its
/// accepted `ranges` (from `starts`) followed by the predecessor's committed
/// rows `committed` (in the domain's slabs), less the rows the domain no
/// longer references at `end` (exactly what committing trims).
fn following_history(
    store: &StateStore,
    floor: usize,
    starts: &[usize],
    ranges: Vec<Vec<(usize, usize)>>,
    committed: Vec<Vec<(usize, usize)>>,
    end: usize,
) -> (Vec<usize>, Vec<Vec<(usize, usize)>>) {
    store
        .domains
        .iter()
        .zip(starts)
        .zip(ranges)
        .zip(committed)
        .map(|(((domain, &start), mut ranges), committed)| {
            append_ranges(&mut ranges, committed, domain.slab_rows);
            let retained = domain.history_start(end, floor).max(start);
            (retained, ranges_from(&ranges, start, retained))
        })
        .unzip()
}

/// What a successor needs of the tentative rows it follows: the state they
/// publish once their committed rows commit. Their front rows never commit:
/// the successor's first rows take over their rows (`front`), each written
/// anew at the same position, so the successor grows in place as the next
/// pass would once the predecessor commits and its front rows' rows free.
struct Predecessor<'a> {
    store: &'a Rc<StateStore>,
    /// The position after the predecessor's committed rows.
    end: usize,
    /// The version the predecessor publishes when its committed rows
    /// commit: its successor bank, at tape 0 (every row after the committed
    /// ones is a front row, recorded on the tape but never accepted).
    bank: usize,
    /// The trim floor, and per domain the first position and the rows of
    /// the history once the predecessor commits.
    floor: usize,
    starts: Vec<usize>,
    ranges: Vec<Vec<(usize, usize)>>,
    /// Per domain, a reference of the successor's own on the predecessor's
    /// rows past its committed rows, in logical order.
    front: Vec<Claims>,
    history: &'a [PlaneBuffer],
    recurrent: &'a Rc<[Tensor]>,
    /// Per domain, the live history (its arena entry) the rows extend.
    live: Vec<u64>,
}

impl<'a> Predecessor<'a> {
    /// The predecessor an advance from `position` holding `claims` makes of
    /// its first `committed` rows, read from accepted `ranges` (from
    /// `starts`, above `floor`) and publishing `bank`.
    #[allow(clippy::too_many_arguments)]
    fn of(
        store: &'a Rc<StateStore>,
        position: usize,
        committed: usize,
        claims: &[Claims],
        floor: usize,
        starts: &[usize],
        ranges: Vec<Vec<(usize, usize)>>,
        bank: usize,
        history: &'a [PlaneBuffer],
        recurrent: &'a Rc<[Tensor]>,
        live: Vec<u64>,
    ) -> Self {
        let end = position + committed;
        let (kept, front): (Vec<_>, Vec<_>) = claims
            .iter()
            .zip(&store.domains)
            .map(|(claims, domain)| {
                // A reference of its own on every row, the committed rows'
                // dropped again with `kept`.
                let mut kept = claims.clone();
                let front = kept.split_off(domain.kind.advance_rows(position, committed));
                (kept.ranges(), front)
            })
            .unzip();
        let (starts, ranges) = following_history(store, floor, starts, ranges, kept, end);
        Self { store, end, bank, floor, starts, ranges, front, history, recurrent, live }
    }

    /// Per domain, where its successor's own rows grow, as the committed
    /// state would once its front rows' rows are taken over: after them,
    /// else after the history's last row, extending the live history.
    fn tails(&self) -> Vec<Tail> {
        self.ranges
            .iter()
            .zip(&self.front)
            .zip(&self.live)
            .map(|((ranges, front), &history)| Tail {
                end: front.end().or(ranges.last().map(|(start, count)| start + count)),
                history: Some(history),
            })
            .collect()
    }

    /// Fit the front rows' rows to a successor of `count` rows: it takes
    /// over only those its own rows reach, and its references on the rest
    /// drop (the running predecessor keeps its own), so a successor
    /// shallower than its predecessor's front costs nothing more.
    fn fit(&mut self, count: usize) {
        for (domain, front) in self.store.domains.iter().zip(&mut self.front) {
            let reached = domain.kind.advance_rows(self.end, count);
            if front.rows() > reached {
                drop(front.split_off(reached));
            }
        }
    }

    /// Per domain, the rows a successor of `count` rows reserves beyond the
    /// front rows' rows it takes over, fitted to it ([`Self::fit`]).
    fn counts(&self, count: usize) -> Vec<usize> {
        self.store
            .domains
            .iter()
            .zip(&self.front)
            .map(|(domain, front)| {
                domain
                    .kind
                    .advance_rows(self.end, count)
                    .checked_sub(front.rows())
                    .expect("invariant violated: a fitted successor takes over more front rows than it reaches")
            })
            .collect()
    }
}

/// The successors of one queued step, each of `committed` committed rows and
/// `front` front rows following its slot's predecessor and writing `writes`,
/// or a fresh successor bank. The step's demands are admitted together
/// before any row is reserved, as provisioning admits a step's
/// ([`StateStore::admit`]), so each successor takes exactly the pages its
/// step takes provisioned alone once its predecessors commit.
fn follow(
    mut slots: Vec<(Predecessor<'_>, usize, usize, Option<BankHandle>)>,
) -> Result<Vec<OwnedSuccessorAdvance>, Error> {
    let mut counts = Vec::with_capacity(slots.len());
    for (predecessor, committed, front, _) in &mut slots {
        let count = committed
            .checked_add(*front)
            .ok_or(LayoutError::ArithmeticOverflow("successor row count"))?;
        if *committed == 0
            || count > predecessor.store.context_capacity.saturating_sub(predecessor.end)
        {
            return Err(Error::from("successor advance exceeds context capacity"));
        }
        predecessor.fit(count);
        counts.push(count);
    }
    let Some((first, ..)) = slots.first() else {
        return Ok(Vec::new());
    };
    let store = first.store.clone();
    if slots
        .iter()
        .any(|(predecessor, ..)| !Rc::ptr_eq(predecessor.store, &store))
    {
        return Err(Error::from("one step's successors follow advances of one store"));
    }
    let demands = slots
        .iter()
        .zip(&counts)
        .flat_map(|((predecessor, ..), &count)| {
            store.demands(&predecessor.tails(), &predecessor.counts(count))
        })
        .collect::<Vec<_>>();
    let admission = Rc::new(Admission {
        slabs: store.admit(&demands)?,
        store: store.clone(),
        kept: Cell::new(false),
    });
    slots
        .into_iter()
        .zip(counts)
        .map(|((predecessor, committed, _, writes), count)| {
            OwnedSuccessorAdvance::reserve(predecessor, committed, count, writes, &admission)
        })
        .collect()
}

/// The one successor of a step of one slot, writing `writes` or a fresh
/// successor bank.
fn follow_one(
    predecessor: Predecessor<'_>,
    committed: usize,
    front: usize,
    writes: Option<BankHandle>,
) -> Result<OwnedSuccessorAdvance, Error> {
    Ok(follow(vec![(predecessor, committed, front, writes)])?
        .pop()
        .expect("one successor follows each predecessor"))
}

/// The slabs one queued step admitted to placement, shared by its
/// successors. Until one of them joins its predecessor's committed state the
/// step has not run: when the last drops, after its rows are released, the
/// slabs still empty return to reserve, as if the step had never been
/// provisioned. Once one attaches they stay, as a provisioned step's do.
struct Admission {
    store: Rc<StateStore>,
    slabs: Vec<(HistoryDomainId, usize)>,
    kept: Cell<bool>,
}

impl Drop for Admission {
    fn drop(&mut self) {
        if !self.kept.get() {
            self.store.withdraw(&self.slabs);
        }
    }
}

/// Tentative rows that follow an in-flight advance (or successor) before it
/// is reconciled, so the next step can be submitted while the previous one
/// executes. It reads the version the predecessor publishes once its
/// committed rows commit and that history, and writes only its own rows and
/// bank: rows it reserved, after the predecessor's front rows' rows, which
/// it takes over (each written anew at the same position, so it grows in
/// place as the next pass would). Its source state is the predecessor's
/// committed state, joined by [`OwnedSuccessorAdvance::attach`] once the
/// predecessor committed its committed rows. Its own rows are `committed`
/// rows that commit, then front rows that never do. Dropping it releases
/// its rows and bank. Its captured bindings pin slab placement until it
/// drops.
pub struct OwnedSuccessorAdvance {
    store: Rc<StateStore>,
    position: usize,
    previous_bank: usize,
    floor: usize,
    starts: Vec<usize>,
    ranges: Vec<Vec<(usize, usize)>>,
    count: usize,
    committed: usize,
    claims: Vec<Claims>,
    following: BankHandle,
    history: Vec<PlaneBuffer>,
    recurrent: Rc<[Tensor]>,
    destinations: Vec<Vec<usize>>,
    /// Per domain, the live history (its arena entry) its rows extend.
    live: Vec<u64>,
    /// Dropped last, after the rows it may return to reserve.
    admission: Rc<Admission>,
}

impl OwnedSuccessorAdvance {
    fn reserve(
        predecessor: Predecessor<'_>,
        committed: usize,
        count: usize,
        writes: Option<BankHandle>,
        admission: &Rc<Admission>,
    ) -> Result<Self, Error> {
        let store = predecessor.store;
        let reserved = store.reserve_rows(&predecessor.tails(), &predecessor.counts(count))?;
        let following = match writes {
            Some(bank) => bank,
            None => store.successor_bank()?,
        };
        let Predecessor { end, bank, floor, starts, ranges, front: mut claims, history, recurrent, live, .. } = predecessor;
        for (claims, reserved) in claims.iter_mut().zip(reserved) {
            claims.append(reserved);
        }
        let destinations = destinations(&claims);
        Ok(Self {
            store: store.clone(),
            position: end,
            previous_bank: bank,
            floor,
            starts,
            ranges,
            count,
            committed,
            claims,
            following,
            history: history.to_vec(),
            recurrent: recurrent.clone(),
            destinations,
            live,
            admission: admission.clone(),
        })
    }

    /// Tentative rows following this successor's committed rows, every one
    /// of which commits.
    pub fn successor(&self, count: usize) -> Result<OwnedSuccessorAdvance, Error> {
        follow_one(self.predecessor(), count, 0, None)
    }

    fn predecessor(&self) -> Predecessor<'_> {
        Predecessor::of(
            &self.store,
            self.position,
            self.committed,
            &self.claims,
            self.floor,
            &self.starts,
            self.ranges.clone(),
            self.following.index(),
            &self.history,
            &self.recurrent,
            self.live.clone(),
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

    /// The rows the successor reads in a domain once its predecessor
    /// commits, from position [`OwnedSuccessorAdvance::history_start`].
    pub fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        self.ranges[domain.0].clone()
    }
    pub fn domain_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        self.ranges.clone()
    }
    pub fn history_start(&self, domain: HistoryDomainId) -> usize {
        self.starts[domain.0]
    }
    pub fn visible_ranges(&self, domain: HistoryDomainId, from: usize) -> Vec<(usize, usize)> {
        ranges_from(&self.ranges[domain.0], self.starts[domain.0], from)
    }
    pub fn span_count(&self) -> usize {
        self.ranges.iter().map(Vec::len).max().unwrap_or(0)
    }

    pub fn bindings(&self) -> OwnedAdvanceBindings<'_> {
        OwnedAdvanceBindings {
            recurrent: &self.recurrent,
            previous_bank: self.previous_bank,
            previous_tape: 0,
            following_bank: self.following.index(),
            stop: self.committed,
            history: &self.history,
            destinations: &self.destinations,
        }
    }

    /// Join the predecessor's committed state: an ordinary advance over the
    /// rows and bank this successor reserved. `state` must be exactly the
    /// state the predecessor published (position, version and history).
    pub fn attach(self, state: SequenceState) -> Result<OwnedStateAdvance, (SequenceState, Error)> {
        if !state.belongs_to(&self.store)
            || state.position() != self.position
            || state.bank_index() != self.previous_bank
            || state.tape_rows() != 0
            || state.history_starts() != self.starts
            || state.domain_ranges() != self.ranges
        {
            return Err((
                state,
                Error::from("successor advance does not follow the committed state"),
            ));
        }
        // Its step runs: what it admitted stays admitted.
        self.admission.kept.set(true);
        let Self {
            count,
            committed,
            claims,
            following,
            history,
            recurrent,
            destinations,
            ..
        } = self;
        Ok(OwnedStateAdvance {
            state,
            count,
            committed,
            acceptable: committed,
            claims,
            following,
            history,
            recurrent,
            destinations,
        })
    }
}

/// A chunk of a prompt group queued behind the chunk before it in one chain
/// ([`OwnedStateAdvance::chained`]). Its rows are its own, as a successor's;
/// its recurrent state alternates between two banks. Behind the group's
/// first chunk it writes a fresh bank, as the first chunk's source is
/// accepted state; behind a chained chunk it writes the bank that chunk
/// read, at each layer after that chunk read it (each chunk reads what the
/// chunk before it wrote there). So the state a chunk publishes is
/// overwritten two chunks later, before its own flight returns: no chunk
/// joins an accepted state of its own, and the chain commits whole
/// ([`OwnedChainAdvance`]).
pub struct OwnedChainedAdvance {
    advance: OwnedSuccessorAdvance,
    /// The bank it reads, which the chunk after it writes.
    reads: BankHandle,
}

impl OwnedChainedAdvance {
    /// The chunk chained behind this one: `count` rows writing the bank
    /// this one reads.
    pub fn chained(&self, count: usize) -> Result<OwnedChainedAdvance, Error> {
        chain_one(self.link(count))
    }

    fn link(&self, count: usize) -> Link<'_> {
        Link {
            predecessor: self.advance.predecessor(),
            count,
            reads: self.advance.following.clone(),
            writes: Some(self.reads.clone()),
        }
    }
}

/// One chunk to chain: `count` rows following `predecessor`, whose
/// published bank is `reads`, writing `writes` or a fresh bank.
struct Link<'a> {
    predecessor: Predecessor<'a>,
    count: usize,
    reads: BankHandle,
    writes: Option<BankHandle>,
}

/// The chunks of one queued step, one per link, their demands admitted
/// together as [`follow`] admits a step's successors.
fn chain(links: Vec<Link<'_>>) -> Result<Vec<OwnedChainedAdvance>, Error> {
    let (slots, reads): (Vec<_>, Vec<_>) = links
        .into_iter()
        .map(|link| ((link.predecessor, link.count, 0, link.writes), link.reads))
        .unzip();
    Ok(follow(slots)?
        .into_iter()
        .zip(reads)
        .map(|(advance, reads)| OwnedChainedAdvance { advance, reads })
        .collect())
}

/// The one chunk of a step of one slot.
fn chain_one(link: Link<'_>) -> Result<OwnedChainedAdvance, Error> {
    Ok(chain(vec![link])?
        .pop()
        .expect("one chunk follows each link"))
}

/// A prompt group's chain, its chunks joined in order as each finishes: the
/// group's first advance and the chunks chained behind it. It publishes
/// nothing until it commits whole: the rows of every chunk and the last
/// chunk's bank; the other bank is released then.
pub struct OwnedChainAdvance {
    first: OwnedStateAdvance,
    chunks: Vec<OwnedChainedAdvance>,
}

impl OwnedChainAdvance {
    /// The chain opened by `first`, whose rows all commit.
    pub fn begin(first: OwnedStateAdvance) -> Result<Self, (OwnedStateAdvance, Error)> {
        if first.committed != first.count {
            return Err((first, Error::from("a chain opens on an advance whose rows all commit")));
        }
        Ok(Self {
            first,
            chunks: Vec::new(),
        })
    }

    /// The position after the last joined chunk.
    pub fn end(&self) -> usize {
        self.chunks
            .last()
            .map_or(self.first.state.position() + self.first.count, |chunk| {
                chunk.advance.position + chunk.advance.count
            })
    }

    /// Join the chunk chained behind the last one joined; any other is
    /// returned.
    pub fn join(
        &mut self,
        chunk: OwnedChainedAdvance,
    ) -> Result<(), (OwnedChainedAdvance, Error)> {
        let bank = self
            .chunks
            .last()
            .map_or(&self.first.following, |last| &last.advance.following);
        if !Rc::ptr_eq(&chunk.advance.store, &self.first.state.snapshot.store)
            || chunk.advance.position != self.end()
            || chunk.advance.previous_bank != bank.index()
        {
            return Err((chunk, Error::from("a joined chunk does not follow its chain")));
        }
        self.chunks.push(chunk);
        Ok(())
    }

    /// Publish every chunk's rows, the recurrent state the last one wrote.
    pub fn commit(self) -> SequenceState {
        let Self { first, chunks } = self;
        let OwnedStateAdvance {
            mut state,
            count,
            mut claims,
            mut following,
            ..
        } = first;
        let store = state.snapshot.store.clone();
        keep_accepted(&store, &mut claims, state.snapshot.position, count);
        let mut rows = count;
        for chunk in chunks {
            // Its step ran: what it admitted stays admitted.
            chunk.advance.admission.kept.set(true);
            let OwnedSuccessorAdvance {
                position,
                count,
                claims: mut chunk_claims,
                following: written,
                ..
            } = chunk.advance;
            keep_accepted(&store, &mut chunk_claims, position, count);
            for (history, chunk_claims) in claims.iter_mut().zip(chunk_claims) {
                history.append(chunk_claims);
            }
            rows += count;
            following = written;
        }
        install_commit(&mut state, claims, &mut following, 0, rows);
        state
    }

    /// Recover the first advance's source state; every chunk's rows and
    /// banks are released.
    pub fn abort(self) -> SequenceState {
        let Self { first, chunks } = self;
        drop(chunks);
        first.abort()
    }
}

/// The tentative rows of one launch slot: an advance of accepted state, a
/// successor of an advance still in flight, or a chunk chained behind one.
pub enum TentativeAdvance {
    Accepted(OwnedStateAdvance),
    Successor(OwnedSuccessorAdvance),
    Chained(OwnedChainedAdvance),
}

impl TentativeAdvance {
    pub fn position(&self) -> usize {
        match self {
            Self::Accepted(advance) => advance.position(),
            Self::Successor(advance) => advance.position(),
            Self::Chained(chunk) => chunk.advance.position(),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Accepted(advance) => advance.rows(),
            Self::Successor(advance) => advance.rows(),
            Self::Chained(chunk) => chunk.advance.rows(),
        }
    }


    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        match self {
            Self::Accepted(advance) => advance.belongs_to(store),
            Self::Successor(advance) => advance.belongs_to(store),
            Self::Chained(chunk) => chunk.advance.belongs_to(store),
        }
    }

    pub fn history_ranges(&self, domain: HistoryDomainId) -> Vec<(usize, usize)> {
        match self {
            Self::Accepted(advance) => advance.history_ranges(domain),
            Self::Successor(advance) => advance.history_ranges(domain),
            Self::Chained(chunk) => chunk.advance.history_ranges(domain),
        }
    }

    pub fn domain_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        match self {
            Self::Accepted(advance) => advance.domain_ranges(),
            Self::Successor(advance) => advance.domain_ranges(),
            Self::Chained(chunk) => chunk.advance.domain_ranges(),
        }
    }

    pub fn history_start(&self, domain: HistoryDomainId) -> usize {
        match self {
            Self::Accepted(advance) => advance.history_start(domain),
            Self::Successor(advance) => advance.history_start(domain),
            Self::Chained(chunk) => chunk.advance.history_start(domain),
        }
    }

    pub fn visible_ranges(&self, domain: HistoryDomainId, from: usize) -> Vec<(usize, usize)> {
        match self {
            Self::Accepted(advance) => advance.visible_ranges(domain, from),
            Self::Successor(advance) => advance.visible_ranges(domain, from),
            Self::Chained(chunk) => chunk.advance.visible_ranges(domain, from),
        }
    }

    /// The most spans any domain's history has.
    pub fn span_count(&self) -> usize {
        match self {
            Self::Accepted(advance) => advance.span_count(),
            Self::Successor(advance) => advance.span_count(),
            Self::Chained(chunk) => chunk.advance.span_count(),
        }
    }

    pub fn bindings(&self) -> OwnedAdvanceBindings<'_> {
        match self {
            Self::Accepted(advance) => advance.bindings(),
            Self::Successor(advance) => advance.bindings(),
            Self::Chained(chunk) => chunk.advance.bindings(),
        }
    }

    pub fn successor(&self, count: usize) -> Result<OwnedSuccessorAdvance, Error> {
        follow_one(self.predecessor()?, count, 0, None)
    }

    /// The successors of one queued step: per slot, `committed` committed
    /// rows then `front` front rows following its advance's committed rows,
    /// admitted and placed as the step would be provisioned and placed
    /// alone once those commit.
    pub fn successors<'a>(
        slots: impl IntoIterator<Item = (&'a TentativeAdvance, usize, usize)>,
    ) -> Result<Vec<OwnedSuccessorAdvance>, Error> {
        follow(
            slots
                .into_iter()
                .map(|(advance, committed, front)| Ok((advance.predecessor()?, committed, front, None)))
                .collect::<Result<_, Error>>()?,
        )
    }

    /// The chunks of one queued step chained behind its slots: per slot,
    /// `count` rows chained behind its advance (the first chunk of a chain
    /// or a chained one), admitted together and placed as the step would be
    /// provisioned and placed alone, as [`Self::successors`] are. A
    /// successor opens no chain: a chain opens where a step launches.
    pub fn chained<'a>(
        slots: impl IntoIterator<Item = (&'a TentativeAdvance, usize)>,
    ) -> Result<Vec<OwnedChainedAdvance>, Error> {
        chain(
            slots
                .into_iter()
                .map(|(advance, count)| match advance {
                    Self::Accepted(advance) => advance.link(count),
                    Self::Chained(advance) => Ok(advance.link(count)),
                    Self::Successor(_) => Err(Error::from("a chain opens where a step launches")),
                })
                .collect::<Result<_, Error>>()?,
        )
    }

    /// What a successor follows. A chained chunk's published state is
    /// overwritten while its chain runs: only its chain follows it.
    fn predecessor(&self) -> Result<Predecessor<'_>, Error> {
        match self {
            Self::Accepted(advance) => advance.predecessor(),
            Self::Successor(advance) => Ok(advance.predecessor()),
            Self::Chained(_) => Err(Error::from("only its chain follows a chained chunk")),
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
        BankCapacity, CodecSpec, ComponentDescriptor, ComponentSpec, HistoryDomainLayout,
        HistoryDomainPlan, LayerRef, OwnDevice, StoreBindings,
    };
    use seismic::DType;

    const TOKEN: HistoryDomainId = HistoryDomainId(0);

    /// A store of one Token domain, as Qwen's stores are.
    fn token_store(
        device: &OwnDevice,
        context: usize,
        rows: usize,
        components: Vec<ComponentDescriptor>,
        specs: Vec<ComponentSpec>,
        banks: BankCapacity,
    ) -> Result<StoreBindings, Error> {
        device
            .store(
                context,
                context,
                vec![HistoryDomainPlan {
                    layout: HistoryDomainLayout::Token { components },
                    logical_rows: rows,
                }],
                specs,
                banks,
            )
            .map(|(_, bindings)| bindings)
    }

    #[test]
    fn abort_recovers_source_and_releases_tentative_capacity() {
        let device = OwnDevice::cpu();
        let store = token_store(
            &device,
            4,
            4,
            vec![ComponentDescriptor::new(
                LayerRef::Target(0),
                CodecSpec::dense(DType::F32, 1, 1),
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
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        assert_eq!(advance.bindings().destinations[0].len(), 2);
        assert_eq!(store.occupied_rows(TOKEN), 2);
        assert_eq!(store.referenced_bytes(), 2 * store.history_row_bytes(TOKEN));
        let state = advance.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(store.occupied_rows(TOKEN), 0);

        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!(state.position(), 2);
        assert_eq!(store.occupied_rows(TOKEN), 2);
        let retained = state.checkpoint();
        let advance = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        // The checkpoint shares the committed rows; the advance adds one.
        assert_eq!(store.referenced_bytes(), 3 * store.history_row_bytes(TOKEN));
        let state = advance.abort();
        assert_eq!(state.position(), 2);
        assert_eq!(store.referenced_bytes(), 2 * store.history_row_bytes(TOKEN));
        drop(retained);
    }

    #[test]
    fn successors_follow_in_flight_advances_and_attach_to_their_commits() {
        let device = OwnDevice::cpu();
        let store = token_store(
            &device,
            8,
            8,
            vec![ComponentDescriptor::new(
                LayerRef::Target(0),
                CodecSpec::dense(DType::F32, 1, 1),
                1,
            )
            .unwrap()],
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
        assert_eq!(
            second.bindings().previous_bank,
            first.bindings().following_bank
        );
        assert_eq!(
            third.bindings().previous_bank,
            second.bindings().following_bank
        );
        assert_eq!(
            second.history_ranges(TOKEN),
            first.bindings().destinations[0]
                .iter()
                .map(|&row| (row, 1))
                .fold(Vec::new(), |mut ranges, range| {
                    append_ranges(&mut ranges, [range], usize::MAX);
                    ranges
                },)
        );
        // Rows follow their predecessors physically: one run.
        assert_eq!(
            second.bindings().destinations[0],
            [first.bindings().destinations[0][1] + 1]
        );
        assert_eq!(
            third.bindings().destinations[0],
            [second.bindings().destinations[0][0] + 1]
        );
        assert_eq!(store.occupied_rows(TOKEN), 4);
        assert_eq!(store.referenced_bytes(), 4 * store.history_row_bytes(TOKEN));

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
        assert_eq!(state.history_ranges(TOKEN).len(), 1);
        // The dropped third successor released its row and bank.
        assert_eq!(store.occupied_rows(TOKEN), 3);
        let next = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let orphan = next.successor(1).unwrap();
        drop(orphan);
        assert_eq!(store.occupied_rows(TOKEN), 4);
        let state = next.abort();
        assert_eq!(state.position(), 3);
        assert_eq!(store.occupied_rows(TOKEN), 3);
    }

    #[test]
    fn codec_conversion_reserves_and_publishes_only_after_completion() {
        let device = OwnDevice::cpu();
        let source_store = token_store(
            &device,
            4,
            4,
            vec![ComponentDescriptor::new(
                LayerRef::Target(0),
                CodecSpec::dense(DType::F16, 32, 32),
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
        let destination_store = token_store(
            &device,
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
        let conversion = OwnedCodecAdvance::begin(source, destination).unwrap_or_else(|(_, _, error)| panic!("the conversion begins: {error}"));
        assert_eq!(conversion.rows(), 2);
        assert_eq!(conversion.source_codec(), KvCodec::Dense);
        assert_eq!(conversion.destination_codec(), KvCodec::AffineK8V4);
        assert_eq!(conversion.conversions().len(), 2);
        // Codes and coefficient planes per vector kind.
        assert_eq!(conversion.bindings().destination_history.len(), 4);
        assert_eq!(destination_store.occupied_rows(TOKEN), 2);
        assert_eq!(
            source_store.referenced_bytes(),
            2 * source_store.history_row_bytes(TOKEN)
        );
        assert_eq!(
            destination_store.referenced_bytes(),
            2 * destination_store.history_row_bytes(TOKEN)
        );
        let (source, destination) = conversion.abort();
        assert_eq!(source.position(), 2);
        assert_eq!(destination.position(), 0);
        assert_eq!(destination_store.occupied_rows(TOKEN), 0);
        let conversion = OwnedCodecAdvance::begin(source, destination).unwrap_or_else(|(_, _, error)| panic!("the conversion begins: {error}"));
        let destination = conversion.commit();
        assert_eq!(destination.position(), 2);
        assert_eq!(
            destination
                .history_ranges(TOKEN)
                .iter()
                .map(|(_, rows)| rows)
                .sum::<usize>(),
            2
        );
        assert_eq!(destination_store.occupied_rows(TOKEN), 2);
    }

    fn recurrent_store(context: usize, in_flight: usize) -> StoreBindings {
        token_store(
            &OwnDevice::cpu(),
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
        .unwrap()
    }

    /// A verification publishes its recurrent state after its committed rows
    /// and records the rest on the tape: any accepted prefix commits as
    /// version (successor bank, accepted - committed), read by the next
    /// advance, with no second pass.
    #[test]
    fn speculative_prefixes_commit_as_tape_versions() {
        let store = recurrent_store(8, 2);
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin_rows(state, 1, 3, 0)
            .ok()
            .unwrap();
        let bank_bytes = store.allocation_trace().unwrap().recurrent_bank_bytes;
        assert_eq!(
            store.referenced_bytes(),
            4 * store.history_row_bytes(TOKEN) + bank_bytes
        );
        assert_eq!(store.occupied_bytes() - store.referenced_bytes(), bank_bytes, "the model's seed bank");
        let bindings = advance.bindings();
        assert_eq!((bindings.stop, bindings.previous_tape), (1, 0));
        let following = bindings.following_bank;
        let OwnedAdvanceResolution::Committed(state) = advance.commit(3).ok().unwrap() else {
            panic!("an accepted prefix past the committed rows commits");
        };
        assert_eq!(
            (state.position(), state.bank_index(), state.tape_rows()),
            (3, following, 2)
        );
        assert_eq!(store.occupied_rows(TOKEN), 3);

        // The next advance reads the version; a checkpoint and its forks keep it.
        let checkpoint = state.checkpoint();
        assert_eq!(checkpoint.tape_rows(), 2);
        assert_eq!(checkpoint.fork().tape_rows(), 2);
        let advance = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        // The checkpoint shares the advance's source rows and bank, counted once.
        assert_eq!(
            store.referenced_bytes(),
            4 * store.history_row_bytes(TOKEN) + 2 * bank_bytes
        );
        let bindings = advance.bindings();
        assert_eq!(
            (
                bindings.previous_bank,
                bindings.previous_tape,
                bindings.stop
            ),
            (following, 2, 1)
        );
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("a plain advance commits");
        };
        assert_eq!((state.position(), state.tape_rows()), (4, 0));

        // A prefix ending before the published state has no version.
        let advance = OwnedStateAdvance::begin_rows(state, 2, 1, 0)
            .ok()
            .unwrap();
        let (state, _) = advance.commit(1).err().unwrap();
        assert_eq!((state.position(), state.tape_rows()), (4, 0));
        assert_eq!(store.occupied_rows(TOKEN), 4);
        let advance = OwnedStateAdvance::begin_rows(state, 2, 1, 0)
            .ok()
            .unwrap();
        let OwnedAdvanceResolution::Aborted(state) = advance.commit(0).ok().unwrap() else {
            panic!("an empty prefix aborts");
        };
        assert_eq!(state.position(), 4);
        // A paired pass commits its exact row, never its front row.
        let advance = OwnedStateAdvance::begin_rows(state, 1, 0, 1).ok().unwrap();
        let (state, _) = advance.commit(2).err().unwrap();
        assert_eq!((state.position(), state.tape_rows()), (4, 0));
        // Every advance commits a row: a front row alone, or acceptable
        // rows with no committed row before them, has none.
        let (state, _) = OwnedStateAdvance::begin_rows(state, 0, 0, 1).err().unwrap();
        let (state, _) = OwnedStateAdvance::begin_rows(state, 0, 1, 1).err().unwrap();
        assert!(OwnedStateAdvance::begin_rows(state, 0, 0, 0).is_err());
    }

    /// The successor of a paired pass, queued while the pass runs, follows
    /// what the pass publishes once its exact row commits: it reads the
    /// successor bank at tape 0, and its exact row takes over the front
    /// row's row, written anew at that position, so its rows run on in
    /// place as the next pass's would once the pass commits. It attaches
    /// to that commit, itself a paired pass, publishing after its exact
    /// row and committing it alone. An advance whose acceptable rows its
    /// reconciliation decides has no successor.
    #[test]
    fn a_paired_pass_is_followed_from_its_exact_row() {
        let store = recurrent_store(8, 3);
        let first = OwnedStateAdvance::begin_rows(store.create().unwrap(), 1, 0, 1).ok().unwrap();
        let rows = first.bindings().destinations[0].clone();
        let pass = TentativeAdvance::Accepted(first);
        let [next]: [OwnedSuccessorAdvance; 1] =
            TentativeAdvance::successors([(&pass, 1, 1)]).unwrap().try_into().ok().unwrap();
        let TentativeAdvance::Accepted(first) = pass else { unreachable!() };
        assert_eq!(next.position(), 1);
        let bindings = next.bindings();
        assert_eq!(
            (bindings.previous_bank, bindings.previous_tape, bindings.stop),
            (first.bindings().following_bank, 0, 1)
        );
        assert_eq!(bindings.destinations[0], [rows[1], rows[1] + 1], "the front row's row, then the next in place");
        assert_eq!(next.history_ranges(TOKEN), [(rows[0], 1)]);
        assert_eq!(store.occupied_rows(TOKEN), 3);

        let OwnedAdvanceResolution::Committed(state) = first.commit(1).ok().unwrap() else {
            panic!("the exact row commits");
        };
        // The front row's row stays the successor's.
        assert_eq!(store.occupied_rows(TOKEN), 3);
        let next = next.attach(state).ok().unwrap();
        assert_eq!(next.bindings().stop, 1);
        let OwnedAdvanceResolution::Committed(state) = next.commit(1).ok().unwrap() else {
            panic!("the exact row commits");
        };
        assert_eq!((state.position(), state.tape_rows(), store.occupied_rows(TOKEN)), (2, 0, 2));

        let verification = OwnedStateAdvance::begin_rows(state, 1, 1, 0).ok().unwrap();
        assert!(verification.successor(1).is_err());
    }

    /// A successor shallower than its predecessor's front (depth 0 queued
    /// behind depth 2) takes over only the front rows' rows its own rows
    /// reach and drops its references on the rest, which the running pass
    /// keeps: getting shallower costs nothing, and they free as the pass
    /// commits.
    #[test]
    fn a_shallower_successor_takes_over_only_the_front_rows_it_reaches() {
        let store = recurrent_store(8, 3);
        let first = OwnedStateAdvance::begin_rows(store.create().unwrap(), 1, 0, 2).ok().unwrap();
        let rows = first.bindings().destinations[0].clone();
        let pass = TentativeAdvance::Accepted(first);
        let [next]: [OwnedSuccessorAdvance; 1] =
            TentativeAdvance::successors([(&pass, 1, 0)]).unwrap().try_into().ok().unwrap();
        let TentativeAdvance::Accepted(first) = pass else { unreachable!() };
        assert_eq!(next.position(), 1);
        assert_eq!(next.bindings().destinations[0], [rows[1]], "the first front row's row, written anew");
        assert_eq!(store.occupied_rows(TOKEN), 3, "the pass keeps its every row; the successor reserves none");

        let OwnedAdvanceResolution::Committed(state) = first.commit(1).ok().unwrap() else {
            panic!("the exact row commits");
        };
        assert_eq!(store.occupied_rows(TOKEN), 2, "the dropped front row's row frees with the pass");
        let next = next.attach(state).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = next.commit(1).ok().unwrap() else {
            panic!("the exact row commits");
        };
        assert_eq!((state.position(), store.occupied_rows(TOKEN)), (2, 2));
    }

    /// A chain of four chunks runs on two successor banks, alternating
    /// behind its first chunk's fresh one and never writing its source;
    /// it publishes nothing until it commits whole, the last chunk's bank
    /// then, or returns its source when aborted.
    #[test]
    fn a_chain_alternates_two_banks_and_commits_whole() {
        let mut store = recurrent_store(8, 2);
        store.provision(&[], 2).unwrap();
        let free = store.available_banks();
        let source = store.create().unwrap();
        let source_bank = source.bank_index();
        let first = OwnedStateAdvance::begin(source, 1).ok().unwrap();
        let second = first.chained(1).unwrap();
        let third = second.chained(1).unwrap();
        let fourth = third.chained(1).unwrap();
        let banks = |chunk: &OwnedChainedAdvance| {
            let bindings = chunk.advance.bindings();
            (bindings.previous_bank, bindings.following_bank)
        };
        let written = first.bindings().following_bank;
        let (read, other) = banks(&second);
        assert_eq!(read, written);
        assert!(other != written && other != source_bank && written != source_bank);
        assert_eq!(banks(&third), (other, written));
        assert_eq!(banks(&fourth), (written, other));
        // Two successor banks in flight, whatever the chain's length.
        assert_eq!(store.available_banks(), free - 2);
        assert_eq!(store.occupied_rows(TOKEN), 4);
        // Only its chain follows a chunk.
        let fourth = TentativeAdvance::Chained(fourth);
        assert!(fourth.successor(1).is_err());
        let TentativeAdvance::Chained(fourth) = fourth else {
            unreachable!()
        };

        let mut chain = OwnedChainAdvance::begin(first).ok().unwrap();
        let (third, _) = chain.join(third).err().expect("a chunk out of order");
        chain.join(second).ok().unwrap();
        chain.join(third).ok().unwrap();
        chain.join(fourth).ok().unwrap();
        assert_eq!(chain.end(), 4);
        let state = chain.commit();
        assert_eq!((state.position(), state.bank_index(), state.tape_rows()), (4, other, 0));
        assert_eq!(store.occupied_rows(TOKEN), 4);
        assert_eq!(state.history_ranges(TOKEN).len(), 1);
        // The other bank was released.
        assert_eq!(store.available_banks(), free - 1);

        store.provision(&state.demands(3), 2).unwrap();
        let first = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let second = first.chained(1).unwrap();
        let mut chain = OwnedChainAdvance::begin(first).ok().unwrap();
        chain.join(second).ok().unwrap();
        let state = chain.abort();
        assert_eq!((state.position(), state.bank_index()), (4, other));
        assert_eq!(store.occupied_rows(TOKEN), 4);
        let speculative = OwnedStateAdvance::begin_rows(state, 1, 1, 0).ok().unwrap();
        assert!(speculative.chained(1).is_err(), "a chain carries no speculation");
        assert!(OwnedChainAdvance::begin(speculative).is_err());
    }

    /// A verification publishes whichever prefix of its rows is accepted
    /// as a version of its one successor bank's tape: every row accepted,
    /// both acceptable rows; a prefix, only those. Nothing follows it
    /// before then: what it commits is decided only by its reconciliation,
    /// so the design refuses it a successor (`OwnedStateAdvance::predecessor`;
    /// no decode queues behind a verify pass), and the next advance begins
    /// from the version it published.
    #[test]
    fn successors_follow_the_tape_version_of_a_speculative_advance() {
        let mut store = recurrent_store(8, 3);
        let first = OwnedStateAdvance::begin_rows(store.create().unwrap(), 1, 2, 0)
            .ok()
            .unwrap();
        assert!(first.successor(1).is_err(), "no successor follows an undecided verification");
        let OwnedAdvanceResolution::Committed(state) = first.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!(state.tape_rows(), 2);
        let second = OwnedStateAdvance::begin(state, 1).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = second.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!((state.position(), state.tape_rows()), (4, 0));

        // A partially accepted verification publishes another version.
        store.provision(&[], 1).unwrap();
        let first = OwnedStateAdvance::begin_rows(state, 1, 2, 0)
            .ok()
            .unwrap();
        assert!(first.successor(1).is_err(), "no successor follows an undecided verification");
        let OwnedAdvanceResolution::Committed(state) = first.commit(2).ok().unwrap() else {
            panic!("an accepted prefix commits");
        };
        assert_eq!((state.position(), state.tape_rows()), (6, 1));
    }

    #[test]
    fn follow_rejects_out_of_context_capacity_successor_before_admitting() {
        let store = recurrent_store(8, 3);
        let first = OwnedStateAdvance::begin(store.create().unwrap(), 2)
            .ok()
            .unwrap();
        let occupied = store.occupied_rows(TOKEN);
        let free_banks = store.available_banks();
        assert!(matches!(
            first.successor(0),
            Err(Error::Request(ref msg)) if msg.contains("context capacity")
        ));
        assert!(matches!(
            first.successor(7),
            Err(Error::Request(ref msg)) if msg.contains("context capacity")
        ));
        assert!(matches!(
            first.successor(usize::MAX),
            Err(Error::Request(ref msg)) if msg.contains("context capacity")
        ));
        assert_eq!(store.occupied_rows(TOKEN), occupied);
        assert_eq!(store.available_banks(), free_banks);
    }
}
