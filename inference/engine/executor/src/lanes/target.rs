//! The sole join of target row semantics, owned state, conditioning, and
//! planned device leases.

use crate::{
    ConditioningRef, FeatureRef, FeatureSpan, GraphOutputTensor, InvariantError,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, PoolClass, ResourceDomainId,
    TargetGraphOutputLease, TargetGraphWorkspaceLease,
};
use crate::operation::{FrontKind, Guess, LaunchWork, Operation, RequestId};
use magnitude_batching::ValidatedTargetBatch;
use magnitude_state::{StateStore, TentativeAdvance};
use std::rc::Rc;

/// Where a launch's input tokens come from.
#[derive(Clone)]
pub enum TargetTokens {
    /// The batch's host tokens, written into the step's upload region.
    Host,
    /// Tokens the device selected for it (`[rows, 2]` token, status rows in
    /// batch row order), read on the device: the step continues the step
    /// before, each of its rows a front pass's exact token or the guess of
    /// the chain made behind that step (`guess_chain`'s `next_selected`);
    /// and of a load that fronts, its front rows' guesses (`[rows, 2]`
    /// guess, kind rows, `guess_chain`'s `next_guesses`), whose kinds the
    /// step's words carry (`routed_step`).
    Selected {
        selected: GraphOutputTensor,
        guesses: Option<GraphOutputTensor>,
    },
}

/// The host's guesses a launch slot's front rows run on, by depth: of a
/// launch of host tokens, the guess each of its operation's front rows
/// carries ([`FrontKind::Host`]), none of an idle row ([`FrontKind::Idle`]),
/// whose kinds the step's words read (`routed_step`); empty of a slot of no
/// front row, and of every slot of selected tokens, whose guesses the
/// device reads ([`TargetTokens::Selected`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotGuess {
    front: Vec<Option<Guess>>,
}

impl SlotGuess {
    /// The host's guesses of a launch's slot of `operation`: its front
    /// rows' where `host` builds the launch's tokens (the operation's
    /// [`FrontKind::Host`], each its front row's token, and
    /// [`FrontKind::Idle`], none), none otherwise ([`FrontKind::Device`]).
    /// Any other pairing, or another operation than a forward, is the
    /// caller's broken invariant.
    pub fn of(operation: &Operation, host: bool) -> Self {
        let Operation::Forward { tokens, front: front_rows, .. } = operation else {
            panic!("a target launch's slot runs a forward");
        };
        let first = tokens.len() - front_rows.len();
        let front = front_rows
            .iter()
            .zip(&tokens[first..])
            .filter_map(|(row, &token)| match (row.kind, host) {
                (FrontKind::Device, false) => None,
                (FrontKind::Host(kind), true) => Some(Some(Guess { token, kind })),
                (FrontKind::Idle, true) => Some(None),
                (kind, host) => panic!("a front row's guess of {kind:?} in a launch whose tokens the host builds: {host}"),
            })
            .collect();
        Self { front }
    }

    /// Every slot of a launch of `operations`.
    pub fn launch(operations: &[Operation], host: bool) -> Vec<Self> {
        operations.iter().map(|operation| Self::of(operation, host)).collect()
    }
}

pub struct TargetLaunchInputs {
    batch: ValidatedTargetBatch,
    /// What its rows are (`LaunchWork::of` its requests' kinds).
    work: LaunchWork,
    /// Per slot, the request whose rows it runs.
    requests: Vec<RequestId>,
    tokens: TargetTokens,
    advances: Vec<TentativeAdvance>,
    conditioning: Vec<Option<ConditioningRef>>,
    conditioning_slices: Vec<Vec<ConditioningSlice>>,
    boundary: Option<FeatureRef>,
    /// Per slot, the accepted tokens its sequence holds before the slot's
    /// first row, oldest first, as many as the model's row rules read (none
    /// of a model without one): the host's form of the n-gram token window.
    preceding: Vec<Vec<u64>>,
    /// Per slot, the host's guesses its front rows run on ([`SlotGuess`]).
    guesses: Vec<SlotGuess>,
    graph_workspace: TargetGraphWorkspaceLease,
    /// The launch's place in its chain's output rotation
    /// (`TargetProgram::submit`).
    graph_output: TargetGraphOutputLease,
    readout_workspace: NativeGraphWorkspaceLease,
    readout_output: Option<NativeGraphOutputLease>,
}

/// A retained feature slice placed at a request-local decoder row offset.
/// The target program overlays these values directly into hidden storage.
#[derive(Clone)]
pub struct ConditioningSlice {
    pub source: FeatureSpan,
    pub destination: usize,
}

impl TargetLaunchInputs {
    pub fn new(
        batch: ValidatedTargetBatch,
        work: LaunchWork,
        requests: Vec<RequestId>,
        tokens: TargetTokens,
        advances: Vec<TentativeAdvance>,
        conditioning: Vec<Option<ConditioningRef>>,
        conditioning_slices: Vec<Vec<ConditioningSlice>>,
        operations: &[Operation],
        graph_workspace: TargetGraphWorkspaceLease,
        graph_output: TargetGraphOutputLease,
        readout_workspace: NativeGraphWorkspaceLease,
        readout_output: NativeGraphOutputLease,
    ) -> Self {
        let guesses = SlotGuess::launch(operations, matches!(tokens, TargetTokens::Host));
        Self {
            batch,
            work,
            requests,
            tokens,
            advances,
            conditioning,
            conditioning_slices,
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output: Some(readout_output),
            boundary: None,
            preceding: Vec::new(),
            guesses,
        }
    }

    pub(crate) fn with_boundary(mut self, boundary: Option<FeatureRef>) -> Self {
        self.boundary = boundary;
        self
    }

    /// Per slot, the accepted tokens before its first row
    /// (`TargetLaunchCore::preceding`).
    pub(crate) fn with_preceding(mut self, preceding: Vec<Vec<u64>>) -> Self {
        self.preceding = preceding;
        self
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        ValidatedTargetBatch,
        Vec<TentativeAdvance>,
        Vec<Option<ConditioningRef>>,
        Vec<Vec<ConditioningSlice>>,
    ) {
        (
            self.batch,
            self.advances,
            self.conditioning,
            self.conditioning_slices,
        )
    }

    fn validate(
        &self,
        store: &Rc<StateStore>,
        domain: &ResourceDomainId,
        hidden_width: usize,
        boundary_width: usize,
    ) -> Result<(), InvariantError> {
        let invalid = |detail: String| InvariantError {
            context: "target launch",
            detail,
        };
        if let Some(boundary) = &self.boundary {
            if self.batch.actual_slots() != 1
                || boundary.domain() != domain
                || boundary.allocation().rows() != 1
                || boundary.allocation().width() != boundary_width
                || boundary.allocation().is_graph_backed()
                || boundary.allocation().tensor().is_err()
                || !self.batch.class().demand().computes_features()
            {
                return Err(invalid(
                    "invalid independent boundary feature destination".into(),
                ));
            }
        }
        let slots = self.batch.actual_slots();
        if slots != self.advances.len()
            || slots != self.requests.len()
            || slots != self.conditioning.len()
            || slots != self.conditioning_slices.len()
            || !(self.preceding.is_empty() || self.preceding.len() == slots)
            || slots != self.guesses.len()
        {
            return Err(invalid(
                "slot, advance, and conditioning counts disagree".into(),
            ));
        }
        // Selected tokens continue a decode pass: each slot is one exact
        // row, then its front rows, each row's token the selection of its
        // predecessor's row at its batch row.
        let front = self.batch.upload().front;
        let fronts = |slot: &magnitude_batching::TargetBatchSlot<'_>| front[slot.span()].iter().filter(|&&front| front).count();
        let host = matches!(self.tokens, TargetTokens::Host);
        if self
            .batch
            .slots()
            .any(|slot| front[slot.span()].first() == Some(&true))
        {
            return Err(invalid(
                "a slot's first row cannot be a front row".into(),
            ));
        }
        if self
            .batch
            .slots()
            .zip(&self.guesses)
            .any(|(slot, guess)| guess.front.len() != if host { fronts(&slot) } else { 0 })
        {
            return Err(invalid("front rows' guesses are the host's exactly where it builds the tokens".into()));
        }
        if let TargetTokens::Selected { selected, guesses } = &self.tokens {
            let rows = self.batch.class().rows() as u64;
            let valid_pairs = |tensor: &GraphOutputTensor| {
                tensor.tensor().element() == seismic::Element::i32()
                    && tensor.tensor().extents() == [rows, 2]
            };
            let has_front = self.batch.slots().any(|slot| fronts(&slot) > 0);
            if !valid_pairs(selected)
                || guesses
                    .as_ref()
                    .map_or(has_front, |guesses| !valid_pairs(guesses))
                || self
                    .batch
                    .slots()
                    .any(|slot| slot.rows() - fronts(&slot) != 1)
                || self.conditioning.iter().any(Option::is_some)
            {
                return Err(invalid(
                    "selected tokens differ from the launch's decode passes".into(),
                ));
            }
        }
        if self.graph_workspace.domain() != domain
            || self.graph_output.domain() != domain
            || self.readout_workspace.domain() != domain
            || self
                .readout_output
                .as_ref()
                .is_none_or(|lease| lease.domain() != domain)
        {
            return Err(invalid(
                "workspace or output lease differs from batch domain/class".into(),
            ));
        }
        let mut expected: Vec<[i32; 2]> = Vec::new();
        let mut occupied: Vec<bool> = Vec::new();
        for (index, (((slot, advance), conditioning), slices)) in self
            .batch
            .slots()
            .zip(&self.advances)
            .zip(&self.conditioning)
            .zip(&self.conditioning_slices)
            .enumerate()
        {
            if !advance.belongs_to(store) || slot.rows() != advance.rows() {
                return Err(invalid(format!(
                    "slot {index} differs from its state advance"
                )));
            }
            let binding = advance.bindings();
            if i32::try_from(binding.previous_bank).ok() != Some(slot.bank()) {
                return Err(invalid(format!("slot {index} uses another recurrent bank")));
            }
            if i32::try_from(binding.following_bank).ok() != Some(slot.following_bank()) {
                return Err(invalid(format!(
                    "slot {index} publishes to another recurrent successor bank"
                )));
            }
            if slot.history_domains() != store.history_reads().len() {
                return Err(invalid(format!(
                    "slot {index} history domain count differs from its store"
                )));
            }
            // A Shared source domain's reads: its accepted rows, then rows its
            // source appends in this advance.
            for (read, domain) in store
                .history_reads()
                .shared_sources()
                .iter()
                .copied()
                .enumerate()
                .map(|(position, domain)| (store.history_domains().count() + position, domain))
            {
                let packed = slot
                    .history(read)
                    .ok_or_else(|| invalid(format!("slot {index} lacks a shared history read")))?;
                let appended = crate::domain::AppendedRuns::of(store, advance, domain);
                for (row, visible) in packed.visible.iter().enumerate() {
                    let (accepted, _) =
                        crate::domain::accepted_history(store, advance, domain, row)
                            .map_err(invalid)?;
                    if visible.len() < accepted.len()
                        || visible[..accepted.len()] != accepted
                        || !visible[accepted.len()..]
                            .iter()
                            .filter(|range| **range != [0, 0])
                            .all(|&[start, end]| appended.covers(start, end))
                    {
                        return Err(invalid(format!(
                            "slot {index} shared visibility differs from accepted state"
                        )));
                    }
                }
            }
            for domain in store.history_domains() {
                let packed = slot
                    .history(domain.0)
                    .ok_or_else(|| invalid(format!("slot {index} lacks a history domain")))?;
                // An indexer's rows read as `indexer_history` forms them.
                if store.is_produced_domain(domain) {
                    let appended = crate::domain::AppendedRuns::of(store, advance, domain);
                    for (row, (destination, visible)) in
                        packed.destinations.iter().zip(packed.visible).enumerate()
                    {
                        let expected = crate::domain::indexer_history(
                            store, advance, &appended, domain, row, 0,
                        )
                        .map_err(invalid)?;
                        if expected.destination != *destination
                            || visible.iter().filter(|range| **range != [0, 0]).ne(expected
                                .visible
                                .iter()
                                .filter(|range| **range != [0, 0]))
                        {
                            return Err(invalid(format!(
                                "slot {index} row {row} indexer history differs from its state"
                            )));
                        }
                    }
                    continue;
                }
                let destinations = &binding.destinations[domain.0];
                if !destinations.is_empty() && destinations.len() != packed.destinations.len() {
                    return Err(invalid(format!("slot {index} destination count differs")));
                }
                let kind = store.history_domain_kind(domain);
                let appended = crate::domain::cached_causal_domain(store, domain)
                    .then(|| crate::domain::AppendedRuns::of(store, advance, domain));
                for (row, (destination, visible)) in
                    packed.destinations.iter().zip(packed.visible).enumerate()
                {
                    let expected_destination = destinations
                        .get(row)
                        .map(|destination| i32::try_from(*destination).ok())
                        .unwrap_or(Some(-1));
                    if expected_destination != Some(*destination) {
                        return Err(invalid(format!(
                            "slot {index} row {row} destination differs"
                        )));
                    }
                    // Each row reads its domain's accepted rows from its
                    // query's window start.
                    let from = kind.visible_from(advance.position() + row);
                    expected.clear();
                    for (start, count) in advance.visible_ranges(domain, from) {
                        let range = start
                            .checked_add(count)
                            .and_then(|end| {
                                Some([i32::try_from(start).ok()?, i32::try_from(end).ok()?])
                            })
                            .ok_or_else(|| {
                                invalid(format!(
                                    "slot {index} history range exceeds packed coordinates"
                                ))
                            })?;
                        expected.push(range);
                    }
                    let supplied = visible.iter().copied().filter(|range| *range != [0, 0]);
                    // Media rows retain their declared dense fresh block;
                    // causal compressed rows may additionally see exactly
                    // their own earlier destinations, never a future row.
                    if supplied.clone().ne(expected.iter().copied()) {
                        let causal_match = if packed.fresh[row][1] - packed.fresh[row][0] == 1 {
                            if let Some(appended) = &appended {
                                // The row's first fresh row in the slot, as
                                // `accepted_history` reports it.
                                let fresh_start = from.saturating_sub(advance.position());
                                expected.extend(
                                    appended.causal_prior(fresh_start, row).map_err(invalid)?,
                                );
                                supplied.eq(expected.iter().copied())
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if !causal_match {
                            return Err(invalid(format!(
                                "slot {index} visibility differs from accepted state"
                            )));
                        }
                    }
                }
            }
            if let Some(lease) = conditioning {
                let allocation = lease.allocation();
                if lease.domain() != domain
                    || allocation.rows() != slot.rows()
                    || allocation.width() != hidden_width
                    || allocation.tensor().is_err()
                {
                    return Err(invalid(format!(
                        "slot {index} conditioning differs from its physical rows"
                    )));
                }
            }
            if !slices.is_empty() {
                occupied.clear();
                occupied.resize(slot.rows(), false);
                for slice in slices {
                    let source = slice.source.features.allocation();
                    let end = slice
                        .destination
                        .checked_add(slice.source.count)
                        .ok_or_else(|| {
                            invalid(format!("slot {index} conditioning destination overflows"))
                        })?;
                    let source_end = slice
                        .source
                        .start
                        .checked_add(slice.source.count)
                        .ok_or_else(|| {
                            invalid(format!("slot {index} conditioning source overflows"))
                        })?;
                    if slice.source.features.domain() != domain
                        || source.width() != hidden_width
                        || source_end > source.rows()
                        || end > slot.rows()
                        || slice.source.count == 0
                        || source.tensor().is_err()
                        || occupied[slice.destination..end].iter().any(|set| *set)
                    {
                        return Err(invalid(format!(
                            "slot {index} conditioning slice is invalid"
                        )));
                    }
                    occupied[slice.destination..end].fill(true);
                }
            }
        }
        Ok(())
    }
}

/// Opaque checked launch. Construction returns unconsumed inputs if its
/// ownership or batch invariants fail.
pub struct ValidatedTargetLaunch {
    core: TargetLaunchCore,
    graph_workspace: TargetGraphWorkspaceLease,
    graph_output: TargetGraphOutputLease,
    readout_workspace: NativeGraphWorkspaceLease,
    readout_output: Option<NativeGraphOutputLease>,
}

impl ValidatedTargetLaunch {
    pub fn new(
        inputs: TargetLaunchInputs,
        store: &Rc<StateStore>,
        domain: &ResourceDomainId,
        hidden_width: usize,
        boundary_width: usize,
    ) -> Result<Self, (TargetLaunchInputs, InvariantError)> {
        if let Err(error) = inputs.validate(store, domain, hidden_width, boundary_width) {
            return Err((inputs, error));
        }
        let TargetLaunchInputs {
            batch,
            work,
            requests,
            tokens,
            advances,
            conditioning,
            conditioning_slices,
            boundary,
            preceding,
            guesses,
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output,
        } = inputs;
        Ok(Self {
            core: TargetLaunchCore {
                batch,
                work,
                requests,
                tokens,
                advances,
                conditioning,
                conditioning_slices,
                boundary,
                preceding,
                guesses,
                store: store.clone(),
                domain: domain.clone(),
            },
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output,
        })
    }

    pub fn class(&self) -> PoolClass {
        PoolClass::Target(self.core.batch.class())
    }

    pub(crate) fn batch(&self) -> &ValidatedTargetBatch {
        self.core.batch()
    }

    pub(crate) fn work(&self) -> LaunchWork {
        self.core.work
    }

    /// Per slot, the request whose rows it runs.
    pub(crate) fn requests(&self) -> &[RequestId] {
        &self.core.requests
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }

    /// The launch's state advances, one per slot: a step chained behind it
    /// follows them.
    pub(crate) fn advances(&self) -> &[TentativeAdvance] {
        self.core.advances()
    }

    pub(crate) fn submission_parts_mut(
        &mut self,
    ) -> (
        &TargetLaunchCore,
        &mut TargetGraphWorkspaceLease,
        &mut TargetGraphOutputLease,
        &mut NativeGraphWorkspaceLease,
        &mut Option<NativeGraphOutputLease>,
    ) {
        (
            &self.core,
            &mut self.graph_workspace,
            &mut self.graph_output,
            &mut self.readout_workspace,
            &mut self.readout_output,
        )
    }

    /// The reconciliation payload and the device leases the submitted work
    /// uses; the leases stay held until the submission is finished. The
    /// launch's output returns to its pool now: every reader of it was
    /// queued by this submission, and the device runs the next launch that
    /// takes it after this one.
    pub(crate) fn into_submission_parts(self) -> (TargetLaunchCore, TargetLaunchWorkspace) {
        (
            self.core,
            TargetLaunchWorkspace {
                _graph_workspace: self.graph_workspace,
                _readout_workspace: self.readout_workspace,
                _readout_output: self.readout_output,
            },
        )
    }
}

/// Device leases of a submitted target launch. Dropping it returns them to
/// their pools, so it is dropped only after the launch's work completed.
pub struct TargetLaunchWorkspace {
    _graph_workspace: TargetGraphWorkspaceLease,
    _readout_workspace: NativeGraphWorkspaceLease,
    _readout_output: Option<NativeGraphOutputLease>,
}

/// Reconciliation payload retained by a submission. It is unavailable before
/// physical finish.
pub struct TargetLaunchCore {
    batch: ValidatedTargetBatch,
    /// What its rows are: policy that tells a prompt from generation reads
    /// this, never the launch's form.
    work: LaunchWork,
    /// Per slot, the request whose rows it runs.
    requests: Vec<RequestId>,
    /// Selected tokens stay held until the launch that reads them completes.
    tokens: TargetTokens,
    advances: Vec<TentativeAdvance>,
    conditioning: Vec<Option<ConditioningRef>>,
    conditioning_slices: Vec<Vec<ConditioningSlice>>,
    boundary: Option<FeatureRef>,
    preceding: Vec<Vec<u64>>,
    guesses: Vec<SlotGuess>,
    /// The store every advance belongs to: it names each attention layer's
    /// history domain and the layer whose regions it binds.
    store: Rc<StateStore>,
    domain: ResourceDomainId,
}

impl TargetLaunchCore {
    pub fn boundary(&self) -> Option<&FeatureRef> {
        self.boundary.as_ref()
    }

    pub fn batch(&self) -> &ValidatedTargetBatch {
        &self.batch
    }

    pub fn work(&self) -> LaunchWork {
        self.work
    }

    pub fn store(&self) -> &Rc<StateStore> {
        &self.store
    }

    pub fn tokens(&self) -> &TargetTokens {
        &self.tokens
    }

    pub fn advances(&self) -> &[TentativeAdvance] {
        &self.advances
    }

    pub fn conditioning(&self) -> &[Option<ConditioningRef>] {
        &self.conditioning
    }

    pub fn conditioning_slices(&self) -> &[Vec<ConditioningSlice>] {
        &self.conditioning_slices
    }

    /// Per slot, the accepted tokens its sequence holds before the slot's
    /// first row, oldest first, as many as the model's row rules read;
    /// empty of a model without one.
    pub fn preceding(&self) -> &[Vec<u64>] {
        &self.preceding
    }

    /// The host's guesses its front rows run on (`[rows, 2]` guess, kind
    /// rows over the class's rows, `routed_step`'s `guesses`, as
    /// `guess_chain` writes them), of a launch of host tokens: each front
    /// row's at the row before it; every other row's zero (an idle row's
    /// included), which the device reads for front rows alone.
    pub fn host_guesses(&self) -> Vec<[i32; 2]> {
        let upload = self.batch.upload();
        let mut rows = vec![[0; 2]; upload.class.rows()];
        for (slot, guess) in self.batch.slots().zip(&self.guesses) {
            let mut front = slot.span().filter(|&row| upload.front[row]);
            let mut matched = 0;
            // An idle row's stays zero: it guessed nothing.
            for (row, guess) in (&mut front).zip(&guess.front) {
                matched += 1;
                if let Some(guess) = guess {
                    rows[row - 1] = [i32::try_from(guess.token.0).expect("a token fits i32"), guess.kind.word()];
                }
            }
            assert!(
                matched == guess.front.len() && front.next().is_none(),
                "each front row of host tokens runs the host's guess"
            );
        }
        rows
    }

    pub fn into_parts(
        self,
    ) -> (
        ValidatedTargetBatch,
        Vec<TentativeAdvance>,
        Vec<Option<ConditioningRef>>,
        Vec<Vec<ConditioningSlice>>,
    ) {
        (
            self.batch,
            self.advances,
            self.conditioning,
            self.conditioning_slices,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation::{FrontRow, GuessKind, Sampling, SelectSpec, Shaping, TokenId, WorkKind};
    use crate::{NativeGraphCharge, NativeGraphPool};
    use magnitude_batching::{ClassLimits, Demand, Row, RowHistory, Slot};
    use magnitude_kernels::{selection_notice, shape_rows};
    use magnitude_state::{
        BankCapacity, ComponentDescriptor, HistoryDomainLayout, HistoryDomainPlan, KvCodec,
        LayerRef, OwnedStateAdvance, RowDemand, StoreBindings,
    };
    use seismic::{
        BackendName, Charges, Device, Element, NativeExecutionArena, NativeGraphFamily,
        NativeSpecialization,
    };

    fn test_pool(
        charges: &Charges,
        domain: &ResourceDomainId,
        family: &NativeGraphFamily,
        activations: usize,
        output_slots: usize,
    ) -> NativeGraphPool {
        let arena = NativeExecutionArena::new(charges, family.workspace_bytes().max(64)).unwrap();
        let charge = NativeGraphCharge {
            workspace_bytes: family.workspace_bytes(),
            sync_bytes: family.sync_bytes(),
            output_bytes: family.output_bytes(),
            upload_regions: crate::planning::serial_upload_regions(family.upload_bytes()),
            activations,
            output_slots,
            committed_bytes: 0,
            graphs: family.graphs(),
        };
        NativeGraphPool::new(
            domain.clone(),
            family,
            &charge,
            &arena,
            charges,
            crate::resources::Outputs::Host,
        )
        .unwrap()
    }

    fn pair_output(charges: &Charges, element: Element, rows: u64) -> GraphOutputTensor {
        let tensor = seismic::Tensor::zeros(charges, element, &[rows, 2]).unwrap();
        GraphOutputTensor::from_test_tensor(tensor)
    }

    fn i32_pair_output(
        _device: &Device,
        charges: &Charges,
        _domain: &ResourceDomainId,
        rows: u64,
    ) -> GraphOutputTensor {
        pair_output(charges, Element::i32(), rows)
    }

    fn f32_pair_output(
        _device: &Device,
        charges: &Charges,
        _domain: &ResourceDomainId,
        rows: u64,
    ) -> GraphOutputTensor {
        pair_output(charges, Element::f32(), rows)
    }

    fn test_store(device: &Device, charges: &Charges) -> (Rc<StateStore>, StoreBindings) {
        let granule = 256 << 10;
        let entry: std::sync::Arc<dyn seismic::memory::Entry> = std::sync::Arc::new(
            seismic::memory::Tally::unbounded(seismic::StoreIdentity::HostRam),
        );
        let pool = seismic::Pool::open(
            &seismic::PoolCharges::one(&device.charges_in(seismic::Contract::Shared, entry)),
            seismic::Contract::Shared,
            granule,
            granule,
        )
        .unwrap();
        StateStore::new(
            charges,
            pool,
            2048,
            32,
            vec![HistoryDomainPlan {
                layout: HistoryDomainLayout::Token {
                    components: vec![ComponentDescriptor::new(
                        LayerRef::Target(0),
                        KvCodec::Dense.spec(seismic::DType::F32, 64, 64),
                        1,
                    )
                    .unwrap()],
                },
                logical_rows: 2048,
            }],
            vec![],
            BankCapacity {
                active: 4,
                in_flight: 4,
                retained: 0,
            },
        )
        .unwrap()
    }

    fn dummy_select() -> SelectSpec {
        SelectSpec {
            sampling: Sampling::Greedy,
            seed: 0,
            position: 0,
            domain: 0,
            mask: None,
            shaping: Shaping::default(),
            history: None,
        }
    }

    fn build_slot_and_advance(
        store: &Rc<StateStore>,
        bindings: &mut StoreBindings,
        rows: usize,
        front: usize,
    ) -> (Slot, TentativeAdvance) {
        let history_domain = store.sole_history_domain().unwrap();
        bindings
            .provision(&[RowDemand { domain: history_domain, rows }], 0)
            .unwrap();
        let advance = TentativeAdvance::Accepted(
            OwnedStateAdvance::begin(store.create().unwrap(), rows)
                .ok()
                .unwrap(),
        );
        let binding = advance.bindings();
        let slot_rows = (0..rows)
            .map(|row| Row {
                token: (row + 1) as i32,
                coordinates: [0, 0, 0, 0],
                histories: vec![RowHistory {
                    visible: Vec::new(),
                    fresh_start: 0,
                    bidirectional_end: None,
                    destination: binding.destinations[history_domain.0][row] as i32,
                }],
                demand: Demand::NONE,
                select: None,
            })
            .collect();
        let slot = Slot {
            rows: slot_rows,
            bank: i32::try_from(binding.previous_bank).unwrap(),
            previous_tape: 0,
            following_bank: i32::try_from(binding.following_bank).unwrap(),
            stop: 1,
            front,
        };
        (slot, advance)
    }

    fn make_launch_inputs(
        store: &Rc<StateStore>,
        bindings: &mut StoreBindings,
        leases: &NativeGraphPool,
        rows: usize,
        front: usize,
        tokens: TargetTokens,
    ) -> TargetLaunchInputs {
        let (slot, advance) = build_slot_and_advance(store, bindings, rows, front);
        let batch = ValidatedTargetBatch::from_slots(
            &[slot],
            32,
            ClassLimits {
                rows: 4,
                segments: 4,
                slots: 2,
            },
        )
        .unwrap();
        let front_kind = match &tokens {
            TargetTokens::Host => FrontKind::Host(GuessKind::Chained),
            TargetTokens::Selected { .. } => FrontKind::Device,
        };
        let operation = Operation::Forward {
            request: RequestId(1),
            kind: WorkKind::Decode,
            tokens: (0..rows).map(|r| TokenId((r + 1) as u32)).collect(),
            position: 0,
            conditioning: None,
            demand: Demand::NONE,
            select: Vec::new(),
            committed: 1,
            prime: None,
            front: (0..front)
                .map(|_| FrontRow {
                    select: dummy_select(),
                    kind: front_kind,
                })
                .collect(),
            chain: None,
        };
        TargetLaunchInputs::new(
            batch,
            LaunchWork::Generation,
            vec![RequestId(1)],
            tokens,
            vec![advance],
            vec![None],
            vec![Vec::new()],
            &[operation],
            leases.acquire_workspace().unwrap(),
            leases.acquire_output().unwrap(),
            leases.acquire_workspace().unwrap(),
            leases.acquire_output().unwrap(),
        )
    }

    #[test]
    fn selected_tokens_validate_element_and_guesses() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap();
        let charges = Charges::tally(&device);
        let domain = ResourceDomainId::new("target-selected-test").unwrap();
        let (store, mut bindings) = test_store(&device, &charges);

        // Lease pool backed by a minimal graph family.
        let implementation = selection_notice::native_implementation(&device)
            .unwrap()
            .unwrap();
        let copy = selection_notice::native_for_device(
            &device,
            &implementation
                .default_specialization(&NativeSpecialization::new())
                .unwrap(),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let selection = graph.port(Element::i32(), &[2, 2]).unwrap();
        let mut notice = graph.local_for(&copy, "notice", &[("M", 2)]).unwrap();
        graph
            .enqueue(
                &copy,
                selection_notice::WorkflowArgs {
                    selection: selection.tensor().into(),
                    notice: notice.tensor_mut().into(),
                },
            )
            .unwrap();
        graph.export(notice.tensor()).unwrap();
        let plan = graph.seal().unwrap();
        let family = NativeGraphFamily::new(&[plan], 0).unwrap();
        let leases = test_pool(&charges, &domain, &family, 2, 2);

        // Determine the class row count for a 2-row (1 exact + 1 front) decode batch.
        let (probe_slot, _) = build_slot_and_advance(&store, &mut bindings, 2, 1);
        let class_rows = ValidatedTargetBatch::from_slots(
            &[probe_slot],
            32,
            ClassLimits {
                rows: 4,
                segments: 4,
                slots: 2,
            },
        )
        .unwrap()
        .class()
        .rows() as u64;

        let valid_i32 = i32_pair_output(&device, &charges, &domain, class_rows);
        let wrong_extent_i32 = i32_pair_output(&device, &charges, &domain, class_rows + 1);
        let wrong_element_f32 = f32_pair_output(&device, &charges, &domain, class_rows);

        // Wrong element on `selected` (f32 instead of i32, matching [rows, 2] extents) is rejected.
        let inputs = make_launch_inputs(
            &store,
            &mut bindings,
            &leases,
            2,
            1,
            TargetTokens::Selected {
                selected: wrong_element_f32.clone(),
                guesses: Some(valid_i32.clone()),
            },
        );
        assert!(ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64).is_err());

        // Wrong extents on `selected` is rejected.
        let inputs = make_launch_inputs(
            &store,
            &mut bindings,
            &leases,
            2,
            1,
            TargetTokens::Selected {
                selected: wrong_extent_i32.clone(),
                guesses: Some(valid_i32.clone()),
            },
        );
        assert!(ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64).is_err());

        // Wrong element on `guesses` (f32 instead of i32) is rejected.
        let inputs = make_launch_inputs(
            &store,
            &mut bindings,
            &leases,
            2,
            1,
            TargetTokens::Selected {
                selected: valid_i32.clone(),
                guesses: Some(wrong_element_f32),
            },
        );
        assert!(ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64).is_err());

        // Wrong extents on `guesses` is rejected.
        let inputs = make_launch_inputs(
            &store,
            &mut bindings,
            &leases,
            2,
            1,
            TargetTokens::Selected {
                selected: valid_i32.clone(),
                guesses: Some(wrong_extent_i32),
            },
        );
        assert!(ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64).is_err());

        // Missing `guesses` when a slot has front rows is rejected.
        let inputs = make_launch_inputs(
            &store,
            &mut bindings,
            &leases,
            2,
            1,
            TargetTokens::Selected {
                selected: valid_i32.clone(),
                guesses: None,
            },
        );
        assert!(ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64).is_err());

        // Valid `selected` and `guesses` (both i32 [class_rows, 2]) with front rows are accepted.
        let inputs = make_launch_inputs(
            &store,
            &mut bindings,
            &leases,
            2,
            1,
            TargetTokens::Selected {
                selected: valid_i32.clone(),
                guesses: Some(valid_i32),
            },
        );
        assert!(ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64).is_ok());

        // Host tokens with a front row validate and `host_guesses()` writes at `row - 1`.
        let inputs = make_launch_inputs(&store, &mut bindings, &leases, 2, 1, TargetTokens::Host);
        let validated = ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64)
            .ok()
            .expect("valid host front launch");
        let (core, _) = validated.into_submission_parts();
        let host_guesses = core.host_guesses();
        assert_eq!(host_guesses[0], [2, GuessKind::Chained.word()]);
        assert_eq!(host_guesses[1], [0, 0]);

        // A slot whose first row would be a front row (`front == rows.len()`) is rejected.
        let (bad_slot, _) = build_slot_and_advance(&store, &mut bindings, 1, 1);
        assert!(
            ValidatedTargetBatch::from_slots(
                &[bad_slot],
                32,
                ClassLimits {
                    rows: 4,
                    segments: 4,
                    slots: 2,
                },
            )
            .is_err()
        );
    }
}
