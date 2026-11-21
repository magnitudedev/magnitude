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

/// Per-slot state and identity common to every target launch.
pub struct TargetSlot {
    /// The request whose rows this slot runs.
    pub request: RequestId,
    /// State advance owned by this slot.
    pub advance: TentativeAdvance,
    /// Accepted tokens the sequence holds before this slot's first row,
    /// oldest first, as many as the model's row rules read (empty for a
    /// model without one): the host's form of the n-gram token window.
    pub preceding: Vec<u64>,
}

/// Per-slot host-only launch inputs: conditioning overlays and host guesses
/// for front rows. Unrepresentable on a [`TargetTokens::Selected`] launch.
pub struct TargetHostSlot {
    pub slot: TargetSlot,
    pub conditioning: Option<ConditioningRef>,
    pub conditioning_slices: Vec<ConditioningSlice>,
    /// The host's guesses this slot's front rows run on ([`SlotGuess`]).
    pub guesses: SlotGuess,
}

/// Iterator over the common per-slot state of a [`TargetTokens`] payload.
pub enum TargetSlotsIter<'a> {
    Host(std::slice::Iter<'a, TargetHostSlot>),
    Selected(std::slice::Iter<'a, TargetSlot>),
}

impl<'a> Iterator for TargetSlotsIter<'a> {
    type Item = &'a TargetSlot;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Host(iter) => iter.next().map(|host| &host.slot),
            Self::Selected(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Host(iter) => iter.size_hint(),
            Self::Selected(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for TargetSlotsIter<'_> {}

/// Where a launch's input tokens come from, coupled with the per-slot
/// payloads valid for that token source. Parallel slot vector mismatches
/// and `Selected` launches carrying host guesses or conditioning are
/// unrepresentable by construction.
pub enum TargetTokens {
    /// The batch's host tokens, written into the step's upload region,
    /// together with each slot's state, conditioning, and host guesses.
    Host(Vec<TargetHostSlot>),
    /// Tokens the device selected for it (`[rows, 2]` token, status rows in
    /// batch row order), read on the device: the step continues the step
    /// before, each of its rows a front pass's exact token or the guess of
    /// the chain made behind that step (`guess_chain`'s `next_selected`);
    /// and of a load that fronts, its front rows' guesses (`[rows, 2]`
    /// guess, kind rows, `guess_chain`'s `next_guesses`), whose kinds the
    /// step's words carry (`routed_step`), together with each slot's state.
    Selected {
        selected: GraphOutputTensor,
        guesses: Option<GraphOutputTensor>,
        slots: Vec<TargetSlot>,
    },
}

impl TargetTokens {
    pub fn slot_count(&self) -> usize {
        match self {
            Self::Host(slots) => slots.len(),
            Self::Selected { slots, .. } => slots.len(),
        }
    }

    pub fn slot(&self, index: usize) -> Option<&TargetSlot> {
        match self {
            Self::Host(slots) => slots.get(index).map(|host| &host.slot),
            Self::Selected { slots, .. } => slots.get(index),
        }
    }

    pub fn slots(&self) -> TargetSlotsIter<'_> {
        match self {
            Self::Host(slots) => TargetSlotsIter::Host(slots.iter()),
            Self::Selected { slots, .. } => TargetSlotsIter::Selected(slots.iter()),
        }
    }

    pub fn into_advances(self) -> Vec<TentativeAdvance> {
        match self {
            Self::Host(slots) => slots.into_iter().map(|host| host.slot.advance).collect(),
            Self::Selected { slots, .. } => slots.into_iter().map(|slot| slot.advance).collect(),
        }
    }
}

/// The host's guesses a launch slot's front rows run on, by depth: of a
/// launch of host tokens, the guess each of its operation's front rows
/// carries ([`FrontKind::Host`]), none of an idle row ([`FrontKind::Idle`]),
/// whose kinds the step's words read (`routed_step`); empty of a slot of no
/// front row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotGuess {
    front: Vec<Option<Guess>>,
}

impl SlotGuess {
    /// The host's guesses of a host-token launch's slot of `operation`: its
    /// front rows' ([`FrontKind::Host`], each its front row's token, and
    /// [`FrontKind::Idle`], none). A [`FrontKind::Device`] front row, or an
    /// operation other than a forward, is the caller's broken invariant.
    pub fn of(operation: &Operation) -> Self {
        let Operation::Forward { tokens, front: front_rows, .. } = operation else {
            panic!("a target launch's slot runs a forward");
        };
        let first = tokens.len() - front_rows.len();
        let front = front_rows
            .iter()
            .zip(&tokens[first..])
            .map(|(row, &token)| match row.kind {
                FrontKind::Host(kind) => Some(Guess { token, kind }),
                FrontKind::Idle => None,
                FrontKind::Device => panic!("a front row's guess of Device in a launch whose tokens the host builds"),
            })
            .collect();
        Self { front }
    }
}

pub struct TargetLaunchInputs {
    batch: ValidatedTargetBatch,
    /// What its rows are (`LaunchWork::of` its requests' kinds).
    work: LaunchWork,
    tokens: TargetTokens,
    boundary: Option<FeatureRef>,
    graph_workspace: TargetGraphWorkspaceLease,
    /// The launch's place in its chain's output rotation
    /// (`TargetProgram::submit`).
    graph_output: TargetGraphOutputLease,
    readout_workspace: NativeGraphWorkspaceLease,
    readout_output: NativeGraphOutputLease,
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
        tokens: TargetTokens,
        graph_workspace: TargetGraphWorkspaceLease,
        graph_output: TargetGraphOutputLease,
        readout_workspace: NativeGraphWorkspaceLease,
        readout_output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            batch,
            work,
            tokens,
            boundary: None,
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output,
        }
    }

    pub(crate) fn with_boundary(mut self, boundary: Option<FeatureRef>) -> Self {
        self.boundary = boundary;
        self
    }

    pub(crate) fn into_advances(self) -> Vec<TentativeAdvance> {
        self.tokens.into_advances()
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
        if slots != self.tokens.slot_count() {
            return Err(invalid("slot count differs from batch".into()));
        }
        // Selected tokens continue a decode pass: each slot is one exact
        // row, then its front rows, each row's token the selection of its
        // predecessor's row at its batch row.
        let front = self.batch.upload().front;
        let fronts = |slot: &magnitude_batching::TargetBatchSlot<'_>| {
            front[slot.span()].iter().filter(|&&front| front).count()
        };
        if self
            .batch
            .slots()
            .any(|slot| front[slot.span()].first() == Some(&true))
        {
            return Err(invalid(
                "a slot's first row cannot be a front row".into(),
            ));
        }
        match &self.tokens {
            TargetTokens::Host(host_slots) => {
                if self
                    .batch
                    .slots()
                    .zip(host_slots)
                    .any(|(slot, host)| host.guesses.front.len() != fronts(&slot))
                {
                    return Err(invalid(
                        "front rows' guesses are the host's exactly where it builds the tokens"
                            .into(),
                    ));
                }
            }
            TargetTokens::Selected {
                selected, guesses, ..
            } => {
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
                {
                    return Err(invalid(
                        "selected tokens differ from the launch's decode passes".into(),
                    ));
                }
            }
        }
        if self.graph_workspace.domain() != domain
            || self.graph_output.domain() != domain
            || self.readout_workspace.domain() != domain
            || self.readout_output.domain() != domain
        {
            return Err(invalid(
                "workspace or output lease differs from batch domain/class".into(),
            ));
        }
        let mut expected: Vec<[i32; 2]> = Vec::new();
        for (index, (slot, target_slot)) in self.batch.slots().zip(self.tokens.slots()).enumerate()
        {
            let advance = &target_slot.advance;
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
        }
        if let TargetTokens::Host(host_slots) = &self.tokens {
            let mut occupied: Vec<bool> = Vec::new();
            for (index, (slot, host)) in self.batch.slots().zip(host_slots).enumerate() {
                if let Some(lease) = &host.conditioning {
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
                if !host.conditioning_slices.is_empty() {
                    occupied.clear();
                    occupied.resize(slot.rows(), false);
                    for slice in &host.conditioning_slices {
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
    readout_output: NativeGraphOutputLease,
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
            tokens,
            boundary,
            graph_workspace,
            graph_output,
            readout_workspace,
            readout_output,
        } = inputs;
        Ok(Self {
            core: TargetLaunchCore {
                batch,
                work,
                tokens,
                boundary,
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

    pub(crate) fn slots(&self) -> TargetSlotsIter<'_> {
        self.core.slots()
    }

    pub(crate) fn slot(&self, index: usize) -> Option<&TargetSlot> {
        self.core.slot(index)
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }

    /// The launch's state advances, one per slot: a step chained behind it
    /// follows them.
    pub(crate) fn advances(&self) -> impl ExactSizeIterator<Item = &TentativeAdvance> {
        self.core.advances()
    }

    pub(crate) fn submission_parts_mut(
        &mut self,
    ) -> (
        &TargetLaunchCore,
        &mut TargetGraphWorkspaceLease,
        &mut TargetGraphOutputLease,
        &mut NativeGraphWorkspaceLease,
        &mut NativeGraphOutputLease,
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
    _readout_output: NativeGraphOutputLease,
}

/// Reconciliation payload retained by a submission. It is unavailable before
/// physical finish.
pub struct TargetLaunchCore {
    batch: ValidatedTargetBatch,
    /// What its rows are: policy that tells a prompt from generation reads
    /// this, never the launch's form.
    work: LaunchWork,
    /// Input token source and per-slot payloads; selected tokens stay held
    /// until the launch that reads them completes.
    tokens: TargetTokens,
    boundary: Option<FeatureRef>,
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

    pub fn slot_count(&self) -> usize {
        self.tokens.slot_count()
    }

    pub fn slot(&self, index: usize) -> Option<&TargetSlot> {
        self.tokens.slot(index)
    }

    pub fn slots(&self) -> TargetSlotsIter<'_> {
        self.tokens.slots()
    }

    pub fn advances(&self) -> impl ExactSizeIterator<Item = &TentativeAdvance> {
        self.tokens.slots().map(|slot| &slot.advance)
    }

    /// The host's guesses its front rows run on (`[rows, 2]` guess, kind
    /// rows over the class's rows, `routed_step`'s `guesses`, as
    /// `guess_chain` writes them), of a launch of host tokens: each front
    /// row's at the row before it; every other row's zero (an idle row's
    /// included), which the device reads for front rows alone.
    pub fn host_guesses(&self) -> Vec<[i32; 2]> {
        let TargetTokens::Host(host_slots) = &self.tokens else {
            panic!("host_guesses called on a Selected target launch");
        };
        let upload = self.batch.upload();
        let mut rows = vec![[0; 2]; upload.class.rows()];
        for (slot, host) in self.batch.slots().zip(host_slots) {
            let mut front = slot.span().filter(|&row| upload.front[row]);
            let mut matched = 0;
            // An idle row's stays zero: it guessed nothing.
            for (row, guess) in (&mut front).zip(&host.guesses.front) {
                matched += 1;
                if let Some(guess) = guess {
                    rows[row - 1] = [
                        i32::try_from(guess.token.0).expect("a token fits i32"),
                        guess.kind.word(),
                    ];
                }
            }
            assert!(
                matched == host.guesses.front.len() && front.next().is_none(),
                "each front row of host tokens runs the host's guess"
            );
        }
        rows
    }

    pub fn into_advances(self) -> Vec<TentativeAdvance> {
        self.tokens.into_advances()
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

    enum TestTokens {
        Host,
        Selected {
            selected: GraphOutputTensor,
            guesses: Option<GraphOutputTensor>,
        },
    }

    fn make_launch_inputs(
        store: &Rc<StateStore>,
        bindings: &mut StoreBindings,
        leases: &NativeGraphPool,
        rows: usize,
        front: usize,
        tokens: TestTokens,
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
            TestTokens::Host => FrontKind::Host(GuessKind::Chained),
            TestTokens::Selected { .. } => FrontKind::Device,
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
        let target_slot = TargetSlot {
            request: RequestId(1),
            advance,
            preceding: Vec::new(),
        };
        let tokens = match tokens {
            TestTokens::Host => TargetTokens::Host(vec![TargetHostSlot {
                slot: target_slot,
                conditioning: None,
                conditioning_slices: Vec::new(),
                guesses: SlotGuess::of(&operation),
            }]),
            TestTokens::Selected { selected, guesses } => TargetTokens::Selected {
                selected,
                guesses,
                slots: vec![target_slot],
            },
        };
        TargetLaunchInputs::new(
            batch,
            LaunchWork::Generation,
            tokens,
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
            TestTokens::Selected {
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
            TestTokens::Selected {
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
            TestTokens::Selected {
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
            TestTokens::Selected {
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
            TestTokens::Selected {
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
            TestTokens::Selected {
                selected: valid_i32.clone(),
                guesses: Some(valid_i32),
            },
        );
        assert!(ValidatedTargetLaunch::new(inputs, &store, &domain, 64, 64).is_ok());

        // Host tokens with a front row validate and `host_guesses()` writes at `row - 1`.
        let inputs = make_launch_inputs(&store, &mut bindings, &leases, 2, 1, TestTokens::Host);
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
