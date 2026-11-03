//! The sole join of target row semantics, owned state, conditioning, and
//! planned device leases.

use crate::{
    ConditioningRef, FeatureRef, FeatureSpan, GraphOutputTensor, InvariantError,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, ResourceDomainId,
    TargetGraphOutputLease, TargetGraphWorkspaceLease,
};
use magnitude_batching::ValidatedTargetBatch;
use magnitude_state::{StateStore, TentativeAdvance};
use std::rc::Rc;

/// Where a launch's input tokens come from.
#[derive(Clone)]
pub enum TargetTokens {
    /// The batch's host tokens, written into the step's upload region.
    Host,
    /// The previous step's selections (`[rows, 2]` token, status rows in slot
    /// order), read on the device: the step continues that one.
    Selected(GraphOutputTensor),
}

/// The rows a target launch computes: one batch, or one advance wider than
/// any launch class in consecutive chunks, layer-major.
pub enum LaunchRows {
    Batch(ValidatedTargetBatch),
    LayerMajor(LayerMajor),
}

/// A lone advance computed in chunks, each a batch of one slot holding the
/// advance rows `rows`, in order, in the class of its own rows: the first's
/// the launch's, no later one's wider. Without barriers every chunk runs
/// every block in turn, recurrent state passing chunk to chunk.
pub struct LayerMajor {
    pub(crate) chunks: Vec<LaunchChunk>,
    pub(crate) barriers: Option<Barriers>,
}

/// The blocks of a layer-major launch every chunk's rows publish to before
/// any chunk's rows attend there, so each reads the whole advance there:
/// `blocks`, ascending and at least one. Between two of them every chunk
/// runs the blocks before the next; `held` keeps every chunk's hidden rows
/// between those stretches, the first chunk's class rows a chunk of the
/// widest input of a barrier block.
pub struct Barriers {
    pub(crate) blocks: Vec<usize>,
    pub(crate) held: seismic::Tensor,
}

/// One chunk of a layer-major launch: its batch and the advance rows it
/// computes.
pub struct LaunchChunk {
    pub(crate) batch: ValidatedTargetBatch,
    pub(crate) rows: std::ops::Range<usize>,
}

impl LaunchRows {
    /// The launch's batches in row order, each with the advance row of its
    /// first slot's first row (0 for a lone batch).
    pub fn batches(&self) -> Vec<(&ValidatedTargetBatch, usize)> {
        match self {
            Self::Batch(batch) => vec![(batch, 0)],
            Self::LayerMajor(layered) => layered
                .chunks
                .iter()
                .map(|chunk| (&chunk.batch, chunk.rows.start))
                .collect(),
        }
    }

    /// Rows the launch computes.
    pub fn actual_rows(&self) -> usize {
        self.batches()
            .iter()
            .map(|(batch, _)| batch.actual_rows())
            .sum()
    }
}

pub struct TargetLaunchInputs {
    rows: LaunchRows,
    tokens: TargetTokens,
    advances: Vec<TentativeAdvance>,
    conditioning: Vec<Option<ConditioningRef>>,
    conditioning_slices: Vec<Vec<ConditioningSlice>>,
    boundary: Option<FeatureRef>,
    /// Per slot, the accepted tokens its sequence holds before the slot's
    /// first row, oldest first, as many as the model's row rules read (none
    /// of a model without one): the host's form of the n-gram token window.
    preceding: Vec<Vec<u64>>,
    /// Per slot, what its rows read on global attention layers; empty when
    /// every slot's rows attend causally.
    visibility: Vec<magnitude_readout::Visibility>,
    /// Per advance, the stores of its sequence's retained rows its readouts
    /// name (`RowReadout::Retain`, `RowReadout::Scores`); none in a launch
    /// given none (`with_stores`).
    stores: Vec<Vec<FeatureRef>>,
    graph_workspace: TargetGraphWorkspaceLease,
    graph_outputs: [TargetGraphOutputLease; 2],
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

impl ConditioningSlice {
    /// What rows `rows` of an advance read of `slices`, placed in its rows:
    /// each slice's part within them, placed in them. A slot computing its
    /// whole advance reads every slice as it is; a layer-major chunk reads
    /// its own rows' part of each.
    pub(crate) fn within(slices: &[Self], rows: &std::ops::Range<usize>) -> Vec<Self> {
        slices
            .iter()
            .filter_map(|slice| {
                let start = slice.destination.max(rows.start);
                let end = (slice.destination + slice.source.count).min(rows.end);
                (start < end).then(|| Self {
                    source: FeatureSpan {
                        features: slice.source.features.clone(),
                        start: slice.source.start + (start - slice.destination),
                        count: end - start,
                    },
                    destination: start - rows.start,
                })
            })
            .collect()
    }
}

impl TargetLaunchInputs {
    pub fn new(
        rows: LaunchRows,
        tokens: TargetTokens,
        advances: Vec<TentativeAdvance>,
        conditioning: Vec<Option<ConditioningRef>>,
        conditioning_slices: Vec<Vec<ConditioningSlice>>,
        graph_workspace: TargetGraphWorkspaceLease,
        graph_outputs: [TargetGraphOutputLease; 2],
        readout_workspace: NativeGraphWorkspaceLease,
        readout_output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            rows,
            tokens,
            advances,
            conditioning,
            conditioning_slices,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output: Some(readout_output),
            boundary: None,
            preceding: Vec::new(),
            visibility: Vec::new(),
            stores: Vec::new(),
        }
    }

    /// Per slot, what its rows read on global attention layers.
    pub(crate) fn with_visibility(mut self, visibility: Vec<magnitude_readout::Visibility>) -> Self {
        self.visibility = visibility;
        self
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

    /// Per slot, the retained-row stores its readouts name
    /// (`TargetLaunchCore::stores`).
    pub(crate) fn with_stores(mut self, stores: Vec<Vec<FeatureRef>>) -> Self {
        self.stores = stores;
        self
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        LaunchRows,
        Vec<TentativeAdvance>,
        Vec<Option<ConditioningRef>>,
        Vec<Vec<ConditioningSlice>>,
    ) {
        (
            self.rows,
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
            let LaunchRows::Batch(batch) = &self.rows else {
                return Err(invalid("a layer-major launch captures no boundary".into()));
            };
            if batch.actual_slots() != 1
                || boundary.domain() != domain
                || boundary.allocation().rows() != 1
                || boundary.allocation().width() != boundary_width
                || boundary.allocation().is_graph_backed()
                || boundary.allocation().tensor().is_err()
                || !batch
                    .class()
                    .demand()
                    .contains(magnitude_batching::Demand::FEATURES)
            {
                return Err(invalid(
                    "invalid independent boundary feature destination".into(),
                ));
            }
        }
        let advances = self.advances.len();
        if advances != self.conditioning.len()
            || advances != self.conditioning_slices.len()
            || !(self.preceding.is_empty() || self.preceding.len() == advances)
            || !(self.visibility.is_empty() || self.visibility.len() == advances)
            || !(self.stores.is_empty() || self.stores.len() == advances)
            || self.stores.iter().flatten().any(|store| {
                store.domain() != domain
                    || store.allocation().width() != hidden_width
                    || store.allocation().is_graph_backed()
            })
        {
            return Err(invalid(
                "slot, advance, conditioning and store counts disagree, or a store is not this \
                 domain's own hidden-width rows"
                    .into(),
            ));
        }
        // Each slot below reads its rows' part of its advance's slices, so
        // every slice holds rows, within its advance's.
        if self.conditioning_slices.iter().zip(&self.advances).any(|(slices, advance)| {
            slices.iter().any(|slice| {
                slice.source.count == 0
                    || slice
                        .destination
                        .checked_add(slice.source.count)
                        .is_none_or(|end| end > advance.rows())
            })
        }) {
            return Err(invalid("a conditioning slice is empty or exceeds its advance".into()));
        }
        // Every slot of the launch: its advance, the advance rows it computes,
        // and its chunk of the launch's chunks.
        let slots = match &self.rows {
            LaunchRows::Batch(batch) => {
                if batch.actual_slots() != advances {
                    return Err(invalid(
                        "slot, advance, and conditioning counts disagree".into(),
                    ));
                }
                batch
                    .slots()
                    .zip(0..)
                    .map(|(slot, advance)| (slot, advance, 0..self.advances[advance].rows(), 0, 1))
                    .collect::<Vec<_>>()
            }
            LaunchRows::LayerMajor(layered) => {
                let rows = layered
                    .chunks
                    .iter()
                    .try_fold(0, |end, chunk| (chunk.rows.start == end).then_some(chunk.rows.end));
                let malformed = || {
                    invalid(
                        "a layer-major launch is not one advance's consecutive one-slot chunks \
                         reading only readout values"
                            .into(),
                    )
                };
                let [first, _, ..] = layered.chunks.as_slice() else {
                    return Err(malformed());
                };
                let class = first.batch.class();
                if advances != 1
                    || rows != Some(self.advances[0].rows())
                    || layered.chunks.iter().any(|chunk| {
                        let other = chunk.batch.class();
                        let demand = other.demand();
                        chunk.batch.actual_slots() != 1
                            || other.rows() > class.rows()
                            || other.segments() != class.segments()
                            || demand != demand & magnitude_batching::Demand::READOUT
                    })
                    || !matches!(self.tokens, TargetTokens::Host)
                    || self.conditioning.iter().any(Option::is_some)
                    || layered.barriers.as_ref().is_some_and(|barriers| {
                        barriers.blocks.is_empty()
                            || barriers.blocks.windows(2).any(|pair| pair[0] >= pair[1])
                    })
                {
                    return Err(malformed());
                }
                let chunks = layered.chunks.len();
                layered
                    .chunks
                    .iter()
                    .enumerate()
                    .map(|(chunk, launch)| {
                        let slot = launch.batch.slot(0).expect("a chunk holds one slot");
                        (slot, 0, launch.rows.clone(), chunk, chunks)
                    })
                    .collect()
            }
        };
        if let (TargetTokens::Selected(selected), LaunchRows::Batch(batch)) = (&self.tokens, &self.rows) {
            let rows = batch.class().rows() as u64;
            if selected.tensor().extents() != [rows, 2]
                || slots.iter().any(|(slot, ..)| slot.rows() != 1)
                || self.conditioning.iter().any(Option::is_some)
            {
                return Err(invalid(
                    "selected tokens differ from the launch's one-row slots".into(),
                ));
            }
        }
        if self.graph_workspace.domain() != domain
            || self
                .graph_outputs
                .iter()
                .any(|lease| lease.domain() != domain)
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
        for (index, (slot, advance_index, computed, chunk, chunks)) in slots.into_iter().enumerate() {
            let advance = &self.advances[advance_index];
            let conditioning = &self.conditioning[advance_index];
            let slices =
                ConditioningSlice::within(&self.conditioning_slices[advance_index], &computed);
            if !advance.belongs_to(store)
                || computed.end > advance.rows()
                || slot.rows() != computed.len()
            {
                return Err(invalid(format!(
                    "slot {index} differs from its state advance"
                )));
            }
            let binding = advance.bindings();
            let version =
                crate::domain::chunk_version(&binding, chunks, chunk, computed.len()).map_err(invalid)?;
            if version.bank != slot.bank() {
                return Err(invalid(format!("slot {index} uses another recurrent bank")));
            }
            if version.following != slot.following_bank() {
                return Err(invalid(format!(
                    "slot {index} publishes to another recurrent successor bank"
                )));
            }
            if slot.history_domains() != store.history_reads() {
                return Err(invalid(format!(
                    "slot {index} history domain count differs from its store"
                )));
            }
            // A blocked slot's rows on a global read read as `blocked_reads`
            // forms them.
            let blocked = self
                .visibility
                .get(advance_index)
                .filter(|visibility| !visibility.is_causal());
            for (read, visibility) in (0..store.history_reads())
                .filter(|read| crate::domain::global_read(store, *read))
                .flat_map(|read| blocked.map(|visibility| (read, visibility)))
            {
                let stored = store.history_domains().count();
                let domain = if read < stored {
                    magnitude_state::HistoryDomainId(read)
                } else {
                    store.shared_source_domains()[read - stored]
                };
                let span = crate::domain::fresh_span(store, read);
                let packed = slot
                    .history(read)
                    .ok_or_else(|| invalid(format!("slot {index} lacks a history read")))?;
                let appended = crate::domain::AppendedRuns::of(store, advance, domain);
                for row in 0..slot.rows() {
                    let position = advance.position() + computed.start + row;
                    let (reads, block) = visibility
                        .reads(position)
                        .zip(visibility.block(position))
                        .ok_or_else(|| invalid(format!("slot {index} row {row} is outside its input")))?;
                    let (expected, own) = crate::domain::blocked_reads(
                        advance,
                        domain,
                        &appended,
                        crate::domain::BlockedRow {
                            reads: &reads,
                            block: &block,
                        },
                        computed.start + row,
                        &computed,
                        span,
                    )
                    .map_err(invalid)?;
                    let supplied = packed.visible[row]
                        .iter()
                        .filter(|range| **range != [0, 0]);
                    let fresh = [
                        slot.start() + own.start - computed.start,
                        slot.start() + own.end - computed.start,
                    ]
                    .map(i32::try_from);
                    let fresh = match fresh {
                        [Ok(start), Ok(end)] => [start, end],
                        _ => return Err(invalid(format!("slot {index} fresh rows exceed i32"))),
                    };
                    if supplied.ne(expected.iter())
                        || (read < stored && packed.fresh[row] != fresh)
                    {
                        return Err(invalid(format!(
                            "slot {index} row {row} blocked visibility differs from its state"
                        )));
                    }
                }
            }
            // A Shared source domain's reads: its accepted rows, then rows its
            // source appends in this advance.
            for (read, domain) in store
                .shared_source_domains()
                .into_iter()
                .enumerate()
                .map(|(position, domain)| (store.history_domains().count() + position, domain))
                .filter(|(read, _)| blocked.is_none() || !crate::domain::global_read(store, *read))
            {
                let packed = slot
                    .history(read)
                    .ok_or_else(|| invalid(format!("slot {index} lacks a shared history read")))?;
                let appended = crate::domain::AppendedRuns::of(store, advance, domain);
                for (row, visible) in packed.visible.iter().enumerate() {
                    let (accepted, _) =
                        crate::domain::accepted_history(store, advance, domain, computed.start + row)
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
                if blocked.is_some() && crate::domain::global_read(store, domain.0) {
                    continue;
                }
                let packed = slot
                    .history(domain.0)
                    .ok_or_else(|| invalid(format!("slot {index} lacks a history domain")))?;
                // An indexer's rows read as `indexer_history` forms them.
                if crate::domain::indexer_domain(store, domain) {
                    let appended = crate::domain::AppendedRuns::of(store, advance, domain);
                    for (row, (destination, visible)) in
                        packed.destinations.iter().zip(packed.visible).enumerate()
                    {
                        let expected = crate::domain::indexer_history(
                            store,
                            advance,
                            &appended,
                            domain,
                            computed.start + row,
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
                if !destinations.is_empty() && destinations.len() != advance.rows() {
                    return Err(invalid(format!("slot {index} destination count differs")));
                }
                let kind = store.history_domain_kind(domain);
                let cached = crate::domain::cached_causal_domain(store, domain);
                let appended = (cached || computed.start > 0)
                    .then(|| crate::domain::AppendedRuns::of(store, advance, domain));
                for (row, (destination, visible)) in
                    packed.destinations.iter().zip(packed.visible).enumerate()
                {
                    let row = computed.start + row;
                    let expected = destinations
                        .get(row)
                        .map(|destination| i32::try_from(*destination).ok())
                        .unwrap_or(Some(-1));
                    if expected != Some(*destination) {
                        return Err(invalid(format!(
                            "slot {index} row {row} destination differs"
                        )));
                    }
                    // Each row reads its domain's accepted rows from its
                    // query's window start.
                    let from = kind.visible_from(advance.position() + row);
                    let expected = advance
                        .visible_ranges(domain, from)
                        .into_iter()
                        .map(|(start, count)| {
                            let end = start.checked_add(count)?;
                            Some([i32::try_from(start).ok()?, i32::try_from(end).ok()?])
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            invalid(format!(
                                "slot {index} history range exceeds packed coordinates"
                            ))
                        })?;
                    let supplied = visible
                        .iter()
                        .copied()
                        .filter(|range| *range != [0, 0])
                        .collect::<Vec<_>>();
                    // The row's first fresh row in the advance, as
                    // `accepted_history` reports it.
                    let fresh_start = from.saturating_sub(advance.position());
                    let fresh = packed.fresh[row - computed.start];
                    let prior = |end: usize| -> Result<Vec<[i32; 2]>, InvariantError> {
                        let mut prepared = expected.clone();
                        if let Some(appended) = &appended {
                            prepared.extend(appended.causal_prior(fresh_start, end).map_err(invalid)?);
                        }
                        Ok(prepared)
                    };
                    let matches = if cached {
                        // Media rows retain their declared dense fresh
                        // block; causal compressed rows may additionally see
                        // exactly their own earlier destinations, never a
                        // future row.
                        supplied == expected || (supplied == prior(row)? && fresh[1] - fresh[0] == 1)
                    } else if fresh_start < computed.start {
                        // Rows of the advance an earlier chunk computed are
                        // read at their destinations.
                        supplied == prior(computed.start)?
                    } else {
                        supplied == expected
                    };
                    if !matches {
                        return Err(invalid(format!(
                            "slot {index} visibility differs from accepted state"
                        )));
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
            let mut occupied = vec![false; slot.rows()];
            for slice in &slices {
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
        Ok(())
    }
}

/// Opaque checked launch. Construction returns unconsumed inputs if its
/// ownership or batch invariants fail.
pub struct ValidatedTargetLaunch {
    core: TargetLaunchCore,
    graph_workspace: TargetGraphWorkspaceLease,
    graph_outputs: [TargetGraphOutputLease; 2],
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
            rows,
            tokens,
            advances,
            conditioning,
            conditioning_slices,
            boundary,
            preceding,
            visibility: _,
            stores,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        } = inputs;
        Ok(Self {
            core: TargetLaunchCore {
                rows,
                tokens,
                advances,
                conditioning,
                conditioning_slices,
                boundary,
                preceding,
                stores,
                store: store.clone(),
                domain: domain.clone(),
            },
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        })
    }


    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }

    pub(crate) fn submission_parts_mut(
        &mut self,
    ) -> (
        &TargetLaunchCore,
        &mut TargetGraphWorkspaceLease,
        &mut [TargetGraphOutputLease; 2],
        &mut NativeGraphWorkspaceLease,
        &mut Option<NativeGraphOutputLease>,
    ) {
        (
            &self.core,
            &mut self.graph_workspace,
            &mut self.graph_outputs,
            &mut self.readout_workspace,
            &mut self.readout_output,
        )
    }

    /// The reconciliation payload and the device leases the submitted work
    /// uses; the leases stay held until the submission is finished. The
    /// residual pair returns to its pool now: every reader of it was queued
    /// by this submission, and the device runs the next launch that takes the
    /// pair after this one.
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
    rows: LaunchRows,
    /// Selected tokens stay held until the launch that reads them completes.
    tokens: TargetTokens,
    advances: Vec<TentativeAdvance>,
    conditioning: Vec<Option<ConditioningRef>>,
    conditioning_slices: Vec<Vec<ConditioningSlice>>,
    boundary: Option<FeatureRef>,
    preceding: Vec<Vec<u64>>,
    stores: Vec<Vec<FeatureRef>>,
    /// The store every advance belongs to: it names each attention layer's
    /// history domain and the layer whose regions it binds.
    store: Rc<StateStore>,
    domain: ResourceDomainId,
}

impl TargetLaunchCore {
    pub fn boundary(&self) -> Option<&FeatureRef> {
        self.boundary.as_ref()
    }

    pub fn rows(&self) -> &LaunchRows {
        &self.rows
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

    /// Slot `slot`'s retained-row stores, which its readouts name by index:
    /// those of the sequence it was forked from, then its own.
    pub fn stores(&self, slot: usize) -> &[FeatureRef] {
        self.stores.get(slot).map_or(&[], Vec::as_slice)
    }

    pub fn into_parts(
        self,
    ) -> (
        LaunchRows,
        Vec<TentativeAdvance>,
        Vec<Option<ConditioningRef>>,
        Vec<Vec<ConditioningSlice>>,
    ) {
        (
            self.rows,
            self.advances,
            self.conditioning,
            self.conditioning_slices,
        )
    }
}
