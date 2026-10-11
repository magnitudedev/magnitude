//! Sole join of head rows, tentative state, conditioning, and pooled memory.

use crate::{
    FeatureRows, InvariantError, NativeGraphOutputLease, NativeGraphWorkspaceLease, PoolClass,
    ResourceDomainId,
};
use magnitude_batching::{HeadPasses, ValidatedHeadBatch};
use magnitude_state::{SequenceState, StateStore, TentativeAdvance};
use std::rc::Rc;

/// What conditions a head launch's entry rows.
pub enum HeadConditioning {
    /// Per slot, the host rows conditioning its entry rows in row order.
    Rows(Vec<FeatureRows>),
    /// A lone slot's entry rows, conditioned by the leading rows of a target
    /// feature output on the device (`[rows, hidden]` activation rows): a
    /// prompt chunk's entry drafted behind the chunk. Holding the output
    /// keeps its arena from recycling until the entry is drafted.
    Features {
        current: crate::GraphOutputTensor,
        previous: Option<crate::FeatureRef>,
    },
}

/// The accepted target history a launch of a head without a store reads:
/// each slot's accepted target state, held unchanged while the launch is in
/// flight, and per head block the target history it binds.
pub struct TargetReads {
    /// Per slot, the request's accepted target state at the anchor.
    pub states: Vec<SequenceState>,
    /// Per head block, in block order.
    pub history: Vec<TargetHistoryRead>,
}

/// The target history one head block's attention binds.
pub struct TargetHistoryRead {
    /// The target store's stored history domain the block reads: the index
    /// of its row tables in the launch's rows.
    pub domain: usize,
    /// The source layer's history planes, in plane-descriptor order.
    pub planes: Vec<seismic::Tensor>,
}

pub struct HeadLaunchInputs {
    batch: ValidatedHeadBatch,
    /// The tentative advances of the head's own store, one per slot; empty
    /// for a launch reading target history.
    advances: Vec<TentativeAdvance>,
    /// The target history a head without a store reads.
    reads: Option<TargetReads>,
    conditioning: HeadConditioning,
    /// Whether windowed history domains keep these entry rows: false for
    /// priming rows every window drops before the first draft reads it.
    windowed: bool,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: Option<NativeGraphOutputLease>,
}

impl HeadLaunchInputs {
    pub fn new(
        batch: ValidatedHeadBatch,
        advances: Vec<TentativeAdvance>,
        conditioning: HeadConditioning,
        windowed: bool,
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            batch,
            advances,
            reads: None,
            conditioning,
            windowed,
            graph_workspace,
            graph_output: Some(graph_output),
        }
    }

    /// A launch of a head without a store: its rows read `reads` and append
    /// nowhere.
    pub fn reading_target(
        batch: ValidatedHeadBatch,
        reads: TargetReads,
        conditioning: HeadConditioning,
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            batch,
            advances: Vec::new(),
            reads: Some(reads),
            conditioning,
            windowed: true,
            graph_workspace,
            graph_output: Some(graph_output),
        }
    }

    /// The target states a refused launch returns.
    pub(crate) fn into_target_reads(self) -> Option<TargetReads> {
        self.reads
    }

    /// Check a launch reading target history: one entry row per slot at its
    /// state's position, reading exactly the state's accepted rows in every
    /// stored domain and appending nowhere.
    fn validate_reads(
        &self,
        store: &Rc<StateStore>,
        domain: &ResourceDomainId,
        row_bytes: usize,
    ) -> Result<(), InvariantError> {
        let invalid = |detail: String| InvariantError {
            context: "head launch",
            detail,
        };
        let reads = self
            .reads
            .as_ref()
            .ok_or_else(|| invalid("a launch reading target history has none".into()))?;
        let HeadConditioning::Rows(conditioning) = &self.conditioning else {
            return Err(invalid(
                "a head reading target history enters host conditioning rows".into(),
            ));
        };
        if self.batch.passes() != HeadPasses::Chained
            || self.batch.steps() == 0
            || self.batch.actual_slots() != reads.states.len()
            || self.batch.actual_slots() != conditioning.len()
            || self.batch.actual_rows() != self.batch.actual_slots()
        {
            return Err(invalid(
                "slot, target state, and conditioning counts disagree".into(),
            ));
        }
        if self.graph_workspace.domain() != domain
            || self
                .graph_output
                .as_ref()
                .is_none_or(|output| output.domain() != domain)
        {
            return Err(invalid(
                "workspace or output lease differs from head domain/class".into(),
            ));
        }
        let domains = store.history_domains().collect::<Vec<_>>();
        if reads
            .history
            .iter()
            .any(|read| read.domain >= domains.len())
        {
            return Err(invalid("a head block reads no stored target domain".into()));
        }
        for (index, ((slot, state), rows)) in self
            .batch
            .slots()
            .zip(&reads.states)
            .zip(conditioning)
            .enumerate()
        {
            if !state.belongs_to(store) {
                return Err(invalid(format!("slot {index} reads another store")));
            }
            if slot.rows() != 1 || rows.rows() != 1 || rows.row_bytes() != row_bytes {
                return Err(invalid(format!("slot {index} is not one conditioned row")));
            }
            for &domain in &domains {
                let from = store
                    .history_domain_kind(domain)
                    .visible_from(state.position());
                let visible = state
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
                let history = slot
                    .history(domain.0)
                    .ok_or_else(|| invalid(format!("slot {index} lacks a history domain")))?;
                if history
                    .destinations
                    .iter()
                    .any(|destination| *destination != -1)
                    || self
                        .batch
                        .chain_destinations(index, domain.0)
                        .iter()
                        .any(|destination| *destination != -1)
                {
                    return Err(invalid(format!("slot {index} appends target history")));
                }
                if history.visible.iter().any(|row| {
                    row.len() < visible.len()
                        || row[..visible.len()] != visible[..]
                        || row[visible.len()..]
                            .iter()
                            .any(|padding| *padding != [0, 0])
                }) {
                    return Err(invalid(format!(
                        "slot {index} visibility differs from accepted target state"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        ValidatedHeadBatch,
        Vec<TentativeAdvance>,
        HeadConditioning,
        NativeGraphWorkspaceLease,
        Option<NativeGraphOutputLease>,
    ) {
        (
            self.batch,
            self.advances,
            self.conditioning,
            self.graph_workspace,
            self.graph_output,
        )
    }

    fn validate(
        &self,
        store: &Rc<StateStore>,
        domain: &ResourceDomainId,
        row_bytes: usize,
    ) -> Result<(), InvariantError> {
        let invalid = |detail: String| InvariantError {
            context: "head launch",
            detail,
        };
        let conditioned = match &self.conditioning {
            HeadConditioning::Rows(rows) => rows.len(),
            HeadConditioning::Features { .. } => 1,
        };
        if self.batch.actual_slots() != self.advances.len()
            || self.batch.actual_slots() != conditioned
        {
            return Err(invalid(
                "slot, advance, and conditioning counts disagree".into(),
            ));
        }
        if self.graph_workspace.domain() != domain
            || self
                .graph_output
                .as_ref()
                .is_none_or(|output| output.domain() != domain)
        {
            return Err(invalid(
                "workspace or output lease differs from head domain/class".into(),
            ));
        }
        for (index, (slot, advance)) in self.batch.slots().zip(&self.advances).enumerate() {
            let binding = advance.bindings();
            if i32::try_from(binding.previous_bank).ok() != Some(slot.bank())
                || i32::try_from(binding.following_bank).ok() != Some(slot.following_bank())
            {
                return Err(invalid(format!("slot {index} uses another state bank")));
            }
            // Bound features' width is checked where the graph binds them.
            let (rows, fits) = match &self.conditioning {
                HeadConditioning::Rows(rows) => (
                    rows[index].rows(),
                    rows[index].rows() == slot.rows() && rows[index].row_bytes() == row_bytes,
                ),
                HeadConditioning::Features { current, previous } => {
                    if let Some(previous) = previous {
                        if self.batch.passes() == HeadPasses::Block
                            || previous.domain() != domain
                            || previous.allocation().rows() != 1
                            || previous.allocation().is_graph_backed()
                            || previous.allocation().tensor().is_err()
                        {
                            return Err(invalid("invalid prefix conditioning feature".into()));
                        }
                    }
                    match current.tensor().extents() {
                        [rows, _] => {
                            let rows = usize::try_from(*rows).unwrap_or(0);
                            (rows, rows + usize::from(previous.is_some()) >= slot.rows())
                        }
                        _ => (0, false),
                    }
                }
            };
            if !fits {
                return Err(invalid(format!(
                    "slot {index} conditioning has {rows} rows for {} entry rows",
                    slot.rows()
                )));
            }
            if self.batch.passes() == HeadPasses::Block {
                // A separate draft's entry rows each inject into every
                // history domain of its store at their own destination,
                // attending nothing else; its block rows append nowhere.
                if !advance.belongs_to(store) || slot.rows() != advance.rows() {
                    return Err(invalid(format!(
                        "slot {index} differs from its state advance"
                    )));
                }
                for domain in store.history_domains() {
                    let history = slot
                        .history(domain.0)
                        .ok_or_else(|| invalid(format!("slot {index} lacks a history domain")))?;
                    let expected = binding.destinations[domain.0]
                        .iter()
                        .map(|destination| i32::try_from(*destination).ok())
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| invalid(format!("slot {index} destination exceeds i32")))?;
                    if history.destinations != expected.as_slice() {
                        return Err(invalid(format!("slot {index} destinations differ")));
                    }
                    if history.visible.iter().flatten().any(|span| *span != [0, 0]) {
                        return Err(invalid(format!(
                            "slot {index} injection rows attend history"
                        )));
                    }
                }
                continue;
            }
            // The draft head's store has one Token domain.
            let domain = store
                .sole_history_domain()
                .map_err(|error| invalid(error.to_string()))?;
            let history = slot
                .history(domain.0)
                .ok_or_else(|| invalid(format!("slot {index} lacks its history domain")))?;
            let chain = self.batch.chain_destinations(index, domain.0);
            let written = chain
                .iter()
                .filter(|destination| **destination >= 0)
                .count();
            if !advance.belongs_to(store) || slot.rows() + written != advance.rows() {
                return Err(invalid(format!(
                    "slot {index} differs from its state advance"
                )));
            }
            // Entry rows append first, then each chained row in step order;
            // a chained row past the request's own proposals appends nowhere.
            let expected = binding.destinations[domain.0]
                .iter()
                .map(|destination| i32::try_from(*destination).ok())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid(format!("slot {index} destination exceeds i32")))?;
            let packed = history
                .destinations
                .iter()
                .copied()
                .chain(
                    chain
                        .iter()
                        .copied()
                        .filter(|destination| *destination >= 0),
                )
                .collect::<Vec<_>>();
            if packed != expected {
                return Err(invalid(format!("slot {index} destinations differ")));
            }
            let visible = advance
                .history_ranges(domain)
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
            if history.visible.iter().any(|row| {
                row.len() < visible.len()
                    || &row[..visible.len()] != visible.as_slice()
                    || row[visible.len()..]
                        .iter()
                        .any(|padding| *padding != [0, 0])
            }) {
                return Err(invalid(format!(
                    "slot {index} visibility differs from accepted state"
                )));
            }
        }
        Ok(())
    }
}

pub struct ValidatedHeadLaunch {
    core: HeadLaunchCore,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: Option<NativeGraphOutputLease>,
}

impl ValidatedHeadLaunch {
    /// Validate a launch against the store its rows address: the head's own
    /// store, or the target's for a launch reading target history.
    pub fn new(
        inputs: HeadLaunchInputs,
        store: &Rc<StateStore>,
        domain: &ResourceDomainId,
        row_bytes: usize,
    ) -> Result<Self, (HeadLaunchInputs, InvariantError)> {
        let validated = if inputs.reads.is_some() {
            inputs.validate_reads(store, domain, row_bytes)
        } else {
            inputs.validate(store, domain, row_bytes)
        };
        if let Err(error) = validated {
            return Err((inputs, error));
        }
        let HeadLaunchInputs {
            batch,
            advances,
            reads,
            conditioning,
            windowed,
            graph_workspace,
            graph_output,
        } = inputs;
        Ok(Self {
            core: HeadLaunchCore {
                batch,
                advances,
                reads,
                conditioning,
                windowed,
                domain: domain.clone(),
            },
            graph_workspace,
            graph_output,
        })
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }
    pub fn class(&self) -> PoolClass {
        PoolClass::Head(self.core.batch.class())
    }
    pub(crate) fn execution_parts_mut(
        &mut self,
    ) -> (
        &HeadLaunchCore,
        &mut NativeGraphWorkspaceLease,
        &mut Option<NativeGraphOutputLease>,
    ) {
        (
            &self.core,
            &mut self.graph_workspace,
            &mut self.graph_output,
        )
    }
    pub(crate) fn into_submission_parts(
        self,
    ) -> (
        HeadLaunchCore,
        NativeGraphWorkspaceLease,
        Option<NativeGraphOutputLease>,
    ) {
        (self.core, self.graph_workspace, self.graph_output)
    }
}

pub struct HeadLaunchCore {
    batch: ValidatedHeadBatch,
    advances: Vec<TentativeAdvance>,
    reads: Option<TargetReads>,
    conditioning: HeadConditioning,
    windowed: bool,
    domain: ResourceDomainId,
}

impl HeadLaunchCore {
    pub fn batch(&self) -> &ValidatedHeadBatch {
        &self.batch
    }
    pub fn advances(&self) -> &[TentativeAdvance] {
        &self.advances
    }
    /// The target history a launch of a head without a store reads.
    pub fn target_reads(&self) -> Option<&TargetReads> {
        self.reads.as_ref()
    }
    /// The target states a finished or failed launch returns.
    pub fn into_target_reads(self) -> Option<TargetReads> {
        self.reads
    }
    pub fn conditioning(&self) -> &HeadConditioning {
        &self.conditioning
    }
    /// Whether windowed history domains keep the entry rows.
    pub fn windowed(&self) -> bool {
        self.windowed
    }
    pub fn into_parts(self) -> (ValidatedHeadBatch, Vec<TentativeAdvance>, HeadConditioning) {
        (self.batch, self.advances, self.conditioning)
    }
}
