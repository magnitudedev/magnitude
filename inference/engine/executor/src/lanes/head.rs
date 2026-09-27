//! Sole join of head rows, tentative state, conditioning, and pooled memory.

use crate::{
    FeatureRows, InvariantError, NativeGraphOutputLease, NativeGraphWorkspaceLease, PoolClass,
    ResourceDomainId,
};
use magnitude_batching::{HeadPasses, ValidatedHeadBatch};
use magnitude_state::{OwnedStateAdvance, StateStore};
use std::rc::Rc;

pub struct HeadLaunchInputs {
    batch: ValidatedHeadBatch,
    advances: Vec<OwnedStateAdvance>,
    /// Per slot, the host rows conditioning its entry rows in row order.
    conditioning: Vec<FeatureRows>,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: Option<NativeGraphOutputLease>,
}

impl HeadLaunchInputs {
    pub fn new(
        batch: ValidatedHeadBatch,
        advances: Vec<OwnedStateAdvance>,
        conditioning: Vec<FeatureRows>,
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            batch,
            advances,
            conditioning,
            graph_workspace,
            graph_output: Some(graph_output),
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        ValidatedHeadBatch,
        Vec<OwnedStateAdvance>,
        Vec<FeatureRows>,
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
        if self.batch.actual_slots() != self.advances.len()
            || self.batch.actual_slots() != self.conditioning.len()
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
        for (index, ((slot, advance), rows)) in self
            .batch
            .slots()
            .zip(&self.advances)
            .zip(&self.conditioning)
            .enumerate()
        {
            let binding = advance.bindings();
            if i32::try_from(binding.previous_bank).ok() != Some(slot.bank())
                || i32::try_from(binding.following_bank).ok() != Some(slot.following_bank())
            {
                return Err(invalid(format!("slot {index} uses another state bank")));
            }
            if rows.rows() != slot.rows() || rows.row_bytes() != row_bytes {
                return Err(invalid(format!(
                    "slot {index} conditioning has {} rows of {} bytes for {} entry rows",
                    rows.rows(),
                    rows.row_bytes(),
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
    pub fn new(
        inputs: HeadLaunchInputs,
        store: &Rc<StateStore>,
        domain: &ResourceDomainId,
        row_bytes: usize,
    ) -> Result<Self, (HeadLaunchInputs, InvariantError)> {
        if let Err(error) = inputs.validate(store, domain, row_bytes) {
            return Err((inputs, error));
        }
        let HeadLaunchInputs {
            batch,
            advances,
            conditioning,
            graph_workspace,
            graph_output,
        } = inputs;
        Ok(Self {
            core: HeadLaunchCore {
                batch,
                advances,
                conditioning,
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
    advances: Vec<OwnedStateAdvance>,
    conditioning: Vec<FeatureRows>,
    domain: ResourceDomainId,
}

impl HeadLaunchCore {
    pub fn batch(&self) -> &ValidatedHeadBatch {
        &self.batch
    }
    pub fn advances(&self) -> &[OwnedStateAdvance] {
        &self.advances
    }
    pub fn conditioning(&self) -> &[FeatureRows] {
        &self.conditioning
    }
    pub fn into_parts(self) -> (ValidatedHeadBatch, Vec<OwnedStateAdvance>, Vec<FeatureRows>) {
        (self.batch, self.advances, self.conditioning)
    }
}
