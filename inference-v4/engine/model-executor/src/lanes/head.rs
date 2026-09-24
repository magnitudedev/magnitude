//! Sole join of head rows, tentative state, conditioning, and pooled memory.

use crate::{
    FeatureSpan, InvariantError, NativeGraphOutputLease, NativeGraphWorkspaceLease, PoolClass,
    ResourceDomainId,
};
use magnitude_model_batching::ValidatedHeadBatch;
use magnitude_model_state::{OwnedStateAdvance, StateStore};
use std::rc::Rc;

pub struct HeadLaunchInputs {
    batch: ValidatedHeadBatch,
    advances: Vec<OwnedStateAdvance>,
    conditioning: Vec<FeatureSpan>,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: Option<NativeGraphOutputLease>,
}

impl HeadLaunchInputs {
    pub fn new(
        batch: ValidatedHeadBatch,
        advances: Vec<OwnedStateAdvance>,
        conditioning: Vec<FeatureSpan>,
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
        Vec<FeatureSpan>,
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
        width: usize,
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
        for (index, ((slot, advance), span)) in self
            .batch
            .slots()
            .zip(&self.advances)
            .zip(&self.conditioning)
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
            for (row, packed) in slot.destinations().iter().enumerate() {
                let expected = binding
                    .destinations
                    .get(row)
                    .map(|destination| i32::try_from(*destination).ok())
                    .unwrap_or(Some(-1));
                if expected != Some(*packed) {
                    return Err(invalid(format!(
                        "slot {index} row {row} destination differs"
                    )));
                }
            }
            let visible = advance
                .history_ranges()
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
            if slot.visible().iter().any(|row| {
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
            let allocation = span.features.allocation();
            if span.features.domain() != domain
                || allocation.width() != width
                || span.count != slot.rows()
                || span
                    .start
                    .checked_add(span.count)
                    .is_none_or(|end| end > allocation.rows())
                || allocation.tensor().is_err()
            {
                return Err(invalid(format!(
                    "slot {index} conditioning differs from its physical rows"
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
        width: usize,
    ) -> Result<Self, (HeadLaunchInputs, InvariantError)> {
        if let Err(error) = inputs.validate(store, domain, width) {
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
    conditioning: Vec<FeatureSpan>,
    domain: ResourceDomainId,
}

impl HeadLaunchCore {
    pub fn batch(&self) -> &ValidatedHeadBatch {
        &self.batch
    }
    pub fn advances(&self) -> &[OwnedStateAdvance] {
        &self.advances
    }
    pub fn conditioning(&self) -> &[FeatureSpan] {
        &self.conditioning
    }
    pub fn into_parts(self) -> (ValidatedHeadBatch, Vec<OwnedStateAdvance>, Vec<FeatureSpan>) {
        (self.batch, self.advances, self.conditioning)
    }
}
