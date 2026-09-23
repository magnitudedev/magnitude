//! The sole join of target row semantics, owned state, conditioning, and
//! planned device leases.

use crate::{
    ConditioningRef, FeatureSpan, InvariantError, NativeGraphOutputLease,
    NativeGraphWorkspaceLease, PoolClass, ResourceDomainId, TargetGraphOutputLease,
    TargetGraphWorkspaceLease,
};
use magnitude_model_batching::ValidatedTargetBatch;
use magnitude_model_state::{OwnedStateAdvance, StateStore};
use std::rc::Rc;

pub struct TargetLaunchInputs {
    batch: ValidatedTargetBatch,
    advances: Vec<OwnedStateAdvance>,
    conditioning: Vec<Option<ConditioningRef>>,
    conditioning_slices: Vec<Vec<ConditioningSlice>>,
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

impl TargetLaunchInputs {
    pub fn new(
        batch: ValidatedTargetBatch,
        advances: Vec<OwnedStateAdvance>,
        conditioning: Vec<Option<ConditioningRef>>,
        conditioning_slices: Vec<Vec<ConditioningSlice>>,
        graph_workspace: TargetGraphWorkspaceLease,
        graph_outputs: [TargetGraphOutputLease; 2],
        readout_workspace: NativeGraphWorkspaceLease,
        readout_output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            batch,
            advances,
            conditioning,
            conditioning_slices,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output: Some(readout_output),
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        ValidatedTargetBatch,
        Vec<OwnedStateAdvance>,
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
    ) -> Result<(), InvariantError> {
        let invalid = |detail: String| InvariantError {
            context: "target launch",
            detail,
        };
        let slots = self.batch.actual_slots();
        if slots != self.advances.len()
            || slots != self.conditioning.len()
            || slots != self.conditioning_slices.len()
        {
            return Err(invalid(
                "slot, advance, and conditioning counts disagree".into(),
            ));
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
            if !binding.destinations.is_empty()
                && binding.destinations.len() != slot.destinations().len()
            {
                return Err(invalid(format!("slot {index} destination count differs")));
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
    ) -> Result<Self, (TargetLaunchInputs, InvariantError)> {
        if let Err(error) = inputs.validate(store, domain, hidden_width) {
            return Err((inputs, error));
        }
        let TargetLaunchInputs {
            batch,
            advances,
            conditioning,
            conditioning_slices,
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        } = inputs;
        Ok(Self {
            core: TargetLaunchCore {
                batch,
                advances,
                conditioning,
                conditioning_slices,
                domain: domain.clone(),
            },
            graph_workspace,
            graph_outputs,
            readout_workspace,
            readout_output,
        })
    }

    pub fn class(&self) -> PoolClass {
        PoolClass::Target(self.core.batch.class())
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

    pub(crate) fn into_submission_parts(self) -> TargetLaunchCore {
        self.core
    }
}

/// Reconciliation payload retained by a submission after its workspace has
/// returned to the pool. It is unavailable before physical finish.
pub struct TargetLaunchCore {
    batch: ValidatedTargetBatch,
    advances: Vec<OwnedStateAdvance>,
    conditioning: Vec<Option<ConditioningRef>>,
    conditioning_slices: Vec<Vec<ConditioningSlice>>,
    domain: ResourceDomainId,
}

impl TargetLaunchCore {
    pub fn batch(&self) -> &ValidatedTargetBatch {
        &self.batch
    }

    pub fn advances(&self) -> &[OwnedStateAdvance] {
        &self.advances
    }

    pub fn conditioning(&self) -> &[Option<ConditioningRef>] {
        &self.conditioning
    }

    pub fn conditioning_slices(&self) -> &[Vec<ConditioningSlice>] {
        &self.conditioning_slices
    }

    pub fn into_parts(
        self,
    ) -> (
        ValidatedTargetBatch,
        Vec<OwnedStateAdvance>,
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
