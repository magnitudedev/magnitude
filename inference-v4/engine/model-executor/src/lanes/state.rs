//! State maintenance owns physical state until a completed submission is reconciled.

use crate::{
    ConditioningRef, ConditioningSlice, InvariantError, NativeGraphWorkspaceLease,
    ResourceDomainId, TargetGraphOutputLease, TargetGraphWorkspaceLease,
};
use magnitude_model_batching::{StateBatchKind, ValidatedStateBatch};
use magnitude_model_state::{OwnedCodecAdvance, OwnedCompaction, OwnedRepairAdvance, StateStore};
use std::rc::Rc;

pub enum StateWork {
    Copy(OwnedCompaction),
    CodecConversion(OwnedCodecAdvance),
    RecurrentRepair {
        advance: OwnedRepairAdvance,
        conditioning: Option<ConditioningRef>,
        conditioning_slices: Vec<ConditioningSlice>,
        graph_workspace: TargetGraphWorkspaceLease,
        graph_outputs: [TargetGraphOutputLease; 2],
    },
}

pub struct StateLaunchInputs {
    batch: ValidatedStateBatch,
    work: StateWork,
    graph_workspace: NativeGraphWorkspaceLease,
}

impl StateLaunchInputs {
    pub fn new(
        batch: ValidatedStateBatch,
        work: StateWork,
        graph_workspace: NativeGraphWorkspaceLease,
    ) -> Self {
        Self {
            batch,
            work,
            graph_workspace,
        }
    }

    pub(crate) fn into_parts(self) -> (ValidatedStateBatch, StateWork, NativeGraphWorkspaceLease) {
        (self.batch, self.work, self.graph_workspace)
    }

    fn validate(
        &self,
        source_store: &Rc<StateStore>,
        destination_store: Option<&Rc<StateStore>>,
        domain: &ResourceDomainId,
        width: usize,
    ) -> Result<(), InvariantError> {
        let invalid = |detail: String| InvariantError {
            context: "state launch",
            detail,
        };
        if self.graph_workspace.domain() != domain {
            return Err(invalid(
                "state lease differs from the planned domain/class".into(),
            ));
        }
        match (&self.work, self.batch.kind()) {
            (StateWork::Copy(work), StateBatchKind::Copy) => {
                if !work.belongs_to(source_store)
                    || work.rows() != self.batch.actual_rows()
                    || self.batch.copies() != Some(work.copies())
                {
                    return Err(invalid(
                        "copy controls differ from the owned compaction".into(),
                    ));
                }
            }
            (
                StateWork::CodecConversion(work),
                StateBatchKind::CodecConversion {
                    source,
                    destination,
                },
            ) => {
                if !work.source_belongs_to(source_store)
                    || !destination_store.is_some_and(|store| work.destination_belongs_to(store))
                    || work.source_codec() != source
                    || work.destination_codec() != destination
                    || work.rows() != self.batch.actual_rows()
                    || self.batch.conversions() != Some(work.conversions())
                {
                    return Err(invalid(
                        "codec controls differ from the owned source/destination transaction"
                            .into(),
                    ));
                }
            }
            (
                StateWork::RecurrentRepair {
                    advance,
                    conditioning,
                    conditioning_slices,
                    graph_workspace,
                    graph_outputs,
                },
                StateBatchKind::RecurrentRepair,
            ) => {
                let replay = self
                    .batch
                    .replay()
                    .ok_or_else(|| invalid("repair has no replay rows".into()))?;
                if !advance.belongs_to(source_store)
                    || advance.rows() != replay.actual_rows()
                    || replay.slot(0).is_none_or(|slot| {
                        i32::try_from(advance.previous_bank()).ok() != Some(slot.bank())
                            || i32::try_from(advance.following_bank()).ok()
                                != Some(slot.following_bank())
                    })
                {
                    return Err(invalid(
                        "repair rows differ from the owned recurrent transaction".into(),
                    ));
                }
                if graph_workspace.domain() != domain
                    || graph_outputs.iter().any(|lease| lease.domain() != domain)
                {
                    return Err(invalid(
                        "repair graph leases differ from the resource domain".into(),
                    ));
                }
                if let Some(lease) = conditioning {
                    if lease.domain() != domain
                        || lease.allocation().rows() != replay.actual_rows()
                        || lease.allocation().width() != width
                        || lease.allocation().tensor().is_err()
                    {
                        return Err(invalid(
                            "repair conditioning differs from the physical replay rows".into(),
                        ));
                    }
                }
                let mut occupied = vec![false; replay.actual_rows()];
                for slice in conditioning_slices {
                    let source = slice.source.features.allocation();
                    let end = slice.destination.checked_add(slice.source.count);
                    if slice.source.features.domain() != domain
                        || source.width() != width
                        || source.tensor().is_err()
                        || slice.source.count == 0
                        || slice
                            .source
                            .start
                            .checked_add(slice.source.count)
                            .is_none_or(|end| end > source.rows())
                        || end.is_none_or(|end| end > replay.actual_rows())
                    {
                        return Err(invalid(
                            "repair feature slice differs from the accepted replay rows".into(),
                        ));
                    }
                    let Some(end) = end else {
                        return Err(invalid("repair feature destination overflows".into()));
                    };
                    if occupied[slice.destination..end].iter().any(|set| *set) {
                        return Err(invalid("repair feature slices overlap".into()));
                    }
                    occupied[slice.destination..end].fill(true);
                }
            }
            _ => {
                return Err(invalid(
                    "state batch and owned maintenance operation differ".into(),
                ));
            }
        }
        Ok(())
    }
}

pub struct ValidatedStateLaunch {
    core: StateLaunchCore,
    graph_workspace: NativeGraphWorkspaceLease,
}

impl ValidatedStateLaunch {
    pub fn new(
        inputs: StateLaunchInputs,
        source_store: &Rc<StateStore>,
        destination_store: Option<&Rc<StateStore>>,
        domain: &ResourceDomainId,
        width: usize,
    ) -> Result<Self, (StateLaunchInputs, InvariantError)> {
        if let Err(error) = inputs.validate(source_store, destination_store, domain, width) {
            return Err((inputs, error));
        }
        let StateLaunchInputs {
            batch,
            work,
            graph_workspace,
        } = inputs;
        Ok(Self {
            core: StateLaunchCore {
                batch,
                work,
                domain: domain.clone(),
            },
            graph_workspace,
        })
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }
    pub(crate) fn execution_parts_mut(
        &mut self,
    ) -> (
        &ValidatedStateBatch,
        &mut StateWork,
        &mut NativeGraphWorkspaceLease,
    ) {
        (
            &self.core.batch,
            &mut self.core.work,
            &mut self.graph_workspace,
        )
    }
    pub(crate) fn into_submission_parts(self) -> (StateLaunchCore, NativeGraphWorkspaceLease) {
        (self.core, self.graph_workspace)
    }
}

pub struct StateLaunchCore {
    batch: ValidatedStateBatch,
    work: StateWork,
    domain: ResourceDomainId,
}

impl StateLaunchCore {
    pub fn batch(&self) -> &ValidatedStateBatch {
        &self.batch
    }
    pub fn work(&self) -> &StateWork {
        &self.work
    }
    pub fn into_parts(self) -> (ValidatedStateBatch, StateWork) {
        (self.batch, self.work)
    }
}
