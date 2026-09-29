//! State maintenance owns physical state until a completed submission is reconciled.

use crate::{InvariantError, NativeGraphWorkspaceLease, ResourceDomainId};
use magnitude_model_batching::{StateBatchKind, ValidatedStateBatch};
use magnitude_model_state::{OwnedCodecAdvance, OwnedCompaction, StateStore};
use std::rc::Rc;

pub enum StateWork {
    Copy(OwnedCompaction),
    CodecConversion(OwnedCodecAdvance),
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

    fn validate(
        &self,
        source_store: &Rc<StateStore>,
        destination_store: Option<&Rc<StateStore>>,
        domain: &ResourceDomainId,
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
    ) -> Result<Self, (StateLaunchInputs, InvariantError)> {
        if let Err(error) = inputs.validate(source_store, destination_store, domain) {
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
