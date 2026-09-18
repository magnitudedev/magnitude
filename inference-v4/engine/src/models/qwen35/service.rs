//! Service adapter for synchronous Qwen generation. All forwards use automatic
//! Seismic selection. Batches share row-independent stages while preserving
//! private sequence successors and independent acceptance.
use super::decoder::{Decoder, GenerationWork};
use crate::{
    models::sequence::OwnedSequence,
    service::{
        owner::{Completion, Executor, PrepareError, Submitted, Work},
        policy::RequestId,
        worker::CompletionWake,
    },
};
use std::collections::{BTreeMap, BTreeSet};

pub struct QwenExecutor {
    decoder: Decoder,
    sequences: BTreeMap<RequestId, Sequence>,
}
pub struct QwenCheckpoint {
    store: std::rc::Rc<crate::state::StateStore>,
    state: Option<crate::state::StateCheckpoint>,
}
enum Sequence {
    Fresh,
    Resident(OwnedSequence),
    Evicted,
}
fn preparation_error(error: seismic_runtime::Error) -> PrepareError {
    match error {
        seismic_runtime::Error::Capacity {
            required,
            available,
        } => PrepareError::Capacity {
            required: required as u64,
            available: available as u64,
        },
        seismic_runtime::Error::Failure(message) => PrepareError::Fatal(message),
    }
}
impl QwenExecutor {
    /// The host sets the device storage budget before importing/compiling the
    /// model. Every allocation retained by this service shares that domain.
    pub fn new(decoder: Decoder) -> Result<Self, String> {
        if decoder.memory_usage().limit.is_none() {
            return Err("Qwen service requires an explicit device storage budget".into());
        }
        Ok(Self {
            decoder,
            sequences: BTreeMap::new(),
        })
    }
    fn selected(&self, requests: &[RequestId]) -> Result<Vec<&OwnedSequence>, String> {
        let mut seen = BTreeSet::new();
        let mut selected = Vec::new();
        for request in requests {
            if !seen.insert(*request) {
                return Err("duplicate eviction request".into());
            }
            match self.sequences.get(request) {
                Some(Sequence::Resident(sequence)) => selected.push(sequence),
                Some(Sequence::Fresh) => {}
                _ => return Err("eviction requires a resident request".into()),
            }
        }
        Ok(selected)
    }
}
struct Completed;
impl Completion for Completed {
    fn is_complete(&self) -> bool {
        true
    }
    fn result(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn notify(&mut self, wake: CompletionWake) {
        wake.complete();
    }
}
impl Executor for QwenExecutor {
    type Checkpoint = QwenCheckpoint;
    fn checkpoint(
        &self,
        request: RequestId,
    ) -> Result<crate::service::owner::NumericalCheckpoint<Self::Checkpoint>, String> {
        let state = match self.sequences.get(&request) {
            Some(Sequence::Resident(sequence)) => Some(sequence.checkpoint()?),
            Some(Sequence::Fresh) => None,
            _ => return Err("checkpoint requires resident numerical state".into()),
        };
        Ok(crate::service::owner::NumericalCheckpoint {
            position: state.as_ref().map_or(0, |state| state.position()),
            state: QwenCheckpoint {
                store: self.decoder.state_store().clone(),
                state,
            },
        })
    }
    fn open_checkpoint(
        &mut self,
        request: RequestId,
        checkpoint: &Self::Checkpoint,
    ) -> Result<(), String> {
        if self.sequences.contains_key(&request) {
            return Err("duplicate numerical request".into());
        }
        if !std::rc::Rc::ptr_eq(&checkpoint.store, self.decoder.state_store()) {
            return Err("checkpoint belongs to another decoder".into());
        }
        let sequence = match &checkpoint.state {
            Some(state) => Sequence::Resident(OwnedSequence::new(state.fork())),
            None => Sequence::Fresh,
        };
        self.sequences.insert(request, sequence);
        Ok(())
    }
    fn open(&mut self, request: RequestId) -> Result<(), String> {
        if self.sequences.contains_key(&request) {
            return Err("duplicate numerical request".into());
        }
        // Numerical allocations belong to negotiated preparation, so admission
        // itself cannot bypass the scheduler's capacity recovery.
        self.sequences.insert(request, Sequence::Fresh);
        Ok(())
    }
    fn prepare(&mut self, work: &[Work]) -> Result<Submitted, PrepareError> {
        let mut seen = BTreeSet::new();
        let mut restored = BTreeMap::new();
        // Validate every member before any execution. Restoration stays private
        // until every row has prepared successfully.
        for row in work {
            let fail = |message: &str| PrepareError::Request {
                request: row.request,
                message: message.into(),
            };
            if !seen.insert(row.request) {
                return Err(fail("duplicate numerical batch member"));
            }
            if row.proposal.tokens().is_empty()
                || row
                    .proposal
                    .position()
                    .checked_add(row.proposal.tokens().len())
                    .is_none_or(|end| end > self.decoder.context_capacity())
                || row
                    .proposal
                    .tokens()
                    .iter()
                    .any(|token| u64::from(token.0) >= self.decoder.geometry().vocabulary)
                || row.mask.as_ref().is_some_and(|mask| {
                    mask.len() != (self.decoder.geometry().vocabulary as usize).div_ceil(32)
                })
            {
                return Err(fail("proposal exceeds decoder context or vocabulary"));
            }
            let sequence = self
                .sequences
                .get(&row.request)
                .ok_or_else(|| fail("unknown numerical request"))?;
            if row.restore != matches!(sequence, Sequence::Evicted) {
                return Err(fail(
                    "numerical residency disagrees with proposed restoration",
                ));
            }
            if let Sequence::Resident(sequence) = sequence {
                if sequence.pending() || sequence.position() != row.proposal.position() {
                    return Err(fail("unresolved or stale numerical proposal"));
                }
            } else {
                if row.proposal.position() != 0 {
                    return Err(fail("restoration must replay from the beginning"));
                }
                let state = self
                    .decoder
                    .state_store()
                    .create()
                    .map_err(preparation_error)?;
                restored.insert(row.request, OwnedSequence::new(state));
            }
        }
        let prepared = work
            .iter()
            .map(|row| {
                let sequence = restored
                    .get(&row.request)
                    .or_else(|| match self.sequences.get(&row.request) {
                        Some(Sequence::Resident(sequence)) => Some(sequence),
                        _ => None,
                    })
                    .expect("validated resident or restored sequence");
                GenerationWork {
                    sequence,
                    proposal: &row.proposal,
                    mask: row.mask.as_deref(),
                }
            })
            .collect::<Vec<_>>();
        let rows = self
            .decoder
            .prepare_generation_batch(&prepared)
            .map_err(preparation_error)?;
        for (request, sequence) in restored {
            self.sequences.insert(request, Sequence::Resident(sequence));
        }
        Ok(Submitted {
            completion: Box::new(Completed),
            rows,
        })
    }
    fn preparation_identity(&self, work: &[Work]) -> Result<String, String> {
        // Retain full inputs because content-conditioned specialization may
        // change physical requirements even for identical token extents.
        Ok(work
            .iter()
            .map(|row| {
                format!(
                    "{:?}:{}:{}:{:?}:{:?}:{}:{}:{:?}:{}",
                    row.request,
                    row.restore,
                    row.proposal.position(),
                    row.proposal.tokens(),
                    row.proposal.sampling(),
                    row.proposal.seed(),
                    row.proposal.sample_position(),
                    row.mask,
                    row.proposal.needs_sample(),
                )
            })
            .collect::<Vec<_>>()
            .join(";"))
    }
    fn reclaim_idle(&mut self) -> Result<u64, String> {
        Ok(self.decoder.reclaim_idle()? as u64)
    }
    fn reclaimable(&self, requests: &[RequestId]) -> Result<u64, String> {
        Ok(
            OwnedSequence::reclaimable(self.decoder.state_store(), &self.selected(requests)?)?
                as u64,
        )
    }
    fn evict(&mut self, requests: &[RequestId]) -> Result<u64, String> {
        let bytes = self.reclaimable(requests)?;
        if bytes != 0 {
            let before = self.decoder.memory_usage().charged;
            // Owner-confined, synchronous execution: no new alias can appear
            // between pricing and dropping these selected state owners.
            for request in requests {
                *self.sequences.get_mut(request).expect("validated request") = Sequence::Evicted;
            }
            let released = (before - self.decoder.memory_usage().charged) as u64;
            debug_assert_eq!(bytes, released);
            return Ok(released);
        }
        Ok(bytes)
    }
    fn close(&mut self, request: RequestId) -> Result<(), String> {
        self.sequences.remove(&request);
        Ok(())
    }
}
