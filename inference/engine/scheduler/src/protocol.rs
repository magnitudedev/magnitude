//! Closed, device-free command vocabulary of the thread-confined execution
//! owner inside a numerical worker.
//!
//! Concrete executor types, device leases, and caller-provided closures cannot
//! be represented here. The engine's worker session is the only dispatcher;
//! the host/worker IPC protocol is defined above this crate.

use crate::prefix_cache::PrefixRetention;
use crate::publication::PublicationReceiver;
use crate::serving::ServingCensus;
use crate::owner::{AdmissionError, Status};
use magnitude_executor::platform::DomainReading;
use magnitude_executor::{MemoryChargeReconciliation, RequestId};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::GenerationSeed;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

pub struct AdmitRequest {
    pub seed: GenerationSeed,
    pub input: PreparedModelInput,
    /// Whether the request resumes from and contributes to the prefix
    /// cache, and where it retains states for later requests.
    pub retention: PrefixRetention,
    pub output_capacity: usize,
}

pub enum WorkerCommand {
    Admit(AdmitRequest),
    Stop { request: RequestId },
    Status { request: RequestId },
    /// The memory heap's classified holdings and standing.
    Observe,
    /// Claim `bytes` of device memory beside the model for the service's
    /// measurement job, in place of any claim before: committed on the
    /// device's ledger, which cache yields to at once, and held until
    /// released.
    ClaimMeasurementRoom { bytes: u64 },
    /// The measurement job ended: its room is free again.
    ReleaseMeasurementRoom,
    /// The measurement job stored new records: price the experts again
    /// from what is stored.
    Reprice,
}

/// The answer to a measurement room's claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoomAnswer {
    /// The room is free beside the model.
    Made,
    /// The room cannot be made, for this reason.
    Refused(RoomRefusal),
    /// The claim was released before it was answered: by its holder, by a
    /// claim in its place, or by the owner ending.
    Released,
}

/// Why a measurement room cannot be made (the ledger's refusal,
/// `memory::ledger::Short`). Cache never refuses it: it yields to the claim
/// at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoomRefusal {
    /// With the model's committed memory, it exceeds what the process may
    /// hold by `bytes`.
    Short { bytes: u64 },
    /// The device's memory reading failed: no figure of it makes room.
    Blind,
}

impl fmt::Display for RoomRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Short { bytes } => write!(f, "the model's committed memory leaves the measurement's room {bytes} bytes short"),
            Self::Blind => f.write_str("the device's memory reading failed, so no room can be made"),
        }
    }
}

/// The owner's answer to a measurement room's claim, which a job waits on.
/// The owner answers once; a claim it drops unanswered is
/// [`RoomAnswer::Released`].
pub struct RoomGrant(Arc<Mutex<GrantState>>);

#[derive(Default)]
struct GrantState {
    answer: Option<RoomAnswer>,
    waker: Option<Waker>,
}

/// The owner's side of a [`RoomGrant`]: its answer, until given (`None`
/// once it was), so it is given once.
pub(crate) struct RoomGranter(Option<Arc<Mutex<GrantState>>>);

impl RoomGrant {
    pub(crate) fn new() -> (RoomGranter, Self) {
        let state = Arc::new(Mutex::new(GrantState::default()));
        (RoomGranter(Some(state.clone())), Self(state))
    }

    /// The grant of a claim no owner holds: released at once.
    pub fn released() -> Self {
        Self::new().1
    }

    pub fn poll_answer(&mut self, context: &mut Context<'_>) -> Poll<RoomAnswer> {
        let mut state = self.0.lock().unwrap();
        match state.answer.take() {
            Some(answer) => Poll::Ready(answer),
            None => {
                state.waker = Some(context.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl RoomGranter {
    /// Answer the claim: made, or refused for why.
    pub(crate) fn answer(mut self, decided: Result<(), RoomRefusal>) {
        self.fill(match decided {
            Ok(()) => RoomAnswer::Made,
            Err(refusal) => RoomAnswer::Refused(refusal),
        });
    }

    /// Give the answer, unless it was given.
    fn fill(&mut self, value: RoomAnswer) {
        let Some(state) = self.0.take() else {
            return;
        };
        let mut state = state.lock().unwrap();
        state.answer = Some(value);
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }
}

impl Drop for RoomGranter {
    /// A claim dropped unanswered is released.
    fn drop(&mut self) {
        self.fill(RoomAnswer::Released);
    }
}

/// A live request's scheduling status and progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestSnapshot {
    pub status: Status,
    pub prompt_tokens: usize,
    /// Leading prompt tokens restored from a retained prefix, not computed.
    pub cached_tokens: usize,
    pub resident_position: usize,
    pub output_tokens: usize,
}

pub enum WorkerReply {
    Admitted {
        request: RequestId,
        /// Its stream, which reads it begun once it becomes resident
        /// (`PublicationReceiver::began`): until then it may yet be refused.
        receiver: PublicationReceiver,
    },
    AdmissionRefused(AdmissionError),
    /// `None` when the request is not live.
    Status(Option<RequestSnapshot>),
    Observed {
        /// Seismic's charge classified by every holder the owner keeps.
        reconciliation: MemoryChargeReconciliation,
        /// A fresh reading of every memory domain the device uses.
        readings: Vec<DomainReading>,
        serving: ServingCensus,
    },
    /// A measurement room's claim is held; its answer follows.
    MeasurementRoom(RoomGrant),
    Acknowledged,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn protocol_is_device_free_and_sendable() {
        assert_send::<WorkerCommand>();
        assert_send::<WorkerReply>();
    }

    /// A claim answers once: an answered claim's granter, dropped, releases
    /// nothing, so the answer taken is the last; one dropped unanswered is
    /// released.
    #[test]
    fn an_answered_claim_is_not_released_after_its_answer() {
        let poll = |grant: &mut RoomGrant| grant.poll_answer(&mut Context::from_waker(Waker::noop()));
        let (granter, mut grant) = RoomGrant::new();
        granter.answer(Ok(()));
        assert_eq!(poll(&mut grant), Poll::Ready(RoomAnswer::Made));
        assert_eq!(poll(&mut grant), Poll::Pending, "no release after the answer");
        let (granter, mut grant) = RoomGrant::new();
        drop(granter);
        assert_eq!(poll(&mut grant), Poll::Ready(RoomAnswer::Released));
        assert_eq!(poll(&mut RoomGrant::released()), Poll::Ready(RoomAnswer::Released));
    }
}
