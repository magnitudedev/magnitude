//! Closed, device-free command vocabulary of the thread-confined execution
//! owner inside a numerical worker.
//!
//! Concrete executor types, device leases, and caller-provided closures cannot
//! be represented here. The engine's worker session is the only dispatcher;
//! the host/worker IPC protocol is defined above this crate.

use crate::publication::PublicationReceiver;
use crate::owner::{AdmissionError, Status};
use magnitude_executor::platform::DomainReading;
use magnitude_executor::{MemoryChargeReconciliation, RequestId};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::GenerationSeed;

pub struct AdmitRequest {
    pub seed: GenerationSeed,
    pub input: PreparedModelInput,
    /// Whether the request resumes from and contributes to the prefix cache.
    pub prefix_cache: bool,
    pub output_capacity: usize,
}

pub enum WorkerCommand {
    Admit(AdmitRequest),
    Stop { request: RequestId },
    Status { request: RequestId },
    /// The memory heap's classified holdings and standing.
    Observe,
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
    },
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
}
