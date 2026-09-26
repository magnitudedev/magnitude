//! Closed, device-free numerical worker protocol.
//!
//! Concrete executor types, device leases, and caller-provided closures cannot
//! be represented here. This is the complete transport surface used by the
//! non-generic root facade.

use crate::publication::PublicationReceiver;
use crate::{
    owner::{AdmissionError, Status},
    retention::RetentionRequest,
};
use magnitude_generation::GenerationSeed;
use magnitude_model_contracts::PreparedModelInput;
use magnitude_model_executor::RequestId;

pub struct AdmitRequest {
    pub seed: GenerationSeed,
    pub input: PreparedModelInput,
    pub retention: Option<RetentionRequest>,
    pub output_capacity: usize,
}

pub enum WorkerCommand {
    Check,
    Admit(AdmitRequest),
    Stop { request: RequestId },
    Cancel { request: RequestId },
    Status { request: RequestId },
    Capacity,
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityStatus {
    pub active: usize,
    pub limit: usize,
}

pub enum WorkerReply {
    Admitted {
        request: RequestId,
        receiver: PublicationReceiver,
    },
    AdmissionRefused(AdmissionError),
    Status(Status),
    Capacity(CapacityStatus),
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
