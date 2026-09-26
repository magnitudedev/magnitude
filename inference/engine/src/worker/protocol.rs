//! The private host/worker protocol (integration spec §8.1).
//!
//! Only immutable, device-free values cross: the execution manifest, prepared
//! numerical input, generation options and the constraint *description*
//! (`ConstraintPlan`); the worker instantiates the matcher with its own
//! vocabulary. Output flows back under per-request credit: the host grants one
//! batch of credit per output batch its consumer drains, and the worker drains
//! the engine's bounded publication queue only while it holds credit, so a slow
//! consumer backs up to the engine's publication bound.
//!
//! The encoding (serde) is versioned with the engine build: both ends
//! exchange [`EngineBuild`] first and refuse a mismatch.

use crate::census::{AllocationCensus, MemoryDomain};
use crate::error::{LoadError, RequestError, UnloadCause};
use crate::options::{ExecutionManifest, ReadyInfo};
use magnitude_chat::ConstraintPlan;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{DetailedUsage, FinishReason, Options, OutputToken};
use serde::{Deserialize, Serialize};
use std::fmt;

/// The engine build identity: the package version and a digest of the engine
/// sources. Two processes speak the protocol only with equal builds.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EngineBuild(pub String);

impl EngineBuild {
    pub fn current() -> Self {
        Self(env!("MAGNITUDE_ENGINE_BUILD").to_owned())
    }
}

impl fmt::Display for EngineBuild {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Host-assigned request identity, unique for one worker connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HostRequestId(pub u64);

impl fmt::Display for HostRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Whether the request's prompt state is kept for later exact-prefix reuse
/// (`cache_prompt`). The worker derives the retention key from its own
/// package, tokenizer and codec identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionPolicy {
    Retain,
    Transient,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Admission {
    pub request_id: HostRequestId,
    pub input: PreparedModelInput,
    pub options: Options,
    pub constraint: Option<ConstraintPlan>,
    pub retention: RetentionPolicy,
    /// Output batches the request's publication queue holds; also the host's
    /// initial credit.
    pub output_capacity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HostMessage {
    Hello { build: EngineBuild },
    Load { manifest: ExecutionManifest },
    Admit(Admission),
    /// The host's consumer drained `batches` output batches.
    Credit { request_id: HostRequestId, batches: u32 },
    /// Ordered stop: accepted output drains, then `Completed`.
    Stop { request_id: HostRequestId },
    /// The host abandoned the request; no further messages arrive for it.
    Cancel { request_id: HostRequestId },
    Status { request_id: HostRequestId },
    Observe,
    Shutdown,
}

/// Engine load progress, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadPhase {
    /// Device resolution and execution planning.
    Planning,
    OpeningDevice,
    /// Program attestation, tuning and graph preparation.
    PreparingPrograms,
    /// Resource allocation and target weight import.
    ImportingWeights,
    /// State allocation, memory policy and warm-up.
    Finalizing,
}

/// Where an admitted request is in scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    Runnable,
    AwaitingCompletion,
    OutputBlocked,
    Preempted,
    CapacityBlocked { required: u64, available: u64 },
    Finished(FinishReason),
}

/// A live request's scheduling state and progress through its input and
/// output. Prefill is complete once `resident_tokens >= prompt_tokens`;
/// its leading `cached_tokens` were restored from a retained prefix rather
/// than computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestProgress {
    pub state: RequestState,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub resident_tokens: usize,
    pub output_tokens: usize,
}

/// Measured physical execution time of a completed request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionTimings {
    pub prompt_ns: u64,
    pub predicted_ns: u64,
}

/// A loaded worker's observation of its memory (integration spec §5.4,
/// §9.2): what the model holds and what each domain it uses has left.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryObservation {
    /// The heap's standing, classified.
    pub census: AllocationCensus,
    /// A fresh reading of every memory domain the loaded device uses: its
    /// allocation domain first, then host RAM for a dedicated device's
    /// staged uploads.
    pub domains: Vec<DomainHeadroom>,
}

/// One memory domain's observed available bytes. Only the worker observes a
/// dedicated device's live memory: it has the device open.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainHeadroom {
    pub domain: MemoryDomain,
    /// Available bytes bounded by every applicable process limit; this
    /// process's existing charges are already absent from it.
    pub headroom_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkerMessage {
    Hello { build: EngineBuild },
    LoadPhase { phase: LoadPhase },
    Ready { ready: ReadyInfo },
    LoadFailed { error: LoadError },
    Admitted { request_id: HostRequestId },
    AdmissionRefused { request_id: HostRequestId, error: RequestError },
    Output { request_id: HostRequestId, tokens: Vec<OutputToken> },
    Completed {
        request_id: HostRequestId,
        finish: FinishReason,
        usage: DetailedUsage,
        method: String,
        timings: ExecutionTimings,
    },
    Failed { request_id: HostRequestId, error: RequestError },
    /// `None` when the request is not live in the worker.
    Status {
        request_id: HostRequestId,
        progress: Option<RequestProgress>,
    },
    Observed {
        observation: Result<MemoryObservation, RequestError>,
    },
    /// The model stopped serving. Every open request has already received
    /// its terminal outcome; the worker exits after this message.
    Unloaded { cause: UnloadCause },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::DomainAllocation;
    use crate::error::InsufficientMemory;
    use crate::options::InputModalities;
    use seismic::DeviceSelector;

    fn round_trip<T: Serialize + serde::de::DeserializeOwned>(value: &T) -> T {
        postcard::from_bytes(&postcard::to_allocvec(value).unwrap()).unwrap()
    }

    /// Memory observations, their failure and a domain-qualified load
    /// refusal survive the process encoding.
    #[test]
    fn memory_values_round_trip_through_the_worker_encoding() {
        let device = MemoryDomain::DeviceLocal {
            device: DeviceSelector::Cuda { uuid: [7; 16] },
        };
        let observation = MemoryObservation {
            census: AllocationCensus {
                domains: vec![DomainAllocation {
                    domain: device,
                    model_bytes: 1,
                    context_bytes: 2,
                    compute_bytes: 3,
                    auxiliary_bytes: 4,
                }],
            },
            domains: vec![
                DomainHeadroom {
                    domain: device,
                    headroom_bytes: 5,
                },
                DomainHeadroom {
                    domain: MemoryDomain::HostRam,
                    headroom_bytes: 6,
                },
            ],
        };
        let WorkerMessage::Observed {
            observation: Ok(decoded),
        } = round_trip(&WorkerMessage::Observed {
            observation: Ok(observation.clone()),
        })
        else {
            panic!("observation changed shape")
        };
        assert_eq!(decoded, observation);
        let failed = RequestError::MemoryObservationUnavailable {
            reason: "blind".into(),
        };
        let WorkerMessage::Observed {
            observation: Err(decoded),
        } = round_trip(&WorkerMessage::Observed {
            observation: Err(failed.clone()),
        })
        else {
            panic!("observation failure changed shape")
        };
        assert_eq!(decoded, failed);
        let refusal = LoadError::InsufficientMemory {
            purpose: "target import".into(),
            domain: MemoryDomain::HostRam,
            memory: InsufficientMemory {
                required: 8,
                available: 9,
            },
        };
        assert_eq!(round_trip(&refusal), refusal);
        let modalities = InputModalities { image: true };
        assert_eq!(round_trip(&modalities), modalities);
    }
}
