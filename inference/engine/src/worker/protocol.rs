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

use super::measure::{MeasureFailure, MeasureRequest};
use crate::census::AllocationCensus;
use crate::error::{LoadError, RequestError, UnloadCause};
use crate::options::{ExecutionManifest, ReadyInfo};
use magnitude_chat::ConstraintPlan;
use magnitude_executor::UnpreparedUnits;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{DetailedUsage, FinishReason, Options, OutputToken};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use seismic::measure::Record;
use seismic::{PointKey, StoreIdentity};
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Admission {
    pub request_id: HostRequestId,
    pub input: PreparedModelInput,
    pub options: Options,
    pub constraint: Option<ConstraintPlan>,
    /// Whether the request's prompt state is kept for later exact-prefix
    /// reuse. The prefix cache belongs to the worker's one loaded model, so
    /// package, tokenizer and codec identities are fixed for every path.
    pub retention: PrefixRetention,
    /// Output batches the request's publication queue holds; also the host's
    /// initial credit.
    pub output_capacity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HostMessage {
    Hello { build: EngineBuild },
    /// Load `manifest`'s model and serve it. A load never searches: each
    /// tuning unit is prepared from its stored record. At a unit the kernel
    /// cache has no record for, the load does what `unprepared` says: it
    /// places nothing, answers `Unprepared` and exits, or it prepares the
    /// unit at its canonical default and goes on.
    Load {
        manifest: ExecutionManifest,
        unprepared: UnpreparedUnits,
    },
    /// Search the tuning units of `manifest`'s model the kernel cache has no
    /// record for, on the device its load selects, publishing each record;
    /// then exit without loading the model. The only search there is.
    Prepare {
        manifest: ExecutionManifest,
    },
    /// Measure this machine as `request` asks (the measurement job,
    /// [`super::measure`]), reporting each record as it is stored; then
    /// exit. Nothing is loaded.
    Measure {
        request: MeasureRequest,
    },
    Admit(Admission),
    /// The host's consumer drained `batches` output batches.
    Credit {
        request_id: HostRequestId,
        batches: u32,
    },
    /// Ordered stop: accepted output drains, then `Completed`.
    Stop {
        request_id: HostRequestId,
    },
    /// The host abandoned the request; no further messages arrive for it.
    Cancel {
        request_id: HostRequestId,
    },
    Status {
        request_id: HostRequestId,
    },
    Observe,
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TuningOperation {
    Discovery,
    Form,
    Measure,
    Promote,
    Confirm,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningWork {
    pub entry: String,
    pub operation: TuningOperation,
}

/// Engine load progress. Tuning reports time budget; import reports resident bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadProgress {
    /// Device selection, planning, opening the device and program preparation.
    Preparing,
    /// Tuning kernels for the device. Reported only by a Prepare that searches, never by a
    /// load: `total` is the tuning time in milliseconds and `completed` the milliseconds
    /// spent. It starts at zero when the census begins; tuning may finish before `total`.
    Tuning {
        completed: u64,
        total: u64,
        work: Option<TuningWork>,
        measured_configurations: u64,
        formed_programs: u64,
    },
    /// Target weight import, in resident bytes. It starts at zero before any import.
    ImportingWeights { completed_bytes: u64, total_bytes: u64 },
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
/// §9.2): what the model holds and what each store it uses has left.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryObservation {
    /// The heap's standing, classified.
    pub census: AllocationCensus,
    /// A fresh reading of every store the loaded executor's placements
    /// reach, and host RAM, in store order.
    pub stores: Vec<StoreHeadroom>,
}

/// One store's observed available bytes. Only the worker observes a
/// device store's live memory: it has the executor open.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreHeadroom {
    pub store: StoreIdentity,
    /// Available bytes bounded by every applicable process limit; this
    /// process's existing charges are already absent from it.
    pub headroom_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkerMessage {
    Hello {
        build: EngineBuild,
    },
    LoadProgress {
        progress: LoadProgress,
    },
    Ready {
        ready: ReadyInfo,
    },
    /// A `Prepare` finished; the worker exits after this message.
    Prepared,
    /// A `Measure` is about to record its attempt of `key` and time it.
    Attempting {
        key: PointKey,
    },
    /// A `Measure` stored `record` of `key`.
    Recorded {
        key: PointKey,
        record: Record,
    },
    /// A `Measure` finished, with what it could not measure; the worker
    /// exits after this message.
    MeasureFinished {
        failures: Vec<MeasureFailure>,
    },
    /// A `Load` asked to report tuning units without a stored record met
    /// one and placed nothing; the worker exits after this message.
    Unprepared,
    LoadFailed {
        error: LoadError,
    },
    Admitted {
        request_id: HostRequestId,
    },
    AdmissionRefused {
        request_id: HostRequestId,
        error: RequestError,
    },
    Output {
        request_id: HostRequestId,
        tokens: Vec<OutputToken>,
    },
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
    use crate::error::InsufficientMemory;
    use crate::options::InputModalities;
    use magnitude_executor::StoreAllocation;

    fn round_trip<T: Serialize + serde::de::DeserializeOwned>(value: &T) -> T {
        postcard::from_bytes(&postcard::to_allocvec(value).unwrap()).unwrap()
    }

    /// Memory observations, their failure and a store-qualified load
    /// refusal survive the process encoding.
    #[test]
    fn memory_values_round_trip_through_the_worker_encoding() {
        let device = StoreIdentity::DeviceMemory(seismic::PhysicalIdentity::Uuid([7; 16]));
        let observation = MemoryObservation {
            census: AllocationCensus {
                stores: vec![StoreAllocation {
                    store: device.clone(),
                    model_bytes: 1,
                    context_bytes: 2,
                    compute_bytes: 3,
                    auxiliary_bytes: 4,
                }],
            },
            stores: vec![
                StoreHeadroom {
                    store: device.clone(),
                    headroom_bytes: 5,
                },
                StoreHeadroom {
                    store: StoreIdentity::HostRam,
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
            store: StoreIdentity::HostRam,
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
