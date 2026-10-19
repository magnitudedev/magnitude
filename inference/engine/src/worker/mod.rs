//! The numerical worker and its host connection (integration spec §8.1).
//!
//! `run_worker` is the worker: it loads the manifest's model on its own
//! device catalog and serves one host over a [`transport::WorkerTransport`].
//! The host side is [`EngineClient`]. The same code runs in-process (CLI,
//! tests: a thread over a channel transport) and in a worker process (the
//! service: framed messages over stdio).
//!
//! A load never searches kernels. A worker process given `Prepare` instead
//! of `Load` runs the engine's only search and exits. Every host loads
//! through [`load_prepared`], which has the Prepare run, to its end, only
//! when a load finds tuning units without a stored record.

mod binding;
mod client;
mod execution;
mod preparation;
pub mod protocol;
mod session;
pub mod transport;

pub use binding::GenerationBinding;
pub use client::{
    connect_worker, EngineClient, EngineRequest, RequestEvent, RequestOptions, WorkerConnection,
};
pub use preparation::{
    load_prepared, prepare, start_worker_process, ChildTransport, Preparation, WorkerProcess,
    WorkerReport,
};

use crate::census::AllocationCensus;
use crate::execution::LoadAttempt;
use magnitude_executor::UnpreparedUnits;
use crate::error::{ArtifactError, LoadError, UnloadCause};
use crate::options::{ExecutionManifest, InputModalities, ReadyInfo, ResourcePlanSummary};
use execution::ExecutionOwner;
use magnitude_artifacts::Package;
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer,
};
use magnitude_grammar::CacheLimits;
use magnitude_scheduler::{
    owner::Owner,
    worker::{Driven, SpawnError, Wakes, Worker, WorkerWakeHandle},
};
use protocol::{EngineBuild, HostMessage, WorkerMessage};
use session::{Loaded, Outbound, Session, Signal};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use transport::{MessageReceiver, MessageSender, WorkerTransport};

/// Compiled constraint matchers the worker keeps for reuse across requests.
const CONSTRAINT_CACHE: CacheLimits = CacheLimits {
    entries: 16,
    bytes: 64 << 20,
};

/// How a worker ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerExit {
    /// The host asked it to shut down.
    Shutdown,
    /// A `Prepare` finished; the host received `Prepared`.
    Prepared,
    /// A `Load` asked to report tuning units without a stored record met
    /// one and placed nothing; the host received `Unprepared`.
    Unprepared,
    /// The model stopped serving (memory pressure or an execution failure).
    Unloaded(UnloadCause),
    /// Loading failed; the host received `LoadFailed`.
    LoadFailed(LoadError),
    /// The host's end of the transport closed.
    HostLost { reason: Option<String> },
    /// The host speaks a different engine build.
    BuildMismatch { host: EngineBuild, worker: EngineBuild },
    /// The host broke the protocol.
    ProtocolViolation(String),
    /// Writing to the host failed.
    TransportFailed(String),
}

fn internal(reason: impl Into<String>) -> LoadError {
    LoadError::Internal {
        reason: reason.into(),
    }
}

/// Load `manifest`'s model and serve the host at the other end of
/// `transport` until shutdown, host loss or unload; or, asked to report
/// tuning units without a stored record, end unprepared at the first.
pub fn run_worker(
    manifest: ExecutionManifest,
    unprepared: UnpreparedUnits,
    transport: impl WorkerTransport,
) -> WorkerExit {
    let (receiver, sender) = transport.split();
    serve(manifest, unprepared, receiver, Box::new(sender))
}

/// A worker process's entry: accept the host's `Hello`, then either run the
/// worker on a `Load`'s manifest or only prepare a `Prepare`'s.
pub fn serve_worker(transport: impl WorkerTransport) -> WorkerExit {
    let (mut receiver, mut sender) = transport.split();
    let worker = EngineBuild::current();
    let host = match receiver.receive() {
        Ok(Some(HostMessage::Hello { build })) => build,
        Ok(Some(_)) => return WorkerExit::ProtocolViolation("expected Hello".into()),
        Ok(None) => return WorkerExit::HostLost { reason: None },
        Err(error) => {
            return WorkerExit::HostLost {
                reason: Some(error.to_string()),
            }
        }
    };
    if host != worker {
        // The host learns the mismatch from our Hello.
        let _ = sender.send(WorkerMessage::Hello {
            build: worker.clone(),
        });
        return WorkerExit::BuildMismatch { host, worker };
    }
    let (manifest, unprepared) = match receiver.receive() {
        Ok(Some(HostMessage::Load {
            manifest,
            unprepared,
        })) => (manifest, unprepared),
        Ok(Some(HostMessage::Prepare { manifest })) => return prepare_only(manifest, sender),
        Ok(Some(_)) => return WorkerExit::ProtocolViolation("expected Load or Prepare".into()),
        Ok(None) => return WorkerExit::HostLost { reason: None },
        Err(error) => {
            return WorkerExit::HostLost {
                reason: Some(error.to_string()),
            }
        }
    };
    serve(manifest, unprepared, receiver, Box::new(sender))
}

/// A Prepare: search the tuning units of `manifest`'s model the kernel cache
/// has no record for, on the device its load selects, and report the
/// outcome. Each record is published as it stands, so a Prepare that is
/// stopped leaves what it reached. Nothing is prepared for serving: no
/// serving kernels are formed and held, no graph is sealed, nothing is
/// allocated or imported for the model. The process ends after the outcome.
fn prepare_only(
    manifest: ExecutionManifest,
    sender: impl MessageSender<WorkerMessage>,
) -> WorkerExit {
    let outbound: Outbound = Arc::new(Mutex::new(Box::new(sender)));
    let hello = WorkerMessage::Hello {
        build: EngineBuild::current(),
    };
    if let Err(error) = outbound.lock().unwrap().send(hello) {
        return WorkerExit::TransportFailed(error.to_string());
    }
    let progress_outbound = outbound.clone();
    let prepared = Package::open_manifest(&manifest.package)
        .map_err(|error| {
            LoadError::Artifact(ArtifactError::from_artifacts(
                error,
                manifest.package.target.path(),
            ))
        })
        .and_then(|package| {
            let catalog =
                seismic::Devices::discover(&[]).map_err(|error| internal(error.to_string()))?;
            crate::execution::search(
                Rc::new(catalog),
                &manifest,
                &package,
                Rc::new(move |progress| {
                    let _ = progress_outbound
                        .lock()
                        .unwrap()
                        .send(WorkerMessage::LoadProgress { progress });
                }),
            )
        });
    let (message, exit) = match prepared {
        Ok(()) => (WorkerMessage::Prepared, WorkerExit::Prepared),
        Err(error) => (
            WorkerMessage::LoadFailed {
                error: error.clone(),
            },
            WorkerExit::LoadFailed(error),
        ),
    };
    let sent = outbound.lock().unwrap().send(message);
    match sent {
        Ok(()) => exit,
        Err(error) => WorkerExit::TransportFailed(error.to_string()),
    }
}

fn serve(
    manifest: ExecutionManifest,
    unprepared: UnpreparedUnits,
    receiver: impl MessageReceiver<HostMessage>,
    sender: Box<dyn MessageSender<WorkerMessage>>,
) -> WorkerExit {
    let outbound: Outbound = Arc::new(Mutex::new(sender));
    let hello = WorkerMessage::Hello {
        build: EngineBuild::current(),
    };
    if let Err(error) = outbound.lock().unwrap().send(hello) {
        return WorkerExit::TransportFailed(error.to_string());
    }
    let signal = Arc::new(Signal::default());
    // Host messages that arrive during the load wait in the session inbox.
    let _reader = session::spawn_reader(receiver, signal.clone());
    match load(manifest, unprepared, &outbound) {
        Ok(LoadAttempt::Loaded((loaded, ready))) => {
            if let Err(error) = outbound.lock().unwrap().send(WorkerMessage::Ready { ready }) {
                return WorkerExit::TransportFailed(error.to_string());
            }
            Session::new(outbound, signal, loaded).run()
        }
        // Nothing of the model was placed, and everything the attempt
        // opened was dropped before this answer.
        Ok(LoadAttempt::Unprepared) => match outbound.lock().unwrap().send(WorkerMessage::Unprepared)
        {
            Ok(()) => WorkerExit::Unprepared,
            Err(transport) => WorkerExit::TransportFailed(transport.to_string()),
        },
        Err(error) => {
            let failed = WorkerMessage::LoadFailed {
                error: error.clone(),
            };
            match outbound.lock().unwrap().send(failed) {
                Ok(()) => WorkerExit::LoadFailed(error),
                Err(transport) => WorkerExit::TransportFailed(transport.to_string()),
            }
        }
    }
}

/// Readiness facts of the constructed execution domain.
struct ExecutionReady {
    resources: ResourcePlanSummary,
    device: seismic::ExecutorSelector,
    backend: seismic::BackendName,
    census: AllocationCensus,
}

/// The chat semantics the worker reads from its own opened package.
struct ChatIdentity {
    template_fingerprint: String,
    modalities: InputModalities,
}

/// Build the execution owner on its own thread while this thread prepares
/// the worker's tokenizer, constraint vocabulary and chat identity.
fn load(
    manifest: ExecutionManifest,
    unprepared: UnpreparedUnits,
    outbound: &Outbound,
) -> Result<LoadAttempt<(Loaded, ReadyInfo)>, LoadError> {
    /// Why the worker's two halves stopped constructing.
    enum Stopped {
        Unprepared,
        Failed(LoadError),
    }
    impl From<LoadError> for Stopped {
        fn from(error: LoadError) -> Self {
            Self::Failed(error)
        }
    }
    impl std::fmt::Display for Stopped {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Unprepared => formatter.write_str("the load stopped unprepared"),
                Self::Failed(error) => error.fmt(formatter),
            }
        }
    }
    let package = Arc::new(Package::open_manifest(&manifest.package).map_err(|error| {
        LoadError::Artifact(ArtifactError::from_artifacts(
            error,
            manifest.package.target.path(),
        ))
    })?);
    let definition = manifest.definition.clone();
    let session_definition = manifest.definition.clone();
    let manifest_identity = manifest.package.identity;
    let ready_model = manifest.model.clone();
    let ready_service = manifest.service.clone();
    let ready_path = manifest.path;
    let host_package = package.clone();
    let progress_outbound = outbound.clone();
    // Readiness and the host-resident tables' bytes.
    type Built = (ExecutionReady, u64);
    let factory = move |wakes: WorkerWakeHandle<UnloadCause>| -> Result<
        (Box<dyn Driven<Stopped = UnloadCause>>, Built),
        Stopped,
    > {
        let catalog = seismic::Devices::discover(&[]).map_err(|error| internal(error.to_string()))?;
        let built = match crate::execution::build(
            catalog,
            &manifest,
            package,
            unprepared,
            Rc::new(move |progress| {
                // A host that is gone is noticed by the session reader.
                let _ = progress_outbound
                    .lock()
                    .unwrap()
                    .send(WorkerMessage::LoadProgress { progress });
            }),
        )? {
            LoadAttempt::Loaded(built) => built,
            LoadAttempt::Unprepared => return Err(Stopped::Unprepared),
        };
        let method = manifest
            .model
            .factory(&manifest.package.identity.to_string())
            .map_err(internal)?;
        if built.domain.execution_path() != manifest.path {
            return Err(
                internal("executor domain differs from the planned execution path").into(),
            );
        }
        let backend = built.domain.execution_backend();
        let host_table_bytes = built.domain.host_table_bytes();
        let resources = ResourcePlanSummary::from_plan(&built.plan).map_err(internal)?;
        let owner = Owner::with_resource_plan(
            built.domain,
            built.bindings,
            manifest.service.clone(),
            &built.plan,
            Arc::new(wakes) as Arc<dyn Wakes>,
        )
        .map_err(internal)?;
        let census = owner
            .reconcile_memory_charge()
            .and_then(|charge| AllocationCensus::classify(&charge, host_table_bytes))
            .map_err(internal)?;
        let ready = ExecutionReady {
            resources,
            device: built.device,
            backend,
            census,
        };
        Ok((
            Box::new(ExecutionOwner::new(owner, method)) as Box<dyn Driven<Stopped = UnloadCause>>,
            (ready, host_table_bytes),
        ))
    };
    let host = move || -> Result<(GenerationBinding, ChatIdentity), Stopped> {
        let unsupported = |reason: String| {
            LoadError::Unsupported(crate::error::UnsupportedModel::Representation { reason })
        };
        // The worker's own reading of its package's chat semantics, which the
        // host verifies against its resolution.
        let family = crate::families::recognize(host_package.target().directory())
            .map_err(LoadError::Unsupported)?;
        let chat = ChatIdentity {
            template_fingerprint: gguf_templates(host_package.templates(), host_package.tokenizer())
                .and_then(|templates| templates.inspect())
                .map_err(unsupported)?
                .fingerprint,
            modalities: InputModalities::of(family, &definition),
        };
        let tokenizer = Arc::new(
            gguf_byte_bpe(
                host_package.tokenizer(),
                definition.artifact_identity.target.to_string(),
            )
            .and_then(ByteBpeTokenizer::new)
            .map_err(|error| unsupported(error.to_string()))?,
        );
        let projection = usize::try_from(definition.decoder.vocabulary)
            .map_err(|_| internal("model vocabulary exceeds the host domain"))?;
        let binding =
            GenerationBinding::new(tokenizer, projection, CONSTRAINT_CACHE).map_err(unsupported)?;
        Ok((binding, chat))
    };
    let (
        execution,
        (execution_ready, host_table_bytes),
        (binding, chat),
    ) = match Worker::spawn_ready_with(factory, usize::MAX, host) {
        Ok(spawned) => spawned,
        Err(SpawnError::Factory(Stopped::Unprepared) | SpawnError::Host(Stopped::Unprepared)) => {
            return Ok(LoadAttempt::Unprepared)
        }
        Err(
            SpawnError::Factory(Stopped::Failed(error)) | SpawnError::Host(Stopped::Failed(error)),
        ) => return Err(error),
        Err(error @ (SpawnError::Capacity | SpawnError::Thread(_) | SpawnError::Panicked)) => {
            return Err(internal(error.to_string()))
        }
    };
    let ready = ReadyInfo {
        package: manifest_identity,
        template_fingerprint: chat.template_fingerprint,
        modalities: chat.modalities,
        model: ready_model,
        service: ready_service,
        resources: execution_ready.resources,
        path: ready_path,
        device: execution_ready.device,
        backend: execution_ready.backend,
        census: execution_ready.census,
    };
    Ok(LoadAttempt::Loaded((
        Loaded {
            execution,
            binding,
            definition: session_definition,
            host_table_bytes,
        },
        ready,
    )))
}
