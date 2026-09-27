//! The numerical worker and its host connection (integration spec §8.1).
//!
//! `run_worker` is the worker: it loads the manifest's model on its own
//! device catalog and serves one host over a [`transport::WorkerTransport`].
//! The host side is [`EngineClient`]. The same code runs in-process (CLI,
//! tests: a thread over a channel transport) and in a worker process (the
//! service: framed messages over stdio).

mod client;
mod execution;
pub mod protocol;
mod session;
pub mod transport;

pub use client::{
    connect_worker, EngineClient, EngineRequest, RequestEvent, RequestOptions, WorkerConnection,
};

use crate::census::{AllocationCensus, MemoryDomain};
use crate::error::{ArtifactError, LoadError, UnloadCause};
use crate::options::{ExecutionManifest, InputModalities, ReadyInfo, ResourcePlanSummary};
use execution::{ExecutionOwner, UnloadNotice};
use magnitude_artifacts::Package;
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer, CacheLimits, PreparedVocabulary, Vocabulary,
};
use magnitude_scheduler::{
    owner::Owner,
    retention::{RetentionKey, TokenizerIdentity},
    worker::{Driven, SpawnError, Worker},
};
use magnitude_state::CodecIdentity;
use protocol::{EngineBuild, HostMessage, WorkerMessage};
use seismic::DeviceCatalog;
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
/// `transport` until shutdown, host loss or unload.
pub fn run_worker(manifest: ExecutionManifest, transport: impl WorkerTransport) -> WorkerExit {
    let (receiver, sender) = transport.split();
    serve(manifest, receiver, Box::new(sender))
}

/// A worker process's entry: accept the host's `Hello` and `Load`, then run
/// the worker on that manifest.
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
    let manifest = match receiver.receive() {
        Ok(Some(HostMessage::Load { manifest })) => manifest,
        Ok(Some(_)) => return WorkerExit::ProtocolViolation("expected Load".into()),
        Ok(None) => return WorkerExit::HostLost { reason: None },
        Err(error) => {
            return WorkerExit::HostLost {
                reason: Some(error.to_string()),
            }
        }
    };
    serve(manifest, receiver, Box::new(sender))
}

fn serve(
    manifest: ExecutionManifest,
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
    match load(manifest, &outbound, &signal) {
        Ok((loaded, ready)) => {
            if let Err(error) = outbound.lock().unwrap().send(WorkerMessage::Ready { ready }) {
                return WorkerExit::TransportFailed(error.to_string());
            }
            Session::new(outbound, signal, loaded).run()
        }
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
    device: seismic::DeviceSelector,
    backend: seismic::BackendName,
    census: AllocationCensus,
}

/// The chat semantics the worker reads from its own opened package.
struct ChatIdentity {
    template_fingerprint: String,
    modalities: InputModalities,
}

/// Build the execution owner on its own thread while this thread prepares
/// the worker's tokenizer, constraint vocabulary, retention key and chat
/// identity.
fn load(
    manifest: ExecutionManifest,
    outbound: &Outbound,
    signal: &Arc<Signal>,
) -> Result<(Loaded, ReadyInfo), LoadError> {
    let package = Arc::new(Package::open_manifest(&manifest.package).map_err(|error| {
        LoadError::Artifact(ArtifactError::from_artifacts(
            error,
            manifest.package.target.path(),
        ))
    })?);
    let method = manifest
        .model
        .method
        .factory(&manifest.package.identity.to_string())
        .map_err(internal)?;
    let notice = Arc::new(UnloadNotice::default());
    let wake = signal.clone();
    notice.install_wake(move || wake.notify());
    let definition = manifest.definition.clone();
    let session_definition = manifest.definition.clone();
    let manifest_identity = manifest.package.identity;
    let ready_model = manifest.model.clone();
    let ready_service = manifest.service.clone();
    let ready_path = manifest.path;
    let codec = manifest.model.kv_codec;
    let host_package = package.clone();
    let progress_outbound = outbound.clone();
    let owner_notice = notice.clone();
    // Readiness, the compute bytes, the host-table bytes and the allocation
    // domain.
    type Built = (ExecutionReady, u64, u64, MemoryDomain);
    let factory = move || -> Result<(Box<dyn Driven>, Built), LoadError> {
        let catalog = DeviceCatalog::discover().map_err(|error| internal(error.to_string()))?;
        let built = crate::execution::build(
            catalog,
            &manifest,
            package,
            Rc::new(move |progress| {
                // A host that is gone is noticed by the session reader.
                let _ = progress_outbound
                    .lock()
                    .unwrap()
                    .send(WorkerMessage::LoadProgress { progress });
            }),
        )?;
        if built.domain.execution_path() != manifest.path {
            return Err(internal("executor domain differs from the planned execution path"));
        }
        let backend = built.domain.execution_backend();
        let compute_bytes = built.plan.bytes().scratch;
        let host_table_bytes = built.domain.host_table_bytes();
        let domain = MemoryDomain::of(built.device, built.pool);
        let resources = ResourcePlanSummary::from_plan(&built.plan).map_err(internal)?;
        let owner = Owner::with_resource_plan(built.domain, manifest.service.clone(), &built.plan)
            .map_err(internal)?;
        let census = owner
            .inspect_domain(|domain_state| {
                let heap = domain_state.memory();
                let standing = heap
                    .standing()
                    .map_err(|error| format!("memory standing is unavailable: {error:?}"))?;
                AllocationCensus::classify(
                    &standing,
                    heap.holdings(),
                    compute_bytes,
                    host_table_bytes,
                    domain,
                )
            })
            .map_err(internal)?;
        let ready = ExecutionReady {
            resources,
            device: built.device,
            backend,
            census,
        };
        Ok((
            Box::new(ExecutionOwner {
                owner: Some(owner),
                method,
                wakes: None,
                notice: owner_notice,
            }) as Box<dyn Driven>,
            (ready, compute_bytes, host_table_bytes, domain),
        ))
    };
    let host = move || -> Result<(Vocabulary, RetentionKey, ChatIdentity), LoadError> {
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
        let retention = RetentionKey::new(
            host_package.identity(),
            TokenizerIdentity::new(tokenizer.identity()).map_err(internal)?,
            CodecIdentity::new(codec.identity()).map_err(internal)?,
        );
        let vocabulary = PreparedVocabulary::new(tokenizer, projection)
            .map_err(unsupported)?
            .with_cache_limits(CONSTRAINT_CACHE);
        Ok((vocabulary, retention, chat))
    };
    let (
        execution,
        (execution_ready, compute_bytes, host_table_bytes, domain),
        (vocabulary, retention, chat),
    ) =
        Worker::spawn_ready_with(factory, usize::MAX, host).map_err(|error| match error {
            SpawnError::Factory(error) | SpawnError::Host(error) => error,
            error @ (SpawnError::Capacity | SpawnError::Thread(_) | SpawnError::Panicked) => {
                internal(error.to_string())
            }
        })?;
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
    Ok((
        Loaded {
            execution,
            vocabulary,
            retention,
            definition: session_definition,
            compute_bytes,
            host_table_bytes,
            domain,
            notice,
        },
        ready,
    ))
}
