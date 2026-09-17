#[cfg(windows)]
use icn_utils::windows_process::OwnedWindowsChild as Child;
use std::collections::HashMap;
use std::io::{Read, Write};
#[cfg(unix)]
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use icn_contracts::inference::{
    InferenceCompletion, InferenceObservation, ResolvedInferenceRequest,
};
use icn_contracts::{
    CompletionBackend, InferenceError, ModelProperties, PreparedChatInfo, ReasoningProfile,
    SpeculativeDecodingConfig, TemplateCapabilities,
};
use icn_engine::{ModelLoadObserver, ModelLoadPhase, NativeBackend};
use icn_hardware::CapacityPolicy;
use serde::{Deserialize, Serialize};

use crate::worker_process::NativeWorkerLauncher;
#[cfg(unix)]
use crate::worker_process::NativeWorkerRole;

const PROTOCOL_VERSION: u32 = 4;
const MAX_FRAME_BYTES: usize = 48 * 1024 * 1024;
// At most one maximum-sized request may wait behind the frame currently being written.
const COMMAND_QUEUE_CAPACITY: usize = 1;
const RESPONSE_QUEUE_CAPACITY: usize = 32;
const REQUEST_EVENT_CAPACITY: usize = 16;
const LOAD_EVENT_CAPACITY: usize = 32;
const MAX_PENDING_REQUESTS: usize = 1024;

#[derive(Debug, Serialize, Deserialize)]
struct Frame<T> {
    protocol_version: u32,
    generation: u64,
    message: T,
}

#[derive(Debug, Serialize, Deserialize)]
enum HostMessage {
    Hello {
        expected_build: String,
    },
    Load {
        model_id: String,
        plan: icn_contracts::ExecutionIntent,
        speculative: SpeculativeDecodingConfig,
        hardware: icn_contracts::HardwareSnapshot,
        template_capabilities: TemplateCapabilities,
        reasoning: ReasoningProfile,
        template_fingerprint: String,
        expected_vision: bool,
        trace: crate::telemetry::TraceCarrier,
    },
    Infer {
        request_id: u64,
        request: ResolvedInferenceRequest,
        trace: crate::telemetry::TraceCarrier,
    },
    ApplyTemplate {
        request_id: u64,
        request: ResolvedInferenceRequest,
        trace: crate::telemetry::TraceCarrier,
    },
    CountTokens {
        request_id: u64,
        request: ResolvedInferenceRequest,
        trace: crate::telemetry::TraceCarrier,
    },
    ObserveModelInstance {
        request_id: u64,
        policy: CapacityPolicy,
        native_build: String,
        enabled_backends: Vec<String>,
        trace: crate::telemetry::TraceCarrier,
    },
    Cancel {
        request_id: u64,
    },
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
enum WorkerMessage {
    Hello {
        build: String,
    },
    Prepared {
        acceleration: String,
        timing_plan_identity: String,
        phases: Vec<ModelLoadPhase>,
    },
    LoadPhase {
        phase: ModelLoadPhase,
        started: bool,
    },
    Loaded {
        properties: ModelProperties,
    },
    LoadFailed {
        message: String,
    },
    InferenceObservation {
        request_id: u64,
        event: InferenceObservation,
    },
    InferenceAdmitted {
        request_id: u64,
        prompt_tokens: u64,
    },
    RequestCompleted {
        request_id: u64,
        completion: InferenceCompletion,
    },
    RequestFailed {
        request_id: u64,
        error: WireInferenceError,
    },
    TemplateCompleted {
        request_id: u64,
        result: Result<PreparedChatInfo, WireInferenceError>,
    },
    TokenCountCompleted {
        request_id: u64,
        result: Result<u64, WireInferenceError>,
    },
    ModelInstanceObserved {
        request_id: u64,
        result: Result<icn_engine::ModelInstanceObservation, String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WireInferenceError {
    InvalidConfig(String),
    ContextLengthExceeded {
        prompt_tokens: u64,
        context_capacity: u64,
    },
    Backend(String),
    Cancelled,
    ModelInstanceStopped,
    Overloaded,
    ExecutorStopped,
    Callback(String),
}

impl From<InferenceError> for WireInferenceError {
    fn from(error: InferenceError) -> Self {
        match error {
            InferenceError::InvalidConfig(message) => Self::InvalidConfig(message),
            InferenceError::ContextLengthExceeded {
                prompt_tokens,
                context_capacity,
            } => Self::ContextLengthExceeded {
                prompt_tokens,
                context_capacity,
            },
            InferenceError::Backend(message) => Self::Backend(message),
            InferenceError::Cancelled => Self::Cancelled,
            InferenceError::ModelInstanceStopped => Self::ModelInstanceStopped,
            InferenceError::Overloaded => Self::Overloaded,
            InferenceError::ExecutorStopped => Self::ExecutorStopped,
            InferenceError::Callback(message) => Self::Callback(message),
        }
    }
}

impl From<WireInferenceError> for InferenceError {
    fn from(error: WireInferenceError) -> Self {
        match error {
            WireInferenceError::InvalidConfig(message) => Self::InvalidConfig(message),
            WireInferenceError::ContextLengthExceeded {
                prompt_tokens,
                context_capacity,
            } => Self::ContextLengthExceeded {
                prompt_tokens,
                context_capacity,
            },
            WireInferenceError::Backend(message) => Self::Backend(message),
            WireInferenceError::Cancelled => Self::Cancelled,
            WireInferenceError::ModelInstanceStopped => Self::ModelInstanceStopped,
            WireInferenceError::Overloaded => Self::Overloaded,
            WireInferenceError::ExecutorStopped => Self::ExecutorStopped,
            WireInferenceError::Callback(message) => Self::Callback(message),
        }
    }
}

fn write_frame<T: Serialize>(
    writer: &mut impl Write,
    generation: u64,
    message: T,
) -> anyhow::Result<()> {
    let payload = serde_json::to_vec(&Frame {
        protocol_version: PROTOCOL_VERSION,
        generation,
        message,
    })?;
    anyhow::ensure!(
        payload.len() <= MAX_FRAME_BYTES,
        "IPC frame is {} bytes; maximum is {MAX_FRAME_BYTES}",
        payload.len()
    );
    let length = u32::try_from(payload.len()).map_err(anyhow::Error::msg)?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

fn read_frame<T: for<'de> Deserialize<'de>>(reader: &mut impl Read) -> anyhow::Result<Frame<T>> {
    let mut length = [0_u8; 4];
    reader.read_exact(&mut length)?;
    let length = u32::from_be_bytes(length) as usize;
    anyhow::ensure!(
        length <= MAX_FRAME_BYTES,
        "IPC peer declared a {length}-byte frame; maximum is {MAX_FRAME_BYTES}"
    );
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload)?;
    let frame: Frame<T> = serde_json::from_slice(&payload)?;
    anyhow::ensure!(
        frame.protocol_version == PROTOCOL_VERSION,
        "IPC protocol version mismatch: got {}, expected {PROTOCOL_VERSION}",
        frame.protocol_version
    );
    Ok(frame)
}

#[derive(Debug)]
pub(crate) enum LoadEvent {
    Prepared {
        acceleration: String,
        timing_plan_identity: String,
        phases: Vec<ModelLoadPhase>,
    },
    Phase {
        phase: ModelLoadPhase,
        started: bool,
    },
    Loaded(Box<ModelProperties>),
    Failed(String),
    Lost {
        code: String,
        message: String,
    },
}

enum RequestReply {
    Admitted(u64),
    Event(InferenceObservation),
    Completed(InferenceCompletion),
    Failed(WireInferenceError),
    Template(Result<PreparedChatInfo, WireInferenceError>),
    TokenCount(Result<u64, WireInferenceError>),
    ModelInstance(Result<icn_engine::ModelInstanceObservation, String>),
}

struct ClientInner {
    commands: mpsc::SyncSender<HostMessage>,
    next_request_id: AtomicU64,
    pending: Mutex<HashMap<u64, mpsc::SyncSender<RequestReply>>>,
    load_events: tokio::sync::mpsc::Sender<LoadEvent>,
    failed: AtomicBool,
}

impl ClientInner {
    fn send(&self, message: HostMessage) -> Result<(), InferenceError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(InferenceError::ExecutorStopped);
        }
        self.commands
            .send(message)
            .map_err(|_| InferenceError::ExecutorStopped)
    }

    fn fail_all(&self, code: &str, reason: impl Into<String>) {
        if self.failed.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.commands.try_send(HostMessage::Shutdown);
        let reason = reason.into();
        let _ = self.load_events.try_send(LoadEvent::Lost {
            code: code.to_owned(),
            message: reason.clone(),
        });
        if let Ok(mut pending) = self.pending.lock() {
            for (_, sender) in pending.drain() {
                let _ = sender.try_send(RequestReply::Failed(WireInferenceError::Backend(
                    reason.clone(),
                )));
            }
        }
    }

    fn stop_all_executions(&self) {
        if self.failed.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.commands.try_send(HostMessage::Shutdown);
        let _ = self.load_events.try_send(LoadEvent::Lost {
            code: "model_instance_stopped".to_owned(),
            message: "model instance was stopped".to_owned(),
        });
        if let Ok(mut pending) = self.pending.lock() {
            for (_, sender) in pending.drain() {
                let _ = sender.try_send(RequestReply::Failed(
                    WireInferenceError::ModelInstanceStopped,
                ));
            }
        }
    }
}

#[derive(Clone)]
struct ClientHandle {
    inner: Arc<ClientInner>,
}

impl ClientHandle {
    fn request(
        &self,
        message: impl FnOnce(u64) -> HostMessage,
    ) -> Result<RequestWaiter, InferenceError> {
        let request_id = self.inner.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::sync_channel(REQUEST_EVENT_CAPACITY);
        {
            let mut pending = self
                .inner
                .pending
                .lock()
                .map_err(|_| InferenceError::ExecutorStopped)?;
            if pending.len() >= MAX_PENDING_REQUESTS {
                return Err(InferenceError::Overloaded);
            }
            pending.insert(request_id, sender);
        }
        if let Err(error) = self.inner.send(message(request_id)) {
            if let Ok(mut pending) = self.inner.pending.lock() {
                pending.remove(&request_id);
            }
            return Err(error);
        }
        Ok(RequestWaiter {
            request_id,
            receiver,
            client: self.clone(),
        })
    }

    fn cancel(&self, request_id: u64) {
        let _ = self.inner.send(HostMessage::Cancel { request_id });
    }
}

struct RequestWaiter {
    request_id: u64,
    receiver: mpsc::Receiver<RequestReply>,
    client: ClientHandle,
}

impl Drop for RequestWaiter {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.client.inner.pending.lock() {
            pending.remove(&self.request_id);
        }
    }
}

#[derive(Clone)]
pub(crate) struct RemoteBackend {
    model_id: String,
    properties: ModelProperties,
    client: ClientHandle,
}

impl RemoteBackend {
    pub(crate) fn observe_model_instance(
        &self,
        policy: CapacityPolicy,
        native_build: String,
        enabled_backends: Vec<String>,
    ) -> Result<icn_engine::ModelInstanceObservation, String> {
        let waiter = self
            .client
            .request(|request_id| HostMessage::ObserveModelInstance {
                request_id,
                policy,
                native_build,
                enabled_backends,
                trace: crate::telemetry::inject_current_trace(),
            })
            .map_err(|error| error.to_string())?;
        match waiter.receiver.recv() {
            Ok(RequestReply::ModelInstance(result)) => result,
            Ok(RequestReply::Failed(error)) => Err(InferenceError::from(error).to_string()),
            Ok(_) => Err("worker returned the wrong model instance response".to_owned()),
            Err(_) => Err("inference worker stopped".to_owned()),
        }
    }
}

impl CompletionBackend for RemoteBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn properties(&self) -> Result<ModelProperties, InferenceError> {
        Ok(self.properties.clone())
    }

    fn apply_template(
        &self,
        request: ResolvedInferenceRequest,
    ) -> Result<PreparedChatInfo, InferenceError> {
        let waiter = self
            .client
            .request(|request_id| HostMessage::ApplyTemplate {
                request_id,
                request,
                trace: crate::telemetry::inject_current_trace(),
            })?;
        match waiter.receiver.recv() {
            Ok(RequestReply::Template(result)) => result.map_err(Into::into),
            Ok(RequestReply::Failed(error)) => Err(error.into()),
            Ok(_) => Err(InferenceError::Backend(
                "worker returned the wrong template response".to_owned(),
            )),
            Err(_) => Err(InferenceError::ExecutorStopped),
        }
    }

    fn count_tokens(&self, request: ResolvedInferenceRequest) -> Result<u64, InferenceError> {
        let waiter = self.client.request(|request_id| HostMessage::CountTokens {
            request_id,
            request,
            trace: crate::telemetry::inject_current_trace(),
        })?;
        match waiter.receiver.recv() {
            Ok(RequestReply::TokenCount(result)) => result.map_err(Into::into),
            Ok(RequestReply::Failed(error)) => Err(error.into()),
            Ok(_) => Err(InferenceError::Backend(
                "worker returned the wrong token-count response".to_owned(),
            )),
            Err(_) => Err(InferenceError::ExecutorStopped),
        }
    }

    fn complete(
        &self,
        request: ResolvedInferenceRequest,
        on_admitted: &mut dyn FnMut(u64) -> Result<(), InferenceError>,
        on_event: &mut dyn FnMut(InferenceObservation) -> Result<(), InferenceError>,
    ) -> Result<InferenceCompletion, InferenceError> {
        let waiter = self.client.request(|request_id| HostMessage::Infer {
            request_id,
            request,
            trace: crate::telemetry::inject_current_trace(),
        })?;
        let mut admitted = false;
        loop {
            match waiter.receiver.recv() {
                Ok(RequestReply::Admitted(prompt_tokens)) => {
                    if admitted {
                        self.client.cancel(waiter.request_id);
                        return Err(InferenceError::Backend(
                            "worker reported inference admission more than once".to_owned(),
                        ));
                    }
                    if let Err(error) = on_admitted(prompt_tokens) {
                        self.client.cancel(waiter.request_id);
                        return Err(error);
                    }
                    admitted = true;
                }
                Ok(RequestReply::Event(event)) => {
                    if !admitted {
                        self.client.cancel(waiter.request_id);
                        return Err(InferenceError::Backend(
                            "worker emitted inference output before admission".to_owned(),
                        ));
                    }
                    if let Err(error) = on_event(event) {
                        self.client.cancel(waiter.request_id);
                        return Err(error);
                    }
                }
                Ok(RequestReply::Completed(completion)) if admitted => return Ok(completion),
                Ok(RequestReply::Completed(_)) => {
                    return Err(InferenceError::Backend(
                        "worker completed inference before admission".to_owned(),
                    ));
                }
                Ok(RequestReply::Failed(error)) => return Err(error.into()),
                Ok(_) => {
                    return Err(InferenceError::Backend(
                        "worker returned the wrong inference response".to_owned(),
                    ));
                }
                Err(_) => return Err(InferenceError::ExecutorStopped),
            }
        }
    }
}

/// Own cleanup from the instant spawn succeeds, including every handshake/IPC setup error.
struct WorkerChild(Child);
impl std::ops::Deref for WorkerChild {
    type Target = Child;
    fn deref(&self) -> &Child {
        &self.0
    }
}
impl std::ops::DerefMut for WorkerChild {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}
impl WorkerChild {
    fn observe_exit(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        #[cfg(unix)]
        {
            self.0.try_wait()
        }
        #[cfg(windows)]
        {
            self.0.try_retirement()
        }
    }
    fn stop(&mut self) -> std::io::Result<std::process::ExitStatus> {
        #[cfg(unix)]
        {
            if self.0.try_wait()?.is_none() {
                self.0.kill()?;
            }
            self.0.wait()
        }
        #[cfg(windows)]
        {
            self.0.retire(std::time::Duration::from_secs(2))
        }
    }
}
impl Drop for WorkerChild {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            tracing::error!(%error, "inference worker cleanup could not prove retirement");
        }
    }
}

enum WorkerProcessState {
    Owned(WorkerChild),
    Retired(std::process::ExitStatus),
}

struct ProcessControl {
    child: Mutex<WorkerProcessState>,
}

impl ProcessControl {
    fn pid(&self) -> Option<u32> {
        self.child.lock().ok().and_then(|state| match &*state {
            WorkerProcessState::Owned(child) => Some(child.id()),
            WorkerProcessState::Retired(_) => None,
        })
    }

    fn try_wait(&self) -> std::io::Result<Option<std::process::ExitStatus>> {
        let mut state = self
            .child
            .lock()
            .map_err(|_| std::io::Error::other("worker process lock poisoned"))?;
        match &mut *state {
            WorkerProcessState::Retired(status) => Ok(Some(*status)),
            WorkerProcessState::Owned(child) => {
                let status = child.observe_exit()?;
                if let Some(status) = status {
                    *state = WorkerProcessState::Retired(status);
                }
                Ok(status)
            }
        }
    }

    fn retire(&self) -> std::io::Result<std::process::ExitStatus> {
        let mut state = self
            .child
            .lock()
            .map_err(|_| std::io::Error::other("worker process lock poisoned"))?;
        match &mut *state {
            WorkerProcessState::Retired(status) => Ok(*status),
            WorkerProcessState::Owned(child) => {
                // An error leaves the exact owner in place. Only proof of retirement replaces it.
                let status = child.stop()?;
                *state = WorkerProcessState::Retired(status);
                Ok(status)
            }
        }
    }

    fn terminate(&self) {
        if let Err(error) = self.retire() {
            tracing::error!(%error, "inference worker retirement failed; retaining its owner");
        }
    }
}

#[derive(Clone)]
pub(crate) struct InferenceWorker {
    client: ClientHandle,
    process: Arc<ProcessControl>,
}

impl InferenceWorker {
    pub(crate) fn spawn(
        generation: u64,
        expected_build: String,
        worker_launcher: &NativeWorkerLauncher,
    ) -> anyhow::Result<(Self, tokio::sync::mpsc::Receiver<LoadEvent>)> {
        #[cfg(unix)]
        let child = {
            let mut command = worker_launcher.command(NativeWorkerRole::Inference)?;
            command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            command.spawn()?
        };
        #[cfg(windows)]
        let child = worker_launcher.spawn_inference()?;
        let mut child = WorkerChild(child);
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("worker stdin was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("worker stdout was not piped"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("worker stderr was not piped"))?;

        let (hello_sender, hello_receiver) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name(format!("icn-worker-handshake-{generation}"))
            .spawn(move || {
                let mut stdout = stdout;
                let hello = read_frame::<WorkerMessage>(&mut stdout);
                let _ = hello_sender.send((hello, stdout));
            })?;
        let (hello, mut stdout) =
            match hello_receiver.recv_timeout(std::time::Duration::from_secs(10 * 60)) {
                Ok((hello, stdout)) => (hello?, stdout),
                Err(_) => {
                    let _ = child.stop();
                    anyhow::bail!("inference worker handshake timed out");
                }
            };
        match hello.message {
            WorkerMessage::Hello { build } if build == expected_build => {}
            WorkerMessage::Hello { build } => {
                let _ = child.stop();
                anyhow::bail!("worker build mismatch: got {build:?}, expected {expected_build:?}");
            }
            _ => {
                let _ = child.stop();
                anyhow::bail!("worker did not begin with a hello frame");
            }
        }
        write_frame(
            &mut stdin,
            generation,
            HostMessage::Hello { expected_build },
        )?;

        let (commands, command_receiver) = mpsc::sync_channel(COMMAND_QUEUE_CAPACITY);
        let (load_events, load_receiver) = tokio::sync::mpsc::channel(LOAD_EVENT_CAPACITY);
        let inner = Arc::new(ClientInner {
            commands,
            next_request_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            load_events,
            failed: AtomicBool::new(false),
        });
        let client = ClientHandle {
            inner: Arc::clone(&inner),
        };
        let process = Arc::new(ProcessControl {
            child: Mutex::new(WorkerProcessState::Owned(child)),
        });

        let writer_inner = Arc::downgrade(&inner);
        thread::Builder::new()
            .name(format!("icn-worker-writer-{generation}"))
            .spawn(move || run_worker_writer(stdin, command_receiver, writer_inner, generation))?;

        let reader_inner = Arc::clone(&inner);
        thread::Builder::new()
            .name(format!("icn-worker-reader-{generation}"))
            .spawn(move || {
                loop {
                    let frame: Frame<WorkerMessage> = match read_frame(&mut stdout) {
                        Ok(frame) => frame,
                        Err(error) => {
                            reader_inner.fail_all(
                                "worker_protocol_error",
                                format!("worker IPC read failed: {error:#}"),
                            );
                            break;
                        }
                    };
                    if frame.generation != generation {
                        reader_inner.fail_all(
                            "worker_protocol_error",
                            format!(
                                "worker generation mismatch: got {}, expected {generation}",
                                frame.generation
                            ),
                        );
                        break;
                    }
                    dispatch_worker_message(&reader_inner, frame.message);
                }
            })?;
        drain_stderr(stderr, generation);

        Ok((Self { client, process }, load_receiver))
    }

    pub(crate) fn start_load(
        &self,
        model_id: String,
        plan: icn_contracts::ExecutionIntent,
        speculative: SpeculativeDecodingConfig,
        hardware: icn_contracts::HardwareSnapshot,
        template_capabilities: TemplateCapabilities,
        reasoning: ReasoningProfile,
        template_fingerprint: String,
        expected_vision: bool,
    ) -> Result<(), InferenceError> {
        self.client.inner.send(HostMessage::Load {
            model_id,
            plan,
            speculative,
            hardware,
            template_capabilities,
            reasoning,
            template_fingerprint,
            expected_vision,
            trace: crate::telemetry::inject_current_trace(),
        })
    }

    pub(crate) fn backend(&self, model_id: String, properties: ModelProperties) -> RemoteBackend {
        RemoteBackend {
            model_id,
            properties,
            client: self.client.clone(),
        }
    }

    pub(crate) fn pid(&self) -> Option<u32> {
        self.process.pid()
    }

    pub(crate) fn try_wait(&self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.process.try_wait()
    }

    pub(crate) fn terminate(&self, code: &str, reason: &str) {
        self.client.inner.fail_all(code, reason.to_owned());
        self.process.terminate();
    }

    pub(crate) fn stop_all_executions(&self) {
        self.client.inner.stop_all_executions();
        self.process.terminate();
    }

    pub(crate) fn shutdown(&self) {
        let _ = self.client.inner.send(HostMessage::Shutdown);
    }
}

fn dispatch_worker_message(inner: &Arc<ClientInner>, message: WorkerMessage) {
    match message {
        WorkerMessage::Prepared {
            acceleration,
            timing_plan_identity,
            phases,
        } => {
            let _ = inner.load_events.blocking_send(LoadEvent::Prepared {
                acceleration,
                timing_plan_identity,
                phases,
            });
        }
        WorkerMessage::LoadPhase { phase, started } => {
            let _ = inner
                .load_events
                .blocking_send(LoadEvent::Phase { phase, started });
        }
        WorkerMessage::Loaded { properties } => {
            let _ = inner
                .load_events
                .blocking_send(LoadEvent::Loaded(Box::new(properties)));
        }
        WorkerMessage::LoadFailed { message } => {
            let _ = inner.load_events.blocking_send(LoadEvent::Failed(message));
        }
        WorkerMessage::InferenceObservation { request_id, event } => {
            send_request_reply(inner, request_id, RequestReply::Event(event), false);
        }
        WorkerMessage::InferenceAdmitted {
            request_id,
            prompt_tokens,
        } => {
            send_request_reply(
                inner,
                request_id,
                RequestReply::Admitted(prompt_tokens),
                false,
            );
        }
        WorkerMessage::RequestCompleted {
            request_id,
            completion,
        } => {
            send_request_reply(inner, request_id, RequestReply::Completed(completion), true);
        }
        WorkerMessage::RequestFailed { request_id, error } => {
            send_request_reply(inner, request_id, RequestReply::Failed(error), true);
        }
        WorkerMessage::TemplateCompleted { request_id, result } => {
            send_request_reply(inner, request_id, RequestReply::Template(result), true);
        }
        WorkerMessage::TokenCountCompleted { request_id, result } => {
            send_request_reply(inner, request_id, RequestReply::TokenCount(result), true);
        }
        WorkerMessage::ModelInstanceObserved { request_id, result } => {
            send_request_reply(inner, request_id, RequestReply::ModelInstance(result), true);
        }
        WorkerMessage::Hello { .. } => {
            inner.fail_all(
                "worker_protocol_error",
                "worker sent an unexpected second hello",
            );
        }
    }
}

fn send_request_reply(
    inner: &Arc<ClientInner>,
    request_id: u64,
    reply: RequestReply,
    terminal: bool,
) {
    let sender = inner.pending.lock().ok().and_then(|mut pending| {
        if terminal {
            pending.remove(&request_id)
        } else {
            pending.get(&request_id).cloned()
        }
    });
    if let Some(sender) = sender {
        let _ = sender.send(reply);
    }
}

fn run_worker_writer(
    mut stdin: impl Write,
    command_receiver: mpsc::Receiver<HostMessage>,
    writer_inner: std::sync::Weak<ClientInner>,
    generation: u64,
) {
    while let Ok(message) = command_receiver.recv() {
        let stopping = matches!(&message, HostMessage::Shutdown);
        if let Err(error) = write_frame(&mut stdin, generation, message) {
            if let Some(inner) = writer_inner.upgrade() {
                inner.fail_all(
                    "worker_protocol_error",
                    format!("worker IPC write failed: {error:#}"),
                );
            }
            break;
        }
        if stopping {
            break;
        }
    }
}

fn drain_stderr(mut stderr: impl Read + Send + 'static, generation: u64) {
    let _ = thread::Builder::new()
        .name(format!("icn-worker-stderr-{generation}"))
        .spawn(move || {
            let mut buffer = [0_u8; 4096];
            loop {
                match stderr.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(length) => {
                        let message = String::from_utf8_lossy(&buffer[..length]);
                        tracing::warn!(worker.generation = generation, %message, "inference worker");
                    }
                }
            }
        });
}

struct IpcLoadObserver {
    responses: mpsc::SyncSender<WorkerMessage>,
}

impl ModelLoadObserver for IpcLoadObserver {
    fn phase_started(&self, phase: ModelLoadPhase) {
        let _ = self.responses.send(WorkerMessage::LoadPhase {
            phase,
            started: true,
        });
    }

    fn phase_completed(&self, phase: ModelLoadPhase) {
        let _ = self.responses.send(WorkerMessage::LoadPhase {
            phase,
            started: false,
        });
    }
}

pub(crate) fn run_worker(build: String, native: NativeBackend) -> anyhow::Result<()> {
    let generation = Arc::new(AtomicU64::new(0));
    let (responses, response_receiver) = mpsc::sync_channel(RESPONSE_QUEUE_CAPACITY);
    let writer_generation = Arc::clone(&generation);
    let writer = thread::Builder::new()
        .name("icn-inference-worker-writer".to_owned())
        .spawn(move || {
            let mut stdout = std::io::stdout().lock();
            while let Ok(message) = response_receiver.recv() {
                let generation = writer_generation.load(Ordering::Acquire);
                if write_frame(&mut stdout, generation, message).is_err() {
                    std::process::exit(1);
                }
            }
        })?;
    responses.send(WorkerMessage::Hello {
        build: build.clone(),
    })?;

    let mut stdin = std::io::stdin().lock();
    let hello: Frame<HostMessage> = read_frame(&mut stdin)?;
    let worker_generation = hello.generation;
    match hello.message {
        HostMessage::Hello { expected_build } if expected_build == build => {
            generation.store(worker_generation, Ordering::Release);
        }
        HostMessage::Hello { expected_build } => {
            anyhow::bail!("host expected build {expected_build:?}, worker build is {build:?}");
        }
        _ => anyhow::bail!("host did not begin with a hello frame"),
    }

    let load: Frame<HostMessage> = read_frame(&mut stdin)?;
    anyhow::ensure!(
        load.generation == worker_generation,
        "load generation mismatch"
    );
    let HostMessage::Load {
        model_id,
        plan,
        speculative,
        hardware,
        template_capabilities,
        reasoning,
        template_fingerprint,
        expected_vision,
        trace,
    } = load.message
    else {
        anyhow::bail!("first worker command was not load");
    };

    let load_span = tracing::info_span!("icn.worker.load", model.id = %model_id);
    crate::telemetry::set_parent_from_carrier(&load_span, &trace);
    let _load_entered = load_span.enter();
    let prepared = match native.prepare_load(
        model_id.clone(),
        plan,
        speculative,
        hardware,
        template_capabilities,
        reasoning,
        template_fingerprint,
        expected_vision,
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            let _ = responses.send(WorkerMessage::LoadFailed {
                message: error.to_string(),
            });
            drop(responses);
            let _ = writer.join();
            return Ok(());
        }
    };
    responses.send(WorkerMessage::Prepared {
        acceleration: prepared.acceleration().to_owned(),
        timing_plan_identity: prepared.timing_plan_identity().to_owned(),
        phases: prepared.phases().to_vec(),
    })?;
    let observer = Arc::new(IpcLoadObserver {
        responses: responses.clone(),
    });
    let backend = match prepared.execute(observer) {
        Ok(backend) => Arc::new(backend),
        Err(error) => {
            let _ = responses.send(WorkerMessage::LoadFailed {
                message: error.to_string(),
            });
            drop(responses);
            let _ = writer.join();
            return Ok(());
        }
    };
    let properties = match backend.properties() {
        Ok(properties) => properties,
        Err(error) => {
            let _ = responses.send(WorkerMessage::LoadFailed {
                message: error.to_string(),
            });
            drop(responses);
            let _ = writer.join();
            return Ok(());
        }
    };
    responses.send(WorkerMessage::Loaded { properties })?;

    let cancellations = Arc::new(Mutex::new(HashMap::<u64, Arc<AtomicBool>>::new()));
    loop {
        let frame: Frame<HostMessage> = match read_frame(&mut stdin) {
            Ok(frame) => frame,
            Err(_) => break,
        };
        if frame.generation != worker_generation {
            break;
        }
        match frame.message {
            HostMessage::Infer {
                request_id,
                request,
                trace,
            } => {
                let cancelled = Arc::new(AtomicBool::new(false));
                if let Ok(mut active) = cancellations.lock() {
                    active.insert(request_id, Arc::clone(&cancelled));
                }
                let backend = Arc::clone(&backend);
                let responses = responses.clone();
                let cancellations = Arc::clone(&cancellations);
                thread::spawn(move || {
                    let span =
                        tracing::info_span!("icn.worker.inference", worker.request.id = request_id);
                    crate::telemetry::set_parent_from_carrier(&span, &trace);
                    let _entered = span.enter();
                    let result = backend.complete(
                        request,
                        &mut |prompt_tokens| {
                            if cancelled.load(Ordering::Acquire) {
                                return Err(InferenceError::Cancelled);
                            }
                            responses
                                .send(WorkerMessage::InferenceAdmitted {
                                    request_id,
                                    prompt_tokens,
                                })
                                .map_err(|_| InferenceError::ExecutorStopped)
                        },
                        &mut |event| {
                            if cancelled.load(Ordering::Acquire) {
                                return Err(InferenceError::Cancelled);
                            }
                            responses
                                .send(WorkerMessage::InferenceObservation { request_id, event })
                                .map_err(|_| InferenceError::ExecutorStopped)
                        },
                    );
                    if let Ok(mut active) = cancellations.lock() {
                        active.remove(&request_id);
                    }
                    let message = match result {
                        Ok(completion) => WorkerMessage::RequestCompleted {
                            request_id,
                            completion,
                        },
                        Err(error) => WorkerMessage::RequestFailed {
                            request_id,
                            error: error.into(),
                        },
                    };
                    let _ = responses.send(message);
                });
            }
            HostMessage::ApplyTemplate {
                request_id,
                request,
                trace,
            } => {
                let backend = Arc::clone(&backend);
                let responses = responses.clone();
                thread::spawn(move || {
                    let span =
                        tracing::info_span!("icn.worker.template", worker.request.id = request_id);
                    crate::telemetry::set_parent_from_carrier(&span, &trace);
                    let _entered = span.enter();
                    let result = backend.apply_template(request).map_err(Into::into);
                    let _ = responses.send(WorkerMessage::TemplateCompleted { request_id, result });
                });
            }
            HostMessage::CountTokens {
                request_id,
                request,
                trace,
            } => {
                let backend = Arc::clone(&backend);
                let responses = responses.clone();
                thread::spawn(move || {
                    let span = tracing::info_span!(
                        "icn.worker.count_tokens",
                        worker.request.id = request_id
                    );
                    crate::telemetry::set_parent_from_carrier(&span, &trace);
                    let _entered = span.enter();
                    let result = backend.count_tokens(request).map_err(Into::into);
                    let _ =
                        responses.send(WorkerMessage::TokenCountCompleted { request_id, result });
                });
            }
            HostMessage::ObserveModelInstance {
                request_id,
                policy,
                native_build,
                enabled_backends,
                trace,
            } => {
                let backend = Arc::clone(&backend);
                let responses = responses.clone();
                thread::spawn(move || {
                    let span = tracing::info_span!(
                        "icn.worker.model_instance_observation",
                        worker.request.id = request_id
                    );
                    crate::telemetry::set_parent_from_carrier(&span, &trace);
                    let _entered = span.enter();
                    let result = backend
                        .observe_model_instance(policy, native_build, enabled_backends)
                        .map_err(|error| error.to_string());
                    let _ =
                        responses.send(WorkerMessage::ModelInstanceObserved { request_id, result });
                });
            }
            HostMessage::Cancel { request_id } => {
                if let Ok(active) = cancellations.lock()
                    && let Some(cancelled) = active.get(&request_id)
                {
                    cancelled.store(true, Ordering::Release);
                }
            }
            HostMessage::Shutdown => break,
            HostMessage::Hello { .. } | HostMessage::Load { .. } => break,
        }
    }
    drop(backend);
    drop(responses);
    let _ = writer.join();
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    #[test]
    fn startup_error_retires_and_reaps_the_acquired_worker() {
        let child = std::process::Command::new("/bin/sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let pid = child.id() as libc::pid_t;
        let fail_startup = || -> anyhow::Result<()> {
            let _owned = super::WorkerChild(child);
            anyhow::bail!("simulated handshake failure")
        };
        assert!(fail_startup().is_err());
        // SAFETY: this only observes whether the exact spawned child still needs reaping.
        assert_eq!(
            unsafe { libc::waitpid(pid, std::ptr::null_mut(), libc::WNOHANG) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
    }
    use super::*;

    fn test_client() -> (ClientHandle, mpsc::Receiver<HostMessage>) {
        let (commands, command_receiver) = mpsc::sync_channel(COMMAND_QUEUE_CAPACITY);
        let (load_events, _load_receiver) = tokio::sync::mpsc::channel(LOAD_EVENT_CAPACITY);
        let inner = Arc::new(ClientInner {
            commands,
            next_request_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            load_events,
            failed: AtomicBool::new(false),
        });
        (ClientHandle { inner }, command_receiver)
    }

    #[test]
    fn writer_exits_when_client_drops_or_retained_client_stops() {
        for explicit_stop in [false, true] {
            let (client, commands) = test_client();
            let inner = Arc::downgrade(&client.inner);
            let (finished, observed) = mpsc::channel();
            let writer = thread::spawn(move || {
                run_worker_writer(std::io::sink(), commands, inner, 1);
                finished.send(()).unwrap();
            });
            if explicit_stop {
                client.inner.stop_all_executions();
                observed
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .unwrap();
                // The retained handle cannot keep an idle writer alive after terminal shutdown.
                assert!(client.inner.failed.load(Ordering::Acquire));
            } else {
                drop(client);
                observed
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .unwrap();
            }
            writer.join().unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn process_control_preserves_retirement_proof_and_clears_live_pid() {
        for forced in [true, false] {
            let child = if forced {
                std::process::Command::new("/bin/sleep")
                    .arg("60")
                    .spawn()
                    .unwrap()
            } else {
                std::process::Command::new("/bin/sh")
                    .args(["-c", "exit 7"])
                    .spawn()
                    .unwrap()
            };
            let pid = child.id();
            let process = ProcessControl {
                child: Mutex::new(WorkerProcessState::Owned(WorkerChild(child))),
            };
            assert_eq!(process.pid(), Some(pid));
            let status = if forced {
                assert!(process.try_wait().unwrap().is_none());
                process.retire().unwrap()
            } else {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                loop {
                    if let Some(status) = process.try_wait().unwrap() {
                        break status;
                    }
                    assert!(std::time::Instant::now() < deadline, "child did not exit");
                    thread::sleep(std::time::Duration::from_millis(5));
                }
            };
            assert_eq!(process.pid(), None);
            assert_eq!(process.try_wait().unwrap(), Some(status));
            assert_eq!(process.retire().unwrap(), status);
            if !forced {
                assert_eq!(status.code(), Some(7));
            }
            assert_eq!(
                unsafe { libc::waitpid(pid as libc::pid_t, std::ptr::null_mut(), libc::WNOHANG) },
                -1
            );
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ECHILD)
            );
        }
    }

    #[test]
    fn terminal_failure_and_explicit_stop_wake_an_idle_writer_once() {
        for explicit_stop in [false, true] {
            let (client, commands) = test_client();
            if explicit_stop {
                client.inner.stop_all_executions();
            } else {
                client.inner.fail_all("worker_lost", "test failure");
            }
            assert!(matches!(commands.try_recv(), Ok(HostMessage::Shutdown)));
            client.inner.fail_all("worker_lost", "repeated failure");
            assert!(commands.try_recv().is_err());
        }
    }

    #[test]
    fn frame_round_trip_and_length_limit() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, 7, HostMessage::Cancel { request_id: 11 }).unwrap();
        let frame: Frame<HostMessage> = read_frame(&mut bytes.as_slice()).unwrap();
        assert_eq!(frame.generation, 7);
        assert!(matches!(
            frame.message,
            HostMessage::Cancel { request_id: 11 }
        ));

        let mut oversized = ((MAX_FRAME_BYTES as u32) + 1).to_be_bytes().to_vec();
        oversized.extend_from_slice(b"{}");
        assert!(read_frame::<HostMessage>(&mut oversized.as_slice()).is_err());
    }

    #[test]
    fn frame_rejects_protocol_version_mismatch_before_dispatch() {
        let payload = serde_json::to_vec(&Frame {
            protocol_version: PROTOCOL_VERSION + 1,
            generation: 1,
            message: HostMessage::Shutdown,
        })
        .unwrap();
        let mut bytes = (payload.len() as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(&payload);
        assert!(read_frame::<HostMessage>(&mut bytes.as_slice()).is_err());
    }

    #[test]
    fn explicit_worker_stop_terminalizes_active_requests_as_model_instance_stopped() {
        let (client, _command_receiver) = test_client();
        let inner = Arc::clone(&client.inner);
        let waiter = client
            .request(|request_id| HostMessage::Cancel { request_id })
            .expect("request must be admitted");

        inner.stop_all_executions();

        assert!(matches!(
            waiter.receiver.recv(),
            Ok(RequestReply::Failed(
                WireInferenceError::ModelInstanceStopped
            ))
        ));
        assert!(inner.failed.load(Ordering::Acquire));
    }

    #[test]
    fn worker_admission_is_forwarded_to_the_matching_request() {
        let (client, _command_receiver) = test_client();
        let inner = Arc::clone(&client.inner);
        let waiter = client
            .request(|request_id| HostMessage::Cancel { request_id })
            .expect("request must be registered");

        dispatch_worker_message(
            &inner,
            WorkerMessage::InferenceAdmitted {
                request_id: waiter.request_id,
                prompt_tokens: 37,
            },
        );

        assert!(matches!(
            waiter.receiver.recv(),
            Ok(RequestReply::Admitted(37))
        ));
    }
}
