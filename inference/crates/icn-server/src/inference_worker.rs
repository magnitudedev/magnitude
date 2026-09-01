use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Child, ChildStderr, Command, Stdio};
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

use crate::worker_process::{NativeWorkerLauncher, NativeWorkerRole};

const PROTOCOL_VERSION: u32 = 3;
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
        on_event: &mut dyn FnMut(InferenceObservation) -> Result<(), InferenceError>,
    ) -> Result<InferenceCompletion, InferenceError> {
        let waiter = self.client.request(|request_id| HostMessage::Infer {
            request_id,
            request,
            trace: crate::telemetry::inject_current_trace(),
        })?;
        loop {
            match waiter.receiver.recv() {
                Ok(RequestReply::Event(event)) => {
                    if let Err(error) = on_event(event) {
                        let _ = self.client.inner.send(HostMessage::Cancel {
                            request_id: waiter.request_id,
                        });
                        return Err(error);
                    }
                }
                Ok(RequestReply::Completed(completion)) => return Ok(completion),
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

struct ProcessControl {
    child: Mutex<Option<Child>>,
    #[cfg(windows)]
    job: WindowsJob,
}

impl ProcessControl {
    fn pid(&self) -> Option<u32> {
        self.child
            .lock()
            .ok()
            .and_then(|child| child.as_ref().map(Child::id))
    }

    fn try_wait(&self) -> std::io::Result<Option<std::process::ExitStatus>> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| std::io::Error::other("worker process lock poisoned"))?;
        match child.as_mut() {
            Some(child) => child.try_wait(),
            None => Ok(None),
        }
    }

    fn terminate(&self) {
        #[cfg(windows)]
        self.job.terminate();
        let Ok(mut slot) = self.child.lock() else {
            return;
        };
        let Some(mut child) = slot.take() else {
            return;
        };
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for ProcessControl {
    fn drop(&mut self) {
        let Ok(slot) = self.child.get_mut() else {
            return;
        };
        if let Some(child) = slot.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
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
        let mut command = worker_launcher.command(NativeWorkerRole::Inference)?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        configure_child(&mut command);
        let mut child = command.spawn()?;
        #[cfg(windows)]
        let job = match WindowsJob::assign(&child) {
            Ok(job) => job,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.into());
            }
        };
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
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!("inference worker handshake timed out");
                }
            };
        match hello.message {
            WorkerMessage::Hello { build } if build == expected_build => {}
            WorkerMessage::Hello { build } => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("worker build mismatch: got {build:?}, expected {expected_build:?}");
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
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
            child: Mutex::new(Some(child)),
            #[cfg(windows)]
            job,
        });

        let writer_inner = Arc::clone(&inner);
        thread::Builder::new()
            .name(format!("icn-worker-writer-{generation}"))
            .spawn(move || {
                while let Ok(message) = command_receiver.recv() {
                    if let Err(error) = write_frame(&mut stdin, generation, message) {
                        writer_inner.fail_all(
                            "worker_protocol_error",
                            format!("worker IPC write failed: {error:#}"),
                        );
                        break;
                    }
                }
            })?;

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

#[cfg(unix)]
fn configure_child(command: &mut Command) {
    use std::os::unix::process::CommandExt as _;

    let expected_parent = std::process::id() as libc::pid_t;
    // SAFETY: this closure performs only async-signal-safe libc calls between fork and exec.
    unsafe {
        command.pre_exec(move || {
            let core_limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_CORE, &core_limit) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            #[cfg(target_os = "linux")]
            {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            if libc::getppid() != expected_parent {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "ICN parent exited while the inference worker was spawning",
                ));
            }
            Ok(())
        });
    }
}

#[cfg(windows)]
fn configure_child(_command: &mut Command) {}

#[cfg(windows)]
struct WindowsJob(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl WindowsJob {
    fn assign(child: &Child) -> std::io::Result<Self> {
        use std::mem::{size_of, zeroed};
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };

        // SAFETY: all handles and structure sizes follow the Win32 Job Object API contract.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                let error = std::io::Error::last_os_error();
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(error);
            }
            if AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0 {
                let error = std::io::Error::last_os_error();
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(error);
            }
            Ok(Self(job))
        }
    }

    fn terminate(&self) {
        // SAFETY: the handle remains owned by this object until Drop.
        unsafe {
            windows_sys::Win32::System::JobObjects::TerminateJobObject(self.0, 1);
        }
    }
}

#[cfg(windows)]
impl Drop for WindowsJob {
    fn drop(&mut self) {
        // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE terminates any surviving worker descendants.
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
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

fn drain_stderr(mut stderr: ChildStderr, generation: u64) {
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
    start_parent_watchdog();
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
                    let result = backend.complete(request, &mut |event| {
                        if cancelled.load(Ordering::Acquire) {
                            return Err(InferenceError::Cancelled);
                        }
                        responses
                            .send(WorkerMessage::InferenceObservation { request_id, event })
                            .map_err(|_| InferenceError::ExecutorStopped)
                    });
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

#[cfg(unix)]
fn start_parent_watchdog() {
    // Linux also has PR_SET_PDEATHSIG. Polling covers platforms such as macOS, which do not expose
    // an equivalent inherited kill-on-parent-death primitive.
    let parent = unsafe { libc::getppid() };
    let _ = thread::Builder::new()
        .name("icn-inference-worker-parent-watchdog".to_owned())
        .spawn(move || {
            loop {
                thread::sleep(std::time::Duration::from_millis(100));
                if unsafe { libc::getppid() } != parent {
                    std::process::exit(1);
                }
            }
        });
}

#[cfg(windows)]
fn start_parent_watchdog() {
    // The worker is assigned to a kill-on-close Job Object before the host completes handshake.
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (commands, _command_receiver) = mpsc::sync_channel(COMMAND_QUEUE_CAPACITY);
        let (load_events, _load_receiver) = tokio::sync::mpsc::channel(LOAD_EVENT_CAPACITY);
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
}
