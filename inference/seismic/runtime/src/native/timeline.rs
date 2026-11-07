//! Host timelines: a monotonic value one device's native work and the host
//! both advance and wait on.
//!
//! A sealed graph carries timeline nodes between its kernel nodes
//! (`NativeGraphDraft::wait_timeline`, `signal_timeline`) and each run binds
//! every node to a timeline and a value:
//!
//! - **wait** (the host gate): the device runs nothing after the node until
//!   the timeline reaches the value, which the host sets
//!   (`TimelineHost::<Gate>::signal`) once what the work after the gate
//!   reads is in place;
//! - **signal**: the timeline reaches the value once the device has completed
//!   the work before the node, its writes visible to the host
//!   (`TimelineHost::<Reached>::wait`).
//!
//! Where the device's side is its own (every backend but the CPU), a host
//! wait on what the host raised would watch a side nothing raises, and a
//! device wait on what the device raises would never open. So a timeline's
//! role ([`TimelineRole`]: [`Gate`], [`Reached`], [`Parks`]) is its type:
//! the host handle has only that role's operations, and a node's port binds
//! only a timeline of its role ([`TimelineNodeKind`]).
//!
//! Per backend:
//!
//! | Backend | Timeline | Wait | Signal |
//! |---|---|---|---|
//! | Metal | two `MTLSharedEvent`s: the device's, which only command buffers signal, and the host's gate, which only the host signals | the command buffer after the node begins with `encodeWaitForEvent` on the gate | `encodeSignalEvent` on the device's |
//! | CUDA | two 32-bit words in mapped pinned host memory: the device's, which only the device's stream writes, and the host's gate, which only the host writes | `cuStreamWaitValue32` on the gate (cyclic `>=`) | `cuStreamWriteValue32` on the device's |
//! | Vulkan | two timeline semaphores: the device's, which only the device raises, and the host's gate, which only the host raises | the queue submission after the node waits on the gate; the first one whose gate is not raised goes to the driver, which starts it as the host raises the gate, and every later one is held on the host until then (the driver holds one wait on the host, with nothing behind it) | the queue submission before the node signals the device's |
//! | CPU | a value under a mutex | the run blocks before the node's launches | set when the launches before it return |
//!
//! Values only grow. CUDA compares the low 32 bits cyclically, so a wait or
//! a reached value is exact while the values in flight lie within 2^31 of
//! one another.
//!
//! A wait never outlives the host's means to answer it. The host signals a
//! timeline through its handles ([`TimelineHost`]); when the last is dropped
//! (a load's teardown, normal, by error or by unwinding), the timeline
//! records the value it had reached and is raised past every value a bound
//! wait node awaits, so every gate opens. The work behind a gate opened so
//! reads what the host never put in place, and is no result: the completion
//! of each run with such a wait fails ([`TimelineInner::released_unsignalled`]).
//! A process that dies is no teardown: Metal documents no release of an
//! encoded event wait whose signaller is gone (`encodeWaitForEvent` pauses
//! the command buffer until the value is reached; only the driver's
//! `MTLCommandBufferErrorTimeout` aborts work, after a time it does not
//! state).

use crate::api::device::DeviceInner;
use crate::api::TensorError;
use crate::backends::OpenedKind;
use seismic_compiler::errors::ExecutionError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

/// One device's host timeline.
pub(crate) struct TimelineInner {
    device: Arc<DeviceInner>,
    kind: TimelineKind,
    /// Whether a host handle lives, and what the gates bound on the timeline
    /// await. A binding and the release take this one lock, so a wait bound
    /// after (or while) the release is opened by it all the same.
    gates: Mutex<Gates>,
    /// The device's refusal of a release's signal or of its query of the
    /// gates (it is lost): the gates it bound stay shut, or none reads as
    /// the host's signal.
    release_refused: OnceLock<ExecutionError>,
    /// The value the next park bound on the timeline takes when its run is
    /// queued ([`Self::next_park`]).
    parks: AtomicU64,
    /// Held across each host signal's read and raise, so signals from any
    /// threads at once only ever raise the timeline: none lowers what
    /// another raised past it (a CUDA gate's store, a Metal gate's value),
    /// nor signals a Vulkan semaphore to less than it holds, which
    /// `vkSignalSemaphore` forbids.
    raising: Mutex<()>,
}

pub use seismic_native_target::{AnsweredBy, ParkAnswer};

/// The release state of a timeline's gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gates {
    /// A host handle lives; `awaited` is the highest value a bound wait node
    /// awaits.
    Held { awaited: u64 },
    /// The last host handle was dropped when the timeline had reached
    /// `from` (`None` where the device could not say: it is lost, and no
    /// gate reads as the host's signal); every wait bound since is opened
    /// as it is bound.
    Released { from: Option<u64> },
}

mod sealed {
    pub trait Sealed {}
}

/// Who raises a timeline and who waits on it: the operations a host handle
/// ([`TimelineHost`]) has and the nodes it binds to follow from its role,
/// so a host wait on what only the host raises (on Metal, CUDA and Vulkan a
/// wait on the device's side, which never moves), or a device wait on what
/// only the device raises, cannot be written.
pub trait TimelineRole: sealed::Sealed + Send + Sync + 'static {}

/// The host raises the timeline ([`TimelineHost::signal`]) and device work
/// waits on it at wait nodes ([`Wait`]): a host gate. The host can see what
/// it raised ([`TimelineHost::signalled`]), never anything of the device's.
pub enum Gate {}

/// Device work raises the timeline at signal nodes ([`Signal`]) and the host
/// waits on it ([`TimelineHost::wait`]); the host raises it only in the
/// device's place, for work that will never run ([`TimelineHost::reach`]).
pub enum Reached {}

/// The timeline of a load's parks and publications ([`Park`], [`Publish`]):
/// the runtime takes its values as runs are queued; the device reaches each
/// park ([`TimelineHost::reached`]) and the host answers it
/// ([`TimelineHost::answer`]).
pub enum Parks {}

impl sealed::Sealed for Gate {}
impl sealed::Sealed for Reached {}
impl sealed::Sealed for Parks {}
impl TimelineRole for Gate {}
impl TimelineRole for Reached {}
impl TimelineRole for Parks {}

/// A role whose nodes each run binds at a value of the host's
/// (`NativeGraphBindings::set_timeline`).
pub trait ValuedRole: TimelineRole {}
impl ValuedRole for Gate {}
impl ValuedRole for Reached {}

/// A kind of timeline node a graph places (`NativeGraphDraft::timeline`),
/// and the role of the timeline each run binds it to.
pub trait TimelineNodeKind: sealed::Sealed + Copy {
    const NODE: TimelineNode;
    type Role: TimelineRole;
}

/// A wait node ([`TimelineNode::Wait`]), bound to a [`Gate`].
#[derive(Clone, Copy, Debug)]
pub struct Wait;
/// A signal node ([`TimelineNode::Signal`]), bound to a [`Reached`].
#[derive(Clone, Copy, Debug)]
pub struct Signal;
/// A park node ([`TimelineNode::Park`]), bound to a [`Parks`] timeline or
/// answered.
#[derive(Clone, Copy, Debug)]
pub struct Park;
/// A publication node ([`TimelineNode::Publish`]), bound to a [`Parks`]
/// timeline.
#[derive(Clone, Copy, Debug)]
pub struct Publish;

impl sealed::Sealed for Wait {}
impl sealed::Sealed for Signal {}
impl sealed::Sealed for Park {}
impl sealed::Sealed for Publish {}
impl TimelineNodeKind for Wait {
    const NODE: TimelineNode = TimelineNode::Wait;
    type Role = Gate;
}
impl TimelineNodeKind for Signal {
    const NODE: TimelineNode = TimelineNode::Signal;
    type Role = Reached;
}
impl TimelineNodeKind for Park {
    const NODE: TimelineNode = TimelineNode::Park;
    type Role = Parks;
}
impl TimelineNodeKind for Publish {
    const NODE: TimelineNode = TimelineNode::Publish;
    type Role = Parks;
}

/// The host's handle of a timeline of role `R`, which has only the
/// operations of its role. Dropping the last releases it: every wait bound
/// on it opens (the module documentation).
pub struct TimelineHost<R: TimelineRole> {
    inner: Arc<TimelineInner>,
    role: PhantomData<fn() -> R>,
}

impl<R: TimelineRole> TimelineHost<R> {
    /// A timeline at 0 on `device`.
    pub fn new(device: &Arc<DeviceInner>) -> Result<Self, TensorError> {
        Ok(Self {
            inner: Arc::new(TimelineInner::new(device)?),
            role: PhantomData,
        })
    }

    pub(crate) fn inner(&self) -> &Arc<TimelineInner> {
        &self.inner
    }
}

impl TimelineHost<Gate> {
    /// Raise the gate to `value`, after every host write before it; a value
    /// it has reached already changes nothing. Any thread may signal at any
    /// time.
    pub fn signal(&self, value: u64) -> Result<(), ExecutionError> {
        self.inner.signal(value)
    }

    /// Whether the host has raised the gate to `value`; the device's loss
    /// where it cannot say.
    pub fn signalled(&self, value: u64) -> Result<bool, ExecutionError> {
        Ok(self.inner.covers(self.inner.gated()?, value))
    }

    /// Wait at most `timeout` for the host to raise the gate to `value`,
    /// after which this thread sees every host write before that signal;
    /// whether it did, or the device's loss where it cannot say.
    pub fn wait_signalled(&self, value: u64, timeout: Duration) -> Result<bool, ExecutionError> {
        self.inner.wait_gated(value, timeout)
    }
}

impl TimelineHost<Reached> {
    /// Whether the device has raised the timeline to `value`; the device's
    /// loss where it cannot say.
    pub fn reached(&self, value: u64) -> Result<bool, ExecutionError> {
        self.inner.reached(value)
    }

    /// The value the device has raised the timeline to (a word timeline's,
    /// its word); the device's loss where it cannot say.
    pub fn current(&self) -> Result<u64, ExecutionError> {
        self.inner.current()
    }

    /// Wait at most `timeout` for the device to raise the timeline to
    /// `value`, after which the host sees every device write before the
    /// signal; whether it did, or the device's loss where it cannot say.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<bool, ExecutionError> {
        self.inner.wait(value, timeout)
    }

    /// Raise the timeline to `value` in the device's place: the host
    /// completes a signal no device work will raise (a step queued but
    /// never run), or plays the device (a test). Precondition: no device
    /// signal of the timeline is pending, so none raises it at once.
    pub fn reach(&self, value: u64) -> Result<(), ExecutionError> {
        self.inner.reach(value)
    }
}

impl TimelineHost<Parks> {
    /// Whether the device reached the park or publication that took `value`
    /// (`NativeGraphOutputs::park`), every write before it visible to the
    /// host. A park's call waits there unless the host answered it
    /// ([`Self::answered`]), which may come first. The device's loss where
    /// it cannot say.
    pub fn reached(&self, value: u64) -> Result<bool, ExecutionError> {
        self.inner.reached(value)
    }

    /// The value the device has raised the timeline to; the device's loss
    /// where it cannot say.
    pub fn current(&self) -> Result<u64, ExecutionError> {
        self.inner.current()
    }

    /// Raise what the device raises to `value`, as the device would: a host
    /// that plays the device (a test).
    pub fn reach(&self, value: u64) -> Result<(), ExecutionError> {
        self.inner.reach(value)
    }

    /// Whether the timeline's device's kernels wait for a park's answer
    /// themselves (`BackendName::parks_in_kernels`), so a park binds no
    /// operation of it.
    pub fn parks_in_kernels(&self) -> bool {
        self.inner.parks_in_kernels()
    }

    /// Answer the park that took `value`, every host write before this
    /// call visible to the call: the timeline's gates raised to
    /// `value + 1`, the call's gate, before or after the device reaches it.
    /// `answer` is why the host answers now, which a trace keeps.
    pub fn answer(&self, value: u64, answer: ParkAnswer) -> Result<(), ExecutionError> {
        self.inner.answer(value, answer)
    }

    /// Whether the host answered the park that took `value`: its gate
    /// raised to `value + 1` by the host's signal, not opened by the
    /// timeline's release ([`TimelineInner::released_unsignalled`]). The
    /// device's loss where it cannot say, or where the release could not
    /// say what the host reached.
    pub fn answered(&self, value: u64) -> Result<bool, ExecutionError> {
        self.inner.answered(value)
    }
}

impl<R: TimelineRole> Drop for TimelineHost<R> {
    fn drop(&mut self) {
        self.inner.release();
    }
}

/// A device timeline's two sides: what the device raises (signal nodes, a
/// park's arrival, a publication) and the host waits on, and the gate the
/// host raises (wait nodes, a park's answer) and the device waits on. They
/// are apart so no device signal can ever satisfy a gate or lower one the
/// host raised past it: a wait whose value only the work behind it would
/// reach is unrepresentable.
pub(crate) struct TimelineSides<R, G = R> {
    pub(crate) reached: R,
    pub(crate) gate: G,
}

pub(crate) enum TimelineKind {
    Cpu(CpuTimeline),
    #[cfg(target_os = "macos")]
    Metal(TimelineSides<seismic_metal::SharedEvent>),
    Cuda(TimelineSides<seismic_cuda::direct::HostWord>),
    #[cfg(not(target_os = "macos"))]
    Vulkan(VulkanTimeline),
}

/// A Vulkan timeline's two semaphores.
#[cfg(not(target_os = "macos"))]
pub(crate) type VulkanTimeline = TimelineSides<
    seismic_vulkan::direct::HostTimeline<seismic_vulkan::direct::Reached>,
    seismic_vulkan::direct::HostTimeline<seismic_vulkan::direct::Gate>,
>;

/// The CPU backend's timeline: its runs execute on the submitting thread,
/// so a wait blocks that thread.
pub(crate) struct CpuTimeline {
    value: Mutex<u64>,
    changed: Condvar,
}

impl CpuTimeline {
    pub(crate) fn signal(&self, value: u64) {
        let mut current = self.value.lock().expect("timeline lock is never poisoned");
        if value > *current {
            *current = value;
            self.changed.notify_all();
        }
    }

    fn value(&self) -> u64 {
        *self.value.lock().expect("timeline lock is never poisoned")
    }

    /// Block until the timeline reaches `value`: a CPU run's host gate.
    pub(crate) fn wait_unbounded(&self, value: u64) {
        let current = self.value.lock().expect("timeline lock is never poisoned");
        let _reached = self
            .changed
            .wait_while(current, |current| *current < value)
            .expect("timeline lock is never poisoned");
    }

    pub(crate) fn wait(&self, value: u64, timeout: Duration) -> bool {
        let current = self.value.lock().expect("timeline lock is never poisoned");
        let (current, _) = self
            .changed
            .wait_timeout_while(current, timeout, |current| *current < value)
            .expect("timeline lock is never poisoned");
        *current >= value
    }
}

/// Whether 32-bit word `word` has reached `value`, compared as CUDA's
/// `CU_STREAM_WAIT_VALUE_GEQ` compares.
fn word_reached(word: u32, value: u64) -> bool {
    (word.wrapping_sub(value as u32) as i32) >= 0
}

impl TimelineInner {
    /// A timeline at 0 on `device`.
    pub(crate) fn new(device: &Arc<DeviceInner>) -> Result<Self, TensorError> {
        let kind = match &device.kind {
            OpenedKind::Cpu(_) => TimelineKind::Cpu(CpuTimeline {
                value: Mutex::new(0),
                changed: Condvar::new(),
            }),
            #[cfg(target_os = "macos")]
            OpenedKind::Metal(opened) => TimelineKind::Metal(TimelineSides {
                reached: seismic_metal::SharedEvent::new(opened.service()).map_err(TensorError::Execution)?,
                gate: seismic_metal::SharedEvent::new(opened.service()).map_err(TensorError::Execution)?,
            }),
            OpenedKind::Cuda(opened) => TimelineKind::Cuda(TimelineSides {
                reached: seismic_cuda::direct::HostWord::new(opened.service())
                    .map_err(TensorError::Execution)?,
                gate: seismic_cuda::direct::HostWord::new(opened.service())
                    .map_err(TensorError::Execution)?,
            }),
            #[cfg(not(target_os = "macos"))]
            OpenedKind::Vulkan(opened) => TimelineKind::Vulkan(TimelineSides {
                reached: seismic_vulkan::direct::HostTimeline::new(opened.service())
                    .map_err(TensorError::Execution)?,
                gate: seismic_vulkan::direct::HostTimeline::new(opened.service())
                    .map_err(TensorError::Execution)?,
            }),
        };
        Ok(Self {
            device: device.clone(),
            kind,
            gates: Mutex::new(Gates::Held { awaited: 0 }),
            release_refused: OnceLock::new(),
            parks: AtomicU64::new(1),
            raising: Mutex::new(()),
        })
    }

    pub(crate) fn device(&self) -> &Arc<DeviceInner> {
        &self.device
    }

    pub(crate) fn kind(&self) -> &TimelineKind {
        &self.kind
    }

    /// Raise the timeline to `value` from the host, after every host write
    /// before it; a value it has reached already changes nothing. Any
    /// thread may signal at any time.
    pub(crate) fn signal(&self, value: u64) -> Result<(), ExecutionError> {
        let _raising = self.raising.lock().expect("timeline raising lock is never poisoned");
        match &self.kind {
            TimelineKind::Cpu(timeline) => timeline.signal(value),
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(timeline) => {
                if timeline.gate.value() < value {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                    timeline.gate.signal(value);
                }
            }
            TimelineKind::Cuda(timeline) => {
                if !word_reached(timeline.gate.load(), value) {
                    timeline.gate.store(value as u32);
                }
            }
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                timeline.gate.signal(value)?;
            }
        }
        Ok(())
    }

    /// Raise what the device raises to `value` in the device's place (no
    /// device signal of it pending, [`TimelineHost::reach`]). On the CPU,
    /// whose one value both raise, [`Self::signal`].
    pub(crate) fn reach(&self, value: u64) -> Result<(), ExecutionError> {
        match &self.kind {
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(timeline) => {
                let _raising = self.raising.lock().expect("timeline raising lock is never poisoned");
                if timeline.reached.value() < value {
                    timeline.reached.signal(value);
                }
                Ok(())
            }
            TimelineKind::Cuda(timeline) => {
                let _raising = self.raising.lock().expect("timeline raising lock is never poisoned");
                if !word_reached(timeline.reached.load(), value) {
                    timeline.reached.store(value as u32);
                }
                Ok(())
            }
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => {
                let _raising = self.raising.lock().expect("timeline raising lock is never poisoned");
                timeline.reached.signal(value)
            }
            _ => self.signal(value),
        }
    }

    /// The value a park queued now takes (`NativeGraphBindings::set_park`),
    /// each park two: the device signals it and waits for the next, the
    /// host's answer. Runs are queued in the order the device runs them, so
    /// it reaches the values in order.
    pub(crate) fn next_park(&self) -> u64 {
        self.parks.fetch_add(2, Ordering::Relaxed)
    }

    /// Whether the timeline's device's kernels wait for a park's answer
    /// themselves (`BackendName::parks_in_kernels`), so a park binds no
    /// operation of it.
    pub(crate) fn parks_in_kernels(&self) -> bool {
        crate::native::backend_name(&self.device.kind).parks_in_kernels()
    }

    /// Answer the park that took `value`, every host write before this
    /// call visible to the call: the timeline's gates raised to
    /// `value + 1`, the call's gate, before or after the device reaches it.
    pub(crate) fn answer(&self, value: u64, answer: ParkAnswer) -> Result<(), ExecutionError> {
        match &self.kind {
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => {
                let _raising = self.raising.lock().expect("timeline raising lock is never poisoned");
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                timeline.gate.answer(value + 1, answer)
            }
            _ => {
                let _ = answer;
                self.signal(value + 1)
            }
        }
    }

    /// Whether the host answered the park that took `value`: its gate
    /// raised to `value + 1` by the host's signal, not opened by the
    /// timeline's release ([`Self::released_unsignalled`]). The device's
    /// loss where it cannot say, or where the release could not say what
    /// the host reached.
    pub(crate) fn answered(&self, value: u64) -> Result<bool, ExecutionError> {
        Ok(self.covers(self.signalled()?, value + 1))
    }

    /// The value the host's own signals raised the gates to: while a host
    /// handle lives, the gates' value; once the last is dropped, the value
    /// its release found, never what the release raised them to. Read
    /// under the gates lock, so no release lands between the two. The
    /// device's loss where it cannot say, or where the release could not.
    fn signalled(&self) -> Result<u64, ExecutionError> {
        match *self.gates() {
            Gates::Held { .. } => self.gated(),
            Gates::Released { from: Some(from) } => Ok(from),
            Gates::Released { from: None } => Err(self
                .release_refused
                .get()
                .expect("a release that could not read the gates keeps the device's loss")
                .clone()),
        }
    }

    /// Whether the timeline has reached `value`; the device's loss where it
    /// cannot say.
    pub(crate) fn reached(&self, value: u64) -> Result<bool, ExecutionError> {
        Ok(self.covers(self.current()?, value))
    }

    /// The value the timeline holds now (a word timeline's, its word); the
    /// device's loss where it cannot say.
    pub(crate) fn current(&self) -> Result<u64, ExecutionError> {
        Ok(match &self.kind {
            TimelineKind::Cpu(timeline) => timeline.value(),
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(timeline) => timeline.reached.value(),
            TimelineKind::Cuda(timeline) => u64::from(timeline.reached.load()),
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => timeline.reached.value()?,
        })
    }

    /// The value the host has raised the gates to (where the device's side
    /// is its own, the gate; elsewhere the one timeline's value); the
    /// device's loss where it cannot say.
    fn gated(&self) -> Result<u64, ExecutionError> {
        Ok(match &self.kind {
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(timeline) => timeline.gate.value(),
            TimelineKind::Cuda(timeline) => u64::from(timeline.gate.load()),
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => timeline.gate.value()?,
            _ => return self.current(),
        })
    }

    /// Wait at most `timeout` for the host to raise the gates to `value`,
    /// after which this thread sees every host write before that signal;
    /// whether it did, or the device's loss where it cannot say.
    fn wait_gated(&self, value: u64, timeout: Duration) -> Result<bool, ExecutionError> {
        let gated = match &self.kind {
            TimelineKind::Cpu(timeline) => timeline.wait(value, timeout),
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(timeline) => timeline.gate.wait(value, timeout),
            // `HostWord::store` wakes its waiters.
            TimelineKind::Cuda(timeline) => timeline.gate.wait(|current| word_reached(current, value), timeout),
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => timeline.gate.wait(value, timeout)?,
        };
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        Ok(gated)
    }

    /// Whether a timeline holding `current` has reached `value`.
    fn covers(&self, current: u64, value: u64) -> bool {
        match &self.kind {
            TimelineKind::Cuda(_) => word_reached(current as u32, value),
            _ => current >= value,
        }
    }

    fn gates(&self) -> std::sync::MutexGuard<'_, Gates> {
        self.gates.lock().expect("timeline gates lock is never poisoned")
    }

    /// Open the gates awaiting at most `value`, keeping a refusal.
    fn open(&self, value: u64) {
        if let Err(refused) = self.signal(value) {
            self.release_refused.get_or_init(|| refused);
        }
    }

    /// A run binds a wait node to `value`: the timeline's release raises it
    /// at least that far, and a timeline already released raises it now.
    pub(crate) fn await_at(&self, value: u64) {
        let mut gates = self.gates();
        match &mut *gates {
            Gates::Held { awaited } => *awaited = (*awaited).max(value),
            Gates::Released { .. } => self.open(value),
        }
    }

    /// The last host handle is gone: record the value reached (or the
    /// device's failure to say), then open every wait bound on the
    /// timeline.
    fn release(&self) {
        let mut gates = self.gates();
        let Gates::Held { awaited } = *gates else {
            unreachable!("only the last host handle's drop releases a timeline")
        };
        let from = match self.gated() {
            Ok(from) => Some(from),
            Err(lost) => {
                self.release_refused.get_or_init(|| lost);
                None
            }
        };
        *gates = Gates::Released { from };
        self.open(awaited);
    }

    /// Why a wait on `value` was not reached by the host's signal: its
    /// timeline's release opened it (its work read what the host never put
    /// in place), or the device refused the release's signal or could not
    /// say what the host reached; `None` when the host reached it.
    pub(crate) fn released_unsignalled(&self, value: u64) -> Option<ExecutionError> {
        let Gates::Released { from } = *self.gates() else {
            return None;
        };
        if from.is_some_and(|from| self.covers(from, value)) {
            return None;
        }
        Some(match self.release_refused.get() {
            Some(refused) => refused.clone(),
            None => ExecutionError::SubmissionFailed(
                "a host gate of the work opened unsignalled: its timeline's last host handle was dropped"
                    .into(),
            ),
        })
    }

    /// Wait at most `timeout` for the timeline to reach `value`, after which
    /// the host sees every device write before the signal; whether it did,
    /// or the device's loss where it cannot say.
    pub(crate) fn wait(&self, value: u64, timeout: Duration) -> Result<bool, ExecutionError> {
        let reached = match &self.kind {
            TimelineKind::Cpu(timeline) => timeline.wait(value, timeout),
            #[cfg(target_os = "macos")]
            TimelineKind::Metal(timeline) => timeline.reached.wait(value, timeout),
            // A mapped word: sleep between its writes, each of which wakes
            // its waiters (`HostWord::wait`).
            TimelineKind::Cuda(timeline) => timeline.reached.wait(|current| word_reached(current, value), timeout),
            #[cfg(not(target_os = "macos"))]
            TimelineKind::Vulkan(timeline) => timeline.reached.wait(value, timeout)?,
        };
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        Ok(reached)
    }
}

/// What one timeline node of a graph does in each run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineNode {
    /// The device waits until the timeline reaches the bound value.
    Wait,
    /// The timeline reaches the bound value once the work before completes.
    Signal,
    /// A park: the host answers the call after it (a call that parks, with
    /// an awaited port, `&await`), which waits for the answer. Bound at
    /// `v`, the park takes `v` and `v + 1`. Where the backend's devices see
    /// host writes while work runs (`BackendName::parks_in_kernels`) the
    /// call's kernel waits for the answer itself and the node does nothing.
    /// Elsewhere it is a submission boundary: the timeline reaches `v` once
    /// the work before completes, its writes visible to the host, which
    /// raises it to `v + 1` once its answer is written; the work after
    /// waits for `v + 1`.
    Park,
    /// A publication of the call before it (the node enqueued last), whose
    /// writes the host reads while the device goes on: bound as a park
    /// (`NativeGraphBindings::set_park`), it takes the timeline's next park
    /// value `v` when the run is queued and encodes a launch of that call.
    /// Where the backend's devices make their writes visible to the host
    /// while work runs (`BackendName::parks_in_kernels`) it does nothing.
    /// Elsewhere it is a submission boundary: the timeline reaches `v` once
    /// the work before completes, its writes visible to the host, and
    /// nothing waits, so the work after runs on.
    Publish,
}

/// What one bound timeline operation of a run does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimelineAction {
    /// The device waits until the timeline reaches the value.
    Wait,
    /// The timeline reaches the value once the work before completes.
    Signal,
}

/// A timeline node's operation bound to its timeline and value for one run.
#[derive(Clone)]
pub(crate) struct TimelineOperation {
    pub(crate) action: TimelineAction,
    pub(crate) timeline: Arc<TimelineInner>,
    pub(crate) value: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_is_compared_cyclically() {
        assert!(word_reached(5, 5));
        assert!(word_reached(6, 5));
        assert!(!word_reached(4, 5));
        // Past 2^32 the low words wrap and still compare.
        assert!(word_reached(2, (1u64 << 32) + 1));
        assert!(!word_reached(u32::MAX, (1u64 << 32) + 1));
    }

    #[test]
    fn a_cpu_timeline_wakes_a_waiter() {
        let timeline = Arc::new(CpuTimeline {
            value: Mutex::new(0),
            changed: Condvar::new(),
        });
        assert!(!timeline.wait(1, Duration::from_millis(1)));
        let signaler = timeline.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            signaler.signal(3);
        });
        assert!(timeline.wait(2, Duration::from_secs(10)));
        thread.join().unwrap();
        // A lower value never lowers it.
        timeline.signal(1);
        assert_eq!(timeline.value(), 3);
    }

    /// A gate of a CPU device, whose runs block at a gate as
    /// `CpuTimeline::wait_unbounded` does.
    fn cpu_gate() -> TimelineHost<Gate> {
        let devices = crate::devices::Devices::discover(&[]).unwrap();
        let device = devices
            .open_backend(seismic_lang::registry::BackendName::Cpu)
            .unwrap();
        TimelineHost::new(&device).unwrap()
    }

    #[test]
    fn dropping_the_last_host_handle_opens_a_pending_gate_and_fails_its_wait() {
        let host = Arc::new(cpu_gate());
        let inner = TimelineHost::inner(&host).clone();
        host.signal(1).unwrap();
        // A run's gate awaits 3, and blocks as a CPU run blocks at its gate:
        // the host signals it no further.
        inner.await_at(3);
        let waiter = {
            let inner = inner.clone();
            std::thread::spawn(move || {
                let TimelineKind::Cpu(timeline) = inner.kind() else {
                    unreachable!("a CPU device's timeline")
                };
                timeline.wait_unbounded(3);
            })
        };
        // A second handle keeps it; the last one's drop releases it.
        let other = host.clone();
        drop(host);
        assert!(!other.signalled(3).unwrap());
        drop(other);
        let started = std::time::Instant::now();
        waiter.join().unwrap();
        assert!(started.elapsed() < Duration::from_secs(1));
        // The gate it opened was not reached; one reached before was.
        assert!(inner.released_unsignalled(3).is_some());
        assert_eq!(inner.released_unsignalled(1), None);
    }

    /// A dead host's gates open as failures, never as answers: on Vulkan,
    /// whose gates are their own semaphore, a park answered before the
    /// device reached it stays answered, the release opens the one bound
    /// after it and the completion of its run fails, and no gate the
    /// release raised reads as the device having reached anything.
    #[test]
    fn a_dead_hosts_gates_open_as_failures_never_answers() {
        let devices = crate::devices::Devices::discover(&[]).unwrap();
        let Ok(device) = devices.open_backend(seismic_lang::registry::BackendName::Vulkan) else {
            eprintln!("no Vulkan device: no timeline has gates of its own");
            return;
        };
        let host = TimelineHost::<Parks>::new(&device).unwrap();
        let inner = host.inner().clone();
        // Park 1 is answered before the device reached it; park 3's run is
        // bound, and its host dies before answering it.
        inner.await_at(2);
        inner.answer(1, ParkAnswer { by: AnsweredBy::Ready, park: "test", layer: 0 }).unwrap();
        inner.await_at(4);
        assert!(inner.answered(1).unwrap() && !inner.reached(1).unwrap(), "an answer comes before the device");
        drop(host);
        assert!(inner.covers(inner.gated().unwrap(), 4), "the release opens the gate bound after the answer");
        assert!(inner.answered(1).unwrap() && !inner.answered(3).unwrap(), "a gate the release opened is no answer");
        assert!(!inner.reached(1).unwrap() && !inner.reached(3).unwrap(), "the release raises nothing the device reaches");
        assert_eq!(inner.released_unsignalled(2), None, "the host's answer is an answer");
        assert!(inner.released_unsignalled(4).is_some(), "the gate the release opened fails its run");
    }

    #[test]
    fn a_wait_bound_after_the_release_opens_as_it_is_bound() {
        let host = cpu_gate();
        let inner = host.inner().clone();
        // The last handle is dropped between a run's binding of the wait
        // node and its registration of the awaited value.
        drop(host);
        inner.await_at(4);
        assert!(inner.covers(inner.gated().unwrap(), 4));
        assert!(inner.released_unsignalled(4).is_some());
    }
}
