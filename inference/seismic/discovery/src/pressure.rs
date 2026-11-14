//! The OS's memory pressure events, which wake a reading of host RAM: no
//! timer. macOS's memory pressure dispatch source, a Linux PSI trigger
//! (`poll()` on the machine's and the limiting cgroup's PSI files), and
//! Windows' memory resource notifications.

/// The OS's memory pressure events, waited on by one thread and woken by any
/// ([`PressureWaker`]), so the thread that reads host RAM waits for the OS
/// and the engine's own events alike.
pub struct PressureEvents {
    source: platform::Source,
}

/// Wakes a [`PressureEvents::wait`]: what the engine's own events (a flight
/// boundary, one of its own changes) wake a reading with. Wakes before a
/// wait coalesce into one.
#[derive(Clone)]
pub struct PressureWaker {
    waker: platform::Waker,
}

/// What ended a wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wake {
    /// The OS signalled a change of its memory pressure (a wake asked
    /// beside it is answered by the same reading).
    Pressure,
    /// A [`PressureWaker`] asked.
    Asked,
    /// The OS's events ended and could not be set again (Linux: a watched
    /// cgroup was removed and the kernel refuses a trigger on the files a
    /// sample reads now): no pressure wake will come from the OS, only
    /// asked ones. Returned once, as soon as the wait that found it ends.
    Lost,
}

impl PressureEvents {
    /// The OS's pressure events, from now on; none where the OS gives this
    /// process none: Linux without a PSI trigger it may set on every PSI
    /// file a sample reads (a kernel built without PSI or booted `psi=0`, a
    /// trigger refused to a process without `CAP_SYS_RESOURCE` before Linux
    /// 6.4, a `/proc/pressure` or cgroup mounted read-only, a cgroup with
    /// its PSI off). Events from only some of those files would leave
    /// pressure in the others unwatched, so there are all or none.
    ///
    /// # Panics
    /// Where the OS facility fails otherwise, naming it.
    pub fn open() -> Option<Self> {
        platform::Source::open().map(|source| Self { source })
    }

    /// Blocks until the OS signals a change of memory pressure or a waker
    /// asks.
    pub fn wait(&self) -> Wake {
        self.source.wait()
    }

    pub fn waker(&self) -> PressureWaker {
        PressureWaker { waker: self.source.waker() }
    }
}

impl PressureWaker {
    /// Ends the current or next wait.
    pub fn wake(&self) {
        self.waker.wake()
    }
}

#[cfg(windows)]
pub(crate) use platform::low_memory_notification;

/// The window of a Linux PSI trigger: PSI's sampling period (2 s), which an
/// unprivileged trigger's window must be a multiple of.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PSI_WINDOW: std::time::Duration = std::time::Duration::from_secs(2);

/// One reading of a PSI file's cumulative stall time (`some total=`, in
/// microseconds).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StallReading {
    at: std::time::Instant,
    total: u64,
}

/// One PSI trigger's firings as edges, decided from PSI's own cumulative
/// stall time: a trigger fires once per window for as long as any stall
/// persists, and later than that where the kernel evaluates it late (an
/// unprivileged trigger is evaluated from PSI's averaging work, whose
/// latency under memory pressure is not bounded), but a pressure event is
/// the transition from no stall to a stall. So a stall stands from the
/// reading that first saw stall time accrue until a reading a whole window
/// ([`PSI_WINDOW`]) after the last that saw it accrue finds none accrued
/// since: only then is the next firing whose stall time accrued a new
/// stall. However late a firing comes, one with stall time accrued across
/// every window since the last is the same stall; one with none accrued
/// since the last reading is no stall at all.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Default)]
struct StallEdges {
    /// The last reading that saw stall time accrue, or that found a whole
    /// window without any.
    last: Option<StallReading>,
    /// Whether a stall stands: no whole window without stall time has been
    /// read since the last that saw it accrue.
    standing: bool,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
impl StallEdges {
    /// The stall time `total` read at `at`, at a firing of the trigger
    /// (`fired`) or at [`Self::quiet_at`]: whether that is a new stall.
    fn observe(&mut self, at: std::time::Instant, total: u64, fired: bool) -> bool {
        let Some(last) = self.last else {
            // The first reading is at a firing: a stall began within the
            // window before it.
            self.last = Some(StallReading { at, total });
            self.standing = fired;
            return fired;
        };
        let accrued = total > last.total;
        let edge = fired && accrued && !self.standing;
        if accrued {
            self.last = Some(StallReading { at, total });
            self.standing = true;
        } else if at.saturating_duration_since(last.at) >= PSI_WINDOW {
            self.standing = false;
        }
        edge
    }

    /// When a standing stall is read again to learn whether a whole window
    /// passed without stall time: a window after the last reading that saw
    /// it accrue. None while no stall stands, when only a firing is read.
    fn quiet_at(&self) -> Option<std::time::Instant> {
        self.last.filter(|_| self.standing).map(|last| last.at + PSI_WINDOW)
    }
}

/// The cumulative stall time a PSI file's text reports: its `some` line's
/// `total=`, in microseconds.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn stall_total(text: &str) -> Option<u64> {
    text.lines()
        .find_map(|line| line.strip_prefix("some "))?
        .split_whitespace()
        .find_map(|field| field.strip_prefix("total="))?
        .parse()
        .ok()
}

/// What a wait has pending, signalled by a callback on another thread.
#[cfg(target_os = "macos")]
#[derive(Default)]
struct Pending {
    pressure: bool,
    asked: bool,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct Shared {
    pending: std::sync::Mutex<Pending>,
    signal: std::sync::Condvar,
}

#[cfg(target_os = "macos")]
impl Shared {
    fn post(&self, post: impl FnOnce(&mut Pending)) {
        post(&mut self.pending.lock().expect("pressure events' lock is never poisoned"));
        self.signal.notify_all();
    }

    fn wait(&self) -> Wake {
        let mut pending = self.pending.lock().expect("pressure events' lock is never poisoned");
        loop {
            let wake = match (pending.pressure, pending.asked) {
                (true, _) => Wake::Pressure,
                (false, true) => Wake::Asked,
                (false, false) => {
                    pending = self.signal.wait(pending).expect("pressure events' lock is never poisoned");
                    continue;
                }
            };
            *pending = Pending::default();
            return wake;
        }
    }
}

/// macOS: a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source on its own serial
/// queue, for the normal, warn and critical levels, whose handler posts to
/// the waiting thread.
#[cfg(target_os = "macos")]
mod platform {
    use super::{Shared, Wake};
    use std::ffi::c_void;
    use std::sync::Arc;

    #[repr(C)]
    struct SourceType {
        _private: [u8; 0],
    }

    extern "C" {
        static _dispatch_source_type_memorypressure: SourceType;
        fn dispatch_queue_create(label: *const std::ffi::c_char, attributes: *const c_void) -> *mut c_void;
        fn dispatch_source_create(kind: *const SourceType, handle: usize, mask: usize, queue: *mut c_void) -> *mut c_void;
        fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
        fn dispatch_source_set_event_handler_f(source: *mut c_void, handler: extern "C" fn(*mut c_void));
        fn dispatch_source_set_cancel_handler_f(source: *mut c_void, handler: extern "C" fn(*mut c_void));
        fn dispatch_resume(object: *mut c_void);
        fn dispatch_source_cancel(source: *mut c_void);
        fn dispatch_release(object: *mut c_void);
    }

    /// `DISPATCH_MEMORYPRESSURE_NORMAL | WARN | CRITICAL`.
    const LEVELS: usize = 0x1 | 0x2 | 0x4;

    pub(super) struct Source {
        shared: Arc<Shared>,
        source: *mut c_void,
    }

    // SAFETY: a dispatch source is thread-safe; it is only cancelled and
    // released on drop.
    unsafe impl Send for Source {}
    unsafe impl Sync for Source {}

    /// The handler's context: a reference of the shared state, released by
    /// the cancel handler, which runs once no event handler can.
    extern "C" fn posted(context: *mut c_void) {
        // SAFETY: the context is the `Arc<Shared>` the source was given,
        // alive until the cancel handler.
        let shared = unsafe { &*(context as *const Shared) };
        shared.post(|pending| pending.pressure = true);
    }

    extern "C" fn cancelled(context: *mut c_void) {
        // SAFETY: the reference given at open, released once.
        drop(unsafe { Arc::from_raw(context as *const Shared) });
    }

    impl Source {
        pub(super) fn open() -> Option<Self> {
            let shared = Arc::new(Shared::default());
            unsafe {
                let queue = dispatch_queue_create(c"magnitude.memory-pressure".as_ptr(), std::ptr::null());
                assert!(!queue.is_null(), "invariant violated: dispatch_queue_create failed");
                let source =
                    dispatch_source_create(&_dispatch_source_type_memorypressure, 0, LEVELS, queue);
                assert!(!source.is_null(), "invariant violated: dispatch_source_create(memorypressure) failed");
                // The source holds its queue.
                dispatch_release(queue);
                dispatch_set_context(source, Arc::into_raw(shared.clone()) as *mut c_void);
                dispatch_source_set_event_handler_f(source, posted);
                dispatch_source_set_cancel_handler_f(source, cancelled);
                dispatch_resume(source);
                Some(Self { shared, source })
            }
        }

        pub(super) fn wait(&self) -> Wake {
            self.shared.wait()
        }

        pub(super) fn waker(&self) -> Waker {
            Waker(self.shared.clone())
        }
    }

    impl Drop for Source {
        fn drop(&mut self) {
            unsafe {
                dispatch_source_cancel(self.source);
                dispatch_release(self.source);
            }
        }
    }

    #[derive(Clone)]
    pub(super) struct Waker(Arc<Shared>);

    impl Waker {
        pub(super) fn wake(&self) {
            self.0.post(|pending| pending.asked = true);
        }
    }
}

/// Linux: a PSI trigger on the machine's PSI file and on the limiting
/// cgroup's, each firing once some task has stalled on memory within PSI's
/// own two-second sampling period, and again each window the stall
/// persists, and an eventfd for wakes, all polled together. Only a trigger's
/// edge is an event ([`super::StallEdges`]): a standing stall is one. A
/// trigger whose file is gone (its cgroup removed once this process moved
/// out of it, where `poll` reports it in error at once and for good) ends
/// the watch on it: the triggers are set again on the files a sample reads
/// now.
#[cfg(target_os = "linux")]
mod platform {
    use super::{stall_total, StallEdges, Wake, PSI_WINDOW};
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Any stall (1 µs) within a window ([`PSI_WINDOW`]), NUL-terminated as
    /// the kernel parses it.
    fn trigger() -> String {
        format!("some 1 {}\0", PSI_WINDOW.as_micros())
    }

    /// A PSI trigger set on one PSI file, and its edges.
    struct Trigger {
        path: PathBuf,
        file: std::fs::File,
        edges: StallEdges,
    }

    /// A trigger on each PSI file a sample reads now
    /// ([`crate::host::pressure_files`]), or none where the kernel refuses
    /// any of them ([`refused`]): events from only some would leave pressure
    /// in the others unwatched.
    ///
    /// # Panics
    /// Where a trigger cannot be set otherwise.
    fn triggers() -> Option<Vec<Trigger>> {
        crate::host::pressure_files()
            .into_iter()
            .map(|path| {
                let set = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                    .and_then(|mut file| file.write_all(trigger().as_bytes()).map(|()| file));
                match set {
                    Ok(file) => Some(Trigger { path, file, edges: StallEdges::default() }),
                    Err(error) if refused(&error) => None,
                    Err(error) => panic!("invariant violated: a PSI trigger on {} cannot be set: {error}", path.display()),
                }
            })
            .collect()
    }

    /// The cumulative stall time the PSI file `path` reports now; none
    /// where the file is gone (its cgroup removed: `ENOENT`, or `ENODEV`
    /// from a cgroup file read as it is removed).
    ///
    /// # Panics
    /// Where it cannot be read otherwise, or reads no total.
    fn read_total(path: &Path) -> Option<u64> {
        match std::fs::read_to_string(path) {
            Ok(text) => Some(stall_total(&text).unwrap_or_else(|| panic!("invariant violated: {} reads {text:?}", path.display()))),
            Err(error) if matches!(error.raw_os_error(), Some(libc::ENOENT | libc::ENODEV)) => None,
            Err(error) => panic!("invariant violated: {}: {error}", path.display()),
        }
    }

    pub(super) struct Source {
        /// The triggers: the one waiting thread's alone. Empty where the
        /// kernel refused a trigger when the watch was set again.
        triggers: Mutex<Vec<Trigger>>,
        wakes: Arc<OwnedFd>,
        /// The triggers could not be set again since the last wait
        /// returned ([`Wake::Lost`]).
        lost: AtomicBool,
    }

    impl Source {
        pub(super) fn open() -> Option<Self> {
            let triggers = triggers()?;
            let wakes = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            assert!(wakes >= 0, "invariant violated: eventfd failed: {}", std::io::Error::last_os_error());
            // SAFETY: a descriptor just opened, owned here alone.
            Some(Self { triggers: Mutex::new(triggers), wakes: Arc::new(unsafe { OwnedFd::from_raw_fd(wakes) }), lost: AtomicBool::new(false) })
        }

        /// Blocks until a trigger's edge or a wake: a trigger firing again
        /// for a stall that persists is consumed and the wait goes on, and a
        /// standing stall's file is read again a window after it last
        /// accrued ([`StallEdges::quiet_at`]).
        pub(super) fn wait(&self) -> Wake {
            let mut triggers = self.triggers.lock().expect("the PSI triggers' lock is never poisoned");
            loop {
                if self.lost.swap(false, Ordering::AcqRel) {
                    return Wake::Lost;
                }
                let now = Instant::now();
                let timeout = triggers.iter().filter_map(|trigger| trigger.edges.quiet_at()).min().map_or(-1, |quiet| {
                    // Rounded up, so a reading at the timeout is past it.
                    let wait = quiet.saturating_duration_since(now);
                    i32::try_from(wait.as_micros().div_ceil(1000)).expect("a PSI window's milliseconds fit i32")
                });
                let mut polled: Vec<libc::pollfd> = triggers
                    .iter()
                    .map(|trigger| libc::pollfd { fd: trigger.file.as_raw_fd(), events: libc::POLLPRI, revents: 0 })
                    .chain(std::iter::once(libc::pollfd { fd: self.wakes.as_raw_fd(), events: libc::POLLIN, revents: 0 }))
                    .collect();
                let ready = unsafe { libc::poll(polled.as_mut_ptr(), polled.len() as libc::nfds_t, timeout) };
                if ready < 0 {
                    let error = std::io::Error::last_os_error();
                    assert_eq!(error.kind(), std::io::ErrorKind::Interrupted, "invariant violated: poll on PSI triggers failed: {error}");
                    continue;
                }
                let at = Instant::now();
                let (wakes, fired) = polled.split_last().expect("the wake descriptor is polled");
                let asked = wakes.revents & libc::POLLIN != 0;
                if asked {
                    let mut count = 0u64;
                    // Resets the count; another thread's wake since is read
                    // by the next wait.
                    unsafe { libc::read(self.wakes.as_raw_fd(), (&mut count as *mut u64).cast(), 8) };
                }
                let mut edge = false;
                // The files of the triggers that ended.
                let mut gone = Vec::new();
                for (trigger, polled) in triggers.iter_mut().zip(fired) {
                    if polled.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                        gone.push(trigger.path.clone());
                        continue;
                    }
                    // Every firing and every due quiet check passes through
                    // its edges, whatever else woke.
                    let fired = polled.revents & libc::POLLPRI != 0;
                    if fired || trigger.edges.quiet_at().is_some_and(|quiet| quiet <= at) {
                        match read_total(&trigger.path) {
                            Some(total) => edge |= trigger.edges.observe(at, total, fired),
                            None => gone.push(trigger.path.clone()),
                        }
                    }
                }
                if !gone.is_empty() {
                    // The watch on a removed file ended: set it again on the
                    // files a sample reads now. A file still watched goes on
                    // with its stall's edges (its cumulative stall time
                    // continues across the new trigger), so a stall that
                    // stands is not taken for a new one. Where the kernel
                    // refuses a trigger no event comes from the OS again.
                    match self::triggers() {
                        Some(mut fresh) => {
                            for trigger in &mut fresh {
                                if gone.contains(&trigger.path) {
                                    continue;
                                }
                                if let Some(old) = triggers.iter_mut().find(|old| old.path == trigger.path) {
                                    trigger.edges = std::mem::take(&mut old.edges);
                                }
                            }
                            *triggers = fresh;
                        }
                        None => {
                            triggers.clear();
                            self.lost.store(true, Ordering::Release);
                        }
                    }
                }
                match (edge, asked) {
                    (true, _) => return Wake::Pressure,
                    (false, true) => return Wake::Asked,
                    (false, false) => continue,
                }
            }
        }

        pub(super) fn waker(&self) -> Waker {
            Waker(self.wakes.clone())
        }
    }

    /// Whether `error`, opening or writing a PSI trigger, is the kernel
    /// giving this process no trigger: no PSI file (`ENOENT`), PSI off
    /// (`EOPNOTSUPP`), a trigger refused without `CAP_SYS_RESOURCE` before
    /// Linux 6.4 (`EACCES`, `EPERM`), a read-only mount (`EROFS`).
    fn refused(error: &std::io::Error) -> bool {
        matches!(error.raw_os_error(), Some(libc::ENOENT | libc::EOPNOTSUPP | libc::EACCES | libc::EPERM | libc::EROFS))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A trigger whose file is gone reports an error to `poll` at once
        /// and for good (a removed cgroup's kernfs file); here a pipe whose
        /// reader is closed reports one the same way. The wait sets the
        /// watch again on the files a sample reads, and never aborts; where
        /// the kernel refuses a trigger, it reports the watch lost.
        #[test]
        fn an_ended_trigger_is_set_again() {
            let mut ends = [0; 2];
            assert_eq!(unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) }, 0);
            // SAFETY: descriptors just opened, owned here alone.
            let (reader, writer) = unsafe { (OwnedFd::from_raw_fd(ends[0]), std::fs::File::from_raw_fd(ends[1])) };
            drop(reader);
            let gone = PathBuf::from("/sys/fs/cgroup/removed/memory.pressure");
            let wakes = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            assert!(wakes >= 0);
            let source = Source {
                triggers: Mutex::new(vec![Trigger { path: gone.clone(), file: writer, edges: StallEdges::default() }]),
                wakes: Arc::new(unsafe { OwnedFd::from_raw_fd(wakes) }),
                lost: AtomicBool::new(false),
            };
            source.waker().wake();
            assert_eq!(source.wait(), Wake::Asked, "the wake ends the wait the ended trigger was found in");
            let triggers = source.triggers.lock().unwrap();
            assert!(triggers.iter().all(|trigger| trigger.path != gone), "the ended trigger is no longer watched");
        }

        /// Only the kernel's refusals of a trigger leave the process
        /// without events; any other failure is no state of the OS.
        #[test]
        fn only_a_refused_trigger_gives_no_events() {
            for code in [libc::ENOENT, libc::EOPNOTSUPP, libc::EACCES, libc::EPERM, libc::EROFS] {
                assert!(super::refused(&std::io::Error::from_raw_os_error(code)));
            }
            for code in [libc::EINVAL, libc::EBADF, libc::ENOMEM] {
                assert!(!super::refused(&std::io::Error::from_raw_os_error(code)));
            }
        }
    }

    #[derive(Clone)]
    pub(super) struct Waker(Arc<OwnedFd>);

    impl Waker {
        pub(super) fn wake(&self) {
            let one = 1u64;
            let written = unsafe { libc::write(self.0.as_raw_fd(), (&one as *const u64).cast(), 8) };
            // A full count (EAGAIN) already wakes the waiter.
            assert!(
                written == 8 || std::io::Error::last_os_error().kind() == std::io::ErrorKind::WouldBlock,
                "invariant violated: a pressure wake could not be written: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

/// Windows: the low- and high-memory resource notifications, each signalled
/// while its state holds, waited on for the state not holding now, so each
/// wait ends at a change; and an auto-reset event for wakes.
#[cfg(windows)]
mod platform {
    use super::Wake;
    use std::sync::Arc;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Memory::{
        CreateMemoryResourceNotification, HighMemoryResourceNotification, LowMemoryResourceNotification,
        QueryMemoryResourceNotification,
    };
    use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent, WaitForMultipleObjects, INFINITE};

    /// An owned handle.
    struct Handle(HANDLE);

    // SAFETY: kernel handles are process-wide.
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.0) };
        }
    }

    fn notification(kind: i32, name: &str) -> Handle {
        let handle = unsafe { CreateMemoryResourceNotification(kind) };
        assert!(
            !handle.is_null(),
            "invariant violated: CreateMemoryResourceNotification({name}) failed: {}",
            std::io::Error::last_os_error()
        );
        Handle(handle)
    }

    /// The process's low-memory notification, created once: what the host
    /// sample's pressure level queries.
    pub(crate) fn low_memory_notification() -> HANDLE {
        static LOW: std::sync::OnceLock<Handle> = std::sync::OnceLock::new();
        LOW.get_or_init(|| notification(LowMemoryResourceNotification, "low")).0
    }

    fn signalled(handle: HANDLE) -> bool {
        let mut state = 0;
        assert!(
            unsafe { QueryMemoryResourceNotification(handle, &mut state) } != 0,
            "invariant violated: QueryMemoryResourceNotification failed: {}",
            std::io::Error::last_os_error()
        );
        state != 0
    }

    pub(super) struct Source {
        high: Handle,
        wakes: Arc<Handle>,
    }

    impl Source {
        pub(super) fn open() -> Option<Self> {
            let wakes = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
            assert!(!wakes.is_null(), "invariant violated: CreateEventW failed: {}", std::io::Error::last_os_error());
            Some(Self { high: notification(HighMemoryResourceNotification, "high"), wakes: Arc::new(Handle(wakes)) })
        }

        pub(super) fn wait(&self) -> Wake {
            let low = low_memory_notification();
            // Each notification is signalled while its state holds: wait for
            // the states that do not hold now.
            let mut handles = vec![self.wakes.0];
            match (signalled(low), signalled(self.high.0)) {
                (true, _) => handles.push(self.high.0),
                (false, true) => handles.push(low),
                (false, false) => handles.extend([low, self.high.0]),
            }
            let ended = unsafe { WaitForMultipleObjects(handles.len() as u32, handles.as_ptr(), 0, INFINITE) };
            match ended.checked_sub(WAIT_OBJECT_0) {
                Some(0) => Wake::Asked,
                Some(index) if (index as usize) < handles.len() => Wake::Pressure,
                _ => panic!("invariant violated: WaitForMultipleObjects returned {ended:#x}: {}", std::io::Error::last_os_error()),
            }
        }

        pub(super) fn waker(&self) -> Waker {
            Waker(self.wakes.clone())
        }
    }

    #[derive(Clone)]
    pub(super) struct Waker(Arc<Handle>);

    impl Waker {
        pub(super) fn wake(&self) {
            assert!(unsafe { SetEvent(self.0 .0) } != 0, "invariant violated: SetEvent failed: {}", std::io::Error::last_os_error());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// A firing sequence as the waiting thread reads it: each firing, and
    /// each quiet check a window after the last reading that saw stall time
    /// accrue, with the PSI file's `some total=` read at it (µs). One stall
    /// persists through a refire the kernel delivers late (4.8 s after the
    /// last, past a window and its polling period), and is one event; a
    /// whole window without stall time ends it, a firing with nothing
    /// accrued since is none, and the next stall is the next event.
    #[test]
    fn a_standing_stall_is_one_event_however_late_it_refires() {
        let start = Instant::now();
        let ms = |ms: u64| start + Duration::from_millis(ms);
        let mut edges = StallEdges::default();
        // (ms, total, fired, a new stall, when the next quiet check falls)
        let sequence: [(u64, u64, bool, bool, Option<u64>); 11] = [
            (0, 400, true, true, Some(2_000)),
            (2_000, 900, false, false, Some(4_000)),
            (2_100, 1_200, true, false, Some(4_100)),
            (4_100, 1_500, false, false, Some(6_100)),
            (6_100, 1_800, false, false, Some(8_100)),
            (6_900, 2_100, true, false, Some(8_900)),
            (8_900, 2_100, false, false, None),
            (9_500, 2_100, true, false, None),
            (12_000, 2_600, true, true, Some(14_000)),
            (14_000, 2_600, false, false, None),
            (15_000, 2_900, true, true, Some(17_000)),
        ];
        for (at, total, fired, new, quiet) in sequence {
            assert_eq!(edges.observe(ms(at), total, fired), new, "the reading at {at} ms");
            assert_eq!(edges.quiet_at(), quiet.map(ms), "the quiet check after {at} ms");
        }
    }

    /// A firing with no stall time accrued since a standing stall's last
    /// reading, within a window of it, neither ends the stall nor starts one.
    #[test]
    fn a_firing_within_a_window_of_the_last_accrual_keeps_the_stall() {
        let start = Instant::now();
        let mut edges = StallEdges::default();
        assert!(edges.observe(start, 10, true));
        assert!(!edges.observe(start + PSI_WINDOW / 2, 10, true));
        assert_eq!(edges.quiet_at(), Some(start + PSI_WINDOW), "still standing");
        assert!(!edges.observe(start + PSI_WINDOW, 25, true), "the same stall accruing again");
    }

    /// A PSI file's cumulative stall time is its `some` line's `total=`.
    #[test]
    fn the_stall_total_is_the_some_lines() {
        let text = "some avg10=1.20 avg60=0.40 avg300=0.10 total=81234\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=512\n";
        assert_eq!(stall_total(text), Some(81_234));
        assert_eq!(stall_total("full avg10=0.00 avg60=0.00 avg300=0.00 total=512\n"), None);
    }

    /// A wake ends a wait, and wakes before it coalesce into one.
    #[test]
    fn a_wake_ends_the_wait() {
        let Some(events) = PressureEvents::open() else {
            // Only Linux may refuse this process its pressure triggers
            // (`platform::refused`).
            assert!(cfg!(target_os = "linux"), "macOS and Windows always give pressure events");
            return;
        };
        let waker = events.waker();
        waker.wake();
        waker.wake();
        // The OS may report a pressure change beside it.
        assert!(matches!(events.wait(), Wake::Asked | Wake::Pressure));
        let thread = std::thread::spawn(move || waker.wake());
        assert!(matches!(events.wait(), Wake::Asked | Wake::Pressure));
        thread.join().unwrap();
    }
}
