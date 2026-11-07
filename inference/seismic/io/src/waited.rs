//! The threads a waiting device's critical path runs on: the read engine's
//! workers, the copy link's issuer and watcher, and the executor's request
//! service, parks' keeper and landing workers. A parked step waits on each
//! of them in turn, so each must wake when its wait ends and run at once:
//!
//! - Linux: a timed wait ends late by the thread's timer slack, 50 µs by
//!   default, longer than a request service's whole rest; the slack is set
//!   to its least, 1 ns.
//! - macOS: a thread of default quality of service may run on an
//!   efficiency core and its timers carry leeway; the thread takes the
//!   user-interactive class, which runs on performance cores with the
//!   least leeway (the CPU backend's pool workers take it for the same
//!   reason).
//!
//! Neither costs the thread any time it does not run.

/// Make the calling thread one a waiting device's critical path runs on
/// (module documentation). Neither call fails for a thread of the process's
/// own (the slack is a valid value; the class is a valid class and the
/// thread no dispatch queue's), so a failure is an invariant violation.
pub fn waited_on() {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: PR_SET_TIMERSLACK takes the slack in nanoseconds and
        // changes only the calling thread's.
        let status = unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, 1 as libc::c_ulong, 0 as libc::c_ulong, 0 as libc::c_ulong, 0 as libc::c_ulong) };
        assert_eq!(status, 0, "a thread's timer slack: {}", std::io::Error::last_os_error());
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: sets the calling thread's own class; no pointers.
        let status = unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
        assert_eq!(status, 0, "a thread's quality of service: {}", std::io::Error::from_raw_os_error(status));
    }
}
