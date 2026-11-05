//! Ending this process.

use std::io::Write;

/// End this process with `code` now: its standard streams flushed, and
/// nothing else run, no exit handler and no destructor, so what it holds is
/// released by the system, as for a kill. For a process whose every durable
/// effect is already made.
pub fn end(code: i32) -> ! {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    // SAFETY: `_exit` takes any status and does not return.
    unsafe {
        libc::_exit(code)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
        // SAFETY: the pseudo-handle of this process; termination does not return.
        unsafe { TerminateProcess(GetCurrentProcess(), code as u32) };
        // It returns only where termination failed: end without it.
        std::process::abort()
    }
}
