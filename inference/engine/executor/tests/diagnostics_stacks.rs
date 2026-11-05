//! The report on demand writes every thread's stack while threads are
//! blocked on a lock and one holds the allocator: a process of its own
//! (this test binary run again as the child), whose global allocator a
//! thread can hold, is sent SIGUSR1, and its report read from its stderr.

#![cfg(unix)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// The system allocator, behind a gate a thread can hold: while it does,
/// every other allocation of the process waits.
struct Gated;

static GATING: AtomicBool = AtomicBool::new(false);
static GATE: Mutex<()> = Mutex::new(());

// SAFETY: forwards to the system allocator, after the gate.
unsafe impl GlobalAlloc for Gated {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if GATING.load(Ordering::Acquire) {
            drop(GATE.lock());
        }
        // SAFETY: the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if GATING.load(Ordering::Acquire) {
            drop(GATE.lock());
        }
        // SAFETY: the caller's.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Gated = Gated;

const CHILD: &str = "MAGNITUDE_STACKS_CHILD";
const BLOCKED: usize = 4;

/// The child: threads blocked on a lock the main thread holds, one thread
/// holding the allocator, and the report installed; it says when it is
/// ready, then waits to be killed, or for its stdin to close as its parent
/// ends by any path, when it exits at once: with the allocator held, no
/// return through the test harness could end the process.
#[test]
fn child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    magnitude_executor::diagnostics::install();
    let held = std::sync::Arc::new(Mutex::new(()));
    let _guard = held.lock().unwrap();
    for index in 0..BLOCKED {
        let held = held.clone();
        std::thread::Builder::new().name(format!("blocked-{index}")).spawn(move || drop(held.lock())).unwrap();
    }
    let (holding, holds) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("holder".into())
        .spawn(move || {
            let _gate = GATE.lock().unwrap();
            holding.send(()).unwrap();
            // Every allocation waits from here, this thread's own too: it
            // allocates nothing more.
            GATING.store(true, Ordering::Release);
            loop {
                std::thread::park();
            }
        })
        .unwrap();
    holds.recv().unwrap();
    while !GATING.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }
    // The allocator is held from here: the readiness is a raw write.
    // SAFETY: writes a static line to stdout.
    unsafe { libc::write(1, b"ready\n".as_ptr().cast(), 6) };
    let mut byte = 0u8;
    // SAFETY: blocks reading stdin, which the parent never writes, until it
    // closes; then ends the process without allocating.
    unsafe {
        libc::read(0, (&mut byte as *mut u8).cast(), 1);
        libc::_exit(0)
    }
}

/// The running child, killed and reaped however the test ends, its failure
/// paths included.
struct Reaped(std::process::Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        // A child that has exited and is not yet reaped is killed as Ok.
        self.0.kill().expect("the child is killed");
        self.0.wait().expect("the child is reaped");
    }
}

#[test]
fn every_thread_is_reported_while_threads_block_and_one_holds_the_allocator() {
    if std::env::var_os(CHILD).is_some() {
        return;
    }
    let mut child = Reaped(Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap());
    let child = &mut child.0;
    let mut ready = String::new();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    while !ready.ends_with("ready\n") {
        ready.clear();
        assert!(stdout.read_line(&mut ready).unwrap() > 0, "the child ended before it was ready");
    }
    let pid = child.id() as libc::pid_t;
    #[cfg(target_os = "linux")]
    let threads = std::fs::read_dir(format!("/proc/{pid}/task")).unwrap().count();
    // SAFETY: a signal to the child.
    unsafe { libc::kill(pid, libc::SIGUSR1) };
    let stderr = child.stderr.take().unwrap();
    let (lines, read) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            let end = line == "magnitude-engine: stacks end";
            if lines.send(line).is_err() || end {
                break;
            }
        }
    });
    let mut report = Vec::new();
    loop {
        match read.recv_timeout(Duration::from_secs(30)) {
            Ok(line) if line == "magnitude-engine: stacks end" => break,
            Ok(line) => report.push(line),
            Err(_) => panic!("the stacks did not end within 30 s:\n{}", report.join("\n")),
        }
    }
    let text = report.join("\n");
    let stacks = report.iter().filter(|line| line.starts_with("  thread ")).collect::<Vec<_>>();
    let header = report.iter().find(|line| line.starts_with("magnitude-engine: the stacks of ")).expect("a header");
    let counted = header["magnitude-engine: the stacks of ".len()..].split(' ').next().unwrap().parse::<usize>().unwrap();
    assert_eq!(stacks.len(), counted, "every thread listed is reported:\n{text}");
    // x86-64 code keeps no frame records: its stacks are unwound through
    // the images' tables.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    assert!(header.contains("through its images' unwind tables"), "{text}");
    #[cfg(target_os = "linux")]
    assert_eq!(counted, threads - 1, "every thread but the reporter is listed:\n{text}");
    for name in (0..BLOCKED).map(|index| format!("blocked-{index}")).chain(["holder".to_owned()]) {
        let stack = stacks.iter().find(|line| line.contains(&format!(" {name}:"))).unwrap_or_else(|| panic!("{name} is not reported:\n{text}"));
        assert!(stack.contains(" 0x"), "{name}'s stack has frames: {stack}");
        assert!(!stack.contains("did not answer") && !stack.contains("could not be read"), "{name} answered: {stack}");
        assert!(stack.matches(" 0x").count() >= 3, "{name}'s stack walks past its wait: {stack}");
    }
    assert!(text.contains("magnitude-engine: the images"), "{text}");
}
