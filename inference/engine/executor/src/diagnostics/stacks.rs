//! Every thread's stack, collected by the report's own thread without
//! allocating, locking or loading anything: a report is wanted when the
//! process is stuck, and a stuck thread may hold the allocator's, the
//! loader's or the unwinder's lock, so a collector that took one would stall
//! with it (in a hung process, a backtrace in each thread's own signal
//! handler stopped after 7 of 97 threads).
//!
//! Each stack is the thread's frame records walked from its registers: the
//! chain every function that calls another keeps on AArch64 (the ABI's on
//! macOS, and Rust's target default on Linux, the standard library's
//! included). Unwinding by tables instead needs the loader's view of the
//! images (`dl_iterate_phdr`, a lock) and allocates. Where a target omits
//! frame records (x86-64 Linux by default) the walk ends at the first
//! function without one: its program counter and its callers up to there.
//! Addresses are printed raw, then the images they fall in, for
//! `addr2line`/`atos`: symbolizing reads the images' tables, which
//! allocates.
//!
//! - **Linux**: the reporter asks each thread in turn (SIGUSR2). Its handler,
//!   which only reads its registers and the stack through the kernel (a
//!   frame record copied through a pipe: an address no longer mapped fails
//!   the copy, never faults) into a preallocated slot, sets an atomic, and
//!   returns. Signal-safe calls only (`gettid`, `write`, `read`). The kernel
//!   stacks (`/proc/self/task/*/stack`) need privileges the containers
//!   refuse, and show no user frame.
//! - **macOS**: the reporter suspends each other thread (`thread_suspend`),
//!   reads its registers (`thread_get_state`) and walks its records through
//!   the kernel (`mach_vm_read_overwrite`), then resumes it: no code runs
//!   on the thread observed.

use std::fmt::Write as _;

/// The most frames of a stack a report prints.
const FRAMES: usize = 64;

/// Bytes of report written at once; the report goes out in pieces of this,
/// from a buffer on the reporter's stack.
const PIECE: usize = 16 << 10;

/// The report's text, written to its descriptor without allocating:
/// formatting into a fixed buffer, flushed whole.
struct Out {
    fd: libc::c_int,
    buffer: [u8; PIECE],
    len: usize,
}

impl Out {
    fn new(fd: libc::c_int) -> Self {
        Self { fd, buffer: [0; PIECE], len: 0 }
    }

    fn flush(&mut self) {
        let mut written = 0;
        while written < self.len {
            // SAFETY: writes initialized bytes of the buffer to the report's descriptor.
            let wrote = unsafe { libc::write(self.fd, self.buffer[written..self.len].as_ptr().cast(), self.len - written) };
            if wrote <= 0 {
                break;
            }
            written += wrote as usize;
        }
        self.len = 0;
    }
}

impl std::fmt::Write for Out {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        for chunk in text.as_bytes().chunks(PIECE) {
            if self.len + chunk.len() > PIECE {
                self.flush();
            }
            self.buffer[self.len..self.len + chunk.len()].copy_from_slice(chunk);
            self.len += chunk.len();
        }
        Ok(())
    }
}

/// A thread's name: bytes up to the first NUL or line end.
fn name(bytes: &[u8]) -> &str {
    let end = bytes.iter().position(|&byte| byte == 0 || byte == b'\n').unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("?")
}

/// A thread's registers a walk begins from: its program counter, its link
/// register (none on x86-64), frame pointer and stack pointer.
#[derive(Clone, Copy)]
pub(super) struct Registers {
    pub(super) pc: u64,
    pub(super) link: Option<u64>,
    pub(super) frame: u64,
    pub(super) sp: u64,
}

/// A fatal signal as its thread took it: the signal, the address it names
/// (the faulting access, for a fault), the thread, and its registers at the
/// faulting instruction.
#[derive(Clone, Copy)]
pub(super) struct Fault {
    pub(super) signal: libc::c_int,
    pub(super) address: u64,
    pub(super) thread: u64,
    pub(super) registers: Registers,
}

/// The fault the reporter writes: stored by the faulting thread's handler
/// before it wakes the reporter, which loads it.
static FAULT: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

impl Fault {
    /// The fault the handler's arguments describe.
    ///
    /// # Safety
    ///
    /// `info` and `context` are a `SA_SIGINFO` handler's.
    pub(super) unsafe fn taken(signal: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) -> Self {
        // SAFETY: the caller's.
        let (address, thread, registers) = unsafe { platform::taken(info, context) };
        Self { signal, address, thread, registers }
    }

    /// Store it for the reporter: signal-safe, atomics only.
    pub(super) fn store(&self) {
        use std::sync::atomic::Ordering;
        let Registers { pc, link, frame, sp } = self.registers;
        // A link register of 0 is no return address: none.
        let words = [self.signal as u64, self.address, self.thread, pc, link.unwrap_or(0), frame, sp];
        for (slot, word) in FAULT.iter().zip(words) {
            slot.store(word, Ordering::Release);
        }
    }

    /// The fault last stored ([`Fault::store`]).
    pub(super) fn load() -> Self {
        use std::sync::atomic::Ordering;
        let [signal, address, thread, pc, link, frame, sp] = std::array::from_fn(|index| FAULT[index].load(Ordering::Acquire));
        Self {
            signal: signal as libc::c_int,
            address,
            thread,
            registers: Registers { pc, link: (link != 0).then_some(link), frame, sp },
        }
    }

    /// The report's line naming it, before the thread's frames.
    fn heading(&self, out: &mut Out, thread: &str) {
        let signal = match self.signal {
            libc::SIGSEGV => "SIGSEGV",
            libc::SIGBUS => "SIGBUS",
            libc::SIGILL => "SIGILL",
            libc::SIGFPE => "SIGFPE",
            libc::SIGABRT => "SIGABRT",
            _ => "a fatal signal",
        };
        let _ = write!(out, "magnitude-engine: thread {} {thread} took {signal} ({}) at address {:#x}:", self.thread, self.signal, self.address);
    }
}

/// What the reporter writes its reports in: the buffers too large for its
/// frames, allocated once, as it starts, so that no report allocates.
pub(super) use platform::Scratch;

/// Write the faulted thread's stack first, when there is one, then every
/// other thread's, then the images their addresses fall in, to `fd`.
/// Allocates nothing, takes no lock a stuck thread can hold.
pub(super) fn write(fd: libc::c_int, fault: Option<&Fault>, scratch: &mut Scratch) {
    let mut out = Out::new(fd);
    platform::write(&mut out, fault, scratch);
    let _ = writeln!(out, "magnitude-engine: stacks end");
    out.flush();
}

/// Ready the stacks' collection, at the report's installation.
pub(super) fn install() {
    platform::install();
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod platform {
    use super::{name, Fault, Out, Registers, FRAMES};
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// The thread asked for its stack now; none (0) between asks.
    static TARGET: AtomicI64 = AtomicI64::new(0);
    /// A handler is walking.
    static BUSY: AtomicBool = AtomicBool::new(false);
    /// The asked thread's walk ended.
    static DONE: AtomicBool = AtomicBool::new(false);
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    static SLOT: [AtomicU64; FRAMES] = [const { AtomicU64::new(0) }; FRAMES];
    /// The asked thread's interrupted program counter, stack pointer and
    /// frame pointer: where the reporter's unwinding through the images'
    /// tables begins (x86-64).
    static REGISTERS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    /// The reporter is done with the asked thread's stack: its handler,
    /// which holds the thread still meanwhile, may return (x86-64).
    static RELEASED: AtomicBool = AtomicBool::new(false);
    /// A pipe a frame record is copied through: the kernel fails the copy
    /// of an unmapped address where a load would fault.
    static PROBE: [AtomicI32; 2] = [AtomicI32::new(-1), AtomicI32::new(-1)];

    /// How long a thread has to answer, and a walk to end: a walk is a few
    /// dozen system calls.
    const ANSWER: Duration = Duration::from_millis(500);

    /// The most threads a report lists.
    const THREADS: usize = 4096;

    /// The process's thread list and, on x86-64, its images and unwinding
    /// context.
    pub(in super::super) struct Scratch {
        tids: Box<[i64]>,
        #[cfg(target_arch = "x86_64")]
        images: super::cfi::Images,
    }

    impl Scratch {
        pub(in super::super) fn new() -> Self {
            Self {
                tids: vec![0; THREADS].into_boxed_slice(),
                #[cfg(target_arch = "x86_64")]
                images: super::cfi::Images::new(),
            }
        }
    }

    pub(super) fn install() {
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: a pipe this process owns from now on.
        if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            panic!("the stack probe pipe opens: {}", std::io::Error::last_os_error());
        }
        PROBE[0].store(fds[0], Ordering::Release);
        PROBE[1].store(fds[1], Ordering::Release);
        // SAFETY: a handler of only signal-safe calls (`handler`).
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler as extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) as libc::sighandler_t;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            // Without the handler SIGUSR2 ends the process, and a report
            // signals every thread with it.
            if libc::sigaction(libc::SIGUSR2, &action, std::ptr::null_mut()) != 0 {
                panic!("the stack handler installs: {}", std::io::Error::last_os_error());
            }
        }
    }

    /// The frame record at `address`: the caller's record and the return
    /// address; none where it is not mapped.
    fn record(address: u64) -> Option<[u64; 2]> {
        let mut bytes = [0u8; 16];
        copy(address, &mut bytes).then(|| [0, 8].map(|at| u64::from_ne_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))))
    }

    /// Copy `into.len()` bytes (at most a page) from `address` into `into`
    /// through the probe pipe: whether all were mapped.
    pub(super) fn copy(address: u64, into: &mut [u8]) -> bool {
        let (read, write) = (PROBE[0].load(Ordering::Acquire), PROBE[1].load(Ordering::Acquire));
        // SAFETY: the kernel copies from `address`, failing (EFAULT) rather
        // than faulting where it is not mapped.
        if unsafe { libc::write(write, address as *const libc::c_void, into.len()) } != into.len() as isize {
            return false;
        }
        // SAFETY: reads the bytes just written into `into`.
        (unsafe { libc::read(read, into.as_mut_ptr().cast(), into.len()) }) == into.len() as isize
    }

    /// An address with its pointer-authentication bits stripped.
    fn stripped(address: u64) -> u64 {
        if cfg!(target_arch = "aarch64") { address & ((1 << 48) - 1) } else { address }
    }

    /// The interrupted thread's registers.
    ///
    /// # Safety
    ///
    /// `context` is the handler's `ucontext_t`.
    unsafe fn registers(context: *mut libc::c_void) -> Registers {
        let context = context as *const libc::ucontext_t;
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: the caller's.
            let machine = unsafe { &(*context).uc_mcontext };
            Registers { pc: machine.pc, link: Some(machine.regs[30]), frame: machine.regs[29], sp: machine.sp }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: the caller's.
            let registers = unsafe { &(*context).uc_mcontext.gregs };
            let register = |index: libc::c_int| registers[index as usize] as u64;
            Registers { pc: register(libc::REG_RIP), link: None, frame: register(libc::REG_RBP), sp: register(libc::REG_RSP) }
        }
    }

    /// The faulting address, the thread and its registers of a fatal
    /// signal's handler ([`super::Fault::taken`]).
    ///
    /// # Safety
    ///
    /// `info` and `context` are a `SA_SIGINFO` handler's.
    pub(super) unsafe fn taken(info: *mut libc::siginfo_t, context: *mut libc::c_void) -> (u64, u64, Registers) {
        // SAFETY: the caller's; `gettid` takes nothing and cannot fail.
        unsafe { ((*info).si_addr() as u64, libc::syscall(libc::SYS_gettid) as u64, registers(context)) }
    }

    /// The stack from `registers` by its frame records: the program
    /// counter, the link register (a leaf interrupted before its record is
    /// made has its caller there, and in no record), then each record's
    /// return address, into `frames`; how many. Signal-safe.
    fn walk(registers: Registers, frames: &mut [u64; FRAMES]) -> usize {
        let mut count = 0;
        // Whether `frames` has room for more.
        let mut push = |address: u64| {
            if count < FRAMES && address != 0 {
                frames[count] = stripped(address);
                count += 1;
            }
            count < FRAMES
        };
        let mut room = push(registers.pc) && registers.link.is_none_or(&mut push);
        let mut frame = registers.frame;
        while room && frame != 0 && frame % 8 == 0 {
            let Some([next, returned]) = record(frame) else { break };
            room = push(returned);
            if next <= frame {
                break;
            }
            frame = next;
        }
        count
    }

    /// SIGUSR2: the asked thread walks its own frame records into the slot.
    /// On x86-64, whose code keeps no frame records by default, it also
    /// leaves its registers and holds still until the reporter has unwound
    /// its stack through the images' tables (`cfi`), or for at most
    /// [`ANSWER`].
    extern "C" fn handler(_: libc::c_int, _: *mut libc::siginfo_t, context: *mut libc::c_void) {
        // SAFETY: `gettid` takes nothing and cannot fail.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) };
        if TARGET.load(Ordering::Acquire) != tid || BUSY.swap(true, Ordering::AcqRel) {
            return;
        }
        // SAFETY: the kernel's context for this handler.
        let registers = unsafe { registers(context) };
        for (slot, value) in REGISTERS.iter().zip([registers.pc, registers.sp, registers.frame]) {
            slot.store(value, Ordering::Relaxed);
        }
        let mut frames = [0u64; FRAMES];
        let count = walk(registers, &mut frames);
        for (slot, frame) in SLOT.iter().zip(&frames[..count]) {
            slot.store(*frame, Ordering::Relaxed);
        }
        COUNT.store(count, Ordering::Release);
        DONE.store(true, Ordering::Release);
        if cfg!(target_arch = "x86_64") {
            hold();
        }
        BUSY.store(false, Ordering::Release);
    }

    /// Hold the handling thread still until the reporter releases it, or
    /// for at most [`ANSWER`]: with signal-safe calls only.
    fn hold() {
        let now = || {
            let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
            // SAFETY: reads the monotonic clock into a local.
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
            Duration::new(time.tv_sec as u64, time.tv_nsec as u32)
        };
        let began = now();
        while !RELEASED.load(Ordering::Acquire) && now() - began < ANSWER {
            // SAFETY: yields the processor; takes nothing.
            unsafe { libc::sched_yield() };
        }
    }

    /// Every thread of the process, by its directory under /proc/self/task,
    /// read with system calls into `tids`; how many.
    fn threads(tids: &mut [i64]) -> usize {
        // SAFETY: opens a directory by a NUL-terminated path.
        let directory = unsafe { libc::open(c"/proc/self/task".as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC) };
        if directory < 0 {
            return 0;
        }
        let mut entries = [0u8; 8192];
        let mut found = 0;
        loop {
            // SAFETY: the kernel writes directory entries into the buffer.
            let read = unsafe { libc::syscall(libc::SYS_getdents64, directory, entries.as_mut_ptr(), entries.len()) };
            if read <= 0 {
                break;
            }
            let mut at = 0;
            while at < read as usize {
                // `linux_dirent64`: inode (8), offset (8), length (2), type
                // (1), then the NUL-terminated name.
                let length = u16::from_ne_bytes([entries[at + 16], entries[at + 17]]) as usize;
                let tid = name(&entries[at + 19..at + length]).parse::<i64>();
                if let (Ok(tid), true) = (tid, found < tids.len()) {
                    tids[found] = tid;
                    found += 1;
                }
                at += length;
            }
        }
        // SAFETY: closes the descriptor opened above.
        unsafe { libc::close(directory) };
        found
    }

    /// Copy what `path` (NUL-terminated) holds into `into`: how many bytes.
    fn file(path: &[u8], into: &mut [u8]) -> usize {
        // SAFETY: opens a NUL-terminated path read only.
        let file = unsafe { libc::open(path.as_ptr().cast(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if file < 0 {
            return 0;
        }
        // SAFETY: reads into the buffer, at most its length.
        let read = unsafe { libc::read(file, into.as_mut_ptr().cast(), into.len()) };
        // SAFETY: closes the descriptor opened above.
        unsafe { libc::close(file) };
        read.max(0) as usize
    }

    /// Thread `tid`'s name, read into `comm`.
    fn thread_name(tid: i64, comm: &mut [u8; 32]) -> &str {
        let mut path = [0u8; 64];
        let mut cursor = std::io::Cursor::new(&mut path[..]);
        let _ = std::io::Write::write_fmt(&mut cursor, format_args!("/proc/self/task/{tid}/comm\0"));
        let length = file(&path, comm);
        name(&comm[..length])
    }

    pub(super) fn write(out: &mut Out, fault: Option<&Fault>, scratch: &mut Scratch) {
        // SAFETY: `getpid` and `gettid` take nothing and cannot fail.
        let (pid, me) = unsafe { (libc::getpid(), libc::syscall(libc::SYS_gettid)) };
        let faulted = fault.map(|fault| fault.thread as i64);
        let tids = &mut scratch.tids[..];
        let found = threads(tids);
        let listed = |tid: &&i64| **tid != me && Some(**tid) != faulted;
        let others = tids[..found].iter().filter(listed).count();
        #[cfg(target_arch = "x86_64")]
        let images = &mut scratch.images;
        #[cfg(target_arch = "x86_64")]
        let read = images.read();
        // The faulted thread first, from its registers at the fault: it is
        // held in its handler, which would only add its own frames.
        if let Some(fault) = fault {
            fault.heading(out, thread_name(fault.thread as i64, &mut [0; 32]));
            let mut frames = [0u64; FRAMES];
            #[allow(unused_mut)]
            let mut count = 0;
            #[cfg(target_arch = "x86_64")]
            if read.is_ok() {
                let Registers { pc, sp, frame, .. } = fault.registers;
                count = images.unwind(pc, sp, frame, &mut frames);
            }
            // Frame records where no table unwinds it.
            if count <= 1 {
                count = walk(fault.registers, &mut frames);
            }
            for frame in &frames[..count] {
                let _ = write!(out, " {frame:#x}");
            }
            let _ = writeln!(out);
        }
        #[cfg(target_arch = "x86_64")]
        let _ = match read {
            Ok(()) => writeln!(out, "magnitude-engine: the stacks of {others} threads, each unwound from its registers through its images' unwind tables (addresses; the images below map them):"),
            Err(why) => writeln!(out, "magnitude-engine: the stacks of {others} threads, each its frame records from its registers, as far as they are kept ({why}; addresses, the images below map them):"),
        };
        #[cfg(not(target_arch = "x86_64"))]
        let _ = writeln!(out, "magnitude-engine: the stacks of {others} threads, each its frame records from its registers (addresses; the images below map them):");
        for &tid in tids[..found].iter().filter(listed) {
            let _ = write!(out, "  thread {tid} {}:", thread_name(tid, &mut [0; 32]));
            DONE.store(false, Ordering::Release);
            RELEASED.store(false, Ordering::Release);
            TARGET.store(tid, Ordering::Release);
            // SAFETY: a signal to a thread of this process, whose handler is
            // `handler`.
            unsafe { libc::syscall(libc::SYS_tgkill, pid as libc::c_long, tid, libc::SIGUSR2) };
            let asked = Instant::now();
            while !DONE.load(Ordering::Acquire) && asked.elapsed() < ANSWER {
                std::thread::sleep(Duration::from_micros(100));
            }
            TARGET.store(0, Ordering::Release);
            if DONE.load(Ordering::Acquire) {
                let mut frames = [0u64; FRAMES];
                #[allow(unused_mut)]
                let mut count = 0;
                #[cfg(target_arch = "x86_64")]
                if read.is_ok() {
                    let [pc, sp, fp] = [0, 1, 2].map(|register| REGISTERS[register].load(Ordering::Relaxed));
                    count = images.unwind(pc, sp, fp, &mut frames);
                }
                // Frame records where no table unwinds it.
                if count <= 1 {
                    count = COUNT.load(Ordering::Acquire);
                    for (frame, slot) in frames.iter_mut().zip(&SLOT[..count]) {
                        *frame = slot.load(Ordering::Relaxed);
                    }
                }
                RELEASED.store(true, Ordering::Release);
                for frame in &frames[..count] {
                    let _ = write!(out, " {frame:#x}");
                }
                let _ = writeln!(out);
                // The released handler returns before the next is asked.
                while BUSY.load(Ordering::Acquire) && asked.elapsed() < ANSWER * 2 {
                    std::thread::sleep(Duration::from_micros(100));
                }
                continue;
            }
            let _ = writeln!(out, " did not answer within {ANSWER:?} (ended, or its signals blocked)");
            // A walk begun past its time ends before the next is asked.
            while BUSY.load(Ordering::Acquire) && asked.elapsed() < ANSWER * 2 {
                std::thread::sleep(Duration::from_micros(100));
            }
            if BUSY.load(Ordering::Acquire) {
                let _ = writeln!(out, "magnitude-engine: a walk did not end; the threads after it are not walked");
                break;
            }
        }
        let _ = writeln!(out, "magnitude-engine: the images (executable mappings of /proc/self/maps):");
        // SAFETY: opens a NUL-terminated path read only.
        let maps = unsafe { libc::open(c"/proc/self/maps".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if maps >= 0 {
            let mut chunk = [0u8; 4096];
            let mut line = [0u8; 512];
            let mut filled = 0;
            loop {
                // SAFETY: reads into the buffer, at most its length.
                let read = unsafe { libc::read(maps, chunk.as_mut_ptr().cast(), chunk.len()) };
                if read <= 0 {
                    break;
                }
                for &byte in &chunk[..read as usize] {
                    if byte != b'\n' {
                        if filled < line.len() {
                            line[filled] = byte;
                            filled += 1;
                        }
                        continue;
                    }
                    let text = std::str::from_utf8(&line[..filled]).unwrap_or("");
                    if text.split_whitespace().nth(1).is_some_and(|permissions| permissions.contains('x')) {
                        let _ = writeln!(out, "  {text}");
                    }
                    filled = 0;
                }
            }
            // SAFETY: closes the descriptor opened above.
            unsafe { libc::close(maps) };
        }
    }

}

/// Unwinding through the images' tables, where code keeps no frame
/// records (x86-64 Linux).
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod cfi;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod platform {
    use super::{name, Fault, Out, Registers, FRAMES};
    use std::fmt::Write as _;

    type Port = u32;

    /// `ARM_THREAD_STATE64`: x0-x28, fp, lr, sp, pc, cpsr and padding.
    #[repr(C)]
    #[derive(Default)]
    struct ThreadState {
        x: [u64; 29],
        fp: u64,
        lr: u64,
        sp: u64,
        pc: u64,
        cpsr: u32,
        pad: u32,
    }

    /// `thread_identifier_info`.
    #[repr(C)]
    #[derive(Default)]
    struct Identifier {
        thread_id: u64,
        thread_handle: u64,
        dispatch_qaddr: u64,
    }

    /// `thread_extended_info`.
    #[repr(C)]
    struct Extended {
        user_time: u64,
        system_time: u64,
        cpu_usage: i32,
        policy: i32,
        run_state: i32,
        flags: i32,
        sleep_time: i32,
        current_priority: i32,
        priority: i32,
        max_priority: i32,
        name: [u8; 64],
    }

    /// `struct __darwin_ucontext`, as a signal handler receives it.
    #[repr(C)]
    struct Context {
        onstack: i32,
        mask: u32,
        stack: libc::stack_t,
        link: *mut libc::c_void,
        machine_bytes: usize,
        machine: *const Machine,
    }

    /// `struct __darwin_mcontext64`: the exception state, then the thread
    /// state (the floating-point state follows, unread).
    #[repr(C)]
    struct Machine {
        fault_address: u64,
        syndrome: u32,
        exception: u32,
        state: ThreadState,
    }

    const ARM_THREAD_STATE64: i32 = 6;
    const THREAD_IDENTIFIER_INFO: i32 = 4;
    const THREAD_EXTENDED_INFO: i32 = 5;

    unsafe extern "C" {
        static mach_task_self_: Port;
        fn task_threads(task: Port, threads: *mut *mut Port, count: *mut u32) -> i32;
        fn mach_thread_self() -> Port;
        fn thread_suspend(thread: Port) -> i32;
        fn thread_resume(thread: Port) -> i32;
        fn thread_get_state(thread: Port, flavor: i32, state: *mut u32, count: *mut u32) -> i32;
        fn thread_info(thread: Port, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
        fn mach_vm_read_overwrite(task: Port, address: u64, size: u64, data: u64, read: *mut u64) -> i32;
        fn mach_port_deallocate(task: Port, name: Port) -> i32;
        fn vm_deallocate(task: Port, address: usize, size: usize) -> i32;
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_header(image: u32) -> *const libc::c_void;
        fn _dyld_get_image_name(image: u32) -> *const libc::c_char;
    }

    pub(super) fn install() {}

    /// An address with its pointer-authentication bits stripped.
    fn stripped(address: u64) -> u64 {
        address & ((1 << 47) - 1)
    }

    /// `thread`'s registers, read while it is suspended.
    fn state(thread: Port) -> Option<ThreadState> {
        let mut state = ThreadState::default();
        let mut count = (size_of::<ThreadState>() / 4) as u32;
        // SAFETY: the kernel writes at most `count` words of the state.
        (unsafe { thread_get_state(thread, ARM_THREAD_STATE64, (&raw mut state).cast(), &mut count) } == 0).then_some(state)
    }

    /// The frame record at `address`: the caller's record and the return
    /// address; none where it is not mapped.
    fn record(address: u64) -> Option<[u64; 2]> {
        let mut words = [0u64; 2];
        let mut read = 0;
        // SAFETY: the kernel copies 16 bytes of this task's memory into the
        // local, failing where `address` is not mapped.
        let status = unsafe { mach_vm_read_overwrite(mach_task_self_, address, 16, words.as_mut_ptr() as u64, &mut read) };
        (status == 0 && read == 16).then_some(words)
    }

    /// A suspended thread's registers ([`state`]).
    fn registers(state: &ThreadState) -> Registers {
        Registers { pc: state.pc, link: Some(state.lr), frame: state.fp, sp: state.sp }
    }

    /// The faulting address, the thread and its registers of a fatal
    /// signal's handler ([`super::Fault::taken`]).
    ///
    /// # Safety
    ///
    /// `info` and `context` are a `SA_SIGINFO` handler's.
    pub(super) unsafe fn taken(info: *mut libc::siginfo_t, context: *mut libc::c_void) -> (u64, u64, Registers) {
        let mut thread = 0u64;
        // SAFETY: the caller's: the kernel's machine context of the
        // interrupted thread; the calling thread's id into a local.
        unsafe {
            libc::pthread_threadid_np(0, &mut thread);
            let state = &(*(*(context as *const Context)).machine).state;
            ((*info).si_addr() as u64, thread, registers(state))
        }
    }

    /// The stack from `registers`: the program counter, the link register,
    /// then each frame record's return address, into `frames`; how many.
    fn walk(registers: Registers, frames: &mut [u64; FRAMES]) -> usize {
        let mut count = 0;
        // Whether `frames` has room for more.
        let mut push = |address: u64| {
            if count < FRAMES && address != 0 {
                frames[count] = stripped(address);
                count += 1;
            }
            count < FRAMES
        };
        let mut room = push(registers.pc) && registers.link.is_none_or(&mut push);
        let mut frame = stripped(registers.frame);
        while room && frame != 0 && frame % 8 == 0 {
            let Some([next, returned]) = record(frame) else { break };
            room = push(returned);
            let next = stripped(next);
            if next <= frame {
                break;
            }
            frame = next;
        }
        count
    }

    /// `thread`'s id and name.
    fn identity(thread: Port) -> (u64, Extended) {
        let mut identifier = Identifier::default();
        let mut identified = (size_of::<Identifier>() / 4) as u32;
        // SAFETY: the kernel writes at most `identified` words.
        unsafe { thread_info(thread, THREAD_IDENTIFIER_INFO, (&raw mut identifier).cast(), &mut identified) };
        // SAFETY: zeroed is a valid `Extended`; the kernel writes at most
        // its words.
        let mut extended: Extended = unsafe { std::mem::zeroed() };
        let mut words = (size_of::<Extended>() / 4) as u32;
        // SAFETY: as above.
        unsafe { thread_info(thread, THREAD_EXTENDED_INFO, (&raw mut extended).cast(), &mut words) };
        (identifier.thread_id, extended)
    }

    /// Nothing: a report here fits the reporter's frames.
    pub(in super::super) struct Scratch;

    impl Scratch {
        pub(in super::super) fn new() -> Self {
            Self
        }
    }

    pub(super) fn write(out: &mut Out, fault: Option<&Fault>, _: &mut Scratch) {
        // SAFETY: a send right to the calling thread, released below.
        let me = unsafe { mach_thread_self() };
        let mut threads: *mut Port = std::ptr::null_mut();
        let mut count = 0u32;
        // SAFETY: the kernel allocates the list (with `vm_allocate`, not the
        // process's allocator), deallocated below with its rights.
        if unsafe { task_threads(mach_task_self_, &mut threads, &mut count) } != 0 {
            let _ = writeln!(out, "magnitude-engine: the threads could not be listed");
            return;
        }
        // SAFETY: `count` ports the kernel wrote.
        let threads = unsafe { std::slice::from_raw_parts(threads, count as usize) };
        // The faulted thread first, from its registers at the fault: it is
        // held in its handler, which would only add its own frames.
        if let Some(fault) = fault {
            let named = threads.iter().map(|&thread| identity(thread)).find(|(id, _)| *id == fault.thread);
            fault.heading(out, named.as_ref().map_or("?", |(_, extended)| name(&extended.name)));
            let mut frames = [0u64; FRAMES];
            let count = walk(fault.registers, &mut frames);
            for frame in &frames[..count] {
                let _ = write!(out, " {frame:#x}");
            }
            let _ = writeln!(out);
        }
        let listed = |thread: &&Port| **thread != me && fault.is_none_or(|fault| identity(**thread).0 != fault.thread);
        let others = threads.iter().filter(listed).count();
        let _ = writeln!(out, "magnitude-engine: the stacks of {others} threads, each its frame records from its registers (addresses; the images below map them):");
        for &thread in threads.iter().filter(listed) {
            let (id, extended) = identity(thread);
            let _ = write!(out, "  thread {id} {}:", name(&extended.name));
            let mut frames = [0u64; FRAMES];
            // SAFETY: suspends another thread of this task, resumed below;
            // nothing this thread does until then needs a lock it may hold.
            let walked = match unsafe { thread_suspend(thread) } {
                0 => {
                    let walked = state(thread).map(|state| walk(registers(&state), &mut frames));
                    // SAFETY: resumes the thread suspended above.
                    unsafe { thread_resume(thread) };
                    walked
                }
                _ => None,
            };
            match walked {
                Some(walked) => {
                    for frame in &frames[..walked] {
                        let _ = write!(out, " {frame:#x}");
                    }
                    let _ = writeln!(out);
                }
                None => {
                    let _ = writeln!(out, " could not be read (it ended)");
                }
            }
        }
        for &thread in threads {
            // SAFETY: releases the send right `task_threads` gave.
            unsafe { mach_port_deallocate(mach_task_self_, thread) };
        }
        // SAFETY: releases the list `task_threads` allocated, and the right
        // to this thread.
        unsafe {
            vm_deallocate(mach_task_self_, threads.as_ptr() as usize, threads.len() * size_of::<Port>());
            mach_port_deallocate(mach_task_self_, me);
        }
        // dyld's image list is read last: a thread stuck in the loader
        // holds it, and every stack is out by then.
        let _ = writeln!(out, "magnitude-engine: the images (load address, path):");
        out.flush();
        // SAFETY: dyld's count and its images' headers and names, live for
        // the process.
        for image in 0..unsafe { _dyld_image_count() } {
            let (header, path) = unsafe { (_dyld_get_image_header(image), _dyld_get_image_name(image)) };
            if path.is_null() {
                continue;
            }
            // SAFETY: a NUL-terminated path dyld holds.
            let path = unsafe { std::ffi::CStr::from_ptr(path) };
            let _ = writeln!(out, "  {header:p} {}", path.to_str().unwrap_or("?"));
        }
    }
}

/// Where no stack can be collected without allocating.
#[cfg(not(any(
    all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")),
    all(target_os = "macos", target_arch = "aarch64")
)))]
mod platform {
    use super::{Fault, Out, Registers};
    use std::fmt::Write as _;

    pub(super) fn install() {}

    /// No registers are read here: a fault reports its signal and address.
    pub(super) unsafe fn taken(info: *mut libc::siginfo_t, _: *mut libc::c_void) -> (u64, u64, Registers) {
        // SAFETY: the caller's.
        (unsafe { (*info).si_addr() } as u64, 0, Registers { pc: 0, link: None, frame: 0, sp: 0 })
    }

    /// Nothing: no stack is collected here.
    pub(in super::super) struct Scratch;

    impl Scratch {
        pub(in super::super) fn new() -> Self {
            Self
        }
    }

    pub(super) fn write(out: &mut Out, fault: Option<&Fault>, _: &mut Scratch) {
        if let Some(fault) = fault {
            fault.heading(out, "?");
            let _ = writeln!(out);
        }
        let _ = writeln!(out, "magnitude-engine: no thread's stack is collected on this platform");
    }
}
