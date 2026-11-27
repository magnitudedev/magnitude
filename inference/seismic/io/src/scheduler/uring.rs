//! Linux `io_uring` batched `O_DIRECT` read engine.
//!
//! Replaces up to 32 blocking `pread` worker threads with a single dedicated
//! ring thread (`seismic-uring`) that:
//! - Batches multiple ready pieces from [`crate::scheduler::Queue`] (preserving
//!   exact priority ordering: demand `Parked` before `Streamed` before
//!   `Expected` before `None`) into a single `io_uring_enter` syscall.
//! - Uses `IORING_OP_READ` for single-part direct or bounced reads and
//!   `IORING_OP_READV` for coalesced multi-part zero-bounce scatter-gather reads.
//! - Issues `IORING_OP_ASYNC_CANCEL` for in-flight speculative reads whose
//!   [`crate::Handle`] was dropped, while keeping destinations and bounce
//!   buffers pinned until the read CQE arrives.
//! - Wakes with zero polling via an `eventfd` armed with `IORING_OP_POLL_ADD`.
//! - Falls back transparently to the `pread` worker pool if `io_uring_setup` is
//!   unavailable (`ENOSYS` / `EPERM` / `EACCES`) or disabled via
//!   `SEISMIC_IO_DISABLE_URING=1`.

use super::engine::{Engine, Flight};
use super::queue::Taken;
use crate::disk::{build_iovecs, refuses_direct, refuses_direct_sink, truncated};
use crate::{Buffer, Error, ALIGNMENT};
use std::collections::{BTreeSet, HashSet};
use std::io;
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Raw Linux `io_uring` UAPI definitions (zero external crate dependencies)
// ---------------------------------------------------------------------------

const IORING_OFF_SQ_RING: i64 = 0;
const IORING_OFF_CQ_RING: i64 = 0x8000_0000;
const IORING_OFF_SQES: i64 = 0x1000_0000;

const IORING_FEAT_SINGLE_MMAP: u32 = 1 << 0;
const IORING_ENTER_GETEVENTS: u32 = 1 << 0;
const IORING_REGISTER_PROBE: libc::c_uint = 8;
const IO_URING_OP_SUPPORTED: u16 = 1 << 0;

const IORING_OP_READV: u8 = 1;
const IORING_OP_POLL_ADD: u8 = 6;
const IORING_OP_ASYNC_CANCEL: u8 = 14;
const IORING_OP_READ: u8 = 22;

/// Ring capacity (SQ entries; kernel allocates `2 * RING_ENTRIES` CQ entries).
const RING_ENTRIES: u32 = 256;

/// Maximum concurrent in-flight read operations on the ring.
/// Sustain QD = 16–32 (up to 64) with ample SQ headroom for `POLL_ADD` and
/// `ASYNC_CANCEL` SQEs.
const MAX_INFLIGHT: usize = 64;

/// Maximum bounce buffers retained locally by the ring thread between ops.
const MAX_LOCAL_SCRATCH: usize = 8;

/// Reserved `user_data` sentinels (in-flight read ops keep generation < 2^55,
/// so `user_data` never collides with these).
const WAKE_USER_DATA: u64 = u64::MAX;
const CANCEL_USER_DATA: u64 = u64::MAX - 1;

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct IoSqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct IoCqringOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct IoUringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: IoSqringOffsets,
    cq_off: IoCqringOffsets,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct IoUringSqe {
    opcode: u8,
    flags: u8,
    ioprio: u16,
    fd: i32,
    off: u64,
    addr: u64,
    len: u32,
    op_flags: u32,
    user_data: u64,
    buf_index: u16,
    personality: u16,
    splice_fd_in: i32,
    addr3: u64,
    resv: u64,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct IoUringCqe {
    user_data: u64,
    res: i32,
    flags: u32,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct IoUringProbeOp {
    op: u8,
    resv: u8,
    flags: u16,
    resv2: u32,
}

const PROBE_OPS_LEN: usize = 32;

#[repr(C)]
struct IoUringProbe {
    last_op: u8,
    ops_len: u8,
    resv: u16,
    resv2: [u32; 3],
    ops: [IoUringProbeOp; PROBE_OPS_LEN],
}

const _: () = assert!(size_of::<IoSqringOffsets>() == 40);
const _: () = assert!(size_of::<IoCqringOffsets>() == 40);
const _: () = assert!(size_of::<IoUringParams>() == 120);
const _: () = assert!(size_of::<IoUringSqe>() == 64);
const _: () = assert!(size_of::<IoUringCqe>() == 16);

fn sys_io_uring_setup(entries: u32, params: &mut IoUringParams) -> io::Result<OwnedFd> {
    // SAFETY: `params` is a valid mutable `IoUringParams` struct for the syscall.
    let fd = unsafe { libc::syscall(libc::SYS_io_uring_setup, entries, params as *mut IoUringParams) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `io_uring_setup` returned a fresh file descriptor owned by us.
    Ok(unsafe { OwnedFd::from_raw_fd(fd as RawFd) })
}

fn sys_io_uring_register_probe(ring_fd: RawFd) -> io::Result<IoUringProbe> {
    let mut probe = IoUringProbe {
        last_op: 0,
        ops_len: 0,
        resv: 0,
        resv2: [0; 3],
        ops: [IoUringProbeOp::default(); PROBE_OPS_LEN],
    };
    // SAFETY: `probe` holds `PROBE_OPS_LEN` trailing `IoUringProbeOp` entries.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_io_uring_register,
            ring_fd,
            IORING_REGISTER_PROBE,
            (&raw mut probe).cast::<libc::c_void>(),
            PROBE_OPS_LEN as libc::c_uint,
        )
    };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(probe)
}

fn probe_supports_required_ops(ring_fd: RawFd) -> io::Result<()> {
    let probe = sys_io_uring_register_probe(ring_fd)?;
    let supported = |op: u8| -> bool {
        let idx = op as usize;
        idx < usize::from(probe.ops_len)
            && idx < PROBE_OPS_LEN
            && (probe.ops[idx].flags & IO_URING_OP_SUPPORTED) != 0
    };
    if supported(IORING_OP_READ)
        && supported(IORING_OP_READV)
        && supported(IORING_OP_POLL_ADD)
        && supported(IORING_OP_ASYNC_CANCEL)
    {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(libc::ENOSYS))
    }
}

/// Owned, single-thread Linux `io_uring` instance with mmapped SQ/CQ rings.
pub(crate) struct IoUring {
    fd: OwnedFd,
    sq_ptr: *mut u8,
    sq_size: usize,
    cq_ptr: *mut u8,
    cq_size: usize,
    single_mmap: bool,
    sqes_ptr: *mut IoUringSqe,
    sqes_size: usize,

    sq_head: *const AtomicU32,
    sq_tail_kernel: *const AtomicU32,
    sq_mask: u32,
    sq_entries: u32,
    sq_array: *mut u32,
    sq_tail: u32,
    pending_sqes: u32,

    cq_head_kernel: *const AtomicU32,
    cq_tail: *const AtomicU32,
    cq_mask: u32,
    cqes: *const IoUringCqe,
    cq_head: u32,
}

// SAFETY: `IoUring` is constructed on the spawning thread and moved exclusively
// to the single `seismic-uring` thread; it is never accessed concurrently.
unsafe impl Send for IoUring {}

impl IoUring {
    /// Probe whether the kernel supports `io_uring` at our working ring size
    /// with the ops we require (verifying both kernel support and `RLIMIT_MEMLOCK`).
    pub(crate) fn probe() -> io::Result<()> {
        let mut params = IoUringParams::default();
        let fd = sys_io_uring_setup(RING_ENTRIES, &mut params)?;
        probe_supports_required_ops(fd.as_raw_fd())
    }

    /// Create and mmap a new `io_uring` ring with `entries` SQ slots.
    pub(crate) fn new(entries: u32) -> io::Result<Self> {
        let mut params = IoUringParams::default();
        let fd = sys_io_uring_setup(entries, &mut params)?;
        probe_supports_required_ops(fd.as_raw_fd())?;

        let sq_size = params.sq_off.array as usize + params.sq_entries as usize * size_of::<u32>();
        let cq_size = params.cq_off.cqes as usize + params.cq_entries as usize * size_of::<IoUringCqe>();
        let single_mmap = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;
        let sq_map_size = if single_mmap { sq_size.max(cq_size) } else { sq_size };

        let mmap_ring = |size: usize, offset: i64| -> io::Result<*mut u8> {
            // SAFETY: mapping kernel-provided io_uring ring region on our owned `fd`.
            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_POPULATE,
                    fd.as_raw_fd(),
                    offset,
                )
            };
            if ptr == libc::MAP_FAILED {
                Err(io::Error::last_os_error())
            } else {
                Ok(ptr.cast::<u8>())
            }
        };

        let sq_ptr = mmap_ring(sq_map_size, IORING_OFF_SQ_RING)?;
        let cq_ptr = if single_mmap {
            sq_ptr
        } else {
            match mmap_ring(cq_size, IORING_OFF_CQ_RING) {
                Ok(ptr) => ptr,
                Err(err) => {
                    // SAFETY: `sq_ptr` was just mapped with `sq_map_size`.
                    unsafe { libc::munmap(sq_ptr.cast(), sq_map_size) };
                    return Err(err);
                }
            }
        };

        let sqes_size = params.sq_entries as usize * size_of::<IoUringSqe>();
        let sqes_ptr = match mmap_ring(sqes_size, IORING_OFF_SQES) {
            Ok(ptr) => ptr.cast::<IoUringSqe>(),
            Err(err) => {
                if !single_mmap {
                    // SAFETY: `cq_ptr` was mapped with `cq_size`.
                    unsafe { libc::munmap(cq_ptr.cast(), cq_size) };
                }
                // SAFETY: `sq_ptr` was mapped with `sq_map_size`.
                unsafe { libc::munmap(sq_ptr.cast(), sq_map_size) };
                return Err(err);
            }
        };

        // SAFETY: all offsets are kernel-validated byte offsets inside the mapped ring regions.
        unsafe {
            let sq_head = sq_ptr.add(params.sq_off.head as usize).cast::<AtomicU32>();
            let sq_tail_kernel = sq_ptr.add(params.sq_off.tail as usize).cast::<AtomicU32>();
            let sq_mask = *sq_ptr.add(params.sq_off.ring_mask as usize).cast::<u32>();
            let sq_entries = *sq_ptr.add(params.sq_off.ring_entries as usize).cast::<u32>();
            let sq_array = sq_ptr.add(params.sq_off.array as usize).cast::<u32>();
            let sq_tail = (*sq_tail_kernel).load(Ordering::Acquire);

            let cq_head_kernel = cq_ptr.add(params.cq_off.head as usize).cast::<AtomicU32>();
            let cq_tail = cq_ptr.add(params.cq_off.tail as usize).cast::<AtomicU32>();
            let cq_mask = *cq_ptr.add(params.cq_off.ring_mask as usize).cast::<u32>();
            let cqes = cq_ptr.add(params.cq_off.cqes as usize).cast::<IoUringCqe>();
            let cq_head = (*cq_head_kernel).load(Ordering::Acquire);

            Ok(Self {
                fd,
                sq_ptr,
                sq_size: sq_map_size,
                cq_ptr,
                cq_size,
                single_mmap,
                sqes_ptr,
                sqes_size,
                sq_head,
                sq_tail_kernel,
                sq_mask,
                sq_entries,
                sq_array,
                sq_tail,
                pending_sqes: 0,
                cq_head_kernel,
                cq_tail,
                cq_mask,
                cqes,
                cq_head,
            })
        }
    }

    fn sq_space(&self) -> usize {
        // SAFETY: `sq_head` points into the live SQ ring mapping.
        let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
        self.sq_entries.saturating_sub(self.sq_tail.wrapping_sub(head)) as usize
    }

    fn has_pending_sqes(&self) -> bool {
        self.pending_sqes > 0
    }

    fn push_sqe(&mut self, sqe: IoUringSqe) -> bool {
        if self.sq_space() == 0 {
            return false;
        }
        let idx = (self.sq_tail & self.sq_mask) as usize;
        // SAFETY: `idx < sq_entries`; we are the sole producer of SQEs.
        unsafe {
            ptr::write(self.sqes_ptr.add(idx), sqe);
            ptr::write(self.sq_array.add(idx), idx as u32);
        }
        self.sq_tail = self.sq_tail.wrapping_add(1);
        self.pending_sqes += 1;
        true
    }

    fn cq_ready(&self) -> u32 {
        // SAFETY: `cq_tail` points into the live CQ ring mapping.
        let tail = unsafe { (*self.cq_tail).load(Ordering::Acquire) };
        tail.wrapping_sub(self.cq_head)
    }

    fn drain_cqes(&mut self, out: &mut Vec<IoUringCqe>) {
        out.clear();
        // SAFETY: `cq_tail` and `cqes` point into the live CQ ring mapping.
        let tail = unsafe { (*self.cq_tail).load(Ordering::Acquire) };
        let mut head = self.cq_head;
        while head != tail {
            let idx = (head & self.cq_mask) as usize;
            // SAFETY: `idx <= cq_mask < cq_entries`.
            let cqe = unsafe { ptr::read(self.cqes.add(idx)) };
            out.push(cqe);
            head = head.wrapping_add(1);
        }
        if head != self.cq_head {
            self.cq_head = head;
            // SAFETY: `cq_head_kernel` points into the live CQ ring mapping.
            unsafe { (*self.cq_head_kernel).store(head, Ordering::Release) };
        }
    }

    fn submit_and_wait(&mut self, min_complete: u32) -> io::Result<u32> {
        if self.pending_sqes > 0 {
            // SAFETY: publishes all SQEs written up to `sq_tail` to the kernel.
            unsafe { (*self.sq_tail_kernel).store(self.sq_tail, Ordering::Release) };
        }
        let to_submit = self.pending_sqes;
        if to_submit == 0 && min_complete == 0 {
            return Ok(0);
        }
        let flags = if min_complete > 0 { IORING_ENTER_GETEVENTS } else { 0 };
        loop {
            // SAFETY: valid `io_uring_enter` syscall on our owned `fd`.
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_io_uring_enter,
                    self.fd.as_raw_fd(),
                    to_submit,
                    min_complete,
                    flags,
                    ptr::null::<libc::sigset_t>(),
                    0usize,
                )
            };
            if ret >= 0 {
                let submitted = ret as u32;
                self.pending_sqes = self.pending_sqes.saturating_sub(submitted);
                return Ok(submitted);
            }
            let err = io::Error::last_os_error();
            match err.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::EBUSY | libc::EAGAIN) => return Ok(0),
                _ => return Err(err),
            }
        }
    }
}

impl Drop for IoUring {
    fn drop(&mut self) {
        // SAFETY: all pointers were mapped in `IoUring::new` with these exact sizes.
        unsafe {
            libc::munmap(self.sqes_ptr.cast(), self.sqes_size);
            if !self.single_mmap {
                libc::munmap(self.cq_ptr.cast(), self.cq_size);
            }
            libc::munmap(self.sq_ptr.cast(), self.sq_size);
        }
    }
}

// ---------------------------------------------------------------------------
// Engine control channel (`UringCtl`)
// ---------------------------------------------------------------------------

/// Producer-facing control handle for the Linux `io_uring` ring thread.
pub(crate) struct UringCtl {
    eventfd: OwnedFd,
    wake_pending: AtomicBool,
    failed: AtomicBool,
    cancel_ids: Mutex<Vec<u64>>,
    flights: Mutex<Box<[Option<Flight>]>>,
}

impl UringCtl {
    /// Probe kernel `io_uring` support (honoring `SEISMIC_IO_DISABLE_URING`)
    /// and construct a control channel if available; returns `None` to fall
    /// back cleanly to the `pread` worker pool.
    pub(crate) fn try_new() -> Option<Self> {
        if let Ok(val) = std::env::var("SEISMIC_IO_DISABLE_URING") {
            let trimmed = val.trim();
            if trimmed == "1" || trimmed.eq_ignore_ascii_case("true") {
                tracing::debug!("io_uring disabled via SEISMIC_IO_DISABLE_URING; using pread worker pool");
                return None;
            }
        }
        Self::probe_and_create()
    }

    fn probe_and_create() -> Option<Self> {
        if let Err(err) = IoUring::probe() {
            tracing::info!(
                error = %err,
                "io_uring unavailable; falling back to pread worker pool"
            );
            return None;
        }
        // SAFETY: `eventfd` creates a new non-blocking close-on-exec descriptor.
        let efd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if efd < 0 {
            let err = io::Error::last_os_error();
            tracing::warn!(
                error = %err,
                "eventfd creation failed; falling back to pread worker pool"
            );
            return None;
        }
        // SAFETY: `efd` is a valid fd owned exclusively by `UringCtl`.
        let eventfd = unsafe { OwnedFd::from_raw_fd(efd) };
        Some(Self {
            eventfd,
            wake_pending: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            cancel_ids: Mutex::new(Vec::new()),
            flights: Mutex::new(vec![None; MAX_INFLIGHT].into_boxed_slice()),
        })
    }

    pub(crate) fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub(crate) fn mark_failed(&self) {
        self.failed.store(true, Ordering::Release);
    }

    /// Create a fresh `IoUring` for the ring thread.
    pub(crate) fn open_ring(&self) -> io::Result<IoUring> {
        IoUring::new(RING_ENTRIES)
    }

    /// Coalesced wakeup: signals `eventfd` at most once until the ring thread
    /// drains it.
    pub(crate) fn wake(&self) {
        if !self.wake_pending.swap(true, Ordering::SeqCst) {
            let val: u64 = 1;
            // SAFETY: `eventfd` is valid for the lifetime of `UringCtl`.
            let _ = unsafe { libc::write(self.eventfd.as_raw_fd(), (&raw const val).cast(), size_of::<u64>()) };
        }
    }

    /// Request asynchronous cancellation of any in-flight reads belonging to
    /// `request_id`.
    pub(crate) fn cancel_inflight(&self, request_id: u64) {
        self.cancel_ids.lock().expect("uring cancel_ids").push(request_id);
        self.wake();
    }

    /// Snapshot live in-flight `Flight`s for [`crate::engine_work`].
    pub(crate) fn snapshot_flights(&self, out: &mut Vec<Flight>) {
        let flights = self.flights.lock().expect("uring flights");
        out.extend(flights.iter().flatten().cloned());
    }

    fn drain_eventfd(&self) {
        let mut buf: u64 = 0;
        loop {
            // SAFETY: `eventfd` is non-blocking; reading 8 bytes drains its counter.
            let n = unsafe { libc::read(self.eventfd.as_raw_fd(), (&raw mut buf).cast(), size_of::<u64>()) };
            if n <= 0 {
                break;
            }
        }
        self.wake_pending.store(false, Ordering::SeqCst);
    }

    fn take_cancels(&self, out: &mut Vec<u64>) {
        out.clear();
        let mut ids = self.cancel_ids.lock().expect("uring cancel_ids");
        out.append(&mut *ids);
    }

    fn insert_flight(&self, slot_idx: usize, flight: Flight) {
        self.flights.lock().expect("uring flights")[slot_idx] = Some(flight);
    }

    fn remove_flight(&self, slot_idx: usize) {
        self.flights.lock().expect("uring flights")[slot_idx] = None;
    }
}

// ---------------------------------------------------------------------------
// In-flight operation state & ring driver loop
// ---------------------------------------------------------------------------

enum OpMode {
    /// Single aligned direct read straight into `taken.piece.parts[0].sink`.
    DirectSingle { sink: NonNull<u8> },
    /// Multi-part contiguous aligned scatter-gather direct read via `IORING_OP_READV`.
    /// `iovecs` is heap-boxed so its pointer stays pinned while in flight.
    DirectVectored { iovecs: Box<[libc::iovec]>, idx: usize },
    /// Read into an aligned host bounce buffer, copied to parts on completion.
    Bounced(Buffer),
}

// SAFETY: `DirectSingle::sink` and `DirectVectored::iovecs` point into
// destinations owned by `InFlightOp::taken.state` until the request ends.
unsafe impl Send for OpMode {}

struct InFlightOp {
    user_data: u64,
    taken: Taken,
    fd: RawFd,
    span: usize,
    wanted: usize,
    done: usize,
    direct: bool,
    drop_after: bool,
    cancelling: bool,
    mode: OpMode,
}

impl InFlightOp {
    /// Build the next SQE for this op (initial submission or short-read / fallback retry).
    fn build_sqe(&self) -> io::Result<IoUringSqe> {
        let offset = self.taken.piece.start + self.done as u64;
        let remaining = self.span - self.done;
        let len = u32::try_from(remaining)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "read piece exceeds u32::MAX"))?;
        match &self.mode {
            OpMode::DirectSingle { sink } => {
                // SAFETY: `self.done < self.span`, stays within the part's destination slice.
                let ptr = unsafe { sink.as_ptr().add(self.done) };
                Ok(IoUringSqe {
                    opcode: IORING_OP_READ,
                    fd: self.fd,
                    off: offset,
                    addr: ptr as u64,
                    len,
                    user_data: self.user_data,
                    ..IoUringSqe::default()
                })
            }
            OpMode::Bounced(buffer) => {
                // SAFETY: `self.done < self.span <= buffer.len()`.
                let base = crate::Destination::region(buffer).0.as_ptr();
                let ptr = unsafe { base.add(self.done) };
                Ok(IoUringSqe {
                    opcode: IORING_OP_READ,
                    fd: self.fd,
                    off: offset,
                    addr: ptr as u64,
                    len,
                    user_data: self.user_data,
                    ..IoUringSqe::default()
                })
            }
            OpMode::DirectVectored { iovecs, idx } => {
                let rem = &iovecs[*idx..];
                Ok(IoUringSqe {
                    opcode: IORING_OP_READV,
                    fd: self.fd,
                    off: offset,
                    addr: rem.as_ptr() as u64,
                    len: rem.len() as u32,
                    user_data: self.user_data,
                    ..IoUringSqe::default()
                })
            }
        }
    }

    fn advance_vectored(&mut self, mut advance: usize) {
        let OpMode::DirectVectored { iovecs, idx } = &mut self.mode else {
            return;
        };
        while advance > 0 && *idx < iovecs.len() {
            if advance >= iovecs[*idx].iov_len {
                advance -= iovecs[*idx].iov_len;
                iovecs[*idx].iov_len = 0;
                *idx += 1;
            } else {
                // SAFETY: `advance < iov_len`, stays within the part's slice.
                iovecs[*idx].iov_base = unsafe { (iovecs[*idx].iov_base as *mut u8).add(advance).cast() };
                iovecs[*idx].iov_len -= advance;
                advance = 0;
            }
        }
    }
}

fn take_scratch(local: &mut Vec<Buffer>, engine: &Engine, span: usize) -> Buffer {
    if let Some(pos) = local.iter().position(|buf| buf.len() >= span) {
        return local.swap_remove(pos);
    }
    engine.buffers.take(span)
}

fn return_scratch(local: &mut Vec<Buffer>, engine: &Engine, buffer: Buffer) {
    if local.len() < MAX_LOCAL_SCRATCH {
        local.push(buffer);
    } else {
        engine.buffers.give(buffer);
    }
}

fn drain_scratch(local: &mut Vec<Buffer>, engine: &Engine) {
    for buffer in local.drain(..) {
        engine.buffers.give(buffer);
    }
}

fn complete_success(engine: &Engine, ctl: &UringCtl, op: InFlightOp, local_scratch: &mut Vec<Buffer>) {
    let now = Instant::now();
    let piece = &op.taken.piece;
    crate::timeline::note_at(
        now,
        crate::timeline::Kind::Read,
        0,
        (piece.start as i64, (piece.end - piece.start) as i64, 0),
    );
    let direct = !matches!(op.mode, OpMode::Bounced(_));
    engine.read_returned_uring(op.taken.returned(now, direct));
    ctl.remove_flight((op.user_data & 0xFF) as usize);

    let land_result = match op.mode {
        OpMode::DirectSingle { .. } | OpMode::DirectVectored { .. } => Ok(()),
        OpMode::Bounced(mut buffer) => {
            let res = op.taken.state.submitter.run(|| {
                let bytes = buffer.bytes_mut();
                for part in &piece.parts {
                    // SAFETY: `part.length` bytes of the part's sink, owned by this request until it ends.
                    unsafe {
                        ptr::copy_nonoverlapping(bytes[part.from..].as_ptr(), part.sink.as_ptr(), part.length);
                    }
                }
            });
            return_scratch(local_scratch, engine, buffer);
            res
        }
    };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match land_result {
        Ok(()) => op.taken.state.pieces_ended(1, [], None),
        Err(panic) => op.taken.state.pieces_ended(1, [], Some(panic)),
    }));
}

fn complete_cancelled(engine: &Engine, ctl: &UringCtl, op: InFlightOp, local_scratch: &mut Vec<Buffer>) {
    if let OpMode::Bounced(buffer) = op.mode {
        return_scratch(local_scratch, engine, buffer);
    }
    let now = Instant::now();
    let piece = &op.taken.piece;
    engine.read_returned_uring(op.taken.returned(now, false));
    ctl.remove_flight((op.user_data & 0xFF) as usize);

    let cancelled = Arc::new(Error::Cancelled);
    let failed: Vec<(usize, Arc<Error>)> = piece
        .parts
        .iter()
        .map(|part| part.range)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|range| (range, cancelled.clone()))
        .collect();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        op.taken.state.pieces_ended(1, failed, None);
    }));
}

fn complete_failure(
    engine: &Engine,
    ctl: &UringCtl,
    op: InFlightOp,
    error: Error,
    local_scratch: &mut Vec<Buffer>,
) {
    if let OpMode::Bounced(buffer) = op.mode {
        return_scratch(local_scratch, engine, buffer);
    }
    let piece = &op.taken.piece;
    let file = &op.taken.file;
    // Isolate failures on multi-part pieces by retrying each part alone (matching `work()`).
    let read = op.taken.state.submitter.run(|| {
        if piece.parts.len() == 1 {
            vec![(piece.parts[0].range, Arc::new(error))]
        } else {
            let mut single_scratch = local_scratch.pop();
            let failed: Vec<_> = piece
                .parts_alone()
                .filter_map(|alone| {
                    let range = alone.parts[0].range;
                    file.read_piece(&alone, &mut single_scratch, &engine.buffers)
                        .err()
                        .map(|err| (range, Arc::new(err)))
                })
                .collect();
            if let Some(buf) = single_scratch {
                return_scratch(local_scratch, engine, buf);
            }
            failed
        }
    });
    let now = Instant::now();
    crate::timeline::note_at(
        now,
        crate::timeline::Kind::Read,
        0,
        (piece.start as i64, (piece.end - piece.start) as i64, 0),
    );
    engine.read_returned_uring(op.taken.returned(now, false));
    ctl.remove_flight((op.user_data & 0xFF) as usize);

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match read {
        Ok(failed) => op.taken.state.pieces_ended(1, failed, None),
        Err(panic) => op.taken.state.pieces_ended(1, [], Some(panic)),
    }));
}

/// Main loop for the dedicated `seismic-uring` thread.
pub(crate) fn run_ring(engine: &'static Engine, ctl: &'static UringCtl, mut ring: IoUring) {
    crate::waited_on();

    let mut slots: Vec<Option<InFlightOp>> = (0..MAX_INFLIGHT).map(|_| None).collect();
    let mut free_slots: Vec<usize> = (0..MAX_INFLIGHT).rev().collect();
    let mut inflight_count: usize = 0;
    let mut generation: u64 = 1;
    let mut poll_armed = false;

    let mut cqe_buf: Vec<IoUringCqe> = Vec::with_capacity(64);
    let mut resubmit: Vec<usize> = Vec::with_capacity(16);
    let mut cancels: Vec<u64> = Vec::with_capacity(16);
    let mut cancel_set: HashSet<u64> = HashSet::new();
    let mut batch_taken: Vec<Taken> = Vec::with_capacity(MAX_INFLIGHT);
    let mut local_scratch: Vec<Buffer> = Vec::with_capacity(MAX_LOCAL_SCRATCH);

    loop {
        // 1. Drain all completed CQEs from the ring (zero syscalls).
        ring.drain_cqes(&mut cqe_buf);
        resubmit.clear();

        for cqe in cqe_buf.drain(..) {
            if cqe.user_data == WAKE_USER_DATA {
                poll_armed = false;
                ctl.drain_eventfd();
                continue;
            }
            if cqe.user_data == CANCEL_USER_DATA {
                continue;
            }
            let slot_idx = (cqe.user_data & 0xFF) as usize;
            let matches = slots
                .get(slot_idx)
                .and_then(Option::as_ref)
                .is_some_and(|op| op.user_data == cqe.user_data);
            if !matches {
                continue;
            }

            if cqe.res == -libc::EINTR || cqe.res == -libc::EAGAIN {
                resubmit.push(slot_idx);
                continue;
            }
            if cqe.res == -libc::ECANCELED {
                let op = slots[slot_idx].take().expect("matched slot");
                free_slots.push(slot_idx);
                inflight_count -= 1;
                complete_cancelled(engine, ctl, op, &mut local_scratch);
                continue;
            }
            if cqe.res < 0 {
                let errno = -cqe.res;
                let io_err = io::Error::from_raw_os_error(errno);
                let op = slots[slot_idx].as_mut().expect("matched slot");

                if op.direct && !(op.done as u64).is_multiple_of(ALIGNMENT) {
                    let op = slots[slot_idx].take().expect("matched slot");
                    free_slots.push(slot_idx);
                    inflight_count -= 1;
                    let err = io::Error::new(
                        io_err.kind(),
                        format!(
                            "a direct read at {}, misaligned {} bytes in by a short read: {io_err}",
                            op.taken.piece.start, op.done
                        ),
                    );
                    complete_failure(engine, ctl, op, err.into(), &mut local_scratch);
                    continue;
                }

                if op.direct && refuses_direct_sink(&io_err) && !matches!(op.mode, OpMode::Bounced(_)) {
                    // Direct I/O into device-mapped sink (`VM_IO`/`VM_PFNMAP`) failed `EFAULT`;
                    // bounce through aligned host scratch while keeping `O_DIRECT` active.
                    let buf = take_scratch(&mut local_scratch, engine, op.span);
                    op.mode = OpMode::Bounced(buf);
                    op.done = 0;
                    resubmit.push(slot_idx);
                    continue;
                }

                if op.direct && refuses_direct(&io_err) {
                    match op.taken.file.activate_direct_fallback(&io_err) {
                        Ok(cached) => {
                            op.fd = cached.as_raw_fd();
                            op.direct = false;
                            op.drop_after = true;
                            op.done = 0;
                            if matches!(op.mode, OpMode::DirectVectored { .. }) {
                                let (arr, count) = build_iovecs(&op.taken.piece.parts);
                                op.mode = OpMode::DirectVectored {
                                    iovecs: arr[..count].to_vec().into_boxed_slice(),
                                    idx: 0,
                                };
                            }
                            resubmit.push(slot_idx);
                            continue;
                        }
                        Err(err) => {
                            let op = slots[slot_idx].take().expect("matched slot");
                            free_slots.push(slot_idx);
                            inflight_count -= 1;
                            complete_failure(engine, ctl, op, err, &mut local_scratch);
                            continue;
                        }
                    }
                }

                let op = slots[slot_idx].take().expect("matched slot");
                free_slots.push(slot_idx);
                inflight_count -= 1;
                complete_failure(engine, ctl, op, io_err.into(), &mut local_scratch);
                continue;
            }

            // cqe.res >= 0
            let n = cqe.res as usize;
            let op = slots[slot_idx].as_mut().expect("matched slot");
            if n == 0 {
                if op.done < op.wanted {
                    let op = slots[slot_idx].take().expect("matched slot");
                    free_slots.push(slot_idx);
                    inflight_count -= 1;
                    complete_failure(engine, ctl, op, truncated().into(), &mut local_scratch);
                } else {
                    let op = slots[slot_idx].take().expect("matched slot");
                    free_slots.push(slot_idx);
                    inflight_count -= 1;
                    finish_read(engine, ctl, op, &mut local_scratch);
                }
                continue;
            }
            op.done += n;
            op.advance_vectored(n);
            if op.done < op.wanted {
                resubmit.push(slot_idx);
            } else {
                let op = slots[slot_idx].take().expect("matched slot");
                free_slots.push(slot_idx);
                inflight_count -= 1;
                finish_read(engine, ctl, op, &mut local_scratch);
            }
        }

        // 2. Re-queue short-read continuations and fallback retries.
        for slot_idx in resubmit.drain(..) {
            if let Some(op) = slots[slot_idx].as_ref() {
                match op.build_sqe() {
                    Ok(sqe) => {
                        if !ring.push_sqe(sqe) {
                            let _ = ring.submit_and_wait(0);
                            assert!(ring.push_sqe(sqe), "io_uring SQ remained full after flush");
                        }
                    }
                    Err(err) => {
                        let op = slots[slot_idx].take().expect("slot present");
                        free_slots.push(slot_idx);
                        inflight_count -= 1;
                        complete_failure(engine, ctl, op, err.into(), &mut local_scratch);
                    }
                }
            }
        }

        // 3. Issue `IORING_OP_ASYNC_CANCEL` for any in-flight ops whose `Handle` dropped.
        ctl.take_cancels(&mut cancels);
        if !cancels.is_empty() {
            cancel_set.clear();
            cancel_set.extend(cancels.drain(..));
            for op in slots.iter_mut().flatten() {
                if !op.cancelling && cancel_set.contains(&op.taken.id) {
                    let cancel_sqe = IoUringSqe {
                        opcode: IORING_OP_ASYNC_CANCEL,
                        addr: op.user_data,
                        user_data: CANCEL_USER_DATA,
                        ..IoUringSqe::default()
                    };
                    if ring.sq_space() < 2 {
                        let _ = ring.submit_and_wait(0);
                    }
                    if ring.push_sqe(cancel_sqe) {
                        op.cancelling = true;
                    }
                }
            }
        }

        // 4. Batch-take ready pieces from `engine.queue()` in priority order.
        batch_taken.clear();
        let mut should_drain_scratch = false;
        {
            let mut queue = engine.queue();
            let room = MAX_INFLIGHT.saturating_sub(inflight_count);
            let sq_budget = ring.sq_space().saturating_sub(2);
            let take_limit = room.min(sq_budget);
            while batch_taken.len() < take_limit {
                let Some(taken) = queue.take() else {
                    break;
                };
                batch_taken.push(taken);
            }
            if batch_taken.is_empty() && inflight_count == 0 {
                if engine.retire_uring(&queue) {
                    drop(queue);
                    drain_scratch(&mut local_scratch, engine);
                    return;
                }
                should_drain_scratch = true;
            }
        }
        if should_drain_scratch {
            // Return all retained bounce buffers while idle so `read_buffers()`
            // drops to 0 once the last reader closes.
            drain_scratch(&mut local_scratch, engine);
        }

        // 5. Prepare & enqueue SQEs for every taken piece in this batch.
        for taken in batch_taken.drain(..) {
            let span = (taken.piece.end - taken.piece.start) as usize;
            let wanted = (taken.piece.end.min(taken.file.size).saturating_sub(taken.piece.start)) as usize;
            let (fd, direct, drop_after) = taken.file.fd_and_mode();

            let slot_idx = free_slots
                .pop()
                .expect("inflight_count < MAX_INFLIGHT guarantees a free slot");
            generation = (generation + 1) & 0x007F_FFFF_FFFF_FFFF;
            if generation == 0 {
                generation = 1;
            }
            let user_data = (generation << 8) | (slot_idx as u64);

            let mode = if taken.file.lands_straight(&taken.piece) {
                match taken.piece.parts.as_slice() {
                    [part] => OpMode::DirectSingle { sink: part.sink },
                    parts => {
                        let (arr, count) = build_iovecs(parts);
                        OpMode::DirectVectored {
                            iovecs: arr[..count].to_vec().into_boxed_slice(),
                            idx: 0,
                        }
                    }
                }
            } else {
                OpMode::Bounced(take_scratch(&mut local_scratch, engine, span))
            };

            let flight = Flight {
                path: taken.file.path.clone(),
                volume: taken.volume.id,
                offset: taken.piece.start,
                bytes: taken.piece.end - taken.piece.start,
                window: if taken.standing.due() {
                    taken.volume.shape.due
                } else {
                    taken.volume.shape.knee
                },
                since: Instant::now(),
            };
            ctl.insert_flight(slot_idx, flight);

            let op = InFlightOp {
                user_data,
                taken,
                fd,
                span,
                wanted,
                done: 0,
                direct,
                drop_after,
                cancelling: false,
                mode,
            };

            if op.wanted == 0 {
                free_slots.push(slot_idx);
                finish_read(engine, ctl, op, &mut local_scratch);
                continue;
            }

            match op.build_sqe() {
                Ok(sqe) => {
                    assert!(ring.push_sqe(sqe), "sq_budget guaranteed SQ space");
                    slots[slot_idx] = Some(op);
                    inflight_count += 1;
                }
                Err(err) => {
                    free_slots.push(slot_idx);
                    complete_failure(engine, ctl, op, err.into(), &mut local_scratch);
                }
            }
        }

        // 6. Arm `eventfd` poll before sleeping so producer wakeups interrupt `io_uring_enter`.
        if !poll_armed && ring.sq_space() >= 1 {
            let poll_sqe = IoUringSqe {
                opcode: IORING_OP_POLL_ADD,
                fd: ctl.eventfd.as_raw_fd(),
                op_flags: libc::POLLIN as u32,
                user_data: WAKE_USER_DATA,
                ..IoUringSqe::default()
            };
            if ring.push_sqe(poll_sqe) {
                poll_armed = true;
            }
        }

        // 7. Single `io_uring_enter`: submit the entire batch and wait for >= 1 CQE
        //    (unless CQEs are already sitting in the CQ ring).
        let min_complete = u32::from(ring.cq_ready() == 0);
        if min_complete > 0 || ring.has_pending_sqes() {
            if let Err(err) = ring.submit_and_wait(min_complete) {
                tracing::error!(error = %err, "fatal io_uring_enter error; quiescing ring and falling back to thread pool");
                ctl.mark_failed();
                let kind = err.kind();
                let msg = format!("io_uring_enter failed: {err}");
                for slot in &mut slots {
                    if let Some(op) = slot.take() {
                        complete_failure(engine, ctl, op, io::Error::new(kind, msg.clone()).into(), &mut local_scratch);
                    }
                }
                for buf in local_scratch.drain(..) {
                    engine.buffers.give(buf);
                }
                engine.fallback_from_uring();
                break;
            }
        }
    }
}

fn finish_read(engine: &Engine, ctl: &UringCtl, op: InFlightOp, local_scratch: &mut Vec<Buffer>) {
    if op.drop_after {
        if let Err(err) = op.taken.file.drop_cached_pages(op.taken.piece.start, op.taken.piece.end) {
            complete_failure(engine, ctl, op, err, local_scratch);
            return;
        }
    }
    complete_success(engine, ctl, op, local_scratch);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn io_uring_batched_raw_read_and_cancel() {
        if IoUring::probe().is_err() {
            return;
        }
        let path = std::env::temp_dir().join(format!("seismic-uring-raw-{}.bin", std::process::id()));
        let mut payload = vec![0u8; 64 * 1024];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xff) as u8;
        }
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
            f.sync_all().unwrap();
        }
        let file = std::fs::File::open(&path).unwrap();
        let mut ring = IoUring::new(32).unwrap();

        // Submit 8 batched 4 KiB reads in a single `io_uring_enter` syscall.
        let mut bufs: Vec<Buffer> = (0..8).map(|_| Buffer::new(4096)).collect();
        for (i, buf) in bufs.iter_mut().enumerate() {
            let sqe = IoUringSqe {
                opcode: IORING_OP_READ,
                fd: file.as_raw_fd(),
                off: (i * 4096) as u64,
                addr: buf.bytes_mut().as_mut_ptr() as u64,
                len: 4096,
                user_data: (i as u64) + 100,
                ..IoUringSqe::default()
            };
            assert!(ring.push_sqe(sqe));
        }

        let mut completed = 0usize;
        let mut cqes = Vec::new();
        while completed < 8 {
            ring.submit_and_wait(1).unwrap();
            ring.drain_cqes(&mut cqes);
            for cqe in &cqes {
                assert_eq!(cqe.res, 4096, "unexpected CQE res for user_data {}", cqe.user_data);
                completed += 1;
            }
        }
        for (i, buf) in bufs.iter().enumerate() {
            // SAFETY: all 8 reads completed above.
            let got = unsafe { buf.bytes() };
            assert_eq!(got, &payload[i * 4096..(i + 1) * 4096]);
        }
        let _ = std::fs::remove_file(&path);
    }
}
