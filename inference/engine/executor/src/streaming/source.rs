//! The file source (spec 5.1): stored records read straight from the model
//! file into staging, around the OS page cache, by persistent workers.
//!
//! Reads come in two classes. A demand read is one a parked step waits on; a
//! prefetch is a prediction. Workers always take a queued demand read before
//! any prefetch, a prefetch that has not started can be cancelled, and a
//! queued prefetch can be promoted to demand when a step turns out to need it.
//! Without demand priority and cancellation, wrong prefetches delay needed
//! reads and prefetching is slower than none (theory §3, §6).

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// Persistent workers. More did not raise bandwidth on any machine measured
/// (theory §2).
pub(crate) const WORKERS: usize = 8;

/// Reads around the page cache take their fast path only for block-aligned
/// file offsets, lengths and buffers; anything else falls back to a slow
/// path (about 50 MB/s against 2.45 GB/s on an M4 Pro instance). A worker
/// reads a range's aligned cover into its own aligned buffer and copies the
/// range's exact bytes into staging.
const BLOCK: usize = 16384;

/// A worker's block-aligned read buffer, grown to the largest cover read.
struct AlignedBuffer {
    base: Option<std::ptr::NonNull<u8>>,
    capacity: usize,
}

impl AlignedBuffer {
    fn new() -> Self {
        Self {
            base: None,
            capacity: 0,
        }
    }

    fn layout(capacity: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(capacity, BLOCK).expect("a block-aligned layout")
    }

    /// At least `bytes` aligned bytes.
    fn get(&mut self, bytes: usize) -> &mut [u8] {
        if bytes > self.capacity {
            self.release();
            let layout = Self::layout(bytes);
            // A nonzero, block-aligned layout.
            let base = unsafe { std::alloc::alloc(layout) };
            self.base = Some(
                std::ptr::NonNull::new(base)
                    .unwrap_or_else(|| std::alloc::handle_alloc_error(layout)),
            );
            self.capacity = bytes;
        }
        let base = self.base.expect("allocated above");
        // `capacity >= bytes` bytes owned by this buffer.
        unsafe { std::slice::from_raw_parts_mut(base.as_ptr(), bytes) }
    }

    fn release(&mut self) {
        if let Some(base) = self.base.take() {
            // Allocated with this layout.
            unsafe { std::alloc::dealloc(base.as_ptr(), Self::layout(self.capacity)) };
        }
        self.capacity = 0;
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        self.release();
    }
}

/// Read `range` through its block-aligned cover. The cover may run past the
/// end of the file; only the range's own bytes must be read.
fn read_range(file: &File, range: &Range, buffer: &mut AlignedBuffer) -> std::io::Result<()> {
    let block = BLOCK as u64;
    let start = range.offset / block * block;
    let head = (range.offset - start) as usize;
    let needed = head + range.bytes;
    let cover = needed.div_ceil(BLOCK) * BLOCK;
    let buffer = buffer.get(cover);
    let mut done = 0;
    while done < needed {
        match file.read_at(&mut buffer[done..cover], start + done as u64) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => done += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    // The destination is staging reserved for this record until its ticket
    // ends.
    let destination = unsafe { std::slice::from_raw_parts_mut(range.destination, range.bytes) };
    destination.copy_from_slice(&buffer[head..needed]);
    Ok(())
}

/// The model file, opened for reads that bypass the page cache.
pub(crate) struct ModelFile(File);

impl ModelFile {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        bypass_page_cache(&file).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self(file))
    }
}

#[cfg(target_os = "macos")]
fn bypass_page_cache(file: &File) -> std::io::Result<()> {
    // Faulting experts in through the page cache is the 3 tok/s case
    // (spec §3 finding 5).
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn bypass_page_cache(file: &File) -> std::io::Result<()> {
    // Linux page-cache advice for a sequential-random reader; O_DIRECT needs
    // aligned destinations the staging layout does not promise.
    let status =
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status));
    }
    Ok(())
}

/// One contiguous range of the model file into host-addressed staging.
pub(crate) struct Range {
    pub offset: u64,
    pub bytes: usize,
    pub destination: *mut u8,
}

// The destination is staging the source's owner keeps alive and does not
// touch until the read completes.
unsafe impl Send for Range {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Priority {
    Demand,
    Prefetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Every range has its bytes.
    Done,
    /// A read failed; the staging record holds nothing usable.
    Failed,
    /// Cancelled before any range started.
    Cancelled,
}

/// A record's reads: complete when every range is.
pub(crate) struct Ticket {
    state: Mutex<TicketState>,
    changed: Condvar,
}

struct TicketState {
    /// Ranges still queued.
    queued: usize,
    /// Ranges still being read.
    running: usize,
    failed: bool,
    cancelled: bool,
    priority: Priority,
}

impl Ticket {
    #[cfg(test)]
    pub(crate) fn controlled() -> Arc<Self> {
        let ticket = Self::done(Priority::Demand);
        { let mut state = ticket.state.lock().unwrap(); state.running = 1; }
        ticket
    }
    #[cfg(test)]
    pub(crate) fn finish_controlled(&self, failed: bool) {
        let mut state = self.state.lock().unwrap();
        state.running = 0; state.failed = failed;
        self.changed.notify_all();
    }
    /// A record whose bytes are already in staging: no read.
    pub(crate) fn done(priority: Priority) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(TicketState {
                queued: 0,
                running: 0,
                failed: false,
                cancelled: false,
                priority,
            }),
            changed: Condvar::new(),
        })
    }

    fn outcome(state: &TicketState) -> Option<Outcome> {
        if state.cancelled {
            return Some(Outcome::Cancelled);
        }
        (state.queued == 0 && state.running == 0).then_some(if state.failed {
            Outcome::Failed
        } else {
            Outcome::Done
        })
    }

    /// The outcome, if the reads have ended.
    pub(crate) fn poll(&self) -> Option<Outcome> {
        Self::outcome(&self.state.lock().expect("ticket lock"))
    }

    /// Wait for the reads to end, at most until `deadline`.
    pub(crate) fn wait_until(&self, deadline: Instant) -> Option<Outcome> {
        let mut state = self.state.lock().expect("ticket lock");
        loop {
            if let Some(outcome) = Self::outcome(&state) {
                return Some(outcome);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            state = self
                .changed
                .wait_timeout(state, deadline - now)
                .expect("ticket lock")
                .0;
        }
    }

    pub(crate) fn priority(&self) -> Priority {
        self.state.lock().expect("ticket lock").priority
    }
}

struct Job {
    ticket: Arc<Ticket>,
    range: Range,
}

struct Queues {
    demand: VecDeque<Job>,
    prefetch: VecDeque<Job>,
    stopping: bool,
}

struct Shared {
    file: ModelFile,
    queues: Mutex<Queues>,
    work: Condvar,
}

pub(crate) struct ExpertSource {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl ExpertSource {
    pub(crate) fn new(file: ModelFile) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            file,
            queues: Mutex::new(Queues {
                demand: VecDeque::new(),
                prefetch: VecDeque::new(),
                stopping: false,
            }),
            work: Condvar::new(),
        });
        let workers = (0..WORKERS)
            .map(|index| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("expert-source-{index}"))
                    .spawn(move || work(&shared))
                    .map_err(|error| format!("expert source worker: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { shared, workers })
    }

    /// Queue a record's ranges.
    pub(crate) fn submit(&self, ranges: Vec<Range>, priority: Priority) -> Arc<Ticket> {
        let ticket = Arc::new(Ticket {
            state: Mutex::new(TicketState {
                queued: ranges.len(),
                running: 0,
                failed: false,
                cancelled: false,
                priority,
            }),
            changed: Condvar::new(),
        });
        let mut queues = self.shared.queues.lock().expect("source queues");
        let queue = match priority {
            Priority::Demand => &mut queues.demand,
            Priority::Prefetch => &mut queues.prefetch,
        };
        for range in ranges {
            queue.push_back(Job {
                ticket: Arc::clone(&ticket),
                range,
            });
        }
        drop(queues);
        self.shared.work.notify_all();
        ticket
    }

    /// A prefetch a parked step now needs: its queued ranges move to the
    /// demand queue, ahead of every other prefetch.
    pub(crate) fn promote(&self, ticket: &Arc<Ticket>) {
        let mut queues = self.shared.queues.lock().expect("source queues");
        ticket.state.lock().expect("ticket lock").priority = Priority::Demand;
        let mut kept = VecDeque::with_capacity(queues.prefetch.len());
        while let Some(job) = queues.prefetch.pop_front() {
            if Arc::ptr_eq(&job.ticket, ticket) {
                queues.demand.push_back(job);
            } else {
                kept.push_back(job);
            }
        }
        queues.prefetch = kept;
    }

    /// Cancel a prefetch none of whose ranges has started; true when
    /// cancelled. A started one runs to the end.
    pub(crate) fn cancel(&self, ticket: &Arc<Ticket>) -> bool {
        let mut queues = self.shared.queues.lock().expect("source queues");
        let mut state = ticket.state.lock().expect("ticket lock");
        if state.running > 0 || state.cancelled || Ticket::outcome(&state).is_some() {
            return state.cancelled;
        }
        queues
            .prefetch
            .retain(|job| !Arc::ptr_eq(&job.ticket, ticket));
        queues.demand.retain(|job| !Arc::ptr_eq(&job.ticket, ticket));
        state.queued = 0;
        state.cancelled = true;
        drop(state);
        ticket.changed.notify_all();
        true
    }
}

impl Drop for ExpertSource {
    fn drop(&mut self) {
        self.shared.queues.lock().expect("source queues").stopping = true;
        self.shared.work.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn work(shared: &Shared) {
    let mut buffer = AlignedBuffer::new();
    loop {
        let job = {
            let mut queues = shared.queues.lock().expect("source queues");
            loop {
                if let Some(job) = queues.demand.pop_front().or_else(|| queues.prefetch.pop_front()) {
                    let mut state = job.ticket.state.lock().expect("ticket lock");
                    state.queued -= 1;
                    state.running += 1;
                    drop(state);
                    break job;
                }
                if queues.stopping {
                    return;
                }
                queues = shared.work.wait(queues).expect("source queues");
            }
        };
        let failed = read_range(&shared.file.0, &job.range, &mut buffer).is_err();
        let mut state = job.ticket.state.lock().expect("ticket lock");
        state.running -= 1;
        state.failed |= failed;
        drop(state);
        job.ticket.changed.notify_all();
    }
}
