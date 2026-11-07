//! The device's transfer queue (`seismic::Transfer`), the one owner of what
//! queues on it: the bank's promotions from a host tier, every one drained
//! by then in one batch, posted only while no batch runs. Nothing a step
//! waits on queues here: every datum a device wait awaits is read by device
//! work already in the graph behind it, so a batch held in the driver
//! behind a parked step delays only a promotion. A scratch-served prompt
//! group's begin waits, on the host before its steps exist, for a batch
//! copying into the lent bank records (`watcher::lent`); none into them
//! queues while it waits, so it waits on at most the batch running.
//!
//! An owner of serving posts a batch and never issues it: the queue's
//! issuer, a thread of its own, issues each in order ([`Copies::start`]).
//! An issue can wait: while host access to its tensors is held, and in the
//! driver behind the device's queue, which a parked step can fill. So no
//! owner of serving calls the device's queue, and none holds a lock another
//! needs while an issue waits. The queue's completions
//! are the load's events (`events`): its watch, a second thread, waits on
//! the queue's own completion of each batch issued (a semaphore wait where
//! the backend has one) and announces it, so no promotion looks at the
//! queue by time, and
//! [`Copies::completed`] reads only what the two threads learned.

use super::events::Events;
use seismic::{Tensor, Transfer, TransferItem, TransferRun};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// The longest one wait of the watch on the queue lasts before it looks
/// whether the load stopped: a bound, not a latency, as the batch's
/// completion ends the wait.
const BOUND: Duration = Duration::from_millis(50);

/// One run of a batch (`seismic::TransferRun`): `bytes` from `source` at
/// `source_offset` to `destination` at `destination_offset`, held by the
/// queue until its batch is issued.
pub(in crate::experts) struct Run {
    pub source: Tensor,
    pub source_offset: u64,
    pub destination: Tensor,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// The device's transfer queue, shared by every holder of it: the batches
/// posted, issued and complete, and the load's events, which each batch
/// posted, issued and complete is one of.
#[derive(Clone)]
pub(in crate::experts) struct Copies {
    inner: Arc<Inner>,
}

struct Inner {
    transfer: Transfer,
    events: Arc<Events>,
    state: Mutex<State>,
}

/// Where the queue's batches are, by ticket (consecutive from one, in the
/// order posted, which is the order issued and the order they complete).
#[derive(Default)]
struct State {
    /// The last ticket posted (0: none).
    posted: u64,
    /// The batches posted and not yet issued, in order.
    waiting: VecDeque<(u64, Vec<Run>)>,
    /// The batch being issued, and since when: an issue that waits shows
    /// here ([`Copies::describe`]).
    issuing: Option<(u64, std::time::Instant)>,
    /// Every ticket through this one is issued.
    issued: u64,
    /// The batches issued and not yet complete: each ticket and the value
    /// its batch completes at, in order.
    running: VecDeque<(u64, u64)>,
    /// The batches the queue refused as issued, and why.
    refused: BTreeMap<u64, String>,
    /// Why the queue faulted, once it has: no batch completes from then.
    fault: Option<String>,
}

impl Copies {
    pub fn new(transfer: Transfer, events: Arc<Events>) -> Self {
        Self { inner: Arc::new(Inner { transfer, events, state: Mutex::default() }) }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.state.lock().expect("the transfer queue's batches")
    }

    /// Post `runs` as a batch: its ticket. The queue's issuer issues it
    /// ([`Self::start`]); nothing that posts waits for the device's queue,
    /// which may hold an issue as long as host access to its tensors is
    /// held, or the device's queue is full behind a parked step.
    fn post(&self, runs: Vec<Run>) -> u64 {
        let ticket = {
            let mut state = self.state();
            state.posted += 1;
            let ticket = state.posted;
            state.waiting.push_back((ticket, runs));
            ticket
        };
        self.inner.events.notify();
        ticket
    }

    /// Post the batch `form` forms, only while no batch is posted and not
    /// complete, formed only
    /// then: what `form` returned beside its runs, with its ticket; none
    /// while one is (the caller offers it again once that one's completion
    /// is announced). Why the queue faulted.
    pub fn background<T>(&self, form: impl FnOnce() -> (Vec<Run>, T)) -> Result<Option<(T, u64)>, String> {
        {
            let state = self.state();
            if let Some(fault) = &state.fault {
                return Err(fault.clone());
            }
            if !state.waiting.is_empty() || !state.running.is_empty() {
                return Ok(None);
            }
        }
        let (runs, formed) = form();
        Ok(Some((formed, self.post(runs))))
    }

    /// Whether no batch is posted and not complete.
    pub fn idle(&self) -> bool {
        let state = self.state();
        state.waiting.is_empty() && state.running.is_empty() && state.issuing.is_none()
    }

    /// The batches posted so far.
    #[cfg(test)]
    pub fn posted(&self) -> u64 {
        self.state().posted
    }

    /// Whether the batch of `ticket` is complete; why it never will be (the
    /// queue refused it, or faulted). It reads what the queue's threads
    /// learned, and calls nothing of the device.
    pub fn completed(&self, ticket: u64) -> Result<bool, String> {
        let state = self.state();
        if let Some(reason) = state.refused.get(&ticket) {
            return Err(reason.clone());
        }
        if ticket > state.issued || state.running.iter().any(|&(running, _)| running == ticket) {
            return match &state.fault {
                Some(fault) => Err(fault.clone()),
                None => Ok(false),
            };
        }
        Ok(true)
    }

    /// Where the queue's batches are, for the exchange's watchdog: posted,
    /// waiting to be issued, being issued and for how long, issued and
    /// running, refused, and the queue's fault.
    pub fn describe(&self) -> String {
        let state = self.state();
        format!(
            "posted through {}, {} waiting to be issued, issuing {:?}, issued through {}, running {:?}, refused {:?}, fault {:?}",
            state.posted,
            state.waiting.len(),
            state.issuing.map(|(ticket, since)| (ticket, since.elapsed())),
            state.issued,
            state.running,
            state.refused,
            state.fault,
        )
    }

    /// The queue's two threads until `stopped` holds, each waiting on the
    /// load's events and on the device alone, so no owner of serving ever
    /// does:
    ///
    /// - its issuer issues each batch posted, in order
    ///   ([`Transfer::copy`], which waits while host access to its tensors
    ///   is held, and can wait in the driver behind the device's queue);
    /// - its watch waits on the queue's own completion of the earliest
    ///   batch issued (a semaphore wait where the backend has one) and
    ///   announces it, or the queue's fault.
    ///
    /// Each issue, refusal, completion and fault is an event. `stopped` is
    /// looked at as each wait ends, the load's stop being an event too.
    pub fn start(&self, stopped: impl Fn() -> bool + Send + Sync + 'static) -> Result<Vec<JoinHandle<()>>, String> {
        let stopped = Arc::new(stopped);
        let issuer = {
            let (copies, stopped) = (self.clone(), stopped.clone());
            std::thread::Builder::new()
                .name("expert-copies-issue".into())
                .spawn(move || {
                    seismic_io::waited_on();
                    copies.issue(&*stopped)
                })
                .map_err(|error| format!("the transfer queue's issuer: {error}"))?
        };
        let copies = self.clone();
        let watch = std::thread::Builder::new()
            .name("expert-copies".into())
            .spawn(move || {
                seismic_io::waited_on();
                copies.watch(&*stopped)
            })
            .map_err(|error| format!("the transfer queue's watch: {error}"))?;
        Ok(vec![issuer, watch])
    }

    /// The issuer ([`Self::start`]).
    fn issue(&self, stopped: &dyn Fn() -> bool) {
        let events = &self.inner.events;
        loop {
            let seen = events.seen();
            if stopped() {
                return;
            }
            let Some((ticket, runs)) = ({
                let mut state = self.state();
                let next = state.waiting.pop_front();
                state.issuing = next.as_ref().map(|&(ticket, _)| (ticket, std::time::Instant::now()));
                next
            }) else {
                events.wait(seen);
                continue;
            };
            let items = runs
                .iter()
                .map(|run| {
                    TransferItem::Run(TransferRun {
                        source: &run.source,
                        source_offset: run.source_offset,
                        destination: &run.destination,
                        destination_offset: run.destination_offset,
                        bytes: run.bytes,
                    })
                })
                .collect::<Vec<_>>();
            let issued = self.inner.transfer.copy(&items);
            {
                let mut state = self.state();
                state.issuing = None;
                state.issued = ticket;
                match issued {
                    Ok(value) => state.running.push_back((ticket, value)),
                    Err(error) => drop(state.refused.insert(ticket, error.to_string())),
                }
            }
            events.notify();
        }
    }

    /// The watch ([`Self::start`]).
    fn watch(&self, stopped: &dyn Fn() -> bool) {
        let events = &self.inner.events;
        loop {
            let seen = events.seen();
            if stopped() {
                return;
            }
            let earliest = {
                let state = self.state();
                state.fault.is_none().then(|| state.running.front().map(|&(_, value)| value)).flatten()
            };
            let Some(value) = earliest else {
                events.wait(seen);
                continue;
            };
            match self.inner.transfer.wait(value, BOUND) {
                Ok(false) => continue,
                Ok(true) => drop(self.state().running.pop_front()),
                Err(error) => self.state().fault = Some(error.to_string()),
            }
            events.notify();
        }
    }
}
