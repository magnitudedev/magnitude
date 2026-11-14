//! The OS's memory pressure events of a store ([`Pressure`]): one watch a
//! store, whichever loads watch it, each event counted once in the store's
//! ledger and waking every load that watches.

use super::*;
use std::sync::atomic::AtomicBool;

/// A store's pressure events (`seismic::PressureEvents`: the macOS
/// memorypressure source, a Linux PSI trigger, Windows' low-memory
/// notification), shared by every load in the scope through the store's
/// one ledger. The first load that watches starts the one watch, a thread
/// of its own; each event is counted once, for the store's next reading to
/// take ([`Ledger::read`]), and wakes every load that watches.
/// The watch stops as the last load's watch drops. Where the OS gives no
/// unprivileged events (a Linux kernel or container without writable PSI
/// triggers), none is watched in the process again: the loads read at
/// their other events.
#[derive(Default)]
pub(super) struct Pressure {
    /// Events the OS delivered that no reading of the store took yet,
    /// counted under the watchers' lock ([`Self::event`]).
    events: AtomicU64,
    /// The times the OS's events ended without a way to set them again
    /// ([`seismic::Wake::Lost`]): from then on the store's pressure is read
    /// at its other events only.
    lost: AtomicU64,
    pub(super) watchers: Mutex<Watchers>,
}

#[derive(Default)]
pub(super) struct Watchers {
    next: u64,
    /// Each watching load's wake, by its watch's id.
    woken: Vec<(u64, Arc<dyn Fn() + Send + Sync>)>,
    /// The watches started so far: the number of the next ([`Thread::watch`]).
    started: u64,
    pub(super) watch: Watch,
}

/// Whether the store's events are watched.
#[derive(Default)]
pub(super) enum Watch {
    /// No load watches.
    #[default]
    Unasked,
    /// The one watch, while a load watches.
    Events(Thread),
    /// The OS gives no unprivileged events.
    Unwatchable,
}

/// The thread waiting for the OS's events: stopped and joined as it drops.
pub(super) struct Thread {
    /// Which of the store's watches it is: an event it delivers counts only
    /// while it is the current one ([`Pressure::event`]).
    watch: u64,
    /// Wakes the thread so it sees it is stopped.
    wake: Box<dyn Fn() + Send>,
    stopped: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Thread {
    /// Stop: the thread ends at its next wake, and an event it delivers
    /// from now on is not counted, its watch no longer current.
    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        (self.wake)();
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        self.stop();
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the memory pressure thread does not panic");
        }
    }
}

impl Pressure {
    /// One event of watch `watch`: counted for the store's next reading
    /// under the watchers' lock, then every watching load woken; dropped
    /// where `watch` is no longer the store's current one, so a watch
    /// restarted beside its stopping thread counts each OS event once.
    fn event(&self, watch: u64) {
        let watchers = self.watchers();
        if !matches!(&watchers.watch, Watch::Events(thread) if thread.watch == watch) {
            return;
        }
        self.count(watchers);
    }

    /// One event a test delivers in the OS's place, counted as a current
    /// watch's is.
    #[cfg(test)]
    pub(super) fn event_now(&self) {
        self.count(self.watchers());
    }

    /// Count one event under `watchers`, then wake every watching load
    /// with the lock let go.
    fn count(&self, watchers: MutexGuard<'_, Watchers>) {
        self.events.fetch_add(1, Ordering::AcqRel);
        let woken = watchers.woken.iter().map(|(_, wake)| wake.clone()).collect::<Vec<_>>();
        drop(watchers);
        woken.iter().for_each(|wake| wake());
    }

    /// The OS's events ended and cannot be set again: no more will come.
    fn lose(&self) {
        tracing::warn!(target: crate::MEMORY_TRACE, "the OS's memory pressure events ended and cannot be set again: the store reads pressure at its other events only");
        self.lost.fetch_add(1, Ordering::AcqRel);
    }

    /// The events no reading took yet, taken now.
    pub(super) fn take(&self) -> u64 {
        self.events.swap(0, Ordering::AcqRel)
    }

    fn watchers(&self) -> MutexGuard<'_, Watchers> {
        self.watchers.lock().expect("pressure watchers lock poisoned")
    }

    /// Watch the store's events from now on: `wake` runs on each, the
    /// watch started where none runs yet. None where the OS gives none.
    fn watch(self: &Arc<Self>, wake: Arc<dyn Fn() + Send + Sync>) -> Option<PressureWatch> {
        let mut watchers = self.watchers();
        if matches!(watchers.watch, Watch::Unasked) {
            watchers.watch = match seismic::PressureEvents::open() {
                Some(events) => {
                    let watch = watchers.started;
                    watchers.started += 1;
                    Watch::Events(self.start(events, watch))
                }
                None => Watch::Unwatchable,
            };
        }
        let watchers = &mut *watchers;
        match &watchers.watch {
            Watch::Unasked => unreachable!("the watch was just asked"),
            Watch::Unwatchable => None,
            Watch::Events(_) => {
                let id = watchers.next;
                watchers.next += 1;
                watchers.woken.push((id, wake));
                Some(PressureWatch { pressure: self.clone(), id })
            }
        }
    }

    fn start(self: &Arc<Self>, events: seismic::PressureEvents, watch: u64) -> Thread {
        let (waker, stopped) = (events.waker(), Arc::new(AtomicBool::new(false)));
        let wake = Box::new(move || waker.wake());
        let (stopping, pressure) = (stopped.clone(), self.clone());
        let thread = std::thread::Builder::new()
            .name("magnitude-memory-pressure".into())
            .spawn(move || loop {
                match events.wait() {
                    _ if stopping.load(Ordering::Acquire) => return,
                    seismic::Wake::Pressure => pressure.event(watch),
                    seismic::Wake::Lost => pressure.lose(),
                    seismic::Wake::Asked => {}
                }
            })
            .expect("invariant violated: the memory pressure thread could not start");
        Thread { watch, wake, stopped, thread: Some(thread) }
    }
}

/// One load's watch of its host RAM store's pressure events
/// ([`Ledger::watch_pressure`]): its wake runs on each until it drops. The
/// last to drop stops the store's watch.
pub struct PressureWatch {
    pressure: Arc<Pressure>,
    id: u64,
}

impl Drop for PressureWatch {
    fn drop(&mut self) {
        let mut watchers = self.pressure.watchers();
        watchers.woken.retain(|(id, _)| *id != self.id);
        if !watchers.woken.is_empty() {
            return;
        }
        // Stopped under the lock, so a watch started after it counts every
        // event from then on alone; joined with the lock let go, as the
        // thread takes it to wake.
        let Watch::Events(thread) = std::mem::take(&mut watchers.watch) else {
            unreachable!("a load's watch exists only while the store's events are watched");
        };
        thread.stop();
        drop(watchers);
        drop(thread);
    }
}

impl Ledger {
    /// Watch the OS's memory pressure events on the store from now on,
    /// where it gives any: each is counted once in the store, whichever
    /// loads watch, for its next reading ([`Self::read`]), and runs `wake`.
    /// The store has one watch while any load watches. None where the OS
    /// gives no unprivileged events.
    pub fn watch_pressure(&self, wake: impl Fn() + Send + Sync + 'static) -> Option<PressureWatch> {
        self.store.pressure.watch(Arc::new(wake))
    }

    /// The times the store's OS pressure events ended with no way to set
    /// them again: a store that read events once reads none from then on.
    pub fn pressure_lost(&self) -> u64 {
        self.store.pressure.lost.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// An event a replaced watch's thread delivers is never counted: the
    /// store's current watch is the second, its first's thread still
    /// delivering; the old watch's event counts nothing and wakes no load,
    /// and the current watch's counts once. The watches' threads are
    /// stand-ins, so the proof depends on no OS facility.
    #[test]
    fn an_event_of_a_replaced_watch_is_not_counted() {
        let pressure = Pressure::default();
        let woken = Arc::new(AtomicUsize::new(0));
        let counted = woken.clone();
        {
            let mut watchers = pressure.watchers();
            let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                counted.fetch_add(1, Ordering::SeqCst);
            });
            watchers.woken.push((0, wake));
            watchers.watch = Watch::Events(Thread { watch: 1, wake: Box::new(|| {}), stopped: Arc::new(AtomicBool::new(false)), thread: None });
        }
        pressure.event(0);
        assert_eq!((pressure.take(), woken.load(Ordering::SeqCst)), (0, 0), "the replaced watch's event counts nothing");
        pressure.event(1);
        assert_eq!((pressure.take(), woken.load(Ordering::SeqCst)), (1, 1), "the current watch's counts once");
    }
}
