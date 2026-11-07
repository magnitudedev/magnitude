//! Host pages a discrete device reads over its link, registered granule by
//! granule (`cuMemHostRegister`), and the device's quiet, the one time a
//! granule is unregistered.
//!
//! `cuMemHostUnregister` waits for the device's work and holds back its copy
//! queue meanwhile: beside a decode flight parked on a pull, a 16 MiB shrink
//! of a registered host tier took 2.1 s and the park gave up (two RTX 3090
//! hosts), where granules
//! created on the host node shrank in 2.6-3.6 ms beside it. So no shrink or
//! drop unregisters inline. It uncommits its granules at once (no device
//! work may reach them) and hands them to its device's [`Quiet`], which
//! unregisters and frees them, and releases their charge, once the device
//! is quiet: no stream of the device's has incomplete work, and no work is
//! issued meanwhile (every enqueue holds its issue lock, [`issue`]).
//! Where the device is quiet at the shrink, as between a single load's
//! flights, that is at once; otherwise a drainer thread waits for it. A
//! growth takes back granules still awaiting it before backing new ones.

use crate::driver::{Context, DriverError, Handle, NOT_READY};
use seismic_reservation::HostRange;
use std::collections::HashMap;
use std::ffi::c_uint;
use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) mod issue;

use issue::{Enqueue, IssueLock, Issuing};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`: pinned for
/// every context, mapped for the device.
const HOST_REGISTER: c_uint = 1 | 2;

/// How long the drainer waits between looks at the device's quiet.
const DRAIN_INTERVAL: Duration = Duration::from_millis(1);

type Hold = Box<dyn seismic_compiler::executable::MemoryHold>;

pub(crate) fn host_pages(failure: seismic_reservation::AllocationFailure) -> DriverError {
    DriverError {
        operation: "host reservation pages",
        code: 0,
        description: failure.to_string(),
    }
}

/// One device ordinal's quiet, shared by every context of the process on
/// it (they share its primary context, and its work).
pub(crate) struct Quiet {
    /// Held shared by every enqueue, exclusively by a drain.
    issue: IssueLock,
    /// Every stream of the device's, as raw handles.
    streams: Mutex<Vec<usize>>,
    /// Reservations with granules awaiting unregistration.
    pending: Mutex<Vec<Arc<Registered>>>,
    /// Whether a drainer thread is waiting for the quiet.
    draining: AtomicBool,
    /// The device's fault: a drain that failed (the context would not
    /// enter, a granule would not unregister). Its granules stay registered
    /// and charged, and every enqueue on the device is refused with it
    /// from then on (`Context::issuing`).
    fault: Mutex<Option<DriverError>>,
}

impl Quiet {
    /// Device `ordinal`'s quiet.
    pub fn of(ordinal: i32) -> Arc<Self> {
        static QUIETS: OnceLock<Mutex<HashMap<i32, Arc<Quiet>>>> = OnceLock::new();
        QUIETS
            .get_or_init(Default::default)
            .lock()
            .expect("device quiet registry lock is never poisoned")
            .entry(ordinal)
            .or_insert_with(|| {
                Arc::new(Self {
                    issue: IssueLock::new(),
                    streams: Mutex::new(Vec::new()),
                    pending: Mutex::new(Vec::new()),
                    draining: AtomicBool::new(false),
                    fault: Mutex::new(None),
                })
            })
            .clone()
    }

    pub fn add_stream(&self, stream: Handle) {
        self.streams().push(stream as usize);
    }

    pub fn remove_stream(&self, stream: Handle) {
        self.streams().retain(|&known| known != stream as usize);
    }

    fn streams(&self) -> std::sync::MutexGuard<'_, Vec<usize>> {
        self.streams.lock().expect("device stream registry lock is never poisoned")
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, Vec<Arc<Registered>>> {
        self.pending.lock().expect("pending unregistration lock is never poisoned")
    }

    /// The right to enqueue on the device through `enqueue`, refused with
    /// its fault once it has one.
    pub fn issuing<'a>(&'a self, enqueue: &'a Enqueue) -> Result<Issuing<'a>, DriverError> {
        self.fault().map_or(Ok(()), Err)?;
        Ok(self.issue.issuing(enqueue))
    }

    /// The device's fault, once a drain failed.
    fn fault(&self) -> Option<DriverError> {
        self.fault.lock().expect("device fault lock is never poisoned").clone()
    }

    /// Record `error` as the device's fault (the first one stands).
    fn faulted(&self, error: DriverError) {
        self.fault.lock().expect("device fault lock is never poisoned").get_or_insert(error);
    }

    /// `registered`'s uncommitted granules await the quiet: unregistered now
    /// where the device is quiet, else by a drainer once it is. Once they
    /// are queued the shrink or drop has happened: a drain that fails, now
    /// or later, is the device's fault, which its next enqueue reports
    /// ([`Self::issuing`]), never the shrink's.
    fn hand(self: &Arc<Self>, registered: Arc<Registered>) {
        {
            let mut pending = self.pending();
            if !pending.iter().any(|known| Arc::ptr_eq(known, &registered)) {
                pending.push(registered);
            }
        }
        match self.drain() {
            Drained::Empty | Drained::Faulted => return,
            Drained::Waiting if self.draining.swap(true, Ordering::AcqRel) => return,
            Drained::Waiting => {}
        }
        let quiet = self.clone();
        let started = std::thread::Builder::new().name("cuda-unregister".into()).spawn(move || loop {
            std::thread::sleep(DRAIN_INTERVAL);
            match quiet.drain() {
                Drained::Waiting => {}
                // Nothing is left to wait for: the device's fault keeps its
                // granules registered.
                Drained::Faulted => {
                    quiet.draining.store(false, Ordering::Release);
                    return;
                }
                Drained::Empty => {
                    quiet.draining.store(false, Ordering::Release);
                    // A hand after the drain found the drainer still running
                    // and left its granules to it.
                    if quiet.pending().is_empty() || quiet.draining.swap(true, Ordering::AcqRel) {
                        return;
                    }
                }
            }
        });
        if let Err(error) = started {
            self.draining.store(false, Ordering::Release);
            self.faulted(DriverError { operation: "unregistration drainer", code: 0, description: error.to_string() });
        }
    }

    /// Unregister every pending granule if the device is quiet: whether
    /// any awaits it still, or the device faulted (its granules then stay
    /// pending, registered and charged, and its fault is recorded for its
    /// next enqueue to report, [`Self::issuing`]).
    fn drain(&self) -> Drained {
        if self.fault().is_some() {
            return Drained::Faulted;
        }
        let Some(_issue) = self.issue.exclusive() else {
            return if self.pending().is_empty() { Drained::Empty } else { Drained::Waiting };
        };
        let mut pending = std::mem::take(&mut *self.pending());
        let Some(first) = pending.first() else { return Drained::Empty };
        let context = first.context.clone();
        let quiet = (|| {
            let _current = context.enter()?;
            for &stream in self.streams().iter() {
                match unsafe { (context.driver.stream_query)(stream as Handle) } {
                    NOT_READY => return Ok(false),
                    status => context.driver.check(status, "stream query")?,
                }
            }
            Ok(true)
        })();
        let released = quiet.and_then(|quiet| {
            if quiet {
                while let Some(registered) = pending.last() {
                    registered.release_uncommitted()?;
                    pending.pop();
                }
            }
            Ok(())
        });
        let mut waiting = self.pending();
        waiting.extend(pending);
        if let Err(error) = released {
            self.faulted(error);
            return Drained::Faulted;
        }
        if waiting.is_empty() { Drained::Empty } else { Drained::Waiting }
    }
}

/// What a drain leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drained {
    /// No granule awaits the quiet.
    Empty,
    /// Granules await the quiet.
    Waiting,
    /// The device faulted: its granules stay registered.
    Faulted,
}

/// A registered reservation's granules: the pages of one reserved host
/// address range ([`HostRange`], large pages where the host has them), the
/// leading `registered` of them backed and registered at their host
/// addresses, the leading `committed` of those committed, and the charge
/// of each registered granule, which falls when it is unregistered.
pub(crate) struct Registered {
    pub range: HostRange,
    context: Arc<Context>,
    state: Mutex<State>,
}

struct State {
    committed: usize,
    /// One charge per registered granule.
    holds: Vec<Hold>,
}

// The range is plain address space; the state is under its lock.
unsafe impl Send for Registered {}
unsafe impl Sync for Registered {}

impl Registered {
    pub fn new(context: &Arc<Context>, range: HostRange) -> Arc<Self> {
        Arc::new(Self {
            range,
            context: context.clone(),
            state: Mutex::new(State { committed: 0, holds: Vec::new() }),
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("registered granule lock is never poisoned")
    }

    /// Commit exactly the leading `wanted` granules. A growth takes back
    /// registered granules awaiting unregistration first, then backs and
    /// registers new ones, each charged by `charge` before it is backed (on
    /// a refusal the reservation is as it was); a shrink uncommits at once
    /// and hands the granules to the device's quiet. Precondition: no
    /// device work reaches granules a shrink uncommits. The granules it
    /// registered before: those from it on are newly backed host pages,
    /// which the host gives zeroed; those before may hold earlier bytes.
    pub fn commit(
        self: &Arc<Self>,
        wanted: usize,
        charge: impl Fn() -> Result<Hold, seismic_compiler::errors::ExecutionError>,
    ) -> Result<usize, seismic_compiler::errors::ExecutionError> {
        let failed = |error: DriverError| crate::executor::allocation_error((wanted * self.range.granule()) as u64, || seismic_compiler::errors::RefusalReading::Host)(error);
        let mut state = self.state();
        let first_new = state.holds.len();
        if wanted <= state.committed {
            let shrunk = wanted < state.committed;
            state.committed = wanted;
            drop(state);
            if shrunk {
                self.context.quiet.hand(self.clone());
            }
            return Ok(first_new);
        }
        let mut holds = Vec::new();
        for _ in first_new..wanted {
            holds.push(charge()?);
        }
        if wanted > first_new {
            let driver = &self.context.driver;
            let _current = self.context.enter().map_err(failed)?;
            self.range.back(first_new, wanted).map_err(|error| failed(host_pages(error)))?;
            let registered = seismic_reservation::time_backing(seismic_reservation::BackingPhase::Pin, || {
                for index in first_new..wanted {
                    let status =
                        unsafe { (driver.host_register)(self.range.at(index).cast(), self.range.granule(), HOST_REGISTER) };
                    if let Err(error) = driver.check(status, "host register") {
                        // Leave the backing as it was: unregister and free
                        // what this call backed.
                        for registered in first_new..index {
                            unsafe { (driver.host_unregister)(self.range.at(registered).cast()) };
                        }
                        return Err(error);
                    }
                }
                Ok(())
            });
            if let Err(error) = registered {
                self.range.unback(first_new, wanted);
                return Err(failed(error));
            }
            state.holds.extend(holds);
        }
        state.committed = wanted;
        Ok(first_new)
    }

    /// The registered granules now.
    #[cfg(all(test, target_os = "linux"))]
    pub fn registered(&self) -> usize {
        self.state().holds.len()
    }

    /// Unregister and free the granules past the committed ones, and
    /// release their charge: the device is quiet ([`Quiet::drain`]). A
    /// granule that would not unregister stays registered and charged.
    fn release_uncommitted(&self) -> Result<(), DriverError> {
        let mut state = self.state();
        let driver = &self.context.driver;
        let _current = self.context.enter()?;
        while state.holds.len() > state.committed {
            let index = state.holds.len() - 1;
            unsafe { driver.check((driver.host_unregister)(self.range.at(index).cast()), "host unregister") }?;
            self.range.unback(index, index + 1);
            state.holds.pop();
        }
        Ok(())
    }

    /// The reservation is dropped: every granule uncommitted and handed to
    /// the device's quiet, the range freed once the last is unregistered.
    pub fn drop_reservation(self: &Arc<Self>) {
        let shrunk = {
            let mut state = self.state();
            std::mem::replace(&mut state.committed, 0) > 0 || !state.holds.is_empty()
        };
        if shrunk {
            self.context.quiet.hand(self.clone());
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::RangeMemory;

    /// After a drain fault, a shrink of a registered tier still succeeds
    /// (its granules queued), every later enqueue is refused with the
    /// fault's reason, and the resizer's and the tier's teardown does not
    /// panic; once the fault is cleared (here only) a drain releases what
    /// waited.
    #[test]
    fn a_drain_fault_refuses_enqueues_not_shrinks() {
        let Some((_exclusive, opened)) = crate::test_device() else { return };
        let device = opened.service;
        let Some(granule) = crate::reservation_granule(device.ordinal(), RangeMemory::Registered).unwrap() else {
            return;
        };
        let (tier, resizer) =
            device.allocate_reserved(3 * granule, 4 * granule, granule, RangeMemory::Registered, std::sync::Arc::new(seismic_compiler::executable::Unbounded)).unwrap();
        let quiet = device.context().quiet.clone();
        let injected = DriverError { operation: "injected drain", code: 0, description: "an injected fault".into() };
        quiet.faulted(injected.clone());
        // The shrink and its release: Ok, no panic.
        let resizer = resizer.prepare(granule).map_err(|(_, error)| error).unwrap().install().release();
        assert_eq!(tier.reservation().unwrap().registered_granules(), Some(3), "the shrunk granules wait, registered");
        let refused = device.context().issuing().err().expect("an enqueue after the fault is refused");
        assert_eq!(refused, injected, "with the fault's reason");
        // Teardown, with the fault standing.
        drop(resizer);
        drop(tier);
        // Cleared, the next drain releases what waited.
        *quiet.fault.lock().unwrap() = None;
        assert_eq!(quiet.drain(), Drained::Empty);
        assert!(device.context().issuing().is_ok());
    }
}
