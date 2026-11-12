//! Host pages a discrete device reads over its link, registered granule by
//! granule (`cuMemHostRegister`), and their unregistration.
//!
//! `cuMemHostUnregister` waits for the device's work and holds back its copy
//! queue meanwhile: beside a decode flight parked on a pull, a 16 MiB shrink
//! of a registered host tier took 2.1 s and the park gave up (two RTX 3090
//! hosts), where granules created on the host node shrank in 2.6-3.6 ms
//! beside it. So only a release unregisters (`Backing::unback`: a shrink's,
//! or the reservation's end), inline on its own thread, and the executor
//! releases a host tier only between its flights (it shrinks only with the
//! idle pipeline's state bindings and asserts no step runs; a load's drop
//! follows its last flight). The wait delays the load's next flight, timed
//! as [`BackingPhase::Unpin`]: with no device work running, 14 MiB in 2 MiB
//! granules unregistered in 0.99 ms and 1 GiB in 48.3 ms, 22 GB/s (RTX 4070,
//! driver 595.99.02). A shrink releases what a deficit owes, so returning
//! 1 GiB to another program costs about one decode step. Another load's work on the device in the
//! same process (tests run many) is waited for meanwhile, and a flight of
//! it parked on a pull stalls until its park gives up; no data is at risk.
//!
//! No device work uses a granule a release unregisters: the runtime records
//! each allocation's newest use per device queue as that submission's end
//! event and waits for those before a resize's release and before the
//! storage drops. No lock of the backend's is held across the driver call,
//! and no enqueue takes one ([`crate::enqueue`]).
//!
//! What a growth backed but did not commit (a failed registration, zeroing
//! or change, an abandoned commitment: `Backing::retract`) stays registered,
//! charged and uncommitted, beside its flights, for the next growth to take
//! back or the next release to unregister.

use crate::driver::{Context, DriverError};
use seismic_reservation::{time_backing, BackingPhase, HostRange};
use std::ffi::c_uint;
use std::sync::{Arc, Mutex, MutexGuard};

/// `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`: pinned for
/// every context, mapped for the device.
const HOST_REGISTER: c_uint = 1 | 2;

type Hold = Box<dyn seismic_compiler::executable::MemoryHold>;

pub(crate) fn host_pages(failure: seismic_reservation::AllocationFailure) -> DriverError {
    DriverError {
        operation: "host reservation pages",
        code: 0,
        description: failure.to_string(),
    }
}

/// A registered reservation's granules: the pages of one reserved host
/// address range ([`HostRange`], large pages where the host has them), the
/// leading `holds.len()` of them backed and registered at their host
/// addresses, and the leading `committed` of those committed.
pub(crate) struct Registered {
    pub range: HostRange,
    context: Arc<Context>,
    state: Mutex<State>,
    /// Where a test parks the next unregistration, as it enters the driver
    /// call.
    #[cfg(all(test, target_os = "linux"))]
    park: Mutex<Option<Park>>,
}

struct State {
    committed: usize,
    /// The charge of each registered granule, committed or not, to the
    /// reservation's account: the one store of its charge, which falls only
    /// when the granule is unregistered.
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
            #[cfg(all(test, target_os = "linux"))]
            park: Mutex::new(None),
        })
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("registered granule lock is never poisoned")
    }

    /// Commit exactly the leading `wanted` granules. A growth takes back
    /// registered granules first, then backs and registers new ones, each
    /// charged by `charge` before it is backed; on a refusal the commitment
    /// is as it was, and what it registered before the refusal stays
    /// registered and charged. A release uncommits and unregisters every
    /// granule past `wanted` at once (precondition: no device work uses
    /// them, and none of the holder's runs). The granules it registered before: those from
    /// it on are newly backed host pages, which the host gives zeroed;
    /// those before may hold earlier bytes.
    pub fn commit(
        &self,
        wanted: usize,
        charge: impl Fn() -> Result<Hold, seismic_compiler::errors::ExecutionError>,
    ) -> Result<usize, seismic_compiler::errors::ExecutionError> {
        let failed = |error: DriverError| crate::executor::allocation_error((wanted * self.range.granule()) as u64, || seismic_compiler::errors::RefusalReading::Host)(error);
        let mut state = self.state();
        let first_new = state.holds.len();
        if wanted <= state.committed {
            state.committed = wanted;
            self.unregister_uncommitted(&mut state);
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
            let registered = time_backing(BackingPhase::Pin, || {
                for index in first_new..wanted {
                    let status =
                        unsafe { (driver.host_register)(self.range.at(index).cast(), self.range.granule(), HOST_REGISTER) };
                    if let Err(error) = driver.check(status, "host register") {
                        return Err((index, error));
                    }
                }
                Ok(())
            });
            if let Err((index, error)) = registered {
                // What registered stays registered and charged (a growth
                // never unregisters); the rest is unbacked.
                self.range.unback(index, wanted);
                holds.truncate(index - first_new);
                state.holds.extend(holds);
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

    /// Uncommit the granules past `wanted`, keeping them registered and
    /// charged ([`seismic_reservation::Backing::retract`]).
    pub fn uncommit(&self, wanted: usize) {
        let mut state = self.state();
        state.committed = state.committed.min(wanted);
    }

    /// The reservation is dropped: every granule uncommitted and
    /// unregistered, after its holder's last work.
    pub fn drop_reservation(&self) {
        let mut state = self.state();
        state.committed = 0;
        self.unregister_uncommitted(&mut state);
    }

    /// Unregister and free the registered granules past the committed
    /// ones, the last first, and release each one's charge. A granule no
    /// work uses that will not unregister is a driver fault the reservation
    /// cannot outlive: its range would free pages the driver still pins.
    fn unregister_uncommitted(&self, state: &mut State) {
        if state.holds.len() == state.committed {
            return;
        }
        time_backing(BackingPhase::Unpin, || {
            let driver = &self.context.driver;
            let _current = self.context.enter().unwrap_or_else(|error| panic!("a registered host granule's context enters: {error}"));
            while state.holds.len() > state.committed {
                let index = state.holds.len() - 1;
                #[cfg(all(test, target_os = "linux"))]
                if let Some(park) = self.park.lock().expect("unregistration park lock is never poisoned").take() {
                    park.entered.send(()).expect("the parking test waits for the unregistration");
                    park.release.recv().expect("the parking test releases the unregistration");
                }
                unsafe { driver.check((driver.host_unregister)(self.range.at(index).cast()), "host unregister") }
                    .unwrap_or_else(|error| panic!("a registered host granule no work uses unregisters: {error}"));
                self.range.unback(index, index + 1);
                state.holds.pop();
            }
        });
    }

    /// Park the next unregistration as it enters the driver call, until the
    /// returned [`Parked`] releases it.
    #[cfg(all(test, target_os = "linux"))]
    pub fn park(&self) -> Parked {
        let (entered, on_entry) = std::sync::mpsc::channel();
        let (release, on_release) = std::sync::mpsc::channel();
        *self.park.lock().expect("unregistration park lock is never poisoned") = Some(Park { entered, release: on_release });
        Parked { entered: on_entry, release }
    }
}

/// The unregistration's side of a test's park ([`Registered::park`]).
#[cfg(all(test, target_os = "linux"))]
struct Park {
    entered: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

/// A test's side of a parked unregistration ([`Registered::park`]).
#[cfg(all(test, target_os = "linux"))]
pub(crate) struct Parked {
    entered: std::sync::mpsc::Receiver<()>,
    release: std::sync::mpsc::Sender<()>,
}

#[cfg(all(test, target_os = "linux"))]
impl Parked {
    /// Whether the unregistration entered the driver call within `timeout`.
    pub fn entered(&self, timeout: std::time::Duration) -> bool {
        self.entered.recv_timeout(timeout).is_ok()
    }

    /// Let the unregistration run.
    pub fn release(self) {
        self.release.send(()).expect("the parked unregistration waits for its release");
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use crate::driver::Stream;
    use crate::{Device, RangeMemory};
    use std::sync::Arc;
    use std::time::Duration;

    /// Fill a fresh device allocation of `device`'s on `stream` and wait
    /// for it, on a thread of its own: whether it completed within `within`.
    /// An allocation work may still reach is never freed.
    fn completes(device: &Device, stream: Arc<Stream>, within: Duration) -> bool {
        const BYTES: usize = 1 << 20;
        let buffer = device.allocate_within(BYTES as u64, 16, None).unwrap();
        let (context, pointer) = (device.context().clone(), buffer.pointer());
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            {
                let _current = context.enter().unwrap();
                let status = unsafe { context.enqueue().memset_d8_async(pointer, 1, BYTES, stream.raw()) };
                context.driver.check(status, "fill").unwrap();
            }
            stream.synchronize().unwrap();
            let _ = done.send(());
        });
        let completed = finished.recv_timeout(within).is_ok();
        if !completed {
            std::mem::forget(buffer);
        }
        completed
    }

    /// A shrink with no device work running unregisters its granules
    /// inline and returns with them unregistered and their pages unbacked:
    /// the recorded case, a 16 MiB registered host tier (2.1 s beside a
    /// parked flight), and a 1 GiB one, whose times give the unregistration
    /// rate a whole host tier's shrink is priced by.
    #[test]
    fn a_shrink_unregisters_inline() {
        let Some((_exclusive, opened)) = crate::test_device() else { return };
        let device = opened.service;
        let Some(granule) = crate::reservation_granule(device.ordinal(), RangeMemory::Registered).unwrap() else {
            return;
        };
        for bytes in [16u64 << 20, 1 << 30] {
            let granules = bytes.div_ceil(granule).max(2);
            let (tier, resizer) = device
                .allocate_reserved(granules * granule, granules * granule, granule, RangeMemory::Registered, Arc::new(seismic_compiler::executable::Unbounded))
                .unwrap();
            assert_eq!(tier.reservation().unwrap().registered_granules(), Some(granules as usize));
            let shrinking = std::time::Instant::now();
            let resizer = resizer.prepare(granule).map_err(|(_, error)| error).unwrap().install().release();
            let took = shrinking.elapsed();
            let released = (granules - 1) * granule;
            eprintln!(
                "unregistered {released} B in {} granules of {granule} B in {took:?}: {:.2} GB/s",
                granules - 1,
                released as f64 / took.as_secs_f64() / 1e9
            );
            assert_eq!(tier.reservation().unwrap().registered_granules(), Some(1), "the shrink unregistered its granules");
            drop(resizer);
            drop(tier);
        }
    }

    /// A shrink beside device work on another stream of the device returns
    /// with its granules unregistered: the release unregisters inline, its
    /// driver call waiting for that work (a stream held on a host word
    /// until 200 ms on), never deferring the unregistration past it.
    #[test]
    fn a_shrink_beside_device_work_returns_with_its_granules_unregistered() {
        let Some((_exclusive, opened)) = crate::test_device() else { return };
        let device = opened.service;
        let Some(granule) = crate::reservation_granule(device.ordinal(), RangeMemory::Registered).unwrap() else {
            return;
        };
        let context = device.context().clone();
        let account: Arc<dyn seismic_compiler::executable::MemoryAccount> = Arc::new(seismic_compiler::executable::Unbounded);
        let word = Arc::new(crate::driver::HostMapped::new(&context, &account, 4).unwrap());
        let held = Stream::new(&context).unwrap();
        {
            let _current = context.enter().unwrap();
            // CU_STREAM_WAIT_VALUE_GEQ
            let Some(status) = (unsafe { context.enqueue().wait_value_32(held.raw(), word.device, 1, 0) }) else {
                return;
            };
            context.driver.check(status, "stream wait on a host word").unwrap();
        }
        let (tier, resizer) =
            device.allocate_reserved(3 * granule, 4 * granule, granule, RangeMemory::Registered, account.clone()).unwrap();
        let releasing = word.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            releasing.write_at(0, &1u32.to_le_bytes());
        });
        let resizer = resizer.prepare(granule).map_err(|(_, error)| error).unwrap().install().release();
        assert_eq!(tier.reservation().unwrap().registered_granules(), Some(1), "the shrink returns with its granules unregistered");
        releaser.join().unwrap();
        held.synchronize().unwrap();
        drop(resizer);
        drop(tier);
    }

    /// While a shrink's unregistration is parked just before the driver
    /// call (the test's hook), work on another stream of the device, and on
    /// a second context of it, is issued and completes: no enqueue waits
    /// for an unregistration, as every enqueue waited behind the device's
    /// issue lock before. Released, the shrink ends with its granules
    /// unregistered.
    #[test]
    fn work_completes_while_a_shrink_unregisters() {
        let Some((_exclusive, opened)) = crate::test_device() else { return };
        let device = opened.service;
        let Some(granule) = crate::reservation_granule(device.ordinal(), RangeMemory::Registered).unwrap() else {
            return;
        };
        let (tier, resizer) =
            device.allocate_reserved(3 * granule, 4 * granule, granule, RangeMemory::Registered, Arc::new(seismic_compiler::executable::Unbounded)).unwrap();
        let parked = tier.reservation().unwrap().registered().unwrap().park();
        let shrinking = std::thread::spawn(move || resizer.prepare(granule).map_err(|(_, error)| error).unwrap().install().release());
        let prompt = Duration::from_secs(10);
        assert!(parked.entered(prompt), "the shrink unregisters");
        let other = Stream::new(device.context()).unwrap();
        assert!(completes(&device, other, prompt), "work on another stream of the device completes");
        let second = Device::open(device.ordinal(), crate::test_accounts()).unwrap();
        let stream = second.stream().clone();
        assert!(completes(&second, stream, prompt), "work on a second context of the device completes");
        parked.release();
        let resizer = shrinking.join().unwrap();
        assert_eq!(tier.reservation().unwrap().registered_granules(), Some(1), "the shrunk granules end unregistered");
        drop(resizer);
        drop(tier);
    }
}
