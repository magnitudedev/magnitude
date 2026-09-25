use crate::error::DeviceError;
use seismic::NativeGraphCompletion;
use std::{
    sync::{mpsc, Arc, Condvar, Mutex},
    thread,
};

/// Exactly one wake for an outstanding executor completion, independent of the
/// bounded control queue. Completion owners may move it across threads.
pub struct CompletionWake(Option<Box<dyn FnOnce() + Send>>);

impl CompletionWake {
    pub fn new(wake: impl FnOnce() + Send + 'static) -> Self {
        Self(Some(Box::new(wake)))
    }

    pub fn complete(mut self) {
        self.0.take().expect("one completion wake")();
    }
}

/// Completion covers numerical execution, deferred transfers, and selection.
/// A result may be read only after completion, including failed submissions.
pub trait Completion {
    fn is_complete(&self) -> bool;
    fn result(&mut self) -> Result<(), DeviceError>;

    /// Register the reserved host wake after all completion obligations.
    /// Invoke it immediately if already complete. This must not poll or block.
    fn notify(&mut self, wake: CompletionWake);
}

/// Synchronous phase-one completion returned after all native calls finish.
pub struct Completed {
    result: Result<(), DeviceError>,
}

impl Completed {
    pub fn success() -> Self {
        Self { result: Ok(()) }
    }

    pub fn failure(error: DeviceError) -> Self {
        Self { result: Err(error) }
    }
}

impl Completion for Completed {
    fn is_complete(&self) -> bool {
        true
    }

    fn result(&mut self) -> Result<(), DeviceError> {
        self.result.clone()
    }

    fn notify(&mut self, wake: CompletionWake) {
        wake.complete();
    }
}

type WaitJob = Box<dyn FnOnce() + Send>;

/// One host thread that blocks on submitted device work so the owning
/// thread never does. It observes completions in submission order and
/// then delivers each reserved wake; it exits once every handle is dropped.
#[derive(Clone)]
pub(crate) struct CompletionWaiter {
    jobs: mpsc::Sender<WaitJob>,
}

impl CompletionWaiter {
    pub(crate) fn spawn() -> Result<Self, String> {
        let (jobs, receiver) = mpsc::channel::<WaitJob>();
        thread::Builder::new()
            .name("magnitude-device-wait".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })
            .map_err(|error| format!("device completion waiter did not start: {error}"))?;
        Ok(Self { jobs })
    }

    /// Completion of `runs`, submitted in this order to one device queue.
    pub(crate) fn completion(&self, runs: Vec<NativeGraphCompletion>) -> DeviceCompletion {
        DeviceCompletion {
            waiter: self.clone(),
            state: DeviceCompletionState::Held(runs),
        }
    }
}

/// Outcome shared between a waiting job and the completion's owner.
struct SharedOutcome {
    result: Mutex<Option<Result<(), DeviceError>>>,
    finished: Condvar,
}

enum DeviceCompletionState {
    /// No wake is registered; the runs are still owned here.
    Held(Vec<NativeGraphCompletion>),
    /// The waiter thread owns the runs and publishes their outcome.
    Waiting(Arc<SharedOutcome>),
    Observed(Result<(), DeviceError>),
}

/// Completion of native graph runs that were submitted without waiting.
/// Registering a wake hands the runs to the waiter thread; reading the result
/// before completion blocks on the device.
pub(crate) struct DeviceCompletion {
    waiter: CompletionWaiter,
    state: DeviceCompletionState,
}

fn wait_runs(runs: Vec<NativeGraphCompletion>) -> Result<(), DeviceError> {
    // Every run is observed, even after a failure, so no run remains in
    // flight once the outcome is published; the first failure is reported.
    runs.into_iter().fold(Ok(()), |outcome, run| {
        let observed = run
            .wait()
            .map_err(|error| DeviceError::Execution(error.to_string()));
        outcome.and(observed)
    })
}

impl Completion for DeviceCompletion {
    fn is_complete(&self) -> bool {
        match &self.state {
            DeviceCompletionState::Held(runs) => {
                runs.iter().all(NativeGraphCompletion::is_complete)
            }
            DeviceCompletionState::Waiting(shared) => shared
                .result
                .lock()
                .expect("device completion outcome lock")
                .is_some(),
            DeviceCompletionState::Observed(_) => true,
        }
    }

    fn result(&mut self) -> Result<(), DeviceError> {
        let observed =
            match std::mem::replace(&mut self.state, DeviceCompletionState::Observed(Ok(()))) {
                DeviceCompletionState::Held(runs) => wait_runs(runs),
                DeviceCompletionState::Waiting(shared) => {
                    let mut result = shared
                        .result
                        .lock()
                        .expect("device completion outcome lock");
                    while result.is_none() {
                        result = shared
                            .finished
                            .wait(result)
                            .expect("device completion outcome lock");
                    }
                    result
                        .clone()
                        .expect("device completion outcome was published")
                }
                DeviceCompletionState::Observed(result) => result,
            };
        self.state = DeviceCompletionState::Observed(observed.clone());
        observed
    }

    fn notify(&mut self, wake: CompletionWake) {
        let runs = match std::mem::replace(&mut self.state, DeviceCompletionState::Observed(Ok(())))
        {
            DeviceCompletionState::Held(runs) => runs,
            DeviceCompletionState::Observed(result) => {
                self.state = DeviceCompletionState::Observed(result);
                wake.complete();
                return;
            }
            DeviceCompletionState::Waiting(_) => {
                panic!("a device completion accepts exactly one wake")
            }
        };
        if runs.iter().all(NativeGraphCompletion::is_complete) {
            self.state = DeviceCompletionState::Observed(wait_runs(runs));
            wake.complete();
            return;
        }
        let shared = Arc::new(SharedOutcome {
            result: Mutex::new(None),
            finished: Condvar::new(),
        });
        let published = shared.clone();
        let job: WaitJob = Box::new(move || {
            let outcome = wait_runs(runs);
            *published
                .result
                .lock()
                .expect("device completion outcome lock") = Some(outcome);
            published.finished.notify_all();
            wake.complete();
        });
        self.state = match self.waiter.jobs.send(job) {
            Ok(()) => DeviceCompletionState::Waiting(shared),
            Err(mpsc::SendError(job)) => {
                // The waiter thread only stops when a job panicked; observe
                // the work here rather than leave the wake undelivered.
                job();
                DeviceCompletionState::Waiting(shared)
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    #[test]
    fn completed_wakes_immediately_and_preserves_result() {
        let fired = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&fired);
        let mut completion =
            Completed::failure(DeviceError::Execution("native stage failed".into()));
        completion.notify(CompletionWake::new(move || {
            observed.store(true, Ordering::SeqCst);
        }));
        assert!(fired.load(Ordering::SeqCst));
        assert!(completion.is_complete());
        assert_eq!(
            completion.result(),
            Err(DeviceError::Execution("native stage failed".into()))
        );
    }
}
