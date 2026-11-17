//! The machine's cores, and the share of them each process the service starts may run its
//! auxiliary threads on. An engine process's auxiliary pools (memory fault-in, native-target
//! workers, repack, formation) size themselves by its share (`seismic::threads::share`), which
//! the process is handed at its spawn ([`ThreadShare`]): the serving worker all of the cores, a
//! background job (the measurement job, a catalog optimization) all but [`SERVING_RESERVE`]. The
//! reserve does not reach a device's own pools (the CPU device's, the expert-compute pool): their
//! size is part of the device's identity, which a job measures and serving then opens, so in the
//! measurement job the reserve reaches only its auxiliary pools. The service sets its own
//! process's share to background work at startup, which sizes its catalog assessment, inventory
//! scans and the engine pools it opens.

use std::num::NonZeroUsize;
use std::sync::LazyLock;

/// The cores background work's auxiliary pools leave the serving worker: its dispatch thread and
/// the service's runtime, which receive a request and preempt the job.
pub const SERVING_RESERVE: usize = 2;

/// What a process the service starts is for, which decides its share of the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Duty {
    /// The worker serving a load: requests wait on it.
    Serving,
    /// Work nothing waits on, which yields to serving: the measurement job, a catalog
    /// optimization, the service's catalog assessment and inventory scans.
    Background,
}

/// The threads one process may run at once, handed to an engine worker at its spawn
/// ([`crate::worker_process::WorkerLauncher::command`]) and set as its share at its startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadShare(NonZeroUsize);

impl ThreadShare {
    pub fn get(self) -> NonZeroUsize {
        self.0
    }
}

/// The machine's cores as the service sees them: its performance cores under its quota of CPUs
/// (`seismic::Processors::cores`), read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MachineBudget {
    cores: NonZeroUsize,
}

static HOST: LazyLock<MachineBudget> = LazyLock::new(|| MachineBudget {
    cores: seismic::Processors::discover()
        .and_then(|processors| processors.cores())
        .unwrap_or_else(|error| panic!("invariant violated: the host's cores cannot be read: {error}")),
});

impl MachineBudget {
    /// This machine's.
    ///
    /// # Panics
    /// Where the OS reports no processor the service may run on.
    pub fn host() -> &'static Self {
        &HOST
    }

    /// The share of a process for `duty`: every core for serving; for background work every
    /// core but [`SERVING_RESERVE`], and one on a machine of no more.
    pub fn share(&self, duty: Duty) -> ThreadShare {
        ThreadShare(match duty {
            Duty::Serving => self.cores,
            Duty::Background => {
                NonZeroUsize::new(self.cores.get().saturating_sub(SERVING_RESERVE)).unwrap_or(NonZeroUsize::MIN)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No share exceeds the machine's cores, and a background job's leaves the serving worker its
    /// reserve wherever the machine has more cores than that.
    #[test]
    fn shares_never_exceed_the_cores_and_background_leaves_the_reserve() {
        for cores in 1..=256 {
            let budget = MachineBudget { cores: NonZeroUsize::new(cores).unwrap() };
            let (serving, background) = (budget.share(Duty::Serving).get().get(), budget.share(Duty::Background).get().get());
            assert_eq!(serving, cores);
            assert!(background <= cores);
            match cores > SERVING_RESERVE {
                true => assert_eq!(background, cores - SERVING_RESERVE),
                false => assert_eq!(background, 1),
            }
        }
        let host = MachineBudget::host();
        assert!(host.share(Duty::Serving).get() <= host.cores);
    }
}
