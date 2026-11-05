//! The host compute executor: the one owner of every host thread that does
//! a streamed load's expert work.
//!
//! Two kinds of work run on its workers:
//!
//! - **Site requests** (`site`): the device writes a layer's
//!   request into its slot, and the one worker noticing finds it by polling
//!   the sequence words and issues it to the others by one word they poll
//!   (no thread relays it, no driver call is on its path). Any worker
//!   claims any unit of a request's stages, its own shares first, so a
//!   request waits only on its own units in progress, never on a worker
//!   inside a job's piece.
//! - **Jobs** ([`Job`]): work the host submits, cut into pieces any worker
//!   runs, then ended once by the worker that ran its last piece. A file
//!   miss's landing into the tier is one (`landing::Converting`). A
//!   submitter a step waits on works its job's pieces beside the workers
//!   ([`HostCompute::run`]), so no wake stands before its first piece.
//!
//! The measurement job sizes the workers ([`Machine::sized`]): among the pools
//! of whole memory nodes (a thread on each performance core of a node with
//! memory, where the tier lies, within the process's affinity) under its
//! quota less the runtime's other busy threads, the one whose site request,
//! measured alone, is the fastest, of the fewest workers on a flat curve.
//! Every other thread of the process runs on the CPUs the workers' cores
//! leave, so no thread shares a core with a worker ([`ComputeThreads`]).
//! While a flight is issued
//! ([`Compute::flight`], held by each decode step of a load with a site)
//! every worker spins (the noticing one on the sequence words, the others
//! on the site's issue word): during decode the site is the
//! misses' latency, and no wake is on a request's path. With no flight
//! issued they park until a job is submitted or a flight is.
//!
//! Liveness is the protocol's (`site`): every taken request ends in an
//! answer. Teardown closes the site's issuing before it stops the workers,
//! and a worker exits only once every request issued is answered, so none
//! is left with units no worker runs. A stuck host is caught by the
//! exchange's watchdog (`exchange::Watchdog`), which reports the site's
//! state ([`Compute::report`]). A panic in a unit or piece ends serving
//! ([`Compute::panicked`]).

use super::site::{WorkerTurn, Site, SiteLayer};
use super::source::Records;
use magnitude_artifacts::Priority;
use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Host work cut into pieces: each piece runs once, on any worker, in any
/// order and concurrently with the others; then [`Self::end`] runs once,
/// on the worker that completed the last piece. A worker takes its next
/// piece from the most urgent job ([`Self::priority`], read as it takes
/// one), in submission order among the equally urgent: the read engine's
/// order, so a step's miss converts ahead of a hint's.
pub(in crate::experts) trait Job: Send + Sync {
    fn pieces(&self) -> usize;
    fn piece(&self, index: usize);
    fn end(&self);
    fn priority(&self) -> Priority;
}

/// A job's urgency, which only rises: shared by its submitter, who may
/// promote it while it waits (a hint's landing, once a request names its
/// expert), and the job. More urgent is less, by `Priority`'s own order.
pub(in crate::experts) struct Urgency(Mutex<Priority>);

impl Urgency {
    pub(in crate::experts) fn new(priority: Priority) -> Arc<Self> {
        Arc::new(Self(Mutex::new(priority)))
    }

    pub(in crate::experts) fn get(&self) -> Priority {
        *self.0.lock().expect("a job's urgency")
    }

    /// At least `priority` from now.
    pub(in crate::experts) fn promote(&self, priority: Priority) {
        let mut urgency = self.0.lock().expect("a job's urgency");
        *urgency = (*urgency).min(priority);
    }
}

/// Where a load's host threads run: the executor's workers, a CPU each,
/// node by node ([`Machine::sized`]), and every other thread of the process
/// on the CPUs the workers' cores leave (`runtime`), so none shares a core
/// with a worker. The resize thread, which backs the host tier, runs on
/// every CPU so the tier spreads over every memory node (`ResizeThread`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ComputeThreads {
    /// Each memory node's CPUs the workers run on.
    pub nodes: Vec<Vec<usize>>,
    /// Every other thread of the process while the executor serves (the
    /// load's serving threads, the device's own, and every thread they
    /// start, `HostCompute::serving`): every CPU the process may run on but
    /// the threads of the workers' cores, so their SMT siblings stay free.
    pub runtime: Arc<[usize]>,
}

impl ComputeThreads {
    pub(crate) fn count(&self) -> usize {
        self.nodes.iter().map(Vec::len).sum()
    }

    /// `workers` threads on the memory-attached cores the process may run
    /// on, in turn (a core twice where there are fewer), the runtime's
    /// threads unplaced: a test's pool, whatever the host.
    #[cfg(test)]
    pub(crate) fn placeable(workers: usize) -> Self {
        let nodes = {
            let _placement = crate::test_support::process_placement();
            seismic::Processors::discover().expect("the host's processors").memory_cores()
        };
        let cpus = nodes.iter().flat_map(|(_, cores)| cores).map(|threads| threads[0]).collect::<Vec<_>>();
        Self::unplaced(vec![(0..workers).map(|worker| cpus[worker % cpus.len()]).collect()])
    }

    /// Workers on `nodes`, nothing else placed: a test's pool.
    #[cfg(test)]
    pub(crate) fn unplaced(nodes: Vec<Vec<usize>>) -> Self {
        Self { nodes, runtime: Arc::new([]) }
    }
}

/// The CPUs a load's executor may take, from the process's real limits:
/// each memory node with memory, its performance cores the process may run
/// on (a core's second thread shares its first's units, so a worker on it
/// would halve both; a node of CPUs alone reads every byte across the
/// fabric), as many cores as the quota (`available_parallelism`, which
/// follows a cgroup's CPU quota and the process's affinity) leaves beside
/// the runtime's busy threads ([`busy_threads`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Machine {
    /// Each memory node's cores: the threads of each the process may run
    /// on, the first a worker's.
    nodes: Vec<Vec<Vec<usize>>>,
    /// Every CPU the process may run on.
    allowed: Vec<usize>,
    /// The most workers the quota leaves: at least one, where the runtime
    /// leaves none, since every file miss's landing runs only on a worker.
    limit: usize,
}

impl Machine {
    /// This machine's beside `device` and reads occupying `read_cores`
    /// (`magnitude_artifacts::read_cores`): the device's threads on the host
    /// (`Device::host_threads`: a discrete GPU's one submitting thread, a
    /// CPU device's pool) and the memory node it is attached to
    /// (`Device::host_node`).
    pub(crate) fn discover(device: &seismic::Device, read_cores: f64) -> Result<Self, String> {
        let node = device.host_node().map_err(|error| format!("the device's memory node: {error}"))?;
        Self::beside(device.host_threads(), read_cores, node)
    }

    /// This machine's beside a device of `device_threads` threads on the
    /// host, attached to node `device_node`, and reads of `read_cores`.
    fn beside(device_threads: usize, read_cores: f64, device_node: seismic::DeviceNode) -> Result<Self, String> {
        let quota = std::thread::available_parallelism().map_err(|error| format!("the process's quota of threads: {error}"))?;
        let processors = seismic::Processors::discover().map_err(|error| format!("the host's processors: {error}"))?;
        Ok(Self::of(&processors, quota.get(), busy_threads(device_threads, read_cores), device_node))
    }

    /// The machine of `processors` under `quota` threads, of which `busy`
    /// run the runtime's work, beside a device attached to `device_node`:
    /// its nodes in the OS's order but the device's last, so the workers
    /// fill it only once every other node is full and the device's own
    /// threads (its submitter, its driver's) run beside it; a device of no
    /// known node orders none.
    pub(crate) fn of(processors: &seismic::Processors, quota: usize, busy: usize, device_node: seismic::DeviceNode) -> Self {
        let mut nodes = processors.memory_cores();
        nodes.sort_by_key(|&(id, _)| device_node == seismic::DeviceNode::Node(id));
        Self {
            nodes: nodes.into_iter().map(|(_, cores)| cores).collect(),
            allowed: processors.allowed().iter().copied().collect(),
            limit: quota.saturating_sub(busy).max(1),
        }
    }

    /// The pools the executor is sized among, fewest workers first: those
    /// that fill whole memory nodes (in the OS's order), and, within the
    /// first node, a power of two of its cores (a single-node machine's
    /// curve has its minimum within the node), each within the quota's
    /// limit.
    fn candidates(&self) -> Vec<ComputeThreads> {
        let cores = self.nodes.iter().map(Vec::len).sum::<usize>();
        let first = self.nodes.first().map_or(0, Vec::len);
        let whole = self.nodes.iter().scan(0, |filled, node| {
            *filled += node.len();
            Some(*filled)
        });
        let halves = std::iter::successors(Some(1usize), |count| count.checked_mul(2)).take_while(|&count| count <= first);
        let mut sizes = whole.chain(halves).filter(|&count| count <= self.limit.min(cores)).collect::<Vec<_>>();
        // Empty only where the process may run on no such core.
        sizes.sort_unstable();
        sizes.dedup();
        sizes.into_iter().map(|count| self.placed(count)).collect()
    }

    /// The most workers the quota leaves, on whole nodes and a power of two
    /// of the first's ([`Self::candidates`]' last): a pool whose stage time
    /// is not measured (a load without a site, whose workers only land file
    /// misses; a qualification).
    pub(crate) fn widest(&self) -> Result<ComputeThreads, String> {
        self.candidates().pop().ok_or_else(|| "the process may run on no memory-attached performance core".to_owned())
    }

    /// The widest pool the quota leaves ([`Self::candidates`]) of at most
    /// `workers` workers: the pool the measurement job sized, which it
    /// sized with no reads beside it, within what a load's reads leave.
    pub(crate) fn at_most(&self, workers: usize) -> Result<ComputeThreads, String> {
        let within = self.candidates().into_iter().rfind(|threads| threads.count() <= workers);
        within.ok_or_else(|| format!("no pool of at most {workers} workers on a memory-attached performance core"))
    }

    /// The candidate ([`Self::candidates`]) whose site stage (`stage`, a
    /// request's time on a pool of its own, measured alone) is the least,
    /// of the fewest workers within the least's resolution ([`fewest_at_least`]).
    pub(crate) fn sized(
        &self,
        mut stage: impl FnMut(&ComputeThreads) -> Result<crate::assessment::Sample, String>,
    ) -> Result<ComputeThreads, String> {
        let candidates = self.candidates();
        if candidates.is_empty() {
            return Err("the process may run on no memory-attached performance core".into());
        }
        let samples = candidates.iter().map(&mut stage).collect::<Result<Vec<_>, _>>()?;
        Ok(candidates.into_iter().nth(fewest_at_least(&samples)).expect("a pick among the candidates"))
    }

    /// `count` workers on the first cores, node by node, each on its core's
    /// first thread; the runtime on every other CPU but those cores'.
    fn placed(&self, count: usize) -> ComputeThreads {
        let mut remaining = count;
        let mut taken = std::collections::BTreeSet::new();
        let nodes = self
            .nodes
            .iter()
            .map(|cores| {
                let take = cores.len().min(remaining);
                remaining -= take;
                cores[..take].iter().inspect(|threads| taken.extend(threads.iter().copied())).map(|threads| threads[0]).collect::<Vec<_>>()
            })
            .filter(|cpus| !cpus.is_empty())
            .collect();
        let runtime = self.allowed.iter().copied().filter(|cpu| !taken.contains(cpu)).collect::<Vec<_>>();
        ComputeThreads { nodes, runtime: runtime.into() }
    }
}

/// The index of the least stage time of `samples` (fewest workers first),
/// or of the fewest workers within its resolution: on a flat curve, more
/// workers buy nothing a measurement resolves and cost the quota.
fn fewest_at_least(samples: &[crate::assessment::Sample]) -> usize {
    let least = samples.iter().min_by(|a, b| a.seconds.total_cmp(&b.seconds)).expect("a sample of each candidate");
    samples.iter().position(|sample| sample.seconds <= least.seconds + least.resolution).expect("the least is within its own resolution")
}

/// The threads the runtime keeps busy beside the executor while a flight
/// runs, whose workers spin hot on the site and so get only what is left of
/// the quota. Each is stated by its owner:
/// - the device's own threads on the host (`device_threads`,
///   `Device::host_threads`): the submitting thread that drives the steps,
///   and a CPU device's worker pool;
/// - the watcher's (`watcher::THREADS`);
/// - the read engine's workers, `read_cores` as the probe of the volumes
///   read measured them (`magnitude_artifacts::read_cores`, the median of
///   its passes), to the nearest whole core: a share that sits near a
///   whole core (GB10 0.77–1.38) reserves the same core in every process,
///   where its ceiling flipped between one and two;
/// - the bank's resize thread, which backs the host tier beside early
///   decode (`bank::RESIZE_THREADS`).
///
/// Landings run on the executor's own workers, between site units.
fn busy_threads(device_threads: usize, read_cores: f64) -> usize {
    assert!(read_cores.is_finite() && read_cores >= 0., "read cores are a measured share of the host, {read_cores}");
    device_threads + super::watcher::THREADS + read_cores.round() as usize + super::bank::RESIZE_THREADS
}

/// A submitted job and how far its pieces are.
struct Running {
    job: Arc<dyn Job>,
    pieces: usize,
    /// The next piece to take (only grows: one taken past the last is
    /// none), and the pieces completed.
    next: AtomicUsize,
    done: AtomicUsize,
}

/// What the workers share.
struct Inner {
    site: Site,
    jobs: Mutex<VecDeque<Arc<Running>>>,
    /// The jobs queued, read before taking `jobs`' lock, so spinning
    /// workers poll no lock while nothing is queued.
    queued: AtomicUsize,
    /// The flights with site work issued ([`Flight`]).
    flights: AtomicUsize,
    /// What a parked worker waits on: a job submitted, a flight issued, the
    /// executor stopping. Changed under `rest`'s lock, so no wake is lost.
    rest: Mutex<()>,
    wake: Condvar,
    stop: AtomicBool,
    panicked: Mutex<Option<String>>,
}

/// The executor: its workers, which serve until it drops.
pub(crate) struct HostCompute {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
    /// The process's placement off the workers' cores ([`Self::serving`]),
    /// which places its threads back as it drops, after the workers.
    placed: Option<seismic::ProcessPlacement>,
}

/// A handle on the executor's work: what the watcher, the steps and the
/// read engine's ends submit to and observe.
#[derive(Clone)]
pub(crate) struct Compute {
    inner: Arc<Inner>,
}

/// A flight with site work issued: while one is held every worker polls
/// the site's requests.
pub(crate) struct Flight {
    inner: Arc<Inner>,
}

impl Drop for Flight {
    fn drop(&mut self) {
        self.inner.flights.fetch_sub(1, Ordering::AcqRel);
    }
}

impl HostCompute {
    /// Workers on `threads`, each placed on its CPU (`seismic::pin`),
    /// serving the site over `layers` (each a streamed layer with a CPU
    /// site) and every job submitted, from now until the executor drops.
    pub(crate) fn start(layers: Vec<SiteLayer>, threads: &ComputeThreads) -> Result<Self, String> {
        let count = threads.count();
        let inner = Arc::new(Inner {
            site: Site::new(layers, threads)?,
            jobs: Mutex::new(VecDeque::new()),
            queued: AtomicUsize::new(0),
            flights: AtomicUsize::new(0),
            rest: Mutex::new(()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            panicked: Mutex::new(None),
        });
        let mut compute = Self { inner, workers: Vec::with_capacity(count), placed: None };
        // Each worker's placement, before it serves: a failed one fails the
        // start (the executor drops, which stops the workers started).
        let (placed, placements) = std::sync::mpsc::channel();
        for (worker, cpu) in threads.nodes.iter().flatten().copied().enumerate() {
            let (shared, placed) = (compute.inner.clone(), placed.clone());
            let spawned = std::thread::Builder::new().name(format!("expert-compute-{cpu}")).spawn(move || {
                let placement = seismic::pin(&[cpu]);
                let serves = placement.is_ok();
                // A send fails only where the start failed already, which
                // then reads no placement.
                let _ = placed.send(placement);
                drop(placed);
                if serves {
                    work(&shared, worker)
                }
            });
            compute.workers.push(spawned.map_err(|error| format!("an expert compute worker: {error}"))?);
        }
        drop(placed);
        for placement in placements {
            placement.map_err(|error| format!("an expert compute worker's placement: {error}"))?;
        }
        Ok(compute)
    }

    /// A load's executor ([`Self::start`]), every other thread of the
    /// process first placed off the workers' cores (`ComputeThreads::runtime`),
    /// so none shares a core with a worker: the load's serving threads and
    /// the device's own (a submitting thread, a Vulkan keeper), and every
    /// thread they start, which inherits it. As it drops, every thread of
    /// the process but those `seismic::pin` placed is placed back on every
    /// CPU it could run on, the threads started while it served among them;
    /// a pinned thread (a worker, the resize thread) keeps its CPUs. The
    /// placement is the process's one (`seismic::ProcessPlacement`): a
    /// second serving executor in one process is an invariant violation. An
    /// executor that measures (`site::measure`) places nothing.
    pub(crate) fn serving(layers: Vec<SiteLayer>, threads: &ComputeThreads) -> Result<Self, String> {
        let placed = match &*threads.runtime {
            [] => None,
            runtime => Some(
                seismic::place_process(runtime).map_err(|error| format!("the process's threads off the workers' cores: {error}"))?,
            ),
        };
        let mut compute = Self::start(layers, threads)?;
        compute.placed = placed;
        Ok(compute)
    }

    pub(crate) fn handle(&self) -> Compute {
        Compute { inner: self.inner.clone() }
    }

}

impl Drop for HostCompute {
    /// Close the site's issuing, then stop the workers (each serves every
    /// request issued before it exits, [`work`]), then drop the jobs never
    /// run: their ends hold what submitted them, which holds this executor's
    /// handle.
    fn drop(&mut self) {
        self.inner.site.close();
        {
            let _rest = self.inner.rest.lock().expect("compute rest");
            self.inner.stop.store(true, Ordering::Release);
            self.inner.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        self.inner.jobs.lock().expect("compute jobs").clear();
        self.inner.queued.store(0, Ordering::Release);
        // The placement drops after this, the workers joined: the process's
        // threads go back on every CPU.
    }
}

impl Compute {
    /// Issue a flight with site work: the workers poll the site's requests
    /// and compute each taken, until the returned flight drops. None where
    /// the executor serves no site layer, as nothing would be asked of it.
    pub(crate) fn flight(&self) -> Option<Flight> {
        if self.inner.site.is_empty() {
            return None;
        }
        let _rest = self.inner.rest.lock().expect("compute rest");
        self.inner.flights.fetch_add(1, Ordering::AcqRel);
        self.inner.wake.notify_all();
        Some(Flight { inner: self.inner.clone() })
    }

    /// Run `job`'s pieces on the workers, then its end. A job submitted once
    /// the executor stopped never runs.
    pub(in crate::experts) fn submit(&self, job: Arc<dyn Job>) {
        self.enqueue(job);
    }

    /// As [`Self::submit`], the calling thread working `job`'s pieces beside
    /// the workers until each is taken: a step waits on it, and no worker's
    /// wake stands before its first piece.
    pub(in crate::experts) fn run(&self, job: Arc<dyn Job>) {
        if let Some(running) = self.enqueue(job) {
            while guarded(&self.inner, || own_piece(&self.inner, &running)) && running.next.load(Ordering::Acquire) < running.pieces {}
        }
    }

    /// `job` queued for the workers, none where it has no piece or the
    /// executor stopped.
    fn enqueue(&self, job: Arc<dyn Job>) -> Option<Arc<Running>> {
        let pieces = job.pieces();
        let running = Arc::new(Running { job, pieces, next: AtomicUsize::new(0), done: AtomicUsize::new(0) });
        if pieces == 0 {
            running.job.end();
            return None;
        }
        let _rest = self.inner.rest.lock().expect("compute rest");
        if self.inner.stop.load(Ordering::Acquire) {
            return None;
        }
        self.inner.jobs.lock().expect("compute jobs").push_back(running.clone());
        self.inner.queued.fetch_add(1, Ordering::AcqRel);
        self.inner.wake.notify_all();
        Some(running)
    }

    /// Why a worker's unit or piece panicked, where one did: serving ends.
    pub(crate) fn panicked(&self) -> Option<String> {
        self.inner.panicked.lock().expect("compute panic").clone()
    }

    /// Each group's tier records `views` address from now (a shrink's,
    /// between flights, or a growth's joined slot, beside the requests
    /// serving, which a growth leaves every record they name; in record
    /// order), for every site layer of the group.
    pub(crate) fn address(&self, views: &[Vec<Records>]) {
        self.inner.site.address(views);
    }

    /// What gates the site's requests and observes its every answer from now
    /// on (`Site::gated`).
    pub(in crate::experts) fn gated(&self, gate: Arc<dyn super::site::SiteGate>) {
        self.inner.site.gated(gate);
    }

    /// What the site failed of step `step`, taken as the step
    /// completes; none where it failed none of its requests.
    pub(crate) fn take_site_failure(&self, step: u64) -> Option<super::site::SiteFailure> {
        self.inner.site.take_failure(step)
    }

    /// What streamed layer `layer`'s CPU site answered of step `step`,
    /// while its slot holds it (`SiteLayer::answered`): none of a layer
    /// without a site or of a step it served nothing of.
    #[cfg(test)]
    pub(crate) fn answered(&self, layer: usize, step: u64) -> Option<super::site::Answered> {
        self.inner.site.layers().iter().find(|site| site.layer() == layer)?.answered(step)
    }

    /// The site's layers.
    pub(crate) fn layers(&self) -> &[SiteLayer] {
        self.inner.site.layers()
    }

    /// The experts streamed layer `layer`'s CPU site served for step `step`,
    /// once it has completed (`Site::sited`): none of a layer without one.
    pub(crate) fn sited(&self, layer: usize, step: u64) -> (u64, std::time::Duration) {
        let site = &self.inner.site;
        site.layers().iter().position(|site| site.layer() == layer).map_or((0, std::time::Duration::ZERO), |index| site.sited(index, step))
    }

    /// The site's state as the exchange's watchdog reports it: each request
    /// issued and not yet answered with its stages' progress, the flights,
    /// the jobs queued and a panic, where one was recorded.
    pub(in crate::experts) fn report(&self) -> String {
        format!(
            "issued (site layer, step, each stage's units done of its units) {:?}; flights {}; jobs queued {}; panicked: {:?}",
            self.inner.site.progress(),
            self.inner.flights.load(Ordering::Acquire),
            self.inner.queued.load(Ordering::Acquire),
            self.inner.panicked.lock().expect("compute panic"),
        )
    }

    /// What the site's answered requests took since the load
    /// (`Site::served`).
    pub(in crate::experts) fn site_served(&self) -> super::site::SiteServed {
        self.inner.site.served()
    }

    /// Whether the site has a request issued and not yet answered.
    #[cfg(test)]
    pub(crate) fn site_busy(&self) -> bool {
        self.inner.site.busy()
    }
}

/// Worker `worker`: a unit of an issued site request first (one issued in a
/// flight is served even should the flight end before), waiting only on a
/// request's units in progress elsewhere; else, in a flight and as the one
/// worker noticing requests (`Site::notice`), a newly published request
/// taken; else a piece of a job. While a flight is issued the noticing worker spins on
/// the sequence words and the others on the site's one issue word, so a
/// request is taken and computed with no wake on its path; with none it
/// parks until a job is submitted or a flight is issued, noticing no more,
/// as it does before it runs a piece. It exits once stopping with no
/// request issued: the stop is read before the requests, and is set only
/// once the site's issuing closed (`HostCompute::drop`), so no request is
/// issued after it found none but its own, which it serves first.
fn work(inner: &Inner, worker: usize) {
    let mut seen = 0;
    loop {
        let stopping = inner.stop.load(Ordering::Acquire);
        let served = inner.site.serve(worker, &mut seen);
        match served {
            WorkerTurn::Ran => continue,
            WorkerTurn::Idle if stopping => {
                inner.site.release(worker);
                return;
            }
            WorkerTurn::Idle | WorkerTurn::Waiting => {}
        }
        let flying = inner.flights.load(Ordering::Acquire) > 0;
        if flying && inner.site.notice(worker) && guarded(inner, || inner.site.take()) {
            continue;
        }
        if served == WorkerTurn::Waiting {
            std::hint::spin_loop();
            continue;
        }
        if inner.queued.load(Ordering::Acquire) > 0 {
            inner.site.release(worker);
        }
        if guarded(inner, || piece(inner)) {
            continue;
        }
        if flying {
            std::hint::spin_loop();
            continue;
        }
        inner.site.release(worker);
        let rest = inner.rest.lock().expect("compute rest");
        let idle = |_: &mut ()| {
            !inner.stop.load(Ordering::Acquire)
                && inner.queued.load(Ordering::Acquire) == 0
                && inner.flights.load(Ordering::Acquire) == 0
                && !inner.site.busy()
        };
        drop(inner.wake.wait_while(rest, idle).expect("compute rest"));
    }
}

/// Run `serve`, recording a panic of it (serving ends) instead of ending
/// the worker. Whether it served anything.
fn guarded(inner: &Inner, serve: impl FnOnce() -> bool) -> bool {
    match std::panic::catch_unwind(AssertUnwindSafe(serve)) {
        Ok(served) => served,
        Err(payload) => {
            let message = super::watcher::panic_message(payload.as_ref());
            inner.panicked.lock().expect("compute panic").get_or_insert(format!("expert compute panicked: {message}"));
            true
        }
    }
}

/// Run one piece of the most urgent job with a piece left (the first
/// submitted among the equally urgent, [`Job::priority`]), its end after
/// its last. True when it ran one.
fn piece(inner: &Inner) -> bool {
    if inner.queued.load(Ordering::Acquire) == 0 {
        return false;
    }
    let (running, index) = {
        let mut jobs = inner.jobs.lock().expect("compute jobs");
        // Pieces are taken only under the lock: a job whose every piece is
        // taken leaves the queue.
        let queued = jobs.len();
        jobs.retain(|job| job.next.load(Ordering::Acquire) < job.pieces);
        inner.queued.fetch_sub(queued - jobs.len(), Ordering::AcqRel);
        let Some(urgent) = jobs.iter().min_by_key(|job| job.job.priority()).cloned() else { return false };
        let index = urgent.next.fetch_add(1, Ordering::AcqRel);
        (urgent, index)
    };
    work_piece(&running, index);
    true
}

/// Run one piece of `running`, its end after its last. True when one was
/// left.
fn own_piece(inner: &Inner, running: &Running) -> bool {
    let index = {
        // Pieces are taken only under the lock ([`piece`]).
        let _jobs = inner.jobs.lock().expect("compute jobs");
        match running.next.load(Ordering::Acquire) < running.pieces {
            true => running.next.fetch_add(1, Ordering::AcqRel),
            false => return false,
        }
    };
    work_piece(running, index);
    true
}

fn work_piece(running: &Running, index: usize) {
    running.job.piece(index);
    if running.done.fetch_add(1, Ordering::AcqRel) + 1 == running.pieces {
        running.job.end();
    }
}

#[cfg(test)]
mod tests;