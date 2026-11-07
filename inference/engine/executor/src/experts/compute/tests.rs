use super::*;
use seismic_io::Want;
use std::time::{Duration, Instant};

/// Recorded topologies (lscpu, 2026-11-03): each node's cores, core `c`'s
/// threads `c` and `c + siblings`.
fn node(id: usize, memory: bool, cores: std::ops::Range<usize>, siblings: usize) -> seismic::MemoryNode {
    let cores = cores.map(|core| seismic::Core { threads: vec![core, core + siblings], performance: true }).collect();
    seismic::MemoryNode { id, memory, cores }
}

/// The read engine's share of the recorded hosts: two cores.
const READS: f64 = 2.;

/// The machine of `nodes`, the process allowed on `allowed`.
fn processors(nodes: Vec<seismic::MemoryNode>, allowed: std::ops::Range<usize>) -> seismic::Processors {
    seismic::Processors::new(nodes, allowed.collect(), seismic::ProcessorsReportedBy::LinuxSysfs).unwrap()
}

/// An EPYC 7313, 4 nodes of 4 cores (`0-3,16-19`, …), every CPU
/// allowed, a cgroup quota of 15.36 CPUs (`available_parallelism` 15).
fn epyc_7313() -> Machine {
    let nodes = (0..4).map(|index| node(index, true, index * 4..index * 4 + 4, 16)).collect::<Vec<_>>();
    Machine::of(&processors(nodes, 0..32), 15, busy_threads(1, READS), seismic::DeviceNode::Unknown)
}

/// A Xeon E5-2696 v3, one node of 18 cores (`0-35`), a quota of
/// 34.56 CPUs.
fn xeon_e5_2696() -> Machine {
    Machine::of(&processors(vec![node(0, true, 0..18, 18)], 0..36), 34, busy_threads(1, READS), seismic::DeviceNode::Node(0))
}

fn counts(candidates: &[ComputeThreads]) -> Vec<usize> {
    candidates.iter().map(ComputeThreads::count).collect()
}

/// A request's time from a recorded line: `expert` µs an expert of a
/// decode request's eight, a 20 µs handshake, resolved to `resolution` µs.
fn line(expert: f64, resolution: f64) -> crate::assessment::Sample {
    crate::assessment::Sample { seconds: (20. + 8. * expert) * 1e-6, resolution: resolution * 1e-6 }
}

/// Sized on the EPYC's line (cold, an expert: 48.7 µs at 4 threads, 47.7
/// at 8): of the quota's 15 less the runtime's busy threads, whole nodes
/// and powers of two of the first fit; 8 workers on nodes 0 and 1, and
/// every other thread on nodes 2 and 3, both hardware threads of each core.
#[test]
fn an_epyc_takes_two_whole_nodes_and_leaves_the_others_to_the_runtime() {
    let machine = epyc_7313();
    assert_eq!(busy_threads(1, READS), 6, "the submitter, the watcher's two, the read engine's two, the resize thread");
    assert_eq!(counts(&machine.candidates()), vec![1, 2, 4, 8], "12 is past 15 less 6");
    let curve = [(1, 160.), (2, 90.), (4, 48.7), (8, 47.7)];
    let threads = machine.sized(|threads| Ok(line(curve.iter().find(|(count, _)| *count == threads.count()).unwrap().1, 0.5))).unwrap();
    assert_eq!(threads.nodes, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    assert_eq!(threads.runtime.to_vec(), (8..16).chain(24..32).collect::<Vec<_>>(), "whole cores: the workers' siblings 16-23 free");
}

/// The EPYC's GPU hangs off node 1: the workers fill nodes 0 and 2, and
/// node 1, the GPU's, is the runtime's with node 3, so the device's own
/// threads sit beside it. A device of no known node (above) orders none.
#[test]
fn an_epyc_keeps_the_gpus_node_for_the_runtime() {
    let nodes = (0..4).map(|index| node(index, true, index * 4..index * 4 + 4, 16)).collect::<Vec<_>>();
    let machine = Machine::of(&processors(nodes, 0..32), 15, busy_threads(1, READS), seismic::DeviceNode::Node(1));
    let threads = machine.candidates().pop().unwrap();
    assert_eq!(threads.nodes, vec![vec![0, 1, 2, 3], vec![8, 9, 10, 11]]);
    assert_eq!(threads.runtime.to_vec(), (4..8).chain(12..16).chain(20..24).chain(28..32).collect::<Vec<_>>());
}

/// Sized on the Xeon's flat line (93.3 µs at 4, 84.4 at 8, 91.3 at 16, about 85
/// at 18): the fewest workers within the least's resolution.
#[test]
fn a_xeon_takes_the_fewest_workers_on_its_flat_curve() {
    let machine = xeon_e5_2696();
    assert_eq!(counts(&machine.candidates()), vec![1, 2, 4, 8, 16, 18], "powers of two within its one node, and the node");
    let curve = [(1, 300.), (2, 170.), (4, 93.3), (8, 84.4), (16, 91.3), (18, 84.9)];
    let threads = machine.sized(|threads| Ok(line(curve.iter().find(|(count, _)| *count == threads.count()).unwrap().1, 1.0))).unwrap();
    assert_eq!(threads.nodes, vec![(0..8).collect::<Vec<_>>()]);
    assert_eq!(threads.runtime.to_vec(), (8..18).chain(26..36).collect::<Vec<_>>(), "whole cores: the workers' siblings 18-25 free");
    // Where 18 resolves faster than 8, it is taken.
    let faster = [(1, 300.), (2, 170.), (4, 93.3), (8, 84.4), (16, 91.3), (18, 80.)];
    let threads = machine.sized(|threads| Ok(line(faster.iter().find(|(count, _)| *count == threads.count()).unwrap().1, 1.0))).unwrap();
    assert_eq!(threads.count(), 18);
}

/// A machine with no quota (`available_parallelism` is its every allowed
/// CPU): the runtime's busy threads still come off it. A 2990WX:
/// two of its four nodes have memory (0 and 2), so only theirs serve, and
/// the dies without memory are the runtime's; an affinity that excludes a
/// core's first thread places its worker on the second.
#[test]
fn a_machine_without_a_quota_and_nodes_without_memory() {
    let quota = 8;
    let small = Machine::of(&processors(vec![node(0, true, 0..4, 4)], 0..8), quota, busy_threads(1, READS), seismic::DeviceNode::Unknown);
    assert_eq!(counts(&small.candidates()), vec![1, 2], "8 CPUs less 6 busy");
    let b = vec![node(0, true, 0..8, 32), node(1, false, 16..24, 32), node(2, true, 8..16, 32), node(3, false, 24..32, 32)];
    let machine = Machine::of(&processors(b.clone(), 1..64), 30, busy_threads(1, READS), seismic::DeviceNode::Unknown);
    assert_eq!(counts(&machine.candidates()), vec![1, 2, 4, 8, 16]);
    let widest = machine.widest().unwrap();
    assert_eq!(widest.nodes, vec![vec![32, 1, 2, 3, 4, 5, 6, 7], (8..16).collect::<Vec<_>>()]);
    assert!(widest.runtime.iter().all(|cpu| (16..32).contains(cpu) || (48..64).contains(cpu)), "{:?}", widest.runtime);
    // Past the quota, one worker for the landings, whatever the runtime.
    assert_eq!(counts(&Machine::of(&processors(b.clone(), 0..64), 3, busy_threads(1, READS), seismic::DeviceNode::Unknown).candidates()), vec![1]);
    // No memory-attached core the process may run on: no pool.
    assert!(Machine::of(&processors(b.to_vec(), 16..32), 30, 1, seismic::DeviceNode::Unknown).widest().is_err());
}

/// The job sizes a pool with no reads beside it: a load whose reads leave
/// fewer workers takes the widest pool within both.
#[test]
fn a_stored_pool_stays_within_what_a_loads_reads_leave() {
    let nodes = (0..2).map(|index| node(index, true, index * 8..index * 8 + 8, 16)).collect::<Vec<_>>();
    let unread = Machine::of(&processors(nodes.clone(), 0..32), 24, busy_threads(1, 0.), seismic::DeviceNode::Unknown);
    assert_eq!(unread.at_most(16).unwrap().count(), 16, "the stored pool itself, whole");
    let read = Machine::of(&processors(nodes, 0..32), 24, busy_threads(1, 6.), seismic::DeviceNode::Unknown);
    assert_eq!(counts(&read.candidates()), vec![1, 2, 4, 8]);
    assert_eq!(read.at_most(16).unwrap().count(), 8, "16 is past 24 less 10");
    assert_eq!(read.at_most(2).unwrap().count(), 2);
}

/// The busy threads are the device's own on the host, the watcher's, the
/// read engine's and the resize thread's: a CPU device's pool takes its
/// participants, so the executor keeps off the cores the device computes
/// on.
#[test]
fn this_machine_sizes_beside_its_device() {
    let cpu = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
    assert!(cpu.host_threads() >= 1, "a CPU device's pool has its submitter");
    // The read engine's share of the host, as a probe measures it, to the
    // nearest whole core: GB10's 0.77–1.38 reserves one in every process.
    let reads = 1.4;
    let others = cpu.host_threads() + super::super::watcher::THREADS + super::super::bank::RESIZE_THREADS;
    assert_eq!(busy_threads(cpu.host_threads(), reads), others + 1);
    assert_eq!(busy_threads(cpu.host_threads(), 0.77), others + 1);
    assert_eq!(busy_threads(cpu.host_threads(), 1.6), others + 2);
    let _placement = crate::test_support::process_placement();
    let processors = seismic::Processors::discover().unwrap();
    let quota = std::thread::available_parallelism().unwrap().get();
    let machine = Machine::discover(&cpu, reads).unwrap();
    assert_eq!(cpu.host_node(), Ok(seismic::DeviceNode::Unknown), "a CPU device sits on no bus");
    assert_eq!(machine, Machine::of(&processors, quota, busy_threads(cpu.host_threads(), reads), seismic::DeviceNode::Unknown));
    assert!(machine.widest().unwrap().count() >= 1);
}

/// A job of `pieces` pieces counting each run, and its ends.
struct Counted {
    pieces: usize,
    runs: Vec<AtomicUsize>,
    ended: AtomicUsize,
    /// The pieces run when the end ran.
    at_end: AtomicUsize,
    urgency: Arc<Urgency>,
    /// Its place among the ends of the jobs sharing `order`.
    order: Arc<AtomicUsize>,
    ended_at: AtomicUsize,
}

impl Counted {
    fn new(pieces: usize) -> Arc<Self> {
        Self::ordered(pieces, None, Arc::default())
    }

    fn ordered(pieces: usize, ahead: Option<Want>, order: Arc<AtomicUsize>) -> Arc<Self> {
        Arc::new(Self {
            pieces,
            runs: (0..pieces).map(|_| AtomicUsize::new(0)).collect(),
            ended: AtomicUsize::new(0),
            at_end: AtomicUsize::new(0),
            urgency: ahead.map_or_else(Urgency::waited, Urgency::ahead),
            order,
            ended_at: AtomicUsize::new(usize::MAX),
        })
    }
}

impl Job for Counted {
    fn pieces(&self) -> usize {
        self.pieces
    }
    fn piece(&self, index: usize) {
        self.runs[index].fetch_add(1, Ordering::AcqRel);
    }
    fn end(&self) {
        let run = self.runs.iter().map(|runs| runs.load(Ordering::Acquire)).sum();
        self.at_end.store(run, Ordering::Release);
        self.ended_at.store(self.order.fetch_add(1, Ordering::AcqRel), Ordering::Release);
        self.ended.fetch_add(1, Ordering::AcqRel);
    }
    fn want(&self) -> Want {
        self.urgency.get()
    }
}

fn until(condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "the executor never got there");
        std::thread::sleep(Duration::from_micros(50));
    }
}

/// Jobs run on parked workers: a submission wakes them, every piece runs
/// once, and the end once after the last; with no flight issued the workers
/// park again. An executor of no site layer
/// issues no flight. A job submitted once the executor stopped never runs.
#[test]
fn jobs_run_each_piece_once_then_end_once() {
    for workers in [1usize, 3] {
        let compute = HostCompute::start(Vec::new(), &ComputeThreads::placeable(workers)).unwrap();
        let handle = compute.handle();
        assert!(handle.flight().is_none(), "no site layer, no flight");
        let jobs = (0..4).map(|job| Counted::new(job * 5)).collect::<Vec<_>>();
        for job in &jobs {
            handle.submit(job.clone());
        }
        until(|| jobs.iter().all(|job| job.ended.load(Ordering::Acquire) == 1));
        for job in &jobs {
            assert!(job.runs.iter().all(|runs| runs.load(Ordering::Acquire) == 1), "{workers} workers: each piece once");
            assert_eq!(job.at_end.load(Ordering::Acquire), job.pieces, "{workers} workers: the end after every piece");
        }
        // Parked: no flight, nothing queued.
        until(|| handle.inner.jobs.lock().unwrap().is_empty());
        drop(compute);
        let late = Counted::new(2);
        handle.submit(late.clone());
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(late.ended.load(Ordering::Acquire), 0, "a stopped executor runs nothing");
    }
}

/// A panicking piece is recorded, which ends serving, and the worker serves
/// on.
#[test]
fn a_panicking_piece_is_recorded_and_the_worker_serves_on() {
    struct Panics;
    impl Job for Panics {
        fn pieces(&self) -> usize {
            1
        }
        fn piece(&self, _: usize) {
            panic!("a piece's defect");
        }
        fn end(&self) {}
        fn want(&self) -> Want {
            Want::now()
        }
    }
    let compute = HostCompute::start(Vec::new(), &ComputeThreads::placeable(1)).unwrap();
    let handle = compute.handle();
    handle.submit(Arc::new(Panics));
    until(|| handle.panicked().is_some());
    assert!(handle.panicked().unwrap().contains("a piece's defect"));
    let after = Counted::new(3);
    handle.submit(after.clone());
    until(|| after.ended.load(Ordering::Acquire) == 1);
}

/// A worker takes the most urgent job's pieces first, a job hastened while
/// it waits among them, and the equally urgent in submission order: on one
/// worker, held busy while the queue fills, the jobs end due first by
/// deadline (the hastened hint due from its hastening), then the hints.
#[test]
fn the_most_urgent_job_runs_first_in_submission_order() {
    struct Gate(Arc<std::sync::Barrier>);
    impl Job for Gate {
        fn pieces(&self) -> usize {
            1
        }
        fn piece(&self, _: usize) {
            self.0.wait();
        }
        fn end(&self) {}
        fn want(&self) -> Want {
            Want::now()
        }
    }
    let compute = HostCompute::start(Vec::new(), &ComputeThreads::placeable(1)).unwrap();
    let handle = compute.handle();
    let gate = Arc::new(std::sync::Barrier::new(2));
    handle.submit(Arc::new(Gate(gate.clone())));
    // The worker holds the gate's one piece.
    until(|| handle.inner.jobs.lock().unwrap().iter().all(|job| job.next.load(Ordering::Acquire) == 1));
    let order = Arc::new(AtomicUsize::new(0));
    let ahead = Some(Want::at(std::time::Instant::now() + std::time::Duration::from_secs(3600), 0.));
    let kinds = [ahead, None, ahead, ahead, None];
    let jobs = kinds.map(|ahead| Counted::ordered(2, ahead, order.clone()));
    for job in &jobs {
        handle.submit(job.clone());
    }
    let due = jobs[1].urgency.get();
    jobs[2].urgency.hasten();
    jobs[1].urgency.hasten();
    assert_eq!(jobs[1].urgency.get(), due, "a due job is not made later");
    assert!(jobs[2].urgency.is_waited() && !jobs[3].urgency.is_waited());
    gate.wait();
    until(|| jobs.iter().all(|job| job.ended.load(Ordering::Acquire) == 1));
    let ends = jobs.each_ref().map(|job| job.ended_at.load(Ordering::Acquire));
    // The due jobs by deadline (the hastened one due last), then those not
    // due in submission order.
    assert_eq!(ends, [3, 0, 2, 4, 1]);
}
