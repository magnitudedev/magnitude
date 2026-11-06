//! The driver memory a device's instantiated graphs hold
//! ([`GraphCost::Pooled`]), measured once per device in a process.
//!
//! Instantiating a graph and uploading it takes device memory no allocation
//! charges, all of it then (a first launch takes none): for the executable
//! graph, and for each kernel node whatever its parameters. On an RTX 3090
//! (2026-11-02, instantiated and uploaded) that was about 90 KiB a graph
//! and 7.9 KiB a node, each linear in its count. The driver takes it in
//! steps of the device's allocation granule, and keeps what a
//! destroyed graph held for the graphs instantiated after it, so a
//! measurement is of fresh memory only before any graph is destroyed.
//!
//! Each term is a slope, measured over enough of its unit that the granule
//! resolves it to [`SPAN`]ths: one-node graphs (a graph and a node each)
//! and chains of [`CHAIN`] nodes (a graph and that many nodes), each
//! instantiated in batches until what the process holds on the device grew
//! by [`SPAN`] granules. Every graph stays instantiated until all are
//! measured.
//!
//! What a slope reads is the memory the device holds for this process, as
//! NVML counts it ([`Holdings`]), never the device's free memory: other
//! programs move that, and on a device whose memory is the host's (GB10)
//! so does every allocation and page the system makes, so a measurement
//! beside other work read 3.96 MB a one-node graph and nothing a chain, a
//! negative term. Where NVML counts no process (no NVML: Windows) it reads
//! what the device holds of everyone's, and says so. Either reading can
//! still move under the measurement (another thread of the process, a
//! graph the process destroyed whose memory the driver hands on), so each
//! slope is measured twice, interleaved, and a measurement whose repeats
//! differ by more than its resolution is refused rather than kept.

use crate::driver::{self, Edge, KernelNodeParams};
use crate::executor::Device;
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::{GraphCensus, GraphCost};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;

/// The granules each measurement spans at least: each term is resolved to
/// a granule in this many, what the free memory's granular readings allow.
const SPAN: u64 = 16;
/// The nodes of each chain the node term is measured over.
const CHAIN: u64 = 1024;
/// The most units a measurement takes: where they span fewer than
/// [`SPAN`] granules, a unit costs less than a 4096th of a granule per
/// granule of span, and the slope is still what they took.
const MOST_UNITS: u64 = 1 << 16;

/// The empty kernel a calibration node launches, taking one word, as PTX
/// the driver loads (NUL-terminated), and its entry.
const KERNEL: (&str, &str) = (
    ".version 7.0\n.target sm_50\n.address_size 64\n.visible .entry seismic_graph(.param .u64 seismic_word)\n{\n\tret;\n}\n\0",
    "seismic_graph",
);

/// What a device's instantiated graphs are charged at before [`graph_cost`]
/// measured it there: a bound above every device measured so far (an RTX
/// 3090 took 100,146 bytes a graph and 8,094 a node, in 2 MiB granules),
/// so what is planned at it still fits once the device's own cost is read.
pub const UNMEASURED: GraphCost = GraphCost::Pooled { graph: 256 << 10, launch: 16 << 10, granule: 2 << 20 };

/// Each device's measured cost ([`graph_cost`]), by ordinal.
static COSTS: Mutex<BTreeMap<u32, GraphCost>> = Mutex::new(BTreeMap::new());

/// What graphs instantiated on `device` hold, measured on its first call
/// in the process and kept for the device's ordinal after.
pub fn graph_cost(device: &Device) -> Result<GraphCost, ExecutionError> {
    let mut costs = COSTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(cost) = costs.get(&device.ordinal()) {
        return Ok(*cost);
    }
    let cost = measure(device)?;
    costs.insert(device.ordinal(), cost);
    Ok(cost)
}

/// The driver memory graphs of `census` take on `device`, at its cost as
/// measured in this process, else at [`UNMEASURED`]'s bound. Measures
/// nothing.
pub(crate) fn census_bytes(device: &Device, census: seismic_compiler::graph_cost::GraphCensus) -> u64 {
    let costs = COSTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let cost = costs.get(&device.ordinal()).copied().unwrap_or(UNMEASURED);
    cost.bytes(census).map_or(u64::MAX, |bytes| bytes.device)
}

/// The driver memory a device's instantiated graphs hold: the graphs and
/// launches of every live graph priced at the device's cost
/// ([`census_bytes`]), rounded to the granule once over them all, as the
/// driver takes it. Never read from the device's memory, which other work
/// moves. Each graph instantiated or dropped moves it by the change in that
/// price, which the device account holds as one charge.
pub(crate) struct GraphPool {
    account: std::sync::Arc<dyn seismic_compiler::executable::MemoryAccount>,
    context: std::sync::Arc<driver::Context>,
    state: Mutex<Pooled>,
    held: std::sync::atomic::AtomicU64,
}

/// The live graphs' census and the charge that holds their price.
struct Pooled {
    census: GraphCensus,
    charge: Option<Box<dyn seismic_compiler::executable::MemoryHold>>,
}

impl GraphPool {
    pub(crate) fn new(
        account: std::sync::Arc<dyn seismic_compiler::executable::MemoryAccount>,
        context: std::sync::Arc<driver::Context>,
    ) -> Self {
        Self { account, context, state: Mutex::new(Pooled { census: GraphCensus::default(), charge: None }), held: 0.into() }
    }

    /// What the live graphs hold now.
    pub(crate) fn bytes(&self) -> u64 {
        self.held.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A graph of `census` instantiated on `device`.
    pub(crate) fn instantiated(&self, device: &Device, census: GraphCensus) -> Result<(), ExecutionError> {
        self.moved(device, |live| live.plus(census))
    }

    /// A graph of `census` dropped on `device`.
    pub(crate) fn dropped(&self, device: &Device, census: GraphCensus) {
        self.moved(device, |live| {
            let left = |live: u64, dropped: u64| live.checked_sub(dropped).expect("a dropped graph was counted live");
            Some(GraphCensus { graphs: left(live.graphs, census.graphs), launches: left(live.launches, census.launches), ..live })
        })
        .expect("fewer live graphs price within what more did");
    }

    fn moved(&self, device: &Device, change: impl FnOnce(GraphCensus) -> Option<GraphCensus>) -> Result<(), ExecutionError> {
        let mut pooled = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let census = change(pooled.census).ok_or_else(|| failure("the live graphs' census overflows"))?;
        let price = census_bytes(device, census);
        if price == u64::MAX {
            return Err(failure("the live graphs' census prices past the address range"));
        }
        let was = self.held.swap(price, std::sync::atomic::Ordering::Relaxed);
        pooled.census = census;
        // The new charge before the old one ends, so the account never shows
        // less than either.
        let charge = self.account.record(price);
        drop(pooled.charge.replace(charge));
        if price > was {
            self.context.driver_allocated(price - was);
        } else {
            self.context.driver_freed(was - price);
        }
        Ok(())
    }
}

fn failure(error: impl std::fmt::Debug) -> ExecutionError {
    ExecutionError::SubmissionFailed(format!("measuring the graphs' driver memory: {error:?}"))
}

/// One graph's kernel node: an empty kernel of `function` taking
/// `parameters`.
fn node(function: driver::Handle, parameters: &mut [*mut c_void; 1]) -> KernelNodeParams {
    KernelNodeParams {
        function,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        parameters: parameters.as_mut_ptr(),
        extra: std::ptr::null_mut(),
        kernel: std::ptr::null_mut(),
        context: std::ptr::null_mut(),
    }
}

/// A chain of `count` nodes launching `launch`.
fn chain(
    context: &std::sync::Arc<driver::Context>,
    launch: &KernelNodeParams,
    count: u64,
) -> Result<driver::Graph, ExecutionError> {
    let mut graph = driver::Graph::new(context).map_err(failure)?;
    for index in 0..count as usize {
        let after: &[usize] = if index == 0 { &[] } else { &[index - 1] };
        graph.push_kernel(launch, Edge::Complete, after).map_err(failure)?;
    }
    Ok(graph)
}

/// `graph` instantiated and uploaded, as a replay prepares its graphs
/// (`DirectGraphBuilder::instantiate`).
fn prepared(device: &Device, graph: driver::Graph) -> Result<driver::GraphExec, ExecutionError> {
    let exec = graph.instantiate().map_err(failure)?;
    exec.upload(device.stream()).map_err(failure)?;
    Ok(exec)
}

/// What the device holds that a slope reads: this process's memory on it as
/// NVML counts it, or, where NVML counts no process, everything it holds.
enum Holdings {
    Process(crate::nvml::NvmlDevice, u32),
    Device,
}

impl Holdings {
    fn of(device: &Device) -> Result<Self, ExecutionError> {
        let uuid = crate::profile::describe(device.ordinal()).map_err(failure)?.uuid;
        let pid = std::process::id();
        Ok(match crate::nvml::NvmlDevice::open(uuid) {
            Some(nvml) if nvml.process_bytes(pid).is_ok() => Self::Process(nvml, pid),
            _ => Self::Device,
        })
    }

    fn read(&self, device: &Device) -> Result<u64, ExecutionError> {
        match self {
            Self::Process(nvml, pid) => nvml.process_bytes(*pid).map_err(failure),
            Self::Device => device.memory_info().map(|info| info.total_bytes - info.free_bytes),
        }
    }
}

fn measure(device: &Device) -> Result<GraphCost, ExecutionError> {
    let context = device.context();
    let granule = driver::allocation_granularity(context).map_err(failure)?;
    let (source, entry) = KERNEL;
    let (module, function) = driver::load_module(context, source.as_bytes(), entry).map_err(failure)?;
    let mut word = [0u8; 8];
    let mut parameters = [word.as_mut_ptr().cast::<c_void>()];
    let launch = node(function, &mut parameters);
    let holdings = Holdings::of(device)?;
    let held = || holdings.read(device);
    let mut kept = Vec::new();
    // What each graph of `count` nodes takes, a slope over as many as span
    // `SPAN` granules: what the holdings grew by over them, per graph.
    let mut slope = |count: u64| -> Result<f64, ExecutionError> {
        let before = held()?;
        let mut graphs = 0u64;
        while held()?.saturating_sub(before) < SPAN * granule && graphs * count < MOST_UNITS {
            kept.push(prepared(device, chain(context, &launch, count)?)?);
            graphs += 1;
        }
        // Holdings that grew by a span before any graph was taken by other
        // work: nothing here measures a graph.
        if graphs == 0 {
            return Err(failure("the device's holdings grew before any graph was instantiated"));
        }
        Ok(held()?.saturating_sub(before) as f64 / graphs as f64)
    };
    let first = (slope(1)?, slope(CHAIN)?);
    let second = (slope(1)?, slope(CHAIN)?);
    drop(kept);
    drop(module);
    let (single, chained) = agreed(first, second, granule)?;
    let (graph, launch) = terms(single, chained)?;
    Ok(GraphCost::Pooled { graph, launch, granule })
}

/// The slopes of two interleaved measurements, `(one-node graph, chain)`
/// each, as one: their means, where each pair agrees within what a span
/// resolves (a [`SPAN`]th of the larger, and at least a granule over the
/// units the larger of a pair's slopes took a span in). Repeats that differ
/// by more read memory something else moved: refused.
fn agreed(first: (f64, f64), second: (f64, f64), granule: u64) -> Result<(f64, f64), ExecutionError> {
    let agree = |a: f64, b: f64| (a - b).abs() <= a.max(b) / SPAN as f64 + granule as f64 / SPAN as f64;
    if !(agree(first.0, second.0) && agree(first.1, second.1)) {
        return Err(failure(format!(
            "the device's memory moved under the measurement: a one-node graph read {} then {} bytes, a chain {} then {}",
            first.0, second.0, first.1, second.1
        )));
    }
    Ok(((first.0 + second.0) / 2., (first.1 + second.1) / 2.))
}

/// The bytes of a graph and of a node, from what a one-node graph and a
/// chain of [`CHAIN`] nodes each took: a one-node graph is a graph and a
/// node; a chain a graph and `CHAIN` nodes. A negative term is no cost:
/// other work moved the device's free memory across the measurement.
fn terms(single: f64, chained: f64) -> Result<(u64, u64), ExecutionError> {
    let node = (chained - single) / (CHAIN - 1) as f64;
    let graph = single - node;
    if graph < 0.0 || node < 0.0 {
        return Err(failure(format!("{single} bytes a graph and {chained} a chain leave a negative term")));
    }
    Ok((graph.round() as u64, node.round() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The terms of a graph and a node are what the slopes imply, and slopes
    /// that imply a negative term are refused.
    #[test]
    fn a_negative_term_is_refused() {
        let chain = 90_000.0 + CHAIN as f64 * 8_000.0;
        assert_eq!(terms(98_000.0, chain).unwrap(), (90_000, 8_000));
        assert!(terms(chain, 98_000.0).is_err());
        assert!(terms(1_000.0, chain).is_err());
    }

    /// Repeats that agree within a span's resolution are one measurement;
    /// repeats that differ by more are refused, as the reading a39d0c took
    /// beside another program's work (3.96 MB a one-node graph).
    #[test]
    fn repeats_that_disagree_are_refused() {
        let granule = 2 << 20;
        let chain = 100_000.0 + CHAIN as f64 * 8_000.0;
        assert_eq!(agreed((100_000.0, chain), (104_000.0, chain), granule).unwrap(), (102_000.0, chain));
        assert!(agreed((3_964_118.9, 0.0), (100_000.0, chain), granule).is_err());
        assert!(agreed((100_000.0, chain), (100_000.0, chain / 2.0), granule).is_err());
    }

    /// A measurement after other work on the same device (graphs
    /// instantiated and destroyed, whose memory the driver keeps, and
    /// memory allocated and freed) reads what one before it did, within a
    /// span's resolution, or is refused: never another cost.
    #[test]
    fn a_measurement_after_other_work_reads_the_same_cost_or_none() {
        let Some((_exclusive, opened)) = crate::test_device() else {
            return;
        };
        let device = opened.service;
        let first = measure(&device).unwrap();
        let context = device.context();
        let (source, entry) = KERNEL;
        let (_module, function) = driver::load_module(context, source.as_bytes(), entry).unwrap();
        let mut word = [0u8; 8];
        let mut parameters = [word.as_mut_ptr().cast::<c_void>()];
        let launch = node(function, &mut parameters);
        let other = (0..256).map(|count| prepared(&device, chain(context, &launch, 1 + count % 64).unwrap()).unwrap()).collect::<Vec<_>>();
        drop(other);
        let buffers = (0..8).map(|_| device.allocate_within(64 << 20, 256, None).unwrap()).collect::<Vec<_>>();
        drop(buffers);
        let (GraphCost::Pooled { graph, launch, granule }, Ok(GraphCost::Pooled { graph: again, launch: launched, .. })) = (first, measure(&device)) else {
            eprintln!("the measurement after other work was refused");
            return;
        };
        let close = |a: u64, b: u64| a.abs_diff(b) as f64 <= a.max(b) as f64 / SPAN as f64 + granule as f64 / SPAN as f64;
        assert!(close(graph, again) && close(launch, launched), "{first:?} then {graph}/{launch} vs {again}/{launched}");
    }

    /// Graphs unlike the calibration's, as a class's are: many graphs of
    /// chains of a length the calibration measures none of. What instantiating them
    /// takes, in a fresh process, is what the measured cost prices their
    /// census at, to within a granule and a [`SPAN`]th (on a device; none
    /// skips). Run alone, before any other graph of the process.
    #[test]
    fn the_measured_cost_prices_a_class_of_graphs() {
        const GRAPHS: u64 = 64;
        const NODES: u64 = 256;
        let Some((_exclusive, opened)) = crate::test_device() else {
            return;
        };
        let device = opened.service;
        let context = device.context();
        let (source, entry) = KERNEL;
        let (_module, function) = driver::load_module(context, source.as_bytes(), entry).unwrap();
        let mut word = [0u8; 8];
        let mut parameters = [word.as_mut_ptr().cast::<c_void>()];
        let launch = node(function, &mut parameters);
        let mut census = GraphCensus::default();
        let mut graphs = Vec::new();
        for _ in 0..GRAPHS {
            let mut graph = driver::Graph::new(context).unwrap();
            let mut launches = GraphCensus::default();
            for index in 0..NODES as usize {
                let after: &[usize] = if index == 0 { &[] } else { &[index - 1] };
                graph.push_kernel(&launch, Edge::Complete, after).unwrap();
                launches = launches.plus(GraphCensus::launch(0)).unwrap();
            }
            census = census.plus(GraphCensus::graph(launches).unwrap()).unwrap();
            graphs.push(graph);
        }
        // Measured before the calibration, whose destroyed graphs' memory
        // the driver would hand these; kept while it measures, so it takes
        // fresh memory too. Read as the calibration reads: the process's own
        // holdings, which no other process's work moves.
        let holdings = Holdings::of(&device).unwrap();
        let before = holdings.read(&device).unwrap();
        let _instantiated = graphs.into_iter().map(|graph| prepared(&device, graph).unwrap()).collect::<Vec<_>>();
        let taken = holdings.read(&device).unwrap() - before;
        let cost = measure(&device).unwrap();
        let GraphCost::Pooled { granule, .. } = cost else {
            panic!("CUDA prices graphs pooled: {cost:?}");
        };
        let priced = cost.bytes(census).unwrap().device;
        eprintln!("{cost:?}: {census:?} took {taken} bytes, priced at {priced}");
        assert!(
            priced.abs_diff(taken) <= granule + taken / SPAN,
            "{census:?} took {taken} bytes; {cost:?} prices it at {priced}"
        );
    }

    /// Where instantiated graphs' device memory goes, step by step, in a
    /// fresh process with nothing destroyed: per one-node graph, per node
    /// by its parameter bytes, and what a first launch adds. Run alone (`--ignored graph_memory_probe`).
    #[test]
    #[ignore = "a measurement of the driver's graph memory"]
    fn graph_memory_probe() {
        let Some((_exclusive, opened)) = crate::test_device() else {
            return;
        };
        let device = opened.service;
        let context = device.context();
        let load = |parameters: u64| {
            let source = format!(
                ".version 7.0\n.target sm_50\n.address_size 64\n\
                 .visible .entry seismic_graph(.param .align 8 .b8 seismic_words[{parameters}])\n{{\n\tret;\n}}\n\0"
            );
            driver::load_module(context, source.as_bytes(), "seismic_graph").unwrap()
        };
        let (_small_module, small) = load(8);
        let (_medium_module, medium) = load(1040);
        let (_large_module, large) = load(4096);
        let mut word = vec![0u8; 8];
        let mut medium_words = vec![0u8; 1040];
        let mut words = vec![0u8; 4096];
        let mut small_parameters = [word.as_mut_ptr().cast::<c_void>()];
        let mut medium_parameters = [medium_words.as_mut_ptr().cast::<c_void>()];
        let mut large_parameters = [words.as_mut_ptr().cast::<c_void>()];
        let small = node(small, &mut small_parameters);
        let medium = node(medium, &mut medium_parameters);
        let large = node(large, &mut large_parameters);
        let chain = |launch: &KernelNodeParams, count: u64| chain(context, launch, count).unwrap();
        let free = || device.memory_info().unwrap().free_bytes as i64;
        let start = free();
        let mut kept = Vec::new();
        let step = |what: &str, graphs: Vec<driver::Graph>, kept: &mut Vec<driver::GraphExec>| {
            let before = free();
            for graph in graphs {
                kept.push(graph.instantiate().unwrap());
            }
            let taken = before - free();
            eprintln!("probe {what}: {taken} bytes ({:.3} MiB)", taken as f64 / 1048576.0);
        };
        step("a chain of 4096 small nodes", vec![chain(&small, 4096)], &mut kept);
        step("a chain of 4096 1040-byte nodes", vec![chain(&medium, 4096)], &mut kept);
        step("a chain of 4096 4096-byte nodes", vec![chain(&large, 4096)], &mut kept);
        step("512 one-node graphs", (0..512).map(|_| chain(&small, 1)).collect(), &mut kept);
        step("64 chains of 64 small nodes", (0..64).map(|_| chain(&small, 64)).collect(), &mut kept);
        let stream = device.stream();
        for (what, graph) in [("512 small", chain(&small, 512)), ("4096 small", chain(&small, 4096))] {
            let exec = graph.instantiate().unwrap();
            let before = free();
            exec.launch(stream).unwrap();
            stream.synchronize().unwrap();
            eprintln!("probe first launch of {what}: {} bytes", before - free());
            kept.push(exec);
        }
        let before = free();
        kept.truncate(1);
        eprintln!("probe destroying all but one returned {} bytes", free() - before);
        step("512 one-node graphs after", (0..512).map(|_| chain(&small, 1)).collect(), &mut kept);
        eprintln!("probe total held: {} bytes", start - free());
    }

    /// What an instantiated graph's parameters may change without a new
    /// instantiation, as shape-shared graphs need (one graph per layout,
    /// every class, place and encoding a parameter): a kernel node's
    /// function, grid and block dimensions and arguments
    /// (`cuGraphExecKernelNodeSetParams`), and whether it runs at all
    /// (`cuGraphNodeSetEnabled`); whether either waits for queued work; and
    /// what a disabled node costs a launch. Run alone
    /// (`--ignored graph_parameters_probe --nocapture`).
    #[test]
    #[ignore = "a probe of the driver's graph parameter updates"]
    fn graph_parameters_probe() {
        let Some((_exclusive, opened)) = crate::test_device() else {
            return;
        };
        let device = opened.service;
        let context = device.context();
        let stream = device.stream();
        // `fill` writes `value` at each thread's index; `twice` writes twice
        // `value`, from a module of its own; `spin` runs for `cycles`.
        let source = |name: &str, scale: u32| {
            format!(
                ".version 7.0\n.target sm_50\n.address_size 64\n\
                 .visible .entry {name}(.param .u64 out, .param .u32 value)\n{{\n\
                 \t.reg .u64 %a, %o;\n\t.reg .u32 %v, %t, %b, %d, %i;\n\
                 \tld.param.u64 %a, [out];\n\tld.param.u32 %v, [value];\n\tmul.lo.u32 %v, %v, {scale};\n\
                 \tmov.u32 %t, %tid.x;\n\tmov.u32 %b, %ctaid.x;\n\tmov.u32 %d, %ntid.x;\n\
                 \tmad.lo.u32 %i, %b, %d, %t;\n\tmul.wide.u32 %o, %i, 4;\n\tadd.u64 %a, %a, %o;\n\
                 \tst.global.u32 [%a], %v;\n\tret;\n}}\n\0"
            )
        };
        let (_fill_module, fill) = driver::load_module(context, source("fill", 1).as_bytes(), "fill").unwrap();
        let (_twice_module, twice) = driver::load_module(context, source("twice", 2).as_bytes(), "twice").unwrap();
        let spin_source = ".version 7.0\n.target sm_50\n.address_size 64\n\
             .visible .entry spin(.param .u64 cycles)\n{\n\t.reg .u64 %s, %n, %c;\n\t.reg .pred %p;\n\
             \tld.param.u64 %n, [cycles];\n\tmov.u64 %s, %clock64;\nLOOP:\n\tmov.u64 %c, %clock64;\n\
             \tsub.u64 %c, %c, %s;\n\tsetp.lt.u64 %p, %c, %n;\n\t@%p bra LOOP;\n\tret;\n}\n\0";
        let (_spin_module, spin) = driver::load_module(context, spin_source.as_bytes(), "spin").unwrap();
        const ELEMENTS: usize = 1024;
        let out = driver::HostMapped::new(context, &device.accounts().upload, ELEMENTS * 4).unwrap();
        out.write_at(0, &[0u8; ELEMENTS * 4]);
        let read = || {
            let mut bytes = vec![0u8; ELEMENTS * 4];
            out.read_at(0, &mut bytes);
            bytes.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>()
        };
        // The driver copies the values the parameters point to at each
        // update, so they are changed in place between updates.
        let (address, value) = (std::cell::Cell::new(out.device), std::cell::Cell::new(7u32));
        let mut parameters = [address.as_ptr().cast::<c_void>(), value.as_ptr().cast::<c_void>()];
        let params = |function, grid: u32, block: u32, parameters: &mut [*mut c_void; 2]| KernelNodeParams {
            function,
            grid: [grid, 1, 1],
            block: [block, 1, 1],
            shared_bytes: 0,
            parameters: parameters.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kernel: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
        };
        // A graph of one `fill` node over 1 block of 32 threads.
        let mut graph = driver::Graph::new(context).unwrap();
        graph.push_kernel(&params(fill, 1, 32, &mut parameters), Edge::Complete, &[]).unwrap();
        let exec = graph.instantiate().unwrap();
        exec.launch(stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(read()[..33], [[7; 32].as_slice(), &[0]].concat(), "the recorded node");

        // 1. Another function, grid and block, and argument.
        value.set(5);
        let changed = exec.set_kernel(graph.node(0), &params(twice, 4, 64, &mut parameters));
        eprintln!("probe set another function, grid 4, block 64: {changed:?}");
        if changed.is_ok() {
            exec.launch(stream).unwrap();
            stream.synchronize().unwrap();
            let read = read();
            eprintln!("probe after: first {:?}, at 255 {}, at 256 {}", &read[..2], read[255], read[256]);
            assert!(read[..256].iter().all(|&word| word == 10), "the updated function, grid and block ran");
        }
        // The same function, other dims only.
        out.write_at(0, &[0u8; ELEMENTS * 4]);
        value.set(3);
        let dims = exec.set_kernel(graph.node(0), &params(fill, 8, 128, &mut parameters));
        exec.launch(stream).unwrap();
        stream.synchronize().unwrap();
        eprintln!("probe set grid 8, block 128 on the same function: {dims:?}, all 1024 written: {}", read().iter().all(|&word| word == 3));

        // 2. Disabled: the node does nothing.
        out.write_at(0, &[0u8; ELEMENTS * 4]);
        exec.set_enabled(graph.node(0), false).unwrap();
        exec.launch(stream).unwrap();
        stream.synchronize().unwrap();
        assert!(read().iter().all(|&word| word == 0), "a disabled node writes nothing");
        exec.set_enabled(graph.node(0), true).unwrap();
        exec.launch(stream).unwrap();
        stream.synchronize().unwrap();
        assert!(read().iter().all(|&word| word == 3), "enabled again, it runs");

        // 3. Neither update waits for queued work: each is timed on the host
        // while a ~0.5 s kernel runs ahead of it on the stream.
        let mut cycles = 1_000_000_000u64;
        let mut spin_parameters = [(&mut cycles as *mut u64).cast::<c_void>()];
        let spin_node = KernelNodeParams {
            function: spin,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_bytes: 0,
            parameters: spin_parameters.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kernel: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
        };
        let mut spinning = driver::Graph::new(context).unwrap();
        spinning.push_kernel(&spin_node, Edge::Complete, &[]).unwrap();
        let spinning = spinning.instantiate().unwrap();
        let start = std::time::Instant::now();
        spinning.launch(stream).unwrap();
        let queued = start.elapsed();
        value.set(9);
        let update = std::time::Instant::now();
        exec.set_kernel(graph.node(0), &params(fill, 8, 128, &mut parameters)).unwrap();
        let set = update.elapsed();
        let toggle = std::time::Instant::now();
        exec.set_enabled(graph.node(0), false).unwrap();
        exec.set_enabled(graph.node(0), true).unwrap();
        let toggled = toggle.elapsed();
        exec.launch(stream).unwrap();
        let before_sync = start.elapsed();
        stream.synchronize().unwrap();
        let spun = start.elapsed();
        eprintln!(
            "probe behind a spinning kernel: queued {queued:?}; set params {set:?}; two toggles {toggled:?}; \
             launched by {before_sync:?}; spin ended {spun:?}"
        );
        assert!(read().iter().all(|&word| word == 9), "the update behind the spin ran");
        assert!(before_sync * 4 < spun, "an update or toggle waited for the queued kernel");

        // What a disabled node costs: a chain of 256 `fill` nodes launched
        // 200 times all enabled, all disabled, and a one-node graph.
        let mut fills = |count: usize| {
            let mut chain = driver::Graph::new(context).unwrap();
            for index in 0..count {
                let after: &[usize] = if index == 0 { &[] } else { &[index - 1] };
                chain.push_kernel(&params(fill, 1, 32, &mut parameters), Edge::Complete, after).unwrap();
            }
            chain
        };
        let timed = |exec: &driver::GraphExec| {
            exec.launch(stream).unwrap();
            stream.synchronize().unwrap();
            let start = std::time::Instant::now();
            for _ in 0..200 {
                exec.launch(stream).unwrap();
            }
            stream.synchronize().unwrap();
            start.elapsed() / 200
        };
        let long = fills(256);
        let long_exec = long.instantiate().unwrap();
        let enabled = timed(&long_exec);
        for node in 0..256 {
            long_exec.set_enabled(long.node(node), false).unwrap();
        }
        let disabled = timed(&long_exec);
        let one = fills(1);
        let one_exec = one.instantiate().unwrap();
        let single = timed(&one_exec);
        eprintln!(
            "probe launch of 256 nodes: enabled {enabled:?}, disabled {disabled:?}; one node {single:?}; \
             a disabled node {:?}",
            disabled.saturating_sub(single) / 255
        );
    }

    /// What a replay's parameter update costs at a decode run's shape: a
    /// chain of 16 nodes, each of 24 pointers and a 512-byte words struct,
    /// launched 400 times alternating two bindings of every pointer (a step
    /// and its lookahead successor on two slots) against the same with no
    /// update, and against two instantiations replayed in turn; host time
    /// per launch, its updates included. Run alone
    /// (`--ignored graph_update_cost_probe --nocapture`).
    #[test]
    #[ignore = "a probe of the driver's graph parameter update cost"]
    fn graph_update_cost_probe() {
        const NODES: usize = 16;
        const POINTERS: usize = 24;
        const WORDS: usize = 512;
        const LAUNCHES: usize = 400;
        let Some((_exclusive, opened)) = crate::test_device() else {
            return;
        };
        let device = opened.service;
        let context = device.context();
        let stream = device.stream();
        let params = (0..POINTERS).map(|index| format!(".param .u64 p{index}")).collect::<Vec<_>>().join(", ");
        let source = format!(
            ".version 7.0\n.target sm_50\n.address_size 64\n\
             .visible .entry decode_node({params}, .param .align 8 .b8 words[{WORDS}])\n{{\n\tret;\n}}\n\0"
        );
        let (_module, function) = driver::load_module(context, source.as_bytes(), "decode_node").unwrap();
        let bindings = [0x7000_0000_0000u64, 0x7100_0000_0000u64];
        let pointers = (0..2).map(|_| std::cell::RefCell::new(vec![0u64; POINTERS])).collect::<Vec<_>>();
        let words = std::cell::RefCell::new(vec![0u8; WORDS]);
        for (binding, pointers) in bindings.iter().zip(&pointers) {
            for (index, pointer) in pointers.borrow_mut().iter_mut().enumerate() {
                *pointer = binding + 256 * index as u64;
            }
        }
        let node_params = |binding: usize, parameters: &mut Vec<*mut c_void>| {
            parameters.clear();
            let mut pointers = pointers[binding].borrow_mut();
            parameters.extend(pointers.iter_mut().map(|pointer| (pointer as *mut u64).cast::<c_void>()));
            parameters.push(words.borrow_mut().as_mut_ptr().cast());
            KernelNodeParams {
                function,
                grid: [1, 1, 1],
                block: [32, 1, 1],
                shared_bytes: 0,
                parameters: parameters.as_mut_ptr(),
                extra: std::ptr::null_mut(),
                kernel: std::ptr::null_mut(),
                context: std::ptr::null_mut(),
            }
        };
        let mut parameters = Vec::new();
        let graph_of = |binding: usize, parameters: &mut Vec<*mut c_void>| {
            let mut graph = driver::Graph::new(context).unwrap();
            for index in 0..NODES {
                let after: &[usize] = if index == 0 { &[] } else { &[index - 1] };
                graph.push_kernel(&node_params(binding, parameters), Edge::Complete, after).unwrap();
            }
            let exec = graph.instantiate().unwrap();
            exec.upload(stream).unwrap();
            (graph, exec)
        };
        let (graph, exec) = graph_of(0, &mut parameters);
        let (_other_graph, other) = graph_of(1, &mut parameters);
        let timed = |each: &mut dyn FnMut(usize)| {
            each(0);
            stream.synchronize().unwrap();
            let start = std::time::Instant::now();
            for launch in 0..LAUNCHES {
                each(launch);
            }
            let host = start.elapsed() / LAUNCHES as u32;
            stream.synchronize().unwrap();
            (host, start.elapsed() / LAUNCHES as u32)
        };
        let unchanged = timed(&mut |_| exec.launch(stream).unwrap());
        let mut set = std::time::Duration::ZERO;
        let updated = timed(&mut |launch| {
            let start = std::time::Instant::now();
            for node in 0..NODES {
                exec.set_kernel(graph.node(node), &node_params(launch % 2, &mut parameters)).unwrap();
            }
            set += start.elapsed();
            exec.launch(stream).unwrap();
        });
        let instances = timed(&mut |launch| [&exec, &other][launch % 2].launch(stream).unwrap());
        eprintln!(
            "probe {NODES} nodes of {POINTERS} pointers and {WORDS} bytes: launch unchanged {:?} host, {:?} with the device; \
             every node updated then launched {:?} host ({:?} of it setting, {:?} a node), {:?} with the device; \
             two instances in turn {:?} host, {:?} with the device",
            unchanged.0, unchanged.1, updated.0, set / (LAUNCHES as u32 + 1), set / ((LAUNCHES as u32 + 1) * NODES as u32), updated.1,
            instances.0, instances.1
        );
    }
}
