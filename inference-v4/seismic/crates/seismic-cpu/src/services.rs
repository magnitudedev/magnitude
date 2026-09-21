//! CPU execution-service vocabulary, exhaustive closed-op demand, and the
//! fixed primitive probe suite acquired after the production worker pool exists.

use crate::workers::{LaunchFrame, TeamBarrier, Workers};
use crate::Cpu;
use seismic_compiler::errors::TargetError;
use seismic_compiler::kernel::ops::{ClosedOpView, ValueType};
use seismic_compiler::target::{
    CompositionQualificationCase, CompositionQualificationParts, ConcreteExecutionDemand,
    DemandMode, DemandScope, DeviceContract, DurationInterval, ExecutionDemand, FactProvenance,
    KernelEmissionLayout, MeasurementBatch, MeasurementSeries, ResourceTopology,
    ServiceAccuracyClass, ServiceClassId, ServiceCorrelationId, ServiceCurve, ServiceCurveRegime,
    ServiceDefinition, ServiceQualificationDomain, SERVICE_COPY, SERVICE_DATA_CHECK, SERVICE_FILL,
    SERVICE_SCALAR_MOVE, SERVICE_SCALAR_READ, SERVICE_SUBMISSION,
};
use seismic_lang::expr::{ExprArena, NatExpr};
use seismic_lang::types::DType;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub(crate) const CONTROL: ServiceClassId = ServiceClassId::new("cpu.control");
pub(crate) const INTEGER: ServiceClassId = ServiceClassId::new("cpu.integer");
pub(crate) const F32_STRICT: ServiceClassId = ServiceClassId::new("cpu.f32.strict");
pub(crate) const NARROW_STRICT: ServiceClassId = ServiceClassId::new("cpu.narrow.strict");
pub(crate) const APPROXIMATE_MATH: ServiceClassId = ServiceClassId::new("cpu.approximate-math");
pub(crate) const MEMORY: ServiceClassId = ServiceClassId::new("cpu.memory");
pub(crate) const REPRESENTATION: ServiceClassId = ServiceClassId::new("cpu.representation");
pub(crate) const ATOMIC: ServiceClassId = ServiceClassId::new("cpu.atomic");
pub(crate) const BARRIER: ServiceClassId = ServiceClassId::new("cpu.barrier");

#[derive(Clone, Copy)]
enum CpuProbeKind {
    Control,
    Integer,
    Float,
    Memory,
    Atomic,
    Barrier,
}

#[derive(Clone, Copy)]
enum CpuProbePath {
    Core,
    Workers,
}

#[derive(Clone, Copy)]
struct CpuServiceProbe {
    class: ServiceClassId,
    kind: CpuProbeKind,
    path: CpuProbePath,
    accuracy: ServiceAccuracyClass,
}

fn backend_probes() -> [CpuServiceProbe; 9] {
    [
        probe(
            CONTROL,
            CpuProbeKind::Control,
            CpuProbePath::Workers,
            ServiceAccuracyClass::Compute,
        ),
        probe(
            INTEGER,
            CpuProbeKind::Integer,
            CpuProbePath::Workers,
            ServiceAccuracyClass::Compute,
        ),
        probe(
            F32_STRICT,
            CpuProbeKind::Float,
            CpuProbePath::Workers,
            ServiceAccuracyClass::Compute,
        ),
        probe(
            NARROW_STRICT,
            CpuProbeKind::Float,
            CpuProbePath::Workers,
            ServiceAccuracyClass::Compute,
        ),
        probe(
            APPROXIMATE_MATH,
            CpuProbeKind::Float,
            CpuProbePath::Workers,
            ServiceAccuracyClass::Compute,
        ),
        probe(
            MEMORY,
            CpuProbeKind::Memory,
            CpuProbePath::Workers,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
        probe(
            REPRESENTATION,
            CpuProbeKind::Integer,
            CpuProbePath::Workers,
            ServiceAccuracyClass::Compute,
        ),
        probe(
            ATOMIC,
            CpuProbeKind::Atomic,
            CpuProbePath::Workers,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
        probe(
            BARRIER,
            CpuProbeKind::Barrier,
            CpuProbePath::Workers,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
    ]
}

fn core_probes() -> [CpuServiceProbe; 6] {
    [
        probe(
            SERVICE_SUBMISSION,
            CpuProbeKind::Control,
            CpuProbePath::Core,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
        probe(
            SERVICE_COPY,
            CpuProbeKind::Memory,
            CpuProbePath::Core,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
        probe(
            SERVICE_FILL,
            CpuProbeKind::Memory,
            CpuProbePath::Core,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
        probe(
            SERVICE_SCALAR_READ,
            CpuProbeKind::Memory,
            CpuProbePath::Core,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
        probe(
            SERVICE_SCALAR_MOVE,
            CpuProbeKind::Integer,
            CpuProbePath::Core,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
        probe(
            SERVICE_DATA_CHECK,
            CpuProbeKind::Memory,
            CpuProbePath::Core,
            ServiceAccuracyClass::MemoryOrTransfer,
        ),
    ]
}

const fn probe(
    class: ServiceClassId,
    kind: CpuProbeKind,
    path: CpuProbePath,
    accuracy: ServiceAccuracyClass,
) -> CpuServiceProbe {
    CpuServiceProbe {
        class,
        kind,
        path,
        accuracy,
    }
}

pub(crate) fn required_service_classes() -> BTreeSet<ServiceClassId> {
    backend_probes()
        .into_iter()
        .map(|probe| probe.class)
        .collect()
}

fn demand(class: ServiceClassId, units: NatExpr, mode: DemandMode) -> ExecutionDemand {
    ExecutionDemand {
        class,
        units,
        mode,
        scope: DemandScope::PerParticipant,
    }
}

fn lanes(arena: &mut ExprArena, ty: ValueType) -> NatExpr {
    arena.nat(match ty {
        ValueType::Vector { lanes, .. } => u64::from(lanes),
        _ => 1,
    })
}

fn arithmetic_class(ty: ValueType) -> ServiceClassId {
    match ty {
        ValueType::Scalar(DType::F32)
        | ValueType::Vector {
            dtype: DType::F32, ..
        } => F32_STRICT,
        ValueType::Scalar(DType::F16 | DType::BF16)
        | ValueType::Vector {
            dtype: DType::F16 | DType::BF16,
            ..
        } => NARROW_STRICT,
        ValueType::Scalar(DType::I32 | DType::U32 | DType::Bool)
        | ValueType::Vector {
            dtype: DType::I32 | DType::U32 | DType::Bool,
            ..
        }
        | ValueType::Index
        | ValueType::Bool => INTEGER,
        ValueType::Opaque { .. } => CONTROL,
    }
}

pub(crate) fn execution_demand(
    _target: &DeviceContract<Cpu>,
    arena: &mut ExprArena,
    _kernel: &seismic_compiler::kernel::Kernel<Cpu>,
    _emission: &KernelEmissionLayout,
    op: ClosedOpView<'_, Cpu>,
) -> Vec<ExecutionDemand> {
    use DemandMode::{DependencyLatency as Latency, SaturatedCapacity as Capacity};
    let one = |arena: &mut ExprArena, class, mode| vec![demand(class, arena.nat(1), mode)];
    match op {
        ClosedOpView::Constant { out, .. }
        | ClosedOpView::Unary { out, .. }
        | ClosedOpView::Binary { out, .. }
        | ClosedOpView::Bit { out, .. }
        | ClosedOpView::Fma { out, .. }
        | ClosedOpView::Cast { out, .. }
        | ClosedOpView::Bitcast { out, .. }
        | ClosedOpView::Select { out, .. } => {
            vec![demand(
                arithmetic_class(out.ty),
                lanes(arena, out.ty),
                Latency,
            )]
        }
        ClosedOpView::VectorSplat { out, .. }
        | ClosedOpView::VectorUnary { out, .. }
        | ClosedOpView::VectorBinary { out, .. }
        | ClosedOpView::VectorBit { out, .. }
        | ClosedOpView::VectorFma { out, .. }
        | ClosedOpView::VectorCast { out, .. }
        | ClosedOpView::VectorLane { out, .. } => {
            vec![demand(
                arithmetic_class(out.ty),
                lanes(arena, out.ty),
                Latency,
            )]
        }
        ClosedOpView::VectorReduceAdd { out, vector } => {
            let reductions = match vector.ty {
                ValueType::Vector { lanes, .. } => u64::from(lanes.saturating_sub(1)),
                _ => unreachable!("typed vector reduction has a scalar operand"),
            };
            vec![demand(
                arithmetic_class(out.ty),
                arena.nat(reductions),
                Latency,
            )]
        }
        ClosedOpView::ApproximateMath { out, .. } => {
            vec![demand(APPROXIMATE_MATH, lanes(arena, out.ty), Latency)]
        }
        ClosedOpView::Cmp { a, .. } => {
            vec![demand(arithmetic_class(a.ty), lanes(arena, a.ty), Latency)]
        }
        ClosedOpView::Logic { .. } | ClosedOpView::Not { .. } => one(arena, INTEGER, Latency),
        ClosedOpView::Geometry { .. }
        | ClosedOpView::NatArg { .. }
        | ClosedOpView::ScalarArg { .. }
        | ClosedOpView::Extent { .. }
        | ClosedOpView::StoreSlot { .. }
        | ClosedOpView::Branch { .. }
        | ClosedOpView::Repeat { .. }
        | ClosedOpView::Yield { .. } => one(arena, CONTROL, Latency),
        ClosedOpView::Read { place, .. } => {
            let mut result = one(arena, MEMORY, Capacity);
            if matches!(
                place.geometry,
                seismic_compiler::target::ReadableRepresentationGeometry::Packed(_)
            ) {
                result.extend(one(arena, REPRESENTATION, Latency));
            }
            result
        }
        ClosedOpView::ReadPlane { .. } => {
            let units = arena.nat(1);
            vec![
                demand(MEMORY, units, Capacity),
                demand(REPRESENTATION, units, Latency),
            ]
        }
        ClosedOpView::VectorRead { out, place, .. } => {
            let units = lanes(arena, out.ty);
            let mut result = vec![demand(MEMORY, units, Capacity)];
            if matches!(
                place.geometry,
                seismic_compiler::target::ReadableRepresentationGeometry::Packed(_)
            ) {
                result.push(demand(REPRESENTATION, units, Latency));
            }
            result
        }
        ClosedOpView::Write { .. } => one(arena, MEMORY, Capacity),
        ClosedOpView::VectorWrite { value, .. } => {
            vec![demand(MEMORY, lanes(arena, value.ty), Capacity)]
        }
        ClosedOpView::RepresentationConvertPacket { source, .. } => {
            let units = arena.nat(u64::from(source.geometry.layout.logical_group));
            vec![
                demand(MEMORY, units, Capacity),
                demand(REPRESENTATION, units, Latency),
            ]
        }
        ClosedOpView::Atomic { .. } => one(arena, ATOMIC, Capacity),
        ClosedOpView::Barrier(_) => one(arena, BARRIER, Latency),
        ClosedOpView::Intrinsic { op, .. } => match *op {},
    }
}

const PROBE_SUITE_REVISION: &str = "cpu-fixed-services-v1";
const OBSERVATIONS: usize = 9;
const ITERATIONS: u64 = 65_536;
const MAX_ACQUISITION_ROUNDS: usize = 8;
const CORE_SETUP_BATCH: u64 = 4_096;
const WORKER_SETUP_BATCH: u64 = 32;
const QUALIFICATION_ITERATIONS: u64 = 131_071;

pub(crate) fn probe_suite_revision() -> &'static str {
    PROBE_SUITE_REVISION
}

/// Runs the fixed batched service suite after the production pool exists.
/// Measurement-control constants affect acquisition only and never enter a
/// duration expression. Every returned numeric service fact is observed on
/// this open rather than copied from a nominal hardware table.
pub(crate) fn probe_services(workers: &mut Workers) -> Result<Vec<ServiceDefinition>, TargetError> {
    let worker_count = u32::try_from(workers.count())
        .map_err(|_| TargetError::UnsupportedDevice("CPU worker count exceeds u32::MAX".into()))?;
    let timer_resolution = timer_resolution()?;
    core_probes()
        .into_iter()
        .chain(backend_probes())
        .map(|probe| probe_service(probe, workers, worker_count, timer_resolution))
        .collect()
}

pub(crate) fn qualify_compositions(
    workers: &mut Workers,
) -> Result<CompositionQualificationParts, TargetError> {
    let integer = backend_probes()
        .into_iter()
        .find(|probe| probe.class == INTEGER)
        .expect("CPU integer probe registration is missing");
    let memory = backend_probes()
        .into_iter()
        .find(|probe| probe.class == MEMORY)
        .expect("CPU memory probe registration is missing");
    let mut observations = Vec::with_capacity(OBSERVATIONS);
    for _ in 0..OBSERVATIONS {
        let started = Instant::now();
        run_probe(
            integer,
            ProbeShape::Capacity,
            workers,
            QUALIFICATION_ITERATIONS,
        )?;
        run_probe(
            memory,
            ProbeShape::Capacity,
            workers,
            QUALIFICATION_ITERATIONS,
        )?;
        observations.push(
            u64::try_from(started.elapsed().as_nanos())
                .map_err(|_| {
                    TargetError::UnsupportedDevice(
                        "CPU composition qualification duration exceeds u64".into(),
                    )
                })?
                .max(1),
        );
    }
    observations.sort_unstable();
    let observed_ns = trimmed_interval(&observations, 1);
    let worker_count = u64::try_from(workers.count())
        .map_err(|_| TargetError::UnsupportedDevice("CPU worker count exceeds u64::MAX".into()))?;
    let units = QUALIFICATION_ITERATIONS
        .checked_mul(worker_count)
        .ok_or_else(|| {
            TargetError::UnsupportedDevice("CPU qualification workload exceeds u64".into())
        })?;
    let mut digest = Sha256::new();
    digest.update(b"cpu-fixed-compositions-v1");
    digest.update(QUALIFICATION_ITERATIONS.to_le_bytes());
    for observation in &observations {
        digest.update(observation.to_le_bytes());
    }
    Ok(CompositionQualificationParts {
        suite_revision: "cpu-fixed-compositions-v1",
        cases: vec![CompositionQualificationCase {
            stable_name: "cpu.integer-memory-capacity",
            demands: vec![
                ConcreteExecutionDemand {
                    class: INTEGER,
                    units,
                    mode: DemandMode::SaturatedCapacity,
                },
                ConcreteExecutionDemand {
                    class: MEMORY,
                    units,
                    mode: DemandMode::SaturatedCapacity,
                },
            ],
            observed_ns,
        }],
        observations_digest: digest.finalize().into(),
    })
}

fn timer_resolution() -> Result<DurationInterval, TargetError> {
    for _ in 0..1_000_000 {
        let start = Instant::now();
        let elapsed = start.elapsed().as_nanos();
        if elapsed != 0 {
            let elapsed = u64::try_from(elapsed).map_err(|_| {
                TargetError::UnsupportedDevice(
                    "host monotonic timer exceeds u64 nanoseconds".into(),
                )
            })?;
            return Ok(DurationInterval::new(elapsed, elapsed, 1));
        }
    }
    Err(TargetError::UnsupportedDevice(
        "host monotonic timer has no observable resolution".into(),
    ))
}

fn probe_service(
    probe: CpuServiceProbe,
    workers: &mut Workers,
    worker_count: u32,
    timer_resolution: DurationInterval,
) -> Result<ServiceDefinition, TargetError> {
    let relative_percent = accuracy_percent(probe.accuracy);
    let setup_started = Instant::now();
    let setup = observe(probe, ProbeShape::Setup, workers, relative_percent)?;
    let setup_acquisition_ns = elapsed_ns(setup_started, "CPU setup-probe acquisition")?;
    let latency_started = Instant::now();
    let latency = observe(probe, ProbeShape::Dependency, workers, relative_percent)?;
    let latency_acquisition_ns = elapsed_ns(latency_started, "CPU latency-probe acquisition")?;
    let capacity_started = Instant::now();
    let capacity = observe(probe, ProbeShape::Capacity, workers, relative_percent)?;
    let capacity_acquisition_ns = elapsed_ns(capacity_started, "CPU capacity-probe acquisition")?;
    let setup_interval = trimmed_interval(&setup, setup_batch(probe.path));
    let dependency_latency = differential_interval(&latency, setup_interval, ITERATIONS)?;
    let capacity_per_wave = differential_interval(&capacity, setup_interval, ITERATIONS)?;
    qualify(probe.class, dependency_latency, relative_percent)?;
    qualify(probe.class, capacity_per_wave, relative_percent)?;
    let latency_units = ITERATIONS;
    let capacity_units = match probe.path {
        CpuProbePath::Core => ITERATIONS,
        CpuProbePath::Workers => {
            ITERATIONS
                .checked_mul(u64::from(worker_count))
                .ok_or_else(|| {
                    TargetError::UnsupportedDevice("CPU probe workload units exceed u64".into())
                })?
        }
    };
    Ok(ServiceDefinition {
        class: probe.class,
        correlation: ServiceCorrelationId::new(probe.class.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units: capacity_units,
            maximum_concurrent_uses: match probe.path {
                CpuProbePath::Core => 1,
                CpuProbePath::Workers => u64::from(worker_count),
            },
        },
        accuracy: probe.accuracy,
        topology: match probe.path {
            CpuProbePath::Core => ResourceTopology {
                resources: 1,
                max_concurrency: 1,
            },
            CpuProbePath::Workers => ResourceTopology {
                resources: worker_count,
                max_concurrency: 1,
            },
        },
        dependency_latency,
        saturated_capacity: ServiceCurve {
            setup: setup_interval,
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit: capacity_per_wave,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: vec![
                measurement_batch(
                    probe,
                    "host-monotonic-fixed-setup-v2",
                    setup_batch(probe.path),
                    timer_resolution,
                    setup,
                    setup_acquisition_ns,
                    worker_count,
                ),
                measurement_batch(
                    probe,
                    "host-monotonic-fixed-dependency-v2",
                    latency_units,
                    timer_resolution,
                    latency,
                    latency_acquisition_ns,
                    worker_count,
                ),
                measurement_batch(
                    probe,
                    "host-monotonic-fixed-capacity-v2",
                    capacity_units,
                    timer_resolution,
                    capacity,
                    capacity_acquisition_ns,
                    worker_count,
                ),
            ]
            .into_boxed_slice(),
        },
    })
}

fn elapsed_ns(started: Instant, what: &str) -> Result<u64, TargetError> {
    u64::try_from(started.elapsed().as_nanos())
        .map_err(|_| TargetError::UnsupportedDevice(format!("{what} exceeds u64")))
}

fn measurement_batch(
    probe: CpuServiceProbe,
    method: &'static str,
    workload_units: u64,
    timer_resolution_ns: DurationInterval,
    observations: Vec<u64>,
    acquisition_duration_ns: u64,
    worker_count: u32,
) -> MeasurementBatch {
    let mut digest = Sha256::new();
    digest.update(PROBE_SUITE_REVISION.as_bytes());
    digest.update(probe.class.stable_name().as_bytes());
    digest.update(method.as_bytes());
    digest.update(worker_count.to_le_bytes());
    for observation in &observations {
        digest.update(observation.to_le_bytes());
    }
    MeasurementBatch {
        probe: probe.class.stable_name(),
        method,
        timer_resolution_ns,
        series: MeasurementSeries::single(workload_units, observations),
        observations_digest: digest.finalize().into(),
        acquisition_duration_ns,
    }
}

const fn accuracy_percent(accuracy: ServiceAccuracyClass) -> u64 {
    match accuracy {
        ServiceAccuracyClass::Compute => 1,
        ServiceAccuracyClass::MemoryOrTransfer => 2,
    }
}

#[derive(Clone, Copy)]
enum ProbeShape {
    Setup,
    Dependency,
    Capacity,
}

fn observe(
    probe: CpuServiceProbe,
    shape: ProbeShape,
    workers: &mut Workers,
    relative_percent: u64,
) -> Result<Vec<u64>, TargetError> {
    let iterations = match shape {
        ProbeShape::Setup => 0,
        ProbeShape::Dependency | ProbeShape::Capacity => ITERATIONS,
    };
    for _ in 0..MAX_ACQUISITION_ROUNDS {
        run_probe(probe, shape, workers, 1_024)?;
        let mut observations = Vec::with_capacity(OBSERVATIONS);
        for _ in 0..OBSERVATIONS {
            let start = Instant::now();
            let repetitions = match shape {
                ProbeShape::Setup => setup_batch(probe.path),
                ProbeShape::Dependency | ProbeShape::Capacity => 1,
            };
            for _ in 0..repetitions {
                run_probe(probe, shape, workers, iterations)?;
            }
            observations.push(
                u64::try_from(start.elapsed().as_nanos())
                    .map_err(|_| {
                        TargetError::UnsupportedDevice(
                            "CPU service probe duration exceeds u64".into(),
                        )
                    })?
                    .max(1),
            );
        }
        observations.sort_unstable();
        let interval = trimmed_interval(&observations, 1);
        let width = u128::from(interval.upper_numerator - interval.lower_numerator);
        let sum = u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
        if width * 100 <= sum * u128::from(relative_percent) {
            return Ok(observations);
        }
    }
    Err(TargetError::DeviceUnavailable(format!(
        "CPU probe `{}` did not meet its {}% relative half-width budget",
        probe.class.stable_name(),
        relative_percent
    )))
}

const fn setup_batch(path: CpuProbePath) -> u64 {
    match path {
        CpuProbePath::Core => CORE_SETUP_BATCH,
        CpuProbePath::Workers => WORKER_SETUP_BATCH,
    }
}

fn trimmed_interval(observations: &[u64], denominator: u64) -> DurationInterval {
    DurationInterval::new(
        observations[1],
        observations[observations.len() - 2],
        denominator,
    )
}

fn differential_interval(
    observations: &[u64],
    setup: DurationInterval,
    denominator: u64,
) -> Result<DurationInterval, TargetError> {
    let measured = trimmed_interval(observations, 1);
    let lower = measured
        .lower_numerator
        .saturating_mul(setup.denominator)
        .saturating_sub(setup.upper_numerator);
    let upper = measured
        .upper_numerator
        .saturating_mul(setup.denominator)
        .saturating_sub(setup.lower_numerator);
    if lower == 0 {
        return Err(TargetError::DeviceUnavailable(
            "CPU service probe did not exceed its measured setup interval".into(),
        ));
    }
    Ok(DurationInterval::new(
        lower,
        upper,
        denominator.saturating_mul(setup.denominator),
    ))
}

fn qualify(
    class: ServiceClassId,
    interval: DurationInterval,
    relative_percent: u64,
) -> Result<(), TargetError> {
    let width = u128::from(interval.upper_numerator - interval.lower_numerator);
    let sum = u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
    if width * 100 > sum * u128::from(relative_percent) {
        return Err(TargetError::DeviceUnavailable(format!(
            "CPU probe `{}` exceeded its {}% relative half-width budget after setup subtraction",
            class.stable_name(),
            relative_percent
        )));
    }
    Ok(())
}

fn run_probe(
    probe: CpuServiceProbe,
    shape: ProbeShape,
    workers: &mut Workers,
    iterations: u64,
) -> Result<(), TargetError> {
    if matches!(probe.path, CpuProbePath::Core) {
        black_box(run_primitive(probe.kind, shape, iterations));
        return Ok(());
    }
    let barrier = matches!(probe.kind, CpuProbeKind::Barrier);
    let workgroups = match (barrier, shape) {
        (true, _) => 1,
        (false, ProbeShape::Capacity) => workers.count(),
        (false, ProbeShape::Setup | ProbeShape::Dependency) => 1,
    };
    let words = [
        iterations,
        probe_kind_tag(probe.kind),
        probe_shape_tag(shape),
    ];
    let participants = if barrier { workers.count() } else { 1 };
    let mut results = vec![0u64; workgroups.max(participants)];
    let frame = LaunchFrame {
        buffers: std::ptr::null(),
        words: words.as_ptr(),
        results: results.as_mut_ptr(),
    };
    workers
        .run(
            worker_probe,
            &frame,
            workgroups as u64,
            participants as u64,
            0,
            0,
            0,
        )
        .map_err(|error| {
            TargetError::DeviceUnavailable(format!("CPU fixed service probe failed: {error:?}"))
        })?;
    black_box(results);
    Ok(())
}

unsafe extern "C-unwind" fn worker_probe(
    frame: *const LaunchFrame,
    barrier: *const TeamBarrier,
    workgroup: u64,
    local: u64,
    _workgroup_scratch: *mut u8,
    _participant_scratch: *mut u8,
    _register_scratch: *mut u8,
) {
    let frame = &*frame;
    let iterations = *frame.words;
    let kind = *frame.words.add(1);
    let shape = *frame.words.add(2);
    let value = if kind == probe_kind_tag(CpuProbeKind::Barrier) {
        let mut value = 0;
        for _ in 0..iterations {
            value += u64::from(crate::workers::seismic_cpu_barrier(barrier) == 0);
        }
        value
    } else {
        run_primitive_tags(kind, shape, iterations)
    };
    let result = if kind == probe_kind_tag(CpuProbeKind::Barrier) {
        local
    } else {
        workgroup
    };
    *frame.results.add(result as usize) = value;
}

fn run_primitive(kind: CpuProbeKind, shape: ProbeShape, iterations: u64) -> u64 {
    run_primitive_tags(probe_kind_tag(kind), probe_shape_tag(shape), iterations)
}

const fn probe_kind_tag(kind: CpuProbeKind) -> u64 {
    match kind {
        CpuProbeKind::Memory => 0,
        CpuProbeKind::Atomic => 1,
        CpuProbeKind::Float => 2,
        CpuProbeKind::Integer => 3,
        CpuProbeKind::Control => 4,
        CpuProbeKind::Barrier => 5,
    }
}

const fn probe_shape_tag(shape: ProbeShape) -> u64 {
    match shape {
        ProbeShape::Setup => 0,
        ProbeShape::Dependency => 1,
        ProbeShape::Capacity => 2,
    }
}

fn run_primitive_tags(kind: u64, shape: u64, iterations: u64) -> u64 {
    if shape == 0 {
        return black_box(kind);
    }
    if kind == 0 {
        let mut words = [0u64; 256];
        let mut values = [1u64, 3, 5, 7];
        for index in 0..iterations {
            let chain = if shape == 1 { 0 } else { index as usize & 3 };
            let slot = if shape == 1 {
                values[0] as usize & 255
            } else {
                index as usize & 255
            };
            values[chain] = black_box(words[slot]).wrapping_add(values[chain]);
            words[slot] = black_box(values[chain]);
        }
        black_box(words);
        values.into_iter().fold(0, u64::wrapping_add)
    } else if kind == 1 {
        let values: [AtomicU64; 4] = std::array::from_fn(|_| AtomicU64::new(0));
        for index in 0..iterations {
            let chain = if shape == 1 { 0 } else { index as usize & 3 };
            black_box(values[chain].fetch_add(1, Ordering::SeqCst));
        }
        values
            .iter()
            .map(|value| value.load(Ordering::Relaxed))
            .fold(0, u64::wrapping_add)
    } else if kind == 2 {
        let mut values = [1.000_001f32, 1.000_003, 1.000_005, 1.000_007];
        for index in 0..iterations {
            let chain = if shape == 1 { 0 } else { index as usize & 3 };
            values[chain] = black_box(values[chain] * 1.000_000_1 + 0.000_000_1);
        }
        black_box(values);
        values
            .into_iter()
            .map(|value| u64::from(value.to_bits()))
            .fold(0, u64::wrapping_add)
    } else {
        let mut values = [1u64, 3, 5, 7];
        for index in 0..iterations {
            let chain = if shape == 1 { 0 } else { index as usize & 3 };
            values[chain] = black_box(values[chain].rotate_left(7) ^ index).wrapping_add(1);
        }
        black_box(values);
        values.into_iter().fold(0, u64::wrapping_add)
    }
}
