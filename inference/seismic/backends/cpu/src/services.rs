//! CPU execution-service vocabulary, exhaustive closed-op demand, and the
//! fixed primitive probe suite acquired after the production worker pool exists.

use crate::Cpu;
use seismic_compiler::executable::DeviceService;
use crate::workers::{LaunchFrame, TeamBarrier, Workers};
use seismic_compiler::errors::TargetError;
use seismic_estimator::{
    AnalyticalService, CoreService, CompositionQualificationCase, CompositionQualificationParts, ConcreteExecutionDemand,
    DemandMode, DemandScope, DurationInterval, ExecutionDemand, FactProvenance, MeasurementBatch,
    MeasurementSeries, ResourceTopology, ServiceAccuracyClass, ServiceClassId,
    ServiceCorrelationId, ServiceCurve, ServiceCurveRegime, ServiceDefinition,
    ServiceQualificationDomain,
};
use seismic_ir::kernel::ops::{ClosedOpView, ValueType};
use seismic_ir::physical_target::KernelEmissionLayout;
use seismic_lang::expr::{ExprArena, NatExpr};
use seismic_lang::types::DType;
use sha2::{Digest, Sha256};
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

seismic_estimator::analytical_services! {
    pub enum CpuService {
        Control => "cpu.control",
        Integer => "cpu.integer",
        F32Strict => "cpu.f32.strict",
        NarrowStrict => "cpu.narrow.strict",
        ApproximateMath => "cpu.approximate-math",
        Memory => "cpu.memory",
        Representation => "cpu.representation",
        Atomic => "cpu.atomic",
        Barrier => "cpu.barrier",
    }
}

#[derive(Clone, Copy)]
enum CpuProbeKind {
    Control,
    Integer,
    Float,
    Memory,
    Atomic,
    Barrier,
    /// A reached allocation instance's backing, as the executor acquires
    /// one: the device service's allocation, released after. Host only.
    Allocation,
    /// A publication as the schedule's execution records it. Host only.
    Publication,
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

/// The probe of every core service, each measuring what this backend does
/// for it.
fn core_probes() -> impl Iterator<Item = CpuServiceProbe> {
    <CoreService as AnalyticalService>::ALL.iter().map(|service| {
        let kind = match service {
            CoreService::Submission => CpuProbeKind::Control,
            CoreService::Copy
            | CoreService::Fill
            | CoreService::ScalarRead
            | CoreService::DataCheck => CpuProbeKind::Memory,
            CoreService::ScalarMove => CpuProbeKind::Integer,
            CoreService::AllocationInstance => CpuProbeKind::Allocation,
            CoreService::TensorPublication => CpuProbeKind::Publication,
        };
        probe(
            service.class(),
            kind,
            CpuProbePath::Core,
            ServiceAccuracyClass::MemoryOrTransfer,
        )
    })
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

fn demand(class: CpuService, units: NatExpr, mode: DemandMode) -> ExecutionDemand<CpuService> {
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

fn arithmetic_class(ty: ValueType) -> CpuService {
    use CpuService::{
        Control as CONTROL, F32Strict as F32_STRICT, Integer as INTEGER,
        NarrowStrict as NARROW_STRICT,
    };
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
        ValueType::Scalar(DType::I32 | DType::U32 | DType::U64 | DType::Bool)
        | ValueType::Vector {
            dtype: DType::I32 | DType::U32 | DType::U64 | DType::Bool,
            ..
        }
        | ValueType::Index
        | ValueType::Bool => INTEGER,
        ValueType::Opaque { .. } => CONTROL,
    }
}

pub(crate) fn operation_cost(
    arena: &mut ExprArena,
    _kernel: &seismic_ir::kernel::Kernel<Cpu>,
    _emission: &KernelEmissionLayout,
    op: ClosedOpView<'_, Cpu>,
) -> seismic_estimator::OperationCost<CpuService> {
    use CpuService::{
        ApproximateMath as APPROXIMATE_MATH, Atomic as ATOMIC, Barrier as BARRIER,
        Control as CONTROL, Integer as INTEGER, Memory as MEMORY, Representation as REPRESENTATION,
    };
    use DemandMode::{DependencyLatency as Latency, SaturatedCapacity as Capacity};
    let one = |arena: &mut ExprArena, class, mode| {
        seismic_estimator::OperationCost::one(demand(class, arena.nat(1), mode))
    };
    let one_demand = seismic_estimator::OperationCost::one;
    let cost = |first, rest| seismic_estimator::OperationCost::demands(first, rest);
    match op {
        ClosedOpView::ScalarBits { .. }
        | ClosedOpView::ScalarFromBits { .. }
        | ClosedOpView::ReadPlaneLane { .. } => one(arena, INTEGER, Latency),
        ClosedOpView::Constant { out, .. }
        | ClosedOpView::Unary { out, .. }
        | ClosedOpView::Binary { out, .. }
        | ClosedOpView::Bit { out, .. }
        | ClosedOpView::Fma { out, .. }
        | ClosedOpView::Cast { out, .. }
        | ClosedOpView::Bitcast { out, .. }
        | ClosedOpView::Select { out, .. } => one_demand(demand(
            arithmetic_class(out.ty),
            lanes(arena, out.ty),
            Latency,
        )),
        ClosedOpView::VectorFromLanes { out, .. }
        | ClosedOpView::VectorSplat { out, .. }
        | ClosedOpView::VectorUnary { out, .. }
        | ClosedOpView::VectorBinary { out, .. }
        | ClosedOpView::VectorBit { out, .. }
        | ClosedOpView::VectorFma { out, .. }
        | ClosedOpView::VectorCast { out, .. }
        | ClosedOpView::VectorLane { out, .. } => one_demand(demand(
            arithmetic_class(out.ty),
            lanes(arena, out.ty),
            Latency,
        )),
        ClosedOpView::VectorReduceAdd { out, vector } => {
            let reductions = match vector.ty {
                ValueType::Vector { lanes, .. } => u64::from(lanes.saturating_sub(1)),
                _ => unreachable!("typed vector reduction has a scalar operand"),
            };
            one_demand(demand(
                arithmetic_class(out.ty),
                arena.nat(reductions),
                Latency,
            ))
        }
        ClosedOpView::ApproximateMath { out, .. } => {
            one_demand(demand(APPROXIMATE_MATH, lanes(arena, out.ty), Latency))
        }
        ClosedOpView::Cmp { a, .. } => {
            one_demand(demand(arithmetic_class(a.ty), lanes(arena, a.ty), Latency))
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
        ClosedOpView::Read { .. } => one(arena, MEMORY, Capacity),
        ClosedOpView::ReadPlaneField { plane_info, .. } => {
            let first = demand(MEMORY, arena.nat(1), Capacity);
            if matches!(
                plane_info.encoding,
                seismic_lang::registry::PlaneEncoding::Dense(_)
            ) {
                one_demand(first)
            } else {
                cost(first, vec![demand(REPRESENTATION, arena.nat(1), Latency)])
            }
        }
        ClosedOpView::ReadPlane { .. } => {
            let units = arena.nat(1);
            cost(
                demand(MEMORY, units, Capacity),
                vec![demand(REPRESENTATION, units, Latency)],
            )
        }
        ClosedOpView::VectorRead { out, .. } => {
            one_demand(demand(MEMORY, lanes(arena, out.ty), Capacity))
        }
        ClosedOpView::Write { .. } => one(arena, MEMORY, Capacity),
        ClosedOpView::VectorWrite { value, .. } => {
            one_demand(demand(MEMORY, lanes(arena, value.ty), Capacity))
        }
        ClosedOpView::RepresentationConvertPacket { source, .. } => {
            let units = arena.nat(u64::from(source.geometry.layout.logical_group));
            cost(
                demand(MEMORY, units, Capacity),
                vec![demand(REPRESENTATION, units, Latency)],
            )
        }
        ClosedOpView::Atomic { .. } => one(arena, ATOMIC, Capacity),
        ClosedOpView::Barrier(_) => one(arena, BARRIER, Latency),
        ClosedOpView::Intrinsic { op, .. } => match *op {},
    }
}

const PROBE_SUITE_REVISION: &str = "cpu-calibrated-services-v3";
const OBSERVATIONS: usize = 9;
const MAX_ACQUISITION_ROUNDS: usize = 8;

pub(crate) fn probe_suite_revision() -> &'static str {
    PROBE_SUITE_REVISION
}

/// Runs the calibrated batched service suite after the production pool exists.
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
    services: &[ServiceDefinition],
) -> Result<CompositionQualificationParts, TargetError> {
    let integer = backend_probes()
        .into_iter()
        .find(|probe| probe.class == INTEGER)
        .expect("CPU integer probe registration is missing");
    let memory = backend_probes()
        .into_iter()
        .find(|probe| probe.class == MEMORY)
        .expect("CPU memory probe registration is missing");
    // One observation: the two capacity probes in turn, each of `iterations`
    // per worker.
    let compose = |workers: &mut Workers, iterations: u64| -> Result<u64, TargetError> {
        let started = Instant::now();
        run_probe(integer, ProbeShape::Capacity, workers, iterations)?;
        run_probe(memory, ProbeShape::Capacity, workers, iterations)?;
        Ok(u64::try_from(started.elapsed().as_nanos())
            .map_err(|_| {
                TargetError::UnsupportedDevice("CPU composition qualification duration exceeds u64".into())
            })?
            .max(1))
    };
    // The composition runs at the size its services were qualified at, the
    // smaller of their capacity probes' iterations per worker: within both
    // services' qualified domains, as composition qualification requires
    // (`assemble_execution`), and at least the size the smaller was qualified
    // to its band at.
    let worker_count = u64::try_from(workers.count())
        .map_err(|_| TargetError::UnsupportedDevice("CPU worker count exceeds u64::MAX".into()))?;
    // Each runs on the workers, so its qualified units are its iterations per
    // worker times the workers ([`probe_service`]): an exact multiple.
    let qualified = |class: ServiceClassId| {
        let units = services
            .iter()
            .find(|service| service.class == class)
            .expect("the CPU's integer and memory services are measured before their composition")
            .qualification
            .maximum_units;
        assert_eq!(units % worker_count, 0, "a worker service qualifies a whole count per worker");
        units / worker_count
    };
    let iterations = qualified(INTEGER).min(qualified(MEMORY));
    compose(workers, iterations)?;
    let mut observations = Vec::with_capacity(OBSERVATIONS);
    for _ in 0..OBSERVATIONS {
        observations.push(compose(workers, iterations)?);
    }
    observations.sort_unstable();
    let observed_ns = trimmed_interval(&observations, 1);
    let units = iterations
        .checked_mul(worker_count)
        .ok_or_else(|| {
            TargetError::UnsupportedDevice("CPU qualification workload exceeds u64".into())
        })?;
    let mut digest = Sha256::new();
    digest.update(b"cpu-qualified-compositions-v2");
    digest.update(iterations.to_le_bytes());
    for observation in &observations {
        digest.update(observation.to_le_bytes());
    }
    Ok(CompositionQualificationParts {
        suite_revision: "cpu-qualified-compositions-v2",
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
    let required_ns = seismic_estimator::required_observation_ns(
        timer_resolution.upper_numerator,
        relative_percent,
    );
    // Each shape is judged by what it spends beyond its own launch: the same
    // shape at zero iterations (a capacity launch spans every worker, a
    // dependency launch one), whose observation repeats it `units` times.
    // What exceeds the setup carries the setup's uncertainty too, which sizes
    // the work as the timer's does.
    let mut work = |shape: ProbeShape| -> Result<(Acquired, Acquired), TargetError> {
        let setup = acquire(
            probe.class,
            relative_percent,
            required_ns,
            0,
            |units| {
                time_run(|| {
                    for _ in 0..units {
                        run_probe(probe, shape, workers, 0)?;
                    }
                    Ok(())
                })
            },
            |observations, units| Some(trimmed_interval(observations, units)),
        )?;
        let setup_ns = setup.interval.upper_numerator.div_ceil(setup.interval.denominator);
        let setup_width_ns = (setup.interval.upper_numerator - setup.interval.lower_numerator)
            .div_ceil(setup.interval.denominator);
        let work = acquire(
            probe.class,
            relative_percent,
            required_ns.max(seismic_estimator::required_beyond_ns(setup_width_ns, relative_percent)),
            setup_ns,
            |units| time_run(|| run_probe(probe, shape, workers, units)),
            |observations, units| differential_interval(observations, setup.interval, units),
        )?;
        Ok((setup, work))
    };
    let (latency_setup, latency) = work(ProbeShape::Dependency)?;
    let (setup, capacity) = work(ProbeShape::Capacity)?;
    let capacity_units = match probe.path {
        CpuProbePath::Core => capacity.units,
        CpuProbePath::Workers => {
            capacity
                .units
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
        dependency_latency: latency.interval,
        saturated_capacity: ServiceCurve {
            setup: setup.interval,
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit: capacity.interval,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: vec![
                measurement_batch(
                    probe,
                    "host-monotonic-calibrated-dependency-setup-v4",
                    latency_setup.units,
                    timer_resolution,
                    latency_setup.observations,
                    latency_setup.acquisition_ns,
                    worker_count,
                ),
                measurement_batch(
                    probe,
                    "host-monotonic-calibrated-capacity-setup-v4",
                    setup.units,
                    timer_resolution,
                    setup.observations,
                    setup.acquisition_ns,
                    worker_count,
                ),
                measurement_batch(
                    probe,
                    "host-monotonic-calibrated-dependency-v4",
                    latency.units,
                    timer_resolution,
                    latency.observations,
                    latency.acquisition_ns,
                    worker_count,
                ),
                measurement_batch(
                    probe,
                    "host-monotonic-calibrated-capacity-v4",
                    capacity_units,
                    timer_resolution,
                    capacity.observations,
                    capacity.acquisition_ns,
                    worker_count,
                ),
            ]
            .into_boxed_slice(),
        },
    })
}

/// One probe shape's accepted round: its sorted observations, each of
/// `units`, and the interval they bound.
struct Acquired {
    units: u64,
    observations: Vec<u64>,
    interval: DurationInterval,
    acquisition_ns: u64,
}

/// Acquires one probe shape in observations of a unit count calibrated until
/// what an observation spends beyond `offset_ns` reaches `required_ns`; a
/// round whose `interval` is absent or does not qualify escalates the count
/// for the next by how far it missed (`escalate_observation_units`).
fn acquire(
    class: ServiceClassId,
    relative_percent: u64,
    required_ns: u64,
    offset_ns: u64,
    mut observe: impl FnMut(u64) -> Result<u64, TargetError>,
    interval: impl Fn(&[u64], u64) -> Option<DurationInterval>,
) -> Result<Acquired, TargetError> {
    let started = Instant::now();
    let mut units = seismic_estimator::calibrate_observation_units::<TargetError>(
        required_ns,
        |units| Ok(observe(units)?.saturating_sub(offset_ns)),
    )?;
    for _ in 0..MAX_ACQUISITION_ROUNDS {
        let round = Instant::now();
        observe(units)?;
        let mut observations = (0..OBSERVATIONS)
            .map(|_| observe(units))
            .collect::<Result<Vec<_>, _>>()?;
        observations.sort_unstable();
        let candidate = interval(&observations, units);
        if let Some(interval) = candidate.filter(|interval| qualified(*interval, relative_percent)) {
            return Ok(Acquired {
                units,
                observations,
                interval,
                acquisition_ns: elapsed_ns(started, "CPU probe acquisition")?,
            });
        }
        match seismic_estimator::escalate_observation_units(units, round.elapsed(), candidate, relative_percent) {
            Some(next) => units = next,
            None => break,
        }
    }
    Err(TargetError::DeviceUnavailable(format!(
        "CPU probe `{}` did not meet its {}% relative half-width budget after setup subtraction within {MAX_ACQUISITION_ROUNDS} rounds of at most {} ms",
        class.stable_name(),
        relative_percent,
        seismic_estimator::MAX_ROUND.as_millis()
    )))
}

/// The duration of one run of `run`, at least a nanosecond.
fn time_run(run: impl FnOnce() -> Result<(), TargetError>) -> Result<u64, TargetError> {
    let start = Instant::now();
    run()?;
    Ok(elapsed_ns(start, "CPU service probe duration")?.max(1))
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
    Dependency,
    Capacity,
}

fn trimmed_interval(observations: &[u64], denominator: u64) -> DurationInterval {
    DurationInterval::new(
        observations[1],
        observations[observations.len() - 2],
        denominator,
    )
}

/// The interval of what `observations` spend beyond `setup`, absent when
/// they do not exceed it.
fn differential_interval(
    observations: &[u64],
    setup: DurationInterval,
    denominator: u64,
) -> Option<DurationInterval> {
    let measured = trimmed_interval(observations, 1);
    let lower = measured
        .lower_numerator
        .saturating_mul(setup.denominator)
        .saturating_sub(setup.upper_numerator);
    let upper = measured
        .upper_numerator
        .saturating_mul(setup.denominator)
        .saturating_sub(setup.lower_numerator);
    (lower != 0).then(|| {
        DurationInterval::new(lower, upper, denominator.saturating_mul(setup.denominator))
    })
}

fn qualified(interval: DurationInterval, relative_percent: u64) -> bool {
    let width = u128::from(interval.upper_numerator - interval.lower_numerator);
    let sum = u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
    width * 100 <= sum * u128::from(relative_percent)
}

fn run_probe(
    probe: CpuServiceProbe,
    shape: ProbeShape,
    workers: &mut Workers,
    iterations: u64,
) -> Result<(), TargetError> {
    if matches!(probe.path, CpuProbePath::Core) {
        match probe.kind {
            CpuProbeKind::Allocation => {
                for _ in 0..iterations {
                    let instance = <crate::executor::Device as DeviceService<Cpu>>::allocate(
                        &crate::executor::Device,
                        seismic_estimator::ALLOCATION_INSTANCE_BYTES,
                        64,
                    )
                    .map_err(|error| {
                        TargetError::DeviceUnavailable(format!(
                            "CPU allocation-instance probe failed: {error:?}"
                        ))
                    })?;
                    drop(black_box(instance));
                }
            }
            CpuProbeKind::Publication => {
                let mut publication = seismic_compiler::executable::TensorPublicationProbe::new();
                for _ in 0..iterations {
                    publication.publish();
                }
            }
            kind => {
                black_box(run_primitive(kind, shape, iterations));
            }
        }
        return Ok(());
    }
    let barrier = matches!(probe.kind, CpuProbeKind::Barrier);
    let workgroups = match (barrier, shape) {
        (true, _) => 1,
        (false, ProbeShape::Capacity) => workers.count(),
        (false, ProbeShape::Dependency) => 1,
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
        CpuProbeKind::Allocation | CpuProbeKind::Publication => {
            panic!("a host-only core probe runs on no worker")
        }
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
        ProbeShape::Dependency => 1,
        ProbeShape::Capacity => 2,
    }
}

fn run_primitive_tags(kind: u64, shape: u64, iterations: u64) -> u64 {
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
