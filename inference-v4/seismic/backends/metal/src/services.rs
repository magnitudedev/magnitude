//! Metal execution-service vocabulary and the exhaustive demand transfer.
//!
//! These classes name the native service families selected by `render`, not
//! source-language spellings. Profile acquisition supplies one measured
//! definition for each class before a plan can be constructed.

use seismic_estimator::{
    CompositionQualificationCase, CompositionQualificationParts, ConcreteExecutionDemand,
    DemandMode, DurationInterval, FactProvenance, MeasurementBatch, MeasurementSeries,
    ResourceTopology, ServiceAccuracyClass, ServiceClassId, ServiceCorrelationId, ServiceCurve,
    ServiceCurveRegime, ServiceDefinition, ServiceQualificationDomain,
};
pub(crate) const SERVICE_SUBMISSION: ServiceClassId = ServiceClassId::new("core.submission");
pub(crate) const SERVICE_COPY: ServiceClassId = ServiceClassId::new("core.copy");
pub(crate) const SERVICE_FILL: ServiceClassId = ServiceClassId::new("core.fill");
pub(crate) const SERVICE_SCALAR_READ: ServiceClassId = ServiceClassId::new("core.scalar-read");
pub(crate) const SERVICE_SCALAR_MOVE: ServiceClassId = ServiceClassId::new("core.scalar-move");
pub(crate) const SERVICE_DATA_CHECK: ServiceClassId = ServiceClassId::new("core.data-check");
pub(crate) const CONTROL: ServiceClassId = ServiceClassId::new("metal.control");
const INTEGER: ServiceClassId = ServiceClassId::new("metal.integer");
pub(crate) const F32_ADD_SUB: ServiceClassId = ServiceClassId::new("metal.f32.add-sub");
const F32_MULTIPLY: ServiceClassId = ServiceClassId::new("metal.f32.multiply");
const F32_DIVIDE: ServiceClassId = ServiceClassId::new("metal.f32.divide");
const F32_REMAINDER: ServiceClassId = ServiceClassId::new("metal.f32.remainder");
const F32_MIN_MAX: ServiceClassId = ServiceClassId::new("metal.f32.min-max");
const F32_FMA: ServiceClassId = ServiceClassId::new("metal.f32.fma");
const F32_COMPARE: ServiceClassId = ServiceClassId::new("metal.f32.compare");
const F32_TO_INTEGER: ServiceClassId = ServiceClassId::new("metal.f32.to-integer");
const INTEGER_TO_F32: ServiceClassId = ServiceClassId::new("metal.integer.to-f32");
const F32_TO_F16: ServiceClassId = ServiceClassId::new("metal.f32.to-f16");
const F16_TO_F32: ServiceClassId = ServiceClassId::new("metal.f16.to-f32");
const F32_TO_BF16: ServiceClassId = ServiceClassId::new("metal.f32.to-bf16");
const F16_STRICT: ServiceClassId = ServiceClassId::new("metal.f16.strict");
const BF16_STRICT: ServiceClassId = ServiceClassId::new("metal.bf16.strict");
const APPROXIMATE_MATH: ServiceClassId = ServiceClassId::new("metal.approximate-math");
const GLOBAL_MEMORY: ServiceClassId = ServiceClassId::new("metal.global-memory");
const WORKGROUP_MEMORY: ServiceClassId = ServiceClassId::new("metal.workgroup-memory");
const REPRESENTATION: ServiceClassId = ServiceClassId::new("metal.representation");
const ATOMIC: ServiceClassId = ServiceClassId::new("metal.atomic");
const BARRIER: ServiceClassId = ServiceClassId::new("metal.barrier");
const SUBGROUP: ServiceClassId = ServiceClassId::new("metal.subgroup");
const MATRIX: ServiceClassId = ServiceClassId::new("metal.simdgroup-matrix");
use std::collections::BTreeSet;
use std::ffi::{c_int, c_void};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSActivityOptions, NSObjectProtocol, NSProcessInfo, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Fixed opened-device probe suite
// ---------------------------------------------------------------------------

const PROBE_METHOD: &str = "adjacent-count-v23-closed-helper-service-single-op-integer-selective-matched-contrast-fixed-three-round-pooled-operation-units-bounded-counterbalanced-steady-queue-paired-95pct-confidence/gpu-command-buffer-time";
const HOST_METHOD: &str = "adjacent-count-v5-bounded-confidence95/host-monotonic-time";
const THREAD_METHOD: &str =
    "adjacent-count-v4-fixed-eight-round-blocked-confidence95/thread-cpu-time";
const STOCHASTIC_HOST_REPEATS: usize = 64;
const GPU_REPEATS: usize = 64;
const MATCHED_GPU_REPEATS: usize = 32;
const MATRIX_GPU_REPEATS: usize = 2_048;
const MAX_HOST_ACQUISITION_ROUNDS: usize = 8;
const MAX_GPU_ACQUISITION_ROUNDS: usize = 3;
const GPU_CONFIDENCE_SIGMA: f64 = 1.96;
const GPU_WARMUP_DISPATCHES: usize = 8;
const SMALL_WORK: u32 = 65_537;
const LARGE_WORK: u32 = 262_145;
/// Operation-minus-baseline contrasts are deliberately isolated from the
/// shared loop scaffold, so their signal is much smaller than either raw
/// command duration. Amplify the fixed adjacent work domain—not the sample
/// count or acceptance threshold—so the retained contrast is resolvable.
const MATCHED_WORK_AMPLIFICATION: u32 = 8;

const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

unsafe extern "C" {
    fn pthread_self() -> *mut c_void;
    fn pthread_get_qos_class_np(
        thread: *mut c_void,
        qos_class: *mut u32,
        relative_priority: *mut c_int,
    ) -> c_int;
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
    fn clock_gettime_nsec_np(clock_id: c_int) -> u64;
}

const CLOCK_MONOTONIC_RAW: c_int = 4;
const CLOCK_THREAD_CPUTIME_ID: c_int = 16;

struct ProfilingQos {
    previous_class: u32,
    previous_priority: c_int,
}

struct ProfilingActivity {
    process: Retained<NSProcessInfo>,
    token: Retained<ProtocolObject<dyn NSObjectProtocol>>,
}

impl ProfilingActivity {
    fn enter() -> Self {
        let process = NSProcessInfo::processInfo();
        let reason = NSString::from_str("Seismic fixed Metal profile acquisition");
        let token =
            process.beginActivityWithOptions_reason(NSActivityOptions::UserInteractive, &reason);
        Self { process, token }
    }
}

impl Drop for ProfilingActivity {
    fn drop(&mut self) {
        unsafe { self.process.endActivity(&self.token) }
    }
}

impl ProfilingQos {
    fn enter() -> Result<Self, seismic_compiler::errors::TargetError> {
        let mut previous_class = 0;
        let mut previous_priority = 0;
        let get_status = unsafe {
            pthread_get_qos_class_np(pthread_self(), &mut previous_class, &mut previous_priority)
        };
        if get_status != 0 {
            return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
                format!("Metal profiler could not inspect thread QoS (status {get_status})"),
            ));
        }
        let status = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
        if status != 0 {
            return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
                format!("Metal profiler could not enter user-interactive QoS (status {status})"),
            ));
        }
        Ok(Self {
            previous_class,
            previous_priority,
        })
    }
}

impl Drop for ProfilingQos {
    fn drop(&mut self) {
        unsafe {
            pthread_set_qos_class_self_np(self.previous_class, self.previous_priority);
        }
    }
}

#[derive(Clone)]
struct Observation {
    latency_small: Vec<u64>,
    latency_large: Vec<u64>,
    capacity_small: Vec<u64>,
    capacity_large: Vec<u64>,
    dependency_interval: DurationInterval,
    capacity_interval: DurationInterval,
    timer_resolution: DurationInterval,
    acquisition_duration_ns: u64,
    dependency_lanes: u64,
    capacity_lanes: u64,
    small_units: u64,
    large_units: u64,
    work_delta: u64,
    iteration_delta: u64,
}

#[derive(Clone)]
struct MatchedObservation {
    operation_latency_small: Vec<u64>,
    operation_latency_large: Vec<u64>,
    baseline_latency_small: Vec<u64>,
    baseline_latency_large: Vec<u64>,
    operation_capacity_small: Vec<u64>,
    operation_capacity_large: Vec<u64>,
    baseline_capacity_small: Vec<u64>,
    baseline_capacity_large: Vec<u64>,
    dependency_interval: DurationInterval,
    capacity_interval: DurationInterval,
    timer_resolution: DurationInterval,
    acquisition_duration_ns: u64,
    dependency_lanes: u64,
    capacity_lanes: u64,
    small_units: u64,
    large_units: u64,
    work_delta: u64,
}

fn measurement_batch(
    probe: &'static str,
    method: &'static str,
    workload_units: u64,
    timer_resolution_ns: DurationInterval,
    observations_ns: Vec<u64>,
    acquisition_duration_ns: u64,
) -> MeasurementBatch {
    let mut digest = Sha256::new();
    digest.update(b"measurement-series/single/v1");
    digest.update(workload_units.to_le_bytes());
    digest.update((observations_ns.len() as u64).to_le_bytes());
    for sample in &observations_ns {
        digest.update(sample.to_le_bytes());
    }
    MeasurementBatch {
        probe,
        method,
        timer_resolution_ns,
        series: MeasurementSeries::single(workload_units, observations_ns),
        observations_digest: digest.finalize().into(),
        acquisition_duration_ns,
    }
}

fn paired_measurement_batch(
    probe: &'static str,
    method: &'static str,
    small_workload_units: u64,
    large_workload_units: u64,
    timer_resolution_ns: DurationInterval,
    small_observations_ns: Vec<u64>,
    large_observations_ns: Vec<u64>,
    acquisition_duration_ns: u64,
) -> MeasurementBatch {
    let mut digest = Sha256::new();
    digest.update(b"measurement-series/paired-adjacent/v1");
    digest.update(small_workload_units.to_le_bytes());
    digest.update(large_workload_units.to_le_bytes());
    digest.update((small_observations_ns.len() as u64).to_le_bytes());
    for (small, large) in small_observations_ns.iter().zip(&large_observations_ns) {
        digest.update(small.to_le_bytes());
        digest.update(large.to_le_bytes());
    }
    MeasurementBatch {
        probe,
        method,
        timer_resolution_ns,
        series: MeasurementSeries::paired_adjacent(
            small_workload_units,
            large_workload_units,
            small_observations_ns,
            large_observations_ns,
        ),
        observations_digest: digest.finalize().into(),
        acquisition_duration_ns,
    }
}

pub(crate) struct AcquisitionMetrics {
    pub probe_build_ns: u64,
    pub probe_execution_ns: u64,
}

pub(crate) struct Acquisition {
    pub services: Vec<ServiceDefinition>,
    pub metrics: AcquisitionMetrics,
    pub composition: CompositionQualificationParts,
}

/// Acquires definitions only for the target-dependent closed service set.
/// One fixed library is compiled once with production compile options; every
/// GPU observation is read from `GPUStartTime`/`GPUEndTime` after completion.
pub(crate) fn acquire(
    device: &crate::device::MetalDevice,
    facts: &crate::facts::MetalFacts,
    required: &BTreeSet<ServiceClassId>,
) -> Result<Acquisition, seismic_compiler::errors::TargetError> {
    let _profiling_activity = ProfilingActivity::enter();
    let _profiling_qos = ProfilingQos::enter()?;
    let build_started = Instant::now();
    let source = probe_source(required.contains(&BF16_STRICT), required.contains(&MATRIX));
    let source = NSString::from_str(&source);
    let library = device
        .handle()
        .raw()
        .newLibraryWithSource_options_error(
            &source,
            Some(&crate::profile::compile_options(facts.language)),
        )
        .map_err(|error| {
            seismic_compiler::errors::TargetError::UnsupportedToolchain(format!(
                "Metal fixed service-probe library failed to compile: {}",
                error.localizedDescription()
            ))
        })?;
    let mut pipelines = Vec::with_capacity(required.len());
    for class in required {
        let function_name = probe_function(*class).unwrap_or_else(|| {
            panic!(
                "Metal service registry lacks a fixed probe for `{}`",
                class.stable_name()
            )
        });
        let function = library
            .newFunctionWithName(&NSString::from_str(function_name))
            .ok_or_else(|| {
                seismic_compiler::errors::TargetError::UnsupportedToolchain(format!(
                    "compiled Metal probe library omitted `{function_name}`"
                ))
            })?;
        let pipeline = device
            .handle()
            .raw()
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| {
                seismic_compiler::errors::TargetError::UnsupportedToolchain(format!(
                    "Metal could not create service probe `{}`: {}",
                    class.stable_name(),
                    error.localizedDescription()
                ))
            })?;
        let baseline_pipeline = matched_baseline_function(*class)
            .map(|baseline_name| {
                let function = library
                    .newFunctionWithName(&NSString::from_str(baseline_name))
                    .ok_or_else(|| {
                        seismic_compiler::errors::TargetError::UnsupportedToolchain(format!(
                            "compiled Metal probe library omitted `{baseline_name}`"
                        ))
                    })?;
                device
                    .handle()
                    .raw()
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|error| {
                        seismic_compiler::errors::TargetError::UnsupportedToolchain(format!(
                            "Metal could not create matched baseline `{baseline_name}`: {}",
                            error.localizedDescription()
                        ))
                    })
            })
            .transpose()?;
        pipelines.push((*class, pipeline, baseline_pipeline));
    }
    let composition_function = library
        .newFunctionWithName(&NSString::from_str("seismic_probe_composition"))
        .ok_or_else(|| {
            seismic_compiler::errors::TargetError::UnsupportedToolchain(
                "compiled Metal probe library omitted `seismic_probe_composition`".into(),
            )
        })?;
    let f32_anchor_function = library
        .newFunctionWithName(&NSString::from_str("seismic_probe_f32_anchor"))
        .ok_or_else(|| {
            seismic_compiler::errors::TargetError::UnsupportedToolchain(
                "compiled Metal probe library omitted `seismic_probe_f32_anchor`".into(),
            )
        })?;
    let f32_anchor_pipeline = device
        .handle()
        .raw()
        .newComputePipelineStateWithFunction_error(&f32_anchor_function)
        .map_err(|error| {
            seismic_compiler::errors::TargetError::UnsupportedToolchain(format!(
                "Metal could not create F32 harness-anchor probe: {}",
                error.localizedDescription()
            ))
        })?;
    let composition_pipeline = device
        .handle()
        .raw()
        .newComputePipelineStateWithFunction_error(&composition_function)
        .map_err(|error| {
            seismic_compiler::errors::TargetError::UnsupportedToolchain(format!(
                "Metal could not create held-out composition probe: {}",
                error.localizedDescription()
            ))
        })?;
    let probe_build_ns = build_started.elapsed().as_nanos() as u64;
    let execution_started = Instant::now();
    let buffer = device.allocate_bytes(4 * 1024 * 1024).map_err(|error| {
        seismic_compiler::errors::TargetError::DeviceUnavailable(error.to_string())
    })?;
    buffer.write_bytes(0, &vec![0u8; 4 * 1024 * 1024]);

    let mut definitions = core_services(device, &buffer)?;
    let mut f32_observation = None;
    for (class, pipeline, baseline_pipeline) in pipelines {
        let collective = matches!(class, BARRIER | SUBGROUP | MATRIX);
        let compute = !matches!(class, GLOBAL_MEMORY | WORKGROUP_MEMORY | ATOMIC | BARRIER);
        let units_per_iteration = if class == F32_ADD_SUB { 3 } else { 1 };
        if let Some(baseline_pipeline) = baseline_pipeline {
            let observation =
                observe_matched_gpu_service(device, &pipeline, &baseline_pipeline, &buffer, class)?;
            definitions.push(measured_f32_matched_definition(class, observation)?);
            continue;
        }
        let observation = observe_gpu_service(
            device,
            &pipeline,
            &buffer,
            collective,
            class,
            compute,
            units_per_iteration,
        )?;
        if class == F32_ADD_SUB {
            f32_observation = Some(observation);
        } else {
            definitions.push(measured_definition(
                class,
                observation,
                compute,
                function_name_for_provenance(class),
                PROBE_METHOD,
            )?);
        }
    }
    let f32_observation = f32_observation
        .as_ref()
        .expect("the required Metal service set omitted strict F32 arithmetic");
    let f32_anchor = observe_gpu_service(
        device,
        &f32_anchor_pipeline,
        &buffer,
        false,
        F32_ADD_SUB,
        true,
        1,
    )?;
    let f32_definition = measured_f32_definition(&f32_anchor, f32_observation)?;
    definitions.push(f32_definition);
    let composition_observation = observe_gpu_service(
        device,
        &composition_pipeline,
        &buffer,
        false,
        F32_ADD_SUB,
        true,
        2,
    )?;
    let composition = qualify_composition(
        &definitions,
        &composition_observation,
        &f32_anchor,
        f32_observation,
    )?;
    Ok(Acquisition {
        services: definitions,
        metrics: AcquisitionMetrics {
            probe_build_ns,
            probe_execution_ns: execution_started.elapsed().as_nanos() as u64,
        },
        composition,
    })
}

fn probe_function(class: ServiceClassId) -> Option<&'static str> {
    Some(if class == CONTROL {
        "seismic_probe_control"
    } else if class == INTEGER {
        "seismic_probe_integer"
    } else if class == F32_ADD_SUB {
        "seismic_probe_f32"
    } else if class == F32_MULTIPLY {
        "seismic_probe_f32_multiply"
    } else if class == F32_DIVIDE {
        "seismic_probe_f32_divide"
    } else if class == F32_REMAINDER {
        "seismic_probe_f32_remainder"
    } else if class == F32_MIN_MAX {
        "seismic_probe_f32_minmax"
    } else if class == F32_FMA {
        "seismic_probe_f32_fma"
    } else if class == F32_COMPARE {
        "seismic_probe_f32_compare"
    } else if class == F32_TO_INTEGER {
        "seismic_probe_f32_to_integer"
    } else if class == INTEGER_TO_F32 {
        "seismic_probe_integer_to_f32"
    } else if class == F32_TO_F16 {
        "seismic_probe_f32_to_f16"
    } else if class == F16_TO_F32 {
        "seismic_probe_f16_to_f32"
    } else if class == F32_TO_BF16 {
        "seismic_probe_f32_to_bf16"
    } else if class == F16_STRICT {
        "seismic_probe_f16"
    } else if class == BF16_STRICT {
        "seismic_probe_bf16"
    } else if class == APPROXIMATE_MATH {
        "seismic_probe_approx"
    } else if class == GLOBAL_MEMORY {
        "seismic_probe_global"
    } else if class == WORKGROUP_MEMORY {
        "seismic_probe_workgroup"
    } else if class == REPRESENTATION {
        "seismic_probe_representation"
    } else if class == ATOMIC {
        "seismic_probe_atomic"
    } else if class == BARRIER {
        "seismic_probe_barrier"
    } else if class == SUBGROUP {
        "seismic_probe_subgroup"
    } else if class == MATRIX {
        "seismic_probe_matrix"
    } else {
        return None;
    })
}

fn matched_baseline_function(class: ServiceClassId) -> Option<&'static str> {
    Some(if class == F32_MULTIPLY {
        "seismic_baseline_f32_multiply"
    } else if class == F32_DIVIDE {
        "seismic_baseline_f32_divide"
    } else if class == F32_REMAINDER {
        "seismic_baseline_f32_remainder"
    } else if class == F32_MIN_MAX {
        "seismic_baseline_f32_minmax"
    } else if class == F32_FMA {
        "seismic_baseline_f32_fma"
    } else if class == F32_TO_INTEGER {
        "seismic_baseline_f32_to_integer"
    } else if class == INTEGER_TO_F32 {
        "seismic_baseline_integer_to_f32"
    } else if class == F32_TO_F16 {
        "seismic_baseline_f32_to_f16"
    } else if class == F16_TO_F32 {
        "seismic_baseline_f16_to_f32"
    } else if class == F32_TO_BF16 {
        "seismic_baseline_f32_to_bf16"
    } else {
        return None;
    })
}

fn function_name_for_provenance(class: ServiceClassId) -> &'static str {
    probe_function(class).expect("required Metal service class has no probe function")
}

fn observe_gpu_service(
    device: &crate::device::MetalDevice,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffer: &crate::device::MetalBuffer,
    collective: bool,
    class: ServiceClassId,
    compute: bool,
    units_per_iteration: u64,
) -> Result<Observation, seismic_compiler::errors::TargetError> {
    let acquisition_started = Instant::now();
    let group = pipeline.threadExecutionWidth().max(1);
    let latency_threads = if collective { group } else { 1 };
    let capacity_threads = group * 128;
    // The adjacent counts already amplify every saturated round across the
    // complete lane population. Confidence acquisition needs independent
    // complete observations, not a second service-specific amplification
    // factor that turns profiling into the workload being measured.
    let (small_work, large_work) = (SMALL_WORK, LARGE_WORK);
    let repeats = if class == MATRIX {
        MATRIX_GPU_REPEATS
    } else {
        GPU_REPEATS
    };
    let percent = if compute { 1 } else { 2 };
    let small_units = u64::from(small_work)
        .checked_mul(units_per_iteration)
        .expect("fixed Metal probe workload exceeds u64");
    let large_units = u64::from(large_work)
        .checked_mul(units_per_iteration)
        .expect("fixed Metal probe workload exceeds u64");
    let work_delta = large_units - small_units;
    let iteration_delta = u64::from(large_work - small_work);
    let total_repeats = repeats * MAX_GPU_ACQUISITION_ROUNDS;
    let mut latency_small = Vec::with_capacity(total_repeats);
    let mut latency_large = Vec::with_capacity(total_repeats);
    let mut capacity_small = Vec::with_capacity(total_repeats);
    let mut capacity_large = Vec::with_capacity(total_repeats);
    for _ in 0..MAX_GPU_ACQUISITION_ROUNDS {
        // Submit each round as one continuous queue batch. Waiting after every
        // dispatch drains the queue and repeatedly crosses GPU power/clock
        // states, which is not a steady service observation.
        let warmup = vec![(large_work, capacity_threads); GPU_WARMUP_DISPATCHES];
        gpu_elapsed_batch(device, pipeline, buffer, &warmup)?;
        let mut requests = Vec::with_capacity(repeats * 4);
        for observation in 0..repeats {
            if observation % 2 == 0 {
                requests.extend_from_slice(&[
                    (small_work, latency_threads),
                    (large_work, latency_threads),
                    (small_work, capacity_threads),
                    (large_work, capacity_threads),
                ]);
            } else {
                requests.extend_from_slice(&[
                    (large_work, latency_threads),
                    (small_work, latency_threads),
                    (large_work, capacity_threads),
                    (small_work, capacity_threads),
                ]);
            }
        }
        let elapsed = gpu_elapsed_batch(device, pipeline, buffer, &requests)?;
        for (observation, sample) in elapsed.chunks_exact(4).enumerate() {
            let (latency_small_ns, latency_large_ns, capacity_small_ns, capacity_large_ns) =
                if observation % 2 == 0 {
                    (sample[0], sample[1], sample[2], sample[3])
                } else {
                    (sample[1], sample[0], sample[3], sample[2])
                };
            latency_small.push(latency_small_ns);
            latency_large.push(latency_large_ns);
            capacity_small.push(capacity_small_ns);
            capacity_large.push(capacity_large_ns);
        }
    }
    let timer_quantum = latency_small
        .iter()
        .chain(&latency_large)
        .chain(&capacity_small)
        .chain(&capacity_large)
        .copied()
        .filter(|value| *value > 0)
        .reduce(greatest_common_divisor)
        .ok_or_else(|| {
            seismic_compiler::errors::TargetError::DeviceUnavailable(
                "Metal GPU timing returned no positive interval".into(),
            )
        })?;
    let timer_resolution = DurationInterval::new(0, timer_quantum, 1);
    let uncertainty = timer_quantum.saturating_mul(2);
    let dependency_interval = paired_difference_confidence_interval(
        &latency_small,
        &latency_large,
        iteration_delta * latency_threads as u64,
        uncertainty,
    )?
    .ok_or_else(|| {
        seismic_compiler::errors::TargetError::DeviceUnavailable(format!(
            "service probe `{}` had a nonpositive adjacent-count mean across its fixed three-round acquisition",
            class.stable_name(),
        ))
    })?;
    let capacity_interval = paired_difference_confidence_interval(
        &capacity_small,
        &capacity_large,
        iteration_delta * capacity_threads as u64,
        uncertainty,
    )?
    .ok_or_else(|| {
        seismic_compiler::errors::TargetError::DeviceUnavailable(format!(
            "service probe `{}` had a nonpositive saturated adjacent-count mean across its fixed three-round acquisition",
            class.stable_name(),
        ))
    })?;
    qualify(class, dependency_interval, percent)?;
    qualify(class, capacity_interval, percent)?;
    Ok(Observation {
        latency_small,
        latency_large,
        capacity_small,
        capacity_large,
        dependency_interval,
        capacity_interval,
        timer_resolution,
        acquisition_duration_ns: acquisition_started.elapsed().as_nanos() as u64,
        dependency_lanes: latency_threads as u64,
        capacity_lanes: capacity_threads as u64,
        small_units,
        large_units,
        work_delta,
        iteration_delta,
    })
}

fn observe_matched_gpu_service(
    device: &crate::device::MetalDevice,
    operation: &ProtocolObject<dyn MTLComputePipelineState>,
    baseline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffer: &crate::device::MetalBuffer,
    class: ServiceClassId,
) -> Result<MatchedObservation, seismic_compiler::errors::TargetError> {
    let acquisition_started = Instant::now();
    let operation_group = operation.threadExecutionWidth().max(1);
    let baseline_group = baseline.threadExecutionWidth().max(1);
    if operation_group != baseline_group {
        return Err(seismic_compiler::errors::TargetError::UnsupportedToolchain(
            format!(
                "matched Metal probe `{}` has operation width {operation_group} but baseline width {baseline_group}",
                class.stable_name()
            ),
        ));
    }
    let latency_threads = 1usize;
    let capacity_threads = operation_group * 128;
    let small_work = SMALL_WORK
        .checked_mul(MATCHED_WORK_AMPLIFICATION)
        .expect("fixed matched Metal probe small work exceeds u32");
    let large_work = LARGE_WORK
        .checked_mul(MATCHED_WORK_AMPLIFICATION)
        .expect("fixed matched Metal probe large work exceeds u32");
    let small_units = u64::from(small_work);
    let large_units = u64::from(large_work);
    let work_delta = large_units - small_units;
    let iteration_delta = u64::from(large_work - small_work);
    let total_repeats = MATCHED_GPU_REPEATS * MAX_GPU_ACQUISITION_ROUNDS;
    let mut operation_latency_small = Vec::with_capacity(total_repeats);
    let mut operation_latency_large = Vec::with_capacity(total_repeats);
    let mut baseline_latency_small = Vec::with_capacity(total_repeats);
    let mut baseline_latency_large = Vec::with_capacity(total_repeats);
    let mut operation_capacity_small = Vec::with_capacity(total_repeats);
    let mut operation_capacity_large = Vec::with_capacity(total_repeats);
    let mut baseline_capacity_small = Vec::with_capacity(total_repeats);
    let mut baseline_capacity_large = Vec::with_capacity(total_repeats);
    for _ in 0..MAX_GPU_ACQUISITION_ROUNDS {
        let mut warmup = Vec::with_capacity(GPU_WARMUP_DISPATCHES * 2);
        for index in 0..GPU_WARMUP_DISPATCHES {
            if index % 2 == 0 {
                warmup.push((operation, large_work, capacity_threads));
                warmup.push((baseline, large_work, capacity_threads));
            } else {
                warmup.push((baseline, large_work, capacity_threads));
                warmup.push((operation, large_work, capacity_threads));
            }
        }
        gpu_elapsed_pipeline_batch(device, buffer, &warmup)?;

        let mut requests = Vec::with_capacity(MATCHED_GPU_REPEATS * 8);
        for observation in 0..MATCHED_GPU_REPEATS {
            if observation % 2 == 0 {
                requests.extend_from_slice(&[
                    (operation, small_work, latency_threads),
                    (baseline, small_work, latency_threads),
                    (operation, large_work, latency_threads),
                    (baseline, large_work, latency_threads),
                    (operation, small_work, capacity_threads),
                    (baseline, small_work, capacity_threads),
                    (operation, large_work, capacity_threads),
                    (baseline, large_work, capacity_threads),
                ]);
            } else {
                requests.extend_from_slice(&[
                    (baseline, large_work, latency_threads),
                    (operation, large_work, latency_threads),
                    (baseline, small_work, latency_threads),
                    (operation, small_work, latency_threads),
                    (baseline, large_work, capacity_threads),
                    (operation, large_work, capacity_threads),
                    (baseline, small_work, capacity_threads),
                    (operation, small_work, capacity_threads),
                ]);
            }
        }
        let elapsed = gpu_elapsed_pipeline_batch(device, buffer, &requests)?;
        for (observation, sample) in elapsed.chunks_exact(8).enumerate() {
            let values = if observation % 2 == 0 {
                [
                    sample[0], sample[2], sample[1], sample[3], sample[4], sample[6], sample[5],
                    sample[7],
                ]
            } else {
                [
                    sample[3], sample[1], sample[2], sample[0], sample[7], sample[5], sample[6],
                    sample[4],
                ]
            };
            operation_latency_small.push(values[0]);
            operation_latency_large.push(values[1]);
            baseline_latency_small.push(values[2]);
            baseline_latency_large.push(values[3]);
            operation_capacity_small.push(values[4]);
            operation_capacity_large.push(values[5]);
            baseline_capacity_small.push(values[6]);
            baseline_capacity_large.push(values[7]);
        }
    }
    let timer_quantum = operation_latency_small
        .iter()
        .chain(&operation_latency_large)
        .chain(&baseline_latency_small)
        .chain(&baseline_latency_large)
        .chain(&operation_capacity_small)
        .chain(&operation_capacity_large)
        .chain(&baseline_capacity_small)
        .chain(&baseline_capacity_large)
        .copied()
        .filter(|value| *value > 0)
        .reduce(greatest_common_divisor)
        .ok_or_else(|| {
            seismic_compiler::errors::TargetError::DeviceUnavailable(
                "Metal matched GPU timing returned no positive interval".into(),
            )
        })?;
    let timer_resolution = DurationInterval::new(0, timer_quantum, 1);
    let uncertainty = timer_quantum.saturating_mul(4);
    let dependency_interval = paired_matched_confidence_interval(
        &operation_latency_small,
        &operation_latency_large,
        &baseline_latency_small,
        &baseline_latency_large,
        iteration_delta * latency_threads as u64,
        uncertainty,
    )?
    .ok_or_else(|| {
        seismic_compiler::errors::TargetError::DeviceUnavailable(format!(
            "matched service probe `{}` had a nonpositive operation-minus-baseline mean across its fixed three-round acquisition",
            class.stable_name()
        ))
    })?;
    let capacity_interval = paired_matched_confidence_interval(
        &operation_capacity_small,
        &operation_capacity_large,
        &baseline_capacity_small,
        &baseline_capacity_large,
        iteration_delta * capacity_threads as u64,
        uncertainty,
    )?
    .ok_or_else(|| {
        seismic_compiler::errors::TargetError::DeviceUnavailable(format!(
            "matched service probe `{}` had a nonpositive saturated operation-minus-baseline mean across its fixed three-round acquisition",
            class.stable_name()
        ))
    })?;
    qualify(class, dependency_interval, 1)?;
    qualify(class, capacity_interval, 1)?;
    Ok(MatchedObservation {
        operation_latency_small,
        operation_latency_large,
        baseline_latency_small,
        baseline_latency_large,
        operation_capacity_small,
        operation_capacity_large,
        baseline_capacity_small,
        baseline_capacity_large,
        dependency_interval,
        capacity_interval,
        timer_resolution,
        acquisition_duration_ns: acquisition_started.elapsed().as_nanos() as u64,
        dependency_lanes: latency_threads as u64,
        capacity_lanes: capacity_threads as u64,
        small_units,
        large_units,
        work_delta,
    })
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn gpu_elapsed_batch(
    device: &crate::device::MetalDevice,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffer: &crate::device::MetalBuffer,
    requests: &[(u32, usize)],
) -> Result<Vec<u64>, seismic_compiler::errors::TargetError> {
    let requests = requests
        .iter()
        .map(|&(iterations, threads)| (pipeline, iterations, threads))
        .collect::<Vec<_>>();
    gpu_elapsed_pipeline_batch(device, buffer, &requests)
}

fn gpu_elapsed_pipeline_batch(
    device: &crate::device::MetalDevice,
    buffer: &crate::device::MetalBuffer,
    requests: &[(&ProtocolObject<dyn MTLComputePipelineState>, u32, usize)],
) -> Result<Vec<u64>, seismic_compiler::errors::TargetError> {
    let mut commands = Vec::with_capacity(requests.len());
    for &(pipeline, iterations, threads) in requests {
        let command = device.queue().commandBuffer().ok_or_else(|| {
            seismic_compiler::errors::TargetError::DeviceUnavailable(
                "Metal could not create a service-probe command buffer".into(),
            )
        })?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            seismic_compiler::errors::TargetError::DeviceUnavailable(
                "Metal could not create a service-probe encoder".into(),
            )
        })?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buffer.raw()), 0, 0);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(&iterations).cast(),
                std::mem::size_of::<u32>(),
                1,
            );
        }
        let group = pipeline.threadExecutionWidth().max(1);
        let groups = threads.div_ceil(group);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        commands.push(command);
    }
    let mut elapsed = Vec::with_capacity(commands.len());
    for command in commands {
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
                error.localizedDescription().to_string(),
            ));
        }
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        if !(start.is_finite() && end.is_finite() && end >= start && start > 0.0) {
            return Err(seismic_compiler::errors::TargetError::UnsupportedDriver(
                "MTLCommandBuffer GPUStartTime/GPUEndTime returned an invalid interval".into(),
            ));
        }
        elapsed.push(((end - start) * 1_000_000_000.0).ceil() as u64);
    }
    Ok(elapsed)
}

fn paired_linear_combination_confidence_interval(
    series: &[(&[u64], &[u64], i64)],
    denominator: u64,
    uncertainty_ns: u64,
) -> Result<DurationInterval, seismic_compiler::errors::TargetError> {
    let count = series.first().map_or(0, |(small, large, _)| {
        assert_eq!(small.len(), large.len());
        small.len()
    });
    if count < 2
        || denominator == 0
        || series
            .iter()
            .any(|(small, large, _)| small.len() != count || large.len() != count)
    {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "paired service contrast requires aligned observations and a nonzero domain".into(),
        ));
    }
    let samples = (0..count)
        .map(|index| {
            series.iter().fold(0.0, |sum, (small, large, coefficient)| {
                sum + (*coefficient as f64) * (large[index] as f64 - small[index] as f64)
            })
        })
        .collect::<Vec<_>>();
    let count = count as f64;
    let mean = samples.iter().sum::<f64>() / count;
    if !mean.is_finite() || mean <= 0.0 {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "paired service contrast has no positive finite mean".into(),
        ));
    }
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = *sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1.0);
    let half_width = GPU_CONFIDENCE_SIGMA * (variance / count).sqrt();
    let lower = (mean - half_width).floor().max(0.0) as u64;
    let upper = (mean + half_width).ceil() as u64;
    Ok(widen(
        DurationInterval::new(lower, upper, denominator),
        uncertainty_ns,
    ))
}

fn observation_batches(observation: Observation, probe: &'static str) -> Vec<MeasurementBatch> {
    vec![
        paired_measurement_batch(
            probe,
            PROBE_METHOD,
            observation.small_units * observation.dependency_lanes,
            observation.large_units * observation.dependency_lanes,
            observation.timer_resolution,
            observation.latency_small,
            observation.latency_large,
            observation.acquisition_duration_ns,
        ),
        paired_measurement_batch(
            probe,
            PROBE_METHOD,
            observation.small_units * observation.capacity_lanes,
            observation.large_units * observation.capacity_lanes,
            observation.timer_resolution,
            observation.capacity_small,
            observation.capacity_large,
            observation.acquisition_duration_ns,
        ),
    ]
}

fn measured_f32_definition(
    anchor: &Observation,
    three: &Observation,
) -> Result<ServiceDefinition, seismic_compiler::errors::TargetError> {
    let iteration_delta = three.iteration_delta;
    let uncertainty = anchor
        .timer_resolution
        .upper_numerator
        .max(three.timer_resolution.upper_numerator)
        .saturating_mul(4);
    let dependency_latency = paired_linear_combination_confidence_interval(
        &[
            (&three.latency_small, &three.latency_large, 1),
            (&anchor.latency_small, &anchor.latency_large, -1),
        ],
        iteration_delta * anchor.dependency_lanes * 2,
        uncertainty,
    )?;
    let saturated = paired_linear_combination_confidence_interval(
        &[
            (&three.capacity_small, &three.capacity_large, 1),
            (&anchor.capacity_small, &anchor.capacity_large, -1),
        ],
        iteration_delta * anchor.capacity_lanes * 2,
        uncertainty,
    )?;
    qualify(F32_ADD_SUB, dependency_latency, 1)?;
    qualify(F32_ADD_SUB, saturated, 1)?;
    let measured_capacity_units = three
        .work_delta
        .checked_mul(three.capacity_lanes)
        .ok_or_else(|| {
            seismic_compiler::errors::TargetError::UnsupportedDevice(
                "Metal F32 service-probe workload exceeds u64".into(),
            )
        })?;
    let maximum_units = measured_capacity_units.checked_mul(2).ok_or_else(|| {
        seismic_compiler::errors::TargetError::UnsupportedDevice(
            "Metal F32 qualified service domain exceeds u64".into(),
        )
    })?;
    let maximum_concurrent_uses = three.capacity_lanes.max(1);
    let mut batches = observation_batches(anchor.clone(), "seismic_probe_f32_anchor");
    batches.extend(observation_batches(three.clone(), "seismic_probe_f32"));
    Ok(ServiceDefinition {
        class: F32_ADD_SUB,
        correlation: ServiceCorrelationId::new(F32_ADD_SUB.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units,
            maximum_concurrent_uses,
        },
        accuracy: ServiceAccuracyClass::Compute,
        topology: ResourceTopology {
            resources: 1,
            max_concurrency: 1,
        },
        dependency_latency,
        saturated_capacity: ServiceCurve {
            setup: DurationInterval::new(0, 0, 1),
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit: saturated,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: batches.into_boxed_slice(),
        },
    })
}

fn measured_f32_matched_definition(
    class: ServiceClassId,
    observation: MatchedObservation,
) -> Result<ServiceDefinition, seismic_compiler::errors::TargetError> {
    let dependency_latency = observation.dependency_interval;
    let saturated = observation.capacity_interval;
    qualify(class, dependency_latency, 1)?;
    qualify(class, saturated, 1)?;
    let measured_capacity_units = observation
        .work_delta
        .checked_mul(observation.capacity_lanes)
        .ok_or_else(|| {
            seismic_compiler::errors::TargetError::UnsupportedDevice(
                "Metal F32 service-probe workload exceeds u64".into(),
            )
        })?;
    let maximum_units = measured_capacity_units.checked_mul(2).ok_or_else(|| {
        seismic_compiler::errors::TargetError::UnsupportedDevice(
            "Metal F32 qualified service domain exceeds u64".into(),
        )
    })?;
    let operation_probe = function_name_for_provenance(class);
    let baseline_probe = matched_baseline_function(class)
        .expect("matched F32 service class has no baseline function");
    let batches = vec![
        paired_measurement_batch(
            operation_probe,
            PROBE_METHOD,
            observation.small_units * observation.dependency_lanes,
            observation.large_units * observation.dependency_lanes,
            observation.timer_resolution,
            observation.operation_latency_small,
            observation.operation_latency_large,
            observation.acquisition_duration_ns,
        ),
        paired_measurement_batch(
            baseline_probe,
            PROBE_METHOD,
            observation.small_units * observation.dependency_lanes,
            observation.large_units * observation.dependency_lanes,
            observation.timer_resolution,
            observation.baseline_latency_small,
            observation.baseline_latency_large,
            observation.acquisition_duration_ns,
        ),
        paired_measurement_batch(
            operation_probe,
            PROBE_METHOD,
            observation.small_units * observation.capacity_lanes,
            observation.large_units * observation.capacity_lanes,
            observation.timer_resolution,
            observation.operation_capacity_small,
            observation.operation_capacity_large,
            observation.acquisition_duration_ns,
        ),
        paired_measurement_batch(
            baseline_probe,
            PROBE_METHOD,
            observation.small_units * observation.capacity_lanes,
            observation.large_units * observation.capacity_lanes,
            observation.timer_resolution,
            observation.baseline_capacity_small,
            observation.baseline_capacity_large,
            observation.acquisition_duration_ns,
        ),
    ];
    Ok(ServiceDefinition {
        class,
        correlation: ServiceCorrelationId::new(class.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units,
            maximum_concurrent_uses: observation.capacity_lanes.max(1),
        },
        accuracy: ServiceAccuracyClass::Compute,
        topology: ResourceTopology {
            resources: 1,
            max_concurrency: 1,
        },
        dependency_latency,
        saturated_capacity: ServiceCurve {
            setup: DurationInterval::new(0, 0, 1),
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit: saturated,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: batches.into_boxed_slice(),
        },
    })
}

fn measured_definition(
    class: ServiceClassId,
    observation: Observation,
    compute: bool,
    probe: &'static str,
    method: &'static str,
) -> Result<ServiceDefinition, seismic_compiler::errors::TargetError> {
    let delta = observation.work_delta;
    let dependency_latency = observation.dependency_interval;
    let saturated = observation.capacity_interval;
    qualify(class, dependency_latency, if compute { 1 } else { 2 })?;
    qualify(class, saturated, if compute { 1 } else { 2 })?;
    let measured_capacity_units =
        delta
            .checked_mul(observation.capacity_lanes)
            .ok_or_else(|| {
                seismic_compiler::errors::TargetError::UnsupportedDevice(
                    "Metal service-probe workload exceeds u64".into(),
                )
            })?;
    let maximum_units = measured_capacity_units.checked_mul(2).ok_or_else(|| {
        seismic_compiler::errors::TargetError::UnsupportedDevice(
            "Metal qualified service domain exceeds u64".into(),
        )
    })?;
    Ok(ServiceDefinition {
        class,
        correlation: ServiceCorrelationId::new(class.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units,
            maximum_concurrent_uses: observation.capacity_lanes.max(1),
        },
        accuracy: if compute {
            ServiceAccuracyClass::Compute
        } else {
            ServiceAccuracyClass::MemoryOrTransfer
        },
        topology: ResourceTopology {
            resources: 1,
            max_concurrency: 1,
        },
        dependency_latency,
        saturated_capacity: ServiceCurve {
            // Launch-level submission/dispatch is already represented once by
            // core.submission. GPU service curves describe incremental work;
            // retaining an empty-command duration here would count the same
            // launch boundary once per service used by a kernel.
            setup: DurationInterval::new(0, 0, 1),
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit: saturated,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: vec![
                paired_measurement_batch(
                    probe,
                    method,
                    observation.small_units * observation.dependency_lanes,
                    observation.large_units * observation.dependency_lanes,
                    observation.timer_resolution,
                    observation.latency_small,
                    observation.latency_large,
                    observation.acquisition_duration_ns,
                ),
                paired_measurement_batch(
                    probe,
                    method,
                    observation.small_units * observation.capacity_lanes,
                    observation.large_units * observation.capacity_lanes,
                    observation.timer_resolution,
                    observation.capacity_small,
                    observation.capacity_large,
                    observation.acquisition_duration_ns,
                ),
            ]
            .into_boxed_slice(),
        },
    })
}

fn qualify_composition(
    _services: &[ServiceDefinition],
    observation: &Observation,
    anchor: &Observation,
    three: &Observation,
) -> Result<CompositionQualificationParts, seismic_compiler::errors::TargetError> {
    // The held-out adjacent difference is deliberately the incremental GPU
    // composition. The kernel is one dependent arithmetic chain, so its case
    // must exercise the same dependency-latency demand that `operation_cost`
    // emits for production F32 arithmetic. Its fixed dispatch is owned and
    // qualified independently by core.submission.
    let uncertainty = observation
        .timer_resolution
        .upper_numerator
        .max(anchor.timer_resolution.upper_numerator)
        .max(three.timer_resolution.upper_numerator)
        .saturating_mul(12);
    // If T_k is the adjacent time for a loop body containing k dependent
    // additions, the independent one/three shapes identify the loop harness
    // as (3*T_1-T_3)/2. The held-out retains its exact two-add body and removes
    // only that independently acquired harness: (2*T_2-3*T_1+T_3)/2.
    let observed_ns = paired_linear_combination_confidence_interval(
        &[
            (&observation.latency_small, &observation.latency_large, 2),
            (&anchor.latency_small, &anchor.latency_large, -3),
            (&three.latency_small, &three.latency_large, 1),
        ],
        2,
        uncertainty,
    )?;
    let mut digest = Sha256::new();
    for sample in observation
        .latency_small
        .iter()
        .chain(&observation.latency_large)
        .chain(&observation.capacity_small)
        .chain(&observation.capacity_large)
    {
        digest.update(sample.to_le_bytes());
    }
    Ok(CompositionQualificationParts {
        suite_revision: "seismic-metal-composition-v1",
        cases: vec![CompositionQualificationCase {
            stable_name: "strict-f32-dependent-pair",
            demands: vec![ConcreteExecutionDemand {
                class: F32_ADD_SUB,
                units: observation.work_delta,
                mode: DemandMode::DependencyLatency,
            }],
            observed_ns,
        }],
        observations_digest: digest.finalize().into(),
    })
}

fn confidence_interval(
    samples: &[u64],
    denominator: u64,
    uncertainty_ns: u64,
) -> Result<DurationInterval, seismic_compiler::errors::TargetError> {
    if samples.len() < 2 || denominator == 0 {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "service confidence interval requires two observations and a nonzero domain".into(),
        ));
    }
    let count = samples.len() as f64;
    let mean = samples.iter().map(|sample| *sample as f64).sum::<f64>() / count;
    if mean <= 0.0 || !mean.is_finite() {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "service confidence interval has no positive finite observation mean".into(),
        ));
    }
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = *sample as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1.0);
    let confidence_half_width = GPU_CONFIDENCE_SIGMA * (variance / count).sqrt();
    let lower = (mean - confidence_half_width).floor().max(0.0) as u64;
    let upper = (mean + confidence_half_width).ceil() as u64;
    Ok(widen(
        DurationInterval::new(lower, upper, denominator),
        uncertainty_ns,
    ))
}

fn paired_difference_confidence_interval(
    small: &[u64],
    large: &[u64],
    denominator: u64,
    uncertainty_ns: u64,
) -> Result<Option<DurationInterval>, seismic_compiler::errors::TargetError> {
    if small.len() != large.len() || small.len() < 2 || denominator == 0 {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "paired service confidence interval requires matching observations and a nonzero domain"
                .into(),
        ));
    }
    let count = small.len() as f64;
    let differences = small
        .iter()
        .zip(large)
        .map(|(small, large)| *large as f64 - *small as f64)
        .collect::<Vec<_>>();
    let mean = differences.iter().sum::<f64>() / count;
    if !mean.is_finite() {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "paired service confidence interval produced a non-finite mean".into(),
        ));
    }
    if mean <= 0.0 {
        return Ok(None);
    }
    let variance = differences
        .iter()
        .map(|difference| {
            let delta = *difference - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1.0);
    let confidence_half_width = GPU_CONFIDENCE_SIGMA * (variance / count).sqrt();
    let lower = (mean - confidence_half_width).floor().max(0.0) as u64;
    let upper = (mean + confidence_half_width).ceil() as u64;
    Ok(Some(widen(
        DurationInterval::new(lower, upper, denominator),
        uncertainty_ns,
    )))
}

fn paired_matched_confidence_interval(
    operation_small: &[u64],
    operation_large: &[u64],
    baseline_small: &[u64],
    baseline_large: &[u64],
    denominator: u64,
    uncertainty_ns: u64,
) -> Result<Option<DurationInterval>, seismic_compiler::errors::TargetError> {
    let count = operation_small.len();
    if count < 2
        || denominator == 0
        || operation_large.len() != count
        || baseline_small.len() != count
        || baseline_large.len() != count
    {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "matched service contrast requires four aligned observation series and a nonzero domain"
                .into(),
        ));
    }
    let contrasts = (0..count)
        .map(|index| {
            (operation_large[index] as f64 - operation_small[index] as f64)
                - (baseline_large[index] as f64 - baseline_small[index] as f64)
        })
        .collect::<Vec<_>>();
    let count = count as f64;
    let mean = contrasts.iter().sum::<f64>() / count;
    if !mean.is_finite() {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "matched service contrast produced a non-finite mean".into(),
        ));
    }
    if mean <= 0.0 {
        return Ok(None);
    }
    let variance = contrasts
        .iter()
        .map(|contrast| {
            let delta = *contrast - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1.0);
    let half_width = GPU_CONFIDENCE_SIGMA * (variance / count).sqrt();
    let lower = (mean - half_width).floor().max(0.0) as u64;
    let upper = (mean + half_width).ceil() as u64;
    Ok(Some(widen(
        DurationInterval::new(lower, upper, denominator),
        uncertainty_ns,
    )))
}

fn paired_batch_mean_confidence_interval(
    small: &[u64],
    large: &[u64],
    units_per_difference: u64,
    batch_size: usize,
    uncertainty_ns: u64,
) -> Result<Option<DurationInterval>, seismic_compiler::errors::TargetError> {
    const STUDENT_T_975_DF7: f64 = 2.364_624;
    if small.len() != large.len()
        || batch_size == 0
        || small.len() != batch_size * 8
        || units_per_difference == 0
    {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "thread service batch-mean interval requires eight equal complete batches and a nonzero domain"
                .into(),
        ));
    }
    let batch_sums = small
        .chunks_exact(batch_size)
        .zip(large.chunks_exact(batch_size))
        .map(|(small, large)| {
            small
                .iter()
                .zip(large)
                .map(|(small, large)| *large as f64 - *small as f64)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let count = batch_sums.len() as f64;
    let mean = batch_sums.iter().sum::<f64>() / count;
    if !mean.is_finite() {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            "thread service batch-mean interval produced a non-finite mean".into(),
        ));
    }
    if mean <= 0.0 {
        return Ok(None);
    }
    let variance = batch_sums
        .iter()
        .map(|batch| {
            let delta = *batch - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1.0);
    let half_width = STUDENT_T_975_DF7 * (variance / count).sqrt();
    let lower = (mean - half_width).floor().max(0.0) as u64;
    let upper = (mean + half_width).ceil() as u64;
    let denominator = units_per_difference
        .checked_mul(batch_size as u64)
        .ok_or_else(|| {
            seismic_compiler::errors::TargetError::UnsupportedDevice(
                "thread service batch domain exceeds u64".into(),
            )
        })?;
    Ok(Some(widen(
        DurationInterval::new(lower, upper, denominator),
        uncertainty_ns,
    )))
}

fn widen(interval: DurationInterval, uncertainty_ns: u64) -> DurationInterval {
    DurationInterval::new(
        interval.lower_numerator.saturating_sub(uncertainty_ns),
        interval.upper_numerator.saturating_add(uncertainty_ns),
        interval.denominator,
    )
}

fn host_timer_resolution(clock_id: c_int) -> DurationInterval {
    let mut minimum = u64::MAX;
    for _ in 0..4_096 {
        let start = unsafe { clock_gettime_nsec_np(clock_id) };
        let elapsed = loop {
            let elapsed = unsafe { clock_gettime_nsec_np(clock_id) }.saturating_sub(start);
            if elapsed != 0 {
                break elapsed;
            }
        };
        minimum = minimum.min(elapsed);
    }
    DurationInterval::new(0, minimum, 1)
}

fn qualify(
    class: ServiceClassId,
    interval: DurationInterval,
    percent: u64,
) -> Result<(), seismic_compiler::errors::TargetError> {
    let width = u128::from(interval.upper_numerator - interval.lower_numerator);
    let sum = u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
    if width * 100 > sum * u128::from(percent) {
        return Err(seismic_compiler::errors::TargetError::DeviceUnavailable(
            format!(
                "service probe `{}` interval {}/{}..{}/{} ns exceeded its {percent}% relative half-width qualification",
                class.stable_name(),
                interval.lower_numerator,
                interval.denominator,
                interval.upper_numerator,
                interval.denominator,
            ),
        ));
    }
    Ok(())
}

fn is_qualified(interval: DurationInterval, percent: u64) -> bool {
    let width = u128::from(interval.upper_numerator - interval.lower_numerator);
    let sum = u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
    width * 100 <= sum * u128::from(percent)
}

fn core_services(
    device: &crate::device::MetalDevice,
    buffer: &crate::device::MetalBuffer,
) -> Result<Vec<ServiceDefinition>, seismic_compiler::errors::TargetError> {
    let mut result = Vec::new();
    result.push(host_definition(
        SERVICE_SUBMISSION,
        "metal.command-submit",
        ServiceAccuracyClass::MemoryOrTransfer,
        1,
        2_048,
        18_432,
        || {
            let command = device.queue().commandBuffer().ok_or_else(|| {
                seismic_compiler::errors::TargetError::DeviceUnavailable(
                    "opened Metal queue stopped creating command buffers".into(),
                )
            })?;
            command.commit();
            command.waitUntilCompleted();
            std::hint::black_box(command.status());
            Ok(())
        },
    )?);
    result.push(host_transfer_definition(
        SERVICE_COPY,
        "metal.shared-copy",
        buffer,
        true,
    )?);
    result.push(host_transfer_definition(
        SERVICE_FILL,
        "metal.shared-fill",
        buffer,
        false,
    )?);
    result.push(host_definition(
        SERVICE_SCALAR_READ,
        "metal.shared-scalar-read",
        ServiceAccuracyClass::MemoryOrTransfer,
        1,
        524_288,
        2_097_152,
        || {
            let mut bytes = [0u8; 4];
            buffer.read_bytes(0, &mut bytes);
            std::hint::black_box(bytes);
            Ok(())
        },
    )?);
    result.push(host_compute_definition(
        SERVICE_SCALAR_MOVE,
        "core.scalar-move",
        1,
        4_194_304,
        16_777_216,
        || {
            let value = std::hint::black_box(0x9e37_79b9_u64);
            std::hint::black_box(value);
            Ok(())
        },
    )?);
    result.push(host_compute_definition(
        SERVICE_DATA_CHECK,
        "core.data-check",
        1,
        8_388_608,
        33_554_432,
        || {
            let value = std::hint::black_box(true);
            std::hint::black_box(if value { 1u32 } else { 0u32 });
            Ok(())
        },
    )?);
    Ok(result)
}

fn host_transfer_definition(
    class: ServiceClassId,
    probe: &'static str,
    buffer: &crate::device::MetalBuffer,
    copy: bool,
) -> Result<ServiceDefinition, seismic_compiler::errors::TargetError> {
    const SMALL: usize = 256 * 1024;
    const LARGE: usize = 2 * 1024 * 1024;
    const AMPLIFICATION: usize = 2_048;
    const SETUP_AMPLIFICATION: usize = 16_384;
    const SETUP_REPEATS_PER_TRANSFER_OBSERVATION: usize = 8;
    let pattern = [0x5au8, 0xa5, 0x3c, 0xc3];
    let mut temporary = vec![0u8; LARGE];
    for (index, byte) in temporary.iter_mut().enumerate() {
        *byte = pattern[index % pattern.len()];
    }
    let mut execute = |bytes: usize| {
        if copy {
            buffer.read_bytes(0, &mut temporary[..bytes]);
            buffer.write_bytes(512 * 1024, &temporary[..bytes]);
        } else {
            buffer.write_bytes(0, &temporary[..bytes]);
        }
    };
    let acquisition_started = Instant::now();
    let timer_resolution = host_timer_resolution(CLOCK_MONOTONIC_RAW);
    let uncertainty = timer_resolution.upper_numerator.saturating_mul(2);
    let workload_units = (LARGE - SMALL) as u64 * AMPLIFICATION as u64;
    let mut accepted = None;
    let mut last_intervals = None;
    for _ in 0..MAX_HOST_ACQUISITION_ROUNDS {
        for _ in 0..AMPLIFICATION {
            execute(LARGE);
        }
        let mut rates = Vec::with_capacity(STOCHASTIC_HOST_REPEATS);
        let mut setups =
            Vec::with_capacity(STOCHASTIC_HOST_REPEATS * SETUP_REPEATS_PER_TRANSFER_OBSERVATION);
        for _ in 0..STOCHASTIC_HOST_REPEATS {
            for _ in 0..SETUP_REPEATS_PER_TRANSFER_OBSERVATION {
                let start = Instant::now();
                // A single zero-byte transfer is comparable to the host clock
                // resolution, so its interval cannot carry the same 2% guarantee
                // as the transfer slope. Measure the unchanged fixed setup in an
                // amplified batch and retain the rational per-operation interval.
                for _ in 0..SETUP_AMPLIFICATION {
                    execute(0);
                }
                setups.push(start.elapsed().as_nanos() as u64);
            }
            let start = Instant::now();
            for _ in 0..AMPLIFICATION {
                execute(SMALL);
            }
            let small = start.elapsed().as_nanos() as u64;
            let start = Instant::now();
            for _ in 0..AMPLIFICATION {
                execute(LARGE);
            }
            let large = start.elapsed().as_nanos() as u64;
            rates.push(large.checked_sub(small).ok_or_else(|| {
                seismic_compiler::errors::TargetError::DeviceUnavailable(
                    "host transfer probe timestamps were not monotone".into(),
                )
            })?);
        }
        let per_unit = confidence_interval(&rates, workload_units, uncertainty)?;
        let setup = confidence_interval(&setups, SETUP_AMPLIFICATION as u64, uncertainty)?;
        last_intervals = Some((per_unit, setup));
        if is_qualified(per_unit, 2) && is_qualified(setup, 2) {
            accepted = Some((rates, setups, per_unit, setup));
            break;
        }
    }
    let (rates, setups, per_unit, setup) = accepted.ok_or_else(|| {
        let (per_unit, setup) =
            last_intervals.expect("bounded transfer acquisition executes at least once");
        seismic_compiler::errors::TargetError::DeviceUnavailable(format!(
            "service probe `{}` transfer interval {}/{}..{}/{} ns or setup interval {}/{}..{}/{} ns exceeded its 2% relative half-width qualification after {MAX_HOST_ACQUISITION_ROUNDS} fixed acquisition rounds",
            class.stable_name(),
            per_unit.lower_numerator,
            per_unit.denominator,
            per_unit.upper_numerator,
            per_unit.denominator,
            setup.lower_numerator,
            setup.denominator,
            setup.upper_numerator,
            setup.denominator,
        ))
    })?;
    Ok(ServiceDefinition {
        class,
        correlation: ServiceCorrelationId::new(class.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units: workload_units,
            maximum_concurrent_uses: 1,
        },
        accuracy: ServiceAccuracyClass::MemoryOrTransfer,
        topology: ResourceTopology {
            resources: 1,
            max_concurrency: 1,
        },
        dependency_latency: per_unit,
        saturated_capacity: ServiceCurve {
            setup,
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: vec![
                measurement_batch(
                    probe,
                    HOST_METHOD,
                    workload_units,
                    timer_resolution,
                    rates,
                    acquisition_started.elapsed().as_nanos() as u64,
                ),
                measurement_batch(
                    probe,
                    HOST_METHOD,
                    SETUP_AMPLIFICATION as u64,
                    timer_resolution,
                    setups,
                    acquisition_started.elapsed().as_nanos() as u64,
                ),
            ]
            .into_boxed_slice(),
        },
    })
}

fn host_definition(
    class: ServiceClassId,
    probe: &'static str,
    accuracy: ServiceAccuracyClass,
    units_per_operation: u64,
    small: usize,
    large: usize,
    mut operation: impl FnMut() -> Result<(), seismic_compiler::errors::TargetError>,
) -> Result<ServiceDefinition, seismic_compiler::errors::TargetError> {
    assert_eq!(accuracy, ServiceAccuracyClass::MemoryOrTransfer);
    let acquisition_started = Instant::now();
    let percent = 2;
    let clock_id = CLOCK_MONOTONIC_RAW;
    let timer_resolution = host_timer_resolution(clock_id);
    let workload_units = (large - small) as u64 * units_per_operation;
    let uncertainty = timer_resolution.upper_numerator.saturating_mul(2);
    let mut accepted = None;
    let mut last_interval = None;
    for _ in 0..MAX_HOST_ACQUISITION_ROUNDS {
        // Re-warm the exact amplified path before each bounded acquisition
        // round. A round is accepted only as a whole: its raw extrema remain
        // the retained interval, with no sample trimming or threshold change.
        for _ in 0..large {
            operation()?;
        }
        let mut samples = Vec::with_capacity(STOCHASTIC_HOST_REPEATS);
        for _ in 0..STOCHASTIC_HOST_REPEATS {
            let start = unsafe { clock_gettime_nsec_np(clock_id) };
            for _ in 0..small {
                operation()?;
            }
            let small_elapsed = unsafe { clock_gettime_nsec_np(clock_id) }.saturating_sub(start);
            let start = unsafe { clock_gettime_nsec_np(clock_id) };
            for _ in 0..large {
                operation()?;
            }
            let large_elapsed = unsafe { clock_gettime_nsec_np(clock_id) }.saturating_sub(start);
            samples.push(large_elapsed.checked_sub(small_elapsed).ok_or_else(|| {
                seismic_compiler::errors::TargetError::DeviceUnavailable(
                    "host service probe timestamps were not monotone".into(),
                )
            })?);
        }
        let candidate = confidence_interval(&samples, workload_units, uncertainty)?;
        last_interval = Some(candidate);
        if is_qualified(candidate, percent) {
            accepted = Some((samples, candidate));
            break;
        }
    }
    let (samples, per_unit) = accepted.ok_or_else(|| {
        let interval = last_interval.expect("bounded host acquisition executes at least once");
        seismic_compiler::errors::TargetError::DeviceUnavailable(format!(
            "service probe `{}` interval {}/{}..{}/{} ns exceeded its {percent}% relative half-width qualification after {MAX_HOST_ACQUISITION_ROUNDS} fixed acquisition rounds",
            class.stable_name(),
            interval.lower_numerator,
            interval.denominator,
            interval.upper_numerator,
            interval.denominator,
        ))
    })?;
    Ok(ServiceDefinition {
        class,
        correlation: ServiceCorrelationId::new(class.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units: workload_units,
            maximum_concurrent_uses: 1,
        },
        accuracy,
        topology: ResourceTopology {
            resources: 1,
            max_concurrency: 1,
        },
        dependency_latency: per_unit,
        saturated_capacity: ServiceCurve {
            setup: DurationInterval::new(0, 0, 1),
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: vec![measurement_batch(
                probe,
                HOST_METHOD,
                workload_units,
                timer_resolution,
                samples,
                acquisition_started.elapsed().as_nanos() as u64,
            )]
            .into_boxed_slice(),
        },
    })
}

fn host_compute_definition(
    class: ServiceClassId,
    probe: &'static str,
    units_per_operation: u64,
    small: usize,
    large: usize,
    mut operation: impl FnMut() -> Result<(), seismic_compiler::errors::TargetError>,
) -> Result<ServiceDefinition, seismic_compiler::errors::TargetError> {
    const OBSERVATIONS_PER_ROUND: usize = STOCHASTIC_HOST_REPEATS;
    let acquisition_started = Instant::now();
    let timer_resolution = host_timer_resolution(CLOCK_THREAD_CPUTIME_ID);
    let timer_tick = timer_resolution.upper_numerator;
    let workload_units = (large - small) as u64 * units_per_operation;
    let total_observations = OBSERVATIONS_PER_ROUND * MAX_HOST_ACQUISITION_ROUNDS;
    let mut small_observations = Vec::with_capacity(total_observations);
    let mut large_observations = Vec::with_capacity(total_observations);
    for _ in 0..MAX_HOST_ACQUISITION_ROUNDS {
        for _ in 0..large {
            operation()?;
        }
        for observation in 0..OBSERVATIONS_PER_ROUND {
            let measure = |count: usize,
                           operation: &mut dyn FnMut() -> Result<
                (),
                seismic_compiler::errors::TargetError,
            >|
             -> Result<u64, seismic_compiler::errors::TargetError> {
                let start = unsafe { clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) };
                for _ in 0..count {
                    operation()?;
                }
                Ok(unsafe { clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) }.saturating_sub(start))
            };
            let (small_elapsed, large_elapsed) = if observation % 2 == 0 {
                (
                    measure(small, &mut operation)?,
                    measure(large, &mut operation)?,
                )
            } else {
                let large_elapsed = measure(large, &mut operation)?;
                let small_elapsed = measure(small, &mut operation)?;
                (small_elapsed, large_elapsed)
            };
            small_observations.push(small_elapsed);
            large_observations.push(large_elapsed);
        }
    }
    let per_unit = paired_batch_mean_confidence_interval(
        &small_observations,
        &large_observations,
        workload_units,
        OBSERVATIONS_PER_ROUND,
        timer_tick.saturating_mul((OBSERVATIONS_PER_ROUND * 2) as u64),
    )?
    .ok_or_else(|| {
        seismic_compiler::errors::TargetError::DeviceUnavailable(format!(
            "service probe `{}` had a nonpositive adjacent-count mean across its fixed eight-round acquisition",
            class.stable_name(),
        ))
    })?;
    qualify(class, per_unit, 1)?;
    let acquisition_duration_ns = acquisition_started.elapsed().as_nanos() as u64;
    Ok(ServiceDefinition {
        class,
        correlation: ServiceCorrelationId::new(class.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units: workload_units,
            maximum_concurrent_uses: 1,
        },
        accuracy: ServiceAccuracyClass::Compute,
        topology: ResourceTopology {
            resources: 1,
            max_concurrency: 1,
        },
        dependency_latency: per_unit,
        saturated_capacity: ServiceCurve {
            setup: DurationInterval::new(0, 0, 1),
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: vec![paired_measurement_batch(
                probe,
                THREAD_METHOD,
                small as u64 * units_per_operation,
                large as u64 * units_per_operation,
                timer_resolution,
                small_observations,
                large_observations,
                acquisition_duration_ns,
            )]
            .into_boxed_slice(),
        },
    })
}

fn probe_source(include_bfloat: bool, include_matrix: bool) -> String {
    let bfloat = include_bfloat.then_some(
        "kernel void seismic_probe_bf16(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) { bfloat x = bfloat(float(tid) + 1.0f); for (uint i=0; i<n; ++i) x = bf16_add(x, bfloat(0.0009765625f)); out[tid] = uint(as_type<ushort>(x)); }\n\
         kernel void seismic_probe_f32_to_bf16(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) { float x=float(tid)+1.0f; bfloat y=bfloat(0.0f); for(uint i=0;i<n;++i) { y=seismic_bf16_narrow(x); x=seismic_bf16_widen(y); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); } out[tid]=as_type<uint>(x); }\n\
         kernel void seismic_baseline_f32_to_bf16(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) { float x=float(tid)+1.0f; bfloat y=bfloat(0.0f); for(uint i=0;i<n;++i) { y=as_type<bfloat>(ushort(as_type<uint>(x))); x=seismic_bf16_widen(y); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); } out[tid]=as_type<uint>(x); }\n\
"
    ).unwrap_or("");
    let matrix = if include_matrix {
        "kernel void seismic_probe_matrix(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]], uint tid [[thread_position_in_grid]]) { threadgroup half a[64]; threadgroup half b[64]; threadgroup float c[64]; for (uint e=lane; e<64; e+=width) { a[e]=half(1); b[e]=half(1); c[e]=0.0f; } threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_matrix<half,8,8> am; simdgroup_matrix<half,8,8> bm; simdgroup_matrix<float,8,8> cm; simdgroup_load(am,a,8); simdgroup_load(bm,b,8); simdgroup_load(cm,c,8); for (uint i=0; i<n; ++i) { simdgroup_matrix<float,8,8> next; simdgroup_multiply_accumulate(next,am,bm,cm); cm=next; } simdgroup_store(cm,c,8); threadgroup_barrier(mem_flags::mem_threadgroup); if (lane==0) out[tid/width]=as_type<uint>(c[0]); }\n".to_string()
    } else {
        String::new()
    };
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n{}\n\
         kernel void seismic_probe_control(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ uint x=tid; for(uint i=0;i<n;++i) x=select(x+1u,x^i,(x&1u)!=0u); out[tid]=x; }}\n\
         kernel void seismic_probe_integer(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ uint x=tid+1u; for(uint i=0;i<n;++i) x=x*1664525u; out[tid]=x; }}\n\
         kernel void seismic_probe_f32_anchor(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=f32_add(x,0.000244140625f); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f32(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=f32_add(x,0.00048828125f); x=f32_add(x,0.00390625f); x=f32_add(x,0.0001220703125f); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f32_multiply(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=f32_mul(x,0.9999999403953552f); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_baseline_f32_multiply(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=as_type<float>(as_type<uint>(x)); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f32_divide(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=f32_div(x,1.0000001192092896f); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_baseline_f32_divide(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=as_type<float>(as_type<uint>(x)); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f32_remainder(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=f32_rem(x,3.25f); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_baseline_f32_remainder(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=as_type<float>(as_type<uint>(x)); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f32_minmax(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ uint rhs=as_type<uint>(x)^1u; x=f32_min(x,as_type<float>(rhs)); x=as_type<float>(as_type<uint>(x)^((rhs^i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_baseline_f32_minmax(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ uint rhs=as_type<uint>(x)^1u; x=as_type<float>(as_type<uint>(x)); x=as_type<float>(as_type<uint>(x)^((rhs^i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f32_fma(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=f32_mulAdd(x,0.9999999403953552f,0.000244140625f); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_baseline_f32_fma(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=as_type<float>(as_type<uint>(x)); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f32_compare(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; uint bits=0u; for(uint i=0;i<n;++i) {{ uint rhs=as_type<uint>(x)^1u; uint predicate=uint(f32_lt(x,as_type<float>(rhs))); bits^=predicate; bits^=(rhs^i^tid)&1u; x=as_type<float>(as_type<uint>(x)^bits); }} out[tid]=as_type<uint>(x)^bits; }}\n\
         kernel void seismic_probe_f32_to_integer(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid&1023u)+1.0f; uint y=0u; for(uint i=0;i<n;++i) {{ y=uint(clamp(trunc(float(x)),0.0f,4294967295.0f)); x=float((y+1u)&1023u); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=y; }}\n\
         kernel void seismic_baseline_f32_to_integer(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid&1023u)+1.0f; uint y=0u; for(uint i=0;i<n;++i) {{ y=as_type<uint>(x); x=float((y+1u)&1023u); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=y; }}\n\
         kernel void seismic_probe_integer_to_f32(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ uint x=tid&1023u; float y=0.0f; for(uint i=0;i<n;++i) {{ y=float(x); x=(as_type<uint>(y)+1u+i)&1023u; }} out[tid]=as_type<uint>(y); }}\n\
         kernel void seismic_baseline_integer_to_f32(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ uint x=tid&1023u; float y=0.0f; for(uint i=0;i<n;++i) {{ y=as_type<float>(x); x=(as_type<uint>(y)+1u+i)&1023u; }} out[tid]=as_type<uint>(y); }}\n\
         kernel void seismic_probe_f32_to_f16(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; half y=half(0.0f); for(uint i=0;i<n;++i) {{ y=f32_to_f16(x); x=f16_to_f32(y); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=uint(as_type<ushort>(y)); }}\n\
         kernel void seismic_baseline_f32_to_f16(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; half y=half(0.0f); for(uint i=0;i<n;++i) {{ y=as_type<half>(ushort(as_type<uint>(x))); x=f16_to_f32(y); x=as_type<float>(as_type<uint>(x)^((i^tid)&1u)); }} out[tid]=uint(as_type<ushort>(y)); }}\n\
         kernel void seismic_probe_f16_to_f32(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ half x=half(float(tid)+1.0f); float y=0.0f; for(uint i=0;i<n;++i) {{ y=f16_to_f32(x); x=f32_to_f16(y); x=as_type<half>(ushort(as_type<ushort>(x)^ushort((i^tid)&1u))); }} out[tid]=as_type<uint>(y); }}\n\
         kernel void seismic_baseline_f16_to_f32(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ half x=half(float(tid)+1.0f); float y=0.0f; for(uint i=0;i<n;++i) {{ y=as_type<float>(uint(as_type<ushort>(x))); x=f32_to_f16(y); x=as_type<half>(ushort(as_type<ushort>(x)^ushort((i^tid)&1u))); }} out[tid]=as_type<uint>(y); }}\n\
         kernel void seismic_probe_composition(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid)+1.0f; for(uint i=0;i<n;++i) {{ x=f32_add(x,0.0009765625f); x=f32_add(x,0.001953125f); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_f16(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ half x=half(float(tid)+1.0f); for(uint i=0;i<n;++i) x=f16_add(x,half(0.0009765625f)); out[tid]=uint(as_type<ushort>(x)); }}\n\
         {bfloat}\
         kernel void seismic_probe_approx(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ float x=float(tid&15u)*0.01f; for(uint i=0;i<n;++i) x=fast::sin(x)+0.01f; out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_global(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ uint x=tid; for(uint i=0;i<n;++i) x=out[(x+i)&0xfffffu]; out[tid]=x; }}\n\
         kernel void seismic_probe_workgroup(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint lane [[thread_index_in_threadgroup]], uint tid [[thread_position_in_grid]]) {{ threadgroup uint x[32]; x[lane]=(lane*5u+1u)&31u; threadgroup_barrier(mem_flags::mem_threadgroup); uint index=lane; for(uint i=0;i<n;++i) index=x[index]; out[tid]=index; }}\n\
         kernel void seismic_probe_representation(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ uint packet=out[tid]; float x=0.0f; for(uint i=0;i<n;++i) {{ int q=int((packet>>((i&7u)*4u))&15u)-8; x=f32_add(x,float(q)*0.0625f); }} out[tid]=as_type<uint>(x); }}\n\
         kernel void seismic_probe_atomic(device atomic_uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{ for(uint i=0;i<n;++i) atomic_fetch_add_explicit(out+(tid&255u),1u,memory_order_relaxed); }}\n\
         kernel void seismic_probe_barrier(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint lane [[thread_index_in_threadgroup]], uint tid [[thread_position_in_grid]]) {{ threadgroup uint x[32]; x[lane]=tid; for(uint i=0;i<n;++i) {{ threadgroup_barrier(mem_flags::mem_threadgroup); x[lane]^=i; }} out[tid]=x[lane]; }}\n\
         kernel void seismic_probe_subgroup(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_position_in_grid]]) {{ uint x=lane; for(uint i=0;i<n;++i) x=simd_sum(x)+lane; out[tid]=x; }}\n\
         {matrix}",
        format!(
            "{}\n{}",
            crate::render::LIBRARY_PRELUDE,
            include_str!("softfloat.metal")
        )
    )
}

#[cfg(test)]
mod tests {
    use super::{
        confidence_interval, is_qualified, paired_batch_mean_confidence_interval,
        paired_difference_confidence_interval, paired_matched_confidence_interval,
        paired_measurement_batch, probe_source, DurationInterval, F16_TO_F32, F32_COMPARE,
        F32_DIVIDE, F32_FMA, F32_MIN_MAX, F32_MULTIPLY, F32_REMAINDER, F32_TO_BF16, F32_TO_F16,
        F32_TO_INTEGER, INTEGER_TO_F32, PROBE_METHOD,
    };

    #[test]
    fn qualification_keeps_exact_compute_and_memory_boundaries() {
        assert!(is_qualified(DurationInterval::new(99, 101, 1), 1));
        assert!(!is_qualified(DurationInterval::new(98, 101, 1), 1));
        assert!(is_qualified(DurationInterval::new(98, 102, 1), 2));
        assert!(!is_qualified(DurationInterval::new(97, 102, 1), 2));
    }

    #[test]
    fn gpu_confidence_interval_retains_timer_uncertainty() {
        let samples = [1_000; 64];
        assert_eq!(
            confidence_interval(&samples, 10, 2).unwrap(),
            DurationInterval::new(998, 1_002, 10)
        );
    }

    #[test]
    fn paired_gpu_confidence_allows_negative_individual_differences() {
        let small = [1_000; 64];
        let mut large = [1_200; 64];
        large[7] = 900;
        let interval = paired_difference_confidence_interval(&small, &large, 10, 0)
            .unwrap()
            .unwrap();
        assert!(interval.lower_numerator > 0);
        assert_eq!(interval.denominator, 10);
    }

    #[test]
    fn paired_gpu_confidence_rejects_a_nonpositive_round_mean() {
        let small = [1_200; 4];
        let large = [1_000; 4];
        assert!(paired_difference_confidence_interval(&small, &large, 10, 0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn matched_gpu_confidence_subtracts_aligned_baseline_differences() {
        let operation_small = [100; 4];
        let operation_large = [250; 4];
        let baseline_small = [80; 4];
        let baseline_large = [130; 4];
        assert_eq!(
            paired_matched_confidence_interval(
                &operation_small,
                &operation_large,
                &baseline_small,
                &baseline_large,
                10,
                2,
            )
            .unwrap()
            .unwrap(),
            DurationInterval::new(98, 102, 10)
        );
    }

    #[test]
    fn fixed_gpu_rounds_are_qualified_once_from_all_pooled_evidence() {
        let operation_small = vec![1_000; 192];
        let baseline_small = vec![1_000; 192];
        let baseline_large = vec![1_100; 192];
        let operation_large = (0..192)
            .map(|index| 1_100 + if index % 2 == 0 { 94 } else { 106 })
            .collect::<Vec<_>>();

        for round in 0..3 {
            let range = round * 64..(round + 1) * 64;
            let interval = paired_matched_confidence_interval(
                &operation_small[range.clone()],
                &operation_large[range.clone()],
                &baseline_small[range.clone()],
                &baseline_large[range],
                1,
                0,
            )
            .unwrap()
            .unwrap();
            assert!(!is_qualified(interval, 1));
        }

        let pooled = paired_matched_confidence_interval(
            &operation_small,
            &operation_large,
            &baseline_small,
            &baseline_large,
            1,
            0,
        )
        .unwrap()
        .unwrap();
        assert!(is_qualified(pooled, 1));
    }

    #[test]
    fn pooled_gpu_evidence_does_not_discard_a_negative_round() {
        let operation_small = vec![1_000; 192];
        let baseline_small = vec![1_000; 192];
        let baseline_large = vec![1_100; 192];
        let operation_large = (0..192)
            .map(|index| if index < 64 { 1_000 } else { 1_300 })
            .collect::<Vec<_>>();
        let pooled = paired_matched_confidence_interval(
            &operation_small,
            &operation_large,
            &baseline_small,
            &baseline_large,
            1,
            0,
        )
        .unwrap()
        .unwrap();
        assert!(!is_qualified(pooled, 1));
    }

    #[test]
    fn matched_probe_provenance_retains_operation_and_baseline_raw_pairs() {
        let timer = DurationInterval::new(0, 1, 1);
        let operation = paired_measurement_batch(
            "operation",
            PROBE_METHOD,
            10,
            20,
            timer,
            vec![100, 110],
            vec![200, 220],
            1_000,
        );
        let baseline = paired_measurement_batch(
            "baseline",
            PROBE_METHOD,
            10,
            20,
            timer,
            vec![70, 80],
            vec![120, 140],
            1_000,
        );
        assert_eq!(
            operation.series.paired_adjacent_parts(),
            Some((10, 20, [100, 110].as_slice(), [200, 220].as_slice()))
        );
        assert_eq!(
            baseline.series.paired_adjacent_parts(),
            Some((10, 20, [70, 80].as_slice(), [120, 140].as_slice()))
        );
        assert_ne!(operation.observations_digest, baseline.observations_digest);
    }

    #[test]
    fn every_split_f32_family_has_a_compiled_matched_baseline() {
        let source = probe_source(true, false);
        for class in [
            F32_MULTIPLY,
            F32_DIVIDE,
            F32_REMAINDER,
            F32_MIN_MAX,
            F32_FMA,
            F32_TO_INTEGER,
            INTEGER_TO_F32,
            F32_TO_F16,
            F16_TO_F32,
            F32_TO_BF16,
        ] {
            let baseline = super::matched_baseline_function(class).unwrap();
            assert!(source.contains(&format!("kernel void {baseline}")));
        }
    }

    #[test]
    fn control_dependent_strict_compare_is_measured_as_its_closed_helper_service() {
        assert!(super::matched_baseline_function(F32_COMPARE).is_none());
        assert!(probe_source(false, false).contains("kernel void seismic_probe_f32_compare"));
    }

    #[test]
    fn bf16_widen_reuses_the_renderer_integer_shift_service() {
        let source = probe_source(true, false);
        assert!(!source.contains("seismic_probe_bf16_to_f32"));
        assert!(!source.contains("seismic_baseline_bf16_to_f32"));
        assert!(source.contains("seismic_probe_f32_to_bf16"));
        assert!(source.contains("seismic_baseline_f32_to_bf16"));
    }

    #[test]
    fn thread_batch_means_preserve_counterbalanced_raw_pairs() {
        let small = [100u64; 64];
        let mut large = [200u64; 64];
        large[1] = 90;
        let interval = paired_batch_mean_confidence_interval(&small, &large, 10, 8, 2)
            .unwrap()
            .unwrap();
        assert!(interval.lower_numerator > 0);
        assert_eq!(interval.denominator, 80);
    }

    #[test]
    fn thread_fixed_round_blocks_use_all_512_pairs() {
        let small = vec![100u64; 512];
        let large = vec![200u64; 512];
        let interval = paired_batch_mean_confidence_interval(&small, &large, 10, 64, 0)
            .unwrap()
            .unwrap();
        assert_eq!(interval, DurationInterval::new(6_400, 6_400, 640));
    }
}
