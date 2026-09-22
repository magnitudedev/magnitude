//! Host-only composition fixture: the production builders, traversal, Metal
//! transfer and emitter all consume the same kernel, without opening a device.
use crate::services::{CONTROL, F32_ADD_SUB, SERVICE_SUBMISSION};
use crate::{Metal, MetalFacts};
use seismic_estimator::*;
use seismic_ir::{
    construction::Construction,
    kernel::{
        ops::{BinaryOp, ClosedOpView, ConstantValue, ValueType},
        Kernel,
    },
    schedule::{Launch, LaunchMode},
    storage::LaunchLocalLayout,
    target::*,
};
use seismic_lang::{
    expr::{Assignment, ExprArena},
    types::DType,
};

fn facts() -> MetalFacts {
    MetalFacts {
        device_name: "fixture".into(),
        architecture: "fixture".into(),
        operating_system: "fixture".into(),
        registry_id: 0,
        unified_memory: true,
        families: Default::default(),
        language: crate::facts::LanguageVersion::V3_0,
        max_threads_per_threadgroup: [1; 3],
        max_threadgroup_bytes: 1024,
        max_buffer_bytes: 1024,
        buffer_alignment: 16,
        scalar_collective_dtypes: Default::default(),
        bfloat_arithmetic: false,
        matrix_dtypes: Default::default(),
        matrix_combinations: Default::default(),
        argument_table_entries: 31,
        reserved_argument_entries: 4,
        backend_revision: "fixture",
    }
}
fn emission(kernel: &Kernel<Metal>) -> KernelEmissionLayout {
    KernelEmissionLayout {
        words: KernelWordLayout::for_kernel(kernel),
        bindings: vec![],
        locals: vec![],
        addressable_resources: vec![],
        scalar_args: vec![],
        result_slots: vec![],
    }
}
struct Model {
    definitions: Vec<ServiceDefinition>,
}
impl ServiceModel for Model {
    fn service(&self, class: ServiceClassId) -> &ServiceDefinition {
        self.definitions.iter().find(|d| d.class == class).unwrap()
    }
    fn maximum_relative_error_basis_points(&self) -> u16 {
        0
    }
}
impl ExecutionModel<Metal> for Model {
    fn emission_layout(&self, kernel: &Kernel<Metal>) -> KernelEmissionLayout {
        emission(kernel)
    }
    fn operation_cost(
        &self,
        arena: &mut ExprArena,
        _: &Kernel<Metal>,
        _: &KernelEmissionLayout,
        _: &Launch,
        _: &LaunchLocalLayout,
        op: ClosedOpView<'_, Metal>,
    ) -> OperationCost {
        seismic_estimator_metal::operation_cost(arena, op)
            .map_services(|service| ServiceClassId::new(service.stable_name()))
    }
}
#[test]
fn same_closed_kernel_reaches_estimation_and_metal_emission_without_a_device() {
    let model = Model {
        definitions: [SERVICE_SUBMISSION, CONTROL, F32_ADD_SUB]
            .into_iter()
            .map(|class| ServiceDefinition {
                class,
                correlation: ServiceCorrelationId::new("fixture"),
                qualification: ServiceQualificationDomain {
                    minimum_units: 1,
                    maximum_units: 1,
                    maximum_concurrent_uses: 1,
                },
                accuracy: ServiceAccuracyClass::Compute,
                topology: ResourceTopology {
                    resources: 1,
                    max_concurrency: 1,
                },
                dependency_latency: DurationInterval::new(7, 7, 1),
                saturated_capacity: ServiceCurve {
                    setup: DurationInterval::new(0, 0, 1),
                    regimes: vec![],
                },
                provenance: FactProvenance::Derived {
                    rule: "synthetic integration coefficient",
                    inputs: Box::new([]),
                },
            })
            .collect(),
    };
    let mut arena = ExprArena::default();
    let mut c = Construction::<Metal>::new(&mut arena, vec![], false, 0);
    let facts = facts();
    let vectors = VectorSupport::default();
    let mut builder = c.portable_kernel(&mut arena, &facts, &[], &vectors);
    let a = builder.constant(ConstantValue::F32(2.0), ValueType::Scalar(DType::F32));
    let b = builder.constant(ConstantValue::F32(3.0), ValueType::Scalar(DType::F32));
    builder.binary(BinaryOp::Add, a, b);
    let id = builder.close();
    let one = arena.nat(1);
    let empty = arena.bool(false);
    let mut builder = c.schedule(&mut arena, 0);
    let launch = builder.launch(Launch {
        kernel: id,
        mode: LaunchMode::Independent,
        grid: [one; 3],
        workgroup: [one; 3],
        empty,
        parallel_extent: None,
        logical_base: None,
    });
    builder.step_launch(launch);
    let token = builder.close();
    let analyzed = c.close(token).analyze_allocations();
    let planned = analyzed.apply_allocation_plan(
        &mut arena,
        seismic_ir::construction::AllocationPlan::distinct(),
    );
    let executable = planned.finish().close_execution(&mut arena);
    let prediction = estimate(&model, &mut arena, executable.view());
    let duration = arena
        .eval_duration(prediction.estimate(), &Assignment::new())
        .unwrap();
    assert_eq!(
        duration.upper().numerator(),
        28 * u128::from(duration.upper().denominator())
    );
    let kernel = executable.kernels().kernel(id);
    let rendered = crate::render::render(0, kernel, &emission(kernel));
    assert!(rendered.source.contains("f32_add("));
    assert!(prediction
        .contributions()
        .iter()
        .any(|term| term.class() == F32_ADD_SUB));
}

#[test]
fn checked_source_refines_without_a_profile_or_native_context() {
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "refinement-fixture.seismic".into(),
        text: "fn inner[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        output[i] = x[i] * factor\n\nfn scale[N](x: &tensor[N] f32, factor: f32, output: &mut tensor[N] f32):\n    inner(x, factor, output)\n".into(),
    }])).unwrap();
    let entry = module
        .entry(
            module.entry_named("scale").unwrap(),
            &seismic_lang::entry::ElementBindings::new(),
        )
        .unwrap();
    let parts = entry.into_parts();
    let mut arena = parts.arena;
    let target = crate::profile::device_description_from_facts(facts()).unwrap();
    let constants = seismic_compiler::target::bind_target_constants(&target, &mut arena);
    let budget = seismic_compiler::preparation_budget::PreparationBudget::default();
    let precision = Default::default();
    let refined = seismic_compiler::refinement::RefinementSession::new(
        seismic_compiler::refinement::RefinementLimits::from(&budget),
    )
    .refine(
        arena,
        seismic_compiler::refinement::RefinementRequest {
            program: &parts.program,
            schema: &parts.schema,
            target: &target,
            registry: crate::profile::registry(),
            constants: &constants,
            precision: &precision,
        },
    )
    .unwrap();
    assert!(refined.universal().kernels().kernels().next().is_some());
    assert!(!refined.universal().schedule().launches().is_empty());
    assert!(refined
        .optimized()
        .iter()
        .all(|family| family.kernels().kernels().next().is_some()));
    assert_eq!(
        refined.report().completion,
        seismic_compiler::refinement::RefinementCompletion::Complete
    );
}
