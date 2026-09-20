//! Targeted strategy tests: construct → elaborate → plan for representative
//! kernels, dialect totality over the portable matrix, and capability
//! alternatives offered only with exact profile facts.

use seismic_compiler::pipeline::{self, Workload};
use seismic_compiler::planning::Budget;
use seismic_lang::{
    intrinsics::PrimitiveId,
    logical::{self},
    precision::PrecisionPolicy,
    program::{self, SourceFile},
    sir::{IntrinsicUse, Program},
    types::{DType, ValueType},
};
use seismic_metal::{
    mapping::{Limits, Metal},
    physical,
};
use seismic_realization::executable::{EffectiveTargetProfile, ExecutableDialect, PlanFamily};

fn check(sources: &[(&str, &str)]) -> Result<Program, String> {
    let files: Vec<SourceFile> = sources
        .iter()
        .map(|(path, text)| SourceFile {
            path: (*path).to_string(),
            text: (*text).to_string(),
        })
        .collect();
    program::compile(&files).map_err(|diagnostics| {
        diagnostics
            .iter()
            .map(|d| d.render())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn supports_all(_: &IntrinsicUse) -> Result<(), String> {
    Ok(())
}

fn target() -> logical::EffectiveTargetIdentity {
    logical::EffectiveTargetIdentity {
        backend: "metal".into(),
        capability_fingerprint: "test-fingerprint".into(),
    }
}

fn shapes(entries: &[(&str, i64)]) -> std::collections::BTreeMap<String, i64> {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect()
}

fn backend() -> Metal {
    Metal::new(Limits::synthetic()).expect("the synthetic Metal target builds")
}

const NEGATE: &str = "fn negate[M, N](io: &mut tensor[M, N] f32):\n    parallel for row in 0..M:\n        for col in 0..N:\n            io[row, col] = 0.0 - io[row, col]\n    return\n";

const SUM_UNORDERED: &str =
    "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(load(x), 0, sum, unordered=true)\n";

const SUM_ORDERED: &str =
    "fn sum[N](x: &tensor[N] f32) -> f32:\n    return reduce(load(x), 0, sum)\n";

fn compile_kernel(
    source: &str,
    entry: &str,
    shape_entries: &[(&str, i64)],
) -> Result<
    seismic_compiler::pipeline::Compiled<
        seismic_metal::physical::MetalDialect,
        seismic_metal::msl::Emitted,
    >,
    String,
> {
    let program = check(&[("kernel.seismic", source)])?;
    let workload = Workload {
        shapes: shapes(shape_entries),
        elems: Default::default(),
        precision: PrecisionPolicy::default(),
        extents: Default::default(),
    };
    pipeline::compile(
        &program,
        entry,
        &workload,
        &backend(),
        &[],
        Budget::default(),
    )
    .map_err(|error| error.to_string())
}

#[test]
fn dialect_legalizes_the_portable_matrix_totally() {
    // Every registry primitive (except the capability route) legalizes to a
    // nonempty opcode set; a universal `Inapplicable` would be a compiler bug.
    let profile: EffectiveTargetProfile = executable_profile();
    let dense = |dtype: DType| {
        ValueType::Tensor(seismic_lang::types::TensorType::new(
            vec![seismic_lang::types::ExtentExpr::Static(4)],
            seismic_lang::types::Elem::Dtype(dtype),
        ))
    };
    let scalar = |dtype: DType| ValueType::Scalar(dtype);
    for signature in seismic_lang::intrinsics::primitives() {
        let id = signature.id.clone();
        let mut inputs: Vec<ValueType> = signature
            .parameters
            .iter()
            .map(|_| ValueType::Scalar(DType::F32))
            .collect();
        if inputs.is_empty() {
            // Variadic payload primitives: a representative place/index/
            // value operand list.
            inputs = match &id {
                seismic_lang::intrinsics::PrimitiveId::Atomic { .. } => {
                    vec![dense(DType::F32), scalar(DType::I32), scalar(DType::F32)]
                }
                seismic_lang::intrinsics::PrimitiveId::ElementRead { .. }
                | seismic_lang::intrinsics::PrimitiveId::ElementWrite { .. } => {
                    vec![dense(DType::F32), scalar(DType::I32)]
                }
                _ => vec![dense(DType::F32)],
            };
        }
        let physical = seismic_realization::executable::PhysicalPrimitive {
            op: seismic_lang::logical::PrimitiveOp::Primitive(id.clone()),
            inputs: inputs.clone(),
            results: vec![inputs.first().cloned().unwrap_or(scalar(DType::F32))],
        };
        let legalized = seismic_metal::physical::MetalDialect::legalize(&physical, &profile);
        if matches!(id, seismic_lang::intrinsics::PrimitiveId::Reduce { .. }) {
            // Reductions are ReductionNodes consumed by `map_reduction`;
            // reaching scalar legalization is the guarded compiler bug.
            assert!(legalized.ops().is_none());
        } else {
            assert!(
                legalized.ops().is_some(),
                "primitive `{}` failed universal legalization",
                id.name()
            );
        }
    }
    // Dense operand classification for arithmetic and cast.
    let arithmetic = seismic_realization::executable::PhysicalPrimitive {
        op: seismic_lang::logical::PrimitiveOp::Primitive(PrimitiveId::Binary(
            seismic_lang::syntax::ast::BinaryOp::Add,
        )),
        inputs: vec![dense(DType::F32), dense(DType::F32)],
        results: vec![dense(DType::F32)],
    };
    assert!(
        seismic_metal::physical::MetalDialect::legalize(&arithmetic, &profile)
            .ops()
            .is_some()
    );
    // Capability applications are the capability route, never universal.
    let capability = seismic_realization::executable::PhysicalPrimitive {
        op: seismic_lang::logical::PrimitiveOp::Capability(seismic_lang::intrinsics::IntrinsicId {
            capability: seismic_lang::intrinsics::CapabilityId::new("metal", "subgroup"),
            name: "simd_sum".into(),
        }),
        inputs: vec![scalar(DType::F32)],
        results: vec![scalar(DType::F32)],
    };
    assert!(
        seismic_metal::physical::MetalDialect::legalize(&capability, &profile)
            .ops()
            .is_none()
    );
}

#[test]
fn representative_kernel_constructs_elaborates_plans_and_emits() {
    let compiled = compile_kernel(NEGATE, "negate", &[("M", 4), ("N", 3)])
        .expect("the representative kernel compiles end to end");
    let emitted = &compiled.native;
    // One root ABI with exactly the parameter buffer; no internal arena
    // (every storage is the caller's), no alias tables, no scratch.
    assert_eq!(emitted.abi.buffers.len(), 1);
    assert_eq!(emitted.arena_bytes, 0);
    assert!(emitted.launches.iter().any(|launch| launch
        .bindings
        .iter()
        .any(|binding| matches!(binding, seismic_metal::msl::LaunchBinding::Buffer { .. }))));
    // The structured execution tree: the leading constant segment as one
    // launch, then Repeat(rows) { Repeat(cols) { Launch } }; dynamic control
    // is never flattened away.
    match &emitted.execution[..] {
        [seismic_metal::msl::ExecutionItem::Launch(_), seismic_metal::msl::ExecutionItem::Repeat { body, .. }] => {
            match &body[..] {
                [_, seismic_metal::msl::ExecutionItem::Repeat { body, .. }] => {
                    assert!(body
                        .iter()
                        .all(|item| matches!(item, seismic_metal::msl::ExecutionItem::Launch(_))));
                }
                other => panic!("the inner schedule is a launch then a repeat, got {other:?}"),
            }
        }
        other => panic!("the entry schedule is a launch then a repeat, got {other:?}"),
    }
    // Three distinct launches (the schedule tree repeats them): the root
    // constant segment, the per-row constant segment, and the per-point
    // read/negate/write segment.
    assert_eq!(emitted.launches.len(), 3);
    // The MSL library renders every opcode.
    assert!(emitted.source.contains("kernel void seismic_metal_0"));
}

#[test]
fn cost_model_carries_a_measured_identity() {
    let identity = physical::cost_model_identity();
    assert!(identity.contains("probe-calibrated"));
    assert!(identity.contains("m4max"));
    // The same identity is fingerprinted into the effective target profile.
    assert!(backend().target_profile().fingerprint().contains(identity));
}

#[test]
fn subgroup_alternative_is_offered_only_with_exact_profile_facts() {
    let program = check(&[("sum.seismic", SUM_UNORDERED)]).expect("checks");
    let logical = logical::construct(
        &program,
        "sum",
        &target(),
        &supports_all,
        shapes(&[("N", 64)]),
        Default::default(),
    )
    .expect("constructs");
    // Profile with the exact simd_sum(f32) signature: the family offers the
    // universal alternative plus the subgroup alternative.
    let admitted = backend();
    let family = physical::elaborate(
        &logical,
        admitted.executable_profile(),
        physical::StrategyLimits {
            max_participants: 1024,
        },
    )
    .expect("elaborates");
    let entry_alternatives = alternatives_of(&family);
    assert_eq!(entry_alternatives, 2, "universal + subgroup");
    // Profile without the exact signature: only the universal alternative.
    let bare = bare_profile();
    let family = physical::elaborate(
        &logical,
        &bare,
        physical::StrategyLimits {
            max_participants: 1024,
        },
    )
    .expect("elaborates");
    assert_eq!(alternatives_of(&family), 1, "universal only");
}

#[test]
fn ordered_reduction_keeps_only_the_universal_alternative() {
    let program = check(&[("sum.seismic", SUM_ORDERED)]).expect("checks");
    let logical = logical::construct(
        &program,
        "sum",
        &target(),
        &supports_all,
        shapes(&[("N", 64)]),
        Default::default(),
    )
    .expect("constructs");
    let admitted = backend();
    let family = physical::elaborate(
        &logical,
        admitted.executable_profile(),
        physical::StrategyLimits {
            max_participants: 1024,
        },
    )
    .expect("elaborates");
    // The ordered reduction never reassociates: no subgroup alternative.
    assert_eq!(alternatives_of(&family), 1);
}

fn alternatives_of(family: &PlanFamily<physical::MetalDialect>) -> usize {
    family
        .choices
        .get(family.entry)
        .map(|choice| choice.alternatives.len())
        .unwrap_or(0)
}

fn executable_profile() -> EffectiveTargetProfile {
    backend().executable_profile().clone()
}

fn bare_profile() -> EffectiveTargetProfile {
    let limits = Limits {
        max_threads_per_threadgroup: 1024,
        max_threadgroup_bytes: 32 * 1024,
        max_private_bytes: 128 * 1024,
    };
    let target_profile = seismic_metal::target::TargetProfile::from_evidence(
        "bare",
        &[],
        &[],
        &[],
        1024,
        32 * 1024,
        u64::MAX,
        128 * 1024,
    );
    EffectiveTargetProfile {
        backend: "metal".into(),
        capability_fingerprint: target_profile.fingerprint().into(),
        toolchain_fingerprint: "bare".into(),
        effective_signatures: target_profile.effective_signatures(),
        limits: seismic_realization::executable::TargetLimits {
            max_participants: limits.max_threads_per_threadgroup as i64,
            max_workgroups_axis: [65_535; 3],
            max_workgroup_bytes: limits.max_threadgroup_bytes as i64,
            max_explicit_private_bytes: limits.max_private_bytes as i64,
            max_direct_bindings: 28,
            max_argument_table_bytes: 1 << 16,
            max_device_bytes: i64::MAX,
        },
    }
}

#[test]
fn resolved_plan_has_one_root_abi_and_no_unplanned_storage() {
    let compiled = compile_kernel(NEGATE, "negate", &[("M", 4), ("N", 3)]).expect("compiles");
    let plan = &compiled.physical;
    // Only root ABI storages; no device-arena, workgroup, or participant
    // storage was invented by the emitter or the strategy.
    for (_, storage) in plan.storage.ids().zip(plan.storage.iter()) {
        assert!(
            matches!(
                storage.placement,
                seismic_realization::executable::ResolvedStoragePlacement::Abi { .. }
            ),
            "unexpected non-ABI storage placement {:?}",
            storage.placement
        );
    }
    assert_eq!(plan.internal_arena.bytes, 0);
    // The structured resolved schedule retains the loop repeats.
    assert!(plan.entry.schedule.steps.iter().any(|step| matches!(
        step,
        seismic_realization::executable::ResolvedStep::Repeat(_)
    )));
    assert!(plan.entry.schedule.steps.iter().any(|step| matches!(
        step,
        seismic_realization::executable::ResolvedStep::Repeat(_)
    )));
    let logical_identity = plan.identity.logical.clone();
    assert_eq!(logical_identity, compiled.logical.identity);
}
