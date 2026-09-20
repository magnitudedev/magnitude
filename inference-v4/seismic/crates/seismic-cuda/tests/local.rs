//! Unit tests that do not require a CUDA device.
//! The device execution test is `#[ignore]`d and runs only where a CUDA
//! device is present.

use seismic_cuda::physical::{
    cuda_target_profile, elaborate, AtomicMode, CudaDialect, CudaOp, MathMode, StrategyFlags,
};
use seismic_lang::{
    logical::{self, EffectiveTargetIdentity, LogicalProgram},
    program::{compile, SourceFile},
    types::Elem,
};
use seismic_realization::executable::{self, EffectiveTargetProfile, ExecutableDialect};
use std::collections::BTreeMap;

fn check(sources: &[(&str, &str)]) -> Result<seismic_lang::sir::Program, String> {
    let files: Vec<SourceFile> = sources
        .iter()
        .map(|(path, text)| SourceFile {
            path: path.to_string(),
            text: text.to_string(),
        })
        .collect();
    compile(&files).map_err(|d| d.iter().map(|d| d.render()).collect::<Vec<_>>().join("\n"))
}

fn target() -> EffectiveTargetProfile {
    let limits = seismic_cuda::mapping::Limits::gb10();
    cuda_target_profile(
        &limits,
        &seismic_cuda::target::TargetProfile::synthetic_baseline(
            seismic_cuda::target::TargetLimits {
                max_threads_per_block: limits.max_threads_per_block,
                max_grid_x: limits.max_grid_x,
                warp_size: limits.warp_size,
                max_scratch_bytes: limits.max_scratch_bytes,
            },
        ),
    )
}

fn supports_all(_: &seismic_lang::sir::IntrinsicUse) -> Result<(), String> {
    Ok(())
}

fn logical_target() -> EffectiveTargetIdentity {
    EffectiveTargetIdentity {
        backend: "cuda".into(),
        capability_fingerprint: "test-fingerprint".into(),
    }
}

fn construct(sources: &[(&str, &str)], entry: &str) -> Result<LogicalProgram, String> {
    let program = check(sources)?;
    logical::construct(
        &program,
        entry,
        &logical_target(),
        &supports_all,
        BTreeMap::from([("M".to_string(), 4), ("N".to_string(), 8)]),
        BTreeMap::new(),
    )
    .map_err(|error| format!("{error:?}"))
}

#[test]
fn universal_legalization_covers_the_portable_matrix() {
    let profile = target();
    let cases: Vec<seismic_lang::logical::PrimitiveOp> = vec![
        seismic_lang::logical::PrimitiveOp::Constant(seismic_lang::sir::Literal::Int(1)),
        seismic_lang::logical::PrimitiveOp::RuntimeExtent(seismic_lang::types::RuntimeExtentId(0)),
        seismic_lang::logical::PrimitiveOp::Primitive(
            seismic_lang::intrinsics::PrimitiveId::Fill {
                value: 0.0,
                dtype: seismic_lang::types::DType::F32,
            },
        ),
        seismic_lang::logical::PrimitiveOp::Primitive(
            seismic_lang::intrinsics::PrimitiveId::ElementWrite { arity: 2 },
        ),
        seismic_lang::logical::PrimitiveOp::Primitive(
            seismic_lang::intrinsics::PrimitiveId::Atomic {
                op: seismic_lang::intrinsics::AtomicOp::Add,
                arity: 1,
            },
        ),
        seismic_lang::logical::PrimitiveOp::Primitive(seismic_lang::intrinsics::PrimitiveId::Math(
            seismic_lang::intrinsics::MathOp::Exp,
        )),
    ];
    for op in cases {
        let primitive = executable::PhysicalPrimitive {
            op,
            inputs: vec![seismic_lang::types::ValueType::Scalar(
                seismic_lang::types::DType::F32,
            )],
            results: vec![seismic_lang::types::ValueType::Scalar(
                seismic_lang::types::DType::F32,
            )],
        };
        let legalized = <CudaDialect as ExecutableDialect>::legalize(&primitive, &profile);
        assert!(
            legalized.ops().is_some(),
            "a universal primitive legalized as inapplicable"
        );
    }
}

#[test]
fn matrix_capability_stays_absent_until_complete() {
    let profile = target();
    let matrix = seismic_lang::intrinsics::IntrinsicId {
        capability: seismic_lang::intrinsics::CapabilityId::new("cuda", "matrix"),
        name: "matmul".into(),
    };
    let primitive = executable::PhysicalPrimitive {
        op: seismic_lang::logical::PrimitiveOp::Capability(matrix),
        inputs: Vec::new(),
        results: vec![seismic_lang::types::ValueType::Scalar(
            seismic_lang::types::DType::F32,
        )],
    };
    let legalized = <CudaDialect as ExecutableDialect>::legalize(&primitive, &profile);
    assert!(legalized.ops().is_none());
}

#[test]
fn cas_atomic_admits_only_in_allocation_word_layouts() {
    use seismic_lang::types::{DType, ExtentExpr};
    // Even f16 element counts: the last element's containing 32-bit word
    // lies inside the tensor's own allocation.
    let even = vec![ExtentExpr::Static(8)];
    assert!(seismic_cuda::physical::cas_admissible(&even, DType::F16));
    // Odd element counts: the final half's word straddles the end.
    let odd = vec![ExtentExpr::Static(7)];
    assert!(!seismic_cuda::physical::cas_admissible(&odd, DType::F16));
    // 32-bit elements always admit the word CAS.
    assert!(seismic_cuda::physical::cas_admissible(&odd, DType::F32));
}

#[test]
fn opcodes_have_exact_universal_consequences_before_solve() {
    for op in [
        CudaOp::NoOp,
        CudaOp::Const {
            dest: seismic_cuda::physical::CudaSsa(0),
            value: seismic_cuda::physical::CudaConst::Int(1),
            dtype: seismic_lang::types::DType::I32,
        },
        CudaOp::Math {
            op: seismic_lang::intrinsics::MathOp::Exp,
            arguments: vec![],
            dest: seismic_cuda::physical::CudaSsa(0),
            mode: MathMode::Software,
        },
        CudaOp::Atomic {
            op: seismic_lang::intrinsics::AtomicOp::Add,
            base: seismic_cuda::physical::CudaOperand::Value(seismic_lang::logical::GraphValueId(
                0,
            )),
            view: seismic_cuda::physical::CudaView::Identity,
            view_shape: vec![],
            indices: vec![],
            value: seismic_cuda::physical::CudaOperand::Ssa(seismic_cuda::physical::CudaSsa(0)),
            dtype: seismic_lang::types::DType::F32,
            mode: AtomicMode::Serialized,
            guard: None,
        },
    ] {
        let consequences = <CudaDialect as ExecutableDialect>::consequences(&op);
        // The universal column is exact and its native contract guarantees
        // at least one resident participant.
        assert_eq!(consequences.native_contract.max_resident_participants.0, 1);
    }
}

#[test]
fn approximate_math_carries_an_approximate_transfer() {
    let op = CudaOp::Math {
        op: seismic_lang::intrinsics::MathOp::ExpFast,
        arguments: vec![],
        dest: seismic_cuda::physical::CudaSsa(0),
        mode: MathMode::FastApprox,
    };
    let consequences = <CudaDialect as ExecutableDialect>::consequences(&op);
    assert!(matches!(
        consequences.numerical,
        seismic_realization::NumericalTransfer::Approximate { .. }
    ));
}

#[test]
fn family_covers_every_choice_with_two_alternatives() {
    let sources = [(
        "kernel.seismic",
        "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n",
    )];
    let logical = construct(&sources, "add").expect("construction succeeds");
    let family = elaborate(&logical, &target()).expect("elaboration succeeds");
    for choice in logical.choices.ids() {
        let physical = family.choices.get(choice).expect("the choice exists");
        assert!(
            physical.alternatives.iter().count() >= 2,
            "every logical alternative carries the universal and optimized columns"
        );
    }
}

#[test]
fn cost_model_identity_is_unqualified_and_explicit() {
    // Uncalibrated costs affect ranking only; the identity says so until a
    // probe-calibrated revision replaces it.
    assert_eq!(
        seismic_cuda::mapping::COST_MODEL_IDENTITY,
        "cuda-estimate-unqualified-v0"
    );
}

/// End-to-end CUDA execution on `sparky` (GB10). Ignored locally: no CUDA
/// device is present on this machine.
#[test]
#[ignore = "requires a CUDA device"]
fn device_execution_of_the_representative_kernel() {
    let sources = [(
        "kernel.seismic",
        "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n",
    )];
    let program = check(&sources).expect("the kernel checks");
    let device = seismic_cuda::Device::open(0).expect("a CUDA device is present");
    let compiler = seismic_cuda::CudaCompiler::new(&device).expect("the backend builds");
    let workload = seismic_compiler::pipeline::Workload {
        shapes: BTreeMap::from([("M".to_string(), 64), ("N".to_string(), 64)]),
        elems: BTreeMap::new(),
        precision: seismic_lang::precision::PrecisionPolicy::Unconstrained,
        extents: BTreeMap::new(),
    };
    let compiled = seismic_compiler::pipeline::compile(
        &program,
        "add",
        &workload,
        &compiler,
        &[],
        seismic_compiler::planning::Budget::default(),
    )
    .expect("compilation succeeds");
    let _ = compiled;
    // Bind ABI buffers, execute, and check the first status word is clean.
    let mut sequence = compiled.native;
    let buffers: Vec<seismic_cuda::Buffer> = (0..sequence.abi_buffers().len())
        .map(|index| {
            let bytes = sequence.abi_buffers()[index].bytes as usize;
            device.buffer(bytes).expect("the buffer allocates")
        })
        .collect();
    let status = sequence
        .execute(&buffers, &[], false)
        .expect("execution succeeds");
    assert!(status.is_none(), "no safety violation was recorded");
    let _ = Elem::Dtype(seismic_lang::types::DType::F32);
    let _ = StrategyFlags::UNIVERSAL;
}
