//! Native matrix operands preserve physical plane projection through views.
use crate::{compile, profile, DeviceHandle, Metal, MetalDevice};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use seismic_compiler::executable::DeviceService;
use seismic_ir::construction::{AllocationPlan, Construction};
use seismic_ir::kernel::dynamic::PortableSliceAxis;
use seismic_ir::kernel::ops::{
    LogicalViewStep, SegmentLaunchDomain, SemanticIntrinsicCall,
    SemanticIntrinsicOperand as Operand, SemanticIntrinsicSink,
};
use seismic_ir::schedule::{Launch, LaunchParticipation};
use seismic_ir::storage::GlobalBufferKind;
use seismic_lang::expr::{Assignment, ExprArena};
use seismic_lang::registry::{self, IntrinsicResultType, OperandCategory, RepresentationKind};
use seismic_lang::types::DType;

#[test]
#[ignore = "requires Metal device"]
fn matrix_operand_preserves_pre_and_post_plane_views() {
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let target = profile::open_device(&device).unwrap();
    let representation = registry::representation("q8g32s").unwrap();
    let RepresentationKind::Packed(layout) = &registry::representation_info(representation).kind
    else {
        unreachable!()
    };
    let (plane, scale) = layout
        .planes
        .iter()
        .enumerate()
        .find(|(_, p)| p.name == "scale")
        .unwrap();
    let dtype = scale.storage_dtype;
    let capability = registry::capability(registry::BackendName::Metal, "matrix").unwrap();
    let signature = registry::intrinsics(capability).iter().find(|s| s.name == "matmul"
        && matches!(s.arguments[0].category, OperandCategory::Readable { representation, .. } if representation == registry::dense(dtype))
        && matches!(s.result, IntrinsicResultType::Owned { representation, .. } if representation == registry::dense(DType::F32))).unwrap();
    let mut arena = ExprArena::default();
    let mut construction = Construction::<Metal>::new(
        &mut arena,
        vec![],
        false,
        target.addressable_resources().len(),
    );
    let ten = arena.nat(10);
    let eight = arena.nat(8);
    let width = arena.nat(u64::from(layout.group) * 8);
    let one = arena.nat(1);
    let thirty_two = arena.nat(32);
    let empty = arena.bool(false);
    let (_, input_view) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        representation,
        vec![ten, width],
    );
    let (_, output_view) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        registry::dense(DType::F32),
        vec![eight, eight],
    );
    let input_view = construction.view(input_view, representation);
    let output_view = construction.view(output_view, registry::dense(DType::F32));
    let mut builder = construction.portable_kernel(
        &mut arena,
        target.facts(),
        target.addressable_resources(),
        target.vectors(),
    );
    let input_place = builder.arg_view(input_view, false);
    let output_place = builder.arg_view(output_view, true);
    let tensor = builder.tensor(input_place);
    let start = builder.index_constant(1);
    let end = builder.index_constant(9);
    let tensor = builder.tensor_slice(
        tensor,
        vec![
            PortableSliceAxis::Range { start, end },
            PortableSliceAxis::Full,
        ],
    );
    let tensor = builder.tensor_plane(tensor, plane as u32);
    let zero = builder.index_constant(0);
    let eight_value = builder.index_constant(8);
    let tensor = builder.tensor_slice(
        tensor,
        vec![
            PortableSliceAxis::Full,
            PortableSliceAxis::Range {
                start: zero,
                end: eight_value,
            },
        ],
    );
    let tensor = builder.tensor_reshape(tensor, vec![eight_value, eight_value]);
    let transpose = builder.tensor_transpose(tensor.clone(), vec![1, 0]);
    let operands = [
        Operand::Readable(builder.semantic_place(tensor, registry::dense(dtype), 2, false)),
        Operand::Readable(builder.semantic_place(transpose, registry::dense(dtype), 2, false)),
    ];
    let output_tensor = builder.tensor(output_place);
    let destination = builder.semantic_place(output_tensor, registry::dense(DType::F32), 2, true);
    let call = SemanticIntrinsicCall {
        signature,
        operands: &operands,
        destination: Some(destination),
    };
    let domain = SegmentLaunchDomain {
        mode: LaunchParticipation::Independent,
        grid: [one; 3],
        workgroup: [thirty_two, one, one],
        empty,
        parallel_extent: thirty_two,
    };
    let mut sink = SemanticIntrinsicSink::open(&mut builder, &call);
    crate::intrinsic::lower_semantic(&target, &domain, call, &mut sink);
    sink.finish();
    let id = builder.close();
    let mut schedule = construction.schedule(&mut arena, 0);
    let launch = schedule.launch(Launch {
        kernel: id,
        descriptor: crate::MetalLaunchMode,
        grid: domain.grid,
        workgroup: domain.workgroup,
        empty,
        parallel_extent: None,
        logical_base: None,
    });
    schedule.step_launch(launch);
    let closed = schedule.close();
    let executable = construction
        .close(closed)
        .normalize_launches(&mut arena, u64::MAX, 64)
        .unwrap()
        .analyze_allocations()
        .apply_allocation_plan(&mut arena, AllocationPlan::distinct())
        .finish()
        .close_execution(&mut arena, target.local_realization(), target.kernel_abi());
    let kernel = executable.kernels().kernel(id);
    let mut saw_projection = false;
    for op in &kernel.block(kernel.root()).ops {
        if let seismic_ir::kernel::ops::Op::Intrinsic {
            op: seismic_ir::metal::MetalIntrinsic::Matrix { left, right, .. },
            ..
        } = op
        {
            assert_eq!(left.representation, registry::dense(dtype));
            assert!(matches!(
                left.steps.as_slice(),
                [
                    LogicalViewStep::Slice(_),
                    LogicalViewStep::Plane { .. },
                    LogicalViewStep::Slice(_),
                    LogicalViewStep::Reshape { .. }
                ]
            ));
            assert!(matches!(
                right.steps.last(),
                Some(LogicalViewStep::Transpose(_))
            ));
            saw_projection = true;
        }
    }
    assert!(saw_projection);
    let emission = target.kernel_emission_layout(kernel);
    let native = compile::compile_kernel(device.handle(), &target, kernel, &emission).unwrap();
    let mut input = vec![0u8; 80 * layout.packet_size as usize];
    for row in 0..10 {
        for column in 0..8 {
            let at = (row * 8 + column) * layout.packet_size as usize + scale.offset as usize;
            let value = seismic_lang::reference_math::float_literal(dtype, (row + 1) as f64);
            input[at..at + dtype.bytes() as usize]
                .copy_from_slice(&value.bits().to_le_bytes()[..dtype.bytes() as usize]);
        }
    }
    let mut words = vec![1u64; emission.words.total as usize];
    for (binding, values) in emission
        .words
        .bindings
        .iter()
        .zip([[10, u64::from(layout.group) * 8, 8, 1], [8, 8, 8, 1]])
    {
        words[binding.first as usize..binding.first as usize + 4].copy_from_slice(&values);
    }
    let locals = executable.launch_resources()[0].layout();
    let values = Assignment::new();
    for (binding, local) in emission.words.locals.iter().zip(&locals.locals) {
        let first = binding.first as usize;
        words[first] = arena.eval_nat_u64(local.offset, &values).unwrap();
        for (i, expression) in local.extents.iter().chain(&local.strides).enumerate() {
            words[first + 1 + i] = arena.eval_nat_u64(*expression, &values).unwrap();
        }
    }
    let local_bytes = arena.eval_nat_u64(locals.workgroup_bytes, &values).unwrap();
    for (i, expression) in [
        locals.workgroup_bytes,
        locals.participant_bytes,
        locals.register_bytes,
    ]
    .iter()
    .enumerate()
    {
        words[emission.words.local_total_first as usize + i] =
            arena.eval_nat_u64(*expression, &values).unwrap();
    }
    words[emission.words.workgroup_first as usize] = 32;
    let input_buffer = device.allocate(input.len() as u64, 16).unwrap();
    device.write(&input_buffer, 0, &input).unwrap();
    let output_buffer = device.allocate(256, 16).unwrap();
    let word_bytes = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let word_buffer = device.allocate(word_bytes.len() as u64, 8).unwrap();
    device.write(&word_buffer, 0, &word_bytes).unwrap();
    let unused = device.allocate(16, 16).unwrap();
    let command = device.queue().commandBuffer().unwrap();
    let encoder = command.computeCommandEncoder().unwrap();
    encoder.setComputePipelineState(&native.pipeline.state);
    for (slot, buffer) in [
        &input_buffer,
        &output_buffer,
        &word_buffer,
        &unused,
        &unused,
        &unused,
    ]
    .iter()
    .enumerate()
    {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buffer.raw()), 0, slot);
        }
    }
    unsafe {
        encoder.setThreadgroupMemoryLength_atIndex(local_bytes as usize, 0);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none(), "{:?}", command.error());
    let mut output = vec![0u8; 256];
    device.read(&output_buffer, 0, &mut output).unwrap();
    for (index, bytes) in output.chunks_exact(4).enumerate() {
        assert_eq!(
            f32::from_le_bytes(bytes.try_into().unwrap()),
            (8 * (index / 8 + 2) * (index % 8 + 2)) as f32
        );
    }
}
