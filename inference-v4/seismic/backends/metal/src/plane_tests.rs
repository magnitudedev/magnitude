//! Execute target storage-plane views through the production Metal builder/compiler.
use crate::{compile, profile, DeviceHandle, Metal, MetalDevice};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};
use seismic_compiler::executable::DeviceService;
use seismic_ir::construction::Construction;
use seismic_ir::kernel::dynamic::PortableSliceAxis;
use seismic_ir::storage::GlobalBufferKind;
use seismic_lang::expr::ExprArena;
use seismic_lang::registry::{self, PlaneEncoding, RepresentationKind};
use seismic_lang::types::DType;

#[test]
#[ignore = "requires Metal device"]
fn plane_storage_views_preserve_types_packet_offsets_and_partial_words() {
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let target = profile::open_device(&device).unwrap();
    for name in ["q4g32", "q4g64", "q4k", "q6k", "nvfp4_e2m1_block16"] {
        let representation = registry::representation(name).unwrap();
        let RepresentationKind::Packed(layout) =
            &registry::representation_info(representation).kind
        else {
            unreachable!()
        };
        let n = u64::from(layout.group + layout.group / 2);
        let mut input = (0..layout.packet_size as usize * 2)
            .map(|i| (i as u8).wrapping_mul(73).wrapping_add(5))
            .collect::<Vec<_>>();
        for plane in &layout.planes {
            if let PlaneEncoding::Dense(dtype) = plane.encoding {
                let payload = match dtype {
                    DType::F16 => 0xfc01u32,
                    DType::BF16 => 0xff81,
                    DType::F32 => 0xff80_0035,
                    _ => 0xffff_fffd,
                };
                let start = layout.packet_size as usize + plane.offset as usize;
                for value in input[start..start + plane.bytes_per_group as usize]
                    .chunks_exact_mut(dtype.bytes() as usize)
                {
                    value.copy_from_slice(&payload.to_le_bytes()[..dtype.bytes() as usize]);
                }
            }
        }
        for slice_start in [0, u64::from(layout.group) - 1, u64::from(layout.group)] {
            let slice_end = n;
            let first_packet = slice_start / u64::from(layout.group);
            let packets = (slice_start % u64::from(layout.group) + slice_end - slice_start)
                .div_ceil(u64::from(layout.group));
            let mut expected = Vec::new();
            let counts = layout
                .planes
                .iter()
                .map(|plane| packets * u64::from(plane.storage_elements_per_packet()))
                .collect::<Vec<_>>();
            let total = counts.iter().map(|n| n + u64::from(*n > 1)).sum::<u64>();

            let mut arena = ExprArena::default();
            let mut construction = Construction::<Metal>::new(
                &mut arena,
                vec![],
                false,
                target.addressable_resources().len(),
            );
            let n_expr = arena.nat(n);
            let output_n = arena.nat(total);
            let (_, input_view) = construction.storage_mut().tensor(
                &mut arena,
                GlobalBufferKind::Arena,
                representation,
                vec![n_expr],
            );
            let (_, output_view) = construction.storage_mut().tensor(
                &mut arena,
                GlobalBufferKind::Arena,
                registry::dense(DType::U32),
                vec![output_n],
            );
            let input_view = construction.view(input_view, representation);
            let output_view = construction.view(output_view, registry::dense(DType::U32));
            let mut builder = construction.portable_kernel(
                &mut arena,
                target.facts(),
                target.addressable_resources(),
                target.vectors(),
            );
            let input_place = builder.arg_view(input_view, false);
            let output_place = builder.arg_view(output_view, true);
            let tensor = builder.tensor(input_place);
            let start = builder.index_constant(slice_start);
            let end = builder.index_constant(slice_end);
            let packet_tail =
                builder.tensor_slice(tensor, vec![PortableSliceAxis::Range { start, end }]);
            let mut out_index = 0;
            for (ordinal, (plane, &count)) in layout.planes.iter().zip(&counts).enumerate() {
                let view = builder.tensor_plane(packet_tail.clone(), ordinal as u32);
                for index in 0..count {
                    let at = builder.index_constant(index);
                    let value = builder.tensor_read(&view, &[at]);
                    let value = builder.scalar_bits(value);
                    let out = builder.index_constant(out_index);
                    builder.write(output_place, &[out], value);
                    out_index += 1;
                    let mut bytes = [0u8; 4];
                    let per_packet = u64::from(plane.storage_elements_per_packet());
                    let packet = first_packet + index / per_packet;
                    let offset =
                        (index % per_packet) as usize * plane.storage_element_bytes() as usize;
                    for byte in 0..plane.storage_element_bytes() as usize {
                        if offset + byte < plane.bytes_per_group as usize {
                            bytes[byte] = input[packet as usize * layout.packet_size as usize
                                + plane.offset as usize
                                + offset
                                + byte];
                        }
                    }
                    expected.push(u32::from_le_bytes(bytes));
                }
                if count > 1 {
                    let start = builder.index_constant(1);
                    let end = builder.index_constant(count);
                    let sliced =
                        builder.tensor_slice(view, vec![PortableSliceAxis::Range { start, end }]);
                    let zero = builder.index_constant(0);
                    let value = builder.tensor_read(&sliced, &[zero]);
                    let value = builder.scalar_bits(value);
                    let out = builder.index_constant(out_index);
                    builder.write(output_place, &[out], value);
                    out_index += 1;
                    expected.push(expected[expected.len() - count as usize + 1]);
                }
            }
            builder.close();
            let kernel = &construction.kernels()[0];
            let emission = target.kernel_emission_layout(kernel);
            let native =
                compile::compile_kernel(device.handle(), &target, kernel, &emission).unwrap();
            let mut words = vec![1u64; emission.words.total as usize];
            for (binding, extent) in emission.words.bindings.iter().zip([n, total]) {
                words[binding.first as usize] = extent;
                words[binding.first as usize + 1] = 1;
            }
            let input_buffer = device.allocate_bytes(input.len() as u64).unwrap();
            device.write(&input_buffer, 0, &input).unwrap();
            let output_buffer = device.allocate_bytes(total * 4).unwrap();
            let word_bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            let word_buffer = device.allocate_bytes(word_bytes.len() as u64).unwrap();
            device.write(&word_buffer, 0, &word_bytes).unwrap();
            let scratch = device.allocate_bytes(8).unwrap();
            let command = device.queue().commandBuffer().unwrap();
            let encoder = command.computeCommandEncoder().unwrap();
            encoder.setComputePipelineState(&native.pipeline.state);
            for (index, buffer) in [
                &input_buffer,
                &output_buffer,
                &word_buffer,
                &scratch,
                &scratch,
                &scratch,
            ]
            .into_iter()
            .enumerate()
            {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer.raw()), 0, index);
                }
            }
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(16, 0);
            }
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "Metal plane kernel failed: {:?}",
                command.error()
            );
            let mut bytes = vec![0u8; total as usize * 4];
            device.read(&output_buffer, 0, &mut bytes).unwrap();
            let output = bytes
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                output, expected,
                "{name} target plane storage projection [{slice_start}:{slice_end}]"
            );
        }
    }
}

#[test]
#[ignore = "requires Metal device"]
fn native_subgroup_ordinal_uses_the_actual_two_dimensional_cohort() {
    use seismic_ir::kernel::ops::{BinaryOp, ValueType};
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let target = profile::open_device(&device).unwrap();
    let mut arena = ExprArena::default();
    let count = arena.nat(128);
    let mut construction = Construction::<Metal>::new(
        &mut arena,
        vec![],
        false,
        target.addressable_resources().len(),
    );
    let (_, output) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        registry::dense(DType::U32),
        vec![count],
    );
    let output = construction.view(output, registry::dense(DType::U32));
    let mut builder = construction.portable_kernel(
        &mut arena,
        target.facts(),
        target.addressable_resources(),
        target.vectors(),
    );
    let output = builder.arg_view(output, true);
    let x = builder.local_id(0);
    let y = builder.local_id(1);
    let width = builder.workgroup_size(0);
    let row = builder.binary(BinaryOp::Mul, y, width);
    let linear = builder.binary(BinaryOp::Add, row, x);
    let ordinal = builder.subgroup_ordinal();
    let ordinal = builder.cast(ordinal, ValueType::Scalar(DType::U32));
    let two = builder.index_constant(2);
    let linear = builder.binary(BinaryOp::Mul, linear, two);
    builder.write(output, &[linear], ordinal);
    let one = builder.index_constant(1);
    let next = builder.binary(BinaryOp::Add, linear, one);
    let subgroup_size = builder.subgroup_size();
    let subgroup_size = builder.cast(subgroup_size, ValueType::Scalar(DType::U32));
    builder.write(output, &[next], subgroup_size);
    builder.close();
    let kernel = &construction.kernels()[0];
    let layout = target.kernel_emission_layout(kernel);
    let native = compile::compile_kernel(device.handle(), &target, kernel, &layout).unwrap();
    let mut words = vec![1u64; layout.words.total as usize];
    words[layout.words.bindings[0].first as usize] = 128;
    let word_bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let table = device.allocate_bytes(word_bytes.len() as u64).unwrap();
    device.write(&table, 0, &word_bytes).unwrap();
    let output = device.allocate_bytes(128 * 4).unwrap();
    let scratch = device.allocate_bytes(8).unwrap();
    let command = device.queue().commandBuffer().unwrap();
    let encoder = command.computeCommandEncoder().unwrap();
    encoder.setComputePipelineState(&native.pipeline.state);
    for (index, buffer) in [&output, &table, &scratch, &scratch, &scratch]
        .into_iter()
        .enumerate()
    {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buffer.raw()), 0, index);
        }
    }
    unsafe {
        encoder.setThreadgroupMemoryLength_atIndex(16, 0);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        },
    );
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none(), "{:?}", command.error());
    let mut bytes = vec![0u8; 128 * 4];
    device.read(&output, 0, &mut bytes).unwrap();
    let words = bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let subgroup_width = native.pipeline.state.threadExecutionWidth() as u32;
    assert_eq!(
        words,
        (0..64)
            .flat_map(|i| [i / subgroup_width, subgroup_width])
            .collect::<Vec<_>>()
    );
}
