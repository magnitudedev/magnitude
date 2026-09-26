//! Execute target storage-plane views through the production typed builder/JIT.
use crate::{compile, open_host, workers::LaunchFrame, Cpu};
use seismic_ir::construction::Construction;
use seismic_ir::kernel::dynamic::PortableSliceAxis;
use seismic_ir::storage::GlobalBufferKind;
use seismic_lang::expr::ExprArena;
use seismic_lang::registry::{self, PlaneEncoding, RepresentationKind};
use seismic_lang::types::DType;

#[test]
fn plane_storage_views_preserve_types_packet_offsets_and_partial_words() {
    let opened = open_host().unwrap();
    let target = &opened.device;
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
            let mut output = vec![0u32; total as usize];
            let mut arena = ExprArena::default();
            let mut construction = Construction::<Cpu>::new(
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
            let native = compile::compile_kernel(target, kernel, &emission).unwrap();
            let mut words = vec![1u64; emission.words.total as usize];
            for (binding, extent) in emission.words.bindings.iter().zip([n, total]) {
                words[binding.first as usize] = extent;
                words[binding.first as usize + 1] = 1;
            }
            let buffers = [input.as_mut_ptr(), output.as_mut_ptr().cast::<u8>()];
            let frame = LaunchFrame {
                buffers: buffers.as_ptr(),
                words: words.as_ptr(),
                results: std::ptr::null_mut(),
            };
            unsafe {
                (native.kernel.entry)(
                    &frame,
                    std::ptr::null(),
                    0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
            }
            assert_eq!(
                output, expected,
                "{name} target plane storage projection [{slice_start}:{slice_end}]"
            );
        }
    }
}

#[test]
fn rectangular_transpose_reshape_preserves_read_and_write_coordinates() {
    use seismic_ir::kernel::ops::{ConstantValue, ValueType};
    let opened = open_host().unwrap();
    let target = &opened.device;
    let mut arena = ExprArena::default();
    let mut construction = Construction::<Cpu>::new(
        &mut arena,
        vec![],
        false,
        target.addressable_resources().len(),
    );
    let two = arena.nat(2);
    let three = arena.nat(3);
    let six = arena.nat(6);
    let representation = registry::dense(DType::F32);
    let (_, source) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        representation,
        vec![two, three],
    );
    let (_, output) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        representation,
        vec![six],
    );
    let source = construction.view(source, representation);
    let output = construction.view(output, representation);
    let mut builder = construction.portable_kernel(
        &mut arena,
        target.facts(),
        target.addressable_resources(),
        target.vectors(),
    );
    let source = builder.arg_view(source, true);
    let output = builder.arg_view(output, true);
    let tensor = builder.tensor(source);
    let transposed = builder.tensor_transpose(tensor, vec![1, 0]);
    let extent = builder.index_constant(6);
    let reshaped = builder.tensor_reshape(transposed, vec![extent]);
    let one = builder.index_constant(1);
    let nine = builder.constant(ConstantValue::F32(9.0), ValueType::Scalar(DType::F32));
    builder.tensor_write(&reshaped, &[one], nine);
    for index in 0..6 {
        let index = builder.index_constant(index);
        let value = builder.tensor_read(&reshaped, &[index]);
        builder.write(output, &[index], value);
    }
    builder.close();
    let kernel = &construction.kernels()[0];
    let emission = target.kernel_emission_layout(kernel);
    let native = compile::compile_kernel(target, kernel, &emission).unwrap();
    let mut words = vec![0u64; emission.words.total as usize];
    for (binding, fields) in emission
        .words
        .bindings
        .iter()
        .zip([vec![2, 3, 3, 1], vec![6, 1]])
    {
        words[binding.first as usize..binding.first as usize + fields.len()]
            .copy_from_slice(&fields);
    }
    let mut input = [1f32, 2., 3., 4., 5., 6.];
    let mut output = [0f32; 6];
    let buffers = [
        input.as_mut_ptr().cast::<u8>(),
        output.as_mut_ptr().cast::<u8>(),
    ];
    let frame = LaunchFrame {
        buffers: buffers.as_ptr(),
        words: words.as_ptr(),
        results: std::ptr::null_mut(),
    };
    unsafe {
        (native.kernel.entry)(
            &frame,
            std::ptr::null(),
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }
    assert_eq!(
        input,
        [1., 2., 3., 9., 5., 6.],
        "write must reach transposed backing coordinate"
    );
    assert_eq!(
        output,
        [1., 9., 2., 5., 3., 6.],
        "reshape follows logical source flatten order"
    );
}
