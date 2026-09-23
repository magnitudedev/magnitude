//! Native execution checks for typed scalar operations outside source-recipe lowering.

use crate::{compile, profile, DeviceHandle, Metal, MetalDevice};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use seismic_compiler::executable::DeviceService;
use seismic_ir::construction::Construction;
use seismic_ir::repr::{DenseI32, Representation};
use seismic_ir::storage::GlobalBufferKind;
use seismic_lang::expr::ExprArena;

#[test]
#[ignore = "requires Metal device"]
fn typed_i32_division_and_remainder_are_euclidean_at_word_extremes() {
    let cases = [
        (i32::MIN, 3),
        (i32::MIN, -3),
        (i32::MIN, i32::MIN),
        (-1, i32::MIN),
        (i32::MAX, i32::MIN),
        (-7, 3),
        (-7, -3),
        (7, -3),
    ];
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let target = profile::open_device(&device).unwrap();
    let mut arena = ExprArena::default();
    let count = arena.nat((cases.len() * 2) as u64);
    let mut construction = Construction::<Metal>::new(
        &mut arena,
        vec![],
        false,
        target.addressable_resources().len(),
    );
    let (_, input) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        DenseI32::id(),
        vec![count],
    );
    let (_, output) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        DenseI32::id(),
        vec![count],
    );
    let input = construction.typed_view::<DenseI32>(construction.view(input, DenseI32::id()));
    let output = construction.typed_view::<DenseI32>(construction.view(output, DenseI32::id()));
    let mut kernel = construction.kernel(
        &mut arena,
        target.facts(),
        target.addressable_resources(),
        target.vectors(),
    );
    let input = kernel.arg_readable(input);
    let output = kernel.arg_writable(output);
    for index in 0..cases.len() {
        let a_index = kernel.constant((index * 2) as u64);
        let b_index = kernel.constant((index * 2 + 1) as u64);
        let a = kernel.read(input, &[a_index]);
        let b = kernel.read(input, &[b_index]);
        let quotient = kernel.div(a, b);
        let remainder = kernel.rem(a, b);
        kernel.write(output, &[a_index], quotient);
        kernel.write(output, &[b_index], remainder);
    }
    kernel.close();

    let kernel = &construction.kernels()[0];
    let layout = target.kernel_emission_layout(kernel);
    let native = compile::compile_kernel(device.handle(), &target, kernel, &layout).unwrap();
    let mut words = vec![1u64; layout.words.total as usize];
    for binding in &layout.words.bindings {
        words[binding.first as usize] = (cases.len() * 2) as u64;
    }
    let input_values = cases.iter().flat_map(|&(a, b)| [a, b]).collect::<Vec<_>>();
    let input_bytes = input_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let input_buffer = device.allocate_bytes(input_bytes.len() as u64).unwrap();
    device.write(&input_buffer, 0, &input_bytes).unwrap();
    let output_buffer = device.allocate_bytes(input_bytes.len() as u64).unwrap();
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
        "Metal kernel failed: {:?}",
        command.error()
    );

    let mut actual_bytes = vec![0u8; input_bytes.len()];
    device.read(&output_buffer, 0, &mut actual_bytes).unwrap();
    let actual = actual_bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = cases
        .iter()
        .flat_map(|&(a, b)| [a.div_euclid(b), a.rem_euclid(b)])
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}
