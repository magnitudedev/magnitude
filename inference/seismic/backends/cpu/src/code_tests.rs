//! JIT code freed with its last kernel is reused for the next kernel
//! compiled; every participant must run the code written there last.
use crate::workers::{LaunchFrame, Workers};
use crate::{compile, open::open_quota, Cpu};
use seismic_ir::construction::Construction;
use seismic_ir::kernel::ops::{ConstantValue, ValueType};
use seismic_ir::storage::GlobalBufferKind;
use seismic_lang::expr::ExprArena;
use seismic_lang::registry;
use seismic_lang::types::DType;

/// Compiles a kernel whose workgroup `w` writes `value` to `output[w]`.
fn constant_writer(
    target: &seismic_native_target::DeviceDescription<Cpu>,
    workgroups: u64,
    value: f32,
) -> (
    compile::NativeCandidate,
    seismic_ir::physical_target::KernelEmissionLayout,
) {
    let mut arena = ExprArena::default();
    let mut construction = Construction::<Cpu>::new(
        &mut arena,
        vec![],
        false,
        target.addressable_resources().len(),
    );
    let extent = arena.nat(workgroups);
    let representation = registry::dense(DType::F32);
    let (_, output) = construction.storage_mut().tensor(
        &mut arena,
        GlobalBufferKind::Arena,
        representation,
        vec![extent],
    );
    let output = construction.view(output, representation);
    let mut builder = construction.portable_kernel(
        &mut arena,
        target.facts(),
        target.addressable_resources(),
        target.vectors(),
    );
    let output = builder.arg_view(output, true);
    let workgroup = builder.workgroup_id(0);
    let value = builder.constant(ConstantValue::F32(value), ValueType::Scalar(DType::F32));
    builder.write(output, &[workgroup], value);
    builder.close();
    let kernel = &construction.kernels()[0];
    let emission = target.kernel_emission_layout(kernel);
    (
        compile::compile_kernel(target, kernel, &emission).unwrap(),
        emission,
    )
}

#[test]
fn every_participant_runs_the_kernel_last_compiled_at_a_reused_code_address() {
    let opened = open_quota().unwrap();
    let target = &opened.device;
    let mut workers = Workers::host(std::thread::available_parallelism().unwrap()).unwrap();
    let workgroups = workers.count() as u64 * 64;
    for round in 0..256u32 {
        // Each round's kernel differs from the last only in its constant,
        // so it is the same size and is usually placed where the last one,
        // freed at the end of its round, was run by every participant.
        let value = round as f32;
        let (native, emission) = constant_writer(target, workgroups, value);
        let words_layout = &emission.words;
        let mut words = vec![0u64; words_layout.total as usize];
        let binding = words_layout.bindings[0];
        words[binding.first as usize] = workgroups;
        words[binding.first as usize + binding.rank as usize] = 1;
        let grid = words_layout.grid_first as usize;
        words[grid..grid + 3].copy_from_slice(&[workgroups, 1, 1]);
        let workgroup = words_layout.workgroup_first as usize;
        words[workgroup..workgroup + 3].copy_from_slice(&[1, 1, 1]);
        let mut output = vec![f32::NAN; workgroups as usize];
        let buffers = [output.as_mut_ptr().cast::<u8>()];
        let frame = LaunchFrame {
            buffers: buffers.as_ptr(),
            words: words.as_ptr(),
            results: std::ptr::null_mut(),
        };
        workers
            .run(native.kernel.entry, &frame, workgroups, 1, 0, 0, 0)
            .unwrap();
        let stale = output.iter().filter(|written| **written != value).count();
        assert_eq!(
            stale, 0,
            "round {round}: {stale} of {workgroups} workgroups ran other code than the kernel compiled"
        );
    }
}
