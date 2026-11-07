mod declared;

use magnitude_kernels::{
    attention_output, gated_delta_chunk, gated_delta_project, gated_delta_step, repack_weight,
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn packed_stages_accept_repacked_q8_weights_and_bf16_activations() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    for backend in [
        seismic::BackendName::Metal,
        seismic::BackendName::Cuda,
        seismic::BackendName::Vulkan,
        seismic::BackendName::Cpu,
    ] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        packed_stages_on(&device);
    }
}

fn packed_stages_on(device: &seismic::Device) {
    let q8_external = seismic::Element::named("gguf_q8_0").unwrap();
    let layout = device.backend().resident_layout();
    let q8 = seismic::Element::stored("q8g32s", layout).unwrap();
    let bf16 = seismic::Element::bf16();
    // One key and four value heads of width 32 over a 64-wide hidden row, a
    // geometry every backend's declarations admit (CUDA's projection takes H
    // in multiples of 64, its step NV in multiples of 4), each entry prepared
    // at its declared default configuration.
    fn at_defaults<E: seismic::Entry>(
        device: &seismic::Device,
        entry: seismic::BoundEntry<E>,
    ) -> seismic::NativeKernel<E> {
        let specialization = declared::declared_default(device, &entry, |name| match name {
            "H" | "D" | "NM" => HIDDEN,
            "W" => WIDTH,
            "NK" => KEYS,
            "NV" | "Q" => VALUES,
            "PT" | "SN" => 1,
            "C" => 2,
            "IN" | "QB" | "ZB" | "OB" | "OS" => 0,
            _ => panic!("unexpected static {name}"),
        });
        entry.prepare(device, &specialization).unwrap()
    }
    let resident = |shape: &[u64]| {
        let values = shape.iter().product::<u64>() as usize;
        assert_eq!(values % 32, 0);
        let mut bytes = Vec::with_capacity(values / 32 * 34);
        for _ in 0..values / 32 {
            bytes.extend_from_slice(&seismic_lang::registry::f16_bits(1.0).to_le_bytes());
            bytes.extend_from_slice(&[0; 32]);
        }
        let external =
            seismic::Tensor::from_host(&device, q8_external, &[1, shape[0], shape[1]], &bytes)
                .unwrap();
        at_defaults(
            device,
            repack_weight::native_entry_with(repack_weight::Elements {
                E: q8_external,
                U: q8,
            }),
        )
        .call(repack_weight::Args { source: &external })
        .unwrap()
        .value
        .reshape(shape)
        .unwrap()
    };
    let bf16_tensor = |shape: &[u64], values: &[f32]| {
        let bytes = values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        seismic::Tensor::from_host(&device, bf16, shape, &bytes).unwrap()
    };
    let epsilon = 1e-5;
    let (hidden_width, width) = (HIDDEN as usize, WIDTH as usize);
    let (keys, values) = (KEYS as usize, VALUES as usize);
    let channels = (2 * keys + values) * width;
    let hidden_values = (0..hidden_width)
        .map(|index| index as f32 / hidden_width as f32 + 0.25)
        .collect::<Vec<_>>();
    let hidden = tensor(&device, &[1, HIDDEN], &hidden_values);
    let input_norm = bf16_tensor(&[HIDDEN], &vec![1.0; hidden_width]);
    let recurrent_norm = bf16_tensor(&[WIDTH], &vec![1.0; width]);

    // A two-tap convolution and a two-bank state arena (bank 0 read, bank 1
    // published) with one tape row (u [NV, W] | k [NK, W] | d [NV]).
    let segments = indices(&device, &[2, 2], &[0, 1, 1, 1]);
    let stop = indices(&device, &[1], &[1]);
    let previous_bank = indices(&device, &[1], &[0]);
    let previous_tape = indices(&device, &[1], &[0]);
    let following_bank = indices(&device, &[1], &[1]);
    let mut slabs = seismic::SlabTensor::new(
        &device,
        2,
        2,
        vec![
            seismic::SlabRegion {
                element: bf16,
                row_shape: vec![2, channels as u64],
            },
            seismic::SlabRegion {
                element: seismic::Element::f32(),
                row_shape: vec![VALUES, WIDTH, WIDTH],
            },
            seismic::SlabRegion {
                element: seismic::Element::f32(),
                row_shape: vec![1, ((values + keys) * width + values) as u64],
            },
        ],
    )
    .unwrap();
    slabs.add_slab().unwrap();
    let mut window = slabs.logical_region(0).unwrap();
    let mut delta = slabs.logical_region(1).unwrap();
    let mut tape = slabs.logical_region(2).unwrap();
    let qkv_weight = resident(&[channels as u64, HIDDEN]);
    let gate_weight = resident(&[VALUES * WIDTH, HIDDEN]);
    let alpha_weight = resident(&[VALUES, HIDDEN]);
    let beta_weight = resident(&[VALUES, HIDDEN]);
    let output_weight = resident(&[HIDDEN, VALUES * WIDTH]);
    let convolution = tensor(&device, &[channels as u64, 2], &vec![0.0; channels * 2]);
    let rate = tensor(&device, &[VALUES], &vec![0.0; values]);
    let time_bias = tensor(&device, &[VALUES], &vec![0.0; values]);
    let projection = at_defaults(
        device,
        gated_delta_project::native_entry_with(gated_delta_project::Elements {
            NW: bf16,
            QW: q8,
            GW: q8,
            AW: q8,
            BW: q8,
            A: bf16,
        }),
    )
    .call(gated_delta_project::Args {
        hidden: &hidden,
        input_norm: &input_norm,
        qkv_weight: &qkv_weight,
        gate_weight: &gate_weight,
        alpha_weight: &alpha_weight,
        beta_weight: &beta_weight,
        epsilon,
        qkv_bound: &tensor(&device, &[0], &[]),
        gate_bound: &tensor(&device, &[0], &[]),
    })
    .unwrap()
    .value;
    let gated = at_defaults(
        device,
        gated_delta_step::native_entry_with(gated_delta_step::Elements { RN: bf16, A: bf16 }),
    )
    .call(gated_delta_step::Args {
        projection: &projection,
        convolution: &convolution,
        rate: &rate,
        time_bias: &time_bias,
        recurrent_norm: &recurrent_norm,
        segments: &segments,
        stop: &stop,
        previous_bank: &previous_bank,
        previous_tape: &previous_tape,
        following_bank: &following_bank,
        window: &mut window,
        delta: &mut delta,
        tape: &mut tape,
        norm_epsilon: epsilon,
        epsilon,
        grouped: true,
        gate_function: 0,
        slab_banks: 2,
    })
    .unwrap()
    .value;
    let recurrent = at_defaults(
        device,
        attention_output::native_entry_with(attention_output::Elements { OW: q8, A: bf16 }),
    )
    .call(attention_output::Args {
        hidden: &hidden,
        inject: &no_inject(&device, hidden.extents()[0]),
        gated: &gated,
        output_weight: &output_weight,
        output_scale: &tensor(&device, &[0], &[]),
        output_bound: &tensor(&device, &[0], &[]),
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&recurrent), &hidden_values);
}

/// The test's geometry: `H`, `W`, `NK` and `NV`.
const HIDDEN: u64 = 64;
const WIDTH: u64 = 32;
const KEYS: u64 = 1;
const VALUES: u64 = 4;

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn tensor(device: &seismic::Device, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    seismic::Tensor::from_host(device, seismic::Element::f32(), shape, &f32_bytes(values)).unwrap()
}

fn indices(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    seismic::Tensor::from_host(device, seismic::Element::i32(), shape, &i32_bytes(values)).unwrap()
}

fn read_f32(tensor: &seismic::Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 3e-5,
            "{index}: {actual} != {expected}"
        );
    }
}

#[test]
fn packed_entries_are_the_only_generated_target_surface() {
    let _ = gated_delta_project::for_device_with;
    let _ = gated_delta_project::native_for_device_with;
    let _ = gated_delta_step::for_device_with;
    let _ = gated_delta_step::native_for_device_with;
    let _ = gated_delta_chunk::for_device_with;
    let _ = gated_delta_chunk::native_for_device_with;
    let sources = [
        include_str!("../kernels/target.seismic"),
        include_str!("../kernels/dense_rows.seismic"),
        include_str!("../kernels/routed.seismic"),
    ]
    .concat();
    for obsolete in [
        "qwen_append_rows",
        "gated_attention_sequence",
        "gated_delta_sequence",
        "dense_suffix",
        "routed_suffix",
        "fn route_topk",
        "fn routed_input",
    ] {
        assert!(
            !sources.contains(obsolete),
            "obsolete generated entry {obsolete}"
        );
    }
}

/// The inject port of a single residual stream: no columns.
fn no_inject(device: &seismic::Device, rows: u64) -> seismic::Tensor {
    seismic::Tensor::zeros(device, seismic::Element::f32(), &[rows, 0]).unwrap()
}
