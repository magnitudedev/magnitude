mod common;
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
    for device in seismic_native_tests::devices(&[seismic::BackendName::Metal, seismic::BackendName::Cuda, seismic::BackendName::Vulkan, seismic::BackendName::Cpu]) {
        packed_stages_on(&device);
    }
}

fn packed_stages_on(device: &seismic::Device) {
    let charges = seismic::Charges::tally(device);
    let q8 = q8_element(device);
    let bf16 = seismic::Element::bf16();
    let resident = |shape: &[u64]| q8_weight(device, shape, 1.0, |_| 0);
    let epsilon = 1e-5;
    let (hidden_width, width) = (HIDDEN as usize, WIDTH as usize);
    let (keys, values) = (KEYS as usize, VALUES as usize);
    let channels = (2 * keys + values) * width;
    let hidden_values = (0..hidden_width)
        .map(|index| index as f32 / hidden_width as f32 + 0.25)
        .collect::<Vec<_>>();
    let hidden = tensor(&device, &[1, HIDDEN], &hidden_values);
    let input_norm = bf16_tensor(device, &[HIDDEN], &vec![1.0; hidden_width]);
    let recurrent_norm = bf16_tensor(device, &[WIDTH], &vec![1.0; width]);

    // A two-tap convolution over one slot that reads bank 0 and publishes
    // bank 1 (`RecurrentState`).
    let segments = indices(&device, &[2, 2], &[0, 1, 1, 1]);
    let stop = indices(&device, &[1], &[1]);
    let previous_bank = indices(&device, &[1], &[0]);
    let previous_tape = indices(&device, &[1], &[0]);
    let following_bank = indices(&device, &[1], &[1]);
    let mut state = RecurrentState::new(device);
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
        &[],
    )
    .call(&charges, gated_delta_project::Args {
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
        &[],
    )
    .call(&charges, gated_delta_step::Args {
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
        window: &mut state.window,
        delta: &mut state.delta,
        tape: &mut state.tape,
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
        &[],
    )
    .call(&charges, attention_output::Args {
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

/// The recurrent decode step and the output projection as one CPU graph,
/// under the INT8 form at one row: the step emits the gated row's operand
/// (its staged rows and their Q8 blocks) in its epilogue, and the output projection reads it in
/// place of forming its own. The gated row and the output have the bits of
/// the two entries called alone, the output forming its own operand.
#[test]
fn cpu_output_reads_the_operand_the_recurrent_step_emits() {
    let Some(device) = seismic_native_tests::fresh_device(seismic::BackendName::Cpu) else { return };
    let device = &device;
    let charges = seismic::Charges::tally(device);
    let (q8, bf16) = (q8_element(device), seismic::Element::bf16());
    let epsilon = 1e-5;
    let (hidden_width, width, keys, values) = (HIDDEN as usize, WIDTH as usize, KEYS as usize, VALUES as usize);
    let channels = (2 * keys + values) * width;
    let pattern = |count: usize, seed: usize| {
        (0..count).map(|index| ((index * 37 + seed * 11) % 64) as f32 / 32.0 - 1.0).collect::<Vec<_>>()
    };
    let weight = |shape: &[u64], seed: usize| {
        q8_weight(device, shape, 1.0 / 64.0, |index| (((index + seed) * 29 + 7) % 61) as i8 - 30)
    };
    let hidden = tensor(device, &[1, HIDDEN], &pattern(hidden_width, 1));
    let input_norm = bf16_tensor(device, &[HIDDEN], &vec![1.0; hidden_width]);
    let recurrent_norm =
        bf16_tensor(device, &[WIDTH], &(0..width).map(|index| 1.0 + index as f32 / 64.0).collect::<Vec<_>>());
    let convolution = tensor(device, &[channels as u64, 2], &pattern(channels * 2, 2));
    let rate = tensor(device, &[VALUES], &pattern(values, 3));
    let time_bias = tensor(device, &[VALUES], &pattern(values, 4));
    let segments = indices(device, &[2, 2], &[0, 1, 1, 1]);
    let stop = indices(device, &[1], &[1]);
    let previous_bank = indices(device, &[1], &[0]);
    let previous_tape = indices(device, &[1], &[0]);
    let following_bank = indices(device, &[1], &[1]);
    let output_weight = weight(&[HIDDEN, VALUES * WIDTH], 5);
    let (absent, inject) = (tensor(device, &[0], &[]), no_inject(device, 1));
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
        &[],
    )
    .call(&charges, gated_delta_project::Args {
        hidden: &hidden,
        input_norm: &input_norm,
        qkv_weight: &weight(&[channels as u64, HIDDEN], 1),
        gate_weight: &weight(&[VALUES * WIDTH, HIDDEN], 2),
        alpha_weight: &weight(&[VALUES, HIDDEN], 3),
        beta_weight: &weight(&[VALUES, HIDDEN], 4),
        epsilon,
        qkv_bound: &absent,
        gate_bound: &absent,
    })
    .unwrap()
    .value;
    let step = at_defaults(
        device,
        gated_delta_step::native_entry_with(gated_delta_step::Elements { RN: bf16, A: bf16 }),
        &[],
    );
    let output = at_defaults(
        device,
        attention_output::native_entry_with(attention_output::Elements { OW: q8, A: bf16 }),
        &[("INT8", 1)],
    );

    // Alone, the output forms its own operand.
    let mut state = RecurrentState::new(device);
    let gated = step
        .call(&charges, gated_delta_step::Args {
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
            window: &mut state.window,
            delta: &mut state.delta,
            tape: &mut state.tape,
            norm_epsilon: epsilon,
            epsilon,
            grouped: true,
            gate_function: 0,
            slab_banks: 2,
        })
        .unwrap()
        .value;
    let alone = output
        .call(&charges, attention_output::Args {
            hidden: &hidden,
            inject: &inject,
            gated: &gated,
            output_weight: &output_weight,
            output_scale: &absent,
            output_bound: &absent,
        })
        .unwrap()
        .value;

    // In one graph, the output reads the operand the step emits.
    let state = RecurrentState::new(device);
    let mut graph = device.native_graph();
    let mut port = |tensor: &seismic::Tensor| graph.port(tensor.element(), &tensor.extents()).unwrap();
    let inputs = [
        &projection,
        &convolution,
        &rate,
        &time_bias,
        &recurrent_norm,
        &segments,
        &stop,
        &previous_bank,
        &previous_tape,
        &following_bank,
        &hidden,
        &inject,
        &output_weight,
        &absent,
    ];
    let [
        projection_port,
        convolution_port,
        rate_port,
        time_bias_port,
        recurrent_norm_port,
        segments_port,
        stop_port,
        previous_bank_port,
        previous_tape_port,
        following_bank_port,
        hidden_port,
        inject_port,
        output_weight_port,
        absent_port,
    ] = inputs.map(&mut port);
    let mut states = [&state.window, &state.delta, &state.tape].map(&mut port);
    let stored = [
        seismic::Tensor::zeros(&charges, gated.element(), &gated.extents()).unwrap(),
        seismic::Tensor::zeros(&charges, alone.element(), &alone.extents()).unwrap(),
    ];
    let stored_ports = [&stored[0], &stored[1]].map(&mut port);
    let [window_port, delta_port, tape_port] = &mut states;
    let gated_in_graph = graph
        .enqueue(
            &step,
            gated_delta_step::WorkflowArgs {
                projection: projection_port.tensor().into(),
                convolution: convolution_port.tensor().into(),
                rate: rate_port.tensor().into(),
                time_bias: time_bias_port.tensor().into(),
                recurrent_norm: recurrent_norm_port.tensor().into(),
                segments: segments_port.tensor().into(),
                stop: stop_port.tensor().into(),
                previous_bank: previous_bank_port.tensor().into(),
                previous_tape: previous_tape_port.tensor().into(),
                following_bank: following_bank_port.tensor().into(),
                window: window_port.tensor_mut().into(),
                delta: delta_port.tensor_mut().into(),
                tape: tape_port.tensor_mut().into(),
                norm_epsilon: seismic::Argument::from(epsilon),
                epsilon: seismic::Argument::from(epsilon),
                grouped: true.into(),
                gate_function: seismic::Argument::from(0i32),
                slab_banks: seismic::Argument::from(2u32),
            },
        )
        .unwrap()
        .value;
    let value = graph
        .enqueue(
            &output,
            attention_output::WorkflowArgs {
                hidden: hidden_port.tensor().into(),
                inject: inject_port.tensor().into(),
                gated: (&gated_in_graph).into(),
                output_weight: output_weight_port.tensor().into(),
                output_scale: absent_port.tensor().into(),
                output_bound: absent_port.tensor().into(),
            },
        )
        .unwrap()
        .value;
    for (result, port) in [&gated_in_graph, &value].into_iter().zip(&stored_ports) {
        graph.store(result, port).unwrap();
    }
    let plan = graph.seal().unwrap();
    let mut bindings = plan.bindings();
    let bound = [
        (&projection_port, &projection),
        (&convolution_port, &convolution),
        (&rate_port, &rate),
        (&time_bias_port, &time_bias),
        (&recurrent_norm_port, &recurrent_norm),
        (&segments_port, &segments),
        (&stop_port, &stop),
        (&previous_bank_port, &previous_bank),
        (&previous_tape_port, &previous_tape),
        (&following_bank_port, &following_bank),
        (&hidden_port, &hidden),
        (&inject_port, &inject),
        (&output_weight_port, &output_weight),
        (&absent_port, &absent),
        (&states[0], &state.window),
        (&states[1], &state.delta),
        (&states[2], &state.tape),
    ];
    for (port, tensor) in bound.into_iter().chain(stored_ports.iter().zip(&stored)) {
        bindings.set(&charges, port, tensor).unwrap();
    }
    let (_, completion) = plan.new_slot(&charges).unwrap().attach(bindings, plan.new_outputs(&charges).unwrap()).unwrap().submit().unwrap();
    completion.wait().unwrap();
    let bytes = |tensor: &seismic::Tensor| tensor.read_to_host().unwrap();
    assert_eq!(bytes(&stored[0]), bytes(&gated), "gated row");
    assert_eq!(bytes(&stored[1]), bytes(&alone), "output");
}

/// A two-bank recurrent state arena (bank 0 read, bank 1 published) with one
/// tape row (u [NV, W] | k [NK, W] | d [NV]), for a two-tap convolution,
/// zero throughout.
struct RecurrentState {
    _slabs: seismic::SlabTensor,
    window: seismic::Tensor,
    delta: seismic::Tensor,
    tape: seismic::Tensor,
}

impl RecurrentState {
    fn new(device: &seismic::Device) -> Self {
        let (width, keys, values) = (WIDTH as usize, KEYS as usize, VALUES as usize);
        let channels = (2 * keys + values) * width;
        let regions = vec![
            seismic::SlabRegion {
                element: seismic::Element::bf16(),
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
        ];
        let slabs = crate::common::slab_tensor(device, 2, 2, &regions);
        slabs.add_slab(seismic::SlabFill::Zeroed).unwrap();
        Self {
            window: slabs.logical_region(0).unwrap(),
            delta: slabs.logical_region(1).unwrap(),
            tape: slabs.logical_region(2).unwrap(),
            _slabs: slabs,
        }
    }
}

/// The element `device`'s kernels read Q8 weights as (`q8g32s` in its
/// resident layout).
fn q8_element(device: &seismic::Device) -> seismic::Element {
    seismic::Element::stored("q8g32s", device.backend().resident_layout()).unwrap()
}

/// A `shape` Q8 weight in `device`'s resident form, repacked from GGUF
/// Q8_0 blocks of scale `scale` whose value `index` is `code(index)`.
fn q8_weight(device: &seismic::Device, shape: &[u64], scale: f32, code: impl Fn(usize) -> i8) -> seismic::Tensor {
    let charges = seismic::Charges::tally(device);
    let q8_external = seismic::Element::named("gguf_q8_0").unwrap();
    let values = shape.iter().product::<u64>() as usize;
    assert_eq!(values % 32, 0);
    let mut bytes = Vec::with_capacity(values / 32 * 34);
    for block in 0..values / 32 {
        bytes.extend_from_slice(&seismic_lang::registry::f16_bits(scale).to_le_bytes());
        bytes.extend((0..32).map(|value| code(block * 32 + value) as u8));
    }
    let external =
        seismic::Tensor::from_host(&charges, q8_external, &[1, shape[0], shape[1]], &bytes).unwrap();
    at_defaults(
        device,
        repack_weight::native_entry_with(repack_weight::Elements {
            E: q8_external,
            U: q8_element(device),
        }),
        &[],
    )
    .call(&charges, repack_weight::Args { source: &external })
    .unwrap()
    .value
    .reshape(shape)
    .unwrap()
}

/// `entry` prepared at its declared defaults over the test's geometry (one
/// key and four value heads of width 32 over a 64-wide hidden row, a
/// geometry every backend's declarations admit: CUDA's projection takes H
/// in multiples of 64, its step NV in multiples of 4), each parameter
/// `pinned` names held at its value.
fn at_defaults<E: seismic::Entry>(
    device: &seismic::Device,
    entry: seismic::BoundEntry<E>,
    pinned: &[(&str, u64)],
) -> seismic::NativeKernel<E> {
    let specialization = declared::declared(device, &entry, |name| match name {
        "H" | "D" | "NM" => HIDDEN,
        "W" => WIDTH,
        "NK" => KEYS,
        "NV" | "Q" => VALUES,
        "PT" | "SN" => 1,
        "C" => 2,
        "IN" | "QB" | "ZB" | "OB" | "OS" => 0,
        _ => panic!("unexpected static {name}"),
    }, pinned).unwrap();
    entry.prepare(device, &specialization).unwrap()
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
    let charges = seismic::Charges::tally(device);
    seismic::Tensor::from_host(&charges, seismic::Element::f32(), shape, &f32_bytes(values)).unwrap()
}

/// `values` rounded to BF16 by truncation.
fn bf16_tensor(device: &seismic::Device, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    let charges = seismic::Charges::tally(device);
    let bytes = values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(&charges, seismic::Element::bf16(), shape, &bytes).unwrap()
}

fn indices(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    let charges = seismic::Charges::tally(device);
    seismic::Tensor::from_host(&charges, seismic::Element::i32(), shape, &i32_bytes(values)).unwrap()
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
        include_str!("../kernels/feed_forward.seismic"),
        include_str!("../kernels/dense.seismic"),
        include_str!("../kernels/short_conv.seismic"),
        include_str!("../kernels/per_layer_gate.seismic"),
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
    let charges = seismic::Charges::tally(device);
    seismic::Tensor::zeros(&charges, seismic::Element::f32(), &[rows, 0]).unwrap()
}
