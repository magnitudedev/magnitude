// `gated_attention_decode` and `gated_attention_prefill` against their portable
// body (the reference interpreter, small shapes) and against a host model of
// that body (Qwen3.5-4B geometry, long contexts), on the local GPU (Metal, or
// Vulkan elsewhere) and the CPU.

use magnitude_model_kernels::{gated_attention_decode, gated_attention_prefill};
use seismic::{BackendName, Device, DeviceCatalog, Element, NativeSpecialization, Tensor};

include!("attention_common/fixtures.rs");

/// Device tensors of one case.
struct Bound {
    query_gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    components: Tensor,
    frequencies: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    history_key: Tensor,
    history_value: Tensor,
}

impl Bound {
    fn new(device: &Device, case: &Case) -> Self {
        let Geometry { kv, g, p, .. } = case.geometry;
        let (m, w, t) = (case.rows, case.geometry.w(), case.history_rows);
        Self {
            query_gate: bf16_tensor(device, &[m, kv * g * 2 * w], &case.query_gate),
            key: bf16_tensor(device, &[m, kv * w], &case.key),
            value: bf16_tensor(device, &[m, kv * w], &case.value),
            query_norm: f32_tensor(device, &[w], &case.query_norm),
            key_norm: f32_tensor(device, &[w], &case.key_norm),
            components: i32_tensor(device, &[p], &case.components),
            frequencies: f32_tensor(device, &[p], &case.frequencies),
            coordinates: i32_tensor(device, &[m, 4], &case.coordinates),
            visible: i32_tensor(device, &[m, case.spans, 2], &case.visible),
            fresh: i32_tensor(device, &[m, 2], &case.fresh),
            destinations: i32_tensor(device, &[m], &case.destinations),
            history_key: bf16_tensor(device, &[t, kv, w], &case.history_key),
            history_value: bf16_tensor(device, &[t, kv, w], &case.history_value),
        }
    }
}

/// The CPU decode's partition counts.
const CPU_PARTS: [u64; 4] = [8, 4, 16, 1];

/// The decode specializations to run on `device`, labelled: on the GPU the
/// statics with each (SPAN, PARTS, SIMDS) of `configs`; on CPU (no statics)
/// each of its declared PARTS.
fn decode_specializations_on(
    device: &Device,
    geometry: Geometry,
    configs: &[(u64, u64, u64)],
) -> Vec<(String, NativeSpecialization)> {
    if is_cpu(device) {
        return CPU_PARTS
            .iter()
            .map(|parts| {
                (
                    format!("Cpu PARTS {parts}"),
                    NativeSpecialization::new().with_param("PARTS", *parts),
                )
            })
            .collect();
    }
    configs
        .iter()
        .map(|&(span, parts, simds)| {
            (
                format!(
                    "{:?} (SPAN, PARTS, SIMDS) {:?}",
                    device.backend(),
                    (span, parts, simds)
                ),
                statics(geometry)
                    .with_param("SPAN", span)
                    .with_param("PARTS", parts)
                    .with_param("SIMDS", simds),
            )
        })
        .collect()
}

/// The prefill specializations to run on `device`, labelled: on the GPU the
/// statics with each (QT, SPLIT_GROUPS) of `configs`; on CPU its one form
/// (no statics, no parameters).
fn prefill_specializations_on(
    device: &Device,
    geometry: Geometry,
    configs: &[(u64, u64)],
) -> Vec<(String, NativeSpecialization)> {
    if is_cpu(device) {
        return vec![("Cpu".to_owned(), NativeSpecialization::new())];
    }
    configs
        .iter()
        .map(|&(query_tile, split_groups)| {
            // Vulkan tiles ROWS = 64 matrix rows (query tile x query heads), the
            // one tile admissible at every tested geometry, whatever QT asks for.
            let tile = if device.backend() == BackendName::Vulkan {
                ("ROWS", 64)
            } else {
                ("QT", query_tile)
            };
            (
                format!(
                    "{:?} (QT, SPLIT_GROUPS) {:?}",
                    device.backend(),
                    (query_tile, split_groups)
                ),
                statics(geometry)
                    .with_param(tile.0, tile.1)
                    .with_param("SPLIT_GROUPS", split_groups),
            )
        })
        .collect()
}

fn decode_kernel(
    device: &Device,
    specialization: &NativeSpecialization,
) -> seismic::NativeKernel<gated_attention_decode::Entry> {
    gated_attention_decode::native_for_device_with(
        device,
        gated_attention_decode::Elements { A: Element::bf16() },
        specialization,
    )
    .unwrap()
}

fn prefill_kernel(
    device: &Device,
    specialization: &NativeSpecialization,
) -> seismic::NativeKernel<gated_attention_prefill::Entry> {
    gated_attention_prefill::native_for_device_with(
        device,
        gated_attention_prefill::Elements { A: Element::bf16() },
        specialization,
    )
    .unwrap()
}

fn run_decode(
    kernel: &seismic::NativeKernel<gated_attention_decode::Entry>,
    bound: &mut Bound,
    case: &Case,
) -> Tensor {
    kernel
        .call(gated_attention_decode::Args {
            query_gate: &bound.query_gate,
            key: &bound.key,
            value: &bound.value,
            query_norm: &bound.query_norm,
            key_norm: &bound.key_norm,
            rotary_components: &bound.components,
            rotary_frequencies: &bound.frequencies,
            coordinates: &bound.coordinates,
            visible: &bound.visible,
            fresh: &bound.fresh,
            destinations: &bound.destinations,
            history_key: &mut bound.history_key,
            history_value: &mut bound.history_value,
            epsilon: case.epsilon,
            scale: case.scale,
        })
        .unwrap()
        .value
}

fn run_prefill(
    kernel: &seismic::NativeKernel<gated_attention_prefill::Entry>,
    bound: &mut Bound,
    case: &Case,
) -> Tensor {
    kernel
        .call(gated_attention_prefill::Args {
            query_gate: &bound.query_gate,
            key: &bound.key,
            value: &bound.value,
            query_norm: &bound.query_norm,
            key_norm: &bound.key_norm,
            rotary_components: &bound.components,
            rotary_frequencies: &bound.frequencies,
            coordinates: &bound.coordinates,
            visible: &bound.visible,
            fresh: &bound.fresh,
            destinations: &bound.destinations,
            history_key: &mut bound.history_key,
            history_value: &mut bound.history_value,
            epsilon: case.epsilon,
            scale: case.scale,
        })
        .unwrap()
        .value
}

/// Gated outputs agree within bf16 publication plus reduced-precision query
/// staging; histories agree exactly except for keys, which may differ by one
/// bf16 step from transcendental rounding.
fn check(
    label: &str,
    case: &Case,
    gated: &Tensor,
    bound: &Bound,
    expected: &(Vec<f32>, Vec<f32>, Vec<f32>),
) {
    let actual = bf16_values(gated);
    assert_eq!(actual.len(), expected.0.len(), "{label}: gated shape");
    let mut worst = 0.0f32;
    for (index, (a, e)) in actual.iter().zip(&expected.0).enumerate() {
        let error = (a - e).abs();
        worst = worst.max(error);
        assert!(
            error <= 4.0e-3 + 1.6e-2 * e.abs(),
            "{label}: gated[{index}] (row {}, head {}, column {}) device {a} expected {e}",
            index / (case.geometry.kv * case.geometry.g * case.geometry.w()),
            index / case.geometry.w() % (case.geometry.kv * case.geometry.g),
            index % case.geometry.w()
        );
    }
    let key = bf16_values(&bound.history_key);
    for (index, (a, e)) in key.iter().zip(&expected.1).enumerate() {
        assert!(
            (a - e).abs() <= 8.0e-3 * e.abs().max(1.0),
            "{label}: history_key[{index}] device {a} expected {e}"
        );
    }
    assert_eq!(
        bf16_values(&bound.history_value),
        expected.2,
        "{label}: history_value"
    );
    eprintln!("{label}: max |gated error| {worst:.2e}");
}

/// The portable body (reference interpreter) agrees with the host model of
/// it, so the host model stands for the body at shapes the interpreter cannot
/// reach.
fn check_host_model_against_portable_body(entry: &str, case: &Case) {
    use seismic_lang::{
        checked::{check_source, SourceFile},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "attention.seismic".into(),
        text: include_str!("../kernels/attention.seismic").into(),
    });
    let module = check_source(sources).unwrap();
    let logical = module
        .entry(
            module.entry_named(entry).unwrap(),
            &ElementBindings::new().bind("A", registry::dense(DType::BF16)),
        )
        .unwrap();
    let Geometry { kv, g, p, .. } = case.geometry;
    let (m, w, t) = (case.rows, case.geometry.w(), case.history_rows);
    let mut interpreter = Interpreter::new(&logical);
    let mut tensor = |dtype, shape: Vec<usize>, values: Vec<f64>| {
        Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
    };
    let floats = |values: &[f32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let ints = |values: &[i32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let args = vec![
        tensor(
            DType::BF16,
            vec![m, kv * g * 2 * w],
            floats(&case.query_gate),
        ),
        tensor(DType::BF16, vec![m, kv * w], floats(&case.key)),
        tensor(DType::BF16, vec![m, kv * w], floats(&case.value)),
        tensor(DType::F32, vec![w], floats(&case.query_norm)),
        tensor(DType::F32, vec![w], floats(&case.key_norm)),
        tensor(DType::I32, vec![p], ints(&case.components)),
        tensor(DType::F32, vec![p], floats(&case.frequencies)),
        tensor(DType::I32, vec![m, 4], ints(&case.coordinates)),
        tensor(DType::I32, vec![m, case.spans, 2], ints(&case.visible)),
        tensor(DType::I32, vec![m, 2], ints(&case.fresh)),
        tensor(DType::I32, vec![m], ints(&case.destinations)),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_key)),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_value)),
        Arg::Scalar(ReferenceScalar::F32(case.epsilon.to_bits())),
        Arg::Scalar(ReferenceScalar::F32(case.scale.to_bits())),
    ];
    let outcome = interpreter.run(&args).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{entry} portable body failed: {failure}");
    }
    let expected = case.expected();
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(gated) = result.value() else {
        panic!("{entry} returns a tensor")
    };
    for (index, e) in expected.0.iter().enumerate() {
        let a = gated.read(index).unwrap() as f32;
        assert!(
            (a - e).abs() <= 1.0e-3 + 8.0e-3 * e.abs(),
            "{entry} portable gated[{index}] {a}, host model {e}"
        );
    }
    let inputs = outcome.inputs().collect::<Vec<_>>();
    for (ordinal, expected, name) in [
        (11, &expected.1, "history_key"),
        (12, &expected.2, "history_value"),
    ] {
        let input = inputs
            .iter()
            .find(|input| input.ordinal() == ordinal)
            .unwrap();
        let tensor = input.tensor();
        for (index, e) in expected.iter().enumerate() {
            let a = tensor.read(index).unwrap() as f32;
            assert!(
                (a - e).abs() <= 8.0e-3 * e.abs().max(1.0),
                "{entry} portable {name}[{index}] {a}, host model {e}"
            );
        }
    }
}

#[test]
fn decode_matches_portable_body() {
    let case = Case::new(SMALL, 64, 2, &decode_rows(5), 11);
    check_host_model_against_portable_body("gated_attention_decode", &case);
    for device in devices() {
        decode_matches_portable_body_on(&device, &case);
    }
}

fn decode_matches_portable_body_on(device: &Device, case: &Case) {
    let expected = case.expected();
    for (config, specialization) in
        decode_specializations_on(device, SMALL, &[(64, 32, 8), (32, 16, 4), (256, 8, 8)])
    {
        let kernel = decode_kernel(device, &specialization);
        let mut bound = Bound::new(device, case);
        let gated = run_decode(&kernel, &mut bound, case);
        check(
            &format!("small decode {config}"),
            case,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn prefill_matches_portable_body() {
    // 300 history rows are 10 key tiles: SPLIT_GROUPS 1 keeps one key
    // partition per query tile; 256 splits the first sequence's history
    // spans over two partitions and merges them, while its short second
    // sequence stays unsplit.
    let case = Case::new(SMALL, 400, 2, &prefill_rows(20, 300), 23);
    check_host_model_against_portable_body("gated_attention_prefill", &case);
    for device in devices() {
        prefill_matches_portable_body_on(&device, &case);
    }
}

fn prefill_matches_portable_body_on(device: &Device, case: &Case) {
    let expected = case.expected();
    for (config, specialization) in
        prefill_specializations_on(device, SMALL, &[(16, 1), (8, 1), (16, 256), (8, 256)])
    {
        let kernel = prefill_kernel(device, &specialization);
        let mut bound = Bound::new(device, case);
        let gated = run_prefill(&kernel, &mut bound, case);
        check(
            &format!("small prefill {config}"),
            case,
            &gated,
            &bound,
            &expected,
        );
    }
}

/// Device time of one decode layer's attention at Qwen3.5-4B geometry
/// (indicative only on a shared development GPU): `cargo test --release
/// --test attention -- --ignored --nocapture decode_timing`. Context 1 (the
/// row sees only itself) is the entry's launch floor.
#[test]
#[ignore]
fn decode_timing() {
    for device in devices() {
        decode_timing_on(&device);
    }
}

fn decode_timing_on(device: &Device) {
    let configs = [
        (32, 16, 4),
        (32, 8, 4),
        (64, 8, 4),
        (64, 16, 4),
        (128, 16, 4),
        (32, 16, 8),
        (64, 32, 4),
    ];
    for context in [1usize, 256, 4096, 16384] {
        let rows = [Row {
            spans: vec![(0, context as i32 - 1)],
            fresh: (0, 1),
            destination: context as i32 - 1,
            position: context as i32 - 1,
        }];
        let case = Case::new(QWEN, context + 64, 1, &rows, 3);
        for (config, specialization) in decode_specializations_on(device, QWEN, &configs) {
            let kernel = decode_kernel(device, &specialization);
            let mut bound = Bound::new(device, &case);
            let measurement = kernel
                .measure(
                    vec![gated_attention_decode::Args {
                        query_gate: &bound.query_gate,
                        key: &bound.key,
                        value: &bound.value,
                        query_norm: &bound.query_norm,
                        key_norm: &bound.key_norm,
                        rotary_components: &bound.components,
                        rotary_frequencies: &bound.frequencies,
                        coordinates: &bound.coordinates,
                        visible: &bound.visible,
                        fresh: &bound.fresh,
                        destinations: &bound.destinations,
                        history_key: &mut bound.history_key,
                        history_value: &mut bound.history_value,
                        epsilon: case.epsilon,
                        scale: case.scale,
                    }],
                    &TIMING,
                )
                .unwrap();
            let kv_bytes = (context * QWEN.kv * QWEN.w() * 2 * 2) as f64;
            eprintln!(
                "decode context {context} {config}: {:.1} us (KV read {:.0} GB/s)",
                measurement.median * 1e6,
                kv_bytes / measurement.median / 1e9
            );
        }
    }
}

/// Device time of one prefill layer's attention at Qwen3.5-4B geometry, rows
/// of one sequence after `history` rows (indicative only on a shared
/// development GPU): `cargo test --release --test attention -- --ignored
/// --nocapture prefill_timing`. FLOP counts both products of every visible
/// (row, key) pair.
#[test]
#[ignore]
fn prefill_timing() {
    for device in devices() {
        prefill_timing_on(&device);
    }
}

fn prefill_timing_on(device: &Device) {
    for (rows, history) in [(128usize, 0i32), (128, 4096), (512, 0), (512, 16384)] {
        let spans = if history > 0 {
            vec![(0, history)]
        } else {
            vec![]
        };
        let rows = (0..rows)
            .map(|row| Row {
                spans: spans.clone(),
                fresh: (0, row as i32 + 1),
                destination: history + row as i32,
                position: history + row as i32,
            })
            .collect::<Vec<_>>();
        let pairs = rows
            .iter()
            .map(|row| (history + row.fresh.1 - row.fresh.0) as f64)
            .sum::<f64>();
        let flop = pairs * (QWEN.kv * QWEN.g * QWEN.w() * 4) as f64;
        let case = Case::new(QWEN, history as usize + rows.len(), 1, &rows, 9);
        let configs = [(16, 1), (16, 128), (16, 256), (16, 512), (8, 1), (8, 256)];
        for (config, specialization) in prefill_specializations_on(device, QWEN, &configs) {
            let kernel = prefill_kernel(device, &specialization);
            let mut bound = Bound::new(device, &case);
            let measurement = kernel
                .measure(
                    vec![gated_attention_prefill::Args {
                        query_gate: &bound.query_gate,
                        key: &bound.key,
                        value: &bound.value,
                        query_norm: &bound.query_norm,
                        key_norm: &bound.key_norm,
                        rotary_components: &bound.components,
                        rotary_frequencies: &bound.frequencies,
                        coordinates: &bound.coordinates,
                        visible: &bound.visible,
                        fresh: &bound.fresh,
                        destinations: &bound.destinations,
                        history_key: &mut bound.history_key,
                        history_value: &mut bound.history_value,
                        epsilon: case.epsilon,
                        scale: case.scale,
                    }],
                    &TIMING,
                )
                .unwrap();
            eprintln!(
                "prefill {} rows after {history} history {config}: {:.1} us ({:.2} TFLOP/s)",
                rows.len(),
                measurement.median * 1e6,
                flop / measurement.median / 1e12
            );
        }
    }
}

#[test]
fn qwen_geometry_speculative_decode_matches_host_model() {
    for device in devices() {
        qwen_geometry_speculative_decode_matches_host_model_on(&device);
    }
}

fn qwen_geometry_speculative_decode_matches_host_model_on(device: &Device) {
    for context in [256, 4096, 16384] {
        for rows in [1, 8] {
            let case = Case::new(
                QWEN,
                context as usize + rows,
                1,
                &speculative_rows(rows, context),
                13,
            );
            let expected = case.expected();
            let configs = [
                (32, 16, 4),
                (32, 16, 8),
                (64, 32, 8),
                (128, 8, 8),
                (256, 32, 4),
            ];
            for (config, specialization) in decode_specializations_on(device, QWEN, &configs) {
                let kernel = decode_kernel(device, &specialization);
                let mut bound = Bound::new(device, &case);
                let gated = run_decode(&kernel, &mut bound, &case);
                check(
                    &format!("speculative decode {rows} rows context {context} {config}"),
                    &case,
                    &gated,
                    &bound,
                    &expected,
                );
            }
        }
    }
}

#[test]
fn qwen_geometry_decode_and_prefill_match_host_model() {
    for device in devices() {
        qwen_geometry_decode_and_prefill_match_host_model_on(&device);
    }
}

fn qwen_geometry_decode_and_prefill_match_host_model_on(device: &Device) {
    for (context, configs) in [
        (256, vec![(64, 32, 8), (32, 16, 4)]),
        (4096, vec![(64, 32, 8), (256, 16, 4)]),
        (16384, vec![(64, 32, 8), (32, 8, 4), (256, 16, 8)]),
    ] {
        let case = Case::new(QWEN, context + 128, 2, &decode_rows(context as i32 - 40), 5);
        let expected = case.expected();
        for (config, specialization) in decode_specializations_on(device, QWEN, &configs) {
            let kernel = decode_kernel(device, &specialization);
            let mut bound = Bound::new(device, &case);
            let gated = run_decode(&kernel, &mut bound, &case);
            check(
                &format!("decode context {context} {config}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
    for (rows, history, configs) in [
        (40, 300, vec![(16, 1), (8, 256)]),
        (128, 1000, vec![(16, 256), (8, 128)]),
        (64, 4096, vec![(16, 256), (8, 512)]),
        (48, 16384, vec![(16, 256), (8, 128)]),
    ] {
        let case = Case::new(
            QWEN,
            history as usize + 256,
            2,
            &prefill_rows(rows, history),
            7,
        );
        let expected = case.expected();
        for (config, specialization) in prefill_specializations_on(device, QWEN, &configs) {
            let kernel = prefill_kernel(device, &specialization);
            let mut bound = Bound::new(device, &case);
            let gated = run_prefill(&kernel, &mut bound, &case);
            check(
                &format!("prefill {rows} rows after {history} {config}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}
