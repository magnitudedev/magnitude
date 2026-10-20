// The WorkList form of `attention_prefill` (`B > 0`): every row reads only
// the keys whose block (`block_rows` keys, in the row's key order) its bitmap
// sets. The portable body against the host model, and every backend this
// host opens against both, over rows of two sequences sharing query tiles,
// with split and unsplit key partitions.

use magnitude_kernels::attention_prefill;
use seismic::{BackendName, Device, Element, NativeSpecialization, SlabRegion, SlabTensor, Tensor};

include!("attention_common/fixtures.rs");

const BLOCK_ROWS: usize = 4;

/// A pseudo-random per-row block bitmap of `words` words reading about
/// three blocks in five.
fn bitmap(rows: usize, words: usize, seed: u64) -> Vec<u32> {
    let mut noise = Noise(seed);
    (0..rows * words)
        .map(|_| (0..32).filter(|_| noise.next() < 0.2).fold(0u32, |word, bit| word | 1 << bit))
        .collect()
}

fn reads(bits: &[u32], words: usize) -> impl Fn(usize, usize) -> bool + '_ {
    move |row, ordinal| {
        let block = ordinal / BLOCK_ROWS;
        block / 32 < words && bits[row * words + block / 32] >> (block % 32) & 1 != 0
    }
}

fn portable(case: &Case, bits: &[u32], words: usize) -> Vec<f32> {
    use seismic_lang::{
        checked::{SourceFile, check_source},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let mut sources = seismic_std::sources();
    for (path, text) in [
        ("attention.seismic", include_str!("../kernels/attention.seismic")),
        ("functions.seismic", include_str!("../kernels/functions.seismic")),
    ] {
        sources.push(SourceFile { path: path.into(), text: text.into() });
    }
    let module = check_source(sources).unwrap();
    let logical = module
        .entry(
            module.entry_named("attention_prefill").unwrap(),
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
        tensor(DType::BF16, vec![m, kv * g, 2 * w], floats(&case.query_gate)),
        tensor(DType::BF16, vec![m, kv * g, 0], Vec::new()),
        tensor(DType::BF16, vec![1, m, kv * w], floats(&case.key)),
        tensor(DType::BF16, vec![1, m, kv * w], floats(&case.value)),
        tensor(DType::F32, vec![1, w], floats(&case.query_norm)),
        tensor(DType::F32, vec![1, w], floats(&case.key_norm)),
        tensor(DType::F32, vec![0, w], Vec::new()),
        tensor(DType::I32, vec![p], ints(&case.components)),
        tensor(DType::F32, vec![p], floats(&case.frequencies)),
        tensor(DType::F32, vec![p], vec![1.0; p]),
        tensor(DType::I32, vec![m, 4], ints(&case.coordinates)),
        tensor(DType::I32, vec![m, case.spans, 2], ints(&case.visible)),
        tensor(DType::I32, vec![m, 2], ints(&case.fresh)),
        tensor(DType::I32, vec![m], ints(&case.destinations)),
        tensor(DType::U32, vec![m, words], bits.iter().map(|word| f64::from(*word)).collect()),
        tensor(DType::I32, vec![0, BLOCK_ROWS], Vec::new()),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_key)),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_value)),
        Arg::Scalar(ReferenceScalar::F32(case.epsilon.to_bits())),
        Arg::Scalar(ReferenceScalar::F32(case.scale.to_bits())),
        Arg::Scalar(ReferenceScalar::I32(0)),
        Arg::Scalar(ReferenceScalar::U32(t as u32)),
    ];
    let outcome = interpreter.run(&args).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable WorkList prefill failed: {failure}");
    }
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(gated) = result.value() else { panic!("a tensor result") };
    (0..gated.element_count()).map(|index| gated.read(index).unwrap() as f32).collect()
}

fn native(device: &Device, specialization: &NativeSpecialization, case: &Case, bits: &[u32], words: usize) -> Vec<f32> {
    let Geometry { kv, g, p, .. } = case.geometry;
    let (m, w, t) = (case.rows, case.geometry.w(), case.history_rows);
    let kernel = attention_prefill::native_for_device_with(
        device,
        attention_prefill::Elements { A: Element::bf16() },
        specialization,
    )
    .unwrap();
    let plane = |values: &[f32]| {
        let mut slabs = SlabTensor::new(
            device,
            t as u64,
            t as u64,
            vec![SlabRegion { element: Element::bf16(), row_shape: vec![kv as u64, w as u64] }],
        )
        .unwrap();
        slabs.add_slab().unwrap();
        slabs
            .region_rows(0, 0, t as u64)
            .unwrap()
            .write_from_host(&values.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        let logical = slabs.logical_region(0).unwrap();
        (slabs, logical)
    };
    let (_key_slabs, mut history_key) = plane(&case.history_key);
    let (_value_slabs, mut history_value) = plane(&case.history_value);
    let selection = Tensor::from_host(
        device,
        Element::u32(),
        &[m as u64, words as u64],
        &bits.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
    let result = kernel
        .call(attention_prefill::Args {
            query: &bf16_tensor(device, &[m, kv * g, 2 * w], &case.query_gate),
            gate: &bf16_tensor(device, &[m, kv * g, 0], &[]),
            key: &bf16_tensor(device, &[1, m, kv * w], &case.key),
            value: &bf16_tensor(device, &[1, m, kv * w], &case.value),
            query_norm: &f32_tensor(device, &[1, w], &case.query_norm),
            key_norm: &f32_tensor(device, &[1, w], &case.key_norm),
            value_norm: &f32_tensor(device, &[0, w], &[]),
            rotary_components: &i32_tensor(device, &[p], &case.components),
            rotary_frequencies: &f32_tensor(device, &[p], &case.frequencies),
            rotary_amplitudes: &f32_tensor(device, &[p], &vec![1.0; p]),
            coordinates: &i32_tensor(device, &[m, 4], &case.coordinates),
            visible: &i32_tensor(device, &[m, case.spans, 2], &case.visible),
            fresh: &i32_tensor(device, &[m, 2], &case.fresh),
            destinations: &i32_tensor(device, &[m], &case.destinations),
            selection: &selection,
            block_slots: &block_slots(device, BLOCK_ROWS),
            history_key: &mut history_key,
            history_value: &mut history_value,
            epsilon: case.epsilon,
            scale: case.scale,
            gate_function: 0,
            slab_rows: t as u32,
        })
        .unwrap()
        .value;
    bf16_values(&result)
}

/// The WorkList specializations of the prefill's GPU forms (the fragment
/// form: no DIRECT), with one and several key partitions.
fn specializations(device: &Device, geometry: Geometry, words: usize) -> Vec<(String, NativeSpecialization)> {
    if is_cpu(device) {
        return vec![(
            "Cpu".into(),
            qwen_form(
                NativeSpecialization::new().with_static("P", geometry.p as u64).with_static("S", geometry.s as u64),
                geometry,
            ),
        )];
    }
    let backend = device.backend();
    [(16u64, 1u64), (8, 1), (16, 256), (8, 256)]
        .into_iter()
        .map(|(query_tile, split)| {
            let base = qwen_form(statics(geometry), geometry).with_static("B", words as u64);
            let specialization = match backend {
                BackendName::Metal => base
                    .with_param("QT", query_tile)
                    .with_param("HEADS", (geometry.g.next_power_of_two() as u64).min(16))
                    .with_param("DIRECT", 0),
                BackendName::Vulkan => base.with_param("ROWS", 64),
                _ => base.with_param("QT", query_tile),
            }
            .with_param("SPLIT_GROUPS", split);
            (format!("{backend:?} QT={query_tile} SPLIT_GROUPS={split}"), specialization)
        })
        .collect()
}

fn check(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 2.0e-3 + 1.6e-2 * e.abs(),
            "{label}: gated[{index}] {a}, expected {e}"
        );
    }
}

fn case() -> (Geometry, Case) {
    let geometry = Geometry { kv: 2, g: 2, p: 4, s: 24 };
    (geometry, Case::new(geometry, 400, 2, &prefill_rows(20, 300), 31))
}

#[test]
fn portable_worklist_prefill_reads_only_selected_blocks() {
    let (_, case) = case();
    let words = 3;
    let bits = bitmap(case.rows, words, 5);
    let expected = case.expected_reading(&reads(&bits, words)).0;
    check("portable", &portable(&case, &bits, words), &expected);
}

#[test]
fn native_worklist_prefill_reads_only_selected_blocks() {
    let (geometry, case) = case();
    for (words, seed) in [(3usize, 5u64), (1, 7)] {
        // One word covers 32 blocks: keys past them are not read.
        let bits = bitmap(case.rows, words, seed);
        let expected = case.expected_reading(&reads(&bits, words)).0;
        for device in devices() {
            for (config, specialization) in specializations(&device, geometry, words) {
                let actual = native(&device, &specialization, &case, &bits, words);
                check(&format!("{config} words={words}"), &actual, &expected);
            }
        }
    }
}
