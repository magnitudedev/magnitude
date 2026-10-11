// `attention_decode` and `attention_prefill` in Qwen's form against their
// portable body (the reference interpreter, small shapes) and against a host
// model of that body (Qwen3.5-4B geometry, long contexts), on the local GPU
// (Metal, or Vulkan elsewhere) and the CPU.

use magnitude_kernels::{attention_append_dense, attention_decode, attention_prefill};
use seismic::{
    BackendName, Device, DeviceCatalog, Element, NativeSpecialization, SlabRegion, SlabTensor,
    Tensor,
};

include!("attention_common/fixtures.rs");

/// Device tensors of one case, in the family's Qwen form (see
/// `attention_k8v4.rs`).
struct Bound {
    query: Tensor,
    gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    value_norm: Tensor,
    components: Tensor,
    frequencies: Tensor,
    amplitudes: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    _history_key_slabs: SlabTensor,
    _history_value_slabs: SlabTensor,
    history_key: Tensor,
    history_value: Tensor,
    slab_rows: u32,
}

/// Slab-stored key and value history. The two tensors' slabs are allocated
/// alternately, so a slab's successor in one tensor never follows it in
/// memory (a read running past a slab's rows lands in the other tensor).
fn history_slabs(
    device: &Device,
    case: &Case,
    slab_rows: u32,
    reuse_middle: bool,
) -> [(SlabTensor, Tensor); 2] {
    let (rows, kv, width) = (case.history_rows, case.geometry.kv, case.geometry.w());
    let slab_rows = u64::from(slab_rows);
    let row_bytes = kv * width * 2;
    let mut tensors = [&case.history_key, &case.history_value].map(|values| {
        let slabs = SlabTensor::new(
            device,
            slab_rows,
            rows as u64,
            vec![SlabRegion {
                element: Element::bf16(),
                row_shape: vec![kv as u64, width as u64],
            }],
        )
        .unwrap();
        let bytes = values
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect::<Vec<_>>();
        (slabs, bytes)
    });
    for index in 0..(rows as u64).div_ceil(slab_rows) {
        for (slabs, bytes) in &mut tensors {
            slabs.add_slab().unwrap();
            let start = index * slab_rows;
            let count = slab_rows.min(rows as u64 - start);
            slabs
                .region_rows(0, start, count)
                .unwrap()
                .write_from_host(
                    &bytes[start as usize * row_bytes..(start + count) as usize * row_bytes],
                )
                .unwrap();
        }
    }
    if reuse_middle {
        assert!(rows as u64 > 2 * slab_rows);
        for (slabs, bytes) in &mut tensors {
            let old = slabs.slab(1).unwrap().observe_storage();
            slabs.free_slab(1).unwrap();
            assert_eq!(slabs.add_slab().unwrap(), 1);
            assert_ne!(
                old.identity(),
                slabs.slab(1).unwrap().observe_storage().identity()
            );
            let start = slab_rows as usize;
            slabs
                .region_rows(0, slab_rows, slab_rows)
                .unwrap()
                .write_from_host(
                    &bytes[start * row_bytes..(start + slab_rows as usize) * row_bytes],
                )
                .unwrap();
        }
    }
    tensors.map(|(slabs, _)| {
        let logical = slabs.logical_region(0).unwrap();
        (slabs, logical)
    })
}

impl Bound {
    fn new(device: &Device, case: &Case) -> Self {
        Self::new_with_slab_rows(device, case, case.history_rows as u32)
    }

    fn new_with_slab_rows(device: &Device, case: &Case, slab_rows: u32) -> Self {
        Self::new_with_options(device, case, slab_rows, false)
    }

    fn new_with_reused_slab_rows(device: &Device, case: &Case, slab_rows: u32) -> Self {
        Self::new_with_options(device, case, slab_rows, true)
    }

    fn new_with_options(device: &Device, case: &Case, slab_rows: u32, reuse_middle: bool) -> Self {
        let Geometry { kv, g, p, .. } = case.geometry;
        let (m, w) = (case.rows, case.geometry.w());
        let [(history_key_slabs, history_key), (history_value_slabs, history_value)] =
            history_slabs(device, case, slab_rows, reuse_middle);
        Self {
            query: bf16_tensor(device, &[m, kv * g, 2 * w], &case.query_gate),
            gate: bf16_tensor(device, &[m, kv * g, 0], &[]),
            key: bf16_tensor(device, &[1, m, kv * w], &case.key),
            value: bf16_tensor(device, &[1, m, kv * w], &case.value),
            query_norm: f32_tensor(device, &[1, w], &case.query_norm),
            key_norm: f32_tensor(device, &[1, w], &case.key_norm),
            value_norm: f32_tensor(device, &[0, w], &[]),
            components: i32_tensor(device, &[p], &case.components),
            frequencies: f32_tensor(device, &[p], &case.frequencies),
            amplitudes: f32_tensor(device, &[p], &vec![1.0; p]),
            coordinates: i32_tensor(device, &[m, 4], &case.coordinates),
            visible: i32_tensor(device, &[m, case.spans, 2], &case.visible),
            fresh: i32_tensor(device, &[m, 2], &case.fresh),
            destinations: i32_tensor(device, &[m], &case.destinations),
            _history_key_slabs: history_key_slabs,
            _history_value_slabs: history_value_slabs,
            history_key,
            history_value,
            slab_rows,
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
                    cpu_statics(geometry).with_param("PARTS", *parts),
                )
            })
            .collect();
    }
    configs
        .iter()
        .flat_map(|&(span, parts, simds)| {
            let base = qwen_form(statics(geometry), geometry)
                .with_param("SPAN", span)
                .with_param("PARTS", parts)
                .with_param("SIMDS", simds)
                .with_param("MATRIX", 0);
            // Metal's vector form also names its (unused) matrix key tile.
            let base = if device.backend() == BackendName::Metal {
                base.with_param("KEYS", 16).with_param("TOKENS", 1)
            } else {
                base
            };
            let label = format!(
                "{:?} (SPAN, PARTS, SIMDS) {:?}",
                device.backend(),
                (span, parts, simds)
            );
            // The query group also splits across simdgroups/subgroups: every
            // admissible slicing.
            decode_slices(geometry, simds)
                .map(|slices| {
                    (
                        format!("{label} SLICES {slices}"),
                        base.clone().with_param("SLICES", slices),
                    )
                })
                .collect::<Vec<_>>()
        })
        .chain(vulkan_matrix_specialization(device, geometry, configs[0].1))
        .chain(metal_matrix_specializations(device, geometry, configs[0].1))
        .collect()
}

/// Metal's grouped-query matrix decode form at `geometry` over `parts`
/// partitions, one row and four rows per threadgroup where admissible: a
/// simdgroup's outputs cover at most 128 columns of every 8-row block, one
/// team of the column slices (four simdgroups for one slice), 16-key tiles.
fn metal_matrix_specializations(
    device: &Device,
    geometry: Geometry,
    parts: u64,
) -> Vec<(String, NativeSpecialization)> {
    if device.backend() != BackendName::Metal {
        return Vec::new();
    }
    let (w, keys) = (geometry.w() as u64, 16u64);
    [1u64, 4]
        .into_iter()
        .filter_map(|tokens| {
            let rows = (tokens * geometry.g as u64).div_ceil(8) * 8;
            let cols = (rows / 8 * w / 128).max(1);
            let simds = if cols > 1 { cols } else { 4 };
            let exchange = if cols > 1 {
                2 * simds * rows * keys * 4
            } else {
                0
            };
            let bytes = rows * w * 2
                + (simds * keys * (w / cols + 8) * 2 + exchange).max(rows * w * 4)
                + simds / cols * rows * 8;
            (w % cols == 0 && (w / cols) % 32 == 0 && bytes <= 32768).then(|| {
                (
                    format!(
                        "Metal matrix (PARTS, SIMDS, TOKENS) {:?}",
                        (parts, simds, tokens)
                    ),
                    qwen_form(statics(geometry), geometry)
                        .with_param("SPAN", 32)
                        .with_param("PARTS", parts)
                        .with_param("SIMDS", simds)
                        .with_param("SLICES", 1)
                        .with_param("MATRIX", 1)
                        .with_param("KEYS", keys)
                        .with_param("TOKENS", tokens),
                )
            })
        })
        .collect()
}

/// Vulkan's grouped-query matrix decode form at `geometry` over `parts`
/// partitions: the fewest subgroups whose 16-row blocks hold the query
/// group.
fn vulkan_matrix_specialization(
    device: &Device,
    geometry: Geometry,
    parts: u64,
) -> Option<(String, NativeSpecialization)> {
    if device.backend() != BackendName::Vulkan {
        return None;
    }
    let simds = [1u64, 2, 4, 8]
        .into_iter()
        .find(|simds| geometry.g as u64 <= simds * 16)?;
    Some((
        format!("Vulkan matrix (PARTS, SIMDS) {:?}", (parts, simds)),
        qwen_form(statics(geometry), geometry)
            .with_param("SPAN", 32)
            .with_param("PARTS", parts)
            .with_param("SIMDS", simds)
            .with_param("SLICES", 1)
            .with_param("MATRIX", 1),
    ))
}

/// The CPU forms' statics: the head width and Qwen's form.
fn cpu_statics(geometry: Geometry) -> NativeSpecialization {
    qwen_form(
        NativeSpecialization::new()
            .with_static("P", geometry.p as u64)
            .with_static("S", geometry.s as u64),
        geometry,
    )
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
        return vec![("Cpu".to_owned(), cpu_statics(geometry))];
    }
    // Metal runs every configuration with staged K/V tiles, and query tiles
    // of whole 16-row simdgroups in the direct form too.
    let metal = device.backend() == BackendName::Metal;
    configs
        .iter()
        .flat_map(|&(query_tile, split_groups)| {
            let direct = metal && query_tile >= 16;
            [0, 1]
                .into_iter()
                .take(1 + usize::from(direct))
                .map(move |direct| ((query_tile, split_groups), direct))
        })
        .map(|((query_tile, split_groups), direct)| {
            // Vulkan tiles ROWS = 64 matrix rows (query tile x query heads), the
            // one tile admissible at every tested geometry, whatever QT asks for.
            let tile = if device.backend() == BackendName::Vulkan {
                ("ROWS", 64)
            } else {
                ("QT", query_tile)
            };
            let specialization = qwen_form(statics(geometry), geometry)
                .with_param(tile.0, tile.1)
                .with_param("SPLIT_GROUPS", split_groups);
            (
                format!(
                    "{:?} (QT, SPLIT_GROUPS, DIRECT) {:?}",
                    device.backend(),
                    (query_tile, split_groups, direct)
                ),
                // Metal splits a kv head's query heads into groups of HEADS:
                // here one group, the smallest declared value holding them.
                if device.backend() == BackendName::Metal {
                    specialization
                        .with_param(
                            "HEADS",
                            (geometry.g.next_power_of_two() as u64).min(16),
                        )
                        .with_param("DIRECT", direct)
                } else {
                    specialization
                },
            )
        })
        .collect()
}

fn decode_kernel(
    device: &Device,
    specialization: &NativeSpecialization,
) -> seismic::NativeKernel<attention_decode::Entry> {
    attention_decode::native_for_device_with(
        device,
        attention_decode::Elements { A: Element::bf16() },
        specialization,
    )
    .unwrap()
}

fn prefill_kernel(
    device: &Device,
    specialization: &NativeSpecialization,
) -> seismic::NativeKernel<attention_prefill::Entry> {
    attention_prefill::native_for_device_with(
        device,
        attention_prefill::Elements { A: Element::bf16() },
        specialization,
    )
    .unwrap()
}

/// The family entries' arguments of a bound case.
macro_rules! args {
    (attention_append_dense, $bound:expr, $case:expr) => {
        attention_append_dense::Args {
            key: &$bound.key,
            value: &$bound.value,
            key_norm: &$bound.key_norm,
            value_norm: &$bound.value_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
            rotary_amplitudes: &$bound.amplitudes,
            coordinates: &$bound.coordinates,
            destinations: &$bound.destinations,
            history_key: &mut $bound.history_key,
            history_value: &mut $bound.history_value,
            epsilon: $case.epsilon,
            slab_rows: $bound.slab_rows,
        }
    };
    ($module:ident, $bound:expr, $case:expr) => {
        $module::Args {
            query: &$bound.query,
            gate: &$bound.gate,
            key: &$bound.key,
            value: &$bound.value,
            query_norm: &$bound.query_norm,
            key_norm: &$bound.key_norm,
            value_norm: &$bound.value_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
            rotary_amplitudes: &$bound.amplitudes,
            coordinates: &$bound.coordinates,
            visible: &$bound.visible,
            fresh: &$bound.fresh,
            destinations: &$bound.destinations,
            history_key: &mut $bound.history_key,
            history_value: &mut $bound.history_value,
            epsilon: $case.epsilon,
            scale: $case.scale,
            gate_function: 0,
            slab_rows: $bound.slab_rows,
        }
    };
}

fn run_decode(
    kernel: &seismic::NativeKernel<attention_decode::Entry>,
    bound: &mut Bound,
    case: &Case,
) -> Tensor {
    kernel
        .call(args!(attention_decode, bound, case))
        .unwrap()
        .value
}

fn run_prefill(
    kernel: &seismic::NativeKernel<attention_prefill::Entry>,
    bound: &mut Bound,
    case: &Case,
) -> Tensor {
    kernel
        .call(args!(attention_prefill, bound, case))
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
            vec![m, kv * g, 2 * w],
            floats(&case.query_gate),
        ),
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
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_key)),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_value)),
        Arg::Scalar(ReferenceScalar::F32(case.epsilon.to_bits())),
        Arg::Scalar(ReferenceScalar::F32(case.scale.to_bits())),
        Arg::Scalar(ReferenceScalar::I32(0)),
        Arg::Scalar(ReferenceScalar::U32(t as u32)),
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
        (14, &expected.1, "history_key"),
        (15, &expected.2, "history_value"),
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
    check_host_model_against_portable_body("attention_decode", &case);
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
    check_host_model_against_portable_body("attention_prefill", &case);
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

#[test]
fn attention_reads_and_writes_across_history_slabs() {
    let rows = [
        Row {
            spans: vec![(0, 16)],
            fresh: (0, 1),
            destination: 49,
            position: 16,
        },
        Row {
            spans: vec![(32, 48)],
            fresh: (1, 2),
            destination: 50,
            position: 16,
        },
    ];
    let case = Case::new(SMALL, 64, 1, &rows, 41);
    check_attention_slab_case(&case, false);
}

/// Visible spans crossing slab edges, as the device measurement binds them
/// (one span over the whole synthetic history, several slabs deep): the
/// entries walk each slab's part of a span.
#[test]
fn attention_reads_spans_crossing_history_slabs() {
    let rows = [
        Row {
            spans: vec![(0, 200)],
            fresh: (0, 1),
            destination: 200,
            position: 200,
        },
        Row {
            spans: vec![(7, 150), (170, 199)],
            fresh: (1, 2),
            destination: 201,
            position: 199,
        },
    ];
    let case = Case::new(SMALL, 224, 2, &rows, 43);
    // A reused middle slab puts consecutive slabs at unrelated addresses.
    check_attention_slab_case(&case, false);
    check_attention_slab_case(&case, true);
    // Every decode configuration's walk.
    let expected = case.expected();
    for device in devices() {
        for (config, specialization) in
            decode_specializations_on(&device, SMALL, &[(32, 16, 4), (64, 8, 8), (128, 32, 4)])
        {
            for reused in [false, true] {
                let mut bound = Bound::new_with_options(&device, &case, 32, reused);
                let gated = run_decode(&decode_kernel(&device, &specialization), &mut bound, &case);
                check(
                    &format!("crossing-span decode {config} reused={reused}"),
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
fn attention_reads_and_writes_after_reusing_an_interior_slab() {
    let rows = [
        Row {
            spans: vec![(0, 16)],
            fresh: (0, 1),
            destination: 81,
            position: 16,
        },
        Row {
            spans: vec![(32, 48)],
            fresh: (1, 2),
            destination: 82,
            position: 16,
        },
    ];
    let case = Case::new(SMALL, 96, 1, &rows, 41);
    check_attention_slab_case(&case, true);
}

fn check_attention_slab_case(case: &Case, reused: bool) {
    let expected = case.expected();
    let catalog = DeviceCatalog::discover().unwrap();
    for backend in [
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
    ] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        // CUDA: the vector and the grouped-query matrix forms.
        let decode_specs = if backend == BackendName::Cuda {
            [(4, 0, 2, 1), (1, 1, 3, 2)]
                .into_iter()
                .map(|(warps, matrix, stages, columns)| {
                    qwen_form(statics(SMALL), SMALL)
                        .with_param("PARTS", 12)
                        .with_param("WARPS", warps)
                        .with_param("SLICES", 1)
                        .with_param("MATRIX", matrix)
                        .with_param("STAGES", stages)
                        .with_param("COLUMNS", columns)
                })
                .collect::<Vec<_>>()
        } else {
            vec![
                decode_specializations_on(&device, SMALL, &[(32, 16, 4)])
                    .into_iter()
                    .next()
                    .unwrap()
                    .1,
            ]
        };
        for decode_spec in &decode_specs {
            let mut bound = if reused {
                Bound::new_with_reused_slab_rows(&device, case, 32)
            } else {
                Bound::new_with_slab_rows(&device, case, 32)
            };
            let gated = run_decode(&decode_kernel(&device, decode_spec), &mut bound, case);
            check(
                &format!("{backend:?} slab decode {decode_spec:?}"),
                case,
                &gated,
                &bound,
                &expected,
            );
        }

        let prefill_spec = if backend == BackendName::Cuda {
            qwen_form(statics(SMALL), SMALL)
                .with_param("WARPS", 4)
                .with_param("SPLIT_GROUPS", 1)
                .with_param("STAGES", 3)
                .with_param("COLUMNS", 2)
                .with_param("QREG", 1)
        } else {
            prefill_specializations_on(&device, SMALL, &[(8, 1)])
                .into_iter()
                .next()
                .unwrap()
                .1
        };
        let mut bound = if reused {
            Bound::new_with_reused_slab_rows(&device, case, 32)
        } else {
            Bound::new_with_slab_rows(&device, case, 32)
        };
        let gated = run_prefill(&prefill_kernel(&device, &prefill_spec), &mut bound, case);
        check(
            &format!("{backend:?} slab prefill"),
            case,
            &gated,
            &bound,
            &expected,
        );
    }
}

/// Qwen3.5-35B-A3B: two key/value heads of eight query heads each. The
/// target stores its history in k8v4; the draft head's dense history is the
/// only dense attention at this group size.
const QWEN35B: Geometry = Geometry {
    kv: 2,
    g: 8,
    p: 32,
    s: 192,
};

#[test]
fn eight_query_group_decode_matches_host_model() {
    let case = Case::new(QWEN35B, 128, 2, &decode_rows(40), 13);
    let expected = case.expected();
    for device in devices() {
        for (config, specialization) in decode_specializations_on(
            &device,
            QWEN35B,
            &[(32, 16, 4), (64, 8, 8), (128, 16, 4), (32, 16, 8)],
        ) {
            let kernel = decode_kernel(&device, &specialization);
            let mut bound = Bound::new(&device, &case);
            let gated = run_decode(&kernel, &mut bound, &case);
            check(
                &format!("group-8 decode {config}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

#[test]
fn eight_query_group_prefill_matches_host_model() {
    let case = Case::new(QWEN35B, 400, 2, &prefill_rows(20, 300), 29);
    let expected = case.expected();
    for device in devices() {
        for (config, specialization) in
            prefill_specializations_on(&device, QWEN35B, &[(16, 1), (8, 256), (16, 256)])
        {
            let kernel = prefill_kernel(&device, &specialization);
            let mut bound = Bound::new(&device, &case);
            let gated = run_prefill(&kernel, &mut bound, &case);
            check(
                &format!("group-8 prefill {config}"),
                &case,
                &gated,
                &bound,
                &expected,
            );
        }
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
                .measure(vec![args!(attention_decode, bound, case)], &TIMING)
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
        let configs = [(16, 1), (16, 128), (16, 256), (16, 512), (8, 1), (8, 256), (32, 1), (32, 256), (32, 512)];
        for (config, specialization) in prefill_specializations_on(device, QWEN, &configs) {
            let kernel = prefill_kernel(device, &specialization);
            let mut bound = Bound::new(device, &case);
            let measurement = kernel
                .measure(vec![args!(attention_prefill, bound, case)], &TIMING)
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


/// State-only priming must publish the same K/V as ordinary attention,
/// including slab boundaries and ignored destinations, without an output.
#[test]
fn dense_append_publishes_only_the_same_history_as_prefill() {
    let rows = [
        Row { spans: vec![(0, 16)], fresh: (0, 1), destination: 31, position: 16 },
        Row { spans: vec![(0, 16)], fresh: (1, 2), destination: 32, position: 17 },
        Row { spans: vec![(0, 16)], fresh: (2, 3), destination: -1, position: 18 },
    ];
    let case = Case::new(SMALL, 64, 1, &rows, 97);
    let expected = case.expected();
    let catalog = DeviceCatalog::discover().unwrap();
    for backend in [BackendName::Cpu, BackendName::Metal] {
        let Ok(device) = catalog.open_backend(backend) else { continue; };
        let specialization = if is_cpu(&device) { cpu_statics(SMALL) }
            else { qwen_form(statics(SMALL), SMALL) };
        let append = attention_append_dense::native_for_device_with(
            &device, attention_append_dense::Elements { A: Element::bf16() }, &append_specialization(&specialization),
        ).unwrap();
        let mut appended = Bound::new_with_slab_rows(&device, &case, 32);
        append.call(args!(attention_append_dense, appended, case)).unwrap();
        let mut full = Bound::new_with_slab_rows(&device, &case, 32);
        let prefill = prefill_specializations_on(&device, SMALL, &[(16, 1)]).remove(0).1;
        let _ = run_prefill(&prefill_kernel(&device, &prefill), &mut full, &case);
        let actual_key = bf16_values(&appended.history_key);
        assert_eq!(actual_key, bf16_values(&full.history_key), "{backend:?}: keys");
        assert_eq!(bf16_values(&appended.history_value), bf16_values(&full.history_value), "{backend:?}: values");
        for (a, e) in actual_key.iter().zip(&expected.1) {
            assert!((a - e).abs() <= 8.0e-3 * e.abs().max(1.0), "{backend:?}: host key {a} != {e}");
        }
        assert_eq!(bf16_values(&appended.history_value), expected.2, "{backend:?}: host values");
        eprintln!("{backend:?}: dense append exact prefill state; host reference agrees");
    }
}

fn append_specialization(specialization: &NativeSpecialization) -> NativeSpecialization {
    specialization.statics().iter().filter(|(name, _)| !matches!(name.as_str(), "G" | "I" | "U"))
        .fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(name, *value))
}
