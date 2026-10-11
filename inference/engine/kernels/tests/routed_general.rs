//! The general routed feed-forward entries (`routed_select`, `routed_gate_up`,
//! `routed_up`, `routed_down`, `routed_experts_up`, `routed_scatter`): the
//! portable bodies define the contract; native implementations are checked
//! against them on the host's GPU backend and on the CPU, at the catalog's
//! real routing shapes (LFM2-MoE, Laguna, Gemma 4 26B, Nemotron Lightning and
//! Super).

use magnitude_kernels::{
    routed_down, routed_experts, routed_experts_up, routed_gate_up, routed_group, routed_scatter,
    routed_select, routed_up,
};
use seismic_lang::{
    checked::{check_source, CheckedModule, SourceFile},
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    for (path, text) in [
        ("routed.seismic", include_str!("../kernels/routed.seismic")),
        (
            "functions.seismic",
            include_str!("../kernels/functions.seismic"),
        ),
    ] {
        sources.push(SourceFile {
            path: path.into(),
            text: text.into(),
        });
    }
    check_source(sources).unwrap()
}

enum Input {
    Tensor(DType, Vec<usize>, Vec<f64>),
    F32(f32),
    I32(i32),
}

fn floats(dtype: DType, shape: &[usize], values: &[f32]) -> Input {
    Input::Tensor(
        dtype,
        shape.to_vec(),
        values.iter().map(|v| f64::from(*v)).collect(),
    )
}

fn ints(shape: &[usize], values: &[i32]) -> Input {
    Input::Tensor(
        DType::I32,
        shape.to_vec(),
        values.iter().map(|v| f64::from(*v)).collect(),
    )
}

fn interpret(
    module: &CheckedModule,
    name: &str,
    bindings: &[(&str, DType)],
    inputs: Vec<Input>,
) -> OracleOutcome {
    let elements = bindings
        .iter()
        .fold(ElementBindings::new(), |elements, (name, dtype)| {
            elements.bind(name, registry::dense(*dtype))
        });
    let logical = module
        .entry(module.entry_named(name).unwrap(), &elements)
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let arguments = inputs
        .into_iter()
        .map(|input| match input {
            Input::Tensor(dtype, shape, values) => {
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
            }
            Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
            Input::I32(value) => Arg::Scalar(ReferenceScalar::I32(value)),
        })
        .collect::<Vec<_>>();
    let outcome = interpreter
        .run(&arguments)
        .unwrap_or_else(|error| panic!("{name} interpreter error: {error}"));
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{name} failed: {failure}");
    }
    outcome
}

fn result(outcome: &OracleOutcome, index: usize) -> Vec<f64> {
    let result = outcome.results().nth(index).unwrap();
    let OutcomeValue::Tensor(tensor) = result.value() else {
        panic!("tensor result")
    };
    (0..tensor.element_count())
        .map(|i| tensor.read(i).unwrap())
        .collect()
}

/// Final contents of the tensor argument at parameter ordinal `ordinal`.
fn input(outcome: &OracleOutcome, ordinal: usize) -> Vec<f64> {
    let input = outcome
        .inputs()
        .find(|input| input.ordinal() == ordinal)
        .unwrap();
    let tensor = input.tensor();
    (0..tensor.element_count())
        .map(|i| tensor.read(i).unwrap())
        .collect()
}

/// Deterministic values in [-scale, scale).
fn pattern(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn element(dtype: DType) -> seismic::Element {
    match dtype {
        DType::F32 => seismic::Element::f32(),
        DType::BF16 => seismic::Element::bf16(),
        other => panic!("unsupported element {other:?}"),
    }
}

fn stored(dtype: DType, values: &[f32]) -> Vec<f32> {
    match dtype {
        DType::F32 => values.to_vec(),
        DType::BF16 => values.iter().map(|v| registry::bf16_round(*v)).collect(),
        other => panic!("unsupported element {other:?}"),
    }
}

fn tensor(
    device: &seismic::Device,
    dtype: DType,
    shape: &[usize],
    values: &[f32],
) -> seismic::Tensor {
    let bytes = match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        DType::BF16 => values
            .iter()
            .flat_map(|v| ((registry::bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        other => panic!("unsupported element {other:?}"),
    };
    let shape = shape.iter().map(|d| *d as u64).collect::<Vec<_>>();
    seismic::Tensor::from_host(device, element(dtype), &shape, &bytes).unwrap()
}

fn i32_tensor(device: &seismic::Device, shape: &[usize], values: &[i32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let shape = shape.iter().map(|d| *d as u64).collect::<Vec<_>>();
    seismic::Tensor::from_host(device, seismic::Element::i32(), &shape, &bytes).unwrap()
}

fn read_f32(tensor: &seismic::Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn read_i32(tensor: &seismic::Tensor) -> Vec<i32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn read_activation(tensor: &seismic::Tensor, dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => read_f32(tensor),
        DType::BF16 => tensor
            .read_to_host()
            .unwrap()
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        other => panic!("unsupported activation {other:?}"),
    }
}

/// CUDA, else Metal; `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan (a CUDA
/// host also has a Vulkan device). Then the CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let gpu = match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        Some("vulkan") => Some(
            catalog
                .open_backend(seismic::BackendName::Vulkan)
                .expect("the Vulkan device opens"),
        ),
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only vulkan is selectable"),
        None => [seismic::BackendName::Cuda, seismic::BackendName::Metal]
            .into_iter()
            .find_map(|backend| catalog.open_backend(backend).ok()),
    };
    gpu.into_iter()
        .chain(std::iter::once(
            catalog.open_backend(seismic::BackendName::Cpu).unwrap(),
        ))
        .collect()
}

fn is_cpu(device: &seismic::Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

fn specialization(
    device: &seismic::Device,
    statics: &[(&str, usize)],
    params: &[(&'static str, u64)],
) -> seismic::NativeSpecialization {
    let statics = if is_cpu(device) { &[][..] } else { statics };
    let specialization = statics.iter().fold(
        seismic::NativeSpecialization::new(),
        |spec, (name, value)| spec.with_static(*name, *value as u64),
    );
    params.iter().fold(specialization, |spec, (name, value)| {
        spec.with_param(*name, *value)
    })
}

/// The route's tuning parameters on `device`.
fn select_mappings(device: &seismic::Device) -> Vec<Vec<(&'static str, u64)>> {
    match device.backend() {
        seismic::BackendName::Cuda => vec![
            vec![("SIMDGROUPS", 8), ("SPLIT", 8)],
            vec![("SIMDGROUPS", 2), ("SPLIT", 4)],
        ],
        seismic::BackendName::Cpu => vec![vec![("ROWS", 8)], vec![("ROWS", 1)]],
        _ => vec![vec![("SIMDGROUPS", 8)], vec![("SIMDGROUPS", 2)]],
    }
}

/// One family's routing: hidden H, experts E, selected K, score function,
/// normalization, post-scale, and whether it has a selection bias, per-expert
/// scales and its own router norm.
#[derive(Clone, Copy)]
struct Routing {
    name: &'static str,
    hidden: usize,
    experts: usize,
    selected: usize,
    score: i32,
    normalization: i32,
    scale: f32,
    biased: bool,
    expert_scales: bool,
    router_norm: bool,
}

const CLAMP: f32 = 6.103_515_6e-5;

const ROUTINGS: [Routing; 5] = [
    Routing {
        name: "lfm2-moe",
        hidden: 2048,
        experts: 32,
        selected: 4,
        score: 1,
        normalization: 3,
        scale: 1.0,
        biased: true,
        expert_scales: false,
        router_norm: false,
    },
    Routing {
        name: "laguna",
        hidden: 3072,
        experts: 256,
        selected: 10,
        score: 1,
        normalization: 3,
        scale: 2.5,
        biased: true,
        expert_scales: false,
        router_norm: false,
    },
    Routing {
        name: "gemma4-26b",
        hidden: 2816,
        experts: 128,
        selected: 8,
        score: 0,
        normalization: 3,
        scale: 1.0,
        biased: false,
        expert_scales: true,
        router_norm: true,
    },
    Routing {
        name: "nemotron-lightning",
        hidden: 2688,
        experts: 128,
        selected: 6,
        score: 1,
        normalization: 3,
        scale: 2.5,
        biased: true,
        expert_scales: true,
        router_norm: false,
    },
    Routing {
        name: "nemotron-super",
        hidden: 4096,
        experts: 512,
        selected: 22,
        score: 1,
        normalization: 3,
        scale: 5.0,
        biased: true,
        expert_scales: false,
        router_norm: false,
    },
];

struct SelectCase {
    rows: usize,
    residual: Vec<f32>,
    norm: Vec<f32>,
    router_norm: Vec<f32>,
    router: Vec<f32>,
    bias: Vec<f32>,
    expert_scale: Vec<f32>,
}

impl SelectCase {
    /// Router rows 5 and 21 are identical and carry the same (largest) bias,
    /// so their ranked values tie exactly (the higher expert must rank
    /// first); the tied rows are large so the pair is selected. The other
    /// selection biases of sigmoid routers are O(1), as trained correction
    /// biases are, so they reorder the unbiased scores.
    fn new(routing: &Routing, rows: usize) -> Self {
        let (h, e) = (routing.hidden, routing.experts);
        let mut router = pattern(e * h, 7, 0.05);
        router[5 * h..6 * h].fill(0.02);
        router.copy_within(5 * h..6 * h, 21 * h);
        let residual = pattern(rows * h, 3, 1.0)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<_>>();
        let norm = pattern(h, 11, 0.5)
            .iter()
            .map(|v| v + 1.0)
            .collect::<Vec<_>>();
        let router_norm = if routing.router_norm {
            pattern(h, 13, 0.02).iter().map(|v| v + 0.03).collect()
        } else {
            norm.clone()
        };
        let mut bias = if routing.biased {
            pattern(e, 17, 1.0)
        } else {
            vec![0.0; e]
        };
        if routing.biased {
            bias[5] = 1.0;
            bias[21] = 1.0;
        }
        let expert_scale = if routing.expert_scales {
            pattern(e, 19, 0.5).iter().map(|v| v + 1.0).collect()
        } else {
            vec![1.0; e]
        };
        Self {
            rows,
            residual,
            norm,
            router_norm,
            router,
            bias,
            expert_scale,
        }
    }
}

/// The portable routes, weights and normalized rows.
fn portable_select(
    module: &CheckedModule,
    routing: &Routing,
    case: &SelectCase,
    activation: DType,
    router: DType,
) -> (Vec<i32>, Vec<f64>, Vec<f64>) {
    let (h, e, k, m) = (routing.hidden, routing.experts, routing.selected, case.rows);
    let outcome = interpret(
        module,
        "routed_select",
        &[
            ("NW", DType::F32),
            ("RNW", DType::F32),
            ("RW", router),
            ("A", activation),
        ],
        vec![
            floats(DType::F32, &[m, h], &case.residual),
            floats(DType::F32, &[h], &case.norm),
            floats(DType::F32, &[h], &case.router_norm),
            floats(router, &[e, h], &stored(router, &case.router)),
            floats(DType::F32, &[e], &case.bias),
            floats(DType::F32, &[e], &case.expert_scale),
            ints(&[m, k], &vec![-7; m * k]),
            floats(DType::F32, &[m, k], &vec![-7.0; m * k]),
            Input::F32(1e-6),
            Input::I32(routing.score),
            Input::I32(routing.normalization),
            Input::F32(CLAMP),
            Input::F32(routing.scale),
        ],
    );
    let routes = input(&outcome, 6).iter().map(|v| *v as i32).collect();
    (routes, input(&outcome, 7), result(&outcome, 0))
}

/// The host (f64) form of the `routed_select` contract with a BF16
/// activation: routes and weights in slot order (rank r at slot K - 1 - r),
/// the normalized rows, and the ranked value (score + bias) of every
/// (row, expert).
struct Selection {
    routes: Vec<i32>,
    weights: Vec<f64>,
    normalized: Vec<f64>,
    ranked: Vec<f64>,
}

/// A selection difference is accepted only between experts whose ranked
/// values are this close: the natives and the interpreter reassociate the RMS
/// and logit sums, which can flip a BF16 rounding of the router input (a
/// logit moves by about ulp(x) * |router| ~ 2e-4 here).
const NEAR_TIE: f64 = 1e-4;

fn host_select(routing: &Routing, case: &SelectCase) -> Selection {
    let (h, e, k, m) = (routing.hidden, routing.experts, routing.selected, case.rows);
    let rms = |norm: &[f32]| {
        case.residual
            .chunks_exact(h)
            .flat_map(|row| {
                let inverse = 1.0
                    / (row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / h as f64 + 1e-6)
                        .sqrt();
                row.iter().zip(norm).map(move |(x, w)| {
                    f64::from(registry::bf16_round(
                        (f64::from(*x) * inverse * f64::from(*w)) as f32,
                    ))
                })
            })
            .collect::<Vec<_>>()
    };
    let (normalized, router_rows) = (rms(&case.norm), rms(&case.router_norm));
    let (mut routes, mut weights, mut ranked) =
        (vec![0; m * k], vec![0.0; m * k], Vec::with_capacity(m * e));
    for row in 0..m {
        let logits = (0..e)
            .map(|expert| {
                (0..h)
                    .map(|j| router_rows[row * h + j] * f64::from(case.router[expert * h + j]))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total = logits.iter().map(|l| (l - maximum).exp()).sum::<f64>();
        let scores = logits
            .iter()
            .map(|logit| match routing.score {
                0 => (logit - maximum).exp() / total,
                1 => 1.0 / (1.0 + (-logit).exp()),
                2 => (logit.max(0.0) + (-logit.abs()).exp().ln_1p()).sqrt(),
                other => panic!("score function {other}"),
            })
            .collect::<Vec<_>>();
        let row_ranked = scores
            .iter()
            .zip(&case.bias)
            .map(|(s, b)| s + f64::from(*b))
            .collect::<Vec<_>>();
        // Descending ranked value; equal values rank the higher expert first.
        let mut order = (0..e).collect::<Vec<_>>();
        order.sort_by(|a, b| row_ranked[*b].total_cmp(&row_ranked[*a]).then(b.cmp(a)));
        for (rank, expert) in order[..k].iter().enumerate() {
            routes[row * k + k - 1 - rank] = *expert as i32;
            weights[row * k + k - 1 - rank] = scores[*expert];
        }
        let denominator = weights[row * k..(row + 1) * k].iter().sum::<f64>();
        for slot in 0..k {
            let divisor = match routing.normalization {
                0 => 1.0,
                1 => denominator,
                2 => denominator + f64::from(CLAMP),
                3 => denominator.max(f64::from(CLAMP)),
                other => panic!("normalization {other}"),
            };
            let expert = routes[row * k + slot] as usize;
            weights[row * k + slot] = weights[row * k + slot] / divisor
                * f64::from(routing.scale)
                * f64::from(case.expert_scale[expert]);
        }
        ranked.extend(row_ranked);
    }
    Selection {
        routes,
        weights,
        normalized,
        ranked,
    }
}

/// Checks a selection (native or portable) against the host reference: slot
/// by slot the same experts except near-ties, the same weights by expert when
/// the selected sets agree, and the normalized rows within one BF16 ulp.
fn assert_selection(
    label: &str,
    routing: &Routing,
    reference: &Selection,
    routes: &[i32],
    weights: &[f64],
    normalized: &[f64],
) {
    let (e, k) = (routing.experts, routing.selected);
    for row in 0..routes.len() / k {
        for slot in 0..k {
            let (actual, expected) = (routes[row * k + slot], reference.routes[row * k + slot]);
            if actual != expected {
                let gap = (reference.ranked[row * e + actual as usize]
                    - reference.ranked[row * e + expected as usize])
                    .abs();
                assert!(gap < NEAR_TIE, "{label} row {row} slot {slot}: expert {actual}, reference {expected}, gap {gap}");
            }
        }
        let mut actual_set = routes[row * k..(row + 1) * k].to_vec();
        let mut expected_set = reference.routes[row * k..(row + 1) * k].to_vec();
        actual_set.sort_unstable();
        expected_set.sort_unstable();
        if actual_set == expected_set {
            for slot in 0..k {
                let expert = routes[row * k + slot];
                let at = (0..k)
                    .find(|s| reference.routes[row * k + s] == expert)
                    .unwrap();
                let (actual, expected) = (weights[row * k + slot], reference.weights[row * k + at]);
                assert!(
                    (actual - expected).abs() <= 5e-4 * expected.abs().max(1e-3),
                    "{label} row {row} slot {slot} weight: {actual}, reference {expected}"
                );
            }
        }
    }
    for (index, (actual, expected)) in normalized.iter().zip(&reference.normalized).enumerate() {
        assert!(
            (actual - expected).abs() <= 8e-3 * expected.abs().max(1e-3),
            "{label} normalized[{index}]: {actual}, reference {expected}"
        );
    }
}

/// The GEMV (<= 8 rows) and GEMM (> 8) logit classes.
const SELECT_ROWS: [usize; 2] = [1, 9];

/// The portable body against the host reference at every routing's real
/// experts and K, over a reduced hidden size (the interpreter's cost is in
/// the router's H-long dot products, not in the selection).
#[test]
fn portable_select_matches_host_reference() {
    let module = module();
    for routing in ROUTINGS.iter().map(|routing| Routing {
        hidden: 256,
        ..*routing
    }) {
        for rows in SELECT_ROWS {
            let case = SelectCase::new(&routing, rows);
            let (routes, weights, normalized) =
                portable_select(&module, &routing, &case, DType::BF16, DType::F32);
            let label = format!("portable {} rows {rows}", routing.name);
            assert_selection(
                &label,
                &routing,
                &host_select(&routing, &case),
                &routes,
                &weights,
                &normalized,
            );
        }
    }
}

/// Every native against the host reference at the catalog's real shapes.
#[test]
fn native_select_matches_host_reference_at_catalog_shapes() {
    let cases = ROUTINGS
        .iter()
        .flat_map(|routing| SELECT_ROWS.map(|rows| (*routing, rows)))
        .map(|(routing, rows)| {
            let case = SelectCase::new(&routing, rows);
            let reference = host_select(&routing, &case);
            (routing, case, reference)
        })
        .collect::<Vec<_>>();
    for device in devices() {
        let started = std::time::Instant::now();
        for (routing, case, reference) in &cases {
            let (h, e, k, rows) = (routing.hidden, routing.experts, routing.selected, case.rows);
            for mapping in select_mappings(&device) {
                let label = format!(
                    "{:?} {} rows {rows} {mapping:?}",
                    device.backend(),
                    routing.name
                );
                let kernel = routed_select::native_for_device_with(
                    &device,
                    routed_select::Elements {
                        NW: seismic::Element::f32(),
                        RNW: seismic::Element::f32(),
                        RW: seismic::Element::f32(),
                        A: seismic::Element::bf16(),
                    },
                    &specialization(&device, &[("H", h), ("E", e), ("K", k)], &mapping),
                )
                .unwrap();
                let mut routes = i32_tensor(&device, &[rows, k], &vec![-7; rows * k]);
                let mut weights = tensor(&device, DType::F32, &[rows, k], &vec![-7.0; rows * k]);
                let outcome = kernel
                    .call(routed_select::Args {
                        residual: &tensor(&device, DType::F32, &[rows, h], &case.residual),
                        norm: &tensor(&device, DType::F32, &[h], &case.norm),
                        router_norm: &tensor(&device, DType::F32, &[h], &case.router_norm),
                        router: &tensor(&device, DType::F32, &[e, h], &case.router),
                        bias: &tensor(&device, DType::F32, &[e], &case.bias),
                        expert_scale: &tensor(&device, DType::F32, &[e], &case.expert_scale),
                        routes: &mut routes,
                        weights: &mut weights,
                        epsilon: 1e-6,
                        score: routing.score,
                        normalization: routing.normalization,
                        normalization_epsilon: CLAMP,
                        scale: routing.scale,
                    })
                    .unwrap();
                let weights = read_f32(&weights)
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                let normalized = read_activation(&outcome.value, DType::BF16)
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                assert_selection(
                    &label,
                    routing,
                    reference,
                    &read_i32(&routes),
                    &weights,
                    &normalized,
                );
            }
        }
        eprintln!(
            "{:?} selection: {:.1}s",
            device.backend(),
            started.elapsed().as_secs_f64()
        );
    }
}

/// The tied router rows 5 and 21 rank the higher expert first in the
/// portable body.
#[test]
fn portable_select_ranks_ties_to_the_higher_expert() {
    let module = module();
    let routing = Routing {
        hidden: 256,
        ..ROUTINGS[1]
    };
    let case = SelectCase::new(&routing, 1);
    let (routes, weights, _) = portable_select(&module, &routing, &case, DType::F32, DType::F32);
    let slot = |expert: i32| {
        routes
            .iter()
            .position(|r| *r == expert)
            .expect("the tied experts are selected")
    };
    let (high, low) = (slot(21), slot(5));
    // Rank r is stored at slot K - 1 - r: the higher expert ranks first.
    assert!(
        high > low,
        "expert 21 must precede expert 5: routes {routes:?}"
    );
    assert_eq!(
        weights[high], weights[low],
        "tied experts carry equal weights"
    );
}

// ---------------------------------------------------------------------------
// Expert projections: decode (`routed_gate_up` / `routed_up`, then
// `routed_down`) and prefill (`routed_group`, `routed_experts` /
// `routed_experts_up`, then `routed_scatter`).

/// One family's routed experts at its real widths: `hidden` is the experts'
/// input width (Nemotron Super's latent width), `features` the expert
/// intermediate width, `selected` K; gated (GLU) or up-only experts with
/// the `functions` activation code. A latent feed-forward also publishes
/// its zero-based sum in A (R = A), the latent-up projection's input.
#[derive(Clone, Copy)]
struct ExpertShape {
    name: &'static str,
    hidden: usize,
    features: usize,
    selected: usize,
    activation: i32,
    gated: bool,
    latent: bool,
}

const EXPERT_SHAPES: [ExpertShape; 5] = [
    ExpertShape {
        name: "lfm2-moe",
        hidden: 2048,
        features: 1792,
        selected: 4,
        activation: 0,
        gated: true,
        latent: false,
    },
    ExpertShape {
        name: "laguna",
        hidden: 3072,
        features: 1024,
        selected: 10,
        activation: 0,
        gated: true,
        latent: false,
    },
    ExpertShape {
        name: "gemma4-26b",
        hidden: 2816,
        features: 704,
        selected: 8,
        activation: 1,
        gated: true,
        latent: false,
    },
    ExpertShape {
        name: "nemotron-lightning",
        hidden: 2688,
        features: 1856,
        selected: 6,
        activation: 2,
        gated: false,
        latent: false,
    },
    ExpertShape {
        name: "nemotron-super",
        hidden: 1024,
        features: 2688,
        selected: 22,
        activation: 2,
        gated: false,
        latent: true,
    },
];

/// The published forms of an output entry: (R, base, reference): F32 over
/// the base, and for a latent feed-forward A over a zero base.
fn published_forms(shape: &ExpertShape, case: &ExpertCase) -> Vec<(DType, Vec<f32>, Vec<f32>)> {
    let mut forms = vec![(DType::F32, case.base.clone(), case.output.clone())];
    if shape.latent {
        let latent = case
            .selected
            .iter()
            .map(|v| registry::bf16_round(*v))
            .collect();
        forms.push((DType::BF16, vec![0.0; case.base.len()], latent));
    }
    forms
}

/// The experts beyond K that routes use: the expert count only indexes the
/// weights, so the test holds K + SPARE_EXPERTS experts, not the model's E.
const SPARE_EXPERTS: usize = 6;

/// Grouped tile rows T of the prefill entries.
const TILE: usize = 32;

fn activate(function: i32, a: f32) -> f32 {
    match function {
        0 => a / (1.0 + (-a).exp()),
        1 => {
            let u = 0.797_884_6 * (a + 0.044715 * (a * a * a));
            a / (1.0 + (-2.0 * u).exp())
        }
        2 => a.max(0.0) * a.max(0.0),
        other => panic!("activation {other}"),
    }
}

fn dot(x: &[f32], w: &[f32]) -> f32 {
    x.iter()
        .zip(w)
        .fold(0.0f32, |total, (x, w)| x.mul_add(*w, total))
}

/// The expanding weights of an expert block: gated (the gate rows
/// [E, F, H] beside the up rows), or up-only with each expert's
/// second-level up scale [E] on the accumulator (NVFP4 `.scale`).
enum Expansion {
    Gated(Vec<f32>),
    Up(Vec<f32>),
}

/// Dense BF16 expert weights: up [E, F, H], down [E, H, F] and the
/// expansion's gate or scales.
struct ExpertBlock {
    shape: ExpertShape,
    experts: usize,
    expansion: Expansion,
    up: Vec<f32>,
    down: Vec<f32>,
}

impl ExpertBlock {
    fn new(shape: ExpertShape) -> Self {
        let experts = shape.selected + SPARE_EXPERTS;
        let count = experts * shape.features * shape.hidden;
        let weights = |seed| {
            pattern(count, seed, 0.05)
                .into_iter()
                .map(registry::bf16_round)
                .collect::<Vec<_>>()
        };
        // Up scales in [0.5, 2), as Lightning's per-expert `ffn_up_exps.scale`.
        let expansion = if shape.gated {
            Expansion::Gated(weights(31))
        } else {
            Expansion::Up(
                pattern(experts, 59, 0.75)
                    .iter()
                    .map(|v| v + 1.25)
                    .collect(),
            )
        };
        Self {
            shape,
            experts,
            expansion,
            up: weights(37),
            down: weights(41),
        }
    }

    /// A(A(act(A(gate))) * A(up)) or A(act(A(scale[e] * up))) of one
    /// normalized row.
    fn product(&self, x: &[f32], expert: usize) -> Vec<f32> {
        let (h, f, function) = (
            self.shape.hidden,
            self.shape.features,
            self.shape.activation,
        );
        (0..f)
            .map(|feature| {
                let row = (expert * f + feature) * h;
                let up = dot(x, &self.up[row..row + h]);
                match &self.expansion {
                    Expansion::Gated(gate) => {
                        let gate = registry::bf16_round(dot(x, &gate[row..row + h]));
                        registry::bf16_round(
                            registry::bf16_round(activate(function, gate))
                                * registry::bf16_round(up),
                        )
                    }
                    Expansion::Up(scales) => registry::bf16_round(activate(
                        function,
                        registry::bf16_round(scales[expert] * up),
                    )),
                }
            })
            .collect()
    }

    /// A(down . product) [H].
    fn projection(&self, product: &[f32], expert: usize) -> Vec<f32> {
        let (h, f) = (self.shape.hidden, self.shape.features);
        (0..h)
            .map(|column| {
                let row = (expert * h + column) * f;
                registry::bf16_round(dot(product, &self.down[row..row + f]))
            })
            .collect()
    }
}

/// The per-row inputs of the expert entries and the host outcome: products
/// [M, K, F], published projections [M, K, H] and the output
/// base + sum_k w_k * published_k [M, H].
struct ExpertCase {
    rows: usize,
    normalized: Vec<f32>,
    routes: Vec<i32>,
    weights: Vec<f32>,
    base: Vec<f32>,
    products: Vec<f32>,
    published: Vec<f32>,
    /// sum_k w_k * published_k: a latent feed-forward's output (zero base).
    selected: Vec<f32>,
    output: Vec<f32>,
}

impl ExpertCase {
    /// Choice k of row m routes to expert (k + 3m) mod E: distinct within a
    /// row, and some experts receive more than one tile at prefill sizes.
    fn new(block: &ExpertBlock, rows: usize) -> Self {
        let (h, f, k, e) = (
            block.shape.hidden,
            block.shape.features,
            block.shape.selected,
            block.experts,
        );
        let normalized = pattern(rows * h, 43 + rows as u32, 1.5)
            .into_iter()
            .map(registry::bf16_round)
            .collect::<Vec<_>>();
        let routes = (0..rows * k)
            .map(|flat| ((flat % k + 3 * (flat / k)) % e) as i32)
            .collect::<Vec<_>>();
        let weights = pattern(rows * k, 47, 0.5)
            .iter()
            .map(|v| v + 0.6)
            .collect::<Vec<_>>();
        let base = pattern(rows * h, 53, 1.0);
        let choices = rows * k;
        let (mut products, mut published) = (vec![0.0; choices * f], vec![0.0; choices * h]);
        let per = choices.div_ceil(std::thread::available_parallelism().unwrap().get());
        std::thread::scope(|scope| {
            for (index, (products, published)) in products
                .chunks_mut(per * f)
                .zip(published.chunks_mut(per * h))
                .enumerate()
            {
                let (normalized, routes) = (&normalized, &routes);
                scope.spawn(move || {
                    for (offset, (product, projection)) in products
                        .chunks_exact_mut(f)
                        .zip(published.chunks_exact_mut(h))
                        .enumerate()
                    {
                        let flat = index * per + offset;
                        let (row, expert) = (flat / k, routes[flat] as usize);
                        product.copy_from_slice(
                            &block.product(&normalized[row * h..(row + 1) * h], expert),
                        );
                        projection.copy_from_slice(&block.projection(product, expert));
                    }
                });
            }
        });
        let selected = (0..rows * h)
            .map(|flat| {
                let (row, column) = (flat / h, flat % h);
                (0..k).fold(0.0f32, |selected, choice| {
                    weights[row * k + choice]
                        .mul_add(published[(row * k + choice) * h + column], selected)
                })
            })
            .collect::<Vec<_>>();
        let output = base
            .iter()
            .zip(&selected)
            .map(|(base, selected)| base + selected)
            .collect();
        Self {
            rows,
            normalized,
            routes,
            weights,
            base,
            products,
            published,
            selected,
            output,
        }
    }

    /// B for T-row tiles: ceil((M * K + min(E, M * K) * (T - 1)) / T).
    fn blocks(&self, block: &ExpertBlock, tile: usize) -> usize {
        let choices = self.rows * block.shape.selected;
        (choices + block.experts.min(choices) * (tile - 1)).div_ceil(tile)
    }
}

/// The host mirror of `routed_group`: (order [B * T], inverse [M * K],
/// blocks [B]).
fn host_group(
    routes: &[i32],
    choices: usize,
    experts: usize,
    tile: usize,
    blocks: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut order, mut inverse, mut table) = (
        vec![-1; blocks * tile],
        vec![0; routes.len()],
        vec![-1; blocks],
    );
    let mut block = 0;
    for expert in 0..experts as i32 {
        let mut lane = 0;
        for (flat, route) in routes.iter().enumerate() {
            if *route == expert {
                table[block] = expert;
                order[block * tile + lane] = (flat / choices) as i32;
                inverse[flat] = (block * tile + lane) as i32;
                lane += 1;
                if lane == tile {
                    lane = 0;
                    block += 1;
                }
            }
        }
        if lane > 0 {
            block += 1;
        }
    }
    (order, inverse, table)
}

/// Results agree within `relative * |reference| + absolute_fraction *
/// max |reference|`.
fn assert_near(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    relative: f32,
    absolute_fraction: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let scale = expected
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let (index, excess) = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            assert!(actual.is_finite(), "{label}: {actual}");
            (actual - expected).abs() - (relative * expected.abs() + absolute_fraction * scale)
        })
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |worst, (index, excess)| {
            if excess > worst.1 {
                (index, excess)
            } else {
                worst
            }
        });
    assert!(
        excess <= 0.0,
        "{label}[{index}]: native {}, reference {} (scale {scale})",
        actual[index],
        expected[index]
    );
}

/// The CPU's INT8 arithmetic variant: the tuner's relative output-norm
/// defect guard.
fn assert_norm(label: &str, actual: &[f32], expected: &[f32]) {
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).powi(2))
        .sum::<f32>();
    let scale = expected.iter().map(|e| e.powi(2)).sum::<f32>();
    assert!(
        error <= 0.05f32.powi(2) * scale,
        "{label}: relative error {}",
        (error / scale).sqrt()
    );
}

fn assert_outcome(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    int8: bool,
    relative: f32,
    absolute_fraction: f32,
) {
    if int8 {
        assert_norm(label, actual, expected);
    } else {
        assert_near(label, actual, expected, relative, absolute_fraction);
    }
}

fn gather_rows(values: &[f32], positions: &[i32], width: usize) -> Vec<f32> {
    positions
        .iter()
        .flat_map(|position| {
            values[*position as usize * width..(*position as usize + 1) * width]
                .iter()
                .copied()
        })
        .collect()
}

/// The portable bodies against the host reference at reduced widths with
/// every family's K and activation: decode at 3 rows, prefill at 5 rows over
/// 4-row tiles.
#[test]
fn portable_expert_entries_match_host_reference() {
    let module = module();
    for shape in EXPERT_SHAPES.map(|shape| ExpertShape {
        hidden: 64,
        features: 32,
        ..shape
    }) {
        let block = ExpertBlock::new(shape);
        let (h, f, k, e) = (shape.hidden, shape.features, shape.selected, block.experts);
        let bf16 = |shape: &[usize], values: &[f32]| floats(DType::BF16, shape, values);
        let weights =
            |values: &[f32], rows: usize, columns: usize| bf16(&[e, rows, columns], values);
        let elements = |names: &[&'static str]| {
            names
                .iter()
                .map(|name| (*name, DType::BF16))
                .collect::<Vec<_>>()
        };

        let case = ExpertCase::new(&block, 3);
        let m = case.rows;
        let product = match &block.expansion {
            Expansion::Gated(gate) => interpret(
                &module,
                "routed_gate_up",
                &elements(&["A", "EGW", "EUW"]),
                vec![
                    bf16(&[m, h], &case.normalized),
                    ints(&[m, k], &case.routes),
                    weights(gate, f, h),
                    weights(&block.up, f, h),
                    Input::I32(shape.activation),
                ],
            ),
            Expansion::Up(scales) => interpret(
                &module,
                "routed_up",
                &elements(&["A", "EUW"]),
                vec![
                    bf16(&[m, h], &case.normalized),
                    ints(&[m, k], &case.routes),
                    weights(&block.up, f, h),
                    floats(DType::F32, &[e], scales),
                    Input::I32(shape.activation),
                ],
            ),
        };
        let product = result(&product, 0)
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        assert_near(
            &format!("portable {} product", shape.name),
            &product,
            &case.products,
            1e-2,
            1e-3,
        );
        for (published, base, expected) in published_forms(&shape, &case) {
            let down = interpret(
                &module,
                "routed_down",
                &[("A", DType::BF16), ("EDW", DType::BF16), ("R", published)],
                vec![
                    floats(DType::F32, &[m, h], &base),
                    bf16(&[m, k, f], &case.products),
                    ints(&[m, k], &case.routes),
                    floats(DType::F32, &[m, k], &case.weights),
                    weights(&block.down, h, f),
                ],
            );
            let down = result(&down, 0)
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>();
            assert_near(
                &format!("portable {} down {published:?}", shape.name),
                &down,
                &expected,
                4e-3,
                1e-4,
            );
        }

        let (case, tile) = (ExpertCase::new(&block, 5), 4);
        let (m, b) = (case.rows, case.blocks(&block, tile));
        let (order, inverse, table) = host_group(&case.routes, k, e, tile, b);
        let grouped = match &block.expansion {
            Expansion::Gated(gate) => interpret(
                &module,
                "routed_experts",
                &elements(&["A", "EGW", "EUW", "EDW"]),
                vec![
                    bf16(&[m, h], &case.normalized),
                    ints(&[b, tile], &order),
                    ints(&[b], &table),
                    weights(gate, f, h),
                    weights(&block.up, f, h),
                    weights(&block.down, h, f),
                    Input::I32(shape.activation),
                ],
            ),
            Expansion::Up(scales) => interpret(
                &module,
                "routed_experts_up",
                &elements(&["A", "EUW", "EDW"]),
                vec![
                    bf16(&[m, h], &case.normalized),
                    ints(&[b, tile], &order),
                    ints(&[b], &table),
                    weights(&block.up, f, h),
                    weights(&block.down, h, f),
                    floats(DType::F32, &[e], scales),
                    Input::I32(shape.activation),
                ],
            ),
        };
        let grouped = result(&grouped, 0)
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        // `routed_experts` forms act(gate) * up unrounded (the Qwen body):
        // each product term differs by up to a BF16 ulp of its factors.
        assert_near(
            &format!("portable {} grouped", shape.name),
            &gather_rows(&grouped, &inverse, h),
            &case.published,
            2e-2,
            5e-3,
        );
        for (published, base, expected) in published_forms(&shape, &case) {
            let scattered = interpret(
                &module,
                "routed_scatter",
                &[("A", DType::BF16), ("R", published)],
                vec![
                    floats(DType::F32, &[m, h], &base),
                    bf16(
                        &[b, tile, h],
                        &scatter_table(&case.published, &inverse, b * tile, h),
                    ),
                    ints(&[m, k], &inverse),
                    floats(DType::F32, &[m, k], &case.weights),
                ],
            );
            let scattered = result(&scattered, 0)
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>();
            assert_near(
                &format!("portable {} scatter {published:?}", shape.name),
                &scattered,
                &expected,
                4e-3,
                1e-4,
            );
        }
    }
}

/// The grouped expert outputs [B * T, H] holding each choice's published
/// projection at its tile position (zeros at padding).
fn scatter_table(published: &[f32], inverse: &[i32], positions: usize, width: usize) -> Vec<f32> {
    let mut table = vec![0.0; positions * width];
    for (flat, position) in inverse.iter().enumerate() {
        let position = *position as usize;
        table[position * width..(position + 1) * width]
            .copy_from_slice(&published[flat * width..(flat + 1) * width]);
    }
    table
}

/// The decode mappings of the expert entries on `device`.
fn decode_mappings(device: &seismic::Device) -> Vec<Vec<(&'static str, u64)>> {
    match device.backend() {
        seismic::BackendName::Cuda => {
            vec![
                vec![("TPW", 1), ("KSPLIT", 1)],
                vec![("TPW", 2), ("KSPLIT", 2)],
                vec![("TPW", 1), ("KSPLIT", 4)],
            ]
        }
        seismic::BackendName::Vulkan => vec![
            vec![("SIMDGROUPS", 2), ("ROWS", 1)],
            vec![("SIMDGROUPS", 8), ("ROWS", 4)],
        ],
        seismic::BackendName::Cpu => vec![vec![("ROWS", 8)], vec![("ROWS", 1)]],
        seismic::BackendName::Metal => vec![
            vec![("SIMDGROUPS", 2), ("ROWS", 1), ("LANES", 32)],
            vec![("SIMDGROUPS", 4), ("ROWS", 4), ("LANES", 16)],
            vec![("SIMDGROUPS", 16), ("ROWS", 1), ("LANES", 16)],
        ],
    }
}

/// The mapping of `routed_down` beside decode mapping `index` for experts of
/// `features` columns: Metal tiles its channels itself (ROWS channels per
/// lane group of LANES lanes, at most 8 weight packets per lane); elsewhere
/// the expansion's mapping.
fn down_mapping(
    device: &seismic::Device,
    index: usize,
    mapping: &[(&'static str, u64)],
    features: usize,
) -> Vec<(&'static str, u64)> {
    match device.backend() {
        seismic::BackendName::Metal => {
            let admissible = [(2u64, 16u64), (4, 32), (1, 16), (8, 16), (1, 32)]
                .into_iter()
                .filter(|(rows, lanes)| (features as u64).div_ceil(32 * lanes) * rows <= 8)
                .collect::<Vec<_>>();
            let (rows, lanes) = admissible[index % admissible.len()];
            vec![("ROWS", rows), ("LANES", lanes)]
        }
        _ => mapping.to_vec(),
    }
}

/// The grouped mappings of the expert entries on `device`.
fn grouped_mappings(device: &seismic::Device) -> Vec<Vec<(&'static str, u64)>> {
    match device.backend() {
        seismic::BackendName::Cuda => vec![vec![]],
        seismic::BackendName::Cpu => vec![vec![("ROWS", 8)], vec![("ROWS", 2)]],
        seismic::BackendName::Vulkan => vec![
            vec![
                ("TILE_M", 32),
                ("TILE_N", 128),
                ("SUB_M", 32),
                ("SUB_N", 32),
            ],
            vec![("TILE_M", 64), ("TILE_N", 64), ("SUB_M", 64), ("SUB_N", 64)],
        ],
        seismic::BackendName::Metal => vec![
            vec![("TILE_M", 32), ("TILE_N", 128)],
            vec![("TILE_M", 64), ("TILE_N", 64)],
        ],
    }
}

/// The CPU runs every mapping in both arithmetic variants.
fn arithmetic_variants(device: &seismic::Device) -> Vec<bool> {
    if is_cpu(device) {
        vec![false, true]
    } else {
        vec![false]
    }
}

/// Metal and CUDA decode projections own their parameters on their sole
/// launch; the CPU also takes its arithmetic variant.
fn expert_specialization(
    device: &seismic::Device,
    statics: &[(&str, usize)],
    mapping: &[(&'static str, u64)],
    launch_scoped: bool,
    int8: bool,
) -> seismic::NativeSpecialization {
    let specialization = if launch_scoped
        && matches!(
            device.backend(),
            seismic::BackendName::Metal | seismic::BackendName::Cuda
        ) {
        mapping.iter().fold(
            specialization(device, statics, &[]),
            |spec, (name, value)| spec.with_launch_param(0, *name, *value),
        )
    } else {
        specialization(device, statics, mapping)
    };
    if is_cpu(device) {
        specialization.with_param("INT8", u64::from(int8))
    } else {
        specialization
    }
}

/// The device tensors of an `ExpertBlock` (`expansion`: the gate rows or
/// the up scales).
struct DeviceExperts {
    expansion: DeviceExpansion,
    up: seismic::Tensor,
    down: seismic::Tensor,
}

enum DeviceExpansion {
    Gated(seismic::Tensor),
    Up(seismic::Tensor),
}

impl DeviceExperts {
    fn new(device: &seismic::Device, block: &ExpertBlock) -> Self {
        let (h, f, e) = (block.shape.hidden, block.shape.features, block.experts);
        let expansion = match &block.expansion {
            Expansion::Gated(gate) => {
                DeviceExpansion::Gated(tensor(device, DType::BF16, &[e, f, h], gate))
            }
            Expansion::Up(scales) => DeviceExpansion::Up(tensor(device, DType::F32, &[e], scales)),
        };
        Self {
            expansion,
            up: tensor(device, DType::BF16, &[e, f, h], &block.up),
            down: tensor(device, DType::BF16, &[e, h, f], &block.down),
        }
    }
}

/// Decode (1 and 8 rows) at every family's real expert widths: the native
/// expansion against the host products, and the native output (from the
/// host products) against the host output.
#[test]
fn native_decode_experts_match_host_reference_at_catalog_shapes() {
    let devices = devices();
    for shape in EXPERT_SHAPES {
        let block = ExpertBlock::new(shape);
        let cases = [1, 8].map(|rows| ExpertCase::new(&block, rows));
        let (h, f, k) = (shape.hidden, shape.features, shape.selected);
        let statics = [("H", h), ("K", k), ("F", f)];
        for device in &devices {
            let started = std::time::Instant::now();
            let experts = DeviceExperts::new(device, &block);
            let bf16 = seismic::Element::bf16();
            for case in &cases {
                let m = case.rows;
                let normalized = tensor(device, DType::BF16, &[m, h], &case.normalized);
                let routes = i32_tensor(device, &[m, k], &case.routes);
                for (index, mapping) in decode_mappings(device).into_iter().enumerate() {
                    let down = down_mapping(device, index, &mapping, f);
                    for int8 in arithmetic_variants(device) {
                        let label = format!(
                            "{:?} {} rows {m} {mapping:?} down {down:?} INT8 {int8}",
                            device.backend(),
                            shape.name
                        );
                        let specialization =
                            expert_specialization(device, &statics, &mapping, true, int8);
                        let down_specialization =
                            expert_specialization(device, &statics, &down, true, int8);
                        let product = match &experts.expansion {
                            DeviceExpansion::Gated(gate) => {
                                routed_gate_up::native_for_device_with(
                                    device,
                                    routed_gate_up::Elements {
                                        A: bf16,
                                        EGW: bf16,
                                        EUW: bf16,
                                    },
                                    &specialization,
                                )
                                .unwrap()
                                .call(routed_gate_up::Args {
                                    normalized: &normalized,
                                    routes: &routes,
                                    expert_gate: gate,
                                    expert_up: &experts.up,
                                    activation: shape.activation,
                                })
                                .unwrap()
                                .value
                            }
                            DeviceExpansion::Up(scales) => {
                                routed_up::native_for_device_with(
                                    device,
                                    routed_up::Elements { A: bf16, EUW: bf16 },
                                    &specialization,
                                )
                                .unwrap()
                                .call(routed_up::Args {
                                    normalized: &normalized,
                                    routes: &routes,
                                    expert_up: &experts.up,
                                    up_scale: scales,
                                    activation: shape.activation,
                                })
                                .unwrap()
                                .value
                            }
                        };
                        let product = read_activation(&product, DType::BF16);
                        assert_outcome(
                            &format!("{label} product"),
                            &product,
                            &case.products,
                            int8,
                            2e-2,
                            2e-3,
                        );
                        for (published, base, expected) in published_forms(&shape, case) {
                            let output = routed_down::native_for_device_with(
                                device,
                                routed_down::Elements {
                                    A: bf16,
                                    EDW: bf16,
                                    R: element(published),
                                },
                                &down_specialization,
                            )
                            .unwrap()
                            .call(routed_down::Args {
                                base: &tensor(device, DType::F32, &[m, h], &base),
                                product: &tensor(device, DType::BF16, &[m, k, f], &case.products),
                                routes: &routes,
                                weights: &tensor(device, DType::F32, &[m, k], &case.weights),
                                expert_down: &experts.down,
                            })
                            .unwrap()
                            .value;
                            let actual = read_activation(&output, published);
                            assert_outcome(
                                &format!("{label} output {published:?}"),
                                &actual,
                                &expected,
                                int8,
                                1e-2,
                                2e-3,
                            );
                        }
                    }
                }
            }
            eprintln!(
                "{:?} {} decode: {:.1}s",
                device.backend(),
                shape.name,
                started.elapsed().as_secs_f64()
            );
        }
    }
}

/// Prefill (64 rows, 32-row tiles, B at the expert-count-aware bound) at
/// every family's real expert widths: the native grouping against the host
/// tables, the grouped outputs against the host projections, and the native
/// scatter (of the native grouped outputs) against the host output.
#[test]
fn native_grouped_experts_match_host_reference_at_catalog_shapes() {
    let devices = devices();
    for shape in EXPERT_SHAPES {
        let block = ExpertBlock::new(shape);
        let case = ExpertCase::new(&block, 64);
        let (h, f, k, e, m) = (
            shape.hidden,
            shape.features,
            shape.selected,
            block.experts,
            case.rows,
        );
        let b = case.blocks(&block, TILE);
        let (order, inverse, table) = host_group(&case.routes, k, e, TILE, b);
        for device in &devices {
            let started = std::time::Instant::now();
            let experts = DeviceExperts::new(device, &block);
            let bf16 = seismic::Element::bf16();
            let routes = i32_tensor(device, &[m, k], &case.routes);
            let fill =
                |shape: &[usize]| i32_tensor(device, shape, &vec![-9; shape.iter().product()]);
            let (mut counts, mut native_order, mut native_inverse, mut native_table) =
                (fill(&[e]), fill(&[b, TILE]), fill(&[m, k]), fill(&[b]));
            let group = if is_cpu(device) {
                seismic::NativeSpecialization::new()
            } else {
                specialization(device, &[("E", e), ("K", k)], &[("PARTS", 4)])
            };
            routed_group::native_for_device(device, &group)
                .unwrap()
                .call(routed_group::Args {
                    routes: &routes,
                    counts: &mut counts,
                    order: &mut native_order,
                    inverse: &mut native_inverse,
                    blocks: &mut native_table,
                })
                .unwrap();
            let label = format!("{:?} {}", device.backend(), shape.name);
            assert_eq!(read_i32(&native_order), order, "{label} order");
            assert_eq!(read_i32(&native_inverse), inverse, "{label} inverse");
            assert_eq!(read_i32(&native_table), table, "{label} blocks");
            let normalized = tensor(device, DType::BF16, &[m, h], &case.normalized);
            for mapping in grouped_mappings(device) {
                for int8 in arithmetic_variants(device) {
                    let label = format!("{label} {mapping:?} INT8 {int8}");
                    let specialization =
                        expert_specialization(device, &[("H", h), ("F", f)], &mapping, false, int8);
                    let grouped = match &experts.expansion {
                        DeviceExpansion::Gated(gate) => {
                            routed_experts::native_for_device_with(
                                device,
                                routed_experts::Elements {
                                    A: bf16,
                                    EGW: bf16,
                                    EUW: bf16,
                                    EDW: bf16,
                                },
                                &specialization,
                            )
                            .unwrap()
                            .call(routed_experts::Args {
                                normalized: &normalized,
                                order: &native_order,
                                blocks: &native_table,
                                expert_gate: gate,
                                expert_up: &experts.up,
                                expert_down: &experts.down,
                                activation: shape.activation,
                            })
                            .unwrap()
                            .value
                        }
                        DeviceExpansion::Up(scales) => {
                            routed_experts_up::native_for_device_with(
                                device,
                                routed_experts_up::Elements {
                                    A: bf16,
                                    EUW: bf16,
                                    EDW: bf16,
                                },
                                &specialization,
                            )
                            .unwrap()
                            .call(routed_experts_up::Args {
                                normalized: &normalized,
                                order: &native_order,
                                blocks: &native_table,
                                expert_up: &experts.up,
                                expert_down: &experts.down,
                                up_scale: scales,
                                activation: shape.activation,
                            })
                            .unwrap()
                            .value
                        }
                    };
                    let rows = gather_rows(&read_activation(&grouped, DType::BF16), &inverse, h);
                    // `routed_experts` forms A(act(gate) * up) unrounded on the
                    // CPU (as its portable body and Qwen's CPU bits), while the
                    // reference rounds gate and up (the GPU forms): each of the
                    // F down terms differs by up to a BF16 ulp of its product.
                    let absolute = if shape.gated { 1e-2 } else { 2e-3 };
                    assert_outcome(
                        &format!("{label} grouped"),
                        &rows,
                        &case.published,
                        int8,
                        2e-2,
                        absolute,
                    );
                    for (published, base, expected) in published_forms(&shape, &case) {
                        let output = routed_scatter::native_for_device_with(
                            device,
                            routed_scatter::Elements {
                                A: bf16,
                                R: element(published),
                            },
                            &seismic::NativeSpecialization::new(),
                        )
                        .unwrap()
                        .call(routed_scatter::Args {
                            base: &tensor(device, DType::F32, &[m, h], &base),
                            expert_output: &grouped,
                            inverse: &native_inverse,
                            weights: &tensor(device, DType::F32, &[m, k], &case.weights),
                        })
                        .unwrap()
                        .value;
                        let actual = read_activation(&output, published);
                        let label = format!("{label} output {published:?}");
                        assert_outcome(&label, &actual, &expected, int8, 1e-2, absolute);
                    }
                }
            }
            eprintln!("{label} grouped: {:.1}s", started.elapsed().as_secs_f64());
        }
    }
}
