//! Routed (mixture-of-experts) entries: the portable bodies define the
//! contract; native implementations are checked against them.

#![allow(
    clippy::disallowed_methods,
    reason = "a test starts threads of its own"
)]

mod common;
mod decision;
mod declared;

use common::bound::{self, Float, Form, Library, Products};
use seismic::BackendName;

use magnitude_kernels::{
    SlotTable, constants, routed_combine, routed_copy, routed_down, routed_expand,
    gated_delta_project, routed_experts, routed_experts_up, routed_gate_up, routed_gate_up_part, routed_group,
    routed_hint_route, routed_hint_select, routed_land, routed_list_grid, routed_output, routed_pass_map, routed_round,
    routed_route, routed_route_shared, routed_route_streamed, routed_scratch_grids, routed_scratch_request,
    routed_seed_commit, routed_select, routed_select_streamed, routed_step, routed_up,
};
use seismic_lang::{
    checked::CheckedModule,
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

mod mapped;
mod operands;
mod packed;
mod streaming_words;

use mapped::Mapped;
use streaming_words::{map_words, step_words};

const H: usize = 64;
const E: usize = 32;
const K: usize = 4;

enum Input {
    Tensor(DType, Vec<usize>, Vec<f64>),
    F32(f32),
    I32(i32),
}

fn floats(shape: &[usize], values: &[f32]) -> Input {
    Input::Tensor(
        DType::F32,
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
        panic!("{name} failed: {failure:?}");
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

struct RouteCase {
    rows: usize,
    experts: usize,
    residual: Vec<f32>,
    norm: Vec<f32>,
    router: Vec<f32>,
    shared_router: Vec<f32>,
    normalize: i32,
}

impl RouteCase {
    /// Router rows 5 and `experts - 11` (21 at 32 experts) are identical, so
    /// their probabilities tie exactly and the higher expert must rank
    /// first. Residuals are positive and the tied rows a positive constant,
    /// so the pair is always selected. Rows 9 and `experts - 2` tie too,
    /// without being forced into the top K.
    fn new(rows: usize, normalize: i32, experts: usize) -> Self {
        let mut router = pattern(experts * H, 7, 0.25);
        router[5 * H..6 * H].fill(0.3);
        router.copy_within(5 * H..6 * H, (experts - 11) * H);
        router.copy_within(9 * H..10 * H, (experts - 2) * H);
        Self {
            rows,
            experts,
            residual: pattern(rows * H, 3, 1.0)
                .into_iter()
                .map(|v| v + 1.5)
                .collect(),
            norm: pattern(H, 11, 0.5).into_iter().map(|v| v + 1.0).collect(),
            router,
            shared_router: pattern(H, 13, 0.2),
            normalize,
        }
    }

    fn portable(
        &self,
        module: &CheckedModule,
        activation: DType,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        self.portable_scaled(module, activation, &[])
    }

    /// (normalized, coefficient, routes, scores) with per-expert score
    /// scales `expert_scale` (none when empty: the absent port).
    fn portable_scaled(
        &self,
        module: &CheckedModule,
        activation: DType,
        expert_scale: &[f32],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let rows = self.rows;
        let outcome = interpret(
            module,
            "routed_route",
            &[("NW", DType::F32), ("RW", DType::F32), ("SRW", DType::F32), ("A", activation)],
            vec![
                floats(&[rows, H], &self.residual),
                floats(&[H], &self.norm),
                floats(&[self.experts, H], &self.router),
                floats(&[H], &self.shared_router),
                ints(&[rows, K], &vec![0; rows * K]),
                floats(&[rows, K], &vec![0.0; rows * K]),
                Input::F32(1e-6),
                Input::I32(self.normalize),
                floats(&[expert_scale.len()], expert_scale),
            ],
        );
        (
            result(&outcome, 0),
            result(&outcome, 1),
            input(&outcome, 4),
            input(&outcome, 5),
        )
    }
}

#[test]
fn portable_route_ranks_by_probability_with_ties_to_the_higher_expert() {
    let module = crate::common::kernels();
    let case = RouteCase::new(3, 0, E);
    let (_, _, routes, scores) = case.portable(&module, DType::F32);
    for row in 0..case.rows {
        let routes = &routes[row * K..(row + 1) * K];
        let scores = &scores[row * K..(row + 1) * K];
        // Ascending-probability slot order; equal probabilities put the
        // higher expert in the later slot (it ranks first).
        for slot in 1..K {
            assert!(
                scores[slot - 1] <= scores[slot],
                "row {row} slot order {scores:?}"
            );
            if scores[slot - 1] == scores[slot] {
                assert!(
                    routes[slot - 1] < routes[slot],
                    "row {row} tie order {routes:?}"
                );
            }
        }
        // The amplified tied pair is always selected, 21 ranked above 5.
        let five = routes
            .iter()
            .position(|&e| e == 5.0)
            .expect("expert 5 selected");
        let twenty_one = routes
            .iter()
            .position(|&e| e == 21.0)
            .expect("expert 21 selected");
        assert_eq!(scores[five], scores[twenty_one]);
        assert_eq!(twenty_one, five + 1, "row {row}: {routes:?}");
    }
    let (_, _, _, normalized_scores) = RouteCase::new(3, 1, E).portable(&module, DType::F32);
    for row in 0..3 {
        let total: f64 = normalized_scores[row * K..(row + 1) * K].iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "row {row} normalized sum {total}"
        );
    }
}

fn read_f32(tensor: &seismic::Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_i32(tensor: &seismic::Tensor) -> Vec<i32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let val = (mant as f32) * 2f32.powi(-24);
            if sign != 0 { -val } else { val }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

fn read_activation(tensor: &seismic::Tensor, dtype: DType) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    match dtype {
        DType::F32 => read_f32(tensor),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        other => panic!("unsupported activation {other:?}"),
    }
}

fn f32_tensor(device: &seismic::Device, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(&seismic::Charges::tally(device), seismic::Element::f32(), shape, &bytes).unwrap()
}

/// A gated expansion's host port `glu_limit` without a clamp: `[0, LB]` at
/// LB = 0.
fn no_limit(device: &seismic::Device) -> seismic::Tensor {
    i32_tensor(device, &[0, 0], &[])
}

/// An absent scale port (`[0] f32`): an unscaled weight's.
fn absent(device: &seismic::Device) -> seismic::Tensor {
    scale_port(device, &[])
}

/// An absent map (`[0, MAP_WORDS] i32`): every expert read from its stacks
/// by index.
fn absent_map(device: &seismic::Device) -> seismic::Tensor {
    i32_tensor(device, &[0, constants::MAP_WORDS], &[])
}

/// The step port beside an absent map (`[0] i32`, QS 0).
fn absent_step(device: &seismic::Device) -> seismic::Tensor {
    i32_tensor(device, &[0], &[])
}

/// A resident decode entry's sources of `rows` rows of `choices` choices:
/// none (`XS` 0), each choice reading its expert's stack row
/// (`decode_source`).
fn no_sources(device: &seismic::Device, rows: u64, choices: u64) -> seismic::Tensor {
    i32_tensor(device, &[0, rows, choices], &[])
}

/// A decode step's sources `[SOURCE_ROWS, rows, choices]` (`resolve`'s
/// snapshot): each choice's record for the GPU (`records`, row-major; a
/// bank record, -2 - t for tier record t, or -1), none the CPU's.
fn decode_sources(device: &seismic::Device, rows: u64, choices: u64, records: &[i32]) -> seismic::Tensor {
    assert_eq!(records.len(), (rows * choices) as usize, "a record a choice");
    i32_tensor(device, &[constants::SOURCE_ROWS, rows, choices], &magnitude_kernels::source_words(records, &vec![-1; records.len()]))
}

/// A decode entry's host tier records of `stack`'s tensor where it has no
/// tier: a view of none of its records (`TR` 0).
fn no_tier(stack: &seismic::Tensor) -> seismic::Tensor {
    stack.slice_leading(0, 0).unwrap()
}

/// A per-expert scale port holding `scales` (absent when empty).
fn scale_port(device: &seismic::Device, scales: &[f32]) -> seismic::Tensor {
    f32_tensor(device, &[scales.len() as u64], scales)
}

/// Distinct per-expert second-level scales (NVFP4 `.scale`), none a power
/// of two, cycled over `experts`.
fn expert_scales(experts: usize) -> Vec<f32> {
    const SCALES: [f32; 4] = [0.4125, 0.7312, 1.3688, 0.2075];
    (0..experts)
        .map(|expert| SCALES[expert % 4] * (1.0 + 0.01 * (expert / 4) as f32))
        .collect()
}

/// The blocks B of `tile` rows that hold `choices` grouped choices when at
/// most `receivers` experts receive one (`routed_group`): a block per
/// `tile` choices, plus one partial block per receiving expert. The
/// executor's bound takes min(E, choices) receivers; E receivers pads for
/// every expert.
fn group_blocks(choices: u64, receivers: u64, tile: u64) -> u64 {
    (choices + receivers * (tile - 1)).div_ceil(tile)
}

/// The first row of a grouping of every row: row 0.
fn first_row(device: &seismic::Device) -> seismic::Tensor {
    i32_tensor(device, &[1], &[0])
}

/// A streamed layer's key records over its `experts` experts, zeroed, which
/// its every route writes.
fn key_records(device: &seismic::Device, experts: u64) -> seismic::Tensor {
    i32_tensor(device, &[FLIGHT, constants::VERIFY_ROWS, experts], &vec![0; (FLIGHT * constants::VERIFY_ROWS * experts) as usize])
}

fn i32_tensor(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(&seismic::Charges::tally(device), seismic::Element::i32(), shape, &bytes).unwrap()
}

/// Each value within `tolerance` of the portable one, relative above 1; a
/// non-finite portable value is matched exactly (NaN by NaN).
fn assert_close(name: &str, actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(*actual);
        let matched = if expected.is_finite() {
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0)
        } else {
            actual == *expected || (actual.is_nan() && expected.is_nan())
        };
        assert!(
            matched,
            "{name}[{index}]: native {actual}, portable {expected}"
        );
    }
}

/// The GPU device under test: the first available of CUDA, Metal and
/// Vulkan, or the backend `SEISMIC_TEST_BACKEND` selects
/// (`seismic_native_tests::devices`).
fn native_device() -> Option<seismic::Device> {
    use seismic::BackendName::{Cuda, Metal, Vulkan};
    seismic_native_tests::devices(&[Cuda, Metal, Vulkan]).into_iter().next()
}

/// `item(index)` of every index below `count`, in index order, computed
/// across the host's cores: the host references are independent per row,
/// case and packet row.
fn parallel<T: Send>(count: usize, item: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(count);
    let mut items = std::thread::scope(|scope| {
        let work = || {
            let mut taken = Vec::new();
            loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index >= count {
                    return taken;
                }
                taken.push((index, item(index)));
            }
        };
        let workers = (0..workers).map(|_| scope.spawn(work)).collect::<Vec<_>>();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
            .collect::<Vec<_>>()
    });
    items.sort_unstable_by_key(|(index, _)| *index);
    items.into_iter().map(|(_, item)| item).collect()
}

fn is_cpu(device: &seismic::Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

/// The specialization of an entry on `device`: CPU implementations have no
/// static dimensions, only their mapping parameters.
fn specialization_on(
    device: &seismic::Device,
    statics: &[(&str, u64)],
    params: &[(&'static str, u64)],
) -> seismic::NativeSpecialization {
    if is_cpu(device) {
        return specialize(&[], params);
    }
    specialize(statics, params)
}

/// The group's specialization on `device` with `parts` (the CPU form has
/// one work item and no parameters).
fn group_specialization(
    device: &seismic::Device,
    experts: u64,
    choices: u64,
    parts: Option<u64>,
) -> seismic::NativeSpecialization {
    if is_cpu(device) {
        return seismic::NativeSpecialization::new();
    }
    let params = parts
        .map(|parts| vec![("PARTS", parts)])
        .unwrap_or_default();
    specialize(&[("E", experts), ("K", choices)], &params)
}

/// The group's parts parameter on `device`: Vulkan tunes it; Metal and CUDA
/// derive their parts from E.
fn group_parts(device: &seismic::Device) -> Vec<Option<u64>> {
    match device.backend() {
        seismic::BackendName::Vulkan => vec![Some(1), Some(2), Some(4)],
        _ => vec![None],
    }
}

/// The group's specialization with the device's default parts.
fn default_group_specialization(
    device: &seismic::Device,
    experts: u64,
    choices: u64,
) -> seismic::NativeSpecialization {
    let parts = match device.backend() {
        seismic::BackendName::Vulkan => Some(4),
        _ => None,
    };
    group_specialization(device, experts, choices, parts)
}

/// The route's hidden-axis splits on the device's backend (CUDA splits its
/// router GEMM; Metal has no split parameter).
fn route_splits(device: &seismic::Device) -> Vec<u64> {
    let _ = device;
    vec![1]
}

/// The route's mapping parameters for `simdgroups` and `split` (CPU and CUDA
/// fix their ordered router mapping, so neither sweeps `SIMDGROUPS`).
fn route_mapping(
    device: &seismic::Device,
    simdgroups: u64,
    split: u64,
) -> Vec<(&'static str, u64)> {
    let _ = split;
    match device.backend() {
        seismic::BackendName::Cpu | seismic::BackendName::Cuda => Vec::new(),
        _ => vec![("SIMDGROUPS", simdgroups)],
    }
}

/// The SIMDGROUPS of `simdgroups` a route mapping sweep runs on `device`:
/// CPU and CUDA run their single fixed mapping once.
fn route_simdgroups<'a>(device: &seismic::Device, simdgroups: &'a [u64]) -> &'a [u64] {
    if is_cpu(device) || device.backend() == seismic::BackendName::Cuda {
        &simdgroups[..1]
    } else {
        simdgroups
    }
}

#[test]
fn native_route_matches_portable_body_for_every_mapping() {
    let checks = RouteCheck::all();
    for device in seismic_native_tests::every_device() {
        native_route_matches_portable_body_for_every_mapping_on(&device, &checks);
    }
}

/// Expert counts of the route checks: a multiple of 32, a small odd count
/// (one partial lane run, every lane's single register) and Nemotron 3.5
/// Bolt's smallest routed layer (240: lanes 16..31 exclude their last
/// register).
const ROUTE_EXPERTS: [usize; 3] = [E, 23, 240];

/// One route check: the activation, case and per-expert score scales, with
/// the portable body's (normalized, coefficient, routes, scores), which no
/// device changes.
struct RouteCheck {
    activation: DType,
    element: seismic::Element,
    expert_scale: Vec<f32>,
    case: RouteCase,
    portable: (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>),
}

impl RouteCheck {
    /// Every route check, each portable body interpreted once for every
    /// device.
    fn all() -> Vec<Self> {
        let checks = [
            (DType::F32, seismic::Element::f32()),
            (DType::BF16, seismic::Element::bf16()),
        ]
        .into_iter()
        .flat_map(|(activation, element)| ROUTE_EXPERTS.map(|experts| (activation, element, experts)))
        .flat_map(|(activation, element, experts)| {
            [(1usize, 1), (5, 0), (9, 1)].into_iter().flat_map(move |(rows, normalize)| {
                [Vec::new(), expert_scales(experts)].map(|expert_scale| {
                    (activation, element, expert_scale, RouteCase::new(rows, normalize, experts))
                })
            })
        })
        .collect::<Vec<_>>();
        let module = crate::common::kernels();
        let portable = parallel(checks.len(), |index| {
            let (activation, _, expert_scale, case) = &checks[index];
            case.portable_scaled(&module, *activation, expert_scale)
        });
        checks
            .into_iter()
            .zip(portable)
            .map(|((activation, element, expert_scale, case), portable)| Self {
                activation,
                element,
                expert_scale,
                case,
                portable,
            })
            .collect()
    }
}

fn native_route_matches_portable_body_for_every_mapping_on(device: &seismic::Device, checks: &[RouteCheck]) {
    let charges = seismic::Charges::tally(device);
    for RouteCheck { activation, element, expert_scale, case, portable } in checks {
        let (activation, element) = (*activation, *element);
        if !is_cpu(device) && activation == DType::F32 {
            continue;
        }
        let (rows, normalize, experts) = (case.rows, case.normalize, case.experts);
        let (normalized, coefficient, routes, scores) = portable;
        for (simdgroups, split) in route_simdgroups(device, &[8, 4, 2]).iter().copied().flat_map(|simdgroups| {
            route_splits(&device)
                .into_iter()
                .map(move |split| (simdgroups, split))
        }) {
            let specialization = specialization_on(
                device,
                &route_statics(
                    &[("H", H as u64), ("E", experts as u64), ("K", K as u64)],
                    H as u64,
                    expert_scale.len() as u64,
                ),
                &route_mapping(&device, simdgroups, split),
            );
            let kernel = declared::prepared(
                &device,
                &routed_route::native_entry_with(routed_route::Elements {
                    NW: seismic::Element::f32(),
                    RW: seismic::Element::f32(),
                    SRW: seismic::Element::f32(),
                    A: element,
                }),
                &specialization,
            )
            .unwrap();
            let mut native_routes =
                i32_tensor(&device, &[rows as u64, K as u64], &vec![-7; rows * K]);
            let mut native_scores =
                f32_tensor(&device, &[rows as u64, K as u64], &vec![-7.0; rows * K]);
            let outcome = kernel
                .call(&charges, routed_route::Args {
                    residual: &f32_tensor(&device, &[rows as u64, H as u64], &case.residual),
                    norm: &f32_tensor(&device, &[H as u64], &case.norm),
                    router: &f32_tensor(&device, &[experts as u64, H as u64], &case.router),
                    shared_router: &f32_tensor(&device, &[H as u64], &case.shared_router),
                    routes: &mut native_routes,
                    scores: &mut native_scores,
                    eps: 1e-6,
                    normalize,
                    expert_scale: &scale_port(device, &expert_scale),
                })
                .unwrap();
            let label = format!(
                "{activation:?} E {experts} ES {} rows {rows} SIMDGROUPS {simdgroups} SPLIT {split}",
                expert_scale.len()
            );
            let expected_routes = routes.iter().map(|v| *v as i32).collect::<Vec<_>>();
            assert_eq!(read_i32(&native_routes), expected_routes, "{label} routes");
            assert_close(
                &format!("{label} scores"),
                &read_f32(&native_scores),
                &scores,
                1e-5,
            );
            let rounding = if activation == DType::F32 { 1e-5 } else { 8e-3 };
            assert_close(
                &format!("{label} normalized"),
                &read_activation(&outcome.r0, activation),
                &normalized,
                rounding,
            );
            assert_close(
                &format!("{label} coefficient"),
                &read_f32(&outcome.r1),
                &coefficient,
                1e-4,
            );
        }
    }
}

/// The rows a non-finite selection check poisons: row 1 holds a NaN and row
/// 3 an infinity in its residual, so every logit of either row is NaN.
const POISONED_ROWS: [usize; 2] = [1, 3];

fn poison(residual: &mut [f32]) {
    residual[POISONED_ROWS[0] * H + 3] = f32::NAN;
    residual[POISONED_ROWS[1] * H] = f32::INFINITY;
}

/// The K highest experts of `experts` in slot order: a poisoned row's
/// choices, every ranked value -inf.
fn highest(experts: usize, choices: usize) -> Vec<i32> {
    (experts - choices..experts).map(|expert| expert as i32).collect()
}

/// A route whose poisoned rows rank every probability (NaN) as -inf, so each
/// routes to the K highest experts, distinct; with `zeros` every router row
/// but the tied pair (5, E - 11) has a -inf logit, so the other rows' last
/// two ranks are probabilities tied at 0, the highest experts first. Every
/// native routes as the portable body, in both logit classes (GEMV, GEMM).
#[test]
fn native_route_ranks_nan_as_negative_infinity() {
    let module = crate::common::kernels();
    let cases = ROUTE_EXPERTS
        .into_iter()
        .flat_map(|experts| [(5, false), (9, true)].map(|(rows, zeros)| (experts, rows, zeros)))
        .map(|(experts, rows, zeros)| {
            let mut case = RouteCase::new(rows, if zeros { 0 } else { 1 }, experts);
            poison(&mut case.residual);
            if zeros {
                for expert in (0..experts).filter(|e| *e != 5 && *e != experts - 11) {
                    case.router[expert * H] = f32::NEG_INFINITY;
                }
            }
            let portable_f32 = case.portable(&module, DType::F32);
            let portable_bf16 = case.portable(&module, DType::BF16);
            for (tag, portable) in [("f32", &portable_f32), ("bf16", &portable_bf16)] {
                let routes = portable.2.iter().map(|v| *v as i32).collect::<Vec<_>>();
                for row in 0..rows {
                    let expected = if POISONED_ROWS.contains(&row) {
                        highest(experts, K)
                    } else if zeros {
                        let e = experts as i32;
                        vec![e - 2, e - 1, 5, e - 11]
                    } else {
                        continue;
                    };
                    assert_eq!(routes[row * K..(row + 1) * K], expected, "portable {tag} E {experts} row {row}");
                }
            }
            (case, portable_f32, portable_bf16)
        })
        .collect::<Vec<_>>();
    for device in seismic_native_tests::every_device() {
        let charges = seismic::Charges::tally(&device);
        for (case, portable_f32, portable_bf16) in &cases {
            let (rows, experts) = (case.rows, case.experts);
            let label = format!("{:?} E {experts} rows {rows}", device.backend());
            let (act_dtype, act_elem, (normalized, coefficient, routes, scores)) = if is_cpu(&device) {
                (DType::F32, seismic::Element::f32(), portable_f32)
            } else {
                (DType::BF16, seismic::Element::bf16(), portable_bf16)
            };
            let kernel = declared::prepared(
                &device,
                &routed_route::native_entry_with(routed_route::Elements {
                    NW: seismic::Element::f32(),
                    RW: seismic::Element::f32(),
                    SRW: seismic::Element::f32(),
                    A: act_elem,
                }),
                &specialization_on(
                    &device,
                    &route_statics(&[("H", H as u64), ("E", experts as u64), ("K", K as u64)], H as u64, 0),
                    &route_mapping(&device, 8, route_splits(&device)[0]),
                ),
            )
            .unwrap();
            let mut native_routes = i32_tensor(&device, &[rows as u64, K as u64], &vec![-7; rows * K]);
            let mut native_scores = f32_tensor(&device, &[rows as u64, K as u64], &vec![-7.0; rows * K]);
            let outcome = kernel
                .call(&charges, routed_route::Args {
                    residual: &f32_tensor(&device, &[rows as u64, H as u64], &case.residual),
                    norm: &f32_tensor(&device, &[H as u64], &case.norm),
                    router: &f32_tensor(&device, &[experts as u64, H as u64], &case.router),
                    shared_router: &f32_tensor(&device, &[H as u64], &case.shared_router),
                    routes: &mut native_routes,
                    scores: &mut native_scores,
                    eps: 1e-6,
                    normalize: case.normalize,
                    expert_scale: &absent(&device),
                })
                .unwrap();
            let expected_routes = routes.iter().map(|v| *v as i32).collect::<Vec<_>>();
            assert_eq!(read_i32(&native_routes), expected_routes, "{label} routes");
            assert_close(&format!("{label} scores"), &read_f32(&native_scores), scores, 1e-5);
            assert_close(&format!("{label} normalized"), &read_activation(&outcome.r0, act_dtype), normalized, 1e-5);
            assert_close(&format!("{label} coefficient"), &read_f32(&outcome.r1), coefficient, 1e-4);
        }
    }
}

/// The general selection ranks score + bias with a NaN as -inf: +inf biases
/// (experts 2 and 10) rank first, the higher first; then the finite ones
/// (experts 2 mod 8, biases -e, well apart), led by 18 and 26, which tie
/// (equal router rows, biases 0), the higher first; then NaN (even experts)
/// and -inf (odd) biases rank together as
/// -inf, the highest experts first (K reaches three into them). A poisoned
/// row's every score is NaN, so it routes to the K highest experts. Every
/// native routes as the portable body, in both logit classes, at a multiple
/// of 32 experts and at one whose last lanes hold an excluded entry.
#[test]
fn native_select_ranks_nan_as_negative_infinity() {
    const CLAMP: f32 = 6.103_515_6e-5;
    let module = crate::common::kernels();
    let cases = [32usize, 240]
        .into_iter()
        .flat_map(|experts| [5usize, 9].map(|rows| (experts, rows)))
        .map(|(experts, rows)| {
            let choices = experts / 8 + 3;
            let mut router = pattern(experts * H, 7, 0.25);
            router.copy_within(18 * H..19 * H, 26 * H);
            let bias = (0..experts)
                .map(|expert| match expert {
                    2 | 10 => f32::INFINITY,
                    18 | 26 => 0.0,
                    _ if expert % 8 == 2 => -(expert as f32),
                    _ if expert % 2 == 0 => f32::NAN,
                    _ => f32::NEG_INFINITY,
                })
                .collect::<Vec<_>>();
            let mut residual = pattern(rows * H, 3, 1.0).into_iter().map(|v| v + 1.5).collect::<Vec<_>>();
            poison(&mut residual);
            let norm = pattern(H, 11, 0.5).into_iter().map(|v| v + 1.0).collect::<Vec<_>>();
            let run_portable = |act: DType| {
                let bindings = [("NW", DType::F32), ("RNW", DType::F32), ("RW", DType::F32), ("A", act)];
                let outcome = interpret(
                    &module,
                    "routed_select",
                    &bindings,
                    vec![
                        floats(&[rows, H], &residual),
                        floats(&[H], &norm),
                        floats(&[H], &norm),
                        floats(&[experts, H], &router),
                        floats(&[experts], &bias),
                        floats(&[0], &[]),
                        ints(&[rows, choices], &vec![-7; rows * choices]),
                        floats(&[rows, choices], &vec![-7.0; rows * choices]),
                        Input::F32(1e-6),
                        Input::I32(1),
                        Input::I32(3),
                        Input::F32(CLAMP),
                        Input::F32(1.0),
                    ],
                );
                let routes = input(&outcome, 6).iter().map(|v| *v as i32).collect::<Vec<_>>();
                let e = experts as i32;
                for row in 0..rows {
                    let routes = &routes[row * choices..(row + 1) * choices];
                    if POISONED_ROWS.contains(&row) {
                        assert_eq!(routes, highest(experts, choices), "portable {act:?} E {experts} row {row}");
                    } else {
                        assert_eq!(routes[..3], [e - 3, e - 2, e - 1], "portable {act:?} E {experts} row {row}");
                        assert_eq!(routes[choices - 4..], [18, 26, 2, 10], "portable {act:?} E {experts} row {row}");
                    }
                }
                (routes, input(&outcome, 7), result(&outcome, 0))
            };
            let portable_f32 = run_portable(DType::F32);
            let portable_bf16 = run_portable(DType::BF16);
            (experts, rows, choices, router, bias, residual, norm, portable_f32, portable_bf16)
        })
        .collect::<Vec<_>>();
    for device in seismic_native_tests::every_device() {
        let charges = seismic::Charges::tally(&device);
        for (experts, rows, choices, router, bias, residual, norm, portable_f32, portable_bf16) in &cases {
            let (experts, rows, choices) = (*experts, *rows, *choices);
            let label = format!("{:?} E {experts} rows {rows}", device.backend());
            let (act_dtype, act_elem, (routes, weights, normalized)) = if is_cpu(&device) {
                (DType::F32, seismic::Element::f32(), portable_f32)
            } else {
                (DType::BF16, seismic::Element::bf16(), portable_bf16)
            };
            let kernel = declared::prepared(
                &device,
                &routed_select::native_entry_with(routed_select::Elements {
                    NW: seismic::Element::f32(),
                    RNW: seismic::Element::f32(),
                    RW: seismic::Element::f32(),
                    A: act_elem,
                }),
                &specialization_on(
                    &device,
                    &[("H", H as u64), ("E", experts as u64), ("K", choices as u64), ("ES", 0)],
                    &route_mapping(&device, 8, route_splits(&device)[0]),
                ),
            )
            .unwrap();
            let shape = [rows as u64, choices as u64];
            let mut native_routes = i32_tensor(&device, &shape, &vec![-7; rows * choices]);
            let mut native_weights = f32_tensor(&device, &shape, &vec![-7.0; rows * choices]);
            let norm = f32_tensor(&device, &[H as u64], norm);
            let outcome = kernel
                .call(&charges, routed_select::Args {
                    residual: &f32_tensor(&device, &[rows as u64, H as u64], residual),
                    norm: &norm,
                    router_norm: &norm,
                    router: &f32_tensor(&device, &[experts as u64, H as u64], router),
                    bias: &f32_tensor(&device, &[experts as u64], bias),
                    expert_scale: &absent(&device),
                    routes: &mut native_routes,
                    weights: &mut native_weights,
                    epsilon: 1e-6,
                    score: 1,
                    normalization: 3,
                    normalization_epsilon: CLAMP,
                    scale: 1.0,
                })
                .unwrap();
            assert_eq!(&read_i32(&native_routes), routes, "{label} routes");
            assert_close(&format!("{label} weights"), &read_f32(&native_weights), weights, 1e-5);
            assert_close(&format!("{label} normalized"), &read_activation(&outcome.value, act_dtype), normalized, 1e-5);
        }
    }
}

struct GroupCase {
    rows: usize,
    experts: usize,
    tile: usize,
    routes: Vec<i32>,
}

impl GroupCase {
    /// Distinct experts per row, deliberately skewed so some experts fill
    /// several tiles and others none.
    fn new(rows: usize, experts: usize, tile: usize) -> Self {
        let mut routes = Vec::with_capacity(rows * K);
        for row in 0..rows {
            let mut chosen = Vec::new();
            let mut candidate = (row * 7 + row * row) % experts;
            while chosen.len() < K {
                let skewed = if row % 3 == 0 {
                    candidate % 5
                } else {
                    candidate
                };
                if !chosen.contains(&(skewed as i32)) {
                    chosen.push(skewed as i32);
                }
                candidate = (candidate + 3) % experts;
            }
            routes.extend(chosen);
        }
        Self {
            rows,
            experts,
            tile,
            routes,
        }
    }

    fn blocks(&self) -> usize {
        group_blocks((self.rows * K) as u64, self.experts as u64, self.tile as u64) as usize
    }

    /// (counts, order, inverse, blocks, used) from the portable body.
    fn portable(&self, module: &CheckedModule) -> [Vec<i32>; 5] {
        let (rows, blocks, tile) = (self.rows, self.blocks(), self.tile);
        let outcome = interpret(
            module,
            "routed_group",
            &[],
            vec![
                ints(&[1], &[0]),
                ints(&[1], &[rows as i32]),
                ints(&[rows, K], &self.routes),
                ints(&[self.experts], &vec![0; self.experts]),
                ints(&[blocks, tile], &vec![0; blocks * tile]),
                ints(&[rows, K], &vec![0; rows * K]),
                ints(&[blocks], &vec![0; blocks]),
                ints(&[1], &[0]),
            ],
        );
        [3, 4, 5, 6, 7].map(|ordinal| {
            input(&outcome, ordinal)
                .into_iter()
                .map(|v| v as i32)
                .collect()
        })
    }

    /// (counts, order, inverse, blocks, used) of the native grouping of the
    /// first `grouped` rows of a graph of `capacity` rows (the case's rows,
    /// then rows routed to expert 0).
    fn native(&self, device: &seismic::Device, parts: Option<u64>, grouped: usize, capacity: usize) -> [Vec<i32>; 5] {
        let charges = seismic::Charges::tally(device);
        let (experts, tile) = (self.experts as u64, self.tile as u64);
        let blocks = group_blocks((capacity * K) as u64, experts, tile);
        let fill = |shape: &[u64]| i32_tensor(device, shape, &vec![-9; shape.iter().product::<u64>() as usize]);
        let mut routes = self.routes.clone();
        routes.resize(capacity * K, 0);
        let (mut counts, mut order) = (fill(&[experts]), fill(&[blocks, tile]));
        let (mut inverse, mut table, mut used) = (fill(&[capacity as u64, K as u64]), fill(&[blocks]), fill(&[1]));
        declared::prepared(device, &routed_group::native_entry(), &group_specialization(device, experts, K as u64, parts))
            .unwrap()
            .call(&charges, routed_group::Args {
                first: &first_row(device),
                rows: &i32_tensor(device, &[1], &[grouped as i32]),
                routes: &i32_tensor(device, &[capacity as u64, K as u64], &routes),
                counts: &mut counts,
                order: &mut order,
                inverse: &mut inverse,
                blocks: &mut table,
                used: &mut used,
            })
            .unwrap();
        [counts, order, inverse, table, used].map(|tensor| read_i32(&tensor))
    }

    /// The tables of the grouping's contract by direct construction (the
    /// portable body's, at row counts its interpretation is too slow for).
    fn host(&self) -> [Vec<i32>; 5] {
        let (tile, experts) = (self.tile, self.experts);
        let blocks = self.blocks();
        let mut counts = vec![0; experts];
        for route in &self.routes {
            counts[*route as usize] += 1;
        }
        let (mut order, mut inverse, mut table) = (vec![-1; blocks * tile], vec![0; self.rows * K], vec![-1; blocks]);
        let mut next = Vec::with_capacity(experts);
        let mut block = 0;
        for (expert, count) in counts.iter().enumerate() {
            let tiles = (*count as usize).div_ceil(tile);
            next.push(block * tile);
            table[block..block + tiles].fill(expert as i32);
            block += tiles;
        }
        for (flat, route) in self.routes.iter().enumerate() {
            let position = &mut next[*route as usize];
            order[*position] = (flat / K) as i32;
            inverse[flat] = *position as i32;
            *position += 1;
        }
        [counts, order, inverse, table, vec![block as i32]]
    }

    /// The tables' contract: blocks past `used` carry none, so their `order`
    /// and `blocks` entries are dropped.
    fn contract(&self, [counts, order, inverse, blocks, used]: [Vec<i32>; 5]) -> [Vec<i32>; 5] {
        let live = used[0] as usize;
        [counts, order[..live * self.tile].to_vec(), inverse, blocks[..live].to_vec(), used]
    }

    /// Every choice maps to a tile row of its own expert holding its row,
    /// experts own contiguous ascending block ranges, and every other slot
    /// of a used block is padding.
    fn check_round_trip(&self, tables: [Vec<i32>; 5]) {
        let [counts, order, inverse, blocks, used] = &self.contract(tables);
        let tile = self.tile as i32;
        let mut claimed = vec![false; order.len()];
        for row in 0..self.rows {
            for choice in 0..K {
                let flat = row * K + choice;
                let position = inverse[flat];
                assert_eq!(
                    order[position as usize], row as i32,
                    "order of choice {flat}"
                );
                assert_eq!(
                    blocks[(position / tile) as usize],
                    self.routes[flat],
                    "expert of choice {flat}"
                );
                assert!(
                    !claimed[position as usize],
                    "position {position} claimed twice"
                );
                claimed[position as usize] = true;
            }
        }
        for (position, claimed) in claimed.iter().enumerate() {
            if !claimed {
                assert_eq!(order[position], -1, "padding at {position}");
            }
        }
        assert_eq!(used[0] as usize, blocks.len(), "used blocks");
        assert!(blocks.iter().all(|&b| b >= 0), "used blocks have experts");
        assert!(blocks.windows(2).all(|w| w[0] <= w[1]), "experts ascend");
        for expert in 0..self.experts as i32 {
            let count = self.routes.iter().filter(|&&e| e == expert).count() as i32;
            assert_eq!(counts[expert as usize], count, "count of expert {expert}");
            let tiles = blocks.iter().filter(|&&b| b == expert).count() as i32;
            assert_eq!(tiles, (count + tile - 1) / tile, "tiles of expert {expert}");
        }
    }
}

#[test]
fn portable_group_is_a_stable_expert_permutation() {
    let module = crate::common::kernels();
    for (rows, experts, tile) in [(1, 8, 4), (13, 8, 4), (37, 16, 8), (64, 32, 16)] {
        let case = GroupCase::new(rows, experts, tile);
        let tables = case.portable(&module);
        case.check_round_trip(tables.clone());
        // Stable: within an expert, flat (row, choice) order.
        let inverse = &tables[2];
        for expert in 0..experts as i32 {
            let positions = (0..rows * K)
                .filter(|&flat| case.routes[flat] == expert)
                .map(|flat| inverse[flat])
                .collect::<Vec<_>>();
            assert!(
                positions.windows(2).all(|w| w[0] < w[1]),
                "expert {expert} stable"
            );
        }
    }
}

#[test]
fn native_group_matches_portable_tables() {
    for device in seismic_native_tests::every_device() {
        native_group_matches_portable_tables_on(&device);
    }
}

fn native_group_matches_portable_tables_on(device: &seismic::Device) {
    let module = crate::common::kernels();
    for (rows, experts, tile) in [
        (1, 8, 4),
        (13, 8, 4),
        (37, 16, 8),
        (64, 32, 16),
        (200, 32, 32),
        (512, 256, 32),
    ] {
        let case = GroupCase::new(rows, experts, tile);
        let expected = case.contract(case.portable(&module));
        assert_eq!(case.contract(case.host()), expected, "host tables: rows {rows} experts {experts} tile {tile}");
        for parts in group_parts(device) {
            let actual = case.contract(case.native(device, parts, rows, rows));
            for (name, (actual, expected)) in ["counts", "order", "inverse", "blocks", "used"]
                .iter()
                .zip(actual.iter().zip(&expected))
            {
                assert_eq!(
                    actual, expected,
                    "{name}: rows {rows} experts {experts} tile {tile} parts {parts:?}"
                );
            }
        }
    }
}

/// The grouping spans as many chunks as its rows need (a single
/// threadgroup's memory held at most ~1,720 rows on Metal), and a graph
/// sealed at a larger capacity groups only its first rows, into the same
/// tables: the tables of 512 rows inside 16,384 are those of 512 alone.
#[test]
fn native_group_spans_chunks_and_groups_rows_below_capacity() {
    for device in seismic_native_tests::every_device() {
        for rows in [512, 4096, 16_384] {
            let case = GroupCase::new(rows, 256, 32);
            let expected = case.contract(case.host());
            for parts in group_parts(&device) {
                let actual = case.contract(case.native(&device, parts, rows, rows));
                assert_eq!(actual, expected, "rows {rows} parts {parts:?}");
            }
        }
        let case = GroupCase::new(512, 256, 32);
        let [counts, order, inverse, blocks, used] =
            case.contract(case.native(&device, group_parts(&device)[0], 512, 512));
        let [c, o, i, b, u] = case.contract(case.native(&device, group_parts(&device)[0], 512, 16_384));
        assert_eq!((c, o, b, u), (counts, order, blocks, used), "tables below capacity");
        assert_eq!(i[..512 * K], inverse[..], "inverse below capacity");
    }
}

/// A small routed block with f32 weights and activations.
struct Block {
    rows: usize,
    hidden: usize,
    experts: usize,
    choices: usize,
    features: usize,
    shared: usize,
    residual: Vec<f32>,
    norm: Vec<f32>,
    router: Vec<f32>,
    shared_router: Vec<f32>,
    expert_gate: Vec<f32>,
    expert_up: Vec<f32>,
    expert_down: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_down: Vec<f32>,
    /// The experts' per-expert gate, up and down scales (empty: the absent
    /// ports; else one per expert).
    gate_scale: Vec<f32>,
    up_scale: Vec<f32>,
    down_scale: Vec<f32>,
}

impl Block {
    /// [`Block::new`] with distinct per-expert gate, up and down scales.
    fn scaled(rows: usize) -> Self {
        let block = Self::new(rows);
        let scales = |offset: usize| {
            let scales = expert_scales(block.experts + offset);
            scales[offset..].to_vec()
        };
        Self {
            gate_scale: scales(0),
            up_scale: scales(1),
            down_scale: scales(2),
            ..block
        }
    }

    fn new(rows: usize) -> Self {
        let (hidden, experts, choices, features, shared) = (16, 8, 3, 8, 8);
        Self {
            rows,
            hidden,
            experts,
            choices,
            features,
            shared,
            residual: pattern(rows * hidden, 21, 1.5),
            norm: pattern(hidden, 22, 0.3)
                .into_iter()
                .map(|v| v + 1.0)
                .collect(),
            router: pattern(experts * hidden, 23, 0.6),
            shared_router: pattern(hidden, 24, 0.4),
            expert_gate: pattern(experts * features * hidden, 25, 0.3),
            expert_up: pattern(experts * features * hidden, 26, 0.3),
            expert_down: pattern(experts * hidden * features, 27, 0.3),
            shared_gate: pattern(shared * hidden, 28, 0.3),
            shared_up: pattern(shared * hidden, 29, 0.3),
            shared_down: pattern(hidden * shared, 30, 0.3),
            gate_scale: Vec::new(),
            up_scale: Vec::new(),
            down_scale: Vec::new(),
        }
    }

    fn values(values: &[f64]) -> Vec<f32> {
        values.iter().map(|v| *v as f32).collect()
    }

    /// (normalized, coefficient, routes, scores)
    fn route(&self, module: &CheckedModule) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<f32>) {
        let (m, h, e, k) = (self.rows, self.hidden, self.experts, self.choices);
        let outcome = interpret(
            module,
            "routed_route",
            &[("NW", DType::F32), ("RW", DType::F32), ("SRW", DType::F32), ("A", DType::F32)],
            vec![
                floats(&[m, h], &self.residual),
                floats(&[h], &self.norm),
                floats(&[e, h], &self.router),
                floats(&[h], &self.shared_router),
                ints(&[m, k], &vec![0; m * k]),
                floats(&[m, k], &vec![0.0; m * k]),
                Input::F32(1e-6),
                Input::I32(1),
                floats(&[self.down_scale.len()], &self.down_scale),
            ],
        );
        (
            Self::values(&result(&outcome, 0)),
            Self::values(&result(&outcome, 1)),
            input(&outcome, 4).into_iter().map(|v| v as i32).collect(),
            Self::values(&input(&outcome, 5)),
        )
    }

    fn decode(&self, module: &CheckedModule) -> Vec<f32> {
        let (m, h, e, k, f, s) = (
            self.rows,
            self.hidden,
            self.experts,
            self.choices,
            self.features,
            self.shared,
        );
        let (normalized, coefficient, routes, scores) = self.route(module);
        let expanded = interpret(
            module,
            "routed_expand",
            &[
                ("A", DType::F32),
                ("EGW", DType::F32),
                ("EUW", DType::F32),
                ("SGW", DType::F32),
                ("SUW", DType::F32),
            ],
            vec![
                floats(&[m, h], &normalized),
                ints(&[m, k], &routes),
                floats(&[e, f, h], &self.expert_gate),
                floats(&[e, f, h], &self.expert_up),
                floats(&[s, h], &self.shared_gate),
                floats(&[s, h], &self.shared_up),
                floats(&[self.gate_scale.len()], &self.gate_scale),
                floats(&[self.up_scale.len()], &self.up_scale),
            ],
        );
        let output = interpret(
            module,
            "routed_output",
            &[("A", DType::F32), ("EDW", DType::F32), ("SDW", DType::F32), ("NW", DType::F32)],
            vec![
                floats(&[m, h], &self.residual),
                floats(&[m, 0], &[]),
                floats(&[m, k, f], &Self::values(&result(&expanded, 0))),
                floats(&[m, s], &Self::values(&result(&expanded, 1))),
                ints(&[m, k], &routes),
                ints(&[0, m, k], &[]),
                floats(&[m, k], &scores),
                floats(&[m], &coefficient),
                floats(&[e, h, f], &self.expert_down),
                floats(&[0, h, f], &[]),
                floats(&[h, s], &self.shared_down),
                floats(&[0], &[]),
                Input::F32(0.0),
            ],
        );
        Self::values(&result(&output, 0))
    }

    fn prefill(&self, module: &CheckedModule, tile: usize) -> Vec<f32> {
        let (m, h, e, k, f, s) = (
            self.rows,
            self.hidden,
            self.experts,
            self.choices,
            self.features,
            self.shared,
        );
        let (normalized, coefficient, routes, scores) = self.route(module);
        let blocks = group_blocks((m * k) as u64, e as u64, tile as u64) as usize;
        let grouped = interpret(
            module,
            "routed_group",
            &[],
            vec![
                ints(&[1], &[0]),
                ints(&[1], &[m as i32]),
                ints(&[m, k], &routes),
                ints(&[e], &vec![0; e]),
                ints(&[blocks, tile], &vec![0; blocks * tile]),
                ints(&[m, k], &vec![0; m * k]),
                ints(&[blocks], &vec![0; blocks]),
                ints(&[1], &[0]),
            ],
        );
        let tables = |ordinal| {
            input(&grouped, ordinal)
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>()
        };
        let (order, inverse, block_experts, used) = (tables(4), tables(5), tables(6), tables(7));
        let experts = interpret(
            module,
            "routed_experts",
            &[
                ("A", DType::F32),
                ("EGW", DType::F32),
                ("EUW", DType::F32),
                ("EDW", DType::F32),
            ],
            vec![
                ints(&[1], &[0]),
                ints(&[1], &[m as i32]),
                floats(&[m, h], &normalized),
                ints(&[blocks, tile], &order),
                ints(&[blocks], &block_experts),
                ints(&[1], &used),
                floats(&[e, f, h], &self.expert_gate),
                floats(&[e, f, h], &self.expert_up),
                floats(&[e, h, f], &self.expert_down),
                ints(&[0, constants::MAP_WORDS as usize], &[]),
                ints(&[0], &[]),
                floats(&[e, f, h], &self.expert_gate),
                floats(&[e, f, h], &self.expert_up),
                floats(&[e, h, f], &self.expert_down),
                Input::I32(0),
                floats(&[self.gate_scale.len()], &self.gate_scale),
                floats(&[self.up_scale.len()], &self.up_scale),
                // No clamp.
                ints(&[0, 0], &[]),
            ],
        );
        let combined = interpret(
            module,
            "routed_combine",
            &[
                ("A", DType::F32),
                ("SGW", DType::F32),
                ("SUW", DType::F32),
                ("SDW", DType::F32),
            ],
            vec![
                floats(&[m, h], &self.residual),
                floats(&[m, 0], &[]),
                floats(&[blocks, tile, h], &Self::values(&result(&experts, 0))),
                ints(&[m, k], &inverse),
                floats(&[m, k], &scores),
                floats(&[m, h], &normalized),
                floats(&[m], &coefficient),
                floats(&[s, h], &self.shared_gate),
                floats(&[s, h], &self.shared_up),
                floats(&[h, s], &self.shared_down),
            ],
        );
        Self::values(&result(&combined, 0))
    }
}

/// With F32 activations the grouped prefill form performs the decode form's
/// exact arithmetic, so the two agree bit for bit: grouping, the expert
/// tiles and the unpermute are a permutation round trip. So do they under
/// per-expert gate, up and down scales, which move the output.
#[test]
fn portable_grouped_prefill_equals_decode_form() {
    let module = crate::common::kernels();
    let unscaled = Block::new(6).decode(&module);
    for block in [Block::new(6), Block::scaled(6)] {
        let decode = block.decode(&module);
        for tile in [1, 2, 4] {
            let prefill = block.prefill(&module, tile);
            assert_eq!(
                prefill.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                decode.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "tile {tile} scaled {}",
                !block.up_scale.is_empty()
            );
        }
        // The block is not the identity: some output moved off the residual.
        assert!(
            decode
                .iter()
                .zip(&block.residual)
                .any(|(out, residual)| (out - residual).abs() > 1e-3)
        );
        if !block.up_scale.is_empty() {
            assert_ne!(decode, unscaled, "the scales move the output");
        }
    }
}

/// `routed_route_shared` is `routed_route` followed by `routed_expand`'s
/// shared rows: with F32 activations every result and table agrees bit for
/// bit, with and without per-expert scales.
#[test]
fn portable_shared_route_equals_route_then_expand() {
    for block in [Block::new(6), Block::scaled(6)] {
        portable_shared_route_equals_route_then_expand_for(&block);
    }
}

fn portable_shared_route_equals_route_then_expand_for(block: &Block) {
    let module = crate::common::kernels();
    let (m, h, e, k, f, s) = (
        block.rows,
        block.hidden,
        block.experts,
        block.choices,
        block.features,
        block.shared,
    );
    let (normalized, coefficient, routes, scores) = block.route(&module);
    let expanded = interpret(
        &module,
        "routed_expand",
        &[
            ("A", DType::F32),
            ("EGW", DType::F32),
            ("EUW", DType::F32),
            ("SGW", DType::F32),
            ("SUW", DType::F32),
        ],
        vec![
            floats(&[m, h], &normalized),
            ints(&[m, k], &routes),
            floats(&[e, f, h], &block.expert_gate),
            floats(&[e, f, h], &block.expert_up),
            floats(&[s, h], &block.shared_gate),
            floats(&[s, h], &block.shared_up),
            floats(&[block.gate_scale.len()], &block.gate_scale),
            floats(&[block.up_scale.len()], &block.up_scale),
        ],
    );
    let shared = interpret(
        &module,
        "routed_route_shared",
        &[
            ("NW", DType::F32),
            ("RW", DType::F32),
            ("SRW", DType::F32),
            ("A", DType::F32),
            ("SGW", DType::F32),
            ("SUW", DType::F32),
        ],
        vec![
            floats(&[m, h], &block.residual),
            floats(&[h], &block.norm),
            floats(&[e, h], &block.router),
            floats(&[h], &block.shared_router),
            floats(&[s, h], &block.shared_gate),
            floats(&[s, h], &block.shared_up),
            ints(&[m, k], &vec![0; m * k]),
            floats(&[m, k], &vec![0.0; m * k]),
            Input::F32(1e-6),
            Input::I32(1),
            floats(&[block.down_scale.len()], &block.down_scale),
        ],
    );
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let tensor = |values: Vec<f64>| bits(&Block::values(&values));
    assert_eq!(tensor(result(&shared, 0)), bits(&normalized), "normalized");
    assert_eq!(
        tensor(result(&shared, 1)),
        bits(&coefficient),
        "coefficient"
    );
    assert_eq!(
        tensor(result(&shared, 2)),
        tensor(result(&expanded, 1)),
        "shared product"
    );
    assert_eq!(
        input(&shared, 6)
            .into_iter()
            .map(|v| v as i32)
            .collect::<Vec<_>>(),
        routes,
        "routes"
    );
    assert_eq!(tensor(input(&shared, 7)), bits(&scores), "scores");
}

// ---------------------------------------------------------------------------
// Host reference of the routed formulas (the portable bodies' arithmetic,
// with every published activation rounded to A by `round`).

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// The SiLU `glu` (functions.seismic) of the scaled F32 accumulators, each
/// publication rounded by `round`.
fn glu(gate: f32, up: f32, round: fn(f32) -> f32) -> f32 {
    round(round(silu(round(gate))) * round(up))
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff_0000)
}

/// Weights and inputs of one routed block at real packed representations.
struct Routed {
    hidden: usize,
    choices: usize,
    features: usize,
    shared: usize,
    expert_gate: Vec<f32>,
    expert_up: Vec<f32>,
    expert_down: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_down: Vec<f32>,
    /// Per-expert factors of the choices' gate and up accumulators (empty:
    /// the absent ports).
    gate_scale: Vec<f32>,
    up_scale: Vec<f32>,
    /// Whether the expert and shared gate/up and down weights take their
    /// rows as NVFP4 operands (`operands::operand_rows`).
    nvfp4: Nvfp4Weights,
}

#[derive(Clone, Copy, Default)]
struct Nvfp4Weights {
    expert_gate_up: bool,
    expert_down: bool,
    shared_gate_up: bool,
    shared_down: bool,
}

impl Nvfp4Weights {
    /// The recipes of weights of the representations (expert gate, up and
    /// down; shared gate, up and down) as the registry gives them.
    fn of(representations: [&str; 6]) -> Self {
        let nvfp4 = |name: &str| {
            registry::representation_info(registry::representation(name).unwrap()).operands
                == registry::OperandRecipe::Nvfp4Rows
        };
        let pair = |gate: &str, up: &str| {
            assert_eq!(nvfp4(gate), nvfp4(up), "a gate and its up share their operands");
            nvfp4(gate)
        };
        Self {
            expert_gate_up: pair(representations[0], representations[1]),
            expert_down: nvfp4(representations[2]),
            shared_gate_up: pair(representations[3], representations[4]),
            shared_down: nvfp4(representations[5]),
        }
    }
}

/// `accumulator` times expert `expert`'s factor in `scales` (unchanged when
/// the port is absent).
fn scaled(scales: &[f32], expert: usize, accumulator: f32) -> f32 {
    scales
        .get(expert)
        .map_or(accumulator, |scale| accumulator * scale)
}

/// Per-row inputs of the projections: normalized rows, routes, scores and
/// the shared coefficient.
struct Routing {
    rows: usize,
    residual: Vec<f32>,
    normalized: Vec<f32>,
    routes: Vec<i32>,
    scores: Vec<f32>,
    coefficient: Vec<f32>,
}

fn dot(x: &[f32], w: &[f32]) -> f32 {
    x.iter()
        .zip(w)
        .fold(0.0f32, |acc, (x, w)| x.mul_add(*w, acc))
}

impl Routed {
    /// The expansions' per-expert scale statics (`GS`, `US`).
    fn expansion_statics(&self) -> [(&'static str, u64); 2] {
        [
            ("GS", self.gate_scale.len() as u64),
            ("US", self.up_scale.len() as u64),
        ]
    }

    /// The expansions' per-expert gate and up scale ports.
    fn expansion_scales(&self, device: &seismic::Device) -> [seismic::Tensor; 2] {
        [
            scale_port(device, &self.gate_scale),
            scale_port(device, &self.up_scale),
        ]
    }

    /// (expert product [M, K, F], shared product [M, S]) with `round` applied
    /// to each published product.
    fn expand(&self, routing: &Routing, round: fn(f32) -> f32) -> (Vec<f32>, Vec<f32>) {
        let (h, k, f, s) = (self.hidden, self.choices, self.features, self.shared);
        let rows = parallel(routing.rows, |m| {
            let normalized = &routing.normalized[m * h..(m + 1) * h];
            let x = operands::operand_rows(normalized, h, self.nvfp4.expert_gate_up);
            let expert = (0..k)
                .flat_map(|choice| {
                    let e = routing.routes[m * k + choice] as usize;
                    (0..f).map(move |feature| (e, feature))
                })
                .map(|(e, feature)| {
                    let row = (e * f + feature) * h;
                    let gate = scaled(&self.gate_scale, e, dot(&x, &self.expert_gate[row..row + h]));
                    let up = scaled(&self.up_scale, e, dot(&x, &self.expert_up[row..row + h]));
                    glu(gate, up, round)
                })
                .collect::<Vec<_>>();
            let x = operands::operand_rows(normalized, h, self.nvfp4.shared_gate_up);
            let shared = (0..s)
                .map(|feature| {
                    let gate = dot(&x, &self.shared_gate[feature * h..(feature + 1) * h]);
                    let up = dot(&x, &self.shared_up[feature * h..(feature + 1) * h]);
                    glu(gate, up, round)
                })
                .collect::<Vec<_>>();
            (expert, shared)
        });
        let (expert, shared): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        (expert.concat(), shared.concat())
    }

    /// Each choice's product row and each shared product row as its down
    /// projection takes it.
    fn down_rows(&self, expert: &[f32], shared: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (
            operands::operand_rows(expert, self.features, self.nvfp4.expert_down),
            operands::operand_rows(shared, self.shared, self.nvfp4.shared_down),
        )
    }

    fn output(
        &self,
        routing: &Routing,
        expert: &[f32],
        shared: &[f32],
        round: fn(f32) -> f32,
    ) -> Vec<f32> {
        let (h, k, f, s) = (self.hidden, self.choices, self.features, self.shared);
        let (expert, shared) = &self.down_rows(expert, shared);
        parallel(routing.rows, |m| {
            (0..h)
                .map(|column| {
                    let mut selected = 0.0f32;
                    for choice in 0..k {
                        let e = routing.routes[m * k + choice] as usize;
                        let row = (e * h + column) * f;
                        let product = &expert[(m * k + choice) * f..(m * k + choice + 1) * f];
                        let published = round(dot(product, &self.expert_down[row..row + f]));
                        selected = routing.scores[m * k + choice].mul_add(published, selected);
                    }
                    let down = round(dot(
                        &shared[m * s..(m + 1) * s],
                        &self.shared_down[column * s..(column + 1) * s],
                    ));
                    routing.residual[m * h + column] + selected + down * routing.coefficient[m]
                })
                .collect::<Vec<_>>()
        })
        .concat()
    }

    /// The bounds of [`Routed::expand`]'s published products (expert,
    /// shared), each publication rounded to `output`, against a native that
    /// multiplies each weight in its form of `forms` over the same rows. Each
    /// gate and up sum is held to its form's bound, scaled in F32 and
    /// published; the SiLU product carries both errors through SiLU's slope,
    /// each side's evaluation, the F32 product and one more publication
    /// ([`bound::silu_product_between`]), `backend` the native's.
    fn expand_bounds(
        &self,
        routing: &Routing,
        forms: &RoutedForms,
        output: Float,
        backend: BackendName,
    ) -> (Vec<f64>, Vec<f64>) {
        let (h, k, f, s) = (self.hidden, self.choices, self.features, self.shared);
        let side = |form: Form, x: &[f32], row: &[f32], scale: Option<f32>| {
            let (acc, sums) = (dot(x, row), form.sums(&Products::of(x, row)));
            let (value, difference) = match scale {
                Some(scale) => (acc * scale, Float::F32.rounded(f64::from(scale.abs()) * sums, acc * scale)),
                None => (acc, sums),
            };
            let published = publish(output, value);
            (published, output.rounded(difference, published))
        };
        let (native, host) = (Library::of(backend), Library::of(BackendName::Cpu));
        let glu = |(g, gate): (f32, f64), (u, up): (f32, f64)| {
            let a = publish(output, silu(g));
            let value = publish(output, a * u);
            bound::silu_product_between(output, (&native, &host), (g, gate), (u, up), (a, value))
        };
        let rows = parallel(routing.rows, |m| {
            let normalized = &routing.normalized[m * h..(m + 1) * h];
            let x = operands::operand_rows(normalized, h, self.nvfp4.expert_gate_up);
            let expert = (0..k)
                .flat_map(|choice| {
                    let e = routing.routes[m * k + choice] as usize;
                    (0..f).map(move |feature| (e, feature))
                })
                .map(|(e, feature)| {
                    let row = (e * f + feature) * h;
                    glu(
                        side(forms.expert_gate, &x, &self.expert_gate[row..row + h], self.gate_scale.get(e).copied()),
                        side(forms.expert_up, &x, &self.expert_up[row..row + h], self.up_scale.get(e).copied()),
                    )
                })
                .collect::<Vec<_>>();
            let x = operands::operand_rows(normalized, h, self.nvfp4.shared_gate_up);
            let shared = (0..s)
                .map(|feature| {
                    let row = feature * h..(feature + 1) * h;
                    glu(
                        side(forms.shared_gate, &x, &self.shared_gate[row.clone()], None),
                        side(forms.shared_up, &x, &self.shared_up[row], None),
                    )
                })
                .collect::<Vec<_>>();
            (expert, shared)
        });
        let (expert, shared): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        (expert.concat(), shared.concat())
    }

    /// The bounds of [`Routed::output`], each down projection published in
    /// `output`, against a native that multiplies each weight in its form of
    /// `forms` over products within `expert_bound` and `shared_bound` of
    /// `expert` and `shared` (zero where it reads these products). Each
    /// published down projection is held to its form's bound plus its
    /// inputs' carried through the weight row; the combination
    /// `residual + Σ score·down + coefficient·shared` of K + 2 terms is an
    /// F32 sum on both sides, each within `γ_{K+2}` of the exact one.
    fn output_bounds(
        &self,
        routing: &Routing,
        (expert, shared): (&[f32], &[f32]),
        (expert_bound, shared_bound): (&[f64], &[f64]),
        forms: &RoutedForms,
        output: Float,
    ) -> Vec<f64> {
        let (h, k, f, s) = (self.hidden, self.choices, self.features, self.shared);
        // An NVFP4 down operand re-quantizes its products, which a
        // perturbation can move by a whole step: outside this model.
        assert!(
            !(self.nvfp4.expert_down && expert_bound.iter().any(|b| *b != 0.0)
                || self.nvfp4.shared_down && shared_bound.iter().any(|b| *b != 0.0)),
            "a carried product bound through an NVFP4 down operand"
        );
        let (expert, shared) = &self.down_rows(expert, shared);
        let gamma = bound::gamma(k + 2);
        let published = |form: Form, x: &[f32], row: &[f32], carried: &[f64]| {
            let value = publish(output, dot(x, row));
            let carried = row.iter().zip(carried).map(|(w, b)| f64::from(w.abs()) * b).sum::<f64>();
            (value, output.rounded(form.sums(&Products::of(x, row)) + carried, value))
        };
        parallel(routing.rows, |m| {
            (0..h)
                .map(|column| {
                    let mut magnitude = f64::from(routing.residual[m * h + column].abs());
                    let mut moved = 0.0;
                    for choice in 0..k {
                        let e = routing.routes[m * k + choice] as usize;
                        let row = (e * h + column) * f;
                        let at = (m * k + choice) * f..(m * k + choice + 1) * f;
                        let (value, difference) = published(
                            forms.expert_down,
                            &expert[at.clone()],
                            &self.expert_down[row..row + f],
                            &expert_bound[at],
                        );
                        let score = f64::from(routing.scores[m * k + choice].abs());
                        magnitude += score * f64::from(value.abs());
                        moved += score * difference;
                    }
                    let (value, difference) = published(
                        forms.shared_down,
                        &shared[m * s..(m + 1) * s],
                        &self.shared_down[column * s..(column + 1) * s],
                        &shared_bound[m * s..(m + 1) * s],
                    );
                    let coefficient = f64::from(routing.coefficient[m].abs());
                    magnitude += coefficient * f64::from(value.abs());
                    moved += coefficient * difference;
                    (1.0 + gamma) * moved + 2.0 * gamma * magnitude
                })
                .collect::<Vec<_>>()
        })
        .concat()
    }
}

/// `value` published in `output`: rounded to BF16, or held in F32.
fn publish(output: Float, value: f32) -> f32 {
    match output {
        Float::Bf16 => bf16_round(value),
        Float::F32 => value,
        Float::F16 => panic!("the routed host model publishes BF16 or F32"),
    }
}

/// The forms a device's routed natives multiply each weight in.
#[derive(Clone, Copy)]
struct RoutedForms {
    expert_gate: Form,
    expert_up: Form,
    expert_down: Form,
    shared_gate: Form,
    shared_up: Form,
    shared_down: Form,
}

impl RoutedForms {
    /// Every weight in `form`.
    fn uniform(form: Form) -> Self {
        Self {
            expert_gate: form,
            expert_up: form,
            expert_down: form,
            shared_gate: form,
            shared_up: form,
            shared_down: form,
        }
    }

    /// Each of the weights (expert gate, up, down; shared gate, up, down)
    /// in `form` of its resident element.
    fn of(weights: [&seismic::Tensor; 6], form: impl Fn(seismic::Element) -> Form) -> Self {
        let [expert_gate, expert_up, expert_down, shared_gate, shared_up, shared_down] =
            weights.map(|weight| form(weight.element()));
        Self { expert_gate, expert_up, expert_down, shared_gate, shared_up, shared_down }
    }

    /// The decode entries (`routed_expand`, `routed_output`) on `device` at
    /// `rows`: the projection family's GEMVs (`Form::projection`).
    fn decode(device: &seismic::Device, weights: [&seismic::Tensor; 6], rows: usize) -> Self {
        Self::of(weights, |weight| Form::projection(device.backend(), rows, Float::Bf16, weight))
    }

    /// The grouped entries (`routed_experts`, `routed_combine`) on `device`:
    /// the family's GEMM at every block (`Form::matrix`), or CUDA's
    /// F16-operand GEMM at `OPERAND` 1.
    fn grouped(device: &seismic::Device, weights: [&seismic::Tensor; 6], operand: u64) -> Self {
        Self::of(weights, |weight| match (device.backend(), operand) {
            (seismic::BackendName::Cuda, 1) => Form::cuda_f16_operands(Float::Bf16, weight),
            (backend, _) => Form::matrix(backend, Float::Bf16, weight),
        })
    }
}

/// Route results (scores after the softmax, the shared expert's sigmoid
/// coefficient) agree within `relative * |reference| + absolute`, with
/// `absolute` a fraction of the reference's largest magnitude.
fn assert_near(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    relative: f32,
    absolute_fraction: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    let scale = expected
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= relative * expected.abs() + absolute_fraction * scale,
            "{name}[{index}]: native {actual}, reference {expected} (scale {scale})"
        );
    }
}

/// `actual` within `bound` of the host model `expected`; the INT8 form,
/// which quantizes its activations (a tuned alternative, not the same
/// arithmetic), to the tuner's relative output-norm defect guard instead.
fn assert_arithmetic(name: &str, actual: &[f32], expected: &[f32], int8: bool, bound: &[f64]) {
    if !int8 {
        return bound::assert_within(name, actual, expected, bound);
    }
    assert_eq!(actual.len(), expected.len(), "{name} length");
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f32>();
    let scale = expected.iter().map(|value| value.powi(2)).sum::<f32>();
    assert!(
        error <= 0.05f32.powi(2) * scale + 1e-6,
        "{name}: relative error {}",
        (error / scale.max(1e-6)).sqrt()
    );
}

/// The reference agrees with the portable bodies: decode chain on the small
/// f32 block (no rounding at f32).
#[test]
fn host_reference_matches_portable_decode_body() {
    let module = crate::common::kernels();
    let block = Block::new(5);
    let (normalized, coefficient, routes, scores) = block.route(&module);
    let routed = Routed {
        hidden: block.hidden,
        choices: block.choices,
        features: block.features,
        shared: block.shared,
        expert_gate: block.expert_gate.clone(),
        expert_up: block.expert_up.clone(),
        expert_down: block.expert_down.clone(),
        shared_gate: block.shared_gate.clone(),
        shared_up: block.shared_up.clone(),
        shared_down: block.shared_down.clone(),
        gate_scale: Vec::new(),
        up_scale: Vec::new(),
        nvfp4: Nvfp4Weights::default(),
    };
    let routing = Routing {
        rows: block.rows,
        residual: block.residual.clone(),
        normalized,
        routes,
        scores,
        coefficient,
    };
    let (expert, shared) = routed.expand(&routing, |v| v);
    let reference = routed.output(&routing, &expert, &shared, |v| v);
    // Both sum in F32 and publish in F32, in their own orders.
    let forms = RoutedForms::uniform(Form::F32);
    let (expert_bound, shared_bound) = routed.expand_bounds(&routing, &forms, Float::F32, BackendName::Cpu);
    let bounds =
        routed.output_bounds(&routing, (&expert, &shared), (&expert_bound, &shared_bound), &forms, Float::F32);
    bound::assert_within("decode", &block.decode(&module), &reference, &bounds);
}

// ---------------------------------------------------------------------------
// Packed weights in the rows16 layout.

/// The resident row layout of the device's backend.
fn resident_layout(device: &seismic::Device) -> registry::Layout {
    device.backend().resident_layout()
}

/// A weight of `representation` in the device's resident row layout with
/// pseudo-random planes, and its decoded values (the registry's reference
/// recipe).
fn packed_weight(
    device: &seismic::Device,
    representation: &str,
    shape: &[u64],
    seed: u32,
) -> (seismic::Tensor, Vec<f32>) {
    let (element, bytes, values) = packed_planes(device, representation, shape, seed);
    (
        seismic::Tensor::from_host(&seismic::Charges::tally(device), element, shape, &bytes).unwrap(),
        values,
    )
}

fn packed_router_matches_dense_decoding(representation: &str, backend: seismic::BackendName) {
    let Some(device) = seismic_native_tests::devices(&[backend]).into_iter().next() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    const HIDDEN: usize = 256;
    let (packed, decoded) = packed_weight(&device, representation, &[E as u64, HIDDEN as u64], 87);
    let dense = f32_tensor(&device, &[E as u64, HIDDEN as u64], &decoded);
    let norm_values = vec![1.0f32; HIDDEN];
    let norm = f32_tensor(&device, &[HIDDEN as u64], &norm_values);
    let shared = f32_tensor(&device, &[HIDDEN as u64], &vec![0.01; HIDDEN]);
    for rows in [1usize, 9] {
        let residual_values = pattern(rows * HIDDEN, 91 + rows as u32, 0.5);
        let residual = f32_tensor(&device, &[rows as u64, HIDDEN as u64], &residual_values);
        let run = |router: &seismic::Tensor| {
            let kernel = declared::prepared(
                &device,
                &routed_route::native_entry_with(routed_route::Elements {
                    NW: seismic::Element::f32(),
                    RW: router.element(),
                    SRW: seismic::Element::f32(),
                    A: if is_cpu(&device) { seismic::Element::f32() } else { seismic::Element::bf16() },
                }),
                &specialization_on(
                    &device,
                    &route_statics(
                        &[("H", HIDDEN as u64), ("E", E as u64), ("K", K as u64)],
                        HIDDEN as u64,
                        0,
                    ),
                    &route_mapping(&device, 4, route_splits(&device)[0]),
                ),
            )
            .unwrap();
            let mut routes = i32_tensor(&device, &[rows as u64, K as u64], &vec![-7; rows * K]);
            let mut scores = f32_tensor(&device, &[rows as u64, K as u64], &vec![-7.0; rows * K]);
            let outcome = kernel
                .call(&charges, routed_route::Args {
                    residual: &residual,
                    norm: &norm,
                    router,
                    shared_router: &shared,
                    routes: &mut routes,
                    scores: &mut scores,
                    eps: 1e-6,
                    normalize: 1,
                    expert_scale: &absent(&device),
                })
                .unwrap();
            (read_i32(&routes), read_f32(&scores), read_f32(&outcome.r1))
        };
        let expected = run(&dense);
        let actual = run(&packed);
        assert_eq!(actual.0, expected.0, "{representation} rows {rows} routes");
        assert_near(
            &format!("{representation} rows {rows} scores"),
            &actual.1,
            &expected.1,
            1e-4,
            1e-4,
        );
        assert_near(
            &format!("{representation} rows {rows} gate"),
            &actual.2,
            &expected.2,
            1e-4,
            1e-4,
        );
    }
}

#[test]
fn metal_q8_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q8g32s", seismic::BackendName::Metal);
}

#[test]
fn metal_q6_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q6k", seismic::BackendName::Metal);
}

#[test]
fn metal_q4_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q4k", seismic::BackendName::Metal);
}

#[test]
fn metal_q5_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q5k", seismic::BackendName::Metal);
}

#[test]
fn metal_iq4_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("iq4xs", seismic::BackendName::Metal);
}

#[test]
fn cuda_q8_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q8g32s", seismic::BackendName::Cuda);
}

#[test]
fn cuda_q6_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q6k", seismic::BackendName::Cuda);
}

#[test]
fn cuda_q4_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q4k", seismic::BackendName::Cuda);
}

#[test]
fn cuda_q5_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q5k", seismic::BackendName::Cuda);
}

#[test]
fn cuda_iq4_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("iq4xs", seismic::BackendName::Cuda);
}

#[test]
fn vulkan_q8_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q8g32s", seismic::BackendName::Vulkan);
}

#[test]
fn vulkan_q6_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q6k", seismic::BackendName::Vulkan);
}

#[test]
fn vulkan_q4_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q4k", seismic::BackendName::Vulkan);
}

#[test]
fn vulkan_q5_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("q5k", seismic::BackendName::Vulkan);
}

#[test]
fn vulkan_iq4_router_matches_dense_decoding() {
    packed_router_matches_dense_decoding("iq4xs", seismic::BackendName::Vulkan);
}

/// The planes of one packed weight: resident element, layout bytes and
/// decoded values.
type Planes = (seismic::Element, Vec<u8>, Vec<f32>);

/// The resident element, layout bytes and decoded values of a
/// `representation` weight of `shape` with pseudo-random planes. Decoding
/// through the reference recipe is slow in a test build, so every device and
/// test shares one decoding of the packets, and every device and test of one
/// resident layout one placement of them.
fn packed_planes(
    device: &seismic::Device,
    representation: &str,
    shape: &[u64],
    seed: u32,
) -> Planes {
    type Key = (String, Vec<u64>, u32);
    static DECODED: std::sync::Mutex<Vec<(Key, Packets)>> = std::sync::Mutex::new(Vec::new());
    let key = (representation.to_owned(), shape.to_vec(), seed);
    let layout = resident_layout(device);
    let mut decoded = DECODED.lock().unwrap();
    let index = match decoded.iter().position(|(known, _)| *known == key) {
        Some(index) => index,
        None => {
            decoded.push((key, Packets::decoded(representation, shape, seed)));
            decoded.len() - 1
        }
    };
    let packets = &mut decoded[index].1;
    let rows_id = registry::storage(representation, layout).unwrap();
    let registry::RepresentationKind::PackedRows(rows) = &registry::representation_info(rows_id).kind else {
        panic!("{representation} resident storage is a row layout")
    };
    let placed = match packets.placed.iter().find(|(known, _)| *known == layout) {
        Some((_, placed)) => placed.clone(),
        None => {
            let placed = rows.place(shape, &packets.bytes);
            packets.placed.push((layout, placed.clone()));
            placed
        }
    };
    let element = seismic::Element::named(registry::representation_info(rows_id).name).unwrap();
    (element, placed, packets.values.clone())
}

/// A packed weight's pseudo-random packets, their decoded values, and their
/// placements in the resident layouts asked for so far.
struct Packets {
    bytes: Vec<u8>,
    values: Vec<f32>,
    placed: Vec<(registry::Layout, Vec<u8>)>,
}

impl Packets {
    /// The packets of a `representation` weight of `shape` from `seed`,
    /// decoded in the packet layout, whose values every row layout's
    /// placement keeps: each packet row decodes on its own, across the
    /// host's cores.
    fn decoded(representation: &str, shape: &[u64], seed: u32) -> Self {
        let packet_id = registry::storage(representation, registry::Layout::Packet).unwrap();
        let registry::RepresentationKind::Packed(packet) = &registry::representation_info(packet_id).kind else {
            panic!("{representation} is packed")
        };
        let bytes = random_packets(packet, shape, seed);
        let (&last, leading) = shape.split_last().unwrap();
        let row_bytes = bytes.len() / leading.iter().product::<u64>() as usize;
        let values = parallel(bytes.len() / row_bytes, |row| {
            TensorData::encoded(packet_id, vec![1, last as usize], bytes[row * row_bytes..(row + 1) * row_bytes].to_vec())
                .unwrap()
                .values()
                .unwrap()
        })
        .into_iter()
        .flatten()
        .map(|value| value as f32)
        .collect();
        Self { bytes, values, placed: Vec::new() }
    }
}

/// Pseudo-random packets of a weight of `shape` in `packet`'s form, with
/// GGUF-like scale magnitudes.
fn random_packets(packet: &registry::PackedPacketLayout, shape: &[u64], seed: u32) -> Vec<u8> {
    let (last, leading) = shape.split_last().unwrap();
    let packets = leading.iter().product::<u64>() * last.div_ceil(u64::from(packet.group));
    let size = packet.packet_size as usize;
    let mut bytes = vec![0u8; packets as usize * size];
    let noise = pattern(bytes.len(), seed, 1.0);
    for (index, packet_bytes) in bytes.chunks_exact_mut(size).enumerate() {
        for plane in &packet.planes {
            let start = plane.offset as usize;
            let end = start + plane.bytes_per_group as usize;
            if plane.storage_dtype == DType::F16 {
                for (field, pair) in packet_bytes[start..end].chunks_exact_mut(2).enumerate() {
                    let unit = noise[(index * size + start + 2 * field) % noise.len()].abs();
                    // GGUF-like magnitudes: decoded weights of order 0.01-0.3.
                    let factor = (1.0 + 3.0 * unit) / 4096.0;
                    pair.copy_from_slice(&registry::f16_bits(factor).to_le_bytes());
                }
            } else if plane.storage_dtype == DType::F32 {
                for (field, word) in packet_bytes[start..end].chunks_exact_mut(4).enumerate() {
                    let unit = noise[(index * size + start + 4 * field) % noise.len()].abs();
                    word.copy_from_slice(&((1.0 + 3.0 * unit) / 4096.0).to_le_bytes());
                }
            } else if plane.name == "block_scale" {
                // Floating scale codes of GGUF-like magnitudes: E8M0 2^-10 ..
                // 2^-7, UE4M3 2^-6 .. 2^-4 with every mantissa.
                for (offset, byte) in packet_bytes[start..end].iter_mut().enumerate() {
                    let unit = noise[(index * size + start + offset) % noise.len()].abs();
                    let code = (unit * 32.0) as u8 % 32;
                    *byte = match plane.encoding {
                        registry::PlaneEncoding::FloatCode {
                            format: registry::FloatCodeFormat::E8M0,
                        } => 117 + code % 4,
                        _ => (1 + code / 8) << 3 | code % 8,
                    };
                }
            } else {
                for (offset, byte) in packet_bytes[start..end].iter_mut().enumerate() {
                    let unit = noise[(index * size + start + offset) % noise.len()];
                    *byte = ((unit + 1.0) * 127.5) as u8;
                }
            }
        }
    }
    bytes
}

fn bf16_tensor(device: &seismic::Device, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(&seismic::Charges::tally(device), seismic::Element::bf16(), shape, &bytes).unwrap()
}

/// A dense bf16 weight of `shape` with pseudo-random values of GGUF-like
/// magnitude, and its values.
fn bf16_weight(device: &seismic::Device, shape: &[u64], seed: u32) -> (seismic::Tensor, Vec<f32>) {
    let count = shape.iter().product::<u64>() as usize;
    let values = pattern(count, seed, 0.1)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    (bf16_tensor(device, shape, &values), values)
}

/// The expert weights of a `PackedBlock`.
#[derive(Clone, Copy, Debug)]
enum Experts {
    /// The 35B-A3B mix: gate/up q4k, down q5k.
    Packed,
    /// IQ4_XS weights, resident as iq4xs.
    Iq4,
    /// IQ4_NL weights, resident as iq4g32s, at a width not a multiple of 256
    /// (a 640-wide expert down).
    Iq4Blocks,
    /// Dense bf16 (GGUF F32/F16/BF16 experts are resident bf16).
    Bf16,
    /// Gemma / Nemotron formats: gate q4g32s (Q4_0), up mxfp4g32, down q5g32
    /// (Q5_1).
    Coded,
    /// Gate and up nvfp4g16, down q5g32s (Q5_0).
    Fp4,
    /// Fp4 calibrated for FP4 activations: gate and up nvfp4g16a4.
    Fp4A4,
    /// Vector codes: gate and down iq2xs (IQ2_XS), up iq3xxs (IQ3_XXS).
    Grid,
    /// Nemotron 3.5 Bolt's vector codes: every expert and shared weight
    /// iq2xxs (IQ2_XXS).
    Iq2Xxs,
    /// A Q2_K quant's experts: gate and up q2k, whose two runs a block each
    /// fold a min term, down q3k.
    Q2,
}

/// The 35B-A3B storage mix at reduced width: expert weights per `Experts`,
/// shared q8.
struct PackedBlock {
    routed: Routed,
    expert_gate: seismic::Tensor,
    expert_up: seismic::Tensor,
    expert_down: seismic::Tensor,
    shared_gate: seismic::Tensor,
    shared_up: seismic::Tensor,
    shared_down: seismic::Tensor,
}

impl PackedBlock {
    const HIDDEN: usize = 512;

    /// The resident weights, as [`RoutedForms::of`] takes them.
    fn weights(&self) -> [&seismic::Tensor; 6] {
        [&self.expert_gate, &self.expert_up, &self.expert_down, &self.shared_gate, &self.shared_up, &self.shared_down]
    }
    const EXPERTS: usize = 16;
    const CHOICES: usize = 4;
    const SHARED: usize = 256;

    /// Expert features: Qwen3.8 Flash Next Q2's 640 (not a multiple of 256)
    /// for IQ4_NL single-block experts, else 256.
    fn features(experts: Experts) -> usize {
        match experts {
            Experts::Iq4Blocks => 640,
            _ => 256,
        }
    }

    fn new(device: &seismic::Device, experts: Experts) -> Self {
        let features = Self::features(experts);
        let (h, e, f, s) = (
            Self::HIDDEN as u64,
            Self::EXPERTS as u64,
            features as u64,
            Self::SHARED as u64,
        );
        // The representations of gate, up and down.
        let representations = match experts {
            Experts::Packed => ["q4k", "q4k", "q5k"],
            Experts::Iq4 => ["iq4xs"; 3],
            Experts::Iq4Blocks => ["iq4g32s"; 3],
            Experts::Bf16 => ["bf16"; 3],
            Experts::Coded => ["q4g32s", "mxfp4g32", "q5g32"],
            Experts::Fp4 => ["nvfp4g16", "nvfp4g16", "q5g32s"],
            Experts::Fp4A4 => ["nvfp4g16a4", "nvfp4g16a4", "q5g32s"],
            Experts::Grid => ["iq2xs", "iq3xxs", "iq2xs"],
            Experts::Iq2Xxs => ["iq2xxs"; 3],
            Experts::Q2 => ["q2k", "q2k", "q3k"],
        };
        let expert_weight = |which: usize, shape: &[u64], seed: u32| match experts {
            Experts::Bf16 => bf16_weight(device, shape, seed),
            _ => packed_weight(device, representations[which], shape, seed),
        };
        let (expert_gate, gate) = expert_weight(0, &[e, f, h], 41);
        let (expert_up, up) = expert_weight(1, &[e, f, h], 42);
        let (expert_down, down) = expert_weight(2, &[e, h, f], 43);
        let shared = match experts {
            Experts::Iq2Xxs => "iq2xxs",
            _ => "q8g32s",
        };
        let (shared_gate, sgate) = packed_weight(device, shared, &[s, h], 44);
        let (shared_up, sup) = packed_weight(device, shared, &[s, h], 45);
        let (shared_down, sdown) = packed_weight(device, shared, &[h, s], 46);
        Self {
            routed: Routed {
                hidden: Self::HIDDEN,
                choices: Self::CHOICES,
                features,
                shared: Self::SHARED,
                expert_gate: gate,
                expert_up: up,
                expert_down: down,
                shared_gate: sgate,
                shared_up: sup,
                shared_down: sdown,
                gate_scale: Vec::new(),
                up_scale: Vec::new(),
                nvfp4: Nvfp4Weights::of([
                    representations[0],
                    representations[1],
                    representations[2],
                    shared,
                    shared,
                    shared,
                ]),
            },
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            shared_up,
            shared_down,
        }
    }

    /// The same block with per-expert gate and up global scales, distinct
    /// per expert and between gate and up.
    fn scaled(mut self) -> Self {
        self.routed.gate_scale = expert_scales(Self::EXPERTS);
        self.routed.up_scale = self.routed.gate_scale.iter().rev().copied().collect();
        self
    }

    /// Synthetic routing of `rows` rows: distinct experts per row, skewed so
    /// some experts fill several tiles.
    fn routing(rows: usize, seed: u32) -> Routing {
        let (h, e, k) = (Self::HIDDEN, Self::EXPERTS, Self::CHOICES);
        let noise = pattern(rows * k * 3, seed, 1.0);
        let mut routes = Vec::with_capacity(rows * k);
        for row in 0..rows {
            let mut chosen = Vec::new();
            let mut index = row * k * 3;
            while chosen.len() < k {
                let pick = ((noise[index % noise.len()] + 1.0) * 0.5 * e as f32) as i32;
                let pick = if row % 2 == 0 { pick % 6 } else { pick }.min(e as i32 - 1);
                if !chosen.contains(&pick) {
                    chosen.push(pick);
                }
                index += 1;
                if index > row * k * 3 + 64 {
                    let next = (0..e as i32)
                        .find(|candidate| !chosen.contains(candidate))
                        .unwrap();
                    chosen.push(next);
                }
            }
            routes.extend(chosen);
        }
        let scores = pattern(rows * k, seed + 1, 1.0)
            .into_iter()
            .map(|v| 0.1 + 0.2 * (v + 1.0))
            .collect();
        Routing {
            rows,
            residual: pattern(rows * h, seed + 2, 2.0),
            normalized: pattern(rows * h, seed + 3, 2.0)
                .into_iter()
                .map(bf16_round)
                .collect(),
            routes,
            scores,
            coefficient: pattern(rows, seed + 4, 1.0)
                .into_iter()
                .map(|v| 0.5 + 0.4 * v)
                .collect(),
        }
    }
}

fn element(device: &seismic::Device, representation: &str) -> seismic::Element {
    element_in(representation, resident_layout(device))
}

/// The element of `representation`'s storage in `layout`.
fn element_in(representation: &str, layout: registry::Layout) -> seismic::Element {
    let id = registry::storage(representation, layout).unwrap();
    seismic::Element::named(registry::representation_info(id).name).unwrap()
}

fn read_bf16(tensor: &seismic::Tensor) -> Vec<f32> {
    read_activation(tensor, DType::BF16)
}

/// Mapping parameters of the decode entries on the device's backend.
fn decode_mappings(device: &seismic::Device) -> Vec<Vec<(&'static str, u64)>> {
    match device.backend() {
        seismic::BackendName::Cuda => vec![
            vec![("TPW", 1), ("KSPLIT", 1)],
            vec![("TPW", 2), ("KSPLIT", 1)],
            vec![("TPW", 1), ("KSPLIT", 2)],
            vec![("TPW", 2), ("KSPLIT", 4)],
        ],
        seismic::BackendName::Vulkan => vec![
            vec![("SIMDGROUPS", 2), ("ROWS", 1)],
            vec![("SIMDGROUPS", 4), ("ROWS", 2)],
            vec![("SIMDGROUPS", 8), ("ROWS", 4)],
        ],
        seismic::BackendName::Cpu => vec![vec![("ROWS", 8)], vec![("ROWS", 4)], vec![("ROWS", 1)]],
        seismic::BackendName::Metal => vec![
            vec![("SIMDGROUPS", 2), ("ROWS", 1), ("LANES", 32)],
            vec![("SIMDGROUPS", 4), ("ROWS", 2), ("LANES", 32)],
            vec![("SIMDGROUPS", 4), ("ROWS", 4), ("LANES", 16)],
            vec![("SIMDGROUPS", 8), ("ROWS", 4), ("LANES", 16)],
            vec![("SIMDGROUPS", 16), ("ROWS", 1), ("LANES", 16)],
        ],
    }
}

/// The decode output's mapping beside decode mapping `index`: the Metal
/// output tiles its channels itself (ROWS channels per lane group of LANES
/// lanes); the Vulkan output takes the expansion's mapping with its own
/// LANES; elsewhere the output takes the expansion's mapping.
fn output_mapping(
    device: &seismic::Device,
    index: usize,
    mapping: &[(&'static str, u64)],
) -> Vec<(&'static str, u64)> {
    const METAL: [[(&str, u64); 2]; 5] = [
        [("ROWS", 2), ("LANES", 16)],
        [("ROWS", 4), ("LANES", 32)],
        [("ROWS", 1), ("LANES", 4)],
        [("ROWS", 2), ("LANES", 8)],
        [("ROWS", 8), ("LANES", 32)],
    ];
    match device.backend() {
        seismic::BackendName::Metal => METAL[index % METAL.len()].to_vec(),
        seismic::BackendName::Vulkan => {
            const LANES: [u64; 3] = [32, 16, 8];
            [mapping, &[("LANES", LANES[index % LANES.len()])]].concat()
        }
        _ => mapping.to_vec(),
    }
}

/// `statics` beside those of the single residual stream that `declared`
/// (an entry's statics) holds and `statics` does not name: the streams'
/// join (`SN` 1, `IN` 0), the input's one normalized part (`PT` 1, `NM` the
/// hidden width `H`), and absent per-expert scales and limit (`ES`, `GS`,
/// `US`, `LB` 0).
fn single_stream_statics<'a>(declared: &[String], statics: &[(&'a str, u64)]) -> Vec<(&'a str, u64)> {
    let hidden = statics.iter().find(|(name, _)| *name == "H").map_or(0, |(_, value)| *value);
    let defaults = [("SN", 1), ("IN", 0), ("PT", 1), ("NM", hidden), ("ES", 0), ("GS", 0), ("US", 0), ("LB", 0)];
    statics
        .iter()
        .copied()
        .chain(defaults.into_iter().filter(|(name, _)| {
            declared.iter().any(|declared| declared == name) && !statics.iter().any(|(given, _)| given == name)
        }))
        .collect()
}

/// `mapping` where entry `E` declares each name on `device`'s backend: as an
/// entry parameter, or in every launch that scopes it. The statics are E's
/// over the single residual stream: with those of the streams' join (`SN` 1,
/// `IN` 0) and input form (`PT` 1, `NM` the hidden width) that E declares,
/// and the per-choice form of an entry with a gathered one (`GATHERED` 0)
/// where `mapping` does not name it; every launch parameter `mapping` does
/// not name at its declared default.
fn decode_specialization<E: seismic::Entry>(
    device: &seismic::Device,
    statics: &[(&str, u64)],
    mapping: &[(&'static str, u64)],
) -> seismic::NativeSpecialization {
    let implementation = seismic::generated::native_implementation::<E>(device)
        .unwrap()
        .unwrap();
    let statics = single_stream_statics(&implementation.statics, statics);
    // The per-choice decode expansion, for an entry with a gathered form
    // the mapping does not name (`GATHERED`).
    let forms = [("GATHERED", 0)].into_iter().filter(|(name, _)| {
        implementation.params.iter().any(|parameter| parameter.name == *name)
            && !mapping.iter().any(|(given, _)| given == name)
    });
    let mapped = mapping.iter().copied().chain(forms).collect::<Vec<_>>().iter().fold(
        specialization_on(device, &statics, &[]),
        |choice, (name, value)| {
            if implementation
                .params
                .iter()
                .any(|parameter| parameter.name == *name)
            {
                return choice.with_param(*name, *value);
            }
            let launches = implementation
                .launches
                .iter()
                .enumerate()
                .filter(|(_, launch)| {
                    launch
                        .params
                        .iter()
                        .any(|parameter| parameter.name == *name)
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            assert!(
                !launches.is_empty(),
                "`{name}` is not a parameter of the entry"
            );
            launches.into_iter().fold(choice, |choice, launch| {
                choice.with_launch_param(launch, *name, *value)
            })
        },
    );
    // Each launch parameter `mapping` leaves unvalued at its declared
    // default: those of Metal's two-row and several-row launches, which
    // their row counts tune apart from the one-row launch `mapping` names.
    let dimension = |name: &str| statics.iter().find(|(given, _)| *given == name).map(|(_, value)| *value);
    let mut defaults = implementation.parameter_defaults(&dimension).unwrap().into_iter().skip(implementation.params.len());
    let mut specialization = mapped;
    for (launch, declaration) in implementation.launches.iter().enumerate() {
        for parameter in &declaration.params {
            let default = defaults.next().expect("a default for every launch parameter");
            if specialization.launch_param(launch, &parameter.name).is_none() {
                specialization = specialization.with_launch_param(launch, parameter.name.as_str(), default);
            }
        }
    }
    specialization
}

/// `mapping` of decode entry `E` at mapping `index`: where its native
/// declares a gathered form (`GATHERED`), the form alternates, per choice
/// then gathered, so a test's mappings cover both; a native of one form
/// takes `mapping` as it is.
fn gathered<E: seismic::Entry>(
    device: &seismic::Device,
    index: usize,
    mapping: &[(&'static str, u64)],
) -> Vec<(&'static str, u64)> {
    let implementation = seismic::generated::native_implementation::<E>(device)
        .unwrap()
        .unwrap();
    let form = implementation
        .params
        .iter()
        .any(|parameter| parameter.name == "GATHERED")
        .then_some(("GATHERED", (index % 2) as u64));
    mapping.iter().copied().chain(form).collect()
}

/// Mapping parameters of the grouped entries on the device's backend.
fn grouped_mappings(device: &seismic::Device) -> Vec<Vec<(&'static str, u64)>> {
    match device.backend() {
        seismic::BackendName::Cuda => vec![vec![]],
        seismic::BackendName::Cpu => vec![vec![("ROWS", 8)], vec![("ROWS", 2)]],
        // Vulkan also maps the tile onto subgroups of SUB_M x SUB_N.
        seismic::BackendName::Vulkan => vec![
            vec![
                ("TILE_M", 32),
                ("TILE_N", 128),
                ("SUB_M", 32),
                ("SUB_N", 32),
            ],
            vec![("TILE_M", 64), ("TILE_N", 128), ("SUB_M", 64), ("SUB_N", 32)],
            vec![("TILE_M", 32), ("TILE_N", 64), ("SUB_M", 32), ("SUB_N", 64)],
        ],
        _ => vec![
            vec![("TILE_M", 32), ("TILE_N", 128)],
            vec![("TILE_M", 64), ("TILE_N", 64)],
            vec![("TILE_M", 32), ("TILE_N", 64)],
        ],
    }
}

fn specialize(
    statics: &[(&str, u64)],
    params: &[(&'static str, u64)],
) -> seismic::NativeSpecialization {
    let specialization = statics.iter().fold(
        seismic::NativeSpecialization::new(),
        |spec, (name, value)| spec.with_static(*name, *value),
    );
    params.iter().fold(specialization, |spec, (name, value)| {
        spec.with_param(*name, *value)
    })
}

/// The statics of a routed route over the single residual stream: `statics`
/// with its input read as one part (`PT` 1) normalized at the hidden width
/// `h` (`NM`), its scores scaled by `expert_scales` per-expert factors
/// (`ES`: 0 absent, else the expert count).
fn route_statics(
    statics: &[(&'static str, u64)],
    h: u64,
    expert_scales: u64,
) -> Vec<(&'static str, u64)> {
    [statics, &[("PT", 1), ("NM", h), ("ES", expert_scales)]].concat()
}

/// The statics of a routed output over the single residual stream: `statics`
/// with one stream (`SN` 1) joined without inject (`IN` 0).
fn joined_statics(statics: &[(&'static str, u64)]) -> Vec<(&'static str, u64)> {
    [statics, &[("SN", 1), ("IN", 0)]].concat()
}

#[test]
fn native_decode_expand_and_output_match_reference() {
    for device in seismic_native_tests::every_device() {
        native_decode_expand_and_output_match_reference_on(&device, false);
        if is_cpu(&device) {
            native_decode_expand_and_output_match_reference_on(&device, true);
        }
    }
}

/// Dense bf16 experts: each expert is one matrix of the [E, N, K] tensor at
/// its own row stride (no tile padding).
#[test]
fn native_decode_expand_and_output_match_reference_with_bf16_experts() {
    for device in seismic_native_tests::every_device() {
        native_decode_expand_and_output_match_reference_rows(
            &device,
            PackedBlock::new(&device, Experts::Bf16),
            false,
            &[1, 3, 8],
            decode_mappings(&device),
        );
    }
}

/// Routed and shared experts all IQ2_XXS (Nemotron 3.5 Bolt).
#[test]
fn native_decode_expand_and_output_match_reference_with_iq2_xxs_experts() {
    for device in seismic_native_tests::every_device() {
        native_decode_expand_and_output_match_reference_rows(
            &device,
            PackedBlock::new(&device, Experts::Iq2Xxs),
            false,
            &[1, 3, 8],
            decode_mappings(&device),
        );
    }
}

/// Experts in the representations of Q4_0, MXFP4, Q5_1, NVFP4, Q5_0, IQ2_XS
/// and IQ3_XXS.
#[test]
fn native_decode_expand_and_output_match_reference_with_coded_experts() {
    for device in seismic_native_tests::every_device() {
        for experts in [Experts::Coded, Experts::Fp4, Experts::Fp4A4, Experts::Grid] {
            native_decode_expand_and_output_match_reference_rows(
                &device,
                PackedBlock::new(&device, experts),
                false,
                &[1, 3, 8],
                decode_mappings(&device),
            );
        }
    }
}

#[test]
fn vulkan_decode_expand_and_output_match_reference_with_iq4_experts() {
    for device in seismic_native_tests::every_device()
        .into_iter()
        .filter(|device| device.backend() == seismic::BackendName::Vulkan)
    {
        native_decode_expand_and_output_match_reference_rows(
            &device,
            PackedBlock::new(&device, Experts::Iq4),
            false,
            &[1, 3, 8],
            decode_mappings(&device),
        );
    }
}

/// A decode block's routing, expansion and output, and the next block's
/// recurrent input projection, as one CPU graph, with and without the INT8
/// form: the router emits the normalized rows' operand that the expansion
/// reads, the expansion the products' that the output reads, and the output
/// the next block's operand, its hidden rows normalized under the next
/// block's input norm (and, under INT8, their Q8 blocks), which the
/// projection reads in place of its form launch: each formed once in
/// its producer's epilogue, in the forms its consumer reads. Every result
/// has the bits of the four entries called alone, which form their own
/// operands: at 1, 3 and 8 rows (the tiled projection pairs at one), every
/// CPU mapping, with experts that take the INT8 dot (the form launches left
/// out) and with NVFP4-operand experts, whose own operands the expansion's
/// and output's form launches still form.
#[test]
fn cpu_decode_reads_the_operands_its_producers_emit() {
    let Some(device) = seismic_native_tests::devices(&[seismic::BackendName::Cpu]).into_iter().next() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    for experts in [Experts::Packed, Experts::Fp4A4] {
        let block = PackedBlock::new(&device, experts);
        let (h, e, k, f, s) = (
            PackedBlock::HIDDEN as u64,
            PackedBlock::EXPERTS as u64,
            PackedBlock::CHOICES as u64,
            block.routed.features as u64,
            PackedBlock::SHARED as u64,
        );
        let activation = seismic::Element::bf16();
        let [gate_scale, up_scale] = block.routed.expansion_scales(&device);
        // The next block's recurrent input projection: one key and sixteen
        // value heads of width 32, Q8 weights, its own input norm.
        let (nk, nv, w) = (1u64, 16u64, 32u64);
        let q8 = element(&device, "q8g32s");
        let next_norm = f32_tensor(&device, &[h], &pattern(h as usize, 74, 1.0).iter().map(|v| 1.0 + 0.25 * v).collect::<Vec<_>>());
        let next_weights = [((2 * nk + nv) * w, 75), (nv * w, 76), (nv, 77), (nv, 78)]
            .map(|(n, seed)| packed_weight(&device, "q8g32s", &[n, h], seed).0);
        let next_epsilon = 1e-6f32;
        for rows in [1u64, 3, 8] {
            let residual = f32_tensor(&device, &[rows, h], &pattern((rows * h) as usize, 70 + rows as u32, 2.0));
            let norm = f32_tensor(&device, &[h], &pattern(h as usize, 71, 1.0).iter().map(|v| 1.0 + 0.25 * v).collect::<Vec<_>>());
            let router = f32_tensor(&device, &[e, h], &pattern((e * h) as usize, 72, 0.5));
            let shared_router = f32_tensor(&device, &[h], &pattern(h as usize, 73, 0.5));
            let expert_scale = absent(&device);
            let inject = no_inject(&device, rows);
            let sources = no_sources(&device, rows, k);
            let tier = no_tier(&block.expert_down);
            for (int8, mapping) in [0u64, 1].into_iter().flat_map(|int8| decode_mappings(&device).into_iter().map(move |mapping| (int8, mapping))) {
                let label = format!("{experts:?} rows {rows} INT8 {int8} mapping {mapping:?}");
                let statics = [("H", h), ("K", k), ("F", f), ("S", s)];
                let route = declared::prepared(
                    &device,
                    &routed_route::native_entry_with(routed_route::Elements { NW: seismic::Element::f32(), RW: seismic::Element::f32(), SRW: seismic::Element::f32(), A: activation }),
                    // The router's rows split over the pool: it takes no mapping.
                    &decode_specialization::<routed_route::Entry>(&device, &[("H", h), ("E", e), ("K", k)], &[]),
                )
                .unwrap();
                let expand = declared::prepared(
                    &device,
                    &routed_expand::native_entry_with(routed_expand::Elements {
                        A: activation,
                        EGW: block.expert_gate.element(),
                        EUW: block.expert_up.element(),
                        SGW: block.shared_gate.element(),
                        SUW: block.shared_up.element(),
                    }),
                    &decode_specialization::<routed_expand::Entry>(
                        &device,
                        &[statics.as_slice(), &block.routed.expansion_statics()].concat(),
                        &mapping,
                    )
                    .with_param("INT8", int8),
                )
                .unwrap();
                let output = declared::prepared(
                    &device,
                    &routed_output::native_entry_with(routed_output::Elements { A: activation, EDW: block.expert_down.element(), SDW: block.shared_down.element(), NW: seismic::Element::f32() }),
                    &decode_specialization::<routed_output::Entry>(&device, &statics, &mapping).with_param("INT8", int8),
                )
                .unwrap();
                let project = declared::prepared(
                    &device,
                    &gated_delta_project::native_entry_with(gated_delta_project::Elements { NW: seismic::Element::f32(), QW: q8, GW: q8, AW: q8, BW: q8, A: activation }),
                    &decode_specialization::<gated_delta_project::Entry>(
                        &device,
                        &[("H", h), ("NK", nk), ("NV", nv), ("W", w)],
                        &mapping,
                    )
                    .with_param("INT8", int8),
                )
                .unwrap();

                // Alone, each forms its own operands.
                let mut routes = i32_tensor(&device, &[rows, k], &vec![0; (rows * k) as usize]);
                let mut scores = f32_tensor(&device, &[rows, k], &vec![0.0; (rows * k) as usize]);
                let routed = route
                    .call(&charges, routed_route::Args {
                        residual: &residual,
                        norm: &norm,
                        router: &router,
                        shared_router: &shared_router,
                        routes: &mut routes,
                        scores: &mut scores,
                        eps: 1e-6,
                        normalize: 1,
                        expert_scale: &expert_scale,
                    })
                    .unwrap();
                let expanded = expand
                    .call(&charges, routed_expand::Args {
                        normalized: &routed.r0,
                        routes: &routes,
                        expert_gate: &block.expert_gate,
                        expert_up: &block.expert_up,
                        shared_gate: &block.shared_gate,
                        shared_up: &block.shared_up,
                        gate_scale: &gate_scale,
                        up_scale: &up_scale,
                    })
                    .unwrap();
                let alone = output
                    .call(&charges, routed_output::Args {
                        residual: &residual,
                        inject: &inject,
                        expert_product: &expanded.r0,
                        shared_product: &expanded.r1,
                        routes: &routes,
                        sources: &sources,
                        scores: &scores,
                        coefficient: &routed.r1,
                        expert_down: &block.expert_down,
                        tier_down: &tier,
                        shared_down: &block.shared_down,
                        operand_norm: &next_norm,
                        operand_epsilon: next_epsilon,
                    })
                    .unwrap()
                    .value;
                let projected = project
                    .call(&charges, gated_delta_project::Args {
                        hidden: &alone,
                        input_norm: &next_norm,
                        qkv_weight: &next_weights[0],
                        gate_weight: &next_weights[1],
                        alpha_weight: &next_weights[2],
                        beta_weight: &next_weights[3],
                        epsilon: next_epsilon,
                        qkv_bound: &expert_scale,
                        gate_bound: &expert_scale,
                    })
                    .unwrap()
                    .value;

                // In one graph, each consumer reads its producer's operand.
                let mut graph = device.native_graph();
                let mut port = |tensor: &seismic::Tensor| graph.port(tensor.element(), &tensor.extents()).unwrap();
                let [residual_port, norm_port, router_port, shared_router_port, expert_scale_port, inject_port, sources_port] =
                    [&residual, &norm, &router, &shared_router, &expert_scale, &inject, &sources].map(&mut port);
                let [gate_port, up_port, shared_gate_port, shared_up_port, gate_scale_port, up_scale_port, down_port, tier_port, shared_down_port] =
                    [
                        &block.expert_gate,
                        &block.expert_up,
                        &block.shared_gate,
                        &block.shared_up,
                        &gate_scale,
                        &up_scale,
                        &block.expert_down,
                        &tier,
                        &block.shared_down,
                    ]
                    .map(&mut port);
                let [next_norm_port, qkv_port, z_port, alpha_port, beta_port] =
                    [&next_norm, &next_weights[0], &next_weights[1], &next_weights[2], &next_weights[3]].map(&mut port);
                let graph_routes = i32_tensor(&device, &[rows, k], &vec![0; (rows * k) as usize]);
                let graph_scores = f32_tensor(&device, &[rows, k], &vec![0.0; (rows * k) as usize]);
                let mut routes_port = port(&graph_routes);
                let mut scores_port = port(&graph_scores);
                let stored = [
                    bf16_tensor(&device, &[rows, h], &vec![0.0; (rows * h) as usize]),
                    bf16_tensor(&device, &[rows, k, f], &vec![0.0; (rows * k * f) as usize]),
                    bf16_tensor(&device, &[rows, s], &vec![0.0; (rows * s) as usize]),
                    f32_tensor(&device, &[rows, h], &vec![0.0; (rows * h) as usize]),
                    seismic::Tensor::zeros(&charges, projected.element(), &projected.extents()).unwrap(),
                ];
                let stored_ports = [&stored[0], &stored[1], &stored[2], &stored[3], &stored[4]].map(&mut port);
                let routed_in_graph = graph
                    .enqueue(
                        &route,
                        routed_route::WorkflowArgs {
                            residual: residual_port.tensor().into(),
                            norm: norm_port.tensor().into(),
                            router: router_port.tensor().into(),
                            shared_router: shared_router_port.tensor().into(),
                            routes: routes_port.tensor_mut().into(),
                            scores: scores_port.tensor_mut().into(),
                            eps: seismic::Argument::from(1e-6f32),
                            normalize: seismic::Argument::from(1i32),
                            expert_scale: expert_scale_port.tensor().into(),
                        },
                    )
                    .unwrap();
                let expanded_in_graph = graph
                    .enqueue(
                        &expand,
                        routed_expand::WorkflowArgs {
                            normalized: (&routed_in_graph.r0).into(),
                            routes: routes_port.tensor().into(),
                            expert_gate: gate_port.tensor().into(),
                            expert_up: up_port.tensor().into(),
                            shared_gate: shared_gate_port.tensor().into(),
                            shared_up: shared_up_port.tensor().into(),
                            gate_scale: gate_scale_port.tensor().into(),
                            up_scale: up_scale_port.tensor().into(),
                        },
                    )
                    .unwrap();
                let value = graph
                    .enqueue(
                        &output,
                        routed_output::WorkflowArgs {
                            residual: residual_port.tensor().into(),
                            inject: inject_port.tensor().into(),
                            expert_product: (&expanded_in_graph.r0).into(),
                            shared_product: (&expanded_in_graph.r1).into(),
                            routes: routes_port.tensor().into(),
                            sources: sources_port.tensor().into(),
                            scores: scores_port.tensor().into(),
                            coefficient: (&routed_in_graph.r1).into(),
                            expert_down: down_port.tensor().into(),
                            tier_down: tier_port.tensor().into(),
                            shared_down: shared_down_port.tensor().into(),
                            operand_norm: next_norm_port.tensor().into(),
                            operand_epsilon: seismic::Argument::from(next_epsilon),
                        },
                    )
                    .unwrap()
                    .value;
                let projected_in_graph = graph
                    .enqueue(
                        &project,
                        gated_delta_project::WorkflowArgs {
                            hidden: (&value).into(),
                            input_norm: next_norm_port.tensor().into(),
                            qkv_weight: qkv_port.tensor().into(),
                            gate_weight: z_port.tensor().into(),
                            alpha_weight: alpha_port.tensor().into(),
                            beta_weight: beta_port.tensor().into(),
                            epsilon: seismic::Argument::from(next_epsilon),
                            qkv_bound: expert_scale_port.tensor().into(),
                            gate_bound: expert_scale_port.tensor().into(),
                        },
                    )
                    .unwrap()
                    .value;
                for (result, port) in [&routed_in_graph.r0, &expanded_in_graph.r0, &expanded_in_graph.r1, &value, &projected_in_graph]
                    .into_iter()
                    .zip(&stored_ports)
                {
                    graph.store(result, port).unwrap();
                }
                let plan = graph.seal().unwrap();
                let mut bindings = plan.bindings();
                let inputs = [
                    (&residual_port, &residual),
                    (&norm_port, &norm),
                    (&router_port, &router),
                    (&shared_router_port, &shared_router),
                    (&expert_scale_port, &expert_scale),
                    (&inject_port, &inject),
                    (&sources_port, &sources),
                    (&gate_port, &block.expert_gate),
                    (&up_port, &block.expert_up),
                    (&shared_gate_port, &block.shared_gate),
                    (&shared_up_port, &block.shared_up),
                    (&gate_scale_port, &gate_scale),
                    (&up_scale_port, &up_scale),
                    (&down_port, &block.expert_down),
                    (&tier_port, &tier),
                    (&shared_down_port, &block.shared_down),
                    (&next_norm_port, &next_norm),
                    (&qkv_port, &next_weights[0]),
                    (&z_port, &next_weights[1]),
                    (&alpha_port, &next_weights[2]),
                    (&beta_port, &next_weights[3]),
                    (&routes_port, &graph_routes),
                    (&scores_port, &graph_scores),
                ];
                for (port, tensor) in inputs.into_iter().chain(stored_ports.iter().zip(&stored)) {
                    bindings.set(&charges, port, tensor).unwrap();
                }
                let (_, completion) =
                    plan.new_slot(&charges).unwrap().attach(bindings, plan.new_outputs(&charges).unwrap()).unwrap().submit().unwrap();
                completion.wait().unwrap();
                let bytes = |tensor: &seismic::Tensor| tensor.read_to_host().unwrap();
                assert_eq!(bytes(&graph_routes), bytes(&routes), "{label} routes");
                assert_eq!(bytes(&stored[0]), bytes(&routed.r0), "{label} normalized");
                assert_eq!(bytes(&stored[1]), bytes(&expanded.r0), "{label} expert products");
                assert_eq!(bytes(&stored[2]), bytes(&expanded.r1), "{label} shared products");
                assert_eq!(bytes(&stored[3]), bytes(&alone), "{label} output");
                assert_eq!(bytes(&stored[4]), bytes(&projected), "{label} next block's projection");
            }
        }
    }
}

/// Three decode blocks, each its own CPU graph of its recurrent input
/// projection and its routed output, carrying the operand of each block's
/// output to the next block's projection, with and without the INT8 form:
/// the first block's projection forms its own operand and its output carries
/// out its rows normalized under the next block's input norm (and their Q8
/// blocks under INT8); the next block's projection reads that carry in place
/// of its form launch and its output carries out again; the last
/// block's projection reads the carry and its output binds no norm, so it
/// emits nothing. Every projection and output has the bits of the entries
/// called alone, which form their own operands. A carry whose forms differ
/// from the projection's at its call is refused.
#[test]
fn cpu_decode_carries_operands_across_block_graphs() {
    let Some(device) = seismic_native_tests::devices(&[seismic::BackendName::Cpu]).into_iter().next() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let block = PackedBlock::new(&device, Experts::Packed);
    let (h, e, k, f, s) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::EXPERTS as u64,
        PackedBlock::CHOICES as u64,
        block.routed.features as u64,
        PackedBlock::SHARED as u64,
    );
    // Decode: one row, where the tiled projection pairs.
    let rows = 1u64;
    let activation = seismic::Element::bf16();
    let [gate_scale, up_scale] = block.routed.expansion_scales(&device);
    let (nk, nv, w) = (1u64, 16u64, 32u64);
    let q8 = element(&device, "q8g32s");
    let epsilon = 1e-6f32;
    const BLOCKS: usize = 3;
    // Each block's input norm, projection weights and residual.
    let norms: [seismic::Tensor; BLOCKS] = std::array::from_fn(|b| {
        let values = pattern(h as usize, 80 + b as u32, 1.0).iter().map(|v| 1.0 + 0.25 * v).collect::<Vec<_>>();
        f32_tensor(&device, &[h], &values)
    });
    let widths = [(2 * nk + nv) * w, nv * w, nv, nv];
    let weights: [[seismic::Tensor; 4]; BLOCKS] = std::array::from_fn(|b| {
        std::array::from_fn(|index| packed_weight(&device, "q8g32s", &[widths[index], h], 84 + (4 * b + index) as u32).0)
    });
    let residuals: [seismic::Tensor; BLOCKS] =
        std::array::from_fn(|b| f32_tensor(&device, &[rows, h], &pattern((rows * h) as usize, 96 + b as u32, 2.0)));
    let first_hidden = f32_tensor(&device, &[rows, h], &pattern((rows * h) as usize, 99, 2.0));
    let no_norm = absent(&device);
    let expert_scale = absent(&device);
    let inject = no_inject(&device, rows);
    let sources = no_sources(&device, rows, k);
    let tier = no_tier(&block.expert_down);
    let statics = [("H", h), ("K", k), ("F", f), ("S", s)];
    for int8 in [0u64, 1] {
        let forms = seismic::OperandForms { quantized: int8 == 1, nvfp4: false };
        let expansion = [statics.as_slice(), &block.routed.expansion_statics()].concat();
        for mapping in decode_mappings(&device) {
            let label = format!("INT8 {int8} mapping {mapping:?}");
            let route = declared::prepared(
                &device,
                &routed_route::native_entry_with(routed_route::Elements { NW: seismic::Element::f32(), RW: seismic::Element::f32(), SRW: seismic::Element::f32(), A: activation }),
                // The router's rows split over the pool: it takes no mapping.
                &decode_specialization::<routed_route::Entry>(&device, &[("H", h), ("E", e), ("K", k)], &[]),
            )
            .unwrap();
            let expand = declared::prepared(
                &device,
                &routed_expand::native_entry_with(routed_expand::Elements {
                    A: activation,
                    EGW: block.expert_gate.element(),
                    EUW: block.expert_up.element(),
                    SGW: block.shared_gate.element(),
                    SUW: block.shared_up.element(),
                }),
                &decode_specialization::<routed_expand::Entry>(&device, &expansion, &mapping),
            )
            .unwrap();
            let output = declared::prepared(
                &device,
                &routed_output::native_entry_with(routed_output::Elements {
                    A: activation,
                    EDW: block.expert_down.element(),
                    SDW: block.shared_down.element(),
                    NW: seismic::Element::f32(),
                }),
                &decode_specialization::<routed_output::Entry>(&device, &statics, &mapping).with_param("INT8", int8),
            )
            .unwrap();
            let project = declared::prepared(
                &device,
                &gated_delta_project::native_entry_with(gated_delta_project::Elements { NW: seismic::Element::f32(), QW: q8, GW: q8, AW: q8, BW: q8, A: activation }),
                &decode_specialization::<gated_delta_project::Entry>(
                    &device,
                    &[("H", h), ("NK", nk), ("NV", nv), ("W", w)],
                    &mapping,
                )
                .with_param("INT8", int8),
            )
            .unwrap();

            // The routed inputs every block's output reads, formed once.
            let mut routes = i32_tensor(&device, &[rows, k], &vec![0; (rows * k) as usize]);
            let mut scores = f32_tensor(&device, &[rows, k], &vec![0.0; (rows * k) as usize]);
            let routed = route
                .call(&charges, routed_route::Args {
                    residual: &residuals[0],
                    norm: &norms[0],
                    router: &f32_tensor(&device, &[e, h], &pattern((e * h) as usize, 100, 0.5)),
                    shared_router: &f32_tensor(&device, &[h], &pattern(h as usize, 101, 0.5)),
                    routes: &mut routes,
                    scores: &mut scores,
                    eps: epsilon,
                    normalize: 1,
                    expert_scale: &expert_scale,
                })
                .unwrap();
            let expanded = expand
                .call(&charges, routed_expand::Args {
                    normalized: &routed.r0,
                    routes: &routes,
                    expert_gate: &block.expert_gate,
                    expert_up: &block.expert_up,
                    shared_gate: &block.shared_gate,
                    shared_up: &block.shared_up,
                    gate_scale: &gate_scale,
                    up_scale: &up_scale,
                })
                .unwrap();
            // The norm each block's output forms the next block's operand
            // under: the last block's none.
            let next_norm = |b: usize| norms.get(b + 1).unwrap_or(&no_norm);
            let next_epsilon = |b: usize| if b + 1 < BLOCKS { epsilon } else { 0.0 };

            // Alone, each block's entries form their own operands.
            let mut alone: Vec<(seismic::Tensor, seismic::Tensor)> = Vec::new();
            for b in 0..BLOCKS {
                let hidden = alone.last().map_or(&first_hidden, |(_, output)| output);
                let projected = project
                    .call(&charges, gated_delta_project::Args {
                        hidden,
                        input_norm: &norms[b],
                        qkv_weight: &weights[b][0],
                        gate_weight: &weights[b][1],
                        alpha_weight: &weights[b][2],
                        beta_weight: &weights[b][3],
                        epsilon,
                        qkv_bound: &expert_scale,
                        gate_bound: &expert_scale,
                    })
                    .unwrap()
                    .value;
                let value = output
                    .call(&charges, routed_output::Args {
                        residual: &residuals[b],
                        inject: &inject,
                        expert_product: &expanded.r0,
                        shared_product: &expanded.r1,
                        routes: &routes,
                        sources: &sources,
                        scores: &scores,
                        coefficient: &routed.r1,
                        expert_down: &block.expert_down,
                        tier_down: &tier,
                        shared_down: &block.shared_down,
                        operand_norm: next_norm(b),
                        operand_epsilon: next_epsilon(b),
                    })
                    .unwrap()
                    .value;
                alone.push((projected, value));
            }

            // Each block its own graph, reading the carry the one before
            // wrote.
            let mut carried: Option<[seismic::Tensor; 2]> = None;
            let mut hidden_in = first_hidden.clone();
            for b in 0..BLOCKS {
                let block_label = format!("{label} block {b}");
                // Block `b`'s graph, its carry in read in `carry_forms` where
                // it reads one.
                let build = |carry_forms: seismic::OperandForms| {
                    let mut graph = device.native_graph();
                    let mut port = |tensor: &seismic::Tensor| graph.port(tensor.element(), &tensor.extents()).unwrap();
                    let [hidden_port, norm_port, qkv_port, z_port, alpha_port, beta_port, bound_port] = [
                        &hidden_in,
                        &norms[b],
                        &weights[b][0],
                        &weights[b][1],
                        &weights[b][2],
                        &weights[b][3],
                        &expert_scale,
                    ]
                    .map(&mut port);
                    let [residual_port, inject_port, expert_product_port, shared_product_port, routes_port, sources_port, scores_port, coefficient_port, down_port, tier_port, shared_down_port, next_norm_port] =
                        [
                            &residuals[b],
                            &inject,
                            &expanded.r0,
                            &expanded.r1,
                            &routes,
                            &sources,
                            &scores,
                            &routed.r1,
                            &block.expert_down,
                            &tier,
                            &block.shared_down,
                            next_norm(b),
                        ]
                        .map(&mut port);
                    let stored = [
                        seismic::Tensor::zeros(&charges, alone[b].0.element(), &alone[b].0.extents()).unwrap(),
                        seismic::Tensor::zeros(&charges, alone[b].1.element(), &alone[b].1.extents()).unwrap(),
                    ];
                    let stored_ports = [&stored[0], &stored[1]].map(&mut port);
                    let carry_in = (b > 0).then(|| graph.operand_carry(rows, h, activation).unwrap());
                    let carry_out = (b + 1 < BLOCKS).then(|| graph.operand_carry(rows, h, activation).unwrap());
                    let projected = graph
                        .enqueue(
                            &project,
                            gated_delta_project::WorkflowArgs {
                                hidden: hidden_port.tensor().into(),
                                input_norm: norm_port.tensor().into(),
                                qkv_weight: qkv_port.tensor().into(),
                                gate_weight: z_port.tensor().into(),
                                alpha_weight: alpha_port.tensor().into(),
                                beta_weight: beta_port.tensor().into(),
                                epsilon: seismic::Argument::from(epsilon),
                                qkv_bound: bound_port.tensor().into(),
                                gate_bound: bound_port.tensor().into(),
                            },
                        )
                        .unwrap()
                        .value;
                    let value = graph
                        .enqueue(
                            &output,
                            routed_output::WorkflowArgs {
                                residual: residual_port.tensor().into(),
                                inject: inject_port.tensor().into(),
                                expert_product: expert_product_port.tensor().into(),
                                shared_product: shared_product_port.tensor().into(),
                                routes: routes_port.tensor().into(),
                                sources: sources_port.tensor().into(),
                                scores: scores_port.tensor().into(),
                                coefficient: coefficient_port.tensor().into(),
                                expert_down: down_port.tensor().into(),
                                tier_down: tier_port.tensor().into(),
                                shared_down: shared_down_port.tensor().into(),
                                operand_norm: next_norm_port.tensor().into(),
                                operand_epsilon: seismic::Argument::from(next_epsilon(b)),
                            },
                        )
                        .unwrap()
                        .value;
                    graph.store(&projected, &stored_ports[0]).unwrap();
                    graph.store(&value, &stored_ports[1]).unwrap();
                    if let Some(carry) = &carry_in {
                        graph.carry_in(&hidden_port, carry, carry_forms, &norm_port, epsilon).unwrap();
                    }
                    if let Some(carry) = &carry_out {
                        graph.carry_out(&value, carry, forms).unwrap();
                    }
                    let inputs = [
                        (hidden_port, &hidden_in),
                        (norm_port, &norms[b]),
                        (qkv_port, &weights[b][0]),
                        (z_port, &weights[b][1]),
                        (alpha_port, &weights[b][2]),
                        (beta_port, &weights[b][3]),
                        (bound_port, &expert_scale),
                        (residual_port, &residuals[b]),
                        (inject_port, &inject),
                        (expert_product_port, &expanded.r0),
                        (shared_product_port, &expanded.r1),
                        (routes_port, &routes),
                        (sources_port, &sources),
                        (scores_port, &scores),
                        (coefficient_port, &routed.r1),
                        (down_port, &block.expert_down),
                        (tier_port, &tier),
                        (shared_down_port, &block.shared_down),
                        (next_norm_port, next_norm(b)),
                    ]
                    .into_iter()
                    .map(|(port, tensor)| (port, tensor.clone()))
                    .chain(stored_ports.into_iter().zip(stored.clone()))
                    .collect::<Vec<_>>();
                    (graph.seal(), inputs, stored, carry_in, carry_out)
                };
                // A carry in of other forms than the projection reads at
                // its call is refused.
                if b > 0 {
                    let other = seismic::OperandForms { quantized: !forms.quantized, ..forms };
                    assert!(
                        matches!(
                            build(other).0.err(),
                            Some(seismic::CallError::Workflow(seismic::WorkflowError::OperandForms { node: 0 }))
                        ),
                        "{block_label}: a carry in of other forms"
                    );
                }
                let (plan, inputs, stored, carry_in, carry_out) = build(forms);
                let plan = plan.unwrap();
                let mut bindings = plan.bindings();
                for (port, tensor) in &inputs {
                    bindings.set(&charges, port, tensor).unwrap();
                }
                if let Some(carry) = &carry_in {
                    let tensors = carried.as_ref().expect("the block before carried out");
                    bindings.set(&charges, &carry.rows, &tensors[0]).unwrap();
                    bindings.set(&charges, &carry.quantized, &tensors[1]).unwrap();
                }
                let out = carry_out.as_ref().map(|carry| carry.allocate(&charges).unwrap());
                if let (Some(carry), Some(tensors)) = (&carry_out, &out) {
                    bindings.set(&charges, &carry.rows, &tensors[0]).unwrap();
                    bindings.set(&charges, &carry.quantized, &tensors[1]).unwrap();
                }
                let (_, completion) =
                    plan.new_slot(&charges).unwrap().attach(bindings, plan.new_outputs(&charges).unwrap()).unwrap().submit().unwrap();
                completion.wait().unwrap();
                let bytes = |tensor: &seismic::Tensor| tensor.read_to_host().unwrap();
                assert_eq!(bytes(&stored[0]), bytes(&alone[b].0), "{block_label} projection");
                assert_eq!(bytes(&stored[1]), bytes(&alone[b].1), "{block_label} output");
                carried = out;
                hidden_in = stored[1].clone();
            }
        }
    }
}

fn native_decode_expand_and_output_match_reference_on(device: &seismic::Device, int8: bool) {
    let ran = native_decode_expand_and_output_match_reference_rows(
        device,
        PackedBlock::new(device, Experts::Packed),
        int8,
        &[1, 2, 3, 8],
        decode_mappings(device),
    );
    // Every value of the Metal output's one top-level LANES, which every
    // launch reads, is checked: the decode mappings value each.
    if device.backend() == seismic::BackendName::Metal {
        let implementation = routed_output::native_implementation(device).unwrap().unwrap();
        let lanes = implementation
            .params
            .iter()
            .find(|parameter| parameter.name == "LANES")
            .expect("the output declares LANES at the top");
        let mut declared = lanes.values.clone();
        declared.sort_unstable();
        assert_eq!(ran, declared, "the decode mappings run every declared LANES");
    }
}

/// Checks the native decode expansion and output against the host reference
/// at each of `row_counts` rows and each of `mappings`, and returns the
/// output's LANES values the checks ran, sorted and distinct.
fn native_decode_expand_and_output_match_reference_rows(
    device: &seismic::Device,
    block: PackedBlock,
    int8: bool,
    row_counts: &[usize],
    mappings: Vec<Vec<(&'static str, u64)>>,
) -> Vec<u64> {
    let charges = seismic::Charges::tally(device);
    let mut ran = Vec::new();
    let (h, k, f, s) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::CHOICES as u64,
        block.routed.features as u64,
        PackedBlock::SHARED as u64,
    );
    for &rows in row_counts {
        let routing = PackedBlock::routing(rows, 50 + rows as u32);
        let m = rows as u64;
        let (expert, shared) = block.routed.expand(&routing, bf16_round);
        let reference = block.routed.output(&routing, &expert, &shared, bf16_round);
        // The output reads the host's products: nothing carried into it.
        let forms = RoutedForms::decode(device, block.weights(), rows);
        let (expert_bound, shared_bound) = block.routed.expand_bounds(&routing, &forms, Float::Bf16, device.backend());
        let output_bound = block.routed.output_bounds(
            &routing,
            (&expert, &shared),
            (&vec![0.0; expert.len()], &vec![0.0; shared.len()]),
            &forms,
            Float::Bf16,
        );
        let normalized = bf16_tensor(&device, &[m, h], &routing.normalized);
        let routes = i32_tensor(&device, &[m, k], &routing.routes);
        let scores = f32_tensor(&device, &[m, k], &routing.scores);
        let coefficient = f32_tensor(&device, &[m], &routing.coefficient);
        let residual = f32_tensor(&device, &[m, h], &routing.residual);
        let expert_product = bf16_tensor(&device, &[m, k, f], &expert);
        let shared_product = bf16_tensor(&device, &[m, s], &shared);
        let [gate_scale, up_scale] = block.routed.expansion_scales(device);
        for (index, mapping) in mappings.iter().enumerate() {
            let output_mapping =
                gathered::<routed_output::Entry>(device, index, &output_mapping(device, index, mapping));
            let mapping = gathered::<routed_expand::Entry>(device, index, mapping);
            let label =
                format!("rows {rows} mapping {mapping:?} output {output_mapping:?} INT8 {int8}");
            let statics = [("H", h), ("K", k), ("F", f), ("S", s)];
            let mut specialization = decode_specialization::<routed_expand::Entry>(
                device,
                &[statics.as_slice(), &block.routed.expansion_statics()].concat(),
                &mapping,
            );
            let mut output_specialization =
                decode_specialization::<routed_output::Entry>(device, &statics, &output_mapping);
            if is_cpu(device) {
                specialization = specialization.with_param("INT8", u64::from(int8));
                output_specialization = output_specialization.with_param("INT8", u64::from(int8));
            }
            let output_elements = || routed_output::Elements {
                A: seismic::Element::bf16(),
                EDW: block.expert_down.element(),
                SDW: block.shared_down.element(),
                NW: seismic::Element::f32(),
            };
            // A mapping the output's declaration does not admit at these
            // statics (a wider expert admits fewer rows per threadgroup).
            if declared::declared_at(&device, &routed_output::native_entry_with(output_elements()), &output_specialization).is_err() {
                continue;
            }
            ran.extend(output_mapping.iter().filter(|(name, _)| *name == "LANES").map(|(_, value)| *value));
            let expanded = declared::prepared(
                &device,
                &routed_expand::native_entry_with(routed_expand::Elements {
                    A: seismic::Element::bf16(),
                    EGW: block.expert_gate.element(),
                    EUW: block.expert_up.element(),
                    SGW: block.shared_gate.element(),
                    SUW: block.shared_up.element(),
                }),
                &specialization,
            )
            .unwrap()
            .call(&charges, routed_expand::Args {
                normalized: &normalized,
                routes: &routes,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
                gate_scale: &gate_scale,
                up_scale: &up_scale,
            })
            .unwrap();
            assert_arithmetic(
                &format!("{label} expert product"),
                &read_bf16(&expanded.r0),
                &expert,
                int8,
                &expert_bound,
            );
            assert_arithmetic(
                &format!("{label} shared product"),
                &read_bf16(&expanded.r1),
                &shared,
                int8,
                &shared_bound,
            );
            let output_native =
                declared::prepared(&device, &routed_output::native_entry_with(output_elements()), &output_specialization).unwrap();
            let output = |sources: &seismic::Tensor, scores: &seismic::Tensor, expert_down: &seismic::Tensor| {
                let value = output_native
                    .call(&charges, routed_output::Args {
                        residual: &residual,
                        inject: &no_inject(&device, residual.extents()[0]),
                        expert_product: &expert_product,
                        shared_product: &shared_product,
                        routes: &routes,
                        sources,
                        scores,
                        coefficient: &coefficient,
                        expert_down,
                        tier_down: &no_tier(expert_down),
                        shared_down: &block.shared_down,
                        operand_norm: &absent(&device),
                        operand_epsilon: 0.0,
                    })
                    .unwrap()
                    .value;
                read_f32(&value)
            };
            let resident = output(&no_sources(device, m, k), &scores, &block.expert_down);
            assert_arithmetic(
                &format!("{label} output"),
                &resident,
                &reference,
                int8,
                &output_bound,
            );
            // Read where its sources name from a bank holding the experts
            // in reverse, the output keeps its bits, but for the choices of
            // the expert the sources name no record for (row 0's first),
            // which contribute nothing: the output whose scores for them
            // are zero. At every mapping, as the streamed launches (several
            // rows' among them) are the mapping's own.
            let experts = PackedBlock::EXPERTS;
            let unmapped = routing.routes[0];
            let stacked = block.expert_down.read_to_host().unwrap();
            let reversed =
                stacked.chunks_exact(stacked.len() / experts).rev().flatten().copied().collect::<Vec<_>>();
            let bank = seismic::Tensor::from_host(
                &charges,
                block.expert_down.element(),
                &block.expert_down.extents(),
                &reversed,
            )
            .unwrap();
            let records = routing
                .routes
                .iter()
                .map(|&expert| if expert == unmapped { -1 } else { experts as i32 - 1 - expert })
                .collect::<Vec<_>>();
            let unscored = routing
                .routes
                .iter()
                .zip(&routing.scores)
                .map(|(&expert, &score)| if expert == unmapped { 0.0 } else { score })
                .collect::<Vec<_>>();
            let bits = |values: Vec<f32>| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(output(&decode_sources(device, m, k, &records), &scores, &bank)),
                bits(output(&no_sources(device, m, k), &f32_tensor(device, &[m, k], &unscored), &block.expert_down)),
                "{label} output where its sources name"
            );
        }
    }
    ran.sort_unstable();
    ran.dedup();
    ran
}

#[test]
#[ignore]
fn cuda_scoped_routed_decode_matches_reference_for_all_code_variants() {
    let Some(device) = native_device() else {
        return;
    };
    if device.backend() != seismic::BackendName::Cuda {
        return;
    }
    let mappings = [1u64, 2]
        .into_iter()
        .flat_map(|tpw| {
            [1u64, 2, 4]
                .into_iter()
                .map(move |split| vec![("TPW", tpw), ("KSPLIT", split)])
        })
        .collect();
    native_decode_expand_and_output_match_reference_rows(
        &device,
        PackedBlock::new(&device, Experts::Packed),
        false,
        &[1],
        mappings,
    );
}

/// What these tests hold a tuned configuration to against its default: 8
/// rounding steps of the `bf16` activations, the engine's standard α.
fn tuning_requirement() -> seismic::NumericalRequirement {
    seismic::NumericalRequirement::RoundingSteps {
        alpha: 8.,
        activation: Some(seismic::DType::BF16),
    }
}

#[test]
#[ignore]
fn cuda_scoped_routed_expand_tunes_declared_candidates() {
    let Some(device) = native_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    if device.backend() != seismic::BackendName::Cuda {
        return;
    }
    let block = PackedBlock::new(&device, Experts::Packed);
    let (h, k, f, s) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::CHOICES as u64,
        block.routed.features as u64,
        PackedBlock::SHARED as u64,
    );
    let rows = [1usize, 3, 8];
    let inputs = rows
        .iter()
        .map(|rows| {
            let routing = PackedBlock::routing(*rows, 70 + *rows as u32);
            (
                bf16_tensor(&device, &[*rows as u64, h], &routing.normalized),
                i32_tensor(&device, &[*rows as u64, k], &routing.routes),
            )
        })
        .collect::<Vec<_>>();
    let unscaled = absent(&device);
    let points: Vec<_> = rows
        .iter()
        .zip(&inputs)
        .map(|(rows, (normalized, routes))| seismic::TuningPoint {
            label: format!("rows{rows}"),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![routed_expand::Args {
                normalized,
                routes,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
                gate_scale: &unscaled,
                up_scale: &unscaled,
            }],
            initialize: None,
            written: Default::default(),
        })
        .collect();
    let search = seismic::Strategy::Search(seismic::SearchPlan {
        allowance: std::time::Duration::from_secs(600),

        settings: seismic::SearchSettings { error_rate: 0.05 },
        min_sample_seconds: 0.001,
        start: Vec::new(),
    });
    let statics = specialize(
        &[("H", h), ("K", k), ("F", f), ("S", s), ("GS", 0), ("US", 0)],
        &[],
    );
    let result = routed_expand::native_tune_with(
        &charges,
        routed_expand::Elements {
            A: seismic::Element::bf16(),
            EGW: element(&device, "q4k"),
            EUW: element(&device, "q4k"),
            SGW: element(&device, "q8g32s"),
            SUW: element(&device, "q8g32s"),
        },
        &statics,
        points,
        tuning_requirement(),
        search,
    )
    .unwrap();
    assert!(matches!(
        &result.method,
        seismic::TuningMethod::Search {
            groups: 1,
            complete: true,
            ..
        }
    ));
    assert_eq!(result.rejections().count(), 0);
    println!(
        "routed_expand CUDA BF16: choice {:?}, method {:?}, time {:?}, records {}",
        result.overall.launches,
        result.method,
        result.time,
        result.configurations.len()
    );
}

#[test]
#[ignore]
fn cuda_scoped_routed_output_tunes_declared_candidates() {
    let Some(device) = native_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    if device.backend() != seismic::BackendName::Cuda {
        return;
    }
    let block = PackedBlock::new(&device, Experts::Packed);
    let (h, k, f, s) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::CHOICES as u64,
        block.routed.features as u64,
        PackedBlock::SHARED as u64,
    );
    let rows = [1usize, 3, 8];
    let inputs = rows
        .iter()
        .map(|rows| {
            let routing = PackedBlock::routing(*rows, 80 + *rows as u32);
            let (expert, shared) = block.routed.expand(&routing, bf16_round);
            let m = *rows as u64;
            (
                f32_tensor(&device, &[m, h], &routing.residual),
                bf16_tensor(&device, &[m, k, f], &expert),
                bf16_tensor(&device, &[m, s], &shared),
                i32_tensor(&device, &[m, k], &routing.routes),
                f32_tensor(&device, &[m, k], &routing.scores),
                f32_tensor(&device, &[m], &routing.coefficient),
                no_inject(&device, m),
                no_sources(&device, m, k),
            )
        })
        .collect::<Vec<_>>();
    let no_down_tier = no_tier(&block.expert_down);
    let no_norm = absent(&device);
    let points: Vec<_> = rows
        .iter()
        .zip(&inputs)
        .map(|(rows, input)| seismic::TuningPoint {
            label: format!("rows{rows}"),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![routed_output::Args {
                residual: &input.0,
                inject: &input.6,
                expert_product: &input.1,
                shared_product: &input.2,
                routes: &input.3,
                sources: &input.7,
                scores: &input.4,
                coefficient: &input.5,
                expert_down: &block.expert_down,
                tier_down: &no_down_tier,
                shared_down: &block.shared_down,
                operand_norm: &no_norm,
                operand_epsilon: 0.0,
            }],
            initialize: None,
            written: Default::default(),
        })
        .collect();
    let search = seismic::Strategy::Search(seismic::SearchPlan {
        allowance: std::time::Duration::from_secs(600),

        settings: seismic::SearchSettings { error_rate: 0.05 },
        min_sample_seconds: 0.001,
        start: Vec::new(),
    });
    let statics = specialize(&[("H", h), ("K", k), ("F", f), ("S", s)], &[]);
    let result = routed_output::native_tune_with(
        &charges,
        routed_output::Elements {
            A: seismic::Element::bf16(),
            EDW: element(&device, "q5k"),
            SDW: element(&device, "q8g32s"),
            NW: seismic::Element::f32(),
        },
        &statics,
        points,
        tuning_requirement(),
        search,
    )
    .unwrap();
    assert!(matches!(
        &result.method,
        seismic::TuningMethod::Search {
            groups: 1,
            complete: true,
            ..
        }
    ));
    assert_eq!(result.rejections().count(), 0);
    println!(
        "routed_output CUDA BF16: choice {:?}, method {:?}, time {:?}, records {}",
        result.overall.launches,
        result.method,
        result.time,
        result.configurations.len()
    );
}

#[test]
fn native_grouped_prefill_matches_reference() {
    for device in seismic_native_tests::every_device() {
        native_grouped_prefill_matches_reference_on(&device, PackedBlock::new(&device, Experts::Packed), 0);
    }
}

/// NVFP4 experts in the grouped (prefill) expert GEMMs, and with per-expert
/// global scales on the gate and up weights in the decode expand and the
/// grouped GEMMs.
#[test]
fn native_routed_experts_match_reference_with_nvfp4_experts() {
    for device in seismic_native_tests::every_device() {
        for experts in [Experts::Fp4, Experts::Fp4A4] {
            native_decode_expand_and_output_match_reference_rows(
                &device,
                PackedBlock::new(&device, experts).scaled(),
                false,
                &[1, 3, 8],
                decode_mappings(&device),
            );
            for block in [
                PackedBlock::new(&device, experts),
                PackedBlock::new(&device, experts).scaled(),
            ] {
                native_grouped_prefill_matches_reference_on(&device, block, 0);
            }
        }
    }
}

/// The grouped tables and expert pass of rows `rows` of `routing` (its
/// normalized rows and routes) placed from row `at` in a graph sealed at
/// `capacity` rows, the other rows zero and routed to the first experts.
struct GroupedPass {
    first: seismic::Tensor,
    grouped: seismic::Tensor,
    normalized: seismic::Tensor,
    order: seismic::Tensor,
    inverse: seismic::Tensor,
    table: seismic::Tensor,
    used: seismic::Tensor,
    glu_limit: seismic::Tensor,
}

impl GroupedPass {
    fn new(device: &seismic::Device, routing: &Routing, rows: std::ops::Range<usize>, at: usize, capacity: usize) -> Self {
        let charges = seismic::Charges::tally(device);
        let (h, e, k) = (PackedBlock::HIDDEN, PackedBlock::EXPERTS, PackedBlock::CHOICES);
        let blocks = group_blocks((capacity * k) as u64, e.min(capacity * k) as u64, 32);
        let mut normalized = vec![0.0; capacity * h];
        normalized[at * h..(at + rows.len()) * h].copy_from_slice(&routing.normalized[rows.start * h..rows.end * h]);
        let mut routes = (0..capacity * k).map(|flat| (flat % k) as i32).collect::<Vec<_>>();
        routes[at * k..(at + rows.len()) * k].copy_from_slice(&routing.routes[rows.start * k..rows.end * k]);
        let fill = |shape: &[u64]| i32_tensor(device, shape, &vec![-9; shape.iter().product::<u64>() as usize]);
        let (capacity, k64) = (capacity as u64, k as u64);
        let mut pass = Self {
            first: i32_tensor(device, &[1], &[at as i32]),
            grouped: i32_tensor(device, &[1], &[rows.len() as i32]),
            normalized: bf16_tensor(device, &[capacity, h as u64], &normalized),
            order: fill(&[blocks, 32]),
            inverse: fill(&[capacity, k64]),
            table: fill(&[blocks]),
            used: fill(&[1]),
            glu_limit: no_limit(device),
        };
        declared::prepared(device, &routed_group::native_entry(), &default_group_specialization(device, e as u64, k64))
            .unwrap()
            .call(&charges, routed_group::Args {
                first: &pass.first,
                rows: &pass.grouped,
                routes: &i32_tensor(device, &[capacity, k64], &routes),
                counts: &mut fill(&[e as u64]),
                order: &mut pass.order,
                inverse: &mut pass.inverse,
                blocks: &mut pass.table,
                used: &mut pass.used,
            })
            .unwrap();
        pass
    }

    fn args<'a>(&'a self, block: &'a PackedBlock, map: &'a [seismic::Tensor; 2], scales: &'a [seismic::Tensor; 2]) -> routed_experts::Args<'a> {
        routed_experts::Args {
            first: &self.first,
            rows: &self.grouped,
            normalized: &self.normalized,
            order: &self.order,
            blocks: &self.table,
            used: &self.used,
            expert_gate: &block.expert_gate,
            expert_up: &block.expert_up,
            expert_down: &block.expert_down,
            map: &map[0],
            step: &map[1],
            records_gate: &block.expert_gate,
            records_up: &block.expert_up,
            records_down: &block.expert_down,
            activation: 0,
            glu_limit: &self.glu_limit,
            gate_scale: &scales[0],
            up_scale: &scales[1],
        }
    }

    /// The bits of each grouped (row, choice)'s expert output row.
    fn rows(&self, output: &seismic::Tensor) -> Vec<u32> {
        let (h, k) = (PackedBlock::HIDDEN, PackedBlock::CHOICES);
        let (at, rows) = (read_i32(&self.first)[0] as usize, read_i32(&self.grouped)[0] as usize);
        let values = read_bf16(output);
        read_i32(&self.inverse)[at * k..(at + rows) * k]
            .iter()
            .flat_map(|position| values[*position as usize * h..(*position as usize + 1) * h].iter().map(|value| value.to_bits()))
            .collect()
    }
}

/// A form of a grouped projection (`routed_experts`, `routed_combine`): the
/// parameters that select it over its declaration's defaults (none: the
/// default form), and whether it quantizes a GEMM's operand rows, which
/// holds its results to the tuner's output-norm guard ([`assert_arithmetic`]).
#[derive(Clone, Debug, Default)]
struct GroupedForm {
    params: Vec<(&'static str, u64)>,
    quantized: bool,
}

/// The grouped projections' forms on `device` beside their defaults, as
/// (`routed_experts`', `routed_combine`'s), each of one GEMM with the other
/// at its default. Metal: the expansion's (`TALL`, `INT8`, `PACK`), then the
/// down GEMM's (`DOWN_TALL`, `DOWN_INT8`, `DOWN_PACK`), dense's forms: the
/// staged tiles against the tall default, INT8 and PACK each on the tall
/// tiles. CUDA and the CPU: INT8, which takes both GEMMs.
fn grouped_forms(device: &seismic::Device) -> Vec<[GroupedForm; 2]> {
    fn form(params: &[(&'static str, u64)], quantized: bool) -> GroupedForm {
        GroupedForm {
            params: params.to_vec(),
            quantized,
        }
    }
    match device.backend() {
        seismic::BackendName::Metal => [("TALL", "INT8", "PACK"), ("DOWN_TALL", "DOWN_INT8", "DOWN_PACK")]
            .into_iter()
            .flat_map(|(tall, int8, pack)| {
                [
                    [form(&[(tall, 0)], false), form(&[(tall, 0)], false)],
                    [form(&[(tall, 1), (int8, 1)], true), form(&[(int8, 1)], true)],
                    [form(&[(tall, 1), (pack, 1)], true), form(&[(pack, 1)], true)],
                ]
            })
            .collect(),
        seismic::BackendName::Cuda | seismic::BackendName::Cpu => {
            vec![[form(&[("INT8", 1)], true), form(&[("INT8", 1)], true)]]
        }
        seismic::BackendName::Vulkan => Vec::new(),
    }
}

/// A grouped projection's configuration: a mapping and a form.
#[derive(Clone, Copy)]
struct Grouped<'a> {
    mapping: &'a [(&'static str, u64)],
    form: &'a GroupedForm,
}

/// Grouped projection `entry` under the specialization each configuration
/// gives it on a device ([`Mapped`]): the statics of `statics` its native
/// declares (the block rows `T` Metal's alone), the configuration's mapping
/// and form, CUDA's operand path `operand`, the rest at their declared
/// defaults.
fn grouped_projection<E: seismic::Entry>(
    entry: seismic::BoundEntry<E>,
    statics: Vec<(&'static str, u64)>,
    operand: u64,
) -> Mapped<E, impl for<'a, 'b> Fn(&'a seismic::Device, Grouped<'b>) -> seismic::NativeSpecialization> {
    Mapped::new(entry, move |device: &seismic::Device, configuration: Grouped<'_>| {
        let implementation = seismic::generated::native_implementation::<E>(device)
            .unwrap()
            .unwrap();
        let statics = statics
            .iter()
            .copied()
            .filter(|(name, _)| implementation.statics.iter().any(|declared| declared == name))
            .collect::<Vec<_>>();
        let operand = (device.backend() == seismic::BackendName::Cuda).then_some(("OPERAND", operand));
        let params = configuration
            .mapping
            .iter()
            .copied()
            .chain(operand)
            .chain(configuration.form.params.iter().copied())
            .collect::<Vec<_>>();
        specialization_on(device, &statics, &params)
    })
}

/// The grouped expert pass of the packed block's experts in its default form
/// (CUDA's operand path 0), at each of `device`'s grouped mappings its
/// declaration admits.
fn grouped_experts(
    device: &seismic::Device,
    block: &PackedBlock,
) -> Vec<(Vec<(&'static str, u64)>, seismic::NativeKernel<routed_experts::Entry>)> {
    let (h, f) = (PackedBlock::HIDDEN as u64, block.routed.features as u64);
    let pass = grouped_projection(
        routed_experts::native_entry_with(routed_experts::Elements {
            A: seismic::Element::bf16(),
            EGW: block.expert_gate.element(),
            EUW: block.expert_up.element(),
            EDW: block.expert_down.element(),
        }),
        [[("H", h), ("F", f), ("T", 32)].as_slice(), &block.routed.expansion_statics(), &[("LB", 0)]].concat(),
        0,
    );
    let form = GroupedForm::default();
    mapped::admitted("routed_experts", grouped_mappings(device), |mapping| pass.admits(device, Grouped { mapping, form: &form }))
        .into_iter()
        .map(|mapping| {
            let kernel = pass.kernel(device, Grouped { mapping: &mapping, form: &form });
            (mapping, kernel)
        })
        .collect()
}

/// A graph sealed at 16,384 rows passes only the blocks its rows fill, and a
/// row's output is the same bits whatever the capacity and whichever rows
/// share its pass: four 512-row chunks passed together at that capacity
/// give each row the bits of its own chunk's 512-row pass.
#[test]
fn native_merged_expert_pass_keeps_each_chunks_bits_at_any_capacity() {
    for device in seismic_native_tests::every_device() {
        let charges = seismic::Charges::tally(&device);
        let block = PackedBlock::new(&device, Experts::Packed);
        let routing = PackedBlock::routing(2048, 77);
        let (scales, map) = (block.routed.expansion_scales(&device), [absent_map(&device), absent_step(&device)]);
        for (mapping, experts) in grouped_experts(&device, &block) {
            let run = |pass: &GroupedPass| pass.rows(&experts.call(&charges, pass.args(&block, &map, &scales)).unwrap().value);
            let chunks = (0..4)
                .map(|chunk| run(&GroupedPass::new(&device, &routing, chunk * 512..(chunk + 1) * 512, 0, 512)))
                .collect::<Vec<_>>();
            let label = format!("{:?} {mapping:?}", device.backend());
            assert_eq!(
                run(&GroupedPass::new(&device, &routing, 0..512, 0, 16_384)),
                chunks[0],
                "{label}: 512 rows at capacity 16,384"
            );
            assert_eq!(
                run(&GroupedPass::new(&device, &routing, 0..2048, 0, 16_384)),
                chunks.concat(),
                "{label}: four chunks merged at capacity 16,384"
            );
            // A pass over a row range from a first row (a merge set's) gives
            // its rows the bits they have passed alone.
            assert_eq!(
                run(&GroupedPass::new(&device, &routing, 1024..2048, 1536, 16_384)),
                chunks[2..].concat(),
                "{label}: two chunks from row 1,536 at capacity 16,384"
            );
        }
    }
}

/// A row's expert outputs are the same bits alone in its tiles, sharing them
/// with one other row, and as the last row of full tiles: 32 rows routed to
/// the same experts, the last one passed alone, with its neighbour and with
/// all 32.
#[test]
fn native_expert_row_keeps_its_bits_at_any_tile_occupancy() {
    let (h, k) = (PackedBlock::HIDDEN, PackedBlock::CHOICES);
    for device in seismic_native_tests::every_device() {
        let charges = seismic::Charges::tally(&device);
        let block = PackedBlock::new(&device, Experts::Packed);
        let mut routing = PackedBlock::routing(32, 79);
        routing.routes = routing.routes[..k].repeat(32);
        let (scales, map) = (block.routed.expansion_scales(&device), [absent_map(&device), absent_step(&device)]);
        for (mapping, experts) in grouped_experts(&device, &block) {
            let last = |rows: std::ops::Range<usize>| {
                let pass = GroupedPass::new(&device, &routing, rows, 0, 32);
                let bits = pass.rows(&experts.call(&charges, pass.args(&block, &map, &scales)).unwrap().value);
                bits[bits.len() - k * h..].to_vec()
            };
            let alone = last(31..32);
            let label = format!("{:?} {mapping:?}", device.backend());
            assert_eq!(last(30..32), alone, "{label}: in 2-row tiles");
            assert_eq!(last(0..32), alone, "{label}: in full tiles");
        }
    }
}

/// IQ4_NL single-block experts (iq4g32s) in the decode expand and output
/// and the grouped (prefill) expert GEMMs.
#[test]
fn native_routed_experts_match_reference_with_iq4_block_experts() {
    for device in seismic_native_tests::every_device() {
        native_decode_expand_and_output_match_reference_rows(
            &device,
            PackedBlock::new(&device, Experts::Iq4Blocks),
            false,
            &[1, 3, 8],
            decode_mappings(&device),
        );
        native_grouped_prefill_matches_reference_on(
            &device,
            PackedBlock::new(&device, Experts::Iq4Blocks),
            0,
        );
    }
}

/// Q2_K experts in the grouped (prefill) expert GEMMs: on Metal the packed
/// GEMM of gate and up weights whose runs each fold a min term.
#[test]
fn native_grouped_prefill_matches_reference_with_q2k_experts() {
    for device in seismic_native_tests::every_device() {
        native_grouped_prefill_matches_reference_on(&device, PackedBlock::new(&device, Experts::Q2), 0);
    }
}

/// Q2_K experts in the ungated grouped expert pass (`routed_experts_up`,
/// relu^2): on Metal its packed GEMMs, over up weights whose two runs a
/// block each fold a min term and q3k down weights, are the packed
/// reference's (`packed`) of each block's expert: the up sums', relu^2 of
/// them rounded to bf16 (each product within what the up sums' budgets
/// leave it), then the down sums' over those products, rounded to bf16.
#[test]
fn metal_packed_experts_up_agrees_with_the_default_form_with_q2k_experts() {
    let Some(device) = seismic_native_tests::every_device()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let block = PackedBlock::new(&device, Experts::Q2);
    let (h, e, k, f) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::EXPERTS as u64,
        PackedBlock::CHOICES as u64,
        block.routed.features as u64,
    );
    let tile = 32u64;
    let experts = grouped_projection(
        routed_experts_up::native_entry_with(routed_experts_up::Elements {
            A: seismic::Element::bf16(),
            EUW: block.expert_up.element(),
            EDW: block.expert_down.element(),
        }),
        [[("H", h), ("F", f), ("T", tile)].as_slice(), &block.routed.expansion_statics()].concat(),
        0,
    );
    let mappings = grouped_mappings(&device);
    let default = GroupedForm::default();
    let packed = GroupedForm {
        params: vec![("TALL", 1), ("PACK", 1), ("DOWN_TALL", 1), ("DOWN_PACK", 1)],
        quantized: true,
    };
    let at = |form| Grouped {
        mapping: mappings[0].as_slice(),
        form,
    };
    assert!(experts.admits(&device, at(&packed)), "routed_experts_up takes PACK over q2k up and q3k down weights");
    let [_, up_scale] = block.routed.expansion_scales(&device);
    // Past 64 rows, where the forms apply; 200 leaves a partial tile.
    for rows in [80usize, 200] {
        let routing = PackedBlock::routing(rows, 190 + rows as u32);
        let m = rows as u64;
        let blocks = group_blocks(m * k, e, tile);
        let fill = |shape: &[u64]| i32_tensor(&device, shape, &vec![-9; shape.iter().product::<u64>() as usize]);
        let (mut counts, mut order, mut inverse, mut table, mut used) =
            (fill(&[e]), fill(&[blocks, tile]), fill(&[m, k]), fill(&[blocks]), fill(&[1]));
        let grouped = i32_tensor(&device, &[1], &[m as i32]);
        routed_group::native_for_device(&device, &default_group_specialization(&device, e, k))
            .unwrap()
            .call(&charges, routed_group::Args {
                first: &first_row(&device),
                rows: &grouped,
                routes: &i32_tensor(&device, &[m, k], &routing.routes),
                counts: &mut counts,
                order: &mut order,
                inverse: &mut inverse,
                blocks: &mut table,
                used: &mut used,
            })
            .unwrap();
        let normalized = bf16_tensor(&device, &[m, h], &routing.normalized);
        // Each choice's output row, at its grouped position.
        let positions = read_i32(&inverse);
        let output = |form| {
            read_bf16(
                &experts
                    .kernel(&device, at(form))
                    .call(&charges, routed_experts_up::Args {
                        first: &first_row(&device),
                        rows: &grouped,
                        normalized: &normalized,
                        order: &order,
                        blocks: &table,
                        used: &used,
                        expert_up: &block.expert_up,
                        expert_down: &block.expert_down,
                        map: &absent_map(&device),
                        step: &absent_step(&device),
                        records_up: &block.expert_up,
                        records_down: &block.expert_down,
                        up_scale: &up_scale,
                        activation: 2,
                    })
                    .unwrap()
                    .value,
            )
        };
        // Each block's tokens at their positions (padding zero), through the
        // packed reference of its expert's up and down weights.
        let (order, table, used) = (read_i32(&order), read_i32(&table), read_i32(&used)[0] as usize);
        let (h, f, tile) = (h as usize, f as usize, tile as usize);
        let up_operand = packed::Packed::of(block.expert_up.element().metal_packing().unwrap(), 1);
        let down_operand = packed::Packed::of(block.expert_down.element().metal_packing().unwrap(), 1);
        let actual = output(&packed);
        let reference = output(&default);
        let mut worst = 0f64;
        for b in 0..used {
            let expert = table[b] as usize;
            let sources = &order[b * tile..(b + 1) * tile];
            let rows_host = &routing.normalized;
            let x = sources
                .iter()
                .flat_map(|&row| (0..h).map(move |c| if row < 0 { 0.0 } else { bf16_round(rows_host[row as usize * h + c]) }))
                .collect::<Vec<_>>();
            let up = packed::reference(
                up_operand,
                h,
                &x,
                &[],
                &block.routed.expert_up[expert * f * h..(expert + 1) * f * h],
                None,
            );
            // Each product as the down GEMM reads it: bf16(relu(bf16(u))^2),
            // monotone in u, over the up sum's interval (padding zero).
            let product = |u: f64| {
                let r = bf16_round(u as f32).max(0.0);
                f64::from(bf16_round(r * r))
            };
            let (products, slack): (Vec<f32>, Vec<f64>) = (0..tile * f)
                .map(|i| {
                    if sources[i / f] < 0 {
                        return (0.0, 0.0);
                    }
                    let (lo, hi) = (product(up.expected[i] - up.budget[i]), product(up.expected[i] + up.budget[i]));
                    if lo == hi { (lo as f32, 0.0) } else { ((0.5 * (lo + hi)) as f32, 0.5 * (hi - lo)) }
                })
                .unzip();
            let down = packed::reference(
                down_operand,
                f,
                &products,
                &slack,
                &block.routed.expert_down[expert * h * f..(expert + 1) * h * f],
                None,
            );
            for (t, &row) in sources.iter().enumerate() {
                if row < 0 {
                    continue;
                }
                let position = b * tile + t;
                let values = actual_positions(&actual, position, h);
                let window = packed::Reference {
                    expected: down.expected[t * h..(t + 1) * h].to_vec(),
                    budget: down.budget[t * h..(t + 1) * h].to_vec(),
                };
                let label = format!("packed routed_experts_up q2k rows {rows} position {position} (block {b}, expert {expert})");
                worst = worst.max(packed::assert_within(&label, &window, &values, |_, expected, budget| {
                    Float::Bf16.rounded_from_exact(budget, expected)
                }));
            }
        }
        eprintln!("packed routed_experts_up q2k rows {rows}: largest difference {worst:.2} of its budget");
        let rms = |values: &mut dyn Iterator<Item = f64>| {
            let (sum, count) = values.fold((0.0, 0usize), |(sum, count), value| (sum + value * value, count + 1));
            (sum / count as f64).sqrt()
        };
        let (reference, actual) = (positioned(&reference, &positions, h), positioned(&actual, &positions, h));
        let scale = rms(&mut reference.iter().copied());
        assert!(scale > 0.0, "rows {rows}: the default form's outputs are not all zero");
        let relative = rms(&mut reference.iter().zip(&actual).map(|(a, b)| a - b)) / scale;
        eprintln!("packed routed_experts_up q2k rows {rows}: relative RMS difference from the default form {relative:.2e}");
    }
}

/// IQ4_NL single-block experts (iq4g32s) at F = 640 (a 128-multiple, not a
/// 256-multiple): Metal admits `DOWN_PACK = 1` on `routed_experts` and the
/// packed down GEMM agrees with its packed reference within budget (covering
/// the partial 128-column tail in `packing_tile_coefficients`).
#[test]
fn metal_down_pack_admitted_and_agrees_with_reference_at_width_640() {
    let Some(device) = seismic_native_tests::every_device()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let block = PackedBlock::new(&device, Experts::Iq4Blocks);
    let (h, e, k, f) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::EXPERTS as u64,
        PackedBlock::CHOICES as u64,
        block.routed.features as u64,
    );
    assert_eq!(f, 640);
    let tile = 32u64;
    let experts = grouped_projection(
        routed_experts::native_entry_with(routed_experts::Elements {
            A: seismic::Element::bf16(),
            EGW: block.expert_gate.element(),
            EUW: block.expert_up.element(),
            EDW: block.expert_down.element(),
        }),
        [[("H", h), ("F", f), ("T", tile)].as_slice(), &block.routed.expansion_statics(), &[("LB", 0)]].concat(),
        0,
    );
    let mappings = grouped_mappings(&device);
    let down_packed = GroupedForm {
        params: vec![("DOWN_TALL", 1), ("DOWN_PACK", 1)],
        quantized: true,
    };
    let at = |form| Grouped {
        mapping: mappings[0].as_slice(),
        form,
    };
    assert!(
        experts.admits(&device, at(&down_packed)),
        "routed_experts admits DOWN_PACK=1 at F={f} over iq4g32s"
    );
    let [gate_scale, up_scale] = block.routed.expansion_scales(&device);
    let rows = 80usize;
    let routing = PackedBlock::routing(rows, 270);
    let m = rows as u64;
    let blocks = group_blocks(m * k, e, tile);
    let fill = |shape: &[u64]| i32_tensor(&device, shape, &vec![-9; shape.iter().product::<u64>() as usize]);
    let (mut counts, mut order, mut inverse, mut table, mut used) =
        (fill(&[e]), fill(&[blocks, tile]), fill(&[m, k]), fill(&[blocks]), fill(&[1]));
    let grouped = i32_tensor(&device, &[1], &[m as i32]);
    routed_group::native_for_device(&device, &default_group_specialization(&device, e, k))
        .unwrap()
        .call(&charges, routed_group::Args {
            first: &first_row(&device),
            rows: &grouped,
            routes: &i32_tensor(&device, &[m, k], &routing.routes),
            counts: &mut counts,
            order: &mut order,
            inverse: &mut inverse,
            blocks: &mut table,
            used: &mut used,
        })
        .unwrap();
    let normalized = bf16_tensor(&device, &[m, h], &routing.normalized);
    let actual = read_bf16(
        &experts
            .kernel(&device, at(&down_packed))
            .call(&charges, routed_experts::Args {
                first: &first_row(&device),
                rows: &grouped,
                normalized: &normalized,
                order: &order,
                blocks: &table,
                used: &used,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                expert_down: &block.expert_down,
                map: &absent_map(&device),
                step: &absent_step(&device),
                records_gate: &block.expert_gate,
                records_up: &block.expert_up,
                records_down: &block.expert_down,
                activation: 0,
                glu_limit: &no_limit(&device),
                gate_scale: &gate_scale,
                up_scale: &up_scale,
            })
            .unwrap()
            .value,
    );
    let (expert, _) = block.routed.expand(&routing, bf16_round);
    let forms = RoutedForms::grouped(&device, block.weights(), 0);
    let (expert_slack, _) = block.routed.expand_bounds(&routing, &forms, Float::Bf16, device.backend());
    let (order, positions, table, used) = (
        read_i32(&order),
        read_i32(&inverse),
        read_i32(&table),
        read_i32(&used)[0] as usize,
    );
    let (h, f, tile) = (h as usize, f as usize, tile as usize);
    let mut flat_at = vec![None; blocks as usize * tile];
    for (flat, &position) in positions.iter().enumerate() {
        flat_at[position as usize] = Some(flat);
    }
    let down_operand = packed::Packed::of(block.expert_down.element().metal_packing().unwrap(), 1);
    let mut worst = 0f64;
    for b in 0..used {
        let expert_id = table[b] as usize;
        let sources = &order[b * tile..(b + 1) * tile];
        let mut products = vec![0.0f32; tile * f];
        let mut slack = vec![0.0f64; tile * f];
        for (t, &row) in sources.iter().enumerate() {
            if row < 0 {
                continue;
            }
            let flat = flat_at[b * tile + t].unwrap();
            products[t * f..(t + 1) * f].copy_from_slice(&expert[flat * f..(flat + 1) * f]);
            slack[t * f..(t + 1) * f].copy_from_slice(&expert_slack[flat * f..(flat + 1) * f]);
        }
        let down = packed::reference(
            down_operand,
            f,
            &products,
            &slack,
            &block.routed.expert_down[expert_id * h * f..(expert_id + 1) * h * f],
            None,
        );
        for (t, &row) in sources.iter().enumerate() {
            if row < 0 {
                continue;
            }
            let position = b * tile + t;
            let values = actual_positions(&actual, position, h);
            let window = packed::Reference {
                expected: down.expected[t * h..(t + 1) * h].to_vec(),
                budget: down.budget[t * h..(t + 1) * h].to_vec(),
            };
            let label = format!("DOWN_PACK F=640 iq4g32s position {position} (block {b}, expert {expert_id})");
            worst = worst.max(packed::assert_within(&label, &window, &values, |_, expected, budget| {
                Float::Bf16.rounded_from_exact(budget, expected)
            }));
        }
    }
    eprintln!("DOWN_PACK F=640 iq4g32s rows {rows}: largest difference {worst:.2} of its budget");
}

/// Output position `position`'s `h` values of a grouped pass's rows.
fn actual_positions(values: &[f32], position: usize, h: usize) -> Vec<f64> {
    values[position * h..(position + 1) * h].iter().map(|&value| f64::from(value)).collect()
}

/// Each choice's output row (`positions`: its flat tile position) of a
/// grouped pass's rows, in choice order.
fn positioned(values: &[f32], positions: &[i32], h: usize) -> Vec<f64> {
    positions.iter().flat_map(|&position| actual_positions(values, position as usize, h)).collect()
}

/// Grid-coded experts in Vulkan's grouped (prefill) expert GEMMs, which stage
/// their weights' tables in shared memory: every expert and shared weight
/// IQ2_XXS (Nemotron 3.5 Bolt), and IQ2_XS beside IQ3_XXS (two tables staged
/// side by side).
#[test]
fn vulkan_grouped_prefill_matches_reference_with_grid_experts() {
    for device in seismic_native_tests::every_device()
        .into_iter()
        .filter(|device| device.backend() == seismic::BackendName::Vulkan)
    {
        for experts in [Experts::Iq2Xxs, Experts::Grid] {
            native_grouped_prefill_matches_reference_on(&device, PackedBlock::new(&device, experts), 0);
        }
    }
}

/// Dense bf16 experts in the grouped (prefill) expert GEMMs.
#[test]
fn native_grouped_prefill_matches_reference_with_bf16_experts() {
    for device in seismic_native_tests::every_device() {
        native_grouped_prefill_matches_reference_on(
            &device,
            PackedBlock::new(&device, Experts::Bf16),
            0,
        );
    }
}

#[test]
fn cuda_grouped_physical_f16_operands_match_reference() {
    let device = seismic_native_tests::every_device()
        .into_iter()
        .find(|device| device.backend() == seismic::BackendName::Cuda);
    if let Some(device) = device {
        for operand in if std::env::var_os("MAGNITUDE_ROUTED_COMPONENT_ONLY").is_some() {
            vec![1]
        } else {
            vec![0, 1]
        } {
            native_grouped_prefill_matches_reference_on(
                &device,
                PackedBlock::new(&device, Experts::Bf16),
                operand,
            );
        }
    }
}

fn native_grouped_prefill_matches_reference_on(
    device: &seismic::Device,
    block: PackedBlock,
    operand: u64,
) {
    let charges = seismic::Charges::tally(device);
    let (h, e, k, f, s) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::EXPERTS as u64,
        PackedBlock::CHOICES as u64,
        block.routed.features as u64,
        PackedBlock::SHARED as u64,
    );
    let tile = 32u64;
    let experts = grouped_projection(
        routed_experts::native_entry_with(routed_experts::Elements {
            A: seismic::Element::bf16(),
            EGW: block.expert_gate.element(),
            EUW: block.expert_up.element(),
            EDW: block.expert_down.element(),
        }),
        [[("H", h), ("F", f), ("T", tile)].as_slice(), &block.routed.expansion_statics(), &[("LB", 0)]].concat(),
        operand,
    );
    let combine = grouped_projection(
        routed_combine::native_entry_with(routed_combine::Elements {
            A: seismic::Element::bf16(),
            SGW: block.shared_gate.element(),
            SUW: block.shared_up.element(),
            SDW: block.shared_down.element(),
        }),
        joined_statics(&[("H", h), ("K", k), ("S", s)]),
        operand,
    );
    // Every mapping in the entries' default forms, then each form at the
    // first mapping on each entry whose declaration admits it, the other
    // entry at its default: a form's launches are its own, and its other
    // GEMM's the default's, which every mapping runs.
    let (mappings, forms, default) = (grouped_mappings(device), grouped_forms(device), GroupedForm::default());
    let at = |form| Grouped {
        mapping: mappings[0].as_slice(),
        form,
    };
    let cases = mappings
        .iter()
        .map(|mapping| {
            [Grouped {
                mapping: mapping.as_slice(),
                form: &default,
            }; 2]
        })
        .chain(forms.iter().filter_map(|[expert_form, combine_form]| {
            let expert_case = at(if experts.admits(device, at(expert_form)) { expert_form } else { &default });
            let combine_case = at(if combine.admits(device, at(combine_form)) { combine_form } else { &default });
            (!expert_case.form.params.is_empty() || !combine_case.form.params.is_empty())
                .then_some([expert_case, combine_case])
        }));
    let cases = mapped::admitted("routed_experts and routed_combine", cases, |[expert_case, combine_case]| {
        experts.admits(device, *expert_case) && combine.admits(device, *combine_case)
    });
    // 80 rows: past 64, where Metal's shared expert takes its forms (to 64
    // rows it runs dense's staged GEMM in every form).
    for rows in if std::env::var_os("MAGNITUDE_ROUTED_COMPONENT_ONLY").is_some() {
        if std::env::var("MAGNITUDE_ROUTED_COMPONENT_ROWS").as_deref() == Ok("40,64") {
            vec![40usize, 64]
        } else {
            vec![16usize]
        }
    } else {
        vec![16usize, 40, 64, 80]
    } {
        let routing = PackedBlock::routing(rows, 90 + rows as u32);
        let m = rows as u64;
        let (expert, shared) = block.routed.expand(&routing, bf16_round);
        let reference = block.routed.output(&routing, &expert, &shared, bf16_round);
        // The combination reads the natives' own products: their bounds
        // carry into it.
        let forms = RoutedForms::grouped(device, block.weights(), operand);
        let products = block.routed.expand_bounds(&routing, &forms, Float::Bf16, device.backend());
        let output_bound = block.routed.output_bounds(
            &routing,
            (&expert, &shared),
            (&products.0, &products.1),
            &forms,
            Float::Bf16,
        );
        let blocks = group_blocks(m * k, e, tile);
        let routes = i32_tensor(&device, &[m, k], &routing.routes);
        let fill = |shape: &[u64]| {
            i32_tensor(
                &device,
                shape,
                &vec![-9; shape.iter().product::<u64>() as usize],
            )
        };
        let mut counts = fill(&[e]);
        let mut order = fill(&[blocks, tile]);
        let mut inverse = fill(&[m, k]);
        let mut table = fill(&[blocks]);
        let mut used = fill(&[1]);
        let grouped = i32_tensor(&device, &[1], &[m as i32]);
        declared::prepared(device, &routed_group::native_entry(), &default_group_specialization(device, e, k))
            .unwrap()
            .call(&charges, routed_group::Args {
                first: &first_row(&device),
                rows: &grouped,
                routes: &routes,
                counts: &mut counts,
                order: &mut order,
                inverse: &mut inverse,
                blocks: &mut table,
                used: &mut used,
            })
            .unwrap();
        let normalized = bf16_tensor(&device, &[m, h], &routing.normalized);
        let [gate_scale, up_scale] = block.routed.expansion_scales(device);
        for &[expert_case, combine_case] in &cases {
            let quantized = expert_case.form.quantized || combine_case.form.quantized;
            let label = format!(
                "rows {rows} mapping {:?} forms {:?} {:?} OPERAND {operand}",
                expert_case.mapping, expert_case.form.params, combine_case.form.params
            );
            let expert_output = experts
                .kernel(device, expert_case)
                .call(&charges, routed_experts::Args {
                    first: &first_row(&device),
                    rows: &grouped,
                    normalized: &normalized,
                    order: &order,
                    blocks: &table,
                    used: &used,
                    expert_gate: &block.expert_gate,
                    expert_up: &block.expert_up,
                    expert_down: &block.expert_down,
                    map: &absent_map(&device),
                    step: &absent_step(&device),
                    records_gate: &block.expert_gate,
                    records_up: &block.expert_up,
                    records_down: &block.expert_down,
                    activation: 0,
                    glu_limit: &no_limit(&device),
                    gate_scale: &gate_scale,
                    up_scale: &up_scale,
                })
                .unwrap()
                .value;
            let combined = combine
                .kernel(device, combine_case)
                .call(&charges, routed_combine::Args {
                    residual: &f32_tensor(&device, &[m, h], &routing.residual),
                    inject: &no_inject(&device, m as u64),
                    expert_output: &expert_output,
                    inverse: &inverse,
                    scores: &f32_tensor(&device, &[m, k], &routing.scores),
                    normalized: &normalized,
                    coefficient: &f32_tensor(&device, &[m], &routing.coefficient),
                    shared_gate: &block.shared_gate,
                    shared_up: &block.shared_up,
                    shared_down: &block.shared_down,
                })
                .unwrap()
                .value;
            let combined_values = read_f32(&combined);
            let worst = combined_values
                .iter()
                .zip(&reference)
                .enumerate()
                .max_by(|(_, (a, b)), (_, (c, d))| (**a - **b).abs().total_cmp(&(**c - **d).abs()))
                .map(|(index, _)| index)
                .unwrap();
            let row = worst / h as usize;
            let column = worst % h as usize;
            let positions = read_i32(&inverse);
            let actual_experts = read_bf16(&expert_output);
            let mut actual_selected = 0.0f32;
            let mut expected_selected = 0.0f32;
            for choice in 0..k as usize {
                let flat = row * k as usize + choice;
                let expert_id = routing.routes[flat] as usize;
                let position = positions[flat] as usize;
                let actual_value = actual_experts[position * h as usize + column];
                let product = &expert[flat * f as usize..(flat + 1) * f as usize];
                let weight_row = (expert_id * h as usize + column) * f as usize;
                let expected_value = bf16_round(dot(
                    product,
                    &block.routed.expert_down[weight_row..weight_row + f as usize],
                ));
                actual_selected = routing.scores[flat].mul_add(actual_value, actual_selected);
                expected_selected = routing.scores[flat].mul_add(expected_value, expected_selected);
            }
            let expected_shared = bf16_round(dot(
                &shared[row * s as usize..(row + 1) * s as usize],
                &block.routed.shared_down[column * s as usize..(column + 1) * s as usize],
            ));
            let actual_shared =
                (combined_values[worst] - routing.residual[worst] - actual_selected)
                    / routing.coefficient[row];
            eprintln!(
                "{label} largest-delta[{worst}] selected native={actual_selected} reference={expected_selected}; shared native={actual_shared} reference={expected_shared}"
            );
            assert_arithmetic(
                &format!("{label} output"),
                &combined_values,
                &reference,
                quantized,
                &output_bound,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The Qwen3.5-35B-A3B routed block end to end at the model's statics. The
// 35B once produced garbage on Metal while the reduced-width tests passed
// and the tuner saw no defects, so this chain runs every native entry at the
// real geometry: route -> group -> experts -> combine (prefill) and
// route -> expand -> output (decode), each stage consuming the previous
// stage's native outputs, against the host reference.

/// A bf16-rounded constant router row for the experts that carry no
/// weights: with positive normalized rows its logit sits ~40 below the
/// weighted experts', so they are never selected.
const UNUSED_ROUTER: f32 = -0.02;

/// The 35B routed block. Only `used` experts hold packed weights (the other
/// experts' bytes are zero and never read), so the host reference decodes
/// just those; the native entries still see all 256 experts.
struct Qwen35b {
    /// Slot in `used` of each expert.
    slot: Vec<Option<usize>>,
    norm: Vec<f32>,
    router: Vec<f32>,
    shared_router: Vec<f32>,
    /// The reference over the used experts, indexed by slot.
    routed: Routed,
    expert_gate: seismic::Tensor,
    expert_up: seismic::Tensor,
    expert_down: seismic::Tensor,
    shared_gate: seismic::Tensor,
    shared_up: seismic::Tensor,
    shared_down: seismic::Tensor,
}

impl Qwen35b {
    const HIDDEN: usize = 2048;

    /// The resident weights, as [`RoutedForms::of`] takes them.
    fn weights(&self) -> [&seismic::Tensor; 6] {
        [&self.expert_gate, &self.expert_up, &self.expert_down, &self.shared_gate, &self.shared_up, &self.shared_down]
    }
    const EXPERTS: usize = 256;
    const CHOICES: usize = 8;
    const FEATURES: usize = 512;
    const SHARED: usize = 512;
    /// The graph's grouped tile rows (`TILE_ROWS`).
    const TILE: usize = 32;
    const USED: usize = 12;
    /// Distinct 32-row tiles per weight pool.
    const POOL: usize = 4;

    fn new(device: &seismic::Device) -> Self {
        let (h, e, f, s) = (Self::HIDDEN, Self::EXPERTS, Self::FEATURES, Self::SHARED);
        let used = (0..Self::USED)
            .map(|i| (37 * i + 5) % e)
            .collect::<Vec<_>>();
        let mut slot = vec![None; e];
        for (index, expert) in used.iter().enumerate() {
            slot[*expert] = Some(index);
        }
        let mut router = vec![bf16_round(UNUSED_ROUTER); e * h];
        for (index, expert) in used.iter().enumerate() {
            let row = pattern(h, 60 + index as u32, 0.06)
                .into_iter()
                .map(bf16_round);
            router[expert * h..(expert + 1) * h]
                .iter_mut()
                .zip(row)
                .for_each(|(to, value)| *to = value);
        }
        let q4k = TilePool::new(device, "q4k", Self::POOL, h, 41);
        let q5k = TilePool::new(device, "q5k", Self::POOL, f, 42);
        let q8_wide = TilePool::new(device, "q8g32s", Self::POOL, h, 43);
        let q8_narrow = TilePool::new(device, "q8g32s", Self::POOL, s, 44);
        let expert_shape = |rows: usize, columns: usize| [e as u64, rows as u64, columns as u64];
        let (expert_gate, gate) = q4k.weight(device, &expert_shape(f, h), &used, 51);
        let (expert_up, up) = q4k.weight(device, &expert_shape(f, h), &used, 52);
        let (expert_down, down) = q5k.weight(device, &expert_shape(h, f), &used, 53);
        let (shared_gate, sgate) = q8_wide.weight(device, &[s as u64, h as u64], &[0], 54);
        let (shared_up, sup) = q8_wide.weight(device, &[s as u64, h as u64], &[0], 55);
        let (shared_down, sdown) = q8_narrow.weight(device, &[h as u64, s as u64], &[0], 56);
        Self {
            routed: Routed {
                hidden: h,
                choices: Self::CHOICES,
                features: f,
                shared: s,
                expert_gate: gate,
                expert_up: up,
                expert_down: down,
                shared_gate: sgate,
                shared_up: sup,
                shared_down: sdown,
                gate_scale: Vec::new(),
                up_scale: Vec::new(),
                nvfp4: Nvfp4Weights::of(["q4k", "q4k", "q5k", "q8g32s", "q8g32s", "q8g32s"]),
            },
            slot,
            norm: pattern(h, 11, 0.5)
                .into_iter()
                .map(|v| bf16_round(v + 1.0))
                .collect(),
            router,
            shared_router: pattern(h, 13, 0.02),
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            shared_up,
            shared_down,
        }
    }

    /// Positive residual rows, so every normalized value is positive.
    fn residual(rows: usize) -> Vec<f32> {
        pattern(rows * Self::HIDDEN, 70 + rows as u32, 1.0)
            .into_iter()
            .map(|v| v + 1.5)
            .collect()
    }

    /// A residual row normalized as the router's norm does, in f64: the one
    /// host reference of `routed_route`'s normalization.
    fn host_normalized(&self, values: &[f32]) -> Vec<f64> {
        let squares = values.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
        let inverse = 1.0 / (squares / Self::HIDDEN as f64 + 1e-6).sqrt();
        values.iter().zip(&self.norm).map(|(value, norm)| f64::from(*value) * inverse * f64::from(*norm)).collect()
    }

    /// The host reference's routing of a normalized row, the one place the
    /// 35B's routing lives on the host: each expert's probability (the
    /// softmax of its logit, an expert without router weights at
    /// `UNUSED_ROUTER` of the row's total) and the experts ranked by it,
    /// ties to the higher expert.
    fn host_ranking(&self, normalized: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let (h, e) = (Self::HIDDEN, Self::EXPERTS);
        let total = normalized.iter().sum::<f64>();
        let logits = (0..e)
            .map(|expert| match self.slot[expert] {
                Some(_) => normalized
                    .iter()
                    .zip(&self.router[expert * h..(expert + 1) * h])
                    .map(|(x, w)| x * f64::from(*w))
                    .sum::<f64>(),
                None => f64::from(bf16_round(UNUSED_ROUTER)) * total,
            })
            .collect::<Vec<_>>();
        let probability = softmax(&logits);
        let ranked = ranked_choices(&probability, e);
        (probability, ranked)
    }

    /// The `CHOICES` experts a row of `values` routes to, ascending, by the
    /// host reference, and the margin of the last choice's probability over
    /// the next expert's, relative to it.
    fn selection(&self, values: &[f32]) -> (Vec<usize>, f64) {
        let k = Self::CHOICES;
        let (probability, ranked) = self.host_ranking(&self.host_normalized(values));
        let margin = (probability[ranked[k - 1]] - probability[ranked[k]]) / probability[ranked[k - 1]];
        let mut chosen = ranked[..k].to_vec();
        chosen.sort_unstable();
        (chosen, margin)
    }

    /// Two residual rows that route to different experts, so a front row has
    /// skips beside its exact row: `residual(2)`'s first row, and the first
    /// of its candidate rows (positive, as `residual`'s) whose host-reference
    /// routing has an absent expert (not the held expert 1) the first lacks,
    /// by a clear margin at the boundary, which no backend's rounding can flip.
    fn diverging_residual_rows(&self) -> Vec<f32> {
        let h = Self::HIDDEN;
        let first = Self::residual(2)[..h].to_vec();
        let (routes, margin) = self.selection(&first);
        assert!(margin > 1e-3, "the first row routes at a near tie");
        let second = (0..256u32)
            .map(|seed| pattern(h, 1000 + seed, 1.0).into_iter().map(|v| v + 1.5).collect::<Vec<f32>>())
            .find(|candidate| {
                let (other, margin) = self.selection(candidate);
                // An absent expert (every one but the held expert 1) only the second row routes to.
                margin > 1e-3 && other.iter().any(|expert| *expert != 1 && !routes.contains(expert))
            })
            .expect("a candidate row routes differently from the first");
        first.into_iter().chain(second).collect()
    }

    /// Native routing of `residual`, checked against the host reference;
    /// every SIMDGROUPS mapping must give the same bits (at the backend's
    /// first split).
    fn route(&self, device: &seismic::Device, residual: &[f32]) -> NativeRouting {
        self.route_mappings(device, residual, &[8, 4, 2])
    }

    fn route_mappings(&self, device: &seismic::Device, residual: &[f32], simdgroups: &[u64]) -> NativeRouting {
        let charges = seismic::Charges::tally(device);
        let (h, e, k) = (
            Self::HIDDEN as u64,
            Self::EXPERTS as u64,
            Self::CHOICES as u64,
        );
        let rows = residual.len() / Self::HIDDEN;
        let m = rows as u64;
        let bf16 = seismic::Element::bf16();
        let residual_tensor = f32_tensor(device, &[m, h], residual);
        let norm = bf16_tensor(device, &[h], &self.norm);
        let router = bf16_tensor(device, &[e, h], &self.router);
        let shared_router = f32_tensor(device, &[h], &self.shared_router);
        let mut outcomes: Vec<NativeRouting> = Vec::new();
        for &simdgroups in route_simdgroups(device, simdgroups) {
            let mut routes = i32_tensor(device, &[m, k], &vec![-7; rows * Self::CHOICES]);
            let mut scores = f32_tensor(device, &[m, k], &vec![-7.0; rows * Self::CHOICES]);
            let outcome = declared::prepared(
                device,
                &routed_route::native_entry_with(routed_route::Elements {
                    NW: bf16,
                    RW: bf16,
                    SRW: seismic::Element::f32(),
                    A: bf16,
                }),
                &specialization_on(
                    device,
                    &route_statics(&[("H", h), ("E", e), ("K", k)], h, 0),
                    &route_mapping(device, simdgroups, route_splits(device)[0]),
                ),
            )
            .unwrap()
            .call(&charges, routed_route::Args {
                residual: &residual_tensor,
                norm: &norm,
                router: &router,
                shared_router: &shared_router,
                routes: &mut routes,
                scores: &mut scores,
                eps: 1e-6,
                normalize: 1,
                expert_scale: &absent(device),
            })
            .unwrap();
            let routing = NativeRouting {
                routes_values: read_i32(&routes),
                scores_values: read_f32(&scores),
                normalized_values: read_bf16(&outcome.r0),
                coefficient_values: read_f32(&outcome.r1),
                routes,
                scores,
                normalized: outcome.r0,
                coefficient: outcome.r1,
            };
            if let Some(first) = outcomes.first() {
                assert!(
                    routing.same_bits(first),
                    "rows {rows}: SIMDGROUPS {simdgroups} changed the routing bits"
                );
            }
            outcomes.push(routing);
        }
        let routing = outcomes.swap_remove(0);
        self.check_route(residual, &routing);
        routing
    }

    fn check_route(&self, residual: &[f32], native: &NativeRouting) {
        let (h, e, k) = (Self::HIDDEN, Self::EXPERTS, Self::CHOICES);
        let rows = residual.len() / h;
        for row in 0..rows {
            let label = format!("rows {rows} row {row}");
            let values = &residual[row * h..(row + 1) * h];
            let normalized = &native.normalized_values[row * h..(row + 1) * h];
            for (index, (expected, actual)) in self.host_normalized(values).into_iter().zip(normalized).enumerate() {
                assert!(
                    (f64::from(*actual) - expected).abs() <= 8e-3 * expected.abs(),
                    "{label} normalized[{index}]: native {actual}, reference {expected}"
                );
            }
            // Probabilities from the native normalized row, in f64.
            let native_row = normalized.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
            let (probability, ranked) = self.host_ranking(&native_row);
            let routes = &native.routes_values[row * k..(row + 1) * k];
            let scores = &native.scores_values[row * k..(row + 1) * k];
            let mut chosen = routes
                .iter()
                .map(|expert| *expert as usize)
                .collect::<Vec<_>>();
            assert!(
                chosen.iter().all(|expert| self.slot[*expert].is_some()),
                "{label}: routes {routes:?}"
            );
            chosen.sort_unstable();
            let mut expected = ranked[..k].to_vec();
            expected.sort_unstable();
            // A different set only where the k-th and the next probability
            // are within rounding of each other (`decision::settled`): each
            // logit is the native's F32 sum of the products in its own order,
            // carried through the softmax's slope, its own and its
            // denominator's.
            let logit_scales = (0..e)
                .map(|expert| match self.slot[expert] {
                    Some(_) => decision::summed(
                        h,
                        normalized
                            .iter()
                            .zip(&self.router[expert * h..(expert + 1) * h])
                            .map(|(x, w)| (f64::from(*x) * f64::from(*w)).abs())
                            .sum::<f64>(),
                    ),
                    None => decision::summed(
                        h,
                        normalized.iter().map(|v| f64::from(*v).abs()).sum::<f64>() * f64::from(bf16_round(UNUSED_ROUTER)).abs(),
                    ),
                })
                .collect::<Vec<_>>();
            let widest = logit_scales.iter().copied().fold(0.0, f64::max);
            let probabilities = probability
                .iter()
                .zip(&logit_scales)
                .map(|(p, scale)| decision::Quantity::carried(*p, p * (scale + widest) + 2.0 * decision::operation(*p)))
                .collect::<Vec<_>>();
            assert!(
                chosen == expected || !decision::settled(&probabilities, k),
                "{label}: routes {routes:?}, reference {:?}",
                &ranked[..k]
            );
            // Ascending-probability slots, scores renormalized over the choice.
            let denominator = routes
                .iter()
                .map(|expert| probability[*expert as usize])
                .sum::<f64>();
            for slot in 0..k {
                let expected = probability[routes[slot] as usize] / denominator;
                assert!(
                    (f64::from(scores[slot]) - expected).abs() <= 1e-3 * expected + 1e-6,
                    "{label} score[{slot}]: native {}, reference {expected}",
                    scores[slot]
                );
                if slot > 0 {
                    assert!(
                        scores[slot - 1] <= scores[slot],
                        "{label}: slot order {scores:?}"
                    );
                }
            }
            let gate = normalized
                .iter()
                .zip(&self.shared_router)
                .map(|(x, w)| f64::from(*x) * f64::from(*w))
                .sum::<f64>();
            let coefficient = 1.0 / (1.0 + (-gate).exp());
            let actual = native.coefficient_values[row];
            assert!(
                (f64::from(actual) - coefficient).abs() <= 1e-4,
                "{label} coefficient: native {actual}, reference {coefficient}"
            );
        }
    }

    /// The host reference inputs of `rows` of a native routing, experts
    /// renamed to their slots.
    fn reference_routing(
        &self,
        residual: &[f32],
        native: &NativeRouting,
        rows: &[usize],
    ) -> Routing {
        let (h, k) = (Self::HIDDEN, Self::CHOICES);
        let gather = |values: &[f32], width: usize| {
            rows.iter()
                .flat_map(|row| values[row * width..(row + 1) * width].iter().copied())
                .collect::<Vec<_>>()
        };
        Routing {
            rows: rows.len(),
            residual: gather(residual, h),
            normalized: gather(&native.normalized_values, h),
            routes: rows
                .iter()
                .flat_map(|row| native.routes_values[row * k..(row + 1) * k].iter())
                .map(|expert| self.slot[*expert as usize].unwrap() as i32)
                .collect(),
            scores: gather(&native.scores_values, k),
            coefficient: gather(&native.coefficient_values, 1),
        }
    }
}

/// One native routing, as device tensors and host values.
struct NativeRouting {
    routes: seismic::Tensor,
    scores: seismic::Tensor,
    normalized: seismic::Tensor,
    coefficient: seismic::Tensor,
    routes_values: Vec<i32>,
    scores_values: Vec<f32>,
    normalized_values: Vec<f32>,
    coefficient_values: Vec<f32>,
}

impl NativeRouting {
    fn same_bits(&self, other: &Self) -> bool {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        self.routes_values == other.routes_values
            && bits(&self.scores_values) == bits(&other.scores_values)
            && bits(&self.normalized_values) == bits(&other.normalized_values)
            && bits(&self.coefficient_values) == bits(&other.coefficient_values)
    }
}

/// Every resident row layout stores 32 consecutive rows in 32 row strides
/// of bytes of their own (`mma16` permutes codes only within a 16-row tile,
/// `rows32` interleaves planes only within a 32-row tile).
const LAYOUT_TILE_ROWS: usize = 32;

/// A few distinct 32-row tiles of `representation` over `columns`, placed
/// and decoded once. Decoding through the reference recipe is slow, so the
/// 35B weights are assembled from pooled tiles instead of decoded whole.
struct TilePool {
    element: seismic::Element,
    tiles: usize,
    columns: usize,
    bytes: Vec<u8>,
    values: Vec<f32>,
}

impl TilePool {
    fn new(
        device: &seismic::Device,
        representation: &str,
        tiles: usize,
        columns: usize,
        seed: u32,
    ) -> Self {
        let shape = [(tiles * LAYOUT_TILE_ROWS) as u64, columns as u64];
        let (element, bytes, values) = packed_planes(device, representation, &shape, seed);
        Self {
            element,
            tiles,
            columns,
            bytes,
            values,
        }
    }

    /// A weight of `shape` ([matrices, rows, columns] or [rows, columns],
    /// one matrix) whose `filled` matrices are sequences of pool tiles picked
    /// pseudo-randomly per (matrix, tile), and whose other matrices are zero
    /// bytes: the tensor and the filled matrices' decoded values in `filled`
    /// order.
    fn weight(
        &self,
        device: &seismic::Device,
        shape: &[u64],
        filled: &[usize],
        seed: u32,
    ) -> (seismic::Tensor, Vec<f32>) {
        let (matrices, rows) = match *shape {
            [matrices, rows, columns] if columns as usize == self.columns => {
                (matrices as usize, rows as usize)
            }
            [rows, columns] if columns as usize == self.columns => (1, rows as usize),
            _ => panic!("pool of {} columns cannot fill {shape:?}", self.columns),
        };
        let tile_bytes = self.bytes.len() / self.tiles;
        let tile_values = LAYOUT_TILE_ROWS * self.columns;
        let tiles = rows / LAYOUT_TILE_ROWS;
        let picks = pattern(filled.len() * tiles, seed, 1.0);
        let mut bytes = vec![0u8; matrices * tiles * tile_bytes];
        let mut values = Vec::with_capacity(filled.len() * rows * self.columns);
        for (index, matrix) in filled.iter().enumerate() {
            for tile in 0..tiles {
                let pick = (((picks[index * tiles + tile] + 1.0) * 0.5 * self.tiles as f32)
                    as usize)
                    .min(self.tiles - 1);
                let at = (matrix * tiles + tile) * tile_bytes;
                bytes[at..at + tile_bytes]
                    .copy_from_slice(&self.bytes[pick * tile_bytes..(pick + 1) * tile_bytes]);
                values
                    .extend_from_slice(&self.values[pick * tile_values..(pick + 1) * tile_values]);
            }
        }
        (
            seismic::Tensor::from_host(&seismic::Charges::tally(device), self.element, shape, &bytes).unwrap(),
            values,
        )
    }
}

#[test]
fn native_35b_geometry_chain_matches_reference() {
    for device in seismic_native_tests::every_device() {
        native_35b_geometry_chain_matches_reference_on(&device);
    }
}

fn native_35b_geometry_chain_matches_reference_on(device: &seismic::Device) {
    let charges = seismic::Charges::tally(device);
    let block = Qwen35b::new(&device);
    let (h, e, k, f, s, t) = (
        Qwen35b::HIDDEN as u64,
        Qwen35b::EXPERTS as u64,
        Qwen35b::CHOICES as u64,
        Qwen35b::FEATURES as u64,
        Qwen35b::SHARED as u64,
        Qwen35b::TILE as u64,
    );
    let bf16 = seismic::Element::bf16();

    // Decode classes: route -> expand -> output.
    for rows in [1usize, 8] {
        let residual = Qwen35b::residual(rows);
        let routing = block.route(&device, &residual);
        let reference_routing =
            block.reference_routing(&residual, &routing, &(0..rows).collect::<Vec<_>>());
        let (expert, shared) = block.routed.expand(&reference_routing, bf16_round);
        let reference = block
            .routed
            .output(&reference_routing, &expert, &shared, bf16_round);
        // The output reads the native's own products: their bounds carry
        // into it.
        let forms = RoutedForms::decode(device, block.weights(), rows);
        let (expert_bound, shared_bound) = block.routed.expand_bounds(&reference_routing, &forms, Float::Bf16, device.backend());
        let output_bound = block.routed.output_bounds(
            &reference_routing,
            (&expert, &shared),
            (&expert_bound, &shared_bound),
            &forms,
            Float::Bf16,
        );
        let residual_tensor = f32_tensor(&device, &[rows as u64, h], &residual);
        let [gate_scale, up_scale] = block.routed.expansion_scales(device);
        for (index, mapping) in decode_mappings(&device).into_iter().enumerate() {
            let output_mapping =
                gathered::<routed_output::Entry>(device, index, &output_mapping(device, index, &mapping));
            let mapping = gathered::<routed_expand::Entry>(device, index, &mapping);
            let label =
                format!("35B decode rows {rows} mapping {mapping:?} output {output_mapping:?}");
            let statics = [("H", h), ("K", k), ("F", f), ("S", s)];
            let mut specialization = decode_specialization::<routed_expand::Entry>(
                device,
                &[statics.as_slice(), &block.routed.expansion_statics()].concat(),
                &mapping,
            );
            let mut output_specialization =
                decode_specialization::<routed_output::Entry>(device, &statics, &output_mapping);
            if is_cpu(device) {
                specialization = specialization.with_param("INT8", 0);
                output_specialization = output_specialization.with_param("INT8", 0);
            }
            let expanded = declared::prepared(
                &device,
                &routed_expand::native_entry_with(routed_expand::Elements {
                    A: bf16,
                    EGW: element(&device, "q4k"),
                    EUW: element(&device, "q4k"),
                    SGW: element(&device, "q8g32s"),
                    SUW: element(&device, "q8g32s"),
                }),
                &specialization,
            )
            .unwrap()
            .call(&charges, routed_expand::Args {
                normalized: &routing.normalized,
                routes: &routing.routes,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
                gate_scale: &gate_scale,
                up_scale: &up_scale,
            })
            .unwrap();
            bound::assert_within(
                &format!("{label} expert product"),
                &read_bf16(&expanded.r0),
                &expert,
                &expert_bound,
            );
            bound::assert_within(
                &format!("{label} shared product"),
                &read_bf16(&expanded.r1),
                &shared,
                &shared_bound,
            );
            let output = declared::prepared(
                &device,
                &routed_output::native_entry_with(routed_output::Elements {
                    A: bf16,
                    EDW: element(&device, "q5k"),
                    SDW: element(&device, "q8g32s"),
                    NW: seismic::Element::f32(),
                }),
                &output_specialization,
            )
            .unwrap()
            .call(&charges, routed_output::Args {
                residual: &residual_tensor,
                inject: &no_inject(&device, residual_tensor.extents()[0]),
                expert_product: &expanded.r0,
                shared_product: &expanded.r1,
                routes: &routing.routes,
                sources: &no_sources(&device, routing.routes.extents()[0], routing.routes.extents()[1]),
                scores: &routing.scores,
                coefficient: &routing.coefficient,
                expert_down: &block.expert_down,
                tier_down: &no_tier(&block.expert_down),
                shared_down: &block.shared_down,
                operand_norm: &absent(&device),
                operand_epsilon: 0.0,
            })
            .unwrap()
            .value;
            bound::assert_within(&format!("{label} output"), &read_f32(&output), &reference, &output_bound);
        }
    }

    // Grouped classes: route -> group -> experts -> combine. 16 rows leave
    // most expert tiles partly padding; at 512 rows every used expert fills
    // many tiles. The reference covers a sample of the 512 rows.
    for (rows, sample) in [(16usize, 1usize), (512, 43)] {
        let residual = Qwen35b::residual(rows);
        let m = rows as u64;
        let routing = block.route(&device, &residual);
        let checked = (0..rows).step_by(sample).collect::<Vec<_>>();
        let reference_routing = block.reference_routing(&residual, &routing, &checked);
        let (expert, shared) = block.routed.expand(&reference_routing, bf16_round);
        let reference = block
            .routed
            .output(&reference_routing, &expert, &shared, bf16_round);
        // The combination reads the natives' own products (CUDA's at
        // `OPERAND` 0): their bounds carry into it.
        let forms = RoutedForms::grouped(device, block.weights(), 0);
        let products = block.routed.expand_bounds(&reference_routing, &forms, Float::Bf16, device.backend());
        let output_bound = block.routed.output_bounds(
            &reference_routing,
            (&expert, &shared),
            (&products.0, &products.1),
            &forms,
            Float::Bf16,
        );
        let blocks = group_blocks(m * k, e, t);
        let fill = |shape: &[u64]| {
            i32_tensor(
                &device,
                shape,
                &vec![-9; shape.iter().product::<u64>() as usize],
            )
        };
        let mut counts = fill(&[e]);
        let mut order = fill(&[blocks, t]);
        let mut inverse = fill(&[m, k]);
        let mut table = fill(&[blocks]);
        let mut used = fill(&[1]);
        let grouped = i32_tensor(&device, &[1], &[m as i32]);
        declared::prepared(device, &routed_group::native_entry(), &default_group_specialization(device, e, k))
            .unwrap()
            .call(&charges, routed_group::Args {
                first: &first_row(&device),
                rows: &grouped,
                routes: &routing.routes,
                counts: &mut counts,
                order: &mut order,
                inverse: &mut inverse,
                blocks: &mut table,
                used: &mut used,
            })
            .unwrap();
        let counts = read_i32(&counts);
        for (expert, count) in counts.iter().enumerate() {
            let expected = routing
                .routes_values
                .iter()
                .filter(|route| **route as usize == expert)
                .count();
            assert_eq!(
                *count as usize, expected,
                "35B rows {rows}: count of expert {expert}"
            );
        }
        let residual_tensor = f32_tensor(&device, &[m, h], &residual);
        let [gate_scale, up_scale] = block.routed.expansion_scales(device);
        let pass = grouped_projection(
            routed_experts::native_entry_with(routed_experts::Elements {
                A: bf16,
                EGW: element(&device, "q4k"),
                EUW: element(&device, "q4k"),
                EDW: element(&device, "q5k"),
            }),
            [[("H", h), ("F", f), ("T", t)].as_slice(), &block.routed.expansion_statics(), &[("LB", 0)]].concat(),
            0,
        );
        let form = GroupedForm::default();
        for mapping in mapped::admitted("routed_experts", grouped_mappings(&device), |mapping| pass.admits(device, Grouped { mapping, form: &form })) {
            let label = format!("35B grouped rows {rows} mapping {mapping:?}");
            // The default forms, CUDA's operand path at its first value: the
            // staged one is the grouped test's (`OPERAND` 1).
            let configuration = Grouped { mapping: &mapping, form: &form };
            let experts = pass
                .kernel(device, configuration)
            .call(&charges, routed_experts::Args {
                first: &first_row(&device),
                rows: &grouped,
                normalized: &routing.normalized,
                order: &order,
                blocks: &table,
                used: &used,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                expert_down: &block.expert_down,
                map: &absent_map(&device),
                step: &absent_step(&device),
                records_gate: &block.expert_gate,
                records_up: &block.expert_up,
                records_down: &block.expert_down,
                activation: 0,
                glu_limit: &no_limit(&device),
                gate_scale: &gate_scale,
                up_scale: &up_scale,
            })
            .unwrap()
            .value;
            let combined = grouped_projection(
                routed_combine::native_entry_with(routed_combine::Elements {
                    A: bf16,
                    SGW: element(&device, "q8g32s"),
                    SUW: element(&device, "q8g32s"),
                    SDW: element(&device, "q8g32s"),
                }),
                joined_statics(&[("H", h), ("K", k), ("S", s)]),
                0,
            )
            .kernel(device, configuration)
            .call(&charges, routed_combine::Args {
                residual: &residual_tensor,
                inject: &no_inject(&device, residual_tensor.extents()[0]),
                expert_output: &experts,
                inverse: &inverse,
                scores: &routing.scores,
                normalized: &routing.normalized,
                coefficient: &routing.coefficient,
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
                shared_down: &block.shared_down,
            })
            .unwrap()
            .value;
            let combined = read_f32(&combined);
            let sampled = checked
                .iter()
                .flat_map(|row| {
                    combined[row * Qwen35b::HIDDEN..(row + 1) * Qwen35b::HIDDEN]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            bound::assert_within(&format!("{label} output"), &sampled, &reference, &output_bound);
        }
    }
}

/// The decode form's mappings on `device`'s backend, one set for each value
/// of the shared expert's arithmetic parameter (Metal LANES, CUDA KSPLIT):
/// `routed_expand`'s mapping at that value (the reference), every
/// `routed_route_shared` mapping at it, and `routed_gate_up` mappings at it.
struct SharedRouteMappings {
    expand: Vec<(&'static str, u64)>,
    routes: Vec<Vec<(&'static str, u64)>>,
    choices: Vec<Vec<(&'static str, u64)>>,
}

fn shared_route_mappings(device: &seismic::Device) -> Vec<SharedRouteMappings> {
    match device.backend() {
        seismic::BackendName::Metal => [32u64, 16]
            .map(|lanes| SharedRouteMappings {
                expand: vec![("SIMDGROUPS", 4), ("ROWS", 2), ("LANES", lanes)],
                routes: [8u64, 16, 4, 2]
                    .into_iter()
                    .flat_map(|simdgroups| {
                        [1u64, 2, 4].map(|rows| {
                            vec![("SIMDGROUPS", simdgroups), ("ROWS", rows), ("LANES", lanes)]
                        })
                    })
                    .collect(),
                choices: vec![
                    vec![("SIMDGROUPS", 4), ("ROWS", 2), ("LANES", lanes)],
                    vec![("SIMDGROUPS", 16), ("ROWS", 1), ("LANES", lanes)],
                ],
            })
            .into(),
        // A shared block of SIMDGROUPS warps forms one tile group of KSPLIT.
        seismic::BackendName::Cuda => [1u64, 2, 4]
            .map(|split| SharedRouteMappings {
                expand: vec![("TPW", 1), ("KSPLIT", split)],
                routes: [1u64, 2]
                    .map(|tpw| vec![("TPW", tpw), ("KSPLIT", split)])
                    .into(),
                choices: vec![
                    vec![("TPW", 1), ("KSPLIT", split)],
                    vec![("TPW", 2), ("KSPLIT", split)],
                ],
            })
            .into(),
        // Vulkan's router and shared projection use their resident recipes;
        // these default mappings differ only in work distribution.
        seismic::BackendName::Vulkan => vec![SharedRouteMappings {
            expand: vec![("SIMDGROUPS", 2), ("ROWS", 1)],
            routes: vec![vec![("SIMDGROUPS", 8), ("ROWS", 1), ("LANES", 32)]],
            choices: vec![vec![("SIMDGROUPS", 8), ("ROWS", 1)]],
        }],
        backend => panic!("{backend:?} declares no `routed_route_shared`"),
    }
}

/// The native device when its backend declares `routed_route_shared`.
fn shared_route_device() -> Option<seismic::Device> {
    native_device().filter(|device| routed_route_shared::native_implementation(device).unwrap().is_some())
}

/// The shared-route decode form over the 35B geometry, on every backend that
/// declares it: `routed_route_shared` and `routed_gate_up` (SiLU) keep every
/// bit of `routed_route` and `routed_expand` for every decode row count and
/// mapping: the routes, scores, normalized rows and coefficient; the shared
/// product at equal arithmetic parameters; the choices' product.
#[test]
fn native_35b_shared_route_keeps_every_bit() {
    let Some(device) = shared_route_device() else {
        return;
    };
    native_35b_shared_route_cases(&device, &(1..=8).collect::<Vec<_>>(), &shared_route_mappings(&device), &[8, 4, 2]);
}

fn native_35b_shared_route_cases(
    device: &seismic::Device,
    row_counts: &[usize],
    mappings: &[SharedRouteMappings],
    reference_simdgroups: &[u64],
) {
    let charges = seismic::Charges::tally(device);
    let block = Qwen35b::new(&device);
    let (h, e, k, f, s) = (
        Qwen35b::HIDDEN as u64,
        Qwen35b::EXPERTS as u64,
        Qwen35b::CHOICES as u64,
        Qwen35b::FEATURES as u64,
        Qwen35b::SHARED as u64,
    );
    let bf16 = seismic::Element::bf16();
    let (q4k, q8) = (element(&device, "q4k"), element(&device, "q8g32s"));
    let norm = bf16_tensor(&device, &[h], &block.norm);
    let router = bf16_tensor(&device, &[e, h], &block.router);
    let shared_router = f32_tensor(&device, &[h], &block.shared_router);
    let unscaled = absent(&device);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for &rows in row_counts {
        let m = rows as u64;
        let residual = Qwen35b::residual(rows);
        let routing = block.route_mappings(&device, &residual, reference_simdgroups);
        let residual_tensor = f32_tensor(&device, &[m, h], &residual);
        for mappings in mappings {
            let expanded = declared::prepared(
                &device,
                &routed_expand::native_entry_with(routed_expand::Elements {
                    A: bf16,
                    EGW: q4k,
                    EUW: q4k,
                    SGW: q8,
                    SUW: q8,
                }),
                &decode_specialization::<routed_expand::Entry>(
                    &device,
                    &[("H", h), ("K", k), ("F", f), ("S", s)],
                    &mappings.expand,
                ),
            )
            .unwrap()
            .call(&charges, routed_expand::Args {
                normalized: &routing.normalized,
                routes: &routing.routes,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
                gate_scale: &unscaled,
                up_scale: &unscaled,
            })
            .unwrap();
            let (expert_product, shared_product) =
                (read_bf16(&expanded.r0), read_bf16(&expanded.r1));
            for mapping in &mappings.routes {
                let label = format!("rows {rows} {mapping:?}");
                let mut routes = i32_tensor(&device, &[m, k], &vec![-7; rows * Qwen35b::CHOICES]);
                let mut scores =
                    f32_tensor(&device, &[m, k], &vec![-7.0; rows * Qwen35b::CHOICES]);
                let shared = declared::prepared(
                    &device,
                    &routed_route_shared::native_entry_with(routed_route_shared::Elements {
                        NW: bf16,
                        RW: bf16,
                        SRW: seismic::Element::f32(),
                        A: bf16,
                        SGW: q8,
                        SUW: q8,
                    }),
                    &specialize(
                        &route_statics(&[("H", h), ("E", e), ("K", k), ("S", s)], h, 0),
                        mapping,
                    ),
                )
                .unwrap()
                .call(&charges, routed_route_shared::Args {
                    residual: &residual_tensor,
                    norm: &norm,
                    router: &router,
                    shared_router: &shared_router,
                    shared_gate: &block.shared_gate,
                    shared_up: &block.shared_up,
                    routes: &mut routes,
                    scores: &mut scores,
                    eps: 1e-6,
                    normalize: 1,
                    expert_scale: &unscaled,
                })
                .unwrap();
                assert_eq!(read_i32(&routes), routing.routes_values, "{label} routes");
                assert_eq!(
                    bits(&read_f32(&scores)),
                    bits(&routing.scores_values),
                    "{label} scores"
                );
                assert_eq!(
                    bits(&read_bf16(&shared.r0)),
                    bits(&routing.normalized_values),
                    "{label} normalized"
                );
                assert_eq!(
                    bits(&read_f32(&shared.r1)),
                    bits(&routing.coefficient_values),
                    "{label} coefficient"
                );
                assert_eq!(
                    bits(&read_bf16(&shared.r2)),
                    bits(&shared_product),
                    "{label} shared product"
                );
            }
            for mapping in &mappings.choices {
                let choices = declared::prepared(
                    &device,
                    &routed_gate_up::native_entry_with(routed_gate_up::Elements {
                        A: bf16,
                        EGW: q4k,
                        EUW: q4k,
                    }),
                    &decode_specialization::<routed_gate_up::Entry>(
                        &device,
                        &[("H", h), ("K", k), ("F", f)],
                        mapping,
                    ),
                )
                .unwrap()
                .call(&charges, routed_gate_up::Args {
                    normalized: &routing.normalized,
                    routes: &routing.routes,
                    sources: &no_sources(&device, routing.routes.extents()[0], routing.routes.extents()[1]),
                    expert_gate: &block.expert_gate,
                    expert_up: &block.expert_up,
                    tier_gate: &no_tier(&block.expert_gate),
                    tier_up: &no_tier(&block.expert_up),
                    activation: 0,
                    glu_limit: &no_limit(&device),
                    gate_scale: &unscaled,
                    up_scale: &unscaled,
                })
                .unwrap()
                .value;
                assert_eq!(
                    bits(&read_bf16(&choices)),
                    bits(&expert_product),
                    "rows {rows} gate_up {mapping:?} choices' product"
                );
            }
        }
    }
}

/// The grouped capacity the executor sizes (`grouped_blocks`): tiles for
/// `m·k` choices plus one partial tile per receiving expert, of which there
/// are at most `min(e, m·k)`. Every row class from 9 rows (the first grouped
/// class) to 32 rows: below 32 rows the bound is smaller than the one padded
/// for all 256 experts, and the combined rows must keep every bit.
#[test]
fn native_35b_grouped_capacity_keeps_every_bit() {
    for device in seismic_native_tests::every_device() {
        native_35b_grouped_capacity_keeps_every_bit_on(&device);
    }
}

fn native_35b_grouped_capacity_keeps_every_bit_on(device: &seismic::Device) {
    let charges = seismic::Charges::tally(device);
    let block = Qwen35b::new(device);
    let (h, e, k, f, s, t) = (
        Qwen35b::HIDDEN as u64,
        Qwen35b::EXPERTS as u64,
        Qwen35b::CHOICES as u64,
        Qwen35b::FEATURES as u64,
        Qwen35b::SHARED as u64,
        Qwen35b::TILE as u64,
    );
    let bf16 = seismic::Element::bf16();
    for rows in 9usize..=32 {
        let residual = Qwen35b::residual(rows);
        let m = rows as u64;
        let routing = block.route(device, &residual);
        let residual_tensor = f32_tensor(device, &[m, h], &residual);
        let padded = group_blocks(m * k, e, t);
        let bounded = group_blocks(m * k, e.min(m * k), t);
        let pass = grouped_projection(
            routed_experts::native_entry_with(routed_experts::Elements {
                A: bf16,
                EGW: element(device, "q4k"),
                EUW: element(device, "q4k"),
                EDW: element(device, "q5k"),
            }),
            vec![("H", h), ("F", f), ("T", t), ("GS", 0), ("US", 0), ("LB", 0)],
            0,
        );
        let form = GroupedForm::default();
        for mapping in mapped::admitted("routed_experts", grouped_mappings(device), |mapping| pass.admits(device, Grouped { mapping, form: &form })) {
            // The default forms, CUDA's operand path at its first value, as
            // the 35B chain's.
            let configuration = Grouped { mapping: &mapping, form: &form };
            let combined = |blocks: u64| {
                let fill = |shape: &[u64]| {
                    i32_tensor(
                        device,
                        shape,
                        &vec![-9; shape.iter().product::<u64>() as usize],
                    )
                };
                let mut counts = fill(&[e]);
                let mut order = fill(&[blocks, t]);
                let mut inverse = fill(&[m, k]);
                let mut table = fill(&[blocks]);
                let mut used = fill(&[1]);
                let grouped = i32_tensor(device, &[1], &[m as i32]);
                declared::prepared(
                    device,
                    &routed_group::native_entry(),
                    &default_group_specialization(device, e, k),
                )
                .unwrap()
                .call(&charges, routed_group::Args {
                    first: &first_row(&device),
                    rows: &grouped,
                    routes: &routing.routes,
                    counts: &mut counts,
                    order: &mut order,
                    inverse: &mut inverse,
                    blocks: &mut table,
                    used: &mut used,
                })
                .unwrap();
                let experts = pass
                    .kernel(device, configuration)
                .call(&charges, routed_experts::Args {
                    first: &first_row(&device),
                    rows: &grouped,
                    normalized: &routing.normalized,
                    order: &order,
                    blocks: &table,
                    used: &used,
                    expert_gate: &block.expert_gate,
                    expert_up: &block.expert_up,
                    expert_down: &block.expert_down,
                    map: &absent_map(device),
                    step: &absent_step(device),
                    records_gate: &block.expert_gate,
                    records_up: &block.expert_up,
                    records_down: &block.expert_down,
                    activation: 0,
                    glu_limit: &no_limit(device),
                    gate_scale: &absent(device),
                    up_scale: &absent(device),
                })
                .unwrap()
                .value;
                let combined = grouped_projection(
                    routed_combine::native_entry_with(routed_combine::Elements {
                        A: bf16,
                        SGW: element(device, "q8g32s"),
                        SUW: element(device, "q8g32s"),
                        SDW: element(device, "q8g32s"),
                    }),
                    joined_statics(&[("H", h), ("K", k), ("S", s)]),
                    0,
                )
                .kernel(device, configuration)
                .call(&charges, routed_combine::Args {
                    residual: &residual_tensor,
                    inject: &no_inject(&device, residual_tensor.extents()[0]),
                    expert_output: &experts,
                    inverse: &inverse,
                    scores: &routing.scores,
                    normalized: &routing.normalized,
                    coefficient: &routing.coefficient,
                    shared_gate: &block.shared_gate,
                    shared_up: &block.shared_up,
                    shared_down: &block.shared_down,
                })
                .unwrap()
                .value;
                read_f32(&combined)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                combined(bounded),
                combined(padded),
                "35B grouped rows {rows} mapping {mapping:?}: capacity {bounded} vs {padded}"
            );
        }
    }
}

/// The inject port of a single residual stream: no columns.
fn no_inject(device: &seismic::Device, rows: u64) -> seismic::Tensor {
    seismic::Tensor::zeros(&seismic::Charges::tally(device), seismic::Element::f32(), &[rows, 0]).unwrap()
}


// ---------------------------------------------------------------------------
// Expert streaming (routed.seismic's streaming section), on each backend
// implementing it.

// ---------------------------------------------------------------------------
// Expert streaming (routed.seismic's streaming section), on each backend
// implementing it.

/// The sequence the tests' steps publish under.
const STREAMING_SEQUENCE: i32 = 7;

/// The routing records the tests' layers keep, one a step in flight
/// (`FLIGHT`): neither the sequence nor a power of two, so the record a
/// step writes is its sequence's residue.
const FLIGHT: u64 = 3;

/// Store `value` as the u64 at `words[at]`, its low word first
/// (streaming_words.seismic's steps).
fn set_pair(words: &mut [i32], at: u64, value: u64) {
    words[at as usize] = value as u32 as i32;
    words[at as usize + 1] = (value >> 32) as u32 as i32;
}

/// The file table of `choices` choices of [`FLIGHT`] slots holding step
/// `q`'s request as a resolve publishes it (the count, then `requested`
/// each choice's word: an absent choice's expert, a held one's -1 -
/// expert), and where `answer` is some, its answer: each of `records`'
/// (choice, record word), then the slot's answer word.
fn file_words(choices: u64, q: u64, requested: &[i32], records: &[(u64, i32)], answer: Option<u64>) -> (SlotTable, Vec<i32>) {
    let table = SlotTable::file(FLIGHT, choices);
    let mut words = vec![0; table.words() as usize];
    let slot = table.lane(q, 0);
    words[(slot + constants::FILE_COUNT) as usize] = requested.len() as i32;
    for (choice, &word) in requested.iter().enumerate() {
        words[(slot + constants::FILE_ENTRY(choice as u64) + constants::FILE_CHOICE) as usize] = word;
    }
    set_pair(&mut words, slot + constants::SLOT_REQUEST, q);
    for &(choice, record) in records {
        words[(slot + constants::FILE_ENTRY(choice) + constants::FILE_RECORD) as usize] = record;
    }
    if let Some(answer) = answer {
        set_pair(&mut words, slot + constants::SLOT_ANSWER, answer);
    }
    (table, words)
}

/// The native device when its backend implements the streaming entries
/// (Metal and CUDA implement every one: one coverage, so `routed_step`
/// stands for them).
fn streaming_device() -> Option<seismic::Device> {
    let device = native_device()?;
    let implemented = routed_step::native_implementation(&device).unwrap().is_some();
    if seismic::testing::requested_backend().is_some() {
        assert!(implemented, "the requested backend must implement expert streaming");
    }
    implemented.then_some(device)
}

/// A host-shared i32 tensor holding `values` (`Tensor::shared`).
fn shared_i32(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    shared_bytes(device, seismic::Element::i32(), shape, &bytes)
}

/// A host-shared tensor of `element` holding `bytes` (`Tensor::shared`).
fn shared_bytes(device: &seismic::Device, element: seismic::Element, shape: &[u64], bytes: &[u8]) -> seismic::Tensor {
    let tensor = seismic::Tensor::shared(&seismic::Charges::tally(device), element, shape).unwrap();
    // SAFETY: no device work uses the tensor yet.
    unsafe { tensor.write_shared(0, bytes).unwrap() };
    tensor
}

fn host_words(device: &seismic::Device, shape: &[u64]) -> seismic::Tensor {
    i32_tensor(device, shape, &vec![0; shape.iter().product::<u64>() as usize])
}

/// Waits (at most ten seconds) until word `index` of a shared tensor holds
/// `value`.
fn await_shared_word(tensor: &seismic::Tensor, index: u64, value: i32) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while tensor.load_acquire(4 * index).unwrap() as i32 != value {
        assert!(
            std::time::Instant::now() < deadline,
            "word {index} never held {value}"
        );
        std::hint::spin_loop();
    }
}

/// A slot table's pulse as a live host's exchange beats it (every lane's
/// SLOT_PULSE every PULSE_MS), until it drops: every table a test's device
/// waits on beats, as a load's do, so no wait here outlives its pulse.
struct Pulse {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn pulse(tensor: &seismic::Tensor, table: SlotTable) -> Pulse {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (beaten, stopped) = (tensor.clone(), stop.clone());
    let lanes = (0..table.slots())
        .flat_map(|slot| (0..table.lanes()).map(move |lane| table.lane(slot, lane) + constants::SLOT_PULSE))
        .collect::<Vec<_>>();
    let beat = move |beat: u32| {
        for &word in &lanes {
            beaten.store_release(4 * word, beat).unwrap();
        }
    };
    beat(1);
    let thread = std::thread::spawn(move || {
        for count in 2u32.. {
            if stopped.load(std::sync::atomic::Ordering::Acquire) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(constants::PULSE_MS));
            beat(count);
        }
    });
    Pulse { stop, thread: Some(thread) }
}

/// A table's words with every lane's pulse zero, as a table is made: what
/// a test compares, its beats aside.
fn unpulsed(mut words: Vec<i32>, table: SlotTable) -> Vec<i32> {
    for slot in 0..table.slots() {
        for lane in 0..table.lanes() {
            words[(table.lane(slot, lane) + constants::SLOT_PULSE) as usize] = 0;
        }
    }
    words
}

impl Drop for Pulse {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// One routed tensor of a model's experts as its GGUF stores it: the
/// stored (GGUF) and resident elements and the [rows, columns] of a matrix.
#[derive(Clone, Copy)]
struct StreamedKind {
    source: seismic::Element,
    resident: seismic::Element,
    matrix: [u64; 2],
}

impl StreamedKind {
    /// Gate, up and down.
    fn qwen35b(device: &seismic::Device) -> [Self; 3] {
        Self::qwen35b_in(resident_layout(device))
    }

    /// Gate, up and down, resident in `layout`.
    fn qwen35b_in(layout: registry::Layout) -> [Self; 3] {
        let (h, f) = (Qwen35b::HIDDEN as u64, Qwen35b::FEATURES as u64);
        [
            ("gguf_q4_k", "q4k", [f, h]),
            ("gguf_q4_k", "q4k", [f, h]),
            ("gguf_q5_k", "q5k", [h, f]),
        ]
        .map(|(source, resident, matrix)| Self {
            source: seismic::Element::named(source).unwrap(),
            resident: element_in(resident, layout),
            matrix,
        })
    }

    /// DeepSeek V4.1's MXFP4 gate (and up, the same form) and down, at its
    /// shape: gate [F 2304, H 5120], rows of 160 packets of 17 bytes (2720),
    /// and down [H 5120, F 2304], rows of 72 (1224). A landing converts an
    /// entry's three tensors together, so their F and H are one record's.
    fn v41_mxfp4(device: &seismic::Device) -> [Self; 2] {
        let layout = resident_layout(device);
        let (h, f) = (160 * 32, 72 * 32);
        [[f, h], [h, f]].map(|matrix| Self {
            source: seismic::Element::named("gguf_mxfp4").unwrap(),
            resident: element_in("mxfp4g32", layout),
            matrix,
        })
    }

    /// GLM-5.3-Flash's FP8 gate (and up, the same form) and down under a
    /// grid of block scales, a planar source (`source`'s codes, then its
    /// grid; the tile is its name's) resident as `resident`, at a shape of
    /// whole tiles: gate [F 256, H 512] and down [H 512, F 256].
    fn glm_fp8(device: &seismic::Device, source: &str, resident: &str) -> [Self; 2] {
        let layout = resident_layout(device);
        let (h, f) = (512, 256);
        [[f, h], [h, f]].map(|matrix| Self {
            source: seismic::Element::named(source).unwrap(),
            resident: element_in(resident, layout),
            matrix,
        })
    }

    fn shape(&self, count: u64) -> [u64; 3] {
        [count, self.matrix[0], self.matrix[1]]
    }

    /// Expert `expert`'s stored matrix: pseudo-random whole packets brought
    /// into the conversion's domain, every byte below 0x78 so that no
    /// 16-bit float scale is infinite or NaN (a NaN row would make a ReLU²
    /// product zero whatever its source); a dense BF16 matrix's values
    /// finite, of both signs.
    fn stored(&self, expert: usize, kind: usize) -> Vec<u8> {
        if self.source == seismic::Element::bf16() {
            return (0..(self.matrix[0] * self.matrix[1]) as u32)
                .flat_map(|index| {
                    let x = index.wrapping_add(expert as u32 * 7919 + kind as u32 * 104_729).wrapping_mul(2_654_435_761);
                    ((0x3c00 + (x >> 20) % 0x0700) as u16 | (((x >> 9) & 1) as u16) << 15).to_le_bytes()
                })
                .collect();
        }
        let bytes = self.source.canonical_byte_len(&self.shape(1)).unwrap() as usize;
        let raw = (0..bytes as u32)
            .map(|index| {
                let x = index
                    .wrapping_add(expert as u32 * 7919 + kind as u32 * 104_729)
                    .wrapping_mul(2_654_435_761);
                ((x >> 13) % 0x78) as u8
            })
            .collect();
        self.resident.repack_source(self.source, &self.shape(1), raw).unwrap()
    }

    /// The resident import of a stored matrix: a dense BF16 matrix's
    /// bytes themselves, or each element widened to F32 (`import_dense`).
    fn import(&self, stored: &[u8]) -> Vec<u8> {
        if self.source == seismic::Element::bf16() {
            return match self.resident == self.source {
                true => stored.to_vec(),
                false => stored.chunks_exact(2).flat_map(|bits| (u32::from(u16::from_le_bytes([bits[0], bits[1]])) << 16).to_le_bytes()).collect(),
            };
        }
        self.resident
            .repack_host(self.source, &self.shape(1), stored)
            .unwrap()
    }

    fn slot_bytes(&self) -> usize {
        self.resident.canonical_byte_len(&self.shape(1)).unwrap() as usize
    }

    /// A [count, rows, columns] tensor of `element` whose matrix i is
    /// `matrix(i)`, zero bytes where it is `None`.
    fn tensor<'a>(
        &self,
        device: &seismic::Device,
        element: seismic::Element,
        count: u64,
        matrix: impl Fn(usize) -> Option<&'a [u8]>,
    ) -> seismic::Tensor {
        let per = element.canonical_byte_len(&self.shape(1)).unwrap() as usize;
        let mut bytes = vec![0u8; per * count as usize];
        for index in 0..count as usize {
            if let Some(matrix) = matrix(index) {
                bytes[index * per..(index + 1) * per].copy_from_slice(matrix);
            }
        }
        seismic::Tensor::from_host(&seismic::Charges::tally(device), element, &self.shape(count), &bytes).unwrap()
    }
}

/// The granule pool of `contract`'s store for record stores of `layouts`'
/// records, its granules each the largest record and its blocks of
/// `per_block` granules.
fn record_pool(device: &seismic::Device, contract: seismic::Contract, per_block: u64, layouts: &[seismic::SlabLayout]) -> seismic::Pool {
    let granule = layouts.iter().map(|layout| layout.slab_bytes.next_multiple_of(layout.slab_alignment)).max().unwrap().next_multiple_of(256);
    seismic::Pool::open(&seismic::PoolCharges::one(&seismic::Charges::tally(device)), contract, granule, per_block * granule).unwrap()
}

/// The regions of a record store of `element` over `shape` (`[records,
/// ...record]`): one, a record a row.
fn record_regions(element: seismic::Element, shape: &[u64]) -> [seismic::SlabRegion; 1] {
    [seismic::SlabRegion { element, row_shape: shape[1..].to_vec() }]
}

/// The layout of one record of `element` over `shape`.
fn record_layout(element: seismic::Element, shape: &[u64]) -> seismic::SlabLayout {
    seismic::SlabLayout::for_regions(1, shape[0], &record_regions(element, shape)).unwrap()
}

/// A record store in `pool` of `element` over `shape`, a slab a record,
/// every record holding `bytes`: the records as a kernel's record port
/// binds them, and the store, which must outlive them.
fn record_store(charges: &seismic::Charges, pool: &seismic::Pool, element: seismic::Element, shape: &[u64], bytes: &[u8]) -> (seismic::Tensor, seismic::SlabTensor) {
    let records = shape[0];
    let store = seismic::SlabTensor::in_pool(charges, pool, 1, records, &record_regions(element, shape)).unwrap();
    let per = bytes.len().checked_div(records as usize).unwrap_or(0);
    assert_eq!(per * records as usize, bytes.len(), "record bytes must divide evenly across {records} records");
    for record in 0..records {
        store.add_slab(seismic::SlabFill::Zeroed).unwrap();
        let start = record as usize * per;
        store.region_rows(0, record, 1).unwrap().write_from_host(&bytes[start..start + per]).unwrap();
    }
    (store.logical_region(0).unwrap(), store)
}

/// A record form a host tier serves on the 3090: its tensors' kinds, gate
/// (gated records only), up and down.
struct ServedForm {
    gate: Option<StreamedKind>,
    up: StreamedKind,
    down: StreamedKind,
}

impl ServedForm {
    /// Nemotron 3.5 Lightning's up-only records (F 1856, H 2688).
    fn lightning(device: &seismic::Device) -> Self {
        let layout = resident_layout(device);
        let (h, f) = (2688u64, 1856u64);
        let [up, down] = [("gguf_q5_0", "q5g32s", [f, h]), ("gguf_q8_0", "q8g32s", [h, f])].map(
            |(source, resident, matrix)| StreamedKind {
                source: seismic::Element::named(source).unwrap(),
                resident: element_in(resident, layout),
                matrix,
            },
        );
        Self { gate: None, up, down }
    }

    /// The record's tensors' kinds: gate (where gated), up and down.
    fn kinds(&self) -> Vec<StreamedKind> {
        self.gate.iter().copied().chain([self.up, self.down]).collect()
    }
}

/// A decode step's sources, each choice's in the bank, in the host tier or
/// some of each (an even expert's in the bank).
#[derive(Clone, Copy, Debug)]
enum Served {
    Bank,
    Tier,
    Mixed,
}

/// Where each expert's records lie in `TierServing`'s stores: expert x in
/// bank record E - 1 - x and in tier record (5 x + 3) mod E.
struct TierServing {
    experts: u64,
    /// Each tensor's bank, tier (with its store) and decoy bank (every
    /// record expert 0's).
    stores: Vec<(seismic::Tensor, (seismic::Tensor, seismic::SlabTensor), seismic::Tensor)>,
}

impl TierServing {
    /// Expert `x`'s bank record.
    fn bank_record(&self, expert: u64) -> u64 {
        self.experts - 1 - expert
    }

    /// Expert `x`'s tier record.
    fn tier_record(&self, expert: u64) -> u64 {
        (5 * expert + 3) % self.experts
    }

    /// The stores of `kinds`' records over `experts` experts (a power of
    /// two, so 5 x + 3 permutes them), each expert's the resident import of
    /// its stored bytes.
    fn new(device: &seismic::Device, kinds: &[StreamedKind], experts: u64) -> Self {
        let charges = seismic::Charges::tally(device);
        let mut serving = Self { experts, stores: Vec::new() };
        let layouts = kinds.iter().map(|kind| record_layout(kind.resident, &kind.shape(experts))).collect::<Vec<_>>();
        let pool = record_pool(device, seismic::Contract::Shared, experts, &layouts);
        for (index, kind) in kinds.iter().enumerate() {
            let imports = (0..experts as usize).map(|expert| kind.import(&kind.stored(expert, index))).collect::<Vec<_>>();
            let bank = kind.tensor(device, kind.resident, experts, |record| Some(&imports[(experts - 1) as usize - record]));
            let mut tier = vec![Vec::new(); experts as usize];
            for expert in 0..experts {
                tier[serving.tier_record(expert) as usize] = imports[expert as usize].clone();
            }
            let tier = record_store(&charges, &pool, kind.resident, &kind.shape(experts), &tier.concat());
            let decoy = kind.tensor(device, kind.resident, experts, |_| Some(&imports[0]));
            serving.stores.push((bank, tier, decoy));
        }
        serving
    }

    /// The sources of `routes` served as `served`.
    fn sources(&self, device: &seismic::Device, rows: u64, choices: u64, routes: &[i32], served: Served) -> seismic::Tensor {
        let records = routes
            .iter()
            .map(|&expert| {
                let expert = expert as u64;
                match (served, expert % 2) {
                    (Served::Bank, _) | (Served::Mixed, 0) => self.bank_record(expert) as i32,
                    _ => -2 - self.tier_record(expert) as i32,
                }
            })
            .collect::<Vec<_>>();
        decode_sources(device, rows, choices, &records)
    }

    /// Tensor `index`'s (bank, tier) as `served` reads them: in the tier,
    /// over the decoy bank no source names.
    fn stores(&self, index: usize, served: Served) -> (&seismic::Tensor, &seismic::Tensor) {
        let (bank, (tier, _), decoy) = &self.stores[index];
        match served {
            Served::Tier => (decoy, tier),
            _ => (bank, tier),
        }
    }
}

/// A streamed layer's gate/up in two passes (`routed_gate_up_part`: the early
/// pass before the tier wait, the late pass after it) forms, bit for bit, the
/// product `routed_gate_up` forms in one pass over the same rows with the
/// sources the wait left, for M = 1 to VERIFY_ROWS rows (Metal's two-row and
/// several-row launches), whichever choices the route left absent (none,
/// alternate experts, all). The rows share experts, and the odd rows are
/// front rows ([`front_sharing_routes`]): a front row's absent choice of an
/// expert an exact row also chose is sourced by the wait, its record shared,
/// as resolve loads it; any other absent choice of it no wait sources (-1
/// after it too), so it holds a zero product. Where the backend declares a
/// gathered form (`GATHERED`), both entries run each form.
#[test]
fn gate_up_in_two_passes_forms_the_one_pass_product() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    // The gathered form, where the passes declare it as the gate/up does.
    let declared = [
        routed_gate_up::native_implementation(&device).unwrap().expect("every backend implements the gate/up"),
        routed_gate_up_part::native_implementation(&device).unwrap().expect("a streaming backend implements the gate/up passes"),
    ]
    .map(|implementation| implementation.params.iter().any(|parameter| parameter.name == "GATHERED"));
    assert_eq!(declared[0], declared[1], "the gate/up passes declare the gathered form as the gate/up does");
    let decode = decode_mappings(&device);
    // Each case's mapping, shared by the one-pass gate/up and the passes, in
    // each declared form.
    let forms: &[u64] = if declared[0] { &[0, 1] } else { &[0] };
    let mappings = [&decode[0], &decode[decode.len() - 1]]
        .into_iter()
        .flat_map(|mapping| {
            forms.iter().map(move |&form| {
                let mut mapping = mapping.clone();
                if declared[0] {
                    mapping.push(("GATHERED", form));
                }
                mapping
            })
        })
        .collect::<Vec<_>>();
    let bf16 = seismic::Element::bf16();
    let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let kinds = StreamedKind::qwen35b(&device);
    let (h, f, k, e) = (Qwen35b::HIDDEN as u64, Qwen35b::FEATURES as u64, Qwen35b::CHOICES as u64, 16u64);
    let (ku, fu) = (k as usize, f as usize);
    let serving = TierServing::new(&device, &kinds, e);
    let (gate, tier_gate) = serving.stores(0, Served::Bank);
    let (up, tier_up) = serving.stores(1, Served::Bank);
    let (limit, gate_scale, up_scale) = (no_limit(&device), absent(&device), absent(&device));
    let mut cases = 0usize;
    // A front row's absent choices: skipped, and sourced by the wait as an
    // exact row chose the same expert.
    let mut fronts = [0usize; 2];
    for mapping in &mappings {
        let one = routed_gate_up::native_for_device_with(
            &device,
            routed_gate_up::Elements { A: bf16, EGW: kinds[0].resident, EUW: kinds[1].resident },
            &decode_specialization::<routed_gate_up::Entry>(&device, &[("H", h), ("K", k), ("F", f)], mapping),
        )
        .unwrap();
        let part = routed_gate_up_part::native_for_device_with(
            &device,
            routed_gate_up_part::Elements { A: bf16, EGW: kinds[0].resident, EUW: kinds[1].resident },
            &decode_specialization::<routed_gate_up_part::Entry>(&device, &[("H", h), ("K", k), ("F", f)], mapping),
        )
        .unwrap();
        for m in 1..=constants::VERIFY_ROWS {
            let routes_values = front_sharing_routes(m, k, e);
            let routes = i32_tensor(&device, &[m, k], &routes_values);
            let normalized_values = pattern((m * h) as usize, 73, 1.0);
            let normalized = bf16_tensor(&device, &[m, h], &normalized_values);
            let choices = (m * k) as usize;
            let records = routes_values.iter().map(|&expert| serving.bank_record(expert as u64) as i32).collect::<Vec<_>>();
            // The experts the route left absent (every choice of one is, as the route copies an
            // expert's first source): none, every other one, all.
            for (label, absent_expert) in [
                ("none absent", (|_: i32| false) as fn(i32) -> bool),
                ("alternate experts absent", |expert| expert % 2 == 1),
                ("all absent", |_| true),
            ] {
                let label = format!("rows {m} {mapping:?} {label}");
                let front = |choice: usize| (choice / ku) % 2 == 1;
                // An exact row's choice of an expert, which its wait sources.
                let exact_chose = |expert: i32| (0..choices).any(|other| !front(other) && routes_values[other] == expert);
                let early_records = (0..choices)
                    .map(|choice| if absent_expert(routes_values[choice]) { -1 } else { records[choice] })
                    .collect::<Vec<_>>();
                // The wait sources every absent choice of an expert an exact
                // row chose, a front row's among them (its record shared),
                // and no other: a front row skips those.
                let waited = (0..choices)
                    .map(|choice| match front(choice) && !exact_chose(routes_values[choice]) {
                        true => early_records[choice],
                        false => records[choice],
                    })
                    .collect::<Vec<_>>();
                for choice in (0..choices).filter(|&choice| front(choice) && early_records[choice] == -1) {
                    fronts[usize::from(waited[choice] != -1)] += 1;
                }
                let (early, full) = (decode_sources(&device, m, k, &early_records), decode_sources(&device, m, k, &waited));
                let flight = 2u64;
                let sequence = 3u64;
                let width = constants::ROUTING_WORDS(choices as u64) as usize;
                let mut ring = vec![0; flight as usize * width];
                let slot = (sequence % flight) as usize;
                for choice in 0..choices {
                    ring[slot * width + constants::ROUTING_SOURCE(choices as u64, choice as u64) as usize] = early_records[choice];
                }
                let routing = shared_i32(&device, &[flight, width as u64], &ring);
                let step = shared_i32(&device, &[constants::STEP_WORDS], &step_words(sequence));
                // The product starts as another step's leftovers: the early pass writes every row of it.
                let mut product = bf16_tensor(&device, &[m, k, f], &vec![0.375; (m * k * f) as usize]);
                for (late, sources) in [(0, &early), (1, &full)] {
                    part.call(&charges, routed_gate_up_part::Args {
                        normalized: &normalized,
                        routes: &routes,
                        sources,
                        expert_gate: gate,
                        expert_up: up,
                        tier_gate,
                        tier_up,
                        activation: 0,
                        glu_limit: &limit,
                        gate_scale: &gate_scale,
                        up_scale: &up_scale,
                        routing: &routing,
                        step: &step,
                        late,
                        product: &mut product,
                    })
                    .unwrap();
                }
                let product = bits(&read_bf16(&product));
                let one_pass = one
                    .call(&charges, routed_gate_up::Args {
                        normalized: &normalized,
                        routes: &routes,
                        sources: &full,
                        expert_gate: gate,
                        expert_up: up,
                        tier_gate,
                        tier_up,
                        activation: 0,
                        glu_limit: &limit,
                        gate_scale: &gate_scale,
                        up_scale: &up_scale,
                    })
                    .unwrap()
                    .value;
                let one_pass = bits(&read_bf16(&one_pass));
                for row in (0..m as usize).filter(|&row| waited[row * ku..(row + 1) * ku].iter().any(|&source| source != -1)) {
                    assert!(one_pass[row * ku * fu..(row + 1) * ku * fu].iter().any(|&value| value != 0), "{label}: row {row} has a product");
                }
                assert_eq!(product, one_pass, "{label}: the two passes against one");
                cases += 1;
            }
        }
    }
    assert!(fronts.iter().all(|&count| count > 0), "a front row's absent choices were skipped and waited for {fronts:?} times");
    println!("gate/up passes on {:?}: {cases} cases against one pass", device.backend());
}

/// Decode rows `rows` of `choices` choices over `experts` experts (at least
/// 4 `choices`), each exact (even) row choosing even experts, row m's choice
/// k (4 m + 2 k) mod E, rows sharing them; each front (odd) row choosing
/// its even choices as the exact row before it does and its odd ones odd
/// experts no exact row chooses ((2 (k + m) + 1) mod E).
fn front_sharing_routes(rows: u64, choices: u64, experts: u64) -> Vec<i32> {
    (0..rows)
        .flat_map(|row| {
            (0..choices).map(move |choice| {
                let expert = match (row % 2, choice % 2) {
                    (0, _) => 4 * row + 2 * choice,
                    (_, 0) => 4 * (row - 1) + 2 * choice,
                    _ => 2 * (choice + row) + 1,
                };
                (expert % experts) as i32
            })
        })
        .collect()
}

/// Decode rows `rows` of `choices` choices over `experts` experts, choice k
/// of row m expert (`stride` m + 2 k) mod E: distinct within a row, rows
/// sharing experts (the gathered forms' case).
fn shared_routes(rows: u64, choices: u64, experts: u64, stride: u64) -> Vec<i32> {
    (0..rows)
        .flat_map(|row| (0..choices).map(move |choice| ((stride * row + 2 * choice) % experts) as i32))
        .collect()
}

/// The decode entries read a choice's records in place from the host tier
/// (sources -2 - t) bit for bit as from the same bytes in bank records, and
/// a launch serving some choices from each as serving all from the bank:
/// the gated `routed_gate_up` and `routed_output` over the 35B's Q4_K and
/// Q5_K records, and the up-only `routed_up` and `routed_down` over Nemotron
/// 3.5 Lightning's Q5_0 and Q8_0 ones, at one row (the per-choice launches)
/// and several (the gathered ones), rows sharing experts, at two decode
/// mappings; the tier read over a decoy bank no source names. On every
/// streaming backend.
#[test]
fn streaming_tier_records_serve_as_bank_records() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let mappings = decode_mappings(&device);
    let mappings = [&mappings[0], &mappings[mappings.len() - 1]];
    let bf16 = seismic::Element::bf16();
    let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let served = [Served::Bank, Served::Tier, Served::Mixed];

    // Gated: the 35B.
    let kinds = StreamedKind::qwen35b(&device);
    let (h, f, k, s, e) = (Qwen35b::HIDDEN as u64, Qwen35b::FEATURES as u64, Qwen35b::CHOICES as u64, Qwen35b::SHARED as u64, 16u64);
    let serving = TierServing::new(&device, &kinds, e);
    let shared_down = bf16_tensor(&device, &[h, s], &pattern((h * s) as usize, 71, 0.05));
    for (index, mapping) in mappings.iter().enumerate() {
        let gate_up = declared::prepared(
            &device,
            &routed_gate_up::native_entry_with(routed_gate_up::Elements { A: bf16, EGW: kinds[0].resident, EUW: kinds[1].resident }),
            &decode_specialization::<routed_gate_up::Entry>(&device, &[("H", h), ("K", k), ("F", f)], mapping),
        )
        .unwrap();
        let output = declared::prepared(
            &device,
            &routed_output::native_entry_with(routed_output::Elements { A: bf16, EDW: kinds[2].resident, SDW: bf16, NW: seismic::Element::f32() }),
            &decode_specialization::<routed_output::Entry>(
                &device,
                &[("H", h), ("K", k), ("F", f), ("S", s)],
                &output_mapping(&device, index, mapping),
            ),
        )
        .unwrap();
        for m in [1u64, 4, 8] {
            let routes_values = shared_routes(m, k, e, 5);
            let routes = i32_tensor(&device, &[m, k], &routes_values);
            let normalized = bf16_tensor(&device, &[m, h], &pattern((m * h) as usize, 73, 1.0));
            let residual = f32_tensor(&device, &[m, h], &pattern((m * h) as usize, 79, 1.0));
            let scores = f32_tensor(&device, &[m, k], &pattern((m * k) as usize, 83, 0.5));
            let coefficient = f32_tensor(&device, &[m], &pattern(m as usize, 89, 0.5));
            let shared_product = bf16_tensor(&device, &[m, s], &pattern((m * s) as usize, 97, 0.5));
            let values = served.map(|served| {
                let sources = serving.sources(&device, m, k, &routes_values, served);
                let (gate, tier_gate) = serving.stores(0, served);
                let (up, tier_up) = serving.stores(1, served);
                let (down, tier_down) = serving.stores(2, served);
                let product = gate_up
                    .call(&charges, routed_gate_up::Args {
                        normalized: &normalized,
                        routes: &routes,
                        sources: &sources,
                        expert_gate: gate,
                        expert_up: up,
                        tier_gate,
                        tier_up,
                        activation: 0,
                        glu_limit: &no_limit(&device),
                        gate_scale: &absent(&device),
                        up_scale: &absent(&device),
                    })
                    .unwrap()
                    .value;
                let value = output
                    .call(&charges, routed_output::Args {
                        residual: &residual,
                        inject: &no_inject(&device, m),
                        expert_product: &product,
                        shared_product: &shared_product,
                        routes: &routes,
                        sources: &sources,
                        scores: &scores,
                        coefficient: &coefficient,
                        expert_down: down,
                        tier_down,
                        shared_down: &shared_down,
                        operand_norm: &absent(&device),
                        operand_epsilon: 0.0,
                    })
                    .unwrap()
                    .value;
                (served, bits(&read_bf16(&product)), bits(&read_f32(&value)))
            });
            let (_, product, value) = &values[0];
            assert!(product.iter().any(|&bits| bits != 0), "35B rows {m} {mapping:?}: a product");
            for (served, served_product, served_value) in &values[1..] {
                let label = format!("35B rows {m} {mapping:?} {served:?}");
                assert_eq!(served_product, product, "{label}: gate/up product");
                assert_eq!(served_value, value, "{label}: output");
            }
        }
    }

    // Up-only: Nemotron 3.5 Lightning.
    let form = ServedForm::lightning(&device);
    let kinds = form.kinds();
    let (h, f, k, e) = (form.down.matrix[0], form.up.matrix[0], 6u64, 8u64);
    let serving = TierServing::new(&device, &kinds, e);
    for (index, mapping) in mappings.iter().enumerate() {
        let up = declared::prepared(
            &device,
            &routed_up::native_entry_with(routed_up::Elements { A: bf16, EUW: kinds[0].resident }),
            &decode_specialization::<routed_up::Entry>(&device, &[("H", h), ("K", k), ("F", f)], mapping),
        )
        .unwrap();
        let down = declared::prepared(
            &device,
            &routed_down::native_entry_with(routed_down::Elements { A: bf16, EDW: kinds[1].resident, R: seismic::Element::f32() }),
            &decode_specialization::<routed_down::Entry>(&device, &[("H", h), ("K", k), ("F", f)], &output_mapping(&device, index, mapping)),
        )
        .unwrap();
        for m in [1u64, 4, 8] {
            let routes_values = shared_routes(m, k, e, 3).iter().map(|&expert| (expert / 2 + 4 * (expert % 2)) % e as i32).collect::<Vec<_>>();
            let routes = i32_tensor(&device, &[m, k], &routes_values);
            let normalized = bf16_tensor(&device, &[m, h], &pattern((m * h) as usize, 101, 1.0));
            let base = f32_tensor(&device, &[m, h], &pattern((m * h) as usize, 103, 1.0));
            let weights = f32_tensor(&device, &[m, k], &pattern((m * k) as usize, 107, 0.5));
            let values = served.map(|served| {
                let sources = serving.sources(&device, m, k, &routes_values, served);
                let (expert_up, tier_up) = serving.stores(0, served);
                let (expert_down, tier_down) = serving.stores(1, served);
                let product = up
                    .call(&charges, routed_up::Args {
                        normalized: &normalized,
                        routes: &routes,
                        sources: &sources,
                        expert_up,
                        tier_up,
                        up_scale: &absent(&device),
                        activation: 2,
                    })
                    .unwrap()
                    .value;
                let value = down
                    .call(&charges, routed_down::Args {
                        base: &base,
                        product: &product,
                        routes: &routes,
                        sources: &sources,
                        weights: &weights,
                        expert_down,
                        tier_down,
                        policy: &i32_tensor(&device, &[constants::POLICY_WORDS], &[0; constants::POLICY_WORDS as usize]),
                    })
                    .unwrap()
                    .value;
                (served, bits(&read_bf16(&product)), bits(&read_f32(&value)))
            });
            let (_, product, value) = &values[0];
            assert!(product.iter().any(|&bits| bits != 0), "up-only rows {m} {mapping:?}: a product");
            for (served, served_product, served_value) in &values[1..] {
                let label = format!("up-only rows {m} {mapping:?} {served:?}");
                assert_eq!(served_product, product, "{label}: up product");
                assert_eq!(served_value, value, "{label}: down output");
            }
        }
    }
}

/// `routed_scratch_request` counts every choice of every row at the shapes
/// serving requests, each choice of a row routed apart: an expert only one
/// choice names is flagged, or its pass reads a scratch slot no round filled.
#[test]
fn a_scratch_request_counts_every_choice_of_every_row() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let sequence = STREAMING_SEQUENCE;
    let step = i32_tensor(&device, &[constants::STEP_WORDS], &step_words(sequence as u64));
    let experts = 128u64;
    let table = SlotTable::scratch(FLIGHT, 3);
    for (rows, k) in [(1u64, 2u64), (2, 4), (128, 2), (512, 2), (512, 4)] {
        let routes = (0..rows * k)
            .map(|choice| i32::try_from((choice * 7 + choice / k * 3) % experts).unwrap())
            .collect::<Vec<_>>();
        let mut flags = shared_i32(&device, &[experts], &vec![-3; experts as usize]);
        let mut scratch_table = shared_i32(&device, &table.extents(), &vec![0; table.words() as usize]);
        declared::prepared(&device, &routed_scratch_request::native_entry(), &specialize(&[("K", k), ("E", experts)], &[]))
            .unwrap()
            .call(&charges, routed_scratch_request::Args {
                routes: &i32_tensor(&device, &[rows, k], &routes),
                step: &step,
                flags: &mut flags,
                scratch: &mut scratch_table,
                resident: &shared_i32(&device, &[1], &[0]),
            })
            .unwrap();
        let mut expected = vec![0; experts as usize];
        for &expert in &routes {
            expected[expert as usize] += 1;
        }
        assert_eq!(read_i32(&flags), expected, "{rows} rows of {k} choices");
    }
}

/// A scratch-served layer's step and non-zero window (`w = 1` of two windows)
/// as serving encodes it: `routed_step` sets the step's sequence;
/// `routed_scratch_request` publishes each expert's choice count and the
/// sequence; the window's round 0 landing launch arrives at lane
/// `ROUND_LANE(w * rounds)`, and its host, having written the count of
/// records the window's rounds convert (`SCRATCH_STAGED` in that round-0
/// lane), appends round 0's records to its landing list and answers it;
/// `routed_scratch_grids` then writes every round's park grid of the window,
/// on which each later round launches while it or the round before has
/// records, converting its records as the host appends them (`routed_round`,
/// every routed tensor of the 35B's gated records). Four records in rounds of
/// two: round 1 launches and converts two, round 2 launches (its arrival,
/// answered before the last round's, proves round 1's records read) and
/// converts none. Of round 1's, expert 3 converts from scratch slot 12, a slot
/// no pass reads, which the host read its stored bytes into at the slot's base
/// (its `from` -1 - 12). Every listed slot then holds its expert's import,
/// byte for byte, slot 12 the stored bytes read there, and every other none;
/// window 0's lanes stay untouched; a failed window's grids launch nothing.
#[test]
fn streaming_scratch_rounds_fill_the_scratch_bank() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let sequence = STREAMING_SEQUENCE;
    let (experts, windows, w, rounds, round_records) = (16u64, 2u64, 1u64, 3u64, 2usize);
    let window = i32_tensor(&device, &[1], &[w as i32]);
    let form = ServedForm::qwen35b(&device);
    // Conveyor records 0, 1 and 2 hold experts 9, 7 and 5, scratch slot 12
    // expert 3's stored bytes; the rounds' (record, scratch slot) entries.
    let conveyed = [9usize, 7, 5];
    let (lent_slot, lent_expert) = (12usize, 3usize);
    let lists = [vec![(1, 7), (2, 5)], vec![(0, 9), (-1 - lent_slot as i32, lent_expert as i32)]];
    let records = lists.iter().map(Vec::len).sum::<usize>();
    let conveyor = form.records(&device, false, conveyed.len() as u64, |tensor, record| Some(form.stored(conveyed[record], tensor)));
    let lent = |tensor: usize| {
        let mut bytes = vec![0u8; form.kind(tensor).slot_bytes()];
        let stored = form.stored(lent_expert, tensor);
        bytes[..stored.len()].copy_from_slice(&stored);
        bytes
    };
    let mut scratch = form.records(&device, true, experts, |tensor, slot| (slot == lent_slot).then(|| lent(tensor)));

    // The step's sequence.
    let mut step = i32_tensor(&device, &[constants::STEP_WORDS], &[0; constants::STEP_WORDS as usize]);
    declared::prepared(&device, &routed_step::native_entry(), &seismic::NativeSpecialization::new())
        .unwrap()
        .call(&charges, routed_step::Args {
            step: &mut step,
            sequence: &i32_tensor(&device, &[constants::STEP_WORDS], &step_words(sequence as u64)),
            guesses: &no_tokens(&device),
        })
        .unwrap();
    assert_eq!(read_i32(&step), step_words(sequence as u64), "step");

    // The request: rows of 4 choices over experts 2, 5, 7 and 9.
    let routes = [2, 5, 7, 9, 2, 7, 5, 2];
    let mut flags = shared_i32(&device, &[experts], &vec![-3; experts as usize]);
    let table = SlotTable::scratch(FLIGHT, windows * rounds);
    let q = sequence as u64;
    let lane = |lane: u64| table.lane(q, lane);
    let ask = lane(constants::SCRATCH_ASK);
    let first = lane(constants::ROUND_LANE(w * rounds));
    let mut scratch_table = shared_i32(&device, &table.extents(), &vec![0; table.words() as usize]);
    let _pulse = pulse(&scratch_table, table);
    let k = 4u64;
    declared::prepared(
        &device,
        &routed_scratch_request::native_entry(),
        &specialize(&[("K", k), ("E", experts)], &[]),
    )
    .unwrap()
    .call(&charges, routed_scratch_request::Args {
        routes: &i32_tensor(&device, &[routes.len() as u64 / k, k], &routes),
        step: &step,
        flags: &mut flags,
        scratch: &mut scratch_table,
        resident: &shared_i32(&device, &[1], &[0]),
    })
    .unwrap();
    let mut expected_flags = vec![0; experts as usize];
    for &expert in &routes {
        expected_flags[expert as usize] += 1;
    }
    assert_eq!(read_i32(&flags), expected_flags, "flags");
    let mut expected = vec![0; table.words() as usize];
    set_pair(&mut expected, ask + constants::SLOT_REQUEST, q);
    assert_eq!(unpulsed(read_i32(&scratch_table), table), expected, "the request in its lane");

    let kernel = form.round_kernel(&device);
    let extent = constants::LANDING_ENTRY(round_records as u64);
    let landings = (0..rounds).map(|_| shared_i32(&device, &[extent], &vec![0; extent as usize])).collect::<Vec<_>>();
    // Round `round`'s launch of window `w` on `grid`. Where kernels park
    // themselves, the launch arrives and spins, and the host serves once the
    // device arrives (its lane's request): before round 0's answer the count
    // of records the window's rounds convert (in the window's round-0 lane),
    // then each of the round's records appended, then the lane's answer. Where
    // a park is a submission boundary (Vulkan), the host serves before the
    // call, as the boundary opens only once its rule holds, and the call is
    // made answered; the launch then records its arrival itself. The round
    // after the last with records is answered with no entry, before the device
    // arrives, as its host answers it. The step's high word is zero, so each
    // u64 is whole at its low word.
    let in_kernels = device.backend().parks_in_kernels();
    let park = |table: &mut seismic::Tensor, scratch: &mut [seismic::Tensor; 3], round: u64, grid: &seismic::Tensor| {
        let at = lane(constants::ROUND_LANE(w * rounds + round));
        let mut landing = landings[round as usize].clone();
        let serve = |served: &seismic::Tensor, landing: &seismic::Tensor| {
            if let Some(pairs) = lists.get(round as usize) {
                if in_kernels {
                    await_shared_word(served, at + constants::SLOT_REQUEST, sequence);
                }
                if round == 0 {
                    served.store_release(4 * (ask + constants::SLOT_ANSWER), sequence as u32).unwrap();
                    served.store_release(4 * (first + constants::SCRATCH_STAGED), records as u32).unwrap();
                }
                for (index, &(from, to)) in pairs.iter().enumerate() {
                    append_landing(landing, sequence, index, from, to);
                }
                served.store_release(4 * (at + constants::SLOT_ANSWER), sequence as u32).unwrap();
            } else if round as usize == lists.len() {
                served.store_release(4 * (at + constants::SLOT_ANSWER), sequence as u32).unwrap();
            }
        };
        let (served, appended) = (table.clone(), landing.clone());
        let tiered = host_words(&device, &[constants::LIST_ENTRY(experts)]);
        let seeds = host_words(&device, &[constants::LIST_ENTRY(experts)]);
        let map = host_words(&device, &[experts, constants::MAP_WORDS]);
        let [gate, up, down] = scratch.each_mut();
        let args = routed_round::Args {
            step: &step,
            scratch: table,
            window: &window,
            rounds: rounds as i32,
            round: round as i32,
            landing: &mut landing,
            tiered: &tiered,
            conveyor_gate: &conveyor[0],
            conveyor_up: &conveyor[1],
            conveyor_down: &conveyor[2],
            gate,
            up,
            down,
            seeds: &seeds,
            map: &map,
            grid,
        };
        if in_kernels {
            std::thread::scope(|scope| {
                scope.spawn(|| serve(&served, &appended));
                kernel.call(&charges, args).unwrap();
            });
        } else {
            serve(&served, &appended);
            kernel.call_answered(&charges, args).unwrap();
        }
    };
    let scratch_grids = declared::prepared(&device, &routed_scratch_grids::native_entry(), &seismic::NativeSpecialization::new()).unwrap();
    let grids_of = |scratch: &seismic::Tensor| {
        let mut grids = i32_tensor(&device, &[rounds, 3], &vec![-7; (rounds * 3) as usize]);
        scratch_grids
            .call(&charges, routed_scratch_grids::Args { step: &step, scratch, window: &window, grids: &mut grids, round_records: round_records as i32 })
            .unwrap();
        grids
    };
    // Every round's park grid for `count` records: LAND_GROUPS threadgroups
    // while n_r = clamp(count - r L, 0, L) > 0 or n_{r-1} > 0 (unfollowed).
    let land_groups = constants::LAND_GROUPS as i32;
    let expected_grids = |count: usize| {
        (0..rounds as usize)
            .flat_map(|round| [land_groups * i32::from(count > round.saturating_sub(1) * round_records), 1, 1])
            .collect::<Vec<_>>()
    };
    let grid = |grids: &seismic::Tensor, round: u64| grids.slice_leading(round, round + 1).unwrap().reshape(&[3]).unwrap();
    let always = i32_tensor(&device, &[3], &[land_groups, 1, 1]);
    let mut grids = None;
    for round in 0..rounds {
        // Round 0 always launches, every later round on its park grid.
        match &grids {
            None => park(&mut scratch_table, &mut scratch, round, &always),
            Some(grids) => park(&mut scratch_table, &mut scratch, round, &grid(grids, round)),
        }
        let written = grids.get_or_insert_with(|| grids_of(&scratch_table));
        assert_eq!(read_i32(written), expected_grids(records), "the rounds' grids");
    }
    // Active rounds and the round after them arrived and were answered; a
    // round after that and window 0's lanes ran nothing.
    set_pair(&mut expected, ask + constants::SLOT_ANSWER, q);
    expected[(first + constants::SCRATCH_STAGED) as usize] = records as i32;
    for round in 0..=lists.len() as u64 {
        let at = lane(constants::ROUND_LANE(w * rounds + round));
        set_pair(&mut expected, at + constants::SLOT_REQUEST, q);
        set_pair(&mut expected, at + constants::SLOT_ANSWER, q);
    }
    assert_eq!(unpulsed(read_i32(&scratch_table), table), expected, "the rounds' arrivals and answers");
    for (tensor, slots) in scratch.iter().enumerate() {
        let bytes = slots.read_to_host().unwrap();
        let width = form.kind(tensor).slot_bytes();
        for expert in 0..experts as usize {
            let slot = &bytes[expert * width..(expert + 1) * width];
            if conveyed.contains(&expert) || expert == lent_expert {
                assert!(slot == form.import(expert, tensor), "tensor {tensor} scratch slot {expert}");
            } else if expert == lent_slot {
                assert!(slot == lent(tensor).as_slice(), "tensor {tensor} lent scratch slot {expert}");
            } else {
                assert!(slot.iter().all(|&byte| byte == 0), "tensor {tensor} unlisted scratch slot {expert}");
            }
        }
    }
    // Every round's entries' DONE words hold the step's low word.
    for (round, pairs) in lists.iter().enumerate() {
        let words = read_i32(&landings[round]);
        for entry in 0..pairs.len() {
            assert_eq!(words[landing_done(entry)], sequence, "round {round} entry {entry}'s DONE");
        }
    }
    // A failed window's rounds launch nothing.
    let mut failed = read_i32(&scratch_table);
    set_pair(&mut failed, first + constants::SLOT_ANSWER, q | (u64::from(constants::SLOT_FAILED as u32) << 32));
    assert_eq!(
        read_i32(&grids_of(&shared_i32(&device, &table.extents(), &failed))),
        expected_grids(0),
        "a failed window's grids"
    );
    // A followed window fill (a sibling fill's round 0 is the next launch on
    // its queue): the round after its last with records launches no more, the
    // sibling's arrival proving its records read.
    let mut followed = read_i32(&scratch_table);
    followed[(first + constants::SCRATCH_FOLLOWED) as usize] = 1;
    let unparked = (0..rounds as usize)
        .flat_map(|round| [land_groups * i32::from(records > round * round_records), 1, 1])
        .collect::<Vec<_>>();
    assert_eq!(
        read_i32(&grids_of(&shared_i32(&device, &table.extents(), &followed))),
        unparked,
        "a followed window's grids"
    );
    // A round-0 lane unanswered past round 0's wait (it ended on its still
    // pulse): every round launches, so the device arrives at each round the
    // host serves, and none converts.
    let mut unanswered = read_i32(&scratch_table);
    set_pair(&mut unanswered, first + constants::SLOT_ANSWER, q - 1);
    let parked = (0..rounds).flat_map(|_| [land_groups, 1, 1]).collect::<Vec<_>>();
    assert_eq!(
        read_i32(&grids_of(&shared_i32(&device, &table.extents(), &unanswered))),
        parked,
        "an expired round 0's grids"
    );
}

/// On every backend that streams, `routed_seed_commit` commits a window's
/// listed bank records with the step's low word iff that window's last round
/// lane (`ROUND_LANE(window[0] * rounds + rounds - 1)`) holds the step's
/// answer and the list names the step: answering another window's last round,
/// only the window's earlier round, failing the lane, or naming another step
/// commits nothing.
#[test]
fn streaming_seed_commit_commits_a_completed_windows_records() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let step = streaming_step(&device);
    let q = STREAMING_SEQUENCE as u64;
    let (windows, w, rounds, capacity, bank) = (2u64, 1u64, 2u64, 4usize, 8u64);
    let window = i32_tensor(&device, &[1], &[w as i32]);
    let table = SlotTable::scratch(FLIGHT, windows * rounds);
    let lane = |index: u64| table.lane(q, constants::ROUND_LANE(index));
    let pairs = [(0, 6), (2, 1), (3, 4)];
    let count = 2usize;
    let kernel = declared::prepared(&device, &routed_seed_commit::native_entry(), &specialize(&[("X", capacity as u64)], &[])).unwrap();
    let failed = q | (u64::from(constants::SLOT_FAILED as u32) << 32);
    for (label, list_seq, answered_lane, answer, commits_listed) in [
        ("completed window", STREAMING_SEQUENCE, w * rounds + rounds - 1, q, true),
        ("other window answered", STREAMING_SEQUENCE, rounds - 1, q, false),
        ("earlier round only", STREAMING_SEQUENCE, w * rounds, q, false),
        ("failed window", STREAMING_SEQUENCE, w * rounds + rounds - 1, failed, false),
        ("stale list", STREAMING_SEQUENCE - 1, w * rounds + rounds - 1, q, false),
    ] {
        let mut scratch = vec![0; table.words() as usize];
        set_pair(&mut scratch, lane(answered_lane) + constants::SLOT_ANSWER, answer);
        let list = shared_i32(&device, &[constants::LIST_ENTRY(capacity as u64)], &list_words(list_seq, count, &pairs, capacity));
        let mut commits = shared_i32(&device, &[bank], &vec![-5; bank as usize]);
        kernel
            .call(&charges, routed_seed_commit::Args {
                list: &list,
                step: &step,
                scratch: &shared_i32(&device, &table.extents(), &scratch),
                window: &window,
                rounds: rounds as i32,
                commits: &mut commits,
            })
            .unwrap();
        let mut expected = vec![-5; bank as usize];
        if commits_listed {
            for &(_, record) in &pairs[..count] {
                expected[record as usize] = STREAMING_SEQUENCE;
            }
        }
        assert_eq!(read_i32(&commits), expected, "{label}");
    }
}

// ---------------------------------------------------------------------------
// Prompt groups: a round's landing launch and `routed_copy`.

/// The resident layouts a bank record takes: Metal's, CUDA's, the others'.
const RESIDENT_LAYOUTS: [registry::Layout; 3] = [
    registry::Layout::Rows32,
    registry::Layout::Mma16,
    registry::Layout::Rows16,
];

/// A list's words (`list[constants::LIST_ENTRY(capacity)]`): the step it was written
/// for, the count, then the pairs, zero past them.
fn list_words(sequence: i32, count: usize, pairs: &[(i32, i32)], capacity: usize) -> Vec<i32> {
    let mut words = vec![0; constants::LIST_ENTRY(capacity as u64) as usize];
    words[constants::LIST_SEQUENCE as usize] = sequence;
    words[constants::LIST_COUNT as usize] = count as i32;
    for (index, &(from, to)) in pairs.iter().enumerate() {
        let entry = constants::LIST_ENTRY(index as u64) as usize;
        words[entry + constants::LIST_FROM as usize] = from;
        words[entry + constants::LIST_TO as usize] = to;
    }
    words
}

/// The 35B's two record forms: gate (and up, the same form) and down.
const RECORD_FORMS: [usize; 2] = [0, 2];

/// A landing list's words (`landing[constants::LANDING_ENTRY(capacity)]`): its head,
/// the step's low word `low` and the count `count` landed, then each (from,
/// to) pair, its DONE word zero, zero past them.
fn landing_words(low: i32, count: usize, pairs: &[(i32, i32)], capacity: usize) -> Vec<i32> {
    let mut words = vec![0; constants::LANDING_ENTRY(capacity as u64) as usize];
    words[constants::LANDING_HEAD as usize] = low;
    words[constants::LANDING_HEAD as usize + 1] = count as i32;
    for (index, &(from, to)) in pairs.iter().enumerate() {
        let entry = constants::LANDING_ENTRY(index as u64) as usize;
        words[entry + constants::LANDING_FROM as usize] = from;
        words[entry + constants::LANDING_TO as usize] = to;
    }
    words
}

/// The word of entry `entry`'s DONE in a landing list.
fn landing_done(entry: usize) -> usize {
    constants::LANDING_ENTRY(entry as u64) as usize + constants::LANDING_DONE as usize
}

/// A host appending entry `index` (from, to) to a running launch's landing
/// list of step low word `low`, as its read lands: the entry's words, then
/// the head (the step's low word, the count) by one 64-bit release store.
fn append_landing(landing: &seismic::Tensor, low: i32, index: usize, from: i32, to: i32) {
    let entry = constants::LANDING_ENTRY(index as u64);
    landing.store_release(4 * (entry + constants::LANDING_FROM), from as u32).unwrap();
    landing.store_release(4 * (entry + constants::LANDING_TO), to as u32).unwrap();
    landing
        .store_release_u64(4 * constants::LANDING_HEAD, u64::from(low as u32) | (index as u64 + 1) << 32)
        .unwrap();
}

impl ServedForm {
    /// Qwen3.6 35B's gated records: Q4_K gate and up, Q5_K down.
    fn qwen35b(device: &seismic::Device) -> Self {
        let [gate, up, down] = StreamedKind::qwen35b(device);
        Self { gate: Some(gate), up, down }
    }

    /// Up-only or gated dense BF16 records, resident in `resident` (BF16
    /// or F32), at F `f` and H `h`.
    fn dense(gated: bool, resident: seismic::Element, f: u64, h: u64) -> Self {
        let kind = |matrix| StreamedKind { source: seismic::Element::bf16(), resident, matrix };
        Self { gate: gated.then(|| kind([f, h])), up: kind([f, h]), down: kind([h, f]) }
    }

    /// DeepSeek V4.1's gated MXFP4 records (`StreamedKind::v41_mxfp4`).
    fn v41_mxfp4(device: &seismic::Device) -> Self {
        let [gate, down] = StreamedKind::v41_mxfp4(device);
        Self { gate: Some(gate), up: gate, down }
    }

    /// DeepSeek V4.1's MXFP4 tensors at row counts no multiple of 16, which a
    /// landing's tied F and H cannot give two MXFP4 tensors (a row count is
    /// the other's columns, a multiple of 32): each is MXFP4 in one form
    /// with its F or H the odd one, and the others dense BF16. Gate and up
    /// of [133, 5120], rows of 2720 bytes, 9 row tiles of a partial last
    /// (down [5120, 133]); down of [139, 2304], rows of 1224 bytes whose
    /// row tile a window holds whole, the last 11 rows, each record of
    /// 170,136 bytes, so every other starts 8 bytes off 16-byte alignment
    /// (gate and up [2304, 139]).
    fn v41_mxfp4_partial(device: &seismic::Device) -> [Self; 2] {
        let layout = resident_layout(device);
        let mxfp4 = |matrix| StreamedKind {
            source: seismic::Element::named("gguf_mxfp4").unwrap(),
            resident: element_in("mxfp4g32", layout),
            matrix,
        };
        let dense = |matrix| StreamedKind { source: seismic::Element::bf16(), resident: seismic::Element::bf16(), matrix };
        let (h, f) = (160 * 32, 72 * 32);
        [
            Self { gate: Some(mxfp4([133, h])), up: mxfp4([133, h]), down: dense([h, 133]) },
            Self { gate: Some(dense([f, 139])), up: dense([f, 139]), down: mxfp4([139, f]) },
        ]
    }

    /// GLM-5.3-Flash's gated FP8 records under a block-scale grid
    /// (`StreamedKind::glm_fp8`).
    fn glm_fp8(device: &seismic::Device, source: &str, resident: &str) -> Self {
        let [gate, down] = StreamedKind::glm_fp8(device, source, resident);
        Self { gate: Some(gate), up: gate, down }
    }

    /// Routed tensor `tensor`'s kind (0 gate, 1 up, 2 down): an up-only
    /// record's gate is the up's, its views empty.
    fn kind(&self, tensor: usize) -> StreamedKind {
        match tensor {
            0 => self.gate.unwrap_or(self.up),
            1 => self.up,
            _ => self.down,
        }
    }

    /// Expert `expert`'s stored matrix of routed tensor `tensor`, and its
    /// resident import.
    fn stored(&self, expert: usize, tensor: usize) -> Vec<u8> {
        self.kind(tensor).stored(expert, tensor)
    }

    fn import(&self, expert: usize, tensor: usize) -> Vec<u8> {
        self.kind(tensor).import(&self.stored(expert, tensor))
    }

    /// Each routed tensor's `count` records, stored or resident: record r of
    /// tensor t holding `matrix(t, r)`, zero where none; an up-only
    /// record's gate empty.
    fn records(
        &self,
        device: &seismic::Device,
        resident: bool,
        count: u64,
        matrix: impl Fn(usize, usize) -> Option<Vec<u8>>,
    ) -> [seismic::Tensor; 3] {
        std::array::from_fn(|tensor| {
            let kind = self.kind(tensor);
            let element = if resident { kind.resident } else { kind.source };
            let count = if tensor == 0 && self.gate.is_none() { 0 } else { count };
            let per = element.canonical_byte_len(&kind.shape(1)).unwrap() as usize;
            let mut bytes = vec![0u8; per * count as usize];
            for record in 0..count as usize {
                if let Some(held) = matrix(tensor, record) {
                    bytes[record * per..record * per + held.len()].copy_from_slice(&held);
                }
            }
            match count {
                0 => seismic::Tensor::zeros(&seismic::Charges::tally(device), element, &kind.shape(0)).unwrap(),
                _ => seismic::Tensor::from_host(&seismic::Charges::tally(device), element, &kind.shape(count), &bytes).unwrap(),
            }
        })
    }

    /// `routed_land` over these records at K choices.
    fn land_kernel(&self, device: &seismic::Device, k: u64) -> seismic::NativeKernel<routed_land::Entry> {
        let [gate, up, down] = [0, 1, 2].map(|tensor| self.kind(tensor));
        declared::prepared(
            device,
            &routed_land::native_entry_with(routed_land::Elements {
                SG: gate.source,
                SU: up.source,
                SD: down.source,
                UG: gate.resident,
                UU: up.resident,
                UD: down.resident,
            }),
            &specialize(&[("K", k)], &[]),
        )
        .unwrap()
    }

    /// `routed_round` over these records.
    fn round_kernel(&self, device: &seismic::Device) -> seismic::NativeKernel<routed_round::Entry> {
        let [gate, up, down] = [0, 1, 2].map(|tensor| self.kind(tensor));
        declared::prepared(
            device,
            &routed_round::native_entry_with(routed_round::Elements {
                SG: gate.source,
                SU: up.source,
                SD: down.source,
                UG: gate.resident,
                UU: up.resident,
                UD: down.resident,
            }),
            &seismic::NativeSpecialization::new(),
        )
        .unwrap()
    }
}

/// Round 0 of step STREAMING_SEQUENCE (`step`) whose host served it before
/// the call: every entry of `landing` appended and its lane answered, so
/// every backend runs it at once (`call_answered`), from `conveyor` into
/// `scratch`.
fn served_round(
    device: &seismic::Device,
    kernel: &seismic::NativeKernel<routed_round::Entry>,
    step: &seismic::Tensor,
    landing: &mut seismic::Tensor,
    conveyor: &[seismic::Tensor; 3],
    scratch: &mut [seismic::Tensor; 3],
) {
    let charges = seismic::Charges::tally(device);
    let q = STREAMING_SEQUENCE as u64;
    let table = SlotTable::scratch(FLIGHT, 1);
    let mut words = vec![0; table.words() as usize];
    set_pair(&mut words, table.lane(q, constants::ROUND_LANE(0)) + constants::SLOT_ANSWER, q);
    let [gate, up, down] = scratch.each_mut();
    kernel
        .call_answered(&charges, routed_round::Args {
            step,
            scratch: &mut shared_i32(device, &table.extents(), &words),
            window: &i32_tensor(device, &[1], &[0]),
            rounds: 1,
            round: 0,
            landing,
            tiered: &host_words(device, &[constants::LIST_ENTRY(1)]),
            conveyor_gate: &conveyor[0],
            conveyor_up: &conveyor[1],
            conveyor_down: &conveyor[2],
            gate,
            up,
            down,
            seeds: &host_words(device, &[constants::LIST_ENTRY(1)]),
            map: &host_words(device, &[1, constants::MAP_WORDS]),
            grid: &i32_tensor(device, &[3], &[constants::LAND_GROUPS as i32, 1, 1]),
        })
        .unwrap();
}

/// The landing launches build at the 35B's record forms: Q4_K and Q5_K
/// stored, their resident forms in every resident layout.
#[test]
fn streaming_landing_entries_build_at_35b_record_forms() {
    let module = crate::common::kernels();
    for name in ["routed_land", "routed_round"] {
        let entry = module.entry_named(name).unwrap();
        for layout in RESIDENT_LAYOUTS {
            let [gate, up, down] = StreamedKind::qwen35b_in(layout);
            let bind = |name: &str, element: seismic::Element| (name.to_string(), registry::representation(element.name()).unwrap());
            let bindings = [
                bind("SG", gate.source),
                bind("SU", up.source),
                bind("SD", down.source),
                bind("UG", gate.resident),
                bind("UU", up.resident),
                bind("UD", down.resident),
            ]
            .into_iter()
            .fold(ElementBindings::new(), |bindings, (name, representation)| bindings.bind(&name, representation));
            module.entry(entry, &bindings).unwrap_or_else(|error| panic!("{name} at {layout:?}: {error}"));
        }
    }
}

/// The launch grid `routed_list_grid` writes of `list` for the step, of a
/// list's whole grid's blocks (`LIST_BLOCKS`, which the host writes), and
/// the list it stages in device memory, which a copy on the grid reads.
fn list_grid(device: &seismic::Device, list: &seismic::Tensor, step: &seismic::Tensor) -> (seismic::Tensor, seismic::Tensor) {
    let charges = seismic::Charges::tally(device);
    let mut grid = i32_tensor(device, &[3], &[magnitude_kernels::LIST_BLOCKS as i32, 0, 0]);
    let mut staged = i32_tensor(device, &list.extents(), &vec![0; list.extents()[0] as usize]);
    declared::prepared(device, &routed_list_grid::native_entry(), &seismic::NativeSpecialization::new())
        .unwrap()
        .call(&charges, routed_list_grid::Args { list, step, grid: &mut grid, staged: &mut staged })
        .unwrap();
    (grid, staged)
}

/// On every backend that streams, `routed_list_grid` writes a list's launch
/// grid: one row of the blocks its host wrote (`grid[0]`, kept) while the
/// step's list has entries, and no row for a list of another step or of no
/// entries; and stages the list in device memory: its sequence, its count
/// of the step's entries and those entries.
#[test]
fn streaming_list_grid_has_a_row_an_entry_of_the_steps_list() {
    let Some(device) = streaming_device() else {
        return;
    };
    let step = streaming_step(&device);
    let blocks = magnitude_kernels::LIST_BLOCKS as i32;
    let pairs = (0..12).map(|entry| (entry, entry)).collect::<Vec<_>>();
    for (sequence, count, expected, staged_count) in [
        (STREAMING_SEQUENCE, 1, 1, 1),
        (STREAMING_SEQUENCE, 12, 1, 12),
        (STREAMING_SEQUENCE, 0, 0, 0),
        (STREAMING_SEQUENCE - 1, 3, 0, 0),
    ] {
        let words = list_words(sequence, count, &pairs, pairs.len());
        let list = shared_i32(&device, &[words.len() as u64], &words);
        let (grid, staged) = list_grid(&device, &list, &step);
        assert_eq!(read_i32(&grid), [blocks, expected, 1], "{count} entries of step {sequence}");
        let staged = read_i32(&staged);
        let entries = constants::LIST_ENTRY(staged_count as u64) as usize;
        let mut expected = words[..entries].to_vec();
        expected[constants::LIST_COUNT as usize] = staged_count;
        assert_eq!(&staged[..entries], expected.as_slice(), "the staged list of {count} entries of step {sequence}");
    }
}

/// On every backend that streams, `routed_pass_map` resolves a layer's map
/// and window row at the step into device words: each expert's source for the
/// window (`held` where `window[e] == -2`, `-2 - s` where `window[e] == s >=
/// 0`, else `-1` even when `map` holds a record), mapped from the step on,
/// past the 256 a block's threads cover at once; and the step.
#[test]
fn streaming_pass_map_resolves_each_experts_record_for_the_step() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let step = streaming_step(&device);
    let (experts, words) = (300u64, constants::MAP_WORDS as usize);
    let q = STREAMING_SEQUENCE as u64;
    let (from, until, record) = (constants::MAP_FROM as usize, constants::MAP_UNTIL as usize, constants::MAP_RECORD as usize);
    // By expert mod 6: bank hit (`window == -2`) live, future or expired;
    // cold scratch slot (`window == s >= 0`) unmapped or overriding a live
    // map row; and outside the window (`window == -1`) overriding a live row.
    let place = |expert: u64| match expert % 6 {
        0..=2 => -2,
        3 | 4 => (expert % 17) as i32,
        _ => -1,
    };
    let row = |expert: u64| {
        let (start, end, held) = match expert % 6 {
            0 => (1, u64::MAX, 2 * expert as i32),
            1 => (q + 1, u64::MAX, 5),
            2 => (1, q, 6),
            3 => (0, u64::MAX, -1),
            4 => (1, u64::MAX, 99),
            _ => (1, u64::MAX, 77),
        };
        let mut words = vec![0; words];
        words[from..from + 2].copy_from_slice(&[start as u32 as i32, (start >> 32) as u32 as i32]);
        words[until..until + 2].copy_from_slice(&[end as u32 as i32, (end >> 32) as u32 as i32]);
        words[record] = held;
        words
    };
    let map = shared_i32(&device, &[experts, words as u64], &(0..experts).flat_map(row).collect::<Vec<_>>());
    let window = shared_i32(&device, &[experts], &(0..experts).map(place).collect::<Vec<_>>());
    let mut resolved = i32_tensor(&device, &[experts, words as u64], &vec![0; experts as usize * words]);
    let mut pass_step = i32_tensor(&device, &[constants::STEP_WORDS], &[0; constants::STEP_WORDS as usize]);
    declared::prepared(&device, &routed_pass_map::native_entry(), &seismic::NativeSpecialization::new())
        .unwrap()
        .call(&charges, routed_pass_map::Args { map: &map, window: &window, step: &step, resolved: &mut resolved, pass_step: &mut pass_step })
        .unwrap();
    assert_eq!(read_i32(&pass_step), step_words(q), "the pass's step");
    let resolved = read_i32(&resolved);
    for expert in 0..experts {
        let at = expert as usize * words;
        let expected = match expert % 6 {
            0 => 2 * expert as i32,
            3 | 4 => -2 - (expert % 17) as i32,
            _ => -1,
        };
        assert_eq!(&resolved[at + from..at + from + 2], &step_words(q)[..2], "expert {expert} from the step");
        assert_eq!(&resolved[at + until..at + until + 2], &[-1, -1], "expert {expert} while mapped");
        assert_eq!(resolved[at + record], expected, "expert {expert}'s record for the step");
    }
}

/// On every backend that streams, `routed_copy` copies every entry of a list
/// longer than the chunk a group reads at once (`STREAMING_LIST_CHUNK`, 32):
/// 72 records, record r into record 71 - r of another tensor, each byte for
/// byte, on its list's grid.
#[test]
fn streaming_copy_walks_a_list_past_its_first_chunk() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (records, per) = (72usize, 1024u64);
    let byte = seismic::Element::bool();
    let step = streaming_step(&device);
    let pattern = |record: usize| (0..per as usize).map(|index| (index * 7 + record * 13) as u8).collect::<Vec<_>>();
    let source = seismic::Tensor::from_host(&charges, byte, &[records as u64, 1, per], &(0..records).flat_map(pattern).collect::<Vec<_>>())
        .unwrap();
    let mut destination = seismic::Tensor::zeros(&charges, byte, &[records as u64, 1, per]).unwrap();
    let pairs = (0..records as i32).map(|record| (record, records as i32 - 1 - record)).collect::<Vec<_>>();
    let words = list_words(STREAMING_SEQUENCE, records, &pairs, records);
    let list = shared_i32(&device, &[words.len() as u64], &words);
    let (grid, staged) = list_grid(&device, &list, &step);
    declared::prepared(&device, &routed_copy::native_entry_with(routed_copy::Elements { U: byte }), &seismic::NativeSpecialization::new())
        .unwrap()
        .call(&charges, routed_copy::Args { source: &source, list: &staged, step: &step, destination: &mut destination, grid: &grid })
        .unwrap();
    let bytes = destination.read_to_host().unwrap();
    for record in 0..records {
        let copied = &bytes[record * per as usize..(record + 1) * per as usize];
        assert!(copied == pattern(records - 1 - record).as_slice(), "record {record}");
    }
}

/// A prompt fill's tier copy (`routed_copy` on its list's grid) as a graph
/// run of its own: `records` records of `per` bytes in a host tier (every
/// byte `record % 251`), `copied` of them listed (record r into slot r) for
/// the step, into device slots, the copy reading the tier in place over the
/// link as a bank's fill does. The copy waits at a host gate each run binds.
struct TierFill {
    plan: seismic::NativeGraphPlan,
    gate: seismic::NativeTimelinePort<seismic::timeline::Gate>,
    ports: [seismic::NativePort; 5],
    tensors: [seismic::Tensor; 5],
    _tier: seismic::SlabTensor,
}

impl TierFill {
    /// On its list's grid where `blocks` is none, else on `blocks` blocks
    /// (the copy strides over its list's parts on any grid).
    fn new(device: &seismic::Device, records: u64, per: u64, copied: usize, blocks: Option<i32>) -> Self {
        let charges = seismic::Charges::tally(device);
        let byte = seismic::Element::bool();
        let shape = [records, 1, per];
        let bytes = (0..records).flat_map(|record| std::iter::repeat_n((record % 251) as u8, per as usize)).collect::<Vec<_>>();
        let pool = record_pool(device, seismic::Contract::Shared, records, &[record_layout(byte, &shape)]);
        let (tier, store) = record_store(&charges, &pool, byte, &shape, &bytes);
        let destination = seismic::Tensor::zeros(&charges, byte, &shape).unwrap();
        let step = streaming_step(device);
        let pairs = (0..copied as i32).map(|record| (record, record)).collect::<Vec<_>>();
        let list = shared_i32(device, &[constants::LIST_ENTRY(records)], &list_words(STREAMING_SEQUENCE, copied, &pairs, records as usize));
        let (grid, staged) = list_grid(device, &list, &step);
        let grid = match blocks {
            None => grid,
            Some(blocks) => i32_tensor(device, &[3], &[blocks, 1, 1]),
        };
        let copy = declared::prepared(device, &routed_copy::native_entry_with(routed_copy::Elements { U: byte }), &seismic::NativeSpecialization::new()).unwrap();
        let mut graph = device.native_graph();
        let mut port = |tensor: &seismic::Tensor| graph.port(tensor.element(), &tensor.extents()).unwrap();
        let ports = [&tier, &staged, &step, &destination, &grid].map(&mut port);
        let [source, list_port, step_port, mut destination_port, grid_port] = ports.clone();
        let gate = graph.timeline(seismic::timeline::Wait);
        graph
            .enqueue(
                &copy,
                routed_copy::WorkflowArgs {
                    source: source.tensor().into(),
                    list: list_port.tensor().into(),
                    step: step_port.tensor().into(),
                    destination: destination_port.tensor_mut().into(),
                    grid: grid_port.tensor().into(),
                },
            )
            .unwrap();
        Self { plan: graph.seal().unwrap(), gate, ports, tensors: [tier, staged, step, destination, grid], _tier: store }
    }

    /// The slots it copies into.
    fn destination(&self) -> &seismic::Tensor {
        &self.tensors[3]
    }

    /// Submit one copy on `queue` from `slot`, held until the host raises
    /// `gate` to `value`, without waiting.
    fn submit(
        &self,
        device: &seismic::Device,
        slot: &mut seismic::NativeGraphSlot,
        queue: seismic::NativeQueue,
        gate: &seismic::Timeline<seismic::timeline::Gate>,
        value: u64,
    ) -> seismic::NativeGraphCompletion {
        let charges = seismic::Charges::tally(device);
        let mut bindings = self.plan.bindings();
        for (port, tensor) in self.ports.iter().zip(&self.tensors) {
            bindings.set(&charges, port, tensor).unwrap();
        }
        bindings.set_timeline(self.gate, gate, value).unwrap();
        let mut sequence = device.native_sequence(queue);
        slot.attach(bindings, self.plan.new_outputs(&charges).unwrap()).unwrap().queue(&mut sequence).unwrap();
        sequence.submit().unwrap()
    }
}

/// Native work on the compute queue that reads what a fill on the fill
/// queue writes runs after it (`Allocation::queue_order`): a copy of the
/// fill's slots on the compute queue, submitted at once behind a 64-record
/// fill of 1.9 MB records, finds every slot filled. Where the backend runs
/// fills on the compute queue the queue orders them.
#[test]
fn compute_reading_a_fills_slots_follows_the_fill() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (records, per) = (64u64, 1_900_544u64);
    let fill = TierFill::new(&device, records, per, records as usize, None);
    let byte = seismic::Element::bool();
    let mut read = seismic::Tensor::zeros(&charges, byte, &[records, 1, per]).unwrap();
    let step = &fill.tensors[2];
    let pairs = (0..records as i32).map(|record| (record, record)).collect::<Vec<_>>();
    let list = shared_i32(&device, &[constants::LIST_ENTRY(records)], &list_words(STREAMING_SEQUENCE, records as usize, &pairs, records as usize));
    let (grid, staged) = list_grid(&device, &list, step);
    let copy = declared::prepared(&device, &routed_copy::native_entry_with(routed_copy::Elements { U: byte }), &seismic::NativeSpecialization::new()).unwrap();
    let mut slot = fill.plan.new_slot(&charges).unwrap();
    let open = seismic::Timeline::new(&device).unwrap();
    open.signal(1).unwrap();
    let filled = fill.submit(&device, &mut slot, seismic::NativeQueue::Fill, &open, 1);
    copy.call(&charges, routed_copy::Args { source: fill.destination(), list: &staged, step, destination: &mut read, grid: &grid })
        .unwrap();
    filled.wait().unwrap();
    let bytes = read.read_to_host().unwrap();
    for record in 0..records as usize {
        let slot = &bytes[record * per as usize..(record + 1) * per as usize];
        assert!(slot.iter().all(|&byte| byte == (record % 251) as u8), "{:?}: slot {record} read before its fill", device.backend());
    }
}

/// Layer 0's front of a chain place writes its own step words
/// (`routed_step`), an allocation of its own (the bank's
/// `PlacedAllocations::step`), and so waits for no fill of another place
/// reading its own: where fills run beside compute, the write returns while
/// a fill on the fill queue is held at its gate, so has provably not run.
/// Writing the fill's own step words instead waits for the fill, as a read
/// must be followed: it has not returned while the gate holds the fill, and
/// returns with the fill complete once it opens. The device orders queues by
/// what each submission touches.
#[test]
fn a_fronts_step_words_wait_for_no_other_places_fill() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    if !device.backend().fills_beside_compute() {
        return;
    }
    let fill = TierFill::new(&device, 4, 1024, 4, None);
    let mut slot = fill.plan.new_slot(&charges).unwrap();
    let step = declared::prepared(&device, &routed_step::native_entry(), &seismic::NativeSpecialization::new()).unwrap();
    let words = step_words(STREAMING_SEQUENCE as u64);
    let sequence = i32_tensor(&device, &[constants::STEP_WORDS], &words);
    let mut other = i32_tensor(&device, &[constants::STEP_WORDS], &[0; constants::STEP_WORDS as usize]);
    let none = no_tokens(&device);
    let gate = seismic::Timeline::new(&device).unwrap();
    let filled = fill.submit(&device, &mut slot, seismic::NativeQueue::Fill, &gate, 1);
    // A call that waited for the fill would return only once the window
    // passed and the gate opened: the window bounds the failure, not the pass.
    let returned = returned_while_held(&gate, 1, std::time::Duration::from_secs(10), || {
        step.call(&charges, routed_step::Args { step: &mut other, sequence: &sequence, guesses: &none }).unwrap();
        assert!(!filled.is_complete(), "the fill ran before its gate opened");
    });
    assert!(returned, "another place's step words waited for the fill");
    assert_eq!(read_i32(&other), words, "another place's step words");
    filled.wait().unwrap();
    let filled = fill.submit(&device, &mut slot, seismic::NativeQueue::Fill, &gate, 2);
    let mut own = fill.tensors[2].clone();
    // An ordered call cannot return while the gate is held, whatever the
    // window; one that did not wait returns within it.
    let returned = returned_while_held(&gate, 2, std::time::Duration::from_millis(200), || {
        step.call(&charges, routed_step::Args { step: &mut own, sequence: &sequence, guesses: &none }).unwrap();
    });
    assert!(!returned, "the fill's own step words were written while its gate held it");
    assert!(filled.is_complete(), "the fill's own step words were written while it read them");
    filled.wait().unwrap();
}

/// Run `call` on this thread while another holds `gate` below `value` until
/// `call` returns or `within` passes, whichever is first, then raises it:
/// whether `call` returned while the gate was held.
fn returned_while_held(gate: &seismic::Timeline<seismic::timeline::Gate>, value: u64, within: std::time::Duration, call: impl FnOnce()) -> bool {
    let (tell, heard) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let opener = scope.spawn(move || {
            let returned = heard.recv_timeout(within).is_ok();
            gate.signal(value).unwrap();
            returned
        });
        call();
        // The opener stops listening once the window passes.
        let _ = tell.send(());
        opener.join().unwrap()
    })
}

/// The guesses of a step whose graph runs no front row (`routed_step`'s
/// `guesses` at T = 0).
fn no_tokens(device: &seismic::Device) -> seismic::Tensor {
    seismic::Tensor::zeros(&seismic::Charges::tally(device), seismic::Element::i32(), &[0, 2]).unwrap()
}

/// The grid a route writes of a step that requested its file slot
/// (`resolve`'s `land_grid`): every landing group where the bank places
/// (`placed`), the one where the host tier places.
fn land_grid(device: &seismic::Device, placed: bool) -> seismic::Tensor {
    i32_tensor(device, &[3], &[magnitude_kernels::land_groups(placed), 1, 1])
}

/// The step word `routed_step` sets from `STREAMING_SEQUENCE`, which it
/// holds once the launch returns.
fn streaming_step(device: &seismic::Device) -> seismic::Tensor {
    let charges = seismic::Charges::tally(device);
    let mut step = i32_tensor(device, &[constants::STEP_WORDS], &[0; constants::STEP_WORDS as usize]);
    declared::prepared(device, &routed_step::native_entry(), &seismic::NativeSpecialization::new())
        .unwrap()
        .call(&charges, routed_step::Args {
            step: &mut step,
            sequence: &i32_tensor(device, &[constants::STEP_WORDS], &step_words(STREAMING_SEQUENCE as u64)),
            guesses: &no_tokens(device),
        })
        .unwrap();
    assert_eq!(read_i32(&step), step_words(STREAMING_SEQUENCE as u64), "step");
    step
}

/// The copy case: S bank records (record s holds pattern s's resident
/// import) and D scratch slots.
const COPY_RECORDS: (u64, u64) = (3, 4);

/// On every backend that streams, `routed_copy` copies each listed (record,
/// slot) pair of its step byte for byte, at the 35B's record forms: record
/// 2 into slot 0 and record 0 into slot 3; an entry past the count is
/// skipped; a list for another step copies nothing.
#[test]
fn streaming_copy_copies_the_listed_records_byte_for_byte() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let pairs = [(2, 0), (0, 3), (1, 2)];
    let count = 2;
    let step = streaming_step(&device);
    for form in RECORD_FORMS {
        let kind = StreamedKind::qwen35b(&device)[form];
        let imports = (0..COPY_RECORDS.0 as usize)
            .map(|pattern| kind.import(&kind.stored(pattern, form)))
            .collect::<Vec<_>>();
        let bank = kind.tensor(&device, kind.resident, COPY_RECORDS.0, |record| Some(imports[record].as_slice()));
        let copy = declared::prepared(
            &device,
            &routed_copy::native_entry_with(routed_copy::Elements { U: kind.resident }),
            &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        for (sequence, copied) in [(STREAMING_SEQUENCE, true), (STREAMING_SEQUENCE - 1, false)] {
            let label = format!("{}, copied {copied}", kind.resident.name());
            let list = list_words(sequence, count, &pairs, pairs.len());
            let mut scratch = kind.tensor(&device, kind.resident, COPY_RECORDS.1, |_| None);
            let list = shared_i32(&device, &[list.len() as u64], &list);
            let (grid, staged) = list_grid(&device, &list, &step);
            copy.call(&charges, routed_copy::Args { source: &bank, list: &staged, step: &step, destination: &mut scratch, grid: &grid })
                .unwrap();
            let bytes = scratch.read_to_host().unwrap();
            let width = kind.slot_bytes();
            for slot in 0..COPY_RECORDS.1 as usize {
                let slot_bytes = &bytes[slot * width..(slot + 1) * width];
                match (copied, slot) {
                    (true, 0) => assert!(slot_bytes == imports[2], "{label}: slot 0"),
                    (true, 3) => assert!(slot_bytes == imports[0], "{label}: slot 3"),
                    _ => assert!(slot_bytes.iter().all(|&byte| byte == 0), "{label}: slot {slot} written"),
                }
            }
        }
    }
}

/// On every backend that streams, a conveyor in device record stores (as
/// bank records are, the host writing them as a transfer does)
/// lands bit for bit as one in host-shared records: the 35B's gated
/// records of four experts landed from each give one scratch tier, every
/// slot its expert's resident import.
#[test]
fn streaming_a_record_store_conveyor_lands_as_a_host_shared_one() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let experts = 4u64;
    let step = streaming_step(&device);
    let form = ServedForm::qwen35b(&device);
    let kernel = form.round_kernel(&device);
    let shared: [seismic::Tensor; 3] = std::array::from_fn(|tensor| {
        let kind = form.kind(tensor);
        let stored = (0..experts as usize).map(|expert| form.stored(expert, tensor)).collect::<Vec<_>>();
        shared_bytes(&device, kind.source, &kind.shape(experts), &stored.concat())
    });
    let layouts = [0, 1, 2].map(|tensor| record_layout(form.kind(tensor).source, &form.kind(tensor).shape(experts)));
    let pool = record_pool(&device, seismic::Contract::DeviceOnly, experts, &layouts);
    let recorded = [0, 1, 2].map(|tensor| {
        let kind = form.kind(tensor);
        let stored = (0..experts as usize).map(|expert| form.stored(expert, tensor)).collect::<Vec<_>>();
        record_store(&charges, &pool, kind.source, &kind.shape(experts), &stored.concat())
    });
    let landed = |conveyor: &[seismic::Tensor; 3]| {
        let pairs = [(0, 0), (1, 1), (2, 2), (3, 3)];
        let mut landing = shared_i32(&device, &[constants::LANDING_ENTRY(4)], &landing_words(STREAMING_SEQUENCE, 4, &pairs, 4));
        let mut scratch = form.records(&device, true, experts, |_, _| None);
        served_round(&device, &kernel, &step, &mut landing, conveyor, &mut scratch);
        scratch.map(|slots| slots.read_to_host().unwrap())
    };
    let from_shared = landed(&shared);
    let [gate, up, down] = &recorded;
    assert!(
        landed(&[gate.0.clone(), up.0.clone(), down.0.clone()]) == from_shared,
        "the record-store conveyor lands as the shared one"
    );
    for (tensor, slots) in from_shared.iter().enumerate() {
        let width = form.kind(tensor).slot_bytes();
        assert!(
            slots.chunks(width).enumerate().all(|(expert, slot)| slot == form.import(expert, tensor)),
            "tensor {tensor}: every slot its expert's import"
        );
    }
}

/// On every backend that streams, a prompt group's windowed passes read each
/// hit from its bank record (in window 0) and stream 8 cold experts across two
/// windows of a bounded 5-slot scratch tier (`W = 5 < 8 < E = 16`), reusing
/// slots 0..3 in window 1 while skipping experts outside each window into one
/// persistent output: at the 35B's record forms, gated, the combined output's
/// rows equal the resident pass's over the same 16 experts bit for bit.
#[test]
fn streaming_pass_reads_cold_experts_from_a_scratch_tier() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let kinds = StreamedKind::qwen35b(&device);
    let (h, f, k, t) = (Qwen35b::HIDDEN as u64, Qwen35b::FEATURES as u64, Qwen35b::CHOICES as u64, Qwen35b::TILE as u64);
    let (experts, rows, window_slots) = (16u64, 32u64, 5u64);
    // Even experts are hits, expert x in bank record H - 1 - x / 2; odd ones
    // are cold (8 experts), conveyor record v holding the v-th, served in two
    // windows of at most 5 scratch slots (5 in window 0, 3 reusing slots 0..3
    // in window 1).
    let hits = experts / 2;
    let bank_record = |expert: u64| hits - 1 - expert / 2;
    let cold = (0..experts).filter(|expert| expert % 2 == 1).collect::<Vec<_>>();
    let step = streaming_step(&device);
    let q = STREAMING_SEQUENCE as u64;

    // The bank's and the bounded scratch tier's records of every kind lie in
    // one pool.
    let layouts = kinds.iter().map(|kind| record_layout(kind.resident, &kind.shape(experts))).collect::<Vec<_>>();
    let pool = record_pool(&device, seismic::Contract::DeviceOnly, window_slots + hits, &layouts);

    let mut reference = Vec::new();
    let mut streamed = Vec::new();
    let mut conveyors = Vec::new();
    for (index, kind) in kinds.iter().enumerate() {
        let stored = (0..experts as usize).map(|expert| kind.stored(expert, index)).collect::<Vec<_>>();
        let imports = stored.iter().map(|bytes| kind.import(bytes)).collect::<Vec<_>>();
        reference.push(kind.tensor(&device, kind.resident, experts, |expert| Some(imports[expert].as_slice())));

        let mut bank_bytes = vec![Vec::new(); hits as usize];
        for expert in (0..experts).step_by(2) {
            bank_bytes[bank_record(expert) as usize] = imports[expert as usize].clone();
        }
        let bank = record_store(&charges, &pool, kind.resident, &kind.shape(hits), &bank_bytes.concat());
        let scratch = record_store(&charges, &pool, kind.resident, &kind.shape(window_slots), &vec![0u8; window_slots as usize * kind.slot_bytes()]);

        conveyors.push(kind.tensor(&device, kind.source, cold.len() as u64, |v| Some(stored[cold[v] as usize].as_slice())));
        streamed.push((bank, scratch));
    }

    // The layer's map: each hit held in its bank record since step 1, each
    // cold expert mapped from the step after.
    let (from, until, at) = (constants::MAP_FROM as usize, constants::MAP_UNTIL as usize, constants::MAP_RECORD as usize);
    let words = constants::MAP_WORDS as usize;
    let row = |expert: u64| {
        let (start, held) = match expert % 2 {
            0 => (1, bank_record(expert) as i32),
            _ => (q + 1, -1),
        };
        let mut row = vec![0; words];
        row[from..from + 2].copy_from_slice(&[start as u32 as i32, (start >> 32) as u32 as i32]);
        row[until..until + 2].copy_from_slice(&[-1, -1]);
        row[at] = held;
        row
    };
    let map = shared_i32(&device, &[experts, words as u64], &(0..experts).flat_map(row).collect::<Vec<_>>());
    let mut resolved = i32_tensor(&device, &[experts, words as u64], &vec![0; experts as usize * words]);
    let mut pass_step = i32_tensor(&device, &[constants::STEP_WORDS], &[0; constants::STEP_WORDS as usize]);
    let pass_map = declared::prepared(&device, &routed_pass_map::native_entry(), &seismic::NativeSpecialization::new()).unwrap();
    let round_kernel = ServedForm::qwen35b(&device).round_kernel(&device);

    // Rows of 8 distinct choices, hits and cold experts mixed: choice c of
    // row r is expert (r + 3 c) mod E.
    let routes = (0..rows).flat_map(|row| (0..k).map(move |choice| ((row + 3 * choice) % experts) as i32)).collect::<Vec<_>>();
    let blocks = group_blocks(rows * k, experts.min(rows * k), t);
    let fill = |shape: &[u64]| i32_tensor(&device, shape, &vec![-9; shape.iter().product::<u64>() as usize]);
    let (grouped, mut order, mut inverse, mut table, mut used) =
        (i32_tensor(&device, &[1], &[rows as i32]), fill(&[blocks, t]), fill(&[rows, k]), fill(&[blocks]), fill(&[1]));
    declared::prepared(&device, &routed_group::native_entry(), &default_group_specialization(&device, experts, k))
        .unwrap()
        .call(&charges, routed_group::Args {
            first: &first_row(&device),
            rows: &grouped,
            routes: &i32_tensor(&device, &[rows, k], &routes),
            counts: &mut fill(&[experts]),
            order: &mut order,
            inverse: &mut inverse,
            blocks: &mut table,
            used: &mut used,
        })
        .unwrap();
    let normalized = bf16_tensor(&device, &[rows, h], &pattern((rows * h) as usize, 93, 1.0));
    let pass = grouped_projection(
        routed_experts::native_entry_with(routed_experts::Elements {
            A: seismic::Element::bf16(),
            EGW: kinds[0].resident,
            EUW: kinds[1].resident,
            EDW: kinds[2].resident,
        }),
        vec![("H", h), ("F", f), ("T", t), ("GS", 0), ("US", 0), ("LB", 0)],
        0,
    )
    .kernel(&device, Grouped { mapping: &grouped_mappings(&device)[0], form: &GroupedForm::default() });
    let scale = absent(&device);
    let gather_bits = |output: &seismic::Tensor| {
        let values = read_bf16(output);
        read_i32(&inverse)
            .iter()
            .flat_map(|&position| values[position as usize * h as usize..(position as usize + 1) * h as usize].iter().map(|value| value.to_bits()))
            .collect::<Vec<_>>()
    };
    let run_into = |expert: [&seismic::Tensor; 3],
                    map: &seismic::Tensor,
                    step: &seismic::Tensor,
                    records: [&seismic::Tensor; 3],
                    output: &mut seismic::Tensor| {
        pass.call_into(
            &charges, routed_experts::Args {
                first: &first_row(&device),
                rows: &grouped,
                normalized: &normalized,
                order: &order,
                blocks: &table,
                used: &used,
                expert_gate: expert[0],
                expert_up: expert[1],
                expert_down: expert[2],
                map,
                step,
                records_gate: records[0],
                records_up: records[1],
                records_down: records[2],
                activation: 0,
                glu_limit: &no_limit(&device),
                gate_scale: &scale,
                up_scale: &scale,
            },
            routed_experts::OutputArgs { value: output },
        )
        .unwrap();
    };
    let resident = [&reference[0], &reference[1], &reference[2]];
    let mut resident_output = bf16_tensor(&device, &[blocks, t, h], &vec![0.0; (blocks * t * h) as usize]);
    run_into(resident, &absent_map(&device), &absent_step(&device), resident, &mut resident_output);
    let expected = gather_bits(&resident_output);

    let conveyor: [seismic::Tensor; 3] = std::array::from_fn(|tensor| conveyors[tensor].clone());
    let mut scratch: [seismic::Tensor; 3] = std::array::from_fn(|tensor| streamed[tensor].1.0.clone());
    let mut windowed_output = bf16_tensor(&device, &[blocks, t, h], &vec![-9.0; (blocks * t * h) as usize]);
    for (w, batch) in cold.chunks(window_slots as usize).enumerate() {
        let base = w * window_slots as usize;
        let pairs = (0..batch.len()).map(|slot| ((base + slot) as i32, slot as i32)).collect::<Vec<_>>();
        let mut landing =
            shared_i32(&device, &[constants::LANDING_ENTRY(pairs.len() as u64)], &landing_words(STREAMING_SEQUENCE, pairs.len(), &pairs, pairs.len()));
        served_round(&device, &round_kernel, &step, &mut landing, &conveyor, &mut scratch);

        let mut window_row = vec![-1i32; experts as usize];
        if w == 0 {
            for expert in (0..experts as usize).step_by(2) {
                window_row[expert] = -2;
            }
        }
        for (slot, &expert) in batch.iter().enumerate() {
            window_row[expert as usize] = slot as i32;
        }
        let window_tensor = shared_i32(&device, &[experts], &window_row);
        pass_map
            .call(&charges, routed_pass_map::Args { map: &map, window: &window_tensor, step: &step, resolved: &mut resolved, pass_step: &mut pass_step })
            .unwrap();
        run_into(
            [&scratch[0], &scratch[1], &scratch[2]],
            &resolved,
            &pass_step,
            [&streamed[0].0.0, &streamed[1].0.0, &streamed[2].0.0],
            &mut windowed_output,
        );
    }
    let actual = gather_bits(&windowed_output);
    assert!(actual == expected, "the windowed streamed passes' rows differ from the resident pass's");
}

/// On every backend that streams, a round lands its records byte-equal to a
/// resident import (`assert_round_lands`), at the 35B's gated records (Q4_K
/// and Q5_K), DeepSeek V4.1's MXFP4 records (17-byte packets: gate/up rows
/// of 2720 bytes, wider than a window holds a row tile of, and down rows of
/// 1224 bytes, whose row tile a window holds whole), each also at a row
/// count no multiple of 16 (`ServedForm::v41_mxfp4_partial`: a partial last
/// row tile, whole at a 1224-byte row pitch, no multiple of 16, and records
/// of 8 bytes off 16-byte alignment), and Nemotron 3.5
/// Lightning's up-only records (Q5_0 and Q8_0, whose windows load in
/// narrower units), and the 35B q8's dense BF16 records ([H 2048, F 512]),
/// into BF16 (the bytes themselves) and F32 (each element widened); and
/// dense BF16 records of 139 by 133, whose row counts are no multiple of
/// 16 (a partial last row tile) and whose records are no multiple of 16
/// bytes (each after the first starts off 16-byte alignment), as no
/// quantized record of whole packets on both axes can be.
#[test]
fn streaming_rounds_land_records_like_a_resident_import() {
    let Some(device) = streaming_device() else {
        return;
    };
    let (bf16, f32) = (seismic::Element::bf16(), seismic::Element::f32());
    let [partial_gate, partial_down] = ServedForm::v41_mxfp4_partial(&device);
    let mut forms = vec![
        ("35B", ServedForm::qwen35b(&device)),
        ("V4.1 MXFP4", ServedForm::v41_mxfp4(&device)),
        ("V4.1 MXFP4 gate and up of 133 rows", partial_gate),
        ("V4.1 MXFP4 down of 139 rows", partial_down),
        ("Lightning", ServedForm::lightning(&device)),
        ("dense BF16", ServedForm::dense(false, bf16, 512, 2048)),
        ("dense F32", ServedForm::dense(false, f32, 512, 2048)),
        ("odd dense BF16", ServedForm::dense(false, bf16, 139, 133)),
    ];
    // Planar FP8 records (a source under a block-scale grid): every
    // backend's window reads their two planes.
    forms.extend([
        ("GLM FP8 128x128 F32", ServedForm::glm_fp8(&device, "gguf_f8_e4m3_f32_128x128", "fp8g128")),
        ("GLM FP8 32x32 E8M0", ServedForm::glm_fp8(&device, "gguf_f8_e4m3_e8m0_32x32", "mxfp8g32")),
        ("GLM FP8 1x32 E8M0", ServedForm::glm_fp8(&device, "gguf_f8_e4m3_e8m0_1x32", "mxfp8g32")),
    ]);
    // `LANDING_FORM=<text>` lands only the forms whose name holds it (to run
    // one form under a sanitizer).
    let only = std::env::var("LANDING_FORM").unwrap_or_default();
    for (name, form) in forms.into_iter().filter(|(name, _)| name.contains(&only)) {
        eprintln!("landing form {name}");
        assert_round_lands(&device, &form, name);
    }
}

/// A round's landing of `form`'s records byte-equal to a resident import:
/// the conveyor a view at an offset into a larger stored tensor (as lent
/// bank records are), conveyor record v holding expert v; its entries out
/// of record order, record 2 into slot 2, then (where every stored record
/// fits a slot) scratch slot 3's stored bytes of expert 0 into slot 1, then
/// record 1 into slot 0; an entry past the count lands nothing; each landed
/// entry's DONE holds the step's low word; a list another step's head names
/// lands nothing.
fn assert_round_lands(device: &seismic::Device, form: &ServedForm, name: &str) {
    let (v, s) = (3u64, 4u64);
    let step = streaming_step(device);
    let kernel = form.round_kernel(device);
    let held = form.records(device, false, v + 2, |tensor, record| {
        (1..=v as usize).contains(&record).then(|| form.stored(record - 1, tensor))
    });
    let conveyor: [seismic::Tensor; 3] = std::array::from_fn(|tensor| match form.gate.is_none() && tensor == 0 {
        true => held[tensor].clone(),
        false => held[tensor].slice_leading(1, 1 + v).unwrap(),
    });
    let lends = (0..3).all(|tensor| form.stored(0, tensor).len() <= form.kind(tensor).slot_bytes());
    let lent = |tensor: usize| {
        let mut bytes = vec![0u8; form.kind(tensor).slot_bytes()];
        let stored = form.stored(0, tensor);
        bytes[..stored.len()].copy_from_slice(&stored);
        bytes
    };
    let mut pairs = vec![(2, 2)];
    if lends {
        pairs.push((-1 - 3, 1));
    }
    pairs.extend([(1, 0), (0, 3)]);
    let count = pairs.len() - 1;
    for (low, landed) in [(STREAMING_SEQUENCE, true), (STREAMING_SEQUENCE - 1, false)] {
        let label = format!("{name}, landed {landed}");
        let mut landing =
            shared_i32(device, &[constants::LANDING_ENTRY(pairs.len() as u64)], &landing_words(low, count, &pairs, pairs.len()));
        let mut scratch = form.records(device, true, s, |tensor, slot| (lends && slot == 3).then(|| lent(tensor)));
        served_round(device, &kernel, &step, &mut landing, &conveyor, &mut scratch);
        for (tensor, slots) in scratch.iter().enumerate() {
            if form.gate.is_none() && tensor == 0 {
                continue;
            }
            let bytes = slots.read_to_host().unwrap();
            let width = form.kind(tensor).slot_bytes();
            for slot in 0..s as usize {
                let held = &bytes[slot * width..(slot + 1) * width];
                let expected = match (landed, slot) {
                    (true, 2) => Some(form.import(2, tensor)),
                    (true, 0) => Some(form.import(1, tensor)),
                    (true, 1) if lends => Some(form.import(0, tensor)),
                    (_, 3) if lends => Some(lent(tensor)),
                    _ => None,
                };
                match expected {
                    Some(expected) => assert!(held == expected.as_slice(), "{label}: tensor {tensor} slot {slot}"),
                    None => assert!(held.iter().all(|&byte| byte == 0), "{label}: tensor {tensor} slot {slot} written"),
                }
            }
        }
        let words = read_i32(&landing);
        for entry in 0..pairs.len() {
            let done = if landed && entry < count { low } else { 0 };
            assert_eq!(words[landing_done(entry)], done, "{label}: entry {entry}'s DONE");
        }
    }
}

/// A decode layer's landing as its host serves it: choice 0's expert 3 held
/// in bank record 0, choices 1-3's experts 9, 5 and 12 absent and read into
/// staging records 0, 1 and 2, whose reads land in the order of `LANDED`
/// (staging record, bank record, choice): not the record order.
const LANDED_EXPERTS: [usize; 4] = [3, 9, 5, 12];
const LANDED: [(i32, i32, u64); 3] = [(2, 5, 3), (0, 1, 1), (1, 4, 2)];

/// `form`'s staging records of `LANDED_EXPERTS`' absent experts.
fn landed_staging(device: &seismic::Device, form: &ServedForm) -> [seismic::Tensor; 3] {
    form.records(device, false, 3, |tensor, record| Some(form.stored(LANDED_EXPERTS[record + 1], tensor)))
}

/// On every backend that streams, a decode layer's landing converts each
/// absent expert's records as its read lands: a host thread appends the
/// entries out of record order and late (20 ms apart, while the launch
/// runs, where kernels park; Vulkan's host appends them before the park
/// opens), the landing list's head first naming an earlier step, then
/// answers each expert's choices and the slot. Every landed bank record
/// then holds its expert's resident import, byte for byte, every other
/// none; every entry's DONE holds the step's low word; and each absent
/// choice is sourced from its bank record. At
/// the 35B's, V4.1's MXFP4 and Lightning's up-only records.
#[test]
fn streaming_a_landing_converts_records_appended_late_out_of_order() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (q, low, k) = (STREAMING_SEQUENCE as u64, STREAMING_SEQUENCE, 4u64);
    let step = streaming_step(&device);
    let requested = [-1 - LANDED_EXPERTS[0] as i32, LANDED_EXPERTS[1] as i32, LANDED_EXPERTS[2] as i32, LANDED_EXPERTS[3] as i32];
    let in_kernels = device.backend().parks_in_kernels();
    for (name, form) in [
        ("35B", ServedForm::qwen35b(&device)),
        ("V4.1 MXFP4", ServedForm::v41_mxfp4(&device)),
        ("Lightning", ServedForm::lightning(&device)),
    ] {
        let kernel = form.land_kernel(&device, k);
        let staging = landed_staging(&device, &form);
        let mut bank = form.records(&device, true, 6, |_, _| None);
        let (table, words) = file_words(k, q, &requested, &[], None);
        let mut file = shared_i32(&device, &table.extents(), &words);
        let _pulse = pulse(&file, table);
        let mut landing = shared_i32(&device, &[constants::LANDING_ENTRY(3)], &landing_words(low - FLIGHT as i32, 3, &[], 3));
        let mut sources = decode_sources(&device, 1, k, &[0, -1, -1, -1]);
        let policy = shared_i32(&device, &[constants::POLICY_WORDS], &[0; constants::POLICY_WORDS as usize]);
        let slot = table.lane(q, 0);
        let serve = |file: &seismic::Tensor, landing: &seismic::Tensor| {
            for (index, &(staged, record, choice)) in LANDED.iter().enumerate() {
                if in_kernels {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                append_landing(landing, low, index, staged, record);
                file.store_release(4 * (slot + constants::FILE_ENTRY(choice) + constants::FILE_RECORD), record as u32).unwrap();
            }
            file.store_release_u64(4 * (slot + constants::SLOT_ANSWER), q).unwrap();
        };
        let (served, appended) = (file.clone(), landing.clone());
        let [gate, up, down] = bank.each_mut();
        let grid = land_grid(&device, true);
        let args = routed_land::Args {
            step: &step,
            file: &mut file,
            sources: &mut sources,
            policy: &policy,
            landing: &mut landing,
            staging_gate: &staging[0],
            staging_up: &staging[1],
            staging_down: &staging[2],
            gate,
            up,
            down,
            grid: &grid,
        };
        if in_kernels {
            std::thread::scope(|scope| {
                scope.spawn(|| serve(&served, &appended));
                kernel.call(&charges, args).unwrap();
            });
        } else {
            serve(&served, &appended);
            kernel.call_answered(&charges, args).unwrap();
        }
        for (tensor, records) in bank.iter().enumerate() {
            if form.gate.is_none() && tensor == 0 {
                continue;
            }
            let bytes = records.read_to_host().unwrap();
            let width = form.kind(tensor).slot_bytes();
            for record in 0..6 {
                let held = &bytes[record * width..(record + 1) * width];
                match LANDED.iter().find(|landed| landed.1 == record as i32) {
                    Some(&(staged, _, _)) => assert!(
                        held == form.import(LANDED_EXPERTS[staged as usize + 1], tensor).as_slice(),
                        "{name}: tensor {tensor} bank record {record}"
                    ),
                    None => assert!(held.iter().all(|&byte| byte == 0), "{name}: tensor {tensor} bank record {record} written"),
                }
            }
        }
        let words = read_i32(&landing);
        for entry in 0..LANDED.len() {
            assert_eq!(words[landing_done(entry)], low, "{name}: entry {entry}'s DONE");
        }
        assert_eq!(read_i32(&sources), magnitude_kernels::source_words(&[0, 1, 4, 5], &[-1; 4]), "{name}: sources");
    }
}

/// On every backend that streams, a landing stops once its slot is answered
/// with fewer entries than its threadgroups claim (one expert landed, two
/// reads failed: their choices sourced none), and once its slot is failed
/// (the entry landed before the failure converted, every absent choice
/// sourced none); each case twice, the second landing as the first, so
/// every launch restores its counters.
#[test]
fn streaming_a_landing_stops_once_its_slot_is_answered_or_failed() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (q, low, k) = (STREAMING_SEQUENCE as u64, STREAMING_SEQUENCE, 4u64);
    let step = streaming_step(&device);
    let form = ServedForm::qwen35b(&device);
    let kernel = form.land_kernel(&device, k);
    let staging = landed_staging(&device, &form);
    let requested = [-1 - LANDED_EXPERTS[0] as i32, LANDED_EXPERTS[1] as i32, LANDED_EXPERTS[2] as i32, LANDED_EXPERTS[3] as i32];
    let failed = q | (u64::from(constants::SLOT_FAILED as u32) << 32);
    // Staging record 0 (choice 1's expert) lands into bank record 1.
    for (case, answer, records) in [("answered", q, [0, 1, -1, -1]), ("failed", failed, [0, -1, -1, -1])] {
        for run in 0..2 {
            let label = format!("{case}, run {run}");
            let (table, words) = file_words(k, q, &requested, &[(1, 1), (2, -1), (3, -1)], Some(answer));
            let mut file = shared_i32(&device, &table.extents(), &words);
            let mut landing = shared_i32(&device, &[constants::LANDING_ENTRY(3)], &landing_words(low, 1, &[(0, 1)], 3));
            let mut sources = decode_sources(&device, 1, k, &[0, -1, -1, -1]);
            let mut bank = form.records(&device, true, 6, |_, _| None);
            let [gate, up, down] = bank.each_mut();
            kernel
                .call_answered(&charges, routed_land::Args {
                    step: &step,
                    file: &mut file,
                    sources: &mut sources,
                    policy: &shared_i32(&device, &[constants::POLICY_WORDS], &[0; constants::POLICY_WORDS as usize]),
                    landing: &mut landing,
                    staging_gate: &staging[0],
                    staging_up: &staging[1],
                    staging_down: &staging[2],
                    gate,
                    up,
                    down,
                    grid: &land_grid(&device, true),
                })
                .unwrap();
            for (tensor, stack) in bank.iter().enumerate() {
                let bytes = stack.read_to_host().unwrap();
                let width = form.kind(tensor).slot_bytes();
                for record in 0..6 {
                    let held = &bytes[record * width..(record + 1) * width];
                    match record {
                        1 => assert!(held == form.import(LANDED_EXPERTS[1], tensor).as_slice(), "{label}: tensor {tensor} record 1"),
                        _ => assert!(held.iter().all(|&byte| byte == 0), "{label}: tensor {tensor} record {record} written"),
                    }
                }
            }
            assert_eq!(read_i32(&landing)[landing_done(0)], low, "{label}: the landed entry's DONE");
            assert_eq!(read_i32(&sources), magnitude_kernels::source_words(&records, &[-1; 4]), "{label}: sources");
        }
    }
}

/// Two consecutive steps in flight (`q` and `q + 1`), queued as a load
/// queues a step and its lookahead successor: each on a family slot of its
/// own over one shared execution arena, every flight's decode graph
/// recorded ahead (`record_decode`). Each slot holds its own `sync` region,
/// so their counters are disjoint, and `HEARD` is published system-ordered:
/// even when both steps are submitted before either completes and their
/// entries land in rapid zero-sleep bursts across all `LAND_GROUPS`
/// threadgroups, neither raises `SLOT_VIOLATED` and both place every record.
#[test]
fn streaming_overlapping_land_steps_isolate_counters_and_order_heads() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let k = 4u64;
    let form = ServedForm::qwen35b(&device);
    let kernel = form.land_kernel(&device, k);
    let staging = landed_staging(&device, &form);
    let requested = [-1 - LANDED_EXPERTS[0] as i32, LANDED_EXPERTS[1] as i32, LANDED_EXPERTS[2] as i32, LANDED_EXPERTS[3] as i32];
    let in_kernels = device.backend().parks_in_kernels();

    let step_proto = streaming_step(&device);
    let (table, proto_words) = file_words(k, STREAMING_SEQUENCE as u64, &requested, &[], None);
    let file_proto = shared_i32(&device, &table.extents(), &proto_words);
    let sources_proto = decode_sources(&device, 1, k, &[0, -1, -1, -1]);
    let policy = shared_i32(&device, &[constants::POLICY_WORDS], &[0; constants::POLICY_WORDS as usize]);
    let landing_proto = shared_i32(&device, &[constants::LANDING_ENTRY(3)], &landing_words(0, 0, &[], 3));
    let bank_proto = form.records(&device, true, 6, |_, _| None);
    let grid = land_grid(&device, true);

    let mut graph = device.native_graph();
    let park_port = graph.timeline(seismic::timeline::Park);
    let mut port = |tensor: &seismic::Tensor| graph.port(tensor.element(), &tensor.extents()).unwrap();
    let step_port = port(&step_proto);
    let mut file_port = port(&file_proto);
    let mut sources_port = port(&sources_proto);
    let policy_port = port(&policy);
    let mut landing_port = port(&landing_proto);
    let grid_port = port(&grid);
    let staging_ports = [&staging[0], &staging[1], &staging[2]].map(&mut port);
    let bank_ports = [&bank_proto[0], &bank_proto[1], &bank_proto[2]].map(&mut port);
    let [mut gate_port, mut up_port, mut down_port] = bank_ports.clone();
    graph
        .enqueue(
            &kernel,
            routed_land::WorkflowArgs {
                step: step_port.tensor().into(),
                file: file_port.tensor_mut().into(),
                sources: sources_port.tensor_mut().into(),
                policy: policy_port.tensor().into(),
                landing: landing_port.tensor_mut().into(),
                staging_gate: staging_ports[0].tensor().into(),
                staging_up: staging_ports[1].tensor().into(),
                staging_down: staging_ports[2].tensor().into(),
                gate: gate_port.tensor_mut().into(),
                up: up_port.tensor_mut().into(),
                down: down_port.tensor_mut().into(),
                grid: grid_port.tensor().into(),
            },
        )
        .unwrap();
    let plan = graph.seal().unwrap();
    let family = seismic::NativeGraphFamily::new(std::slice::from_ref(&plan), 1).unwrap();
    let arena = seismic::NativeExecutionArena::new(&charges, family.workspace_bytes()).unwrap();
    let mut flights = [(); 2].map(|()| family.new_slot_in(&charges, &arena, &[family.upload_bytes()]).unwrap());
    family.record_decode().unwrap();
    let parks = seismic::Timeline::<seismic::timeline::Parks>::new(&device).unwrap();

    for round in 0..4u64 {
        let q0 = STREAMING_SEQUENCE as u64 + 2 * round;
        let q1 = q0 + 1;
        let step0 = i32_tensor(&device, &[constants::STEP_WORDS], &step_words(q0));
        let step1 = i32_tensor(&device, &[constants::STEP_WORDS], &step_words(q1));
        let (_, words0) = file_words(k, q0, &requested, &[], None);
        let (_, words1) = file_words(k, q1, &requested, &[], None);
        let file0 = shared_i32(&device, &table.extents(), &words0);
        let file1 = shared_i32(&device, &table.extents(), &words1);
        let _pulse0 = pulse(&file0, table);
        let _pulse1 = pulse(&file1, table);
        let landing0 = shared_i32(&device, &[constants::LANDING_ENTRY(3)], &landing_words((q0 - FLIGHT) as i32, 3, &[], 3));
        let landing1 = shared_i32(&device, &[constants::LANDING_ENTRY(3)], &landing_words((q1 - FLIGHT) as i32, 3, &[], 3));
        let sources0 = decode_sources(&device, 1, k, &[0, -1, -1, -1]);
        let sources1 = decode_sources(&device, 1, k, &[0, -1, -1, -1]);
        let bank0 = form.records(&device, true, 6, |_, _| None);
        let bank1 = form.records(&device, true, 6, |_, _| None);

        let serve_one = |q: u64, file: &seismic::Tensor, landing: &seismic::Tensor| {
            let low = q as i32;
            let slot = table.lane(q, 0);
            for (index, &(staged, record, choice)) in LANDED.iter().enumerate() {
                append_landing(landing, low, index, staged, record);
                file.store_release(4 * (slot + constants::FILE_ENTRY(choice) + constants::FILE_RECORD), record as u32)
                    .unwrap();
            }
            file.store_release_u64(4 * (slot + constants::SLOT_ANSWER), q).unwrap();
        };

        let bind_and_submit = |slot: &mut seismic::NativeGraphFamilySlot,
                               step: &seismic::Tensor,
                               file: &seismic::Tensor,
                               sources: &seismic::Tensor,
                               landing: &seismic::Tensor,
                               bank: &[seismic::Tensor; 3]| {
            let mut bindings = plan.bindings();
            if in_kernels {
                bindings.set_park(park_port, &parks).unwrap();
            } else {
                bindings.set_answered(park_port).unwrap();
            }
            for (port, tensor) in [
                (&step_port, step),
                (&file_port, file),
                (&sources_port, sources),
                (&policy_port, &policy),
                (&landing_port, landing),
                (&staging_ports[0], &staging[0]),
                (&staging_ports[1], &staging[1]),
                (&staging_ports[2], &staging[2]),
                (&bank_ports[0], &bank[0]),
                (&bank_ports[1], &bank[1]),
                (&bank_ports[2], &bank[2]),
                (&grid_port, &grid),
            ] {
                bindings.set(&charges, port, tensor).unwrap();
            }
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
            let mut active = slot.activate(&plan).unwrap();
            active
                .attach(bindings, plan.new_outputs(&charges).unwrap())
                .unwrap()
                .queue(&mut sequence)
                .unwrap();
            sequence.submit().unwrap()
        };

        if in_kernels {
            let c0 = bind_and_submit(&mut flights[0], &step0, &file0, &sources0, &landing0, &bank0);
            let c1 = bind_and_submit(&mut flights[1], &step1, &file1, &sources1, &landing1, &bank1);
            let (f0, l0, f1, l1) = (file0.clone(), landing0.clone(), file1.clone(), landing1.clone());
            std::thread::scope(|scope| {
                // Serve q1 while q0 is still parked, then burst-serve q0 so all
                // LAND_GROUPS threadgroups race HEARD -> LANDING_HEAD.
                scope.spawn(|| serve_one(q1, &f1, &l1));
                scope.spawn(|| {
                    std::thread::yield_now();
                    serve_one(q0, &f0, &l0);
                });
            });
            c0.wait().unwrap();
            c1.wait().unwrap();
        } else {
            serve_one(q0, &file0, &landing0);
            serve_one(q1, &file1, &landing1);
            let c0 = bind_and_submit(&mut flights[0], &step0, &file0, &sources0, &landing0, &bank0);
            let c1 = bind_and_submit(&mut flights[1], &step1, &file1, &sources1, &landing1, &bank1);
            c0.wait().unwrap();
            c1.wait().unwrap();
        }

        for (q, file, landing, sources, bank) in [
            (q0, &file0, &landing0, &sources0, &bank0),
            (q1, &file1, &landing1, &sources1, &bank1),
        ] {
            let slot = table.lane(q, 0) as usize;
            assert_eq!(
                read_i32(file)[slot + constants::SLOT_VIOLATED as usize],
                0,
                "round {round} step {q}: SLOT_VIOLATED set"
            );
            let words = read_i32(landing);
            for entry in 0..LANDED.len() {
                assert_eq!(words[landing_done(entry)], q as i32, "round {round} step {q}: entry {entry} DONE");
            }
            assert_eq!(
                read_i32(sources),
                magnitude_kernels::source_words(&[0, 1, 4, 5], &[-1; 4]),
                "round {round} step {q}: sources"
            );
            for (tensor, records) in bank.iter().enumerate() {
                let bytes = records.read_to_host().unwrap();
                let width = form.kind(tensor).slot_bytes();
                for &(staged, record, _) in &LANDED {
                    let record = record as usize;
                    let held = &bytes[record * width..(record + 1) * width];
                    assert!(
                        held == form.import(LANDED_EXPERTS[staged as usize + 1], tensor).as_slice(),
                        "round {round} step {q}: tensor {tensor} record {record}"
                    );
                }
            }
        }
    }
}

/// On every backend that streams, each landing launch matches its portable
/// body on the same words: dense BF16 records landed into F32 ones (the
/// interpreter binds dense elements alone), at the smallest shapes reaching
/// every path of the launch: a gated record (gate, up and down), an
/// up-only record (its gate views empty, GV = GS = 0) and, for a decode
/// layer, the host tier's placing (V = S = 0: nothing lands, the launch
/// only sources, a CPU choice's site named by its tier record). Each
/// landing's entries land out of record order, an entry past the count; a
/// decode layer's one read failed.
#[test]
fn native_landings_match_the_portable_body() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let module = crate::common::kernels();
    let (k, q, low) = (4usize, STREAMING_SEQUENCE as u64, STREAMING_SEQUENCE);
    let pairs = [(2, 4), (0, 1), (1, 0)];
    // A tensor's interpreter input: its values, of `dtype`.
    let dense = |tensor: &seismic::Tensor, dtype: DType| {
        let shape = tensor.extents().iter().map(|&extent| extent as usize).collect();
        Input::Tensor(dtype, shape, read_activation(tensor, dtype).into_iter().map(f64::from).collect())
    };
    let form = |gated: bool| ServedForm::dense(gated, seismic::Element::f32(), 4, 40);
    let check = |label: &str, outcome: &OracleOutcome, first: usize, landed: &[seismic::Tensor; 3]| {
        for (tensor, records) in landed.iter().enumerate() {
            let expected = input(outcome, first + tensor).into_iter().map(|value| value as f32).collect::<Vec<_>>();
            assert_eq!(read_f32(records), expected, "{label}: tensor {tensor}");
        }
    };

    // A decode layer: choice 0 held, 1 and 3 landed (staging 2 into bank 4,
    // staging 0 into bank 1), 2's read failed; where the host tier places,
    // 1 and 3 read in place from tier records 6 and 2, 2 the CPU's (its
    // site pending) from tier record 3.
    for (case, form, staged) in [
        ("gated", form(true), 3),
        ("up-only", form(false), 3),
        ("tier-placed", form(false), 0),
    ] {
        let label = format!("decode, {case}");
        let placed = staged > 0;
        let staging = form.records(&device, false, staged, |tensor, record| Some(form.stored(record, tensor)));
        let mut bank = form.records(&device, true, 5 * u64::from(placed), |_, _| None);
        let landed: &[(u64, i32)] = if placed { &[(1, 1), (2, -1), (3, 4)] } else { &[(1, -2 - 6), (2, -2 - 3), (3, -2 - 2)] };
        let (table, file) = file_words(k as u64, q, &[-1 - 3, 9, 5, 12], landed, Some(q));
        let site = if placed { -1 } else { -2 };
        let source_words = magnitude_kernels::source_words(&[0, -1, -1, -1], &[-1, -1, site, -1]);
        let landing = if placed { landing_words(low, 2, &pairs, 3) } else { landing_words(low, 0, &[], 3) };
        let [slots, width] = table.extents().map(|extent| extent as usize);
        let mut inputs = vec![
            ints(&[constants::STEP_WORDS as usize], &step_words(q)),
            ints(&[slots, width], &file),
            ints(&[constants::SOURCE_ROWS as usize, 1, k], &source_words),
            ints(&[constants::POLICY_WORDS as usize], &[0; constants::POLICY_WORDS as usize]),
            ints(&[landing.len()], &landing),
        ];
        inputs.extend(staging.iter().map(|tensor| dense(tensor, DType::BF16)));
        inputs.extend(bank.iter().map(|tensor| dense(tensor, DType::F32)));
        let groups = magnitude_kernels::land_groups(placed);
        inputs.push(ints(&[3], &[groups, 1, 1]));
        let outcome = interpret(&module, "routed_land", &DENSE_LANDING, inputs);

        let mut sources = i32_tensor(&device, &[constants::SOURCE_ROWS, 1, k as u64], &source_words);
        let mut landing_tensor = shared_i32(&device, &[landing.len() as u64], &landing);
        let [gate, up, down] = bank.each_mut();
        form.land_kernel(&device, k as u64)
            .call_answered(&charges, routed_land::Args {
                step: &streaming_step(&device),
                file: &mut shared_i32(&device, &table.extents(), &file),
                sources: &mut sources,
                policy: &shared_i32(&device, &[constants::POLICY_WORDS], &[0; constants::POLICY_WORDS as usize]),
                landing: &mut landing_tensor,
                staging_gate: &staging[0],
                staging_up: &staging[1],
                staging_down: &staging[2],
                gate,
                up,
                down,
                grid: &land_grid(&device, placed),
            })
            .unwrap();
        assert_eq!(read_i32(&sources), as_i32(input(&outcome, 2)), "{label}: sources");
        assert_eq!(read_i32(&landing_tensor), as_i32(input(&outcome, 4)), "{label}: landing list");
        check(&label, &outcome, 8, &bank);
    }

    // A round of window 1, round 1 (of 2 rounds per window): conveyor record
    // 2 into scratch slot 4, 0 into 1, then one past the count.
    for (case, form) in [("gated", form(true)), ("up-only", form(false))] {
        let label = format!("round, {case}");
        let conveyor = form.records(&device, false, 3, |tensor, record| Some(form.stored(record, tensor)));
        let mut scratch = form.records(&device, true, 5, |_, _| None);
        let (w, rounds, round) = (1i32, 2i32, 1i32);
        let table = SlotTable::scratch(FLIGHT, 4);
        let mut words = vec![0; table.words() as usize];
        let lane = constants::ROUND_LANE((w * rounds + round) as u64);
        set_pair(&mut words, table.lane(q, lane) + constants::SLOT_ANSWER, q);
        let landing = landing_words(low, 2, &pairs, 3);
        let list = constants::LIST_ENTRY(1) as usize;
        let [slots, width] = table.extents().map(|extent| extent as usize);
        let mut inputs = vec![
            ints(&[constants::STEP_WORDS as usize], &step_words(q)),
            ints(&[slots, width], &words),
            ints(&[1], &[w]),
            Input::I32(rounds),
            Input::I32(round),
            ints(&[landing.len()], &landing),
            ints(&[list], &vec![0; list]),
        ];
        inputs.extend(conveyor.iter().map(|tensor| dense(tensor, DType::BF16)));
        inputs.extend(scratch.iter().map(|tensor| dense(tensor, DType::F32)));
        inputs.extend([
            ints(&[list], &vec![0; list]),
            ints(&[1, constants::MAP_WORDS as usize], &vec![0; constants::MAP_WORDS as usize]),
            ints(&[3], &[constants::LAND_GROUPS as i32, 1, 1]),
        ]);
        let outcome = interpret(&module, "routed_round", &DENSE_LANDING, inputs);

        let mut scratch_tensor = shared_i32(&device, &table.extents(), &words);
        let mut landing_tensor = shared_i32(&device, &[landing.len() as u64], &landing);
        let [gate, up, down] = scratch.each_mut();
        form.round_kernel(&device)
            .call_answered(&charges, routed_round::Args {
                step: &streaming_step(&device),
                scratch: &mut scratch_tensor,
                window: &i32_tensor(&device, &[1], &[w]),
                rounds,
                round,
                landing: &mut landing_tensor,
                tiered: &host_words(&device, &[constants::LIST_ENTRY(1)]),
                conveyor_gate: &conveyor[0],
                conveyor_up: &conveyor[1],
                conveyor_down: &conveyor[2],
                gate,
                up,
                down,
                seeds: &host_words(&device, &[constants::LIST_ENTRY(1)]),
                map: &host_words(&device, &[1, constants::MAP_WORDS]),
                grid: &i32_tensor(&device, &[3], &[constants::LAND_GROUPS as i32, 1, 1]),
            })
            .unwrap();
        assert_eq!(read_i32(&scratch_tensor), as_i32(input(&outcome, 1)), "{label}: scratch arrival");
        assert_eq!(read_i32(&landing_tensor), as_i32(input(&outcome, 5)), "{label}: landing list");
        check(&label, &outcome, 10, &scratch);
    }
}

// ---------------------------------------------------------------------------
// The decode protocol (streaming_words.seismic's contract), by the
// portable bodies: the step-stamped map, the assignment, the landing's
// sources, the site request and the combine's answer.

/// Every backend encodes a streamed decode layer's landing only while an
/// expert of it can be absent (`unless (policy[POLICY_RESOLVED])`): where
/// its group is whole or its host tier keeps every expert, the landing is
/// left out, and with it any park before it, which takes a value only
/// where its call's launch is encoded (seismic's
/// `the_host_answers_a_parked_call`). A 40-layer decode step's 40 landing
/// parks are then none.
#[test]
fn the_landing_is_encoded_only_while_an_expert_can_be_absent() {
    let module = crate::common::kernels();
    let entry = module.entry_named("routed_land").unwrap();
    let resolved = seismic_lang::checked::LaunchCondition {
        word: "policy".into(),
        index: constants::POLICY_RESOLVED as u64,
        encoded: seismic_lang::checked::Encoded::WhileClear,
    };
    for backend in [registry::BackendName::Metal, registry::BackendName::Cuda, registry::BackendName::Vulkan] {
        let native = module.native_implementation(entry, backend).unwrap();
        let conditions = native.launches.iter().map(|launch| launch.conditions.clone()).collect::<Vec<_>>();
        assert_eq!(conditions, [vec![resolved.clone()]], "{backend:?}");
    }
}

/// `routed_hint_route` and `routed_hint_select` write each lead's hint ring
/// as `publish_hints` defines it, against a host reference: the lead's
/// router ranks each row's experts (`rank_experts`: best first, routes stored
/// worst first), and the ring names every predicted expert once, at its
/// first appearance in rank order (every row's best, then every row's
/// second, ...), with its rank, its row and its weight's word (the softmax
/// probability for the route; for the selection, the score it ranked on less
/// its bias), after its header (the count, and the step's words the ring
/// carries). Lead 0 reads the `next_*` router, lead 1 the `after_*` one;
/// with the next layer's POLICY_RESOLVED set no ring is written.
/// Words no entry writes keep their sentinel. Every M = 1 to VERIFY_ROWS
/// (the launch's capacity), AHEAD 1 and 2, and every admissible
/// configuration. The routers' logits are exact in every backend's
/// arithmetic and never tie ([`hint_router`]), so experts, ranks, rows and
/// the header compare exactly; weights, the routing's floats, compare
/// within a relative 1e-3.
#[test]
fn hint_rings_publish_each_leads_ranked_experts() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (h, e, k) = (Qwen35b::HIDDEN as u64, Qwen35b::EXPERTS as u64, Qwen35b::CHOICES as u64);
    let (hu, eu, ku) = (h as usize, e as usize, k as usize);
    let bf16 = seismic::Element::bf16();
    // Each lead's router (`hint_router`) and its selection bias, which
    // favours one expert past every score.
    let classes: [fn(usize) -> i32; 2] = [
        |expert| match expert % 21 {
            5 => 1,
            9 => -1,
            _ => 0,
        },
        |expert| match expert % 19 {
            3 => 1,
            7 => -1,
            _ => 0,
        },
    ];
    let favoured = [17usize, 40];
    let routers = classes.map(|class| bf16_tensor(&device, &[e, h], &hint_router(eu, hu, class)));
    let biases = favoured.map(|favoured| (0..eu).map(|expert| if expert == favoured { 8.0 } else { 0.0 }).collect::<Vec<f32>>());
    let bias_tensors = biases.clone().map(|bias| f32_tensor(&device, &[e], &bias));
    let norm = bf16_tensor(&device, &[h], &vec![1.0; hu]);
    let activation = bf16_tensor(&device, &[0, k], &[]);
    let residual = hint_residual(constants::VERIFY_ROWS as usize, hu);
    let mut unresolved = policy_words(0, 0, [0; 3]);
    unresolved[constants::POLICY_RESOLVED as usize] = 0;
    let mut resolved = unresolved.clone();
    resolved[constants::POLICY_RESOLVED as usize] = 1;
    // An entry's declared statics, from `given` and the single stream's.
    let statics_of = |names: &[String], given: &[(&str, u64)]| {
        let given = single_stream_statics(names, given);
        names.iter().fold(seismic::NativeSpecialization::new(), |specialization, name| {
            let (_, value) = given
                .iter()
                .find(|(candidate, _)| *candidate == name.as_str())
                .unwrap_or_else(|| panic!("the case gives no static `{name}`"));
            specialization.with_static(name.clone(), *value)
        })
    };
    // The rings of one call over `m` rows: each case's step, policy and
    // rings, the kernel run by `call`.
    let case = |label: &str, m: u64, ahead: u64, scored: &dyn Fn(usize, usize) -> Vec<(usize, f64)>, call: &dyn Fn(&seismic::Tensor, &seismic::Tensor, &mut seismic::Tensor)| {
        let ring = constants::HINT_ENTRY(m * k) as usize;
        let step = guessed_step(0b1010_1010 & ((1 << m) - 1));
        for (policy, written) in [(&unresolved, true), (&resolved, false)] {
            let mut rings = shared_i32(&device, &[ahead, ring as u64], &vec![HINT_SENTINEL; ahead as usize * ring]);
            call(
                &i32_tensor(&device, &[constants::STEP_WORDS], &step),
                &shared_i32(&device, &[constants::POLICY_WORDS], policy),
                &mut rings,
            );
            let got = read_i32(&rings);
            for lead in 0..ahead as usize {
                let label = format!("{label} rows {m} AHEAD {ahead} lead {lead} written {written}");
                let ring_words = &got[lead * ring..(lead + 1) * ring];
                match written {
                    true => {
                        let ranked = (0..m as usize).map(|row| scored(lead, row)).collect::<Vec<_>>();
                        assert_hint_ring(&label, ring_words, &expected_hint_ring(&step, &ranked, ku, ring));
                    }
                    false => assert!(ring_words.iter().all(|&word| word == HINT_SENTINEL), "{label}: a resolved layer's ring is untouched: {ring_words:?}"),
                }
            }
        }
    };
    let mut cases = 0usize;
    for ahead in [1u64, 2] {
        let entry = routed_hint_route::native_entry_with(routed_hint_route::Elements { NW: bf16, RW: bf16, A: bf16 });
        let implementation = entry.offered(&device).unwrap().unwrap();
        let statics = statics_of(&implementation.statics, &[("H", h), ("NM", h), ("K", k), ("AHEAD", ahead), ("EN", e)]);
        let mut configurations = Vec::new();
        implementation
            .walk_admissible(&implementation.implementation().with_owned_defaults(statics), |configuration| {
                configurations.push(configuration.clone());
                true
            })
            .unwrap();
        for configuration in configurations {
            let kernel = entry.prepare(&device, &configuration).unwrap();
            let label = format!("routed_hint_route {:?}", configuration.params());
            for m in 1..=constants::VERIFY_ROWS {
                let scored = |lead: usize, row: usize| {
                    let logits = (0..eu).map(|expert| hint_logit(row, expert, classes[lead])).collect::<Vec<_>>();
                    let probabilities = softmax(&logits);
                    ranked_choices(&logits, ku).into_iter().map(|expert| (expert, probabilities[expert])).collect::<Vec<_>>()
                };
                case(&label, m, ahead, &scored, &|step, policy, rings| {
                    kernel
                        .call(&charges, routed_hint_route::Args {
                            residual: &f32_tensor(&device, &[m, h], &residual[..m as usize * hu]),
                            activation: &activation,
                            next_norm: &norm,
                            next_router: &routers[0],
                            after_norm: &norm,
                            after_router: &routers[1],
                            eps: 1e-6,
                            step,
                            next_policy: policy,
                            rings,
                        })
                        .unwrap();
                });
                cases += 1;
            }
        }
        let entry = routed_hint_select::native_entry_with(routed_hint_select::Elements { RNW: bf16, RW: bf16, A: bf16 });
        let implementation = entry.offered(&device).unwrap().unwrap();
        let statics = statics_of(&implementation.statics, &[("H", h), ("K", k), ("AHEAD", ahead), ("EN", e)]);
        let mut configurations = Vec::new();
        implementation
            .walk_admissible(&implementation.implementation().with_owned_defaults(statics), |configuration| {
                configurations.push(configuration.clone());
                true
            })
            .unwrap();
        for configuration in configurations {
            let kernel = entry.prepare(&device, &configuration).unwrap();
            for score in 0..3 {
                let label = format!("routed_hint_select score {score} {:?}", configuration.params());
                for m in 1..=constants::VERIFY_ROWS {
                    let scored = |lead: usize, row: usize| {
                        let logits = (0..eu).map(|expert| hint_logit(row, expert, classes[lead])).collect::<Vec<_>>();
                        let scores = selection_scores(&logits, score);
                        let ranked = scores.iter().zip(&biases[lead]).map(|(score, &bias)| score + bias as f64).collect::<Vec<_>>();
                        ranked_choices(&ranked, ku).into_iter().map(|expert| (expert, scores[expert])).collect::<Vec<_>>()
                    };
                    case(&label, m, ahead, &scored, &|step, policy, rings| {
                        kernel
                            .call(&charges, routed_hint_select::Args {
                                residual: &f32_tensor(&device, &[m, h], &residual[..m as usize * hu]),
                                activation: &activation,
                                next_router_norm: &norm,
                                next_router: &routers[0],
                                next_bias: &bias_tensors[0],
                                after_router_norm: &norm,
                                after_router: &routers[1],
                                after_bias: &bias_tensors[1],
                                score: score as i32,
                                epsilon: 1e-6,
                                step,
                                next_policy: policy,
                                rings,
                            })
                            .unwrap();
                    });
                    cases += 1;
                }
            }
        }
    }
    println!("hint rings on {:?}: {cases} cases", device.backend());
}

/// A hint ring word no launch writes, which every word past the entries
/// keeps.
const HINT_SENTINEL: i32 = -9;

/// Row `row`'s signs of the low and high halves of the hidden width
/// ([`hint_residual`]), taking turns over the rows: + +, - +, + -, - -.
fn hint_signs(row: usize) -> (f32, f32) {
    ([1.0, -1.0, 1.0, -1.0][row % 4], [1.0, 1.0, -1.0, -1.0][row % 4])
}

/// `rows` residual rows of `hidden` columns, each +1 or -1 on each half
/// ([`hint_signs`]): every row's norm is 1, so its normalized row is its
/// residual exactly.
fn hint_residual(rows: usize, hidden: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            let (low, high) = hint_signs(row);
            (0..hidden).map(move |column| if column < hidden / 2 { low } else { high })
        })
        .collect()
}

/// A router of `experts` (at most 256) over `hidden` columns under which a
/// row of [`hint_residual`] has logit [`hint_logit`]: expert e weighs the low
/// half e / (64 H) and the high half 4 class(e) / H, both exact in bf16 and
/// summed exactly in f32 in any order.
fn hint_router(experts: usize, hidden: usize, class: fn(usize) -> i32) -> Vec<f32> {
    assert!(experts <= 256, "the low half's weights are exact for at most 256 experts");
    (0..experts)
        .flat_map(|expert| {
            (0..hidden).map(move |column| match column < hidden / 2 {
                true => expert as f32 / (64 * hidden) as f32,
                false => 4.0 * class(expert) as f32 / hidden as f32,
            })
        })
        .collect()
}

/// Row `row`'s logit of `expert` under [`hint_router`]: s_low e / 128 +
/// 2 s_high class(e). Distinct across experts (128 times it is e plus a
/// multiple of 256, up to sign), and at least 1/128 apart.
fn hint_logit(row: usize, expert: usize, class: fn(usize) -> i32) -> f64 {
    let (low, high) = hint_signs(row);
    low as f64 * expert as f64 / 128.0 + 2.0 * high as f64 * class(expert) as f64
}

/// The softmax of `logits`.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = logits.iter().map(|logit| (logit - maximum).exp()).collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f64>();
    exponentials.iter().map(|value| value / total).collect()
}

/// `select_experts`' scores of `logits` under score function `score`: 0
/// the softmax, 1 the sigmoid, 2 sqrt(softplus).
fn selection_scores(logits: &[f64], score: usize) -> Vec<f64> {
    match score {
        0 => softmax(logits),
        1 => logits.iter().map(|logit| 1.0 / (1.0 + (-logit).exp())).collect(),
        _ => logits.iter().map(|logit| (logit.max(0.0) + (-logit.abs()).exp().ln_1p()).sqrt()).collect(),
    }
}

/// The `k` best experts of `values`, best first, ties to the higher expert
/// (`rank_experts`).
fn ranked_choices(values: &[f64], k: usize) -> Vec<usize> {
    let mut experts = (0..values.len()).collect::<Vec<_>>();
    experts.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(b.cmp(&a)));
    experts.truncate(k);
    experts
}

/// The hint ring `publish_hints` writes for `ranked` (each row's choices,
/// best first, with their weights) under step words `step`: the header,
/// then each expert once, at its first appearance in rank order (rank-major,
/// then row), as (expert, rank, row, weight word); [`HINT_SENTINEL`] past
/// them.
fn expected_hint_ring(step: &[i32], ranked: &[Vec<(usize, f64)>], k: usize, ring: usize) -> Vec<i32> {
    let mut words = vec![HINT_SENTINEL; ring];
    let mut named = Vec::new();
    for rank in 0..k {
        for (row, choices) in ranked.iter().enumerate() {
            let (expert, weight) = choices[rank];
            if !named.contains(&expert) {
                let at = constants::HINT_ENTRY(named.len() as u64) as usize;
                words[at + constants::HINT_EXPERT as usize] = expert as i32;
                words[at + constants::HINT_RANK as usize] = rank as i32;
                words[at + constants::HINT_ROW as usize] = row as i32;
                words[at + constants::HINT_WEIGHT as usize] = ((weight as f32).to_bits() & 0x7fff_ffff) as i32;
                named.push(expert);
            }
        }
    }
    words[constants::RING_SEQUENCE as usize] = step[constants::STEP_LOW as usize];
    words[constants::RING_COUNT as usize] = named.len() as i32;
    words[constants::RING_GUESSES as usize] = step[constants::STEP_GUESSES as usize];
    words
}

/// `got` against `expected` ([`expected_hint_ring`]): every word exactly
/// but the entries' weight words, which hold a non-negative f32 within a
/// relative 1e-3 of the reference's.
fn assert_hint_ring(label: &str, got: &[i32], expected: &[i32]) {
    let count = expected[constants::RING_COUNT as usize] as u64;
    let weight_word = |index: usize| {
        (0..count).any(|entry| index == (constants::HINT_ENTRY(entry) + constants::HINT_WEIGHT) as usize)
    };
    let unlike = (0..expected.len())
        .filter(|&index| match weight_word(index) {
            true => {
                let (weight, reference) = (f32::from_bits(got[index] as u32), f32::from_bits(expected[index] as u32));
                got[index] < 0 || (weight - reference).abs() > 1e-3 * reference + 1e-6
            }
            false => got[index] != expected[index],
        })
        .map(|index| format!("word {index}: {} against {}", got[index], expected[index]))
        .collect::<Vec<_>>();
    assert!(unlike.is_empty(), "{label}: hint ring unlike its reference:\n{}", unlike.join("\n"));
}

/// A layer's policy words: whole, home, then the assignment's lines priced
/// (CPU, link, one hit, in nanoseconds; no row term and no handshake).
fn policy_words(whole: i32, home: i32, [cpu, link, hit]: [i32; 3]) -> Vec<i32> {
    let mut words = vec![0; constants::POLICY_WORDS as usize];
    words[constants::POLICY_WHOLE as usize] = whole;
    words[constants::POLICY_RESOLVED as usize] = whole;
    words[constants::POLICY_HOME as usize] = home;
    words[constants::POLICY_PRICED as usize] = 1;
    words[constants::POLICY_CPU_NS as usize] = cpu;
    words[constants::POLICY_LINK_NS as usize] = link;
    words[constants::POLICY_HIT_NS as usize] = hit;
    words
}

fn as_i32(values: Vec<f64>) -> Vec<i32> {
    values.into_iter().map(|value| value as i32).collect()
}

/// A map row holds its record for exactly the steps FROM <= q < UNTIL, both
/// words of each pair compared.
#[test]
fn portable_held_holds_a_record_within_its_steps() {
    let module = crate::common::kernels();
    let high = 1u64 << 32;
    let map = map_words(&[(3, 5, u64::MAX), (4, 0, 10), (-1, 0, 0), (2, high + 1, high + 3), (-2 - 5, 1, 10)]);
    let cases = [
        (0, 4, -1),
        (0, 5, 3),
        (0, high + 5, 3),
        (1, 9, 4),
        (1, 10, -1),
        (2, 7, -1),
        (3, 1, -1),
        (3, high, -1),
        (3, high + 1, 2),
        (3, high + 2, 2),
        (3, high + 3, -1),
        (4, 0, -1),
        (4, 1, -7),
        (4, 9, -7),
        (4, 10, -1),
    ];
    for (row, q, expected) in cases {
        let entry = module.entry(module.entry_named("held").unwrap(), &ElementBindings::new()).unwrap();
        let mut interpreter = Interpreter::new(&entry);
        let map = interpreter.add_tensor(TensorData::dense(
            DType::I32,
            vec![5, constants::MAP_WORDS as usize],
            map.iter().map(|&word| f64::from(word)).collect(),
        ));
        let outcome = interpreter
            .run(&[Arg::Tensor(map), Arg::Scalar(ReferenceScalar::I32(row)), Arg::Scalar(ReferenceScalar::U64(q))])
            .unwrap();
        let held = match outcome.results().next().unwrap().value() {
            OutcomeValue::Scalar(ReferenceScalar::I32(held)) => held,
            _ => panic!("held returns an i32"),
        };
        assert_eq!(held, expected, "row {row} at step {q}");
    }
}

/// `assign` splits a layer's misses by its policy lines, the cost
/// function's (whose own test runs this body against it,
/// `assessment::cost`): the CPU takes every miss that ends there no later
/// than in place, the link the rest; every choice of an expert goes to its
/// expert's side; an expert only front rows choose and the step lacks is
/// skipped, served by neither side. Front rows are no part of the assignment:
/// the exact rows' choices come out as the same rows routed alone at one row,
/// whatever the policy lines (a row term that a front row's choices of an
/// expert would double, a hit only a front row has).
#[test]
fn portable_assign_splits_the_misses_by_the_policy_lines() {
    let module = crate::common::kernels();
    // `policy_words` with the CPU's row term.
    let cpu_row = |lines: [i32; 3], cpu_row_ns: i32| {
        let mut words = policy_words(0, 0, lines);
        words[constants::POLICY_CPU_ROW_NS as usize] = cpu_row_ns;
        words
    };
    // (rows, front rows, routes, records, policy, expected records,
    // expected sites)
    let cases: [(&str, usize, i32, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>); 7] = [
        (
            "the CPU takes every miss",
            1,
            0,
            vec![10, 11, 12, 13],
            vec![5, -2, -3, -1],
            policy_words(0, 0, [10, 100, 40]),
            vec![5, -1, -1, -1],
            vec![-1, 0, 1, -2],
        ),
        (
            "each miss to the side it ends on first: CPU, link, CPU",
            1,
            0,
            vec![10, 11, 12, 13],
            vec![5, -2, -3, -1],
            policy_words(0, 0, [20, 30, 0]),
            vec![5, -1, -3, -1],
            vec![-1, 0, -1, -2],
        ),
        (
            "an expert's choices share its side",
            2,
            0,
            vec![11, 12, 11, 10],
            vec![-2, -3, -2, 5],
            policy_words(0, 0, [10, 100, 0]),
            vec![-1, -1, -1, 5],
            vec![0, 1, 0, -1],
        ),
        (
            "a front row's absent expert no exact row chooses is no miss",
            2,
            0b10,
            vec![10, 13, 13, 14],
            vec![5, -1, -1, -1],
            policy_words(0, 0, [10, 100, 40]),
            vec![5, -1, -1, -1],
            vec![-1, -2, -2, -1],
        ),
        (
            "a front row's choice of an exact row's expert adds no row to it",
            2,
            0b10,
            vec![10, 11, 10, 12],
            vec![-2, -3, -2, -4],
            cpu_row([10, 100, 0], 1000),
            vec![-1, -1, -1, -4],
            vec![0, 1, 0, -1],
        ),
        (
            "a hit only a front row has queues nothing on the GPU",
            2,
            0b10,
            vec![10, 11, 14, 15],
            vec![-2, -3, 7, 8],
            policy_words(0, 0, [60, 50, 500]),
            vec![-2, -1, 7, 8],
            vec![-1, 1, -1, -1],
        ),
        (
            "a front row before the exact row changes nothing for it",
            2,
            0b01,
            vec![10, 12, 10, 11],
            vec![-2, -4, -2, -3],
            cpu_row([10, 100, 0], 1000),
            vec![-1, -4, -1, -1],
            vec![0, -1, 0, 1],
        ),
    ];
    for (case, m, front, routes, records, policy, expected_records, expected_sites) in cases {
        let k = routes.len() / m;
        let inputs = vec![
            ints(&[m, k], &routes),
            ints(&[constants::POLICY_WORDS as usize], &policy),
            Input::I32(front),
            ints(&[constants::SOURCE_ROWS as usize, m, k], &magnitude_kernels::source_words(&records, &vec![-1; records.len()])),
        ];
        let outcome = interpret(&module, "assign", &[], inputs);
        let sources = as_i32(input(&outcome, 3));
        let count = routes.len();
        assert_eq!(
            sources,
            magnitude_kernels::source_words(&expected_records, &expected_sites),
            "{case}: records then sites ({count} choices)"
        );
        // The first exact row, routed alone at one row, is sourced and sited
        // as it is beside the front rows.
        if let Some(row) = (0..m).find(|row| front != 0 && (front >> row) & 1 == 0) {
            let at = row * k;
            let alone = vec![
                ints(&[1, k], &routes[at..at + k]),
                ints(&[constants::POLICY_WORDS as usize], &policy),
                Input::I32(0),
                ints(&[constants::SOURCE_ROWS as usize, 1, k], &magnitude_kernels::source_words(&records[at..at + k], &vec![-1; k])),
            ];
            let alone = as_i32(input(&interpret(&module, "assign", &[], alone), 3));
            assert_eq!(alone[..k], sources[at..at + k], "{case}: the exact row's records as at one row");
            assert_eq!(alone[k..2 * k], sources[count + at..count + at + k], "{case}: the exact row's sites as at one row");
        }
    }
}

/// A sited layer's route assigns misses by its policy's lines only once they
/// are priced (`POLICY_PRICED`): until then every miss stays the GPU's in
/// place and the CPU's site is off, however cheap the CPU looks. Every expert
/// is a miss the host tier holds, record t for expert t. Priced, an expert
/// only a front row chooses and the step lacks is skipped: no side serves
/// it.
#[test]
fn a_streamed_route_assigns_its_misses_only_once_priced() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (h, e, k) = (64u64, 32u64, 4u64);
    let table = SlotTable::file(FLIGHT, k);
    let statics = [("H", h), ("E", e), ("K", k), ("ES", 0), ("TM", e)];
    let entry = routed_select_streamed::native_entry_with(routed_select_streamed::Elements {
        NW: seismic::Element::f32(),
        RNW: seismic::Element::f32(),
        RW: seismic::Element::f32(),
        A: seismic::Element::bf16(),
    });
    let implementation = entry.offered(&device).unwrap().unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(seismic::NativeSpecialization::new(), |specialization, name| {
            let (_, value) = statics.iter().find(|(candidate, _)| *candidate == name.as_str()).unwrap();
            specialization.with_static(name.clone(), *value)
        });
    let kernel = entry.prepare(&device, &implementation.default_specialization(&statics).unwrap()).unwrap();
    let zeros = |element, shape: &[u64]| seismic::Tensor::zeros(&charges, element, shape).unwrap();
    let words = |shape: &[u64]| shared_i32(&device, shape, &vec![0; shape.iter().product::<u64>() as usize]);
    let f32 = seismic::Element::f32();
    let unmapped = map_words(&vec![(-1, 0, 0); e as usize]);
    let unpriced = {
        let mut words = policy_words(0, 0, [1, 100, 0]);
        words[constants::POLICY_PRICED as usize] = 0;
        words
    };
    for (policy, sited) in [(unpriced, false), (policy_words(0, 0, [1, 100, 0]), true)] {
        let tier = (0..e as i32).collect::<Vec<_>>();
        let mut routes = zeros(seismic::Element::i32(), &[1, k]);
        let mut sources = zeros(seismic::Element::i32(), &[constants::SOURCE_ROWS, 1, k]);
        kernel
            .call(&charges, routed_select_streamed::Args {
                residual: &f32_tensor(&device, &[1, h], &vec![1.0; h as usize]),
                norm: &zeros(f32, &[h]),
                router_norm: &zeros(f32, &[h]),
                router: &zeros(f32, &[e, h]),
                bias: &zeros(f32, &[e]),
                expert_scale: &zeros(f32, &[0]),
                routes: &mut routes,
                weights: &mut zeros(f32, &[1, k]),
                epsilon: 1e-6,
                score: 0,
                normalization: 0,
                normalization_epsilon: 1e-6,
                scale: 1.0,
                map: &shared_i32(&device, &[e, constants::MAP_WORDS], &unmapped),
                tier: &shared_i32(&device, &[e], &tier),
                policy: &shared_i32(&device, &[constants::POLICY_WORDS], &policy),
                sited: 1,
                step: &i32_tensor(&device, &[constants::STEP_WORDS], &step_words(STREAMING_SEQUENCE as u64)),
                file: &mut words(&table.extents()),
                sources: &mut sources,
                routing: &mut words(&[FLIGHT, constants::ROUTING_WORDS(k)]),
                keys: &mut key_records(&device, e),
                front: &mut words(&[constants::HINT_ENTRY(k)]),
                land_grid: &mut words(&[3]),
                land_groups: 1,
            })
            .unwrap();
        let routes = read_i32(&routes);
        let (records, sites): (Vec<i32>, Vec<i32>) =
            routes.iter().map(|&expert| if sited { (-1, expert) } else { (-2 - expert, -1) }).unzip();
        assert_eq!(read_i32(&sources), magnitude_kernels::source_words(&records, &sites), "{:?}, sited {sited}", device.backend());
    }
    // Priced, with a front row (STEP_FRONT): row 0 (exact) routes to
    // experts 1, 6, 2, 3 and row 1 (front) to 5, 6, 1, 4, best first (each
    // row's residual names its own router column). Expert 5 is absent, the
    // tier holding every other (7 stays unrouted here): the front row skips
    // 5, which neither side serves. The assignment runs over the exact row
    // alone, as at one row: its experts are the CPU's, the front row's
    // choices of them too, and expert 4, which only the front row chose, is
    // no part of it and stays in place across the link.
    let m = 2u64;
    let preferred: [[i32; 4]; 2] = [[1, 6, 2, 3], [5, 6, 1, 4]];
    let mut residual = vec![0.0; (m * h) as usize];
    let mut router = vec![0.0; (e * h) as usize];
    for (row, experts) in preferred.iter().enumerate() {
        residual[row * h as usize + row] = 1.0;
        for (rank, &expert) in experts.iter().enumerate() {
            router[expert as usize * h as usize + row] = (k as usize - rank) as f32;
        }
    }
    let tier = (0..e as i32).map(|expert| if expert == 5 || expert == 7 { -1 } else { expert }).collect::<Vec<_>>();
    let step = magnitude_kernels::step_words(STREAMING_SEQUENCE as u64, 0b10).to_vec();
    let mut routes = zeros(seismic::Element::i32(), &[m, k]);
    let mut sources = zeros(seismic::Element::i32(), &[constants::SOURCE_ROWS, m, k]);
    kernel
        .call(&charges, routed_select_streamed::Args {
            residual: &f32_tensor(&device, &[m, h], &residual),
            norm: &f32_tensor(&device, &[h], &vec![1.0; h as usize]),
            router_norm: &f32_tensor(&device, &[h], &vec![1.0; h as usize]),
            router: &f32_tensor(&device, &[e, h], &router),
            bias: &zeros(f32, &[e]),
            expert_scale: &zeros(f32, &[0]),
            routes: &mut routes,
            weights: &mut zeros(f32, &[m, k]),
            epsilon: 1e-6,
            score: 0,
            normalization: 0,
            normalization_epsilon: 1e-6,
            scale: 1.0,
            map: &shared_i32(&device, &[e, constants::MAP_WORDS], &unmapped),
            tier: &shared_i32(&device, &[e], &tier),
            policy: &shared_i32(&device, &[constants::POLICY_WORDS], &policy_words(0, 0, [1, 100, 0])),
            sited: 1,
            step: &i32_tensor(&device, &[constants::STEP_WORDS], &step),
            file: &mut words(&SlotTable::file(FLIGHT, m * k).extents()),
            sources: &mut sources,
            routing: &mut words(&[FLIGHT, constants::ROUTING_WORDS(m * k)]),
            keys: &mut key_records(&device, e),
            front: &mut words(&[constants::HINT_ENTRY(m * k)]),
            land_grid: &mut words(&[3]),
            land_groups: 1,
        })
        .unwrap();
    let routes = read_i32(&routes);
    let row_experts = |row: usize| {
        let mut experts = routes[row * k as usize..][..k as usize].to_vec();
        experts.sort_unstable();
        experts
    };
    assert_eq!((row_experts(0), row_experts(1)), (vec![1, 2, 3, 6], vec![1, 4, 5, 6]), "{:?}: routes", device.backend());
    let (records, sites): (Vec<i32>, Vec<i32>) =
        routes.iter().map(|&expert| match expert { 5 => (-1, -1), 4 => (-2 - expert, -1), _ => (-1, expert) }).unzip();
    assert_eq!(read_i32(&sources), magnitude_kernels::source_words(&records, &sites), "{:?}, a front row", device.backend());
}

/// Every choice of an expert has its expert's one source while the host
/// maps and unmaps its tier record during the route, as a landing's end
/// and a park do: every decode row routes to the same experts (a zero
/// router), and a host thread flips each expert's tier word between none
/// and record e with release stores for as long as the launches run. A
/// route that read the tier word once per choice could source two rows of
/// one expert differently, one absent and one tier-held, whose landing
/// and assignment then drop the one or serve it twice.
#[test]
fn a_streamed_route_sources_every_choice_of_an_expert_alike_under_tier_flips() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (h, e, k, m) = (64u64, 32u64, 8u64, constants::VERIFY_ROWS);
    let choices = m * k;
    let table = SlotTable::file(FLIGHT, choices);
    let statics = [("H", h), ("E", e), ("K", k), ("ES", 0), ("TM", e)];
    let entry = routed_select_streamed::native_entry_with(routed_select_streamed::Elements {
        NW: seismic::Element::f32(),
        RNW: seismic::Element::f32(),
        RW: seismic::Element::f32(),
        A: seismic::Element::bf16(),
    });
    let implementation = entry.offered(&device).unwrap().unwrap();
    let statics = implementation
        .statics
        .iter()
        .fold(seismic::NativeSpecialization::new(), |specialization, name| {
            let (_, value) = statics.iter().find(|(candidate, _)| *candidate == name.as_str()).unwrap();
            specialization.with_static(name.clone(), *value)
        });
    let kernel = entry.prepare(&device, &implementation.default_specialization(&statics).unwrap()).unwrap();
    let zeros = |element, shape: &[u64]| seismic::Tensor::zeros(&charges, element, shape).unwrap();
    let words = |shape: &[u64]| shared_i32(&device, shape, &vec![0; shape.iter().product::<u64>() as usize]);
    let f32 = seismic::Element::f32();
    let tier = seismic::Tensor::words(&charges, seismic::Element::i32(), &[e]).unwrap();
    let store = |expert: u64, record: i32| tier.store_release(expert * 4, record as u32).unwrap();
    (0..e).for_each(|expert| store(expert, -1));
    let mut unpriced = policy_words(0, 0, [1, 100, 0]);
    unpriced[constants::POLICY_PRICED as usize] = 0;
    let map = shared_i32(&device, &[e, constants::MAP_WORDS], &map_words(&vec![(-1, 0, 0); e as usize]));
    let policy = shared_i32(&device, &[constants::POLICY_WORDS], &unpriced);
    let step = i32_tensor(&device, &[constants::STEP_WORDS], &step_words(STREAMING_SEQUENCE as u64));
    let residual = f32_tensor(&device, &[m, h], &vec![1.0; (m * h) as usize]);
    let flipping = std::sync::atomic::AtomicBool::new(true);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut held = false;
            while flipping.load(std::sync::atomic::Ordering::Relaxed) {
                held = !held;
                (0..e).for_each(|expert| store(expert, if held { expert as i32 } else { -1 }));
            }
        });
        let finish = || flipping.store(false, std::sync::atomic::Ordering::Relaxed);
        for launch in 0..4000 {
            let mut routes = zeros(seismic::Element::i32(), &[m, k]);
            let mut sources = zeros(seismic::Element::i32(), &[constants::SOURCE_ROWS, m, k]);
            kernel
                .call(&charges, routed_select_streamed::Args {
                    residual: &residual,
                    norm: &zeros(f32, &[h]),
                    router_norm: &zeros(f32, &[h]),
                    router: &zeros(f32, &[e, h]),
                    bias: &zeros(f32, &[e]),
                    expert_scale: &zeros(f32, &[0]),
                    routes: &mut routes,
                    weights: &mut zeros(f32, &[m, k]),
                    epsilon: 1e-6,
                    score: 0,
                    normalization: 0,
                    normalization_epsilon: 1e-6,
                    scale: 1.0,
                    map: &map,
                    tier: &tier,
                    policy: &policy,
                    sited: 1,
                    step: &step,
                    file: &mut words(&table.extents()),
                    sources: &mut sources,
                    routing: &mut words(&[FLIGHT, constants::ROUTING_WORDS(choices)]),
                    keys: &mut key_records(&device, e),
                    front: &mut words(&[constants::HINT_ENTRY(choices)]),
                    land_grid: &mut words(&[3]),
                    land_groups: 1,
                })
                .inspect_err(|_| finish())
                .unwrap();
            let routes = read_i32(&routes);
            let records = &read_i32(&sources)[..choices as usize];
            let first = |expert: i32| routes.iter().position(|&routed| routed == expert).unwrap();
            let split = (0..choices as usize).find(|&c| records[c] != records[first(routes[c])]);
            if let Some(c) = split {
                finish();
                panic!(
                    "{:?}, launch {launch}: choice {c} of expert {} sourced {}, its first choice {}",
                    device.backend(),
                    routes[c],
                    records[c],
                    records[first(routes[c])]
                );
            }
        }
        finish();
    });
}

/// The landing groups a route is given, which its grid holds where it
/// requests.
const LANDING: i32 = 5;

/// A step's front row (STEP_FRONT) computes only the experts the step holds
/// or an exact row chooses. Any other choice stays unsourced (its
/// routing record keeps its source as resolved), and an absent expert no
/// exact row chooses is skipped: its request words name
/// it held, so neither the wait nor the host reads it now, and the layer's
/// front ring, which names every choice of every front row (rank first,
/// then row, each with its rank and row, an expert two front rows chose
/// named by both), names it for the next step. An absent expert an exact
/// row chooses is requested for every row, and a held one serves all.
/// Without a front row every absent expert is requested, and the front ring
/// is not written. Both streamed routes resolve so
/// ([`assert_front_resolve`]): `routed_select_streamed` over three rows,
/// the last two front rows, whose row 0 routes to experts 1, 6, 2, 3, row 1
/// to 5, 6, 1, 7 and row 2 to 6, 5, 4, 1, best first (each row's residual
/// names its own router column), and `routed_route_streamed` over the
/// 35B's first two decode rows, the second a front row. Expert 1 is held
/// (record 4) and every other is absent, the tier holding none.
#[test]
fn a_streamed_route_skips_what_only_a_front_row_lacks() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let zeros = |element, shape: &[u64]| seismic::Tensor::zeros(&charges, element, shape).unwrap();
    let words = |shape: &[u64]| shared_i32(&device, shape, &vec![0; shape.iter().product::<u64>() as usize]);
    let prepare = |entry_statics: &[String], given: &[(&str, u64)]| {
        entry_statics.iter().fold(seismic::NativeSpecialization::new(), |specialization, name| {
            let (_, value) = given.iter().find(|(candidate, _)| *candidate == name.as_str()).unwrap();
            specialization.with_static(name.clone(), *value)
        })
    };
    let step = |front: i32| i32_tensor(&device, &[constants::STEP_WORDS], &guessed_step(front));
    let policy = shared_i32(&device, &[constants::POLICY_WORDS], &policy_words(0, 0, [0; 3]));
    let held = |e: u64| shared_i32(&device, &[e, constants::MAP_WORDS], &map_words(&one_held(e)));
    // The bank's ports of a route of `m` rows of `k` choices over `e`
    // experts: the file table, sources, routing record, front ring, tier
    // map and landing grid.
    let ports = |m: u64, e: u64, k: u64| {
        let choices = m * k;
        (
            SlotTable::file(FLIGHT, choices),
            words(&SlotTable::file(FLIGHT, choices).extents()),
            zeros(seismic::Element::i32(), &[constants::SOURCE_ROWS, m, k]),
            words(&[FLIGHT, constants::ROUTING_WORDS(choices)]),
            words(&[constants::HINT_ENTRY(choices)]),
            shared_i32(&device, &[e], &vec![-1; e as usize]),
            words(&[3]),
        )
    };

    // routed_select_streamed.
    let (m, h, e, k) = THREE_ROWS;
    let (residual, router) = three_rows();
    for front in [0b110, 0b000] {
        let ThreeRowResolve { table, routes, weights, sources, file, routing, front: front_ring, land_grid } =
            resolve_three_rows(&device, &policy_words(0, 0, [0; 3]), front, &one_held(e));
        let case = format!("routed_select_streamed on {:?}, front rows {front:#b}", device.backend());
        let routes = read_i32(&routes);
        let skipped = assert_front_resolve(
            &case,
            front,
            k,
            &routes,
            &read_f32(&weights),
            &table,
            [&sources, &file, &routing, &front_ring, &land_grid],
        );
        assert_eq!(skipped, if front != 0 { vec![4, 5, 7] } else { vec![] }, "{case}: skipped");
        // The reference resolves the same words (`resolve` in
        // routed.seismic), its front ring's weights aside (the reference's
        // own float arithmetic).
        let reference = reference_select_resolve(m, h, e, k, &residual, &router, front, &table);
        let weightless = |ring: Vec<i32>| {
            ring.into_iter()
                .enumerate()
                .map(|(word, value)| {
                    let entry = word.checked_sub(constants::HINT_ENTRY(0) as usize);
                    match entry.map(|at| at % constants::HINT_ENTRY_WORDS as usize) {
                        Some(at) if at == constants::HINT_WEIGHT as usize => 0,
                        _ => value,
                    }
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(reference.routes, routes, "{case}: reference routes");
        assert_eq!(reference.sources, read_i32(&sources), "{case}: reference sources");
        assert_eq!(reference.file, read_i32(&file), "{case}: reference file table");
        assert_eq!(reference.routing, read_i32(&routing), "{case}: reference routing record");
        assert_eq!(weightless(reference.front), weightless(read_i32(&front_ring)), "{case}: reference front ring");
        assert_eq!(reference.land_grid, read_i32(&land_grid), "{case}: reference landing grid");
    }

    // routed_route_streamed, over the 35B's geometry and weights.
    let m = 2u64;
    let block = Qwen35b::new(&device);
    let (h, e, k, s) = (Qwen35b::HIDDEN as u64, Qwen35b::EXPERTS as u64, Qwen35b::CHOICES as u64, Qwen35b::SHARED as u64);
    let (f32, bf16, q8) = (seismic::Element::f32(), seismic::Element::bf16(), element(&device, "q8g32s"));
    let entry = routed_route_streamed::native_entry_with(routed_route_streamed::Elements { NW: bf16, RW: bf16, SRW: f32, SGW: q8, SUW: q8, A: bf16 });
    let implementation = entry.offered(&device).unwrap().unwrap();
    let statics = prepare(
        &implementation.statics,
        &[("H", h), ("PT", 1), ("NM", h), ("E", e), ("K", k), ("S", s), ("ES", 0), ("AHEAD", 0), ("EN", e), ("TM", e)],
    );
    let kernel = entry.prepare(&device, &implementation.default_specialization(&statics).unwrap()).unwrap();
    let norm = bf16_tensor(&device, &[h], &block.norm);
    let router = bf16_tensor(&device, &[e, h], &block.router);
    // Two rows that route differently: the same pattern's rows are positive and normalize alike.
    let rows = block.diverging_residual_rows();
    let mut skips = 0;
    for front in [0b10, 0b00] {
        let (table, mut file, mut sources, mut routing, mut front_ring, tier, mut land_grid) = ports(m, e, k);
        let mut routes = zeros(seismic::Element::i32(), &[m, k]);
        let mut scores = zeros(f32, &[m, k]);
        kernel
            .call(&charges, routed_route_streamed::Args {
                residual: &f32_tensor(&device, &[m, h], &rows),
                norm: &norm,
                router: &router,
                shared_router: &f32_tensor(&device, &[h], &block.shared_router),
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
                routes: &mut routes,
                scores: &mut scores,
                eps: 1e-6,
                normalize: 1,
                expert_scale: &absent(&device),
                map: &held(e),
                tier: &tier,
                policy: &policy,
                sited: 0,
                step: &step(front),
                file: &mut file,
                sources: &mut sources,
                routing: &mut routing,
                keys: &mut key_records(&device, e),
                front: &mut front_ring,
                land_grid: &mut land_grid,
                land_groups: LANDING,
            })
            .unwrap();
        let case = format!("routed_route_streamed on {:?}, front rows {front:#b}", device.backend());
        let routes = read_i32(&routes);
        skips += assert_front_resolve(
            &case,
            front,
            k,
            &routes,
            &read_f32(&scores),
            &table,
            [&sources, &file, &routing, &front_ring, &land_grid],
        )
        .len();
    }
    assert!(skips > 0, "the 35B's front row skipped nothing: its rows route alike despite `diverging_residual_rows`");
}

/// The three rows [`a_streamed_route_skips_what_only_a_front_row_lacks`]'s
/// general route resolves (rows, hidden, experts, choices): row 0 routes to
/// experts 1, 6, 2, 3, row 1 to 5, 6, 1, 7 and row 2 to 6, 5, 4, 1, best
/// first.
const THREE_ROWS: (u64, u64, u64, u64) = (3, 64, 32, 4);
const THREE_ROW_ROUTES: [[i32; 4]; 3] = [[1, 6, 2, 3], [5, 6, 1, 7], [6, 5, 4, 1]];

/// [`THREE_ROWS`]' residual and router: each row's residual names its own
/// router column, which ranks the row's experts.
fn three_rows() -> (Vec<f32>, Vec<f32>) {
    let (m, h, e, k) = THREE_ROWS;
    let mut residual = vec![0.0; (m * h) as usize];
    let mut router = vec![0.0; (e * h) as usize];
    for (row, experts) in THREE_ROW_ROUTES.iter().enumerate() {
        residual[row * h as usize + row] = 1.0;
        for (rank, &expert) in experts.iter().enumerate() {
            router[expert as usize * h as usize + row] = (k as usize - rank) as f32;
        }
    }
    (residual, router)
}

/// Step `STREAMING_SEQUENCE`'s words with front rows `front`, each front
/// row's guess's kind too ([`front_kinds`]): they change nothing the route
/// computes, and every ring it writes carries them.
fn guessed_step(front: i32) -> [i32; constants::STEP_WORDS as usize] {
    let mut words = magnitude_kernels::step_words(STREAMING_SEQUENCE as u64, front as u32);
    words[constants::STEP_GUESSES as usize] = front_kinds(front);
    words
}

/// The STEP_GUESSES word of front rows `front`, each front row's guess the
/// head's.
fn front_kinds(front: i32) -> i32 {
    magnitude_kernels::step_guesses(
        (0..constants::VERIFY_ROWS as usize).filter(|&row| (front >> row) & 1 != 0).map(|row| (row, constants::GUESS_HEAD as i32)),
    )
}

/// The map rows of `e` experts of which the step holds expert 1 alone, in
/// record 4.
fn one_held(e: u64) -> Vec<(i32, u64, u64)> {
    let mut map = vec![(-1, 0, 0); e as usize];
    map[1] = (4, 0, u64::MAX);
    map
}

/// What `routed_select_streamed` resolved of [`three_rows`] at step
/// [`guessed_step`]`(front)` under the layer's `policy`, its map rows `map`
/// and a tier holding none, from zeroed words and a landing grid each of
/// whose words is 7: the file table's slots, and the words it wrote.
struct ThreeRowResolve {
    table: SlotTable,
    routes: seismic::Tensor,
    weights: seismic::Tensor,
    sources: seismic::Tensor,
    file: seismic::Tensor,
    routing: seismic::Tensor,
    front: seismic::Tensor,
    land_grid: seismic::Tensor,
}

fn resolve_three_rows(device: &seismic::Device, policy: &[i32], front: i32, map: &[(i32, u64, u64)]) -> ThreeRowResolve {
    let charges = seismic::Charges::tally(device);
    let (m, h, e, k) = THREE_ROWS;
    let (residual, router) = three_rows();
    let (f32, i32) = (seismic::Element::f32(), seismic::Element::i32());
    let zeros = |element, shape: &[u64]| seismic::Tensor::zeros(&charges, element, shape).unwrap();
    let words = |shape: &[u64]| shared_i32(device, shape, &vec![0; shape.iter().product::<u64>() as usize]);
    let entry = routed_select_streamed::native_entry_with(routed_select_streamed::Elements {
        NW: f32,
        RNW: f32,
        RW: f32,
        A: seismic::Element::bf16(),
    });
    let implementation = entry.offered(device).unwrap().unwrap();
    let given = [("H", h), ("E", e), ("K", k), ("ES", 0), ("AHEAD", 0), ("EN", e), ("TM", e)];
    let statics = implementation.statics.iter().fold(seismic::NativeSpecialization::new(), |specialization, name| {
        let (_, value) = given.iter().find(|(candidate, _)| *candidate == name.as_str()).unwrap();
        specialization.with_static(name.clone(), *value)
    });
    let kernel = entry.prepare(device, &implementation.default_specialization(&statics).unwrap()).unwrap();
    let choices = m * k;
    let table = SlotTable::file(FLIGHT, choices);
    let mut resolved = ThreeRowResolve {
        file: words(&table.extents()),
        table,
        routes: zeros(i32, &[m, k]),
        weights: zeros(f32, &[m, k]),
        sources: zeros(i32, &[constants::SOURCE_ROWS, m, k]),
        routing: words(&[FLIGHT, constants::ROUTING_WORDS(choices)]),
        front: words(&[constants::HINT_ENTRY(choices)]),
        land_grid: shared_i32(device, &[3], &[7; 3]),
    };
    kernel
        .call(&charges, routed_select_streamed::Args {
            residual: &f32_tensor(device, &[m, h], &residual),
            norm: &f32_tensor(device, &[h], &vec![1.0; h as usize]),
            router_norm: &f32_tensor(device, &[h], &vec![1.0; h as usize]),
            router: &f32_tensor(device, &[e, h], &router),
            bias: &zeros(f32, &[e]),
            expert_scale: &zeros(f32, &[0]),
            routes: &mut resolved.routes,
            weights: &mut resolved.weights,
            epsilon: 1e-6,
            score: 0,
            normalization: 0,
            normalization_epsilon: 1e-6,
            scale: 1.0,
            map: &shared_i32(device, &[e, constants::MAP_WORDS], &map_words(map)),
            tier: &shared_i32(device, &[e], &vec![-1; e as usize]),
            policy: &shared_i32(device, &[constants::POLICY_WORDS], policy),
            sited: 0,
            step: &i32_tensor(device, &[constants::STEP_WORDS], &guessed_step(front)),
            file: &mut resolved.file,
            sources: &mut resolved.sources,
            routing: &mut resolved.routing,
            keys: &mut key_records(device, e),
            front: &mut resolved.front,
            land_grid: &mut resolved.land_grid,
            land_groups: LANDING,
        })
        .unwrap();
    let routes = read_i32(&resolved.routes);
    let row_experts = |row: usize| {
        let mut experts = routes[row * k as usize..][..k as usize].to_vec();
        experts.sort_unstable();
        experts
    };
    let expected = |row: usize| {
        let mut experts = THREE_ROW_ROUTES[row].to_vec();
        experts.sort_unstable();
        experts
    };
    assert_eq!((0..m as usize).map(row_experts).collect::<Vec<_>>(), (0..m as usize).map(expected).collect::<Vec<_>>(), "routes");
    resolved
}

/// A whole layer (POLICY_WHOLE) holds every expert at home (POLICY_HOME +
/// expert), and its route sources every choice from there, a front row's
/// too (each expert held), none with a CPU site. Nothing misses: it requests nothing
/// and its landing grid is empty, `[0, 1, 1]` (its routing record, which names
/// its home records, is `a_streamed_route_keys_every_expert_as_it_ranks_them`'s).
/// `routed_select_streamed` over [`three_rows`], the last two front rows.
#[test]
fn a_whole_layers_front_rows_compute_every_choice() {
    let Some(device) = streaming_device() else {
        return;
    };
    const HOME: i32 = 100;
    let (_, _, e, _) = THREE_ROWS;
    let resolved = resolve_three_rows(&device, &policy_words(1, HOME, [0; 3]), 0b110, &vec![(-1, 0, 0); e as usize]);
    let case = format!("routed_select_streamed on {:?}", device.backend());
    let routes = read_i32(&resolved.routes);
    let records = routes.iter().map(|&expert| HOME + expert).collect::<Vec<_>>();
    let sites = vec![-1; routes.len()];
    assert_eq!(read_i32(&resolved.sources), magnitude_kernels::source_words(&records, &sites), "{case}: sources");
    assert!(read_i32(&resolved.file).iter().all(|&word| word == 0), "{case}: file table");
    assert_eq!(read_i32(&resolved.land_grid), [0, 1, 1], "{case}: landing grid");
}

/// A streamed route writes each row's selection
/// key of every expert into the step's key record as the reference resolve
/// keys them (the ranked softmax probability of the fused form, score plus
/// bias of the general form), and its routing record names, row by row and
/// rank by rank, the top K of the keys it recorded (descending, ties to the
/// higher expert). At a decode pass of an exact row and a front row over
/// Qwen3.8 Flash Next's experts and choices (E 512, K 10), at a streamed
/// layer and at a whole one, which writes its routing record too, each
/// choice's source its home record.
#[test]
fn a_streamed_route_keys_every_expert_as_it_ranks_them() {
    let Some(device) = streaming_device() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    const HOME: i32 = 100;
    let (m, h, e, k, s) = (2u64, 64u64, 512u64, 10u64, 64u64);
    let (mu, hu, eu, ku, su) = (m as usize, h as usize, e as usize, k as usize, s as usize);
    let (choices, rows) = (m * k, constants::VERIFY_ROWS);
    let front = 0b10;
    let (f32, bf16, i32) = (seismic::Element::f32(), seismic::Element::bf16(), seismic::Element::i32());
    let zeros = |element, shape: &[u64]| seismic::Tensor::zeros(&charges, element, shape).unwrap();
    let words = |shape: &[u64]| shared_i32(&device, shape, &vec![0; shape.iter().product::<u64>() as usize]);
    // Rows of signs (rms 1, so their bf16 normal form is themselves) and
    // weights of multiples of 1/512: every logit is exact on every side, so
    // a tie is one everywhere and every other key differs past rounding.
    let residual = (0..mu * hu)
        .map(|i| if i < hu || (i * 7 + i / hu) % 3 != 0 { 1.0 } else { -1.0 })
        .collect::<Vec<f32>>();
    let router = (0..eu * hu).map(|i| ((i * 37 + (i / hu) * (i % hu) * 11) % 33) as f32 / 512.0 - 16.0 / 512.0).collect::<Vec<f32>>();
    let bias = (0..eu).map(|x| (x % 5) as f32 / 1024.0).collect::<Vec<f32>>();
    let shared_router = (0..hu).map(|i| (i % 9) as f32 / 64.0 - 4.0 / 64.0).collect::<Vec<f32>>();
    let shared = (0..su * hu).map(|i| (i % 17) as f32 / 64.0 - 8.0 / 64.0).collect::<Vec<f32>>();
    let ones = vec![1.0f32; hu];
    let step = guessed_step(front);
    let keys_words = (FLIGHT * rows * e) as usize;
    let slot = STREAMING_SEQUENCE as u64 % FLIGHT;
    let routing_words = constants::ROUTING_WORDS(choices);
    let held = map_words(&one_held(e));
    let bf16s = |shape: &[usize], values: &[f32]| Input::Tensor(DType::BF16, shape.to_vec(), values.iter().map(|&v| f64::from(v)).collect());

    for (whole, home) in [(0, 0), (1, HOME)] {
        let policy = policy_words(whole, home, [0; 3]);
        for general in [false, true] {
            let case = format!(
                "{} on {:?}, {}",
                if general { "routed_select_streamed" } else { "routed_route_streamed" },
                device.backend(),
                if whole != 0 { "whole" } else { "streamed" }
            );
            let table = SlotTable::file(FLIGHT, choices);
            let (mut file, mut sources, mut routing, mut keys, mut front_ring, mut land_grid) = (
                words(&table.extents()),
                zeros(i32, &[constants::SOURCE_ROWS, m, k]),
                words(&[FLIGHT, routing_words]),
                words(&[FLIGHT, rows, e]),
                words(&[constants::HINT_ENTRY(choices)]),
                words(&[3]),
            );
            let mut routes = zeros(i32, &[m, k]);
            let mut weights = zeros(f32, &[m, k]);
            let (map, tier, policy_tensor) = (
                shared_i32(&device, &[e, constants::MAP_WORDS], &held),
                shared_i32(&device, &[e], &vec![-1; eu]),
                shared_i32(&device, &[constants::POLICY_WORDS], &policy),
            );
            let step_tensor = i32_tensor(&device, &[constants::STEP_WORDS], &step);
            // The reference's key record: the entry's keys port, its final
            // contents (`input`, by the port's ordinal).
            let (reference_routes, reference_keys) = if general {
                let entry = routed_select_streamed::native_entry_with(routed_select_streamed::Elements { NW: f32, RNW: f32, RW: f32, A: bf16 });
                let implementation = entry.offered(&device).unwrap().unwrap();
                let given = [("H", h), ("E", e), ("K", k), ("ES", 0), ("TM", e)];
                let statics = implementation.statics.iter().fold(seismic::NativeSpecialization::new(), |specialization, name| {
                    let (_, value) = given.iter().find(|(candidate, _)| *candidate == name.as_str()).unwrap();
                    specialization.with_static(name.clone(), *value)
                });
                let kernel = entry.prepare(&device, &implementation.default_specialization(&statics).unwrap()).unwrap();
                kernel
                    .call(&charges, routed_select_streamed::Args {
                        residual: &f32_tensor(&device, &[m, h], &residual),
                        norm: &f32_tensor(&device, &[h], &ones),
                        router_norm: &f32_tensor(&device, &[h], &ones),
                        router: &f32_tensor(&device, &[e, h], &router),
                        bias: &f32_tensor(&device, &[e], &bias),
                        expert_scale: &zeros(f32, &[0]),
                        routes: &mut routes,
                        weights: &mut weights,
                        epsilon: 1e-6,
                        score: 0,
                        normalization: 1,
                        normalization_epsilon: 1e-6,
                        scale: 1.0,
                        map: &map,
                        tier: &tier,
                        policy: &policy_tensor,
                        sited: 0,
                        step: &step_tensor,
                        file: &mut file,
                        sources: &mut sources,
                        routing: &mut routing,
                        keys: &mut keys,
                        front: &mut front_ring,
                        land_grid: &mut land_grid,
                        land_groups: LANDING,
                    })
                    .unwrap();
                let [slots, width] = table.extents().map(|extent| extent as usize);
                let inputs = vec![
                    floats(&[mu, hu], &residual),
                    floats(&[hu], &ones),
                    floats(&[hu], &ones),
                    floats(&[eu, hu], &router),
                    floats(&[eu], &bias),
                    floats(&[0], &[]),
                    ints(&[mu, ku], &vec![0; mu * ku]),
                    floats(&[mu, ku], &vec![0.0; mu * ku]),
                    Input::F32(1e-6),
                    Input::I32(0),
                    Input::I32(1),
                    Input::F32(1e-6),
                    Input::F32(1.0),
                    ints(&[eu, constants::MAP_WORDS as usize], &held),
                    ints(&[eu], &vec![-1; eu]),
                    ints(&[constants::POLICY_WORDS as usize], &policy),
                    Input::I32(0),
                    ints(&[constants::STEP_WORDS as usize], &step),
                    ints(&[slots, width], &vec![0; slots * width]),
                    ints(&[constants::SOURCE_ROWS as usize, mu, ku], &vec![0; constants::SOURCE_ROWS as usize * mu * ku]),
                    ints(&[FLIGHT as usize, routing_words as usize], &vec![0; FLIGHT as usize * routing_words as usize]),
                    ints(&[FLIGHT as usize, rows as usize, eu], &vec![0; keys_words]),
                    ints(&[constants::HINT_ENTRY(choices) as usize], &vec![0; constants::HINT_ENTRY(choices) as usize]),
                    ints(&[3], &[0; 3]),
                    Input::I32(LANDING),
                ];
                let bindings = [("NW", DType::F32), ("RNW", DType::F32), ("RW", DType::F32), ("A", DType::BF16)];
                let outcome = interpret(crate::common::kernels(), "routed_select_streamed", &bindings, inputs);
                (as_i32(input(&outcome, 6)), as_i32(input(&outcome, 21)))
            } else {
                let entry = routed_route_streamed::native_entry_with(routed_route_streamed::Elements { NW: bf16, RW: bf16, SRW: f32, SGW: bf16, SUW: bf16, A: bf16 });
                let implementation = entry.offered(&device).unwrap().unwrap();
                let given = [("H", h), ("PT", 1), ("NM", h), ("E", e), ("K", k), ("S", s), ("ES", 0), ("TM", e)];
                let statics = implementation.statics.iter().fold(seismic::NativeSpecialization::new(), |specialization, name| {
                    let (_, value) = given.iter().find(|(candidate, _)| *candidate == name.as_str()).unwrap();
                    specialization.with_static(name.clone(), *value)
                });
                let kernel = entry.prepare(&device, &implementation.default_specialization(&statics).unwrap()).unwrap();
                kernel
                    .call(&charges, routed_route_streamed::Args {
                        residual: &f32_tensor(&device, &[m, h], &residual),
                        norm: &bf16_tensor(&device, &[h], &ones),
                        router: &bf16_tensor(&device, &[e, h], &router),
                        shared_router: &f32_tensor(&device, &[h], &shared_router),
                        shared_gate: &bf16_tensor(&device, &[s, h], &shared),
                        shared_up: &bf16_tensor(&device, &[s, h], &shared),
                        routes: &mut routes,
                        scores: &mut weights,
                        eps: 1e-6,
                        normalize: 1,
                        expert_scale: &absent(&device),
                        map: &map,
                        tier: &tier,
                        policy: &policy_tensor,
                        sited: 0,
                        step: &step_tensor,
                        file: &mut file,
                        sources: &mut sources,
                        routing: &mut routing,
                        keys: &mut keys,
                        front: &mut front_ring,
                        land_grid: &mut land_grid,
                        land_groups: LANDING,
                    })
                    .unwrap();
                let [slots, width] = table.extents().map(|extent| extent as usize);
                let inputs = vec![
                    floats(&[mu, hu], &residual),
                    bf16s(&[hu], &ones),
                    bf16s(&[eu, hu], &router),
                    floats(&[hu], &shared_router),
                    bf16s(&[su, hu], &shared),
                    bf16s(&[su, hu], &shared),
                    ints(&[mu, ku], &vec![0; mu * ku]),
                    floats(&[mu, ku], &vec![0.0; mu * ku]),
                    Input::F32(1e-6),
                    Input::I32(1),
                    floats(&[0], &[]),
                    ints(&[eu, constants::MAP_WORDS as usize], &held),
                    ints(&[eu], &vec![-1; eu]),
                    ints(&[constants::POLICY_WORDS as usize], &policy),
                    Input::I32(0),
                    ints(&[constants::STEP_WORDS as usize], &step),
                    ints(&[slots, width], &vec![0; slots * width]),
                    ints(&[constants::SOURCE_ROWS as usize, mu, ku], &vec![0; constants::SOURCE_ROWS as usize * mu * ku]),
                    ints(&[FLIGHT as usize, routing_words as usize], &vec![0; FLIGHT as usize * routing_words as usize]),
                    ints(&[FLIGHT as usize, rows as usize, eu], &vec![0; keys_words]),
                    ints(&[constants::HINT_ENTRY(choices) as usize], &vec![0; constants::HINT_ENTRY(choices) as usize]),
                    ints(&[3], &[0; 3]),
                    Input::I32(LANDING),
                ];
                let bindings = [
                    ("NW", DType::BF16),
                    ("RW", DType::BF16),
                    ("SRW", DType::F32),
                    ("SGW", DType::BF16),
                    ("SUW", DType::BF16),
                    ("A", DType::BF16),
                ];
                let outcome = interpret(crate::common::kernels(), "routed_route_streamed", &bindings, inputs);
                (as_i32(input(&outcome, 6)), as_i32(input(&outcome, 19)))
            };
            let routes = read_i32(&routes);
            assert_eq!(routes, reference_routes, "{case}: routes against the reference");
            let key = |words: &[i32], row: usize, expert: usize| {
                f32::from_bits(words[(slot as usize * rows as usize + row) * eu + expert] as u32)
            };
            let keyed = read_i32(&keys);
            let record = read_i32(&routing);
            let record = &record[slot as usize * routing_words as usize..][..routing_words as usize];
            assert_eq!(record[constants::RING_SEQUENCE as usize], STREAMING_SEQUENCE, "{case}: the record's step");
            assert_eq!(record[constants::RING_COUNT as usize], choices as i32, "{case}: the record's count");
            for row in 0..mu {
                for expert in 0..eu {
                    let (device_key, reference_key) = (key(&keyed, row, expert), key(&reference_keys, row, expert));
                    assert!(
                        (device_key - reference_key).abs() <= 1e-4 * reference_key.abs(),
                        "{case}: row {row}'s key of expert {expert}: {device_key} against the reference's {reference_key}"
                    );
                }
                // The record's choices of the row, best first (rank r at
                // slot K - 1 - r), and the top K of the row's recorded keys.
                let chosen = (0..ku)
                    .map(|rank| record[constants::ROUTING_EXPERT((row * ku + ku - 1 - rank) as u64) as usize])
                    .collect::<Vec<_>>();
                let mut ranked = (0..eu).collect::<Vec<_>>();
                ranked.sort_by(|&a, &b| key(&keyed, row, b).total_cmp(&key(&keyed, row, a)).then(b.cmp(&a)));
                let top = ranked[..ku].iter().map(|&expert| expert as i32).collect::<Vec<_>>();
                assert_eq!(chosen, top, "{case}: row {row}'s recorded choices against its keys' top {k}");
                assert_eq!(chosen, (0..ku).map(|rank| routes[row * ku + ku - 1 - rank]).collect::<Vec<_>>(), "{case}: row {row}'s routes");
                if whole != 0 {
                    for choice in 0..ku {
                        let c = (row * ku + choice) as u64;
                        assert_eq!(
                            record[constants::ROUTING_SOURCE(choices, c) as usize],
                            HOME + record[constants::ROUTING_EXPERT(c) as usize],
                            "{case}: a whole layer's source of choice {c}"
                        );
                    }
                }
            }
        }
    }
}

/// A streamed step whose every choice is held requests nothing, with or
/// without front rows: its file slot is not requested, and its route writes
/// its landing grid empty, `[0, 1, 1]`, over whatever the grid held, so
/// its landing launch dispatches nothing. `routed_select_streamed` over
/// [`three_rows`].
#[test]
fn a_streamed_step_that_requests_nothing_lands_nothing() {
    let Some(device) = streaming_device() else {
        return;
    };
    let (_, _, e, _) = THREE_ROWS;
    let held = (0..e).map(|expert| (expert as i32, 0, u64::MAX)).collect::<Vec<_>>();
    for front in [0b000, 0b110] {
        let resolved = resolve_three_rows(&device, &policy_words(0, 0, [0; 3]), front, &held);
        let case = format!("routed_select_streamed on {:?}, front rows {front:#b}", device.backend());
        assert!(read_i32(&resolved.file).iter().all(|&word| word == 0), "{case}: file table");
        assert_eq!(read_i32(&resolved.land_grid), [0, 1, 1], "{case}: landing grid");
    }
}

/// What the reference `routed_select_streamed` wrote of the words
/// [`a_streamed_route_skips_what_only_a_front_row_lacks`]'s general route
/// resolves (its `m` rows of `residual` over `e` experts of `router`, front
/// rows `front`, its guesses' kinds too), from the same zeroed words.
struct ReferenceResolve {
    routes: Vec<i32>,
    sources: Vec<i32>,
    file: Vec<i32>,
    routing: Vec<i32>,
    front: Vec<i32>,
    land_grid: Vec<i32>,
}

#[allow(clippy::too_many_arguments)]
fn reference_select_resolve(
    m: u64,
    h: u64,
    e: u64,
    k: u64,
    residual: &[f32],
    router: &[f32],
    front: i32,
    table: &SlotTable,
) -> ReferenceResolve {
    let (mu, hu, eu, ku) = (m as usize, h as usize, e as usize, k as usize);
    let choices = mu * ku;
    let map = one_held(e);
    let step = guessed_step(front);
    let policy = policy_words(0, 0, [0; 3]);
    let [slots, width] = table.extents().map(|extent| extent as usize);
    let routing_words = constants::ROUTING_WORDS(choices as u64) as usize;
    let ring = constants::HINT_ENTRY(choices as u64) as usize;
    let inputs = vec![
        floats(&[mu, hu], residual),
        floats(&[hu], &vec![1.0; hu]),
        floats(&[hu], &vec![1.0; hu]),
        floats(&[eu, hu], router),
        floats(&[eu], &vec![0.0; eu]),
        floats(&[0], &[]),
        ints(&[mu, ku], &vec![0; choices]),
        floats(&[mu, ku], &vec![0.0; choices]),
        Input::F32(1e-6),
        Input::I32(0),
        Input::I32(0),
        Input::F32(1e-6),
        Input::F32(1.0),
        ints(&[eu, constants::MAP_WORDS as usize], &map_words(&map)),
        ints(&[eu], &vec![-1; eu]),
        ints(&[constants::POLICY_WORDS as usize], &policy),
        Input::I32(0),
        ints(&[constants::STEP_WORDS as usize], &step),
        ints(&[slots, width], &vec![0; slots * width]),
        ints(&[constants::SOURCE_ROWS as usize, mu, ku], &vec![0; constants::SOURCE_ROWS as usize * choices]),
        ints(&[FLIGHT as usize, routing_words], &vec![0; FLIGHT as usize * routing_words]),
        ints(&[FLIGHT as usize, constants::VERIFY_ROWS as usize, eu], &vec![0; FLIGHT as usize * constants::VERIFY_ROWS as usize * eu]),
        ints(&[ring], &vec![0; ring]),
        ints(&[3], &[0; 3]),
        Input::I32(LANDING),
    ];
    let bindings = [("NW", DType::F32), ("RNW", DType::F32), ("RW", DType::F32), ("A", DType::BF16)];
    let outcome = interpret(crate::common::kernels(), "routed_select_streamed", &bindings, inputs);
    ReferenceResolve {
        routes: as_i32(input(&outcome, 6)),
        file: as_i32(input(&outcome, 18)),
        sources: as_i32(input(&outcome, 19)),
        routing: as_i32(input(&outcome, 20)),
        front: as_i32(input(&outcome, 22)),
        land_grid: as_i32(input(&outcome, 23)),
    }
}

/// What a streamed route of `routes` (rows of `k` choices) and their routing
/// `weights` (the weights it wrote, which its front ring names by their
/// magnitudes' bits) resolved with front rows `front` (STEP_FRONT's bits,
/// its guesses' kinds too), expert 1 held in record 4 and every other absent: the `[sources, file, routing record,
/// front ring, landing grid]` it wrote at step `STREAMING_SEQUENCE` of
/// `table`, as [`a_streamed_route_skips_what_only_a_front_row_lacks`]
/// states. The experts the front rows skipped, ascending.
fn assert_front_resolve(
    case: &str,
    front: i32,
    k: u64,
    routes: &[i32],
    weights: &[f32],
    table: &SlotTable,
    [sources, file, routing, front_ring, land_grid]: [&seismic::Tensor; 5],
) -> Vec<i32> {
    let (ku, choices) = (k as usize, routes.len());
    let rows = choices / ku;
    let sequence = STREAMING_SEQUENCE as u64;
    let held = |expert: i32| -1 - expert;
    let is_front = |row: usize| front >> row & 1 == 1;
    let exact_chooses = |expert: i32| (0..rows).any(|row| !is_front(row) && routes[row * ku..][..ku].contains(&expert));
    // Expert 1's source as resolved, record 4; any other's none.
    let resolved = |expert: i32| if expert == 1 { 4 } else { -1 };
    // Whether a choice is computed (`resolve`'s `computes`).
    let computes = |row: usize, expert: i32| !is_front(row) || expert == 1 || exact_chooses(expert);
    let expected_sources = (0..choices)
        .map(|choice| if computes(choice / ku, routes[choice]) { resolved(routes[choice]) } else { -1 })
        .collect::<Vec<_>>();
    assert_eq!(&read_i32(sources)[..choices], &expected_sources, "{case}: sources");
    let requested = (0..choices)
        .map(|choice| {
            let expert = routes[choice];
            if expert != 1 && computes(choice / ku, expert) { expert } else { held(expert) }
        })
        .collect::<Vec<_>>();
    let file = read_i32(file);
    let slot = table.lane(sequence, 0) as usize;
    let words = (0..choices as u64)
        .map(|choice| file[slot + (constants::FILE_ENTRY(choice) + constants::FILE_CHOICE) as usize])
        .collect::<Vec<_>>();
    assert_eq!(words, requested, "{case}: request words");
    let routing_words = constants::ROUTING_WORDS(choices as u64) as usize;
    let record = (sequence % FLIGHT) as usize * routing_words;
    let routing = read_i32(routing);
    assert_eq!(&routing[record + constants::ROUTING_EXPERT(0) as usize..][..choices], routes, "{case}: routing record");
    // Every choice's source as resolved, whatever the front rows compute.
    assert_eq!(
        &routing[record + constants::ROUTING_SOURCE(choices as u64, 0) as usize..][..choices],
        &routes.iter().map(|&expert| resolved(expert)).collect::<Vec<_>>(),
        "{case}: routing record sources"
    );
    assert_eq!(routing[record + constants::RING_GUESSES as usize], front_kinds(front), "{case}: routing record's guesses' kinds");
    // Row 0, exact, always routes to an absent expert: the step requests,
    // so its landing launch lands.
    assert_eq!(read_i32(land_grid), [LANDING, 1, 1], "{case}: landing grid");
    // Rank-major, then row: every front row's choice, with its rank, row
    // and weight's word.
    let entries = (0..ku)
        .flat_map(|rank| (0..rows).map(move |row| (rank, row)))
        .filter(|&(_, row)| is_front(row))
        .map(|(rank, row)| {
            let choice = row * ku + ku - 1 - rank;
            (routes[choice], rank, row, (weights[choice].to_bits() & 0x7fff_ffff) as i32)
        })
        .collect::<Vec<_>>();
    let ring = constants::HINT_ENTRY(choices as u64) as usize;
    let mut expected = vec![0; ring];
    if front != 0 {
        expected[constants::RING_SEQUENCE as usize] = sequence as u32 as i32;
        expected[constants::RING_COUNT as usize] = entries.len() as i32;
        expected[constants::RING_GUESSES as usize] = front_kinds(front);
        for (index, &(expert, rank, row, weight)) in entries.iter().enumerate() {
            let entry = constants::HINT_ENTRY(index as u64) as usize;
            expected[entry + constants::HINT_EXPERT as usize] = expert;
            expected[entry + constants::HINT_RANK as usize] = rank as i32;
            expected[entry + constants::HINT_ROW as usize] = row as i32;
            expected[entry + constants::HINT_WEIGHT as usize] = weight;
        }
    }
    assert_eq!(read_i32(front_ring), expected, "{case}: front ring");
    let mut skipped = entries.into_iter().map(|(expert, ..)| expert).filter(|&expert| expert != 1 && !exact_chooses(expert)).collect::<Vec<_>>();
    skipped.sort_unstable();
    skipped.dedup();
    skipped
}

/// A landing sources each landed choice where the route assigned its
/// expert: the CPU's pending site names its tier record, the link's record
/// reads it in place; a failed read, or a failed step, sources none. A
/// layer the host tier places (V = S = 0), its step's high word set.
#[test]
fn portable_landing_routes_a_landed_record_to_its_side() {
    let module = crate::common::kernels();
    let (k, step) = (4usize, (1u64 << 32) + 7);
    // Choice 0 held, 1-3 absent: 1 the CPU's (site -2), 2 the link's, 3 the
    // CPU's with a failed read.
    let choices = [-1 - 10, 11, 12, 13];
    let records = [5, -1, -1, -1];
    let sites = [-1, -2, -1, -2];
    let landed = [(1, -2 - 4), (2, -2 - 6), (3, -1)];
    let failed = step | (u64::from(constants::SLOT_FAILED as u32) << 32);
    for (case, answer, expected_records, expected_sites) in [
        ("answered", step, [5, -1, -8, -1], [-1, 4, -1, -1]),
        ("a failed step", failed, [5, -1, -1, -1], [-1, -1, -1, -1]),
    ] {
        let (table, file) = file_words(k as u64, step, &choices, &landed, Some(answer));
        let [slots, width] = table.extents().map(|extent| extent as usize);
        let landing = landing_words(step as i32, 0, &[], 1);
        let mut inputs = vec![
            ints(&[constants::STEP_WORDS as usize], &step_words(step)),
            ints(&[slots, width], &file),
            ints(&[constants::SOURCE_ROWS as usize, 1, k], &magnitude_kernels::source_words(&records, &sites)),
            ints(&[constants::POLICY_WORDS as usize], &policy_words(0, 0, [0; 3])),
            ints(&[landing.len()], &landing),
        ];
        inputs.extend(DENSE_LANDING.map(|(_, dtype)| Input::Tensor(dtype, vec![0, 1, 1], vec![])));
        inputs.push(ints(&[3], &[1, 1, 1]));
        let outcome = interpret(&module, "routed_land", &DENSE_LANDING, inputs);
        assert_eq!(
            as_i32(input(&outcome, 2)),
            magnitude_kernels::source_words(&expected_records, &expected_sites),
            "{case}"
        );
    }
}

/// The site request writes, into its step's slot, the rows and each CPU
/// choice's (row, tier record, expert, weight bits), the count and the step
/// pair; a step with no CPU choice writes nothing.
#[test]
fn portable_site_request_writes_its_slot() {
    let module = crate::common::kernels();
    let (m, k, h) = (2usize, 2usize, 3usize);
    let q = (1u64 << 32) + 7;
    let slot = (q % FLIGHT) as usize;
    let table = SlotTable::site(FLIGHT, (m * k) as u64);
    let words = table.width() as usize;
    let rows = constants::VERIFY_ROWS as usize;
    let routes = [10, 11, 12, 13];
    let weights = [0.5f32, 0.25, 0.125, 0.75];
    let normalized = pattern(m * h, 31, 1.0);
    for (case, sites) in [("two CPU choices", [4, -1, -1, 9]), ("no CPU choice", [-1; 4])] {
        let inputs = vec![
            ints(&[constants::STEP_WORDS as usize], &step_words(q)),
            ints(&[m, k], &routes),
            ints(&[constants::SOURCE_ROWS as usize, m, k], &magnitude_kernels::source_words(&[-1; 4], &sites)),
            floats(&[m, k], &weights),
            floats(&[m, h], &normalized),
            ints(&[FLIGHT as usize, words], &vec![-9; FLIGHT as usize * words]),
            floats(&[FLIGHT as usize, rows, h], &vec![-9.0; FLIGHT as usize * rows * h]),
        ];
        let outcome = interpret(&module, "routed_site_request", &[("A", DType::F32)], inputs);
        let mut request = vec![-9; FLIGHT as usize * words];
        let mut request_rows = vec![-9.0f32; FLIGHT as usize * rows * h];
        if sites.iter().any(|&site| site >= 0) {
            let at = table.lane(q, 0);
            set_pair(&mut request, at + constants::SLOT_REQUEST, q);
            let at = at as usize;
            request[at + constants::SLOT_VIOLATED as usize] = 0;
            request[at + constants::SITE_COUNT as usize] = 2;
            for (entry, (row, record, expert, weight)) in [(0, 4, 10, 0.5f32), (1, 9, 13, 0.75)].into_iter().enumerate() {
                let entry = at + constants::SITE_ENTRY(entry as u64) as usize;
                request[entry + constants::SITE_ROW as usize] = row;
                request[entry + constants::SITE_TIER_RECORD as usize] = record;
                request[entry + constants::SITE_EXPERT as usize] = expert;
                request[entry + constants::SITE_WEIGHT_BITS as usize] = weight.to_bits() as i32;
            }
            request_rows[slot * rows * h..][..m * h].copy_from_slice(&normalized);
        }
        assert_eq!(as_i32(input(&outcome, 5)), request, "{case}: request");
        assert_eq!(
            input(&outcome, 6).into_iter().map(|value| value as f32).collect::<Vec<_>>(),
            request_rows,
            "{case}: rows"
        );
    }
}

/// The general combine adds the CPU's answer after the slot-order weighted
/// sum of the GPU's choices, only when a choice is the CPU's and its slot
/// is requested for the step and answered with it (not its failure, nor
/// another step's answer or request).
#[test]
fn portable_sited_combine_adds_an_answered_step_alone() {
    let module = crate::common::kernels();
    let (m, k, h, q) = (2usize, 2usize, 3usize, 7u64);
    let slot = (q % FLIGHT) as usize;
    let rows = constants::VERIFY_ROWS as usize;
    let base = pattern(m * h, 41, 1.0);
    let published = pattern(m * k * h, 43, 1.0);
    let weights = pattern(m * k, 47, 0.5);
    let output = pattern(FLIGHT as usize * rows * h, 53, 1.0);
    let failed = q | (u64::from(constants::SLOT_FAILED as u32) << 32);
    let table = SlotTable::site(FLIGHT, (m * k) as u64);
    for (case, sites, request, answer, added) in [
        ("answered", [-1, 4, -1, -1], q, q, true),
        ("failed", [-1, 4, -1, -1], q, failed, false),
        ("another step's answer", [-1, 4, -1, -1], q, q - 3, false),
        ("another step's request", [-1, 4, -1, -1], q - 3, q, false),
        ("no CPU choice", [-1; 4], q, q, false),
    ] {
        let mut site = vec![0; table.words() as usize];
        set_pair(&mut site, table.lane(q, 0) + constants::SLOT_REQUEST, request);
        set_pair(&mut site, table.lane(q, 0) + constants::SLOT_ANSWER, answer);
        let [slots, width] = table.extents().map(|extent| extent as usize);
        let inputs = vec![
            floats(&[m, h], &base),
            floats(&[m, k, h], &published),
            ints(&[constants::SOURCE_ROWS as usize, m, k], &magnitude_kernels::source_words(&[-1; 4], &sites)),
            floats(&[m, k], &weights),
            ints(&[constants::STEP_WORDS as usize], &step_words(q)),
            ints(&[slots, width], &site),
            floats(&[FLIGHT as usize, rows, h], &output),
        ];
        let outcome = interpret(&module, "routed_down_sited", &[("A", DType::F32), ("R", DType::F32)], inputs);
        let expected = (0..m * h)
            .map(|at| {
                let (row, column) = (at / h, at % h);
                let mut selected = 0.0f32;
                for choice in 0..k {
                    selected = weights[row * k + choice].mul_add(published[(row * k + choice) * h + column], selected);
                }
                if added {
                    selected += output[(slot * rows + row) * h + column];
                }
                (base[at] + selected).to_bits()
            })
            .collect::<Vec<_>>();
        let value = result(&outcome, 0).into_iter().map(|value| (value as f32).to_bits()).collect::<Vec<_>>();
        assert_eq!(value, expected, "{case}");
    }
}

/// Set in the environment of the host a dead-host test ends.
const DEAD_HOST: &str = "MAGNITUDE_TEST_DEAD_HOST";

/// What the host prints once its waiting launch is set up, and once its
/// wait returned.
const WAITING: &str = "dead host: waiting";
const RETURNED: &str = "dead host: its wait returned";
const EXPIRED: &str = "dead host: its wait stored its expiry";
/// What the host prints once it measured its device's patience
/// ([`patience`]), then the milliseconds.
const PATIENCE: &str = "dead host: patience ms";

/// This device's patience, as its own wait records it: a landing's wait on
/// a requested slot never answered nor pulsed, from its call until it ends on
/// its still pulse, having stored its expiry (PULSE_SPINS spins at the
/// device's own cost of a spin, which is longer across a discrete device's
/// bus than on unified memory). What a test of a wait outlasting its
/// patience times itself by.
fn patience(device: &seismic::Device) -> std::time::Duration {
    let q = STREAMING_SEQUENCE as u64;
    let (table, words) = file_words(1, q, &[0], &[], None);
    let mut wait = TierLanding::new(device);
    let mut file = shared_i32(device, &table.extents(), &words);
    let mut sources = i32_tensor(device, &[constants::SOURCE_ROWS, 1, 1], &[-1; constants::SOURCE_ROWS as usize]);
    let began = std::time::Instant::now();
    wait.call(&mut file, &mut sources).unwrap();
    let patience = began.elapsed();
    assert_eq!(
        file.load_acquire_u64(4 * (table.lane(q, 0) + constants::SLOT_EXPIRED)).unwrap(),
        q,
        "an unpulsed wait returned without storing its expiry"
    );
    patience
}

/// A live host's long wait is answered, not given up: its pulse beats
/// throughout (the acquire load the wait reads it by sees each beat), so a
/// landing's wait the host answers at twice the device's patience, as a still
/// pulse's wait on this device recorded it ([`patience`]), returns the
/// answer, its choice sourced from the record the answer names, and stores
/// no expiry.
#[test]
fn a_live_hosts_long_wait_is_answered() {
    let Some(device) = streaming_device() else {
        return;
    };
    if !device.backend().parks_in_kernels() {
        return;
    }
    let q = STREAMING_SEQUENCE as u64;
    let (table, words) = file_words(1, q, &[0], &[], None);
    let mut wait = TierLanding::new(&device);
    let mut file = shared_i32(&device, &table.extents(), &words);
    let mut sources = i32_tensor(&device, &[constants::SOURCE_ROWS, 1, 1], &[-1; constants::SOURCE_ROWS as usize]);
    let late = 2 * patience(&device);
    let _pulse = pulse(&file, table);
    let slot = table.lane(q, 0);
    let answered = std::time::Instant::now();
    std::thread::scope(|scope| {
        let answerer = file.clone();
        scope.spawn(move || {
            std::thread::sleep(late);
            // Bank record 5 for choice 0, then the answer.
            answerer.store_release(4 * (slot + constants::FILE_ENTRY(0) + constants::FILE_RECORD), 5).unwrap();
            answerer.store_release(4 * (slot + constants::SLOT_ANSWER), q as u32).unwrap();
        });
        wait.call(&mut file, &mut sources).unwrap();
    });
    assert!(answered.elapsed() >= late, "the wait returned before its answer");
    assert_ne!(file.load_acquire_u64(4 * (slot + constants::SLOT_EXPIRED)).unwrap(), q, "a pulsed wait stored an expiry");
    assert_eq!(read_i32(&sources), magnitude_kernels::source_words(&[5], &[-1]), "the wait returned the answer");
}

/// [`dead_hosts_wait_ends`], the host killed (a stopped run's process tree).
#[test]
#[ignore = "kills a process mid spinning wait and watches its GPU go idle; needs the GPU to itself: run it by name"]
fn a_killed_hosts_wait_ends() {
    dead_hosts_wait_ends("kill");
}

/// [`dead_hosts_wait_ends`], the host alive but its pulse stopped: its wait
/// returns as on its failure (the give-up itself, which on CUDA the driver's
/// end of a dead host's kernel would hide).
#[test]
#[ignore = "stops a host's pulse mid spinning wait and watches the wait end; needs the GPU to itself: run it by name"]
fn a_stalled_hosts_wait_ends() {
    dead_hosts_wait_ends("stall");
}

/// [`dead_hosts_wait_ends`], the host aborting itself (the exchange's
/// watchdog, a panic).
#[test]
#[ignore = "aborts a process mid spinning wait and watches its GPU go idle; needs the GPU to itself: run it by name"]
fn an_aborted_hosts_wait_ends() {
    dead_hosts_wait_ends("abort");
}

/// A host that dies while its launch waits on a slot it never answers
/// leaves no kernel spinning: its process ends at once, but Metal does not
/// end a dead process's running kernel (its GPU stayed busy, its memory
/// held, until a reboot; NVIDIA's driver ends it within a second), so the
/// wait ends once its slot's pulse, which a live host advances every
/// PULSE_MS, has stood still for its patience, PULSE_SPINS spins at the
/// device's own cost of a spin, which the host measures first
/// ([`patience`]) and the test times itself by. The test runs itself as
/// that host (one landing's wait on a requested slot, pulsed), ends it by
/// `death` at twice that patience, and requires its GPU idle again within
/// twice its patience and the five seconds its settling is read over; it
/// needs the GPU to itself.
fn dead_hosts_wait_ends(death: &str) {
    use std::io::BufRead;
    // One host at a time: each test reads the whole GPU's business.
    static ALONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if std::env::var_os(DEAD_HOST).is_some() {
        dead_host(death);
    }
    let _alone = ALONE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(device) = streaming_device() else {
        return;
    };
    if !device.backend().parks_in_kernels() {
        return;
    }
    drop(device);
    let test = match death {
        "kill" => "a_killed_hosts_wait_ends",
        "stall" => "a_stalled_hosts_wait_ends",
        _ => "an_aborted_hosts_wait_ends",
    };
    assert!(gpu_settles(std::time::Duration::from_secs(30)).is_some(), "the GPU is idle before the host");
    let mut host = std::process::Command::new(std::env::current_exe().unwrap())
        .args([test, "--exact", "--ignored", "--nocapture", "--test-threads=1"])
        .env(DEAD_HOST, death)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = host.stdout.take().unwrap();
    let (lines, ready) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = lines.send(line);
        }
    });
    // Whether a line of the host's holds `wanted` within `within`.
    let await_line = |wanted: &str, within: std::time::Duration| {
        let deadline = std::time::Instant::now() + within;
        while let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) {
            match ready.recv_timeout(left) {
                Ok(line) if line.contains(wanted) => return true,
                Ok(_) => {}
                Err(_) => return false,
            }
        }
        false
    };
    // The host's patience, then its wait.
    let patience = {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut measured = None;
        while let (None, Some(left)) = (measured, deadline.checked_duration_since(std::time::Instant::now())) {
            match ready.recv_timeout(left) {
                Ok(line) => {
                    measured = line.find(PATIENCE).and_then(|at| line[at + PATIENCE.len()..].split_whitespace().next()?.parse::<u64>().ok())
                }
                Err(_) => break,
            }
        }
        std::time::Duration::from_millis(measured.unwrap_or_else(|| {
            let _ = host.kill();
            panic!("the host never measured its patience: {:?}", host.wait())
        }))
    };
    if !await_line(WAITING, std::time::Duration::from_secs(60)) {
        let _ = host.kill();
        panic!("the host never began its wait: {:?}", host.wait());
    }
    // While the host pulses, its wait spins on past its patience; an
    // aborting host aborts, a stalling one stops its pulse, at twice its
    // patience after its call.
    std::thread::sleep(patience * 3 / 2);
    assert_eq!(gpu_busy(), Some(true), "the live host's wait spins on");
    match death {
        "kill" => host.kill().unwrap(),
        "stall" => {
            let within = 3 * patience;
            let returned = await_line(RETURNED, within);
            let expired = returned && await_line(EXPIRED, std::time::Duration::from_secs(1));
            host.kill().unwrap();
            assert!(returned, "the host's wait did not return within {within:?} of the point its pulse stops at");
            assert!(expired, "the host's wait returned without storing its expiry");
        }
        _ => {}
    }
    let status = host.wait().unwrap();
    let bound = 2 * patience + std::time::Duration::from_secs(5);
    let busy = gpu_settles(bound).unwrap_or_else(|| panic!("the dead host's GPU is still busy {bound:?} after it ended ({status})"));
    eprintln!("{death}: the dead host ({status}) left its GPU busy {busy:?}");
}

/// How long the GPU stayed busy before it was idle for two seconds on end,
/// within `within`; none where it never was.
fn gpu_settles(within: std::time::Duration) -> Option<std::time::Duration> {
    let began = std::time::Instant::now();
    let mut idle_since = None;
    while began.elapsed() < within {
        match gpu_busy() {
            Some(false) => {
                let since = *idle_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() >= std::time::Duration::from_secs(2) {
                    return Some(since - began);
                }
            }
            _ => idle_since = None,
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    None
}

/// Whether the GPU runs anything, as its driver reports it: Metal's
/// accelerator's device utilization, NVIDIA's utilization; none where
/// neither reports.
fn gpu_busy() -> Option<bool> {
    let read = |program: &str, args: &[&str]| {
        std::process::Command::new(program).args(args).output().ok().map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
    };
    let percent = if cfg!(target_os = "macos") {
        let report = read("ioreg", &["-r", "-d", "1", "-c", "IOAccelerator"])?;
        let at = report.find("\"Device Utilization %\"=")? + "\"Device Utilization %\"=".len();
        report[at..].split(|c: char| !c.is_ascii_digit()).next()?.parse::<u32>().ok()?
    } else {
        read("nvidia-smi", &["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])?.lines().next()?.trim().parse::<u32>().ok()?
    };
    Some(percent > 0)
}

/// The element bindings a dense landing's portable body runs under
/// (`ServedForm::dense` into F32): BF16 stored, F32 resident.
const DENSE_LANDING: [(&str, DType); 6] =
    [("SG", DType::BF16), ("SU", DType::BF16), ("SD", DType::BF16), ("UG", DType::F32), ("UU", DType::F32), ("UD", DType::F32)];

/// A decode layer's landing of one choice where the host tier places
/// (V = S = 0): nothing lands, the launch waits on its slot and sources,
/// one threadgroup; what the pulse's tests wait on.
struct TierLanding {
    kernel: seismic::NativeKernel<routed_land::Entry>,
    charges: seismic::Charges,
    step: seismic::Tensor,
    policy: seismic::Tensor,
    landing: seismic::Tensor,
    staging: [seismic::Tensor; 3],
    bank: [seismic::Tensor; 3],
    grid: seismic::Tensor,
}

impl TierLanding {
    fn new(device: &seismic::Device) -> Self {
        let form = ServedForm::dense(false, seismic::Element::f32(), 1, 1);
        Self {
            kernel: form.land_kernel(device, 1),
            charges: seismic::Charges::tally(device),
            step: streaming_step(device),
            policy: shared_i32(device, &[constants::POLICY_WORDS], &[0; constants::POLICY_WORDS as usize]),
            landing: shared_i32(device, &[constants::LANDING_ENTRY(1)], &landing_words(STREAMING_SEQUENCE, 0, &[], 1)),
            staging: form.records(device, false, 0, |_, _| None),
            bank: form.records(device, true, 0, |_, _| None),
            grid: land_grid(device, false),
        }
    }

    fn call(&mut self, file: &mut seismic::Tensor, sources: &mut seismic::Tensor) -> Result<(), seismic::CallError> {
        let [gate, up, down] = self.bank.each_mut();
        self.kernel
            .call(&self.charges, routed_land::Args {
                step: &self.step,
                file,
                sources,
                policy: &self.policy,
                landing: &mut self.landing,
                staging_gate: &self.staging[0],
                staging_up: &self.staging[1],
                staging_down: &self.staging[2],
                gate,
                up,
                down,
                grid: &self.grid,
            })
            .map(|_| ())
    }
}

/// The host a dead-host test ends: one landing's wait on step q's
/// requested, unanswered slot of a one-choice file table, its pulse
/// advanced every PULSE_MS as a live host's watchdog does, then nothing but
/// its `death`: the kill, its own abort at twice its device's patience
/// ([`patience`], measured first) after its call, or its pulse stopped then
/// (a stall).
fn dead_host(death: &str) -> ! {
    let device = streaming_device().unwrap();
    let measured = patience(&device);
    println!("{PATIENCE} {}", measured.as_millis());
    let late = 2 * measured;
    let q = STREAMING_SEQUENCE as u64;
    let (table, words) = file_words(1, q, &[0], &[], None);
    let mut wait = TierLanding::new(&device);
    let mut file = shared_i32(&device, &table.extents(), &words);
    let mut sources = i32_tensor(&device, &[constants::SOURCE_ROWS, 1, 1], &[-1; constants::SOURCE_ROWS as usize]);
    // A stalling host stops its pulse at twice its patience after its call;
    // any other beats until it dies.
    let beating = pulse(&file, table);
    let _beating = if death == "stall" {
        std::thread::spawn(move || {
            std::thread::sleep(late);
            drop(beating);
        });
        None
    } else {
        Some(beating)
    };
    if death == "abort" {
        std::thread::spawn(move || {
            std::thread::sleep(late);
            std::process::abort();
        });
    }
    println!("{WAITING}");
    let returned = wait.call(&mut file, &mut sources);
    println!("{RETURNED} {:?}", returned.map(|_| ()));
    // A wait that ended on its still pulse stores its step as its slot's
    // SLOT_EXPIRED, by which the live host interrupts exactly that step.
    if file.load_acquire_u64(4 * (table.lane(q, 0) + constants::SLOT_EXPIRED)).unwrap() == q {
        println!("{EXPIRED}");
    }
    loop {
        std::thread::park();
    }
}


/// The time of `routed_select_streamed` on Metal at DeepSeek V4.1 Flash's
/// decode shape: one row (M = 1) over its 5120-wide residual, a BF16 router
/// of 384 experts and BF16 norms (as its GGUF binds them), top 6 by
/// sqrt-softplus. Three launches as a step encodes them: a whole group's
/// layer whose next layer holds every expert (no leads; the resident load),
/// a whole group's layer with both leads' hints (AHEAD 2), and a streamed
/// layer with both leads, every expert absent (each choice resolved absent
/// and requested). SIMDGROUPS stays at its default 8: each launch is timed
/// as the default of its own survey, every parameter fixed, the formed
/// default its only program. It prints each one's median device seconds a
/// call and, as context, the rate at which it reads its routers.
#[test]
#[ignore = "times routed_select_streamed at V4.1's decode shape on Metal; run it by name under a timed hold"]
fn routed_select_streamed_at_deepseek_v41_decode_shape_times() {
    let Some(device) = streaming_device().filter(|device| device.backend() == seismic::BackendName::Metal) else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (h, e, k, m) = (5120u64, 384u64, 6u64, 1u64);
    let choices = m * k;
    let bf16 = seismic::Element::bf16();
    let f32 = seismic::Element::f32();
    let elements = || routed_select_streamed::Elements { NW: bf16, RNW: bf16, RW: bf16, A: bf16 };
    let entry = routed_select_streamed::native_entry_with(elements());
    let table = SlotTable::file(FLIGHT, choices);
    let zeros = |element, shape: &[u64]| seismic::Tensor::zeros(&charges, element, shape).unwrap();
    let words = |shape: &[u64]| shared_i32(&device, shape, &vec![0; shape.iter().product::<u64>() as usize]);
    let residual = f32_tensor(&device, &[m, h], &(0..h).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect::<Vec<_>>());
    let router = |seed: u64| {
        let values = (0..e * h).map(|i| (((i * 2654435761 + seed) % 1009) as f32 - 504.0) / 5040.0).collect::<Vec<_>>();
        bf16_tensor(&device, &[e, h], &values)
    };
    let ones = bf16_tensor(&device, &[h], &vec![1.0; h as usize]);
    let router0 = router(1);
    let map = shared_i32(&device, &[e, constants::MAP_WORDS], &map_words(&vec![(-1, 0, 0); e as usize]));
    let tier = seismic::Tensor::words(&charges, seismic::Element::i32(), &[e]).unwrap();
    (0..e).for_each(|expert| tier.store_release(expert * 4, u32::MAX).unwrap());
    let step = i32_tensor(&device, &[constants::STEP_WORDS], &step_words(STREAMING_SEQUENCE as u64));
    for (name, whole) in [("whole", 1), ("streamed", 0)] {
        let policy = shared_i32(&device, &[constants::POLICY_WORDS], &policy_words(whole, 0, [0; 3]));
        let statics = [("H", h), ("E", e), ("K", k), ("ES", 0), ("TM", e)];
        let implementation = entry.offered(&device).unwrap().unwrap();
        let statics = implementation
            .statics
            .iter()
            .fold(seismic::NativeSpecialization::new(), |specialization, name| {
                let (_, value) = statics.iter().find(|(candidate, _)| *candidate == name.as_str()).unwrap();
                specialization.with_static(name.clone(), *value)
            });
        let configuration = implementation
            .default_specialization(&statics)
            .unwrap()
            .with_param("SIMDGROUPS", 8);
        let mut routes = zeros(seismic::Element::i32(), &[m, k]);
        let mut weights = zeros(f32, &[m, k]);
        let mut file = words(&table.extents());
        let mut sources = zeros(seismic::Element::i32(), &[constants::SOURCE_ROWS, m, k]);
        let mut front = words(&[constants::HINT_ENTRY(choices)]);
        let mut routing = words(&[FLIGHT, constants::ROUTING_WORDS(choices)]);
        let mut keys = key_records(&device, e);
        let mut land_grid = words(&[3]);
        let bias = zeros(f32, &[e]);
        let expert_scale = zeros(f32, &[0]);
        let point = seismic::TuningPoint {
            label: "m1".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![routed_select_streamed::Args {
                residual: &residual,
                norm: &ones,
                router_norm: &ones,
                router: &router0,
                bias: &bias,
                expert_scale: &expert_scale,
                routes: &mut routes,
                weights: &mut weights,
                epsilon: 1e-6,
                score: 2,
                normalization: 1,
                normalization_epsilon: 1e-6,
                scale: 1.0,
                map: &map,
                tier: &tier,
                policy: &policy,
                sited: 0,
                step: &step,
                file: &mut file,
                front: &mut front,
                sources: &mut sources,
                routing: &mut routing,
                keys: &mut keys,
                land_grid: &mut land_grid,
                land_groups: 1,
            }],
            initialize: None,
            written: Default::default(),
        };
        let result = routed_select_streamed::native_tune_with(
            &charges,
            elements(),
            &configuration,
            vec![point],
            tuning_requirement(),
            seismic::Strategy::Survey(seismic::SurveyPlan {
                samples: 15,
                min_sample_seconds: 0.001,
                domains: Default::default(),
            }),
        )
        .unwrap();
        let [record] = &result.configurations[..] else {
            panic!("{name}: a fixed configuration surveys itself alone");
        };
        let seismic::Outcome::Measured { points, .. } = &record.outcome else {
            panic!("{name}: the default is measured");
        };
        let median = points[0].median_seconds;
        let bytes = e * h * 2;
        println!(
            "routed_select_streamed v41 m1 {name}: {:.1} us a call; {bytes} router bytes at {:.1} GB/s",
            median * 1e6,
            bytes as f64 / median / 1e9
        );
    }
}


/// Every backend's `routed_route` and `routed_select` accumulate both the RMS
/// square sum and every router dot in the CPU 8-lane reference order
/// (`((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))`), so ulp-sensitive
/// cancellations that flip under 1-, 2-, 4-, 16- or 32-lane reductions or a
/// different butterfly order route to the exact same experts as CPU across
/// decode GEMV (`M <= 8`), prefill GEMM (`M > 8`), partial tiles, multi-chunk
/// widths (`H = 2048, 4096`), `bf16` and `f16` activations, and `q8g32s`
/// packed router weights.
#[test]
fn route_choice_follows_the_reference_order() {
    let catalog = seismic::Devices::discover(&[]).expect("discover devices");
    let cpu = catalog
        .open_backend(seismic::BackendName::Cpu)
        .expect("cpu reference device");
    let gpus = seismic_native_tests::devices(&[
        seismic::BackendName::Cpu,
        seismic::BackendName::Metal,
        seismic::BackendName::Cuda,
        seismic::BackendName::Vulkan,
    ])
    .into_iter()
    .filter(|d| d.backend() != seismic::BackendName::Cpu)
    .collect::<Vec<_>>();
    let choices = 5usize;
    let shapes = [
        (64usize, 23usize, 1usize),
        (64, 128, 5),
        (64, 240, 9),
        (2048, 128, 1),
        (2048, 23, 33),
        (4096, 240, 1),
        (4096, 128, 9),
    ];

    for &(h, e, m) in &shapes {
        let residual_vals = vec![1.0f32; m * h];
        let mut norm_vals = vec![1.0f32; h];
        norm_vals[0] = 4096.0;
        norm_vals[4] = 4096.0;
        norm_vals[8] = 4096.0;
        let shared_vals = {
            let mut s = vec![0.0f32; h];
            s[0] = 4096.0;
            s[8] = -4096.0;
            s[16] = 1.0;
            s
        };

        let tie_lo = 2usize;
        let tie_hi = e - 3;
        let trap_16_32 = 5usize;
        let trap_1_2_4 = e / 2;
        let trap_tree = e - 7;
        let chunk_tail = h - 16;

        let mut router_vals = vec![0.0f32; e * h];
        for ex in 0..e {
            router_vals[ex * h + chunk_tail] = 0.5;
        }
        for &ex in &[tie_lo, tie_hi] {
            router_vals[ex * h + chunk_tail] = 0.0;
            router_vals[ex * h + 24] = 2.0;
        }
        router_vals[trap_16_32 * h + chunk_tail] = 0.0;
        router_vals[trap_16_32 * h] = 4096.0;
        router_vals[trap_16_32 * h + 8] = -4096.0;
        router_vals[trap_16_32 * h + 16] = 1.0;

        router_vals[trap_1_2_4 * h + chunk_tail] = 0.0;
        router_vals[trap_1_2_4 * h] = 4096.0;
        router_vals[trap_1_2_4 * h + 4] = 0.000244140625;
        router_vals[trap_1_2_4 * h + 8] = -4096.0;

        router_vals[trap_tree * h + chunk_tail] = 0.0;
        router_vals[trap_tree * h] = 4096.0;
        router_vals[trap_tree * h + 1] = 1.0;
        router_vals[trap_tree * h + 4] = -4096.0;

        let bias_vals = vec![0.0f32; e];

        for (act_dtype, act_elem) in [
            (DType::BF16, seismic::Element::bf16()),
            (DType::F16, seismic::Element::f16()),
        ] {
            for rw_bf16 in [false, true] {
                let make_router = |dev: &seismic::Device| {
                    if rw_bf16 {
                        bf16_tensor(dev, &[e as u64, h as u64], &router_vals)
                    } else {
                        f32_tensor(dev, &[e as u64, h as u64], &router_vals)
                    }
                };
                let run_route = |dev: &seismic::Device| {
                    let rw_tensor = make_router(dev);
                    let spec = specialization_on(
                        dev,
                        &route_statics(
                            &[("H", h as u64), ("E", e as u64), ("K", choices as u64)],
                            h as u64,
                            0,
                        ),
                        &route_mapping(dev, 8, 1),
                    );
                    let kernel = declared::prepared(
                        dev,
                        &routed_route::native_entry_with(routed_route::Elements {
                            NW: seismic::Element::f32(),
                            RW: rw_tensor.element(),
                            SRW: seismic::Element::f32(),
                            A: act_elem,
                        }),
                        &spec,
                    )
                    .unwrap();
                    let mut routes = i32_tensor(dev, &[m as u64, choices as u64], &vec![-1; m * choices]);
                    let mut scores = f32_tensor(dev, &[m as u64, choices as u64], &vec![-1.0; m * choices]);
                    let out = kernel
                        .call(&seismic::Charges::tally(dev), routed_route::Args {
                            residual: &f32_tensor(dev, &[m as u64, h as u64], &residual_vals),
                            norm: &f32_tensor(dev, &[h as u64], &norm_vals),
                            router: &rw_tensor,
                            shared_router: &f32_tensor(dev, &[h as u64], &shared_vals),
                            routes: &mut routes,
                            scores: &mut scores,
                            eps: 0.0,
                            normalize: 1,
                            expert_scale: &absent(dev),
                        })
                        .unwrap();
                    (
                        read_i32(&routes),
                        read_f32(&scores),
                        read_activation(&out.r0, act_dtype),
                        read_f32(&out.r1),
                    )
                };

                let run_select = |dev: &seismic::Device| {
                    let rw_tensor = make_router(dev);
                    let spec = specialization_on(
                        dev,
                        &[("H", h as u64), ("E", e as u64), ("K", choices as u64), ("ES", 0)],
                        &route_mapping(dev, 8, 1),
                    );
                    let kernel = declared::prepared(
                        dev,
                        &routed_select::native_entry_with(routed_select::Elements {
                            NW: seismic::Element::f32(),
                            RNW: seismic::Element::f32(),
                            RW: rw_tensor.element(),
                            A: act_elem,
                        }),
                        &spec,
                    )
                    .unwrap();
                    let mut routes = i32_tensor(dev, &[m as u64, choices as u64], &vec![-1; m * choices]);
                    let mut weights = f32_tensor(dev, &[m as u64, choices as u64], &vec![-1.0; m * choices]);
                    let norm = f32_tensor(dev, &[h as u64], &norm_vals);
                    let out = kernel
                        .call(&seismic::Charges::tally(dev), routed_select::Args {
                            residual: &f32_tensor(dev, &[m as u64, h as u64], &residual_vals),
                            norm: &norm,
                            router_norm: &norm,
                            router: &rw_tensor,
                            bias: &f32_tensor(dev, &[e as u64], &bias_vals),
                            expert_scale: &absent(dev),
                            routes: &mut routes,
                            weights: &mut weights,
                            epsilon: 0.0,
                            score: 0,
                            normalization: 1,
                            normalization_epsilon: 1e-6,
                            scale: 1.0,
                        })
                        .unwrap();
                    (
                        read_i32(&routes),
                        read_f32(&weights),
                        read_activation(&out.value, act_dtype),
                    )
                };

                let cpu_route = run_route(&cpu);
                let mut expected_set = [
                    tie_lo as i32,
                    tie_hi as i32,
                    trap_16_32 as i32,
                    trap_1_2_4 as i32,
                    trap_tree as i32,
                ];
                expected_set.sort_unstable();
                for r in 0..m {
                    let mut got = cpu_route.0[r * choices..(r + 1) * choices].to_vec();
                    got.sort_unstable();
                    assert_eq!(got, expected_set, "cpu reference winners H={h} E={e} M={m} {act_dtype:?} bf16_rw={rw_bf16}");
                    assert!((cpu_route.3[r] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
                }

                let cpu_select = run_select(&cpu);
                for r in 0..m {
                    let mut got = cpu_select.0[r * choices..(r + 1) * choices].to_vec();
                    got.sort_unstable();
                    assert_eq!(got, expected_set, "cpu select winners H={h} E={e} M={m} {act_dtype:?} bf16_rw={rw_bf16}");
                }

                for gpu in &gpus {
                    let label = format!("{:?} H={h} E={e} M={m} A={act_dtype:?} bf16_rw={rw_bf16}", gpu.backend());
                    let gpu_route = run_route(gpu);
                    assert_eq!(gpu_route.0, cpu_route.0, "{label} routed_route routes");
                    assert_near(&format!("{label} routed_route scores"), &gpu_route.1, &cpu_route.1, 1e-6, 1e-6);
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&gpu_route.2), bits(&cpu_route.2), "{label} routed_route normalized bits");
                    assert_near(&format!("{label} routed_route shared gate"), &gpu_route.3, &cpu_route.3, 1e-6, 1e-6);

                    let gpu_select = run_select(gpu);
                    assert_eq!(gpu_select.0, cpu_select.0, "{label} routed_select routes");
                    assert_near(&format!("{label} routed_select weights"), &gpu_select.1, &cpu_select.1, 1e-6, 1e-6);
                    assert_eq!(bits(&gpu_select.2), bits(&cpu_select.2), "{label} routed_select normalized bits");
                }
            }
        }

        let seed = (h * 31 + e * 17 + m) as u32;
        let patterned_res = pattern(m * h, seed ^ 0x5a5a, 0.75);
        let patterned_norm = (0..h).map(|c| 0.75 + 0.5 * ((c % 13) as f32 / 13.0)).collect::<Vec<_>>();
        let patterned_shared = pattern(h, seed ^ 0x1234, 0.25);
        let (cpu_packed, _) = packed_weight(&cpu, "q8g32s", &[e as u64, h as u64], seed);
        let cpu_q8_kernel = declared::prepared(
            &cpu,
            &routed_route::native_entry_with(routed_route::Elements {
                NW: seismic::Element::f32(),
                RW: cpu_packed.element(),
                SRW: seismic::Element::f32(),
                A: seismic::Element::bf16(),
            }),
            &specialization_on(
                &cpu,
                &route_statics(&[("H", h as u64), ("E", e as u64), ("K", choices as u64)], h as u64, 0),
                &[],
            ),
        )
        .unwrap();
        let mut cpu_q8_routes = i32_tensor(&cpu, &[m as u64, choices as u64], &vec![-1; m * choices]);
        let mut cpu_q8_scores = f32_tensor(&cpu, &[m as u64, choices as u64], &vec![-1.0; m * choices]);
        let cpu_q8_out = cpu_q8_kernel
            .call(&seismic::Charges::tally(&cpu), routed_route::Args {
                residual: &f32_tensor(&cpu, &[m as u64, h as u64], &patterned_res),
                norm: &f32_tensor(&cpu, &[h as u64], &patterned_norm),
                router: &cpu_packed,
                shared_router: &f32_tensor(&cpu, &[h as u64], &patterned_shared),
                routes: &mut cpu_q8_routes,
                scores: &mut cpu_q8_scores,
                eps: 1e-6,
                normalize: 1,
                expert_scale: &absent(&cpu),
            })
            .unwrap();
        let expected_q8_routes = read_i32(&cpu_q8_routes);
        let expected_q8_scores = read_f32(&cpu_q8_scores);
        let expected_q8_gate = read_f32(&cpu_q8_out.r1);

        for gpu in &gpus {
            let label = format!("{:?} q8g32s H={h} E={e} M={m}", gpu.backend());
            let (gpu_packed, _) = packed_weight(gpu, "q8g32s", &[e as u64, h as u64], seed);
            let gpu_q8_kernel = declared::prepared(
                gpu,
                &routed_route::native_entry_with(routed_route::Elements {
                    NW: seismic::Element::f32(),
                    RW: gpu_packed.element(),
                    SRW: seismic::Element::f32(),
                    A: seismic::Element::bf16(),
                }),
                &specialization_on(
                    gpu,
                    &route_statics(&[("H", h as u64), ("E", e as u64), ("K", choices as u64)], h as u64, 0),
                    &route_mapping(gpu, 8, 1),
                ),
            )
            .unwrap();
            let mut gpu_q8_routes = i32_tensor(gpu, &[m as u64, choices as u64], &vec![-1; m * choices]);
            let mut gpu_q8_scores = f32_tensor(gpu, &[m as u64, choices as u64], &vec![-1.0; m * choices]);
            let gpu_q8_out = gpu_q8_kernel
                .call(&seismic::Charges::tally(gpu), routed_route::Args {
                    residual: &f32_tensor(gpu, &[m as u64, h as u64], &patterned_res),
                    norm: &f32_tensor(gpu, &[h as u64], &patterned_norm),
                    router: &gpu_packed,
                    shared_router: &f32_tensor(gpu, &[h as u64], &patterned_shared),
                    routes: &mut gpu_q8_routes,
                    scores: &mut gpu_q8_scores,
                    eps: 1e-6,
                    normalize: 1,
                    expert_scale: &absent(gpu),
                })
                .unwrap();
            assert_eq!(read_i32(&gpu_q8_routes), expected_q8_routes, "{label} routes");
            assert_near(&format!("{label} scores"), &read_f32(&gpu_q8_scores), &expected_q8_scores, 1e-5, 1e-5);
            assert_near(&format!("{label} gate"), &read_f32(&gpu_q8_out.r1), &expected_q8_gate, 1e-5, 1e-5);
        }
    }
}

/// Metal's routed `INT8` and `DOWN_INT8` admit partial 256/512-column shapes
/// (`H = 320`, `F = 640`, multiples of 32) for both Q4_K and Q8_0 experts.
#[test]
fn metal_routed_experts_int8_partial_k_block_admitted() {
    let Some(device) = seismic_native_tests::every_device()
        .into_iter()
        .find(|device| device.backend() == seismic::BackendName::Metal)
    else {
        return;
    };
    let (h, f, e) = (320u64, 640u64, 2u64);
    for repr in ["q4k", "q8g32s"] {
        let (gate, _) = packed_weight(&device, repr, &[e, f, h], 211);
        let (up, _) = packed_weight(&device, repr, &[e, f, h], 212);
        let (down, _) = packed_weight(&device, repr, &[e, h, f], 213);
        let entry = routed_experts::native_entry_with(routed_experts::Elements {
            A: seismic::Element::bf16(),
            EGW: gate.element(),
            EUW: up.element(),
            EDW: down.element(),
        });
        let statics = [("H", h), ("F", f), ("T", 32), ("GS", 0), ("US", 0), ("LB", 0)];
        for (label, params) in [
            ("INT8", &[("TALL", 1u64), ("INT8", 1)][..]),
            ("DOWN_INT8", &[("DOWN_TALL", 1u64), ("DOWN_INT8", 1)][..]),
            (
                "INT8+DOWN_INT8",
                &[("TALL", 1u64), ("INT8", 1), ("DOWN_TALL", 1), ("DOWN_INT8", 1)][..],
            ),
        ] {
            let prepared = declared::prepared(
                &device,
                &entry,
                &specialization_on(&device, &statics, params),
            );
            assert!(
                prepared.is_ok(),
                "routed_experts {label} must admit {repr} at H={h}, F={f}: {:?}",
                prepared.err()
            );
        }
    }
}
