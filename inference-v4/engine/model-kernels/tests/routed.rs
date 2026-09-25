//! Routed (mixture-of-experts) entries: the portable bodies define the
//! contract; native implementations are checked against them.

use magnitude_model_kernels::{
    routed_combine, routed_expand, routed_experts, routed_group, routed_output, routed_route,
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

const H: usize = 64;
const E: usize = 32;
const K: usize = 4;

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "routed.seismic".into(),
        text: include_str!("../kernels/routed.seismic").into(),
    });
    check_source(sources).unwrap()
}

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

struct RouteCase {
    rows: usize,
    residual: Vec<f32>,
    norm: Vec<f32>,
    router: Vec<f32>,
    shared_router: Vec<f32>,
    normalize: i32,
}

impl RouteCase {
    /// Router rows 5 and 21 are identical, so their probabilities tie
    /// exactly and the higher expert must rank first. Residuals are positive
    /// and the tied rows a positive constant, so the pair is always selected.
    /// Rows 9 and 30 tie too, without being forced into the top K.
    fn new(rows: usize, normalize: i32) -> Self {
        let mut router = pattern(E * H, 7, 0.25);
        router[5 * H..6 * H].fill(0.3);
        router.copy_within(5 * H..6 * H, 21 * H);
        router.copy_within(9 * H..10 * H, 30 * H);
        Self {
            rows,
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
        let rows = self.rows;
        let outcome = interpret(
            module,
            "routed_route",
            &[("NW", DType::F32), ("RW", DType::F32), ("A", activation)],
            vec![
                floats(&[rows, H], &self.residual),
                floats(&[H], &self.norm),
                floats(&[E, H], &self.router),
                floats(&[H], &self.shared_router),
                ints(&[rows, K], &vec![0; rows * K]),
                floats(&[rows, K], &vec![0.0; rows * K]),
                Input::F32(1e-6),
                Input::I32(self.normalize),
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
    let module = module();
    let case = RouteCase::new(3, 0);
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
    let (_, _, _, normalized_scores) = RouteCase::new(3, 1).portable(&module, DType::F32);
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

fn read_activation(tensor: &seismic::Tensor, dtype: DType) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    match dtype {
        DType::F32 => read_f32(tensor),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        other => panic!("unsupported activation {other:?}"),
    }
}

fn f32_tensor(device: &seismic::Device, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(device, seismic::Element::f32(), shape, &bytes).unwrap()
}

fn i32_tensor(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(device, seismic::Element::i32(), shape, &bytes).unwrap()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(*actual) - expected).abs();
        assert!(
            error <= tolerance * expected.abs().max(1.0),
            "{name}[{index}]: native {actual}, portable {expected}"
        );
    }
}

/// CUDA, else Metal; `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan (a CUDA
/// host also has a Vulkan device).
fn native_device() -> Option<seismic::Device> {
    let catalog = seismic::DeviceCatalog::discover().ok()?;
    match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        // A selected backend must open: never skip silently.
        Some("vulkan") => Some(
            catalog
                .open_backend(seismic::BackendName::Vulkan)
                .expect("the Vulkan device opens"),
        ),
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only vulkan is selectable"),
        None => [seismic::BackendName::Cuda, seismic::BackendName::Metal]
            .into_iter()
            .find_map(|backend| catalog.open_backend(backend).ok()),
    }
}

/// The GPU device of `native_device` when present, and the CPU device.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    native_device()
        .into_iter()
        .chain(std::iter::once(
            catalog.open_backend(seismic::BackendName::Cpu).unwrap(),
        ))
        .collect()
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
    parts: u64,
) -> seismic::NativeSpecialization {
    if is_cpu(device) {
        return seismic::NativeSpecialization::new();
    }
    specialize(&[("E", experts), ("K", choices)], &[("PARTS", parts)])
}

/// The route's hidden-axis splits on the device's backend (CUDA splits its
/// router GEMM; Metal has no split parameter).
fn route_splits(device: &seismic::Device) -> Vec<u64> {
    match device.backend() {
        seismic::BackendName::Cuda => vec![8, 4, 16],
        _ => vec![1],
    }
}

/// The route's mapping parameters for `simdgroups` and `split` (on CPU,
/// `simdgroups` router rows per work item).
fn route_mapping(
    device: &seismic::Device,
    simdgroups: u64,
    split: u64,
) -> Vec<(&'static str, u64)> {
    match device.backend() {
        seismic::BackendName::Cuda => vec![("SIMDGROUPS", simdgroups), ("SPLIT", split)],
        seismic::BackendName::Cpu => vec![("ROWS", simdgroups)],
        _ => vec![("SIMDGROUPS", simdgroups)],
    }
}

#[test]
fn native_route_matches_portable_body_for_every_mapping() {
    for device in devices() {
        native_route_matches_portable_body_for_every_mapping_on(&device);
    }
}

fn native_route_matches_portable_body_for_every_mapping_on(device: &seismic::Device) {
    let module = module();
    for (activation, element) in [
        (DType::F32, seismic::Element::f32()),
        (DType::BF16, seismic::Element::bf16()),
    ] {
        for (rows, normalize) in [(1usize, 1), (5, 0), (9, 1)] {
            let case = RouteCase::new(rows, normalize);
            let (normalized, coefficient, routes, scores) = case.portable(&module, activation);
            for (simdgroups, split) in [8, 4, 2].into_iter().flat_map(|simdgroups| {
                route_splits(&device)
                    .into_iter()
                    .map(move |split| (simdgroups, split))
            }) {
                let specialization = specialization_on(
                    device,
                    &[("H", H as u64), ("E", E as u64), ("K", K as u64)],
                    &route_mapping(&device, simdgroups, split),
                );
                let kernel = routed_route::native_for_device_with(
                    &device,
                    routed_route::Elements {
                        NW: seismic::Element::f32(),
                        RW: seismic::Element::f32(),
                        A: element,
                    },
                    &specialization,
                )
                .unwrap();
                let mut native_routes =
                    i32_tensor(&device, &[rows as u64, K as u64], &vec![-7; rows * K]);
                let mut native_scores =
                    f32_tensor(&device, &[rows as u64, K as u64], &vec![-7.0; rows * K]);
                let outcome = kernel
                    .call(routed_route::Args {
                        residual: &f32_tensor(&device, &[rows as u64, H as u64], &case.residual),
                        norm: &f32_tensor(&device, &[H as u64], &case.norm),
                        router: &f32_tensor(&device, &[E as u64, H as u64], &case.router),
                        shared_router: &f32_tensor(&device, &[H as u64], &case.shared_router),
                        routes: &mut native_routes,
                        scores: &mut native_scores,
                        eps: 1e-6,
                        normalize,
                    })
                    .unwrap();
                let label =
                    format!("{activation:?} rows {rows} SIMDGROUPS {simdgroups} SPLIT {split}");
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
        (self.rows * K + self.experts * (self.tile - 1)).div_ceil(self.tile)
    }

    /// (counts, order, inverse, blocks) from the portable body.
    fn portable(&self, module: &CheckedModule) -> [Vec<i32>; 4] {
        let (rows, blocks, tile) = (self.rows, self.blocks(), self.tile);
        let outcome = interpret(
            module,
            "routed_group",
            &[],
            vec![
                ints(&[rows, K], &self.routes),
                ints(&[self.experts], &vec![0; self.experts]),
                ints(&[blocks, tile], &vec![0; blocks * tile]),
                ints(&[rows, K], &vec![0; rows * K]),
                ints(&[blocks], &vec![0; blocks]),
            ],
        );
        [1, 2, 3, 4].map(|ordinal| {
            input(&outcome, ordinal)
                .into_iter()
                .map(|v| v as i32)
                .collect()
        })
    }

    /// Every choice maps to a tile row of its own expert holding its row,
    /// experts own contiguous ascending block ranges, and every other slot
    /// is padding.
    fn check_round_trip(&self, [counts, order, inverse, blocks]: &[Vec<i32>; 4]) {
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
        let used = blocks.iter().filter(|&&b| b >= 0).count();
        assert!(
            blocks[used..].iter().all(|&b| b == -1),
            "unused blocks trail"
        );
        assert!(
            blocks[..used].windows(2).all(|w| w[0] <= w[1]),
            "experts ascend"
        );
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
    let module = module();
    for (rows, experts, tile) in [(1, 8, 4), (13, 8, 4), (37, 16, 8), (64, 32, 16)] {
        let case = GroupCase::new(rows, experts, tile);
        let tables = case.portable(&module);
        case.check_round_trip(&tables);
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
    for device in devices() {
        native_group_matches_portable_tables_on(&device);
    }
}

fn native_group_matches_portable_tables_on(device: &seismic::Device) {
    let module = module();
    for (rows, experts, tile) in [
        (1, 8, 4),
        (13, 8, 4),
        (37, 16, 8),
        (64, 32, 16),
        (200, 32, 32),
        (512, 256, 32),
    ] {
        let case = GroupCase::new(rows, experts, tile);
        let expected = case.portable(&module);
        let blocks = case.blocks();
        for parts in [1, 2, 4] {
            let kernel = routed_group::native_for_device(
                device,
                &group_specialization(device, experts as u64, K as u64, parts),
            )
            .unwrap();
            let fill = |shape: &[u64]| {
                let count = shape.iter().product::<u64>() as usize;
                i32_tensor(&device, shape, &vec![-9; count])
            };
            let mut counts = fill(&[experts as u64]);
            let mut order = fill(&[blocks as u64, tile as u64]);
            let mut inverse = fill(&[rows as u64, K as u64]);
            let mut table = fill(&[blocks as u64]);
            kernel
                .call(routed_group::Args {
                    routes: &i32_tensor(&device, &[rows as u64, K as u64], &case.routes),
                    counts: &mut counts,
                    order: &mut order,
                    inverse: &mut inverse,
                    blocks: &mut table,
                })
                .unwrap();
            let actual = [
                read_i32(&counts),
                read_i32(&order),
                read_i32(&inverse),
                read_i32(&table),
            ];
            for (name, (actual, expected)) in ["counts", "order", "inverse", "blocks"]
                .iter()
                .zip(actual.iter().zip(&expected))
            {
                assert_eq!(
                    actual, expected,
                    "{name}: rows {rows} experts {experts} tile {tile} parts {parts}"
                );
            }
        }
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
}

impl Block {
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
            &[("NW", DType::F32), ("RW", DType::F32), ("A", DType::F32)],
            vec![
                floats(&[m, h], &self.residual),
                floats(&[h], &self.norm),
                floats(&[e, h], &self.router),
                floats(&[h], &self.shared_router),
                ints(&[m, k], &vec![0; m * k]),
                floats(&[m, k], &vec![0.0; m * k]),
                Input::F32(1e-6),
                Input::I32(1),
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
            ],
        );
        let output = interpret(
            module,
            "routed_output",
            &[("A", DType::F32), ("EDW", DType::F32), ("SDW", DType::F32)],
            vec![
                floats(&[m, h], &self.residual),
                floats(&[m, k, f], &Self::values(&result(&expanded, 0))),
                floats(&[m, s], &Self::values(&result(&expanded, 1))),
                ints(&[m, k], &routes),
                floats(&[m, k], &scores),
                floats(&[m], &coefficient),
                floats(&[e, h, f], &self.expert_down),
                floats(&[h, s], &self.shared_down),
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
        let blocks = (m * k + e * (tile - 1)).div_ceil(tile);
        let grouped = interpret(
            module,
            "routed_group",
            &[],
            vec![
                ints(&[m, k], &routes),
                ints(&[e], &vec![0; e]),
                ints(&[blocks, tile], &vec![0; blocks * tile]),
                ints(&[m, k], &vec![0; m * k]),
                ints(&[blocks], &vec![0; blocks]),
            ],
        );
        let tables = |ordinal| {
            input(&grouped, ordinal)
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>()
        };
        let (order, inverse, block_experts) = (tables(2), tables(3), tables(4));
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
                floats(&[m, h], &normalized),
                ints(&[blocks, tile], &order),
                ints(&[blocks], &block_experts),
                floats(&[e, f, h], &self.expert_gate),
                floats(&[e, f, h], &self.expert_up),
                floats(&[e, h, f], &self.expert_down),
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
/// tiles and the unpermute are a permutation round trip.
#[test]
fn portable_grouped_prefill_equals_decode_form() {
    let module = module();
    let block = Block::new(6);
    let decode = block.decode(&module);
    for tile in [1, 2, 4] {
        let prefill = block.prefill(&module, tile);
        assert_eq!(
            prefill.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            decode.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "tile {tile}"
        );
    }
    // The block is not the identity: some output moved off the residual.
    assert!(decode
        .iter()
        .zip(&block.residual)
        .any(|(out, residual)| (out - residual).abs() > 1e-3));
}

// ---------------------------------------------------------------------------
// Host reference of the routed formulas (the portable bodies' arithmetic,
// with every published activation rounded to A by `round`).

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff_0000)
}

/// Weights and inputs of one routed block at real packed representations.
struct Routed {
    hidden: usize,
    experts: usize,
    choices: usize,
    features: usize,
    shared: usize,
    expert_gate: Vec<f32>,
    expert_up: Vec<f32>,
    expert_down: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_down: Vec<f32>,
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
    /// (expert product [M, K, F], shared product [M, S]) with `round` applied
    /// to each published product.
    fn expand(&self, routing: &Routing, round: fn(f32) -> f32) -> (Vec<f32>, Vec<f32>) {
        let (h, k, f, s) = (self.hidden, self.choices, self.features, self.shared);
        let mut expert = vec![0.0; routing.rows * k * f];
        let mut shared = vec![0.0; routing.rows * s];
        for m in 0..routing.rows {
            let x = &routing.normalized[m * h..(m + 1) * h];
            for choice in 0..k {
                let e = routing.routes[m * k + choice] as usize;
                for feature in 0..f {
                    let row = (e * f + feature) * h;
                    let gate = dot(x, &self.expert_gate[row..row + h]);
                    let up = dot(x, &self.expert_up[row..row + h]);
                    expert[(m * k + choice) * f + feature] = round(silu(gate) * up);
                }
            }
            for feature in 0..s {
                let gate = dot(x, &self.shared_gate[feature * h..(feature + 1) * h]);
                let up = dot(x, &self.shared_up[feature * h..(feature + 1) * h]);
                shared[m * s + feature] = round(silu(gate) * up);
            }
        }
        (expert, shared)
    }

    fn output(
        &self,
        routing: &Routing,
        expert: &[f32],
        shared: &[f32],
        round: fn(f32) -> f32,
    ) -> Vec<f32> {
        let (h, k, f, s) = (self.hidden, self.choices, self.features, self.shared);
        let mut value = vec![0.0; routing.rows * h];
        for m in 0..routing.rows {
            for column in 0..h {
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
                value[m * h + column] =
                    routing.residual[m * h + column] + selected + down * routing.coefficient[m];
            }
        }
        value
    }
}

/// Floating results agree within `relative * |reference| + absolute`, with
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
    let mut worst = (0usize, 0.0f32);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{name}[{index}] is {actual}");
        let excess =
            (actual - expected).abs() - (relative * expected.abs() + absolute_fraction * scale);
        if excess > worst.1 {
            worst = (index, excess);
        }
    }
    assert!(
        worst.1 <= 0.0,
        "{name}[{}]: native {}, reference {} (scale {scale})",
        worst.0,
        actual[worst.0],
        expected[worst.0]
    );
}

/// Arithmetic variants use the tuner's relative output-norm defect guard.
fn assert_arithmetic(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    int8: bool,
    relative: f32,
    absolute_fraction: f32,
) {
    if !int8 {
        return assert_near(name, actual, expected, relative, absolute_fraction);
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
    let module = module();
    let block = Block::new(5);
    let (normalized, coefficient, routes, scores) = block.route(&module);
    let routed = Routed {
        hidden: block.hidden,
        experts: block.experts,
        choices: block.choices,
        features: block.features,
        shared: block.shared,
        expert_gate: block.expert_gate.clone(),
        expert_up: block.expert_up.clone(),
        expert_down: block.expert_down.clone(),
        shared_gate: block.shared_gate.clone(),
        shared_up: block.shared_up.clone(),
        shared_down: block.shared_down.clone(),
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
    assert_near("decode", &block.decode(&module), &reference, 1e-6, 1e-6);
}

// ---------------------------------------------------------------------------
// Packed weights in the rows16 layout.

/// The resident row layout of the device's backend: `mma16` on CUDA,
/// `rows16` on Metal.
fn resident_layout(device: &seismic::Device) -> registry::Layout {
    match device.backend() {
        seismic::BackendName::Cuda => registry::Layout::Mma16,
        _ => registry::Layout::Rows16,
    }
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
        seismic::Tensor::from_host(device, element, shape, &bytes).unwrap(),
        values,
    )
}

/// The planes of one packed weight: resident element, layout bytes and
/// decoded values.
type Planes = (seismic::Element, Vec<u8>, Vec<f32>);

/// The resident element, layout bytes and decoded values of a
/// `representation` weight of `shape` with pseudo-random planes. Decoding
/// through the reference recipe is slow in a test build, so every device and
/// test of one resident layout shares one decoding.
fn packed_planes(
    device: &seismic::Device,
    representation: &str,
    shape: &[u64],
    seed: u32,
) -> Planes {
    type Key = (registry::Layout, String, Vec<u64>, u32);
    static DECODED: std::sync::Mutex<Vec<(Key, Planes)>> = std::sync::Mutex::new(Vec::new());
    let key = (
        resident_layout(device),
        representation.to_owned(),
        shape.to_vec(),
        seed,
    );
    let mut decoded = DECODED.lock().unwrap();
    if let Some((_, planes)) = decoded.iter().find(|(known, _)| *known == key) {
        return planes.clone();
    }
    let planes = decode_packed_planes(key.0, representation, shape, seed);
    decoded.push((key, planes.clone()));
    planes
}

fn decode_packed_planes(
    layout: registry::Layout,
    representation: &str,
    shape: &[u64],
    seed: u32,
) -> Planes {
    use registry::{Layout, RepresentationKind};
    let packet_id = registry::storage(representation, Layout::Packet).unwrap();
    let RepresentationKind::Packed(packet) = &registry::representation_info(packet_id).kind else {
        panic!("{representation} is packed")
    };
    let rows_id = registry::storage(representation, layout).unwrap();
    let RepresentationKind::PackedRows(rows) = &registry::representation_info(rows_id).kind else {
        panic!("{representation} resident storage is a row layout")
    };
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
            } else {
                for (offset, byte) in packet_bytes[start..end].iter_mut().enumerate() {
                    let unit = noise[(index * size + start + offset) % noise.len()];
                    *byte = ((unit + 1.0) * 127.5) as u8;
                }
            }
        }
    }
    let placed = rows.place(shape, &bytes);
    let shape_usize = shape
        .iter()
        .map(|extent| *extent as usize)
        .collect::<Vec<_>>();
    let values = TensorData::encoded(rows_id, shape_usize, placed.clone())
        .unwrap()
        .values()
        .unwrap()
        .into_iter()
        .map(|value| value as f32)
        .collect();
    let element = seismic::Element::named(registry::representation_info(rows_id).name).unwrap();
    (element, placed, values)
}

fn bf16_tensor(device: &seismic::Device, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(device, seismic::Element::bf16(), shape, &bytes).unwrap()
}

/// The 35B-A3B storage mix at reduced width: expert gate/up q4k, down q5k,
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
    const EXPERTS: usize = 16;
    const CHOICES: usize = 4;
    const FEATURES: usize = 256;
    const SHARED: usize = 256;

    fn new(device: &seismic::Device) -> Self {
        let (h, e, f, s) = (
            Self::HIDDEN as u64,
            Self::EXPERTS as u64,
            Self::FEATURES as u64,
            Self::SHARED as u64,
        );
        let (expert_gate, gate) = packed_weight(device, "q4k", &[e, f, h], 41);
        let (expert_up, up) = packed_weight(device, "q4k", &[e, f, h], 42);
        let (expert_down, down) = packed_weight(device, "q5k", &[e, h, f], 43);
        let (shared_gate, sgate) = packed_weight(device, "q8g32s", &[s, h], 44);
        let (shared_up, sup) = packed_weight(device, "q8g32s", &[s, h], 45);
        let (shared_down, sdown) = packed_weight(device, "q8g32s", &[h, s], 46);
        Self {
            routed: Routed {
                hidden: Self::HIDDEN,
                experts: Self::EXPERTS,
                choices: Self::CHOICES,
                features: Self::FEATURES,
                shared: Self::SHARED,
                expert_gate: gate,
                expert_up: up,
                expert_down: down,
                shared_gate: sgate,
                shared_up: sup,
                shared_down: sdown,
            },
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            shared_up,
            shared_down,
        }
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
    let id = registry::storage(representation, resident_layout(device)).unwrap();
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
            vec![("TILE_M", 64), ("TILE_N", 64), ("SUB_M", 64), ("SUB_N", 64)],
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

#[test]
fn native_decode_expand_and_output_match_reference() {
    for device in devices() {
        native_decode_expand_and_output_match_reference_on(&device, false);
        if is_cpu(&device) {
            native_decode_expand_and_output_match_reference_on(&device, true);
        }
    }
}

fn native_decode_expand_and_output_match_reference_on(device: &seismic::Device, int8: bool) {
    let block = PackedBlock::new(&device);
    let (h, k, f, s) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::CHOICES as u64,
        PackedBlock::FEATURES as u64,
        PackedBlock::SHARED as u64,
    );
    for rows in [1usize, 3, 8] {
        let routing = PackedBlock::routing(rows, 50 + rows as u32);
        let m = rows as u64;
        let (expert, shared) = block.routed.expand(&routing, bf16_round);
        let reference = block.routed.output(&routing, &expert, &shared, bf16_round);
        let normalized = bf16_tensor(&device, &[m, h], &routing.normalized);
        let routes = i32_tensor(&device, &[m, k], &routing.routes);
        let scores = f32_tensor(&device, &[m, k], &routing.scores);
        let coefficient = f32_tensor(&device, &[m], &routing.coefficient);
        let residual = f32_tensor(&device, &[m, h], &routing.residual);
        let expert_product = bf16_tensor(&device, &[m, k, f], &expert);
        let shared_product = bf16_tensor(&device, &[m, s], &shared);
        for mapping in decode_mappings(&device) {
            let label = format!("rows {rows} mapping {mapping:?} INT8 {int8}");
            let mut specialization =
                specialization_on(device, &[("H", h), ("K", k), ("F", f), ("S", s)], &mapping);
            if is_cpu(device) {
                specialization = specialization.with_param("INT8", u64::from(int8));
            }
            let expanded = routed_expand::native_for_device_with(
                &device,
                routed_expand::Elements {
                    A: seismic::Element::bf16(),
                    EGW: element(&device, "q4k"),
                    EUW: element(&device, "q4k"),
                    SGW: element(&device, "q8g32s"),
                    SUW: element(&device, "q8g32s"),
                },
                &specialization,
            )
            .unwrap()
            .call(routed_expand::Args {
                normalized: &normalized,
                routes: &routes,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
            })
            .unwrap();
            assert_arithmetic(
                &format!("{label} expert product"),
                &read_bf16(&expanded.r0),
                &expert,
                int8,
                2e-2,
                2e-3,
            );
            assert_arithmetic(
                &format!("{label} shared product"),
                &read_bf16(&expanded.r1),
                &shared,
                int8,
                2e-2,
                2e-3,
            );
            let output = routed_output::native_for_device_with(
                &device,
                routed_output::Elements {
                    A: seismic::Element::bf16(),
                    EDW: element(&device, "q5k"),
                    SDW: element(&device, "q8g32s"),
                },
                &specialization,
            )
            .unwrap()
            .call(routed_output::Args {
                residual: &residual,
                expert_product: &expert_product,
                shared_product: &shared_product,
                routes: &routes,
                scores: &scores,
                coefficient: &coefficient,
                expert_down: &block.expert_down,
                shared_down: &block.shared_down,
            })
            .unwrap()
            .value;
            assert_arithmetic(
                &format!("{label} output"),
                &read_f32(&output),
                &reference,
                int8,
                1e-2,
                2e-3,
            );
        }
    }
}

#[test]
fn native_grouped_prefill_matches_reference() {
    for device in devices() {
        native_grouped_prefill_matches_reference_on(&device, false);
        if is_cpu(&device) {
            native_grouped_prefill_matches_reference_on(&device, true);
        }
    }
}

fn native_grouped_prefill_matches_reference_on(device: &seismic::Device, int8: bool) {
    let block = PackedBlock::new(&device);
    let (h, e, k, f, s) = (
        PackedBlock::HIDDEN as u64,
        PackedBlock::EXPERTS as u64,
        PackedBlock::CHOICES as u64,
        PackedBlock::FEATURES as u64,
        PackedBlock::SHARED as u64,
    );
    let tile = 32u64;
    for rows in [16usize, 40, 64] {
        let routing = PackedBlock::routing(rows, 90 + rows as u32);
        let m = rows as u64;
        let (expert, shared) = block.routed.expand(&routing, bf16_round);
        let reference = block.routed.output(&routing, &expert, &shared, bf16_round);
        let blocks = (m * k + e * (tile - 1)).div_ceil(tile);
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
        routed_group::native_for_device(device, &group_specialization(device, e, k, 4))
            .unwrap()
            .call(routed_group::Args {
                routes: &routes,
                counts: &mut counts,
                order: &mut order,
                inverse: &mut inverse,
                blocks: &mut table,
            })
            .unwrap();
        let normalized = bf16_tensor(&device, &[m, h], &routing.normalized);
        for mapping in grouped_mappings(&device) {
            let label = format!("rows {rows} mapping {mapping:?} INT8 {int8}");
            let specialization = |statics: &[(&str, u64)]| {
                let specialization = specialization_on(device, statics, &mapping);
                if is_cpu(device) {
                    specialization.with_param("INT8", u64::from(int8))
                } else {
                    specialization
                }
            };
            let experts = routed_experts::native_for_device_with(
                &device,
                routed_experts::Elements {
                    A: seismic::Element::bf16(),
                    EGW: element(&device, "q4k"),
                    EUW: element(&device, "q4k"),
                    EDW: element(&device, "q5k"),
                },
                &specialization(&[("H", h), ("F", f)]),
            )
            .unwrap()
            .call(routed_experts::Args {
                normalized: &normalized,
                order: &order,
                blocks: &table,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                expert_down: &block.expert_down,
            })
            .unwrap()
            .value;
            let combined = routed_combine::native_for_device_with(
                &device,
                routed_combine::Elements {
                    A: seismic::Element::bf16(),
                    SGW: element(&device, "q8g32s"),
                    SUW: element(&device, "q8g32s"),
                    SDW: element(&device, "q8g32s"),
                },
                &specialization(&[("H", h), ("K", k), ("S", s)]),
            )
            .unwrap()
            .call(routed_combine::Args {
                residual: &f32_tensor(&device, &[m, h], &routing.residual),
                expert_output: &experts,
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
            assert_arithmetic(
                &format!("{label} output"),
                &read_f32(&combined),
                &reference,
                int8,
                2e-2,
                4e-3,
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
    const EXPERTS: usize = 256;
    const CHOICES: usize = 8;
    const FEATURES: usize = 512;
    const SHARED: usize = 512;
    /// The graph's grouped tile rows (`TILE_ROWS`).
    const TILE: usize = 32;
    const USED: usize = 12;
    /// Distinct 16-row tiles per weight pool.
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
                experts: Self::USED,
                choices: Self::CHOICES,
                features: f,
                shared: s,
                expert_gate: gate,
                expert_up: up,
                expert_down: down,
                shared_gate: sgate,
                shared_up: sup,
                shared_down: sdown,
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

    /// Native routing of `residual`, checked against the host reference;
    /// every SIMDGROUPS mapping must give the same bits (at the backend's
    /// first split).
    fn route(&self, device: &seismic::Device, residual: &[f32]) -> NativeRouting {
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
        for simdgroups in [8, 4, 2] {
            let mut routes = i32_tensor(device, &[m, k], &vec![-7; rows * Self::CHOICES]);
            let mut scores = f32_tensor(device, &[m, k], &vec![-7.0; rows * Self::CHOICES]);
            let outcome = routed_route::native_for_device_with(
                device,
                routed_route::Elements {
                    NW: bf16,
                    RW: bf16,
                    A: bf16,
                },
                &specialization_on(
                    device,
                    &[("H", h), ("E", e), ("K", k)],
                    &route_mapping(device, simdgroups, route_splits(device)[0]),
                ),
            )
            .unwrap()
            .call(routed_route::Args {
                residual: &residual_tensor,
                norm: &norm,
                router: &router,
                shared_router: &shared_router,
                routes: &mut routes,
                scores: &mut scores,
                eps: 1e-6,
                normalize: 1,
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
            let squares = values
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>();
            let inverse = 1.0 / (squares / h as f64 + 1e-6).sqrt();
            let normalized = &native.normalized_values[row * h..(row + 1) * h];
            for (index, (value, actual)) in values.iter().zip(normalized).enumerate() {
                let expected = f64::from(*value) * inverse * f64::from(self.norm[index]);
                assert!(
                    (f64::from(*actual) - expected).abs() <= 8e-3 * expected.abs(),
                    "{label} normalized[{index}]: native {actual}, reference {expected}"
                );
            }
            // Logits from the native normalized row, in f64.
            let total = normalized.iter().map(|v| f64::from(*v)).sum::<f64>();
            let logits = (0..e)
                .map(|expert| match self.slot[expert] {
                    Some(_) => normalized
                        .iter()
                        .zip(&self.router[expert * h..(expert + 1) * h])
                        .map(|(x, w)| f64::from(*x) * f64::from(*w))
                        .sum::<f64>(),
                    None => f64::from(bf16_round(UNUSED_ROUTER)) * total,
                })
                .collect::<Vec<_>>();
            let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exponentials = logits
                .iter()
                .map(|logit| (logit - maximum).exp())
                .collect::<Vec<_>>();
            let sum = exponentials.iter().sum::<f64>();
            let probability = exponentials
                .iter()
                .map(|value| value / sum)
                .collect::<Vec<_>>();
            let mut ranked = (0..e).collect::<Vec<_>>();
            ranked.sort_by(|a, b| probability[*b].total_cmp(&probability[*a]).then(b.cmp(a)));
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
            // A different set is only admissible at a near tie on the boundary.
            let (last, next) = (probability[ranked[k - 1]], probability[ranked[k]]);
            assert!(
                chosen == expected || last - next <= 1e-4 * last,
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

/// Rows of a resident row layout come in tiles of 16 that each occupy
/// 16 row strides of bytes (`mma16` permutes codes only within a tile).
const LAYOUT_TILE_ROWS: usize = 16;

/// A few distinct 16-row tiles of `representation` over `columns`, placed
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
            seismic::Tensor::from_host(device, self.element, shape, &bytes).unwrap(),
            values,
        )
    }
}

#[test]
fn native_35b_geometry_chain_matches_reference() {
    for device in devices() {
        native_35b_geometry_chain_matches_reference_on(&device);
    }
}

fn native_35b_geometry_chain_matches_reference_on(device: &seismic::Device) {
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
        let residual_tensor = f32_tensor(&device, &[rows as u64, h], &residual);
        for mapping in decode_mappings(&device) {
            let label = format!("35B decode rows {rows} mapping {mapping:?}");
            let mut specialization =
                specialization_on(device, &[("H", h), ("K", k), ("F", f), ("S", s)], &mapping);
            if is_cpu(device) {
                specialization = specialization.with_param("INT8", 0);
            }
            let expanded = routed_expand::native_for_device_with(
                &device,
                routed_expand::Elements {
                    A: bf16,
                    EGW: element(&device, "q4k"),
                    EUW: element(&device, "q4k"),
                    SGW: element(&device, "q8g32s"),
                    SUW: element(&device, "q8g32s"),
                },
                &specialization,
            )
            .unwrap()
            .call(routed_expand::Args {
                normalized: &routing.normalized,
                routes: &routing.routes,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                shared_gate: &block.shared_gate,
                shared_up: &block.shared_up,
            })
            .unwrap();
            assert_near(
                &format!("{label} expert product"),
                &read_bf16(&expanded.r0),
                &expert,
                2e-2,
                2e-3,
            );
            assert_near(
                &format!("{label} shared product"),
                &read_bf16(&expanded.r1),
                &shared,
                2e-2,
                2e-3,
            );
            let output = routed_output::native_for_device_with(
                &device,
                routed_output::Elements {
                    A: bf16,
                    EDW: element(&device, "q5k"),
                    SDW: element(&device, "q8g32s"),
                },
                &specialization,
            )
            .unwrap()
            .call(routed_output::Args {
                residual: &residual_tensor,
                expert_product: &expanded.r0,
                shared_product: &expanded.r1,
                routes: &routing.routes,
                scores: &routing.scores,
                coefficient: &routing.coefficient,
                expert_down: &block.expert_down,
                shared_down: &block.shared_down,
            })
            .unwrap()
            .value;
            assert_near(
                &format!("{label} output"),
                &read_f32(&output),
                &reference,
                2e-2,
                4e-3,
            );
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
        let blocks = (m * k + e * (t - 1)).div_ceil(t);
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
        routed_group::native_for_device(device, &group_specialization(device, e, k, 4))
            .unwrap()
            .call(routed_group::Args {
                routes: &routing.routes,
                counts: &mut counts,
                order: &mut order,
                inverse: &mut inverse,
                blocks: &mut table,
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
        for mapping in grouped_mappings(&device) {
            let label = format!("35B grouped rows {rows} mapping {mapping:?}");
            let specialization = |statics: &[(&str, u64)]| {
                let specialization = specialization_on(device, statics, &mapping);
                if is_cpu(device) {
                    specialization.with_param("INT8", 0)
                } else {
                    specialization
                }
            };
            let experts = routed_experts::native_for_device_with(
                &device,
                routed_experts::Elements {
                    A: bf16,
                    EGW: element(&device, "q4k"),
                    EUW: element(&device, "q4k"),
                    EDW: element(&device, "q5k"),
                },
                &specialization(&[("H", h), ("F", f)]),
            )
            .unwrap()
            .call(routed_experts::Args {
                normalized: &routing.normalized,
                order: &order,
                blocks: &table,
                expert_gate: &block.expert_gate,
                expert_up: &block.expert_up,
                expert_down: &block.expert_down,
            })
            .unwrap()
            .value;
            let combined = routed_combine::native_for_device_with(
                &device,
                routed_combine::Elements {
                    A: bf16,
                    SGW: element(&device, "q8g32s"),
                    SUW: element(&device, "q8g32s"),
                    SDW: element(&device, "q8g32s"),
                },
                &specialization(&[("H", h), ("K", k), ("S", s)]),
            )
            .unwrap()
            .call(routed_combine::Args {
                residual: &residual_tensor,
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
            assert_near(&format!("{label} output"), &sampled, &reference, 2e-2, 4e-3);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel timings at the Qwen3.5-35B-A3B routed shape (opt-in):
//     cargo test --release -p magnitude-model-kernels --test routed \
//         routed_kernel_timings -- --ignored --nocapture
// Every entry is tuned over its declared domain at its row points; each
// point rotates four layers of distinct zero weights and distinct routing so
// the decode reads stream from memory. Prints each configuration's median
// device time per point.

const LAYERS: usize = 4;

fn print_timings(result: &seismic::TuningResult) {
    println!("== {} ({})", result.entry, result.backend);
    println!("   overall {:?}", result.overall.params);
    for record in &result.configurations {
        match &record.outcome {
            seismic::Outcome::Measured { points, .. } => {
                let cells = points
                    .iter()
                    .map(|point| format!("{} {:.1}us", point.point, point.median_seconds * 1e6))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("   {:?}: {cells}", record.configuration.params);
            }
            seismic::Outcome::Excluded(exclusion) => {
                println!(
                    "   {:?}: excluded {exclusion:?}",
                    record.configuration.params
                );
            }
        }
    }
}

/// Distinct experts per row, a different spread per layer.
fn spread_routes(rows: usize, experts: usize, choices: usize, layer: usize) -> Vec<i32> {
    (0..rows)
        .flat_map(|row| {
            (0..choices)
                .map(move |choice| ((37 * row + 101 * choice + 13 * layer) % experts) as i32)
        })
        .collect()
}

#[test]
#[ignore]
fn routed_kernel_timings() {
    let Some(device) = native_device() else {
        return;
    };
    let (h, e, k, f, s) = (2048u64, 256u64, 8u64, 512u64, 512u64);
    let bf16 = seismic::Element::bf16();
    let zeros = |element: seismic::Element, shape: &[u64]| {
        seismic::Tensor::zeros(&device, element, shape).unwrap()
    };
    let activation = |element: seismic::Element, shape: &[u64], seed: u32| {
        let count = shape.iter().product::<u64>() as usize;
        let values = pattern(count, seed, 1.0);
        match element.dtype() {
            Some(DType::F32) => f32_tensor(&device, shape, &values),
            _ => bf16_tensor(&device, shape, &values),
        }
    };
    let layers = (0..LAYERS)
        .map(|_| {
            (
                zeros(element(&device, "q4k"), &[e, f, h]),
                zeros(element(&device, "q4k"), &[e, f, h]),
                zeros(element(&device, "q5k"), &[e, h, f]),
                zeros(element(&device, "q8g32s"), &[s, h]),
                zeros(element(&device, "q8g32s"), &[s, h]),
                zeros(element(&device, "q8g32s"), &[h, s]),
                activation(bf16, &[e, h], 5),
            )
        })
        .collect::<Vec<_>>();
    // Every admissible configuration is measured, so the timings cover the
    // whole declared space.
    let measure = seismic::Strategy::Survey(seismic::SurveyPlan {
        samples: 7,
        min_sample_seconds: 0.002,
        domains: Default::default(),
    });
    let validation = seismic::Validation::Relative { error: 0.05 };
    let statics = |pairs: &[(&str, u64)]| {
        pairs.iter().fold(
            seismic::NativeSpecialization::new(),
            |spec, (name, value)| spec.with_static(*name, *value),
        )
    };

    // Route (decode and grouped rows): tables are fully overwritten, so the
    // initializer has nothing to restore.
    let route_rows = [1u64, 8, 64, 512];
    let route_inputs = route_rows
        .iter()
        .map(|&m| {
            (0..LAYERS)
                .map(|layer| {
                    (
                        activation(seismic::Element::f32(), &[m, h], 10 + layer as u32),
                        activation(bf16, &[h], 20),
                        activation(seismic::Element::f32(), &[h], 21),
                        zeros(seismic::Element::i32(), &[m, k]),
                        zeros(seismic::Element::f32(), &[m, k]),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut route_inputs = route_inputs;
    let points = route_rows
        .iter()
        .zip(route_inputs.iter_mut())
        .map(|(m, inputs)| seismic::TuningPoint {
            label: format!("m{m}"),
            weight: 1.0,
            class: None,
            rotation: inputs
                .iter_mut()
                .zip(&layers)
                .map(|((residual, norm, shared_router, routes, scores), layer)| {
                    routed_route::Args {
                        residual,
                        norm,
                        router: &layer.6,
                        shared_router,
                        routes,
                        scores,
                        eps: 1e-6,
                        normalize: 1,
                    }
                })
                .collect(),
            initialize: Some(Box::new(|| Ok(()))),
        })
        .collect();
    print_timings(
        &routed_route::native_tune_with(
            &device,
            routed_route::Elements {
                NW: bf16,
                RW: bf16,
                A: bf16,
            },
            &statics(&[("H", h), ("E", e), ("K", k)]),
            points,
            validation,
            measure.clone(),
        )
        .unwrap(),
    );

    // Decode expand and output.
    let decode_rows = [1u64, 2, 4, 8];
    let decode = decode_rows
        .iter()
        .map(|&m| {
            (0..LAYERS)
                .map(|layer| {
                    let routes = spread_routes(m as usize, e as usize, k as usize, layer);
                    (
                        activation(bf16, &[m, h], 30 + layer as u32),
                        i32_tensor(&device, &[m, k], &routes),
                        activation(bf16, &[m, k, f], 40 + layer as u32),
                        activation(bf16, &[m, s], 50 + layer as u32),
                        activation(seismic::Element::f32(), &[m, h], 60 + layer as u32),
                        activation(seismic::Element::f32(), &[m, k], 70 + layer as u32),
                        activation(seismic::Element::f32(), &[m], 80 + layer as u32),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let decode_statics = statics(&[("H", h), ("K", k), ("F", f), ("S", s)]);
    let points = decode_rows
        .iter()
        .zip(&decode)
        .map(|(m, inputs)| seismic::TuningPoint {
            label: format!("m{m}"),
            weight: 1.0,
            class: None,
            rotation: inputs
                .iter()
                .zip(&layers)
                .map(|(input, layer)| routed_expand::Args {
                    normalized: &input.0,
                    routes: &input.1,
                    expert_gate: &layer.0,
                    expert_up: &layer.1,
                    shared_gate: &layer.3,
                    shared_up: &layer.4,
                })
                .collect(),
            initialize: None,
        })
        .collect();
    print_timings(
        &routed_expand::native_tune_with(
            &device,
            routed_expand::Elements {
                A: bf16,
                EGW: element(&device, "q4k"),
                EUW: element(&device, "q4k"),
                SGW: element(&device, "q8g32s"),
                SUW: element(&device, "q8g32s"),
            },
            &decode_statics,
            points,
            validation,
            measure.clone(),
        )
        .unwrap(),
    );
    let points = decode_rows
        .iter()
        .zip(&decode)
        .map(|(m, inputs)| seismic::TuningPoint {
            label: format!("m{m}"),
            weight: 1.0,
            class: None,
            rotation: inputs
                .iter()
                .zip(&layers)
                .map(|(input, layer)| routed_output::Args {
                    residual: &input.4,
                    expert_product: &input.2,
                    shared_product: &input.3,
                    routes: &input.1,
                    scores: &input.5,
                    coefficient: &input.6,
                    expert_down: &layer.2,
                    shared_down: &layer.5,
                })
                .collect(),
            initialize: None,
        })
        .collect();
    print_timings(
        &routed_output::native_tune_with(
            &device,
            routed_output::Elements {
                A: bf16,
                EDW: element(&device, "q5k"),
                SDW: element(&device, "q8g32s"),
            },
            &decode_statics,
            points,
            validation,
            measure.clone(),
        )
        .unwrap(),
    );

    // Grouped rows: tables from the host mirror of the group contract.
    let tile = 32u64;
    let grouped_rows = [64u64, 128, 512];
    let grouped = grouped_rows
        .iter()
        .map(|&m| {
            (0..LAYERS)
                .map(|layer| {
                    let routes = spread_routes(m as usize, e as usize, k as usize, layer);
                    let blocks = (m * k + e * (tile - 1)).div_ceil(tile);
                    let (order, inverse, table) = host_group(
                        &routes,
                        m as usize,
                        e as usize,
                        k as usize,
                        tile as usize,
                        blocks as usize,
                    );
                    (
                        activation(bf16, &[m, h], 90 + layer as u32),
                        i32_tensor(&device, &[blocks, tile], &order),
                        i32_tensor(&device, &[blocks], &table),
                        i32_tensor(&device, &[m, k], &inverse),
                        activation(bf16, &[blocks, tile, h], 100 + layer as u32),
                        activation(seismic::Element::f32(), &[m, h], 110 + layer as u32),
                        activation(seismic::Element::f32(), &[m, k], 120 + layer as u32),
                        activation(seismic::Element::f32(), &[m], 130 + layer as u32),
                        i32_tensor(&device, &[m, k], &routes),
                        blocks,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let points = grouped_rows
        .iter()
        .zip(&grouped)
        .map(|(m, inputs)| seismic::TuningPoint {
            label: format!("m{m}"),
            weight: 1.0,
            class: None,
            rotation: inputs
                .iter()
                .zip(&layers)
                .map(|(input, layer)| routed_experts::Args {
                    normalized: &input.0,
                    order: &input.1,
                    blocks: &input.2,
                    expert_gate: &layer.0,
                    expert_up: &layer.1,
                    expert_down: &layer.2,
                })
                .collect(),
            initialize: None,
        })
        .collect();
    print_timings(
        &routed_experts::native_tune_with(
            &device,
            routed_experts::Elements {
                A: bf16,
                EGW: element(&device, "q4k"),
                EUW: element(&device, "q4k"),
                EDW: element(&device, "q5k"),
            },
            &statics(&[("H", h), ("F", f)]),
            points,
            validation,
            measure.clone(),
        )
        .unwrap(),
    );
    let points = grouped_rows
        .iter()
        .zip(&grouped)
        .map(|(m, inputs)| seismic::TuningPoint {
            label: format!("m{m}"),
            weight: 1.0,
            class: None,
            rotation: inputs
                .iter()
                .zip(&layers)
                .map(|(input, layer)| routed_combine::Args {
                    residual: &input.5,
                    expert_output: &input.4,
                    inverse: &input.3,
                    scores: &input.6,
                    normalized: &input.0,
                    coefficient: &input.7,
                    shared_gate: &layer.3,
                    shared_up: &layer.4,
                    shared_down: &layer.5,
                })
                .collect(),
            initialize: None,
        })
        .collect();
    print_timings(
        &routed_combine::native_tune_with(
            &device,
            routed_combine::Elements {
                A: bf16,
                SGW: element(&device, "q8g32s"),
                SUW: element(&device, "q8g32s"),
                SDW: element(&device, "q8g32s"),
            },
            &statics(&[("H", h), ("K", k), ("S", s)]),
            points,
            validation,
            measure.clone(),
        )
        .unwrap(),
    );

    // Group: its tables are fully overwritten.
    let mut group_tables = grouped_rows
        .iter()
        .zip(&grouped)
        .map(|(&m, inputs)| {
            inputs
                .iter()
                .map(|input| {
                    let blocks = input.9;
                    (
                        zeros(seismic::Element::i32(), &[e]),
                        zeros(seismic::Element::i32(), &[blocks, tile]),
                        zeros(seismic::Element::i32(), &[m, k]),
                        zeros(seismic::Element::i32(), &[blocks]),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let points = grouped_rows
        .iter()
        .zip(&grouped)
        .zip(group_tables.iter_mut())
        .map(|((m, inputs), tables)| seismic::TuningPoint {
            label: format!("m{m}"),
            weight: 1.0,
            class: None,
            rotation: inputs
                .iter()
                .zip(tables.iter_mut())
                .map(
                    |(input, (counts, order, inverse, blocks))| routed_group::Args {
                        routes: &input.8,
                        counts,
                        order,
                        inverse,
                        blocks,
                    },
                )
                .collect(),
            initialize: Some(Box::new(|| Ok(()))),
        })
        .collect();
    print_timings(
        &routed_group::native_tune(
            &device,
            &statics(&[("E", e), ("K", k)]),
            points,
            seismic::Validation::BitExact,
            measure,
        )
        .unwrap(),
    );
}

/// The host mirror of `routed_group`: (order [B * T], inverse [M * K],
/// blocks [B]).
fn host_group(
    routes: &[i32],
    rows: usize,
    experts: usize,
    choices: usize,
    tile: usize,
    blocks: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut order = vec![-1; blocks * tile];
    let mut inverse = vec![0; rows * choices];
    let mut table = vec![-1; blocks];
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
