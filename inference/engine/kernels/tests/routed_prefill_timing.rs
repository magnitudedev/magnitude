//! Prefill-layer timings of the general routed entries (opt-in):
//!     ROUTED_ROWS=512 cargo test --release -p magnitude-kernels \
//!         --test routed_prefill_timing -- --ignored --nocapture
//! The Gemma 4 26B-A4B routed sublayer (H 2816, E 128, K 8, F 704; or
//! `ROUTED_SHAPE=35b`, Qwen3.6-35B-A3B's H 2048, E 256, K 8, F 512) with
//! experts in `ROUTED_EXPERTS` (default q4g32s) and a bf16 router, at prefill
//! rows: the selection, the grouping, the
//! grouped expert projections and the scatter over every declared
//! configuration, rotating four layers of distinct zero weights so every
//! weight read streams from memory. The routes are skewed like a trained
//! router's (a few hot experts, a long tail), so experts span whole and
//! partial tiles.

use magnitude_kernels::{routed_experts, routed_group, routed_scatter, routed_select};
use seismic_lang::registry;

const LAYERS: usize = 4;
const TILE: u64 = 32;

/// The routed geometry `ROUTED_SHAPE` names: (H, E, K, F) of the Gemma 4
/// 26B-A4B (default) or Qwen3.6-35B-A3B blocks.
fn shape() -> (u64, u64, u64, u64) {
    match std::env::var("ROUTED_SHAPE").as_deref() {
        Err(_) | Ok("26b") => (2816, 128, 8, 704),
        Ok("35b") => (2048, 256, 8, 512),
        Ok(other) => panic!("ROUTED_SHAPE={other}: expected 26b or 35b"),
    }
}

fn native_device() -> seismic::Device {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    [seismic::BackendName::Cuda, seismic::BackendName::Metal]
        .into_iter()
        .find_map(|backend| catalog.open_backend(backend).ok())
        .expect("a CUDA or Metal device")
}

fn expert_element(device: &seismic::Device) -> seismic::Element {
    let layout = match device.backend() {
        seismic::BackendName::Cuda => registry::Layout::Mma16,
        _ => registry::Layout::Rows32,
    };
    let representation = std::env::var("ROUTED_EXPERTS").unwrap_or_else(|_| "q4g32s".into());
    let id = registry::storage(&representation, layout).unwrap();
    seismic::Element::named(registry::representation_info(id).name).unwrap()
}

fn rows() -> Vec<u64> {
    std::env::var("ROUTED_ROWS")
        .map(|value| value.split(',').map(|row| row.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![512])
}

fn entry(name: &str) -> bool {
    std::env::var("ROUTED_ENTRIES").map_or(true, |entries| entries.split(',').any(|e| e == name))
}

/// A deterministic generator in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// K distinct experts per row drawn with Zipf-like popularity (exponent
/// 0.8 over a per-layer permutation of the experts).
#[allow(non_snake_case)]
fn skewed_routes(rows: u64, layer: usize, E: u64, K: u64) -> Vec<i32> {
    // `ROUTED_SPREAD=uniform`: every expert takes the same share of choices.
    if std::env::var("ROUTED_SPREAD").is_ok_and(|spread| spread == "uniform") {
        return (0..rows)
            .flat_map(|row| {
                (0..K).map(move |choice| ((row * K + choice + 13 * layer as u64) % E) as i32)
            })
            .collect();
    }
    let mut random = Lcg(0x9e37_79b9 + layer as u64);
    let mut permutation: Vec<i32> = (0..E as i32).collect();
    for index in (1..permutation.len()).rev() {
        let other = (random.next() * (index + 1) as f64) as usize;
        permutation.swap(index, other);
    }
    let popularity: Vec<f64> = (0..E)
        .map(|rank| 1.0 / ((rank + 4) as f64).powf(0.8))
        .collect();
    let total: f64 = popularity.iter().sum();
    let mut routes = Vec::with_capacity((rows * K) as usize);
    for _ in 0..rows {
        let mut chosen: Vec<i32> = Vec::new();
        while chosen.len() < K as usize {
            let mut draw = random.next() * total;
            let mut rank = 0;
            while rank + 1 < popularity.len() && draw >= popularity[rank] {
                draw -= popularity[rank];
                rank += 1;
            }
            let expert = permutation[rank];
            if !chosen.contains(&expert) {
                chosen.push(expert);
            }
        }
        routes.extend(chosen);
    }
    routes
}

/// The grouped tables (order, inverse, blocks) of `routes`: the group
/// contract's host mirror.
#[allow(non_snake_case)]
fn host_group(
    routes: &[i32],
    rows: usize,
    blocks: usize,
    E: u64,
    K: u64,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, usize) {
    let (k, tile) = (K as usize, TILE as usize);
    let mut order = vec![-1; blocks * tile];
    let mut inverse = vec![0; rows * k];
    let mut table = vec![-1; blocks];
    let mut block = 0;
    for expert in 0..E as i32 {
        let mut lane = 0;
        for (flat, route) in routes.iter().enumerate() {
            if *route == expert {
                table[block] = expert;
                order[block * tile + lane] = (flat / k) as i32;
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
    (order, inverse, table, block)
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_tensor(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    seismic::Tensor::from_host(device, seismic::Element::i32(), shape, &bytes).unwrap()
}

fn pattern(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut random = Lcg(seed);
    (0..count)
        .map(|_| (random.next() as f32 * 2.0 - 1.0) * scale)
        .collect()
}

fn f32_tensor(device: &seismic::Device, shape: &[u64], seed: u64, scale: f32) -> seismic::Tensor {
    let count = shape.iter().product::<u64>() as usize;
    seismic::Tensor::from_host(
        device,
        seismic::Element::f32(),
        shape,
        &bytes_f32(&pattern(count, seed, scale)),
    )
    .unwrap()
}

fn bf16_tensor(device: &seismic::Device, shape: &[u64], seed: u64, scale: f32) -> seismic::Tensor {
    let count = shape.iter().product::<u64>() as usize;
    let bytes: Vec<u8> = pattern(count, seed, scale)
        .iter()
        .flat_map(|v| ((registry::bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    seismic::Tensor::from_host(device, seismic::Element::bf16(), shape, &bytes).unwrap()
}

fn print_timings(result: &seismic::TuningResult, flop: &dyn Fn(&str) -> Option<f64>) {
    println!("== {} ({})", result.entry, result.backend);
    for record in &result.configurations {
        let configuration = &record.configuration;
        match &record.outcome {
            seismic::Outcome::Measured { points, .. } => {
                let cells = points
                    .iter()
                    .map(|point| {
                        let rate = flop(&point.point)
                            .map(|flop| {
                                format!(" ({:.1} TFLOP/s)", flop / point.median_seconds / 1e12)
                            })
                            .unwrap_or_default();
                        format!("{} {:.1}us{rate}", point.point, point.median_seconds * 1e6)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "   {:?} {:?}: {cells}",
                    configuration.params, configuration.launches
                );
            }
            seismic::Outcome::Excluded(exclusion) => {
                println!(
                    "   {:?} {:?}: excluded {exclusion:?}",
                    configuration.params, configuration.launches
                );
            }
        }
    }
}

struct Layer {
    gate: seismic::Tensor,
    up: seismic::Tensor,
    down: seismic::Tensor,
    router: seismic::Tensor,
}

struct Inputs {
    residual: seismic::Tensor,
    normalized: seismic::Tensor,
    routes: seismic::Tensor,
    weights: seismic::Tensor,
    order: seismic::Tensor,
    blocks: seismic::Tensor,
    inverse: seismic::Tensor,
    expert_output: seismic::Tensor,
    base: seismic::Tensor,
    capacity: u64,
    live: usize,
}

#[test]
#[ignore]
fn routed_prefill_timings() {
    #[allow(non_snake_case)]
    let (H, E, K, F) = shape();
    println!("(H, E, K, F) = {:?}", (H, E, K, F));
    let device = native_device();
    let bf16 = seismic::Element::bf16();
    let q4 = expert_element(&device);
    let zeros = |element: seismic::Element, shape: &[u64]| {
        seismic::Tensor::zeros(&device, element, shape).unwrap()
    };
    let layers: Vec<Layer> = (0..LAYERS)
        .map(|layer| Layer {
            gate: zeros(q4, &[E, F, H]),
            up: zeros(q4, &[E, F, H]),
            down: zeros(q4, &[E, H, F]),
            router: bf16_tensor(&device, &[E, H], 7 + layer as u64, 0.05),
        })
        .collect();
    let norm = bf16_tensor(&device, &[H], 3, 1.0);
    let bias = zeros(seismic::Element::f32(), &[E]);
    let expert_scale = f32_tensor(&device, &[E], 4, 1.0);
    let measure = seismic::Strategy::Survey(seismic::SurveyPlan {
        samples: 9,
        min_sample_seconds: 0.002,
        domains: Default::default(),
    });
    let statics = |pairs: &[(&str, u64)]| {
        pairs.iter().fold(
            seismic::NativeSpecialization::new(),
            |spec, (name, value)| spec.with_static(*name, *value),
        )
    };
    // Timing only: the zero weights make every candidate agree, so a loose
    // bound suffices.
    let loose = seismic::NumericalRequirement::RoundingSteps {
        alpha: f64::MAX,
        activation: None,
    };
    let all_rows = rows();
    let mut inputs: Vec<Vec<Inputs>> = all_rows
        .iter()
        .map(|&m| {
            (0..LAYERS)
                .map(|layer| {
                    let routes = skewed_routes(m, layer, E, K);
                    let capacity = (m * K + E.min(m * K) * (TILE - 1)).div_ceil(TILE);
                    let (order, inverse, table, live) =
                        host_group(&routes, m as usize, capacity as usize, E, K);
                    let weights = pattern((m * K) as usize, 50 + layer as u64, 0.2);
                    Inputs {
                        residual: f32_tensor(&device, &[m, H], 10 + layer as u64, 1.0),
                        normalized: bf16_tensor(&device, &[m, H], 20 + layer as u64, 1.0),
                        routes: i32_tensor(&device, &[m, K], &routes),
                        weights: seismic::Tensor::from_host(
                            &device,
                            seismic::Element::f32(),
                            &[m, K],
                            &bytes_f32(&weights),
                        )
                        .unwrap(),
                        order: i32_tensor(&device, &[capacity, TILE], &order),
                        blocks: i32_tensor(&device, &[capacity], &table),
                        inverse: i32_tensor(&device, &[m, K], &inverse),
                        expert_output: bf16_tensor(
                            &device,
                            &[capacity, TILE, H],
                            30 + layer as u64,
                            1.0,
                        ),
                        base: f32_tensor(&device, &[m, H], 40 + layer as u64, 1.0),
                        capacity,
                        live,
                    }
                })
                .collect()
        })
        .collect();
    for (m, layer_inputs) in all_rows.iter().zip(&inputs) {
        let live: Vec<usize> = layer_inputs.iter().map(|input| input.live).collect();
        println!(
            "m{m}: capacity {} blocks, live blocks per layer {live:?}",
            layer_inputs[0].capacity
        );
    }
    let routed_flop = |label: &str| {
        let m: f64 = label.trim_start_matches('m').parse().ok()?;
        Some(m * K as f64 * 2.0 * 3.0 * (H * F) as f64)
    };
    let router_flop = |label: &str| {
        let m: f64 = label.trim_start_matches('m').parse().ok()?;
        Some(m * 2.0 * (E * H) as f64)
    };
    let none = |_: &str| None;

    if entry("select") {
        let points: Vec<_> = all_rows
            .iter()
            .zip(inputs.iter_mut())
            .map(|(m, rows)| seismic::TuningPoint {
                label: format!("m{m}"),
                weight: 1.0,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: rows
                    .iter_mut()
                    .zip(&layers)
                    .map(|(input, layer)| routed_select::Args {
                        residual: &input.residual,
                        norm: &norm,
                        router_norm: &norm,
                        router: &layer.router,
                        bias: &bias,
                        expert_scale: &expert_scale,
                        routes: &mut input.routes,
                        weights: &mut input.weights,
                        epsilon: 1e-6,
                        score: 0,
                        normalization: 1,
                        normalization_epsilon: 0.0,
                        scale: 1.0,
                    })
                    .collect(),
                initialize: Some(Box::new(|| Ok(()))),
                written: Default::default(),
            })
            .collect();
        print_timings(
            &routed_select::native_tune_with(
                &device,
                routed_select::Elements {
                    NW: bf16,
                    RNW: bf16,
                    RW: bf16,
                    A: bf16,
                },
                &statics(&[("H", H), ("E", E), ("K", K)]),
                points,
                loose.clone(),
                measure.clone(),
            )
            .unwrap(),
            &router_flop,
        );
    }

    if entry("group") {
        let mut tables: Vec<Vec<_>> = all_rows
            .iter()
            .zip(&inputs)
            .map(|(&m, rows)| {
                rows.iter()
                    .map(|input| {
                        (
                            zeros(seismic::Element::i32(), &[E]),
                            zeros(seismic::Element::i32(), &[input.capacity, TILE]),
                            zeros(seismic::Element::i32(), &[m, K]),
                            zeros(seismic::Element::i32(), &[input.capacity]),
                        )
                    })
                    .collect()
            })
            .collect();
        let points: Vec<_> = all_rows
            .iter()
            .zip(&inputs)
            .zip(tables.iter_mut())
            .map(|((m, rows), tables)| seismic::TuningPoint {
                label: format!("m{m}"),
                weight: 1.0,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: rows
                    .iter()
                    .zip(tables.iter_mut())
                    .map(
                        |(input, (counts, order, inverse, blocks))| routed_group::Args {
                            routes: &input.routes,
                            counts,
                            order,
                            inverse,
                            blocks,
                        },
                    )
                    .collect(),
                initialize: Some(Box::new(|| Ok(()))),
                written: Default::default(),
            })
            .collect();
        print_timings(
            &routed_group::native_tune(
                &device,
                &statics(&[("E", E), ("K", K)]),
                points,
                seismic::NumericalRequirement::BitExact,
                measure.clone(),
            )
            .unwrap(),
            &none,
        );
    }

    if entry("experts") {
        let points: Vec<_> = all_rows
            .iter()
            .zip(&inputs)
            .map(|(m, rows)| seismic::TuningPoint {
                label: format!("m{m}"),
                weight: 1.0,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: rows
                    .iter()
                    .zip(&layers)
                    .map(|(input, layer)| routed_experts::Args {
                        normalized: &input.normalized,
                        order: &input.order,
                        blocks: &input.blocks,
                        expert_gate: &layer.gate,
                        expert_up: &layer.up,
                        expert_down: &layer.down,
                        activation: 1,
                    })
                    .collect(),
                initialize: None,
                written: Default::default(),
            })
            .collect();
        print_timings(
            &routed_experts::native_tune_with(
                &device,
                routed_experts::Elements {
                    A: bf16,
                    EGW: q4,
                    EUW: q4,
                    EDW: q4,
                },
                &statics(&[("H", H), ("F", F)]),
                points,
                loose.clone(),
                measure.clone(),
            )
            .unwrap(),
            &routed_flop,
        );
    }

    if entry("scatter") {
        let points: Vec<_> = all_rows
            .iter()
            .zip(&inputs)
            .map(|(m, rows)| seismic::TuningPoint {
                label: format!("m{m}"),
                weight: 1.0,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: rows
                    .iter()
                    .map(|input| routed_scatter::Args {
                        base: &input.base,
                        expert_output: &input.expert_output,
                        inverse: &input.inverse,
                        weights: &input.weights,
                    })
                    .collect(),
                initialize: None,
                written: Default::default(),
            })
            .collect();
        print_timings(
            &routed_scatter::native_tune_with(
                &device,
                routed_scatter::Elements {
                    A: bf16,
                    R: seismic::Element::f32(),
                },
                &statics(&[]),
                points,
                loose.clone(),
                measure,
            )
            .unwrap(),
            &none,
        );
    }
}
