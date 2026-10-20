//! The progressive head readout (`readout.seismic`) on Metal (macOS) and CUDA
//! (elsewhere): the natives against their portable bodies (the Seismic
//! interpreter), and certified selections against the full planes readout.
//! Tests return early without a Metal or CUDA device.

use magnitude_kernels::{
    readout_exact_rows, readout_planes_rows, readout_refine_rows, readout_top_rows,
};
use seismic::{BackendName, Device, Element, NativeSpecialization, Tensor};
use seismic_lang::{
    checked::{CheckedModule, SourceFile, check_source},
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry::{self, bf16_round, f16_bits},
    types::DType,
};

mod timing_common;

const EPSILON: f32 = 1e-6;

fn device() -> Option<Device> {
    let backend = if cfg!(target_os = "macos") {
        BackendName::Metal
    } else {
        BackendName::Cuda
    };
    seismic::Devices::discover(&[]).ok()?.open_backend(backend).ok()
}

/// The launch of the top level's and the full pass's vocabulary projection
/// that takes the scan parameters: Metal's library GEMV (after the full
/// pass's stage launch), CUDA's scan (after its form launch).
fn scan_launch(device: &Device, planes: bool) -> usize {
    match (device.backend(), planes) {
        (BackendName::Metal, false) => 0,
        _ => 1,
    }
}

/// A top level's or full pass's specialization with the scan parameters;
/// on Metal also the first row count its batched GEMV serves (3 with four
/// rows per simdgroup, so the batched path is exercised, else never).
fn scan_spec(
    device: &Device,
    head: &Head,
    planes: bool,
    scan: u64,
    rows: u64,
) -> NativeSpecialization {
    let launch = scan_launch(device, planes);
    let spec = formed_statics(head)
        .with_launch_param(launch, width_param(device), scan)
        .with_launch_param(launch, "ROWS", rows);
    if device.backend() != BackendName::Metal {
        return spec;
    }
    // Metal's every launch: the GEMV, the batched GEMV and the full pass's
    // GEMM.
    let (gemv, batch) = (launch, launch + 1);
    let spec = spec
        .with_param("BATCH_FROM", if rows == 4 { 3 } else { 9 })
        .with_launch_param(gemv, "LANES", 32)
        .with_launch_param(batch, "BATCH_SIMDGROUPS", scan)
        .with_launch_param(batch, "BATCH_ROWS", rows.min(2));
    if planes {
        spec.with_launch_param(batch + 1, "TILE_M", 64)
            .with_launch_param(batch + 1, "TILE_N", 64)
    } else {
        spec
    }
}

/// The threadgroup-size parameter of the scan and gather launches.
fn width_param(device: &Device) -> &'static str {
    if device.backend() == BackendName::Metal {
        "SIMDGROUPS"
    } else {
        "WARPS"
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() as f32 / (1u64 << 31) as f32)
    }
}

fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * if exponent == 0 {
        mantissa * 2f32.powi(-24)
    } else {
        (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

/// A Q8_0 head (codes `u = c + 128`, an f16 scale per 32) in progressive
/// planes, with its exact rows and the radii of its 4- and 5-bit views.
struct Head {
    v: usize,
    d: usize,
    bits: usize,
    top: Vec<u32>,
    bit3: Vec<u32>,
    rest: Vec<u32>,
    scales: Vec<u16>,
    exact: Vec<f32>,
    radius: Vec<f32>,
}

impl Head {
    /// Rows of random codes and scales.
    fn new(v: usize, d: usize, seed: u64) -> Self {
        Self::with_bits(v, d, seed, 8)
    }
    fn with_bits(v: usize, d: usize, seed: u64, bits: usize) -> Self {
        assert!(matches!(bits, 4 | 8));
        let half = bits / 2;
        let low = half - 1;
        let per_word = 32 / half;
        let offset = (1u32 << (bits - 1)) as f32;
        let mut rng = Rng(seed);
        let groups = d / 32;
        let mut codes = vec![0u8; v * d];
        let mut scales = vec![0u16; v * groups];
        for n in 0..v {
            for g in 0..groups {
                let sign = if bits == 4 && (n + g) % 2 == 0 {
                    -1.0
                } else {
                    1.0
                };
                scales[n * groups + g] =
                    f16_bits(sign * rng.uniform(0.5, 1.5) * 0.02 / (d as f32).sqrt());
                for i in 0..32 {
                    codes[n * d + 32 * g + i] = (rng.next() % (1u32 << bits)) as u8;
                }
            }
        }
        let value = |n: usize, k: usize, code: f32| {
            f16_value(scales[n * groups + k / 32]) * (code - offset)
        };
        let mut top = vec![0u32; v * d * half / 32];
        let mut bit3 = vec![0u32; v * groups];
        let mut rest = vec![0u32; v * groups * low];
        let mut exact = vec![0f32; v * d];
        let mut views = [vec![0f32; v * d], vec![0f32; v * d]];
        for n in 0..v {
            for k in 0..d {
                let u = u32::from(codes[n * d + k]);
                let (g, within) = (k / 32, k % 32);
                let shift = 4 * (within % 8) + within / 8;
                top[n * d / per_word + k / per_word] |= (u >> half) << (half * (k % per_word));
                bit3[n * groups + g] |= ((u >> low) & 1) << shift;
                for j in 0..low {
                    rest[(n * groups + g) * low + j] |= ((u >> (low - 1 - j)) & 1) << shift;
                }
                exact[n * d + k] = value(n, k, u as f32);
                views[0][n * d + k] = value(
                    n,
                    k,
                    (u >> half) as f32 * (1u32 << half) as f32 + 0.5 * ((1u32 << half) - 1) as f32,
                );
                views[1][n * d + k] = value(
                    n,
                    k,
                    (u >> low) as f32 * (1u32 << low) as f32 + 0.5 * ((1u32 << low) - 1) as f32,
                );
            }
        }
        let norm =
            |values: &mut dyn Iterator<Item = f64>| values.map(|x| x * x).sum::<f64>().sqrt();
        let mut radius = vec![0f32; 2 * v];
        for (level, view) in views.iter().enumerate() {
            for n in 0..v {
                let row = n * d..(n + 1) * d;
                let error = norm(&mut row.clone().map(|i| f64::from(exact[i] - view[i])));
                let length = norm(&mut row.map(|i| f64::from(exact[i])));
                radius[2 * n + level] =
                    (error * (1.0 + 1e-6) + 4.0 * d as f64 * 2f64.powi(-24) * length) as f32;
            }
        }
        Self {
            v,
            d,
            bits,
            top,
            bit3,
            rest,
            scales,
            exact,
            radius,
        }
    }
    fn row(&self, n: usize) -> &[f32] {
        &self.exact[n * self.d..(n + 1) * self.d]
    }
}

/// The selection side of O rows: row o samples (temperature 0.8) when o is
/// odd, else greedy; row 2 (when present) competes only over a random half of
/// the vocabulary.
struct Selection {
    draws: Vec<u32>,
    temperature: Vec<f32>,
    mask: Vec<u32>,
    constrained: Vec<i32>,
}

impl Selection {
    fn new(o: usize, v: usize, rng: &mut Rng) -> Self {
        let words = v.div_ceil(32);
        let mut draws = vec![0u32; o * 6];
        for row in 0..o {
            if row % 2 == 1 {
                draws[row * 6] = 1;
                for word in 1..6 {
                    draws[row * 6 + word] = rng.next();
                }
            }
        }
        Self {
            draws,
            temperature: (0..o)
                .map(|row| if row % 2 == 1 { 0.8 } else { 1.0 })
                .collect(),
            mask: (0..o * words).map(|_| rng.next()).collect(),
            constrained: (0..o).map(|row| i32::from(row == 2)).collect(),
        }
    }
    fn competes(&self, row: usize, token: usize, v: usize) -> bool {
        self.constrained[row] == 0
            || (self.mask[row * v.div_ceil(32) + token / 32] >> (token % 32)) & 1 != 0
    }
    /// The sampler's Gumbel noise (`gumbel_rows`).
    fn noise(&self, row: usize, token: usize) -> f32 {
        let draw = &self.draws[row * 6..row * 6 + 6];
        if draw[0] != 1 {
            return 0.0;
        }
        let (mut c0, mut c1, mut c2, mut c3) = (token as u32, draw[3], draw[4], draw[5]);
        let (mut k0, mut k1) = (draw[1], draw[2]);
        for _ in 0..10 {
            let p0 = u64::from(3528531795u32) * u64::from(c0);
            let p1 = u64::from(3449720151u32) * u64::from(c2);
            let next0 = (p1 >> 32) as u32 ^ c1 ^ k0;
            let next2 = (p0 >> 32) as u32 ^ c3 ^ k1;
            c0 = next0;
            c1 = p1 as u32;
            c2 = next2;
            c3 = p0 as u32;
            k0 = k0.wrapping_add(2654435769);
            k1 = k1.wrapping_add(3144134277);
        }
        let uniform = ((c0 >> 9) as f32 + 0.5) * 0.00000011920928955078125;
        -(-uniform.ln()).ln()
    }
    /// What the sampler selects from row `row` of `logits`: the competing
    /// finite logit with the largest `logit / temperature + noise`.
    fn select(&self, row: usize, logits: &[f32], v: usize) -> Option<usize> {
        (0..v)
            .filter(|&token| logits[row * v + token].is_finite() && self.competes(row, token, v))
            .map(|token| {
                (
                    logits[row * v + token] / self.temperature[row] + self.noise(row, token),
                    token,
                )
            })
            .fold(None, |best: Option<(f32, usize)>, candidate| match best {
                Some(best) if best.0 >= candidate.0 => Some(best),
                _ => Some(candidate),
            })
            .map(|(_, token)| token)
    }
}

struct Tensors {
    hidden: Tensor,
    norm: Tensor,
    top: Tensor,
    bit3: Tensor,
    rest: Tensor,
    scales: Tensor,
    radius: Tensor,
    out_rows: Tensor,
    draws: Tensor,
    temperature: Tensor,
    mask: Tensor,
    constrained: Tensor,
}

fn tensor(device: &Device, element: Element, shape: &[usize], bytes: Vec<u8>) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    Tensor::from_host(device, element, &shape, &bytes).unwrap()
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16))
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

struct Case {
    o: usize,
    m: usize,
    hidden: Vec<f32>,
    norm: Vec<f32>,
    out_rows: Vec<i32>,
    selection: Selection,
}

impl Case {
    /// O rows gathered from M = O + 1 residual rows; `peaked` rows lean
    /// towards head row `v / 3`, so few vocabulary rows compete.
    fn new(o: usize, head: &Head, peaked: bool, rng: &mut Rng) -> Self {
        let (m, d) = (o + 1, head.d);
        let mut hidden = (0..m * d)
            .map(|_| rng.uniform(-1.0, 1.0))
            .collect::<Vec<_>>();
        if peaked {
            let lead = head.row(head.v / 3);
            let length = lead.iter().map(|x| x * x).sum::<f32>().sqrt();
            for row in 0..m {
                for k in 0..d {
                    hidden[row * d + k] = 0.05 * hidden[row * d + k] + 8.0 * lead[k] / length;
                }
            }
        }
        Self {
            o,
            m,
            hidden,
            norm: (0..d).map(|_| rng.uniform(0.75, 1.25)).collect(),
            out_rows: (0..o).map(|i| ((i * 2 + 1) % m) as i32).collect(),
            selection: Selection::new(o, head.v, rng),
        }
    }
    fn tensors(&self, device: &Device, head: &Head) -> Tensors {
        let (v, d, o) = (head.v, head.d, self.o);
        let groups = d / 32;
        let selection = &self.selection;
        Tensors {
            // The levels read the rows as `[M, PT, D]` parts (one part).
            hidden: tensor(device, Element::f32(), &[self.m, 1, d], f32s(&self.hidden)),
            norm: tensor(device, Element::bf16(), &[d], bf16_bytes(&self.norm)),
            top: tensor(
                device,
                Element::u32(),
                &[v, groups, head.bits / 2],
                words(&head.top),
            ),
            bit3: tensor(device, Element::u32(), &[v, groups], words(&head.bit3)),
            rest: tensor(
                device,
                Element::u32(),
                &[v, groups, head.bits / 2 - 1],
                words(&head.rest),
            ),
            scales: tensor(
                device,
                Element::f16(),
                &[v, groups],
                head.scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
            ),
            radius: tensor(device, Element::f32(), &[v, 2], f32s(&head.radius)),
            out_rows: tensor(
                device,
                Element::i32(),
                &[o],
                self.out_rows.iter().flat_map(|r| r.to_le_bytes()).collect(),
            ),
            draws: tensor(device, Element::u32(), &[o, 6], words(&selection.draws)),
            temperature: tensor(device, Element::f32(), &[o], f32s(&selection.temperature)),
            mask: tensor(
                device,
                Element::u32(),
                &[o, v.div_ceil(32)],
                words(&selection.mask),
            ),
            constrained: tensor(
                device,
                Element::i32(),
                &[o],
                selection
                    .constrained
                    .iter()
                    .flat_map(|c| c.to_le_bytes())
                    .collect(),
            ),
        }
    }
}

/// The natives' launch parameters: (scan width, scan rows, gather width).
type Params = (u64, u64, u64);

struct Levels {
    coarse: Vec<f32>,
    threshold4: Vec<f32>,
    features: Tensor,
    length: Tensor,
    fine: Vec<f32>,
    threshold5: Vec<f32>,
    exact: Vec<f32>,
    planes: Vec<f32>,
}

fn statics(head: &Head) -> NativeSpecialization {
    NativeSpecialization::new()
        .with_static("V", head.v as u64)
        .with_static("D", head.d as u64)
        .with_static("P", (head.bits / 2 - 1) as u64)
}

/// The statics of the entries that form their input (top, planes): one
/// part normalized at the row width.
fn formed_statics(head: &Head) -> NativeSpecialization {
    statics(head)
        .with_static("PT", 1)
        .with_static("NM", head.d as u64)
}

/// Every level natively, each from the previous native level.
fn run(device: &Device, head: &Head, t: &Tensors, params: Params) -> Levels {
    run_with_softcap(device, head, t, params, 0.0)
}

fn run_with_softcap(
    device: &Device,
    head: &Head,
    t: &Tensors,
    (scan, rows, gather): Params,
    softcap: f32,
) -> Levels {
    let width = width_param(device);
    let top = readout_top_rows::native_for_device_with(
        device,
        readout_top_rows::Elements {
            NW: Element::bf16(),
            A: Element::bf16(),
        },
        &scan_spec(device, head, false, scan, rows),
    )
    .unwrap()
    .call(readout_top_rows::Args {
        softcap,
        hidden: &t.hidden,
        norm: &t.norm,
        top: &t.top,
        bit3: &t.bit3,
        rest: &t.rest,
        scales: &t.scales,
        radius: &t.radius,
        out_rows: &t.out_rows,
        draws: &t.draws,
        temperature: &t.temperature,
        mask: &t.mask,
        constrained: &t.constrained,
        epsilon: EPSILON,
    })
    .unwrap();
    let refine = readout_refine_rows::native_for_device_with(
        device,
        readout_refine_rows::Elements { A: Element::bf16() },
        &statics(head).with_launch_param(0, width, gather),
    )
    .unwrap()
    .call(readout_refine_rows::Args {
        softcap,
        features: &top.r2,
        top: &t.top,
        bit3: &t.bit3,
        rest: &t.rest,
        scales: &t.scales,
        radius: &t.radius,
        coarse: &top.r0,
        floor: &top.r1,
        length: &top.r3,
        draws: &t.draws,
        temperature: &t.temperature,
        mask: &t.mask,
        constrained: &t.constrained,
    })
    .unwrap();
    let exact = readout_exact_rows::native_for_device_with(
        device,
        readout_exact_rows::Elements { A: Element::bf16() },
        &statics(head).with_launch_param(0, width, gather),
    )
    .unwrap()
    .call(readout_exact_rows::Args {
        softcap,
        features: &top.r2,
        top: &t.top,
        bit3: &t.bit3,
        rest: &t.rest,
        scales: &t.scales,
        radius: &t.radius,
        fine: &refine.r0,
        floor: &refine.r1,
        length: &top.r3,
        draws: &t.draws,
        temperature: &t.temperature,
        mask: &t.mask,
        constrained: &t.constrained,
    })
    .unwrap()
    .value;
    let planes = readout_planes_rows::native_for_device_with(
        device,
        readout_planes_rows::Elements {
            NW: Element::bf16(),
            A: Element::bf16(),
        },
        &scan_spec(device, head, true, scan, rows),
    )
    .unwrap()
    .call(readout_planes_rows::Args {
        softcap,
        hidden: &t.hidden,
        norm: &t.norm,
        top: &t.top,
        bit3: &t.bit3,
        rest: &t.rest,
        scales: &t.scales,
        out_rows: &t.out_rows,
        epsilon: EPSILON,
    })
    .unwrap()
    .value;
    Levels {
        coarse: read_f32(&top.r0),
        threshold4: read_f32(&top.r1),
        features: top.r2,
        length: top.r3,
        fine: read_f32(&refine.r0),
        threshold5: read_f32(&refine.r1),
        exact: read_f32(&exact),
        planes: read_f32(&planes),
    }
}

/// A certified selection keeps every row its exact level can select, with the
/// full planes readout's logits bit for bit, so it selects what the full
/// readout selects; on peaked rows it keeps few rows.
#[test]
fn progressive_selection_selects_what_the_full_readout_selects() {
    let Some(device) = device() else { return };
    let head = Head::new(3001, 2048, 41);
    let mut rng = Rng(5);
    for peaked in [false, true] {
        for o in [1usize, 2, 3, 5, 8] {
            let case = Case::new(o, &head, peaked, &mut rng);
            let t = case.tensors(&device, &head);
            for params in [(8u64, 2u64, 8u64), (4, 1, 4), (8, 4, 4)] {
                let context = format!("peaked {peaked} O {o} params {params:?}");
                let levels = run(&device, &head, &t, params);
                let (v, d) = (head.v, head.d);
                let x = read_bf16(&levels.features);
                let length = read_f32(&levels.length);
                for row in 0..o {
                    let features = &x[row * d..(row + 1) * d];
                    let host_length = features
                        .iter()
                        .map(|x| f64::from(*x).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        (f64::from(length[row]) - host_length).abs() <= 1e-5 * host_length,
                        "{context}: row {row} length {} host {host_length}",
                        length[row]
                    );
                    let mut kept = 0;
                    for token in 0..v {
                        let i = row * v + token;
                        let planes = levels.planes[i];
                        let host = head
                            .row(token)
                            .iter()
                            .zip(features)
                            .map(|(w, x)| f64::from(*w) * f64::from(*x))
                            .sum::<f64>();
                        let magnitude = head
                            .row(token)
                            .iter()
                            .zip(features)
                            .map(|(w, x)| f64::from((w * x).abs()))
                            .sum::<f64>();
                        assert!(
                            (f64::from(planes) - host).abs() <= 1e-4 * (magnitude + 1e-6),
                            "{context}: row {row} token {token} planes {planes} host {host}"
                        );
                        let exact = levels.exact[i];
                        if exact.is_finite() {
                            kept += 1;
                            // CUDA's exact level and full pass share one routine,
                            // so they agree bit for bit; Metal's full pass is the
                            // projection library's, within its rounding.
                            if device.backend() == BackendName::Cuda {
                                assert_eq!(
                                    exact.to_bits(),
                                    planes.to_bits(),
                                    "{context}: row {row} token {token} exact {exact} planes {planes}"
                                );
                            } else {
                                assert!(
                                    (f64::from(exact) - f64::from(planes)).abs()
                                        <= 1e-5 * (magnitude + 1e-6),
                                    "{context}: row {row} token {token} exact {exact} planes {planes}"
                                );
                            }
                        } else {
                            assert_eq!(
                                exact,
                                f32::NEG_INFINITY,
                                "{context}: row {row} token {token}"
                            );
                        }
                        // Each level keeps the rows the next keeps.
                        if exact.is_finite() {
                            assert!(
                                levels.fine[i].is_finite(),
                                "{context}: row {row} token {token} skipped level two"
                            );
                        }
                    }
                    assert!(
                        levels.threshold4[row] <= levels.threshold5[row],
                        "{context}: row {row} thresholds"
                    );
                    assert_eq!(
                        case.selection.select(row, &levels.exact, v),
                        case.selection.select(row, &levels.planes, v),
                        "{context}: row {row} selects differently from the full readout"
                    );
                    // A peaked row whose mask admits the leading token.
                    if peaked && case.selection.competes(row, v / 3, v) {
                        assert!(kept * 100 < v, "{context}: row {row} keeps {kept} of {v}");
                    }
                }
                let coarse_finite = levels.coarse.iter().all(|z| z.is_finite());
                assert!(coarse_finite, "{context}: level one covers every row");
            }
        }
    }
}

/// A monotone cap can turn distinct raw logits into equal F32 scores.
/// Bounds must keep every tied competitor, including one preferred by the
/// sampler's token tie rule, while sampled/constrained rows use the same draws.
#[test]
fn progressive_softcap_preserves_full_selection_and_saturated_ties() {
    let Some(device) = device() else { return };
    for bits in [4, 8] {
        let head = Head::with_bits(67, 256, 41, bits);
        let mut rng = Rng(5);
        for rows in [1, 3] {
            let case = Case::new(rows, &head, false, &mut rng);
            let t = case.tensors(&device, &head);
            for cap in [0.0, 0.001, 0.1, 30.0] {
                let levels = run_with_softcap(&device, &head, &t, (8, 2, 8), cap);
                for (token, (&selected, &full)) in
                    levels.exact.iter().zip(&levels.planes).enumerate()
                {
                    if selected.is_finite() {
                        assert_eq!(
                            selected.to_bits(),
                            full.to_bits(),
                            "cap {cap} token {token}: survivor and full scan must publish the same dot"
                        );
                    }
                }
                for row in 0..case.o {
                    assert_eq!(
                        case.selection.select(row, &levels.exact, head.v),
                        case.selection.select(row, &levels.planes, head.v),
                        "cap {cap} row {row}: certified selection differs"
                    );
                    assert!(levels.threshold4[row] <= levels.threshold5[row]);
                }
                if cap == 0.001 {
                    let maximum = levels.planes[..head.v]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let tied = levels.planes[..head.v]
                        .iter()
                        .filter(|&&z| z == maximum)
                        .count();
                    assert!(tied > 1, "the fixture must create actual published ties");
                }
            }
        }
    }
}

/// The full planes readout projects every row class, in blocks of eight rows:
/// each row's logits are its features' exact dot products with the head.
#[test]
fn planes_readout_projects_every_row_class() {
    let Some(device) = device() else { return };
    let head = Head::new(301, 1088, 43);
    let mut rng = Rng(11);
    for o in [9usize, 16, 21] {
        let case = Case::new(o, &head, false, &mut rng);
        let t = case.tensors(&device, &head);
        let (v, d) = (head.v, head.d);
        let planes = read_f32(
            &readout_planes_rows::native_for_device_with(
                &device,
                readout_planes_rows::Elements {
                    NW: Element::bf16(),
                    A: Element::bf16(),
                },
                &scan_spec(&device, &head, true, 4, 2),
            )
            .unwrap()
            .call(readout_planes_rows::Args {
                softcap: 0.0,
                hidden: &t.hidden,
                norm: &t.norm,
                top: &t.top,
                bit3: &t.bit3,
                rest: &t.rest,
                scales: &t.scales,
                out_rows: &t.out_rows,
                epsilon: EPSILON,
            })
            .unwrap()
            .value,
        );
        for row in 0..o {
            let source = case.out_rows[row] as usize;
            let hidden = &case.hidden[source * d..(source + 1) * d];
            let mean = hidden.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / d as f64;
            let inverse = 1.0 / (mean + f64::from(EPSILON)).sqrt();
            let features = hidden
                .iter()
                .zip(&case.norm)
                .map(|(x, w)| {
                    f64::from(bf16_round(
                        (f64::from(*x) * inverse * f64::from(bf16_round(*w))) as f32,
                    ))
                })
                .collect::<Vec<_>>();
            for token in 0..v {
                let (host, magnitude) = head.row(token).iter().zip(&features).fold(
                    (0f64, 0f64),
                    |(sum, size), (w, x)| {
                        (sum + f64::from(*w) * x, size + (f64::from(*w) * x).abs())
                    },
                );
                let device = f64::from(planes[row * v + token]);
                // The features may round to a neighbouring bf16 value.
                assert!(
                    (device - host).abs() <= 4e-3 * magnitude + 1e-6,
                    "O {o} row {row} token {token}: planes {device} host {host}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Portable bodies.

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    for (path, text) in [
        (
            "dense_rows.seismic",
            include_str!("../kernels/dense_rows.seismic"),
        ),
        (
            "readout.seismic",
            include_str!("../kernels/readout.seismic"),
        ),
        ("target.seismic", include_str!("../kernels/target.seismic")),
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
    Data(TensorData),
    F32(f32),
}

fn dense(dtype: DType, shape: &[usize], values: impl Iterator<Item = f64>) -> Input {
    Input::Data(TensorData::dense(dtype, shape.to_vec(), values.collect()))
}

fn interpret(
    module: &CheckedModule,
    name: &str,
    bindings: &[&str],
    inputs: Vec<Input>,
) -> OracleOutcome {
    let elements = bindings
        .iter()
        .fold(ElementBindings::new(), |elements, name| {
            elements.bind(name, registry::dense(DType::BF16))
        });
    let logical = module
        .entry(module.entry_named(name).unwrap(), &elements)
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let arguments = inputs
        .into_iter()
        .map(|input| match input {
            Input::Data(data) => Arg::Tensor(interpreter.add_tensor(data)),
            Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
        })
        .collect::<Vec<_>>();
    let outcome = interpreter
        .run(&arguments)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{name} failed: {failure}");
    }
    outcome
}

fn result(outcome: &OracleOutcome, index: usize) -> Vec<f32> {
    let result = outcome.results().nth(index).unwrap();
    let OutcomeValue::Tensor(tensor) = result.value() else {
        panic!("tensor result")
    };
    (0..tensor.element_count())
        .map(|i| tensor.read(i).unwrap() as f32)
        .collect()
}

/// Native values agree with portable ones: equal non-finite values, finite
/// values within `tolerance` relative to `scale`.
fn assert_close(label: &str, native: &[f32], portable: &[f32], scale: f32) {
    assert_eq!(native.len(), portable.len(), "{label}: lengths");
    for (i, (a, b)) in native.iter().zip(portable).enumerate() {
        let close =
            a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-4 * (scale + b.abs()));
        assert!(close, "{label} {i}: native {a} portable {b}");
    }
}

/// Native survivors cover the portable ones (the natives' margin keeps extra
/// rows near a threshold, never fewer), with agreeing logits where both keep.
fn assert_covers(label: &str, native: &[f32], portable: &[f32], scale: f32) {
    for (i, (a, b)) in native.iter().zip(portable).enumerate() {
        if b.is_finite() {
            assert!(
                a.is_finite() && (a - b).abs() <= 1e-4 * (scale + b.abs()),
                "{label} {i}: native {a} portable {b}"
            );
        }
    }
}

#[test]
fn progressive_portable_bodies_match_the_natives() {
    let Some(device) = device() else { return };
    let module = module();
    let softcap = 30.0;
    // A vocabulary that is no multiple of 32 or of the scan's rows, and a
    // width with a partial 1,024-column chunk.
    for (v, d, bits) in [
        (67usize, 256usize, 4usize),
        (67, 256, 8),
        (45, 1088, 4),
        (45, 1088, 8),
    ] {
        let head = Head::with_bits(v, d, 77, bits);
        let mut rng = Rng(9);
        let case = Case::new(3, &head, false, &mut rng);
        let (o, m, groups) = (case.o, case.m, d / 32);
        let selection = &case.selection;
        let words = |values: &[u32]| values.iter().map(|w| f64::from(*w)).collect::<Vec<_>>();
        let planes = || {
            vec![
                dense(
                    DType::U32,
                    &[v, groups, head.bits / 2],
                    words(&head.top).into_iter(),
                ),
                dense(DType::U32, &[v, groups], words(&head.bit3).into_iter()),
                dense(
                    DType::U32,
                    &[v, groups, head.bits / 2 - 1],
                    words(&head.rest).into_iter(),
                ),
                dense(
                    DType::F16,
                    &[v, groups],
                    head.scales.iter().map(|s| f64::from(f16_value(*s))),
                ),
                dense(
                    DType::F32,
                    &[v, 2],
                    head.radius.iter().map(|r| f64::from(*r)),
                ),
            ]
        };
        let selecting = || {
            vec![
                dense(DType::U32, &[o, 6], words(&selection.draws).into_iter()),
                dense(
                    DType::F32,
                    &[o],
                    selection.temperature.iter().map(|t| f64::from(*t)),
                ),
                dense(
                    DType::U32,
                    &[o, v.div_ceil(32)],
                    words(&selection.mask).into_iter(),
                ),
                dense(
                    DType::I32,
                    &[o],
                    selection.constrained.iter().map(|c| f64::from(*c)),
                ),
            ]
        };
        let rows = || {
            (
                dense(
                    DType::F32,
                    &[m, 1, d],
                    case.hidden.iter().map(|x| f64::from(*x)),
                ),
                dense(
                    DType::BF16,
                    &[d],
                    case.norm.iter().map(|x| f64::from(bf16_round(*x))),
                ),
                dense(
                    DType::I32,
                    &[o],
                    case.out_rows.iter().map(|r| f64::from(*r)),
                ),
            )
        };
        let (hidden, norm, out_rows) = rows();
        let mut inputs = vec![hidden, norm];
        inputs.extend(planes());
        inputs.push(out_rows);
        inputs.extend(selecting());
        inputs.push(Input::F32(EPSILON));
        inputs.push(Input::F32(softcap));
        let top = interpret(&module, "readout_top_rows", &["NW", "A"], inputs);
        let (coarse, threshold4, features, length) = (
            result(&top, 0),
            result(&top, 1),
            result(&top, 2),
            result(&top, 3),
        );
        let level = |name: &str, previous: &[f32], floor: &[f32]| {
            let mut inputs = vec![dense(
                DType::BF16,
                &[o, d],
                features.iter().map(|x| f64::from(*x)),
            )];
            inputs.extend(planes());
            inputs.push(dense(
                DType::F32,
                &[o, v],
                previous.iter().map(|x| f64::from(*x)),
            ));
            inputs.push(dense(DType::F32, &[o], floor.iter().map(|x| f64::from(*x))));
            inputs.push(dense(
                DType::F32,
                &[o],
                length.iter().map(|x| f64::from(*x)),
            ));
            inputs.extend(selecting());
            inputs.push(Input::F32(softcap));
            interpret(&module, name, &["A"], inputs)
        };
        let refine = level("readout_refine_rows", &coarse, &threshold4);
        let (fine, threshold5) = (result(&refine, 0), result(&refine, 1));
        let exact = result(&level("readout_exact_rows", &fine, &threshold5), 0);
        let (hidden, norm, out_rows) = rows();
        let mut inputs = vec![hidden, norm];
        inputs.extend(planes().into_iter().take(4));
        inputs.push(out_rows);
        inputs.push(Input::F32(EPSILON));
        inputs.push(Input::F32(softcap));
        let full = result(
            &interpret(&module, "readout_planes_rows", &["NW", "A"], inputs),
            0,
        );
        assert!(exact.iter().any(|z| z.is_finite()) && exact.iter().any(|z| !z.is_finite()));

        let t = case.tensors(&device, &head);
        let scale = full.iter().fold(0f32, |a, b| a.max(b.abs()));
        let context = format!("V {v} D {d}");
        for params in [(4u64, 1u64, 4u64), (8, 2, 8)] {
            let levels = run_with_softcap(&device, &head, &t, params, softcap);
            assert_close(
                &format!("{context} top logits"),
                &levels.coarse,
                &coarse,
                scale,
            );
            assert_close(
                &format!("{context} top threshold"),
                &levels.threshold4,
                &threshold4,
                scale,
            );
            assert_close(
                &format!("{context} features"),
                &read_bf16(&levels.features),
                &features,
                1.0,
            );
            assert_close(
                &format!("{context} length"),
                &read_f32(&levels.length),
                &length,
                1.0,
            );
            assert_close(&format!("{context} planes"), &levels.planes, &full, scale);
            assert_close(
                &format!("{context} refine threshold"),
                &levels.threshold5,
                &threshold5,
                scale,
            );
        }
        // Each native level from the portable previous level.
        let width = width_param(&device);
        let features_tensor = tensor(&device, Element::bf16(), &[o, d], bf16_bytes(&features));
        let length_tensor = tensor(&device, Element::f32(), &[o], f32s(&length));
        let native_refine = readout_refine_rows::native_for_device_with(
            &device,
            readout_refine_rows::Elements { A: Element::bf16() },
            &statics(&head).with_launch_param(0, width, 4),
        )
        .unwrap()
        .call(readout_refine_rows::Args {
            softcap,
            features: &features_tensor,
            top: &t.top,
            bit3: &t.bit3,
            rest: &t.rest,
            scales: &t.scales,
            radius: &t.radius,
            coarse: &tensor(&device, Element::f32(), &[o, v], f32s(&coarse)),
            floor: &tensor(&device, Element::f32(), &[o], f32s(&threshold4)),
            length: &length_tensor,
            draws: &t.draws,
            temperature: &t.temperature,
            mask: &t.mask,
            constrained: &t.constrained,
        })
        .unwrap();
        assert_covers(
            &format!("{context} refine from portable"),
            &read_f32(&native_refine.r0),
            &fine,
            scale,
        );
        assert_close(
            &format!("{context} refine threshold from portable"),
            &read_f32(&native_refine.r1),
            &threshold5,
            scale,
        );
        let native_exact = readout_exact_rows::native_for_device_with(
            &device,
            readout_exact_rows::Elements { A: Element::bf16() },
            &statics(&head).with_launch_param(0, width, 4),
        )
        .unwrap()
        .call(readout_exact_rows::Args {
            softcap,
            features: &features_tensor,
            top: &t.top,
            bit3: &t.bit3,
            rest: &t.rest,
            scales: &t.scales,
            radius: &t.radius,
            fine: &tensor(&device, Element::f32(), &[o, v], f32s(&fine)),
            floor: &tensor(&device, Element::f32(), &[o], f32s(&threshold5)),
            length: &length_tensor,
            draws: &t.draws,
            temperature: &t.temperature,
            mask: &t.mask,
            constrained: &t.constrained,
        })
        .unwrap()
        .value;
        assert_covers(
            &format!("{context} exact from portable"),
            &read_f32(&native_exact),
            &exact,
            scale,
        );
    }
}

// ---------------------------------------------------------------------------
// Timing.

const TIMING_OPTIONS: seismic::measure::Measure = seismic::measure::Measure {
    stop: seismic::measure::Stop::Count(std::num::NonZeroUsize::new(9).unwrap()),
    min_sample_seconds: 0.02,
};

/// The device time of `point`'s calls.
fn timed(
    point: Result<seismic::KernelPoint, seismic::CallError>,
) -> seismic::measure::Measurement {
    timing_common::measured(point.unwrap(), &TIMING_OPTIONS)
}

/// Device time of every level at the 35B-A3B head (V = 248320, D = 2048) over
/// random planes: `cargo test --release -p magnitude-kernels --test
/// progressive_readout -- --ignored --nocapture timings`. The later levels
/// run at given shares of surviving rows.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn progressive_timings() {
    let Some(device) = device() else { return };
    let (v, d) = (248_320usize, 2048usize);
    let groups = d / 32;
    let mut rng = Rng(3);
    let mut random = |count: usize| {
        (0..count)
            .map(|_| rng.next() ^ (rng.next() << 16))
            .collect::<Vec<_>>()
    };
    let head = Head {
        v,
        d,
        bits: 8,
        top: random(v * d / 8),
        bit3: random(v * groups),
        rest: random(v * groups * 3),
        scales: vec![f16_bits(0.001); v * groups],
        exact: Vec::new(),
        radius: vec![0.01; 2 * v],
    };
    let width = width_param(&device);
    let report = |label: &str, seconds: f64, bytes: f64| {
        eprintln!(
            "timing {label}: {:.1} us ({:.0} GB/s)",
            seconds * 1e6,
            bytes / seconds / 1e9
        );
    };
    let top_bytes = (v * d) as f64 * (0.5 + 0.0625);
    let full_bytes = (v * d) as f64 * 1.0625;
    for o in [1usize, 2, 4, 8] {
        let case = Case::new(o, &head, false, &mut Rng(o as u64));
        let t = case.tensors(&device, &head);
        for (scan, rows) in [(8u64, 2u64), (4, 2), (8, 4), (4, 4), (8, 1)] {
            let spec = |planes| scan_spec(&device, &head, planes, scan, rows);
            let top = readout_top_rows::native_for_device_with(
                &device,
                readout_top_rows::Elements {
                    NW: Element::bf16(),
                    A: Element::bf16(),
                },
                &spec(false),
            )
            .unwrap();
            let args = readout_top_rows::Args {
                softcap: 0.0,
                hidden: &t.hidden,
                norm: &t.norm,
                top: &t.top,
                bit3: &t.bit3,
                rest: &t.rest,
                scales: &t.scales,
                radius: &t.radius,
                out_rows: &t.out_rows,
                draws: &t.draws,
                temperature: &t.temperature,
                mask: &t.mask,
                constrained: &t.constrained,
                epsilon: EPSILON,
            };
            let measured = timed(top.point(vec![args]));
            report(
                &format!("top O {o} {width} {scan} ROWS {rows}"),
                measured.median,
                top_bytes,
            );
            let planes = readout_planes_rows::native_for_device_with(
                &device,
                readout_planes_rows::Elements {
                    NW: Element::bf16(),
                    A: Element::bf16(),
                },
                &spec(true),
            )
            .unwrap();
            let args = readout_planes_rows::Args {
                softcap: 0.0,
                hidden: &t.hidden,
                norm: &t.norm,
                top: &t.top,
                bit3: &t.bit3,
                rest: &t.rest,
                scales: &t.scales,
                out_rows: &t.out_rows,
                epsilon: EPSILON,
            };
            let measured = timed(planes.point(vec![args]));
            report(
                &format!("planes O {o} {width} {scan} ROWS {rows}"),
                measured.median,
                full_bytes,
            );
        }
        let features = tensor(
            &device,
            Element::bf16(),
            &[o, d],
            bf16_bytes(&vec![0.01; o * d]),
        );
        let length = tensor(&device, Element::f32(), &[o], f32s(&vec![1.0; o]));
        let floor = tensor(
            &device,
            Element::f32(),
            &[o],
            f32s(&vec![f32::NEG_INFINITY; o]),
        );
        // `share` of the rows survive: finite previous logits there, -inf
        // elsewhere, under a floor every finite row reaches.
        for share in [1.0f64, 0.084, 0.01, 0.001] {
            let step = (1.0 / share).round() as usize;
            let previous = (0..o * v)
                .map(|i| {
                    if (i % v) % step == 0 {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>();
            let previous = tensor(&device, Element::f32(), &[o, v], f32s(&previous));
            for gather in [8u64, 4] {
                let spec = statics(&head).with_launch_param(0, width, gather);
                let refine = readout_refine_rows::native_for_device_with(
                    &device,
                    readout_refine_rows::Elements { A: Element::bf16() },
                    &spec,
                )
                .unwrap();
                let args = readout_refine_rows::Args {
                    softcap: 0.0,
                    features: &features,
                    top: &t.top,
                    bit3: &t.bit3,
                    rest: &t.rest,
                    scales: &t.scales,
                    radius: &t.radius,
                    coarse: &previous,
                    floor: &floor,
                    length: &length,
                    draws: &t.draws,
                    temperature: &t.temperature,
                    mask: &t.mask,
                    constrained: &t.constrained,
                };
                let measured = timed(refine.point(vec![args]));
                report(
                    &format!("refine O {o} share {share} {width} {gather}"),
                    measured.median,
                    (v * d) as f64 * share * (0.125 + 0.0625),
                );
                let exact = readout_exact_rows::native_for_device_with(
                    &device,
                    readout_exact_rows::Elements { A: Element::bf16() },
                    &spec,
                )
                .unwrap();
                let args = readout_exact_rows::Args {
                    softcap: 0.0,
                    features: &features,
                    top: &t.top,
                    bit3: &t.bit3,
                    rest: &t.rest,
                    scales: &t.scales,
                    radius: &t.radius,
                    fine: &previous,
                    floor: &floor,
                    length: &length,
                    draws: &t.draws,
                    temperature: &t.temperature,
                    mask: &t.mask,
                    constrained: &t.constrained,
                };
                let measured = timed(exact.point(vec![args]));
                report(
                    &format!("exact O {o} share {share} {width} {gather}"),
                    measured.median,
                    (v * d) as f64 * share * 1.0625,
                );
            }
        }
    }
}

/// Canonical selection changes the physical row lookup, not the projection's
/// scale/cap semantics. Invalid sentinels must never become weight addresses.
#[test]
fn prepared_full_and_indexed_share_projection_and_mask_invalid_ids() {
    let Some(device) = device() else { return };
    let (o, v, d, s) = (2usize, 9usize, 64usize, 5usize);
    let features = (0..o * d)
        .map(|i| bf16_round(((i * 13 % 31) as f32 - 15.0) / 32.0))
        .collect::<Vec<_>>();
    let weights = (0..v * d)
        .map(|i| bf16_round(((i * 7 % 43) as f32 - 21.0) / 64.0))
        .collect::<Vec<_>>();
    let features_tensor = tensor(&device, Element::bf16(), &[o, d], bf16_bytes(&features));
    let weight_tensor = tensor(&device, Element::bf16(), &[v, d], bf16_bytes(&weights));
    let ids = [8i32, 2, 2, -1, 9, 0, 7, 3, i32::MAX, -17];
    let scale_tensor = tensor(&device, Element::f32(), &[1], f32s(&[-0.75]));
    let project = |indexed: bool, cap: f32| {
        let columns = if indexed { s } else { v };
        let tokens = tensor(
            &device,
            Element::i32(),
            &[o, usize::from(indexed), columns],
            if indexed {
                ids.iter().flat_map(|id| id.to_le_bytes()).collect()
            } else {
                Vec::new()
            },
        );
        let spec = NativeSpecialization::new()
            .with_static("V", v as u64)
            .with_static("D", d as u64)
            .with_static("S", columns as u64)
            .with_static("I", u64::from(indexed))
            .with_static("WS", 1);
        let spec = match device.backend() {
            BackendName::Metal => spec
                .with_launch_param(0, "SIMDGROUPS", 4)
                .with_launch_param(0, "ROWS", 1)
                .with_launch_param(0, "LANES", 32),
            BackendName::Cuda => spec
                .with_param("OPERAND", 0)
                .with_launch_param(1, "KSPLIT", 4)
                .with_launch_param(2, "KSPLIT", 4)
                .with_launch_param(4, "ROTATE", 2)
                .with_launch_param(5, "KSPLIT", 4),
            _ => spec,
        };
        let kernel = magnitude_kernels::readout_project_rows::native_for_device_with(
            &device,
            magnitude_kernels::readout_project_rows::Elements {
                A: Element::bf16(),
                OW: Element::bf16(),
            },
            &spec,
        )
        .unwrap();
        read_f32(
            &kernel
                .call(magnitude_kernels::readout_project_rows::Args {
                    features: &features_tensor,
                    weight: &weight_tensor,
                    token_rows: &tokens,
                    weight_scale: &scale_tensor,
                    softcap: cap,
                })
                .unwrap()
                .value,
        )
    };
    for cap in [0.0f32, 0.1, 30.0] {
        let full = project(false, cap);
        let indexed = project(true, cap);
        for row in 0..o {
            for column in 0..s {
                let id = ids[row * s + column];
                let actual = indexed[row * s + column];
                if id < 0 || id as usize >= v {
                    assert_eq!(actual, f32::NEG_INFINITY);
                    continue;
                }
                let sum = (0..d)
                    .map(|k| features[row * d + k] * weights[id as usize * d + k])
                    .sum::<f32>()
                    * -0.75;
                let expected = if cap > 0.0 {
                    cap * (sum / cap).tanh()
                } else {
                    sum
                };
                assert!(
                    (actual - expected).abs() <= 2e-5,
                    "indexed {actual} oracle {expected}"
                );
                assert!(
                    (actual - full[row * v + id as usize]).abs() <= 2e-5,
                    "canonical projection differs between Full and Indexed"
                );
            }
        }
    }
}
