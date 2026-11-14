// `pool_rows` (pooled.seismic) against an independent host model of pooled
// rows: each row's u (LayerNorm-ed per token under a token norm, and gate
// scores) appended to the window, each complete group of RATE tokens pooled
// (softmax-gated, with a per-place gate term or none, or mean), RMS-normed
// unless its tokens were,
// rotated at its first token's coordinates, rounded by the independent
// rounding model (a rounded row from its pooled row, norm and rotation each
// held in the activation element) and stored in the row element. The portable body
// equals the host model, and every backend this host opens equals the
// portable body: the window planes (but a token-normed u, within F32
// arithmetic order), block rows written and origin coordinates exactly, the
// latents within F32 arithmetic order, the rounded
// block rows identical but where an F32 input sits on a rounding boundary.

mod common;
mod rounding_model;
mod versions_common;

use magnitude_kernels::pool_rows;
use rounding_model::{COMPRESSED, NONE, Rounding};
use common::bound::{self, Library};
use seismic::{BackendName, Device, Element, NativeSpecialization};
use versions_common::*;

const EPSILON: f32 = 1e-6;
/// Physical window rows; position p's is (7 p) mod 64, so no two pending
/// tokens are adjacent and every member is a span of its own.
const WINDOW_ROWS: usize = 64;
const WINDOW_SLAB_ROWS: usize = 8;
const BLOCK_ROWS: usize = 256;
const BLOCK_SLAB_ROWS: usize = 32;

/// One launch over the tokens at positions `first..first + rows` of one
/// sequence; the earlier tokens of the first row's group are pending in the
/// window.
#[derive(Clone)]
struct Case {
    /// The activation element `A`, which a rounded row is held in.
    element: Element,
    /// The block plane's row element `Y`.
    row: Element,
    p: usize,
    s: usize,
    adjacent: bool,
    trailing: bool,
    gated: bool,
    /// The per-place gate term `[rate, W]` (`position`), if any.
    places: Vec<Vec<f32>>,
    /// `PoolNorm::Token`'s bias: the tokens are LayerNorm-ed before pooling.
    bias: Option<Vec<f32>>,
    rate: usize,
    rounding: Rounding,
    first: usize,
    rows: usize,
    /// Per position from the first pending one: u, g (gated) and the
    /// coordinates.
    u: Vec<Vec<f32>>,
    g: Vec<Vec<f32>>,
    norm: Vec<f32>,
    components: Vec<i32>,
    frequencies: Vec<f32>,
    amplitudes: Vec<f32>,
}

fn coordinates(position: usize) -> [i32; 4] {
    [position as i32, position as i32 / 2 + 3, position as i32 % 5, 0]
}

fn window_of(position: usize) -> usize {
    position * 7 % WINDOW_ROWS
}

fn block_of(group: usize) -> usize {
    (3 + group * 2) % BLOCK_ROWS
}

impl Case {
    #[allow(clippy::too_many_arguments)]
    fn new(
        element: Element,
        (p, s, adjacent, trailing): (usize, usize, bool, bool),
        gated: bool,
        rate: usize,
        rounding: Rounding,
        first: usize,
        rows: usize,
        seed: u64,
    ) -> Self {
        let w = 2 * p + s;
        let mut random = Random(seed);
        let start = first - first % rate;
        let mut draw = |scale: f32| (0..w).map(|_| random.next() * scale).collect::<Vec<_>>();
        let u = (start..first + rows).map(|_| draw(1.5)).collect();
        let g = (start..first + rows).map(|_| if gated { draw(2.0) } else { Vec::new() }).collect();
        let norm = draw(0.5).iter().map(|x| 1.0 + x).collect();
        Self {
            element,
            row: element,
            places: Vec::new(),
            bias: None,
            p,
            s,
            adjacent,
            trailing,
            gated,
            rate,
            rounding,
            first,
            rows,
            u,
            g,
            norm,
            // Three coordinate axes, a rotary base of 1e4.
            components: (0..p).map(|pair| (pair % 3) as i32).collect(),
            frequencies: (0..p).map(|pair| 1.0e4f64.powf(-((2 * pair) as f64) / (2 * p) as f64) as f32).collect(),
            amplitudes: vec![1.0; p],
        }
    }

    /// This case's rotary table as a YaRN-extended layer derives it: one
    /// position axis, the high frequencies kept, the low ones divided by the
    /// extension factor 16 over a linear ramp, every pair scaled by the
    /// attention factor 0.1 ln 16 + 1.
    fn with_yarn(mut self) -> Self {
        let p = self.p;
        self.components = vec![0; p];
        self.frequencies = (0..p)
            .map(|pair| {
                let base = 1.0e4f64.powf(-((2 * pair) as f64) / (2 * p) as f64);
                let ramp = ((pair as f64 - 4.0) / 16.0).clamp(0.0, 1.0);
                (base * ((1.0 - ramp) + ramp / 16.0)) as f32
            })
            .collect();
        self.amplitudes = vec![(0.1 * 16f64.ln() + 1.0) as f32; p];
        self
    }

    /// GLM-5.3-Flash's indexer keys: each token LayerNorm-ed with a bias,
    /// gated with a per-place term, stored in an F32 key plane.
    fn with_token_keys(mut self, seed: u64) -> Self {
        let mut random = Random(seed);
        let w = self.w();
        self.row = Element::f32();
        self.places = (0..self.rate).map(|_| (0..w).map(|_| random.next() * 2.0).collect()).collect();
        self.bias = Some((0..w).map(|_| random.next() * 0.25).collect());
        self
    }

    fn w(&self) -> usize {
        2 * self.p + self.s
    }
    /// The launch of this case's first `rows` rows alone, over the same
    /// tokens: the accepted prefix of a verify.
    fn prefix(&self, rows: usize) -> Self {
        let kept = self.first - self.start() + rows;
        let mut prefix = self.clone();
        prefix.rows = rows;
        prefix.u.truncate(kept);
        prefix.g.truncate(kept);
        prefix
    }
    fn tn(&self) -> usize {
        usize::from(self.bias.is_some())
    }
    /// A token's u as the window holds it: LayerNorm-ed under a token norm.
    fn token(&self, u: &[f32]) -> Vec<f32> {
        let Some(bias) = &self.bias else { return u.to_vec() };
        let w = u.len() as f32;
        let mean = u.iter().sum::<f32>() / w;
        let variance = u.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / w;
        let inverse = 1.0 / (variance + EPSILON).sqrt();
        u.iter().zip(&self.norm).zip(bias).map(|((x, n), b)| (x - mean) * inverse * n + b).collect()
    }
    /// Whether a value is held in BF16: a rounded row's model holds its
    /// pooled row, norm and rotation in the activation element (DeepSeek
    /// V4.1's BF16), an unrounded row's nothing.
    fn holds(&self) -> bool {
        self.rounding.block != 0 && self.element == Element::bf16()
    }
    fn hold(&self, x: f32) -> f32 {
        if self.holds() { bf16_round(x) } else { x }
    }
    fn gt(&self) -> usize {
        usize::from(self.gated)
    }
    /// The window plane's row width: u, then g.
    fn window_width(&self) -> usize {
        (1 + self.gt()) * self.w()
    }
    fn start(&self) -> usize {
        self.first - self.first % self.rate
    }
    fn position(&self, row: usize) -> usize {
        self.first + row
    }
    fn completes(&self, position: usize) -> bool {
        (position + 1) % self.rate == 0
    }
    /// The launch's spans per row: a completing row's earlier members.
    fn spans(&self) -> usize {
        self.rate - 1
    }

    /// Per row: window destination (a pending token's), window spans (its
    /// group's earlier members, one span each), block destination, its
    /// coordinates.
    fn controls(&self) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut window_destinations = Vec::new();
        let mut window_visible = Vec::new();
        let mut block_destinations = Vec::new();
        let mut row_coordinates = Vec::new();
        for row in 0..self.rows {
            let position = self.position(row);
            let complete = self.completes(position);
            window_destinations.push(if complete { -1 } else { window_of(position) as i32 });
            block_destinations.push(if complete { block_of(position / self.rate) as i32 } else { -1 });
            let members = position - position % self.rate..position;
            let spans = members.map(|member| [window_of(member) as i32, window_of(member) as i32 + 1]);
            window_visible.extend(spans.chain(std::iter::repeat([0, 0])).take(self.spans()).flatten());
            row_coordinates.extend(coordinates(position));
        }
        (window_destinations, window_visible, block_destinations, row_coordinates)
    }

    /// The window planes before the launch: the pending tokens before it.
    fn window_planes(&self) -> (Vec<f32>, Vec<i32>) {
        let width = self.window_width();
        let mut rows = vec![SENTINEL; WINDOW_ROWS * width];
        let mut coordinate_rows = vec![-7; WINDOW_ROWS * 4];
        for position in self.start()..self.first {
            self.write_window(&mut rows, &mut coordinate_rows, position);
        }
        (rows, coordinate_rows)
    }

    fn write_window(&self, rows: &mut [f32], coordinate_rows: &mut [i32], position: usize) {
        let (width, at, index) = (self.window_width(), window_of(position), position - self.start());
        rows[at * width..][..self.w()].copy_from_slice(&self.token(&self.u[index]));
        rows[at * width + self.w()..][..self.gt() * self.w()].copy_from_slice(&self.g[index]);
        coordinate_rows[at * 4..][..4].copy_from_slice(&coordinates(position));
    }

    /// The case's rotary layout: pair `pair`'s (lo, hi) columns.
    fn columns(&self, pair: usize) -> (usize, usize) {
        let first = if self.trailing { self.s } else { 0 };
        if self.adjacent {
            (first + 2 * pair, first + 2 * pair + 1)
        } else {
            (first + pair, first + pair + self.p)
        }
    }
}

/// The outcome of a launch: its three results and the planes it wrote.
#[derive(Debug)]
struct Outcome {
    latent: Vec<f32>,
    latent_rows: Vec<i32>,
    latent_coordinates: Vec<i32>,
    window: Vec<f32>,
    window_coordinates: Vec<i32>,
    blocks: Vec<f32>,
}

/// The host model, in F32 in the portable body's order.
fn model(case: &Case) -> Outcome {
    let w = case.w();
    let (mut window, mut window_coordinates) = case.window_planes();
    let mut blocks = vec![SENTINEL; BLOCK_ROWS * w];
    let mut latent = vec![0.0; case.rows * w];
    let mut latent_rows = vec![-1; case.rows];
    let mut latent_coordinates = vec![0; case.rows * 4];
    for row in 0..case.rows {
        let position = case.position(row);
        if !case.completes(position) {
            case.write_window(&mut window, &mut window_coordinates, position);
            continue;
        }
        let group = position - position % case.rate;
        let members = (group..=position).map(|member| member - case.start()).collect::<Vec<_>>();
        let u = members.iter().map(|&m| case.token(&case.u[m])).collect::<Vec<_>>();
        // Member i's gate scores plus its place's term.
        let gate = |i: usize, c: usize| case.g[members[i]][c] + case.places.get(i).map_or(0.0, |place| place[c]);
        let pooled = (0..w)
            .map(|c| {
                if case.gated {
                    let top = (0..case.rate).fold(f32::NEG_INFINITY, |top, i| top.max(gate(i, c)));
                    let total = (0..case.rate).fold(0.0f32, |total, i| total + (gate(i, c) - top).exp());
                    (0..case.rate).fold(0.0f32, |sum, i| sum + (gate(i, c) - top).exp() / total * u[i][c])
                } else {
                    u.iter().fold(0.0f32, |sum, u| sum + u[c]) / case.rate as f32
                }
            })
            .map(|x| case.hold(x))
            .collect::<Vec<_>>();
        let normed = if case.bias.is_some() {
            pooled
        } else {
            let squares = pooled.iter().fold(0.0f32, |sum, x| sum + x * x);
            let inverse = 1.0 / (squares / w as f32 + EPSILON).sqrt();
            pooled.iter().zip(&case.norm).map(|(x, n)| case.hold(x * inverse * n)).collect::<Vec<_>>()
        };
        let origin = coordinates(group);
        let mut rotated = normed.clone();
        for pair in 0..case.p {
            let (lo, hi) = case.columns(pair);
            let angle = origin[case.components[pair] as usize] as f32 * case.frequencies[pair];
            let (c, s) = (angle.cos() * case.amplitudes[pair], angle.sin() * case.amplitudes[pair]);
            rotated[lo] = normed[lo] * c - normed[hi] * s;
            rotated[hi] = normed[hi] * c + normed[lo] * s;
        }
        let rotated = rotated.into_iter().map(|x| case.hold(x)).collect::<Vec<_>>();
        let destination = block_of(position / case.rate);
        for (target, value) in blocks[destination * w..][..w].iter_mut().zip(rounding_model::round(case.rounding, &rotated)) {
            *target = if case.row == Element::bf16() { bf16_round(value) } else { value };
        }
        latent[row * w..][..w].copy_from_slice(&normed);
        latent_rows[row] = destination as i32;
        latent_coordinates[row * 4..][..4].copy_from_slice(&origin);
    }
    Outcome { latent, latent_rows, latent_coordinates, window, window_coordinates, blocks }
}

fn portable(case: &Case) -> Outcome {
    use seismic_lang::{
        checked::SourceFile,
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let dtype = |element: Element| if element == Element::bf16() { DType::BF16 } else { DType::F32 };
    let (element, row) = (dtype(case.element), dtype(case.row));
    let mut sources = seismic_std::sources();
    for (path, text) in [
        ("pooled.seismic", include_str!("../kernels/pooled.seismic")),
        ("attention.seismic", include_str!("../kernels/attention.seismic")),
        ("functions.seismic", include_str!("../kernels/functions.seismic")),
        ("rounding.seismic", include_str!("../kernels/rounding.seismic")),
    ] {
        sources.push(SourceFile { path: path.into(), text: text.into() });
    }
    let module = crate::common::check(sources);
    let logical = module
        .entry(
            module.entry_named("pool_rows").unwrap(),
            &ElementBindings::new().bind("A", registry::dense(element)).bind("Y", registry::dense(row)),
        )
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let (m, w, gt) = (case.rows, case.w(), case.gt());
    let (window_destinations, window_visible, block_destinations, row_coordinates) = case.controls();
    let (window, window_coordinates) = case.window_planes();
    let launch = |values: &[Vec<f32>]| {
        values[case.first - case.start()..].iter().flatten().map(|x| f64::from(*x)).collect::<Vec<_>>()
    };
    let mut tensor = |dtype, shape: Vec<usize>, values: Vec<f64>| {
        Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
    };
    let floats = |values: &[f32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let ints = |values: &[i32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let args = vec![
        tensor(DType::F32, vec![m, w], launch(&case.u)),
        tensor(DType::F32, vec![m, gt * w], launch(&case.g)),
        tensor(DType::F32, vec![case.places.len(), w], floats(&case.places.concat())),
        tensor(DType::F32, vec![1, w], floats(&case.norm)),
        tensor(DType::F32, vec![case.tn(), w], floats(case.bias.as_deref().unwrap_or(&[]))),
        tensor(DType::I32, vec![case.p], ints(&case.components)),
        tensor(DType::F32, vec![case.p], floats(&case.frequencies)),
        tensor(DType::F32, vec![case.p], floats(&case.amplitudes)),
        tensor(DType::I32, vec![m, 4], ints(&row_coordinates)),
        tensor(DType::I32, vec![m, case.spans(), 2], ints(&window_visible)),
        tensor(DType::I32, vec![m], ints(&window_destinations)),
        tensor(DType::I32, vec![m], ints(&block_destinations)),
        tensor(DType::F32, vec![WINDOW_ROWS, case.window_width()], floats(&window)),
        tensor(DType::I32, vec![WINDOW_ROWS, 4], ints(&window_coordinates)),
        tensor(row, vec![BLOCK_ROWS, w], vec![f64::from(SENTINEL); BLOCK_ROWS * w]),
        tensor(DType::I32, vec![0, usize::from(case.adjacent), usize::from(case.trailing), case.rate, gt], Vec::new()),
        Arg::Scalar(ReferenceScalar::F32(EPSILON.to_bits())),
        Arg::Scalar(ReferenceScalar::U32(WINDOW_SLAB_ROWS as u32)),
        Arg::Scalar(ReferenceScalar::U32(BLOCK_SLAB_ROWS as u32)),
        Arg::Scalar(ReferenceScalar::I32(case.rounding.block)),
        Arg::Scalar(ReferenceScalar::I32(case.rounding.element)),
        Arg::Scalar(ReferenceScalar::I32(case.rounding.scale)),
        Arg::Scalar(ReferenceScalar::F32(case.rounding.amax_floor.to_bits())),
    ];
    let outcome = interpreter.run_bounded(&args, u64::MAX).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable pool_rows failed: {failure}");
    }
    let values = |value: OutcomeValue<'_>| match value {
        OutcomeValue::Tensor(reader) => (0..reader.element_count()).map(|i| reader.read(i).unwrap()).collect::<Vec<_>>(),
        _ => panic!("a tensor result"),
    };
    let mut results = outcome.results();
    let latent = values(results.next().unwrap().value()).into_iter().map(|v| v as f32).collect();
    let latent_rows = values(results.next().unwrap().value()).into_iter().map(|v| v as i32).collect();
    let latent_coordinates = values(results.next().unwrap().value()).into_iter().map(|v| v as i32).collect();
    let plane = |ordinal: usize| {
        let input = outcome.inputs().find(|input| input.ordinal() == ordinal).unwrap();
        let reader = input.tensor();
        (0..reader.element_count()).map(|i| reader.read(i).unwrap()).collect::<Vec<_>>()
    };
    Outcome {
        latent,
        latent_rows,
        latent_coordinates,
        window: plane(12).into_iter().map(|v| v as f32).collect(),
        window_coordinates: plane(13).into_iter().map(|v| v as i32).collect(),
        blocks: plane(14).into_iter().map(|v| v as f32).collect(),
    }
}

/// `case`'s `pool_rows` static values.
fn statics(case: &Case) -> NativeSpecialization {
    [
        ("P", case.p as u64),
        ("S", case.s as u64),
        ("GT", case.gt() as u64),
        ("PR", case.places.len() as u64),
        ("TN", case.tn() as u64),
        ("RA", u64::from(case.adjacent)),
        ("RT", u64::from(case.trailing)),
        ("RATE", case.rate as u64),
    ]
    .iter()
    .fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value))
}

fn native(device: &Device, case: &Case) -> Outcome {
    let (m, w, gt) = (case.rows as u64, case.w() as u64, case.gt() as u64);
    let entry = pool_rows::native_entry_with(pool_rows::Elements { A: case.element, Y: case.row });
    let specialization = entry.offered(device).unwrap().unwrap().default_specialization(&statics(case)).unwrap();
    let kernel = entry.prepare(device, &specialization).unwrap();
    let (window_destinations, window_visible, block_destinations, row_coordinates) = case.controls();
    let (window, window_coordinates) = case.window_planes();
    let plane = |rows: usize, slab_rows: usize, element: Element, width: u64, values: &[f32]| {
        let (slabs, logical) = slab_arenas(
            device,
            rows,
            slab_rows as u64,
            &[Arena { element, bank_shape: vec![width], values }],
            false,
        );
        (slabs, logical.into_iter().next().unwrap())
    };
    let (_window_slabs, mut window_plane) =
        plane(WINDOW_ROWS, WINDOW_SLAB_ROWS, Element::f32(), case.window_width() as u64, &window);
    // 32-bit words travel as the bits of f32 values (`bytes`).
    let coordinate_words = window_coordinates.iter().map(|value| f32::from_bits(*value as u32)).collect::<Vec<_>>();
    let (_coordinate_slabs, mut coordinate_plane) =
        plane(WINDOW_ROWS, WINDOW_SLAB_ROWS, Element::i32(), 4, &coordinate_words);
    let (_block_slabs, mut block_plane) =
        plane(BLOCK_ROWS, BLOCK_SLAB_ROWS, case.row, w, &vec![SENTINEL; BLOCK_ROWS * case.w()]);
    let launch = |values: &[Vec<f32>]| values[case.first - case.start()..].concat();
    let results = kernel
        .call(pool_rows::Args {
            u: &tensor(device, Element::f32(), &[m, w], &launch(&case.u)),
            g: &tensor(device, Element::f32(), &[m, gt * w], &launch(&case.g)),
            position: &tensor(device, Element::f32(), &[case.places.len() as u64, w], &case.places.concat()),
            norm: &tensor(device, Element::f32(), &[1, w], &case.norm),
            bias: &tensor(device, Element::f32(), &[case.tn() as u64, w], case.bias.as_deref().unwrap_or(&[])),
            rotary_components: &ints(device, &[case.p as u64], &case.components),
            rotary_frequencies: &tensor(device, Element::f32(), &[case.p as u64], &case.frequencies),
            rotary_amplitudes: &tensor(device, Element::f32(), &[case.p as u64], &case.amplitudes),
            coordinates: &ints(device, &[m, 4], &row_coordinates),
            window_visible: &ints(device, &[m, case.spans() as u64, 2], &window_visible),
            window_destinations: &ints(device, &[m], &window_destinations),
            block_destinations: &ints(device, &[m], &block_destinations),
            window_rows: &mut window_plane,
            window_coordinates: &mut coordinate_plane,
            block_rows: &mut block_plane,
            form: &ints(device, &[0, u64::from(case.adjacent), u64::from(case.trailing), case.rate as u64, gt], &[]),
            epsilon: EPSILON,
            window_slab_rows: WINDOW_SLAB_ROWS as u32,
            block_slab_rows: BLOCK_SLAB_ROWS as u32,
            block: case.rounding.block,
            element: case.rounding.element,
            scale: case.rounding.scale,
            amax_floor: case.rounding.amax_floor,
        })
        .unwrap();
    let words = |values: Vec<f32>| values.into_iter().map(|value| value.to_bits() as i32).collect::<Vec<_>>();
    Outcome {
        latent: read(&results.r0),
        latent_rows: words(read(&results.r1)),
        latent_coordinates: words(read(&results.r2)),
        window: read(&window_plane),
        window_coordinates: words(read(&coordinate_plane)),
        blocks: read(&block_plane),
    }
}

/// `actual` agrees with `expected` for `case`: the window planes, written
/// block rows and origins exactly; the latents within F32 arithmetic order.
/// A rounded block plane is identical but for one value in 512 and one
/// rounding run, each within one code of its element (an F32 input on a
/// rounding boundary). An unrounded one is the rotated latent in its
/// element, so it agrees as the latents do (the backends' rotary sines
/// differ in their last bits), within one code in BF16.
fn agree(label: &str, case: &Case, backend: BackendName, actual: &Outcome, expected: &Outcome) {
    if let Some(bias) = &case.bias {
        // Under a token norm the window holds each token's computed
        // LayerNorm, within the LayerNorm bound of `backend`'s library and
        // the host's; its gate scores and unwritten rows are copies, held
        // exactly.
        assert_eq!(actual.window.len(), expected.window.len(), "{label}: window plane");
        let libraries = (&Library::of(backend), &Library::of(BackendName::Cpu));
        let mut bounds = vec![0.0; expected.window.len()];
        for (index, u) in case.u.iter().enumerate() {
            let at = window_of(case.start() + index) * case.window_width();
            bounds[at..][..case.w()]
                .copy_from_slice(&bound::layer_norm_between(libraries, u, &case.norm, bias, EPSILON));
        }
        for (index, ((a, e), bound)) in actual.window.iter().zip(&expected.window).zip(&bounds).enumerate() {
            let held = a.to_bits() == e.to_bits() || (f64::from(*a) - f64::from(*e)).abs() <= *bound;
            assert!(held, "{label}: window[{index}] {a} vs {e}, bound {bound}");
        }
    } else {
        assert!(same_bits(&actual.window, &expected.window), "{label}: window plane");
    }
    assert_eq!(actual.window_coordinates, expected.window_coordinates, "{label}: window coordinates");
    assert_eq!(actual.latent_rows, expected.latent_rows, "{label}: latent rows");
    assert_eq!(actual.latent_coordinates, expected.latent_coordinates, "{label}: latent coordinates");
    // A held latent may sit one BF16 code (2^-7 |x|) apart where its F32
    // value straddles a BF16 rounding boundary.
    let held = if case.holds() { 2f32.powi(-7) } else { 0.0 };
    for (index, (a, e)) in actual.latent.iter().zip(&expected.latent).enumerate() {
        assert!((a - e).abs() <= 1e-6 + (1e-5 + held) * e.abs(), "{label}: latent[{index}] {a} vs {e}");
    }
    assert_eq!(actual.blocks.len(), expected.blocks.len(), "{label}: block plane");
    if case.rounding.block == 0 {
        // A stored value is a latent pair (v, w) turned, v c - w s: with v, w
        // within the latent bar b(x) = 1e-6 + 1e-5 |x| and c, s within the
        // fast sine's 2^-13 absolute error on the reduced angle (Metal Shading
        // Language, fast-math sin/cos on [-pi, pi]), it is within
        // 2 b(m) + 2^-12 m of the oracle's, m the row's largest |value|; a BF16
        // plane adds one code (2^-7 |x|) of its publication.
        let w = case.w();
        let publication = if case.row == Element::f32() { 0.0 } else { 2f32.powi(-7) };
        for (row, (a, e)) in actual.blocks.chunks(w).zip(expected.blocks.chunks(w)).enumerate() {
            let m = e.iter().fold(0f32, |m, x| m.max(x.abs()));
            let bound = 2.0 * (1e-6 + 1e-5 * m) + 2f32.powi(-12) * m;
            for (column, (a, e)) in a.iter().zip(e).enumerate() {
                let limit = bound + publication * e.abs();
                assert!((a - e).abs() <= limit, "{label}: block [{row}, {column}] {a} vs {e}");
            }
        }
        return;
    }
    let differing = actual
        .blocks
        .iter()
        .zip(&expected.blocks)
        .filter(|(a, e)| a.to_bits() != e.to_bits())
        .inspect(|(a, e)| assert!((*a - *e).abs() <= a.abs().max(e.abs()), "{label}: block value {a} vs {e}"))
        .count();
    assert!(differing <= actual.blocks.len() / 512 + 16, "{label}: {differing} block values differ");
}

fn check(case: &Case, label: &str) {
    let expected = model(case);
    let oracle = portable(case);
    agree(&format!("{label} portable"), case, BackendName::Cpu, &oracle, &expected);
    for device in offering(&pool_rows::native_entry_with(pool_rows::Elements { A: case.element, Y: case.row })) {
        let backend = device.backend();
        agree(&format!("{label} {}", backend.as_str()), case, backend, &native(&device, case), &oracle);
    }
}

/// DeepSeek V4.1's compressed rows: W = 512, its 32 rotary pairs adjacent
/// over the trailing 64 columns, YaRN-extended.
const V41: (usize, usize, bool, bool) = (32, 448, true, true);
/// Qwen4exp's indexer blocks: W = 128, 32 half-split pairs over the leading
/// columns, multi-axis coordinates.
const QWEN4: (usize, usize, bool, bool) = (32, 64, false, false);

#[test]
fn softmax_gated_rows_at_v41_shape() {
    // Rate 2, E2M1/16/E4M3 rounding: nine rows from an odd position, so the
    // first row completes a group its previous launch began.
    check(&Case::new(Element::bf16(), V41, true, 2, COMPRESSED, 101, 9, 1).with_yarn(), "v41 gated");
    // One row: a decode step completing its group.
    check(&Case::new(Element::bf16(), V41, true, 2, COMPRESSED, 7, 1, 2).with_yarn(), "v41 gated one row");
}

#[test]
fn mean_rows_at_qwen4_shape() {
    // Qwen4exp's mean of four raw keys, unrounded, in an F32 block plane:
    // sixteen rows from a group's third token, finishing four groups.
    check(&Case::new(Element::f32(), QWEN4, false, 4, NONE, 42, 16, 4), "qwen4 mean");
    // In the activation element.
    check(&Case::new(Element::bf16(), QWEN4, false, 4, NONE, 42, 16, 5), "qwen4 mean bf16");
}

#[test]
fn groups_straddle_launches() {
    // A group's tokens over three launches: the first two rows only append,
    // then one row completes the group from three pending tokens.
    check(&Case::new(Element::f32(), QWEN4, false, 4, NONE, 45, 2, 6), "qwen4 pending rows");
    check(&Case::new(Element::f32(), QWEN4, false, 4, NONE, 47, 1, 7), "qwen4 one completing row");
}

/// GLM-5.3-Flash's indexer keys: W = 128 without a rotary, groups of four
/// tokens, each LayerNorm-ed with a bias, softmax-gated with a per-place
/// term, unrounded, in an F32 key plane beside a BF16 activation.
const GLM: (usize, usize, bool, bool) = (0, 128, false, false);

#[test]
fn token_normed_keys_gated_by_place_at_glm_shape() {
    check(&Case::new(Element::bf16(), GLM, true, 4, NONE, 42, 16, 8).with_token_keys(9), "glm keys");
    // A decode step completing its group from three pending tokens.
    check(&Case::new(Element::bf16(), GLM, true, 4, NONE, 47, 1, 10).with_token_keys(11), "glm keys one row");
}

/// A verify that accepts a prefix of its rows leaves, bit for bit on every
/// backend this host opens, the state a launch of that prefix alone leaves:
/// each accepted row's latent and origin, each group it completes' block
/// row, and each pending token's window row and coordinates. The rows after
/// it write only the window rows of their own tokens and the block rows of
/// the groups they complete, which the state releases with them; so the
/// next advance from the accepted prefix reads what a run that stopped
/// there would have, and rolling back a rejected verify is exact. Each key
/// form: V4.1's rounded gated rows, Qwen4exp's mean keys and GLM-5.3's
/// token-normed keys gated by place, each over launches whose prefixes end
/// inside a group and on its last token.
#[test]
fn an_accepted_prefix_leaves_the_state_its_own_launch_does() {
    let cases = [
        (Case::new(Element::bf16(), V41, true, 2, COMPRESSED, 101, 9, 1).with_yarn(), "v41 gated"),
        (Case::new(Element::f32(), QWEN4, false, 4, NONE, 42, 16, 4), "qwen4 mean"),
        (Case::new(Element::bf16(), GLM, true, 4, NONE, 42, 16, 8).with_token_keys(9), "glm keys"),
    ];
    let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    for (case, label) in &cases {
        let (w, width) = (case.w(), case.window_width());
        for device in offering(&pool_rows::native_entry_with(pool_rows::Elements { A: case.element, Y: case.row })) {
            let whole = native(&device, case);
            for rows in 1..case.rows {
                let prefix = native(&device, &case.prefix(rows));
                for row in 0..rows {
                    let position = case.position(row);
                    let label = format!("{label} {} prefix of {rows} at {position}", device.backend().as_str());
                    assert_eq!(bits(&prefix.latent[row * w..][..w]), bits(&whole.latent[row * w..][..w]), "{label}: latent");
                    assert_eq!(prefix.latent_rows[row], whole.latent_rows[row], "{label}: latent row");
                    assert_eq!(prefix.latent_coordinates[row * 4..][..4], whole.latent_coordinates[row * 4..][..4], "{label}: origin");
                    if case.completes(position) {
                        let at = block_of(position / case.rate) * w;
                        assert_eq!(bits(&prefix.blocks[at..][..w]), bits(&whole.blocks[at..][..w]), "{label}: block row");
                    } else {
                        let at = window_of(position);
                        assert_eq!(bits(&prefix.window[at * width..][..width]), bits(&whole.window[at * width..][..width]), "{label}: window row");
                        assert_eq!(prefix.window_coordinates[at * 4..][..4], whole.window_coordinates[at * 4..][..4], "{label}: window coordinates");
                    }
                }
            }
        }
    }
}
