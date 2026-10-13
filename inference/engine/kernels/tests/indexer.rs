// `indexer_select` against an independent host model of selected attention
// (raw keys appended, blocks finished and rotated at their first row, scores,
// the K best blocks with ties to the lower block, then the Segments and
// WorkList read forms), and every backend this host opens against the
// portable body: the read forms equal bit for bit, the appended raw keys bit
// for bit and the finished F32 block keys within F32 arithmetic order. The
// indexer reads its projections in the activation element and computes the
// rest in F32, as the reference does: a case whose near-tie scores an
// indexer rounding to the activation would order differently selects the
// F32 reference's blocks.

mod versions_common;

use magnitude_kernels::indexer_select;
use seismic::{Device, Element, NativeSpecialization, Tensor};
use versions_common::*;

const EPSILON: f32 = 1e-6;
const H: usize = 4;
const P: usize = 32;
const S: usize = 64;
const W: usize = 2 * P + S;
const SB: usize = 4;
/// Physical rows of a token-history slab: spans break at its edges.
const SLAB_ROWS: usize = 64;
const WINDOW_ROWS: usize = 64;
const BLOCK_ROWS: usize = 32;

/// One launch over rows at positions `first..first + rows` of one sequence.
struct Case {
    /// The indexer's element `A`: its projected query and raw key inputs.
    element: Element,
    rows: usize,
    first: usize,
    /// Selected blocks per row.
    k: usize,
    /// Block capacity (scores and bitmap extent).
    nb: usize,
    /// Segments per row (decode) or bitmap words per row (prefill).
    segments: usize,
    words: usize,
    query: Vec<f32>,
    key: Vec<f32>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    components: Vec<i32>,
    frequencies: Vec<f32>,
    /// Physical token-history row of each history position.
    history_rows: Vec<i32>,
    /// Window rows of every position before the launch (only the last
    /// SB - 1 are read) and of the launch's rows.
    window_of: Vec<i32>,
    window: Vec<f32>,
    /// Block row of every block finished before the launch, and of the
    /// launch's own finished blocks.
    block_of: Vec<i32>,
    blocks: Vec<f32>,
}

fn coordinates(position: usize) -> [i32; 4] {
    // Distinct multi-axis coordinates, as an image row's are.
    [position as i32, position as i32 / 2 + 3, position as i32 % 7, 0]
}

/// A value as an `element` plane or tensor holds it.
fn round(element: Element, value: f32) -> f32 {
    if element == Element::bf16() { bf16_round(value) } else { value }
}

impl Case {
    fn new(rows: usize, first: usize, k: usize, nb: usize, decode: bool, seed: u64) -> Self {
        Self::of(Element::bf16(), rows, first, k, nb, decode, seed)
    }

    fn of(element: Element, rows: usize, first: usize, k: usize, nb: usize, decode: bool, seed: u64) -> Self {
        let mut random = Random(seed);
        let mut draw = |count: usize, scale: f32| {
            (0..count).map(|_| round(element, random.next() * scale)).collect::<Vec<_>>()
        };
        let query = draw(rows * H * W, 2.0);
        let key = draw(rows * W, 1.5);
        let query_norm = draw(W, 1.0).iter().map(|x| 1.0 + 0.5 * x).collect();
        let key_norm = draw(W, 1.0).iter().map(|x| 1.0 + 0.5 * x).collect();
        let components = (0..P).map(|pair| [0, 1, 2][pair % 3]).collect();
        let frequencies = (0..P).map(|pair| 10_000f32.powf(-(2.0 * pair as f32) / (2.0 * P as f32))).collect();
        // History positions in two physical runs, the first crossing a slab
        // edge.
        let history_rows = (0..first)
            .map(|position| if position < 20 { 50 + position as i32 } else { 200 + position as i32 })
            .collect();
        // Window rows: every earlier position, then the launch's rows.
        let window_of = (0..first + rows).map(|position| ((position * 7) % 40) as i32 + if position >= first { 24 } else { 0 }).collect::<Vec<_>>();
        let mut window = vec![SENTINEL; WINDOW_ROWS * W];
        for position in first.saturating_sub(SB - 1)..first {
            let at = window_of[position] as usize;
            window[at * W..][..W].copy_from_slice(&draw(W, 1.5));
        }
        let finished = (first + rows) / SB;
        let block_of = (0..finished)
            .map(|block| if block < 5 { block as i32 } else { 10 + block as i32 })
            .collect::<Vec<_>>();
        let mut blocks = vec![SENTINEL; BLOCK_ROWS * W];
        for block in 0..first / SB {
            let at = block_of[block] as usize;
            blocks[at * W..][..W].copy_from_slice(&draw(W, 1.0));
        }
        // Every class that scores has the bitmap, over 32 blocks a word.
        let words = (nb + 1).div_ceil(32);
        let nb = 32 * words;
        let segments = if decode { k + 1 + 2 + rows + 4 } else { 0 };
        Self {
            element,
            rows,
            first,
            k,
            nb,
            segments,
            words,
            query,
            key,
            query_norm,
            key_norm,
            components,
            frequencies,
            history_rows,
            window_of,
            window,
            block_of,
            blocks,
        }
    }

    fn position(&self, row: usize) -> usize {
        self.first + row
    }

    /// The row's token spans (history in physical runs) and fresh rows.
    fn token_runs(&self) -> Vec<[i32; 2]> {
        let mut runs: Vec<[i32; 2]> = Vec::new();
        for &at in &self.history_rows {
            match runs.last_mut() {
                Some(run) if run[1] == at && at as usize % SLAB_ROWS != 0 => run[1] += 1,
                _ => runs.push([at, at + 1]),
            }
        }
        runs
    }

    /// The launch's spans per row `R`: the most any of its reads (token,
    /// window, block) has, as one class pads them all.
    fn spans(&self) -> usize {
        let controls = self.read_runs();
        [self.token_runs().len()]
            .into_iter()
            .chain(controls.0.iter().chain(&controls.1).map(Vec::len))
            .max()
            .unwrap_or(1)
            .max(1)
    }

    fn token_spans(&self) -> (Vec<i32>, Vec<i32>, usize) {
        let runs = self.token_runs();
        let spans = self.spans();
        let mut visible = Vec::new();
        let mut fresh = Vec::new();
        for row in 0..self.rows {
            for index in 0..spans {
                visible.extend(runs.get(index).copied().unwrap_or([0, 0]));
            }
            fresh.extend([0, row as i32 + 1]);
        }
        (visible, fresh, spans)
    }

    /// The window rows of every position before the launch, with their
    /// coordinates, as the window planes hold them.
    fn window_planes(&self) -> Vec<i32> {
        let mut plane = vec![-7; WINDOW_ROWS * 4];
        for position in self.first.saturating_sub(SB - 1)..self.first {
            let at = self.window_of[position] as usize;
            plane[at * 4..][..4].copy_from_slice(&coordinates(position));
        }
        plane
    }

    /// Per-row controls: the window destinations and spans (the window rows
    /// of the SB - 1 positions before each row, in runs), the block spans
    /// (the blocks complete before each row, in runs), the block
    /// destinations and the rows' coordinates.
    /// Per row, its window runs and its block runs.
    fn read_runs(&self) -> (Vec<Vec<[i32; 2]>>, Vec<Vec<[i32; 2]>>) {
        let runs = |rows: &[i32]| {
            let mut runs: Vec<[i32; 2]> = Vec::new();
            for &at in rows {
                match runs.last_mut() {
                    Some(run) if run[1] == at => run[1] += 1,
                    _ => runs.push([at, at + 1]),
                }
            }
            runs
        };
        let positions = (0..self.rows).map(|row| self.position(row));
        (
            positions
                .clone()
                .map(|position| runs(&self.window_of[position.saturating_sub(SB - 1)..position]))
                .collect(),
            positions.map(|position| runs(&self.block_of[..position / SB])).collect(),
        )
    }

    fn controls(&self) -> Controls {
        let spans = self.spans();
        let pad = |per_row: &[Vec<[i32; 2]>]| {
            per_row
                .iter()
                .flat_map(|runs| (0..spans).flat_map(|index| runs.get(index).copied().unwrap_or([0, 0])))
                .collect::<Vec<_>>()
        };
        let (window_runs, block_runs) = self.read_runs();
        let positions = (0..self.rows).map(|row| self.position(row)).collect::<Vec<_>>();
        Controls {
            window_destinations: positions.iter().map(|&position| self.window_of[position]).collect(),
            window_visible: pad(&window_runs),
            block_visible: pad(&block_runs),
            spans,
            block_destinations: positions
                .iter()
                .map(|&position| if (position + 1) % SB == 0 { self.block_of[position / SB] } else { -1 })
                .collect(),
            coordinates: positions.iter().flat_map(|&position| coordinates(position)).collect(),
        }
    }
}

struct Controls {
    window_destinations: Vec<i32>,
    window_visible: Vec<i32>,
    block_visible: Vec<i32>,
    /// `R`: every read's spans per row.
    spans: usize,
    block_destinations: Vec<i32>,
    coordinates: Vec<i32>,
}

// ---- host model ---------------------------------------------------------------------------------

/// The portable `head_rotary`, its result rounded to `element`.
fn head_rotary(x: &[f32], norm: &[f32], coordinates: [i32; 4], case: &Case, element: Element) -> Vec<f32> {
    let squares: f32 = x.iter().map(|v| v * v).sum();
    let inverse = 1.0 / (squares / W as f32 + EPSILON).sqrt();
    let normed = x.iter().zip(norm).map(|(v, w)| v * inverse * w).collect::<Vec<_>>();
    let mut rotated = normed.clone();
    for pair in 0..P {
        let angle = coordinates[case.components[pair] as usize] as f32 * case.frequencies[pair];
        let (s, c) = angle.sin_cos();
        rotated[pair] = normed[pair] * c - normed[pair + P] * s;
        rotated[pair + P] = normed[pair + P] * c + normed[pair] * s;
    }
    rotated.into_iter().map(|value| round(element, value)).collect()
}

struct Model {
    /// Finished block keys (block index -> key).
    block_keys: Vec<(usize, Vec<f32>)>,
    /// Per row, the blocks it reads and its complete blocks' scores.
    reads: Vec<Vec<bool>>,
    scores: Vec<Vec<f32>>,
    segments: Vec<i32>,
    bitmap: Vec<u32>,
}

/// The host model of `indexer_select`: F32 from its inputs on.
fn model(case: &Case) -> Model {
    model_rounding(case, Element::f32())
}

/// The host model rounding its prepared queries, block means and block keys
/// to `element` (F32: none).
fn model_rounding(case: &Case, element: Element) -> Model {
    // Raw keys by position.
    let mut raw = vec![Vec::new(); case.first + case.rows];
    for position in case.first.saturating_sub(SB - 1)..case.first {
        let at = case.window_of[position] as usize;
        raw[position] = case.window[at * W..][..W].to_vec();
    }
    for row in 0..case.rows {
        raw[case.position(row)] = case.key[row * W..][..W].to_vec();
    }
    let mut keys = (0..case.first / SB)
        .map(|block| {
            let at = case.block_of[block] as usize;
            case.blocks[at * W..][..W].to_vec()
        })
        .collect::<Vec<_>>();
    let mut block_keys = Vec::new();
    for row in 0..case.rows {
        let position = case.position(row);
        if (position + 1) % SB != 0 {
            continue;
        }
        let start = position / SB * SB;
        let mean = (0..W)
            .map(|c| round(element, (start..start + SB).map(|m| raw[m][c]).sum::<f32>() / SB as f32))
            .collect::<Vec<_>>();
        let finished = head_rotary(&mean, &case.key_norm, coordinates(start), case, element);
        keys.push(finished.clone());
        block_keys.push((position / SB, finished));
    }
    let mut reads = Vec::new();
    let mut all_scores = Vec::new();
    let (visible, _, spans) = case.token_spans();
    let mut segments = vec![0; case.rows * case.segments * 2];
    let mut bitmap = vec![0u32; case.rows * case.words];
    for row in 0..case.rows {
        let position = case.position(row);
        let complete = (position + 1) / SB;
        let q = (0..H)
            .map(|h| {
                head_rotary(&case.query[(row * H + h) * W..][..W], &case.query_norm, coordinates(position), case, element)
            })
            .collect::<Vec<_>>();
        let scores = (0..complete)
            .map(|block| {
                (0..H)
                    .map(|h| q[h].iter().zip(&keys[block]).map(|(a, b)| a * b).sum::<f32>().max(0.0))
                    .sum::<f32>()
                    / (W as f32).sqrt()
            })
            .collect::<Vec<_>>();
        let mut read = vec![true; case.nb + 1];
        if complete > case.k {
            let mut order = (0..complete).collect::<Vec<_>>();
            order.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap().then(a.cmp(b)));
            read[..complete].fill(false);
            for &block in &order[..case.k] {
                read[block] = true;
            }
        }
        for word in 0..case.words {
            bitmap[row * case.words + word] =
                (0..32).filter(|bit| read[(word * 32 + bit).min(case.nb)]).fold(0, |w, bit| w | 1 << bit);
        }
        if case.segments > 0 {
            // The keys of the visible spans, in position order (attention
            // reads the fresh rows itself).
            let mut keys_of = Vec::new();
            for index in 0..spans {
                let [lo, hi] = [visible[(row * spans + index) * 2], visible[(row * spans + index) * 2 + 1]];
                keys_of.extend(lo..hi);
            }
            let mut runs: Vec<[i32; 2]> = Vec::new();
            let mut open = false;
            for (key, stored) in keys_of.into_iter().enumerate() {
                if !read[(key / SB).min(case.nb)] {
                    open = false;
                    continue;
                }
                match runs.last_mut() {
                    Some(run) if open && run[1] == stored && stored as usize % SLAB_ROWS != 0 => run[1] += 1,
                    _ => runs.push([stored, stored + 1]),
                }
                open = true;
            }
            assert!(runs.len() <= case.segments, "the case's segment capacity holds its runs");
            for (entry, run) in runs.iter().enumerate() {
                segments[(row * case.segments + entry) * 2] = run[0];
                segments[(row * case.segments + entry) * 2 + 1] = run[1];
            }
        }
        reads.push(read);
        all_scores.push(scores);
    }
    Model { block_keys, reads, scores: all_scores, segments, bitmap }
}

// ---- portable body and natives -----------------------------------------------------------------

struct Outcome {
    segments: Vec<i32>,
    bitmap: Vec<u32>,
    window: Vec<f32>,
    blocks: Vec<f32>,
}

fn portable(case: &Case) -> Outcome {
    use seismic_lang::{
        checked::{SourceFile, check_source},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let element = if case.element == Element::bf16() { DType::BF16 } else { DType::F32 };
    let mut sources = seismic_std::sources();
    for (path, text) in [
        ("indexer.seismic", include_str!("../kernels/indexer.seismic")),
        ("attention.seismic", include_str!("../kernels/attention.seismic")),
        ("functions.seismic", include_str!("../kernels/functions.seismic")),
    ] {
        sources.push(SourceFile { path: path.into(), text: text.into() });
    }
    let module = check_source(sources).unwrap();
    let logical = module
        .entry(
            module.entry_named("indexer_select").unwrap(),
            &ElementBindings::new().bind("A", registry::dense(element)),
        )
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let m = case.rows;
    let (visible, _, spans) = case.token_spans();
    let controls = case.controls();
    let mut tensor = |dtype, shape: Vec<usize>, values: Vec<f64>| {
        Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
    };
    let floats = |values: &[f32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let ints = |values: &[i32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let args = vec![
        tensor(element, vec![m, H, W], floats(&case.query)),
        tensor(element, vec![m, W], floats(&case.key)),
        tensor(DType::F32, vec![1, W], floats(&case.query_norm)),
        tensor(DType::F32, vec![1, W], floats(&case.key_norm)),
        tensor(DType::I32, vec![P], ints(&case.components)),
        tensor(DType::F32, vec![P], floats(&case.frequencies)),
        tensor(DType::F32, vec![P], vec![1.0; P]),
        tensor(DType::I32, vec![m, 4], ints(&controls.coordinates)),
        tensor(DType::I32, vec![m, controls.spans, 2], ints(&controls.window_visible)),
        tensor(DType::I32, vec![m], ints(&controls.window_destinations)),
        tensor(DType::I32, vec![m, controls.spans, 2], ints(&controls.block_visible)),
        tensor(DType::I32, vec![m], ints(&controls.block_destinations)),
        tensor(DType::I32, vec![m, spans, 2], ints(&visible)),
        tensor(DType::I32, vec![0, SB], Vec::new()),
        tensor(DType::I32, vec![0, case.segments], Vec::new()),
        tensor(DType::U32, vec![0, case.words], Vec::new()),
        tensor(DType::F32, vec![WINDOW_ROWS, W], floats(&case.window)),
        tensor(DType::I32, vec![WINDOW_ROWS, 4], ints(&case.window_planes())),
        tensor(DType::F32, vec![BLOCK_ROWS, W], floats(&case.blocks)),
        Arg::Scalar(ReferenceScalar::I32(case.k as i32)),
        Arg::Scalar(ReferenceScalar::F32(EPSILON.to_bits())),
        Arg::Scalar(ReferenceScalar::U32(SLAB_ROWS as u32)),
        Arg::Scalar(ReferenceScalar::U32(WINDOW_ROWS as u32)),
        Arg::Scalar(ReferenceScalar::U32(BLOCK_ROWS as u32)),
    ];
    let outcome = interpreter.run_bounded(&args, u64::MAX).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable indexer_select failed: {failure}");
    }
    let mut results = outcome.results();
    let read_ints = |value: OutcomeValue<'_>| match value {
        OutcomeValue::Tensor(reader) => {
            (0..reader.element_count()).map(|i| reader.read(i).unwrap() as i64).collect::<Vec<_>>()
        }
        _ => panic!("a tensor result"),
    };
    let segments = read_ints(results.next().unwrap().value()).into_iter().map(|v| v as i32).collect();
    let bitmap = read_ints(results.next().unwrap().value()).into_iter().map(|v| v as u32).collect();
    let plane = |ordinal: usize| {
        let input = outcome.inputs().find(|input| input.ordinal() == ordinal).unwrap();
        let reader = input.tensor();
        (0..reader.element_count()).map(|i| reader.read(i).unwrap() as f32).collect::<Vec<_>>()
    };
    Outcome { segments, bitmap, window: plane(16), blocks: plane(18) }
}

fn native(device: &Device, case: &Case) -> Outcome {
    let m = case.rows as u64;
    let statics = [("H", H as u64), ("P", P as u64), ("S", S as u64), ("SB", SB as u64)];
    let specialization = if is_cpu(device) {
        NativeSpecialization::new()
    } else {
        statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value))
    };
    let specialization = seismic::generated::native_implementation::<indexer_select::Entry>(device)
        .unwrap()
        .unwrap()
        .default_specialization(&specialization)
        .unwrap();
    let kernel = indexer_select::native_for_device_with(
        device,
        indexer_select::Elements { A: case.element },
        &specialization,
    )
    .unwrap();
    let (visible, _, spans) = case.token_spans();
    let controls = case.controls();
    let rows_tensor = |rows: usize, values: &[f32]| {
        let (slabs, logical) = slab_arenas(
            device,
            rows,
            rows as u64,
            &[Arena { element: Element::f32(), bank_shape: vec![W as u64], values }],
            false,
        );
        (slabs, logical.into_iter().next().unwrap())
    };
    let (_window_slabs, mut window) = rows_tensor(WINDOW_ROWS, &case.window);
    let (_block_slabs, mut block_plane) = rows_tensor(BLOCK_ROWS, &case.blocks);
    let window_coordinates = case.window_planes();
    let (_coordinate_slabs, window_coordinate_planes) = slab_arenas(
        device,
        WINDOW_ROWS,
        WINDOW_ROWS as u64,
        &[Arena {
            element: Element::i32(),
            bank_shape: vec![4],
            // 32-bit words travel as the bits of f32 values (`bytes`).
            values: &window_coordinates.iter().map(|value| f32::from_bits(*value as u32)).collect::<Vec<_>>(),
        }],
        false,
    );
    let mut window_coordinate_plane = window_coordinate_planes.into_iter().next().unwrap();
    let results = kernel
        .call(indexer_select::Args {
            query: &tensor(device, case.element, &[m, H as u64, W as u64], &case.query),
            key: &tensor(device, case.element, &[m, W as u64], &case.key),
            query_norm: &tensor(device, Element::f32(), &[1, W as u64], &case.query_norm),
            key_norm: &tensor(device, Element::f32(), &[1, W as u64], &case.key_norm),
            rotary_components: &ints(device, &[P as u64], &case.components),
            rotary_frequencies: &tensor(device, Element::f32(), &[P as u64], &case.frequencies),
            rotary_amplitudes: &tensor(device, Element::f32(), &[P as u64], &vec![1.0; P]),
            coordinates: &ints(device, &[m, 4], &controls.coordinates),
            window_visible: &ints(device, &[m, controls.spans as u64, 2], &controls.window_visible),
            window_destinations: &ints(device, &[m], &controls.window_destinations),
            block_visible: &ints(device, &[m, controls.spans as u64, 2], &controls.block_visible),
            block_destinations: &ints(device, &[m], &controls.block_destinations),
            visible: &ints(device, &[m, spans as u64, 2], &visible),
            block_slots: &ints(device, &[0, SB as u64], &[]),
            segment_slots: &ints(device, &[0, case.segments as u64], &[]),
            bitmap_words: &Tensor::from_host(device, Element::u32(), &[0, case.words as u64], &[]).unwrap(),
            window_keys: &mut window,
            window_coordinates: &mut window_coordinate_plane,
            block_keys: &mut block_plane,
            selected: case.k as i32,
            epsilon: EPSILON,
            slab_rows: SLAB_ROWS as u32,
            window_slab_rows: WINDOW_ROWS as u32,
            block_slab_rows: BLOCK_ROWS as u32,
        })
        .unwrap();
    let words = |tensor: &Tensor| {
        tensor
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    Outcome {
        segments: words(&results.r0).into_iter().map(|v| v as i32).collect(),
        bitmap: words(&results.r1),
        window: read(&window),
        blocks: read(&block_plane),
    }
}

fn check(case: &Case, label: &str) {
    let expected = model(case);
    let oracle = portable(case);
    assert_eq!(oracle.segments, expected.segments, "{label}: portable segments differ from the host model");
    assert_eq!(oracle.bitmap, expected.bitmap, "{label}: portable bitmap differs from the host model");
    for (block, key) in &expected.block_keys {
        let at = case.block_of[*block] as usize;
        let stored = &oracle.blocks[at * W..][..W];
        assert!(relative_max(stored, key) <= 1e-2, "{label}: portable block {block} key differs from the host model");
    }
    for row in 0..case.rows {
        let position = case.position(row);
        let at = case.window_of[position] as usize;
        assert!(same_bits(&oracle.window[at * W..][..W], &case.key[row * W..][..W]), "{label}: raw key appended");
        let read = expected.reads[row].iter().take((position + 1) / SB).filter(|r| **r).count();
        assert_eq!(read, ((position + 1) / SB).min(case.k), "{label}: row {row} reads min(K, complete) blocks");
    }
    for device in devices() {
        let backend = device.backend().as_str();
        let outcome = native(&device, case);
        assert_eq!(outcome.segments, oracle.segments, "{backend} {label}: segments");
        assert_eq!(outcome.bitmap, oracle.bitmap, "{backend} {label}: bitmap");
        assert!(same_bits(&outcome.window, &oracle.window), "{backend} {label}: window plane");
        for (block, _) in &expected.block_keys {
            let at = case.block_of[*block] as usize;
            let (native_key, portable_key) = (&outcome.blocks[at * W..][..W], &oracle.blocks[at * W..][..W]);
            // F32 arithmetic order.
            let step = 1e-5;
            for (a, e) in native_key.iter().zip(portable_key) {
                assert!((a - e).abs() <= e.abs() * step + 1e-6, "{backend} {label}: block {block} key {a} vs {e}");
            }
        }
    }
}

#[test]
fn decode_rows_select_segments() {
    // Two rows past the dense limit, the second finishing block 10.
    check(&Case::new(2, 42, 4, 16, true, 1), "decode");
    // One row whose block is complete at its own position.
    check(&Case::new(1, 47, 4, 16, true, 2), "decode own block");
}

#[test]
fn dense_rows_read_every_visible_row() {
    check(&Case::new(2, 42, 16, 16, true, 3), "dense decode");
    check(&Case::new(16, 20, 16, 16, false, 4), "dense prefill");
}

#[test]
fn prefill_rows_select_a_bitmap() {
    // Sixteen rows finishing four blocks, mixing dense and selecting rows.
    check(&Case::new(16, 12, 5, 16, false, 5), "prefill mixed");
    check(&Case::new(16, 36, 4, 16, false, 6), "prefill");
}

#[test]
fn f32_indexer_selects_the_reference_blocks_of_near_ties() {
    // A bf16-input case (the engine's binding) whose near-tie scores an
    // indexer rounding its block keys and queries to bf16 would order
    // differently, while every selecting row's K-th and (K + 1)-th F32
    // scores lie farther apart than F32 summation order moves them.
    let case = (0..400)
        .map(|seed| Case::of(Element::bf16(), 8, 40, 3, 16, false, 100 + seed))
        .find(|case| {
            let reference = model(case);
            let rounded = model_rounding(case, Element::bf16());
            let robust = reference.scores.iter().all(|scores| {
                let mut sorted = scores.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                sorted.len() <= case.k
                    || sorted[case.k - 1] - sorted[case.k] > 1e-4 * sorted[0].abs().max(1e-6)
            });
            reference.bitmap != rounded.bitmap && robust
        })
        .expect("a case whose bf16 rounding reorders its near ties");
    check(&case, "near ties");
    check(&Case::of(Element::f32(), 2, 42, 4, 16, true, 8), "f32 inputs decode");
}
