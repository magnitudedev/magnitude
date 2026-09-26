//! Kernel-level golden harness for bit-exact refactoring of the native kernels.
//!
//! For every entry of the selected families it forms the native
//! implementation of the opened device's backend at a covering sample of the
//! declared parameter configurations (every configuration when the domain is
//! small; otherwise the defaults, each single-parameter variation and a fixed
//! pseudo-random sample, all filtered by `where`), runs representative shapes
//! at the real Qwen3.5 geometry (real 4B GGUF weights where the model has
//! them), and hashes every result and every `&mut` argument after the call.
//! `record` writes the hashes (running each call twice to reject
//! nondeterministic outputs); `compare` requires identical hashes for the
//! same key set.
//!
//! Sources are read from `kernels/` (or `GOLDEN_KERNELS`) at run time through
//! the dynamic API, so a record, an edit of the kernel sources and a compare
//! run use one test binary.
//!
//! ```text
//! GOLDEN_MODE=record|compare GOLDEN_DIR=<dir> GOLDEN_FAMILIES=projection,readout \
//!   [GOLDEN_ENTRIES=dense_expand,...] [GOLDEN_GGUF=<Qwen3.5-4B-Q4_K_M.gguf>] \
//!   cargo test --release -p magnitude-model-kernels --test golden -- --ignored --nocapture
//! ```
//!
//! Families: projection, attention, recurrent, readout, routed, sampling,
//! import, state, vision, plus `library` (every entry built on the projection
//! library) and `all`.

use seismic::dynamic::{Function, Module, Scalar, SignatureType, Tensor, TensorAccess, Value};
use seismic::{BackendName, Device, DeviceCatalog, Element, Layout, NativeSpecialization};
use seismic_lang::checked::{CheckedModule, NativeImplementation};
use seismic_lang::registry::{bf16_round, f16_bits};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, bound: usize) -> usize {
        self.next() as usize % bound
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
}

/// Activation-like values: uniform spread plus a few large channels, as real
/// hidden states carry.
fn activations(rng: &mut Rng, rows: usize, width: usize, scale: f32) -> Vec<f32> {
    let outliers = [width / 7, width / 3, width / 2 + 1, width - 5];
    (0..rows * width)
        .map(|index| {
            let value = rng.unit() * scale;
            if width > 64 && outliers.contains(&(index % width)) {
                value * 40.0
            } else {
                value
            }
        })
        .collect()
}

fn uniform(rng: &mut Rng, count: usize, low: f32, high: f32) -> Vec<f32> {
    (0..count)
        .map(|_| low + (high - low) * (rng.unit() + 1.0) * 0.5)
        .collect()
}

fn dense_bytes(element: Element, values: &[f32]) -> Vec<u8> {
    match element.name() {
        "f32" => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        "bf16" => values
            .iter()
            .flat_map(|v| ((bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        "f16" => values
            .iter()
            .flat_map(|v| f16_bits(*v).to_le_bytes())
            .collect(),
        other => panic!("no dense encoding for {other}"),
    }
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A 128-bit content hash (two independent multiply-xorshift lanes over
/// 8-byte words, then the length).
fn hash(bytes: &[u8]) -> String {
    let mix = |mut x: u64| {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^ (x >> 33)
    };
    let (mut a, mut b) = (0x243f_6a88_85a3_08d3u64, 0x1319_8a2e_0370_7344u64);
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        a = mix(a ^ word).wrapping_add(0x9e37_79b9_7f4a_7c15);
        b = mix(b.rotate_left(29) ^ word.wrapping_mul(0x2545_f491_4f6c_dd1d));
    }
    let mut tail = [0u8; 8];
    tail[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
    let word = u64::from_le_bytes(tail);
    a = mix(a ^ word ^ bytes.len() as u64);
    b = mix(b ^ word.rotate_left(17) ^ (bytes.len() as u64).wrapping_mul(3));
    format!("{a:016x}{b:016x}")
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Q4K,
    Q5K,
    Q6K,
    Q8,
    Iq4,
}

impl Format {
    fn from_gguf(kind: u32) -> Self {
        match kind {
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            8 => Self::Q8,
            23 => Self::Iq4,
            other => panic!("GGUF type {other}"),
        }
    }
    fn external(self) -> Element {
        Element::named(match self {
            Self::Q4K => "gguf_q4_k",
            Self::Q5K => "gguf_q5_k",
            Self::Q6K => "gguf_q6_k",
            Self::Q8 => "gguf_q8_0",
            Self::Iq4 => "gguf_iq4_xs",
        })
        .unwrap()
    }
    fn representation(self) -> &'static str {
        match self {
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
            Self::Q8 => "q8g32s",
            Self::Iq4 => "iq4g32",
        }
    }
    fn block_values(self) -> usize {
        match self {
            Self::Q8 => 32,
            _ => 256,
        }
    }
    fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8 => 34,
            Self::Iq4 => 136,
        }
    }
    /// One pseudo-random GGUF block with GGUF-like (small positive) factors.
    fn block(self, rng: &mut Rng) -> Vec<u8> {
        let f16 = |value: f32| f16_bits(value).to_le_bytes();
        let mut block = Vec::new();
        let factor =
            |rng: &mut Rng, low: f32, high: f32| low + (high - low) * (rng.unit() + 1.0) * 0.5;
        match self {
            Self::Q4K | Self::Q5K => {
                block.extend(f16(factor(rng, 0.0005, 0.003)));
                block.extend(f16(factor(rng, 0.0, 0.002)));
                let bytes = if self == Self::Q4K {
                    12 + 128
                } else {
                    12 + 32 + 128
                };
                block.extend((0..bytes).map(|_| rng.byte()));
            }
            Self::Q6K => {
                block.extend((0..128 + 64).map(|_| rng.byte()));
                block.extend((0..16).map(|_| (rng.next() % 128) as u8 ^ 0xC0));
                block.extend(f16(factor(rng, 0.0002, 0.001)));
            }
            Self::Q8 => {
                block.extend(f16(factor(rng, 0.001, 0.005)));
                block.extend((0..32).map(|_| rng.byte()));
            }
            Self::Iq4 => {
                block.extend(f16(factor(rng, 0.0005, 0.003)));
                block.extend((0..2 + 4 + 128).map(|_| rng.byte()));
            }
        }
        block
    }
}

/// The tensors of a GGUF file, read on demand.
struct Gguf {
    path: PathBuf,
    data: u64,
    tensors: HashMap<String, (u32, Vec<u64>, u64)>,
}

impl Gguf {
    fn open(path: &Path) -> Self {
        let mut file = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        // The header of the pinned file (metadata incl. the tokenizer) is ~10 MB.
        let mut bytes = vec![0u8; 32 << 20];
        let read = file.read(&mut bytes).unwrap();
        let mut at = 0usize;
        let take = |n: usize, at: &mut usize| {
            let slice = bytes[*at..*at + n].to_vec();
            *at += n;
            slice
        };
        let u32_at = |b: Vec<u8>| u32::from_le_bytes(b.try_into().unwrap());
        let u64_at = |b: Vec<u8>| u64::from_le_bytes(b.try_into().unwrap());
        assert_eq!(take(4, &mut at), b"GGUF");
        take(4, &mut at);
        let count = u64_at(take(8, &mut at));
        let metadata = u64_at(take(8, &mut at));
        fn skip(kind: u32, at: &mut usize, bytes: &[u8]) -> Option<u64> {
            let u32v = |at: &mut usize| {
                let v = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
                *at += 4;
                v
            };
            let u64v = |at: &mut usize| {
                let v = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap());
                *at += 8;
                v
            };
            match kind {
                0 | 1 | 7 => {
                    *at += 1;
                    Some(u64::from(bytes[*at - 1]))
                }
                2 | 3 => {
                    *at += 2;
                    None
                }
                4 | 5 | 6 => Some(u64::from(u32v(at))),
                8 => {
                    let n = u64v(at) as usize;
                    *at += n;
                    None
                }
                9 => {
                    let inner = u32v(at);
                    let n = u64v(at);
                    for _ in 0..n {
                        skip(inner, at, bytes);
                    }
                    None
                }
                10 | 11 | 12 => Some(u64v(at)),
                other => panic!("GGUF metadata type {other}"),
            }
        }
        let string = |at: &mut usize, bytes: &[u8]| {
            let n = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap()) as usize;
            *at += 8;
            let s = String::from_utf8(bytes[*at..*at + n].to_vec()).unwrap();
            *at += n;
            s
        };
        let mut alignment = 32u64;
        for _ in 0..metadata {
            let key = string(&mut at, &bytes);
            let kind = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            at += 4;
            let value = skip(kind, &mut at, &bytes);
            if key == "general.alignment" {
                alignment = value.unwrap();
            }
        }
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let name = string(&mut at, &bytes);
            let rank = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            at += 4;
            let dims = (0..rank)
                .map(|_| {
                    let v = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
                    at += 8;
                    v
                })
                .collect::<Vec<_>>();
            let kind = u32_at(bytes[at..at + 4].to_vec());
            at += 4;
            let offset = u64_at(bytes[at..at + 8].to_vec());
            at += 8;
            tensors.insert(name, (kind, dims, offset));
        }
        assert!(at <= read, "GGUF header exceeds the read window");
        let data = (at as u64).div_ceil(alignment) * alignment;
        Self {
            path: path.to_path_buf(),
            data,
            tensors,
        }
    }

    fn read(&self, offset: u64, len: usize) -> Vec<u8> {
        let mut file = File::open(&self.path).unwrap();
        file.seek(SeekFrom::Start(self.data + offset)).unwrap();
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).unwrap();
        bytes
    }

    /// Rows [first, first + rows) of a quantized matrix (GGUF dims [K, N]):
    /// (format, K, external bytes).
    fn rows(&self, name: &str, first: usize, rows: usize) -> (Format, usize, Vec<u8>) {
        let (kind, dims, offset) = &self.tensors[name];
        let format = Format::from_gguf(*kind);
        let k = dims[0] as usize;
        let row_bytes = k / format.block_values() * format.block_bytes();
        (
            format,
            k,
            self.read(offset + (first * row_bytes) as u64, rows * row_bytes),
        )
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        let (kind, dims, offset) = &self.tensors[name];
        assert_eq!(*kind, 0, "{name} is not F32");
        let n = dims.iter().product::<u64>() as usize;
        self.read(*offset, 4 * n)
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// One argument. Shared tensors are uploaded once per case; mutable ones
/// are recreated from their bytes for every run.
#[derive(Clone)]
enum Arg {
    Shared(Tensor),
    Mutable {
        element: Element,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    },
    Scalar(Scalar),
}

/// One call shape.
struct Case {
    label: String,
    args: Vec<Arg>,
}

/// Element bindings and static dimensions shared by a list of cases.
struct Variant {
    label: String,
    elements: Vec<(&'static str, Element)>,
    statics: Vec<(&'static str, u64)>,
    /// Configurations to run: every sampled one (true) or the first three.
    every_configuration: bool,
    cases: Vec<Case>,
}

struct Ctx {
    device: Device,
    backend: BackendName,
    gguf: Option<Gguf>,
    /// Resident real weights by (tensor name, rows).
    weights: std::cell::RefCell<HashMap<(String, usize), (Tensor, Format)>>,
}

impl Ctx {
    fn layout(&self) -> Layout {
        match self.backend {
            BackendName::Cuda => Layout::Mma16,
            _ => Layout::Rows16,
        }
    }
    fn tensor(&self, element: Element, shape: &[u64], bytes: &[u8]) -> Tensor {
        Tensor::from_host(&self.device, element, shape, bytes)
            .unwrap_or_else(|e| panic!("upload {} {shape:?}: {e}", element.name()))
    }
    fn dense(&self, element: Element, shape: &[u64], values: &[f32]) -> Arg {
        Arg::Shared(self.tensor(element, shape, &dense_bytes(element, values)))
    }
    fn dense_mut(&self, element: Element, shape: &[u64], values: &[f32]) -> Arg {
        Arg::Mutable {
            element,
            shape: shape.to_vec(),
            bytes: dense_bytes(element, values),
        }
    }
    fn ints(&self, shape: &[u64], values: &[i32]) -> Arg {
        Arg::Shared(self.tensor(Element::i32(), shape, &i32_bytes(values)))
    }
    fn resident(&self, format: Format) -> Element {
        Element::stored(format.representation(), self.layout()).unwrap()
    }
    /// A resident packed weight of `shape` from external GGUF bytes.
    fn packed(&self, format: Format, shape: &[u64], external: &[u8]) -> Tensor {
        let resident = self.resident(format);
        let bytes = resident
            .repack_host(format.external(), shape, external)
            .unwrap_or_else(|| panic!("{format:?} {shape:?} repack"));
        self.tensor(resident, shape, &bytes)
    }
    /// A dense weight of `shape` holding the decoded values of a packed one.
    /// A dense weight of `shape` with GGUF-like magnitudes (the host decode
    /// of a packed weight is far too slow at model widths).
    fn dense_weight(&self, shape: &[u64], seed: u64, element: Element) -> Tensor {
        let mut rng = Rng::new(seed);
        let values = uniform(
            &mut rng,
            shape.iter().product::<u64>() as usize,
            -0.04,
            0.04,
        );
        self.tensor(element, shape, &dense_bytes(element, &values))
    }
    fn gguf(&self) -> &Gguf {
        self.gguf
            .as_ref()
            .expect("GOLDEN_GGUF names the pinned Qwen3.5-4B Q4_K_M file")
    }
    /// Rows of a real GGUF matrix, resident; or a dense weight of its shape.
    fn real(&self, name: &str, rows: usize, dense: Option<Element>) -> (Tensor, Format) {
        let (kind, dims, _) = &self.gguf().tensors[name];
        let (format, k) = (Format::from_gguf(*kind), dims[0]);
        if let Some(element) = dense {
            let shape = [rows as u64, k];
            return (
                self.dense_weight(&shape, name.len() as u64 * 131 + rows as u64, element),
                format,
            );
        }
        let key = (name.to_string(), rows);
        if let Some(cached) = self.weights.borrow().get(&key) {
            return cached.clone();
        }
        let (format, k, external) = self.gguf().rows(name, 0, rows);
        let weight = (
            self.packed(format, &[rows as u64, k as u64], &external),
            format,
        );
        self.weights.borrow_mut().insert(key, weight.clone());
        weight
    }
    /// A synthetic packed weight (rank 2 or 3), or a dense one.
    fn synthetic(
        &self,
        format: Format,
        shape: &[u64],
        seed: u64,
        dense: Option<Element>,
    ) -> Tensor {
        if let Some(element) = dense {
            return self.dense_weight(shape, seed, element);
        }
        let mut rng = Rng::new(seed);
        let values = shape.iter().product::<u64>() as usize;
        let external = (0..values / format.block_values())
            .flat_map(|_| format.block(&mut rng))
            .collect::<Vec<_>>();
        self.packed(format, shape, &external)
    }
    fn norm(&self, name: &str, element: Element) -> Arg {
        let values = self.gguf().f32s(name);
        self.dense(element, &[values.len() as u64], &values)
    }
}

fn f32s(value: f32) -> Arg {
    Arg::Scalar(Scalar::F32(value.to_bits()))
}

fn i32s(value: i32) -> Arg {
    Arg::Scalar(Scalar::I32(value))
}

/// `out_rows` for O of M rows: the identity when O == M, else a scattered
/// ascending selection that ends at the last row.
fn out_rows(m: usize, o: usize) -> Vec<i32> {
    if o == m {
        return (0..m as i32).collect();
    }
    (0..o)
        .map(|i| if i + 1 == o { m - 1 } else { i * m / o } as i32)
        .collect()
}

const PROJECTION_ROWS: [(usize, usize); 12] = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (8, 8),
    (9, 9),
    (12, 4),
    (33, 33),
    (40, 17),
    (64, 64),
    (130, 130),
];

fn bf16() -> Element {
    Element::bf16()
}
fn f16() -> Element {
    Element::f16()
}
fn f32e() -> Element {
    Element::f32()
}

// The Qwen3.5-4B geometry.
const HIDDEN: usize = 2560;
const FFN: usize = 9216;
const KV: usize = 4;
const GROUP: usize = 4;
const HEAD: usize = 256;
const PAIRS: usize = 32;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const STATE: usize = 128;
const CONVOLUTION: usize = 4;
const VOCABULARY_SLICE: usize = 8200;

/// Activation element variants of a projection entry: (label, A, norm
/// element, dense weights, every configuration).
fn projection_variants() -> Vec<(&'static str, Element, Element, Option<Element>, bool)> {
    vec![
        ("bf16", bf16(), f32e(), None, true),
        ("f16", f16(), bf16(), None, false),
        ("f32", f32e(), f32e(), None, false),
        ("dense-bf16", bf16(), f32e(), Some(bf16()), false),
        ("dense-f16", f16(), f32e(), Some(f16()), false),
    ]
}

fn dense_expand(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, nw, dense, every)| {
            let (gate, gf) = ctx.real("blk.0.ffn_gate.weight", FFN, dense);
            let (up, uf) = ctx.real("blk.0.ffn_up.weight", FFN, dense);
            let norm = ctx.norm("blk.0.post_attention_norm.weight", nw);
            let weight_element = |f: Format| dense.unwrap_or_else(|| ctx.resident(f));
            let mut rng = Rng::new(1);
            let cases = PROJECTION_ROWS
                .iter()
                .map(|&(m, o)| Case {
                    label: format!("m{m}o{o}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, HIDDEN as u64],
                            &activations(&mut rng, m, HIDDEN, 1.0),
                        ),
                        norm.clone(),
                        Arg::Shared(gate.clone()),
                        Arg::Shared(up.clone()),
                        ctx.ints(&[o as u64], &out_rows(m, o)),
                        f32s(1e-6),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![
                    ("NW", nw),
                    ("GW", weight_element(gf)),
                    ("UW", weight_element(uf)),
                    ("A", a),
                ],
                statics: vec![("H", HIDDEN as u64), ("F", FFN as u64)],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn dense_output(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, _, dense, every)| {
            let (down, df) = ctx.real("blk.0.ffn_down.weight", HIDDEN, dense);
            let mut rng = Rng::new(2);
            let cases = PROJECTION_ROWS
                .iter()
                .map(|&(m, o)| Case {
                    label: format!("m{m}o{o}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, HIDDEN as u64],
                            &activations(&mut rng, m, HIDDEN, 1.0),
                        ),
                        ctx.dense(
                            a,
                            &[o as u64, FFN as u64],
                            &activations(&mut rng, o, FFN, 0.3),
                        ),
                        Arg::Shared(down.clone()),
                        ctx.ints(&[o as u64], &out_rows(m, o)),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![("DW", dense.unwrap_or_else(|| ctx.resident(df))), ("A", a)],
                statics: vec![("H", HIDDEN as u64), ("F", FFN as u64)],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn rows_only() -> Vec<usize> {
    PROJECTION_ROWS
        .iter()
        .filter(|(m, o)| m == o)
        .map(|(m, _)| *m)
        .collect()
}

fn attention_project(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, nw, dense, every)| {
            let (q, qf) = ctx.real("blk.3.attn_q.weight", KV * GROUP * 2 * HEAD, dense);
            let (k, kf) = ctx.real("blk.3.attn_k.weight", KV * HEAD, dense);
            let (v, vf) = ctx.real("blk.3.attn_v.weight", KV * HEAD, dense);
            let norm = ctx.norm("blk.3.attn_norm.weight", nw);
            let query_norm = ctx.norm("blk.3.attn_q_norm.weight", f32e());
            let element = |f: Format| dense.unwrap_or_else(|| ctx.resident(f));
            let mut rng = Rng::new(3);
            let cases = rows_only()
                .into_iter()
                .map(|m| Case {
                    label: format!("m{m}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, HIDDEN as u64],
                            &activations(&mut rng, m, HIDDEN, 1.0),
                        ),
                        norm.clone(),
                        query_norm.clone(),
                        Arg::Shared(q.clone()),
                        Arg::Shared(k.clone()),
                        Arg::Shared(v.clone()),
                        f32s(1e-6),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![
                    ("NW", nw),
                    ("QW", element(qf)),
                    ("KW", element(kf)),
                    ("VW", element(vf)),
                    ("A", a),
                ],
                statics: vec![
                    ("D", HIDDEN as u64),
                    ("KV", KV as u64),
                    ("G", GROUP as u64),
                    ("W", HEAD as u64),
                ],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn attention_output(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, _, dense, every)| {
            let (weight, wf) = ctx.real("blk.3.attn_output.weight", HIDDEN, dense);
            let mut rng = Rng::new(4);
            let q = KV * GROUP;
            let cases = rows_only()
                .into_iter()
                .map(|m| Case {
                    label: format!("m{m}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, HIDDEN as u64],
                            &activations(&mut rng, m, HIDDEN, 1.0),
                        ),
                        ctx.dense(
                            a,
                            &[m as u64, q as u64, HEAD as u64],
                            &activations(&mut rng, m, q * HEAD, 0.5),
                        ),
                        Arg::Shared(weight.clone()),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![("OW", dense.unwrap_or_else(|| ctx.resident(wf))), ("A", a)],
                statics: vec![("D", HIDDEN as u64), ("Q", q as u64), ("W", HEAD as u64)],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

/// One attention row: visible history spans, fresh span, destination and
/// position.
struct AttentionRow {
    spans: Vec<(i32, i32)>,
    fresh: (i32, i32),
    destination: i32,
    position: i32,
}

fn speculative_rows(rows: usize, context: i32) -> Vec<AttentionRow> {
    (0..rows)
        .map(|row| AttentionRow {
            spans: vec![(0, context)],
            fresh: (0, row as i32 + 1),
            destination: context + row as i32,
            position: context + row as i32,
        })
        .collect()
}

fn mixed_decode_rows(base: i32) -> Vec<AttentionRow> {
    vec![
        AttentionRow {
            spans: vec![(0, base), (base + 7, base + 19)],
            fresh: (0, 1),
            destination: base + 40,
            position: base + 12,
        },
        AttentionRow {
            spans: vec![(0, base), (base + 7, base + 19)],
            fresh: (0, 2),
            destination: base + 41,
            position: base + 13,
        },
        AttentionRow {
            spans: vec![(base + 20, base + 33)],
            fresh: (2, 3),
            destination: -1,
            position: 13,
        },
        AttentionRow {
            spans: vec![],
            fresh: (0, 0),
            destination: -1,
            position: 0,
        },
    ]
}

fn prefill_rows(rows: usize, history: i32) -> Vec<AttentionRow> {
    let boundary = rows * 2 / 3;
    (0..rows)
        .map(|row| {
            let r = row as i32;
            if row + 2 >= rows {
                AttentionRow {
                    spans: vec![],
                    fresh: (0, 0),
                    destination: -1,
                    position: 0,
                }
            } else if row < boundary {
                AttentionRow {
                    spans: if history == 0 {
                        vec![]
                    } else {
                        vec![(0, history / 2), (history / 2 + 9, history)]
                    },
                    fresh: (0, r + 1),
                    destination: if row % 5 == 3 { -1 } else { history + 64 + r },
                    position: history - 9 + r,
                }
            } else {
                let first = boundary as i32;
                AttentionRow {
                    spans: if history == 0 {
                        vec![]
                    } else {
                        vec![(history + 3, history + 17)]
                    },
                    fresh: (first, r + 1),
                    destination: history + 64 + r,
                    position: 14 + r - first,
                }
            }
        })
        .collect()
}

fn attention_case(
    ctx: &Ctx,
    a: Element,
    label: String,
    history: usize,
    spans: usize,
    rows: &[AttentionRow],
    seed: u64,
) -> Case {
    let mut rng = Rng::new(seed);
    let m = rows.len();
    let w = 2 * PAIRS + (HEAD - 2 * PAIRS);
    let heads = KV * GROUP;
    let components = (0..PAIRS)
        .map(|pair| match pair % 3 {
            1 if pair < 3 * PAIRS * 11 / 32 => 1,
            2 if pair < 3 * PAIRS * 10 / 32 => 2,
            _ => 0,
        })
        .collect::<Vec<i32>>();
    let frequencies = (0..PAIRS)
        .map(|pair| 1.0e7f64.powf(-((2 * pair) as f64) / (2 * PAIRS) as f64) as f32)
        .collect::<Vec<_>>();
    let coordinates = rows
        .iter()
        .flat_map(|row| [row.position, row.position + 3, row.position / 2, 0])
        .collect::<Vec<_>>();
    let visible = rows
        .iter()
        .flat_map(|row| {
            row.spans
                .iter()
                .copied()
                .chain(std::iter::repeat((0, 0)))
                .take(spans)
                .flat_map(|(lo, hi)| [lo, hi])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let values = |rng: &mut Rng, count: usize, scale: f32| {
        (0..count).map(|_| rng.unit() * scale).collect::<Vec<_>>()
    };
    let (m64, w64, t64) = (m as u64, w as u64, history as u64);
    Case {
        label,
        args: vec![
            ctx.dense(
                a,
                &[m64, (heads * 2 * w) as u64],
                &values(&mut rng, m * heads * 2 * w, 2.0),
            ),
            ctx.dense(
                a,
                &[m64, (KV * w) as u64],
                &values(&mut rng, m * KV * w, 2.0),
            ),
            ctx.dense(
                a,
                &[m64, (KV * w) as u64],
                &values(&mut rng, m * KV * w, 1.0),
            ),
            ctx.norm("blk.3.attn_q_norm.weight", f32e()),
            ctx.norm("blk.3.attn_k_norm.weight", f32e()),
            ctx.ints(&[PAIRS as u64], &components),
            ctx.dense(f32e(), &[PAIRS as u64], &frequencies),
            ctx.ints(&[m64, 4], &coordinates),
            ctx.ints(&[m64, spans as u64, 2], &visible),
            ctx.ints(
                &[m64, 2],
                &rows
                    .iter()
                    .flat_map(|r| [r.fresh.0, r.fresh.1])
                    .collect::<Vec<_>>(),
            ),
            ctx.ints(
                &[m64],
                &rows.iter().map(|r| r.destination).collect::<Vec<_>>(),
            ),
            ctx.dense_mut(
                a,
                &[t64, KV as u64, w64],
                &values(&mut rng, history * KV * w, 3.0),
            ),
            ctx.dense_mut(
                a,
                &[t64, KV as u64, w64],
                &values(&mut rng, history * KV * w, 1.0),
            ),
            f32s(1e-6),
            f32s(1.0 / (w as f32).sqrt()),
        ],
    }
}

fn attention_statics() -> Vec<(&'static str, u64)> {
    vec![
        ("KV", KV as u64),
        ("G", GROUP as u64),
        ("P", PAIRS as u64),
        ("S", (HEAD - 2 * PAIRS) as u64),
    ]
}

fn attention_decode(ctx: &Ctx) -> Vec<Variant> {
    [("bf16", bf16(), true), ("f16", f16(), false)]
        .into_iter()
        .map(|(label, a, every)| {
            let cases = vec![
                attention_case(
                    ctx,
                    a,
                    "spec1@256".into(),
                    256 + 64,
                    1,
                    &speculative_rows(1, 256),
                    11,
                ),
                attention_case(
                    ctx,
                    a,
                    "spec8@4096".into(),
                    4096 + 64,
                    1,
                    &speculative_rows(8, 4096),
                    12,
                ),
                attention_case(
                    ctx,
                    a,
                    "mixed4@300".into(),
                    300 + 128,
                    2,
                    &mixed_decode_rows(300),
                    13,
                ),
                attention_case(
                    ctx,
                    a,
                    "spec2@16384".into(),
                    16384 + 64,
                    1,
                    &speculative_rows(2, 16384),
                    14,
                ),
                attention_case(ctx, a, "spec1@1".into(), 64, 1, &speculative_rows(1, 1), 15),
            ];
            Variant {
                label: label.into(),
                elements: vec![("A", a)],
                statics: attention_statics(),
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn attention_prefill(ctx: &Ctx) -> Vec<Variant> {
    [("bf16", bf16(), true), ("f16", f16(), false)]
        .into_iter()
        .map(|(label, a, every)| {
            let cases = [(16usize, 0usize), (40, 300), (128, 1000), (64, 4096)]
                .into_iter()
                .enumerate()
                .map(|(i, (rows, history))| {
                    attention_case(
                        ctx,
                        a,
                        format!("m{rows}@{history}"),
                        history + 256,
                        2,
                        &prefill_rows(rows, history as i32),
                        20 + i as u64,
                    )
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![("A", a)],
                statics: attention_statics(),
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn recurrent_channels() -> usize {
    (2 * KEY_HEADS + VALUE_HEADS) * STATE
}

fn recurrent_width() -> usize {
    recurrent_channels() + VALUE_HEADS * STATE + 2 * VALUE_HEADS
}

fn recurrent_statics_projection() -> Vec<(&'static str, u64)> {
    vec![
        ("H", HIDDEN as u64),
        ("NK", KEY_HEADS as u64),
        ("NV", VALUE_HEADS as u64),
        ("W", STATE as u64),
    ]
}

fn recurrent_project(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, nw, dense, every)| {
            let (qkv, qf) = ctx.real("blk.0.attn_qkv.weight", recurrent_channels(), dense);
            let (gate, gf) = ctx.real("blk.0.attn_gate.weight", VALUE_HEADS * STATE, dense);
            let (alpha, af) = ctx.real("blk.0.ssm_alpha.weight", VALUE_HEADS, dense);
            let (beta, bf) = ctx.real("blk.0.ssm_beta.weight", VALUE_HEADS, dense);
            let norm = ctx.norm("blk.0.attn_norm.weight", nw);
            let element = |f: Format| dense.unwrap_or_else(|| ctx.resident(f));
            let mut rng = Rng::new(5);
            let cases = rows_only()
                .into_iter()
                .map(|m| Case {
                    label: format!("m{m}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, HIDDEN as u64],
                            &activations(&mut rng, m, HIDDEN, 1.0),
                        ),
                        norm.clone(),
                        Arg::Shared(qkv.clone()),
                        Arg::Shared(gate.clone()),
                        Arg::Shared(alpha.clone()),
                        Arg::Shared(beta.clone()),
                        f32s(1e-6),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![
                    ("NW", nw),
                    ("QW", element(qf)),
                    ("GW", element(gf)),
                    ("AW", element(af)),
                    ("BW", element(bf)),
                    ("A", a),
                ],
                statics: recurrent_statics_projection(),
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn recurrent_output(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, nw, dense, every)| {
            let (weight, wf) = ctx.real("blk.0.ssm_out.weight", HIDDEN, dense);
            let norm = ctx.norm("blk.0.ssm_norm.weight", nw);
            let mut rng = Rng::new(6);
            let cases = rows_only()
                .into_iter()
                .map(|m| Case {
                    label: format!("m{m}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, HIDDEN as u64],
                            &activations(&mut rng, m, HIDDEN, 1.0),
                        ),
                        ctx.dense(
                            a,
                            &[m as u64, VALUE_HEADS as u64, STATE as u64],
                            &activations(&mut rng, m, VALUE_HEADS * STATE, 0.2),
                        ),
                        ctx.dense(
                            a,
                            &[m as u64, recurrent_width() as u64],
                            &activations(&mut rng, m, recurrent_width(), 1.0),
                        ),
                        norm.clone(),
                        Arg::Shared(weight.clone()),
                        f32s(1e-6),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![
                    ("RN", nw),
                    ("OW", dense.unwrap_or_else(|| ctx.resident(wf))),
                    ("A", a),
                ],
                statics: recurrent_statics_projection(),
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

/// One recurrent slot: rows, published prefix, accepted and successor bank.
struct Slot {
    rows: usize,
    stop: usize,
    previous: usize,
    following: usize,
}

fn recurrent_case(
    ctx: &Ctx,
    a: Element,
    label: &str,
    rows: usize,
    slots: &[Slot],
    banks: usize,
    grouped: bool,
    seed: u64,
) -> Case {
    let mut rng = Rng::new(seed);
    let channels = recurrent_channels();
    let width = recurrent_width();
    let gates = channels + VALUE_HEADS * STATE;
    let mut projection = (0..rows * width).map(|_| rng.unit()).collect::<Vec<_>>();
    for row in 0..rows {
        for head in 0..VALUE_HEADS {
            projection[row * width + gates + head] = rng.unit() * 3.0;
            projection[row * width + gates + VALUE_HEADS + head] = rng.unit() * 4.0;
        }
    }
    // Two tape rows per bank: slots that stop early record them. Every slot
    // reads its bank's own state (previous_tape = 0).
    const TAPE: usize = 2;
    let window_bank = (CONVOLUTION - 1 + TAPE) * channels;
    let delta_bank = VALUE_HEADS * STATE * STATE;
    let tape_row = (VALUE_HEADS + KEY_HEADS) * STATE + VALUE_HEADS;
    let mut window = vec![7.25f32; banks * window_bank];
    let mut delta = vec![7.25f32; banks * delta_bank];
    let mut tape = vec![7.25f32; banks * TAPE * tape_row];
    window[..window_bank].fill(0.0);
    delta[..delta_bank].fill(0.0);
    tape[..TAPE * tape_row].fill(0.0);
    for slot in slots {
        if slot.previous > 0 {
            for value in &mut window[slot.previous * window_bank..][..window_bank] {
                *value = rng.unit();
            }
            for value in &mut delta[slot.previous * delta_bank..][..delta_bank] {
                *value = rng.unit() * 0.3;
            }
        }
    }
    let mut segments = Vec::new();
    let mut row = 0;
    for slot in slots {
        segments.extend([row as i32, (row + slot.rows) as i32]);
        row += slot.rows;
    }
    segments.extend([rows as i32, rows as i32]);
    let b = slots.len() as u64;
    let gguf = ctx.gguf();
    let convolution = gguf.f32s("blk.0.ssm_conv1d.weight");
    Case {
        label: label.into(),
        args: vec![
            ctx.dense(a, &[rows as u64, width as u64], &projection),
            ctx.dense(f32e(), &[channels as u64, CONVOLUTION as u64], &convolution),
            ctx.dense(f32e(), &[VALUE_HEADS as u64], &gguf.f32s("blk.0.ssm_a")),
            ctx.dense(
                f32e(),
                &[VALUE_HEADS as u64],
                &gguf.f32s("blk.0.ssm_dt.bias"),
            ),
            ctx.ints(&[b + 1, 2], &segments),
            ctx.ints(
                &[b],
                &slots.iter().map(|s| s.stop as i32).collect::<Vec<_>>(),
            ),
            ctx.ints(
                &[b],
                &slots.iter().map(|s| s.previous as i32).collect::<Vec<_>>(),
            ),
            ctx.ints(&[b], &vec![0; slots.len()]),
            ctx.ints(
                &[b],
                &slots.iter().map(|s| s.following as i32).collect::<Vec<_>>(),
            ),
            ctx.dense_mut(
                a,
                &[
                    banks as u64,
                    (CONVOLUTION - 1 + TAPE) as u64,
                    channels as u64,
                ],
                &window,
            ),
            ctx.dense_mut(
                f32e(),
                &[banks as u64, VALUE_HEADS as u64, STATE as u64, STATE as u64],
                &delta,
            ),
            ctx.dense_mut(f32e(), &[banks as u64, TAPE as u64, tape_row as u64], &tape),
            f32s(1e-6 * STATE as f32),
            Arg::Scalar(Scalar::Bool(grouped)),
        ],
    }
}

fn recurrent_state(ctx: &Ctx) -> Vec<Variant> {
    let slot = |rows, stop, previous, following| Slot {
        rows,
        stop,
        previous,
        following,
    };
    [("bf16", bf16(), true), ("f32", f32e(), false)]
        .into_iter()
        .map(|(label, a, every)| {
            let mut cases = vec![
                recurrent_case(ctx, a, "m1", 1, &[slot(1, 1, 1, 3)], 5, false, 31),
                recurrent_case(
                    ctx,
                    a,
                    "m8x2",
                    8,
                    &[slot(4, 2, 1, 3), slot(3, 3, 0, 4)],
                    5,
                    false,
                    32,
                ),
                recurrent_case(
                    ctx,
                    a,
                    "m8x8",
                    8,
                    &(0..8).map(|i| slot(1, 1, i % 3, 8 + i)).collect::<Vec<_>>(),
                    16,
                    true,
                    33,
                ),
                recurrent_case(ctx, a, "m16g", 16, &[slot(16, 16, 2, 3)], 5, true, 34),
                recurrent_case(ctx, a, "m128", 128, &[slot(128, 128, 1, 3)], 5, false, 35),
                recurrent_case(
                    ctx,
                    a,
                    "m300x2",
                    300,
                    &[slot(170, 101, 1, 3), slot(130, 0, 2, 4)],
                    5,
                    false,
                    36,
                ),
            ];
            cases.push(recurrent_case(
                ctx,
                a,
                "m20stop0",
                20,
                &[slot(19, 0, 0, 1)],
                3,
                false,
                37,
            ));
            Variant {
                label: label.into(),
                elements: vec![("A", a)],
                statics: vec![
                    ("NK", KEY_HEADS as u64),
                    ("NV", VALUE_HEADS as u64),
                    ("W", STATE as u64),
                    ("C", CONVOLUTION as u64),
                ],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn readout_rows() -> Vec<(usize, usize)> {
    vec![
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 4),
        (8, 8),
        (9, 9),
        (40, 17),
        (64, 64),
    ]
}

fn features_rows(ctx: &Ctx) -> Vec<Variant> {
    [
        ("bf16", bf16(), f32e()),
        ("f16", f16(), bf16()),
        ("f32", f32e(), f16()),
    ]
    .into_iter()
    .map(|(label, a, nw)| {
        let norm = ctx.norm("output_norm.weight", nw);
        let mut rng = Rng::new(7);
        let cases = readout_rows()
            .into_iter()
            .map(|(m, o)| Case {
                label: format!("m{m}o{o}"),
                args: vec![
                    ctx.dense(
                        f32e(),
                        &[m as u64, HIDDEN as u64],
                        &activations(&mut rng, m, HIDDEN, 1.0),
                    ),
                    norm.clone(),
                    ctx.ints(&[o as u64], &out_rows(m, o)),
                    f32s(1e-6),
                ],
            })
            .collect();
        Variant {
            label: label.into(),
            elements: vec![("NW", nw), ("A", a)],
            statics: vec![("D", HIDDEN as u64)],
            every_configuration: true,
            cases,
        }
    })
    .collect()
}

fn head_rows(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, nw, dense, every)| {
            let (weight, wf) = ctx.real("token_embd.weight", VOCABULARY_SLICE, dense);
            let norm = ctx.norm("output_norm.weight", nw);
            let mut rng = Rng::new(8);
            let cases = PROJECTION_ROWS
                .iter()
                .map(|&(m, o)| Case {
                    label: format!("m{m}o{o}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, HIDDEN as u64],
                            &activations(&mut rng, m, HIDDEN, 1.0),
                        ),
                        norm.clone(),
                        Arg::Shared(weight.clone()),
                        ctx.ints(&[o as u64], &out_rows(m, o)),
                        f32s(1e-6),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![
                    ("NW", nw),
                    ("OW", dense.unwrap_or_else(|| ctx.resident(wf))),
                    ("A", a),
                ],
                statics: vec![("V", VOCABULARY_SLICE as u64), ("D", HIDDEN as u64)],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn selected_rows(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, nw, dense, every)| {
            let (weight, wf) = ctx.real("token_embd.weight", VOCABULARY_SLICE, dense);
            let norm = ctx.norm("output_norm.weight", nw);
            let mut rng = Rng::new(9);
            let cases = PROJECTION_ROWS
                .iter()
                .flat_map(|&(m, o)| [(m, o, 37usize), (m, o, 300)])
                .map(|(m, o, sv)| {
                    let selected = (0..sv)
                        .map(|_| rng.below(VOCABULARY_SLICE) as i32)
                        .collect::<Vec<_>>();
                    Case {
                        label: format!("m{m}o{o}s{sv}"),
                        args: vec![
                            ctx.dense(
                                f32e(),
                                &[m as u64, HIDDEN as u64],
                                &activations(&mut rng, m, HIDDEN, 1.0),
                            ),
                            norm.clone(),
                            Arg::Shared(weight.clone()),
                            ctx.ints(&[o as u64], &out_rows(m, o)),
                            ctx.ints(&[sv as u64], &selected),
                            f32s(1e-6),
                        ],
                    }
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![
                    ("NW", nw),
                    ("OW", dense.unwrap_or_else(|| ctx.resident(wf))),
                    ("A", a),
                ],
                statics: vec![("V", VOCABULARY_SLICE as u64), ("D", HIDDEN as u64)],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn head_logits_rows(ctx: &Ctx) -> Vec<Variant> {
    projection_variants()
        .into_iter()
        .map(|(label, a, _, dense, every)| {
            let (weight, wf) = ctx.real("token_embd.weight", VOCABULARY_SLICE, dense);
            let mut rng = Rng::new(10);
            let cases = rows_only()
                .into_iter()
                .map(|o| Case {
                    label: format!("o{o}"),
                    args: vec![
                        ctx.dense(
                            a,
                            &[o as u64, HIDDEN as u64],
                            &activations(&mut rng, o, HIDDEN, 1.0),
                        ),
                        Arg::Shared(weight.clone()),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![("OW", dense.unwrap_or_else(|| ctx.resident(wf))), ("A", a)],
                statics: vec![("V", VOCABULARY_SLICE as u64), ("D", HIDDEN as u64)],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn draft_rows(ctx: &Ctx) -> Vec<Variant> {
    [
        ("bf16", bf16(), f32e(), None, None),
        ("f16", f16(), bf16(), None, Some(Format::Q8)),
        ("dense", bf16(), f32e(), Some(bf16()), None),
        ("f32", f32e(), f32e(), Some(f32e()), Some(Format::Q6K)),
    ]
    .into_iter()
    .map(|(label, a, norm_element, dense, combine_format)| {
        let (table, tf) = ctx.real("token_embd.weight", VOCABULARY_SLICE, dense);
        let combine_format = combine_format.unwrap_or(Format::Q4K);
        let combine = ctx.synthetic(
            combine_format,
            &[HIDDEN as u64, 2 * HIDDEN as u64],
            41,
            dense,
        );
        let element = |f: Format| dense.unwrap_or_else(|| ctx.resident(f));
        let mut rng = Rng::new(11);
        let cases = [1usize, 2, 4, 8, 16]
            .into_iter()
            .map(|m| Case {
                label: format!("m{m}"),
                args: vec![
                    ctx.ints(
                        &[m as u64],
                        &(0..m)
                            .map(|_| rng.below(VOCABULARY_SLICE) as i32)
                            .collect::<Vec<_>>(),
                    ),
                    Arg::Shared(table.clone()),
                    ctx.dense(
                        a,
                        &[m as u64, HIDDEN as u64],
                        &activations(&mut rng, m, HIDDEN, 1.0),
                    ),
                    ctx.norm("output_norm.weight", norm_element),
                    ctx.norm("blk.0.attn_norm.weight", norm_element),
                    Arg::Shared(combine.clone()),
                    f32s(1e-6),
                ],
            })
            .collect();
        Variant {
            label: label.into(),
            elements: vec![
                ("EW", element(tf)),
                ("A", a),
                ("EN", norm_element),
                ("HN", norm_element),
                ("CW", element(combine_format)),
            ],
            statics: vec![],
            every_configuration: true,
            cases,
        }
    })
    .collect()
}

fn embedding_rows(ctx: &Ctx) -> Vec<Variant> {
    [
        ("q6k-bf16", bf16(), None),
        ("q6k-f16", f16(), None),
        ("q6k-f32", f32e(), None),
        ("bf16-bf16", bf16(), Some(bf16())),
        ("f16-f32", f32e(), Some(f16())),
        ("f32-bf16", bf16(), Some(f32e())),
    ]
    .into_iter()
    .map(|(label, a, dense)| {
        let (table, tf) = ctx.real("token_embd.weight", VOCABULARY_SLICE, dense);
        let mut rng = Rng::new(12);
        let cases = [1usize, 3, 8, 64]
            .into_iter()
            .map(|m| Case {
                label: format!("m{m}"),
                args: vec![
                    Arg::Shared(table.clone()),
                    // (token, status) rows, the `sample_rows` result layout.
                    ctx.ints(
                        &[m as u64, 2],
                        &(0..m)
                            .flat_map(|_| [rng.below(VOCABULARY_SLICE) as i32, 0])
                            .collect::<Vec<_>>(),
                    ),
                ],
            })
            .collect();
        Variant {
            label: label.into(),
            elements: vec![("EW", dense.unwrap_or_else(|| ctx.resident(tf))), ("A", a)],
            statics: vec![("D", HIDDEN as u64)],
            every_configuration: true,
            cases,
        }
    })
    .collect()
}

// The routed geometry: Qwen3.5-35B-A3B widths with a reduced expert count
// for the expert entries (routing uses all 256 experts).
const ROUTED_HIDDEN: usize = 2048;
const ROUTED_EXPERTS: usize = 256;
const ROUTED_EXPERTS_SMALL: usize = 64;
const ROUTED_CHOICES: usize = 8;
const ROUTED_FFN: usize = 512;
const ROUTED_SHARED: usize = 512;
const ROUTED_TILE: usize = 32;

/// Distinct experts per row, skewed so some experts fill several tiles.
fn routes(rows: usize, experts: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    let mut routes = Vec::with_capacity(rows * ROUTED_CHOICES);
    for row in 0..rows {
        let mut chosen = Vec::new();
        while chosen.len() < ROUTED_CHOICES {
            let candidate = if row % 3 == 0 {
                rng.below(ROUTED_CHOICES + 3)
            } else {
                rng.below(experts)
            } as i32;
            if !chosen.contains(&candidate) {
                chosen.push(candidate);
            }
        }
        routes.extend(chosen);
    }
    routes
}

fn group_blocks(rows: usize, experts: usize) -> usize {
    (rows * ROUTED_CHOICES + experts * (ROUTED_TILE - 1)).div_ceil(ROUTED_TILE)
}

fn host_group(routes: &[i32], experts: usize, blocks: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut order = vec![-1; blocks * ROUTED_TILE];
    let mut inverse = vec![0; routes.len()];
    let mut table = vec![-1; blocks];
    let mut block = 0;
    for expert in 0..experts as i32 {
        let mut lane = 0;
        for (flat, route) in routes.iter().enumerate() {
            if *route == expert {
                table[block] = expert;
                order[block * ROUTED_TILE + lane] = (flat / ROUTED_CHOICES) as i32;
                inverse[flat] = (block * ROUTED_TILE + lane) as i32;
                lane += 1;
                if lane == ROUTED_TILE {
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

fn routed_route(ctx: &Ctx) -> Vec<Variant> {
    [
        ("bf16", bf16(), f32e(), bf16(), true),
        ("f32", f32e(), bf16(), f32e(), false),
        ("f16", f16(), f32e(), f16(), false),
    ]
    .into_iter()
    .map(|(label, a, nw, rw, every)| {
        let mut rng = Rng::new(13);
        let router = ctx.dense(
            rw,
            &[ROUTED_EXPERTS as u64, ROUTED_HIDDEN as u64],
            &uniform(&mut rng, ROUTED_EXPERTS * ROUTED_HIDDEN, -0.05, 0.05),
        );
        let norm = ctx.dense(
            nw,
            &[ROUTED_HIDDEN as u64],
            &uniform(&mut rng, ROUTED_HIDDEN, 0.5, 1.5),
        );
        let shared = ctx.dense(
            f32e(),
            &[ROUTED_HIDDEN as u64],
            &uniform(&mut rng, ROUTED_HIDDEN, -0.05, 0.05),
        );
        let cases = [1usize, 3, 8, 9, 64]
            .into_iter()
            .flat_map(|m| [(m, 1), (m, 0)])
            .map(|(m, normalize)| {
                let mut residual = activations(&mut rng, m, ROUTED_HIDDEN, 1.0);
                if m > 1 {
                    // Two identical rows exercise routing ties identically.
                    let (first, rest) = residual.split_at_mut(ROUTED_HIDDEN);
                    rest[..ROUTED_HIDDEN].copy_from_slice(first);
                }
                Case {
                    label: format!("m{m}n{normalize}"),
                    args: vec![
                        ctx.dense(f32e(), &[m as u64, ROUTED_HIDDEN as u64], &residual),
                        norm.clone(),
                        router.clone(),
                        shared.clone(),
                        Arg::Mutable {
                            element: Element::i32(),
                            shape: vec![m as u64, ROUTED_CHOICES as u64],
                            bytes: i32_bytes(&vec![-7; m * ROUTED_CHOICES]),
                        },
                        ctx.dense_mut(
                            f32e(),
                            &[m as u64, ROUTED_CHOICES as u64],
                            &vec![-3.0; m * ROUTED_CHOICES],
                        ),
                        f32s(1e-6),
                        i32s(normalize),
                    ],
                }
            })
            .collect();
        Variant {
            label: label.into(),
            elements: vec![("NW", nw), ("RW", rw), ("A", a)],
            statics: vec![
                ("H", ROUTED_HIDDEN as u64),
                ("E", ROUTED_EXPERTS as u64),
                ("K", ROUTED_CHOICES as u64),
            ],
            every_configuration: every,
            cases,
        }
    })
    .collect()
}

/// The routed expert weights: (gate, up, down, shared gate, shared up,
/// shared down) with their elements.
struct RoutedWeights {
    tensors: [Tensor; 6],
    elements: [Element; 6],
}

fn routed_weights(ctx: &Ctx, formats: [Format; 6], dense: Option<Element>) -> RoutedWeights {
    let (e, h, f, s) = (
        ROUTED_EXPERTS_SMALL as u64,
        ROUTED_HIDDEN as u64,
        ROUTED_FFN as u64,
        ROUTED_SHARED as u64,
    );
    let shapes: [Vec<u64>; 6] = [
        vec![e, f, h],
        vec![e, f, h],
        vec![e, h, f],
        vec![s, h],
        vec![s, h],
        vec![h, s],
    ];
    let tensors =
        std::array::from_fn(|i| ctx.synthetic(formats[i], &shapes[i], 50 + i as u64, dense));
    let elements = std::array::from_fn(|i| dense.unwrap_or_else(|| ctx.resident(formats[i])));
    RoutedWeights { tensors, elements }
}

fn routed_weight_variants() -> Vec<(&'static str, Element, [Format; 6], Option<Element>, bool)> {
    use Format::*;
    vec![
        ("35b-mix", bf16(), [Q4K, Q4K, Q5K, Q8, Q8, Q8], None, true),
        ("k-mix", bf16(), [Q5K, Q6K, Q4K, Q4K, Q6K, Q5K], None, false),
        ("f16", f16(), [Q4K, Q4K, Q5K, Q8, Q8, Q8], None, false),
        ("dense-bf16", bf16(), [Q4K; 6], Some(bf16()), false),
    ]
}

fn routed_expand(ctx: &Ctx) -> Vec<Variant> {
    routed_weight_variants()
        .into_iter()
        .map(|(label, a, formats, dense, every)| {
            let weights = routed_weights(ctx, formats, dense);
            let mut rng = Rng::new(14);
            let cases = [1usize, 2, 3, 8]
                .into_iter()
                .map(|m| Case {
                    label: format!("m{m}"),
                    args: vec![
                        ctx.dense(
                            a,
                            &[m as u64, ROUTED_HIDDEN as u64],
                            &activations(&mut rng, m, ROUTED_HIDDEN, 1.0),
                        ),
                        ctx.ints(
                            &[m as u64, ROUTED_CHOICES as u64],
                            &routes(m, ROUTED_EXPERTS_SMALL, 60 + m as u64),
                        ),
                        Arg::Shared(weights.tensors[0].clone()),
                        Arg::Shared(weights.tensors[1].clone()),
                        Arg::Shared(weights.tensors[3].clone()),
                        Arg::Shared(weights.tensors[4].clone()),
                    ],
                })
                .collect();
            let el = weights.elements;
            Variant {
                label: label.into(),
                elements: vec![
                    ("EGW", el[0]),
                    ("EUW", el[1]),
                    ("SGW", el[3]),
                    ("SUW", el[4]),
                    ("A", a),
                ],
                statics: vec![
                    ("H", ROUTED_HIDDEN as u64),
                    ("K", ROUTED_CHOICES as u64),
                    ("F", ROUTED_FFN as u64),
                    ("S", ROUTED_SHARED as u64),
                ],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn routed_output(ctx: &Ctx) -> Vec<Variant> {
    routed_weight_variants()
        .into_iter()
        .map(|(label, a, formats, dense, every)| {
            let weights = routed_weights(ctx, formats, dense);
            let mut rng = Rng::new(15);
            let cases = [1usize, 2, 3, 8]
                .into_iter()
                .map(|m| {
                    let k = ROUTED_CHOICES;
                    Case {
                        label: format!("m{m}"),
                        args: vec![
                            ctx.dense(
                                f32e(),
                                &[m as u64, ROUTED_HIDDEN as u64],
                                &activations(&mut rng, m, ROUTED_HIDDEN, 1.0),
                            ),
                            ctx.dense(
                                a,
                                &[m as u64, k as u64, ROUTED_FFN as u64],
                                &activations(&mut rng, m * k, ROUTED_FFN, 0.3),
                            ),
                            ctx.dense(
                                a,
                                &[m as u64, ROUTED_SHARED as u64],
                                &activations(&mut rng, m, ROUTED_SHARED, 0.3),
                            ),
                            ctx.ints(
                                &[m as u64, k as u64],
                                &routes(m, ROUTED_EXPERTS_SMALL, 70 + m as u64),
                            ),
                            ctx.dense(
                                f32e(),
                                &[m as u64, k as u64],
                                &uniform(&mut rng, m * k, 0.0, 0.3),
                            ),
                            ctx.dense(f32e(), &[m as u64], &uniform(&mut rng, m, 0.1, 0.9)),
                            Arg::Shared(weights.tensors[2].clone()),
                            Arg::Shared(weights.tensors[5].clone()),
                        ],
                    }
                })
                .collect();
            let el = weights.elements;
            Variant {
                label: label.into(),
                elements: vec![("EDW", el[2]), ("SDW", el[5]), ("A", a)],
                statics: vec![
                    ("H", ROUTED_HIDDEN as u64),
                    ("K", ROUTED_CHOICES as u64),
                    ("F", ROUTED_FFN as u64),
                    ("S", ROUTED_SHARED as u64),
                ],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn routed_group(ctx: &Ctx) -> Vec<Variant> {
    let cases = [(1usize, 64usize), (9, 64), (64, 64), (300, 256), (512, 64)]
        .into_iter()
        .map(|(m, e)| {
            let b = group_blocks(m, e);
            let k = ROUTED_CHOICES;
            Case {
                label: format!("m{m}e{e}"),
                args: vec![
                    ctx.ints(&[m as u64, k as u64], &routes(m, e, 80 + m as u64)),
                    Arg::Mutable {
                        element: Element::i32(),
                        shape: vec![e as u64],
                        bytes: i32_bytes(&vec![-5; e]),
                    },
                    Arg::Mutable {
                        element: Element::i32(),
                        shape: vec![b as u64, ROUTED_TILE as u64],
                        bytes: i32_bytes(&vec![-5; b * ROUTED_TILE]),
                    },
                    Arg::Mutable {
                        element: Element::i32(),
                        shape: vec![m as u64, k as u64],
                        bytes: i32_bytes(&vec![-5; m * k]),
                    },
                    Arg::Mutable {
                        element: Element::i32(),
                        shape: vec![b as u64],
                        bytes: i32_bytes(&vec![-5; b]),
                    },
                ],
            }
        })
        .collect::<Vec<_>>();
    // E is static: one variant per expert count.
    let mut variants = Vec::new();
    for e in [64usize, 256] {
        variants.push(Variant {
            label: format!("e{e}"),
            elements: vec![],
            statics: vec![("E", e as u64), ("K", ROUTED_CHOICES as u64)],
            every_configuration: true,
            cases: vec![],
        });
    }
    for case in cases {
        let e = if case.label.ends_with("e256") { 1 } else { 0 };
        variants[e].cases.push(case);
    }
    variants
}

fn routed_experts(ctx: &Ctx) -> Vec<Variant> {
    routed_weight_variants()
        .into_iter()
        .map(|(label, a, formats, dense, every)| {
            let weights = routed_weights(ctx, formats, dense);
            let mut rng = Rng::new(16);
            let e = ROUTED_EXPERTS_SMALL;
            let cases = [9usize, 64, 300]
                .into_iter()
                .map(|m| {
                    let b = group_blocks(m, e);
                    let routes = routes(m, e, 90 + m as u64);
                    let (order, _, blocks) = host_group(&routes, e, b);
                    Case {
                        label: format!("m{m}"),
                        args: vec![
                            ctx.dense(
                                a,
                                &[m as u64, ROUTED_HIDDEN as u64],
                                &activations(&mut rng, m, ROUTED_HIDDEN, 1.0),
                            ),
                            ctx.ints(&[b as u64, ROUTED_TILE as u64], &order),
                            ctx.ints(&[b as u64], &blocks),
                            Arg::Shared(weights.tensors[0].clone()),
                            Arg::Shared(weights.tensors[1].clone()),
                            Arg::Shared(weights.tensors[2].clone()),
                        ],
                    }
                })
                .collect();
            let el = weights.elements;
            Variant {
                label: label.into(),
                elements: vec![("EGW", el[0]), ("EUW", el[1]), ("EDW", el[2]), ("A", a)],
                statics: vec![("H", ROUTED_HIDDEN as u64), ("F", ROUTED_FFN as u64)],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

fn routed_combine(ctx: &Ctx) -> Vec<Variant> {
    routed_weight_variants()
        .into_iter()
        .map(|(label, a, formats, dense, every)| {
            let weights = routed_weights(ctx, formats, dense);
            let mut rng = Rng::new(17);
            let e = ROUTED_EXPERTS_SMALL;
            let k = ROUTED_CHOICES;
            let cases = [9usize, 64, 300]
                .into_iter()
                .map(|m| {
                    let b = group_blocks(m, e);
                    let routes = routes(m, e, 100 + m as u64);
                    let (_, inverse, _) = host_group(&routes, e, b);
                    Case {
                        label: format!("m{m}"),
                        args: vec![
                            ctx.dense(
                                f32e(),
                                &[m as u64, ROUTED_HIDDEN as u64],
                                &activations(&mut rng, m, ROUTED_HIDDEN, 1.0),
                            ),
                            ctx.dense(
                                a,
                                &[b as u64, ROUTED_TILE as u64, ROUTED_HIDDEN as u64],
                                &activations(&mut rng, b * ROUTED_TILE, ROUTED_HIDDEN, 0.2),
                            ),
                            ctx.ints(&[m as u64, k as u64], &inverse),
                            ctx.dense(
                                f32e(),
                                &[m as u64, k as u64],
                                &uniform(&mut rng, m * k, 0.0, 0.3),
                            ),
                            ctx.dense(
                                a,
                                &[m as u64, ROUTED_HIDDEN as u64],
                                &activations(&mut rng, m, ROUTED_HIDDEN, 1.0),
                            ),
                            ctx.dense(f32e(), &[m as u64], &uniform(&mut rng, m, 0.1, 0.9)),
                            Arg::Shared(weights.tensors[3].clone()),
                            Arg::Shared(weights.tensors[4].clone()),
                            Arg::Shared(weights.tensors[5].clone()),
                        ],
                    }
                })
                .collect();
            let el = weights.elements;
            Variant {
                label: label.into(),
                elements: vec![("SGW", el[3]), ("SUW", el[4]), ("SDW", el[5]), ("A", a)],
                statics: vec![
                    ("H", ROUTED_HIDDEN as u64),
                    ("K", k as u64),
                    ("S", ROUTED_SHARED as u64),
                ],
                every_configuration: every,
                cases,
            }
        })
        .collect()
}

/// Logits with ties, -inf and a NaN-free spread.
fn logits(rng: &mut Rng, rows: usize, vocabulary: usize) -> Vec<f32> {
    (0..rows * vocabulary)
        .map(|i| match i % 97 {
            0 => f32::NEG_INFINITY,
            1 | 2 => 3.5,
            _ => (rng.unit() * 8.0 * 64.0).round() / 64.0,
        })
        .collect()
}

fn shape_rows(ctx: &Ctx) -> Vec<Variant> {
    // Rows of params: temperature, top_k, top_p, min_p, repetition,
    // presence, frequency, unused.
    let params: [[f32; 8]; 6] = [
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.7, 40.0, 0.9, 0.05, 1.1, 0.2, 0.1, 0.0],
        [1.0, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.3, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.6, 500.0, 0.95, 0.1, 1.3, 0.0, 0.5, 0.0],
        [1.0, 3.0, 0.5, 0.2, 0.9, 1.0, 0.0, 0.0],
    ];
    [1000usize, 248320]
        .into_iter()
        .map(|v| {
            let mut rng = Rng::new(18 + v as u64);
            let cases = [1usize, 6]
                .into_iter()
                .map(|sx| {
                    let hn = 24;
                    let history = (0..sx * hn)
                        .map(|i| {
                            if i % 5 == 4 {
                                -1
                            } else {
                                rng.below(v.min(300)) as i32
                            }
                        })
                        .collect::<Vec<_>>();
                    let rows = (0..sx)
                        .flat_map(|r| params[(r + 1) % 6])
                        .collect::<Vec<_>>();
                    Case {
                        label: format!("s{sx}"),
                        args: vec![
                            ctx.dense(f32e(), &[sx as u64, v as u64], &logits(&mut rng, sx, v)),
                            ctx.dense(f32e(), &[sx as u64, 8], &rows),
                            ctx.ints(&[sx as u64, hn as u64], &history),
                            ctx.dense_mut(f32e(), &[sx as u64, v as u64], &vec![11.0; sx * v]),
                        ],
                    }
                })
                .collect();
            Variant {
                label: format!("v{v}"),
                elements: vec![],
                statics: vec![("V", v as u64), ("Hn", 24)],
                every_configuration: true,
                cases,
            }
        })
        .collect()
}

fn sample_rows(ctx: &Ctx) -> Vec<Variant> {
    [1000usize, 248320]
        .into_iter()
        .map(|v| {
            let mut rng = Rng::new(19 + v as u64);
            let words = v.div_ceil(32);
            let cases = [1usize, 5]
                .into_iter()
                .map(|m| {
                    let mask = (0..m * words)
                        .map(|_| rng.next() | 0x0101_0101)
                        .collect::<Vec<u32>>();
                    let constrained = (0..m).map(|r| (r % 2) as i32).collect::<Vec<_>>();
                    let draws = (0..m)
                        .flat_map(|r| {
                            [
                                if r % 3 == 0 { 0 } else { 1 },
                                1234 + r as u32,
                                99,
                                r as u32,
                                7,
                                0,
                            ]
                        })
                        .collect::<Vec<u32>>();
                    Case {
                        label: format!("m{m}"),
                        args: vec![
                            ctx.dense(f32e(), &[m as u64, v as u64], &logits(&mut rng, m, v)),
                            Arg::Shared(ctx.tensor(
                                Element::u32(),
                                &[m as u64, words as u64],
                                &u32_bytes(&mask),
                            )),
                            ctx.ints(&[m as u64], &constrained),
                            Arg::Shared(ctx.tensor(
                                Element::u32(),
                                &[m as u64, 6],
                                &u32_bytes(&draws),
                            )),
                            Arg::Mutable {
                                element: Element::i32(),
                                shape: vec![m as u64, 2],
                                bytes: i32_bytes(&vec![-9; 2 * m]),
                            },
                        ],
                    }
                })
                .collect();
            Variant {
                label: format!("v{v}"),
                elements: vec![],
                statics: vec![("V", v as u64)],
                every_configuration: true,
                cases,
            }
        })
        .collect()
}

fn import_dense(ctx: &Ctx) -> Vec<Variant> {
    [
        (f32e(), bf16()),
        (f32e(), f16()),
        (bf16(), f32e()),
        (f16(), f32e()),
        (bf16(), bf16()),
        (f32e(), f32e()),
        (f16(), bf16()),
    ]
    .into_iter()
    .map(|(e, u)| {
        let mut rng = Rng::new(20);
        let cases = [[1u64, 1, 2560], [1, 37, 768], [3, 16, 512]]
            .into_iter()
            .map(|shape| {
                let n = shape.iter().product::<u64>() as usize;
                let mut values = activations(&mut rng, 1, n, 3.0);
                values[0] = 1.0e30;
                values[1] = -0.0;
                values[2] = 7.0e-41;
                Case {
                    label: format!("{shape:?}"),
                    args: vec![ctx.dense(e, &shape, &values)],
                }
            })
            .collect();
        Variant {
            label: format!("{}-{}", e.name(), u.name()),
            elements: vec![("E", e), ("U", u)],
            statics: vec![],
            every_configuration: true,
            cases,
        }
    })
    .collect()
}

fn repack_weight(ctx: &Ctx) -> Vec<Variant> {
    use Format::*;
    [Q4K, Q5K, Q6K, Q8, Iq4]
        .into_iter()
        .map(|format| {
            let mut rng = Rng::new(21);
            let cases = [[1u64, 48, 512], [1, 37, 768], [3, 16, 256], [1, 1, 2560]]
                .into_iter()
                .map(|shape| {
                    let n = shape.iter().product::<u64>() as usize;
                    let external = (0..n / format.block_values())
                        .flat_map(|_| format.block(&mut rng))
                        .collect::<Vec<_>>();
                    Case {
                        label: format!("{shape:?}"),
                        args: vec![Arg::Shared(ctx.tensor(
                            format.external(),
                            &shape,
                            &external,
                        ))],
                    }
                })
                .collect();
            Variant {
                label: format.representation().into(),
                elements: vec![("E", format.external()), ("U", ctx.resident(format))],
                statics: vec![],
                every_configuration: true,
                cases,
            }
        })
        .collect()
}

fn copy_rows(ctx: &Ctx) -> Vec<Variant> {
    [Element::u32(), bf16(), f16(), f32e(), Element::i32()]
        .into_iter()
        .map(|element| {
            let mut rng = Rng::new(22);
            let cases = [
                (2usize, 6usize, 5usize, 4usize, 64usize),
                (5, 40, 40, 4, 256),
                (1, 3, 9, 1, 12),
            ]
            .into_iter()
            .map(|(n, ts, td, kv, w)| {
                let bytes_per = if element.name() == "bf16" || element.name() == "f16" {
                    2
                } else {
                    4
                };
                let src = (0..ts * kv * w * bytes_per)
                    .map(|_| rng.byte())
                    .collect::<Vec<_>>();
                let dst = (0..td * kv * w * bytes_per)
                    .map(|_| rng.byte())
                    .collect::<Vec<_>>();
                let from = (0..n)
                    .map(|i| ((i * 7 + 1) % ts) as i32)
                    .collect::<Vec<_>>();
                let to = (0..n)
                    .map(|i| ((i * 3 + 2) % td) as i32)
                    .collect::<Vec<_>>();
                Case {
                    label: format!("n{n}ts{ts}td{td}kv{kv}w{w}"),
                    args: vec![
                        Arg::Shared(ctx.tensor(element, &[ts as u64, kv as u64, w as u64], &src)),
                        Arg::Mutable {
                            element,
                            shape: vec![td as u64, kv as u64, w as u64],
                            bytes: dst,
                        },
                        ctx.ints(&[n as u64], &from),
                        ctx.ints(&[n as u64], &to),
                    ],
                }
            })
            .collect();
            Variant {
                label: element.name().into(),
                elements: vec![("A", element)],
                statics: vec![],
                every_configuration: true,
                cases,
            }
        })
        .collect()
}

fn conditioning_overlay(ctx: &Ctx) -> Vec<Variant> {
    let mut rng = Rng::new(23);
    let cases = [(1usize, 2560usize), (7, 2560), (3, 100)]
        .into_iter()
        .map(|(m, d)| Case {
            label: format!("m{m}d{d}"),
            args: vec![
                ctx.dense(
                    f32e(),
                    &[m as u64, d as u64],
                    &activations(&mut rng, m, d, 2.0),
                ),
                ctx.dense_mut(f32e(), &[m as u64, d as u64], &vec![5.0; m * d]),
            ],
        })
        .collect();
    vec![Variant {
        label: "f32".into(),
        elements: vec![],
        statics: vec![],
        every_configuration: true,
        cases,
    }]
}

fn vision_stem(ctx: &Ctx) -> Vec<Variant> {
    // Qwen3.5 vision stem at a reduced width: C 3, P 16.
    let (c, p, h, l) = (3usize, 16usize, 64usize, 49usize);
    [("f16", f16(), f32e()), ("bf16", bf16(), bf16())]
        .into_iter()
        .map(|(label, w, b)| {
            let mut rng = Rng::new(24);
            let cases = [1usize, 5]
                .into_iter()
                .map(|m| Case {
                    label: format!("m{m}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, c as u64, 2, p as u64, p as u64],
                            &uniform(&mut rng, m * c * 2 * p * p, -1.0, 1.0),
                        ),
                        ctx.dense(
                            w,
                            &[h as u64, c as u64, p as u64, p as u64],
                            &uniform(&mut rng, p * p * c * h, -0.05, 0.05),
                        ),
                        ctx.dense(
                            w,
                            &[h as u64, c as u64, p as u64, p as u64],
                            &uniform(&mut rng, p * p * c * h, -0.05, 0.05),
                        ),
                        ctx.dense(b, &[h as u64], &uniform(&mut rng, h, -0.1, 0.1)),
                        ctx.dense(
                            b,
                            &[l as u64, h as u64],
                            &uniform(&mut rng, h * l, -0.5, 0.5),
                        ),
                        ctx.ints(
                            &[m as u64, 4],
                            &(0..m * 4).map(|i| (i * 5 % l) as i32).collect::<Vec<_>>(),
                        ),
                        ctx.dense(f32e(), &[m as u64, 4], &uniform(&mut rng, m * 4, 0.0, 0.5)),
                    ],
                })
                .collect();
            Variant {
                label: label.into(),
                elements: vec![("W0", w), ("W1", w), ("B", b), ("PE", b)],
                statics: vec![("C", c as u64), ("P", p as u64), ("H", h as u64)],
                every_configuration: true,
                cases,
            }
        })
        .collect()
}

fn vision_block(ctx: &Ctx) -> Vec<Variant> {
    // Two heads of 64 (the kernels specialize on H, P, F).
    let (h, p, f) = (2usize, 16usize, 128usize);
    let width = h * 4 * p;
    [
        ("f16", f16(), f16(), f32e()),
        ("bf16", bf16(), bf16(), bf16()),
    ]
    .into_iter()
    .map(|(label, a, w, n)| {
        let mut rng = Rng::new(25);
        let cases = [2usize, 70]
            .into_iter()
            .map(|m| {
                let mut vector = |len: usize, low: f32, high: f32, element: Element| {
                    ctx.dense(element, &[len as u64], &uniform(&mut rng, len, low, high))
                };
                let n1w = vector(width, 0.5, 1.5, n);
                let n1b = vector(width, -0.1, 0.1, n);
                let qb = vector(3 * width, -0.1, 0.1, n);
                let pb = vector(width, -0.1, 0.1, n);
                let n2w = vector(width, 0.5, 1.5, n);
                let n2b = vector(width, -0.1, 0.1, n);
                let ub = vector(f, -0.1, 0.1, n);
                let db = vector(width, -0.1, 0.1, n);
                Case {
                    label: format!("m{m}"),
                    args: vec![
                        ctx.dense(
                            f32e(),
                            &[m as u64, h as u64, 4, p as u64],
                            &uniform(&mut rng, m * width, -1.0, 1.0),
                        ),
                        ctx.ints(
                            &[m as u64, 2],
                            &(0..m * 2).map(|i| (i % 9) as i32).collect::<Vec<_>>(),
                        ),
                        n1w,
                        n1b,
                        ctx.dense(
                            w,
                            &[3 * width as u64, width as u64],
                            &uniform(&mut rng, width * 3 * width, -0.1, 0.1),
                        ),
                        qb,
                        ctx.dense(
                            w,
                            &[width as u64, width as u64],
                            &uniform(&mut rng, width * width, -0.1, 0.1),
                        ),
                        pb,
                        n2w,
                        n2b,
                        ctx.dense(
                            w,
                            &[f as u64, width as u64],
                            &uniform(&mut rng, width * f, -0.1, 0.1),
                        ),
                        ub,
                        ctx.dense(
                            w,
                            &[width as u64, f as u64],
                            &uniform(&mut rng, f * width, -0.1, 0.1),
                        ),
                        db,
                        f32s(1e-6),
                    ],
                }
            })
            .collect();
        Variant {
            label: label.into(),
            elements: vec![
                ("N1W", n),
                ("N1B", n),
                ("QW", w),
                ("QB", n),
                ("PW", w),
                ("PB", n),
                ("N2W", n),
                ("N2B", n),
                ("UW", w),
                ("UB", n),
                ("DW", w),
                ("DB", n),
                ("A", a),
            ],
            statics: vec![("H", h as u64), ("P", p as u64), ("F", f as u64)],
            every_configuration: true,
            cases,
        }
    })
    .collect()
}

fn vision_merger(ctx: &Ctx) -> Vec<Variant> {
    let (g, h, d) = (4usize, 32usize, 48usize);
    [
        ("f16", f16(), f16(), f32e()),
        ("bf16", bf16(), bf16(), bf16()),
    ]
    .into_iter()
    .map(|(label, a, w, n)| {
        let mut rng = Rng::new(26);
        let cases = [1usize, 3]
            .into_iter()
            .map(|m| Case {
                label: format!("m{m}"),
                args: vec![
                    ctx.dense(
                        f32e(),
                        &[(m * g) as u64, h as u64],
                        &uniform(&mut rng, m * g * h, -1.0, 1.0),
                    ),
                    ctx.dense(n, &[h as u64], &uniform(&mut rng, h, 0.5, 1.5)),
                    ctx.dense(n, &[h as u64], &uniform(&mut rng, h, -0.1, 0.1)),
                    ctx.dense(
                        w,
                        &[(g * h) as u64, (g * h) as u64],
                        &uniform(&mut rng, g * h * g * h, -0.2, 0.2),
                    ),
                    ctx.dense(n, &[(g * h) as u64], &uniform(&mut rng, g * h, -0.1, 0.1)),
                    ctx.dense(
                        w,
                        &[d as u64, (g * h) as u64],
                        &uniform(&mut rng, g * h * d, -0.2, 0.2),
                    ),
                    ctx.dense(n, &[d as u64], &uniform(&mut rng, d, -0.1, 0.1)),
                    f32s(1e-6),
                ],
            })
            .collect();
        Variant {
            label: label.into(),
            elements: vec![
                ("NW", n),
                ("NB", n),
                ("UW", w),
                ("UB", n),
                ("DW", w),
                ("DB", n),
                ("A", a),
            ],
            statics: vec![("G", g as u64), ("H", h as u64), ("D", d as u64)],
            every_configuration: true,
            cases,
        }
    })
    .collect()
}

/// A harness entry: a stable id (golden file name), the function names it
/// may carry (current first, then names before a rename), its family, and
/// its case builder.
struct EntrySpec {
    id: &'static str,
    names: &'static [&'static str],
    family: &'static str,
    /// Built on the projection library (`packets`/`projection` headers).
    library: bool,
    build: fn(&Ctx) -> Vec<Variant>,
}

const ENTRIES: &[EntrySpec] = &[
    EntrySpec {
        id: "dense_expand",
        names: &["dense_expand"],
        family: "projection",
        library: true,
        build: dense_expand,
    },
    EntrySpec {
        id: "dense_output",
        names: &["dense_output"],
        family: "projection",
        library: true,
        build: dense_output,
    },
    EntrySpec {
        id: "attention_project",
        names: &["gated_attention_project"],
        family: "attention",
        library: true,
        build: attention_project,
    },
    EntrySpec {
        id: "attention_output",
        names: &["attention_output"],
        family: "attention",
        library: true,
        build: attention_output,
    },
    EntrySpec {
        id: "attention_decode",
        names: &["gated_attention_decode"],
        family: "attention",
        library: false,
        build: attention_decode,
    },
    EntrySpec {
        id: "attention_prefill",
        names: &["gated_attention_prefill"],
        family: "attention",
        library: false,
        build: attention_prefill,
    },
    EntrySpec {
        id: "recurrent_project",
        names: &["gated_delta_project"],
        family: "recurrent",
        library: true,
        build: recurrent_project,
    },
    EntrySpec {
        id: "recurrent_output",
        names: &["gated_delta_output"],
        family: "recurrent",
        library: true,
        build: recurrent_output,
    },
    EntrySpec {
        id: "recurrent_step",
        names: &["gated_delta_step"],
        family: "recurrent",
        library: false,
        build: recurrent_state,
    },
    EntrySpec {
        id: "recurrent_chunk",
        names: &["gated_delta_chunk"],
        family: "recurrent",
        library: false,
        build: recurrent_state,
    },
    EntrySpec {
        id: "features_rows",
        names: &["readout_features_rows", "qwen_features_rows"],
        family: "readout",
        library: true,
        build: features_rows,
    },
    EntrySpec {
        id: "head_rows",
        names: &["readout_head_rows", "qwen_head_rows"],
        family: "readout",
        library: true,
        build: head_rows,
    },
    EntrySpec {
        id: "selected_rows",
        names: &["readout_selected_rows", "qwen_selected_rows"],
        family: "readout",
        library: true,
        build: selected_rows,
    },
    EntrySpec {
        id: "head_logits_rows",
        names: &["head_logits_rows"],
        family: "readout",
        library: true,
        build: head_logits_rows,
    },
    EntrySpec {
        id: "draft_rows",
        names: &["draft_rows"],
        family: "readout",
        library: true,
        build: draft_rows,
    },
    EntrySpec {
        id: "embedding_rows",
        names: &["embedding_rows", "qwen_embedding_rows"],
        family: "readout",
        library: true,
        build: embedding_rows,
    },
    EntrySpec {
        id: "routed_route",
        names: &["routed_route"],
        family: "routed",
        library: false,
        build: routed_route,
    },
    EntrySpec {
        id: "routed_expand",
        names: &["routed_expand"],
        family: "routed",
        library: true,
        build: routed_expand,
    },
    EntrySpec {
        id: "routed_output",
        names: &["routed_output"],
        family: "routed",
        library: true,
        build: routed_output,
    },
    EntrySpec {
        id: "routed_group",
        names: &["routed_group"],
        family: "routed",
        library: false,
        build: routed_group,
    },
    EntrySpec {
        id: "routed_experts",
        names: &["routed_experts"],
        family: "routed",
        library: true,
        build: routed_experts,
    },
    EntrySpec {
        id: "routed_combine",
        names: &["routed_combine"],
        family: "routed",
        library: true,
        build: routed_combine,
    },
    EntrySpec {
        id: "shape_rows",
        names: &["shape_rows"],
        family: "sampling",
        library: false,
        build: shape_rows,
    },
    EntrySpec {
        id: "sample_rows",
        names: &["sample_rows"],
        family: "sampling",
        library: false,
        build: sample_rows,
    },
    EntrySpec {
        id: "import_dense",
        names: &["import_dense"],
        family: "import",
        library: false,
        build: import_dense,
    },
    EntrySpec {
        id: "repack_weight",
        names: &["repack_weight"],
        family: "import",
        library: false,
        build: repack_weight,
    },
    EntrySpec {
        id: "copy_rows",
        names: &["copy_rows"],
        family: "state",
        library: false,
        build: copy_rows,
    },
    EntrySpec {
        id: "conditioning_overlay",
        names: &["conditioning_overlay", "qwen_conditioning_overlay"],
        family: "state",
        library: false,
        build: conditioning_overlay,
    },
    EntrySpec {
        id: "vision_stem",
        names: &["qwen_vision_stem"],
        family: "vision",
        library: false,
        build: vision_stem,
    },
    EntrySpec {
        id: "vision_block",
        names: &["qwen_vision_block"],
        family: "vision",
        library: false,
        build: vision_block,
    },
    EntrySpec {
        id: "vision_merger",
        names: &["qwen_vision_merger"],
        family: "vision",
        library: false,
        build: vision_merger,
    },
];

// ---------------------------------------------------------------------------
// Configurations
// ---------------------------------------------------------------------------

/// Every configuration when the domain has at most 48; otherwise the
/// defaults, each single-parameter variation of them and 8 pseudo-random
/// configurations. Inadmissible configurations (`where`) are dropped.
fn configurations(
    native: &NativeImplementation,
    statics: &[(&str, u64)],
    seed: &str,
) -> Vec<Vec<(String, u64)>> {
    let params = &native.params;
    let total = params.iter().map(|p| p.values.len()).product::<usize>();
    let default = params.iter().map(|p| p.values[0]).collect::<Vec<_>>();
    let mut chosen: Vec<Vec<u64>> = Vec::new();
    let push = |config: Vec<u64>, chosen: &mut Vec<Vec<u64>>| {
        if !chosen.contains(&config) {
            chosen.push(config);
        }
    };
    if total <= 48 {
        let mut index = vec![0usize; params.len()];
        loop {
            push(
                index
                    .iter()
                    .zip(params)
                    .map(|(i, p)| p.values[*i])
                    .collect(),
                &mut chosen,
            );
            let mut digit = 0;
            loop {
                if digit == params.len() {
                    break;
                }
                index[digit] += 1;
                if index[digit] < params[digit].values.len() {
                    break;
                }
                index[digit] = 0;
                digit += 1;
            }
            if digit == params.len() {
                break;
            }
        }
    } else {
        push(default.clone(), &mut chosen);
        for (i, parameter) in params.iter().enumerate() {
            for value in &parameter.values[1..] {
                let mut config = default.clone();
                config[i] = *value;
                push(config, &mut chosen);
            }
        }
        let mut rng = Rng::new(
            seed.bytes()
                .fold(7u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b))),
        );
        for _ in 0..8 {
            push(
                params
                    .iter()
                    .map(|p| p.values[rng.below(p.values.len())])
                    .collect(),
                &mut chosen,
            );
        }
    }
    chosen
        .into_iter()
        .map(|config| {
            params
                .iter()
                .map(|p| p.name.clone())
                .zip(config)
                .collect::<Vec<_>>()
        })
        .filter(|config| native.validate(&specialization(statics, config)).is_ok())
        .collect()
}

fn specialization(statics: &[(&str, u64)], config: &[(String, u64)]) -> NativeSpecialization {
    let with_statics = statics
        .iter()
        .fold(NativeSpecialization::new(), |s, (name, value)| {
            s.with_static(*name, *value)
        });
    config.iter().fold(with_statics, |s, (name, value)| {
        s.with_param(name.clone(), *value)
    })
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

fn outputs(function: &Function, args: &[Value], result: Value, out: &mut Vec<(String, Vec<u8>)>) {
    fn flatten(value: Value, index: &mut usize, out: &mut Vec<(String, Vec<u8>)>) {
        match value {
            Value::Unit => {}
            Value::Tuple(values) => {
                for value in values {
                    flatten(value, index, out);
                }
            }
            Value::Tensor(tensor) | Value::Move(tensor) => {
                out.push((format!("r{index}"), tensor.read().unwrap()));
                *index += 1;
            }
            Value::Scalar(scalar) => {
                out.push((format!("r{index}"), format!("{scalar:?}").into_bytes()));
                *index += 1;
            }
        }
    }
    flatten(result, &mut 0, out);
    for ((name, ty), value) in function.parameters().iter().zip(args) {
        if let (
            SignatureType::Tensor {
                access: TensorAccess::Mutable,
                ..
            },
            Value::Tensor(tensor),
        ) = (ty, value)
        {
            out.push((name.clone(), tensor.read().unwrap()));
        }
    }
}

fn run(
    ctx: &Ctx,
    kernel: &seismic::dynamic::Kernel,
    case: &Case,
) -> Result<Vec<(String, String)>, String> {
    let args = case
        .args
        .iter()
        .map(|arg| match arg {
            Arg::Shared(tensor) => Value::Tensor(tensor.clone()),
            Arg::Mutable {
                element,
                shape,
                bytes,
            } => Value::Tensor(ctx.tensor(*element, shape, bytes)),
            Arg::Scalar(scalar) => Value::Scalar(scalar.clone()),
        })
        .collect::<Vec<_>>();
    let result = kernel.call(&args).map_err(|e| format!("ERR {}", e.kind))?;
    let mut out = Vec::new();
    outputs(kernel.function(), &args, result, &mut out);
    Ok(out
        .into_iter()
        .map(|(name, bytes)| (name, hash(&bytes)))
        .collect())
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Mode {
    Record,
    Compare,
}

struct Report {
    compared: usize,
    mismatches: Vec<String>,
    nondeterministic: Vec<String>,
}

fn run_entry(
    ctx: &Ctx,
    checked: &CheckedModule,
    module: &Module,
    spec: &EntrySpec,
    mode: Mode,
    dir: &Path,
    report: &mut Report,
) {
    let Some(name) = spec
        .names
        .iter()
        .copied()
        .find(|name| checked.entry_named(name).is_some())
    else {
        panic!("{}: none of {:?} is an entry", spec.id, spec.names);
    };
    let id = checked.entry_named(name).unwrap();
    let Some(native) = checked.native_implementation(id, ctx.backend) else {
        println!("{}: no {:?} implementation", spec.id, ctx.backend);
        return;
    };
    let function = module.function(name).unwrap();
    let started = std::time::Instant::now();
    let variants = (spec.build)(ctx);
    let mut lines: BTreeMap<String, String> = BTreeMap::new();
    let mut prepared = 0usize;
    for variant in &variants {
        // A variant names the statics of either backend's declaration.
        let statics = variant
            .statics
            .iter()
            .copied()
            .filter(|(name, _)| native.statics.iter().any(|s| s == name))
            .collect::<Vec<_>>();
        let mut configs = configurations(native, &statics, spec.id);
        if !variant.every_configuration {
            configs.truncate(3);
        }
        let elements = variant
            .elements
            .iter()
            .map(|(n, e)| (n.to_string(), *e))
            .collect::<BTreeMap<_, _>>();
        for config in &configs {
            let config_label = config
                .iter()
                .map(|(n, v)| format!("{n}={v}"))
                .collect::<Vec<_>>()
                .join(",");
            let prefix = format!("{}|{}", variant.label, config_label);
            let kernel = match function.prepare_native(
                &ctx.device,
                elements.clone(),
                specialization(&statics, config),
            ) {
                Ok(kernel) => kernel,
                Err(error) => {
                    lines.insert(format!("{prefix}|prepare"), format!("ERR {}", error.kind));
                    if std::env::var_os("GOLDEN_VERBOSE").is_some() {
                        println!("{}: {prefix}: {}", spec.id, error.message);
                    }
                    continue;
                }
            };
            prepared += 1;
            for case in &variant.cases {
                let key = format!("{prefix}|{}", case.label);
                match run(ctx, &kernel, case) {
                    Ok(outputs) => {
                        let again = if mode == Mode::Record {
                            Some(run(ctx, &kernel, case))
                        } else {
                            None
                        };
                        for (output, digest) in &outputs {
                            let stable = match &again {
                                Some(Ok(second)) => {
                                    second.iter().any(|(n, d)| n == output && d == digest)
                                }
                                _ => true,
                            };
                            if !stable {
                                report
                                    .nondeterministic
                                    .push(format!("{}: {key}|{output}", spec.id));
                            }
                            lines.insert(
                                format!("{key}|{output}"),
                                if stable {
                                    digest.clone()
                                } else {
                                    "NONDET".into()
                                },
                            );
                        }
                    }
                    Err(error) => {
                        if std::env::var_os("GOLDEN_VERBOSE").is_some() {
                            println!("{}: {key}: {error}", spec.id);
                        }
                        lines.insert(format!("{key}|call"), error);
                    }
                }
            }
        }
    }
    let path = dir.join(format!("{}.golden", spec.id));
    match mode {
        Mode::Record => {
            let mut text = String::new();
            for (key, value) in &lines {
                writeln!(text, "{key}\t{value}").unwrap();
            }
            std::fs::write(&path, text).unwrap();
        }
        Mode::Compare => {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let golden = text
                .lines()
                .map(|line| {
                    let (key, value) = line.split_once('\t').unwrap();
                    (key.to_string(), value.to_string())
                })
                .collect::<BTreeMap<_, _>>();
            let keys = golden.keys().chain(lines.keys()).collect::<BTreeSet<_>>();
            for key in keys {
                let (expected, actual) = (golden.get(key), lines.get(key));
                report.compared += 1;
                if expected == Some(&"NONDET".to_string()) {
                    continue;
                }
                if expected != actual {
                    report.mismatches.push(format!(
                        "{}: {key}: golden {expected:?} now {actual:?}",
                        spec.id
                    ));
                }
            }
        }
    }
    let errors = lines.values().filter(|v| v.starts_with("ERR")).count();
    println!(
        "{}: {} keys ({errors} error keys), {prepared} configurations prepared, {:.1}s",
        spec.id,
        lines.len(),
        started.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "golden capture/compare; run explicitly with GOLDEN_MODE and GOLDEN_DIR"]
fn golden() {
    let mode = match std::env::var("GOLDEN_MODE").as_deref() {
        Ok("record") => Mode::Record,
        Ok("compare") => Mode::Compare,
        other => panic!("GOLDEN_MODE must be record or compare, got {other:?}"),
    };
    let catalog = DeviceCatalog::discover().unwrap();
    // `GOLDEN_BACKEND=vulkan` selects Vulkan (never chosen by default: a CUDA
    // host also has a Vulkan device).
    let backends = match std::env::var("GOLDEN_BACKEND").ok().as_deref() {
        Some("vulkan") => vec![BackendName::Vulkan],
        Some("cuda") => vec![BackendName::Cuda],
        Some("metal") => vec![BackendName::Metal],
        Some(other) => panic!("GOLDEN_BACKEND={other}: expected metal, cuda or vulkan"),
        None => vec![BackendName::Metal, BackendName::Cuda],
    };
    let (backend, device) = backends
        .into_iter()
        .find_map(|backend| {
            catalog
                .open_backend(backend)
                .ok()
                .map(|device| (backend, device))
        })
        .expect("a device of the selected backend");
    let dir = PathBuf::from(std::env::var("GOLDEN_DIR").expect("GOLDEN_DIR"))
        .join(format!("{backend:?}").to_lowercase());
    std::fs::create_dir_all(&dir).unwrap();
    let families = std::env::var("GOLDEN_FAMILIES").unwrap_or_else(|_| "all".into());
    let families = families.split(',').map(str::trim).collect::<Vec<_>>();
    let entries = std::env::var("GOLDEN_ENTRIES")
        .ok()
        .filter(|list| !list.is_empty());
    let entries = entries
        .as_deref()
        .map(|list| list.split(',').map(str::trim).collect::<Vec<_>>());
    let gguf = std::env::var_os("GOLDEN_GGUF").map(|path| Gguf::open(Path::new(&path)));
    let ctx = Ctx {
        device,
        backend,
        gguf,
        weights: Default::default(),
    };
    // A harness built from a snapshot of the tree may check the live sources.
    let kernels = std::env::var_os("GOLDEN_KERNELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"));
    // `GOLDEN_SOURCES` names the declaration files to load (so another
    // family's in-progress declarations cannot block a run); default: all.
    let sources = match std::env::var("GOLDEN_SOURCES")
        .ok()
        .filter(|list| !list.is_empty())
    {
        Some(list) => list
            .split(',')
            .map(|name| kernels.join(name.trim()))
            .collect::<Vec<_>>(),
        None => vec![kernels],
    };
    let checked = seismic_lang::source::load(&sources, seismic_std::sources())
        .unwrap()
        .module;
    let module = Module::load(&sources, true).unwrap();
    let mut report = Report {
        compared: 0,
        mismatches: Vec::new(),
        nondeterministic: Vec::new(),
    };
    for spec in ENTRIES {
        let selected = families.iter().any(|family| {
            *family == "all" || *family == spec.family || (*family == "library" && spec.library)
        }) && entries.as_ref().is_none_or(|list| list.contains(&spec.id));
        if selected {
            run_entry(&ctx, &checked, &module, spec, mode, &dir, &mut report);
        }
    }
    for line in &report.nondeterministic {
        println!("NONDETERMINISTIC {line}");
    }
    if mode == Mode::Compare {
        for line in &report.mismatches {
            println!("MISMATCH {line}");
        }
        println!(
            "compared {} keys, {} mismatches",
            report.compared,
            report.mismatches.len()
        );
        assert!(
            report.mismatches.is_empty(),
            "{} golden mismatches",
            report.mismatches.len()
        );
    }
}
