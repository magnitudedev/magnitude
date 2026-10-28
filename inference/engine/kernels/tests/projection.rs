//! K1/K2 projection entries on Metal (macOS) and Vulkan (elsewhere). Weights
//! are built host-side in the registry's `rows16` layout; every entry is
//! checked against its portable body (the Seismic interpreter) on small
//! shapes, and against a host reference over the row classes M in {1, 2, 3,
//! 8, 16, 64, 128}, across weight representations and mapping parameters.

use magnitude_kernels::{
    attention_output, attention_project, dense_expand, dense_output, dense_up, embedding_rows,
    gated_delta_project, packing_bound, project_rows, readout_features_rows, readout_head_rows,
    readout_project_rows, readout_selected_rows,
};
use seismic::{BackendName, Device, Element, NativeSpecialization, Tensor};
use seismic_lang::{
    checked::{CheckedModule, SourceFile, check_source},
    entry::ElementBindings,
    failure::SourceTermination,
    ids::RepresentationId,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry::{self, RepresentationKind},
    types::DType,
};

mod operands;

// ---------------------------------------------------------------------------
// Host values.

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * if exponent == 0 {
        mantissa * 2f32.powi(-24)
    } else {
        (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

fn f16_bits(value: f32) -> u16 {
    registry::f16_bits(value)
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    Bf16,
    F16,
}

impl Act {
    fn round(self, value: f32) -> f32 {
        match self {
            Act::Bf16 => bf16_round(value),
            Act::F16 => f16_to_f32(f16_bits(value)),
        }
    }
    fn element(self) -> Element {
        match self {
            Act::Bf16 => Element::bf16(),
            Act::F16 => Element::f16(),
        }
    }
    fn dtype(self) -> DType {
        match self {
            Act::Bf16 => DType::BF16,
            Act::F16 => DType::F16,
        }
    }
    fn bytes(self, values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| match self {
                Act::Bf16 => ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes(),
                Act::F16 => f16_bits(*value).to_le_bytes(),
            })
            .collect()
    }
    fn decode(self, bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|pair| {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                match self {
                    Act::Bf16 => f32::from_bits(u32::from(bits) << 16),
                    Act::F16 => f16_to_f32(bits),
                }
            })
            .collect()
    }
    /// Relative size of one unit in the last place.
    fn ulp(self) -> f32 {
        match self {
            Act::Bf16 => 1.0 / 128.0,
            Act::F16 => 1.0 / 1024.0,
        }
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        )
    }
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn symmetric(&mut self) -> f32 {
        self.uniform() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// rows16 weights.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Repr {
    Q4k,
    Q5k,
    Q6k,
    Q8,
    Bf16,
    // Built from GGUF blocks through the registry's conversion
    // (`gguf_weight`).
    Iq4xs,
    Q4g32s,
    Q4g32,
    Iq4g32s,
    Q5g32s,
    Q5g32,
    Mxfp4,
    Nvfp4,
    /// NVFP4's W4A4 element: nvfp4g16's storage, NVFP4 operands.
    Nvfp4A4,
    Iq2xs,
    Iq3xxs,
    Iq2xxs,
    Q2k,
    Q3k,
}

/// The vector-coded formats (256-value GGUF blocks).
const GRID_REPRS: [Repr; 3] = [Repr::Iq2xs, Repr::Iq3xxs, Repr::Iq2xxs];

/// The k-quant formats imported from GGUF blocks of 256 values.
const K_QUANT_REPRS: [Repr; 2] = [Repr::Q2k, Repr::Q3k];

/// The formats with a representation of their own, imported from GGUF.
const GGUF_REPRS: [Repr; 8] = [
    Repr::Q4g32s,
    Repr::Q4g32,
    Repr::Iq4g32s,
    Repr::Q5g32s,
    Repr::Q5g32,
    Repr::Mxfp4,
    Repr::Nvfp4,
    Repr::Nvfp4A4,
];

impl Repr {
    fn name(self) -> &'static str {
        match self {
            Repr::Q4k => "q4k",
            Repr::Q5k => "q5k",
            Repr::Q6k => "q6k",
            Repr::Q8 => "q8g32s",
            Repr::Iq4xs => "iq4xs",
            Repr::Bf16 => "bf16",
            Repr::Q4g32s => "q4g32s",
            Repr::Q4g32 => "q4g32",
            Repr::Iq4g32s => "iq4g32s",
            Repr::Q5g32s => "q5g32s",
            Repr::Q5g32 => "q5g32",
            Repr::Mxfp4 => "mxfp4g32",
            Repr::Nvfp4 => "nvfp4g16",
            Repr::Nvfp4A4 => "nvfp4g16a4",
            Repr::Iq2xs => "iq2xs",
            Repr::Iq3xxs => "iq3xxs",
            Repr::Iq2xxs => "iq2xxs",
            Repr::Q2k => "q2k",
            Repr::Q3k => "q3k",
        }
    }
    /// The GGUF source of a representation built from GGUF blocks, with its
    /// block bytes and values.
    fn gguf_source(self) -> Option<(&'static str, usize, usize)> {
        match self {
            Repr::Iq4xs => Some(("gguf_iq4_xs", 136, 256)),
            Repr::Q4g32s => Some(("gguf_q4_0", 18, 32)),
            Repr::Q4g32 => Some(("gguf_q4_1", 20, 32)),
            Repr::Iq4g32s => Some(("gguf_iq4_nl", 18, 32)),
            Repr::Q5g32s => Some(("gguf_q5_0", 22, 32)),
            Repr::Q5g32 => Some(("gguf_q5_1", 24, 32)),
            Repr::Mxfp4 => Some(("gguf_mxfp4", 17, 32)),
            Repr::Nvfp4 | Repr::Nvfp4A4 => Some(("gguf_nvfp4", 36, 64)),
            Repr::Iq2xs => Some(("gguf_iq2_xs", 74, 256)),
            Repr::Iq3xxs => Some(("gguf_iq3_xxs", 98, 256)),
            Repr::Iq2xxs => Some(("gguf_iq2_xxs", 66, 256)),
            Repr::Q2k => Some(("gguf_q2_k", 84, 256)),
            Repr::Q3k => Some(("gguf_q3_k", 110, 256)),
            _ => None,
        }
    }
    fn storage(self) -> RepresentationId {
        match self {
            Repr::Bf16 => registry::dense(DType::BF16),
            _ => registry::storage(self.name(), registry::Layout::Rows16).expect("rows16 storage"),
        }
    }
    /// The element of the host-built bytes (`rows16`).
    fn element(self) -> Element {
        self.element_in(registry::Layout::Rows16)
    }
    fn element_in(self, layout: registry::Layout) -> Element {
        match self {
            Repr::Bf16 => Element::bf16(),
            _ => Element::stored(self.name(), layout).expect("row layout element"),
        }
    }
    /// The element `device`'s native kernels read this weight as.
    fn resident(self, device: &Device) -> Element {
        self.element_in(resident_layout(device))
    }
    /// Whether this format's packed forms take two residue bits (codes of
    /// 8-bit values, as the executor's `packing_residue` decides).
    fn takes_residue(self, device: &Device) -> bool {
        self.resident(device)
            .integer_codes()
            .is_some_and(|codes| codes.max - codes.min > i32::from(i8::MAX))
    }
}

/// The row layout `device`'s native kernels read packed weights in.
fn resident_layout(device: &Device) -> registry::Layout {
    device.backend().resident_layout()
}

/// Plane byte offsets of one row, in the frozen rows16 order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Planes {
    stride: usize,
    codes: usize,
    high: usize,
    scales: usize,
    supers: usize,
}

fn planes(repr: Repr, k: usize) -> Planes {
    let blocks = k / 256;
    let mut planes = Planes::default();
    let mut at = 0;
    let mut place = |bytes: usize| {
        let offset = at;
        at += bytes.div_ceil(16) * 16;
        offset
    };
    match repr {
        Repr::Q4k => {
            planes.codes = place(k / 2);
            planes.scales = place(12 * blocks);
            planes.supers = place(4 * blocks);
        }
        Repr::Q5k => {
            planes.codes = place(k / 2);
            planes.high = place(k / 8);
            planes.scales = place(12 * blocks);
            planes.supers = place(4 * blocks);
        }
        Repr::Q6k => {
            planes.codes = place(k / 2);
            planes.high = place(k / 4);
            planes.scales = place(k / 16);
            planes.supers = place(2 * blocks);
        }
        Repr::Q8 => {
            planes.codes = place(k);
            planes.supers = place(2 * (k / 32));
        }
        Repr::Bf16 => at = 2 * k,
        Repr::Iq4xs
        | Repr::Q4g32s
        | Repr::Q4g32
        | Repr::Iq4g32s
        | Repr::Q5g32s
        | Repr::Q5g32
        | Repr::Mxfp4
        | Repr::Nvfp4
        | Repr::Nvfp4A4
        | Repr::Iq2xs
        | Repr::Iq3xxs
        | Repr::Iq2xxs
        | Repr::Q2k
        | Repr::Q3k => {
            unreachable!("{repr:?} takes the registry's geometry (`gguf_weight`)")
        }
    }
    planes.stride = at;
    planes
}

struct Weight {
    repr: Repr,
    rows: usize,
    k: usize,
    bytes: Vec<u8>,
    /// Decoded logical values [rows, k].
    values: Vec<f32>,
}

/// An absent packing bound (`<weight>_bound`, extent 0).
fn absent_bound(device: &Device) -> Tensor {
    f32_tensor(device, &[0], &[])
}

/// An absent accumulator-scale port (static extent 0).
fn no_scale(device: &Device) -> Tensor {
    f32_tensor(device, &[0], &[])
}

impl Weight {
    /// Rows `x` of `k` as products over this weight take them: as NVFP4
    /// operands where the registry's recipe of its representation says so.
    fn operand_rows(&self, x: &[f32]) -> Vec<f32> {
        let representation = registry::representation(self.repr.name()).unwrap();
        let nvfp4 = registry::representation_info(representation).operands
            == registry::OperandRecipe::Nvfp4Rows;
        operands::operand_rows(x, self.k, nvfp4)
    }

    /// The packing bound operand of this weight on `device` and its extent,
    /// as loading makes it (the executor's `packing_run`): on Metal,
    /// `packing_bound` of a resident weight with integer codes whose runs
    /// shorter than a 32-column block are unbiased (one bound per run of the
    /// registry's code group); an absent bound elsewhere.
    fn bound(&self, device: &Device) -> (Tensor, usize) {
        let resident = self.repr.resident(device);
        let Some(run) = resident.integer_codes().and_then(|codes| codes.packing_run()) else {
            return (absent_bound(device), 0);
        };
        if device.backend() != BackendName::Metal {
            return (absent_bound(device), 0);
        }
        let runs = self.k / run as usize;
        let mut bound = f32_tensor(device, &[runs], &vec![0.0; runs]);
        packing_bound::native_for_device_with(
            device,
            packing_bound::Elements { W: resident },
            &NativeSpecialization::new(),
        )
        .unwrap()
        .call(packing_bound::Args {
            weight: &self.tensor(device),
            bound: &mut bound,
        })
        .unwrap();
        (bound, runs)
    }

    /// This weight in `device`'s resident layout.
    fn tensor(&self, device: &Device) -> Tensor {
        resident_tensor(device, self.repr, [self.rows, self.k], &self.bytes)
    }
    fn oracle(&self) -> TensorData {
        match self.repr {
            Repr::Bf16 => TensorData::dense(
                DType::BF16,
                vec![self.rows, self.k],
                self.values.iter().map(|value| f64::from(*value)).collect(),
            ),
            _ => TensorData::encoded(
                self.repr.storage(),
                vec![self.rows, self.k],
                self.bytes.clone(),
            )
            .unwrap(),
        }
    }
    fn row(&self, n: usize) -> &[f32] {
        &self.values[n * self.k..(n + 1) * self.k]
    }
}

/// `repr`'s host-built `rows16` bytes of a `[rows, k]` matrix placed in
/// `device`'s resident layout through the registry's packet form.
fn resident_tensor(device: &Device, repr: Repr, [rows, k]: [usize; 2], bytes: &[u8]) -> Tensor {
    let layout = resident_layout(device);
    let shape = [rows as u64, k as u64];
    if repr == Repr::Bf16 || layout == registry::Layout::Rows16 {
        return Tensor::from_host(device, repr.element(), &shape, bytes).unwrap();
    }
    let registry::RepresentationKind::PackedRows(source) =
        &registry::representation_info(repr.storage()).kind
    else {
        unreachable!()
    };
    let resident = registry::representation_info(registry::storage(repr.name(), layout).unwrap());
    let registry::RepresentationKind::PackedRows(destination) = &resident.kind else {
        unreachable!()
    };
    let packets = source.packets(&shape, bytes);
    let bytes = destination.place(&shape, &packets);
    Tensor::from_host(device, repr.element_in(layout), &shape, &bytes).unwrap()
}

fn put_bits(bytes: &mut [u8], bit: usize, width: usize, value: u32) {
    for i in 0..width {
        if (value >> i) & 1 != 0 {
            bytes[(bit + i) / 8] |= 1 << ((bit + i) % 8);
        }
    }
}

/// A random weight whose rows have dot products of order `scale` with unit
/// activations.
fn weight(repr: Repr, rows: usize, k: usize, seed: u64, scale: f32) -> Weight {
    let scale = scale * 2.0 / (k as f32).sqrt();
    let mut rng = Rng::new(seed);
    if repr.gguf_source().is_some() {
        return gguf_weight(repr, rows, k, &mut rng, scale);
    }
    let p = planes(repr, k);
    let mut bytes = vec![0u8; p.stride * rows];
    let mut values = vec![0f32; rows * k];
    for r in 0..rows {
        let row = &mut bytes[r * p.stride..(r + 1) * p.stride];
        let out = &mut values[r * k..(r + 1) * k];
        match repr {
            Repr::Q4k | Repr::Q5k => {
                let bits = if repr == Repr::Q4k { 4 } else { 5 };
                let center = if bits == 5 { 15.5 } else { 7.5 };
                for block in 0..k / 256 {
                    let d_value = scale / (center * 32.0) * (0.5 + rng.uniform());
                    let d = f16_bits(d_value);
                    let dmin = f16_bits(d_value * center * (0.75 + 0.5 * rng.uniform()));
                    row[p.supers + 4 * block..p.supers + 4 * block + 2]
                        .copy_from_slice(&d.to_le_bytes());
                    row[p.supers + 4 * block + 2..p.supers + 4 * block + 4]
                        .copy_from_slice(&dmin.to_le_bytes());
                    for group in 0..8 {
                        let local = rng.below(64);
                        let minimum = rng.below(64);
                        let fields = &mut row[p.scales + 12 * block..p.scales + 12 * block + 12];
                        put_bits(fields, 12 * group, 6, local);
                        put_bits(fields, 12 * group + 6, 6, minimum);
                        let group_scale = f16_to_f32(d) * local as f32;
                        let group_bias = -(f16_to_f32(dmin) * minimum as f32);
                        for i in 0..32 {
                            let column = 256 * block + 32 * group + i;
                            let code = rng.below(1 << bits);
                            put_bits(&mut row[p.codes..], 4 * column, 4, code & 15);
                            if bits == 5 {
                                put_bits(&mut row[p.high..], column, 1, code >> 4);
                            }
                            out[column] = group_scale.mul_add(code as f32, group_bias);
                        }
                    }
                }
            }
            Repr::Q6k => {
                for block in 0..k / 256 {
                    let d = f16_bits(scale / 32.0 / 48.0 * (0.5 + rng.uniform()));
                    row[p.supers + 2 * block..p.supers + 2 * block + 2]
                        .copy_from_slice(&d.to_le_bytes());
                    for group in 0..16 {
                        let local = (rng.below(255) as i32 - 127) as i8;
                        row[p.scales + 16 * block + group] = local as u8;
                        let group_scale = f16_to_f32(d) * f32::from(local);
                        for i in 0..16 {
                            let column = 256 * block + 16 * group + i;
                            let code = rng.below(64);
                            put_bits(&mut row[p.codes..], 4 * column, 4, code & 15);
                            put_bits(&mut row[p.high..], 2 * column, 2, code >> 4);
                            out[column] = group_scale * (code as i32 - 32) as f32;
                        }
                    }
                }
            }
            Repr::Q8 => {
                for group in 0..k / 32 {
                    let factor = f16_bits(scale / 96.0 * (0.5 + rng.uniform()));
                    row[p.supers + 2 * group..p.supers + 2 * group + 2]
                        .copy_from_slice(&factor.to_le_bytes());
                    for i in 0..32 {
                        let column = 32 * group + i;
                        let code = (rng.below(255) as i32 - 127) as i8;
                        row[p.codes + column] = code as u8;
                        out[column] = f16_to_f32(factor) * f32::from(code);
                    }
                }
            }
            Repr::Bf16 => {
                for column in 0..k {
                    let value = bf16_round(rng.symmetric() * scale);
                    row[2 * column..2 * column + 2]
                        .copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                    out[column] = value;
                }
            }
            Repr::Iq4xs
            | Repr::Q4g32s
            | Repr::Q4g32
            | Repr::Iq4g32s
            | Repr::Q5g32s
            | Repr::Q5g32
            | Repr::Mxfp4
            | Repr::Nvfp4
            | Repr::Nvfp4A4
            | Repr::Iq2xs
            | Repr::Iq3xxs
            | Repr::Iq2xxs
            | Repr::Q2k
            | Repr::Q3k => {
                unreachable!("built by `gguf_weight`")
            }
        }
    }
    Weight {
        repr,
        rows,
        k,
        bytes,
        values,
    }
}

/// A random weight of a representation imported from GGUF: random blocks of
/// its source format (with scale fields of magnitude about `scale / 8`)
/// through the registry's conversion, so the rows16 bytes are the ones the
/// device repack produces.
fn gguf_weight(repr: Repr, rows: usize, k: usize, rng: &mut Rng, scale: f32) -> Weight {
    let bytes = gguf_bytes(repr, rows, k, rng, scale);
    let values = repr
        .element()
        .decode_host(&[rows as u64, k as u64], &bytes)
        .unwrap()
        .into_iter()
        .map(|value| value as f32)
        .collect();
    Weight {
        repr,
        rows,
        k,
        bytes,
        values,
    }
}

/// The rows16 bytes of `gguf_weight`.
fn gguf_bytes(repr: Repr, rows: usize, k: usize, rng: &mut Rng, scale: f32) -> Vec<u8> {
    let (source, block_bytes, block_values) = repr.gguf_source().unwrap();
    assert_eq!(k % block_values, 0, "{repr:?} rows of whole GGUF blocks");
    let mut external = Vec::with_capacity(rows * k / block_values * block_bytes);
    for _ in 0..rows * k / block_values {
        let mut block = (0..block_bytes)
            .map(|_| rng.next() as u8)
            .collect::<Vec<_>>();
        let factor = scale * (0.5 + rng.uniform());
        match repr {
            Repr::Q4g32s => block[..2].copy_from_slice(&f16_bits(factor / 8.0).to_le_bytes()),
            // d: the table's values reach 127 d (32 d (ls - 32) under IQ4_XS's
            // six-bit local scales).
            Repr::Iq4xs => block[..2].copy_from_slice(&f16_bits(factor / 4096.0).to_le_bytes()),
            Repr::Iq4g32s => block[..2].copy_from_slice(&f16_bits(factor / 128.0).to_le_bytes()),
            Repr::Q5g32s => block[..2].copy_from_slice(&f16_bits(factor / 16.0).to_le_bytes()),
            Repr::Q4g32 => {
                block[..2].copy_from_slice(&f16_bits(factor / 8.0).to_le_bytes());
                block[2..4].copy_from_slice(&f16_bits(-factor).to_le_bytes());
            }
            Repr::Q5g32 => {
                block[..2].copy_from_slice(&f16_bits(factor / 16.0).to_le_bytes());
                block[2..4].copy_from_slice(&f16_bits(-factor).to_le_bytes());
            }
            // E8M0: the power of two nearest factor / 6 (the E2M1 range).
            Repr::Mxfp4 => block[0] = (127.0 + (factor / 6.0).log2().round()) as u8,
            // UE4M3 exponent of factor / 6 (normal), random mantissas.
            Repr::Nvfp4 | Repr::Nvfp4A4 => {
                for field in &mut block[..4] {
                    let exponent = ((factor / 6.0).log2().floor() as i32 + 7).clamp(1, 14) as u8;
                    *field = (exponent << 3) | rng.below(8) as u8;
                }
            }
            // d: values reach d (2 s + 1) 43 / 8 and d (2 s + 1) 62 / 4.
            Repr::Iq2xs | Repr::Iq2xxs => {
                block[..2].copy_from_slice(&f16_bits(factor / 40.0).to_le_bytes())
            }
            Repr::Iq3xxs => block[..2].copy_from_slice(&f16_bits(factor / 120.0).to_le_bytes()),
            // d: values reach d 32 4.
            Repr::Q3k => block[108..110].copy_from_slice(&f16_bits(factor / 128.0).to_le_bytes()),
            // d and dmin: values reach d 15 3 and dmin 15.
            Repr::Q2k => {
                block[80..82].copy_from_slice(&f16_bits(factor / 45.0).to_le_bytes());
                block[82..84].copy_from_slice(&f16_bits(factor / 30.0).to_le_bytes());
            }
            _ => unreachable!("{repr:?} is not imported from GGUF"),
        }
        external.extend(block);
    }
    repr.element()
        .repack_host(
            Element::named(source).unwrap(),
            &[rows as u64, k as u64],
            &external,
        )
        .unwrap()
}

// ---------------------------------------------------------------------------
// References.

/// Ordered F32 FMA dot and the magnitude sum bounding its reassociation error.
fn dot(x: &[f32], w: &[f32]) -> (f32, f32) {
    let mut acc = 0f32;
    let mut magnitude = 0f32;
    for (a, b) in x.iter().zip(w) {
        acc = a.mul_add(*b, acc);
        magnitude += (a * b).abs();
    }
    (acc, magnitude)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// RMS normalization of one row rounded to the activation, with, per element,
/// the rounding flip a device may legitimately take because its square sum is
/// ordered differently (zero away from rounding boundaries).
fn rms_row(act: Act, x: &[f32], norm: &[f32], epsilon: f32) -> (Vec<f32>, Vec<f32>) {
    let squares = x.iter().fold(0f32, |sum, value| value.mul_add(*value, sum));
    let inverse = 1.0 / (squares / x.len() as f32 + epsilon).sqrt();
    let values = x
        .iter()
        .zip(norm)
        .map(|(v, w)| act.round(v * inverse * w))
        .collect();
    let slack = x
        .iter()
        .zip(norm)
        .map(|(v, w)| {
            let exact = v * inverse * w;
            (act.round(exact * (1.0 + 1e-5)) - act.round(exact * (1.0 - 1e-5))).abs()
        })
        .collect();
    (values, slack)
}

fn slack_dot(slack: &[f32], w: &[f32]) -> f32 {
    slack.iter().zip(w).map(|(s, w)| s * w.abs()).sum()
}

/// Bound for an A-rounded projection: reassociation plus one rounding flip.
fn rounded_bound(act: Act, value: f32, magnitude: f32) -> f32 {
    reassociation_bound(magnitude) + value.abs() * act.ulp() + 1e-7
}

fn assert_within(label: &str, actual: &[f32], expected: &[f32], bound: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let failures = (0..expected.len())
        .filter(|i| !((actual[*i] - expected[*i]).abs() <= bound[*i]))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "{label}: {} of {} outside tolerance; first {:?}",
        failures.len(),
        expected.len(),
        failures
            .iter()
            .take(4)
            .map(|i| (*i, actual[*i], expected[*i], bound[*i]))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Device helpers.

/// Metal on macOS, elsewhere Vulkan when present (rows16 is the resident
/// layout of both), and the CPU device.
fn devices() -> Vec<Device> {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let gpu = if cfg!(target_os = "macos") {
        Some(catalog.open_backend(seismic::BackendName::Metal).unwrap())
    } else {
        catalog.open_backend(seismic::BackendName::Vulkan).ok()
    };
    gpu.into_iter()
        .chain(std::iter::once(
            catalog.open_backend(seismic::BackendName::Cpu).unwrap(),
        ))
        .collect()
}

fn is_cpu(device: &Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

fn f32_tensor(device: &Device, shape: &[usize], values: &[f32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), &shape, &bytes).unwrap()
}

fn i32_tensor(device: &Device, shape: &[usize], values: &[i32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), &shape, &bytes).unwrap()
}

fn act_tensor(device: &Device, act: Act, shape: &[usize], values: &[f32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    Tensor::from_host(device, act.element(), &shape, &act.bytes(values)).unwrap()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_act(act: Act, tensor: &Tensor) -> Vec<f32> {
    act.decode(&tensor.read_to_host().unwrap())
}

/// A K1 mapping: the GEMV threadgroup (SIMDGROUPS x ROWS lane groups of
/// LANES lanes), the first row count served by the batched GEMV (BATCH_FROM;
/// smaller classes run the GEMV), the batched threadgroup (BATCH_SIMDGROUPS x
/// BATCH_ROWS blocks of 8 weight rows), the GEMM tile (TILE_M x TILE_N; on
/// Vulkan over subgroups of SUB_M x SUB_N), for the small-N output
/// projections the split-K factor, and for the Metal dense entries the tall
/// GEMM when they take that form.
#[derive(Clone, Copy, Debug)]
struct Mapping {
    simdgroups: u64,
    rows: u64,
    lanes: u64,
    batch_from: u64,
    batch_simdgroups: u64,
    batch_rows: u64,
    tile_m: u64,
    tile_n: u64,
    sub_m: u64,
    sub_n: u64,
    split: u64,
    tall: Option<Tall>,
    /// The Metal dense entries' GEMV form over row tiles (TILED), for one or
    /// two rows of `rows32` weights.
    tiled: u64,
    /// Simdgroups sharing a weight fragment in `dense_output`'s matrix GEMV
    /// (BATCH_PARTS).
    batch_parts: u64,
    /// `dense_output`'s INT8 tile rows on Metal (INT8_TILE).
    int8_tile: u64,
}

/// A tall GEMM threadgroup: TALL_M rows, TALL_K columns per stage, STAGERS
/// staging simdgroups.
#[derive(Clone, Copy, Debug)]
struct Tall {
    rows: u64,
    columns: u64,
    stagers: u64,
}

/// The tall launch's declaration defaults.
const TALL_DEFAULT: Tall = Tall {
    rows: 128,
    columns: 64,
    stagers: 2,
};

/// Decode defaults with the given GEMM tile and split.
const fn gemm_mapping(tile_m: u64, tile_n: u64, split: u64) -> Mapping {
    Mapping {
        simdgroups: 16,
        rows: 1,
        lanes: 16,
        batch_from: 5,
        batch_simdgroups: 8,
        batch_rows: 2,
        tile_m,
        tile_n,
        sub_m: 32,
        sub_n: 32,
        split,
        tall: None,
        tiled: 0,
        batch_parts: 2,
        int8_tile: 128,
    }
}

const MAPPINGS: [Mapping; 4] = [
    Mapping {
        simdgroups: 4,
        rows: 1,
        lanes: 32,
        batch_from: 3,
        batch_simdgroups: 4,
        batch_rows: 1,
        tile_m: 64,
        tile_n: 64,
        sub_m: 64,
        sub_n: 64,
        split: 1,
        tall: None,
        tiled: 0,
        batch_parts: 2,
        int8_tile: 128,
    },
    Mapping {
        simdgroups: 2,
        rows: 4,
        lanes: 16,
        batch_from: 9,
        batch_simdgroups: 16,
        batch_rows: 2,
        tile_m: 128,
        tile_n: 64,
        sub_m: 64,
        sub_n: 32,
        split: 2,
        tall: None,
        tiled: 1,
        batch_parts: 1,
        int8_tile: 128,
    },
    Mapping {
        simdgroups: 32,
        rows: 2,
        lanes: 32,
        batch_from: 5,
        batch_simdgroups: 8,
        batch_rows: 4,
        tile_m: 32,
        tile_n: 128,
        sub_m: 32,
        sub_n: 64,
        split: 4,
        tall: None,
        tiled: 1,
        batch_parts: 4,
        int8_tile: 128,
    },
    Mapping {
        simdgroups: 16,
        rows: 2,
        lanes: 16,
        batch_from: 3,
        batch_simdgroups: 32,
        batch_rows: 1,
        tile_m: 64,
        tile_n: 128,
        sub_m: 32,
        sub_n: 32,
        split: 2,
        tall: None,
        tiled: 0,
        batch_parts: 2,
        int8_tile: 128,
    },
];

/// Relative bound on the reassociation error of a dot product over
/// `outputs` output rows: F32 accumulation in the GEMV (outputs <= 2), plus
/// half-rounded operands on the matrix units (batched GEMV and GEMM).
fn reassociation(outputs: usize) -> f32 {
    if outputs > 2 { 7e-4 } else { 3e-5 }
}

thread_local! {
    static REASSOCIATION: std::cell::Cell<f32> = const { std::cell::Cell::new(3e-5) };
}

fn reassociation_bound(magnitude: f32) -> f32 {
    REASSOCIATION.with(|factor| factor.get()) * magnitude
}

/// Evaluate a reference under the reassociation bound of `outputs` rows.
fn under<T>(outputs: usize, reference: impl FnOnce() -> T) -> T {
    REASSOCIATION.with(|factor| factor.set(reassociation(outputs)));
    let value = reference();
    REASSOCIATION.with(|factor| factor.set(3e-5));
    value
}

/// The static dimensions of an entry on `device`: CPU implementations have
/// none.
fn statics_on(device: &Device, statics: &[(&str, usize)]) -> NativeSpecialization {
    if is_cpu(device) {
        return NativeSpecialization::new();
    }
    statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, (name, value)| {
            spec.with_static(*name, *value as u64)
        })
}

/// An output entry's statics over the single residual stream: one stream
/// (`SN` 1) joined without inject (`IN` 0). The CPU form has no statics.
fn joined(device: &Device, spec: NativeSpecialization) -> NativeSpecialization {
    if is_cpu(device) {
        return spec;
    }
    spec.with_static("SN", 1).with_static("IN", 0)
}

/// Rows `[M, D]` as the `[M, 1, D]` parts the features entry reads.
fn one_part(rows: &Tensor) -> Tensor {
    let extents = rows.extents();
    rows.reshape(&[extents[0], 1, extents[1]]).unwrap()
}

/// The row width a consumer entry's statics name: `D`, else `H`.
fn row_width(statics: &[(&str, usize)]) -> usize {
    ["D", "H"]
        .iter()
        .find_map(|name| statics.iter().find(|(static_name, _)| static_name == name))
        .map(|(_, value)| *value)
        .expect("a consumer entry's statics name its row width")
}

/// A consumer entry's statics over the single residual stream: its input
/// read as one part (`PT` 1) normalized at the row width `width` (`NM`). The
/// CPU form has no statics.
fn formed(device: &Device, spec: NativeSpecialization, width: usize) -> NativeSpecialization {
    if is_cpu(device) {
        return spec;
    }
    spec.with_static("PT", 1).with_static("NM", width as u64)
}

/// The specialization of a projection entry on `device` under `mapping`. CPU
/// implementations have no static dimensions and one mapping parameter, the
/// weight rows per work item.
fn specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    if is_cpu(device) {
        return NativeSpecialization::new()
            .with_param("ROWS", mapping.rows)
            .with_param("INT8", 0);
    }
    let specialization = statics_on(device, statics)
        .with_param("SIMDGROUPS", mapping.simdgroups)
        .with_param("ROWS", mapping.rows)
        .with_param("LANES", mapping.lanes)
        .with_param("TILE_M", mapping.tile_m)
        .with_param("TILE_N", mapping.tile_n);
    // Vulkan has no batched GEMV class (its GEMV serves M <= 8) and maps the
    // GEMM tile onto subgroup tiles.
    if cfg!(target_os = "macos") {
        specialization
            .with_param("BATCH_FROM", mapping.batch_from)
            .with_param("BATCH_SIMDGROUPS", mapping.batch_simdgroups)
            .with_param("BATCH_ROWS", mapping.batch_rows)
    } else {
        specialization
            .with_param("SUB_M", mapping.sub_m)
            .with_param("SUB_N", mapping.sub_n)
    }
}

/// The specialization of an output projection with split-K on `device`.
fn split_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    if is_cpu(device) {
        return specialization_on(device, statics, mapping);
    }
    specialization_on(device, statics, mapping).with_param("SPLIT", mapping.split)
}

struct ProjectionLaunches {
    gemv: usize,
    batch: usize,
    gemm: usize,
}

fn scoped_projection_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
    launches: ProjectionLaunches,
    split: bool,
) -> NativeSpecialization {
    if device.backend() != BackendName::Metal {
        return if split {
            split_specialization_on(device, statics, mapping)
        } else {
            specialization_on(device, statics, mapping)
        };
    }
    let choice = statics
        .iter()
        .fold(NativeSpecialization::new(), |choice, (name, value)| {
            choice.with_static(*name, *value as u64)
        })
        .with_param("BATCH_FROM", mapping.batch_from)
        .with_launch_param(launches.gemv, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(launches.gemv, "ROWS", mapping.rows)
        .with_launch_param(launches.gemv, "LANES", mapping.lanes)
        .with_launch_param(launches.batch, "BATCH_SIMDGROUPS", mapping.batch_simdgroups)
        .with_launch_param(launches.batch, "BATCH_ROWS", mapping.batch_rows)
        .with_launch_param(launches.gemm, "TILE_M", mapping.tile_m)
        .with_launch_param(launches.gemm, "TILE_N", mapping.tile_n);
    if split {
        choice.with_param("SPLIT", mapping.split)
    } else {
        choice
    }
}

/// `mapping`'s GEMV threadgroup on the Metal launches for three rows up to
/// BATCH_FROM (`launch`) and for two rows (the next) of an entry that launches
/// its GEMV per row count.
fn with_gemv_rows(
    device: &Device,
    specialization: NativeSpecialization,
    launch: usize,
    mapping: Mapping,
) -> NativeSpecialization {
    if device.backend() != BackendName::Metal {
        return specialization;
    }
    specialization
        .with_launch_param(launch, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(launch, "ROWS", mapping.rows)
        .with_launch_param(launch, "LANES", mapping.lanes)
        .with_launch_param(launch + 1, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(launch + 1, "ROWS", mapping.rows)
        .with_launch_param(launch + 1, "LANES", mapping.lanes)
}

/// The `gemv_rows` launches on Metal, each followed by its entry's last
/// launch, `gemv_pair`.
const DENSE_OUTPUT_GEMV_ROWS_LAUNCH: usize = 12;
const DENSE_EXPAND_GEMV_ROWS_LAUNCH: usize = 10;
const ATTENTION_OUTPUT_GEMV_ROWS_LAUNCH: usize = 9;
const ATTENTION_PROJECT_GEMV_ROWS_LAUNCH: usize = 7;
const READOUT_HEAD_GEMV_ROWS_LAUNCH: usize = 4;

/// The TALL form of a Metal dense entry and the tile of its tall launch
/// (`launch`), which takes the declaration defaults under the staged form.
fn with_tall(
    device: &Device,
    specialization: NativeSpecialization,
    launch: usize,
    tall: Option<Tall>,
) -> NativeSpecialization {
    if device.backend() != BackendName::Metal {
        return specialization;
    }
    let tile = tall.unwrap_or(TALL_DEFAULT);
    specialization
        .with_param("TALL", u64::from(tall.is_some()))
        .with_launch_param(launch, "TALL_M", tile.rows)
        .with_launch_param(launch, "TALL_K", tile.columns)
        .with_launch_param(launch, "STAGERS", tile.stagers)
}

/// The packing bound statics `names` of a Metal projection entry (static
/// there only), absent; a test binding a bound sets its own.
fn with_absent_bounds(
    device: &Device,
    specialization: NativeSpecialization,
    names: &[&str],
) -> NativeSpecialization {
    if device.backend() != BackendName::Metal {
        return specialization;
    }
    names
        .iter()
        .fold(specialization, |specialization, name| specialization.with_static(*name, 0))
}

/// `mapping` as Metal's `dense_expand` and `dense_output` admit it: their GEMV
/// lane groups own one or two weight rows.
fn dense_gemv_mapping(device: &Device, mapping: Mapping) -> Mapping {
    if device.backend() != BackendName::Metal {
        return mapping;
    }
    Mapping {
        rows: mapping.rows.min(2),
        ..mapping
    }
}

/// `dense_output`'s INT8 tile launch on Metal, and the dense entries' PACK
/// tile launches.
const DENSE_OUTPUT_INT8_LAUNCH: usize = 11;
const DENSE_OUTPUT_PACKED_LAUNCH: usize = 16;
const DENSE_EXPAND_PACKED_LAUNCH: usize = 14;

fn dense_output_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    let mapping = dense_gemv_mapping(device, mapping);
    let specialization = scoped_projection_specialization_on(
        device,
        statics,
        mapping,
        ProjectionLaunches {
            gemv: 0,
            batch: 2,
            gemm: 5,
        },
        true,
    )
    .with_static("DS", 0);
    let specialization = with_absent_bounds(device, specialization, &["DB"]);
    let specialization = with_tall(device, specialization, 8, mapping.tall);
    let specialization = with_gemv_rows(
        device,
        specialization,
        DENSE_OUTPUT_GEMV_ROWS_LAUNCH,
        mapping,
    );
    if device.backend() == BackendName::Metal {
        with_pack(specialization, DENSE_OUTPUT_PACKED_LAUNCH, PACK_DEFAULT)
            .with_param("PACK", 0)
            .with_param("INT8", 0)
            .with_launch_param(0, "TILED", mapping.tiled)
            .with_launch_param(DENSE_OUTPUT_GEMV_ROWS_LAUNCH, "TILED", mapping.tiled.min(1))
            .with_launch_param(DENSE_OUTPUT_GEMV_ROWS_LAUNCH + 1, "TILED", mapping.tiled)
            .with_launch_param(2, "BATCH_PARTS", mapping.batch_parts)
            .with_launch_param(DENSE_OUTPUT_INT8_LAUNCH, "INT8_TILE", mapping.int8_tile)
    } else {
        specialization
    }
}

fn attention_output_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    // The weight's scale port: absent unless `statics` names it.
    let scale = static_or_absent(statics, "OS");
    // The CPU form's statics: the gated heads and width.
    if is_cpu(device) {
        return statics
            .iter()
            .filter(|(name, _)| matches!(*name, "Q" | "W"))
            .fold(specialization_on(device, statics, mapping), |spec, (name, value)| {
                spec.with_static(*name, *value as u64)
            })
            .with_static("OS", scale);
    }
    let specialization = scoped_projection_specialization_on(
        device,
        statics,
        mapping,
        ProjectionLaunches {
            gemv: 0,
            batch: 2,
            gemm: 5,
        },
        true,
    )
    .with_static("OS", scale);
    let specialization = with_absent_bounds(device, specialization, &["OB"]);
    let specialization = with_tall(device, specialization, 8, mapping.tall);
    let specialization = with_gemv_rows(
        device,
        specialization,
        ATTENTION_OUTPUT_GEMV_ROWS_LAUNCH,
        mapping,
    );
    let specialization = joined(device, specialization);
    if device.backend() == BackendName::Metal {
        with_pack(specialization, ATTENTION_OUTPUT_PACKED_LAUNCH, PACK_DEFAULT)
            .with_param("PACK", 0)
            .with_launch_param(2, "BATCH_PARTS", mapping.batch_parts)
    } else {
        specialization
    }
}

/// The PACK tile launches on Metal: `attention_output`'s, `attention_project`'s
/// and `gated_delta_project`'s.
const ATTENTION_OUTPUT_PACKED_LAUNCH: usize = 13;
const ATTENTION_PROJECT_PACKED_LAUNCH: usize = 11;
const RECURRENT_PROJECT_PACKED_LAUNCH: usize = 9;

/// The specialization of a segmented projection (`attention_project`,
/// `gated_delta_project`) on `device`; `packed` is its PACK tile launch on
/// Metal.
fn segmented_projection_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
    packed: usize,
) -> NativeSpecialization {
    let specialization = scoped_projection_specialization_on(
        device,
        statics,
        mapping,
        ProjectionLaunches {
            gemv: 1,
            batch: 2,
            gemm: 4,
        },
        false,
    );
    let bounds: &[&str] = if packed == ATTENTION_PROJECT_PACKED_LAUNCH {
        &["QB", "GB", "KB", "VB"]
    } else {
        &["QB", "ZB"]
    };
    let specialization = with_absent_bounds(device, specialization, bounds);
    let specialization = with_tall(device, specialization, 6, mapping.tall);
    let specialization = formed(device, specialization, row_width(statics));
    if device.backend() == BackendName::Metal {
        with_pack(specialization, packed, PACK_DEFAULT).with_param("PACK", 0)
    } else {
        specialization
    }
}

/// The specialization of `attention_project` on `device`: the segmented
/// projection's, with the GEMV launch for two rows up to BATCH_FROM.
fn attention_project_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    // The segments' scale ports: absent unless `statics` names them.
    let scaled = |specialization: NativeSpecialization| {
        ["QS", "GS", "KS", "VS"]
            .into_iter()
            .fold(specialization, |specialization, name| {
                specialization.with_static(name, static_or_absent(statics, name))
            })
    };
    // The CPU form's other static: the row width its `[M, PT * D]` input
    // does not name alone.
    if is_cpu(device) {
        return scaled(
            specialization_on(device, statics, mapping)
                .with_static("D", row_width(statics) as u64),
        );
    }
    scaled(with_gemv_rows(
        device,
        segmented_projection_specialization_on(
            device,
            statics,
            mapping,
            ATTENTION_PROJECT_PACKED_LAUNCH,
        ),
        ATTENTION_PROJECT_GEMV_ROWS_LAUNCH,
        mapping,
    ))
}

/// The value `statics` gives the scale port static `name`, else 0 (absent).
fn static_or_absent(statics: &[(&str, usize)], name: &str) -> u64 {
    statics
        .iter()
        .find(|(static_name, _)| *static_name == name)
        .map_or(0, |(_, value)| *value as u64)
}

fn dense_expand_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    let mapping = dense_gemv_mapping(device, mapping);
    let specialization = scoped_projection_specialization_on(
        device,
        statics,
        mapping,
        ProjectionLaunches {
            gemv: 1,
            batch: 2,
            gemm: 4,
        },
        false,
    )
    .with_static("GS", 0)
    .with_static("US", 0);
    let specialization = with_absent_bounds(device, specialization, &["GB", "UB"]);
    let specialization = with_tall(device, specialization, 6, mapping.tall);
    let specialization = with_gemv_rows(
        device,
        specialization,
        DENSE_EXPAND_GEMV_ROWS_LAUNCH,
        mapping,
    );
    if device.backend() == BackendName::Metal {
        with_pack(specialization, DENSE_EXPAND_PACKED_LAUNCH, PACK_DEFAULT)
            .with_param("PACK", 0)
            .with_param("INT8", 0)
            .with_launch_param(2, "BATCH_PARTS", mapping.batch_parts)
            .with_launch_param(1, "TILED", mapping.tiled)
            .with_launch_param(DENSE_EXPAND_GEMV_ROWS_LAUNCH, "TILED", mapping.tiled.min(1))
            .with_launch_param(DENSE_EXPAND_GEMV_ROWS_LAUNCH + 1, "TILED", mapping.tiled)
    } else {
        specialization
    }
}

/// A PACK tile launch of the Metal projection entries: operand fragments per
/// simdgroup (4 x 2 or 2 x 4 fragments), whether the weight words are
/// loaded one block ahead, and the form's residue width (`RESIDUE`: 2 only
/// for weights with a packing bound, whose values are 8 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pack {
    tokens: u64,
    weights_ahead: u64,
    residue: u64,
}

const PACK_DEFAULT: Pack = Pack {
    tokens: 4,
    weights_ahead: 0,
    residue: 1,
};

/// The default tile launch with two residue bits.
const PACK_WIDE: Pack = Pack {
    residue: 2,
    ..PACK_DEFAULT
};

/// The exact forms' launches at their defaults, where a PACK configuration
/// holds them.
fn pack_mapping() -> Mapping {
    Mapping {
        tall: Some(TALL_DEFAULT),
        ..gemm_mapping(64, 64, 1)
    }
}

/// Every PACK tile launch.
fn pack_tiles() -> Vec<Pack> {
    let mut all = Vec::new();
    for tokens in [4, 2] {
        for weights_ahead in [0, 1] {
            all.push(Pack {
                tokens,
                weights_ahead,
                residue: 1,
            });
        }
    }
    all
}

/// `specialization` under the PACK form with the tile launch `pack`.
fn with_pack(
    specialization: NativeSpecialization,
    launch: usize,
    pack: Pack,
) -> NativeSpecialization {
    specialization
        .with_param("PACK", 1)
        .with_param("RESIDUE", pack.residue)
        .with_launch_param(launch, "PACK_TOKENS", pack.tokens)
        .with_launch_param(launch, "WEIGHTS_AHEAD", pack.weights_ahead)
}

fn readout_projection_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    let specialization = scoped_projection_specialization_on(
        device,
        statics,
        mapping,
        ProjectionLaunches {
            gemv: 1,
            batch: 2,
            gemm: 3,
        },
        false,
    );
    formed(device, specialization, row_width(statics))
}

/// The specialization of `readout_head_rows` on `device`: the readout
/// projection's, with the GEMV launch for two rows up to BATCH_FROM.
fn readout_head_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    let specialization = with_gemv_rows(
        device,
        readout_projection_specialization_on(device, statics, mapping),
        READOUT_HEAD_GEMV_ROWS_LAUNCH,
        mapping,
    );
    if device.backend() == BackendName::Metal {
        specialization.with_launch_param(2, "BATCH_PARTS", mapping.batch_parts)
    } else {
        specialization
    }
}

fn prepared_projection_specialization_on(
    device: &Device,
    statics: &[(&str, usize)],
    mapping: Mapping,
) -> NativeSpecialization {
    let v = statics.iter().find(|(key, _)| *key == "V").unwrap().1;
    let mut spec = NativeSpecialization::new()
        .with_static("S", v as u64)
        .with_static("I", 0)
        .with_static("WS", 0);
    for (key, value) in statics {
        spec = spec.with_static(*key, *value as u64);
    }
    match device.backend() {
        BackendName::Metal => spec
            .with_launch_param(0, "SIMDGROUPS", mapping.simdgroups)
            .with_launch_param(0, "ROWS", mapping.rows)
            .with_launch_param(0, "LANES", mapping.lanes),
        BackendName::Vulkan => spec
            .with_param("SIMDGROUPS", mapping.simdgroups)
            .with_param("ROWS", mapping.rows)
            .with_param("LANES", mapping.lanes),
        _ => spec,
    }
}

const ROW_CLASSES: [usize; 9] = [1, 2, 3, 8, 9, 16, 17, 64, 128];

/// The decode and verify rows checked at the pinned Qwen3.5 4B geometry: the
/// GEMV (1, 2, 4), the batched class (5, 16) and the first GEMM class (17).
const FOUR_B_ROWS: [usize; 6] = [1, 2, 4, 5, 16, 17];

/// The row classes and mappings a geometry is checked over: every mapping at
/// the small geometry, the decode mapping over the decode and verify rows at
/// the 4B geometry.
fn checked_classes(four_b: bool) -> (Vec<usize>, Vec<Mapping>) {
    if four_b {
        (FOUR_B_ROWS.to_vec(), vec![gemm_mapping(64, 64, 1)])
    } else {
        (ROW_CLASSES.to_vec(), MAPPINGS.to_vec())
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
            "recurrent.seismic",
            include_str!("../kernels/recurrent.seismic"),
        ),
        (
            "attention.seismic",
            include_str!("../kernels/attention.seismic"),
        ),
        (
            "functions.seismic",
            include_str!("../kernels/functions.seismic"),
        ),
        ("import.seismic", include_str!("../kernels/import.seismic")),
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
    I32(i32),
}

fn floats(shape: &[usize], values: &[f32]) -> Input {
    Input::Data(TensorData::dense(
        DType::F32,
        shape.to_vec(),
        values.iter().map(|v| f64::from(*v)).collect(),
    ))
}

fn ints(shape: &[usize], values: &[i32]) -> Input {
    Input::Data(TensorData::dense(
        DType::I32,
        shape.to_vec(),
        values.iter().map(|v| f64::from(*v)).collect(),
    ))
}

fn activations(act: Act, shape: &[usize], values: &[f32]) -> Input {
    Input::Data(TensorData::dense(
        act.dtype(),
        shape.to_vec(),
        values.iter().map(|v| f64::from(*v)).collect(),
    ))
}

fn interpret(
    module: &CheckedModule,
    name: &str,
    bindings: &[(&str, RepresentationId)],
    inputs: Vec<Input>,
) -> OracleOutcome {
    let elements = bindings
        .iter()
        .fold(ElementBindings::new(), |elements, (name, id)| {
            elements.bind(name, *id)
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
            Input::I32(value) => Arg::Scalar(ReferenceScalar::I32(value)),
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

fn norm_values(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| bf16_round(0.5 + rng.uniform())).collect()
}

fn bf16_norm(device: &Device, values: &[f32]) -> Tensor {
    act_tensor(device, Act::Bf16, &[values.len()], values)
}

/// `attention_project`'s statics in Qwen's form: the query segment holds the
/// interleaved query and gate rows, no separate gate segment.
fn attention_project_statics(
    d: usize,
    kv: usize,
    g: usize,
    w: usize,
) -> [(&'static str, usize); 5] {
    [
        ("D", d),
        ("Q", kv * g * 2 * w),
        ("GR", 0),
        ("K", kv * w),
        ("V", kv * w),
    ]
}

// ---------------------------------------------------------------------------
// Tests.

#[test]
fn rows16_builder_matches_the_registry_layout_and_decode() {
    for repr in [Repr::Q4k, Repr::Q5k, Repr::Q6k, Repr::Q8] {
        let k = 512;
        let w = weight(repr, 3, k, 1, 1.0);
        let info = registry::representation_info(repr.storage());
        let RepresentationKind::PackedRows(layout) = &info.kind else {
            panic!("{repr:?} rows16")
        };
        let expected = planes(repr, k);
        assert_eq!(
            layout.row_stride_bytes(k as u64).unwrap() as usize,
            expected.stride,
            "{repr:?} stride"
        );
        for (ordinal, plane) in layout.planes.iter().enumerate() {
            let offset = layout.plane_row_offset(ordinal, k as u64).unwrap() as usize;
            let mine = match plane.name {
                "codes_lo" | "codes" => expected.codes,
                "codes_hi" => expected.high,
                "scales" => expected.scales,
                "supers" => expected.supers,
                other => panic!("unexpected plane {other}"),
            };
            assert_eq!(offset, mine, "{repr:?} plane {}", plane.name);
        }
        let decoded = w.oracle().values().unwrap();
        for (i, (oracle, host)) in decoded.iter().zip(&w.values).enumerate() {
            assert_eq!(*oracle as f32, *host, "{repr:?} value {i}");
        }
    }
}

fn dense_expand_native(
    device: &Device,
    act: Act,
    gate: &Weight,
    up: &Weight,
    residual: &[f32],
    rows: usize,
    norm: &[f32],
    out_rows: &[i32],
    eps: f32,
    mapping: Mapping,
) -> Vec<f32> {
    dense_expand_native_arithmetic(
        device, act, gate, up, residual, rows, norm, out_rows, eps, mapping, false,
    )
}

/// `dense_expand` on `device`; `int8` selects Metal's INT8 form.
#[allow(clippy::too_many_arguments)]
fn dense_expand_native_arithmetic(
    device: &Device,
    act: Act,
    gate: &Weight,
    up: &Weight,
    residual: &[f32],
    rows: usize,
    norm: &[f32],
    out_rows: &[i32],
    eps: f32,
    mapping: Mapping,
    int8: bool,
) -> Vec<f32> {
    let (h, f) = (gate.k, gate.rows);
    let mut specialization = dense_expand_specialization_on(device, &[("H", h), ("F", f)], mapping);
    if device.backend() == BackendName::Metal {
        specialization = specialization.with_param("INT8", u64::from(int8));
    }
    let kernel = dense_expand::native_for_device_with(
        device,
        dense_expand::Elements {
            NW: Element::bf16(),
            GW: gate.repr.resident(&device),
            UW: up.repr.resident(&device),
            A: act.element(),
        },
        &specialization,
    )
    .unwrap();
    let out = kernel
        .call(dense_expand::Args {
            residual: &f32_tensor(device, &[rows, h], residual),
            norm: &bf16_norm(device, norm),
            gate_weight: &gate.tensor(device),
            up_weight: &up.tensor(device),
            out_rows: &i32_tensor(device, &[out_rows.len()], out_rows),
            eps,
            activation: 0,
            gate_scale: &f32_tensor(device, &[0], &[]),
            up_scale: &f32_tensor(device, &[0], &[]),
            gate_bound: &absent_bound(device),
            up_bound: &absent_bound(device),
        })
        .unwrap()
        .value;
    read_act(act, &out)
}

fn dense_expand_reference(
    act: Act,
    gate: &Weight,
    up: &Weight,
    residual: &[f32],
    norm: &[f32],
    out_rows: &[i32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (h, f) = (gate.k, gate.rows);
    let mut expected = Vec::new();
    let mut bound = Vec::new();
    for row in out_rows {
        let source = *row as usize;
        let (x, slack) = rms_row(act, &residual[source * h..(source + 1) * h], norm, eps);
        let (xg, xu) = (gate.operand_rows(&x), up.operand_rows(&x));
        for n in 0..f {
            let (g, gm) = dot(&xg, gate.row(n));
            let (u, um) = dot(&xu, up.row(n));
            let (g, u) = (act.round(g), act.round(u));
            let a = act.round(silu(g));
            let value = act.round(a * u);
            expected.push(value);
            let gate_error =
                reassociation_bound(gm) + g.abs() * act.ulp() + slack_dot(&slack, gate.row(n));
            let up_error =
                reassociation_bound(um) + u.abs() * act.ulp() + slack_dot(&slack, up.row(n));
            bound.push(
                1.2 * (gate_error * u.abs() + up_error * a.abs()) + value.abs() * act.ulp() + 1e-6,
            );
        }
    }
    (expected, bound)
}

#[test]
fn dense_expand_matches_its_portable_body() {
    for device in devices() {
        dense_expand_matches_its_portable_body_on(
            &device,
            &[(Repr::Q4k, Repr::Q6k), (Repr::Q5k, Repr::Q8)],
        );
    }
}

/// IQ4_XS weights are resident as iq4xs (the table decode).
#[test]
fn dense_expand_iq4_matches_its_portable_body() {
    for device in devices() {
        dense_expand_matches_its_portable_body_on(&device, &[(Repr::Iq4xs, Repr::Iq4xs)]);
    }
}

fn dense_expand_matches_its_portable_body_on(device: &Device, pairs: &[(Repr, Repr)]) {
    let module = module();
    let act = Act::Bf16;
    let (h, f) = (256, 40);
    for &(gate_repr, up_repr) in pairs {
        let gate = weight(gate_repr, f, h, 11, 1.0);
        let up = weight(up_repr, f, h, 12, 1.0);
        let norm = norm_values(h, 3);
        for (rows, out_rows) in [
            (3usize, vec![2, 0]),
            (18, (0..16).map(|i| (i * 5 + 1) % 18).collect::<Vec<i32>>()),
        ] {
            let mut rng = Rng::new(rows as u64);
            let residual = (0..rows * h)
                .map(|_| rng.symmetric() * 3.0)
                .collect::<Vec<_>>();
            let outcome = interpret(
                &module,
                "dense_expand",
                &[
                    ("NW", registry::dense(DType::BF16)),
                    ("GW", gate_repr.storage()),
                    ("UW", up_repr.storage()),
                    ("A", registry::dense(DType::BF16)),
                ],
                vec![
                    floats(&[rows, h], &residual),
                    activations(Act::Bf16, &[h], &norm),
                    Input::Data(gate.oracle()),
                    Input::Data(up.oracle()),
                    ints(&[out_rows.len()], &out_rows),
                    Input::F32(1e-6),
                    Input::I32(0),
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                    // No packing bounds.
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                ],
            );
            let oracle = result(&outcome, 0);
            for mapping in MAPPINGS {
                let (_, bound) = under(out_rows.len(), || {
                    dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6)
                });
                let native = dense_expand_native(
                    &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, mapping,
                );
                assert_within(
                    &format!("dense_expand {gate_repr:?}/{up_repr:?} rows {rows} {mapping:?}"),
                    &native,
                    &oracle,
                    &bound,
                );
            }
        }
    }
}

#[test]
fn dense_expand_matches_the_host_reference_across_row_classes() {
    for device in devices() {
        dense_expand_matches_the_host_reference_across_row_classes_on(&device);
    }
}

fn dense_expand_matches_the_host_reference_across_row_classes_on(device: &Device) {
    let (h, f) = (512, 1000);
    for (act, gate_repr, up_repr) in [
        (Act::Bf16, Repr::Q4k, Repr::Q4k),
        (Act::Bf16, Repr::Q5k, Repr::Q6k),
        (Act::Bf16, Repr::Q8, Repr::Bf16),
        (Act::Bf16, Repr::Iq4xs, Repr::Q4k),
        (Act::F16, Repr::Q4k, Repr::Q4k),
    ] {
        let gate = weight(gate_repr, f, h, 21, 1.0);
        let up = weight(up_repr, f, h, 22, 1.0);
        let norm = norm_values(h, 4);
        for rows in ROW_CLASSES {
            let mut rng = Rng::new(rows as u64 + 7);
            let residual = (0..rows * h)
                .map(|_| rng.symmetric() * 3.0)
                .collect::<Vec<_>>();
            let out_rows = (0..rows)
                .map(|i| ((i * 7 + 3) % rows) as i32)
                .collect::<Vec<_>>();
            for mapping in MAPPINGS {
                let (expected, bound) = under(out_rows.len(), || {
                    dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6)
                });
                let native = dense_expand_native(
                    &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, mapping,
                );
                assert_within(
                    &format!("dense_expand {act:?} {gate_repr:?}/{up_repr:?} M {rows} {mapping:?}"),
                    &native,
                    &expected,
                    &bound,
                );
            }
        }
    }
}

#[test]
fn metal_dense_expand_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let (h, f) = (256, 128);
    let gate = weight(Repr::Q4k, f, h, 61, 1.0);
    let up = weight(Repr::Q4k, f, h, 62, 1.0);
    let norm = norm_values(h, 63);
    for rows in [1usize, 8, 32, 128] {
        let mut rng = Rng::new(rows as u64);
        let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
        let out_rows = (0..rows as i32).collect::<Vec<_>>();
        let (expected, bound) = under(rows, || {
            dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6)
        });
        for mapping in [MAPPINGS[0], MAPPINGS[2]] {
            let actual = dense_expand_native(
                &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, mapping,
            );
            assert_within(
                &format!("scoped dense_expand rows {rows} {mapping:?}"),
                &actual,
                &expected,
                &bound,
            );
        }
    }
}

/// The paired compressed-code matrix path keeps both projection sums in
/// F32 until the original GLU epilogue. Cover its two activation fragments,
/// split-K exchange, distinct gate/up formats, and generic batch fallback.
#[test]
fn metal_dense_expand_matrix_pair_matches_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|d| d.backend() == BackendName::Metal)
    else {
        return;
    };
    for act in [Act::Bf16, Act::F16] {
        for (h, f) in [(256usize, 131usize), (2560, 67)] {
            let norm = norm_values(h, 63);
            for (gate_repr, up_repr) in [
                (Repr::Q4g32s, Repr::Q4g32s),
                (Repr::Q4g32s, Repr::Q4k),
                (Repr::Q4k, Repr::Q4k),
                (Repr::Q5k, Repr::Q4k),
                (Repr::Q6k, Repr::Q5k),
                (Repr::Q5k, Repr::Q6k),
                (Repr::Q8, Repr::Q8),
            ] {
                let gate = weight(gate_repr, f, h, 61, 1.0);
                let up = weight(up_repr, f, h, 62, 1.0);
                for rows in [3usize, 5, 8, 13, 16] {
                    let residual = uniform_values(rows * h, rows as u64, 1.0);
                    let out_rows = (0..rows as i32).rev().collect::<Vec<_>>();
                    let (expected, bound) = under(rows, || {
                        dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6)
                    });
                    for parts in [1, 2, 4] {
                        let mapping = Mapping {
                            batch_from: 3,
                            batch_parts: parts,
                            ..gemm_mapping(64, 64, 1)
                        };
                        let actual = dense_expand_native(
                            &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6,
                            mapping,
                        );
                        assert_within(
                            &format!("matrix pair {gate_repr:?}/{up_repr:?} M{rows} parts{parts}"),
                            &actual,
                            &expected,
                            &bound,
                        );
                    }
                }
            }
        }
    }
}

/// The class walk (TILED 2) adds the row mapping's partial sums in the row
/// mapping's order, so it reproduces TILED 0 bit for bit under the same
/// LANES: `dense_expand` (paired) and `dense_output` (plain) over Q8_0 and
/// Q4_K, at one row and at two (the launches that offer it), with weight rows
/// that fill the last tile and that do not, in mappings the walk serves
/// (whole lane blocks a simdgroup, one and two rows a lane).
#[test]
fn metal_class_walk_is_bit_identical_to_the_row_mapping() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
    let mappings = [
        (8, 1, 16),
        (8, 1, 8),
        (4, 1, 16),
        (4, 1, 8),
        (4, 2, 16),
        (16, 1, 16),
    ]
    .map(|(simdgroups, rows, lanes)| Mapping {
        simdgroups,
        rows,
        lanes,
        batch_from: 9,
        ..gemm_mapping(64, 64, 1)
    });
    for repr in [Repr::Q8, Repr::Q4k] {
        for (h, f) in [(2048usize, 96usize), (2048, 72)] {
            let gate = weight(repr, f, h, 71, 1.0);
            let up = weight(repr, f, h, 72, 1.0);
            let down = weight(repr, f, h, 73, 1.0);
            let norm = norm_values(h, 74);
            for rows in [1usize, 2] {
                let mut rng = Rng::new(rows as u64 + 75);
                let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
                let narrow = (0..rows * f).map(|_| rng.symmetric()).collect::<Vec<_>>();
                let product = (0..rows * h)
                    .map(|_| act.round(rng.symmetric()))
                    .collect::<Vec<_>>();
                let out_rows = (0..rows as i32).collect::<Vec<_>>();
                for mapping in mappings {
                    let walked = Mapping {
                        tiled: 2,
                        ..mapping
                    };
                    let expand = |mapping| {
                        dense_expand_native(
                            &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6,
                            mapping,
                        )
                    };
                    assert!(
                        same(&expand(walked), &expand(mapping)),
                        "dense_expand {repr:?} {f}x{h} rows {rows} {mapping:?}"
                    );
                    let output = |mapping| {
                        dense_output_native(
                            &device, act, &down, &narrow, rows, &product, &out_rows, mapping,
                        )
                    };
                    assert!(
                        same(&output(walked), &output(mapping)),
                        "dense_output {repr:?} {f}x{h} rows {rows} {mapping:?}"
                    );
                }
            }
        }
    }
}

/// Every tall tile of the Metal dense entries.
const TALL_TILES: [Tall; 6] = [
    TALL_DEFAULT,
    Tall {
        rows: 256,
        columns: 128,
        stagers: 1,
    },
    Tall {
        rows: 512,
        columns: 64,
        stagers: 1,
    },
    Tall {
        rows: 256,
        columns: 64,
        stagers: 2,
    },
    Tall {
        rows: 128,
        columns: 128,
        stagers: 1,
    },
    Tall {
        rows: 512,
        columns: 128,
        stagers: 2,
    },
];

/// The tall GEMM multiplies the staged tile's operand values in its K order
/// per output, so TALL 1 reproduces TALL 0 bit for bit: paired and plain, in
/// every packed format and tall tile, over whole row tiles (128, 192), a row
/// count past its last 64-row and 16-row groups (200), and weight rows past
/// the last tile (200 features, 72 outputs).
#[test]
fn metal_tall_gemm_is_bit_identical_to_the_staged_tiles() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
    let staged = gemm_mapping(64, 64, 1);
    for repr in [
        Repr::Q4k,
        Repr::Q5k,
        Repr::Q6k,
        Repr::Q8,
        Repr::Iq4xs,
        Repr::Bf16,
    ] {
        let (h, f) = (256, 200);
        let gate = weight(repr, f, h, 71, 1.0);
        let up = weight(repr, f, h, 72, 1.0);
        let norm = norm_values(h, 73);
        let (outputs, k) = (72, 512);
        let down = weight(repr, outputs, k, 74, 1.0);
        for rows in [128usize, 192, 200] {
            let mut rng = Rng::new(rows as u64 + 75);
            let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let narrow = (0..rows * outputs)
                .map(|_| rng.symmetric())
                .collect::<Vec<_>>();
            let product = (0..rows * k).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let out_rows = (0..rows as i32).collect::<Vec<_>>();
            let expand = |mapping| {
                dense_expand_native(
                    &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, mapping,
                )
            };
            let output = |mapping| {
                dense_output_native(
                    &device, act, &down, &narrow, rows, &product, &out_rows, mapping,
                )
            };
            let (expand_staged, output_staged) = (expand(staged), output(staged));
            let (expand_expected, expand_bound) = under(rows, || {
                dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6)
            });
            let (output_expected, output_bound) = under(rows, || {
                dense_output_reference(act, &down, &narrow, &product, &out_rows)
            });
            for tile in TALL_TILES {
                let mapping = Mapping {
                    tall: Some(tile),
                    ..staged
                };
                let (expand_tall, output_tall) = (expand(mapping), output(mapping));
                assert_within(
                    &format!("tall dense_expand {repr:?} rows {rows} {tile:?}"),
                    &expand_tall,
                    &expand_expected,
                    &expand_bound,
                );
                assert_within(
                    &format!("tall dense_output {repr:?} rows {rows} {tile:?}"),
                    &output_tall,
                    &output_expected,
                    &output_bound,
                );
                assert!(
                    same(&expand_tall, &expand_staged),
                    "dense_expand {repr:?} rows {rows} {tile:?}"
                );
                assert!(
                    same(&output_tall, &output_staged),
                    "dense_output {repr:?} rows {rows} {tile:?}"
                );
            }
        }
    }
}

/// The attention and recurrent projections under TALL 1 reproduce TALL 0 bit
/// for bit, as the dense entries do: the segmented projections (query | key |
/// value, with and without the zeroed query; qkv | z | alpha | beta), and the
/// output projection, over whole row tiles and a row count past its last
/// 64-row and 16-row groups.
#[test]
fn metal_tall_segmented_projections_are_bit_identical_to_the_staged_tiles() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
    let staged = gemm_mapping(64, 64, 1);
    let (d, kv, g, w) = (256, 1, 3, 40);
    let rows_of = [kv * g * 2 * w, kv * w, kv * w];
    let reprs = [Repr::Q4k, Repr::Q8, Repr::Q6k];
    let weights = (0..3)
        .map(|i| weight(reprs[i], rows_of[i], d, 81 + i as u64, 1.0))
        .collect::<Vec<_>>();
    let norm = norm_values(d, 84);
    let (outputs, heads, width) = (72, 8, 64);
    let out_weight = weight(Repr::Q5k, outputs, heads * width, 85, 1.0);
    let (nk, nv, head_width) = (1, 3, 24);
    let recurrent_reprs = [Repr::Q5k, Repr::Q4k, Repr::Q8, Repr::Bf16];
    let recurrent_rows = [(2 * nk + nv) * head_width, nv * head_width, nv, nv];
    let recurrent_weights = (0..4)
        .map(|i| weight(recurrent_reprs[i], recurrent_rows[i], d, 90 + i as u64, 1.0))
        .collect::<Vec<_>>();
    for rows in [128usize, 200] {
        let mut rng = Rng::new(rows as u64 + 86);
        let hidden = (0..rows * d).map(|_| rng.symmetric()).collect::<Vec<_>>();
        let narrow = (0..rows * outputs)
            .map(|_| rng.symmetric())
            .collect::<Vec<_>>();
        let gated = (0..rows * heads * width)
            .map(|_| act.round(rng.symmetric()))
            .collect::<Vec<_>>();
        let project = |mapping, project_mode| {
            let out = attention_project::native_for_device_with(
                &device,
                attention_project::Elements {
                    NW: Element::bf16(),
                    QW: reprs[0].resident(&device),
                    GW: Element::bf16(),
                    KW: reprs[1].resident(&device),
                    VW: reprs[2].resident(&device),
                    A: act.element(),
                },
                &attention_project_specialization_on(
                    &device,
                    &attention_project_statics(d, kv, g, w),
                    mapping,
                ),
            )
            .unwrap()
            .call(attention_project::Args {
                hidden: &f32_tensor(&device, &[rows, d], &hidden),
                input_norm: &bf16_norm(&device, &norm),
                query_weight: &weights[0].tensor(&device),
                gate_weight: &act_tensor(&device, Act::Bf16, &[0, d], &[]),
                key_weight: &weights[1].tensor(&device),
                value_weight: &weights[2].tensor(&device),
                epsilon: 1e-6,
                project_mode,
                query_scale: &no_scale(&device),
                gate_scale: &no_scale(&device),
                key_scale: &no_scale(&device),
                value_scale: &no_scale(&device),
                query_bound: &absent_bound(&device),
                gate_bound: &absent_bound(&device),
                key_bound: &absent_bound(&device),
                value_bound: &absent_bound(&device),
            })
            .unwrap();
            [
                read_act(act, &out.r0),
                read_act(act, &out.r2),
                read_act(act, &out.r3),
            ]
        };
        let output = |mapping| {
            let out = attention_output::native_for_device_with(
                &device,
                attention_output::Elements {
                    OW: out_weight.repr.resident(&device),
                    A: act.element(),
                },
                &attention_output_specialization_on(
                    &device,
                    &[("D", outputs), ("Q", heads), ("W", width)],
                    mapping,
                ),
            )
            .unwrap()
            .call(attention_output::Args {
                hidden: &f32_tensor(&device, &[rows, outputs], &narrow),
                inject: &no_inject(&device, rows as u64),
                gated: &act_tensor(&device, act, &[rows, heads, width], &gated),
                output_weight: &out_weight.tensor(&device),
                output_scale: &no_scale(&device),
                output_bound: &absent_bound(&device),
            })
            .unwrap()
            .value;
            read_f32(&out)
        };
        let recurrent = |mapping| {
            let out = gated_delta_project::native_for_device_with(
                &device,
                gated_delta_project::Elements {
                    NW: Element::bf16(),
                    QW: recurrent_reprs[0].resident(&device),
                    GW: recurrent_reprs[1].resident(&device),
                    AW: recurrent_reprs[2].resident(&device),
                    BW: recurrent_reprs[3].resident(&device),
                    A: act.element(),
                },
                &segmented_projection_specialization_on(
                    &device,
                    &[("H", d), ("NK", nk), ("NV", nv), ("W", head_width)],
                    mapping,
                    RECURRENT_PROJECT_PACKED_LAUNCH,
                ),
            )
            .unwrap()
            .call(gated_delta_project::Args {
                hidden: &f32_tensor(&device, &[rows, d], &hidden),
                input_norm: &bf16_norm(&device, &norm),
                qkv_weight: &recurrent_weights[0].tensor(&device),
                gate_weight: &recurrent_weights[1].tensor(&device),
                alpha_weight: &recurrent_weights[2].tensor(&device),
                beta_weight: &recurrent_weights[3].tensor(&device),
                epsilon: 1e-6,
                qkv_bound: &absent_bound(&device),
                gate_bound: &absent_bound(&device),
            })
            .unwrap()
            .value;
            read_act(act, &out)
        };
        let project_staged = [project(staged, 0), project(staged, 1)];
        let output_staged = output(staged);
        let recurrent_staged = recurrent(staged);
        for tile in TALL_TILES {
            let mapping = Mapping {
                tall: Some(tile),
                ..staged
            };
            assert!(
                same(&recurrent(mapping), &recurrent_staged),
                "gated_delta_project rows {rows} {tile:?}"
            );
            for (mode, expected) in project_staged.iter().enumerate() {
                let actual = project(mapping, mode as i32);
                for (segment, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        same(actual, expected),
                        "attention_project segment {segment} mode {mode} rows {rows} {tile:?}"
                    );
                }
            }
            assert!(
                same(&output(mapping), &output_staged),
                "attention_output rows {rows} {tile:?}"
            );
        }
    }
}

fn dense_output_native(
    device: &Device,
    act: Act,
    down: &Weight,
    residual: &[f32],
    rows: usize,
    product: &[f32],
    out_rows: &[i32],
    mapping: Mapping,
) -> Vec<f32> {
    dense_output_native_arithmetic(
        device, act, down, residual, rows, product, out_rows, mapping, false,
    )
}

fn dense_output_native_arithmetic(
    device: &Device,
    act: Act,
    down: &Weight,
    residual: &[f32],
    rows: usize,
    product: &[f32],
    out_rows: &[i32],
    mapping: Mapping,
    int8: bool,
) -> Vec<f32> {
    let (h, f) = (down.rows, down.k);
    let mut specialization = dense_output_specialization_on(device, &[("H", h), ("F", f)], mapping);
    if is_cpu(device) || device.backend() == BackendName::Metal {
        specialization = specialization.with_param("INT8", u64::from(int8));
    }
    let kernel = dense_output::native_for_device_with(
        device,
        dense_output::Elements {
            A: act.element(),
            DW: down.repr.resident(&device),
        },
        &specialization,
    )
    .unwrap();
    let out = kernel
        .call(dense_output::Args {
            residual: &f32_tensor(device, &[rows, h], residual),
            product: &act_tensor(device, act, &[out_rows.len(), f], product),
            down_weight: &down.tensor(device),
            out_rows: &i32_tensor(device, &[out_rows.len()], out_rows),
            down_scale: &f32_tensor(device, &[0], &[]),
            down_bound: &absent_bound(device),
        })
        .unwrap()
        .value;
    read_f32(&out)
}

#[test]
fn cpu_int8_dense_output_agrees_with_exact_projection() {
    let Some(device) = devices().into_iter().find(is_cpu) else {
        return;
    };
    let (h, f, rows) = (37, 512, 9);
    let down = weight(Repr::Q4k, h, f, 313, 1.0);
    let mut random = Rng::new(314);
    let product = (0..rows * f)
        .map(|_| random.symmetric())
        .collect::<Vec<_>>();
    let residual = vec![0.0; rows * h];
    let out_rows = (0..rows as i32).collect::<Vec<_>>();
    let exact = dense_output_native_arithmetic(
        &device,
        Act::Bf16,
        &down,
        &residual,
        rows,
        &product,
        &out_rows,
        Mapping {
            rows: 8,
            ..MAPPINGS[0]
        },
        false,
    );
    let int8 = dense_output_native_arithmetic(
        &device,
        Act::Bf16,
        &down,
        &residual,
        rows,
        &product,
        &out_rows,
        Mapping {
            rows: 8,
            ..MAPPINGS[0]
        },
        true,
    );
    let error = exact
        .iter()
        .zip(&int8)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>();
    let scale = exact.iter().map(|value| value.powi(2)).sum::<f32>();
    assert!(
        error <= 0.05f32.powi(2) * scale,
        "relative CPU INT8 error {}",
        (error / scale).sqrt()
    );
}

/// Metal's INT8 form of the down projection: int8 activations per (row, 32
/// columns) against the exact weights. On tensor operations with Q4_K weights
/// its results differ from the tall form's by the activation quantization
/// alone; other weights and devices without tensor operations run the staged
/// 64 x 64 tile, bit for bit.
#[test]
fn metal_int8_dense_output_agrees_with_the_tall_form() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let staged = gemm_mapping(64, 64, 1);
    let tall = Mapping {
        tall: Some(TALL_DEFAULT),
        ..staged
    };
    for repr in [Repr::Q4k, Repr::Q6k, Repr::Q8] {
        let (outputs, k) = (192, 1024);
        let down = weight(repr, outputs, k, 91, 1.0);
        for rows in [128usize, 200, 512] {
            let mut rng = Rng::new(rows as u64 + 92);
            let residual = (0..rows * outputs)
                .map(|_| rng.symmetric())
                .collect::<Vec<_>>();
            let product = (0..rows * k).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let out_rows = (0..rows as i32).collect::<Vec<_>>();
            let run = |mapping, int8| {
                dense_output_native_arithmetic(
                    &device, act, &down, &residual, rows, &product, &out_rows, mapping, int8,
                )
            };
            let exact = run(staged, false);
            let int8 = run(tall, true);
            // The projection alone (the residual rows are exact in both).
            let projected = |values: &[f32]| {
                values
                    .iter()
                    .zip(&residual)
                    .map(|(value, residual)| f64::from(value - residual))
                    .collect::<Vec<_>>()
            };
            let (reference, actual) = (projected(&exact), projected(&int8));
            let scale = reference.iter().map(|value| value * value).sum::<f64>();
            let error = reference
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>();
            let relative = (error / scale).sqrt();
            let worst = reference
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
                / (scale / reference.len() as f64).sqrt();
            eprintln!(
                "int8 dense_output {repr:?} rows {rows}: relative RMS error {relative:.2e}, largest {worst:.2e} reference RMS"
            );
            assert!(
                relative <= 1.5e-2 && worst <= 0.1,
                "int8 dense_output {repr:?} rows {rows}: relative RMS error {relative:.2e}, largest {worst:.2e}"
            );
        }
    }
}

/// The relative RMS difference of `actual` from `reference` and the largest
/// element difference in units of the reference RMS.
fn relative_difference(reference: &[f64], actual: &[f64]) -> (f64, f64) {
    let scale = reference.iter().map(|value| value * value).sum::<f64>();
    let error = reference
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>();
    let worst = reference
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    (
        (error / scale).sqrt(),
        worst / (scale / reference.len() as f64).sqrt(),
    )
}

/// `project_rows`' launches on Metal: the tall tile and the PACK tile.
const PROJECT_ROWS_TALL_LAUNCH: usize = 8;
const PROJECT_ROWS_PACKED_LAUNCH: usize = 11;

/// `project_rows` (activation rows in, F32 rows out, no scale port) on Metal
/// under `mapping`, with the PACK tile launch `pack` or in an exact form.
fn project_rows_native(
    device: &Device,
    act: Act,
    weight: &Weight,
    source: &[f32],
    mapping: Mapping,
    pack: Option<Pack>,
) -> Vec<f32> {
    let (n, k) = (weight.rows, weight.k);
    let rows = source.len() / k;
    let (bound, bound_extent) = weight.bound(device);
    let specialization = with_tall(
        device,
        scoped_projection_specialization_on(
            device,
            &[("K", k), ("N", n)],
            mapping,
            ProjectionLaunches {
                gemv: 0,
                batch: 2,
                gemm: 5,
            },
            true,
        )
        .with_static("WS", 0),
        PROJECT_ROWS_TALL_LAUNCH,
        mapping.tall,
    );
    let specialization = with_absent_bounds(device, specialization, &["WB"]);
    let specialization = if bound_extent != 0 {
        specialization.with_static("WB", bound_extent as u64)
    } else {
        specialization
    };
    let specialization = with_pack(
        specialization,
        PROJECT_ROWS_PACKED_LAUNCH,
        pack.unwrap_or(PACK_DEFAULT),
    )
    .with_param("PACK", u64::from(pack.is_some()))
    .with_launch_param(2, "BATCH_PARTS", mapping.batch_parts);
    let kernel = project_rows::native_for_device_with(
        device,
        project_rows::Elements {
            A: act.element(),
            W: weight.repr.resident(device),
            Y: Element::f32(),
        },
        &specialization,
    )
    .unwrap();
    let out = kernel
        .call(project_rows::Args {
            source: &act_tensor(device, act, &[rows, k], source),
            weight: &weight.tensor(device),
            weight_scale: &f32_tensor(device, &[0], &[]),
            weight_bound: &bound,
        })
        .unwrap()
        .value;
    read_f32(&out)
}

/// Metal's tall and PACK forms of the plain projection: the tall tile gives
/// the staged tile's bits; the PACK form differs from them by the packed
/// form's error over Q4_K, Q5_K, Q6_K, q4g32s, Q8_0 or IQ4 codebook weights.
#[test]
fn metal_tall_and_packed_project_rows_agree_with_the_staged_form() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let staged = gemm_mapping(64, 64, 1);
    for repr in [
        Repr::Q4k,
        Repr::Q5k,
        Repr::Q6k,
        Repr::Q4g32s,
        Repr::Q8,
        Repr::Iq4xs,
        Repr::Iq4g32s,
    ] {
        let (outputs, k) = (192, 1024);
        let weight = weight(repr, outputs, k, 97, 1.0);
        for rows in [128usize, 200, 512] {
            let mut rng = Rng::new(rows as u64 + 98);
            let source = (0..rows * k).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let exact = project_rows_native(&device, act, &weight, &source, staged, None);
            let tall = project_rows_native(&device, act, &weight, &source, pack_mapping(), None);
            assert!(
                exact
                    .iter()
                    .zip(&tall)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "tall project_rows {repr:?} rows {rows} differs from the staged tile"
            );
            let mut first: Option<Vec<f32>> = None;
            for pack in pack_tiles() {
                let packed =
                    project_rows_native(&device, act, &weight, &source, pack_mapping(), Some(pack));
                // Every tile launch gives the same bits.
                let first = first.get_or_insert_with(|| packed.clone());
                assert!(
                    first
                        .iter()
                        .zip(&packed)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "packed project_rows {repr:?} rows {rows} {pack:?} differs from {PACK_DEFAULT:?}"
                );
                let wide = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| f64::from(*value))
                        .collect::<Vec<_>>()
                };
                let (relative, worst) = relative_difference(&wide(&exact), &wide(&packed));
                if pack == PACK_DEFAULT {
                    eprintln!(
                        "packed project_rows {repr:?} rows {rows}: relative RMS error {relative:.2e}, largest {worst:.2e} reference RMS"
                    );
                }
                assert!(
                    relative <= 1.5e-2 && worst <= 0.1,
                    "packed project_rows {repr:?} rows {rows} {pack:?}: relative RMS error {relative:.2e}, largest {worst:.2e}"
                );
            }
        }
    }
}

/// `dense_output` under Metal's PACK form with the tile launch `pack`.
#[allow(clippy::too_many_arguments)]
fn dense_output_packed(
    device: &Device,
    act: Act,
    down: &Weight,
    residual: &[f32],
    rows: usize,
    product: &[f32],
    out_rows: &[i32],
    pack: Pack,
) -> Vec<f32> {
    let (h, f) = (down.rows, down.k);
    let (bound, bound_extent) = down.bound(device);
    let specialization = with_pack(
        dense_output_specialization_on(device, &[("H", h), ("F", f)], pack_mapping()),
        DENSE_OUTPUT_PACKED_LAUNCH,
        pack,
    )
    .with_static("DB", bound_extent as u64);
    let kernel = dense_output::native_for_device_with(
        device,
        dense_output::Elements {
            A: act.element(),
            DW: down.repr.resident(device),
        },
        &specialization,
    )
    .unwrap();
    let out = kernel
        .call(dense_output::Args {
            residual: &f32_tensor(device, &[rows, h], residual),
            product: &act_tensor(device, act, &[out_rows.len(), f], product),
            down_weight: &down.tensor(device),
            out_rows: &i32_tensor(device, &[out_rows.len()], out_rows),
            down_scale: &f32_tensor(device, &[0], &[]),
            down_bound: &bound,
        })
        .unwrap()
        .value;
    read_f32(&out)
}

/// Metal's PACK form of the down projection: integer activation codes per
/// (row, 32 columns), two rows packed per operand element against the exact
/// Q4_K, Q5_K, Q6_K, q4g32s, Q8_0 or IQ4 codebook codes. Its results differ from the staged
/// form's by the activation rounding, the packed accumulator's rounding and
/// the F16 fold.
#[test]
fn metal_packed_dense_output_agrees_with_the_staged_form() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let staged = gemm_mapping(64, 64, 1);
    for repr in [
        Repr::Q4k,
        Repr::Q5k,
        Repr::Q6k,
        Repr::Q4g32s,
        Repr::Q8,
        Repr::Iq4xs,
        Repr::Iq4g32s,
    ] {
        let (outputs, k) = (192, 1024);
        let down = weight(repr, outputs, k, 91, 1.0);
        for rows in [128usize, 200, 512] {
            let mut rng = Rng::new(rows as u64 + 92);
            let residual = (0..rows * outputs)
                .map(|_| rng.symmetric())
                .collect::<Vec<_>>();
            let product = (0..rows * k).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let out_rows = (0..rows as i32).collect::<Vec<_>>();
            let exact = dense_output_native_arithmetic(
                &device, act, &down, &residual, rows, &product, &out_rows, staged, false,
            );
            let mut first: Option<Vec<f32>> = None;
            // The projection alone (the residual rows are exact in both).
            let projected = |values: &[f32]| {
                values
                    .iter()
                    .zip(&residual)
                    .map(|(value, residual)| f64::from(value - residual))
                    .collect::<Vec<_>>()
            };
            let mut narrow = 0.0;
            for pack in pack_tiles() {
                let packed = dense_output_packed(
                    &device, act, &down, &residual, rows, &product, &out_rows, pack,
                );
                // A sum is its token pair's and weight row's alone: every
                // tile launch gives the same bits.
                let first = first.get_or_insert_with(|| packed.clone());
                assert!(
                    first
                        .iter()
                        .zip(&packed)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "packed dense_output {repr:?} rows {rows} {pack:?} differs from {PACK_DEFAULT:?}"
                );
                let (relative, worst) =
                    relative_difference(&projected(&exact), &projected(&packed));
                if pack == PACK_DEFAULT {
                    narrow = relative;
                    eprintln!(
                        "packed dense_output {repr:?} rows {rows}: relative RMS error {relative:.2e}, largest {worst:.2e} reference RMS"
                    );
                }
                assert!(
                    relative <= 1.5e-2 && worst <= 0.1,
                    "packed dense_output {repr:?} rows {rows} {pack:?}: relative RMS error {relative:.2e}, largest {worst:.2e}"
                );
            }
            // 8-bit values: two residue bits give the top sum one more bit,
            // the top token half its step. With the pair's two errors alike
            // the form's falls to sqrt((1/4 + 1) / 2) of it, 0.79.
            if repr.takes_residue(&device) {
                let packed = dense_output_packed(
                    &device, act, &down, &residual, rows, &product, &out_rows, PACK_WIDE,
                );
                let (relative, worst) =
                    relative_difference(&projected(&exact), &projected(&packed));
                eprintln!(
                    "packed dense_output {repr:?} rows {rows} RESIDUE 2: relative RMS error {relative:.2e}, largest {worst:.2e} reference RMS"
                );
                assert!(
                    relative <= 0.8 * narrow && worst <= 0.1,
                    "packed dense_output {repr:?} rows {rows} RESIDUE 2: relative RMS error {relative:.2e} against {narrow:.2e} with one"
                );
            }
        }
    }
}

/// Metal's PACK form of the paired gate/up projection, as for the down
/// projection: the GLU outputs differ from the staged form's by the packed
/// form's error with gate and up weights one packed operand serves (Q4_K,
/// Q5_K and q4g32s together, both Q6_K, or Q8_0 and IQ4 codebook weights together), and are
/// the staged tile's bit for bit otherwise.
#[test]
fn metal_packed_dense_expand_agrees_with_the_staged_form() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let staged = gemm_mapping(64, 64, 1);
    for (gate_repr, up_repr, served) in [
        (Repr::Q4k, Repr::Q4k, true),
        (Repr::Q5k, Repr::Q4k, true),
        (Repr::Q6k, Repr::Q6k, true),
        (Repr::Q4g32s, Repr::Q4g32s, true),
        (Repr::Q4g32s, Repr::Q4k, true),
        (Repr::Q8, Repr::Q8, true),
        (Repr::Iq4xs, Repr::Iq4xs, true),
        (Repr::Iq4xs, Repr::Q8, true),
        (Repr::Iq4g32s, Repr::Iq4g32s, true),
        (Repr::Q4k, Repr::Q6k, false),
        (Repr::Q8, Repr::Q4k, false),
    ] {
        let (h, f) = (512, 192);
        let gate = weight(gate_repr, f, h, 93, 1.0);
        let up = weight(up_repr, f, h, 94, 1.0);
        let norm = norm_values(h, 95);
        for rows in [128usize, 200, 512] {
            let mut rng = Rng::new(rows as u64 + 96);
            let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let out_rows = (0..rows as i32).collect::<Vec<_>>();
            let exact = dense_expand_native_arithmetic(
                &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, staged, false,
            );
            let mut first: Option<Vec<f32>> = None;
            let (gate_bound, gate_bound_extent) = gate.bound(&device);
            let (up_bound, up_bound_extent) = up.bound(&device);
            // 8-bit values also take two residue bits.
            let bytes = served
                && gate_bound_extent > 0
                && up_bound_extent > 0
                && gate.repr.takes_residue(&device)
                && up.repr.takes_residue(&device);
            let mut narrow = 0.0;
            for pack in pack_tiles().into_iter().chain(bytes.then_some(PACK_WIDE)) {
                let specialization = with_pack(
                    dense_expand_specialization_on(&device, &[("H", h), ("F", f)], pack_mapping()),
                    DENSE_EXPAND_PACKED_LAUNCH,
                    pack,
                )
                .with_static("GB", gate_bound_extent as u64)
                .with_static("UB", up_bound_extent as u64);
                let kernel = dense_expand::native_for_device_with(
                    &device,
                    dense_expand::Elements {
                        NW: Element::bf16(),
                        GW: gate.repr.resident(&device),
                        UW: up.repr.resident(&device),
                        A: act.element(),
                    },
                    &specialization,
                )
                .unwrap();
                let packed = read_act(
                    act,
                    &kernel
                        .call(dense_expand::Args {
                            residual: &f32_tensor(&device, &[rows, h], &residual),
                            norm: &bf16_norm(&device, &norm),
                            gate_weight: &gate.tensor(&device),
                            up_weight: &up.tensor(&device),
                            out_rows: &i32_tensor(&device, &[rows], &out_rows),
                            eps: 1e-6,
                            activation: 0,
                            gate_scale: &f32_tensor(&device, &[0], &[]),
                            up_scale: &f32_tensor(&device, &[0], &[]),
                            gate_bound: &gate_bound,
                            up_bound: &up_bound,
                        })
                        .unwrap()
                        .value,
                );
                if !served {
                    assert_eq!(exact, packed, "{gate_repr:?} {up_repr:?} rows {rows}");
                    continue;
                }
                let wide = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| f64::from(*value))
                        .collect::<Vec<_>>()
                };
                let (relative, _) = relative_difference(&wide(&exact), &wide(&packed));
                if pack.residue == 2 {
                    // Two residue bits halve the top token's step (the
                    // form's error to 0.79 of it with the pair's alike).
                    eprintln!(
                        "packed dense_expand {gate_repr:?} {up_repr:?} rows {rows} RESIDUE 2: relative RMS error {relative:.2e}"
                    );
                    assert!(
                        relative <= 0.8 * narrow,
                        "packed dense_expand {gate_repr:?} {up_repr:?} rows {rows} RESIDUE 2: relative RMS error {relative:.2e} against {narrow:.2e} with one"
                    );
                    continue;
                }
                // Every tile launch gives the same bits.
                let first = first.get_or_insert_with(|| packed.clone());
                assert!(
                    first
                        .iter()
                        .zip(&packed)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "packed dense_expand {gate_repr:?} {up_repr:?} rows {rows} {pack:?} differs from {PACK_DEFAULT:?}"
                );
                if pack == PACK_DEFAULT {
                    narrow = relative;
                    eprintln!(
                        "packed dense_expand {gate_repr:?} {up_repr:?} rows {rows}: relative RMS error {relative:.2e}"
                    );
                }
                assert!(
                    relative <= 3e-2,
                    "packed dense_expand {gate_repr:?} {up_repr:?} rows {rows} {pack:?}: relative RMS error {relative:.2e}"
                );
            }
        }
    }
}

fn recurrent_project_elements(
    device: &Device,
    reprs: [Repr; 4],
    act: Act,
) -> gated_delta_project::Elements {
    gated_delta_project::Elements {
        NW: Element::bf16(),
        QW: reprs[0].resident(device),
        GW: reprs[1].resident(device),
        AW: reprs[2].resident(device),
        BW: reprs[3].resident(device),
        A: act.element(),
    }
}

/// Metal's PACK form of the recurrent projection (qkv | z | alpha | beta), the
/// attention projection (query | key | value, with and without the zeroed
/// query) and the output projection past 64 rows. A segment whose weights the
/// packed operand serves on the device differs from the staged tiles by the
/// form's error; every other segment, and every segment on a device without
/// the form, is the staged tiles' bit for bit.
#[test]
fn metal_packed_recurrent_and_attention_projections_agree_with_the_staged_tiles() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let staged = gemm_mapping(64, 64, 1);
    let d = 512;
    let norm = norm_values(d, 84);
    let (kv, g, w) = (1, 1, 64);
    let project_rows = [kv * g * 2 * w, kv * w, kv * w];
    let project_reprs = [Repr::Q4k, Repr::Q5k, Repr::Q6k];
    let project_weights = (0..3)
        .map(|i| weight(project_reprs[i], project_rows[i], d, 181 + i as u64, 1.0))
        .collect::<Vec<_>>();
    let (outputs, heads, width) = (128, 8, 64);
    let out_weights = [Repr::Q5k, Repr::Q4k, Repr::Q6k, Repr::Q8, Repr::Iq4xs]
        .map(|repr| weight(repr, outputs, heads * width, 185, 1.0));
    let (nk, nv, head_width) = (1, 2, 32);
    let recurrent_reprs = [Repr::Q5k, Repr::Q4k, Repr::Q8, Repr::Bf16];
    let recurrent_rows = [(2 * nk + nv) * head_width, nv * head_width, nv, nv];
    let recurrent_weights = (0..4)
        .map(|i| {
            weight(
                recurrent_reprs[i],
                recurrent_rows[i],
                d,
                190 + i as u64,
                1.0,
            )
        })
        .collect::<Vec<_>>();
    let recurrent_columns = recurrent_rows.iter().sum::<usize>();
    let tiles = [
        PACK_DEFAULT,
        Pack {
            tokens: 2,
            weights_ahead: 1,
            residue: 1,
        },
    ];
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
    let check = |label: String, pack: Pack, reference: &[f32], actual: &[f32]| {
        let wide = |values: &[f32]| {
            values
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>()
        };
        let (relative, worst) = relative_difference(&wide(reference), &wide(actual));
        if pack == PACK_DEFAULT {
            eprintln!(
                "packed {label}: relative RMS error {relative:.2e}, largest {worst:.2e} reference RMS"
            );
        }
        assert!(
            relative <= 2e-2 && worst <= 0.2,
            "packed {label} {pack:?}: relative RMS error {relative:.2e}, largest {worst:.2e}"
        );
    };
    for rows in [128usize, 200, 512] {
        let mut rng = Rng::new(rows as u64 + 186);
        let hidden = (0..rows * d).map(|_| rng.symmetric()).collect::<Vec<_>>();
        let narrow = (0..rows * outputs)
            .map(|_| rng.symmetric())
            .collect::<Vec<_>>();
        let gated = (0..rows * heads * width)
            .map(|_| act.round(rng.symmetric()))
            .collect::<Vec<_>>();
        let project = |pack: Option<Pack>, project_mode| {
            let specialization = attention_project_specialization_on(
                &device,
                &attention_project_statics(d, kv, g, w),
                if pack.is_some() {
                    pack_mapping()
                } else {
                    staged
                },
            );
            let out = attention_project::native_for_device_with(
                &device,
                attention_project::Elements {
                    NW: Element::bf16(),
                    QW: project_reprs[0].resident(&device),
                    GW: Element::bf16(),
                    KW: project_reprs[1].resident(&device),
                    VW: project_reprs[2].resident(&device),
                    A: act.element(),
                },
                &match pack {
                    Some(pack) => with_pack(specialization, ATTENTION_PROJECT_PACKED_LAUNCH, pack),
                    None => specialization,
                },
            )
            .unwrap()
            .call(attention_project::Args {
                hidden: &f32_tensor(&device, &[rows, d], &hidden),
                input_norm: &bf16_norm(&device, &norm),
                query_weight: &project_weights[0].tensor(&device),
                gate_weight: &act_tensor(&device, Act::Bf16, &[0, d], &[]),
                key_weight: &project_weights[1].tensor(&device),
                value_weight: &project_weights[2].tensor(&device),
                epsilon: 1e-6,
                project_mode,
                query_scale: &no_scale(&device),
                gate_scale: &no_scale(&device),
                key_scale: &no_scale(&device),
                value_scale: &no_scale(&device),
                query_bound: &absent_bound(&device),
                gate_bound: &absent_bound(&device),
                key_bound: &absent_bound(&device),
                value_bound: &absent_bound(&device),
            })
            .unwrap();
            [
                read_act(act, &out.r0),
                read_act(act, &out.r2),
                read_act(act, &out.r3),
            ]
        };
        let output = |pack: Option<Pack>, weight: &Weight| {
            let (bound, bound_extent) = weight.bound(&device);
            let specialization = attention_output_specialization_on(
                &device,
                &[("D", outputs), ("Q", heads), ("W", width)],
                if pack.is_some() {
                    pack_mapping()
                } else {
                    staged
                },
            )
            .with_static("OB", bound_extent as u64);
            let out = attention_output::native_for_device_with(
                &device,
                attention_output::Elements {
                    OW: weight.repr.resident(&device),
                    A: act.element(),
                },
                &match pack {
                    Some(pack) => with_pack(specialization, ATTENTION_OUTPUT_PACKED_LAUNCH, pack),
                    None => specialization,
                },
            )
            .unwrap()
            .call(attention_output::Args {
                hidden: &f32_tensor(&device, &[rows, outputs], &narrow),
                inject: &no_inject(&device, rows as u64),
                gated: &act_tensor(&device, act, &[rows, heads, width], &gated),
                output_weight: &weight.tensor(&device),
                output_scale: &no_scale(&device),
                output_bound: &bound,
            })
            .unwrap()
            .value;
            // The projection alone (the residual rows are exact in every form).
            read_f32(&out)
                .iter()
                .zip(&narrow)
                .map(|(value, residual)| value - residual)
                .collect::<Vec<_>>()
        };
        let recurrent = |pack: Option<Pack>| {
            let specialization = segmented_projection_specialization_on(
                &device,
                &[("H", d), ("NK", nk), ("NV", nv), ("W", head_width)],
                if pack.is_some() {
                    pack_mapping()
                } else {
                    staged
                },
                RECURRENT_PROJECT_PACKED_LAUNCH,
            );
            let out = gated_delta_project::native_for_device_with(
                &device,
                recurrent_project_elements(&device, recurrent_reprs, act),
                &match pack {
                    Some(pack) => with_pack(specialization, RECURRENT_PROJECT_PACKED_LAUNCH, pack),
                    None => specialization,
                },
            )
            .unwrap()
            .call(gated_delta_project::Args {
                hidden: &f32_tensor(&device, &[rows, d], &hidden),
                input_norm: &bf16_norm(&device, &norm),
                qkv_weight: &recurrent_weights[0].tensor(&device),
                gate_weight: &recurrent_weights[1].tensor(&device),
                alpha_weight: &recurrent_weights[2].tensor(&device),
                beta_weight: &recurrent_weights[3].tensor(&device),
                epsilon: 1e-6,
                qkv_bound: &absent_bound(&device),
                gate_bound: &absent_bound(&device),
            })
            .unwrap()
            .value;
            // The row-major projection as its segments' columns.
            let values = read_act(act, &out);
            let mut first = 0;
            recurrent_rows.map(|columns| {
                let segment = values
                    .chunks(recurrent_columns)
                    .flat_map(|row| row[first..first + columns].iter().copied())
                    .collect::<Vec<_>>();
                first += columns;
                segment
            })
        };
        let project_staged = [project(None, 0), project(None, 1)];
        let output_staged = out_weights.each_ref().map(|weight| output(None, weight));
        let recurrent_staged = recurrent(None);
        for pack in tiles {
            let actual = recurrent(Some(pack));
            for (segment, name) in ["qkv", "z"].into_iter().enumerate() {
                check(
                    format!("gated_delta_project {name} rows {rows}"),
                    pack,
                    &recurrent_staged[segment],
                    &actual[segment],
                );
            }
            for segment in 2..4 {
                assert!(
                    same(&actual[segment], &recurrent_staged[segment]),
                    "gated_delta_project segment {segment} rows {rows} {pack:?}"
                );
            }
            for (mode, expected) in project_staged.iter().enumerate() {
                let actual = project(Some(pack), mode as i32);
                for (segment, name) in ["query", "key"].into_iter().enumerate() {
                    if segment == 0 && mode == 1 {
                        assert!(
                            same(&actual[0], &expected[0]),
                            "attention_project zeroed query rows {rows} {pack:?}"
                        );
                        continue;
                    }
                    check(
                        format!("attention_project {name} mode {mode} rows {rows}"),
                        pack,
                        &expected[segment],
                        &actual[segment],
                    );
                }
                // The operand is laid out for the query's Q4_K runs, which
                // do not serve the Q6_K value weights.
                assert!(
                    same(&actual[2], &expected[2]),
                    "attention_project value mode {mode} rows {rows} {pack:?}"
                );
            }
            for (weight, expected) in out_weights.iter().zip(&output_staged) {
                check(
                    format!("attention_output {:?} rows {rows}", weight.repr),
                    pack,
                    expected,
                    &output(Some(pack), weight),
                );
            }
        }
        // 8-bit values: two residue bits halve the top token's step, so the
        // form's error falls.
        let projection = |values: &[f32]| values.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
        for (weight, expected) in out_weights.iter().zip(&output_staged) {
            if !weight.repr.takes_residue(&device) {
                continue;
            }
            let (narrow, _) =
                relative_difference(&projection(expected), &projection(&output(Some(PACK_DEFAULT), weight)));
            let (relative, worst) =
                relative_difference(&projection(expected), &projection(&output(Some(PACK_WIDE), weight)));
            eprintln!(
                "packed attention_output {:?} rows {rows} RESIDUE 2: relative RMS error {relative:.2e} (one bit {narrow:.2e}), largest {worst:.2e}",
                weight.repr
            );
            assert!(
                relative <= 0.8 * narrow && worst <= 0.2,
                "packed attention_output {:?} rows {rows} RESIDUE 2: relative RMS error {relative:.2e} against {narrow:.2e} with one",
                weight.repr
            );
        }
    }
}

/// Metal's INT8 form of the paired gate/up projection, as for the down
/// projection: the GLU outputs differ from the tall form's by the activation
/// quantization on tensor operations with Q4_K gate and up weights, and are
/// the staged tile's bit for bit otherwise.
#[test]
fn metal_int8_dense_expand_agrees_with_the_tall_form() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let staged = gemm_mapping(64, 64, 1);
    let tall = Mapping {
        tall: Some(TALL_DEFAULT),
        ..staged
    };
    for repr in [Repr::Q4k, Repr::Q8] {
        let (h, f) = (512, 192);
        let gate = weight(repr, f, h, 93, 1.0);
        let up = weight(repr, f, h, 94, 1.0);
        let norm = norm_values(h, 95);
        for rows in [128usize, 200, 512] {
            let mut rng = Rng::new(rows as u64 + 96);
            let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let out_rows = (0..rows as i32).collect::<Vec<_>>();
            let run = |mapping, int8| {
                dense_expand_native_arithmetic(
                    &device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, mapping,
                    int8,
                )
            };
            let exact = run(staged, false);
            let int8 = run(tall, true);
            let scale = exact
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>();
            let error = exact
                .iter()
                .zip(&int8)
                .map(|(a, b)| f64::from(a - b) * f64::from(a - b))
                .sum::<f64>();
            let relative = (error / scale).sqrt();
            eprintln!("int8 dense_expand {repr:?} rows {rows}: relative RMS error {relative:.2e}");
            assert!(
                relative <= 3e-2,
                "int8 dense_expand {repr:?} rows {rows}: relative RMS error {relative:.2e}"
            );
        }
    }
}

fn dense_output_reference(
    act: Act,
    down: &Weight,
    residual: &[f32],
    product: &[f32],
    out_rows: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let (h, f) = (down.rows, down.k);
    let mut expected = Vec::new();
    let mut bound = Vec::new();
    for (r, row) in out_rows.iter().enumerate() {
        let x = down.operand_rows(&product[r * f..(r + 1) * f]);
        for n in 0..h {
            let (acc, magnitude) = dot(&x, down.row(n));
            expected.push(residual[*row as usize * h + n] + act.round(acc));
            bound.push(rounded_bound(act, acc, magnitude));
        }
    }
    (expected, bound)
}

/// `packing_bound` on Metal equals its portable body over every format with
/// integer codes, at a row count that leaves padding rows in a tile, with
/// the runs the registry gives the format.
#[test]
fn packing_bound_matches_its_portable_body() {
    let module = module();
    for device in devices() {
        if device.backend() != BackendName::Metal {
            continue;
        }
        for repr in [
            Repr::Q4k,
            Repr::Q5k,
            Repr::Q6k,
            Repr::Q8,
            Repr::Iq4xs,
            Repr::Q4g32s,
            Repr::Iq4g32s,
            Repr::Q5g32s,
            Repr::Q5g32,
            Repr::Q4g32,
        ] {
            let (rows, k) = (40, 512);
            let w = weight(repr, rows, k, 17, 1.0);
            let resident = repr.resident(&device);
            let runs = k / resident.integer_codes().unwrap().group as usize;
            let outcome = interpret(
                &module,
                "packing_bound",
                &[("W", repr.storage())],
                vec![Input::Data(w.oracle()), floats(&[runs], &vec![0.0; runs])],
            );
            let input = outcome.inputs().nth(1).unwrap();
            let tensor = input.tensor();
            let expected = (0..tensor.element_count())
                .map(|i| tensor.read(i).unwrap() as f32)
                .collect::<Vec<_>>();
            let mut bound = f32_tensor(&device, &[runs], &vec![0.0; runs]);
            packing_bound::native_for_device_with(
                &device,
                packing_bound::Elements { W: resident },
                &NativeSpecialization::new(),
            )
            .unwrap()
            .call(packing_bound::Args {
                weight: &w.tensor(&device),
                bound: &mut bound,
            })
            .unwrap();
            let actual = bound.read_to_host().unwrap();
            let actual = actual
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{:?} {repr:?}", device.backend());
        }
    }
}

#[test]
fn dense_output_matches_its_portable_body_and_the_host_reference() {
    for device in devices() {
        dense_output_matches_its_portable_body_and_the_host_reference_on(
            &device,
            &[Repr::Q4k, Repr::Q5k, Repr::Q6k, Repr::Q8, Repr::Bf16],
        );
    }
}

#[test]
fn dense_output_iq4_matches_its_portable_body_and_the_host_reference() {
    for device in devices() {
        dense_output_matches_its_portable_body_and_the_host_reference_on(&device, &[Repr::Iq4xs]);
    }
}

/// Q4_0, Q4_1, Q5_0, Q5_1, MXFP4 and NVFP4 weights, imported into
/// representations of their own.
#[test]
fn dense_output_gguf_formats_match_their_portable_body_and_the_host_reference() {
    for device in devices() {
        dense_output_matches_its_portable_body_and_the_host_reference_on(&device, &GGUF_REPRS);
    }
}

/// IQ2_XS, IQ3_XXS and IQ2_XXS weights, imported into their vector-coded
/// representations.
#[test]
fn dense_output_grid_formats_match_their_portable_body_and_the_host_reference() {
    for device in devices() {
        dense_output_matches_its_portable_body_and_the_host_reference_on(&device, &GRID_REPRS);
    }
}

/// Q2_K weights, imported into q2k.
#[test]
fn dense_output_k_quant_formats_match_their_portable_body_and_the_host_reference() {
    for device in devices() {
        dense_output_matches_its_portable_body_and_the_host_reference_on(&device, &K_QUANT_REPRS);
    }
}

#[test]
fn dense_expand_k_quant_formats_match_their_portable_body() {
    for device in devices() {
        dense_expand_matches_its_portable_body_on(
            &device,
            &[(Repr::Q2k, Repr::Q2k), (Repr::Q3k, Repr::Q3k)],
        );
    }
}

#[test]
fn dense_expand_grid_formats_match_their_portable_body() {
    for device in devices() {
        dense_expand_matches_its_portable_body_on(
            &device,
            &[(Repr::Iq2xs, Repr::Iq3xxs), (Repr::Iq2xxs, Repr::Iq2xxs)],
        );
    }
}

#[test]
fn dense_expand_gguf_formats_match_their_portable_body() {
    for device in devices() {
        dense_expand_matches_its_portable_body_on(
            &device,
            &[
                (Repr::Q4g32s, Repr::Q4g32s),
                (Repr::Iq4g32s, Repr::Q4g32s),
                (Repr::Mxfp4, Repr::Q4g32s),
                // Weights that share their rows share an operand recipe.
                (Repr::Nvfp4, Repr::Nvfp4),
                (Repr::Nvfp4A4, Repr::Nvfp4A4),
                (Repr::Q5g32s, Repr::Q5g32),
                (Repr::Q4g32, Repr::Q4g32),
            ],
        );
    }
}

/// `dense_up` (Nemotron-H's ReLU² shared expert) over NVFP4 and Q4_0 weights
/// at every row class, at each backend's declared default configuration,
/// against its portable body.
#[test]
fn dense_up_gguf_formats_match_their_portable_body() {
    let module = module();
    let act = Act::Bf16;
    let (h, f) = (256, 40);
    for repr in [Repr::Nvfp4, Repr::Nvfp4A4, Repr::Q4g32s] {
        let up = weight(repr, f, h, 21, 1.0);
        let norm = norm_values(h, 5);
        for outputs in [1usize, 2, 3, 6, 12, 18, 70] {
            let rows = outputs + 2;
            let out_rows = (0..outputs).map(|i| ((i * 5 + 1) % rows) as i32).collect::<Vec<_>>();
            let mut rng = Rng::new(rows as u64 + 7);
            let residual = (0..rows * h).map(|_| rng.symmetric() * 3.0).collect::<Vec<_>>();
            let outcome = interpret(
                &module,
                "dense_up",
                &[
                    ("NW", registry::dense(DType::BF16)),
                    ("UW", repr.storage()),
                    ("A", registry::dense(DType::BF16)),
                ],
                vec![
                    floats(&[rows, h], &residual),
                    activations(Act::Bf16, &[h], &norm),
                    Input::Data(up.oracle()),
                    ints(&[outputs], &out_rows),
                    Input::F32(1e-6),
                    Input::I32(2),
                    floats(&[0], &[]),
                ],
            );
            let oracle = result(&outcome, 0);
            // The host model's bound: reassociation, the rounding flips of
            // the normalized rows, and one rounding of the product.
            let mut bound = Vec::new();
            for row in &out_rows {
                let source = *row as usize;
                let (x, slack) = rms_row(act, &residual[source * h..(source + 1) * h], &norm, 1e-6);
                let x = up.operand_rows(&x);
                for n in 0..f {
                    let (u, magnitude) = dot(&x, up.row(n));
                    let u = act.round(u);
                    let error = reassociation_bound(magnitude) + u.abs() * act.ulp()
                        + slack_dot(&slack, up.row(n));
                    let value = act.round(u.max(0.0) * u.max(0.0));
                    bound.push(1.2 * 2.0 * u.abs() * error + value.abs() * act.ulp() + 1e-6);
                }
            }
            for device in devices() {
                let implementation = dense_up::native_implementation(&device).unwrap().unwrap();
                let statics = implementation.statics.iter().fold(
                    NativeSpecialization::new(),
                    |statics, name| {
                        let value = match name.as_str() {
                            "H" => h,
                            "F" => f,
                            "US" => 0,
                            _ => panic!("unexpected static {name}"),
                        };
                        statics.with_static(name.clone(), value as u64)
                    },
                );
                let native = dense_up::native_for_device_with(
                    &device,
                    dense_up::Elements {
                        NW: Element::bf16(),
                        UW: repr.resident(&device),
                        A: act.element(),
                    },
                    &implementation.default_specialization(&statics).unwrap(),
                )
                .unwrap()
                .call(dense_up::Args {
                    residual: &f32_tensor(&device, &[rows, h], &residual),
                    norm: &bf16_norm(&device, &norm),
                    up_weight: &up.tensor(&device),
                    out_rows: &i32_tensor(&device, &[outputs], &out_rows),
                    eps: 1e-6,
                    activation: 2,
                    up_scale: &f32_tensor(&device, &[0], &[]),
                })
                .unwrap()
                .value;
                assert_within(
                    &format!("dense_up {repr:?} {:?} rows {outputs}", device.backend()),
                    &read_act(act, &native),
                    &oracle,
                    &bound,
                );
            }
        }
    }
}

/// Rows whose K is not a multiple of 256 (a partial last 256-value block):
/// Gemma's 704 (expert down) and 2112 (dense down) and Nemotron's 2688, on
/// the portable body at K = 704 and on the host reference over the row
/// classes.
#[test]
fn dense_output_gguf_formats_at_32_granular_k() {
    for device in devices() {
        dense_output_gguf_formats_at_32_granular_k_on(&device);
    }
}

fn dense_output_gguf_formats_at_32_granular_k_on(device: &Device) {
    let module = module();
    let act = Act::Bf16;
    for repr in GGUF_REPRS {
        let (h, f) = (16, 704);
        let down = weight(repr, h, f, 41, 1.0);
        let (rows, out_rows) = (3usize, vec![2, 0]);
        let mut rng = Rng::new(43);
        let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
        let product = (0..out_rows.len() * f)
            .map(|_| act.round(rng.symmetric()))
            .collect::<Vec<_>>();
        let outcome = interpret(
            &module,
            "dense_output",
            &[("A", registry::dense(DType::BF16)), ("DW", repr.storage())],
            vec![
                floats(&[rows, h], &residual),
                activations(act, &[out_rows.len(), f], &product),
                Input::Data(down.oracle()),
                ints(&[out_rows.len()], &out_rows),
                floats(&[0], &[]),
                // No packing bound.
                floats(&[0], &[]),
            ],
        );
        let oracle = result(&outcome, 0);
        let (_, bound) = dense_output_reference(act, &down, &residual, &product, &out_rows);
        for mapping in MAPPINGS {
            let native = dense_output_native(
                &device, act, &down, &residual, rows, &product, &out_rows, mapping,
            );
            assert_within(
                &format!("dense_output portable {repr:?} K {f} {mapping:?}"),
                &native,
                &oracle,
                &bound,
            );
        }
        for f in [704usize, 2112, 2688] {
            let h = 72;
            let down = weight(repr, h, f, 42 + f as u64, 1.0);
            for rows in ROW_CLASSES {
                let mut rng = Rng::new(rows as u64 + f as u64);
                let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
                let out_rows = (0..rows)
                    .map(|i| ((i * 5 + 1) % rows) as i32)
                    .collect::<Vec<_>>();
                let product = (0..rows * f)
                    .map(|_| act.round(rng.symmetric()))
                    .collect::<Vec<_>>();
                let (expected, bound) = under(out_rows.len(), || {
                    dense_output_reference(act, &down, &residual, &product, &out_rows)
                });
                for mapping in MAPPINGS {
                    let native = dense_output_native(
                        &device, act, &down, &residual, rows, &product, &out_rows, mapping,
                    );
                    assert_within(
                        &format!("dense_output {repr:?} K {f} M {rows} {mapping:?}"),
                        &native,
                        &expected,
                        &bound,
                    );
                }
            }
        }
    }
}

#[test]
fn metal_dense_output_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let (h, f) = (40, 256);
    let down = weight(Repr::Q4k, h, f, 31, 1.0);
    for rows in [1usize, 8, 32, 128] {
        let mut rng = Rng::new(rows as u64);
        let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
        let product = (0..rows * f)
            .map(|_| act.round(rng.symmetric()))
            .collect::<Vec<_>>();
        let out_rows = (0..rows as i32).collect::<Vec<_>>();
        let (expected, bound) = under(rows, || {
            dense_output_reference(act, &down, &residual, &product, &out_rows)
        });
        let splits: &[u64] = if rows == 128 { &[1, 2, 4] } else { &[1] };
        for &split in splits {
            let actual = dense_output_native(
                &device,
                act,
                &down,
                &residual,
                rows,
                &product,
                &out_rows,
                gemm_mapping(64, 64, split),
            );
            assert_within(
                &format!("scoped dense_output rows {rows} split {split}"),
                &actual,
                &expected,
                &bound,
            );
        }
    }
}

/// `dense_output`'s matrix GEMV (the batched launch over Q4_K, Q5_K and Q6_K
/// weights), from two rows (BATCH_FROM 2) to sixteen, with every BATCH_PARTS
/// and weight rows that do not fill a fragment, against the host reference.
#[test]
fn metal_dense_output_matrix_gemv_matches_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    for repr in [Repr::Q4k, Repr::Q5k, Repr::Q6k] {
        for (h, f) in [(40usize, 256usize), (67, 512)] {
            let down = weight(repr, h, f, 37, 1.0);
            for rows in [2usize, 3, 4, 8, 9, 16] {
                let mut rng = Rng::new(rows as u64 + 11);
                let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
                let product = (0..rows * f)
                    .map(|_| act.round(rng.symmetric()))
                    .collect::<Vec<_>>();
                let out_rows = (0..rows as i32).collect::<Vec<_>>();
                let (expected, bound) = under(rows, || {
                    dense_output_reference(act, &down, &residual, &product, &out_rows)
                });
                for (batch_simdgroups, batch_parts) in [(8, 2), (4, 1), (16, 4), (4, 4), (32, 2)] {
                    let actual = dense_output_native(
                        &device,
                        act,
                        &down,
                        &residual,
                        rows,
                        &product,
                        &out_rows,
                        Mapping {
                            batch_from: 2,
                            batch_simdgroups,
                            batch_parts,
                            ..gemm_mapping(64, 64, 1)
                        },
                    );
                    assert_within(
                        &format!(
                            "matrix dense_output {repr:?} {h}x{f} rows {rows} simdgroups {batch_simdgroups} parts {batch_parts}"
                        ),
                        &actual,
                        &expected,
                        &bound,
                    );
                }
            }
        }
    }
}

fn dense_output_matches_its_portable_body_and_the_host_reference_on(
    device: &Device,
    reprs: &[Repr],
) {
    let module = module();
    let act = Act::Bf16;
    for &repr in reprs {
        // Portable body at small shapes.
        let (h, f) = (40, 256);
        let down = weight(repr, h, f, 31, 1.0);
        for (rows, out_rows) in [
            (3usize, vec![1, 2]),
            (20, (0..16).map(|i| (i * 3 + 2) % 20).collect::<Vec<i32>>()),
        ] {
            let mut rng = Rng::new(rows as u64);
            let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let product = (0..out_rows.len() * f)
                .map(|_| act.round(rng.symmetric()))
                .collect::<Vec<_>>();
            let outcome = interpret(
                &module,
                "dense_output",
                &[("A", registry::dense(DType::BF16)), ("DW", repr.storage())],
                vec![
                    floats(&[rows, h], &residual),
                    activations(act, &[out_rows.len(), f], &product),
                    Input::Data(down.oracle()),
                    ints(&[out_rows.len()], &out_rows),
                    floats(&[0], &[]),
                    // No packing bound.
                    floats(&[0], &[]),
                ],
            );
            let oracle = result(&outcome, 0);
            for mapping in MAPPINGS {
                let (_, bound) = under(out_rows.len(), || {
                    dense_output_reference(act, &down, &residual, &product, &out_rows)
                });
                let native = dense_output_native(
                    &device, act, &down, &residual, rows, &product, &out_rows, mapping,
                );
                assert_within(
                    &format!("dense_output portable {repr:?} rows {rows} {mapping:?}"),
                    &native,
                    &oracle,
                    &bound,
                );
            }
        }
        // Host reference over the row classes, with a row tail (H = 300).
        let (h, f) = (300, 1024);
        let down = weight(repr, h, f, 32, 1.0);
        for rows in ROW_CLASSES {
            let mut rng = Rng::new(rows as u64 + 100);
            let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let out_rows = (0..rows)
                .map(|i| ((i * 5 + 1) % rows) as i32)
                .collect::<Vec<_>>();
            let product = (0..rows * f)
                .map(|_| act.round(rng.symmetric()))
                .collect::<Vec<_>>();
            for mapping in MAPPINGS {
                let (expected, bound) = under(out_rows.len(), || {
                    dense_output_reference(act, &down, &residual, &product, &out_rows)
                });
                let native = dense_output_native(
                    &device, act, &down, &residual, rows, &product, &out_rows, mapping,
                );
                assert_within(
                    &format!("dense_output {repr:?} M {rows} {mapping:?}"),
                    &native,
                    &expected,
                    &bound,
                );
            }
        }
    }
}

/// The decode and verify row classes at the pinned Qwen3.5 4B geometry
/// (H 2560, F 9216) with the decode mapping: every M from 1 to 16 and the
/// first GEMM class 17, against the host reference. The GEMV class (M below
/// BATCH_FROM) must reproduce each row's M = 1 result bit for bit, so a
/// speculative verify row computes exactly what plain decode computes.
#[test]
fn dense_entries_at_4b_geometry_match_the_host_reference_and_single_rows() {
    for device in devices() {
        dense_entries_at_4b_geometry_match_the_host_reference_and_single_rows_on(&device);
    }
}

fn dense_entries_at_4b_geometry_match_the_host_reference_and_single_rows_on(device: &Device) {
    let act = Act::Bf16;
    let (h, f) = (2560, 9216);
    let mapping = gemm_mapping(64, 64, 1);
    let gate = weight(Repr::Q4k, f, h, 71, 1.0);
    let up = weight(Repr::Q4k, f, h, 72, 1.0);
    let norm = norm_values(h, 8);
    let down6 = weight(Repr::Q6k, h, f, 73, 1.0);
    let down4 = weight(Repr::Q4k, h, f, 74, 1.0);
    let rows_max = 17usize;
    let mut rng = Rng::new(75);
    let residual = (0..rows_max * h)
        .map(|_| rng.symmetric() * 3.0)
        .collect::<Vec<_>>();
    let product = (0..rows_max * f)
        .map(|_| act.round(rng.symmetric()))
        .collect::<Vec<_>>();
    let expand_single = (0..rows_max)
        .map(|r| {
            dense_expand_native(
                &device,
                act,
                &gate,
                &up,
                &residual,
                rows_max,
                &norm,
                &[r as i32],
                1e-6,
                mapping,
            )
        })
        .collect::<Vec<_>>();
    let output_single = |down: &Weight| {
        (0..rows_max)
            .map(|r| {
                dense_output_native(
                    &device,
                    act,
                    down,
                    &residual,
                    rows_max,
                    &product[r * f..(r + 1) * f],
                    &[r as i32],
                    mapping,
                )
            })
            .collect::<Vec<_>>()
    };
    let (single6, single4) = (output_single(&down6), output_single(&down4));
    for rows in 1..=rows_max {
        let out_rows = (0..rows as i32).collect::<Vec<_>>();
        let (expected, bound) = under(rows, || {
            dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6)
        });
        let native = dense_expand_native(
            &device, act, &gate, &up, &residual, rows_max, &norm, &out_rows, 1e-6, mapping,
        );
        assert_within(
            &format!("dense_expand 4B M {rows}"),
            &native,
            &expected,
            &bound,
        );
        for (label, down, single) in [("q6k", &down6, &single6), ("q4k", &down4, &single4)] {
            let (expected, bound) = under(rows, || {
                dense_output_reference(act, down, &residual, &product[..rows * f], &out_rows)
            });
            let output = dense_output_native(
                &device,
                act,
                down,
                &residual,
                rows_max,
                &product[..rows * f],
                &out_rows,
                mapping,
            );
            assert_within(
                &format!("dense_output {label} 4B M {rows}"),
                &output,
                &expected,
                &bound,
            );
            if (rows as u64) < mapping.batch_from {
                for r in 0..rows {
                    assert!(
                        output[r * h..(r + 1) * h]
                            .iter()
                            .zip(&single[r])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "dense_output {label} 4B M {rows} row {r} differs from its M = 1 result"
                    );
                }
            }
        }
        if (rows as u64) < mapping.batch_from {
            for r in 0..rows {
                assert!(
                    native[r * f..(r + 1) * f]
                        .iter()
                        .zip(&expand_single[r])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "dense_expand 4B M {rows} row {r} differs from its M = 1 result"
                );
            }
        }
    }
}

/// Which cases a segmented projection check runs: the small and 4B
/// geometries over every mapping, the scoped launches, or the IQ4_NL
/// weights at a K that is not a multiple of 256.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cases {
    Checked,
    Scoped,
    Iq4Blocks,
}

#[test]
fn recurrent_project_matches_its_portable_body_and_the_host_reference() {
    for device in devices() {
        recurrent_project_matches_its_portable_body_and_the_host_reference_on(&device);
    }
}

/// QKV and alpha weights in the IQ4_NL form (iq4g32s) beside
/// weights of other kinds in one segmented launch.
#[test]
fn recurrent_project_over_iq4_blocks_matches_its_portable_body_and_the_host_reference() {
    for device in devices() {
        recurrent_project_cases_on(&device, Cases::Iq4Blocks);
    }
}

#[test]
fn metal_gated_delta_project_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    recurrent_project_cases_on(&device, Cases::Scoped);
}

fn recurrent_project_matches_its_portable_body_and_the_host_reference_on(device: &Device) {
    recurrent_project_cases_on(device, Cases::Checked);
}

fn recurrent_project_cases_on(device: &Device, cases: Cases) {
    let module = module();
    let act = Act::Bf16;
    let scoped = cases == Cases::Scoped;
    // A small geometry over every mapping, then the pinned 4B geometry over
    // the decode and verify rows with the decode mapping; each with its qkv,
    // gate, alpha and beta representations.
    let k_quants = [Repr::Q5k, Repr::Q4k, Repr::Q8, Repr::Q8];
    let iq4_blocks = [Repr::Iq4g32s, Repr::Q4g32s, Repr::Iq4g32s, Repr::Q8];
    let geometries = match cases {
        Cases::Scoped => vec![(256usize, 1usize, 1usize, 64usize, false, k_quants)],
        Cases::Checked => vec![
            (512, 2, 4, 32, false, k_quants),
            (2560, 16, 32, 128, true, k_quants),
        ],
        Cases::Iq4Blocks => vec![(640, 2, 4, 32, false, iq4_blocks)],
    };
    for (h, nk, nv, w, four_b, reprs) in geometries {
        let rows_of = [(2 * nk + nv) * w, nv * w, nv, nv];
        let total: usize = rows_of.iter().sum();
        let weights = (0..4)
            .map(|i| weight(reprs[i], rows_of[i], h, 40 + i as u64, 1.0))
            .collect::<Vec<_>>();
        let norm = norm_values(h, 5);
        let reference = |hidden: &[f32], rows: usize| {
            let mut expected = Vec::new();
            let mut bound = Vec::new();
            for r in 0..rows {
                let (x, slack) = rms_row(act, &hidden[r * h..(r + 1) * h], &norm, 1e-6);
                for wt in &weights {
                    let x = wt.operand_rows(&x);
                    for n in 0..wt.rows {
                        let (acc, magnitude) = dot(&x, wt.row(n));
                        expected.push(act.round(acc));
                        bound.push(
                            rounded_bound(act, acc, magnitude) + slack_dot(&slack, wt.row(n)),
                        );
                    }
                }
            }
            (expected, bound)
        };
        let native = |hidden: &[f32], rows: usize, mapping: Mapping| {
            let kernel = gated_delta_project::native_for_device_with(
                &device,
                gated_delta_project::Elements {
                    NW: Element::bf16(),
                    QW: reprs[0].resident(&device),
                    GW: reprs[1].resident(&device),
                    AW: reprs[2].resident(&device),
                    BW: reprs[3].resident(&device),
                    A: act.element(),
                },
                &segmented_projection_specialization_on(
                    device,
                    &[("H", h), ("NK", nk), ("NV", nv), ("W", w)],
                    mapping,
                    RECURRENT_PROJECT_PACKED_LAUNCH,
                ),
            )
            .unwrap();
            let out = kernel
                .call(gated_delta_project::Args {
                    hidden: &f32_tensor(&device, &[rows, h], hidden),
                    input_norm: &bf16_norm(&device, &norm),
                    qkv_weight: &weights[0].tensor(&device),
                    gate_weight: &weights[1].tensor(&device),
                    alpha_weight: &weights[2].tensor(&device),
                    beta_weight: &weights[3].tensor(&device),
                    epsilon: 1e-6,
                    qkv_bound: &absent_bound(&device),
                    gate_bound: &absent_bound(&device),
                })
                .unwrap()
                .value;
            read_act(act, &out)
        };
        let (row_classes, mappings) = if scoped {
            (vec![1, 8, 32, 128], vec![MAPPINGS[0], MAPPINGS[2]])
        } else {
            checked_classes(four_b)
        };
        let portable_rows: &[usize] = if four_b || scoped { &[] } else { &[3, 17] };
        for &rows in portable_rows {
            let mut rng = Rng::new(rows as u64 + 3);
            let hidden = (0..rows * h)
                .map(|_| rng.symmetric() * 2.0)
                .collect::<Vec<_>>();
            let outcome = interpret(
                &module,
                "gated_delta_project",
                &[
                    ("NW", registry::dense(DType::BF16)),
                    ("QW", reprs[0].storage()),
                    ("GW", reprs[1].storage()),
                    ("AW", reprs[2].storage()),
                    ("BW", reprs[3].storage()),
                    ("A", registry::dense(DType::BF16)),
                ],
                vec![
                    floats(&[rows, h], &hidden),
                    activations(Act::Bf16, &[h], &norm),
                    Input::Data(weights[0].oracle()),
                    Input::Data(weights[1].oracle()),
                    Input::Data(weights[2].oracle()),
                    Input::Data(weights[3].oracle()),
                    Input::F32(1e-6),
                    // No packing bounds.
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                ],
            );
            let oracle = result(&outcome, 0);
            assert_eq!(oracle.len(), rows * total);
            for mapping in MAPPINGS {
                let (_, bound) = under(rows, || reference(&hidden, rows));
                assert_within(
                    &format!("recurrent_project portable rows {rows} {mapping:?}"),
                    &native(&hidden, rows, mapping),
                    &oracle,
                    &bound,
                );
            }
        }
        for &rows in &row_classes {
            let mut rng = Rng::new(rows as u64 + 9);
            let hidden = (0..rows * h)
                .map(|_| rng.symmetric() * 2.0)
                .collect::<Vec<_>>();
            for &mapping in &mappings {
                let (expected, bound) = under(rows, || reference(&hidden, rows));
                assert_within(
                    &format!("recurrent_project H {h} M {rows} {mapping:?}"),
                    &native(&hidden, rows, mapping),
                    &expected,
                    &bound,
                );
            }
        }
    }
}

/// The recurrent output projection: `attention_output` over the gated rows
/// [M, NV, W] the recurrent state entries publish.
#[test]
fn recurrent_output_matches_its_portable_body_and_the_host_reference() {
    for device in devices() {
        recurrent_output_matches_its_portable_body_and_the_host_reference_on(&device);
    }
}

#[test]
fn metal_recurrent_output_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    recurrent_output_cases_on(&device, true);
}

fn recurrent_output_matches_its_portable_body_and_the_host_reference_on(device: &Device) {
    recurrent_output_cases_on(device, false);
}

fn recurrent_output_cases_on(device: &Device, scoped: bool) {
    let module = module();
    let act = Act::Bf16;
    // A small geometry over every representation and mapping, then the pinned
    // 4B geometry (q5k) over the decode and verify rows.
    let geometries = if scoped {
        vec![(300usize, 2usize, 128usize, false)]
    } else {
        vec![(300usize, 2usize, 128usize, false), (2560, 32, 128, true)]
    };
    for (h, nv, w, four_b) in geometries {
        let k = nv * w;
        let (row_classes, mappings) = if scoped {
            (vec![1, 8, 32, 128], vec![MAPPINGS[0], MAPPINGS[2]])
        } else {
            checked_classes(four_b)
        };
        let reprs: &[Repr] = if four_b || scoped {
            &[Repr::Q5k]
        } else {
            &[Repr::Q5k, Repr::Q6k, Repr::Bf16]
        };
        for &repr in reprs {
            let out_weight = weight(repr, h, k, 50, 1.0);
            let case = |rows: usize| {
                let mut rng = Rng::new(rows as u64 + 11);
                let hidden = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
                let gated = (0..rows * k)
                    .map(|_| act.round(rng.symmetric()))
                    .collect::<Vec<_>>();
                (hidden, gated)
            };
            let reference = |hidden: &[f32], gated: &[f32], rows: usize| {
                let mut expected = Vec::new();
                let mut bound = Vec::new();
                for r in 0..rows {
                    for n in 0..h {
                        let (acc, magnitude) = dot(&out_weight.operand_rows(&gated[r * k..(r + 1) * k]), out_weight.row(n));
                        expected.push(hidden[r * h + n] + act.round(acc));
                        bound.push(rounded_bound(act, acc, magnitude));
                    }
                }
                (expected, bound)
            };
            let native = |hidden: &[f32], gated: &[f32], rows: usize, mapping: Mapping| {
                let kernel = attention_output::native_for_device_with(
                    &device,
                    attention_output::Elements {
                        OW: repr.resident(&device),
                        A: act.element(),
                    },
                    &attention_output_specialization_on(
                        device,
                        &[("D", h), ("Q", nv), ("W", w)],
                        mapping,
                    ),
                )
                .unwrap();
                let out = kernel
                    .call(attention_output::Args {
                        hidden: &f32_tensor(&device, &[rows, h], hidden),
                        inject: &no_inject(&device, rows as u64),
                        gated: &act_tensor(&device, act, &[rows, nv, w], gated),
                        output_weight: &out_weight.tensor(&device),
                        output_scale: &no_scale(&device),
                        output_bound: &absent_bound(&device),
                    })
                    .unwrap()
                    .value;
                read_f32(&out)
            };
            let portable_rows: &[usize] = if four_b || scoped { &[] } else { &[2, 12] };
            for &rows in portable_rows {
                let (hidden, gated) = case(rows);
                let outcome = interpret(
                    &module,
                    "attention_output",
                    &[("OW", repr.storage()), ("A", registry::dense(DType::BF16))],
                    vec![
                        floats(&[rows, h], &hidden),
                        floats(&[rows, 0], &[]),
                        activations(act, &[rows, nv, w], &gated),
                        Input::Data(out_weight.oracle()),
                        // No scale, no packing bound.
                        floats(&[0], &[]),
                        floats(&[0], &[]),
                    ],
                );
                let oracle = result(&outcome, 0);
                for mapping in MAPPINGS {
                    let (_, bound) = under(rows, || reference(&hidden, &gated, rows));
                    assert_within(
                        &format!("recurrent_output portable {repr:?} rows {rows} {mapping:?}"),
                        &native(&hidden, &gated, rows, mapping),
                        &oracle,
                        &bound,
                    );
                }
            }
            for &rows in &row_classes {
                let (hidden, gated) = case(rows);
                for &mapping in &mappings {
                    let (expected, bound) = under(rows, || reference(&hidden, &gated, rows));
                    assert_within(
                        &format!("recurrent_output {repr:?} H {h} M {rows} {mapping:?}"),
                        &native(&hidden, &gated, rows, mapping),
                        &expected,
                        &bound,
                    );
                }
            }
        }
    }
}

#[test]
fn attention_projections_match_their_portable_bodies_and_the_host_reference() {
    for device in devices() {
        attention_projections_match_their_portable_bodies_and_the_host_reference_on(
            &device,
            Cases::Checked,
        );
    }
}

/// Query and value weights in the IQ4_NL form (iq4g32s, rows
/// whose width is not a multiple of 256) beside a key weight of another
/// kind in one segmented launch, and an iq4g32s output projection.
#[test]
fn attention_projections_over_iq4_blocks_match_their_portable_bodies_and_the_host_reference() {
    for device in devices() {
        attention_projections_match_their_portable_bodies_and_the_host_reference_on(
            &device,
            Cases::Iq4Blocks,
        );
    }
}

#[test]
fn metal_attention_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    attention_projections_match_their_portable_bodies_and_the_host_reference_on(
        &device,
        Cases::Scoped,
    );
}

#[test]
fn cuda_attention_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Cuda)
    else {
        return;
    };
    attention_projections_match_their_portable_bodies_and_the_host_reference_on(
        &device,
        Cases::Scoped,
    );
}

#[test]
fn vulkan_attention_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Vulkan)
    else {
        return;
    };
    attention_projections_match_their_portable_bodies_and_the_host_reference_on(
        &device,
        Cases::Scoped,
    );
}

#[test]
fn cpu_attention_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Cpu)
    else {
        return;
    };
    attention_projections_match_their_portable_bodies_and_the_host_reference_on(
        &device,
        Cases::Scoped,
    );
}

fn attention_projections_match_their_portable_bodies_and_the_host_reference_on(
    device: &Device,
    cases: Cases,
) {
    let module = module();
    let act = Act::Bf16;
    let scoped = cases == Cases::Scoped;
    // A small geometry over every mapping, then the pinned 4B geometry over
    // the decode and verify rows with the decode mapping; each with its
    // query, key and value representations.
    let k_quants = [Repr::Q4k, Repr::Q4k, Repr::Q6k];
    let iq4_blocks = [Repr::Iq4g32s, Repr::Q8, Repr::Iq4g32s];
    let geometries = match cases {
        Cases::Scoped => vec![(256usize, 1usize, 4usize, 64usize, false, k_quants)],
        Cases::Checked => vec![
            (512, 2, 2, 64, false, k_quants),
            (2560, 4, 4, 256, true, k_quants),
        ],
        Cases::Iq4Blocks => vec![(640, 2, 2, 64, false, iq4_blocks)],
    };
    for (d, kv, g, w, four_b, reprs) in geometries {
        let (row_classes, mappings) = if scoped {
            (vec![1, 8, 32, 128], vec![MAPPINGS[0], MAPPINGS[2]])
        } else {
            checked_classes(four_b)
        };
        let rows_of = [kv * g * 2 * w, kv * w, kv * w];
        let weights = (0..3)
            .map(|i| weight(reprs[i], rows_of[i], d, 60 + i as u64, 1.0))
            .collect::<Vec<_>>();
        let norm = norm_values(d, 7);
        let project = |hidden: &[f32], rows: usize, mapping: Mapping, project_mode: i32| {
            let kernel = attention_project::native_for_device_with(
                &device,
                attention_project::Elements {
                    NW: Element::bf16(),
                    QW: reprs[0].resident(&device),
                    GW: Element::bf16(),
                    KW: reprs[1].resident(&device),
                    VW: reprs[2].resident(&device),
                    A: act.element(),
                },
                &attention_project_specialization_on(
                    device,
                    &attention_project_statics(d, kv, g, w),
                    mapping,
                ),
            )
            .unwrap();
            let out = kernel
                .call(attention_project::Args {
                    hidden: &f32_tensor(&device, &[rows, d], hidden),
                    input_norm: &bf16_norm(&device, &norm),
                    query_weight: &weights[0].tensor(&device),
                    gate_weight: &act_tensor(&device, Act::Bf16, &[0, d], &[]),
                    key_weight: &weights[1].tensor(&device),
                    value_weight: &weights[2].tensor(&device),
                    epsilon: 1e-6,
                    project_mode,
                    query_scale: &no_scale(&device),
                    gate_scale: &no_scale(&device),
                    key_scale: &no_scale(&device),
                    value_scale: &no_scale(&device),
                    query_bound: &absent_bound(&device),
                    gate_bound: &absent_bound(&device),
                    key_bound: &absent_bound(&device),
                    value_bound: &absent_bound(&device),
                })
                .unwrap();
            [
                read_act(act, &out.r0),
                read_act(act, &out.r2),
                read_act(act, &out.r3),
            ]
        };
        let reference = |hidden: &[f32], rows: usize| {
            let mut out = [
                (Vec::new(), Vec::new()),
                (Vec::new(), Vec::new()),
                (Vec::new(), Vec::new()),
            ];
            for r in 0..rows {
                let (x, slack) = rms_row(act, &hidden[r * d..(r + 1) * d], &norm, 1e-6);
                for (i, wt) in weights.iter().enumerate() {
                    let x = wt.operand_rows(&x);
                    for n in 0..wt.rows {
                        let (acc, magnitude) = dot(&x, wt.row(n));
                        out[i].0.push(act.round(acc));
                        out[i].1.push(
                            rounded_bound(act, acc, magnitude) + slack_dot(&slack, wt.row(n)),
                        );
                    }
                }
            }
            out
        };
        let portable_rows: &[usize] = if four_b || scoped { &[] } else { &[2, 16] };
        for &rows in portable_rows {
            let mut rng = Rng::new(rows as u64 + 5);
            let hidden = (0..rows * d)
                .map(|_| rng.symmetric() * 2.0)
                .collect::<Vec<_>>();
            let outcome = interpret(
                &module,
                "attention_project",
                &[
                    ("NW", registry::dense(DType::BF16)),
                    ("QW", reprs[0].storage()),
                    ("GW", registry::dense(DType::BF16)),
                    ("KW", reprs[1].storage()),
                    ("VW", reprs[2].storage()),
                    ("A", registry::dense(DType::BF16)),
                ],
                vec![
                    floats(&[rows, d], &hidden),
                    activations(Act::Bf16, &[d], &norm),
                    Input::Data(weights[0].oracle()),
                    activations(Act::Bf16, &[0, d], &[]),
                    Input::Data(weights[1].oracle()),
                    Input::Data(weights[2].oracle()),
                    Input::F32(1e-6),
                    Input::I32(0),
                    // No scales, no packing bounds.
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                    floats(&[0], &[]),
                ],
            );
            for mapping in MAPPINGS {
                let expected = under(rows, || reference(&hidden, rows));
                let native = project(&hidden, rows, mapping, 0);
                for i in 0..3 {
                    assert_within(
                        &format!("attention_project[{i}] portable rows {rows} {mapping:?}"),
                        &native[i],
                        // The results after the empty gate segment.
                        &result(&outcome, [0, 2, 3][i]),
                        &expected[i].1,
                    );
                }
            }
        }
        for &rows in &row_classes {
            let mut rng = Rng::new(rows as u64 + 55);
            let hidden = (0..rows * d)
                .map(|_| rng.symmetric() * 2.0)
                .collect::<Vec<_>>();
            for &mapping in &mappings {
                let expected = under(rows, || reference(&hidden, rows));
                let native = project(&hidden, rows, mapping, 0);
                for i in 0..3 {
                    assert_within(
                        &format!("attention_project[{i}] D {d} M {rows} {mapping:?}"),
                        &native[i],
                        &expected[i].0,
                        &expected[i].1,
                    );
                }
            }
        }
        if scoped {
            for rows in [1, 8, 128] {
                let mut rng = Rng::new(rows as u64 + 155);
                let hidden = (0..rows * d)
                    .map(|_| rng.symmetric() * 2.0)
                    .collect::<Vec<_>>();
                let expected = under(rows, || reference(&hidden, rows));
                let native = project(&hidden, rows, MAPPINGS[0], 1);
                assert!(native[0].iter().all(|&value| value == 0.0));
                for i in 1..3 {
                    assert_within(
                        &format!("attention_project injection[{i}] M {rows}"),
                        &native[i],
                        &expected[i].0,
                        &expected[i].1,
                    );
                }
            }
        }
        // Output projection over Q = KV * G gated heads.
        let q = kv * g;
        let reprs: &[Repr] = if four_b || scoped {
            &[Repr::Q4k]
        } else if cases == Cases::Iq4Blocks {
            &[Repr::Iq4g32s]
        } else {
            &[Repr::Q4k, Repr::Q8, Repr::Bf16]
        };
        for &repr in reprs {
            let (d_out, k) = (if four_b { d } else { 300usize }, q * w);
            let out_weight = weight(repr, d_out, k, 70, 1.0);
            let native = |hidden: &[f32], gated: &[f32], rows: usize, mapping: Mapping| {
                let kernel = attention_output::native_for_device_with(
                    &device,
                    attention_output::Elements {
                        A: act.element(),
                        OW: repr.resident(&device),
                    },
                    &attention_output_specialization_on(
                        device,
                        &[("D", d_out), ("Q", q), ("W", w)],
                        mapping,
                    ),
                )
                .unwrap();
                let out = kernel
                    .call(attention_output::Args {
                        hidden: &f32_tensor(&device, &[rows, d_out], hidden),
                        inject: &no_inject(&device, rows as u64),
                        gated: &act_tensor(&device, act, &[rows, q, w], gated),
                        output_weight: &out_weight.tensor(&device),
                        output_scale: &no_scale(&device),
                        output_bound: &absent_bound(&device),
                    })
                    .unwrap()
                    .value;
                read_f32(&out)
            };
            let reference = |hidden: &[f32], gated: &[f32], rows: usize| {
                let mut expected = Vec::new();
                let mut bound = Vec::new();
                for r in 0..rows {
                    for n in 0..d_out {
                        let (acc, magnitude) = dot(&out_weight.operand_rows(&gated[r * k..(r + 1) * k]), out_weight.row(n));
                        expected.push(hidden[r * d_out + n] + act.round(acc));
                        bound.push(rounded_bound(act, acc, magnitude));
                    }
                }
                (expected, bound)
            };
            for &rows in &row_classes {
                let mut rng = Rng::new(rows as u64 + 77);
                let hidden = (0..rows * d_out)
                    .map(|_| rng.symmetric())
                    .collect::<Vec<_>>();
                let gated = (0..rows * k)
                    .map(|_| act.round(rng.symmetric()))
                    .collect::<Vec<_>>();
                if !four_b && !scoped && (rows == 3 || rows == 16) {
                    let (_, bound) = under(rows, || reference(&hidden, &gated, rows));
                    let outcome = interpret(
                        &module,
                        "attention_output",
                        &[("A", registry::dense(DType::BF16)), ("OW", repr.storage())],
                        vec![
                            floats(&[rows, d_out], &hidden),
                            floats(&[rows, 0], &[]),
                            activations(act, &[rows, q, w], &gated),
                            Input::Data(out_weight.oracle()),
                            // No scale, no packing bound.
                            floats(&[0], &[]),
                            floats(&[0], &[]),
                        ],
                    );
                    let oracle = result(&outcome, 0);
                    assert_within(
                        &format!("attention_output portable {repr:?} rows {rows}"),
                        &native(&hidden, &gated, rows, MAPPINGS[0]),
                        &oracle,
                        &bound,
                    );
                }
                for &mapping in &mappings {
                    let (expected, bound) = under(rows, || reference(&hidden, &gated, rows));
                    assert_within(
                        &format!("attention_output {repr:?} D {d_out} M {rows} {mapping:?}"),
                        &native(&hidden, &gated, rows, mapping),
                        &expected,
                        &bound,
                    );
                }
            }
        }
    }
}

/// Projections under their weights' second-level scales (Nemotron's NVFP4
/// `.scale`s): every segment of `attention_project` (QS = GS = KS = VS = 1,
/// one scale each) and `attention_output` (OS = 1) multiply their F32
/// accumulators by their scale before rounding, against the portable bodies
/// and the host reference at every launch class, and on Metal in the PACK
/// form too. One test per backend; CUDA's is `cuda_projection_blocks.rs`'.
#[test]
fn cpu_scaled_projections_match_the_host() {
    scaled_projections_on_backend(BackendName::Cpu);
}

#[test]
fn metal_scaled_projections_match_the_host() {
    scaled_projections_on_backend(BackendName::Metal);
}

#[test]
fn vulkan_scaled_projections_match_the_host() {
    scaled_projections_on_backend(BackendName::Vulkan);
}

fn scaled_projections_on_backend(backend: BackendName) {
    if let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == backend)
    {
        scaled_projections_on(&device);
    }
}

fn scaled_projections_on(device: &Device) {
    /// The segments' second-level scales, query, gate, key, value: none a
    /// power of two, each distinct, so a scale bound to the wrong segment
    /// fails.
    const SCALES: [f32; 4] = [0.412_5, 0.731_2, 1.368_8, 0.207_5];
    /// The output projection's.
    const SCALE: f32 = 0.412_5;
    let module = module();
    let act = Act::Bf16;
    // Query, gate, key and value segments of 128, 64, 64 and 64 rows over a
    // 256-wide residual; an output projection of 4 heads of 64 channels
    // back to it.
    let (d, heads, width) = (256usize, 4usize, 64usize);
    let segments = [128usize, 64, 64, 64];
    let k = heads * width;
    let norm = norm_values(d, 17);
    let rows_classes = [1usize, 2, 8, 17, 64, 128];
    let mappings = [MAPPINGS[0], MAPPINGS[2]];
    let metal = device.backend() == BackendName::Metal;
    for repr in [Repr::Nvfp4, Repr::Nvfp4A4, Repr::Q4k] {
        let projections = [301, 311, 312, 313]
            .into_iter()
            .zip(segments)
            .map(|(seed, rows)| weight(repr, rows, d, seed, 1.0))
            .collect::<Vec<_>>();
        let output = weight(repr, d, k, 302, 1.0);
        let project = |hidden: &[f32], rows: usize, mapping: Mapping, pack: Option<Pack>| {
            let specialization = attention_project_specialization_on(
                device,
                &[
                    ("D", d),
                    ("Q", segments[0]),
                    ("GR", segments[1]),
                    ("K", segments[2]),
                    ("V", segments[3]),
                    ("QS", 1),
                    ("GS", 1),
                    ("KS", 1),
                    ("VS", 1),
                ],
                mapping,
            );
            let weights = projections
                .iter()
                .map(|projection| projection.tensor(device))
                .collect::<Vec<_>>();
            let scales = SCALES.map(|scale| f32_tensor(device, &[1], &[scale]));
            let out = attention_project::native_for_device_with(
                device,
                attention_project::Elements {
                    NW: Element::bf16(),
                    QW: repr.resident(device),
                    GW: repr.resident(device),
                    KW: repr.resident(device),
                    VW: repr.resident(device),
                    A: act.element(),
                },
                &match pack {
                    Some(pack) => with_pack(specialization, ATTENTION_PROJECT_PACKED_LAUNCH, pack),
                    None => specialization,
                },
            )
            .unwrap()
            .call(attention_project::Args {
                hidden: &f32_tensor(device, &[rows, d], hidden),
                input_norm: &bf16_norm(device, &norm),
                query_weight: &weights[0],
                gate_weight: &weights[1],
                key_weight: &weights[2],
                value_weight: &weights[3],
                epsilon: 1e-6,
                project_mode: 0,
                query_scale: &scales[0],
                gate_scale: &scales[1],
                key_scale: &scales[2],
                value_scale: &scales[3],
                query_bound: &absent_bound(device),
                gate_bound: &absent_bound(device),
                key_bound: &absent_bound(device),
                value_bound: &absent_bound(device),
            })
            .unwrap();
            [&out.r0, &out.r1, &out.r2, &out.r3].map(|segment| read_act(act, segment))
        };
        let output_native =
            |hidden: &[f32], gated: &[f32], rows: usize, mapping: Mapping, pack: Option<Pack>| {
                let specialization = attention_output_specialization_on(
                    device,
                    &[("D", d), ("Q", heads), ("W", width), ("OS", 1)],
                    mapping,
                );
                let out = attention_output::native_for_device_with(
                    device,
                    attention_output::Elements {
                        OW: repr.resident(device),
                        A: act.element(),
                    },
                    &match pack {
                        Some(pack) => {
                            with_pack(specialization, ATTENTION_OUTPUT_PACKED_LAUNCH, pack)
                        }
                        None => specialization,
                    },
                )
                .unwrap()
                .call(attention_output::Args {
                    hidden: &f32_tensor(device, &[rows, d], hidden),
                    inject: &no_inject(device, rows as u64),
                    gated: &act_tensor(device, act, &[rows, heads, width], gated),
                    output_weight: &output.tensor(device),
                    output_scale: &f32_tensor(device, &[1], &[SCALE]),
                    output_bound: &absent_bound(device),
                })
                .unwrap()
                .value;
                read_f32(&out)
            };
        // Each segment's scaled accumulators rounded to A, and their bounds.
        let project_reference = |hidden: &[f32], rows: usize| {
            std::array::from_fn::<_, 4, _>(|segment| {
                let (projection, scale) = (&projections[segment], SCALES[segment]);
                let (mut expected, mut bound) = (Vec::new(), Vec::new());
                for r in 0..rows {
                    let (x, slack) = rms_row(act, &hidden[r * d..(r + 1) * d], &norm, 1e-6);
                    let x = projection.operand_rows(&x);
                    for n in 0..segments[segment] {
                        let (acc, magnitude) = dot(&x, projection.row(n));
                        expected.push(act.round(scale * acc));
                        bound.push(
                            rounded_bound(act, scale * acc, scale * magnitude)
                                + scale * slack_dot(&slack, projection.row(n)),
                        );
                    }
                }
                (expected, bound)
            })
        };
        let output_reference = |hidden: &[f32], gated: &[f32], rows: usize| {
            let (mut expected, mut bound) = (Vec::new(), Vec::new());
            for r in 0..rows {
                for n in 0..d {
                    let (acc, magnitude) = dot(&output.operand_rows(&gated[r * k..(r + 1) * k]), output.row(n));
                    expected.push(hidden[r * d + n] + act.round(SCALE * acc));
                    bound.push(rounded_bound(act, SCALE * acc, SCALE * magnitude));
                }
            }
            (expected, bound)
        };
        let case = |rows: usize| {
            let mut rng = Rng::new(rows as u64 + 303);
            let hidden = (0..rows * d)
                .map(|_| rng.symmetric() * 2.0)
                .collect::<Vec<_>>();
            let gated = (0..rows * k)
                .map(|_| act.round(rng.symmetric()))
                .collect::<Vec<_>>();
            (hidden, gated)
        };

        // The portable bodies scale as the natives do.
        let rows = 2;
        let (hidden, gated) = case(rows);
        let projected = interpret(
            &module,
            "attention_project",
            &[
                ("NW", registry::dense(DType::BF16)),
                ("QW", repr.storage()),
                ("GW", repr.storage()),
                ("KW", repr.storage()),
                ("VW", repr.storage()),
                ("A", registry::dense(DType::BF16)),
            ],
            [
                floats(&[rows, d], &hidden),
                activations(Act::Bf16, &[d], &norm),
            ]
            .into_iter()
            .chain(projections.iter().map(|projection| Input::Data(projection.oracle())))
            .chain([Input::F32(1e-6), Input::I32(0)])
            // Every segment's scale; no bound.
            .chain(SCALES.map(|scale| floats(&[1], &[scale])))
            .chain((0..4).map(|_| floats(&[0], &[])))
            .collect(),
        );
        let outputted = interpret(
            &module,
            "attention_output",
            &[("OW", repr.storage()), ("A", registry::dense(DType::BF16))],
            vec![
                floats(&[rows, d], &hidden),
                floats(&[rows, 0], &[]),
                activations(act, &[rows, heads, width], &gated),
                Input::Data(output.oracle()),
                floats(&[1], &[SCALE]),
                // No packing bound.
                floats(&[0], &[]),
            ],
        );
        let project_reference_at = under(rows, || project_reference(&hidden, rows));
        let (_, output_bound) = under(rows, || output_reference(&hidden, &gated, rows));
        let projected_native = project(&hidden, rows, MAPPINGS[0], None);
        for (segment, (_, project_bound)) in project_reference_at.iter().enumerate() {
            assert_within(
                &format!("scaled attention_project portable {repr:?} segment {segment}"),
                &projected_native[segment],
                &result(&projected, segment),
                project_bound,
            );
        }
        assert_within(
            &format!("scaled attention_output portable {repr:?}"),
            &output_native(&hidden, &gated, rows, MAPPINGS[0], None),
            &result(&outputted, 0),
            &output_bound,
        );

        // Every launch class against the host reference.
        for rows in rows_classes {
            let (hidden, gated) = case(rows);
            let project_expected = under(rows, || project_reference(&hidden, rows));
            let (output_expected, output_bound) =
                under(rows, || output_reference(&hidden, &gated, rows));
            for mapping in mappings {
                let projected = project(&hidden, rows, mapping, None);
                for (segment, (expected, bound)) in project_expected.iter().enumerate() {
                    assert_within(
                        &format!(
                            "scaled attention_project {repr:?} M {rows} {mapping:?} segment {segment}"
                        ),
                        &projected[segment],
                        expected,
                        bound,
                    );
                }
                assert_within(
                    &format!("scaled attention_output {repr:?} M {rows} {mapping:?}"),
                    &output_native(&hidden, &gated, rows, mapping, None),
                    &output_expected,
                    &output_bound,
                );
            }
            // The PACK form rounds its operand: within the packed error of the
            // scaled projection.
            if metal && rows > 64 {
                let wide = |values: &[f32]| values.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
                let packed = project(&hidden, rows, pack_mapping(), Some(PACK_DEFAULT));
                for (segment, (expected, _)) in project_expected.iter().enumerate() {
                    let (relative, worst) =
                        relative_difference(&wide(expected), &wide(&packed[segment]));
                    assert!(
                        relative <= 2e-2 && worst <= 0.2,
                        "packed scaled attention_project {repr:?} M {rows} segment {segment}: \
                         {relative:.2e}, {worst:.2e}"
                    );
                }
                let projection_only = |values: &[f32]| {
                    values
                        .iter()
                        .zip(&hidden)
                        .map(|(value, hidden)| f64::from(value - hidden))
                        .collect::<Vec<_>>()
                };
                let packed = output_native(&hidden, &gated, rows, pack_mapping(), Some(PACK_DEFAULT));
                let (relative, worst) = relative_difference(
                    &projection_only(&output_expected),
                    &projection_only(&packed),
                );
                assert!(
                    relative <= 2e-2 && worst <= 0.2,
                    "packed scaled attention_output {repr:?} M {rows}: {relative:.2e}, {worst:.2e}"
                );
            }
        }
    }
}

/// One readout check: vocabulary V and width D, head representations, output
/// row classes, mappings, the softcap (0 = none), and whether the portable
/// bodies are interpreted at 2 output rows (Q6K).
struct ReadoutCheck {
    v: usize,
    d: usize,
    reprs: &'static [Repr],
    output_classes: &'static [usize],
    mappings: &'static [Mapping],
    softcap: f32,
    /// The head's second-level weight scale (`weight_scale`, WS = 1).
    weight_scale: Option<f32>,
    portable: bool,
}

#[test]
fn readout_entries_match_their_portable_bodies_and_the_host_reference() {
    let check = ReadoutCheck {
        v: 1000,
        d: 512,
        reprs: &[Repr::Q6k, Repr::Q4k, Repr::Q8, Repr::Q2k, Repr::Q4g32, Repr::Q3k],
        output_classes: &ROW_CLASSES,
        mappings: &MAPPINGS,
        softcap: 0.0,
        weight_scale: None,
        portable: true,
    };
    for device in devices() {
        readout_entries_match_their_portable_bodies_and_the_host_reference_on(&device, &check);
    }
}

/// Gemma's softcap 30: 2 rows (GEMV) and 12 (Metal's batched GEMV, Vulkan's
/// GEMM); every class publishes through the same logits epilogue.
#[test]
fn readout_softcap_matches_the_portable_bodies_and_the_host_reference() {
    const MAPPING: [Mapping; 1] = [MAPPINGS[0]];
    let check = ReadoutCheck {
        v: 256,
        d: 256,
        reprs: &[Repr::Q6k],
        output_classes: &[2, 12],
        mappings: &MAPPING,
        softcap: 30.0,
        weight_scale: None,
        portable: true,
    };
    for device in devices() {
        readout_entries_match_their_portable_bodies_and_the_host_reference_on(&device, &check);
    }
}

/// An NVFP4 vocabulary projection's second-level scale (Nemotron's
/// `output.scale` magnitude) on the F32 projection, before the softcap: the
/// portable body and every native against the host reference, on the GEMV,
/// batched and GEMM classes.
#[test]
fn readout_weight_scale_matches_the_portable_bodies_and_the_host_reference() {
    const MAPPING: [Mapping; 1] = [MAPPINGS[0]];
    for softcap in [0.0, 30.0] {
        let check = ReadoutCheck {
            v: 256,
            d: 256,
            reprs: &[Repr::Q6k],
            output_classes: &[1, 12, 40],
            mappings: &MAPPING,
            softcap,
            weight_scale: Some(3.0517578e-4),
            portable: true,
        };
        for device in devices() {
            readout_entries_match_their_portable_bodies_and_the_host_reference_on(&device, &check);
        }
    }
}

#[test]
fn metal_readout_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    const SCOPED: [Mapping; 2] = [MAPPINGS[0], MAPPINGS[2]];
    let check = ReadoutCheck {
        v: 256,
        d: 256,
        reprs: &[Repr::Q6k],
        output_classes: &[1, 8, 32, 128],
        mappings: &SCOPED,
        softcap: 0.0,
        weight_scale: None,
        portable: false,
    };
    readout_entries_match_their_portable_bodies_and_the_host_reference_on(&device, &check);
}

#[test]
fn metal_head_logits_scoped_launches_match_the_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    head_logits_match_the_host(
        &device,
        Repr::Q6k,
        (256, 256),
        &[1, 8, 32, 128],
        &[MAPPINGS[0], MAPPINGS[2]],
    );
}

/// The draft head readout (`readout_project_rows`) at the widths of real
/// draft heads (Qwen3.5 0.8B: 1024, Gemma 4 E2B: 1536, and a width whose
/// staging spans two GEMV chunks) over every weight form a tied embedding
/// table takes. The decode rows of a head's slots stage one activation row
/// per workgroup, with the per-16 sums of the biased kinds (q4g32s, q4k,
/// q6k) after the staged values.
#[test]
fn draft_head_logits_match_the_host_at_head_widths() {
    for device in devices() {
        for repr in [Repr::Q4g32s, Repr::Q4k, Repr::Q6k, Repr::Q8, Repr::Bf16] {
            for d in [1024, 1536, 9472] {
                head_logits_match_the_host(&device, repr, (40, d), &[1, 3], &MAPPINGS[..2]);
            }
        }
    }
}

fn head_logits_match_the_host(
    device: &Device,
    repr: Repr,
    (v, d): (usize, usize),
    row_classes: &[usize],
    mappings: &[Mapping],
) {
    let act = Act::Bf16;
    let head = weight(repr, v, d, 81, 1.0);
    for &rows in row_classes {
        let mut rng = Rng::new(rows as u64 + 31);
        let features = (0..rows * d)
            .map(|_| act.round(rng.symmetric()))
            .collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(rows * v);
        for row in 0..rows {
            for column in 0..v {
                expected.push(dot(&head.operand_rows(&features[row * d..(row + 1) * d]), head.row(column)));
            }
        }
        let values = expected.iter().map(|(value, _)| *value).collect::<Vec<_>>();
        let bounds = expected
            .iter()
            .map(|(_, magnitude)| reassociation(rows) * magnitude + 1e-7)
            .collect::<Vec<_>>();
        let mappings = if is_cpu(device) {
            &mappings[..1]
        } else {
            mappings
        };
        for &mapping in mappings {
            let specialization = if is_cpu(device) {
                let implementation = readout_project_rows::native_implementation(device)
                    .unwrap()
                    .unwrap();
                let statics = implementation
                    .statics
                    .iter()
                    .fold(NativeSpecialization::new(), |statics, name| {
                        statics.with_static(name.clone(), 0)
                    });
                implementation
                    .default_specialization(&statics)
                    .unwrap()
                    .with_param("INT8", 0)
            } else {
                prepared_projection_specialization_on(device, &[("V", v), ("D", d)], mapping)
            };
            let kernel = readout_project_rows::native_for_device_with(
                device,
                readout_project_rows::Elements {
                    A: act.element(),
                    OW: repr.resident(&device),
                },
                &specialization,
            )
            .unwrap();
            let logits = kernel
                .call(readout_project_rows::Args {
                    features: &act_tensor(device, act, &[rows, d], &features),
                    weight: &head.tensor(device),
                    token_rows: &Tensor::zeros(device, Element::i32(), &[rows as u64, 0, v as u64])
                        .unwrap(),
                    weight_scale: &Tensor::zeros(device, Element::f32(), &[0]).unwrap(),
                    softcap: 0.0,
                })
                .unwrap()
                .value;
            assert_within(
                &format!(
                    "{:?} head_logits {repr:?} V {v} D {d} O {rows} {mapping:?}",
                    device.backend()
                ),
                &read_f32(&logits),
                &values,
                &bounds,
            );
        }
    }
}

fn readout_entries_match_their_portable_bodies_and_the_host_reference_on(
    device: &Device,
    check: &ReadoutCheck,
) {
    let module = module();
    let act = Act::Bf16;
    let (v, d, cap) = (check.v, check.d, check.softcap);
    for &repr in check.reprs {
        let head = weight(repr, v, d, 80, 1.0);
        let norm = norm_values(d, 8);
        let selected = (0..37)
            .map(|i| ((i * 37 + 11) % v) as i32)
            .collect::<Vec<_>>();
        for &outputs in check.output_classes {
            let rows = outputs.max(2) + 1;
            let mut rng = Rng::new(outputs as u64 + 13);
            let hidden = (0..rows * d)
                .map(|_| rng.symmetric() * 2.0)
                .collect::<Vec<_>>();
            let out_rows = (0..outputs)
                .map(|i| ((i * 3 + 2) % rows) as i32)
                .collect::<Vec<_>>();
            let mut features = Vec::new();
            let mut feature_bound = Vec::new();
            let mut logits = Vec::new();
            let mut logit_bound = Vec::new();
            let mut chosen = Vec::new();
            let mut chosen_bound = Vec::new();
            for row in &out_rows {
                let source = *row as usize;
                let (x, slack) = rms_row(act, &hidden[source * d..(source + 1) * d], &norm, 1e-6);
                features.extend_from_slice(&x);
                feature_bound.extend_from_slice(&slack);
                let x = head.operand_rows(&x);
                for n in 0..v {
                    let (acc, magnitude) = dot(&x, head.row(n));
                    logits.push(acc);
                    logit_bound.push((magnitude, 1e-7 + slack_dot(&slack, head.row(n))));
                }
                for n in &selected {
                    let (acc, magnitude) = dot(&x, head.row(*n as usize));
                    chosen.push(acc);
                    chosen_bound.push((magnitude, 1e-7 + slack_dot(&slack, head.row(*n as usize))));
                }
            }
            let hidden_tensor = f32_tensor(&device, &[rows, d], &hidden);
            let rows_tensor = i32_tensor(&device, &[outputs], &out_rows);
            let features_native = readout_features_rows::native_for_device_with(
                &device,
                readout_features_rows::Elements {
                    NW: Element::bf16(),
                    A: act.element(),
                },
                &formed(device, statics_on(device, &[("D", d)]), d),
            )
            .unwrap()
            .call(readout_features_rows::Args {
                hidden: &one_part(&hidden_tensor),
                norm: &bf16_norm(&device, &norm),
                out_rows: &rows_tensor,
                epsilon: 1e-6,
            })
            .unwrap()
            .value;
            assert_within(
                &format!("features_rows O {outputs}"),
                &read_act(act, &features_native),
                &features,
                &feature_bound,
            );
            // The head's weight scale multiplies each exact logit and its
            // bound.
            let factor = check.weight_scale.unwrap_or(1.0);
            let scale_extent = usize::from(check.weight_scale.is_some());
            let scale_values = check.weight_scale.into_iter().collect::<Vec<_>>();
            let scaled_logits = logits.iter().map(|z| z * factor).collect::<Vec<_>>();
            let scaled_logit_bound = logit_bound
                .iter()
                .map(|(magnitude, rest)| (magnitude * factor, rest * factor))
                .collect::<Vec<_>>();
            // The softcap c * tanh(z / c) has slope at most 1, so a logit's
            // bound carries over; its exp form adds a few ulp of c.
            let bounds = |terms: &[(f32, f32)]| {
                terms
                    .iter()
                    .map(|(magnitude, rest)| reassociation(outputs) * magnitude + rest + 1e-6 * cap)
                    .collect::<Vec<_>>()
            };
            let capped = |values: &[f32]| {
                values
                    .iter()
                    .map(|z| {
                        if cap > 0.0 {
                            (f64::from(cap) * (f64::from(*z) / f64::from(cap)).tanh()) as f32
                        } else {
                            *z
                        }
                    })
                    .collect::<Vec<_>>()
            };
            for &mapping in check.mappings {
                let specialization = readout_head_specialization_on(
                    device,
                    &[("V", v), ("D", d), ("WS", scale_extent)],
                    mapping,
                );
                // The CPU native's one static is the scale port's extent.
                let specialization = if is_cpu(device) {
                    specialization.with_static("WS", scale_extent as u64)
                } else {
                    specialization
                };
                let head_native = readout_head_rows::native_for_device_with(
                    &device,
                    readout_head_rows::Elements {
                        NW: Element::bf16(),
                        OW: repr.resident(&device),
                        A: act.element(),
                    },
                    &specialization,
                )
                .unwrap()
                .call(readout_head_rows::Args {
                    hidden: &hidden_tensor,
                    norm: &bf16_norm(&device, &norm),
                    weight: &head.tensor(&device),
                    out_rows: &rows_tensor,
                    epsilon: 1e-6,
                    softcap: cap,
                    weight_scale: &f32_tensor(device, &[scale_extent], &scale_values),
                })
                .unwrap()
                .value;
                assert_within(
                    &format!(
                        "head_rows {repr:?} O {outputs} {mapping:?} softcap {cap} scale {:?}",
                        check.weight_scale
                    ),
                    &read_f32(&head_native),
                    &capped(&scaled_logits),
                    &bounds(&scaled_logit_bound),
                );
                let selected_native = readout_selected_rows::native_for_device_with(
                    &device,
                    readout_selected_rows::Elements {
                        NW: Element::bf16(),
                        OW: repr.resident(&device),
                        A: act.element(),
                    },
                    &readout_projection_specialization_on(device, &[("V", v), ("D", d)], mapping),
                )
                .unwrap()
                .call(readout_selected_rows::Args {
                    hidden: &hidden_tensor,
                    norm: &bf16_norm(&device, &norm),
                    weight: &head.tensor(&device),
                    out_rows: &rows_tensor,
                    selected: &i32_tensor(&device, &[selected.len()], &selected),
                    epsilon: 1e-6,
                    softcap: cap,
                })
                .unwrap()
                .value;
                assert_within(
                    &format!("selected_rows {repr:?} O {outputs} {mapping:?} softcap {cap}"),
                    &read_f32(&selected_native),
                    &capped(&chosen),
                    &bounds(&chosen_bound),
                );
            }
            if check.portable
                && (outputs == 2 || check.weight_scale.is_some() && outputs == 1)
                && repr == Repr::Q6k
            {
                let bindings = [
                    ("NW", registry::dense(DType::BF16)),
                    ("OW", repr.storage()),
                    ("A", registry::dense(DType::BF16)),
                ];
                let inputs = || {
                    vec![
                        floats(&[rows, d], &hidden),
                        activations(Act::Bf16, &[d], &norm),
                        Input::Data(head.oracle()),
                        ints(&[outputs], &out_rows),
                    ]
                };
                let mut head_inputs = inputs();
                head_inputs.push(Input::F32(1e-6));
                head_inputs.push(Input::F32(cap));
                head_inputs.push(floats(&[scale_extent], &scale_values));
                let oracle = result(
                    &interpret(&module, "readout_head_rows", &bindings, head_inputs),
                    0,
                );
                assert_within(
                    &format!(
                        "head_rows portable softcap {cap} scale {:?}",
                        check.weight_scale
                    ),
                    &oracle,
                    &capped(&scaled_logits),
                    &bounds(&scaled_logit_bound),
                );
                let mut selected_inputs = inputs();
                selected_inputs.push(ints(&[selected.len()], &selected));
                selected_inputs.push(Input::F32(1e-6));
                selected_inputs.push(Input::F32(cap));
                let oracle = result(
                    &interpret(&module, "readout_selected_rows", &bindings, selected_inputs),
                    0,
                );
                assert_within(
                    &format!("selected_rows portable softcap {cap}"),
                    &oracle,
                    &capped(&chosen),
                    &bounds(&chosen_bound),
                );
                let oracle = result(
                    &interpret(
                        &module,
                        "readout_features_rows",
                        &[
                            ("NW", registry::dense(DType::BF16)),
                            ("A", registry::dense(DType::BF16)),
                        ],
                        vec![
                            floats(&[rows, 1, d], &hidden),
                            activations(Act::Bf16, &[d], &norm),
                            ints(&[outputs], &out_rows),
                            Input::F32(1e-6),
                        ],
                    ),
                    0,
                );
                assert_within("features_rows portable", &oracle, &features, &feature_bound);
            }
        }
    }
}

fn uniform_values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| rng.symmetric() * scale).collect()
}

/// A read of a gathered table through a directory reads, for each selected
/// row, its slot (row 0 for a row without one and for a failed selection),
/// exactly as a direct read of that slot does, on every rows16 backend the
/// host has, over packed and dense tables; the portable body reads the same
/// rows. (CUDA's mma16 tables: `cuda_projection.rs`.)
#[test]
fn embedding_rows_read_the_slots_their_directory_names() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let module = module();
    let act = Act::Bf16;
    let (v, d) = (300usize, 512usize);
    // Logical rows past the table's slots; every third has a slot.
    let logical = 1000usize;
    let directory_values = (0..logical)
        .map(|row| {
            if row % 3 == 0 {
                ((7 * row + 3) % v) as i32
            } else {
                -1
            }
        })
        .collect::<Vec<_>>();
    for backend in [BackendName::Metal, BackendName::Vulkan, BackendName::Cpu] {
        let device = match catalog.open_backend(backend) {
            Ok(device) => device,
            Err(error) if backend != BackendName::Cpu => {
                eprintln!("gathered rows: {backend:?} unavailable: {error}");
                continue;
            }
            Err(error) => panic!("gathered rows: CPU unavailable: {error}"),
        };
        eprintln!("gathered rows: testing {backend:?}");
        let directory = i32_tensor(&device, &[logical, 1], &directory_values);
        let direct = i32_tensor(&device, &[0, 1], &[]);
        for repr in [Repr::Q4k, Repr::Q6k, Repr::Bf16] {
            let table = weight(repr, v, d, 91, 8.0);
            let resident = table.tensor(&device);
            let kernel = embedding_rows::native_for_device_with(
                &device,
                embedding_rows::Elements {
                    EW: repr.resident(&device),
                    A: act.element(),
                },
                &statics_on(&device, &[("D", d)]),
            )
            .unwrap();
            for rows in [1usize, 3, 8, 64, 513] {
                // The second pair is a failed selection; statuses vary.
                let selected = (0..rows)
                    .flat_map(|i| match i {
                        1 => [-1, 7],
                        _ => [((i * 97 + 6) % logical) as i32, (i % 2) as i32],
                    })
                    .collect::<Vec<_>>();
                let slots = selected
                    .chunks_exact(2)
                    .flat_map(|pair| {
                        let slot =
                            usize::try_from(pair[0]).map_or(0, |row| directory_values[row].max(0));
                        [slot, pair[1]]
                    })
                    .collect::<Vec<_>>();
                let read = |pairs: &[i32], directory: &Tensor| {
                    let out = kernel
                        .call(embedding_rows::Args {
                            table: &resident,
                            tokens: &i32_tensor(&device, &[rows, 2], pairs),
                            directory,
                            scale: 1.0,
                            normalize: 0,
                            epsilon: 0.0,
                        })
                        .unwrap();
                    (
                        out.r0.read_to_host().unwrap(),
                        out.r1.read_to_host().unwrap(),
                    )
                };
                assert_eq!(
                    read(&selected, &directory),
                    read(&slots, &direct),
                    "{backend:?} {repr:?} rows {rows}: a read through the directory"
                );
                if rows == 3 && backend == BackendName::Cpu {
                    let expected = slots
                        .chunks_exact(2)
                        .flat_map(|pair| {
                            table
                                .row(pair[0] as usize)
                                .iter()
                                .map(|value| act.round(*value))
                        })
                        .collect::<Vec<_>>();
                    let outcome = interpret(
                        &module,
                        "embedding_rows",
                        &[("EW", repr.storage()), ("A", registry::dense(DType::BF16))],
                        vec![
                            Input::Data(table.oracle()),
                            ints(&[rows, 2], &selected),
                            ints(&[logical, 1], &directory_values),
                            Input::F32(1.0),
                            Input::I32(0),
                            Input::F32(0.0),
                        ],
                    );
                    assert_within(
                        &format!("gathered {repr:?} portable"),
                        &result(&outcome, 1),
                        &expected,
                        &vec![0.0; expected.len()],
                    );
                }
            }
        }
    }
}

#[test]
fn embedding_rows_decode_exactly_as_the_portable_body() {
    for device in devices() {
        embedding_rows_decode_exactly_as_the_portable_body_on(&device);
    }
}

fn embedding_rows_decode_exactly_as_the_portable_body_on(device: &Device) {
    let module = module();
    let act = Act::Bf16;
    let (v, d) = (300usize, 2560usize);
    for repr in [
        Repr::Q6k,
        Repr::Q4k,
        Repr::Q5k,
        Repr::Q8,
        Repr::Iq4xs,
        Repr::Q2k,
        Repr::Q3k,
        Repr::Bf16,
    ] {
        let table = weight(repr, v, d, 90, 8.0);
        for rows in [1usize, 3, 8, 64] {
            let tokens = (0..rows)
                .map(|i| ((i * 97 + 5) % v) as i32)
                .collect::<Vec<_>>();
            // (token, status) rows, the `sample_rows` result layout.
            let pairs = tokens
                .iter()
                .flat_map(|token| [*token, 0])
                .collect::<Vec<_>>();
            let out = embedding_rows::native_for_device_with(
                &device,
                embedding_rows::Elements {
                    EW: repr.resident(&device),
                    A: act.element(),
                },
                &statics_on(device, &[("D", d)]),
            )
            .unwrap()
            .call(embedding_rows::Args {
                table: &table.tensor(&device),
                tokens: &i32_tensor(&device, &[rows, 2], &pairs),
                directory: &i32_tensor(&device, &[0, 1], &[]),
                scale: 1.0,
                normalize: 0,
                epsilon: 0.0,
            })
            .unwrap();
            let expected = tokens
                .iter()
                .flat_map(|t| table.row(*t as usize).iter().map(|value| act.round(*value)))
                .collect::<Vec<_>>();
            // The scaled (Gemma, √D) and normalized (Muse, weightless RMS)
            // forms: the square sum reassociates, so within an A rounding.
            for (scale, normalize) in [(39.191835f32, 0), (1.0, 1)] {
                let transformed = embedding_rows::native_for_device_with(
                    &device,
                    embedding_rows::Elements {
                        EW: repr.resident(&device),
                        A: act.element(),
                    },
                    &statics_on(device, &[("D", d)]),
                )
                .unwrap()
                .call(embedding_rows::Args {
                    table: &table.tensor(&device),
                    tokens: &i32_tensor(&device, &[rows, 2], &pairs),
                    directory: &i32_tensor(&device, &[0, 1], &[]),
                    scale,
                    normalize,
                    epsilon: 1e-5,
                })
                .unwrap();
                let (values, bound): (Vec<f32>, Vec<f32>) = tokens
                    .iter()
                    .flat_map(|t| {
                        let row = table
                            .row(*t as usize)
                            .iter()
                            .map(|v| v * scale)
                            .collect::<Vec<_>>();
                        let inverse = if normalize != 0 {
                            let squares = row
                                .iter()
                                .map(|v| f64::from(*v) * f64::from(*v))
                                .sum::<f64>();
                            (1.0 / (squares / d as f64 + 1e-5).sqrt()) as f32
                        } else {
                            1.0
                        };
                        row.into_iter()
                            .map(move |v| (act.round(v * inverse), (v * inverse).abs() / 128.0))
                            .collect::<Vec<_>>()
                    })
                    .unzip();
                assert_within(
                    &format!("embedding {repr:?} rows {rows} scale {scale} normalize {normalize}"),
                    &read_act(act, &transformed.r0),
                    &values,
                    &bound,
                );
            }
            let exact = vec![0.0; expected.len()];
            assert_within(
                &format!("embedding {repr:?} rows {rows} (A)"),
                &read_act(act, &out.r0),
                &expected,
                &exact,
            );
            assert_within(
                &format!("embedding {repr:?} rows {rows} (F32)"),
                &read_f32(&out.r1),
                &expected,
                &exact,
            );
            if rows == 3 {
                let outcome = interpret(
                    &module,
                    "embedding_rows",
                    &[("EW", repr.storage()), ("A", registry::dense(DType::BF16))],
                    vec![
                        Input::Data(table.oracle()),
                        ints(&[rows, 2], &pairs),
                        ints(&[0, 1], &[]),
                        Input::F32(1.0),
                        Input::I32(0),
                        Input::F32(0.0),
                    ],
                );
                assert_within(
                    &format!("embedding {repr:?} portable"),
                    &result(&outcome, 1),
                    &expected,
                    &exact,
                );
            }
        }
    }
}

/// Matrix column gathering preserves the rounded projection, convolution,
/// and every window bank for partial fragments and taped, split slots.
#[test]
fn metal_convolved_matrix_columns_match_host() {
    use magnitude_kernels::gated_delta_project_convolved;
    use seismic::{SlabRegion, SlabTensor};
    let Some(device) = devices()
        .into_iter()
        .find(|d| d.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let (h, nk, nv, w, c, banks, tape) =
        (512usize, 1usize, 2usize, 33usize, 4usize, 5usize, 3usize);
    let channels = (2 * nk + nv) * w;
    let segment_rows = [channels, nv * w, nv, nv];
    let total: usize = segment_rows.iter().sum();
    let norm = norm_values(h, 73);
    let window_rows = c - 1 + tape;
    let initial = uniform_values(banks * window_rows * channels, 74, 0.5)
        .into_iter()
        .map(|v| act.round(v))
        .collect::<Vec<_>>();
    let convolution = uniform_values(channels * c, 75, 0.25);
    for (q, z) in [
        (Repr::Q4k, Repr::Q6k),
        (Repr::Q5k, Repr::Q4k),
        (Repr::Q6k, Repr::Q5k),
        (Repr::Q8, Repr::Bf16),
    ] {
        let reprs = [q, z, Repr::Q8, Repr::Q8];
        let weights = (0..4)
            .map(|i| weight(reprs[i], segment_rows[i], h, 76 + i as u64, 1.0))
            .collect::<Vec<_>>();
        let tensors = weights
            .iter()
            .map(|v| v.tensor(&device))
            .collect::<Vec<_>>();
        for rows in [8usize, 13] {
            let hidden = uniform_values(rows * h, rows as u64, 1.0);
            let (expected, bound) = under(rows, || {
                let mut expected = Vec::new();
                let mut bound = Vec::new();
                for row in hidden.chunks_exact(h) {
                    let (x, slack) = rms_row(act, row, &norm, 1e-6);
                    for wt in &weights {
                        let x = wt.operand_rows(&x);
                        for n in 0..wt.rows {
                            let (acc, magnitude) = dot(&x, wt.row(n));
                            expected.push(act.round(acc));
                            bound.push(
                                rounded_bound(act, acc, magnitude) + slack_dot(&slack, wt.row(n)),
                            );
                        }
                    }
                }
                (expected, bound)
            });
            // (lo, count, stop, source bank, source tape, target bank).
            // The final row is padding and must publish a zero convolution.
            let slots = [
                (0usize, 2usize, 1usize, 1usize, 2usize, 3usize),
                (2, rows - 3, rows - 5, 2, 1, 4),
            ];
            let segments = i32_tensor(
                &device,
                &[3, 2],
                &[0, 2, 2, (rows - 1) as i32, rows as i32, rows as i32],
            );
            let stop = i32_tensor(&device, &[2], &[1, (rows - 5) as i32]);
            let previous = i32_tensor(&device, &[2], &[1, 2]);
            let previous_tape = i32_tensor(&device, &[2], &[2, 1]);
            let following = i32_tensor(&device, &[2], &[3, 4]);
            for parts in [1u64, 2, 4] {
                let mut slabs = SlabTensor::new(
                    &device,
                    2,
                    banks as u64,
                    vec![SlabRegion {
                        element: act.element(),
                        row_shape: vec![window_rows as u64, channels as u64],
                    }],
                )
                .unwrap();
                for slab in 0..banks.div_ceil(2) {
                    slabs.add_slab().unwrap();
                    let start = slab * 2;
                    let count = 2.min(banks - start);
                    slabs
                        .region_rows(0, start as u64, count as u64)
                        .unwrap()
                        .write_from_host(&act.bytes(
                            &initial[start * window_rows * channels
                                ..(start + count) * window_rows * channels],
                        ))
                        .unwrap();
                }
                let mut window = slabs.logical_region(0).unwrap();
                let mut spec = formed(
                    &device,
                    statics_on(
                        &device,
                        &[("H", h), ("NK", nk), ("NV", nv), ("W", w), ("C", c)],
                    ),
                    h,
                )
                .with_param("BATCH_FROM", 3)
                .with_launch_param(2, "BATCH_SIMDGROUPS", 8)
                .with_launch_param(2, "BATCH_ROWS", 2)
                .with_launch_param(2, "BATCH_PARTS", parts);
                for launch in [1, 3, 4] {
                    spec = spec
                        .with_launch_param(launch, "SIMDGROUPS", 8)
                        .with_launch_param(launch, "ROWS", 2)
                        .with_launch_param(launch, "LANES", 16);
                }
                let output = gated_delta_project_convolved::native_for_device_with(
                    &device,
                    gated_delta_project_convolved::Elements {
                        NW: act.element(),
                        QW: reprs[0].resident(&device),
                        GW: reprs[1].resident(&device),
                        AW: reprs[2].resident(&device),
                        BW: reprs[3].resident(&device),
                        A: act.element(),
                    },
                    &spec,
                )
                .unwrap()
                .call(gated_delta_project_convolved::Args {
                    hidden: &f32_tensor(&device, &[rows, h], &hidden),
                    input_norm: &bf16_norm(&device, &norm),
                    qkv_weight: &tensors[0],
                    gate_weight: &tensors[1],
                    alpha_weight: &tensors[2],
                    beta_weight: &tensors[3],
                    convolution: &f32_tensor(&device, &[channels, c], &convolution),
                    segments: &segments,
                    stop: &stop,
                    previous_bank: &previous,
                    previous_tape: &previous_tape,
                    following_bank: &following,
                    window: &mut window,
                    epsilon: 1e-6,
                    slab_banks: 2,
                })
                .unwrap();
                let projection = read_act(act, &output.r0);
                let label = format!("convolved {q:?}/{z:?} M{rows} parts{parts}");
                assert_within(&label, &projection, &expected, &bound);
                let mut expected_window = initial.clone();
                let mut expected_convolved = vec![0.0f32; rows * channels];
                for (lo, count, stop, source, taped, target) in slots {
                    for local in 0..count {
                        for n in 0..channels {
                            let mut sum = 0.0f32;
                            for tap in 0..c {
                                let position = local as isize + tap as isize - (c - 1) as isize;
                                let input = if position < 0 {
                                    initial[(source * window_rows + (taped + c - 1)
                                        - position.unsigned_abs())
                                        * channels
                                        + n]
                                } else {
                                    projection[(lo + position as usize) * total + n]
                                };
                                sum = convolution[n * c + tap].mul_add(input, sum);
                            }
                            expected_convolved[(lo + local) * channels + n] =
                                sum / (1.0 + (-sum).exp());
                        }
                    }
                    for kept in 0..c - 1 + count - stop {
                        let position = stop as isize + kept as isize - (c - 1) as isize;
                        for n in 0..channels {
                            expected_window[(target * window_rows + kept) * channels + n] =
                                if position < 0 {
                                    initial[(source * window_rows + taped + c
                                        - 1
                                        - position.unsigned_abs())
                                        * channels
                                        + n]
                                } else {
                                    projection[(lo + position as usize) * total + n]
                                };
                        }
                    }
                }
                assert_eq!(
                    read_act(act, &window),
                    expected_window,
                    "{label}: all window banks"
                );
                // Same F32 tap chain; allow the host/Metal exponential's final rounding.
                let convolution_bound = expected_convolved
                    .iter()
                    .map(|v| 2e-6 * v.abs() + 1e-7)
                    .collect::<Vec<_>>();
                assert_within(
                    &format!("{label}: convolution"),
                    &read_f32(&output.r1),
                    &expected_convolved,
                    &convolution_bound,
                );
            }
        }
    }
}

#[test]
fn metal_absent_query_projection_preserves_key_value() {
    let Some(device) = devices()
        .into_iter()
        .find(|d| d.backend() == BackendName::Metal)
    else {
        return;
    };
    let (d, q, kv) = (256usize, 256usize, 64usize);
    for rows in [1usize, 2, 7, 128] {
        let mut rng = Rng::new(776);
        let values = (0..rows * d).map(|_| rng.symmetric()).collect::<Vec<_>>();
        let hidden = f32_tensor(&device, &[rows, d], &values);
        let norm = bf16_norm(&device, &norm_values(d, 13));
        let query = weight(Repr::Bf16, q, d, 777, 1.0).tensor(&device);
        let key = weight(Repr::Bf16, kv, d, 778, 1.0).tensor(&device);
        let value = weight(Repr::Bf16, kv, d, 779, 1.0).tensor(&device);
        let empty = query.slice_leading(0, 0).unwrap();
        let run = |absent: bool| {
            let spec = attention_project_specialization_on(
                &device,
                &[
                    ("D", d),
                    ("Q", if absent { 0 } else { q }),
                    ("GR", 0),
                    ("K", kv),
                    ("V", kv),
                ],
                gemm_mapping(64, 64, 1),
            );
            let kernel = attention_project::native_for_device_with(
                &device,
                attention_project::Elements {
                    NW: Element::bf16(),
                    QW: Element::bf16(),
                    GW: Element::bf16(),
                    KW: Element::bf16(),
                    VW: Element::bf16(),
                    A: Element::bf16(),
                },
                &spec,
            )
            .unwrap();
            let out = kernel
                .call(attention_project::Args {
                    hidden: &hidden,
                    input_norm: &norm,
                    query_weight: if absent { &empty } else { &query },
                    gate_weight: &empty,
                    key_weight: &key,
                    value_weight: &value,
                    epsilon: 1e-6,
                    project_mode: if absent { 0 } else { 1 },
                    query_scale: &no_scale(&device),
                    gate_scale: &no_scale(&device),
                    key_scale: &no_scale(&device),
                    value_scale: &no_scale(&device),
                    query_bound: &absent_bound(&device),
                    gate_bound: &absent_bound(&device),
                    key_bound: &absent_bound(&device),
                    value_bound: &absent_bound(&device),
                })
                .unwrap();
            (
                out.r2.read_to_host().unwrap(),
                out.r3.read_to_host().unwrap(),
            )
        };
        let full = run(false);
        let absent = run(true);
        assert_eq!(full, absent, "K/V changed at {rows} rows");
    }
}

#[test]
fn metal_attention_output_matrix_matches_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|d| d.backend() == BackendName::Metal)
    else {
        return;
    };
    let (d, q, w) = (131usize, 4usize, 128usize);
    let k = q * w;
    for act in [Act::Bf16, Act::F16] {
        for repr in [Repr::Q4k, Repr::Q5k, Repr::Q6k, Repr::Q8] {
            let weights = weight(repr, d, k, 714, 1.0);
            for rows in [8usize, 13] {
                let hidden = uniform_values(rows * d, 715, 1.0);
                let gated = uniform_values(rows * k, 716, 1.0)
                    .into_iter()
                    .map(|v| act.round(v))
                    .collect::<Vec<_>>();
                let (expected, bound) = under(rows, || {
                    let mut expected = Vec::new();
                    let mut bound = Vec::new();
                    for r in 0..rows {
                        for n in 0..d {
                            let (acc, magnitude) = dot(&weights.operand_rows(&gated[r * k..(r + 1) * k]), weights.row(n));
                            expected.push(hidden[r * d + n] + act.round(acc));
                            bound.push(rounded_bound(act, acc, magnitude));
                        }
                    }
                    (expected, bound)
                });
                for parts in [1, 2, 4] {
                    let mapping = Mapping {
                        batch_from: 3,
                        batch_parts: parts,
                        ..gemm_mapping(64, 64, 1)
                    };
                    let kernel = attention_output::native_for_device_with(
                        &device,
                        attention_output::Elements {
                            A: act.element(),
                            OW: repr.resident(&device),
                        },
                        &attention_output_specialization_on(
                            &device,
                            &[("D", d), ("Q", q), ("W", w)],
                            mapping,
                        ),
                    )
                    .unwrap();
                    let output = kernel
                        .call(attention_output::Args {
                            hidden: &f32_tensor(&device, &[rows, d], &hidden),
                            inject: &no_inject(&device, rows as u64),
                            gated: &act_tensor(&device, act, &[rows, q, w], &gated),
                            output_weight: &weights.tensor(&device),
                            output_scale: &no_scale(&device),
                            output_bound: &absent_bound(&device),
                        })
                        .unwrap()
                        .value;
                    assert_within(
                        &format!("attention output {act:?} {repr:?} M{rows} parts{parts}"),
                        &read_f32(&output),
                        &expected,
                        &bound,
                    );
                }
            }
        }
    }
}

/// Q4 verification: exact integer code operands, partial weight fragments,
/// gathered RMS inputs, both activation fragments, and split-K reduction.
#[test]
fn metal_q4_verify_rows_match_host() {
    let Some(device) = devices()
        .into_iter()
        .find(|d| d.backend() == BackendName::Metal)
    else {
        return;
    };
    let act = Act::Bf16;
    let (h, f) = (2560, 131);
    let gate = weight(Repr::Q4g32s, f, h, 61, 1.0);
    let up = weight(Repr::Q4g32s, f, h, 62, 1.0);
    let norm = norm_values(h, 63);
    for rows in [2usize, 4, 8, 16] {
        let source = uniform_values(rows * h, rows as u64, 1.0)
            .into_iter()
            .map(|x| act.round(x))
            .collect::<Vec<_>>();
        let out_rows = (0..rows as i32).rev().collect::<Vec<_>>();
        let (expected, bounds) = under(rows, || {
            dense_expand_reference(act, &gate, &up, &source, &norm, &out_rows, 1e-6)
        });
        for parts in [1, 4] {
            let mapping = Mapping {
                batch_from: 2,
                batch_parts: parts,
                ..gemm_mapping(64, 64, 1)
            };
            let actual = dense_expand_native(
                &device, act, &gate, &up, &source, rows, &norm, &out_rows, 1e-6, mapping,
            );
            assert_within("Q4 paired verify", &actual, &expected, &bounds);
            let actual = project_rows_native(&device, act, &gate, &source, mapping, None);
            let (expected, bounds) = under(rows, || {
                let mut expected = Vec::new();
                let mut bounds = Vec::new();
                for m in 0..rows {
                    for n in 0..f {
                        let (sum, magnitude) = dot(&gate.operand_rows(&source[m * h..(m + 1) * h]), gate.row(n));
                        expected.push(sum);
                        bounds.push(reassociation_bound(magnitude) + 1e-7);
                    }
                }
                (expected, bounds)
            });
            assert_within("Q4 plain verify", &actual, &expected, &bounds);
            // Changing a peer row cannot change this row's result.
            let mut changed = source.clone();
            changed[h..2 * h].fill(0.0);
            let peer = project_rows_native(&device, act, &gate, &changed, mapping, None);
            assert_eq!(&actual[..f], &peer[..f]);
        }
    }
}


/// Identity weights isolate the RMS prologue: each dot product has one
/// nonzero term, so accumulation reassociation cannot explain a difference.
#[test]
fn metal_normalize_prepass_matches_small_row_tree() {
    let Some(device) = devices()
        .into_iter()
        .find(|d| d.backend() == BackendName::Metal)
    else {
        return;
    };
    let h = 3840usize;
    let mut identity = Weight {
        repr: Repr::Bf16,
        rows: h,
        k: h,
        bytes: vec![0; h * h * 2],
        values: vec![0.0; h * h],
    };
    for i in 0..h {
        identity.values[i * h + i] = 1.0;
        identity.bytes[(i * h + i) * 2..(i * h + i) * 2 + 2]
            .copy_from_slice(&0x3f80u16.to_le_bytes());
    }
    let norm = vec![1.0; h];
    let rows = [0, 1, 2, 3];
    let mut differences = 0;
    for seed in 1..=16 {
        let mut source = vec![0.001f32; 4 * h];
        for m in 0..4 {
            let target = [1.00390625f64, 2.0078125, 3.0078125, 3.8671875][m];
            let rest = 240.0f64 + (h - 256) as f64 * 0.001f64.powi(2);
            let probe = (target * target * (rest + h as f64 * 1e-6)
                / (h as f64 - 16.0 * target * target))
                .sqrt() as f32;
            for k in 0..256 {
                source[m * h + k] = 1.0;
            }
            for k in 0..16 {
                source[m * h + k] =
                    f32::from_bits(probe.to_bits().wrapping_add(seed as u32).wrapping_sub(8));
            }
        }
        let hidden = f32_tensor(&device, &[4, h], &source);
        let nt = bf16_norm(&device, &norm);
        let selected = i32_tensor(&device, &[4], &rows);
        let features = readout_features_rows::native_for_device_with(
            &device,
            readout_features_rows::Elements {
                NW: Element::bf16(),
                A: Element::bf16(),
            },
            &formed(
                &device,
                NativeSpecialization::new().with_static("D", h as u64),
                h,
            ),
        )
        .unwrap()
        .call(readout_features_rows::Args {
            hidden: &one_part(&hidden),
            norm: &nt,
            out_rows: &selected,
            epsilon: 1e-6,
        })
        .unwrap()
        .value;
        let reference = readout_head_rows::native_for_device_with(
            &device,
            readout_head_rows::Elements {
                NW: Element::bf16(),
                OW: Element::bf16(),
                A: Element::bf16(),
            },
            &readout_head_specialization_on(
                &device,
                &[("D", h), ("V", h), ("WS", 0)],
                Mapping {
                    batch_from: 5,
                    ..gemm_mapping(64, 64, 1)
                },
            ),
        )
        .unwrap()
        .call(readout_head_rows::Args {
            hidden: &hidden,
            norm: &nt,
            weight: &identity.tensor(&device),
            out_rows: &selected,
            epsilon: 1e-6,
            softcap: 0.0,
            weight_scale: &f32_tensor(&device, &[0], &[]),
        })
        .unwrap()
        .value;
        assert_eq!(
            read_act(Act::Bf16, &features),
            read_f32(&reference),
            "feature publication and scalar identity projection must normalize identically"
        );
        let scalar = dense_expand_native(
            &device,
            Act::Bf16,
            &identity,
            &identity,
            &source,
            4,
            &norm,
            &rows,
            1e-6,
            Mapping {
                batch_from: 5,
                ..gemm_mapping(64, 64, 1)
            },
        );
        let staged = dense_expand_native(
            &device,
            Act::Bf16,
            &identity,
            &identity,
            &source,
            4,
            &norm,
            &rows,
            1e-6,
            Mapping {
                batch_from: 3,
                ..gemm_mapping(64, 64, 1)
            },
        );
        differences += scalar
            .iter()
            .zip(&staged)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    assert_eq!(
        differences, 0,
        "identity projection must preserve each rounded RMS activation across the staging boundary"
    );
}

/// The inject port of a single residual stream: no columns.
fn no_inject(device: &seismic::Device, rows: u64) -> seismic::Tensor {
    seismic::Tensor::zeros(device, seismic::Element::f32(), &[rows, 0]).unwrap()
}

/// `dense_output`'s time per call on every device this host has, over the
/// projection shapes of Gemma 4 E2B Q2_K_P (H x F: its gate/up 12288 x 1536
/// and 6144 x 1536, query 2048 x 1536, key 256 x 1536, per-layer projection
/// 1536 x 256), for q2k and q3k, the existing forms nearest them in bits
/// (iq2xxs, iq2xs), q4k and q4g32s, and q6k (the form Q3_K was held in
/// before q3k), at one row (the GEMV) and at 512 (the GEMM):
/// per form the best over `MAPPINGS` of the best of eleven batches of 20
/// calls.
#[test]
#[ignore = "timing"]
fn dense_output_timing_at_model_shapes() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let devices = [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan, BackendName::Cpu]
        .into_iter()
        .filter_map(|backend| catalog.open_backend(backend).ok())
        .collect::<Vec<_>>();
    let act = Act::Bf16;
    for device in &devices {
        for (h, f) in [(12288usize, 1536usize), (6144, 1536), (2048, 1536), (256, 1536), (1536, 256)] {
            for rows in [1usize, 512] {
                if is_cpu(device) && rows > 1 {
                    continue;
                }
                let mut rng = Rng::new(7);
                let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
                let product = (0..rows * f).map(|_| act.round(rng.symmetric())).collect::<Vec<_>>();
                let out_rows = (0..rows as i32).collect::<Vec<_>>();
                let residual = f32_tensor(device, &[rows, h], &residual);
                let product = act_tensor(device, act, &[rows, f], &product);
                let out_rows = i32_tensor(device, &[rows], &out_rows);
                let down_scale = f32_tensor(device, &[0], &[]);
                let down_bound = absent_bound(device);
                let mut line = format!("{:>7} {h:>5} x {f:>4} M {rows:>3}:", device.backend().as_str());
                for repr in [Repr::Q2k, Repr::Q3k, Repr::Iq2xxs, Repr::Iq2xs, Repr::Q4k, Repr::Q4g32s, Repr::Q6k] {
                    // Bytes alone: a time does not read the values.
                    let weight = match repr.gguf_source() {
                        Some(_) => resident_tensor(
                            device,
                            repr,
                            [h, f],
                            &gguf_bytes(repr, h, f, &mut Rng::new(11), 2.0 / (f as f32).sqrt()),
                        ),
                        None => weight(repr, h, f, 11, 1.0).tensor(device),
                    };
                    let best = MAPPINGS
                        .iter()
                        .map(|mapping| {
                            let kernel = dense_output::native_for_device_with(
                                device,
                                dense_output::Elements {
                                    A: act.element(),
                                    DW: repr.resident(device),
                                },
                                &dense_output_specialization_on(device, &[("H", h), ("F", f)], *mapping),
                            )
                            .unwrap();
                            let args = || dense_output::Args {
                                residual: &residual,
                                product: &product,
                                down_weight: &weight,
                                out_rows: &out_rows,
                                down_scale: &down_scale,
                                down_bound: &down_bound,
                            };
                            let mut value = kernel.call(args()).unwrap().value;
                            const CALLS: u32 = 20;
                            (0..12)
                                .map(|_| {
                                    let mut batch = seismic::NativeTensorBatch::new(device);
                                    for _ in 0..CALLS {
                                        batch
                                            .push(&kernel, args(), dense_output::OutputArgs { value: &mut value })
                                            .unwrap();
                                    }
                                    let started = std::time::Instant::now();
                                    batch.submit().unwrap().wait().unwrap();
                                    started.elapsed() / CALLS
                                })
                                .skip(1)
                                .min()
                                .unwrap()
                        })
                        .min()
                        .unwrap();
                    line += &format!(" {repr:?} {:>8.2} us", best.as_secs_f64() * 1e6);
                }
                println!("{line}");
            }
        }
    }
}
