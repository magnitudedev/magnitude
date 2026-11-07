//! E4M3 decode on every device against the format's reference decode, bit
//! for bit
//! (`mxfp8g32`: E4M3 codes under one E8M0 scale per 32): every byte code
//! (NaN, ±0, subnormals, ±448) under every E8M0 exponent, as a gather
//! decodes each value alone (`embedding_rows`) and as the GEMV's products
//! take the codes (`project_rows`, under BF16 and F16 activations).
//!
//! The reference restores a weight as its E4M3 codes' values
//! (`e4m3::value`, the `float8_e4m3fn` format) times their UE8M0 scales,
//! each byte `b` read as 2^(b - 127) in F32: an F32 product, exact. The
//! host decode (`Element::decode_host`) agrees with it at every case the
//! tests take. They differ at scale byte 255 alone, which that reading
//! makes 2^128 (F32 inf) and the registry, as OCP MX's E8M0, NaN; no test
//! takes it.

use magnitude_ggml::Encoding;
use magnitude_kernels::{embedding_rows, project_rows};
use seismic::{BackendName, Device, Element, GgufScales, Layout, NativeSpecialization, Tensor};
use seismic_lang::registry::bf16_round;

mod e4m3;

/// A row's values: eight blocks of 32, so every byte code once.
const K: usize = 256;

/// The reference's UE8M0 scale: `ldexp(1.0, bits - 127)` in
/// F32, the subnormal 2^-127 at 0 and inf at 255.
fn ue8m0(bits: u8) -> f32 {
    let exponent = i32::from(bits) - 127;
    (2f64.powi(exponent)) as f32
}

/// The E8M0 exponents up to `last` (at most 254, where the reference and the
/// registry agree) a device decodes every code under exactly. From 10
/// every finite code's value, at least 2^-9 2^(e - 127), is an F32 normal;
/// below it some are subnormal, which devices that flush F32 subnormals
/// (NVIDIA's Vulkan without `DenormPreserve`) lose. CUDA keeps them (its
/// natives compile with `--ftz=false`, and its scales carry no code
/// exponent, `packets::CodeExponentOf`), so it takes every exponent.
fn exponents(device: &Device, last: u8) -> Vec<u8> {
    assert!(last <= 254, "scale byte 255 is inf to the reference and NaN to the registry");
    let first = if device.backend() == BackendName::Cuda { 0 } else { 10 };
    (first..=last).collect()
}

/// A table of one row per `(exponent, codes)`: each of its eight assembled
/// blocks (`gguf_f8_e4m3_e8m0`: the scale byte, then 32 codes) under the
/// row's exponent.
struct Table {
    rows: Vec<(u8, [u8; K])>,
    blocks: Vec<u8>,
}

impl Table {
    fn new(rows: Vec<(u8, [u8; K])>) -> Self {
        let mut blocks = Vec::with_capacity(rows.len() * K / 32 * 33);
        for (exponent, codes) in &rows {
            for block in codes.chunks_exact(32) {
                blocks.push(*exponent);
                blocks.extend_from_slice(block);
            }
        }
        Self { rows, blocks }
    }

    fn shape(&self) -> [u64; 2] {
        [self.rows.len() as u64, K as u64]
    }

    /// The reference values of the table, row-major.
    fn values(&self) -> Vec<f32> {
        self.rows
            .iter()
            .flat_map(|(exponent, codes)| codes.map(|code| e4m3::value(code) * ue8m0(*exponent)))
            .collect()
    }

    /// The table's bytes in `layout`, whose host decode is the reference
    /// values.
    fn resident(&self, layout: Layout) -> (Element, Vec<u8>) {
        let source = Element::gguf_packet_source(Encoding::F8E4M3, GgufScales::E8m0Grid).unwrap();
        let element = Element::stored("mxfp8g32", layout).unwrap();
        let bytes = element.repack_host(source, &self.shape(), &self.blocks).unwrap();
        let host = element
            .decode_host(&self.shape(), &bytes)
            .unwrap()
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_same(&format!("host decode in {layout:?}"), &host, &self.values());
        (element, bytes)
    }
}

/// Row `r`'s codes: `codes` rotated by 7 r, so a code meets every column
/// (and lane) across the rows.
fn rotated(codes: &[u8; K], r: usize) -> [u8; K] {
    std::array::from_fn(|k| codes[(k + 7 * r) % K])
}

/// Bit equality, NaN as NaN.
fn same(actual: f32, expected: f32) -> bool {
    actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan())
}

fn assert_same(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let failures = (0..expected.len())
        .filter(|i| !same(actual[*i], expected[*i]))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "{label}: {} of {} differ; first {:?}",
        failures.len(),
        expected.len(),
        failures
            .iter()
            .take(4)
            .map(|i| (*i, actual[*i], expected[*i]))
            .collect::<Vec<_>>()
    );
}

fn i32_tensor(device: &Device, shape: &[u64], values: &[i32]) -> Tensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), shape, &bytes).unwrap()
}

fn f32_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), shape, &bytes).unwrap()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// The default specialization of an entry's native on `device` at
/// `extents` (each of its statics').
fn defaults<E: seismic::Entry>(
    device: &Device,
    entry: &seismic::BoundEntry<E>,
    statics: &[String],
    extents: &[(&str, u64)],
) -> NativeSpecialization {
    let fixed = statics.iter().fold(NativeSpecialization::new(), |fixed, name| {
        let (_, value) = extents
            .iter()
            .find(|(extent, _)| extent == name)
            .unwrap_or_else(|| panic!("static {name} has no extent"));
        fixed.with_static(name.clone(), *value)
    });
    entry.default_specialization(device, &fixed).unwrap().unwrap()
}

/// Every code under every exponent, gathered: each row's F32 values are
/// the reference's rounded to the gather's BF16 (its F32 result is the A row
/// widened), bit for bit (inf where 448 2^(e - 127) passes F32's range, NaN
/// for a NaN code, and a value below BF16's subnormals rounded as BF16
/// rounds it), in every layout a device gathers from (CUDA's streamed slots
/// too, `rows16`).
#[test]
fn e4m3_gathers_match_the_reference_at_every_code_and_exponent() {
    let codes: [u8; K] = std::array::from_fn(|k| k as u8);
    for device in seismic_native_tests::every_device() {
        let table = Table::new(
            exponents(&device, 254)
                .into_iter()
                .enumerate()
                .map(|(r, exponent)| (exponent, rotated(&codes, r)))
                .collect(),
        );
        let expected = table.values().into_iter().map(bf16_round).collect::<Vec<_>>();
        let rows = table.rows.len();
        let pairs = (0..rows as i32).flat_map(|row| [row, 0]).collect::<Vec<_>>();
        let tokens = i32_tensor(&device, &[rows as u64, 2], &pairs);
        let directory = i32_tensor(&device, &[0, 1], &[]);
        let mut layouts = vec![device.backend().resident_layout()];
        if device.backend() == BackendName::Cuda {
            layouts.push(Layout::Rows16);
        }
        for layout in layouts {
            let (element, bytes) = table.resident(layout);
            let elements = || embedding_rows::Elements {
                EW: element,
                A: Element::bf16(),
            };
            let statics = embedding_rows::native_implementation(&device)
                .unwrap()
                .unwrap()
                .statics
                .clone();
            let specialization = defaults(
                &device,
                &embedding_rows::native_entry_with(elements()),
                &statics,
                &[("D", K as u64)],
            );
            let out = embedding_rows::native_for_device_with(&device, elements(), &specialization)
                .unwrap()
                .call(embedding_rows::Args {
                    table: &Tensor::from_host(&device, element, &table.shape(), &bytes).unwrap(),
                    tokens: &tokens,
                    directory: &directory,
                    scale: 1.0,
                    normalize: 0,
                    epsilon: 0.0,
                })
                .unwrap();
            assert_same(
                &format!("{:?} gather from {layout:?}", device.backend()),
                &read_f32(&out.r1),
                &expected,
            );
        }
    }
}

/// Every finite code under every exponent whose values stay finite, as the
/// GEMV's products take them: a one-hot row reads one column of the weight,
/// so each output is its code's reference value exactly (+0 for either
/// zero, the sum of the row's products). A NaN code anywhere in a row makes
/// its every output NaN.
#[test]
fn e4m3_gemv_products_match_the_reference_at_every_code_and_exponent() {
    let finite = (0..=u8::MAX)
        .filter(|code| code & 0x7f != 0x7f)
        .chain([0x00, 0x80])
        .collect::<Vec<_>>();
    let finite: [u8; K] = finite.try_into().unwrap();
    for device in seismic_native_tests::every_device() {
        // 448 2^(e - 127) is finite up to e = 245.
        let mut rows = exponents(&device, 245)
            .into_iter()
            .enumerate()
            .map(|(r, exponent)| (exponent, rotated(&finite, r)))
            .collect::<Vec<_>>();
        let nan_rows = rows.len();
        // One NaN code of each sign, among zeros.
        let mut positive = [0x00; K];
        positive[5] = 0x7f;
        let mut negative = [0x80; K];
        negative[200] = 0xff;
        rows.extend([(127, positive), (127, negative)]);
        let table = Table::new(rows);
        let n = table.rows.len();
        let values = table.values();
        let (element, bytes) = table.resident(device.backend().resident_layout());
        let weight = Tensor::from_host(&device, element, &table.shape(), &bytes).unwrap();
        for act in [Element::bf16(), Element::f16()] {
            let elements = || project_rows::Elements {
                A: act,
                W: element,
                Y: Element::f32(),
            };
            let statics = project_rows::native_implementation(&device)
                .unwrap()
                .unwrap()
                .statics
                .clone();
            let specialization = defaults(
                &device,
                &project_rows::native_entry_with(elements()),
                &statics,
                &[("K", K as u64), ("N", n as u64), ("WS", 0), ("WB", 0)],
            );
            let kernel =
                project_rows::native_for_device_with(&device, elements(), &specialization).unwrap();
            // 1.0 in BF16 and F16.
            let one = if act == Element::bf16() { 0x3f80u16 } else { 0x3c00u16 };
            let mut actual = Vec::with_capacity(K * n);
            for column in 0..K {
                let source = (0..K)
                    .flat_map(|k| (if k == column { one } else { 0 }).to_le_bytes())
                    .collect::<Vec<_>>();
                let out = kernel
                    .call(project_rows::Args {
                        source: &Tensor::from_host(&device, act, &[1, K as u64], &source).unwrap(),
                        weight: &weight,
                        weight_scale: &f32_tensor(&device, &[0], &[]),
                        weight_bound: &f32_tensor(&device, &[0], &[]),
                    })
                    .unwrap()
                    .value;
                actual.extend(read_f32(&out));
            }
            let expected = (0..K)
                .flat_map(|column| {
                    let values = &values;
                    (0..n).map(move |row| {
                        if row >= nan_rows {
                            f32::NAN
                        } else {
                            values[row * K + column] + 0.0
                        }
                    })
                })
                .collect::<Vec<_>>();
            assert_same(
                &format!("{:?} GEMV under {act:?}", device.backend()),
                &actual,
                &expected,
            );
        }
    }
}
