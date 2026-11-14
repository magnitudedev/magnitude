//! Every non-finite block-scale product decodes to the registry's canonical
//! NaN on every device, bit for bit. A row gathered (`embedding_rows`, each
//! value decoded alone, then rounded to BF16) from blocks whose stored scale
//! is not finite holds, at every value, the registry's reference decode
//! (`Element::decode_host`: the scale times the code, one F32 product, a NaN
//! the one canonical NaN, 0x7fc00000, whatever the scale's sign and payload,
//! and an infinity the infinity): a NaN scale (any payload, either sign,
//! signaling too), an infinite scale (its product with a zero code is NaN, with
//! another code an infinity of the product's sign).
//!
//! The scales are every stored float encoding a GGUF format carries: an F16
//! scale (`Q8_0`), an F32 grid scale (`fp8g128`), an E8M0 byte (`mxfp8g32`,
//! `mxfp4`) and UE4M3 fields (`nvfp4`).

use magnitude_kernels::{embedding_rows, stored_rows};
use seismic::{BackendName, Device, Element, GgufScales, Grid, Layout, NativeSpecialization, Tensor};
use seismic_lang::registry::bf16_round;

mod gguf_formats;

use gguf_formats::{Format, Rng};
use magnitude_ggml::Encoding;

/// Values per gathered row: whole blocks of every format here.
const K: usize = 256;

/// A format's block with its stored scale set to a raw `bits` pattern.
struct Case {
    format: Format,
    /// Raw non-finite scale patterns (in the scale's width).
    scales: &'static [u32],
    /// Writes `bits` as the scale of `block`.
    set: fn(&mut [u8], u32),
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            format: Format::Q8,
            // ±inf, quiet and signaling NaN of either sign, all-ones.
            scales: &[0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7c01, 0xfdff, 0xffff],
            set: |block, bits| block[..2].copy_from_slice(&(bits as u16).to_le_bytes()),
        },
        Case {
            format: Format::F8_E4M3_F32,
            scales: &[0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0000, 0x7f80_0001, 0xff80_0001, 0x7fff_ffff, 0xffff_ffff],
            set: |block, bits| block[..4].copy_from_slice(&bits.to_le_bytes()),
        },
        Case {
            format: Format::F8_E4M3,
            // E8M0's NaN.
            scales: &[0xff],
            set: |block, bits| block[0] = bits as u8,
        },
        Case {
            format: Format::rounded(Encoding::Mxfp4),
            scales: &[0xff],
            set: |block, bits| block[0] = bits as u8,
        },
        Case {
            format: Format::NVFP4,
            // UE4M3's NaN, with and without the (ignored) sign bit.
            scales: &[0x7f, 0xff],
            set: |block, bits| block[..4].fill(bits as u8),
        },
    ]
}

fn i32_tensor(device: &Device, shape: &[u64], values: &[i32]) -> Tensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), shape, &bytes).unwrap()
}

/// A result's values as F32, whatever its element (the gather's wide result is F32, a row of
/// `stored_rows` the activation's BF16).
fn read_f32(tensor: &Tensor) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    if tensor.element() == Element::bf16() {
        bytes
            .chunks_exact(2)
            .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(word.try_into().unwrap())) << 16))
            .collect()
    } else {
        bytes.chunks_exact(4).map(|word| f32::from_le_bytes(word.try_into().unwrap())).collect()
    }
}

/// The reference's value as the BF16 gather returns it: the registry's one
/// NaN for any NaN.
fn expected(value: f64) -> u32 {
    let value = value as f32;
    if value.is_nan() { 0x7fc0_0000 } else { bf16_round(value).to_bits() }
}

#[test]
fn non_finite_block_scales_decode_to_the_canonical_nan_on_every_device() {
    for device in seismic_native_tests::every_device() {
        let mut layouts = vec![device.backend().resident_layout()];
        if device.backend() == BackendName::Cuda {
            layouts.push(Layout::Rows16);
        }
        for case in cases() {
            for layout in &layouts {
                gather(&device, &case, *layout);
            }
            read_stored(&device, &case);
        }
    }
}

/// A case's rows in a layout: the format's blocks of the case's scales, as resident
/// storage, and the registry's decode of them.
struct Fixture {
    element: Element,
    rows: usize,
    shape: [u64; 2],
    resident: Vec<u8>,
    reference: Vec<f64>,
    /// Row `r`'s selection of row `r`.
    tokens: Vec<i32>,
}

fn fixture(case: &Case, layout: Layout) -> Option<Fixture> {
    let format = case.format;
    let mut rng = Rng(0x5eed);
    let blocks = K / format.block_values();
    // One row per scale pattern; each row's blocks cycle through the
    // patterns and a finite control (the format's own random scale). Under a
    // grid the patterns go to its tiles, each a scale of tile rows and
    // columns, so there are rows enough for a tile to each.
    let patterns = case.scales.len() + 1;
    let (rows, external) = match format.scales {
        GgufScales::Grid(Grid { tile, .. }) => {
            let (tile_rows, column_tiles) = (tile[0] as usize, K / tile[1] as usize);
            let rows = tile_rows * patterns.div_ceil(column_tiles);
            let shape = [rows as u64, K as u64];
            let mut bytes = format.external(&mut rng, &shape, 0.05);
            format.set_grid_scales(
                &mut bytes,
                &shape,
                |row, packet| case.scales.get(((row / tile_rows) * column_tiles + packet) % patterns).copied(),
                case.set,
            );
            (rows, bytes)
        }
        _ => {
            let rows = case.scales.len();
            let mut external = Vec::new();
            for row in 0..rows {
                for block in 0..blocks {
                    let mut bytes = format.block(&mut rng, 0.05);
                    if let Some(bits) = case.scales.get((row + block) % patterns) {
                        (case.set)(&mut bytes, *bits);
                    }
                    external.extend(bytes);
                }
            }
            (rows, external)
        }
    };
    let shape = [rows as u64, K as u64];
    let Some(element) = Element::gguf_packet_source(format.encoding, format.scales)
        .and_then(|source| source.converted(layout, format.operands))
    else {
        return None;
    };
    let resident = element
        .repack_host(format.source(), &shape, &external)
        .expect("the registry's repack takes blocks of a non-finite scale");
    let reference = element.decode_host(&shape, &resident).unwrap();
    assert!(
        reference.iter().any(|value| value.is_nan()),
        "{format:?}: the blocks hold non-finite products"
    );
    let tokens = (0..rows as i32).flat_map(|row| [row, 0]).collect::<Vec<_>>();
    Some(Fixture { element, rows, shape, resident, reference, tokens })
}

/// The rows gathered by `embedding_rows` in `layout`.
fn gather(device: &Device, case: &Case, layout: Layout) {
    let Some(Fixture { element, rows, shape, resident, reference, tokens }) = fixture(case, layout) else {
        return;
    };
    let format = case.format;
    let elements = || embedding_rows::Elements { EW: element, A: Element::bf16() };
    let statics = embedding_rows::native_implementation(device)
        .unwrap()
        .unwrap()
        .statics
        .clone();
    let fixed = statics.iter().fold(NativeSpecialization::new(), |fixed, name| {
        assert_eq!(name, "D", "the gather's only static is its width");
        fixed.with_static(name.clone(), K as u64)
    });
    let specialization = embedding_rows::native_entry_with(elements())
        .default_specialization(device, &fixed)
        .unwrap()
        .unwrap();
    let out = embedding_rows::native_for_device_with(device, elements(), &specialization)
        .unwrap()
        .call(embedding_rows::Args {
            table: &Tensor::from_host(device, element, &shape, &resident).unwrap(),
            tokens: &i32_tensor(device, &[rows as u64, 2], &tokens),
            directory: &i32_tensor(device, &[0, 1], &[]),
            scale: 1.0,
            normalize: 0,
            epsilon: 0.0,
        })
        .unwrap();
    check(device, "embedding_rows", format, layout, &read_f32(&out.r1), &reference);
}

/// The rows read by `stored_rows` from resident slots (one record holding every row, in the
/// rows16 layout every backend's `stored_rows` reads) through a dense directory.
fn read_stored(device: &Device, case: &Case) {
    let Some(Fixture { element, rows, resident, reference, tokens, .. }) = fixture(case, Layout::Rows16) else {
        return;
    };
    let stored = stored_rows::native_entry_with(stored_rows::Elements { E: element, A: Element::bf16() });
    let Some(offered) = stored.offered(device).unwrap() else {
        return;
    };
    let specialization = offered
        .default_specialization(&NativeSpecialization::new().with_static("W", K as u64))
        .unwrap();
    let slots = (0..rows as i32).collect::<Vec<_>>();
    let result = stored
        .prepare(device, &specialization)
        .unwrap()
        .call(stored_rows::Args {
            slots: &Tensor::from_host(device, element, &[1, rows as u64, K as u64], &resident).unwrap(),
            tokens: &i32_tensor(device, &[rows as u64, 2], &tokens),
            directory: &i32_tensor(device, &[rows as u64, 1], &slots),
        })
        .unwrap()
        .value;
    check(device, "stored_rows", case.format, Layout::Rows16, &read_f32(&result), &reference);
}

/// Asserts `actual` is the reference's values as a BF16 gather returns them.
fn check(device: &Device, kernel: &str, format: Format, layout: Layout, actual: &[f32], reference: &[f64]) {
    let failures = (0..actual.len())
        .filter(|i| actual[*i].to_bits() != expected(reference[*i]))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "{:?} {kernel} of {format:?} in {layout:?}: {} of {} differ; first (index, actual, expected): {:?}",
        device.backend(),
        failures.len(),
        actual.len(),
        failures
            .iter()
            .take(4)
            .map(|i| (*i, format!("{:#010x}", actual[*i].to_bits()), format!("{:#010x}", expected(reference[*i]))))
            .collect::<Vec<_>>()
    );
}
