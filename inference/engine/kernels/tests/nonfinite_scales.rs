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

use magnitude_kernels::embedding_rows;
use seismic::{BackendName, Device, Element, Layout, NativeSpecialization, Tensor};
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

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
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
        }
    }
}

fn gather(device: &Device, case: &Case, layout: Layout) {
    let format = case.format;
    let mut rng = Rng(0x5eed);
    let blocks = K / format.block_values();
    // One row per scale pattern; each row's blocks cycle through the
    // patterns and a finite control (the format's own random scale).
    let mut external = Vec::new();
    let rows = case.scales.len();
    for row in 0..rows {
        for block in 0..blocks {
            let mut bytes = format.block(&mut rng, 0.05);
            let pattern = (row + block) % (case.scales.len() + 1);
            if let Some(bits) = case.scales.get(pattern) {
                (case.set)(&mut bytes, *bits);
            }
            external.extend(bytes);
        }
    }
    let shape = [rows as u64, K as u64];
    let Some(element) = Element::gguf_packet_source(format.encoding, format.scales)
        .and_then(|source| source.converted(layout, format.operands))
    else {
        return;
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
    let actual = read_f32(&out.r1);
    let failures = (0..actual.len())
        .filter(|i| actual[*i].to_bits() != expected(reference[*i]))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "{:?} gather of {format:?} in {layout:?}: {} of {} differ; first (index, actual, expected): {:?}",
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
