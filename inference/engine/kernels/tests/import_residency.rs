use magnitude_kernels::{import_dense, repack_weight};

fn dense_bytes(name: &str, values: &[f32]) -> Vec<u8> {
    match name {
        "f32" => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        "f16" => [0xc280u16, 0x8000, 0x3000, 0x3e00, 0x4cc0]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        "bf16" => values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

/// GGUF sources and their resident representations.
const FORMATS: [(&str, &str); 8] = [
    ("gguf_q8_0", "q8g32s"),
    ("gguf_q3_k", "q6k"),
    ("gguf_q4_k", "q4k"),
    ("gguf_q5_k", "q5k"),
    ("gguf_q6_k", "q6k"),
    ("gguf_iq3_s", "q6k"),
    ("gguf_iq4_nl", "iq4g32"),
    ("gguf_iq4_xs", "iq4g32"),
];

/// ggml's `dequantize_row_q3_K` for one 110-byte block, in its operation
/// order: `dl = d * (scale - 32)`, then `dl * q`.
fn ggml_q3_k(block: &[u8]) -> Vec<f32> {
    let d = f16_value(u16::from_le_bytes([block[108], block[109]]));
    let (hmask, qs, packed) = (&block[..32], &block[32..96], &block[96..108]);
    let mut aux = [0u32; 4];
    for (word, bytes) in aux.iter_mut().zip(packed.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    let tmp = aux[2];
    let (kmask1, kmask2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let scales: Vec<i8> = aux
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .map(|byte| byte as i8)
        .collect();
    let mut out = Vec::with_capacity(256);
    let mut scale = 0;
    for half in 0..2 {
        let q = &qs[half * 32..];
        for j in 0..4 {
            let (shift, mask) = (2 * j, 1u8 << (4 * half + j));
            for part in 0..2 {
                let dl = d * f32::from(scales[scale] - 32);
                scale += 1;
                for l in 16 * part..16 * part + 16 {
                    let low = ((q[l] >> shift) & 3) as i8;
                    let value = low - if hmask[l] & mask != 0 { 0 } else { 4 };
                    out.push(dl * f32::from(value));
                }
            }
        }
    }
    out
}

/// ggml's `dequantize_row_iq4_nl` for one 18-byte block.
fn ggml_iq4_nl(block: &[u8]) -> Vec<f32> {
    const VALUES: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let d = f16_value(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..18];
    let low = qs.iter().map(|q| d * f32::from(VALUES[usize::from(q & 15)]));
    let high = qs.iter().map(|q| d * f32::from(VALUES[usize::from(q >> 4)]));
    low.chain(high).collect()
}

fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    assert!(exponent < 31, "reference blocks use finite scales");
    if exponent == 0 {
        sign * mantissa * 2f32.powi(-24)
    } else {
        sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

/// Q3_K and IQ4_NL have no resident representation of their own: they import
/// into q6k and iq4g32. That is exact only if every stored value decodes to
/// ggml's own dequantization bit for bit, in every layout, including the
/// extremes (scale -32 with q = -4 is +128 d, outside int8, and each
/// sixteen-value sub-block keeps its own scale).
#[test]
fn q3_k_and_iq4_nl_import_exactly_as_ggml_dequantizes() {
    let rows = 3u64;
    let k = 512u64;
    // Q3_K: pseudo-random blocks plus one block of every extreme: all
    // scales 0 (s - 32 = -32) with every q = -4 (low 0, high clear), and
    // all scales 63 with q = 3.
    let mut q3 = source_bytes(rows * (k / 256) * 110, 7);
    for block in q3.chunks_exact_mut(110) {
        block[108..110].copy_from_slice(&0x3555u16.to_le_bytes());
    }
    q3[..108].fill(0);
    q3[110..142].fill(0xff);
    q3[142..206].fill(0xff);
    q3[206..218].fill(0xff);
    let q3_reference: Vec<f32> = q3.chunks_exact(110).flat_map(ggml_q3_k).collect();
    assert!(q3_reference[..256].iter().all(|v| *v == 128.0 * f16_value(0x3555)));
    assert!(q3_reference[256..512].iter().all(|v| *v == 93.0 * f16_value(0x3555)));
    // IQ4_NL: eight blocks per packet, each block with its own finite scale.
    let mut nl = source_bytes(rows * (k / 32) * 18, 11);
    for block in nl.chunks_exact_mut(18) {
        block[1] &= 0xbb;
    }
    let nl_reference: Vec<f32> = nl.chunks_exact(18).flat_map(ggml_iq4_nl).collect();
    for (source_name, resident, bytes, reference) in [
        ("gguf_q3_k", "q6k", &q3, &q3_reference),
        ("gguf_iq4_nl", "iq4g32", &nl, &nl_reference),
    ] {
        let source = seismic::Element::named(source_name).unwrap();
        let shape = [rows, k];
        assert_eq!(source.canonical_byte_len(&shape).unwrap(), bytes.len() as u64);
        for layout in seismic::Layout::ALL {
            let element = seismic::Element::stored(resident, layout).unwrap();
            let stored = element.repack_host(source, &shape, bytes).unwrap();
            let values = element.decode_host(&shape, &stored).unwrap();
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference
                    .iter()
                    .map(|v| f64::from(*v).to_bits())
                    .collect::<Vec<_>>(),
                "{source_name} -> {}",
                element.name()
            );
        }
    }
}

/// The formats imported into another format's representation decode, in
/// every layout, to exactly the values ggml's reference dequantization (gguf-py
/// `quants.dequantize`) produces for 64 pseudo-random blocks, whose f16 scale
/// is set to 0x3555: an FNV-1a digest of those values' f32 bits. The IQ3_S
/// blocks reach every grid point and sign. Regenerate a digest with gguf-py
/// over `golden_bytes(blocks * block_bytes, 12345)`.
#[test]
fn exact_imports_match_the_gguf_reference_dequantization() {
    for (source_name, resident, block_bytes, block_values, scale_offset, digest) in [
        ("gguf_q3_k", "q6k", 110, 256, 108, 0x8a3b_79aa_c79e_9c25u64),
        ("gguf_iq3_s", "q6k", 110, 256, 0, 0x35ce_a941_f3e6_18c5),
        ("gguf_iq4_nl", "iq4g32", 18, 32, 0, 0xf925_723c_01ab_1e25),
    ] {
        let blocks = 64u64;
        let mut bytes = golden_bytes(blocks * block_bytes, 12345);
        for block in bytes.chunks_exact_mut(block_bytes as usize) {
            block[scale_offset..scale_offset + 2].copy_from_slice(&0x3555u16.to_le_bytes());
        }
        let source = seismic::Element::named(source_name).unwrap();
        let shape = [1, blocks * block_values];
        for layout in seismic::Layout::ALL {
            let element = seismic::Element::stored(resident, layout).unwrap();
            let stored = element.repack_host(source, &shape, &bytes).unwrap();
            let values = element.decode_host(&shape, &stored).unwrap();
            let actual = values.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| {
                (hash ^ u64::from((*value as f32).to_bits())).wrapping_mul(0x0100_0000_01b3)
            });
            assert_eq!(actual, digest, "{source_name} -> {}", element.name());
        }
    }
}

/// Every byte of a full range: byte `i` is the top byte of
/// `(i * 2654435761 + seed) mod 2^32`.
fn golden_bytes(length: u64, seed: u32) -> Vec<u8> {
    (0..length as u32)
        .map(|index| (index.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 24) as u8)
        .collect()
}

/// Deterministic source bytes whose f16 factors are finite (so every decoded
/// value is a number and bit comparisons are meaningful).
fn source_bytes(length: u64, seed: u32) -> Vec<u8> {
    (0..length as u32)
        .map(|index| {
            let x = index.wrapping_add(seed).wrapping_mul(2_654_435_761);
            ((x >> 13) as u8) & 0x7b
        })
        .collect()
}

#[test]
fn k_quant_six_bit_locals_decode_as_gguf_defines_them() {
    // These coefficient bytes exercise both GGUF's low six-bit groups and
    // its split high-bit groups. A raw 12-byte copy decodes differently.
    let expected_scales = [1f64, 2., 3., 4., 48., 34., 20., 6.];
    let expected_minima = [5f64, 6., 7., 8., 33., 51., 5., 23.];
    for (source_name, resident, size, code) in [
        ("gguf_q4_k", "q4k", 144, 15.0f64),
        ("gguf_q5_k", "q5k", 176, 31.0f64),
    ] {
        let mut input = vec![0xffu8; size];
        input[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        input[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
        input[4..8].copy_from_slice(&[0xc1, 0x82, 0x43, 0x04]);
        input[8..12].copy_from_slice(&[0x85, 0xc6, 0x07, 0x48]);
        input[12..16].copy_from_slice(&[0x10, 0x32, 0x54, 0x76]);
        let source = seismic::Element::named(source_name).unwrap();
        for layout in seismic::Layout::ALL {
            let element = seismic::Element::stored(resident, layout).unwrap();
            let shape = [1, 256];
            let stored = element.repack_host(source, &shape, &input).unwrap();
            let values = element.decode_host(&shape, &stored).unwrap();
            for (position, value) in values.iter().enumerate() {
                let group = position / 32;
                let expected = code * expected_scales[group] - 0.5 * expected_minima[group];
                assert_eq!(*value, expected, "{} position {position}", element.name());
            }
        }
    }
}

#[test]
fn generated_surface_is_one_dense_import_and_one_exact_repack() {
    let _ = import_dense::for_device_with;
    let _ = import_dense::native_for_device_with;
    let _ = repack_weight::for_device_with;
    let _ = repack_weight::native_for_device_with;
    let source = include_str!("../kernels/import.seismic");
    assert!(!source.contains("copy_to_f32"));
    assert!(!source.contains("NegativeExp"));
}

/// The opened device of `backend`, when this host has one.
fn device(backend: seismic::BackendName) -> Option<seismic::Device> {
    seismic::DeviceCatalog::discover()
        .ok()?
        .open_backend(backend)
        .ok()
}

#[cfg(target_os = "macos")]
#[test]
fn metal_dense_import_matches_host_for_all_nine_pairs() {
    dense_import_matches_host_for_all_nine_pairs(&device(seismic::BackendName::Metal).unwrap());
}

#[test]
fn cpu_dense_import_matches_host_for_all_nine_pairs() {
    dense_import_matches_host_for_all_nine_pairs(&device(seismic::BackendName::Cpu).unwrap());
}

#[test]
fn cuda_dense_import_matches_host_for_all_nine_pairs() {
    if let Some(device) = device(seismic::BackendName::Cuda) {
        dense_import_matches_host_for_all_nine_pairs(&device);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_repack_matches_the_registered_conversion_for_every_format_and_layout() {
    repack_matches_the_registered_conversion(
        &device(seismic::BackendName::Metal).unwrap(),
        &[seismic::Layout::Rows16, seismic::Layout::Mma16],
    );
}

#[test]
fn cuda_repack_matches_the_registered_conversion_for_every_format_and_layout() {
    if let Some(device) = device(seismic::BackendName::Cuda) {
        repack_matches_the_registered_conversion(
            &device,
            &[seismic::Layout::Rows16, seismic::Layout::Mma16],
        );
    }
}

#[test]
fn vulkan_dense_import_matches_host_for_all_nine_pairs() {
    if let Some(device) = device(seismic::BackendName::Vulkan) {
        dense_import_matches_host_for_all_nine_pairs(&device);
    }
}

/// The CPU repacks into both supported row layouts.
#[test]
fn cpu_repack_matches_the_registered_conversion_for_every_format() {
    repack_matches_the_registered_conversion(
        &device(seismic::BackendName::Cpu).unwrap(),
        &[seismic::Layout::Rows16, seismic::Layout::Rows8],
    );
}

/// Vulkan repacks into both supported row layouts.
#[test]
fn vulkan_repack_matches_the_registered_conversion_for_every_format() {
    if let Some(device) = device(seismic::BackendName::Vulkan) {
        repack_matches_the_registered_conversion(&device, &[seismic::Layout::Rows16]);
    }
}

fn dense_import_matches_host_for_all_nine_pairs(device: &seismic::Device) {
    let device = device.clone();
    let values = [-3.25f32, -0.0, 0.125, 1.5, 19.0];
    // One row, and 21 rows of the same values over two matrices (rows of
    // several row blocks).
    for (matrices, rows) in [(1usize, 1usize), (3, 7)] {
        let repeated = |bytes: Vec<u8>| bytes.repeat(matrices * rows);
        for source_name in ["f32", "f16", "bf16"] {
            for destination_name in ["f32", "f16", "bf16"] {
                let source_element = seismic::Element::named(source_name).unwrap();
                let destination_element = seismic::Element::named(destination_name).unwrap();
                let source = seismic::Tensor::from_host(
                    &device,
                    source_element,
                    &[matrices as u64, rows as u64, values.len() as u64],
                    &repeated(dense_bytes(source_name, &values)),
                )
                .unwrap();
                let elements = import_dense::Elements {
                    E: source_element,
                    U: destination_element,
                };
                let native = import_dense::native_for_device_with(
                    &device,
                    elements,
                    &seismic::NativeSpecialization::new(),
                )
                .unwrap()
                .call(import_dense::Args { source: &source })
                .unwrap()
                .value;
                assert_eq!(
                    native.read_to_host().unwrap(),
                    repeated(dense_bytes(destination_name, &values)),
                    "{source_name}->{destination_name} over {matrices}x{rows} rows"
                );
            }
        }
    }
}

/// K8: the native repack of every (format, layout) conversion equals the
/// registry's host reference byte for byte, and its storage decodes to the
/// source's values. Shapes cover a row count off the 16-row tile, several
/// matrices, a packing axis with an odd number of q8 packets (mma16 pads
/// rows to whole 64-column k-blocks) and a partial trailing packet.
fn repack_matches_the_registered_conversion(device: &seismic::Device, layouts: &[seismic::Layout]) {
    let device = device.clone();
    for (source_name, resident) in FORMATS {
        let source_element = seismic::Element::named(source_name).unwrap();
        let group = source_element.logical_group().unwrap();
        for shape in [[1, 17, 3 * group], [3, 5, 2 * group - 8], [2, 16, group]] {
            let length = source_element.canonical_byte_len(&shape).unwrap();
            let bytes = source_bytes(length, shape[1] as u32);
            let source =
                seismic::Tensor::from_host(&device, source_element, &shape, &bytes).unwrap();
            let packet = seismic::Element::stored(resident, seismic::Layout::Packet).unwrap();
            let expected_values = packet
                .decode_host(
                    &shape,
                    &packet.repack_host(source_element, &shape, &bytes).unwrap(),
                )
                .unwrap();
            for &layout in layouts {
                let destination = seismic::Element::stored(resident, layout).unwrap();
                let native = repack_weight::native_for_device_with(
                    &device,
                    repack_weight::Elements {
                        E: source_element,
                        U: destination,
                    },
                    &seismic::NativeSpecialization::new(),
                )
                .unwrap()
                .call(repack_weight::Args { source: &source })
                .unwrap()
                .value;
                let actual = native.read_to_host().unwrap();
                let label = format!(
                    "{source_name} -> {} over {shape:?}",
                    destination.name()
                );
                assert_eq!(
                    actual,
                    destination
                        .repack_host(source_element, &shape, &bytes)
                        .unwrap(),
                    "{label}"
                );
                let values = destination.decode_host(&shape, &actual).unwrap();
                assert_eq!(
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{label}"
                );
            }
        }
    }
}
