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

/// Every conversion the registry declares: each GGUF source with each
/// operand recipe its conversion writes an element for.
fn conversions() -> Vec<(seismic::Element, seismic::OperandRecipe)> {
    seismic::Element::external_sources()
        .flat_map(|source| {
            seismic::OperandRecipe::ALL
                .into_iter()
                .filter(move |operands| source.converted(seismic::Layout::Packet, *operands).is_some())
                .map(move |operands| (source, operands))
        })
        .collect()
}

/// The reference blocks of `tests/support/gguf_vectors/<TYPE>.bin`
/// (`tests/support/gguf_reference.py vectors`): their bytes and gguf-py's
/// dequantization of them.
fn reference_vectors(encoding: magnitude_ggml::Encoding) -> (Vec<u8>, Vec<f32>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/support/gguf_vectors")
        .join(format!("{}.bin", encoding.ggml_name()));
    let file = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let blocks = u64::from(u32::from_le_bytes(file[..4].try_into().unwrap()));
    let bytes = (blocks * encoding.block_bytes()) as usize;
    let values = file[4 + bytes..]
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values.len() as u64, blocks * encoding.block_elements(), "{}", path.display());
    (file[4..4 + bytes].to_vec(), values)
}

/// Every GGUF source the registry converts from imports, in every layout and
/// for every operand recipe, to exactly the reference dequantization of its
/// type's vectors, in rows of 128 values (of one source packet, when
/// wider).
#[test]
fn every_gguf_source_imports_as_the_reference_dequantizes() {
    for (source, operands) in conversions() {
        let width = source.logical_group().unwrap().max(128);
        let encoding = magnitude_ggml::Encoding::ALL
            .into_iter()
            .find(|encoding| seismic::Element::gguf_source(*encoding, width) == Some(source))
            .unwrap_or_else(|| panic!("{} is no encoding's source", source.name()));
        let (bytes, reference) = reference_vectors(encoding);
        let shape = [reference.len() as u64 / width, width];
        for layout in seismic::Layout::ALL {
            let element = source.converted(layout, operands).unwrap();
            let stored = element.repack_host(source, &shape, &bytes).unwrap();
            let values = element.decode_host(&shape, &stored).unwrap();
            assert_eq!(
                values.iter().map(|value| (*value as f32).to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "{} -> {}",
                source.name(),
                element.name()
            );
        }
    }
}

/// Real tensors of released GGUF files import, in every layout, to exactly
/// the reference dequantization of their blocks. The dump comes from
/// `tests/support/gguf_reference.py dump <file> <dir>`, named by
/// `CODEC_REAL_DIR`.
#[test]
#[ignore = "needs CODEC_REAL_DIR, a real-tensor dump of tests/support/gguf_reference.py"]
fn real_gguf_tensors_import_as_the_reference_dequantizes() {
    let directory =
        std::path::PathBuf::from(std::env::var("CODEC_REAL_DIR").expect("CODEC_REAL_DIR"));
    let index = std::fs::read_to_string(directory.join("index.tsv")).unwrap();
    for line in index.lines() {
        let fields = line.split('\t').collect::<Vec<_>>();
        let [type_name, rows, k, stem] = fields.as_slice() else {
            panic!("index line `{line}`")
        };
        let shape = [rows.parse::<u64>().unwrap(), k.parse::<u64>().unwrap()];
        let blocks = std::fs::read(directory.join(format!("{stem}.blocks"))).unwrap();
        let reference = std::fs::read(directory.join(format!("{stem}.values"))).unwrap();
        let encoding = magnitude_ggml::Encoding::ALL
            .into_iter()
            .find(|encoding| encoding.ggml_name() == *type_name)
            .unwrap_or_else(|| panic!("type `{type_name}`"));
        let source = seismic::Element::gguf_source(encoding, shape[1]).unwrap();
        for layout in seismic::Layout::ALL {
            let element = source
                .converted(layout, seismic::OperandRecipe::Rounded)
                .unwrap();
            let stored = element.repack_host(source, &shape, &blocks).unwrap();
            let values = element.decode_host(&shape, &stored).unwrap();
            let mismatches = values
                .iter()
                .zip(reference.chunks_exact(4))
                .filter(|(value, bytes)| {
                    (**value as f32).to_bits() != u32::from_le_bytes((*bytes).try_into().unwrap())
                })
                .count();
            assert_eq!(values.len() * 4, reference.len(), "{stem}");
            assert_eq!(mismatches, 0, "{stem} -> {}", element.name());
        }
        println!("{stem}: {shape:?} exact in every layout");
    }
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

/// A GGUF Q6_K block of factor `d`, sub-block scales `scales` (per 16
/// values) and codes `codes` (`q + 32`, per value), as ggml lays it out:
/// ql[128] | qh[64] | scales[16] | d.
fn q6_k_block(d: [u8; 2], scales: [i8; 16], codes: [u8; 256]) -> Vec<u8> {
    let mut block = vec![0u8; 210];
    for (position, code) in codes.iter().enumerate() {
        let (half, rest) = (position / 128, position % 128);
        let (quarter, l) = (rest / 32, rest % 32);
        block[half * 64 + l + 32 * (quarter % 2)] |= (code & 15) << (4 * (quarter / 2));
        block[128 + half * 32 + l] |= (code >> 4) << (2 * quarter);
    }
    for (j, scale) in scales.iter().enumerate() {
        block[192 + j] = *scale as u8;
    }
    block[208..].copy_from_slice(&d);
    block
}

/// Q3_K blocks decode, in every layout, to exactly the values of their Q6_K
/// re-expression (q6k, the form they were held in before q3k): Q3_K's
/// `d (s - 32) q` is Q6_K's with scale `s - 32` and code `q + 32`.
#[test]
fn q3_k_decodes_as_its_q6_k_reexpression() {
    let encoding = magnitude_ggml::Encoding::Q3K;
    let reexpress = |block: &[u8]| {
        let scales = std::array::from_fn(|j| {
            let low = (block[96 + j % 8] >> (4 * (j / 8))) & 15;
            let high = (block[104 + j % 4] >> (2 * (j / 4))) & 3;
            (low | (high << 4)) as i8 - 32
        });
        let codes = std::array::from_fn(|p| {
            let low = (block[32 + 32 * (p / 128) + p % 32] >> (2 * (p % 128 / 32))) & 3;
            let high = (block[p % 32] >> (p / 32)) & 1;
            low + 4 * high - 4 + 32
        });
        q6_k_block([block[108], block[109]], scales, codes)
    };
    let source = seismic::Element::gguf_source(encoding, 256).unwrap();
    let (vectors, _) = reference_vectors(encoding);
    let bytes = [vectors, source_bytes(64 * encoding.block_bytes(), 7)].concat();
    let shape = [bytes.len() as u64 / encoding.block_bytes(), 256];
    let reexpressed = bytes
        .chunks_exact(encoding.block_bytes() as usize)
        .flat_map(reexpress)
        .collect::<Vec<_>>();
    let q6k = seismic::Element::stored("q6k", seismic::Layout::Packet).unwrap();
    let q6_k = seismic::Element::named("gguf_q6_k").unwrap();
    let expected = q6k
        .decode_host(&shape, &q6k.repack_host(q6_k, &shape, &reexpressed).unwrap())
        .unwrap();
    for layout in seismic::Layout::ALL {
        let element = source.converted(layout, seismic::OperandRecipe::Rounded).unwrap();
        let values = element
            .decode_host(&shape, &element.repack_host(source, &shape, &bytes).unwrap())
            .unwrap();
        assert_eq!(
            values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "{} -> {}",
            source.name(),
            element.name()
        );
    }
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
    seismic::Devices::discover(&[])
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
    // Metal's repack writes every storage of these layouts but CUDA's
    // `records`.
    repack_matches_the_registered_conversion(
        &device(seismic::BackendName::Metal).unwrap(),
        &[
            seismic::Layout::Packet,
            seismic::Layout::Rows16,
            seismic::Layout::Mma16,
            seismic::Layout::Rows32,
        ],
        false,
    );
}

/// The Metal repack's time and source read rate over a 4096 x 4096 tensor
/// of every GGUF source, into its packet form and the Metal resident
/// layout (`repack_timing`).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "timing"]
fn metal_repack_timing() {
    repack_timing(
        &device(seismic::BackendName::Metal).unwrap(),
        &[seismic::Layout::Packet, seismic::Layout::Rows32],
    );
}

/// The Vulkan repack's time and source read rate over a 4096 x 4096 tensor
/// of every GGUF source, into the Vulkan resident layout (`repack_timing`).
#[test]
#[ignore = "timing"]
fn vulkan_repack_timing() {
    if let Some(device) = device(seismic::BackendName::Vulkan) {
        repack_timing(&device, &[seismic::Layout::Rows16]);
    }
}

/// The CUDA repack's time and source read rate over a 4096 x 4096 tensor
/// of every GGUF source, into its packet form and both CUDA resident
/// layouts (`repack_timing`).
#[test]
#[ignore = "timing"]
fn cuda_repack_timing() {
    if let Some(device) = device(seismic::BackendName::Cuda) {
        repack_timing(
            &device,
            &[
                seismic::Layout::Packet,
                seismic::Layout::Rows16,
                seismic::Layout::Mma16,
            ],
        );
    }
}

/// The repack's time and source read rate on `device` over a 4096 x 4096
/// tensor of every GGUF source, into each of `layouts`: the best of eleven
/// batches of 20 repacks.
fn repack_timing(device: &seismic::Device, layouts: &[seismic::Layout]) {
    let shape = [1, 4096, 4096];
    for (source_element, operands) in conversions() {
        let length = source_element.canonical_byte_len(&shape).unwrap();
        let source =
            seismic::Tensor::from_host(device, source_element, &shape, &source_bytes(length, 7))
                .unwrap();
        for &layout in layouts {
            let destination = source_element.converted(layout, operands).unwrap();
            let kernel = repack_weight::native_for_device_with(
                device,
                repack_weight::Elements {
                    E: source_element,
                    U: destination,
                },
                &seismic::NativeSpecialization::new(),
            )
            .unwrap();
            let mut result = seismic::Tensor::zeros(device, destination, &shape).unwrap();
            const CALLS: u32 = 20;
            let best = (0..12)
                .map(|_| {
                    let mut batch = seismic::NativeTensorBatch::new(device);
                    for _ in 0..CALLS {
                        batch
                            .push(
                                &kernel,
                                repack_weight::Args { source: &source },
                                repack_weight::OutputArgs { value: &mut result },
                            )
                            .unwrap();
                    }
                    let started = std::time::Instant::now();
                    batch.submit().unwrap().wait().unwrap();
                    started.elapsed() / CALLS
                })
                .skip(1)
                .min()
                .unwrap();
            println!(
                "{:>18} -> {:<16} {:>8.1} us {:>7.1} GB/s",
                source_element.name(),
                destination.name(),
                best.as_secs_f64() * 1e6,
                length as f64 / best.as_secs_f64() / 1e9
            );
        }
    }
}

#[test]
fn cuda_repack_matches_the_registered_conversion_for_every_format_and_layout() {
    if let Some(device) = device(seismic::BackendName::Cuda) {
        repack_matches_the_registered_conversion(
            &device,
            &[
                seismic::Layout::Packet,
                seismic::Layout::Rows16,
                seismic::Layout::Mma16,
            ],
            true,
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
        false,
    );
}

/// Vulkan repacks into both supported row layouts.
#[test]
fn vulkan_repack_matches_the_registered_conversion_for_every_format() {
    if let Some(device) = device(seismic::BackendName::Vulkan) {
        repack_matches_the_registered_conversion(&device, &[seismic::Layout::Rows16], false);
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

/// Whether `element`'s storage holds whole `records` (CUDA's iq2xxs mma16).
fn has_records(element: seismic::Element) -> bool {
    use seismic_lang::registry::{self, RepresentationKind, RowPlaneContent};
    let info = registry::representation_info(registry::representation(element.name()).unwrap());
    matches!(&info.kind, RepresentationKind::PackedRows(rows)
        if rows.planes.iter().any(|plane| matches!(plane.content, RowPlaneContent::Records { .. })))
}

/// K8: the native repack of every (format, layout) conversion equals the
/// registry's host reference byte for byte, and its storage decodes to the
/// source's values. Shapes cover a row count off the 16-row tile, several
/// matrices, a packing axis with an odd number of q8 packets (mma16 pads
/// rows to whole 64-column k-blocks) and a partial trailing packet.
fn repack_matches_the_registered_conversion(
    device: &seismic::Device,
    layouts: &[seismic::Layout],
    records: bool,
) {
    let device = device.clone();
    for (source_element, operands) in conversions() {
        let source_name = source_element.name();
        let group = source_element.logical_group().unwrap();
        for shape in [[1, 17, 3 * group], [3, 5, 2 * group - 8], [2, 16, group]] {
            let length = source_element.canonical_byte_len(&shape).unwrap();
            let bytes = source_bytes(length, shape[1] as u32);
            let source =
                seismic::Tensor::from_host(&device, source_element, &shape, &bytes).unwrap();
            let packet = source_element.converted(seismic::Layout::Packet, operands).unwrap();
            let expected_values = packet
                .decode_host(
                    &shape,
                    &packet.repack_host(source_element, &shape, &bytes).unwrap(),
                )
                .unwrap();
            for &layout in layouts {
                let destination = source_element.converted(layout, operands).unwrap();
                if !records && has_records(destination) {
                    continue;
                }
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
                let label = format!("{source_name} -> {} over {shape:?}", destination.name());
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
