use magnitude_kernels::{constants, import_dense, landing_extent, repack_weight, routed_round, routed_step, SlotTable};

mod e4m3;

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
    let path = std::path::Path::new(
        &std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    )
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

/// E8M0 scales across their range (0, the F32 subnormal 2^-127; 254, where
/// the largest codes overflow; 255, NaN), each its byte and its value as
/// OCP MX defines it, 2^(scale - 127).
fn e8m0_scales() -> Vec<(Vec<u8>, f32)> {
    let e8m0 = |scale: u8| match scale {
        255 => f32::NAN,
        0 => f32::from_bits(0x0040_0000),
        scale => f32::from_bits(u32::from(scale) << 23),
    };
    [0u8, 1, 100, 120, 127, 140, 254, 255].map(|scale| (vec![scale], e8m0(scale))).to_vec()
}

/// F32 scales across the range the importer admits (`f32_scale`: zero, ±0,
/// 2^-117, amax / 448 values, values whose products round, one whose
/// largest products overflow), each its bytes and itself.
fn f32_scales() -> Vec<(Vec<u8>, f32)> {
    let least = f32::from_bits(10 << 23);
    [0.0f32, -0.0, least, -least, 2.232_142_9e-3, 0.1, -0.333_333_34, 1.000_000_1, 1234.567, 3.0e38]
        .map(|scale| (scale.to_le_bytes().to_vec(), scale))
        .to_vec()
}

/// Every E4M3 code under each scale of `grid`'s kind (`e8m0_scales`,
/// `f32_scales`), as the planar source of `grid` stores them: a tile's rows
/// of 256 values per scale, every row every code once, every tile of them
/// under the scale. Its bytes (the codes, then the grid), its shape, and
/// each value as the reference holds it: the code's value
/// (`e4m3::value`) times the scale, one F32 product.
fn grid_vectors(grid: seismic::Grid) -> (Vec<u8>, [u64; 2], Vec<f32>) {
    const COLUMNS: u64 = 256;
    let scales = match grid.scale {
        seismic::GridScale::E8m0 => e8m0_scales(),
        seismic::GridScale::F32 => f32_scales(),
    };
    let [tile_rows, tile_columns] = grid.tile;
    let (mut codes, mut scale_bytes, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for (bytes, scale) in &scales {
        for _ in 0..tile_rows {
            codes.extend(0..=u8::MAX);
            values.extend((0..=u8::MAX).map(|code| e4m3::value(code) * scale));
        }
        for _ in 0..COLUMNS / tile_columns {
            scale_bytes.extend_from_slice(bytes);
        }
    }
    ([codes, scale_bytes].concat(), [scales.len() as u64 * tile_rows, COLUMNS], values)
}

/// Every GGUF source the registry converts from imports, in every layout and
/// for every operand recipe, to exactly the reference dequantization of its
/// type's vectors: a GGUF type's gguf-py's, in rows of 128 values (of one
/// source packet, when wider), and a scale grid's planar codes and scales
/// (`grid_vectors`) as OCP MX defines them or the F32 reference does, NaN
/// where it is NaN.
#[test]
fn every_gguf_source_imports_as_the_reference_dequantizes() {
    use seismic::GgufScales;
    for (source, operands) in conversions() {
        let width = source.logical_group().unwrap().max(128);
        let encoding = |scales| {
            magnitude_ggml::Encoding::ALL
                .into_iter()
                .find(|encoding| seismic::Element::gguf_source(*encoding, scales, width) == Some(source))
        };
        let scales = GgufScales::ALL.into_iter().filter(|scales| encoding(*scales).is_some()).collect::<Vec<_>>();
        let (bytes, shape, reference) = match scales[..] {
            [GgufScales::Own] => {
                let (bytes, reference) = reference_vectors(encoding(GgufScales::Own).unwrap());
                let shape = [reference.len() as u64 / width, width];
                (bytes, shape, reference)
            }
            [GgufScales::Grid(grid)] => grid_vectors(grid),
            _ => panic!("{} is no one encoding's source", source.name()),
        };
        let bits = |value: f32| if value.is_nan() { f32::NAN.to_bits() } else { value.to_bits() };
        for layout in seismic::Layout::ALL {
            let element = source.converted(layout, operands).unwrap();
            let stored = element.repack_host(source, &shape, &bytes).unwrap();
            let values = element.decode_host(&shape, &stored).unwrap();
            assert_eq!(
                values.iter().map(|value| bits(*value as f32)).collect::<Vec<_>>(),
                reference.iter().map(|value| bits(*value)).collect::<Vec<_>>(),
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
        let source = seismic::Element::gguf_source(encoding, seismic::GgufScales::Own, shape[1]).unwrap();
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

/// A record port whose records lie in a graph's workspace (an import's
/// result read by the next node) binds through the slot's workspace record
/// table: f32 -> f16 -> f32 over 3 matrices of 7 rows returns the source.
#[cfg(target_os = "macos")]
#[test]
fn metal_record_ports_bind_graph_workspace_records() {
    let device = device(seismic::BackendName::Metal).unwrap();
    let values = [-3.25f32, -0.0, 0.125, 1.5, 19.0];
    let extents = [3, 7, values.len() as u64];
    let bytes = dense_bytes("f32", &values).repeat(21);
    let [f32, f16] = ["f32", "f16"].map(|name| seismic::Element::named(name).unwrap());
    let import = |source, destination| {
        import_dense::native_for_device_with(
            &device,
            import_dense::Elements { E: source, U: destination },
            &seismic::NativeSpecialization::new(),
        )
        .unwrap()
    };
    let (narrow, widen) = (import(f32, f16), import(f16, f32));
    let mut graph = device.native_graph();
    let source_port = graph.port(f32, &extents).unwrap();
    let result_port = graph.port(f32, &extents).unwrap();
    let narrowed = graph
        .enqueue(&narrow, import_dense::WorkflowArgs { source: source_port.tensor().into() })
        .unwrap()
        .value;
    let widened = graph
        .enqueue(&widen, import_dense::WorkflowArgs { source: (&narrowed).into() })
        .unwrap()
        .value;
    graph.store(&widened, &result_port).unwrap();
    let plan = graph.seal().unwrap();
    let source = seismic::Tensor::from_host(&device, f32, &extents, &bytes).unwrap();
    let result = seismic::Tensor::from_host(&device, f32, &extents, &vec![0; bytes.len()]).unwrap();
    let mut bindings = plan.bindings();
    bindings.set(&source_port, &source).unwrap();
    bindings.set(&result_port, &result).unwrap();
    let (_, completion) = plan
        .new_slot()
        .unwrap()
        .attach(bindings, plan.new_outputs().unwrap())
        .unwrap()
        .submit()
        .unwrap();
    completion.wait().unwrap();
    assert_eq!(result.read_to_host().unwrap(), bytes);
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

/// The CUDA repack into mma16 of the GGUF sources that keep their own size
/// there, at a real tensor's shape (Qwen3.5-2B's `attn_output`, 2048 x
/// 2048), equals the registry's host conversion byte for byte. IQ3_S comes
/// last: its rows' 24-byte k-blocks of grid entries are the only ones not a
/// power of two, which the mma16 grid placement once stored as 16-byte
/// units (a misaligned address on the device).
#[test]
fn cuda_repack_into_mma16_matches_the_host_at_a_real_shape() {
    let Some(device) = device(seismic::BackendName::Cuda) else {
        return;
    };
    use magnitude_ggml::Encoding;
    let shape = [1, 2048, 2048];
    for encoding in [
        Encoding::Iq2S,
        Encoding::Iq1S,
        Encoding::Iq1M,
        Encoding::Q3K,
        Encoding::Iq4Xs,
        Encoding::Q1_0,
        Encoding::Iq3S,
    ] {
        let source_element =
            seismic::Element::gguf_source(encoding, seismic::GgufScales::Own, shape[2]).unwrap();
        let destination = source_element
            .converted(seismic::Layout::Mma16, seismic::OperandRecipe::Rounded)
            .unwrap();
        let length = source_element.canonical_byte_len(&shape).unwrap();
        let bytes = source_bytes(length, 11);
        let source = seismic::Tensor::from_host(&device, source_element, &shape, &bytes).unwrap();
        let label = format!("{} -> {} over {shape:?}", source_element.name(), destination.name());
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
        .unwrap_or_else(|error| panic!("{label}: {error:?}"))
        .value;
        assert!(
            native.read_to_host().unwrap() == destination.repack_host(source_element, &shape, &bytes).unwrap(),
            "{label}"
        );
        println!("{label}: exact");
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

/// A planar source's streamed conversion (`routed_round`, its window holding
/// the codes and the scales of their tiles) at GLM-5.3-Flash's real expert
/// extents, up/gate and down, equals the registry's host conversion byte for
/// byte: the planar windows' tile grids at a width of many windows, which the
/// small shapes of the every-format test do not reach.
fn planar_streamed_conversion_matches_the_host_at_real_extents(device: &seismic::Device) {
    let resident = device.backend().resident_layout();
    for (source_element, operands) in conversions().into_iter().filter(|(element, _)| element.is_planar()) {
        let destination = source_element.converted(resident, operands).unwrap();
        if has_records(destination) {
            continue;
        }
        for shape in [[1, 16, 256], [1, 16, 6144], [1, 48, 6144], [2, 2048, 6144], [2, 6144, 2048]] {
            let bytes = source_bytes(source_element.canonical_byte_len(&shape).unwrap(), shape[1] as u32);
            let source = seismic::Tensor::from_host(device, source_element, &shape, &bytes).unwrap();
            let label = format!("{} -> {} over {shape:?}", source_element.name(), destination.name());
            let actual = streamed_conversion(device, &source, destination, &shape);
            let expected = destination.repack_host(source_element, &shape, &bytes).unwrap();
            assert!(
                actual == expected,
                "{label}: streamed: {} of {} bytes differ, the first at {:?}",
                actual.iter().zip(&expected).filter(|(a, b)| a != b).count(),
                expected.len(),
                actual.iter().zip(&expected).position(|(a, b)| a != b)
            );
        }
    }
}

#[test]
fn cuda_planar_streamed_conversion_matches_the_host_at_real_extents() {
    if let Some(device) = device(seismic::BackendName::Cuda) {
        planar_streamed_conversion_matches_the_host_at_real_extents(&device);
    }
}

#[test]
fn metal_planar_streamed_conversion_matches_the_host_at_real_extents() {
    if let Some(device) = device(seismic::BackendName::Metal) {
        planar_streamed_conversion_matches_the_host_at_real_extents(&device);
    }
}

#[test]
fn vulkan_planar_streamed_conversion_matches_the_host_at_real_extents() {
    if let Some(device) = device(seismic::BackendName::Vulkan) {
        planar_streamed_conversion_matches_the_host_at_real_extents(&device);
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
    use seismic_lang::registry::{self, RepresentationKind};
    let info = registry::representation_info(registry::representation(element.name()).unwrap());
    matches!(&info.kind, RepresentationKind::PackedRows(rows) if rows.records())
}

/// The columns of a repack unit into `element`'s row layout.
fn unit_columns(element: seismic::Element) -> u32 {
    use seismic_lang::{registry, repack};
    let info = registry::representation_info(registry::representation(element.name()).unwrap());
    repack::conversion_unit_columns(registry::ConversionLayout::of(&info.kind).unwrap())
}

/// The rows of a conversion's row tile: repack's 16-row tile on every
/// streaming backend.
const ROW_TILE: u64 = 16;

/// The scratch slots a round lands (`routed_round`) from `source` (a stored
/// [records, N, K] stack, the conveyor's up-only records, their down dense
/// BF16 zeros), each record into the slot of its index, the round served
/// before its call.
fn streamed_conversion(
    device: &seismic::Device,
    source: &seismic::Tensor,
    destination: seismic::Element,
    shape: &[u64; 3],
) -> Vec<u8> {
    const SEQUENCE: i32 = 1;
    let words = |values: &[i32]| {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        seismic::Tensor::from_host(device, seismic::Element::i32(), &[values.len() as u64], &bytes).unwrap()
    };
    let shared = |shape: &[u64], values: &[i32]| {
        let tensor = seismic::Tensor::shared(device, seismic::Element::i32(), shape).unwrap();
        // SAFETY: no device work uses the tensor yet.
        unsafe { tensor.write_shared(0, &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap() };
        tensor
    };
    let zeros = |element: seismic::Element, shape: &[u64]| seismic::Tensor::zeros(device, element, shape).unwrap();
    let mut step = words(&[0, 0]);
    routed_step::native_for_device(device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(routed_step::Args {
            step: &mut step,
            sequence: &words(&[SEQUENCE, 0]),
        })
        .unwrap();
    // Every record into its own slot, all landed.
    let records = shape[0];
    let entry = |index: u64| landing_extent(index) as usize;
    let mut landing = vec![0; entry(records)];
    landing[constants::LANDING_HEAD as usize] = SEQUENCE;
    landing[constants::LANDING_HEAD as usize + 1] = records as i32;
    for record in 0..records {
        landing[entry(record) + constants::LANDING_FROM as usize] = record as i32;
        landing[entry(record) + constants::LANDING_TO as usize] = record as i32;
    }
    // The round's lane answered for the step.
    let table = SlotTable::scratch(1, 1);
    let mut scratch = vec![0; table.words() as usize];
    scratch[(table.lane(SEQUENCE as u64, constants::SCRATCH_ROUND) + constants::SLOT_ANSWER) as usize] = SEQUENCE;
    let bf16 = seismic::Element::bf16();
    let down = [records, shape[2], shape[1]];
    let mut gate = zeros(destination, &[0, shape[1], shape[2]]);
    let mut slots = zeros(destination, shape);
    let mut down_slots = zeros(bf16, &down);
    let list = magnitude_kernels::list_extent(1);
    let native = routed_round::native_for_device_with(
        device,
        routed_round::Elements {
            SG: source.element(),
            SU: source.element(),
            SD: bf16,
            UG: destination,
            UU: destination,
            UD: bf16,
        },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap();
    native
    .call_answered(routed_round::Args {
        step: &step,
        scratch: &mut shared(&table.extents(), &scratch),
        round: 0,
        landing: &mut shared(&[landing.len() as u64], &landing),
        tiered: &words(&vec![0; list as usize]),
        conveyor_gate: &zeros(source.element(), &[0, shape[1], shape[2]]),
        conveyor_up: source,
        conveyor_down: &zeros(bf16, &down),
        gate: &mut gate,
        up: &mut slots,
        down: &mut down_slots,
        seeds: &words(&vec![0; list as usize]),
        map: &words(&vec![0; constants::MAP_WORDS as usize]).reshape(&[1, constants::MAP_WORDS]).unwrap(),
        grid: &words(&[constants::LAND_GROUPS as i32, 1, 1]),
    })
    .unwrap();
    slots.read_to_host().unwrap()
}

/// K8: the native repack of every (format, layout) conversion equals the
/// registry's host reference byte for byte, and its storage decodes to the
/// source's values; where the backend streams, so does its streamed
/// conversion (`routed_round`, record by record into scratch slots),
/// so a streamed record holds what the resident import places. Shapes cover
/// a row count off the 16-row tile, several matrices, a packing axis with an
/// odd number of q8 packets (mma16 pads rows to whole 64-column k-blocks),
/// a partial trailing packet; the streamed conversion of whole tiles of
/// several records is at a row tile of several windows that a converting
/// block's mirror does not hold.
fn repack_matches_the_registered_conversion(
    device: &seismic::Device,
    layouts: &[seismic::Layout],
    records: bool,
) {
    let device = device.clone();
    let streams = routed_round::native_implementation(&device)
        .unwrap()
        .is_some();
    for (source_element, operands) in conversions() {
        let source_name = source_element.name();
        let group = source_element.logical_group().unwrap();
        // Whole tiles of several records: their streamed conversion runs at
        // the wide record's width (below), so the streamed entry converts as
        // many shapes as the others.
        let whole_tiles = [2, 16, group];
        for shape in [[1, 17, 3 * group], [3, 5, 2 * group - 8], whole_tiles] {
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
                // Metal places mma16 code planes as 32-column units only.
                if device.backend() == seismic::BackendName::Metal
                    && layout == seismic::Layout::Mma16
                    && unit_columns(destination) != 32
                {
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
                let expected = destination
                    .repack_host(source_element, &shape, &bytes)
                    .unwrap();
                assert_eq!(actual, expected, "{label}");
                if streams && layout == device.backend().resident_layout() && shape != whole_tiles {
                    assert!(
                        streamed_conversion(&device, &source, destination, &shape) == expected,
                        "{label}: streamed"
                    );
                }
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
        // A streamed record whose row tile is wider than a converting
        // block's mirror (CUDA then converts it in place) and spans several
        // windows, as the catalog's widest records do.
        let resident = device.backend().resident_layout();
        let destination = source_element.converted(resident, operands).filter(|_| streams && layouts.contains(&resident));
        if let Some(destination) = destination.filter(|&destination| records || !has_records(destination)) {
            let mut wide = group;
            while destination.canonical_byte_len(&[1, ROW_TILE, wide]).unwrap() <= constants::MIRROR_BYTES {
                wide *= 2;
            }
            let shape = [whole_tiles[0], whole_tiles[1], wide];
            let bytes = source_bytes(source_element.canonical_byte_len(&shape).unwrap(), shape[1] as u32);
            let source = seismic::Tensor::from_host(&device, source_element, &shape, &bytes).unwrap();
            assert!(
                streamed_conversion(&device, &source, destination, &shape)
                    == destination.repack_host(source_element, &shape, &bytes).unwrap(),
                "{source_name} -> {} over {shape:?}: streamed",
                destination.name()
            );
        }
    }
}
