//! `draft_rows` on every device the host has (each accelerator and the CPU),
//! over its resident layout (Metal `rows32`, Vulkan and CPU `rows16`, CUDA
//! `mma16`): the embedding table and the combine weight are GGUF packets
//! repacked to that layout, and the result is checked against a host model of
//! the entry's contract. The table takes every form a target embedding table
//! a draft head reads is stored in (Q8_0, Q4_0, Q4_K, Q6_K, BF16).

use magnitude_kernels::{draft_rows, repack_weight};
use seismic::{Device, Element, Layout, NativeSpecialization, Tensor};

fn f16_bits(value: f32) -> u16 {
    seismic_lang::registry::f16_bits(value)
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn code(&mut self) -> i8 {
        (self.next() % 255) as i8
    }
    fn symmetric(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }
}

/// The resident layout of the device's backend.
fn resident_layout(device: &Device) -> Layout {
    match device.backend() {
        seismic::BackendName::Cuda => Layout::Mma16,
        seismic::BackendName::Metal => Layout::Rows32,
        _ => Layout::Rows16,
    }
}

/// The resident q8 element of the device's backend.
fn resident_q8(device: &Device) -> Element {
    Element::stored("q8g32s", resident_layout(device)).unwrap()
}

/// The form of an embedding table: a GGUF block format, or BF16.
#[derive(Clone, Copy, Debug)]
enum TableForm {
    Q8_0,
    Q4_0,
    Q4K,
    Q6K,
    Bf16,
}

/// A GGUF block format: its external element, the resident representation
/// it loads as, its block bytes and values, and the offset of each f16 block
/// scale with the largest code magnitude that scale multiplies.
struct GgufBlocks {
    external: &'static str,
    resident: &'static str,
    block_bytes: usize,
    block_values: usize,
    scales: &'static [(usize, f32)],
}

impl TableForm {
    const ALL: [TableForm; 5] = [Self::Q8_0, Self::Q4_0, Self::Q4K, Self::Q6K, Self::Bf16];

    fn gguf(self) -> Option<GgufBlocks> {
        let blocks = |external, resident, block_bytes, block_values, scales| GgufBlocks {
            external,
            resident,
            block_bytes,
            block_values,
            scales,
        };
        match self {
            Self::Q8_0 => Some(blocks("gguf_q8_0", "q8g32s", 34, 32, &[(0, 127.0)])),
            Self::Q4_0 => Some(blocks("gguf_q4_0", "q4g32s", 18, 32, &[(0, 8.0)])),
            Self::Q4K => Some(blocks(
                "gguf_q4_k",
                "q4k",
                144,
                256,
                &[(0, 63.0 * 15.0), (2, 63.0 * 63.0)],
            )),
            Self::Q6K => Some(blocks("gguf_q6_k", "q6k", 210, 256, &[(208, 127.0 * 32.0)])),
            Self::Bf16 => None,
        }
    }
}

/// A `[rows, columns]` embedding table of `form` with values of magnitude
/// about `scale`, in the device's resident form (a GGUF table through the
/// device repack, as loading places it), and its logical values (the host
/// reference decode of the registry's conversion).
fn table_weight(
    device: &Device,
    form: TableForm,
    rows: usize,
    columns: usize,
    scale: f32,
    random: &mut Random,
) -> (Tensor, Vec<f32>) {
    let Some(GgufBlocks {
        external,
        resident,
        block_bytes,
        block_values,
        scales,
    }) = form.gguf()
    else {
        let values = (0..rows * columns)
            .map(|_| bf16_round(scale * random.symmetric()))
            .collect::<Vec<_>>();
        return (
            bf16_tensor(device, &[rows as u64, columns as u64], &values),
            values,
        );
    };
    assert_eq!(columns % block_values, 0);
    let mut bytes = Vec::with_capacity(rows * columns / block_values * block_bytes);
    for _ in 0..rows * columns / block_values {
        let mut block = (0..block_bytes)
            .map(|_| random.next() as u8)
            .collect::<Vec<_>>();
        for &(offset, codes) in scales {
            let factor = scale / codes * (0.5 + 0.5 * (random.symmetric() + 1.0));
            block[offset..offset + 2].copy_from_slice(&f16_bits(factor).to_le_bytes());
        }
        bytes.extend(block);
    }
    let external = Element::named(external).unwrap();
    let shape = [rows as u64, columns as u64];
    let host = Element::stored(resident, Layout::Rows16).unwrap();
    let bytes = host.repack_source(external, bytes).unwrap();
    let values = host
        .decode_host(&shape, &host.repack_host(external, &shape, &bytes).unwrap())
        .unwrap()
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let source =
        Tensor::from_host(device, external, &[1, rows as u64, columns as u64], &bytes).unwrap();
    let table = repack_weight::native_for_device_with(
        device,
        repack_weight::Elements {
            E: external,
            U: Element::stored(resident, resident_layout(device)).unwrap(),
        },
        &repack_weight::native_implementation(device)
            .unwrap()
            .expect("repack_weight has a native implementation on every device")
            .default_specialization(&NativeSpecialization::new())
            .unwrap(),
    )
    .unwrap()
    .call(repack_weight::Args { source: &source })
    .unwrap()
    .value
    .reshape(&shape)
    .unwrap();
    (table, values)
}

/// A `[rows, columns]` GGUF Q8_0 weight repacked to the resident q8 layout,
/// and its logical values.
fn q8_weight(
    device: &Device,
    rows: usize,
    columns: usize,
    scale: f32,
    random: &mut Random,
) -> (Tensor, Vec<f32>) {
    assert_eq!(columns % 32, 0);
    let scale_bits = f16_bits(scale);
    let scale = half_to_f32(scale_bits);
    let mut bytes = Vec::new();
    let mut values = Vec::with_capacity(rows * columns);
    for _ in 0..rows * columns / 32 {
        bytes.extend_from_slice(&scale_bits.to_le_bytes());
        for _ in 0..32 {
            let code = random.code();
            bytes.push(code as u8);
            values.push(scale * f32::from(code));
        }
    }
    let external = Element::named("gguf_q8_0").unwrap();
    let source =
        Tensor::from_host(device, external, &[1, rows as u64, columns as u64], &bytes).unwrap();
    let weight = repack_weight::native_for_device_with(
        device,
        repack_weight::Elements {
            E: external,
            U: resident_q8(device),
        },
        // Declared parameters (the CPU's rows per work item) take their
        // first values.
        &repack_weight::native_implementation(device)
            .unwrap()
            .expect("repack_weight has a native implementation on every device")
            .default_specialization(&NativeSpecialization::new())
            .unwrap(),
    )
    .unwrap()
    .call(repack_weight::Args { source: &source })
    .unwrap()
    .value
    .reshape(&[rows as u64, columns as u64])
    .unwrap();
    (weight, values)
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

fn i32_tensor(device: &Device, shape: &[u64], values: &[i32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), shape, &bytes).unwrap()
}

fn bf16_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::bf16(), shape, &bytes).unwrap()
}

/// The entry's contract on the host: both RMS norms rounded to bf16, then the
/// F32 combine projection. Returns the result and, per output, the sum of
/// |w * x| that bounds the effect of one-ulp rounding differences.
fn reference(
    tokens: &[i32],
    table: &[f32],
    conditioning: &[f32],
    embedding_norm: &[f32],
    hidden_norm: &[f32],
    combine: &[f32],
    width: usize,
    epsilon: f32,
) -> (Vec<f32>, Vec<f32>) {
    let normalized = |values: &[f32], norm: &[f32]| {
        let squares = values.iter().map(|v| v * v).sum::<f32>();
        let inverse = 1.0 / (squares / width as f32 + epsilon).sqrt();
        values
            .iter()
            .zip(norm)
            .map(|(v, w)| bf16_round(v * inverse * w))
            .collect::<Vec<_>>()
    };
    let mut result = Vec::new();
    let mut magnitude = Vec::new();
    for (row, &token) in tokens.iter().enumerate() {
        let embedded = table[token as usize * width..][..width]
            .iter()
            .map(|v| bf16_round(*v))
            .collect::<Vec<_>>();
        let mut joined = normalized(&embedded, embedding_norm);
        joined.extend(normalized(
            &conditioning[row * width..][..width],
            hidden_norm,
        ));
        for output in 0..width {
            let weights = &combine[output * 2 * width..][..2 * width];
            result.push(weights.iter().zip(&joined).map(|(w, x)| w * x).sum());
            magnitude.push(
                weights
                    .iter()
                    .zip(&joined)
                    .map(|(w, x)| (w * x).abs())
                    .sum(),
            );
        }
    }
    (result, magnitude)
}

#[test]
fn draft_rows_match_the_host_model_over_resident_weights() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    for backend in [
        seismic::BackendName::Metal,
        seismic::BackendName::Cuda,
        seismic::BackendName::Vulkan,
        seismic::BackendName::Cpu,
    ] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        draft_rows_match_on(&device);
    }
}

fn draft_rows_match_on(device: &Device) {
    // Vulkan fixes the width D at preparation (it sizes the shared joined
    // row); the other backends declare no statics. Declared parameters (the
    // CPU's weight rows per work item) take their first values.
    let implementation = draft_rows::native_implementation(device)
        .unwrap()
        .expect("draft_rows has a native implementation on every device");
    let prepare = |width: usize, int8: bool| {
        let statics =
            implementation
                .statics
                .iter()
                .fold(NativeSpecialization::new(), |statics, name| {
                    assert_eq!(name, "D", "draft_rows may only fix D");
                    statics.with_static(name.clone(), width as u64)
                });
        let mut statics = implementation.default_specialization(&statics).unwrap();
        if device.backend() == seismic::BackendName::Cpu {
            statics = statics.with_param("INT8", u64::from(int8));
        }
        draft_rows::native_for_device_with(
            device,
            draft_rows::Elements {
                C: Element::bf16(),
                EW: resident_q8(device),
                A: Element::bf16(),
                EN: Element::bf16(),
                HN: Element::bf16(),
                CW: resident_q8(device),
            },
            &statics,
        )
        .unwrap()
    };
    let epsilon = 1.0e-6;
    // Widths of whole 64-column k-blocks, below and past one block's 32
    // outputs, decode rows and the partial/full tiles of Metal prompt GEMM.
    for (width, vocabulary, rows) in [
        (64, 5, 1),
        (128, 7, 3),
        (192, 4, 2),
        (64, 5, 17),
        (128, 7, 64),
        (192, 4, 65),
    ] {
        let kernel = prepare(width, false);
        let mut random = Random(width as u64 * 31 + rows as u64);
        let (table, table_values) = q8_weight(&device, vocabulary, width, 0.03125, &mut random);
        let (combine, combine_values) =
            q8_weight(&device, width, 2 * width, 0.0078125, &mut random);
        let embedding_norm_values = (0..width)
            .map(|_| bf16_round(1.0 + 0.25 * random.symmetric()))
            .collect::<Vec<_>>();
        let hidden_norm_values = (0..width)
            .map(|_| bf16_round(1.0 + 0.25 * random.symmetric()))
            .collect::<Vec<_>>();
        let conditioning_values = (0..rows * width)
            .map(|_| bf16_round(2.0 * random.symmetric()))
            .collect::<Vec<_>>();
        // Selection rows (token, status); the last is a failed selection,
        // which embeds token 0.
        let selections = (0..rows)
            .map(|row| {
                if row + 1 == rows {
                    [-1, 1]
                } else {
                    [(random.next() % vocabulary as u32) as i32, 0]
                }
            })
            .collect::<Vec<_>>();
        let token_values = selections
            .iter()
            .map(|[token, _]| (*token).max(0))
            .collect::<Vec<_>>();
        let token_bytes = selections
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tokens =
            Tensor::from_host(&device, Element::i32(), &[rows as u64, 2], &token_bytes).unwrap();
        let conditioning = bf16_tensor(&device, &[rows as u64, width as u64], &conditioning_values);
        let embedding_norm = bf16_tensor(&device, &[width as u64], &embedding_norm_values);
        let hidden_norm = bf16_tensor(&device, &[width as u64], &hidden_norm_values);
        // An empty directory: the selections are the table's rows.
        let direct = i32_tensor(device, &[0, 1], &[]);
        let result = kernel
            .call(draft_rows::Args {
                tokens: &tokens,
                directory: &direct,
                table: &table,
                conditioning: &conditioning,
                embedding_norm: &embedding_norm,
                hidden_norm: &hidden_norm,
                combine: &combine,
                epsilon,
            })
            .unwrap()
            .value;
        let actual = result
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let (expected, magnitude) = reference(
            &token_values,
            &table_values,
            &conditioning_values,
            &embedding_norm_values,
            &hidden_norm_values,
            &combine_values,
            width,
            epsilon,
        );
        assert_eq!(actual.len(), expected.len());
        for (index, ((actual, expected), magnitude)) in
            actual.iter().zip(&expected).zip(&magnitude).enumerate()
        {
            // One bf16 ulp of any joined input, plus F32 reassociation.
            let bound = magnitude * 2f32.powi(-8) + 1.0e-5;
            assert!(
                (actual - expected).abs() <= bound,
                "{:?} width {width} rows {rows} index {index}: {actual} vs {expected} (bound {bound})",
                device.backend()
            );
        }
        // Through a directory of a table's slots: a selection reads its
        // row's slot, exactly as a direct read of that slot does. Every
        // other logical row has a slot; a row without one and a failed
        // selection read slot 0.
        let logical = 3 * vocabulary;
        let directory_values = (0..logical)
            .map(|row| {
                if row % 2 == 0 {
                    ((row / 2 + 1) % vocabulary) as i32
                } else {
                    -1
                }
            })
            .collect::<Vec<_>>();
        let selected = (0..rows)
            .map(|row| match row % 4 {
                3 => [-1, 1],
                _ => [((row * 5 + 1) % logical) as i32, (row % 2) as i32],
            })
            .collect::<Vec<_>>();
        let slots = selected
            .iter()
            .map(|[row, status]| {
                let slot = usize::try_from(*row).map_or(0, |row| directory_values[row].max(0));
                [slot, *status]
            })
            .collect::<Vec<_>>();
        let read = |tokens: &[[i32; 2]], directory: &Tensor| {
            kernel
                .call(draft_rows::Args {
                    tokens: &i32_tensor(device, &[rows as u64, 2], tokens.as_flattened()),
                    directory,
                    table: &table,
                    conditioning: &conditioning,
                    embedding_norm: &embedding_norm,
                    hidden_norm: &hidden_norm,
                    combine: &combine,
                    epsilon,
                })
                .unwrap()
                .value
                .read_to_host()
                .unwrap()
        };
        assert_eq!(
            read(
                &selected,
                &i32_tensor(device, &[logical as u64, 1], &directory_values)
            ),
            read(&slots, &direct),
            "{:?} width {width} rows {rows}: a read through the directory",
            device.backend()
        );
        if device.backend() == seismic::BackendName::Cpu {
            let quantized = prepare(width, true)
                .call(draft_rows::Args {
                    tokens: &tokens,
                    directory: &direct,
                    table: &table,
                    conditioning: &conditioning,
                    embedding_norm: &embedding_norm,
                    hidden_norm: &hidden_norm,
                    combine: &combine,
                    epsilon,
                })
                .unwrap()
                .value;
            let quantized = quantized.read_to_host().unwrap();
            let quantized = quantized
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            let error = actual
                .iter()
                .zip(&quantized)
                .map(|(exact, int8)| (exact - int8).powi(2))
                .sum::<f32>();
            let scale = actual.iter().map(|value| value.powi(2)).sum::<f32>();
            assert!(
                error <= 0.05f32.powi(2) * scale,
                "CPU INT8 draft width {width} rows {rows}: relative error {}",
                (error / scale).sqrt()
            );
        }
    }
}

#[test]
fn draft_rows_preserve_distinct_f32_streams() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    for backend in [
        seismic::BackendName::Cuda,
        seismic::BackendName::Metal,
        seismic::BackendName::Cpu,
        seismic::BackendName::Vulkan,
    ] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        let width = 64;
        let streams = 4;
        for rows in [1, 4, 5, 16, 17, 65] {
            let mut random = Random(291 + rows as u64);
            let (table, tv) = q8_weight(&device, 3, width, 0.03125, &mut random);
            let (combine, cv) = q8_weight(&device, width, 2 * width, 0.0078125, &mut random);
            let hv = (0..rows * streams * width)
                .map(|i| random.symmetric() * (1 + i / width % streams) as f32 + 0.00317)
                .collect::<Vec<_>>();
            let hn = (0..streams * width)
                .map(|i| bf16_round(0.5 + i as f32 / (streams * width) as f32))
                .collect::<Vec<_>>();
            let en = vec![1.0; width];
            let tokens = i32_tensor(
                &device,
                &[rows as u64, 2],
                &(0..rows)
                    .flat_map(|i| [(i % 3) as i32, 0])
                    .collect::<Vec<_>>(),
            );
            let conditioning = Tensor::from_host(
                &device,
                Element::f32(),
                &[rows as u64, (streams * width) as u64],
                &hv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )
            .unwrap();
            let embedding_norm = bf16_tensor(&device, &[width as u64], &en);
            let hidden_norm = bf16_tensor(&device, &[(streams * width) as u64], &hn);
            let direct = i32_tensor(&device, &[0, 1], &[]);
            let implementation = draft_rows::native_implementation(&device).unwrap().unwrap();
            let statics = implementation
                .statics
                .iter()
                .fold(NativeSpecialization::new(), |s, name| {
                    s.with_static(name.clone(), width as u64)
                });
            let mut statics = implementation.default_specialization(&statics).unwrap();
            if backend == seismic::BackendName::Cpu {
                statics = statics.with_param("INT8", 0);
            }
            let kernel = draft_rows::native_for_device_with(
                &device,
                draft_rows::Elements {
                    C: Element::f32(),
                    A: Element::bf16(),
                    EW: resident_q8(&device),
                    EN: Element::bf16(),
                    HN: Element::bf16(),
                    CW: resident_q8(&device),
                },
                &statics,
            )
            .unwrap();
            let result = kernel
                .call(draft_rows::Args {
                    tokens: &tokens,
                    directory: &direct,
                    table: &table,
                    conditioning: &conditioning,
                    embedding_norm: &embedding_norm,
                    hidden_norm: &hidden_norm,
                    combine: &combine,
                    epsilon: 1e-6,
                })
                .unwrap()
                .value;
            assert_eq!(result.extents(), &[rows as u64, (streams * width) as u64]);
            let actual = result
                .read_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            for row in 0..rows {
                for stream in 0..streams {
                    let offset = (row * streams + stream) * width;
                    let (expected, magnitude) = reference(
                        &[(row % 3) as i32],
                        &tv,
                        &hv[offset..offset + width],
                        &en,
                        &hn[stream * width..(stream + 1) * width],
                        &cv,
                        width,
                        1e-6,
                    );
                    for column in 0..width {
                        let bound = magnitude[column] * 2f32.powi(-8) + 1e-5;
                        assert!(
                            (actual[offset + column] - expected[column]).abs() <= bound,
                            "{backend:?}: row {row} stream {stream} column {column}: {} vs {}",
                            actual[offset + column],
                            expected[column]
                        );
                    }
                }
            }
        }
    }
}

/// Every table form at the widths of real draft heads (Qwen3.5 0.8B: 1024,
/// Gemma 4 E2B: 1536): decode rows and a prompt class, tokens spread over a
/// table whose rows are not a whole number of any tile.
#[test]
fn draft_rows_match_the_host_model_over_every_table_form() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    for backend in [
        seismic::BackendName::Metal,
        seismic::BackendName::Cuda,
        seismic::BackendName::Vulkan,
        seismic::BackendName::Cpu,
    ] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        for form in TableForm::ALL {
            for width in [1024, 1536] {
                for rows in [1, 3, 17] {
                    draft_rows_match_with_table(&device, form, width, 37, rows);
                }
            }
        }
    }
}

fn draft_rows_match_with_table(
    device: &Device,
    form: TableForm,
    width: usize,
    vocabulary: usize,
    rows: usize,
) {
    let implementation = draft_rows::native_implementation(device)
        .unwrap()
        .expect("draft_rows has a native implementation on every device");
    let statics = implementation
        .statics
        .iter()
        .fold(NativeSpecialization::new(), |statics, name| {
            statics.with_static(name.clone(), width as u64)
        });
    let mut statics = implementation.default_specialization(&statics).unwrap();
    if device.backend() == seismic::BackendName::Cpu {
        statics = statics.with_param("INT8", 0);
    }
    let mut random = Random(width as u64 * 131 + rows as u64 * 7 + form as u64);
    let (table, table_values) = table_weight(device, form, vocabulary, width, 1.0, &mut random);
    let kernel = draft_rows::native_for_device_with(
        device,
        draft_rows::Elements {
            C: Element::bf16(),
            EW: table.element(),
            A: Element::bf16(),
            EN: Element::bf16(),
            HN: Element::bf16(),
            CW: resident_q8(device),
        },
        &statics,
    )
    .unwrap();
    let (combine, combine_values) = q8_weight(device, width, 2 * width, 0.0009765625, &mut random);
    let embedding_norm_values = (0..width)
        .map(|_| bf16_round(1.0 + 0.25 * random.symmetric()))
        .collect::<Vec<_>>();
    let hidden_norm_values = (0..width)
        .map(|_| bf16_round(1.0 + 0.25 * random.symmetric()))
        .collect::<Vec<_>>();
    let conditioning_values = (0..rows * width)
        .map(|_| bf16_round(2.0 * random.symmetric()))
        .collect::<Vec<_>>();
    // The last and odd rows of the table, then spread rows.
    let token_values = (0..rows)
        .map(|row| match row {
            0 => vocabulary as i32 - 1,
            1 => 1,
            _ => (random.next() % vocabulary as u32) as i32,
        })
        .collect::<Vec<_>>();
    let tokens = i32_tensor(
        device,
        &[rows as u64, 2],
        &token_values
            .iter()
            .flat_map(|token| [*token, 0])
            .collect::<Vec<_>>(),
    );
    let actual = kernel
        .call(draft_rows::Args {
            tokens: &tokens,
            directory: &i32_tensor(device, &[0, 1], &[]),
            table: &table,
            conditioning: &bf16_tensor(device, &[rows as u64, width as u64], &conditioning_values),
            embedding_norm: &bf16_tensor(device, &[width as u64], &embedding_norm_values),
            hidden_norm: &bf16_tensor(device, &[width as u64], &hidden_norm_values),
            combine: &combine,
            epsilon: 1.0e-6,
        })
        .unwrap()
        .value
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let (expected, magnitude) = reference(
        &token_values,
        &table_values,
        &conditioning_values,
        &embedding_norm_values,
        &hidden_norm_values,
        &combine_values,
        width,
        1.0e-6,
    );
    assert_eq!(actual.len(), expected.len());
    let failures = (0..expected.len())
        .filter(|&index| {
            let bound = magnitude[index] * 2f32.powi(-8) + 1.0e-5;
            !((actual[index] - expected[index]).abs() <= bound)
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "{:?} table {form:?} width {width} rows {rows}: {} of {} outside tolerance; first {:?}",
        device.backend(),
        failures.len(),
        expected.len(),
        failures
            .iter()
            .take(4)
            .map(|&index| (index, actual[index], expected[index]))
            .collect::<Vec<_>>()
    );
}
