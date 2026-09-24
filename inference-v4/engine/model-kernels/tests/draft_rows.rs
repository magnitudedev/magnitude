//! `qwen_draft_rows` on every accelerator the host has, over its resident
//! layout (Metal `rows16`, CUDA `mma16`): the embedding table and the combine
//! weight are GGUF Q8_0 packets repacked on the device, and the result is
//! checked against a host model of the entry's contract.

use magnitude_model_kernels::{qwen_draft_rows, repack_weight};
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
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn code(&mut self) -> i8 {
        (self.next() % 255) as i8
    }
    fn symmetric(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }
}

/// The resident q8 element of the device's backend.
fn resident_q8(device: &Device) -> Element {
    let layout = match device.backend() {
        seismic::BackendName::Cuda => Layout::Mma16,
        _ => Layout::Rows16,
    };
    Element::stored("q8g32s", layout).unwrap()
}

/// A `[rows, columns]` GGUF Q8_0 weight repacked to the resident q8 layout,
/// and its logical values.
fn q8_weight(device: &Device, rows: usize, columns: usize, scale: f32, random: &mut Random) -> (Tensor, Vec<f32>) {
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
        &NativeSpecialization::new(),
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
        joined.extend(normalized(&conditioning[row * width..][..width], hidden_norm));
        for output in 0..width {
            let weights = &combine[output * 2 * width..][..2 * width];
            result.push(weights.iter().zip(&joined).map(|(w, x)| w * x).sum());
            magnitude.push(weights.iter().zip(&joined).map(|(w, x)| (w * x).abs()).sum());
        }
    }
    (result, magnitude)
}

#[test]
fn draft_rows_match_the_host_model_over_resident_weights() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    for backend in [seismic::BackendName::Metal, seismic::BackendName::Cuda] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        draft_rows_match_on(&device);
    }
}

fn draft_rows_match_on(device: &Device) {
    let kernel = qwen_draft_rows::native_for_device_with(
        &device,
        qwen_draft_rows::Elements {
            EW: resident_q8(&device),
            A: Element::bf16(),
            EN: Element::bf16(),
            HN: Element::bf16(),
            CW: resident_q8(&device),
        },
        &NativeSpecialization::new(),
    )
    .unwrap();
    let epsilon = 1.0e-6;
    // Widths of whole 64-column k-blocks, below and past one block's 32
    // outputs, rows 1..3.
    for (width, vocabulary, rows) in [(64, 5, 1), (128, 7, 3), (192, 4, 2)] {
        let mut random = Random(width as u64 * 31 + rows as u64);
        let (table, table_values) = q8_weight(&device, vocabulary, width, 0.03125, &mut random);
        let (combine, combine_values) = q8_weight(&device, width, 2 * width, 0.0078125, &mut random);
        let embedding_norm_values = (0..width)
            .map(|_| bf16_round(1.0 + 0.25 * random.symmetric()))
            .collect::<Vec<_>>();
        let hidden_norm_values = (0..width)
            .map(|_| bf16_round(1.0 + 0.25 * random.symmetric()))
            .collect::<Vec<_>>();
        let conditioning_values = (0..rows * width)
            .map(|_| bf16_round(2.0 * random.symmetric()))
            .collect::<Vec<_>>();
        let token_values = (0..rows)
            .map(|_| (random.next() % vocabulary as u32) as i32)
            .collect::<Vec<_>>();
        let token_bytes = token_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tokens = Tensor::from_host(&device, Element::i32(), &[rows as u64], &token_bytes).unwrap();
        let conditioning = bf16_tensor(&device, &[rows as u64, width as u64], &conditioning_values);
        let embedding_norm = bf16_tensor(&device, &[width as u64], &embedding_norm_values);
        let hidden_norm = bf16_tensor(&device, &[width as u64], &hidden_norm_values);
        let result = kernel
            .call(qwen_draft_rows::Args {
                tokens: &tokens,
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
    }
}
