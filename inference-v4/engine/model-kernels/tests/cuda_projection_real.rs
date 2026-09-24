//! CUDA K1 projections over real Qwen3.5-4B GGUF weights, against an f64
//! reference of the exact operands and a host emulation of each operand
//! path. Every mapping must agree with its own emulation to within output
//! roundings (a decode, scale or fragment-order defect shows up as a
//! departure that dwarfs them); the report also gives each path's error
//! against the exact product, relative to the output RMS. Run with
//! `MAGNITUDE_GGUF_4B` naming the pinned unsloth Q4_K_M file; returns early
//! without it or without CUDA.

mod cuda_common;

use cuda_common::*;
use magnitude_model_kernels::qwen_dense_expand;
use seismic::{Element, NativeSpecialization, Tensor};
use seismic_lang::registry::bf16_round;
use std::collections::HashMap;
use std::fs;

/// The tensors of a GGUF file: name -> (GGUF type id, dims, bytes).
struct Gguf {
    bytes: Vec<u8>,
    tensors: HashMap<String, (u32, Vec<u64>, usize)>,
    data: usize,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        slice
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn string(&mut self) -> String {
        let n = self.u64() as usize;
        String::from_utf8(self.take(n).to_vec()).unwrap()
    }
    /// Skip one metadata value of GGUF type `kind`, returning it when it is an
    /// unsigned integer.
    fn value(&mut self, kind: u32) -> Option<u64> {
        match kind {
            0 | 1 | 7 => Some(u64::from(self.take(1)[0])),
            2 | 3 => Some(u64::from(u16::from_le_bytes(self.take(2).try_into().unwrap()))),
            4 | 5 | 6 => Some(u64::from(self.u32())),
            8 => {
                self.string();
                None
            }
            9 => {
                let inner = self.u32();
                let n = self.u64();
                for _ in 0..n {
                    self.value(inner);
                }
                None
            }
            10 | 11 | 12 => Some(self.u64()),
            other => panic!("GGUF metadata type {other}"),
        }
    }
}

impl Gguf {
    fn open(path: &str) -> Self {
        let bytes = fs::read(path).unwrap();
        let mut cursor = Cursor { bytes: &bytes, at: 0 };
        assert_eq!(cursor.take(4), b"GGUF");
        cursor.u32();
        let count = cursor.u64();
        let metadata = cursor.u64();
        let mut alignment = 32u64;
        for _ in 0..metadata {
            let key = cursor.string();
            let kind = cursor.u32();
            let value = cursor.value(kind);
            if key == "general.alignment" {
                alignment = value.unwrap();
            }
        }
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let name = cursor.string();
            let dims = (0..cursor.u32()).map(|_| cursor.u64()).collect::<Vec<_>>();
            let kind = cursor.u32();
            let offset = cursor.u64() as usize;
            tensors.insert(name, (kind, dims, offset));
        }
        let data = (cursor.at as u64).div_ceil(alignment) * alignment;
        Self { data: data as usize, tensors, bytes }
    }

    /// The external format and raw bytes of rows [first, first + rows) of a
    /// quantized matrix (GGUF dims are [K, N]).
    fn rows(&self, name: &str, first: usize, rows: usize) -> (Format, usize, Vec<u8>) {
        let (kind, dims, offset) = &self.tensors[name];
        let format = match kind {
            12 => Format::Q4K,
            13 => Format::Q5K,
            14 => Format::Q6K,
            8 => Format::Q8,
            other => panic!("{name}: GGUF type {other}"),
        };
        let k = dims[0] as usize;
        let block_bytes = match format {
            Format::Q4K => 144,
            Format::Q5K => 176,
            Format::Q6K => 210,
            Format::Q8 => 34,
        };
        let row_bytes = k / format.block_values() * block_bytes;
        let start = self.data + offset + first * row_bytes;
        (format, k, self.bytes[start..start + rows * row_bytes].to_vec())
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        let (kind, dims, offset) = &self.tensors[name];
        assert_eq!(*kind, 0, "{name} is not F32");
        let n = dims.iter().product::<u64>() as usize;
        let start = self.data + offset;
        self.bytes[start..start + 4 * n].chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().unwrap())).collect()
    }
}

/// A resident mma16 weight from raw GGUF rows.
fn real_weight(device: &seismic::Device, format: Format, rows: usize, k: usize, external: &[u8]) -> Weight {
    let shape = [rows as u64, k as u64];
    let resident = format.resident();
    let bytes = resident.repack_host(Element::named(format.external()).unwrap(), &shape, external).unwrap();
    let values = resident.decode_host(&shape, &bytes).unwrap();
    let tensor = Tensor::from_host(device, resident, &shape, &bytes).unwrap();
    Weight { tensor, values, bytes }
}

/// Token-embedding rows as a residual, with `outliers` channels scaled up the
/// way massive activations are.
fn residual(gguf: &Gguf, m: usize, outliers: bool) -> Vec<f32> {
    let (format, k, external) = gguf.rows("token_embd.weight", 1000, m);
    let packet = Element::named(format.representation()).unwrap();
    let bytes = packet.repack_host(Element::named(format.external()).unwrap(), &[m as u64, k as u64], &external).unwrap();
    let mut values: Vec<f32> =
        packet.decode_host(&[m as u64, k as u64], &bytes).unwrap().into_iter().map(|v| v as f32 * 40.0).collect();
    if outliers {
        for row in 0..m {
            for channel in [7usize, 911, 1700, 2301] {
                values[row * k + channel] *= 60.0;
            }
        }
    }
    values
}

struct Error {
    rms: f64,
    max: f64,
}

/// |a - b| over the RMS of `reference`: RMS and max.
fn relative(a: &[f64], b: &[f64], reference: &[f64]) -> Error {
    let scale = (reference.iter().map(|v| v * v).sum::<f64>() / reference.len() as f64).sqrt();
    let rms = (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    Error { rms: rms / scale, max: max / scale }
}

fn mapping_label(mapping: Mapping, rows: usize) -> String {
    let path = if mapping.quantizes(rows) { "int8" } else { "16-bit" };
    format!("{path}{}", if rows > GEMV_ROWS && mapping.bm == 128 { " bm128" } else { "" })
}

#[test]
fn real_4b_projections_match_their_operand_emulation() {
    let Ok(path) = std::env::var("MAGNITUDE_GGUF_4B") else {
        eprintln!("MAGNITUDE_GGUF_4B unset; skipping");
        return;
    };
    let Some(device) = cuda() else { return };
    let gguf = Gguf::open(&path);
    let norm_values = gguf.f32s("blk.0.post_attention_norm.weight");
    let h = norm_values.len();
    // Real matrices as the gate and up streams: (gate, first row, up, first
    // row, rows).
    let pairs = [
        ("blk.0.ffn_gate.weight", 0usize, "blk.0.ffn_up.weight", 0usize, 512usize),
        ("blk.0.attn_qkv.weight", 512, "blk.0.attn_qkv.weight", 6144, 512),
        ("token_embd.weight", 20000, "token_embd.weight", 90000, 512),
        ("blk.0.ssm_alpha.weight", 0, "blk.0.ssm_beta.weight", 0, 32),
        ("blk.3.ffn_gate.weight", 4096, "blk.3.ffn_up.weight", 4096, 512),
    ];
    let mut failures = Vec::new();
    for (gate_name, gate_first, up_name, up_first, f) in pairs {
        let (gate_format, k, gate_bytes) = gguf.rows(gate_name, gate_first, f);
        let (up_format, _, up_bytes) = gguf.rows(up_name, up_first, f);
        assert_eq!(k, h);
        let gate = real_weight(&device, gate_format, f, k, &gate_bytes);
        let up = real_weight(&device, up_format, f, k, &up_bytes);
        for outliers in [false, true] {
            for o in [1usize, 3, 12, 64] {
                let values = residual(&gguf, o, outliers);
                let exact = normalized(&values, &norm_values, o, h);
                let residual_tensor = f32_tensor(&device, &[o as u64, h as u64], &values);
                let norm = f32_tensor(&device, &[h as u64], &norm_values);
                let out_rows = i32_tensor(&device, &[o as u64], &(0..o as i32).collect::<Vec<_>>());
                let (g_exact, _) = project(&exact, &gate.values, o, f, h);
                let (u_exact, _) = project(&exact, &up.values, o, f, h);
                let reference: Vec<f64> = (0..o * f).map(|i| silu(g_exact[i]) * u_exact[i]).collect();
                for &mapping in mappings(o) {
                    let (x, _) = operand_rows(&exact, h, mapping);
                    let (g, _) = project(&x, &gate.values, o, f, h);
                    let (u, _) = project(&x, &up.values, o, f, h);
                    // The operand path and the epilogue's bf16 roundings.
                    let round = |v: f64| f64::from(bf16_round(v as f32));
                    let emulated: Vec<f64> =
                        (0..o * f).map(|i| round(round(silu(round(g[i]))) * round(u[i]))).collect();
                    let kernel = qwen_dense_expand::native_for_device_with(
                        &device,
                        qwen_dense_expand::Elements {
                            NW: Element::f32(),
                            GW: gate_format.resident(),
                            UW: up_format.resident(),
                            A: Element::bf16(),
                        },
                        &mapping.params(NativeSpecialization::new().with_static("H", h as u64).with_static("F", f as u64), false),
                    )
                    .unwrap();
                    let actual: Vec<f64> = read_bf16(
                        &kernel
                            .call(qwen_dense_expand::Args {
                                residual: &residual_tensor,
                                norm: &norm,
                                gate_weight: &gate.tensor,
                                up_weight: &up.tensor,
                                out_rows: &out_rows,
                                eps: EPSILON,
                            })
                            .unwrap()
                            .value,
                    )
                    .into_iter()
                    .map(f64::from)
                    .collect();
                    let device_error = relative(&actual, &reference, &reference);
                    let path_error = relative(&emulated, &reference, &reference);
                    let departure = relative(&actual, &emulated, &reference);
                    println!(
                        "{gate_name} {gate_format:?} outliers={outliers} O={o} {}: device vs exact rms {:.2e} max {:.2e}; \
                         emulation vs exact rms {:.2e} max {:.2e}; device vs emulation rms {:.2e} max {:.2e}",
                        mapping_label(mapping, o),
                        device_error.rms,
                        device_error.max,
                        path_error.rms,
                        path_error.max,
                        departure.rms,
                        departure.max
                    );
                    // Accumulation order across bf16 rounding boundaries, the
                    // rare activation-rounding tie that moves a q8_1 group, and
                    // on the 16-bit GEMM the weights' dequantization to bf16
                    // (2^-9 of each weight, not emulated; 1.9-5.0e-3 of the
                    // output RMS on these rows).
                    let limit = if o > GEMV_ROWS && mapping.int8 == 0 { 8e-3 } else { 1e-3 };
                    if departure.rms > limit {
                        failures.push(format!("{gate_name} outliers={outliers} O={o} {}", mapping_label(mapping, o)));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "paths departing from their emulation: {failures:?}");
}
