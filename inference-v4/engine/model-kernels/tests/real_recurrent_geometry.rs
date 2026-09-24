//! One actual Qwen3.5-4B block-zero recurrent path, independent CPU GGUF fixture.
//! Run only with MAGNITUDE_REAL_RECURRENT_FIXTURE set to its generated directory.

use magnitude_model_kernels::{
    qwen_recurrent_output, qwen_recurrent_project, qwen_recurrent_step, repack_weight,
};
use seismic::{BackendName, Device, DeviceCatalog, Element, Tensor};
use std::{fs, path::Path};

fn bytes(path: &Path, name: &str, extension: &str) -> Vec<u8> {
    fs::read(path.join(format!("{name}.{extension}"))).unwrap()
}

fn expected(path: &Path, name: &str) -> Vec<f32> {
    bytes(path, name, "f32")
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn f32_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    Tensor::from_host(
        device,
        Element::f32(),
        shape,
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn bf16_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    Tensor::from_host(
        device,
        Element::bf16(),
        shape,
        &values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn repacked(
    device: &Device,
    path: &Path,
    name: &str,
    source: &str,
    resident: &str,
    shape: &[u64],
) -> Tensor {
    let source_element = Element::named(source).unwrap();
    let resident_element = Element::named(resident).unwrap();
    let wire = bytes(path, name, "wire");
    let logical = shape.iter().product::<u64>();
    let input = Tensor::from_host(device, source_element, &[1, 1, logical], &wire).unwrap();
    repack_weight::native_for_device_with(
        device,
        repack_weight::Elements {
            E: source_element,
            U: resident_element,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(repack_weight::Args { source: &input })
    .unwrap()
    .value
    .reshape(shape)
    .unwrap()
}

fn read_f32(actual: &Tensor) -> Vec<f32> {
    actual.read_to_host().unwrap().chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap())).collect()
}

fn read_bf16(actual: &Tensor) -> Vec<f32> {
    actual.read_to_host().unwrap().chunks_exact(2)
        .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(word.try_into().unwrap())) << 16)).collect()
}

fn report(path: &Path, label: &str, actual: Vec<f32>) {
    let current = expected(path, &format!("{label}_v4"));
    let reference = expected(path, &format!("{label}_v3"));
    assert_eq!(actual.len(), current.len(), "{label} shape");
    assert_eq!(actual.len(), reference.len(), "{label} reference shape");
    for (name, target) in [("current", current), ("reference", reference)] {
        let max = actual
            .iter()
            .zip(&target)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rms = (actual
            .iter()
            .zip(&target)
            .map(|(a, b)| (a - b).powi(2) as f64)
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let different = actual
            .iter()
            .zip(&target)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "{label} vs {name}: n={} max_abs={max:.8} rms={rms:.8} bits_different={different}",
            actual.len()
        );
        if name == "reference" {
            // The independent CPU fixture uses ordered GGUF dequantization and
            // BF16 publication. Metal's parallel reductions can cross a BF16
            // rounding boundary; the oracle bounds that effect at each stage.
            let (max_limit, rms_limit) = match label {
                "projection" => (0.0625, 0.0010),
                "mixed" => (0.00025, 0.000005),
                "delta" => (0.003, 0.00001),
                "output" => (0.001, 0.00015),
                _ => panic!("unexpected recurrent stage {label}"),
            };
            assert!(max <= max_limit && rms <= rms_limit,
                "{label} departs from CPU BF16 reference: max={max}, rms={rms}");
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
fn actual_4b_recurrent_stage_boundaries_vs_cpu_gguf() {
    let Ok(path) = std::env::var("MAGNITUDE_REAL_RECURRENT_FIXTURE") else {
        eprintln!("real GGUF fixture absent; set MAGNITUDE_REAL_RECURRENT_FIXTURE to run oracle");
        return;
    };
    let path = Path::new(&path);
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let f32e = Element::f32();
    let bf16 = Element::bf16();
    let q5 = Element::named("q5k").unwrap();
    let q4 = Element::named("q4k").unwrap();
    let q8 = Element::named("q8g32s").unwrap();
    let hidden = f32_tensor(&device, &[1, 2560], &expected(path, "hidden"));
    let norm = f32_tensor(&device, &[2560], &expected(path, "norm"));
    let qkv = repacked(&device, path, "qkv", "gguf_q5_k", "q5k", &[8192, 2560]);
    let gate = repacked(&device, path, "gate", "gguf_q4_k", "q4k", &[4096, 2560]);
    let alpha = repacked(&device, path, "alpha", "gguf_q8_0", "q8g32s", &[32, 2560]);
    let beta = repacked(&device, path, "beta", "gguf_q8_0", "q8g32s", &[32, 2560]);
    let output = repacked(&device, path, "output", "gguf_q5_k", "q5k", &[2560, 4096]);
    let convolution = f32_tensor(&device, &[8192, 4], &expected(path, "convolution"));
    let rate = f32_tensor(&device, &[32], &expected(path, "rate"));
    let time_bias = f32_tensor(&device, &[32], &expected(path, "time_bias"));
    let recurrent_norm = f32_tensor(&device, &[128], &expected(path, "recurrent_norm"));
    let indices = |shape: &[u64], values: &[i32]| {
        Tensor::from_host(
            &device,
            Element::i32(),
            shape,
            &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let segments = indices(&[2, 2], &[0, 1, 1, 1]);
    // Bank 0 is the zero seed the advance reads; bank 1 is its successor.
    let mut window = bf16_tensor(&device, &[2, 3, 8192], &vec![0.0; 2 * 3 * 8192]);
    let mut delta = f32_tensor(&device, &[2, 32, 128, 128], &vec![0.0; 2 * 32 * 128 * 128]);

    let projection = qwen_recurrent_project::native_for_device_with(
        &device,
        qwen_recurrent_project::Elements {
            NW: f32e,
            QW: q5,
            GW: q4,
            AW: q8,
            BW: q8,
            A: bf16,
        },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_project::Args {
        hidden: &hidden,
        input_norm: &norm,
        qkv_weight: &qkv,
        gate_weight: &gate,
        alpha_weight: &alpha,
        beta_weight: &beta,
        epsilon: 1e-6,
    })
    .unwrap()
    .value;
    report(path, "projection", read_bf16(&projection));

    let mixed = qwen_recurrent_step::native_for_device_with(
        &device,
        qwen_recurrent_step::Elements { A: bf16 },
        &seismic::NativeSpecialization::new()
            .with_static("NK", 16)
            .with_static("NV", 32)
            .with_static("W", 128)
            .with_static("C", 4)
            .with_param("ROWS", 8),
    )
    .unwrap()
    .call(qwen_recurrent_step::Args {
        projection: &projection,
        convolution: &convolution,
        rate: &rate,
        time_bias: &time_bias,
        segments: &segments,
        stop: &indices(&[1], &[1]),
        previous_bank: &indices(&[1], &[0]),
        following_bank: &indices(&[1], &[1]),
        window: &mut window,
        delta: &mut delta,
        norm_epsilon: 128e-6,
        grouped: false,
    })
    .unwrap()
    .value;
    report(path, "mixed", read_bf16(&mixed));
    report(path, "delta", read_f32(&delta.slice_leading(1, 2).unwrap()));
    let projected = qwen_recurrent_output::native_for_device_with(
        &device,
        qwen_recurrent_output::Elements {
            RN: f32e,
            OW: q5,
            A: bf16,
        },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_output::Args {
        hidden: &hidden,
        mixed: &mixed,
        projection: &projection,
        recurrent_norm: &recurrent_norm,
        output_weight: &output,
        epsilon: 1e-6,
    })
    .unwrap()
    .value;
    report(path, "output", read_f32(&projected));
}
