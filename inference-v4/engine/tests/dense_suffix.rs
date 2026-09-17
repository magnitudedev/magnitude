use seismic_lang::{
    lower::Options,
    plan::plan_specialized,
    types::{DType, Elem},
};
use seismic_runtime::{
    plan::{Bindings, CompiledPlan},
    Buffer, Candidate, Device,
};
use std::collections::HashMap;
struct Bound(HashMap<String, Buffer>);
impl Bindings for Bound {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        assert!(plane.is_empty());
        self.0.get(root)
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        (name == "eps").then_some(1e-6)
    }
}
// Arithmetic rounding reference, independent of the compiler's bit conversions.
fn rounded(x: f32, dtype: DType) -> f32 {
    if x == 0.0 {
        return x;
    }
    let precision = if dtype == DType::BF16 { 7 } else { 10 };
    let minimum = if dtype == DType::BF16 { -126 } else { -14 };
    let exponent = (f64::from(x.abs()).log2().floor() as i32).max(minimum);
    let step = 2f64.powi(exponent - precision);
    ((f64::from(x) / step).round_ties_even() * step) as f32
}
fn encode(values: &[f32], dtype: DType) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&x| {
            let x = rounded(x, dtype);
            let bits = if dtype == DType::BF16 {
                (x.to_bits() >> 16) as u16
            } else if x == 0.0 {
                ((x.to_bits() >> 16) & 0x8000) as u16
            } else {
                let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
                let exponent = ((x.to_bits() >> 23) & 255) as i32 - 127;
                if exponent < -14 {
                    sign | (x.abs() * 2f32.powi(24)) as u16
                } else {
                    sign | (((exponent + 15) as u16) << 10) | ((x.to_bits() >> 13) as u16 & 1023)
                }
            };
            bits.to_le_bytes()
        })
        .collect()
}
fn floats(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn exercise(device: Device, candidate: Candidate) {
    let device = std::rc::Rc::new(device);
    exercise_packed(&device, candidate.clone());
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    let (m, h, f) = (2usize, 8usize, 12usize);
    let residual: Vec<f32> = (0..m * h)
        .map(|i| (i as f32 * 0.391 + 0.13).sin() * 3.0)
        .collect();
    let norm: Vec<f32> = (0..h).map(|i| 0.9 + i as f32 * 0.027).collect();
    let gate: Vec<f32> = (0..f * h)
        .map(|i| ((i as f32 * 0.17).sin()) * 0.2)
        .collect();
    let up: Vec<f32> = (0..f * h)
        .map(|i| ((i as f32 * 0.23).cos()) * 0.3)
        .collect();
    let down: Vec<f32> = (0..h * f)
        .map(|i| ((i as f32 * 0.31).sin()) * 0.1)
        .collect();
    for dtype in [DType::BF16, DType::F16] {
        let elements = HashMap::from([
            ("A".into(), Elem::Dtype(dtype)),
            ("NW".into(), Elem::Dtype(DType::F32)),
            ("GW".into(), Elem::Dtype(DType::F32)),
            ("UW".into(), Elem::Dtype(DType::F32)),
            ("DW".into(), Elem::Dtype(DType::F32)),
        ]);
        let shapes = HashMap::from([
            ("M".into(), m as i64),
            ("H".into(), h as i64),
            ("F".into(), f as i64),
        ]);
        assert!(plan_specialized(&program, "qwen_dense_suffix", &shapes, &HashMap::new()).is_err());
        let plan = plan_specialized(&program, "qwen_dense_suffix", &shapes, &elements).unwrap();
        assert_eq!(plan.steps.len(), 7);
        let account = seismic_accounting::derive_specialized(
            &program,
            "qwen_dense_suffix",
            &shapes,
            &elements,
            &seismic_accounting::memory::Bindings::new(),
            1_000_000,
        )
        .unwrap();
        assert!(
            account.memory.is_exact(),
            "{:?}",
            account.memory.unavailable
        );
        assert!(account.work.is_exact(), "{:?}", account.work);
        for (name, reads, writes) in [
            ("residual", m * h * 4, 0),
            ("normalized", m * h * 2, m * h * 2),
            ("activated", m * f * 2, m * f * 2),
            ("out", 0, m * h * 4),
        ] {
            let access = account
                .memory
                .accesses
                .iter()
                .find(|(id, _)| *id == name)
                .unwrap()
                .1;
            assert_eq!(
                (access.reads.bytes(), access.writes.bytes()),
                (reads as u64, writes as u64)
            );
        }

        let mut compiled = CompiledPlan::compile(
            &device,
            &program,
            &plan,
            &Options::default(),
            candidate.clone(),
        )
        .unwrap();
        let mut bindings = Bound(HashMap::new());
        for (name, values) in [
            ("residual", &residual),
            ("norm", &norm),
            ("gate_weight", &gate),
            ("up_weight", &up),
            ("down_weight", &down),
        ] {
            bindings
                .0
                .insert(name.into(), device.buffer_from(&floats(values)).unwrap());
        }
        for (name, count) in [
            ("normalized", m * h),
            ("gate", m * f),
            ("up", m * f),
            ("activated", m * f),
            ("product", m * f),
            ("projected", m * h),
        ] {
            bindings
                .0
                .insert(name.into(), device.buffer(count * 2).unwrap());
        }
        bindings
            .0
            .insert("out".into(), device.buffer(m * h * 4).unwrap());
        compiled.execute(&bindings).unwrap();
        let mut normalized = vec![];
        for row in residual.chunks_exact(h) {
            let sum: f32 = row.iter().map(|x| x * x).sum();
            let inv = (sum / h as f32 + 1e-6).sqrt().recip();
            normalized.extend(
                row.iter()
                    .zip(&norm)
                    .map(|(x, w)| rounded(x * inv * w, dtype)),
            );
        }
        let linear = |x: &[f32], w: &[f32], k: usize| -> Vec<f32> {
            x.chunks_exact(k)
                .flat_map(|row| {
                    w.chunks_exact(k).map(|wr| {
                        rounded(
                            row.iter()
                                .zip(wr)
                                .fold(0f32, |acc, (x, w)| x.mul_add(*w, acc)),
                            dtype,
                        )
                    })
                })
                .collect()
        };
        let g = linear(&normalized, &gate, h);
        let u = linear(&normalized, &up, h);
        let activated: Vec<f32> = g
            .iter()
            .map(|x| rounded(x / (1.0 + (-x).exp()), dtype))
            .collect();
        let product: Vec<f32> = activated
            .iter()
            .zip(&u)
            .map(|(a, b)| rounded(a * b, dtype))
            .collect();
        let projected = linear(&product, &down, f);
        for (name, expected) in [
            ("normalized", normalized),
            ("gate", g),
            ("up", u),
            ("activated", activated),
            ("product", product),
            ("projected", projected.clone()),
        ] {
            let mut actual = vec![0; expected.len() * 2];
            bindings.0[name].read(&mut actual).unwrap();
            assert_eq!(actual, encode(&expected, dtype), "{name} {dtype:?}");
        }
        let expected: Vec<f32> = residual
            .iter()
            .zip(&projected)
            .map(|(a, b)| a + b)
            .collect();
        let mut actual = vec![0; expected.len() * 4];
        bindings.0["out"].read(&mut actual).unwrap();
        assert_eq!(actual, floats(&expected), "F32 residual {dtype:?}");
        // All intermediate publications repeat correctly with retained native code.
        compiled.execute(&bindings).unwrap();
        bindings.0["out"].read(&mut actual).unwrap();
        assert_eq!(actual, floats(&expected));
        // The engine resource owner imports real stored tensors and retains the
        // same native composition across invocations, including in-place residuals.
        use seismic_engine::{
            models::qwen35::dense::{DenseInvocation, DenseSuffix, DenseWeights},
            weights::{
                descriptor::{Stored, StoredTensor, Transform, WeightDescriptor},
                residency::Importer,
                source::FileSource,
            },
        };
        let fixture =
            Fixture::new(&[floats(&norm), floats(&gate), floats(&up), floats(&down)].concat());
        let source = std::sync::Arc::new(FileSource::open(&fixture.0).unwrap());
        let mut importer = Importer::new(device.clone(), candidate.clone()).unwrap();
        let mut offset = 0u64;
        let mut import = |name: &str, shape: Vec<u64>| {
            let nbytes = shape.iter().product::<u64>() * 4;
            let stored = Stored::Dense(StoredTensor {
                source: source.clone(),
                offset,
                nbytes,
                dtype: DType::F32,
                shape: shape.clone(),
            });
            offset += nbytes;
            importer
                .import(
                    &WeightDescriptor {
                        name: name.into(),
                        shape,
                        transform: Transform::Identity,
                    },
                    &stored,
                    DType::F32,
                )
                .unwrap()
        };
        let weights = DenseWeights {
            norm: import("norm", vec![h as u64]),
            gate: import("gate", vec![f as u64, h as u64]),
            up: import("up", vec![f as u64, h as u64]),
            down: import("down", vec![h as u64, f as u64]),
        };
        let mut owned = DenseSuffix::compile(
            &device,
            &program,
            DenseInvocation {
                rows: m,
                activation: dtype,
                epsilon: 1e-6,
            },
            weights,
            candidate.clone(),
            &Options::default(),
        )
        .unwrap();
        drop(importer);
        owned
            .execute(&bindings.0["residual"], &bindings.0["out"])
            .unwrap();
        bindings.0["out"].read(&mut actual).unwrap();
        assert_eq!(actual, floats(&expected));
        owned
            .execute(&bindings.0["residual"], &bindings.0["residual"])
            .unwrap();
        bindings.0["residual"].read(&mut actual).unwrap();
        assert_eq!(actual, floats(&expected));
    }
}
#[test]
fn cpu_dense_suffix() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_dense_suffix() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_dense_suffix() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "seismic-dense-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn exercise_packed(device: &std::rc::Rc<Device>, candidate: Candidate) {
    use seismic_engine::{
        models::qwen35::dense::{DenseInvocation, DenseSuffix, DenseWeights},
        weights::{
            descriptor::{Stored, StoredTensor, Transform, WeightDescriptor},
            residency::Importer,
            source::FileSource,
        },
    };
    let mut bytes = vec![];
    bytes.extend((0..64).flat_map(|_| 0x3f80u16.to_le_bytes()));
    for code in [1u8, 2, 1] {
        bytes.extend(vec![code | code << 4; 2048]);
        bytes.extend((0..64).flat_map(|_| 0x3c80u16.to_le_bytes()));
        bytes.extend([0; 128]);
    }
    let fixture = Fixture::new(&bytes);
    let source = std::sync::Arc::new(FileSource::open(&fixture.0).unwrap());
    let tensor = |dtype, shape, offset, nbytes| StoredTensor {
        source: source.clone(),
        dtype,
        shape,
        offset,
        nbytes,
    };
    let mut importer = Importer::new(device.clone(), candidate.clone()).unwrap();
    let descriptor = |name: &str, shape| WeightDescriptor {
        name: name.into(),
        shape,
        transform: Transform::Identity,
    };
    let norm = importer
        .import(
            &descriptor("norm", vec![64]),
            &Stored::Dense(tensor(DType::BF16, vec![64], 0, 128)),
            DType::BF16,
        )
        .unwrap();
    let mut offset = 128;
    let mut packed = |name| {
        let stored = Stored::AffinePlanes {
            bits: 4,
            group: 64,
            codes: tensor(DType::U32, vec![64, 8], offset, 2048),
            scales: tensor(DType::BF16, vec![64, 1], offset + 2048, 128),
            biases: tensor(DType::BF16, vec![64, 1], offset + 2176, 128),
        };
        offset += 2304;
        importer
            .import(&descriptor(name, vec![64, 64]), &stored, DType::BF16)
            .unwrap()
    };
    let weights = DenseWeights {
        norm,
        gate: packed("gate"),
        up: packed("up"),
        down: packed("down"),
    };
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    let mut suffix = DenseSuffix::compile(
        device,
        &program,
        DenseInvocation {
            rows: 1,
            activation: DType::BF16,
            epsilon: 1e-6,
        },
        weights,
        candidate,
        &Options::default(),
    )
    .unwrap();
    let residual = device.buffer_from(&floats(&[1.; 64])).unwrap();
    let out = device.buffer(256).unwrap();
    suffix.execute(&residual, &out).unwrap();
    let mut actual = vec![0; 256];
    out.read(&mut actual).unwrap();
    let activated = rounded(1.0 / (1.0 + (-1f32).exp()), DType::BF16);
    let expected = 1.0 + rounded(activated * 2.0, DType::BF16);
    assert_eq!(
        actual,
        floats(&[expected; 64]),
        "packed weights retain explicit publications"
    );
}
