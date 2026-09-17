use seismic_engine::weights::{
    descriptor::{Stored, StoredTensor, Transform, WeightDescriptor},
    residency::Importer,
    source::FileSource,
};
use seismic_lang::types::{DType, Elem};
use seismic_runtime::{Candidate, Device};
use std::{
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "seismic-resident-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
    fn tensor(&self, dtype: DType, shape: Vec<u64>, offset: u64, nbytes: u64) -> StoredTensor {
        StoredTensor {
            source: Arc::new(FileSource::open(&self.0).unwrap()),
            offset,
            nbytes,
            dtype,
            shape,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn exercise(device: Device, candidate: Candidate) {
    let device = Rc::new(device);
    let mut importer = Importer::new(device.clone(), candidate).unwrap();
    let fixture = Fixture::new(
        &[0x3c00u16, 0xc000, 0x3800, 0x4400]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let tensor = fixture.tensor(DType::F16, vec![1, 4], 0, 8);
    let mut descriptor = WeightDescriptor {
        name: "test".into(),
        shape: vec![4],
        transform: Transform::Identity,
    };
    let stored = Stored::Dense(tensor.clone());
    let resident = importer.import(&descriptor, &stored, DType::BF16).unwrap();
    assert_eq!(resident.element(), &Elem::Dtype(DType::BF16));
    let mut actual = [0; 8];
    resident.plane("").unwrap().read(&mut actual).unwrap();
    assert_eq!(
        &actual,
        [0x3f80u16, 0xc000, 0x3f00, 0x4080]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
            .as_slice()
    );
    descriptor.transform = Transform::NegativeExp;
    let transformed = importer.import(&descriptor, &stored, DType::F32).unwrap();
    let mut actual = [0; 16];
    transformed.plane("").unwrap().read(&mut actual).unwrap();
    for (bytes, input) in actual.chunks_exact(4).zip([1f32, -2., 0.5, 4.]) {
        let got = f32::from_le_bytes(bytes.try_into().unwrap());
        let expected = -(f64::from(input).exp() as f32);
        assert!(
            (got - expected).abs() <= expected.abs() * 3e-7,
            "{got} != {expected}"
        );
    }
    let malformed = Stored::Dense(StoredTensor {
        nbytes: 4,
        ..tensor
    });
    assert!(importer
        .import(&descriptor, &malformed, DType::F32)
        .is_err());
    // Canonical affine storage is transferred byte-for-byte without host decoding.
    let mut bytes = vec![0x21; 32];
    bytes.extend(0x3f80u16.to_le_bytes());
    bytes.extend(0xbf80u16.to_le_bytes());
    let packed = Fixture::new(&bytes);
    let stored = Stored::AffinePlanes {
        bits: 4,
        group: 64,
        codes: packed.tensor(DType::U32, vec![1, 8], 0, 32),
        scales: packed.tensor(DType::BF16, vec![1, 1], 32, 2),
        biases: packed.tensor(DType::BF16, vec![1, 1], 34, 2),
    };
    let descriptor = WeightDescriptor {
        name: "packed".into(),
        shape: vec![1, 64],
        transform: Transform::Identity,
    };
    let packed = importer.import(&descriptor, &stored, DType::BF16).unwrap();
    assert_eq!(packed.element(), &Elem::Repr("q4g64".into()));
    drop(importer);
    drop(device);
    drop(stored);
    for (plane, expected) in [
        ("words", &bytes[..32]),
        ("scale", &bytes[32..34]),
        ("bias", &bytes[34..]),
    ] {
        let mut got = vec![0; expected.len()];
        packed.plane(plane).unwrap().read(&mut got).unwrap();
        assert_eq!(&got, expected);
    }
}
#[test]
fn cpu_weight_import() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_weight_import() {
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
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_weight_import() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
