use seismic_engine::weights::{descriptor::Stored, mlx::MlxArtifact};
use std::{
    fs,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "seismic-mlx-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn shard(&self, name: &str, header: &str, size: usize) {
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header.as_bytes());
        b.resize(b.len() + size, 0);
        fs::write(self.0.join(name), b).unwrap();
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
const CONFIG: &str = r#"{"quantization":{"group_size":64,"bits":4,"mode":"affine"}}"#;
const CODES: &str = r#"{"projection.weight":{"dtype":"U32","shape":[2,8],"data_offsets":[0,64]}}"#;
const COEFFICIENTS: &str = r#"{"projection.scales":{"dtype":"BF16","shape":[2,1],"data_offsets":[0,4]},"projection.biases":{"dtype":"BF16","shape":[2,1],"data_offsets":[4,8]},"norm":{"dtype":"F32","shape":[1,2],"data_offsets":[8,16]}}"#;
fn fixture() -> Temp {
    let t = Temp::new();
    fs::write(t.0.join("config.json"), CONFIG).unwrap();
    t.shard("z.safetensors", CODES, 64);
    t.shard("a.safetensors", COEFFICIENTS, 16);
    t
}
#[test]
fn identity_matches_python_reference_and_planes_outlive_artifact() {
    let t = fixture();
    let artifact = MlxArtifact::open(&t.0).unwrap();
    assert_eq!(
        artifact.identity().to_string(),
        "d998aa0bf0933aa829a85be24dd37e44f75027f23fe193e187790597a11e89d5"
    );
    let descriptor = artifact.descriptor("projection.weight", &[2, 64]).unwrap();
    let stored = artifact.stored(&descriptor).unwrap();
    artifact.descriptor("norm", &[2]).unwrap();
    assert!(artifact.descriptor("projection.weight", &[2, 63]).is_err());
    assert!(artifact.descriptor("projection.weight", &[1, 128]).is_err());
    assert!(artifact.descriptor("norm", &[3]).is_err());
    drop(artifact);
    let Stored::AffinePlanes {
        bits,
        group,
        codes,
        scales,
        biases,
    } = stored
    else {
        panic!("expected encoded planes")
    };
    assert_eq!((bits, group), (4, 64));
    assert_eq!(codes.read().unwrap(), vec![0; 64]);
    assert_eq!(scales.read().unwrap(), vec![0; 4]);
    assert_eq!(biases.read().unwrap(), vec![0; 4]);
}
#[test]
fn rejects_cross_shard_duplicates_and_wrong_coefficient_shape() {
    let t = fixture();
    t.shard("b.safetensors", CODES, 64);
    assert!(MlxArtifact::open(&t.0).is_err());
    fs::remove_file(t.0.join("b.safetensors")).unwrap();
    t.shard("a.safetensors", &COEFFICIENTS.replace("[2,1]", "[1,2]"), 16);
    let a = MlxArtifact::open(&t.0).unwrap();
    assert!(a.descriptor("projection.weight", &[2, 64]).is_err());
}
#[test]
fn requires_declared_uniform_quantization_and_nonempty_artifact() {
    let t = fixture();
    fs::write(t.0.join("config.json"), CONFIG.replace("64", "32")).unwrap();
    assert!(MlxArtifact::open(&t.0).is_err());
    fs::write(t.0.join("config.json"), CONFIG).unwrap();
    fs::remove_file(t.0.join("a.safetensors")).unwrap();
    fs::remove_file(t.0.join("z.safetensors")).unwrap();
    assert!(MlxArtifact::open(&t.0).is_err());
}
