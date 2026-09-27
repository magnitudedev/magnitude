//! Persistence of a measured basis. The caller names the directory; each
//! basis is one JSON file whose name is the content address of its identity
//! and the backend's declared plan, so a change to either is a new file.
//! Elements are stored by name. An absent, unreadable or unparseable file,
//! an unknown class or element name, or an identity or protocol mismatch is
//! a cache miss.

use super::basis::{
    BasisIdentity, ClassCost, ClassMeasurement, CostModel, MeasuredPoint, MeasurementBasis,
    MeasurementKey, OperationClass, MEASUREMENT_PROTOCOL_VERSION,
};
use super::plan::measurement_plan;
use crate::StreamingCost;
use seismic::{BackendName, Element};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The cache file name of a basis with `identity`, or `None` when the
/// identity names no backend this build knows.
pub fn basis_file_name(identity: &BasisIdentity) -> Option<String> {
    let backend = BackendName::parse(&identity.backend)?;
    let mut material = format!(
        "engine {}\nbackend {}\ndevice {}\nprotocol {}\nplan",
        identity.engine_build, identity.backend, identity.device, identity.protocol_version
    );
    for key in measurement_plan(backend) {
        material.push('\n');
        material.push_str(&key_json(&key).to_string());
    }
    let digest = Sha256::digest(material.as_bytes());
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Some(format!("basis-{hex}.json"))
}

fn key_json(key: &MeasurementKey) -> Value {
    json!({
        "class": key.class.name(),
        "bindings": key.bindings.iter().map(|element| element.name()).collect::<Vec<_>>(),
        "geometry": key.geometry.iter().map(|(name, value)| json!([name, value])).collect::<Vec<_>>(),
    })
}

fn identity_json(identity: &BasisIdentity) -> Value {
    json!({
        "engine_build": identity.engine_build,
        "backend": identity.backend,
        "device": identity.device,
        "protocol_version": identity.protocol_version,
    })
}

fn measurement_json(measurement: &ClassMeasurement) -> Value {
    match measurement {
        ClassMeasurement::Unsupported { reason } => json!({ "unsupported": reason }),
        ClassMeasurement::Measured { points, cost } => json!({
            "points": points
                .iter()
                .map(|point| json!({ "bytes": point.bytes, "samples": point.samples }))
                .collect::<Vec<_>>(),
            "cost": match &cost.model {
                CostModel::PerLaunch { seconds } => json!({ "per_launch_seconds": seconds }),
                CostModel::Linear(linear) => json!({
                    "launch_seconds": linear.launch_seconds,
                    "seconds_per_byte": linear.seconds_per_byte,
                }),
                CostModel::Curve(points) => json!({
                    "launch_curve": points
                        .iter()
                        .map(|(bytes, seconds)| json!([bytes, seconds]))
                        .collect::<Vec<_>>(),
                }),
            },
            "slow_factor": cost.slow_factor,
            "fast_factor": cost.fast_factor,
        }),
    }
}

/// The JSON document of `basis`.
pub fn basis_json(basis: &MeasurementBasis) -> Value {
    json!({
        "identity": identity_json(&basis.identity),
        "classes": basis
            .classes
            .iter()
            .map(|(key, measurement)| json!({
                "key": key_json(key),
                "measurement": measurement_json(measurement),
            }))
            .collect::<Vec<_>>(),
    })
}

fn parse_key(value: &Value) -> Option<MeasurementKey> {
    let class = OperationClass::named(value.get("class")?.as_str()?)?;
    let bindings = value
        .get("bindings")?
        .as_array()?
        .iter()
        .map(|name| Element::named(name.as_str()?))
        .collect::<Option<Vec<_>>>()?;
    let geometry = value
        .get("geometry")?
        .as_array()?
        .iter()
        .map(|entry| {
            let [name, value] = entry.as_array()?.as_slice() else {
                return None;
            };
            Some((name.as_str()?.to_owned(), value.as_u64()?))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(MeasurementKey {
        class,
        bindings,
        geometry,
    })
}

fn parse_measurement(value: &Value) -> Option<ClassMeasurement> {
    if let Some(reason) = value.get("unsupported") {
        return Some(ClassMeasurement::Unsupported {
            reason: reason.as_str()?.to_owned(),
        });
    }
    let points = value
        .get("points")?
        .as_array()?
        .iter()
        .map(|point| {
            Some(MeasuredPoint {
                bytes: point.get("bytes")?.as_u64()?,
                samples: point
                    .get("samples")?
                    .as_array()?
                    .iter()
                    .map(Value::as_f64)
                    .collect::<Option<Vec<_>>>()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let cost = value.get("cost")?;
    let model = if let Some(seconds) = cost.get("per_launch_seconds") {
        CostModel::PerLaunch {
            seconds: seconds.as_f64()?,
        }
    } else if let Some(curve) = cost.get("launch_curve") {
        CostModel::Curve(
            curve
                .as_array()?
                .iter()
                .map(|entry| {
                    let [bytes, seconds] = entry.as_array()?.as_slice() else {
                        return None;
                    };
                    Some((bytes.as_u64()?, seconds.as_f64()?))
                })
                .collect::<Option<Vec<_>>>()?,
        )
    } else {
        CostModel::Linear(StreamingCost {
            launch_seconds: cost.get("launch_seconds")?.as_f64()?,
            seconds_per_byte: cost.get("seconds_per_byte")?.as_f64()?,
        })
    };
    Some(ClassMeasurement::Measured {
        points,
        cost: ClassCost {
            model,
            slow_factor: value.get("slow_factor")?.as_f64()?,
            fast_factor: value.get("fast_factor")?.as_f64()?,
        },
    })
}

/// The basis in `document`, when it is well formed and has `identity`.
pub fn parse_basis(document: &Value, identity: &BasisIdentity) -> Option<MeasurementBasis> {
    let stored = document.get("identity")?;
    let parsed = BasisIdentity {
        engine_build: stored.get("engine_build")?.as_str()?.to_owned(),
        backend: stored.get("backend")?.as_str()?.to_owned(),
        device: stored.get("device")?.as_str()?.to_owned(),
        protocol_version: u32::try_from(stored.get("protocol_version")?.as_u64()?).ok()?,
    };
    if parsed != *identity || parsed.protocol_version != MEASUREMENT_PROTOCOL_VERSION {
        return None;
    }
    let classes = document
        .get("classes")?
        .as_array()?
        .iter()
        .map(|entry| {
            Some((
                parse_key(entry.get("key")?)?,
                parse_measurement(entry.get("measurement")?)?,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(MeasurementBasis {
        identity: parsed,
        classes,
    })
}

/// The cached basis of `identity` in `dir`; `None` on any miss.
pub fn load_basis(dir: &Path, identity: &BasisIdentity) -> Option<MeasurementBasis> {
    let path = dir.join(basis_file_name(identity)?);
    let bytes = std::fs::read(path).ok()?;
    let document = serde_json::from_slice::<Value>(&bytes).ok()?;
    parse_basis(&document, identity)
}

/// Distinguishes this process's concurrent temporary files.
static WRITES: AtomicU64 = AtomicU64::new(0);

/// Write `basis` into `dir` under its identity's file name, through a
/// temporary file renamed into place so readers never see a partial file.
pub fn store_basis(dir: &Path, basis: &MeasurementBasis) -> Result<(), io::Error> {
    let name = basis_file_name(&basis.identity).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("basis backend {:?} is unknown", basis.identity.backend),
        )
    })?;
    std::fs::create_dir_all(dir)?;
    let temporary: PathBuf = dir.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&serde_json::to_vec(&basis_json(basis)).map_err(io::Error::other)?)?;
        file.sync_all()?;
        std::fs::rename(&temporary, dir.join(&name))
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::Layout;

    fn identity() -> BasisIdentity {
        BasisIdentity {
            engine_build: "magnitude-model-executor@0.0.0+kernels.test".into(),
            backend: "metal".into(),
            device: "Apple M4 Max;metal".into(),
            protocol_version: MEASUREMENT_PROTOCOL_VERSION,
        }
    }

    fn basis() -> MeasurementBasis {
        let q4k = Element::stored("q4k", Layout::Rows16).unwrap();
        MeasurementBasis {
            identity: identity(),
            classes: vec![
                (
                    MeasurementKey::dense_output(q4k, Element::bf16()),
                    ClassMeasurement::Measured {
                        points: vec![
                            MeasuredPoint {
                                bytes: 2_359_296,
                                samples: vec![9.68e-6, 9.7123456789e-6, 1.0e-5],
                            },
                            MeasuredPoint {
                                bytes: 67_108_864,
                                samples: vec![1.6635e-4, 1.7e-4],
                            },
                        ],
                        cost: ClassCost {
                            model: CostModel::Linear(StreamingCost {
                                launch_seconds: 3.99e-6,
                                seconds_per_byte: 2.41e-12,
                            }),
                            slow_factor: 1.0312,
                            fast_factor: 0.9876,
                        },
                    },
                ),
                (
                    MeasurementKey::dense_expand(Element::bf16(), q4k, Element::bf16()),
                    ClassMeasurement::Measured {
                        points: vec![
                            MeasuredPoint {
                                bytes: 1_048_576,
                                samples: vec![1.7e-5],
                            },
                            MeasuredPoint {
                                bytes: 2_097_152,
                                samples: vec![2.0e-5],
                            },
                        ],
                        cost: ClassCost {
                            model: CostModel::Curve(vec![(1_048_576, 1.7e-5), (2_097_152, 2.0e-5)]),
                            slow_factor: 1.0,
                            fast_factor: 1.0,
                        },
                    },
                ),
                (
                    MeasurementKey::delta_step(16, 32, 128, 4, Element::bf16()),
                    ClassMeasurement::Measured {
                        points: vec![MeasuredPoint {
                            bytes: 3,
                            samples: vec![1.0e-5],
                        }],
                        cost: ClassCost {
                            model: CostModel::PerLaunch { seconds: 1.0e-5 },
                            slow_factor: 1.0,
                            fast_factor: 1.0,
                        },
                    },
                ),
                (
                    MeasurementKey::dense_expand(
                        Element::bf16(),
                        Element::stored("iq4g32", Layout::Rows16).unwrap(),
                        Element::bf16(),
                    ),
                    ClassMeasurement::Unsupported {
                        reason: "no formation".into(),
                    },
                ),
            ],
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "magnitude-basis-{name}-{}-{}",
            std::process::id(),
            WRITES.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn basis_round_trips_exactly_through_its_cache_file() {
        let dir = scratch("round-trip");
        let basis = basis();
        store_basis(&dir, &basis).unwrap();
        assert_eq!(load_basis(&dir, &basis.identity), Some(basis));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn identity_protocol_and_content_mismatches_are_misses() {
        let dir = scratch("miss");
        let basis = basis();
        assert_eq!(load_basis(&dir, &basis.identity), None);
        store_basis(&dir, &basis).unwrap();
        let mut other = identity();
        other.device = "another device".into();
        assert_eq!(load_basis(&dir, &other), None);
        let mut older = identity();
        older.protocol_version = MEASUREMENT_PROTOCOL_VERSION - 1;
        assert_eq!(load_basis(&dir, &older), None);
        let mut unknown = identity();
        unknown.backend = "abacus".into();
        assert_eq!(load_basis(&dir, &unknown), None);

        // A stored file whose identity disagrees with its name, or whose
        // element or class names are unknown, is not a basis.
        let path = dir.join(basis_file_name(&basis.identity).unwrap());
        let mut document = basis_json(&basis);
        document["identity"]["protocol_version"] = json!(MEASUREMENT_PROTOCOL_VERSION + 1);
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert_eq!(load_basis(&dir, &basis.identity), None);
        let mut document = basis_json(&basis);
        document["classes"][0]["key"]["bindings"][0] = json!("q3z@rows16");
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert_eq!(load_basis(&dir, &basis.identity), None);
        let mut document = basis_json(&basis);
        document["classes"][0]["key"]["class"] = json!("dense_teleport");
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert_eq!(load_basis(&dir, &basis.identity), None);
        std::fs::write(&path, b"{ truncated").unwrap();
        assert_eq!(load_basis(&dir, &basis.identity), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn file_names_follow_identity() {
        let name = basis_file_name(&identity()).unwrap();
        assert_eq!(basis_file_name(&identity()).unwrap(), name);
        let mut other = identity();
        other.engine_build.push('+');
        assert_ne!(basis_file_name(&other).unwrap(), name);
    }
}
