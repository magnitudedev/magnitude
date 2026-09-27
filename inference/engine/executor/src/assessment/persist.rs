//! Persistence of a measured basis. The caller names the directory; each
//! device's basis is one JSON file whose name is the content address of its
//! identity. A file holds what was measured (every entry's points, or that it
//! was formed or is unsupported); costs are fitted from the points when the
//! file is read. Elements are stored by name. An absent, unreadable or
//! unparseable file, an unknown class or element name, points that do not fit
//! their class, or an identity or protocol mismatch is a cache miss.

use super::basis::{
    BasisIdentity, ClassCost, ClassMeasurement, HeadGeometry, MeasuredPoint, MeasurementBasis,
    MeasurementKey, OperationClass, PointShape, MEASUREMENT_PROTOCOL_VERSION,
};
use seismic::{BackendName, Element};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The cache file name of a basis with `identity`, or `None` when the
/// identity names no backend this build knows.
pub fn basis_file_name(identity: &BasisIdentity) -> Option<String> {
    BackendName::parse(&identity.backend)?;
    let material = format!(
        "engine {}\nbackend {}\ndevice {}\nprotocol {}",
        identity.engine_build, identity.backend, identity.device, identity.protocol_version
    );
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

fn shape_json(shape: &PointShape) -> Value {
    match shape {
        PointShape::Size => json!("size"),
        PointShape::Launch {
            rows,
            reduction,
            weight,
        } => json!({ "launch": { "rows": rows, "reduction": reduction, "weight": weight.name() } }),
        PointShape::Heads(heads) => json!({ "heads": {
            "kv_heads": heads.kv_heads,
            "group": heads.group,
            "width": heads.width,
        } }),
    }
}

fn measurement_json(measurement: &ClassMeasurement) -> Value {
    match measurement {
        ClassMeasurement::Unsupported { reason } => json!({ "unsupported": reason }),
        ClassMeasurement::Formed => json!({ "formed": true }),
        ClassMeasurement::Measured { points, .. } => json!({
            "points": points
                .iter()
                .map(|point| json!({
                    "shape": shape_json(&point.shape),
                    "bytes": point.bytes,
                    "samples": point.samples,
                }))
                .collect::<Vec<_>>(),
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
    Some(MeasurementKey { class, bindings })
}

fn parse_shape(value: &Value) -> Option<PointShape> {
    if value.as_str() == Some("size") {
        return Some(PointShape::Size);
    }
    if let Some(launch) = value.get("launch") {
        return Some(PointShape::Launch {
            rows: launch.get("rows")?.as_u64()?,
            reduction: launch.get("reduction")?.as_u64()?,
            weight: Element::named(launch.get("weight")?.as_str()?)?,
        });
    }
    let heads = value.get("heads")?;
    Some(PointShape::Heads(HeadGeometry {
        kv_heads: heads.get("kv_heads")?.as_u64()?,
        group: heads.get("group")?.as_u64()?,
        width: heads.get("width")?.as_u64()?,
    }))
}

fn parse_measurement(class: OperationClass, value: &Value) -> Option<ClassMeasurement> {
    if let Some(reason) = value.get("unsupported") {
        return Some(ClassMeasurement::Unsupported {
            reason: reason.as_str()?.to_owned(),
        });
    }
    if value.get("formed").and_then(Value::as_bool) == Some(true) {
        return Some(ClassMeasurement::Formed);
    }
    let points = value
        .get("points")?
        .as_array()?
        .iter()
        .map(|point| {
            Some(MeasuredPoint {
                shape: parse_shape(point.get("shape")?)?,
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
    let cost = ClassCost::from_points(class, &points).ok()?;
    Some(ClassMeasurement::Measured { points, cost })
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
            let key = parse_key(entry.get("key")?)?;
            let measurement = parse_measurement(key.class, entry.get("measurement")?)?;
            Some((key, measurement))
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
            engine_build: "magnitude-executor@0.0.0+kernels.test".into(),
            backend: "metal".into(),
            device: "Apple M4 Max;metal".into(),
            protocol_version: MEASUREMENT_PROTOCOL_VERSION,
        }
    }

    fn measured(class: OperationClass, points: Vec<MeasuredPoint>) -> ClassMeasurement {
        let cost = ClassCost::from_points(class, &points).unwrap();
        ClassMeasurement::Measured { points, cost }
    }

    fn basis() -> MeasurementBasis {
        let q4k = Element::stored("q4k", Layout::Rows16).unwrap();
        let bf16 = Element::bf16();
        let launch = |rows, reduction, seconds| MeasuredPoint {
            shape: PointShape::Launch {
                rows,
                reduction,
                weight: q4k,
            },
            bytes: rows * reduction / 2,
            samples: vec![seconds, 2.0e-4],
        };
        let heads = |kv_heads, bytes, seconds| MeasuredPoint {
            shape: PointShape::Heads(HeadGeometry {
                kv_heads,
                group: 8,
                width: 256,
            }),
            bytes,
            samples: vec![seconds],
        };
        MeasurementBasis {
            identity: identity(),
            classes: vec![
                (
                    MeasurementKey::new(OperationClass::DenseOutput, &[bf16]),
                    measured(
                        OperationClass::DenseOutput,
                        vec![
                            launch(768, 4096, 9.7123456789e-6),
                            launch(6144, 4096, 3.1e-5),
                            launch(49_152, 4096, 1.9e-4),
                            launch(6144, 1024, 1.2e-5),
                        ],
                    ),
                ),
                (
                    MeasurementKey::new(OperationClass::AttentionDecode, &[bf16]),
                    measured(
                        OperationClass::AttentionDecode,
                        vec![
                            heads(2, 1_000_000, 2e-5),
                            heads(2, 8_000_000, 9e-5),
                            heads(4, 16_000_000, 1.2e-4),
                        ],
                    ),
                ),
                (
                    MeasurementKey::delta_step(bf16),
                    measured(
                        OperationClass::DeltaStep,
                        vec![
                            MeasuredPoint {
                                shape: PointShape::Size,
                                bytes: 3_000,
                                samples: vec![1.0e-5],
                            },
                            MeasuredPoint {
                                shape: PointShape::Size,
                                bytes: 9_000,
                                samples: vec![1.5e-5],
                            },
                        ],
                    ),
                ),
                (MeasurementKey::dense_output(q4k, bf16), ClassMeasurement::Formed),
                (
                    MeasurementKey::dense_expand(
                        bf16,
                        Element::stored("iq4g32", Layout::Rows16).unwrap(),
                        bf16,
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

        // A stored file whose identity disagrees with its name, whose
        // element or class names are unknown, or whose points do not fit
        // their class, is not a basis.
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
        let mut document = basis_json(&basis);
        document["classes"][2]["measurement"]["points"] = json!([]);
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
