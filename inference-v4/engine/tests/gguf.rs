//! Cases mirror V3 tests/weights/test_gguf.py, including its source byte layout.
use seismic_engine::weights::gguf::{self, ByteOrder, Encoding, Scalar, Value};
use std::io::{Cursor, Read, Seek, SeekFrom};
struct Bytes {
    value: Vec<u8>,
    big: bool,
}
impl Bytes {
    fn raw(&mut self, b: &[u8]) {
        self.value.extend_from_slice(b)
    }
    fn u32(&mut self, n: u32) {
        self.raw(&if self.big {
            n.to_be_bytes()
        } else {
            n.to_le_bytes()
        })
    }
    fn u64(&mut self, n: u64) {
        self.raw(&if self.big {
            n.to_be_bytes()
        } else {
            n.to_le_bytes()
        })
    }
    fn string(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.raw(s.as_bytes())
    }
}
type Entry<'a> = (&'a str, Vec<u64>, u32, u64);
fn container(
    big: bool,
    entries: &[Entry<'_>],
    metadata: &[(&str, u32, Vec<u8>)],
    alignment: usize,
) -> Vec<u8> {
    let mut b = Bytes {
        value: Vec::new(),
        big,
    };
    b.raw(b"GGUF");
    b.u32(3);
    b.u64(entries.len() as u64);
    b.u64(metadata.len() as u64);
    for (name, kind, value) in metadata {
        b.string(name);
        b.u32(*kind);
        b.raw(value)
    }
    let mut size = 0;
    for (name, dims, encoding, offset) in entries {
        b.string(name);
        b.u32(dims.len() as u32);
        for dim in dims {
            b.u64(*dim)
        }
        b.u32(*encoding);
        b.u64(*offset);
        size = size.max(*offset as usize + 288)
    }
    b.value
        .resize(b.value.len().div_ceil(alignment) * alignment, 0);
    b.value.resize(b.value.len() + size, 0);
    b.value
}
fn entry() -> Entry<'static> {
    ("weight", vec![256, 2], Encoding::Q4K as u32, 0)
}
fn read(bytes: Vec<u8>) -> Result<gguf::Directory, seismic_engine::weights::Error> {
    gguf::read_directory(&mut Cursor::new(bytes), gguf::DEFAULT_HEADER_LIMIT)
}
#[test]
fn portable_directory_reverses_axes_without_reading_tensor_storage() {
    struct Counted {
        source: Cursor<Vec<u8>>,
        read: usize,
    }
    impl Read for Counted {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let n = self.source.read(out)?;
            self.read += n;
            Ok(n)
        }
    }
    impl Seek for Counted {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            self.source.seek(to)
        }
    }
    for big in [false, true] {
        let mut source = Counted {
            source: Cursor::new(container(big, &[entry()], &[], 32)),
            read: 0,
        };
        let d = gguf::read_directory(&mut source, gguf::DEFAULT_HEADER_LIMIT).unwrap();
        assert_eq!(
            d.byte_order,
            if big {
                ByteOrder::Big
            } else {
                ByteOrder::Little
            }
        );
        assert_eq!(d.tensor("weight").unwrap().shape, [2, 256]);
        assert_eq!(d.tensor("weight").unwrap().nbytes, 288);
        assert_eq!(d.data_offset % 32, 0);
        assert!((source.read as u64) <= d.data_offset);
        assert_eq!(d.require_execution_byte_order().is_err(), big);
    }
}
#[test]
fn metadata_types_arrays_and_full_width_integers_are_preserved() {
    let mut array = 6u32.to_le_bytes().to_vec();
    array.extend(3u64.to_le_bytes());
    for n in [0.5f32, 1.0, -2.0] {
        array.extend(n.to_le_bytes())
    }
    let d = read(container(
        false,
        &[entry()],
        &[
            ("signed", 11, (-4i64).to_le_bytes().to_vec()),
            ("flag", 7, vec![1]),
            ("values", 9, array),
            ("unsigned", 10, u64::MAX.to_le_bytes().to_vec()),
        ],
        32,
    ))
    .unwrap();
    assert_eq!(d.value("signed"), Some(&Value::Scalar(Scalar::Signed(-4))));
    assert_eq!(d.value("flag"), Some(&Value::Scalar(Scalar::Bool(true))));
    assert_eq!(
        d.value("values"),
        Some(&Value::Array(vec![
            Scalar::Float(0.5),
            Scalar::Float(1.0),
            Scalar::Float(-2.0)
        ]))
    );
    assert_eq!(d.value("unsigned").unwrap().unsigned(), Some(u64::MAX));
}
#[test]
fn rejects_truncation_geometry_overlap_and_duplicate_names() {
    let full = container(false, &[entry()], &[], 32);
    for n in [0, 3, 7, 8, 23, 35, 69, 351] {
        assert!(read(full[..n].to_vec()).is_err(), "length {n}")
    }
    for entries in [
        vec![("weight", vec![255, 2], 12, 0)],
        vec![("weight", vec![0, 2], 12, 0)],
        vec![("weight", vec![256, 2], 12, 1)],
        vec![("weight", vec![256, 2], 999, 0)],
        vec![entry(), ("weight", vec![256, 2], 12, 288)],
        vec![
            ("first", vec![256, 2], 12, 0),
            ("second", vec![256, 2], 12, 32),
        ],
    ] {
        assert!(read(container(false, &entries, &[], 32)).is_err())
    }
}
#[test]
fn rejects_invalid_alignment_boolean_duplicate_and_nested_metadata() {
    for n in [0u32, 3, 17] {
        assert!(read(container(
            false,
            &[entry()],
            &[("general.alignment", 4, n.to_le_bytes().to_vec())],
            32
        ))
        .is_err())
    }
    for metadata in [
        vec![("flag", 7, vec![2])],
        vec![("a", 7, vec![0]), ("a", 7, vec![1])],
        vec![(
            "nested",
            9,
            [
                9u32.to_le_bytes().as_slice(),
                1u64.to_le_bytes().as_slice(),
                &[0],
            ]
            .concat(),
        )],
    ] {
        assert!(read(container(false, &[entry()], &metadata, 32)).is_err())
    }
}
#[test]
fn oversized_header_requests_and_size_overflow_are_rejected_before_allocation() {
    let mut raw = b"GGUF".to_vec();
    raw.extend(3u32.to_le_bytes());
    raw.extend(0u64.to_le_bytes());
    raw.extend(1u64.to_le_bytes());
    raw.extend(u64::MAX.to_le_bytes());
    raw.extend([0; 100]);
    assert!(read(raw).is_err());
    assert!(read(container(
        false,
        &[("overflow", vec![u64::MAX, 2], 0, 0)],
        &[],
        32
    ))
    .is_err());
    let bytes = container(false, &[entry()], &[], 32);
    assert!(gguf::read_directory(&mut Cursor::new(bytes), 32).is_err());
}
#[test]
fn declaration_order_does_not_depend_on_physical_order() {
    let d = read(container(
        false,
        &[
            ("second", vec![256, 2], 12, 288),
            ("first", vec![256, 2], 12, 0),
        ],
        &[],
        32,
    ))
    .unwrap();
    assert_eq!(
        d.tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["second", "first"]
    );
}

#[test]
fn artifact_pins_validated_source_and_codec_descriptions() {
    use seismic_engine::weights::descriptor::{Stored, Transform, WeightDescriptor};
    let root = std::env::temp_dir().join(format!("seismic-gguf-artifact-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("model.gguf");
    let original = container(false, &[entry()], &[], 32);
    std::fs::write(&path, &original).unwrap();
    let artifact = gguf::GgufArtifact::open(&path).unwrap();
    let descriptor = WeightDescriptor {
        name: "weight".into(),
        shape: vec![2, 256],
        transform: Transform::Identity,
    };
    let stored = artifact.stored(&descriptor).unwrap();
    // Replacing the pathname cannot redirect stored reads to a different inode.
    let replacement = root.join("replacement.gguf");
    std::fs::write(&replacement, vec![0xa5; original.len()]).unwrap();
    std::fs::rename(&replacement, &path).unwrap();
    let Stored::GgmlBlocks {
        source,
        offset,
        nbytes,
        shape,
        encoding,
    } = stored
    else {
        panic!()
    };
    assert_eq!(shape, [2, 256]);
    assert_eq!(encoding, Encoding::Q4K);
    assert_eq!(nbytes, 288);
    drop(artifact);
    assert_eq!(source.read(offset, nbytes as usize).unwrap(), vec![0; 288]);
    std::fs::write(&path, container(true, &[entry()], &[], 32)).unwrap();
    assert!(gguf::GgufArtifact::open(&path).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
