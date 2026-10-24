//! Cases mirror V3 tests/weights/test_gguf.py, including its source byte layout.
use magnitude_artifacts::{
    gguf::{self, ByteOrder, Encoding, Scalar, Value},
    Error,
};
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
fn read(bytes: Vec<u8>) -> Result<gguf::Directory, Error> {
    gguf::read_directory(&mut Cursor::new(bytes))
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
        let d = gguf::read_directory(&mut source).unwrap();
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
        // The header's read-ahead (the walk's estimate of its rest) stops short of the end of
        // the tensor storage.
        assert!((source.read as u64) < d.data_offset + d.tensor("weight").unwrap().nbytes);
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
        Some(&Value::Array(vec![0.5, 1.0, -2.0].into()))
    );
    assert_eq!(d.value("unsigned").unwrap().unsigned(), Some(u64::MAX));
}
#[test]
fn arrays_are_held_by_their_element_type() {
    let array = |kind: u32, count: u64, elements: &[u8]| {
        let mut bytes = kind.to_le_bytes().to_vec();
        bytes.extend(count.to_le_bytes());
        bytes.extend(elements);
        bytes
    };
    let mut pieces = Vec::new();
    for piece in ["a", "", "héllo"] {
        pieces.extend((piece.len() as u64).to_le_bytes());
        pieces.extend(piece.as_bytes());
    }
    let kinds = [1i32, -3].map(i32::to_le_bytes).concat();
    let d = read(container(
        false,
        &[entry()],
        &[
            ("pieces", 9, array(8, 3, &pieces)),
            ("kinds", 9, array(5, 2, &kinds)),
            ("bytes", 9, array(0, 1, &[7])),
        ],
        32,
    ))
    .unwrap();
    let Some(Value::Array(gguf::Array::String(strings))) = d.value("pieces") else {
        panic!("a text array is held as text");
    };
    assert_eq!(strings.iter().collect::<Vec<_>>(), ["a", "", "héllo"]);
    assert_eq!(strings.get(2), Some("héllo"));
    assert_eq!(strings.get(3), None);
    assert_eq!(d.value("kinds"), Some(&Value::Array(vec![1_i64, -3].into())));
    assert_eq!(d.value("bytes"), Some(&Value::Array(vec![7_u64].into())));
    assert_eq!(
        gguf::Array::of([Scalar::Unsigned(1), Scalar::Unsigned(2)]).unwrap(),
        vec![1_u64, 2].into()
    );
    assert!(gguf::Array::of([Scalar::Unsigned(1), Scalar::Bool(true)]).is_err());

    // A count no header can hold at its element's size is refused before anything is read for
    // it; one the file merely ends before is a truncated header. Neither allocates its elements.
    let declaring = |count: u64| {
        read(container(
            false,
            &[entry()],
            &[("kinds", 9, array(5, count, &kinds))],
            32,
        ))
    };
    assert!(matches!(declaring(1 << 40), Err(Error::Invalid(message)) if message.contains("oversized")));
    assert!(matches!(declaring(1 << 20), Err(Error::Truncated { .. })));
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
    let root = std::env::temp_dir().join(format!("seismic-gguf-artifact-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("model.gguf");
    let original = container(false, &[entry()], &[], 32);
    std::fs::write(&path, &original).unwrap();
    let artifact = gguf::GgufArtifact::open(&path).unwrap();
    let stored = artifact.tensor("weight").unwrap();
    // Replacing the pathname cannot redirect stored reads to a different inode.
    let replacement = root.join("replacement.gguf");
    std::fs::write(&replacement, vec![0xa5; original.len()]).unwrap();
    std::fs::rename(&replacement, &path).unwrap();
    assert_eq!(stored.shape, [2, 256]);
    assert_eq!(stored.encoding, Encoding::Q4K);
    assert_eq!(stored.nbytes, 288);
    drop(artifact);
    assert_eq!(stored.read().unwrap(), vec![0; 288]);
    std::fs::write(&path, container(true, &[entry()], &[], 32)).unwrap();
    assert!(gguf::GgufArtifact::open(&path).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

/// On every real header fixture: the walk of the whole header is the directory the engine reads
/// from it, and the walk of any prefix is the whole header's walk cut short: the entries before
/// the cut, the entry it falls in, and a need the header's remaining bytes cover, which a reader
/// growing the prefix by the estimate follows to the whole header.
#[test]
fn every_prefix_of_a_real_header_walks_as_the_whole_header_cut_short() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/assessment/headers");
    let mut fixtures = std::fs::read_dir(fixtures)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    fixtures.sort();
    assert!(!fixtures.is_empty());
    for path in fixtures {
        let name = path.display();
        let mut bytes = Vec::new();
        let data_offset = gguf::read_header_bytes(
            &mut flate2::read::GzDecoder::new(std::fs::File::open(&path).unwrap()),
            &mut bytes,
        )
        .unwrap();
        bytes.truncate(data_offset);
        let Ok(gguf::Walk::Complete(header)) = gguf::walk(&bytes) else {
            panic!("{name}: a whole header");
        };

        let directory = match gguf::read_header(&bytes) {
            Ok(directory) => directory,
            Err(Error::UnknownTensorTypes { directory, .. }) => *directory,
            Err(error) => panic!("{name}: {error}"),
        };
        assert_eq!(directory.data_offset, header.data_offset as u64, "{name}");
        assert_eq!(directory.alignment, header.alignment, "{name}");
        assert_eq!(
            directory.metadata,
            header.entries.iter().map(gguf::Entry::metadata).collect::<Vec<_>>(),
            "{name}"
        );
        let known = header
            .tensors
            .iter()
            .filter(|tensor| Encoding::try_from(tensor.type_id).is_ok());
        assert_eq!(directory.tensors.len(), known.clone().count(), "{name}");
        for (descriptor, tensor) in directory.tensors.iter().zip(known) {
            let mut shape = tensor.dimensions().collect::<Vec<_>>();
            shape.reverse();
            assert_eq!(
                (descriptor.name.as_str(), &descriptor.shape, descriptor.encoding as u32, descriptor.offset),
                (tensor.name, &shape, tensor.type_id, tensor.offset),
                "{name}"
            );
        }

        let mut cuts = (0..data_offset).step_by(data_offset / 32 + 1).collect::<Vec<_>>();
        for entry in &header.entries {
            cuts.extend([entry.head.end, (entry.head.end + entry.end) / 2]);
        }
        for cut in cuts {
            let prefix = match gguf::walk(&bytes[..cut]).unwrap() {
                gguf::Walk::Complete(whole) => {
                    assert_eq!(whole, header, "{name} at {cut}");
                    continue;
                }
                gguf::Walk::Incomplete(prefix) => prefix,
            };
            let held = prefix.entries.len();
            assert_eq!(prefix.entries, header.entries[..held], "{name} at {cut}");
            assert!(
                header.entries.get(held).is_none_or(|next| next.end > cut),
                "{name} at {cut}: an entry in hand is left out"
            );
            if let Some(open) = prefix.open {
                assert_eq!(open, header.entries[held].head, "{name} at {cut}");
            }
            assert_eq!(prefix.preamble.is_some(), cut >= 24, "{name} at {cut}");
            assert!(1 <= prefix.minimum && prefix.minimum <= prefix.estimate, "{name} at {cut}");
            assert!(cut + prefix.minimum <= data_offset, "{name} at {cut}: {} more", prefix.minimum);
        }
    }
}
