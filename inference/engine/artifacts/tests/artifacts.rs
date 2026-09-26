use magnitude_artifacts::{
    gguf::{self, ByteOrder, Encoding, Scalar, Value},
    Error, Package, PackageHeaders,
};
use std::{
    io::{Cursor, Read, Seek, SeekFrom},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct Bytes {
    value: Vec<u8>,
    big: bool,
}

impl Bytes {
    fn raw(&mut self, bytes: &[u8]) {
        self.value.extend_from_slice(bytes);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&if self.big {
            value.to_be_bytes()
        } else {
            value.to_le_bytes()
        });
    }

    fn u64(&mut self, value: u64) {
        self.raw(&if self.big {
            value.to_be_bytes()
        } else {
            value.to_le_bytes()
        });
    }

    fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.raw(value.as_bytes());
    }
}

type Entry<'a> = (&'a str, Vec<u64>, u32, u64);

fn string_value(value: &str) -> Vec<u8> {
    let mut bytes = Bytes {
        value: Vec::new(),
        big: false,
    };
    bytes.string(value);
    bytes.value
}

fn container(big: bool, entries: &[Entry<'_>], metadata: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
    let mut bytes = Bytes {
        value: Vec::new(),
        big,
    };
    bytes.raw(b"GGUF");
    bytes.u32(3);
    bytes.u64(entries.len() as u64);
    bytes.u64(metadata.len() as u64);
    for (name, kind, value) in metadata {
        bytes.string(name);
        bytes.u32(*kind);
        bytes.raw(value);
    }
    let mut stored_size = 0;
    for (name, dimensions, encoding, offset) in entries {
        bytes.string(name);
        bytes.u32(dimensions.len() as u32);
        for dimension in dimensions {
            bytes.u64(*dimension);
        }
        bytes.u32(*encoding);
        bytes.u64(*offset);
        // Leave enough payload space for two blocks of every audited encoding;
        // individual parser assertions still verify the exact computed size.
        stored_size = stored_size.max(*offset as usize + 1024);
    }
    bytes.value.resize(bytes.value.len().div_ceil(32) * 32, 0);
    bytes.value.resize(bytes.value.len() + stored_size, 0);
    bytes.value
}

fn entry() -> Entry<'static> {
    ("weight", vec![256, 2], Encoding::Q4K as u32, 0)
}

fn temporary_directory(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "magnitude-artifacts-{label}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn directory_is_bounded_portable_and_preserves_metadata() {
    struct Counted {
        source: Cursor<Vec<u8>>,
        read: usize,
    }
    impl Read for Counted {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let count = self.source.read(out)?;
            self.read += count;
            Ok(count)
        }
    }
    impl Seek for Counted {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.source.seek(position)
        }
    }

    for big in [false, true] {
        let mut source = Counted {
            source: Cursor::new(container(big, &[entry()], &[])),
            read: 0,
        };
        let directory = gguf::read_directory(&mut source, gguf::DEFAULT_HEADER_LIMIT).unwrap();
        assert_eq!(
            directory.byte_order,
            if big {
                ByteOrder::Big
            } else {
                ByteOrder::Little
            }
        );
        assert_eq!(directory.tensor("weight").unwrap().shape, [2, 256]);
        assert!((source.read as u64) <= directory.data_offset);
        assert_eq!(directory.require_execution_byte_order().is_err(), big);
    }

    let directory = gguf::read_directory(
        &mut Cursor::new(container(
            false,
            &[entry()],
            &[("general.name", 8, string_value("fixture"))],
        )),
        gguf::DEFAULT_HEADER_LIMIT,
    )
    .unwrap();
    assert_eq!(
        directory.value("general.name"),
        Some(&Value::Scalar(Scalar::String("fixture".into())))
    );
}

#[test]
fn header_inspection_accepts_declared_weights_without_payload() {
    let root = temporary_directory("header-only");
    let path = root.join("header.gguf");
    let full = container(false, &[entry()], &[]);
    let directory =
        gguf::read_directory(&mut Cursor::new(&full), gguf::DEFAULT_HEADER_LIMIT).unwrap();
    std::fs::write(&path, &full[..directory.data_offset as usize]).unwrap();
    let inspected = gguf::inspect_header(&path).unwrap();
    assert_eq!(inspected.tensors, directory.tensors);
    let headers = PackageHeaders::open(&path, None).unwrap();
    assert_eq!(headers.target().tensors, directory.tensors);
    assert_eq!(headers.identity().projector, None);
    assert!(gguf::GgufArtifact::open(&path).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_keeps_payloads_generic_and_composes_component_identity() {
    let root = temporary_directory("package");
    let target_path = root.join("model.gguf");
    let projector_path = root.join("mmproj-model.gguf");
    std::fs::write(
        &target_path,
        container(
            false,
            &[entry()],
            &[
                ("tokenizer.ggml.pre", 8, string_value("family-owned-value")),
                ("tokenizer.chat_template", 8, string_value("{{ messages }}")),
                (
                    "tokenizer.chat_template.tools",
                    8,
                    string_value("{{ tools }}"),
                ),
            ],
        ),
    )
    .unwrap();
    std::fs::write(
        &projector_path,
        container(
            false,
            &[entry()],
            &[("general.type", 8, string_value("mmproj"))],
        ),
    )
    .unwrap();

    let package = Package::open(&target_path).unwrap();
    assert!(package.projector().is_some());
    assert_eq!(
        package.tokenizer().value("tokenizer.ggml.pre"),
        Some(&Value::Scalar(Scalar::String("family-owned-value".into())))
    );
    assert_eq!(
        package.tokenizer().value("tokenizer.chat_template"),
        None,
        "template sources have one package-owned payload, not a duplicate tokenizer entry"
    );
    assert_eq!(
        package
            .templates()
            .sources
            .iter()
            .map(|source| source.name.as_str())
            .collect::<Vec<_>>(),
        ["default", "tools"]
    );
    assert_eq!(package.identity().to_string().matches(':').count(), 1);
    assert_ne!(
        package.identity().to_string(),
        Package::open_without_projector(&target_path)
            .unwrap()
            .identity()
            .to_string()
    );
    assert!(matches!(
        Package::open_with_projector(&target_path, &target_path),
        Err(Error::Invalid(message)) if message.contains("distinct package components")
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn opened_package_remains_bound_to_admitted_file() {
    let root = temporary_directory("manifest-race");
    let path = root.join("model.gguf");
    let admitted = container(
        false,
        &[entry()],
        &[("general.name", 8, string_value("admitted"))],
    );
    std::fs::write(&path, admitted).unwrap();

    let package = Package::open_without_projector(&path).unwrap();
    let manifest = package.manifest();
    assert!(manifest.target.path().is_absolute());

    let replacement = root.join("replacement.gguf");
    std::fs::write(
        &replacement,
        container(
            false,
            &[entry()],
            &[("general.name", 8, string_value("replacement"))],
        ),
    )
    .unwrap();
    std::fs::rename(replacement, &path).unwrap();

    assert_ne!(
        Package::open_without_projector(&path).unwrap().identity(),
        package.identity()
    );
    // The already-open package still refers to the admitted source and payload.
    assert_eq!(package.manifest(), manifest);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn hardlinked_components_are_rejected() {
    let root = temporary_directory("hardlinked-components");
    let target = root.join("model.gguf");
    let projector = root.join("mmproj-model.gguf");
    std::fs::write(&target, container(false, &[entry()], &[])).unwrap();
    std::fs::hard_link(&target, &projector).unwrap();
    assert!(matches!(
        Package::open_with_projector(&target, &projector),
        Err(Error::Invalid(message)) if message.contains("distinct package components")
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn mapped_window_keeps_open_source_after_path_replacement() {
    let root = temporary_directory("mapped-window");
    let path = root.join("source.bin");
    let original = (0..8193).map(|n| (n % 251) as u8).collect::<Vec<_>>();
    std::fs::write(&path, &original).unwrap();
    let source = std::sync::Arc::new(magnitude_artifacts::FileSource::open(&path).unwrap());
    let window = source.map_window(17, 4097).unwrap();
    assert_eq!(window.data(), &original[17..4114]);
    assert_eq!(window.as_ref().as_ref().as_ptr() as usize % 4096, 0);
    assert!(window.mapped_len() >= window.data_offset() + window.data_len());
    std::fs::write(root.join("replacement.bin"), b"replacement").unwrap();
    std::fs::rename(root.join("replacement.bin"), &path).unwrap();
    drop(source);
    assert_eq!(window.data(), &original[17..4114]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn open_source_pins_tensor_bytes_and_projector_discovery_is_unambiguous() {
    let root = temporary_directory("ownership");
    let path = root.join("model.gguf");
    let original = container(false, &[entry()], &[]);
    std::fs::write(&path, &original).unwrap();
    let artifact = gguf::GgufArtifact::open(&path).unwrap();
    let stored = artifact.tensor("weight").unwrap();

    let replacement = root.join("replacement.gguf");
    std::fs::write(&replacement, vec![0xa5; original.len()]).unwrap();
    std::fs::rename(&replacement, &path).unwrap();
    drop(artifact);
    assert_eq!(stored.read().unwrap(), vec![0; 288]);

    std::fs::write(&path, &original).unwrap();
    std::fs::write(root.join("mmproj-one.gguf"), &original).unwrap();
    std::fs::write(root.join("mmproj-two.gguf"), &original).unwrap();
    assert!(matches!(
        Package::open(&path),
        Err(Error::AmbiguousProjectors(projectors)) if projectors.len() == 2
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn bf16_is_a_dense_two_byte_container_encoding() {
    let bytes = container(
        false,
        &[("weight", vec![4, 2], Encoding::BF16 as u32, 0)],
        &[],
    );
    let directory =
        gguf::read_directory(&mut Cursor::new(bytes), gguf::DEFAULT_HEADER_LIMIT).unwrap();
    let tensor = directory.tensor("weight").unwrap();
    assert_eq!(tensor.encoding, Encoding::BF16);
    assert_eq!(tensor.nbytes, 16);
}

#[test]
fn audited_encoding_ids_and_block_geometry_match_pinned_ggml() {
    // Pinned source of truth: ggml/llama.cpp
    // 5e6c0e18b6b11f109411401239b3b0ef61058dae, specifically
    // `ggml/include/ggml.h` and `ggml/src/ggml-common.h`.
    let cases = [
        (Encoding::F32, 0, 1, 4),
        (Encoding::F16, 1, 1, 2),
        (Encoding::Q4_0, 2, 32, 18),
        (Encoding::Q4_1, 3, 32, 20),
        (Encoding::Q5_0, 6, 32, 22),
        (Encoding::Q5_1, 7, 32, 24),
        (Encoding::Q8_0, 8, 32, 34),
        (Encoding::Q8_1, 9, 32, 36),
        (Encoding::Q2K, 10, 256, 84),
        (Encoding::Q3K, 11, 256, 110),
        (Encoding::Q4K, 12, 256, 144),
        (Encoding::Q5K, 13, 256, 176),
        (Encoding::Q6K, 14, 256, 210),
        (Encoding::Q8K, 15, 256, 292),
        (Encoding::Iq2Xxs, 16, 256, 66),
        (Encoding::Iq2Xs, 17, 256, 74),
        (Encoding::Iq3Xxs, 18, 256, 98),
        (Encoding::Iq1S, 19, 256, 50),
        (Encoding::Iq4Nl, 20, 32, 18),
        (Encoding::Iq3S, 21, 256, 110),
        (Encoding::Iq2S, 22, 256, 82),
        (Encoding::Iq4Xs, 23, 256, 136),
        (Encoding::I8, 24, 1, 1),
        (Encoding::I16, 25, 1, 2),
        (Encoding::I32, 26, 1, 4),
        (Encoding::I64, 27, 1, 8),
        (Encoding::F64, 28, 1, 8),
        (Encoding::Iq1M, 29, 256, 56),
        (Encoding::BF16, 30, 1, 2),
        (Encoding::Tq1_0, 34, 256, 54),
        (Encoding::Tq2_0, 35, 256, 66),
        (Encoding::Mxfp4, 39, 32, 17),
        (Encoding::Nvfp4, 40, 64, 36),
        (Encoding::Q1_0, 41, 128, 18),
    ];
    assert_eq!(
        cases.map(|(encoding, ..)| encoding),
        Encoding::ALL,
        "every GGUF tensor type is audited"
    );

    for (encoding, type_id, block_elements, block_bytes) in cases {
        assert_eq!(encoding as u32, type_id);
        assert_eq!(encoding.block_elements(), block_elements);
        assert_eq!(encoding.block_bytes(), block_bytes);

        let bytes = container(
            false,
            &[("weight", vec![block_elements, 2], type_id, 0)],
            &[],
        );
        let directory =
            gguf::read_directory(&mut Cursor::new(bytes), gguf::DEFAULT_HEADER_LIMIT).unwrap();
        let tensor = directory.tensor("weight").unwrap();
        assert_eq!(tensor.encoding, encoding);
        assert_eq!(tensor.nbytes, block_bytes * 2);
    }
}

#[test]
fn removed_or_unknown_encoding_ids_are_invalid() {
    for type_id in [4, 5, 31, 32, 33, 36, 37, 38, 42, u32::MAX] {
        let error = gguf::read_directory(
            &mut Cursor::new(container(false, &[("weight", vec![32], type_id, 0)], &[])),
            gguf::DEFAULT_HEADER_LIMIT,
        )
        .unwrap_err();
        assert!(
            matches!(error, Error::Invalid(message) if message == format!("unknown GGUF tensor type {type_id}"))
        );
    }
}

fn u16_value(value: u16) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// One shard of a `gguf-split` GGUF (split.no/split.count are u16, the tensor
/// total i32), with each tensor's payload filled with `fill`.
fn shard(
    index: u16,
    count: u16,
    total: i32,
    entries: &[Entry<'_>],
    metadata: &[(&str, u32, Vec<u8>)],
    fill: u8,
) -> Vec<u8> {
    let mut all = vec![
        ("split.no", 2, u16_value(index)),
        ("split.count", 2, u16_value(count)),
        ("split.tensors.count", 5, total.to_le_bytes().to_vec()),
    ];
    all.extend(metadata.iter().cloned());
    let mut bytes = container(false, entries, &all);
    let directory =
        gguf::read_directory(&mut Cursor::new(&bytes), gguf::DEFAULT_HEADER_LIMIT).unwrap();
    for tensor in &directory.tensors {
        let start = (directory.data_offset + tensor.offset) as usize;
        bytes[start..start + tensor.nbytes as usize].fill(fill);
    }
    bytes
}

/// A three-file split GGUF: metadata only in the first shard, one tensor in
/// each of the others. Returns the paths in shard order.
fn write_split(root: &std::path::Path, header_only: bool) -> Vec<PathBuf> {
    let shards = [
        shard(
            0,
            3,
            2,
            &[],
            &[("general.name", 8, string_value("split"))],
            0,
        ),
        shard(
            1,
            3,
            2,
            &[("first", vec![256, 2], Encoding::Q4K as u32, 0)],
            &[],
            0x11,
        ),
        shard(
            2,
            3,
            2,
            &[("second", vec![32, 3], Encoding::Q8_0 as u32, 0)],
            &[],
            0x22,
        ),
    ];
    shards
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            let path = root.join(format!("model-{:05}-of-00003.gguf", index + 1));
            let bytes = if header_only {
                let directory =
                    gguf::read_directory(&mut Cursor::new(bytes), gguf::DEFAULT_HEADER_LIMIT)
                        .unwrap();
                &bytes[..directory.data_offset as usize]
            } else {
                &bytes[..]
            };
            std::fs::write(&path, bytes).unwrap();
            path
        })
        .collect()
}

#[test]
fn split_gguf_opens_as_one_component() {
    let root = temporary_directory("split");
    let paths = write_split(&root, false);
    let artifact = gguf::GgufArtifact::open(&paths[0]).unwrap();
    assert_eq!(artifact.sources().len(), 3);
    let directory = artifact.directory();
    assert_eq!(
        directory.value("general.name"),
        Some(&Value::Scalar(Scalar::String("split".into())))
    );
    assert_eq!(
        directory
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(directory.data_offset, 0);
    let first = artifact.tensor("first").unwrap();
    let second = artifact.tensor("second").unwrap();
    assert_eq!(first.read().unwrap(), vec![0x11; 288]);
    assert_eq!(second.read().unwrap(), vec![0x22; 102]);
    assert!(!std::sync::Arc::ptr_eq(&first.source, &second.source));

    // The package, its manifest and a reopen by manifest cover every shard.
    let package = Package::open_without_projector(&paths[0]).unwrap();
    let manifest = package.manifest();
    assert_eq!(manifest.target.files.len(), 3);
    assert_eq!(manifest.target.path(), paths[0].canonicalize().unwrap());
    let reopened = Package::open_manifest(&manifest).unwrap();
    assert_eq!(reopened.identity(), package.identity());
    assert_eq!(reopened.manifest(), manifest);

    // Only the first shard names the component.
    assert!(matches!(
        gguf::GgufArtifact::open(&paths[1]),
        Err(Error::Invalid(message)) if message.contains("open its first shard")
    ));
    std::fs::remove_file(&paths[2]).unwrap();
    assert!(matches!(
        gguf::GgufArtifact::open(&paths[0]),
        Err(Error::Io(_))
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn split_gguf_headers_match_the_payload_backed_directory() {
    let full_root = temporary_directory("split-full");
    let header_root = temporary_directory("split-headers");
    let full = write_split(&full_root, false);
    let headers = write_split(&header_root, true);
    let opened = gguf::GgufArtifact::open(&full[0]).unwrap();
    let inspected = PackageHeaders::open(&headers[0], None).unwrap();
    assert_eq!(inspected.target().tensors, opened.directory().tensors);
    assert_eq!(inspected.target().metadata, opened.directory().metadata);
    assert!(gguf::GgufArtifact::open(&headers[0]).is_err());
    std::fs::remove_dir_all(full_root).unwrap();
    std::fs::remove_dir_all(header_root).unwrap();
}

#[test]
fn split_gguf_requires_the_split_naming_and_consistent_shards() {
    let root = temporary_directory("split-naming");
    let paths = write_split(&root, false);
    let renamed = root.join("model.gguf");
    std::fs::copy(&paths[0], &renamed).unwrap();
    assert!(matches!(
        gguf::GgufArtifact::open(&renamed),
        Err(Error::Invalid(message)) if message.contains("is not named")
    ));
    // A shard declaring another position is rejected.
    std::fs::copy(&paths[1], &paths[2]).unwrap();
    assert!(matches!(
        gguf::GgufArtifact::open(&paths[0]),
        Err(Error::Invalid(message)) if message.contains("does not declare itself shard 3")
    ));
    std::fs::remove_dir_all(root).unwrap();
}
