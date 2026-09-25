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
        stored_size = stored_size.max(*offset as usize + 512);
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
    assert!(manifest.target.path.is_absolute());

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
    // ff0dbb975e93a9a2899efa34bdd32d1c5cfbc183, specifically
    // `ggml/include/ggml.h` and `ggml/src/ggml-common.h`.
    let cases = [
        (Encoding::F32, 0, 1, 4),
        (Encoding::F16, 1, 1, 2),
        (Encoding::Q4_0, 2, 32, 18),
        (Encoding::Q5_0, 6, 32, 22),
        (Encoding::Q5_1, 7, 32, 24),
        (Encoding::Q8_0, 8, 32, 34),
        (Encoding::Q3K, 11, 256, 110),
        (Encoding::Q4K, 12, 256, 144),
        (Encoding::Q5K, 13, 256, 176),
        (Encoding::Q6K, 14, 256, 210),
        (Encoding::Iq4Nl, 20, 32, 18),
        (Encoding::Iq3S, 21, 256, 110),
        (Encoding::Iq4Xs, 23, 256, 136),
        (Encoding::I32, 26, 1, 4),
        (Encoding::BF16, 30, 1, 2),
        (Encoding::Mxfp4, 39, 32, 17),
        (Encoding::Nvfp4, 40, 64, 36),
        (Encoding::Q1_0, 41, 128, 18),
    ];

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
fn unaudited_or_removed_encoding_ids_remain_unsupported() {
    for type_id in [3, 4, 5, 9, 42, u32::MAX] {
        let error = gguf::read_directory(
            &mut Cursor::new(container(false, &[("weight", vec![32], type_id, 0)], &[])),
            gguf::DEFAULT_HEADER_LIMIT,
        )
        .unwrap_err();
        assert!(
            matches!(error, Error::Invalid(message) if message == format!("unsupported GGUF encoding {type_id}"))
        );
    }
}
