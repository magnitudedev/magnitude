use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let root = std::env::temp_dir().join(format!(
                "magnitude-gsq-rco-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&root) {
                Ok(()) => return Self(root),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("{root:?}: {error}"),
            }
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn text(name: &str, value: &str) -> StoredMetadata {
    scalar(name, ValueType::String, Scalar::String(value.into()))
}

fn tensor(name: &str, shape: &[u64], encoding: Encoding, bytes: impl Into<Vec<u8>>) -> Tensor {
    Tensor {
        name: name.into(),
        shape: shape.to_vec(),
        encoding,
        bytes: Bytes::Generated(bytes.into()),
    }
}

fn write_fixture(path: &Path, metadata: Vec<StoredMetadata>, tensors: &[Tensor]) {
    let header =
        StoredHeader::new(GGUF_VERSION, metadata, tensors.iter().map(Tensor::info)).unwrap();
    write_shard(path, &header, &tensors.iter().collect::<Vec<_>>()).unwrap();
}

/// A tiny four-shard legacy bundle with both Engram tables, grouped output,
/// and an ordinary quantized tensor. Uses the real on-disk legacy shapes.
fn legacy(root: &Path, wrong_offsets: bool, revision: &str) -> (PathBuf, PathBuf) {
    let mut metadata = vec![
        text("general.architecture", ARCH),
        text(&key("engram.source"), "matching revision"),
        text(TEMPLATE, "the old template"),
        array("kv_source_layer_ids", ValueType::U32, vec![2_u64]),
        array("index_source_layer_ids", ValueType::U32, vec![2_u64, 3]),
        array("engram.layer_ids", ValueType::U32, vec![1_u64, 3]),
        array("engram.num_embeddings", ValueType::U64, vec![4_u64, 6]),
        StoredMetadata {
            metadata: Metadata {
                name: "tokenizer.ggml.tokens".into(),
                value: Value::Array(
                    Array::of(["a", "b", "c"].map(|s| Scalar::String(s.into()))).unwrap(),
                ),
            },
            stored: StoredType::Array(ValueType::String),
        },
    ];
    for (key_suffix, value) in [
        ("block_count", 4),
        ("expert_count", 2),
        ("attention.candidate_block_size", 4),
        ("attention.candidate_topk_blocks", 16),
        ("attention.candidate_source_layer_id", 2),
        ("engram_max_ngram_size", 3),
        ("engram_n_heads", 1),
        ("engram_head_dim", 256),
        ("engram_vocab_size", 16_000_000),
        ("engram_compressed_vocab_size", 2),
        ("engram_pad_token_id", 0),
    ] {
        metadata.push(scalar(
            &key(key_suffix),
            ValueType::U32,
            Scalar::Unsigned(value),
        ));
    }
    let [map, primes, offsets, multipliers] = CONSTANTS;
    let mut tensors = vec![tensor(
        map,
        &[3],
        Encoding::I32,
        [0_i32, 1, 1].map(i32::to_le_bytes).concat(),
    )];
    for (name, shape, values) in [
        (primes, [2, 2], vec![2_i64, 2, 2, 4]),
        (offsets, [2, 2], vec![0, 2, i64::from(wrong_offsets), 2]),
        (multipliers, [2, 3], vec![1, 3, 5, 1, 5, 7]),
    ] {
        tensors.push(tensor(
            name,
            &shape,
            Encoding::I64,
            values
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect::<Vec<_>>(),
        ));
    }
    for layer in [1, 3] {
        for (stem, bf16) in [("q", 0x4000_u16), ("k", 0x4040_u16)] {
            tensors.push(tensor(
                &format!("blk.{layer}.engram_{stem}.weight"),
                &[1, 2],
                Encoding::BF16,
                bf16.to_le_bytes().repeat(2),
            ));
        }
    }
    for layer in 0..4 {
        tensors.push(tensor(
            &format!("blk.{layer}.exp_probs_b.bias"),
            &[2],
            Encoding::F32,
            [0; 8],
        ));
    }
    tensors.extend([
        tensor("blk.1.engram_wkv.weight", &[2, 4], Encoding::F32, [1; 32]),
        tensor(
            "blk.2.attn_output_a.weight",
            &[2, 2, 2],
            Encoding::F32,
            [3; 32],
        ),
        tensor("token_embd.weight", &[1, 256], Encoding::Iq2Xxs, [5; 66]),
    ]);
    let total = tensors.len() as i64;
    for (i, chunk) in tensors.chunks(4).enumerate() {
        let mut shard_kv = if i == 0 {
            std::mem::take(&mut metadata)
        } else {
            Vec::new()
        };
        shard_kv.extend([
            scalar("split.no", ValueType::U16, Scalar::Unsigned(i as u64)),
            scalar("split.count", ValueType::U16, Scalar::Unsigned(4)),
            scalar("split.tensors.count", ValueType::I32, Scalar::Signed(total)),
        ]);
        write_fixture(
            &root.join(format!("legacy-{:05}-of-00004.gguf", i + 1)),
            shard_kv,
            chunk,
        );
    }
    let table = root.join("legacy-engram.gguf");
    write_fixture(
        &table,
        vec![
            text("general.architecture", "deepseek41-engram"),
            text(&key("engram.source"), revision),
        ],
        &[
            tensor(
                "blk.1.engram_embd.weight",
                &[4, 256],
                Encoding::Iq4Nl,
                vec![9; 4 * 144],
            ),
            tensor(
                "blk.3.engram_embd.weight",
                &[6, 256],
                Encoding::Iq4Nl,
                vec![10; 6 * 144],
            ),
        ],
    );
    (root.join("legacy-00001-of-00004.gguf"), table)
}

fn build(model: &Path, sidecar: &Path) -> Result<Plan> {
    Plan::build(
        &GgufArtifact::open(model)?,
        &GgufArtifact::open(sidecar)?,
        "the new template".into(),
        &(0..32).collect::<Vec<_>>(),
    )
}

#[test]
fn rewrites_only_the_legacy_layout_and_publishes_verified_gguf_shards() {
    let root = Scratch::new();
    let (model, sidecar) = legacy(&root.0, false, "matching revision");
    let original = GgufArtifact::open(&model).unwrap();
    let original_shards = original
        .sources()
        .map(|s| (s.path().to_owned(), fs::read(s.path()).unwrap()))
        .collect::<Vec<_>>();
    let original_table = fs::read(&sidecar).unwrap();
    let manifest = write_manifest(&root.0, &sidecar);
    let trusted = TrustedSources::read(&model, "legacy", &sidecar, &manifest).unwrap();
    let inputs = opened_inputs(&trusted, &sidecar);
    let sources = inputs.iter().collect::<Vec<_>>();
    let plan = build(&model, &sidecar).unwrap();
    let shards = plan.shards(128).unwrap();
    let output = root.0.join("imported");
    let first = {
        let mut staged = Output::create(&output).unwrap();
        trusted.authenticate(&sources).unwrap();
        let names = staged.write_shards(&plan, &shards, "modern").unwrap();
        trusted.check(&sources).unwrap();
        staged.publish(&names).unwrap()
    };
    assert_eq!(first, output.join("modern-00001-of-00006.gguf"));
    let converted = GgufArtifact::open(&first).unwrap();
    assert_eq!(converted.sources().len(), 6);
    assert!(!output.join(".partial").exists());
    let directory = converted.directory();
    let value = |name: &str| directory.value(&key(name));
    assert!(value("kv_source_layer_ids").is_none());
    for (name, expected) in [
        ("attention.kv_source_layers", vec![2]),
        ("attention.indexer.source_layers", vec![2, 3]),
        ("ple.head_offsets", vec![0, 2, 4, 6]),
    ] {
        let Some(Value::Array(counts)) = value(name) else {
            panic!("missing {name}");
        };
        assert_eq!(
            counts
                .elements()
                .unwrap()
                .iter()
                .map(|x| x.unsigned().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
    }
    assert_eq!(
        value("attention.indexer.candidate_source_layer").and_then(Value::unsigned),
        Some(2)
    );
    assert_eq!(
        directory.value(TEMPLATE).and_then(Value::string),
        Some("the new template")
    );
    for name in CONSTANTS
        .into_iter()
        .chain(["blk.1.engram_q.weight", "blk.1.engram_k.weight"])
    {
        assert!(directory.tensor(name).is_none(), "legacy tensor {name}");
    }
    let read = |name| converted.tensor(name).unwrap().read().unwrap();
    assert_eq!(
        read("blk.1.engram_gate.weight"),
        6_f32.to_le_bytes().repeat(2)
    );
    assert_eq!(read("blk.1.engram_kv.weight"), [1; 32]);
    assert_eq!(
        directory
            .tensor("blk.2.attn_output_a.weight")
            .unwrap()
            .shape,
        [4, 2]
    );
    assert_eq!(read("blk.2.attn_output_a.weight"), [3; 32]);
    assert_eq!(
        read("blk.3.exp_probs_b_vl.bias"),
        (24..32).collect::<Vec<u8>>()
    );
    assert_eq!(read("token_embd.weight"), [5; 66]);
    assert_eq!(directory.tensor(TABLE).unwrap().shape, [10, 256]);
    assert_eq!(read(TABLE), [vec![9; 4 * 144], vec![10; 6 * 144]].concat());
    assert!(Output::create(&output).is_err());
    assert!(GgufArtifact::open(first).is_ok());
    assert_eq!(fs::read(sidecar).unwrap(), original_table);
    for (path, bytes) in original_shards {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
}

#[test]
fn refuses_bad_engram_offsets_wrong_revision_and_untrusted_data_sidecar() {
    let offsets = Scratch::new();
    let (model, sidecar) = legacy(&offsets.0, true, "matching revision");
    let Err(error) = build(&model, &sidecar) else {
        panic!("wrong offsets were accepted")
    };
    assert!(
        error.to_string().contains("layer-local offsets"),
        "{error:#}"
    );

    let revision = Scratch::new();
    let (model, sidecar) = legacy(&revision.0, false, "another revision");
    let Err(error) = build(&model, &sidecar) else {
        panic!("wrong revision was accepted")
    };
    assert!(
        error.to_string().contains("another checkpoint revision"),
        "{error:#}"
    );

    let fake_data = revision.0.join("exp_probs_b_vl.f32");
    fs::write(&fake_data, [0; 32]).unwrap();
    let error = checked_sidecar(&fake_data, BIAS_SHA256).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not match the pinned data sidecar"),
        "{error:#}"
    );
}

#[test]
fn refuses_shard_bytes_that_differ_from_the_copy_stream() {
    let root = Scratch::new();
    let path = root.0.join("stage.gguf");
    let bytes = b"expected output bytes";
    let digest = Sha256::digest(bytes);
    fs::write(&path, bytes).unwrap();
    verify_shard(&path, &digest).unwrap();
    fs::write(&path, b"tampered output bytes").unwrap();
    let error = verify_shard(&path, &digest).unwrap_err();
    assert!(
        error.to_string().contains("verification failed on reread"),
        "{error:#}"
    );
}

fn write_manifest(root: &Path, engram: &Path) -> PathBuf {
    let contents = (1..=BACKBONE_SHARDS)
        .map(|i| root.join(format!("legacy-{i:05}-of-00004.gguf")))
        .chain([engram.to_owned()])
        .map(|path| {
            format!(
                "{:x}  {}\n",
                Sha256::digest(fs::read(&path).unwrap()),
                path.file_name().unwrap().to_str().unwrap()
            )
        })
        .collect::<String>();
    let manifest = root.join("SHA256SUMS");
    fs::write(&manifest, contents).unwrap();
    manifest
}

fn opened_inputs(trusted: &TrustedSources, sidecar: &Path) -> Vec<magnitude_artifacts::FileSource> {
    trusted
        .backbone()
        .into_iter()
        .chain([sidecar.to_owned()])
        .map(|path| magnitude_artifacts::FileSource::open(path).unwrap())
        .collect()
}

#[test]
fn manifest_cannot_select_paths_duplicate_entries_or_omit_inputs() {
    let root = Scratch::new();
    let (model, engram) = legacy(&root.0, false, "matching revision");
    let manifest = write_manifest(&root.0, &engram);
    let contents = fs::read_to_string(&manifest).unwrap();
    let first = contents.lines().next().unwrap();
    let (_, first_name) = first.split_once("  ").unwrap();
    for (invalid, message) in [
        (
            format!("{contents}{first}\n"),
            "not a remaining input basename",
        ),
        (
            contents.replacen(first_name, &format!("../{first_name}"), 1),
            "not a remaining input basename",
        ),
        (
            contents.strip_prefix(&format!("{first}\n")).unwrap().into(),
            "missing SHA-256",
        ),
        (format!("g{}", &contents[1..]), "expected a SHA-256"),
    ] {
        fs::write(&manifest, invalid).unwrap();
        let error = TrustedSources::read(&model, "legacy", &engram, &manifest)
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains(message), "{error:#}");
    }
    let binary = contents
        .lines()
        .map(|line| {
            let (digest, name) = line.split_once("  ").unwrap();
            format!("{} *{name}\n", digest.to_ascii_uppercase())
        })
        .collect::<String>();
    fs::write(&manifest, binary).unwrap();
    assert!(TrustedSources::read(&model, "legacy", &engram, &manifest).is_ok());
}

#[test]
fn authenticates_engram_bytes_and_refuses_input_changes() {
    let root = Scratch::new();
    let (model, engram) = legacy(&root.0, false, "matching revision");
    let manifest = write_manifest(&root.0, &engram);
    let mut bytes = fs::read(&engram).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&engram, &bytes).unwrap();
    let trusted = TrustedSources::read(&model, "legacy", &engram, &manifest).unwrap();
    let inputs = opened_inputs(&trusted, &engram);
    let sources = inputs.iter().collect::<Vec<_>>();
    let output = root.0.join("would-be-import");
    {
        let _staged = Output::create(&output).unwrap();
        let error = trusted.authenticate(&sources).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("legacy-engram.gguf: source SHA-256 mismatch"),
            "{error:#}"
        );
    }
    assert!(!output.exists());
    bytes.push(0);
    fs::write(&engram, bytes).unwrap();
    let error = trusted.check(&sources).unwrap_err();
    assert!(
        error.to_string().contains("changed during import"),
        "{error:#}"
    );
}

#[cfg(unix)]
#[test]
fn refuses_recovery_journals_beside_a_supplied_name_or_its_symlink_target() {
    let root = Scratch::new();
    let (model, engram) = legacy(&root.0, false, "matching revision");
    let manifest = write_manifest(&root.0, &engram);
    let blobs = root.0.join("blobs");
    fs::create_dir(&blobs).unwrap();
    let shard = root.0.join("legacy-00004-of-00004.gguf");
    let blob = blobs.join("shard-blob");
    fs::rename(&shard, &blob).unwrap();
    std::os::unix::fs::symlink(&blob, &shard).unwrap();
    assert!(TrustedSources::read(&model, "legacy", &engram, &manifest).is_ok());
    for journal in [
        root.0
            .join(".legacy-00004-of-00004.gguf.chat-template-recovery"),
        blobs.join(".shard-blob.chat-template-recovery"),
    ] {
        fs::create_dir(&journal).unwrap();
        let error = TrustedSources::read(&model, "legacy", &engram, &manifest)
            .err()
            .unwrap();
        let detail = format!("{error:#}");
        assert!(
            detail.contains("unfinished chat-template update")
                && detail.contains(journal.file_name().unwrap().to_str().unwrap()),
            "{error:#}"
        );
        fs::remove_dir(&journal).unwrap();
    }
}

#[test]
fn publication_conflict_rolls_back_only_our_links() {
    let root = Scratch::new();
    let out = root.0.join("import");
    let mut output = Output::create(&out).unwrap();
    assert!(!out.join(".hard-link-check").exists());
    let names = ["discoverable", "conflict", "late"].map(str::to_owned);
    for name in &names {
        fs::write(output.stage.join(name), b"verified").unwrap();
    }
    fs::write(out.join("conflict"), b"existing file").unwrap();
    let error = output.publish(&names).unwrap_err();
    assert!(error.to_string().contains("without replacing"), "{error:#}");
    drop(output);
    assert_eq!(fs::read(out.join("conflict")).unwrap(), b"existing file");
    assert_eq!(fs::read_dir(out).unwrap().count(), 1);
}
