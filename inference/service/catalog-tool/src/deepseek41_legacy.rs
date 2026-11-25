//! Explicit import of ISTA-DASLab's old four-shard DeepSeek V4.1 GSQ-RCO layout.
//! The result is an ordinary upstream six-shard GGUF, not a second loading path.

mod source;

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use magnitude_artifacts::gguf::{
    self, Array, Directory, Encoding, GgufArtifact, Metadata, Scalar, StoredHeader, StoredMetadata,
    StoredType, TensorSlice, Value, ValueType,
};
use magnitude_artifacts::{ArtifactIdentity, PackageHeaders, PackageIdentity};
use sha2::{Digest, Sha256};
use source::{BACKBONE_SHARDS, TrustedSources};

const ARCH: &str = "deepseek41";
const TABLE: &str = "per_layer_token_embd.weight";
const SHARDS: usize = 6;
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;
const BACKBONE_LIMIT: u64 = 40_000_000_000;
const TEMPLATE: &str = "tokenizer.chat_template";
// The data sidecars at ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF,
// revision 65d53bc8be65123e255ff02f3ded958b6dbf54fd. Never fetched by this tool.
const TEMPLATE_SHA256: &str = "4d61d22c90ea0cdf15c7e298f7215f23eeffd20f8eed8fb517fda5fb64693d99";
const BIAS_SHA256: &str = "710b11267b4db392d1fdb558f36ba8e0e91a91f4e8180c953c912e18eacf843d";
// The vision biases are 40 rows of 384 values, from this checkpoint only.
const BIAS_CHECKPOINT: &str = "1bae0e1088993b9b82de08f59de5be5a34e42170";
const CONSTANTS: [&str; 4] = [
    "engram_token_map.weight",
    "engram_primes.weight",
    "engram_offsets.weight",
    "engram_multipliers.weight",
];
const LEGACY_KEYS: &[&str] = &[
    "attention.candidate_block_size",
    "attention.candidate_topk_blocks",
    "attention.candidate_source_layer_id",
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "dspark_target_layer_ids",
    "expert_scoring_func",
    "engram.present",
    "engram.layer_ids",
    "engram.num_embeddings",
    "engram_head_dim",
    "engram_n_heads",
    "engram_max_ngram_size",
    "engram_vocab_size",
    "engram_compressed_vocab_size",
    "engram_pad_token_id",
    "engram.source",
    "engram.file",
];

#[derive(Debug, clap::Args)]
pub struct Options {
    /// The first of the original four backbone shards.
    first_shard: PathBuf,
    /// Original shared Engram GGUF (read-only).
    #[arg(long)]
    engram: PathBuf,
    /// Local chat_template.jinja from the pinned ISTA-DASLab revision.
    #[arg(long)]
    chat_template: PathBuf,
    /// Local exp_probs_b_vl.f32 from the pinned ISTA-DASLab revision.
    #[arg(long)]
    vision_bias: PathBuf,
    /// Trusted sha256sum manifest naming the four original shards and the original Engram basename.
    #[arg(long)]
    source_sha256: PathBuf,
    /// A new directory for the six verified shards. Must not exist.
    #[arg(long)]
    output: PathBuf,
    /// Check the manifest and layout without authenticating full source files or writing output.
    #[arg(long)]
    dry_run: bool,
}

pub fn run(options: Options) -> Result<()> {
    let prefix = options
        .first_shard
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix("-00001-of-00004.gguf"))
        .filter(|name| !name.is_empty())
        .context("expected the original first backbone shard (*-00001-of-00004.gguf)")?;
    ensure!(
        !options.output.try_exists()?,
        "{} already exists; choose a new output directory",
        options.output.display()
    );
    let trusted = TrustedSources::read(
        &options.first_shard,
        prefix,
        &options.engram,
        &options.source_sha256,
    )?;
    let model = GgufArtifact::open_shards(&trusted.backbone()).with_context(|| {
        format!(
            "open backbone starting at {}",
            options.first_shard.display()
        )
    })?;
    ensure!(
        model.sources().len() == BACKBONE_SHARDS,
        "expected four backbone shards"
    );
    let legacy = Legacy {
        directory: model.directory(),
    };
    ensure!(
        legacy.engram_source()? == BIAS_CHECKPOINT
            && legacy.count("block_count")? == 40
            && legacy.count("expert_count")? == 384,
        "the pinned vision bias sidecar requires the DeepSeek V4.1 Flash 1017 checkpoint"
    );
    let engram_header = gguf::inspect_header(&options.engram)
        .with_context(|| format!("read original Engram header {}", options.engram.display()))?;
    ensure!(
        engram_header.value("split.no").is_none() && engram_header.value("split.count").is_none(),
        "{}: expected the separate, unsplit legacy Engram file",
        options.engram.display()
    );
    let sidecar = GgufArtifact::open(&options.engram)
        .with_context(|| format!("open original Engram file {}", options.engram.display()))?;
    let opened = model
        .sources()
        .chain(sidecar.sources())
        .map(std::sync::Arc::as_ref)
        .collect::<Vec<_>>();
    trusted.check(&opened)?;
    let chat = checked_sidecar(&options.chat_template, TEMPLATE_SHA256)?;
    let bias = checked_sidecar(&options.vision_bias, BIAS_SHA256)?;
    let plan = Plan::build(&model, &sidecar, String::from_utf8(chat)?, &bias)?;
    let shards = plan.shards(BACKBONE_LIMIT)?;
    // Validate the normalized family definition before writing hundreds of GB.
    let header = StoredHeader::new(
        GGUF_VERSION,
        plan.metadata.clone(),
        shards.iter().flatten().map(|t| t.info()),
    )?;
    let directory = gguf::read_header(header.encode()?)?;
    magnitude_family_deepseek41::inspect_components(
        &directory,
        None,
        PackageIdentity {
            target: ArtifactIdentity([0; 32]),
            projector: None,
        },
    )
    .context("normalized DeepSeek V4.1 header failed family validation")?;

    let mut bytes = 0_u64;
    for (index, tensors) in shards.iter().enumerate() {
        let header = plan.header(index, tensors)?;
        let size = header.encode()?.len() as u64
            + tensors
                .iter()
                .map(|t| t.len().next_multiple_of(ALIGNMENT))
                .sum::<u64>();
        bytes += size;
        println!(
            "{}: {} tensors, {size} bytes",
            shard_name(prefix, index),
            tensors.len()
        );
    }
    println!("Output needs {bytes} bytes; inputs are kept unchanged.");
    if options.dry_run {
        println!(
            "Dry run: checked the manifest, headers and small conversion inputs. Full source SHA-256 verification was not run; no output was written."
        );
    } else {
        let mut output = Output::create(&options.output)?;
        trusted.authenticate(&opened)?;
        let names = output.write_shards(&plan, &shards, prefix)?;
        trusted.check(&opened)?;
        let first = output.publish(&names)?;
        println!("Verified and written: {}", first.display());
    }
    Ok(())
}

fn checked_sidecar(path: &Path, digest: &str) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    File::open(path)
        .and_then(|file| file.take(128 * 1024 + 1).read_to_end(&mut data))
        .with_context(|| format!("read data sidecar {}", path.display()))?;
    ensure!(
        data.len() <= 128 * 1024,
        "{} is too large for a data sidecar",
        path.display()
    );
    ensure!(
        format!("{:x}", Sha256::digest(&data)) == digest,
        "{} does not match the pinned data sidecar (SHA-256 {digest})",
        path.display()
    );
    Ok(data)
}

fn key(suffix: &str) -> String {
    format!("{ARCH}.{suffix}")
}

fn scalar(name: &str, kind: ValueType, value: Scalar) -> StoredMetadata {
    StoredMetadata {
        metadata: Metadata {
            name: name.into(),
            value: Value::Scalar(value),
        },
        stored: StoredType::Scalar(kind),
    }
}

fn array(suffix: &str, kind: ValueType, values: impl Into<Array>) -> StoredMetadata {
    StoredMetadata {
        metadata: Metadata {
            name: key(suffix),
            value: Value::Array(values.into()),
        },
        stored: StoredType::Array(kind),
    }
}

struct Legacy<'a> {
    directory: &'a Directory,
}

impl Legacy<'_> {
    fn value(&self, suffix: &str) -> Result<&Value> {
        let key = key(suffix);
        self.directory
            .value(&key)
            .with_context(|| format!("missing {key}"))
    }

    fn count(&self, suffix: &str) -> Result<u64> {
        self.value(suffix)?
            .unsigned()
            .filter(|&n| n > 0)
            .with_context(|| format!("{} must be a positive integer", key(suffix)))
    }

    fn list(&self, suffix: &str) -> Result<Vec<u64>> {
        let Value::Array(values) = self.value(suffix)? else {
            anyhow::bail!("{} must be an integer array", key(suffix));
        };
        values
            .elements()?
            .iter()
            .map(|item| {
                item.unsigned()
                    .with_context(|| format!("{} must contain nonnegative integers", key(suffix)))
            })
            .collect()
    }

    fn engram_source(&self) -> Result<&str> {
        self.value("engram.source")?
            .string()
            .filter(|value| !value.is_empty())
            .context("missing Engram source revision")
    }
}

struct Engram {
    layers: Vec<u64>,
    rows: Vec<u64>,
    width: u64,
    upstream: Vec<StoredMetadata>,
}

impl Engram {
    fn read(model: &GgufArtifact, old: &Legacy<'_>, block_count: u64) -> Result<Self> {
        let layers = old.list("engram.layer_ids")?;
        let rows = old.list("engram.num_embeddings")?;
        let order = old.count("engram_max_ngram_size")?;
        let heads_per_order = old.count("engram_n_heads")?;
        let width = old.count("engram_head_dim")?;
        let heads = order
            .checked_sub(1)
            .and_then(|n| n.checked_mul(heads_per_order))
            .filter(|&n| n > 0)
            .context("invalid Engram head geometry")?;
        ensure!(
            !layers.is_empty()
                && layers.len() == rows.len()
                && layers
                    .iter()
                    .all(|&layer| (1..block_count).contains(&layer))
                && layers.windows(2).all(|pair| pair[0] < pair[1]),
            "invalid Engram layer/row mapping"
        );
        let n = layers.len() as u64;
        // engram_vocab_size is the nominal bucket size, not the vocabulary of
        // the input token map. The map has one entry per tokenizer token.
        let Some(Value::Array(tokens)) = model.directory().value("tokenizer.ggml.tokens") else {
            anyhow::bail!("missing tokenizer.ggml.tokens");
        };
        let vocab = tokens.len() as u64;
        let compressed = old.count("engram_compressed_vocab_size")?;
        ensure!(
            compressed <= vocab,
            "Engram vocabulary does not match the tokenizer"
        );
        let [token_map, primes, offsets, multipliers] = CONSTANTS;
        let map = constant(model, token_map, Encoding::I32, &[vocab])?
            .chunks_exact(4)
            .map(|x| i64::from(i32::from_le_bytes(x.try_into().unwrap())))
            .collect::<Vec<_>>();
        ensure!(
            map.iter()
                .all(|&token| (0..compressed as i64).contains(&token))
                && map.iter().max() == Some(&(compressed as i64 - 1)),
            "invalid compressed Engram token IDs"
        );
        let pad = old
            .value("engram_pad_token_id")?
            .unsigned()
            .filter(|&p| p < vocab)
            .context("invalid Engram padding token")?;
        let primes = nonnegative_constants(model, primes, &[n, heads])?;
        let offsets = nonnegative_constants(model, offsets, &[n, heads])?;
        let multipliers = nonnegative_constants(model, multipliers, &[n, order])?;
        ensure!(
            multipliers.iter().all(|multiplier| multiplier % 2 == 1),
            "Engram multipliers must be positive odd integers"
        );
        let heads = usize::try_from(heads)?;
        let mut global = 0_u64;
        let mut global_offsets = Vec::new();
        for ((primes, offsets), &rows) in primes
            .chunks_exact(heads)
            .zip(offsets.chunks_exact(heads))
            .zip(&rows)
        {
            let start = global;
            for (&prime, &offset) in primes.iter().zip(offsets) {
                ensure!(
                    prime > 0 && offset == global - start,
                    "invalid Engram layer-local offsets"
                );
                global_offsets.push(global);
                global = global
                    .checked_add(prime)
                    .context("Engram offsets overflow")?;
            }
            ensure!(
                global - start == rows,
                "Engram head sizes do not match its layer's rows"
            );
        }
        let upstream = vec![
            array("ple.layers", ValueType::U32, layers.clone()),
            scalar(
                &key("ple.ngram_size"),
                ValueType::U32,
                Scalar::Unsigned(order),
            ),
            scalar(
                &key("ple.heads_per_ngram"),
                ValueType::U32,
                Scalar::Unsigned(heads_per_order),
            ),
            scalar(
                &key("embedding_length_per_layer_input"),
                ValueType::U32,
                Scalar::Unsigned(width),
            ),
            array("ple.layer_multipliers", ValueType::U64, multipliers),
            array("ple.head_offsets", ValueType::U64, global_offsets),
            array("ple.head_vocab_sizes", ValueType::U64, primes),
            scalar(
                &key("engram.pad_token_id"),
                ValueType::U32,
                Scalar::Unsigned(pad),
            ),
            array("engram.token_map", ValueType::I32, map),
        ];
        Ok(Self {
            layers,
            rows,
            width,
            upstream,
        })
    }

    fn table(&self, source: &GgufArtifact) -> Result<Tensor> {
        let directory = source.directory();
        ensure!(
            source.sources().len() == 1 && directory.tensors.len() == self.layers.len(),
            "Engram file must contain one table per legacy Engram layer"
        );
        let mut ranges = Vec::new();
        let mut end = None;
        for (&layer, &rows) in self.layers.iter().zip(&self.rows) {
            let name = format!("blk.{layer}.engram_embd.weight");
            let table = source.tensor(&name)?;
            ensure!(
                table.shape == [rows, self.width] && table.encoding == Encoding::Iq4Nl,
                "{name}: wrong Engram table shape or encoding"
            );
            ensure!(
                end.is_none_or(|end| table.offset == end),
                "{name}: Engram tables must be contiguous in layer order"
            );
            end = Some(
                table
                    .offset
                    .checked_add(table.nbytes)
                    .context("Engram range overflows")?,
            );
            ranges.push(table);
        }
        Ok(Tensor {
            name: TABLE.into(),
            shape: vec![self.rows.iter().sum(), self.width],
            encoding: Encoding::Iq4Nl,
            bytes: Bytes::Copy(ranges),
        })
    }
}

fn constant(
    model: &GgufArtifact,
    name: &str,
    encoding: Encoding,
    shape: &[u64],
) -> Result<Vec<u8>> {
    let tensor = model.tensor(name)?;
    ensure!(
        tensor.encoding == encoding && tensor.shape == shape && tensor.nbytes <= 4 * 1024 * 1024,
        "{name}: invalid Engram constant shape or encoding"
    );
    Ok(tensor.read()?)
}

fn nonnegative_constants(model: &GgufArtifact, name: &str, shape: &[u64]) -> Result<Vec<u64>> {
    constant(model, name, Encoding::I64, shape)?
        .chunks_exact(8)
        .map(|bytes| {
            u64::try_from(i64::from_le_bytes(bytes.try_into().unwrap()))
                .with_context(|| format!("{name}: negative Engram constant"))
        })
        .collect()
}

struct Plan {
    metadata: Vec<StoredMetadata>,
    engram_source: String,
    backbone: Vec<Tensor>,
    table: Tensor,
}

impl Plan {
    fn build(
        model: &GgufArtifact,
        sidecar: &GgufArtifact,
        chat: String,
        bias: &[u8],
    ) -> Result<Self> {
        let directory = model.directory();
        directory.require_execution_byte_order()?;
        sidecar.directory().require_execution_byte_order()?;
        let old = Legacy { directory };
        ensure!(
            directory
                .value("general.architecture")
                .and_then(Value::string)
                == Some(ARCH),
            "the backbone is not DeepSeek V4.1"
        );
        ensure!(
            sidecar
                .directory()
                .value("general.architecture")
                .and_then(Value::string)
                == Some("deepseek41-engram"),
            "expected the original deepseek41-engram sidecar"
        );
        let engram_source = old.engram_source()?;
        ensure!(
            engram_source
                == (Legacy {
                    directory: sidecar.directory()
                })
                .engram_source()?,
            "Engram file is from another checkpoint revision"
        );
        let block_count = old.count("block_count")?;
        let experts = old.count("expert_count")?;
        ensure!(
            block_count
                .checked_mul(experts)
                .and_then(|n| n.checked_mul(4))
                == Some(bias.len() as u64),
            "vision bias sidecar size does not match the backbone"
        );
        let engram = Engram::read(model, &old, block_count)?;
        let table = engram.table(sidecar)?;
        let backbone = backbone(model, &engram.layers, block_count, experts, bias)?;
        let metadata = upstream_metadata(model, &old, engram.upstream, chat)?;
        Ok(Self {
            metadata,
            engram_source: engram_source.into(),
            backbone,
            table,
        })
    }

    fn shards(&self, limit: u64) -> Result<Vec<Vec<&Tensor>>> {
        let mut shards = vec![Vec::new(), Vec::new()]; // shard 1 is metadata only
        let mut used = 0_u64;
        for tensor in &self.backbone {
            let size = tensor.len().next_multiple_of(ALIGNMENT);
            ensure!(
                size <= limit,
                "{} exceeds the backbone shard limit",
                tensor.name
            );
            if size > limit - used {
                shards.push(Vec::new());
                used = 0;
            }
            shards.last_mut().unwrap().push(tensor);
            used += size;
        }
        ensure!(
            shards.len() == SHARDS - 1,
            "expected four backbone data shards (got {})",
            shards.len() - 1
        );
        shards.push(vec![&self.table]);
        Ok(shards)
    }

    fn header(&self, index: usize, tensors: &[&Tensor]) -> Result<StoredHeader> {
        let mut metadata = match index {
            0 => self.metadata.clone(),
            index if index == SHARDS - 1 => vec![
                scalar(
                    "general.name",
                    ValueType::String,
                    Scalar::String("DeepSeek-V4.1-Flash-1017 Engram tables".into()),
                ),
                scalar(
                    "general.description",
                    ValueType::String,
                    Scalar::String("Shard 6 of 6 of every GSQ-RCO build".into()),
                ),
                scalar(
                    &key("engram.source"),
                    ValueType::String,
                    Scalar::String(self.engram_source.clone()),
                ),
            ],
            _ => Vec::new(),
        };
        metadata.extend([
            scalar("split.no", ValueType::U16, Scalar::Unsigned(index as u64)),
            scalar(
                "split.count",
                ValueType::U16,
                Scalar::Unsigned(SHARDS as u64),
            ),
            scalar(
                "split.tensors.count",
                ValueType::I32,
                Scalar::Signed(i64::try_from(self.backbone.len())? + 1),
            ),
        ]);
        Ok(StoredHeader::new(
            GGUF_VERSION,
            metadata,
            tensors.iter().map(|tensor| tensor.info()),
        )?)
    }
}

fn upstream_metadata(
    model: &GgufArtifact,
    old: &Legacy<'_>,
    engram: Vec<StoredMetadata>,
    chat: String,
) -> Result<Vec<StoredMetadata>> {
    let source = model.sources().next().context("missing backbone source")?;
    let stored = StoredHeader::read(&mut source.reader())?;
    let candidate = old.value("attention.candidate_source_layer_id")?;
    let candidate = match candidate {
        Value::Scalar(Scalar::Signed(-1)) => Scalar::Signed(-1),
        _ => Scalar::Signed(i64::try_from(
            candidate
                .unsigned()
                .context("invalid candidate pool source layer")?,
        )?),
    };
    let mut replacement = vec![
        array(
            "attention.kv_source_layers",
            ValueType::U32,
            old.list("kv_source_layer_ids")?,
        ),
        array(
            "attention.indexer.source_layers",
            ValueType::U32,
            old.list("index_source_layer_ids")?,
        ),
        scalar(
            &key("attention.indexer.candidate_source_layer"),
            ValueType::I32,
            candidate,
        ),
        scalar(
            &key("attention.indexer.candidate_top_k_blocks"),
            ValueType::U32,
            Scalar::Unsigned(old.count("attention.candidate_topk_blocks")?),
        ),
        scalar(
            &key("attention.indexer.candidate_block_size"),
            ValueType::U32,
            Scalar::Unsigned(old.count("attention.candidate_block_size")?),
        ),
    ];
    replacement.extend(engram);
    let mut metadata = Vec::new();
    for entry in stored.metadata {
        let name = entry.metadata.name.as_str();
        if name.starts_with("split.") {
            continue;
        }
        if name
            .strip_prefix("deepseek41.")
            .is_some_and(|suffix| LEGACY_KEYS.contains(&suffix))
        {
            metadata.append(&mut replacement);
        } else if name == TEMPLATE {
            metadata.push(scalar(
                TEMPLATE,
                ValueType::String,
                Scalar::String(chat.clone()),
            ));
        } else if name == "general.alignment" {
            metadata.push(scalar(name, ValueType::U32, Scalar::Unsigned(ALIGNMENT)));
        } else {
            metadata.push(entry);
        }
    }
    let mut names = BTreeSet::new();
    for entry in &metadata {
        ensure!(
            names.insert(entry.metadata.name.as_str()),
            "mixed legacy/upstream metadata key {:?}",
            entry.metadata.name
        );
    }
    ensure!(
        names.contains(TEMPLATE),
        "legacy GGUF is missing {TEMPLATE}"
    );
    Ok(metadata)
}

struct Tensor {
    name: String,
    shape: Vec<u64>,
    encoding: Encoding,
    bytes: Bytes,
}

enum Bytes {
    Copy(Vec<TensorSlice>),
    Generated(Vec<u8>),
}

impl Tensor {
    fn len(&self) -> u64 {
        match &self.bytes {
            Bytes::Copy(parts) => parts.iter().map(|part| part.nbytes).sum(),
            Bytes::Generated(bytes) => bytes.len() as u64,
        }
    }

    fn info(&self) -> (String, Vec<u64>, Encoding) {
        (self.name.clone(), self.shape.clone(), self.encoding)
    }

    fn stream(&self, mut sink: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        match &self.bytes {
            Bytes::Copy(parts) => {
                for part in parts {
                    part.source.stream(
                        part.offset,
                        part.nbytes,
                        seismic_io::Want::now(),
                        |_, data| sink(data),
                    )?;
                }
                Ok(())
            }
            Bytes::Generated(bytes) => sink(bytes),
        }
    }
}

fn block_tensor(name: &str) -> Option<(u64, &str)> {
    let (block, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
    Some((block.parse().ok()?, suffix))
}

fn backbone(
    model: &GgufArtifact,
    engram: &[u64],
    blocks: u64,
    experts: u64,
    bias: &[u8],
) -> Result<Vec<Tensor>> {
    let mut output = Vec::new();
    for entry in &model.directory().tensors {
        if CONSTANTS.contains(&entry.name.as_str()) {
            continue;
        }
        if let Some((layer, "engram_q.weight" | "engram_k.weight")) = block_tensor(&entry.name) {
            ensure!(
                engram.contains(&layer),
                "unexpected Engram gate input {}",
                entry.name
            );
            continue;
        }
        let mut shape = entry.shape.clone();
        if entry.name.ends_with(".attn_output_a.weight")
            && let [groups, rank, width] = shape[..]
        {
            shape = vec![
                groups
                    .checked_mul(rank)
                    .context("attention output shape overflows")?,
                width,
            ];
        }
        output.push(Tensor {
            name: entry.name.replace(".engram_wkv.", ".engram_kv."),
            encoding: entry.encoding,
            shape,
            bytes: Bytes::Copy(vec![model.tensor(&entry.name)?]),
        });
    }
    for &layer in engram {
        output.push(gate(model, layer)?);
    }
    for (layer, row) in (0..blocks).zip(bias.chunks_exact(usize::try_from(experts * 4)?)) {
        let name = format!("blk.{layer}.exp_probs_b.bias");
        ensure!(
            model
                .directory()
                .tensor(&name)
                .is_some_and(|tensor| tensor.shape == [experts]),
            "missing or invalid {name}"
        );
        output.push(Tensor {
            name: format!("blk.{layer}.exp_probs_b_vl.bias"),
            shape: vec![experts],
            encoding: Encoding::F32,
            bytes: Bytes::Generated(row.to_vec()),
        });
    }
    output.sort_unstable_by(|a, b| {
        (block_tensor(&a.name).map(|(block, _)| block), &a.name)
            .cmp(&(block_tensor(&b.name).map(|(block, _)| block), &b.name))
    });
    for pair in output.windows(2) {
        ensure!(
            pair[0].name != pair[1].name,
            "duplicate normalized tensor {}",
            pair[0].name
        );
    }
    ensure!(
        !output.iter().any(|tensor| tensor.name == TABLE),
        "backbone already contains an upstream Engram table"
    );
    Ok(output)
}

fn gate(model: &GgufArtifact, layer: u64) -> Result<Tensor> {
    let q = model.tensor(&format!("blk.{layer}.engram_q.weight"))?;
    let k = model.tensor(&format!("blk.{layer}.engram_k.weight"))?;
    ensure!(
        q.encoding == Encoding::BF16
            && k.encoding == Encoding::BF16
            && q.shape == k.shape
            && q.shape.len() == 2
            && q.nbytes <= 4 * 1024 * 1024,
        "blk.{layer} Engram q and k must be BF16 matrices of the same shape"
    );
    let read = |bytes: &[u8]| {
        f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16)
    };
    let mut product = Vec::new();
    for (q, k) in q.read()?.chunks_exact(2).zip(k.read()?.chunks_exact(2)) {
        let value = read(q) * read(k);
        ensure!(
            value.is_finite(),
            "blk.{layer} Engram gate contains a nonfinite product"
        );
        product.extend(value.to_le_bytes());
    }
    Ok(Tensor {
        name: format!("blk.{layer}.engram_gate.weight"),
        shape: q.shape,
        encoding: Encoding::F32,
        bytes: Bytes::Generated(product),
    })
}

fn shard_name(prefix: &str, index: usize) -> String {
    format!("{prefix}-{:05}-of-{SHARDS:05}.gguf", index + 1)
}

fn write_shard(path: &Path, header: &StoredHeader, tensors: &[&Tensor]) -> Result<()> {
    let mut writer = BufWriter::new(File::options().create_new(true).write(true).open(path)?);
    let mut digest = Sha256::new();
    let mut write = |bytes: &[u8]| -> Result<()> {
        writer.write_all(bytes)?;
        digest.update(bytes);
        Ok(())
    };
    write(&header.encode()?)?;
    for (descriptor, tensor) in header.tensors.iter().zip(tensors) {
        ensure!(
            descriptor.nbytes == tensor.len(),
            "{}: payload does not fit output shape",
            tensor.name
        );
        tensor.stream(&mut write)?;
        let padding = tensor.len().next_multiple_of(ALIGNMENT) - tensor.len();
        write(&[0; ALIGNMENT as usize][..padding as usize])?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    verify_shard(path, &digest.finalize())
}

fn verify_shard(path: &Path, expected: &[u8]) -> Result<()> {
    let copied = magnitude_artifacts::FileSource::open(path)
        .with_context(|| format!("open staged shard {} for verification", path.display()))?;
    ensure!(
        &source::sha256(&copied)?[..] == expected,
        "{}: verification failed on reread",
        path.display()
    );
    Ok(())
}

struct Output<'a> {
    root: &'a Path,
    stage: PathBuf,
    linked: Vec<PathBuf>,
    complete: bool,
}

impl<'a> Output<'a> {
    fn create(root: &'a Path) -> Result<Self> {
        fs::create_dir(root)
            .with_context(|| format!("create new output directory {}", root.display()))?;
        let stage = root.join(".partial");
        if let Err(error) = fs::create_dir(&stage) {
            let _ = fs::remove_dir(root); // only if still empty
            return Err(error)
                .with_context(|| format!("create staging directory {}", stage.display()));
        }
        let mut output = Self {
            root,
            stage,
            linked: Vec::new(),
            complete: false,
        };
        // Check capability in the actual output directory before reading or
        // writing hundreds of GB. Use the same no-clobber link as publication.
        let probe = ".hard-link-check";
        let source = output.stage.join(probe);
        File::options()
            .create_new(true)
            .write(true)
            .open(&source)
            .with_context(|| format!("create hard-link probe {}", source.display()))?;
        output
            .link(probe)
            .with_context(|| format!("{} must support creating hard links", root.display()))?;
        let destination = output.root.join(probe);
        fs::remove_file(&destination)
            .with_context(|| format!("remove hard-link probe {}", destination.display()))?;
        output.linked.pop();
        fs::remove_file(&source)
            .with_context(|| format!("remove hard-link probe {}", source.display()))?;
        Ok(output)
    }

    fn write_shards(
        &self,
        plan: &Plan,
        shards: &[Vec<&Tensor>],
        prefix: &str,
    ) -> Result<Vec<String>> {
        let names = (0..SHARDS)
            .map(|index| shard_name(prefix, index))
            .collect::<Vec<_>>();
        for (index, tensors) in shards.iter().enumerate() {
            let path = self.stage.join(&names[index]);
            println!("Writing and verifying {}", path.display());
            write_shard(&path, &plan.header(index, tensors)?, tensors)
                .with_context(|| format!("stage shard {}", path.display()))?;
        }
        let first = self.stage.join(&names[0]);
        PackageHeaders::open(&first, None)
            .with_context(|| format!("open staged package {}", first.display()))?;
        Ok(names)
    }

    fn link(&mut self, name: &str) -> Result<()> {
        let path = self.root.join(name);
        fs::hard_link(self.stage.join(name), &path).with_context(|| {
            format!(
                "publish {} without replacing any existing path",
                path.display()
            )
        })?;
        self.linked.push(path);
        Ok(())
    }

    fn publish(&mut self, names: &[String]) -> Result<PathBuf> {
        // A waiting scanner must not see the first shard until its siblings
        // exist. On a returned error, Drop unlinks only paths we created.
        for name in names.iter().rev() {
            self.link(name)?;
        }
        self.complete = true;
        Ok(self.root.join(&names[0]))
    }
}

impl Drop for Output<'_> {
    fn drop(&mut self) {
        if !self.complete {
            for path in self.linked.iter().rev() {
                if let Err(error) = fs::remove_file(path) {
                    eprintln!(
                        "Could not remove incomplete output {}: {error}",
                        path.display()
                    );
                }
            }
        }
        if let Err(error) = fs::remove_dir_all(&self.stage) {
            eprintln!(
                "Could not remove staging directory {}: {error}",
                self.stage.display()
            );
        }
        if !self.complete {
            let _ = fs::remove_dir(self.root); // leave any unrelated files alone
        }
    }
}

#[cfg(test)]
mod tests;
