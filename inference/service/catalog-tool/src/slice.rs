//! Truncated real-weight GGUFs for models too large to run whole.
//!
//! Reads the header of every shard and fetches only the bytes of the tensors it keeps: the
//! global tensors (embedding, final norm, head, rope tables, ...), the selected main layers
//! renumbered 0..n-1, and the trailing MTP (`nextn`) layers renumbered after them. Metadata is
//! copied unchanged, each value in its stored type, except `block_count`, per-layer arrays
//! (entries of the kept layers) and the dropped `split.*` keys, so the slice is recognized as the
//! same family.
//!
//! Default selection: the shortest prefix of the main layers that contains every layer *variant*
//! and at least `--minimum` layers. A variant is a layer's tensor names and shapes plus its
//! entries in every per-layer metadata array; tensor types are not part of it. A prefix keeps
//! implicit periodic layer patterns (a sliding-window period, a full-attention interval, a
//! dense-first layer) intact, which `--minimum 4` covers for the in-scope families. `--layers`
//! picks an explicit ascending list instead; the caller then owns the pattern keys. `--list`
//! prints the variants and the default selection.
//!
//! A provenance record `<output>.json` (the output's extension replaced) lists the source files,
//! kept layers, renaming and output sha256.

use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Write};
use std::path::PathBuf;

use anyhow::Context;
use futures_util::{StreamExt, stream};
use magnitude_artifacts::gguf::{Array, Element, Scalar, StoredHeader, StoredMetadata, Value};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::gguf_source::{Gguf, GgufSource};

/// Metadata counting trailing main layers (Gemma 4's KV-shared layers). A slice keeps a trailing
/// layer's role only when every kept layer after it is also trailing, which a prefix-plus-tail
/// selection guarantees and `--layers` must respect.
const TRAILING_COUNTS: [&str; 1] = ["attention.shared_kv_layers"];
/// Global tensors that pack one slice per layer (Gemma 4 per-layer embeddings); a slice would
/// have to cut them too, so such models are only sliced with every main layer kept (they are
/// small anyway).
const LAYER_PACKED: [&str; 2] = ["per_layer_token_embd.weight", "per_layer_model_proj.weight"];
/// The GGUF version a slice is written in.
const SLICE_VERSION: u32 = 3;

pub struct Options {
    pub source: GgufSource,
    pub output: Option<PathBuf>,
    pub layers: Option<Vec<usize>>,
    pub minimum: usize,
    pub no_mtp: bool,
    pub list: bool,
    pub parallel: usize,
}

/// A layer tensor's layer and the name after `blk.<layer>.`.
fn layer_of(name: &str) -> Option<(usize, &str)> {
    let (layer, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    (!layer.is_empty() && layer.bytes().all(|byte| byte.is_ascii_digit()) && !rest.is_empty())
        .then(|| Some((layer.parse().ok()?, rest)))
        .flatten()
}

/// A per-layer metadata entry compared as a value: floats by their bits, as their JSON text
/// differs exactly when their bits do.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Entry {
    String(String),
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Float(u64),
}

impl From<Element<'_>> for Entry {
    fn from(element: Element<'_>) -> Self {
        match element {
            Element::String(value) => Self::String(value.to_owned()),
            Element::Bool(value) => Self::Bool(value),
            Element::Unsigned(value) => Self::Unsigned(value),
            Element::Signed(value) => Self::Signed(value),
            Element::Float(value) => Self::Float(value.to_bits()),
        }
    }
}

/// A layer's tensor names (after `blk.<layer>.`, sorted) and shapes, and its entries in every
/// per-layer metadata array.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Variant {
    tensors: Vec<(String, Vec<u64>)>,
    metadata: Vec<Entry>,
}

/// What a layer completes for coverage: its variant, and the variant pair it forms with the
/// previous layer (a sublayer family such as Nemotron-H pairs consecutive layers into blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Unit {
    Variant(usize),
    Pair(usize, usize),
}

struct Slicer<'a> {
    gguf: &'a Gguf,
    architecture: String,
    total: usize,
    mtp: usize,
    main: usize,
    per_layer_keys: Vec<String>,
    layer_tensors: HashMap<usize, Vec<String>>,
    global_tensors: Vec<String>,
    /// Distinct variants in order of their first layer, with their layers.
    variants: Vec<(Variant, Vec<usize>)>,
    /// Each main layer's index in `variants`.
    layer_variants: Vec<usize>,
}

impl<'a> Slicer<'a> {
    fn new(gguf: &'a Gguf) -> anyhow::Result<Self> {
        let first = gguf.first();
        let architecture = first
            .value("general.architecture")
            .and_then(Value::string)
            .context("missing metadata general.architecture")?
            .to_owned();
        let key = |suffix: &str| first.value(&format!("{architecture}.{suffix}"));
        let total = usize::try_from(
            key("block_count")
                .and_then(Value::unsigned)
                .with_context(|| format!("missing metadata {architecture}.block_count"))?,
        )?;
        let mtp = usize::try_from(
            key("nextn_predict_layers")
                .map(|value| {
                    value
                        .unsigned()
                        .context("nextn_predict_layers is not a count")
                })
                .transpose()?
                .unwrap_or(0),
        )?;
        let main = total
            .checked_sub(mtp)
            .context("more MTP layers than layers")?;
        let per_layer_keys = first
            .metadata
            .iter()
            .filter(|entry| {
                matches!(&entry.value, Value::Array(array) if array.len() == total)
                    && entry.name.starts_with(&format!("{architecture}."))
            })
            .map(|entry| entry.name.clone())
            .collect::<Vec<_>>();
        let mut layer_tensors = HashMap::<usize, Vec<String>>::new();
        let mut global_tensors = Vec::new();
        for tensor in gguf.tensors() {
            match layer_of(&tensor.name) {
                Some((layer, _)) => layer_tensors
                    .entry(layer)
                    .or_default()
                    .push(tensor.name.clone()),
                None => global_tensors.push(tensor.name.clone()),
            }
        }
        let mut slicer = Self {
            gguf,
            architecture,
            total,
            mtp,
            main,
            per_layer_keys,
            layer_tensors,
            global_tensors,
            variants: Vec::new(),
            layer_variants: Vec::new(),
        };
        for layer in 0..main {
            let variant = slicer.variant(layer)?;
            let index = match slicer
                .variants
                .iter()
                .position(|(known, _)| *known == variant)
            {
                Some(index) => index,
                None => {
                    slicer.variants.push((variant, Vec::new()));
                    slicer.variants.len() - 1
                }
            };
            slicer.variants[index].1.push(layer);
            slicer.layer_variants.push(index);
        }
        Ok(slicer)
    }

    fn layer(&self, layer: usize) -> anyhow::Result<&[String]> {
        self.layer_tensors
            .get(&layer)
            .map(Vec::as_slice)
            .with_context(|| format!("layer {layer} has no tensors"))
    }

    fn variant(&self, layer: usize) -> anyhow::Result<Variant> {
        let mut tensors = self
            .layer(layer)?
            .iter()
            .map(|name| {
                let tensor = self.gguf.tensor(name).expect("a listed tensor");
                let (_, rest) = layer_of(name).expect("a layer tensor");
                (rest.to_owned(), tensor.shape.clone())
            })
            .collect::<Vec<_>>();
        tensors.sort();
        let metadata = self
            .per_layer_keys
            .iter()
            .map(|key| match self.gguf.first().value(key) {
                Some(Value::Array(array)) => {
                    Entry::from(array.get(layer).expect("a per-layer entry"))
                }
                _ => unreachable!("a per-layer key names an array"),
            })
            .collect();
        Ok(Variant { tensors, metadata })
    }

    fn units(&self, layer: usize) -> Vec<Unit> {
        let variant = self.layer_variants[layer];
        let mut units = vec![Unit::Variant(variant)];
        if layer > 0 {
            units.push(Unit::Pair(self.layer_variants[layer - 1], variant));
        }
        units
    }

    fn default_layers(&self, minimum: usize) -> Vec<usize> {
        let wanted = (0..self.main)
            .flat_map(|layer| self.units(layer))
            .collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        for layer in 0..self.main {
            seen.extend(self.units(layer));
            if seen == wanted && layer + 1 >= minimum {
                return (0..=layer).collect();
            }
        }
        (0..self.main).collect()
    }

    /// The slice's metadata: the first shard's, each value in its stored type, without
    /// `split.*`, with `block_count`, the per-layer arrays and the trailing-layer counts of the
    /// `kept` layers.
    fn metadata(
        &self,
        stored: StoredHeader,
        kept: &[usize],
        no_mtp: bool,
    ) -> anyhow::Result<Vec<StoredMetadata>> {
        let architecture = &self.architecture;
        let trailing = TRAILING_COUNTS.map(|suffix| format!("{architecture}.{suffix}"));
        let count = |value: usize| Value::Scalar(Scalar::Unsigned(value as u64));
        let mut metadata = stored
            .metadata
            .into_iter()
            .filter(|entry| !entry.metadata.name.starts_with("split."))
            .collect::<Vec<_>>();
        for entry in &mut metadata {
            let name = entry.metadata.name.as_str();
            let value = &mut entry.metadata.value;
            if name == format!("{architecture}.block_count") {
                *value = count(kept.len());
            } else if self.per_layer_keys.iter().any(|key| key == name) {
                let Value::Array(array) = value else {
                    unreachable!("a per-layer key names an array")
                };
                *value =
                    Value::Array(Array::of(kept.iter().map(|layer| {
                        array.get(*layer).expect("a per-layer entry").to_scalar()
                    }))?);
            } else if trailing.iter().any(|key| key == name) {
                // Counts of trailing layers keep meaning "the last n layers": recount over the
                // kept layers.
                let first = self.main
                    - usize::try_from(value.unsigned().context("a trailing-layer count")?)?;
                *value = count(
                    kept.iter()
                        .filter(|layer| (first..self.main).contains(*layer))
                        .count(),
                );
            } else if no_mtp
                && self.mtp > 0
                && name == format!("{architecture}.nextn_predict_layers")
            {
                *value = count(0);
            }
        }
        Ok(metadata)
    }

    /// (source name, output name) in output order: globals first, then layers in the new order.
    fn tensors(&self, kept: &[usize]) -> anyhow::Result<Vec<(String, String)>> {
        let mut pairs = self
            .global_tensors
            .iter()
            .map(|name| (name.clone(), name.clone()))
            .collect::<Vec<_>>();
        for (new, old) in kept.iter().enumerate() {
            for name in self.layer(*old)? {
                let (_, rest) = layer_of(name).expect("a layer tensor");
                pairs.push((name.clone(), format!("blk.{new}.{rest}")));
            }
        }
        Ok(pairs)
    }
}

pub async fn run(options: Options) -> anyhow::Result<()> {
    let gguf = Gguf::open(&options.source).await?;
    let slicer = Slicer::new(&gguf)?;
    let main_layers = match options.layers {
        Some(layers) => layers,
        None => slicer.default_layers(options.minimum),
    };
    anyhow::ensure!(
        main_layers.windows(2).all(|pair| pair[0] < pair[1])
            && main_layers.last().is_some_and(|last| *last < slicer.main),
        "--layers must be ascending main-layer indices below {}",
        slicer.main
    );
    let packed = LAYER_PACKED
        .into_iter()
        .filter(|name| gguf.tensor(name).is_some())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        packed.is_empty() || main_layers.len() == slicer.main,
        "{} pack every layer; keep all main layers",
        packed.join(", ")
    );
    if options.list {
        let variants = slicer
            .variants
            .iter()
            .map(|(variant, members)| {
                let tensors = variant.tensors.iter().map(|(name, _)| name);
                json!({ "layers": members, "tensors": tensors.collect::<Vec<_>>() })
            })
            .collect::<Vec<_>>();
        let listing = json!({
            "architecture": slicer.architecture,
            "main_layers": slicer.main,
            "mtp_layers": slicer.mtp,
            "per_layer_keys": slicer.per_layer_keys,
            "variants": variants,
            "default_layers": main_layers,
        });
        println!("{}", serde_json::to_string_pretty(&listing)?);
        return Ok(());
    }
    let output = options
        .output
        .context("--output is required unless --list")?;
    let missing = slicer
        .variants
        .iter()
        .filter(|(_, members)| !members.iter().any(|layer| main_layers.contains(layer)))
        .map(|(_, members)| members[0])
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        println!(
            "note: layer variants not in the selection (first layers): [{}]",
            missing
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let mtp = if options.no_mtp {
        Vec::new()
    } else {
        (slicer.main..slicer.total).collect()
    };
    let kept = [main_layers.as_slice(), &mtp].concat();
    let stored = StoredHeader::read(&mut Cursor::new(&gguf.shards[0].header))?;
    let metadata = slicer.metadata(stored, &kept, options.no_mtp)?;
    let pairs = slicer.tensors(&kept)?;
    let header = StoredHeader::new(
        SLICE_VERSION,
        metadata,
        pairs.iter().map(|(old, new)| {
            let tensor = gguf.tensor(old).expect("a listed tensor");
            (new.clone(), tensor.shape.clone(), tensor.encoding)
        }),
    )?;
    let alignment = usize::try_from(header.alignment()?)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::io::BufWriter::new(std::fs::File::create(&output)?);
    let mut digest = Sha256::new();
    let mut write = |bytes: &[u8]| -> std::io::Result<()> {
        digest.update(bytes);
        file.write_all(bytes)
    };
    write(&header.encode()?)?;
    let mut tensor_bytes = 0u64;
    // At most `parallel` tensors are in flight or waiting to be written, so memory stays bounded.
    let gguf = &gguf;
    let mut reads = stream::iter(&pairs)
        .map(|(old, new)| async move {
            Ok::<_, anyhow::Error>((old, new, gguf.tensor_bytes(old).await?))
        })
        .buffered(options.parallel);
    while let Some(read) = reads.next().await {
        let (old, new, data) = read?;
        write(&data)?;
        write(&vec![
            0;
            data.len().next_multiple_of(alignment) - data.len()
        ])?;
        tensor_bytes += data.len() as u64;
        println!(
            "{old} -> {new} ({:.1} MiB)",
            data.len() as f64 / f64::from(1 << 20)
        );
    }
    file.flush()?;
    drop(file);
    let sha256 = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut record = json!({
        "source": gguf.describe(),
        "architecture": slicer.architecture,
        "main_layers": main_layers,
        "mtp_layers": mtp,
        "tensor_bytes": tensor_bytes,
        "output": output.display().to_string(),
        "sha256": sha256,
    });
    let summary = serde_json::to_string_pretty(&record)?;
    record["renaming"] = kept
        .iter()
        .enumerate()
        .map(|(new, old)| (old.to_string(), json!(new)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    std::fs::write(
        output.with_extension("json"),
        serde_json::to_string_pretty(&record)? + "\n",
    )?;
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_tensors_are_named_by_their_layer() {
        assert_eq!(
            layer_of("blk.12.attn_q.weight"),
            Some((12, "attn_q.weight"))
        );
        assert_eq!(
            layer_of("blk.0.nextn.eh_proj.weight"),
            Some((0, "nextn.eh_proj.weight"))
        );
        assert_eq!(layer_of("token_embd.weight"), None);
        assert_eq!(layer_of("blk.x.attn_q.weight"), None);
        assert_eq!(layer_of("blk.3."), None);
    }
}
