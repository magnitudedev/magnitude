//! Weight lineage: whether two GGUFs hold weights of one base model, from a few values of their
//! early normalization weights.
//!
//! Normalization weights are tiny one-dimensional tensors that converters store unquantized, so
//! every quantization of one model carries them bit for bit. Fine-tunes, abliterations and
//! quantization-aware training leave them almost unchanged, at most rescaled, while another
//! generation of the same architecture and size (continued pretraining included) moves them well
//! away. The first values of each early normalization tensor, compared by correlation, which is
//! blind to a rescale, tell the two apart where the shape fingerprint cannot.
//!
//! Measured over the first 64 values of `blk.0`, `blk.1` and `output_norm` normalization weights,
//! as the median of one minus each shared tensor's correlation: every requantization, Ollama blob
//! and abliteration of a catalog target diverges 0; full fine-tunes and quantization-aware
//! training diverge at most 1.2e-4 (a distillation of Qwen3.5 9B); successive generations diverge
//! at least 4.7e-4 (Qwen3.5 to Qwen3.6 35B-A3B), 3e-3 to 1.3e-2 for Qwen3.5, 3.6 and 3.8 27B, and
//! 3.5e-2 for LFM2 to LFM2.5. The limit sits between the two groups on a log scale.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use magnitude_artifacts::gguf::{Directory, Encoding, read_header};
use magnitude_service_contracts::inventory::HuggingFaceRepositorySnapshot;
use magnitude_service_contracts::inventory::InventoryError;
use serde::{Deserialize, Serialize};

use crate::cache::ModelIndexKind;
use crate::inventory::ManagedModelStore;
use crate::preview::RemoteRanges;

/// Values sampled from the start of each tensor.
const SAMPLE_VALUES: u64 = 64;
/// Blocks whose normalization weights are sampled, with the output normalization.
const SAMPLED_BLOCKS: [&str; 2] = ["blk.0.", "blk.1."];
const OUTPUT_NORM: &str = "output_norm.weight";
/// Fewest tensors two samples must share to be compared.
const MIN_SHARED_TENSORS: usize = 3;
/// The largest divergence that is weight lineage (see the module documentation).
const LINEAGE_DIVERGENCE: f64 = 2.5e-4;

/// The first values of a GGUF's early normalization weights, by tensor name. Constant tensors
/// carry no evidence and are left out, so a file whose tensor data is absent (a header extended
/// with zeros) has no sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WeightSample(BTreeMap<String, Vec<f32>>);

/// One byte range of a GGUF file a weight sample reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleRead {
    pub tensor: String,
    pub start: u64,
    pub end: u64,
    encoding: Encoding,
}

impl WeightSample {
    /// The byte ranges of the file `directory` describes (one file's own header, so its data
    /// offset addresses that file) that hold the sampled values: the leading values of each
    /// sampled one-dimensional tensor stored as F32, F16 or BF16.
    #[must_use]
    pub fn reads(directory: &Directory) -> Vec<SampleRead> {
        directory
            .tensors
            .iter()
            .filter(|tensor| is_sampled(&tensor.name) && tensor.shape.len() == 1)
            .filter_map(|tensor| {
                let width = match tensor.encoding {
                    Encoding::F32 => 4,
                    Encoding::F16 | Encoding::BF16 => 2,
                    _ => return None,
                };
                let start = directory.data_offset.checked_add(tensor.offset)?;
                let length = tensor.shape[0].min(SAMPLE_VALUES) * width;
                Some(SampleRead {
                    tensor: tensor.name.clone(),
                    start,
                    end: start.checked_add(length)?,
                    encoding: tensor.encoding,
                })
            })
            .collect()
    }

    /// The sample from the bytes of each read, or `None` when too few tensors carry values.
    #[must_use]
    pub fn from_reads(reads: impl IntoIterator<Item = (SampleRead, Vec<u8>)>) -> Option<Self> {
        let tensors = reads
            .into_iter()
            .filter_map(|(read, bytes)| {
                let values = decode(read.encoding, &bytes)?;
                (!is_constant(&values)).then_some((read.tensor, values))
            })
            .collect::<BTreeMap<_, _>>();
        (tensors.len() >= MIN_SHARED_TENSORS).then_some(Self(tensors))
    }

    /// The sample of the GGUF file `source`, which `directory` describes.
    pub fn read<R: Read + Seek>(
        directory: &Directory,
        source: &mut R,
    ) -> std::io::Result<Option<Self>> {
        let mut parts = Vec::new();
        for read in Self::reads(directory) {
            let mut bytes = vec![0; usize::try_from(read.end - read.start).unwrap_or(0)];
            source.seek(SeekFrom::Start(read.start))?;
            match source.read_exact(&mut bytes) {
                Ok(()) => parts.push((read, bytes)),
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {}
                Err(error) => return Err(error),
            }
        }
        Ok(Self::from_reads(parts))
    }

    /// The sample of the GGUF at `path`, which `directory` describes.
    pub fn read_file(directory: &Directory, path: &Path) -> std::io::Result<Option<Self>> {
        Self::read(directory, &mut std::fs::File::open(path)?)
    }

    /// How far these weights are from `reference`'s: the median over their shared tensors of one
    /// minus the tensors' correlation, or `None` when they share too few tensors to tell.
    #[must_use]
    pub fn divergence(&self, reference: &Self) -> Option<f64> {
        let mut divergences = self
            .0
            .iter()
            .filter_map(|(name, values)| {
                let other = reference
                    .0
                    .get(name)
                    .filter(|other| other.len() == values.len())?;
                Some(1.0 - correlation(values, other))
            })
            .collect::<Vec<_>>();
        if divergences.len() < MIN_SHARED_TENSORS {
            return None;
        }
        divergences.sort_by(f64::total_cmp);
        Some(divergences[divergences.len() / 2])
    }

    /// Whether these are weights of `reference`'s model: the same weights requantized, or
    /// fine-tuned, abliterated or quantization-aware trained from them, never another generation.
    #[must_use]
    pub fn shares_lineage_with(&self, reference: &Self) -> bool {
        self.divergence(reference)
            .is_some_and(|divergence| divergence <= LINEAGE_DIVERGENCE)
    }

    /// The sampled values, by tensor name.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.0
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice()))
    }

    /// The sample's bytes as the release bundle stores them.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a weight sample serializes")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

fn is_sampled(name: &str) -> bool {
    name == OUTPUT_NORM
        || SAMPLED_BLOCKS.iter().any(|block| {
            name.strip_prefix(block)
                .and_then(|rest| rest.strip_suffix(".weight"))
                .is_some_and(|rest| rest.ends_with("norm") && !rest.contains('.'))
        })
}

fn decode(encoding: Encoding, bytes: &[u8]) -> Option<Vec<f32>> {
    let values = match encoding {
        Encoding::F32 => bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("four bytes")))
            .collect::<Vec<_>>(),
        Encoding::F16 => bytes
            .chunks_exact(2)
            .map(|value| f16_to_f32(u16::from_le_bytes([value[0], value[1]])))
            .collect(),
        Encoding::BF16 => bytes
            .chunks_exact(2)
            .map(|value| f32::from_bits(u32::from(u16::from_le_bytes([value[0], value[1]])) << 16))
            .collect(),
        _ => return None,
    };
    (!values.is_empty() && values.iter().all(|value| value.is_finite())).then_some(values)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        0x1f if mantissa == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

fn is_constant(values: &[f32]) -> bool {
    values.iter().all(|value| *value == values[0])
}

/// Pearson correlation; zero when either side is constant, so such a pair counts as unrelated.
fn correlation(left: &[f32], right: &[f32]) -> f64 {
    let count = left.len() as f64;
    let mean = |values: &[f32]| values.iter().map(|value| f64::from(*value)).sum::<f64>() / count;
    let (left_mean, right_mean) = (mean(left), mean(right));
    let (mut covariance, mut left_variance, mut right_variance) = (0.0, 0.0, 0.0);
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left) - left_mean;
        let right = f64::from(*right) - right_mean;
        covariance += left * right;
        left_variance += left * left;
        right_variance += right * right;
    }
    if left_variance == 0.0 || right_variance == 0.0 {
        return 0.0;
    }
    covariance / (left_variance * right_variance).sqrt()
}

/// A Hub file's weight sample as cached with its commit: `None` when it has none.
#[derive(Serialize, Deserialize)]
struct CachedWeightSample {
    sample: Option<WeightSample>,
}

impl ManagedModelStore {
    /// The weight sample of the GGUF at `path`, from its header and a few hundred bytes of its
    /// tensor data.
    pub fn local_weight_sample(&self, path: &Path) -> Result<Option<WeightSample>, InventoryError> {
        let (_, header) = self.local_gguf_header(path)?;
        let directory = read_header(&header)
            .map_err(|error| InventoryError::Integrity(format!("{}: {error}", path.display())))?;
        WeightSample::read_file(&directory, path)
            .map_err(|error| InventoryError::Io(format!("{}: {error}", path.display())))
    }

    /// The weight sample of the Hub file at `path` in `snapshot`: its header (cached with the
    /// commit) and one exact range request per sampled tensor, through the Hub's rate-limit
    /// handling. The answer is cached with the commit, so each file is sampled once.
    pub async fn hub_weight_sample(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        path: &Path,
    ) -> Result<Option<WeightSample>, InventoryError> {
        let evidence = format!(
            "{}@{}:{}{}",
            snapshot.repository,
            snapshot.commit,
            path.display(),
            self.restricted_access_scope(snapshot.gated || snapshot.private)
        );
        if let Some(cached) = self
            .cache
            .read_index::<CachedWeightSample>(ModelIndexKind::HuggingFaceWeightSample, &evidence)
        {
            return Ok(cached.sample);
        }
        let file = snapshot
            .gguf_files
            .iter()
            .find(|file| file.path == path)
            .ok_or_else(|| {
                InventoryError::NotFound(format!(
                    "{} has no {}",
                    snapshot.repository,
                    path.display()
                ))
            })?;
        let header = self
            .hugging_face_gguf_header(&snapshot.repository, &snapshot.commit, file, &[])
            .await?;
        let directory = read_header(&header.bytes)
            .map_err(|error| InventoryError::Integrity(format!("{}: {error}", path.display())))?;
        let token = self.hub_token();
        let mut remote = RemoteRanges::new(
            &self.http,
            token.as_deref(),
            self.client.endpoint(),
            &snapshot.repository,
            &snapshot.commit,
            path,
            file.size_bytes,
        )?;
        let sample = remote_weight_sample(&mut remote, &directory).await?;
        self.cache.write_index(
            ModelIndexKind::HuggingFaceWeightSample,
            &evidence,
            &CachedWeightSample {
                sample: sample.clone(),
            },
        );
        Ok(sample)
    }

    /// The weight sample of the Ollama registry blob `digest` (`sha256:<hex>`, a GGUF model
    /// layer) of the library model `model`: its header and one exact range request per sampled
    /// tensor.
    pub async fn ollama_registry_weight_sample(
        &self,
        model: &str,
        digest: &str,
    ) -> Result<Option<WeightSample>, InventoryError> {
        let url = reqwest::Url::parse(OLLAMA_REGISTRY)
            .and_then(|registry| registry.join(&format!("v2/library/{model}/blobs/{digest}")))
            .map_err(|error| InventoryError::InvalidRequest(error.to_string()))?;
        let mut remote = RemoteRanges::open_url(&self.http, url).await?;
        let header = crate::remote_header::read_header(&mut remote, &[]).await?;
        let directory = read_header(&header)
            .map_err(|error| InventoryError::Integrity(format!("{model}@{digest}: {error}")))?;
        remote_weight_sample(&mut remote, &directory).await
    }
}

/// The Ollama registry, whose library models' layers are blobs addressed by digest.
const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai/";

/// The weight sample of the remote GGUF file `directory` describes, one exact range request per
/// sampled tensor the file holds.
async fn remote_weight_sample(
    remote: &mut RemoteRanges<'_>,
    directory: &Directory,
) -> Result<Option<WeightSample>, InventoryError> {
    let mut parts = Vec::new();
    for read in WeightSample::reads(directory) {
        let (Ok(start), Ok(end)) = (usize::try_from(read.start), usize::try_from(read.end)) else {
            continue;
        };
        if end > remote.logical_length() {
            continue;
        }
        let bytes = remote.read(start, end).await?;
        parts.push((read, bytes));
    }
    Ok(WeightSample::from_reads(parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real file's weight sample (see the assessment fixtures' README).
    fn sample(name: &str) -> WeightSample {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/fixtures/assessment/weight-samples")
            .join(format!("{name}.json"));
        WeightSample::decode(&std::fs::read(path).expect("weight sample fixture"))
            .expect("weight sample")
    }

    /// Requantizations, an Ollama blob and an abliteration of a catalog target carry its sampled
    /// weights bit for bit; a full fine-tune and the model before quantization-aware training
    /// diverge a little. Every one is the catalog target's lineage.
    #[test]
    fn variants_of_one_model_share_its_weight_lineage() {
        for (variant, catalog, identical) in [
            ("ollama-qwen3.6-35b-a3b", "Qwen3.6-35B-A3B-UD-Q4_K_XL", true),
            (
                "MiniCPM5-2B-heretic-abliterated-Q5_K_M",
                "MiniCPM5-2B-Q4_K_M",
                true,
            ),
            ("Qwen_Qwen3.5-9B-IQ4_XS", "Qwen3.5-9B-UD-Q4_K_XL", true),
            (
                "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M",
                "Qwen3.5-9B-UD-Q4_K_XL",
                false,
            ),
            (
                "gemma-4-12B-it-Q4_K_M",
                "gemma-4-12B-it-qat-UD-Q4_K_XL",
                false,
            ),
        ] {
            let (variant_sample, reference) = (sample(variant), sample(catalog));
            let divergence = variant_sample.divergence(&reference).expect("comparable");
            assert_eq!(divergence == 0.0, identical, "{variant}: {divergence:e}");
            assert!(
                variant_sample.shares_lineage_with(&reference),
                "{variant}: {divergence:e}"
            );
        }
    }

    /// Another generation of the same architecture and size, continued pretraining included, has
    /// the catalog target's shape and even its token table, but not its weights.
    #[test]
    fn another_generation_of_one_shape_is_not_its_weight_lineage() {
        for (other, catalog) in [
            ("Qwen3.6-27B-Q4_K_M", "Qwen3.8-27B-UD-Q4_K_XL"),
            ("Qwen3.5-27B-UD-Q4_K_XL", "Qwen3.8-27B-UD-Q4_K_XL"),
            ("Qwen3.5-35B-A3B-UD-Q4_K_XL", "Qwen3.6-35B-A3B-UD-Q4_K_XL"),
        ] {
            let (other_sample, reference) = (sample(other), sample(catalog));
            let divergence = other_sample.divergence(&reference).expect("comparable");
            assert!(
                !other_sample.shares_lineage_with(&reference),
                "{other}: {divergence:e}"
            );
        }
    }

    #[test]
    fn unrelated_models_are_far_apart_and_samples_sharing_too_few_tensors_are_not_compared() {
        let (minicpm, qwen) = (
            sample("MiniCPM5-2B-Q4_K_M"),
            sample("Qwen3.8-27B-UD-Q4_K_XL"),
        );
        assert!(minicpm.divergence(&qwen).expect("comparable") > 0.1);
        let few = WeightSample(
            minicpm
                .0
                .iter()
                .take(2)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );
        assert_eq!(few.divergence(&minicpm), None);
        assert!(!few.shares_lineage_with(&minicpm));
    }

    /// The sample is read from the tensor data the header locates; a file holding the header but
    /// zeros where its tensors go (a header-only preview) has none.
    #[test]
    fn a_file_is_sampled_where_its_header_locates_the_tensors() {
        let header = crate::test_gguf::fixture_header("Qwen3.6-27B-Q4_K_M");
        let directory = read_header(&header).expect("header");
        let expected = sample("Qwen3.6-27B-Q4_K_M");
        let folder = tempfile::tempdir().expect("folder");
        let path = folder.path().join("model.gguf");
        let reads = WeightSample::reads(&directory);
        let end = reads
            .iter()
            .map(|read| read.end)
            .max()
            .expect("sampled tensors");
        let mut bytes = header.clone();
        bytes.resize(usize::try_from(end).unwrap(), 0);
        std::fs::write(&path, &bytes).expect("header-only file");
        assert_eq!(
            WeightSample::read_file(&directory, &path).expect("read"),
            None
        );
        for (tensor, values) in expected.tensors() {
            let read = reads
                .iter()
                .find(|read| read.tensor == tensor)
                .expect("read");
            assert_eq!(read.encoding, Encoding::F32);
            let start = usize::try_from(read.start).unwrap();
            for (index, value) in values.iter().enumerate() {
                bytes[start + index * 4..start + index * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).expect("file");
        assert_eq!(
            WeightSample::read_file(&directory, &path).expect("read"),
            Some(expected)
        );
    }

    #[test]
    fn only_early_one_dimensional_normalization_weights_are_sampled() {
        for (name, sampled) in [
            ("output_norm.weight", true),
            ("blk.0.attn_norm.weight", true),
            ("blk.1.post_attention_norm.weight", true),
            ("blk.1.attn_q_norm.weight", true),
            ("blk.2.attn_norm.weight", false),
            ("blk.10.attn_norm.weight", false),
            ("blk.0.attn_q.weight", false),
            ("blk.0.attn_norm.bias", false),
            ("token_embd.weight", false),
        ] {
            assert_eq!(is_sampled(name), sampled, "{name}");
        }
    }
}
