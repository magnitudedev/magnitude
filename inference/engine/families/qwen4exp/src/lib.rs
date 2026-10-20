//! Qwen4 experimental (`qwen4exp`, Qwen3.8 Flash Next) artifact recognition
//! and numerical interpretation.
//!
//! The decoder is the Qwen 3.5 routed decoder built from the shared Qwen
//! builders, with four differences read from the header: hyper-connection
//! streams around every sublayer, a sigmoid gated delta output gate, selected
//! attention history scored by an indexer, and hashed n-gram rows gated into
//! the streams at the layers `ple.layers` names.

mod family;

pub use family::Qwen4ExpFamily;

use magnitude_artifacts::{
    Package, PackageIdentity,
    gguf::{Directory, Value},
};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    ActivationDType, Block, Decoder, EmbeddingScale, EntryForm, ExitForm, ExitNorm, FamilyId,
    GateFunction, HeadComponent, HistoryReads, HyperConnectionMixing, HyperConnectionWeights,
    HyperConnections, Indexer, InputNorm, InputSemantics, ModelDefinition, NGramHash, NGramInput,
    Operator, OutputForm, ResidualForm, Sublayer, WeightDescriptor,
};
use magnitude_family_qwen::{
    GEOMETRY_KEYS, Geometry, MIXER_KEYS, Metadata, MixerKind, ROUTED_KEYS, Weight,
    check_projector_width, describe_projector, mixer_kinds, prediction_blocks, product,
    stores_prediction_blocks,
};

pub use magnitude_family_qwen::Error;

mod head;
pub use head::inspect_head;

pub const FAMILY_ID: &str = "qwen4exp";

/// The family ids of every Qwen4 experimental architecture.
pub const FAMILY_IDS: &[&str] = &[FAMILY_ID];

/// The keys a target interprets beside the shared Qwen geometry, mixer and
/// routed keys.
const TARGET_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "nextn_predict_layers",
    "attention.compress_ratios",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "embedding_length_per_layer_input",
    "hyper_connection.count",
    "hyper_connection.low_rank",
    "ple.conv_kernel",
    "ple.eos_token_id",
    "ple.image_token_id",
    "ple.head_offsets",
    "ple.head_vocab_sizes",
    "ple.heads_per_ngram",
    "ple.layer_multipliers",
    "ple.layers",
    "ple.ngram_size",
];

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

pub fn recognize(directory: &Directory) -> Result<(), Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
        .ok_or_else(|| invalid("missing GGUF architecture"))?
    {
        FAMILY_ID => Ok(()),
        architecture => Err(invalid(format!(
            "unsupported Qwen4 GGUF architecture {architecture:?}"
        ))),
    }
}

pub fn inspect_package(package: &Package) -> Result<ModelDefinition, Error> {
    inspect_components(
        package.target().directory(),
        package.projector().map(|projector| projector.directory()),
        package.identity(),
    )
}

/// The header facts beyond the Qwen 3.5 geometry.
struct Extensions {
    streams: u64,
    rank: u64,
    indexer_heads: u64,
    indexer_width: u64,
    /// Selected tokens per row.
    selected: u64,
    /// Per layer, its selection block (0 where it has none).
    blocks: Vec<u64>,
    ngram: NGramSite,
}

struct NGramSite {
    layer: u64,
    hash: NGramHash,
    width: u64,
    kernel: u64,
    dilation: u64,
}

impl Extensions {
    /// `predictions` blocks follow the decoder's layers; their ratios are
    /// the prediction head's to check.
    fn read(m: &Metadata, layers: &[MixerKind], predictions: u64) -> Result<Self, Error> {
        let blocks = m.integers("attention.compress_ratios")?;
        if blocks.len() as u64 != layers.len() as u64 + predictions
            || layers
                .iter()
                .zip(&blocks)
                .any(|(kind, block)| (*kind == MixerKind::Attention) != (*block > 0))
        {
            return Err(invalid(
                "qwen4exp compress ratios must name exactly the attention layers",
            ));
        }
        let ngram_layers = m.integers("ple.layers")?;
        let [layer] = ngram_layers[..] else {
            return Err(invalid("qwen4exp must name exactly one n-gram layer"));
        };
        if layer >= layers.len() as u64 {
            return Err(invalid("qwen4exp n-gram layer exceeds the block count"));
        }
        let ngram_size = m.integer("ple.ngram_size")?;
        if ngram_size < 2 {
            return Err(invalid("qwen4exp n-grams span at least two tokens"));
        }
        let hash = NGramHash {
            context: ngram_size - 1,
            heads_per_order: m.integer("ple.heads_per_ngram")?,
            multipliers: m.integers("ple.layer_multipliers")?,
            sizes: m.integers("ple.head_vocab_sizes")?,
            offsets: m.integers("ple.head_offsets")?,
            reset: m
                .value("ple.eos_token_id")?
                .unsigned()
                .ok_or_else(|| m.malformed("ple.eos_token_id", "an integer"))?,
        };
        // Media rows hash the token that stands in their place, which this
        // key names.
        if m.optional("ple.image_token_id")
            .is_some_and(|token| token.unsigned().is_none())
        {
            return Err(m.malformed("ple.image_token_id", "an integer"));
        }
        Ok(Self {
            streams: m.integer("hyper_connection.count")?,
            rank: m.integer("hyper_connection.low_rank")?,
            indexer_heads: m.integer("attention.indexer.head_count")?,
            indexer_width: m.integer("attention.indexer.key_length")?,
            selected: m.integer("attention.indexer.top_k")?,
            blocks,
            ngram: NGramSite {
                layer,
                hash,
                width: m.integer("embedding_length_per_layer_input")?,
                kernel: m.integer("ple.conv_kernel")?,
                // The convolution steps over whole n-grams.
                dilation: ngram_size,
            },
        })
    }

    fn wide(&self, hidden: u64) -> Result<u64, Error> {
        product(&[self.streams, hidden])
    }

    /// One hyper-connection site named `prefix` (`hc_attn`, `hc_ffn`,
    /// `output_hc`).
    fn site(
        &self,
        geometry: &Geometry,
        weight: &mut Weight,
        prefix: &str,
        inject: bool,
    ) -> Result<HyperConnectionWeights, Error> {
        hyper_site(geometry, weight, prefix, self.streams, self.rank, inject)
    }

    fn reads(
        &self,
        geometry: &Geometry,
        weight: &mut Weight,
        block: u64,
    ) -> Result<HistoryReads, Error> {
        if !self.selected.is_multiple_of(block) {
            return Err(invalid(
                "qwen4exp selection is not a whole number of blocks",
            ));
        }
        let (heads, width, hidden) = (self.indexer_heads, self.indexer_width, geometry.hidden);
        Ok(HistoryReads::Selected {
            block,
            blocks: self.selected / block,
            indexer: Indexer {
                heads,
                width,
                query: weight(
                    "indexer.q_proj.weight",
                    &[product(&[heads, width])?, hidden],
                )?,
                key: weight("indexer.k_proj.weight", &[width, hidden])?,
                query_norm: geometry.rms(weight("indexer.q_norm.weight", &[width])?),
                key_norm: geometry.rms(weight("indexer.k_norm.weight", &[width])?),
            },
        })
    }

    fn ngram(
        &self,
        geometry: &Geometry,
        weight: &mut Weight,
        table: &WeightDescriptor,
    ) -> Result<Sublayer, Error> {
        let site = &self.ngram;
        let hidden = geometry.hidden;
        let wide = self.wide(hidden)?;
        let embedding = product(&[site.hash.context, site.hash.heads_per_order, site.width])?;
        Ok(Sublayer {
            input: InputNorm::None,
            op: Operator::NGramInput(Box::new(NGramInput {
                hash: site.hash.clone(),
                width: site.width,
                table: table.clone(),
                key: weight("ple_key.weight", &[wide, embedding])?,
                value: weight("ple_value.weight", &[hidden, embedding])?,
                key_norm: geometry.rms(weight("ple_norm_key.weight", &[wide])?),
                query_norm: geometry.rms(weight("ple_norm_query.weight", &[wide])?),
                convolution_norm: geometry.rms(weight("ple_norm_conv.weight", &[wide])?),
                convolution: weight("ple_conv1d.weight", &[wide, site.kernel])?,
                kernel: site.kernel,
                dilation: site.dilation,
            })),
            output: OutputForm::Residual,
        })
    }
}

/// One decoder layer: the mixer and the routed feed-forward, each between
/// its own mix site and scatter.
fn block(
    geometry: &Geometry,
    extensions: &Extensions,
    weight: &mut Weight,
    layer: u64,
    kind: MixerKind,
) -> Result<Block, Error> {
    let mut sublayers = Vec::with_capacity(3);
    let mixer = match kind {
        MixerKind::Attention => {
            let mut attention = geometry.attention(weight)?;
            attention.reads =
                extensions.reads(geometry, weight, extensions.blocks[layer as usize])?;
            Operator::Attention(Box::new(attention))
        }
        MixerKind::Recurrent => Operator::GatedDelta(Box::new(
            geometry.gated_delta(weight, GateFunction::Sigmoid)?,
        )),
    };
    sublayers.push(Sublayer {
        input: InputNorm::HyperConnectionMix(extensions.site(geometry, weight, "hc_attn", true)?),
        op: mixer,
        output: OutputForm::Scatter,
    });
    sublayers.push(Sublayer {
        input: InputNorm::HyperConnectionMix(extensions.site(geometry, weight, "hc_ffn", true)?),
        op: geometry.feed_forward(weight, geometry.intermediate)?,
        output: OutputForm::Scatter,
    });
    Ok(Block { sublayers })
}

/// Inspect already-opened package components. This exists for composition roots
/// and header-only fixtures; component ownership remains with `Package`.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, Error> {
    recognize(directory)?;
    let m = Metadata::new(
        directory,
        FAMILY_ID,
        &[TARGET_KEYS, GEOMETRY_KEYS, MIXER_KEYS, ROUTED_KEYS].concat(),
    )?;
    let block_count = m.integer("block_count")?;
    let predictions = prediction_blocks(&m, block_count)?;
    let layer_count = block_count - predictions;
    // Each real layer requires distinct stored weights. Reject impossible counts
    // before allocating architecture vectors from untrusted metadata.
    if layer_count > directory.tensors.len() as u64 {
        return Err(invalid(
            "qwen4exp layer count exceeds available weight roles",
        ));
    }
    let layers = mixer_kinds(&m, layer_count, predictions)?;
    let extensions = Extensions::read(&m, &layers, predictions)?;
    let geometry = Geometry::read(&m, true)?;
    let hidden = geometry.hidden;
    let embedding = directory
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid("missing qwen4exp embedding"))?;
    let table = directory
        .tensor("per_layer_token_embd.weight")
        .ok_or_else(|| invalid("missing qwen4exp n-gram table"))?;
    let (&[vocabulary, _], &[table_rows, _]) = (&embedding.shape[..], &table.shape[..]) else {
        return Err(invalid("qwen4exp embedding tables must be matrices"));
    };
    let context_limit = m.integer("context_length")?;
    let mut tensors = Tensors::new(directory);
    let mut tensor = |name: &str, shape: &[u64]| -> Result<WeightDescriptor, Error> {
        Ok(tensors.bind(name, shape)?)
    };
    let embedding = tensor("token_embd.weight", &[vocabulary, hidden])?;
    let ngram_table = tensor(
        "per_layer_token_embd.weight",
        &[table_rows, extensions.ngram.width],
    )?;
    let output = tensor("output.weight", &[vocabulary, hidden])?;
    let exit = extensions.site(&geometry, &mut tensor, "output_hc", false)?;
    let mut blocks = layers
        .iter()
        .enumerate()
        .map(|(layer, kind)| {
            let p = format!("blk.{layer}.");
            block(
                &geometry,
                &extensions,
                &mut |name, shape| tensor(&format!("{p}{name}"), shape),
                layer as u64,
                *kind,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The n-gram input enters the streams between layers: the trailing
    // sublayer of the block before the layer the header names (its weights
    // keep that layer's names).
    let layer = extensions.ngram.layer;
    let before = layer
        .checked_sub(1)
        .and_then(|before| blocks.get_mut(before as usize))
        .ok_or_else(|| invalid("qwen4exp n-gram input layer must follow a decoder layer"))?;
    let p = format!("blk.{layer}.");
    before.sublayers.push(extensions.ngram(
        &geometry,
        &mut |name, shape| tensor(&format!("{p}{name}"), shape),
        &ngram_table,
    )?);
    if let Some(projector) = projector {
        check_projector_width(projector, hidden)?;
    }
    let vision = projector.map(describe_projector).transpose()?;
    let mut definition = ModelDefinition {
        family: FamilyId(FAMILY_ID.into()),
        artifact_identity: identity,
        inputs: InputSemantics { coordinate_axes: 3 },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary,
            context_limit,
            residual: ResidualForm::HyperConnections(HyperConnections {
                streams: extensions.streams,
                mixing: HyperConnectionMixing::LowRankGated {
                    rank: extensions.rank,
                },
            }),
            entry: EntryForm {
                embedding,
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
                hash_routing: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::HyperConnectionHead(exit),
                output,
                softcap: None,
            },
        },
        head: None,
        vision,
        draft: None,
    };
    if stores_prediction_blocks(directory, layer_count, predictions) {
        let p = format!("blk.{layer_count}.");
        definition.head = Some(head::prediction_head(
            &m,
            &definition,
            HeadComponent::Target,
            &mut |name, shape| tensor(&format!("{p}{name}"), shape),
        )?);
    }
    tensors.finish()?;
    definition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(definition)
}

/// A low-rank gated mixing site, shared by target and prediction blocks.
fn hyper_site(
    geometry: &Geometry,
    weight: &mut Weight,
    prefix: &str,
    streams: u64,
    rank: u64,
    inject: bool,
) -> Result<HyperConnectionWeights, Error> {
    let wide = product(&[streams, geometry.hidden])?;
    Ok(HyperConnectionWeights {
        norm: geometry.rms(weight(&format!("{prefix}_norm.weight"), &[wide])?),
        down: weight(&format!("{prefix}_down.weight"), &[rank, wide])?,
        up: weight(&format!("{prefix}_up.weight"), &[wide, rank])?,
        inject: inject.then(|| weight(&format!("{prefix}_inject.weight"), &[streams, wide])).transpose()?,
    })
}
