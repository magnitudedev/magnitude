//! Qwen 3.5 artifact recognition and family-specific numerical interpretation.
//!
//! This adapter contains only facts that can vary across model families. It
//! recognizes Qwen 3.5 GGUF metadata and binds stored tensor names to the
//! family-neutral numerical contracts consumed by the engine, through the
//! builders the Qwen families share.

mod family;

pub use family::Qwen35Family;

use magnitude_artifacts::{
    Package, PackageIdentity,
    gguf::{Directory, Value},
};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    ActivationDType, Block, Decoder, EmbeddingScale, EntryForm, ExitForm, ExitNorm, FamilyId,
    GateFunction, Head, HeadBlock, HeadChain, HeadComponent, HeadHistory, HeadInput, HeadReadout,
    InputNorm, InputSemantics, ModelDefinition, Operator, OutputForm, ResidualForm, Sublayer,
    VocabularyProjection, WeightDescriptor,
};
use magnitude_family_qwen::{
    Geometry, Metadata, MixerKind, Weight, check_projector_base, describe_projector, mixer_kinds,
    product,
};

pub use magnitude_family_qwen::Error;

/// Qwen 3.5 architecture recognized from literal artifact metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Dense,
    Routed,
}

impl Architecture {
    pub const fn family_id(self) -> &'static str {
        match self {
            Self::Dense => "qwen35",
            Self::Routed => "qwen35moe",
        }
    }

    const fn is_routed(self) -> bool {
        matches!(self, Self::Routed)
    }
}

/// The family ids of every Qwen 3.5 architecture.
pub const FAMILY_IDS: &[&str] = &["qwen35", "qwen35moe"];

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

/// Recognize the supported Qwen 3.5 architecture without interpreting its
/// geometry or tensor roles.
pub fn recognize(directory: &Directory) -> Result<Architecture, Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
        .ok_or_else(|| invalid("missing GGUF architecture"))?
    {
        "qwen35" => Ok(Architecture::Dense),
        "qwen35moe" => Ok(Architecture::Routed),
        architecture => Err(invalid(format!(
            "unsupported Qwen GGUF architecture {architecture:?}"
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

/// One attention or gated delta sublayer and one feed-forward sublayer, as
/// every target layer and every MTP block has them.
fn block(
    geometry: &Geometry,
    weight: &mut Weight,
    kind: MixerKind,
    intermediate: u64,
) -> Result<Block, Error> {
    let mixer = match kind {
        MixerKind::Attention => Operator::Attention(Box::new(geometry.attention(weight)?)),
        MixerKind::Recurrent => {
            Operator::GatedDelta(Box::new(geometry.gated_delta(weight, GateFunction::Silu)?))
        }
    };
    let feed_forward = geometry.feed_forward(weight, intermediate)?;
    let hidden = geometry.hidden;
    Ok(Block {
        sublayers: vec![
            Sublayer {
                input: InputNorm::Rms(geometry.rms(weight("attn_norm.weight", &[hidden])?)),
                op: mixer,
                output: OutputForm::Residual,
            },
            Sublayer {
                input: InputNorm::Rms(
                    geometry.rms(weight("post_attention_norm.weight", &[hidden])?),
                ),
                op: feed_forward,
                output: OutputForm::Residual,
            },
        ],
    })
}

/// Inspect already-opened package components. This exists for composition roots
/// and header-only fixtures; component ownership remains with `Package`.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, Error> {
    let architecture = recognize(directory)?;
    let family_id = architecture.family_id();
    let m = Metadata::new(directory, family_id);
    let block_count = m.integer("block_count")?;
    let speculative = match m.optional("nextn_predict_layers") {
        None => 0,
        Some(v) => v
            .unsigned()
            .filter(|n| *n < block_count)
            .ok_or_else(|| invalid("nextn_predict_layers must be below block count"))?,
    };
    if speculative > 1 {
        return Err(invalid(
            "Qwen artifacts with more than one nextn prediction layer are not admitted until their GGUF roles and execution semantics are qualified",
        ));
    }
    let layer_count = block_count - speculative;
    // Each real layer requires distinct stored weights. Reject impossible counts
    // before allocating architecture vectors from untrusted metadata.
    if layer_count > directory.tensors.len() as u64 {
        return Err(invalid("Qwen layer count exceeds available weight roles"));
    }
    let layers = mixer_kinds(&m, layer_count)?;
    let embedding = directory
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid("missing Qwen embedding"))?;
    if embedding.shape.len() != 2 {
        return Err(invalid("Qwen embedding must be a matrix"));
    }
    let geometry = Geometry::read(&m, architecture.is_routed())?;
    let hidden = geometry.hidden;
    let vocabulary = embedding.shape[0];
    let context_limit = m.integer("context_length")?;
    let mut tensors = Tensors::new(directory);
    let mut tensor = |name: &str, shape: &[u64]| -> Result<WeightDescriptor, Error> {
        Ok(tensors.bind(name, shape)?)
    };
    let embedding = tensor(&embedding.name, &[vocabulary, hidden])?;
    let output_norm = tensor("output_norm.weight", &[hidden])?;
    let output = tensor(
        if directory.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            &embedding.name
        },
        &[vocabulary, hidden],
    )?;
    let blocks = layers
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            let p = format!("blk.{i}.");
            block(
                &geometry,
                &mut |name, shape| tensor(&format!("{p}{name}"), shape),
                *kind,
                geometry.intermediate,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let head = if speculative == 0 {
        None
    } else {
        let mut head_blocks = Vec::with_capacity(speculative as usize);
        for offset in 0..speculative {
            let p = format!("blk.{}.", layer_count + offset);
            let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
            let block = block(
                &geometry,
                &mut weight,
                MixerKind::Attention,
                geometry.intermediate,
            )?;
            head_blocks.push(HeadBlock {
                residual: magnitude_family_contracts::ResidualForm::Single,
                input: HeadInput::Combine {
                    embedding_norm: geometry.rms(weight("nextn.enorm.weight", &[hidden])?),
                    hidden_norm: geometry.rms(weight("nextn.hnorm.weight", &[hidden])?),
                    combine: weight("nextn.eh_proj.weight", &[hidden, product(&[2, hidden])?])?,
                },
                blocks: vec![block],
                output_norm: ExitNorm::Rms(
                    geometry.rms(weight("nextn.shared_head_norm.weight", &[hidden])?),
                ),
                readout: HeadReadout::full(VocabularyProjection::Target),
                chain: HeadChain::Output,
            });
        }
        Some(Head {
            blocks: head_blocks,
            history: HeadHistory::Own,
            component: HeadComponent::Target,
        })
    };
    tensors.finish()?;
    if let Some(projector) = projector {
        check_projector_base(directory, projector)?;
    }
    let vision = projector.map(describe_projector).transpose()?;
    let definition = ModelDefinition {
        family: FamilyId(family_id.into()),
        artifact_identity: identity,
        inputs: InputSemantics { coordinate_axes: 3 },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary,
            context_limit,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding,
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
                hash_routing: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::Rms(geometry.rms(output_norm)),
                output,
                softcap: None,
            },
        },
        head,
        vision,
        draft: None,
    };
    definition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(definition)
}
