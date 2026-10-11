//! Family-neutral materialization of a validated model definition.
//!
//! The family adapter names semantic roles in [`ModelDefinition`]. This module
//! applies the planned resident representation to those roles and imports every
//! weight through [`ResidencyStore`]. It deliberately contains no model-name,
//! tensor-name, or family-specific branching.

use crate::operators;
use crate::{ResidencyStore, ResidentWeight, WeightImportError};
use magnitude_artifacts::{gguf::GgufArtifact, Package};
use magnitude_family_contracts::{
    ActivationDType, ModelDefinition, ProgressivePlane, VisionDescription, WeightKind, WeightRole,
};
use seismic::DType;
use std::collections::HashMap;
use std::{error, fmt};

#[derive(Debug)]
pub enum ResidencyError {
    Invalid(String),
    Import(WeightImportError),
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => formatter.write_str(message),
            Self::Import(error) => write!(formatter, "model residency: {error}"),
        }
    }
}

impl error::Error for ResidencyError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Import(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<WeightImportError> for ResidencyError {
    fn from(error: WeightImportError) -> Self {
        Self::Import(error)
    }
}

fn invalid(message: impl Into<String>) -> ResidencyError {
    ResidencyError::Invalid(message.into())
}

/// Resident sublayer weights keyed by their semantic role. Prepared stages
/// bind a weight by the role their graph fragment named it with.
#[derive(Clone, Default)]
pub struct ResidentRoles(HashMap<WeightRole, ResidentWeight>);

impl ResidentRoles {
    pub fn get(&self, role: WeightRole) -> Result<&ResidentWeight, String> {
        self.0
            .get(&role)
            .ok_or_else(|| format!("resident weight {role:?} is absent"))
    }

    fn values(&self) -> impl Iterator<Item = &ResidentWeight> {
        self.0.values()
    }

    fn import(
        weights: Vec<(WeightRole, &magnitude_family_contracts::WeightDescriptor)>,
        residency: &mut ResidencyStore,
        artifact: &GgufArtifact,
        activation: DType,
    ) -> Result<Self, ResidencyError> {
        let mut roles = HashMap::with_capacity(weights.len());
        for (role, descriptor) in weights {
            let weight = residency.import_gguf(
                artifact,
                descriptor,
                operators::resident_dtype(role.kind, activation),
            )?;
            if roles.insert(role, weight).is_some() {
                return Err(invalid(format!("weight role {role:?} is bound twice")));
            }
        }
        Ok(Self(roles))
    }
}

#[derive(Clone)]
pub struct ResidentHead {
    /// Tied target embedding; cloned from the sole ResidencyStore cache.
    pub embedding: ResidentWeight,
    /// Head blocks, whose weights are in `weights`.
    pub depth: usize,
    pub weights: ResidentRoles,
    /// Tied target vocabulary projection; cloned from the sole ResidencyStore cache.
    pub output: ResidentOutput,
}

/// Every projector weight, keyed by its vision role.
#[derive(Clone)]
pub struct ResidentVision {
    pub weights: ResidentRoles,
}

/// All immutable weights needed by the always-resident target decoder. Stored
/// tensors keep their semantic roles so prepared stages do not need to
/// rediscover or reinterpret artifact names. Optional components have
/// explicit demand-driven import paths below.
#[derive(Clone)]
pub struct ResidentTarget {
    pub embedding: ResidentWeight,
    /// Decoder blocks, whose weights are in `sublayers`.
    pub blocks: usize,
    pub sublayers: ResidentRoles,
    pub output_norm: ResidentWeight,
    pub output: ResidentOutput,
    /// A separate draft's fusion of the target taps, which the target
    /// readout applies; imported from the draft component.
    pub fusion: Option<ResidentFusion>,
    /// The per-layer entry, when the model has per-layer inputs.
    pub per_layer: Option<ResidentPerLayer>,
}

/// The target's vocabulary projection, as its plan places it.
#[derive(Clone)]
pub enum ResidentOutput {
    /// The output projection in its resident representation.
    Packed(ResidentWeight),
    /// Its progressive planes (`ModelLoadPlan::with_progressive_head`), which
    /// replace it.
    Progressive(ResidentPlanes),
}

/// The planes of a progressive head (`ProgressivePlane`).
#[derive(Clone)]
pub struct ResidentPlanes {
    pub top: ResidentWeight,
    pub bit3: ResidentWeight,
    pub rest: ResidentWeight,
    pub scales: ResidentWeight,
    pub radius: ResidentWeight,
}

impl ResidentPlanes {
    pub(crate) fn plane(&self, plane: ProgressivePlane) -> &ResidentWeight {
        match plane {
            ProgressivePlane::Top => &self.top,
            ProgressivePlane::Bit3 => &self.bit3,
            ProgressivePlane::Rest => &self.rest,
            ProgressivePlane::Scales => &self.scales,
            ProgressivePlane::Radius => &self.radius,
        }
    }
}

/// The per-layer entry's device weights and its host-resident table.
#[derive(Clone)]
pub struct ResidentPerLayer {
    pub projection: ResidentWeight,
    pub norm: ResidentWeight,
    pub table: crate::host_tables::HostTable,
}

/// The draft's fusion projection of the concatenated taps, and the norm the
/// draft applies to the fused rows at injection.
#[derive(Clone)]
pub struct ResidentFusion {
    pub projection: ResidentWeight,
    pub norm: ResidentWeight,
}

impl ResidentTarget {
    /// Visit the materialized target weights, including tied roles. Callers
    /// that count physical storage must deduplicate shared allocations.
    pub(crate) fn visit_weights<'a>(&'a self, mut visit: impl FnMut(&'a ResidentWeight)) {
        visit(&self.embedding);
        self.sublayers.values().for_each(&mut visit);
        visit(&self.output_norm);
        match &self.output {
            ResidentOutput::Packed(output) => visit(output),
            ResidentOutput::Progressive(planes) => {
                ProgressivePlane::ALL.into_iter().for_each(|plane| visit(planes.plane(plane)))
            }
        }
        if let Some(fusion) = &self.fusion {
            visit(&fusion.projection);
            visit(&fusion.norm);
        }
        if let Some(per_layer) = &self.per_layer {
            visit(&per_layer.projection);
            visit(&per_layer.norm);
        }
    }

    pub(crate) fn storage_bytes(&self) -> Result<u64, &'static str> {
        let mut seen = Vec::<&seismic::Tensor>::new();
        let mut total = Some(0u64);
        self.visit_weights(|weight| {
            let tensor = weight.tensor();
            if seen.iter().any(|other| other.shares_allocation(tensor)) {
                return;
            }
            total = total.and_then(|bytes| bytes.checked_add(weight.storage_bytes()));
            seen.push(tensor);
        });
        total.ok_or("resident target charge overflows")
    }
}

pub(crate) fn import_target(
    definition: &ModelDefinition,
    package: &Package,
    residency: &mut ResidencyStore,
) -> Result<ResidentTarget, ResidencyError> {
    validate_definition_package(definition, package)?;
    let decoder = &definition.decoder;
    let activation = activation_dtype(decoder.activation_dtype);
    let target = package.target();
    let embedding = residency.import_gguf(target, &decoder.entry.embedding, activation)?;
    let sublayers = ResidentRoles::import(
        operators::decoder_weights(decoder),
        residency,
        target,
        activation,
    )?;
    let output_norm = residency.import_gguf(target, decoder.exit.norm.weight(), activation)?;
    let output = import_output(decoder, target, residency, activation)?;
    let fusion = definition
        .draft
        .as_ref()
        .map(|draft| {
            let artifact = package
                .draft()
                .expect("validated: a bound draft has its component");
            let mut import = |kind, descriptor| {
                residency.import_gguf(
                    artifact,
                    descriptor,
                    operators::resident_dtype(kind, activation),
                )
            };
            Ok::<_, ResidencyError>(ResidentFusion {
                projection: import(WeightKind::DraftFusion, &draft.fusion)?,
                norm: import(WeightKind::DraftFusionNorm, &draft.fusion_norm.weight)?,
            })
        })
        .transpose()?;
    let per_layer = decoder
        .entry
        .per_layer
        .as_ref()
        .map(|entry| {
            let stored = crate::residency::Stored::from_gguf(target, &entry.table)
                .map_err(|error| invalid(error.to_string()))?;
            Ok::<_, ResidencyError>(ResidentPerLayer {
                projection: residency.import_gguf(target, &entry.projection, activation)?,
                norm: residency.import_gguf(
                    target,
                    &entry.projection_norm.weight,
                    operators::resident_dtype(WeightKind::PerLayerProjectionNorm, activation),
                )?,
                table: crate::host_tables::HostTable::open(&stored)
                    .map_err(|error| invalid(error.to_string()))?,
            })
        })
        .transpose()?;
    Ok(ResidentTarget {
        embedding,
        blocks: decoder.blocks.len(),
        sublayers,
        output_norm,
        output,
        fusion,
        per_layer,
    })
}

/// The target's output head as the plan places it: its progressive planes
/// (imported once; the residency cache returns them again) or the packed
/// projection.
fn import_output(
    decoder: &magnitude_family_contracts::Decoder,
    target: &GgufArtifact,
    residency: &mut ResidencyStore,
    activation: DType,
) -> Result<ResidentOutput, ResidencyError> {
    let planes = ProgressivePlane::ALL
        .into_iter()
        .map(|plane| {
            residency
                .planned_target(WeightKind::OutputPlane(plane))
                .map(|weight| weight.descriptor.clone())
        })
        .collect::<Option<Vec<_>>>();
    Ok(match planes {
        Some(planes) => {
            let [top, bit3, rest, scales, radius] = planes
                .iter()
                .map(|plane| residency.import_progressive(target, plane))
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| invalid("a progressive head has five planes"))?;
            ResidentOutput::Progressive(ResidentPlanes {
                top,
                bit3,
                rest,
                scales,
                radius,
            })
        }
        None => ResidentOutput::Packed(residency.import_gguf(target, &decoder.exit.output, activation)?),
    })
}

pub(crate) fn import_optional_head(
    definition: &ModelDefinition,
    package: &Package,
    residency: &mut ResidencyStore,
) -> Result<Option<ResidentHead>, ResidencyError> {
    validate_definition_package(definition, package)?;
    let decoder = &definition.decoder;
    let activation = activation_dtype(decoder.activation_dtype);
    if let Some(draft) = &definition.draft {
        return import_draft(definition, draft, package, residency).map(Some);
    }
    definition
        .head
        .as_ref()
        .map(|head| {
            let embedding =
                residency.import_gguf(package.target(), &decoder.entry.embedding, activation)?;
            let output = import_output(decoder, package.target(), residency, activation)?;
            let weights = operators::head_weights(head)
                .map_err(|error| invalid(error.to_string()))?;
            Ok(ResidentHead {
                embedding,
                depth: head.depth(),
                weights: ResidentRoles::import(weights, residency, package.target(), activation)?,
                output,
            })
        })
        .transpose()
}

/// A separate draft's drafter weights, as the head lane binds them: the
/// block's embedding table (the target's or the draft's own), the draft
/// layers' and heads' weights, the fusion norm its injection reads (shared
/// with the target's import), and the target's vocabulary projection.
fn import_draft(
    definition: &ModelDefinition,
    draft: &magnitude_family_contracts::DraftDefinition,
    package: &Package,
    residency: &mut ResidencyStore,
) -> Result<ResidentHead, ResidencyError> {
    let decoder = &definition.decoder;
    let activation = activation_dtype(decoder.activation_dtype);
    let target = package.target();
    let artifact = package
        .draft()
        .expect("validated: a bound draft has its component");
    let embedding = match &draft.embedding {
        magnitude_family_contracts::DraftEmbedding::Target => {
            residency.import_gguf(target, &decoder.entry.embedding, activation)?
        }
        magnitude_family_contracts::DraftEmbedding::Own(table) => residency.import_gguf(
            artifact,
            table,
            operators::resident_dtype(WeightKind::Embedding, activation),
        )?,
    };
    // A separate draft's readouts read the packed projection (it admits no
    // progressive head).
    let output = ResidentOutput::Packed(residency.import_gguf(target, &decoder.exit.output, activation)?);
    let mut weights = operators::draft::draft_weights(draft)
        .map_err(|error| invalid(error.to_string()))?;
    // The embedding is bound as `embedding`, not by role.
    weights.retain(|(role, _)| role.kind != WeightKind::Embedding);
    let [_, fusion_norm] = operators::draft::fusion_weights(draft);
    weights.push(fusion_norm);
    Ok(ResidentHead {
        embedding,
        depth: draft.blocks.len(),
        weights: ResidentRoles::import(weights, residency, artifact, activation)?,
        output,
    })
}

pub(crate) fn import_optional_vision(
    definition: &ModelDefinition,
    package: &Package,
    residency: &mut ResidencyStore,
) -> Result<Option<ResidentVision>, ResidencyError> {
    validate_definition_package(definition, package)?;
    definition
        .vision
        .as_ref()
        .zip(package.projector())
        .map(|(vision, projector)| materialize_vision(residency, projector, vision))
        .transpose()
}

pub(crate) fn validate_definition_package(
    definition: &ModelDefinition,
    package: &Package,
) -> Result<(), ResidencyError> {
    definition
        .validate()
        .map_err(|error| invalid(format!("invalid model definition: {error}")))?;
    validate_projector_binding(definition.vision.is_some(), package.projector().is_some())?;
    // A package may carry a draft the definition leaves unselected; a bound
    // draft needs its component.
    if definition.draft.is_some() && package.draft().is_none() {
        return Err(invalid(
            "model definition binds a draft, but the package has no draft component",
        ));
    }
    if definition.artifact_identity != package.identity() {
        return Err(invalid(format!(
            "model definition identifies package {}, but opened package is {}",
            definition.artifact_identity,
            package.identity()
        )));
    }
    Ok(())
}

fn validate_projector_binding(
    definition_has_vision: bool,
    package_has_projector: bool,
) -> Result<(), ResidencyError> {
    match (definition_has_vision, package_has_projector) {
        (true, false) => Err(invalid(
            "model definition requires a projector component, but the package has none",
        )),
        (false, true) => Err(invalid(
            "package contains a projector component not bound by the model definition",
        )),
        _ => Ok(()),
    }
}

pub(crate) fn activation_dtype(dtype: ActivationDType) -> DType {
    match dtype {
        ActivationDType::F16 => DType::F16,
        ActivationDType::BF16 => DType::BF16,
    }
}

// Projector weights are imported into the dense element their admitted plan
// chose (their stored element).
fn materialize_vision(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    vision: &VisionDescription,
) -> Result<ResidentVision, ResidencyError> {
    let mut roles = HashMap::new();
    for (role, descriptor) in vision.weights() {
        let weight = residency.import_gguf_planned(artifact, descriptor)?;
        if roles.insert(role, weight).is_some() {
            return Err(invalid(format!("weight role {role:?} is bound twice")));
        }
    }
    Ok(ResidentVision {
        weights: ResidentRoles(roles),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_no_family_or_tensor_name_policy() {
        let source = include_str!("resident_weights.rs");
        let source = source.split("#[cfg(test)]").next().unwrap();
        for forbidden in ["qwen", "Qwen", "ssm_", "attn_"] {
            assert!(
                !source.contains(forbidden),
                "family detail leaked: {forbidden}"
            );
        }
    }

    #[test]
    fn projector_presence_must_exactly_match_the_family_contract() {
        assert!(validate_projector_binding(false, false).is_ok());
        assert!(validate_projector_binding(true, true).is_ok());
        assert_eq!(
            validate_projector_binding(true, false)
                .unwrap_err()
                .to_string(),
            "model definition requires a projector component, but the package has none"
        );
        assert_eq!(
            validate_projector_binding(false, true)
                .unwrap_err()
                .to_string(),
            "package contains a projector component not bound by the model definition"
        );
    }
}
