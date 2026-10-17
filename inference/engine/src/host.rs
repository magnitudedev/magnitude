//! Host-side model artifacts: everything chat semantics need, device-free.
//!
//! One resolution produces them (`EngineConfiguration::resolve`); request
//! preparation, host-only operations (count, template apply, properties) and
//! loads all share that value. They never open a device.

use crate::error::{ArtifactError, RequestError, ResolveError, UnsupportedModel, UnsupportedReason};
use crate::families;
use crate::options::InputModalities;
use magnitude_artifacts::{
    ImageProcessor, InputLayout, Package, TokenId, TokenizerPayload,
    gguf::{Directory, Value},
};
use magnitude_chat::{
    ByteBpeTokenizer, SpecialTokens, TemplateBundle, TemplateInspection,
    artifacts::{gguf_byte_bpe, gguf_templates},
    request::ImageInput,
};
use magnitude_executor::image_cell_limit;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, MarkerTokens, ModelDefinition, ModelFamily,
    PreparedModelInput, TokenPlan,
};
use std::{fmt, sync::Arc};

/// The package's chat and input semantics: tokenizer, templates, media
/// processor and the recognized family's input adapter.
pub struct HostArtifacts {
    package: Arc<Package>,
    family: &'static dyn ModelFamily,
    /// The definition at the served context bound.
    definition: ModelDefinition,
    draft_unavailable: Option<DraftUnavailable>,
    /// The same definition at the artifact's declared capability, for
    /// host-only sizing (counting).
    declared: ModelDefinition,
    tokenizer: Arc<ByteBpeTokenizer>,
    templates: TemplateBundle,
    /// The templates' header-only chat facts, established once.
    inspection: TemplateInspection,
    media: Option<ImageProcessor>,
    input_adapter: FamilyInputAdapter,
}

struct TokenizerMarkers<'a>(&'a ByteBpeTokenizer);

impl MarkerTokens for TokenizerMarkers<'_> {
    fn marker(&self, text: &str) -> Result<TokenId, FamilyError> {
        let tokens = self
            .0
            .encode(text, SpecialTokens::Recognize)
            .map_err(FamilyError::invalid)?;
        let [token] = tokens.as_slice() else {
            return Err(FamilyError::invalid(format!("special token {text:?} is not atomic")));
        };
        if self.0.piece(*token, false).map_err(FamilyError::invalid)? != text.as_bytes() {
            return Err(FamilyError::invalid(format!(
                "special token {text:?} has a different vocabulary value"
            )));
        }
        Ok(*token)
    }
}

/// Host admission retains optional-component failures outside the numerical
/// model. A load whose draft does not bind serves the target without it; the
/// same binding makes assessment report that drafter `Rejected`. A drafter
/// assessment rejects for any other reason (its weight encodings, its plan)
/// is one the host leaves out of the load it requests.
pub struct AdmittedDefinition {
    pub definition: ModelDefinition,
    pub draft_unavailable: Option<DraftUnavailable>,
}

impl AdmittedDefinition {
    fn bound(definition: ModelDefinition) -> Self {
        Self {
            definition,
            draft_unavailable: None,
        }
    }

    fn without(declared: ModelDefinition, reason: DraftUnavailable) -> Self {
        Self {
            definition: declared,
            draft_unavailable: Some(reason),
        }
    }
}

/// Why a package's separate draft is not bound: the drafter family its
/// `general.architecture` names could not interpret it against the target,
/// or it names none. The target serves without it.
#[derive(Clone, Debug, PartialEq)]
pub enum DraftUnavailable {
    /// A DFlash-family block drafter (`dflash`).
    Block(magnitude_family_dflash::Error),
    /// A Gemma 4 assistant (`gemma4-assistant`) that is not a draft head of
    /// this target.
    Assistant(magnitude_family_gemma4::Error),
    PredictionHead(magnitude_family_qwen4exp::Error),
    /// An architecture no drafter family interprets.
    Architecture(Option<String>),
}

impl DraftUnavailable {
    /// The refusal as the engine reports it for a drafter.
    pub fn unsupported(&self) -> UnsupportedModel {
        let reasons = match self {
            Self::Block(error) => error.reasons(),
            Self::Assistant(error) | Self::PredictionHead(error) => error.reasons.clone(),
            Self::Architecture(architecture) => vec![UnsupportedReason::Architecture {
                architecture: architecture.clone().unwrap_or_default(),
            }],
        };
        UnsupportedModel::new(reasons, self.to_string())
    }
}

impl fmt::Display for DraftUnavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Block(error) => write!(formatter, "DFlash draft: {error}"),
            Self::Assistant(error) => write!(formatter, "Gemma assistant draft: {error}"),
            Self::PredictionHead(error) => write!(formatter, "prediction head: {error}"),
            Self::Architecture(Some(architecture)) => write!(
                formatter,
                "draft GGUF architecture {architecture:?} names no drafter family"
            ),
            Self::Architecture(None) => formatter.write_str("draft has no GGUF architecture"),
        }
    }
}

/// An opened package's family and declared definition: the recognized
/// target family's definition, with the package's separate draft bound
/// against it (`bind_draft`). Reads no tokenizer or template.
pub fn package_definition(
    package: &Package,
) -> Result<(&'static dyn ModelFamily, AdmittedDefinition), ResolveError> {
    let target = package.target().directory();
    let family = families::recognize(target).map_err(ResolveError::Unsupported)?;
    let declared = family
        .inspect(
            target,
            package.projector().map(|projector| projector.directory()),
            package.identity(),
        )
        .map_err(|error| ResolveError::Unsupported(error.into()))?;
    let declared = bind_draft(
        family,
        declared,
        package.draft().map(|draft| draft.directory()),
    );
    Ok((family, declared))
}

/// Bind an optional draft after target interpretation has succeeded. A draft
/// failure cannot hide a target failure or change target numerical semantics.
/// The draft's own `general.architecture` names the family that interprets
/// it: a block drafter's tapped layers the target family maps onto its
/// sublayers; a Gemma assistant is a draft head reading its target's history.
/// A payload-backed load and a header-only assessment bind the same draft.
pub fn bind_draft(
    family: &dyn ModelFamily,
    declared: ModelDefinition,
    draft: Option<&Directory>,
) -> AdmittedDefinition {
    let Some(draft) = draft else {
        return AdmittedDefinition::bound(declared);
    };
    match draft.value("general.architecture").and_then(Value::string) {
        Some(magnitude_family_qwen4exp::FAMILY_ID) => {
            match magnitude_family_qwen4exp::inspect_head(draft, &declared) {
                Ok(head) => AdmittedDefinition::bound(ModelDefinition { head: Some(head), ..declared }),
                Err(error) => AdmittedDefinition::without(declared, DraftUnavailable::PredictionHead(error)),
            }
        }
        Some(magnitude_family_dflash::ARCHITECTURE) => {
            match magnitude_family_dflash::inspect(draft, &declared, &|layer| {
                family.layer_entry(&declared, layer)
            }) {
                Ok(draft) => AdmittedDefinition::bound(ModelDefinition {
                    draft: Some(draft),
                    ..declared
                }),
                Err(error) => AdmittedDefinition::without(declared, DraftUnavailable::Block(error)),
            }
        }
        Some(magnitude_family_gemma4::ASSISTANT_ARCHITECTURE) => {
            match magnitude_family_gemma4::inspect_assistant(draft, &declared) {
                Ok(head) => AdmittedDefinition::bound(ModelDefinition {
                    head: Some(head),
                    ..declared
                }),
                Err(error) => {
                    AdmittedDefinition::without(declared, DraftUnavailable::Assistant(error))
                }
            }
        }
        architecture => AdmittedDefinition::without(
            declared,
            DraftUnavailable::Architecture(architecture.map(str::to_owned)),
        ),
    }
}

impl HostArtifacts {
    /// Interpret an opened package: family recognition, definition,
    /// tokenizer, templates, media processor and input adapter.
    /// `served_context` bounds the served definition within the declared one;
    /// `launch_rows`, a load's launch row bound, bounds its image resize to the
    /// cells that load's vision graphs encode.
    pub(crate) fn interpret(
        package: Package,
        served_context: Option<usize>,
        launch_rows: Option<usize>,
    ) -> Result<Self, ResolveError> {
        let (family, admitted) = package_definition(&package)?;
        let AdmittedDefinition {
            definition: declared,
            draft_unavailable,
        } = admitted;
        if let Some(reason) = &draft_unavailable {
            eprintln!("magnitude-engine: separate drafter disabled: {reason}");
        }
        let mut definition = declared.clone();
        definition.decoder.context_limit =
            resolve_served_context(served_context, declared.decoder.context_limit)?;
        let unsupported =
            |detail: String| ResolveError::Unsupported(UnsupportedModel::unreadable(detail));
        if let Some(launch_rows) = launch_rows {
            let cells = image_cell_limit(&definition, launch_rows)
                .map_err(|error| unsupported(error.to_string()))?;
            if let Some(vision) = definition.vision.as_mut() {
                vision
                    .bound_cells(cells as u64)
                    .map_err(|error| unsupported(error.to_string()))?;
            }
        }
        let tokenizer = Arc::new(
            gguf_byte_bpe(
                package.tokenizer(),
                definition.artifact_identity.target.to_string(),
            )
            .and_then(ByteBpeTokenizer::new)
            .map_err(|error| ResolveError::Unsupported(error.into()))?,
        );
        if u64::try_from(tokenizer.vocabulary()).ok() != Some(definition.decoder.vocabulary) {
            return Err(ResolveError::Unsupported(vocabulary_mismatch(
                package.tokenizer(),
            )));
        }
        let templates = gguf_templates(package.templates(), package.tokenizer())
            .map_err(|detail| ResolveError::Unsupported(UnsupportedModel::chat_template(detail)))?;
        let inspection = templates
            .inspect()
            .map_err(|detail| ResolveError::Unsupported(UnsupportedModel::chat_template(detail)))?;
        let media = definition
            .vision
            .as_ref()
            .map(|vision| {
                let config = vision
                    .image_processor_config()
                    .map_err(|error| unsupported(error.to_string()))?;
                ImageProcessor::new(config).map_err(unsupported)
            })
            .transpose()?;
        let input_adapter = family
            .input_adapter(&definition, &TokenizerMarkers(&tokenizer))
            .map_err(|error| ResolveError::Unsupported(error.into()))?;
        Ok(Self {
            package: Arc::new(package),
            family,
            definition,
            draft_unavailable,
            declared,
            tokenizer,
            templates,
            inspection,
            media,
            input_adapter,
        })
    }

    /// Open and interpret a package at its declared context.
    pub fn open(package: &crate::options::PackageOptions) -> Result<Self, ResolveError> {
        let opened = package.open().map_err(|error| {
            ResolveError::Artifact(ArtifactError::from_artifacts(error, &package.target))
        })?;
        Self::interpret(opened, None, None)
    }

    pub fn package(&self) -> &Package {
        &self.package
    }

    pub fn shared_package(&self) -> Arc<Package> {
        self.package.clone()
    }

    pub fn family(&self) -> &'static dyn ModelFamily {
        self.family
    }

    /// The definition at the served context bound.
    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

    pub(crate) fn draft_unavailable(&self) -> Option<&DraftUnavailable> {
        self.draft_unavailable.as_ref()
    }

    /// The artifact's declared context capability.
    pub fn declared_context_limit(&self) -> u64 {
        self.declared.decoder.context_limit
    }

    pub fn tokenizer(&self) -> &ByteBpeTokenizer {
        self.tokenizer.as_ref()
    }

    pub fn shared_tokenizer(&self) -> Arc<ByteBpeTokenizer> {
        self.tokenizer.clone()
    }

    pub fn templates(&self) -> &TemplateBundle {
        &self.templates
    }

    /// The templates' chat facts, including their fingerprint.
    pub fn template_inspection(&self) -> &TemplateInspection {
        &self.inspection
    }

    /// The text one image renders as in the chat template, when the model
    /// accepts images.
    pub fn media_placeholder(&self) -> Option<&'static str> {
        self.media
            .as_ref()
            .and_then(|_| self.family.media_placeholder(&self.definition))
    }

    /// The inputs the model accepts besides text.
    pub fn modalities(&self) -> InputModalities {
        InputModalities::of(self.family, &self.definition)
    }

    /// Run the processor bound into the package identity over the request's
    /// images and let the family adapter install the resulting spans and
    /// coordinates. Bounded by the served context.
    pub fn prepare_input(
        &self,
        tokens: Vec<TokenId>,
        images: &[ImageInput],
    ) -> Result<PreparedModelInput, RequestError> {
        self.prepare_with(&self.definition, tokens, images)
    }

    /// The same preparation bounded by the artifact's declared capability,
    /// for host-only sizing; it never enlarges the served bound.
    pub fn prepare_count_input(
        &self,
        tokens: Vec<TokenId>,
        images: &[ImageInput],
    ) -> Result<PreparedModelInput, RequestError> {
        self.prepare_with(&self.declared, tokens, images)
    }

    fn prepare_with(
        &self,
        definition: &ModelDefinition,
        tokens: Vec<TokenId>,
        images: &[ImageInput],
    ) -> Result<PreparedModelInput, RequestError> {
        let invalid = |reason: String| RequestError::InvalidRequest { reason };
        let layout = InputLayout::new(tokens.len(), Vec::new()).map_err(invalid)?;
        let token_plan =
            TokenPlan::new(tokens, layout).map_err(|error| invalid(error.to_string()))?;
        let media = if images.is_empty() {
            Vec::new()
        } else {
            let processor = self
                .media
                .as_ref()
                .ok_or_else(|| invalid("the loaded model has no image processor".into()))?;
            let encoded = images
                .iter()
                .map(|image| image.bytes.to_vec())
                .collect::<Vec<_>>();
            vec![processor.prepare(&encoded).map_err(invalid)?]
        };
        self.input_adapter
            .prepare(definition, token_plan, &media)
            .map_err(|error| invalid(error.to_string()))
    }
}

/// A tokenizer whose vocabulary is not the model's: the package's tokenizer
/// scheme is one this model cannot use.
pub(crate) fn vocabulary_mismatch(payload: &TokenizerPayload) -> UnsupportedModel {
    tokenizer_refusal(payload, "tokenizer vocabulary differs from model vocabulary")
}

/// The package's tokenizer scheme, refused for `detail`.
pub(crate) fn tokenizer_refusal(
    payload: &TokenizerPayload,
    detail: impl Into<String>,
) -> UnsupportedModel {
    let text = |key: &str| payload.value(key).and_then(Value::string).map(str::to_owned);
    UnsupportedModel::reason(
        UnsupportedReason::Tokenizer {
            model: text("tokenizer.ggml.model").unwrap_or_default(),
            pre: text("tokenizer.ggml.pre"),
        },
        detail,
    )
}

fn resolve_served_context(
    requested: Option<usize>,
    artifact_limit: u64,
) -> Result<u64, ResolveError> {
    let Some(requested) = requested else {
        return Ok(artifact_limit);
    };
    let invalid = || ResolveError::InvalidConfiguration {
        reason: format!(
            "serving context {requested} is outside the artifact limit {artifact_limit}"
        ),
    };
    let requested = u64::try_from(requested).map_err(|_| invalid())?;
    if requested == 0 || requested > artifact_limit {
        return Err(invalid());
    }
    Ok(requested)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_admission_retains_target_and_distinguishes_missing_from_invalid() {
        use magnitude_artifacts::{
            ArtifactIdentity, PackageIdentity,
            gguf::{Metadata, Scalar, Value},
        };
        use magnitude_family_common::headers;
        let family = magnitude_family_qwen35::Qwen35Family;
        let target = family
            .inspect(
                &headers::directory("qwen3.6-35b-a3b__target-gguf_q4.json"),
                None,
                PackageIdentity {
                    target: ArtifactIdentity([7; 32]),
                    projector: None,
                },
            )
            .unwrap();
        let mut directory = headers::directory("qwen3.6-35b-a3b__draft.json");
        directory
            .metadata
            .retain(|entry| entry.name != "dflash.attention.causal");
        let missing = bind_draft(&family, target.clone(), Some(&directory));
        assert_eq!(missing.definition, target);
        assert_eq!(
            missing.draft_unavailable,
            Some(DraftUnavailable::Block(
                magnitude_family_dflash::Error::MissingCausality
            ))
        );
        directory.metadata.push(Metadata {
            name: "dflash.attention.causal".into(),
            value: Value::Scalar(Scalar::Unsigned(1)),
        });
        let invalid = bind_draft(&family, target.clone(), Some(&directory));
        assert_eq!(invalid.definition, target);
        assert!(matches!(
            invalid.draft_unavailable,
            Some(DraftUnavailable::Block(
                magnitude_family_dflash::Error::Metadata { .. }
            ))
        ));
        directory.metadata.last_mut().unwrap().value = Value::Array(
            vec![true, true, true, true, true, false]
                .into_iter()
                .map(Scalar::Bool)
                .collect(),
        );
        let admitted = bind_draft(&family, target.clone(), Some(&directory));
        assert!(admitted.draft_unavailable.is_none());
        assert!(admitted.definition.draft.is_some());
        assert_eq!(admitted.definition.decoder, target.decoder);
    }

    #[test]
    fn a_draft_is_interpreted_by_the_family_its_architecture_names() {
        use magnitude_artifacts::{
            ArtifactIdentity, PackageIdentity,
            gguf::{Scalar, Value},
        };
        use magnitude_family_common::headers;
        let identity = PackageIdentity {
            target: ArtifactIdentity([7; 32]),
            projector: None,
        };
        let qwen = magnitude_family_qwen35::Qwen35Family;
        let target = qwen
            .inspect(
                &headers::directory("qwen3.6-35b-a3b__target-gguf_q4.json"),
                None,
                identity,
            )
            .unwrap();
        // A Gemma assistant reaches the assistant family, which names why it
        // is not a draft of this target; the target is kept whole.
        let mut assistant = headers::directory("qwen3.6-35b-a3b__draft.json");
        headers::set(
            &mut assistant,
            "general.architecture",
            Value::Scalar(Scalar::String(
                magnitude_family_gemma4::ASSISTANT_ARCHITECTURE.into(),
            )),
        );
        let admitted = bind_draft(&qwen, target.clone(), Some(&assistant));
        assert_eq!(admitted.definition, target);
        let Some(DraftUnavailable::Assistant(error)) = &admitted.draft_unavailable else {
            panic!(
                "an assistant is interpreted as one: {:?}",
                admitted.draft_unavailable
            );
        };
        assert!(error.to_string().contains("drafts for a Gemma 4 target"));
        // A Gemma target's assistant failure is the assistant's, not DFlash's.
        let gemma = magnitude_family_gemma4::Gemma4Family;
        let gemma_target = gemma
            .inspect(
                &headers::directory("gemma-4-12b-it-qat__target-gguf_q4.json"),
                None,
                identity,
            )
            .unwrap();
        let admitted = bind_draft(&gemma, gemma_target.clone(), Some(&assistant));
        assert_eq!(admitted.definition, gemma_target);
        assert!(matches!(
            admitted.draft_unavailable,
            Some(DraftUnavailable::Assistant(_))
        ));
        let mut unknown = assistant;
        headers::set(
            &mut unknown,
            "general.architecture",
            Value::Scalar(Scalar::String("eagle3".into())),
        );
        assert_eq!(
            bind_draft(&gemma, gemma_target, Some(&unknown)).draft_unavailable,
            Some(DraftUnavailable::Architecture(Some("eagle3".into())))
        );
    }

    #[test]
    fn draft_refusals_name_their_reasons() {
        use magnitude_family_dflash::Error;
        let reasons = |unavailable: DraftUnavailable| unavailable.unsupported().reasons;
        assert_eq!(
            reasons(DraftUnavailable::Architecture(Some("eagle3".into()))),
            [UnsupportedReason::Architecture {
                architecture: "eagle3".into()
            }]
        );
        assert_eq!(
            reasons(DraftUnavailable::Block(Error::Unadmitted {
                metadata: vec!["dflash.rope.dimension_count".into()],
                tensors: vec!["d2t".into()],
            })),
            [
                UnsupportedReason::UnrecognizedMetadata {
                    key: "dflash.rope.dimension_count".into()
                },
                UnsupportedReason::UnrecognizedTensor { name: "d2t".into() },
            ]
        );
        for error in [
            Error::Target("draft width 1024 differs from the target's 2048".into()),
            Error::Geometry("draft taps no target layer".into()),
        ] {
            let expected = error.to_string();
            let reasons = reasons(DraftUnavailable::Block(error));
            assert!(
                matches!(
                    reasons.as_slice(),
                    [UnsupportedReason::TargetMismatch { detail }] if detail.contains(&expected)
                ),
                "{reasons:?}"
            );
        }
        assert_eq!(
            reasons(DraftUnavailable::Block(Error::MissingCausality)),
            [UnsupportedReason::MissingMetadata {
                key: "dflash.attention.causal".into()
            }]
        );
        assert_eq!(
            reasons(DraftUnavailable::Block(Error::MissingWeight("fc.weight".into()))),
            [UnsupportedReason::Unreadable]
        );
        assert_eq!(
            reasons(DraftUnavailable::PredictionHead(FamilyError::mismatch(
                "not a head of this target"
            ))),
            [UnsupportedReason::TargetMismatch {
                detail: "not a head of this target".into()
            }]
        );
    }

    #[test]
    fn served_context_is_an_explicit_bound_within_artifact_capability() {
        assert_eq!(resolve_served_context(None, 262_144).unwrap(), 262_144);
        assert_eq!(
            resolve_served_context(Some(16_384), 262_144).unwrap(),
            16_384
        );
        assert!(resolve_served_context(Some(0), 262_144).is_err());
        assert!(resolve_served_context(Some(262_145), 262_144).is_err());
    }
}
