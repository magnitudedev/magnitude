//! Host-side model artifacts: everything chat semantics need, device-free.
//!
//! One resolution produces them (`EngineConfiguration::resolve`); request
//! preparation, host-only operations (count, template apply, properties) and
//! loads all share that value, a load's bounded to the images its device
//! encodes. They never open a device.

use crate::error::{ArtifactError, RequestError, ResolveError, UnsupportedModel, UnsupportedReason};
use crate::families;
use crate::options::InputModalities;
use magnitude_artifacts::{
    DecisionMetadata, ImageProcessor, InputLayout, Package, PackageIdentity, TokenId,
    TokenizerPayload,
    gguf::{Directory, Value},
};
use magnitude_family_common::{readout_head, HeaderError};
use magnitude_decision::Decider;
use magnitude_chat::{
    ByteBpeTokenizer, SpecialTokens, TemplateBundle, TemplateInspection,
    artifacts::{gguf_byte_bpe, gguf_templates},
    request::ImageInput,
};
use magnitude_executor::{image_cell_limit, MAX_IMAGE_CELLS};
use magnitude_family_contracts::{
    DecisionBinding, FamilyError, MarkerTokens, ModelDefinition, ModelFamily, ModelInputAdapter,
    PreparedModelInput, TokenPlan, WeightDescriptor,
};
use std::{fmt, sync::Arc};

/// The package's chat and input semantics: tokenizer, templates, the
/// recognized family's input adapter and each definition's media processor
/// ([`media_processor`]), so the media a request prepares is the media its
/// input adapter reads under the same definition.
#[derive(Clone)]
pub struct HostArtifacts {
    package: Arc<Package>,
    family: &'static dyn ModelFamily,
    /// The definition at the served context bound.
    definition: ModelDefinition,
    /// `definition`'s media processor.
    media: Option<ImageProcessor>,
    draft_unavailable: Option<DraftUnavailable>,
    /// The same definition at the artifact's declared capability, for
    /// host-only sizing (counting).
    declared: ModelDefinition,
    /// `declared`'s media processor.
    declared_media: Option<ImageProcessor>,
    tokenizer: Arc<ByteBpeTokenizer>,
    templates: Arc<TemplateBundle>,
    /// The templates' header-only chat facts, established once.
    inspection: TemplateInspection,
    input_adapter: Arc<dyn ModelInputAdapter + Send + Sync>,
    /// The decision description the GGUF carries, bound to the tokenizer:
    /// present exactly when the model serves decisions.
    decider: Option<Arc<Decider>>,
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
/// is one the host leaves out of the load it requests. A projector that does
/// not bind ([`interpret_target`]) leaves the target serving text only, and
/// assessment reports that projector `Rejected`.
pub struct AdmittedDefinition {
    pub definition: ModelDefinition,
    pub draft_unavailable: Option<DraftUnavailable>,
    pub projector_unavailable: Option<UnsupportedModel>,
}

impl AdmittedDefinition {
    fn bound(definition: ModelDefinition) -> Self {
        Self {
            definition,
            draft_unavailable: None,
            projector_unavailable: None,
        }
    }

    fn without(declared: ModelDefinition, reason: DraftUnavailable) -> Self {
        Self {
            definition: declared,
            draft_unavailable: Some(reason),
            projector_unavailable: None,
        }
    }
}

/// A target interpreted with its projector, and why the projector did not
/// bind when it did not.
pub struct InterpretedTarget {
    pub definition: ModelDefinition,
    pub projector_unavailable: Option<UnsupportedModel>,
}

/// Interpret a recognized target with its optional projector. A projector the
/// family cannot bind (a text-only family, an output width that is not the
/// target's embedding width, a malformed projector header) leaves the target
/// interpreted without it, taking text input only, with the projector's
/// refusal kept apart. A target the family refuses on its own is refused
/// whatever its projector, so a projector's refusal never becomes the
/// target's verdict and never hides the target's. A payload-backed load and a
/// header-only assessment interpret the same way.
///
/// The target's decision description, readout head and whether its
/// description's plans read chosen logits and scores are read here, the
/// same way for every family: the family's definition leaves them, and its
/// strict reading leaves the keys and tensor a decision description owns. A decision model
/// generates nothing, so nothing drafts for it: its definition keeps no
/// draft head, and [`bind_draft`] binds it no separate draft.
pub fn interpret_target(
    family: &dyn ModelFamily,
    target: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<InterpretedTarget, UnsupportedModel> {
    let decision = DecisionMetadata::of(target)
        .map_err(|error| FamilyError::from(HeaderError::from(error)))?;
    let bound = projector.map(|projector| family.inspect(target, Some(projector), identity));
    let (definition, projector_unavailable) = match bound {
        Some(Ok(definition)) => (definition, None),
        Some(Err(error)) => (
            family.inspect(target, None, identity)?,
            Some(UnsupportedModel::new(
                error.reasons,
                format!("projector: {}", error.detail),
            )),
        ),
        None => (family.inspect(target, None, identity)?, None),
    };
    let readout_head =
        readout_head(target, definition.decoder.hidden).map_err(FamilyError::from)?;
    let decision = decision
        .map(|metadata| {
            let facts =
                magnitude_decision::load_facts(&metadata, head_rows(readout_head.as_ref()))?;
            let binding = DecisionBinding {
                metadata,
                readout_head,
                reads_logits: facts.reads_logits,
                reads_scores: facts.reads_scores,
            };
            Ok::<_, String>((binding, facts.image_pixels))
        })
        .transpose()
        .map_err(UnsupportedModel::unreadable)?;
    // A decision description's picture cap is the model's: every picture
    // it reads, through any processor of its definition, is read within it.
    let mut vision = definition.vision;
    if let (Some(vision), Some(pixels)) = (
        vision.as_mut(),
        decision.as_ref().and_then(|(_, pixels)| *pixels),
    ) {
        vision
            .cap_pixels(pixels as u64)
            .map_err(|error| UnsupportedModel::unreadable(error.to_string()))?;
    }
    let decision = decision.map(|(binding, _)| binding);
    Ok(InterpretedTarget {
        definition: ModelDefinition {
            head: definition.head.filter(|_| decision.is_none()),
            decision,
            vision,
            ..definition
        },
        projector_unavailable,
    })
}

/// The rows of a bound readout head, one per code it reads.
fn head_rows(head: Option<&WeightDescriptor>) -> Option<usize> {
    head.map(|head| head.shape[0] as usize)
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
    /// An architecture no drafter family interprets: an unknown one, or a
    /// model architecture whose prediction heads the engine binds only
    /// built into their target, if at all (a separate Nemotron-H MTP file).
    Architecture(Option<String>),
    /// A target that serves decisions, which generates nothing to draft.
    Decisions,
}

impl DraftUnavailable {
    /// The refusal as the engine reports it for a drafter.
    pub fn unsupported(&self) -> UnsupportedModel {
        let reasons = match self {
            Self::Block(error) => error.reasons(),
            Self::Assistant(error) | Self::PredictionHead(error) => error.reasons.clone(),
            Self::Architecture(Some(architecture)) => vec![UnsupportedReason::DraftArchitecture {
                architecture: architecture.clone(),
            }],
            Self::Architecture(None) => vec![UnsupportedReason::Unreadable],
            Self::Decisions => vec![UnsupportedReason::TargetMismatch {
                detail: self.to_string(),
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
            Self::Decisions => {
                formatter.write_str("a decision model generates nothing for a drafter to draft")
            }
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
    let interpreted = interpret_target(
        family,
        target,
        package.projector().map(|projector| projector.directory()),
        package.identity(),
    )
    .map_err(ResolveError::Unsupported)?;
    let admitted = bind_draft(
        family,
        interpreted.definition,
        package.draft().map(|draft| draft.directory()),
    );
    Ok((
        family,
        AdmittedDefinition {
            projector_unavailable: interpreted.projector_unavailable,
            ..admitted
        },
    ))
}

/// Bind an optional draft after target interpretation has succeeded. A draft
/// failure cannot hide a target failure or change target numerical semantics.
/// The draft's own `general.architecture` names the family that interprets
/// it: a block drafter's tapped layers the target family maps onto its
/// sublayers; a Gemma assistant is a draft head reading its target's history.
/// A payload-backed load and a header-only assessment bind the same draft.
/// A decision target binds none: it generates nothing to draft.
pub fn bind_draft(
    family: &dyn ModelFamily,
    declared: ModelDefinition,
    draft: Option<&Directory>,
) -> AdmittedDefinition {
    let Some(draft) = draft else {
        return AdmittedDefinition::bound(declared);
    };
    if declared.decision.is_some() {
        return AdmittedDefinition::without(declared, DraftUnavailable::Decisions);
    }
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
    /// cells a load on any device encodes ([`Self::within_image_cells`]
    /// bounds it to one device's).
    pub(crate) fn interpret(
        package: Package,
        served_context: Option<usize>,
        launch_rows: Option<usize>,
    ) -> Result<Self, ResolveError> {
        let (family, admitted) = package_definition(&package)?;
        let AdmittedDefinition {
            definition: declared,
            draft_unavailable,
            projector_unavailable,
        } = admitted;
        if let Some(reason) = &draft_unavailable {
            eprintln!("magnitude-engine: separate drafter disabled: {reason}");
        }
        if let Some(reason) = &projector_unavailable {
            eprintln!("magnitude-engine: image input disabled: {}", reason.detail);
        }
        let mut definition = declared.clone();
        definition.decoder.context_limit =
            resolve_served_context(served_context, declared.decoder.context_limit)?;
        let unsupported =
            |detail: String| ResolveError::Unsupported(UnsupportedModel::unreadable(detail));
        if let Some(launch_rows) = launch_rows {
            let cells = image_cell_limit(&definition, launch_rows, MAX_IMAGE_CELLS)
                .map_err(|error| unsupported(error.to_string()))?;
            bound_images(&mut definition, cells).map_err(unsupported)?;
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
        let decider = definition
            .decision
            .as_ref()
            .map(|decision| {
                Decider::new(
                    &decision.metadata,
                    head_rows(decision.readout_head.as_ref()),
                    tokenizer.clone(),
                    definition
                        .vision
                        .as_ref()
                        .and_then(|_| family.media_placeholder(&definition)),
                )
                .map(Arc::new)
                .map_err(|detail| ResolveError::Unsupported(UnsupportedModel::unreadable(detail)))
            })
            .transpose()?;
        let media = media_processor(&definition).map_err(unsupported)?;
        let declared_media = media_processor(&declared).map_err(unsupported)?;
        let input_adapter = family
            .input_adapter(&definition, &TokenizerMarkers(&tokenizer))
            .map_err(|error| ResolveError::Unsupported(error.into()))?
            .into();
        Ok(Self {
            package: Arc::new(package),
            family,
            definition,
            media,
            draft_unavailable,
            declared,
            declared_media,
            tokenizer,
            templates: Arc::new(templates),
            inspection,
            input_adapter,
            decider,
        })
    }

    /// These artifacts for a load whose device encodes at most `cells` cells
    /// per image: the served image resize and its media processor bounded to
    /// them, so a larger image is resized down rather than refused. Shared
    /// unchanged when the served resize admits no more.
    pub(crate) fn within_image_cells(self: Arc<Self>, cells: usize) -> Result<Arc<Self>, String> {
        let Some(vision) = &self.definition.vision else {
            return Ok(self);
        };
        let served = vision.max_cells().map_err(|error| error.to_string())?;
        if served <= cells as u64 {
            return Ok(self);
        }
        let mut bounded = (*self).clone();
        bound_images(&mut bounded.definition, cells)?;
        bounded.media = media_processor(&bounded.definition)?;
        Ok(Arc::new(bounded))
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
        self.definition
            .vision
            .as_ref()
            .and_then(|_| self.family.media_placeholder(&self.definition))
    }

    /// The model's decider: present exactly when it serves decisions.
    pub fn decider(&self) -> Option<&Arc<Decider>> {
        self.decider.as_ref()
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
        self.prepare_with(&self.definition, self.media.as_ref(), tokens, images)
    }

    /// The same preparation bounded by the artifact's declared capability,
    /// for host-only sizing; it never enlarges the served bound.
    pub fn prepare_count_input(
        &self,
        tokens: Vec<TokenId>,
        images: &[ImageInput],
    ) -> Result<PreparedModelInput, RequestError> {
        self.prepare_with(&self.declared, self.declared_media.as_ref(), tokens, images)
    }

    fn prepare_with(
        &self,
        definition: &ModelDefinition,
        media: Option<&ImageProcessor>,
        tokens: Vec<TokenId>,
        images: &[ImageInput],
    ) -> Result<PreparedModelInput, RequestError> {
        prepare_model_input(definition, media, self.input_adapter.as_ref(), tokens, images)
    }
}

/// `tokens` and their `images` as `adapter` reads them under `definition`,
/// the images prepared by `processor`, `definition`'s own media processor.
fn prepare_model_input(
    definition: &ModelDefinition,
    processor: Option<&ImageProcessor>,
    adapter: &dyn ModelInputAdapter,
    tokens: Vec<TokenId>,
    images: &[ImageInput],
) -> Result<PreparedModelInput, RequestError> {
    let invalid = |reason: String| RequestError::InvalidRequest { reason };
    let layout = InputLayout::new(tokens.len(), Vec::new()).map_err(invalid)?;
    let token_plan = TokenPlan::new(tokens, layout).map_err(|error| invalid(error.to_string()))?;
    let media = if images.is_empty() {
        Vec::new()
    } else {
        let processor =
            processor.ok_or_else(|| invalid("the loaded model has no image processor".into()))?;
        let encoded = images
            .iter()
            .map(|image| image.bytes.to_vec())
            .collect::<Vec<_>>();
        vec![processor.prepare(&encoded).map_err(invalid)?]
    };
    adapter
        .prepare(definition, token_plan, &media)
        .map_err(|error| invalid(error.to_string()))
}

/// Bound `definition`'s image resize to at most `cells` cells per image.
fn bound_images(definition: &mut ModelDefinition, cells: usize) -> Result<(), String> {
    match definition.vision.as_mut() {
        Some(vision) => vision
            .bound_cells(cells as u64)
            .map_err(|error| error.to_string()),
        None => Ok(()),
    }
}

/// The media processor of `definition`'s image preprocessing, when it reads
/// images: the processor its input adapter checks prepared media against.
fn media_processor(definition: &ModelDefinition) -> Result<Option<ImageProcessor>, String> {
    definition
        .vision
        .as_ref()
        .map(|vision| {
            let config = vision
                .image_processor_config()
                .map_err(|error| error.to_string())?;
            ImageProcessor::new(config)
        })
        .transpose()
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

    /// A noul-only decision description (`decision.*` keys without their
    /// namespace), read at the last token through the LM head.
    fn noul_description() -> [(&'static str, magnitude_artifacts::gguf::Value); 13] {
        use magnitude_artifacts::gguf::{Array, Scalar, Strings, Value};
        let string = |value: &str| Value::Scalar(Scalar::String(value.into()));
        [
            ("escape.method", string("none")),
            ("segments", string("whole")),
            ("set_size", Value::Scalar(Scalar::Unsigned(1))),
            ("state_attends", string("state")),
            ("question_attends", string("question")),
            ("codes.sequence", string("letters")),
            ("codes.probe", Value::Scalar(Scalar::Bool(false))),
            ("codes.count", Value::Scalar(Scalar::Unsigned(2))),
            (
                "kinds",
                Value::Array(Array::from(["noul"].into_iter().collect::<Strings>())),
            ),
            ("noul.codes", string("sequence")),
            ("noul.true_option", Value::Scalar(Scalar::Unsigned(0))),
            ("noul.slots", string("last")),
            ("temperature.noul", Value::Scalar(Scalar::Float(1.0))),
        ]
    }

    /// A target reads logits of chosen tokens and scores exactly when its
    /// description's plans do; a description the decision crate does not
    /// implement leaves it unreadable.
    #[test]
    fn a_target_reads_what_its_decision_description_plans() {
        use magnitude_artifacts::{
            ArtifactIdentity, DECISION_NAMESPACE, DECISION_TEMPLATE, PackageIdentity,
            gguf::{Scalar, Value},
        };
        use magnitude_family_common::headers;
        let family = magnitude_family_qwen35::Qwen35Family;
        let string = |value: &str| Value::Scalar(Scalar::String(value.into()));
        let described = noul_description();
        let interpreted = |keys: &[(&str, Value)]| {
            let mut target = headers::target("qwen3.6-35b-a3b:gguf:q4");
            let architecture = target
                .value("general.architecture")
                .and_then(Value::string)
                .unwrap()
                .to_owned();
            if !keys.is_empty() {
                headers::set(&mut target, DECISION_TEMPLATE, string("{{ state }}"));
                headers::set(
                    &mut target,
                    &format!("{architecture}.attention.causal"),
                    Value::Scalar(Scalar::Bool(false)),
                );
            }
            for (key, value) in keys {
                headers::set(
                    &mut target,
                    &format!("{DECISION_NAMESPACE}{key}"),
                    value.clone(),
                );
            }
            interpret_target(
                &family,
                &target,
                None,
                PackageIdentity {
                    target: ArtifactIdentity([7; 32]),
                    projector: None,
                },
            )
            .map(|interpreted| {
                interpreted
                    .definition
                    .decision
                    .map(|decision| (decision.reads_logits, decision.reads_scores))
            })
        };
        assert_eq!(interpreted(&[]).unwrap(), None);
        assert_eq!(interpreted(&described).unwrap(), Some((true, false)));
        let unimplemented = [described.as_slice(), &[("boxes.max", Value::Scalar(Scalar::Unsigned(1)))]].concat();
        assert_eq!(
            interpreted(&unimplemented).unwrap_err().reasons,
            vec![UnsupportedReason::Unreadable]
        );
    }

    /// A decision description's picture cap is its definition's: a decision
    /// request with a picture prepares through the processor the input
    /// adapter checks its media against, at the served and the declared
    /// bound alike (an LFM3-VL target and projector, described as reading
    /// pictures before the state within d1-2B's cap).
    #[test]
    fn a_decision_request_with_a_picture_prepares_within_its_described_cap() {
        use magnitude_artifacts::{
            ArtifactIdentity, DECISION_NAMESPACE, DECISION_TEMPLATE, PackageIdentity,
            gguf::{Array, Scalar, Strings, Value},
        };
        use magnitude_decision::{
            DecisionRequest, Question, QuestionKind,
            json::{Segment, Text, Value as Content},
        };
        use magnitude_family_common::headers;
        const CAP: u64 = 1_638_400;
        let family = magnitude_family_lfm2::Lfm2Family;
        let string = |value: &str| Value::Scalar(Scalar::String(value.into()));
        let mut target = headers::target("lfm3-vl-2b:gguf:q8");
        let projector = headers::projector("lfm3-vl-2b:gguf:q8");
        headers::set(
            &mut target,
            DECISION_TEMPLATE,
            string(concat!(
                "<|im_start|>user\n{% for image in images %}\u{FDD0}image:{{ image }}\u{FDD1}{% endfor %}",
                "\u{FDD0}read:text\u{FDD1}\n\u{FDD0}state\u{FDD1}",
                "{% for question in questions %}{{ question.instructions.stripped }}{% endfor %}"
            )),
        );
        headers::set(&mut target, "lfm3.attention.causal", Value::Scalar(Scalar::Bool(false)));
        for (key, value) in noul_description().into_iter().chain([
            (
                "image_placements",
                Value::Array(Array::from(["front"].into_iter().collect::<Strings>())),
            ),
            ("image_max_pixels", Value::Scalar(Scalar::Unsigned(CAP))),
        ]) {
            headers::set(&mut target, &format!("{DECISION_NAMESPACE}{key}"), value);
        }
        let interpreted = interpret_target(
            &family,
            &target,
            Some(&projector),
            PackageIdentity {
                target: ArtifactIdentity([7; 32]),
                projector: Some(ArtifactIdentity([8; 32])),
            },
        )
        .unwrap();
        assert!(interpreted.projector_unavailable.is_none());
        let declared = interpreted.definition;
        assert_eq!(
            declared.vision.as_ref().unwrap().preprocessing.pixel_cap,
            Some(CAP)
        );
        let tokenizer = Arc::new(
            gguf_byte_bpe(
                &magnitude_artifacts::TokenizerPayload::from_directory(&target),
                "lfm3-vl-2b".into(),
            )
            .and_then(ByteBpeTokenizer::new)
            .unwrap(),
        );
        let decider = Decider::new(
            &declared.decision.as_ref().unwrap().metadata,
            None,
            tokenizer.clone(),
            family.media_placeholder(&declared),
        )
        .unwrap();
        let mut served = declared.clone();
        let cells = image_cell_limit(&served, 512, MAX_IMAGE_CELLS).unwrap();
        bound_images(&mut served, cells).unwrap();
        let literal = |text: &str| Content::String(Text::new([Segment::Literal(text.into())]));
        let request = DecisionRequest {
            state: literal("A picture."),
            images: vec![ImageInput {
                media_type: "image/png".into(),
                bytes: Arc::from(
                    &include_bytes!("../artifacts/tests/fixtures/cap-source.png")[..],
                ),
            }],
            front: vec![0],
            questions: vec![Question {
                instructions: literal("Is it a picture?"),
                kind: QuestionKind::Predicate {
                    truthy: Content::Null,
                    falsy: Content::Null,
                },
            }],
        };
        let prepared = decider.prepare(&request).unwrap();
        for definition in [&served, &declared] {
            let adapter = family
                .input_adapter(definition, &TokenizerMarkers(&tokenizer))
                .unwrap();
            let plan = prepared
                .readout_plan(&request, |tokens, images| {
                    let media = media_processor(definition).unwrap();
                    prepare_model_input(definition, media.as_ref(), adapter.as_ref(), tokens, images)
                })
                .unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(plan.sequences().len(), 1);
        }
    }

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
                &headers::target("qwen3.6-35b-a3b:gguf:q4"),
                None,
                PackageIdentity {
                    target: ArtifactIdentity([7; 32]),
                    projector: None,
                },
            )
            .unwrap();
        let mut directory = headers::draft("qwen3.6-35b-a3b:gguf:q4");
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
        directory.metadata.last_mut().unwrap().value =
            Value::Array(vec![true, true, true, true, true, false].into());
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
                &headers::target("qwen3.6-35b-a3b:gguf:q4"),
                None,
                identity,
            )
            .unwrap();
        // A Gemma assistant reaches the assistant family, which names why it
        // is not a draft of this target; the target is kept whole.
        let mut assistant = headers::draft("qwen3.6-35b-a3b:gguf:q4");
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
                &headers::target("gemma-4-12b-it-qat:gguf:q4"),
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
            [UnsupportedReason::DraftArchitecture {
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

    /// QA (pq-deb13, 4 vCPU): a 2560×1600 image was one 4000-cell encode
    /// that held the CPU engine for ~57 minutes after its client left. A
    /// load's host resizes images to the cells its device encodes: on the
    /// CPU a large image is resized down to them rather than refused, and on
    /// a GPU it encodes as the resolved host resizes it.
    #[test]
    fn a_large_image_is_resized_to_the_cells_its_device_encodes() {
        use magnitude_artifacts::{ArtifactIdentity, PackageIdentity};
        use magnitude_executor::{max_image_cells, CPU_IMAGE_CELLS};
        use magnitude_family_common::headers;
        use seismic::BackendName;
        let mut resolved = magnitude_family_qwen35::Qwen35Family
            .inspect(
                &headers::target("qwen3.5-4b:gguf:q4"),
                Some(&headers::projector("qwen3.5-4b:gguf:q4")),
                PackageIdentity {
                    target: ArtifactIdentity([7; 32]),
                    projector: Some(ArtifactIdentity([8; 32])),
                },
            )
            .unwrap();
        let any_device = image_cell_limit(&resolved, 512, MAX_IMAGE_CELLS).unwrap();
        bound_images(&mut resolved, any_device).unwrap();
        let cell = {
            let p = &resolved.vision.as_ref().unwrap().preprocessing;
            (p.patch * p.merge) as usize
        };
        let encoded = |definition: &ModelDefinition, height, width| {
            let (height, width) = media_processor(definition)
                .unwrap()
                .unwrap()
                .resized_size(height, width)
                .unwrap();
            height * width / (cell * cell)
        };
        assert_eq!(encoded(&resolved, 1600, 2560), 4000);
        for (backend, large, small) in [
            (BackendName::Cpu, CPU_IMAGE_CELLS, 64),
            (BackendName::Metal, 4000, 64),
            (BackendName::Cuda, 4000, 64),
            (BackendName::Vulkan, 4000, 64),
        ] {
            let cells = image_cell_limit(&resolved, 512, max_image_cells(backend)).unwrap();
            let mut load = resolved.clone();
            bound_images(&mut load, cells).unwrap();
            let resized = encoded(&load, 1600, 2560);
            assert!(resized <= cells, "{backend:?}: {resized} cells over {cells}");
            assert!(resized * 4 >= large * 3, "{backend:?}: {resized} cells, not ~{large}");
            assert_eq!(encoded(&load, 256, 256), small, "{backend:?}");
        }
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
