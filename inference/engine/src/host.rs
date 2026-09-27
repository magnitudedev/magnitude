//! Host-side model artifacts: everything chat semantics need, device-free.
//!
//! One resolution produces them (`EngineConfiguration::resolve`); request
//! preparation, host-only operations (count, template apply, properties) and
//! loads all share that value. They never open a device.

use crate::error::{ArtifactError, RequestError, ResolveError, UnsupportedModel};
use crate::families;
use crate::options::InputModalities;
use magnitude_artifacts::{ImageProcessor, InputLayout, Package, TokenId};
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    request::ImageInput,
    ByteBpeTokenizer, SpecialTokens, TemplateBundle, TemplateInspection,
};
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, MarkerTokens, ModelDefinition, ModelFamily,
    PreparedModelInput, TokenPlan,
};
use std::sync::Arc;

/// The package's chat and input semantics: tokenizer, templates, media
/// processor and the recognized family's input adapter.
pub struct HostArtifacts {
    package: Arc<Package>,
    family: &'static dyn ModelFamily,
    /// The definition at the served context bound.
    definition: ModelDefinition,
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
            .map_err(FamilyError)?;
        let [token] = tokens.as_slice() else {
            return Err(FamilyError(format!("special token {text:?} is not atomic")));
        };
        if self.0.piece(*token, false).map_err(FamilyError)? != text.as_bytes() {
            return Err(FamilyError(format!(
                "special token {text:?} has a different vocabulary value"
            )));
        }
        Ok(*token)
    }
}

/// An opened package's family and declared definition: the recognized
/// target family's definition, with the package's separate draft bound
/// against it (the target family maps the draft's tapped layers onto its
/// sublayers). Reads no tokenizer or template.
pub fn package_definition(
    package: &Package,
) -> Result<(&'static dyn ModelFamily, ModelDefinition), ResolveError> {
    let unsupported =
        |reason: String| ResolveError::Unsupported(UnsupportedModel::Representation { reason });
    let target = package.target().directory();
    let family = families::recognize(target).map_err(ResolveError::Unsupported)?;
    let declared = family
        .inspect(
            target,
            package.projector().map(|projector| projector.directory()),
            package.identity(),
        )
        .map_err(|error| unsupported(error.0))?;
    let Some(draft) = package.draft() else {
        return Ok((family, declared));
    };
    let draft = magnitude_family_dflash::inspect(draft.directory(), &declared, &|layer| {
        family.layer_entry(&declared, layer)
    })
    .map_err(|error| unsupported(format!("draft: {error}")))?;
    Ok((
        family,
        ModelDefinition {
            draft: Some(draft),
            ..declared
        },
    ))
}

impl HostArtifacts {
    /// Interpret an opened package: family recognition, definition,
    /// tokenizer, templates, media processor and input adapter.
    /// `served_context` bounds the served definition within the declared one.
    pub(crate) fn interpret(
        package: Package,
        served_context: Option<usize>,
    ) -> Result<Self, ResolveError> {
        let (family, declared) = package_definition(&package)?;
        let mut definition = declared.clone();
        definition.decoder.context_limit =
            resolve_served_context(served_context, declared.decoder.context_limit)?;
        let unsupported = |reason: String| {
            ResolveError::Unsupported(UnsupportedModel::Representation { reason })
        };
        let tokenizer = Arc::new(
            gguf_byte_bpe(
                package.tokenizer(),
                definition.artifact_identity.target.to_string(),
            )
            .and_then(ByteBpeTokenizer::new)
            .map_err(|error| unsupported(error.to_string()))?,
        );
        if u64::try_from(tokenizer.vocabulary()).ok() != Some(definition.decoder.vocabulary) {
            return Err(unsupported(
                "tokenizer vocabulary differs from model vocabulary".into(),
            ));
        }
        let templates =
            gguf_templates(package.templates(), package.tokenizer()).map_err(unsupported)?;
        let inspection = templates.inspect().map_err(unsupported)?;
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
            .map_err(|error| unsupported(error.0))?;
        Ok(Self {
            package: Arc::new(package),
            family,
            definition,
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
        let opened = package
            .open()
            .map_err(|error| ResolveError::Artifact(ArtifactError::from_artifacts(error, &package.target)))?;
        Self::interpret(opened, None)
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

fn resolve_served_context(requested: Option<usize>, artifact_limit: u64) -> Result<u64, ResolveError> {
    let Some(requested) = requested else {
        return Ok(artifact_limit);
    };
    let invalid = || ResolveError::InvalidConfiguration {
        reason: format!("serving context {requested} is outside the artifact limit {artifact_limit}"),
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
