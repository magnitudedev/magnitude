//! Device-free composition of a recognized package with generic chat and
//! media preparation. Numerical executors are attached by the execution
//! composition layer; model-family crates remain limited to varying facts.

use magnitude_artifacts::{ImageProcessor, InputLayout, Package, TokenId};
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    wire::PreparedGeneration,
    ByteBpeTokenizer, PreparedVocabulary, SpecialTokens, TemplateBundle,
};
use magnitude_model_contracts::{
    ModelDefinition, ModelInputAdapter, PreparedModelInput, TokenPlan,
};
use magnitude_model_executor::{platform::DeviceRequest, ExecutionPath};
use magnitude_model_qwen35::{
    inputs::{QwenImageTokens, QwenInputAdapter},
    inspect_package,
};
use magnitude_service::ServiceLimits;
use std::path::{Path, PathBuf};
use std::{rc::Rc, sync::Arc};

use crate::options::{ExecutionManifest, ModelPolicy, PackageOptions, ReadyInfo};
use crate::{
    chat::CacheLimits,
    service::EngineService,
    serving::{Config as ServerConfig, CountInput, Server},
};

const MAX_IMAGE_SOURCE_BYTES: usize = 24 << 20;

/// Host-owned authority for resolving chat image sources. Data URLs are
/// always local and bounded; filesystem access is opt-in and confined to
/// canonical roots chosen by the host.
#[derive(Clone)]
pub struct MediaSourcePolicy {
    file_roots: Vec<std::path::PathBuf>,
}

impl MediaSourcePolicy {
    pub const fn data_urls_only() -> Self {
        Self {
            file_roots: Vec::new(),
        }
    }

    pub fn with_file_roots(
        roots: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> Result<Self, String> {
        let file_roots = roots
            .into_iter()
            .map(|root| {
                root.as_ref()
                    .canonicalize()
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if file_roots.is_empty() {
            return Err("file media policy requires at least one canonical root".into());
        }
        Ok(Self { file_roots })
    }

    pub fn resolve(&self, source: &str) -> Result<Vec<u8>, String> {
        if let Some(data) = source.strip_prefix("data:") {
            let (metadata, payload) = data
                .split_once(',')
                .ok_or("image data URL has no payload delimiter")?;
            if !matches!(
                metadata,
                "image/png;base64" | "image/jpeg;base64" | "image/webp;base64"
            ) {
                return Err("image data URL requires PNG, JPEG, or WebP base64".into());
            }
            let decoded = base64::decode(payload)
                .map_err(|error| format!("image data URL base64 is invalid: {error}"))?;
            if decoded.is_empty() || decoded.len() > MAX_IMAGE_SOURCE_BYTES {
                return Err("decoded image data is empty or exceeds its limit".into());
            }
            return Ok(decoded);
        }
        if self.file_roots.is_empty() {
            return Err("file image sources are disabled".into());
        }
        let path = source.strip_prefix("file://").unwrap_or(source);
        let path = Path::new(path)
            .canonicalize()
            .map_err(|error| format!("image file cannot be resolved: {error}"))?;
        if !self.file_roots.iter().any(|root| path.starts_with(root)) {
            return Err("image file is outside the configured roots".into());
        }
        let metadata = path
            .metadata()
            .map_err(|error| format!("image file metadata failed: {error}"))?;
        let length =
            usize::try_from(metadata.len()).map_err(|_| "image file size exceeds host domain")?;
        if length == 0 || length > MAX_IMAGE_SOURCE_BYTES {
            return Err("image file is empty or exceeds its limit".into());
        }
        let bytes =
            std::fs::read(path).map_err(|error| format!("image file read failed: {error}"))?;
        if bytes.len() != length {
            return Err("image file changed while it was being read".into());
        }
        Ok(bytes)
    }
}

pub struct LoadedArtifacts {
    package: Arc<Package>,
    definition: ModelDefinition,
    declared_context_limit: u64,
    tokenizer: Arc<ByteBpeTokenizer>,
    templates: TemplateBundle,
    media: Option<ImageProcessor>,
    input_adapter: QwenInputAdapter,
}

/// Package and model interpretation needed to admit a numerical worker.
/// Host-only tokenizer and template construction is completed during loading.
pub struct PreparedArtifacts {
    package: Arc<Package>,
    definition: ModelDefinition,
    declared_context_limit: u64,
}

impl PreparedArtifacts {
    fn from_package(package: Package) -> Result<Self, String> {
        let definition = inspect_package(&package).map_err(|error| error.to_string())?;
        let declared_context_limit = definition.geometry.context_limit;
        Ok(Self {
            package: Arc::new(package),
            definition,
            declared_context_limit,
        })
    }

    pub fn package(&self) -> &Package {
        &self.package
    }

    pub fn shared_package(&self) -> Arc<Package> {
        self.package.clone()
    }

    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

    pub(crate) fn finish(self) -> Result<LoadedArtifacts, String> {
        let Self {
            package,
            definition,
            declared_context_limit,
        } = self;
        LoadedArtifacts::from_prepared(package, definition, declared_context_limit)
    }
}

/// Host-owned inputs to engine construction. Resolving this value performs all
/// filesystem and family interpretation needed before the numerical worker is
/// started, but it never opens a device or imports a tensor.
#[derive(Clone, Debug)]
pub struct EngineConfiguration {
    pub package: PackageOptions,
    pub model: ModelPolicy,
    /// Host-selected serving context. The executor sizes all context-bound
    /// state to this limit. `None` serves the artifact's declared maximum.
    pub context_tokens: Option<usize>,
    pub service: ServiceLimits,
    pub path: ExecutionPath,
    /// The device the numerical worker opens; the native path runs on its
    /// backend.
    pub device: DeviceRequest,
    pub control_capacity: usize,
    /// The directory the host keeps formed kernels and tuning results in
    /// (`--cache-dir`; ACN names `<dataDir>/cache/kernels`). `None` caches
    /// nothing: every load forms and tunes every entry.
    pub kernel_cache: Option<PathBuf>,
}

/// The two authorities produced by host resolution: local chat artifacts stay
/// on the host, while the owned device-free manifest is sent to the numerical
/// worker. Neither side reparses the other's representation.
pub struct ResolvedEngineConfiguration {
    pub artifacts: PreparedArtifacts,
    pub manifest: ExecutionManifest,
    pub control_capacity: usize,
}

/// Fully committed engine: host chat/media assets remain host-local while the
/// numerical service owns its device, package mapping, weights, and executors.
pub struct ReadyEngine {
    artifacts: LoadedArtifacts,
    vocabulary: PreparedVocabulary,
    service: EngineService,
    ready: ReadyInfo,
}

impl ReadyEngine {
    pub(crate) fn new(
        artifacts: LoadedArtifacts,
        vocabulary: PreparedVocabulary,
        service: EngineService,
        ready: ReadyInfo,
    ) -> Self {
        Self {
            artifacts,
            vocabulary,
            service,
            ready,
        }
    }

    pub const fn ready_info(&self) -> &ReadyInfo {
        &self.ready
    }

    pub fn client(&self) -> crate::service::EngineClient {
        self.service.client()
    }

    pub fn into_server(
        self,
        media_policy: MediaSourcePolicy,
        constraint_cache: CacheLimits,
        config: ServerConfig,
    ) -> Result<Server, String> {
        let expected_context = usize::try_from(self.artifacts.definition.geometry.context_limit)
            .map_err(|_| "model context limit exceeds host domain")?;
        let expected_vocabulary = usize::try_from(self.artifacts.definition.geometry.vocabulary)
            .map_err(|_| "model vocabulary exceeds host domain")?;
        if config.context_tokens != expected_context
            || config.vocabulary != expected_vocabulary
            || config.method != self.ready.model.method.policy()
        {
            return Err("server model limits differ from the ready engine".into());
        }
        let LoadedArtifacts {
            package: _,
            definition,
            declared_context_limit,
            input_adapter,
            tokenizer,
            templates,
            media,
        } = self.artifacts;
        let vocabulary = self.vocabulary.with_cache_limits(constraint_cache);
        let count_context_tokens = usize::try_from(declared_context_limit)
            .map_err(|_| "artifact context limit exceeds host domain")?;
        let mut count_definition = definition.clone();
        count_definition.geometry.context_limit = declared_context_limit;
        let media_marker = media.as_ref().map(|_| QWEN_IMAGE_PLACEHOLDER.to_owned());
        let count_adapter = input_adapter.clone();
        let count_media = media.clone();
        let count_policy = media_policy.clone();
        let count_input = CountInput {
            context_tokens: count_context_tokens,
            prepare: Rc::new(move |request: &PreparedGeneration| {
                prepare_with_media(
                    &count_definition,
                    &count_adapter,
                    count_media.as_ref(),
                    request,
                    |source| count_policy.resolve(source),
                )
            }),
        };
        let prepare_input = Rc::new(move |request: &PreparedGeneration| {
            prepare_with_media(
                &definition,
                &input_adapter,
                media.as_ref(),
                request,
                |source| media_policy.resolve(source),
            )
        });
        Server::new(
            self.service,
            tokenizer,
            templates,
            vocabulary,
            prepare_input,
            count_input,
            media_marker,
            config,
        )
    }
}

/// The text one image renders as in the Qwen chat template: the image pad
/// between the vision delimiters, which input preparation expands to the
/// image's merged patch rows.
const QWEN_IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

impl EngineConfiguration {
    pub fn resolve(self) -> Result<ResolvedEngineConfiguration, String> {
        if self.control_capacity == 0 {
            return Err("engine control capacity must be positive".into());
        }
        let package = self.package.open().map_err(|error| error.to_string())?;
        let mut artifacts = PreparedArtifacts::from_package(package)?;
        artifacts.definition.geometry.context_limit = resolve_served_context(
            self.context_tokens,
            artifacts.definition.geometry.context_limit,
        )?;
        let model = self.model.resolve(artifacts.definition())?;
        let manifest = ExecutionManifest::new(
            artifacts.package().manifest(),
            artifacts.definition().clone(),
            model,
            self.service,
            self.path,
            self.device,
            self.kernel_cache,
        )?;
        Ok(ResolvedEngineConfiguration {
            artifacts,
            manifest,
            control_capacity: self.control_capacity,
        })
    }
}

fn resolve_served_context(requested: Option<usize>, artifact_limit: u64) -> Result<u64, String> {
    let Some(requested) = requested else {
        return Ok(artifact_limit);
    };
    let requested =
        u64::try_from(requested).map_err(|_| "serving context exceeds the model domain")?;
    if requested == 0 || requested > artifact_limit {
        return Err(format!(
            "serving context {requested} is outside the artifact limit {artifact_limit}"
        ));
    }
    Ok(requested)
}

impl ResolvedEngineConfiguration {
    pub fn start(self) -> Result<ReadyEngine, String> {
        crate::execution::start(self)
    }
}

impl LoadedArtifacts {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let package = Package::open(path).map_err(|error| error.to_string())?;
        Self::from_package(package)
    }

    pub fn open_without_projector(path: impl AsRef<Path>) -> Result<Self, String> {
        let package = Package::open_without_projector(path).map_err(|error| error.to_string())?;
        Self::from_package(package)
    }

    pub fn open_with_projector(
        target: impl AsRef<Path>,
        projector: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let package =
            Package::open_with_projector(target, projector).map_err(|error| error.to_string())?;
        Self::from_package(package)
    }

    fn from_package(package: Package) -> Result<Self, String> {
        PreparedArtifacts::from_package(package)?.finish()
    }

    fn from_prepared(
        package: Arc<Package>,
        definition: ModelDefinition,
        declared_context_limit: u64,
    ) -> Result<Self, String> {
        let artifact_identity = definition.artifact_identity.target.to_string();
        let tokenizer = Arc::new(ByteBpeTokenizer::new(gguf_byte_bpe(
            package.tokenizer(),
            artifact_identity,
        )?)?);
        if u64::try_from(tokenizer.vocabulary()).ok() != Some(definition.geometry.vocabulary) {
            return Err("tokenizer vocabulary differs from model vocabulary".into());
        }
        let templates = gguf_templates(package.templates(), package.tokenizer())?;
        let media = definition
            .vision
            .as_ref()
            .map(|vision| {
                let processor = ImageProcessor::new(
                    vision
                        .image_processor_config()
                        .map_err(|error| error.to_string())?,
                )?;
                Ok::<_, String>(processor)
            })
            .transpose()?;
        let image_tokens = if media.is_some() {
            Some(
                QwenImageTokens::new(
                    special_token(&tokenizer, "<|image_pad|>")?,
                    special_token(&tokenizer, "<|vision_start|>")?,
                    special_token(&tokenizer, "<|vision_end|>")?,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        Ok(Self {
            package,
            definition,
            declared_context_limit,
            tokenizer,
            templates,
            media,
            input_adapter: QwenInputAdapter::new(image_tokens),
        })
    }

    pub fn package(&self) -> &Package {
        &self.package
    }

    pub fn shared_package(&self) -> Arc<Package> {
        self.package.clone()
    }

    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

    pub fn tokenizer(&self) -> &ByteBpeTokenizer {
        self.tokenizer.as_ref()
    }

    pub(crate) fn shared_tokenizer(&self) -> Arc<ByteBpeTokenizer> {
        self.tokenizer.clone()
    }

    pub fn templates(&self) -> &TemplateBundle {
        &self.templates
    }

    /// Resolve validated image sources under host policy, run the processor
    /// bound into the package identity, and let the family adapter install the
    /// resulting spans and rotary coordinates. Text-only input never touches
    /// the media path.
    pub fn prepare_input(
        &self,
        request: &PreparedGeneration,
        resolve: impl FnMut(&str) -> Result<Vec<u8>, String>,
    ) -> Result<PreparedModelInput, String> {
        prepare_with_media(
            &self.definition,
            &self.input_adapter,
            self.media.as_ref(),
            request,
            resolve,
        )
    }
}

fn prepare_with_media(
    definition: &ModelDefinition,
    adapter: &QwenInputAdapter,
    media: Option<&ImageProcessor>,
    request: &PreparedGeneration,
    mut resolve: impl FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<PreparedModelInput, String> {
    let tokens = request.chat.input().tokens.clone();
    let layout = InputLayout::new(tokens.len(), Vec::new())?;
    let token_plan = TokenPlan::new(tokens, layout).map_err(|error| error.to_string())?;
    if request.image_sources.is_empty() {
        return adapter
            .prepare(definition, token_plan, &[])
            .map_err(|error| error.to_string());
    }
    let processor = media.ok_or("loaded model has no image processor")?;
    let encoded = request
        .image_sources
        .iter()
        .map(|source| resolve(source))
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = processor.prepare(&encoded)?;
    adapter
        .prepare(definition, token_plan, &[prepared])
        .map_err(|error| error.to_string())
}

fn special_token(tokenizer: &ByteBpeTokenizer, value: &str) -> Result<TokenId, String> {
    let tokens = tokenizer.encode(value, SpecialTokens::Recognize)?;
    let [token] = tokens.as_slice() else {
        return Err(format!("Qwen special token {value:?} is not atomic"));
    };
    if tokenizer.piece(*token, false)? != value.as_bytes() {
        return Err(format!(
            "Qwen special token {value:?} has a different vocabulary value"
        ));
    }
    Ok(*token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_policy_accepts_only_bounded_supported_data_urls() {
        let policy = MediaSourcePolicy::data_urls_only();
        assert_eq!(
            policy.resolve("data:image/png;base64,AA==").unwrap(),
            vec![0]
        );
        assert!(policy
            .resolve("data:image/svg+xml;base64,PHN2Zy8+")
            .is_err());
        assert!(policy.resolve("data:image/png,raw").is_err());
        assert!(policy.resolve("/tmp/image.png").is_err());
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
