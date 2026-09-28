//! Configuration split by authority and resolution time.
//!
//! Package selection is resolved by the host. Model policy is resolved against
//! a validated family definition. The resulting execution manifest is the only value
//! transported into the numerical worker.

use magnitude_artifacts::{Package, PackageIdentity, PackageManifest};
use magnitude_chat::generation::MethodPolicy;
use magnitude_generation::{DFlash, Method, Mtp, Plain};
use magnitude_family_contracts::{DraftVariant, ModelDefinition, ModelFamily, Operator};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath, ResourcePlan, MAX_DRAFT_PROPOSALS,
};

/// Dense-target MTP width when none is requested.
const DEFAULT_PROPOSALS: u8 = 3;
use magnitude_state::KvCodec;
use magnitude_scheduler::ServiceLimits;
use crate::census::AllocationCensus;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc};

/// The standalone engine's service limits. Loading and metadata-only
/// assessment use the same token and time-share policy.
pub fn standard_service_limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 512,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 1.0,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectorSelection {
    Discover,
    Disabled,
    Explicit(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageOptions {
    pub target: PathBuf,
    pub projector: ProjectorSelection,
    /// A separate draft model (DFlash, DSpark) for the target.
    pub draft: Option<PathBuf>,
}

impl PackageOptions {
    pub fn open(&self) -> Result<Package, magnitude_artifacts::Error> {
        let package = match &self.projector {
            ProjectorSelection::Discover => Package::open(&self.target),
            ProjectorSelection::Disabled => Package::open_without_projector(&self.target),
            ProjectorSelection::Explicit(projector) => {
                Package::open_with_projector(&self.target, projector)
            }
        }?;
        match &self.draft {
            Some(draft) => package.with_draft(draft),
            None => Ok(package),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelMethod {
    /// A package's separate draft when it has one, else its draft head,
    /// else plain generation.
    #[default]
    Auto,
    Plain,
    Mtp,
    /// A separate draft of the named variant. The draft's own definition
    /// decides its variant; requesting one the package's draft is not fails.
    DFlash,
    DSpark,
    DFlash2,
}

impl ModelMethod {
    /// The separate-draft variant this method requests, if it requests one.
    fn draft_variant(self) -> Option<DraftVariant> {
        match self {
            Self::DFlash => Some(DraftVariant::DFlash),
            Self::DSpark => Some(DraftVariant::DSpark),
            Self::DFlash2 => Some(DraftVariant::DFlash2),
            Self::Auto | Self::Plain | Self::Mtp => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPolicy {
    pub method: ModelMethod,
    /// Explicit proposal width of the drafter (MTP or DFlash) for both greedy
    /// and sampled requests. When absent, the measured fixed-width policy is
    /// resolved from model shape.
    pub mtp_proposals: Option<u8>,
    pub kv_codec: KvCodec,
    /// Queue each plain decode step's successor on the device before the
    /// step completes (cross-step pipelining).
    pub lookahead: bool,
}

impl Default for ModelPolicy {
    fn default() -> Self {
        Self {
            method: ModelMethod::Auto,
            mtp_proposals: None,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolvedMethod {
    Plain,
    Mtp {
        greedy_proposals: u8,
        sampled_proposals: u8,
    },
    /// A separate draft drafting `proposals` tokens per block.
    DFlash { proposals: u8 },
}

impl ResolvedMethod {
    /// The widest draft any request may use: the head graphs are sealed for
    /// every width up to it. Zero for plain generation.
    pub fn proposals(self) -> usize {
        match self {
            Self::Plain => 0,
            Self::Mtp {
                greedy_proposals,
                sampled_proposals,
            } => usize::from(greedy_proposals.max(sampled_proposals)),
            Self::DFlash { proposals } => usize::from(proposals),
        }
    }

    pub const fn policy(self) -> MethodPolicy {
        match self {
            Self::Plain => MethodPolicy::Plain,
            Self::Mtp {
                greedy_proposals,
                sampled_proposals,
                ..
            } => MethodPolicy::Mtp {
                greedy_proposals,
                sampled_proposals,
            },
            Self::DFlash { proposals } => MethodPolicy::DFlash { proposals },
        }
    }

    pub fn factory(self, artifact_identity: &str) -> Result<Arc<dyn Method>, String> {
        match self {
            Self::Plain => Ok(Arc::new(Plain)),
            Self::Mtp { .. } => Ok(Arc::new(Mtp::new(artifact_identity, self.proposals())?)),
            Self::DFlash { .. } => Ok(Arc::new(DFlash::new(artifact_identity, self.proposals())?)),
        }
    }

    /// The definition a load of this method executes: it carries at most
    /// the drafter the method runs (a plain load keeps its draft head, which
    /// it never selects).
    pub fn executed(self, definition: ModelDefinition) -> ModelDefinition {
        match self {
            Self::DFlash { .. } => ModelDefinition {
                head: None,
                ..definition
            },
            Self::Plain | Self::Mtp { .. } => ModelDefinition {
                draft: None,
                ..definition
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModelPolicy {
    pub method: ResolvedMethod,
    #[serde(with = "KvCodecEncoding")]
    pub kv_codec: KvCodec,
    pub lookahead: bool,
}

impl ModelPolicy {
    pub fn resolve(&self, definition: &ModelDefinition) -> Result<ResolvedModelPolicy, String> {
        definition.validate().map_err(|error| error.to_string())?;
        let method = resolve_method(self.method, self.mtp_proposals, definition)?;
        Ok(ResolvedModelPolicy {
            method,
            kv_codec: self.kv_codec,
            // A per-layer entry gathers host-table rows by host tokens, so
            // its steps cannot chain on device-selected tokens.
            lookahead: self.lookahead && definition.decoder.entry.per_layer.is_none(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionManifest {
    pub package: PackageManifest,
    pub definition: ModelDefinition,
    pub model: ResolvedModelPolicy,
    #[serde(with = "ServiceLimitsEncoding")]
    pub service: ServiceLimits,
    pub path: ExecutionPath,
    pub device: DeviceRequest,
    /// The kernel cache directory the host names; `None` caches nothing.
    pub kernel_cache: Option<PathBuf>,
    /// The host's threshold policy: every engine claim keeps each domain's
    /// headroom above its planning reserve.
    pub reserves: MemoryReserves,
}

/// Device-free readiness evidence returned only after worker construction has
/// committed the complete resource plan and qualified execution pack.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePlanSummary {
    pub domain_capacity_bytes: u64,
    /// Planned charges when every selected component is resident, not the
    /// opened device's current Seismic charge.
    pub planned_bytes: u64,
    pub immutable_bytes: u64,
    pub state_bytes: u64,
    pub scratch_bytes: u64,
    pub active_sequences: usize,
    pub in_flight_sequences: usize,
    pub retained_sequences: usize,
    pub retention_entries: usize,
    pub history_rows: usize,
}

impl ResourcePlanSummary {
    pub fn from_plan(plan: &ResourcePlan) -> Result<Self, String> {
        let bytes = plan.bytes();
        let capacity = plan.capacity();
        Ok(Self {
            domain_capacity_bytes: plan.domain_capacity_bytes(),
            planned_bytes: bytes.total()?,
            immutable_bytes: bytes
                .target_weights
                .checked_add(bytes.head_weights)
                .and_then(|total| total.checked_add(bytes.vision_weights))
                .ok_or("immutable resource summary overflow")?,
            state_bytes: bytes
                .history
                .checked_add(bytes.recurrent_banks)
                .ok_or("state resource summary overflow")?,
            scratch_bytes: bytes.scratch,
            active_sequences: capacity.live_requests,
            in_flight_sequences: capacity.live_requests,
            retained_sequences: capacity
                .checkpoints
                .checked_add(capacity.retention_entries)
                .ok_or("retained sequence summary overflow")?,
            retention_entries: capacity.retention_entries,
            history_rows: capacity.history_rows,
        })
    }
}

/// The inputs a model accepts besides text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputModalities {
    /// Images: the definition has a vision component and its family renders
    /// images in the chat template.
    pub image: bool,
}

impl InputModalities {
    pub fn of(family: &dyn ModelFamily, definition: &ModelDefinition) -> Self {
        Self {
            image: definition.vision.is_some() && family.media_placeholder(definition).is_some(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadyInfo {
    pub package: PackageIdentity,
    /// [`magnitude_chat::TemplateInspection::fingerprint`] of the chat
    /// templates the worker read from its own opened package.
    pub template_fingerprint: String,
    /// The inputs the worker's loaded model accepts, from its own opened
    /// package's family and the definition it executes.
    pub modalities: InputModalities,
    pub model: ResolvedModelPolicy,
    #[serde(with = "ServiceLimitsEncoding")]
    pub service: ServiceLimits,
    pub resources: ResourcePlanSummary,
    pub path: ExecutionPath,
    /// The device the worker resolved and opened.
    pub device: seismic::DeviceSelector,
    /// The backend of the opened device the native path executes on.
    pub backend: seismic::BackendName,
    /// The memory heap's standing when readiness was published.
    pub census: AllocationCensus,
}

/// The serialized form of [`ServiceLimits`] (a scheduler value).
#[derive(Serialize, Deserialize)]
#[serde(remote = "ServiceLimits")]
struct ServiceLimitsEncoding {
    prefill_tokens: usize,
    decode_tokens: usize,
    decode_share: f64,
    locality_seconds: f64,
}

/// The serialized form of [`KvCodec`] (a state value).
#[derive(Serialize, Deserialize)]
#[serde(remote = "KvCodec")]
enum KvCodecEncoding {
    Dense,
    AffineK8V4,
    RotatedK4V4,
}

impl ExecutionManifest {
    pub fn new(
        package: PackageManifest,
        definition: ModelDefinition,
        model: ResolvedModelPolicy,
        service: ServiceLimits,
        path: ExecutionPath,
        device: DeviceRequest,
        kernel_cache: Option<PathBuf>,
        reserves: MemoryReserves,
    ) -> Result<Self, String> {
        let definition = model.method.executed(definition);
        definition.validate().map_err(|error| error.to_string())?;
        service.validate()?;
        if package.identity != definition.artifact_identity {
            return Err("package manifest and model definition identities differ".into());
        }
        Ok(Self {
            package,
            definition,
            model,
            service,
            path,
            device,
            kernel_cache,
            reserves,
        })
    }
}

fn resolve_method(
    requested: ModelMethod,
    override_width: Option<u8>,
    definition: &ModelDefinition,
) -> Result<ResolvedMethod, String> {
    let head = definition.head.as_ref();
    let use_draft = match requested {
        ModelMethod::Auto => definition.draft.is_some(),
        ModelMethod::DFlash | ModelMethod::DSpark | ModelMethod::DFlash2 => true,
        ModelMethod::Plain | ModelMethod::Mtp => false,
    };
    if use_draft {
        let draft = definition
            .draft
            .as_ref()
            .ok_or("a separate-draft method was requested but the package has no draft")?;
        let variant = draft.method.variant();
        if let Some(requested) = requested.draft_variant() {
            if requested != variant {
                return Err(format!(
                    "{requested} was requested but the package's draft is {variant}"
                ));
            }
        }
        let bound = u8::try_from(draft.max_proposals())
            .map_err(|_| "the draft's block exceeds the proposal width range")?;
        let proposals = match override_width {
            Some(0) => return Err("mtp_proposals must be positive".into()),
            Some(width) if width > bound => {
                return Err(format!("the draft proposes at most {bound} tokens per block"))
            }
            Some(width) => width,
            None => DEFAULT_PROPOSALS.min(bound),
        };
        return Ok(ResolvedMethod::DFlash { proposals });
    }
    let use_mtp = match requested {
        // A head this executor does not run leaves the model plain.
        ModelMethod::Auto => magnitude_executor::head_admitted(definition),
        ModelMethod::Plain
        | ModelMethod::DFlash
        | ModelMethod::DSpark
        | ModelMethod::DFlash2 => false,
        ModelMethod::Mtp => true,
    };
    if !use_mtp {
        if override_width.is_some() {
            return Err("mtp_proposals requires the MTP method".into());
        }
        return Ok(ResolvedMethod::Plain);
    }
    head.ok_or("MTP was requested but the artifact has no draft head")?;
    let routed = definition
        .decoder
        .sublayers()
        .any(|(_, sublayer)| matches!(sublayer.op, Operator::RoutedFfn(_)));
    let (greedy_proposals, sampled_proposals) = match override_width {
        Some(0) => return Err("mtp_proposals must be positive".into()),
        Some(width) if width > MAX_DRAFT_PROPOSALS => {
            return Err(format!("mtp_proposals exceeds {MAX_DRAFT_PROPOSALS}"))
        }
        Some(width) => (width, width),
        // Every verify row of a routed target streams more experts.
        None if routed => (1, 1),
        None => (DEFAULT_PROPOSALS, DEFAULT_PROPOSALS),
    };
    Ok(ResolvedMethod::Mtp {
        greedy_proposals,
        sampled_proposals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::PackageIdentity;
    use magnitude_family_contracts::{
        ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
        EmbeddingScale, EntryForm, ExitForm, ExitNorm, FeedForwardUp, GateFunction, Head,
        HeadBlock, HeadNorm, HistoryDomain, HistoryReads, InputNorm, KeyValue, MediaRowAttention,
        OutputForm, ResidualForm, RmsNorm, Rotary, Sublayer, ValueNorm, ValueSource,
        WeightDescriptor,
    };

    fn weight(name: &str, shape: &[u64]) -> WeightDescriptor {
        WeightDescriptor::stored(name, shape)
    }

    fn rms(name: &str) -> RmsNorm {
        RmsNorm {
            weight: weight(name, &[2]),
            epsilon: 1e-6,
        }
    }

    /// One attention and one dense sublayer, their weights named with `p`.
    fn block(p: &str) -> Block {
        let attention = Attention {
            heads: 1,
            kv_heads: 1,
            width: 2,
            query: weight(&format!("{p}qg"), &[4, 2]),
            gate: AttentionGate::Interleaved {
                function: GateFunction::Sigmoid,
            },
            query_norm: HeadNorm::Rms(rms(&format!("{p}qn"))),
            key_value: KeyValue::Owned {
                key: weight(&format!("{p}k"), &[2, 2]),
                value: ValueSource::Projected(weight(&format!("{p}v"), &[2, 2])),
                key_norm: HeadNorm::Rms(rms(&format!("{p}kn"))),
                value_norm: ValueNorm::None,
                domain: HistoryDomain::Token,
            },
            rotary: Rotary::Interleaved {
                width: 2,
                base: 10_000.0,
                sections: vec![1],
                axis_pattern: vec![0],
            },
            scale: 1.0 / 2f64.sqrt(),
            reads: HistoryReads::Visible,
            media_rows: MediaRowAttention::Causal,
            output: weight(&format!("{p}o"), &[2, 2]),
        };
        let dense = DenseFfn {
            intermediate: 4,
            up: FeedForwardUp::Gated {
                activation: ActivationFunction::Silu,
                gate: weight(&format!("{p}g"), &[4, 2]),
                up: weight(&format!("{p}u"), &[4, 2]),
            },
            down: weight(&format!("{p}d"), &[2, 4]),
        };
        Block {
            sublayers: vec![
                Sublayer {
                    input: InputNorm::Rms(rms(&format!("{p}in"))),
                    op: Operator::Attention(Box::new(attention)),
                    output: OutputForm::Residual,
                },
                Sublayer {
                    input: InputNorm::Rms(rms(&format!("{p}fn"))),
                    op: Operator::DenseFfn(Box::new(dense)),
                    output: OutputForm::Residual,
                },
            ],
        }
    }

    fn definition(with_head: bool) -> ModelDefinition {
        let head = with_head.then(|| Head {
            blocks: vec![HeadBlock {
                embedding_norm: rms("en"),
                hidden_norm: rms("hn"),
                combine: weight("combine", &[2, 4]),
                block: block("h"),
                output_norm: ExitNorm::Rms(rms("hon")),
            }],
        });
        ModelDefinition {
            family: magnitude_family_contracts::FamilyId("fixture".into()),
            artifact_identity: PackageIdentity {
                target: magnitude_artifacts::ArtifactIdentity([1; 32]),
                projector: None,
            },
            inputs: magnitude_family_contracts::InputSemantics {
                coordinate_axes: 1,
                text_coordinates:
                    magnitude_family_contracts::TextCoordinateSemantics::ReplicatedPosition,
            },
            decoder: Decoder {
                activation_dtype: ActivationDType::BF16,
                hidden: 2,
                vocabulary: 8,
                context_limit: 128,
                residual: ResidualForm::Single,
                entry: EntryForm {
                    embedding: weight("embedding", &[8, 2]),
                    scale: EmbeddingScale::Unit,
                    norm: None,
                    per_layer: None,
                    hash_routing: None,
                },
                blocks: vec![block("")],
                exit: ExitForm {
                    norm: ExitNorm::Rms(rms("on")),
                    output: weight("out", &[8, 2]),
                    softcap: None,
                },
            },
            head,
            vision: None,
            draft: None,
        }
    }

    #[test]
    fn defaults_resolve_after_artifact_inspection() {
        let plain = ModelPolicy::default().resolve(&definition(false)).unwrap();
        assert_eq!(plain.method, ResolvedMethod::Plain);
        assert_eq!(plain.kv_codec, KvCodec::AffineK8V4);

        let mtp = ModelPolicy::default().resolve(&definition(true)).unwrap();
        assert_eq!(
            mtp.method,
            ResolvedMethod::Mtp {
                greedy_proposals: DEFAULT_PROPOSALS,
                sampled_proposals: DEFAULT_PROPOSALS,
            }
        );
    }

    #[test]
    fn explicit_method_and_budget_are_strict() {
        let mut options = ModelPolicy {
            method: ModelMethod::Mtp,
            mtp_proposals: Some(2),
            ..ModelPolicy::default()
        };
        assert!(options.resolve(&definition(false)).is_err());
        assert!(options.resolve(&definition(true)).is_ok());
        options.mtp_proposals = Some(MAX_DRAFT_PROPOSALS + 1);
        assert!(options.resolve(&definition(true)).is_err());
        options.method = ModelMethod::Plain;
        assert!(options.resolve(&definition(true)).is_err());
    }

    fn with_draft(method: magnitude_family_contracts::DraftMethod) -> ModelDefinition {
        use magnitude_family_contracts::{
            BlockLayout, DraftDefinition, DraftEmbedding, SublayerIndex, TapPoint, TokenId,
        };
        ModelDefinition {
            draft: Some(DraftDefinition {
                method,
                taps: vec![TapPoint::Sublayer(SublayerIndex {
                    block: 0,
                    sublayer: 0,
                })],
                fusion: weight("fc", &[2, 2]),
                fusion_norm: rms("enc"),
                embedding: DraftEmbedding::Target,
                blocks: vec![block("d")],
                output_norm: rms("don"),
                block_size: 4,
                mask_token: TokenId(7),
                layout: BlockLayout::MaskSlots,
            }),
            ..definition(true)
        }
    }

    /// A separate-draft method names the draft's variant; the package's
    /// draft must be of it, and `auto` takes whichever the draft is.
    #[test]
    fn a_requested_draft_variant_must_be_the_packages() {
        use magnitude_family_contracts::DraftMethod;
        let draft = with_draft(DraftMethod::DFlash);
        let policy = |method| ModelPolicy {
            method,
            ..ModelPolicy::default()
        };
        for method in [ModelMethod::Auto, ModelMethod::DFlash] {
            assert_eq!(
                policy(method).resolve(&draft).unwrap().method,
                ResolvedMethod::DFlash { proposals: 3 }
            );
        }
        for method in [ModelMethod::DSpark, ModelMethod::DFlash2] {
            let error = policy(method).resolve(&draft).unwrap_err();
            assert!(error.contains("the package's draft is DFlash"), "{error}");
        }
        assert!(policy(ModelMethod::DFlash)
            .resolve(&definition(true))
            .is_err());
        // Plain and MTP loads ignore the draft.
        assert_eq!(
            policy(ModelMethod::Plain).resolve(&draft).unwrap().method,
            ResolvedMethod::Plain
        );
    }

    #[test]
    fn execution_manifest_is_worker_transportable() {
        fn assert_send<T: Send>() {}
        assert_send::<ExecutionManifest>();
    }

    /// The manifest crosses to a worker process in the protocol encoding and
    /// arrives unchanged.
    #[test]
    fn execution_manifest_round_trips_through_the_worker_encoding() {
        use magnitude_artifacts::{
            gguf::{Encoding, TensorDescriptor},
            ComponentFile, ComponentManifest,
        };
        let definition = definition(true);
        let package = PackageManifest {
            identity: definition.artifact_identity,
            target: ComponentManifest {
                files: vec![ComponentFile {
                    path: PathBuf::from("/models/fixture.gguf"),
                    size: 4096,
                }],
                identity: definition.artifact_identity.target,
                tensors: vec![TensorDescriptor {
                    name: "embedding".into(),
                    shape: vec![8, 2],
                    encoding: Encoding::F32,
                    offset: 0,
                    nbytes: 64,
                }],
            },
            projector: None,
            draft: None,
        };
        let model = ModelPolicy::default().resolve(&definition).unwrap();
        let manifest = ExecutionManifest::new(
            package,
            definition,
            model,
            standard_service_limits(),
            ExecutionPath::Native,
            "metal:00000001000004a5".parse().unwrap(),
            Some(PathBuf::from("/cache/kernels")),
            MemoryReserves::standard(),
        )
        .unwrap();
        let encoded = postcard::to_allocvec(&manifest).unwrap();
        let decoded: ExecutionManifest = postcard::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.package, manifest.package);
        assert_eq!(decoded.definition, manifest.definition);
        assert_eq!(decoded.model, manifest.model);
        assert_eq!(decoded.device, manifest.device);
        assert_eq!(decoded.reserves, manifest.reserves);
        assert_eq!(decoded.kernel_cache, manifest.kernel_cache);
        assert_eq!(postcard::to_allocvec(&decoded).unwrap(), encoded);
    }
}
