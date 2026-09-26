//! Configuration split by authority and resolution time.
//!
//! Package selection is resolved by the host. Model policy is resolved against
//! a validated family definition. The resulting execution manifest is the only value
//! transported into the numerical worker.

use magnitude_artifacts::{Package, PackageIdentity, PackageManifest};
use magnitude_chat::generation::MethodPolicy;
use magnitude_generation::{Method, Mtp, Plain};
use magnitude_family_contracts::{FeedForwardGeometry, ModelDefinition, ModelFamily};
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
}

impl PackageOptions {
    pub fn open(&self) -> Result<Package, magnitude_artifacts::Error> {
        match &self.projector {
            ProjectorSelection::Discover => Package::open(&self.target),
            ProjectorSelection::Disabled => Package::open_without_projector(&self.target),
            ProjectorSelection::Explicit(projector) => {
                Package::open_with_projector(&self.target, projector)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelMethod {
    #[default]
    Auto,
    Plain,
    Mtp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPolicy {
    pub method: ModelMethod,
    /// Explicit proposal width for both greedy and sampled requests. When
    /// absent, the measured fixed-width policy is resolved from model shape.
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
        }
    }

    pub fn factory(self, artifact_identity: &str) -> Result<Arc<dyn Method>, String> {
        match self {
            Self::Plain => Ok(Arc::new(Plain)),
            Self::Mtp { .. } => Ok(Arc::new(Mtp::new(artifact_identity, self.proposals())?)),
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
            lookahead: self.lookahead,
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
    let use_mtp = match requested {
        ModelMethod::Auto => head.is_some(),
        ModelMethod::Plain => false,
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
        .geometry
        .blocks
        .iter()
        .any(|block| matches!(&block.feedforward, FeedForwardGeometry::Routed(_)));
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
        ActivationDType, AttentionGeometry, AttentionWeights, BlockGeometry, BlockWeights,
        DecoderGeometry, DenseFeedForwardWeights, FeedForwardWeights, HeadBlock, HeadWeights,
        MixerGeometry, MixerWeights, RotarySemantics, WeightDescriptor,
    };

    fn weight(name: &str, shape: &[u64]) -> WeightDescriptor {
        WeightDescriptor {
            name: name.into(),
            shape: shape.to_vec(),
        }
    }

    fn definition(with_head: bool) -> ModelDefinition {
        let attention_geometry = AttentionGeometry {
            heads: 1,
            kv_heads: 1,
            width: 2,
            rotary: RotarySemantics::Interleaved {
                width: 2,
                base: 10_000.0,
                sections: vec![1],
                axis_pattern: vec![0],
            },
        };
        let attention = || AttentionWeights {
            query_gate: weight("qg", &[4, 2]),
            key: weight("k", &[2, 2]),
            value: weight("v", &[2, 2]),
            query_norm: weight("qn", &[2]),
            key_norm: weight("kn", &[2]),
            output: weight("o", &[2, 2]),
        };
        let dense = || DenseFeedForwardWeights {
            gate: weight("g", &[4, 2]),
            up: weight("u", &[4, 2]),
            down: weight("d", &[2, 4]),
        };
        let geometry = DecoderGeometry {
            activation_dtype: ActivationDType::BF16,
            hidden: 2,
            vocabulary: 8,
            context_limit: 128,
            epsilon: 1e-6,
            blocks: vec![BlockGeometry {
                mixer: MixerGeometry::Attention(attention_geometry),
                feedforward: FeedForwardGeometry::Dense { intermediate: 4 },
            }],
        };
        let head = with_head.then(|| HeadWeights {
            blocks: vec![HeadBlock {
                embedding_norm: weight("en", &[2]),
                hidden_norm: weight("hn", &[2]),
                combine: weight("combine", &[2, 4]),
                input_norm: weight("hin", &[2]),
                attention: attention(),
                feedforward_norm: weight("hfn", &[2]),
                feedforward_geometry: FeedForwardGeometry::Dense { intermediate: 4 },
                feedforward: FeedForwardWeights::Dense(Box::new(dense())),
                output_norm: weight("hon", &[2]),
            }],
        });
        ModelDefinition {
            family: magnitude_family_contracts::FamilyId("fixture".into()),
            artifact_identity: PackageIdentity {
                target: magnitude_artifacts::ArtifactIdentity([1; 32]),
                projector: None,
            },
            geometry,
            inputs: magnitude_family_contracts::InputSemantics {
                coordinate_axes: 1,
                text_coordinates:
                    magnitude_family_contracts::TextCoordinateSemantics::ReplicatedPosition,
            },
            embedding: weight("embedding", &[8, 2]),
            blocks: vec![BlockWeights {
                input_norm: weight("in", &[2]),
                mixer: MixerWeights::Attention(Box::new(attention())),
                feedforward_norm: weight("fn", &[2]),
                feedforward: FeedForwardWeights::Dense(Box::new(dense())),
            }],
            output_norm: weight("on", &[2]),
            output: weight("out", &[8, 2]),
            head,
            vision: None,
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
