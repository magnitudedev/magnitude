//! The declared measurement plan: the complete, fixed list of classes a
//! backend's basis measures. It never depends on a model being assessed.
//!
//! Weight-streaming classes are measured for every resident representation
//! the load planner can produce for the backend. Geometry-keyed classes are
//! measured at the head geometries of the declared Qwen3.5 configurations,
//! which cover the release catalog's Qwen3.5-family models. A model whose
//! decode demand needs a key outside this plan is incompatible.

use super::basis::MeasurementKey;
use crate::{resident_element, resident_layout, AttentionShape, ExecutionPath};
use magnitude_artifacts::gguf::Encoding;
use seismic::{BackendName, DType, Element};

/// The activation dtype the Qwen3.5 decoder binds (`qwen35` definition).
pub const ACTIVATIONS: [DType; 1] = [DType::BF16];

/// Artifact encodings the load planner admits for decoder weights. Dense
/// encodings all become the activation dtype.
const ENCODINGS: [Encoding; 6] = [
    Encoding::BF16,
    Encoding::Q8_0,
    Encoding::Q4K,
    Encoding::Q5K,
    Encoding::Q6K,
    Encoding::Iq4Xs,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclaredFeedForward {
    Dense {
        intermediate: u64,
    },
    Routed {
        experts: u64,
        selected: u64,
        intermediate: u64,
        shared_intermediate: u64,
    },
}

/// Decoder geometry of one release-catalog model, from its GGUF header:
/// `embedding_length`, `vocab_size`, `attention.head_count[_kv]`,
/// `attention.key_length`, `rope.dimension_count`, `ssm.group_count`,
/// `ssm.time_step_rank`, `ssm.state_size`, `ssm.conv_kernel`,
/// `full_attention_interval` and the (expert) feed-forward lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeclaredConfiguration {
    pub model: &'static str,
    pub blocks: u64,
    /// Every `attention_interval`-th block is attention; the rest recurrent.
    pub attention_interval: u64,
    pub hidden: u64,
    pub vocabulary: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_width: u64,
    pub rotary_width: u64,
    pub key_heads: u64,
    pub value_heads: u64,
    pub state_width: u64,
    pub convolution_width: u64,
    pub feed_forward: DeclaredFeedForward,
}

impl DeclaredConfiguration {
    pub fn attention_shape(&self) -> AttentionShape {
        AttentionShape {
            hidden: self.hidden,
            kv_heads: self.kv_heads,
            group: self.heads / self.kv_heads,
            rotary_pairs: self.rotary_width / 2,
            width: self.head_width,
        }
    }
}

const fn qwen35(
    model: &'static str,
    blocks: u64,
    hidden: u64,
    heads: u64,
    kv_heads: u64,
    value_heads: u64,
    feed_forward: DeclaredFeedForward,
) -> DeclaredConfiguration {
    DeclaredConfiguration {
        model,
        blocks,
        attention_interval: 4,
        hidden,
        vocabulary: 248_320,
        heads,
        kv_heads,
        head_width: 256,
        rotary_width: 64,
        key_heads: 16,
        value_heads,
        state_width: 128,
        convolution_width: 4,
        feed_forward,
    }
}

/// The release catalog's Qwen3.5-family (`qwen35`, `qwen35moe`) models, read
/// from the pinned planner stubs of `inference/catalog/models.json`
/// (qwen3.8-flash-next is `qwen4exp`, muse-glimmer-30b its own family; the
/// V4 engine plans neither). `blocks` excludes the MTP layer.
pub const QWEN35_CONFIGURATIONS: [DeclaredConfiguration; 5] = [
    qwen35(
        "qwen3.5-4b",
        32,
        2560,
        16,
        4,
        32,
        DeclaredFeedForward::Dense { intermediate: 9216 },
    ),
    qwen35(
        "qwen3.5-9b",
        32,
        4096,
        16,
        4,
        32,
        DeclaredFeedForward::Dense {
            intermediate: 12_288,
        },
    ),
    qwen35(
        "qwen3.8-27b",
        64,
        5120,
        24,
        4,
        48,
        DeclaredFeedForward::Dense {
            intermediate: 17_408,
        },
    ),
    qwen35(
        "qwen3.6-35b-a3b",
        40,
        2048,
        16,
        2,
        32,
        DeclaredFeedForward::Routed {
            experts: 256,
            selected: 8,
            intermediate: 512,
            shared_intermediate: 512,
        },
    ),
    qwen35(
        "qwen3.5-122b-a10b",
        48,
        3072,
        32,
        2,
        64,
        DeclaredFeedForward::Routed {
            experts: 256,
            selected: 8,
            intermediate: 1024,
            shared_intermediate: 1024,
        },
    ),
];

/// Resident representations the native load planner produces on `backend`
/// with `activation` as the dense resident dtype, in declaration order.
pub fn resident_representations(backend: BackendName, activation: DType) -> Vec<Element> {
    let layout = resident_layout(ExecutionPath::Native, backend);
    let mut elements = Vec::new();
    for encoding in ENCODINGS {
        if let Some(element) = resident_element(encoding, activation, layout) {
            if !elements.contains(&element) {
                elements.push(element);
            }
        }
    }
    elements
}

/// Every class the basis of `backend` measures, in a fixed order.
pub fn measurement_plan(backend: BackendName) -> Vec<MeasurementKey> {
    let mut keys: Vec<MeasurementKey> = Vec::new();
    let mut push = |key: MeasurementKey| {
        if !keys.contains(&key) {
            keys.push(key);
        }
    };
    for dtype in ACTIVATIONS {
        let activation = Element::dense(dtype);
        // Norm vectors are resident in the activation dtype.
        let norm = activation;
        let representations = resident_representations(backend, dtype);
        for &weight in &representations {
            push(MeasurementKey::embedding_rows(weight, activation));
            push(MeasurementKey::attention_project(norm, weight, activation));
            push(MeasurementKey::attention_output(weight, activation));
            push(MeasurementKey::delta_project(norm, weight, activation));
            push(MeasurementKey::delta_output(norm, weight, activation));
            push(MeasurementKey::dense_expand(norm, weight, activation));
            push(MeasurementKey::dense_output(weight, activation));
            push(MeasurementKey::routed_expand(weight, activation));
            push(MeasurementKey::routed_output(weight, activation));
            push(MeasurementKey::readout_head(norm, weight, activation));
        }
        push(MeasurementKey::readout_features(norm, activation));
        push(MeasurementKey::sample_rows());
        push(MeasurementKey::launch_dependency());
        push(MeasurementKey::step_submission());
        for configuration in QWEN35_CONFIGURATIONS {
            for affine in [false, true] {
                push(MeasurementKey::attention_decode(
                    affine,
                    configuration.attention_shape(),
                    activation,
                ));
            }
            push(MeasurementKey::delta_step(
                configuration.key_heads,
                configuration.value_heads,
                configuration.state_width,
                configuration.convolution_width,
                activation,
            ));
            if let DeclaredFeedForward::Routed {
                experts, selected, ..
            } = configuration.feed_forward
            {
                for &router in &representations {
                    push(MeasurementKey::routed_route(
                        norm,
                        router,
                        activation,
                        configuration.hidden,
                        experts,
                        selected,
                    ));
                }
            }
        }
    }
    keys
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::assessment::DecodeDemand;
    use crate::{source_element, ComponentSelection, ModelLoadPlan};
    use magnitude_artifacts::{
        gguf::TensorDescriptor, ArtifactIdentity, ComponentManifest, PackageIdentity,
        PackageManifest,
    };
    use magnitude_model_contracts::{
        ActivationDType, AttentionGeometry, AttentionWeights, BlockGeometry, BlockWeights,
        DecoderGeometry, DenseFeedForwardWeights, ExpertGeometry, FeedForwardGeometry,
        FeedForwardWeights, InputSemantics, MixerGeometry, MixerWeights, ModelDefinition,
        RecurrentGeometry, RecurrentHeadMapping, RecurrentWeights, RotarySemantics,
        RoutedFeedForwardWeights, TextCoordinateSemantics, WeightDescriptor,
    };
    use magnitude_model_state::KvCodec;

    /// A header-shaped definition and manifest of a declared configuration.
    /// Matrices take packed encodings in rotation (so segmented entries bind
    /// mixed representations as the catalog's UD quantizations do); vectors
    /// and fixed-f32 roles are F32.
    pub(crate) fn declared_model(
        configuration: &DeclaredConfiguration,
    ) -> (ModelDefinition, PackageManifest) {
        const PACKED: [Encoding; 5] = [
            Encoding::Q4K,
            Encoding::Q5K,
            Encoding::Q6K,
            Encoding::Q8_0,
            Encoding::BF16,
        ];
        let mut tensors = Vec::new();
        let mut offset = 0u64;
        let mut rotation = 0usize;
        let mut tensor = |name: String, shape: Vec<u64>, packed: bool| {
            let encoding = if packed {
                rotation += 1;
                PACKED[rotation % PACKED.len()]
            } else {
                Encoding::F32
            };
            let nbytes = source_element(encoding)
                .unwrap()
                .canonical_byte_len(&shape)
                .unwrap();
            tensors.push(TensorDescriptor {
                name: name.clone(),
                shape: shape.clone(),
                encoding,
                offset,
                nbytes,
            });
            offset += nbytes;
            WeightDescriptor { name, shape }
        };
        let c = configuration;
        let channels = (2 * c.key_heads + c.value_heads) * c.state_width;
        let inner = c.value_heads * c.state_width;
        let mut geometry_blocks = Vec::new();
        let mut blocks = Vec::new();
        for index in 0..c.blocks {
            let name = |suffix: &str| format!("blk.{index}.{suffix}");
            let attention = (index + 1) % c.attention_interval == 0;
            let mixer_geometry = if attention {
                MixerGeometry::Attention(AttentionGeometry {
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    width: c.head_width,
                    rotary: RotarySemantics::Interleaved {
                        width: c.rotary_width,
                        base: 10_000_000.0,
                        sections: vec![c.rotary_width / 2],
                        axis_pattern: vec![0],
                    },
                })
            } else {
                MixerGeometry::Recurrent(RecurrentGeometry {
                    convolution_width: c.convolution_width,
                    key_heads: c.key_heads,
                    value_heads: c.value_heads,
                    width: c.state_width,
                    head_mapping: RecurrentHeadMapping::Tiled,
                })
            };
            let input_norm = tensor(name("attn_norm"), vec![c.hidden], false);
            let mixer = if attention {
                MixerWeights::Attention(Box::new(AttentionWeights {
                    query_gate: tensor(
                        name("attn_q"),
                        vec![2 * c.heads * c.head_width, c.hidden],
                        true,
                    ),
                    key: tensor(
                        name("attn_k"),
                        vec![c.kv_heads * c.head_width, c.hidden],
                        true,
                    ),
                    value: tensor(
                        name("attn_v"),
                        vec![c.kv_heads * c.head_width, c.hidden],
                        true,
                    ),
                    query_norm: tensor(name("attn_q_norm"), vec![c.head_width], false),
                    key_norm: tensor(name("attn_k_norm"), vec![c.head_width], false),
                    output: tensor(
                        name("attn_output"),
                        vec![c.hidden, c.heads * c.head_width],
                        true,
                    ),
                }))
            } else {
                MixerWeights::Recurrent(Box::new(RecurrentWeights {
                    query_key_value: tensor(name("attn_qkv"), vec![channels, c.hidden], true),
                    gate: tensor(name("attn_gate"), vec![inner, c.hidden], true),
                    alpha: tensor(name("ssm_alpha"), vec![c.value_heads, c.hidden], true),
                    beta: tensor(name("ssm_beta"), vec![c.value_heads, c.hidden], true),
                    convolution: tensor(
                        name("ssm_conv1d"),
                        vec![channels, c.convolution_width],
                        false,
                    ),
                    decay: tensor(name("ssm_a"), vec![c.value_heads], false),
                    time_bias: tensor(name("ssm_dt"), vec![c.value_heads], false),
                    norm: tensor(name("ssm_norm"), vec![c.state_width], false),
                    output: tensor(name("ssm_out"), vec![c.hidden, inner], true),
                }))
            };
            let feedforward_norm = tensor(name("post_attention_norm"), vec![c.hidden], false);
            let (feedforward_geometry, feedforward) = match c.feed_forward {
                DeclaredFeedForward::Dense { intermediate } => (
                    FeedForwardGeometry::Dense { intermediate },
                    FeedForwardWeights::Dense(Box::new(DenseFeedForwardWeights {
                        gate: tensor(name("ffn_gate"), vec![intermediate, c.hidden], true),
                        up: tensor(name("ffn_up"), vec![intermediate, c.hidden], true),
                        down: tensor(name("ffn_down"), vec![c.hidden, intermediate], true),
                    })),
                ),
                DeclaredFeedForward::Routed {
                    experts,
                    selected,
                    intermediate,
                    shared_intermediate,
                } => (
                    FeedForwardGeometry::Routed(ExpertGeometry {
                        count: experts,
                        selected,
                        intermediate,
                        shared_intermediate,
                        normalize_selected: true,
                    }),
                    FeedForwardWeights::Routed(Box::new(RoutedFeedForwardWeights {
                        router: tensor(name("ffn_gate_inp"), vec![experts, c.hidden], true),
                        shared_router: tensor(name("ffn_gate_inp_shexp"), vec![c.hidden], false),
                        expert_gate: tensor(
                            name("ffn_gate_exps"),
                            vec![experts, intermediate, c.hidden],
                            true,
                        ),
                        expert_up: tensor(
                            name("ffn_up_exps"),
                            vec![experts, intermediate, c.hidden],
                            true,
                        ),
                        expert_down: tensor(
                            name("ffn_down_exps"),
                            vec![experts, c.hidden, intermediate],
                            true,
                        ),
                        shared_gate: tensor(
                            name("ffn_gate_shexp"),
                            vec![shared_intermediate, c.hidden],
                            true,
                        ),
                        shared_up: tensor(
                            name("ffn_up_shexp"),
                            vec![shared_intermediate, c.hidden],
                            true,
                        ),
                        shared_down: tensor(
                            name("ffn_down_shexp"),
                            vec![c.hidden, shared_intermediate],
                            true,
                        ),
                    })),
                ),
            };
            geometry_blocks.push(BlockGeometry {
                mixer: mixer_geometry,
                feedforward: feedforward_geometry,
            });
            blocks.push(BlockWeights {
                input_norm,
                mixer,
                feedforward_norm,
                feedforward,
            });
        }
        let embedding = tensor("token_embd".into(), vec![c.vocabulary, c.hidden], true);
        let output_norm = tensor("output_norm".into(), vec![c.hidden], false);
        let output = tensor("output".into(), vec![c.vocabulary, c.hidden], true);
        let identity = PackageIdentity {
            target: ArtifactIdentity([3; 32]),
            projector: None,
        };
        let definition = ModelDefinition {
            family: magnitude_model_contracts::FamilyId("declared-qwen35".into()),
            artifact_identity: identity,
            inputs: InputSemantics {
                text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
                coordinate_axes: 1,
            },
            geometry: DecoderGeometry {
                activation_dtype: ActivationDType::BF16,
                hidden: c.hidden,
                vocabulary: c.vocabulary,
                context_limit: 262_144,
                epsilon: 1e-6,
                blocks: geometry_blocks,
            },
            embedding,
            output_norm,
            output,
            blocks,
            head: None,
            vision: None,
        };
        let manifest = PackageManifest {
            identity,
            target: ComponentManifest {
                path: format!("{}.gguf", c.model).into(),
                identity: identity.target,
                size: offset,
                tensors,
            },
            projector: None,
        };
        (definition, manifest)
    }

    #[test]
    fn every_declared_configuration_decode_demand_is_in_the_plan() {
        for backend in [
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
            BackendName::Cpu,
        ] {
            let plan = measurement_plan(backend);
            for configuration in &QWEN35_CONFIGURATIONS {
                let (definition, manifest) = declared_model(configuration);
                let load = ModelLoadPlan::derive(
                    &manifest,
                    &definition,
                    ComponentSelection {
                        head: false,
                        vision: false,
                    },
                    resident_layout(ExecutionPath::Native, backend),
                )
                .unwrap();
                for codec in [KvCodec::Dense, KvCodec::AffineK8V4] {
                    let demand = DecodeDemand::from_model(&definition, &load, codec).unwrap();
                    for term in &demand.terms {
                        assert!(
                            plan.contains(&term.key),
                            "{} on {} needs {:?} outside the plan",
                            configuration.model,
                            backend.as_str(),
                            term.key
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn plan_is_fixed_and_duplicate_free() {
        let plan = measurement_plan(BackendName::Metal);
        assert_eq!(plan, measurement_plan(BackendName::Metal));
        for (index, key) in plan.iter().enumerate() {
            assert!(!plan[..index].contains(key));
        }
        // Six representations × ten streaming classes, the feature norm,
        // sampling, launch dependency and step submission, two codecs × four attention geometries, three recurrent
        // geometries and two routing geometries × six routers.
        let representations = resident_representations(BackendName::Metal, DType::BF16).len();
        assert_eq!(representations, 6);
        assert_eq!(
            plan.len(),
            representations * 10 + 4 + 2 * 4 + 3 + 2 * representations
        );
    }
}
