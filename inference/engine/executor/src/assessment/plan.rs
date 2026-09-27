//! The measurement plan: the fixed list of what a backend's basis holds. It
//! never depends on a model, so it is known before any header is read.
//!
//! - Every class is timed under its cost key ([`MeasurementKey::cost`]):
//!   entries binding a stored representation at the reference
//!   representation, everything else at its own bindings.
//! - Every representation the load planner produces for the backend is
//!   timed once as a `WeightFormat`, the factor that carries a
//!   weight-streaming class from the reference to it.
//! - Every exact binding of a representation-binding entry (each entry at
//!   each representation, and each conversion the importer runs) is formed
//!   untimed: the basis is the compatibility set.
//!
//! Decoders publish BF16 activations (every family declares it), and norm
//! weights are resident in the activation dtype. A model binding anything
//! else is outside the basis and incompatible.

use super::basis::{MeasurementKey, OperationClass};
use crate::{resident_element, resident_layout, source_element, ExecutionPath};
use magnitude_artifacts::gguf::Encoding;
use seismic::{BackendName, Element};

/// The activation (and norm) element the basis is measured at.
pub fn activation() -> Element {
    Element::bf16()
}

/// Every resident weight representation the native load planner produces on
/// `backend`, in encoding order.
pub fn representations(backend: BackendName) -> Vec<Element> {
    let layout = resident_layout(ExecutionPath::Native, backend);
    let activation = activation().dtype().expect("the activation is dense");
    let mut elements = Vec::new();
    for encoding in Encoding::ALL {
        if let Some(element) = resident_element(encoding, activation, layout) {
            if !elements.contains(&element) {
                elements.push(element);
            }
        }
    }
    elements
}

/// The representation weight-streaming classes are timed at: the one most
/// catalog weights stream.
pub fn reference_weight(backend: BackendName) -> Element {
    resident_element(
        Encoding::Q4K,
        activation().dtype().expect("the activation is dense"),
        resident_layout(ExecutionPath::Native, backend),
    )
    .expect("q4_k has a resident representation")
}

/// The conversions the importer runs into resident rows: each dense source
/// into the activation, and each packed source into its representation.
fn conversions(backend: BackendName) -> Vec<MeasurementKey> {
    let layout = resident_layout(ExecutionPath::Native, backend);
    let activation = activation();
    let dense = activation.dtype().expect("the activation is dense");
    let mut keys = Vec::new();
    for encoding in Encoding::ALL {
        let (Some(source), Some(resident)) = (
            source_element(encoding),
            resident_element(encoding, dense, layout),
        ) else {
            continue;
        };
        let key = if source.dtype().is_some() {
            MeasurementKey::import_rows(source, activation)
        } else {
            MeasurementKey::repack_rows(source, resident)
        };
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

/// Every exact binding of `class` at `weight`, one representation.
fn representation_bindings(class: OperationClass, weight: Element) -> Option<MeasurementKey> {
    use OperationClass as C;
    let a = activation();
    Some(match class {
        C::EmbeddingRows => MeasurementKey::embedding_rows(weight, a),
        C::AttentionProject => MeasurementKey::attention_project(a, weight, a),
        C::AttentionOutput => MeasurementKey::attention_output(weight, a),
        C::DeltaProject => MeasurementKey::delta_project(a, weight, a),
        C::DeltaOutput => MeasurementKey::delta_output(a, weight, a),
        C::ShortConvProject => MeasurementKey::short_conv_project(a, weight, a),
        C::DenseExpand => MeasurementKey::dense_expand(a, weight, a),
        C::DenseUp => MeasurementKey::dense_up(a, weight, a),
        C::DenseOutput => MeasurementKey::dense_output(weight, a),
        C::RoutedSelect => MeasurementKey::routed_select(a, weight, a),
        C::RoutedGateUp => MeasurementKey::routed_expansion(true, weight, a),
        C::RoutedUp => MeasurementKey::routed_expansion(false, weight, a),
        C::RoutedDown => MeasurementKey::routed_down(weight, a),
        C::RoutedRoute => MeasurementKey::routed_route(a, weight, a),
        C::RoutedExpand => MeasurementKey::routed_expand(weight, a),
        C::RoutedOutput => MeasurementKey::routed_output(weight, a),
        C::ProjectRows => MeasurementKey::project_rows(weight, a),
        C::PerLayerGate => MeasurementKey::per_layer_gate(weight, a),
        C::PerLayerInputs => MeasurementKey::per_layer_inputs(weight, a),
        C::ReadoutHead => MeasurementKey::readout_head(a, weight, a),
        _ => return None,
    })
}

/// The cost key every class is timed under.
fn cost_key(class: OperationClass) -> MeasurementKey {
    use OperationClass as C;
    let a = activation();
    match class {
        C::WeightFormat => unreachable!("weight formats are keyed by representation"),
        C::ImportRows | C::RepackRows | C::CopyRows | C::TableUpload | C::SampleRows => {
            MeasurementKey::new(class, &[])
        }
        C::LaunchDependency | C::StepSubmission => MeasurementKey::new(class, &[]),
        C::PostNormResidual | C::MoeTail => MeasurementKey::new(class, &[a]),
        C::ReadoutFeatures => MeasurementKey::readout_features(a, a),
        _ => MeasurementKey::new(class, &[a]),
    }
}

/// One entry of the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedKey {
    /// A class timed under its cost key, or a `WeightFormat`.
    Timed(MeasurementKey),
    /// An exact binding formed for compatibility, untimed.
    Formed(MeasurementKey),
}

impl PlannedKey {
    pub fn key(&self) -> &MeasurementKey {
        match self {
            Self::Timed(key) | Self::Formed(key) => key,
        }
    }
}

/// The plan of `backend`: every class timed in declaration order, the weight
/// formats, then every exact representation binding formed.
pub fn measurement_plan(backend: BackendName) -> Vec<PlannedKey> {
    let representations = representations(backend);
    let mut timed = Vec::new();
    let mut formed = Vec::new();
    for class in OperationClass::ALL {
        if class == OperationClass::WeightFormat {
            timed.extend(
                representations
                    .iter()
                    .map(|&weight| PlannedKey::Timed(MeasurementKey::weight_format(weight, activation()))),
            );
            continue;
        }
        timed.push(PlannedKey::Timed(cost_key(class)));
        if !class.binds_representation() {
            continue;
        }
        let bindings = match class {
            OperationClass::ImportRows | OperationClass::RepackRows => conversions(backend)
                .into_iter()
                .filter(|key| key.class == class)
                .collect(),
            _ => representations
                .iter()
                .filter_map(|&weight| representation_bindings(class, weight))
                .collect::<Vec<_>>(),
        };
        formed.extend(bindings.into_iter().map(PlannedKey::Formed));
    }
    timed.extend(formed);
    timed
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::assessment::demand::DecodeDemand;
    use crate::{ComponentSelection, ModelLoadPlan};
    use magnitude_state::KvCodec;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum DeclaredFeedForward {
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

    /// Header geometry of a Qwen3.5-family model, the fixture the plan and
    /// graph tests build header-shaped definitions from.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct DeclaredConfiguration {
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

    /// The release catalog's Qwen3.5-family (`qwen35`, `qwen35moe`) models
    /// as their headers declare them; `blocks` excludes the MTP layer.
    pub(crate) const QWEN35_CONFIGURATIONS: [DeclaredConfiguration; 5] = [
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
    use magnitude_artifacts::{
        gguf::TensorDescriptor, ArtifactIdentity, ComponentFile, ComponentManifest,
        PackageIdentity, PackageManifest,
    };
    use magnitude_family_contracts::{
        ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
        EmbeddingScale, EntryForm, ExitForm, ExitNorm, ExpertSelection, FeedForwardUp,
        GateFunction, GatedDelta, HeadNorm, HistoryDomain, HistoryReads, InputNorm,
        InputSemantics, KeyValue, MediaRowAttention, ModelDefinition, Operator, OutputForm,
        RecurrentHeadMapping, ResidualForm, RmsNorm, Rotary, RouteNormalization, RoutedFfn,
        Router, RouterInput,
        ScoreFunction, SharedExpert, SharedExpertGate, Sublayer, TextCoordinateSemantics,
        ValueNorm, ValueSource, WeightDescriptor,
    };

    /// A header-shaped definition and manifest of a declared configuration.
    /// Matrices take packed encodings in rotation (so segmented entries bind
    /// mixed representations as the catalog's UD quantizations do); vectors
    /// and fixed-f32 roles are F32.
    pub(crate) fn declared_model(
        configuration: &DeclaredConfiguration,
    ) -> (ModelDefinition, PackageManifest) {
        const PACKED: [Encoding; 6] = [
            Encoding::Q4K,
            Encoding::Q5K,
            Encoding::Q6K,
            Encoding::Q8_0,
            Encoding::Iq4Xs,
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
            WeightDescriptor::stored(name, shape)
        };
        let c = configuration;
        let channels = (2 * c.key_heads + c.value_heads) * c.state_width;
        let inner = c.value_heads * c.state_width;
        let rms = |weight| RmsNorm {
            weight,
            epsilon: 1e-6,
        };
        let silu = |gate, up| FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            gate,
            up,
        };
        let mut blocks = Vec::new();
        for index in 0..c.blocks {
            let name = |suffix: &str| format!("blk.{index}.{suffix}");
            let input_norm = tensor(name("attn_norm"), vec![c.hidden], false);
            let mixer = if (index + 1) % c.attention_interval == 0 {
                Operator::Attention(Box::new(Attention {
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    width: c.head_width,
                    query: tensor(
                        name("attn_q"),
                        vec![2 * c.heads * c.head_width, c.hidden],
                        true,
                    ),
                    gate: AttentionGate::Interleaved {
                        function: GateFunction::Sigmoid,
                    },
                    key_value: KeyValue::Owned {
                        key: tensor(
                            name("attn_k"),
                            vec![c.kv_heads * c.head_width, c.hidden],
                            true,
                        ),
                        value: ValueSource::Projected(tensor(
                            name("attn_v"),
                            vec![c.kv_heads * c.head_width, c.hidden],
                            true,
                        )),
                        key_norm: HeadNorm::Rms(rms(tensor(
                            name("attn_k_norm"),
                            vec![c.head_width],
                            false,
                        ))),
                        value_norm: ValueNorm::None,
                        domain: HistoryDomain::Token,
                    },
                    query_norm: HeadNorm::Rms(rms(tensor(
                        name("attn_q_norm"),
                        vec![c.head_width],
                        false,
                    ))),
                    rotary: Rotary::Interleaved {
                        width: c.rotary_width,
                        base: 10_000_000.0,
                        sections: vec![c.rotary_width / 2],
                        axis_pattern: vec![0],
                    },
                    scale: 1.0 / (c.head_width as f64).sqrt(),
                    reads: HistoryReads::Visible,
                    media_rows: MediaRowAttention::Causal,
                    output: tensor(
                        name("attn_output"),
                        vec![c.hidden, c.heads * c.head_width],
                        true,
                    ),
                }))
            } else {
                Operator::GatedDelta(Box::new(GatedDelta {
                    convolution_width: c.convolution_width,
                    key_heads: c.key_heads,
                    value_heads: c.value_heads,
                    width: c.state_width,
                    head_mapping: RecurrentHeadMapping::Tiled,
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
                    norm: rms(tensor(name("ssm_norm"), vec![c.state_width], false)),
                    output: tensor(name("ssm_out"), vec![c.hidden, inner], true),
                }))
            };
            let feedforward_norm = tensor(name("post_attention_norm"), vec![c.hidden], false);
            let feedforward = match c.feed_forward {
                DeclaredFeedForward::Dense { intermediate } => Operator::DenseFfn(Box::new(DenseFfn {
                    intermediate,
                    up: silu(
                        tensor(name("ffn_gate"), vec![intermediate, c.hidden], true),
                        tensor(name("ffn_up"), vec![intermediate, c.hidden], true),
                    ),
                    down: tensor(name("ffn_down"), vec![c.hidden, intermediate], true),
                })),
                DeclaredFeedForward::Routed {
                    experts,
                    selected,
                    intermediate,
                    shared_intermediate,
                } => Operator::RoutedFfn(Box::new(RoutedFfn {
                    experts,
                    selected,
                    intermediate,
                    router: Router {
                        weight: tensor(name("ffn_gate_inp"), vec![experts, c.hidden], true),
                        input: RouterInput::Operator,
                        score: ScoreFunction::Softmax,
                        selection: ExpertSelection::TopK { bias: None },
                        normalization: RouteNormalization::Sum,
                        scale: 1.0,
                    },
                    expert_up: silu(
                        tensor(
                            name("ffn_gate_exps"),
                            vec![experts, intermediate, c.hidden],
                            true,
                        ),
                        tensor(
                            name("ffn_up_exps"),
                            vec![experts, intermediate, c.hidden],
                            true,
                        ),
                    ),
                    expert_down: tensor(
                        name("ffn_down_exps"),
                        vec![experts, c.hidden, intermediate],
                        true,
                    ),
                    expert_scale: None,
                    latent: None,
                    shared: Some(SharedExpert {
                        intermediate: shared_intermediate,
                        up: silu(
                            tensor(
                                name("ffn_gate_shexp"),
                                vec![shared_intermediate, c.hidden],
                                true,
                            ),
                            tensor(
                                name("ffn_up_shexp"),
                                vec![shared_intermediate, c.hidden],
                                true,
                            ),
                        ),
                        down: tensor(
                            name("ffn_down_shexp"),
                            vec![c.hidden, shared_intermediate],
                            true,
                        ),
                        gate: SharedExpertGate::Sigmoid(tensor(
                            name("ffn_gate_inp_shexp"),
                            vec![c.hidden],
                            false,
                        )),
                    }),
                })),
            };
            blocks.push(Block {
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::Rms(rms(input_norm)),
                        op: mixer,
                        output: OutputForm::Residual,
                    },
                    Sublayer {
                        input: InputNorm::Rms(rms(feedforward_norm)),
                        op: feedforward,
                        output: OutputForm::Residual,
                    },
                ],
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
            family: magnitude_family_contracts::FamilyId("declared-qwen35".into()),
            artifact_identity: identity,
            inputs: InputSemantics {
                text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
                coordinate_axes: 1,
            },
            decoder: Decoder {
                activation_dtype: ActivationDType::BF16,
                hidden: c.hidden,
                vocabulary: c.vocabulary,
                context_limit: 262_144,
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
                    norm: ExitNorm::Rms(rms(output_norm)),
                    output,
                    softcap: None,
                },
            },
            head: None,
            vision: None,
            draft: None,
        };
        let manifest = PackageManifest {
            identity,
            target: ComponentManifest {
                files: vec![ComponentFile {
                    path: format!("{}.gguf", c.model).into(),
                    size: offset,
                }],
                identity: identity.target,
                tensors,
            },
            projector: None,
            draft: None,
        };
        (definition, manifest)
    }

    const BACKENDS: [BackendName; 4] = [
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
        BackendName::Cpu,
    ];

    #[test]
    fn the_plan_is_fixed_and_duplicate_free() {
        for backend in BACKENDS {
            let plan = measurement_plan(backend);
            assert_eq!(plan, measurement_plan(backend));
            for (index, entry) in plan.iter().enumerate() {
                assert!(
                    plan[..index].iter().all(|earlier| earlier.key() != entry.key()),
                    "{}: {} is planned twice",
                    backend.as_str(),
                    entry.key()
                );
            }
            // Every class is timed under its cost key.
            for class in OperationClass::ALL {
                assert!(plan.iter().any(|entry| matches!(entry, PlannedKey::Timed(key)
                    if key.class == class)));
            }
        }
    }

    #[test]
    fn every_declared_qwen_demand_is_covered_by_the_plan() {
        for backend in BACKENDS {
            let plan = measurement_plan(backend);
            let timed = |key: &MeasurementKey| plan.contains(&PlannedKey::Timed(key.clone()));
            for configuration in QWEN35_CONFIGURATIONS {
                let (definition, manifest) = declared_model(&configuration);
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
                    for term in DecodeDemand::from_model(&definition, &load, codec).unwrap().terms {
                        assert!(timed(&term.key.cost()), "{}: {}", backend.as_str(), term.key);
                        if term.key.class.binds_representation() {
                            assert!(
                                plan.contains(&PlannedKey::Formed(term.key.clone())),
                                "{}: {} is not formed",
                                backend.as_str(),
                                term.key
                            );
                        }
                        if let Some(weight) = term.weight() {
                            assert!(timed(&MeasurementKey::weight_format(weight, activation())));
                        }
                    }
                }
            }
        }
    }
}
