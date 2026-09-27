//! Short-convolution (LFM2) block graph: the normed `u = B⊙X | C`
//! projection rows, the gated convolution over the layer's window with its
//! publication, and the output projection plus residual
//! (`operators::short_conv`). Slots and versions follow the recurrent
//! contract, so the block binds the recurrent control ports; its bank is the
//! window alone.

use crate::operators::gated_delta::graph::RecurrentControlPorts;
use crate::programs::graph::draft::GraphDraft;
use crate::programs::native_target_graph::weight;
use crate::{
    native::ShortConvKernels, ModelLoadPlan, ShortConvBinding, ShortConvShape, StateResourcePlan,
};
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{attention_output, short_conv_project, short_conv_rows};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

pub(crate) struct ShortConvGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, short_conv_project::Entry>,
    pub rows: G::Binding<'a, short_conv_rows::Entry>,
    pub output: G::Binding<'a, attention_output::Entry>,
}

impl<'a> From<&'a ShortConvKernels> for ShortConvGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a ShortConvKernels) -> Self {
        Self {
            project: &kernels.project,
            rows: &kernels.rows,
            output: &kernels.output,
        }
    }
}

/// Entry element assignments from the program binding.
pub(crate) struct CheckedShortConvEntries {
    project: [(&'static str, Element); 5],
    rows: [(&'static str, Element); 1],
    output: [(&'static str, Element); 2],
}

impl CheckedShortConvEntries {
    pub(crate) fn new(binding: ShortConvBinding) -> Self {
        Self {
            project: [
                ("NW", binding.norm),
                ("BW", binding.input_gate),
                ("CW", binding.output_gate),
                ("XW", binding.value),
                ("A", binding.activation),
            ],
            rows: [("A", binding.activation)],
            output: [("OW", binding.output), ("A", binding.activation)],
        }
    }

    pub(crate) fn entries(&self) -> ShortConvGraphEntries<'_, NativeGraphMetadata> {
        ShortConvGraphEntries {
            project: &self.project,
            rows: &self.rows,
            output: &self.output,
        }
    }
}

/// One short-convolution block at one graph class.
pub(crate) struct ShortConvBlock {
    pub rows: u64,
    pub slots: u64,
    pub slab_banks: u32,
    pub shape: ShortConvShape,
    /// The input norm's epsilon.
    pub epsilon: f32,
    /// Index of this layer's window in the store's recurrent components.
    pub component_index: usize,
}

pub(crate) fn short_conv<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: ShortConvGraphEntries<'a, G>,
    state: &StateResourcePlan,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    weight_scale: &WorkflowTensor,
    hidden: &WorkflowTensor,
    block: ShortConvBlock,
) -> Result<(WorkflowTensor, NativePort, RecurrentControlPorts), String> {
    let shape = block.shape;
    let mut weight = |graph: &mut G, kind| weight(graph, load, scope, kind, weights);
    let input_norm = weight(graph, WeightKind::InputNorm)?;
    let input_gate = weight(graph, WeightKind::ShortConvInputGate)?;
    let output_gate = weight(graph, WeightKind::ShortConvOutputGate)?;
    let value = weight(graph, WeightKind::ShortConvValue)?;
    let convolution = weight(graph, WeightKind::RecurrentConvolution)?;
    let output_weight = weight(graph, WeightKind::RecurrentOutput)?;

    let store = state.target_state();
    let banks = u64::try_from(
        store
            .bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?,
    )
    .map_err(|_| "recurrent bank count exceeds u64")?;
    let component = store
        .recurrent_components
        .get(block.component_index)
        .ok_or("short convolution window component is absent")?;
    let extents = std::iter::once(Ok(banks))
        .chain(component.shape.iter().map(|extent| {
            u64::try_from(*extent).map_err(|_| "recurrent state extent exceeds u64".to_owned())
        }))
        .collect::<Result<Vec<_>, String>>()?;
    let tape_rows = extents
        .get(1)
        .and_then(|rows| shape.tape_rows(*rows))
        .ok_or("short convolution window is narrower than its taps")?;
    let mut window = graph.port(Element::dense(component.dtype), &extents)?;

    let projection = graph
        .enqueue(
            kernels.project,
            &[shape.project_dimensions(block.rows).as_slice(), &[("BS", 0), ("CS", 0), ("XS", 0)]].concat(),
            short_conv_project::WorkflowArgs {
                residual: hidden.into(),
                norm: (&input_norm).into(),
                b_weight: (&input_gate).into(),
                c_weight: (&output_gate).into(),
                x_weight: (&value).into(),
                eps: block.epsilon,
                b_scale: weight_scale.into(),
                c_scale: weight_scale.into(),
                x_scale: weight_scale.into(),
            },
        )?
        .value;
    let dimensions = shape.rows_dimensions(block.rows, block.slots, banks, tape_rows);
    let input = |graph: &mut G, name: &str| graph.input_for(kernels.rows, name, &dimensions);
    let controls = RecurrentControlPorts {
        segments: input(graph, "segments")?,
        stop: input(graph, "stop")?,
        previous_bank: input(graph, "previous_bank")?,
        previous_tape: input(graph, "previous_tape")?,
        following_bank: input(graph, "following_bank")?,
    };
    let gated = graph
        .enqueue(
            kernels.rows,
            &dimensions,
            short_conv_rows::WorkflowArgs {
                projection: (&projection).into(),
                convolution: (&convolution).into(),
                segments: controls.segments.tensor().into(),
                stop: controls.stop.tensor().into(),
                previous_bank: controls.previous_bank.tensor().into(),
                previous_tape: controls.previous_tape.tensor().into(),
                following_bank: controls.following_bank.tensor().into(),
                window: window.tensor_mut().into(),
                slab_banks: block.slab_banks,
            },
        )?
        .value;
    // One output head of every channel.
    let gated = gated.reshape(&[block.rows, 1, shape.channels]);
    let output = graph
        .enqueue(
            kernels.output,
            &shape.output_dimensions(block.rows),
            attention_output::WorkflowArgs {
                hidden: hidden.into(),
                gated: (&gated).into(),
                output_weight: (&output_weight).into(),
            },
        )?
        .value;
    Ok((output, window, controls))
}

#[cfg(test)]
mod tests {
    use crate::programs::native_target_graph::checked_target_family_storage;
    use crate::{
        resident_layout, source_element, ComponentSelection, ExecutionPath,
        FeedForwardProgramSlot, MixerProgramSlot, ModelLoadPlan, PlannedMethod, ResourceCapacity,
        ResourceLimits, ResourcePlanner,
    };
    use magnitude_artifacts::{
        gguf::{Encoding, TensorDescriptor},
        ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
    };
    use magnitude_family_contracts::{
        ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
        EmbeddingScale, EntryForm, ExitForm, ExitNorm, ExpertSelection, FamilyId, FeedForwardUp,
        HeadNorm, HistoryDomain, HistoryReads, ImportTransform, InputNorm, InputSemantics,
        KeyValue, MediaRowAttention, ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm,
        Rotary, RotaryPair, RouteNormalization, RoutedFfn, Router, RouterInput, RowRange,
        ScoreFunction, ShortConv, Sublayer, TextCoordinateSemantics, ValueNorm, ValueSource,
        WeightDescriptor,
    };
    use magnitude_state::KvCodec;
    use seismic::BackendName;

    const HIDDEN: u64 = 256;

    /// An LFM2-shaped decoder: short-convolution blocks (the fused `in_proj`
    /// split into B, C, X rows) with a dense and a routed feed-forward, and
    /// an ungated NEOX attention block with a routed feed-forward (sigmoid
    /// scores ranked with a selection bias, no shared expert).
    fn lfm2() -> (ModelDefinition, PackageManifest) {
        let mut tensors = Vec::new();
        let mut offset = 0;
        let mut tensor = |name: String, shape: Vec<u64>| {
            let encoding = if shape.len() > 1 && !name.contains("conv.") {
                Encoding::BF16
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
        let rms = |weight| RmsNorm {
            weight,
            epsilon: 1e-5,
        };
        let mut short_conv = |layer: usize| {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let fused = tensor(name("shortconv.in_proj"), vec![3 * HIDDEN, HIDDEN]);
            let chunk = |block: u64| WeightDescriptor {
                name: fused.name.clone(),
                shape: vec![HIDDEN, HIDDEN],
                transforms: vec![ImportTransform::Rows(RowRange {
                    start: block * HIDDEN,
                    rows: HIDDEN,
                })],
            };
            Sublayer {
                input: InputNorm::Rms(rms(tensor(name("attn_norm"), vec![HIDDEN]))),
                op: Operator::ShortConv(Box::new(ShortConv {
                    channels: HIDDEN,
                    width: 3,
                    input_gate: chunk(0),
                    output_gate: chunk(1),
                    value: chunk(2),
                    convolution: tensor(name("shortconv.conv"), vec![HIDDEN, 3]),
                    output: tensor(name("shortconv.out_proj"), vec![HIDDEN, HIDDEN]),
                })),
                output: OutputForm::Residual,
            }
        };
        let conv = [short_conv(0), short_conv(2)];
        let attention = {
            let name = |suffix: &str| format!("blk.1.{suffix}");
            Sublayer {
                input: InputNorm::Rms(rms(tensor(name("attn_norm"), vec![HIDDEN]))),
                op: Operator::Attention(Box::new(Attention {
                    heads: 4,
                    kv_heads: 2,
                    width: 64,
                    query: tensor(name("attn_q"), vec![HIDDEN, HIDDEN]),
                    gate: AttentionGate::None,
                    query_norm: HeadNorm::Rms(rms(tensor(name("attn_q_norm"), vec![64]))),
                    key_value: KeyValue::Owned {
                        key: tensor(name("attn_k"), vec![128, HIDDEN]),
                        value: ValueSource::Projected(tensor(name("attn_v"), vec![128, HIDDEN])),
                        key_norm: HeadNorm::Rms(rms(tensor(name("attn_k_norm"), vec![64]))),
                        value_norm: ValueNorm::None,
                        domain: HistoryDomain::Token,
                    },
                    rotary: Rotary::Table {
                        pairs: (0..32)
                            .map(|pair| RotaryPair {
                                frequency: 1.0e6f64.powf(-(2 * pair) as f64 / 64.0),
                                amplitude: 1.0,
                            })
                            .collect(),
                        divisors: None,
                    },
                    scale: 1.0 / 8.0,
                    reads: HistoryReads::Visible,
                    media_rows: MediaRowAttention::Causal,
                    output: tensor(name("attn_output"), vec![HIDDEN, HIDDEN]),
                })),
                output: OutputForm::Residual,
            }
        };
        let dense = {
            let name = |suffix: &str| format!("blk.0.{suffix}");
            Sublayer {
                input: InputNorm::Rms(rms(tensor(name("ffn_norm"), vec![HIDDEN]))),
                op: Operator::DenseFfn(Box::new(DenseFfn {
                    intermediate: 512,
                    up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        gate: tensor(name("ffn_gate"), vec![512, HIDDEN]),
                        up: tensor(name("ffn_up"), vec![512, HIDDEN]),
                    },
                    down: tensor(name("ffn_down"), vec![HIDDEN, 512]),
                })),
                output: OutputForm::Residual,
            }
        };
        let mut routed = |layer: usize| {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let (experts, features) = (32, 128);
            Sublayer {
                input: InputNorm::Rms(rms(tensor(name("ffn_norm"), vec![HIDDEN]))),
                op: Operator::RoutedFfn(Box::new(RoutedFfn {
                    experts,
                    selected: 4,
                    intermediate: features,
                    router: Router {
                        weight: tensor(name("ffn_gate_inp"), vec![experts, HIDDEN]),
                        input: RouterInput::Operator,
                        score: ScoreFunction::Sigmoid,
                        selection: ExpertSelection::TopK {
                            bias: Some(tensor(name("exp_probs_b"), vec![experts])),
                        },
                        normalization: RouteNormalization::SumPlusEpsilon(1e-6),
                        scale: 1.0,
                    },
                    expert_up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        gate: tensor(name("ffn_gate_exps"), vec![experts, features, HIDDEN]),
                        up: tensor(name("ffn_up_exps"), vec![experts, features, HIDDEN]),
                    },
                    expert_down: tensor(name("ffn_down_exps"), vec![experts, HIDDEN, features]),
                    expert_scale: None,
                    latent: None,
                    shared: None,
                })),
                output: OutputForm::Residual,
            }
        };
        let [first_conv, last_conv] = conv;
        let blocks = vec![
            Block {
                sublayers: vec![first_conv, dense],
            },
            Block {
                sublayers: vec![attention, routed(1)],
            },
            Block {
                sublayers: vec![last_conv, routed(2)],
            },
        ];
        let embedding = tensor("token_embd".into(), vec![512, HIDDEN]);
        let output_norm = tensor("token_embd_norm".into(), vec![HIDDEN]);
        let identity = PackageIdentity {
            target: ArtifactIdentity([6; 32]),
            projector: None,
        };
        let definition = ModelDefinition {
            family: FamilyId("lfm2-shaped".into()),
            artifact_identity: identity,
            inputs: InputSemantics {
                text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
                coordinate_axes: 1,
            },
            decoder: Decoder {
                activation_dtype: ActivationDType::BF16,
                hidden: HIDDEN,
                vocabulary: 512,
                context_limit: 4096,
                residual: ResidualForm::Single,
                entry: EntryForm {
                    embedding: embedding.clone(),
                    scale: EmbeddingScale::Unit,
                    norm: None,
                    per_layer: None,
                    hash_routing: None,
                },
                blocks,
                exit: ExitForm {
                    norm: ExitNorm::Rms(rms(output_norm)),
                    output: embedding,
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
                    path: "lfm2.gguf".into(),
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

    /// Short-convolution blocks plan with a one-component bank each, and
    /// their graphs (the checked metadata route of the production fragment)
    /// certify on every backend at every row class.
    #[test]
    fn short_convolution_blocks_plan_and_certify_on_every_backend() {
        let (definition, manifest) = lfm2();
        definition.validate().unwrap();
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 8,
            max_projected_rows: 64,
            max_images_per_request: 1,
            lookahead: false,
        };
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
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
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let blocks = plan.target().blocks();
            assert!(matches!(blocks[0].mixer(), MixerProgramSlot::ShortConv(_)));
            assert!(matches!(
                blocks[0].feed_forward(),
                Some(FeedForwardProgramSlot::Dense(_))
            ));
            assert!(matches!(blocks[1].mixer(), MixerProgramSlot::Attention(_)));
            assert!(matches!(
                blocks[2].feed_forward(),
                Some(FeedForwardProgramSlot::GeneralRouted(binding))
                    if binding.shape.shared.is_none() && binding.shape.score == 1
            ));
            let state = ResourcePlanner::state_plan(
                &definition,
                &load,
                PlannedMethod::Plain,
                KvCodec::Dense,
                limits,
                ResourceCapacity {
                    domain_bytes: 16 * 1024 * 1024 * 1024,
                },
            )
            .unwrap();
            // One F32 window per short-convolution layer.
            assert_eq!(state.target_state().recurrent_components.len(), 2);
            checked_target_family_storage(
                backend,
                &load,
                &definition.decoder,
                &state,
                plan.target(),
                limits,
            )
            .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            let classes = crate::assessment::DecodeDemand::from_model(
                &definition,
                &load,
                magnitude_state::KvCodec::Dense,
            )
            .unwrap()
            .terms
            .into_iter()
            .map(|term| term.key.class.name())
            .collect::<Vec<_>>();
            for class in [
                "short_conv_project",
                "short_conv_rows",
                "attention_output",
                "routed_select",
                "routed_gate_up",
                "routed_down",
            ] {
                assert!(classes.contains(&class), "{backend:?} lacks {class}");
            }
        }
    }
}
