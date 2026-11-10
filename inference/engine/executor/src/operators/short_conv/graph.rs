//! Short-convolution (LFM2, LFM3) block graph: the normed `u = B⊙X | C`
//! projection rows, the gated convolution over the layer's window with its
//! publication, and the output projection plus residual, or for a post-norm
//! sublayer the projection normalized into the residual
//! (`operators::short_conv`). Slots and versions follow the recurrent
//! contract, so the block binds the recurrent control ports; its bank is the
//! window alone.

use crate::operators::gated_delta::graph::{RecurrentControlPorts, bank_ports, slab_banks};
use crate::operators::{Join, RecurrentBank, split_bank};
use crate::programs::graph::facts::Numbers;
use seismic::Argument;
use crate::operators::output::{CheckedPostNormEntries, PostNorm, PostNormShape, TailEntries, post_norm};
use crate::programs::graph::{
    GraphError,
    draft::{ControlSource, GraphDraft},
};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_target_graph::{
    PlannedWeights, WeightPort, packed_weight, scaled_weight, weight,
};
use crate::{
    ShortConvBinding, ShortConvShape, StateResourcePlan, SublayerTail, native::ShortConvKernels,
};
use magnitude_family_contracts::{ShortConv, WeightKind, WeightScope};
use magnitude_kernels::{attention_output, short_conv_project, short_conv_rows};
use seismic::{Dim, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

pub(crate) struct ShortConvGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, short_conv_project::Entry>,
    pub rows: G::Binding<'a, short_conv_rows::Entry>,
    pub output: TailEntries<'a, G, attention_output::Entry>,
}

impl<'a> From<&'a ShortConvKernels> for ShortConvGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a ShortConvKernels) -> Self {
        Self {
            project: &kernels.project,
            rows: &kernels.rows,
            output: (&kernels.output).into(),
        }
    }
}

/// Entry element assignments from the program binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedShortConvEntries {
    project: [(&'static str, Element); 5],
    rows: [(&'static str, Element); 1],
    output: [(&'static str, Element); 2],
    post_norm: Option<CheckedPostNormEntries>,
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
            post_norm: match binding.tail {
                SublayerTail::Residual => None,
                SublayerTail::PostNorm { norm, .. } => Some(CheckedPostNormEntries::new(
                    binding.output,
                    binding.activation,
                    norm,
                )),
            },
        }
    }

    pub(crate) fn entries(&self) -> ShortConvGraphEntries<'_, NativeGraphMetadata> {
        ShortConvGraphEntries {
            project: &self.project,
            rows: &self.rows,
            output: match &self.post_norm {
                None => TailEntries::Residual(&self.output[..]),
                Some(post_norm) => TailEntries::PostNorm(post_norm.entries()),
            },
        }
    }
}

/// What decides a short-convolution block's graph beside its entries: no
/// model number. Its reals `R` (`RecurrentTopology`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShortConvTopology<R = Argument<f64>> {
    /// The input norm's epsilon.
    pub epsilon: R,
    /// The post-norm tail's reals; none for a residual tail.
    pub post_norm: Option<PostNorm<R>>,
    /// The element of the layer's window.
    pub window: [Element; 1],
    /// How the graph takes its launch's slot tables.
    pub controls: ControlSource,
}

impl<R> Numbers<R> for ShortConvTopology<R> {
    type Of<E> = ShortConvTopology<E>;
    fn map<E>(self, f: &mut impl FnMut(R) -> E) -> ShortConvTopology<E> {
        ShortConvTopology {
            epsilon: f(self.epsilon),
            post_norm: self.post_norm.map(|post_norm| post_norm.map(&mut *f)),
            window: self.window,
            controls: self.controls,
        }
    }
}

/// The numbers of one short-convolution block at one graph class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShortConvBlock<D = Dim> {
    pub rows: D,
    pub slots: D,
    pub shape: ShortConvShape<D>,
    /// The layer's window, also its tape.
    pub bank: RecurrentBank<D, 1>,
}

impl<D> Numbers<D> for ShortConvBlock<D> {
    type Of<E> = ShortConvBlock<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> ShortConvBlock<E> {
        ShortConvBlock {
            rows: f(self.rows),
            slots: f(self.slots),
            shape: self.shape.map(f),
            bank: self.bank.map(f),
        }
    }
}

/// The topology and numbers of the short-convolution block of `conv` over a
/// `hidden`-wide residual at a class of `rows` rows and `slots` slots, its
/// window `state`'s component `component_index`, with `post_norm`'s reals
/// where its output joins the residual through a post-norm.
pub(crate) fn split(
    conv: &ShortConv,
    hidden: u64,
    rows: u64,
    slots: u64,
    epsilon: f32,
    post_norm: Option<PostNorm<f64>>,
    state: &StateResourcePlan,
    component_index: usize,
    controls: ControlSource,
) -> Result<(ShortConvTopology<f64>, ShortConvBlock<u64>), GraphError> {
    let (window, bank) = split_bank(state, component_index)?;
    Ok((
        ShortConvTopology {
            epsilon: f64::from(epsilon),
            post_norm,
            window,
            controls,
        },
        ShortConvBlock {
            rows,
            slots,
            shape: ShortConvShape::of(hidden, conv).map_err(|error| error.to_string())?,
            bank,
        },
    ))
}

pub(crate) fn short_conv<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: ShortConvGraphEntries<'a, G>,
    topology: &ShortConvTopology,
    block: &ShortConvBlock,
    weights: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
    hidden: &WorkflowTensor,
    join: &Join,
) -> Result<(WorkflowTensor, NativePort, RecurrentControlPorts), GraphError> {
    let shape = &block.shape;
    let mut scaled =
        |graph: &mut G, kind| scaled_weight(graph, weights, scope, kind, ports, constants);
    let input_gate = scaled(graph, WeightKind::ShortConvInputGate)?;
    let output_gate = scaled(graph, WeightKind::ShortConvOutputGate)?;
    let value = scaled(graph, WeightKind::ShortConvValue)?;
    let output = packed_weight(
        graph,
        weights,
        scope,
        WeightKind::RecurrentOutput,
        ports,
        constants,
    )?;
    let mut weight = |graph: &mut G, kind| weight(graph, weights, scope, kind, ports);
    let input_norm = weight(graph, WeightKind::InputNorm)?;
    let convolution = weight(graph, WeightKind::RecurrentConvolution)?;

    let [mut window] = bank_ports(graph, &topology.window, &block.bank)?;
    let [window_extents] = &block.bank.components;
    let tape_rows = window_extents
        .first()
        .and_then(|rows| shape.tape_rows(rows))
        .ok_or("short convolution window is narrower than its taps")?;
    let slab_banks = slab_banks(&block.bank)?;

    let projection = graph
        .enqueue(
            kernels.project,
            &[
                shape.project_dimensions(&block.rows).as_slice(),
                &[
                    ("BS", input_gate.scale.extent.clone()),
                    ("CS", output_gate.scale.extent.clone()),
                    ("XS", value.scale.extent.clone()),
                ],
            ]
            .concat(),
            short_conv_project::WorkflowArgs {
                residual: hidden.into(),
                norm: (&input_norm).into(),
                b_weight: (&input_gate.weight).into(),
                c_weight: (&output_gate.weight).into(),
                x_weight: (&value.weight).into(),
                eps: topology.epsilon.f32(),
                b_scale: (&input_gate.scale.tensor).into(),
                c_scale: (&output_gate.scale.tensor).into(),
                x_scale: (&value.scale.tensor).into(),
            },
        )?
        .value;
    let dimensions =
        shape.rows_dimensions(&block.rows, &block.slots, &block.bank.banks, &tape_rows);
    let input = |graph: &mut G, name: &str| {
        graph.control_for(topology.controls, kernels.rows, name, &dimensions)
    };
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
                slab_banks,
            },
        )?
        .value;
    let mixed = match (kernels.output, &topology.post_norm) {
        (TailEntries::Residual(entry), None) => {
            // One output head of every channel.
            let gated = gated.reshape(&[block.rows.clone(), Dim::from(1u64), shape.channels.clone()]);
            let inject = join.inject(graph, &block.rows, constants)?;
            graph
                .enqueue(
                    entry,
                    &[
                        shape.output_dimensions(&block.rows).as_slice(),
                        &[
                            ("OS", output.operands.scale.extent.clone()),
                            ("OB", output.operands.bound.extent.clone()),
                        ],
                    ]
                    .concat(),
                    attention_output::WorkflowArgs {
                        hidden: hidden.into(),
                        inject: (&inject).into(),
                        gated: (&gated).into(),
                        output_weight: (&output.weight).into(),
                        output_scale: (&output.operands.scale.tensor).into(),
                        output_bound: (&output.operands.bound.tensor).into(),
                    },
                )?
                .value
        }
        (TailEntries::PostNorm(entries), Some(tail)) => {
            let norm = weight(graph, WeightKind::PostNorm)?;
            let out_rows = GraphConstant::row_identity(graph, &block.rows, Some("M"))?;
            let mixed = post_norm(
                graph,
                entries,
                hidden,
                (&gated).into(),
                &output,
                &norm,
                out_rows.port().tensor(),
                PostNormShape {
                    rows: block.rows.clone(),
                    out: block.rows.clone(),
                    inputs: shape.channels.clone(),
                    outputs: shape.hidden.clone(),
                },
                tail,
            )?;
            constants.push(out_rows);
            mixed
        }
        _ => return Err("a short convolution's output entries disagree with its post-norm tail".into()),
    };
    Ok((mixed, window, controls))
}

#[cfg(test)]
mod tests {
    use crate::programs::native_target_graph::checked_target_family_storage;
    use crate::{
        ComponentSelection, ExecutionPath, FeedForwardProgramSlot, MixerProgramSlot, ModelLoadPlan,
        PlannedMethod, ResourceCapacity, ResourceLimits, ResourcePlanner, resident_layout,
        source_element,
    };
    use magnitude_artifacts::{
        ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
        gguf::{Encoding, TensorDescriptor},
    };
    use magnitude_family_contracts::{
        ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
        EmbeddingScale, EntryForm, ExitForm, ExitNorm, ExpertSelection, FamilyId, FeedForwardUp,
        HeadNorm, HistoryRows, HistorySource, ImportTransform, InputNorm, InputSemantics,
        MediaRowAttention, ModelDefinition, Operator, OutputForm, OutputProjection,
        OutputRotary, ProjectedRows, QueryProjection, ResidualForm, RmsNorm, Rotary,
        RotaryLayout, RotaryPair, RouteNormalization, RoutedFfn, Router, RouterInput,
        RowProducer, RowRange, RowRounding, TokenDomain,
        ScoreFunction, ShortConv, Sublayer, ValueNorm, ValueSource, WeightDescriptor,
    };
    use magnitude_state::KvCodec;
    use seismic::BackendName;

    const HIDDEN: u64 = 256;

    /// An LFM2-shaped decoder: short-convolution blocks (the fused `in_proj`
    /// split into B, C, X rows) with a dense and a routed feed-forward, and
    /// an ungated NEOX attention block with a routed feed-forward (sigmoid
    /// scores ranked with a selection bias, no shared expert). With
    /// `post_norm` the short convolutions normalize their output before the
    /// residual add (LFM3's sandwich norm).
    fn lfm2(post_norm: bool) -> (ModelDefinition, PackageManifest) {
        let mut tensors = Vec::new();
        let mut offset = 0;
        let mut tensor = |name: String, shape: Vec<u64>| {
            let encoding = if shape.len() > 1 && !name.contains("conv.") {
                Encoding::BF16
            } else {
                Encoding::F32
            };
            let nbytes = source_element(encoding, *shape.last().unwrap())
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
                output: if post_norm {
                    OutputForm::PostNorm(rms(tensor(name("post_attention_norm"), vec![HIDDEN])))
                } else {
                    OutputForm::Residual
                },
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
                    query: QueryProjection::Full(tensor(name("attn_q"), vec![HIDDEN, HIDDEN])),
                    gate: AttentionGate::None,
                    query_norm: HeadNorm::Rms(rms(tensor(name("attn_q_norm"), vec![64]))),
                    history: vec![HistorySource::visible(HistoryRows::Own(
                        RowProducer::Projected(ProjectedRows {
                            key: tensor(name("attn_k"), vec![128, HIDDEN]),
                            value: ValueSource::Projected(tensor(
                                name("attn_v"),
                                vec![128, HIDDEN],
                            )),
                            key_norm: HeadNorm::Rms(rms(tensor(name("attn_k_norm"), vec![64]))),
                            value_norm: ValueNorm::None,
                            domain: TokenDomain::Token,
                            rounding: RowRounding::None,
                        }),
                    ))],
                    rotary: Rotary::Table {
                        pairs: (0..32)
                            .map(|pair| RotaryPair {
                                frequency: 1.0e6f64.powf(-(2 * pair) as f64 / 64.0),
                                amplitude: 1.0,
                            })
                            .collect(),
                        layout: RotaryLayout::HALF_SPLIT_LEADING,
                        divisors: None,
                        output: OutputRotary::None,
                    },
                    scale: 1.0 / 8.0,
                    media_rows: MediaRowAttention::Causal,
                    sinks: None,
                    output: OutputProjection::Full(tensor(
                        name("attn_output"),
                        vec![HIDDEN, HIDDEN],
                    )),
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
                        limit: None,
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
                        media_bias: None,
                        normalization: RouteNormalization::SumPlusEpsilon(1e-6),
                        scale: 1.0,
                    },
                    expert_up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        limit: None,
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
            inputs: InputSemantics { coordinate_axes: 1 },
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
    /// certify on every backend at every row class, joining the residual
    /// directly or through a post-norm.
    #[test]
    fn short_convolution_blocks_plan_and_certify_on_every_backend() {
        for post_norm in [false, true] {
            short_convolution_blocks_plan_and_certify(post_norm);
        }
    }

    fn short_convolution_blocks_plan_and_certify(post_norm: bool) {
        let (definition, manifest) = lfm2(post_norm);
        definition.validate().unwrap();
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 8,
            max_selected_rows: 64,
            max_drafting_slots: 8,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
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
                crate::PlannedMethod::Plain,
                crate::planning::fixtures::LIMITS,
            )
            .unwrap();
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let blocks = plan.target().blocks();
            assert!(matches!(
                blocks[0].mixer(),
                MixerProgramSlot::ShortConv(binding)
                    if matches!(binding.tail, crate::SublayerTail::PostNorm { .. }) == post_norm
            ));
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
                    tensor_operations: crate::TensorOperations::Absent,
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
            // The decode step streams every block's windows and weights.
            let demand = crate::assessment::demand::tests::demand_on(
                &definition,
                &load,
                magnitude_state::KvCodec::Dense,
                backend,
            )
            .unwrap();
            assert!(
                crate::assessment::demand::tests::streamed(&demand) > 0,
                "{backend:?}"
            );
        }
    }
}
