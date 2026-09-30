//! State-space (Mamba-2) block graph: the normed `z | x | B | C | dt`
//! projection row, the in-place state advance over the layer's bank, the
//! gated group norm and the output projection plus residual
//! (`operators::state_space`). The bank and its per-slot tables follow the
//! recurrent contract, so the block binds the recurrent state and control
//! ports.

use crate::operators::gated_delta::graph::{
    bank_ports, chunked, RecurrentControlPorts, RecurrentStatePorts,
};
use crate::programs::graph::draft::GraphDraft;
use crate::programs::native_target_graph::{weight, WeightPort};
use crate::{
    native::StateSpaceKernels, ModelLoadPlan, StateResourcePlan, StateSpaceBinding, StateSpaceShape,
};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{
    attention_output, attention_project, state_space_chunk, state_space_gate, state_space_step,
};
use seismic::{Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

pub(crate) struct StateSpaceGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub step: G::Binding<'a, state_space_step::Entry>,
    pub chunk: G::Binding<'a, state_space_chunk::Entry>,
    pub gate: G::Binding<'a, state_space_gate::Entry>,
    pub output: G::Binding<'a, attention_output::Entry>,
}

impl<'a> From<&'a StateSpaceKernels> for StateSpaceGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a StateSpaceKernels) -> Self {
        Self {
            project: &kernels.project,
            step: &kernels.step,
            chunk: &kernels.chunk,
            gate: &kernels.gate,
            output: &kernels.output,
        }
    }
}

/// Entry element assignments from the program binding. The projection's
/// empty segments are views of the projection weight.
pub(crate) struct CheckedStateSpaceEntries {
    project: [(&'static str, Element); 6],
    activation: [(&'static str, Element); 1],
    output: [(&'static str, Element); 2],
}

impl CheckedStateSpaceEntries {
    pub(crate) fn new(binding: StateSpaceBinding) -> Self {
        Self {
            project: [
                ("NW", binding.norm),
                ("QW", binding.projection),
                ("GW", binding.projection),
                ("KW", binding.projection),
                ("VW", binding.projection),
                ("A", binding.activation),
            ],
            activation: [("A", binding.activation)],
            output: [("OW", binding.output), ("A", binding.activation)],
        }
    }

    pub(crate) fn entries(&self) -> StateSpaceGraphEntries<'_, NativeGraphMetadata> {
        StateSpaceGraphEntries {
            project: &self.project,
            step: &self.activation,
            chunk: &self.activation,
            gate: &self.activation,
            output: &self.output,
        }
    }
}

/// One state-space block at one graph class.
pub(crate) struct StateSpaceBlock {
    pub rows: u64,
    pub slots: u64,
    pub slab_banks: u32,
    pub shape: StateSpaceShape,
    /// The input norm's epsilon.
    pub epsilon: f32,
    /// The output group norm's epsilon.
    pub norm_epsilon: f32,
    /// Index of this layer's window component in the store's recurrent
    /// components; its state and tape components follow it.
    pub component_index: usize,
}

pub(crate) fn state_space<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: StateSpaceGraphEntries<'a, G>,
    state: &StateResourcePlan,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    hidden: &WorkflowTensor,
    block: StateSpaceBlock,
) -> Result<(WorkflowTensor, RecurrentStatePorts, RecurrentControlPorts), String> {
    let shape = block.shape;
    let mut weight = |graph: &mut G, kind| weight(graph, load, scope, kind, weights);
    let input_norm = weight(graph, WeightKind::InputNorm)?;
    let projection_weight = weight(graph, WeightKind::StateSpaceProjection)?;
    let convolution = weight(graph, WeightKind::RecurrentConvolution)?;
    let convolution_bias = weight(graph, WeightKind::RecurrentConvolutionBias)?;
    let time_bias = weight(graph, WeightKind::RecurrentTimeBias)?;
    let rate = weight(graph, WeightKind::RecurrentDecay)?;
    let skip = weight(graph, WeightKind::StateSpaceSkip)?;
    let state_norm = weight(graph, WeightKind::StateSpaceNorm)?;
    let output_weight = weight(graph, WeightKind::RecurrentOutput)?;

    let (mut ports, banks, tape_rows) = bank_ports(graph, state, block.component_index)?;
    // The projection has only its query segment: the others read zero rows
    // of the projection weight.
    let empty = projection_weight.slice_leading(0, 0);
    let projection = graph
        .enqueue(
            kernels.project,
            &shape.project_dimensions(block.rows),
            attention_project::WorkflowArgs {
                hidden: hidden.into(),
                input_norm: (&input_norm).into(),
                query_weight: (&projection_weight).into(),
                gate_weight: (&empty).into(),
                key_weight: (&empty).into(),
                value_weight: (&empty).into(),
                epsilon: block.epsilon,
                project_mode: 0,
            },
        )?
        .r0;
    let dimensions = shape.state_dimensions(block.rows, block.slots, banks, tape_rows);
    // The step and chunk entries share one contract, so their inputs have the
    // same geometry whichever advances this class.
    let input = |graph: &mut G, name: &str| graph.input_for(kernels.step, name, &dimensions);
    let controls = RecurrentControlPorts {
        segments: input(graph, "segments")?,
        stop: input(graph, "stop")?,
        previous_bank: input(graph, "previous_bank")?,
        previous_tape: input(graph, "previous_tape")?,
        following_bank: input(graph, "following_bank")?,
    };
    macro_rules! advance {
        ($module:ident, $entry:expr) => {
            graph
                .enqueue(
                    $entry,
                    &dimensions,
                    $module::WorkflowArgs {
                        projection: (&projection).into(),
                        convolution: (&convolution).into(),
                        convolution_bias: (&convolution_bias).into(),
                        rate: (&rate).into(),
                        time_bias: (&time_bias).into(),
                        skip: (&skip).into(),
                        segments: controls.segments.tensor().into(),
                        stop: controls.stop.tensor().into(),
                        previous_bank: controls.previous_bank.tensor().into(),
                        previous_tape: controls.previous_tape.tensor().into(),
                        following_bank: controls.following_bank.tensor().into(),
                        window: ports.window.tensor_mut().into(),
                        state: ports.delta.tensor_mut().into(),
                        tape: ports.tape.tensor_mut().into(),
                        slab_banks: block.slab_banks,
                    },
                )?
                .value
        };
    }
    let mixed = if chunked(block.rows) {
        advance!(state_space_chunk, kernels.chunk)
    } else {
        advance!(state_space_step, kernels.step)
    };
    // The stored `[groups, group width]` norm, flattened by the family, is
    // the gate's `[G, U, P]`.
    let state_norm = state_norm.reshape(&[shape.norm_groups(), shape.norm_heads, shape.head_width]);
    let gated = graph
        .enqueue(
            kernels.gate,
            &shape.gate_dimensions(block.rows),
            state_space_gate::WorkflowArgs {
                mixed: (&mixed).into(),
                projection: (&projection).into(),
                state_norm: (&state_norm).into(),
                epsilon: block.norm_epsilon,
            },
        )?
        .value;
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
    Ok((output, ports, controls))
}

#[cfg(test)]
mod tests {
    use crate::programs::native_target_graph::checked_target_family_storage;
    use crate::{
        resident_layout, source_element, ComponentSelection, ExecutionPath, FeedForwardProgramSlot,
        MixerProgramSlot, ModelLoadPlan, PlannedMethod, ResourceCapacity, ResourceLimits,
        ResourcePlanner,
    };
    use crate::{DenseScales, GeneralRoutedScales, PlanError};
    use magnitude_artifacts::{
        gguf::{Encoding, TensorDescriptor},
        ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
    };
    use magnitude_family_contracts::{
        ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder,
        EmbeddingScale, EntryForm, ExitForm, ExitNorm, ExpertSelection, FamilyId, FeedForwardUp,
        HeadNorm, HistoryDomain, HistoryReads, ImportTransform, InputNorm, InputSemantics,
        KeyValue, LatentExperts, MediaRowAttention, ModelDefinition, Operator, OutputForm,
        ResidualForm, RmsNorm, Rotary, RouteNormalization, RoutedFfn, Router, RouterInput,
        ScoreFunction, SharedExpert, SharedExpertGate, StateSpace, Sublayer, SublayerIndex,
        ValueNorm, ValueSource, WeightDescriptor, WeightKind, WeightRole, WeightScope,
    };
    use magnitude_state::KvCodec;
    use seismic::BackendName;

    const HIDDEN: u64 = 256;
    const HEADS: u64 = 8;
    const HEAD_WIDTH: u64 = 64;
    const STATE: u64 = 128;
    const GROUPS: u64 = 2;

    /// A Nemotron-H-shaped decoder: a lone state-space block, a state-space
    /// block with a feed-forward, and a NoPE attention block with one.
    fn hybrid() -> (ModelDefinition, PackageManifest) {
        hybrid_stored(|_| None)
    }

    /// [`hybrid`] with the matrices `scaled` names stored NVFP4 beside a
    /// second-level scale of the stored shape it returns
    /// (`ImportTransform::ScaleByTensor`), as Nemotron's NVFP4 files are.
    fn hybrid_stored(
        scaled: impl Fn(&str) -> Option<Vec<u64>>,
    ) -> (ModelDefinition, PackageManifest) {
        let mut tensors = Vec::new();
        let mut offset = 0;
        let mut tensor = |name: String, shape: Vec<u64>| {
            let mut stored = |name: String, shape: Vec<u64>, encoding: Encoding| {
                let nbytes = source_element(encoding)
                    .unwrap()
                    .canonical_byte_len(&shape)
                    .unwrap();
                tensors.push(TensorDescriptor {
                    name,
                    shape,
                    encoding,
                    offset,
                    nbytes,
                });
                offset += nbytes;
            };
            let mut descriptor = WeightDescriptor::stored(name.clone(), shape.clone());
            match scaled(&name) {
                Some(scale) => {
                    stored(name.clone(), shape, Encoding::Nvfp4);
                    let scale_name = format!("{name}.scale");
                    stored(scale_name.clone(), scale, Encoding::F32);
                    descriptor
                        .transforms
                        .push(ImportTransform::ScaleByTensor { tensor: scale_name });
                }
                None if shape.len() > 1 => stored(name, shape, Encoding::BF16),
                None => stored(name, shape, Encoding::F32),
            }
            descriptor
        };
        let rms = |weight| RmsNorm {
            weight,
            epsilon: 1e-5,
        };
        let inner = HEADS * HEAD_WIDTH;
        let channels = inner + 2 * GROUPS * STATE;
        let mut state_space = |layer: usize| {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            Sublayer {
                input: InputNorm::Rms(rms(tensor(name("attn_norm"), vec![HIDDEN]))),
                op: Operator::StateSpace(Box::new(StateSpace {
                    heads: HEADS,
                    head_width: HEAD_WIDTH,
                    state: STATE,
                    groups: GROUPS,
                    convolution_width: 4,
                    projection: tensor(name("ssm_in"), vec![inner + channels + HEADS, HIDDEN]),
                    convolution: tensor(name("ssm_conv1d"), vec![channels, 4]),
                    convolution_bias: tensor(name("ssm_conv1d.bias"), vec![channels]),
                    time_bias: tensor(name("ssm_dt.bias"), vec![HEADS]),
                    decay: tensor(name("ssm_a"), vec![HEADS]),
                    skip: tensor(name("ssm_d"), vec![HEADS]),
                    norm: rms(tensor(name("ssm_norm"), vec![inner])),
                    norm_group: inner / GROUPS,
                    output: tensor(name("ssm_out"), vec![HIDDEN, inner]),
                })),
                output: OutputForm::Residual,
            }
        };
        let blocks_state_space = [state_space(0), state_space(1)];
        // Nemotron-H's LatentMoE form: sigmoid scores ranked with a
        // selection bias, sum + ε normalization and a scale, up-only ReLU²
        // experts (in a latent width when `latent`), an ungated up-only
        // shared expert.
        let mut routed = |layer: usize, latent: Option<u64>| {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let (experts, features, shared) = (32, 128, 256);
            let width = latent.unwrap_or(HIDDEN);
            let relu2 = |up| FeedForwardUp::Plain {
                activation: ActivationFunction::ReluSquared,
                up,
            };
            Sublayer {
                input: InputNorm::Rms(rms(tensor(name("attn_norm"), vec![HIDDEN]))),
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
                        normalization: RouteNormalization::SumPlusEpsilon(1e-20),
                        scale: 2.5,
                    },
                    expert_up: relu2(tensor(name("ffn_up_exps"), vec![experts, features, width])),
                    expert_down: tensor(name("ffn_down_exps"), vec![experts, width, features]),
                    expert_scale: None,
                    latent: latent.map(|width| LatentExperts {
                        width,
                        down: tensor(name("ffn_latent_down"), vec![width, HIDDEN]),
                        up: tensor(name("ffn_latent_up"), vec![HIDDEN, width]),
                    }),
                    shared: Some(SharedExpert {
                        intermediate: shared,
                        up: relu2(tensor(name("ffn_up_shexp"), vec![shared, HIDDEN])),
                        down: tensor(name("ffn_down_shexp"), vec![HIDDEN, shared]),
                        gate: SharedExpertGate::None,
                    }),
                })),
                output: OutputForm::Residual,
            }
        };
        let feed_forwards = [routed(2, Some(128)), routed(4, None)];
        let attention = {
            let name = |suffix: &str| format!("blk.3.{suffix}");
            Sublayer {
                input: InputNorm::Rms(rms(tensor(name("attn_norm"), vec![HIDDEN]))),
                op: Operator::Attention(Box::new(Attention {
                    heads: 4,
                    kv_heads: 2,
                    width: 128,
                    query: tensor(name("attn_q"), vec![512, HIDDEN]),
                    gate: AttentionGate::None,
                    query_norm: HeadNorm::None,
                    key_value: KeyValue::Owned {
                        key: tensor(name("attn_k"), vec![256, HIDDEN]),
                        value: ValueSource::Projected(tensor(name("attn_v"), vec![256, HIDDEN])),
                        key_norm: HeadNorm::None,
                        value_norm: ValueNorm::None,
                        domain: HistoryDomain::Token,
                    },
                    rotary: Rotary::None,
                    scale: 1.0 / 128f64.sqrt(),
                    reads: HistoryReads::Visible,
                    media_rows: MediaRowAttention::Causal,
                    output: tensor(name("attn_output"), vec![HIDDEN, 512]),
                })),
                output: OutputForm::Residual,
            }
        };
        let [lone, paired] = blocks_state_space;
        let [first_dense, second_dense] = feed_forwards;
        let blocks = vec![
            Block {
                sublayers: vec![lone],
            },
            Block {
                sublayers: vec![paired, first_dense],
            },
            Block {
                sublayers: vec![attention, second_dense],
            },
        ];
        let embedding = tensor("token_embd".into(), vec![512, HIDDEN]);
        let output_norm = tensor("output_norm".into(), vec![HIDDEN]);
        let output = tensor("output".into(), vec![512, HIDDEN]);
        let identity = PackageIdentity {
            target: ArtifactIdentity([5; 32]),
            projector: None,
        };
        let definition = ModelDefinition {
            family: FamilyId("hybrid-state-space".into()),
            artifact_identity: identity,
            inputs: InputSemantics { coordinate_axes: 1 },
            decoder: Decoder {
                activation_dtype: ActivationDType::BF16,
                hidden: HIDDEN,
                vocabulary: 512,
                context_limit: 4096,
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
                    path: "hybrid.gguf".into(),
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

    /// Lone and paired state-space blocks plan, and their graphs (the
    /// checked metadata route of the production fragment) certify on every
    /// backend at every row class.
    #[test]
    fn hybrid_state_space_blocks_plan_and_certify_on_every_backend() {
        let (definition, manifest) = hybrid();
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
            assert!(matches!(blocks[0].mixer(), MixerProgramSlot::StateSpace(_)));
            assert!(blocks[0].feed_forward().is_none());
            assert!(matches!(
                blocks[1].feed_forward(),
                Some(FeedForwardProgramSlot::GeneralRouted(binding)) if binding.shape.latent
            ));
            assert!(matches!(
                blocks[2].feed_forward(),
                Some(FeedForwardProgramSlot::GeneralRouted(binding)) if !binding.shape.latent
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
            checked_target_family_storage(
                backend,
                &load,
                &definition.decoder,
                &state,
                plan.target(),
                limits,
            )
            .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            // Every launch of a decode step has a measured class.
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
                "state_space_step",
                "state_space_gate",
                "routed_select",
                "routed_up",
                "routed_down",
                "dense_up",
                "project_rows",
            ] {
                assert!(classes.contains(&class), "{backend:?} lacks {class}");
            }
        }
    }

    /// The matrices Nemotron's NVFP4 files scale: routed experts per expert
    /// (block 2's up weight stores one value for its whole stack), shared
    /// experts, latent projections and the vocabulary projection per tensor.
    fn nemotron_scales(name: &str) -> Option<Vec<u64>> {
        match name {
            "blk.4.ffn_up_exps" => Some(vec![1]),
            name if name.ends_with("_exps") => Some(vec![32]),
            name if name.ends_with("_shexp") || name.contains("ffn_latent") => Some(vec![1]),
            "output" => Some(vec![1]),
            _ => None,
        }
    }

    fn feed_forward_scope(block: u32) -> WeightScope {
        WeightScope::TargetSublayer(SublayerIndex { block, sublayer: 1 })
    }

    /// Second-level scales become resident beside their weights (one F32
    /// value per matrix, charged in the component's weight bytes), bind the
    /// accumulator-scale ports of every entry that reads them, and the
    /// scaled graphs certify on every backend with the same decode demand
    /// classes as unscaled weights.
    #[test]
    fn nvfp4_second_level_scales_plan_bind_and_certify_on_every_backend() {
        let (definition, manifest) = hybrid_stored(nemotron_scales);
        let (unscaled, unscaled_manifest) = hybrid();
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 8,
            max_projected_rows: 64,
            max_images_per_request: 1,
            lookahead: false,
        };
        let selection = ComponentSelection {
            head: false,
            vision: false,
        };
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let layout = resident_layout(ExecutionPath::Native, backend);
            let load = ModelLoadPlan::derive(&manifest, &definition, selection, layout).unwrap();
            let extent = |scope, kind| {
                let plan = load
                    .target()
                    .iter()
                    .find(|weight| weight.role == WeightRole { scope, kind })
                    .unwrap();
                (
                    plan.scale_extent(),
                    plan.scale.as_ref().map(|scale| scale.resident_bytes),
                )
            };
            let (latent, plain) = (feed_forward_scope(1), feed_forward_scope(2));
            assert_eq!(extent(latent, WeightKind::ExpertUp), (32, Some(128)));
            assert_eq!(extent(latent, WeightKind::ExpertDown), (32, Some(128)));
            // A stored single value is resident once per matrix.
            assert_eq!(extent(plain, WeightKind::ExpertUp), (32, Some(128)));
            assert_eq!(extent(plain, WeightKind::SharedUp), (1, Some(4)));
            assert_eq!(extent(latent, WeightKind::LatentDown), (1, Some(4)));
            assert_eq!(
                extent(WeightScope::Target, WeightKind::Output),
                (1, Some(4))
            );
            assert_eq!(extent(plain, WeightKind::Router), (0, None));
            let scale_bytes = load
                .target()
                .iter()
                .filter_map(|weight| weight.scale.as_ref())
                .map(|scale| scale.resident_bytes)
                .sum::<u64>();
            let packed_bytes = load
                .target()
                .iter()
                .map(|weight| weight.storage_bytes().unwrap())
                .sum::<u64>();
            assert_eq!(
                packed_bytes,
                load.target()
                    .iter()
                    .map(|weight| weight.resident_bytes)
                    .sum::<u64>()
                    + scale_bytes
            );

            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let blocks = plan.target().blocks();
            let Some(FeedForwardProgramSlot::GeneralRouted(latent_binding)) =
                blocks[1].feed_forward()
            else {
                panic!("block 1 is general routed");
            };
            assert_eq!(
                latent_binding.scales,
                GeneralRoutedScales {
                    expert_up: 32,
                    expert_down: 32,
                    shared: DenseScales {
                        gate: 0,
                        up: 1,
                        down: 1
                    },
                    latent: (1, 1),
                }
            );
            assert_eq!(plan.target().readout().weight_scale, 1);

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
            checked_target_family_storage(
                backend,
                &load,
                &definition.decoder,
                &state,
                plan.target(),
                limits,
            )
            .unwrap_or_else(|error| panic!("{backend:?}: {error}"));

            // Scales change no demand class: the measurement keys are the
            // representation's.
            let classes = |definition: &ModelDefinition, load: &ModelLoadPlan| {
                crate::assessment::DecodeDemand::from_model(definition, load, KvCodec::Dense)
                    .unwrap()
                    .terms
                    .into_iter()
                    .map(|term| term.key.class.name())
                    .collect::<std::collections::BTreeSet<_>>()
            };
            let unscaled_load =
                ModelLoadPlan::derive(&unscaled_manifest, &unscaled, selection, layout).unwrap();
            assert_eq!(
                classes(&definition, &load),
                classes(&unscaled, &unscaled_load)
            );
        }
    }

    /// A scaled weight whose entry has no accumulator-scale port (the
    /// attention projections, the router) is refused at plan time.
    #[test]
    fn a_scaled_weight_without_a_scale_port_is_refused_at_plan_time() {
        for (name, role) in [
            (
                "blk.3.attn_q",
                WeightRole {
                    scope: WeightScope::TargetSublayer(SublayerIndex {
                        block: 2,
                        sublayer: 0,
                    }),
                    kind: WeightKind::Query,
                },
            ),
            (
                "blk.4.ffn_gate_inp",
                WeightRole {
                    scope: feed_forward_scope(2),
                    kind: WeightKind::Router,
                },
            ),
        ] {
            let (definition, manifest) = hybrid_stored(|stored| (stored == name).then(|| vec![1]));
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                resident_layout(ExecutionPath::Native, BackendName::Metal),
            )
            .unwrap();
            assert_eq!(
                load.program_plan(&definition, KvCodec::Dense).unwrap_err(),
                PlanError::UnportedScale(role)
            );
        }
    }

    /// A scale that is not one F32 value, or one per matrix, is refused
    /// when the load is planned.
    #[test]
    fn malformed_second_level_scales_are_refused_when_the_load_is_planned() {
        let derive = |scale: fn(&str) -> Option<Vec<u64>>| {
            let (definition, manifest) = hybrid_stored(scale);
            ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                resident_layout(ExecutionPath::Native, BackendName::Cpu),
            )
        };
        // Neither one value nor one per expert.
        let error = derive(|name| (name == "blk.4.ffn_up_exps").then(|| vec![3])).unwrap_err();
        assert!(error.contains("not F32 [1] or [32]"), "{error}");
        // A matrix has one scale.
        let error = derive(|name| (name == "blk.4.ffn_up_shexp").then(|| vec![2])).unwrap_err();
        assert!(error.contains("not F32 [1] or [1]"), "{error}");
        // A named scale the component lacks.
        let (definition, mut manifest) = hybrid_stored(|name| (name == "output").then(|| vec![1]));
        manifest
            .target
            .tensors
            .retain(|tensor| tensor.name != "output.scale");
        let error = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            resident_layout(ExecutionPath::Native, BackendName::Cpu),
        )
        .unwrap_err();
        assert!(error.contains("absent from its component"), "{error}");
    }
}
