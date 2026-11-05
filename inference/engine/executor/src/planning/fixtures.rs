//! The planning fixture's model: the smallest geometry every native kernel
//! admits, and its package manifest. Built for the crate's tests and, under
//! the `fixtures` feature, for the domain fixtures other crates test on.

use magnitude_artifacts::{
    ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
    gguf::{Encoding, TensorDescriptor},
};
use magnitude_family_contracts::{
    ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
    EmbeddingScale, EntryForm, ExitForm, ExitNorm, FeedForwardUp, GateFunction, HeadNorm,
    HistoryDomain, HistoryReads, InputNorm, InputSemantics, KeyValue, MediaRowAttention,
    ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm, Rotary, Sublayer, ValueNorm,
    ValueSource, WeightDescriptor,
};

/// The limits the fixture's loads plan with.
pub(crate) const LIMITS: crate::ResourceLimits = crate::ResourceLimits {
    max_launch_rows: 512,
    max_launch_slots: 32,
    max_selected_rows: 32,
    max_drafting_slots: 32,
    exported_logits_rows: 0,
    max_images_per_request: 1,
    max_image_cells: 0,
    lookahead: true,
    chain_capacity: std::num::NonZeroUsize::MIN,
    conversations: std::num::NonZeroUsize::MIN,
};

pub(crate) fn descriptor(name: &str, shape: &[u64]) -> WeightDescriptor {
    WeightDescriptor::stored(name, shape)
}

/// The smallest geometry every native kernel admits: hidden and
/// feed-forward widths a multiple of 64 (CUDA K1 k-blocks), an attention
/// head width a multiple of 32 (attention lanes), and two query heads
/// sharing one key-value head (Vulkan prefill needs a query group of at
/// least two).
pub(crate) fn fixture_definition() -> ModelDefinition {
    const HIDDEN: u64 = 128;
    const WIDTH: u64 = 64;
    const FEATURES: u64 = 128;
    const VOCABULARY: u64 = 256;
    let rms = |name: &str, width| RmsNorm {
        weight: descriptor(name, &[width]),
        epsilon: 1e-6,
    };
    let attention = Attention {
        heads: 2,
        kv_heads: 1,
        width: WIDTH,
        query: descriptor("qg", &[2 * 2 * WIDTH, HIDDEN]),
        gate: AttentionGate::Interleaved {
            function: GateFunction::Sigmoid,
        },
        query_norm: HeadNorm::Rms(rms("qn", WIDTH)),
        key_value: KeyValue::Owned {
            key: descriptor("k", &[WIDTH, HIDDEN]),
            value: ValueSource::Projected(descriptor("v", &[WIDTH, HIDDEN])),
            key_norm: HeadNorm::Rms(rms("kn", WIDTH)),
            value_norm: ValueNorm::None,
            domain: HistoryDomain::Token,
        },
        rotary: Rotary::Interleaved {
            width: WIDTH / 2,
            base: 10_000.0,
            sections: vec![WIDTH / 4],
            axis_pattern: vec![0],
        },
        scale: 1.0 / (WIDTH as f64).sqrt(),
        reads: HistoryReads::Visible,
        media_rows: MediaRowAttention::Causal,
        sinks: None,
        output: descriptor("o", &[HIDDEN, 2 * WIDTH]),
    };
    let dense = DenseFfn {
        intermediate: FEATURES,
        up: FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            gate: descriptor("gate", &[FEATURES, HIDDEN]),
            up: descriptor("up", &[FEATURES, HIDDEN]),
        },
        down: descriptor("down", &[HIDDEN, FEATURES]),
    };
    ModelDefinition {
        family: magnitude_family_contracts::FamilyId("resource-fixture".into()),
        artifact_identity: PackageIdentity {
            target: ArtifactIdentity([7; 32]),
            projector: None,
        },
        inputs: InputSemantics { coordinate_axes: 1 },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden: HIDDEN,
            vocabulary: VOCABULARY,
            context_limit: 128,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding: descriptor("embedding", &[VOCABULARY, HIDDEN]),
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
                hash_routing: None,
            },
            blocks: vec![Block {
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::Rms(rms("input_norm", HIDDEN)),
                        op: Operator::Attention(Box::new(attention)),
                        output: OutputForm::Residual,
                    },
                    Sublayer {
                        input: InputNorm::Rms(rms("ffn_norm", HIDDEN)),
                        op: Operator::DenseFfn(Box::new(dense)),
                        output: OutputForm::Residual,
                    },
                ],
            }],
            exit: ExitForm {
                norm: ExitNorm::Rms(rms("output_norm", HIDDEN)),
                output: descriptor("output", &[VOCABULARY, HIDDEN]),
                softcap: None,
            },
        },
        head: None,
        vision: None,
        draft: None,
    }
}

pub(crate) fn fixture_manifest(definition: &ModelDefinition) -> PackageManifest {
    let decoder = &definition.decoder;
    let head = definition.head.as_ref().map_or_else(Vec::new, |head| {
        crate::operators::head_weights(head).expect("fixture head block indices fit u32")
    });
    let descriptors = std::iter::once(&decoder.entry.embedding)
        .chain(
            crate::operators::decoder_weights(decoder)
                .into_iter()
                .chain(head)
                .map(|(_, descriptor)| descriptor),
        )
        .chain(
            crate::operators::exit_weights(decoder)
                .into_iter()
                .map(|(_, descriptor)| descriptor),
        )
        .chain([&decoder.exit.output]);
    // A tensor several roles read (tied weights) is stored once.
    let mut named = std::collections::HashSet::new();
    let mut offset = 0;
    let tensors = descriptors
        .filter(|descriptor| named.insert(descriptor.name.clone()))
        .map(|descriptor| {
            let nbytes = descriptor.shape.iter().product::<u64>() * 2;
            let tensor = TensorDescriptor {
                name: descriptor.name.clone(),
                shape: descriptor.shape.clone(),
                encoding: Encoding::F16,
                offset,
                nbytes,
            };
            offset += nbytes;
            tensor
        })
        .collect();
    PackageManifest {
        identity: definition.artifact_identity,
        target: ComponentManifest {
            files: vec![ComponentFile {
                path: "planning-fixture.gguf".into(),
                size: offset,
            }],
            identity: definition.artifact_identity.target,
            tensors,
        },
        projector: None,
        draft: None,
    }
}

/// The planning fixture over four hyper-connection streams: its lone
/// attention mixer reads through a mix site and scatters into the
/// streams, and its exit is a hyper-connection head, with no output
/// norm.
pub fn hyper_connection_fixture_definition() -> ModelDefinition {
    use magnitude_family_contracts::{HyperConnectionMixing, HyperConnections};
    let mut definition = fixture_definition();
    let decoder = &mut definition.decoder;
    decoder.residual = ResidualForm::HyperConnections(HyperConnections {
        streams: STREAMS,
        mixing: HyperConnectionMixing::LowRankGated { rank: RANK },
    });
    // A dense feed-forward has no streams form: the block is its mixer.
    let sublayers = &mut decoder.blocks[0].sublayers;
    sublayers.truncate(1);
    sublayers[0].input = InputNorm::HyperConnectionMix(site("attn_hc", true));
    sublayers[0].output = OutputForm::Scatter;
    decoder.exit.norm = ExitNorm::HyperConnectionHead(site("output_hc", false));
    definition
}

/// The streams and low rank of the hyper-connection fixtures.
const STREAMS: u64 = 4;
const RANK: u64 = 64;

/// A mix site of the fixture geometry, with inject weights when
/// `injected`.
fn site(prefix: &str, injected: bool) -> magnitude_family_contracts::HyperConnectionWeights {
    let wide = STREAMS * fixture_definition().decoder.hidden;
    magnitude_family_contracts::HyperConnectionWeights {
        norm: RmsNorm {
            weight: descriptor(&format!("{prefix}_norm"), &[wide]),
            epsilon: 1e-6,
        },
        down: descriptor(&format!("{prefix}_down"), &[RANK, wide]),
        up: descriptor(&format!("{prefix}_up"), &[wide, RANK]),
        inject: injected.then(|| descriptor(&format!("{prefix}_inject"), &[STREAMS, wide])),
    }
}

/// A qwen4exp-shaped decoder over the hyper-connection fixture's
/// streams: a gated delta block whose routed feed-forward is followed by
/// an n-gram input, then an attention block with a routed feed-forward,
/// every mixer and feed-forward reading through its mix site.
pub fn qwen4exp_fixture_definition() -> ModelDefinition {
    qwen4exp_fixture_definition_of(1, 32)
}

/// [`qwen4exp_fixture_definition`] with `attention_blocks` attention
/// blocks after its gated delta block, each with its own experts, and
/// `experts` experts in every routed layer (a multiple of 32: Metal routes
/// over whole simdgroups): the first `attention_blocks - HINT_LEADS`
/// attention blocks of one shape, so they share one sealed graph (each
/// later one differs: its route predicts fewer layers,
/// `ExpertBankPlan::hints`).
pub fn qwen4exp_fixture_definition_of(attention_blocks: u64, experts: u64) -> ModelDefinition {
    use magnitude_family_contracts::{
        ExpertSelection, GatedDelta, NGramHash, NGramInput, RecurrentHeadMapping,
        RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction, SharedExpert,
        SharedExpertGate,
    };
    const FEATURES: u64 = 64;
    const STATE: u64 = 128;
    const NGRAM_WIDTH: u64 = 32;
    const NGRAM_ROWS: u64 = 64;
    let mut definition = hyper_connection_fixture_definition();
    let decoder = &mut definition.decoder;
    let hidden = decoder.hidden;
    let wide = STREAMS * hidden;
    let rms = |name: &str, width| RmsNorm {
        weight: descriptor(name, &[width]),
        epsilon: 1e-6,
    };
    let silu = |gate: WeightDescriptor, up: WeightDescriptor| FeedForwardUp::Gated {
        activation: ActivationFunction::Silu,
        gate,
        up,
    };
    let mixing = |prefix: &str, op: Operator| Sublayer {
        input: InputNorm::HyperConnectionMix(site(prefix, true)),
        op,
        output: OutputForm::Scatter,
    };
    let routed = |block: u64| {
        let name = |part: &str| format!("blk.{block}.{part}");
        Operator::RoutedFfn(Box::new(RoutedFfn {
            experts,
            selected: 2,
            intermediate: FEATURES,
            router: Router {
                weight: descriptor(&name("ffn_gate_inp"), &[experts, hidden]),
                input: RouterInput::Operator,
                score: ScoreFunction::Softmax,
                selection: ExpertSelection::TopK { bias: None },
                normalization: RouteNormalization::Sum,
                scale: 1.0,
            },
            expert_up: silu(
                descriptor(&name("ffn_gate_exps"), &[experts, FEATURES, hidden]),
                descriptor(&name("ffn_up_exps"), &[experts, FEATURES, hidden]),
            ),
            expert_down: descriptor(&name("ffn_down_exps"), &[experts, hidden, FEATURES]),
            expert_scale: None,
            latent: None,
            shared: Some(SharedExpert {
                intermediate: FEATURES,
                up: silu(
                    descriptor(&name("ffn_gate_shexp"), &[FEATURES, hidden]),
                    descriptor(&name("ffn_up_shexp"), &[FEATURES, hidden]),
                ),
                down: descriptor(&name("ffn_down_shexp"), &[hidden, FEATURES]),
                gate: SharedExpertGate::Sigmoid(descriptor(
                    &name("ffn_gate_inp_shexp"),
                    &[hidden],
                )),
            }),
        }))
    };
    // CUDA's gated delta step admits value heads in fours.
    let (key_heads, value_heads) = (1, 4);
    let channels = (2 * key_heads + value_heads) * STATE;
    let gated_delta = Operator::GatedDelta(Box::new(GatedDelta {
        convolution_width: 4,
        key_heads,
        value_heads,
        width: STATE,
        head_mapping: RecurrentHeadMapping::Tiled,
        query_key_value: descriptor("blk.0.attn_qkv", &[channels, hidden]),
        gate: descriptor("blk.0.attn_gate", &[value_heads * STATE, hidden]),
        alpha: descriptor("blk.0.ssm_alpha", &[value_heads, hidden]),
        beta: descriptor("blk.0.ssm_beta", &[value_heads, hidden]),
        convolution: descriptor("blk.0.ssm_conv1d", &[channels, 4]),
        decay: descriptor("blk.0.ssm_a", &[value_heads]),
        time_bias: descriptor("blk.0.ssm_dt", &[value_heads]),
        norm: rms("blk.0.ssm_norm", STATE),
        output_gate: GateFunction::Sigmoid,
        output: descriptor("blk.0.ssm_out", &[hidden, value_heads * STATE]),
    }));
    let (context, heads_per_order) = (2, 2);
    let heads = context * heads_per_order;
    let ngram = Operator::NGramInput(Box::new(NGramInput {
        hash: NGramHash {
            context,
            heads_per_order,
            multipliers: vec![1, 7, 13],
            sizes: vec![NGRAM_ROWS / heads; heads as usize],
            offsets: (0..heads).map(|head| head * NGRAM_ROWS / heads).collect(),
            reset: 1,
        },
        width: NGRAM_WIDTH,
        table: descriptor("per_layer_token_embd.weight", &[NGRAM_ROWS, NGRAM_WIDTH]),
        key: descriptor("blk.0.ngram_k", &[wide, heads * NGRAM_WIDTH]),
        value: descriptor("blk.0.ngram_v", &[hidden, heads * NGRAM_WIDTH]),
        key_norm: rms("blk.0.ngram_k_norm", wide),
        query_norm: rms("blk.0.ngram_q_norm", wide),
        convolution_norm: rms("blk.0.ngram_conv_norm", wide),
        convolution: descriptor("blk.0.ngram_conv", &[wide, 4]),
        kernel: 4,
        dilation: 1,
    }));
    let attention = decoder.blocks[0].sublayers[0].op.clone();
    decoder.blocks = vec![
        Block {
            sublayers: vec![
                mixing("blk.0.attn_hc", gated_delta),
                mixing("blk.0.ffn_hc", routed(0)),
                Sublayer {
                    input: InputNorm::None,
                    op: ngram,
                    output: OutputForm::Residual,
                },
            ],
        },
    ];
    // Each attention block's mixer and mix sites are block 1's (tied), its
    // experts its own.
    decoder.blocks.extend((1..=attention_blocks).map(|block| Block {
        sublayers: vec![
            mixing("blk.1.attn_hc", attention.clone()),
            mixing("blk.1.ffn_hc", routed(block)),
        ],
    }));
    definition
}

/// The fixture manifest of `definition`, its n-gram tables stored as
/// IQ4_NL blocks (the stored form a streamed table reads).
pub fn stored_table_manifest(definition: &ModelDefinition) -> PackageManifest {
    let tables = definition
        .decoder
        .sublayers()
        .filter_map(|(_, sublayer)| match &sublayer.op {
            Operator::NGramInput(input) => Some(input.table.name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    relaid_manifest(fixture_manifest(definition), &tables, Encoding::Iq4Nl, 18)
}

/// [`stored_table_manifest`] with every routed layer's expert stacks
/// stored as Q8_0 blocks: a packed record form an expert bank streams.
pub fn streamed_experts_manifest(definition: &ModelDefinition) -> PackageManifest {
    let stacks = expert_stacks(definition).into_iter().flatten().collect::<Vec<_>>();
    relaid_manifest(stored_table_manifest(definition), &stacks, Encoding::Q8_0, 34)
}

/// Each routed layer's expert stacks' names (gate, up, down), in decoder
/// order.
pub fn expert_stacks(definition: &ModelDefinition) -> Vec<[String; 3]> {
    definition
        .decoder
        .sublayers()
        .filter_map(|(_, sublayer)| match &sublayer.op {
            Operator::RoutedFfn(routed) => Some(routed),
            _ => None,
        })
        .map(|routed| {
            let FeedForwardUp::Gated { gate, up, .. } = &routed.expert_up else {
                unreachable!("the fixture's experts are gated");
            };
            [&gate.name, &up.name, &routed.expert_down.name].map(Clone::clone)
        })
        .collect()
}

/// `manifest` with the tensors `names` stored as `encoding`, whose
/// 32-element blocks take `block_bytes` each, every tensor re-laid.
pub fn relaid_manifest(
    mut manifest: PackageManifest,
    names: &[String],
    encoding: Encoding,
    block_bytes: u64,
) -> PackageManifest {
    let mut offset = 0;
    for tensor in &mut manifest.target.tensors {
        if names.contains(&tensor.name) {
            tensor.encoding = encoding;
            tensor.nbytes = tensor.shape.iter().product::<u64>() / 32 * block_bytes;
        }
        tensor.offset = offset;
        offset += tensor.nbytes;
    }
    manifest.target.files[0].size = offset;
    manifest
}
