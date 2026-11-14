//! The planning fixture's model: the smallest geometry every native kernel
//! admits, and its package manifest. Built for the crate's tests and, under
//! the `fixtures` feature, for the domain fixtures other crates test on.

use magnitude_artifacts::{
    ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
    gguf::{Encoding, TensorDescriptor},
};
use magnitude_family_contracts::{
    ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, DenseFfn,
    EmbeddingScale, EntryForm, ExitForm, ExitNorm, FeedForwardUp, GateFunction, GateProjection,
    RecurrentDecay, RecurrentGate, HeadNorm,
    HistoryRows, HistorySource, InputNorm, InputSemantics, MediaRowAttention, ModelDefinition,
    Operator, OutputForm, OutputProjection, ProjectedRows, QueryKeys, QueryProjection, ResidualForm,
    RmsNorm, Rotary, RowProducer, RowRounding, Sublayer, TokenDomain, ValueNorm, ValueSource,
    WeightDescriptor,
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
    fixture_definition_of(HIDDEN)
}

/// The planning fixture's hidden width.
pub(crate) const HIDDEN: u64 = 128;

/// [`fixture_definition`] at hidden width `hidden`.
pub(crate) fn fixture_definition_of(hidden: u64) -> ModelDefinition {
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
        query: QueryProjection::Full(descriptor("qg", &[2 * 2 * WIDTH, hidden])),
        query_keys: QueryKeys::Direct,
        gate: AttentionGate::Interleaved {
            function: GateFunction::Sigmoid,
        },
        query_norm: HeadNorm::Rms(rms("qn", WIDTH)),
        history: vec![HistorySource::visible(HistoryRows::Own(RowProducer::Projected(
            ProjectedRows {
                key: descriptor("k", &[WIDTH, hidden]),
                value: ValueSource::Projected(descriptor("v", &[WIDTH, hidden])),
                key_norm: HeadNorm::Rms(rms("kn", WIDTH)),
                value_norm: ValueNorm::None,
                domain: TokenDomain::Token,
                rounding: RowRounding::None,
            },
        )))],
        rotary: Rotary::Interleaved {
            width: WIDTH / 2,
            base: 10_000.0,
            sections: vec![WIDTH / 4],
            axis_pattern: vec![0],
        },
        scale: 1.0 / (WIDTH as f64).sqrt(),
        media_rows: MediaRowAttention::Causal,
        sinks: None,
        output: OutputProjection::Full(descriptor("o", &[hidden, 2 * WIDTH])),
    };
    let dense = DenseFfn {
        intermediate: FEATURES,
        up: FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            limit: None,
            gate: descriptor("gate", &[FEATURES, hidden]),
            up: descriptor("up", &[FEATURES, hidden]),
        },
        down: descriptor("down", &[hidden, FEATURES]),
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
            hidden,
            vocabulary: VOCABULARY,
            context_limit: 128,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding: descriptor("embedding", &[VOCABULARY, hidden]),
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
            },
            blocks: vec![Block {
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::Rms(rms("input_norm", hidden)),
                        op: Operator::Attention(Box::new(attention)),
                        output: OutputForm::Residual,
                    },
                    Sublayer {
                        input: InputNorm::Rms(rms("ffn_norm", hidden)),
                        op: Operator::DenseFfn(Box::new(dense)),
                        output: OutputForm::Residual,
                    },
                ],
            }],
            exit: ExitForm {
                norm: ExitNorm::Rms(rms("output_norm", hidden)),
                output: descriptor("output", &[VOCABULARY, hidden]),
                softcap: None,
            },
        },
        head: None,
        vision: None,
        draft: None,
    }
}

/// `definition` with an MTP head of its own history, stored in the target
/// component: its first block (attention, dense) under head names, between
/// the head's own norms and combine, reading out through the target's
/// vocabulary projection and chaining its output.
#[cfg(test)]
pub(crate) fn with_mtp_head(mut definition: ModelDefinition) -> ModelDefinition {
    use magnitude_family_contracts::{
        Head, HeadBlock, HeadChain, HeadComponent, HeadHistory, HeadInput, HeadReadout,
        VocabularyProjection,
    };
    let hidden = definition.decoder.hidden;
    let named = |name: &str, shape: &[u64]| descriptor(&format!("head_{name}"), shape);
    let rms = |name: &str| RmsNorm {
        weight: named(name, &[hidden]),
        epsilon: 1e-6,
    };
    let mut head_layer = definition.decoder.blocks[0].clone();
    let rename = |weight: &mut WeightDescriptor| weight.name = format!("head_{}", weight.name);
    for sublayer in &mut head_layer.sublayers {
        if let InputNorm::Rms(norm) = &mut sublayer.input {
            rename(&mut norm.weight);
        }
        match &mut sublayer.op {
            Operator::Attention(attention) => {
                let (QueryProjection::Full(query), OutputProjection::Full(output)) =
                    (&mut attention.query, &mut attention.output)
                else {
                    unreachable!("the fixture projects its queries and output whole")
                };
                rename(query);
                rename(output);
                if let HeadNorm::Rms(norm) = &mut attention.query_norm {
                    rename(&mut norm.weight);
                }
                if let HistoryRows::Own(RowProducer::Projected(ProjectedRows {
                    key,
                    value: ValueSource::Projected(value),
                    key_norm: HeadNorm::Rms(key_norm),
                    ..
                })) = &mut attention.history[0].rows
                {
                    rename(key);
                    rename(value);
                    rename(&mut key_norm.weight);
                }
            }
            Operator::DenseFfn(dense) => {
                if let FeedForwardUp::Gated { gate, up, .. } = &mut dense.up {
                    rename(gate);
                    rename(up);
                }
                rename(&mut dense.down);
            }
            _ => unreachable!("the fixture's first block is attention then dense"),
        }
    }
    definition.head = Some(Head {
        blocks: vec![HeadBlock {
            residual: ResidualForm::Single,
            input: HeadInput::Combine {
                embedding_norm: rms("embedding_norm"),
                hidden_norm: rms("hidden_norm"),
                combine: named("combine", &[hidden, 2 * hidden]),
            },
            blocks: vec![head_layer],
            output_norm: ExitNorm::Rms(rms("output_norm")),
            readout: HeadReadout::full(VocabularyProjection::Target),
            chain: HeadChain::Output,
        }],
        history: HeadHistory::Own,
        component: HeadComponent::Target,
    });
    definition
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

/// The planning fixture with a separate DFlash draft, which owns its output
/// projection or reads the target's (`owns_output`). The projection the
/// draft reads is Q8_0, the form a progressive readout is offered for.
#[cfg(test)]
pub(crate) fn separate_draft_fixture(owns_output: bool) -> (ModelDefinition, PackageManifest) {
    use magnitude_family_contracts::{
        BlockAttention, BlockLayout, DraftDefinition, DraftEmbedding, DraftMethod, TapPoint,
        TokenId,
    };
    let mut definition = fixture_definition();
    let mut manifest = fixture_manifest(&definition);
    let (hidden, vocabulary) = (definition.decoder.hidden, definition.decoder.vocabulary);
    let norm = |name: &str| RmsNorm {
        weight: WeightDescriptor::stored(name, &[hidden]),
        epsilon: 1e-6,
    };
    definition.draft = Some(DraftDefinition {
        method: DraftMethod::DFlash,
        convolutions: None,
        taps: vec![TapPoint::Exit],
        fusion: WeightDescriptor::stored("draft_fusion", &[hidden, hidden]),
        fusion_norm: norm("draft_fusion_norm"),
        embedding: DraftEmbedding::Target,
        output: owns_output
            .then(|| WeightDescriptor::stored("draft_output", &[vocabulary, hidden])),
        embedding_scale: 1.0,
        blocks: definition.decoder.blocks.clone(),
        block_attention: vec![BlockAttention::Bidirectional],
        output_norm: norm("draft_output_norm"),
        logit_softcap: Some(0.1),
        block_size: 2,
        mask_token: TokenId(0),
        layout: BlockLayout::MaskSlots,
    });
    definition.validate().unwrap();
    let draft = definition.draft.as_ref().unwrap();
    let mut offset = 0;
    let tensors = crate::operators::draft::fusion_weights(draft)
        .into_iter()
        .chain(crate::operators::draft::draft_weights(draft).unwrap())
        .map(|(_, descriptor)| {
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
    manifest.draft = Some(ComponentManifest {
        files: vec![ComponentFile {
            path: "draft-fixture.gguf".into(),
            size: offset,
        }],
        identity: ArtifactIdentity([9; 32]),
        tensors,
    });
    let (component, descriptor) = match &draft.output {
        Some(output) => (manifest.draft.as_mut().unwrap(), output),
        None => (&mut manifest.target, &definition.decoder.exit.output),
    };
    let output = component
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == descriptor.name)
        .unwrap();
    output.encoding = Encoding::Q8_0;
    output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
    (definition, manifest)
}

/// The planning fixture over four hyper-connection streams: its lone
/// attention mixer reads through a mix site and scatters into the
/// streams, and its exit is a hyper-connection head, with no output
/// norm. Its hidden width is the mix sites' (`mix_site_hidden`), so
/// every backend's kernels admit it.
pub fn hyper_connection_fixture_definition() -> ModelDefinition {
    use magnitude_family_contracts::{HyperConnectionMixing, HyperConnections};
    let hidden = mix_site_hidden();
    let mut definition = fixture_definition_of(hidden);
    let decoder = &mut definition.decoder;
    decoder.residual = ResidualForm::HyperConnections(HyperConnections {
        streams: STREAMS,
        mixing: HyperConnectionMixing::LowRankGated { rank: RANK },
    });
    // A dense feed-forward has no streams form: the block is its mixer.
    let sublayers = &mut decoder.blocks[0].sublayers;
    sublayers.truncate(1);
    sublayers[0].input = mix_input(site("attn_hc", true, hidden));
    sublayers[0].output = OutputForm::Scatter;
    decoder.exit.norm = exit_norm(site("output_hc", false, hidden));
    definition
}

/// The streams and low rank of the hyper-connection fixtures.
pub(crate) const STREAMS: u64 = 4;
const RANK: u64 = 64;

/// The smallest multiple of the planning fixture's hidden width at which
/// every backend's `stream_mix_down` declares a configuration for the
/// hyper-connection fixtures' mix sites (`STREAMS` streams, rank `RANK`),
/// with and without inject weights: read from each backend's declared
/// domain, so no fixture built at it names a shape a backend has no
/// configuration for.
pub(crate) fn mix_site_hidden() -> u64 {
    use seismic::{BackendName, NativeSpecialization, NativeSpecializationError};
    let implementations = BackendName::ALL.map(|backend| {
        seismic::generated::native_implementation_for_backend::<magnitude_kernels::stream_mix_down::Entry>(backend)
            .expect("the kernels bundle resolves `stream_mix_down`")
            .unwrap_or_else(|| panic!("{backend:?} implements `stream_mix_down`"))
    });
    let admits = |implementation: &seismic::NativeImplementation, hidden: u64, injected: bool| {
        let statics = implementation.statics.iter().fold(NativeSpecialization::new(), |statics, name| {
            let value = match name.as_str() {
                "H" => hidden,
                "S" => STREAMS,
                "LR" => RANK,
                "C" => u64::from(injected),
                other => unreachable!("`stream_mix_down` declares no static `{other}`"),
            };
            statics.with_static(name.clone(), value)
        });
        match implementation.admits(&statics) {
            Ok(()) => true,
            Err(NativeSpecializationError::Inadmissible) => false,
            Err(error) => panic!("`stream_mix_down`'s domain is evaluable: {error}"),
        }
    };
    (1..=64)
        .map(|multiple| multiple * HIDDEN)
        .find(|&hidden| {
            implementations
                .iter()
                .all(|implementation| admits(implementation, hidden, true) && admits(implementation, hidden, false))
        })
        .expect("every backend's `stream_mix_down` admits a mix site at some fixture hidden width")
}

/// A mix site at hidden width `hidden`, with inject weights when
/// `injected`.
fn site(prefix: &str, injected: bool, hidden: u64) -> magnitude_family_contracts::LowRankGatedSite {
    let wide = STREAMS * hidden;
    magnitude_family_contracts::LowRankGatedSite {
        norm: RmsNorm {
            weight: descriptor(&format!("{prefix}_norm"), &[wide]),
            epsilon: 1e-6,
        },
        down: descriptor(&format!("{prefix}_down"), &[RANK, wide]),
        up: descriptor(&format!("{prefix}_up"), &[wide, RANK]),
        inject: injected.then(|| descriptor(&format!("{prefix}_inject"), &[STREAMS, wide])),
    }
}

/// A sublayer's input through the low-rank `site`, unnormalized after it.
pub fn mix_input(site: magnitude_family_contracts::LowRankGatedSite) -> InputNorm {
    InputNorm::HyperConnectionMix(magnitude_family_contracts::HyperConnectionInput {
        site: magnitude_family_contracts::HyperConnectionWeights::LowRankGated(site),
        norm: None,
    })
}

/// The exit through its own low-rank `site`, unnormalized after it.
pub fn exit_norm(site: magnitude_family_contracts::LowRankGatedSite) -> ExitNorm {
    ExitNorm::HyperConnectionHead(magnitude_family_contracts::HyperConnectionExit {
        site: magnitude_family_contracts::ExitSite::Own(site),
        norm: None,
    })
}

/// A DeepSeek V4.1-shaped decoder over the fixture geometry: `blocks`
/// blocks of attention then a general routed feed-forward of 32 experts,
/// all selected, every sublayer reading through its Sinkhorn site over the
/// hyper-connection fixtures' streams (20 iterations) under its own norm,
/// mixing by the previous site's pre, and an exit that collapses the
/// streams by the last site's pre under the output norm.
pub fn sinkhorn_fixture_definition(blocks: u64) -> ModelDefinition {
    use magnitude_family_contracts::{ExitSite, SinkhornInput};
    sinkhorn_fixture_definition_of(blocks, SinkhornInput::PreviousSite, ExitSite::PreviousSite)
}

/// [`sinkhorn_fixture_definition`] with every site mixing by `input`'s pre
/// and the exit at `exit` (GLM-5.3's: each site's own, the streams' mean).
pub fn sinkhorn_fixture_definition_of(
    blocks: u64,
    input: magnitude_family_contracts::SinkhornInput,
    exit: magnitude_family_contracts::ExitSite,
) -> ModelDefinition {
    sinkhorn_fixture_definition_over(STREAMS, blocks, input, exit)
}

/// [`sinkhorn_fixture_definition_of`] over `streams` streams rather than the
/// hyper-connection fixtures' [`STREAMS`]: every site's weights are sized
/// from the count, so the definition stays consistent at any of them.
pub fn sinkhorn_fixture_definition_over(
    streams: u64,
    blocks: u64,
    input: magnitude_family_contracts::SinkhornInput,
    exit: magnitude_family_contracts::ExitSite,
) -> ModelDefinition {
    use magnitude_family_contracts::{
        ExpertSelection, HyperConnectionExit, HyperConnectionInput, HyperConnectionMixing,
        HyperConnectionWeights, HyperConnections, RouteNormalization, RoutedFfn, Router, RouterInput,
        ScoreFunction, SinkhornSite,
    };
    const EXPERTS: u64 = 32;
    const FEATURES: u64 = 64;
    let mut definition = fixture_definition();
    let decoder = &mut definition.decoder;
    let (hidden, wide) = (decoder.hidden, streams * decoder.hidden);
    decoder.residual = ResidualForm::HyperConnections(HyperConnections {
        streams,
        mixing: HyperConnectionMixing::Sinkhorn { iterations: 20, epsilon: 1e-6, norm_epsilon: 1e-6, input },
    });
    let rms = |name: &str| RmsNorm { weight: descriptor(name, &[hidden]), epsilon: 1e-6 };
    let mixes = (2 + streams) * streams;
    let sited = |prefix: &str, op: Operator| Sublayer {
        input: InputNorm::HyperConnectionMix(HyperConnectionInput {
            site: HyperConnectionWeights::Sinkhorn(SinkhornSite {
                function: descriptor(&format!("{prefix}_hc_fn"), &[mixes, wide]),
                base: descriptor(&format!("{prefix}_hc_base"), &[mixes]),
                scale: descriptor(&format!("{prefix}_hc_scale"), &[3]),
            }),
            norm: Some(rms(&format!("{prefix}_norm"))),
        }),
        op,
        output: OutputForm::Scatter,
    };
    let attention = decoder.blocks[0].sublayers[0].op.clone();
    decoder.blocks = (0..blocks)
        .map(|block| {
            let name = |part: &str| format!("blk.{block}.{part}");
            let routed = Operator::RoutedFfn(Box::new(RoutedFfn {
                experts: EXPERTS,
                selected: EXPERTS,
                intermediate: FEATURES,
                router: Router {
                    weight: descriptor(&name("ffn_gate_inp"), &[EXPERTS, hidden]),
                    input: RouterInput::Operator,
                    score: ScoreFunction::Softmax,
                    selection: ExpertSelection::TopK { bias: None },
                    media_bias: None,
                    normalization: RouteNormalization::Sum,
                    scale: 1.0,
                },
                expert_up: FeedForwardUp::Gated {
                    activation: ActivationFunction::Silu,
                    limit: None,
                    gate: descriptor(&name("ffn_gate_exps"), &[EXPERTS, FEATURES, hidden]),
                    up: descriptor(&name("ffn_up_exps"), &[EXPERTS, FEATURES, hidden]),
                },
                expert_down: descriptor(&name("ffn_down_exps"), &[EXPERTS, hidden, FEATURES]),
                expert_scale: None,
                latent: None,
                shared: None,
            }));
            Block { sublayers: vec![sited(&name("attn"), attention.clone()), sited(&name("ffn"), routed)] }
        })
        .collect();
    decoder.exit.norm = ExitNorm::HyperConnectionHead(HyperConnectionExit {
        site: exit,
        norm: Some(rms("output_norm")),
    });
    definition
}

/// GLM-5.3-Flash's slice A over the fixture geometry: one block of Kimi
/// Delta Attention (per-channel decay under its lower bound, low-rank decay
/// and sigmoid output gate) and a clamped dense feed-forward, each reading
/// through its own Sinkhorn site by that site's own pre, and the streams'
/// mean exit.
pub fn glm_slice_fixture_definition() -> ModelDefinition {
    use magnitude_family_contracts::{
        DecayGate, ExitSite, GatedDelta, GluLimit, LowRank, RecurrentHeadMapping, SinkhornInput,
    };
    const STATE: u64 = 128;
    const RANK: u64 = 64;
    const FEATURES: u64 = 64;
    let mut definition = sinkhorn_fixture_definition_of(1, SinkhornInput::OwnSite, ExitSite::Mean);
    let hidden = definition.decoder.hidden;
    let heads = 4;
    let inner = heads * STATE;
    let low_rank = |down: &str, up: &str| LowRank {
        down: descriptor(down, &[RANK, hidden]),
        up: descriptor(up, &[inner, RANK]),
    };
    let kda = Operator::GatedDelta(Box::new(GatedDelta {
        convolution_width: 4,
        key_heads: heads,
        value_heads: heads,
        width: STATE,
        head_mapping: RecurrentHeadMapping::Grouped,
        query_key_value: descriptor("blk.0.attn_qkv", &[3 * inner, hidden]),
        beta: descriptor("blk.0.ssm_beta", &[heads, hidden]),
        convolution: descriptor("blk.0.ssm_conv1d", &[3 * inner, 4]),
        decay: RecurrentDecay::PerChannel {
            projection: low_rank("blk.0.ssm_f_a", "blk.0.ssm_f_b"),
            bias: descriptor("blk.0.ssm_dt", &[inner]),
            decay: descriptor("blk.0.ssm_a", &[heads]),
            gate: DecayGate::BoundedSigmoid { lower: -5.0 },
        },
        key_norm_epsilon: 1e-6,
        // The recurrent norm shares its sublayer's input norm epsilon.
        norm: RmsNorm { weight: descriptor("blk.0.ssm_norm", &[STATE]), epsilon: 1e-6 },
        output_gate: RecurrentGate {
            projection: GateProjection::LowRank(low_rank("blk.0.ssm_g_a", "blk.0.ssm_g_b")),
            function: GateFunction::Sigmoid,
        },
        output: descriptor("blk.0.attn_output", &[hidden, inner]),
    }));
    let dense = Operator::DenseFfn(Box::new(DenseFfn {
        intermediate: FEATURES,
        up: FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            limit: Some(GluLimit::new(10.0).expect("a positive limit")),
            gate: descriptor("blk.0.ffn_gate", &[FEATURES, hidden]),
            up: descriptor("blk.0.ffn_up", &[FEATURES, hidden]),
        },
        down: descriptor("blk.0.ffn_down", &[hidden, FEATURES]),
    }));
    let [mixer, feed_forward] = definition.decoder.blocks[0].sublayers.as_mut_slice() else {
        unreachable!("the Sinkhorn fixture's blocks are a mixer and a feed-forward")
    };
    mixer.op = kda;
    feed_forward.op = dense;
    definition
}

/// GLM-5.3-Flash's slice B over the fixture geometry, with its package
/// manifest: slice A's block, then a block of GLM's latent attention and its
/// routed feed-forward, each behind its own Sinkhorn site.
/// - The attention: 4 query heads 64 wide off a rank-128 `qr`, absorbed per
///   head into the 128-wide latent row stored once as key and value, read
///   without positions over the rows its indexer picks: 4 heads 64 wide
///   querying from `qr`, each token's key LayerNorm-ed with a bias, pooled
///   per 4 tokens by a softmax gate with a per-place term, weighted per
///   head by a projection of the row.
/// - The feed-forward: 32 experts choosing 8 by sigmoid scores ranked on a
///   selection bias, weighted over their sum plus 1e-20 times 2.5, the
///   experts and the ungated shared expert clamped.
///
/// Its query, latent and output projections, experts and shared expert are
/// F8_E4M3 codes under F32 grids of 128×128 tiles, as GLM stores them.
///
/// Its indexer pools `GLM_SLICE_B_RATE` tokens to a unit and picks the
/// `GLM_SLICE_B_TOP` best.
pub fn glm_slice_b_fixture() -> (ModelDefinition, PackageManifest) {
    use magnitude_family_contracts::{
        Candidates, ExpertSelection, GluLimit, HeadWeights, HistoryReads, ImportTransform, Indexer,
        IndexerInput, IndexerKeys, IndexerQuery, LayerNorm, Pool, PoolNorm, PooledRows,
        RouteNormalization, RoutedFfn, Router, RouterInput, ScaleCodes, ScoreFunction, Selection,
        SharedExpert, SharedExpertGate,
    };
    const HEADS: u64 = 4;
    const HEAD: u64 = 64;
    const LATENT: u64 = 128;
    const QUERY_RANK: u64 = 128;
    const INDEX_WIDTH: u64 = 64;
    const RATE: u64 = GLM_SLICE_B_RATE;
    const EXPERTS: u64 = 32;
    const FEATURES: u64 = 128;
    const TILE: u64 = 128;
    let mut definition = glm_slice_fixture_definition();
    let hidden = definition.decoder.hidden;
    let name = |part: &str| format!("blk.1.{part}");
    let weight = |part: &str, shape: &[u64]| descriptor(&name(part), shape);
    // An FP8 weight under its `.scale` grid of whole tiles.
    let fp8 = |part: &str, shape: &[u64]| WeightDescriptor {
        transforms: vec![ImportTransform::ScaleBlocks {
            tensor: name(&format!("{part}.scale")),
            tile: [TILE, TILE],
            codes: ScaleCodes::F32,
        }],
        ..weight(part, shape)
    };
    let merged = |part: &str, stack: &[u64; 3]| WeightDescriptor {
        transforms: vec![ImportTransform::MergeLeadingAxes],
        ..weight(part, &[stack[0] * stack[1], stack[2]])
    };
    let rms = |part: &str, width| RmsNorm { weight: weight(part, &[width]), epsilon: 1e-6 };
    let limit = || Some(GluLimit::new(10.0).expect("a positive limit"));
    let indexer = Indexer {
        heads: HEADS,
        width: INDEX_WIDTH,
        query: IndexerQuery {
            input: IndexerInput::QueryRank,
            projection: weight("indexer.attn_q_b", &[HEADS * INDEX_WIDTH, QUERY_RANK]),
            norm: HeadNorm::None,
            rounding: RowRounding::None,
        },
        keys: IndexerKeys::Pooled(PooledRows {
            rate: RATE,
            projection: weight("indexer.attn_k", &[INDEX_WIDTH, hidden]),
            pool: Pool::SoftmaxGated {
                gate: weight("indexer_compressor_gate", &[INDEX_WIDTH, hidden]),
                position: Some(weight("indexer_compressor_ape", &[RATE, INDEX_WIDTH])),
            },
            norm: PoolNorm::Token {
                norm: LayerNorm { weight: weight("indexer.k_norm", &[INDEX_WIDTH]), epsilon: 1e-6 },
                bias: weight("indexer.k_norm.bias", &[INDEX_WIDTH]),
            },
            rounding: RowRounding::None,
        }),
        head_weights: HeadWeights::Projected(weight("indexer.proj", &[HEADS, hidden])),
        scale: ((HEADS * INDEX_WIDTH) as f64).powf(-0.5),
        top: GLM_SLICE_B_TOP,
        candidates: Candidates::All,
    };
    let attention = Operator::Attention(Box::new(Attention {
        heads: HEADS,
        kv_heads: 1,
        width: LATENT,
        query: QueryProjection::LowRank {
            down: fp8("attn_q_a", &[QUERY_RANK, hidden]),
            norm: rms("attn_q_a_norm", QUERY_RANK),
            up: fp8("attn_q_b", &[HEADS * HEAD, QUERY_RANK]),
        },
        query_keys: QueryKeys::Absorbed { weight: merged("attn_k_b", &[HEADS, LATENT, HEAD]), width: HEAD },
        gate: AttentionGate::None,
        query_norm: HeadNorm::None,
        history: vec![HistorySource {
            rows: HistoryRows::Own(RowProducer::Projected(ProjectedRows {
                key: fp8("attn_kv_a_mqa", &[LATENT, hidden]),
                value: ValueSource::StoredKey,
                key_norm: HeadNorm::Rms(rms("attn_kv_a_norm", LATENT)),
                value_norm: ValueNorm::None,
                domain: TokenDomain::Token,
                rounding: RowRounding::None,
            })),
            reads: HistoryReads::Selected(Selection::Own(Box::new(indexer))),
        }],
        rotary: Rotary::None,
        scale: (HEAD as f64).powf(-0.5),
        media_rows: MediaRowAttention::Causal,
        sinks: None,
        output: OutputProjection::GroupedLowRank {
            groups: HEADS,
            rank: HEAD,
            a: merged("attn_v_b", &[HEADS, HEAD, LATENT]),
            b: fp8("attn_output", &[hidden, HEADS * HEAD]),
        },
    }));
    let gated = |gate, up| FeedForwardUp::Gated { activation: ActivationFunction::Silu, limit: limit(), gate, up };
    let stacked = [EXPERTS, FEATURES, hidden];
    let routed = Operator::RoutedFfn(Box::new(RoutedFfn {
        experts: EXPERTS,
        selected: 8,
        intermediate: FEATURES,
        router: Router {
            weight: weight("ffn_gate_inp", &[EXPERTS, hidden]),
            input: RouterInput::Operator,
            score: ScoreFunction::Sigmoid,
            selection: ExpertSelection::TopK { bias: Some(weight("exp_probs_b.bias", &[EXPERTS])) },
            media_bias: None,
            normalization: RouteNormalization::SumPlusEpsilon(1e-20),
            scale: 2.5,
        },
        expert_up: gated(fp8("ffn_gate_exps", &stacked), fp8("ffn_up_exps", &stacked)),
        expert_down: fp8("ffn_down_exps", &[EXPERTS, hidden, FEATURES]),
        expert_scale: None,
        latent: None,
        shared: Some(SharedExpert {
            intermediate: FEATURES,
            up: gated(fp8("ffn_gate_shexp", &[FEATURES, hidden]), fp8("ffn_up_shexp", &[FEATURES, hidden])),
            down: fp8("ffn_down_shexp", &[hidden, FEATURES]),
            gate: SharedExpertGate::None,
        }),
    }));
    let mut block = definition.decoder.blocks[0].clone();
    let [mixer, feed_forward] = block.sublayers.as_mut_slice() else {
        unreachable!("slice A's block is a mixer and a feed-forward")
    };
    // Each sublayer keeps its Sinkhorn site's form under block 1's names.
    for (sublayer, site, op) in [(mixer, "attn", attention), (feed_forward, "ffn", routed)] {
        let InputNorm::HyperConnectionMix(input) = &mut sublayer.input else {
            unreachable!("slice A's sublayers read through Sinkhorn sites")
        };
        let magnitude_family_contracts::HyperConnectionWeights::Sinkhorn(weights) = &mut input.site else {
            unreachable!("slice A's sites are Sinkhorn sites")
        };
        for (descriptor, part) in [(&mut weights.function, "fn"), (&mut weights.base, "base"), (&mut weights.scale, "scale")] {
            descriptor.name = name(&format!("{site}_hc_{part}"));
        }
        if let Some(norm) = &mut input.norm {
            norm.weight.name = name(&format!("{site}_norm"));
        }
        sublayer.op = op;
    }
    definition.decoder.blocks.push(block);
    definition.validate().expect("the slice B fixture validates");
    let manifest = glm_slice_b_manifest(&definition);
    (definition, manifest)
}

/// Slice B's package (`glm_slice_b_fixture`): the stored weights of
/// `definition`'s decoder and head, its FP8 weights' codes each under its
/// grid of scales.
fn glm_slice_b_manifest(definition: &ModelDefinition) -> PackageManifest {
    use magnitude_family_contracts::ImportTransform;
    const HEADS: u64 = 4;
    const HEAD: u64 = 64;
    const LATENT: u64 = 128;
    const TILE: u64 = 128;
    let name = |part: &str| format!("blk.1.{part}");
    // Each FP8 weight's codes, one byte a value, then its grid of scales.
    let grids = crate::operators::decoder_weights(&definition.decoder)
        .into_iter()
        .filter_map(|(_, descriptor)| match descriptor.transforms.as_slice() {
            [ImportTransform::ScaleBlocks { tensor, .. }] => Some((descriptor.name.clone(), tensor.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let codes = grids.iter().map(|(codes, _)| codes.clone()).collect::<Vec<_>>();
    let mut manifest = relaid_manifest(fixture_manifest(definition), &codes, Encoding::F8E4M3, 32);
    let target = &mut manifest.target;
    // Every vector (a norm, a bias) is F32, as GLM stores them.
    let mut offset = 0;
    for tensor in &mut target.tensors {
        if let [width] = tensor.shape[..] {
            tensor.encoding = Encoding::F32;
            tensor.nbytes = 4 * width;
        }
        tensor.offset = offset;
        offset += tensor.nbytes;
    }
    target.files[0].size = offset;
    // The per-head stacks are stored as stacks.
    for (part, stack) in [("attn_k_b", [HEADS, LATENT, HEAD]), ("attn_v_b", [HEADS, HEAD, LATENT])] {
        let tensor = target.tensors.iter_mut().find(|tensor| tensor.name == name(part)).expect("a stored stack");
        tensor.shape = stack.to_vec();
    }
    for (codes, scale) in grids {
        let shape = &target.tensors.iter().find(|tensor| tensor.name == codes).expect("a stored weight").shape;
        let (leading, [rows, columns]) = shape.split_at(shape.len() - 2) else {
            unreachable!("a matrix or a stack of them")
        };
        let grid = leading.iter().copied().chain([rows / TILE, columns / TILE]).collect::<Vec<_>>();
        let (offset, nbytes) = (target.files[0].size, 4 * grid.iter().product::<u64>());
        target.tensors.push(TensorDescriptor { name: scale, shape: grid, encoding: Encoding::F32, offset, nbytes });
        target.files[0].size += nbytes;
    }
    manifest
}

/// `op`'s own indexer's weights, each renamed by `rename`.
#[cfg(test)]
fn rename_indexer_weights(op: &mut Operator, rename: &dyn Fn(&str) -> String) {
    use magnitude_family_contracts::{HeadWeights, HistoryReads, IndexerKeys, Pool, PoolNorm, Selection};
    let Operator::Attention(attention) = op else { return };
    let [HistorySource { reads: HistoryReads::Selected(Selection::Own(indexer)), .. }] = attention.history.as_mut_slice()
    else {
        return;
    };
    let head = |weight: &mut WeightDescriptor| weight.name = rename(&weight.name);
    head(&mut indexer.query.projection);
    let IndexerKeys::Pooled(keys) = &mut indexer.keys else { panic!("the fixture's indexer pools its own keys") };
    head(&mut keys.projection);
    let Pool::SoftmaxGated { gate, position } = &mut keys.pool else { panic!("the fixture's indexer pools by a gate") };
    head(gate);
    position.iter_mut().for_each(head);
    let PoolNorm::Token { norm, bias } = &mut keys.norm else { panic!("the fixture's indexer norms its tokens") };
    head(&mut norm.weight);
    head(bias);
    let HeadWeights::Projected(weights) = &mut indexer.head_weights else { panic!("the fixture's indexer projects head weights") };
    head(weights);
}

/// Tokens an indexer unit of [`glm_slice_b_fixture`] pools.
pub const GLM_SLICE_B_RATE: u64 = 4;
/// Units its indexer picks.
pub const GLM_SLICE_B_TOP: u64 = 2;

/// Slice B with an MTP head of the shape GLM-5.3-Flash's is: a head block
/// of block 1's mixer and general routed feed-forward (the latent attention
/// over its history, selected by an indexer of the head's own, then the
/// routed FP8 experts and the shared expert), plain-normed over a single
/// residual, read out through the target vocabulary projection and chaining
/// its output. Its operators' weights are block 1's (tied: stored once);
/// its norms, combine and indexer's are its own.
#[cfg(test)]
pub fn glm_slice_b_mtp_fixture() -> (ModelDefinition, PackageManifest) {
    glm_slice_b_mtp_fixture_of(true)
}

/// [`glm_slice_b_mtp_fixture`], its head's indexer reading the body's tied
/// weights where `own_indexer` is false. Both lay out the same tensors, the
/// head's own indexer's stored beside the body's, so a file written of either
/// holds every other tensor alike.
#[cfg(test)]
pub fn glm_slice_b_mtp_fixture_of(own_indexer: bool) -> (ModelDefinition, PackageManifest) {
    use magnitude_family_contracts::{
        Head, HeadBlock, HeadChain, HeadComponent, HeadHistory, HeadInput, HeadReadout, VocabularyProjection,
    };
    let (mut definition, _) = glm_slice_b_fixture();
    let hidden = definition.decoder.hidden;
    let named = |name: &str, shape: &[u64]| descriptor(&format!("head_{name}"), shape);
    let rms = |name: &str| RmsNorm { weight: named(name, &[hidden]), epsilon: 1e-6 };
    let trunk = &definition.decoder.blocks[1];
    // The head keeps its own pooled indexer state and selects as block 1
    // does, by indexer weights of its own.
    let plain = |norm: &str, sublayer: &Sublayer| {
        let mut op = sublayer.op.clone();
        rename_indexer_weights(&mut op, &|name| format!("head_{name}"));
        Sublayer { input: InputNorm::Rms(rms(norm)), op, output: OutputForm::Residual }
    };
    let head_block = Block {
        sublayers: vec![plain("attn_norm", &trunk.sublayers[0]), plain("ffn_norm", &trunk.sublayers[1])],
    };
    definition.head = Some(Head {
        blocks: vec![HeadBlock {
            residual: ResidualForm::Single,
            input: HeadInput::Combine {
                embedding_norm: rms("enorm"),
                hidden_norm: rms("hnorm"),
                combine: named("eh_proj", &[hidden, 2 * hidden]),
            },
            blocks: vec![head_block],
            output_norm: ExitNorm::Rms(rms("shared_head_norm")),
            readout: HeadReadout::full(VocabularyProjection::Target),
            chain: HeadChain::Output,
        }],
        history: HeadHistory::Own,
        component: HeadComponent::Target,
    });
    let manifest = glm_slice_b_manifest(&definition);
    if !own_indexer {
        // The head's indexer reads the body's tensors; its own stay stored.
        let head = definition.head.as_mut().expect("the fixture has a head");
        for sublayer in &mut head.blocks[0].blocks[0].sublayers {
            rename_indexer_weights(&mut sublayer.op, &|name| {
                name.strip_prefix("head_").expect("an indexer weight the head owns").to_owned()
            });
        }
    }
    definition.validate().expect("the slice B MTP fixture validates");
    (definition, manifest)
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
        ExpertSelection, GatedDelta, NGramConvolution, NGramGate, NGramHash, NGramInput,
        NGramMedia, RecurrentHeadMapping,
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
        limit: None,
        gate,
        up,
    };
    let mixing = |prefix: &str, op: Operator| Sublayer {
        input: mix_input(site(prefix, true, hidden)),
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
                media_bias: None,
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
        beta: descriptor("blk.0.ssm_beta", &[value_heads, hidden]),
        convolution: descriptor("blk.0.ssm_conv1d", &[channels, 4]),
        decay: RecurrentDecay::PerHead {
            alpha: descriptor("blk.0.ssm_alpha", &[value_heads, hidden]),
            time_bias: descriptor("blk.0.ssm_dt", &[value_heads]),
            decay: descriptor("blk.0.ssm_a", &[value_heads]),
        },
        key_norm_epsilon: 1e-6 * STATE as f64,
        norm: rms("blk.0.ssm_norm", STATE),
        output_gate: RecurrentGate {
            projection: GateProjection::Full(descriptor(
                "blk.0.attn_gate",
                &[value_heads * STATE, hidden],
            )),
            function: GateFunction::Sigmoid,
        },
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
            token_map: None,
            fill: 1,
            boundary: Some(1),
            media: NGramMedia::Hashed,
        },
        width: NGRAM_WIDTH,
        table: descriptor("per_layer_token_embd.weight", &[NGRAM_ROWS, NGRAM_WIDTH]),
        key: descriptor("blk.0.ngram_k", &[wide, heads * NGRAM_WIDTH]),
        value: descriptor("blk.0.ngram_v", &[hidden, heads * NGRAM_WIDTH]),
        gate: NGramGate::Norms {
            key: descriptor("blk.0.ngram_k_norm", &[wide]),
            query: descriptor("blk.0.ngram_q_norm", &[wide]),
        },
        epsilon: 1e-6,
        mixing: Some(NGramConvolution {
            norm: descriptor("blk.0.ngram_conv_norm", &[wide]),
            taps: descriptor("blk.0.ngram_conv", &[wide, 4]),
            kernel: 4,
            dilation: 1,
        }),
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

/// A DeepSeek V4.1-shaped decoder: the Sinkhorn fixture's two attention
/// blocks, block 0 ending with an Engram input (the trailing sublayer of
/// the block before its layer) that hashes mapped ids with a fill and no
/// boundary, gated by one stored product weight and added without a
/// convolution. The value projection is `ENGRAM_VALUE`.
pub fn engram_fixture_definition() -> ModelDefinition {
    use magnitude_family_contracts::{NGramGate, NGramHash, NGramInput, NGramMedia};
    const WIDTH: u64 = 32;
    const ROWS: u64 = 96;
    let mut definition = sinkhorn_fixture_definition(2);
    let decoder = &mut definition.decoder;
    let (hidden, wide) = (decoder.hidden, STREAMS * decoder.hidden);
    let (context, heads_per_order) = (3, 2);
    let heads = context * heads_per_order;
    // Raw ids sharing hashed ids, as a compressed vocabulary's do.
    let token_map = (0..decoder.vocabulary as u32).map(|token| (token * 7 + 2) % 61).collect();
    let ngram = Operator::NGramInput(Box::new(NGramInput {
        hash: NGramHash {
            context,
            heads_per_order,
            multipliers: vec![(1 << 40) + 3, 1_000_003, (1 << 41) + 17, 999_983],
            sizes: vec![ROWS / heads; heads as usize],
            offsets: (0..heads).map(|head| head * ROWS / heads).collect(),
            token_map: Some(token_map),
            fill: 2,
            boundary: None,
            media: NGramMedia::Blocking,
        },
        width: WIDTH,
        table: descriptor("per_layer_token_embd.weight", &[ROWS, WIDTH]),
        key: descriptor("blk.1.engram_k", &[wide, heads * WIDTH]),
        value: descriptor(ENGRAM_VALUE, &[hidden, heads * WIDTH]),
        gate: NGramGate::Product(descriptor("blk.1.engram_gate", &[wide])),
        epsilon: 1e-6,
        mixing: None,
    }));
    decoder.blocks[0].sublayers.push(Sublayer {
        input: InputNorm::None,
        op: ngram,
        output: OutputForm::Residual,
    });
    definition
}

/// The value projection of [`engram_fixture_definition`]'s Engram input.
pub const ENGRAM_VALUE: &str = "blk.1.engram_v";

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

/// Gemma 4 26B-A4B q4's routed shape, synthetic (no header of it is
/// committed): 30 blocks of attention then a routed feed-forward of 128
/// experts choosing 8, hidden 2,816, its experts' intermediate widened to
/// 1,152 (the routed kernels take a width in whole 64s) so the Q4_0 stacks
/// ([`gemma_moe_fixture_manifest`]) total about 21.0 GB, as the model's
/// own bank (about 21.1 GB), the bank a 24 GB device holds whole beside
/// little else. Its
/// attention (8 query heads sharing 2 key-value heads of 128) and its
/// 256-token vocabulary are small; a 16k context keeps its history about
/// 0.5 GB, and its whole-context prompt group 32 launches of 512 rows.
#[cfg(test)]
pub fn gemma_moe_fixture_definition() -> ModelDefinition {
    routed_fixture_definition(RoutedShape { layers: 30, experts: 128, selected: 8, hidden: 2816, intermediate: 1152 })
}

/// A routed model's shape: its layers, each layer's experts and those a
/// row selects, and its widths.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub struct RoutedShape {
    pub layers: u64,
    pub experts: u64,
    pub selected: u64,
    pub hidden: u64,
    pub intermediate: u64,
}

/// A model of `shape`, every block attention and a routed feed-forward, as
/// [`gemma_moe_fixture_definition`]'s are.
#[cfg(test)]
pub fn routed_fixture_definition(shape: RoutedShape) -> ModelDefinition {
    use magnitude_family_contracts::{ExpertSelection, RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction};
    let RoutedShape { layers, experts, selected, hidden, intermediate } = shape;
    const HEADS: u64 = 8;
    const KV_HEADS: u64 = 2;
    const WIDTH: u64 = 128;
    let mut definition = fixture_definition();
    let decoder = &mut definition.decoder;
    decoder.hidden = hidden;
    decoder.context_limit = 16_384;
    let vocabulary = decoder.vocabulary;
    let rms = |name: String, width| RmsNorm { weight: descriptor(&name, &[width]), epsilon: 1e-6 };
    decoder.entry.embedding = descriptor("embedding", &[vocabulary, hidden]);
    decoder.exit.norm = ExitNorm::Rms(rms("output_norm".into(), hidden));
    decoder.exit.output = descriptor("output", &[vocabulary, hidden]);
    let block = |block: u64| {
        let name = |part: &str| format!("blk.{block}.{part}");
        let attention = Attention {
            heads: HEADS,
            kv_heads: KV_HEADS,
            width: WIDTH,
            query: QueryProjection::Full(descriptor(&name("attn_qg"), &[2 * HEADS * WIDTH, hidden])),
            query_keys: QueryKeys::Direct,
            gate: AttentionGate::Interleaved { function: GateFunction::Sigmoid },
            query_norm: HeadNorm::Rms(rms(name("attn_q_norm"), WIDTH)),
            history: vec![HistorySource::visible(HistoryRows::Own(RowProducer::Projected(ProjectedRows {
                key: descriptor(&name("attn_k"), &[KV_HEADS * WIDTH, hidden]),
                value: ValueSource::Projected(descriptor(&name("attn_v"), &[KV_HEADS * WIDTH, hidden])),
                key_norm: HeadNorm::Rms(rms(name("attn_k_norm"), WIDTH)),
                value_norm: ValueNorm::None,
                domain: TokenDomain::Token,
                rounding: RowRounding::None,
            })))],
            rotary: Rotary::Interleaved { width: WIDTH / 2, base: 10_000.0, sections: vec![WIDTH / 4], axis_pattern: vec![0] },
            scale: 1.0 / (WIDTH as f64).sqrt(),
            media_rows: MediaRowAttention::Causal,
            sinks: None,
            output: OutputProjection::Full(descriptor(&name("attn_output"), &[hidden, HEADS * WIDTH])),
        };
        let routed = RoutedFfn {
            experts,
            selected,
            intermediate,
            router: Router {
                weight: descriptor(&name("ffn_gate_inp"), &[experts, hidden]),
                input: RouterInput::Operator,
                score: ScoreFunction::Softmax,
                selection: ExpertSelection::TopK { bias: None },
                media_bias: None,
                normalization: RouteNormalization::Sum,
                scale: 1.0,
            },
            expert_up: FeedForwardUp::Gated {
                activation: ActivationFunction::Silu,
                limit: None,
                gate: descriptor(&name("ffn_gate_exps"), &[experts, intermediate, hidden]),
                up: descriptor(&name("ffn_up_exps"), &[experts, intermediate, hidden]),
            },
            expert_down: descriptor(&name("ffn_down_exps"), &[experts, hidden, intermediate]),
            expert_scale: None,
            latent: None,
            shared: None,
        };
        Block {
            sublayers: vec![
                Sublayer {
                    input: InputNorm::Rms(rms(name("attn_norm"), hidden)),
                    op: Operator::Attention(Box::new(attention)),
                    output: OutputForm::Residual,
                },
                Sublayer {
                    input: InputNorm::Rms(rms(name("ffn_norm"), hidden)),
                    op: Operator::RoutedFfn(Box::new(routed)),
                    output: OutputForm::Residual,
                },
            ],
        }
    };
    decoder.blocks = (0..layers).map(block).collect();
    definition
}

/// [`gemma_moe_fixture_definition`]'s manifest: its expert stacks Q4_0
/// blocks (18 bytes a 32-element block), everything else F16.
#[cfg(test)]
pub fn gemma_moe_fixture_manifest(definition: &ModelDefinition) -> PackageManifest {
    let stacks = expert_stacks(definition).into_iter().flatten().collect::<Vec<_>>();
    relaid_manifest(fixture_manifest(definition), &stacks, Encoding::Q4_0, 18)
}
