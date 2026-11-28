//! Regression test for banked MTP head routing-record allocation:
//! when `places` is smaller than the MTP head's chained drafting steps
//! (e.g. `places = 7` -> `2 * 7 = 14` routing records while MTP drafts
//! `15` steps on `repeat:4k`), the expert bank's routing-record ring
//! (`flight`) must size for `flights.steps(places.max(head_steps))`.

use magnitude_artifacts::{
    gguf::{Encoding, TensorDescriptor},
    ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
};
use magnitude_executor::{
    resident_layout, BankAllocation, BankSubstrate, ComponentSelection, DraftWidths, ExecutionPath,
    HostTier, ModelLoadPlan, PlannedMethod, ResourceLimits,
};
use magnitude_family_contracts::{
    ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder, EmbeddingScale,
    EntryForm, ExitForm, ExitNorm, ExpertSelection, FeedForwardUp, GateFunction, Head, HeadBlock,
    HeadChain, HeadComponent, HeadHistory, HeadInput, HeadNorm, HeadReadout, HistoryRows,
    HistorySource, InputNorm, InputSemantics, MediaRowAttention, ModelDefinition, Operator,
    OutputForm, OutputProjection, ProjectedRows, QueryKeys, QueryProjection, ResidualForm, RmsNorm,
    Rotary, RouteNormalization, RoutedFfn, Router, RouterInput, RowProducer, RowRounding,
    ScoreFunction, Sublayer, TokenDomain, ValueNorm, ValueSource, VocabularyProjection,
    WeightDescriptor, WeightScope,
};
use seismic::{BackendName, NativeAllocations};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;

fn limits() -> ResourceLimits {
    ResourceLimits {
        max_launch_rows: 512,
        max_launch_slots: 16,
        max_selected_rows: 16,
        max_drafting_slots: 16,
        exported_logits_rows: 0,
        max_images_per_request: 1,
        max_image_cells: 0,
        lookahead: true,
        chain_capacity: NonZeroUsize::MIN,
        conversations: NonZeroUsize::MIN,
    }
}

fn desc(
    unique: &mut BTreeMap<String, Vec<u64>>,
    name: &str,
    shape: &[u64],
) -> WeightDescriptor {
    unique
        .entry(name.to_owned())
        .or_insert_with(|| shape.to_vec());
    WeightDescriptor::stored(name, shape)
}

fn rms(unique: &mut BTreeMap<String, Vec<u64>>, name: &str, width: u64) -> RmsNorm {
    RmsNorm {
        weight: desc(unique, name, &[width]),
        epsilon: 1e-6,
    }
}

fn routed_block(
    unique: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    hidden: u64,
    experts: u64,
    features: u64,
) -> Block {
    const WIDTH: u64 = 64;
    let name = |part: &str| format!("{prefix}.{part}");
    let attention = Attention {
        heads: 2,
        kv_heads: 1,
        width: WIDTH,
        query: QueryProjection::Full(desc(unique, &name("qg"), &[2 * 2 * WIDTH, hidden])),
        query_keys: QueryKeys::Direct,
        gate: AttentionGate::Interleaved {
            function: GateFunction::Sigmoid,
        },
        query_norm: HeadNorm::Rms(rms(unique, &name("qn"), WIDTH)),
        history: vec![HistorySource::visible(HistoryRows::Own(
            RowProducer::Projected(ProjectedRows {
                key: desc(unique, &name("k"), &[WIDTH, hidden]),
                value: ValueSource::Projected(desc(unique, &name("v"), &[WIDTH, hidden])),
                key_conv: None,
                value_conv: None,
                key_norm: HeadNorm::Rms(rms(unique, &name("kn"), WIDTH)),
                value_norm: ValueNorm::None,
                domain: TokenDomain::Token,
                rounding: RowRounding::None,
            }),
        ))],
        rotary: Rotary::Interleaved {
            width: WIDTH / 2,
            base: 10_000.0,
            sections: vec![WIDTH / 4],
            axis_pattern: vec![0],
        },
        relative_bias: None,
        log_scale: None,
        scale: 1.0 / (WIDTH as f64).sqrt(),
        media_rows: MediaRowAttention::Causal,
        sinks: magnitude_family_contracts::AttentionSinks::None,
        output: OutputProjection::Full(desc(unique, &name("o"), &[hidden, 2 * WIDTH])),
    };
    let routed = RoutedFfn {
        experts,
        selected: 2,
        intermediate: features,
        router: Router {
            weight: desc(unique, &name("router"), &[experts, hidden]),
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
            gate: desc(unique, &name("gate_exps"), &[experts, features, hidden]),
            up: desc(unique, &name("up_exps"), &[experts, features, hidden]),
        },
        expert_down: desc(unique, &name("down_exps"), &[experts, hidden, features]),
        expert_scale: None,
        latent: None,
        shared: None,
    };
    Block {
        sublayers: vec![
            Sublayer {
                input: InputNorm::Rms(rms(unique, &name("attn_norm"), hidden)),
                op: Operator::Attention(Box::new(attention)),
                post_conv: None,
                output: OutputForm::Residual,
            },
            Sublayer {
                input: InputNorm::Rms(rms(unique, &name("ffn_norm"), hidden)),
                op: Operator::RoutedFfn(Box::new(routed)),
                post_conv: None,
                output: OutputForm::Residual,
            },
        ],
        transition: None,
    }
}

fn routed_mtp_fixture() -> (ModelDefinition, PackageManifest) {
    const HIDDEN: u64 = 128;
    const VOCAB: u64 = 256;
    const EXPERTS: u64 = 64;
    const FEATURES: u64 = 64;

    let mut unique = BTreeMap::<String, Vec<u64>>::new();
    let identity = PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    };
    let target_block = routed_block(&mut unique, "blk.0", HIDDEN, EXPERTS, FEATURES);
    let head_layer = routed_block(&mut unique, "head.0", HIDDEN, EXPERTS, FEATURES);
    let head = Head {
        component: HeadComponent::Target,
        history: HeadHistory::Own,
        blocks: vec![HeadBlock {
            residual: ResidualForm::Single,
            input: HeadInput::Combine {
                embedding_norm: rms(&mut unique, "head_token_norm", HIDDEN),
                hidden_norm: rms(&mut unique, "head_cond_norm", HIDDEN),
                combine: desc(&mut unique, "head_combine", &[HIDDEN, 2 * HIDDEN]),
            },
            blocks: vec![head_layer],
            output_norm: ExitNorm::Rms(rms(&mut unique, "head_out_norm", HIDDEN)),
            readout: HeadReadout::full(VocabularyProjection::Target),
            chain: HeadChain::Output,
        }],
    };
    let definition = ModelDefinition {
        family: magnitude_family_contracts::FamilyId("routed-mtp-fixture".into()),
        artifact_identity: identity,
        inputs: InputSemantics { coordinate_axes: 1 },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden: HIDDEN,
            vocabulary: VOCAB,
            context_limit: 4096,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding: desc(&mut unique, "embedding", &[VOCAB, HIDDEN]),
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
            },
            blocks: vec![target_block],
            exit: ExitForm {
                norm: ExitNorm::Rms(rms(&mut unique, "output_norm", HIDDEN)),
                output: desc(&mut unique, "output", &[VOCAB, HIDDEN]),
                unpadded_vocabulary: None,
                softcap: None,
            },
        },
        head: Some(head),
        vision: None,
        draft: None,
    };

    let mut offset = 0u64;
    let mut tensors = Vec::with_capacity(unique.len());
    for (name, shape) in unique {
        let is_expert_stack =
            name.ends_with(".gate_exps") || name.ends_with(".up_exps") || name.ends_with(".down_exps");
        let elements: u64 = shape.iter().product();
        let (encoding, nbytes) = if is_expert_stack {
            (Encoding::Q4_0, elements / 32 * 18)
        } else {
            (Encoding::BF16, elements * 2)
        };
        tensors.push(TensorDescriptor {
            name,
            shape,
            encoding,
            offset,
            nbytes,
        });
        offset += nbytes;
    }
    let manifest = PackageManifest {
        identity,
        target: ComponentManifest {
            files: vec![ComponentFile {
                path: PathBuf::from("routed-mtp-fixture.gguf"),
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

#[test]
fn planned_method_head_steps_is_single_authority() {
    assert_eq!(PlannedMethod::Plain.head_steps(), 1);
    assert_eq!(
        PlannedMethod::DFlash {
            proposals: 15,
            widths: DraftWidths::Adaptive,
        }
        .head_steps(),
        1
    );
    assert_eq!(
        PlannedMethod::Mtp {
            greedy_proposals: 15,
            sampled_proposals: 3,
            widths: DraftWidths::Adaptive,
        }
        .head_steps(),
        15
    );
    assert_eq!(
        PlannedMethod::Mtp {
            greedy_proposals: 3,
            sampled_proposals: 12,
            widths: DraftWidths::Fixed(3),
        }
        .head_steps(),
        12
    );
}

#[test]
fn banked_mtp_head_sizes_routing_records_for_max_head_steps() {
    let (definition, manifest) = routed_mtp_fixture();
    let backend = BackendName::Cuda;
    let layout = resident_layout(ExecutionPath::Native, backend);
    let substrate = BankSubstrate::new(backend, NativeAllocations::EXACT, HostTier::Linked);
    let method = PlannedMethod::Mtp {
        greedy_proposals: 15,
        sampled_proposals: 15,
        widths: DraftWidths::Adaptive,
    };
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: true,
            vision: false,
        },
        layout,
        method,
        limits(),
    )
    .expect("derive routed MTP load plan");

    // Exact regression scenario: `places = 7` with pipelined depth (`groups = 2`)
    // previously allocated `2 * 7 = 14` routing records while MTP submitted
    // 15-step head groups (`repeat:4k` high-acceptance MTP).
    let banked = load
        .clone()
        .with_whole_expert_bank(&definition, None, 7, limits().flights(), substrate)
        .expect("bank routed MTP with places = 7");
    let bank = banked.expert_bank().expect("banked plan holds its bank");
    assert!(
        bank.layers()
            .iter()
            .any(|layer| matches!(layer.scope, WeightScope::HeadSublayer(_))),
        "fixture must bank its HeadSublayer"
    );
    assert_eq!(bank.places, 7);
    assert_eq!(bank.head_steps, 15);
    assert_eq!(bank.group_steps(), 15);
    assert_eq!(bank.flight(), 30);
    for layer in 0..bank.layers().len() {
        assert_eq!(bank.binding(layer).unwrap().words.flight, 30);
    }
    let bank_layout = bank.layout().unwrap();
    for layer in &bank_layout.layers {
        let BankAllocation::Words { extents, .. } = &layer.routing else {
            panic!("routing must be Words");
        };
        assert_eq!(extents[0], 30);
        let BankAllocation::Words { extents, .. } = &layer.keys else {
            panic!("keys must be Words");
        };
        assert_eq!(extents[0], 30);
    }
    assert_eq!(
        bank.storage_demand_at(bank.slots, bank.places, bank.paths)
            .unwrap(),
        bank.storage_demand().unwrap(),
        "storage_demand_at must match storage_demand when head_steps > places"
    );

    // When `places > head_steps`, `group_steps` follows `places`.
    let wider = load
        .with_whole_expert_bank(&definition, None, 20, limits().flights(), substrate)
        .expect("bank routed MTP with places = 20");
    let wider_bank = wider.expert_bank().unwrap();
    assert_eq!(wider_bank.group_steps(), 20);
    assert_eq!(wider_bank.flight(), 40);

    // Without a head component (`Plain`), `head_steps` stays 1 (no extra routing records).
    let plain = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        layout,
        PlannedMethod::Plain,
        limits(),
    )
    .expect("derive plain load plan")
    .with_whole_expert_bank(&definition, None, 7, limits().flights(), substrate)
    .expect("bank plain load plan");
    let plain_bank = plain.expert_bank().unwrap();
    assert_eq!(plain_bank.head_steps, 1);
    assert_eq!(plain_bank.group_steps(), 7);
    assert_eq!(plain_bank.flight(), 14);
}
