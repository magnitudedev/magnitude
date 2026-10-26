//! LiLiCorr drafts (NVIDIA Nemotron 3.5 Bolt and Super-VL) at the catalog's
//! exact headers (`headers::draft`), and their DSpark drafts at pinned
//! artifacts' (`headers::artifact`): a convolved backbone without a
//! candidate selector, per-head attention sinks (Bolt), and the candidate
//! corrector. Each reads against its own target's header.

use magnitude_artifacts::{
    gguf::{Directory, Encoding, Scalar, TensorDescriptor, Value},
    ArtifactIdentity, PackageIdentity, TokenId,
};
use magnitude_family_common::headers;
use magnitude_family_contracts::{
    BlockAttention, BlockLayout, CandidateCorrector, DraftDefinition, DraftEmbedding, DraftMethod,
    DraftVariant, HistoryDomain, ImportTransform, KeyValue, ModelDefinition, ModelFamily, Operator,
    Rotary, CORRECTOR_FEATURES,
};
use magnitude_family_dflash::{inspect, recognize, Error};
use magnitude_family_nemotron_h::NemotronHFamily;

const BOLT: &str = "nemotron-3.5-bolt-74b-a9b:gguf:iq2-qat";
const SUPER: &str = "nemotron-3.5-super-vl-120b-a12b:gguf:q4";

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

fn target(model: &str) -> ModelDefinition {
    NemotronHFamily
        .inspect(&headers::target(model), None, identity())
        .unwrap_or_else(|error| panic!("target: {error}"))
}

fn read(draft: &Directory, model: &str) -> Result<DraftDefinition, Error> {
    let definition = target(model);
    inspect(draft, &definition, &|layer| {
        NemotronHFamily.layer_entry(&definition, layer)
    })
}

fn bolt() -> Directory {
    headers::draft(BOLT)
}

fn super_vl() -> Directory {
    headers::draft(SUPER)
}

fn corrector(draft: &DraftDefinition) -> &CandidateCorrector {
    match &draft.method {
        DraftMethod::LiLiCorr { corrector } => corrector,
        other => panic!("expected a LiLiCorr draft, read {other:?}"),
    }
}

fn scaled(weight: &magnitude_family_contracts::WeightDescriptor) -> bool {
    weight
        .transforms
        .iter()
        .any(|transform| matches!(transform, ImportTransform::ScaleByTensor { .. }))
}

fn attention(draft: &DraftDefinition, layer: usize) -> &magnitude_family_contracts::Attention {
    match &draft.blocks[layer].sublayers[0].op {
        Operator::Attention(attention) => attention,
        _ => panic!("a draft layer starts with attention"),
    }
}

#[test]
fn both_drafts_bind_and_validate_against_their_targets() {
    for (draft, model) in [(bolt(), BOLT), (super_vl(), SUPER)] {
        recognize(&draft).unwrap();
        let definition = read(&draft, model).unwrap();
        assert_eq!(definition.method.variant(), DraftVariant::LiLiCorr);
        let mut model = target(model);
        model.draft = Some(definition);
        model.validate().unwrap();
    }
}

#[test]
fn bolt_reads_a_selectorless_convolved_backbone_with_sinks() {
    let draft = read(&bolt(), BOLT).unwrap();
    assert_eq!(draft.layout, BlockLayout::MaskSlots);
    assert_eq!((draft.block_size, draft.max_proposals()), (8, 7));
    // The corrector scores every mask slot: a block is always whole.
    for proposals in 1..=7 {
        assert_eq!(draft.block_rows(proposals), 8);
    }
    assert_eq!(draft.mask_token, TokenId(990));
    assert_eq!(draft.taps.len(), 6);
    assert_eq!(draft.fusion.shape, [4096, 6 * 4096]);
    assert!(scaled(&draft.fusion));
    assert_eq!(draft.embedding, DraftEmbedding::Target);
    assert!(draft.output.is_none());
    assert_eq!(draft.blocks.len(), 5);
    let convolutions = draft.convolutions.as_ref().unwrap();
    assert_eq!((convolutions.kernel, convolutions.group), (2, 16));
    assert_eq!(convolutions.layers.len(), 5);
    assert_eq!(convolutions.layers[0].attention.base.shape, [2, 2, 4096]);
    assert_eq!(
        convolutions.layers[0].attention.projection.shape,
        [2 * 2 * 256, 4096]
    );
    assert_eq!(draft.block_attention, [BlockAttention::Bidirectional; 5]);
    for layer in 0..5 {
        let attention = attention(&draft, layer);
        assert_eq!(attention.sinks.as_ref().unwrap().shape, [32]);
        assert_eq!(
            (attention.heads, attention.kv_heads, attention.width),
            (32, 2, 128)
        );
        let KeyValue::Owned { domain, .. } = &attention.key_value else {
            panic!("a draft layer owns its history");
        };
        assert_eq!(*domain, HistoryDomain::Window { tokens: 1024 });
    }
}

#[test]
fn bolt_reads_its_corrector() {
    let draft = read(&bolt(), BOLT).unwrap();
    let corrector = corrector(&draft);
    assert_eq!(
        (
            corrector.width,
            corrector.top_k,
            corrector.heads,
            corrector.factor,
            corrector.mlp()
        ),
        (1024, 8, 8, 1024, 2048)
    );
    assert_eq!(corrector.vector_epsilon as f32, 1e-4);
    assert_eq!(corrector.logit_scale, 8.0);
    assert_eq!(corrector.feature_norm.epsilon, 1e-5);
    assert_eq!(corrector.feature_norm.weight.shape, [CORRECTOR_FEATURES]);
    let token = corrector.token.as_ref().unwrap();
    assert_eq!(token.weight.shape, [1024, 4096]);
    assert!(scaled(&token.weight));
    assert_eq!(corrector.pass.weight.shape, [1024, 4096]);
    assert_eq!(corrector.context.weight.shape, [1024, 4096]);
    assert_eq!(
        corrector.feature_up.weight.shape,
        [1024, CORRECTOR_FEATURES]
    );
    assert_eq!(corrector.feature_down.weight.shape, [1024, 1024]);
    assert_eq!(corrector.slot_embedding.shape, [7, 1024]);
    assert_eq!(corrector.rank_embedding.shape, [8, 1024]);
    assert_eq!(corrector.attention_bias.shape, [8, 56, 56]);
    assert_eq!(corrector.layers.len(), 2);
    for layer in &corrector.layers {
        assert_eq!(layer.attention_input.weight.shape, [3072, 1024]);
        assert_eq!(layer.attention_input.bias.shape, [3072]);
        assert!(!scaled(&layer.attention_input.weight));
        assert_eq!(layer.attention_output.weight.shape, [1024, 1024]);
        assert!(scaled(&layer.attention_output.weight));
        assert_eq!(layer.up.weight.shape, [2048, 1024]);
        assert_eq!(layer.down.weight.shape, [1024, 2048]);
        assert_eq!(layer.attention_norm.epsilon as f32, 1e-6);
    }
    assert_eq!(corrector.factor_input.weight.shape, [1024, 3072]);
    for head in [&corrector.outgoing, &corrector.incoming, &corrector.anchor] {
        assert_eq!(head.weight.shape, [1024, 1024]);
    }
}

#[test]
fn super_vl_reads_windowed_layers_without_sinks_and_its_own_projection() {
    let draft = read(&super_vl(), SUPER).unwrap();
    assert_eq!(corrector(&draft).layers.len(), 2);
    assert_eq!(draft.taps.len(), 5);
    assert_eq!(draft.block_attention, [BlockAttention::Bidirectional; 5]);
    let output = draft.output.as_ref().unwrap();
    assert_eq!(output.shape, [131072, 4096]);
    assert!(scaled(output));
    for layer in 0..5 {
        let attention = attention(&draft, layer);
        assert!(attention.sinks.is_none());
        assert_eq!((attention.heads, attention.kv_heads), (32, 8));
        let KeyValue::Owned { domain, .. } = &attention.key_value else {
            panic!("a draft layer owns its history");
        };
        assert_eq!(*domain, HistoryDomain::Window { tokens: 8192 });
        assert!(matches!(attention.rotary, Rotary::Table { .. }));
    }
}

const BOLT_DSPARK: &str = "nemotron-3.5-bolt-74b-a9b-dspark";
const SUPER_DSPARK: &str = "nemotron-3.5-super-vl-120b-a12b-dspark";

/// The published DSpark drafts: Bolt's bidirectional with the anchor's own
/// mask token, Super-VL's causal over a 2048-token window, both sampling
/// from the anchor with a confidence head.
#[test]
fn both_dspark_drafts_bind_against_their_targets() {
    for (draft, model, layers, attention_kind, mask, window) in [
        (
            BOLT_DSPARK,
            BOLT,
            6,
            BlockAttention::Bidirectional,
            18,
            None,
        ),
        (
            SUPER_DSPARK,
            SUPER,
            5,
            BlockAttention::Causal,
            999,
            Some(2048),
        ),
    ] {
        let draft = headers::artifact(draft);
        recognize(&draft).unwrap();
        let definition = read(&draft, model).unwrap();
        assert_eq!(definition.method.variant(), DraftVariant::DSpark);
        assert_eq!(definition.layout, BlockLayout::AnchorFirst);
        assert_eq!(definition.mask_token, TokenId(mask));
        assert_eq!(definition.block_attention, vec![attention_kind; layers]);
        for layer in 0..layers {
            let KeyValue::Owned { domain, .. } = &attention(&definition, layer).key_value else {
                panic!("a draft layer owns its history");
            };
            if let Some(tokens) = window {
                assert_eq!(*domain, HistoryDomain::Window { tokens });
            }
        }
        let mut model = target(model);
        model.draft = Some(definition);
        model.validate().unwrap();
    }
}

fn tensor(name: &str, shape: &[u64]) -> TensorDescriptor {
    TensorDescriptor {
        name: name.into(),
        shape: shape.to_vec(),
        encoding: Encoding::F32,
        offset: 0,
        nbytes: shape.iter().product::<u64>() * 4,
    }
}

#[test]
fn unknown_corrector_metadata_is_refused() {
    let mut draft = bolt();
    headers::set(
        &mut draft,
        "lilicorr.dropout",
        Value::Scalar(Scalar::Float(0.1)),
    );
    assert_eq!(
        recognize(&draft),
        Err(Error::Unadmitted {
            metadata: vec!["lilicorr.dropout".into()],
            tensors: Vec::new(),
        })
    );
}

#[test]
fn a_missing_corrector_weight_is_refused() {
    let mut draft = bolt();
    headers::remove_tensor(&mut draft, "lilicorr.in_head.bias");
    assert_eq!(
        read(&draft, BOLT),
        Err(Error::MissingWeight("lilicorr.in_head.bias".into()))
    );
}

#[test]
fn a_corrector_layer_beyond_the_declared_count_is_unbound() {
    let mut draft = bolt();
    draft
        .tensors
        .push(tensor("blk.8.attn_norm.weight", &[1024]));
    assert_eq!(
        read(&draft, BOLT),
        Err(Error::Unadmitted {
            metadata: Vec::new(),
            tensors: vec!["blk.8.attn_norm.weight".into()],
        })
    );
}

#[test]
fn selector_metadata_without_its_roles_is_refused() {
    let mut draft = bolt();
    headers::set(
        &mut draft,
        "dflash.selector_top_k",
        Value::Scalar(Scalar::Unsigned(8)),
    );
    assert!(matches!(read(&draft, BOLT), Err(Error::Geometry(_))));
}

#[test]
fn a_corrector_proposes_from_its_mask_slots() {
    let mut draft = bolt();
    headers::set(
        &mut draft,
        "dflash.sample_from_anchor",
        Value::Scalar(Scalar::Bool(true)),
    );
    assert!(matches!(read(&draft, BOLT), Err(Error::Geometry(_))));
}

#[test]
fn sinks_take_one_logit_per_query_head() {
    let mut draft = bolt();
    headers::reshape_tensor(&mut draft, "blk.2.attn_sinks.weight", &[16]);
    assert!(matches!(
        read(&draft, BOLT),
        Err(Error::WeightShape { name, .. }) if name == "blk.2.attn_sinks.weight"
    ));
}

#[test]
fn a_convolved_backbone_declares_its_kernel_and_group() {
    let mut draft = bolt();
    draft
        .metadata
        .retain(|entry| entry.name != "dflash.conv_group_size");
    assert!(matches!(read(&draft, BOLT), Err(Error::Geometry(_))));
}
