//! Dialect-perturbation tests across all model families and vision projectors.
//!
//! Verifies that benign GGUF dialect variations (omitting defaulted keys,
//! adding benign keys with their invariant values, and reshaping singleton
//! axes) are admitted cleanly, while mutating architectural invariants is
//! strictly rejected.

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    ArtifactIdentity, PackageIdentity,
};
use magnitude_family_common::{
    dialect::{
        assert_dialect_equivalent, assert_dialect_rejected, remove_metadata, reshape_tensor,
        set_metadata,
    },
    headers,
};
use magnitude_family_contracts::{FamilyError, ModelDefinition};
use magnitude_family_registry::FAMILIES;

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([9; 32]),
        projector: None,
    }
}

fn inspect_target(directory: &Directory) -> Result<ModelDefinition, FamilyError> {
    let family = FAMILIES
        .iter()
        .find(|family| family.recognizes(directory))
        .expect("a family recognizes the header");
    family.inspect(directory, None, identity())
}

fn inspect_with_projector(
    target: &Directory,
    projector: &Directory,
) -> Result<ModelDefinition, FamilyError> {
    let family = FAMILIES
        .iter()
        .find(|family| family.recognizes(target))
        .expect("a family recognizes the header");
    family.inspect(target, Some(projector), identity())
}

#[test]
fn deepseek41_dialect_equivalence_and_invariant_rejection() {
    let base = headers::target("deepseek-v4.1-flash:gguf:mxfp4");
    assert_dialect_equivalent(&base, inspect_target, |dir| {
        remove_metadata(dir, "deepseek41.expert_gating_func");
        remove_metadata(dir, "deepseek41.hash_layer_count");
        remove_metadata(dir, "deepseek41.attention.head_count_kv");
        remove_metadata(dir, "deepseek41.attention.value_length");
        set_metadata(
            dir,
            "deepseek41.rope.scaling.yarn_attn_factor",
            Value::Scalar(Scalar::Float(1.0)),
        );
        let vocab = dir
            .tensor("token_embd.weight")
            .unwrap()
            .shape[0];
        set_metadata(
            dir,
            "deepseek41.vocab_size",
            Value::Scalar(Scalar::Unsigned(vocab)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "deepseek41.expert_gating_func",
            Value::Scalar(Scalar::Unsigned(2)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "deepseek41.hash_layer_count",
            Value::Scalar(Scalar::Unsigned(1)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "deepseek41.vocab_size",
            Value::Scalar(Scalar::Unsigned(17)),
        );
    });
}

#[test]
fn glm5next_dialect_equivalence_and_invariant_rejection() {
    let base = headers::target("glm-5.3-flash:gguf:fp8");
    assert_dialect_equivalent(&base, inspect_target, |dir| {
        remove_metadata(dir, "glm5-next.vocab_size");
        remove_metadata(dir, "glm5-next.rope.dimension_count");
        remove_metadata(dir, "glm5-next.expert_gating_func");
        remove_metadata(dir, "glm5-next.expert_group_count");
        remove_metadata(dir, "glm5-next.expert_group_used_count");
        remove_metadata(dir, "glm5-next.attention.key_length");
        remove_metadata(dir, "glm5-next.attention.value_length");
        remove_metadata(dir, "glm5-next.attention.indexer.kpool_select_tail");
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "glm5-next.rope.dimension_count",
            Value::Scalar(Scalar::Unsigned(64)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "glm5-next.expert_gating_func",
            Value::Scalar(Scalar::Unsigned(1)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "glm5-next.expert_group_count",
            Value::Scalar(Scalar::Unsigned(4)),
        );
    });

    // Projector dialect equivalence & rejection.
    let proj = headers::projector("glm-5.3-flash:gguf:fp8");
    assert_dialect_equivalent(&proj, |p| inspect_with_projector(&base, p), |dir| {
        remove_metadata(dir, "clip.use_silu");
        remove_metadata(dir, "clip.vision.image_size");
        set_metadata(
            dir,
            "clip.has_audio_encoder",
            Value::Scalar(Scalar::Bool(false)),
        );
    });
    assert_dialect_rejected(&proj, |p| inspect_with_projector(&base, p), |dir| {
        set_metadata(dir, "clip.use_silu", Value::Scalar(Scalar::Bool(false)));
    });
    assert_dialect_rejected(&proj, |p| inspect_with_projector(&base, p), |dir| {
        set_metadata(
            dir,
            "clip.has_audio_encoder",
            Value::Scalar(Scalar::Bool(true)),
        );
    });
}

#[test]
fn nemotron_h_dialect_equivalence_and_invariant_rejection() {
    let base = headers::target("nemotron-3.5-super-vl-120b-a12b:gguf:q4");
    assert_dialect_equivalent(&base, inspect_target, |dir| {
        remove_metadata(dir, "nemotron_h_moe.vocab_size");
        remove_metadata(dir, "nemotron_h_moe.attention.value_length");
        remove_metadata(dir, "nemotron_h_moe.expert_group_count");
        remove_metadata(dir, "nemotron_h_moe.expert_group_used_count");
        remove_metadata(dir, "nemotron_h_moe.expert_gating_func");
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "nemotron_h_moe.expert_group_count",
            Value::Scalar(Scalar::Unsigned(2)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "nemotron_h_moe.attention.value_length",
            Value::Scalar(Scalar::Unsigned(32)),
        );
    });

    // Singleton-axis reshaping on conv1d, ssm_a/ssm_d, and ssm_norm.
    let mut reshaped = base.clone();
    let conv_shape = reshaped
        .tensor("blk.0.ssm_conv1d.weight")
        .unwrap()
        .shape
        .clone();
    reshape_tensor(
        &mut reshaped,
        "blk.0.ssm_conv1d.weight",
        &[conv_shape[0], 1, conv_shape[1]],
    );
    let a_len = reshaped.tensor("blk.0.ssm_a").unwrap().shape[0];
    reshape_tensor(&mut reshaped, "blk.0.ssm_a", &[a_len]);
    let norm_len: u64 = reshaped
        .tensor("blk.0.ssm_norm.weight")
        .unwrap()
        .shape
        .iter()
        .product();
    reshape_tensor(&mut reshaped, "blk.0.ssm_norm.weight", &[norm_len]);
    assert!(inspect_target(&reshaped).is_ok());

    // Projector dialect equivalence & rejection.
    let proj = headers::projector("nemotron-3.5-super-vl-120b-a12b:gguf:q4");
    assert_dialect_equivalent(&proj, |p| inspect_with_projector(&base, p), |dir| {
        remove_metadata(dir, "clip.use_gelu");
        remove_metadata(dir, "clip.vision.image_size");
        set_metadata(
            dir,
            "clip.has_audio_encoder",
            Value::Scalar(Scalar::Bool(false)),
        );
    });
    assert_dialect_rejected(&proj, |p| inspect_with_projector(&base, p), |dir| {
        set_metadata(dir, "clip.use_gelu", Value::Scalar(Scalar::Bool(false)));
    });
}

#[test]
fn gemma4_dialect_equivalence_and_invariant_rejection() {
    let base = headers::target("gemma-4-12b-it-qat:gguf:q4");
    assert_dialect_equivalent(&base, inspect_target, |dir| {
        remove_metadata(dir, "gemma4.attention.value_length");
        remove_metadata(dir, "gemma4.attention.value_length_swa");
        remove_metadata(dir, "gemma4.rope.dimension_count");
        remove_metadata(dir, "gemma4.rope.dimension_count_swa");
        set_metadata(
            dir,
            "gemma4.expert_count",
            Value::Scalar(Scalar::Unsigned(0)),
        );
        let vocab = dir.tensor("token_embd.weight").unwrap().shape[0];
        set_metadata(
            dir,
            "gemma4.vocab_size",
            Value::Scalar(Scalar::Unsigned(vocab)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "gemma4.attention.value_length",
            Value::Scalar(Scalar::Unsigned(32)),
        );
    });
    assert_dialect_rejected(&base, inspect_target, |dir| {
        set_metadata(
            dir,
            "gemma4.vocab_size",
            Value::Scalar(Scalar::Unsigned(13)),
        );
    });

    let proj = headers::projector("gemma-4-12b-it-qat:gguf:q4");
    assert_dialect_equivalent(&proj, |p| inspect_with_projector(&base, p), |dir| {
        set_metadata(dir, "clip.use_gelu", Value::Scalar(Scalar::Bool(true)));
    });
    assert_dialect_rejected(&proj, |p| inspect_with_projector(&base, p), |dir| {
        set_metadata(dir, "clip.use_gelu", Value::Scalar(Scalar::Bool(false)));
    });
}

#[test]
fn lfm2_and_muse_and_qwen_dialect_equivalence_and_rejection() {
    // LFM2 MoE: rope.scaling.type = "none", expert_weights_norm = true,
    // expert_weights_scale = 1.0, and 3D shortconv.conv.weight.
    let lfm2 = headers::target("lfm2.5-24b-a2b:gguf:q4");
    assert_dialect_equivalent(&lfm2, inspect_target, |dir| {
        set_metadata(
            dir,
            "lfm2moe.rope.scaling.type",
            Value::Scalar(Scalar::String("none".into())),
        );
        set_metadata(
            dir,
            "lfm2moe.expert_weights_norm",
            Value::Scalar(Scalar::Bool(true)),
        );
        set_metadata(
            dir,
            "lfm2moe.expert_weights_scale",
            Value::Scalar(Scalar::Float(1.0)),
        );
    });
    assert_dialect_rejected(&lfm2, inspect_target, |dir| {
        set_metadata(
            dir,
            "lfm2moe.rope.scaling.type",
            Value::Scalar(Scalar::String("linear".into())),
        );
    });
    assert_dialect_rejected(&lfm2, inspect_target, |dir| {
        set_metadata(
            dir,
            "lfm2moe.expert_weights_norm",
            Value::Scalar(Scalar::Bool(false)),
        );
    });
    assert_dialect_rejected(&lfm2, inspect_target, |dir| {
        set_metadata(
            dir,
            "lfm2moe.expert_weights_scale",
            Value::Scalar(Scalar::Float(2.0)),
        );
    });
    let mut lfm2_conv3d = lfm2.clone();
    let conv_name = lfm2_conv3d
        .tensors
        .iter()
        .find(|t| t.name.ends_with("shortconv.conv.weight"))
        .unwrap()
        .name
        .clone();
    let conv_shape = lfm2_conv3d.tensor(&conv_name).unwrap().shape.clone();
    reshape_tensor(
        &mut lfm2_conv3d,
        &conv_name,
        &[conv_shape[0], 1, conv_shape[1]],
    );
    assert!(inspect_target(&lfm2_conv3d).is_ok());

    // Muse Glimmer: omit attention.value_length; projector clip.use_gelu & batched pos embd.
    let muse = headers::target("muse-glimmer-30b:gguf:q4");
    assert_dialect_equivalent(&muse, inspect_target, |dir| {
        remove_metadata(dir, "muse-glimmer.attention.value_length");
    });
    let muse_proj = headers::projector("muse-glimmer-30b:gguf:q4");
    assert_dialect_equivalent(&muse_proj, |p| inspect_with_projector(&muse, p), |dir| {
        set_metadata(dir, "clip.use_gelu", Value::Scalar(Scalar::Bool(true)));
        set_metadata(
            dir,
            "clip.has_audio_encoder",
            Value::Scalar(Scalar::Bool(false)),
        );
    });
    let mut muse_batched = muse_proj.clone();
    let pos_shape = muse_batched
        .tensor("v.position_embd.weight")
        .unwrap()
        .shape
        .clone();
    reshape_tensor(
        &mut muse_batched,
        "v.position_embd.weight",
        &[1, pos_shape[0], pos_shape[1]],
    );
    assert!(inspect_with_projector(&muse, &muse_batched).is_ok());

    // Qwen 3.5: vocab_size, 3D ssm_conv1d.weight, projector batched v.position_embd.weight.
    let qwen = headers::target("qwen3.5-4b:gguf:q4");
    assert_dialect_equivalent(&qwen, inspect_target, |dir| {
        remove_metadata(dir, "qwen35.attention.value_length");
        remove_metadata(dir, "qwen35.ssm.inner_size");
        let vocab = dir.tensor("token_embd.weight").unwrap().shape[0];
        set_metadata(
            dir,
            "qwen35.vocab_size",
            Value::Scalar(Scalar::Unsigned(vocab)),
        );
    });
    let mut qwen_conv3d = qwen.clone();
    let qconv_name = qwen_conv3d
        .tensors
        .iter()
        .find(|t| t.name.ends_with("ssm_conv1d.weight"))
        .unwrap()
        .name
        .clone();
    let qconv_shape = qwen_conv3d.tensor(&qconv_name).unwrap().shape.clone();
    reshape_tensor(
        &mut qwen_conv3d,
        &qconv_name,
        &[qconv_shape[0], 1, qconv_shape[1]],
    );
    assert!(inspect_target(&qwen_conv3d).is_ok());

    let qwen_proj = headers::projector("qwen3.5-4b:gguf:q4");
    let mut qwen_batched = qwen_proj.clone();
    let qpos_shape = qwen_batched
        .tensor("v.position_embd.weight")
        .unwrap()
        .shape
        .clone();
    reshape_tensor(
        &mut qwen_batched,
        "v.position_embd.weight",
        &[1, qpos_shape[0], qpos_shape[1]],
    );
    assert!(inspect_with_projector(&qwen, &qwen_batched).is_ok());
}
