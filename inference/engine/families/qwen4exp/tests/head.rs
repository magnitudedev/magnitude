use magnitude_artifacts::{ArtifactIdentity, PackageIdentity};
use magnitude_family_common::headers;
use magnitude_family_contracts::{
    HeadChain, HeadComponent, HeadHistory, HistoryReads, Operator, ResidualForm,
};

fn target() -> magnitude_family_contracts::ModelDefinition {
    magnitude_family_qwen4exp::inspect_components(
        &headers::directory(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/target-q2.json"
        )),
        None,
        PackageIdentity {
            target: ArtifactIdentity([7; 32]),
            projector: None,
        },
    )
    .unwrap()
}

#[test]
fn shared_prediction_package_retains_raw_streams_and_owns_causal_history() {
    let mut target = target();
    let head = magnitude_family_qwen4exp::inspect_head(
        &headers::directory(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/mtp-q8.json"
        )),
        &target,
    )
    .unwrap();
    assert_eq!(head.component, HeadComponent::Draft);
    assert_eq!(head.history, HeadHistory::Own);
    assert!(head.residual_conditioning());
    assert_eq!(head.conditioning_width(&target.decoder), 10240);
    assert_eq!(head.blocks[0].chain, HeadChain::Residual);
    assert_eq!(head.blocks[0].residual, target.decoder.residual);
    let Operator::Attention(attention) = &head.blocks[0].blocks[0].sublayers[0].op else {
        panic!("prediction must attend")
    };
    assert_eq!(attention.reads, HistoryReads::Visible);
    target.head = Some(head);
    target.validate().unwrap();
    assert!(target.deferred_forms().is_empty());
}

#[test]
fn prediction_package_rejects_mismatched_target_and_missing_weights() {
    let mut target = target();
    let mut draft = headers::directory(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mtp-q8.json"
    ));
    target.decoder.residual = ResidualForm::Single;
    assert!(magnitude_family_qwen4exp::inspect_head(&draft, &target).is_err());
    let target = self::target();
    draft
        .tensors
        .retain(|tensor| !tensor.name.contains("nextn.hnorm"));
    assert!(magnitude_family_qwen4exp::inspect_head(&draft, &target).is_err());
}
