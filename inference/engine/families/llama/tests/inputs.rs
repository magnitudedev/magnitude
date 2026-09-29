use magnitude_artifacts::{
    media::{DType, PreparedMedia, PreparedTensor},
    ArtifactIdentity, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_contracts::{
    FamilyError, FamilyId, InputPreparationError, MarkerTokens, ModelDefinition, ModelFamily,
    TokenPlan,
};
use magnitude_family_common::headers;
use magnitude_family_llama::{inspect_components, LlamaFamily};

struct NoMarkers;

impl MarkerTokens for NoMarkers {
    fn marker(&self, text: &str) -> Result<TokenId, FamilyError> {
        Err(FamilyError(format!("no marker {text:?}")))
    }
}

fn model() -> ModelDefinition {
    inspect_components(
        &headers::directory("minicpm5-2b__target-gguf_q4.json"),
        None,
        PackageIdentity {
            target: ArtifactIdentity([1; 32]),
            projector: None,
        },
    )
    .unwrap()
}

fn plan(count: u32) -> TokenPlan {
    TokenPlan::new(
        (0..count).map(TokenId).collect(),
        InputLayout::new(count as usize, Vec::new()).unwrap(),
    )
    .unwrap()
}

#[test]
fn text_rows_sit_at_their_absolute_positions() {
    let model = model();
    let adapter = LlamaFamily.input_adapter(&model, &NoMarkers).unwrap();
    let input = adapter.prepare(&model, plan(4), &[]).unwrap();
    assert_eq!(input.coordinates(), &[[0; 3], [1; 3], [2; 3], [3; 3]]);
    assert_eq!(input.continuation(), 4);
    assert_eq!(input.coordinates_at(4, 2).unwrap(), [[4; 3], [5; 3]]);
}

#[test]
fn media_and_foreign_definitions_are_rejected() {
    let model = model();
    let adapter = LlamaFamily.input_adapter(&model, &NoMarkers).unwrap();
    let media = PreparedMedia::new(
        "0".repeat(64),
        vec![PreparedTensor::new("pixel_values".into(), DType::F32, vec![1], vec![0; 4]).unwrap()],
    )
    .unwrap();
    assert_eq!(
        adapter.prepare(&model, plan(2), &[media]).unwrap_err(),
        InputPreparationError::UnsupportedMedia
    );
    let foreign = ModelDefinition {
        family: FamilyId("qwen35".into()),
        ..model
    };
    assert_eq!(
        adapter.prepare(&foreign, plan(2), &[]).unwrap_err(),
        InputPreparationError::InputAlignment
    );
}
