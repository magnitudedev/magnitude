//! Layers that reuse an earlier layer's history never compute keys or values.
//! Converters that keep the checkpoint's projections for those layers (for
//! example bartowski's Gemma 4 E2B quantizations) store `attn_k`, `attn_v`
//! and `attn_k_norm` there; the family checks and discards them, so such a
//! file interprets to the same definition as one without them.

use magnitude_artifacts::{gguf::Directory, ArtifactIdentity, PackageIdentity, UnsupportedReason};
use magnitude_family_common::headers::{add_tensor, artifact, catalog as header};
use magnitude_family_contracts::{KeyValue, ModelDefinition, Operator};
use magnitude_family_gemma4::{inspect_components, Error};

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

fn inspect(directory: &Directory) -> Result<ModelDefinition, Error> {
    inspect_components(directory, None, identity())
}

fn catalog(model: &str) -> Directory {
    header(&format!("gemma-4-{model}-it-qat__target-gguf:q4"))
}

/// `(layer, key/value rows, head width)` of every sharing layer.
fn sharing_layers(definition: &ModelDefinition) -> Vec<(usize, u64, u64)> {
    definition
        .decoder
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(layer, block)| match &block.sublayers[0].op {
            Operator::Attention(attention)
                if matches!(attention.key_value, KeyValue::Shared { .. }) =>
            {
                Some((layer, attention.kv_heads * attention.width, attention.width))
            }
            _ => None,
        })
        .collect()
}

fn with_sharing_projections(model: &str, hidden: u64) -> (Directory, ModelDefinition) {
    let plain = catalog(model);
    let definition = inspect(&plain).unwrap();
    let mut directory = plain;
    for (layer, rows, width) in sharing_layers(&definition) {
        add_tensor(
            &mut directory,
            &format!("blk.{layer}.attn_k.weight"),
            &[rows, hidden],
        );
        add_tensor(
            &mut directory,
            &format!("blk.{layer}.attn_v.weight"),
            &[rows, hidden],
        );
        add_tensor(
            &mut directory,
            &format!("blk.{layer}.attn_k_norm.weight"),
            &[width],
        );
    }
    (directory, definition)
}

#[test]
fn sharing_layer_projections_are_accepted_and_never_imported() {
    for (model, hidden, sharing) in [("e2b", 1536, 20), ("e4b", 2560, 18)] {
        let (directory, plain) = with_sharing_projections(model, hidden);
        assert_eq!(sharing_layers(&plain).len(), sharing, "{model}");
        assert_eq!(inspect(&directory).unwrap(), plain, "{model}");
    }
}

#[test]
fn a_sharing_layer_projection_of_another_geometry_is_unrecognized() {
    let (mut directory, plain) = with_sharing_projections("e2b", 1536);
    let (layer, rows, _) = sharing_layers(&plain)[0];
    let name = format!("blk.{layer}.attn_v.weight");
    directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap()
        .shape = vec![rows * 2, 1536];
    assert_eq!(
        inspect(&directory).unwrap_err().reasons,
        [UnsupportedReason::UnrecognizedTensor { name }]
    );
}

#[test]
fn an_owning_layer_still_requires_its_key_projection() {
    let mut directory = catalog("e2b");
    directory
        .tensors
        .retain(|tensor| tensor.name != "blk.14.attn_k.weight");
    assert_eq!(
        inspect(&directory).unwrap_err().to_string(),
        "missing weight \"blk.14.attn_k.weight\""
    );
}

/// The released bartowski `google_gemma-4-E2B-it-Q4_K_M.gguf` header
/// (repository revision 81012ba3538e061d5ee003f11f25335b17f82e2d) carries the
/// 60 sharing-layer tensors of layers 15..=34 the catalog's file omits, and
/// otherwise the same tensor names and Gemma 4 metadata.
#[test]
fn the_released_bartowski_e2b_header_interprets_like_the_catalog_file() {
    let bartowski = artifact("gemma-4-e2b-it-bartowski__target-gguf:q4_k_m");
    let catalog = catalog("e2b");
    let extra = bartowski
        .tensors
        .iter()
        .filter(|tensor| catalog.tensor(&tensor.name).is_none())
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    let expected = (15..35)
        .flat_map(|layer| {
            ["attn_k", "attn_k_norm", "attn_v"].map(|role| format!("blk.{layer}.{role}.weight"))
        })
        .collect::<Vec<_>>();
    assert_eq!(extra, expected);
    assert_eq!(inspect(&bartowski).unwrap(), inspect(&catalog).unwrap());
}
