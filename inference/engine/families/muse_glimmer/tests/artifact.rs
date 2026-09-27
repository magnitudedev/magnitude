//! Interpretation of a real downloaded Muse Glimmer GGUF header.
//! Run on a host holding the model:
//! `MUSE_GLIMMER_GGUF=<file> cargo test -p magnitude-family-muse-glimmer --test artifact -- --ignored`

use magnitude_artifacts::PackageHeaders;
use magnitude_family_muse_glimmer::inspect_components;

#[test]
#[ignore = "needs MUSE_GLIMMER_GGUF, a downloaded Muse Glimmer GGUF"]
fn a_downloaded_artifact_binds_every_tensor() {
    let path = std::env::var("MUSE_GLIMMER_GGUF").expect("MUSE_GLIMMER_GGUF names a Muse Glimmer GGUF");
    let headers = PackageHeaders::open(&path, None).unwrap();
    let model = inspect_components(headers.target(), None, headers.identity()).unwrap();
    assert_eq!(model.decoder.blocks.len(), 52);
}
