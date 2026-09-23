use std::{fs, path::PathBuf};

#[test]
fn manifest_keeps_the_executor_below_model_and_product_layers() {
    let manifest =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    for forbidden in [
        "magnitude-chat",
        "magnitude-generation",
        "magnitude-model-qwen35",
        "magnitude-service",
        "magnitude-templates",
    ] {
        assert!(
            !manifest.contains(forbidden),
            "model-executor must not depend on {forbidden}"
        );
    }
    assert!(manifest.contains("magnitude-model-batching"));
    assert!(manifest.contains("magnitude-artifacts"));
    assert!(manifest.contains("magnitude-model-contracts"));
    assert!(manifest.contains("magnitude-model-kernels"));
}
