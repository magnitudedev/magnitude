use std::{fs, path::PathBuf};

#[test]
fn manifest_keeps_the_executor_below_model_and_product_layers() {
    let manifest =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    for forbidden in [
        "magnitude-chat",
        "magnitude-generation",
        "magnitude-family-qwen35",
        "magnitude-scheduler",
        "magnitude-templates",
    ] {
        assert!(
            !manifest.contains(forbidden),
            "executor must not depend on {forbidden}"
        );
    }
    assert!(manifest.contains("magnitude-batching"));
    assert!(manifest.contains("magnitude-artifacts"));
    assert!(manifest.contains("magnitude-family-contracts"));
    assert!(manifest.contains("magnitude-kernels"));
}
