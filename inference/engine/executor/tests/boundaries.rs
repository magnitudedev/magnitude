use std::process::Command;

#[test]
fn manifest_keeps_the_executor_below_model_and_product_layers() {
    // Cargo resolves dependency tables, target conditions and renamed packages for us.
    // Model families used only by numerical fixtures do not create a production edge.
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO when it runs tests");
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .expect("cargo sets CARGO_MANIFEST_DIR when it runs tests");
    let output = Command::new(cargo)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
            "--locked",
        ])
        .current_dir(manifest_dir)
        .output()
        .expect("read Cargo dependency metadata");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let package = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["name"] == env!("CARGO_PKG_NAME"))
        .unwrap();
    let dependencies: Vec<_> = package["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|dependency| dependency["kind"] != "dev")
        .map(|dependency| dependency["name"].as_str().unwrap())
        .collect();
    for forbidden in [
        "magnitude-chat",
        "magnitude-generation",
        "magnitude-family-qwen35",
        "magnitude-scheduler",
        "magnitude-templates",
    ] {
        assert!(
            !dependencies.contains(&forbidden),
            "executor must not depend on {forbidden}"
        );
    }
    assert!(dependencies.contains(&"magnitude-batching"));
    assert!(dependencies.contains(&"magnitude-artifacts"));
    assert!(dependencies.contains(&"magnitude-family-contracts"));
    assert!(dependencies.contains(&"magnitude-kernels"));
}
