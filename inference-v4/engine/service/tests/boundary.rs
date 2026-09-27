#[test]
fn direct_dependencies_preserve_the_service_boundary() {
    let manifest = include_str!("../Cargo.toml");
    let forbidden = [
        "seismic",
        "magnitude-artifacts",
        "magnitude-chat",
        "magnitude-templates",
        "magnitude-model-qwen35",
    ];
    for dependency in forbidden {
        assert!(
            !manifest.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with(dependency)
                    && line[dependency.len()..]
                        .starts_with(|character: char| character == ' ' || character == '=')
            }),
            "service must not depend on {dependency}"
        );
    }
}

#[test]
fn service_sources_do_not_import_forbidden_implementation_layers() {
    let sources = [
        include_str!("../src/lib.rs"),
        include_str!("../src/domain.rs"),
        include_str!("../src/owner.rs"),
        include_str!("../src/policy.rs"),
        include_str!("../src/round_driver.rs"),
        include_str!("../src/retention.rs"),
        include_str!("../src/worker.rs"),
    ]
    .join("\n");
    for forbidden in [
        "seismic",
        "magnitude_artifacts",
        "magnitude_model_qwen35",
        "crate::chat",
        "crate::templates",
    ] {
        assert!(
            !sources.contains(forbidden),
            "service source imported {forbidden}"
        );
    }
}
