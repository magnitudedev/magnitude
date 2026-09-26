#[test]
fn direct_dependencies_preserve_the_generic_chat_boundary() {
    let manifest = include_str!("../Cargo.toml");
    for dependency in [
        "seismic",
        "magnitude-engine",
        "magnitude-scheduler",
        "magnitude-executor",
        "magnitude-family-qwen35",
        "magnitude-family-contracts",
        "magnitude-kernels",
        "magnitude-state",
        "magnitude-batching",
    ] {
        assert!(
            !manifest.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with(dependency) && line[dependency.len()..].starts_with([' ', '='])
            }),
            "chat must not depend directly on {dependency}"
        );
    }
}

#[test]
fn sources_do_not_import_model_or_execution_implementations() {
    let sources = [
        include_str!("../src/lib.rs"),
        include_str!("../src/artifacts.rs"),
        include_str!("../src/error.rs"),
        include_str!("../src/generation.rs"),
        include_str!("../src/output.rs"),
        include_str!("../src/preparation.rs"),
        include_str!("../src/reasoning.rs"),
        include_str!("../src/request.rs"),
        include_str!("../src/stream.rs"),
        include_str!("../src/templates.rs"),
        include_str!("../src/tokenizer.rs"),
    ]
    .join("\n");
    for forbidden in [
        "seismic",
        "magnitude_engine",
        "magnitude_scheduler",
        "magnitude_family_qwen35",
        "magnitude_executor",
        "crate::models",
    ] {
        assert!(
            !sources.contains(forbidden),
            "chat source imported {forbidden}"
        );
    }
}

#[test]
fn artifact_adapters_consume_package_payloads_instead_of_raw_directories() {
    let source = include_str!("../src/artifacts.rs");
    assert!(source.contains("TokenizerPayload"));
    assert!(source.contains("TemplatePayload"));
    assert!(!source.contains("Directory"));
    assert!(!source.contains("qwen35_gguf"));
}
