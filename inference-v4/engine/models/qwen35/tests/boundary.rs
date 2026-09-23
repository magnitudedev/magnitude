#[test]
fn direct_dependencies_preserve_the_family_adapter_boundary() {
    let manifest = include_str!("../Cargo.toml");
    let forbidden = [
        "seismic",
        "magnitude-model-executor",
        "magnitude-model-state",
        "magnitude-model-batching",
        "magnitude-generation",
        "magnitude-service",
        "magnitude-chat",
        "magnitude-templates",
    ];
    for dependency in forbidden {
        assert!(
            !manifest.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with(dependency) && line[dependency.len()..].starts_with([' ', '='])
            }),
            "Qwen family adapter must not depend on {dependency}"
        );
    }
}

#[test]
fn adapter_sources_do_not_import_engine_implementation_layers() {
    let source = include_str!("../src/lib.rs");
    for implementation in [
        "seismic",
        "model_executor",
        "model_state",
        "model_batching",
        "generation",
        "service",
        "chat",
        "templates",
    ] {
        assert!(
            !source.contains(implementation),
            "Qwen family adapter source imported {implementation}"
        );
    }

    let all_sources = [source, include_str!("../src/inputs.rs")].join("\n");
    for generic_payload in ["tokenizer.ggml", "tokenizer.chat_template"] {
        assert!(
            !all_sources.contains(generic_payload),
            "Qwen numerical interpretation must not parse generic payload {generic_payload}"
        );
    }
}
