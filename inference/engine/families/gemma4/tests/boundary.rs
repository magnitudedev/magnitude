#[test]
fn direct_dependencies_preserve_the_family_adapter_boundary() {
    let manifest = include_str!("../Cargo.toml");
    let forbidden = [
        "seismic",
        "magnitude-executor",
        "magnitude-state",
        "magnitude-batching",
        "magnitude-generation",
        "magnitude-scheduler",
        "magnitude-chat",
        "magnitude-templates",
    ];
    for dependency in forbidden {
        assert!(
            !manifest.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with(dependency) && line[dependency.len()..].starts_with([' ', '='])
            }),
            "Gemma family adapter must not depend on {dependency}"
        );
    }
}

#[test]
fn adapter_sources_do_not_import_engine_implementation_layers() {
    let sources = [
        include_str!("../src/lib.rs"),
        include_str!("../src/family.rs"),
        include_str!("../src/inputs.rs"),
    ]
    .join("\n");
    for implementation in [
        "seismic",
        "magnitude_executor",
        "magnitude_state",
        "magnitude_batching",
        "generation",
        "scheduler",
        "chat",
        "templates",
        // Tokenizer payloads are generic; numerical interpretation never
        // parses them.
        "tokenizer.ggml",
    ] {
        assert!(
            !sources.contains(implementation),
            "Gemma family adapter source mentions {implementation}"
        );
    }
}
