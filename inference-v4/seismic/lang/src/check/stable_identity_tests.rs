use crate::checked::{check_source, SourceFile, SourceSet};

#[test]
fn function_identity_delimits_path_from_source_text() {
    let first = "fn first() -> f32:\n    return 1.0\n\n";
    let second = "fn second() -> f32:\n    return 2.0\n";
    let combined = check_source(SourceSet::new(vec![SourceFile {
        path: "a".into(),
        text: format!("{first}{second}"),
    }]))
    .unwrap();
    let shifted = check_source(SourceSet::new(vec![SourceFile {
        path: format!("a{first}"),
        text: second.into(),
    }]))
    .unwrap();

    assert_eq!(combined.internal().definitions[0].name, "first");
    assert_eq!(shifted.internal().definitions[0].name, "second");
    assert_ne!(
        combined.internal().definitions[0].stable,
        shifted.internal().definitions[0].stable,
        "different checked functions must not alias when a source prefix moves into the path"
    );
}

#[test]
fn function_identity_includes_other_source_files_that_define_callees() {
    fn checked(helper_result: &str) -> crate::checked::CheckedModule {
        check_source(SourceSet::new(vec![
            SourceFile {
                path: "a-main.seismic".into(),
                text: "fn probe() -> f32:\n    return helper()\n".into(),
            },
            SourceFile {
                path: "z-helper.seismic".into(),
                text: format!("fn helper() -> f32:\n    return {helper_result}\n"),
            },
        ]))
        .unwrap()
    }
    let first = checked("1.0");
    let second = checked("2.0");
    assert_eq!(first.internal().definitions[0].name, "probe");
    assert_eq!(second.internal().definitions[0].name, "probe");
    assert_ne!(
        first.internal().definitions[0].stable,
        second.internal().definitions[0].stable,
        "a caller must not retain its source label when a checked callee changes"
    );
}

#[test]
fn lowered_function_retains_source_ordinal_across_independent_checks() {
    use crate::entry::ElementBindings;
    let source = SourceSet::new(vec![SourceFile {
        path: "definitions.seismic".into(),
        text: "fn helper(x: f32) -> f32:\n    return x + 1.0\n\nfn probe(x: f32) -> f32:\n    return helper(x)\n".into(),
    }]);
    let first = check_source(source.clone()).unwrap();
    let second = check_source(source).unwrap();
    let ordinal_map = |module: &crate::checked::CheckedModule| {
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        assert_eq!(entry.program().sources(), module.sources());
        entry
            .program()
            .functions()
            .map(|(_, function)| (function.name().to_owned(), function.source_definition()))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let expected =
        std::collections::BTreeMap::from([("helper".to_owned(), 0), ("probe".to_owned(), 1)]);
    assert_eq!(ordinal_map(&first), expected);
    assert_eq!(ordinal_map(&second), expected);
}

#[test]
fn exact_program_subject_distinguishes_element_instantiations() {
    use crate::entry::ElementBindings;
    use crate::types::DType;
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "polymorphic.seismic".into(),
        text: "fn probe(x: &tensor[2] T) -> tensor[2] T:\n    return load(x)\n".into(),
    }]))
    .unwrap();
    let id = module.entry_named("probe").unwrap();
    let f32_entry = module
        .entry(
            id,
            &ElementBindings::new().bind("T", crate::registry::dense(DType::F32)),
        )
        .unwrap();
    let i32_entry = module
        .entry(
            id,
            &ElementBindings::new().bind("T", crate::registry::dense(DType::I32)),
        )
        .unwrap();
    assert_eq!(f32_entry.program().sources(), i32_entry.program().sources());
    assert_ne!(f32_entry.program().subject(), i32_entry.program().subject());
}
