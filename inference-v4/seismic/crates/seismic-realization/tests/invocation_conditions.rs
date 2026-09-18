use seismic_lang::{
    Scope,
    lowered_ir::AliasRequirement,
    program::{SourceFile, compile},
};
use seismic_realization::{InvocationConditions, storage};

fn conditions(
    left: &str,
    right: &str,
    exact_allowed: bool,
) -> (InvocationConditions, Vec<seismic_realization::BufferSpec>) {
    let program = compile(
        &[SourceFile {
            path: "bindings.seismic.portable".into(),
            scope: Scope::Portable,
            text: format!("fn evaluate(x: {left}, y: {right}):\n  z = 1\n"),
        }],
        &[],
    )
    .unwrap();
    let mut function =
        seismic_lang::lower::lower(&program, "evaluate", "cpu", &Default::default()).unwrap();
    function.alias_requirements.push(AliasRequirement {
        left: 0,
        right: 1,
        exact_allowed,
    });
    let specs = storage::parameters(&function).unwrap().0;
    (
        InvocationConditions::from_lowered(&function).unwrap(),
        specs,
    )
}

#[test]
fn source_alias_requirements_validate_used_byte_ranges() {
    let (conditions, specs) = conditions("tensor[4] f32", "tensor[4] f32", true);
    for offsets in [[0, 0], [0, 16], [16, 0]] {
        conditions
            .validate_aliases(&specs, |i| (1, offsets[i]))
            .unwrap();
    }
    assert!(
        conditions
            .validate_aliases(&specs, |i| (1, [0, 4][i]))
            .unwrap_err()
            .contains("overlapping")
    );
    conditions
        .validate_aliases(&specs, |i| (i as u64, 0))
        .unwrap();
    let (conditions, specs) = self::conditions("tensor[4] f32", "tensor[4] f32", false);
    assert!(conditions.validate_aliases(&specs, |_| (1, 0)).is_err());
    assert!(
        conditions
            .validate_aliases(&specs, |_| (1, u64::MAX))
            .is_err()
    );
}

#[test]
fn packed_alias_requirements_cover_every_storage_plane() {
    let (conditions, specs) = conditions("tensor[64] q4g64", "tensor[64] q4g64", true);
    assert_eq!(specs.len(), 6);
    assert_eq!(conditions.alias_pairs().len(), 9);
    assert_eq!(
        conditions
            .alias_pairs()
            .iter()
            .filter(|(_, _, exact)| *exact)
            .count(),
        3
    );
    conditions
        .validate_aliases(&specs, |i| ((i % 3) as u64, 0))
        .unwrap();
    assert!(
        conditions.validate_aliases(&specs, |_| (0, 0)).is_err(),
        "overlapping distinct representation planes are not exact aliases"
    );
}
