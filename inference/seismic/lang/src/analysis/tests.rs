use super::*;
use crate::checked::{CheckedModule, SourceFile, SourceSet, check_source};
use crate::entry::ElementBindings;

pub(super) fn entry(source: &str) -> LogicalEntry {
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "analysis.seismic".into(),
        text: source.into(),
    }]))
    .unwrap();
    module
        .entry(
            module.entry_named("probe").unwrap(),
            &ElementBindings::new(),
        )
        .unwrap()
}

/// The engine's kernel sources, checked once.
pub(super) fn kernels() -> &'static CheckedModule {
    static KERNELS: std::sync::OnceLock<CheckedModule> = std::sync::OnceLock::new();
    KERNELS.get_or_init(|| {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let roots = [
            manifest.join("../../seismic-std/lib"),
            manifest.join("../../engine/kernels/kernels"),
        ];
        crate::source::load(&roots, SourceSet::default()).unwrap().module
    })
}

/// Kernel entry `name` with each element parameter bound to the named
/// representation.
fn kernel(name: &str, elements: &[(&str, &str)]) -> LogicalEntry {
    let module = kernels();
    let bindings = elements.iter().fold(ElementBindings::new(), |bindings, (parameter, representation)| {
        bindings.bind(parameter, registry::representation(representation).unwrap())
    });
    module.entry(module.entry_named(name).unwrap(), &bindings).unwrap()
}

/// Every tensor parameter with the kind of its stored elements.
fn kinds(analysis: &EntryAnalysis) -> Vec<(&str, &'static str)> {
    analysis
        .parameters
        .iter()
        .map(|parameter| {
            let kind = match &parameter.stored {
                Stored::Float { dtype } => dtype.name(),
                Stored::Packed { .. } => "packed",
                Stored::Codes { .. } => "codes",
                Stored::Discrete => "discrete",
                Stored::Converted => "converted",
                Stored::External => "external",
            };
            (parameter.name.as_str(), kind)
        })
        .collect()
}

fn stored<'a>(analysis: &'a EntryAnalysis, name: &str) -> &'a Stored {
    &analysis.parameters.iter().find(|parameter| parameter.name == name).unwrap().stored
}

/// The planes holding codes of each grid of `parameter`, in order.
fn coded(analysis: &EntryAnalysis, parameter: &str) -> Vec<Option<(PlaneField, u32)>> {
    analysis
        .grids
        .iter()
        .filter(|grid| grid.parameter == parameter)
        .map(|grid| grid.plane)
        .collect()
}

fn integers(min: i64, max: i64) -> Lattice {
    Lattice::Codes(CodeLevels::Integers { min, max })
}

fn plane(field: PlaneField, group: u32, fields: u32, lattice: Lattice) -> Plane {
    Plane {
        field,
        group,
        fields,
        lattice,
    }
}

fn q4k_planes() -> Vec<Plane> {
    vec![
        plane(PlaneField::Words, 1, 1, integers(0, 15)),
        plane(PlaneField::Coefficients, 32, 2, integers(0, 63)),
        plane(PlaneField::ScaleFactor, 256, 1, Lattice::Float(DType::F16)),
        plane(PlaneField::BiasFactor, 256, 1, Lattice::Float(DType::F16)),
    ]
}

const Q4K_CODES: [Option<(PlaneField, u32)>; 3] = [
    Some((PlaneField::Words, 0)),
    Some((PlaneField::Coefficients, 0)),
    Some((PlaneField::Coefficients, 1)),
];

/// The value of plane entry `(plane, field)` at element zero, for evaluating
/// a packed grid.
fn planes(values: &'static [(PlaneField, u32, f64)]) -> impl Fn(&Variable) -> Option<f64> {
    |variable| match variable {
        Variable::Binder(_) => Some(0.0),
        Variable::Plane { plane, field, .. } => values
            .iter()
            .find(|(known, entry, _)| known == plane && entry == field)
            .map(|(_, _, value)| *value),
        _ => None,
    }
}

#[test]
fn dense_expand_weights_are_packed_planes_and_its_row_table_is_discrete() {
    let entry = kernel("dense_expand", &[("NW", "f32"), ("GW", "q4k"), ("UW", "q8g32s"), ("A", "bf16")]);
    let analysis = analyze(&entry);
    assert_eq!(
        kinds(&analysis),
        [
            ("residual", "f32"),
            ("norm", "f32"),
            ("gate_weight", "packed"),
            ("up_weight", "packed"),
            ("out_rows", "discrete"),
            ("gate_scale", "f32"),
            ("up_scale", "f32"),
            ("gate_bound", "f32"),
            ("up_bound", "f32"),
        ]
    );
    assert_eq!(
        stored(&analysis, "gate_weight"),
        &Stored::Packed {
            planes: q4k_planes()
        }
    );
    assert_eq!(
        stored(&analysis, "up_weight"),
        &Stored::Packed {
            planes: vec![
                plane(PlaneField::Words, 1, 1, integers(-128, 127)),
                plane(PlaneField::Scale, 32, 1, Lattice::Float(DType::F16)),
            ]
        }
    );
    assert_eq!(coded(&analysis, "gate_weight"), Q4K_CODES);
    assert_eq!(coded(&analysis, "up_weight"), [Some((PlaneField::Words, 0))]);
    assert_eq!(analysis.grids.len(), 4);

    // q4k decodes scale_factor * scale_code * code - bias_factor * bias_code:
    // every code's step is its own slope, and the bias never enters a slope.
    let point = planes(&[
        (PlaneField::Words, 0, 7.0),
        (PlaneField::Coefficients, 0, 3.0),
        (PlaneField::Coefficients, 1, 63.0),
        (PlaneField::ScaleFactor, 0, -0.5),
        (PlaneField::BiasFactor, 0, 0.25),
    ]);
    let steps = analysis.grids[..3].iter().map(|grid| grid.steps(&point).unwrap()).collect::<Vec<_>>();
    assert_eq!(steps[0], Steps { below: Some(1.5), above: Some(1.5) });
    assert_eq!(steps[1], Steps { below: Some(3.5), above: Some(3.5) });
    assert_eq!(steps[2], Steps { below: Some(0.25), above: None });
    for grid in &analysis.grids {
        assert_eq!(grid.coordinates.len(), 2);
        assert_eq!(grid.index.iter().map(Term::to_string).collect::<Vec<_>>(), ["i0", "i1"]);
        assert_eq!(grid.coordinates[0].end.to_string(), "F");
        assert_eq!(grid.coordinates[1].end.to_string(), "H");
    }
}

#[test]
fn attention_project_classifies_each_weight_by_its_own_representation() {
    let entry = kernel(
        "attention_project",
        &[("NW", "f32"), ("QW", "q4k"), ("GW", "q8g32s"), ("KW", "iq4g32"), ("VW", "bf16"), ("A", "bf16")],
    );
    let analysis = analyze(&entry);
    assert_eq!(
        kinds(&analysis),
        [
            ("hidden", "f32"),
            ("input_norm", "f32"),
            ("query_weight", "packed"),
            ("gate_weight", "packed"),
            ("key_weight", "packed"),
            ("value_weight", "bf16"),
            ("query_bound", "f32"),
            ("gate_bound", "f32"),
            ("key_bound", "f32"),
            ("value_bound", "f32"),
        ]
    );
    assert_eq!(coded(&analysis, "query_weight"), Q4K_CODES);
    assert_eq!(coded(&analysis, "gate_weight"), [Some((PlaneField::Words, 0))]);
    assert_eq!(coded(&analysis, "key_weight"), [Some((PlaneField::Words, 0))]);
    assert_eq!(analysis.grids.len(), 5);

    // The IQ4 codebook is not uniform: the steps to the two adjacent levels
    // of a code differ, and each is reported.
    let table = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    let levels = CodeLevels::Table(table.map(f64::from).to_vec());
    assert_eq!(
        stored(&analysis, "key_weight"),
        &Stored::Packed {
            planes: vec![
                plane(PlaneField::Words, 1, 1, Lattice::Codes(levels.clone())),
                plane(PlaneField::Scale, 32, 1, Lattice::Float(DType::F32)),
            ]
        }
    );
    let grid = analysis.grids.iter().find(|grid| grid.parameter == "key_weight").unwrap();
    assert_eq!(grid.levels, levels);
    let at = |code: f64| {
        grid.steps(&|variable| match variable {
            Variable::Binder(_) => Some(0.0),
            Variable::Plane { plane: PlaneField::Words, .. } => Some(code),
            Variable::Plane { plane: PlaneField::Scale, .. } => Some(-0.5),
            _ => None,
        })
        .unwrap()
    };
    assert_eq!(at(1.0), Steps { below: Some(5.5), above: Some(6.0) });
    assert_eq!(at(-127.0), Steps { below: None, above: Some(11.5) });
    assert_eq!(at(113.0), Steps { below: Some(12.0), above: None });
}

#[test]
fn dense_output_has_one_packed_weight_and_a_discrete_row_table() {
    let entry = kernel("dense_output", &[("A", "bf16"), ("DW", "q4k")]);
    let analysis = analyze(&entry);
    assert_eq!(
        kinds(&analysis),
        [
            ("residual", "f32"),
            ("product", "bf16"),
            ("down_weight", "packed"),
            ("out_rows", "discrete"),
            ("down_scale", "f32"),
            ("down_bound", "f32"),
        ]
    );
    assert_eq!(
        stored(&analysis, "down_weight"),
        &Stored::Packed {
            planes: q4k_planes()
        }
    );
    assert_eq!(coded(&analysis, "down_weight"), Q4K_CODES);
    assert_eq!(analysis.grids.len(), 3);
}

/// The stored kinds of the affine K8/V4 attention entries, and the one grid
/// of each code plane: 8-bit key codes four to a word, 4-bit value codes
/// eight to a word, each scaled by the coefficient pair of its 32-value
/// group.
fn assert_affine_history(analysis: &EntryAnalysis) {
    for (name, kind) in [
        ("query", "bf16"),
        ("key", "bf16"),
        ("query_norm", "f32"),
        ("rotary_components", "discrete"),
        ("rotary_frequencies", "f32"),
        ("coordinates", "converted"),
        ("visible", "discrete"),
        ("fresh", "discrete"),
        ("destinations", "discrete"),
        ("history_key_codes", "codes"),
        ("history_key_coefficients", "f16"),
        ("history_value_codes", "codes"),
        ("history_value_coefficients", "f16"),
    ] {
        assert!(kinds(analysis).contains(&(name, kind)), "{name} is not {kind}");
    }
    assert_eq!(analysis.grids.len(), 2);
    for (codes, coefficients, per_word, bits, maximum) in [
        ("history_key_codes", "history_key_coefficients", 4, 8, 255),
        ("history_value_codes", "history_value_coefficients", 8, 4, 15),
    ] {
        let levels = CodeLevels::Integers { min: 0, max: maximum };
        assert_eq!(
            stored(analysis, codes),
            &Stored::Codes {
                levels: levels.clone(),
                coefficients: vec![coefficients.to_owned()],
            }
        );
        assert_eq!(coded(analysis, codes), [None]);
        let grid = analysis.grids.iter().find(|grid| grid.parameter == codes).unwrap();
        assert_eq!(grid.levels, levels);
        // i0 is the value within the head, i1 the history row, i3 the head.
        assert_eq!(
            grid.index.iter().map(Term::to_string).collect::<Vec<_>>(),
            ["i1".to_owned(), "i3".to_owned(), format!("(i0 div {per_word})")]
        );
        assert_eq!(
            grid.code.to_string(),
            format!("(({codes}[i1, i3, (i0 div {per_word})] >> ((i0 mod {per_word}) * {bits})) & {maximum})")
        );
        assert_eq!(grid.slope.to_string(), format!("Abs({coefficients}[i1, i3, (2 * (i0 div 32))])"));
        assert_eq!(grid.coordinates.len(), 5);
        assert_eq!(grid.coordinates[1].start.to_string(), "visible[i4, i2, 0]");
        assert_eq!(grid.coordinates[1].end.to_string(), "visible[i4, i2, 1]");

        // Value 37 of head 2 of history row 5: group 1 of the head.
        let steps = |value: f64, word: u32| {
            grid.steps(&|variable| match variable {
                Variable::Binder(0) => Some(value),
                Variable::Binder(1) => Some(5.0),
                Variable::Binder(3) => Some(2.0),
                Variable::Element { parameter, index } => {
                    let index = index.iter().map(|term| term.as_constant().unwrap()).collect::<Vec<_>>();
                    if parameter == codes && index == [5.0, 2.0, f64::from(37 / per_word)] {
                        Some(f64::from(word))
                    } else if parameter == coefficients && index == [5.0, 2.0, 2.0] {
                        Some(-0.25)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .unwrap()
        };
        let shift = (37 % per_word) * bits;
        let word = |code: u32| (code << shift) | !((maximum as u32) << shift);
        assert_eq!(steps(37.0, word(3)), Steps { below: Some(0.25), above: Some(0.25) });
        assert_eq!(steps(37.0, word(0)), Steps { below: None, above: Some(0.25) });
        assert_eq!(steps(37.0, word(maximum as u32)), Steps { below: Some(0.25), above: None });
    }
}

#[test]
fn attention_decode_k8v4_has_one_grid_per_code_plane() {
    let analysis = analyze(&kernel("attention_decode_k8v4", &[("A", "bf16")]));
    assert_eq!(analysis.parameters.len(), 18);
    assert_affine_history(&analysis);
}

#[test]
fn attention_prefill_k8v4_decodes_history_it_has_already_appended_to() {
    let analysis = analyze(&kernel("attention_prefill_k8v4", &[("A", "bf16")]));
    // The WorkList bitmap and its block keys' extent-only table among them.
    assert_eq!(analysis.parameters.len(), 21);
    assert_eq!(stored(&analysis, "history_tiles"), &Stored::Discrete);
    assert_affine_history(&analysis);
}

#[test]
fn dense_attention_history_is_floating_and_has_no_grid() {
    let analysis = analyze(&kernel("attention_decode", &[("A", "bf16")]));
    assert_eq!(
        kinds(&analysis),
        [
            ("query", "bf16"),
            ("gate", "bf16"),
            ("key", "bf16"),
            ("value", "bf16"),
            ("query_norm", "f32"),
            ("key_norm", "f32"),
            ("value_norm", "f32"),
            ("rotary_components", "discrete"),
            ("rotary_frequencies", "f32"),
            ("rotary_amplitudes", "f32"),
            ("coordinates", "converted"),
            ("visible", "discrete"),
            ("fresh", "discrete"),
            ("destinations", "discrete"),
            ("history_key", "bf16"),
            ("history_value", "bf16"),
        ]
    );
    assert!(analysis.grids.is_empty());
    assert!(!analysis.reductions.is_empty());
}

#[test]
fn gated_delta_state_is_floating_and_its_segment_tables_are_discrete() {
    let analysis = analyze(&kernel("gated_delta_advance", &[("A", "bf16"), ("RN", "f32")]));
    assert_eq!(
        kinds(&analysis),
        [
            ("projection", "bf16"),
            ("convolved", "f32"),
            ("rate", "f32"),
            ("time_bias", "f32"),
            ("recurrent_norm", "f32"),
            ("segments", "discrete"),
            ("stop", "discrete"),
            ("previous_bank", "discrete"),
            ("previous_tape", "discrete"),
            ("following_bank", "discrete"),
            ("delta", "f32"),
            ("tape", "f32"),
        ]
    );
    assert!(analysis.grids.is_empty());
}

#[test]
fn routed_expert_weights_are_packed_and_routes_are_discrete() {
    let entry = kernel(
        "routed_expand",
        &[("A", "bf16"), ("EGW", "q4k"), ("EUW", "q4k"), ("SGW", "q8g32s"), ("SUW", "f32")],
    );
    let analysis = analyze(&entry);
    assert_eq!(
        kinds(&analysis),
        [
            ("normalized", "bf16"),
            ("routes", "discrete"),
            ("expert_gate", "packed"),
            ("expert_up", "packed"),
            ("shared_gate", "packed"),
            ("shared_up", "f32"),
        ]
    );
    assert_eq!(coded(&analysis, "expert_gate"), Q4K_CODES);
    assert_eq!(coded(&analysis, "expert_up"), Q4K_CODES);
    assert_eq!(coded(&analysis, "shared_gate"), [Some((PlaneField::Words, 0))]);
    assert_eq!(analysis.grids.len(), 7);
    let grid = &analysis.grids[0];
    assert_eq!(grid.coordinates.iter().map(|c| c.end.to_string()).collect::<Vec<_>>(), ["E", "F", "H"]);
}

#[test]
fn every_packed_representation_has_one_grid_per_code_it_stores() {
    for info in registry::representations() {
        if !matches!(info.kind, RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_)) {
            continue;
        }
        let (planes, grids) = packed(info.id, "w", Vec::new());
        assert!(matches!(planes[0].lattice, Lattice::Codes(_)), "{}", info.name);
        for plane in &planes {
            let codes = grids.iter().filter(|grid| grid.plane.is_some_and(|(field, _)| field == plane.field));
            let expected = match plane.lattice {
                Lattice::Codes(_) => plane.fields as usize,
                Lattice::Float(_) => 0,
            };
            assert_eq!(codes.clone().count(), expected, "{} {:?}", info.name, plane.field);
            for grid in codes {
                assert!(grid.slope.is_resolved(), "{} {:?}: {}", info.name, plane.field, grid.slope);
                assert_eq!(Lattice::Codes(grid.levels.clone()), plane.lattice);
            }
        }
    }
}

#[test]
fn a_floating_codebook_reports_its_two_adjacent_gaps() {
    // MXFP4: E2M1 codes scaled by one E8M0 power of two per 32 values.
    let (planes, grids) = packed(registry::representation("mxfp4g32").unwrap(), "w", Vec::new());
    let e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut levels = e2m1.iter().rev().take(7).map(|level: &f64| -level).collect::<Vec<_>>();
    levels.extend(e2m1);
    assert_eq!(planes[0], plane(PlaneField::Words, 1, 1, Lattice::Codes(CodeLevels::Table(levels))));
    assert_eq!(planes[1].field, PlaneField::BlockScale);
    assert_eq!(planes[1].group, 32);
    let Lattice::Codes(CodeLevels::Table(scales)) = &planes[1].lattice else {
        panic!("the block scale is a floating code");
    };
    assert_eq!(scales.len(), 255);
    assert!(scales.windows(2).all(|pair| pair[1] == 2.0 * pair[0]));

    let [codes, scale] = grids.as_slice() else {
        panic!("one grid per coded plane");
    };
    let point = planes_of(2.0, 0.25);
    assert_eq!(codes.plane, Some((PlaneField::Words, 0)));
    assert_eq!(codes.steps(&point), Ok(Steps { below: Some(0.125), above: Some(0.25) }));
    // The scale is itself a code: its neighbours are half and double.
    assert_eq!(scale.plane, Some((PlaneField::BlockScale, 0)));
    assert_eq!(scale.steps(&point), Ok(Steps { below: Some(0.25), above: Some(0.5) }));
}

fn planes_of(code: f64, scale: f64) -> impl Fn(&Variable) -> Option<f64> {
    move |variable| match variable {
        Variable::Plane { plane: PlaneField::Words, .. } => Some(code),
        Variable::Plane { plane: PlaneField::BlockScale, .. } => Some(scale),
        _ => None,
    }
}

#[test]
fn integer_elements_are_codes_only_through_an_affine_decode() {
    let source = |body: &str| {
        format!(
            "fn probe[N](codes: &tensor[N] u32, scale: &tensor[N] f32, picks: &tensor[N] i32, out: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        {body}\n"
        )
    };
    let kinds_of = |body: &str| {
        let analysis = analyze(&entry(&source(body)));
        (
            stored(&analysis, "codes").clone(),
            stored(&analysis, "picks").clone(),
            analysis.grids.len(),
        )
    };
    // Scaled and offset by stored values: a decode.
    let (codes, picks, grids) = kinds_of("out[i] = f32(codes[picks[i]] & u32(7)) * scale[i] + scale[0]");
    assert_eq!(
        codes,
        Stored::Codes {
            levels: CodeLevels::Integers { min: 0, max: 7 },
            coefficients: vec!["scale".into()],
        }
    );
    assert_eq!((picks, grids), (Stored::Discrete, 1));
    // Converted, but through a cosine: no decoded value has a step in it.
    let (codes, picks, grids) = kinds_of("out[i] = cos(f32(codes[picks[i]]) * scale[i])");
    assert_eq!((codes, picks, grids), (Stored::Converted, Stored::Discrete, 0));
    // Published through a narrower floating type: a rounding boundary.
    let (codes, _, grids) = kinds_of("out[i] = f32(f16(f32(codes[i]) * scale[i]))");
    assert_eq!((codes, grids), (Stored::Converted, 0));
    // Never converted.
    let (codes, picks, grids) = kinds_of("if codes[i] > u32(3):\n            out[i] = scale[picks[i]]");
    assert_eq!((codes, picks, grids), (Stored::Discrete, Stored::Discrete, 0));
}

#[test]
fn placeholder_terms_are_unresolved() {
    let term = Term::placeholder(1, Vec::new());
    assert!(!term.is_resolved());
    assert_eq!(term.evaluate(&|_| None), Err(TermError::Unresolved));
}

#[test]
fn analysis_and_term_evaluation_observe_the_preparation_deadline() {
    use crate::cancellation::{Expired, within};
    use std::time::{Duration, Instant};

    let small = entry("fn probe(x: f32) -> f32:\n    return x\n");
    let deadline = Instant::now() + Duration::from_millis(5);
    let result = within(Some(deadline), || {
        std::thread::sleep(Duration::from_millis(10));
        analyze(&small)
    });
    assert_eq!(result, Err(Expired));

    // A deadline that passes inside the trace of a large entry abandons it.
    let large = kernel("attention_prefill_k8v4", &[("A", "bf16")]);
    let deadline = Instant::now() + Duration::from_micros(200);
    assert_eq!(within(Some(deadline), || analyze(&large)), Err(Expired));

    let term = Term::fold(term::Fold::Sum, 1, Term::constant(0.0), Term::constant(100.0),
        Term::variable(Variable::Parameter("x".into())));
    let deadline = Instant::now() + Duration::from_millis(5);
    let result = within(Some(deadline), || term.evaluate(&|_| {
        std::thread::sleep(Duration::from_millis(10));
        Some(1.0)
    }));
    assert_eq!(result, Err(Expired));
}
