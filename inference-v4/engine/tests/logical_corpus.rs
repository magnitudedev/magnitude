//! System gates for the semantic-to-logical boundary.
//!
//! These tests deliberately stop before backend planning. They prove that the
//! complete authored library can be specialized for production geometries and
//! target capability profiles without a device being present.

use seismic_lang::{
    logical::{
        self,
        boundary::BoundaryLeaf,
        EffectiveTargetIdentity, TaskGraph,
        specialization::{ShapeBinding, SpecializationDomain},
    },
    program::{compile, SourceFile},
    sir::IntrinsicUse,
    types::{DType, Elem, ValuePath, ValueType},
};
use std::collections::BTreeMap;

fn shape_bindings(values: &[(&str, i64)]) -> BTreeMap<String, ShapeBinding> {
    values
        .iter()
        .map(|(name, value)| {
            (
                (*name).to_owned(),
                ShapeBinding::Exact(u64::try_from(*value).expect("a test shape is a valid extent")),
            )
        })
        .collect()
}

fn elems(values: &[(&str, Elem)]) -> BTreeMap<String, Elem> {
    values
        .iter()
        .map(|(name, elem)| ((*name).to_owned(), elem.clone()))
        .collect()
}

fn dense<'a>(names: &'a [&'a str]) -> Vec<(&'a str, Elem)> {
    names
        .iter()
        .map(|name| (*name, Elem::Dtype(DType::BF16)))
        .collect()
}

fn packed<'a>(names: &'a [&'a str]) -> Vec<(&'a str, Elem)> {
    names
        .iter()
        .map(|name| (*name, Elem::Repr("q4g64".into())))
        .collect()
}

fn accept_all(_: &IntrinsicUse) -> Result<(), String> {
    Ok(())
}

fn domain(
    program: &seismic_lang::sir::Program,
    entry: &str,
    shape_values: &[(&str, i64)],
    elem_values: &[(&str, Elem)],
) -> SpecializationDomain {
    SpecializationDomain::new(program, entry, shape_bindings(shape_values), elems(elem_values))
        .unwrap_or_else(|error| panic!("{entry}: {error}"))
}

fn construct<'a>(
    program: &'a seismic_lang::sir::Program,
    entry: &str,
    target: &str,
    fingerprint: &str,
    supports: &dyn Fn(&IntrinsicUse) -> Result<(), String>,
    domain: &SpecializationDomain,
) -> logical::LogicalProgram {
    let identity = EffectiveTargetIdentity {
        backend: target.to_string(),
        capability_fingerprint: fingerprint.to_string(),
    };
    logical::construct(program, &identity, supports, domain)
        .unwrap_or_else(|error| panic!("{entry} for {target}: {error:?}"))
}

/// Tensor leaf paths of one value type, in canonical traversal order.
fn tensor_leaf_paths(ty: &ValueType) -> Vec<ValuePath> {
    fn walk(ty: &ValueType, prefix: &mut Vec<u32>, out: &mut Vec<ValuePath>) {
        match ty {
            ValueType::Tensor(_) => out.push(ValuePath(prefix.clone())),
            ValueType::Tuple(children) => {
                for (ordinal, child) in children.iter().enumerate() {
                    prefix.push(ordinal as u32);
                    walk(child, prefix, out);
                    prefix.pop();
                }
            }
            _ => (),
        }
    }
    let mut out = Vec::new();
    walk(ty, &mut Vec::new(), &mut out);
    out
}

/// The entry occurrence's task graph.
fn entry_graph(logical: &logical::LogicalProgram) -> &TaskGraph {
    logical
        .graphs()
        .find(|(_, graph)| graph.choice == logical.entry_choice)
        .map(|(_, graph)| graph)
        .expect("the entry choice has a task graph")
}

#[test]
fn active_qwen_corpus_specializes_at_production_geometry_for_every_target() {
    let program = magnitude_engine::models::qwen35::program::program()
        .expect("the active seismic-std and engine source corpus must compile");

    let mut embedding_elems = dense(&["A"]);
    embedding_elems.extend(packed(&["EW"]));
    let mut dense_elems = dense(&["A", "NW"]);
    dense_elems.extend(packed(&["GW", "UW", "DW"]));
    let mut recurrent_elems = dense(&["A", "NW", "RN"]);
    recurrent_elems.extend(packed(&["QW", "GW", "AW", "BW", "OW"]));
    let mut attention_elems = dense(&["A", "NW"]);
    attention_elems.extend(packed(&["QW", "KW", "VW", "OW"]));
    let mut readout_elems = dense(&["A", "NW"]);
    readout_elems.extend(packed(&["OW"]));

    // Qwen3.5-4B geometry, including the 16K history that originally exposed
    // the split between selection accounting and physical realization.
    let cases: Vec<(&str, &[(&str, i64)], &[(&str, Elem)], usize)> = vec![
        (
            "qwen_embedding_rows",
            &[("M", 128), ("V", 248_320), ("D", 2_560)],
            &embedding_elems,
            2,
        ),
        (
            "qwen_dense_suffix",
            &[("M", 128), ("H", 2_560), ("F", 9_216)],
            &dense_elems,
            7,
        ),
        (
            "qwen_recurrent_sequence",
            &[
                ("M", 128),
                ("H", 2_560),
                ("NK", 16),
                ("GV", 2),
                ("W", 128),
                ("C", 4),
            ],
            &recurrent_elems,
            3,
        ),
        (
            "qwen_attention_sequence",
            &[
                ("M", 128),
                ("D", 2_560),
                ("T", 16_384),
                ("G", 4),
                ("KV", 4),
                ("P", 32),
                ("S", 192),
                ("SH", 11),
                ("SW", 10),
                ("R", 1),
            ],
            &attention_elems,
            1,
        ),
        (
            "qwen_readout_rows",
            &[("M", 128), ("V", 248_320), ("D", 2_560)],
            &readout_elems,
            2,
        ),
    ];

    for target in ["cpu", "metal", "cuda"] {
        let fingerprint = format!("logical-corpus-{target}-v1");
        for (entry, shape_values, elem_values, expected_result_leaves) in &cases {
            let domain = domain(&program, entry, shape_values, elem_values);
            let logical = construct(&program, entry, target, &fingerprint, &accept_all, &domain);
            logical
                .verify()
                .unwrap_or_else(|errors| panic!("{entry} for {target}: {errors:?}"));
            assert_eq!(logical.target.backend, target);
            let interface = logical.choice(logical.entry_choice).interface.clone();
            let leaves = tensor_leaf_paths(&interface.result);
            assert_eq!(leaves.len(), *expected_result_leaves, "{entry}");
            let graph = entry_graph(&logical);
            let published = graph
                .boundary
                .results()
                .keys()
                .filter_map(|leaf| match leaf {
                    BoundaryLeaf::Result { leaf } => Some(leaf.clone()),
                    BoundaryLeaf::Input { .. } => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                published, leaves,
                "{entry}: every tensor result leaf is a published boundary result"
            );
        }
    }
}

#[test]
fn logical_abi_covers_scalar_range_and_nested_tuple_results() {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "logical-corpus-boundaries.seismic".into(),
        text: "fn scalar_result[N](values: tensor[N] f32) -> f32:\n    return reduce(f32(values), 0, sum)\n\nfn range_input[N](selected: range[N]):\n    return\n\nfn nested_results[N](value: tensor[N] f32) -> (tensor[N] f32, (tensor[N] f32, tensor[N] f32)):\n    return value, (value, value)\n".into(),
    });
    let program = compile(&sources).unwrap_or_else(|diagnostics| {
        panic!(
            "{}",
            diagnostics
                .iter()
                .map(|diagnostic| diagnostic.render())
                .collect::<Vec<_>>()
                .join("\n")
        )
    });
    let geometry: &[(&str, i64)] = &[("N", 8)];

    let scalar = construct(
        &program,
        "scalar_result",
        "cpu",
        "logical-corpus-boundaries-v1",
        &accept_all,
        &domain(&program, "scalar_result", geometry, &[]),
    );
    let interface = scalar.choice(scalar.entry_choice).interface.clone();
    assert_eq!(interface.result, ValueType::Scalar(DType::F32));
    assert!(tensor_leaf_paths(&interface.result).is_empty());

    let range = construct(
        &program,
        "range_input",
        "cpu",
        "logical-corpus-boundaries-v1",
        &accept_all,
        &domain(&program, "range_input", geometry, &[]),
    );
    let interface = range.choice(range.entry_choice).interface.clone();
    let selected = interface
        .params
        .iter()
        .position(|param| matches!(param.ty, ValueType::Range { .. }))
        .expect("the entry interface declares a range parameter");
    assert!(entry_graph(&range)
        .boundary
        .inputs()
        .contains_key(&BoundaryLeaf::Input {
            param: selected as u32,
            leaf: ValuePath(Vec::new()),
        }));

    let nested = construct(
        &program,
        "nested_results",
        "cpu",
        "logical-corpus-boundaries-v1",
        &accept_all,
        &domain(&program, "nested_results", geometry, &[]),
    );
    let interface = nested.choice(nested.entry_choice).interface.clone();
    assert_eq!(
        tensor_leaf_paths(&interface.result)
            .iter()
            .map(|path| path.0.clone())
            .collect::<Vec<_>>(),
        [vec![0], vec![1, 0], vec![1, 1]]
    );
}

#[test]
fn unavailable_capability_removes_only_the_dependent_implementation() {
    let program = seismic_std::program().expect("active seismic-std corpus");
    let geometry: &[(&str, i64)] = &[("M", 32), ("N", 64), ("K", 128)];
    let element_values: &[(&str, Elem)] = &[
        ("T", Elem::Dtype(DType::BF16)),
        ("U", Elem::Dtype(DType::BF16)),
    ];
    let domain = domain(&program, "matmul", geometry, element_values);

    let supported = construct(
        &program,
        "matmul",
        "metal",
        "metal-matrix-supported-v1",
        &accept_all,
        &domain,
    );
    assert!(supported
        .choice(supported.entry_choice)
        .alternatives
        .iter()
        .any(|alternative| alternative
            .required_capabilities
            .iter()
            .any(|capability| capability.path() == "metal.matrix")));

    let reject_matrix = |used: &IntrinsicUse| {
        if used.id.capability.path() == "metal.matrix" {
            Err("test profile has no matrix capability".into())
        } else {
            Ok(())
        }
    };
    let unsupported = construct(
        &program,
        "matmul",
        "metal",
        "metal-matrix-unavailable-v1",
        &reject_matrix,
        &domain,
    );

    assert!(unsupported
        .choice(unsupported.entry_choice)
        .alternatives
        .iter()
        .all(|alternative| alternative.required_capabilities.is_empty()));
    assert!(
        unsupported
            .choice(unsupported.entry_choice)
            .alternatives
            .len()
        < supported.choice(supported.entry_choice).alternatives.len()
    );
}
