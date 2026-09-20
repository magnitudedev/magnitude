//! System gates for the semantic-to-logical boundary.
//!
//! These tests deliberately stop before backend planning. They prove that the
//! complete authored library can be specialized for production geometries and
//! target capability profiles without a device being present.

use seismic_compiler::pipeline::Workload;
use seismic_lang::{
    logical::{self, EffectiveTargetIdentity, RegionParameter, RegionResult},
    program::{compile, SourceFile},
    sir::IntrinsicUse,
    types::{DType, Elem, ValuePath, ValueType},
};
use std::collections::BTreeMap;

fn shapes(values: &[(&str, i64)]) -> BTreeMap<String, i64> {
    values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
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

fn workload(shape_values: &[(&str, i64)], elem_values: &[(&str, Elem)]) -> Workload {
    Workload {
        shapes: shapes(shape_values),
        elems: elems(elem_values),
        ..Workload::default()
    }
}

fn accept_all(_: &IntrinsicUse) -> Result<(), String> {
    Ok(())
}

fn construct<'a>(
    program: &'a seismic_lang::sir::Program,
    entry: &str,
    target: &str,
    fingerprint: &str,
    supports: &dyn Fn(&IntrinsicUse) -> Result<(), String>,
    workload: &Workload,
) -> logical::LogicalProgram {
    let identity = EffectiveTargetIdentity {
        backend: target.to_string(),
        capability_fingerprint: fingerprint.to_string(),
    };
    logical::construct(
        program,
        entry,
        &identity,
        supports,
        workload.shapes.clone(),
        workload.elems.clone(),
    )
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
fn entry_graph(logical: &logical::LogicalProgram) -> &logical::TaskGraph {
    logical
        .graphs
        .iter()
        .find(|graph| graph.choice == logical.entry_choice)
        .expect("the entry choice has a task graph")
}

#[test]
fn active_qwen_corpus_specializes_at_production_geometry_for_every_target() {
    let program = seismic_engine::models::qwen35::program::program()
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
    let cases = [
        (
            "qwen_embedding_rows",
            workload(
                &[("M", 128), ("V", 248_320), ("D", 2_560)],
                &embedding_elems,
            ),
            2,
        ),
        (
            "qwen_dense_suffix",
            workload(&[("M", 128), ("H", 2_560), ("F", 9_216)], &dense_elems),
            7,
        ),
        (
            "qwen_recurrent_sequence",
            workload(
                &[
                    ("M", 128),
                    ("H", 2_560),
                    ("NK", 16),
                    ("GV", 2),
                    ("W", 128),
                    ("C", 4),
                ],
                &recurrent_elems,
            ),
            3,
        ),
        (
            "qwen_attention_sequence",
            workload(
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
            ),
            1,
        ),
        (
            "qwen_readout_rows",
            workload(&[("M", 128), ("V", 248_320), ("D", 2_560)], &readout_elems),
            2,
        ),
    ];

    for target in ["cpu", "metal", "cuda"] {
        let fingerprint = format!("logical-corpus-{target}-v1");
        for (entry, workload, expected_result_leaves) in &cases {
            let logical = construct(&program, entry, target, &fingerprint, &accept_all, workload);
            logical
                .verify()
                .unwrap_or_else(|errors| panic!("{entry} for {target}: {errors:?}"));
            assert_eq!(logical.target.backend, target);
            let interface = logical.choice(logical.entry_choice).interface.clone();
            let leaves = tensor_leaf_paths(&interface.result);
            assert_eq!(leaves.len(), *expected_result_leaves, "{entry}");
            let graph = entry_graph(&logical);
            assert_eq!(
                graph
                    .results
                    .iter()
                    .filter(|result| matches!(
                        result,
                        RegionResult::Value {
                            ty: ValueType::Tensor(_),
                            ..
                        }
                    ))
                    .count(),
                leaves.len(),
                "{entry}: every tensor result leaf is a published region result"
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
    let workload = workload(&[("N", 8)], &[]);

    let scalar = construct(
        &program,
        "scalar_result",
        "cpu",
        "logical-corpus-boundaries-v1",
        &accept_all,
        &workload,
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
        &workload,
    );
    assert!(entry_graph(&range)
        .parameters
        .iter()
        .any(|parameter| matches!(
            parameter,
            RegionParameter::Value {
                ty: ValueType::Range { .. },
                ..
            }
        )));

    let nested = construct(
        &program,
        "nested_results",
        "cpu",
        "logical-corpus-boundaries-v1",
        &accept_all,
        &workload,
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
    let workload = workload(
        &[("M", 32), ("N", 64), ("K", 128)],
        &[
            ("T", Elem::Dtype(DType::BF16)),
            ("U", Elem::Dtype(DType::BF16)),
        ],
    );

    let supported = construct(
        &program,
        "matmul",
        "metal",
        "metal-matrix-supported-v1",
        &accept_all,
        &workload,
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
        &workload,
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
