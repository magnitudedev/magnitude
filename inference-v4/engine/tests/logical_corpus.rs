//! System gates for the semantic-to-logical boundary.
//!
//! These tests deliberately stop before backend planning. They prove that the
//! complete authored library can be specialized for production geometries and
//! target capability profiles without a device being present.

use seismic_lang::{
    family::{TargetEnvironment, Workload},
    logical::{self, StorageOrigin, Type, ValueKind},
    program::{compile, SourceFile},
    sir::IntrinsicUse,
    types::{DType, Elem},
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
        let environment = TargetEnvironment {
            target,
            capability_fingerprint: &fingerprint,
            supports_intrinsic: &accept_all,
        };
        for (entry, workload, expected_result_slots) in &cases {
            let logical =
                logical::specialize_entry_contract(&program, entry, &environment, workload)
                    .unwrap_or_else(|error| panic!("{entry} for {target}: {error}"));
            logical
                .verify()
                .unwrap_or_else(|error| panic!("{entry} for {target}: {error}"));
            assert_eq!(logical.target, target);
            assert_eq!(
                logical.result_slots.len(),
                *expected_result_slots,
                "{entry}"
            );
            assert!(logical.result_slots.iter().all(|slot| matches!(
                &logical.storage[slot.storage.0 as usize].origin,
                StorageOrigin::Result { path } if path == &slot.path
            )));
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
    let environment = TargetEnvironment {
        target: "cpu",
        capability_fingerprint: "logical-corpus-boundaries-v1",
        supports_intrinsic: &accept_all,
    };
    let workload = workload(&[("N", 8)], &[]);

    let scalar =
        logical::specialize_entry_contract(&program, "scalar_result", &environment, &workload)
            .expect("scalar result specialization");
    assert!(scalar.result_slots.is_empty());
    assert_eq!(
        scalar.choices[0].interface.results,
        [Type::Scalar(DType::F32)]
    );

    let range =
        logical::specialize_entry_contract(&program, "range_input", &environment, &workload)
            .expect("range input specialization");
    assert!(range
        .values
        .iter()
        .any(|value| matches!(value.kind, ValueKind::RangeParameter { ordinal: 0, .. })));

    let nested =
        logical::specialize_entry_contract(&program, "nested_results", &environment, &workload)
            .expect("nested tuple specialization");
    assert_eq!(
        nested
            .result_slots
            .iter()
            .map(|slot| slot.path.clone())
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

    let supported_environment = TargetEnvironment {
        target: "metal",
        capability_fingerprint: "metal-matrix-supported-v1",
        supports_intrinsic: &accept_all,
    };
    let supported =
        logical::specialize_entry_contract(&program, "matmul", &supported_environment, &workload)
            .expect("matmul with matrix capability");
    assert!(supported.choices[0]
        .alternatives
        .iter()
        .any(|alternative| { alternative.capabilities.contains("metal.matrix") }));

    let reject_matrix = |used: &IntrinsicUse| {
        if used.id.capability.path() == "metal.matrix" {
            Err("test profile has no matrix capability".into())
        } else {
            Ok(())
        }
    };
    let unsupported_environment = TargetEnvironment {
        target: "metal",
        capability_fingerprint: "metal-matrix-unavailable-v1",
        supports_intrinsic: &reject_matrix,
    };
    let unsupported =
        logical::specialize_entry_contract(&program, "matmul", &unsupported_environment, &workload)
            .expect("portable matmul remains applicable");

    assert!(unsupported.choices[0]
        .alternatives
        .iter()
        .all(|alternative| alternative.capabilities.is_empty()));
    assert!(unsupported.choices[0].alternatives.len() < supported.choices[0].alternatives.len());
}
