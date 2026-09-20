//! End-to-end semantic -> logical -> physical -> native corpus gates.
//!
//! These compile the production Qwen entry points without allocating model
//! buffers. The CPU backend is always available and therefore serves as the
//! workspace gate that every mapped construct reaches a real native encoder.

use seismic_compiler::pipeline::Workload;
use seismic_compiler::{pipeline, planning::Budget};
use seismic_cpu::mapping::Cpu;
use seismic_cuda::mapping::Cuda;
use seismic_lang::{
    precision::PrecisionPolicy,
    types::{DType, Elem},
};
use std::collections::BTreeMap;

fn workload(shapes: &[(&str, i64)], dense: &[&str], packed: &[&str]) -> Workload {
    Workload {
        shapes: shapes
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect(),
        elems: dense
            .iter()
            .map(|name| ((*name).to_string(), Elem::Dtype(DType::BF16)))
            .chain(
                packed
                    .iter()
                    .map(|name| ((*name).to_string(), Elem::Repr("q4g64".into()))),
            )
            .collect::<BTreeMap<_, _>>(),
        ..Workload::default()
    }
}

fn cases() -> Vec<(&'static str, Workload)> {
    vec![
        (
            "qwen_embedding_rows",
            workload(&[("M", 128), ("V", 248_320), ("D", 2_560)], &["A"], &["EW"]),
        ),
        (
            "qwen_dense_suffix",
            workload(
                &[("M", 128), ("H", 2_560), ("F", 9_216)],
                &["A", "NW"],
                &["GW", "UW", "DW"],
            ),
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
                &["A", "NW", "RN"],
                &["QW", "GW", "AW", "BW", "OW"],
            ),
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
                &["A", "NW"],
                &["QW", "KW", "VW", "OW"],
            ),
        ),
        (
            "qwen_readout_rows",
            workload(
                &[("M", 128), ("V", 248_320), ("D", 2_560)],
                &["A", "NW"],
                &["OW"],
            ),
        ),
    ]
}

#[test]
fn active_qwen_corpus_reaches_native_cpu_encoding() {
    let program =
        seismic_engine::models::qwen35::program::program().expect("active Qwen Seismic corpus");
    let backend = Cpu::host(1).expect("host CPU profile");
    for (entry, workload) in cases() {
        pipeline::compile(
            &program,
            entry,
            &workload,
            &backend,
            &[],
            Budget {
                work: 200_000,
                time: None,
            },
        )
        .unwrap_or_else(|error| panic!("{entry}: {error}"));
    }
}

#[test]
fn active_qwen_corpus_reaches_resolved_cuda_ptx() {
    let program =
        seismic_engine::models::qwen35::program::program().expect("active Qwen Seismic corpus");
    let backend = Cuda::new(
        seismic_cuda::mapping::Limits::gb10(),
        seismic_cuda::mapping::EstimateModel::default(),
    )
    .expect("documented GB10 profile");
    for (entry, mut workload) in cases() {
        // This gate exercises semantic -> logical -> physical -> PTX coverage.
        // Numerical qualification has separate evidence gates; CUDA's available
        // exponential lowering is explicitly approximate.
        workload.precision = PrecisionPolicy::Unconstrained;
        pipeline::compile(
            &program,
            entry,
            &workload,
            &backend,
            &[],
            Budget {
                work: 200_000,
                time: None,
            },
        )
        .unwrap_or_else(|error| panic!("{entry}: {error}"));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn active_qwen_corpus_reaches_resolved_msl() {
    let program =
        seismic_engine::models::qwen35::program::program().expect("active Qwen Seismic corpus");
    let device = seismic_metal::runtime::Device::open().expect("local Metal device");
    let backend = seismic_metal::mapping::MetalCompiler::from_device(&device)
        .expect("queried Metal compiler");
    for (entry, workload) in cases() {
        pipeline::compile(
            &program,
            entry,
            &workload,
            &backend,
            &[],
            Budget {
                work: 200_000,
                time: None,
            },
        )
        .unwrap_or_else(|error| panic!("{entry}: {error}"));
    }
}
