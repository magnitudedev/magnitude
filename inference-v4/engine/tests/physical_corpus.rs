//! End-to-end semantic -> logical -> physical -> native corpus gates.
//!
//! These compile the production Qwen entry points without allocating model
//! buffers. The CPU backend is always available and therefore serves as the
//! workspace gate that every mapped construct reaches a real native encoder.
//! The CUDA gate needs a live device: native assembly compiles every PTX
//! launch through the driver, so there is no device-free CUDA compilation.

use seismic_compiler::{pipeline, planning::Budget};
use seismic_cuda::CudaCompiler;
use seismic_lang::{
    logical::specialization::{ShapeBinding, SpecializationDomain},
    precision::PrecisionPolicy,
    types::{DType, Elem},
};
use seismic_cpu::Cpu;
use std::collections::BTreeMap;

type Geometry = &'static [(&'static str, i64)];

fn domain(
    program: &seismic_lang::sir::Program,
    entry: &str,
    shapes: Geometry,
    dense: &[&str],
    packed: &[&str],
) -> SpecializationDomain {
    let shapes: BTreeMap<String, ShapeBinding> = shapes
        .iter()
        .map(|(name, value)| {
            (
                (*name).to_owned(),
                ShapeBinding::Exact(u64::try_from(*value).expect("a corpus shape is an extent")),
            )
        })
        .collect();
    let elems: BTreeMap<String, Elem> = dense
        .iter()
        .map(|name| ((*name).to_string(), Elem::Dtype(DType::BF16)))
        .chain(
            packed
                .iter()
                .map(|name| ((*name).to_string(), Elem::Repr("q4g64".into()))),
        )
        .collect();
    SpecializationDomain::new(program, entry, shapes, elems)
        .unwrap_or_else(|error| panic!("{entry}: {error}"))
}

fn cases() -> Vec<(&'static str, Geometry, &'static [&'static str], &'static [&'static str])> {
    vec![
        (
            "qwen_embedding_rows",
            &[("M", 128), ("V", 248_320), ("D", 2_560)],
            &["A"],
            &["EW"],
        ),
        (
            "qwen_dense_suffix",
            &[("M", 128), ("H", 2_560), ("F", 9_216)],
            &["A", "NW"],
            &["GW", "UW", "DW"],
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
            &["A", "NW", "RN"],
            &["QW", "GW", "AW", "BW", "OW"],
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
            &["A", "NW"],
            &["QW", "KW", "VW", "OW"],
        ),
        (
            "qwen_readout_rows",
            &[("M", 128), ("V", 248_320), ("D", 2_560)],
            &["A", "NW"],
            &["OW"],
        ),
    ]
}

fn budget() -> Budget {
    Budget {
        work: 200_000,
        time: None,
    }
}

#[test]
fn active_qwen_corpus_reaches_native_cpu_encoding() {
    let program =
        magnitude_engine::models::qwen35::program::program().expect("active Qwen Seismic corpus");
    let backend = Cpu::host(1).expect("host CPU profile");
    for (entry, shapes, dense, packed) in cases() {
        pipeline::compile(
            &program,
            &domain(&program, entry, shapes, dense, packed),
            &PrecisionPolicy::default(),
            &backend,
            &[],
            budget(),
        )
        .unwrap_or_else(|error| panic!("{entry}: {error}"));
    }
}

#[test]
fn active_qwen_corpus_reaches_resolved_cuda_ptx() {
    let program =
        magnitude_engine::models::qwen35::program::program().expect("active Qwen Seismic corpus");
    let Ok(device) = seismic_cuda::Device::open(0) else {
        eprintln!("no CUDA device present; skipping the CUDA corpus gate");
        return;
    };
    let backend = CudaCompiler::new(&device).expect("device-bound CUDA compiler");
    for (entry, shapes, dense, packed) in cases() {
        // This gate exercises semantic -> logical -> physical -> PTX coverage.
        // Numerical qualification has separate evidence gates; CUDA's available
        // exponential lowering is explicitly approximate.
        pipeline::compile(
            &program,
            &domain(&program, entry, shapes, dense, packed),
            &PrecisionPolicy::Unconstrained,
            &backend,
            &[],
            budget(),
        )
        .unwrap_or_else(|error| panic!("{entry}: {error}"));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn active_qwen_corpus_reaches_resolved_msl() {
    let program =
        magnitude_engine::models::qwen35::program::program().expect("active Qwen Seismic corpus");
    let device = seismic_metal::runtime::Device::open().expect("local Metal device");
    let backend = seismic_metal::catalog::MetalCompiler::from_device(&device);
    for (entry, shapes, dense, packed) in cases() {
        pipeline::compile(
            &program,
            &domain(&program, entry, shapes, dense, packed),
            &PrecisionPolicy::default(),
            &backend,
            &[],
            budget(),
        )
        .unwrap_or_else(|error| panic!("{entry}: {error}"));
    }
}
