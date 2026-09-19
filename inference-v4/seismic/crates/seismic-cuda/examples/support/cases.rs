//! Kernel cases of the CUDA sweeps: the std kernel table of `seismic-runtime/tests/kernels.rs`
//! (copied so this crate does not depend on the runtime crate), minus the real-geometry
//! k-quant `linear` cases, which exist to exercise Metal lowerings at Qwen dimensions.
use seismic_lang::family::Workload;
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::repr;
use seismic_lang::sir::{Definition, Program};
use seismic_lang::types::{DType, Elem};

/// Inputs are consumed by the native sweep only; the device-free example reads the rest.
#[allow(dead_code)]
pub struct Case {
    pub label: &'static str,
    pub entry: &'static str,
    pub shapes: &'static [(&'static str, i64)],
    pub elems: &'static [(&'static str, &'static str)],
    pub scalars: &'static [(&'static str, f64)],
    /// Explicit contents of control tensors (ranges, coordinates).
    pub contents: &'static [(&'static str, &'static [f64])],
    /// GGUF import: `data` is random raw words and `halves` is the f16 view of the same bytes.
    pub raw: bool,
}

const fn case(
    label: &'static str,
    entry: &'static str,
    shapes: &'static [(&'static str, i64)],
    elems: &'static [(&'static str, &'static str)],
) -> Case {
    Case {
        label,
        entry,
        shapes,
        elems,
        scalars: &[],
        contents: &[],
        raw: false,
    }
}

const TUV: &[(&str, &str)] = &[("T", "bf16"), ("U", "bf16"), ("V", "bf16")];
const MN: &[(&str, i64)] = &[("M", 3), ("N", 7)];
const NK: &[(&str, i64)] = &[("N", 5), ("K", 64)];
const EPS: &[(&str, f64)] = &[("eps", 1e-6)];
const A: &[(&str, &str)] = &[("A", "bf16")];

pub fn cases() -> Vec<Case> {
    let mut all = vec![
        Case {
            scalars: EPS,
            ..case("rms_norm", "rms_norm", &[("R", 3), ("W", 8)], TUV)
        },
        Case {
            scalars: EPS,
            ..case(
                "rms_norm W=96 f32",
                "rms_norm",
                &[("R", 3), ("W", 96)],
                &[("T", "f32"), ("U", "bf16"), ("V", "f32")],
            )
        },
        Case {
            scalars: &[("grouped", 1.0)],
            ..case(
                "delta_step W=64",
                "delta_step",
                &[("NK", 2), ("GV", 2), ("W", 64)],
                A,
            )
        },
        case(
            "linear bf16 M=1",
            "linear",
            &[("M", 1), ("N", 5), ("K", 64)],
            TUV,
        ),
        case(
            "linear bf16 M=4",
            "linear",
            &[("M", 4), ("N", 5), ("K", 64)],
            TUV,
        ),
        case(
            "linear q4g64 M=1",
            "linear",
            &[("M", 1), ("N", 5), ("K", 64)],
            &[("T", "bf16"), ("U", "q4g64"), ("V", "bf16")],
        ),
        case(
            "linear q4g64 M=4",
            "linear",
            &[("M", 4), ("N", 5), ("K", 64)],
            &[("T", "bf16"), ("U", "q4g64"), ("V", "bf16")],
        ),
        case(
            "linear q4g64 M=1 K=1024",
            "linear",
            &[("M", 1), ("N", 12), ("K", 1024)],
            &[("T", "bf16"), ("U", "q4g64"), ("V", "f32")],
        ),
        case(
            "linear q4g64 M=1 K=1088",
            "linear",
            &[("M", 1), ("N", 12), ("K", 1088)],
            &[("T", "bf16"), ("U", "q4g64"), ("V", "f32")],
        ),
        case(
            "linear q4g64 M=8 K=1024",
            "linear",
            &[("M", 8), ("N", 16), ("K", 1024)],
            &[("T", "bf16"), ("U", "q4g64"), ("V", "bf16")],
        ),
        case(
            "linear_bias",
            "linear_bias",
            &[("M", 2), ("N", 5), ("K", 64)],
            &[("T", "bf16"), ("U", "bf16"), ("B", "bf16"), ("V", "bf16")],
        ),
        case("projection", "projection", NK, &[]),
        case("gated_projection", "gated_projection", NK, &[]),
        Case {
            scalars: EPS,
            ..case("norm_gated_projection", "norm_gated_projection", NK, &[])
        },
        case("projection_add", "projection_add", NK, &[]),
        case("silu", "silu", MN, TUV),
        case("sigmoid", "sigmoid", MN, TUV),
        case("multiply", "multiply", MN, TUV),
        case("add", "add", MN, TUV),
        case(
            "cast_rows",
            "cast_rows",
            &[("M", 3), ("K", 7)],
            &[("T", "bf16"), ("U", "f32")],
        ),
        Case {
            scalars: &[("token", 4.0)],
            ..case(
                "embedding_row",
                "embedding_row",
                &[("V", 6), ("K", 64)],
                &[("T", "q4g64"), ("A", "bf16")],
            )
        },
        Case {
            scalars: &[("base", 10000.0), ("epsilon", 1e-6)],
            contents: &[("coordinates", &[3.0, 1.0, 2.0, 0.0, 7.0, 4.0, 5.0, 0.0])],
            ..case(
                "rotary_prepare",
                "rotary_prepare",
                &[
                    ("Q", 2),
                    ("H", 2),
                    ("KV", 1),
                    ("P", 3),
                    ("S", 2),
                    ("SH", 1),
                    ("SW", 1),
                ],
                A,
            )
        },
        Case {
            scalars: &[("pos", 3.0), ("theta", 10000.0), ("eps", 1e-6)],
            ..case(
                "attention_prepare",
                "attention_prepare",
                &[("H", 2), ("G", 1), ("R", 4), ("S", 4), ("T", 6)],
                &[],
            )
        },
        Case {
            scalars: &[("scale", 0.35)],
            contents: &[("visible", &[0.0, 3.0, 0.0, 4.0, 1.0, 5.0])],
            ..case(
                "attention",
                "attention",
                &[("Q", 3), ("T", 5), ("H", 4), ("KV", 2), ("W", 8)],
                A,
            )
        },
        Case {
            scalars: &[("scale", 0.35)],
            contents: &[("visible", &[1.0, 4.0])],
            ..case(
                "attention_decode",
                "attention_decode",
                &[("T", 5), ("H", 4), ("KV", 2), ("W", 8)],
                A,
            )
        },
        Case {
            scalars: &[("destination", 3.0)],
            ..case(
                "kv_append",
                "kv_append",
                &[("T", 5), ("KV", 2), ("W", 8)],
                A,
            )
        },
        Case {
            scalars: EPS,
            ..case(
                "recurrent_prepare",
                "recurrent_prepare",
                &[("NK", 2), ("NV", 4), ("W", 4), ("C", 4)],
                &[("A", "bf16"), ("CW", "bf16")],
            )
        },
        Case {
            scalars: &[("grouped", 1.0)],
            ..case(
                "delta_step",
                "delta_step",
                &[("NK", 2), ("GV", 2), ("W", 4)],
                A,
            )
        },
        Case {
            scalars: EPS,
            ..case("gated_norm", "gated_norm", &[("NV", 2), ("W", 8)], &[])
        },
        case(
            "attention_gate",
            "attention_gate",
            &[("H", 2), ("W", 8)],
            &[],
        ),
        case("argmax_row", "argmax_row", &[("V", 64), ("B", 16)], &[]),
        case("logits", "logits", NK, &[]),
        Case {
            scalars: &[("epsilon", 1e-5)],
            ..case(
                "layer_norm",
                "layer_norm",
                &[("R", 3), ("W", 8)],
                &[("T", "bf16"), ("U", "bf16"), ("B", "bf16"), ("V", "bf16")],
            )
        },
        case("gelu", "gelu", MN, &[("T", "bf16"), ("U", "bf16")]),
    ];
    // Every packed representation through `linear` and `embedding_row`, the GGUF and dense
    // weight imports.
    for name in ["q4g32", "q4k", "q5k", "q6k", "q8g32s", "iq4g32", "q8g32"] {
        let elems: &'static [(&str, &str)] = vec![("T", "bf16"), ("U", name), ("V", "f32")].leak();
        for m in [1, 3] {
            let shapes: &'static [(&str, i64)] = vec![("M", m), ("N", 96), ("K", 768)].leak();
            all.push(case(
                format!("linear {name} M={m}").leak(),
                "linear",
                shapes,
                elems,
            ));
        }
        all.push(Case {
            scalars: &[("token", 4.0)],
            ..case(
                format!("embedding_row {name}").leak(),
                "embedding_row",
                &[("V", 6), ("K", 768)],
                vec![("T", name), ("A", "bf16")].leak(),
            )
        });
    }
    for entry in [
        "import_q4k",
        "import_q5k",
        "import_q6k",
        "import_q8_0",
        "import_iq4_xs",
    ] {
        all.push(Case {
            raw: true,
            ..case(entry, entry, &[("B", 3)], &[])
        });
    }
    for (source, negative) in [("f16", 0.0), ("bf16", 0.0), ("f32", 0.0), ("f32", 1.0)] {
        all.push(Case {
            scalars: if negative == 0.0 {
                &[("negative_exp", 0.0)]
            } else {
                &[("negative_exp", 1.0)]
            },
            ..case(
                format!("import_weight {source} neg={negative}").leak(),
                "import_weight",
                &[("N", 100)],
                vec![("T", source), ("U", "bf16")].leak(),
            )
        });
    }
    all
}

pub fn element(name: &str) -> Result<Elem, String> {
    match DType::from_name(name) {
        Some(dtype) => Ok(Elem::Dtype(dtype)),
        None if repr::lookup(name).is_some() => Ok(Elem::Repr(name.into())),
        None => Err(format!("unknown element {name}")),
    }
}

pub fn workload(case: &Case, precision: PrecisionPolicy) -> Result<Workload, String> {
    Ok(Workload {
        shapes: case
            .shapes
            .iter()
            .map(|(n, v)| (n.to_string(), *v))
            .collect(),
        elems: case
            .elems
            .iter()
            .map(|(n, e)| Ok((n.to_string(), element(e)?)))
            .collect::<Result<_, String>>()?,
        precision,
    })
}

/// A family definition whose parameters are the entry's invocation ABI.
#[allow(dead_code)]
pub fn abi<'a>(program: &'a Program, entry: &str) -> Result<&'a Definition, String> {
    let family = program.resolve_family(entry)?;
    family
        .bodies
        .iter()
        .chain(&family.lowerings)
        .map(|id| program.definition(*id))
        .next()
        .ok_or_else(|| format!("{entry} has no implementation"))
}

/// seismic-std checked for the CUDA target.
pub fn program() -> Result<Program, String> {
    seismic_lang::program::compile(&seismic_std::sources()).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.render())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// `PRECISION=unconstrained` enables exploration instead of the default exact policy.
pub fn precision() -> Result<PrecisionPolicy, String> {
    match std::env::var("PRECISION").as_deref() {
        Err(_) | Ok("exact") => Ok(PrecisionPolicy::Exact),
        Ok("unconstrained") => Ok(PrecisionPolicy::Unconstrained),
        Ok(other) => Err(format!("PRECISION={other}: expected `exact` or `unconstrained`")),
    }
}

/// Cases kept by `KERNELS=<substring>` and not dropped by `KERNELS_SKIP=<substring>`.
pub fn selected() -> Vec<Case> {
    let (filter, skip) = (
        std::env::var("KERNELS").ok(),
        std::env::var("KERNELS_SKIP").ok(),
    );
    cases()
        .into_iter()
        .filter(|c| {
            filter.as_ref().is_none_or(|f| c.label.contains(f.as_str()))
                && !skip.as_ref().is_some_and(|s| c.label.contains(s.as_str()))
        })
        .collect()
}

/// Outcome of one case with panics reported as errors (the pipeline is under test).
pub fn guarded<T>(run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|panic| {
        let text = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()));
        Err(format!(
            "panic: {}",
            text.unwrap_or_else(|| "non-string payload".into())
        ))
    })
}
