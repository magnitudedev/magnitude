//! The canonical seismic-std sweep corpus: every std kernel, every packed representation,
//! and the GGUF/dense weight imports, with per-case precision modes. One table consumed by
//! every correctness sweep — `seismic-runtime`'s gate tests and the CUDA sweeps — so kernel
//! coverage cannot drift between consumers.
use seismic_lang::precision::PrecisionPolicy;

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
    /// Every output compares exactly even under unconstrained exploration.
    pub exact: bool,
    pub modes: &'static [PrecisionPolicy],
}

pub const fn case(
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
        exact: false,
        modes: &[PrecisionPolicy::Unconstrained],
    }
}

const TUV: &[(&str, &str)] = &[("T", "bf16"), ("U", "bf16"), ("V", "bf16")];
const MN: &[(&str, i64)] = &[("M", 3), ("N", 7)];
const NK: &[(&str, i64)] = &[("N", 5), ("K", 64)];
const EPS: &[(&str, f64)] = &[("eps", 1e-6)];
const A: &[(&str, &str)] = &[("A", "bf16")];

pub const BOTH: &[PrecisionPolicy] = &[PrecisionPolicy::Unconstrained, PrecisionPolicy::Exact];

pub fn cases() -> Vec<Case> {
    vec![
        Case {
            scalars: EPS,
            ..case("rms_norm", "rms_norm", &[("R", 3), ("W", 8)], TUV)
        },
        // Rows of at least one subgroup of elements: the lane-distributed tile and `sum_any_order`'s collective body.
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
        // Direct coverage for the logical Metal matrix lowering: one complete native 8x8x8
        // atom and a shape where every axis has a scalar tail. Both compare against the
        // interpreter under the ordinary qualified tolerance policy; native MMA is not Exact.
        Case {
            modes: BOTH,
            ..case(
                "linear logical matrix full atom",
                "linear",
                &[("M", 8), ("N", 8), ("K", 8)],
                &[("T", "f32"), ("U", "f32"), ("V", "f32")],
            )
        },
        Case {
            modes: BOTH,
            ..case(
                "linear logical matrix tails",
                "linear",
                &[("M", 9), ("N", 9), ("K", 9)],
                &[("T", "f32"), ("U", "f32"), ("V", "f32")],
            )
        },
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
        // Whole 512-column chunks, and a 64-column tail: the packet vector lowering of
        // `matmul`'s reassociating body. F32 outputs keep the admitted summation rounding
        // visible instead of hiding it in bf16.
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
                &[("T", 5), ("G", 2), ("KV", 2), ("W", 8)],
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
    ]
}

/// Every packed representation through `linear` and `embedding_row`, the GGUF and dense
/// weight imports, and the k-quants at Qwen3.5-4B projection geometry.
pub fn packed_cases() -> Vec<Case> {
    let mut all = Vec::new();
    for name in ["q4g32", "q4k", "q5k", "q6k", "q8g32s", "iq4g32", "q8g32"] {
        let elems: &'static [(&str, &str)] = vec![("T", "bf16"), ("U", name), ("V", "f32")].leak();
        for m in [1, 3] {
            let shapes: &'static [(&str, i64)] = vec![("M", m), ("N", 96), ("K", 768)].leak();
            all.push(Case {
                modes: BOTH,
                ..case(
                    format!("linear {name} M={m}").leak(),
                    "linear",
                    shapes,
                    elems,
                )
            });
        }
        all.push(Case {
            scalars: &[("token", 4.0)],
            modes: BOTH,
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
            exact: true,
            modes: BOTH,
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
            exact: negative == 0.0,
            modes: BOTH,
            ..case(
                format!("import_weight {source} neg={negative}").leak(),
                "import_weight",
                &[("N", 100)],
                vec![("T", source), ("U", "bf16")].leak(),
            )
        });
    }
    const EXACT: &[PrecisionPolicy] = &[PrecisionPolicy::Exact];
    for (name, n, k) in [
        ("q4k", 9216, 2560),
        ("q5k", 9216, 2560),
        ("q6k", 9216, 2560),
        ("q6k", 2560, 9216),
    ] {
        let shapes: &'static [(&str, i64)] = vec![("M", 1), ("N", n), ("K", k)].leak();
        let elems: &'static [(&str, &str)] = vec![("T", "bf16"), ("U", name), ("V", "f32")].leak();
        all.push(Case {
            modes: EXACT,
            ..case(
                format!("linear {name} {n}x{k}").leak(),
                "linear",
                shapes,
                elems,
            )
        });
    }
    all
}
