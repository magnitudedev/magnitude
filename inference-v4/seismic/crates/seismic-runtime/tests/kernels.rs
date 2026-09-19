//! Gate G3: every linked seismic-std kernel the Qwen path uses, selected and run on
//! Metal, agrees with the reference interpreter. The same case table runs on the CPU device
//! under exact precision, where every output must be bit-identical to the interpreter.
use seismic_lang::family::Workload;
use seismic_lang::precision::{compare_dense, Limit, PrecisionPolicy, SpecialPolicy, Tolerance};
use seismic_lang::interp::{Arg, Interpreter, Rng, TensorData, Uniform};
use seismic_lang::repr;
use seismic_lang::sir::{Definition, Program};
use seismic_lang::syntax::ast::Mode;
use seismic_lang::types::{DType, Elem};
use seismic_lang::types::{Extent, Ty};
use seismic_runtime::plan::{Bindings, PlanCompiler, Settings};
use seismic_runtime::{Buffer, Device};
use std::collections::HashMap;

struct Case {
    label: &'static str,
    entry: &'static str,
    shapes: &'static [(&'static str, i64)],
    elems: &'static [(&'static str, &'static str)],
    scalars: &'static [(&'static str, f64)],
    /// Explicit contents of control tensors (ranges, coordinates).
    contents: &'static [(&'static str, &'static [f64])],
    /// GGUF import: `data` is random raw words and `halves` is the f16 view of the same bytes.
    raw: bool,
    /// Every output compares exactly even under unconstrained exploration.
    exact: bool,
    modes: &'static [PrecisionPolicy],
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
        exact: false,
        modes: &[PrecisionPolicy::Unconstrained],
    }
}

const TUV: &[(&str, &str)] = &[("T", "bf16"), ("U", "bf16"), ("V", "bf16")];
const MN: &[(&str, i64)] = &[("M", 3), ("N", 7)];
const NK: &[(&str, i64)] = &[("N", 5), ("K", 64)];
const EPS: &[(&str, f64)] = &[("eps", 1e-6)];
const A: &[(&str, &str)] = &[("A", "bf16")];

fn cases() -> Vec<Case> {
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
        // Whole 512-column chunks, and a 64-column tail: the packet vector lowering of `matmul_any_order`.
        // F32 outputs keep the admitted summation rounding visible instead of hiding it in bf16.
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
    ]
}

const BOTH: &[PrecisionPolicy] = &[PrecisionPolicy::Unconstrained, PrecisionPolicy::Exact];

/// Every packed representation through `linear` and `embedding_row`, the GGUF and dense
/// weight imports, and the k-quants at Qwen3.5-4B projection geometry.
fn packed_cases() -> Vec<Case> {
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

fn abi<'a>(program: &'a Program, entry: &str) -> Result<&'a Definition, String> {
    let family = program.resolve_family(entry)?;
    family
        .bodies
        .iter()
        .chain(&family.lowerings)
        .map(|id| program.definition(*id))
        .next()
        .ok_or_else(|| format!("{entry} has no implementation"))
}

fn element(name: &str) -> Result<Elem, String> {
    match DType::from_name(name) {
        Some(dtype) => Ok(Elem::Dtype(dtype)),
        None if repr::lookup(name).is_some() => Ok(Elem::Repr(name.into())),
        None => Err(format!("unknown element {name}")),
    }
}

struct Bound {
    buffers: HashMap<(String, String), Buffer>,
    scalars: HashMap<String, f64>,
}
impl Bindings for Bound {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        self.buffers.get(&(root.to_string(), plane.to_string()))
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        self.scalars.get(name).copied()
    }
}

/// Max absolute difference over all out/inout tensors, or the failing stage and its error.
fn run(
    program: &Program,
    device: &Device,
    case: &Case,
    precision: PrecisionPolicy,
    seed: u64,
) -> Result<f64, String> {
    let definition = abi(program, case.entry)?;
    let workload = Workload {
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
        precision: precision.clone(),
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
    let mut tensors: Vec<(String, bool, TensorData)> = Vec::new();
    let mut args = Vec::new();
    let mut scalars = HashMap::new();
    for param in &definition.params {
        match &param.ty {
            Ty::Tensor(shaped) => {
                let shape = shaped
                    .axes
                    .iter()
                    .map(|axis| match axis {
                        Extent::Semantic(sym) => sym
                            .eval(&|n| workload.shapes.get(n).copied())
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or_else(|| format!("{}: unresolved extent {sym}", param.name)),
                        Extent::Structural(_) => {
                            Err(format!("{}: structural entry extent", param.name))
                        }
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let elem = match &shaped.elem {
                    Elem::Param(p) => workload
                        .elems
                        .get(p)
                        .ok_or_else(|| format!("unbound element {p}"))?,
                    concrete => concrete,
                };
                let mut tensor = match elem {
                    Elem::Dtype(dtype) => TensorData::random_dense(&mut rng, *dtype, shape),
                    Elem::Repr(name) => TensorData::random_packed(
                        &mut rng,
                        repr::lookup(name).ok_or("unknown representation")?,
                        shape,
                    ),
                    Elem::Param(p) => return Err(format!("element {p} is not concrete")),
                };
                if let Some((_, values)) = case.contents.iter().find(|(n, _)| *n == param.name) {
                    for (flat, value) in values.iter().enumerate() {
                        tensor.set(flat, *value);
                    }
                }
                args.push(Arg::Tensor(tensors.len()));
                tensors.push((param.name.clone(), param.mode != Mode::In, tensor));
            }
            Ty::Scalar(_) | Ty::Index(_) => {
                let (_, value) = case
                    .scalars
                    .iter()
                    .find(|(n, _)| *n == param.name)
                    .ok_or_else(|| format!("case binds no scalar {}", param.name))?;
                args.push(Arg::Scalar(*value));
                scalars.insert(param.name.clone(), *value);
            }
            other => {
                return Err(format!(
                    "{}: unsupported entry parameter type {other}",
                    param.name
                ))
            }
        }
    }

    if case.raw {
        let words = tensors
            .iter()
            .position(|(n, _, _)| n == "data")
            .ok_or("raw case has no `data`")?;
        let count = tensors[words].2.shape().iter().product::<usize>();
        let raw: Vec<u32> = (0..count).map(|_| rng.next() as u32).collect();
        for (flat, word) in raw.iter().enumerate() {
            tensors[words].2.set(flat, f64::from(*word));
        }
        let bytes: Vec<u8> = raw.iter().flat_map(|w| w.to_le_bytes()).collect();
        let halves = tensors
            .iter()
            .position(|(n, _, _)| n == "halves")
            .ok_or("raw case has no `halves`")?;
        for flat in 0..tensors[halves].2.shape().iter().product::<usize>() {
            let half = u16::from_le_bytes([bytes[2 * flat], bytes[2 * flat + 1]]);
            tensors[halves]
                .2
                .set(flat, f64::from(seismic_lang::numeric::f16_to_f32(half)));
        }
    }

    let mut interpreter = Interpreter::new(program);
    interpreter.partitioner = Box::new(Uniform(3));
    interpreter.tensors = tensors.iter().map(|(_, _, t)| t.clone()).collect();
    interpreter
        .run(case.entry, &args, &workload)
        .map_err(|e| format!("interpreter: {e}"))?;

    let shapes: HashMap<String, i64> = workload.shapes.clone().into_iter().collect();
    let elems: HashMap<String, Elem> = workload.elems.clone().into_iter().collect();
    let mut plan = PlanCompiler::new(
        device,
        program,
        Settings {
            precision: precision.clone(),
            ..Settings::default()
        },
    )
    .compile_entry(case.entry, &shapes, &elems)?;
    plan.kernel().map_err(|e| format!("compile: {e}"))?;
    let mut buffers = HashMap::new();
    for (name, _, tensor) in &tensors {
        let planes: Vec<String> = match tensor {
            TensorData::Dense { .. } => vec![String::new()],
            TensorData::Packed { repr, .. } => {
                repr.planes().iter().map(|p| p.name.to_string()).collect()
            }
        };
        for (plane, bytes) in planes.into_iter().zip(tensor.device_bytes()) {
            buffers.insert(
                (name.clone(), plane),
                device
                    .buffer_from(&bytes)
                    .map_err(|e| format!("upload {name}: {e}"))?,
            );
        }
    }
    let bound = Bound { buffers, scalars };
    plan.execute(&bound).map_err(|e| format!("execute: {e}"))?;

    let mut worst = 0.0f64;
    for (index, (name, written, tensor)) in tensors.iter().enumerate() {
        if !written {
            continue;
        }
        let TensorData::Dense { dtype, data, .. } = tensor else {
            return Err(format!("{name}: packed output"));
        };
        let mut bytes = vec![0u8; data.len() * dtype.bytes() as usize];
        bound.buffers[&(name.clone(), String::new())].read(&mut bytes)?;
        let mut actual = tensor.clone();
        actual.load_device_bytes(&bytes);
        let expected = &interpreter.tensors[index];
        let strict =
            dtype.is_int() || *dtype == DType::Bool || case.exact || precision == PrecisionPolicy::Exact;
        let tolerance = if strict {
            Tolerance::EXACT
        } else {
            Tolerance { absolute: Limit::new(2e-4)?, relative: Limit::new(2e-3)?, relative_floor: Limit::ZERO, ulps: None }
        };
        let reference: Vec<f64> = (0..data.len()).map(|flat| expected.get(flat)).collect();
        let candidate: Vec<f64> = (0..data.len()).map(|flat| actual.get(flat)).collect();
        let comparison = compare_dense(&reference, &candidate, *dtype, Some(tolerance), SpecialPolicy::PRESERVE)?;
        worst = worst.max(comparison.metrics.maximum_absolute);
        if comparison.beyond_tolerance != 0 {
            let flat = comparison.worst_element.unwrap_or(0);
            let (want, got) = (expected.get(flat), actual.get(flat));
            return Err(format!(
                "mismatch: {name} {}/{} elements, max abs {:.3e}, max rel {:.3e}, max ulps {}, worst [{flat}] expected {want} {} {got}",
                comparison.beyond_tolerance, data.len(), comparison.metrics.maximum_absolute, comparison.metrics.maximum_relative, comparison.metrics.maximum_ulps, device.backend()
            ));
        }
    }
    Ok(worst)
}

/// Run the case table on `device`. `precision` overrides each case's own modes.
fn sweep(device: &Device, precision: Option<PrecisionPolicy>) {
    let program = seismic_std::program().expect("seismic-std checks");
    // `KERNELS=<substring>` restricts the sweep to matching labels; `KERNELS_SKIP` excludes them.
    let (filter, skip) = (
        std::env::var("KERNELS").ok(),
        std::env::var("KERNELS_SKIP").ok(),
    );
    let all: Vec<Case> = cases()
        .into_iter()
        .chain(packed_cases())
        .filter(|c| {
            filter.as_ref().is_none_or(|f| c.label.contains(f.as_str()))
                && !skip.as_ref().is_some_and(|s| c.label.contains(s.as_str()))
        })
        .collect();
    let outcomes: Vec<(String, Result<f64, String>)> = all
        .iter()
        .enumerate()
        .flat_map(|(seed, case)| {
            let modes: Vec<PrecisionPolicy> =
                precision.clone().map_or_else(|| case.modes.to_vec(), |only| vec![only]);
            modes
                .into_iter()
                .map(move |precision| (seed, case, precision))
        })
        .map(|(seed, case, precision)| {
            let start = std::time::Instant::now();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&program, device, case, precision.clone(), seed as u64)
            }))
            .unwrap_or_else(|panic| {
                let text = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()));
                Err(format!(
                    "panic: {}",
                    text.unwrap_or_else(|| "non-string payload".into())
                ))
            });
            eprintln!(
                "{} {precision:?}: {:.1}s",
                case.label,
                start.elapsed().as_secs_f64()
            );
            (format!("{} [{precision:?}]", case.label), outcome)
        })
        .collect();
    println!(
        "\n{:<36} outcome on {}",
        "kernel [precision]",
        device.backend()
    );
    for (label, outcome) in &outcomes {
        match outcome {
            Ok(worst) => println!("{label:<36} ok (max abs diff {worst:.3e})"),
            // Native compiler output: keep the first line and the error lines, not warnings.
            Err(error) => println!(
                "{label:<36} FAIL {}",
                error
                    .lines()
                    .enumerate()
                    .filter(|(i, l)| *i == 0 || l.contains("error:"))
                    .map(|(_, l)| l.trim())
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
        }
    }
    let failed = outcomes.iter().filter(|(_, o)| o.is_err()).count();
    assert_eq!(failed, 0, "{failed} of {} kernels failed", outcomes.len());
}

#[test]
#[ignore = "requires a Metal device"]
fn metal_matches_interpreter() {
    sweep(&Device::metal().expect("Metal device"), None);
}

/// Exact precision on the CPU: every case is bit-identical to the interpreter. Cases whose
/// Metal selection uses Metal lowerings select portable bodies here.
#[test]
fn cpu_matches_interpreter() {
    sweep(&Device::cpu().expect("CPU device"), Some(PrecisionPolicy::Exact));
}
