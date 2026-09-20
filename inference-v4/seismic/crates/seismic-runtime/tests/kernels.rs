//! Gate G3: every linked seismic-std kernel the Qwen path uses, selected and run on
//! Metal, agrees with the reference interpreter. The same case table runs on the CPU device
//! under exact precision, where every output must be bit-identical to the interpreter.
use seismic_lang::interp::{Arg, Bindings as InterpBindings, Interpreter, Rng, TensorData};
use seismic_lang::precision::{compare_dense, Limit, PrecisionPolicy, SpecialPolicy, Tolerance};
use seismic_lang::repr;
use seismic_lang::sir::Mode;
use seismic_lang::sir::{Definition, Program};
use seismic_lang::types::{DType, Elem, ExtentExpr, ValueType};
use seismic_runtime::plan::{Bindings, PlanCompiler, Settings};
use seismic_runtime::{Buffer, Device, Workload};
use seismic_std::sweep::{cases, packed_cases, Case};
use std::collections::HashMap;

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
        extents: Default::default(),
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
    let mut tensors: Vec<(String, bool, TensorData)> = Vec::new();
    let mut args = Vec::new();
    let mut scalars = HashMap::new();
    for param in &definition.params {
        match &param.ty {
            ValueType::Tensor(shaped) => {
                let shape = shaped
                    .axes
                    .iter()
                    .map(|axis| match axis {
                        ExtentExpr::Sym(sym) => sym
                            .eval(&|n| workload.shapes.get(n).copied())
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or_else(|| format!("{}: unresolved extent {sym}", param.name)),
                        ExtentExpr::Static(value) => usize::try_from(*value)
                            .map_err(|_| format!("{}: extent exceeds address range", param.name)),
                        ExtentExpr::Runtime(_) => {
                            Err(format!("{}: runtime-dependent entry extent", param.name))
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
                        tensor.set(flat, *value).map_err(|e| e.to_string())?;
                    }
                }
                args.push(Arg::Tensor(tensors.len()));
                tensors.push((param.name.clone(), param.mode != Mode::In, tensor));
            }
            ValueType::Scalar(_) | ValueType::Index { .. } => {
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
            tensors[words]
                .2
                .set(flat, f64::from(*word))
                .map_err(|e| e.to_string())?;
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
                .set(flat, f64::from(seismic_lang::numeric::f16_to_f32(half)))
                .map_err(|e| e.to_string())?;
        }
    }

    let mut interpreter = Interpreter::new(program);
    interpreter.tensors = tensors.iter().map(|(_, _, t)| t.clone()).collect();
    let interpreter_bindings = InterpBindings {
        shapes: workload.shapes.clone(),
        elems: workload.elems.clone(),
    };
    interpreter
        .run(case.entry, &args, &interpreter_bindings)
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
        let strict = dtype.is_int()
            || *dtype == DType::Bool
            || case.exact
            || precision == PrecisionPolicy::Exact;
        let tolerance = if strict {
            Tolerance::EXACT
        } else {
            Tolerance {
                absolute: Limit::new(2e-4)?,
                relative: Limit::new(2e-3)?,
                relative_floor: Limit::ZERO,
                ulps: None,
            }
        };
        let reference: Vec<f64> = (0..data.len()).map(|flat| expected.get(flat)).collect();
        let candidate: Vec<f64> = (0..data.len()).map(|flat| actual.get(flat)).collect();
        let comparison = compare_dense(
            &reference,
            &candidate,
            *dtype,
            Some(tolerance),
            SpecialPolicy::PRESERVE,
        )?;
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
            let modes: Vec<PrecisionPolicy> = precision
                .clone()
                .map_or_else(|| case.modes.to_vec(), |only| vec![only]);
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
    sweep(
        &Device::cpu().expect("CPU device"),
        Some(PrecisionPolicy::Exact),
    );
}
