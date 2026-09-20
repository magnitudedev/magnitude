//! Native check (needs an NVIDIA driver and device): every std kernel case, selected on
//! the CUDA backend with the queried device limits, realized to PTX, compiled by the
//! driver and executed through this crate's runtime, against the reference interpreter
//! on the same host. Exact precision by default (`PRECISION=unconstrained` to explore).
//!
//! Integer and boolean outputs compare exactly. Float outputs are expected bit-exact; the
//! bound `2e-4 + 2e-3 * |reference|` admits the documented difference of the bundled PTX
//! libm port (`exp` within 2 ULP). The table reports the max absolute difference and the
//! number of differing elements, so "within tolerance" is never reported as "exact".
#[path = "support/cases.rs"]
mod cases;

use seismic_compiler::selection::{select, Budget};
use seismic_cuda::mapping::Cuda;
use seismic_cuda::{Buffer, Device};
use seismic_lang::precision::{compare_dense, Limit, PrecisionPolicy, SpecialPolicy, Tolerance};
use seismic_lang::interp::{Arg, Interpreter, Rng, TensorData, Uniform};
use seismic_lang::repr;
use seismic_lang::sir::Program;
use seismic_lang::sir::Mode;
use seismic_lang::types::{DType, Elem, Extent, Ty};
use std::collections::HashMap;

struct Agreement {
    worst: f64,
    differing: usize,
    elements: usize,
    launches: usize,
}

fn run(program: &Program, device: &Device, backend: &Cuda, case: &cases::Case, precision: PrecisionPolicy, seed: u64) -> Result<Agreement, String> {
    let definition = cases::abi(program, case.entry)?;
    let workload = cases::workload(case, precision)?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
    let mut tensors: Vec<(String, bool, TensorData)> = Vec::new();
    let mut args = Vec::new();
    let mut scalars = HashMap::new();
    for param in &definition.params {
        match &param.ty {
            Ty::Tensor(shaped) | Ty::View(shaped) => {
                let shape = shaped.axes.iter().map(|axis| match axis {
                    Extent::Semantic(sym) => sym.eval(&|n| workload.shapes.get(n).copied()).and_then(|v| usize::try_from(v).ok()).ok_or_else(|| format!("{}: unresolved extent {sym}", param.name)),
                    Extent::Structural(_) => Err(format!("{}: structural entry extent", param.name)),
                }).collect::<Result<Vec<_>, String>>()?;
                let elem = match &shaped.elem {
                    Elem::Param(p) => workload.elems.get(p).ok_or_else(|| format!("unbound element {p}"))?,
                    concrete => concrete,
                };
                let mut tensor = match elem {
                    Elem::Dtype(dtype) => TensorData::random_dense(&mut rng, *dtype, shape),
                    Elem::Repr(name) => TensorData::random_packed(&mut rng, repr::lookup(name).ok_or("unknown representation")?, shape),
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
                let (_, value) = case.scalars.iter().find(|(n, _)| *n == param.name).ok_or_else(|| format!("case binds no scalar {}", param.name))?;
                args.push(Arg::Scalar(*value));
                scalars.insert(param.name.clone(), *value);
            }
            other => return Err(format!("{}: unsupported entry parameter type {other}", param.name)),
        }
    }
    if case.raw {
        let words = tensors.iter().position(|(n, _, _)| n == "data").ok_or("raw case has no `data`")?;
        let count = tensors[words].2.shape().iter().product::<usize>();
        let raw: Vec<u32> = (0..count).map(|_| rng.next() as u32).collect();
        for (flat, word) in raw.iter().enumerate() {
            tensors[words].2.set(flat, f64::from(*word));
        }
        let bytes: Vec<u8> = raw.iter().flat_map(|w| w.to_le_bytes()).collect();
        let halves = tensors.iter().position(|(n, _, _)| n == "halves").ok_or("raw case has no `halves`")?;
        for flat in 0..tensors[halves].2.shape().iter().product::<usize>() {
            let half = u16::from_le_bytes([bytes[2 * flat], bytes[2 * flat + 1]]);
            tensors[halves].2.set(flat, f64::from(seismic_lang::numeric::f16_to_f32(half)));
        }
    }

    let mut interpreter = Interpreter::new(program);
    interpreter.partitioner = Box::new(Uniform(3));
    interpreter.tensors = tensors.iter().map(|(_, _, t)| t.clone()).collect();
    interpreter.run(case.entry, &args, &workload).map_err(|e| format!("interpreter: {e}"))?;

    let selected = select(program, case.entry, &workload, backend, Budget::default()).map_err(|e| format!("select: {e}"))?;
    let launches = selected.execution.phases.len();
    let mut sequence = device.compile_launches(selected.execution).map_err(|e| format!("compile: {e}"))?;

    let mut uploaded: HashMap<(String, String), Buffer> = HashMap::new();
    for (name, _, tensor) in &tensors {
        let planes: Vec<String> = match tensor {
            TensorData::Dense { .. } => vec![String::new()],
            TensorData::Packed { repr, .. } => repr.planes().iter().map(|p| p.name.to_string()).collect(),
        };
        for (plane, bytes) in planes.into_iter().zip(tensor.device_bytes()) {
            uploaded.insert((name.clone(), plane), device.buffer_from(&bytes).map_err(|e| format!("upload {name}: {e}"))?);
        }
    }
    let buffers = sequence.buffers().iter()
        .map(|spec| uploaded.get(&(spec.parameter.clone(), spec.plane.clone())).cloned().ok_or_else(|| format!("no tensor bound to {}.{}", spec.parameter, spec.plane)))
        .collect::<Result<Vec<_>, String>>()?;
    let values = sequence.scalars().iter()
        .map(|s| scalars.get(&s.name).copied().ok_or_else(|| format!("unbound scalar {}", s.name)))
        .collect::<Result<Vec<_>, String>>()?;
    sequence.execute(&buffers, &values, false).map_err(|e| format!("execute: {e}"))?;

    let mut agreement = Agreement { worst: 0.0, differing: 0, elements: 0, launches };
    for (index, (name, written, tensor)) in tensors.iter().enumerate() {
        if !written {
            continue;
        }
        let TensorData::Dense { dtype, data, .. } = tensor else { return Err(format!("{name}: packed output")) };
        let mut bytes = vec![0u8; data.len() * dtype.bytes() as usize];
        uploaded.get(&(name.clone(), String::new())).ok_or_else(|| format!("{name}: no output buffer"))?.read(&mut bytes)?;
        let mut actual = tensor.clone();
        actual.load_device_bytes(&bytes);
        let expected = &interpreter.tensors[index];
        let strict = dtype.is_int() || *dtype == DType::Bool;
        agreement.elements += data.len();
        let tolerance = if strict { Tolerance::EXACT } else { Tolerance { absolute: Limit::new(2e-4)?, relative: Limit::new(2e-3)?, relative_floor: Limit::ZERO, ulps: None } };
        let reference: Vec<f64> = (0..data.len()).map(|flat| expected.get(flat)).collect();
        let candidate: Vec<f64> = (0..data.len()).map(|flat| actual.get(flat)).collect();
        let comparison = compare_dense(&reference, &candidate, *dtype, Some(tolerance), SpecialPolicy::PRESERVE)?;
        agreement.differing += comparison.metrics.differing as usize;
        agreement.worst = agreement.worst.max(comparison.metrics.maximum_absolute);
        if comparison.beyond_tolerance != 0 {
            let flat = comparison.worst_element.unwrap_or(0);
            return Err(format!("mismatch: {name} {}/{} elements beyond tolerance, max abs {:.3e}, max rel {:.3e}, max ulps {}, worst [{flat}] expected {} CUDA {}", comparison.beyond_tolerance, data.len(), comparison.metrics.maximum_absolute, comparison.metrics.maximum_relative, comparison.metrics.maximum_ulps, expected.get(flat), actual.get(flat)));
        }
    }
    Ok(agreement)
}

fn main() -> Result<(), String> {
    let program = cases::program()?;
    let precision = cases::precision()?;
    let device = Device::open(0)?;
    println!("device: {:?}", device.info);
    let backend = Cuda::from_device(&device.info).map_err(|e| e.to_string())?;
    println!("limits: {:?}; estimate model {}", backend.limits(), seismic_cuda::mapping::IDENTITY);
    let all = cases::selected();
    let mut failed = 0usize;
    let mut worst = 0.0f64;
    let mut diagnostics = Vec::new();
    println!("\n{:<30} outcome ({precision:?} precision)", "kernel");
    for (seed, case) in all.iter().enumerate() {
        match cases::guarded(|| run(&program, &device, &backend, case, precision.clone(), seed as u64)) {
            Ok(a) if a.differing == 0 => println!("{:<30} ok    bit-exact ({} elements, {} launch(es))", case.label, a.elements, a.launches),
            Ok(a) => {
                worst = worst.max(a.worst);
                println!("{:<30} ok    within tolerance: {}/{} elements differ, max abs diff {:.3e} ({} launch(es))", case.label, a.differing, a.elements, a.worst, a.launches);
            }
            Err(error) => {
                failed += 1;
                println!("{:<30} FAIL  {}", case.label, error.lines().next().unwrap_or(""));
                if error.lines().count() > 1 {
                    diagnostics.push(format!("--- {}\n{error}", case.label));
                }
            }
        }
    }
    // Driver compilation logs, in full, after the table.
    diagnostics.iter().for_each(|text| println!("{text}"));
    println!("\n{} of {} kernels agree with the interpreter; max abs diff over agreeing kernels {worst:.3e}", all.len() - failed, all.len());
    if failed > 0 { Err(format!("{failed} kernels failed")) } else { Ok(()) }
}
