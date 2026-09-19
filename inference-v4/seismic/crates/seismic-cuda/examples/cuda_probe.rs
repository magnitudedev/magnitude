//! Native timing probe (needs an NVIDIA driver and device): one std entry at a stated
//! workload, selected on the CUDA backend, compiled and launched repeatedly; prints the
//! selection estimate, every launch's geometry and the CUDA event interval of every launch.
//! This is the instrument behind the estimate coefficients in `mapping/estimate.rs`.
//!
//! usage: cuda_probe ENTRY SHAPES ELEMENTS [SCALARS] [REPEATS]
//!   e.g. cuda_probe linear M=1,N=9728,K=2560 T=bf16,U=q4g64,V=bf16 - 20
#[path = "support/cases.rs"]
mod cases;

use seismic_compiler::selection::{select, Budget};
use seismic_cuda::mapping::Cuda;
use seismic_cuda::{Buffer, Device};
use seismic_lang::family::Workload;
use seismic_lang::interp::{Rng, TensorData};
use seismic_lang::repr;
use seismic_lang::types::{Elem, Extent, Ty};
use std::collections::HashMap;

fn pairs(text: &str) -> Result<Vec<(String, String)>, String> {
    text.split(',').filter(|p| !p.is_empty() && *p != "-").map(|p| p.split_once('=').map(|(n, v)| (n.to_string(), v.to_string())).ok_or_else(|| format!("`{p}` is not NAME=VALUE"))).collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: cuda_probe ENTRY SHAPES ELEMENTS [SCALARS] [REPEATS]".into());
    }
    let entry = args[1].as_str();
    let workload = Workload {
        shapes: pairs(&args[2])?.into_iter().map(|(n, v)| Ok((n, v.parse::<i64>().map_err(|e| e.to_string())?))).collect::<Result<_, String>>()?,
        elems: pairs(&args[3])?.into_iter().map(|(n, e)| Ok((n, cases::element(&e)?))).collect::<Result<_, String>>()?,
        numerics: cases::numerics()?,
    };
    let scalars: HashMap<String, f64> = pairs(args.get(4).map_or("", String::as_str))?.into_iter().map(|(n, v)| Ok((n, v.parse::<f64>().map_err(|e| e.to_string())?))).collect::<Result<_, String>>()?;
    let repeats: usize = args.get(5).map_or(Ok(10), |r| r.parse()).map_err(|e| format!("repeats: {e}"))?;

    let program = cases::program()?;
    let device = Device::open(0)?;
    let backend = Cuda::from_device(&device.info).map_err(|e| e.to_string())?;
    let definition = cases::abi(&program, entry)?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut uploaded: HashMap<(String, String), Buffer> = HashMap::new();
    for param in &definition.params {
        let Ty::Tensor(shaped) = &param.ty else { continue };
        let shape = shaped.axes.iter().map(|axis| match axis {
            Extent::Semantic(sym) => sym.eval(&|n| workload.shapes.get(n).copied()).and_then(|v| usize::try_from(v).ok()).ok_or_else(|| format!("{}: unresolved extent {sym}", param.name)),
            Extent::Structural(_) => Err(format!("{}: structural entry extent", param.name)),
        }).collect::<Result<Vec<_>, String>>()?;
        let elem = match &shaped.elem {
            Elem::Param(p) => workload.elems.get(p).ok_or_else(|| format!("unbound element {p}"))?,
            concrete => concrete,
        };
        let tensor = match elem {
            Elem::Dtype(dtype) => TensorData::random_dense(&mut rng, *dtype, shape),
            Elem::Repr(name) => TensorData::random_packed(&mut rng, repr::lookup(name).ok_or("unknown representation")?, shape),
            Elem::Param(p) => return Err(format!("element {p} is not concrete")),
        };
        let planes: Vec<String> = match &tensor {
            TensorData::Dense { .. } => vec![String::new()],
            TensorData::Packed { repr, .. } => repr.planes().iter().map(|p| p.name.to_string()).collect(),
        };
        for (plane, bytes) in planes.into_iter().zip(tensor.device_bytes()) {
            uploaded.insert((param.name.clone(), plane), device.buffer_from(&bytes)?);
        }
    }
    let selected = select(&program, entry, &workload, &backend, Budget::default()).map_err(|e| format!("select: {e}"))?;
    println!("{entry} {} {}: {:?}, estimate {} ns (seed {} ns), model {}", args[2], args[3], selected.status, selected.estimate, selected.seed_estimate, seismic_cuda::mapping::IDENTITY);
    let geometry: Vec<String> = selected.execution.phases.iter().map(|p| format!("{}x{} scratch {}", p.dispatch().groups, p.dispatch().threads_per_group, p.program().scratch_bytes)).collect();
    let mut sequence = device.compile_launches(selected.execution)?;
    println!("  device {:?}", device.info);
    for (ordinal, (_, native)) in sequence.realizations().enumerate() {
        println!("  launch {ordinal} native {native:?}");
    }
    let buffers = sequence.buffers().iter()
        .map(|spec| uploaded.get(&(spec.parameter.clone(), spec.plane.clone())).cloned().ok_or_else(|| format!("no tensor bound to {}.{}", spec.parameter, spec.plane)))
        .collect::<Result<Vec<_>, String>>()?;
    let values = sequence.scalars().iter().map(|s| scalars.get(&s.name).copied().ok_or_else(|| format!("unbound scalar {}", s.name))).collect::<Result<Vec<_>, String>>()?;
    let mut best = vec![f64::INFINITY; geometry.len()];
    let mut wall = f64::INFINITY;
    for _ in 0..repeats {
        let start = std::time::Instant::now();
        let seconds = sequence.execute_launches(&buffers, &values)?;
        wall = wall.min(start.elapsed().as_secs_f64());
        for (best, seconds) in best.iter_mut().zip(seconds) {
            *best = best.min(seconds);
        }
    }
    for (ordinal, (geometry, seconds)) in geometry.iter().zip(&best).enumerate() {
        println!("  launch {ordinal}: blocks x threads {geometry}: {:.1} us", seconds * 1e6);
    }
    println!("  device total {:.1} us, host wall {:.1} us (best of {repeats})", best.iter().sum::<f64>() * 1e6, wall * 1e6);
    Ok(())
}
