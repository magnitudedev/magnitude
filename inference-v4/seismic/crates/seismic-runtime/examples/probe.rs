//! Timing probe for one exported entry on Metal: selects, compiles, binds random tensors and
//! reports the selected witness, its estimate, and measured GPU time per dispatch (best of
//! several profiled runs) and per invocation inside one batched command buffer.
//!
//! usage: probe <source dir>... -- <entry> <K=V,...> <NAME=elem,...|-> [scalar=v,...|-] [name=v:v:...;...|-] [exact]
use seismic_lang::repr;
use seismic_lang::interp::{Rng, TensorData};
use seismic_lang::program::{collect_files, compile};
use seismic_lang::syntax::ast::Mode;
use seismic_lang::types::{Extent, Ty};
use seismic_lang::types::{DType, Elem};
use seismic_runtime::plan::{Bindings, PlanCompiler, Settings};
use seismic_runtime::{Buffer, Device};
use std::collections::HashMap;

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

fn pairs(text: &str) -> Vec<(String, String)> {
    text.split(',').filter(|p| !p.is_empty() && *p != "-").filter_map(|p| p.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args.iter().position(|a| a == "--").ok_or("usage: probe <dir>... -- <entry> <shapes> <elems> [scalars] [contents]")?;
    let (dirs, rest) = (&args[..split], &args[split + 1..]);
    let entry = rest.first().ok_or("missing entry")?;
    let shapes: HashMap<String, i64> = pairs(rest.get(1).ok_or("missing shapes")?).into_iter().map(|(k, v)| Ok((k, v.parse::<i64>().map_err(|e| e.to_string())?))).collect::<Result<_, String>>()?;
    let elems: HashMap<String, Elem> = pairs(rest.get(2).map_or("-", String::as_str)).into_iter().map(|(k, v)| {
        let elem = match DType::from_name(&v) {
            Some(d) => Elem::Dtype(d),
            None if repr::lookup(&v).is_some() => Elem::Repr(v.clone()),
            None => return Err(format!("unknown element {v}")),
        };
        Ok((k, elem))
    }).collect::<Result<_, String>>()?;
    let scalars: HashMap<String, f64> = pairs(rest.get(3).map_or("-", String::as_str)).into_iter().map(|(k, v)| Ok((k, v.parse::<f64>().map_err(|e| e.to_string())?))).collect::<Result<_, String>>()?;
    let contents: HashMap<String, Vec<f64>> = rest.get(4).map_or("", String::as_str).split(';').filter_map(|p| p.split_once('=')).map(|(k, v)| (k.to_string(), v.split(':').filter_map(|x| x.parse().ok()).collect())).collect();

    let paths: Vec<std::path::PathBuf> = dirs.iter().map(std::path::PathBuf::from).collect();
    let files = collect_files(&paths).map_err(|e| format!("{e:?}"))?;
    let program = compile(&files, &["metal".into()]).map_err(|errors| errors.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n"))?;
    let family = program.export(entry).ok_or_else(|| format!("no exported entry {entry}"))?;
    let definition = family.bodies.iter().chain(&family.contracts).chain(&family.lowerings).map(|id| program.definition(*id)).find(|d| d.export).ok_or("entry has no exported definition")?;
    let device = Device::metal()?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buffers = HashMap::new();
    for param in &definition.params {
        let Ty::Tensor(shaped) = &param.ty else { continue };
        let shape = shaped.axes.iter().map(|axis| match axis {
            Extent::Semantic(sym) => sym.eval(&|n| shapes.get(n).copied()).and_then(|v| usize::try_from(v).ok()).ok_or_else(|| format!("{}: unresolved extent {sym}", param.name)),
            Extent::Structural(_) => Err(format!("{}: structural entry extent", param.name)),
        }).collect::<Result<Vec<_>, String>>()?;
        let elem = match &shaped.elem {
            Elem::Param(p) => elems.get(p).ok_or_else(|| format!("unbound element {p}"))?,
            concrete => concrete,
        };
        let mut tensor = match elem {
            Elem::Dtype(dtype) => TensorData::random_dense(&mut rng, *dtype, shape),
            Elem::Repr(name) => TensorData::random_packed(&mut rng, repr::lookup(name).ok_or("unknown representation")?, shape),
            Elem::Param(p) => return Err(format!("element {p} is not concrete")),
        };
        if let Some(values) = contents.get(&param.name) {
            let count = match &tensor { TensorData::Dense { data, .. } => data.len(), TensorData::Packed { .. } => 0 };
            for flat in 0..count {
                tensor.set(flat, values[flat % values.len()]);
            }
        }
        let _ = param.mode != Mode::In;
        let planes: Vec<String> = match &tensor {
            TensorData::Dense { .. } => vec![String::new()],
            TensorData::Packed { repr, .. } => repr.planes().iter().map(|p| p.name.to_string()).collect(),
        };
        for (plane, bytes) in planes.into_iter().zip(tensor.device_bytes()) {
            buffers.insert((param.name.clone(), plane), device.buffer_from(&bytes).map_err(|e| format!("upload {}: {e}", param.name))?);
        }
    }
    let bound = Bound { buffers, scalars };
    let numerics = if rest.get(5).is_some_and(|mode| mode == "exact") { seismic_lang::family::Numerics::Exact } else { seismic_lang::family::Numerics::Admitted };
    let mut plan = PlanCompiler::new(&device, &program, Settings { numerics, ..Settings::default() }).compile_entry(entry, &shapes, &elems)?;
    let kernel = plan.kernel().map_err(|e| format!("compile: {e}"))?;
    {
        let kernel = kernel.borrow();
        let selection = kernel.selection();
        println!("estimate {} ns (seed {}), status {:?}", selection.estimate, selection.seed_estimate, selection.status);
        println!("choices {:?}", selection.witness.choices.iter().map(|(o, c)| (o.0, *c)).collect::<Vec<_>>());
        println!("sites {:?}", selection.witness.sites.iter().map(|(s, v)| (s.0, *v)).collect::<Vec<_>>());
        println!("covers {:?}", selection.witness.covers.iter().filter(|(_, c)| c.iter().any(|(a, b)| b - a > 1)).map(|(s, c)| (s.0, c.clone())).collect::<Vec<_>>());
    }
    let mut best: Vec<(String, u64, f64)> = Vec::new();
    for _ in 0..12 {
        let steps = plan.execute_observed(&bound)?;
        let dispatches: Vec<_> = steps.into_iter().flat_map(|s| s.dispatches).collect();
        if best.is_empty() {
            best = dispatches.iter().map(|d| (d.kernel.clone(), d.threadgroups, d.device_seconds)).collect();
        }
        for (slot, d) in best.iter_mut().zip(&dispatches) {
            slot.2 = slot.2.min(d.device_seconds);
        }
    }
    for (kernel, groups, seconds) in &best {
        println!("  {kernel:<36} tg={groups:<6} best {:.1} us", seconds * 1e6);
    }
    println!("profiled sum {:.1} us", best.iter().map(|b| b.2).sum::<f64>() * 1e6);
    let mut batched = f64::INFINITY;
    // Long command buffers keep the GPU at its steady clocks, as a batched forward does.
    for _ in 0..8 {
        let mut submission = plan.prepare(&bound)?;
        for _ in 1..200 {
            submission.append(plan.prepare(&bound)?);
        }
        if let Some(seconds) = submission.execute_batched()?.device_seconds {
            batched = batched.min(seconds / 200.0);
        }
    }
    println!("batched x200: {:.1} us per invocation", batched * 1e6);
    Ok(())
}
