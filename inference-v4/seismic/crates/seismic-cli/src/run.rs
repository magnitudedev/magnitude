//! `seismic lower` and `seismic run`.

use crate::{load_program, options};
use seismic_lang::interp::{Arg, Interpreter, Rng, TensorData};
use seismic_lang::lower::Choice;
use seismic_lang::repr;
use seismic_lang::types::{Elem, Ty};
use seismic_metal::msl;
#[cfg(target_os = "macos")]
use seismic_metal::runtime::{Buffer, Device};
use std::time::Instant;

pub fn lower(args: &[String]) -> Result<(), String> {
    let o = options(args)?;
    let program = load_program(&o)?;
    let name = o.function.as_ref().ok_or("--fn is required")?;
    let lowered = seismic_lang::lower::lower_with(&program, name, &o.target, &o.shapes, &seismic_lang::lower::Options { piece: o.piece })?;
    for sp in seismic_lang::split::splittable(&lowered.body, &lowered.vars) {
        let names: Vec<&str> = sp.carried.iter().map(|v| lowered.vars[*v].name.as_str()).collect();
        eprintln!("splittable: streamed range carrying {}", names.join(", "));
    }
    for s in &lowered.selections {
        let choice = match s.choice {
            Choice::Block(i) => format!("block {i}"),
            Choice::Portable => "portable body".to_string(),
        };
        eprintln!("{}[{}]: {choice}", s.construct, s.shape_args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "));
    }
    if o.target == "cpu" {
        let kernel = seismic_cpu::compile_candidate(&lowered,o.loads)?;
        print!("{}", kernel.ir);
        return Ok(());
    }
    if o.target == "cuda" {
        print!("{}", seismic_cuda::ptx::lower_candidate(&lowered, seismic_realization::ScalarOptions {dispatch:seismic_realization::Dispatch::ParallelRoot,loads:o.loads})?);
        return Ok(());
    }
    if o.target != "metal" { return Err(format!("target `{}` has no printer yet", o.target)); }
    // `lower` prints without a device, so the architectural maximum stands in for a query.
    let emitted = msl::emit_with(&lowered, msl::Config { tile_piece: None, sg_per_tg: o.sg_per_tg, piece: o.piece, per_item: o.per_item, split: o.split, max_threads_per_threadgroup: 1024, max_threadgroup_bytes: 32768 })?;
    print!("{}", emitted.source);
    for l in &emitted.launches {
        eprintln!("launch {}: {} threadgroups x {} threads", l.kernel, l.threadgroups, l.threads_per_threadgroup);
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), String> {
    let o = options(args)?;
    let program = load_program(&o)?;
    let name = o.function.as_ref().ok_or("--fn is required")?;
    let f = program.functions.iter().find(|f| &f.name == name).ok_or_else(|| format!("no function `{name}`"))?.clone();
    let lowered = seismic_lang::lower::lower_with(&program, name, &o.target, &o.shapes, &seismic_lang::lower::Options { piece: o.piece })?;
    for p in &f.shape_params {
        if !o.shapes.contains_key(p) {
            return Err(format!("shape parameter `{p}` is not bound; pass --shape {p}=<value>"));
        }
    }
    let shape_env = |s: &seismic_lang::sym::Sym| -> Result<usize, String> {
        s.eval(&|p| o.shapes.get(p).copied()).map(|v| v as usize).ok_or_else(|| format!("cannot evaluate shape `{s}`"))
    };

    // Inputs: every tensor parameter gets random contents; scalars from --scalar.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut tensors: Vec<(String, TensorData)> = Vec::new();
    let mut interp_args = Vec::new();
    let mut interp = Interpreter::new(&program);
    let mut scalar_schema = Vec::new();
    let mut scalar_values = Vec::new();
    for (pname, ty) in &f.params {
        match ty {
            Ty::Tensor(s) => {
                let shape: Vec<usize> = s.shape.iter().map(shape_env).collect::<Result<_, _>>()?;
                let spec = o.inputs.get(pname).map(String::as_str).unwrap_or("random");
                let data = match (&s.elem, spec) {
                    (Elem::Dtype(d), "random") => TensorData::random_dense(&mut rng, *d, shape),
                    (Elem::Dtype(d), "zeros") => TensorData::dense(*d, shape.clone(), vec![0.0; shape.iter().product()]),
                    // Causal visibility for a [Q, 2] i32 tensor over a history of shape parameter T:
                    // row q sees [0, T - Q + q + 1).
                    (Elem::Dtype(d), "causal") if d.is_int() && shape.len() == 2 && shape[1] == 2 => {
                        let t = *o.shapes.get("T").ok_or("causal visibility needs shape parameter T")? as usize;
                        let q = shape[0];
                        let mut data = Vec::with_capacity(q * 2);
                        for row in 0..q {
                            data.push(0.0);
                            data.push((t - q + row + 1) as f64);
                        }
                        TensorData::dense(*d, shape, data)
                    }
                    (Elem::Dtype(d), other) if other.starts_with("prefix:") => {
                        let n: f64 = other["prefix:".len()..].parse().map_err(|_| "bad prefix length")?;
                        let mut data = Vec::with_capacity(shape.iter().product());
                        for _ in 0..shape[0] {
                            data.push(0.0);
                            data.push(n);
                        }
                        TensorData::dense(*d, shape, data)
                    }
                    (Elem::Repr(r), "random") => TensorData::random_packed(&mut rng, repr::lookup(r).unwrap(), shape),
                    (Elem::Param(p), _) => return Err(format!("parameter `{pname}` has generic element type `{p}`; run a kernel, not a construct")),
                    (_, other) => return Err(format!("unsupported input spec `{other}` for `{pname}`")),
                };
                let id = interp.add_tensor(data.clone());
                interp_args.push(Arg::Tensor(id));
                tensors.push((pname.clone(), data));
            }
            Ty::Scalar(d) => {
                let v = *o.scalars.get(pname).ok_or_else(|| format!("scalar parameter `{pname}` is not bound; pass --scalar {pname}=<value>"))?;
                interp_args.push(Arg::Scalar(v));
                scalar_schema.push(seismic_lang::abi::ScalarParameter::from_lowered(&lowered,pname,*d)?);
                scalar_values.push(v);
            }
            other => return Err(format!("parameter `{pname}` of type {other} cannot be run")),
        }
    }

    let scalar_bytes = seismic_lang::abi::ScalarLayout::natural(&scalar_schema)?.encode(&scalar_values)?;

    // Reference.
    if o.check {
        let t0 = Instant::now();
        interp.run(name, &interp_args, &o.shapes)?;
        eprintln!("interpreter: {:.3} s", t0.elapsed().as_secs_f64());
    }

    // Lower, emit, compile.

    for s in &lowered.selections {
        let choice = match s.choice {
            Choice::Block(i) => format!("block {i}"),
            Choice::Portable => "portable body".to_string(),
        };
        eprintln!("{}[{}]: {choice}", s.construct, s.shape_args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "));
    }
    if o.target == "cpu" { return run_cpu(&o, &program, name, &lowered, &tensors, &interp); }
    if o.target == "cuda" { return run_cuda(&o, &program, name, &lowered, &tensors, &interp); }
    if o.target != "metal" { return Err(format!("target `{}` has no runtime yet", o.target)); }
    #[cfg(target_os = "macos")]
    { run_metal(&o, &program, name, &lowered, &tensors, &interp, &scalar_bytes) }
    #[cfg(not(target_os = "macos"))]
    { Err("Metal execution requires macOS".into()) }
}

#[cfg(target_os = "macos")]
fn run_metal(o: &crate::Options, program: &seismic_lang::program::Program, name: &str,
    lowered: &seismic_lang::lower::Lowered, tensors: &[(String, TensorData)], interp: &Interpreter<'_>, scalar_bytes: &[u8]) -> Result<(), String> {
    let device = Device::open()?;
    let info = device.info();
    eprintln!("device: {} (unified memory: {})", info.name, info.unified_memory);
    let emitted = msl::emit_with(&lowered, msl::Config { tile_piece: None, sg_per_tg: o.sg_per_tg, piece: o.piece, per_item: o.per_item, split: o.split, max_threads_per_threadgroup: info.max_threads_per_threadgroup as i64, max_threadgroup_bytes: info.max_threadgroup_bytes as i64 })?;
    let t1 = Instant::now();
    let pipeline = device.compile(emitted)?;
    eprintln!("metal compile: {:.3} s", t1.elapsed().as_secs_f64());

    // Buffers in ABI order.
    let mut buffers: Vec<Buffer> = Vec::new();
    let mut total_bytes = 0usize;
    for slot in &pipeline.emitted.buffers {
        let (_, data) = tensors.iter().find(|(n, _)| *n == slot.parameter).unwrap();
        let parts = data.device_bytes();
        let part = match slot.plane.as_str() {
            "" | "words" => &parts[0],
            "scale" => &parts[1],
            _ => &parts[2],
        };
        total_bytes += part.len();
        buffers.push(device.buffer_from(part)?);
    }
    let refs: Vec<&Buffer> = buffers.iter().collect();

    // Run once, compare, then time.
    let gpu = device.run(&pipeline, &refs, &scalar_bytes, 1)?;
    let mut worst_abs = 0f64;
    let mut worst_rel = 0f64;
    for (pname, data) in tensors.iter() {
        if !o.check {
            break;
        }
        let TensorData::Dense { .. } = data else { continue };
        let index = pipeline.emitted.buffers.iter().position(|s| s.parameter == *pname && s.plane.is_empty()).unwrap();
        let bytes = buffers[index].read(data.bytes());
        let mut got = data.clone();
        got.load_device_bytes(&bytes);
        let id = tensors.iter().position(|(n, _)| n == pname).unwrap();
        let expect = &interp.tensors[id];
        let n = data.shape().iter().product::<usize>();
        let (mut max_abs, mut max_rel, mut at) = (0f64, 0f64, 0usize);
        let mut nans = 0usize;
        for i in 0..n {
            let e = expect.get(i);
            let g = got.get(i);
            if !g.is_finite() || !e.is_finite() {
                nans += 1;
                max_rel = f64::INFINITY;
                continue;
            }
            let abs = (e - g).abs();
            let rel = abs / e.abs().max(1e-6);
            if abs > max_abs {
                max_abs = abs;
                at = i;
            }
            max_rel = max_rel.max(rel.min(abs * 1e3));
        }
        if nans > 0 {
            eprintln!("{pname}: {nans} non-finite value(s)");
        }
        let changed = (0..n).any(|i| expect.get(i) != data.get(i));
        if changed {
            eprintln!("{pname}: max abs err {max_abs:.3e} (at {at}: expected {:.6}, got {:.6}), max rel err {max_rel:.3e}", expect.get(at), got.get(at));
            worst_abs = worst_abs.max(max_abs);
            worst_rel = worst_rel.max(max_rel);
        }
    }
    let mut times = Vec::new();
    for _ in 0..o.iters {
        times.push(device.run(&pipeline, &refs, &scalar_bytes, o.repeat)? / o.repeat as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    eprintln!("first run: {:.3} ms; median of {}: {:.3} ms; min {:.3} ms", gpu * 1e3, o.iters, median * 1e3, times[0] * 1e3);
    eprintln!("bound buffer storage: {:.1} MB (allocation size, not measured traffic)", total_bytes as f64 / 1e6);
    // The former byte/FLOP shortcut did not establish a roofline. Do not publish a
    // percentage until checked region/operation accounts and a compatible profile exist.
    crate::account::print_work(&program, name, &o.shapes, o.analysis_steps)?;
    let ok = worst_rel <= 2e-2;
    println!("{}: {name} on {}: max rel err {worst_rel:.2e}, {:.3} ms", if !o.check { "unchecked" } else if ok { "ok" } else { "MISMATCH" }, o.target, median * 1e3);
    if ok { Ok(()) } else { Err("device result differs from the reference".into()) }
}

#[cfg(target_os = "macos")]
pub fn calibrate() -> Result<(), String> {
    let device = Device::open()?;
    let info = device.info();
    for mb in [64usize, 256, 1024] {
        let gbps = seismic_metal::runtime::calibrate_bandwidth(&device, mb << 20)?;
        println!("{}: streaming read of {mb} MB: {gbps:.1} GB/s", info.name);
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn calibrate() -> Result<(), String> { Err("Metal calibration requires macOS".into()) }

fn run_cpu(o: &crate::Options, program: &seismic_lang::program::Program, name: &str,
    lowered: &seismic_lang::lower::Lowered, tensors: &[(String, TensorData)], reference: &Interpreter<'_>) -> Result<(), String> {
    if o.iters == 0 || o.repeat == 0 { return Err("--iters and --repeat must be positive".into()); }
    let start = Instant::now();
    let mut kernel = seismic_cpu::compile_candidate(lowered,o.loads)?;
    eprintln!("CPU native compile: {:.3} s; scratch {} bytes; scalar realization (automatic selection pending)", start.elapsed().as_secs_f64(), kernel.scratch_bytes());
    let specs = kernel.buffers().to_vec();
    let mut buffers = scalar_buffers(&specs, tensors)?;
    let scalars = kernel.scalars().iter().map(|parameter| o.scalars.get(&parameter.name).copied().ok_or_else(||format!("unbound scalar `{}`",parameter.name))).collect::<Result<Vec<_>,_>>()?;
    kernel.run(&mut buffers.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(), &scalars)?;
    if o.check { check_scalar_buffers(&specs, &buffers, tensors, reference)?; }
    let mut times = Vec::with_capacity(o.iters);
    let mut bindings = buffers.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
    for _ in 0..o.iters {
        let start = Instant::now();
        for _ in 0..o.repeat { kernel.run(&mut bindings, &scalars)?; }
        times.push(start.elapsed().as_secs_f64() / o.repeat as f64);
    }
    times.sort_by(f64::total_cmp);
    crate::account::print_work(program, name, &o.shapes, o.analysis_steps)?;
    println!("{}: {name} on CPU {}: median {:.3} ms including invocation, scratch {} bytes", if o.check {"ok"} else {"unchecked"}, std::env::consts::ARCH, times[times.len()/2]*1000.0,kernel.scratch_bytes());
    Ok(())
}

fn scalar_buffers(specs: &[seismic_realization::BufferSpec], tensors: &[(String, TensorData)]) -> Result<Vec<Vec<u8>>, String> {
    specs.iter().map(|slot| {
        let (_, tensor) = tensors.iter().find(|(n,_)| n == &slot.parameter).ok_or("unresolved parameter")?;
        let parts = tensor.device_bytes();
        let index = match slot.plane.as_str() { "" | "words" => 0, "scale" => 1, "bias" => 2, _ => return Err("unknown storage plane".into()) };
        parts.get(index).cloned().ok_or_else(|| "missing storage plane".into())
    }).collect()
}
fn check_scalar_buffers(specs: &[seismic_realization::BufferSpec], buffers: &[Vec<u8>], tensors: &[(String, TensorData)], reference: &Interpreter<'_>) -> Result<(), String> {
    for (slot, bytes) in specs.iter().zip(buffers).filter(|(s,_)| s.plane.is_empty()) {
        let id = tensors.iter().position(|(n,_)| n == &slot.parameter).ok_or("missing reference tensor")?;
        let mut got = tensors[id].1.clone(); got.load_device_bytes(bytes);
        let expected = &reference.tensors[id];
        let mut worst = 0.0_f64;
        for i in 0..got.shape().iter().product::<usize>() {
            let (a,b) = (got.get(i), expected.get(i));
            if !a.is_finite() || !b.is_finite() || (a-b).abs() > 0.002 + 0.005*b.abs() {
                return Err(format!("native mismatch {}[{i}]: expected {b}, got {a}",slot.parameter));
            }
            worst = worst.max((a-b).abs());
        }
        eprintln!("{}: max abs err {worst:.3e}", slot.parameter);
    }
    Ok(())
}
fn run_cuda(o: &crate::Options, program: &seismic_lang::program::Program, name: &str,
    lowered: &seismic_lang::lower::Lowered, tensors: &[(String, TensorData)], reference: &Interpreter<'_>) -> Result<(), String> {
    let threads = o.threads_per_block.ok_or("CUDA scalar baseline requires an explicit --threads-per-block candidate; automatic selection is not implemented yet")?;
    let device = seismic_cuda::Device::open(0)?;
    eprintln!("CUDA device: {:?}", device.info);
    let start = Instant::now();
    let mut kernel = device.compile_candidate(lowered, seismic_realization::ScalarOptions {dispatch:seismic_realization::Dispatch::ParallelRoot,loads:o.loads}, threads)?;
    eprintln!("CUDA compile: {:.3} s; scratch {} bytes; native {:?}; explicit scalar/SIMT candidate", start.elapsed().as_secs_f64(), kernel.scratch_bytes(), kernel.native);
    let specs = kernel.buffers().to_vec();
    let mut buffers = scalar_buffers(&specs, tensors)?;
    let scalars = kernel.scalars().iter().map(|parameter| o.scalars.get(&parameter.name).copied().ok_or_else(||format!("unbound scalar `{}`",parameter.name))).collect::<Result<Vec<_>,_>>()?;
    kernel.run(&mut buffers.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(), &scalars)?;
    if o.check { check_scalar_buffers(&specs, &buffers, tensors, reference)?; }
    let mut times = Vec::with_capacity(o.iters);
    let mut device_times = Vec::with_capacity(o.iters);
    for _ in 0..o.iters {
        let start = Instant::now();
        let mut device_seconds=0.0;
        for _ in 0..o.repeat { device_seconds+=kernel.launch_timed()?; }
        device_times.push(device_seconds / o.repeat as f64);
        times.push(start.elapsed().as_secs_f64() / o.repeat as f64);
    }
    times.sort_by(f64::total_cmp);
    device_times.sort_by(f64::total_cmp);
    crate::account::print_work(program, name, &o.shapes, o.analysis_steps)?;
    println!("{}: {name} on CUDA {}: median {:.3} ms including launch, synchronization and status download; scratch {} bytes", if o.check {"ok"} else {"unchecked"}, device.info.name, times[times.len()/2]*1000.0, kernel.scratch_bytes());
    println!("CUDA event interval: median {:.3} ms/kernel; input reset/transfer and host validation excluded",device_times[device_times.len()/2]*1000.0);
    Ok(())
}

pub fn plan(args: &[String]) -> Result<(), String> {
    let o = options(args)?;
    let program = load_program(&o)?;
    let name = o.function.as_ref().ok_or("--fn is required")?;
    let plan = seismic_lang::plan::plan(&program, name, &o.shapes)?;
    let mut distinct = std::collections::BTreeMap::new();
    for (i, step) in plan.steps.iter().enumerate() {
        let mut shapes: Vec<String> = step.shapes.iter().map(|(k, v)| format!("{k}={v}")).collect();
        shapes.sort();
        *distinct.entry(format!("{}[{}]", step.kernel, shapes.join(","))).or_insert(0usize) += 1;
        let tensors: Vec<String> = step.tensors.iter().map(|t| if t.elem_offset == 0 { format!("{}={}", t.param, t.root) } else { format!("{}={}+{}", t.param, t.root, t.elem_offset) }).collect();
        let scalars: Vec<String> = step.scalars.iter().map(|(n, s)| match s {
            seismic_lang::plan::ScalarSource::Literal(x) => format!("{n}={x}"),
            seismic_lang::plan::ScalarSource::Param(p) => format!("{n}=<{p}>"),
        }).collect();
        println!("{i:4}: {}({}{}{})", step.kernel, tensors.join(", "), if scalars.is_empty() { "" } else { ", " }, scalars.join(", "));
    }
    println!("{} step(s), {} distinct kernel instantiation(s):", plan.steps.len(), distinct.len());
    for (k, n) in distinct {
        println!("  {n:4} x {k}");
    }
    Ok(())
}

/// Rust bindings for a composition function: a shapes struct, an arguments struct holding
/// buffers and scalars, byte sizes per tensor part, and the `Bindings` implementation the
/// Metal plan executor consumes.
pub fn bindings(args: &[String]) -> Result<(), String> {
    let o = options(args)?;
    let program = load_program(&o)?;
    let name = o.function.as_ref().ok_or("--fn is required")?;
    let f = program.functions.iter().find(|f| &f.name == name).ok_or_else(|| format!("no function `{name}`"))?;
    let camel: String = name.split('_').map(|w| { let mut c = w.chars(); match c.next() { Some(h) => h.to_uppercase().collect::<String>() + c.as_str(), None => String::new() } }).collect();
    let rustify = |s: &seismic_lang::sym::Sym| -> String {
        let text = format!("{s}");
        let mut out = String::new();
        let mut ident = String::new();
        for ch in text.chars().chain(std::iter::once(' ')) {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
            } else {
                if !ident.is_empty() {
                    if ident.chars().next().unwrap().is_ascii_digit() { out.push_str(&ident); } else { out.push_str("s."); out.push_str(&ident); }
                    ident.clear();
                }
                out.push(ch);
            }
        }
        format!("({})", out.trim_end())
    };
    let mut out = String::new();
    out.push_str(&format!("// Generated by `seismic bindings` for `{name}`. Do not edit.\n"));
    out.push_str("#![allow(non_snake_case, dead_code)]\nuse seismic_metal::plan_exec::Bindings;\nuse seismic_metal::runtime::Buffer;\nuse std::collections::HashMap;\n\n");
    out.push_str("/// A packed tensor's parts on the device.\npub struct Packed {\n    pub words: Buffer,\n    pub scale: Buffer,\n    pub bias: Buffer,\n}\n\n");
    out.push_str(&format!("#[derive(Clone, Copy, Debug)]\npub struct {camel}Shapes {{\n"));
    for p in &f.shape_params {
        out.push_str(&format!("    pub {p}: i64,\n"));
    }
    out.push_str("}\n\n");
    out.push_str(&format!("impl {camel}Shapes {{\n    pub fn map(&self) -> HashMap<String, i64> {{\n        let mut m = HashMap::new();\n"));
    for p in &f.shape_params {
        out.push_str(&format!("        m.insert(\"{p}\".to_string(), self.{p});\n"));
    }
    out.push_str("        m\n    }\n}\n\n");
    out.push_str(&format!("pub struct {camel}<'a> {{\n"));
    let mut sizes = Vec::new();
    let mut buffer_arms = Vec::new();
    let mut scalar_arms = Vec::new();
    for (pname, ty) in &f.params {
        match ty {
            Ty::Tensor(s) => {
                let rows: seismic_lang::sym::Sym = s.shape[..s.shape.len() - 1].iter().fold(seismic_lang::sym::Sym::constant(1), |a, d| a.mul(d));
                let last = s.shape.last().unwrap().clone();
                match &s.elem {
                    Elem::Dtype(d) => {
                        out.push_str(&format!("    pub {pname}: &'a Buffer,\n"));
                        sizes.push(format!("        (\"{pname}\", \"\", ({} * {} * {}) as usize),", rustify(&rows), rustify(&last), d.bytes()));
                        buffer_arms.push(format!("            (\"{pname}\", \"\") => Some(self.{pname}),"));
                    }
                    Elem::Repr(r) => {
                        let rep = repr::lookup(r).unwrap();
                        out.push_str(&format!("    pub {pname}: &'a Packed,\n"));
                        sizes.push(format!("        (\"{pname}\", \"words\", ({} * {} * 4) as usize),", rustify(&rows), rustify(&rep.words_extent(&last))));
                        sizes.push(format!("        (\"{pname}\", \"scale\", ({} * {} * {}) as usize),", rustify(&rows), rustify(&rep.groups_extent(&last)), rep.coefficient.bytes()));
                        if rep.has_bias {
                            sizes.push(format!("        (\"{pname}\", \"bias\", ({} * {} * {}) as usize),", rustify(&rows), rustify(&rep.groups_extent(&last)), rep.coefficient.bytes()));
                        }
                        buffer_arms.push(format!("            (\"{pname}\", \"words\") => Some(&self.{pname}.words),\n            (\"{pname}\", \"scale\") => Some(&self.{pname}.scale),\n            (\"{pname}\", \"bias\") => Some(&self.{pname}.bias),"));
                    }
                    Elem::Param(p) => return Err(format!("parameter `{pname}` has generic element `{p}`")),
                }
            }
            Ty::Scalar(d) => {
                let rust = if d.is_int() { "i64" } else { "f64" };
                out.push_str(&format!("    pub {pname}: {rust},\n"));
                scalar_arms.push(format!("            \"{pname}\" => Some(self.{pname} as f64),"));
            }
            other => return Err(format!("parameter `{pname}` of type {other} has no binding")),
        }
    }
    out.push_str("}\n\n");
    out.push_str(&format!("impl<'a> {camel}<'a> {{\n    /// Byte size of every tensor part: (parameter, part, bytes).\n    pub fn sizes(s: &{camel}Shapes) -> Vec<(&'static str, &'static str, usize)> {{\n        vec![\n{}\n        ]\n    }}\n}}\n\n", sizes.join("\n")));
    out.push_str(&format!("impl<'a> Bindings for {camel}<'a> {{\n    fn buffer(&self, root: &str, part: &str) -> Option<&Buffer> {{\n        match (root, part) {{\n{}\n            _ => None,\n        }}\n    }}\n\n    fn scalar(&self, name: &str) -> Option<f64> {{\n        match name {{\n{}\n            _ => None,\n        }}\n    }}\n}}\n", buffer_arms.join("\n"), scalar_arms.join("\n")));
    print!("{out}");
    Ok(())
}
