//! `seismic lower` and `seismic run`.

use crate::{load_program, options};
use seismic_lang::interp::{Arg, Interpreter, Rng, TensorData};
use seismic_lang::lower::Choice;
use seismic_lang::repr;
use seismic_lang::types::{Elem, Ty};
use seismic_metal::msl;
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
    if o.target != "metal" {
        return Err(format!("target `{}` has no printer yet", o.target));
    }
    // `lower` prints without a device, so the architectural maximum stands in for a query.
    let emitted = msl::emit_with(&lowered, msl::Config { sg_per_tg: o.sg_per_tg, piece: o.piece, per_item: o.per_item, split: o.split, cores: 40, max_threads_per_threadgroup: 1024, max_threadgroup_bytes: 32768 })?;
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
    let mut scalar_bytes = Vec::new();
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
                match d {
                    seismic_lang::types::DType::I32 => scalar_bytes.extend_from_slice(&(v as i32).to_le_bytes()),
                    seismic_lang::types::DType::U32 => scalar_bytes.extend_from_slice(&(v as u32).to_le_bytes()),
                    _ => scalar_bytes.extend_from_slice(&(v as f32).to_le_bytes()),
                }
            }
            other => return Err(format!("parameter `{pname}` of type {other} cannot be run")),
        }
    }

    // Reference.
    if o.check {
        let t0 = Instant::now();
        interp.run(name, &interp_args, &o.shapes)?;
        eprintln!("interpreter: {:.3} s", t0.elapsed().as_secs_f64());
    }

    // Lower, emit, compile.
    let lowered = seismic_lang::lower::lower_with(&program, name, &o.target, &o.shapes, &seismic_lang::lower::Options { piece: o.piece })?;
    for s in &lowered.selections {
        let choice = match s.choice {
            Choice::Block(i) => format!("block {i}"),
            Choice::Portable => "portable body".to_string(),
        };
        eprintln!("{}[{}]: {choice}", s.construct, s.shape_args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "));
    }
    if o.target != "metal" {
        return Err(format!("target `{}` has no runtime yet", o.target));
    }
    let device = Device::open()?;
    let info = device.info();
    eprintln!("device: {} (unified memory: {})", info.name, info.unified_memory);
    let emitted = msl::emit_with(&lowered, msl::Config { sg_per_tg: o.sg_per_tg, piece: o.piece, per_item: o.per_item, split: o.split, cores: info.cores as i64, max_threads_per_threadgroup: info.max_threads_per_threadgroup as i64, max_threadgroup_bytes: info.max_threadgroup_bytes as i64 })?;
    let t1 = Instant::now();
    let pipeline = device.compile(emitted)?;
    eprintln!("metal compile: {:.3} s", t1.elapsed().as_secs_f64());

    // Buffers in ABI order.
    let mut buffers: Vec<Buffer> = Vec::new();
    let mut total_bytes = 0usize;
    for slot in &pipeline.emitted.buffers {
        let (_, data) = tensors.iter().find(|(n, _)| *n == slot.param).unwrap();
        let parts = data.device_bytes();
        let part = match slot.part {
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
        let slot = pipeline.emitted.buffers.iter().find(|s| s.param == *pname && s.part == "").unwrap();
        let bytes = buffers[slot.index].read(data.bytes());
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
    let gbps = total_bytes as f64 / median / 1e9;
    eprintln!("first run: {:.3} ms; median of {}: {:.3} ms; min {:.3} ms", gpu * 1e3, o.iters, median * 1e3, times[0] * 1e3);
    eprintln!("buffer bytes: {:.1} MB; achieved {gbps:.1} GB/s over buffers", total_bytes as f64 / 1e6);
    // Demand and bound.
    let d = seismic_model::demand::demand(&program, name, &o.shapes)?;
    let supply = match seismic_model::supply::Supply::load(&info.name) {
        Some(s) => s,
        None => {
            let bw = seismic_metal::runtime::calibrate_bandwidth(&device, 256 << 20)? * 1e9;
            let s = seismic_model::supply::Supply { device: info.name.clone(), bandwidth: bw };
            s.store()?;
            s
        }
    };
    let bound = d.bytes() / supply.bandwidth;
    eprintln!(
        "demand: {:.2} MB read, {:.2} MB written, {:.1} MFLOP; bound {:.3} ms at {:.0} GB/s; achieved {:.0}% of bound",
        d.bytes_read / 1e6,
        d.bytes_written / 1e6,
        d.flops / 1e6,
        bound * 1e3,
        supply.bandwidth / 1e9,
        100.0 * bound / median
    );
    for (t, r, w) in &d.per_tensor {
        eprintln!("  {t}: read {:.2} MB, write {:.2} MB", r / 1e6, w / 1e6);
    }
    let ok = worst_rel <= 2e-2;
    println!("{}: {name} on {}: max rel err {worst_rel:.2e}, {:.3} ms, {:.0}% of bound", if ok { "ok" } else { "MISMATCH" }, o.target, median * 1e3, 100.0 * bound / median);
    if ok { Ok(()) } else { Err("device result differs from the reference".into()) }
}

pub fn calibrate() -> Result<(), String> {
    let device = Device::open()?;
    let info = device.info();
    for mb in [64usize, 256, 1024] {
        let gbps = seismic_metal::runtime::calibrate_bandwidth(&device, mb << 20)?;
        println!("{}: streaming read of {mb} MB: {gbps:.1} GB/s", info.name);
    }
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
