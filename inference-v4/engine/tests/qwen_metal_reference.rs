//! Localizes interpreter/Metal divergence at real Qwen3.5-4B dimensions on seeded
//! random data. Every written tensor of every entry is compared; nothing stops early.
#[path = "support/reference.rs"]
mod reference;
use reference::{Backend, TensorData, Workload};
use seismic_lang::{
    family,
    interp::Rng,
    sir::Program,
    span::line_col,
    syntax::ast::Mode,
    types::{DType, Elem, Ty},
};
use seismic_runtime::{plan::{PlanCompiler, Settings}, Device, Selection};
use std::collections::HashMap;

struct Row {
    case: String,
    tensor: String,
    dtype: DType,
    max_abs: f64,
    max_rel: f64,
    worst: usize,
    beyond: usize,
    beyond_compact: usize,
    count: usize,
}
fn describe(program: &Program, target: &str, entry: &str, selection: &Selection, workload: &Workload) -> String {
    let mut text = format!(
        "  selection {:?} estimate={} seed={} bound={} model={} unresolved={:?}\n",
        selection.status, selection.estimate, selection.seed_estimate, selection.lower_bound, selection.estimate_model, selection.unresolved
    );
    let family = match family::construct(program, entry, target, workload) {
        Ok(family) => family,
        Err(e) => return format!("{text}  family reconstruction failed: {e}\n  raw witness {:?}\n", selection.witness),
    };
    let named = |id| {
        let d = program.definition(id);
        let (path, source) = &program.files[d.file];
        format!("{}@{}:{} {:?}", d.name, path, line_col(source, d.span.start).0, d.kind)
    };
    for (occurrence, choice) in &selection.witness.choices {
        let Some(candidate) = family.occurrences.get(occurrence.0 as usize).and_then(|o| o.candidates.get(*choice as usize)) else {
            text += &format!("  occ {} -> candidate {choice} (not in reconstructed family)\n", occurrence.0);
            continue;
        };
        let template = &family.templates[candidate.template.0 as usize];
        text += &format!(
            "  occ {} -> [{choice}] {} via {} shapes={:?} structural={:?}\n",
            occurrence.0, named(template.definition), named(candidate.via), template.shapes, template.structural
        );
    }
    for (site, value) in &selection.witness.sites {
        match family.sites.get(site.0 as usize) {
            Some(s) => text += &format!("  site {} = {value} (occ {} cand {}, {:?}, extent {})\n", site.0, s.owner.occurrence.0, s.owner.candidate, s.kind, s.extent),
            None => text += &format!("  site {} = {value}\n", site.0),
        }
    }
    text + &format!("  covers {:?}\n", selection.witness.covers)
}
struct Harness<'a> {
    program: &'a Program,
    device: &'a Device,
    rows: Vec<Row>,
}
impl Harness<'_> {
    fn compare(&mut self, case: &str, entry: &str, shapes: &[(&str, i64)], tensors: HashMap<String, TensorData>, scalars: &[(&str, f64)]) {
        let shapes: HashMap<String, i64> = shapes.iter().map(|(n, v)| (n.to_string(), *v)).collect();
        let scalars: HashMap<String, f64> = scalars.iter().map(|(n, v)| (n.to_string(), *v)).collect();
        let started = std::time::Instant::now();
        let mut expected = tensors.clone();
        Backend::Interpreter(self.program, 64).run(entry, &shapes, &mut expected, &scalars);
        let interpreted = started.elapsed();
        let initial = tensors.clone();
        let mut actual = tensors;
        let (selection, workload) = Backend::Metal(PlanCompiler::new(self.device, self.program, Settings { precision: seismic_lang::precision::PrecisionPolicy::Exact, ..Settings::default() }))
            .run(entry, &shapes, &mut actual, &scalars)
            .unwrap();
        println!("== {case}: {entry} {shapes:?} (interpreter {interpreted:.1?})");
        print!("{}", describe(self.program, self.device.backend(), entry, &selection, &workload));
        if let Ok(directory) = std::env::var("QWEN_REFERENCE_MSL") {
            let seismic_runtime::DeviceFacts::Metal(info) = self.device.facts() else { panic!("the Metal reference needs a Metal device") };
            let backend = seismic_metal::mapping::Metal::from_device(&info).unwrap();
            let selected = seismic_compiler::selection::select(self.program, entry, &workload, &backend, Default::default()).unwrap();
            let emitted = seismic_metal::msl::emit_execution(&selected.execution).unwrap();
            std::fs::write(format!("{directory}/{}.metal", case.replace(' ', "_")), &emitted.source).unwrap();
            std::fs::write(format!("{directory}/{}.launches.txt", case.replace(' ', "_")), format!("{:#?}", emitted.launches)).unwrap();
        }
        // A/B the emitted source: `from=>to` replacements separated by `;;`, run through the
        // native Metal runtime directly. Diagnostic only; selection and emission are untouched.
        if let Ok(patch) = std::env::var("QWEN_REFERENCE_PATCH") {
            let seismic_runtime::DeviceFacts::Metal(info) = self.device.facts() else { panic!("the Metal reference needs a Metal device") };
            let backend = seismic_metal::mapping::Metal::from_device(&info).unwrap();
            let selected = seismic_compiler::selection::select(self.program, entry, &workload, &backend, Default::default()).unwrap();
            let mut emitted = seismic_metal::msl::emit_execution(&selected.execution).unwrap();
            for replacement in patch.split(";;") {
                let (from, to) = replacement.split_once("=>").unwrap();
                println!("   patch `{from}` => `{to}`: {} sites", emitted.source.matches(from).count());
                emitted.source = emitted.source.replace(from, to);
            }
            let native = seismic_metal::runtime::Device::open().unwrap();
            let pipeline = native.compile(emitted).unwrap();
            let buffers = pipeline.emitted.buffers.iter().map(|slot| {
                let tensor = &initial[&slot.parameter];
                let plane = match tensor {
                    TensorData::Dense { .. } => 0,
                    TensorData::Packed { repr, .. } => repr.plane_index(&slot.plane).unwrap(),
                };
                native.buffer_from(&tensor.device_bytes()[plane]).unwrap()
            }).collect::<Vec<_>>();
            let values = pipeline.emitted.scalars.iter().map(|p| scalars[&p.name]).collect::<Vec<_>>();
            native.run(&pipeline, &buffers.iter().collect::<Vec<_>>(), &pipeline.emitted.encode_scalars(&values).unwrap(), 1).unwrap();
            for (slot, buffer) in pipeline.emitted.buffers.iter().zip(&buffers) {
                let mut tensor = initial[&slot.parameter].clone();
                if !matches!(tensor, TensorData::Dense { .. }) { continue; }
                tensor.load_device_bytes(&buffer.read(buffer.len()));
                let (e, a) = (reference::values(&expected[&slot.parameter]), reference::values(&tensor));
                let bad = e.iter().zip(&a).filter(|(e, a)| e != a && !(e.is_nan() && a.is_nan())).count();
                let worst = e.iter().zip(&a).map(|(e, a)| (e - a).abs()).fold(0f32, f32::max);
                println!("   patched {}: {bad} mismatches, max_abs {worst:e}", slot.parameter);
            }
        }
        for param in &reference::entry(self.program, entry).params {
            if !matches!(param.ty, Ty::Tensor(_)) || param.mode == Mode::In {
                continue;
            }
            let TensorData::Dense { dtype, .. } = &expected[&param.name] else { continue };
            let (e, a) = (reference::values(&expected[&param.name]), reference::values(&actual[&param.name]));
            let mut row = Row { case: case.into(), tensor: param.name.clone(), dtype: *dtype, max_abs: 0., max_rel: 0., worst: 0, beyond: 0, beyond_compact: 0, count: e.len() };
            for (i, (e, a)) in e.iter().zip(&a).enumerate() {
                let (e, a) = (f64::from(*e), f64::from(*a));
                let abs = if e == a || (e.is_nan() && a.is_nan()) { 0. } else if (e - a).is_nan() { f64::INFINITY } else { (e - a).abs() };
                if abs > row.max_abs {
                    (row.max_abs, row.worst) = (abs, i);
                }
                row.max_rel = row.max_rel.max(if abs == 0. { 0. } else { abs / e.abs().max(1e-30) });
                row.beyond += usize::from(abs > 2e-4 + 2e-3 * e.abs());
                row.beyond_compact += usize::from(abs > 2e-4 + 1.6e-2 * e.abs());
            }
            if std::env::var("QWEN_REFERENCE_DUMP").is_ok() {
                let bad = e.iter().zip(&a).enumerate().filter(|(_, (e, a))| e != a).map(|(i, _)| i).collect::<Vec<_>>();
                let mut runs: Vec<(usize, usize)> = Vec::new();
                for &i in &bad {
                    match runs.last_mut() {
                        Some((_, end)) if *end == i => *end += 1,
                        _ => runs.push((i, i + 1)),
                    }
                }
                println!("   {} mismatches {}: runs {:?}", row.tensor, bad.len(), &runs[..runs.len().min(40)]);
                for &i in bad.iter().take(12) {
                    println!("     [{i}] ref {} metal {}", e[i], a[i]);
                }
            }
            if row.max_abs > 0. {
                println!("   {}[{}]: ref {} metal {}", row.tensor, row.worst, e[row.worst], a[row.worst]);
            }
            self.rows.push(row);
        }
    }
}
fn random(program: &Program, entry: &str, shapes: &[(&str, i64)], seed: u64, element: impl Fn(&str) -> Elem) -> HashMap<String, TensorData> {
    let shapes: HashMap<String, i64> = shapes.iter().map(|(n, v)| (n.to_string(), *v)).collect();
    let mut rng = Rng(seed);
    reference::entry(program, entry)
        .params
        .iter()
        .filter_map(|param| {
            let Ty::Tensor(tensor) = &param.ty else { return None };
            let shape = reference::extents(tensor, &shapes);
            let element = match &tensor.elem {
                Elem::Param(p) => element(p),
                concrete => concrete.clone(),
            };
            let data = match element {
                Elem::Dtype(dtype) if param.mode == Mode::In || param.mode == Mode::Inout => TensorData::random_dense(&mut rng, dtype, shape),
                Elem::Dtype(dtype) => { let n = shape.iter().product(); TensorData::dense(dtype, shape, vec![0.; n]) }
                Elem::Repr(name) => TensorData::random_packed(&mut rng, seismic_lang::repr::lookup(&name).unwrap(), shape),
                Elem::Param(_) => unreachable!(),
            };
            Some((param.name.clone(), data))
        })
        .collect()
}
fn integers(tensors: &mut HashMap<String, TensorData>, name: &str, values: impl IntoIterator<Item = i64>) {
    reference::fill(tensors.get_mut(name).unwrap(), values.into_iter().map(|v| v as f64));
}
fn packed(parameter: &str, packed_parameters: &[&str]) -> Elem {
    if packed_parameters.contains(&parameter) { Elem::Repr("q4g64".into()) } else { Elem::Dtype(DType::BF16) }
}
#[test]
#[ignore = "requires a Metal device"]
fn real_geometry_entries_match_the_interpreter_on_metal() {
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    // `QWEN_REFERENCE_DEVICE=cpu` holds the CPU device to the same table.
    let device = Device::open(&std::env::var("QWEN_REFERENCE_DEVICE").unwrap_or_else(|_| "metal".into())).unwrap();
    let mut h = Harness { program: &program, device: &device, rows: Vec::new() };
    let only = std::env::var("QWEN_REFERENCE_ONLY").ok();
    let wanted = |case: &str| only.as_deref().is_none_or(|o| o.split(',').any(|o| case.contains(o)));
    for m in [1i64, 3] {
        if wanted("embedding") {
            let shapes = [("M", m), ("V", 1024), ("D", std::env::var("QWEN_REFERENCE_D").map_or(2560, |d| d.parse().unwrap()))];
            let mut t = random(&program, "qwen_embedding_rows", &shapes, 11, |p| packed(p, &["EW"]));
            integers(&mut t, "tokens", (0..m).map(|i| std::env::var("QWEN_REFERENCE_TOKEN").map_or([1023, 0, 517][i as usize], |t| t.parse().unwrap())));
            h.compare(&format!("embedding M={m}"), "qwen_embedding_rows", &shapes, t, &[]);
        }
        if wanted("dense") {
            let shapes = [("M", m), ("H", 2560), ("F", 9216)];
            let t = random(&program, "qwen_dense_suffix", &shapes, 12, |p| packed(p, &["GW", "UW", "DW"]));
            h.compare(&format!("dense M={m}"), "qwen_dense_suffix", &shapes, t, &[("eps", 1e-6)]);
        }
        if wanted("recurrent") {
            let shapes = [("M", m), ("H", 2560), ("NK", 16), ("GV", 2), ("W", 128), ("C", 4)];
            let t = random(&program, "qwen_recurrent_sequence", &shapes, 13, |p| packed(p, &["QW", "GW", "AW", "BW", "OW"]));
            h.compare(&format!("recurrent M={m}"), "qwen_recurrent_sequence", &shapes, t, &[("epsilon", 1e-6), ("preparation_epsilon", 1e-6), ("grouped", 1.)]);
        }
        if wanted("attention") {
            let shapes = [("M", m), ("D", 2560), ("T", 16), ("R", 1), ("H", 16), ("KV", 4), ("P", 32), ("S", 192), ("SH", 11), ("SW", 10)];
            let mut t = random(&program, "qwen_attention_sequence", &shapes, 14, |p| packed(p, &["QW", "KW", "VW", "OW"]));
            let position = 5;
            integers(&mut t, "coordinates", (0..m).flat_map(|i| [position + i; 4]));
            integers(&mut t, "visible", (0..m).flat_map(|_| [0, position]));
            integers(&mut t, "destinations", (0..m).map(|i| position + i));
            h.compare(&format!("attention M={m}"), "qwen_attention_sequence", &shapes, t, &[("base", 1e7), ("epsilon", 1e-6), ("scale", 0.0625)]);
        }
        // Opt-in only: at these shapes Metal pipeline creation currently fails with
        // "Compute function exceeds available stack space", which would abort the table.
        if only.as_deref().is_some_and(|o| o.contains("selected")) {
            let shapes = [("M", m), ("V", 1024), ("D", 2560), ("S", 5)];
            let mut t = random(&program, "qwen_readout_selected", &shapes, 16, |p| packed(p, &["OW"]));
            integers(&mut t, "selected", [1023, 0, 517, 0, 64]);
            h.compare(&format!("selected M={m}"), "qwen_readout_selected", &shapes, t, &[("epsilon", 1e-6)]);
        }
        if wanted("readout") {
            let shapes = [("M", m), ("V", 1024), ("D", 2560)];
            let t = random(&program, "qwen_readout_rows", &shapes, 15, |p| packed(p, &["OW"]));
            h.compare(&format!("readout M={m}"), "qwen_readout_rows", &shapes, t, &[("epsilon", 1e-6)]);
        }
    }
    println!("\n{:<18} {:<18} {:<5} {:>9} {:>12} {:>12} {:>10} {:>10}", "case", "tensor", "dtype", "elements", "max_abs", "max_rel", ">tol", ">bf16tol");
    for r in &h.rows {
        println!("{:<18} {:<18} {:<5} {:>9} {:>12.4e} {:>12.4e} {:>10} {:>10}", r.case, r.tensor, r.dtype.name(), r.count, r.max_abs, r.max_rel, r.beyond, r.beyond_compact);
    }
    // Metal selects under exact precision, so every tensor is held to the strict bound.
    let failed = h.rows.iter().filter(|r| r.beyond > 0).map(|r| format!("{} {}", r.case, r.tensor)).collect::<Vec<_>>();
    assert!(failed.is_empty(), "beyond tolerance (2e-4 + 2e-3*|ref|): {failed:?}");
}
