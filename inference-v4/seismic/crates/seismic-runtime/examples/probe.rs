//! Timing probe for one linked entry on Metal: compiles one specialization
//! domain, binds random tensors, and reports the resolved physical
//! assignment, its estimate, and measured time per invocation (best of
//! several observed runs) and per invocation inside one sequential batch.
//!
//! usage: probe <source dir>... -- <entry> <K=V,...> <NAME=elem,...|-> [scalar=v,...|-] [name=v:v:...;...|-] [exact]
use seismic_lang::interp::{Rng, TensorData};
use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
use seismic_lang::program::{collect_files, compile};
use seismic_lang::repr;
use seismic_lang::sir::Mode;
use seismic_lang::types::{DType, Elem};
use seismic_lang::types::{ExtentExpr, ValueType};
use seismic_runtime::invocation::Bindings;
use seismic_runtime::plan::{CompiledArtifact, PlanCompiler, Settings};
use seismic_runtime::submission::Submission;
use seismic_runtime::{Buffer, Device};
use seismic_realization::kernel::ExecutableDialect;
use seismic_realization::physical::PhysicalPlan;
use std::collections::{BTreeMap, HashMap};

fn report<D: ExecutableDialect>(physical: &PhysicalPlan<D>) {
    println!(
        "physical estimate {}, optimal={}, launches={}",
        physical.estimated_cost(),
        physical.optimal(),
        physical.launches().len(),
    );
    println!("assignment {:?}", physical.identity().selections);
    println!("resources {:?}", physical.resources());
    println!("numerics {:?}", physical.numerical());
}

struct Bound {
    buffers: HashMap<(String, String), Buffer>,
    scalars: HashMap<String, f64>,
    shapes: HashMap<String, i64>,
}
impl Bindings for Bound {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        self.buffers.get(&(root.to_string(), plane.to_string()))
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        self.scalars.get(name).copied()
    }
    fn shape(&self, name: &str) -> Option<u64> {
        self.shapes.get(name).and_then(|v| u64::try_from(*v).ok())
    }
}

fn pairs(text: &str) -> Vec<(String, String)> {
    text.split(',')
        .filter(|p| !p.is_empty() && *p != "-")
        .filter_map(|p| p.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args
        .iter()
        .position(|a| a == "--")
        .ok_or("usage: probe <dir>... -- <entry> <shapes> <elems> [scalars] [contents]")?;
    let (dirs, rest) = (&args[..split], &args[split + 1..]);
    let entry = rest.first().ok_or("missing entry")?.clone();
    let shapes: HashMap<String, i64> = pairs(rest.get(1).ok_or("missing shapes")?)
        .into_iter()
        .map(|(k, v)| Ok((k, v.parse::<i64>().map_err(|e| e.to_string())?)))
        .collect::<Result<_, String>>()?;
    let elems: BTreeMap<String, Elem> = pairs(rest.get(2).map_or("-", String::as_str))
        .into_iter()
        .map(|(k, v)| {
            let elem = match DType::from_name(&v) {
                Some(d) => Elem::Dtype(d),
                None if repr::lookup(&v).is_some() => Elem::Repr(v.clone()),
                None => return Err(format!("unknown element {v}")),
            };
            Ok((k, elem))
        })
        .collect::<Result<_, String>>()?;
    let scalars: HashMap<String, f64> = pairs(rest.get(3).map_or("-", String::as_str))
        .into_iter()
        .map(|(k, v)| Ok((k, v.parse::<f64>().map_err(|e| e.to_string())?)))
        .collect::<Result<_, String>>()?;
    let contents: HashMap<String, Vec<f64>> = rest
        .get(4)
        .map_or("", String::as_str)
        .split(';')
        .filter_map(|p| p.split_once('='))
        .map(|(k, v)| {
            (
                k.to_string(),
                v.split(':').filter_map(|x| x.parse().ok()).collect(),
            )
        })
        .collect();

    let paths: Vec<std::path::PathBuf> = dirs.iter().map(std::path::PathBuf::from).collect();
    let files = collect_files(&paths).map_err(|e| format!("{e:?}"))?;
    let program = compile(&files).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.render())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let family = program.resolve_family(&entry)?;
    let definition = family
        .bodies
        .iter()
        .chain(&family.lowerings)
        .map(|id| program.definition(*id))
        .next()
        .ok_or("entry has no implementation")?;
    let device = Device::metal().map_err(|e| e.to_string())?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buffers = HashMap::new();
    for param in &definition.params {
        let ValueType::Tensor(shaped) = &param.ty else {
            continue;
        };
        let shape = shaped
            .axes
            .iter()
            .map(|axis| match axis {
                ExtentExpr::Sym(sym) => sym
                    .eval(&|n| shapes.get(n).copied())
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
            Elem::Param(p) => elems
                .get(p)
                .ok_or_else(|| format!("unbound element {p}"))?,
            concrete => concrete,
        };
        let mut tensor = match elem {
            Elem::Dtype(dtype) => TensorData::random_dense(&mut rng, *dtype, shape),
            Elem::Repr(name) => TensorData::random_packed(
                &mut rng,
                repr::lookup(name.as_str()).ok_or("unknown representation")?,
                shape,
            ),
            Elem::Param(p) => return Err(format!("element {p} is not concrete")),
        };
        if let Some(values) = contents.get(&param.name) {
            let count = match &tensor {
                TensorData::Dense { data, .. } => data.len(),
                TensorData::Packed { .. } => 0,
            };
            for flat in 0..count {
                tensor
                    .set(flat, values[flat % values.len()])
                    .map_err(|e| e.to_string())?;
            }
        }
        let _ = param.mode != Mode::In;
        let planes: Vec<String> = match &tensor {
            TensorData::Dense { .. } => vec![String::new()],
            TensorData::Packed { repr, .. } => {
                repr.planes().iter().map(|p| p.name.to_string()).collect()
            }
        };
        for (plane, bytes) in planes.into_iter().zip(tensor.device_bytes()) {
            buffers.insert(
                (param.name.clone(), plane),
                device
                    .buffer_from(&bytes)
                    .map_err(|e| format!("upload {}: {e}", param.name))?,
            );
        }
    }
    let precision = if rest.get(5).is_some_and(|mode| mode == "unconstrained") {
        seismic_lang::precision::PrecisionPolicy::Unconstrained
    } else {
        seismic_lang::precision::PrecisionPolicy::Exact
    };
    let domain = SpecializationDomain::new(
        &program,
        &entry,
        shapes
            .iter()
            .map(|(n, v)| {
                let value = u64::try_from(*v)
                    .map_err(|_| format!("shape {n}={v} is not a valid extent"))?;
                Ok((n.clone(), ShapeBinding::Exact(value)))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?,
        elems,
    )
    .map_err(|e| e.to_string())?;
    let bound = Bound {
        buffers,
        scalars,
        shapes,
    };
    let plan = PlanCompiler::new(
        &device,
        &program,
        Settings {
            precision,
            ..Settings::default()
        },
    )
    .compile_entry(&domain)
    .map_err(|e| e.to_string())?;
    match plan.artifact() {
        artifact => {
            if let Some(physical) = artifact.physical_cpu() {
                report(physical);
            }
            #[cfg(target_os = "macos")]
            if let Some(physical) = artifact.physical_metal() {
                report(physical);
            }
            if let Some(physical) = artifact.physical_cuda() {
                report(physical);
            }
        }
    }
    let mut best_host = f64::INFINITY;
    for _ in 0..12 {
        let observed = Submission::single(
            plan.prepare(&bound).map_err(|e| e.to_string())?,
        )
        .execute_observed()
        .map_err(|e| e.to_string())?;
        for (_, observation) in observed {
            best_host = best_host.min(observation.host_seconds);
        }
    }
    println!("best host {:.1} us", best_host * 1e6);
    let mut batched = f64::INFINITY;
    for _ in 0..8 {
        let mut submission = Submission::single(
            plan.prepare(&bound).map_err(|e| e.to_string())?,
        );
        for _ in 1..200 {
            submission.append(Submission::single(
                plan.prepare(&bound).map_err(|e| e.to_string())?,
            ));
        }
        let observed = submission
            .execute_observed()
            .map_err(|e| e.to_string())?;
        let total: f64 = observed
            .into_iter()
            .map(|(_, observation)| observation.host_seconds)
            .sum();
        batched = batched.min(total / 200.0);
    }
    println!("batched x200: {:.1} us per invocation", batched * 1e6);
    Ok(())
}
