//! Actual authored Qwen kernel family construction, not a synthetic cost model.
//! One case per process lets the runner bound source construction as well as search.
use magnitude_solver::model::{FactorKind, ModelBuilder};
use seismic_compiler::tuner::{source::Binding, Input};
use seismic_lang::{
    program::{self, SourceFile},
    types::{DType, Elem},
    Scope,
};
use serde_json::json;
use std::{collections::HashMap, time::Instant};

fn shapes(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}
fn elements(pairs: &[(&str, Elem)]) -> HashMap<String, Elem> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}
fn main() {
    if let Err(error) = run() {
        println!("{}", json!({"stage":"error", "error":error}));
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("dense");
    let size: i64 = args
        .get(2)
        .map(|v| v.parse().map_err(|_| "invalid size"))
        .transpose()?
        .unwrap_or(1);
    let step: usize = args
        .get(3)
        .map(|v| v.parse().map_err(|_| "invalid step"))
        .transpose()?
        .unwrap_or(0);
    if size <= 0 {
        return Err("size must be positive".into());
    }
    // These geometry values are from the pinned 4B artifact, recorded in the report.
    let bf = Elem::Dtype(DType::BF16);
    let weight = if args.get(4).is_some_and(|s| s == "bf16") {
        bf.clone()
    } else {
        Elem::Repr("q4g64".into())
    };
    println!(
        "{}",
        json!({"stage":"ready", "mode":mode,"size":size,"step":step,"weight":weight.to_string()})
    );
    let start = Instant::now();
    let mut sources = seismic_std::sources();
    for (path, text) in [
        (
            "dense_suffix",
            include_str!("../../../../engine/lib/dense_suffix.seismic.portable"),
        ),
        (
            "attention_step",
            include_str!("../../../../engine/lib/attention_step.seismic.portable"),
        ),
        (
            "recurrent_step",
            include_str!("../../../../engine/lib/recurrent_step.seismic.portable"),
        ),
    ] {
        sources.push(SourceFile {
            path: format!("qwen35/{path}.seismic.portable").into(),
            text: text.into(),
            scope: Scope::Portable,
        });
    }
    let program = program::compile(&sources, &["cpu".into(), "cuda".into(), "metal".into()])
        .map_err(|e| e.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n"))?;
    println!(
        "{}",
        json!({"stage":"program","milliseconds":start.elapsed().as_secs_f64()*1000.0})
    );
    let (entry, dims, types) = match mode.strip_suffix("-entry").unwrap_or(mode) {
        "dense" => (
            "qwen_dense_suffix",
            shapes(&[("M", size), ("H", 2560), ("F", 9216)]),
            elements(&[
                ("A", bf.clone()),
                ("NW", bf.clone()),
                ("GW", weight.clone()),
                ("UW", weight.clone()),
                ("DW", weight.clone()),
            ]),
        ),
        "decode" => (
            "qwen_attention_step",
            shapes(&[
                ("D", 2560),
                ("T", size),
                ("H", 16),
                ("KV", 4),
                ("P", 32),
                ("S", 192),
                ("SH", 11),
                ("SW", 10),
            ]),
            elements(&[
                ("A", bf.clone()),
                ("NW", bf.clone()),
                ("QW", weight.clone()),
                ("KW", weight.clone()),
                ("VW", weight.clone()),
                ("OW", weight.clone()),
            ]),
        ),
        "recurrent" => (
            "qwen_recurrent_step",
            shapes(&[("H", 2560), ("NK", 16), ("GV", 2), ("W", 128), ("C", 4)]),
            elements(&[
                ("A", bf.clone()),
                ("NW", bf.clone()),
                ("RN", bf.clone()),
                ("QW", weight.clone()),
                ("GW", weight.clone()),
                ("AW", weight.clone()),
                ("BW", weight.clone()),
                ("OW", weight.clone()),
            ]),
        ),
        "prefill" => (
            "attention",
            shapes(&[("Q", size), ("T", size), ("H", 16), ("KV", 4), ("W", 256)]),
            elements(&[("A", bf)]),
        ),
        _ => return Err("expected dense, decode, recurrent or prefill".into()),
    };
    let mut options = seismic_lang::lower::Options::default();
    let plan_start = Instant::now();
    let (kernel, dims, types) = if mode == "prefill" {
        (entry.to_string(), dims, types)
    } else {
        let plan = seismic_lang::plan::plan_specialized(&program, entry, &dims, &types)?;
        println!(
            "{}",
            json!({"stage":"plan","milliseconds":plan_start.elapsed().as_secs_f64()*1000.0,
            "kernels":plan.steps.iter().map(|s|s.kernel.as_str()).collect::<Vec<_>>()})
        );
        if mode.ends_with("-entry") {
            options.ownership = plan.ownership;
            (entry.to_string(), dims, types)
        } else {
            let selected = plan.steps.get(step).ok_or("step outside plan")?;
            (
                selected.kernel.clone(),
                selected.shapes.clone(),
                selected.elements.clone(),
            )
        }
    };
    println!(
        "{}",
        json!({"stage":"specialization","kernel":kernel,"shapes":dims,
        "elements":types.iter().map(|(k,v)|(k.clone(),v.to_string())).collect::<HashMap<_,_>>()})
    );
    let begin = Instant::now();
    let family = Binding::construct(
        Input::Portable {
            program: &program,
            entry: &kernel,
            shapes: &dims,
            elements: &types,
            options: &options,
        },
        "metal",
    )?;
    let construct_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let mut classes = std::collections::BTreeMap::<String, usize>::new();
    let mut log10_product = 0.0;
    let mut maximum_domain = 0;
    for d in family.decisions() {
        *classes.entry(format!("{:?}", d.id.class)).or_default() += 1;
        let n = d.domain.alternatives.len();
        maximum_domain = maximum_domain.max(n);
        log10_product += (n.max(1) as f64).log10();
    }
    println!(
        "{}",
        json!({"stage":"family","milliseconds":construct_ms,"regions":family.regions().len(),
        "decisions":family.decisions().len(),"classes":classes,"largest_domain":maximum_domain,
        "log10_cartesian_upper_bound":log10_product,"requirements":family.requirements().len(),
        "dependencies":family.dependencies().len(),
        "obligations":family.obligations().iter().map(|o|json!({"class":format!("{:?}",o.class),"reason":o.reason})).collect::<Vec<_>>()})
    );
    let begin = Instant::now();
    let mut builder = ModelBuilder::new();
    let _binding = Binding::append(&mut builder, family)?;
    let model = builder.build().map_err(|e| e.to_string())?;
    let obligations: Vec<_> = model
        .factors()
        .iter()
        .filter_map(|f| match &f.kind {
            FactorKind::Unresolved { reason, .. } => Some(reason.clone()),
            _ => None,
        })
        .collect();
    println!(
        "{}",
        json!({"stage":"source_export","milliseconds":begin.elapsed().as_secs_f64()*1000.0,
        "variables":model.variables().len(),"factors":model.factors().len(),"retained_bytes":model.retained_bytes(),
        "obligations":obligations,"end_to_end_ms":start.elapsed().as_secs_f64()*1000.0,
        "quality_eligible":false,"reason":"Source-only model: target conditional schedules and costs are not exported; no optimization quality claim."})
    );
    Ok(())
}
