//! Device-free check: every std kernel case selects on the CUDA backend under the
//! documented GB10 default limits and realizes to PTX text with launch geometry.
//! Prints one row per kernel and exits nonzero if any kernel fails.
#[path = "support/cases.rs"]
mod cases;

use seismic_compiler::selection::{select, Budget};
use seismic_cuda::mapping::{Cuda, EstimateModel, Limits};

fn main() -> Result<(), String> {
    let program = cases::program()?;
    let numerics = cases::numerics()?;
    let backend = Cuda::new(Limits::gb10(), EstimateModel::default()).map_err(|e| e.to_string())?;
    let mut failed = 0usize;
    let all = cases::selected();
    println!("{:<30} outcome ({numerics:?} numerics, limits {:?})", "kernel", backend.limits());
    for case in &all {
        let outcome = cases::guarded(|| {
            let workload = cases::workload(case, numerics)?;
            let selected = select(&program, case.entry, &workload, &backend, Budget::default()).map_err(|e| e.to_string())?;
            let texts = selected.execution.ptx();
            // `PTX_DIR=<dir>` keeps every launch's text for inspection.
            if let Ok(directory) = std::env::var("PTX_DIR") {
                for (ordinal, text) in texts.iter().enumerate() {
                    let name = case.label.replace([' ', '='], "_");
                    std::fs::write(format!("{directory}/{name}.{ordinal}.ptx"), text).map_err(|e| format!("{directory}: {e}"))?;
                }
            }
            if texts.is_empty() || texts.iter().any(|text| !text.contains("seismic_kernel")) {
                return Err("realization printed no PTX kernel".into());
            }
            let geometry: Vec<String> = selected.execution.phases.iter().map(|p| format!("{}x{}", p.dispatch().groups, p.dispatch().threads_per_group)).collect();
            Ok(format!("{} launch(es) [blocks x threads: {}], {} PTX bytes, {:?}, estimate {} ns", texts.len(), geometry.join(" "), texts.iter().map(String::len).sum::<usize>(), selected.status, selected.estimate))
        });
        match outcome {
            Ok(summary) => println!("{:<30} ok    {summary}", case.label),
            Err(error) => {
                failed += 1;
                println!("{:<30} ERROR {}", case.label, error.lines().next().unwrap_or(""));
            }
        }
    }
    println!("{} of {} kernels emitted PTX", all.len() - failed, all.len());
    if failed > 0 { Err(format!("{failed} kernels failed")) } else { Ok(()) }
}
