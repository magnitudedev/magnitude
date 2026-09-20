//! Unified logical -> family -> solve -> resolve -> native compilation
//! inspection commands.

use crate::{load_program, options, Options};
use seismic_compiler::{
    pipeline::{self, Backend, Compiled},
    planning::Budget,
};
use seismic_cpu::mapping::Cpu;
use seismic_cuda::mapping::Cuda;
use seismic_lang::sir::Program;
use seismic_metal::mapping::{Limits, Metal};
use seismic_realization::executable::{ExecutableDialect, ResolvedSchedule, ResolvedStep};

const FLAGS: &[&str] = &[
    "--fn",
    "--shape",
    "--element",
    "--extent",
    "--precision",
    "--atol",
    "--rtol",
    "--relative-floor",
    "--ulps",
    "--evidence",
    "--output-tolerance",
    "--input-range",
    "--allow-special-changes",
    "--target",
];

const DEFAULT_MAX_THREADS_PER_THREADGROUP: u64 = 1024;
const DEFAULT_MAX_THREADGROUP_BYTES: u64 = 32 * 1024;

fn metal() -> Result<(Metal, String), String> {
    #[cfg(target_os = "macos")]
    if let Ok(device) = seismic_metal::runtime::Device::open() {
        let info = device.info();
        let origin = format!(
            "device `{}` (max_threads_per_threadgroup={}, max_threadgroup_bytes={})",
            info.name, info.max_threads_per_threadgroup, info.max_threadgroup_bytes
        );
        return Ok((
            Metal::from_device(&info).map_err(|error| error.to_string())?,
            origin,
        ));
    }
    let limits = Limits {
        max_threads_per_threadgroup: DEFAULT_MAX_THREADS_PER_THREADGROUP,
        max_threadgroup_bytes: DEFAULT_MAX_THREADGROUP_BYTES,
        max_private_bytes: seismic_metal::mapping::CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES,
    };
    Ok((
        Metal::new(limits).map_err(|error| error.to_string())?,
        "documented Metal baseline (no live device facts available)".into(),
    ))
}

fn cpu() -> Result<(Cpu, String), String> {
    let workers = std::thread::available_parallelism()
        .map_err(|error| format!("host parallelism: {error}"))?
        .get() as u64;
    let backend = Cpu::host(workers).map_err(|error| error.to_string())?;
    let origin = format!(
        "host ({} workers, {} scratch bytes per worker)",
        backend.limits().workers,
        backend.limits().max_scratch_bytes
    );
    Ok((backend, origin))
}

fn cuda() -> Result<(Cuda, String), String> {
    if let Ok(device) = seismic_cuda::Device::open(0) {
        let origin = format!(
            "device `{}` (max_threads_per_block={}, max_grid_x={}, warp_size={})",
            device.info.name,
            device.info.max_threads_per_block,
            device.info.max_grid_x,
            device.info.warp_size
        );
        return Ok((
            Cuda::from_device(&device.info).map_err(|error| error.to_string())?,
            origin,
        ));
    }
    Ok((
        Cuda::new(
            seismic_cuda::mapping::Limits::gb10(),
            seismic_cuda::mapping::EstimateModel::default(),
        )
        .map_err(|error| error.to_string())?,
        "documented GB10 baseline (no live device facts available)".into(),
    ))
}

fn compile<B: Backend>(
    options: &Options,
    program: &Program,
    backend: &B,
) -> Result<Compiled<B::Dialect, B::NativeArtifact>, String> {
    pipeline::compile(
        program,
        options.entry()?,
        &options.workload,
        backend,
        &[],
        Budget::default(),
    )
    .map_err(|error| error.to_string())
}

/// Launches of the retained structured schedule, in retained order.
fn launches<D: ExecutableDialect>(schedule: &ResolvedSchedule<D>) -> Vec<&ResolvedStep<D>> {
    let mut out = Vec::new();
    fn walk<'a, D: ExecutableDialect>(
        schedule: &'a ResolvedSchedule<D>,
        out: &mut Vec<&'a ResolvedStep<D>>,
    ) {
        for step in schedule.steps.iter() {
            match step {
                ResolvedStep::Launch(_) => out.push(step),
                ResolvedStep::Call(call) => walk(&call.body.schedule, out),
                ResolvedStep::If(if_step) => {
                    walk(&if_step.then_schedule, out);
                    walk(&if_step.else_schedule, out);
                }
                ResolvedStep::Repeat(repeat) => walk(&repeat.body, out),
            }
        }
    }
    walk(schedule, &mut out);
    out
}

fn report<D: ExecutableDialect, A>(
    compiled: &Compiled<D, A>,
    target: &str,
    capacities: &str,
) -> String {
    let physical = &compiled.physical;
    let mut text = format!(
        "entry: {}\ntarget: {target}\ntarget facts: {capacities}\ncapability fingerprint: {}\nestimated cost: {}\noptimal: {}\nnumerical assessment: {:?}\n",
        compiled.logical.entry,
        compiled.logical.target.capability_fingerprint,
        physical.estimated_cost,
        physical.optimal,
        physical.numerical,
    );
    text.push_str("assignment:\n");
    for (choice, selected) in &physical.identity.selections {
        text.push_str(&format!(
            "  choice#{} = logical#{} / physical#{}\n",
            choice.0, selected.0, selected.1
        ));
    }
    text.push_str("resources:\n");
    let device_bytes = physical.internal_arena.bytes;
    for step in launches(&physical.entry.schedule) {
        if let ResolvedStep::Launch(launch) = step {
            text.push_str(&format!(
                "  launch#{}: device={device_bytes} B, workgroup={} B, private/participant={} B, bindings={}\n",
                launch.id.0,
                launch.kernel.resources.workgroup_bytes,
                launch.kernel.resources.private_bytes_per_participant,
                launch.bindings.len()
            ));
        }
    }
    let nested = {
        fn calls<D: ExecutableDialect>(schedule: &ResolvedSchedule<D>) -> usize {
            schedule
                .steps
                .iter()
                .map(|step| match step {
                    ResolvedStep::Call(call) => 1 + calls(&call.body.schedule),
                    ResolvedStep::If(if_step) => {
                        calls(&if_step.then_schedule) + calls(&if_step.else_schedule)
                    }
                    ResolvedStep::Repeat(repeat) => calls(&repeat.body),
                    ResolvedStep::Launch(_) => 0,
                })
                .sum()
        }
        calls(&physical.entry.schedule)
    };
    text.push_str(&format!(
        "executable plan: {} launches, {} nested calls\n",
        launches(&physical.entry.schedule).len(),
        nested
    ));
    text
}

pub fn select(args: &[String]) -> Result<(), String> {
    let options = options(args, FLAGS)?;
    let (_, program) = load_program(&options)?;
    let text = match options.target.as_str() {
        "cpu" => {
            let (backend, facts) = cpu()?;
            report(&compile(&options, &program, &backend)?, "cpu", &facts)
        }
        "cuda" => {
            let (backend, facts) = cuda()?;
            report(&compile(&options, &program, &backend)?, "cuda", &facts)
        }
        "metal" => {
            let (backend, facts) = metal()?;
            report(&compile(&options, &program, &backend)?, "metal", &facts)
        }
        target => return Err(format!("unknown target `{target}`")),
    };
    print!("{text}");
    Ok(())
}

pub fn emit(args: &[String]) -> Result<(), String> {
    let options = options(args, FLAGS)?;
    let (_, program) = load_program(&options)?;
    let text = match options.target.as_str() {
        "cpu" => {
            let (backend, facts) = cpu()?;
            let compiled = compile(&options, &program, &backend)?;
            let mut text = report(&compiled, "cpu", &facts);
            text.push_str(&format!(
                "\nnative CPU artifact: {} launches\n",
                compiled.native.kernel.launch_count()
            ));
            text
        }
        "cuda" => {
            let (backend, _) = cuda()?;
            compile(&options, &program, &backend)?
                .native
                .launches
                .iter()
                .enumerate()
                .map(|(index, launch)| format!("// launch {index}\n{}\n", launch.ptx))
                .collect()
        }
        "metal" => compile(&options, &program, &metal()?.0)?.native.source,
        target => return Err(format!("unknown target `{target}`")),
    };
    print!("{text}");
    Ok(())
}

pub fn analyze_search(args: &[String]) -> Result<(), String> {
    let options = options(args, FLAGS)?;
    let (_, program) = load_program(&options)?;
    let text = match options.target.as_str() {
        "cpu" => {
            let (backend, facts) = cpu()?;
            search_report(&compile(&options, &program, &backend)?, "cpu", &facts)
        }
        "cuda" => {
            let (backend, facts) = cuda()?;
            search_report(&compile(&options, &program, &backend)?, "cuda", &facts)
        }
        "metal" => {
            let (backend, facts) = metal()?;
            search_report(&compile(&options, &program, &backend)?, "metal", &facts)
        }
        target => return Err(format!("unknown target `{target}`")),
    };
    print!("{text}");
    Ok(())
}

fn search_report<D: ExecutableDialect, A>(
    compiled: &Compiled<D, A>,
    target: &str,
    capacities: &str,
) -> String {
    let physical = &compiled.physical;
    format!(
        "executable search of `{}` on {target}\n  target facts       {capacities}\n  selected choices   {}\n  selected cost      {}\n  optimum proven     {}\n",
        compiled.logical.entry,
        physical.identity.selections.len(),
        physical.estimated_cost,
        physical.optimal,
    )
}
