//! Unified logical -> physical -> native compilation inspection commands.

use crate::{load_program, options, Options};
use seismic_compiler::{
    pipeline::{self, Backend, Compiled},
    planning::Budget,
};
use seismic_cpu::mapping::Cpu;
use seismic_cuda::mapping::Cuda;
use seismic_lang::sir::Program;
use seismic_metal::mapping::{EstimateModel, Limits, Metal};
use seismic_realization::executable::{
    ExecutableDialect, ResolvedPlan, ResolvedScheduleItem, StorageScope,
};

const FLAGS: &[&str] = &[
    "--fn",
    "--shape",
    "--element",
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
    "--strategy",
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
        Metal::new(limits, EstimateModel::default()).map_err(|error| error.to_string())?,
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
        Budget {
            strategy: options.strategy,
            ..Budget::default()
        },
    )
    .map_err(|error| error.to_string())
}

fn report<D: ExecutableDialect, A>(
    compiled: &Compiled<D, A>,
    target: &str,
    capacities: &str,
) -> String {
    let physical = &compiled.physical;
    fn counts<D: ExecutableDialect>(plan: &ResolvedPlan<D>) -> (usize, usize, usize) {
        let mut phases = 0;
        let mut launches = 0;
        let mut subplans = 0;
        for item in plan.items().iter() {
            match item {
                ResolvedScheduleItem::Phase(phase) => {
                    phases += 1;
                    launches += phase.launches.len();
                }
                ResolvedScheduleItem::Subplan(subplan) => {
                    subplans += 1;
                    let child = counts(&subplan.plan);
                    phases += child.0;
                    launches += child.1;
                    subplans += child.2;
                }
            }
        }
        (phases, launches, subplans)
    }
    fn resources<D: ExecutableDialect>(plan: &ResolvedPlan<D>, text: &mut String) {
        let device_bytes = plan
            .device_storage()
            .allocations
            .iter()
            .filter(|storage| storage.scope == StorageScope::Device)
            .map(|storage| storage.bytes)
            .sum::<u64>();
        for item in plan.items().iter() {
            match item {
                ResolvedScheduleItem::Phase(phase) => {
                    for launch in phase.launches.iter() {
                        let threads = launch
                            .geometry
                            .participants_per_workgroup
                            .iter()
                            .product::<u64>();
                        text.push_str(&format!(
                            "  launch#{}: workgroups={:?}, threads/group={threads}, device={device_bytes} B, workgroup={} B, private/thread={} B, bindings={}\n",
                            launch.id.0,
                            launch.geometry.workgroups,
                            launch.kernel.resources.workgroup_bytes,
                            launch.kernel.resources.private_bytes,
                            launch.binding_groups.len()
                        ));
                    }
                }
                ResolvedScheduleItem::Subplan(subplan) => resources(&subplan.plan, text),
            }
        }
    }
    let counts = counts(physical);
    let mut text = format!(
        "entry: {}\ntarget: {target}\ntarget facts: {capacities}\ncapability fingerprint: {}\nestimated cost: {}\noptimal: {}\nnumerical assessment: {:?}\n",
        compiled.logical.entry, compiled.logical.capability_fingerprint,
        physical.estimated_cost(), physical.optimal(), physical.numerical_assessment(),
    );
    text.push_str(&format!(
        "numerical evidence: {}:{}\n",
        physical.identity().precision.method_revision,
        physical.identity().precision.evidence_domain
    ));
    text.push_str("assignment:\n");
    for (choice, selected) in physical.identity().assignment.selections() {
        text.push_str(&format!(
            "  choice#{} = logical#{} / physical#{}\n",
            choice.0, selected.logical_alternative, selected.physical_alternative
        ));
    }
    for (symbol, value) in physical.identity().assignment.symbols() {
        text.push_str(&format!("  {symbol} = {value}\n"));
    }
    text.push_str("resources:\n");
    resources(physical, &mut text);
    text.push_str(&format!(
        "executable plan: {} phases, {} launches, {} nested plans\n",
        counts.0, counts.1, counts.2
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
                "\nnative CPU artifact: {} phases\n",
                compiled.native.kernel.phase_count()
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
    let assignment = &compiled.physical.identity().assignment;
    format!(
        "executable search of `{}` on {target}\n  target facts       {capacities}\n  selected choices   {}\n  resolved symbols   {}\n  selected cost      {}\n  optimum proven     {}\n",
        compiled.logical.entry,
        assignment.selections().len(),
        assignment.symbols().len(),
        compiled.physical.estimated_cost(),
        compiled.physical.optimal(),
    )
}
