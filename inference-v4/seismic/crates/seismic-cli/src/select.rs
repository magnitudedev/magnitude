//! Unified semantic -> logical -> plan-space -> physical-plan -> native compilation
//! inspection commands.

use crate::{load_program, options, Options};
use seismic_compiler::{
    pipeline::{self, Backend, Compiled},
    planning::Budget,
};
use seismic_cpu::Cpu;
use seismic_cuda::{CudaCompiler, Device as CudaDevice};
use seismic_lang::sir::Program;
use seismic_realization::ids::{DenseIndex, LaunchIx};
use seismic_realization::kernel::ExecutableDialect;
use seismic_realization::physical::PhysicalStep;

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
];

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

fn cuda() -> Result<(CudaDevice, String), String> {
    let device = CudaDevice::open(0)
        .map_err(|error| format!("CUDA target needs a CUDA device: {error}"))?;
    let origin = format!(
        "device `{}` (max_threads_per_block={}, max_grid_x={}, warp_size={})",
        device.info.name,
        device.info.max_threads_per_block,
        device.info.max_grid_x,
        device.info.warp_size
    );
    Ok((device, origin))
}

#[cfg(target_os = "macos")]
fn metal() -> Result<(seismic_metal::runtime::Device, String), String> {
    let device = seismic_metal::runtime::Device::open()
        .map_err(|error| format!("Metal target needs a Metal device: {error}"))?;
    let info = device.info();
    let origin = format!(
        "device `{}` (max_threads_per_threadgroup={}, max_threadgroup_bytes={})",
        info.name, info.max_threads_per_threadgroup, info.max_threadgroup_bytes
    );
    Ok((device, origin))
}

fn compile<B: Backend>(
    options: &Options,
    program: &Program,
    backend: &B,
) -> Result<Compiled<B::Dialect, B::NativeArtifact>, String> {
    let domain = options.domain(program)?;
    pipeline::compile(
        program,
        &domain,
        &options.precision,
        backend,
        &[],
        Budget::default(),
    )
    .map_err(|error| error.to_string())
}

fn report<D: ExecutableDialect, A>(
    compiled: &Compiled<D, A>,
    target: &str,
    capacities: &str,
) -> String {
    let physical = &compiled.physical;
    let identity = physical.identity();
    let mut text = format!(
        "entry: {}\ntarget: {target}\ntarget facts: {capacities}\ncapability fingerprint: {}\nestimated cost: {}\noptimal: {}\nnumerical assessment: {:?}\n",
        compiled.logical.entry(),
        identity.target.capability_fingerprint,
        physical.estimated_cost(),
        physical.optimal(),
        physical.numerical(),
    );
    text.push_str("assignment:\n");
    for (occurrence, selected) in &identity.selections {
        let selected = selected
            .map(|strategy| format!("physical#{}", strategy.0))
            .unwrap_or_else(|| "none".to_string());
        text.push_str(&format!(
            "  occurrence#{} = {}\n",
            occurrence.0, selected
        ));
    }
    text.push_str("resources:\n");
    let device_bytes = physical.resources().arena_bytes;
    for launch in physical.launches() {
        text.push_str(&format!(
            "  launch#{}: device={device_bytes} B, workgroup={} B, private/participant={} B, bindings={}\n",
            launch.id.index(),
            launch.resources.workgroup_bytes,
            launch.resources.private_bytes_per_participant,
            launch.bindings.len(),
        ));
    }
    let nested = {
        fn calls<D: ExecutableDialect>(steps: &[PhysicalStep<D>]) -> usize {
            steps
                .iter()
                .map(|step| match step {
                    PhysicalStep::Call(call) => 1 + calls(&call.body.steps),
                    PhysicalStep::If(branch) => {
                        calls(&branch.then_schedule.steps) + calls(&branch.else_schedule.steps)
                    }
                    PhysicalStep::Repeat(repeat) => calls(&repeat.body.steps),
                    _ => 0,
                })
                .sum()
        }
        calls(&physical.schedule().steps)
    };
    text.push_str(&format!(
        "executable plan: {} launches, {} nested calls\n",
        physical.launches().len(),
        nested,
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
            let (device, facts) = cuda()?;
            let backend = CudaCompiler::new(&device).map_err(|error| error.to_string())?;
            report(&compile(&options, &program, &backend)?, "cuda", &facts)
        }
        #[cfg(target_os = "macos")]
        "metal" => {
            let (device, facts) = metal()?;
            let backend = seismic_metal::catalog::MetalCompiler::from_device(&device);
            report(&compile(&options, &program, &backend)?, "metal", &facts)
        }
        #[cfg(not(target_os = "macos"))]
        "metal" => return Err("Metal target requires macOS".to_string()),
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
                compiled.native.launch_count()
            ));
            text
        }
        "cuda" => {
            let (device, _) = cuda()?;
            let backend = CudaCompiler::new(&device).map_err(|error| error.to_string())?;
            let compiled = compile(&options, &program, &backend)?;
            compiled
                .native
                .launches()
                .iter()
                .enumerate()
                .map(|(index, launch)| {
                    format!("// launch {index}\n{}\n", launch.launch.ptx)
                })
                .collect()
        }
        #[cfg(target_os = "macos")]
        "metal" => {
            let (device, _) = metal()?;
            let backend = seismic_metal::catalog::MetalCompiler::from_device(&device);
            let compiled = compile(&options, &program, &backend)?;
            (0..compiled.native.launch_count())
                .map(|index| {
                    let launch = compiled.native.launch(LaunchIx::from_index(index));
                    format!("// launch {index}\n{}\n", launch.kernel())
                })
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        "metal" => return Err("Metal target requires macOS".to_string()),
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
            let (device, facts) = cuda()?;
            let backend = CudaCompiler::new(&device).map_err(|error| error.to_string())?;
            search_report(&compile(&options, &program, &backend)?, "cuda", &facts)
        }
        #[cfg(target_os = "macos")]
        "metal" => {
            let (device, facts) = metal()?;
            let backend = seismic_metal::catalog::MetalCompiler::from_device(&device);
            search_report(&compile(&options, &program, &backend)?, "metal", &facts)
        }
        #[cfg(not(target_os = "macos"))]
        "metal" => return Err("Metal target requires macOS".to_string()),
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
        compiled.logical.entry(),
        physical.identity().selections.len(),
        physical.estimated_cost(),
        physical.optimal(),
    )
}
