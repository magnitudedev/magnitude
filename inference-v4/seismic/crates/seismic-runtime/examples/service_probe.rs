//! Explicit calibration experiment. This does not infer physical DRAM traffic,
//! saturated capacity or general applicability from a copy kernel's throughput.
use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_realization::{Dispatch, LoadStrategy, ScalarOptions};
use seismic_runtime::{Candidate, Device};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
const SOURCE:&str="fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    value = load(x[row:row+1])\n    store(value,out[row:row+1])\n";
fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 6 {
        return Err("usage: service_probe cpu|cuda|metal ELEMENTS REPEATS WORKGROUP_SIZE ACCESS_ANALYSIS_BUDGET (explicit experiment inputs)".into());
    }
    let n = args[2].parse::<usize>()?;
    let repeats = args[3].parse::<usize>()?;
    let group = args[4].parse::<u32>()?;
    let budget = args[5].parse::<usize>()?;
    if n == 0 || repeats == 0 {
        return Err("positive shape/repeat count required".into());
    }
    let (device, candidate) = match args[1].as_str() {
        "cpu" => (
            Device::cpu(),
            Candidate::Cpu {
                loads: LoadStrategy::BorrowProvenReadOnly,
            },
        ),
        "cuda" => (
            Device::cuda(0)?,
            Candidate::Cuda {
                options: ScalarOptions {
                    dispatch: Dispatch::ParallelRoot,
                    loads: LoadStrategy::BorrowProvenReadOnly,
                },
                threads_per_block: group,
            },
        ),
        #[cfg(target_os = "macos")]
        "metal" => {
            let config = seismic_metal::execution::Config {
                sg_per_tg: i64::from(group),
                ..Default::default()
            };
            (Device::metal()?, Candidate::Metal(config))
        }
        _ => return Err("unsupported backend".into()),
    };
    let program = compile(
        &[SourceFile {
            path: "copy-probe.seismic.portable".into(),
            scope: Scope::Portable,
            text: SOURCE.into(),
        }],
        &[],
    )
    .map_err(|e| format!("{e:?}"))?;
    let shapes = HashMap::from([("N".into(), i64::try_from(n)?)]);
    let start = std::time::Instant::now();
    let memory =
        seismic_accounting::memory::derive(&program, "copy", &shapes, &HashMap::new(), budget)?;
    let analysis_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let lowered = seismic_lang::lower::lower(&program, "copy", device.backend(), &shapes)?;
    let mut kernel = device.compile(&lowered, candidate)?;
    let compile_seconds = start.elapsed().as_secs_f64();
    let values = (0..n)
        .map(|i| (i % 127) as f32)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let input = device.buffer_from(&values)?;
    let output = device.buffer(values.len())?;
    let buffers = [input, output.clone()];
    let mut samples = Vec::new();
    for _ in 0..repeats {
        let observation = kernel.execute_observed(&buffers, &[])?;
        samples.push(serde_json::json!({"host_seconds":observation.host_seconds,"device_seconds":observation.device_seconds,"device_scope":observation.device_scope.map(|scope|format!("{scope:?}"))}));
    }
    let mut got = vec![0; values.len()];
    output.read(&mut got)?;
    if got != values {
        return Err("probe copy verification failed".into());
    }
    let accesses=memory.accesses.iter().map(|(backing,a)|serde_json::json!({"backing":backing,"unique_read_bytes":a.reads.bytes(),"unique_write_bytes":a.writes.bytes()})).collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::json!({"probe":"seismic-copy-v1","source_sha256":hex(Sha256::digest(SOURCE.as_bytes())),"executable_sha256":hex(Sha256::digest(std::fs::read(std::env::current_exe()?)?)),"backend":device.backend(),"device_facts":format!("{:?}",device.facts()),"elements":n,"workgroup_input":group,"workgroup_input_units":if device.backend()=="metal" {"subgroups per threadgroup"} else if device.backend()=="cuda" {"threads per block"} else {"unused on sequential CPU"},"conditions":"same resident source/destination reused; no cache flush; all samples retained in order; no claim of saturation","analysis_seconds":analysis_seconds,"compile_seconds":compile_seconds,"logical_accesses":accesses,"accesses_exact":memory.is_exact(),"accesses_unavailable":memory.unavailable,"samples":samples,"verified":true,"physical_traffic":"unavailable; logical access bytes are not DRAM or cache transaction measurements"})
    );
    Ok(())
}
