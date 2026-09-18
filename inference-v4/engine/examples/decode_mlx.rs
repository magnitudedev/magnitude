//! Explicit-candidate, single-row decoder qualification. No automatic tuning or
//! throughput/parity claim is made by this token-ID/logit inspection tool.
use seismic_engine::{
    models::qwen35::{self, decoder::Decoder},
    weights::{
        descriptor::{Stored, WeightDescriptor},
        gguf::GgufArtifact,
        mlx::MlxArtifact,
        residency::{Importer, ResidentWeight},
    },
};
use seismic_lang::{lower::Options, types::DType};
use seismic_runtime::{Candidate, Device, plan::Diagnostic};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, rc::Rc, time::Instant};

enum Artifact {
    Mlx(MlxArtifact),
    Gguf(GgufArtifact),
}
impl Artifact {
    fn stored(
        &self,
        descriptor: &WeightDescriptor,
    ) -> Result<Stored, seismic_engine::weights::Error> {
        match self {
            Self::Mlx(a) => a.stored(descriptor),
            Self::Gguf(a) => a.stored(descriptor),
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(args.len() == 6 || args.len() == 7) {
        return Err("usage: decode_mlx cpu|cuda|metal ARTIFACT PIECE CONTEXT TOKEN_IDS_COMMA_SEPARATED [batch]".into());
    }
    if args.get(6).is_some_and(|mode| mode != "batch") {
        return Err("execution mode must be batch when specified".into());
    }
    let parse_piece = |value: &str| -> Result<Option<i64>, Box<dyn std::error::Error>> {
        if value == "whole" {
            return Ok(None);
        }
        let n = value.parse::<i64>()?;
        if n <= 0 {
            return Err("piece must be positive".into());
        }
        Ok(Some(n))
    };
    let piece = parse_piece(&args[3])?;
    let (device, candidate) = match args[1].as_str() {
        "cpu" => (
            Device::cpu(),
            Candidate::Cpu {
                loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
            },
        ),
        "cuda" => (
            Device::cuda(0)?,
            Candidate::Cuda {
                options: seismic_realization::ScalarOptions {
                    dispatch: seismic_realization::Dispatch::ParallelRoot,
                    loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
                },
                threads_per_block: 32,
            },
        ),
        #[cfg(target_os = "macos")]
        "metal" => {
            let device = Device::metal()?;
            let seismic_runtime::DeviceFacts::Metal(facts) = device.facts() else {
                unreachable!()
            };
            let mut config = match Candidate::Metal(Default::default()) {
                Candidate::Metal(c) => c,
                _ => unreachable!(),
            };
            config.piece = piece;
            config.max_threads_per_threadgroup = i64::try_from(facts.max_threads_per_threadgroup)?;
            config.max_threadgroup_bytes = i64::try_from(facts.max_threadgroup_bytes)?;
            (device, Candidate::Metal(config))
        }
        _ => return Err("unknown backend".into()),
    };
    let device = Rc::new(device);
    let artifact = if std::path::Path::new(&args[2]).is_dir() {
        Artifact::Mlx(MlxArtifact::open(&args[2])?)
    } else {
        Artifact::Gguf(GgufArtifact::open(&args[2])?)
    };
    let description = match &artifact {
        Artifact::Mlx(a) => qwen35::mlx::describe(a)?,
        Artifact::Gguf(a) => qwen35::gguf::inspect(a.directory(), a.identity())?,
    };
    let mut importer = Importer::new(device.clone(), candidate.clone())?;
    let mut cache: HashMap<(String, String, String), ResidentWeight> = HashMap::new();
    let start = Instant::now();
    let choices = Diagnostic {
        candidate: candidate.clone(),
        lowering: Options {
            piece,
            ..Default::default()
        },
    };
    let mut decoder = Decoder::compile_diagnostic(
        device,
        &description,
        |descriptor, target: DType| {
            let key = (
                descriptor.name.clone(),
                format!("{:?}", descriptor.transform),
                target.name().to_string(),
            );
            if let Some(weight) = cache.get(&key) {
                return Ok(weight.clone());
            }
            eprintln!(
                "import {} {:?} -> {}",
                descriptor.name,
                descriptor.shape,
                target.name()
            );
            let stored = artifact.stored(descriptor).map_err(|e| e.to_string())?;
            let resident = importer
                .import(descriptor, &stored, target)
                .map_err(|e| e.to_string())?;
            cache.insert(key, resident.clone());
            Ok(resident)
        },
        choices,
        args[4].parse()?,
        1,
    )?;
    drop(cache);
    drop(importer);
    println!(
        "{}",
        serde_json::json!({"kind":"compiled","artifact":description.artifact_identity.to_string(),"seconds":start.elapsed().as_secs_f64(),"kernel_count":decoder.compiled_kernel_count(),"pieces":args[3],"context":args[4],"backend":args[1]})
    );
    let mut state = decoder.state_store().create()?;
    for token in args[5].split(',').map(str::parse::<u32>) {
        let token = token?;
        let start = Instant::now();
        let (proposal, observations, batch) = if args.get(6).map(String::as_str) == Some("batch") {
            let (proposal, observation) = decoder.propose_batched(&mut state, token)?;
            (proposal, Vec::new(), Some(observation))
        } else {
            let (proposal, observations) = decoder.propose_observed(&mut state, token)?;
            (proposal, observations, None)
        };
        let seconds = start.elapsed().as_secs_f64();
        if let Some(observation) = batch {
            println!(
                "{}",
                serde_json::json!({"kind":"batch","input_token":token,"host_seconds":observation.host_seconds,
                "device_seconds":observation.device_seconds,"device_scope":format!("{:?}",observation.device_scope)})
            );
        }
        for observation in observations {
            println!(
                "{}",
                serde_json::json!({"kind":"step","input_token":token,"stage":observation.stage,"block":observation.block,
                "kernel":observation.step.entry,"host_seconds":observation.step.execution.host_seconds,
                "device_seconds":observation.step.execution.device_seconds,"device_scope":format!("{:?}",observation.step.execution.device_scope)})
            );
        }
        let logits = proposal.logits();
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("decoder produced nonfinite logits".into());
        }
        let mut top = (0..logits.len()).collect::<Vec<_>>();
        top.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b)));
        top.truncate(8);
        let record = serde_json::json!({"kind":"logits","input_token":token,"seconds":seconds,"logits_sha256":Sha256::digest(logits.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()).iter().map(|b|format!("{b:02x}")).collect::<String>(),"top":top.iter().map(|i|serde_json::json!({"token":i,"logit":logits[*i]})).collect::<Vec<_>>()});
        proposal.commit()?;
        println!("{record}");
    }
    Ok(())
}
