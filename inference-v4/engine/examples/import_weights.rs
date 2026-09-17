//! Qualification only: import representative real MLX roles and independently
//! verify bytes/transforms. This is not model inference or a benchmark.
use seismic_engine::{
    models::qwen35::{self, FeedForwardWeights, MixerWeights},
    weights::{
        descriptor::{Stored, Transform},
        mlx::MlxArtifact,
        residency::Importer,
    },
};
use seismic_lang::types::DType;
use seismic_runtime::{Candidate, Device};
use std::rc::Rc;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let (device, candidate) = match args.get(1).map(String::as_str) {
        Some("cpu") => (
            Device::cpu(),
            Candidate::Cpu {
                loads: seismic_realization::LoadStrategy::Materialize,
            },
        ),
        Some("cuda") => (
            Device::cuda(0)?,
            Candidate::Cuda {
                options: seismic_realization::ScalarOptions {
                    dispatch: seismic_realization::Dispatch::ParallelRoot,
                    loads: seismic_realization::LoadStrategy::Materialize,
                },
                threads_per_block: 32,
            },
        ),
        #[cfg(target_os = "macos")]
        Some("metal") => (Device::metal()?, Candidate::Metal(Default::default())),
        _ => return Err("usage: import_weights cpu|cuda|metal MLX_DIRECTORY".into()),
    };
    let artifact = MlxArtifact::open(args.get(2).ok_or("missing MLX directory")?)?;
    let model = qwen35::mlx::describe(&artifact)?;
    let block = &model.blocks[0];
    let MixerWeights::Recurrent(recurrent) = &block.mixer else {
        return Err("first block is not recurrent".into());
    };
    let FeedForwardWeights::Dense(feedforward) = &block.feedforward else {
        return Err("first feedforward is not dense".into());
    };
    let mut importer = Importer::new(Rc::new(device), candidate)?;
    for descriptor in [&block.input_norm, &recurrent.decay, &feedforward.gate] {
        let stored = artifact.stored(descriptor)?;
        let resident = importer.import(descriptor, &stored, DType::F32)?;
        match stored {
            Stored::GgmlBlocks {..} => return Err("this qualification example expects an MLX artifact".into()),
            Stored::Dense(tensor) => {
                let input = tensor.read()?;
                let mut output = vec![0; resident.plane("").unwrap().len()];
                resident.plane("").unwrap().read(&mut output)?;
                let mut maximum = 0f64;
                for (raw, out) in input
                    .chunks_exact(tensor.dtype.bytes() as usize)
                    .zip(output.chunks_exact(4))
                {
                    let value = match tensor.dtype {
                        DType::BF16 => {
                            f32::from_bits(u32::from(u16::from_le_bytes(raw.try_into()?)) << 16)
                        }
                        DType::F32 => f32::from_le_bytes(raw.try_into()?),
                        _ => return Err("reference dtype unsupported".into()),
                    };
                    let expected = match descriptor.transform {
                        Transform::Identity => value,
                        Transform::NegativeExp => -(f64::from(value).exp() as f32),
                    };
                    let got = f32::from_le_bytes(out.try_into()?);
                    if !got.is_finite() || (got - expected).abs() > expected.abs().max(1.0) * 3e-7 {
                        return Err(format!("{}: {got} != {expected}", descriptor.name).into());
                    }
                    maximum = maximum.max(f64::from((got - expected).abs()));
                }
                println!(
                    "{}",
                    serde_json::json!({"name":descriptor.name,"shape":descriptor.shape,"max_absolute_error":maximum,"bytes":output.len()})
                );
            }
            Stored::AffinePlanes {
                codes,
                scales,
                biases,
                ..
            } => {
                for (name, tensor) in [("words", codes), ("scale", scales), ("bias", biases)] {
                    let expected = tensor.read()?;
                    let mut got = vec![0; expected.len()];
                    resident.plane(name).unwrap().read(&mut got)?;
                    if got != expected {
                        return Err("packed bytes changed".into());
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"name":descriptor.name,"shape":descriptor.shape,"packed_planes_exact":true})
                );
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"artifact_identity":artifact.identity(),"qualification":"representative resident imports only"})
    );
    Ok(())
}
