//! Retain the execution path's linked native images for external inspection.
use crate::{load_program, options};
use seismic_lang::lower::{lower_specialized, Options};
use seismic_realization::{Dispatch, ScalarOptions};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

pub fn native(args: &[String]) -> Result<(), String> {
    let mut remaining = Vec::new();
    let mut directory = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--artifact-dir" {
            if directory.is_some() {
                return Err("duplicate --artifact-dir".into());
            }
            directory = Some(PathBuf::from(
                args.next().ok_or("--artifact-dir requires a path")?,
            ));
        } else {
            remaining.push(arg.clone());
        }
    }
    let directory = directory.ok_or("--artifact-dir is required (must not already exist)")?;
    let options = options(&remaining)?;
    if !matches!(options.target.as_str(), "cuda" | "cpu") {
        return Err("native image export requires --target cuda|cpu".into());
    }
    let program = load_program(&options)?;
    let name = options.function.as_deref().ok_or("--fn is required")?;
    let lowered = lower_specialized(
        &program,
        name,
        &options.target,
        &options.shapes,
        &options.elements,
        &Options { piece: options.piece, ..Default::default() },
    )?;
    if options.target == "cpu" {
        return cpu(&options, &lowered, &directory);
    }
    let threads = options
        .threads_per_block
        .ok_or("--threads-per-block is required for CUDA")?;
    let device = seismic_cuda::Device::open(0)?;
    let artifacts = device.compile_artifacts(
        &lowered,
        ScalarOptions {
            dispatch: Dispatch::ParallelRoot,
            loads: options.loads,
        },
        threads,
    )?;
    fs::create_dir(&directory).map_err(|e| format!("{}: {e}", directory.display()))?;
    let mut phases = Vec::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        let stem = format!("phase-{index}");
        for (suffix, data) in [
            ("cubin", artifact.image.cubin.as_slice()),
            ("ptx", artifact.ptx.as_bytes()),
            ("log", artifact.image.compilation_log.as_bytes()),
        ] {
            let path = directory.join(format!("{stem}.{suffix}"));
            fs::write(&path, data).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        let native = &artifact.native;
        phases.push(json!({
            "phase":index,"source_statement":artifact.source_statement,
            "cubin":format!("{stem}.cubin"),"ptx":format!("{stem}.ptx"),
            "cubin_sha256":hash(&artifact.image.cubin),
            "ptx_sha256":hash(artifact.ptx.as_bytes()),
            "work_items":artifact.work_items,"threads_per_block":artifact.threads_per_block,
            "blocks":artifact.blocks,"registers_per_thread":native.registers_per_thread,
            "local_bytes_per_thread":native.local_bytes_per_thread,
            "shared_bytes_per_block":native.shared_bytes_per_block,
            "max_threads_per_block":native.max_threads_per_block,
            "max_active_blocks_per_multiprocessor":native.max_active_blocks_per_multiprocessor
        }));
    }
    let manifest = json!({
        "kind":"cuda_native_images","kernel":name,"shapes":options.shapes,
        "elements":options.elements.iter().map(|(n,e)|(n,e.to_string())).collect::<std::collections::BTreeMap<_,_>>(),
        "device":device.info.name,"compute_capability":device.info.compute_capability,
        "driver_version":device.info.driver_version,"load_strategy":format!("{:?}",options.loads),
        "lowering_decisions":format!("{:?}",lowered.decisions),"phases":phases,
        "optimality_certified":false,"performance_prediction":null,
        "limitations":["deterministic baseline lowering, not optimized selection", "no execution or timing performed", "native capacities do not establish instruction service rates or latency"]
    });
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{manifest}");
    Ok(())
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn cpu(
    options: &crate::Options,
    lowered: &seismic_lang::lowered_ir::LoweredIr,
    directory: &std::path::Path,
) -> Result<(), String> {
    let artifact = seismic_cpu::compile_artifact(lowered, options.loads)?;
    fs::create_dir(directory).map_err(|e| format!("{}: {e}", directory.display()))?;
    for (name, data) in [
        ("kernel.linked.bin", artifact.machine_code.as_slice()),
        (
            "kernel.unrelocated.bin",
            artifact.unrelocated_code.as_slice(),
        ),
        ("kernel.clif", artifact.ir.as_bytes()),
        ("kernel.optimized.clif", artifact.optimized_ir.as_bytes()),
        ("kernel.vcode", artifact.vcode.as_bytes()),
    ] {
        let path = directory.join(name);
        fs::write(&path, data).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let manifest = json!({
        "kind":"cpu_native_image", "kernel":lowered.name, "shapes":options.shapes,
        "elements":options.elements.iter().map(|(n,e)|(n,e.to_string())).collect::<std::collections::BTreeMap<_,_>>(),
        "load_strategy":format!("{:?}",options.loads), "lowering_decisions":format!("{:?}",lowered.decisions),
        "target":artifact.target,"compiler_flags":artifact.compiler_flags,"isa_flags":artifact.isa_flags,
        "linked_sha256":hash(&artifact.machine_code), "unrelocated_sha256":hash(&artifact.unrelocated_code),
        "ir_sha256":hash(artifact.ir.as_bytes()),"optimized_ir_sha256":hash(artifact.optimized_ir.as_bytes()),
        "native_bytes":artifact.machine_code.len(),"frame_bytes":artifact.frame_bytes,"scratch_bytes":artifact.scratch_bytes,
        "block_starts":artifact.block_starts,"block_edges":artifact.block_edges,
        "origins":artifact.origins.iter().map(|o|json!({"start":o.start,"end":o.end,"ssa_instruction":o.ssa_instruction})).collect::<Vec<_>>(),
        "relocations":artifact.relocations.iter().map(|r|format!("{r:?}")).collect::<Vec<_>>(),
        "imports":artifact.imports.iter().map(|i|format!("{i:?}")).collect::<Vec<_>>(),
        "optimality_certified":false,"performance_prediction":null,
        "limitations":["deterministic baseline lowering, not optimized selection", "linked bytes contain process-local addresses and cannot be reloaded", "imported math implementations are outside the retained function image", "source attribution is not a complete dependency or instruction cost model", "no execution or timing performed"]
    });
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{manifest}");
    Ok(())
}
