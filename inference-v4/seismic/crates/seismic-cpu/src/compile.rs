//! Direct typed-kernel to native compilation.

use crate::command::{CompiledKernel, JitMemory};
use crate::emit;
use crate::{numeric, Cpu};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use seismic_compiler::errors::NativeCompilationError;
use seismic_compiler::kernel::Kernel;
use seismic_compiler::target::{DeviceContract, KernelEmissionLayout};
use seismic_lang::types::DType;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;

pub struct NativeCandidate {
    pub kernel: CompiledKernel,
    pub artifact_digest: [u8; 32],
    pub frame_size: u64,
    pub code_bytes: u64,
    pub metadata_bytes: u64,
    pub compilation_ns: u64,
}

pub(crate) fn compile_kernel(
    target: &DeviceContract<Cpu>,
    kernel: &Kernel<Cpu>,
    layout: &KernelEmissionLayout,
) -> Result<NativeCandidate, NativeCompilationError> {
    let started = Instant::now();
    let isa = target.facts().codegen.isa()?;
    let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
    for (name, address) in numeric::host_symbols() {
        builder.symbol(name, address);
    }
    let mut module = JITModule::new(builder);
    let emitted = emit::emit(
        &mut module,
        kernel,
        layout,
        target.facts().codegen.call_conv,
    )?;
    let declared = module
        .declare_function(
            "seismic_cpu_kernel",
            Linkage::Local,
            &emitted.function.signature,
        )
        .map_err(toolchain)?;
    let mut context = module.make_context();
    context.func = emitted.function;
    context.func.name = cranelift_codegen::ir::UserFuncName::user(0, declared.as_u32());
    for import in emitted.imports {
        let imported = module
            .declare_function(import.name, Linkage::Import, &import.signature)
            .map_err(toolchain)?;
        let native = module.declare_func_in_func(imported, &mut context.func);
        context.func.dfg.ext_funcs[import.reference] = context.func.dfg.ext_funcs[native].clone();
    }
    module
        .define_function(declared, &mut context)
        .map_err(|error| {
            NativeCompilationError::ToolchainFailure(format!("Cranelift definition: {error:?}"))
        })?;
    let compiled = context.compiled_code().ok_or_else(|| {
        NativeCompilationError::ToolchainFailure(
            "Cranelift did not retain authoritative compiled-code metadata".into(),
        )
    })?;
    let frame_size = u64::from(compiled.frame_size);
    let code_bytes = u64::from(compiled.code_info().total_size);
    if code_bytes == 0 {
        return Err(NativeCompilationError::ToolchainFailure(
            "Cranelift produced an empty native function".into(),
        ));
    }
    let artifact_digest: [u8; 32] = Sha256::digest(compiled.code_buffer()).into();
    module.clear_context(&mut context);
    module.finalize_definitions().map_err(toolchain)?;
    let entry = unsafe {
        std::mem::transmute::<*const u8, crate::workers::LaunchEntry>(
            module.get_finalized_function(declared),
        )
    };
    let metadata_bytes = layout_metadata_bytes(layout)?;
    Ok(NativeCandidate {
        kernel: CompiledKernel {
            entry,
            memory: Arc::new(JitMemory::new(module)),
            layout: layout.clone(),
        },
        artifact_digest,
        frame_size,
        code_bytes,
        metadata_bytes,
        compilation_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

fn layout_metadata_bytes(layout: &KernelEmissionLayout) -> Result<u64, NativeCompilationError> {
    let bytes = std::mem::size_of::<CompiledKernel>()
        .checked_add(std::mem::size_of::<KernelEmissionLayout>())
        .and_then(|bytes| {
            bytes.checked_add(
                layout.bindings.len()
                    * std::mem::size_of::<seismic_compiler::target::BindingEmissionLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.locals.len()
                    * std::mem::size_of::<seismic_compiler::target::LocalEmissionLayout>(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.addressable_resources.len()
                    * std::mem::size_of::<
                        seismic_compiler::target::AddressableResourceEmissionLayout,
                    >(),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(layout.scalar_args.len() * std::mem::size_of::<DType>())
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout.result_slots.len()
                    * std::mem::size_of::<(seismic_compiler::schedule::AnyScalarSlot, DType)>(),
            )
        })
        .ok_or_else(|| {
            NativeCompilationError::ToolchainFailure(
                "CPU native artifact metadata footprint exceeds usize".into(),
            )
        })?;
    u64::try_from(bytes).map_err(|_| {
        NativeCompilationError::ToolchainFailure(
            "CPU native artifact metadata footprint exceeds u64".into(),
        )
    })
}

fn toolchain(error: impl std::fmt::Display) -> NativeCompilationError {
    NativeCompilationError::ToolchainFailure(error.to_string())
}
