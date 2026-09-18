//! Native CPU compilation of checked Seismic through Cranelift. The initial scalar
//! realization establishes executable semantics; SIMD and microkernel choices follow
//! through the same lowering/accounting contracts. No reference interpreter is called.

use cranelift_codegen::{
    ir::{self, types},
    settings::{self, Configurable},
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use seismic_lang::abi::ScalarParameter;
use seismic_lang::lowered_ir::LoweredIr;

mod buffer;
pub use buffer::Buffer;
pub use seismic_realization::BufferSpec;

/// Executable memory and scratch have one owner. Mutable invocation serializes reuse
/// of its scratch. Generated pointers cannot escape this wrapper.
pub struct Kernel {
    _memory: ExecutableMemory,
    entry: unsafe extern "C" fn(*const *mut u8, *const u64, *mut u8) -> i32,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    scratch: Vec<u8>,
    artifact: NativeArtifact,
}

/// Native evidence from the exact JIT compilation used for execution.
/// Linked bytes contain process-local relocations and are inspection evidence,
/// not a portable executable or a cache key.
pub struct NativeArtifact {
    pub ir: String,
    pub optimized_ir: String,
    pub machine_code: Vec<u8>,
    pub unrelocated_code: Vec<u8>,
    pub relocations: Vec<cranelift_codegen::FinalizedMachReloc>,
    pub vcode: String,
    pub frame_bytes: u32,
    /// Compiler attribution, not a one-to-one instruction mapping or a cost model.
    pub origins: Vec<NativeOrigin>,
    pub block_starts: Vec<u32>,
    pub block_edges: Vec<(u32, u32)>,
    pub target: String,
    pub compiler_flags: String,
    pub isa_flags: Vec<String>,
    pub imports: Vec<seismic_realization::MathFunction>,
    pub scratch_bytes: usize,
}
pub struct NativeOrigin {
    pub start: u32,
    pub end: u32,
    pub ssa_instruction: Option<u32>,
}
struct CompiledCode {
    memory: ExecutableMemory,
    entry: unsafe extern "C" fn(*const *mut u8, *const u64, *mut u8) -> i32,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    artifact: NativeArtifact,
}
impl Kernel {
    pub fn native_artifact(&self) -> &NativeArtifact {
        &self.artifact
    }

    pub fn buffers(&self) -> &[BufferSpec] {
        &self.buffers
    }
    pub fn scalars(&self) -> &[ScalarParameter] {
        &self.scalars
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.len()
    }
    pub fn run(&mut self, buffers: &mut [&mut [u8]], scalars: &[f64]) -> Result<(), String> {
        if buffers.len() != self.buffers.len() || scalars.len() != self.scalars.len() {
            return Err("CPU binding count does not match compiled signature".into());
        }
        for (b, s) in buffers.iter().zip(&self.buffers) {
            if !(b.as_ptr() as usize).is_multiple_of(s.alignment) {
                return Err("CPU buffer violates typed storage alignment".into());
            }
            if b.len() < s.bytes {
                return Err(format!(
                    "buffer {}.{} has {} bytes; needs {}",
                    s.parameter,
                    s.plane,
                    b.len(),
                    s.bytes
                ));
            }
        }
        let scalar_words = seismic_realization::encode_scalars(&self.scalars, scalars)?;
        let pointers = buffers
            .iter_mut()
            .map(|b| b.as_mut_ptr())
            .collect::<Vec<_>>();
        self.invoke(&pointers, &scalar_words)
    }
    /// Execute over retained storage views. Each backing allocation is borrowed
    /// once; aliases use raw pointers rather than overlapping Rust mutable slices.
    /// The kernel's checked effects still govern whether any aliasing is legal.
    pub fn run_resident(&mut self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        if buffers.len() != self.buffers.len() {
            return Err("CPU resident binding count mismatch".into());
        }
        let words = seismic_realization::encode_scalars(&self.scalars, scalars)?;
        let mut roots = Vec::new();
        let mut root_indices = Vec::new();
        for (buffer, spec) in buffers.iter().zip(&self.buffers) {
            if buffer.len() < spec.bytes {
                return Err(format!(
                    "CPU resident {}.{} needs {} bytes, has {}",
                    spec.parameter,
                    spec.plane,
                    spec.bytes,
                    buffer.len()
                ));
            }
            if buffer.offset % spec.alignment != 0 {
                return Err("CPU resident view violates typed storage alignment".into());
            }
            let index = roots
                .iter()
                .position(|root| std::rc::Rc::ptr_eq(root, &buffer.storage))
                .unwrap_or_else(|| {
                    roots.push(buffer.storage.clone());
                    roots.len() - 1
                });
            root_indices.push(index);
        }
        let mut borrows = roots
            .iter()
            .map(|root| {
                root.try_borrow_mut()
                    .map_err(|_| "CPU resident storage is already in use".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pointers = buffers
            .iter()
            .zip(root_indices)
            .map(|(buffer, index)| {
                // Views were checked at construction; the retained mutable borrow
                // prevents host access or reallocation through physical completion.
                unsafe { borrows[index].as_mut_ptr().cast::<u8>().add(buffer.offset) }
            })
            .collect::<Vec<_>>();
        self.invoke(&pointers, &words)
    }
    fn invoke(&mut self, pointers: &[*mut u8], scalar_words: &[u64]) -> Result<(), String> {
        // All pointer capacities and scalar encodings match the checked compiled ABI.
        // The emitter rejects unresolved indexing and unsupported control semantics.
        let status = unsafe {
            (self.entry)(
                pointers.as_ptr(),
                scalar_words.as_ptr(),
                self.scratch.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(
                "CPU execution rejected an out-of-bounds view; outputs may be partially written"
                    .into(),
            );
        }
        Ok(())
    }
}
struct ExecutableMemory(Option<JITModule>);
impl std::ops::Deref for ExecutableMemory {
    type Target = JITModule;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("live executable owner")
    }
}
impl std::ops::DerefMut for ExecutableMemory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("live executable owner")
    }
}
impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            // No generated pointer escapes its Kernel owner. Also release allocations
            // on failed compilation/finalization before a Kernel can be constructed.
            unsafe {
                module.free_memory();
            }
        }
    }
}

extern "C" fn exp(x: f32) -> f32 {
    x.exp()
}
extern "C" fn log(x: f32) -> f32 {
    x.ln()
}
extern "C" fn sin(x: f32) -> f32 {
    x.sin()
}
extern "C" fn cos(x: f32) -> f32 {
    x.cos()
}

/// Prepare the scalar execution without compiling native instructions.
pub fn prepare(
    lowered: &LoweredIr,
    loads: seismic_realization::LoadStrategy,
) -> Result<seismic_realization::ScalarProgram, String> {
    if lowered.backend != "cpu" {
        return Err("CPU preparation requires a CPU-lowered function".into());
    }
    seismic_compiler::scalar_candidate(
        lowered,
        host_call_conv()?,
        seismic_realization::ScalarOptions {
            dispatch: seismic_realization::Dispatch::Sequential,
            loads,
        },
    )
}
fn host_call_conv() -> Result<seismic_realization::CallConv, String> {
    let target = cranelift_native::builder_with_options(false).map_err(str::to_owned)?;
    Ok(seismic_realization::CallConv::triple_default(target.triple()))
}

pub fn compile(lowered: &LoweredIr) -> Result<Kernel, String> {
    compile_candidate(lowered, seismic_realization::LoadStrategy::Materialize)
}
pub fn compile_candidate(
    lowered: &LoweredIr,
    loads: seismic_realization::LoadStrategy,
) -> Result<Kernel, String> {
    compile_execution(prepare(lowered, loads)?)
}
/// Compile an already selected scalar program; no source preparation occurs here.
pub fn compile_execution(program: seismic_realization::ScalarProgram) -> Result<Kernel, String> {
    let CompiledCode {
        memory,
        entry,
        buffers,
        scalars,
        artifact,
    } = compile_code(program)?;
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(artifact.scratch_bytes)
        .map_err(|e| format!("CPU scratch allocation: {e}"))?;
    scratch.resize(artifact.scratch_bytes, 0);
    Ok(Kernel {
        _memory: memory,
        entry,
        buffers,
        scalars,
        scratch,
        artifact,
    })
}
/// Inspect the execution compiler without allocating invocation scratch.
pub fn compile_artifact(
    lowered: &LoweredIr,
    loads: seismic_realization::LoadStrategy,
) -> Result<NativeArtifact, String> {
    Ok(compile_code(prepare(lowered, loads)?)?.artifact)
}
fn compile_code(program: seismic_realization::ScalarProgram) -> Result<CompiledCode, String> {
    if program.dispatch != seismic_realization::Dispatch::Sequential
        || program.function.signature.call_conv != host_call_conv()?
        || program.function.signature.params.len() != 3
        || program.function.signature.params.iter().any(|p| p.value_type != types::I64)
        || program.function.signature.returns.len() != 1
        || program.function.signature.returns[0].value_type != types::I32
    {
        return Err("selected CPU execution has an incompatible invocation ABI".into());
    }
    let mut flags = settings::builder();
    flags
        .set("use_colocated_libcalls", "false")
        .map_err(|e| e.to_string())?;
    flags.set("is_pic", "false").map_err(|e| e.to_string())?;
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
    flags
        .set("machine_code_cfg_info", "true")
        .map_err(|e| e.to_string())?;
    let isa = cranelift_native::builder()
        .map_err(str::to_owned)?
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())?;
    let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
    for (name, address) in [
        ("seismic_exp", exp as *const u8),
        ("seismic_log", log as *const u8),
        ("seismic_sin", sin as *const u8),
        ("seismic_cos", cos as *const u8),
    ] {
        jb.symbol(name, address);
    }
    let mut module = ExecutableMemory(Some(JITModule::new(jb)));
    let mut context = module.make_context();
    context.set_disasm(true);
    let pointer_type = module.target_config().pointer_type();
    if pointer_type != types::I64 {
        return Err("CPU backend currently requires 64-bit pointers".into());
    }
    let seismic_realization::ScalarProgram {
        function,
        buffers,
        scalars,
        scratch_bytes,
        imports,
        ..
    } = program;
    context.func = function;
    let math_imports = imports.iter().map(|(_, op)| *op).collect();
    for (reference, operation) in imports {
        let signature =
            context.func.dfg.signatures[context.func.dfg.ext_funcs[reference].signature].clone();
        let id = module
            .declare_function(operation.symbol(), Linkage::Import, &signature)
            .map_err(|e| e.to_string())?;
        let native = module.declare_func_in_func(id, &mut context.func);
        context.func.dfg.ext_funcs[reference] = context.func.dfg.ext_funcs[native].clone();
    }
    let id = module
        .declare_function("seismic_kernel", Linkage::Local, &context.func.signature)
        .map_err(|e| e.to_string())?;
    context.func.name = ir::UserFuncName::user(0, id.as_u32());
    // Use the input SSA instruction ID as the native compiler's source token.
    // Optimizations can merge/remove instructions; retain the compiler's actual
    // attribution without manufacturing a one-to-one correspondence.
    let instructions: Vec<_> = context
        .func
        .layout
        .blocks()
        .flat_map(|block| context.func.layout.block_insts(block))
        .collect();
    for inst in instructions {
        context
            .func
            .set_srcloc(inst, ir::SourceLoc::new(inst.as_u32()));
    }
    let ir = context.func.display().to_string();
    module
        .define_function(id, &mut context)
        .map_err(|e| format!("CPU compilation: {e}\n{ir}"))?;
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let entry = unsafe {
        std::mem::transmute::<
            *const u8,
            unsafe extern "C" fn(*const *mut u8, *const u64, *mut u8) -> i32,
        >(module.get_finalized_function(id))
    };
    let compiled = context
        .compiled_code()
        .ok_or("CPU compiler returned no native code")?;
    let code = compiled.code_buffer();
    // Finalization applies relocations in the retained JIT allocation. The known
    // code length comes from this same compilation; the module stays alive here.
    let machine_code =
        unsafe { std::slice::from_raw_parts(module.get_finalized_function(id), code.len()) }
            .to_vec();
    let artifact = NativeArtifact {
        ir,
        optimized_ir: context.func.display().to_string(),
        machine_code,
        unrelocated_code: code.to_vec(),
        relocations: compiled.buffer.relocs().to_vec(),
        vcode: compiled
            .vcode
            .clone()
            .ok_or("CPU compiler omitted requested instruction listing")?,
        frame_bytes: compiled.frame_size,
        origins: compiled
            .buffer
            .get_srclocs_sorted()
            .iter()
            .map(|origin| NativeOrigin {
                start: origin.start,
                end: origin.end,
                ssa_instruction: (!origin.loc.is_default()).then(|| origin.loc.bits()),
            })
            .collect(),
        block_starts: compiled.bb_starts.clone(),
        block_edges: compiled.bb_edges.clone(),
        target: module.isa().triple().to_string(),
        compiler_flags: module.isa().flags().to_string(),
        isa_flags: module
            .isa()
            .isa_flags()
            .iter()
            .map(ToString::to_string)
            .collect(),
        imports: math_imports,
        scratch_bytes,
    };
    Ok(CompiledCode {
        memory: module,
        entry,
        buffers,
        scalars,
        artifact,
    })
}
