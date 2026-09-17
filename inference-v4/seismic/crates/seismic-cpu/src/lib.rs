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
use seismic_lang::lower::Lowered;

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
    pub ir: String,
}

impl Kernel {
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

pub fn compile(lowered: &Lowered) -> Result<Kernel, String> {
    compile_candidate(lowered, seismic_realization::LoadStrategy::Materialize)
}
pub fn compile_candidate(
    lowered: &Lowered,
    loads: seismic_realization::LoadStrategy,
) -> Result<Kernel, String> {
    if lowered.backend != "cpu" {
        return Err("CPU compiler requires a CPU-lowered function".into());
    }
    let mut flags = settings::builder();
    flags
        .set("use_colocated_libcalls", "false")
        .map_err(|e| e.to_string())?;
    flags.set("is_pic", "false").map_err(|e| e.to_string())?;
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
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
    let pointer_type = module.target_config().pointer_type();
    if pointer_type != types::I64 {
        return Err("CPU backend currently requires 64-bit pointers".into());
    }
    let program = seismic_compiler::scalar_candidate(
        lowered,
        context.func.signature.call_conv,
        seismic_realization::ScalarOptions {
            dispatch: seismic_realization::Dispatch::Sequential,
            loads,
        },
    )?;
    let seismic_realization::ScalarProgram {
        function,
        buffers,
        scalars,
        scratch_bytes,
        imports,
        ..
    } = program;
    context.func = function;
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
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(scratch_bytes)
        .map_err(|e| format!("CPU scratch allocation: {e}"))?;
    scratch.resize(scratch_bytes, 0);
    Ok(Kernel {
        _memory: module,
        entry,
        buffers,
        scalars,
        scratch,
        ir,
    })
}
