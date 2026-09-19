//! The CPU backend: the `selection::Backend` mapping (`mapping::Cpu`) and native compilation
//! of its realized execution through Cranelift. Coverage is scalar: every phase is the shared
//! scalar realization of the instantiated execution IR. No reference interpreter is called.

use cranelift_codegen::ir::{self, types};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use seismic_lang::abi::ScalarParameter;

mod buffer;
pub mod codegen;
pub mod mapping;
mod workers;
pub use buffer::Buffer;
pub use seismic_realization::BufferSpec;
pub use workers::Workers;

struct Phase {
    entry: workers::PhaseEntry,
    work_items: u64,
    scratch_bytes: usize,
}

/// Executable memory and retained phase storage have one owner. Generated pointers cannot
/// escape this wrapper; mutable invocation serializes reuse of the retained storage.
pub struct Kernel {
    conditions: seismic_realization::InvocationConditions,
    _memory: ExecutableMemory,
    phases: Vec<Phase>,
    /// The entry ABI: the source-visible prefix of every phase's buffer table.
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    /// Invocation-owned storage of values that cross a phase boundary, after the entry ABI.
    retained: Vec<Vec<u64>>,
}

impl Kernel {
    pub fn buffers(&self) -> &[BufferSpec] {
        &self.buffers
    }
    pub fn scalars(&self) -> &[ScalarParameter] {
        &self.scalars
    }
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }
    /// Execute over retained storage views. Each backing allocation is borrowed once; aliases
    /// use raw pointers rather than overlapping Rust mutable slices. The kernel's checked
    /// effects still govern whether any aliasing is legal.
    pub fn run(&mut self, workers: &mut Workers, buffers: &[&Buffer], scalars: &[f64]) -> Result<(), String> {
        if buffers.len() != self.buffers.len() {
            return Err("CPU binding count differs from the compiled entry ABI".into());
        }
        let words = seismic_realization::encode_scalars(&self.scalars, scalars)?;
        let mut roots = Vec::new();
        let mut root_indices = Vec::new();
        for (buffer, spec) in buffers.iter().zip(&self.buffers) {
            if buffer.len() < spec.bytes {
                return Err(format!("CPU buffer {}.{} needs {} bytes, has {}", spec.parameter, spec.plane, spec.bytes, buffer.len()));
            }
            if buffer.offset % spec.alignment != 0 {
                return Err(format!("CPU buffer {}.{} violates its typed storage alignment", spec.parameter, spec.plane));
            }
            let index = match roots.iter().position(|root| std::rc::Rc::ptr_eq(root, &buffer.storage)) {
                Some(index) => index,
                None => {
                    roots.push(buffer.storage.clone());
                    roots.len() - 1
                }
            };
            root_indices.push(index);
        }
        let mut borrows = roots
            .iter()
            .map(|root| root.try_borrow_mut().map_err(|_| "CPU storage is already in use".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut pointers = buffers
            .iter()
            .zip(root_indices)
            .map(|(buffer, index)| {
                // Views were checked at construction; the retained mutable borrow prevents
                // host access or reallocation through physical completion.
                unsafe { borrows[index].as_mut_ptr().cast::<u8>().add(buffer.offset) }
            })
            .collect::<Vec<_>>();
        self.conditions.validate_aliases(&self.buffers, |i| (0, pointers[i] as u64))?;
        pointers.extend(self.retained.iter_mut().map(|storage| storage.as_mut_ptr().cast::<u8>()));
        for (ordinal, phase) in self.phases.iter().enumerate() {
            // Pointer capacities and scalar encodings match the checked compiled ABI; phases
            // run in source order, each to completion.
            let status = workers.run(phase.entry, &pointers, &words, phase.work_items, phase.scratch_bytes)?;
            if status != 0 {
                return Err(format!("CPU execution phase {ordinal} rejected an out-of-bounds view or index; outputs may be partially written"));
            }
        }
        Ok(())
    }
}

struct ExecutableMemory(Option<JITModule>);
impl ExecutableMemory {
    fn module(&mut self) -> Result<&mut JITModule, String> {
        self.0.as_mut().ok_or_else(|| "CPU executable memory was already released".to_string())
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

// The reference evaluates transcendental functions in binary64 and rounds once.
extern "C" fn exp(x: f32) -> f32 {
    f64::from(x).exp() as f32
}
extern "C" fn log(x: f32) -> f32 {
    f64::from(x).ln() as f32
}
extern "C" fn sin(x: f32) -> f32 {
    f64::from(x).sin() as f32
}
extern "C" fn cos(x: f32) -> f32 {
    f64::from(x).cos() as f32
}

/// Natively compile exactly the realized execution. Nothing here selects or re-lowers.
pub fn compile(execution: mapping::Execution) -> Result<Kernel, String> {
    let mapping::Execution { sequence, conditions, codegen } = execution;
    let call_conv = codegen.call_conv();
    let isa = codegen.target()?.isa;
    let mut jit = JITBuilder::with_isa(isa, default_libcall_names());
    for (name, address) in [
        ("seismic_exp", exp as *const u8),
        ("seismic_log", log as *const u8),
        ("seismic_sin", sin as *const u8),
        ("seismic_cos", cos as *const u8),
    ] {
        jit.symbol(name, address);
    }
    let mut memory = ExecutableMemory(Some(JITModule::new(jit)));
    let module = memory.module()?;
    if module.target_config().pointer_type() != types::I64 {
        return Err("the CPU backend requires 64-bit pointers".into());
    }
    let public = sequence.public_buffer_count;
    let mut abi: Option<(Vec<BufferSpec>, Vec<ScalarParameter>)> = None;
    let mut declared = Vec::new();
    for (ordinal, phase) in sequence.phases.into_iter().enumerate() {
        let program = phase.program;
        let signature = &program.function.signature;
        if program.dispatch != seismic_realization::Dispatch::ParallelRoot
            || signature.call_conv != call_conv
            || signature.params.len() != 4
            || signature.params.iter().any(|p| p.value_type != types::I64)
            || signature.returns.len() != 1
            || signature.returns[0].value_type != types::I32
            || program.public_buffer_count != public
        {
            return Err(format!("{} phase {ordinal} has an incompatible invocation ABI", sequence.name));
        }
        codegen::imports(&program)?;
        let seismic_realization::ScalarProgram { function, buffers, scalars, scratch_bytes, imports, work_items, .. } = program;
        match &abi {
            None => abi = Some((buffers, scalars)),
            Some((first_buffers, first_scalars)) if *first_buffers == buffers && *first_scalars == scalars => {}
            Some(_) => return Err(format!("{} phase {ordinal} disagrees with the invocation ABI of phase 0", sequence.name)),
        }
        let mut context = module.make_context();
        context.func = function;
        for (reference, operation) in imports {
            let signature = context.func.dfg.signatures[context.func.dfg.ext_funcs[reference].signature].clone();
            let id = module.declare_function(operation.symbol(), Linkage::Import, &signature).map_err(|e| e.to_string())?;
            let native = module.declare_func_in_func(id, &mut context.func);
            context.func.dfg.ext_funcs[reference] = context.func.dfg.ext_funcs[native].clone();
        }
        let id = module
            .declare_function(&format!("seismic_phase_{ordinal}"), Linkage::Local, &context.func.signature)
            .map_err(|e| e.to_string())?;
        context.func.name = ir::UserFuncName::user(0, id.as_u32());
        module.define_function(id, &mut context).map_err(|e| format!("CPU compilation of {} phase {ordinal}: {e:?}", sequence.name))?;
        module.clear_context(&mut context);
        declared.push((id, work_items, scratch_bytes));
    }
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let phases = declared
        .into_iter()
        .map(|(id, work_items, scratch_bytes)| Phase {
            // The finalized function has exactly the checked four-argument signature above.
            entry: unsafe { std::mem::transmute::<*const u8, workers::PhaseEntry>(module.get_finalized_function(id)) },
            work_items,
            scratch_bytes,
        })
        .collect();
    let (mut buffers, scalars) = abi.ok_or_else(|| format!("{} has no phase", sequence.name))?;
    let mut retained = Vec::new();
    for spec in buffers.split_off(public) {
        let mut storage: Vec<u64> = Vec::new();
        let words = spec.bytes.div_ceil(8);
        storage.try_reserve_exact(words).map_err(|e| format!("CPU retained storage {}: {e}", spec.parameter))?;
        storage.resize(words, 0);
        retained.push(storage);
    }
    Ok(Kernel { conditions, _memory: memory, phases, buffers, scalars, retained })
}
