//! Constructive CPU physical planning and native Cranelift compilation.

use cranelift_codegen::ir::{self, types};
use cranelift_codegen::isa::CallConv;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use seismic_lang::abi::ScalarParameter;

mod buffer;
pub mod codegen;
pub mod mapping;
mod native;
pub mod physical;
mod workers;
pub use buffer::Buffer;
pub use seismic_realization::BufferSpec;
pub use workers::Workers;

struct Phase {
    entry: workers::PhaseEntry,
    work_items: u64,
    scratch_bytes: usize,
    bindings: Vec<usize>,
}

/// Executable memory and retained phase storage have one owner. Generated pointers cannot
/// escape this wrapper; mutable invocation serializes reuse of the retained storage.
pub struct Kernel {
    conditions: seismic_realization::InvocationConditions,
    _memory: ExecutableMemory,
    phases: Vec<Phase>,
    /// The entry ABI: the source-visible prefix of every phase's buffer table.
    buffers: Vec<BufferSpec>,
    external_ids: Vec<seismic_realization::executable::ResolvedStorageId>,
    scalars: Vec<ScalarParameter>,
    /// Invocation-owned storage of values that cross a phase boundary, after the entry ABI.
    retained: Vec<Vec<u64>>,
}

/// CPU-native output of mapped terminal instruction selection. Unlike the
/// legacy realization artifact this contains Cranelift functions and runtime
/// metadata only; no semantic or execution IR survives this boundary.
pub struct NativeExecution {
    pub name: String,
    pub conditions: seismic_realization::InvocationConditions,
    pub call_conv: CallConv,
    pub phases: Vec<NativePhase>,
    pub buffers: Vec<BufferSpec>,
    pub scalars: Vec<ScalarParameter>,
    pub external_ids: Vec<seismic_realization::executable::ResolvedStorageId>,
    pub scalar_ids: Vec<seismic_realization::executable::ResolvedStorageId>,
    pub retained: Vec<(seismic_realization::executable::ResolvedStorageId, u64, u64)>,
}

pub struct NativePhase {
    pub function: ir::Function,
    pub imports: Vec<(ir::FuncRef, seismic_realization::MathFunction)>,
    pub work_items: u64,
    pub scratch_bytes: usize,
    pub bindings: Vec<seismic_realization::executable::ResolvedStorageId>,
}

impl Kernel {
    pub fn buffers(&self) -> &[BufferSpec] {
        &self.buffers
    }
    pub fn scalars(&self) -> &[ScalarParameter] {
        &self.scalars
    }
    pub fn external_ids(&self) -> &[seismic_realization::executable::ResolvedStorageId] {
        &self.external_ids
    }
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }
    /// Execute over retained storage views. Each backing allocation is borrowed once; aliases
    /// use raw pointers rather than overlapping Rust mutable slices. The kernel's checked
    /// effects still govern whether any aliasing is legal.
    pub fn run(
        &mut self,
        workers: &mut Workers,
        buffers: &[&Buffer],
        scalars: &[f64],
    ) -> Result<(), String> {
        if buffers.len() != self.buffers.len() {
            return Err("CPU binding count differs from the compiled entry ABI".into());
        }
        let mut words = seismic_realization::encode_scalars(&self.scalars, scalars)?;
        let mut roots = Vec::new();
        let mut root_indices = Vec::new();
        for (buffer, spec) in buffers.iter().zip(&self.buffers) {
            if buffer.len() < spec.bytes {
                return Err(format!(
                    "CPU buffer {}.{} needs {} bytes, has {}",
                    spec.parameter,
                    spec.plane,
                    spec.bytes,
                    buffer.len()
                ));
            }
            if buffer.offset % spec.alignment != 0 {
                return Err(format!(
                    "CPU buffer {}.{} violates its typed storage alignment",
                    spec.parameter, spec.plane
                ));
            }
            let index = match roots
                .iter()
                .position(|root| std::rc::Rc::ptr_eq(root, &buffer.storage))
            {
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
            .map(|root| {
                root.try_borrow_mut()
                    .map_err(|_| "CPU storage is already in use".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut pointers = buffers
            .iter()
            .zip(root_indices.iter().copied())
            .map(|(buffer, index)| {
                // Views were checked at construction; the retained mutable borrow prevents
                // host access or reallocation through physical completion.
                unsafe { borrows[index].as_mut_ptr().cast::<u8>().add(buffer.offset) }
            })
            .collect::<Vec<_>>();
        self.conditions.validate_aliases(&self.buffers, |i| {
            (root_indices[i] as u64, buffers[i].offset as u64)
        })?;
        pointers.extend(words.iter_mut().map(|word| (word as *mut u64).cast::<u8>()));
        pointers.extend(
            self.retained
                .iter_mut()
                .map(|storage| storage.as_mut_ptr().cast::<u8>()),
        );
        for (ordinal, phase) in self.phases.iter().enumerate() {
            let phase_pointers = phase
                .bindings
                .iter()
                .map(|index| pointers[*index])
                .collect::<Vec<_>>();
            // Pointer capacities and scalar encodings match the checked compiled ABI; phases
            // run in source order, each to completion.
            let status = workers.run(
                phase.entry,
                &phase_pointers,
                &words,
                phase.work_items,
                phase.scratch_bytes,
            )?;
            if status != 0 {
                return Err(format!(
                    "CPU execution phase {ordinal} rejected an out-of-bounds view or index; outputs may be partially written"
                ));
            }
        }
        Ok(())
    }
}

struct ExecutableMemory(Option<JITModule>);
impl ExecutableMemory {
    fn module(&mut self) -> Result<&mut JITModule, String> {
        self.0
            .as_mut()
            .ok_or_else(|| "CPU executable memory was already released".to_string())
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

/// Finalize already-selected native Cranelift functions. This stage performs
/// no lowering, placement, scheduling, or implementation choice.
pub fn compile_native(execution: NativeExecution) -> Result<Kernel, String> {
    let NativeExecution {
        name,
        conditions,
        call_conv,
        phases: native_phases,
        buffers,
        scalars,
        external_ids,
        scalar_ids,
        retained,
    } = execution;
    if native_phases.is_empty() {
        return Err(format!("{name} has no native CPU phase"));
    }
    let isa = codegen::Policy::host()?.target()?.isa;
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
    let mut declared = Vec::new();
    for (ordinal, phase) in native_phases.into_iter().enumerate() {
        let signature = &phase.function.signature;
        if signature.call_conv != call_conv
            || signature.params.len() != 4
            || signature
                .params
                .iter()
                .any(|parameter| parameter.value_type != types::I64)
            || signature.returns.len() != 1
            || signature.returns[0].value_type != types::I32
        {
            return Err(format!(
                "{name} native phase {ordinal} has an incompatible invocation ABI"
            ));
        }
        let mut context = module.make_context();
        context.func = phase.function;
        for (reference, operation) in phase.imports {
            let signature = context.func.dfg.signatures
                [context.func.dfg.ext_funcs[reference].signature]
                .clone();
            let imported = module
                .declare_function(operation.symbol(), Linkage::Import, &signature)
                .map_err(|error| error.to_string())?;
            let native = module.declare_func_in_func(imported, &mut context.func);
            context.func.dfg.ext_funcs[reference] = context.func.dfg.ext_funcs[native].clone();
        }
        let id = module
            .declare_function(
                &format!("seismic_native_phase_{ordinal}"),
                Linkage::Local,
                &context.func.signature,
            )
            .map_err(|error| error.to_string())?;
        context.func.name = ir::UserFuncName::user(0, id.as_u32());
        module.define_function(id, &mut context).map_err(|error| {
            format!("CPU compilation of {name} native phase {ordinal}: {error:?}")
        })?;
        module.clear_context(&mut context);
        declared.push((id, phase.work_items, phase.scratch_bytes, phase.bindings));
    }
    module
        .finalize_definitions()
        .map_err(|error| error.to_string())?;
    let phases = declared
        .into_iter()
        .map(|(id, work_items, scratch_bytes, bindings)| {
            let bindings = bindings
                .into_iter()
                .map(|id| {
                    external_ids
                        .iter()
                        .position(|candidate| *candidate == id)
                        .or_else(|| {
                            scalar_ids
                                .iter()
                                .position(|candidate| *candidate == id)
                                .map(|index| external_ids.len() + index)
                        })
                        .or_else(|| {
                            retained
                                .iter()
                                .position(|(candidate, _, _)| *candidate == id)
                                .map(|index| external_ids.len() + scalar_ids.len() + index)
                        })
                        .ok_or_else(|| format!("CPU phase references absent storage#{}", id.0))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Phase {
                entry: unsafe {
                    std::mem::transmute::<*const u8, workers::PhaseEntry>(
                        module.get_finalized_function(id),
                    )
                },
                work_items,
                scratch_bytes,
                bindings,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let retained = retained
        .into_iter()
        .map(|(_, bytes, _)| {
            usize::try_from(bytes.div_ceil(8))
                .map(|words| vec![0u64; words])
                .map_err(|_| "CPU retained storage exceeds usize".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Kernel {
        conditions,
        _memory: memory,
        phases,
        buffers,
        external_ids,
        scalars,
        retained,
    })
}
