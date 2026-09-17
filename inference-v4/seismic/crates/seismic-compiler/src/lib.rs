//! Lower checked Seismic into a concrete shared scalar realization. Native
//! emitters own machine instructions and external math implementations.
mod scalar;
use cranelift_codegen::ir::{self, types, AbiParam, InstBuilder};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use seismic_lang::lower::Lowered;
use seismic_realization::{CallConv, Dispatch, LoadStrategy, ScalarOptions, ScalarProgram};

pub fn scalar(lowered: &Lowered, call_conv: CallConv) -> Result<ScalarProgram, String> {
    scalar_with(lowered, call_conv, Dispatch::Sequential)
}
pub fn scalar_with(
    lowered: &Lowered,
    call_conv: CallConv,
    dispatch: Dispatch,
) -> Result<ScalarProgram, String> {
    scalar_candidate(
        lowered,
        call_conv,
        ScalarOptions {
            dispatch,
            loads: LoadStrategy::Materialize,
        },
    )
}
pub fn scalar_candidate(
    lowered: &Lowered,
    call_conv: CallConv,
    options: ScalarOptions,
) -> Result<ScalarProgram, String> {
    let ScalarOptions { dispatch, loads } = options;
    let mut function = ir::Function::new();
    function.signature.call_conv = call_conv;
    for _ in 0..3 {
        function.signature.params.push(AbiParam::new(types::I64));
    }
    if dispatch == Dispatch::ParallelRoot {
        function.signature.params.push(AbiParam::new(types::I64));
    }
    function.signature.returns.push(AbiParam::new(types::I32));
    let mut frontend = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut function, &mut frontend);
    let entry = builder.create_block();
    builder.switch_to_block(entry);
    builder.append_block_params_for_function_params(entry);
    let args = builder.block_params(entry).to_vec();
    let mut emitter = scalar::Emitter::new(lowered, builder, args[0], args[1], args[2], loads)?;
    let work_items = match dispatch {
        Dispatch::Sequential => {
            emitter.body(&lowered.body)?;
            1
        }
        Dispatch::ParallelRoot => emitter.parallel_root(&lowered.body, args[3])?,
    };
    let success = emitter.builder.ins().iconst(types::I32, 0);
    emitter.builder.ins().return_(&[success]);
    emitter.builder.seal_all_blocks();
    let (buffers, scalars, scratch_bytes, imports, execution) = (
        emitter.buffers,
        emitter.scalars,
        emitter.scratch_bytes,
        emitter.imports,
        emitter.execution,
    );
    emitter.builder.finalize();
    cranelift_codegen::verify_function(
        &function,
        &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    )
    .map_err(|e| format!("invalid scalar realization: {e}"))?;
    Ok(ScalarProgram {
        function,
        buffers,
        scalars,
        scratch_bytes: scratch_bytes
            .checked_add(7)
            .map(|n| n & !7)
            .ok_or("scratch alignment overflow")?,
        imports,
        work_items,
        dispatch,
        loads,
        execution,
    })
}

/// Lower a natural sequence of outer parallel domains to ordered native phases.
/// Local storage crossing domains requires a different realization and is rejected
/// by each phase's ordinary binding validation; no kernel-source rewrite is needed.
pub fn scalar_sequence(
    lowered: &Lowered,
    call_conv: CallConv,
    options: ScalarOptions,
) -> Result<seismic_realization::ScalarSequence, String> {
    use seismic_realization::{ScalarPhase, ScalarSequence};
    if options.dispatch == Dispatch::Sequential {
        return Ok(ScalarSequence {
            name: lowered.name.clone(),
            phases: vec![ScalarPhase {
                source_statement: 0,
                program: scalar_candidate(lowered, call_conv, options)?,
            }],
        });
    }
    if lowered.body.is_empty() {
        return Err("parallel sequence has no domains".into());
    }
    let mut phases = Vec::new();
    for (source_statement, statement) in lowered.body.iter().enumerate() {
        if !matches!(statement.kind, seismic_lang::hir::StmtKind::Parallel { .. }) {
            return Err(format!("{}: statement {source_statement} needs cross-phase storage/control lowering; expected an outer parallel domain",lowered.name));
        }
        let mut phase = lowered.clone();
        phase.body = vec![statement.clone()];
        let program = scalar_candidate(&phase, call_conv, options)
            .map_err(|error| format!("{} phase {source_statement}: {error}", lowered.name))?;
        phases.push(ScalarPhase {
            source_statement,
            program,
        });
    }
    Ok(ScalarSequence {
        name: lowered.name.clone(),
        phases,
    })
}
