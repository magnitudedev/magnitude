//! Lower checked Seismic into a concrete shared scalar realization. Native
//! emitters own machine instructions and external math implementations.
mod scalar;
pub mod selection;
mod participants;
use cranelift_codegen::ir::{self, types, AbiParam, InstBuilder};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_realization::{CallConv, Dispatch, LoadStrategy, ScalarOptions, ScalarProgram};

/// The load rule every backend applies in `realize`: borrow where the borrow proof holds, else
/// materialize. The proof reads the modes already written (re-executing a load in a loop
/// ends its own loan only when that load borrows), so the rule is the greatest fixpoint of
/// the proof: every load starts as `Borrow`, a load whose proof fails becomes `Materialize`,
/// until stable. Failures only remove aliases, so the iteration is monotone. This selects
/// nothing by profit; it is one deterministic function of the instantiated execution.
pub fn load_rule(lowered: &mut LoweredIr) -> Result<(), String> {
    use seismic_lang::exec::ir::LoadMode;
    use seismic_lang::exec::normalize::loads;
    let mut modes = vec![LoadMode::Borrow; loads::sites(&lowered.body).len()];
    loop {
        // Write the candidate modes without the legality check, then read the proof under them.
        let permissive: Vec<LoadMode> = modes.clone();
        write_load_modes(&mut lowered.body, &mut permissive.iter())?;
        let failed: Vec<usize> = loads::sites(&lowered.body).iter().enumerate()
            .filter(|(site, proof)| modes[*site] == LoadMode::Borrow && !proof.can_borrow)
            .map(|(site, _)| site)
            .collect();
        if failed.is_empty() {
            return loads::resolve(&mut lowered.body, &modes).map(|_| ());
        }
        failed.into_iter().for_each(|site| modes[site] = LoadMode::Materialize);
    }
}

/// The load rule over the scalar realization's value bindings: nested loads and reductions
/// are bound to statements first, then `load_rule` applies.
pub fn scalar_load_rule(lowered: &mut LoweredIr) -> Result<(), String> {
    seismic_lang::exec::normalize::bind_values(&mut lowered.body, &mut lowered.vars);
    load_rule(lowered)
}

/// Write one mode per load site, in the lexical order of `loads::sites`.
fn write_load_modes(body: &mut [seismic_lang::exec::ir::Stmt], modes: &mut std::slice::Iter<'_, seismic_lang::exec::ir::LoadMode>) -> Result<(), String> {
    use seismic_lang::exec::ir::{Builtin, ExprKind, StmtKind};
    use seismic_lang::syntax::ast::AssignOp;
    let next = |modes: &mut std::slice::Iter<'_, _>| modes.next().copied().ok_or_else(|| "load site count changed while applying the load rule".to_string());
    for statement in body {
        match &mut statement.kind {
            StmtKind::Assign { target, op: AssignOp::Assign, value } if matches!(target.kind, ExprKind::Var(_)) => {
                let view = match &value.kind {
                    ExprKind::Builtin { name: Builtin::Load, args } => args.first().cloned(),
                    ExprKind::Load { view, .. } => Some((**view).clone()),
                    _ => None,
                };
                if let Some(view) = view {
                    value.kind = ExprKind::Load { view: Box::new(view), mode: next(modes)? };
                }
            }
            StmtKind::LoadLoop { vars, modes: selected, body, .. } => {
                *selected = Some((0..vars.len()).map(|_| next(modes)).collect::<Result<_, _>>()?);
                write_load_modes(body, modes)?;
            }
            StmtKind::Parallel { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => write_load_modes(body, modes)?,
            StmtKind::If { then, els, .. } => {
                write_load_modes(then, modes)?;
                write_load_modes(els, modes)?;
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn scalar(lowered: &LoweredIr, call_conv: CallConv) -> Result<ScalarProgram, String> {
    scalar_with(lowered, call_conv, Dispatch::Sequential)
}
pub fn scalar_with(
    lowered: &LoweredIr,
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
    lowered: &LoweredIr,
    call_conv: CallConv,
    options: ScalarOptions,
) -> Result<ScalarProgram, String> {
    let mut normalized = lowered.clone();
    seismic_lang::exec::normalize::bind_values(&mut normalized.body, &mut normalized.vars);
    seismic_lang::exec::normalize::select_loads(&mut normalized.body, options.loads == LoadStrategy::BorrowProvenReadOnly);
    scalar_resolved(&normalized, call_conv, options.dispatch)
}

/// Compile the selected load operations directly. This boundary validates their
/// legality and introduces no load/storage preferences.
pub fn scalar_resolved(
    lowered: &LoweredIr,
    call_conv: CallConv,
    dispatch: Dispatch,
) -> Result<ScalarProgram, String> {
    scalar_participants_resolved(lowered, call_conv, dispatch, seismic_realization::dispatch::Participation::Thread)
}
/// A source intrinsic commits to subgroup participation; this expands through
/// the same scalar instruction builder, preserving semantic backend calls.
pub fn subgroup_required(lowered: &LoweredIr) -> bool { participants::required(lowered) }
pub fn scalar_participants_resolved(
    lowered: &LoweredIr, call_conv: CallConv, dispatch: Dispatch,
    participation: seismic_realization::dispatch::Participation,
) -> Result<ScalarProgram, String> {
    let mut normalized = lowered.clone();
    if let seismic_realization::dispatch::Participation::Subgroup { lanes } = participation {
        if lanes != 32 || dispatch != Dispatch::ParallelRoot { return Err("subgroup scalar form requires 32 lanes per parallel work item".into()); }
        participants::validate(&normalized)?;
    }
    if dispatch == Dispatch::ParallelRoot {
        seismic_lang::exec::normalize::work_domain(&mut normalized.body);
    }
    seismic_lang::exec::normalize::bind_values(&mut normalized.body, &mut normalized.vars);
    let loads = seismic_lang::exec::normalize::loads::selected(&normalized.body)?;
    seismic_lang::exec::verify::lowered(&normalized)?;
    seismic_lang::exec::normalize::identify(&mut normalized.body, &mut 0);
    let lowered = &normalized;
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
    let mut emitter = scalar::Emitter::new(lowered, builder, args[0], args[1], args[2], participation)?;
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
    let (buffers, scalars, scratch_bytes, imports, backend_calls, execution) = (
        emitter.buffers,
        emitter.scalars,
        emitter.scratch_bytes,
        emitter.imports,
        emitter.backend_calls,
        emitter.execution,
    );
    emitter.builder.finalize();
    cranelift_codegen::verify_function(
        &function,
        &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    )
    .map_err(|e| format!("invalid scalar realization: {e}"))?;
    Ok(ScalarProgram {
        conditions: seismic_realization::InvocationConditions::from_lowered(lowered)?,
        function,
        public_buffer_count: buffers.len(),
        buffers,
        scalars,
        scratch_bytes: scratch_bytes
            .checked_add(7)
            .map(|n| n & !7)
            .ok_or("scratch alignment overflow")?,
        imports,
        backend_calls,
        participation,
        work_items,
        dispatch,
        loads,
        execution,
    })
}

/// Lower serial setup and parallel domains through the shared phase plan. Values
/// crossing a launch boundary use invocation-owned storage in the native ABI.
pub fn scalar_sequence(
    lowered: &LoweredIr,
    call_conv: CallConv,
    options: ScalarOptions,
) -> Result<seismic_realization::ScalarSequence, String> {
    let mut normalized = lowered.clone();
    seismic_lang::exec::normalize::bind_values(&mut normalized.body, &mut normalized.vars);
    seismic_lang::exec::normalize::select_loads(&mut normalized.body, options.loads == LoadStrategy::BorrowProvenReadOnly);
    scalar_sequence_resolved(&normalized, call_conv, options.dispatch)
}

pub fn scalar_sequence_resolved(
    lowered: &LoweredIr,
    call_conv: CallConv,
    dispatch: Dispatch,
) -> Result<seismic_realization::ScalarSequence, String> {
    scalar_sequence_participants_resolved(lowered, call_conv, dispatch, seismic_realization::dispatch::Participation::Thread)
}
pub fn scalar_sequence_participants_resolved(
    lowered: &LoweredIr, call_conv: CallConv, dispatch: Dispatch,
    participation: seismic_realization::dispatch::Participation,
) -> Result<seismic_realization::ScalarSequence, String> {
    use seismic_realization::{ScalarPhase, ScalarSequence};
    let public_buffer_count = seismic_realization::storage::parameters(lowered)?.0.len();
    let source_conditions = seismic_realization::InvocationConditions::from_lowered(lowered)?;
    if dispatch == Dispatch::Sequential {
        return Ok(ScalarSequence {
            name: lowered.name.clone(),
            public_buffer_count,
            retained: Vec::new(),
            phases: vec![ScalarPhase {
                source_statement: 0,
                program: scalar_participants_resolved(lowered, call_conv, dispatch, participation)?,
            }],
        });
    }
    let plan = seismic_realization::phases::construct(lowered)?;
    let lowered = &plan.function;
    let mut phases = Vec::new();
    for (source_statement, statement) in lowered.body.iter().enumerate() {
        let mut phase = lowered.clone();
        phase.body = vec![statement.clone()];
        let mut program = scalar_participants_resolved(&phase, call_conv, dispatch, participation)
            .map_err(|error| format!("{} phase {source_statement}: {error}", lowered.name))?;
        program.public_buffer_count = public_buffer_count;
        program.conditions = source_conditions.clone();
        phases.push(ScalarPhase {
            source_statement,
            program,
        });
    }
    Ok(ScalarSequence {
        name: lowered.name.clone(),
        phases,
        public_buffer_count,
        retained: plan.retained,
    })
}
