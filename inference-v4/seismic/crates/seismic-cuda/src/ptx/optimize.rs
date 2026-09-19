//! Target-neutral cleanup of the shared scalar SSA before PTX selection. The driver assembles
//! PTX as written: it does not fold constants, remove checks whose operands are constants,
//! hoist invariant address arithmetic or forward stored values, and every SSA value is one
//! PTX register. The scalar realization states every coordinate, bound and stride as its own
//! instruction, so the unoptimized function is dominated by arithmetic on constants.
//!
//! One fixed pipeline, no target facts and no alternatives: immediate instruction forms are
//! expanded to a constant and the plain instruction (the e-graph rules match plain forms only;
//! Cranelift's own expansion sits behind an ISA), then Cranelift's ISA-independent
//! mid-end (unreachable-code and constant-phi removal, then the e-graph pass: constant
//! folding, algebraic identities, GVN, loop-invariant code motion, redundant-load
//! elimination), folding of branches on constants (which the e-graph pass leaves in place),
//! and the same mid-end once more over the reduced control flow. Floating-point results are
//! unchanged: the e-graph rules neither reassociate nor contract floating-point arithmetic.
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{self, InstBuilder, InstructionData, Opcode};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;

fn flags() -> Result<settings::Flags, String> {
    let mut builder = settings::builder();
    builder.set("opt_level", "speed").map_err(|e| format!("scalar SSA optimization flags: {e}"))?;
    Ok(settings::Flags::new(builder))
}

fn mid_end(context: &mut Context, flags: &settings::Flags) -> Result<(), String> {
    let failed = |e| format!("scalar SSA optimization: {e:?}");
    context.compute_cfg();
    context.compute_domtree();
    context.eliminate_unreachable_code(flags).map_err(failed)?;
    context.remove_constant_phis(flags).map_err(failed)?;
    context.func.dfg.resolve_all_aliases();
    context.egraph_pass(flags, &mut ControlPlane::default()).map_err(failed)
}

/// `op_imm x, c` becomes `op x, (iconst c)`, the constant zero-extended from the operand type
/// as Cranelift's legalizer states it.
fn expand_immediate_forms(function: &mut ir::Function) -> Result<(), String> {
    let mut cursor = FuncCursor::new(function);
    while let Some(_block) = cursor.next_block() {
        while let Some(inst) = cursor.next_inst() {
            let (arg, imm, opcode, condition) = match cursor.func.dfg.insts[inst] {
                InstructionData::BinaryImm64 { opcode, arg, imm } => (arg, imm, opcode, None),
                InstructionData::IntCompareImm { opcode, arg, cond, imm } => (arg, imm, opcode, Some(cond)),
                _ => continue,
            };
            let ty = cursor.func.dfg.value_type(arg);
            let bits = match ty {
                ir::types::I8 => i64::from(imm.bits() as u8),
                ir::types::I16 => i64::from(imm.bits() as u16),
                ir::types::I32 => i64::from(imm.bits() as u32),
                ir::types::I64 => imm.bits(),
                other => return Err(format!("immediate form over {other}")),
            };
            cursor.goto_inst(inst);
            let constant = cursor.ins().iconst(ty, bits);
            let replace = cursor.func.dfg.replace(inst);
            match (opcode, condition) {
                (_, Some(condition)) => replace.icmp(condition, arg, constant),
                (Opcode::IaddImm, _) => replace.iadd(arg, constant),
                (Opcode::IrsubImm, _) => replace.isub(constant, arg),
                (Opcode::ImulImm, _) => replace.imul(arg, constant),
                (Opcode::UdivImm, _) => replace.udiv(arg, constant),
                (Opcode::SdivImm, _) => replace.sdiv(arg, constant),
                (Opcode::UremImm, _) => replace.urem(arg, constant),
                (Opcode::SremImm, _) => replace.srem(arg, constant),
                (Opcode::BandImm, _) => replace.band(arg, constant),
                (Opcode::BorImm, _) => replace.bor(arg, constant),
                (Opcode::BxorImm, _) => replace.bxor(arg, constant),
                (Opcode::IshlImm, _) => replace.ishl(arg, constant),
                (Opcode::UshrImm, _) => replace.ushr(arg, constant),
                (Opcode::SshrImm, _) => replace.sshr(arg, constant),
                (other, _) => return Err(format!("immediate form {other}")),
            };
        }
    }
    Ok(())
}

/// Replace every `brif` whose condition is an integer constant by the jump it always takes.
fn fold_constant_branches(function: &mut ir::Function) -> bool {
    let mut folded = false;
    let blocks: Vec<ir::Block> = function.layout.blocks().collect();
    for block in blocks {
        let Some(inst) = function.layout.last_inst(block) else { continue };
        let InstructionData::Brif { arg, blocks, .. } = function.dfg.insts[inst] else { continue };
        let condition = function.dfg.resolve_aliases(arg);
        let ir::ValueDef::Result(definition, _) = function.dfg.value_def(condition) else { continue };
        let InstructionData::UnaryImm { opcode: Opcode::Iconst, imm } = function.dfg.insts[definition] else { continue };
        let taken = blocks[usize::from(imm.bits() == 0)];
        let target = taken.block(&function.dfg.value_lists);
        let args: Vec<ir::BlockArg> = taken.args(&function.dfg.value_lists).collect();
        function.dfg.replace(inst).jump(target, &args);
        folded = true;
    }
    folded
}

pub(super) fn optimize(function: &ir::Function) -> Result<ir::Function, String> {
    let flags = flags()?;
    let mut context = Context::for_function(function.clone());
    expand_immediate_forms(&mut context.func)?;
    mid_end(&mut context, &flags)?;
    if fold_constant_branches(&mut context.func) {
        mid_end(&mut context, &flags)?;
    }
    if std::env::var_os("SEISMIC_CUDA_DUMP_SSA").is_some() {
        eprintln!("{}", context.func.display());
    }
    Ok(context.func)
}
