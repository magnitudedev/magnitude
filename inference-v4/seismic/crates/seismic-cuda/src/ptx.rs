//! Translate the shared scalar SSA realization directly to PTX. The baseline uses
//! one CUDA thread per explicit outer parallel point and disjoint global scratch.
//! This is an inspectable legal realization, not an automatic performance choice.
use cranelift_codegen::ir::{
    self,
    condcodes::{FloatCC, IntCC},
    types, BlockArg, BlockCall, Inst, InstructionData as D, Opcode as O, Type, Value,
};
use seismic_realization::{Dispatch, MathFunction, ScalarProgram};
use std::fmt::Write;

/// Compile the explicit SIMT baseline without requiring a driver or device.
pub fn lower(lowered: &seismic_lang::lowered_ir::LoweredIr, dispatch: Dispatch) -> Result<String, String> {
    lower_candidate(
        lowered,
        seismic_realization::ScalarOptions {
            dispatch,
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    )
}
pub fn lower_candidate(
    lowered: &seismic_lang::lowered_ir::LoweredIr,
    options: seismic_realization::ScalarOptions,
) -> Result<String, String> {
    if lowered.backend != "cuda" {
        return Err("PTX emitter requires CUDA lowering".into());
    }
    let program = seismic_compiler::scalar_candidate(
        lowered,
        cranelift_codegen::isa::CallConv::SystemV,
        options,
    )?;
    emit(&program)
}

pub fn emit(program: &ScalarProgram) -> Result<String, String> {
    let f = &program.function;
    let mut e = Emitter {
        f,
        program,
        text: String::new(),
        temporary: 0,
    };
    // PTX 7.0 / sm_80 uses only the documented baseline instruction set. The
    // driver accepts compatible targets on newer devices; runtime checks SM >=80.
    e.text.push_str(".version 7.0\n.target sm_80\n.address_size 64\n\n.visible .entry seismic_kernel(\n .param .u64 buffers,\n .param .u64 scalars,\n .param .u64 scratch,\n .param .u64 statuses\n) {\n");
    if program
        .imports
        .iter()
        .any(|(_, m)| matches!(m, MathFunction::Exp | MathFunction::ExpFast))
    {
        let entry = e.text.find(".visible .entry").ok_or("missing PTX entry")?;
        e.text.insert_str(entry, include_str!("math/exp.ptx"));
    }
    if program.imports.iter().any(|(_, math)| {
        matches!(
            math,
            MathFunction::Log | MathFunction::Sin | MathFunction::Cos
        )
    }) {
        let entry = e.text.find(".visible .entry").ok_or("missing PTX entry")?;
        e.text
            .insert_str(entry, include_str!("math/portable_math.ptx"));
    }
    for v in f.dfg.values() {
        if !f.dfg.value_is_real(v) {
            continue;
        }
        let t = reg_type(f.dfg.value_type(v))?;
        writeln!(e.text, " .reg .{t} %v{};", v.as_u32()).unwrap();
    }
    e.text.push_str(" .reg .u32 %bx, %bd, %tx;\n .reg .u64 %linear, %scratch_offset, %status_addr, %status_offset;\n .reg .pred %pred;\n mov.u32 %bx, %ctaid.x;\n mov.u32 %bd, %ntid.x;\n mov.u32 %tx, %tid.x;\n mul.wide.u32 %linear, %bx, %bd;\n cvt.u64.u32 %status_offset, %tx;\n add.u64 %linear, %linear, %status_offset;\n");
    writeln!(
        e.text,
        " setp.ge.u64 %pred, %linear, {};\n @%pred ret;",
        program.work_items
    )
    .unwrap();
    e.text.push_str(" ld.param.u64 %status_addr, [statuses];\n mul.lo.u64 %status_offset, %linear, 4;\n add.u64 %status_addr, %status_addr, %status_offset;\n");
    let entry = f.layout.entry_block().ok_or("empty scalar program")?;
    let params = f.dfg.block_params(entry);
    for (v, name) in params.iter().take(3).zip(["buffers", "scalars", "scratch"]) {
        writeln!(e.text, " ld.param.u64 {}, [{name}];", e.v(*v)).unwrap();
    }
    if program.dispatch == Dispatch::ParallelRoot {
        writeln!(e.text, " mov.u64 {}, %linear;\n mul.lo.u64 %scratch_offset, %linear, {};\n add.u64 {scratch}, {scratch}, %scratch_offset;", e.v(params[3]),program.scratch_bytes,scratch=e.v(params[2])).unwrap();
    }
    for block in f.layout.blocks() {
        writeln!(e.text, "{block}:").unwrap();
        for inst in f.layout.block_insts(block) {
            e.instruction(inst)
                .map_err(|error| format!("PTX {}: {error}", f.dfg.display_inst(inst)))?;
        }
    }
    e.text.push_str("}\n");
    Ok(e.text)
}
fn reg_type(t: Type) -> Result<&'static str, String> {
    Ok(match t {
        types::I8 | types::I16 | types::I32 => "b32",
        types::I64 => "b64",
        types::F32 => "f32",
        _ => return Err(format!("unsupported PTX scalar type {t}")),
    })
}
fn uint(t: Type) -> &'static str {
    if t == types::I64 {
        "u64"
    } else {
        "u32"
    }
}
fn sint(t: Type) -> &'static str {
    if t == types::I64 {
        "s64"
    } else {
        "s32"
    }
}
fn bits(t: Type) -> &'static str {
    if t == types::I64 {
        "b64"
    } else {
        "b32"
    }
}
fn memory_type(t: Type) -> Result<&'static str, String> {
    Ok(match t {
        types::I8 => "u8",
        types::I16 => "u16",
        types::I32 => "u32",
        types::I64 => "u64",
        types::F32 => "f32",
        _ => return Err(format!("unsupported memory type {t}")),
    })
}
struct Emitter<'a> {
    f: &'a ir::Function,
    program: &'a ScalarProgram,
    text: String,
    temporary: usize,
}
impl Emitter<'_> {
    fn v(&self, v: Value) -> String {
        format!("%v{}", self.f.dfg.resolve_aliases(v).as_u32())
    }
    fn temp(&mut self, t: Type) -> Result<String, String> {
        let name = format!("%tmp{}", self.temporary);
        self.temporary += 1;
        writeln!(self.text, " .reg .{} {name};", reg_type(t)?).unwrap();
        Ok(name)
    }
    fn edge(&mut self, edge: BlockCall) -> Result<(), String> {
        let block = edge.block(&self.f.dfg.value_lists);
        let args = edge.args(&self.f.dfg.value_lists).collect::<Vec<_>>();
        let params = self.f.dfg.block_params(block).to_vec();
        // Block parameters are simultaneous assignments; temporaries preserve
        // swaps and cycles on loop backedges.
        let mut moves = Vec::new();
        for (p, a) in params.into_iter().zip(args) {
            let BlockArg::Value(a) = a else {
                return Err("exception block argument".into());
            };
            let t = self.f.dfg.value_type(p);
            let tmp = self.temp(t)?;
            let suffix = if t == types::F32 { "f32" } else { bits(t) };
            writeln!(self.text, " mov.{suffix} {tmp}, {};", self.v(a)).unwrap();
            moves.push((p, tmp, suffix));
        }
        for (p, tmp, suffix) in moves {
            writeln!(self.text, " mov.{suffix} {}, {tmp};", self.v(p)).unwrap();
        }
        writeln!(self.text, " bra {block};").unwrap();
        Ok(())
    }
    fn comparison(&mut self, out: &str, op: &str, ty: &str, a: &str, b: &str) {
        writeln!(
            self.text,
            " setp.{op}.{ty} %pred, {a}, {b};\n selp.u32 {out}, 1, 0, %pred;"
        )
        .unwrap();
    }
    fn instruction(&mut self, inst: Inst) -> Result<(), String> {
        let data = &self.f.dfg.insts[inst];
        let op = data.opcode();
        let result = self.f.dfg.inst_results(inst).first().copied();
        let out = result.map(|v| self.v(v)).unwrap_or_default();
        let t = result
            .map(|v| self.f.dfg.value_type(v))
            .unwrap_or(types::INVALID);
        match data {
            D::UnaryImm { imm, .. } if op == O::Iconst => {
                writeln!(self.text, " mov.{} {out}, {};", bits(t), imm.bits()).unwrap();
            }
            D::UnaryIeee32 { imm, .. } => {
                writeln!(self.text, " mov.f32 {out}, 0f{:08x};", imm.bits()).unwrap();
            }
            D::Load { arg, offset, .. } => {
                writeln!(
                    self.text,
                    " ld.global.{} {out}, [{}{:+}];",
                    memory_type(t)?,
                    self.v(*arg),
                    i32::from(*offset)
                )
                .unwrap();
            }
            D::Store { args, offset, .. } => {
                writeln!(
                    self.text,
                    " st.global.{} [{}{:+}], {};",
                    memory_type(self.f.dfg.value_type(args[0]))?,
                    self.v(args[1]),
                    i32::from(*offset),
                    self.v(args[0])
                )
                .unwrap();
            }
            D::Jump { destination, .. } => self.edge(*destination)?,
            D::Brif { arg, blocks, .. } => {
                let label = format!("edge_{}", self.temporary);
                self.temporary += 1;
                writeln!(
                    self.text,
                    " setp.ne.u32 %pred, {}, 0;\n @%pred bra {label};",
                    self.v(*arg)
                )
                .unwrap();
                self.edge(blocks[1])?;
                writeln!(self.text, "{label}:").unwrap();
                self.edge(blocks[0])?;
            }
            D::MultiAry { args, .. } if op == O::Return => {
                let value = args.as_slice(&self.f.dfg.value_lists)[0];
                writeln!(
                    self.text,
                    " st.global.u32 [%status_addr], {};\n ret;",
                    self.v(value)
                )
                .unwrap();
            }
            D::IntCompare { args, cond, .. } => {
                let (cc, signed) = int_cc(*cond);
                let at = self.f.dfg.value_type(args[0]);
                self.comparison(
                    &out,
                    cc,
                    if signed { sint(at) } else { uint(at) },
                    &self.v(args[0]),
                    &self.v(args[1]),
                );
            }
            D::IntCompareImm { arg, cond, imm, .. } => {
                let (cc, signed) = int_cc(*cond);
                let at = self.f.dfg.value_type(*arg);
                self.comparison(
                    &out,
                    cc,
                    if signed { sint(at) } else { uint(at) },
                    &self.v(*arg),
                    &imm.bits().to_string(),
                );
            }
            D::FloatCompare { args, cond, .. } => {
                self.comparison(
                    &out,
                    float_cc(*cond),
                    "f32",
                    &self.v(args[0]),
                    &self.v(args[1]),
                );
            }
            D::Ternary { args, .. } if op == O::Select => {
                writeln!(
                    self.text,
                    " setp.ne.u32 %pred, {}, 0;\n selp.{} {out}, {}, {}, %pred;",
                    self.v(args[0]),
                    if t == types::F32 { "f32" } else { bits(t) },
                    self.v(args[1]),
                    self.v(args[2])
                )
                .unwrap();
            }
            D::Ternary { args, .. } if op == O::Fma => {
                writeln!(
                    self.text,
                    " fma.rn.f32 {out}, {}, {}, {};",
                    self.v(args[0]),
                    self.v(args[1]),
                    self.v(args[2])
                )
                .unwrap();
            }
            D::Binary { args, .. } => {
                self.binary(op, &out, t, &self.v(args[0]), &self.v(args[1]))?
            }
            D::BinaryImm64 { arg, imm, .. } => {
                self.binary(op, &out, t, &self.v(*arg), &imm.bits().to_string())?
            }
            D::Unary { arg, .. } => self.unary(op, &out, t, *arg)?,
            D::LoadNoOffset { arg, .. } if op == O::Bitcast => {
                writeln!(self.text, " mov.{} {out}, {};", bits(t), self.v(*arg)).unwrap();
            }
            D::Call { func_ref, .. } => {
                let (_, math) = self
                    .program
                    .imports
                    .iter()
                    .find(|(r, _)| r == func_ref)
                    .ok_or("unrecognized external function")?;
                let arg = self.f.dfg.inst_args(inst)[0];
                // Fast exp currently shares the accurate implementation; semantic
                // identity stays distinct in the realization account.
                writeln!(self.text, " {{ .param .b32 arg; .param .b32 result;\n st.param.f32 [arg], {};\n call.uni (result), {}, (arg);\n ld.param.f32 {out}, [result];\n }}", self.v(arg), math.symbol()).unwrap();
            }
            _ => return Err(format!("unsupported instruction {op}")),
        }
        Ok(())
    }
    fn unary(&mut self, op: O, out: &str, t: Type, arg: Value) -> Result<(), String> {
        let a = self.v(arg);
        let at = self.f.dfg.value_type(arg);
        let instruction = match op {
            O::Fneg => "neg.f32".into(),
            O::Fabs => "abs.f32".into(),
            O::Sqrt => "sqrt.rn.f32".into(),
            O::Ineg => format!("neg.{}", sint(t)),
            O::Uextend => format!("cvt.{}.{}", uint(t), uint(at)),
            O::Sextend if at == types::I32 && t == types::I64 => "cvt.s64.s32".into(),
            O::Ireduce if t.bits() < 32 => {
                writeln!(
                    self.text,
                    " and.b32 {out}, {a}, {};",
                    (1u64 << t.bits()) - 1
                )
                .unwrap();
                return Ok(());
            }
            O::Ireduce => format!("cvt.{}.{}", uint(t), uint(at)),
            O::FcvtFromUint => format!("cvt.rn.f32.{}", uint(at)),
            O::FcvtFromSint => format!("cvt.rn.f32.{}", sint(at)),
            O::FcvtToSintSat => format!("cvt.rzi.{}.f32", sint(t)),
            O::FcvtToUintSat => format!("cvt.rzi.{}.f32", uint(t)),
            _ => return Err(format!("unsupported unary {op}")),
        };
        writeln!(self.text, " {instruction} {out}, {a};").unwrap();
        Ok(())
    }
    fn binary(&mut self, op: O, out: &str, t: Type, a: &str, b: &str) -> Result<(), String> {
        let mnemonic = match op {
            O::Iadd | O::IaddImm => format!("add.{}", uint(t)),
            O::Isub => format!("sub.{}", uint(t)),
            O::Imul | O::ImulImm => format!("mul.lo.{}", uint(t)),
            O::Udiv | O::UdivImm => format!("div.{}", uint(t)),
            O::Urem | O::UremImm => format!("rem.{}", uint(t)),
            O::Sdiv => format!("div.{}", sint(t)),
            O::Srem => format!("rem.{}", sint(t)),
            O::Band | O::BandImm => format!("and.{}", bits(t)),
            O::Bor | O::BorImm => format!("or.{}", bits(t)),
            O::Bxor | O::BxorImm => format!("xor.{}", bits(t)),
            O::Ishl | O::IshlImm => format!("shl.{}", bits(t)),
            O::Ushr | O::UshrImm => format!("shr.{}", uint(t)),
            O::Sshr | O::SshrImm => format!("shr.{}", sint(t)),
            O::Fadd => "add.rn.f32".into(),
            O::Fsub => "sub.rn.f32".into(),
            O::Fmul => "mul.rn.f32".into(),
            O::Fdiv => "div.rn.f32".into(),
            O::Fmin => "min.NaN.f32".into(),
            O::Fmax => "max.NaN.f32".into(),
            _ => return Err(format!("unsupported binary {op}")),
        };
        writeln!(self.text, " {mnemonic} {out}, {a}, {b};").unwrap();
        Ok(())
    }
}
fn int_cc(cc: IntCC) -> (&'static str, bool) {
    match cc {
        IntCC::Equal => ("eq", false),
        IntCC::NotEqual => ("ne", false),
        IntCC::SignedLessThan => ("lt", true),
        IntCC::SignedLessThanOrEqual => ("le", true),
        IntCC::SignedGreaterThan => ("gt", true),
        IntCC::SignedGreaterThanOrEqual => ("ge", true),
        IntCC::UnsignedLessThan => ("lt", false),
        IntCC::UnsignedLessThanOrEqual => ("le", false),
        IntCC::UnsignedGreaterThan => ("gt", false),
        IntCC::UnsignedGreaterThanOrEqual => ("ge", false),
    }
}
fn float_cc(cc: FloatCC) -> &'static str {
    match cc {
        FloatCC::Equal => "eq",
        FloatCC::NotEqual => "neu",
        FloatCC::LessThan => "lt",
        FloatCC::LessThanOrEqual => "le",
        FloatCC::GreaterThan => "gt",
        FloatCC::GreaterThanOrEqual => "ge",
        FloatCC::Ordered => "num",
        FloatCC::Unordered => "nan",
        FloatCC::OrderedNotEqual => "ne",
        FloatCC::UnorderedOrEqual => "equ",
        FloatCC::UnorderedOrLessThan => "ltu",
        FloatCC::UnorderedOrLessThanOrEqual => "leu",
        FloatCC::UnorderedOrGreaterThan => "gtu",
        FloatCC::UnorderedOrGreaterThanOrEqual => "geu",
    }
}
