//! Select a terminal PTX program from shared scalar SSA before accounting and
//! printing. The selected plan includes instruction expansion, call ABI traffic,
//! dispatch guards, status writes and scratch address arithmetic.
mod plan;
mod print;
pub mod family;
use cranelift_codegen::ir::{
    self, BlockArg, BlockCall, Inst, InstructionData as D, Opcode as O, Type, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
pub use plan::*;
pub use print::print;
use seismic_realization::{Dispatch, ScalarProgram};
use std::collections::HashMap;

/// Pure IR planning: no driver, native compilation, resource query or measurement.
pub fn prepare(program: &ScalarProgram) -> Result<TargetPlan, String> {
    let f = &program.function;
    let mut builder = Builder {
        f,
        program,
        plan: TargetPlan {
            target: Target::Sm80Ptx70,
            registers: Vec::new(),
            parameters: Vec::new(),
            body: Vec::new(),
            libraries: Vec::new(),
            domain: WorkDomain {
                work_items: program.work_items,
                lanes_per_item: program.participation.lanes(),
                dispatch: program.dispatch,
                scratch_bytes_per_item: program.scratch_bytes,
            },
        },
        values: HashMap::new(),
        temporary: 0,
        origin: Origin::InvocationAbi,
        predicate: RegisterId(0),
    };
    for value in f.dfg.values() {
        if f.dfg.value_is_real(value) {
            let id = builder.register(
                RegisterName::Ssa(value.as_u32()),
                class(f.dfg.value_type(value))?,
            );
            builder.values.insert(value, id);
        }
    }
    let bx = builder.register(
        RegisterName::Abi(AbiRegister::BlockIndex),
        RegisterClass::Bits32,
    );
    let bd = builder.register(
        RegisterName::Abi(AbiRegister::BlockWidth),
        RegisterClass::Bits32,
    );
    let tx = builder.register(
        RegisterName::Abi(AbiRegister::ThreadIndex),
        RegisterClass::Bits32,
    );
    let linear = builder.register(
        RegisterName::Abi(AbiRegister::LinearIndex),
        RegisterClass::Bits64,
    );
    let scratch_offset = builder.register(
        RegisterName::Abi(AbiRegister::ScratchOffset),
        RegisterClass::Bits64,
    );
    let status_address = builder.register(
        RegisterName::Abi(AbiRegister::StatusAddress),
        RegisterClass::Bits64,
    );
    let status_offset = builder.register(
        RegisterName::Abi(AbiRegister::StatusOffset),
        RegisterClass::Bits64,
    );
    builder.predicate = builder.register(
        RegisterName::Abi(AbiRegister::Predicate),
        RegisterClass::Predicate,
    );
    let buffers = builder.parameter(ParameterRole::Buffers, DataType::U64);
    let scalars = builder.parameter(ParameterRole::Scalars, DataType::U64);
    let scratch = builder.parameter(ParameterRole::Scratch, DataType::U64);
    let statuses = builder.parameter(ParameterRole::Statuses, DataType::U64);
    builder.origin = Origin::Dispatch;
    for (destination, source) in [
        (bx, SpecialRegister::BlockIndexX),
        (bd, SpecialRegister::BlockWidthX),
        (tx, SpecialRegister::ThreadIndexX),
    ] {
        builder.mov(DataType::U32, destination, Operand::Special(source));
    }
    builder.binary_op(
        Binary::Multiply(Multiply::Wide),
        DataType::U32,
        Rounding::Default,
        linear,
        bx.into(),
        bd.into(),
    );
    builder.push(Operation::Convert {
        destination_type: DataType::U64,
        source_type: DataType::U32,
        rounding: Rounding::Default,
        destination: status_offset,
        source: tx.into(),
    });
    builder.binary_op(
        Binary::Add,
        DataType::U64,
        Rounding::Default,
        linear,
        linear.into(),
        status_offset.into(),
    );
    builder.compare(
        Comparison::GreaterEqual,
        DataType::U64,
        linear.into(),
        Operand::Unsigned(program.work_items.checked_mul(u64::from(program.participation.lanes())).ok_or("CUDA participant count overflow")?),
    );
    builder.predicated(Operation::Return, false);
    builder.origin = Origin::InvocationAbi;
    builder.push(Operation::Load {
        space: Space::Parameter,
        data_type: DataType::U64,
        destination: status_address,
        address: Address::parameter(statuses),
    });
    builder.binary_op(
        Binary::Multiply(Multiply::Low),
        DataType::U64,
        Rounding::Default,
        status_offset,
        linear.into(),
        Operand::Unsigned(4),
    );
    builder.binary_op(
        Binary::Add,
        DataType::U64,
        Rounding::Default,
        status_address,
        status_address.into(),
        status_offset.into(),
    );
    let entry = f.layout.entry_block().ok_or("empty scalar program")?;
    let params = f.dfg.block_params(entry);
    let expected = if program.dispatch == Dispatch::ParallelRoot {
        4
    } else {
        3
    };
    if params.len() != expected || params.iter().any(|&p| f.dfg.value_type(p) != types::I64) {
        return Err(
            "CUDA scalar entry ABI must have pointer parameters and an optional I64 work index"
                .into(),
        );
    }
    for (&value, parameter) in params.iter().take(3).zip([buffers, scalars, scratch]) {
        builder.push(Operation::Load {
            space: Space::Parameter,
            data_type: DataType::U64,
            destination: builder.value(value),
            address: Address::parameter(parameter),
        });
    }
    if program.dispatch == Dispatch::ParallelRoot {
        if program.participation.lanes() > 1 {
            builder.binary_op(Binary::Divide, DataType::U64, Rounding::Default, builder.value(params[3]), linear.into(), Operand::Unsigned(u64::from(program.participation.lanes())));
        } else { builder.mov(DataType::U64, builder.value(params[3]), linear.into()); }
        builder.binary_op(
            Binary::Multiply(Multiply::Low),
            DataType::U64,
            Rounding::Default,
            scratch_offset,
            linear.into(),
            Operand::Unsigned(program.scratch_bytes as u64),
        );
        builder.binary_op(
            Binary::Add,
            DataType::U64,
            Rounding::Default,
            builder.value(params[2]),
            builder.value(params[2]).into(),
            scratch_offset.into(),
        );
    }
    for block in f.layout.blocks() {
        builder
            .plan
            .body
            .push(Item::Label(Label::Block(block.as_u32())));
        for inst in f.layout.block_insts(block) {
            builder.origin = Origin::Ssa {
                instruction: inst.as_u32(),
                block: block.as_u32(),
            };
            builder
                .instruction(inst, status_address)
                .map_err(|e| format!("PTX {}: {e}", f.dfg.display_inst(inst)))?;
        }
    }
    builder.plan.validate()?;
    Ok(builder.plan)
}
fn class(t: Type) -> Result<RegisterClass, String> {
    Ok(match t {
        types::I8 | types::I16 | types::I32 => RegisterClass::Bits32,
        types::I64 => RegisterClass::Bits64,
        types::F32 => RegisterClass::Float32,
        _ => return Err(format!("unsupported PTX register type {t}")),
    })
}
fn uint(t: Type) -> DataType {
    if t == types::I64 {
        DataType::U64
    } else {
        DataType::U32
    }
}
fn sint(t: Type) -> DataType {
    if t == types::I64 {
        DataType::S64
    } else {
        DataType::S32
    }
}
fn bits(t: Type) -> DataType {
    if t == types::I64 {
        DataType::B64
    } else {
        DataType::B32
    }
}
fn value_type(t: Type) -> DataType {
    if t == types::F32 {
        DataType::F32
    } else {
        bits(t)
    }
}
fn memory_type(t: Type) -> Result<DataType, String> {
    Ok(match t {
        types::I8 => DataType::U8,
        types::I16 => DataType::U16,
        types::I32 => DataType::U32,
        types::I64 => DataType::U64,
        types::F32 => DataType::F32,
        _ => return Err(format!("unsupported PTX memory type {t}")),
    })
}
struct Builder<'a> {
    f: &'a ir::Function,
    program: &'a ScalarProgram,
    plan: TargetPlan,
    values: HashMap<Value, RegisterId>,
    temporary: usize,
    origin: Origin,
    predicate: RegisterId,
}
impl Builder<'_> {
    fn value(&self, value: Value) -> RegisterId {
        self.values[&self.f.dfg.resolve_aliases(value)]
    }
    fn register(&mut self, name: RegisterName, class: RegisterClass) -> RegisterId {
        let id = RegisterId(self.plan.registers.len());
        self.plan.registers.push(Register { name, class });
        id
    }
    fn temp(&mut self, t: Type) -> Result<RegisterId, String> {
        let name = RegisterName::Temporary(self.temporary);
        self.temporary += 1;
        Ok(self.register(name, class(t)?))
    }
    fn parameter(&mut self, role: ParameterRole, data_type: DataType) -> ParameterId {
        let id = ParameterId(self.plan.parameters.len());
        self.plan.parameters.push(Parameter { role, data_type });
        id
    }
    fn push(&mut self, operation: Operation) {
        self.plan.body.push(Item::Instruction(Instruction {
            origin: self.origin,
            predicate: None,
            operation,
        }));
    }
    fn predicated(&mut self, operation: Operation, inverted: bool) {
        self.plan.body.push(Item::Instruction(Instruction {
            origin: self.origin,
            predicate: Some(Predicate {
                register: self.predicate,
                inverted,
            }),
            operation,
        }));
    }
    fn mov(&mut self, data_type: DataType, destination: RegisterId, source: Operand) {
        self.push(Operation::Unary {
            operation: Unary::Move,
            data_type,
            rounding: Rounding::Default,
            destination,
            source,
        });
    }
    fn binary_op(
        &mut self,
        operation: Binary,
        data_type: DataType,
        rounding: Rounding,
        destination: RegisterId,
        lhs: Operand,
        rhs: Operand,
    ) {
        self.push(Operation::Binary {
            operation,
            data_type,
            rounding,
            destination,
            lhs,
            rhs,
        });
    }
    fn compare(&mut self, comparison: Comparison, data_type: DataType, lhs: Operand, rhs: Operand) {
        self.push(Operation::Compare {
            comparison,
            data_type,
            destination: self.predicate,
            lhs,
            rhs,
        });
    }
    fn comparison(
        &mut self,
        out: RegisterId,
        comparison: Comparison,
        data_type: DataType,
        lhs: Operand,
        rhs: Operand,
    ) {
        self.compare(comparison, data_type, lhs, rhs);
        self.push(Operation::Select {
            data_type: DataType::U32,
            destination: out,
            when_true: Operand::Unsigned(1),
            when_false: Operand::Unsigned(0),
            predicate: self.predicate,
        });
    }
    fn edge(&mut self, edge: BlockCall) -> Result<(), String> {
        let block = edge.block(&self.f.dfg.value_lists);
        let args = edge.args(&self.f.dfg.value_lists).collect::<Vec<_>>();
        let params = self.f.dfg.block_params(block).to_vec();
        if params.len() != args.len() {
            return Err("block argument count differs from parameters".into());
        }
        // Simultaneous block-parameter assignment preserves backedge swaps/cycles.
        let mut moves = Vec::new();
        for (p, a) in params.into_iter().zip(args) {
            let BlockArg::Value(a) = a else {
                return Err("exception block argument".into());
            };
            let t = self.f.dfg.value_type(p);
            let temp = self.temp(t)?;
            self.mov(value_type(t), temp, self.value(a).into());
            moves.push((p, temp, t));
        }
        for (p, temp, t) in moves {
            self.mov(value_type(t), self.value(p), temp.into());
        }
        self.push(Operation::Branch {
            target: Label::Block(block.as_u32()),
        });
        Ok(())
    }
    fn instruction(&mut self, inst: Inst, status_address: RegisterId) -> Result<(), String> {
        let data = &self.f.dfg.insts[inst];
        let op = data.opcode();
        let result = self.f.dfg.inst_results(inst).first().copied();
        let out = result.map(|v| self.value(v));
        let t = result
            .map(|v| self.f.dfg.value_type(v))
            .unwrap_or(types::INVALID);
        let output = || out.ok_or_else(|| "instruction has no destination register".to_string());
        match data {
            D::UnaryImm { imm, .. } if op == O::Iconst => {
                self.mov(bits(t), output()?, Operand::Signed(imm.bits()))
            }
            D::UnaryIeee32 { imm, .. } => {
                self.mov(DataType::F32, output()?, Operand::Float32Bits(imm.bits()))
            }
            D::Load { arg, offset, .. } => self.push(Operation::Load {
                space: Space::Global,
                data_type: memory_type(t)?,
                destination: output()?,
                address: Address::register(self.value(*arg), i32::from(*offset)),
            }),
            D::Store { args, offset, .. } => self.push(Operation::Store {
                space: Space::Global,
                data_type: memory_type(self.f.dfg.value_type(args[0]))?,
                address: Address::register(self.value(args[1]), i32::from(*offset)),
                value: self.value(args[0]).into(),
            }),
            D::Jump { destination, .. } => self.edge(*destination)?,
            D::Brif { arg, blocks, .. } => {
                let label = Label::Edge(self.temporary);
                self.temporary += 1;
                self.compare(
                    Comparison::NotEqual,
                    DataType::U32,
                    self.value(*arg).into(),
                    Operand::Unsigned(0),
                );
                self.predicated(Operation::Branch { target: label }, false);
                self.edge(blocks[1])?;
                self.plan.body.push(Item::Label(label));
                self.edge(blocks[0])?;
            }
            D::MultiAry { args, .. } if op == O::Return => {
                let args = args.as_slice(&self.f.dfg.value_lists);
                if args.len() != 1 {
                    return Err("CUDA scalar return ABI expects one status value".into());
                }
                self.push(Operation::Store {
                    space: Space::Global,
                    data_type: DataType::U32,
                    address: Address::register(status_address, 0),
                    value: self.value(args[0]).into(),
                });
                self.push(Operation::Return);
            }
            D::IntCompare { args, cond, .. } => {
                let (cc, signed) = int_cc(*cond);
                let at = self.f.dfg.value_type(args[0]);
                self.comparison(
                    output()?,
                    cc,
                    if signed { sint(at) } else { uint(at) },
                    self.value(args[0]).into(),
                    self.value(args[1]).into(),
                );
            }
            D::IntCompareImm { arg, cond, imm, .. } => {
                let (cc, signed) = int_cc(*cond);
                let at = self.f.dfg.value_type(*arg);
                self.comparison(
                    output()?,
                    cc,
                    if signed { sint(at) } else { uint(at) },
                    self.value(*arg).into(),
                    Operand::Signed(imm.bits()),
                );
            }
            D::FloatCompare { args, cond, .. } => self.comparison(
                output()?,
                float_cc(*cond),
                DataType::F32,
                self.value(args[0]).into(),
                self.value(args[1]).into(),
            ),
            D::Ternary { args, .. } if op == O::Select => {
                self.compare(
                    Comparison::NotEqual,
                    DataType::U32,
                    self.value(args[0]).into(),
                    Operand::Unsigned(0),
                );
                self.push(Operation::Select {
                    data_type: value_type(t),
                    destination: output()?,
                    when_true: self.value(args[1]).into(),
                    when_false: self.value(args[2]).into(),
                    predicate: self.predicate,
                });
            }
            D::Ternary { args, .. } if op == O::Fma => self.push(Operation::Fma {
                destination: output()?,
                a: self.value(args[0]).into(),
                b: self.value(args[1]).into(),
                c: self.value(args[2]).into(),
            }),
            D::Binary { args, .. } => self.binary(
                op,
                output()?,
                t,
                self.value(args[0]).into(),
                self.value(args[1]).into(),
            )?,
            D::BinaryImm64 { arg, imm, .. } => self.binary(
                op,
                output()?,
                t,
                self.value(*arg).into(),
                Operand::Signed(imm.bits()),
            )?,
            D::Unary { arg, .. } => self.unary(op, output()?, t, *arg)?,
            D::LoadNoOffset { arg, .. } if op == O::Bitcast => {
                self.mov(bits(t), output()?, self.value(*arg).into())
            }
            D::Call { func_ref, .. } => {
                if let Some((_, operation)) = self.program.backend_calls.iter().find(|(reference,_)|reference==func_ref) {
                    let args=self.f.dfg.inst_args(inst);
                    match operation {
                        seismic_realization::ParticipantOperation::LaneIndex => {
                            if !args.is_empty() || t != types::I32 { return Err("invalid lane-index call ABI".into()); }
                            self.mov(DataType::U32, output()?, Operand::Special(SpecialRegister::LaneIndex));
                        }
                        seismic_realization::ParticipantOperation::ShuffleIndex => {
                            if args.len()!=2 || t!=types::F32 || self.f.dfg.value_type(args[0])!=types::F32 || self.f.dfg.value_type(args[1])!=types::I32 {return Err("shuffle-index requires f32 value and i32 lane".into());}
                            let source=self.temp(types::I32)?;let result=self.temp(types::I32)?;
                            self.mov(DataType::B32,source,self.value(args[0]).into());
                            self.push(Operation::Shuffle{mode:ShuffleMode::Index,destination:result,source,lane:self.value(args[1]).into()});
                            self.mov(DataType::B32,output()?,result.into());
                        }
                        seismic_realization::ParticipantOperation::Reduce(seismic_lang::ir::ReduceOp::Sum) => {
                            if args.len()!=1 || t!=types::F32 || self.f.dfg.value_type(args[0])!=types::F32 { return Err("warp sum requires f32 input and output".into()); }
                            let accumulator=self.temp(types::F32)?;
                            self.mov(DataType::F32,accumulator,self.value(args[0]).into());
                            for delta in [16,8,4,2,1] {
                                let input=self.temp(types::I32)?;let exchange=self.temp(types::I32)?;let received=self.temp(types::F32)?;
                                self.mov(DataType::B32,input,accumulator.into());
                                self.push(Operation::Shuffle{mode:ShuffleMode::Butterfly,destination:exchange,source:input,lane:Operand::Unsigned(delta)});
                                self.mov(DataType::B32,received,exchange.into());
                                self.binary_op(Binary::Add,DataType::F32,Rounding::NearestEven,accumulator,accumulator.into(),received.into());
                            }
                            // Uniform source results include NaN payloads: publish the
                            // same lane-zero tree result to every participant.
                            let input=self.temp(types::I32)?;let broadcast=self.temp(types::I32)?;
                            self.mov(DataType::B32,input,accumulator.into());
                            self.push(Operation::Shuffle{mode:ShuffleMode::Index,destination:broadcast,source:input,lane:Operand::Unsigned(0)});
                            self.mov(DataType::B32,output()?,broadcast.into());
                        }
                        _=>return Err("participant operation has no selected PTX implementation".into()),
                    }
                    return Ok(());
                }
                let (_, math) = self
                    .program
                    .imports
                    .iter()
                    .find(|(r, _)| r == func_ref)
                    .ok_or("unrecognized external function")?;
                let math = *math;
                let args = self.f.dfg.inst_args(inst);
                if args.len() != 1
                    || t != types::F32
                    || self.f.dfg.value_type(args[0]) != types::F32
                {
                    return Err("CUDA math helper ABI requires f32 -> f32".into());
                }
                let library = Helper::for_function(math).library;
                if !self.plan.libraries.contains(&library) {
                    self.plan.libraries.push(library);
                }
                let call = inst.as_u32() as usize;
                let argument = self.parameter(ParameterRole::CallArgument(call), DataType::B32);
                let result = self.parameter(ParameterRole::CallResult(call), DataType::B32);
                self.push(Operation::Store {
                    space: Space::Parameter,
                    data_type: DataType::F32,
                    address: Address::parameter(argument),
                    value: self.value(args[0]).into(),
                });
                self.push(Operation::Call {
                    function: math,
                    argument,
                    result,
                });
                self.push(Operation::Load {
                    space: Space::Parameter,
                    data_type: DataType::F32,
                    destination: output()?,
                    address: Address::parameter(result),
                });
            }
            _ => return Err(format!("unsupported instruction {op}")),
        }
        Ok(())
    }
    fn unary(&mut self, op: O, out: RegisterId, t: Type, arg: Value) -> Result<(), String> {
        let source = self.value(arg).into();
        let at = self.f.dfg.value_type(arg);
        let instruction = match op {
            O::Fneg => Operation::Unary {
                operation: Unary::Negate,
                data_type: DataType::F32,
                rounding: Rounding::Default,
                destination: out,
                source,
            },
            O::Fabs => Operation::Unary {
                operation: Unary::Absolute,
                data_type: DataType::F32,
                rounding: Rounding::Default,
                destination: out,
                source,
            },
            O::Sqrt => Operation::Unary {
                operation: Unary::Sqrt,
                data_type: DataType::F32,
                rounding: Rounding::NearestEven,
                destination: out,
                source,
            },
            O::Ineg => Operation::Unary {
                operation: Unary::Negate,
                data_type: sint(t),
                rounding: Rounding::Default,
                destination: out,
                source,
            },
            O::Uextend | O::Ireduce if t.bits() >= 32 => Operation::Convert {
                destination_type: uint(t),
                source_type: uint(at),
                rounding: Rounding::Default,
                destination: out,
                source,
            },
            O::Uextend => Operation::Convert {
                destination_type: uint(t),
                source_type: uint(at),
                rounding: Rounding::Default,
                destination: out,
                source,
            },
            O::Sextend if at == types::I32 && t == types::I64 => Operation::Convert {
                destination_type: DataType::S64,
                source_type: DataType::S32,
                rounding: Rounding::Default,
                destination: out,
                source,
            },
            O::Ireduce if t.bits() < 32 => Operation::Binary {
                operation: Binary::And,
                data_type: DataType::B32,
                rounding: Rounding::Default,
                destination: out,
                lhs: source,
                rhs: Operand::Unsigned((1u64 << t.bits()) - 1),
            },
            O::FcvtFromUint => Operation::Convert {
                destination_type: DataType::F32,
                source_type: uint(at),
                rounding: Rounding::NearestEven,
                destination: out,
                source,
            },
            O::FcvtFromSint => Operation::Convert {
                destination_type: DataType::F32,
                source_type: sint(at),
                rounding: Rounding::NearestEven,
                destination: out,
                source,
            },
            O::FcvtToSintSat => Operation::Convert {
                destination_type: sint(t),
                source_type: DataType::F32,
                rounding: Rounding::TowardZeroInteger,
                destination: out,
                source,
            },
            O::FcvtToUintSat => Operation::Convert {
                destination_type: uint(t),
                source_type: DataType::F32,
                rounding: Rounding::TowardZeroInteger,
                destination: out,
                source,
            },
            _ => return Err(format!("unsupported unary {op}")),
        };
        self.push(instruction);
        Ok(())
    }
    fn binary(
        &mut self,
        op: O,
        out: RegisterId,
        t: Type,
        a: Operand,
        b: Operand,
    ) -> Result<(), String> {
        if op == O::Srem {
            return self.signed_remainder(out, t, a, b);
        }
        if op == O::IrsubImm {
            self.binary_op(Binary::Subtract, uint(t), Rounding::Default, out, b, a);
            return Ok(());
        }
        let (operation, data_type, rounding) = match op {
            O::Iadd | O::IaddImm => (Binary::Add, uint(t), Rounding::Default),
            O::Isub => (Binary::Subtract, uint(t), Rounding::Default),
            O::Imul | O::ImulImm => (Binary::Multiply(Multiply::Low), uint(t), Rounding::Default),
            O::Udiv | O::UdivImm => (Binary::Divide, uint(t), Rounding::Default),
            O::Urem | O::UremImm => (Binary::Remainder, uint(t), Rounding::Default),
            O::Sdiv => (Binary::Divide, sint(t), Rounding::Default),
            O::Band | O::BandImm => (Binary::And, bits(t), Rounding::Default),
            O::Bor | O::BorImm => (Binary::Or, bits(t), Rounding::Default),
            O::Bxor | O::BxorImm => (Binary::Xor, bits(t), Rounding::Default),
            O::Ishl | O::IshlImm => (Binary::ShiftLeft, bits(t), Rounding::Default),
            O::Ushr | O::UshrImm => (Binary::ShiftRight, uint(t), Rounding::Default),
            O::Sshr | O::SshrImm => (Binary::ShiftRight, sint(t), Rounding::Default),
            O::Fadd => (Binary::Add, DataType::F32, Rounding::NearestEven),
            O::Fsub => (Binary::Subtract, DataType::F32, Rounding::NearestEven),
            O::Fmul => (
                Binary::Multiply(Multiply::Floating),
                DataType::F32,
                Rounding::NearestEven,
            ),
            O::Fdiv => (Binary::Divide, DataType::F32, Rounding::NearestEven),
            O::Fmin => (Binary::MinimumNaN, DataType::F32, Rounding::Default),
            O::Fmax => (Binary::MaximumNaN, DataType::F32, Rounding::Default),
            _ => return Err(format!("unsupported binary {op}")),
        };
        self.binary_op(operation, data_type, rounding, out, a, b);
        Ok(())
    }

    /// Scalar SSA defines truncating signed remainder, including MIN % -1 = 0.
    /// PTX rem.s leaves negative operands machine-dependent. Derive unsigned
    /// magnitudes, rem.u, and restore the dividend sign using explicit target
    /// instructions so printing and resource analysis see the same expansion.
    fn signed_remainder(
        &mut self,
        out: RegisterId,
        t: Type,
        mut a: Operand,
        mut b: Operand,
    ) -> Result<(), String> {
        if t.bits() < 32 {
            // I8/I16 occupy PTX b32 registers. Normalize their signed value
            // before comparisons, independently of upper register bits.
            for operand in [&mut a, &mut b] {
                let normalized = self.temp(t)?;
                let shift = Operand::Unsigned(u64::from(32 - t.bits()));
                self.binary_op(
                    Binary::ShiftLeft,
                    DataType::B32,
                    Rounding::Default,
                    normalized,
                    *operand,
                    shift,
                );
                self.binary_op(
                    Binary::ShiftRight,
                    DataType::S32,
                    Rounding::Default,
                    normalized,
                    normalized.into(),
                    shift,
                );
                *operand = normalized.into();
            }
        }
        let magnitudes = [self.temp(t)?, self.temp(t)?];
        for (operand, magnitude) in [a, b].into_iter().zip(magnitudes) {
            // Unsigned subtraction preserves MIN's magnitude without overflow.
            self.binary_op(
                Binary::Subtract,
                uint(t),
                Rounding::Default,
                magnitude,
                Operand::Unsigned(0),
                operand,
            );
            self.compare(Comparison::Less, sint(t), operand, Operand::Signed(0));
            self.push(Operation::Select {
                data_type: uint(t),
                destination: magnitude,
                when_true: magnitude.into(),
                when_false: operand,
                predicate: self.predicate,
            });
        }
        self.binary_op(
            Binary::Remainder,
            uint(t),
            Rounding::Default,
            out,
            magnitudes[0].into(),
            magnitudes[1].into(),
        );
        let negative = self.temp(t)?;
        self.binary_op(
            Binary::Subtract,
            uint(t),
            Rounding::Default,
            negative,
            Operand::Unsigned(0),
            out.into(),
        );
        self.compare(Comparison::Less, sint(t), a, Operand::Signed(0));
        self.push(Operation::Select {
            data_type: uint(t),
            destination: out,
            when_true: negative.into(),
            when_false: out.into(),
            predicate: self.predicate,
        });
        if t.bits() < 32 {
            self.binary_op(
                Binary::And,
                DataType::B32,
                Rounding::Default,
                out,
                out.into(),
                Operand::Unsigned((1u64 << t.bits()) - 1),
            );
        }
        Ok(())
    }
}
fn int_cc(cc: IntCC) -> (Comparison, bool) {
    match cc {
        IntCC::Equal => (Comparison::Equal, false),
        IntCC::NotEqual => (Comparison::NotEqual, false),
        IntCC::SignedLessThan => (Comparison::Less, true),
        IntCC::SignedLessThanOrEqual => (Comparison::LessEqual, true),
        IntCC::SignedGreaterThan => (Comparison::Greater, true),
        IntCC::SignedGreaterThanOrEqual => (Comparison::GreaterEqual, true),
        IntCC::UnsignedLessThan => (Comparison::Less, false),
        IntCC::UnsignedLessThanOrEqual => (Comparison::LessEqual, false),
        IntCC::UnsignedGreaterThan => (Comparison::Greater, false),
        IntCC::UnsignedGreaterThanOrEqual => (Comparison::GreaterEqual, false),
    }
}
fn float_cc(cc: FloatCC) -> Comparison {
    match cc {
        FloatCC::Equal => Comparison::Equal,
        FloatCC::NotEqual => Comparison::NotEqualOrUnordered,
        FloatCC::LessThan => Comparison::Less,
        FloatCC::LessThanOrEqual => Comparison::LessEqual,
        FloatCC::GreaterThan => Comparison::Greater,
        FloatCC::GreaterThanOrEqual => Comparison::GreaterEqual,
        FloatCC::Ordered => Comparison::Number,
        FloatCC::Unordered => Comparison::NaN,
        FloatCC::OrderedNotEqual => Comparison::NotEqual,
        FloatCC::UnorderedOrEqual => Comparison::EqualOrUnordered,
        FloatCC::UnorderedOrLessThan => Comparison::LessOrUnordered,
        FloatCC::UnorderedOrLessThanOrEqual => Comparison::LessEqualOrUnordered,
        FloatCC::UnorderedOrGreaterThan => Comparison::GreaterOrUnordered,
        FloatCC::UnorderedOrGreaterThanOrEqual => Comparison::GreaterEqualOrUnordered,
    }
}
