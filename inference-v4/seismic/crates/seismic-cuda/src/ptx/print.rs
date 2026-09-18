//! Pure formatting of the selected target program. No SSA traversal, instruction
//! selection, helper choice, temporary allocation or storage planning occurs here.
use super::plan::*;
use std::fmt::Write;

pub fn print(plan: &TargetPlan) -> String {
    let mut out = String::from(match plan.target() {
        Target::Sm80Ptx70 => ".version 7.0\n.target sm_80\n.address_size 64\n\n",
    });
    for library in plan.libraries() {
        out.push_str(library.source());
        out.push('\n');
    }
    out.push_str(".visible .entry seismic_kernel(\n");
    let entry = plan
        .parameters()
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            matches!(
                p.role,
                ParameterRole::Buffers
                    | ParameterRole::Scalars
                    | ParameterRole::Scratch
                    | ParameterRole::Statuses
            )
        })
        .collect::<Vec<_>>();
    for (position, (id, p)) in entry.iter().enumerate() {
        writeln!(
            out,
            " .param .{} {}{}",
            p.data_type.spelling(),
            parameter(plan, ParameterId(*id)),
            if position + 1 == entry.len() { "" } else { "," }
        )
        .unwrap();
    }
    out.push_str(") {\n");
    for (id, r) in plan.registers().iter().enumerate() {
        writeln!(
            out,
            " .reg .{} {};",
            match r.class {
                RegisterClass::Bits32 => "b32",
                RegisterClass::Bits64 => "b64",
                RegisterClass::Float32 => "f32",
                RegisterClass::Predicate => "pred",
            },
            register(plan, RegisterId(id))
        )
        .unwrap();
    }
    for (id, p) in plan.parameters().iter().enumerate() {
        if matches!(
            p.role,
            ParameterRole::CallArgument(_) | ParameterRole::CallResult(_)
        ) {
            writeln!(
                out,
                " .param .{} {};",
                p.data_type.spelling(),
                parameter(plan, ParameterId(id))
            )
            .unwrap();
        }
    }
    for item in plan.body() {
        match item {
            Item::Label(l) => {
                writeln!(out, "{}:", label(*l)).unwrap();
            }
            Item::Instruction(instruction) => {
                out.push(' ');
                if let Some(p) = instruction.predicate {
                    write!(
                        out,
                        "@{}{} ",
                        if p.inverted { "!" } else { "" },
                        register(plan, p.register)
                    )
                    .unwrap();
                }
                let op = &instruction.operation;
                match *op {
                    Operation::Unary {
                        operation,
                        data_type,
                        rounding,
                        destination,
                        source,
                    } => write!(
                        out,
                        "{}{}.{ty} {}, {}",
                        match operation {
                            Unary::Move => "mov",
                            Unary::Negate => "neg",
                            Unary::Absolute => "abs",
                            Unary::Sqrt => "sqrt",
                        },
                        round(rounding),
                        register(plan, destination),
                        operand(plan, source),
                        ty = data_type.spelling()
                    )
                    .unwrap(),
                    Operation::Binary {
                        operation,
                        data_type,
                        rounding,
                        destination,
                        lhs,
                        rhs,
                    } => {
                        let mnemonic = match operation {
                            Binary::Add => "add",
                            Binary::Subtract => "sub",
                            Binary::Multiply(Multiply::Low) => "mul.lo",
                            Binary::Multiply(Multiply::Wide) => "mul.wide",
                            Binary::Multiply(Multiply::Floating) => "mul",
                            Binary::Divide => "div",
                            Binary::Remainder => "rem",
                            Binary::And => "and",
                            Binary::Or => "or",
                            Binary::Xor => "xor",
                            Binary::ShiftLeft => "shl",
                            Binary::ShiftRight => "shr",
                            Binary::MinimumNaN => "min.NaN",
                            Binary::MaximumNaN => "max.NaN",
                        };
                        write!(
                            out,
                            "{mnemonic}{}.{ty} {}, {}, {}",
                            round(rounding),
                            register(plan, destination),
                            operand(plan, lhs),
                            operand(plan, rhs),
                            ty = data_type.spelling()
                        )
                        .unwrap();
                    }
                    Operation::Convert {
                        destination_type,
                        source_type,
                        rounding,
                        destination,
                        source,
                    } => write!(
                        out,
                        "cvt{}.{}.{} {}, {}",
                        round(rounding),
                        destination_type.spelling(),
                        source_type.spelling(),
                        register(plan, destination),
                        operand(plan, source)
                    )
                    .unwrap(),
                    Operation::Compare {
                        comparison,
                        data_type,
                        destination,
                        lhs,
                        rhs,
                    } => write!(
                        out,
                        "setp.{}.{} {}, {}, {}",
                        comparison_name(comparison),
                        data_type.spelling(),
                        register(plan, destination),
                        operand(plan, lhs),
                        operand(plan, rhs)
                    )
                    .unwrap(),
                    Operation::Select {
                        data_type,
                        destination,
                        when_true,
                        when_false,
                        predicate,
                    } => write!(
                        out,
                        "selp.{} {}, {}, {}, {}",
                        data_type.spelling(),
                        register(plan, destination),
                        operand(plan, when_true),
                        operand(plan, when_false),
                        register(plan, predicate)
                    )
                    .unwrap(),
                    Operation::Shuffle { mode, destination, source, lane } => write!(out, "shfl.sync.{}.b32 {}, {}, {}, 31, 0xffffffff", match mode { ShuffleMode::Butterfly => "bfly", ShuffleMode::Index => "idx" }, register(plan,destination), register(plan,source), operand(plan,lane)).unwrap(),
                    Operation::Fma {
                        destination,
                        a,
                        b,
                        c,
                    } => write!(
                        out,
                        "fma.rn.f32 {}, {}, {}, {}",
                        register(plan, destination),
                        operand(plan, a),
                        operand(plan, b),
                        operand(plan, c)
                    )
                    .unwrap(),
                    Operation::Load {
                        space,
                        data_type,
                        destination,
                        address: at,
                    } => write!(
                        out,
                        "ld.{}.{} {}, {}",
                        space_name(space),
                        data_type.spelling(),
                        register(plan, destination),
                        address(plan, at)
                    )
                    .unwrap(),
                    Operation::Store {
                        space,
                        data_type,
                        address: at,
                        value,
                    } => write!(
                        out,
                        "st.{}.{} {}, {}",
                        space_name(space),
                        data_type.spelling(),
                        address(plan, at),
                        operand(plan, value)
                    )
                    .unwrap(),
                    Operation::Branch { target } => write!(out, "bra {}", label(target)).unwrap(),
                    Operation::Return => out.push_str("ret"),
                    Operation::Call {
                        function,
                        argument,
                        result,
                    } => write!(
                        out,
                        "call.uni ({}), {}, ({})",
                        parameter(plan, result),
                        function.symbol(),
                        parameter(plan, argument)
                    )
                    .unwrap(),
                }
                out.push_str(";\n");
            }
        }
    }
    out.push_str("}\n");
    out
}
fn round(rounding: Rounding) -> &'static str {
    match rounding {
        Rounding::Default => "",
        Rounding::NearestEven => ".rn",
        Rounding::TowardZeroInteger => ".rzi",
    }
}
fn space_name(space: Space) -> &'static str {
    match space {
        Space::Global => "global",
        Space::Parameter => "param",
    }
}
fn comparison_name(c: Comparison) -> &'static str {
    match c {
        Comparison::Equal => "eq",
        Comparison::NotEqual => "ne",
        Comparison::Less => "lt",
        Comparison::LessEqual => "le",
        Comparison::Greater => "gt",
        Comparison::GreaterEqual => "ge",
        Comparison::Number => "num",
        Comparison::NaN => "nan",
        Comparison::NotEqualOrUnordered => "neu",
        Comparison::EqualOrUnordered => "equ",
        Comparison::LessOrUnordered => "ltu",
        Comparison::LessEqualOrUnordered => "leu",
        Comparison::GreaterOrUnordered => "gtu",
        Comparison::GreaterEqualOrUnordered => "geu",
    }
}
fn label(l: Label) -> String {
    match l {
        Label::Block(b) => format!("block{b}"),
        Label::Edge(e) => format!("edge_{e}"),
    }
}
fn parameter(plan: &TargetPlan, p: ParameterId) -> String {
    match plan.parameters()[p.0].role {
        ParameterRole::Buffers => "buffers".into(),
        ParameterRole::Scalars => "scalars".into(),
        ParameterRole::Scratch => "scratch".into(),
        ParameterRole::Statuses => "statuses".into(),
        ParameterRole::CallArgument(i) => format!("call_arg_{i}"),
        ParameterRole::CallResult(i) => format!("call_result_{i}"),
    }
}
fn register(plan: &TargetPlan, r: RegisterId) -> String {
    match plan.registers()[r.0].name {
        RegisterName::Ssa(v) => format!("%v{v}"),
        RegisterName::Temporary(i) => format!("%tmp{i}"),
        RegisterName::Abi(abi) => match abi {
            AbiRegister::BlockIndex => "%bx",
            AbiRegister::BlockWidth => "%bd",
            AbiRegister::ThreadIndex => "%tx",
            AbiRegister::LinearIndex => "%linear",
            AbiRegister::ScratchOffset => "%scratch_offset",
            AbiRegister::StatusAddress => "%status_addr",
            AbiRegister::StatusOffset => "%status_offset",
            AbiRegister::Predicate => "%pred",
        }
        .into(),
    }
}
fn operand(plan: &TargetPlan, op: Operand) -> String {
    match op {
        Operand::Register(r) => register(plan, r),
        Operand::Signed(n) => n.to_string(),
        Operand::Unsigned(n) => n.to_string(),
        Operand::Float32Bits(bits) => format!("0f{bits:08x}"),
        Operand::Special(s) => match s {
            SpecialRegister::LaneIndex => "%laneid",
            SpecialRegister::BlockIndexX => "%ctaid.x",
            SpecialRegister::BlockWidthX => "%ntid.x",
            SpecialRegister::ThreadIndexX => "%tid.x",
        }
        .into(),
    }
}
fn address(plan: &TargetPlan, at: Address) -> String {
    let base = match at.base {
        AddressBase::Register(r) => register(plan, r),
        AddressBase::Parameter(p) => parameter(plan, p),
    };
    if at.offset == 0 {
        format!("[{base}]")
    } else {
        format!("[{base}{:+}]", at.offset)
    }
}
