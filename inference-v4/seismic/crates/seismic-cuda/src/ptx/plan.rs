//! Terminal PTX implementation representation. Register names, instruction forms,
//! ABI traffic, control edges and helper bodies are selected before printing.
//! PTX virtual-register counts are not physical register allocation or native cost.
use seismic_realization::{Dispatch, MathFunction};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegisterId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegisterClass {
    Bits32,
    Bits64,
    Float32,
    /// Binary64 values: the shared scalar realization evaluates `rsqrt` in binary64.
    Float64,
    Predicate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    B32,
    B64,
    U8,
    U16,
    U32,
    U64,
    S32,
    S64,
    F32,
    F64,
    Pred,
}
impl DataType {
    pub fn bits(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::B64 | Self::U64 | Self::S64 | Self::F64 => 64,
            Self::Pred => 1,
            _ => 32,
        }
    }
    pub fn spelling(self) -> &'static str {
        match self {
            Self::B32 => "b32",
            Self::B64 => "b64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::S32 => "s32",
            Self::S64 => "s64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Pred => "pred",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbiRegister {
    BlockIndex,
    BlockWidth,
    ThreadIndex,
    LinearIndex,
    ScratchOffset,
    StatusAddress,
    StatusOffset,
    Predicate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterName {
    Ssa(u32),
    Temporary(usize),
    Abi(AbiRegister),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Register {
    pub name: RegisterName,
    pub class: RegisterClass,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterRole {
    Buffers,
    Scalars,
    Scratch,
    Statuses,
    CallArgument(usize),
    CallResult(usize),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parameter {
    pub role: ParameterRole,
    pub data_type: DataType,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialRegister {
    LaneIndex,
    BlockIndexX,
    BlockWidthX,
    ThreadIndexX,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Register(RegisterId),
    Signed(i64),
    Unsigned(u64),
    Float32Bits(u32),
    Float64Bits(u64),
    Special(SpecialRegister),
}
impl From<RegisterId> for Operand {
    fn from(value: RegisterId) -> Self {
        Self::Register(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressBase {
    Register(RegisterId),
    Parameter(ParameterId),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address {
    pub base: AddressBase,
    pub offset: i32,
}
impl Address {
    pub fn register(register: RegisterId, offset: i32) -> Self {
        Self {
            base: AddressBase::Register(register),
            offset,
        }
    }
    pub fn parameter(parameter: ParameterId) -> Self {
        Self {
            base: AddressBase::Parameter(parameter),
            offset: 0,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Space {
    Global,
    Parameter,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rounding {
    Default,
    NearestEven,
    TowardZeroInteger,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Multiply {
    Low,
    Wide,
    Floating,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Unary {
    Move,
    Negate,
    Absolute,
    Sqrt,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Binary {
    Add,
    Subtract,
    Multiply(Multiply),
    Divide,
    Remainder,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
    MinimumNaN,
    MaximumNaN,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Comparison {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Number,
    NaN,
    NotEqualOrUnordered,
    EqualOrUnordered,
    LessOrUnordered,
    LessEqualOrUnordered,
    GreaterOrUnordered,
    GreaterEqualOrUnordered,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Label {
    Block(u32),
    Edge(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Predicate {
    pub register: RegisterId,
    pub inverted: bool,
}
/// An exact implementation form key; consumers must supply a mapping/resource
/// contract for this form. No guessed latency or throughput is attached here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Primitive {
    Shuffle {
        mode: ShuffleMode,
    },
    Unary {
        operation: Unary,
        data_type: DataType,
        rounding: Rounding,
    },
    Binary {
        operation: Binary,
        data_type: DataType,
        rounding: Rounding,
    },
    Convert {
        destination: DataType,
        source: DataType,
        rounding: Rounding,
    },
    Compare {
        comparison: Comparison,
        data_type: DataType,
    },
    Select {
        data_type: DataType,
    },
    Fma {
        data_type: DataType,
        rounding: Rounding,
    },
    Load {
        space: Space,
        data_type: DataType,
    },
    Store {
        space: Space,
        data_type: DataType,
    },
    Branch,
    Return,
    CallUniform,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Dispatch,
    InvocationAbi,
    Ssa { instruction: u32, block: u32 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShuffleMode {
    Butterfly,
    Index,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    /// Full-warp, convergent register communication with completion before use.
    Shuffle {
        mode: ShuffleMode,
        destination: RegisterId,
        source: RegisterId,
        lane: Operand,
    },
    Unary {
        operation: Unary,
        data_type: DataType,
        rounding: Rounding,
        destination: RegisterId,
        source: Operand,
    },
    Binary {
        operation: Binary,
        data_type: DataType,
        rounding: Rounding,
        destination: RegisterId,
        lhs: Operand,
        rhs: Operand,
    },
    Convert {
        destination_type: DataType,
        source_type: DataType,
        rounding: Rounding,
        destination: RegisterId,
        source: Operand,
    },
    Compare {
        comparison: Comparison,
        data_type: DataType,
        destination: RegisterId,
        lhs: Operand,
        rhs: Operand,
    },
    Select {
        data_type: DataType,
        destination: RegisterId,
        when_true: Operand,
        when_false: Operand,
        predicate: RegisterId,
    },
    Fma {
        destination: RegisterId,
        a: Operand,
        b: Operand,
        c: Operand,
    },
    Load {
        space: Space,
        data_type: DataType,
        destination: RegisterId,
        address: Address,
    },
    Store {
        space: Space,
        data_type: DataType,
        address: Address,
        value: Operand,
    },
    Branch {
        target: Label,
    },
    Return,
    Call {
        function: MathFunction,
        argument: ParameterId,
        result: ParameterId,
    },
}
impl Operation {
    pub fn primitive(&self) -> Primitive {
        match *self {
            Self::Shuffle { mode, .. } => Primitive::Shuffle { mode },
            Self::Unary {
                operation,
                data_type,
                rounding,
                ..
            } => Primitive::Unary {
                operation,
                data_type,
                rounding,
            },
            Self::Binary {
                operation,
                data_type,
                rounding,
                ..
            } => Primitive::Binary {
                operation,
                data_type,
                rounding,
            },
            Self::Convert {
                destination_type,
                source_type,
                rounding,
                ..
            } => Primitive::Convert {
                destination: destination_type,
                source: source_type,
                rounding,
            },
            Self::Compare {
                comparison,
                data_type,
                ..
            } => Primitive::Compare {
                comparison,
                data_type,
            },
            Self::Select { data_type, .. } => Primitive::Select { data_type },
            Self::Fma { .. } => Primitive::Fma {
                data_type: DataType::F32,
                rounding: Rounding::NearestEven,
            },
            Self::Load {
                space, data_type, ..
            } => Primitive::Load { space, data_type },
            Self::Store {
                space, data_type, ..
            } => Primitive::Store { space, data_type },
            Self::Branch { .. } => Primitive::Branch,
            Self::Return => Primitive::Return,
            Self::Call { .. } => Primitive::CallUniform,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub origin: Origin,
    pub predicate: Option<Predicate>,
    pub operation: Operation,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Label(Label),
    Instruction(Instruction),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryDirection {
    Read,
    Write,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryEffect {
    pub space: Space,
    pub direction: MemoryDirection,
    pub address: Address,
    pub bytes: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Next,
    Branch(Label),
    Return,
    Call(MathFunction),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Effects {
    pub register_reads: Vec<RegisterId>,
    pub register_writes: Vec<RegisterId>,
    pub parameter_reads: Vec<ParameterId>,
    pub parameter_writes: Vec<ParameterId>,
    pub special_reads: Vec<SpecialRegister>,
    pub memory: Option<MemoryEffect>,
    pub control: Control,
}
impl Instruction {
    pub fn effects(&self) -> Effects {
        let mut effect = Effects {
            register_reads: Vec::new(),
            register_writes: Vec::new(),
            parameter_reads: Vec::new(),
            parameter_writes: Vec::new(),
            special_reads: Vec::new(),
            memory: None,
            control: Control::Next,
        };
        if let Some(predicate) = self.predicate {
            effect.register_reads.push(predicate.register);
        }
        fn operand(effect: &mut Effects, operand: Operand) {
            match operand {
                Operand::Register(r) => effect.register_reads.push(r),
                Operand::Special(r) => effect.special_reads.push(r),
                _ => {}
            }
        }
        fn address(effect: &mut Effects, address: Address, direction: MemoryDirection) {
            match address.base {
                AddressBase::Register(r) => effect.register_reads.push(r),
                AddressBase::Parameter(p) => {
                    if direction == MemoryDirection::Read {
                        effect.parameter_reads.push(p)
                    } else {
                        effect.parameter_writes.push(p)
                    }
                }
            }
        }
        match self.operation {
            Operation::Shuffle {
                destination,
                source,
                lane,
                ..
            } => {
                effect.register_writes.push(destination);
                effect.register_reads.push(source);
                operand(&mut effect, lane);
            }
            Operation::Unary {
                destination,
                source,
                ..
            }
            | Operation::Convert {
                destination,
                source,
                ..
            } => {
                effect.register_writes.push(destination);
                operand(&mut effect, source);
            }
            Operation::Binary {
                destination,
                lhs,
                rhs,
                ..
            }
            | Operation::Compare {
                destination,
                lhs,
                rhs,
                ..
            } => {
                effect.register_writes.push(destination);
                operand(&mut effect, lhs);
                operand(&mut effect, rhs);
            }
            Operation::Select {
                destination,
                when_true,
                when_false,
                predicate,
                ..
            } => {
                effect.register_writes.push(destination);
                operand(&mut effect, when_true);
                operand(&mut effect, when_false);
                effect.register_reads.push(predicate);
            }
            Operation::Fma {
                destination,
                a,
                b,
                c,
            } => {
                effect.register_writes.push(destination);
                for value in [a, b, c] {
                    operand(&mut effect, value);
                }
            }
            Operation::Load {
                space,
                data_type,
                destination,
                address: at,
            } => {
                effect.register_writes.push(destination);
                address(&mut effect, at, MemoryDirection::Read);
                effect.memory = Some(MemoryEffect {
                    space,
                    direction: MemoryDirection::Read,
                    address: at,
                    bytes: data_type.bits() / 8,
                });
            }
            Operation::Store {
                space,
                data_type,
                address: at,
                value,
            } => {
                operand(&mut effect, value);
                address(&mut effect, at, MemoryDirection::Write);
                effect.memory = Some(MemoryEffect {
                    space,
                    direction: MemoryDirection::Write,
                    address: at,
                    bytes: data_type.bits() / 8,
                });
            }
            Operation::Branch { target } => effect.control = Control::Branch(target),
            Operation::Return => effect.control = Control::Return,
            Operation::Call {
                function,
                argument,
                result,
            } => {
                effect.parameter_reads.push(argument);
                effect.parameter_writes.push(result);
                effect.control = Control::Call(function);
            }
        }
        effect
    }
    pub fn requirements(&self) -> impl Iterator<Item = Requirement> {
        let body = match self.operation {
            Operation::Call { function, .. } => {
                Some(Requirement::WholeBody(Helper::for_function(function)))
            }
            _ => None,
        };
        std::iter::once(Requirement::Instruction(self.operation.primitive())).chain(body)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Library {
    ExpLibm,
    PortableMusl,
}
impl Library {
    pub fn source(self) -> &'static str {
        match self {
            Self::ExpLibm => include_str!("../math/exp.ptx"),
            Self::PortableMusl => include_str!("../math/portable_math.ptx"),
        }
    }
    pub fn identity(self) -> &'static str {
        match self {
            Self::ExpLibm => "libm-0.2.16-expf-ptx-v1",
            Self::PortableMusl => "musl-1.2.5-portable-math-ptx-v1",
        }
    }
}
/// A bundled function body with internal control, private storage and (for musl)
/// global tables. It is not one primitive instruction. Its complete cost needs a
/// whole-body contract or structured expansion before cost-based selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Helper {
    pub function: MathFunction,
    pub library: Library,
}
impl Helper {
    pub fn for_function(function: MathFunction) -> Self {
        Self {
            function,
            library: match function {
                MathFunction::Exp | MathFunction::ExpFast => Library::ExpLibm,
                MathFunction::Log | MathFunction::Sin | MathFunction::Cos => Library::PortableMusl,
            },
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requirement {
    Instruction(Primitive),
    WholeBody(Helper),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkDomain {
    pub work_items: u64,
    pub lanes_per_item: u32,
    pub dispatch: Dispatch,
    pub scratch_bytes_per_item: usize,
}
pub type Target = crate::target::PtxTarget;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetPlan {
    pub(super) target: Target,
    pub(super) registers: Vec<Register>,
    pub(super) parameters: Vec<Parameter>,
    pub(super) body: Vec<Item>,
    pub(super) libraries: Vec<Library>,
    pub(super) domain: WorkDomain,
}
impl TargetPlan {
    pub fn target(&self) -> Target {
        self.target
    }
    pub fn registers(&self) -> &[Register] {
        &self.registers
    }
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }
    pub fn body(&self) -> &[Item] {
        &self.body
    }
    pub fn libraries(&self) -> &[Library] {
        &self.libraries
    }
    pub fn domain(&self) -> WorkDomain {
        self.domain
    }
    pub fn instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.body.iter().filter_map(|item| match item {
            Item::Instruction(i) => Some(i),
            _ => None,
        })
    }
    pub fn requirements(&self) -> impl Iterator<Item = Requirement> + '_ {
        self.instructions().flat_map(Instruction::requirements)
    }
}

impl TargetPlan {
    /// Check definition/reference and control integrity before this artifact can
    /// be retained by an execution. This validates the virtual target program,
    /// not the downstream driver's physical mapping.
    pub fn validate(&self) -> Result<(), String> {
        use std::collections::BTreeSet;
        let labels = self
            .body
            .iter()
            .filter_map(|item| {
                if let Item::Label(label) = item {
                    Some(*label)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let unique = labels.iter().copied().collect::<BTreeSet<_>>();
        if labels.len() != unique.len() {
            return Err("duplicate PTX label".into());
        }
        for instruction in self.instructions() {
            let effect = instruction.effects();
            for &register in effect.register_reads.iter().chain(&effect.register_writes) {
                if register.0 >= self.registers.len() {
                    return Err("PTX instruction references an undeclared register".into());
                }
            }
            for &parameter in effect
                .parameter_reads
                .iter()
                .chain(&effect.parameter_writes)
            {
                if parameter.0 >= self.parameters.len() {
                    return Err("PTX instruction references an undeclared parameter".into());
                }
            }
            if let Some(predicate) = instruction.predicate {
                if self.registers[predicate.register.0].class != RegisterClass::Predicate {
                    return Err("PTX predicate has non-predicate register class".into());
                }
            }
            if let Control::Branch(target) = effect.control {
                if !unique.contains(&target) {
                    return Err("PTX branch target has no label".into());
                }
            }
            if let Some(memory) = effect.memory {
                match (memory.space, memory.address.base) {
                    (Space::Global, AddressBase::Register(register))
                        if self.registers[register.0].class == RegisterClass::Bits64 => {}
                    (Space::Parameter, AddressBase::Parameter(parameter)) => {
                        if memory.bytes * 8 > self.parameters[parameter.0].data_type.bits() {
                            return Err("PTX parameter access exceeds declaration".into());
                        }
                    }
                    _ => return Err("PTX address space/base mismatch".into()),
                }
            }
            match instruction.operation {
                Operation::Shuffle {
                    destination,
                    source,
                    lane,
                    ..
                } => {
                    if self.domain.lanes_per_item != 32
                        || instruction.predicate.is_some()
                        || !match lane {
                            Operand::Register(r) => {
                                self.registers[r.0].class == RegisterClass::Bits32
                            }
                            Operand::Unsigned(n) => n < 32,
                            _ => false,
                        }
                        || self.registers[destination.0].class != RegisterClass::Bits32
                        || self.registers[source.0].class != RegisterClass::Bits32
                    {
                        return Err("PTX shuffle requires a convergent full warp and 32-bit register operands".into());
                    }
                }
                Operation::Compare { destination, .. }
                    if self.registers[destination.0].class != RegisterClass::Predicate =>
                {
                    return Err("PTX compare must define a predicate".into());
                }
                Operation::Select { predicate, .. }
                    if self.registers[predicate.0].class != RegisterClass::Predicate =>
                {
                    return Err("PTX select needs a predicate".into());
                }
                Operation::Call {
                    function,
                    argument,
                    result,
                } => {
                    if !self
                        .libraries
                        .contains(&Helper::for_function(function).library)
                    {
                        return Err("PTX call lacks its whole-body implementation".into());
                    }
                    if !matches!(
                        self.parameters[argument.0].role,
                        ParameterRole::CallArgument(_)
                    ) || !matches!(self.parameters[result.0].role, ParameterRole::CallResult(_))
                    {
                        return Err(
                            "PTX helper parameter roles differ from the selected ABI".into()
                        );
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
