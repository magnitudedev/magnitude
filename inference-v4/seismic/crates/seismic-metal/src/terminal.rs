//! MSL operations shared by emission and accounting. Completed backend programs
//! contain only typed nodes; opaque diagnostic nodes never enter execution.
use crate::support::Helper;
use seismic_lang::{
    ast::{BinaryOp, UnaryOp},
    ir::OperationId,
    types::DType,
};
use std::fmt;
mod simplify;
pub(crate) mod synchronize;
pub(crate) mod helper;
pub(crate) mod rewrite;
mod validate;
pub mod traversal;
pub mod transfer;
pub(crate) mod family;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Type {
    Bool,
    I32,
    U32,
    I64,
    U64,
    F16,
    BF16,
    F32,
}
impl Type {
    pub fn arithmetic(a: Self, b: Self) -> Self {
        if a == b {
            return a;
        }
        use Type::*;
        match (a, b) {
            (F32, _) | (_, F32) | (F16, BF16) | (BF16, F16) => F32,
            (F16, _) | (_, F16) => F16,
            (BF16, _) | (_, BF16) => BF16,
            (U64, _) | (_, U64) => U64,
            (I64, _) | (_, I64) => I64,
            (U32, _) | (_, U32) => U32,
            _ => I32,
        }
    }
    pub fn metal(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::I64 => "long",
            Self::U64 => "ulong",
            Self::F16 => "half",
            Self::BF16 => "bfloat",
            Self::F32 => "float",
        }
    }
    pub fn bytes(self) -> u64 {
        match self {
            Self::Bool => 1,
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }
}
impl From<DType> for Type {
    fn from(t: DType) -> Self {
        match t {
            DType::Bool => Self::Bool,
            DType::I32 => Self::I32,
            DType::U32 => Self::U32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Space {
    Private,
    Threadgroup,
    Device,
    Constant,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Primitive {
    Binary {
        operation: BinaryOp,
        ty: Type,
    },
    Unary {
        operation: UnaryOp,
        ty: Type,
    },
    Cast {
        from: Type,
        to: Type,
    },
    Read {
        space: Space,
        ty: Type,
    },
    VectorRead { space: Space, ty: Type, components: u8 },
    Write {
        space: Space,
        ty: Type,
    },
    Address {
        space: Space,
        ty: Type,
    },
    Branch,
    Return,
    Select,
    Bitcast {
        from: Type,
        to: Type,
    },
    Barrier,
    Launch,
    Group,
    Builtin {
        name: String,
        inputs: Vec<Type>,
        result: Type,
    },
    /// A complete MSL subgroup matrix intrinsic, not an asserted native
    /// instruction. Its layout supplies logical bytes/FMA work structurally.
    MatrixLoad {
        layout: crate::collective::FragmentLayout,
        space: Space,
        transpose: bool,
    },
    MatrixStore {
        layout: crate::collective::FragmentLayout,
        space: Space,
    },
    MatrixMultiplyAccumulate {
        layouts: [crate::collective::FragmentLayout; 4],
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expression {
    Integer(i64, Type),
    Float(u64, Type),
    Variable(String, Type),
    VectorElement { name: String, component: u8, ty: Type },
    Parameter {
        name: String,
        ty: Type,
    },
    Binary(BinaryOp, Box<Expression>, Box<Expression>, Type),
    Unary(UnaryOp, Box<Expression>, Type),
    Cast(Type, Box<Expression>),
    Builtin(String, Vec<Expression>, Type),
    Select(Box<Expression>, Box<Expression>, Box<Expression>),
    /// Eager scalar value selection: condition and both values execute with
    /// identical participation. An unknown condition changes only the value.
    EagerSelect(Box<Expression>, Box<Expression>, Box<Expression>),
    ShortCircuit {
        or: bool,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Bitcast(Type, Box<Expression>),
    Helper(Helper, Vec<Expression>, Type),
    Read {
        name: String,
        index: Box<Expression>,
        space: Space,
        ty: Type,
    },
    /// An incomplete diagnostic node, rejected at the completed-program boundary.
    Unmapped(String, Type),
}
impl Expression {
    pub fn ty(&self) -> Type {
        match self {
            Self::Integer(_, t)
            | Self::Float(_, t)
            | Self::Variable(_, t)
            | Self::Binary(_, _, _, t)
            | Self::Unary(_, _, t)
            | Self::Cast(t, _)
            | Self::Builtin(_, _, t)
            | Self::Helper(_, _, t)
            | Self::Unmapped(_, t) => *t,
            Self::Read { ty, .. } | Self::Parameter { ty, .. } | Self::VectorElement { ty, .. } => *ty,
            Self::Select(_, a, _) | Self::EagerSelect(_, a, _) => a.ty(),
            Self::ShortCircuit { .. } => Type::Bool,
            Self::Bitcast(t, _) => *t,
        }
    }
    pub fn variable(name: impl Into<String>, ty: Type) -> Self {
        Self::Variable(name.into(), ty)
    }
    pub fn integer(n: i64) -> Self {
        Self::Integer(
            n,
            if i32::try_from(n).is_ok() {
                Type::I32
            } else {
                Type::I64
            },
        )
    }
    pub fn binary(op: BinaryOp, a: Self, b: Self, ty: Type) -> Self {
        let operand = if matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        ) {
            Type::arithmetic(a.ty(), b.ty())
        } else {
            ty
        };
        let right = if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
            b
        } else {
            b.cast(operand)
        };
        Self::Binary(op, Box::new(a.cast(operand)), Box::new(right), ty)
    }
    pub fn cast(self, ty: Type) -> Self {
        if self.ty() == ty {
            self
        } else {
            Self::Cast(ty, Box::new(self))
        }
    }
    pub fn render(&self) -> String {
        match self {
            Self::Integer(n, Type::Bool) => if *n == 0 { "false" } else { "true" }.into(),
            Self::Integer(n, t) => match t {
                Type::U32 => format!("{}u", *n as u32),
                Type::I64 => format!("{n}l"),
                Type::U64 => format!("{}ul", *n as u64),
                _ => n.to_string(),
            },
            Self::Float(bits, t) => {
                let n = f64::from_bits(*bits);
                format!(
                    "{}({})",
                    t.metal(),
                    if n.is_infinite() {
                        if n > 0.0 {
                            "INFINITY".into()
                        } else {
                            "-INFINITY".into()
                        }
                    } else {
                        format!("{n:?}f")
                    }
                )
            }
            Self::Variable(n, _) | Self::Unmapped(n, _) => n.clone(),
            Self::VectorElement { name, component, .. } => format!("{name}[{component}]"),
            Self::Parameter { name, .. } => name.clone(),
            Self::Binary(op, a, b, _) => match op {
                BinaryOp::And => format!("bool(({a}) & ({b}))"),
                BinaryOp::Or => format!("bool(({a}) | ({b}))"),
                _ => format!("({a} {} {b})", op.text()),
            },
            Self::Unary(op, x, _) => format!(
                "({}{x})",
                match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "!",
                    UnaryOp::BitNot => "~",
                }
            ),
            Self::Cast(t, x) => format!("{}({x})", t.metal()),
            Self::ShortCircuit { or, left, right } => {
                format!("({left} {} {right})", if *or { "||" } else { "&&" })
            }
            Self::Select(c, a, b) => format!("({c} ? {a} : {b})"),
            Self::EagerSelect(c, a, b) => format!("seismic_value_select({c}, {a}, {b})"),
            Self::Bitcast(t, e) => format!("as_type<{}>({e})", t.metal()),
            Self::Builtin(name, args, _) => format!(
                "{name}({})",
                args.iter().map(Self::render).collect::<Vec<_>>().join(", ")
            ),
            Self::Helper(helper, args, _) => {
                let d = crate::support::Definition::new(*helper);
                let mut args = args.iter().map(Self::render).collect::<Vec<_>>();
                if d.parameters
                    .last()
                    .is_some_and(|(_, t)| *t == crate::support::Type::StatusPointer)
                {
                    args.push("seismic_status".into());
                }
                format!("{}({})", d.name, args.join(", "))
            }
            Self::Read { name, index, .. } => format!("{name}[{index}]"),
        }
    }
}
/// By-value arguments establish eager evaluation for every admitted scalar
/// type, including bool and bfloat, without relying on Metal select overloads.
pub(crate) const VALUE_SELECTION_SUPPORT: &str =
    "template<typename T> inline T seismic_value_select(bool condition, T yes, T no) { return condition ? yes : no; }\n\n";
impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Statement {
    Let {
        name: String,
        ty: Type,
        value: Expression,
    },
    Assign {
        name: String,
        value: Expression,
    },
    Array {
        name: String,
        ty: Type,
        elements: u64,
    },
    Pointer {
        name: String,
        base: String,
        index: Expression,
        space: Space,
        ty: Type,
    },
    VectorRead { name: String, base: String, index: Expression, ty: Type, components: u8 },
    Write {
        name: String,
        index: Expression,
        space: Space,
        ty: Type,
        value: Expression,
    },
    Evaluate(Expression),
    For {
        name: String,
        start: Expression,
        end: Expression,
        step: i64,
    },
    If(Expression),
    Else,
    Scope,
    End,
    ReturnIf(Expression),
    Return(Option<Expression>),
    FailureStatus,
    Barrier,
    Fragment {
        name: String,
        layout: crate::collective::FragmentLayout,
    },
    MatrixLoad {
        fragment: String,
        layout: crate::collective::FragmentLayout,
        base: String,
        offset: Expression,
        leading: Expression,
        space: Space,
        transpose: bool,
    },
    MatrixStore {
        fragment: String,
        layout: crate::collective::FragmentLayout,
        base: String,
        offset: Expression,
        leading: Expression,
        space: Space,
    },
    MatrixMultiplyAccumulate {
        fragments: [String; 4],
        layouts: [crate::collective::FragmentLayout; 4],
    },
    Unmapped(String),
}
impl Statement {
    pub(crate) fn realized(self) -> Self {
        simplify::statement(self)
    }
    pub fn render(&self) -> String {
        match self {
            Self::Let { name, ty, value } => format!("{} {name} = {value};", ty.metal()),
            Self::Assign { name, value } => format!("{name} = {value};"),
            Self::Pointer {
                name,
                base,
                index,
                space,
                ty,
            } => format!(
                "{} {}* {name} = &{base}[{index}];",
                match space {
                    Space::Private => "thread",
                    Space::Threadgroup => "threadgroup",
                    Space::Device => "device",
                    Space::Constant => "constant",
                },
                ty.metal()
            ),
            Self::Array { name, ty, elements } => format!("{} {name}[{elements}];", ty.metal()),
            Self::VectorRead { name, base, index, ty, components } => format!("packed_{}{components} {name} = *reinterpret_cast<const device packed_{}{components}*>({base} + ({index}));", ty.metal(), ty.metal()),
            Self::Write {
                name, index, value, ..
            } => format!("{name}[{index}] = {value};"),
            Self::Evaluate(e) => format!("{e};"),
            Self::For {
                name,
                start,
                end,
                step,
            } => format!("for (int {name} = {start}; {name} < {end}; {name} += {step}) {{"),
            Self::If(e) => format!("if ({e}) {{"),
            Self::Scope => "{".into(),
            Self::Else => "} else {".into(),
            Self::End => "}".into(),
            Self::ReturnIf(e) => format!("if ({e}) return;"),
            Self::Return(value) => match value {
                Some(e) => format!("return {e};"),
                None => "return;".into(),
            },
            Self::FailureStatus => {
                "atomic_store_explicit(seismic_status, 1u, memory_order_relaxed);".into()
            }
            Self::Barrier => "simdgroup_barrier(mem_flags::mem_threadgroup);".into(),
            Self::Fragment { name, layout } => format!(
                "simdgroup_matrix<{}, {}, {}> {name};",
                Type::from(layout.dtype).metal(),
                layout.rows,
                layout.columns
            ),
            Self::MatrixLoad {
                fragment,
                base,
                offset,
                leading,
                transpose,
                ..
            } => format!(
                "simdgroup_load({fragment}, {base} + ({offset}), {leading}, ulong2(0, 0), {transpose});"
            ),
            Self::MatrixStore {
                fragment,
                base,
                offset,
                leading,
                ..
            } => format!("simdgroup_store({fragment}, {base} + ({offset}), {leading});"),
            Self::MatrixMultiplyAccumulate { fragments, .. } => format!(
                "simdgroup_multiply_accumulate({}, {}, {}, {});",
                fragments[0], fragments[1], fragments[2], fragments[3]
            ),
            Self::Unmapped(text) => text.clone(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub operation: Option<OperationId>,
    pub statement: Statement,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    pub(crate) launches: Vec<Vec<Site>>,
}
impl Program {
    /// A completed backend program must describe every executable operation.
    /// Missing hardware analysis is a separate accounting result, never opaque text.
    pub fn validate_typed(&self) -> Result<(), String> {
        validate::program(self)
    }
    pub(crate) fn validate_template(&self) -> Result<(), String> {
        validate::template(self)
    }
    pub(crate) fn realize_launch(&mut self, index: usize) {
        simplify::launch(&mut self.launches[index]);
    }
    pub fn launches(&self) -> &[Vec<Site>] {
        &self.launches
    }
}
