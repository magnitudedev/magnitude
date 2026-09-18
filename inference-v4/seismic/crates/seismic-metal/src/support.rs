//! Structured implementation of compiler-introduced validity and scalar helpers.
//! These are operation graphs, not timing constants. A backend renderer and its
//! resource mapping consume the same definitions, including failure-path atomics.
use seismic_lang::types::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Helper {
    Index,
    ShiftSigned,
    ShiftUnsigned,
    DivideSigned,
    DivideUnsigned,
    Read,
    Write,
    SliceValid,
    SliceStart,
    SliceExtent,
    WorkItemLive,
    Validate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    Bool,
    I32,
    U32,
    I64,
    U64,
    Element,
    DevicePointer,
    StatusPointer,
    Void,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Subtract,
    Divide,
    Remainder,
    ShiftLeft,
    ShiftRight,
    Less,
    GreaterEqual,
    LessEqual,
    Equal,
    And,
    Or,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expression {
    Value(&'static str),
    Integer(i64),
    Bool(bool),
    Cast(Type, Box<Expression>),
    Bitcast(Type, Box<Expression>),
    Binary(Binary, Box<Expression>, Box<Expression>),
    Negate(Box<Expression>),
    Select(Box<Expression>, Box<Expression>, Box<Expression>),
    Read {
        pointer: Box<Expression>,
        index: Box<Expression>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Statement {
    Let {
        name: &'static str,
        ty: Type,
        value: Expression,
    },
    Assign {
        name: &'static str,
        value: Expression,
    },
    If {
        condition: Expression,
        body: Vec<Statement>,
    },
    /// Relaxed device status store; never a missing zero-cost failure path.
    FailureStatus,
    Return(Option<Expression>),
    Write {
        pointer: Expression,
        index: Expression,
        value: Expression,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Definition {
    pub helper: Helper,
    pub name: &'static str,
    pub result: Type,
    pub parameters: Vec<(&'static str, Type)>,
    pub generic_element: bool,
    pub body: Vec<Statement>,
}
#[derive(Clone, Debug)]
pub struct Plan {
    definitions: Vec<Definition>,
}
impl Plan {
    pub fn new() -> Self {
        Self {
            definitions: [
                Helper::Index,
                Helper::ShiftSigned,
                Helper::ShiftUnsigned,
                Helper::DivideSigned,
                Helper::DivideUnsigned,
                Helper::Read,
                Helper::Write,
                Helper::SliceValid,
                Helper::SliceStart,
                Helper::SliceExtent,
                Helper::WorkItemLive,
                Helper::Validate,
            ]
            .into_iter()
            .map(Definition::new)
            .collect(),
        }
    }
    pub fn definitions(&self) -> &[Definition] {
        &self.definitions
    }
    pub fn get(&self, helper: Helper) -> &Definition {
        self.definitions
            .iter()
            .find(|d| d.helper == helper)
            .expect("closed helper vocabulary is exhaustive")
    }
}
impl Default for Plan {
    fn default() -> Self {
        Self::new()
    }
}
fn v(name: &'static str) -> Expression {
    Expression::Value(name)
}
fn n(value: i64) -> Expression {
    Expression::Integer(value)
}
fn cast(ty: Type, x: Expression) -> Expression {
    Expression::Cast(ty, Box::new(x))
}
fn bitcast(ty: Type, x: Expression) -> Expression {
    Expression::Bitcast(ty, Box::new(x))
}
fn binary(op: Binary, a: Expression, b: Expression) -> Expression {
    Expression::Binary(op, Box::new(a), Box::new(b))
}
fn select(c: Expression, a: Expression, b: Expression) -> Expression {
    Expression::Select(Box::new(c), Box::new(a), Box::new(b))
}
fn ret(value: Expression) -> Statement {
    Statement::Return(Some(value))
}
fn let_(name: &'static str, ty: Type, value: Expression) -> Statement {
    Statement::Let { name, ty, value }
}
fn failure(condition: Expression, result: Type) -> Statement {
    Statement::If {
        condition,
        body: vec![
            Statement::FailureStatus,
            Statement::Return(if result == Type::Void {
                None
            } else {
                Some(cast(result, n(0)))
            }),
        ],
    }
}
impl Definition {
    fn new(helper: Helper) -> Self {
        use Binary::*;
        use Type::*;
        let status = ("status", StatusPointer);
        let (name, result, parameters, generic_element, body) = match helper {
            Helper::Index => (
                "seismic_index",
                I64,
                vec![("index", I64), ("extent", I64), status],
                false,
                vec![
                    failure(
                        binary(
                            Or,
                            binary(Less, v("index"), n(0)),
                            binary(GreaterEqual, v("index"), v("extent")),
                        ),
                        I64,
                    ),
                    ret(v("index")),
                ],
            ),
            Helper::ShiftSigned | Helper::ShiftUnsigned => {
                let signed = helper == Helper::ShiftSigned;
                let ty = if signed { I32 } else { U32 };
                let left = binary(
                    ShiftLeft,
                    if signed { bitcast(U32, v("a")) } else { v("a") },
                    cast(U32, v("b")),
                );
                let left = if signed { bitcast(I32, left) } else { left };
                (
                    "seismic_shift",
                    ty,
                    vec![("a", ty), ("b", I64), ("left", Bool), status],
                    false,
                    vec![
                        failure(
                            binary(
                                Or,
                                binary(Less, v("b"), n(0)),
                                binary(GreaterEqual, v("b"), n(32)),
                            ),
                            ty,
                        ),
                        ret(select(
                            v("left"),
                            left,
                            binary(ShiftRight, v("a"), cast(U32, v("b"))),
                        )),
                    ],
                )
            }
            Helper::DivideUnsigned => (
                "seismic_integer_division",
                U32,
                vec![("a", U32), ("b", U32), ("remainder", Bool), status],
                false,
                vec![
                    failure(binary(Equal, v("b"), n(0)), U32),
                    ret(select(
                        v("remainder"),
                        binary(Remainder, v("a"), v("b")),
                        binary(Divide, v("a"), v("b")),
                    )),
                ],
            ),
            Helper::DivideSigned => {
                let overflow = binary(
                    And,
                    binary(Equal, v("a"), n(i64::from(i32::MIN))),
                    binary(Equal, v("b"), n(-1)),
                );
                (
                    "seismic_integer_division",
                    I32,
                    vec![("a", I32), ("b", I32), ("remainder", Bool), status],
                    false,
                    vec![
                        failure(binary(Or, binary(Equal, v("b"), n(0)), overflow), I32),
                        let_(
                            "q",
                            I64,
                            binary(Divide, cast(I64, v("a")), cast(I64, v("b"))),
                        ),
                        let_(
                            "r",
                            I64,
                            binary(Remainder, cast(I64, v("a")), cast(I64, v("b"))),
                        ),
                        Statement::If {
                            condition: binary(Less, v("r"), n(0)),
                            body: vec![
                                Statement::Assign {
                                    name: "q",
                                    value: binary(
                                        Add,
                                        v("q"),
                                        select(binary(Less, v("b"), n(0)), n(1), n(-1)),
                                    ),
                                },
                                Statement::Assign {
                                    name: "r",
                                    value: binary(
                                        Add,
                                        v("r"),
                                        select(
                                            binary(Less, v("b"), n(0)),
                                            Expression::Negate(Box::new(cast(I64, v("b")))),
                                            cast(I64, v("b")),
                                        ),
                                    ),
                                },
                            ],
                        },
                        ret(cast(I32, select(v("remainder"), v("r"), v("q")))),
                    ],
                )
            }
            Helper::Read | Helper::Write => {
                let read = helper == Helper::Read;
                let result = if read { Element } else { Void };
                let mut parameters =
                    vec![("pointer", DevicePointer), ("index", I64), ("count", U64)];
                if !read {
                    parameters.push(("value", Element));
                }
                parameters.push(status);
                let mut body = vec![failure(
                    binary(
                        Or,
                        binary(Less, v("index"), n(0)),
                        binary(GreaterEqual, cast(U64, v("index")), v("count")),
                    ),
                    result,
                )];
                body.push(if read {
                    ret(Expression::Read {
                        pointer: Box::new(v("pointer")),
                        index: Box::new(v("index")),
                    })
                } else {
                    Statement::Write {
                        pointer: v("pointer"),
                        index: v("index"),
                        value: v("value"),
                    }
                });
                (
                    if read {
                        "seismic_read"
                    } else {
                        "seismic_write"
                    },
                    result,
                    parameters,
                    true,
                    body,
                )
            }
            Helper::SliceValid => (
                "seismic_slice_valid",
                Bool,
                vec![("start", I64), ("end", I64), ("extent", I64)],
                false,
                vec![ret(binary(
                    And,
                    binary(
                        And,
                        binary(GreaterEqual, v("start"), n(0)),
                        binary(LessEqual, v("start"), v("end")),
                    ),
                    binary(LessEqual, v("end"), v("extent")),
                ))],
            ),
            Helper::SliceStart => (
                "seismic_slice_start",
                I64,
                vec![("valid", Bool), ("start", I64), status],
                false,
                vec![
                    failure(binary(Equal, v("valid"), Expression::Bool(false)), I64),
                    ret(v("start")),
                ],
            ),
            Helper::SliceExtent => (
                "seismic_slice_extent",
                I32,
                vec![("valid", Bool), ("start", I64), ("end", I64)],
                false,
                vec![ret(select(
                    v("valid"),
                    cast(I32, binary(Subtract, v("end"), v("start"))),
                    n(0),
                ))],
            ),
            Helper::WorkItemLive => (
                "seismic_work_item_live",
                Bool,
                vec![("item", U32), ("extent", U32)],
                false,
                vec![ret(binary(Less, v("item"), v("extent")))],
            ),
            Helper::Validate => (
                "seismic_validate",
                Void,
                vec![("valid", Bool), status],
                false,
                vec![Statement::If {
                    condition: binary(Equal, v("valid"), Expression::Bool(false)),
                    body: vec![Statement::FailureStatus],
                }],
            ),
        };
        Self {
            helper,
            name,
            result,
            parameters,
            generic_element,
            body,
        }
    }
}
/// Generic helper instantiations carry their actual element precision in model
/// requests; `Element` alone is never a hardware instruction width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invocation {
    pub helper: Helper,
    pub element: Option<DType>,
}

/// Exact unsigned arithmetic used to enter a selected Metal work item. These
/// operations are shared source-level requests, not a native instruction count:
/// target mappings may implement constant division with different instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchValue {
    Group,
    Subgroup,
    Constant(u32),
    Result(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchOperation {
    Add(LaunchValue, LaunchValue),
    Multiply(LaunchValue, LaunchValue),
    Divide(LaunchValue, LaunchValue),
    Remainder(LaunchValue, LaunchValue),
    SignedIndex(LaunchValue),
    Live(LaunchValue, LaunchValue),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchStep {
    pub operation: LaunchOperation,
    /// The guard itself and its dependencies execute on padding lanes too.
    pub participating_only: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchProgram {
    pub steps: Vec<LaunchStep>,
    pub guard: usize,
    pub item: LaunchValue,
    pub part: Option<LaunchValue>,
    pub coordinates: Vec<LaunchValue>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchRecipe {
    mapping: seismic_realization::dispatch::WorkMapping,
    parts: u64,
}
impl LaunchRecipe {
    pub fn new(
        mapping: seismic_realization::dispatch::WorkMapping,
        parts: u64,
    ) -> Result<Self, String> {
        if parts == 0 {
            return Err("launch part count must be positive".into());
        }
        Ok(Self { mapping, parts })
    }
    pub fn instantiate(
        &self,
        dispatch: &seismic_realization::dispatch::GroupDispatch,
    ) -> Result<LaunchProgram, String> {
        use LaunchOperation as O;
        use LaunchValue as V;
        if self.mapping.work_items().checked_mul(self.parts) != Some(dispatch.work_items) {
            return Err("launch recipe and dispatch describe different work domains".into());
        }
        let uint = |n| {
            u32::try_from(n)
                .map(V::Constant)
                .map_err(|_| "Metal launch coordinate exceeds uint range".to_string())
        };
        // All intermediate slot values, including padding, must be representable.
        if dispatch
            .groups
            .checked_mul(dispatch.items_per_group)
            .is_none_or(|n| n > u64::from(u32::MAX))
        {
            return Err("Metal launch slot domain exceeds uint range".into());
        }
        let mut steps = Vec::new();
        let mut push = |operation, participating_only| {
            let result = V::Result(steps.len());
            steps.push(LaunchStep {
                operation,
                participating_only,
            });
            result
        };
        let base = push(
            O::Multiply(V::Group, uint(dispatch.items_per_group)?),
            false,
        );
        let slot = push(O::Add(base, V::Subgroup), false);
        let V::Result(guard) = push(O::Live(slot, uint(dispatch.work_items)?), false) else {
            unreachable!()
        };
        let (item, part) = if self.parts == 1 {
            (slot, None)
        } else {
            let part = push(O::Remainder(slot, uint(self.parts)?), true);
            let part = push(O::SignedIndex(part), true);
            (push(O::Divide(slot, uint(self.parts)?), true), Some(part))
        };
        let mut coordinates = Vec::new();
        for axis in self.mapping.axes() {
            if self.mapping.work_items() == 0 {
                coordinates.push(V::Constant(0));
                continue;
            }
            if (axis.extent - 1)
                .checked_mul(axis.step)
                .is_none_or(|n| n > i32::MAX as u64)
            {
                return Err("Metal work coordinate exceeds signed index range".into());
            }
            let divided = push(O::Divide(item, uint(axis.stride)?), true);
            let remainder = push(O::Remainder(divided, uint(axis.extent)?), true);
            let scaled = push(O::Multiply(remainder, uint(axis.step)?), true);
            coordinates.push(push(O::SignedIndex(scaled), true));
        }
        Ok(LaunchProgram {
            steps,
            guard,
            item,
            part,
            coordinates,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_realization::dispatch::{GroupDispatch, WorkMapping};
    fn evaluate(
        program: &LaunchProgram,
        group: u32,
        subgroup: u32,
    ) -> Option<(u32, Option<u32>, Vec<u32>)> {
        use LaunchOperation as O;
        let mut results = Vec::new();
        let get = |v: LaunchValue, results: &Vec<u32>| match v {
            LaunchValue::Group => group,
            LaunchValue::Subgroup => subgroup,
            LaunchValue::Constant(n) => n,
            LaunchValue::Result(n) => results[n],
        };
        for (n, step) in program.steps.iter().enumerate() {
            let value = match step.operation {
                O::Add(a, b) => get(a, &results) + get(b, &results),
                O::Multiply(a, b) => get(a, &results) * get(b, &results),
                O::Divide(a, b) => get(a, &results) / get(b, &results),
                O::Remainder(a, b) => get(a, &results) % get(b, &results),
                O::SignedIndex(a) => get(a, &results),
                O::Live(a, b) => u32::from(get(a, &results) < get(b, &results)),
            };
            results.push(value);
            if n == program.guard && value == 0 {
                return None;
            }
        }
        Some((
            get(program.item, &results),
            program.part.map(|p| get(p, &results)),
            program
                .coordinates
                .iter()
                .map(|c| get(*c, &results))
                .collect(),
        ))
    }
    #[test]
    fn launch_graph_covers_split_coordinates_once_and_excludes_padding() {
        let mapping = WorkMapping::new(&[4, 6], &[2, 2]).unwrap();
        let recipe = LaunchRecipe::new(mapping, 3).unwrap();
        let dispatch = GroupDispatch::new(18, 32, 4).unwrap();
        let program = recipe.instantiate(&dispatch).unwrap();
        let mut actual = Vec::new();
        for group in 0..5 {
            for subgroup in 0..4 {
                if let Some(value) = evaluate(&program, group, subgroup) {
                    actual.push(value);
                }
            }
        }
        let expected: Vec<_> = (0..2)
            .flat_map(|row| {
                (0..3).flat_map(move |column| {
                    (0..3)
                        .map(move |part| (row * 3 + column, Some(part), vec![row * 2, column * 2]))
                })
            })
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(
            program
                .steps
                .iter()
                .filter(|s| !s.participating_only)
                .count(),
            3
        );
        assert!(recipe
            .instantiate(&GroupDispatch::new(19, 32, 4).unwrap())
            .is_err());
    }
    #[test]
    fn zero_domain_never_evaluates_axis_division_and_large_coordinates_reject() {
        let recipe = LaunchRecipe::new(WorkMapping::new(&[0, 6], &[1, 1]).unwrap(), 1).unwrap();
        let program = recipe
            .instantiate(&GroupDispatch::new(0, 32, 4).unwrap())
            .unwrap();
        assert_eq!(evaluate(&program, 0, 0), None);
        assert!(!program.steps.iter().any(|s| matches!(
            s.operation,
            LaunchOperation::Divide(..) | LaunchOperation::Remainder(..)
        )));
        let recipe =
            LaunchRecipe::new(WorkMapping::new(&[u32::MAX as u64], &[1]).unwrap(), 1).unwrap();
        assert!(recipe
            .instantiate(&GroupDispatch::new(u32::MAX as u64, 32, 4).unwrap())
            .is_err());
    }
}

// MSL rendering of the same typed helper bodies exposed to resource modeling.

fn ty(ty: Type) -> &'static str {
    match ty {
        Type::Bool => "bool",
        Type::I32 => "int",
        Type::U32 => "uint",
        Type::I64 => "long",
        Type::U64 => "ulong",
        Type::Element => "T",
        Type::DevicePointer => "device T*",
        Type::StatusPointer => "device atomic_uint*",
        Type::Void => "void",
    }
}
fn expression(e: &Expression) -> String {
    match e {
        Expression::Value(name) => (*name).into(),
        Expression::Integer(n) => {
            if *n == i64::from(i32::MIN) {
                "(-2147483647 - 1)".into()
            } else {
                n.to_string()
            }
        }
        Expression::Bool(b) => b.to_string(),
        Expression::Cast(t, e) => format!("{}({})", ty(*t), expression(e)),
        Expression::Bitcast(t, e) => format!("as_type<{}>({})", ty(*t), expression(e)),
        Expression::Negate(e) => format!("(-{})", expression(e)),
        Expression::Select(c, a, b) => format!(
            "({} ? {} : {})",
            expression(c),
            expression(a),
            expression(b)
        ),
        Expression::Read { pointer, index } => {
            format!("{}[{}]", expression(pointer), expression(index))
        }
        Expression::Binary(op, a, b) => format!(
            "({} {} {})",
            expression(a),
            match op {
                Binary::Add => "+",
                Binary::Subtract => "-",
                Binary::Divide => "/",
                Binary::Remainder => "%",
                Binary::ShiftLeft => "<<",
                Binary::ShiftRight => ">>",
                Binary::Less => "<",
                Binary::GreaterEqual => ">=",
                Binary::LessEqual => "<=",
                Binary::Equal => "==",
                Binary::And => "&&",
                Binary::Or => "||",
            },
            expression(b)
        ),
    }
}
fn body(stmts: &[Statement], indent: usize, out: &mut String) {
    for stmt in stmts {
        out.push_str(&"    ".repeat(indent));
        match stmt {
            Statement::Let { name, ty: t, value } => {
                out.push_str(&format!("{} {name} = {};\n", ty(*t), expression(value)))
            }
            Statement::Assign { name, value } => {
                out.push_str(&format!("{name} = {};\n", expression(value)))
            }
            Statement::Return(value) => out.push_str(&match value {
                Some(v) => format!("return {};\n", expression(v)),
                None => "return;\n".into(),
            }),
            Statement::FailureStatus => {
                out.push_str("atomic_store_explicit(status, 1u, memory_order_relaxed);\n")
            }
            Statement::Write {
                pointer,
                index,
                value,
            } => out.push_str(&format!(
                "{}[{}] = {};\n",
                expression(pointer),
                expression(index),
                expression(value)
            )),
            Statement::If {
                condition,
                body: then,
            } => {
                out.push_str(&format!("if ({}) {{\n", expression(condition)));
                body(then, indent + 1, out);
                out.push_str(&"    ".repeat(indent));
                out.push_str("}\n");
            }
        }
    }
}
fn definition(d: &Definition, out: &mut String) {
    if d.generic_element {
        out.push_str("template<typename T> ");
    }
    out.push_str(&format!(
        "inline {} {}({}) {{\n",
        ty(d.result),
        d.name,
        d.parameters
            .iter()
            .map(|(name, t)| {
                if *t == Type::DevicePointer && d.helper == crate::support::Helper::Read {
                    format!("device const T* {name}")
                } else {
                    format!("{} {name}", ty(*t))
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    ));
    body(&d.body, 1, out);
    out.push_str("}\n");
}
pub(crate) fn render(plan: &Plan) -> String {
    let mut out = String::new();
    for d in plan.definitions() {
        definition(d, &mut out);
    }
    out
}
