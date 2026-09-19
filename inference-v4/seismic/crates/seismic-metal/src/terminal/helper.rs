//! Lower the shared helper definition into the same typed terminal body used
//! for local realization, helper invocation accounting, and requirement discovery.
use super::{Expression, Statement, Type};
use seismic_lang::syntax::ast::{BinaryOp, UnaryOp};
use std::collections::BTreeMap;

fn ty(t: crate::support::Type, element: Type) -> Type {
    use crate::support::Type as T;
    match t {
        T::Bool => Type::Bool,
        T::I32 => Type::I32,
        T::U32 => Type::U32,
        T::I64 => Type::I64,
        T::U64 | T::DevicePointer | T::StatusPointer => Type::U64,
        T::Element => element,
        T::Void => Type::Bool,
    }
}

/// Single-expression helpers have no local lifetime or control to retain.
/// Callers must establish that argument evaluation is pure and total before
/// substitution can duplicate or omit it under the returned expression.
pub(crate) fn single_expression(helper: crate::support::Helper, args: &[Expression], element: Type) -> Option<Expression> {
    let definition = crate::support::Definition::new(helper);
    if definition.parameters.len() != args.len() || definition.parameters.iter().any(|(_, ty)| matches!(ty, crate::support::Type::DevicePointer | crate::support::Type::StatusPointer)) { return None; }
    let args: Vec<_> = definition.parameters.iter().zip(args).map(|((_, parameter), argument)| argument.clone().cast(ty(*parameter, element))).collect();
    let statements = body(&definition, element).ok()?;
    let [super::Site { statement: Statement::Return(Some(value)), .. }] = statements.as_slice() else { return None; };
    // Parameter names belong to the helper's scope. Simultaneous substitution
    // prevents a caller's identically named scalar from becoming another argument.
    let mut value = value.clone();
    let mut placeholders = Vec::new();
    for (index, (name, _)) in definition.parameters.iter().enumerate() {
        let placeholder = format!("$seismic_helper_argument_{index}");
        value = super::rewrite::substitute(&value, name, &Expression::variable(&placeholder, args[index].ty()), None);
        placeholders.push(placeholder);
    }
    for (placeholder, argument) in placeholders.iter().zip(&args) {
        value = super::rewrite::substitute(&value, placeholder, argument, None);
    }
    Some(value)
}

pub(crate) fn body(
    definition: &crate::support::Definition,
    element: Type,
) -> Result<Vec<crate::terminal::Site>, String> {
    use crate::support::{Binary as B, Expression as E, Statement as S};

    fn expr(e: &E, types: &BTreeMap<&str, Type>, element: Type) -> Result<Expression, String> {
        Ok(match e {
            E::Value(name) => {
                Expression::variable(*name, *types.get(name).ok_or("helper value has no type")?)
            }
            E::Integer(n) => Expression::Integer(
                *n,
                if i32::try_from(*n).is_ok() {
                    Type::I32
                } else {
                    Type::I64
                },
            ),
            E::Bool(v) => Expression::Integer(i64::from(*v), Type::Bool),
            E::Cast(t, e) => expr(e, types, element)?.cast(ty(*t, element)),
            E::Bitcast(t, e) => {
                Expression::Bitcast(ty(*t, element), Box::new(expr(e, types, element)?))
            }
            E::Negate(e) => {
                let e = expr(e, types, element)?;
                let t = e.ty();
                Expression::Unary(UnaryOp::Neg, Box::new(e), t)
            }
            E::Select(c, a, b) => Expression::Select(
                Box::new(expr(c, types, element)?),
                Box::new(expr(a, types, element)?),
                Box::new(expr(b, types, element)?),
            ),
            E::Binary(op, a, b) if matches!(op, B::And | B::Or) => Expression::ShortCircuit {
                or: *op == B::Or,
                left: Box::new(expr(a, types, element)?),
                right: Box::new(expr(b, types, element)?),
            },
            E::Binary(op, a, b) => {
                let a = expr(a, types, element)?;
                let b = expr(b, types, element)?;
                let (op, comparison) = match op {
                    B::Add => (BinaryOp::Add, false),
                    B::Subtract => (BinaryOp::Sub, false),
                    B::Divide => (BinaryOp::Div, false),
                    B::Remainder => (BinaryOp::Rem, false),
                    B::ShiftLeft => (BinaryOp::Shl, false),
                    B::ShiftRight => (BinaryOp::Shr, false),
                    B::Less => (BinaryOp::Lt, true),
                    B::GreaterEqual => (BinaryOp::Ge, true),
                    B::LessEqual => (BinaryOp::Le, true),
                    B::Equal => (BinaryOp::Eq, true),
                    B::And => (BinaryOp::And, true),
                    B::Or => (BinaryOp::Or, true),
                };
                let t = if comparison { Type::Bool } else { a.ty() };
                Expression::binary(op, a, b, t)
            }
            E::Read { pointer, index } => {
                let E::Value(name) = &**pointer else {
                    return Err("unsupported helper pointer expression".into());
                };
                Expression::Read {
                    name: (*name).into(),
                    index: Box::new(expr(index, types, element)?),
                    space: crate::terminal::Space::Device,
                    ty: element,
                }
            }
        })
    }
    fn statements(
        body: &[S],
        types: &mut BTreeMap<&'static str, Type>,
        element: Type,
        out: &mut Vec<crate::terminal::Site>,
    ) -> Result<(), String> {
        let push = |out: &mut Vec<crate::terminal::Site>, statement| {
            out.push(crate::terminal::Site {
                operation: None,
                statement,
            })
        };
        for s in body {
            match s {
                S::Let { name, ty: t, value } => {
                    let value = expr(value, types, element)?;
                    let t = ty(*t, element);
                    types.insert(name, t);
                    push(
                        out,
                        Statement::Let {
                            name: (*name).into(),
                            ty: t,
                            value,
                        },
                    );
                }
                S::Assign { name, value } => push(
                    out,
                    Statement::Assign {
                        name: (*name).into(),
                        value: expr(value, types, element)?,
                    },
                ),
                S::If { condition, body } => {
                    push(out, Statement::If(expr(condition, types, element)?));
                    statements(body, types, element, out)?;
                    push(out, Statement::End);
                }
                S::Return(value) => push(
                    out,
                    Statement::Return(
                        value
                            .as_ref()
                            .map(|v| expr(v, types, element))
                            .transpose()?,
                    ),
                ),
                S::FailureStatus => push(out, Statement::FailureStatus),
                S::Write {
                    pointer,
                    index,
                    value,
                } => {
                    let E::Value(name) = pointer else {
                        return Err("unsupported helper write pointer".into());
                    };
                    push(
                        out,
                        Statement::Write {
                            name: (*name).into(),
                            index: expr(index, types, element)?,
                            space: crate::terminal::Space::Device,
                            ty: element,
                            value: expr(value, types, element)?,
                        },
                    );
                }
            }
        }
        Ok(())
    }
    let mut types = definition
        .parameters
        .iter()
        .map(|(n, t)| (*n, ty(*t, element)))
        .collect();
    let mut out = Vec::new();
    statements(&definition.body, &mut types, element, &mut out)?;
    Ok(out)
}
