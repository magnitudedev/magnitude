//! Structural transfer of nodes between arenas.
//!
//! A definition's arena is private to it; a call binds the callee's
//! dimensions to caller expressions, and the entry builder rebuilds every
//! reachable definition in the entry's arena. Both walk the frozen
//! [`NodeView`] projection of the source node and rebuild it through the
//! destination arena's constructors, mapping each symbol through `map`. The
//! transfer is total over every node kind an arena can hold.

use crate::expr::{
    AnyExpr, BinaryOp, BoolExpr, CmpOp, DurationExpr, DurationTerm, ExprArena, IntExpr, NaryOp,
    NatExpr, NodeView, SymbolId, UnaryOp,
};

/// Maps a source symbol to an expression of the same sort in the destination.
pub(crate) type SymbolMap<'m> = dyn FnMut(SymbolId, &mut ExprArena) -> AnyExpr + 'm;

pub(crate) fn transfer_int(
    src: &ExprArena,
    node: IntExpr,
    dst: &mut ExprArena,
    map: &mut SymbolMap<'_>,
) -> IntExpr {
    int_of(src, AnyExpr::Int(node), dst, map)
}

fn int_of(src: &ExprArena, node: AnyExpr, dst: &mut ExprArena, map: &mut SymbolMap<'_>) -> IntExpr {
    match transfer(src, node, dst, map) {
        AnyExpr::Int(e) => e,
        _ => panic!("expression transfer changed an Int expression's sort"),
    }
}

pub(crate) fn transfer_nat(
    src: &ExprArena,
    node: NatExpr,
    dst: &mut ExprArena,
    map: &mut SymbolMap<'_>,
) -> NatExpr {
    nat_of(src, AnyExpr::Nat(node), dst, map)
}

fn nat_of(src: &ExprArena, node: AnyExpr, dst: &mut ExprArena, map: &mut SymbolMap<'_>) -> NatExpr {
    match transfer(src, node, dst, map) {
        AnyExpr::Nat(n) => n,
        _ => panic!("expression transfer changed a Nat expression's sort"),
    }
}

pub(crate) fn transfer_bool(
    src: &ExprArena,
    node: BoolExpr,
    dst: &mut ExprArena,
    map: &mut SymbolMap<'_>,
) -> BoolExpr {
    bool_of(src, AnyExpr::Bool(node), dst, map)
}

fn bool_of(
    src: &ExprArena,
    node: AnyExpr,
    dst: &mut ExprArena,
    map: &mut SymbolMap<'_>,
) -> BoolExpr {
    match transfer(src, node, dst, map) {
        AnyExpr::Bool(b) => b,
        _ => panic!("expression transfer changed a Bool expression's sort"),
    }
}

fn duration_of(
    src: &ExprArena,
    node: AnyExpr,
    dst: &mut ExprArena,
    map: &mut SymbolMap<'_>,
) -> DurationExpr {
    match transfer(src, node, dst, map) {
        AnyExpr::Duration(duration) => duration,
        _ => panic!("expression transfer changed a Duration expression's sort"),
    }
}

/// Rebuilds `node` in `dst`. The result has the sort of the source node
/// except for symbols, whose sort is whatever `map` supplies.
pub(crate) fn transfer(
    src: &ExprArena,
    node: AnyExpr,
    dst: &mut ExprArena,
    map: &mut SymbolMap<'_>,
) -> AnyExpr {
    match src.view(node) {
        NodeView::NatConst(c) => AnyExpr::Nat(dst.nat(c)),
        NodeView::IntConst(c) => AnyExpr::Int(dst.int(c)),
        NodeView::BoolConst(b) => AnyExpr::Bool(dst.bool(b)),
        NodeView::ScalarConst { dtype, bits } => AnyExpr::Scalar(match dtype {
            crate::types::DType::F32 => dst
                .scalar_const::<crate::expr::F32>(f32::from_bits(bits))
                .erase(),
            crate::types::DType::F16 => dst.scalar_const::<crate::expr::F16>(bits as u16).erase(),
            crate::types::DType::BF16 => dst.scalar_const::<crate::expr::BF16>(bits as u16).erase(),
            crate::types::DType::I32 => dst.scalar_const::<crate::expr::I32>(bits as i32).erase(),
            crate::types::DType::U32 => dst.scalar_const::<crate::expr::U32>(bits).erase(),
            crate::types::DType::Bool => dst
                .scalar_const::<crate::expr::BoolScalar>(bits != 0)
                .erase(),
        }),
        NodeView::Symbol(s) => map(s, dst),
        NodeView::Unary { op, operand } => match op {
            UnaryOp::Not => {
                let operand = bool_of(src, operand, dst, map);
                AnyExpr::Bool(dst.not(operand))
            }
            UnaryOp::NatFromInt => {
                let operand = int_of(src, operand, dst, map);
                AnyExpr::Nat(dst.nat_from_int(operand))
            }
            UnaryOp::IntFromNat => {
                let operand = nat_of(src, operand, dst, map);
                AnyExpr::Int(dst.int_from_nat(operand))
            }
        },
        NodeView::Binary { op, lhs, rhs } => match (node, op) {
            (_, BinaryOp::And | BinaryOp::Or | BinaryOp::Implies | BinaryOp::Iff) => {
                let l = bool_of(src, lhs, dst, map);
                let r = bool_of(src, rhs, dst, map);
                AnyExpr::Bool(match op {
                    BinaryOp::And => dst.and(l, r),
                    BinaryOp::Or => dst.or(l, r),
                    BinaryOp::Implies => dst.implies(l, r),
                    _ => dst.iff(l, r),
                })
            }
            (AnyExpr::Int(_), _) => {
                let l = int_of(src, lhs, dst, map);
                let r = int_of(src, rhs, dst, map);
                AnyExpr::Int(match op {
                    BinaryOp::Add => dst.int_add(l, r),
                    BinaryOp::Sub => dst.int_sub(l, r),
                    BinaryOp::Mul => dst.int_mul(l, r),
                    BinaryOp::Div => dst.int_div(l, r),
                    BinaryOp::Rem => dst.int_rem(l, r),
                    BinaryOp::Min => dst.int_min(l, r),
                    BinaryOp::Max => dst.int_max(l, r),
                    // Ceiling division and alignment have no `Int` constructor;
                    // they never arise on integer nodes.
                    BinaryOp::CeilDiv => {
                        let one = dst.int(1);
                        let sum = dst.int_add(l, r);
                        let numerator = dst.int_sub(sum, one);
                        dst.int_div(numerator, r)
                    }
                    BinaryOp::AlignUp => {
                        let one = dst.int(1);
                        let sum = dst.int_add(l, r);
                        let numerator = dst.int_sub(sum, one);
                        let q = dst.int_div(numerator, r);
                        dst.int_mul(q, r)
                    }
                    BinaryOp::And | BinaryOp::Or | BinaryOp::Implies | BinaryOp::Iff => {
                        panic!("boolean operator appeared in an integer expression")
                    }
                })
            }
            (_, _) => {
                let l = nat_of(src, lhs, dst, map);
                let r = nat_of(src, rhs, dst, map);
                AnyExpr::Nat(match op {
                    BinaryOp::Add => dst.nat_add(l, r),
                    BinaryOp::Sub => dst.nat_sub(l, r),
                    BinaryOp::Mul => dst.nat_mul(l, r),
                    BinaryOp::Div => dst.nat_div(l, r),
                    BinaryOp::CeilDiv => dst.nat_ceil_div(l, r),
                    BinaryOp::Rem => dst.nat_rem(l, r),
                    BinaryOp::Min => dst.nat_min(l, r),
                    BinaryOp::Max => dst.nat_max(l, r),
                    BinaryOp::AlignUp => dst.nat_align_up(l, r),
                    BinaryOp::And | BinaryOp::Or | BinaryOp::Implies | BinaryOp::Iff => {
                        panic!("boolean operator appeared in a natural expression")
                    }
                })
            }
        },
        NodeView::Nary { op, operands } => match op {
            NaryOp::All | NaryOp::Any => {
                let terms: Vec<BoolExpr> = operands
                    .iter()
                    .map(|o| bool_of(src, *o, dst, map))
                    .collect();
                AnyExpr::Bool(if op == NaryOp::All {
                    dst.all(&terms)
                } else {
                    dst.any(&terms)
                })
            }
            NaryOp::Product => {
                let factors: Vec<NatExpr> =
                    operands.iter().map(|o| nat_of(src, *o, dst, map)).collect();
                AnyExpr::Nat(dst.nat_product(&factors))
            }
            NaryOp::DurationAdd => {
                let durations: Vec<DurationExpr> = operands
                    .iter()
                    .map(|operand| duration_of(src, *operand, dst, map))
                    .collect();
                let mut acc = dst.duration(&[]);
                for duration in durations {
                    acc = dst.duration_add(acc, duration);
                }
                AnyExpr::Duration(acc)
            }
        },
        NodeView::Select {
            cond,
            then,
            otherwise,
        } => {
            let cond = bool_of(src, AnyExpr::Bool(cond), dst, map);
            match node {
                AnyExpr::Int(_) => {
                    let t = int_of(src, then, dst, map);
                    let e = int_of(src, otherwise, dst, map);
                    AnyExpr::Int(dst.int_select(cond, t, e))
                }
                AnyExpr::Duration(_) => {
                    let t = duration_of(src, then, dst, map);
                    let e = duration_of(src, otherwise, dst, map);
                    AnyExpr::Duration(dst.duration_select(cond, t, e))
                }
                _ => {
                    let t = nat_of(src, then, dst, map);
                    let e = nat_of(src, otherwise, dst, map);
                    AnyExpr::Nat(dst.nat_select(cond, t, e))
                }
            }
        }
        NodeView::Cmp { op, lhs, rhs } => AnyExpr::Bool(match lhs {
            AnyExpr::Int(_) => {
                let l = int_of(src, lhs, dst, map);
                let r = int_of(src, rhs, dst, map);
                dst.int_cmp(op, l, r)
            }
            _ => {
                let l = nat_of(src, lhs, dst, map);
                let r = nat_of(src, rhs, dst, map);
                dst.nat_cmp(op, l, r)
            }
        }),
        NodeView::In { operand, values } => AnyExpr::Bool(match operand {
            AnyExpr::Nat(n) => {
                let n = nat_of(src, AnyExpr::Nat(n), dst, map);
                let values: Vec<u64> = values
                    .iter()
                    .filter_map(|v| u64::try_from(*v).ok())
                    .collect();
                dst.nat_in(n, &values)
            }
            _ => {
                let e = int_of(src, operand, dst, map);
                let terms: Vec<BoolExpr> = values
                    .iter()
                    .map(|v| {
                        let c = dst.int(*v);
                        dst.int_cmp(CmpOp::Eq, e, c)
                    })
                    .collect();
                dst.any(&terms)
            }
        }),
        NodeView::Fold {
            op,
            binder,
            start,
            extent,
            body,
        } => {
            let start = nat_of(src, AnyExpr::Nat(start), dst, map);
            let extent = nat_of(src, AnyExpr::Nat(extent), dst, map);
            let body = nat_of(src, AnyExpr::Nat(body), dst, map);
            AnyExpr::Nat(dst.nat_fold_range(op, binder, start, extent, body))
        }
        NodeView::Duration(terms) => {
            let terms: Vec<DurationTerm> = terms
                .iter()
                .map(|term| DurationTerm {
                    demand: nat_of(src, AnyExpr::Nat(term.demand), dst, map),
                    lower_numerator: term.lower_numerator,
                    upper_numerator: term.upper_numerator,
                    denominator: term.denominator,
                })
                .collect();
            AnyExpr::Duration(dst.duration(&terms))
        }
        NodeView::DurationScale { duration, by } => {
            let duration = duration_of(src, AnyExpr::Duration(duration), dst, map);
            let by = nat_of(src, AnyExpr::Nat(by), dst, map);
            AnyExpr::Duration(dst.duration_scale(duration, by))
        }
    }
}
