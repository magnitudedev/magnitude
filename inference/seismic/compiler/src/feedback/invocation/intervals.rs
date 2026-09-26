//! Conservative interval contraction. This is navigation, never admission:
//! unsupported/partial arithmetic remains unknown and every witness is checked
//! by the entry's compiled predicate. The program is a derived DAG, not a new
//! domain or solver authority.
use super::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
struct Bounds(i128, i128);
#[derive(Clone, Debug)]
enum Op {
    Unknown,
    Constant(i128),
    Axis(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Alias(usize),
}
pub(super) struct Program {
    nodes: Vec<Op>,
    constraints: Vec<(CmpOp, usize, usize)>,
}
fn integer(value: SymbolValue) -> Option<i128> {
    match value {
        SymbolValue::Nat(v) => i128::try_from(v).ok(),
        SymbolValue::Int(v) => i128::try_from(v).ok(),
        _ => None,
    }
}
impl Program {
    pub(super) fn new(
        arena: &ExprArena,
        predicate: AnyExpr,
        axes: &[Axis],
        aliases: &[(SymbolId, SymbolId)],
        constants: &[(SymbolId, SymbolValue)],
    ) -> Self {
        let mut program = Self {
            nodes: Vec::new(),
            constraints: Vec::new(),
        };
        let mut comparisons = Vec::new();
        let mut pending = vec![predicate];
        let mut seen = HashSet::new();
        while let Some(node) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            match arena.view(node) {
                NodeView::Binary {
                    op: expr::BinaryOp::And,
                    lhs,
                    rhs,
                } => pending.extend([lhs, rhs]),
                NodeView::Nary {
                    op: NaryOp::All,
                    operands,
                } => pending.extend(operands),
                NodeView::Cmp { op, lhs, rhs } => comparisons.push((op, lhs, rhs)),
                _ => {}
            }
        }
        let mut ids = HashMap::new();
        for (cmp, lhs, rhs) in comparisons {
            for root in [lhs, rhs] {
                let mut pending = vec![(root, false)];
                while let Some((node, expanded)) = pending.pop() {
                    if ids.contains_key(&node) {
                        continue;
                    }
                    let children: Vec<_> = match arena.view(node) {
                        NodeView::Binary {
                            op: expr::BinaryOp::Add | expr::BinaryOp::Sub | expr::BinaryOp::Mul,
                            lhs,
                            rhs,
                        } => vec![lhs, rhs],
                        NodeView::Unary {
                            op: expr::UnaryOp::IntFromNat | expr::UnaryOp::NatFromInt,
                            operand,
                        } => vec![operand],
                        NodeView::Nary {
                            op: NaryOp::Product,
                            operands,
                        } => operands.to_vec(),
                        _ => Vec::new(),
                    };
                    if !expanded && !children.is_empty() {
                        pending.push((node, true));
                        pending.extend(children.into_iter().map(|n| (n, false)));
                        continue;
                    }
                    let op = match arena.view(node) {
                        NodeView::NatConst(v) => Op::Constant(i128::from(v)),
                        NodeView::IntConst(v) => Op::Constant(i128::from(v)),
                        NodeView::Symbol(symbol) => {
                            let symbol = aliases
                                .iter()
                                .find(|(a, _)| *a == symbol)
                                .map(|(_, r)| *r)
                                .unwrap_or(symbol);
                            if let Some(i) = axes.iter().position(|a| {
                                a.symbol == symbol
                                    && matches!(a.sort, SymbolSort::Nat | SymbolSort::Int)
                            }) {
                                Op::Axis(i)
                            } else {
                                constants
                                    .iter()
                                    .find(|(s, _)| *s == symbol)
                                    .and_then(|(_, v)| integer(v.clone()))
                                    .map(Op::Constant)
                                    .unwrap_or(Op::Unknown)
                            }
                        }
                        NodeView::Unary {
                            op: expr::UnaryOp::IntFromNat | expr::UnaryOp::NatFromInt,
                            operand,
                        } => Op::Alias(ids[&operand]),
                        NodeView::Binary { op, lhs, rhs }
                            if ids.contains_key(&lhs) && ids.contains_key(&rhs) =>
                        {
                            match op {
                                expr::BinaryOp::Add => Op::Add(ids[&lhs], ids[&rhs]),
                                expr::BinaryOp::Sub => Op::Sub(ids[&lhs], ids[&rhs]),
                                expr::BinaryOp::Mul => Op::Mul(ids[&lhs], ids[&rhs]),
                                _ => Op::Unknown,
                            }
                        }
                        NodeView::Nary {
                            op: NaryOp::Product,
                            operands,
                        } => {
                            let mut accumulator = program.nodes.len();
                            program.nodes.push(Op::Constant(1));
                            for operand in operands {
                                let next = program.nodes.len();
                                program.nodes.push(Op::Mul(accumulator, ids[operand]));
                                accumulator = next;
                            }
                            Op::Alias(accumulator)
                        }
                        _ => Op::Unknown,
                    };
                    ids.insert(node, program.nodes.len());
                    program.nodes.push(op);
                }
            }
            program.constraints.push((cmp, ids[&lhs], ids[&rhs]));
        }
        program
    }
    fn ranges(&self, axes: &[Axis], cell: &Cell) -> Vec<Option<Bounds>> {
        let mut values: Vec<Option<Bounds>> = Vec::with_capacity(self.nodes.len());
        for op in &self.nodes {
            let value = (|| {
                Some(match *op {
                    Op::Unknown => return None,
                    Op::Constant(v) => Bounds(v, v),
                    Op::Axis(i) => Bounds(
                        integer(decode(axes[i].sort, cell.bounds[i].0))?,
                        integer(decode(axes[i].sort, cell.bounds[i].1))?,
                    ),
                    Op::Alias(i) => values[i]?,
                    Op::Add(a, b) => {
                        let (a, b) = (values[a]?, values[b]?);
                        Bounds(a.0.checked_add(b.0)?, a.1.checked_add(b.1)?)
                    }
                    Op::Sub(a, b) => {
                        let (a, b) = (values[a]?, values[b]?);
                        Bounds(a.0.checked_sub(b.1)?, a.1.checked_sub(b.0)?)
                    }
                    Op::Mul(a, b) => {
                        let (a, b) = (values[a]?, values[b]?);
                        let products = [
                            a.0.checked_mul(b.0)?,
                            a.0.checked_mul(b.1)?,
                            a.1.checked_mul(b.0)?,
                            a.1.checked_mul(b.1)?,
                        ];
                        Bounds(*products.iter().min()?, *products.iter().max()?)
                    }
                })
            })();
            values.push(value);
        }
        values
    }
    fn restrict(
        &self,
        node: usize,
        bound: Bounds,
        ranges: &[Option<Bounds>],
        axes: &[Axis],
        cell: &mut Cell,
    ) -> bool {
        let mut pending = vec![(node, bound)];
        let mut work = 0usize;
        while let Some((node, bound)) = pending.pop() {
            work += 1;
            if work > self.nodes.len().saturating_mul(4).max(64) {
                break;
            }
            let Some(old) = ranges[node] else {
                continue;
            };
            let bound = Bounds(old.0.max(bound.0), old.1.min(bound.1));
            if bound.0 > bound.1 {
                return false;
            }
            match self.nodes[node] {
                Op::Axis(i) => {
                    let rank =
                        |v| match axes[i].sort {
                            SymbolSort::Nat => encode(SymbolValue::Nat(
                                u64::try_from(v)
                                    .expect(
                                        "restricted axis value is nonnegative and within its rank",
                                    )
                                    .into(),
                            ))
                            .expect("axis value has rank")
                            .1,
                            SymbolSort::Int => {
                                encode(SymbolValue::Int(
                                    i64::try_from(v)
                                        .expect("restricted axis value is within its rank")
                                        .into(),
                                ))
                                .expect("axis value has rank")
                                .1
                            }
                            _ => unreachable!(),
                        };
                    cell.bounds[i].0 = cell.bounds[i].0.max(rank(bound.0));
                    cell.bounds[i].1 = cell.bounds[i].1.min(rank(bound.1));
                    if cell.bounds[i].0 > cell.bounds[i].1 {
                        return false;
                    }
                }
                Op::Alias(i) => pending.push((i, bound)),
                Op::Add(a, b) | Op::Sub(a, b) => {
                    let (Some(x), Some(y)) = (ranges[a], ranges[b]) else {
                        continue;
                    };
                    let subtract = matches!(self.nodes[node], Op::Sub(..));
                    let left = if subtract {
                        bound.0.checked_add(y.0).zip(bound.1.checked_add(y.1))
                    } else {
                        bound.0.checked_sub(y.1).zip(bound.1.checked_sub(y.0))
                    };
                    let right = if subtract {
                        x.0.checked_sub(bound.1).zip(x.1.checked_sub(bound.0))
                    } else {
                        bound.0.checked_sub(x.1).zip(bound.1.checked_sub(x.0))
                    };
                    if let Some((l, h)) = left {
                        pending.push((a, Bounds(l, h)));
                    }
                    if let Some((l, h)) = right {
                        pending.push((b, Bounds(l, h)));
                    }
                }
                Op::Mul(a, b) => {
                    // Positive-factor inversion with directed integer rounding.
                    for (child, factor) in [(a, b), (b, a)] {
                        if let Some(Bounds(lo, hi)) = ranges[factor] {
                            if lo > 0 && bound.0 >= 0 {
                                let lower = bound.0 / hi + i128::from(bound.0 % hi != 0);
                                pending.push((child, Bounds(lower, bound.1 / lo)));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        true
    }
    pub(super) fn contract(&self, axes: &[Axis], cell: &mut Cell) -> bool {
        // A bounded fixed point: unfinished contraction is unknown, never empty.
        for _ in 0..8 {
            let before = cell.bounds.clone();
            let ranges = self.ranges(axes, cell);
            for &(op, a, b) in &self.constraints {
                let (Some(x), Some(y)) = (ranges[a], ranges[b]) else {
                    continue;
                };
                let impossible = match op {
                    CmpOp::Eq => x.1 < y.0 || y.1 < x.0,
                    CmpOp::Ne => x.0 == x.1 && y.0 == y.1 && x.0 == y.0,
                    CmpOp::Lt => x.0 >= y.1,
                    CmpOp::Le => x.0 > y.1,
                    CmpOp::Gt => x.1 <= y.0,
                    CmpOp::Ge => x.1 < y.0,
                };
                if impossible {
                    return false;
                }
                let pair = match op {
                    CmpOp::Eq => Some((y, x)),
                    CmpOp::Le => Some((Bounds(i128::MIN, y.1), Bounds(x.0, i128::MAX))),
                    CmpOp::Lt => {
                        y.1.checked_sub(1)
                            .zip(x.0.checked_add(1))
                            .map(|(h, l)| (Bounds(i128::MIN, h), Bounds(l, i128::MAX)))
                    }
                    CmpOp::Ge => Some((Bounds(y.0, i128::MAX), Bounds(i128::MIN, x.1))),
                    CmpOp::Gt => {
                        y.0.checked_add(1)
                            .zip(x.1.checked_sub(1))
                            .map(|(l, h)| (Bounds(l, i128::MAX), Bounds(i128::MIN, h)))
                    }
                    CmpOp::Ne => None,
                };
                if let Some((left, right)) = pair {
                    if !self.restrict(a, left, &ranges, axes, cell)
                        || !self.restrict(b, right, &ranges, axes, cell)
                    {
                        return false;
                    }
                }
            }
            if cell.bounds == before {
                break;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupled_affine_bounds_propagate_and_disjoint_scope_is_proven_empty() {
        let mut arena = ExprArena::new();
        let (_, sx) = arena.target_constant(SymbolSort::Nat);
        let (_, sy) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(sx);
        let y = arena.nat_symbol(sy);
        let sum = arena.nat_add(x, y);
        let limit = arena.nat(100);
        let predicate = arena.nat_cmp(CmpOp::Eq, sum, limit);
        let axes = vec![
            Axis::new(sx, SymbolSort::Nat),
            Axis::new(sy, SymbolSort::Nat),
        ];
        let program = Program::new(&arena, predicate.into(), &axes, &[], &[]);
        let mut cell = Cell {
            bounds: vec![(0, 1_000_000), (70, 80)],
        };
        assert!(program.contract(&axes, &mut cell));
        assert_eq!(cell.bounds, vec![(20, 30), (70, 80)]);
        let mut empty = Cell {
            bounds: vec![(31, 90), (70, 80)],
        };
        assert!(!program.contract(&axes, &mut empty));
    }

    #[test]
    fn nonlinear_product_contraction_preserves_every_legal_integer_pair() {
        let mut arena = ExprArena::new();
        let (_, sx) = arena.target_constant(SymbolSort::Nat);
        let (_, sy) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(sx);
        let y = arena.nat_symbol(sy);
        let product = arena.nat_product(&[x, y]);
        let limit = arena.nat(143);
        let predicate = arena.nat_cmp(CmpOp::Eq, product, limit);
        let axes = vec![
            Axis::new(sx, SymbolSort::Nat),
            Axis::new(sy, SymbolSort::Nat),
        ];
        let program = Program::new(&arena, predicate.into(), &axes, &[], &[]);
        let mut cell = Cell {
            bounds: vec![(1, 200), (10, 15)],
        };
        assert!(program.contract(&axes, &mut cell));
        assert_eq!(cell.bounds[0], (11, 13));
        for x in 1..=200 {
            for y in 10..=15 {
                if x * y == 143 {
                    assert!(
                        cell.bounds[0].0 <= x
                            && x <= cell.bounds[0].1
                            && cell.bounds[1].0 <= y
                            && y <= cell.bounds[1].1
                    );
                }
            }
        }
    }

    #[test]
    fn unsupported_arithmetic_does_not_prove_an_empty_region() {
        let mut arena = ExprArena::new();
        let (_, sx) = arena.target_constant(SymbolSort::Nat);
        let x = arena.nat_symbol(sx);
        let two = arena.nat(2);
        let one = arena.nat(1);
        let remainder = arena.nat_rem(x, two);
        let predicate = arena.nat_cmp(CmpOp::Eq, remainder, one);
        let axes = vec![Axis::new(sx, SymbolSort::Nat)];
        let program = Program::new(&arena, predicate.into(), &axes, &[], &[]);
        let mut cell = Cell {
            bounds: vec![(0, 100)],
        };
        assert!(program.contract(&axes, &mut cell));
        assert_eq!(cell.bounds, vec![(0, 100)]);
    }

    #[test]
    fn signed_contraction_never_loses_a_legal_assignment() {
        for op in [
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
        ] {
            let mut arena = ExprArena::new();
            let (_, sx) = arena.target_constant(SymbolSort::Int);
            let (_, sy) = arena.target_constant(SymbolSort::Int);
            let x = arena.int_symbol(sx);
            let y = arena.int_symbol(sy);
            let sum = arena.int_add(x, y);
            let limit = arena.int(-3);
            let predicate = arena.int_cmp(op, sum, limit);
            let axes = vec![
                Axis::new(sx, SymbolSort::Int),
                Axis::new(sy, SymbolSort::Int),
            ];
            let program = Program::new(&arena, predicate.into(), &axes, &[], &[]);
            let rank = |v: i64| encode(SymbolValue::Int(v.into())).unwrap().1;
            let mut cell = Cell {
                bounds: vec![(rank(-10), rank(5)), (rank(-5), rank(10))],
            };
            let kept = program.contract(&axes, &mut cell);
            let compiled = arena.compile_bool_with(predicate, &PartialAssignment::new());
            for x in -10..=5 {
                for y in -5..=10 {
                    let mut values = InvocationValues::new();
                    values.bind(sx, SymbolValue::Int(x.into()));
                    values.bind(sy, SymbolValue::Int(y.into()));
                    if compiled.evaluate(&values).unwrap() {
                        assert!(
                            kept && cell.bounds[0].0 <= rank(x)
                                && rank(x) <= cell.bounds[0].1
                                && cell.bounds[1].0 <= rank(y)
                                && rank(y) <= cell.bounds[1].1
                        );
                    }
                }
            }
        }
    }
}
