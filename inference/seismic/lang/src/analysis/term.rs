//! Symbolic real-valued expressions derived from checked portable bodies.
//!
//! A [`Term`] is exact mathematical arithmetic over named [`Variable`]s: it
//! carries no rounding except where the source publishes a value in a
//! narrower floating type ([`Term::evaluate`] applies that rounding). Integer
//! quantities are represented exactly while they stay within `2^53`.

use crate::registry::{PlaneField, QuantizedComponent};
use crate::types::DType;
use std::fmt;
use std::sync::Arc;

/// A free name in a [`Term`].
#[derive(Clone, Debug, PartialEq)]
pub enum Variable {
    /// A scalar parameter of the entry, by name.
    Parameter(String),
    /// A dimension of the entry, by name.
    Dimension(String),
    /// A loop coordinate; its range is reported alongside the term.
    Binder(u32),
    /// The value of one element of an entry tensor parameter, as the body
    /// reads it: the stored element of dense storage, the decoded value of a
    /// packed representation.
    Element { parameter: String, index: Vec<Term> },
    /// One registry decode component of an element of a packed entry tensor
    /// parameter (its scale, code value, ...), as a body reads it.
    Component {
        parameter: String,
        component: QuantizedComponent,
        index: Vec<Term>,
    },
    /// What the registry decode of the logical element at `index` of a packed
    /// entry tensor parameter reads from entry `field` of its group in
    /// `plane`: the interpreted code of a coded plane, the stored value of a
    /// floating plane.
    Plane {
        parameter: String,
        plane: PlaneField,
        field: u32,
        index: Vec<Term>,
    },
}

/// The values a stored code takes. Adjacent codes are adjacent levels.
#[derive(Clone, Debug, PartialEq)]
pub enum CodeLevels {
    /// Every integer from `min` to `max`.
    Integers { min: i64, max: i64 },
    /// These values, ascending and distinct.
    Table(Vec<f64>),
}

/// The levels adjacent to a code value. An end of the code range has no
/// neighbour on its outer side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbours {
    /// The greatest level below the value.
    pub below: Option<f64>,
    /// The least level above the value.
    pub above: Option<f64>,
}

impl CodeLevels {
    /// The levels `values` takes: consecutive integers as a range, anything
    /// else as a table.
    pub(crate) fn of(values: impl IntoIterator<Item = f64>) -> CodeLevels {
        let mut levels = values.into_iter().collect::<Vec<_>>();
        levels.sort_by(f64::total_cmp);
        levels.dedup_by(|a, b| a == b);
        let consecutive = levels.iter().all(|level| level.fract() == 0.0)
            && levels.windows(2).all(|pair| pair[1] - pair[0] == 1.0);
        match (levels.first(), levels.last()) {
            (Some(min), Some(max)) if consecutive => CodeLevels::Integers {
                min: *min as i64,
                max: *max as i64,
            },
            _ => CodeLevels::Table(levels),
        }
    }

    /// The levels adjacent to `value`.
    pub fn neighbours(&self, value: f64) -> Neighbours {
        match self {
            CodeLevels::Integers { min, max } => {
                let (min, max) = (*min as f64, *max as f64);
                let below = (value.ceil() - 1.0).min(max);
                let above = (value.floor() + 1.0).max(min);
                Neighbours {
                    below: (below >= min).then_some(below),
                    above: (above <= max).then_some(above),
                }
            }
            CodeLevels::Table(levels) => Neighbours {
                below: levels.iter().rev().find(|level| **level < value).copied(),
                above: levels.iter().find(|level| **level > value).copied(),
            },
        }
    }
}

/// Why a term has no value.
#[derive(Clone, Debug, PartialEq)]
pub enum TermError {
    /// The caller supplied no value for this variable.
    Unbound(Variable),
    /// The term depends on its inputs in a way the analysis did not resolve.
    Unresolved,
}

impl fmt::Display for TermError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermError::Unbound(variable) => write!(f, "no value for {variable}"),
            TermError::Unresolved => f.write_str("the term was not resolved by the analysis"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unary {
    Neg,
    Abs,
    Exp,
    Exp2,
    Log,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Rint,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    /// Euclidean integer quotient.
    Quotient,
    /// Euclidean integer remainder.
    Remainder,
    Min,
    Max,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    BitXor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fold {
    Sum,
    Max,
    Min,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Node {
    Constant(f64),
    Variable(Variable),
    /// A value the analysis did not resolve, with the variables it reads.
    Opaque(Vec<Variable>),
    Unary(Unary, Term),
    Binary(Binary, Term, Term),
    Select(Term, Term, Term),
    /// Publication in a narrower floating type: the value is rounded.
    Round(DType, Term),
    /// Conversion to an integer word: truncation toward zero.
    Truncate(Term),
    /// `op` over `body` for `binder` in `start..end`.
    Fold {
        op: Fold,
        binder: u32,
        start: Term,
        end: Term,
        body: Term,
    },
    /// A read of an entry tensor parameter that an earlier write of this
    /// call may have replaced, where the analysis did not resolve whether it
    /// did. `earlier` is the parameter's content before that write; `by` are
    /// the variables the write's value and position read.
    Overwritten { by: Vec<Variable>, earlier: Term },
    /// The analysis's own stand-in for loop-carried state while it resolves
    /// a loop; never present in a published term.
    Placeholder(u32, Vec<Term>),
}

/// A symbolic real-valued expression.
#[derive(Clone, PartialEq)]
pub struct Term(pub(crate) Arc<Node>);

impl fmt::Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl Term {
    pub(crate) fn node(&self) -> &Node {
        &self.0
    }

    fn new(node: Node) -> Term {
        Term(Arc::new(node))
    }

    pub fn constant(value: f64) -> Term {
        Term::new(Node::Constant(value))
    }

    pub fn variable(variable: Variable) -> Term {
        Term::new(Node::Variable(variable))
    }

    pub(crate) fn opaque(dependencies: Vec<Variable>) -> Term {
        Term::new(Node::Opaque(without_repeats(dependencies)))
    }

    /// The constant value of this term, when it is one.
    pub fn as_constant(&self) -> Option<f64> {
        match self.node() {
            Node::Constant(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn unary(op: Unary, operand: Term) -> Term {
        if let Some(value) = operand.as_constant() {
            if let Some(value) = apply_unary(op, value) {
                return Term::constant(value);
            }
        }
        if op == Unary::Neg {
            if let Node::Unary(Unary::Neg, inner) = operand.node() {
                return inner.clone();
            }
        }
        Term::new(Node::Unary(op, operand))
    }

    pub(crate) fn binary(op: Binary, left: Term, right: Term) -> Term {
        if let (Some(a), Some(b)) = (left.as_constant(), right.as_constant()) {
            if let Some(value) = apply_binary(op, a, b) {
                return Term::constant(value);
            }
        }
        let zero = |term: &Term| term.as_constant() == Some(0.0);
        let one = |term: &Term| term.as_constant() == Some(1.0);
        match op {
            Binary::Add if zero(&left) => return right,
            Binary::Add | Binary::Sub if zero(&right) => return left,
            Binary::Sub if zero(&left) => return Term::unary(Unary::Neg, right),
            Binary::Mul if zero(&left) || zero(&right) => return Term::constant(0.0),
            Binary::Mul if one(&left) => return right,
            Binary::Mul | Binary::Div | Binary::Quotient if one(&right) => return left,
            _ => {}
        }
        Term::new(Node::Binary(op, left, right))
    }

    pub(crate) fn add(self, other: Term) -> Term {
        Term::binary(Binary::Add, self, other)
    }

    pub(crate) fn sub(self, other: Term) -> Term {
        Term::binary(Binary::Sub, self, other)
    }

    pub(crate) fn mul(self, other: Term) -> Term {
        Term::binary(Binary::Mul, self, other)
    }

    pub(crate) fn div(self, other: Term) -> Term {
        Term::binary(Binary::Div, self, other)
    }

    pub(crate) fn abs(self) -> Term {
        Term::unary(Unary::Abs, self)
    }

    pub(crate) fn select(condition: Term, then: Term, otherwise: Term) -> Term {
        match condition.as_constant() {
            Some(value) if value != 0.0 => return then,
            Some(_) => return otherwise,
            None => {}
        }
        if then == otherwise {
            return then;
        }
        Term::new(Node::Select(condition, then, otherwise))
    }

    pub(crate) fn round(dtype: DType, value: Term) -> Term {
        match value.node() {
            Node::Constant(constant) => Term::constant(round_to(dtype, *constant)),
            Node::Round(inner, _) if *inner == dtype => value,
            _ => Term::new(Node::Round(dtype, value)),
        }
    }

    pub(crate) fn truncate(value: Term) -> Term {
        match value.as_constant() {
            Some(constant) => Term::constant(constant.trunc()),
            None => Term::new(Node::Truncate(value)),
        }
    }

    pub(crate) fn placeholder(id: u32, index: Vec<Term>) -> Term {
        Term::new(Node::Placeholder(id, index))
    }

    /// Whether a placeholder with `id` occurs in this term.
    pub(crate) fn carries(&self, id: u32) -> bool {
        self.any(&|term| matches!(term.node(), Node::Placeholder(found, _) if *found == id))
    }

    /// Integer-affine normal form: `constant + Σ coefficient * atom`, where
    /// atoms are the non-affine subterms. `None` when a coefficient would not
    /// be a constant.
    pub(crate) fn affine(&self) -> Option<Affine> {
        Some(match self.node() {
            Node::Constant(value) => Affine {
                constant: *value,
                terms: Vec::new(),
            },
            Node::Unary(Unary::Neg, a) => a.affine()?.scaled(-1.0),
            Node::Binary(Binary::Add, a, b) => a.affine()?.plus(&b.affine()?, 1.0),
            Node::Binary(Binary::Sub, a, b) => a.affine()?.plus(&b.affine()?, -1.0),
            Node::Binary(Binary::Mul, a, b) => {
                let (a, b) = (a.affine()?, b.affine()?);
                if a.terms.is_empty() {
                    b.scaled(a.constant)
                } else if b.terms.is_empty() {
                    a.scaled(b.constant)
                } else {
                    Affine::atom(self.clone())
                }
            }
            _ => Affine::atom(self.clone()),
        })
    }

    /// `Some(true)` when the two terms are provably equal, `Some(false)`
    /// when provably different, `None` when undecided.
    pub(crate) fn compare(&self, other: &Term) -> Option<bool> {
        if self == other {
            return Some(true);
        }
        let difference = self.affine()?.plus(&other.affine()?, -1.0);
        if difference.terms.is_empty() {
            Some(difference.constant == 0.0)
        } else {
            None
        }
    }

    pub(crate) fn fold(op: Fold, binder: u32, start: Term, end: Term, body: Term) -> Term {
        Term::new(Node::Fold {
            op,
            binder,
            start,
            end,
            body,
        })
    }

    pub(crate) fn overwritten(by: Vec<Variable>, earlier: Term) -> Term {
        Term::new(Node::Overwritten {
            by: without_repeats(by),
            earlier,
        })
    }

    /// Whether the analysis resolved every part of this term.
    pub fn is_resolved(&self) -> bool {
        !self.any(&|term| {
            matches!(
                term.node(),
                Node::Opaque(_) | Node::Placeholder(..) | Node::Overwritten { .. }
            )
        })
    }

    /// Every variable this term reads, outermost first, without repeats.
    /// Variables inside element indices are included.
    pub fn variables(&self) -> Vec<Variable> {
        let mut out: Vec<Variable> = Vec::new();
        self.visit(&mut |term| match term.node() {
            Node::Variable(variable) => {
                if !out.contains(variable) {
                    out.push(variable.clone());
                }
            }
            Node::Opaque(dependencies) | Node::Overwritten { by: dependencies, .. } => {
                for variable in dependencies {
                    if !out.contains(variable) {
                        out.push(variable.clone());
                    }
                }
            }
            _ => {}
        });
        out
    }

    /// Visit this term and every subterm, including those inside variable
    /// indices.
    pub(crate) fn visit(&self, f: &mut impl FnMut(&Term)) {
        crate::cancellation::check();
        f(self);
        match self.node() {
            Node::Constant(_) | Node::Opaque(_) => {}
            Node::Placeholder(_, index) => {
                for term in index {
                    term.visit(f);
                }
            }
            Node::Variable(variable) => match variable {
                Variable::Element { index, .. }
                | Variable::Component { index, .. }
                | Variable::Plane { index, .. } => {
                    for term in index {
                        term.visit(f);
                    }
                }
                _ => {}
            },
            Node::Unary(_, a)
            | Node::Round(_, a)
            | Node::Truncate(a)
            | Node::Overwritten { earlier: a, .. } => a.visit(f),
            Node::Binary(_, a, b) => {
                a.visit(f);
                b.visit(f);
            }
            Node::Select(a, b, c) => {
                a.visit(f);
                b.visit(f);
                c.visit(f);
            }
            Node::Fold {
                start, end, body, ..
            } => {
                start.visit(f);
                end.visit(f);
                body.visit(f);
            }
        }
    }

    /// Whether `predicate` holds for some subterm.
    pub(crate) fn any(&self, predicate: &impl Fn(&Term) -> bool) -> bool {
        let mut found = false;
        self.visit(&mut |term| found |= predicate(term));
        found
    }

    /// Whether `by` occurs in this term, including as a dependency of an
    /// unresolved part.
    pub(crate) fn mentions(&self, by: &Term) -> bool {
        let variable = match by.node() {
            Node::Variable(variable) => Some(variable),
            _ => None,
        };
        self.any(&|term| {
            term == by
                || matches!(
                    (term.node(), variable),
                    (Node::Opaque(dependencies) | Node::Overwritten { by: dependencies, .. }, Some(variable))
                        if dependencies.contains(variable)
                )
        })
    }

    /// This term with every subterm equal to `from` replaced by `to`.
    pub(crate) fn replace(&self, from: &Term, to: &Term) -> Term {
        if self == from {
            return to.clone();
        }
        self.map_children(&mut |child| child.replace(from, to))
    }

    /// This term with `Variable::Binder(binder)` replaced by `value`.
    pub(crate) fn bind(&self, binder: u32, value: &Term) -> Term {
        self.replace(&Term::variable(Variable::Binder(binder)), value)
    }

    fn map_children(&self, f: &mut impl FnMut(&Term) -> Term) -> Term {
        match self.node() {
            Node::Constant(_) => self.clone(),
            Node::Opaque(dependencies) => Term::opaque(
                dependencies
                    .iter()
                    .map(|variable| map_variable(variable, f))
                    .collect(),
            ),
            Node::Variable(variable) => Term::variable(map_variable(variable, f)),
            Node::Placeholder(id, index) => {
                Term::placeholder(*id, index.iter().map(f).collect())
            }
            Node::Unary(op, a) => Term::unary(*op, f(a)),
            Node::Binary(op, a, b) => Term::binary(*op, f(a), f(b)),
            Node::Select(a, b, c) => Term::select(f(a), f(b), f(c)),
            Node::Round(dtype, a) => Term::round(*dtype, f(a)),
            Node::Truncate(a) => Term::truncate(f(a)),
            Node::Overwritten { by, earlier } => Term::overwritten(
                by.iter().map(|variable| map_variable(variable, f)).collect(),
                f(earlier),
            ),
            Node::Fold {
                op,
                binder,
                start,
                end,
                body,
            } => Term::fold(*op, *binder, f(start), f(end), f(body)),
        }
    }

    /// Evaluate with `value` supplying each variable. A [`Variable::Binder`]
    /// bound by a fold inside the term is supplied by the fold itself.
    pub fn evaluate(&self, value: &impl Fn(&Variable) -> Option<f64>) -> Result<f64, TermError> {
        self.evaluate_with(value, &mut Vec::new())
    }

    fn evaluate_with(
        &self,
        value: &impl Fn(&Variable) -> Option<f64>,
        folds: &mut Vec<(u32, f64)>,
    ) -> Result<f64, TermError> {
        crate::cancellation::check();
        Ok(match self.node() {
            Node::Constant(constant) => *constant,
            Node::Opaque(_) | Node::Placeholder(..) | Node::Overwritten { .. } => {
                return Err(TermError::Unresolved)
            }
            Node::Variable(variable) => {
                if let Variable::Binder(binder) = variable {
                    if let Some((_, coordinate)) =
                        folds.iter().rev().find(|(bound, _)| bound == binder)
                    {
                        return Ok(*coordinate);
                    }
                }
                let resolved = match variable {
                    Variable::Element { parameter, index } => Variable::Element {
                        parameter: parameter.clone(),
                        index: constant_index(index, value, folds)?,
                    },
                    Variable::Component {
                        parameter,
                        component,
                        index,
                    } => Variable::Component {
                        parameter: parameter.clone(),
                        component: *component,
                        index: constant_index(index, value, folds)?,
                    },
                    Variable::Plane {
                        parameter,
                        plane,
                        field,
                        index,
                    } => Variable::Plane {
                        parameter: parameter.clone(),
                        plane: *plane,
                        field: *field,
                        index: constant_index(index, value, folds)?,
                    },
                    other => other.clone(),
                };
                value(&resolved).ok_or(TermError::Unbound(resolved))?
            }
            Node::Unary(op, a) => {
                let a = a.evaluate_with(value, folds)?;
                apply_unary(*op, a).ok_or(TermError::Unresolved)?
            }
            Node::Binary(op, a, b) => {
                let a = a.evaluate_with(value, folds)?;
                if matches!(op, Binary::And) && a == 0.0 {
                    return Ok(0.0);
                }
                if matches!(op, Binary::Or) && a != 0.0 {
                    return Ok(1.0);
                }
                let b = b.evaluate_with(value, folds)?;
                apply_binary(*op, a, b).ok_or(TermError::Unresolved)?
            }
            Node::Select(condition, then, otherwise) => {
                if condition.evaluate_with(value, folds)? != 0.0 {
                    then.evaluate_with(value, folds)?
                } else {
                    otherwise.evaluate_with(value, folds)?
                }
            }
            Node::Round(dtype, a) => round_to(*dtype, a.evaluate_with(value, folds)?),
            Node::Truncate(a) => a.evaluate_with(value, folds)?.trunc(),
            Node::Fold {
                op,
                binder,
                start,
                end,
                body,
            } => {
                let start = start.evaluate_with(value, folds)?;
                let end = end.evaluate_with(value, folds)?;
                let mut accumulator = match op {
                    Fold::Sum => 0.0,
                    Fold::Max => f64::NEG_INFINITY,
                    Fold::Min => f64::INFINITY,
                };
                let mut coordinate = start;
                while coordinate < end {
                    folds.push((*binder, coordinate));
                    let item = body.evaluate_with(value, folds);
                    folds.pop();
                    let item = item?;
                    accumulator = match op {
                        Fold::Sum => accumulator + item,
                        Fold::Max => accumulator.max(item),
                        Fold::Min => accumulator.min(item),
                    };
                    coordinate += 1.0;
                }
                accumulator
            }
        })
    }

    /// This term with every binder, free or bound by a fold, renumbered.
    /// `number` must be injective over the binders present.
    pub(crate) fn rename_binders(&self, number: &impl Fn(u32) -> u32) -> Term {
        match self.node() {
            Node::Variable(Variable::Binder(binder)) => {
                Term::variable(Variable::Binder(number(*binder)))
            }
            Node::Fold {
                op,
                binder,
                start,
                end,
                body,
            } => Term::fold(
                *op,
                number(*binder),
                start.rename_binders(number),
                end.rename_binders(number),
                body.rename_binders(number),
            ),
            _ => self.map_children(&mut |child| child.rename_binders(number)),
        }
    }

    /// This term where no unresolved earlier write replaced what it reads:
    /// every [`Node::Overwritten`] read is the parameter's earlier content.
    pub(crate) fn unshadowed(&self) -> Term {
        match self.node() {
            Node::Overwritten { earlier, .. } => earlier.unshadowed(),
            _ => self.map_children(&mut |child| child.unshadowed()),
        }
    }

    /// `(slope, intercept)` with `self = slope * by + intercept` and neither
    /// reading `by`. `None` when the term is not affine in `by`, when a
    /// rounding boundary lies between them, or when an unresolved part of the
    /// term may read the storage `by` reads.
    pub(crate) fn affine_in(&self, by: &Term) -> Option<(Term, Term)> {
        let reads = by.variables();
        self.affine_parts(by, &reads)
    }

    fn affine_parts(&self, by: &Term, reads: &[Variable]) -> Option<(Term, Term)> {
        crate::cancellation::check();
        if self == by {
            return Some((Term::constant(1.0), Term::constant(0.0)));
        }
        let free = !self.any(&|term| {
            term == by
                || matches!(
                    term.node(),
                    Node::Opaque(dependencies) | Node::Overwritten { by: dependencies, .. }
                        if dependencies.iter().any(|dependency| {
                            reads.iter().any(|read| {
                                dependency == read
                                    || (dependency.storage().is_some()
                                        && dependency.storage() == read.storage())
                            })
                        })
                )
        });
        if free {
            return Some((Term::constant(0.0), self.clone()));
        }
        let parts = |term: &Term| term.affine_parts(by, reads);
        let constant = |term: &Term| match parts(term)? {
            (slope, value) if slope.as_constant() == Some(0.0) => Some(value),
            _ => None,
        };
        Some(match self.node() {
            Node::Unary(Unary::Neg, a) => {
                let (slope, intercept) = parts(a)?;
                (Term::unary(Unary::Neg, slope), Term::unary(Unary::Neg, intercept))
            }
            Node::Binary(op @ (Binary::Add | Binary::Sub), a, b) => {
                let ((slope_a, intercept_a), (slope_b, intercept_b)) = (parts(a)?, parts(b)?);
                (
                    Term::binary(*op, slope_a, slope_b),
                    Term::binary(*op, intercept_a, intercept_b),
                )
            }
            Node::Binary(Binary::Mul, a, b) => {
                let (factor, (slope, intercept)) = match (constant(a), constant(b)) {
                    (Some(factor), _) => (factor, parts(b)?),
                    (None, Some(factor)) => (factor, parts(a)?),
                    (None, None) => return None,
                };
                (factor.clone().mul(slope), factor.mul(intercept))
            }
            Node::Binary(Binary::Div, a, b) => {
                let divisor = constant(b)?;
                let (slope, intercept) = parts(a)?;
                (slope.div(divisor.clone()), intercept.div(divisor))
            }
            Node::Select(condition, then, otherwise) => {
                let condition = constant(condition)?;
                let ((slope_a, intercept_a), (slope_b, intercept_b)) =
                    (parts(then)?, parts(otherwise)?);
                (
                    Term::select(condition.clone(), slope_a, slope_b),
                    Term::select(condition, intercept_a, intercept_b),
                )
            }
            _ => return None,
        })
    }

    /// The tensor parameter values this term's value is computed from:
    /// operands of arithmetic and the selected arms of a selection, not the
    /// positions they are read at and not the conditions that select them.
    /// `None` when an unresolved part contributes to the value.
    pub(crate) fn value_reads(&self) -> Option<Vec<Variable>> {
        let mut reads = Vec::new();
        self.collect_value_reads(&mut reads).then_some(reads)
    }

    fn collect_value_reads(&self, reads: &mut Vec<Variable>) -> bool {
        crate::cancellation::check();
        match self.node() {
            Node::Constant(_) => true,
            Node::Variable(variable) => {
                if variable.storage().is_some()
                    && !matches!(variable, Variable::Parameter(_))
                    && !reads.contains(variable)
                {
                    reads.push(variable.clone());
                }
                true
            }
            Node::Opaque(_) | Node::Placeholder(..) | Node::Overwritten { .. } => false,
            Node::Unary(Unary::Not, _) => true,
            Node::Unary(_, a) | Node::Round(_, a) | Node::Truncate(a) => {
                a.collect_value_reads(reads)
            }
            Node::Binary(
                Binary::Eq
                | Binary::Ne
                | Binary::Lt
                | Binary::Le
                | Binary::Gt
                | Binary::Ge
                | Binary::And
                | Binary::Or,
                _,
                _,
            ) => true,
            Node::Binary(_, a, b) | Node::Select(_, a, b) => {
                a.collect_value_reads(reads) && b.collect_value_reads(reads)
            }
            Node::Fold { body, .. } => body.collect_value_reads(reads),
        }
    }

    /// Bounds every value of this integer term lies within, given the bounds
    /// of its variables. `None` when the term's form bounds nothing.
    pub(crate) fn bounds(
        &self,
        variable: &impl Fn(&Variable) -> Option<(f64, f64)>,
    ) -> Option<(f64, f64)> {
        crate::cancellation::check();
        let of = |term: &Term| term.bounds(variable);
        let nonnegative = |term: &Term| of(term).filter(|(low, _)| *low >= 0.0);
        match self.node() {
            Node::Constant(value) => Some((*value, *value)),
            Node::Variable(found) => variable(found),
            Node::Round(_, a) | Node::Truncate(a) => of(a),
            Node::Select(_, a, b) => {
                let ((low_a, high_a), (low_b, high_b)) = (of(a)?, of(b)?);
                Some((low_a.min(low_b), high_a.max(high_b)))
            }
            Node::Binary(op, a, b) => match op {
                // A nonnegative operand bounds the bits of the result.
                Binary::BitAnd => match (nonnegative(a), nonnegative(b)) {
                    (Some((_, high_a)), Some((_, high_b))) => Some((0.0, high_a.min(high_b))),
                    (Some((_, high)), None) | (None, Some((_, high))) => Some((0.0, high)),
                    (None, None) => None,
                },
                Binary::Shr => nonnegative(a).map(|(_, high)| (0.0, high)),
                Binary::Remainder => of(b)
                    .filter(|(low, _)| *low > 0.0)
                    .map(|(_, high)| (0.0, high - 1.0)),
                Binary::Min | Binary::Max | Binary::Add | Binary::Sub | Binary::Mul => {
                    let ((low_a, high_a), (low_b, high_b)) = (of(a)?, of(b)?);
                    Some(match op {
                        Binary::Min => (low_a.min(low_b), high_a.min(high_b)),
                        Binary::Max => (low_a.max(low_b), high_a.max(high_b)),
                        Binary::Add => (low_a + low_b, high_a + high_b),
                        Binary::Sub => (low_a - high_b, high_a - low_b),
                        _ => {
                            let products =
                                [low_a * low_b, low_a * high_b, high_a * low_b, high_a * high_b];
                            (
                                products.iter().copied().fold(f64::INFINITY, f64::min),
                                products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                            )
                        }
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }
}

impl Variable {
    /// The entry parameter whose storage this variable reads.
    pub(crate) fn storage(&self) -> Option<&str> {
        match self {
            Variable::Element { parameter, .. }
            | Variable::Component { parameter, .. }
            | Variable::Plane { parameter, .. }
            | Variable::Parameter(parameter) => Some(parameter),
            Variable::Dimension(_) | Variable::Binder(_) => None,
        }
    }
}

/// `constant + Σ coefficient * atom`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Affine {
    pub(crate) constant: f64,
    pub(crate) terms: Vec<(Term, f64)>,
}

impl Affine {
    fn atom(term: Term) -> Affine {
        Affine {
            constant: 0.0,
            terms: vec![(term, 1.0)],
        }
    }

    fn scaled(mut self, factor: f64) -> Affine {
        self.constant *= factor;
        for (_, coefficient) in &mut self.terms {
            *coefficient *= factor;
        }
        self.terms.retain(|(_, coefficient)| *coefficient != 0.0);
        self
    }

    fn plus(mut self, other: &Affine, factor: f64) -> Affine {
        self.constant += other.constant * factor;
        for (term, coefficient) in &other.terms {
            match self.terms.iter_mut().find(|(existing, _)| existing == term) {
                Some((_, existing)) => *existing += coefficient * factor,
                None => self.terms.push((term.clone(), coefficient * factor)),
            }
        }
        self.terms.retain(|(_, coefficient)| *coefficient != 0.0);
        self
    }

    /// The coefficient of `atom`.
    pub(crate) fn coefficient(&self, atom: &Term) -> f64 {
        self.terms
            .iter()
            .find(|(term, _)| term == atom)
            .map_or(0.0, |(_, coefficient)| *coefficient)
    }

    /// This form without `atom`, as a term.
    pub(crate) fn without(&self, atom: &Term) -> Term {
        let mut term = Term::constant(self.constant);
        for (existing, coefficient) in &self.terms {
            if existing != atom {
                term = term.add(existing.clone().mul(Term::constant(*coefficient)));
            }
        }
        term
    }
}

fn without_repeats(variables: Vec<Variable>) -> Vec<Variable> {
    let mut unique: Vec<Variable> = Vec::new();
    for variable in variables {
        if !unique.contains(&variable) {
            unique.push(variable);
        }
    }
    unique
}

fn map_variable(variable: &Variable, f: &mut impl FnMut(&Term) -> Term) -> Variable {
    match variable {
        Variable::Element { parameter, index } => Variable::Element {
            parameter: parameter.clone(),
            index: index.iter().map(f).collect(),
        },
        Variable::Component {
            parameter,
            component,
            index,
        } => Variable::Component {
            parameter: parameter.clone(),
            component: *component,
            index: index.iter().map(f).collect(),
        },
        Variable::Plane {
            parameter,
            plane,
            field,
            index,
        } => Variable::Plane {
            parameter: parameter.clone(),
            plane: *plane,
            field: *field,
            index: index.iter().map(f).collect(),
        },
        other => other.clone(),
    }
}

fn constant_index(
    index: &[Term],
    value: &impl Fn(&Variable) -> Option<f64>,
    folds: &mut Vec<(u32, f64)>,
) -> Result<Vec<Term>, TermError> {
    index
        .iter()
        .map(|term| term.evaluate_with(value, folds).map(Term::constant))
        .collect()
}

fn apply_unary(op: Unary, a: f64) -> Option<f64> {
    Some(match op {
        Unary::Neg => -a,
        Unary::Abs => a.abs(),
        Unary::Exp => a.exp(),
        Unary::Exp2 => a.exp2(),
        Unary::Log => a.ln(),
        Unary::Sqrt => a.sqrt(),
        Unary::Rsqrt => 1.0 / a.sqrt(),
        Unary::Sin => a.sin(),
        Unary::Cos => a.cos(),
        Unary::Rint => a.round_ties_even(),
        Unary::Not => f64::from(u8::from(a == 0.0)),
    })
}

fn integer(value: f64) -> Option<i64> {
    (value.fract() == 0.0 && value.abs() < 9.007_199_254_740_992e15).then_some(value as i64)
}

fn apply_binary(op: Binary, a: f64, b: f64) -> Option<f64> {
    let truth = |value: bool| f64::from(u8::from(value));
    Some(match op {
        Binary::Add => a + b,
        Binary::Sub => a - b,
        Binary::Mul => a * b,
        Binary::Div => a / b,
        Binary::Quotient => {
            let (a, b) = (integer(a)?, integer(b)?);
            (b != 0).then(|| a.div_euclid(b) as f64)?
        }
        Binary::Remainder => {
            let (a, b) = (integer(a)?, integer(b)?);
            (b != 0).then(|| a.rem_euclid(b) as f64)?
        }
        Binary::Min => a.min(b),
        Binary::Max => a.max(b),
        Binary::Shl => (integer(a)? << (integer(b)? & 63)) as f64,
        Binary::Shr => (integer(a)? >> (integer(b)? & 63)) as f64,
        Binary::BitAnd => (integer(a)? & integer(b)?) as f64,
        Binary::BitOr => (integer(a)? | integer(b)?) as f64,
        Binary::BitXor => (integer(a)? ^ integer(b)?) as f64,
        Binary::Eq => truth(a == b),
        Binary::Ne => truth(a != b),
        Binary::Lt => truth(a < b),
        Binary::Le => truth(a <= b),
        Binary::Gt => truth(a > b),
        Binary::Ge => truth(a >= b),
        Binary::And => truth(a != 0.0 && b != 0.0),
        Binary::Or => truth(a != 0.0 || b != 0.0),
    })
}

/// `value` rounded to nearest-even in floating type `dtype`; integer and
/// boolean types keep the value.
pub(crate) fn round_to(dtype: DType, value: f64) -> f64 {
    match dtype {
        DType::F32 => f64::from(value as f32),
        DType::F16 => f64::from(crate::registry::f16_round(value as f32)),
        DType::BF16 => f64::from(crate::registry::bf16_round(value as f32)),
        _ => value,
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |f: &mut fmt::Formatter<'_>, index: &[Term]| {
            for (ordinal, term) in index.iter().enumerate() {
                if ordinal > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{term}")?;
            }
            Ok(())
        };
        match self {
            Variable::Plane {
                parameter,
                plane,
                field,
                index,
            } => {
                write!(f, "{parameter}.{}.{field}[", plane.name())?;
                list(f, index)?;
                f.write_str("]")
            }
            Variable::Parameter(name) | Variable::Dimension(name) => f.write_str(name),
            Variable::Binder(binder) => write!(f, "i{binder}"),
            Variable::Element { parameter, index } => {
                write!(f, "{parameter}[")?;
                list(f, index)?;
                f.write_str("]")
            }
            Variable::Component {
                parameter,
                component,
                index,
            } => {
                write!(f, "{parameter}.{}[", component.name())?;
                list(f, index)?;
                f.write_str("]")
            }
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node() {
            Node::Constant(value) => write!(f, "{value}"),
            Node::Variable(variable) => write!(f, "{variable}"),
            Node::Opaque(dependencies) => {
                f.write_str("opaque(")?;
                for (ordinal, variable) in dependencies.iter().enumerate() {
                    if ordinal > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{variable}")?;
                }
                f.write_str(")")
            }
            Node::Unary(op, a) => write!(f, "{op:?}({a})"),
            Node::Binary(op, a, b) => {
                let symbol = match op {
                    Binary::Add => "+",
                    Binary::Sub => "-",
                    Binary::Mul => "*",
                    Binary::Div => "/",
                    Binary::Quotient => "div",
                    Binary::Remainder => "mod",
                    Binary::Shl => "<<",
                    Binary::Shr => ">>",
                    Binary::BitAnd => "&",
                    Binary::BitOr => "|",
                    Binary::BitXor => "^",
                    Binary::Eq => "==",
                    Binary::Ne => "!=",
                    Binary::Lt => "<",
                    Binary::Le => "<=",
                    Binary::Gt => ">",
                    Binary::Ge => ">=",
                    Binary::And => "and",
                    Binary::Or => "or",
                    Binary::Min => return write!(f, "min({a}, {b})"),
                    Binary::Max => return write!(f, "max({a}, {b})"),
                };
                write!(f, "({a} {symbol} {b})")
            }
            Node::Select(c, t, e) => write!(f, "select({c}, {t}, {e})"),
            Node::Round(dtype, a) => write!(f, "{}({a})", dtype.name()),
            Node::Truncate(a) => write!(f, "trunc({a})"),
            Node::Overwritten { earlier, .. } => write!(f, "overwritten({earlier})"),
            Node::Placeholder(id, _) => write!(f, "carried{id}"),
            Node::Fold {
                op,
                binder,
                start,
                end,
                body,
            } => write!(f, "{op:?}(i{binder} in {start}..{end}: {body})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(name: &str) -> Term {
        Term::variable(Variable::Parameter(name.into()))
    }

    fn element(parameter: &str) -> Term {
        Term::variable(Variable::Element {
            parameter: parameter.into(),
            index: vec![Term::constant(0.0)],
        })
    }

    #[test]
    fn affine_parts_are_exact_and_refuse_what_is_not_affine() {
        let code = Term::round(DType::F32, element("codes"));
        let decoded = code
            .clone()
            .mul(named("scale"))
            .add(named("zero"))
            .div(named("divisor"));
        let (slope, intercept) = decoded.affine_in(&code).unwrap();
        let values = |variable: &Variable| match variable {
            Variable::Parameter(name) => Some(match name.as_str() {
                "scale" => 0.5,
                "zero" => -3.0,
                _ => 4.0,
            }),
            _ => None,
        };
        assert_eq!(slope.evaluate(&values), Ok(0.125));
        assert_eq!(intercept.evaluate(&values), Ok(-0.75));

        // A product of two readers of the code, a rounding boundary and a
        // transcendental use are not affine in it.
        assert!(code.clone().mul(code.clone()).affine_in(&code).is_none());
        assert!(Term::round(DType::F16, decoded.clone()).affine_in(&code).is_none());
        assert!(Term::unary(Unary::Cos, code.clone()).affine_in(&code).is_none());
        // An unresolved part that reads the code's storage may depend on it.
        let unresolved = Term::opaque(vec![Variable::Element {
            parameter: "codes".into(),
            index: Vec::new(),
        }]);
        assert!(code.clone().add(unresolved).affine_in(&code).is_none());
        let elsewhere = Term::opaque(vec![Variable::Parameter("scale".into())]);
        assert!(code.clone().add(elsewhere).affine_in(&code).is_some());
    }

    #[test]
    fn value_reads_exclude_positions_and_conditions() {
        let position = element("routes");
        let read = Term::variable(Variable::Element {
            parameter: "codes".into(),
            index: vec![position],
        });
        let term = Term::select(
            Term::binary(Binary::Gt, element("mask"), Term::constant(0.0)),
            Term::binary(Binary::BitAnd, read.clone(), Term::constant(15.0)),
            Term::constant(0.0),
        );
        let reads = term.value_reads().unwrap();
        assert_eq!(reads.len(), 1);
        assert_eq!(Term::variable(reads[0].clone()), read);
        assert!(term.add(Term::opaque(Vec::new())).value_reads().is_none());
    }

    #[test]
    fn integer_bounds_follow_masks_shifts_and_remainders() {
        let word = |variable: &Variable| {
            matches!(variable, Variable::Element { .. }).then_some((0.0, 4_294_967_295.0))
        };
        let shifted = Term::binary(Binary::Shr, element("codes"), named("shift"));
        assert_eq!(shifted.bounds(&word), Some((0.0, 4_294_967_295.0)));
        let field = Term::binary(Binary::BitAnd, shifted, Term::constant(255.0));
        assert_eq!(field.bounds(&word), Some((0.0, 255.0)));
        let remainder = Term::binary(Binary::Remainder, named("anything"), Term::constant(16.0));
        assert_eq!(remainder.bounds(&word), Some((0.0, 15.0)));
        assert_eq!(field.sub(remainder).bounds(&word), Some((-15.0, 255.0)));
        assert_eq!(named("anything").bounds(&word), None);
    }

    #[test]
    fn an_overwritten_read_is_unresolved_until_taken_at_its_earlier_content() {
        let read = Term::overwritten(vec![Variable::Binder(3)], element("codes"));
        assert!(!read.is_resolved());
        assert_eq!(read.evaluate(&|_| Some(1.0)), Err(TermError::Unresolved));
        assert!(read.mentions(&Term::variable(Variable::Binder(3))));
        let doubled = read.mul(Term::constant(2.0)).unshadowed();
        assert!(doubled.is_resolved());
        assert_eq!(doubled.evaluate(&|_| Some(1.5)), Ok(3.0));
    }

    #[test]
    fn folds_bind_their_own_coordinate_and_unresolved_parts_do_not_evaluate() {
        let sum = Term::fold(
            Fold::Sum,
            7,
            Term::constant(0.0),
            Term::variable(Variable::Dimension("K".into())),
            Term::variable(Variable::Binder(7)),
        );
        let value = sum
            .evaluate(&|variable| match variable {
                Variable::Dimension(name) if name == "K" => Some(4.0),
                _ => None,
            })
            .unwrap();
        assert_eq!(value, 6.0);
        assert_eq!(
            Term::opaque(vec![Variable::Parameter("x".into())]).evaluate(&|_| Some(1.0)),
            Err(TermError::Unresolved)
        );
    }

    #[test]
    fn the_levels_adjacent_to_a_code_are_reported_separately() {
        let table = CodeLevels::of([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 2.0]);
        assert_eq!(
            table,
            CodeLevels::Table(vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        );
        let at = |value| table.neighbours(value);
        assert_eq!(at(0.0), Neighbours { below: None, above: Some(0.5) });
        assert_eq!(at(2.0), Neighbours { below: Some(1.5), above: Some(3.0) });
        assert_eq!(at(6.0), Neighbours { below: Some(4.0), above: None });

        let integers = CodeLevels::of((-8..8).map(f64::from));
        assert_eq!(integers, CodeLevels::Integers { min: -8, max: 7 });
        let at = |value| integers.neighbours(value);
        assert_eq!(at(-8.0), Neighbours { below: None, above: Some(-7.0) });
        assert_eq!(at(0.0), Neighbours { below: Some(-1.0), above: Some(1.0) });
        assert_eq!(at(7.0), Neighbours { below: Some(6.0), above: None });
    }
}
