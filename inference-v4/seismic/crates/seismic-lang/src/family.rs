//! Occurrence applicability: which authored implementations of one contract
//! are applicable at one call occurrence on one effective target.
//!
//! Portable bodies and same-backend lowerings are peer semantic alternatives,
//! a backend helper is reachable only from the same backend, and every
//! candidate must satisfy its `where` predicates (under the occurrence's bound
//! shapes), its caller-element requirements, and the effective capability
//! environment.

use crate::intrinsics::{CapabilityId, NumericalTransfer};
use crate::logical::ImplementationKind;
use crate::sir::{
    CandidateBinding, CheckedCall, DefId, DefKind, Definition, IntrinsicUse, Predicate, Program,
};
use crate::sym::Sym;
use crate::types::{Elem, ExtentExpr};
use std::collections::BTreeSet;

/// One candidate of an occurrence: a definition plus how its generic
/// parameters bind at this call.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub definition: DefId,
    pub kind: ImplementationKind,
    /// Callee shape parameter -> symbolic extent in the caller's space.
    pub shape_args: Vec<(String, Sym)>,
    /// Callee element parameter -> concrete element.
    pub elem_args: Vec<(String, Elem)>,
    /// Caller element parameters that must equal these concrete elements.
    pub requires_elems: Vec<(String, Elem)>,
    /// Argument expression ordinal for each candidate parameter. The logical
    /// boundary requires every candidate of an occurrence to share the
    /// contract's positional order.
    pub arg_order: Vec<usize>,
}

impl Candidate {
    fn of(program: &Program, binding: &CandidateBinding) -> Option<Candidate> {
        let definition = program.definition(binding.definition);
        let kind = kind_of(program, definition)?;
        Some(Candidate {
            definition: binding.definition,
            kind,
            shape_args: binding.shape_args.clone(),
            elem_args: binding.elem_args.clone(),
            requires_elems: binding.requires_elems.clone(),
            arg_order: binding.arg_order.clone(),
        })
    }
}

fn kind_of(program: &Program, definition: &Definition) -> Option<ImplementationKind> {
    let _ = program;
    match definition.kind {
        DefKind::Body { target: None } => Some(ImplementationKind::PortableBody),
        DefKind::Body { target: Some(_) } => Some(ImplementationKind::BackendBody),
        DefKind::Lower { .. } => Some(ImplementationKind::Lowering),
    }
}

/// Whether a definition is reachable on `target`.
fn reachable_on(kind: ImplementationKind, definition: &Definition, target: &str) -> bool {
    match kind {
        ImplementationKind::PortableBody => true,
        ImplementationKind::BackendBody | ImplementationKind::Lowering => {
            definition.kind.target() == Some(target)
        }
    }
}

/// The candidates a checked call occurrence carries, filtered to the effective
/// target (portable bodies stay; backend bodies and lowerings apply only on
/// their own backend).
pub fn candidates_of_call(program: &Program, call: &CheckedCall, target: &str) -> Vec<Candidate> {
    call.bindings
        .iter()
        .filter_map(|binding| Candidate::of(program, binding))
        .filter(|candidate| {
            reachable_on(
                candidate.kind,
                program.definition(candidate.definition),
                target,
            )
        })
        .collect()
}

/// The candidates of the entry itself: every body and lowering of the family
/// reachable on the target, plus the definitions unreachable on it (with the
/// reason, for applicability reports). The entry's generic parameters bind to
/// the workload's concrete shapes/elements.
pub fn entry_candidates(
    program: &Program,
    family: usize,
    target: &str,
) -> (Vec<Candidate>, Vec<(DefId, String)>) {
    let family = &program.families[family];
    let mut out = Vec::new();
    let mut rejected = Vec::new();
    for id in family.bodies.iter().chain(family.lowerings.iter()) {
        let definition = program.definition(*id);
        let Some(kind) = kind_of(program, definition) else {
            continue;
        };
        if !reachable_on(kind, definition, target) {
            rejected.push((
                *id,
                format!(
                    "`{}` is authored for `{}` and is not reachable on `{}`",
                    definition.name,
                    definition.kind.target().unwrap_or_default(),
                    target
                ),
            ));
            continue;
        }
        let shape_args = definition
            .shape_params
            .iter()
            .map(|p| (p.clone(), Sym::param(p)))
            .collect();
        let elem_args = definition
            .elem_params
            .iter()
            .map(|p| (p.clone(), Elem::Param(p.clone())))
            .collect();
        let arg_order = (0..definition.params.len()).collect();
        out.push(Candidate {
            definition: *id,
            kind,
            shape_args,
            elem_args,
            requires_elems: Vec::new(),
            arg_order,
        });
    }
    (out, rejected)
}

/// One applicable alternative of an occurrence.
#[derive(Clone, Debug)]
pub struct ResolvedAlternative {
    pub candidate: Candidate,
    pub required_capabilities: BTreeSet<CapabilityId>,
    pub authored_numerical_effects: Vec<NumericalTransfer>,
}

/// The applicability answer for one occurrence.
#[derive(Clone, Debug, Default)]
pub struct OccurrenceAlternatives {
    pub alternatives: Vec<ResolvedAlternative>,
    /// Inapplicable definitions with the reason, for inspection and reports.
    pub rejected: Vec<(DefId, String)>,
}

/// Evaluate applicability of `candidates` at one occurrence.
///
/// `caller_shape` answers the concrete value of one caller shape parameter
/// (`None` when it is runtime-determined or unknown); `caller_elem` answers
/// the concrete element of one caller element parameter. A predicate over a
/// value that is not concretely known is undecidable, so the candidate is
/// inapplicable.
pub fn applicable(
    program: &Program,
    target: &str,
    supports_intrinsic: &dyn Fn(&IntrinsicUse) -> Result<(), String>,
    candidates: &[Candidate],
    caller_shape: &dyn Fn(&str) -> Option<i64>,
    caller_elem: &dyn Fn(&str) -> Option<Elem>,
) -> OccurrenceAlternatives {
    let _ = target;
    let mut out = OccurrenceAlternatives::default();
    for candidate in candidates {
        let definition = program.definition(candidate.definition);
        if let Err(reason) = predicates_hold(definition, candidate, caller_shape) {
            out.rejected.push((candidate.definition, reason));
            continue;
        }
        if let Err(reason) = elements_hold(candidate, caller_elem) {
            out.rejected.push((candidate.definition, reason));
            continue;
        }
        let mut required = BTreeSet::new();
        for capability in &definition.requires {
            required.insert(capability.clone());
        }
        let mut supported = true;
        for use_ in &definition.intrinsic_uses {
            required.insert(use_.id.capability.clone());
            if let Err(reason) = supports_intrinsic(use_) {
                out.rejected.push((
                    candidate.definition,
                    format!("capability `{}`: {reason}", use_.id.path()),
                ));
                supported = false;
                break;
            }
        }
        if !supported {
            continue;
        }
        out.alternatives.push(ResolvedAlternative {
            candidate: candidate.clone(),
            required_capabilities: required,
            authored_numerical_effects: authored_effects(definition, &definition.intrinsic_uses),
        });
    }
    out
}

/// Substitute the candidate's shape bindings into the definition's `where`
/// predicates and decide them under the caller's concrete shapes.
fn predicates_hold(
    definition: &Definition,
    candidate: &Candidate,
    caller_shape: &dyn Fn(&str) -> Option<i64>,
) -> Result<(), String> {
    for predicate in &definition.predicates {
        let bound = |name: &str| -> Option<Sym> {
            candidate
                .shape_args
                .iter()
                .find(|(p, _)| p == name)
                .map(|(_, sym)| sym.clone())
        };
        let substituted = match predicate {
            Predicate::NonNegative(expr) => substitute(expr, &bound),
            Predicate::Zero(expr) => substitute(expr, &bound),
            Predicate::NonZero(expr) => substitute(expr, &bound),
        };
        let eval = |expr: &Sym| -> Result<i64, String> {
            expr.eval(caller_shape).ok_or_else(|| {
                format!(
                    "the `where` predicate `{expr}` depends on a value that is not a concrete shape at this occurrence"
                )
            })
        };
        match predicate {
            Predicate::NonNegative(expr) => {
                let value = eval(&substituted).map_err(|reason| {
                    format!("predicate `0 <= {expr}` is undecidable: {reason}")
                })?;
                if value < 0 {
                    return Err(format!(
                        "predicate `0 <= {expr}` fails: the bound value is {value}"
                    ));
                }
            }
            Predicate::Zero(expr) => {
                let value = eval(&substituted).map_err(|reason| {
                    format!("predicate `{expr} == 0` is undecidable: {reason}")
                })?;
                if value != 0 {
                    return Err(format!(
                        "predicate `{expr} == 0` fails: the bound value is {value}"
                    ));
                }
            }
            Predicate::NonZero(expr) => {
                let value = eval(&substituted).map_err(|reason| {
                    format!("predicate `{expr} != 0` is undecidable: {reason}")
                })?;
                if value == 0 {
                    return Err(format!(
                        "predicate `{expr} != 0` fails: the bound value is 0"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Substitute the candidate's bound shape parameters into one predicate
/// expression; unbound parameters are kept symbolic (and therefore
/// undecidable).
fn substitute(expr: &Sym, bound: &dyn Fn(&str) -> Option<Sym>) -> Sym {
    let mut out = Sym::constant(0);
    for (monomial, coefficient) in expr.monomials() {
        let mut term = Sym::constant(coefficient);
        for (atom, power) in monomial {
            let atom_value = match atom {
                crate::sym::Atom::Param(name) => {
                    bound(name).unwrap_or_else(|| Sym::atom(atom.clone()))
                }
                crate::sym::Atom::Quot(..) | crate::sym::Atom::Rem(..) => Sym::atom(atom.clone()),
            };
            for _ in 0..*power {
                term = term.mul(&atom_value);
            }
        }
        out = out.add(&term);
    }
    out
}

/// The caller element parameters this candidate requires must be bound to
/// exactly these concrete elements.
fn elements_hold(
    candidate: &Candidate,
    caller_elem: &dyn Fn(&str) -> Option<Elem>,
) -> Result<(), String> {
    for (param, required) in &candidate.requires_elems {
        match caller_elem(param) {
            None => {
                return Err(format!(
                    "element parameter `{param}` is not concrete at this occurrence"
                ))
            }
            Some(actual) if &actual != required => {
                return Err(format!(
                    "requires `{param} = {required}` but the occurrence binds `{actual}`"
                ));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// The authored numerical effects of one alternative. The portable reference
/// body contributes the registry's reference numerics; its capability uses
/// carry their registry transfer. A non-reference authored implementation
/// (backend body or lowering) starts with `Unknown` whole-candidate
/// equivalence.
fn authored_effects(definition: &Definition, uses: &[IntrinsicUse]) -> Vec<NumericalTransfer> {
    match definition.kind {
        DefKind::Body { target: None } => uses
            .iter()
            .map(|use_| {
                crate::intrinsics::lookup(
                    &use_.id.capability.backend,
                    &use_.id.capability.name,
                    &use_.id.name,
                )
                .into_iter()
                .find(|signature| {
                    signature.arguments == use_.arguments && signature.result == use_.result
                })
                .map(|signature| signature.numerical)
                .unwrap_or(NumericalTransfer::Capability {
                    signature: use_.id.clone(),
                    bound: None,
                })
            })
            .collect(),
        DefKind::Body { target: Some(_) } | DefKind::Lower { .. } => {
            vec![NumericalTransfer::Unknown {
                reason: format!(
                    "`{}` is a non-reference authored implementation; whole-candidate numerical equivalence starts unknown",
                    definition.name
                ),
            }]
        }
    }
}

/// The concrete extent environment of one specialization, for predicate
/// evaluation: shape parameter name -> concrete value when static.
pub fn static_shape_of(extent: &ExtentExpr) -> Option<i64> {
    match extent {
        ExtentExpr::Static(n) => Some(*n as i64),
        _ => None,
    }
}
