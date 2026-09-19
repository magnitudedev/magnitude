//! Unresolved, solver-independent computation. Construction retains local
//! alternatives; it never chooses a path through completed lowerings.
//!
//! A coverage obligation is part of the result, not a discarded alternative.
//! Consumers must export every obligation before claiming complete coverage.
mod build;
mod materialize;
mod expand;
mod registry;

pub use registry::{DecisionClass, Migration};
use crate::{ir::*, lowered_ir::*, sym::{Atom, Sym}};
use std::collections::{BTreeMap, BTreeSet};

/// Identity includes the checked definition, every call/body occurrence, and
/// the defining operation. A lexical ordinal only disambiguates equal siblings.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OccurrenceId {
    pub definition: String,
    pub topology: Vec<Origin>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Origin {
    pub role: String,
    pub sibling: usize,
    pub operation: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DecisionId {
    pub occurrence: OccurrenceId,
    pub class: DecisionClass,
    pub slot: usize,
}

/// Original ordinals, not ordinal paths through a traversal.
pub type Assignment = BTreeMap<DecisionId, usize>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Guard {
    pub choices: Vec<(DecisionId, usize)>,
    pub predicates: Vec<Predicate>,
    /// Each inner list is a disjunction; all lists must hold. This preserves
    /// one shared preparation window when any operand requests window storage.
    pub one_of: Vec<Vec<(DecisionId, usize)>>,
}
/// A derived presence condition over ORIGINAL numeric choices. It is not an
/// additional implementation choice and cannot narrow its parameter domains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Predicate {
    pub nonnegative: Sym,
    pub parameters: Vec<(String, DecisionId, Alternatives)>,
}
impl Guard {
    pub fn active(&self, assignment: &Assignment) -> Result<bool, String> {
        // Parents precede children. An inactive parent makes descendants
        // inactive without demanding assignments for their local decisions.
        let mut missing = None;
        for (decision, ordinal) in &self.choices {
            match assignment.get(decision) {
                Some(value) if value == ordinal => {},
                Some(_) => return Ok(false),
                None => missing = Some(format!("missing activation decision {decision:?}")),
            }
        }
        for predicate in &self.predicates {
            let mut values = BTreeMap::new();
            for (name, decision, domain) in &predicate.parameters {
                if let Some(&ordinal) = assignment.get(decision) {
                    let value = domain.numeric().and_then(|numeric| numeric.value(ordinal))
                        .ok_or_else(|| format!("invalid numeric activation decision {decision:?}"))?;
                    values.insert(name.as_str(), value);
                }
            }
            match predicate.nonnegative.eval(&|name| values.get(name).copied()) {
                Some(value) if value < 0 => return Ok(false),
                Some(_) => {},
                None => missing = Some("missing original numeric activation parameter".into()),
            }
        }
        for alternatives in &self.one_of {
            if alternatives.iter().any(|(decision, ordinal)| assignment.get(decision) == Some(ordinal)) { continue; }
            if alternatives.iter().all(|(decision, _)| assignment.contains_key(decision)) { return Ok(false); }
            missing = Some("missing categorical activation operand".into());
        }
        if let Some(missing) = missing { return Err(missing); }
        Ok(true)
    }
    fn with(&self, decision: DecisionId, ordinal: usize) -> Self {
        let mut next = self.clone();
        next.choices.push((decision, ordinal));
        next
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId(pub DecisionId);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumericParameter {
    pub id: ParameterId,
    /// The atom appears in the shared typed `Sym` expressions below. Exporters
    /// bind it directly to this original decision's numeric value.
    pub atom: Atom,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilyDecision {
    pub id: DecisionId,
    pub domain: Decision,
    pub guard: Guard,
    pub numeric: Option<NumericParameter>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverageObligation {
    pub occurrence: OccurrenceId,
    pub guard: Guard,
    pub class: DecisionClass,
    pub reason: String,
}

/// A shape precondition of an alternative. Runtime piece variables are
/// universally quantified over their declared repeat domain, never choices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirement {
    pub guard: Guard,
    pub nonnegative: Sym,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub usize);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    pub reads: BTreeSet<VarId>,
    pub writes: BTreeSet<VarId>,
    pub tensor_effect: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub occurrence: OccurrenceId,
    pub guard: Guard,
    pub effects: Effects,
    pub kind: RegionKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RegionKind {
    Sequence(Vec<RegionId>),
    Statement(Stmt),
    Choice { decision: DecisionId, arms: Vec<RegionId> },
    /// The header is the original typed control operation with an empty body.
    Repeated { header: Stmt, body: RegionId, repetition: Repetition },
    /// Static copies of one ordinary body. Numeric count belongs to the source
    /// decision, and every copy substitutes the declared index in that body.
    /// This is code replication, not a serial runtime loop.
    Replicated { index: VarId, count: Sym, body: RegionId },
    Conditional { header: Stmt, then: RegionId, els: RegionId },
    Stream { header: Stmt, body: RegionId, capacity: DecisionId, geometry: StreamGeometry },
    Reduction(Box<ReductionFamily>),
}

/// The original merge/step computation and local ownership choices are shared
/// by every tree. Segment counts and tails depend on the original capacity.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionFamily {
    pub operation: crate::reduction::structured::Reduction,
    pub tree: DecisionId,
    pub segments: Vec<(Guard, DecisionId, StreamGeometry)>,
    pub operands: Vec<(usize, DecisionId)>,
    pub state: Option<DecisionId>,
    pub decomposition: Option<ReductionDecomposition>,
    pub dynamic_decomposition: Option<DynamicReductionDecomposition>,
    pub callbacks: Vec<ReductionCallback>,
    pub preparation: Vec<FoldPreparationFamily>,
    /// Compact ordered-tree coverage. Each active frontier is the exclusive
    /// leaf endpoint emitted immediately before one merge. Together these
    /// frontiers encode every ordered binary tree without interval domains.
    pub explicit: Option<ExplicitReductionFamily>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExplicitReductionFamily {
    pub guard: Guard,
    pub leaves: Sym,
    pub merges: Vec<ExplicitReductionMerge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExplicitReductionMerge {
    pub guard: Guard,
    pub frontier: DecisionId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FoldPreparationFamily {
    pub guard: Guard,
    pub segment: Sym,
    pub inputs: Vec<(usize, DecisionId)>,
    pub window: Option<DecisionId>,
    pub traversals: Vec<(Guard, DecisionId)>,
    pub packets: Vec<PacketPreparationFamily>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PacketPreparationFamily {
    pub guard: Guard,
    pub input: usize,
    pub width: DecisionId,
    pub decoder: DecisionId,
    pub coefficients: DecisionId,
    pub words: DecisionId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReductionCallback {
    pub role: CallbackRole,
    pub body: RegionId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackRole { Merge, Step }

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicReductionDecomposition {
    pub guard: Guard,
    pub decision: DecisionId,
    pub geometry: StreamGeometry,
    /// Evaluates the already captured logical source extent; it never repeats
    /// the source endpoint expressions or takes a new source-state snapshot.
    pub header: Stmt,
    pub operation: crate::reduction::structured::Reduction,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReductionDecomposition {
    pub guard: Guard,
    pub decision: DecisionId,
    pub geometry: StreamGeometry,
    pub index: VarId,
    /// The original unpartitioned operation applies when this expression is
    /// nonnegative. The other branch contains full pieces and an optional tail.
    pub whole_condition: Sym,
    pub full: ReductionPiece,
    pub tail: ReductionPiece,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReductionPiece {
    pub setup: Vec<Stmt>,
    pub operation: crate::reduction::structured::Reduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepeatOrder { Serial, Parallel }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repetition {
    pub order: RepeatOrder,
    pub indices: Vec<VarId>,
    pub counts: Vec<Sym>,
    /// Values written and subsequently consumed across visits. They belong to
    /// the repeat occurrence, not merely to a shared body definition.
    pub carried: BTreeSet<VarId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamGeometry {
    pub parameter: NumericParameter,
    pub extent: Sym,
    pub capacity: Sym,
    pub complete_pieces: Sym,
    pub tail_extent: Sym,
    pub visits: Sym,
    pub piece: Atom,
}
impl StreamGeometry {
    /// Preserve the endpoint correlation of the final partial piece. For the
    /// positive capacity domain, this equals capacity * floor(extent/capacity),
    /// while adding tail_extent cancels symbolically to the original extent.
    pub fn tail_start(&self) -> Sym { self.extent.sub(&self.tail_extent) }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Value(VarId),
    Effect,
    /// The source and target denote adjacent dynamic visits of one body.
    Recurrence(VarId),
    /// Source order between independent work domains is a phase boundary.
    Phase,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependency {
    pub from: RegionId,
    pub to: RegionId,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub producer: RegionId,
    pub variable: VarId,
    pub source: Expr,
    pub guard: Guard,
    pub consumers: Vec<RegionId>,
}

/// Immutable owner shared by model export and selected reconstruction.
#[derive(Clone, Debug)]
pub struct ExecutionFamily {
    template: LoweredIr,
    root: RegionId,
    regions: Vec<Region>,
    decisions: Vec<FamilyDecision>,
    obligations: Vec<CoverageObligation>,
    requirements: Vec<Requirement>,
    dependencies: Vec<Dependency>,
    snapshots: Vec<Snapshot>,
}

impl ExecutionFamily {
    pub fn new(input: super::alternatives::Specialization<'_>) -> Result<Self, String> {
        build::construct(input)
    }
    /// An explicitly supplied, already selected source artifact has no source
    /// optimization freedom. Backend decisions remain unresolved by its owner.
    pub fn from_lowered(function: impl std::borrow::Borrow<LoweredIr>) -> Result<Self, String> {
        build::fixed(function.borrow())
    }
    /// Specialized ABI and union of local variables. The body is deliberately
    /// absent: no concrete body has been selected at this boundary.
    pub fn template(&self) -> &LoweredIr { &self.template }
    pub fn root(&self) -> RegionId { self.root }
    pub fn regions(&self) -> &[Region] { &self.regions }
    pub fn decisions(&self) -> &[FamilyDecision] { &self.decisions }
    pub fn obligations(&self) -> &[CoverageObligation] { &self.obligations }
    pub fn requirements(&self) -> &[Requirement] { &self.requirements }
    pub fn dependencies(&self) -> &[Dependency] { &self.dependencies }
    pub fn snapshots(&self) -> &[Snapshot] { &self.snapshots }
    pub fn instantiate(&self, assignment: &Assignment) -> Result<LoweredIr, String> {
        materialize::instantiate(self, assignment)
    }
    /// Expand semantic reductions and streamed transfers once into ordinary
    /// guarded typed regions over the ORIGINAL source parameters. Backend
    /// exporters and reconstruction consume this same immutable owner.
    pub fn expanded(&self) -> Result<Self, String> { expand::expand(self) }
}
