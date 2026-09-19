//! The joint selection family of one entry on one target for one workload:
//! guarded static call occurrences with their applicable authored implementations,
//! numerical sites, and ordered execution-unit sequences. Target-neutral.
//!
//! This is a finite description of what the authors supplied. It is not a rewriting
//! space: nothing here invents calls, producers, stages, partitions or groupings.
//! Dynamic repetition (visits, elements, tokens) never creates occurrences or sites.
//!
//! Backends add site domains, legality constraints, fusion intervals and cost factors
//! (`seismic-compiler::selection::Backend`); the solver picks a `Witness`;
//! `instantiate` reproduces exactly that witness.

use super::sir::{CallId, DefId, Program};
use super::types::{Elem, RegionId, SliceId};
use std::collections::BTreeMap;

mod construct;
mod normalize;

pub use construct::construct;

/// Which numerical latitude selection may use.
///
/// `Exact` removes the latitude admitted contracts grant: within any contract family whose
/// definitions are `admit`, a candidate is inapplicable if it is a target `Lower` body (a
/// lowering of an admitted contract may reassociate) or its body contains a
/// `Reduce { unordered: true }` anywhere; the rejection reason is "exact numerics
/// requested". Everything else is unchanged, including coverage: a library keeps an ordered
/// portable body in every admitted family or exact selection reports missing coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Numerics {
    #[default]
    Admitted,
    Exact,
}

/// Concrete semantic specialization of an entry. Every field is part of selection identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Workload {
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
    pub numerics: Numerics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemplateId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OccurrenceId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SiteId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceId(pub u32);

/// A candidate of an occurrence: `(occurrence, ordinal into its candidates)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CandidateRef {
    pub occurrence: OccurrenceId,
    pub candidate: u32,
}

#[derive(Clone, Debug)]
pub struct Family {
    pub entry: String,
    pub target: String,
    pub workload: Workload,
    /// Interned specialized bodies. Two occurrences of one definition under equal
    /// bindings share a template and keep separate choices, sites and costs.
    pub templates: Vec<Template>,
    /// `occurrences[0]` is the entry: its candidates are all applicable portable bodies,
    /// same-target function bodies, and same-target lowerings.
    pub occurrences: Vec<Occurrence>,
    pub sites: Vec<Site>,
    /// `(refinement, refined)`: the width site of a binder over an enclosing slice of the same
    /// body (`parallel [r] in rows:`) and the width site of that slice. A refinement
    /// partitions the pieces of the refined binder, so its value divides the refined value.
    pub refinements: Vec<(SiteId, SiteId)>,
    pub sequences: Vec<Sequence>,
    /// Supported-looking candidates the construction could not analyze. Never silently
    /// dropped: selection reports them and cannot claim full-family coverage.
    pub obligations: Vec<Obligation>,
}

/// One definition specialized to concrete semantic shapes/elements and a pattern of
/// structural shape arguments (`structural` lists shape parameters bound to caller slices).
#[derive(Clone, Debug)]
pub struct Template {
    pub id: TemplateId,
    pub definition: DefId,
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
    pub structural: Vec<String>,
    /// Shape parameters bound to a runtime-valued semantic extent of the caller (the length
    /// of a runtime-bounded range such as visible history). Semantic, numerically usable,
    /// not static and never a site: the value is the caller's extent expression at the call
    /// occurrence. Predicates over such a parameter are undecidable at selection, so a
    /// candidate whose applicability depends on one is inapplicable.
    pub dynamic: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Occurrence {
    pub id: OccurrenceId,
    /// The candidate whose body contains this call; `None` for the entry.
    pub parent: Option<CandidateRef>,
    /// The call within the parent's template body; `None` for the entry.
    pub call: Option<CallId>,
    pub family: usize,
    /// Every applicable authored implementation. Empty means missing coverage on this path
    /// (the parent candidate is then unselectable; at the entry it is an error).
    pub candidates: Vec<Candidate>,
    /// Inapplicable definitions with the reason, for inspection.
    pub rejected: Vec<(DefId, String)>,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub template: TemplateId,
    /// The function or lowering declaration that contributes this candidate.
    pub via: DefId,
    /// Caller slice bound to each structural shape parameter of the template.
    pub structural: Vec<(String, SiteRef)>,
    /// Applicability that depends on numbers: holds for the selected site values or the
    /// candidate is unselectable.
    pub requirements: Vec<Requirement>,
    pub children: Vec<OccurrenceId>,
    pub sites: Vec<SiteId>,
    pub sequences: Vec<SequenceId>,
}

/// A site visible from a candidate: its own, or one owned by an ancestor candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SiteRef(pub SiteId);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Requirement {
    /// Site value is a multiple of `unit` (atom / packet alignment from a `where`).
    Multiple {
        site: SiteId,
        unit: i64,
    },
    AtLeast {
        site: SiteId,
        value: i64,
    },
    AtMost {
        site: SiteId,
        value: i64,
    },
    Equal {
        site: SiteId,
        value: i64,
    },
    /// `full(P)`: the site value divides the slice's parent extent.
    Divides {
        site: SiteId,
        extent: i64,
    },
}

#[derive(Clone, Debug)]
pub struct Site {
    pub id: SiteId,
    pub owner: CandidateRef,
    pub kind: SiteKind,
    /// Static semantic extent of the partitioned domain when known (upper bound for a
    /// runtime extent). Widths range over `1..=extent`; backends narrow the domain.
    pub extent: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SiteKind {
    /// Width of a `parallel`/`ordered`/`pipeline` binder over a domain or refinement.
    Width { region: RegionId, slice: SliceId },
    /// Partition count of a `merge` region axis.
    Parts { region: RegionId, slice: SliceId },
}

/// The normalized execution units of one block, in authored order.
#[derive(Clone, Debug)]
pub struct Sequence {
    pub id: SequenceId,
    pub owner: CandidateRef,
    /// Path of the block inside the template body: region/stage/branch steps from the root.
    pub scope: Vec<ScopeStep>,
    pub units: Vec<Unit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeStep {
    Region(RegionId),
    Stage(usize),
    Then(usize),
    Else(usize),
    Loop(usize),
}

#[derive(Clone, Debug)]
pub struct Unit {
    /// Statement ordinals of the normalized block this unit covers (contiguous).
    pub statements: std::ops::Range<usize>,
    pub kind: UnitKind,
    /// Completion that must hold before the next unit starts (stage/region boundary at
    /// this scope). A fused interval may cross it only with a realization preserving it.
    pub completion_after: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitKind {
    /// Tile-valued binding or state update computed elementwise over its axes.
    Elementwise,
    /// Reduction, scalar work, or other local computation that is not elementwise.
    Local,
    /// Static call occurrence whose implementation is selected from its linked family.
    Call(OccurrenceId),
    Publish,
    Region(RegionId),
    Stage(usize),
}

#[derive(Clone, Debug)]
pub struct Obligation {
    pub occurrence: OccurrenceId,
    pub definition: DefId,
    pub reason: String,
}

/// A complete joint assignment: one implementation per active occurrence, a value per
/// active site, and a contiguous exact cover per active sequence. Inactive occurrences,
/// sites and sequences are absent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Witness {
    pub choices: BTreeMap<OccurrenceId, u32>,
    pub sites: BTreeMap<SiteId, i64>,
    /// Half-open unit intervals `[start, end)`, ascending, covering every unit once.
    pub covers: BTreeMap<SequenceId, Vec<(u32, u32)>>,
}

impl Family {
    pub fn occurrence(&self, id: OccurrenceId) -> &Occurrence {
        &self.occurrences[id.0 as usize]
    }

    pub fn candidate(&self, r: CandidateRef) -> &Candidate {
        &self.occurrence(r.occurrence).candidates[r.candidate as usize]
    }

    pub fn template(&self, id: TemplateId) -> &Template {
        &self.templates[id.0 as usize]
    }

    /// Whether `candidate` is active under `witness` (it and all its ancestors selected).
    pub fn active(&self, witness: &Witness, candidate: CandidateRef) -> bool {
        let mut current = Some(candidate);
        while let Some(c) = current {
            if witness.choices.get(&c.occurrence) != Some(&c.candidate) {
                return false;
            }
            current = self.occurrence(c.occurrence).parent;
        }
        true
    }

    /// Structural validity of a witness against this family: exactly one applicable choice
    /// per active occurrence, values for exactly the active sites, requirements hold, and
    /// exact contiguous covers for exactly the active sequences. Backend legality is separate.
    pub fn validate(&self, witness: &Witness) -> Result<(), String> {
        normalize::validate(self, witness)
    }
}

/// Program handle used by consumers that need definition bodies for a template.
pub fn body<'a>(
    program: &'a Program,
    family: &Family,
    template: TemplateId,
) -> &'a super::sir::Body {
    &program
        .definition(family.template(template).definition)
        .body
}
