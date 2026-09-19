//! Construction of the joint selection family: the finite description of what the authors
//! supplied for one entry, target and workload. No ranking, no invented structure.
//!
//! The occurrence tree is built first (candidates that cannot be selected never enter it),
//! then numbered in pre-order and emitted with compact site and sequence ids.

use super::normalize::{owning_slice, sequences, BlockUnits};
use super::{
    Candidate, CandidateRef, Family, Obligation, Occurrence, OccurrenceId, Requirement,
    Sequence, SequenceId, Site, SiteId, SiteKind, SiteRef, Template, TemplateId, UnitKind,
    Workload,
};
use crate::repr;
use crate::precision::NumericalEffect;
use crate::sir::{
    Body, CallId, CallSite, DefId, DefKind, Definition, ExprKind, Index, Predicate, Program,
    SliceParent,
};
use crate::span::line_col;
use crate::sym::Sym;
use crate::types::{Elem, Extent, RegionId, SliceId};
use applicability::{Binding, Bounds, SiteExtent};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

mod applicability;
pub(super) mod walk;

/// Build the family of linked `entry` on `target` for `workload`. Unions applicable
/// portable functions, functions for this target, and lowerings for this target; evaluates
/// applicability predicates, interns templates, and creates guarded
/// occurrences recursively (finite expansion checked), sites and unit sequences.
pub fn construct(
    program: &Program,
    entry: &str,
    target: &str,
    workload: &Workload,
) -> Result<Family, String> {
    let contract = program.family_index(entry)?;
    let mut builder = Builder {
        program,
        target,
        templates: Vec::new(),
        interned: HashMap::new(),
        sites: Vec::new(),
        path: Vec::new(),
        units: HashMap::new(),
    };
    let mut root = builder.occurrence(contract, None, &Source::Workload(workload))?;
    if root.candidates.is_empty() {
        let shapes: Vec<String> = workload
            .shapes
            .iter()
            .map(|(name, value)| format!("{name} = {value}"))
            .collect();
        return Err(format!(
            "missing coverage: no implementation of `{entry}` is selectable on `{target}` for [{}]: {}",
            shapes.join(", "),
            builder.coverage(&root)
        ));
    }
    number(&mut root, &mut 0);
    let mut family = Family {
        entry: entry.to_string(),
        target: target.to_string(),
        workload: workload.clone(),
        allow_numerical_effects: matches!(workload.precision, crate::precision::PrecisionPolicy::Unconstrained),
        templates: Vec::new(),
        occurrences: Vec::new(),
        sites: Vec::new(),
        refinements: Vec::new(),
        sequences: Vec::new(),
        obligations: Vec::new(),
    };
    builder.emit(root, None, &mut family, &mut BTreeMap::new())?;
    family.templates = builder.templates;
    family.refinements = refinements(program, &family);
    Ok(family)
}

/// Why a definition is not a candidate of an occurrence.
struct Reject {
    reason: String,
    /// The construction could not analyze a supported-looking candidate (an obligation),
    /// as opposed to known inapplicability.
    unanalyzed: bool,
    /// Obligations of the removed subtree; they stay reported on the surviving occurrence.
    obligations: Vec<(DefId, String)>,
}

impl Reject {
    fn inapplicable(reason: String) -> Reject {
        Reject {
            reason,
            unanalyzed: false,
            obligations: Vec::new(),
        }
    }

    fn unanalyzed(reason: String) -> Reject {
        Reject {
            reason,
            unanalyzed: true,
            obligations: Vec::new(),
        }
    }
}

/// A fatal construction error outside, a rejected candidate inside.
type Attempt<T> = Result<Result<T, Reject>, String>;

struct OccurrenceNode {
    id: OccurrenceId,
    call: Option<CallId>,
    family: usize,
    candidates: Vec<CandidateNode>,
    rejected: Vec<(DefId, String)>,
    obligations: Vec<(DefId, String)>,
}

impl OccurrenceNode {
    fn reject(&mut self, definition: DefId, reject: Reject) {
        if reject.unanalyzed {
            self.obligations.push((definition, reject.reason.clone()));
        }
        self.obligations.extend(reject.obligations);
        if !self
            .rejected
            .iter()
            .any(|(d, r)| *d == definition && *r == reject.reason)
        {
            self.rejected.push((definition, reject.reason));
        }
    }

    fn drain_obligations(self, out: &mut Vec<(DefId, String)>) {
        out.extend(self.obligations);
        for child in self.candidates.into_iter().flat_map(|c| c.children) {
            child.drain_obligations(out);
        }
    }
}

/// Sites carry provisional ids (indices into `Builder::sites`) until emission.
struct CandidateNode {
    template: TemplateId,
    definition: DefId,
    via: DefId,
    reference: bool,
    numerical_effects: Vec<NumericalEffect>,
    structural: Vec<(String, SiteId)>,
    requirements: Vec<Requirement>,
    children: Vec<OccurrenceNode>,
    sites: Vec<(SiteId, SiteKind)>,
}

/// The candidate body containing a call: what callee shape arguments are evaluated under.
struct Caller<'s> {
    binding: &'s Binding,
    body: &'s Body,
    /// Provisional site of each site-owning slice of the body.
    own: &'s BTreeMap<SliceId, SiteId>,
}

enum Source<'s> {
    Workload(&'s Workload),
    Call {
        site: &'s CallSite,
        caller: &'s Caller<'s>,
    },
}

type TemplateKey = (
    DefId,
    Vec<(String, i64)>,
    Vec<(String, Elem)>,
    Vec<String>,
    Vec<String>,
);

struct Builder<'a> {
    program: &'a Program,
    target: &'a str,
    templates: Vec<Template>,
    interned: HashMap<TemplateKey, TemplateId>,
    /// Provisional sites, including those of candidates removed later.
    sites: Vec<SiteExtent>,
    /// Templates on the current expansion path.
    path: Vec<TemplateId>,
    units: HashMap<DefId, Rc<Vec<BlockUnits>>>,
}

impl<'a> Builder<'a> {
    fn describe(&self, id: DefId) -> String {
        let def = self.program.definition(id);
        match self.program.files.get(def.file) {
            Some((path, text)) => format!(
                "`{}` at {path}:{}",
                def.name,
                line_col(text, def.span.start).0
            ),
            None => format!("`{}`", def.name),
        }
    }

    /// Portable semantic reference of one linked function family.
    fn reference(&self, family: usize) -> Option<DefId> {
        let contract = &self.program.families[family];
        contract
            .bodies
            .iter()
            .copied()
            .find(|id| matches!(self.program.definition(*id).kind, DefKind::Body { target: None }))
    }

    fn coverage(&self, node: &OccurrenceNode) -> String {
        if node.rejected.is_empty() {
            return format!(
                "`{}` declares no implementation for `{}`",
                self.program.families[node.family].name, self.target
            );
        }
        node.rejected
            .iter()
            .map(|(def, reason)| format!("{}: {reason}", self.describe(*def)))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// The occurrence of contract family `family` with every selectable function and
    /// lowering for the target, in declaration order.
    fn occurrence(
        &mut self,
        family: usize,
        call: Option<CallId>,
        source: &Source,
    ) -> Result<OccurrenceNode, String> {
        let program = self.program;
        let contract = program
            .families
            .get(family)
            .ok_or_else(|| format!("call names missing contract family #{family}"))?;
        let mut node = OccurrenceNode {
            id: OccurrenceId(0),
            call,
            family,
            candidates: Vec::new(),
            rejected: Vec::new(),
            obligations: Vec::new(),
        };
        for id in &contract.bodies {
            let def = program.definition(*id);
            if matches!(&def.kind, DefKind::Body { target: Some(required) } if required != self.target)
            {
                continue;
            }
            self.attempt(&mut node, def, def.id, source)?;
        }
        for id in &contract.lowerings {
            let def = program.definition(*id);
            if matches!(&def.kind, DefKind::Lower { target } if target == self.target) {
                self.attempt(&mut node, def, def.id, source)?;
            }
        }
        Ok(node)
    }

    fn attempt(
        &mut self,
        node: &mut OccurrenceNode,
        def: &'a Definition,
        via: DefId,
        source: &Source,
    ) -> Result<(), String> {
        match self.candidate(def, via, source)? {
            Ok(candidate) => node.candidates.push(candidate),
            Err(reject) => node.reject(def.id, reject),
        }
        Ok(())
    }

    fn candidate(
        &mut self,
        def: &'a Definition,
        via: DefId,
        source: &Source,
    ) -> Attempt<CandidateNode> {
        let body = &def.body;
        let reference = self.reference(def.family) == Some(via);
        let mut numerical_effects = Vec::new();
        if !reference {
            numerical_effects.push(NumericalEffect::AlternativeImplementation);
        }
        walk::block(&body.block, true, &mut |expr| {
            let effect = match &expr.kind {
                ExprKind::Reduce { unordered: true, .. } => Some(NumericalEffect::ReassociatedReduction),
                ExprKind::Math { op: crate::sir::Math::ExpFast, .. } => Some(NumericalEffect::ApproximateTranscendental("exp".into())),
                ExprKind::Intrinsic { op, .. } => Some(NumericalEffect::BackendIntrinsic {
                    target: def.kind.target().unwrap_or("unknown").to_string(),
                    operation: format!("{op:?}"),
                }),
                _ => None,
            };
            if let Some(effect) = effect.filter(|effect| !numerical_effects.contains(effect)) {
                numerical_effects.push(effect);
            }
        });
        let binding = match self.bind(def, source) {
            Ok(binding) => binding,
            Err(reject) => return Ok(Err(reject)),
        };
        let requirements = match self.requirements(&def.predicates, &binding) {
            Ok(found) => found,
            Err(reject) => return Ok(Err(reject)),
        };
        let template = self.intern(def, &binding);
        if let Some(at) = self.path.iter().position(|t| *t == template) {
            let cycle: Vec<String> = self.path[at..]
                .iter()
                .chain([&template])
                .map(|t| self.describe(self.templates[t.0 as usize].definition))
                .collect();
            return Err(format!(
                "recursive candidate cycle without a decreasing static measure: {}",
                cycle.join(" -> ")
            ));
        }
        self.path.push(template);
        let expanded = self.expand(def, body, via, reference, numerical_effects, template, binding, requirements);
        self.path.pop();
        expanded
    }

    fn requirements(
        &self,
        predicates: &[Predicate],
        binding: &Binding,
    ) -> Result<Vec<Requirement>, Reject> {
        applicability::requirements(predicates, binding, &|site| self.sites[site.0 as usize])
            .map_err(Reject::inapplicable)
    }

    fn bind(&self, def: &Definition, source: &Source) -> Result<Binding, Reject> {
        let mut binding = Binding {
            shapes: BTreeMap::new(),
            structural: Vec::new(),
            dynamic: Vec::new(),
            elems: BTreeMap::new(),
        };
        match source {
            Source::Workload(workload) => {
                for name in &def.shape_params {
                    let value = workload.shapes.get(name).ok_or_else(|| {
                        Reject::inapplicable(format!(
                            "workload does not bind shape parameter `{name}`"
                        ))
                    })?;
                    binding.shapes.insert(name.clone(), *value);
                }
                for name in &def.elem_params {
                    if let Some(elem) = workload.elems.get(name) {
                        binding.elems.insert(name.clone(), elem.clone());
                    }
                }
            }
            Source::Call { site, caller } => {
                let args = site
                    .bindings
                    .iter()
                    .find(|b| b.definition == def.id)
                    .ok_or_else(|| {
                        Reject::inapplicable("parameters do not unify with this call".into())
                    })?;
                for name in &def.shape_params {
                    let (_, extent) = args
                        .shape_args
                        .iter()
                        .find(|(arg, _)| arg == name)
                        .ok_or_else(|| {
                            Reject::inapplicable(format!(
                                "call does not determine shape parameter `{name}`"
                            ))
                        })?;
                    match extent {
                        Extent::Structural(slice) => {
                            let site = owning_slice(caller.body, *slice)
                                .and_then(|owner| caller.own.get(&owner))
                                .ok_or_else(|| Reject::unanalyzed(format!("`{name}` is bound to slice#{} which has no numerical site in the caller", slice.0)))?;
                            binding.structural.push((name.clone(), *site));
                        }
                        Extent::Semantic(sym) => {
                            let inherited = caller
                                .binding
                                .structural
                                .iter()
                                .find(|(param, _)| *sym == Sym::param(param));
                            let params = sym.params();
                            if let Some((_, site)) = inherited {
                                binding.structural.push((name.clone(), *site));
                            } else if caller
                                .binding
                                .structural
                                .iter()
                                .any(|(param, _)| params.contains(param))
                            {
                                return Err(Reject::unanalyzed(format!(
                                    "`{name} = {sym}` computes with a structural extent"
                                )));
                            } else if let Some(value) =
                                sym.eval(&|param| caller.binding.shapes.get(param).copied())
                            {
                                binding.shapes.insert(name.clone(), value);
                            } else {
                                // A runtime-valued semantic extent of the caller: a runtime-bounded
                                // range, a runtime index, or a parameter that is itself dynamic.
                                binding.dynamic.push(name.clone());
                            }
                        }
                    }
                }
                for (param, required) in &args.requires_elems {
                    match caller.binding.elems.get(param) {
                        Some(bound) if bound == required => {}
                        Some(bound) => return Err(Reject::inapplicable(format!("element `{param}` is `{bound}`, candidate requires `{required}`"))),
                        None => return Err(Reject::inapplicable(format!("element `{param}` is unbound in the caller, candidate requires `{required}`"))),
                    }
                }
                for (name, elem) in &args.elem_args {
                    let elem = resolve_elem(elem, &caller.binding.elems).ok_or_else(|| {
                        Reject::unanalyzed(format!(
                            "element argument `{name} = {elem}` is unbound in the caller"
                        ))
                    })?;
                    binding.elems.insert(name.clone(), elem);
                }
            }
        }
        for (name, fixed) in &def.elem_bindings {
            match binding.elems.get(name) {
                Some(bound) if bound != fixed => {
                    return Err(Reject::inapplicable(format!(
                        "definition fixes `{name} = {fixed}`, bound `{bound}`"
                    )));
                }
                _ => {
                    binding.elems.insert(name.clone(), fixed.clone());
                }
            }
        }
        if let Some(name) = def
            .elem_params
            .iter()
            .find(|name| !binding.elems.contains_key(*name))
        {
            return Err(Reject::inapplicable(format!(
                "element parameter `{name}` is unbound"
            )));
        }
        Ok(binding)
    }

    fn intern(&mut self, def: &Definition, binding: &Binding) -> TemplateId {
        let structural: Vec<String> = binding
            .structural
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let key: TemplateKey = (
            def.id,
            binding
                .shapes
                .iter()
                .map(|(name, value)| (name.clone(), *value))
                .collect(),
            binding
                .elems
                .iter()
                .map(|(name, elem)| (name.clone(), elem.clone()))
                .collect(),
            structural.clone(),
            binding.dynamic.clone(),
        );
        let next = TemplateId(self.templates.len() as u32);
        let id = *self.interned.entry(key).or_insert(next);
        if id == next {
            self.templates.push(Template {
                id,
                definition: def.id,
                shapes: binding.shapes.clone(),
                elems: binding.elems.clone(),
                structural,
                dynamic: binding.dynamic.clone(),
            });
        }
        id
    }

    /// Sites, packed alignment and child occurrences of one candidate body.
    fn expand(
        &mut self,
        def: &'a Definition,
        body: &'a Body,
        via: DefId,
        reference: bool,
        numerical_effects: Vec<NumericalEffect>,
        template: TemplateId,
        binding: Binding,
        mut requirements: Vec<Requirement>,
    ) -> Attempt<CandidateNode> {
        let merges: BTreeSet<RegionId> = walk::regions(&body.block)
            .iter()
            .filter(|r| r.merge.is_some())
            .map(|r| r.id)
            .collect();
        let bounds = Bounds::new(body, &binding.shapes);
        let mut own = BTreeMap::new();
        let mut sites = Vec::new();
        for (index, decl) in body.slices.iter().enumerate() {
            let slice = SliceId(index as u32);
            if matches!(decl.parent, SliceParent::Rebind(_)) {
                continue;
            }
            let Some(extent) = slice_extent(body, slice, &bounds, body.slices.len()) else {
                let name = body.vars.get(decl.var).map_or("?", |v| v.name.as_str());
                return Ok(Err(Reject::unanalyzed(format!(
                    "domain of binder `{name}` has no static extent or upper bound"
                ))));
            };
            let site = SiteId(self.sites.len() as u32);
            self.sites.push(extent);
            own.insert(slice, site);
            let kind = if merges.contains(&decl.region) {
                SiteKind::Parts {
                    region: decl.region,
                    slice,
                }
            } else {
                SiteKind::Width {
                    region: decl.region,
                    slice,
                }
            };
            sites.push((site, kind));
        }

        let mut unanalyzed = None;
        walk::block(&body.block, true, &mut |e| {
            let ExprKind::Index { base, indices } = &e.kind else {
                return;
            };
            let Some(shape) = base.ty.shaped() else {
                return;
            };
            let Some(Elem::Repr(name)) = resolve_elem(&shape.elem, &binding.elems) else {
                return;
            };
            let axis = shape
                .packed_axis
                .unwrap_or(shape.axes.len().saturating_sub(1));
            let Some(Index::Slice(slice)) = indices.get(axis) else {
                return;
            };
            let site = owning_slice(body, *slice).and_then(|owner| own.get(&owner));
            match (repr::lookup(&name), site) {
                (Some(repr), Some(site)) => {
                    let aligned = Requirement::Multiple {
                        site: *site,
                        unit: i64::from(repr.storage_group()),
                    };
                    if !requirements.contains(&aligned) {
                        requirements.push(aligned);
                    }
                }
                (None, _) => unanalyzed = Some(format!("unknown packed representation `{name}`")),
                (_, None) => {
                    unanalyzed = Some(format!(
                        "slice#{} on a packed axis has no numerical site in this body",
                        slice.0
                    ))
                }
            }
        });
        if let Some(reason) = unanalyzed {
            return Ok(Err(Reject::unanalyzed(reason)));
        }

        let caller = Caller {
            binding: &binding,
            body,
            own: &own,
        };
        let mut children: Vec<OccurrenceNode> = Vec::new();
        for (index, site) in body.calls.iter().enumerate() {
            let child = self.occurrence(
                site.family,
                Some(CallId(index as u32)),
                &Source::Call {
                    site,
                    caller: &caller,
                },
            )?;
            if child.candidates.is_empty() {
                let callee = self.program.families[child.family].name.as_str();
                let reason = format!(
                    "call to `{callee}` has no selectable implementation on `{}` ({})",
                    self.target,
                    self.coverage(&child)
                );
                let mut obligations = Vec::new();
                children
                    .into_iter()
                    .chain([child])
                    .for_each(|removed| removed.drain_obligations(&mut obligations));
                return Ok(Err(Reject {
                    reason,
                    unanalyzed: false,
                    obligations,
                }));
            }
            children.push(child);
        }
        let structural = binding.structural;
        Ok(Ok(CandidateNode {
            template,
            definition: def.id,
            via,
            reference,
            numerical_effects,
            structural,
            requirements,
            children,
            sites,
        }))
    }

    fn units(&mut self, def: DefId, body: &Body) -> Rc<Vec<BlockUnits>> {
        self.units
            .entry(def)
            .or_insert_with(|| Rc::new(sequences(body)))
            .clone()
    }

    /// Emit a numbered tree into `family`. `sites` maps provisional to final site ids;
    /// ancestors are emitted first, so every site a candidate can name is already mapped.
    fn emit(
        &mut self,
        node: OccurrenceNode,
        parent: Option<CandidateRef>,
        family: &mut Family,
        sites: &mut BTreeMap<SiteId, SiteId>,
    ) -> Result<(), String> {
        let id = node.id;
        if family.occurrences.len() != id.0 as usize {
            return Err(format!(
                "internal: occurrence {} emitted out of pre-order",
                id.0
            ));
        }
        family
            .obligations
            .extend(
                node.obligations
                    .into_iter()
                    .map(|(definition, reason)| Obligation {
                        occurrence: id,
                        definition,
                        reason,
                    }),
            );
        let mut candidates = Vec::new();
        let mut subtrees = Vec::new();
        for (ordinal, node) in node.candidates.into_iter().enumerate() {
            let owner = CandidateRef {
                occurrence: id,
                candidate: ordinal as u32,
            };
            let mut own = Vec::new();
            for (provisional, kind) in node.sites {
                let site = SiteId(family.sites.len() as u32);
                family.sites.push(Site {
                    id: site,
                    owner,
                    kind,
                    extent: self.sites[provisional.0 as usize].extent,
                });
                sites.insert(provisional, site);
                own.push(site);
            }
            let resolve = |provisional: SiteId| {
                sites.get(&provisional).copied().ok_or_else(|| {
                    format!(
                        "internal: site of a removed candidate is referenced from occurrence {}",
                        id.0
                    )
                })
            };
            let structural = node
                .structural
                .into_iter()
                .map(|(name, site)| Ok((name, SiteRef(resolve(site)?))))
                .collect::<Result<Vec<_>, String>>()?;
            let requirements = node
                .requirements
                .iter()
                .map(|r| Ok(with_site(r, resolve(r.site())?)))
                .collect::<Result<Vec<_>, String>>()?;

            let definition = self.program.definition(node.definition);
            let body = &definition.body;
            let mut sequences = Vec::new();
            for block in self.units(node.definition, body).iter() {
                let mut units = block.units.clone();
                for unit in &mut units {
                    if let UnitKind::Call(call) = &mut unit.kind {
                        let child = node
                            .children
                            .iter()
                            .find(|child| child.call == Some(CallId(call.0)));
                        *call = child.map(|child| child.id).ok_or_else(|| {
                            format!(
                                "internal: call #{} of {} has no occurrence",
                                call.0,
                                self.describe(node.definition)
                            )
                        })?;
                    }
                }
                let sequence = SequenceId(family.sequences.len() as u32);
                family.sequences.push(Sequence {
                    id: sequence,
                    owner,
                    scope: block.scope.clone(),
                    units,
                });
                sequences.push(sequence);
            }
            candidates.push(Candidate {
                template: node.template,
                via: node.via,
                reference: node.reference,
                numerical_effects: node.numerical_effects,
                structural,
                requirements,
                children: node.children.iter().map(|child| child.id).collect(),
                sites: own,
                sequences,
            });
            subtrees.push((owner, node.children));
        }
        family.occurrences.push(Occurrence {
            id,
            parent,
            call: node.call,
            family: node.family,
            candidates,
            rejected: node.rejected,
        });
        for (owner, children) in subtrees {
            for child in children {
                self.emit(child, Some(owner), family, sites)?;
            }
        }
        Ok(())
    }
}

/// Every `(refinement, refined)` width-site pair of the family: a binder whose domain is an
/// enclosing slice of its own body, with the site that owns that slice.
fn refinements(program: &Program, family: &Family) -> Vec<(SiteId, SiteId)> {
    let mut out = Vec::new();
    for site in &family.sites {
        let SiteKind::Width { slice, .. } = site.kind else {
            continue;
        };
        let candidate = family.candidate(site.owner);
        let body = &program
            .definition(family.template(candidate.template).definition)
            .body;
        let Some(SliceParent::Refine(parent)) = body
            .slices
            .get(slice.0 as usize)
            .map(|declared| &declared.parent)
        else {
            continue;
        };
        let refined = owning_slice(body, *parent).and_then(|owner| {
            candidate.sites.iter().copied().find(|id| matches!(family.sites[id.0 as usize].kind, SiteKind::Width { slice, .. } if slice == owner))
        });
        if let Some(refined) = refined {
            out.push((site.id, refined));
        }
    }
    out
}

fn number(node: &mut OccurrenceNode, next: &mut u32) {
    node.id = OccurrenceId(*next);
    *next += 1;
    for child in node
        .candidates
        .iter_mut()
        .flat_map(|c| c.children.iter_mut())
    {
        number(child, next);
    }
}

fn resolve_elem(elem: &Elem, elems: &BTreeMap<String, Elem>) -> Option<Elem> {
    match elem {
        Elem::Param(name) => elems.get(name).cloned(),
        concrete => Some(concrete.clone()),
    }
}

/// Extent of a site-owning slice: its domain's, or for a refinement its parent site's.
fn slice_extent(body: &Body, slice: SliceId, bounds: &Bounds, depth: usize) -> Option<SiteExtent> {
    match &body.slices.get(slice.0 as usize)?.parent {
        SliceParent::Domain { lo, hi } => bounds.extent(lo, hi),
        SliceParent::Refine(parent) => slice_extent(
            body,
            owning_slice(body, *parent)?,
            bounds,
            depth.checked_sub(1)?,
        ),
        SliceParent::Rebind(_) => None,
    }
}

fn with_site(requirement: &Requirement, site: SiteId) -> Requirement {
    match *requirement {
        Requirement::Multiple { unit, .. } => Requirement::Multiple { site, unit },
        Requirement::AtLeast { value, .. } => Requirement::AtLeast { site, value },
        Requirement::AtMost { value, .. } => Requirement::AtMost { site, value },
        Requirement::Equal { value, .. } => Requirement::Equal { site, value },
        Requirement::Divides { extent, .. } => Requirement::Divides { site, extent },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{ScopeStep, Witness};
    use crate::program::{compile, SourceFile};

    const DOTS: &str = "\
fn row_dot[N](x: tensor[N] f32, w: tensor[N] f32) -> f32
    where N >= 1:
    let mut result = f32(0.0)
    for i in 0..N:
        result = fma(x[i], w[i], result)
    return result

fn row_dot[N](x: tensor[N] f32, w: tensor[N] f32) -> f32
    where N >= 2 and N % 2 == 0:
    let mut result = f32(0.0)
    for pair in 0..(N / 2):
        result = fma(x[2 * pair], w[2 * pair], result)
        result = fma(x[2 * pair + 1], w[2 * pair + 1], result)
    return result

fn two_dots[N](x: tensor[N] f32, w0: tensor[N] f32, w1: tensor[N] f32) -> (f32, f32)
    where N >= 1:
    let first = row_dot(x, w0)
    let second = row_dot(x, w1)
    return first, second

";

    fn family(text: &str, entry: &str, shapes: &[(&str, i64)]) -> Result<Family, String> {
        let file = SourceFile {
            path: "test.seismic".into(),
            text: text.into(),
        };
        let program = compile(&[file]).map_err(|diagnostics| {
            diagnostics
                .iter()
                .map(|d| d.render())
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        let workload = Workload {
            shapes: shapes
                .iter()
                .map(|(name, value)| (name.to_string(), *value))
                .collect(),
            ..Workload::default()
        };
        construct(&program, entry, "cpu", &workload)
    }

    #[test]
    fn overlapping_bodies_are_candidates_wherever_they_apply() -> Result<(), String> {
        for (n, expected) in [(8, 2), (5, 1)] {
            let family = family(DOTS, "two_dots", &[("N", n)])?;
            assert_eq!(family.occurrences.len(), 3);
            assert_eq!(family.occurrences[0].candidates.len(), 1);
            for child in &family.occurrences[1..] {
                assert_eq!(
                    child.parent,
                    Some(CandidateRef {
                        occurrence: OccurrenceId(0),
                        candidate: 0
                    })
                );
                assert_eq!(child.candidates.len(), expected);
                assert_eq!(child.rejected.len(), 2 - expected);
            }
            assert_eq!(
                family.occurrences[1].candidates[0].template,
                family.occurrences[2].candidates[0].template
            );
            let mut witness = Witness::default();
            witness.choices = family
                .occurrences
                .iter()
                .map(|o| (o.id, o.candidates.len() as u32 - 1))
                .collect();
            // Covers belong to active sequences only: those of the selected candidates.
            let covers = family
                .sequences
                .iter()
                .filter(|s| family.active(&witness, s.owner))
                .map(|s| (s.id, vec![(0, s.units.len() as u32)]))
                .collect();
            witness.covers = covers;
            // Entry root; per child the scalar body's root, and the paired body's root and loop body.
            assert_eq!(family.sequences.len(), if expected == 2 { 7 } else { 3 });
            family.validate(&witness)?;
            witness.choices.remove(&OccurrenceId(2));
            assert!(family.validate(&witness).is_err());
        }
        Ok(())
    }

    #[test]
    fn portable_body_and_matching_lowering_are_union_candidates() -> Result<(), String> {
        let text = "\
fn pick[N](x: tensor[N] f32) -> f32 where N >= 1:
    return x[0]

lower pick[N](x: tensor[N] f32) -> f32 for cpu where N >= 1:
    return x[N - 1]
";
        let file = SourceFile {
            path: "pick.seismic".into(),
            text: text.into(),
        };
        let program = compile(&[file]).map_err(|diagnostics| {
            diagnostics
                .iter()
                .map(|d| d.render())
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        let workload = Workload {
            shapes: [("N".into(), 8)].into(),
            ..Workload::default()
        };

        let cpu = construct(&program, "pick", "cpu", &workload)?;
        assert_eq!(cpu.occurrences[0].candidates.len(), 2);
        assert!(cpu.occurrences[0]
            .candidates
            .iter()
            .any(|candidate| matches!(
                program.definition(candidate.via).kind,
                DefKind::Body { target: None }
            )));
        assert!(cpu.occurrences[0]
            .candidates
            .iter()
            .any(|candidate| matches!(program.definition(candidate.via).kind, DefKind::Lower { ref target } if target == "cpu")));

        let metal = construct(&program, "pick", "metal", &workload)?;
        assert_eq!(metal.occurrences[0].candidates.len(), 1);
        assert!(matches!(
            program
                .definition(metal.occurrences[0].candidates[0].via)
                .kind,
            DefKind::Body { target: None }
        ));
        Ok(())
    }

    #[test]
    fn name_only_root_rejects_disjoint_overload_families() -> Result<(), String> {
        let text = "\
fn choose(x: f32) -> f32:
    return x

fn choose[N](x: tensor[N] f32) -> f32 where N >= 1:
    return x[0]
";
        let file = SourceFile {
            path: "choose.seismic".into(),
            text: text.into(),
        };
        let program = compile(&[file]).map_err(|diagnostics| {
            diagnostics
                .iter()
                .map(|d| d.render())
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        let error = construct(&program, "choose", "cpu", &Workload::default()).unwrap_err();
        assert!(
            error.contains("ambiguous linked function `choose`"),
            "{error}"
        );
        assert!(program.family("choose").is_none());
        Ok(())
    }

    #[test]
    fn missing_coverage_names_the_uncovered_path() {
        let error = family(DOTS, "two_dots", &[("N", 0)])
            .map(|_| ())
            .unwrap_err();
        assert!(
            error.contains("missing coverage") && error.contains("two_dots"),
            "{error}"
        );
    }

    #[test]
    fn packed_axis_slices_align_to_the_representation_group() -> Result<(), String> {
        let text = "\
fn unpack[M, N](w: tensor[M, N] q4g64, out y: tensor[M, N] f32):
    parallel [cols] in 0..N:
        publish decode(w[:, cols]) to y[:, cols]
";
        let family = family(text, "unpack", &[("M", 4), ("N", 256)])?;
        let candidate = &family.occurrences[0].candidates[0];
        assert_eq!(candidate.sites.len(), 1);
        assert_eq!(family.sites[0].extent, 256);
        assert_eq!(
            candidate.requirements,
            vec![Requirement::Multiple {
                site: candidate.sites[0],
                unit: 64
            }]
        );
        Ok(())
    }

    #[test]
    fn fan_out_keeps_the_shared_producer_and_folds_single_consumers() -> Result<(), String> {
        let text = "\
fn fan[N](x: tensor[N] f32, out y: tensor[N] f32):
    parallel [cols] in 0..N:
        let a = exp(f32(x[cols]))
        let b = a * f32(2.0)
        let c = a + f32(1.0)
        publish b + c to y[cols]
";
        let family = family(text, "fan", &[("N", 64)])?;
        assert_eq!(family.sequences.len(), 1);
        let sequence = &family.sequences[0];
        assert_eq!(sequence.scope, vec![ScopeStep::Region(RegionId(0))]);
        let units: Vec<_> = sequence
            .units
            .iter()
            .map(|u| (u.statements.clone(), u.kind.clone(), u.completion_after))
            .collect();
        assert_eq!(
            units,
            vec![
                (0..1, UnitKind::Elementwise, false),
                (1..4, UnitKind::Publish, false)
            ]
        );
        Ok(())
    }
}
