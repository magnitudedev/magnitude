//! Independent load realization choices. Legality is a lifetime/effect fact;
//! choosing a mode never supplies a performance preference.
use crate::{ast::AssignOp, ir::*, lowered_ir::LoweredIr};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub variable: VarId,
    pub can_borrow: bool,
    pub selected: Option<LoadMode>,
}

/// All load ownership domains over one immutable normalized execution. This
/// owner performs no implementation search and discovers no new sites while
/// reconstructing an assignment. A previously selected operation is a singleton.
#[derive(Clone, Debug)]
pub struct Family {
    function: LoweredIr,
    sites: Vec<Site>,
    domains: Vec<Vec<LoadMode>>,
}
impl Family {
    pub fn new(mut function: LoweredIr) -> Result<Self, String> {
        super::bind_values(&mut function.body, &mut function.vars);
        let sites = sites(&function.body);
        let domains = sites.iter().map(|site| {
            if let Some(mode) = site.selected {
                if mode == LoadMode::Borrow && !site.can_borrow {
                    return Err(format!("selected load of variable {} violates its snapshot lifetime", site.variable));
                }
                Ok(vec![mode])
            } else if site.can_borrow {
                Ok(vec![LoadMode::Materialize, LoadMode::Borrow])
            } else {
                Ok(vec![LoadMode::Materialize])
            }
        }).collect::<Result<_, String>>()?;
        Ok(Self { function, sites, domains })
    }
    /// Carry original ownership domains through a computation-preserving
    /// phase transform. Improved local lifetime facts may admit more modes,
    /// but do not introduce new source freedom or change original ordinals.
    pub fn with_domains(function: LoweredIr, domains: Vec<Vec<LoadMode>>) -> Result<Self, String> {
        let mut family = Self::new(function)?;
        if domains.len() != family.sites.len() {
            return Err("retained load domains do not cover exactly the transformed sites".into());
        }
        for (site, (supplied, legal)) in domains.iter().zip(&family.domains).enumerate() {
            if supplied.is_empty() { return Err(format!("retained load site {site} has an empty original domain")); }
            for (ordinal, mode) in supplied.iter().enumerate() {
                if supplied[..ordinal].contains(mode) { return Err(format!("retained load site {site} repeats an original mode")); }
                if !legal.contains(mode) { return Err(format!("retained load site {site} no longer admits its original mode {mode:?}")); }
            }
        }
        family.domains = domains;
        Ok(family)
    }
    /// Restrict one site to a single legal mode. A guard-conditioned lifetime
    /// proof may be unavailable; the conservative mode must then be sound on
    /// its own. This never admits a borrow the original domain did not allow.
    pub fn restrict(&mut self, site: usize, mode: LoadMode) -> Result<(), String> {
        let legal = self.domains.get(site).ok_or("load site is outside the family")?;
        if !legal.contains(&mode) { return Err("restricted load mode is not legal at this site".into()); }
        self.domains[site] = vec![mode];
        Ok(())
    }
    pub fn function(&self) -> &LoweredIr { &self.function }
    pub fn sites(&self) -> &[Site] { &self.sites }
    pub fn domains(&self) -> &[Vec<LoadMode>] { &self.domains }
    pub fn instantiate(&self, ordinals: &[usize]) -> Result<LoweredIr, String> {
        if ordinals.len() != self.domains.len() { return Err("load assignment does not cover exactly the retained sites".into()); }
        let modes = self.domains.iter().zip(ordinals).enumerate().map(|(site, (domain, &ordinal))| {
            domain.get(ordinal).copied().ok_or_else(|| format!("load site {site} choice is outside its original domain"))
        }).collect::<Result<Vec<_>, _>>()?;
        let mut selected = self.function.clone();
        resolve(&mut selected.body, &modes)?;
        Ok(selected)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub site: usize,
    pub variable: VarId,
    pub mode: LoadMode,
}

pub fn sites(body: &[Stmt]) -> Vec<Site> {
    fn visit(body: &[Stmt], lifetimes: &crate::effects::LoadLifetimes<'_>, out: &mut Vec<Site>) {
        for stmt in body {
            match &stmt.kind {
                StmtKind::Reduction(reduction) => {
                    for implementation in reduction.implementations() { visit(&implementation.body, lifetimes, out); }
                }
                StmtKind::Assign {
                    target,
                    op: AssignOp::Assign,
                    value,
                } => {
                    if let ExprKind::Var(variable) = target.kind {
                        let selected = match value.kind {
                            ExprKind::Builtin {
                                name: Builtin::Load,
                                ..
                            } => Some(None),
                            ExprKind::Load { mode, .. } => Some(Some(mode)),
                            _ => None,
                        };
                        if let Some(selected) = selected {
                            out.push(Site {
                                variable,
                                can_borrow: lifetimes.can_borrow(stmt),
                                selected,
                            });
                        }
                    }
                }
                StmtKind::LoadLoop {
                    vars, views, modes, body, ..
                } => {
                    for (i, &variable) in vars.iter().enumerate() {
                        out.push(Site {
                            variable,
                            can_borrow: views.get(i).is_some_and(|view|crate::effects::stream_load_can_borrow(body, variable, view)),
                            selected: modes.as_ref().and_then(|m| m.get(i)).copied(),
                        });
                    }
                    visit(body, lifetimes, out);
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, lifetimes, out),
                StmtKind::If { then, els, .. } => {
                    visit(then, lifetimes, out);
                    visit(els, lifetimes, out);
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    let lifetimes = crate::effects::LoadLifetimes::new(body);
    visit(body, &lifetimes, &mut out);
    out
}

/// Check every decision before changing the tree. Site order is lexical and
/// includes each operand of a streamed load separately.
pub fn resolve(body: &mut [Stmt], modes: &[LoadMode]) -> Result<Vec<Decision>, String> {
    fn validate(body:&[Stmt])->Result<(),String> {
        for statement in body {
            match &statement.kind {
                StmtKind::LoadLoop{domain,vars,views,axes,modes,body,..}=>{
                    let extent=domain.view.ty.shaped().and_then(|s|s.shape.get(domain.axis)).ok_or("invalid logical iteration domain")?;
                    if vars.len()!=views.len() || axes.len()!=views.len() || modes.as_ref().is_some_and(|m|m.len()!=views.len()) {return Err("iteration transfer binding counts disagree".into());}
                    for (view,axis) in views.iter().zip(axes) {
                        let active=view.ty.shaped().and_then(|s|s.shape.get(*axis)).ok_or("invalid iteration transfer axis")?;
                        if extent.as_constant().zip(active.as_constant()).is_some_and(|(a,b)|a!=b) {return Err("iteration transfer extent differs from logical domain".into());}
                    }
                    validate(body)?;
                }
                StmtKind::Reduction(r)=>for body in r.bodies(){validate(body)?;},
                StmtKind::Parallel{body,..}|StmtKind::Owned{body,..}|StmtKind::Range{body,..}|StmtKind::Lanes{body,..}=>validate(body)?,
                StmtKind::If{then,els,..}=>{validate(then)?;validate(els)?;},
                _=>{}
            }
        }
        Ok(())
    }
    validate(body)?;
    let sites = sites(body);
    if sites.len() != modes.len() {
        return Err("load decision count does not match the bound IR".into());
    }
    let decisions = sites
        .iter()
        .zip(modes)
        .enumerate()
        .map(|(site, (s, &mode))| {
            if mode == LoadMode::Borrow && !s.can_borrow {
                return Err(format!("load site {site} requires snapshot storage"));
            }
            Ok(Decision {
                site,
                variable: s.variable,
                mode,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    fn apply(body: &mut [Stmt], modes: &mut std::slice::Iter<'_, LoadMode>) {
        for stmt in body {
            match &mut stmt.kind {
                StmtKind::Reduction(reduction) => {
                    for implementation in reduction.implementations_mut() { apply(&mut implementation.body, modes); }
                }
                StmtKind::Assign {
                    target,
                    op: AssignOp::Assign,
                    value,
                } if matches!(target.kind, ExprKind::Var(_)) => {
                    let view = match &value.kind {
                        ExprKind::Builtin {
                            name: Builtin::Load,
                            args,
                        } => Some(args[0].clone()),
                        ExprKind::Load { view, .. } => Some((**view).clone()),
                        _ => None,
                    };
                    if let Some(view) = view {
                        value.kind = ExprKind::Load {
                            view: Box::new(view),
                            mode: *modes.next().unwrap(),
                        };
                    }
                }
                StmtKind::LoadLoop {
                    vars,
                    modes: selected,
                    body,
                    ..
                } => {
                    *selected = Some((0..vars.len()).map(|_| *modes.next().unwrap()).collect());
                    apply(body, modes);
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => apply(body, modes),
                StmtKind::If { then, els, .. } => {
                    apply(then, modes);
                    apply(els, modes);
                }
                _ => {}
            }
        }
    }
    apply(body, &mut modes.iter());
    Ok(decisions)
}

/// Validate already selected operations without choosing or changing any mode.
pub fn selected(body: &[Stmt]) -> Result<Vec<Decision>, String> {
    sites(body)
        .into_iter()
        .enumerate()
        .map(|(site, s)| {
            let mode = s
                .selected
                .ok_or_else(|| format!("unresolved load site {site}"))?;
            if mode == LoadMode::Borrow && !s.can_borrow {
                return Err(format!(
                    "load site {site} has an invalid borrowing lifetime"
                ));
            }
            Ok(Decision {
                site,
                variable: s.variable,
                mode,
            })
        })
        .collect()
}

/// The unresolved choice at an actual normalized load site. Its alternatives
/// are owned here and used by both explicit selection and optimization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub site: usize,
    pub variable: VarId,
}
impl Choice {
    pub fn modes(&self) -> &'static [LoadMode] {
        &[LoadMode::Materialize, LoadMode::Borrow]
    }
}

pub enum Expansion {
    Choice(Choice),
    Selected {
        function: LoweredIr,
        consumed: usize,
    },
}

/// Every eligible site has both alternatives, including mixed policies. Forced
/// snapshots do not add duplicate branches. The returned tree owns all choices.
pub fn expand(function: &LoweredIr, prefix: &[usize]) -> Result<Expansion, String> {
    let mut function = function.clone();
    super::bind_values(&mut function.body, &mut function.vars);
    let mut consumed = 0;
    let mut modes = Vec::new();
    for (site, s) in sites(&function.body).into_iter().enumerate() {
        modes.push(if s.can_borrow {
            let Some(&choice) = prefix.get(consumed) else {
                return Ok(Expansion::Choice(Choice {
                    site,
                    variable: s.variable,
                }));
            };
            consumed += 1;
            Choice {
                site,
                variable: s.variable,
            }
            .modes()
            .get(choice)
            .copied()
            .ok_or("load choice is outside its domain")?
        } else {
            LoadMode::Materialize
        });
    }
    resolve(&mut function.body, &modes)?;
    Ok(Expansion::Selected { function, consumed })
}
