//! Independent load realization choices. Legality is a lifetime/effect fact;
//! choosing a mode never supplies a performance preference.
use crate::{ast::AssignOp, ir::*, lowered_ir::LoweredIr};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub variable: VarId,
    pub can_borrow: bool,
    pub selected: Option<LoadMode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub site: usize,
    pub variable: VarId,
    pub mode: LoadMode,
}

pub fn sites(body: &[Stmt]) -> Vec<Site> {
    fn visit(body: &[Stmt], root: &[Stmt], out: &mut Vec<Site>) {
        for stmt in body {
            match &stmt.kind {
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
                                can_borrow: crate::effects::load_can_borrow(root, variable),
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
                    visit(body, root, out);
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, root, out),
                StmtKind::If { then, els, .. } => {
                    visit(then, root, out);
                    visit(els, root, out);
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    visit(body, body, &mut out);
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
