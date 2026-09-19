//! Fold ownership is a local compile-time region. Each admitted ownership arm
//! is expanded once, preserving the enclosing computation and original state
//! bindings; independent fold sites never multiply complete source programs.
use crate::execution::{self, FoldChoice, FoldOwnership, Stage};
use magnitude_solver::model::{Constraint, Domain, Literal, ModelBuilder, VarId};
use seismic_accounting::choices::Choices;
use seismic_lang::{ast::AssignOp, ir::*, lowered_ir::LoweredIr, types::{DType, Ty}};
use std::{collections::BTreeMap, sync::Arc};

pub(super) struct Retained {
    pub stage: Stage,
    pub predicates: Vec<(seismic_lang::ir::VarId, Literal)>,
    pub private_values: Vec<(seismic_lang::ir::VarId, Literal)>,
}

pub(super) fn construct(
    stage: &Stage,
    parameters: &BTreeMap<usize, VarId>,
    builder: &mut ModelBuilder,
    name: &str,
    selectors: &BTreeMap<String, VarId>,
    phase_presence: &[Option<VarId>],
) -> Result<Option<Retained>, String> {
    let Some(prepared) = stage.prepared_folds() else { return Ok(None); };
    let mut prepared = prepared.clone();
    let domains = stage.local_domains()?.into_iter().map(|domain| {
        let crate::choices::Decision::Fold(choice) = domain.decision else { unreachable!() };
        (choice.site, choice)
    }).collect::<BTreeMap<_, _>>();
    let mut retained = Constructor { function: prepared.function.clone(), domains, parameters,
        builder, name, selectors, guards: Vec::new(), site: 0, predicates: Vec::new(), private_values: Vec::new() };
    for (phase_index, (statement, phase)) in prepared.function.body.iter_mut().zip(&mut prepared.phases).enumerate() {
        let phase_guard = phase_presence.get(phase_index).copied().flatten().map(|active| Literal::new(active, 1)).into_iter().collect::<Vec<_>>();
        retained.guards = phase_guard.clone();
        let StmtKind::Parallel { body, .. } = &mut statement.kind else { return Err("retained fold lost its launch boundary".into()); };
        if let Some(split) = &mut phase.split {
            let ordinary_at = split.retained.as_ref().map_or(body.len(), |retained| retained.ordinary_at);
            if split.loop_at >= ordinary_at || ordinary_at > body.len() { return Err("retained split boundaries are invalid".into()); }
            if let Some(split) = &split.retained {
                let active = *selectors.get(&split.selector).ok_or("retained split fold has no compiler selector")?;
                retained.guards.push(Literal::new(active, 1));
            }
            let mut transformed = retained.block(&body[..split.loop_at])?;
            let loop_at = transformed.len();
            transformed.extend(retained.block(&body[split.loop_at..=split.loop_at])?);
            transformed.extend(retained.block(&body[split.loop_at + 1..ordinary_at])?);
            if let Some(split) = &mut split.retained { split.ordinary_at = transformed.len(); }
            retained.guards = phase_guard;
            transformed.extend(retained.block(&body[ordinary_at..])?);
            split.loop_at = loop_at;
            *body = transformed;
        } else { *body = retained.block(body)?; }
    }
    prepared.function.vars = retained.function.vars;
    // All generated arm bodies are ordinary typed statements. Preserve the
    // selected split's original statement coordinates while lifting empty forms.
    for (statement, phase) in prepared.function.body.iter_mut().zip(&mut prepared.phases) {
        let StmtKind::Parallel { body, .. } = &mut statement.kind else { return Err("retained fold lost its launch boundary".into()); };
        let positions = seismic_lang::normalize::lift_owned_reductions(body);
        if let Some(split) = &mut phase.split { split.loop_at = positions[split.loop_at]; if let Some(retained) = &mut split.retained { retained.ordinary_at = positions[retained.ordinary_at]; } }
        let positions = seismic_lang::normalize::remove_empty_ranges(body);
        if let Some(split) = &mut phase.split { split.loop_at = positions[split.loop_at]; if let Some(retained) = &mut split.retained { retained.ordinary_at = positions[retained.ordinary_at]; } }
    }
    Ok(Some(Retained { stage: Stage::Loads(Arc::new(prepared)), predicates: retained.predicates,
        private_values: retained.private_values }))
}

struct Constructor<'a> {
    function: LoweredIr,
    domains: BTreeMap<usize, FoldChoice>,
    parameters: &'a BTreeMap<usize, VarId>,
    builder: &'a mut ModelBuilder,
    name: &'a str,
    selectors: &'a BTreeMap<String, VarId>,
    guards: Vec<Literal>,
    site: usize,
    predicates: Vec<(seismic_lang::ir::VarId, Literal)>,
    private_values: Vec<(seismic_lang::ir::VarId, Literal)>,
}
impl Constructor<'_> {
    fn block(&mut self, body: &[Stmt]) -> Result<Vec<Stmt>, String> {
        let mut output = Vec::new();
        for original in body {
            let mut statement = original.clone();
            match &mut statement.kind {
                StmtKind::Reduction(reduction) => {
                    let site = self.site; self.site += 1;
                    let Some(choice) = self.domains.get(&site).cloned() else {
                        let expanded = reduction.expand(reduction.tree.ok_or("fold tree is unresolved")?, &mut self.function.vars)?;
                        output.extend(expanded); continue;
                    };
                    let parameter = *self.parameters.get(&site).ok_or("fold ownership has no retained parameter")?;
                    let active = super::layout::conjunction(self.builder, &format!("{}.fold{site}.active", self.name), &self.guards)?;
                    self.builder.constraint(Constraint::InactiveValue { active: Literal::new(active, 1), variable: parameter, inactive: 0 });
                    let mut chain = Vec::new();
                    for ordinal in (0..choice.len()).rev() {
                        let ownership = choice.get(ordinal).ok_or("empty fold ownership arm")?;
                        let mut local = self.function.clone();
                        local.body = vec![original.clone()];
                        let selected = match ownership {
                            FoldOwnership::Serial => Vec::new(),
                            ownership => {
                                use seismic_lang::reduction::structured::participants::{Completion, SeedPlacement, Selection};
                                let (seed, completion) = match ownership {
                                    FoldOwnership::Participants => (SeedPlacement::LeadingLeaf, Completion::RetainLeaves),
                                    FoldOwnership::ParticipantsInsertSeed => (SeedPlacement::InsertAfterSegments, Completion::RetainLeaves),
                                    FoldOwnership::ParticipantsWavefront => (SeedPlacement::LeadingLeaf, Completion::CompleteWaves),
                                    FoldOwnership::ParticipantsWavefrontInsertSeed => (SeedPlacement::InsertAfterSegments, Completion::CompleteWaves),
                                    FoldOwnership::ParticipantsRootSeed => (SeedPlacement::AtRoot, Completion::RetainLeaves),
                                    FoldOwnership::ParticipantsWavefrontRootSeed => (SeedPlacement::AtRoot, Completion::CompleteWaves),
                                    FoldOwnership::Serial => unreachable!(),
                                };
                                vec![Selection { site: 0, seed, completion }]
                            }
                        };
                        let refinement = seismic_lang::reduction::structured::participants::apply(&local, &selected, execution::SUBGROUP as u32)?;
                        let local = seismic_lang::reduction::structured::materialize(&refinement.function)?;
                        self.function.vars = local.vars;
                        let guard = Literal::new(parameter, ordinal as i64);
                        self.private_values.extend(refinement.private_values.into_iter().map(|variable| (variable, guard.clone())));
                        let predicate = self.function.vars.len();
                        let variable = self.builder.local_variable(format!("{}.fold{site}.arm{ordinal}", self.name), Domain::boolean())
                            .map_err(|error| error.to_string())?;
                        let value = self.builder.variable(format!("{}.fold{site}.ordinal{ordinal}", self.name), Domain::singleton(ordinal as i64));
                        self.builder.guarded_constraint(vec![Literal::new(variable, 1)], Constraint::Equal { left: parameter, right: value });
                        self.builder.guarded_constraint(vec![Literal::new(variable, 0)], Constraint::NotEqual { left: parameter, right: value });
                        self.function.vars.push(Var { name: format!("fold_choice_{predicate}"), ty: Ty::Scalar(DType::Bool),
                            kind: VarKind::Local, span: original.span });
                        self.predicates.push((predicate, Literal::new(variable, 1)));
                        let condition = Expr { kind: ExprKind::Var(predicate), ty: Ty::Scalar(DType::Bool), sym: None, span: original.span };
                        chain = vec![Stmt { id: None, span: original.span, kind: StmtKind::Assign {
                            target: condition.clone(), op: AssignOp::Assign, value: Expr { kind: ExprKind::Bool(false),
                                ty: Ty::Scalar(DType::Bool), sym: None, span: original.span } } },
                            Stmt { id: None, span: original.span, kind: StmtKind::If { cond: condition, then: local.body, els: chain } }];
                    }
                    output.extend(chain); continue;
                }
                StmtKind::Parallel { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. } | StmtKind::Lanes { body, .. } => *body = self.block(body)?,
                StmtKind::If { cond, then, els } => {
                    let predicate = match cond.kind {
                        ExprKind::Var(variable) => self.selectors.get(&crate::msl::variable_symbol(&self.function.vars[variable], variable)).copied(),
                        _ => None,
                    };
                    let enclosing = self.guards.clone();
                    if let Some(predicate) = predicate { self.guards.push(Literal::new(predicate, 1)); }
                    *then = self.block(then)?;
                    self.guards = enclosing.clone();
                    if let Some(predicate) = predicate { self.guards.push(Literal::new(predicate, 0)); }
                    *els = self.block(els)?;
                    self.guards = enclosing;
                }
                _ => {},
            }
            output.push(statement);
        }
        Ok(output)
    }
}
