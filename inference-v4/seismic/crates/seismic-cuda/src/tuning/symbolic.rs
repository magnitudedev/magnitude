//! CUDA target choices and their shared dispatch equations. Numeric domains are
//! original legal domains; no block size or ownership alternative is sampled.
use super::*;
use magnitude_solver::model::{
    Arithmetic, Constraint, Domain, LinearTerm, Literal, ModelBuilder, ObligationKind, VarId,
};
use seismic_accounting::algebra::{Algebra, Symbolic, Value};
use seismic_compiler::tuner::{
    family::Decision,
    geometry::{Error, Geometry},
};
use seismic_lang::ir::{LoadMode, Stmt, StmtKind};

pub struct LoadChoice {
    pub source_variable: seismic_lang::ir::VarId,
    pub identity: String,
    pub variable: VarId,
    pub modes: Vec<LoadMode>,
}
pub struct FoldBinding {
    pub identity: String,
    pub variable: VarId,
    pub choice: FoldChoice,
}
pub struct LaunchRegion {
    pub dispatch: Dispatch,
    pub presence: VarId,
    pub phases: Vec<Geometry>,
    pub obligation: Option<String>,
}
/// Shared source owner plus every CUDA-local decision, including alternatives
/// whose terminal operations still need a parameterized accounting relation.
pub struct Binding {
    pub loads: Vec<LoadChoice>,
    pub folds: Vec<FoldBinding>,
    pub dispatch: VarId,
    pub lanes: Value,
    pub regions: Vec<LaunchRegion>,
    function: LoweredIr,
}
impl Binding {
    pub fn append(
        builder: &mut ModelBuilder,
        function: &LoweredIr,
        device: &DeviceInfo,
    ) -> Result<Self, String> {
        validate_device(device)?;
        if function.backend != "cuda" { return Err("CUDA family requires CUDA source".into()); }
        seismic_lang::verify::lowered(function, seismic_lang::verify::Stage::Expanded)?;
        let mut function = function.clone();
        seismic_lang::normalize::bind_values(&mut function.body, &mut function.vars);
        let mut loads = Vec::new();
        let ownership = loads::Family::new(function.clone())?;
        for (site, (load, domain)) in ownership.sites().iter().zip(ownership.domains()).enumerate() {
            let var = &function.vars[load.variable];
            let identity = format!(
                "cuda.{}.load.{}@{}:{}.site{}",
                function.name, var.name, var.span.start, var.span.end, site
            );
            let modes = domain.clone();
            let variable = builder.variable(
                &identity,
                Domain::interval(0, modes.len() as i64 - 1).map_err(|e| e.to_string())?,
            );
            loads.push(LoadChoice {
                source_variable: load.variable,
                identity,
                variable,
                modes,
            });
        }
        let explicit_subgroup = seismic_compiler::subgroup_required(&function);
        if explicit_subgroup && device.warp_size != 32 {
            return Err("CUDA subgroup source requires a 32-lane warp".into());
        }
        let mut subgroup = builder.variable(
            "cuda.source.subgroup",
            Domain::singleton(i64::from(explicit_subgroup)),
        );
        let fold_sites = reduction_sites(&function.body);
        let mut folds = Vec::new();
        for choice in fold_choices(&function, device) {
            let span = fold_sites
                .get(choice.site)
                .ok_or("missing CUDA reduction occurrence")?;
            let identity = format!(
                "cuda.{}.fold.{}:{}.site{}",
                function.name, span.start, span.end, choice.site
            );
            let variable = builder.variable(
                &identity,
                Domain::interval(0, choice.len() as i64 - 1).map_err(|e| e.to_string())?,
            );
            let active = builder.variable(format!("{identity}.subgroup"), Domain::boolean());
            builder.constraint(Constraint::Table {
                variables: vec![variable, active],
                tuples: (0..choice.len())
                    .map(|i| vec![i as i64, i64::from(i != 0)])
                    .collect(),
            });
            let combined = builder.variable(format!("{identity}.any_subgroup"), Domain::boolean());
            builder.constraint(Constraint::Arithmetic(Arithmetic::Maximum {
                left: subgroup,
                right: active,
                result: combined,
            }));
            subgroup = combined;
            folds.push(FoldBinding {
                identity,
                variable,
                choice,
            });
        }
        let lane_domain = if explicit_subgroup {
            Domain::singleton(32)
        } else if folds.is_empty() {
            Domain::singleton(1)
        } else {
            Domain::interval(1, 32).map_err(|e| e.to_string())?
        };
        let lane_id = builder.variable("cuda.participants", lane_domain.clone());
        for sign in [1, -1] {
            builder.constraint(Constraint::LinearLe {
                terms: vec![
                    LinearTerm::new(lane_id, sign),
                    LinearTerm::new(subgroup, -31 * sign),
                ],
                rhs: i128::from(sign),
            });
        }
        let lanes = Value::binding(lane_id, &lane_domain).map_err(|e| e.to_string())?;
        let dispatch = builder.variable("cuda.dispatch", Domain::boolean());
        builder.constraint(Constraint::Implies {
            premise: Literal::new(subgroup, 1),
            consequence: Literal::new(dispatch, 1),
        });
        let mut regions = Vec::new();
        for (ordinal, kind) in [(0, Dispatch::Sequential), (1, Dispatch::ParallelRoot)] {
            let presence = equals(
                builder,
                dispatch,
                ordinal,
                &format!("cuda.dispatch{ordinal}.active"),
            )?;
            let work = if kind == Dispatch::Sequential {
                Ok(vec![1])
            } else {
                match seismic_realization::phases::assess(&function)? {
                    seismic_realization::phases::Applicability::Supported(plan) => plan
                        .function
                        .body
                        .iter()
                        .map(|statement| {
                            let StmtKind::Parallel { extents, .. } = &statement.kind else {
                                return Err("CUDA phase lacks its work domain".to_string());
                            };
                            extents.iter().try_fold(1u64, |count, extent| {
                                count
                                    .checked_mul(
                                        extent
                                            .as_constant()
                                            .and_then(|n| u64::try_from(n).ok())
                                            .ok_or("CUDA phase extent is not bounded")?,
                                    )
                                    .ok_or_else(|| "CUDA work domain exceeds u64".to_string())
                            })
                        })
                        .collect(),
                    seismic_realization::phases::Applicability::Unresolved { reason } => {
                        Err(reason)
                    }
                }
            };
            let mut phases = Vec::new();
            let mut obligation = None;
            match work {
                Ok(work) if work.is_empty() => {
                    obligation=Some("CUDA dispatch has no retained execution phase".into());
                }
                Ok(work) => {
                    // Conditional arithmetic is transactional. An unsupported
                    // integer width never leaves constraints narrowing the family.
                    let mut next = builder.clone();
                    let result =
                        next.when(Literal::new(presence, 1), |builder| -> Result<(), Error> {
                            for (phase, work) in work.into_iter().enumerate() {
                                let maximum =
                                    u64::from(device.max_threads_per_block) / lanes.bounds().0;
                                let minimum = work
                                    .div_ceil(u64::from(device.max_grid_x))
                                    .max(1)
                                    .min(maximum);
                                let mut a = Symbolic::new(builder, "cuda.launch");
                                let one = a.constant(1)?;
                                let items = a.positive(
                                    &format!("dispatch{ordinal}.phase{phase}.items"),
                                    Domain::interval(minimum as i64, maximum as i64)
                                        .map_err(|e| Error::Invalid(e.to_string()))?,
                                )?;
                                let geometry = Geometry::from_values(
                                    builder,
                                    &format!("cuda.dispatch{ordinal}.phase{phase}"),
                                    &[work],
                                    &[one],
                                    lanes,
                                    items,
                                    &[],
                                )?;
                                builder.constraint(Constraint::LinearLe {
                                    terms: vec![LinearTerm::new(
                                        geometry.threads_per_group.id(),
                                        1,
                                    )],
                                    rhs: i128::from(device.max_threads_per_block),
                                });
                                builder.constraint(Constraint::LinearLe {
                                    terms: vec![LinearTerm::new(geometry.groups.id(), 1)],
                                    rhs: i128::from(device.max_grid_x),
                                });
                                phases.push(geometry);
                            }
                            Ok(())
                        });
                    match result {
                        Ok(()) => *builder = next,
                        Err(Error::Unsupported(reason)) => {
                            phases.clear();
                            obligation = Some(reason);
                        }
                        Err(error) => return Err(error.to_string()),
                    }
                }
                Err(reason) => obligation = Some(format!("CUDA phase construction: {reason}")),
            }
            if let Some(reason) = &obligation {
                builder.obligation(
                    vec![Literal::new(presence, 1)],
                    ObligationKind::Analysis,
                    reason,
                );
            }
            regions.push(LaunchRegion {
                dispatch: kind,
                presence,
                phases,
                obligation,
            });
        }
        Ok(Self {
            loads,
            folds,
            dispatch,
            lanes,
            regions,
            function,
        })
    }
    pub fn function(&self) -> &LoweredIr {
        &self.function
    }
    pub fn reconstruct(
        &self,
        values: &[i64],
    ) -> Result<(Selection, Vec<u32>, Vec<Decision>, usize), String> {
        let read = |id: VarId| {
            values
                .get(id.0)
                .copied()
                .ok_or("missing CUDA family assignment".to_string())
        };
        let mut decisions = Vec::new();
        let mut modes = Vec::new();
        for choice in &self.loads {
            let value = read(choice.variable)?;
            let mode = choice
                .modes
                .get(usize::try_from(value).map_err(|_| "negative CUDA load choice")?)
                .ok_or("CUDA load choice outside its original domain")?;
            modes.push(*mode);
            decisions.push(Decision {
                identity: choice.identity.clone(),
                value,
            });
        }
        let mut folds = Vec::new();
        for choice in &self.folds {
            let value = read(choice.variable)?;
            folds.push((
                choice.choice.site,
                choice
                    .choice
                    .get(usize::try_from(value).map_err(|_| "negative CUDA fold choice")?)
                    .ok_or("CUDA fold choice outside its original domain")?,
            ));
            decisions.push(Decision {
                identity: choice.identity.clone(),
                value,
            });
        }
        let index =
            usize::try_from(read(self.dispatch)?).map_err(|_| "negative CUDA dispatch choice")?;
        let region = self
            .regions
            .get(index)
            .ok_or("CUDA dispatch choice outside its original domain")?;
        if region.obligation.is_some() {
            return Err("CUDA phase construction remains unresolved".into());
        }
        if read(region.presence)? != 1 {
            return Err("CUDA selected dispatch is inactive".into());
        }
        let geometries = region
            .phases
            .iter()
            .map(|geometry| geometry.reconstruct(values).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let selection = Selection {
            loads: modes,
            folds,
            dispatch: region.dispatch,
        };
        let threads = geometries
            .iter()
            .map(|g| {
                u32::try_from(g.dispatch.threads_per_group)
                    .map_err(|_| "CUDA block exceeds u32".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        decisions.push(Decision {
            identity: "cuda.dispatch".into(),
            value: index as i64,
        });
        for (phase, geometry) in geometries.iter().enumerate() {
            decisions.push(Decision {
                identity: format!("cuda.dispatch{index}.phase{phase}.items"),
                value: geometry.dispatch.items_per_group as i64,
            });
        }
        Ok((selection, threads, decisions, index))
    }
}
fn equals(
    builder: &mut ModelBuilder,
    variable: VarId,
    value: i64,
    name: &str,
) -> Result<VarId, String> {
    let output = builder.variable(name, Domain::boolean());
    let constant = builder.variable(format!("{name}.value"), Domain::singleton(value));
    builder.guarded_constraint(
        vec![Literal::new(output, 1)],
        Constraint::Equal {
            left: variable,
            right: constant,
        },
    );
    builder.guarded_constraint(
        vec![Literal::new(output, 0)],
        Constraint::NotEqual {
            left: variable,
            right: constant,
        },
    );
    Ok(output)
}
fn reduction_sites(body: &[Stmt]) -> Vec<seismic_lang::span::Span> {
    fn visit(body: &[Stmt], sites: &mut Vec<seismic_lang::span::Span>) {
        for statement in body {
            match &statement.kind {
                StmtKind::Reduction(_) => sites.push(statement.span),
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, sites),
                StmtKind::If { then, els, .. } => {
                    visit(then, sites);
                    visit(els, sites);
                }
                _ => {}
            }
        }
    }
    let mut sites = Vec::new();
    visit(body, &mut sites);
    sites
}
