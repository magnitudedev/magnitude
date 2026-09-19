//! One source/target/accounting export. Local control alternatives share retained
//! PTX and the caller's schedule; there is no search or assignment re-lowering here.
use super::*;
use compiler::{family::{Export, Reconstructed, Reconstruction}, source};
use magnitude_solver::{FeasibleSolution, model::{Constraint, Cost, Domain, LinearTerm, Literal, ModelBuilder, ObligationKind, VarId}};
use seismic_accounting::{schedule::{self, guarded::Binding as Account, symbolic::Encoding}, workload::DerivationError};
use seismic_compiler::template::{ScalarFamily, SequenceFamily, PhaseParameter};
use std::collections::BTreeMap;

struct TerminalPhase {
    template: crate::ptx::family::TargetFamily,
    parameters: Vec<VarId>,
}
struct Terminal { phases: Vec<TerminalPhase> }
struct Original {
    source: source::Binding,
    target: Option<symbolic::Binding>,
    terminals: Vec<Option<Terminal>>,
    accounts: Vec<Option<Account>>,
    limits: Limits,
}
impl Reconstruction<Vec<Execution>> for Original {
    fn reconstruct(&self, witness: &FeasibleSolution, lower_bound: u64) -> Result<Reconstructed<Vec<Execution>>, String> {
        let values = witness.values();
        let (source, mut decisions) = self.source.reconstruct(values)?;
        let target = self.target.as_ref().ok_or("CUDA terminal family construction remains unresolved")?;
        let (selection, widths, target_decisions, region) = target.reconstruct(values)?;
        if selection.folds.iter().any(|(_,fold)|*fold != FoldImplementation::Thread) {
            return Err("CUDA fold participation template remains unresolved".into());
        }
        let terminal = self.terminals.get(region).and_then(Option::as_ref).ok_or("CUDA terminal region remains unresolved")?;
        if widths.len()!=terminal.phases.len() {return Err("CUDA retained terminal phase arity differs from selected geometry".into());}
        let conditions=seismic_realization::InvocationConditions::from_lowered(&source)?;
        let mut execution=Vec::new();
        for ((phase,&width),geometry) in terminal.phases.iter().zip(&widths).zip(&target.regions[region].phases) {
            let assignment=phase.parameters.iter().map(|id|values.get(id.0).and_then(|&value|usize::try_from(value).ok()).ok_or("CUDA terminal parameter is absent")).collect::<Result<Vec<_>,_>>()?;
            let (mut program,plan)=phase.template.instantiate(&assignment)?;
            program.conditions=conditions.clone();
            let selected=Execution::from_plan(Arc::new(program),Arc::new(plan),width,self.limits)?;
            if selected.dispatch()!=&geometry.reconstruct(values).map_err(|e|e.to_string())?.dispatch {return Err("CUDA reconstruction changed exported launch geometry".into());}
            execution.push(selected);
        }
        let objective = self.accounts.get(region).and_then(Option::as_ref).ok_or("CUDA terminal accounting remains unresolved")?.reconstruct(values,lower_bound)?;
        decisions.extend(target_decisions);
        Ok(Reconstructed { execution, source, objective, decisions })
    }
}

pub(super) fn export(input: compiler::Input<'_>, conditions: &Conditions, workload: &ScalarWorkload, limits: DerivationLimits) -> Result<Export<Vec<Execution>>, String> {
    workload.validate()?;
    let family = source::Binding::construct(input, "cuda")?;
    let mut builder = ModelBuilder::new();
    builder.units(format!("CUDA ticks ({} / {} seconds)",conditions.hardware.timebase.seconds_numerator,conditions.hardware.timebase.seconds_denominator));
    let source = source::Binding::append(&mut builder, family.clone())?;
    let mut original = Original { source, target:None, terminals:Vec::new(), accounts:Vec::new(), limits:Limits {max_threads_per_block:conditions.device.max_threads_per_block,max_grid_x:conditions.device.max_grid_x} };
    append_source_counts(&mut builder,&mut original.source)?;
    let fixed = if family.decisions().is_empty() && family.obligations().is_empty() { Some(family.instantiate(&Default::default())?) } else { None };
    // Target domains use the typed union before any scalar launch is emitted.
    let union=if fixed.is_none() {
        match SequenceFamily::source_from_family(&family) {
            Ok(source)=>Some(source),
            Err(error)=>{builder.obligation(vec![],ObligationKind::Construction,error.to_string());return finish(builder,original);},
        }
    } else {None};
    let target_source=fixed.as_ref().or(union.as_ref()).ok_or("missing CUDA source template")?;
    let target = symbolic::Binding::append(&mut builder,target_source,&conditions.device)?;
    if fixed.is_none() && family.regions().iter().any(|region|matches!(&region.kind,seismic_lang::family::RegionKind::Reduction(_))) {
        builder.obligation(vec![],ObligationKind::Construction,"CUDA source-dependent fold ownership needs retained participation regions before its original target alternatives can be exported");
    }
    let mut models = Vec::new();
    let mut resources = Vec::<schedule::Resource>::new();
    let mut horizon = 0;
    for (index,region) in target.regions.iter().enumerate() {
        let region_guard = vec![Literal::new(region.presence,1)];
        if region.obligation.is_some() {original.terminals.push(None);models.push(None);continue;}
        let participation=if seismic_compiler::subgroup_required(target.function()) {seismic_realization::dispatch::Participation::Subgroup {lanes:32}} else {seismic_realization::dispatch::Participation::Thread};
        let terminal=if region.dispatch==Dispatch::ParallelRoot {
            let sequence=if fixed.is_some() {SequenceFamily::new(target.function(),CallConv::SystemV,participation)} else {SequenceFamily::from_family(&family,CallConv::SystemV,participation)};
            sequence.map_err(|e|e.to_string()).and_then(|sequence|bind_sequence(&mut builder,&mut original.source,&target,sequence))
        } else {
            let scalar=if fixed.is_some() {ScalarFamily::new(target.function(),CallConv::SystemV,region.dispatch,participation)} else {ScalarFamily::from_family(&family,CallConv::SystemV,region.dispatch,participation).map_err(|e|e.to_string())};
            scalar.and_then(|scalar|bind_terminal(&mut builder,&mut original.source,&target,scalar))
        };
        let terminal=match terminal {
            Ok(terminal)=>terminal,
            Err(reason)=>{builder.obligation(region_guard,ObligationKind::Construction,reason);original.terminals.push(None);models.push(None);continue;},
        };
        if terminal.phases.len()!=region.phases.len() {return Err("CUDA publication template differs from original phase geometry".into());}
        for fold in &target.folds {
            for ordinal in 1..fold.choice.len() {
                builder.obligation(vec![Literal::new(region.presence,1),Literal::new(fold.variable,ordinal as i64)],ObligationKind::Construction,"CUDA reduction participation needs retained scalar/PTX ownership arms");
            }
        }
        let lanes=target.lanes.bounds();
        if region.phases.iter().any(|phase| {let width=phase.threads_per_group.bounds();width.0!=width.1}) || lanes.0!=lanes.1 {
            builder.obligation(region_guard,ObligationKind::Analysis,format!("CUDA {:?} retains original block-size and participation equations; terminal cohort membership, service and residency must be expressed in those variables",region.dispatch));
            original.terminals.push(Some(terminal));models.push(None);continue;
        }
        let executions=terminal.phases.iter().zip(&region.phases).map(|(phase,geometry)| {
            Execution::from_plan(Arc::new(phase.template.scalar().program().clone()),Arc::new(phase.template.plan().clone()),u32::try_from(geometry.threads_per_group.bounds().0).map_err(|_|"CUDA block width overflow")?,original.limits)
        }).collect::<Result<Vec<_>,String>>()?;
        let templates=terminal.phases.iter().map(|phase|&phase.template).collect::<Vec<_>>();
        let derived=crate::model::derive_family_sequence(&executions,&templates,&conditions.hardware,workload,limits);
        let mut derived = match derived {
            Ok(derived)=>derived,
            Err(DerivationError::Analysis(reason))=>return Err(reason),
            Err(DerivationError::Exhausted(limit))=>{builder.obligation(region_guard,ObligationKind::Construction,format!("CUDA terminal construction exhausted {limit:?}"));original.terminals.push(Some(terminal));models.push(None);continue;},
            Err(error)=>{builder.obligation(region_guard,ObligationKind::Analysis,error.to_string());original.terminals.push(Some(terminal));models.push(None);continue;},
        };
        // This homogeneous pool is a reduction of the region's complete
        // residency vector, so its capacity may differ between exclusive
        // dispatch regions. Physical service resource capacities stay shared.
        for resource in &mut derived.model.resources {
            if resource.name.ends_with("cuda.homogeneous-resident-slots") { resource.name=format!("dispatch{index}:{}",resource.name); }
        }
        let mut remap = Vec::new();
        for resource in &derived.model.resources {
            let id = if let Some(id)=resources.iter().position(|old|old.name==resource.name) {
                if resources[id]!=*resource { return Err("CUDA regions disagree on shared resource capacity".into()); } id
            } else {let id=resources.len();resources.push(resource.clone());id};
            remap.push(id);
        }
        for operation in &mut derived.model.operations {for reservation in &mut operation.reservations {reservation.resource=remap[reservation.resource];}}
        for lifetime in &mut derived.model.lifetimes {lifetime.resource=remap[lifetime.resource];}
        horizon=horizon.max(schedule::export::horizon(&derived.model.clone().into())?);
        let presence=derived.guards.iter().map(|guard|bound_activation(&mut builder,guard,Some(region.presence))).collect::<Result<Vec<_>,_>>()?;
        original.terminals.push(Some(terminal));models.push(Some((derived.model,presence)));
    }
    let mut encoding=Encoding::new(&mut builder,"cuda.sequence",&resources,horizon).map_err(|e|e.to_string())?;
    encoding.bind_timebase(&conditions.hardware.timebase).map_err(|e|e.to_string())?;
    for model in models {
        original.accounts.push(if let Some((mut model,presence))=model {
            model.resources=resources.clone();Some(Account::append(&mut builder,&mut encoding,Arc::new(model),presence)?)
        } else {None});
    }
    let completion=encoding.finish(&mut builder).map_err(|e|e.to_string())?;
    builder.cost(Cost::Linear {constant:0,terms:vec![LinearTerm::new(completion,1)]});
    original.target=Some(target);
    finish(builder,original)
}
fn finish(builder:ModelBuilder,original:Original)->Result<Export<Vec<Execution>>,String> {Export::new(Arc::new(builder.build().map_err(|e|e.to_string())?),original)}
fn bind_parameters(
    builder:&mut ModelBuilder,source:&mut source::Binding,target:&symbolic::Binding,
    sites:&[seismic_lang::normalize::loads::Site],domains:&[Vec<seismic_lang::ir::LoadMode>],
    source_parameters:&[(seismic_lang::family::DecisionId,usize)],load_guards:&[BTreeMap<usize,bool>],
)->Result<Vec<VarId>,String> {
    let mut parameters=Vec::new();
    if sites.len()!=target.loads.len() {return Err("CUDA scalar template changed the original load ownership occurrences".into());}
    for ((site,domain),choice) in sites.iter().zip(domains).zip(&target.loads) {
        if choice.source_variable!=site.variable || &choice.modes!=domain {return Err("CUDA retained load occurrence or domain differs from its original ownership family".into());}
        parameters.push(choice.variable);
    }
    for (decision,ordinal) in source_parameters {
        let mut guard=source.family().decisions().iter().find(|entry|&entry.id==decision).ok_or("unknown CUDA source selector")?.guard.clone();
        guard.choices.push((decision.clone(),*ordinal));
        parameters.push(source.presence(builder,&guard)?.ok_or("CUDA source selector has no activation condition")?);
    }
    for (index,guard) in load_guards.iter().enumerate() {
        if let Some(active)=activation(builder,&parameters,guard,None)? {
            let zero=builder.variable("cuda.inactive-load",Domain::singleton(0));
            builder.guarded_constraint(vec![Literal::new(active,0)],Constraint::Equal {left:parameters[index],right:zero});
        }
    }
    Ok(parameters)
}
fn phase(scalar:ScalarFamily,parameters:Vec<VarId>)->Result<TerminalPhase,String> {
    let mut template=crate::ptx::family::TargetFamily::new(scalar)?;
    template.bind_parameters(&parameters.iter().map(|id|id.0).collect::<Vec<_>>())?;
    Ok(TerminalPhase {template,parameters})
}
fn bind_terminal(builder:&mut ModelBuilder,source:&mut source::Binding,target:&symbolic::Binding,scalar:ScalarFamily)->Result<Terminal,String> {
    let parameters=bind_parameters(builder,source,target,scalar.sites(),scalar.domains(),scalar.source_parameters(),scalar.load_guards())?;
    Ok(Terminal {phases:vec![phase(scalar,parameters)?]})
}
fn bind_sequence(builder:&mut ModelBuilder,source:&mut source::Binding,target:&symbolic::Binding,sequence:SequenceFamily)->Result<Terminal,String> {
    let original=bind_parameters(builder,source,target,sequence.sites(),sequence.domains(),sequence.source_parameters(),sequence.load_guards())?;
    let mut phases=Vec::new();
    for definition in sequence.into_phases() {
        let parameters=definition.parameters.iter().map(|parameter|match parameter {
            PhaseParameter::Original(index)=>original.get(*index).copied().ok_or("CUDA publication parameter lost its original identity".to_string()),
            PhaseParameter::Fixed(value)=>Ok(builder.variable("cuda.publication.fixed-load",Domain::singleton(i64::try_from(*value).map_err(|_|"CUDA publication ordinal overflow")?))),
        }).collect::<Result<Vec<_>,String>>()?;
        phases.push(phase(definition.scalar,parameters)?);
    }
    Ok(Terminal {phases})
}
fn bound_activation(builder:&mut ModelBuilder,guard:&BTreeMap<usize,bool>,parent:Option<VarId>)->Result<Option<VarId>,String> {
    let mut terms=parent.into_iter().collect::<Vec<_>>();
    for (&parameter,&value) in guard {
        let variable=VarId(parameter);
        terms.push(if value {variable} else {let inverse=builder.variable("cuda.parameter.false",Domain::boolean());builder.constraint(Constraint::NotEqual {left:inverse,right:variable});inverse});
    }
    Ok(match terms.as_slice() {[]=>None,[only]=>Some(*only),_=>{let active=builder.variable("cuda.region.active",Domain::boolean());builder.constraint(Constraint::BoolAnd {output:active,inputs:terms});Some(active)}})
}
fn activation(builder:&mut ModelBuilder,parameters:&[VarId],guard:&BTreeMap<usize,bool>,parent:Option<VarId>)->Result<Option<VarId>,String> {
    let mut terms=parent.into_iter().collect::<Vec<_>>();
    for (&parameter,&value) in guard {
        let variable=*parameters.get(parameter).ok_or("CUDA terminal guard references an unknown parameter")?;
        terms.push(if value {variable} else {
            let inverse=builder.variable("cuda.parameter.false",Domain::boolean());
            builder.constraint(Constraint::NotEqual {left:inverse,right:variable});inverse
        });
    }
    Ok(match terms.as_slice() {[]=>None,[only]=>Some(*only),_=>{let active=builder.variable("cuda.region.active",Domain::boolean());builder.constraint(Constraint::BoolAnd {output:active,inputs:terms});Some(active)}})
}

/// Retain original repeat/tail equations even while the corresponding terminal
/// template needs conversion. Runtime controls remain universal workload facts.
fn append_source_counts(
    builder: &mut ModelBuilder,
    source: &mut source::Binding,
) -> Result<(), String> {
    use seismic_lang::family::RegionKind;
    let regions = source.family().regions().to_vec();
    for region in regions {
        let counts = match &region.kind {
            RegionKind::Repeated { repetition, .. } => repetition.counts.clone(),
            RegionKind::Stream { geometry, .. } => vec![
                geometry.complete_pieces.clone(),
                geometry.tail_extent.clone(),
                geometry.visits.clone(),
            ],
            _ => Vec::new(),
        };
        if counts.is_empty() {
            continue;
        }
        let presence = source.presence(builder, &region.guard)?;
        let mut next = builder.clone();
        let mut expressions = compiler::expressions::Expressions::new(source.parameters().clone());
        let mut append =
            |builder: &mut ModelBuilder| -> Result<(), seismic_accounting::algebra::Error> {
                for (axis, count) in counts.iter().enumerate() {
                    expressions.nonnegative(
                        builder,
                        &format!("cuda.{:?}.count{axis}", region.occurrence),
                        count,
                    )?;
                }
                Ok(())
            };
        let result = match presence {
            Some(active) => next.when(Literal::new(active, 1), append),
            None => append(&mut next),
        };
        match result {
            Ok(()) => *builder = next,
            Err(seismic_accounting::algebra::Error::Unsupported(reason)) => {
                let guard = source.literals(builder, &region.guard)?;
                builder.obligation(
                    guard,
                    ObligationKind::Analysis,
                    reason,
                );
            }
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
}
