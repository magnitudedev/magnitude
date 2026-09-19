//! CPU family contribution: original load domains, one conditional SSA graph,
//! guarded terminal accounting and joint instruction scheduling. Reconstruction
//! substitutes retained operands and materializes the common static order.
use magnitude_solver::model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder, Literal, ObligationKind, VarId};
use seismic_accounting::{execution_model::{self, ScalarHardware}, schedule::{self, guarded::Binding as Account, symbolic::Encoding}, workload::{DerivationError, DerivationLimits, ScalarWorkload}};
use seismic_compiler::{template::ScalarFamily, tuner::{family::{Decision, Export, Reconstructed, Reconstruction}, source::Binding, Input}};
use seismic_realization::{scheduling, ScalarProgram};
use std::{collections::BTreeMap, sync::Arc};

struct Plan {
    source: Binding,
    template: Option<ScalarFamily>,
    parameters: Vec<VarId>,
    account: Option<Account>,
}
impl Reconstruction<ScalarProgram> for Plan {
    fn reconstruct(&self, witness:&magnitude_solver::FeasibleSolution, lower_bound:u64)->Result<Reconstructed<ScalarProgram>,String> {
        let values=witness.values();
        let (source,mut decisions)=self.source.reconstruct(values)?;
        let template=self.template.as_ref().ok_or("CPU scalar family is not constructed")?;
        let assignment=self.parameters.iter().map(|id|values.get(id.0).and_then(|&n|usize::try_from(n).ok()).ok_or("CPU scalar parameter assignment missing")).collect::<Result<Vec<_>,_>>()?;
        let mut program=template.instantiate(&assignment)?;
        program.conditions=seismic_realization::InvocationConditions::from_lowered(&source)?;
        let objective=self.account.as_ref().ok_or("CPU accounting remains unresolved")?.reconstruct(values,lower_bound)?;
        let (model,schedule)=objective.flat()?;
        let order=scheduling::Order {blocks:schedule::static_order::orders(model,schedule)?};
        let mut execution=program.clone();
        scheduling::apply(&mut execution,&order)?;
        scheduling::check_materialization(&program,&execution,&order)?;
        for ((site,&ordinal),guard) in template.sites().iter().zip(&assignment).zip(template.load_guards()) {
            if !guard.iter().all(|(parameter,value)|assignment[*parameter]==usize::from(*value)) {continue;}
            decisions.push(Decision {identity:format!("cpu.load.{}",site.variable),value:ordinal as i64});
        }
        Ok(Reconstructed {source,execution,objective,decisions})
    }
}

pub(super) fn build(input:Input<'_>,hardware:&ScalarHardware,workload:&ScalarWorkload,limits:DerivationLimits)->Result<Export<ScalarProgram>,String> {
    let family=Binding::construct(input,"cpu")?;
    let mut builder=ModelBuilder::new();
    builder.units(format!("{} / {} seconds",hardware.timebase.seconds_numerator,hardware.timebase.seconds_denominator));
    let source=Binding::append(&mut builder,family.clone())?;
    let mut plan=Plan {source,template:None,parameters:Vec::new(),account:None};
    let template=match ScalarFamily::from_family(&family,crate::host_call_conv()?,seismic_realization::Dispatch::Sequential,seismic_realization::dispatch::Participation::Thread) {
        Ok(template)=>template,
        Err(seismic_compiler::template::Error::Unsupported(reason))=>{
            builder.obligation(vec![],ObligationKind::Construction,reason);
            return Export::new(Arc::new(builder.build().map_err(|e|e.to_string())?),plan);
        },
        Err(error)=>return Err(error.to_string()),
    };
        let mut source_parameters=Vec::new();
        for (decision,ordinal) in template.source_parameters() {
            let mut guard=family.decisions().iter().find(|entry|&entry.id==decision).ok_or("unknown retained source selector")?.guard.clone();
            guard.choices.push((decision.clone(),*ordinal));
            let presence=plan.source.presence(&mut builder,&guard)?.ok_or("source arm selector must have a presence condition")?;
            source_parameters.push(presence);
        }
        for ((site,domain),guard) in template.sites().iter().zip(template.domains()).zip(template.load_guards()) {
            let maximum=i64::try_from(domain.len().checked_sub(1).ok_or("empty CPU load domain")?).map_err(|_|"CPU load domain exceeds integer range")?;
            let local_guard=guard.iter().map(|(&parameter,&value)|Ok((parameter.checked_sub(template.domains().len()).ok_or("load activation depends on another ownership decision")?,value))).collect::<Result<BTreeMap<_,_>,String>>()?;
            let active=reify_guard(&mut builder,&source_parameters,&local_guard)?;
            let append=|builder:&mut ModelBuilder|builder.local_variable(format!("cpu.load.{}",site.variable),Domain::interval(0,maximum).map_err(|error|error.to_string())?).map_err(|error|error.to_string());
            let parameter=match active {Some(active)=>builder.when(Literal::new(active,1),append),None=>append(&mut builder)}?;
            plan.parameters.push(parameter);
        }
        plan.parameters.extend(source_parameters);
        match execution_model::family::derive(template.program(),hardware,workload,limits,
            template.literals().iter().map(|literal|(literal.instruction,literal.parameter)),template.joins().iter().copied()) {
            Ok(derived)=>{
                for (guard,reason) in &derived.obligations {
                    let active=reify_guard(&mut builder,&plan.parameters,guard)?;
                    builder.obligation(active.into_iter().map(|active|Literal::new(active,1)).collect(),ObligationKind::Analysis,reason.clone());
                }
                let original=Arc::new(derived.execution.model);
                let horizon=schedule::export::horizon(&original.as_ref().clone().into())?;
                let mut encoding=Encoding::new(&mut builder,"cpu",&original.resources,horizon).map_err(|e|e.to_string())?;
                let mut guards=BTreeMap::new();
                let mut presence=Vec::new();
                for guard in &derived.guards {
                    if guard.is_empty() {presence.push(None);continue;}
                    let active=if let Some(&active)=guards.get(guard) {active} else {
                        let active=reify_guard(&mut builder,&plan.parameters,guard)?.ok_or("nonempty operation guard has no activation")?;
                        guards.insert(guard.clone(),active);active
                    };
                    presence.push(Some(active));
                }
                plan.account=Some(Account::append(&mut builder,&mut encoding,original,presence)?);
                let completion=encoding.finish(&mut builder).map_err(|e|e.to_string())?;
                builder.cost(Cost::Linear {constant:0,terms:vec![LinearTerm::new(completion,1)]});
            },
            Err(DerivationError::Analysis(reason))=>return Err(reason),
            Err(error)=>{builder.obligation(vec![],ObligationKind::Analysis,error.to_string());},
        }
        plan.template=Some(template);
    Export::new(Arc::new(builder.build().map_err(|e|e.to_string())?),plan)
}

fn reify_guard(builder:&mut ModelBuilder,parameters:&[VarId],guard:&BTreeMap<usize,bool>)->Result<Option<VarId>,String> {
    if guard.is_empty() {return Ok(None);}
    let mut terms=Vec::new();
    for (&index,&selected) in guard {
        let parameter=*parameters.get(index).ok_or("scalar guard names an absent parameter")?;
        if selected {terms.push(parameter);} else {
            let inverse=builder.variable("cpu.parameter.inverse",Domain::boolean());
            builder.constraint(Constraint::NotEqual {left:inverse,right:parameter});terms.push(inverse);
        }
    }
    let active=builder.variable("cpu.region.active",Domain::boolean());
    builder.constraint(Constraint::BoolAnd {output:active,inputs:terms});
    Ok(Some(active))
}
