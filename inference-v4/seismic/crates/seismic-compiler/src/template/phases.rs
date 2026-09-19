//! Retained scalar families across ordered publication boundaries. Original
//! source/load parameters keep one identity across every phase. Publication
//! restores are the phase plan's explicit materializations, not new choices.
use super::*;
use seismic_lang::family::ExecutionFamily;
use std::collections::{BTreeMap,HashMap};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum PhaseParameter { Original(usize), Fixed(usize) }
pub struct PhaseFamily {
    pub source_statement: usize,
    pub scalar: ScalarFamily,
    pub parameters: Vec<PhaseParameter>,
}
pub struct SequenceFamily {
    source: LoweredIr,
    loads: loads::Family,
    source_parameters: Vec<(seismic_lang::family::DecisionId,usize)>,
    load_guards: Vec<BTreeMap<usize,bool>>,
    phases: Vec<PhaseFamily>,
    retained: Vec<seismic_realization::phases::RetainedValue>,
}
impl SequenceFamily {
    /// Typed union source before launch lowering, for target domain equations.
    pub fn source_from_family(family:&ExecutionFamily)->Result<LoweredIr,Error> {
        if !family.obligations().is_empty() {return Err(Error::Unsupported("source transformations still carry construction obligations".into()));}
        let mut retained=SourceRegions {family,function:family.template().clone(),parameters:Vec::new(),variables:Vec::new()};
        retained.function.body=retained.region(family.root())?;
        Ok(seismic_lang::reduction::structured::materialize(&retained.function)?)
    }
    pub fn new(source:&LoweredIr,convention:CallConv,participation:Participation)->Result<Self,Error> {
        let source=seismic_lang::reduction::structured::materialize(source)?;
        Self::construct(source,Vec::new(),Vec::new(),convention,participation)
    }
    pub fn from_family(family:&ExecutionFamily,convention:CallConv,participation:Participation)->Result<Self,Error> {
        if !family.obligations().is_empty() {return Err(Error::Unsupported("source transformations still carry construction obligations".into()));}
        let mut retained=SourceRegions {family,function:family.template().clone(),parameters:Vec::new(),variables:Vec::new()};
        retained.function.body=retained.region(family.root())?;
        let source=seismic_lang::reduction::structured::materialize(&retained.function)?;
        Self::construct(source,retained.parameters,retained.variables,convention,participation)
    }
    fn construct(source:LoweredIr,source_parameters:Vec<(seismic_lang::family::DecisionId,usize)>,selectors:Vec<VarId>,convention:CallConv,participation:Participation)->Result<Self,Error> {
        let loads=loads::Family::new(source)?;
        if !selectors.is_empty() && loads.sites().iter().any(|site|!site.can_borrow && site.selected.is_none() && choice_affects_snapshot(&loads.function().body,site.variable,&selectors)) {
            return Err(Error::Unsupported("source-dependent phase ownership needs a guard-conditioned snapshot lifetime proof".into()));
        }
        let original_load_count=loads.domains().len();
        let source_choices:HashMap<_,_>=selectors.iter().enumerate().map(|(index,&variable)|(variable,original_load_count+index)).collect();
        let load_guards=load_activation(&loads.function().body,&source_choices);
        if load_guards.len()!=original_load_count {return Err("publication load guards do not cover original occurrences".into());}
        let source=loads.function().clone();
        let public_buffer_count=seismic_realization::storage::parameters(&source)?.0.len();
        let conditions=seismic_realization::InvocationConditions::from_lowered(&source)?;
        let plan=match seismic_realization::phases::assess(&source)? {
            seismic_realization::phases::Applicability::Supported(plan)=>plan,
            seismic_realization::phases::Applicability::Unresolved {reason}=>return Err(Error::Unsupported(reason)),
        };
        let mut phases=Vec::new();
        let mut represented=vec![false;original_load_count];
        for (source_statement,statement) in plan.function.body.iter().enumerate() {
            let mut phase=plan.function.clone();phase.body=vec![statement.clone()];
            let local_loads=loads::Family::new(phase)?;
            let local_count=local_loads.domains().len();
            let local_source_choices:HashMap<_,_>=selectors.iter().enumerate().map(|(index,&variable)|(variable,local_count+index)).collect();
            let local_guards=load_activation(&local_loads.function().body,&local_source_choices);
            if local_guards.len()!=local_count {return Err("phase load guard arity differs from retained occurrences".into());}
            let mut parameters=Vec::new();
            for (local,((site,domain),guard)) in local_loads.sites().iter().zip(local_loads.domains()).zip(&local_guards).enumerate() {
                let global_guard=guard.iter().map(|(&parameter,&value)| {
                    let source=parameter.checked_sub(local_count).ok_or("phase load activation names a non-source selector")?;
                    Ok((original_load_count+source,value))
                }).collect::<Result<BTreeMap<_,_>,String>>()?;
                let matches=loads.sites().iter().zip(loads.domains()).enumerate().filter(|(index,(original,original_domain))| {
                    !represented[*index] && original.variable==site.variable && *original_domain==domain && load_guards[*index]==global_guard
                }).map(|(index,_)|index).collect::<Vec<_>>();
                if let Some(&index)=matches.first() {
                    represented[index]=true;parameters.push(PhaseParameter::Original(index));
                } else if site.selected.is_some() && domain.len()==1 {
                    parameters.push(PhaseParameter::Fixed(0));
                } else {
                    return Err(Error::Unsupported(format!("publication phase {source_statement} load {local} lacks an original ownership occurrence")));
                }
            }
            parameters.extend((0..source_parameters.len()).map(|index|PhaseParameter::Original(original_load_count+index)));
            let choices=local_loads.sites().iter().zip(local_loads.domains()).enumerate().filter(|(_,(_,domain))|domain.len()>1).map(|(site,(definition,_))|(definition.variable,site)).collect();
            let function=local_loads.instantiate(&vec![0;local_count])?;
            let (mut program,literals,joins)=crate::scalar_load_template(&function,convention,Dispatch::ParallelRoot,participation,choices,local_source_choices)?;
            program.public_buffer_count=public_buffer_count;
            program.conditions=conditions.clone();
            let scalar=ScalarFamily {loads:local_loads,program,literals,joins,source_parameters:source_parameters.clone(),load_guards:local_guards};
            phases.push(PhaseFamily {source_statement,scalar,parameters});
        }
        if represented.iter().any(|represented|!*represented) {return Err(Error::Unsupported("publication phases did not retain every original load ownership occurrence".into()));}
        Ok(Self {source,loads,source_parameters,load_guards,phases,retained:plan.retained})
    }
    pub fn source(&self)->&LoweredIr {&self.source}
    pub fn sites(&self)->&[loads::Site] {self.loads.sites()}
    pub fn domains(&self)->&[Vec<LoadMode>] {self.loads.domains()}
    pub fn source_parameters(&self)->&[(seismic_lang::family::DecisionId,usize)] {&self.source_parameters}
    pub fn load_guards(&self)->&[BTreeMap<usize,bool>] {&self.load_guards}
    pub fn phases(&self)->&[PhaseFamily] {&self.phases}
    pub fn retained(&self)->&[seismic_realization::phases::RetainedValue] {&self.retained}
    pub fn into_phases(self)->Vec<PhaseFamily> {self.phases}
}
