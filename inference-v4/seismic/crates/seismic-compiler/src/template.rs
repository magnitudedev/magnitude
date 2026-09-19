//! One typed scalar CFG containing original local ownership alternatives.
//! Parameters belong to static source sites and remain shared by every dynamic
//! visit. Reconstruction substitutes literal operands; it never repeats lowering.
mod phases;
pub use phases::{SequenceFamily, PhaseFamily, PhaseParameter};
use cranelift_codegen::ir::{Inst, InstructionData, Opcode};
use seismic_lang::{ir::{LoadMode, VarId}, lowered_ir::LoweredIr, normalize::loads};
use seismic_realization::{dispatch::Participation, CallConv, Dispatch, ScalarProgram};

#[derive(Clone, Debug)]
pub struct ParameterLiteral {
    pub parameter: usize,
    pub variable: VarId,
    pub instruction: Inst,
}

pub struct ScalarFamily {
    loads: loads::Family,
    program: ScalarProgram,
    literals: Vec<ParameterLiteral>,
    joins: Vec<(Inst, cranelift_codegen::ir::Block)>,
    source_parameters: Vec<(seismic_lang::family::DecisionId, usize)>,
    load_guards: Vec<std::collections::BTreeMap<usize,bool>>,
}
impl ScalarFamily {
    pub fn new(source: &LoweredIr, convention: CallConv, dispatch: Dispatch, participation: Participation) -> Result<Self, String> {
        let source = seismic_lang::reduction::structured::materialize(source)?;
        let loads = loads::Family::new(source)?;
        // The operand is only a placeholder in the union CFG. Both ownership
        // bodies are emitted, and no complete assignment is selected here.
        let placeholders = vec![0; loads.domains().len()];
        let template = loads.instantiate(&placeholders)?;
        let choices = loads.sites().iter().zip(loads.domains()).enumerate()
            .filter(|(_, (_, domain))| domain.len() > 1)
            .map(|(site, (definition, _))| (definition.variable, site)).collect();
        let (program, literals, joins) = crate::scalar_load_template(&template, convention, dispatch, participation, choices, Default::default())?;
        let load_guards=vec![Default::default();loads.sites().len()];
        Ok(Self { loads, program, literals, joins, source_parameters:Vec::new(), load_guards })
    }
    pub fn load_guards(&self)->&[std::collections::BTreeMap<usize,bool>] { &self.load_guards }
    pub fn source_parameters(&self) -> &[(seismic_lang::family::DecisionId,usize)] { &self.source_parameters }
    pub fn source(&self) -> &LoweredIr { self.loads.function() }
    pub fn domains(&self) -> &[Vec<LoadMode>] { self.loads.domains() }
    pub fn sites(&self) -> &[loads::Site] { self.loads.sites() }
    pub fn program(&self) -> &ScalarProgram { &self.program }
    pub fn literals(&self) -> &[ParameterLiteral] { &self.literals }
    pub fn joins(&self) -> &[(Inst, cranelift_codegen::ir::Block)] { &self.joins }
    pub fn instantiate(&self, assignment: &[usize]) -> Result<ScalarProgram, String> {
        if assignment.len()!=self.loads.domains().len()+self.source_parameters.len() { return Err("scalar template parameter assignment arity mismatch".into()); }
        for (index,guard) in self.load_guards.iter().enumerate() {
            if !guard.iter().all(|(parameter,value)|assignment.get(*parameter).copied()==Some(usize::from(*value))) && assignment[index]!=0 {
                return Err("inactive scalar load has a noncanonical ownership choice".into());
            }
        }
        let selected = self.loads.instantiate(&assignment[..self.loads.domains().len()])?;
        let mut program = self.program.clone();
        for literal in &self.literals {
            let selected = *assignment.get(literal.parameter).ok_or("missing retained scalar parameter")?;
            let value = if let Some(domain)=self.loads.domains().get(literal.parameter) {
                i64::from(*domain.get(selected).ok_or("load choice lies outside its original domain")?==LoadMode::Borrow)
            } else {
                if selected>1 { return Err("source predicate is not boolean".into()); }
                selected as i64
            };
            let node = &mut program.function.dfg.insts[literal.instruction];
            if !matches!(node, InstructionData::UnaryImm { opcode: Opcode::Iconst, .. }) {
                return Err("load choice operand changed after family construction".into());
            }
            *node = InstructionData::UnaryImm { opcode: Opcode::Iconst,
                imm: value.into() };
        }
        program.loads = loads::selected(&selected.body)?.into_iter().filter(|decision|self.load_guards[decision.site].iter().all(|(parameter,value)|assignment.get(*parameter).copied()==Some(usize::from(*value)))).collect();
        Ok(program)
    }
}

/// Unsupported retained structure is coverage, while malformed typed IR is a
/// construction error. Callers must not turn either into a rejected candidate.
#[derive(Debug)]
pub enum Error { Unsupported(String), Invalid(String) }
impl From<String> for Error { fn from(value:String)->Self {Self::Invalid(value)} }
impl From<&str> for Error { fn from(value:&str)->Self {Self::Invalid(value.into())} }
impl std::fmt::Display for Error {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {match self {Self::Unsupported(s)|Self::Invalid(s)=>f.write_str(s)}}
}
impl std::error::Error for Error {}

impl ScalarFamily {
    /// Retain categorical source topology in the same scalar graph. The graph
    /// size is additive in local arms, never the product of source decisions.
    pub fn from_family(family:&seismic_lang::family::ExecutionFamily,convention:CallConv,dispatch:Dispatch,participation:Participation)->Result<Self,Error> {
        if !family.obligations().is_empty() {return Err(Error::Unsupported("source transformations still carry construction obligations".into()));}
        let mut retained=SourceRegions {family,function:family.template().clone(),parameters:Vec::new(),variables:Vec::new()};
        retained.function.body=retained.region(family.root())?;
        let normalized=seismic_lang::reduction::structured::materialize(&retained.function)?;
        let loads=loads::Family::new(normalized)?;
        // Effect analysis must preserve guarded legality before borrowing is
        // admitted. A topology-sensitive proof cannot be replaced with the
        // conservative result of analyzing the union as one runtime program.
        if !retained.parameters.is_empty() && loads.sites().iter().any(|site|!site.can_borrow && site.selected.is_none() && choice_affects_snapshot(&loads.function().body,site.variable,&retained.variables)) {
            return Err(Error::Unsupported("source-dependent load ownership needs a guard-conditioned snapshot lifetime proof".into()));
        }
        let function=loads.instantiate(&vec![0;loads.domains().len()])?;
        let choices=loads.sites().iter().zip(loads.domains()).enumerate().filter(|(_,(_,domain))|domain.len()>1).map(|(site,(definition,_))|(definition.variable,site)).collect();
        let source_choices:std::collections::HashMap<_,_>=retained.variables.iter().enumerate().map(|(index,&variable)|(variable,loads.domains().len()+index)).collect();
        let load_guards=load_activation(&function.body,&source_choices);
        if load_guards.len()!=loads.sites().len() {return Err("retained load guards do not cover the original sites".into());}
        let (program,literals,joins)=crate::scalar_load_template(&function,convention,dispatch,participation,choices,source_choices)?;
        Ok(Self {loads,program,literals,joins,source_parameters:retained.parameters,load_guards})
    }
}
struct SourceRegions<'a> {
    family:&'a seismic_lang::family::ExecutionFamily,
    function:LoweredIr,
    parameters:Vec<(seismic_lang::family::DecisionId,usize)>,
    variables:Vec<VarId>,
}
impl SourceRegions<'_> {
    fn region(&mut self,id:seismic_lang::family::RegionId)->Result<Vec<seismic_lang::ir::Stmt>,Error> {
        use seismic_lang::{ast::AssignOp,family::RegionKind,ir::{Expr,ExprKind,Stmt,StmtKind,Var,VarKind},types::{DType,Ty}};
        let region=self.family.regions().get(id.0).ok_or("invalid retained source region")?;
        match &region.kind {
            RegionKind::Sequence(children)=>{
                let mut body=Vec::new();for &child in children {body.extend(self.region(child)?);}Ok(body)
            },
            RegionKind::Statement(statement)=>Ok(vec![statement.clone()]),
            RegionKind::Choice {decision,arms}=>{
                if arms.is_empty() {return Err("empty retained source topology".into());}
                if arms.len()==1 {return self.region(arms[0]);}
                let mut body=self.region(*arms.last().unwrap())?;
                for (ordinal,&arm) in arms.iter().enumerate().rev().skip(1) {
                    let then=self.region(arm)?;
                    let span=then.first().or(body.first()).map(|statement|statement.span).unwrap_or_default();
                    let variable=self.function.vars.len();
                    self.function.vars.push(Var {name:format!("source_choice_{variable}"),ty:Ty::Scalar(DType::Bool),kind:VarKind::Local,span});
                    self.parameters.push((decision.clone(),ordinal));self.variables.push(variable);
                    let value=Expr {kind:ExprKind::Var(variable),ty:Ty::Scalar(DType::Bool),sym:None,span};
                    body=vec![Stmt {id:None,span,kind:StmtKind::Assign {target:value.clone(),op:AssignOp::Assign,value:Expr {kind:ExprKind::Bool(false),ty:Ty::Scalar(DType::Bool),sym:None,span}}},
                        Stmt {id:None,span,kind:StmtKind::If {cond:value,then,els:body}}];
                }
                Ok(body)
            },
            RegionKind::Repeated {header,body,..}=>{
                let mut statement=header.clone();let selected=self.region(*body)?;
                match &mut statement.kind {StmtKind::Parallel {body,..}|StmtKind::Owned {body,..}|StmtKind::Range {body,..}|StmtKind::Lanes {body,..}=>*body=selected,
                    _=>return Err("invalid retained repetition header".into())}
                Ok(vec![statement])
            },
            RegionKind::Replicated {index,count,body}=>{
                let count=count.as_constant().ok_or_else(||Error::Unsupported("scalar static replication needs original symbolic occurrence guards".into()))?;
                let count=usize::try_from(count).map_err(|_|Error::Invalid("negative retained replication count".into()))?;
                let variable=self.function.vars.get(*index).ok_or("retained replication index is absent")?.clone();
                let VarKind::Index(atom)=variable.kind else {return Err("retained replication index has no symbolic identity".into())};
                let original=self.region(*body)?;
                let mut output=Vec::new();
                for ordinal in 0..count {
                    output.extend(seismic_lang::widen::parameterized::replicate(&original,*index,&atom,ordinal as i64,variable.span));
                }
                Ok(output)
            },
            RegionKind::Conditional {header,then,els}=>{
                let mut statement=header.clone();let yes=self.region(*then)?;let no=self.region(*els)?;
                let StmtKind::If {then,els,..}=&mut statement.kind else {return Err("invalid retained conditional header".into())};
                *then=yes;*els=no;Ok(vec![statement])
            },
            RegionKind::Stream {..}=>Err(Error::Unsupported("scalar stream templates need original symbolic capacity operands and storage strides".into())),
            RegionKind::Reduction(_)=>Err(Error::Unsupported("scalar reduction templates need guarded tree, segment and preparation regions".into())),
        }
    }
}

fn choice_affects_snapshot(body:&[seismic_lang::ir::Stmt],variable:VarId,selectors:&[VarId])->bool {
    use seismic_lang::ir::{ExprKind,StmtKind};
    fn contains(body:&[seismic_lang::ir::Stmt],selectors:&[VarId])->bool {
        body.iter().any(|statement|match &statement.kind {
            StmtKind::If {cond,then,els}=>matches!(&cond.kind,ExprKind::Var(id) if selectors.contains(id)) || contains(then,selectors) || contains(els,selectors),
            StmtKind::Parallel {body,..}|StmtKind::Owned {body,..}|StmtKind::Range {body,..}|StmtKind::LoadLoop {body,..}|StmtKind::Lanes {body,..}=>contains(body,selectors),
            _=>false,
        })
    }
    if let Some(definition)=body.iter().position(|statement|matches!(&statement.kind,StmtKind::Assign {target,..} if matches!(target.kind,ExprKind::Var(id) if id==variable))) {
        let last=body.iter().rposition(|statement|seismic_lang::effects::uses(statement,variable)).unwrap_or(definition);
        return last>definition && contains(&body[definition+1..=last],selectors);
    }
    body.iter().any(|statement|match &statement.kind {
        StmtKind::If {then,els,..}=>choice_affects_snapshot(then,variable,selectors)||choice_affects_snapshot(els,variable,selectors),
        StmtKind::LoadLoop {vars,body,..} if vars.contains(&variable)=>contains(body,selectors),
        StmtKind::Parallel {body,..}|StmtKind::Owned {body,..}|StmtKind::Range {body,..}|StmtKind::LoadLoop {body,..}|StmtKind::Lanes {body,..}=>choice_affects_snapshot(body,variable,selectors),
        _=>false,
    })
}

/// Lexical ownership decisions exist under the same source predicates as their
/// defining load. Runtime branches do not become optimizer choices.
fn load_activation(body:&[seismic_lang::ir::Stmt],selectors:&std::collections::HashMap<VarId,usize>)->Vec<std::collections::BTreeMap<usize,bool>> {
    use seismic_lang::ir::{Builtin,ExprKind,StmtKind};
    type Guard=std::collections::BTreeMap<usize,bool>;
    fn visit(body:&[seismic_lang::ir::Stmt],selectors:&std::collections::HashMap<VarId,usize>,guard:&Guard,out:&mut Vec<Guard>) {
        for statement in body {
            match &statement.kind {
                StmtKind::Assign {target,value,..} if matches!(target.kind,ExprKind::Var(_)) && matches!(value.kind,ExprKind::Builtin {name:Builtin::Load,..}|ExprKind::Load {..})=>out.push(guard.clone()),
                StmtKind::LoadLoop {vars,body,..}=>{out.extend(vars.iter().map(|_|guard.clone()));visit(body,selectors,guard,out);},
                StmtKind::If {cond,then,els}=>{
                    if let ExprKind::Var(variable)=cond.kind {
                        if let Some(&parameter)=selectors.get(&variable) {
                            let mut yes=guard.clone();yes.insert(parameter,true);
                            let mut no=guard.clone();no.insert(parameter,false);
                            visit(then,selectors,&yes,out);visit(els,selectors,&no,out);continue;
                        }
                    }
                    visit(then,selectors,guard,out);visit(els,selectors,guard,out);
                },
                StmtKind::Parallel {body,..}|StmtKind::Owned {body,..}|StmtKind::Range {body,..}|StmtKind::Lanes {body,..}=>visit(body,selectors,guard,out),
                StmtKind::Reduction(reduction)=>for implementation in reduction.implementations() {visit(&implementation.body,selectors,guard,out);},
                _=>{},
            }
        }
    }
    let mut out=Vec::new();visit(body,selectors,&Guard::new(),&mut out);out
}
