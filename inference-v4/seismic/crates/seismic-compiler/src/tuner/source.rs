//! Original source decisions, topology guards and reconstruction bindings for
//! the enclosing backend model. There is no traversal of selected lowerings.
use super::{choices::Choice, expressions::Expressions, family::Decision, Input};
use magnitude_solver::model::{Constraint, Domain, Literal, ModelBuilder, ObligationKind, VarId};
use seismic_accounting::algebra::Value;
use seismic_lang::{family::{Assignment, DecisionId, ExecutionFamily, Guard}, lowered_ir::LoweredIr, sym::Atom};
use std::{collections::BTreeMap, sync::Arc};

pub struct Binding {
    family: Arc<ExecutionFamily>,
    choices: BTreeMap<DecisionId, Choice>,
    activation: BTreeMap<DecisionId, Option<VarId>>,
    parameters: BTreeMap<String, Value>,
    guards: BTreeMap<Vec<(usize, i64)>, VarId>,
    equalities: BTreeMap<(VarId, i64), VarId>,
    constants: BTreeMap<i64, VarId>,
    complements: BTreeMap<VarId, VarId>,
    disjunctions:BTreeMap<Vec<VarId>,VarId>,
    predicates: BTreeMap<(seismic_lang::sym::Sym, Vec<(String,VarId)>, Vec<(usize,i64)>),VarId>,
}
impl Binding {
    pub fn construct(input: Input<'_>, backend: &str) -> Result<Arc<ExecutionFamily>, String> {
        Ok(Arc::new(match input {
            Input::Portable { program, entry, shapes, elements, options } => ExecutionFamily::new(
                seismic_lang::lower::alternatives::Specialization { program, entry, backend, shapes, elements, options })?,
            Input::Lowered(function) => {
                if function.backend != backend { return Err("source/backend family mismatch".into()); }
                ExecutionFamily::from_lowered(function.clone())?
            }
        }))
    }
    pub fn append(builder: &mut ModelBuilder, family: Arc<ExecutionFamily>) -> Result<Self, String> {
        let mut binding = Self { family, choices: BTreeMap::new(), activation: BTreeMap::new(),
            parameters: BTreeMap::new(), guards: BTreeMap::new(), equalities:BTreeMap::new(),
            constants:BTreeMap::new(), complements:BTreeMap::new(), predicates:BTreeMap::new(),disjunctions:BTreeMap::new() };
        for decision in binding.family.decisions().to_vec() {
            if binding.choices.contains_key(&decision.id) { return Err("duplicate source decision identity".into()); }
            let presence = binding.presence(builder, &decision.guard)?;
            let choice = Choice::append(builder, &format!("source.{:?}", decision.id), &decision.domain, presence)
                .map_err(|e| e.to_string())?;
            if let Some(parameter) = &decision.numeric {
                let Atom::Param(name) = &parameter.atom else { return Err("family parameter is not a parameter atom".into()); };
                let value = choice.numeric.ok_or("numeric source parameter has no numeric domain")?;
                if binding.parameters.insert(name.clone(), value).is_some() { return Err("duplicate source numeric parameter".into()); }
            }
            binding.activation.insert(decision.id.clone(), presence);
            binding.choices.insert(decision.id, choice);
        }
        for obligation in binding.family.obligations().to_vec() {
            let literals=binding.literals(builder,&obligation.guard)?;
            builder.obligation(literals, ObligationKind::Construction,
                format!("{:?}: {}", obligation.occurrence, obligation.reason));
        }
        for requirement in binding.family.requirements().to_vec() {
            let presence = binding.presence(builder, &requirement.guard)?;
            // Partial arithmetic appends are transactional: an unsupported
            // equation cannot leave restrictive remnants in the legal family.
            let mut next = builder.clone();
            let mut expressions = Expressions::new(binding.parameters.clone());
            let mut append = |builder: &mut ModelBuilder| expressions.nonnegative(builder, "source.requirement", &requirement.nonnegative);
            let result = match presence {
                Some(active) => next.when(Literal::new(active, 1), append),
                None => append(&mut next),
            };
            match result {
                Ok(_) => *builder = next,
                Err(seismic_accounting::algebra::Error::Unsupported(reason)) => {
                    let literals=binding.literals(builder,&requirement.guard)?;
                    builder.obligation(literals, ObligationKind::Analysis, reason);
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        Ok(binding)
    }
    pub fn family(&self) -> &Arc<ExecutionFamily> { &self.family }
    pub fn parameters(&self) -> &BTreeMap<String, Value> { &self.parameters }
    pub fn choice(&self, id: &DecisionId) -> Option<&Choice> { self.choices.get(id) }
    pub fn literals(&mut self, builder:&mut ModelBuilder, guard: &Guard) -> Result<Vec<Literal>, String> {
        let mut literals = Vec::new();
        for (id, ordinal) in &guard.choices {
            let choice = self.choices.get(id).ok_or("source guard references a decision before its definition")?;
            if choice.source().alternatives.get(*ordinal).is_none() { return Err("source guard ordinal is outside its original domain".into()); }
            if let Some(active) = self.activation.get(id).copied().flatten() { literals.push(Literal::new(active, 1)); }
            literals.push(Literal::new(choice.ordinal, i64::try_from(*ordinal).map_err(|_| "source ordinal exceeds i64")?));
        }
        for clause in &guard.one_of {
            let mut terms=Vec::new();
            for (id,ordinal) in clause {
                let choice=self.choices.get(id).ok_or("guard disjunction names an absent decision")?;
                if choice.source().alternatives.get(*ordinal).is_none() {return Err("guard disjunction ordinal is outside its original domain".into());}
                let mut term=vec![Literal::new(choice.ordinal,i64::try_from(*ordinal).map_err(|_|"guard ordinal exceeds i64")?)];
                if let Some(active)=self.activation.get(id).copied().flatten() {term.push(Literal::new(active,1));}
                terms.push(self.conjunction(builder,term)?.ok_or("guard disjunction has an empty term")?);
            }
            terms.sort();terms.dedup();
            let none=if let Some(&none)=self.disjunctions.get(&terms) {none} else {
                let mut absent=Vec::new();
                for &term in &terms {
                    let inverse=if let Some(&inverse)=self.complements.get(&term) {inverse} else {
                        let inverse=builder.variable("source.guard.absent",Domain::boolean());
                        builder.constraint(Constraint::NotEqual {left:term,right:inverse});
                        self.complements.insert(term,inverse);inverse
                    };
                    absent.push(inverse);
                }
                let none=if absent.len()==1 {absent[0]} else {
                    let none=builder.variable("source.guard.none",Domain::boolean());
                    builder.constraint(Constraint::BoolAnd {output:none,inputs:absent});none
                };
                self.disjunctions.insert(terms,none);none
            };
            literals.push(Literal::new(none,0));
        }
        for predicate in &guard.predicates {
            let mut parameters=self.parameters.clone();
            let mut parent=literals.clone();
            for (name,id,domain) in &predicate.parameters {
                let choice=self.choices.get(id).ok_or("numeric guard references an absent original decision")?;
                if &choice.source().alternatives != domain { return Err("numeric guard domain differs from its original decision".into()); }
                parameters.insert(name.clone(),choice.numeric.ok_or("numeric guard names a categorical decision")?);
                if let Some(active)=self.activation.get(id).copied().flatten() {parent.push(Literal::new(active,1));}
            }
            parent.sort_by_key(|literal|(literal.variable.0,literal.value));parent.dedup();
            let key=(predicate.nonnegative.clone(),parameters.iter().map(|(name,value)|(name.clone(),value.id())).collect(),parent.iter().map(|literal|(literal.variable.0,literal.value)).collect());
            if let Some(&active)=self.predicates.get(&key) {literals.extend(parent);literals.push(Literal::new(active,1));continue;}
            let mut next=builder.clone();
            let mut expressions=Expressions::new(parameters);
            let result=under(&mut next,&parent,|builder|expressions.predicate(builder,"source.guard",&predicate.nonnegative));
            let active=match result {
                Ok(active)=>{*builder=next;active},
                Err(seismic_accounting::algebra::Error::Unsupported(reason))=>{
                    builder.obligation(parent.clone(),ObligationKind::Analysis,reason);
                    under(builder,&parent,|builder|builder.local_variable("source.guard.unresolved",Domain::boolean()).map_err(|error|error.to_string()))?
                },
                Err(error)=>return Err(error.to_string()),
            };
            self.predicates.insert(key,active);
            literals.extend(parent);
            literals.push(Literal::new(active,1));
        }
        literals.sort_by_key(|literal| (literal.variable.0, literal.value));
        literals.dedup();
        Ok(literals)
    }
    /// Reify the complete activation condition. Including ancestor presence
    /// prevents a canonical inactive ordinal from activating a descendant.
    pub fn presence(&mut self, builder: &mut ModelBuilder, guard: &Guard) -> Result<Option<VarId>, String> {
        let literals = self.literals(builder,guard)?;
        self.conjunction(builder,literals)
    }
    fn conjunction(&mut self,builder:&mut ModelBuilder,mut literals:Vec<Literal>)->Result<Option<VarId>,String> {
        literals.sort_by_key(|literal|(literal.variable.0,literal.value));literals.dedup();
        if literals.is_empty() { return Ok(None); }
        let key: Vec<_> = literals.iter().map(|l| (l.variable.0, l.value)).collect();
        if let Some(&presence) = self.guards.get(&key) { return Ok(Some(presence)); }
        let mut terms = Vec::new();
        for literal in literals {
            let equal = if let Some(&equal) = self.equalities.get(&(literal.variable, literal.value)) { equal } else {
                let equal = builder.variable("source.guard.equal", Domain::boolean());
                let constant = *self.constants.entry(literal.value).or_insert_with(||
                    builder.variable("source.guard.value", Domain::singleton(literal.value)));
                builder.guarded_constraint(vec![Literal::new(equal, 1)], Constraint::Equal { left: literal.variable, right: constant });
                builder.guarded_constraint(vec![Literal::new(equal, 0)], Constraint::NotEqual { left: literal.variable, right: constant });
                self.equalities.insert((literal.variable, literal.value), equal);
                equal
            };
            terms.push(equal);
        }
        let presence = if terms.len() == 1 { terms[0] } else {
            let presence = builder.variable("source.guard.active", Domain::boolean());
            builder.constraint(Constraint::BoolAnd { output: presence, inputs: terms });
            presence
        };
        self.guards.insert(key, presence);
        Ok(Some(presence))
    }
    pub fn assignment(&self, values: &[i64]) -> Result<Assignment, String> {
        let mut assignment = Assignment::new();
        for (id, choice) in &self.choices {
            if let Some((ordinal, _)) = choice.reconstruct(values).map_err(|e| e.to_string())? {
                assignment.insert(id.clone(), ordinal);
            }
        }
        Ok(assignment)
    }
    pub fn reconstruct(&self, values: &[i64]) -> Result<(LoweredIr, Vec<Decision>), String> {
        let assignment = self.assignment(values)?;
        let source = self.family.instantiate(&assignment)?;
        let decisions = assignment.into_iter().map(|(id, ordinal)| Ok(Decision {
            identity: format!("source.{id:?}"), value: i64::try_from(ordinal).map_err(|_| "source ordinal exceeds i64")?,
        })).collect::<Result<_, String>>()?;
        Ok((source, decisions))
    }
}

fn under<T>(builder:&mut ModelBuilder, guards:&[Literal], append:impl FnOnce(&mut ModelBuilder)->T)->T {
    match guards.split_first() {
        Some((&first,rest))=>builder.when(first,|builder|under(builder,rest,append)),
        None=>append(builder),
    }
}
