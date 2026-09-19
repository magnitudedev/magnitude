//! A terminal PTX family derived once from the common scalar family. Selection
//! changes retained literal operands; it does not repeat source or target lowering.
use super::*;
use seismic_compiler::template::ScalarFamily;
use std::collections::BTreeMap;

pub struct TargetFamily {
    scalar: ScalarFamily,
    plan: TargetPlan,
    literals: BTreeMap<usize, usize>,
    joins: BTreeMap<usize, usize>,
    parameter_scope: Option<Vec<usize>>,
}
impl TargetFamily {
    pub fn new(scalar: ScalarFamily) -> Result<Self, String> {
        let plan = prepare(scalar.program())?;
        let mut literals = BTreeMap::new();
        for literal in scalar.literals() {
            let positions = plan.body.iter().enumerate().filter_map(|(position, item)| {
                let Item::Instruction(instruction) = item else { return None; };
                matches!(instruction.origin, Origin::Ssa { instruction, .. } if instruction == literal.instruction.as_u32()).then_some(position)
            }).collect::<Vec<_>>();
            let [position] = positions.as_slice() else { return Err("CUDA family literal does not have one retained terminal definition".into()); };
            if !matches!(&plan.body[*position], Item::Instruction(Instruction { predicate: None, operation: Operation::Unary { operation: Unary::Move, source: Operand::Signed(_), .. }, .. })) {
                return Err("CUDA family parameter did not retain its integer literal form".into());
            }
            literals.insert(*position, literal.parameter);
        }
        let mut joins = BTreeMap::new();
        for (branch, block) in scalar.joins() {
            let join = plan.body.iter().position(|item| matches!(item, Item::Label(Label::Block(id)) if *id == block.as_u32())).ok_or("CUDA family join label is absent")?;
            for (position, item) in plan.body.iter().enumerate() {
                if matches!(item, Item::Instruction(Instruction { origin: Origin::Ssa { instruction, .. }, predicate: Some(_), operation: Operation::Branch { .. } }) if *instruction == branch.as_u32()) {
                    joins.insert(position, join);
                }
            }
        }
        Ok(Self { scalar, plan, literals, joins, parameter_scope: None })
    }
    /// Bind local scalar operands to one shared parameter identity across
    /// publication phases. Scalar reconstruction still consumes local ordinals.
    pub fn bind_parameters(&mut self, parameters: &[usize]) -> Result<(), String> {
        if parameters.len()!=self.scalar.domains().len()+self.scalar.source_parameters().len() {return Err("CUDA terminal scope has a different parameter arity".into());}
        if self.parameter_scope.is_some() {return Err("CUDA terminal parameter scope is immutable".into());}
        for parameter in self.literals.values_mut() { *parameter=*parameters.get(*parameter).ok_or("CUDA terminal parameter lies outside its retained scope")?; }
        self.parameter_scope=Some(parameters.to_vec());
        Ok(())
    }
    pub fn parameter_scope(&self)->Option<&[usize]> {self.parameter_scope.as_deref()}
    pub fn scalar(&self) -> &ScalarFamily { &self.scalar }
    pub fn plan(&self) -> &TargetPlan { &self.plan }
    pub fn literals(&self) -> &BTreeMap<usize, usize> { &self.literals }
    pub fn joins(&self) -> &BTreeMap<usize, usize> { &self.joins }
    pub fn instantiate(&self, assignment: &[usize]) -> Result<(ScalarProgram, TargetPlan), String> {
        let scalar = self.scalar.instantiate(assignment)?;
        let mut plan = self.plan.clone();
        for (&position, _) in &self.literals {
            let Item::Instruction(instruction)=&self.plan.body[position] else {return Err("CUDA literal lost its terminal instruction".into());};
            let Origin::Ssa {instruction,..}=instruction.origin else {return Err("CUDA literal lost its scalar origin".into());};
            let parameter=self.scalar.literals().iter().find(|literal|literal.instruction.as_u32()==instruction).ok_or("CUDA literal lost its original parameter")?.parameter;
            let ordinal = *assignment.get(parameter).ok_or("missing CUDA terminal parameter")?;
            let value = if let Some(domain) = self.scalar.domains().get(parameter) {
                i64::from(*domain.get(ordinal).ok_or("CUDA load parameter exceeds its original domain")? == seismic_lang::ir::LoadMode::Borrow)
            } else {
                if ordinal > 1 { return Err("CUDA source predicate must be boolean".into()); }
                ordinal as i64
            };
            let Item::Instruction(Instruction { operation: Operation::Unary { source, .. }, .. }) = &mut plan.body[position] else { return Err("CUDA retained parameter definition changed".into()); };
            *source = Operand::Signed(value);
        }
        Ok((scalar, plan))
    }
}
