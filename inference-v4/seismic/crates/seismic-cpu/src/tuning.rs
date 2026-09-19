//! CPU implementation choices and analysis of the same prepared scalar IR.
use seismic_accounting::{
    execution_model::ScalarHardware,
    workload::{DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner::{self as compiler, family, Input};
use seismic_realization::ScalarProgram;
mod export;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conditions {
    pub hardware: ScalarHardware,
    pub codegen: crate::codegen::Policy,
}
pub struct Backend {
    conditions: Conditions,
}
impl Backend {
    pub fn new(hardware: &ScalarHardware) -> Result<Self, String> {
        hardware.validate()?;
        Ok(Self {
            conditions: Conditions {
                hardware: hardware.clone(),
                codegen: crate::codegen::Policy::host()?,
            },
        })
    }
}
impl compiler::Backend for Backend {
    type Execution = ScalarProgram;
    type Conditions = Conditions;
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn conditions(&self) -> Conditions {
        self.conditions.clone()
    }
    fn description(&self) -> compiler::Description {
        compiler::Description {
            target: self.conditions.codegen.triple().into(),
            contracts: self.conditions.hardware.identity.clone(),
            form: "direct scalar".into(),
            objective: "conditional scalar completion; native mapping unqualified".into(),
            scheduling: seismic_accounting::authority::ScheduleInterpretation {
                instruction_order: seismic_accounting::authority::InstructionOrder::MaterializedFromWitness,
                timing: seismic_accounting::authority::TimingSemantics::IdealResourceFeasible,
            },
            timebase: self.conditions.hardware.timebase.clone(),
        }
    }
    fn export(&self, input: Input<'_>, workload: &ScalarWorkload, limits: DerivationLimits)
        -> Result<family::Export<ScalarProgram>, String> {
        export::build(input, &self.conditions.hardware, workload, limits)
    }
}
