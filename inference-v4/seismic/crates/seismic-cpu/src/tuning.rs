//! CPU implementation choices and analysis of the same prepared scalar IR.
use seismic_accounting::{
    execution_model::{self, ScalarHardware},
    schedule,
    selection::Objective,
    workload::{DerivationError, DerivationLimits, ScalarWorkload},
};
use seismic_compiler::tuner::{self as compiler, Preparation};
use seismic_lang::lowered_ir::LoweredIr;
use seismic_realization::{ScalarProgram, scheduling};

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
            timebase: self.conditions.hardware.timebase.clone(),
        }
    }
    fn prepare(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<Preparation<ScalarProgram>, String> {
        prepare(function, path)
    }
    fn analyze(
        &self,
        execution: &ScalarProgram,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<schedule::evaluation::Model, DerivationError> {
        Ok(
            execution_model::derive_scalar(execution, &self.conditions.hardware, workload, limits)?
                .model.into(),
        )
    }
    fn materialize(
        &self,
        source: &ScalarProgram,
        objective: &Objective,
    ) -> Result<ScalarProgram, String> {
        let order = scheduling::Order {
            blocks: schedule::static_order::orders(objective.flat()?.0, objective.flat()?.1)?,
        };
        let mut execution = source.clone();
        scheduling::apply(&mut execution, &order)?;
        Ok(execution)
    }
    fn check_materialization(
        &self,
        source: &ScalarProgram,
        selected: &ScalarProgram,
        objective: &Objective,
    ) -> Result<(), String> {
        let order = scheduling::Order {
            blocks: schedule::static_order::orders(objective.flat()?.0, objective.flat()?.1)?,
        };
        scheduling::check_materialization(source, selected, &order)
    }
}

/// Prepare explicit choices through the same implementation path used by tuning.
/// No hardware timings or native compilation are needed to construct this IR.
pub fn prepare(function: &LoweredIr, path: &[usize]) -> Result<Preparation<ScalarProgram>, String> {
    if function.backend != "cpu" {
        return Err("CPU preparation requires CPU Lowered IR".into());
    }
    match seismic_lang::normalize::loads::expand(function, path)? {
        seismic_lang::normalize::loads::Expansion::Choice(choice) => Ok(Preparation::Choice {
            name: format!("load site {} (variable {})", choice.site, choice.variable),
            alternatives: seismic_accounting::selection::Domain::new(choice)?,
        }),
        seismic_lang::normalize::loads::Expansion::Selected { function, consumed } => {
            if path.len() != consumed {
                return Err("unused CPU decisions".into());
            }
            Ok(Preparation::Execution(crate::prepare_resolved(&function)?))
        }
    }
}
