//! Metal's implementation choices, shared by diagnostics and compiler selection.
use crate::{
    execution::{Config, Execution},
    model,
};
use seismic_accounting::workload::{DerivationLimits, ScalarWorkload};
use seismic_compiler::tuner as compiler;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decomposition {
    pub per_item: i64,
    pub split: i64,
    pub tile_piece: Option<i64>,
}
impl Default for Decomposition {
    fn default() -> Self {
        Self {
            per_item: 1,
            split: 1,
            tile_piece: None,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Form {
    #[default]
    Automatic,
    /// Explicit diagnostic restriction, never the default production domain.
    Fixed(Decomposition),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capacities {
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
}
impl Capacities {
    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Self {
        Self {
            max_threads_per_threadgroup: device.max_threads_per_threadgroup,
            max_threadgroup_bytes: device.max_threadgroup_bytes,
        }
    }
    pub(crate) fn config(&self, form: &Decomposition) -> Result<Config, String> {
        Ok(Config {
            loads: seismic_realization::LoadStrategy::Materialize,
            sg_per_tg: 1,
            piece: None,
            per_item: form.per_item,
            split: form.split,
            tile_piece: form.tile_piece,
            max_threads_per_threadgroup: self
                .max_threads_per_threadgroup
                .try_into()
                .map_err(|_| "Metal thread capacity exceeds i64")?,
            max_threadgroup_bytes: self
                .max_threadgroup_bytes
                .try_into()
                .map_err(|_| "Metal shared capacity exceeds i64")?,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conditions {
    pub target: String,
    pub capacities: Capacities,
    pub form: Form,
    pub hardware: model::Hardware,
}
pub struct Backend {
    conditions: Conditions,
}
impl Backend {
    pub fn with_conditions(conditions: Conditions) -> Result<Self, String> {
        conditions.capacities.config(&Decomposition::default())?;
        conditions.hardware.validate()?;
        if conditions.target.is_empty() {
            return Err("Metal target identity is empty".into());
        }
        Ok(Self { conditions })
    }
    #[cfg(target_os = "macos")]
    pub fn new(
        device: &crate::runtime::DeviceInfo,
        hardware: &model::Hardware,
        form: &Form,
    ) -> Result<Self, String> {
        Self::with_conditions(Conditions {
            target: device.name.clone(),
            capacities: Capacities::from_device(device),
            hardware: hardware.clone(),
            form: form.clone(),
        })
    }
}
impl compiler::Backend for Backend {
    type Execution = Execution;
    type Conditions = Conditions;
    fn name(&self) -> &'static str {
        "metal"
    }
    fn conditions(&self) -> Conditions {
        self.conditions.clone()
    }
    fn description(&self) -> compiler::Description {
        compiler::Description {
            target: self.conditions.target.clone(),
            contracts: self.conditions.hardware.identity.clone(),
            form: format!("Metal {:?}", self.conditions.form),
            objective: "conditional retained MSL completion; native mapping unqualified".into(),
            scheduling: seismic_accounting::authority::ScheduleInterpretation {
                instruction_order:
                    seismic_accounting::authority::InstructionOrder::FixedByRealization,
                timing: seismic_accounting::authority::TimingSemantics::IdealResourceFeasible,
            },
            timebase: self.conditions.hardware.timebase.clone(),
        }
    }
    fn export(
        &self,
        input: compiler::Input<'_>,
        workload: &ScalarWorkload,
        limits: DerivationLimits,
    ) -> Result<compiler::family::Export<Execution>, String> {
        crate::family::export::export(input, self.conditions.clone(), workload.clone(), limits)
    }
}
