use crate::Cpu;
use seismic_estimator::{AnalyticalModelDefinition, OperationCost};
use seismic_ir::kernel::{ops::ClosedOpView, Kernel};
use seismic_ir::schedule::Launch;
use seismic_ir::storage::LaunchLocalLayout;
use seismic_ir::physical_target::KernelEmissionLayout;

pub struct CpuAnalyticalModel;

impl AnalyticalModelDefinition<Cpu> for CpuAnalyticalModel {
    type Service = crate::services::CpuService;

    fn model_revision(&self) -> &'static str {
        "seismic-cpu-analytical-model-v1"
    }

    fn operation_cost(
        &self,
        _facts: &<Cpu as seismic_ir::physical_target::PhysicalDialect>::Facts,
        _supported_intrinsics: &std::collections::BTreeSet<seismic_lang::ids::IntrinsicId>,
        arena: &mut seismic_lang::expr::ExprArena,
        kernel: &Kernel<Cpu>,
        emission: &KernelEmissionLayout,
        _launch: &Launch<Cpu>,
        _locals: &LaunchLocalLayout,
        op: ClosedOpView<'_, Cpu>,
    ) -> Result<OperationCost<Self::Service>, seismic_estimator::ModelLimitation> {
        Ok(crate::services::operation_cost(arena, kernel, emission, op))
    }
}
