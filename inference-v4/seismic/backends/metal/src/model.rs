use crate::Metal;
use seismic_estimator::{AnalyticalModelDefinition, OperationCost};
use seismic_ir::kernel::{ops::ClosedOpView, Kernel};
use seismic_ir::schedule::Launch;
use seismic_ir::storage::LaunchLocalLayout;
use seismic_ir::physical_target::KernelEmissionLayout;
use std::collections::BTreeSet;

pub struct MetalAnalyticalModel;

impl AnalyticalModelDefinition<Metal> for MetalAnalyticalModel {
    type Service = seismic_estimator_metal::MetalService;

    fn model_revision(&self) -> &'static str {
        "seismic-metal-analytical-model-v2"
    }

    fn service_state(
        &self,
        facts: &<Metal as seismic_ir::physical_target::PhysicalDialect>::Facts,
        supported_intrinsics: &BTreeSet<seismic_lang::ids::IntrinsicId>,
        service: Self::Service,
    ) -> seismic_estimator::AnalyticalServiceState {
        if seismic_estimator_metal::service_available(
            facts.bfloat_arithmetic,
            supported_intrinsics,
            service,
        ) {
            seismic_estimator::AnalyticalServiceState::Available
        } else {
            seismic_estimator::AnalyticalServiceState::Unsupported {
                reason: "the authoritative device description does not support this service",
            }
        }
    }

    fn operation_cost(
        &self,
        _facts: &<Metal as seismic_ir::physical_target::PhysicalDialect>::Facts,
        _supported_intrinsics: &BTreeSet<seismic_lang::ids::IntrinsicId>,
        arena: &mut seismic_lang::expr::ExprArena,
        kernel: &Kernel<Metal>,
        _emission: &KernelEmissionLayout,
        _launch: &Launch<Metal>,
        _locals: &LaunchLocalLayout,
        op: ClosedOpView<'_, Metal>,
    ) -> Result<OperationCost<Self::Service>, seismic_estimator::ModelLimitation> {
        seismic_estimator_metal::operation_cost(arena, kernel, op)
    }
}
