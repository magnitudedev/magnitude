//! Specialized physical composition. No placement search or logical publication.
use super::{
    ExecutableStage, PipelineModel, PipelineRefusal, StageAllocation, StageResources,
    ValidatedPipelineLaunch,
};
use crate::{
    programs::native_target::{pipeline::StageResult, TargetOutput},
    SubmitError, TargetLaunchCore, TargetLaunchWorkspace,
};
use magnitude_family_contracts::{EmbeddingScale, ExitNorm, ResidualForm};
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

pub struct TwoStageCudaPipeline {
    pub(crate) stages: [ExecutableStage; 2],
}
#[derive(Clone, Copy, Debug)]
pub struct HandoffMeasurement {
    pub bytes: usize,
    pub device_to_host: Duration,
    pub host_to_device: Duration,
}
/// All physical work completed; tentative logical state is still unpublished.
/// The normal Owner must reconcile both advances together or discard both.
pub struct CompletedPipelineStep {
    stages: [crate::ValidatedTargetLaunch; 2],
    pub(crate) output: TargetOutput,
    pub measurement: HandoffMeasurement,
}
impl CompletedPipelineStep {
    pub(crate) fn into_parts(
        self,
    ) -> (
        [TargetLaunchCore; 2],
        [TargetLaunchWorkspace; 2],
        TargetOutput,
    ) {
        let [(a, aw), (b, bw)] = self
            .stages
            .map(crate::ValidatedTargetLaunch::into_submission_parts);
        ([a, b], [aw, bw], self.output)
    }
}
impl TwoStageCudaPipeline {
    fn qualify_model(model: &PipelineModel) -> Result<(), PipelineRefusal> {
        if model.ranges.len() != 2 {
            return Err(PipelineRefusal::StageCount);
        }
        let d = &model.definition.decoder;
        if d.residual != ResidualForm::Single
            || d.entry.scale != EmbeddingScale::Unit
            || d.entry.norm.is_some()
            || d.entry.per_layer.is_some()
            || d.entry.hash_routing.is_some()
            || !matches!(d.exit.norm, ExitNorm::Rms(_))
            || d.exit.softcap.is_some()
            || model.definition.vision.is_some()
            || model.definition.draft.is_some()
        {
            return Err(PipelineRefusal::UnsupportedProfile);
        }
        Ok(())
    }

    /// Metadata-only capability gate for this concrete executor. Must run
    /// before opening either device; construction rechecks actual owners.
    pub fn qualify(
        model: &PipelineModel,
        devices: &[seismic::DeviceSelector],
    ) -> Result<(), PipelineRefusal> {
        Self::qualify_model(model)?;
        let [first, last] = devices else {
            return Err(PipelineRefusal::StageCount);
        };
        if !matches!(first, seismic::DeviceSelector::Cuda { .. })
            || !matches!(last, seismic::DeviceSelector::Cuda { .. })
            || first == last
        {
            return Err(PipelineRefusal::ForeignDevice);
        }
        Ok(())
    }

    pub fn new(
        stages: [StageResources; 2],
    ) -> Result<(Self, [StageAllocation; 2]), PipelineRefusal> {
        let [a, b] = stages.each_ref().map(|s| &s.executable);
        if !Rc::ptr_eq(a.assignment.original_model(), b.assignment.original_model())
            || a.state_plan.limits() != b.state_plan.limits()
        {
            return Err(PipelineRefusal::ForeignModel);
        }
        let model = PipelineModel::new(
            a.assignment.original_model().clone(),
            stages
                .iter()
                .map(|s| s.executable.assignment.view().global_range())
                .collect(),
        )?;
        Self::qualify(
            &model,
            &stages
                .each_ref()
                .map(|s| s.executable.device.info().selector.clone()),
        )?;
        if a.domain == b.domain
            || a.handoff.is_some()
            || b.handoff.is_none()
            || a.graphs.readout.is_some()
            || b.graphs.readout.is_none()
            || stages[0].allocation.readout_pool.is_some()
            || stages[1].allocation.readout_pool.is_none()
        {
            return Err(PipelineRefusal::UnsupportedProfile);
        }
        let [(a, aa), (b, ba)] = stages.map(StageResources::into_parts);
        Ok((Self { stages: [a, b] }, [aa, ba]))
    }
    /// A failure consumes/discards the whole request launch. Completed GPU
    /// writes are not rolled back, and neither stage's logical state is accepted.
    pub fn execute(
        &self,
        mut launch: ValidatedPipelineLaunch,
    ) -> Result<CompletedPipelineStep, SubmitError> {
        let mut commits = None;
        let first = self.stages[0].execute_stage(&mut launch.stages[0], None, &mut commits)?;
        let StageResult::Boundary {
            rows,
            bytes,
            elapsed,
        } = first
        else {
            return Err(invalid("prefix unexpectedly published readout"));
        };
        let began = Instant::now();
        let activation = self.stages[1]
            .handoff
            .as_ref()
            .ok_or_else(|| invalid("consumer has no bounded activation buffer"))?
            .borrow_mut()
            .receive(&self.stages[1].device, rows, &bytes)
            .map_err(|e| invalid(e.to_string()))?;
        let measurement = HandoffMeasurement {
            bytes: bytes.len(),
            device_to_host: elapsed,
            host_to_device: began.elapsed(),
        };
        // The qualification-only seam exercises the normal fatal/discard path
        // after real prefix GPU writes and transfer, not pre-launch refusal.
        #[cfg(feature = "pipeline-fault-injection")]
        if let Some(raw) = std::env::var_os("MAGNITUDE_PIPELINE_FAIL_AFTER_PREFIX_POSITION") {
            let raw = raw
                .into_string()
                .map_err(|_| invalid("fault position is not UTF-8"))?;
            let positions = launch
                .stages
                .each_ref()
                .map(|s| s.core().advances()[0].position());
            if fail_at(&raw, positions[0])? {
                eprintln!("pipeline qualification suffix failure: prefix_completed=true suffix_submitted=false logical_positions={positions:?} handoff_bytes={}", measurement.bytes);
                return Err(invalid(
                    "qualification-injected suffix failure after completed prefix",
                ));
            }
        }
        let result =
            self.stages[1].execute_stage(&mut launch.stages[1], Some(&activation), &mut commits)?;
        // The consumer has physically completed before activation or host staging
        // goes away. Both stores' tentative advances remain in the checked launch.
        drop(activation);
        drop(bytes);
        let StageResult::Readout(readout) = result else {
            return Err(invalid("suffix did not reach readout"));
        };
        if std::env::var_os("MAGNITUDE_TRACE_PIPELINE").is_some() {
            eprintln!(
                "pipeline completed devices={:?} ranges={:?} handoff_bytes={} d2h_ns={} h2d_ns={}",
                self.stages.each_ref().map(|s| &s.device.info().selector),
                self.stages
                    .each_ref()
                    .map(|s| s.assignment.view().global_range()),
                measurement.bytes,
                measurement.device_to_host.as_nanos(),
                measurement.host_to_device.as_nanos()
            );
        }
        Ok(CompletedPipelineStep {
            stages: launch.stages,
            output: TargetOutput {
                readout,
                commits: commits
                    .ok_or_else(|| invalid("complete pipeline submitted no device work"))?,
            },
            measurement,
        })
    }
}
fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(crate::InvariantError {
        context: "two-stage CUDA execution",
        detail: detail.into(),
    })
}

#[cfg(feature = "pipeline-fault-injection")]
fn fail_at(raw: &str, position: usize) -> Result<bool, SubmitError> {
    let selected = raw
        .parse::<usize>()
        .map_err(|_| invalid("fault position must be a nonnegative integer"))?;
    Ok(selected == position)
}

#[cfg(all(test, feature = "pipeline-fault-injection"))]
mod failure_tests {
    use super::*;
    #[test]
    fn fault_selection_is_exact_and_invalid_configuration_refuses() {
        assert!(fail_at("2", 2).unwrap());
        assert!(!fail_at("2", 0).unwrap());
        assert!(!fail_at("2", 4).unwrap());
        for bad in ["", "-1", "decode", "184467440737095516160"] {
            assert!(fail_at(bad, 2).is_err());
        }
    }
}

#[cfg(test)]
mod placement_tests {
    use super::*;
    use crate::placement::{ModelPlacement, ModelRegion, PartitionAssignment, PlacementGroup};
    use seismic::DeviceSelector;
    fn definition() -> Rc<magnitude_family_contracts::ModelDefinition> {
        let mut definition = crate::planning::tests::fixture_definition();
        definition.decoder.blocks = vec![definition.decoder.blocks[0].clone(); 4];
        Rc::new(definition)
    }
    #[test]
    fn pipeline_projection_is_backend_neutral_but_cuda_admission_is_explicit() {
        let a = DeviceSelector::Cuda { uuid: [1; 16] };
        let b = DeviceSelector::Cuda { uuid: [2; 16] };
        for devices in [
            [a, b],
            [a, a],
            [a, DeviceSelector::HostCpu],
            [a, DeviceSelector::Vulkan { uuid: [2; 16] }],
        ] {
            let placement = ModelPlacement::pipeline(devices.into_iter().zip([0..2, 2..4]));
            let (model, assigned) =
                PipelineModel::from_placement(definition(), &placement).unwrap();
            assert_eq!(assigned, devices);
            assert_eq!(
                TwoStageCudaPipeline::qualify(&model, &devices).is_ok(),
                devices == [a, b]
            );
        }
        let model = PipelineModel::new(definition(), vec![0..2, 2..4]).unwrap();
        for devices in [vec![], vec![a], vec![a, b, a]] {
            assert_eq!(
                TwoStageCudaPipeline::qualify(&model, &devices),
                Err(PipelineRefusal::StageCount)
            );
        }
    }
    #[test]
    fn tensor_and_mixed_groups_refuse_before_any_device_is_opened() {
        use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
        let whole = PartitionAssignment {
            device: DeviceSelector::HostCpu,
            region: ModelRegion::DecoderBlocks(0..4),
        };
        let tensor = PartitionAssignment {
            device: DeviceSelector::HostCpu,
            region: ModelRegion::TensorSlice {
                role: WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::Output,
                },
                axis: 0,
                elements: 0..16,
            },
        };
        for partitions in [
            vec![],
            vec![tensor.clone()],
            vec![tensor.clone(), tensor],
            vec![whole.clone(), whole],
        ] {
            let placement = ModelPlacement {
                groups: vec![PlacementGroup { partitions }],
            };
            assert!(matches!(
                PipelineModel::from_placement(definition(), &placement),
                Err(PipelineRefusal::UnsupportedProfile)
            ));
        }
        let placement = ModelPlacement::pipeline([(DeviceSelector::HostCpu, 0..2)]);
        assert!(matches!(
            PipelineModel::from_placement(definition(), &placement),
            Err(PipelineRefusal::Coverage)
        ));
    }
}
