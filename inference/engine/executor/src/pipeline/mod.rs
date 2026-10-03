//! Explicit pipeline ownership. No partition search or automatic selection.
mod graphs;
mod launch;
mod execution;
mod model;
mod stage;
mod transfer;

pub use model::{PipelineModel, PipelineRefusal, StageAssignment};
pub use stage::{StageResources, ExecutableStage, StageAllocation};

pub use launch::{PipelineLaunchRefusal, ValidatedPipelineLaunch};
pub use execution::{TwoStageCudaPipeline, CompletedPipelineStep, HandoffMeasurement};

#[cfg(test)]
mod qualification;
