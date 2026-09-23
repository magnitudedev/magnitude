//! The worker binds one program family before constructing its numerical domain.
//! Every lane keeps its concrete submission type through completion and finish.

use crate::programs::{
    CompletedHeadWork, CompletedProjectWork, CompletedStateWork, CompletedTargetWork,
    CompletedVisionWork, HeadProgram, ProgramSubmission, StateProgram, TargetProgram,
    VisionProgram,
};
use crate::{
    AttestedPrograms, ResidentHead, ResidentTarget, ResidentVision, SubmitError,
    ValidatedHeadLaunch, ValidatedProjectionLaunch, ValidatedStateLaunch, ValidatedTargetLaunch,
    ValidatedVisionLaunch,
};
use magnitude_model_contracts::{DecoderGeometry, ModelDefinition};
use magnitude_model_state::StateStore;
use std::rc::Rc;

pub trait ProgramFamily: 'static {
    type TargetSubmission: ProgramSubmission<CompletedWork = CompletedTargetWork> + 'static;
    type HeadSubmission: ProgramSubmission<CompletedWork = CompletedHeadWork> + 'static;
    type ProjectSubmission: ProgramSubmission<CompletedWork = CompletedProjectWork> + 'static;
    type VisionSubmission: ProgramSubmission<CompletedWork = CompletedVisionWork> + 'static;
    type StateSubmission: ProgramSubmission<CompletedWork = CompletedStateWork> + 'static;

    fn bind_head(
        &mut self,
        resident: ResidentHead,
        definition: &ModelDefinition,
    ) -> Result<(), String>;
    fn bind_vision(
        &mut self,
        resident: ResidentVision,
        definition: &ModelDefinition,
    ) -> Result<(), String>;
    fn head_is_bound(&self) -> bool;
    fn vision_is_bound(&self) -> bool;
    fn state_is_bound(&self) -> bool;

    fn submit_target(
        &mut self,
        launch: ValidatedTargetLaunch,
    ) -> Result<Self::TargetSubmission, (SubmitError, ValidatedTargetLaunch)>;
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)>;
    fn submit_project(
        &mut self,
        launch: ValidatedProjectionLaunch,
    ) -> Result<Self::ProjectSubmission, (SubmitError, ValidatedProjectionLaunch)>;
    fn submit_vision(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (SubmitError, ValidatedVisionLaunch)>;
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (SubmitError, ValidatedStateLaunch)>;
}

pub struct NativeFamily {
    programs: Rc<AttestedPrograms>,
    target: crate::programs::native_target::NativeTargetProgram,
    state: crate::programs::native_state::NativeStateProgram,
    head: Option<crate::programs::native_head::NativeHeadProgram>,
    vision: Option<crate::programs::native_vision::NativeVisionProgram>,
}

impl NativeFamily {
    pub(crate) fn new(
        programs: Rc<AttestedPrograms>,
        resident: ResidentTarget,
        geometry: DecoderGeometry,
        store: Rc<StateStore>,
    ) -> Result<Self, String> {
        let target = programs
            .bind_target(resident, geometry)
            .map_err(|error| error.to_string())?;
        let state = programs.bind_state_with_repair(&target, store);
        Ok(Self {
            programs,
            target,
            state,
            head: None,
            vision: None,
        })
    }
}

impl ProgramFamily for NativeFamily {
    type TargetSubmission =
        <crate::programs::native_target::NativeTargetProgram as TargetProgram>::Submission;
    type HeadSubmission =
        <crate::programs::native_head::NativeHeadProgram as HeadProgram>::Submission;
    type ProjectSubmission =
        <crate::programs::native_head::NativeHeadProgram as HeadProgram>::ProjectSubmission;
    type VisionSubmission =
        <crate::programs::native_vision::NativeVisionProgram as VisionProgram>::Submission;
    type StateSubmission =
        <crate::programs::native_state::NativeStateProgram as StateProgram>::Submission;

    fn bind_head(
        &mut self,
        resident: ResidentHead,
        definition: &ModelDefinition,
    ) -> Result<(), String> {
        if self.head.is_none() {
            self.head = Some(
                self.programs
                    .bind_head(resident, definition)
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(())
    }
    fn bind_vision(
        &mut self,
        resident: ResidentVision,
        definition: &ModelDefinition,
    ) -> Result<(), String> {
        if self.vision.is_none() {
            self.vision = Some(
                self.programs
                    .bind_vision(resident, definition)
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(())
    }
    fn head_is_bound(&self) -> bool {
        self.head.is_some()
    }
    fn vision_is_bound(&self) -> bool {
        self.vision.is_some()
    }
    fn state_is_bound(&self) -> bool {
        true
    }

    fn submit_target(
        &mut self,
        launch: ValidatedTargetLaunch,
    ) -> Result<Self::TargetSubmission, (SubmitError, ValidatedTargetLaunch)> {
        self.target.submit(launch)
    }
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)> {
        self.head
            .as_mut()
            .expect("head was bound before submission")
            .submit(launch)
    }
    fn submit_project(
        &mut self,
        launch: ValidatedProjectionLaunch,
    ) -> Result<Self::ProjectSubmission, (SubmitError, ValidatedProjectionLaunch)> {
        self.head
            .as_mut()
            .expect("head was bound before projection")
            .project(launch)
    }
    fn submit_vision(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (SubmitError, ValidatedVisionLaunch)> {
        self.vision
            .as_mut()
            .expect("vision was bound before submission")
            .submit(launch)
    }
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (SubmitError, ValidatedStateLaunch)> {
        self.state.submit(launch)
    }
}
