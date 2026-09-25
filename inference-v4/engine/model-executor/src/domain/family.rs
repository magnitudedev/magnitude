//! The worker binds one program family before constructing its numerical domain.
//! Every lane keeps its concrete submission type through completion and finish.

use crate::programs::{
    CompletedHeadWork, CompletedStateWork, CompletedTargetWork, CompletedVisionWork, HeadProgram,
    ProgramSubmission, StateProgram, SubmittedTarget, TargetProgram, VisionProgram,
};
use crate::{
    AttestedPrograms, ResidentHead, ResidentTarget, ResidentVision, SubmitError,
    ValidatedHeadLaunch, ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch,
};
use magnitude_model_contracts::{DecoderGeometry, ModelDefinition};
use std::rc::Rc;

pub trait ProgramFamily: 'static {
    type TargetSubmission: ProgramSubmission<CompletedWork = CompletedTargetWork>
        + SubmittedTarget
        + 'static;
    type HeadSubmission: ProgramSubmission<CompletedWork = CompletedHeadWork> + 'static;
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
    fn unbind_optional(&mut self);

    /// Physical target tensors held by the bound family. Test families with
    /// no resident model have no target charge.
    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn prepared_program_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn bound_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn submit_target(
        &mut self,
        launch: ValidatedTargetLaunch,
    ) -> Result<Self::TargetSubmission, (SubmitError, ValidatedTargetLaunch)>;
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)>;
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
    resident: ResidentTarget,
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
    ) -> Result<Self, String> {
        let binding_constants = programs
            .target_graphs()
            .ok_or("target graph family was not prepared")?
            .binding_constant_bytes()?;
        let target = programs
            .bind_target(resident.clone(), geometry)
            .map_err(|error| error.to_string())?;
        if target.constant_bytes()? != binding_constants {
            return Err("bound target graph constants differ from the claimed charge".into());
        }
        let state = programs.bind_state();
        Ok(Self {
            programs,
            resident,
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
            let binding_constants = self
                .programs
                .head_graphs()
                .ok_or("head graph family was not prepared")?
                .binding_constant_bytes()?;
            let head = self
                .programs
                .bind_head(resident, definition)
                .map_err(|error| error.to_string())?;
            if head.constant_bytes()? != binding_constants {
                return Err("bound head graph constants differ from the claimed charge".into());
            }
            self.head = Some(head);
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

    fn unbind_optional(&mut self) {
        self.head = None;
        self.vision = None;
    }

    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        self.resident.storage_bytes()
    }

    fn prepared_program_bytes(&self) -> Result<u64, &'static str> {
        self.programs.device_storage_bytes()
    }

    fn bound_constant_bytes(&self) -> Result<u64, &'static str> {
        self.target
            .constant_bytes()?
            .checked_add(
                self.head
                    .as_ref()
                    .map(|head| head.constant_bytes())
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or("bound graph constant charge overflows")
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
