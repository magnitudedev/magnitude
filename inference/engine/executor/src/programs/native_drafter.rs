//! The drafter a load runs in the head lane: an embedded draft head (MTP)
//! or a separate draft (DFlash, DSpark). Both run one head transaction per
//! submission and publish step-major selections.

use super::native_draft::{NativeDraftProgram, PreparedDraftGraphs};
use super::native_head::{NativeHeadProgram, PreparedHeadGraphs};
use super::{DeviceSubmission, HeadProgram};
use crate::{GraphOutputTensor, HeadLaunchCore, NativeGraphWorkspaceLease, SubmitError, ValidatedHeadLaunch};
use seismic::NativeGraphFamily;
use std::rc::Rc;

pub enum PreparedDrafterGraphs {
    Head(Rc<PreparedHeadGraphs>),
    Draft(Rc<PreparedDraftGraphs>),
}

impl PreparedDrafterGraphs {
    pub fn family(&self) -> &NativeGraphFamily {
        match self {
            Self::Head(graphs) => graphs.family(),
            Self::Draft(graphs) => graphs.family(),
        }
    }

    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        match self {
            Self::Head(graphs) => graphs.binding_constant_bytes(),
            Self::Draft(graphs) => graphs.binding_constant_bytes(),
        }
    }

    pub fn class_count(&self) -> usize {
        match self {
            Self::Head(graphs) => graphs.class_count(),
            Self::Draft(graphs) => graphs.class_count(),
        }
    }
}

pub enum NativeDrafterProgram {
    Head(NativeHeadProgram),
    Draft(NativeDraftProgram),
}

impl NativeDrafterProgram {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        match self {
            Self::Head(program) => program.constant_bytes(),
            Self::Draft(program) => program.constant_bytes(),
        }
    }
}

impl HeadProgram for NativeDrafterProgram {
    type Submission =
        DeviceSubmission<HeadLaunchCore, NativeGraphWorkspaceLease, Option<GraphOutputTensor>>;

    fn submit(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)> {
        match self {
            Self::Head(program) => program.submit(launch),
            Self::Draft(program) => program.submit(launch),
        }
    }
}
