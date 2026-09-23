//! Checked MTP feature projection with the same pooled submission lifecycle as
//! head forward work. A projection never owns or advances sequence state.

use crate::{
    FeatureRef, InvariantError, NativeGraphOutputLease, NativeGraphWorkspaceLease, PoolClass,
    RequestId, ResourceDomainId, SelectSpec,
};

pub struct ProjectionRequest {
    request: RequestId,
    features: FeatureRef,
    select: SelectSpec,
}

impl ProjectionRequest {
    pub fn new(request: RequestId, features: FeatureRef, select: SelectSpec) -> Self {
        Self {
            request,
            features,
            select,
        }
    }
    pub fn request(&self) -> RequestId {
        self.request
    }
    pub fn features(&self) -> &FeatureRef {
        &self.features
    }
    pub fn select(&self) -> &SelectSpec {
        &self.select
    }
    pub fn rows(&self) -> usize {
        self.features.allocation().rows()
    }
}

pub struct ProjectionLaunchInputs {
    requests: Vec<ProjectionRequest>,
    class: crate::LaunchClass,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: Option<NativeGraphOutputLease>,
}

impl ProjectionLaunchInputs {
    pub fn new(
        requests: Vec<ProjectionRequest>,
        class: crate::LaunchClass,
        graph_workspace: NativeGraphWorkspaceLease,
        graph_output: NativeGraphOutputLease,
    ) -> Self {
        Self {
            requests,
            class,
            graph_workspace,
            graph_output: Some(graph_output),
        }
    }
}

pub struct ValidatedProjectionLaunch {
    core: ProjectLaunchCore,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: Option<NativeGraphOutputLease>,
}

impl ValidatedProjectionLaunch {
    pub fn new(
        inputs: ProjectionLaunchInputs,
        domain: &ResourceDomainId,
        hidden_width: usize,
        vocabulary: usize,
    ) -> Result<Self, (ProjectionLaunchInputs, InvariantError)> {
        let invalid = |detail: &str| InvariantError {
            context: "projection launch",
            detail: detail.into(),
        };
        let capacity = inputs.class;
        if inputs.graph_workspace.domain() != domain
            || inputs
                .graph_output
                .as_ref()
                .is_none_or(|output| output.domain() != domain)
            || inputs.requests.is_empty()
            || inputs.requests.len() > capacity.segments()
            || vocabulary == 0
            || hidden_width == 0
        {
            return Err((
                inputs,
                invalid(
                    "projection class, domain, or request count differs from the admitted plan",
                ),
            ));
        }
        let mut rows = 0usize;
        for request in &inputs.requests {
            let feature = request.features.allocation();
            let Some(next) = rows.checked_add(feature.rows()) else {
                return Err((inputs, invalid("projection row count overflows")));
            };
            rows = next;
            let selection = &request.select;
            if request.features.domain() != domain
                || feature.tensor().is_err()
                || feature.rows() == 0
                || feature.width() != hidden_width
                || selection.domain == 0
                || selection
                    .mask
                    .as_ref()
                    .is_some_and(|mask| mask.len() != vocabulary.div_ceil(32))
                || selection
                    .history
                    .as_ref()
                    .is_some_and(|history| history.len() > 64)
                || selection.shaping.validate().is_err()
            {
                return Err((
                    inputs,
                    invalid("projection feature or selection controls are invalid"),
                ));
            }
        }
        if rows > capacity.rows() {
            return Err((
                inputs,
                invalid("projection rows exceed the planned head class"),
            ));
        }
        let ProjectionLaunchInputs {
            requests,
            class: _,
            graph_workspace,
            graph_output,
        } = inputs;
        Ok(Self {
            core: ProjectLaunchCore {
                requests,
                rows,
                domain: domain.clone(),
                class: capacity,
            },
            graph_workspace,
            graph_output,
        })
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }
    pub fn class(&self) -> PoolClass {
        PoolClass::Head(self.core.class)
    }
    pub(crate) fn execution_parts_mut(
        &mut self,
    ) -> (
        &ProjectLaunchCore,
        &mut NativeGraphWorkspaceLease,
        &mut Option<NativeGraphOutputLease>,
    ) {
        (
            &self.core,
            &mut self.graph_workspace,
            &mut self.graph_output,
        )
    }
    pub(crate) fn into_submission_parts(
        self,
    ) -> (
        ProjectLaunchCore,
        NativeGraphWorkspaceLease,
        Option<NativeGraphOutputLease>,
    ) {
        (self.core, self.graph_workspace, self.graph_output)
    }
}

pub struct ProjectLaunchCore {
    requests: Vec<ProjectionRequest>,
    rows: usize,
    domain: ResourceDomainId,
    class: crate::LaunchClass,
}

impl ProjectLaunchCore {
    pub fn requests(&self) -> &[ProjectionRequest] {
        &self.requests
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn physical_class(&self) -> crate::LaunchClass {
        self.class
    }
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }
    pub fn into_requests(self) -> Vec<ProjectionRequest> {
        self.requests
    }
}
