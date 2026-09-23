use magnitude_artifacts::ArtifactIdentity;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArtifactComponentKind {
    Target,
    Projector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactComponent {
    pub kind: ArtifactComponentKind,
    pub identity: ArtifactIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentPlan {
    pub(super) target: ArtifactComponent,
    pub(super) head: Option<ArtifactComponent>,
    pub(super) vision: Option<ArtifactComponent>,
}

impl ComponentPlan {
    pub fn target(&self) -> ArtifactComponent {
        self.target
    }

    pub fn head(&self) -> Option<ArtifactComponent> {
        self.head
    }

    pub fn vision(&self) -> Option<ArtifactComponent> {
        self.vision
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentSelection {
    pub head: bool,
    pub vision: bool,
}
