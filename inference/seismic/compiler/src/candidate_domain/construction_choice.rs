//! Stable names for choices made while constructing a checked source body.
//! These name source structure, never an expression arena's decision allocation.

use seismic_lang::entry::{CheckedProgramSubject, SemanticFunction, SemanticProgram};
use seismic_lang::ids::StableFunctionId;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodyMapping {
    Sequential,
    Independent,
    Authored,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BodySelection {
    pub subject: Arc<CheckedProgramSubject>,
    pub body: StableFunctionId,
    pub source_definition: u64,
    pub mapping: BodyMapping,
}

impl BodySelection {
    pub(crate) fn new(
        program: &SemanticProgram,
        function: &SemanticFunction,
        mode: crate::portable::SemanticMode,
    ) -> Self {
        use crate::portable::SemanticMode;
        Self {
            subject: program.subject().clone(),
            body: function.stable(),
            source_definition: function.source_definition(),
            mapping: match mode {
                SemanticMode::Portable => BodyMapping::Sequential,
                SemanticMode::PortableParallel => BodyMapping::Independent,
                SemanticMode::AuthoredBackend => BodyMapping::Authored,
            },
        }
    }

    pub(crate) fn mode(&self) -> crate::portable::SemanticMode {
        use crate::portable::SemanticMode;
        match self.mapping {
            BodyMapping::Sequential => SemanticMode::Portable,
            BodyMapping::Independent => SemanticMode::PortableParallel,
            BodyMapping::Authored => SemanticMode::AuthoredBackend,
        }
    }
}

/// A checked source node instantiated during physical construction. The
/// occurrence distinguishes construction under different selected input arms;
/// runtime repeat visits execute one closed body and do not create occurrences.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallLocation {
    pub body: StableFunctionId,
    pub source_definition: u64,
    pub node: Vec<u32>,
    pub occurrence: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallPath(pub Vec<CallLocation>);

#[derive(Clone, Debug)]
pub struct BodyChoice {
    pub path: CallPath,
    pub alternatives: Vec<BodySelection>,
}

/// A partial source construction path. Missing calls are unresolved choices,
/// not missing members of the domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstructionCoordinate {
    pub root: BodySelection,
    pub calls: Vec<(CallPath, BodySelection)>,
}

impl ConstructionCoordinate {
    pub fn root(root: BodySelection) -> Self {
        Self {
            root,
            calls: Vec::new(),
        }
    }

    pub fn select(&self, choice: &BodyChoice, selected: BodySelection) -> Self {
        assert!(
            choice.alternatives.contains(&selected),
            "body is outside the requested choice"
        );
        assert!(
            !self.calls.iter().any(|(path, _)| path == &choice.path),
            "call path already selected"
        );
        let mut result = self.clone();
        result.calls.push((choice.path.clone(), selected));
        result
    }
}

impl ConstructionCoordinate {
    pub fn digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        fn body(digest: &mut Sha256, value: &BodySelection) {
            digest.update(value.subject.digest());
            digest.update(value.body.digest());
            digest.update(value.source_definition.to_le_bytes());
            digest.update([match value.mapping {
                BodyMapping::Sequential => 0,
                BodyMapping::Independent => 1,
                BodyMapping::Authored => 2,
            }]);
        }
        let mut digest = Sha256::new();
        digest.update(b"seismic-source-construction-coordinate-v1");
        body(&mut digest, &self.root);
        digest.update((self.calls.len() as u64).to_le_bytes());
        for (path, selected) in &self.calls {
            digest.update((path.0.len() as u64).to_le_bytes());
            for call in &path.0 {
                digest.update(call.body.digest());
                digest.update(call.source_definition.to_le_bytes());
                digest.update((call.node.len() as u64).to_le_bytes());
                for node in &call.node {
                    digest.update(node.to_le_bytes());
                }
                digest.update(call.occurrence.to_le_bytes());
            }
            body(&mut digest, selected);
        }
        digest.finalize().into()
    }
}
