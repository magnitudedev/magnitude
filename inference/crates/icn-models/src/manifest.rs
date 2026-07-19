use std::path::PathBuf;

use icn_contracts::{
    ComponentRelationship, ComponentRole, ContentId, ContentIdentity, ModelComponent, ModelId,
};
use serde::{Deserialize, Serialize};

pub const MANIFEST_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedManifest {
    pub version: u32,
    pub model_id: ModelId,
    pub content_id: ContentId,
    pub repository: String,
    pub requested_revision: String,
    pub commit: String,
    pub components: Vec<ModelComponent>,
    pub created_at: u64,
    pub ready_at: u64,
}

impl ManagedManifest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != MANIFEST_VERSION {
            return Err("unsupported managed manifest version");
        }
        if self.components.is_empty() {
            return Err("managed manifest has no components");
        }
        if self.repository.is_empty() || self.commit.is_empty() {
            return Err("managed manifest is missing repository identity");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationManifest {
    pub version: u32,
    pub operation_id: String,
    pub model_id: ModelId,
    pub content_id: ContentId,
    pub repository: String,
    pub requested_revision: String,
    pub commit: String,
    pub components: Vec<OperationComponent>,
    pub stage: String,
    pub started_at: u64,
    pub updated_at: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationComponent {
    pub path: PathBuf,
    pub role: ComponentRole,
    pub content: ContentIdentity,
    pub shard_index: Option<u32>,
    pub relationship: Option<ComponentRelationship>,
    pub expected_size: u64,
    pub content_key: String,
    pub completed_bytes: u64,
}
