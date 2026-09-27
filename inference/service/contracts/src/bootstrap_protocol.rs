use serde::{Deserialize, Serialize};

use crate::ExecutionBackend;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct IcnBinaryIdentity {
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub version: String,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub api_version: u32,
    /// Engine build identity: engine version plus kernel bundle identity.
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub native_build: String,
    #[cfg_attr(feature = "openapi", schema(min_items = 1))]
    pub capabilities: Vec<String>,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub target: String,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub profile: String,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub rustc: String,
    /// Backends compiled into the engine build.
    #[cfg_attr(feature = "openapi", schema(min_items = 1))]
    pub backends: Vec<ExecutionBackend>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct IcnStartupRecord {
    #[serde(rename = "type")]
    pub record_type: IcnStartupRecordType,
    #[cfg_attr(feature = "openapi", schema(minimum = 1, maximum = 1))]
    pub protocol_version: u32,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub origin: String,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub instance_id: String,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub pid: u32,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub api_version: u32,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub native_build: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum IcnStartupRecordType {
    IcnReady,
}

/// Private parent-to-child control; never exposed as a management HTTP endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct IcnParentCommand {
    #[serde(rename = "type")]
    pub command_type: IcnParentCommandType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum IcnParentCommandType {
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct IcnStartupProgressRecord {
    #[serde(rename = "type")]
    pub record_type: IcnStartupProgressRecordType,
    pub backend: IcnStartupBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum IcnStartupProgressRecordType {
    PreparingBackend,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum IcnStartupBackend {
    Cpu {
        #[serde(rename = "hardwareLabel")]
        hardware_label: String,
    },
    Metal {
        #[serde(rename = "hardwareLabel")]
        hardware_label: String,
    },
    Cuda {
        #[serde(rename = "hardwareLabel")]
        hardware_label: String,
    },
    Vulkan {
        #[serde(rename = "hardwareLabel")]
        hardware_label: String,
    },
}

/// Installation declaration written by the installer or the local build beside the installed
/// service. The service selects its execution device at runtime, so the declaration names no
/// backend; it binds the installation to one engine build.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct IcnInstallationDeclaration {
    #[cfg_attr(feature = "openapi", schema(minimum = 1, maximum = 1))]
    pub schema_version: u32,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub native_build: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_and_installation_use_their_public_field_names() {
        let startup = serde_json::to_value(IcnStartupRecord {
            record_type: IcnStartupRecordType::IcnReady,
            protocol_version: 1,
            origin: "http://127.0.0.1:1".to_owned(),
            instance_id: "instance".to_owned(),
            pid: 1,
            api_version: 1,
            native_build: "native".to_owned(),
        })
        .expect("serialize startup");
        assert_eq!(startup["type"], "icn_ready");
        assert_eq!(startup["protocolVersion"], 1);
        assert_eq!(startup["instanceId"], "instance");

        let progress = serde_json::to_value(IcnStartupProgressRecord {
            record_type: IcnStartupProgressRecordType::PreparingBackend,
            backend: IcnStartupBackend::Cuda {
                hardware_label: "NVIDIA GPU".to_owned(),
            },
        })
        .expect("serialize progress");
        assert_eq!(progress["type"], "preparing_backend");
        assert_eq!(progress["backend"]["type"], "cuda");
        assert_eq!(progress["backend"]["hardwareLabel"], "NVIDIA GPU");

        let installation = serde_json::to_value(IcnInstallationDeclaration {
            schema_version: 1,
            native_build: "native".to_owned(),
        })
        .expect("serialize installation");
        assert_eq!(
            installation,
            serde_json::json!({ "schemaVersion": 1, "nativeBuild": "native" })
        );
    }

    #[test]
    fn identity_names_compiled_backends_in_lowercase() {
        let identity = serde_json::to_value(IcnBinaryIdentity {
            version: "0.0.0".to_owned(),
            api_version: 1,
            native_build: "native".to_owned(),
            capabilities: vec!["hardware".to_owned()],
            target: "aarch64-apple-darwin".to_owned(),
            profile: "release".to_owned(),
            rustc: "rustc".to_owned(),
            backends: vec![ExecutionBackend::Cpu, ExecutionBackend::Metal],
        })
        .expect("serialize identity");
        assert_eq!(identity["backends"], serde_json::json!(["cpu", "metal"]));
    }
}
