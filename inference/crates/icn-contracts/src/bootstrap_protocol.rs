use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BackendEligibilityReport {
    #[cfg_attr(feature = "openapi", schema(minimum = 1, maximum = 1))]
    pub schema_version: u32,
    pub cuda: CudaEligibility,
    pub vulkan: VulkanEligibility,
    pub metal: MetalEligibility,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(
    tag = "state",
    rename_all = "lowercase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum CudaEligibility {
    Usable {
        #[serde(rename = "driverApi")]
        #[cfg_attr(feature = "openapi", schema(minimum = 1))]
        driver_api: i32,
        #[cfg_attr(feature = "openapi", schema(min_items = 1))]
        architectures: Vec<String>,
        #[serde(rename = "driverLibrary")]
        #[cfg_attr(feature = "openapi", schema(min_length = 1))]
        driver_library: String,
    },
    Absent {
        diagnostic: String,
    },
    Failed {
        diagnostic: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(
    tag = "state",
    rename_all = "lowercase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum VulkanEligibility {
    Usable {
        #[serde(rename = "loaderApi")]
        #[cfg_attr(feature = "openapi", schema(minimum = 1))]
        loader_api: u32,
    },
    Absent {
        diagnostic: String,
    },
    Failed {
        diagnostic: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "state", rename_all = "lowercase", deny_unknown_fields)]
pub enum MetalEligibility {
    Usable,
    Absent { diagnostic: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct IcnBinaryIdentity {
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub version: String,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub api_version: u32,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub native_build: String,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub backend_module_abi: String,
    #[cfg_attr(feature = "openapi", schema(min_items = 1))]
    pub capabilities: Vec<String>,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub target: String,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub profile: String,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub rustc: String,
    #[cfg_attr(feature = "openapi", schema(min_items = 1))]
    pub backends: Vec<String>,
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct IcnInstallationDeclaration {
    #[cfg_attr(feature = "openapi", schema(minimum = 1, maximum = 1))]
    pub schema_version: u32,
    pub backend: IcnInstallationBackend,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub native_build: String,
    #[cfg_attr(feature = "openapi", schema(min_length = 1))]
    pub backend_module_abi: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum IcnInstallationBackend {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

impl IcnInstallationBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligibility_uses_the_state_discriminator() {
        let report = BackendEligibilityReport {
            schema_version: 1,
            cuda: CudaEligibility::Absent {
                diagnostic: "unavailable".to_owned(),
            },
            vulkan: VulkanEligibility::Usable { loader_api: 1 },
            metal: MetalEligibility::Usable,
        };
        let value = serde_json::to_value(report).expect("serialize eligibility");
        assert_eq!(value["cuda"]["state"], "absent");
        assert_eq!(value["vulkan"]["state"], "usable");
        assert_eq!(value["metal"]["state"], "usable");
        assert!(value["cuda"].get("_tag").is_none());
    }

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
            backend: IcnInstallationBackend::Cpu,
            native_build: "native".to_owned(),
            backend_module_abi: "abi".to_owned(),
        })
        .expect("serialize installation");
        assert_eq!(installation["schemaVersion"], 1);
        assert_eq!(installation["backend"], "cpu");
        assert_eq!(installation["backendModuleAbi"], "abi");
    }
}
