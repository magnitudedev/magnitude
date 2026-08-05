use ash::{Entry, vk};
use icn_contracts::bootstrap_protocol::{
    BackendEligibilityReport, CudaEligibility, MetalEligibility, VulkanEligibility,
};

use crate::cuda_driver::{self, CudaDriverFailureKind};

fn bounded(value: impl std::fmt::Display) -> String {
    value
        .to_string()
        .replace(['\r', '\n'], " ")
        .chars()
        .take(240)
        .collect()
}

fn cuda() -> CudaEligibility {
    match cuda_driver::resolve() {
        Ok(driver) => CudaEligibility::Usable {
            driver_api: driver.driver_api,
            architectures: driver.architectures.clone(),
            driver_library: driver.path.display().to_string(),
        },
        Err(failure) if failure.kind == CudaDriverFailureKind::Absent => CudaEligibility::Absent {
            diagnostic: failure.diagnostic.clone(),
        },
        Err(failure) => CudaEligibility::Failed {
            diagnostic: failure.diagnostic.clone(),
        },
    }
}

pub(crate) fn probe_cuda() -> CudaEligibility {
    cuda()
}

fn classify_vulkan_instance_error(error: vk::Result) -> VulkanEligibility {
    if error == vk::Result::ERROR_INCOMPATIBLE_DRIVER {
        VulkanEligibility::Absent {
            diagnostic: "Vulkan driver is unavailable".to_owned(),
        }
    } else {
        VulkanEligibility::Failed {
            diagnostic: bounded(error),
        }
    }
}

fn vulkan() -> VulkanEligibility {
    let entry = match unsafe { Entry::load() } {
        Ok(entry) => entry,
        Err(error) => {
            return VulkanEligibility::Absent {
                diagnostic: bounded(error),
            };
        }
    };
    let api = unsafe { entry.try_enumerate_instance_version() }
        .ok()
        .flatten()
        .unwrap_or(vk::API_VERSION_1_0);
    let application = vk::ApplicationInfo::default()
        .application_name(c"magnitude-icn")
        .api_version(api.min(vk::API_VERSION_1_1));
    let create = vk::InstanceCreateInfo::default().application_info(&application);
    let instance = match unsafe { entry.create_instance(&create, None) } {
        Ok(instance) => instance,
        Err(error) => return classify_vulkan_instance_error(error),
    };
    let devices = unsafe { instance.enumerate_physical_devices() };
    let usable = devices.as_ref().is_ok_and(|devices| {
        devices.iter().any(|device| {
            unsafe { instance.get_physical_device_properties(*device) }.device_type
                != vk::PhysicalDeviceType::CPU
        })
    });
    unsafe { instance.destroy_instance(None) };
    match devices {
        Err(error) => VulkanEligibility::Failed {
            diagnostic: bounded(error),
        },
        Ok(_) if usable => VulkanEligibility::Usable { loader_api: api },
        Ok(_) => VulkanEligibility::Absent {
            diagnostic: "no non-CPU Vulkan device is available".to_owned(),
        },
    }
}

pub(crate) fn probe_vulkan() -> VulkanEligibility {
    vulkan()
}

pub(crate) fn probe_metal() -> MetalEligibility {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        MetalEligibility::Usable
    } else {
        MetalEligibility::Absent {
            diagnostic: "Metal requires Apple Silicon".to_owned(),
        }
    }
}

pub(crate) fn probe() -> BackendEligibilityReport {
    BackendEligibilityReport {
        schema_version: 1,
        cuda: probe_cuda(),
        vulkan: probe_vulkan(),
        metal: probe_metal(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_vulkan_driver_is_absent() {
        assert_eq!(
            classify_vulkan_instance_error(vk::Result::ERROR_INCOMPATIBLE_DRIVER),
            VulkanEligibility::Absent {
                diagnostic: "Vulkan driver is unavailable".to_owned(),
            }
        );
    }

    #[test]
    fn other_vulkan_instance_errors_still_fail() {
        assert!(matches!(
            classify_vulkan_instance_error(vk::Result::ERROR_INITIALIZATION_FAILED),
            VulkanEligibility::Failed { .. }
        ));
    }
}
