use ash::{Entry, vk};
use icn_contracts::bootstrap_protocol::{
    BackendEligibilityReport, CudaEligibility, MetalEligibility, VulkanEligibility,
};
use libloading::Library;

const CUDA_ERROR_STUB_LIBRARY: i32 = 34;
const CUDA_ERROR_NO_DEVICE: i32 = 100;

type CudaProbeResult = Result<(i32, i32, i32, i32, i32, Vec<String>), &'static str>;

fn bounded(value: impl std::fmt::Display) -> String {
    value
        .to_string()
        .replace(['\r', '\n'], " ")
        .chars()
        .take(240)
        .collect()
}

fn cuda_library_names() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["nvcuda.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        &["libcuda.so.1", "libcuda.so"]
    }
    #[cfg(target_os = "macos")]
    {
        &[]
    }
}

fn classify_cuda(result: CudaProbeResult) -> CudaEligibility {
    match result {
        Ok((init, version, count, driver_api, devices, architectures))
            if init == 0 && version == 0 && count == 0 && devices > 0 && driver_api > 0 =>
        {
            CudaEligibility::Usable {
                driver_api,
                architectures,
            }
        }
        Ok((0, _, 0, _, 0, _)) | Ok((CUDA_ERROR_NO_DEVICE, _, _, _, _, _)) => {
            CudaEligibility::Absent {
                diagnostic: "no CUDA device is available".to_owned(),
            }
        }
        Ok((CUDA_ERROR_STUB_LIBRARY, _, _, _, _, _)) => CudaEligibility::Absent {
            diagnostic: "only the CUDA stub driver is available".to_owned(),
        },
        Ok((init, version, count, _, _, _)) => CudaEligibility::Failed {
            diagnostic: bounded(format!(
                "CUDA probe failed (init={init}, version={version}, count={count})"
            )),
        },
        Err(message) => CudaEligibility::Failed {
            diagnostic: message.to_owned(),
        },
    }
}

fn cuda() -> CudaEligibility {
    type Init = unsafe extern "C" fn(u32) -> i32;
    type DriverVersion = unsafe extern "C" fn(*mut i32) -> i32;
    type DeviceCount = unsafe extern "C" fn(*mut i32) -> i32;
    type DeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type DeviceAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;

    let library = cuda_library_names()
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() });
    let Some(library) = library else {
        return CudaEligibility::Absent {
            diagnostic: "CUDA driver library is unavailable".to_owned(),
        };
    };
    let result = unsafe {
        let init = library.get::<Init>(b"cuInit\0");
        let version = library.get::<DriverVersion>(b"cuDriverGetVersion\0");
        let count = library.get::<DeviceCount>(b"cuDeviceGetCount\0");
        let get = library.get::<DeviceGet>(b"cuDeviceGet\0");
        let attribute = library.get::<DeviceAttribute>(b"cuDeviceGetAttribute\0");
        match (init, version, count, get, attribute) {
            (Ok(init), Ok(version), Ok(count), Ok(get), Ok(attribute)) => {
                let init_code = init(0);
                let mut driver_api = 0;
                let version_code = version(&mut driver_api);
                let mut device_count = 0;
                let count_code = if init_code == 0 {
                    count(&mut device_count)
                } else {
                    init_code
                };
                let mut architectures = Vec::new();
                if count_code == 0 {
                    for ordinal in 0..device_count {
                        let mut device = 0;
                        let mut major = 0;
                        let mut minor = 0;
                        if get(&mut device, ordinal) == 0
                            && attribute(&mut major, 75, device) == 0
                            && attribute(&mut minor, 76, device) == 0
                            && major > 0
                        {
                            architectures.push(format!("{major}{minor}"));
                        }
                    }
                }
                Ok((
                    init_code,
                    version_code,
                    count_code,
                    driver_api,
                    device_count,
                    architectures,
                ))
            }
            _ => Err("CUDA driver is missing required API symbols"),
        }
    };
    classify_cuda(result)
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

pub(crate) fn probe() -> BackendEligibilityReport {
    let metal = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    BackendEligibilityReport {
        schema_version: 1,
        cuda: cuda(),
        vulkan: vulkan(),
        metal: if metal {
            MetalEligibility::Usable
        } else {
            MetalEligibility::Absent {
                diagnostic: "Metal requires Apple Silicon".to_owned(),
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_stub_library_is_absent() {
        assert_eq!(
            classify_cuda(Ok((CUDA_ERROR_STUB_LIBRARY, 34, 34, 0, 0, vec![]))),
            CudaEligibility::Absent {
                diagnostic: "only the CUDA stub driver is available".to_owned(),
            }
        );
    }

    #[test]
    fn cuda_no_device_is_absent() {
        assert_eq!(
            classify_cuda(Ok((
                CUDA_ERROR_NO_DEVICE,
                0,
                CUDA_ERROR_NO_DEVICE,
                0,
                0,
                vec![],
            ))),
            CudaEligibility::Absent {
                diagnostic: "no CUDA device is available".to_owned(),
            }
        );
    }

    #[test]
    fn other_cuda_probe_errors_still_fail() {
        assert_eq!(
            classify_cuda(Ok((35, 0, 35, 0, 0, vec![]))),
            CudaEligibility::Failed {
                diagnostic: "CUDA probe failed (init=35, version=0, count=35)".to_owned(),
            }
        );
    }

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
