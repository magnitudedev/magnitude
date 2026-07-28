use ash::{Entry, vk};
use libloading::Library;
use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Report {
    schema_version: u32,
    cuda: Cuda,
    vulkan: Vulkan,
    metal: Metal,
}

#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "lowercase", rename_all_fields = "camelCase")]
enum Cuda {
    Usable {
        driver_api: i32,
        architectures: Vec<String>,
    },
    Absent {
        diagnostic: String,
    },
    Failed {
        diagnostic: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "lowercase", rename_all_fields = "camelCase")]
enum Vulkan {
    Usable { loader_api: u32 },
    Absent { diagnostic: String },
    Failed { diagnostic: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
enum Metal {
    Usable,
    Absent { diagnostic: String },
}

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

fn cuda() -> Cuda {
    type Init = unsafe extern "C" fn(u32) -> i32;
    type DriverVersion = unsafe extern "C" fn(*mut i32) -> i32;
    type DeviceCount = unsafe extern "C" fn(*mut i32) -> i32;
    type DeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type DeviceAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;

    let library = cuda_library_names()
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() });
    let Some(library) = library else {
        return Cuda::Absent {
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
    match result {
        Ok((init, version, count, driver_api, devices, architectures))
            if init == 0 && version == 0 && count == 0 && devices > 0 && driver_api > 0 =>
        {
            Cuda::Usable {
                driver_api,
                architectures,
            }
        }
        Ok((0, _, 0, _, 0, _)) => Cuda::Absent {
            diagnostic: "no CUDA device is available".to_owned(),
        },
        Ok((init, version, count, _, _, _)) => Cuda::Failed {
            diagnostic: bounded(format!(
                "CUDA probe failed (init={init}, version={version}, count={count})"
            )),
        },
        Err(message) => Cuda::Failed {
            diagnostic: message.to_owned(),
        },
    }
}

fn vulkan() -> Vulkan {
    let entry = match unsafe { Entry::load() } {
        Ok(entry) => entry,
        Err(error) => {
            return Vulkan::Absent {
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
        Err(error) => {
            return Vulkan::Failed {
                diagnostic: bounded(error),
            };
        }
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
        Err(error) => Vulkan::Failed {
            diagnostic: bounded(error),
        },
        Ok(_) if usable => Vulkan::Usable { loader_api: api },
        Ok(_) => Vulkan::Absent {
            diagnostic: "no non-CPU Vulkan device is available".to_owned(),
        },
    }
}

pub(crate) fn probe() -> Report {
    let metal = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    Report {
        schema_version: 1,
        cuda: cuda(),
        vulkan: vulkan(),
        metal: if metal {
            Metal::Usable
        } else {
            Metal::Absent {
                diagnostic: "Metal requires Apple Silicon".to_owned(),
            }
        },
    }
}
