//! The Vulkan loader and instance.

use ash::vk;
use std::fmt;

/// Why the Vulkan loader or instance is unusable on this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoaderError {
    /// `libvulkan.so.1` / `vulkan-1.dll` could not be loaded.
    Missing(String),
    /// The loader supports only Vulkan 1.0.
    Version(u32),
    Call {
        call: &'static str,
        result: vk::Result,
    },
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(reason) => write!(f, "the Vulkan loader is unavailable: {reason}"),
            Self::Version(version) => write!(
                f,
                "the Vulkan loader supports only {}.{}; 1.3 is required",
                vk::api_version_major(*version),
                vk::api_version_minor(*version)
            ),
            Self::Call { call, result } => write!(f, "{call} failed: {result}"),
        }
    }
}

impl std::error::Error for LoaderError {}

/// The process's Vulkan 1.3 instance, created on first use and kept for the
/// process, as its logical devices are (see `Device::open`).
pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
}

/// The process's instance, or why the loader is unusable.
pub(crate) fn instance() -> Result<&'static Instance, LoaderError> {
    static INSTANCE: std::sync::OnceLock<Result<Instance, LoaderError>> =
        std::sync::OnceLock::new();
    INSTANCE
        .get_or_init(Instance::create)
        .as_ref()
        .map_err(Clone::clone)
}

impl Instance {
    fn create() -> Result<Self, LoaderError> {
        // SAFETY: loading the system Vulkan loader library.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|error| LoaderError::Missing(error.to_string()))?;
        let version = unsafe { entry.try_enumerate_instance_version() }
            .map_err(|result| LoaderError::Call {
                call: "vkEnumerateInstanceVersion",
                result,
            })?
            .unwrap_or(vk::API_VERSION_1_0);
        if vk::api_version_major(version) == 1 && vk::api_version_minor(version) == 0 {
            return Err(LoaderError::Version(version));
        }
        let application = vk::ApplicationInfo::default()
            .application_name(c"seismic")
            .engine_name(c"seismic")
            .api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&application);
        let instance =
            unsafe { entry.create_instance(&info, None) }.map_err(|result| LoaderError::Call {
                call: "vkCreateInstance",
                result,
            })?;
        Ok(Self { entry, instance })
    }

    pub(crate) fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub(crate) fn raw(&self) -> &ash::Instance {
        &self.instance
    }

    pub(crate) fn physical_devices(&self) -> Result<Vec<vk::PhysicalDevice>, LoaderError> {
        unsafe { self.instance.enumerate_physical_devices() }.map_err(|result| LoaderError::Call {
            call: "vkEnumeratePhysicalDevices",
            result,
        })
    }
}
