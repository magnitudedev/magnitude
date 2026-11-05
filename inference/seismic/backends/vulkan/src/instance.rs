//! The Vulkan loader and instance.

use ash::vk;
use std::fmt;
use std::sync::{Arc, Mutex, Weak};

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

/// The process's Vulkan 1.3 instance while anything holds it: discovery's
/// device handles and every logical device created from it. Its last
/// holder's drop destroys it; a logical device holds it, so it outlives
/// every device created from it, and nothing can use it once destroyed.
pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
}

/// The process's instance, created when none is held, or why the loader is
/// unusable.
pub(crate) fn instance() -> Result<Arc<Instance>, LoaderError> {
    static CURRENT: Mutex<Weak<Instance>> = Mutex::new(Weak::new());
    let mut current = CURRENT.lock().expect("the instance lock is never poisoned");
    if let Some(instance) = current.upgrade() {
        return Ok(instance);
    }
    let instance = Arc::new(Instance::create()?);
    *current = Arc::downgrade(&instance);
    Ok(instance)
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: every object created from the instance holds it, so none
        // is left.
        unsafe { self.instance.destroy_instance(None) };
    }
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
