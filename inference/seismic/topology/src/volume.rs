//! Volumes: the filesystem a served path is on, its size and the space left
//! on it (spec §2.1).

use crate::host::ObservationError;
use crate::model::{Reported, ReportedBy, VolumeId};
use std::path::Path;

/// The identity of the volume holding `path`.
pub(crate) fn identify(path: &Path) -> std::io::Result<VolumeId> {
    platform::identify(path)
}

/// The size of the filesystem holding `path`.
pub(crate) fn size(path: &Path) -> std::io::Result<Reported<u64>> {
    platform::space(path).map(|space| Reported::new(space.size, ReportedBy::FilesystemSize))
}

/// The space this process can still write on the filesystem holding `path`:
/// the reading a [`crate::Sample`] takes of each volume, for a caller that
/// needs one volume's free space without an inventory.
pub fn volume_available(path: &Path) -> Result<Reported<u64>, ObservationError> {
    platform::space(path)
        .map(|space| Reported::new(space.available, ReportedBy::FilesystemAvailable))
        .map_err(|error| ObservationError::Failed(format!("{}: {error}", path.display())))
}

struct Space {
    size: u64,
    available: u64,
}

#[cfg(unix)]
fn statvfs(path: &Path) -> std::io::Result<Space> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let mut status = unsafe { std::mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut status) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let fragment = status.f_frsize as u64;
    Ok(Space {
        size: status.f_blocks as u64 * fragment,
        available: status.f_bavail as u64 * fragment,
    })
}

#[cfg(target_os = "macos")]
mod platform {
    use super::Space;
    use crate::model::VolumeId;
    use std::ffi::{CStr, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    use std::path::Path;

    pub(super) fn identify(path: &Path) -> std::io::Result<VolumeId> {
        let device = std::fs::metadata(path)?.dev();
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        let mut status = unsafe { std::mem::zeroed::<libc::statfs>() };
        if unsafe { libc::statfs(c_path.as_ptr(), &mut status) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // The mount point's volume UUID; a volume without one (some network
        // and FAT volumes) is known by its device alone.
        let mount = unsafe { CStr::from_ptr(status.f_mntonname.as_ptr()) };
        Ok(VolumeId {
            device,
            uuid: volume_uuid(mount)?,
        })
    }

    pub(super) fn space(path: &Path) -> std::io::Result<Space> {
        super::statvfs(path)
    }

    /// `ATTR_VOL_UUID` of the volume mounted at `mount`: none where the
    /// volume reports no UUID (the attribute left out of the reply, or all
    /// zeros); a failed call is an error.
    fn volume_uuid(mount: &CStr) -> std::io::Result<Option<String>> {
        #[repr(C)]
        struct Reply {
            length: u32,
            uuid: [u8; 16],
        }
        let mut request = unsafe { std::mem::zeroed::<libc::attrlist>() };
        request.bitmapcount = libc::ATTR_BIT_MAP_COUNT;
        request.volattr = libc::ATTR_VOL_INFO | libc::ATTR_VOL_UUID;
        let mut reply = Reply {
            length: 0,
            uuid: [0; 16],
        };
        let status = unsafe {
            libc::getattrlist(
                mount.as_ptr(),
                (&mut request as *mut libc::attrlist).cast(),
                (&mut reply as *mut Reply).cast(),
                std::mem::size_of::<Reply>(),
                0,
            )
        };
        if status != 0 {
            return Err(std::io::Error::last_os_error());
        }
        if (reply.length as usize) < std::mem::size_of::<Reply>() || reply.uuid == [0; 16] {
            return Ok(None);
        }
        let hex = reply
            .uuid
            .iter()
            .map(|byte| format!("{byte:02X}"))
            .collect::<String>();
        Ok(Some(format!(
            "{}-{}-{}-{}-{}",
            &hex[0..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..32]
        )))
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::Space;
    use crate::model::VolumeId;
    use std::os::unix::fs::MetadataExt;
    use std::path::Path;

    pub(super) fn identify(path: &Path) -> std::io::Result<VolumeId> {
        let device = std::fs::metadata(path)?.dev();
        let block = Path::new("/sys/dev/block").join(format!(
            "{}:{}",
            libc::major(device),
            libc::minor(device)
        ));
        Ok(VolumeId {
            device,
            uuid: filesystem_uuid(&block)?,
        })
    }

    pub(super) fn space(path: &Path) -> std::io::Result<Space> {
        super::statvfs(path)
    }

    /// The UUID udev names the block device by (`/dev/disk/by-uuid`). None
    /// for a filesystem without one: one on no block device (network,
    /// tmpfs, overlay), a block device udev names no device node of, a host
    /// without udev's UUID links, or a device no link resolves to. Any
    /// other read failure is an error.
    fn filesystem_uuid(block: &Path) -> std::io::Result<Option<String>> {
        let absent = |error: &std::io::Error| error.kind() == std::io::ErrorKind::NotFound;
        let uevent = match std::fs::read_to_string(block.join("uevent")) {
            Ok(uevent) => uevent,
            Err(error) if absent(&error) => return Ok(None),
            Err(error) => return Err(error),
        };
        let Some(name) = uevent
            .lines()
            .find_map(|line| line.strip_prefix("DEVNAME="))
        else {
            return Ok(None);
        };
        let device = Path::new("/dev").join(name);
        let links = match std::fs::read_dir("/dev/disk/by-uuid") {
            Ok(links) => links,
            Err(error) if absent(&error) => return Ok(None),
            Err(error) => return Err(error),
        };
        for link in links {
            let link = link?;
            match std::fs::canonicalize(link.path()) {
                Ok(target) if target == device => {
                    return Ok(Some(link.file_name().to_string_lossy().into_owned()));
                }
                // Another device's link, or one whose device is gone.
                Ok(_) => {}
                Err(error) if absent(&error) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(None)
    }
}

#[cfg(windows)]
mod platform {
    use super::Space;
    use crate::model::VolumeId;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use windows_sys::Win32::Storage::FileSystem::{
        GetDiskFreeSpaceExW, GetVolumeInformationW, GetVolumeNameForVolumeMountPointW,
        GetVolumePathNameW,
    };

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain([0]).collect()
    }

    fn text(wide: &[u16]) -> String {
        let end = wide
            .iter()
            .position(|&unit| unit == 0)
            .unwrap_or(wide.len());
        String::from_utf16_lossy(&wide[..end])
    }

    /// The volume's serial number, and its `\\?\Volume{GUID}\` name's GUID.
    pub(super) fn identify(path: &Path) -> std::io::Result<VolumeId> {
        let path = wide(path);
        let mut root = [0u16; 261];
        if unsafe { GetVolumePathNameW(path.as_ptr(), root.as_mut_ptr(), root.len() as u32) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut serial = 0u32;
        if unsafe {
            GetVolumeInformationW(
                root.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut serial,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let mut name = [0u16; 50];
        let uuid = (unsafe {
            GetVolumeNameForVolumeMountPointW(root.as_ptr(), name.as_mut_ptr(), name.len() as u32)
        } != 0)
            .then(|| text(&name))
            .and_then(|name| {
                let open = name.find('{')?;
                let close = name.find('}')?;
                Some(name[open + 1..close].to_owned())
            });
        Ok(VolumeId {
            device: u64::from(serial),
            uuid,
        })
    }

    pub(super) fn space(path: &Path) -> std::io::Result<Space> {
        let path = wide(path);
        let (mut available, mut size) = (0u64, 0u64);
        if unsafe {
            GetDiskFreeSpaceExW(
                path.as_ptr(),
                &mut available,
                &mut size,
                std::ptr::null_mut(),
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Space { size, available })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory's volume identifies the same twice, and its free space is
    /// within its size.
    #[test]
    fn a_volume_is_identified_and_sized() {
        let directory = std::env::temp_dir().join(format!("seismic-volume-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let volume = identify(&directory).unwrap();
        assert_eq!(identify(&directory).unwrap(), volume);
        assert!(size(&directory).unwrap().value >= volume_available(&directory).unwrap().value);
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
