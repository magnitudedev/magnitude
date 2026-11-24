//! Filesystem space for a path's volume: its total size and the bytes this
//! process may still write on it.

use std::path::Path;

/// The total size and writer-available bytes of the filesystem holding a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumeSpace {
    pub size: u64,
    pub available: u64,
}

/// The total size and space this process can still write on the filesystem
/// holding `path`.
pub fn volume_space(path: &Path) -> std::io::Result<VolumeSpace> {
    platform::space(path)
}

#[cfg(target_os = "macos")]
mod platform {
    use super::VolumeSpace;
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    pub(super) fn space(path: &Path) -> std::io::Result<VolumeSpace> {
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        // Use 64-bit `statfs` rather than `statvfs` (whose 32-bit `f_blocks`
        // truncates volumes above 16 TiB at 4 KiB block size).
        let mut status = unsafe { std::mem::zeroed::<libc::statfs>() };
        if unsafe { libc::statfs(c_path.as_ptr(), &mut status) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let block = u64::from(status.f_bsize);
        Ok(VolumeSpace {
            size: status.f_blocks.saturating_mul(block),
            available: status.f_bavail.saturating_mul(block),
        })
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use super::VolumeSpace;
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    pub(super) fn space(path: &Path) -> std::io::Result<VolumeSpace> {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        let mut status = unsafe { std::mem::zeroed::<libc::statvfs>() };
        if unsafe { libc::statvfs(path.as_ptr(), &mut status) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let fragment = if status.f_frsize != 0 {
            status.f_frsize as u64
        } else {
            status.f_bsize as u64
        };
        Ok(VolumeSpace {
            size: (status.f_blocks as u64).saturating_mul(fragment),
            available: (status.f_bavail as u64).saturating_mul(fragment),
        })
    }
}

#[cfg(windows)]
mod platform {
    use super::VolumeSpace;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    pub(super) fn space(path: &Path) -> std::io::Result<VolumeSpace> {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let (mut available, mut size) = (0u64, 0u64);
        if unsafe {
            GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut available,
                &mut size,
                std::ptr::null_mut(),
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(VolumeSpace { size, available })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_space_reports_available_within_size() {
        let space = volume_space(&std::env::temp_dir()).unwrap();
        assert!(space.size > 0);
        assert!(space.size >= space.available);
    }
}
