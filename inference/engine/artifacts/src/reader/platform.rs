use super::file::Through;
use super::volume::VolumeKey;
#[cfg(doc)]
use super::ArtifactReader;
use crate::Error;
use std::fs::File;
use std::io;
use std::path::Path;

/// The pages wholly within `length` bytes at `offset` of the file at `path`
/// that the file cache holds, and all those pages (`mincore` of a read-only
/// mapping of them, which faults none in). A page the range shares with the
/// bytes beside it is not the range's alone, so it is not counted.
#[cfg(unix)]
pub fn cached_pages(path: &Path, offset: u64, length: u64) -> Result<(u64, u64), Error> {
    use std::os::fd::AsRawFd;
    let file = File::open(path)?;
    // SAFETY: `sysconf` reads a constant of the system.
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .map_err(|_| std::io::Error::last_os_error())?;
    let outside = || Error::Invalid("the range exceeds the host file domain".into());
    let end = offset.checked_add(length).ok_or_else(outside)? / page * page;
    let start = offset.checked_next_multiple_of(page).ok_or_else(outside)?;
    if end <= start {
        return Ok((0, 0));
    }
    let span = usize::try_from(end - start)
        .map_err(|_| Error::Invalid("the range exceeds the host address range".into()))?;
    let first = libc::off_t::try_from(start).map_err(|_| outside())?;
    let mut pages = vec![0; span.div_ceil(page as usize)];
    // SAFETY: a read-only shared mapping of an open file, unmapped before
    // returning; `pages` holds one entry per page of it.
    unsafe {
        let mapped = libc::mmap(
            std::ptr::null_mut(),
            span,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            first,
        );
        if mapped == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let read = libc::mincore(mapped, span, pages.as_mut_ptr());
        let error = std::io::Error::last_os_error();
        libc::munmap(mapped, span);
        if read != 0 {
            return Err(error.into());
        }
    }
    let cached = pages.iter().filter(|state| **state & 1 == 1).count();
    Ok((cached as u64, pages.len() as u64))
}

/// [`ArtifactReader::evict_cached`] of the `size` bytes of `file`.
pub(super) fn evict(file: &File, size: u64) -> Result<(), Error> {
    Ok(drop_pages(file, 0, size)?)
}

/// [`cached_pages`] where the platform has no `mincore`.
#[cfg(not(unix))]
pub fn cached_pages(_path: &Path, _offset: u64, _length: u64) -> Result<(u64, u64), Error> {
    Err(Error::Invalid(
        "the file cache's pages are read only on unix".into(),
    ))
}

/// The processor seconds the calling thread has run.
#[cfg(unix)]
pub(super) fn thread_cpu_seconds() -> Result<f64, Error> {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: the clock writes the one `timespec` it is given.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(time.tv_sec as f64 + time.tv_nsec as f64 * 1e-9)
}

#[cfg(windows)]
pub(super) fn thread_cpu_seconds() -> Result<f64, Error> {
    use std::os::windows::io::RawHandle;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> RawHandle;
        fn GetThreadTimes(thread: RawHandle, creation: *mut u64, exit: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
    }
    let (mut creation, mut exit, mut kernel, mut user) = (0u64, 0u64, 0u64, 0u64);
    // SAFETY: the calling thread's pseudo-handle, and four `FILETIME`s
    // (each 100 ns ticks in a u64) the call writes.
    if unsafe { GetThreadTimes(GetCurrentThread(), &mut creation, &mut exit, &mut kernel, &mut user) } == 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok((kernel + user) as f64 * 1e-7)
}

/// The volume `file` lies on.
#[cfg(unix)]
pub(super) fn volume_key(file: &File) -> Result<VolumeKey, Error> {
    use std::os::unix::fs::MetadataExt;
    Ok(VolumeKey(file.metadata()?.dev()))
}

#[cfg(windows)]
pub(super) fn volume_key(file: &File) -> Result<VolumeKey, Error> {
    use std::os::windows::io::{AsRawHandle, RawHandle};
    /// `BY_HANDLE_FILE_INFORMATION`.
    #[repr(C)]
    #[derive(Default)]
    struct Information {
        attributes: u32,
        times: [u32; 6],
        volume_serial_number: u32,
        size: [u32; 2],
        links: u32,
        index: [u32; 2],
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetFileInformationByHandle(file: RawHandle, information: *mut Information) -> i32;
    }
    let mut information = Information::default();
    // SAFETY: a handle `file` owns, and a structure of the layout the call
    // writes.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(VolumeKey(u64::from(information.volume_serial_number)))
}

/// How an uncached reader reads `file`, a source's open file: that file
/// opened again for reads without the cache, or, where its filesystem
/// refuses direct I/O (tmpfs, some overlays), `file` itself, read through
/// the cache.
#[cfg(any(target_os = "linux", windows))]
pub(super) fn uncached(file: &File) -> Result<Through, Error> {
    match reopen(file, true) {
        Ok(direct) => Ok(Through::direct(direct)),
        Err(error) if refuses_direct(&error) => Ok(Through::Dropped(file.try_clone()?)),
        Err(error) => Err(error.into()),
    }
}

/// macOS opens the file again by its identity or its name (`reopen`); one
/// neither names any more (replaced or removed under its source) is read
/// through the cache, each read's pages dropped.
#[cfg(target_os = "macos")]
pub(super) fn uncached(file: &File) -> Result<Through, Error> {
    match reopen(file, true) {
        Ok(direct) => Ok(Through::direct(direct)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Through::Dropped(file.try_clone()?)),
        Err(error) => Err(error.into()),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(super) fn uncached(file: &File) -> Result<Through, Error> {
    Ok(Through::Dropped(file.try_clone()?))
}

/// The file `file` has open, opened again (a new open file, whose flags are
/// its own): past the cache where `direct`, through it where not. It is
/// never what `file`'s path names now, which may be another file.
#[cfg(target_os = "linux")]
pub(super) fn reopen(file: &File, direct: bool) -> io::Result<File> {
    use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
    // `/proc/self/fd/N` opens the file descriptor N has open.
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(if direct { libc::O_DIRECT } else { 0 })
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

/// macOS opens no file by its descriptor, but by its identity: the
/// volume file system's path of its device and inode (`/.vol/<dev>/<ino>`),
/// whatever its path names now. A file system without that path (one not
/// APFS or HFS+) opens the descriptor's path (`F_GETPATH`) instead, refused
/// unless it is still the same file.
#[cfg(target_os = "macos")]
pub(super) fn reopen(file: &File, direct: bool) -> io::Result<File> {
    use std::os::{
        fd::AsRawFd,
        unix::{ffi::OsStrExt, fs::MetadataExt},
    };
    let open = file.metadata()?;
    let reopened = match File::open(format!("/.vol/{}/{}", open.dev(), open.ino())) {
        Ok(reopened) => reopened,
        Err(_) => {
            let mut path = [0 as libc::c_char; libc::PATH_MAX as usize];
            // SAFETY: `F_GETPATH` writes at most `PATH_MAX` bytes, the path
            // and its terminating NUL, into `path`.
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETPATH, path.as_mut_ptr()) } == -1 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `F_GETPATH` succeeded, so `path` holds a NUL-terminated
            // string.
            let path = unsafe { std::ffi::CStr::from_ptr(path.as_ptr()) };
            File::open(std::ffi::OsStr::from_bytes(path.to_bytes()))?
        }
    };
    let named = reopened.metadata()?;
    if (open.dev(), open.ino()) != (named.dev(), named.ino()) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "the open artifact's path names another file",
        ));
    }
    // SAFETY: an fcntl on a descriptor this function owns.
    if direct && unsafe { libc::fcntl(reopened.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(reopened)
}

#[cfg(windows)]
pub(super) fn reopen(file: &File, direct: bool) -> io::Result<File> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, RawHandle};
    const GENERIC_READ: u32 = 0x8000_0000;
    // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, as `File::open` shares.
    const SHARE_ALL: u32 = 0x7;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    #[link(name = "kernel32")]
    extern "system" {
        fn ReOpenFile(original: RawHandle, access: u32, share: u32, flags: u32) -> RawHandle;
    }
    let flags = if direct { FILE_FLAG_NO_BUFFERING } else { 0 };
    // SAFETY: a reopen of a handle `file` owns; the new handle is the
    // returned file's alone.
    let handle = unsafe { ReOpenFile(file.as_raw_handle(), GENERIC_READ, SHARE_ALL, flags) };
    // INVALID_HANDLE_VALUE.
    if handle as isize == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `handle` is a valid handle nothing else owns.
    Ok(unsafe { File::from_raw_handle(handle) })
}

/// No direct I/O here: every reader's file already reads through the
/// cache, so a direct reopen is refused and any other shares `file`.
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(super) fn reopen(file: &File, direct: bool) -> io::Result<File> {
    if direct {
        return Err(io::ErrorKind::Unsupported.into());
    }
    file.try_clone()
}

/// The file opened for reads through the cache, advised random access: the
/// cache reads ahead of no range. Every cached read of a model file opens it
/// so (an [`ArtifactReader`] and a [`crate::FileSource`]): read-ahead past a
/// resident weight filled the cache with the expert stacks beside it, 5716
/// pages of a streamed load on a GB10.
#[cfg(target_os = "linux")]
pub(crate) fn open_cached(path: &Path) -> Result<File, Error> {
    use std::os::fd::AsRawFd;
    let file = File::open(path)?;
    // SAFETY: advice on the whole of a descriptor this function owns.
    let advised =
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
    if advised != 0 {
        return Err(io::Error::from_raw_os_error(advised).into());
    }
    Ok(file)
}

/// The file opened for reads through the cache, with the platform's
/// read-ahead.
#[cfg(not(target_os = "linux"))]
pub(crate) fn open_cached(path: &Path) -> Result<File, Error> {
    Ok(File::open(path)?)
}

/// Whether `error` is a filesystem's refusal of direct I/O, rather than a
/// failure of the storage.
#[cfg(target_os = "linux")]
pub(super) fn refuses_direct(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(libc::EINVAL | libc::EOPNOTSUPP))
}

#[cfg(windows)]
pub(super) fn refuses_direct(error: &io::Error) -> bool {
    const ERROR_INVALID_PARAMETER: i32 = 87;
    error.raw_os_error() == Some(ERROR_INVALID_PARAMETER)
}

/// `F_NOCACHE` has no alignment to refuse.
#[cfg(not(any(target_os = "linux", windows)))]
pub(super) fn refuses_direct(_error: &io::Error) -> bool {
    false
}

/// Drops the file cache's pages of `start..end`, or the error that kept
/// them: pages left cached would be served from memory the reads were meant
/// not to fill.
#[cfg(target_os = "linux")]
pub(super) fn drop_pages(file: &File, start: u64, end: u64) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let (Ok(first), Ok(count)) = (libc::off_t::try_from(start), libc::off_t::try_from(end - start))
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the range exceeds the host file domain",
        ));
    };
    // SAFETY: advice on a range of a descriptor this source owns.
    let code =
        unsafe { libc::posix_fadvise(file.as_raw_fd(), first, count, libc::POSIX_FADV_DONTNEED) };
    // posix_fadvise returns the error number itself, never setting errno.
    match code {
        0 => Ok(()),
        code => Err(io::Error::from_raw_os_error(code)),
    }
}

/// macOS drops a range's cached pages by invalidating them through a
/// mapping of them (`msync` with `MS_INVALIDATE`), which touches none.
#[cfg(target_os = "macos")]
pub(super) fn drop_pages(file: &File, start: u64, end: u64) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: `sysconf` reads a constant of the system.
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).map_err(|_| io::Error::last_os_error())?;
    let first = start / page * page;
    if end <= first {
        return Ok(());
    }
    let outside = || io::Error::new(io::ErrorKind::InvalidInput, "the range exceeds the host address range");
    let span = usize::try_from(end - first).map_err(|_| outside())?;
    let offset = libc::off_t::try_from(first).map_err(|_| outside())?;
    // SAFETY: a read-only shared mapping of an open file, unmapped before
    // returning, whose pages are only invalidated, never touched.
    unsafe {
        let mapped = libc::mmap(std::ptr::null_mut(), span, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), offset);
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let invalidated = libc::msync(mapped, span, libc::MS_INVALIDATE);
        let error = io::Error::last_os_error();
        libc::munmap(mapped, span);
        if invalidated != 0 {
            return Err(error);
        }
    }
    Ok(())
}

/// Elsewhere reads have no cache of their own to drop.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn drop_pages(_file: &File, _start: u64, _end: u64) -> io::Result<()> {
    Ok(())
}
