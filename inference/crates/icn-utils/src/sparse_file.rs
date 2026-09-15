use std::fs::File;
use std::io;

/// Extend a header-only file without allocating its unwritten model payload.
pub fn set_sparse_len(file: &File, length: u64) -> io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use std::ptr::{null, null_mut};
        use windows_sys::Win32::System::{IO::DeviceIoControl, Ioctl::FSCTL_SET_SPARSE};

        let mut returned = 0;
        // A null input marks the file sparse. The borrowed file and output remain
        // alive for this synchronous operation; no overlapped I/O is requested.
        if unsafe {
            DeviceIoControl(
                file.as_raw_handle(),
                FSCTL_SET_SPARSE,
                null(),
                0,
                null_mut(),
                0,
                &mut returned,
                null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    file.set_len(length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    #[test]
    fn large_preview_preserves_header_and_zero_tail_without_payload_allocation() {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"GGUF header").unwrap();
        let length = 8 * 1024 * 1024 * 1024;
        set_sparse_len(&file, length).unwrap();
        assert_eq!(file.metadata().unwrap().len(), length);
        file.rewind().unwrap();
        let mut header = [0; 11];
        file.read_exact(&mut header).unwrap();
        assert_eq!(&header, b"GGUF header");
        file.seek(SeekFrom::End(-16)).unwrap();
        let mut tail = [1; 16];
        file.read_exact(&mut tail).unwrap();
        assert_eq!(tail, [0; 16]);

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert!(file.metadata().unwrap().blocks() * 512 < 1024 * 1024);
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{
                FILE_STANDARD_INFO, FileStandardInfo, GetFileInformationByHandleEx,
            };
            let mut info: FILE_STANDARD_INFO = unsafe { std::mem::zeroed() };
            // The writable output has the exact native structure size.
            assert_ne!(
                unsafe {
                    GetFileInformationByHandleEx(
                        file.as_raw_handle(),
                        FileStandardInfo,
                        (&mut info as *mut FILE_STANDARD_INFO).cast(),
                        std::mem::size_of::<FILE_STANDARD_INFO>() as u32,
                    )
                },
                0
            );
            assert!(info.AllocationSize < 1024 * 1024);
        }
    }
}
