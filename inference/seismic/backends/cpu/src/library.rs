//! The C library functions programs call (Cranelift's library calls),
//! resolved in this process as its dynamic loader would.

/// The address of the C library function `name` in this process. A name no
/// loaded module exports is an invariant violation: the process links the C
/// library every library call Cranelift emits is part of.
pub(crate) fn address(name: &str) -> usize {
    let symbol = std::ffi::CString::new(name).expect("a library function name holds no NUL");
    lookup(&symbol).unwrap_or_else(|| {
        panic!(
            "a CPU program calls the library function `{name}`, which the process does not export"
        )
    })
}

/// A lookup in the process's global symbol scope.
#[cfg(unix)]
fn lookup(symbol: &std::ffi::CStr) -> Option<usize> {
    // SAFETY: a lookup in the global scope by a NUL-terminated name.
    let address = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr()) };
    (!address.is_null()).then_some(address as usize)
}

/// A lookup in every module the process has loaded, in load order (the
/// executable first): Windows has no global symbol scope, and the C
/// runtime's functions are exported by its DLLs.
#[cfg(windows)]
fn lookup(symbol: &std::ffi::CStr) -> Option<usize> {
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    use windows_sys::Win32::System::ProcessStatus::K32EnumProcessModules;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut modules: Vec<HMODULE> = Vec::new();
    loop {
        let capacity = u32::try_from(std::mem::size_of_val(modules.as_slice()))
            .expect("the process's module list fits in a u32 of bytes");
        let mut needed = 0u32;
        // SAFETY: the buffer holds `capacity` bytes of module handles.
        if unsafe {
            K32EnumProcessModules(
                GetCurrentProcess(),
                modules.as_mut_ptr(),
                capacity,
                &mut needed,
            )
        } == 0
        {
            panic!(
                "listing the process's modules failed: {}",
                std::io::Error::last_os_error()
            );
        }
        let count = needed as usize / std::mem::size_of::<HMODULE>();
        if needed <= capacity {
            modules.truncate(count);
            break;
        }
        modules.resize(count, std::ptr::null_mut());
    }
    modules.into_iter().find_map(|module| {
        // SAFETY: a loaded module's handle and a NUL-terminated name.
        unsafe { GetProcAddress(module, symbol.as_ptr().cast()) }.map(|address| address as usize)
    })
}
