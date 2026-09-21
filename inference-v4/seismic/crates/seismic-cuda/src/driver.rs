//! Minimal dynamically loaded driver ABI. Symbols are resolved once; all owned
//! objects keep their driver/context alive. No CUDA headers or libraries are
//! required to build this crate or load hardware-independent tools.
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_uchar, c_uint, c_void, CStr},
    fmt,
    rc::Rc,
};
pub type Handle = *mut c_void;
type ResultCode = c_int;
macro_rules! driver {
    ($( $field:ident: $ty:ty => $symbol:literal ),* $(,)?) => {
        pub(crate) struct Driver { $(pub $field:$ty,)* _library:Library }
        impl Driver {
            pub fn load()->Result<Rc<Self>,String> {
                #[cfg(target_os="windows")] let names=&["nvcuda.dll"];
                #[cfg(not(target_os="windows"))] let names=&["libcuda.so.1"];
                let mut errors=Vec::new();
                for name in names {
                    // CUDA driver symbols have the documented C ABI. Keeping the
                    // library in Driver preserves every copied function pointer.
                    let library=match unsafe {Library::new(name)} {Ok(l)=>l,Err(e)=>{errors.push(e.to_string());continue}};
                    unsafe {
                        $(let $field: $ty=*library.get(concat!($symbol,"\0").as_bytes()).map_err(|e|format!("CUDA driver symbol {}: {e}",$symbol))?;)*
                        let driver=Rc::new(Self { $($field,)* _library:library });
                        driver.check((driver.init)(0),"initialization")?;
                        return Ok(driver);
                    }
                }
                Err(format!("CUDA driver unavailable: {}",errors.join("; ")))
            }
        }
    }
}
driver! {
    init: unsafe extern "system" fn(c_uint)->ResultCode => "cuInit",
    device_total_memory: unsafe extern "system" fn(*mut usize,c_int)->ResultCode => "cuDeviceTotalMem_v2",
    occupancy_blocks: unsafe extern "system" fn(*mut c_int,Handle,c_int,usize)->ResultCode => "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    device_get: unsafe extern "system" fn(*mut c_int,c_int)->ResultCode => "cuDeviceGet",
    device_name: unsafe extern "system" fn(*mut c_char,c_int,c_int)->ResultCode => "cuDeviceGetName",
    device_attribute: unsafe extern "system" fn(*mut c_int,c_int,c_int)->ResultCode => "cuDeviceGetAttribute",
    driver_version: unsafe extern "system" fn(*mut c_int)->ResultCode => "cuDriverGetVersion",
    context_create: unsafe extern "system" fn(*mut Handle,c_uint,c_int)->ResultCode => "cuCtxCreate_v2",
    context_destroy: unsafe extern "system" fn(Handle)->ResultCode => "cuCtxDestroy_v2",
    context_get: unsafe extern "system" fn(*mut Handle)->ResultCode => "cuCtxGetCurrent",
    context_set: unsafe extern "system" fn(Handle)->ResultCode => "cuCtxSetCurrent",
    synchronize: unsafe extern "system" fn()->ResultCode => "cuCtxSynchronize",
    allocate: unsafe extern "system" fn(*mut u64,usize)->ResultCode => "cuMemAlloc_v2",
    free: unsafe extern "system" fn(u64)->ResultCode => "cuMemFree_v2",
    upload: unsafe extern "system" fn(u64,*const c_void,usize)->ResultCode => "cuMemcpyHtoD_v2",
    download: unsafe extern "system" fn(*mut c_void,u64,usize)->ResultCode => "cuMemcpyDtoH_v2",
    module_load: unsafe extern "system" fn(*mut Handle,*const c_void,c_uint,*mut c_int,*mut *mut c_void)->ResultCode => "cuModuleLoadDataEx",
    link_create: unsafe extern "system" fn(c_uint,*mut c_int,*mut *mut c_void,*mut Handle)->ResultCode => "cuLinkCreate_v2",
    link_add_data: unsafe extern "system" fn(Handle,c_int,*mut c_void,usize,*const c_char,c_uint,*mut c_int,*mut *mut c_void)->ResultCode => "cuLinkAddData_v2",
    link_complete: unsafe extern "system" fn(Handle,*mut *mut c_void,*mut usize)->ResultCode => "cuLinkComplete",
    link_destroy: unsafe extern "system" fn(Handle)->ResultCode => "cuLinkDestroy",
    module_unload: unsafe extern "system" fn(Handle)->ResultCode => "cuModuleUnload",
    module_function: unsafe extern "system" fn(*mut Handle,Handle,*const c_char)->ResultCode => "cuModuleGetFunction",
    function_attribute: unsafe extern "system" fn(*mut c_int,c_int,Handle)->ResultCode => "cuFuncGetAttribute",
    launch: unsafe extern "system" fn(Handle,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,Handle,*mut *mut c_void,*mut *mut c_void)->ResultCode => "cuLaunchKernel",
    launch_cooperative: unsafe extern "system" fn(Handle,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,Handle,*mut *mut c_void,*mut *mut c_void)->ResultCode => "cuLaunchCooperativeKernel",
    memcpy_device: unsafe extern "system" fn(u64,u64,usize)->ResultCode => "cuMemcpyDtoD_v2",
    memset_d8: unsafe extern "system" fn(u64,c_uchar,usize)->ResultCode => "cuMemsetD8_v2",
    error_string: unsafe extern "system" fn(ResultCode,*mut *const c_char)->ResultCode => "cuGetErrorString",
}
impl Driver {
    pub fn check_typed(&self, result: ResultCode, operation: &str) -> Result<(), DriverError> {
        if result == 0 {
            return Ok(());
        }
        let mut text = std::ptr::null();
        let description = unsafe {
            if (self.error_string)(result, &mut text) == 0 && !text.is_null() {
                CStr::from_ptr(text).to_string_lossy().into_owned()
            } else {
                "unknown driver error".into()
            }
        };
        Err(DriverError {
            operation: operation.into(),
            code: result,
            description,
        })
    }

    pub fn check(&self, result: ResultCode, operation: &str) -> Result<(), String> {
        self.check_typed(result, operation)
            .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriverError {
    pub operation: String,
    pub code: i32,
    pub description: String,
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA {}: {} ({})",
            self.operation, self.description, self.code
        )
    }
}

impl std::error::Error for DriverError {}
pub(crate) struct Context {
    pub driver: Rc<Driver>,
    raw: Handle,
}
impl Context {
    pub fn new(driver: Rc<Driver>, device: c_int) -> Result<Rc<Self>, String> {
        let mut previous = std::ptr::null_mut();
        let mut raw = std::ptr::null_mut();
        unsafe {
            driver.check((driver.context_get)(&mut previous), "current context query")?;
            driver.check(
                (driver.context_create)(&mut raw, 0, device),
                "context creation",
            )?;
            let context = Rc::new(Self {
                driver: driver.clone(),
                raw,
            });
            driver.check((driver.context_set)(previous), "restore context")?;
            Ok(context)
        }
    }
    pub fn enter(&self) -> Result<Current<'_>, String> {
        let mut previous = std::ptr::null_mut();
        unsafe {
            self.driver.check(
                (self.driver.context_get)(&mut previous),
                "current context query",
            )?;
            self.driver
                .check((self.driver.context_set)(self.raw), "set current context")?;
        }
        Ok(Current {
            driver: &self.driver,
            previous,
        })
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            (self.driver.context_destroy)(self.raw);
        }
    }
}
pub(crate) struct Current<'a> {
    driver: &'a Driver,
    previous: Handle,
}
impl Drop for Current<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.driver.context_set)(self.previous);
        }
    }
}

pub(crate) struct Allocation {
    pub pointer: u64,
    pub bytes: usize,
    pub context: Rc<Context>,
}
impl Allocation {
    pub fn new(context: &Rc<Context>, bytes: usize) -> Result<Self, String> {
        let _current = context.enter()?;
        let mut pointer = 0;
        unsafe {
            context.driver.check(
                (context.driver.allocate)(&mut pointer, bytes.max(1)),
                "allocation",
            )?;
        }
        Ok(Self {
            pointer,
            bytes,
            context: context.clone(),
        })
    }
    pub fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        if offset
            .checked_add(bytes.len())
            .is_none_or(|end| end > self.bytes)
        {
            return Err("CUDA upload exceeds allocation".into());
        }
        let _current = self.context.enter()?;
        if !bytes.is_empty() {
            unsafe {
                self.context.driver.check(
                    (self.context.driver.upload)(
                        self.pointer + offset as u64,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                    ),
                    "upload",
                )?;
            }
        }
        Ok(())
    }
    pub fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), String> {
        if offset
            .checked_add(bytes.len())
            .is_none_or(|end| end > self.bytes)
        {
            return Err("CUDA download exceeds allocation".into());
        }
        let _current = self.context.enter()?;
        if !bytes.is_empty() {
            unsafe {
                self.context.driver.check(
                    (self.context.driver.download)(
                        bytes.as_mut_ptr().cast(),
                        self.pointer + offset as u64,
                        bytes.len(),
                    ),
                    "download",
                )?;
            }
        }
        Ok(())
    }
}
impl Drop for Allocation {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.free)(self.pointer);
            }
        }
    }
}
pub(crate) struct Module {
    pub raw: Handle,
    pub context: Rc<Context>,
}
impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.module_unload)(self.raw);
            }
        }
    }
}

/// Retains JIT log buffers until the link state is destroyed. The driver owns the
/// completed cubin until destruction; callers receive an owned byte-for-byte copy.
struct Linker {
    raw: Handle,
    context: Rc<Context>,
    info: Vec<u8>,
    error: Vec<u8>,
}
impl Drop for Linker {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.link_destroy)(self.raw);
            }
        }
    }
}

pub(crate) fn compile_image(
    context: &Rc<Context>,
    source: &str,
) -> Result<(Vec<u8>, String), String> {
    let _current = context.enter()?;
    let driver = &context.driver;
    let mut info = vec![0u8; 16384];
    let mut error = vec![0u8; 16384];
    // CUDA driver ABI: INFO_LOG_BUFFER/SIZE, ERROR_LOG_BUFFER/SIZE,
    // TARGET_FROM_CUCONTEXT and LOG_VERBOSE. No resource cap or fallback policy.
    let mut options = [3, 4, 5, 6, 8, 12];
    let mut values = [
        info.as_mut_ptr().cast(),
        info.len() as *mut c_void,
        error.as_mut_ptr().cast(),
        error.len() as *mut c_void,
        std::ptr::null_mut(),
        std::ptr::without_provenance_mut::<c_void>(1),
    ];
    let mut raw = std::ptr::null_mut();
    unsafe {
        driver.check(
            (driver.link_create)(
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
                &mut raw,
            ),
            "JIT link creation",
        )?;
    }
    let linker = Linker {
        raw,
        context: context.clone(),
        info,
        error,
    };
    let mut input = std::ffi::CString::new(source)
        .map_err(|_| "PTX contains NUL")?
        .into_bytes_with_nul();
    let status = unsafe {
        (driver.link_add_data)(
            linker.raw,
            1,
            input.as_mut_ptr().cast(),
            input.len(),
            c"seismic.ptx".as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    let check = |status, operation| {
        driver.check(status, operation).map_err(|error| {
            let end = linker
                .error
                .iter()
                .position(|b| *b == 0)
                .unwrap_or(linker.error.len());
            format!("{error}\n{}", String::from_utf8_lossy(&linker.error[..end]))
        })
    };
    check(status, "PTX compilation")?;
    let mut image = std::ptr::null_mut();
    let mut size = 0;
    check(
        unsafe { (driver.link_complete)(linker.raw, &mut image, &mut size) },
        "native image linking",
    )?;
    if image.is_null() || size == 0 || size > isize::MAX as usize {
        return Err("driver returned an invalid native image".into());
    }
    // cuLinkComplete's image remains valid until cuLinkDestroy. Copy before the
    // RAII state releases it; loaded modules never borrow this driver's pointer.
    let cubin = unsafe { std::slice::from_raw_parts(image.cast::<u8>(), size) }.to_vec();
    let end = linker
        .info
        .iter()
        .position(|b| *b == 0)
        .unwrap_or(linker.info.len());
    Ok((
        cubin,
        String::from_utf8_lossy(&linker.info[..end]).into_owned(),
    ))
}
