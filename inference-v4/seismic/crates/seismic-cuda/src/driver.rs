//! Minimal dynamically loaded driver ABI. Symbols are resolved once; all owned
//! objects keep their driver/context alive. No CUDA headers or libraries are
//! required to build this crate or load hardware-independent tools.
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr},
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
    memset: unsafe extern "system" fn(u64,u8,usize)->ResultCode => "cuMemsetD8_v2",
    module_load: unsafe extern "system" fn(*mut Handle,*const c_void,c_uint,*mut c_int,*mut *mut c_void)->ResultCode => "cuModuleLoadDataEx",
    module_unload: unsafe extern "system" fn(Handle)->ResultCode => "cuModuleUnload",
    module_function: unsafe extern "system" fn(*mut Handle,Handle,*const c_char)->ResultCode => "cuModuleGetFunction",
    function_attribute: unsafe extern "system" fn(*mut c_int,c_int,Handle)->ResultCode => "cuFuncGetAttribute",
    launch: unsafe extern "system" fn(Handle,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,Handle,*mut *mut c_void,*mut *mut c_void)->ResultCode => "cuLaunchKernel",
    event_create: unsafe extern "system" fn(*mut Handle,c_uint)->ResultCode => "cuEventCreate",
    event_destroy: unsafe extern "system" fn(Handle)->ResultCode => "cuEventDestroy_v2",
    event_record: unsafe extern "system" fn(Handle,Handle)->ResultCode => "cuEventRecord",
    event_elapsed: unsafe extern "system" fn(*mut f32,Handle,Handle)->ResultCode => "cuEventElapsedTime",
    error_string: unsafe extern "system" fn(ResultCode,*mut *const c_char)->ResultCode => "cuGetErrorString",
}
impl Driver {
    pub fn check(&self, result: ResultCode, operation: &str) -> Result<(), String> {
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
        Err(format!("CUDA {operation}: {description} ({result})"))
    }
}
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
    pub fn upload(&self, bytes: &[u8]) -> Result<(), String> {
        self.upload_at(0, bytes)
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
    pub fn download(&self, bytes: &mut [u8]) -> Result<(), String> {
        self.download_at(0, bytes)
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
    pub fn fill(&self, byte: u8) -> Result<(), String> {
        let _current = self.context.enter()?;
        if self.bytes > 0 {
            unsafe {
                self.context.driver.check(
                    (self.context.driver.memset)(self.pointer, byte, self.bytes),
                    "memory fill",
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

/// Event ownership is tied to the private context. A recorded event is retired
/// only after the synchronous launch boundary has drained submitted work.
pub(crate) struct Event {
    pub raw: Handle,
    context: Rc<Context>,
}
impl Event {
    pub fn new(context: &Rc<Context>) -> Result<Self, String> {
        let _current = context.enter()?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            context
                .driver
                .check((context.driver.event_create)(&mut raw, 0), "event creation")?;
        }
        Ok(Self {
            raw,
            context: context.clone(),
        })
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.event_destroy)(self.raw);
            }
        }
    }
}
