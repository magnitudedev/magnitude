//! One CPU native kernel. Schedule and emission layout remain core-owned.

use crate::workers::LaunchEntry;
use cranelift_jit::{BranchProtection, JITMemoryProvider, JITModule, SystemMemoryProvider};
use cranelift_module::ModuleResult;
use seismic_ir::physical_target::KernelEmissionLayout;
use std::sync::Arc;

/// JIT memory whose executable bytes every core fetches as last written.
///
/// A variant's code is freed with its last kernel, and the allocator hands
/// the same addresses to the next variant compiled. On aarch64 instruction
/// caches are not coherent with data writes, and Cranelift's system provider
/// leaves their maintenance to `mprotect` (its cache clear is a no-op), which
/// does not invalidate a page that already held code: a core that ran the
/// earlier variant then executes its stale instructions over the new
/// variant's frame. This provider invalidates every executable range once
/// its final bytes are written, before it becomes executable; each
/// participant entering a launch then discards instructions it fetched
/// before ([`crate::workers`]).
pub(crate) struct CoherentCode {
    memory: SystemMemoryProvider,
    /// Executable ranges written since the last finalization, as
    /// (address, bytes).
    written: Vec<(usize, usize)>,
}

impl CoherentCode {
    pub(crate) fn new() -> Self {
        Self {
            memory: SystemMemoryProvider::new(),
            written: Vec::new(),
        }
    }
}

impl JITMemoryProvider for CoherentCode {
    fn allocate_readexec(&mut self, size: usize, align: u64) -> std::io::Result<*mut u8> {
        let address = self.memory.allocate_readexec(size, align)?;
        self.written.push((address as usize, size));
        Ok(address)
    }

    fn allocate_readwrite(&mut self, size: usize, align: u64) -> std::io::Result<*mut u8> {
        self.memory.allocate_readwrite(size, align)
    }

    fn allocate_readonly(&mut self, size: usize, align: u64) -> std::io::Result<*mut u8> {
        self.memory.allocate_readonly(size, align)
    }

    unsafe fn free_memory(&mut self) {
        self.written.clear();
        unsafe { self.memory.free_memory() };
    }

    /// Cranelift finalizes after writing and relocating every definition.
    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        for (address, bytes) in self.written.drain(..) {
            invalidate_instructions(address as *const u8, bytes);
        }
        self.memory.finalize(branch_protection)
    }
}

/// Makes the instruction caches of every core agree with the data written
/// to `bytes` bytes at `address`.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn invalidate_instructions(address: *const u8, bytes: usize) {
    unsafe extern "C" {
        fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
    }
    // SAFETY: cache maintenance over memory this provider allocated.
    unsafe { sys_icache_invalidate(address.cast_mut().cast(), bytes) };
}

/// Makes the instruction caches of every core agree with the data written
/// to `bytes` bytes at `address`: clean the data lines to the point of
/// unification, then invalidate the instruction lines, both broadcast to
/// the inner-shareable domain.
#[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
fn invalidate_instructions(address: *const u8, bytes: usize) {
    let start = address as usize;
    let end = start + bytes;
    let ctr: u64;
    // SAFETY: CTR_EL0 is readable at EL0, and cache maintenance by address
    // is permitted at EL0 over this process's mapped memory.
    unsafe {
        std::arch::asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr);
        // IDC: data cleaning to the point of unification is not required.
        if ctr >> 28 & 1 == 0 {
            let line = 4usize << (ctr >> 16 & 0xf);
            for address in (start & !(line - 1)..end).step_by(line) {
                std::arch::asm!("dc cvau, {address}", address = in(reg) address);
            }
        }
        std::arch::asm!("dsb ish");
        // DIC: instruction invalidation to the point of unification is not
        // required.
        if ctr >> 29 & 1 == 0 {
            let line = 4usize << (ctr & 0xf);
            for address in (start & !(line - 1)..end).step_by(line) {
                std::arch::asm!("ic ivau, {address}", address = in(reg) address);
            }
            std::arch::asm!("dsb ish");
        }
    }
}

/// x86 instruction fetch is coherent with data writes.
#[cfg(target_arch = "x86_64")]
fn invalidate_instructions(_: *const u8, _: usize) {}

/// The JIT memory of one variant, freed when its last kernel drops.
pub(crate) struct JitMemory {
    module: Option<JITModule>,
}

// Finalized code memory is immutable and freed once after the last kernel.
unsafe impl Send for JitMemory {}
unsafe impl Sync for JitMemory {}

impl JitMemory {
    pub(crate) fn new(module: JITModule) -> Self {
        Self {
            module: Some(module),
        }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // Every function pointer into this memory is retained by the same
            // CompiledKernel Arc, so none can outlive the allocation.
            unsafe { module.free_memory() };
        }
    }
}

pub struct CompiledKernel {
    pub(crate) entry: LaunchEntry,
    /// Drop guard: keeps the JIT code `entry` points into alive.
    pub(crate) _memory: Arc<JitMemory>,
    pub(crate) layout: KernelEmissionLayout,
}

impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("words", &self.layout.words.total)
            .finish()
    }
}
