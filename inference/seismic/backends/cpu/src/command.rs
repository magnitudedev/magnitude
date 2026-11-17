//! One CPU native kernel and the executable memory of its program.
//! Schedule and emission layout remain core-owned.

use crate::toolchain::CpuProgram;
use crate::workers::LaunchEntry;
use seismic_ir::physical_target::KernelEmissionLayout;
use seismic_native_target::Formed;

/// Executable memory holding one program's image, whose bytes every core
/// fetches as last written.
///
/// A program's code is freed with its last kernel, and the system hands the
/// same addresses to the next program loaded. On aarch64 instruction
/// caches are not coherent with data writes, and changing a page's
/// protection does not invalidate a page that already held code: a core
/// that ran the earlier program would then execute its stale instructions
/// over the new program's frame. The image is invalidated once its final
/// bytes are written, before it becomes executable; each participant
/// entering a launch then discards instructions it fetched before
/// ([`crate::workers`]).
pub(crate) struct CoherentCode {
    pages: seismic_memory::ExecutablePages,
}

impl CoherentCode {
    /// `bytes` bytes of executable memory, written by `write` (given the
    /// writable image and the address it executes at) before they become
    /// executable. `branch_targets` marks the pages guarded (aarch64 BTI)
    /// where the host enforces it.
    pub(crate) fn new(
        bytes: usize,
        branch_targets: bool,
        write: impl FnOnce(&mut [u8], usize),
    ) -> Self {
        assert!(bytes > 0, "a program image holds code");
        let mut pages = seismic_memory::CodePages::allocate(bytes).unwrap_or_else(|failure| {
            panic!("mapping {bytes} bytes for program code failed: {failure}")
        });
        let base = pages.base().as_ptr();
        write(&mut pages.image()[..bytes], base as usize);
        invalidate_instructions(base, bytes);
        let pages = pages.seal(branch_targets).unwrap_or_else(|error| {
            panic!("making {bytes} bytes of program code executable failed: {error}")
        });
        Self { pages }
    }

    /// The address the image executes at.
    pub(crate) fn base(&self) -> *const u8 {
        self.pages.base().as_ptr()
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

pub struct CompiledKernel {
    pub(crate) entry: LaunchEntry,
    /// Keeps the program `entry` points into alive.
    pub(crate) _program: Formed<CpuProgram>,
    pub(crate) layout: KernelEmissionLayout,
}

impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("words", &self.layout.words.total)
            .finish()
    }
}
