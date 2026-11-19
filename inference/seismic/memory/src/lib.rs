//! Seismic's memory from the embedder's view: the ledger entries every
//! allocation is charged to ([`entry`]), each store's granule pool the
//! expert bank's records and the KV slabs live in ([`pool`]), and host
//! blocks ([`HostBlock`]): one plain allocation of the OS each, backed whole
//! as it is made and returned whole as it drops, in large pages where a
//! device reads it over its link; and program code ([`CodePages`]): pages
//! written once, then executable.

pub mod entry;
pub mod pool;

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Why a host allocation could not be made: the only real failure of a
/// host buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationFailure {
    /// The size and alignment do not form a valid host layout.
    Unrepresentable,
    /// The host allocator refused.
    OutOfMemory,
    /// The host did not report its page size, which blocks are whole
    /// numbers of.
    PageUnknown,
    /// The kernel refused to back the allocation in large pages it reports
    /// having.
    LargePagesRefused,
}

impl std::fmt::Display for AllocationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unrepresentable => {
                f.write_str("CPU allocation size is not representable on the host")
            }
            Self::OutOfMemory => f.write_str("CPU allocation failed"),
            Self::PageUnknown => f.write_str("the host page size could not be read"),
            Self::LargePagesRefused => f.write_str("the host refused large pages it reports having"),
        }
    }
}

/// The pages a [`HostBlock`] asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageSize {
    /// The host page: memory the host itself reads (the CPU backend's).
    Host,
    /// Large pages where the block is whole ones and the host has them:
    /// host memory a device reads over its link through an IOMMU (CUDA's
    /// registered host pages).
    Large,
}

/// One host block: a plain allocation of the OS (an anonymous mapping, a
/// `VirtualAlloc`), every page backed and zero before it is handed out and
/// none wired, returned to the OS whole and at once as it drops. Nothing of
/// it is reserved ahead or released piecewise: a block is the unit memory
/// leaves the process in.
///
/// A block of whole large pages (Linux's transparent huge page) asking for
/// them is backed by them, at an address aligned to one: a device reading
/// host memory over its link through an IOMMU that maps the pages it pins
/// one by one (DMA translation) pays a translation per small page it
/// touches. Measured on an RTX 3090 host (Xeon E5-2696 v3, VT-d in DMA-FQ
/// mode, PCIe gen 3), zero-copy reads and the copy engine alike: 4 KiB
/// pages copy at 10.7 GB/s and fall to 6.1 GB/s when 328 blocks read
/// 4 x 512 B a warp at once (1.0 GB/s in 32 B reads), where 2 MiB pages
/// copy and read at 12.4 GB/s in every pattern; on a 2990WX (AMD-Vi,
/// DMA-FQ) and an EPYC 7313 (identity) the two are equal.
pub struct HostBlock {
    base: NonNull<u8>,
    bytes: usize,
    /// Where the block is whole large pages: how many of them the kernel
    /// backed large, the rest small ([`large_page_backing`]).
    large: Option<LargeBacking>,
}

/// A large-page block's backing, counted into the process's
/// [`LargePageBacking`] while it lives.
#[derive(Clone, Copy)]
struct LargeBacking {
    page: usize,
    large: usize,
    small: usize,
}

// The block is plain memory owned by this struct; its bytes are accessed
// under the holder's discipline.
unsafe impl Send for HostBlock {}
unsafe impl Sync for HostBlock {}

/// The host's large page (Linux's transparent huge page, its PMD size),
/// decided once: none where the kernel has no transparent huge pages or
/// cannot collapse a range into them on demand (`MADV_COLLAPSE`, Linux
/// 6.1), whose backing would be left to its fault-time policy alone. A
/// probe that could not run is the error, decided again at the next call.
fn large_page() -> Result<Option<usize>, AllocationFailure> {
    static LARGE: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(&large) = LARGE.get() {
        return Ok(large);
    }
    let large = host::large_page()?;
    Ok(*LARGE.get_or_init(|| large))
}

/// The span of one last-level page table: the host page squared over a
/// page-table entry (2 MiB on 4 KiB pages, 32 MiB on 16 KiB, 512 MiB on
/// 64 KiB), which is the kernel's PMD size, its transparent huge page.
fn table_span(page: usize) -> usize {
    page * (page / std::mem::size_of::<u64>())
}

/// The unit a device registers host memory in (CUDA's registered host
/// blocks over a link): the large page, which a block of them is then
/// backed in. Where the host has none it is a size choice only, not a
/// measured one: the span of one last-level page table ([`table_span`]),
/// as many registrations as large pages would take; no host without large
/// pages was measured.
pub fn registration_granule() -> Result<u64, AllocationFailure> {
    let page = host::page()?;
    Ok(large_page()?.unwrap_or(table_span(page)) as u64)
}

/// Bytes of every live block that asks for large pages
/// ([`PageSize::Large`]: host memory devices read over their link) and is
/// whole ones, backed in large pages and in small ones: what the process's
/// large-page backing came to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LargePageBacking {
    pub large_bytes: u64,
    pub small_bytes: u64,
}

static LARGE_BYTES: AtomicU64 = AtomicU64::new(0);
static SMALL_BYTES: AtomicU64 = AtomicU64::new(0);

/// The process's [`LargePageBacking`] now.
pub fn large_page_backing() -> LargePageBacking {
    LargePageBacking {
        large_bytes: LARGE_BYTES.load(Ordering::Relaxed),
        small_bytes: SMALL_BYTES.load(Ordering::Relaxed),
    }
}

/// Whether a host read's `error` is its source missing (`ENOENT`,
/// `ENOTDIR`, `ENOSYS`, `EOPNOTSUPP`; Windows' file, path or function not
/// found or not supported) or refused (`EACCES`, `EPERM`, `EROFS`; Windows'
/// access denied, a privilege not held): either way the reader goes to the
/// fact's next source. The one rule every host reader shares; any other
/// error is no answer of the source's but an invariant violation.
pub fn unanswered(error: &std::io::Error) -> bool {
    use std::io::ErrorKind::{NotADirectory, NotFound, PermissionDenied, ReadOnlyFilesystem, Unsupported};
    if matches!(error.kind(), NotFound | PermissionDenied | ReadOnlyFilesystem | Unsupported | NotADirectory) {
        return true;
    }
    #[cfg(unix)]
    let codes = [libc::ENOSYS, libc::EOPNOTSUPP];
    #[cfg(windows)]
    let codes = [
        windows_sys::Win32::Foundation::ERROR_INVALID_FUNCTION as i32,
        windows_sys::Win32::Foundation::ERROR_NOT_SUPPORTED as i32,
        windows_sys::Win32::Foundation::ERROR_PRIVILEGE_NOT_HELD as i32,
    ];
    error.raw_os_error().is_some_and(|code| codes.contains(&code))
}

/// The host page, which every block is a whole number of.
pub fn host_page() -> Result<u64, AllocationFailure> {
    host::page().map(|page| page as u64)
}

impl HostBlock {
    /// A block of `bytes` (at least one) in whole host pages, in pages of
    /// `pages`, every page backed and zero, none wired.
    pub fn allocate(bytes: u64, pages: PageSize) -> Result<Self, AllocationFailure> {
        let page = host::page()?;
        let bytes = usize::try_from(bytes)
            .ok()
            .and_then(|bytes| bytes.max(1).checked_next_multiple_of(page))
            .ok_or(AllocationFailure::Unrepresentable)?;
        let large_page = large_page()?.filter(|&large| pages == PageSize::Large && bytes % large == 0);
        let base = host::map(bytes, large_page.unwrap_or(page))?;
        // Unmapped by its drop from here, whatever fails.
        let mut block = Self { base, bytes, large: None };
        #[cfg(target_os = "linux")]
        if large_page.is_some() {
            // Faulted in large pages where the kernel has them; collapsing
            // then makes what faulted small large.
            unsafe { host::prefer_large(base.as_ptr(), bytes)? };
        }
        fault_in(base, bytes, page);
        if let Some(large_page) = large_page {
            let large = host::collapse(base.as_ptr(), bytes, large_page);
            let backing = LargeBacking { page: large_page, large, small: bytes / large_page - large };
            LARGE_BYTES.fetch_add((backing.large * backing.page) as u64, Ordering::Relaxed);
            SMALL_BYTES.fetch_add((backing.small * backing.page) as u64, Ordering::Relaxed);
            block.large = Some(backing);
        }
        Ok(block)
    }

    pub fn base(&self) -> NonNull<u8> {
        self.base
    }

    /// Its bytes: whole pages, at least those asked for.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// How many of its large pages the kernel backed large, where it is
    /// whole large pages asked for as such (none otherwise).
    #[cfg(all(test, target_os = "linux"))]
    fn large_pages(&self) -> Option<usize> {
        self.large.map(|backing| backing.large)
    }
}

impl Drop for HostBlock {
    fn drop(&mut self) {
        if let Some(backing) = self.large {
            LARGE_BYTES.fetch_sub((backing.large * backing.page) as u64, Ordering::Relaxed);
            SMALL_BYTES.fetch_sub((backing.small * backing.page) as u64, Ordering::Relaxed);
        }
        unsafe { host::unmap(self.base, self.bytes) };
    }
}

/// Pages holding one program's code, still writable: its image is written
/// ([`CodePages::image`]), then the pages are sealed executable
/// ([`CodePages::seal`]) and never written again. Returned to the OS whole
/// as they drop. Nothing is backed until written.
pub struct CodePages {
    base: NonNull<u8>,
    bytes: usize,
}

/// Program code sealed executable: read-only, at a fixed address, until it
/// drops.
pub struct ExecutablePages(CodePages);

// The pages are plain memory owned by the value: written only through
// `&mut` while writable, immutable once executable.
unsafe impl Send for CodePages {}
unsafe impl Sync for CodePages {}

impl CodePages {
    /// Readable, writable zero pages covering `bytes` (at least one), at a
    /// host page.
    pub fn allocate(bytes: usize) -> Result<Self, AllocationFailure> {
        let page = host::page()?;
        let bytes = bytes.max(1).checked_next_multiple_of(page).ok_or(AllocationFailure::Unrepresentable)?;
        Ok(Self { base: host::map(bytes, page)?, bytes })
    }

    /// The address the code executes at.
    pub fn base(&self) -> NonNull<u8> {
        self.base
    }

    /// The writable image: whole pages, at least the bytes asked for.
    pub fn image(&mut self) -> &mut [u8] {
        // SAFETY: the pages are mapped writable and owned by this value.
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr(), self.bytes) }
    }

    /// Make the pages readable and executable, no longer writable. Where
    /// the host enforces branch-target landing pads (aarch64 BTI on Linux),
    /// `branch_targets` marks the pages guarded: their code has the pads.
    /// The caller has made instruction fetch agree with the written image.
    /// The OS's refusal is the error; the pages are then released.
    pub fn seal(self, branch_targets: bool) -> std::io::Result<ExecutablePages> {
        // SAFETY: the whole mapping, owned by this value.
        unsafe { host::seal_executable(self.base, self.bytes, branch_targets)? };
        Ok(ExecutablePages(self))
    }
}

impl ExecutablePages {
    /// The address the code executes at.
    pub fn base(&self) -> NonNull<u8> {
        self.0.base
    }
}

impl Drop for CodePages {
    fn drop(&mut self) {
        // SAFETY: every entry into executable pages is held by the owner of
        // the `ExecutablePages`, so none outlives the mapping.
        unsafe { host::unmap(self.base, self.bytes) };
    }
}

/// Back every page of a fresh block of `bytes`, whole pages of `page`
/// bytes starting at a page, at once: a write to the first byte of each
/// makes the kernel back it with a zero page, so from a page-aligned start
/// every page, the last included, is written. Faulted on every thread at
/// once, so the kernel's zeroing, and the reclaim and compaction a large
/// page's fault may run, go at memory speed rather than one core's. The
/// ledger counts a block from the moment it is made, so a block backed
/// later would read as other programs' memory falling and then rising.
fn fault_in(start: NonNull<u8>, bytes: usize, page: usize) {
    // The block's address, which every thread writes its own part of,
    // contiguous pages each.
    let base = start.as_ptr() as usize;
    assert!(
        base % page == 0 && bytes % page == 0,
        "invariant violated: a block of {bytes} bytes at {base:#x} is not whole {page}-byte pages"
    );
    parallel(bytes / page, |first, end| {
        for index in first..end {
            // SAFETY: within the fresh block, which this call alone touches,
            // each part on its own thread; zero is what it reads.
            unsafe { std::ptr::write_volatile((base + index * page) as *mut u8, 0) };
        }
    });
}

/// Run `work` over `[0, items)` in contiguous parts, one on each of the
/// process's threads (`seismic_threads::share`, at most one an item): on
/// the calling thread where that is one.
fn parallel(items: usize, work: impl Fn(usize, usize) + Sync) {
    let threads = seismic_threads::share().get().min(items).max(1);
    if threads == 1 {
        return work(0, items);
    }
    let part = items.div_ceil(threads);
    std::thread::scope(|scope| {
        for first in (0..items).step_by(part.max(1)) {
            let work = &work;
            seismic_threads::spawn_scoped(scope, "memory-split", move || work(first, (first + part).min(items))).expect("invariant violated: the host starts a thread");
        }
    });
}

/// Host virtual memory for blocks: one anonymous mapping each.
#[cfg(unix)]
mod host {
    use super::AllocationFailure;
    use std::ptr::NonNull;

    pub(super) fn page() -> Result<usize, AllocationFailure> {
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .map_err(|_| AllocationFailure::PageUnknown)
    }

    /// The kernel's transparent huge page size.
    #[cfg(target_os = "linux")]
    const THP_SIZE: &str = "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size";

    /// The transparent huge page size `read` of [`THP_SIZE`] gives on a
    /// host of `page`: the size it reads; where it is missing (a kernel
    /// without transparent huge pages, a sandbox's `/sys`) or refused, the
    /// PMD size the kernel's page tables fix ([`super::table_span`]).
    ///
    /// # Panics
    /// Where the size present does not parse, or the read fails otherwise.
    #[cfg(target_os = "linux")]
    pub(super) fn thp_size(read: std::io::Result<String>, page: usize) -> usize {
        match read {
            Ok(size) => size.trim().parse().unwrap_or_else(|error| panic!("invariant violated: {THP_SIZE} reads {size:?}: {error}")),
            Err(error) if super::unanswered(&error) => super::table_span(page),
            Err(error) => panic!("invariant violated: {THP_SIZE}: {error}"),
        }
    }

    /// The large page transparent huge pages back memory in ([`thp_size`]),
    /// where the kernel collapses a range into them on demand: probed on a
    /// block of its own. None where the kernel refuses the range large
    /// pages (`MADV_HUGEPAGE`: `EINVAL` from a kernel without transparent
    /// huge pages, or a filter refusing the call) or has no `MADV_COLLAPSE`
    /// (`EINVAL`, whatever the range); any other refusal of the collapse
    /// (want of memory now) is still a kernel that collapses.
    #[cfg(target_os = "linux")]
    pub(super) fn large_page() -> Result<Option<usize>, AllocationFailure> {
        let large = thp_size(std::fs::read_to_string(THP_SIZE), page()?);
        let base = map(large, large)?;
        let collapses = unsafe {
            let collapses = prefer_large(base.as_ptr(), large).is_ok() && {
                std::ptr::write_volatile(base.as_ptr(), 0);
                libc::madvise(base.as_ptr().cast(), large, libc::MADV_COLLAPSE) == 0
                    || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINVAL)
            };
            unmap(base, large);
            collapses
        };
        Ok(collapses.then_some(large))
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) fn large_page() -> Result<Option<usize>, AllocationFailure> {
        Ok(None)
    }

    /// Map `bytes` of readable, writable zero pages, nothing backed yet, at
    /// a multiple of `align` (a whole number of pages): the excess of a
    /// larger mapping is returned on both sides.
    pub(super) fn map(bytes: usize, align: usize) -> Result<NonNull<u8>, AllocationFailure> {
        let page = page()?;
        let span = bytes.checked_add(align - page).ok_or(AllocationFailure::Unrepresentable)?;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                span,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(AllocationFailure::OutOfMemory);
        }
        let lead = (align - base as usize % align) % align;
        unsafe {
            if lead > 0 {
                libc::munmap(base, lead);
            }
            if span - lead > bytes {
                libc::munmap(base.cast::<u8>().add(lead + bytes).cast(), span - lead - bytes);
            }
        }
        Ok(NonNull::new(unsafe { base.cast::<u8>().add(lead) }).expect("mmap never returns null on success"))
    }

    /// Have the mapped `[start, start + bytes)`, whole large pages, fault
    /// in them where the kernel has them ([`collapse`] then makes what
    /// faulted small large).
    #[cfg(target_os = "linux")]
    pub(super) unsafe fn prefer_large(start: *mut u8, bytes: usize) -> Result<(), AllocationFailure> {
        match unsafe { libc::madvise(start.cast(), bytes, libc::MADV_HUGEPAGE) } {
            0 => Ok(()),
            _ => Err(AllocationFailure::LargePagesRefused),
        }
    }

    /// Make the backed `[start, start + bytes)`, whole large pages of
    /// `large` bytes, large (`MADV_COLLAPSE`), each on its own thread's
    /// part: how many are. One call per page is the kernel's whole effort:
    /// its allocation reclaims and compacts directly, whatever the
    /// transparent huge page `defrag` setting, before it refuses (madvise(2),
    /// `MADV_COLLAPSE`). A page it refuses stays in host pages, which are
    /// memory all the same.
    #[cfg(target_os = "linux")]
    pub(super) fn collapse(start: *mut u8, bytes: usize, large: usize) -> usize {
        let base = start as usize;
        let collapsed = std::sync::atomic::AtomicUsize::new(0);
        super::parallel(bytes / large, |first, end| {
            for index in first..end {
                let at = (base + index * large) as *mut libc::c_void;
                if unsafe { libc::madvise(at, large, libc::MADV_COLLAPSE) } == 0 {
                    collapsed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
        collapsed.into_inner()
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) fn collapse(_start: *mut u8, _bytes: usize, _large: usize) -> usize {
        unreachable!("only Linux reports large pages")
    }

    /// Make the mapped `[base, base + bytes)` readable and executable,
    /// guarded where `branch_targets` asks and the host enforces it.
    pub(super) unsafe fn seal_executable(base: NonNull<u8>, bytes: usize, branch_targets: bool) -> std::io::Result<()> {
        let protection = libc::PROT_READ | libc::PROT_EXEC | guarded(branch_targets);
        match unsafe { libc::mprotect(base.as_ptr().cast(), bytes, protection) } {
            0 => Ok(()),
            _ => Err(std::io::Error::last_os_error()),
        }
    }

    /// The protection guarding branch targets (aarch64 BTI) of code that
    /// has landing pads, where the host enforces it.
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    fn guarded(branch_targets: bool) -> libc::c_int {
        const PROT_BTI: libc::c_int = 0x10;
        if branch_targets && std::arch::is_aarch64_feature_detected!("bti") { PROT_BTI } else { 0 }
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
    fn guarded(_: bool) -> libc::c_int {
        0
    }

    /// Unmap a whole block, returning its pages at once.
    pub(super) unsafe fn unmap(base: NonNull<u8>, bytes: usize) {
        unsafe { libc::munmap(base.as_ptr().cast(), bytes) };
    }
}

#[cfg(windows)]
mod host {
    use super::AllocationFailure;
    use std::ptr::NonNull;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        VirtualProtect,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

    /// Commit granularity: the system's page size.
    pub(super) fn page() -> Result<usize, AllocationFailure> {
        let mut info = std::mem::MaybeUninit::<SYSTEM_INFO>::uninit();
        // SAFETY: `GetSystemInfo` fills the whole structure and cannot fail.
        let info = unsafe {
            GetSystemInfo(info.as_mut_ptr());
            info.assume_init()
        };
        usize::try_from(info.dwPageSize).map_err(|_| AllocationFailure::PageUnknown)
    }

    /// Windows backs no block in large pages.
    pub(super) fn large_page() -> Result<Option<usize>, AllocationFailure> {
        Ok(None)
    }

    /// One committed allocation of `bytes`, reading zero, at the page:
    /// an allocation's own alignment covers it.
    pub(super) fn map(bytes: usize, align: usize) -> Result<NonNull<u8>, AllocationFailure> {
        assert_eq!(align, page()?, "a Windows block aligns to its page");
        let base = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        NonNull::new(base.cast()).ok_or(AllocationFailure::OutOfMemory)
    }

    pub(super) fn collapse(_start: *mut u8, _bytes: usize, _large: usize) -> usize {
        unreachable!("only Linux reports large pages")
    }

    /// Make the committed `[base, base + bytes)` readable and executable.
    /// Windows enforces no branch-target landing pads.
    pub(super) unsafe fn seal_executable(base: NonNull<u8>, bytes: usize, _branch_targets: bool) -> std::io::Result<()> {
        let mut previous = 0;
        match unsafe { VirtualProtect(base.as_ptr().cast(), bytes, PAGE_EXECUTE_READ, &mut previous) } {
            0 => Err(std::io::Error::last_os_error()),
            _ => Ok(()),
        }
    }

    /// Release a whole block, returning its pages at once.
    pub(super) unsafe fn unmap(base: NonNull<u8>, _bytes: usize) {
        unsafe { VirtualFree(base.as_ptr().cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// The address-space mapping `[start, end)` of `/proc/self/maps` that
    /// holds `address`.
    fn mapping(address: usize) -> (usize, usize) {
        std::fs::read_to_string("/proc/self/maps")
            .unwrap()
            .lines()
            .filter_map(|line| {
                let (start, end) = line.split_whitespace().next()?.split_once('-')?;
                Some((usize::from_str_radix(start, 16).ok()?, usize::from_str_radix(end, 16).ok()?))
            })
            .find(|&(start, end)| start <= address && address < end)
            .expect("a block's address is mapped")
    }

    /// Held by each test that reads the process's [`large_page_backing`],
    /// which every large-page block counts into: one at a time.
    fn counting() -> std::sync::MutexGuard<'static, ()> {
        static COUNTING: std::sync::Mutex<()> = std::sync::Mutex::new(());
        COUNTING.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A block of whole large pages lies at a large page, its excess
    /// returned on both sides (the mapping is exactly its pages), and each
    /// of its large pages is counted large or small, uncounted as it drops.
    #[test]
    fn large_page_blocks_are_aligned_trimmed_and_counted() {
        let _counting = counting();
        let Some(large) = large_page().unwrap() else {
            eprintln!("no large pages on this host");
            return;
        };
        let pages = 3;
        let before = large_page_backing();
        let block = HostBlock::allocate((pages * large) as u64, PageSize::Large).unwrap();
        let base = block.base().as_ptr() as usize;
        assert_eq!(base % large, 0, "aligned to a large page");
        assert_eq!(mapping(base), (base, base + pages * large), "the excess is returned on both sides");
        let after = large_page_backing();
        let counted = (after.large_bytes - before.large_bytes) + (after.small_bytes - before.small_bytes);
        assert_eq!(counted, (pages * large) as u64, "every large page is counted once");
        assert_eq!(block.large_pages().unwrap() as u64 * large as u64, after.large_bytes - before.large_bytes);
        // The pages read zero and take writes.
        unsafe {
            assert_eq!(*block.base().as_ptr().add(2 * large), 0);
            block.base().as_ptr().add(2 * large).write(7);
        }
        drop(block);
        assert_eq!(large_page_backing(), before, "a dropped block is uncounted");
    }

    /// The transparent huge page size is `hpage_pmd_size`'s; where it is
    /// missing or refused, the PMD size: the page squared over an 8-byte
    /// page-table entry.
    #[test]
    fn a_missing_thp_size_is_the_pmd_size() {
        use std::io::{Error, ErrorKind};
        assert_eq!(host::thp_size(Ok("2097152\n".into()), 4096), 2 << 20);
        assert_eq!(host::thp_size(Err(Error::from(ErrorKind::NotFound)), 4096), 2 << 20);
        assert_eq!(host::thp_size(Err(Error::from(ErrorKind::PermissionDenied)), 16384), 32 << 20);
        assert_eq!(host::thp_size(Err(Error::from(ErrorKind::NotFound)), 65536), 512 << 20);
    }

    /// A block that is not whole large pages, or that asks for host pages,
    /// lies at the host page and counts nothing.
    #[test]
    fn small_blocks_count_nothing() {
        let _counting = counting();
        let page = host::page().unwrap();
        let large = large_page().unwrap().unwrap_or(page);
        for (bytes, pages) in [(page, PageSize::Large), (large, PageSize::Host)] {
            let before = large_page_backing();
            let block = HostBlock::allocate((4 * bytes) as u64, pages).unwrap();
            assert!(block.large_pages().is_none());
            assert_eq!(large_page_backing(), before);
        }
    }
}
