//! Seismic's memory from the embedder's view: the ledger entries every
//! allocation is charged to ([`entry`]), each store's granule pool the
//! expert bank's records and the KV slabs live in ([`pool`]), and host
//! blocks ([`HostBlock`]): one plain allocation of the OS each, backed whole
//! as it is made and returned whole as it drops, in large pages where a
//! device reads it over its link; and program code ([`CodePages`]): pages
//! written once, then executable.

pub mod entry;
pub mod placement;
pub mod pool;
pub use pool::SlabFill;

/// Alignment of the base of every workspace buffer placed in a graph arena.
pub const BUFFER_ALIGNMENT: u64 = 256;

use std::cell::Cell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr::NonNull;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

thread_local! {
    /// Nesting depth of no-blocking-allocation scopes on this thread.
    static NO_BLOCKING_ALLOC_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// RAII guard marking a scope on the current thread where blocking memory
/// allocations, frees, driver locks, and pool locks are forbidden.
pub struct NoBlockingAllocationGuard {
    _not_send: PhantomData<*const ()>,
}

/// Enter a no-blocking-allocation scope on this thread.
pub fn enter_no_blocking_allocation() -> NoBlockingAllocationGuard {
    NO_BLOCKING_ALLOC_DEPTH.with(|depth| depth.set(depth.get() + 1));
    NoBlockingAllocationGuard { _not_send: PhantomData }
}

impl Drop for NoBlockingAllocationGuard {
    fn drop(&mut self) {
        NO_BLOCKING_ALLOC_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "NoBlockingAllocationGuard depth underflow");
            depth.set(current - 1);
        });
    }
}

/// Assert that the current thread is not inside a no-blocking-allocation
/// scope.
#[track_caller]
pub fn assert_blocking_allocation_allowed(operation: &str) {
    let depth = NO_BLOCKING_ALLOC_DEPTH.with(Cell::get);
    assert!(
        depth == 0,
        "invariant violated: {operation} called inside a no-blocking-allocation scope"
    );
}

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
    /// Where the block is whole large pages: the large page size it is tracked
    /// under in [`live_large_blocks`] ([`large_page_backing`]).
    large_page: Option<usize>,
}

// The block is plain memory owned by this struct; its bytes are accessed
// under the holder's discipline.
unsafe impl Send for HostBlock {}
unsafe impl Sync for HostBlock {}

/// The host's large page (Linux's transparent huge page, its PMD size),
/// decided once: none where the kernel has no transparent huge pages or
/// refuses `MADV_HUGEPAGE`. A probe that could not run is the error, decided
/// again at the next call.
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

/// Live `PageSize::Large` whole-large-page blocks: `base -> (bytes, large_page)`.
/// Maintained in O(1) on [`HostBlock::allocate`] and [`HostBlock::drop`] so
/// allocation never walks `/proc/self/smaps`; [`large_page_backing`] measures
/// all live blocks in one single `/proc/self/smaps` pass on demand.
fn live_large_blocks() -> &'static Mutex<HashMap<usize, (usize, usize)>> {
    static LIVE: OnceLock<Mutex<HashMap<usize, (usize, usize)>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The process's [`LargePageBacking`] now, measured lazily in a single
/// `/proc/self/smaps` pass across all live large-page blocks.
pub fn large_page_backing() -> LargePageBacking {
    let mut ranges: Vec<(usize, usize, usize)> = {
        let live = live_large_blocks().lock().unwrap_or_else(PoisonError::into_inner);
        live.iter().map(|(&base, &(bytes, large))| (base, bytes, large)).collect()
    };
    if ranges.is_empty() {
        return LargePageBacking::default();
    }
    ranges.sort_unstable_by_key(|&(base, _, _)| base);
    host::measure_large_page_backing(&ranges)
}

/// Latched `true` on Linux once a `MADV_HUGEPAGE` block under a blocking
/// `/sys/kernel/mm/transparent_hugepage/defrag` policy hits synchronous kernel
/// compaction during `fault_in`, so subsequent blocks skip `MADV_HUGEPAGE` and
/// fault at full RAM speed.
#[cfg(target_os = "linux")]
static COMPACTION_STALLED: AtomicBool = AtomicBool::new(false);

/// Below this `fault_in` throughput (500 MB/s), `MADV_HUGEPAGE` under a
/// blocking `thp/defrag` policy has hit synchronous kernel compaction.
#[cfg(target_os = "linux")]
const MIN_THP_FAULT_BYTES_PER_SEC: u128 = 500_000_000;

#[cfg(target_os = "linux")]
fn fault_stalled(bytes: usize, elapsed: std::time::Duration) -> bool {
    (bytes as u128).saturating_mul(1_000_000_000)
        < MIN_THP_FAULT_BYTES_PER_SEC.saturating_mul(elapsed.as_nanos())
}

/// Whether a host read's `error` is its source missing (`ENOENT`,
/// `ENOTDIR`, `ENOSYS`, `EOPNOTSUPP`; Windows' file, path or function not
/// found or not supported) or refused (`EACCES`, `EPERM`, `EROFS`; Windows'
/// access denied, a privilege not held): either way the reader goes to the
/// reading's next source. The one rule every host reader shares; any other
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
        let mut block = Self { base, bytes, large_page: None };
        #[cfg(target_os = "linux")]
        let advised_blocking = if large_page.is_some() {
            let blocking = host::thp_defrag().blocks_on_madvise();
            if !blocking || !COMPACTION_STALLED.load(Ordering::Relaxed) {
                // Faulted in large pages opportunistically when free; under a
                // blocking `thp/defrag` policy, once synchronous kernel
                // compaction stalls a block's `fault_in`, subsequent blocks
                // skip `MADV_HUGEPAGE` to fault at full RAM speed.
                unsafe { host::prefer_large(base.as_ptr(), bytes)? };
                blocking
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(target_os = "linux")]
        let fault_start = advised_blocking.then(std::time::Instant::now);
        fault_in(base, bytes, page);
        #[cfg(target_os = "linux")]
        if let Some(start) = fault_start {
            if fault_stalled(bytes, start.elapsed()) {
                COMPACTION_STALLED.store(true, Ordering::Relaxed);
            }
        }
        if let Some(large_page) = large_page {
            let mut live = live_large_blocks().lock().unwrap_or_else(PoisonError::into_inner);
            let prev = live.insert(base.as_ptr() as usize, (bytes, large_page));
            debug_assert!(prev.is_none(), "live HostBlock address already registered");
            block.large_page = Some(large_page);
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
        let large = self.large_page?;
        let backing = host::measure_large_page_backing(&[(self.base.as_ptr() as usize, self.bytes, large)]);
        Some((backing.large_bytes / large as u64) as usize)
    }
}

impl Drop for HostBlock {
    fn drop(&mut self) {
        if self.large_page.is_some() {
            // Remove before unmapping so the OS cannot hand the same virtual
            // address to another `HostBlock::allocate` while we still hold the
            // entry.
            let mut live = live_large_blocks().lock().unwrap_or_else(PoisonError::into_inner);
            live.remove(&(self.base.as_ptr() as usize));
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
    parallel(bytes / page, bytes, |first, end| {
        for index in first..end {
            // SAFETY: within the fresh block, which this call alone touches,
            // each part on its own thread; zero is what it reads.
            unsafe { std::ptr::write_volatile((base + index * page) as *mut u8, 0) };
        }
    });
}

/// Minimum bytes per thread when faulting a block in parallel:
/// blocks under twice this size (4 MiB) run inline on the calling thread with
/// zero thread-spawn overhead, and larger blocks scale at most one thread per
/// 2 MiB up to `seismic_threads::share()`.
const MIN_PARALLEL_BYTES_PER_THREAD: usize = 2 << 20;

/// Run `work` over `[0, items)` covering `bytes` in contiguous parts across at
/// most `seismic_threads::share()` threads (at least
/// [`MIN_PARALLEL_BYTES_PER_THREAD`] per thread, and at most one thread per
/// item): inline on the calling thread when one thread suffices, with the
/// calling thread running the first slice and falling back inline for any
/// remaining slices if the host refuses to start a thread.
fn parallel(items: usize, bytes: usize, work: impl Fn(usize, usize) + Sync) {
    let threads = seismic_threads::share()
        .get()
        .min(items)
        .min(bytes / MIN_PARALLEL_BYTES_PER_THREAD)
        .max(1);
    if threads == 1 {
        return work(0, items);
    }
    let part = items.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut first = part;
        while first < items {
            let end = (first + part).min(items);
            let work = &work;
            if seismic_threads::spawn_scoped(scope, "memory-split", move || work(first, end)).is_err() {
                work(first, items);
                break;
            }
            first = end;
        }
        work(0, part);
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

    /// The kernel's transparent huge page defragmentation policy.
    #[cfg(target_os = "linux")]
    const THP_DEFRAG: &str = "/sys/kernel/mm/transparent_hugepage/defrag";

    /// Active token parsed from `/sys/kernel/mm/transparent_hugepage/defrag`.
    #[cfg(target_os = "linux")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum ThpDefrag {
        Never,
        Defer,
        Madvise,
        DeferMadvise,
        Always,
        Unknown,
    }

    #[cfg(target_os = "linux")]
    impl ThpDefrag {
        /// Whether `madvise(MADV_HUGEPAGE)` may trigger synchronous direct
        /// reclaim / compaction on page fault under this policy.
        pub(super) fn blocks_on_madvise(self) -> bool {
            !matches!(self, Self::Never | Self::Defer)
        }
    }

    /// Parse the active `[token]` from `/sys/kernel/mm/transparent_hugepage/defrag`.
    #[cfg(target_os = "linux")]
    pub(super) fn parse_thp_defrag(content: &str) -> ThpDefrag {
        let active = content
            .split_whitespace()
            .find_map(|tok| tok.strip_prefix('[').and_then(|s| s.strip_suffix(']')));
        match active {
            Some("never") => ThpDefrag::Never,
            Some("defer") => ThpDefrag::Defer,
            Some("madvise") => ThpDefrag::Madvise,
            Some("defer+madvise") => ThpDefrag::DeferMadvise,
            Some("always") => ThpDefrag::Always,
            _ => ThpDefrag::Unknown,
        }
    }

    /// Read `/sys/kernel/mm/transparent_hugepage/defrag` once per process.
    #[cfg(target_os = "linux")]
    pub(super) fn thp_defrag() -> ThpDefrag {
        static DEFRAG: std::sync::OnceLock<ThpDefrag> = std::sync::OnceLock::new();
        *DEFRAG.get_or_init(|| match std::fs::read_to_string(THP_DEFRAG) {
            Ok(content) => parse_thp_defrag(&content),
            Err(_) => ThpDefrag::Unknown,
        })
    }

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
    /// where the kernel accepts `MADV_HUGEPAGE`: probed on a block of its
    /// own. None where the kernel refuses the range large pages (`EINVAL`
    /// from a kernel without transparent huge pages, or a filter refusing
    /// the call).
    #[cfg(target_os = "linux")]
    pub(super) fn large_page() -> Result<Option<usize>, AllocationFailure> {
        let large = thp_size(std::fs::read_to_string(THP_SIZE), page()?);
        let base = map(large, large)?;
        let supported = unsafe {
            let supported = prefer_large(base.as_ptr(), large).is_ok();
            unmap(base, large);
            supported
        };
        Ok(supported.then_some(large))
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
    /// in them where the kernel has them (`MADV_HUGEPAGE`).
    #[cfg(target_os = "linux")]
    pub(super) unsafe fn prefer_large(start: *mut u8, bytes: usize) -> Result<(), AllocationFailure> {
        match unsafe { libc::madvise(start.cast(), bytes, libc::MADV_HUGEPAGE) } {
            0 => Ok(()),
            _ => Err(AllocationFailure::LargePagesRefused),
        }
    }

    /// Measure large-page and small-page backing for `ranges` — sorted,
    /// non-overlapping `(base, bytes, large_page)` entries — in a single pass
    /// over `/proc/self/smaps`. If `/proc/self/smaps` cannot be opened, every
    /// tracked range is counted as small-page backed.
    #[cfg(target_os = "linux")]
    pub(super) fn measure_large_page_backing(ranges: &[(usize, usize, usize)]) -> super::LargePageBacking {
        if ranges.is_empty() {
            return super::LargePageBacking::default();
        }
        let Ok(file) = std::fs::File::open("/proc/self/smaps") else {
            let small_bytes = ranges.iter().map(|&(_, bytes, _)| bytes as u64).sum();
            return super::LargePageBacking { large_bytes: 0, small_bytes };
        };
        parse_smaps_large_pages_batch(std::io::BufReader::new(file), ranges)
    }

    /// Parse `AnonHugePages:` entries from `smaps` for all sorted,
    /// non-overlapping `ranges` (`(base, bytes, large_page)`) in one pass and
    /// return the aggregate [`super::LargePageBacking`].
    #[cfg(target_os = "linux")]
    pub(super) fn parse_smaps_large_pages_batch(
        mut reader: impl std::io::BufRead,
        ranges: &[(usize, usize, usize)],
    ) -> super::LargePageBacking {
        use std::io::BufRead;
        if ranges.is_empty() {
            return super::LargePageBacking::default();
        }
        let max_end = ranges
            .iter()
            .map(|&(base, bytes, _)| base.saturating_add(bytes))
            .max()
            .unwrap_or(0);
        let mut anon_huge_bytes = vec![0usize; ranges.len()];
        let mut cursor = 0usize;
        let mut current_vma: Option<(usize, usize)> = None;
        let mut line = String::new();
        while let Ok(read) = reader.read_line(&mut line) {
            if read == 0 {
                break;
            }
            if let Some(first) = line.split_whitespace().next() {
                if !first.ends_with(':') {
                    current_vma = None;
                    if let Some((start_hex, end_hex)) = first.split_once('-') {
                        if let (Ok(vma_start), Ok(vma_end)) = (
                            usize::from_str_radix(start_hex, 16),
                            usize::from_str_radix(end_hex, 16),
                        ) {
                            if vma_start >= max_end {
                                break;
                            }
                            while cursor < ranges.len()
                                && ranges[cursor].0.saturating_add(ranges[cursor].1) <= vma_start
                            {
                                cursor += 1;
                            }
                            if cursor < ranges.len() && ranges[cursor].0 < vma_end {
                                current_vma = Some((vma_start, vma_end));
                            }
                        }
                    }
                } else if let Some((vma_start, vma_end)) = current_vma {
                    if let Some(rest) = line.strip_prefix("AnonHugePages:") {
                        if let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<usize>().ok()) {
                            let huge_bytes = kb.saturating_mul(1024);
                            let vma_len = vma_end.saturating_sub(vma_start);
                            for (&(base, bytes, _), huge_acc) in
                                ranges[cursor..].iter().zip(&mut anon_huge_bytes[cursor..])
                            {
                                if base >= vma_end {
                                    break;
                                }
                                let end = base.saturating_add(bytes);
                                if vma_start >= base && vma_end <= end {
                                    *huge_acc = huge_acc.saturating_add(huge_bytes);
                                } else if vma_len > 0 {
                                    let overlap = vma_end.min(end).saturating_sub(vma_start.max(base));
                                    let share =
                                        ((huge_bytes as u128 * overlap as u128) / vma_len as u128) as usize;
                                    *huge_acc = huge_acc.saturating_add(share);
                                }
                            }
                        }
                    }
                }
            }
            line.clear();
        }
        let mut large_bytes = 0u64;
        let mut small_bytes = 0u64;
        for (&(_, bytes, large), &huge) in ranges.iter().zip(&anon_huge_bytes) {
            let total_pages = bytes / large;
            let large_pages = (huge / large).min(total_pages);
            let small_pages = total_pages - large_pages;
            large_bytes += (large_pages * large) as u64;
            small_bytes += (small_pages * large) as u64;
        }
        super::LargePageBacking { large_bytes, small_bytes }
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn measure_large_page_backing(_ranges: &[(usize, usize, usize)]) -> super::LargePageBacking {
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

    pub(super) fn measure_large_page_backing(_ranges: &[(usize, usize, usize)]) -> super::LargePageBacking {
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

    #[test]
    fn smaps_parser_counts_exact_split_and_merged_vmas() {
        let large = 2usize << 20;
        let smaps = "\
10000000-10600000 rw-p 00000000 00:00 0
Size:               6144 kB
AnonHugePages:      4096 kB
VmFlags: rd wr mr mw me ac hg sd
20000000-20200000 rw-p 00000000 00:00 0
Size:               2048 kB
AnonHugePages:      2048 kB
20200000-20400000 rw-p 00000000 00:00 0
Size:               2048 kB
AnonHugePages:         0 kB
30000000-30800000 rw-p 00000000 00:00 0
Size:               8192 kB
AnonHugePages:      8192 kB
";
        let l = large as u64;
        // Exact single VMA with 2 of 3 huge pages backed.
        assert_eq!(
            host::parse_smaps_large_pages_batch(smaps.as_bytes(), &[(0x1000_0000, 3 * large, large)]),
            LargePageBacking { large_bytes: 2 * l, small_bytes: l },
        );
        // Two split VMAs covering a 2-large-page block: 1 huge + 0 huge.
        assert_eq!(
            host::parse_smaps_large_pages_batch(smaps.as_bytes(), &[(0x2000_0000, 2 * large, large)]),
            LargePageBacking { large_bytes: l, small_bytes: l },
        );
        // Merged VMA twice as large as the block: proportionally attributed and never exceeds block's pages.
        assert_eq!(
            host::parse_smaps_large_pages_batch(smaps.as_bytes(), &[(0x3000_0000, 2 * large, large)]),
            LargePageBacking { large_bytes: 2 * l, small_bytes: 0 },
        );
        // Unmapped address range counts all pages as small.
        assert_eq!(
            host::parse_smaps_large_pages_batch(smaps.as_bytes(), &[(0x4000_0000, 2 * large, large)]),
            LargePageBacking { large_bytes: 0, small_bytes: 2 * l },
        );
        // Single-pass batch across all ranges, including two adjacent blocks sharing a merged VMA.
        assert_eq!(
            host::parse_smaps_large_pages_batch(
                smaps.as_bytes(),
                &[
                    (0x1000_0000, 3 * large, large),
                    (0x2000_0000, 2 * large, large),
                    (0x3000_0000, 2 * large, large),
                    (0x3040_0000, 2 * large, large),
                    (0x4000_0000, 2 * large, large),
                ],
            ),
            LargePageBacking { large_bytes: 7 * l, small_bytes: 4 * l },
        );
    }

    #[test]
    fn large_page_backing_tracks_multiple_live_blocks_and_drops_to_zero() {
        let _counting = counting();
        let Some(large) = large_page().unwrap() else {
            eprintln!("no large pages on this host");
            return;
        };
        let before = large_page_backing();
        let a = HostBlock::allocate((2 * large) as u64, PageSize::Large).unwrap();
        let b = HostBlock::allocate((3 * large) as u64, PageSize::Large).unwrap();
        let both = large_page_backing();
        let counted_both = (both.large_bytes - before.large_bytes) + (both.small_bytes - before.small_bytes);
        assert_eq!(counted_both, (5 * large) as u64, "both live blocks counted in one smaps pass");
        drop(a);
        let only_b = large_page_backing();
        let counted_b = (only_b.large_bytes - before.large_bytes) + (only_b.small_bytes - before.small_bytes);
        assert_eq!(counted_b, (3 * large) as u64, "dropping first block removes only its bytes");
        drop(b);
        assert_eq!(large_page_backing(), before, "dropping all blocks returns backing to baseline");
    }

    #[test]
    fn thp_defrag_parser_recognizes_active_token_and_blocking_policy() {
        use host::{ThpDefrag, parse_thp_defrag};
        assert_eq!(parse_thp_defrag("[madvise]"), ThpDefrag::Madvise);
        assert!(parse_thp_defrag("[madvise]").blocks_on_madvise());

        assert_eq!(parse_thp_defrag("always defer defer+madvise [madvise] never\n"), ThpDefrag::Madvise);
        assert!(parse_thp_defrag("always defer defer+madvise [madvise] never\n").blocks_on_madvise());

        assert_eq!(parse_thp_defrag("[never]"), ThpDefrag::Never);
        assert!(!parse_thp_defrag("[never]").blocks_on_madvise());

        assert_eq!(parse_thp_defrag("[defer]"), ThpDefrag::Defer);
        assert!(!parse_thp_defrag("[defer]").blocks_on_madvise());

        assert_eq!(parse_thp_defrag("always defer [defer+madvise] madvise never"), ThpDefrag::DeferMadvise);
        assert!(parse_thp_defrag("always defer [defer+madvise] madvise never").blocks_on_madvise());

        assert_eq!(parse_thp_defrag("[always] defer defer+madvise madvise never"), ThpDefrag::Always);
        assert!(parse_thp_defrag("[always] defer defer+madvise madvise never").blocks_on_madvise());

        assert_eq!(parse_thp_defrag(""), ThpDefrag::Unknown);
        assert!(parse_thp_defrag("").blocks_on_madvise());

        // Fault throughput stall threshold: 500 MB/s.
        let block = 64usize << 20; // 67_108_864 bytes -> 500 MB/s threshold is ~134.2 ms
        assert!(!fault_stalled(block, std::time::Duration::from_millis(16))); // ~4 GB/s
        assert!(!fault_stalled(block, std::time::Duration::ZERO));
        assert!(fault_stalled(block, std::time::Duration::from_millis(200))); // ~335 MB/s
    }
}
