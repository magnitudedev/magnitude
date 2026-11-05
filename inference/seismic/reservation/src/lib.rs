//! A reservation of granules, its leading ones backed: the one resize
//! between commitments every backend shares ([`Resizer`]). What backs a
//! granule, and what device work binds over the backed ones, is its holder's
//! [`Backing`]: for the CPU backend and Metal a host address range reserved
//! once ([`HostRange`]), whose address never changes and whose released
//! granules' pages are freed at once; for Vulkan device memory allocated
//! chunk by chunk; for CUDA granules mapped into one address range.

use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// Why a host allocation could not be made: the only real failure of a
/// host buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationFailure {
    /// The size and alignment do not form a valid host layout.
    Unrepresentable,
    /// The host allocator refused.
    OutOfMemory,
    /// The host did not report its page size, in which reservations are
    /// made.
    PageUnknown,
    /// The kernel refused to wire the allocation's pages.
    #[cfg(target_os = "macos")]
    WireRefused,
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
            #[cfg(target_os = "macos")]
            Self::WireRefused => f.write_str("CPU allocation could not be wired"),
        }
    }
}

/// What backs a reservation's granules, and what device work binds over the
/// backed ones.
pub trait Backing: Send + Sync + 'static {
    type Error: Send;
    /// What device work binds over the backed granules, changed in place
    /// where they change.
    type View: Send + Sync;
    /// A change of the view from one granule count to another, formed
    /// beside device work that binds the view.
    type Change: Send + Sync;
    /// What applying a change leaves the view no longer holding, kept until
    /// no device work submitted before the change uses it.
    type Retired: Send + Sync;
    /// Bytes of one granule.
    fn granule(&self) -> usize;
    /// The granules the reservation spans.
    fn granules(&self) -> usize;
    /// Back granules `[from, to)`, zero, every byte present: a backed
    /// granule is charged, so it is held. On failure nothing new stays
    /// backed.
    fn back(&self, from: usize, to: usize) -> Result<(), Self::Error>;
    /// Return granules `[from, to)` (none where `from >= to`) to unbacked,
    /// freeing what backed them at once. No device work uses them.
    fn unback(&self, from: usize, to: usize);
    /// Write zero over `bytes` of backed granules, beside device work that
    /// uses the reservation but not those bytes.
    fn zero(&self, bytes: Range<u64>) -> Result<(), Self::Error>;
    /// The view over the first `granules` granules, when the reservation is
    /// formed.
    fn view(&self, granules: usize) -> Result<Self::View, Self::Error>;
    /// The change of the view from its `installed` granules to the first
    /// `granules` (another count), formed without changing what device work
    /// or the host see. Every granule the view or the change spans is
    /// backed.
    fn change(&self, installed: usize, granules: usize) -> Result<Self::Change, Self::Error>;
    /// Apply `change` to `view`: device work encoded from now binds the
    /// result. What it retires is dropped by the change's release, before
    /// the granules it releases are unbacked.
    fn apply(&self, view: &mut Self::View, change: Self::Change) -> Self::Retired;
}

/// A reserved host address range, unmapped when dropped: what the CPU
/// backend's, Metal's and the discrete devices' host storage back granules
/// of.
///
/// A granule that is a whole number of the host's large pages (Linux's
/// transparent huge page) is backed by them, at a range aligned to one: a
/// device reading host memory over its link through an IOMMU that maps
/// the pages it pins one by one (DMA translation) pays a translation per
/// small page it touches. Measured on an RTX 3090 host (Xeon E5-2696 v3, VT-d in
/// DMA-FQ mode, PCIe gen 3), zero-copy reads and the copy engine alike:
/// 4 KiB pages copy at 10.7 GB/s and fall to 6.1 GB/s when 328 blocks read
/// 4 x 512 B a warp at once (1.0 GB/s in 32 B reads), where 2 MiB pages
/// copy and read at 12.4 GB/s in every pattern; on a 2990WX (AMD-Vi,
/// DMA-FQ) and an EPYC 7313 (identity) the two are equal.
pub struct HostRange {
    base: NonNull<u8>,
    /// The host page, which a granule is a whole number of.
    page: usize,
    granule: usize,
    granules: usize,
    /// Where the granules are whole large pages: each one's backing
    /// (unbacked, or backed in large or in small pages).
    large: Option<Box<[AtomicU8]>>,
}

/// The pages a [`HostRange`] asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageSize {
    /// The host page: memory the host itself reads (the CPU backend's,
    /// Metal's).
    Host,
    /// Large pages where its granules are whole ones and the host has them:
    /// host memory a device reads over its link through an IOMMU (CUDA's
    /// registered host pages, Vulkan's imported ones).
    Large,
}

/// What backs one granule of a range whose granules are whole large pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum GranulePages {
    Unbacked = 0,
    /// Host pages: the kernel had no contiguous memory to make them large,
    /// once retried.
    Small = 1,
    Large = 2,
}

/// The host's large page (Linux's transparent huge page, its PMD size),
/// decided once: none where the kernel has no transparent huge pages or
/// cannot collapse a range into them on demand (`MADV_COLLAPSE`, Linux
/// 6.1), whose backing would be left to its fault-time policy alone. A
/// probe that could not run is the error, decided again at the next call.
pub fn large_page() -> Result<Option<usize>, AllocationFailure> {
    static LARGE: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(&large) = LARGE.get() {
        return Ok(large);
    }
    let large = host::large_page()?;
    Ok(*LARGE.get_or_init(|| large))
}

/// The granule a device registers host memory in (`seismic_cuda`'s host
/// reservations over a link): the large page, which its granules are then
/// backed in. Where the host has none it is a size choice only, not a
/// measured one: the span of one last-level page table (host page squared
/// over a page-table entry, 2 MiB on 4 KiB pages), as many registrations
/// as large pages would take; no host without large pages was measured.
pub fn registration_granule() -> Result<u64, AllocationFailure> {
    let page = host::page()?;
    Ok(large_page()?.unwrap_or(page * (page / std::mem::size_of::<u64>())) as u64)
}

/// Bytes of granules backed now of every range that asks for large pages
/// ([`PageSize::Large`]: host memory devices read over their link) and has
/// them (its granules whole large pages), in large pages and in small ones:
/// what the process's large-page backing came to ([`large_page_backing`]).
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

/// The time the process has spent backing reserved storage, by phase
/// ([`time_backing`]), each a wall time on the thread that backed it:
/// faulting host pages in, collapsing them into large pages, making them
/// the device's (CUDA's registration, Vulkan's import and binding with its
/// records' table entries), and zeroing bytes the backing did not give
/// zeroed. What a span of backing cost is the difference of two readings
/// ([`Self::since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackingTimes {
    pub fault: Duration,
    pub collapse: Duration,
    pub pin: Duration,
    pub zero: Duration,
}

impl BackingTimes {
    /// What was spent since `earlier`, an earlier reading.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            fault: self.fault.saturating_sub(earlier.fault),
            collapse: self.collapse.saturating_sub(earlier.collapse),
            pin: self.pin.saturating_sub(earlier.pin),
            zero: self.zero.saturating_sub(earlier.zero),
        }
    }
}

impl std::ops::AddAssign for BackingTimes {
    fn add_assign(&mut self, more: Self) {
        self.fault += more.fault;
        self.collapse += more.collapse;
        self.pin += more.pin;
        self.zero += more.zero;
    }
}

/// A phase of backing reserved storage ([`BackingTimes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackingPhase {
    Fault,
    Collapse,
    Pin,
    Zero,
}

static BACKING_NANOS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Do `work`, a phase of backing reserved storage, counting its wall time
/// into the process's [`BackingTimes`].
pub fn time_backing<T>(phase: BackingPhase, work: impl FnOnce() -> T) -> T {
    let began = std::time::Instant::now();
    let done = work();
    let nanos = u64::try_from(began.elapsed().as_nanos()).expect("a phase's nanoseconds fit 64 bits");
    BACKING_NANOS[phase as usize].fetch_add(nanos, Ordering::Relaxed);
    done
}

/// The process's [`BackingTimes`] now.
pub fn backing_times() -> BackingTimes {
    let read = |phase: BackingPhase| Duration::from_nanos(BACKING_NANOS[phase as usize].load(Ordering::Relaxed));
    BackingTimes {
        fault: read(BackingPhase::Fault),
        collapse: read(BackingPhase::Collapse),
        pin: read(BackingPhase::Pin),
        zero: read(BackingPhase::Zero),
    }
}

// The range is plain address space owned by this struct; its bytes are
// accessed under the holder's discipline.
unsafe impl Send for HostRange {}
unsafe impl Sync for HostRange {}

impl HostRange {
    /// `reserved` bytes of inaccessible address space (at least one
    /// granule) in `granule`-byte granules, a whole number of host pages,
    /// nothing backed, in pages of `pages`.
    pub fn reserve(reserved: u64, granule: u64, pages: PageSize) -> Result<Self, AllocationFailure> {
        let page = host::page()?;
        let granule = usize::try_from(granule)
            .ok()
            .filter(|&granule| granule != 0 && granule % page == 0)
            .ok_or(AllocationFailure::Unrepresentable)?;
        let granules = usize::try_from(reserved)
            .ok()
            .map(|bytes| bytes.div_ceil(granule).max(1))
            .filter(|granules| granules.checked_mul(granule).is_some())
            .ok_or(AllocationFailure::Unrepresentable)?;
        let large = large_page()?.filter(|&large| pages == PageSize::Large && granule % large == 0);
        Ok(Self {
            base: host::reserve(granules * granule, large.unwrap_or(page))?,
            page,
            granule,
            granules,
            large: large.map(|_| (0..granules).map(|_| AtomicU8::new(GranulePages::Unbacked as u8)).collect()),
        })
    }

    pub fn base(&self) -> NonNull<u8> {
        self.base
    }

    /// Bytes of one host page, which a granule is a whole number of.
    pub fn page(&self) -> usize {
        self.page
    }

    /// Bytes of one granule.
    pub fn granule(&self) -> usize {
        self.granule
    }

    /// The granules the range spans.
    pub fn granules(&self) -> usize {
        self.granules
    }

    /// Write zero over `bytes` of backed granules.
    pub fn zero(&self, bytes: Range<u64>) {
        assert!(
            bytes.end <= (self.granules * self.granule) as u64,
            "zeroed bytes {bytes:?} exceed the range"
        );
        unsafe {
            std::ptr::write_bytes(
                self.base.as_ptr().add(bytes.start as usize),
                0,
                (bytes.end - bytes.start) as usize,
            )
        };
    }

    /// The address of granule `index`.
    pub fn at(&self, index: usize) -> *mut u8 {
        assert!(index <= self.granules, "granule {index} of {}", self.granules);
        unsafe { self.base.as_ptr().add(index * self.granule) }
    }

    /// Back granules `[from, to)` with accessible zero pages, held: large
    /// pages where the granules are whole ones, granule by granule, and
    /// where the kernel has no contiguous memory for one (its collapse
    /// refused twice) host pages, which are memory all the same; each
    /// outcome is counted ([`large_page_backing`], [`Self::large_granules`]).
    pub fn back(&self, from: usize, to: usize) -> Result<(), AllocationFailure> {
        if from < to {
            assert!(to <= self.granules, "granule {to} of {}", self.granules);
            let bytes = (to - from) * self.granule;
            // Granules of whole large pages fault in them where the kernel
            // has them; collapsing then makes what faulted small large.
            #[cfg(target_os = "linux")]
            if self.large.is_some() {
                unsafe { host::prefer_large(self.at(from), bytes)? };
            }
            time_backing(BackingPhase::Fault, || unsafe { host::commit(self.at(from), bytes) })?;
            #[cfg(target_os = "linux")]
            if let Some(large) = &self.large {
                // Each granule collapsed on its own thread's part, as its
                // pages were faulted.
                time_backing(BackingPhase::Collapse, || host::parallel(to - from, |first, end| {
                    for index in from + first..from + end {
                        let pages = match unsafe { host::collapse(self.at(index), self.granule) } {
                            true => GranulePages::Large,
                            false => GranulePages::Small,
                        };
                        let counter = if pages == GranulePages::Large { &LARGE_BYTES } else { &SMALL_BYTES };
                        counter.fetch_add(self.granule as u64, Ordering::Relaxed);
                        large[index].store(pages as u8, Ordering::Relaxed);
                    }
                }));
            }
        }
        Ok(())
    }

    /// How many backed granules are large pages, where the granules are
    /// whole ones (none otherwise).
    pub fn large_granules(&self) -> Option<usize> {
        self.large
            .as_ref()
            .map(|large| large.iter().filter(|pages| pages.load(Ordering::Relaxed) == GranulePages::Large as u8).count())
    }

    /// Forget granules `[from, to)`'s backing in the counts.
    fn uncount(&self, from: usize, to: usize) {
        if let Some(large) = &self.large {
            for pages in &large[from..to] {
                match pages.swap(GranulePages::Unbacked as u8, Ordering::Relaxed) {
                    x if x == GranulePages::Large as u8 => LARGE_BYTES.fetch_sub(self.granule as u64, Ordering::Relaxed),
                    x if x == GranulePages::Small as u8 => SMALL_BYTES.fetch_sub(self.granule as u64, Ordering::Relaxed),
                    _ => 0,
                };
            }
        }
    }

    /// Return granules `[from, to)` to inaccessible, unbacked address space,
    /// freeing their pages at once.
    pub fn unback(&self, from: usize, to: usize) {
        if from < to {
            assert!(to <= self.granules, "granule {to} of {}", self.granules);
            self.uncount(from, to);
            unsafe { host::decommit(self.at(from), (to - from) * self.granule) };
        }
    }
}

impl Drop for HostRange {
    fn drop(&mut self) {
        self.uncount(0, self.granules);
        unsafe { host::release(self.base, self.granules * self.granule) };
    }
}

/// A reserved range of granules, the leading ones backed, and the view
/// device work binds over them.
pub struct Reserved<B: Backing> {
    // Dropped first: the view (Metal's buffers) leaves before what backs
    // it.
    state: Mutex<State<B::View>>,
    backing: B,
}

/// What a reservation backs now.
pub struct State<V> {
    /// Committed bytes, the buffer's length.
    pub bytes: u64,
    /// The granules the view spans, every one backed. Beyond them only a
    /// prepared growth or an installed shrink backs granules, until it is
    /// abandoned or released.
    granules: usize,
    pub view: V,
}

impl<B: Backing> Reserved<B> {
    /// The reservation `backing` spans, its leading `committed` bytes
    /// backed, and its resizer.
    ///
    /// # Panics
    /// If `committed` exceeds the reservation.
    pub fn new(backing: B, committed: u64) -> Result<(Arc<Self>, Resizer<B>), B::Error> {
        let backed = granules_of(&backing, committed);
        backing.back(0, backed)?;
        let view = match backing.view(backed) {
            Ok(view) => view,
            Err(error) => {
                backing.unback(0, backed);
                return Err(error);
            }
        };
        let reserved = Arc::new(Self {
            state: Mutex::new(State {
                bytes: committed,
                granules: backed,
                view,
            }),
            backing,
        });
        Ok((reserved.clone(), Resizer { reserved }))
    }

    pub fn state(&self) -> MutexGuard<'_, State<B::View>> {
        self.state.lock().expect("reservation lock poisoned")
    }

    pub fn backing(&self) -> &B {
        &self.backing
    }
}

/// The granules of `backing` covering its leading `bytes`.
///
/// # Panics
/// If `bytes` exceed the reservation.
fn granules_of<B: Backing>(backing: &B, bytes: u64) -> usize {
    usize::try_from(bytes)
        .ok()
        .map(|bytes| bytes.div_ceil(backing.granule()))
        .filter(|&granules| granules <= backing.granules())
        .unwrap_or_else(|| panic!("{bytes} bytes exceed the reservation"))
}

/// The right to resize one reservation: its holder holds it
/// ([`Reserved::new`]), a resize consumes it, and the resize's release or
/// abandonment returns it, so its resizes follow one another.
pub struct Resizer<B: Backing> {
    reserved: Arc<Reserved<B>>,
}

impl<B: Backing> Resizer<B> {
    /// Prepare backing exactly the granules covering the leading `committed`
    /// bytes, without changing what device work or the host see, so it runs
    /// beside device work that uses the reservation: a growth's granules are
    /// backed (zero), bytes a shrink left in a kept granule that the
    /// commitment makes visible again are zeroed, and where the granules
    /// change the view's change is formed. A refusal returns the resizer.
    ///
    /// # Panics
    /// If `committed` exceeds the reservation.
    pub fn prepare(self, committed: u64) -> Result<Prepared<B>, (Self, B::Error)> {
        let backing = &self.reserved.backing;
        let granules = granules_of(backing, committed);
        let (bytes, installed) = {
            let state = self.reserved.state();
            (state.bytes, state.granules)
        };
        if granules > installed {
            if let Err(error) = backing.back(installed, granules) {
                return Err((self, error));
            }
        }
        // Bytes a shrink left inside a kept granule still hold old contents.
        let stale = bytes.min(committed)..committed.min((installed * backing.granule()) as u64);
        let zeroed = match stale.is_empty() {
            true => Ok(()),
            false => time_backing(BackingPhase::Zero, || backing.zero(stale)),
        };
        let change = zeroed.and_then(|()| match granules == installed {
            true => Ok(None),
            false => backing.change(installed, granules).map(Some),
        });
        let change = match change {
            Ok(change) => change,
            Err(error) => {
                backing.unback(installed, granules);
                return Err((self, error));
            }
        };
        Ok(Prepared {
            next: Some(Next {
                reserved: self.reserved,
                bytes: committed,
                granules,
                installed,
                change,
            }),
        })
    }
}

/// A reservation's next commitment, prepared beside device work that uses
/// it ([`Resizer::prepare`]): its bytes and, where its granules change, the
/// view's change. Dropped, it is abandoned, its resizer with it.
pub struct Prepared<B: Backing> {
    next: Option<Next<B>>,
}

struct Next<B: Backing> {
    reserved: Arc<Reserved<B>>,
    bytes: u64,
    granules: usize,
    /// The view's granules when it was prepared.
    installed: usize,
    change: Option<B::Change>,
}

impl<B: Backing> Prepared<B> {
    /// Publish the commitment: its length, and the view's change, which
    /// device work encoded from now binds. A shrink's granules and what the
    /// change retires stay until the returned release, so device work
    /// submitted before keeps them, and dropping what it retires (on Metal,
    /// a buffer's removal from the residency set and the set's commit; an
    /// install that dropped it held its thread 10–50 ms under memory
    /// pressure, M4 Pro, 2026-10-26) is the release's, not the publisher's.
    /// Precondition: no host access or encoding uses the reservation
    /// meanwhile.
    pub fn install(mut self) -> Pending<B> {
        let next = self.next.take().expect("a prepared commitment installs once");
        let reserved = &next.reserved;
        let mut state = reserved.state();
        state.bytes = next.bytes;
        state.granules = next.granules;
        let retired = next
            .change
            .map(|change| reserved.backing.apply(&mut state.view, change));
        drop(state);
        Pending {
            released: Some((next.reserved, next.granules, next.granules.max(next.installed))),
            retired,
        }
    }

    /// Abandon the commitment: its change is dropped and the granules a
    /// growth backed are released. It returns the resizer.
    pub fn abandon(mut self) -> Resizer<B> {
        Self::unprepare(self.next.take().expect("a prepared commitment is abandoned once"))
    }

    fn unprepare(next: Next<B>) -> Resizer<B> {
        drop(next.change);
        next.reserved.backing.unback(next.installed, next.granules);
        Resizer {
            reserved: next.reserved,
        }
    }
}

impl<B: Backing> Drop for Prepared<B> {
    fn drop(&mut self) {
        if let Some(next) = self.next.take() {
            drop(Self::unprepare(next));
        }
    }
}

/// An installed commitment's release ([`Prepared::install`]): the granules a
/// shrink left backed beyond it. Dropped, it is released, its resizer with
/// it.
pub struct Pending<B: Backing> {
    /// The reservation, its view's granules and the granules backed.
    released: Option<(Arc<Reserved<B>>, usize, usize)>,
    /// What the install's change retired, where the granules changed.
    retired: Option<B::Retired>,
}

impl<B: Backing> Pending<B> {
    /// Drop what the install retired, then release the granules beyond the
    /// view's, freeing their pages. Precondition: no device work uses them.
    /// It returns the resizer.
    pub fn release(mut self) -> Resizer<B> {
        drop(self.retired.take());
        Self::unback(self.released.take().expect("a commitment is released once"))
    }

    fn unback((reserved, kept, backed): (Arc<Reserved<B>>, usize, usize)) -> Resizer<B> {
        reserved.backing.unback(kept, backed);
        Resizer { reserved }
    }
}

impl<B: Backing> Drop for Pending<B> {
    fn drop(&mut self) {
        drop(self.retired.take());
        if let Some(released) = self.released.take() {
            drop(Self::unback(released));
        }
    }
}

/// The CPU backend's backing: host pages, wired on macOS, read in place.
pub struct Pages(pub HostRange);

impl Backing for Pages {
    type Error = AllocationFailure;
    type View = ();
    type Change = ();
    type Retired = ();

    fn granule(&self) -> usize {
        self.0.granule
    }

    fn granules(&self) -> usize {
        self.0.granules
    }

    fn back(&self, from: usize, to: usize) -> Result<(), AllocationFailure> {
        self.0.back(from, to)
    }

    fn unback(&self, from: usize, to: usize) {
        self.0.unback(from, to);
    }

    fn zero(&self, bytes: Range<u64>) -> Result<(), AllocationFailure> {
        self.0.zero(bytes);
        Ok(())
    }

    fn view(&self, _: usize) -> Result<(), AllocationFailure> {
        Ok(())
    }

    fn change(&self, _: usize, _: usize) -> Result<(), AllocationFailure> {
        Ok(())
    }

    fn apply(&self, _: &mut (), _: ()) {}
}

/// The host page, which a [`HostRange`]'s granules are a whole number of.
pub fn host_page() -> Result<u64, AllocationFailure> {
    host::page().map(|page| page as u64)
}

/// Host virtual memory for reservations. Every range lies inside one
/// reservation and is a whole number of pages.
#[cfg(unix)]
mod host {
    use super::AllocationFailure;
    use std::ptr::NonNull;

    /// Address space reserved but not committed is not charged against
    /// overcommit.
    #[cfg(target_os = "linux")]
    const RESERVE_FLAGS: libc::c_int = libc::MAP_NORESERVE;
    #[cfg(not(target_os = "linux"))]
    const RESERVE_FLAGS: libc::c_int = 0;

    pub(super) fn page() -> Result<usize, AllocationFailure> {
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .map_err(|_| AllocationFailure::PageUnknown)
    }

    /// The large page transparent huge pages back memory in, where the
    /// kernel has them and collapses a range into them on demand: probed
    /// on a range of its own. None where the kernel has no transparent
    /// huge pages (no `hpage_pmd_size`) or no `MADV_COLLAPSE` (refused as
    /// invalid, whatever the range); a refusal for want of memory now is
    /// still a kernel that collapses.
    #[cfg(target_os = "linux")]
    pub(super) fn large_page() -> Result<Option<usize>, AllocationFailure> {
        const SIZE: &str = "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size";
        let large: usize = match std::fs::read_to_string(SIZE) {
            Ok(size) => size.trim().parse().unwrap_or_else(|error| panic!("{SIZE} reads {size:?}: {error}")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => panic!("reading {SIZE}: {error}"),
        };
        let base = reserve(large, large)?;
        let collapses = unsafe {
            let probed = prefer_large(base.as_ptr(), large).and_then(|()| commit(base.as_ptr(), large)).map(|()| {
                libc::madvise(base.as_ptr().cast(), large, libc::MADV_COLLAPSE) == 0
                    || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINVAL)
            });
            release(base, large);
            probed?
        };
        Ok(collapses.then_some(large))
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) fn large_page() -> Result<Option<usize>, AllocationFailure> {
        Ok(None)
    }

    /// Reserve `bytes` of inaccessible address space with nothing backed,
    /// at a multiple of `align` (a whole number of pages): the excess of a
    /// larger reservation is returned on both sides.
    pub(super) fn reserve(bytes: usize, align: usize) -> Result<NonNull<u8>, AllocationFailure> {
        let page = page()?;
        let span = bytes.checked_add(align - page).ok_or(AllocationFailure::Unrepresentable)?;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                span,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | RESERVE_FLAGS,
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

    /// Have the reserved `[start, start + bytes)`, whole large pages, fault
    /// in them where the kernel has them, once committed ([`collapse`] then
    /// makes what faulted small large).
    #[cfg(target_os = "linux")]
    pub(super) unsafe fn prefer_large(start: *mut u8, bytes: usize) -> Result<(), AllocationFailure> {
        match unsafe { libc::madvise(start.cast(), bytes, libc::MADV_HUGEPAGE) } {
            0 => Ok(()),
            _ => Err(AllocationFailure::LargePagesRefused),
        }
    }

    /// Make `[start, start + bytes)` accessible zero pages, backed (wired on
    /// macOS): what is committed is charged, so it is held.
    pub(super) unsafe fn commit(start: *mut u8, bytes: usize) -> Result<(), AllocationFailure> {
        #[cfg(not(target_os = "macos"))]
        let page = page()?;
        let protection = libc::PROT_READ | libc::PROT_WRITE;
        if unsafe { libc::mprotect(start.cast(), bytes, protection) } != 0 {
            return Err(AllocationFailure::OutOfMemory);
        }
        #[cfg(target_os = "macos")]
        if unsafe { libc::mlock(start.cast(), bytes) } != 0 {
            unsafe { decommit(start, bytes) };
            return Err(AllocationFailure::WireRefused);
        }
        // A write to each page makes the kernel back it with a zero page:
        // faulted on every thread at once (`parallel`), so the kernel's
        // zeroing, and the reclaim and compaction a large page's fault may
        // run, go at memory speed rather than one core's.
        #[cfg(not(target_os = "macos"))]
        {
            // The range's address, which every thread writes its own part of,
            // contiguous pages each.
            let base = start as usize;
            parallel(bytes.div_ceil(page), |first, end| {
                let (from, to) = (first * page, (end * page).min(bytes));
                for offset in (from..to).step_by(page) {
                    // SAFETY: within the range just made accessible, which
                    // this call alone touches, each part on its own thread.
                    unsafe { std::ptr::write_volatile((base + offset) as *mut u8, 0) };
                }
            });
        }
        Ok(())
    }

    /// Run `work` over `[0, items)` in contiguous parts, one on each of the
    /// host's threads (`available_parallelism`, at most one an item).
    #[cfg(not(target_os = "macos"))]
    pub(super) fn parallel(items: usize, work: impl Fn(usize, usize) + Sync) {
        let threads = std::thread::available_parallelism().map_or(1, |threads| threads.get()).min(items).max(1);
        let part = items.div_ceil(threads);
        std::thread::scope(|scope| {
            for first in (0..items).step_by(part.max(1)) {
                let work = &work;
                scope.spawn(move || work(first, (first + part).min(items)));
            }
        });
    }

    /// Make the committed `[start, start + bytes)`, whole large pages, large
    /// (`MADV_COLLAPSE`, which compacts for them): whether it is. A refusal
    /// for want of contiguous memory is retried once, after the kernel's
    /// own compaction has had the first attempt's reclaim.
    #[cfg(target_os = "linux")]
    pub(super) unsafe fn collapse(start: *mut u8, bytes: usize) -> bool {
        (0..2).any(|_| unsafe { libc::madvise(start.cast(), bytes, libc::MADV_COLLAPSE) } == 0)
    }

    /// Return `[start, start + bytes)` to inaccessible, unbacked address
    /// space. Replacing the range with a fresh mapping frees its pages at
    /// once on Linux and macOS alike (Metal's granules too, whatever maps
    /// them), and a later commit reads zero. (`MADV_FREE_REUSABLE` only
    /// marks pages reclaimable while they stay present, and needs
    /// `MADV_FREE_REUSE` and zeroing to back them again.)
    pub(super) unsafe fn decommit(start: *mut u8, bytes: usize) {
        let pointer = unsafe {
            libc::mmap(
                start.cast(),
                bytes,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED | RESERVE_FLAGS,
                -1,
                0,
            )
        };
        // Replacing a mapping inside the process's own reservation fails
        // only on exhausted kernel resources; a half-released range cannot
        // be reported while keeping the old prefix valid.
        assert!(
            pointer != libc::MAP_FAILED,
            "releasing reserved pages failed: {}",
            std::io::Error::last_os_error()
        );
    }

    /// Unmap a whole reservation, which also unwires it.
    pub(super) unsafe fn release(base: NonNull<u8>, bytes: usize) {
        unsafe { libc::munmap(base.as_ptr().cast(), bytes) };
    }
}

#[cfg(windows)]
mod host {
    use super::AllocationFailure;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_DECOMMIT: u32 = 0x4000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(address: *mut c_void, size: usize, kind: u32, protect: u32)
            -> *mut c_void;
        fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
        fn GetSystemInfo(info: *mut SystemInfo);
    }

    /// `SYSTEM_INFO`, of which only the page size is read.
    #[repr(C)]
    struct SystemInfo {
        _oem_id: u32,
        page_size: u32,
        _minimum_application_address: *mut c_void,
        _maximum_application_address: *mut c_void,
        _active_processor_mask: usize,
        _number_of_processors: u32,
        _processor_type: u32,
        _allocation_granularity: u32,
        _processor_level: u16,
        _processor_revision: u16,
    }

    /// Commit granularity: the system's page size.
    pub(super) fn page() -> Result<usize, AllocationFailure> {
        let mut info = std::mem::MaybeUninit::<SystemInfo>::uninit();
        // SAFETY: `GetSystemInfo` fills the whole structure and cannot fail.
        let info = unsafe {
            GetSystemInfo(info.as_mut_ptr());
            info.assume_init()
        };
        usize::try_from(info.page_size).map_err(|_| AllocationFailure::PageUnknown)
    }

    /// Windows backs no reservation in large pages.
    pub(super) fn large_page() -> Result<Option<usize>, AllocationFailure> {
        Ok(None)
    }

    /// At `align`, the page: an allocation's own alignment covers it.
    pub(super) fn reserve(bytes: usize, align: usize) -> Result<NonNull<u8>, AllocationFailure> {
        assert_eq!(align, page()?, "a Windows reservation aligns to its page");
        let base =
            unsafe { VirtualAlloc(std::ptr::null_mut(), bytes, MEM_RESERVE, PAGE_NOACCESS) };
        NonNull::new(base.cast()).ok_or(AllocationFailure::OutOfMemory)
    }

    /// Committed pages read zero.
    pub(super) unsafe fn commit(start: *mut u8, bytes: usize) -> Result<(), AllocationFailure> {
        let committed = unsafe { VirtualAlloc(start.cast(), bytes, MEM_COMMIT, PAGE_READWRITE) };
        if committed.is_null() {
            return Err(AllocationFailure::OutOfMemory);
        }
        Ok(())
    }

    /// Decommitting frees the pages at once; a later commit reads zero.
    pub(super) unsafe fn decommit(start: *mut u8, bytes: usize) {
        let freed = unsafe { VirtualFree(start.cast(), bytes, MEM_DECOMMIT) };
        assert!(
            freed != 0,
            "releasing reserved pages failed: {}",
            std::io::Error::last_os_error()
        );
    }

    pub(super) unsafe fn release(base: NonNull<u8>, _bytes: usize) {
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
            .expect("a reserved address is mapped")
    }

    /// Held by each test that reads the process's [`large_page_backing`],
    /// which every backed range counts into: one at a time.
    fn counting() -> std::sync::MutexGuard<'static, ()> {
        static COUNTING: std::sync::Mutex<()> = std::sync::Mutex::new(());
        COUNTING.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A range of whole large-page granules is reserved at a large page,
    /// its excess returned on both sides (the mapping is exactly its
    /// granules), and each backed granule is counted large or small,
    /// uncounted when unbacked or dropped.
    #[test]
    fn large_page_granules_are_aligned_trimmed_and_counted() {
        let _counting = counting();
        let Some(large) = large_page().unwrap() else {
            eprintln!("no large pages on this host");
            return;
        };
        let granules = 5;
        let range = HostRange::reserve((granules * large) as u64, large as u64, PageSize::Large).unwrap();
        let base = range.base().as_ptr() as usize;
        assert_eq!(base % large, 0, "aligned to a large page");
        assert_eq!(mapping(base), (base, base + granules * large), "the excess is returned on both sides");
        let before = large_page_backing();
        range.back(0, 3).unwrap();
        let after = large_page_backing();
        let counted = (after.large_bytes - before.large_bytes) + (after.small_bytes - before.small_bytes);
        assert_eq!(counted, (3 * large) as u64, "every backed granule is counted once");
        assert_eq!(range.large_granules().unwrap() as u64 * large as u64, after.large_bytes - before.large_bytes);
        // The pages read zero and take writes.
        unsafe {
            assert_eq!(*range.at(2), 0);
            range.at(2).write(7);
        }
        range.unback(1, 3);
        assert!(range.large_granules().unwrap() <= 1, "unbacked granules are uncounted");
        drop(range);
        assert_eq!(large_page_backing(), before, "a dropped range is uncounted");
    }

    /// A granule that is not whole large pages, or a range that asks for
    /// host pages, is reserved at the host page and counts nothing.
    #[test]
    fn small_granules_count_nothing() {
        let _counting = counting();
        let page = host::page().unwrap();
        let large = large_page().unwrap().unwrap_or(page);
        for (granule, pages) in [(page, PageSize::Large), (large, PageSize::Host)] {
            let range = HostRange::reserve((4 * granule) as u64, granule as u64, pages).unwrap();
            assert!(range.large_granules().is_none());
            let before = large_page_backing();
            range.back(0, 4).unwrap();
            assert_eq!(large_page_backing(), before);
        }
    }
}
