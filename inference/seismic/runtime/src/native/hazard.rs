//! The order a submission's launches need, derived from what each launch
//! reaches (`docs/seismic/execution/binding-and-workflows.md`, "Workflow
//! composition").
//!
//! A launch reaches byte ranges of allocations ([`Reach`]): each bound
//! buffer's view, and through a slab view's address table each backed slab
//! region, reading or writing them as the kernel's ABI declares. It depends
//! on an earlier launch when one of them writes bytes the other reaches
//! (read after write, write after read, write after write).
//!
//! A host port (`&host tensor`, `&host mut tensor`) is the host's access:
//! once it sees the launch, the host reads the tensor or rewrites it. The
//! device never writes a host port, but a launch whose host rewrites one is
//! ordered as its writer ([`abi::host_writes`]): it follows every earlier
//! launch that reaches those bytes, and every later one that reads them
//! follows it. A protocol the host serves while work runs is ordered by
//! these declarations alone.
//!
//! [`abi::host_writes`]: super::abi::host_writes
//!
//! A launch depends on exactly the earlier launches it conflicts with
//! ([`Order`]). [`schedule`] runs each launch at its dependency level:
//! every launch of a level completes before any launch of a later level
//! starts, and the launches of one level are independent. A launch's level
//! is one past the highest level of an earlier launch it depends on, so a
//! launch may run before earlier launches it does not depend on, never
//! before one it does. CUDA's graphs keep stream order. Timeline operations
//! bound the reordering: no launch crosses one.
//!
//! A chained list ([`DispatchList::chained`]) runs each call after the one
//! before it, whatever they reach: standalone calls, which time a tuning
//! point's rotation of a model's layers as the served step runs them, each
//! after its predecessor. A rotation of independent argument sets run at
//! one level overlaps its calls, and ranks configurations by a cost the
//! step does not pay: on an M4 Pro one GEMV phase read 14.09 us four at a
//! time and 26.65 us in order, its form's fixed cost hidden by the overlap
//! (2026-11-07). Vulkan's direct launches each follow a barrier, and CUDA's
//! stream keeps order, so only Metal's levels overlapped them.

use super::Dispatch;
use super::DispatchList;
use crate::api::tensor::Records;
use crate::driver::Allocation;
use std::collections::HashMap;
use std::ops::Range;

/// One step of encoding a submission ([`schedule`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scheduled {
    /// The timeline operations at a boundary (before dispatch `.0`, or after
    /// the last).
    Timelines(usize),
    /// The next `.0` launches form a level.
    Level(usize),
    /// Launch `ordinal` of dispatch `index`.
    Launch { index: usize, ordinal: usize },
}

/// `list`'s launches repeated `repetitions` times, ordered for encoding:
/// between timeline operations, the launches by level, each level in
/// submission order. Launches that do no device work (inactive, or an empty
/// grid) are left out.
pub(crate) fn schedule(list: &impl DispatchList, repetitions: usize) -> Vec<Scheduled> {
    let mut schedule = Vec::new();
    let mut order = Order::default();
    // The open segment's launches: (level, index, ordinal), in submission
    // order.
    let mut segment = Vec::new();
    let mut buffers = Vec::new();
    let mut reaches = Vec::new();
    let mut accesses = Vec::new();
    let chained = list.chained();
    // A chained list's launches of the call before, which each launch of
    // the next follows.
    let mut previous = Vec::new();
    let mut current = Vec::new();
    for _ in 0..repetitions {
        for index in 0..=list.count() {
            if !list.timelines(index).is_empty() {
                order_segment(&mut segment, &mut schedule);
                order = Order::default();
                previous.clear();
                schedule.push(Scheduled::Timelines(index));
            }
            if index == list.count() {
                break;
            }
            buffers.clear();
            let dispatch = list.dispatch(index, &mut buffers);
            reaches.clear();
            list.reaches(index, &mut reaches);
            accesses.clear();
            dispatch_accesses(&dispatch, &reaches, &mut accesses);
            for ordinal in dispatch.kernel.implementation.dispatch_order(dispatch.rounds) {
                if dispatch.launches[ordinal].is_none_or(|launch| launch.is_empty()) {
                    continue;
                }
                let mut after = order.after(&accesses);
                if chained {
                    after.extend(&previous);
                    current.push(order.placed());
                }
                let alone = dispatch.launches[ordinal].is_some_and(|launch| launch.barrier.is_some());
                segment.push((order.place(&accesses, &after, alone), index, ordinal));
            }
            // An inactive call is no link of the chain.
            if !current.is_empty() {
                std::mem::swap(&mut previous, &mut current);
                current.clear();
            }
        }
    }
    order_segment(&mut segment, &mut schedule);
    schedule
}

/// The schedule of a sealed-plan submission, kept with the device's replays
/// (`graph_replays`): with its plans, what every dispatch reaches up to
/// where it lies ([`relative_placement`]) fixes it, so every submission of
/// those plans placed alike, wherever its storage lies, encodes one
/// schedule, and only a change in how its reaches meet derives levels
/// again. Deriving them costs a 64k decode step up to 1.2 ms of host time
/// on an M4 Pro, above the step's resolution.
pub(crate) struct Schedule(pub(crate) Vec<Scheduled>);

impl super::graph_replays::RecordedGraph for Schedule {
    /// One graph, of host data alone: Metal records no launch into device
    /// memory (`GraphCost::None`), so only the schedules' count is bounded.
    fn census(&self) -> seismic_compiler::graph_cost::GraphCensus {
        seismic_compiler::graph_cost::GraphCensus { graphs: 1, ..Default::default() }
    }
    fn idle(&self) -> Result<bool, crate::api::CallError> {
        Ok(true)
    }
    /// A schedule binds nothing, so one serves every run of its shape.
    fn updates_in_flight() -> bool {
        seismic_lang::registry::BackendName::Metal.graphs_update_in_flight()
    }
}

/// What fixes the order of `calls` of `list` beside their sealed plans:
/// every range each dispatch reaches, in dispatch order, as its
/// allocation's place among the allocations reached before (first reached
/// first) and its offset from that allocation's first reach, with its
/// bytes. Two placements alike in these words meet alike: each
/// allocation's reaches are the other's moved by one distance, and reaches
/// of different allocations never meet, so every dependency of one is the
/// other's ([`Order`]). A slab view's reaches name the slabs its table
/// backs.
pub(crate) fn relative_placement(list: &impl DispatchList, calls: Range<usize>) -> Vec<u64> {
    let mut placement = Vec::new();
    // Per allocation reached, its place and its first reach's offset.
    let mut first: HashMap<u64, (u64, u64)> = HashMap::new();
    let mut reaches = Vec::new();
    for index in calls {
        reaches.clear();
        list.reaches(index, &mut reaches);
        placement.push(reaches.len() as u64);
        for reach in &reaches {
            let places = first.len() as u64;
            let (place, origin) = *first.entry(reach.allocation.identity()).or_insert((places, reach.offset));
            // Two's complement: a reach before the allocation's first lies
            // at a distance below it.
            placement.extend([place, reach.offset.wrapping_sub(origin), reach.bytes, reach.through as u64]);
        }
    }
    placement
}

/// Append `segment`'s launches to `schedule` level by level, each level in
/// submission order, and empty it.
fn order_segment(segment: &mut Vec<(usize, usize, usize)>, schedule: &mut Vec<Scheduled>) {
    // Stable: a level keeps submission order.
    segment.sort_by_key(|(level, ..)| *level);
    for level in segment.chunk_by(|first, second| first.0 == second.0) {
        schedule.push(Scheduled::Level(level.len()));
        schedule.extend(
            level
                .iter()
                .map(|&(_, index, ordinal)| Scheduled::Launch { index, ordinal }),
        );
    }
    segment.clear();
}

/// How a launch reaches bytes through one of its buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Through {
    /// The bound view itself.
    Binding,
    /// A slab view's address-table entries, its native binding, which the
    /// launch only reads.
    Table,
    /// A backed slab region, or a record port's records, reached by the
    /// addresses its table holds.
    Slab,
}

/// What a slab view's binding reaches ([`TensorInner::slab_reach`]): its
/// address-table entries and, by address, each backed slab region
/// (allocation, byte offset, bytes).
///
/// [`TensorInner::slab_reach`]: crate::api::tensor::TensorInner::slab_reach
pub(crate) struct SlabReach {
    pub(crate) table_bytes: u64,
    pub(crate) regions: Vec<(std::sync::Arc<Allocation>, u64, u64)>,
}

impl SlabReach {
    /// The reaches of buffer `buffer`, bound to this slab view's table at
    /// `offset` of `table`, into `reaches`.
    pub(crate) fn extend<'a>(
        &'a self,
        buffer: usize,
        table: &'a Allocation,
        offset: u64,
        reaches: &mut Vec<Reach<'a>>,
    ) {
        reaches.push(Reach {
            buffer,
            allocation: table,
            offset,
            bytes: self.table_bytes,
            through: Through::Table,
        });
        reaches.extend(self.regions.iter().map(|(allocation, offset, bytes)| Reach {
            buffer,
            allocation,
            offset: *offset,
            bytes: *bytes,
            through: Through::Slab,
        }));
    }
}

impl Records {
    /// The reach of buffer `buffer`, bound to the table entries of records
    /// `[first, first + count)` of this view: the bytes the records span,
    /// reached by their addresses. The entries themselves order nothing:
    /// device work only reads a record table (the host writes it).
    pub(crate) fn extend<'a>(
        &'a self,
        buffer: usize,
        first: u64,
        count: u64,
        reaches: &mut Vec<Reach<'a>>,
    ) {
        let (offset, bytes) = self.span(first, count);
        reaches.push(Reach {
            buffer,
            allocation: &self.storage,
            offset,
            bytes,
            through: Through::Slab,
        });
    }
}

/// Bytes one launch reaches through its buffer `buffer` (ABI order).
pub(crate) struct Reach<'a> {
    pub(crate) buffer: usize,
    pub(crate) allocation: &'a Allocation,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    pub(crate) through: Through,
}

/// What one launch reaches of one allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Access {
    /// [`Allocation::identity`].
    allocation: u64,
    bytes: Range<u64>,
    write: bool,
}

impl Access {
    fn of(allocation: &Allocation, offset: u64, bytes: u64, write: bool) -> Self {
        Self {
            allocation: allocation.identity(),
            bytes: offset..offset.checked_add(bytes).expect("a reached range's end fits u64"),
            write,
        }
    }

    /// Whether it writes any of `bytes` of the allocation `allocation`
    /// ([`Allocation::identity`]).
    pub(crate) fn writes(&self, allocation: u64, bytes: &Range<u64>) -> bool {
        self.write && self.allocation == allocation && self.bytes.start < bytes.end && bytes.start < self.bytes.end
    }
}

/// What every launch of `dispatch` reaches, into `accesses`: its `reaches`,
/// written as the kernel's ABI writes their buffers or its host rewrites
/// them, and its scalar-result slots when the kernel publishes scalars.
pub(crate) fn dispatch_accesses(
    dispatch: &Dispatch<'_>,
    reaches: &[Reach<'_>],
    accesses: &mut Vec<Access>,
) {
    let kernel = dispatch.kernel;
    accesses.extend(reaches.iter().map(|reach| {
        let written = kernel.buffer_writes[reach.buffer] || kernel.host_writes[reach.buffer];
        let write = written && reach.through != Through::Table;
        Access::of(reach.allocation, reach.offset, reach.bytes, write)
    }));
    if let Some(scalars) = &kernel.scalars {
        accesses.push(Access::of(&scalars.slots, 0, scalars.bytes(), true));
    }
}

/// Bytes an earlier launch of the segment reached: the launch, by its
/// place among the segment's placed launches.
#[derive(Clone, Debug)]
struct Reached {
    bytes: Range<u64>,
    launch: usize,
}

impl Reached {
    /// Whether these bytes contain `other`'s.
    fn contains(&self, other: &Reached) -> bool {
        self.bytes.start <= other.bytes.start && other.bytes.end <= self.bytes.end
    }

    /// Whether these bytes meet `bytes`.
    fn meets(&self, bytes: &Range<u64>) -> bool {
        self.bytes.start < bytes.end && bytes.start < self.bytes.end
    }
}

/// The ranges of one allocation earlier launches of the segment reached that
/// a later launch can still depend on. A later read depends on the writes
/// it meets; a later write on the reads and writes. A write that contains a
/// range reached before decides for its bytes: whatever later meets them
/// meets the write, which depends on that range's launch.
#[derive(Default)]
struct Allocated {
    reads: Vec<Reached>,
    writes: Vec<Reached>,
}

/// The dependencies and levels of one segment's launches, in submission
/// order ([`schedule`]; CUDA's graphs, `cuda::issue`).
#[derive(Default)]
pub(crate) struct Order {
    reached: HashMap<u64, Allocated>,
    /// Each placed launch's level.
    levels: Vec<usize>,
    /// One past the highest level placed.
    top: usize,
    /// The lowest level a later launch may take: one past the last resident
    /// launch's.
    floor: usize,
}

impl Order {
    /// The earlier placed launches the next one, which reaches `accesses`,
    /// depends on, ascending.
    pub(crate) fn after(&self, accesses: &[Access]) -> Vec<usize> {
        let mut after = Vec::new();
        for access in accesses {
            let Some(allocated) = self.reached.get(&access.allocation) else {
                continue;
            };
            let reads = access.write.then_some(&allocated.reads).into_iter().flatten();
            after.extend(
                allocated
                    .writes
                    .iter()
                    .chain(reads)
                    .filter(|reached| reached.meets(&access.bytes))
                    .map(|reached| reached.launch),
            );
        }
        after.sort_unstable();
        after.dedup();
        after
    }

    /// How many launches are placed: the next one's index.
    pub(crate) fn placed(&self) -> usize {
        self.levels.len()
    }

    /// Place the next launch, which reaches `accesses` and depends on
    /// `after` (its [`Self::after`]): its level, one past the highest of
    /// those launches' (0 with none), and past the last resident launch's.
    /// An `alone` (resident) launch takes a level above every placed one,
    /// and no later launch shares it.
    pub(crate) fn place(&mut self, accesses: &[Access], after: &[usize], alone: bool) -> usize {
        let needed = after.iter().map(|&launch| self.levels[launch] + 1).max().unwrap_or(0);
        let level = if alone { needed.max(self.top) } else { needed.max(self.floor) };
        if alone {
            self.floor = level + 1;
        }
        self.top = self.top.max(level + 1);
        let launch = self.levels.len();
        self.levels.push(level);
        for access in accesses {
            let new = Reached {
                bytes: access.bytes.clone(),
                launch,
            };
            let allocated = self.reached.entry(access.allocation).or_default();
            if access.write {
                allocated.reads.retain(|earlier| !new.contains(earlier));
                allocated.writes.retain(|earlier| !new.contains(earlier));
                allocated.writes.push(new);
            } else {
                allocated.reads.push(new);
            }
        }
        level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn access(allocation: u64, bytes: Range<u64>, write: bool) -> Access {
        Access {
            allocation,
            bytes,
            write,
        }
    }

    fn read(allocation: u64, bytes: Range<u64>) -> Access {
        access(allocation, bytes, false)
    }

    fn write(allocation: u64, bytes: Range<u64>) -> Access {
        access(allocation, bytes, true)
    }

    /// The level of each launch of `launches`, in submission order.
    fn levels(launches: &[Vec<Access>]) -> Vec<usize> {
        let mut order = Order::default();
        launches
            .iter()
            .map(|launch| {
                let after = order.after(launch);
                order.place(launch, &after, false)
            })
            .collect()
    }

    /// The launches each launch of `launches` depends on, in submission
    /// order.
    fn dependencies(launches: &[Vec<Access>]) -> Vec<Vec<usize>> {
        let mut order = Order::default();
        launches
            .iter()
            .map(|launch| {
                let after = order.after(launch);
                order.place(launch, &after, false);
                after
            })
            .collect()
    }

    /// A resident launch shares its level with no launch, earlier or
    /// later, though none of them depends on another.
    #[test]
    fn a_resident_launch_is_alone_in_its_level() {
        let mut order = Order::default();
        let placed = [(1, false), (2, true), (3, false), (4, false)]
            .map(|(allocation, alone)| {
                let launch = [write(allocation, 0..64)];
                let after = order.after(&launch);
                order.place(&launch, &after, alone)
            });
        assert_eq!(placed, [0, 1, 2, 2]);
    }

    #[test]
    fn a_launch_follows_the_writes_it_reads() {
        assert_eq!(
            levels(&[vec![write(1, 0..64)], vec![read(1, 0..64), write(2, 0..64)]]),
            [0, 1]
        );
    }

    #[test]
    fn a_write_follows_earlier_reads_and_writes() {
        // Write after read.
        assert_eq!(levels(&[vec![read(1, 0..64)], vec![write(1, 0..64)]]), [0, 1]);
        // Write after write.
        assert_eq!(levels(&[vec![write(1, 0..64)], vec![write(1, 0..64)]]), [0, 1]);
    }

    #[test]
    fn reads_of_one_storage_share_a_level() {
        assert_eq!(
            levels(&[
                vec![read(1, 0..64), write(2, 0..64)],
                vec![read(1, 0..64), write(3, 0..64)],
            ]),
            [0, 0]
        );
    }

    #[test]
    fn disjoint_views_of_one_buffer_are_independent() {
        // Two results of one workspace arena, side by side: the ranges are
        // half open, so touching ends do not overlap.
        assert_eq!(
            levels(&[vec![write(1, 0..64)], vec![write(1, 64..128)], vec![read(1, 128..256)]]),
            [0, 0, 0]
        );
    }

    #[test]
    fn aliasing_views_of_one_buffer_conflict() {
        // A later result placed over an earlier one's bytes (liveness reuse
        // of one arena), and a narrower view inside a written one.
        assert_eq!(levels(&[vec![read(1, 0..64)], vec![write(1, 32..96)]]), [0, 1]);
        assert_eq!(levels(&[vec![write(1, 0..256)], vec![read(1, 100..101)]]), [0, 1]);
    }

    #[test]
    fn distinct_allocations_are_independent_at_equal_offsets() {
        assert_eq!(levels(&[vec![write(1, 0..64)], vec![write(2, 0..64)]]), [0, 0]);
    }

    #[test]
    fn slab_regions_conflict_only_where_their_bytes_meet() {
        // Slab storage 7 holds two slabs, [0, 4096) and [4096, 8192); a
        // slab view reaches table 6 (read) and both regions.
        let view = |write| {
            vec![
                read(6, 0..16),
                access(7, 0..4096, write),
                access(7, 4096..8192, write),
            ]
        };
        // A write into the second slab's view against a reader through the
        // table: real.
        assert_eq!(levels(&[view(false), vec![write(7, 5000..5100)]]), [0, 1]);
        // A writer through the table against another slab storage, and
        // against the table's other region entries: none.
        assert_eq!(
            levels(&[view(true), vec![write(8, 0..4096)], vec![read(6, 16..32)]]),
            [0, 0, 0]
        );
        // Two readers through one table: none.
        assert_eq!(levels(&[view(false), view(false)]), [0, 0]);
    }

    #[test]
    fn a_launch_rises_to_the_earliest_level_after_its_dependencies() {
        // A router (reads the residual, writes its choices), a fill that
        // reads them, then a shared expert that reads only the residual:
        // the shared expert runs beside the router, before the fill.
        assert_eq!(
            levels(&[
                vec![read(1, 0..64), write(2, 0..32)],
                vec![read(2, 0..32), write(3, 0..64)],
                vec![read(1, 0..64), write(4, 0..64)],
                // The expert product reads the fill's and the shared's.
                vec![read(3, 0..64), read(4, 0..64), write(5, 0..64)],
            ]),
            [0, 1, 0, 2]
        );
    }

    #[test]
    fn a_launch_follows_its_deepest_dependency() {
        // A chain of three, then a launch reading the chain's head and tail.
        assert_eq!(
            levels(&[
                vec![write(1, 0..8)],
                vec![read(1, 0..8), write(2, 0..8)],
                vec![read(2, 0..8), write(3, 0..8)],
                vec![read(1, 0..8), read(3, 0..8), write(4, 0..8)],
            ]),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn a_contained_later_access_decides_for_its_bytes() {
        // A write at level 1 over a read at level 0 of the same bytes: a
        // later write follows the level-1 write; reads of other bytes of
        // the first range still see the level-0 read only through a write.
        assert_eq!(
            levels(&[
                vec![read(1, 0..64), write(2, 0..8)],
                vec![read(2, 0..8), write(1, 0..32)],
                vec![write(1, 0..64)],
                vec![read(1, 40..48)],
            ]),
            [0, 1, 2, 3]
        );
        // A read at a higher level does not hide an earlier write from a
        // later read.
        assert_eq!(
            levels(&[
                vec![write(1, 0..64)],
                vec![write(9, 0..8)],
                vec![read(9, 0..8), read(1, 0..64)],
                vec![read(1, 0..64)],
            ]),
            [0, 0, 1, 1]
        );
    }

    /// A launch depends on exactly the earlier launches it conflicts with:
    /// a long independent launch beside a chain is no dependency of the
    /// chain, and the launch that reads both depends on both.
    #[test]
    fn a_launch_depends_on_exactly_the_launches_it_conflicts_with() {
        assert_eq!(
            dependencies(&[
                // A route: writes its choices and a placement word.
                vec![write(1, 0..8), write(9, 0..4)],
                // A prefetch beside it: reads the choices, rewrites the word.
                vec![read(1, 0..8), write(9, 0..4)],
                // The layer's entries: read the choices only.
                vec![read(1, 0..8), write(2, 0..8)],
                vec![read(2, 0..8), write(3, 0..8)],
                // The next route reads the entries' output and the word.
                vec![read(3, 0..8), read(9, 0..4), write(4, 0..8)],
            ]),
            [vec![], vec![0], vec![0], vec![2], vec![1, 3]]
        );
        // A read contained in a later write is decided by the write: a
        // later write depends on the write alone.
        assert_eq!(
            dependencies(&[vec![read(1, 8..16)], vec![write(1, 0..64)], vec![write(1, 0..8)]]),
            [vec![], vec![0], vec![1]]
        );
        // Reads of one range are each a later write's dependency.
        assert_eq!(
            dependencies(&[vec![read(1, 0..64)], vec![read(1, 0..64)], vec![write(1, 0..8)]]),
            [vec![], vec![], vec![0, 1]]
        );
    }

    #[test]
    fn a_launch_of_several_bindings_follows_through_any() {
        assert_eq!(
            levels(&[
                vec![read(1, 0..64), write(2, 0..64)],
                vec![read(3, 0..64), read(4, 0..64), read(2, 60..61), write(5, 0..8)],
            ]),
            [0, 1]
        );
    }

    /// A CPU kernel of one launch reading buffer 0 and writing buffer 1, its
    /// function never run: `schedule` reads only its ABI and launches.
    fn copy_kernel(device: &std::sync::Arc<crate::api::device::DeviceInner>) -> std::sync::Arc<super::super::NativePrepared> {
        use crate::native::{CpuInvocation, CpuLaunchVariants, CpuNativeKernels, CpuVariant};
        use seismic_native_cpu::Tier;
        unsafe fn unused(_: &CpuInvocation<'_>, _: [u64; 3], _: &mut [u8]) {}
        const fn variant(tier: Tier) -> CpuVariant {
            CpuVariant {
                tier,
                elements: &[],
                function: unused,
            }
        }
        static KERNELS: CpuNativeKernels = CpuNativeKernels {
            digest: "hazard-schedule-test",
            elements: &[],
            launches: &[CpuLaunchVariants {
                kernel: "copy",
                variants: &[
                    variant(Tier::X86V2),
                    variant(Tier::X86V3),
                    variant(Tier::X86V4),
                    variant(Tier::X86V4Vnni),
                    variant(Tier::Neon),
                ],
            }],
        };
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "hazard-schedule.seismic".into(),
            text: "fn copy[N](x: &tensor[N] f32, y: &mut tensor[N] f32):\n    for i in 0..N:\n        y[i] = x[i]\n\nnative copy for cpu from \"copy.rs\":\n    static (N)\n    launch copy:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".into(),
        }]))
        .unwrap();
        let kernel = super::super::NativePrepared::prepare(
            device,
            &module,
            module.entry_named("copy").unwrap(),
            seismic_lang::entry::ElementBindings::new(),
            seismic_lang::checked::NativeSpecialization::new().with_static("N", 8),
            Some(&KERNELS),
        )
        .unwrap();
        assert!(kernel.scalars.is_none());
        assert_eq!(kernel.buffer_writes, [false, true]);
        kernel
    }

    /// Calls of one kernel, each reading `(allocation, range)` and writing
    /// another, `None` for a call whose launch is inactive; timeline
    /// operations at the boundaries named.
    struct Calls<'a> {
        kernel: &'a super::super::NativePrepared,
        calls: Vec<Option<[(&'a Allocation, Range<u64>); 2]>>,
        timelines: HashMap<usize, Vec<super::super::timeline::TimelineOperation>>,
        /// Each call after the one before, as standalone calls run.
        chained: bool,
    }

    const ACTIVE: [Option<super::super::LaunchGeometry>; 1] = [Some(super::super::LaunchGeometry {
        groups: [1, 1, 1],
        grid: None,
        barrier: None,
        threads: [1, 1, 1],
        shared_bytes: 0,
    })];
    const INACTIVE: [Option<super::super::LaunchGeometry>; 1] = [None];

    impl DispatchList for Calls<'_> {
        fn count(&self) -> usize {
            self.calls.len()
        }
        fn dispatch<'s>(&'s self, index: usize, _: &mut Vec<(&'s Allocation, u64)>) -> Dispatch<'s> {
            Dispatch {
                kernel: self.kernel,
                words: &[],
                launches: if self.calls[index].is_some() { &ACTIVE } else { &INACTIVE },
                rounds: 1,
                representations: &[],
            }
        }
        fn reaches<'s>(&'s self, index: usize, reaches: &mut Vec<Reach<'s>>) {
            for (buffer, (allocation, range)) in self.calls[index].iter().flatten().enumerate() {
                reaches.push(Reach {
                    buffer,
                    allocation,
                    offset: range.start,
                    bytes: range.end - range.start,
                    through: Through::Binding,
                });
            }
        }
        fn encodes(&self, _: usize, _: usize) -> bool {
            true
        }
        fn runs(&self) -> Option<Vec<super::super::PlanRun>> {
            None
        }
        fn timelines(&self, boundary: usize) -> &[super::super::timeline::TimelineOperation] {
            self.timelines.get(&boundary).map_or(&[], Vec::as_slice)
        }
        fn chained(&self) -> bool {
            self.chained
        }
    }

    fn cpu_device() -> std::sync::Arc<crate::api::device::DeviceInner> {
        crate::devices::Devices::discover(&[])
            .unwrap()
            .open_backend(seismic_lang::registry::BackendName::Cpu)
            .unwrap()
    }

    fn launch(index: usize) -> Scheduled {
        Scheduled::Launch { index, ordinal: 0 }
    }

    /// A timeline operation begins a new segment whose levels start again:
    /// a launch after it that depends on one before it is not held behind
    /// independent launches after it, and none crosses it.
    #[test]
    fn a_timeline_boundary_starts_the_levels_again() {
        let device = cpu_device();
        let kernel = copy_kernel(&device);
        let storage = (0..4)
            .map(|_| device.allocate(64, 64).unwrap())
            .collect::<Vec<_>>();
        let [a, b, c, d] = [&*storage[0], &*storage[1], &*storage[2], &*storage[3]];
        let timeline = std::sync::Arc::new(super::super::timeline::TimelineInner::new(&device).unwrap());
        let wait = super::super::timeline::TimelineOperation {
            action: super::super::timeline::TimelineAction::Wait,
            timeline,
            value: 1,
        };
        let calls = Calls {
            kernel: &kernel,
            calls: vec![
                Some([(a, 0..64), (b, 0..64)]),
                // Reads what the first writes: level 1 of its segment.
                Some([(b, 0..64), (c, 0..64)]),
                // After the gate: reads what the second wrote, and an
                // independent call; both at level 0 of the new segment.
                Some([(c, 0..64), (d, 0..32)]),
                Some([(a, 0..64), (d, 32..64)]),
            ],
            timelines: HashMap::from([(2, vec![wait])]),
            chained: false,
        };
        assert_eq!(
            schedule(&calls, 1),
            [
                Scheduled::Level(1),
                launch(0),
                Scheduled::Level(1),
                launch(1),
                Scheduled::Timelines(2),
                Scheduled::Level(2),
                launch(2),
                launch(3),
            ]
        );
    }

    /// Inactive launches do no device work and are left out; repetitions
    /// are one segment, each repetition's launches following the earlier
    /// ones they depend on.
    #[test]
    fn a_schedule_leaves_out_inactive_launches_and_carries_levels_across_repetitions() {
        let device = cpu_device();
        let kernel = copy_kernel(&device);
        let storage = (0..2)
            .map(|_| device.allocate(64, 64).unwrap())
            .collect::<Vec<_>>();
        let [a, b] = [&*storage[0], &*storage[1]];
        let calls = Calls {
            kernel: &kernel,
            calls: vec![
                Some([(a, 0..64), (b, 0..64)]),
                None,
                // Rewrites what the first reads: write after read.
                Some([(b, 0..64), (a, 0..64)]),
            ],
            timelines: HashMap::new(),
            chained: false,
        };
        assert_eq!(
            schedule(&calls, 2),
            [
                Scheduled::Level(1),
                launch(0),
                Scheduled::Level(1),
                launch(2),
                Scheduled::Level(1),
                launch(0),
                Scheduled::Level(1),
                launch(2),
            ]
        );
    }

    /// Calls of a chained list run each after the one before, as a served
    /// step runs one kernel's calls, though they reach nothing in common:
    /// a rotation of independent argument sets is timed in served order,
    /// never overlapped (an M4 Pro's in-order phase of one GEMV read 26.65
    /// us against 14.09 us four at a time).
    #[test]
    fn a_chained_list_runs_each_call_after_the_one_before() {
        let device = cpu_device();
        let kernel = copy_kernel(&device);
        let storage = (0..4)
            .map(|_| device.allocate(64, 64).unwrap())
            .collect::<Vec<_>>();
        let [a, b, c, d] = [&*storage[0], &*storage[1], &*storage[2], &*storage[3]];
        let calls = |chained| Calls {
            kernel: &kernel,
            calls: vec![Some([(a, 0..64), (b, 0..64)]), None, Some([(c, 0..64), (d, 0..64)])],
            timelines: HashMap::new(),
            chained,
        };
        assert_eq!(
            schedule(&calls(false), 2),
            [
                Scheduled::Level(2),
                launch(0),
                launch(2),
                Scheduled::Level(2),
                launch(0),
                launch(2),
            ],
            "independent calls share a level"
        );
        assert_eq!(
            schedule(&calls(true), 2),
            [
                Scheduled::Level(1),
                launch(0),
                Scheduled::Level(1),
                launch(2),
                Scheduled::Level(1),
                launch(0),
                Scheduled::Level(1),
                launch(2),
            ],
            "an inactive call is no link of the chain"
        );
    }

    /// The relative placement is alike exactly where the reaches meet alike:
    /// each allocation's reaches moved by one distance keep it, as does
    /// another allocation in an allocation's place; reaches moved apart, of
    /// another length, sharing an allocation they did not, or fewer change
    /// it.
    #[test]
    fn the_relative_placement_follows_how_reaches_meet() {
        let device = cpu_device();
        let kernel = copy_kernel(&device);
        let storage = (0..3)
            .map(|_| device.allocate(256, 64).unwrap())
            .collect::<Vec<_>>();
        let [a, b, other] = [&*storage[0], &*storage[1], &*storage[2]];
        let placed = |calls: Vec<Option<[(&Allocation, Range<u64>); 2]>>| {
            let count = calls.len();
            relative_placement(
                &Calls {
                    kernel: &kernel,
                    calls,
                    timelines: HashMap::new(),
                    chained: false,
                },
                0..count,
            )
        };
        let base = placed(vec![Some([(a, 0..64), (b, 0..64)]), Some([(a, 64..128), (b, 64..128)])]);
        for alike in [
            // Every reach of `a` moved by 64 bytes, of `b` by 128.
            placed(vec![Some([(a, 64..128), (b, 128..192)]), Some([(a, 128..192), (b, 192..256)])]),
            // Another allocation in `a`'s place.
            placed(vec![Some([(other, 0..64), (b, 0..64)]), Some([(other, 64..128), (b, 64..128)])]),
        ] {
            assert_eq!(base, alike);
        }
        for changed in [
            // `a`'s second reach moved apart from its first: now it meets it.
            placed(vec![Some([(a, 0..64), (b, 0..64)]), Some([(a, 32..96), (b, 64..128)])]),
            // Of another length.
            placed(vec![Some([(a, 0..64), (b, 0..64)]), Some([(a, 64..128), (b, 64..96)])]),
            // One allocation where there were two.
            placed(vec![Some([(a, 0..64), (a, 128..192)]), Some([(a, 64..128), (a, 192..256)])]),
            // A reach fewer.
            placed(vec![Some([(a, 0..64), (b, 0..64)]), None]),
        ] {
            assert_ne!(base, changed);
        }
    }

    #[test]
    fn a_segment_orders_its_launches_by_level() {
        let mut schedule = Vec::new();
        let mut segment = vec![(0, 0, 0), (1, 1, 0), (0, 2, 0), (2, 3, 0), (1, 4, 0)];
        order_segment(&mut segment, &mut schedule);
        let launch = |index| Scheduled::Launch { index, ordinal: 0 };
        assert_eq!(
            schedule,
            [
                Scheduled::Level(2),
                launch(0),
                launch(2),
                Scheduled::Level(2),
                launch(1),
                launch(4),
                Scheduled::Level(1),
                launch(3),
            ]
        );
        assert!(segment.is_empty());
    }
}
