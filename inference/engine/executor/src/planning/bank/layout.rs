use super::{BankBinding, ExpertBankPlan, RecordTensors, RoutedTensor};
use super::records::{record_bytes, reserved_record};
use crate::memory::ClaimDemand;
use crate::planning::resources::FlightDepth;
use crate::planning::weights::representation_bytes;
use magnitude_kernels::{constants, list_extent};
use seismic::Element;
use std::num::NonZeroU64;

/// A part of a streamed routed layer's bank its graphs bind (`BankPort`),
/// beside its banked expert stacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BankPart {
    /// A chain place's step word, which the run's first streamed layer sets
    /// to its sequence and every streamed entry of the run reads.
    Step,
    /// Whether the prompt group being encoded is resident: every streamed
    /// layer whole and its experts run in the group's pass, so its fronts
    /// request nothing (`routed_scratch_request`). The host writes it as it
    /// begins a group, before encoding any of its runs.
    Resident,
    /// The layer's map, `[E, MAP_WORDS]`: its view of its group's map
    /// ([`Self::Map`]), the layer's expert `e` at the group's entry
    /// `home + e` (`ExpertBankPlan::home`).
    SlotMap,
    /// The layer's bank group's map, `[GM, MAP_WORDS]`: every member
    /// layer's map, end to end, each entry the bank record holding its
    /// expert and the steps it holds it for, the host its one writer
    /// (`streaming_words.seismic`'s map), which a decode route resolves
    /// through (`routed_route_streamed`'s `map`).
    Map,
    /// The layer's tier map: the host tier record holding each of its
    /// experts whole, or -1, where the bank has a host tier
    /// (`routed_route_streamed`'s `tier`).
    TierMap,
    /// The layer's bank group's host tier records of routed tensor `kind`,
    /// where the bank has a host tier: what the decode entries read a
    /// tier-served choice from in place, and the host's promotions copy
    /// into the bank.
    Tier { kind: RoutedTensor },
    /// The layer's tables of the exchange (streaming_words.seismic): its
    /// file kind's (`file`, the decode route's requests of the experts
    /// neither the bank nor the tier holds, `routed_land`'s slot) and
    /// its scratch kind's (`scratch`, a scratch-served step's request and
    /// rounds), each a slot for each step in flight (`RegionWords::file`,
    /// `RegionWords::scratch`).
    FileTable,
    ScratchTable,
    /// The layer's hint rings, one per layer it predicts
    /// (`ExpertBankPlan::ahead`), and its routing records, one for each
    /// step in flight (`RegionWords::flight`), ring words each.
    Rings,
    Routing,
    /// The layer's policy words (`POLICY_WORDS`), the host's between
    /// flights (the watcher's `publish_policy`): its POLICY_WHOLE word,
    /// nonzero while the layer's group is whole (nothing misses there, so
    /// its landing launch and the hints of it are not encoded, and its route is
    /// the whole form: the resident route and its routing record,
    /// `routed_route_whole`, `routed_select_whole`); its home; and the
    /// lines the route's assignment reads (`streaming_words.seismic`'s
    /// policy).
    Policy,
    /// The layer's site kind's table of the exchange (`site [FLIGHT, R]`,
    /// `RegionWords::site`), the rows its requests carry (`site_rows
    /// [FLIGHT, DECODE_ROWS, H]` f32) and the CPU's answers' sums
    /// (`site_output [FLIGHT, DECODE_ROWS, H]` f32), one slot for each group
    /// in flight (`routed_site_request`, `routed_down_sited`,
    /// `routed_output_sited`; streaming_words.seismic's site kind), where
    /// the layer has a CPU site: host-shared, the CPU pool's
    /// (`experts::site`).
    SiteTable,
    SiteRows,
    SiteOutput,
    /// The layer's bank group's commits: each record's step whose seed the
    /// device committed in it (`routed_seed_commit`).
    Commits,
    /// The conveyor of routed tensor `kind` in the layer's stored form: its
    /// records, a view of the lender's (`ConveyorCarrier`), which a scratch
    /// round's landing launch (`routed_round`) converts from.
    Conveyor { kind: RoutedTensor },
    /// The staging pool of routed tensor `kind`, the layer's stored form's,
    /// which the layer's decode landing launch (`routed_land`) converts
    /// from where the host places, a miss's record then the bank's own
    /// (`experts::landing`); and that launch's landing list
    /// (`streaming_words.seismic`'s, its entries each (staging record, bank
    /// record, DONE)), which the host appends to as each record's read
    /// lands, while the launch runs. Staging only where the host places;
    /// the list for every layer, which every landing launch declares.
    Staging { kind: RoutedTensor },
    LandingList,
    /// The scratch tier's view of routed tensor `kind` in the layer's form:
    /// slot `e` holding expert `e`, a view of the scratch the lender lends
    /// every prompt group ([`ScratchPlan`]), which the rounds convert cold
    /// experts into and the pass reads them from.
    Scratch { kind: RoutedTensor },
    /// A chain place's scratch flags, scratch round `round`'s landing list
    /// (its landing launch's, `routed_round`: each conveyed expert's
    /// (conveyor record or scratch slot, scratch slot), appended as its read
    /// lands), the tier list (a step's experts the host tier keeps whole, each
    /// (tier record, scratch slot): its tier copy after round 0) and the
    /// seed list (a prompt group's last step beside a host tier: its seeds'
    /// copy from their scratch slots into their bank records, then commit),
    /// the last two each layer's own.
    ScratchFlags,
    RoundList { round: u8 },
    TierList,
    SeedList,
    /// The launch grid a scratch request's round 0 writes of the tier list
    /// (`routed_list_grid`), on which the tier copy launches: none of a list
    /// of no entries for the step; and the list it stages in device memory
    /// beside it, which the tier copy reads.
    TierGrid,
    TierStaged,
    /// The seed list's launch grid and its staging in device memory, as the
    /// tier list's (`routed_list_grid`), on which the seed copy launches.
    SeedGrid,
    SeedStaged,
    /// The round grids a scratch request's round 0 writes
    /// (`routed_scratch_grids`), each round's park grid, which its later
    /// rounds' landing launches dispatch on.
    ScratchGrids,
    /// The layer's group reads: of each record of a group in flight
    /// (`FlightDepth::groups`) and each chain place, the choices of
    /// the merge set its pass grouped from that place routed to each expert
    /// (`routed_group`'s `counts`, `[E]`), zero where no pass began there.
    /// A group holds its record from its beginning, when the host zeroes
    /// it, until every step of it has ended, when the host reads it.
    GroupReads,
}

impl BankPart {
    /// Whether the part is a chain place's, which each run binds at its own
    /// place: a prompt group's steps' graphs interleave, each reading its
    /// own step, and every step requests at a layer before it is served
    /// there (`routed_scratch_request`); a group's reads at each record's
    /// places, a pass binding its group's record at its merge set's first.
    pub fn placed(self) -> bool {
        matches!(self, Self::Step | Self::ScratchFlags | Self::GroupReads)
    }

    /// The instances of a placed part of a load of `places` chain places
    /// flying at `flights`.
    pub fn instances(self, flights: FlightDepth, places: u64) -> u64 {
        match self {
            Self::GroupReads => flights.steps(places),
            _ => places,
        }
    }

    /// Every part a streamed layer's bank of `rounds` convert rounds may
    /// hold, of which [`ExpertBankPlan::held_parts`] keeps those a layer
    /// holds.
    pub(crate) fn every(rounds: u64) -> impl Iterator<Item = Self> {
        // Exhaustive, so a part added to the bank is listed below.
        let _listed = |part: Self| match part {
            Self::Step
            | Self::Resident
            | Self::SlotMap
            | Self::Map
            | Self::TierMap
            | Self::Tier { .. }
            | Self::FileTable
            | Self::ScratchTable
            | Self::Rings
            | Self::Routing
            | Self::Policy
            | Self::SiteTable
            | Self::SiteRows
            | Self::SiteOutput
            | Self::Commits
            | Self::Conveyor { .. }
            | Self::Staging { .. }
            | Self::LandingList
            | Self::Scratch { .. }
            | Self::ScratchFlags
            | Self::RoundList { .. }
            | Self::TierList
            | Self::SeedList
            | Self::TierGrid
            | Self::TierStaged
            | Self::SeedGrid
            | Self::SeedStaged
            | Self::ScratchGrids
            | Self::GroupReads => (),
        };
        let records = RoutedTensor::ALL.into_iter().flat_map(|kind| {
            [Self::Tier { kind }, Self::Conveyor { kind }, Self::Staging { kind }, Self::Scratch { kind }]
        });
        let rounds = (0..=u8::MAX).take_while(move |&round| u64::from(round) < rounds).map(|round| Self::RoundList { round });
        [
            Self::Step,
            Self::Resident,
            Self::SlotMap,
            Self::Map,
            Self::TierMap,
            Self::FileTable,
            Self::ScratchTable,
            Self::Rings,
            Self::Routing,
            Self::Policy,
            Self::SiteTable,
            Self::SiteRows,
            Self::SiteOutput,
            Self::Commits,
            Self::LandingList,
            Self::ScratchFlags,
            Self::TierList,
            Self::SeedList,
            Self::TierGrid,
            Self::TierStaged,
            Self::SeedGrid,
            Self::SeedStaged,
            Self::ScratchGrids,
            Self::GroupReads,
        ]
        .into_iter()
        .chain(records)
        .chain(rounds)
    }
}

/// One allocation of a bank (`ExpertBankPlan::layout`): how it is held, and
/// so what it is charged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BankAllocation {
    /// Host-shared (`Tensor::shared`): its bytes.
    Shared { element: Element, extents: Vec<u64> },
    /// Coherent words the host and the device exchange (`Tensor::words`):
    /// its bytes.
    Words { element: Element, extents: Vec<u64> },
    /// Coherent words whose order across the device's queues the
    /// exchange's protocol gives (`Tensor::protocol_words`): a layer's
    /// scratch table, whose slots a step's front requests on the compute
    /// queue and its fill waits on beside it. Its bytes.
    Protocol { element: Element, extents: Vec<u64> },
    /// The device's own, zeroed (`Tensor::zeros`): its bytes.
    Device { element: Element, extents: Vec<u64> },
    /// `records` records of `matrix` reserved under `contract`, for the
    /// tier `kind`, the leading `committed` backed
    /// (`Tensor::reserved_in`); their bytes in whole granules of its
    /// reservation.
    Reserved {
        kind: crate::planning::BankTier,
        contract: seismic::ReservedContract,
        element: Element,
        records: u64,
        matrix: [u64; 2],
        committed: u64,
        /// The largest view its records carry beside their own
        /// (`seismic::ReservedRecords`): the conveyor's stored records, in
        /// its lender's.
        carries: Option<NonZeroU64>,
    },
    /// A banked stack's per-expert second-level scale, imported as its
    /// resident plan holds it: its resident bytes.
    Imported(super::super::weights::WeightScalePlan),
}

/// Where a streamed load's file misses land, by its memory: the host maps
/// every bank record on both (streaming_words.seismic's map).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placer {
    /// Across a link: a file miss lands into the host tier (`tier[e]`), which
    /// the CPU computes from and the GPU reads in place; the host promotes
    /// tier records into the bank on the copy engine, off every step's path,
    /// and the device copies a prompt group's seeds into it.
    Tier,
    /// Unified memory: there is no tier, and the host lands a file miss
    /// straight into a bank record, which the device reads where it lies.
    Bank,
}

/// Which of a bank's reserved tiers a holding lies in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BankTier {
    /// The bank's own records and tables.
    Bank,
    /// A linked host tier's records.
    Host,
}

impl BankTier {
    pub const COUNT: usize = 2;
    pub const ALL: [Self; Self::COUNT] = [Self::Bank, Self::Host];

    /// The tier's place in [`Self::ALL`], for what is kept per tier.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// A contract whose store the tier lies in, which claims against it are
    /// placed under: `DeviceOnly`'s, which on a unified device is `Shared`'s
    /// too, or `Shared`'s.
    pub const fn storage(self) -> seismic::Contract {
        match self {
            Self::Bank => seismic::Contract::DeviceOnly,
            Self::Host => seismic::Contract::Shared,
        }
    }

    /// The contract of the tier where shared ranges lie in the device's
    /// memory or not (`unified`): a bank the host writes in place is `Shared`,
    /// one across a link `DeviceOnly`; a host tier is `Shared`.
    pub const fn contract_where(self, unified: bool) -> seismic::ReservedContract {
        match (self, unified) {
            (Self::Bank, false) => seismic::ReservedContract::DeviceOnly,
            (Self::Bank, true) | (Self::Host, _) => seismic::ReservedContract::Shared,
        }
    }

    /// The contract of the tier on `executor`, from the rule
    /// (`DeviceTopology::shared_ranges_lie_in_device_memory`).
    pub fn contract(self, topology: &seismic::DeviceTopology, executor: &seismic::Executor) -> seismic::ReservedContract {
        self.contract_where(topology.shared_ranges_lie_in_device_memory(executor))
    }

    /// [`Self::contract`] on `device`.
    pub fn contract_on(self, device: &seismic::Device) -> seismic::ReservedContract {
        self.contract(device.topology(), device.info())
    }
}

/// What a device gives an expert bank: the backend its launches are, how its
/// record tensors are reserved, and the host tier beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankSubstrate {
    pub backend: seismic::BackendName,
    pub reservation: seismic::Reservation,
    pub host: HostTier,
}

impl BankSubstrate {
    /// Whether a whole bank's resident forms can read its home records:
    /// its backend's resident launches reach expert stacks through record
    /// ports, or bind them to storage that is one buffer
    /// (`native::ExpertKernels::bind_resident_stacks`).
    pub fn reads_homes(&self) -> bool {
        !(crate::native::ExpertKernels::bind_resident_stacks(self.backend) && self.reservation.chunked())
    }
}

/// How a bank's misses are held on its device beside the bank, where its
/// device storage and host RAM differ or are one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostTier {
    /// Host RAM a discrete device reads over its link, a tier of its own
    /// reserved as host storage its backend reserves: the host lands every
    /// file miss into it, a step serves a miss from its records, and the
    /// host promotes the records the ranking keeps into the bank on the copy
    /// engine (`watcher::promotions`).
    Linked(seismic::Reservation),
    /// Unified memory (Metal's, an integrated CUDA device's such as
    /// GB10's): device storage is host RAM, so the bank is the one store.
    /// Every file miss lands straight into a bank record, which the device
    /// reads at the bank's own bandwidth, placed by the map write alone:
    /// the host reads it into staging and the device's landing launch
    /// converts it in place (`experts::landing`); there is no tier. Where
    /// the device parks ([`Parking`]) sets how soon the host learns a
    /// converted record's staging is free.
    Unified(Parking),
}

/// Where the device waits for the host (`seismic::BackendName::parks_in_kernels`),
/// and so when the host sees what a running submission stored: inside its
/// kernels (Metal, CUDA), where host and device see each other's stores as
/// they land; or at submission boundaries (Vulkan), where the device's
/// stores are the host's only once its submission ends, past the park that
/// waits on every entry of a landing (`ExpertBankPlan::demand_records`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parking {
    InKernels,
    AtBoundaries,
}

impl Parking {
    /// Where `backend`'s device parks.
    pub fn of(backend: seismic::BackendName) -> Self {
        match backend.parks_in_kernels() {
            true => Self::InKernels,
            false => Self::AtBoundaries,
        }
    }
}

impl HostTier {
    /// Where a file miss lands.
    pub const fn placer(self) -> Placer {
        match self {
            Self::Linked(_) => Placer::Tier,
            Self::Unified(_) => Placer::Bank,
        }
    }

    /// The reservation of its tier's own records: a linked tier's; unified
    /// memory has none.
    pub const fn tier(self) -> Option<seismic::Reservation> {
        match self {
            Self::Linked(reservation) => Some(reservation),
            Self::Unified(_) => None,
        }
    }

    /// The contract of `tier`'s records.
    pub const fn contract(self, tier: BankTier) -> seismic::ReservedContract {
        tier.contract_where(matches!(self, Self::Unified(_)))
    }
}

impl BankAllocation {
    /// `extents` reserved under `contract` on `device`, the leading
    /// `committed` rows backed, and its resizer: the one way a bank's
    /// reserved records, and the fills measured over them, are allocated.
    pub fn reserve(
        device: &seismic::Device,
        contract: seismic::ReservedContract,
        element: Element,
        extents: &[u64],
        committed: u64,
        carries: Option<NonZeroU64>,
    ) -> Result<(seismic::Tensor, seismic::Resizer), seismic::TensorError> {
        seismic::Tensor::reserved_in(contract, device, element, extents, committed, carries)
    }

    /// Physical storage demanded by this allocation. Reserved backing cannot
    /// consume free space inside ordinary allocator blocks; its record table
    /// is an independent ordinary allocation in the declared placement.
    pub fn storage_demand(&self, reservation: &seismic::Reservation) -> Result<ClaimDemand, String> {
        Ok(match self {
            Self::Shared { element, extents } | Self::Words { element, extents } | Self::Protocol { element, extents } => {
                ClaimDemand { shared: representation_bytes(*element, extents)?, ..ClaimDemand::default() }
            }
            Self::Device { element, extents } => ClaimDemand { device: representation_bytes(*element, extents)?, ..ClaimDemand::default() },
            Self::Imported(scale) => ClaimDemand { device: scale.resident_bytes, ..ClaimDemand::default() },
            Self::Reserved { kind, contract, element, records, matrix, committed, carries } => {
                let held = seismic::ReservedRecords { record: reserved_record(record_bytes(*element, *matrix)?)?, carries: *carries };
                let charges = held.record.get().checked_mul(*committed)
                    .and_then(|bytes| reservation.charges(*contract, held, *records, bytes))
                    .ok_or_else(|| "reserved bank bytes overflow".to_owned())?;
                ClaimDemand::reservation(*kind, charges)
            }
        })
    }

    /// What the allocation is charged, a reservation backed as `reservation`
    /// backs its storage kind's (Seismic's rule).
    pub fn charged_bytes(&self, reservation: &seismic::Reservation) -> Result<u64, String> {
        self.storage_demand(reservation)?.checked_bytes()
            .ok_or_else(|| "bank allocation bytes overflow".to_owned())
    }
}

/// A bank's allocations by what holds each (`ExpertBankPlan::layout`): the
/// plan's of each, and the bank's tensors it allocates from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankLayout<T> {
    pub load: LoadAllocations<T>,
    pub groups: Vec<GroupAllocations<T>>,
    pub layers: Vec<LayerAllocations<T>>,
    /// Each staging pool (`ExpertBankPlan::staging_pools`).
    pub staging: Vec<T>,
    /// Each group's host tier record tensors, where the bank has a tier
    /// (`ExpertBankPlan::host`).
    pub host: Option<Vec<RecordTensors<T>>>,
}

/// The load's words: whether the group being encoded is resident, the tier
/// and seed lists' grids and staged copies, and its scratch rounds' grids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadAllocations<T> {
    pub resident: T,
    pub tier_grid: T,
    pub tier_staged: T,
    pub seed_grid: T,
    pub seed_staged: T,
    pub grids: T,
}

/// The placed parts' words (`BankPart::placed`) of a count of chain places
/// (`ExpertBankPlan::placed`): each place's step, every place's scratch
/// flags, and each streamed layer's group reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacedAllocations<T> {
    pub places: u64,
    /// Each place's step words, an allocation of its own: the first streamed
    /// layer's front of a place writes them (`routed_step`) while another
    /// place's fill, on the fill queue where it runs beside compute, reads
    /// its own, so no place's front orders after another's fill.
    pub step: Vec<T>,
    pub flags: T,
    pub layers: Vec<PlacedLayer<T>>,
}

/// A streamed layer's placed words: its group reads of every record's
/// places.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacedLayer<T> {
    pub group_reads: T,
}

impl<T> PlacedAllocations<T> {
    /// Every allocation.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.step
            .iter()
            .chain([&self.flags])
            .chain(self.layers.iter().map(|layer| &layer.group_reads))
    }

    /// The allocations of what `make` makes of each.
    pub fn try_map<U, E>(&self, mut make: impl FnMut(&T) -> Result<U, E>) -> Result<PlacedAllocations<U>, E> {
        Ok(PlacedAllocations {
            places: self.places,
            step: self.step.iter().map(&mut make).collect::<Result<Vec<_>, E>>()?,
            flags: make(&self.flags)?,
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    Ok(PlacedLayer { group_reads: make(&layer.group_reads)? })
                })
                .collect::<Result<Vec<_>, E>>()?,
        })
    }
}

/// A group's record tensor of each routed tensor, its map and commits, and
/// its move list (`routed_move`, [`ExpertBankPlan::move_records`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupAllocations<T> {
    pub records: RecordTensors<T>,
    /// Every member layer's map, end to end (`BankPart::Map`).
    pub map: T,
    pub commits: T,
    pub moves: T,
}

/// A layer's per-expert scale of each stack that has one, and its words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerAllocations<T> {
    pub expert_scales: RecordTensors<Option<T>>,
    /// The tier map, where the bank has a host tier (`BankPart::TierMap`).
    pub tier: Option<T>,
    /// The file and scratch kinds' tables (`BankPart::FileTable`,
    /// `ScratchTable`).
    pub file: T,
    pub scratch: T,
    pub rings: T,
    pub routing: T,
    /// The policy words (`BankPart::Policy`).
    pub policy: T,
    /// The CPU site's words and buffers, where the layer has a CPU site
    /// (`ExpertBankPlan::sited`).
    pub sited: Option<SiteAllocations<T>>,
    /// Its decode landing launch's landing list (`BankPart::LandingList`),
    /// and each scratch round's (`BankPart::RoundList`): the layer's own, so
    /// a step's round at the next layer, which launches before the host
    /// appends for it there, finds no entry of this layer's under its head.
    pub landing: T,
    pub rounds: Vec<T>,
    /// Its scratch request's tier and seed lists (`BankPart::TierList`,
    /// `SeedList`): each layer's own, so a list a step's copy at the layer
    /// launches on was written for the layer.
    pub tiered: T,
    pub seeds: T,
}

/// A streamed layer's CPU site's words and buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteAllocations<T> {
    pub table: T,
    pub rows: T,
    pub output: T,
}

impl<T> SiteAllocations<T> {
    fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.table, &self.rows, &self.output].into_iter()
    }

    fn try_map<U, E>(&self, make: &mut impl FnMut(&T) -> Result<U, E>) -> Result<SiteAllocations<U>, E> {
        Ok(SiteAllocations {
            table: make(&self.table)?,
            rows: make(&self.rows)?,
            output: make(&self.output)?,
        })
    }
}

impl<T> BankLayout<T> {
    /// Every allocation.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.owned().into_iter().map(|(_, allocation)| allocation)
    }

    /// Every allocation with the part of the bank that owns it.
    pub fn owned(&self) -> Vec<(&'static str, &T)> {
        let Self {
            load:
                LoadAllocations {
                    resident,
                    tier_grid,
                    tier_staged,
                    seed_grid,
                    seed_staged,
                    grids,
                },
            groups,
            layers,
            staging,
            host,
        } = self;
        let mut owned = vec![("load: resident words", resident)];
        owned.extend([
            ("load: tier grid", tier_grid),
            ("load: tier staged list", tier_staged),
            ("load: seed grid", seed_grid),
            ("load: seed staged list", seed_staged),
            ("load: grids", grids),
        ]);
        for GroupAllocations { records, map, commits, moves } in groups {
            owned.extend(records.iter().map(|(_, records)| ("group: records", records)));
            owned.extend([("group: map", map), ("group: commits", commits), ("group: moves", moves)]);
        }
        for LayerAllocations { expert_scales, tier, file, scratch, rings, routing, policy, sited, landing, rounds, tiered, seeds } in layers {
            owned.extend(expert_scales.iter().filter_map(|(_, scale)| scale.as_ref()).map(|scale| ("layer: expert scales", scale)));
            owned.extend(tier.iter().map(|tier| ("layer: tier words", tier)));
            owned.extend([
                ("layer: file words", file),
                ("layer: scratch words", scratch),
                ("layer: hint rings", rings),
                ("layer: routing records", routing),
                ("layer: policy", policy),
            ]);
            owned.extend(sited.iter().flat_map(SiteAllocations::iter).map(|site| ("layer: CPU site", site)));
            owned.push(("layer: landing list", landing));
            owned.extend(rounds.iter().map(|list| ("layer: round landing lists", list)));
            owned.extend([("layer: tier list", tiered), ("layer: seeds", seeds)]);
        }
        owned.extend(staging.iter().map(|pool| ("staging", pool)));
        owned.extend(host.iter().flatten().flat_map(|tier| tier.iter().map(|(_, records)| ("host tier", records))));
        owned
    }

    /// The layout of what `make` makes of each allocation.
    pub fn try_map<U, E>(&self, mut make: impl FnMut(&T) -> Result<U, E>) -> Result<BankLayout<U>, E> {
        fn each<T, U, E>(values: &RecordTensors<T>, make: &mut impl FnMut(&T) -> Result<U, E>) -> Result<RecordTensors<U>, E> {
            values.as_ref().try_map(|_, value| make(value))
        }
        fn optional<T, U, E>(
            values: &RecordTensors<Option<T>>,
            make: &mut impl FnMut(&T) -> Result<U, E>,
        ) -> Result<RecordTensors<Option<U>>, E> {
            values.as_ref().try_map(|_, value| value.as_ref().map(&mut *make).transpose())
        }
        let load = &self.load;
        let load = LoadAllocations {
            resident: make(&load.resident)?,
            tier_grid: make(&load.tier_grid)?,
            tier_staged: make(&load.tier_staged)?,
            seed_grid: make(&load.seed_grid)?,
            seed_staged: make(&load.seed_staged)?,
            grids: make(&load.grids)?,
        };
        let mut groups = Vec::with_capacity(self.groups.len());
        for group in &self.groups {
            groups.push(GroupAllocations {
                records: each(&group.records, &mut make)?,
                map: make(&group.map)?,
                commits: make(&group.commits)?,
                moves: make(&group.moves)?,
            });
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(LayerAllocations {
                expert_scales: optional(&layer.expert_scales, &mut make)?,
                tier: layer.tier.as_ref().map(&mut make).transpose()?,
                file: make(&layer.file)?,
                scratch: make(&layer.scratch)?,
                rings: make(&layer.rings)?,
                routing: make(&layer.routing)?,
                policy: make(&layer.policy)?,
                sited: layer.sited.as_ref().map(|sited| sited.try_map(&mut make)).transpose()?,
                landing: make(&layer.landing)?,
                rounds: layer.rounds.iter().map(&mut make).collect::<Result<Vec<_>, E>>()?,
                tiered: make(&layer.tiered)?,
                seeds: make(&layer.seeds)?,
            });
        }
        let staging = self.staging.iter().map(&mut make).collect::<Result<Vec<_>, E>>()?;
        let host = match &self.host {
            Some(tiers) => Some(tiers.iter().map(|tier| each(tier, &mut make)).collect::<Result<Vec<_>, E>>()?),
            None => None,
        };
        Ok(BankLayout {
            load,
            groups,
            layers,
            staging,
            host,
        })
    }
}

impl ExpertBankPlan {
    /// The element and extents of `part` of streamed layer `layer` (an
    /// index of [`Self::layers`]): what its graph ports declare and its
    /// allocation ([`Self::layout`]) holds, a placed part's for each of the
    /// bank's places, end to end.
    pub fn part_shape(&self, layer: usize, part: BankPart) -> Result<(Element, Vec<u64>), String> {
        self.shape_of(layer, part, || self.binding(layer))
    }

    /// Every part streamed layer `layer` holds ([`BankPart::every`]), with
    /// its element and extents ([`Self::part_shape`]).
    pub(crate) fn held_parts(&self, layer: usize) -> Result<Vec<(BankPart, Element, Vec<u64>)>, String> {
        // A layer's binding is derived once for all its parts.
        let binding = self.binding(layer)?;
        let mut parts = Vec::new();
        for part in BankPart::every(self.rounds()) {
            if let Some((element, extents)) = self.held_shape(layer, part, || Ok(binding))? {
                parts.push((part, element, extents));
            }
        }
        Ok(parts)
    }

    /// [`Self::part_shape`] with the layer's binding from `binding`, called
    /// only for the parts that read it: a placed part's shape is bound at
    /// every run of every step (`ExpertBank::placed`), and deriving the
    /// binding (its floor, which lays out the scratch) there costs
    /// milliseconds a step; [`Self::layout`], which reads it of every part,
    /// derives it once a layer. A part the layer does not hold is an error.
    fn shape_of(
        &self,
        layer: usize,
        part: BankPart,
        binding: impl Fn() -> Result<BankBinding, String>,
    ) -> Result<(Element, Vec<u64>), String> {
        self.held_shape(layer, part, binding)?
            .ok_or_else(|| format!("bank part {part:?} is not held by streamed layer {layer}"))
    }

    /// [`Self::shape_of`], none of a part the layer does not hold: a host
    /// tier's of a bank without one, a CPU site's of a layer without one,
    /// staging where the device places, a round's landing list past its rounds,
    /// a record part of a kind its record lacks.
    fn held_shape(
        &self,
        layer: usize,
        part: BankPart,
        binding: impl Fn() -> Result<BankBinding, String>,
    ) -> Result<Option<(Element, Vec<u64>)>, String> {
        let words = |count: u64| Ok(Some((Element::i32(), vec![count])));
        match part {
            BankPart::Step => words(constants::STEP_WORDS),
            BankPart::SlotMap => Ok(Some((Element::i32(), vec![self.layers()[layer].experts, constants::MAP_WORDS]))),
            BankPart::GroupReads => words(self.layers()[layer].experts),
            // The load's flags of each place, which every layer's request
            // counts into in turn: the widest layer's.
            BankPart::ScratchFlags => words(self.widest().get()),
            BankPart::FileTable => Ok(Some((Element::i32(), binding()?.words.file().extents().to_vec()))),
            BankPart::ScratchTable => Ok(Some((Element::i32(), binding()?.words.scratch().extents().to_vec()))),
            BankPart::Rings => Ok(Some((Element::i32(), vec![self.ahead(layer), binding()?.words.ring_words()]))),
            BankPart::Routing => {
                let words = binding()?.words;
                Ok(Some((Element::i32(), vec![words.flight, words.routing_words()])))
            }
            BankPart::Map => Ok(Some((Element::i32(), vec![self.group_experts(self.layers()[layer].group), constants::MAP_WORDS]))),
            BankPart::TierMap | BankPart::Tier { .. } if self.tier_reservation().is_none() => Ok(None),
            BankPart::TierMap => words(self.layers()[layer].experts),
            BankPart::SiteTable | BankPart::SiteRows | BankPart::SiteOutput if !self.sited(layer) => Ok(None),
            BankPart::SiteTable => Ok(Some((Element::i32(), binding()?.words.site().extents().to_vec()))),
            // A slot's rows: every decode class's, so a slot's row m lies
            // where it does for every class (the experts' width H, of the
            // down matrix `[H, F]`); and its answer's, the weighted sum of
            // the CPU's choices of each row.
            BankPart::SiteRows | BankPart::SiteOutput => {
                let binding = binding()?;
                Ok(Some((Element::f32(), vec![binding.words.site().slots(), constants::DECODE_ROWS, binding.record.down.matrix[0]])))
            }
            BankPart::Tier { kind } => {
                let record = binding()?.record;
                Ok(record
                    .get(kind)
                    .map(|tensor| tensor.resident_tensor(self.host_extent(self.layers()[layer].group)))
                    .map(|(element, extents)| (element, extents.to_vec())))
            }
            BankPart::Resident => words(1),
            BankPart::Policy => words(constants::POLICY_WORDS),
            BankPart::Commits => words(binding()?.records),
            BankPart::Conveyor { kind } => {
                let binding = binding()?;
                Ok(binding.record.get(kind).map(|tensor| tensor.stored_tensor(binding.conveyor)).map(|(element, extents)| (element, extents.to_vec())))
            }
            BankPart::Scratch { kind } => {
                let binding = binding()?;
                Ok(binding.record.get(kind).map(|tensor| tensor.resident_tensor(binding.experts)).map(|(element, extents)| (element, extents.to_vec())))
            }
            BankPart::Staging { .. } if self.placer() != Placer::Bank => Ok(None),
            BankPart::Staging { kind } => {
                let binding = binding()?;
                Ok(binding.record.get(kind).map(|tensor| tensor.stored_tensor(binding.staging)).map(|(element, extents)| (element, extents.to_vec())))
            }
            BankPart::LandingList => words(magnitude_kernels::landing_extent(self.choices())),
            BankPart::RoundList { round } if u64::from(round) < self.rounds() => words(self.round_list_words()),
            BankPart::RoundList { .. } => Ok(None),
            BankPart::TierList | BankPart::SeedList | BankPart::TierStaged | BankPart::SeedStaged => words(self.expert_list_words()),
            BankPart::TierGrid | BankPart::SeedGrid => words(3),
            BankPart::ScratchGrids => Ok(Some((Element::i32(), vec![self.rounds(), 3]))),
        }
    }

    /// Every allocation the bank holds whatever places a group holds, by
    /// what holds it: the one list `ExpertBank::open` allocates and the plan
    /// charges ([`Self::bytes`]) beside the floor's placed words
    /// ([`Self::placed`]). Each word part the host writes or reads is host-shared (each
    /// layer's scratch table as protocol words, whose cross-queue order is
    /// the exchange's), the
    /// device's own grids and staged lists its own; each group's record tensors are
    /// reserved for its ceiling, its committed records backed, which a
    /// prompt group's scratch and, on unified memory, its conveyor are views
    /// of the lender's (no expert memory is reserved for prefill or decode
    /// alone); each layer's per-expert scales are imported as its resident
    /// stacks plan them; the staging pools are host-shared; a host tier's
    /// record tensors, where its conveyor lies beside a discrete device, are
    /// reserved in host storage for every expert of their group, their
    /// floor committed ([`Self::host_records`] at no host slots): the tier
    /// opens at its floor, empty, and the bank backs it to its load-time
    /// size ([`Self::host_slots`]) beside its fill, each slot filled as it
    /// joins (`ExpertBank::back_host`).
    pub fn layout(&self) -> Result<BankLayout<BankAllocation>, String> {
        let bindings = (0..self.layers().len()).map(|layer| self.binding(layer)).collect::<Result<Vec<_>, _>>()?;
        let shape = |layer: usize, part: BankPart| self.shape_of(layer, part, || Ok(bindings[layer]));
        let words = |layer: usize, part: BankPart| {
            let (element, extents) = shape(layer, part)?;
            Ok::<_, String>(BankAllocation::Words { element, extents })
        };
        // The words only the device writes and reads (the launch grids
        // the device writes, the lists it stages): the device's own, so its
        // many readers read no host memory.
        let device = |layer: usize, part: BankPart| {
            let (element, extents) = shape(layer, part)?;
            Ok::<_, String>(BankAllocation::Device { element, extents })
        };
        let groups = self
            .groups()
            .iter()
            .enumerate()
            .map(|(index, group)| {
                let first = group.layers[0];
                Ok(GroupAllocations {
                    records: group.record.try_map(|kind, tensor| {
                        Ok::<_, String>(BankAllocation::Reserved {
                            kind: crate::planning::BankTier::Bank,
                            contract: self.host().contract(crate::planning::BankTier::Bank),
                            element: tensor.resident,
                            records: self.records(index),
                            matrix: tensor.matrix,
                            committed: self.committed_records(index),
                            carries: self.reserved_records(index, kind, tensor, crate::planning::BankTier::Bank)?.carries,
                        })
                    })?,
                    map: words(first, BankPart::Map)?,
                    commits: words(first, BankPart::Commits)?,
                    moves: BankAllocation::Words {
                        element: Element::i32(),
                        extents: vec![list_extent(self.move_records(index))],
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let layers = self
            .layers()
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                Ok(LayerAllocations {
                    expert_scales: layer
                        .resident
                        .as_ref()
                        .map(|_, plan| plan.scale.clone().map(BankAllocation::Imported)),
                    tier: self.tier_reservation().map(|_| words(index, BankPart::TierMap)).transpose()?,
                    file: words(index, BankPart::FileTable)?,
                    scratch: {
                        let (element, extents) = shape(index, BankPart::ScratchTable)?;
                        BankAllocation::Protocol { element, extents }
                    },
                    rings: words(index, BankPart::Rings)?,
                    routing: words(index, BankPart::Routing)?,
                    policy: words(index, BankPart::Policy)?,
                    sited: self
                        .sited(index)
                        .then(|| {
                            Ok::<_, String>(SiteAllocations {
                                table: words(index, BankPart::SiteTable)?,
                                rows: words(index, BankPart::SiteRows)?,
                                output: words(index, BankPart::SiteOutput)?,
                            })
                        })
                        .transpose()?,
                    landing: words(index, BankPart::LandingList)?,
                    rounds: (0..self.rounds())
                        .map(|round| {
                            let round = u8::try_from(round).map_err(|_| "scratch rounds exceed u8")?;
                            words(index, BankPart::RoundList { round })
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                    tiered: words(index, BankPart::TierList)?,
                    seeds: words(index, BankPart::SeedList)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let load = LoadAllocations {
            resident: words(0, BankPart::Resident)?,
            tier_grid: device(0, BankPart::TierGrid)?,
            tier_staged: device(0, BankPart::TierStaged)?,
            seed_grid: device(0, BankPart::SeedGrid)?,
            seed_staged: device(0, BankPart::SeedStaged)?,
            grids: device(0, BankPart::ScratchGrids)?,
        };
        let staging = self
            .staging_pools()
            .0
            .iter()
            .map(|pool| BankAllocation::Shared {
                element: pool.source,
                extents: vec![self.staging_records(), pool.matrix[0], pool.matrix[1]],
            })
            .collect();
        let host = self.tier_reservation().map(|_| {
            self.groups()
                .iter()
                .enumerate()
                .map(|(index, group)| {
                    group.record.try_map(|kind, tensor| {
                        Ok::<_, String>(BankAllocation::Reserved {
                            kind: crate::planning::BankTier::Host,
                            contract: self.host().contract(crate::planning::BankTier::Host),
                            element: tensor.resident,
                            records: self.host_extent(index),
                            matrix: tensor.matrix,
                            committed: self.host_records(index, 0),
                            carries: self.reserved_records(index, kind, tensor, crate::planning::BankTier::Host)?.carries,
                        })
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        }).transpose()?;
        Ok(BankLayout {
            load,
            groups,
            layers,
            staging,
            host,
        })
    }

    /// The words from one instance of placed part `part` of `layer`
    /// (`BankPart::placed`, each a run of i32 words) to the next: its
    /// planned words rounded up to whole u64s, so every instance begins
    /// 8-byte aligned, as every host-shared word view the host accesses
    /// (`experts::Words`) is, whatever its words.
    pub fn place_stride(&self, layer: usize, part: BankPart) -> Result<u64, String> {
        if !part.placed() {
            return Err(format!("bank part {part:?} is not placed"));
        }
        let (_, extents) = self.part_shape(layer, part)?;
        Ok(extents.iter().product::<u64>().next_multiple_of(2))
    }

    /// The placed parts' words (`BankPart::placed`) of `places` chain places,
    /// at most the bank's: each instance the words of the part's planned
    /// shape at its stride ([`Self::place_stride`]), a group reads' each
    /// record's places, end to end, host-shared; each place's step words
    /// apart (`PlacedAllocations::step`). The load holds the floor's
    /// one place, and a group the places it holds
    /// (`ExpertBank::hold_places`).
    pub fn placed(&self, places: std::num::NonZeroU64) -> Result<PlacedAllocations<(Element, Vec<u64>)>, String> {
        if places.get() > self.places {
            return Err(format!("{places} chain places of a bank of {}", self.places));
        }
        let words = |layer: usize, part: BankPart| {
            let (element, _) = self.part_shape(layer, part)?;
            let words = self
                .place_stride(layer, part)?
                .checked_mul(part.instances(self.flights, places.get()))
                .ok_or("placed bank words overflow")?;
            Ok::<_, String>((element, vec![words]))
        };
        Ok(PlacedAllocations {
            places: places.get(),
            step: {
                let (element, _) = self.part_shape(0, BankPart::Step)?;
                let place = (element, vec![self.place_stride(0, BankPart::Step)?]);
                vec![place; usize::try_from(places.get()).map_err(|_| "chain places exceed the host")?]
            },
            flags: words(0, BankPart::ScratchFlags)?,
            layers: (0..self.layers().len())
                .map(|layer| Ok(PlacedLayer { group_reads: words(layer, BankPart::GroupReads)? }))
                .collect::<Result<Vec<_>, String>>()?,
        })
    }

    /// The placed parts' words of `places` chain places as the bank
    /// allocates them ([`Self::placed`]): host-shared.
    pub(super) fn placed_allocations(&self, places: std::num::NonZeroU64) -> Result<Vec<BankAllocation>, String> {
        let placed = self.placed(places)?;
        Ok(placed
            .iter()
            .map(|(element, extents)| BankAllocation::Words { element: *element, extents: extents.clone() })
            .collect())
    }

    /// The bytes the placed parts' words of `places` chain places are
    /// charged ([`Self::placed`]).
    pub fn placed_bytes(&self, places: std::num::NonZeroU64) -> Result<u64, String> {
        self.placed_demand(places)?.checked_bytes().ok_or_else(|| "placed bank bytes overflow".to_owned())
    }

    /// The physical demand of the placed parts' words of `places` places.
    pub fn placed_demand(&self, places: std::num::NonZeroU64) -> Result<ClaimDemand, String> {
        self.demand(self.placed_allocations(places)?.iter())
    }
}
