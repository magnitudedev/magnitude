pub(super) use super::super::*;
pub(super) use super::super::scratch::conveyor_forms;
pub(super) use super::super::steps::{FailedRows, StepFailure};
pub(super) use super::super::telemetry::GroupFacts;
pub(super) use crate::experts::exchange::Answer;
pub(super) use crate::experts::{ExchangeTable, Words};
pub(super) use crate::experts::conveyor::{Carrier, LentPool, Lending};
pub(super) use crate::experts::landing::TierTensor;
pub(super) use crate::experts::map::{LandingMap, MapView, StampedMap};
pub(super) use crate::experts::source::{ExpertSource, Outcome, Read, Records, Ticket};
pub(super) use crate::operation::LaunchWork;
pub(super) use crate::planning::{Placer, RegionWords};
pub(super) use magnitude_kernels::constants;
pub(super) use seismic::{BackendName, Element, Tensor};
pub(super) use std::time::{Duration, Instant};

/// A file request's choice `choice`'s word: the route's request.
pub(super) fn choice_word(choice: u64) -> u64 {
    magnitude_kernels::SlotTable::file_entry(choice) + constants::FILE_CHOICE
}

/// A file request's choice `choice`'s record word: the host's answer.
pub(super) fn record_word(choice: u64) -> u64 {
    magnitude_kernels::SlotTable::file_entry(choice) + constants::FILE_RECORD
}

pub(super) const EXPERTS: usize = 8;

/// Bytes of one expert's record of each routed tensor.
pub(super) const RECORD: u64 = 16;

/// The tests' decode request: one record of one stored range a choice.
fn urgent() -> seismic_io::Urgent {
    seismic_io::Urgent { parts: vec![RECORD], records: 1 }
}

pub(super) const STAGING: u32 = 4;

/// The host tier records of a harness of [`STAGING`] staging records: its
/// reserve (the staging records) and as many again, which the tier keeps.
pub(super) const TIER: u32 = 2 * STAGING;

/// The harness's chain places.
pub(super) const PLACES: u64 = 2;

pub(super) const WORDS: RegionWords = RegionWords {
    choices: 4,
    rounds: 4,
    flight: crate::FlightDepth::PIPELINED.steps(PLACES),
    sites: crate::FlightDepth::PIPELINED.steps(1),
};

/// A scratch round's records.
pub(super) const ROUND_RECORDS: u64 = 2;

/// The lane of scratch round `index` of [`WORDS`].
pub(super) fn round(index: u64) -> u64 {
    assert!(index < WORDS.rounds, "a round of the table");
    scratch::round_lane(index)
}

/// The last round's lane, whose answer is the request's verdict.
fn last_round() -> u64 {
    round(WORDS.rounds - 1)
}

/// `slots`' table of the exchange on the CPU device, beating with `steps`'
/// pulse.
fn cpu_slots(table: magnitude_kernels::SlotTable, steps: &Steps) -> ExchangeTable {
    ExchangeTable::new(Tensor::shared(&cpu(), Element::i32(), &[table.words()]).unwrap(), table, steps.pulse()).unwrap()
}

/// Spin until lane `lane` of `step`'s slot of `slots` is answered, by the
/// step or its failure.
pub(super) fn until_answered(slots: &ExchangeTable, step: Step, lane: u64) -> Answer {
    awaited(&format!("step {step}'s lane {lane} answered"), || slots.answer_of(step, lane))
}

/// A watcher over one layer of `EXPERTS` experts in a group of `records`
/// records beside its host tier, decode requests of 4 choices, scratch
/// rounds of 2 records, and a model file holding `stored` experts of each
/// routed tensor; with the words a test plays the device through.
pub(super) struct Harness {
    pub(super) watcher: Watcher,
    /// Each layer's barrier's convert words, where the host places.
    pub(super) device_words: Vec<Option<Arc<crate::experts::landing::DeviceWords>>>,
    /// The first layer's words, and every layer's: map, file and scratch
    /// tables, rings, routing records and policy words.
    pub(super) layer: LayerWords,
    pub(super) layers: Vec<LayerWords>,
    /// Each layer's group reads (`[records, places, EXPERTS]`).
    pub(super) reads: Vec<Words>,
    /// The group's commits.
    commits: Words,
    /// Scratch flags and convert lists.
    pub(super) load: (Words, Vec<Words>),
    /// Each layer's tier list: a request's experts the host tier keeps
    /// whole, which the device's tier copy reads (the harness plays it at
    /// round 0).
    pub(super) tiered: Vec<Words>,
    /// Each layer's seed list.
    pub(super) seeds: Vec<Words>,
    /// The second chain place's scratch flags (the first's are `load.0`).
    pub(super) second: Words,
    staging: Tensor,
    conveyor: Tensor,
    /// The conveyor's records of each tensor.
    conveyor_records: u64,
    pub(super) staging_records: u32,
    pub(super) file: std::path::PathBuf,
    /// What the device shares with the host tier.
    pub(super) host: HostWords,
    /// Each tensor's experts as the device's pass reads them, every
    /// layer's in turn: slot `e` its expert `e`, which the tier copy plays
    /// into.
    pub(super) scratch: Vec<Tensor>,
    /// The device's transfer queue promotions copy by.
    pub(super) copies: crate::experts::copies::Copies,
    /// The first bank record the seed list names records from
    /// (`ExpertBankPlan::seeded_from`): past the lent ones, the bank's
    /// beside a host tier, the one store's where the host places.
    seeded_from: usize,
    /// Its share of the read engine's pool, none where its test holds the
    /// pool alone ([`alone`]).
    _pool: Option<std::sync::RwLockReadGuard<'static, ()>>,
}

/// The read engine's pool is the process's one
/// (`seismic_io`): reads a test queues ahead in it
/// ([`Ahead`]) delay every other harness's demand reads, whose times refit
/// that harness's read line and so the price its hints are admitted by
/// (`telemetry::pays`). A test that queues reads ahead holds the pool
/// alone; every other harness shares it.
static POOL: std::sync::RwLock<()> = std::sync::RwLock::new(());

thread_local! {
    /// Whether this thread's test holds [`POOL`] alone.
    static ALONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`POOL`] held alone by this thread's test, until dropped.
pub(super) struct Alone {
    _pool: std::sync::RwLockWriteGuard<'static, ()>,
}

/// Hold the read engine's pool alone for this test ([`POOL`]): taken before
/// its harness, which then shares nothing.
pub(super) fn alone() -> Alone {
    let pool = POOL.write().unwrap_or_else(std::sync::PoisonError::into_inner);
    ALONE.set(true);
    Alone { _pool: pool }
}

impl Drop for Alone {
    fn drop(&mut self) {
        ALONE.set(false);
    }
}

/// The host tier's records and words the harness plays the device
/// through: each routed tensor's tier records, each layer's tier map, and
/// each routed tensor's bank records, which promotions copy into.
pub(super) struct HostWords {
    /// Each routed tensor's tier records, reserved as a bank's are and every
    /// one backed, so the host addresses only views of them, and their
    /// resizers, which outlive them.
    tier: Vec<Tensor>,
    _resizers: Vec<seismic::Resizer>,
    pub(super) maps: Vec<Words>,
    bank: Vec<Tensor>,
}

impl HostWords {
    /// Each routed tensor's first `records` tier records as the host
    /// addresses them: a view pinning them, as a bank's are.
    pub(super) fn views(&self, records: u32) -> Vec<Records> {
        let records = u64::from(records);
        self.tier
            .iter()
            .map(|tensor| Records::new(tensor.region(Element::i32(), &[records, RECORD / 4], 0).unwrap(), RECORD, records).unwrap())
            .collect()
    }
}

type LayerWords = (MapView, ExchangeTable, ExchangeTable, Words, Words, Words);

impl Drop for Harness {
    fn drop(&mut self) {
        std::fs::remove_file(&self.file).unwrap();
    }
}

pub(super) fn harness(name: &str, stored: usize, records: u64) -> Harness {
    layered(name, stored, records, 1, STAGING)
}

/// [`harness`] of `layers` layers, one group, `staging_records` staging
/// records and twice as many tier records, short of every expert of its
/// layers: a tier its misses and hints fill.
pub(super) fn layered(
    name: &str,
    stored: usize,
    records: u64,
    layers: usize,
    staging_records: u32,
) -> Harness {
    layered_over(name, stored, records, layers, staging_records, DECODE)
}

/// [`layered`] conveying over `conveying`.
pub(super) fn layered_over(
    name: &str,
    stored: usize,
    records: u64,
    layers: usize,
    staging_records: u32,
    conveying: Conveying,
) -> Harness {
    let tier = (2 * staging_records).min((EXPERTS * layers) as u32 - 1);
    lending(name, stored, records, layers, staging_records, conveying, tier, Placer::Tier, false)
}

/// [`harness`] that serves prompt groups.
pub(super) fn prompted(name: &str, stored: usize, records: u64) -> Harness {
    layered_over(name, stored, records, 1, STAGING, WHOLE)
}

/// [`hosted_layers`] that serves prompt groups.
pub(super) fn hosted_prompted(name: &str, records: u64, layers: usize, tier: u32) -> Harness {
    lending(name, EXPERTS, records, layers, STAGING, WHOLE, tier, Placer::Tier, false)
}

/// [`harness`] of one layer beside a host tier of `tier` records.
pub(super) fn hosted(name: &str, records: u64, tier: u32) -> Harness {
    hosted_layers(name, records, 1, tier)
}

/// [`hosted`] of `layers` layers, one group.
pub(super) fn hosted_layers(name: &str, records: u64, layers: usize, tier: u32) -> Harness {
    lending(name, EXPERTS, records, layers, STAGING, DECODE, tier, Placer::Tier, false)
}

/// [`hosted`] whose watcher promotes what its ranking keeps into the bank
/// (`promotions`); every other harness's tier serves its misses alone, so
/// what it holds is what its steps place.
pub(super) fn promoting(name: &str, records: u64, tier: u32) -> Harness {
    lending(name, EXPERTS, records, 1, STAGING, DECODE, tier, Placer::Tier, true)
}

/// A harness's conveyor of `conveyed` records of each form, which the host
/// reads into and the rounds convert from, lying on its group's bottom `lent`
/// records of `bytes` each, beneath the tier the test sizes.
#[derive(Clone, Copy)]
pub(super) struct Conveying {
    pub(super) conveyed: u64,
    pub(super) lent: usize,
    pub(super) bytes: u64,
    /// Whether the host addresses the scratch, as on unified memory: its
    /// slots no pass reads carry experts too (`Scratch::lendable_slot`).
    pub(super) scratch: bool,
}

impl Conveying {
    /// A conveyor record on each lent record.
    pub(super) const fn of(conveyed: u64) -> Self {
        Self { conveyed, lent: conveyed as usize, bytes: RECORD, scratch: false }
    }

    /// The conveyor, the host addressing the scratch.
    pub(super) const fn lending_scratch(self) -> Self {
        Self { scratch: true, ..self }
    }
}

/// A conveyor of a record of every expert.
pub(super) const WHOLE: Conveying = Conveying::of(EXPERTS as u64);

/// What a harness that serves no prompt group holds: no lent record, so its
/// tier is exactly what the test sizes.
pub(super) const DECODE: Conveying = Conveying { conveyed: 1, lent: 0, bytes: RECORD, scratch: false };

/// The CPU device the harness's words and records live on.
pub(super) fn cpu() -> seismic::Device {
    crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), BackendName::Cpu).unwrap()
}

/// `count` words on `device`, each `initial`.
pub(super) fn cpu_words(device: &seismic::Device, count: u64, initial: i32) -> Words {
    Words::new(Tensor::shared(device, Element::i32(), &[count]).unwrap(), count, initial).unwrap()
}

/// [`layered`], its conveyor carried as `conveying` says, beside a host
/// tier of `tier` records; lent records each of `bytes` bytes as the
/// conveyor's records lie on them: conveyor record `e` of each pool lies
/// on lent record `e · RECORD / bytes`. Beside a host tier its watcher
/// promotes into the bank where `promotes` says.
#[allow(clippy::too_many_arguments)]
pub(super) fn lending(
    name: &str,
    stored: usize,
    records: u64,
    layers: usize,
    staging_records: u32,
    conveying: Conveying,
    tier: u32,
    placer: Placer,
    promotes: bool,
) -> Harness {
    lending_bank(name, stored, records, layers, staging_records, conveying, tier, placer, promotes, 0, false)
}

/// [`lending`], beside a host tier the scratch's lender lending a prompt
/// group its bank's bottom `bank_lent` records (the fixture's scratch tier
/// still lies apart, so only their claims are played); its bank imported
/// `whole` (each expert at its home) where that says.
#[allow(clippy::too_many_arguments)]
pub(super) fn lending_bank(
    name: &str,
    stored: usize,
    records: u64,
    layers: usize,
    staging_records: u32,
    conveying: Conveying,
    tier: u32,
    placer: Placer,
    promotes: bool,
    bank_lent: usize,
    whole: bool,
) -> Harness {
    let device = cpu();
    let words = |count: u64, initial| cpu_words(&device, count, initial);
    let file =
        std::env::temp_dir().join(format!("expert-watcher-{name}-{}", std::process::id()));
    // Tensor `kind`'s expert `e` holds bytes 100 kind + e.
    let bytes = (0..3)
        .flat_map(|kind| {
            (0..stored).flat_map(move |expert| [(100 * kind + expert) as u8; RECORD as usize])
        })
        .collect::<Vec<_>>();
    std::fs::write(&file, bytes).unwrap();
    let staging =
        Tensor::shared(&device, Element::i32(), &[u64::from(staging_records) * 3 * 4]).unwrap();
    let stored_tensor = |kind: usize| StoredTensor {
        file: 0,
        base: kind as u64 * stored as u64 * RECORD,
        pitch: RECORD,
        bytes: RECORD,
        pool: kind,
    };
    // The steps, under whose lock every map entry is written, and whose
    // pulse every table beats with: driven, as the harness plays the
    // device, so no beat comes late but a test's.
    let steps = Arc::new(Steps::new(crate::FlightDepth::PIPELINED, PLACES, crate::experts::exchange::Pulse::driven()));
    // Every layer reads the same experts; layer `l` hints the streamed
    // layers after it, a ring row a lead (`ExpertBankPlan::ahead`).
    // Each layer's map's one writer: the bank's records beside a host tier,
    // the one store's landing map where the host places.
    let mut maps = (0..layers).map(|_| StampedMap::new(words(EXPERTS as u64 * constants::MAP_WORDS, 0))).collect::<Vec<_>>();
    let all = (0..layers)
        .map(|index| {
            let ahead = crate::planning::HINT_LEADS.min((layers - 1 - index) as u64);
            (
                maps[index].view().clone(),
                cpu_slots(WORDS.file(), &steps),
                cpu_slots(WORDS.scratch(), &steps),
                words(ahead * WORDS.ring_words(), 0),
                words(WORDS.flight * WORDS.routing_words(), 0),
                words(constants::POLICY_WORDS, 0),
            )
        })
        .collect::<Vec<_>>();
    let layer = all[0].clone();
    // Each layer's group reads, of both chain places.
    let reads = (0..layers)
        .map(|_| words(crate::FlightDepth::PIPELINED.steps(PLACES) * EXPERTS as u64, 0))
        .collect::<Vec<_>>();
    let second = words(EXPERTS as u64, 0);
    // The records the conveyor's lending lends it, the host tier's (or the
    // one store's) bottom ones beneath the `tier` the test sizes: conveyor
    // record `e` lies on lent record `e · RECORD / bytes`.
    let Conveying { conveyed: conveyor_records, lent, bytes, scratch: addressed } = conveying;
    let tier = tier + lent as u32;
    // The slots the test sizes: the bank's records (its scratch lies apart,
    // the fixture's scratch tier).
    let slots = records as usize / layers;
    // Its commits span the records seeds lie in: where the host places, the
    // one store's.
    let bank_records = match placer { Placer::Tier => records, Placer::Bank => u64::from(tier).max(records) };
    let commits = words(bank_records, 0);
    // The host tier: each tensor's tier records, reserved for every expert
    // of every layer or its records if more (as a load reserves the most it
    // grows to), in the bank's resident form, which here is the stored one;
    // each layer's tier map, none kept; and each tensor's bank records.
    let reserved = u64::from(tier).max(EXPERTS as u64 * layers as u64);
    let reserve = |records: u64| Tensor::reserved_in(crate::planning::BankTier::Bank.contract_on(&device), &device, Element::i32(), &[records, RECORD / 4], records, None).unwrap();
    let (tensors, _resizers) = (0..3).map(|_| reserve(reserved)).unzip::<_, _, Vec<_>, Vec<_>>();
    let (bank, bank_resizers) = (0..3).map(|_| reserve(records)).unzip::<_, _, Vec<_>, Vec<_>>();
    let host = HostWords {
        tier: tensors,
        _resizers: _resizers.into_iter().chain(bank_resizers).collect(),
        maps: (0..layers).map(|_| words(EXPERTS as u64, -1)).collect(),
        bank,
    };
    // Beside a host tier, what promotions copy with: the device's transfers
    // (the CPU's, complete as queued), from each tensor's tier records into
    // its bank records.
    let copies = crate::experts::copies::Copies::new(seismic::Transfer::new(&device).unwrap(), steps.events().clone());
    let copier = (placer == Placer::Tier && promotes).then(|| super::Copier {
        copies: copies.clone(),
        groups: vec![
            host.tier
                .iter()
                .zip(&host.bank)
                .map(|(tier, bank)| super::CopyTensor {
                    tier: tier.clone(),
                    tier_at: tier.record_placement().unwrap(),
                    bank: bank.clone(),
                    bank_at: bank.record_placement().unwrap(),
                })
                .collect(),
        ],
    });
    let tier_tensors = host
        .views(tier)
        .into_iter()
        .map(|records| TierTensor::new(records, Element::i32(), Element::i32(), [1, RECORD / 4], RECORD).unwrap())
        .collect::<Vec<_>>();
    let load = (
        words(EXPERTS as u64, 0),
        (0..WORDS.rounds).map(|_| words(magnitude_kernels::list_extent(2), 0)).collect::<Vec<_>>(),
    );
    let lists = || (0..layers).map(|_| words(magnitude_kernels::list_extent(EXPERTS as u64), 0)).collect::<Vec<_>>();
    let (tiered, seeds) = (lists(), lists());
    // Each tensor's conveyor: the bytes of the lent records it lies on,
    // which the harness addresses apart from the tier's.
    let conveyor = Tensor::shared(&device, Element::i32(), &[conveyor_records * 3 * 4]).unwrap();
    let view = |tensor: &Tensor, kind: u64, records: u64| {
        let view = tensor
            .region(Element::i32(), &[records, RECORD / 4], kind * records * RECORD)
            .unwrap();
        Records::new(view, RECORD, records).unwrap()
    };
    let pools = (0..3)
        .map(|kind| Pool {
            staging: view(&staging, kind, u64::from(staging_records)),
        })
        .collect::<Vec<_>>();
    let conveyed = || (0..3).map(|kind| view(&conveyor, kind, conveyor_records)).collect::<Vec<_>>();
    let lending = |records, bytes, pools: &[Records]| Lending {
        group: 0,
        records,
        pools: pools
            .iter()
            .map(|pool| LentPool { record: bytes, offset: 0, conveyor: pool.placement() })
            .collect(),
    };
    // Each tensor's experts as the device's pass reads them, by expert:
    // the scratch slots the harness's tier copy plays into.
    let scratch = (0..3).map(|_| Tensor::shared(&device, Element::i32(), &[EXPERTS as u64, RECORD / 4]).unwrap()).collect::<Vec<_>>();
    let conveyors = conveyed();
    let carrier = Carrier {
        records: conveyor_records as usize,
        slots: if addressed { EXPERTS } else { 0 },
        lending: lending(lent, bytes, &conveyors),
        pools: conveyors,
    };
    // Who places, with the maps each placement writes: beside a host tier
    // the bank's records write their maps and the host records the tier's,
    // the lender lending its bank records; where the host lands into the one
    // store, its host records write the map.
    let placement = match placer {
        Placer::Tier => crate::experts::directory::Placement::Tier {
            bank_maps: std::mem::take(&mut maps),
            tier_maps: host.maps.iter().map(|words| LandingMap::Tier(words.clone())).collect(),
            lent: (0, bank_lent),
        },
        Placer::Bank => crate::experts::directory::Placement::Bank {
            landing_maps: std::mem::take(&mut maps)
                .into_iter()
                .map(|map| LandingMap::Bank { map: Arc::new(map), steps: steps.clone() })
                .collect(),
        },
    };
    // Each layer's barrier's convert words, where the host places.
    let device_words = all
        .iter()
        .map(|_| {
            (placer == Placer::Bank).then(|| {
                Arc::new(crate::experts::landing::DeviceWords {
                    list: words(magnitude_kernels::list_extent(EXPERTS as u64), 0),
                    grids: words(constants::ROUND_GRIDS * 3, 0),
                    rows_groups: 3,
                    columns_groups: 4,
                    bounds: crate::experts::ListBounds { from: u64::from(staging_records), to: bank_records, lent: 0 },
                })
            })
        })
        .collect::<Vec<_>>();
    // A hung test reports every owner and every thread's stack on SIGUSR1.
    crate::diagnostics::install();
    let watcher = Watcher::start(WatcherParts {
        layers: all
            .iter()
            .enumerate()
            .map(|(index, layer)| Layer {
                group: 0,
                experts: EXPERTS,
                map: layer.0.clone(),
                file: layer.1.clone(),
                scratch: layer.2.clone(),
                rings: layer.3.clone(),
                // Each ring row hints the layer its lead is ahead.
                hints: (0..layer.3.len() / WORDS.ring_words() as usize).map(|lead| index + 1 + lead).collect(),
                routing: layer.4.clone(),
                policy: layer.5.clone(),
                site: None,
                stored: (0..3).map(stored_tensor).collect(),
                slots: addressed.then(|| scratch.iter().map(|slots| Records::new(slots.clone(), RECORD, EXPERTS as u64).unwrap()).collect()),
                scratch_form: vec![0; 3],
                landing: device_words[index].clone(),
                lists: ListRecords {
                    bank: bank_records,
                    // `BankBinding::tier`: the one store has no tier to copy from.
                    tier: match placer {
                        Placer::Tier => reserved,
                        Placer::Bank => 0,
                    },
                    conveyor: conveyor_records,
                },
                tiered: tiered[index].clone(),
                seeds: seeds[index].clone(),
            })
            .collect(),
        group_reads: reads.iter().map(|reads| PlacedLayerWords { group_reads: reads.clone(), stride: EXPERTS }).collect(),
        groups: vec![Group {
            commits: commits.clone(),
            tier: Some(tier_tensors),
            records: match placer {
                Placer::Tier => records as usize,
                Placer::Bank => 0,
            },
        }],
        load: LoadWords {
            flags: vec![load.0.clone(), second.clone()],
            lists: load.1.clone(),
        },
        pools,
        extents: Extents {
            widest: EXPERTS,
            round_records: ROUND_RECORDS as usize,
            words: WORDS,
            // One choice a row.
            selected: 1,
            demand: WORDS.choices as usize,
            // The tier's reserve: one record for each staging record, so the
            // tests exercise the refill's unmapping and the reads ahead
            // beside a request at any size (a load's is its reads-ahead
            // window, `ExpertBankPlan::tier_reserve`).
            reserve: staging_records as usize,
            placer,
        },
        slots: Arc::new(std::sync::atomic::AtomicU64::new(slots as u64)),
        whole,
        staging: staging_records,
        host: crate::experts::directory::HostTier { records: vec![tier], lent: vec![lent as u32], placement },
        carrier,
        source: ExpertSource::open(std::slice::from_ref(&file), &urgent()).unwrap(),
        // The landings' executor, of no site layer.
        compute: crate::experts::HostCompute::start(Vec::new(), &crate::experts::ComputeThreads::unplaced(vec![vec![0, 1]])).unwrap(),
        copier,
        steps,
        spans: false,
    })
    .unwrap();
    // Its steps compute nothing: its hints are priced on an idle volume.
    watcher.shared().telemetry.price_at(0.);
    Harness {
        _pool: (!ALONE.get()).then(|| POOL.read().unwrap_or_else(std::sync::PoisonError::into_inner)),
        watcher,
        device_words,
        layer,
        layers: all,
        reads,
        commits,
        load,
        tiered,
        seeds,
        second,
        staging,
        conveyor,
        conveyor_records,
        staging_records,
        file,
        host,
        scratch,
        copies,
        seeded_from: match placer {
            Placer::Tier => bank_lent,
            Placer::Bank => lent,
        },
    }
}

/// Spin until `word` of `words` holds `value`.
pub(super) fn until(words: &Words, word: usize, value: i32) {
    eventually(&format!("word {word} holding {value}"), || words.load(word) == value);
}

/// Spin until `check` holds, `what` naming it ([`awaited`]).
pub(super) fn eventually(what: &str, check: impl Fn() -> bool) {
    awaited(what, || check().then_some(()))
}

/// How long a wait goes between notes of what it still awaits.
const NOTED: Duration = Duration::from_secs(10);

/// How long a wait may go unmet before its test fails naming it: a wait is a
/// poll of state the watcher publishes within milliseconds, so one unmet for
/// this long is a hang, which would otherwise hold the read engine's pool
/// (`POOL`) shared and starve every test that needs it alone.
const HANG: Duration = Duration::from_secs(60);

/// Spin until `ready` gives a value, `what` naming it. A wait unmet past
/// each [`NOTED`] notes what it awaits, and past [`HANG`] fails its test
/// naming it, so a hang reports itself in a minute.
pub(super) fn awaited<T>(what: &str, ready: impl Fn() -> Option<T>) -> T {
    let (began, mut noted) = (Instant::now(), Instant::now());
    loop {
        if let Some(value) = ready() {
            return value;
        }
        assert!(began.elapsed() < HANG, "hung for {HANG:?} awaiting: {what}");
        if noted.elapsed() >= NOTED {
            eprintln!("still awaiting: {what}");
            noted = Instant::now();
        }
        std::thread::yield_now();
    }
}

/// Begin a group of steps of `routes`, at least one, `scratch`-served
/// (`Shared::begin`): the steps, in sequence order. Its grouped steps are a
/// prompt's, its decode steps generation's.
pub(super) fn begin(shared: &Arc<Shared>, routes: &[StepRoute], scratch: bool) -> Result<Vec<ExpertStep>, String> {
    let work = |route: StepRoute| match route {
        StepRoute::Grouped => LaunchWork::Prompt,
        StepRoute::Decode => LaunchWork::Generation,
    };
    let launches = routes.iter().map(|&route| (route, work(route))).collect::<Vec<_>>();
    begin_as(shared, &launches, scratch)
}

/// [`begin`] of steps each of its route and its work.
pub(super) fn begin_as(shared: &Arc<Shared>, launches: &[(StepRoute, LaunchWork)], scratch: bool) -> Result<Vec<ExpertStep>, String> {
    let (first, rest) = launches.split_first().expect("a group of steps");
    let mut chain = crate::programs::Chain::one(*first);
    rest.iter().for_each(|&launch| chain.push(launch));
    let steps = shared.begin(chain, |&(route, _)| route, |&(_, work)| work, || Ok(scratch)).map_err(|(error, _)| error)?.0;
    Ok(steps.into_iter().map(|(_, step)| step).collect())
}

/// Demand reads standing in the read engine's pool ahead of a test's until
/// it finishes them: one piece of a file each (its volume's due read), as
/// many at once as the pool has workers once the line stands, and two more,
/// into one record nothing reads. Each read is submitted again as it ends,
/// by the worker that ended it, before that worker takes another piece
/// (`seismic_io::read_workers`). So a read of the line not queued has its
/// piece on a worker, or its end runs on one, or the line's first
/// submissions are still being made on the test's thread: at most one more
/// than the workers. Two more always leave one queued, parked, on the
/// volume's link, however fast the volume or late a resubmission: a read
/// the test issues behind it at a lesser priority (a hint's) is
/// not taken until the test finishes the line ("Order and depth"), and one
/// at demand, or hastened to it, waits for the reads queued as it was
/// issued: at most the pool's workers and two pieces, never the line's
/// every resubmission. The pool grows only as reads are submitted, to at
/// most what their links can hold in flight, so the line first grows it
/// there, and stands two reads longer than it; a test that queues a line
/// holds the pool alone (`alone`).
pub(super) struct Ahead {
    line: Arc<AheadLine>,
    path: std::path::PathBuf,
}

/// [`Ahead`]'s reads: every one's ticket, which keeps it queued, until the
/// line is done with (`done`), when a read ending submits no other.
struct AheadLine {
    source: ExpertSource,
    records: Mutex<Records>,
    done: std::sync::atomic::AtomicBool,
    tickets: Mutex<Vec<Ticket>>,
    /// The bytes each read reads: its volume's due read, one piece.
    bytes: u64,
    /// How long a read's end waits before it submits the read again.
    late: Duration,
}

impl AheadLine {
    fn tickets(&self) -> std::sync::MutexGuard<'_, Vec<Ticket>> {
        self.tickets.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Read the file, and again as that read ends, until the line is done
    /// with. A read that ended before its hook was registered runs the hook
    /// at once, on this thread: the next read is then submitted by this
    /// loop, not from inside the hook, so no line of reads ending as fast as
    /// they are submitted grows the stack.
    fn stand(self: &Arc<Self>) {
        use std::sync::atomic::{AtomicU8, Ordering};
        const REGISTERING: u8 = 0;
        const REGISTERED: u8 = 1;
        const ENDED: u8 = 2;
        while !self.done.load(Ordering::Acquire) {
            let read = Read::new(0, 0, self.bytes, &self.records.lock().unwrap(), 0);
            let ticket = self.source.request(vec![read], seismic_io::Want::now()).unwrap();
            let (line, state) = (self.clone(), Arc::new(AtomicU8::new(REGISTERING)));
            let ending = state.clone();
            ticket.on_end(move |_, _| {
                // Ended while its hook was registered: the loop reads again.
                if ending.compare_exchange(REGISTERING, ENDED, Ordering::AcqRel, Ordering::Acquire).is_err() {
                    std::thread::sleep(line.late);
                    line.stand();
                }
            });
            self.tickets().push(ticket);
            if state.compare_exchange(REGISTERING, REGISTERED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return;
            }
        }
    }
}

impl Ahead {
    const FILE: u64 = 16 << 20;

    pub(super) fn queue(name: &str) -> Self {
        Self::resubmitting_after(name, Duration::ZERO)
    }

    /// [`Self::queue`], each read submitted again `late` after it ended.
    pub(super) fn resubmitting_after(name: &str, late: Duration) -> Self {
        assert!(ALONE.get(), "a test queuing reads ahead holds the pool alone (`alone`)");
        let path = std::env::temp_dir().join(format!("expert-watcher-{name}-{}", std::process::id()));
        std::fs::write(&path, vec![0u8; Self::FILE as usize]).unwrap();
        let source = ExpertSource::open(std::slice::from_ref(&path), &urgent()).unwrap();
        let sink = Tensor::shared(&cpu(), Element::i32(), &[1, Self::FILE / 4]).unwrap();
        let records = Records::new(sink, Self::FILE, 1).unwrap();
        // The pool grown to what its devices' links can hold in flight
        // before the line stands, so no read a test makes grows it past the
        // line: whole reads of the file, read ahead, queue their pieces far
        // faster than workers take them, each submission growing the pool to
        // the most pieces queued or in flight at once, until one grows it no
        // further; dropped, what of them is queued is cancelled.
        let mut growing = Vec::new();
        loop {
            let workers = seismic_io::read_workers();
            let read = Read::new(0, 0, Self::FILE, &records, 0);
            growing.push(source.request(vec![read], seismic_io::Want::undated(0.)).unwrap());
            if seismic_io::read_workers() == workers {
                break;
            }
        }
        drop(growing);
        let shape = seismic_io::link_shape(&seismic_io::OpenFile::open(&path).unwrap(), Some(&urgent())).unwrap();
        let bytes = shape.due_read.min(Self::FILE);
        let line = Arc::new(AheadLine { source, records: Mutex::new(records), done: Default::default(), tickets: Mutex::default(), bytes, late });
        let mut standing = 0;
        while standing < seismic_io::read_workers() + 2 {
            line.stand();
            standing += 1;
        }
        Self { line, path }
    }

    /// A read of the line's file behind it, wanted as `want`, outside the
    /// line: the test's to observe and to end.
    pub(super) fn read(&self, want: seismic_io::Want) -> Ticket {
        let read = Read::new(0, 0, Self::FILE, &self.line.records.lock().unwrap(), 0);
        self.line.source.request(vec![read], want).unwrap()
    }

    /// End the line: every read in it lands, and none follows it.
    pub(super) fn finish(self) {
        self.line.done.store(true, std::sync::atomic::Ordering::Release);
        let tickets = std::mem::take(&mut *self.line.tickets());
        assert!(tickets.iter().all(|ticket| matches!(ticket.wait(), Outcome::Read)));
        std::fs::remove_file(&self.path).unwrap();
    }
}

impl Drop for Ahead {
    /// A test that failed before it finished the line leaves no read behind.
    fn drop(&mut self) {
        self.line.done.store(true, std::sync::atomic::Ordering::Release);
        drop(std::mem::take(&mut *self.line.tickets()));
    }
}

impl Harness {
    /// Wait until nothing reads or lands into staging, every record free:
    /// a hint's read takes staging beyond a request's choices only as the
    /// ring is read, and a hint refused then is never read.
    pub(super) fn staging_free(&self) {
        let staging = || {
            let state = residency::lock(&self.watcher.state);
            (state.prefetch.staging.free(), state.prefetch.staging.records())
        };
        eventually("staging never came free", || {
            let (free, records) = staging();
            free == records
        });
    }

    /// Wait until the tier holds its whole reserve ready (the fixture's, a
    /// record per staging record, or the experts it lacks where fewer),
    /// every record the refill unmapped retired from its grace: a hint takes a ready record only as
    /// its ring is read (`Prefetch::admit`), and a hint refused then is never
    /// read, so a test hinting after a step must find the graces retired,
    /// not race their retirement.
    pub(super) fn tier_ready(&self) {
        let reserve = self.staging_records as usize;
        eventually("the tier's graces never retired", || {
            let state = residency::lock(&self.watcher.state);
            let directory = &state.core.directory;
            directory.host(0).graced() == 0 && !directory.refill_short(reserve)
        });
    }

    /// Wait until residency has retired every step ended so far
    /// (`Residency::retire`): each landing their requests waited for is
    /// settled, kept, and no longer an outstanding answer, so a test that
    /// takes residency's lock finds the tier as those steps left it.
    pub(super) fn retired(&self) {
        let shared = self.watcher.shared();
        eventually("residency never retired the steps ended", || {
            let frontier = shared.steps.frontier();
            residency::lock(&self.watcher.state).core.retired >= frontier
        });
    }

    /// Begin a decode step, a group of one.
    pub(super) fn step(&self) -> ExpertStep {
        self.group(&[StepRoute::Decode]).pop().unwrap()
    }

    /// Begin a prompt step, a group of one.
    pub(super) fn prompt_step(&self) -> ExpertStep {
        self.group(&[StepRoute::Grouped]).pop().unwrap()
    }

    /// Begin a scratch-served group of steps of `routes`.
    pub(super) fn group(&self, routes: &[StepRoute]) -> Vec<ExpertStep> {
        begin(self.watcher.shared(), routes, true).unwrap()
    }

    /// Play a begun decode step's route of `experts` at its one layer, and
    /// the tier wait of its request ([`Self::request_at`]).
    pub(super) fn request(&self, experts: &[usize]) -> ExpertStep {
        let step = self.step();
        self.request_at(&step, 0, experts);
        step
    }

    /// Play `routed_route_streamed`'s resolution of `experts` at layer
    /// `layer` in `step`, which every decode step's route makes at every
    /// streamed layer: one the map holds for the step is a held choice
    /// (`-1 - e`) sourced from its record; one the tier map names is a held
    /// choice sourced from its tier record `t` (as `-2 - t`), served in
    /// place; any other is absent (`e`), sourced from none. Its routing
    /// record holds them; the choices.
    pub(super) fn route_at(&self, step: &ExpertStep, layer: usize, experts: &[usize]) -> Vec<i32> {
        let (map, tiers) = (&self.layers[layer].0, &self.host.maps[layer]);
        let sequence = step.sequence();
        let (choices, sources): (Vec<_>, Vec<_>) = experts
            .iter()
            .map(|&expert| {
                let held = -1 - expert as i32;
                match (map.resolve(expert, sequence), tiers.load(expert)) {
                    (Some(record), _) => (held, record as i32),
                    (None, tier) if tier >= 0 => (held, -2 - tier),
                    _ => (expert as i32, -1),
                }
            })
            .unzip();
        self.route_sourced(step, layer, &choices, &sources);
        choices
    }

    /// Play layer `layer`'s routing record of `choices` (a held one as
    /// `-1 - expert`) in `step`, each choice's source as the route resolved
    /// it (`SOURCE_RECORD`'s word: a tier record `t` as `-2 - t`).
    pub(super) fn route_sourced(&self, step: &ExpertStep, layer: usize, choices: &[i32], sources: &[i32]) {
        let ring = WORDS.routing_words() as usize;
        let record = &self.layers[layer].4;
        let base = (step.sequence() % WORDS.flight) as usize * ring;
        for (choice, (&word, &source)) in choices.iter().zip(sources).enumerate() {
            let expert = if word < 0 { -1 - word } else { word };
            record.store(base + WORDS.routing_expert(choice as u64) as usize, expert);
            record.store(base + WORDS.routing_source(choice as u64) as usize, source);
        }
        record.store(base + constants::RING_COUNT as usize, choices.len() as i32);
        record.store(base + constants::RING_SEQUENCE as usize, low_word(step.sequence()));
    }

    /// Play the route of `experts` at layer `layer` in `step`
    /// ([`Self::route_at`]) and, where any is absent, its request of the
    /// choices, without parking. Whether it requested any.
    pub(super) fn ask_at(&self, step: &ExpertStep, layer: usize, experts: &[usize]) -> bool {
        let choices = self.route_at(step, layer, experts);
        let asks = choices.iter().any(|&choice| choice >= 0);
        if asks {
            self.publish_request(step, layer, &choices);
        }
        asks
    }

    /// Play `routed_route_streamed`'s request of layer `layer`'s `choices`
    /// in `step`'s slot of its file table: the choices and their count,
    /// then the request.
    pub(super) fn publish_request(&self, step: &ExpertStep, layer: usize, choices: &[i32]) {
        let (file, sequence) = (&self.layers[layer].1, step.sequence());
        for (choice, &word) in choices.iter().enumerate() {
            file.store(sequence, 0, choice_word(choice as u64), word);
        }
        file.store(sequence, 0, constants::FILE_COUNT, choices.len() as i32);
        file.request(sequence, 0);
    }

    /// Play the route of `experts` at layer `layer` in `step` and, where it
    /// requests any, its tier wait: parked until the slot is answered, by
    /// the step or its failure (the exchange's rule).
    pub(super) fn request_at(&self, step: &ExpertStep, layer: usize, experts: &[usize]) {
        if self.ask_at(step, layer, experts) {
            let file = &self.layers[layer].1;
            eventually(&format!("step {}'s request at layer {layer} was never answered", step.sequence()), || {
                file.answered(step.sequence(), 0)
            });
        }
    }

    /// What step `sequence`'s file slot at layer `layer` holds: none while
    /// it is unanswered.
    pub(super) fn verdict(&self, layer: usize, sequence: Step) -> Option<Answer> {
        self.layers[layer].1.answer_of(sequence, 0)
    }

    /// Choice `choice`'s answer at layer `layer` in step `sequence`'s
    /// slot: the tier record its record word names (-2 - t, the device
    /// placing; -1 for a failed read), and whether the slot is answered.
    pub(super) fn answer(&self, layer: usize, sequence: Step, choice: usize) -> (i32, Option<Answer>) {
        let source = self.layers[layer].1.load(sequence, 0, record_word(choice as u64));
        (if source == -1 { -1 } else { -2 - source }, self.verdict(layer, sequence))
    }

    /// The first byte of tier record `record` of tensor `kind`.
    pub(super) fn tiered(&self, kind: usize, record: i32) -> u8 {
        let mut byte = [0u8];
        unsafe { self.host.tier[kind].read_shared(record as u64 * RECORD, &mut byte) }.unwrap();
        byte[0]
    }

    /// The tier record layer `layer`'s tier map names for `expert`, every
    /// tensor's bytes of it whole there: none while it names none.
    pub(super) fn in_tier(&self, layer: usize, expert: usize) -> Option<i32> {
        let record = self.host.maps[layer].load(expert);
        (record >= 0).then(|| {
            for kind in 0..3 {
                assert_eq!(self.tiered(kind, record), (100 * kind + expert) as u8, "expert {expert}'s tier record");
            }
            record
        })
    }

    /// Spin until layer `layer`'s tier map names each of `experts`.
    pub(super) fn tiering(&self, layer: usize, experts: &[usize]) {
        eventually(&format!("experts {experts:?} of layer {layer} never landed in the tier"), || {
            experts.iter().all(|&expert| self.in_tier(layer, expert).is_some())
        });
    }

    /// The experts layer `layer`'s map holds.
    fn held_count(&self, layer: usize) -> usize {
        (0..EXPERTS).filter(|&expert| self.layers[layer].0.held(expert).is_some()).count()
    }

    /// Spin until the watcher holds what layer `layer`'s map holds (it
    /// retired the steps that committed them, or promoted them).
    fn holding_all(&self, layer: usize) {
        let shared = self.watcher.shared();
        eventually("the bank never held what its map holds", || shared.holdings().0[layer] == self.held_count(layer));
    }

    /// The first byte of bank record `record` of tensor `kind`.
    pub(super) fn banked(&self, kind: usize, record: usize) -> u8 {
        let mut byte = [0u8];
        unsafe { self.host.bank[kind].read_shared(record as u64 * RECORD, &mut byte) }.unwrap();
        byte[0]
    }

    /// Bank `experts` of layer `layer` as decode does, a request's experts
    /// at a time: a step lands them into the tier, and once it completed the
    /// ranking, which has its reads, promotes each into a record; every
    /// other layer routes nothing. Each is mapped to a record holding its
    /// bytes.
    pub(super) fn promoted(&self, layer: usize, experts: &[usize]) {
        for chunk in experts.chunks(WORDS.choices as usize) {
            let landing = self.step();
            for index in 0..self.layers.len() {
                match index == layer {
                    true => self.request_at(&landing, index, chunk),
                    false => drop(self.route_at(&landing, index, &[])),
                }
            }
            landing.complete().unwrap();
            let map = &self.layers[layer].0;
            eventually(&format!("experts {chunk:?} of layer {layer} were never promoted"), || {
                chunk.iter().all(|&expert| map.held(expert).is_some())
            });
            for &expert in chunk {
                let record = map.held(expert).unwrap();
                for kind in 0..3 {
                    assert_eq!(self.banked(kind, record), (100 * kind + expert) as u8, "expert {expert}'s bank record");
                }
            }
        }
        self.holding_all(layer);
    }

    /// The (expert, bank record) pairs of layer `layer`'s seed list for
    /// `step`, as `routed_copy` copies and `routed_seed_commit` commits
    /// them after its last round there: none where the list is another
    /// step's.
    pub(super) fn seed_pairs(&self, step: &ExpertStep, layer: usize) -> Vec<(i32, i32)> {
        let seeds = &self.seeds[layer];
        if seeds.load(constants::LIST_SEQUENCE as usize) != low_word(step.sequence()) {
            return Vec::new();
        }
        let first = self.seeded_from as i32;
        listed(seeds).into_iter().map(|(expert, record)| (expert, first + record)).collect()
    }

    /// Play `routed_seed_commit` of `step` at layer `layer`, after its last
    /// round there: each seed the step's list names is committed by the
    /// step, which the host maps once the device is past it. The seeds;
    /// none where the list is another step's.
    pub(super) fn commit_seeds(&self, step: &ExpertStep, layer: usize) -> Vec<(i32, i32)> {
        let pairs = self.seed_pairs(step, layer);
        for &(_, record) in &pairs {
            self.commits.store(record as usize, low_word(step.sequence()));
        }
        pairs
    }

    /// Bank `experts` of layer `layer` as a prompt group does: a group of
    /// one prompt step routing them there (and nothing elsewhere) seeds
    /// them, its seed commits played. The seeds.
    pub(super) fn seeded(&self, layer: usize, experts: &[usize]) -> Vec<(i32, i32)> {
        let step = self.prompt_step();
        let mut seeds = Vec::new();
        for index in 0..self.layers.len() {
            self.scratch(&step, index, if index == layer { experts } else { &[] });
            let committed = self.commit_seeds(&step, index);
            if index == layer {
                seeds = committed;
            }
        }
        step.complete().unwrap();
        let map = &self.layers[layer].0;
        eventually("the seeds were never mapped", || {
            seeds.iter().all(|&(expert, record)| map.held(expert as usize) == Some(record as usize))
        });
        self.holding_all(layer);
        seeds
    }

    /// Play `routed_scratch_request` at layer `layer` of `step`, routing
    /// to `flagged`, and the device arriving at each round, then parking
    /// until it is acknowledged: the (staging or conveyor record, expert)
    /// pairs each round converts.
    pub(super) fn scratch(&self, step: &ExpertStep, layer: usize, flagged: &[usize]) -> Vec<Vec<(i32, i32)>> {
        self.publish(step, layer, flagged);
        self.rounds(step, layer)
    }

    /// The flags of `step`'s chain place, its group's `n`th.
    fn flags(&self, step: &ExpertStep) -> &Words {
        let (first, _) = self.watcher.shared().steps.group_of(step.sequence());
        match step.sequence() - first {
            0 => &self.load.0,
            1 => &self.second,
            place => panic!("the harness has two chain places, not {}", place + 1),
        }
    }

    /// Play `routed_scratch_request`'s publication at layer `layer` of
    /// `step`, routing to `flagged`: its place's flags, then the request in
    /// its slot's SCRATCH_ASK lane.
    pub(super) fn publish(&self, step: &ExpertStep, layer: usize, flagged: &[usize]) {
        let flags = self.flags(step);
        for expert in 0..EXPERTS {
            flags.store(expert, i32::from(flagged.contains(&expert)));
        }
        self.layers[layer].2.request(step.sequence(), constants::SCRATCH_ASK);
    }

    /// Play the device arriving at each round of `step`'s published request
    /// at layer `layer` that parks, then parking until it is acknowledged:
    /// round 0, and the rounds the count of records round 0's return
    /// carries gives records (`routed_scratch_grids`). The lists
    /// [`Self::scratch`] returns, one a round with records.
    pub(super) fn rounds(&self, step: &ExpertStep, layer: usize) -> Vec<Vec<(i32, i32)>> {
        self.rounds_arriving(step, layer, true)
    }

    /// [`Self::rounds`], each round's experts checked whole where it
    /// converts them from ([`Self::from_conveyor`]) as it is answered,
    /// before the device arrives at the next, whose arrival frees them for
    /// other reads: where records or slots cycle within a request.
    pub(super) fn rounds_checked(&self, step: &ExpertStep, layer: usize) -> Vec<Vec<(i32, i32)>> {
        let sequence = low_word(step.sequence());
        let (_, lists) = &self.load;
        let mut rounds = Vec::new();
        self.park_answering(step, layer, true, |index| {
            let list = &lists[index as usize];
            assert_eq!(list.load(constants::LIST_SEQUENCE as usize), sequence);
            let pairs = listed(list);
            self.from_conveyor(&pairs);
            rounds.push(pairs);
        });
        rounds
    }

    /// [`Self::rounds`], the device arriving at the round after the last
    /// with records only where `after` says; else it arrives there later
    /// ([`Self::arrive_after`]).
    pub(super) fn rounds_arriving(&self, step: &ExpertStep, layer: usize, after: bool) -> Vec<Vec<(i32, i32)>> {
        let sequence = low_word(step.sequence());
        let (_, lists) = &self.load;
        let converting = self.park_arriving(step, layer, after);
        let pairs = |list: &Words| {
            assert_eq!(list.load(constants::LIST_SEQUENCE as usize), sequence);
            listed(list)
        };
        lists[..converting].iter().map(pairs).collect()
    }

    /// Play the device arriving at every round of `step`'s request at layer
    /// `layer`, each once the one before is answered (a failed step's runs
    /// on unparked): what the last round's lane holds.
    pub(super) fn every_round(&self, step: &ExpertStep, layer: usize) -> Answer {
        let (scratch, sequence) = (&self.layers[layer].2, step.sequence());
        for index in 0..WORDS.rounds {
            scratch.request(sequence, round(index));
            until_answered(scratch, sequence, round(index));
        }
        scratch.answer_of(sequence, last_round()).expect("every round is answered")
    }

    /// Play the parks of [`Self::rounds`]: the rounds with records, then
    /// the round after them, its slot's next, which its host answered
    /// before the last (its arrival frees the last round's records).
    pub(super) fn park(&self, step: &ExpertStep, layer: usize) -> usize {
        self.park_arriving(step, layer, true)
    }

    /// [`Self::park`], arriving at the round after the last with records
    /// only where `after` says.
    fn park_arriving(&self, step: &ExpertStep, layer: usize, after: bool) -> usize {
        self.park_answering(step, layer, after, |_| {})
    }

    /// [`Self::park_arriving`], `answered` called with each round with
    /// records as it is answered, before the device arrives at the next.
    fn park_answering(&self, step: &ExpertStep, layer: usize, after: bool, mut answered: impl FnMut(u64)) -> usize {
        let (scratch, sequence) = (&self.layers[layer].2, step.sequence());
        let (mut converting, mut index) = (1, 0);
        while index < (converting + u64::from(after)).min(WORDS.rounds) {
            scratch.request(sequence, round(index));
            assert_eq!(until_answered(scratch, sequence, round(index)), Answer::Answered);
            if index == 0 {
                self.tier_copy(sequence, layer);
                converting = (scratch.load(sequence, constants::SCRATCH_ASK, constants::SCRATCH_STAGED) as u64)
                    .div_ceil(ROUND_RECORDS);
                if converting == 0 {
                    break;
                }
            }
            if index < converting {
                answered(index);
            }
            index += 1;
        }
        assert_eq!(scratch.answer_of(sequence, last_round()), Some(Answer::Answered));
        converting as usize
    }

    /// Play the device arriving at round `round` of `step`'s request at
    /// `layer`, the round after its last with records, which its host
    /// answered before the last.
    pub(super) fn arrive_after(&self, step: &ExpertStep, layer: usize, round_index: u64) {
        let (scratch, sequence) = (&self.layers[layer].2, step.sequence());
        scratch.request(sequence, round(round_index));
        assert_eq!(until_answered(scratch, sequence, round(round_index)), Answer::Answered);
    }

    /// Play the device's tier copy after `sequence`'s round 0: each
    /// (tier record, scratch slot) the tier list names for the step, the
    /// tier record's bytes of every routed tensor into the expert's slot.
    fn tier_copy(&self, sequence: u64, layer: usize) {
        let tiered = &self.tiered[layer];
        if tiered.load(constants::LIST_SEQUENCE as usize) != low_word(sequence) {
            return;
        }
        let records = u32::try_from(self.host.tier[0].extents()[0]).unwrap();
        for (records, slots) in self.host.views(records).iter().zip(&self.scratch) {
            // Slot `e` holds expert `e`.
            for (record, expert) in listed(tiered) {
                let mut bytes = vec![0u8; RECORD as usize];
                // SAFETY: the harness plays the device; nothing else
                // writes the record or the slot meanwhile.
                unsafe {
                    records.tensor().read_shared(records.offset(record as u64), &mut bytes).unwrap();
                    slots.write_shared(expert as u64 * RECORD, &bytes).unwrap();
                }
            }
        }
    }

    /// The experts each convert round of `scratch` lists.
    pub(super) fn converted(
        &self,
        step: &ExpertStep,
        layer: usize,
        flagged: &[usize],
    ) -> Vec<Vec<i32>> {
        self.scratch(step, layer, flagged)
            .into_iter()
            .map(|round| round.into_iter().map(|(_, expert)| expert).collect())
            .collect()
    }

    /// Spin until a conveyor record holds expert `expert` whole (the
    /// watcher took a request that reads it).
    pub(super) fn conveying(&self, expert: i32) {
        eventually(&format!("expert {expert} was never conveyed"), || self.holding(expert).is_some());
    }

    /// The conveyor record whose every tensor holds expert `expert`'s
    /// bytes, if one does.
    pub(super) fn holding(&self, expert: i32) -> Option<i32> {
        (0..self.conveyor_records as i32)
            .find(|&record| (0..3).all(|kind| self.conveyed(kind, record) == (100 * kind + expert as u64) as u8))
    }

    /// The first byte of conveyor record `record` of tensor `kind`.
    fn conveyed(&self, kind: u64, record: i32) -> u8 {
        let mut byte = [0u8];
        let at = kind * self.conveyor_records * RECORD + record as u64 * RECORD;
        unsafe { self.conveyor.read_shared(at, &mut byte) }.unwrap();
        byte[0]
    }

    /// The first byte of scratch slot `slot` of tensor `kind`.
    fn slotted(&self, kind: u64, slot: i32) -> u8 {
        let mut byte = [0u8];
        unsafe { self.scratch[kind as usize].read_shared(slot as u64 * RECORD, &mut byte) }.unwrap();
        byte[0]
    }

    /// Each listed `(from, expert)` converts from a conveyor record or, a
    /// negative `from`, scratch slot -1 - `from`, holding the expert whole;
    /// the experts, ascending.
    pub(super) fn from_conveyor(&self, listed: &[(i32, i32)]) -> Vec<i32> {
        let mut experts = listed
            .iter()
            .map(|&(from, expert)| {
                assert!(from < self.conveyor_records as i32, "expert {expert} converted from record {from}");
                for kind in 0..3 {
                    let held = match from {
                        0.. => self.conveyed(kind, from),
                        _ => self.slotted(kind, -1 - from),
                    };
                    assert_eq!(held, (100 * kind + expert as u64) as u8, "expert {expert} from {from}");
                }
                expert
            })
            .collect::<Vec<_>>();
        experts.sort_unstable();
        experts
    }

    /// The experts of listed `(from, expert)`, ascending.
    pub(super) fn experts(listed: &[(i32, i32)]) -> Vec<i32> {
        let mut experts = listed.iter().map(|&(_, expert)| expert).collect::<Vec<_>>();
        experts.sort_unstable();
        experts
    }

    /// Of listed `(from, expert)`, each expert converted from a scratch
    /// slot, with the slot.
    pub(super) fn from_slots(listed: &[(i32, i32)]) -> Vec<(i32, i32)> {
        listed.iter().filter(|&&(from, _)| from < 0).map(|&(from, expert)| (expert, -1 - from)).collect()
    }

    /// Play a decode step whose second layer's route requests `expert`, read
    /// on demand; every other layer chooses nothing.
    pub(super) fn demand_step(&self, expert: usize) {
        let step = self.step();
        self.route_at(&step, 0, &[]);
        self.request_at(&step, 1, &[expert]);
        for layer in 2..self.layers.len() {
            self.route_at(&step, layer, &[]);
        }
        step.complete().unwrap();
    }

    /// Play layer `layer`'s ring hinting `experts` of the next in `step`.
    pub(super) fn hint(&self, step: &ExpertStep, layer: usize, experts: &[i32]) {
        self.hint_at(step, layer, 0, experts);
    }

    /// Play `step`'s tier wait at layer `layer` ending on its still pulse,
    /// as its kernel does when the host's pulse stood still past its
    /// patience.
    pub(super) fn expire(&self, step: &ExpertStep, layer: usize) {
        self.layers[layer].1.expire(step.sequence(), 0);
    }

    /// Play layer `layer`'s ring row of lead `lead` hinting `experts` of
    /// the layer `lead + 1` ahead in `step`: a row each, its one choice
    /// (rank 0) at the harness's one choice a row.
    pub(super) fn hint_at(&self, step: &ExpertStep, layer: usize, lead: usize, experts: &[i32]) {
        let ring = &self.layers[layer].3;
        let base = lead * WORDS.ring_words() as usize;
        for (index, &expert) in experts.iter().enumerate() {
            let entry = base + (constants::RING_EXPERTS + constants::HINT_ENTRY_WORDS * index as u64) as usize;
            ring.store(entry + constants::HINT_EXPERT as usize, expert);
            ring.store(entry + constants::HINT_RANK as usize, 0);
        }
        ring.store(base + constants::RING_COUNT as usize, experts.len() as i32);
        ring.store(base + constants::RING_SEQUENCE as usize, low_word(step.sequence()));
    }

    pub(super) fn staged(&self, kind: u64, record: i32) -> u8 {
        let mut byte = [0u8];
        let at = kind * u64::from(self.staging_records) * RECORD + record as u64 * RECORD;
        unsafe { self.staging.read_shared(at, &mut byte) }.unwrap();
        byte[0]
    }
}

/// A list's (from, to) entries, up to its count.
pub(super) fn listed(list: &Words) -> Vec<(i32, i32)> {
    (0..list.load(constants::LIST_COUNT as usize) as usize)
        .map(|index| {
            let entry = magnitude_kernels::list_extent(index as u64) as usize;
            (
                list.load(entry + constants::LIST_FROM as usize),
                list.load(entry + constants::LIST_TO as usize),
            )
        })
        .collect()
}

/// A line holds back every read ahead behind it however late its reads are
/// submitted again: each read's end waits a second before it resubmits,
/// longer than a volume takes to read one of its reads, so its ends pile up
/// on the workers, and a read of the line stays queued all the same
/// (`Ahead`).
#[test]
fn a_line_holds_back_a_read_ahead_however_late_its_reads_resubmit() {
    let _alone = alone();
    let line = Ahead::resubmitting_after("late-line", Duration::from_secs(1));
    let held = line.read(seismic_io::Want::undated(1.));
    std::thread::sleep(Duration::from_secs(2));
    assert!(held.running(), "a read ahead was taken past the line");
    drop(held);
    line.finish();
}
