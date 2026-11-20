use super::{ExpertBank, PoolConveyor, allocate_placed, homes, placed_words, slot_maps, write_blocks};
use crate::experts::conveyor::{Carrier, LentPool, Lending};
use crate::experts::source::{ExpertSource, Records};
use crate::experts::watcher::{Copier, Extents, Group, Layer, ListRecords, LoadWords, PlacedWords, Pool, Steps, StoredPart, StoredTensor, Watcher, WatcherParts};
use crate::experts::directory::{HostTier as TierStart, Placement};
use crate::experts::landing::{LandingList, TierTensor};
use crate::experts::map::{LandingMap, StampedMap};
use crate::experts::{ExchangeTable, Words};
use crate::planning::{BankAllocation, BankLayout, BankTier, ConveyorCarrier, ExpertBankPlan, Lender, PlacedAllocations, Placer, RecordTensors, StagingPool};
use magnitude_artifacts::gguf::GgufArtifact;
use seismic::{Device, SlabTensor, Tensor};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use super::resize::{Commitment, HostTier, Stores};
use crate::storage_thread::StorageThread;
use crate::experts::site::site_lines;

/// A load's granule pools, each opened once for the load, each class at
/// its own granule (§4.7): its device store's state pool at the state's
/// granule (`StateStorePlan::granule`), which the target and head state
/// slabs claim from; where the load has a bank, the records pool of the
/// store its records are placed in, at the bank's granule
/// (`ExpertBankPlan::granule`), which its records and its banked stored
/// tables claim from; and a linked host tier's own records pool of host
/// storage (`ExpertBankPlan::host_granule`).
#[derive(Clone)]
pub struct LoadPools {
    pub state: seismic::Pool,
    pub records: Option<seismic::Pool>,
    pub host: Option<seismic::Pool>,
}

impl LoadPools {
    /// `execution`'s pools on `device`, charged to the pool entries of
    /// `ledgers`' stores (`DeviceLedgers::open_pool`).
    pub fn open(device: &Device, ledgers: &crate::memory::ledger::DeviceLedgers, execution: &crate::planning::ExecutionPlan) -> Result<Self, String> {
        Self::open_at(device, ledgers, None, execution).map_err(|error| error.to_string())
    }

    /// `execution`'s pools on `device`, claiming each pool's absent granule at
    /// `reading` (the load's reading when provided) before opening it.
    pub fn open_at(
        device: &Device,
        ledgers: &crate::memory::ledger::DeviceLedgers,
        reading: Option<&crate::memory::ledger::DeviceReading>,
        execution: &crate::planning::ExecutionPlan,
    ) -> Result<Self, crate::memory::ledger::OpenPoolError> {
        use crate::memory::ledger::PoolClass;
        let granule = execution.resources().target_state().granule;
        let state = ledgers.open_pool_at(reading, device, seismic::Contract::DeviceOnly, PoolClass::State, granule)?;
        let Some(plan) = execution.load().expert_bank() else {
            return Ok(Self { state, records: None, host: None });
        };
        let records = ledgers.open_pool_at(reading, device, plan.host().contract(BankTier::Bank), PoolClass::Records, plan.granule())?;
        let host = plan
            .host_granule()
            .map(|granule| ledgers.open_pool_at(reading, device, plan.host().contract(BankTier::Host), PoolClass::Records, granule))
            .transpose()?;
        Ok(Self { state, records: Some(records), host })
    }
}

impl ExpertBank {
    /// Allocate `plan`'s bank on `device`, its records claimed from
    /// `pools` up to its committed slots, and start its watcher over the
    /// files `artifact` stores the expert stacks in; each streamed layer's
    /// admitted CPU site (`sites`) served by its host compute executor, its
    /// lines those the job stored (`cpu`).
    pub(crate) fn open(
        charges: &seismic::Charges,
        plan: &ExpertBankPlan,
        artifact: &GgufArtifact,
        sites: &[Option<crate::native::BankSite>],
        lines: std::sync::Arc<[crate::assessment::BankLines]>,
        cpu: &crate::assessment::StoredCpuSite,
        pools: LoadPools,
        (changes, host_backing): (std::sync::Arc<crate::memory::ledger::Changes>, Option<crate::memory::ledger::Claim>),
    ) -> Result<Arc<Self>, crate::residency::WeightImportError> {
        // A layer whose CPU site its qualification refused serves every miss
        // over the link: never silently, and with why.
        for (layer, reason) in plan.refused_sites() {
            tracing::warn!("streamed layer {layer}'s CPU site was refused, so it reads every miss over the link: {reason}");
        }
        let (served, awaiting) = sites.iter().flatten().fold((0, 0), |(served, awaiting), site| match site.admitted {
            Some(_) => (served + 1, awaiting),
            None => (served, awaiting + 1),
        });
        tracing::info!("{served} streamed layers' CPU sites admitted, {awaiting} awaiting their verdicts");
        use crate::residency::WeightImportError;
        let failed = |error: &dyn std::fmt::Display| format!("expert bank allocation failed: {error}");
        // Every allocation of the plan's layout but the records: words and
        // staging host-shared, each lender's lent records under its tier's
        // contract, the stacks' per-expert scales imported. The records are
        // their stores' regions, formed once the stores are.
        let planned = plan.layout()?;
        let allocated = planned.try_map(|allocation| {
            let held = |contract, tensor: Result<Tensor, seismic::TensorError>| {
                tensor.map(|tensor| (allocation.clone(), Some(tensor))).map_err(|error| WeightImportError::allocating(contract, &error))
            };
            match allocation {
                BankAllocation::Shared { element, extents } => held(seismic::Contract::Shared, Tensor::shared(charges, *element, extents)),
                BankAllocation::Words { element, extents } => held(seismic::Contract::Words, Tensor::words(charges, *element, extents)),
                BankAllocation::Protocol { element, extents } => held(seismic::Contract::Words, Tensor::protocol_words(charges, *element, extents)),
                BankAllocation::Device { element, extents } => held(seismic::Contract::DeviceOnly, Tensor::zeros(charges, *element, extents)),
                BankAllocation::Lent { tier, element, extents, .. } => match plan.host().contract(*tier) {
                    seismic::Contract::DeviceOnly => held(seismic::Contract::DeviceOnly, Tensor::zeros(charges, *element, extents)),
                    contract => held(contract, Tensor::shared(charges, *element, extents)),
                },
                BankAllocation::Records { .. } => Ok((allocation.clone(), None)),
                BankAllocation::Imported(scale) => crate::residency::import_scale(charges, artifact, scale).map(|tensor| (allocation.clone(), Some(tensor))),
            }
        })?;
        // Each lender's lent records, by the tier and group lending them.
        let lent = allocated
            .lent
            .iter()
            .map(|tensors| {
                let mut of = None;
                let tensors = tensors.as_ref().try_map(|_, (allocation, tensor)| {
                    let BankAllocation::Lent { tier, group, .. } = allocation else {
                        return Err("a lent allocation is not lent records".to_owned());
                    };
                    of = Some((*tier, *group));
                    tensor.clone().ok_or_else(|| "lent records allocate a tensor".to_owned())
                })?;
                Ok(((of.ok_or("lent records of no routed tensor")?), tensors))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let lent_of = |tier: BankTier, group: usize| lent.iter().find(|((of, at), _)| (*of, *at) == (tier, group)).map(|(_, tensors)| tensors);
        // Each tier's stores: a group's records a slab each of its pool's
        // granules, every expert of its layers (`ExpertBankPlan::records`,
        // `host_extent`), its leading records its lent ones.
        let stores = |tier: BankTier, pool: seismic::Pool| -> Result<Arc<Stores>, WeightImportError> {
            let groups = (0..plan.groups().len())
                .map(|group| {
                    let (extent, lent) = match tier {
                        BankTier::Bank => (plan.records(group), plan.lent(group)),
                        BankTier::Host => (plan.host_extent(group), plan.host_lent(group)),
                    };
                    let record = &plan.groups()[group].record;
                    let (kinds, regions): (Vec<_>, Vec<_>) = plan.regions(group).into_iter().unzip();
                    let bytes = kinds
                        .iter()
                        .map(|&kind| record.get(kind).ok_or("a region of a tensor the record lacks")?.record_bytes())
                        .collect::<Result<Vec<_>, String>>()?;
                    let store = match lent {
                        0 => SlabTensor::in_pool(charges, &pool, 1, extent, &regions),
                        _ => {
                            let tensors = lent_of(tier, group).ok_or("a group's lent records beside no lent allocation")?;
                            let views = kinds
                                .iter()
                                .map(|&kind| tensors.get(kind).cloned().ok_or("lent records of a tensor the record lacks"))
                                .collect::<Result<Vec<_>, _>>()?;
                            SlabTensor::in_pool_over(charges, &pool, 1, extent, &regions, views)
                        }
                    }
                    .map_err(|error| WeightImportError::allocating(plan.host().contract(tier), &error))?;
                    Ok((store, lent, bytes))
                })
                .collect::<Result<Vec<_>, WeightImportError>>()?;
            Ok(Arc::new(Stores::new(pool, plan.block(), groups)))
        };
        let LoadPools { records, host: host_pool, .. } = pools;
        let bank_pool = records.expect("invariant violated: a banked load opens its records pool");
        let bank_stores = stores(BankTier::Bank, bank_pool)?;
        let host_stores = match (plan.tiered(), host_pool) {
            (true, Some(pool)) => Some(stores(BankTier::Host, pool)?),
            (false, None) => None,
            (tiered, _) => return Err(format!("a bank whose host tier is {tiered} opened with the other's pools").into()),
        };
        // Derived once here, so no resize derives them on the step's thread.
        let deriving = std::time::Instant::now();
        let counts = Arc::new(plan.counts()?);
        tracing::info!("derived the bank's counts up to {} slots in {:.1} ms", plan.ceiling, deriving.elapsed().as_secs_f64() * 1e3);
        // Each store's records at the load: the bank's at its planned slots,
        // a host tier's at no host slots (its backing to the plan's beside
        // the fill, `ExpertBank::back_host`), claimed lowest first, so each
        // group holds records 0 onward. The pool charges their granules,
        // so the claims are one change.
        changes.change(|_| {
            (0..plan.groups().len()).try_for_each(|group| {
                let claimed = |tier: BankTier| move |error: seismic::TensorError| WeightImportError::allocating(plan.host().contract(tier), &error);
                bank_stores.claim_to(group, counts.records_at(group, plan.slots)).map_err(claimed(BankTier::Bank))?;
                if let Some(host) = &host_stores {
                    host.claim_to(group, counts.host_records(group, 0)).map_err(claimed(BankTier::Host))?;
                }
                Ok::<_, WeightImportError>(())
            })
        })?;
        // The layout as kernels bind it: each records allocation its store's
        // region, through which every record held now or later is reached.
        let layout = allocated.try_map(|(allocation, tensor)| match (tensor, allocation) {
            (Some(tensor), _) => Ok(tensor.clone()),
            (None, BankAllocation::Records { tier, group, kind }) => {
                let stores = match tier {
                    BankTier::Bank => &bank_stores,
                    BankTier::Host => host_stores.as_ref().ok_or("host tier records beside no tier")?,
                };
                let region = plan.regions(*group).iter().position(|(of, _)| of == kind).ok_or("records of a tensor the group's regions lack")?;
                stores.region(*group, region).map_err(|error| failed(&error))
            }
            (None, _) => Err("an allocation allocated no tensor".to_owned()),
        })?;
        // Each list copy's blocks, its grid's first word (`routed_list_grid`
        // writes the rest): a list's whole grid, until a tier copy's over the
        // link are measured (`Self::tier_copy_blocks`).
        for grid in [&layout.load.tier_grid, &layout.load.seed_grid] {
            write_blocks(grid, magnitude_kernels::LIST_BLOCKS)?;
        }
        for (w, tensor) in layout.load.window.iter().enumerate() {
            let word = i32::try_from(w).map_err(|_| "window index exceeds i32")?.to_le_bytes();
            tensor.clone().write_from_host(&word).map_err(|error| format!("a window index: {error}"))?;
        }
        let (staging_pools, pool_of) = plan.staging_pools();
        let conveyor = plan.conveyor();
        // Each pool's conveyor: its view of the lender's lent records as the
        // conveyor's layout lays it there, plainly: the one store's on
        // unified memory, the host tier's beside a discrete device
        // (`ConveyorCarrier`).
        let carriers = match conveyor.carrier {
            ConveyorCarrier::Bank(lender) => lent_of(BankTier::Bank, lender.group),
            ConveyorCarrier::HostTier(lender) => lent_of(BankTier::Host, lender.group),
        }
        .ok_or("a conveyor beside no lent records")?;
        let conveyors = staging_pools
            .iter()
            .zip(&conveyor.conveyor.offsets)
            .map(|(pool, &offset)| {
                let view = carriers
                    .get(plan.form().carrier(pool.kind))
                    .ok_or("a staging pool of a tensor the records lack")?
                    .carried_region(pool.source, &[conveyor.conveyor.records, pool.matrix[0], pool.matrix[1]], offset)
                    .map_err(|error| format!("conveyor view failed: {error}"))?;
                Ok(PoolConveyor { conveyor: view })
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Each scratch form's view of the lender's lent bank records as the
        // scratch's layout lays it there, and each layer's view of its form
        // (`ExpertBankPlan::scratch`).
        let scratch_plan = plan.scratch();
        let lender = lent_of(BankTier::Bank, scratch_plan.lender.group).ok_or("a scratch beside no lent bank records")?;
        let forms = plan.layer_scratch_forms();
        let scratch = plan
            .layers()
            .iter()
            .zip(&forms)
            .enumerate()
            .map(|(index, (_, forms))| {
                let binding = plan.binding(index)?;
                let layer_window = plan.layer_window(index);
                binding.record.try_map(|kind, tensor| {
                    let form = *forms.get(kind).ok_or("a scratch form of a tensor the record lacks")?;
                    lender
                        .get(kind)
                        .ok_or("scratch of a tensor the lender lacks")?
                        .carried_region(tensor.resident, &[layer_window, tensor.matrix[0], tensor.matrix[1]], scratch_plan.layout.offsets[form])
                        .map_err(|error| format!("scratch view failed: {error}"))
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let homes = plan.resident.then(|| homes(plan, &layout)).transpose()?;
        let slot_maps = slot_maps(plan, &layout)?;
        let floor = allocate_placed(charges, plan, std::num::NonZeroU64::MIN)
            .map_err(|error| crate::residency::WeightImportError::of_allocation(seismic::Contract::Words, error))?;
        // The slots the bank commits, which its watcher's telemetry reads.
        let slots = Arc::new(AtomicU64::new(plan.slots));
        let landed = match plan.placer() {
            Placer::Tier => host_stores.as_ref().ok_or("a bank the device places beside no host tier")?,
            Placer::Bank => &bank_stores,
        };
        let (watcher, lines, scratch_tables) =
            watch(plan, artifact, &layout, (&bank_stores, landed), &slot_maps, &floor, &pool_of, &conveyors, &scratch, sites, lines, cpu, slots.clone())?;
        let host = host_stores
            .map(|stores| {
                let ceiling = plan.host_ceiling().ok_or("a host tier's records beside a plan without one")?;
                Ok::<_, String>(Mutex::new(HostTier {
                    stores,
                    commitment: Commitment::new(Arc::new(AtomicU64::new(0))),
                    ceiling,
                    thread: StorageThread::start("expert-bank-host-tier")?,
                    backing: false,
                    last_resize: None,
                }))
            })
            .transpose()?;
        let bank = Self {
            layout,
            floor,
            held: Mutex::new(None),
            device: charges.device(),
            pool_of,
            conveyors,
            scratch,
            homes,
            #[cfg(test)]
            resident_decodes: std::sync::atomic::AtomicU64::new(0),
            stores: bank_stores,
            host,
            slot_maps,
            watcher,
            lines,
            scratch_tables,
            commitment: Mutex::new(Commitment::new(slots)),
            last_resize: Mutex::new(None),
            resize_thread: StorageThread::start("expert-bank-resize")?,
            rises: Arc::new(Mutex::new(())),
            plan: plan.clone(),
            counts,
            cache: Arc::new(super::cache::BankCache::new(changes)),
        };
        let bank = Arc::new(bank);
        bank.cache.set_reclaimer(BankTier::Bank, super::resize::BankReclaimer::new(&bank, BankTier::Bank));
        if bank.host.is_some() {
            bank.cache.set_reclaimer(BankTier::Host, super::resize::BankReclaimer::new(&bank, BankTier::Host));
        }
        bank.cache.open_host(plan.host_slots, host_backing);
        if plan.host_slots > 0 {
            bank.back_host().map_err(|error| format!("the host tier's backing: {error}"))?;
        }
        Ok(bank)
    }
}

/// Start the watcher over the bank's words, those of an empty bank (every
/// map entry unmapped, every other word 0) or, of a whole one, each map
/// naming its layer's home records, and the files its experts are stored
/// in, beside an empty host tier where the bank has one, and the host
/// compute executor serving its landings and its CPU site: each sited
/// layer's CPU lines as `cpu` holds them into `lines` ([`site_lines`]); its
/// telemetry reads the bank's committed `slots` (`Commitment::slots`). The
/// bank's stores, which promotions copy into, and those the host lands
/// into (the host tier's, or where the host places the bank's own).
#[allow(clippy::too_many_arguments)]
fn watch(
    plan: &ExpertBankPlan,
    artifact: &GgufArtifact,
    layout: &BankLayout<Tensor>,
    (bank, landed): (&Arc<Stores>, &Arc<Stores>),
    slot_maps: &[Tensor],
    floor: &PlacedAllocations<Tensor>,
    pool_of: &[RecordTensors<usize>],
    conveyors: &[PoolConveyor],
    scratch: &[RecordTensors<Tensor>],
    sites: &[Option<crate::native::BankSite>],
    lines: std::sync::Arc<[crate::assessment::BankLines]>,
    cpu: &crate::assessment::StoredCpuSite,
    slots: Arc<AtomicU64>,
) -> Result<(Watcher, std::sync::Arc<[crate::assessment::BankLines]>, Vec<ExchangeTable>), String> {
    let words = |tensor: &Tensor, initial| Words::new(tensor.clone(), tensor.byte_len() / 4, initial);
    let binding = plan.binding(0)?;
    // The steps the watcher serves, under whose lock the host writes every
    // map entry (`experts::map`).
    let steps = Arc::new(Steps::new(plan.flights, plan.places, crate::experts::exchange::Pulse::of(layout.load.resident.device().backend())));
    // The floor's placed words, which the bank holds until a group holds
    // more places.
    let PlacedWords { flags, layers: group_reads } = placed_words(plan, floor)?;
    // Each group's tier tensors, which landings convert its stored experts
    // into: each routed tensor's records the landed stores hold at the load
    // (a growth's views replace them, `watcher::Join`; a shrink evicts
    // them, `Shared::shrink`).
    let tiers = plan
        .groups()
        .iter()
        .zip(landed.records()?)
        .map(|(planned, views)| {
            planned
                .record
                .iter()
                .zip(views)
                .map(|((_, record), records)| TierTensor::new(records, record.source, record.resident, record.matrix, record.stored_bytes()?))
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    // The host compute executor's threads, sized once for the load: the
    // executor lands every file miss into the tier and serves every sited
    // layer's CPU misses.
    let device = layout.load.resident.device();
    // A decode request reads at most a row's selected experts of one layer,
    // each a record stored as its routed tensors' ranges: the read engine
    // reads the files for it.
    let urgent = plan.urgent()?;
    let read_cores = seismic_io::read_cores(artifact.sources().map(|source| &**source), &urgent)
        .map_err(|error| format!("the read engine's cores: {error}"))?;
    let machine = super::Machine::discover(&device, read_cores)?;
    let (site_layers, lines, threads) = site_lines(plan, layout, Some(&tiers), sites, lines, &machine, cpu, steps.pulse())?;
    let mut files = Vec::<std::path::PathBuf>::new();
    let mut watched = Vec::with_capacity(layout.layers.len());
    let mut bank_maps = Vec::new();
    let mut landing_maps = Vec::new();
    let stacks = plan.stored_stacks(artifact)?;
    let scratch_forms = plan.layer_scratch_forms();
    // Where the host addresses the scratch (the one store's on unified
    // memory, beside the conveyor), its slots carry experts too.
    let addressed = matches!(plan.conveyor().carrier, ConveyorCarrier::Bank(_));
    for (index, layer) in layout.layers.iter().enumerate() {
        let group = plan.layers()[index].group;
        let stored = plan.groups()[group]
            .record
            .iter()
            .zip(stacks[index].iter())
            .zip(pool_of[index].iter())
            .map(|(((_, _), (_, stack)), (_, &pool))| {
                let parts = stack
                    .parts
                    .iter()
                    .map(|part| {
                        let file = match files.iter().position(|path| *path == part.path) {
                            Some(file) => file,
                            None => {
                                files.push(part.path.clone());
                                files.len() - 1
                            }
                        };
                        StoredPart { file, base: part.offset + part.first, pitch: part.pitch, bytes: part.bytes }
                    })
                    .collect();
                Ok::<_, String>(StoredTensor { parts, pool })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let experts = plan.layers()[index].experts;
        let layer_window = plan.layer_window(index);
        let slots = addressed
            .then(|| {
                plan.groups()[group]
                    .record
                    .iter()
                    .map(|(kind, _)| {
                        let view = scratch[index].get(kind).ok_or("a scratch view of a tensor the record lacks")?;
                        Records::rows(view, view.byte_len() / layer_window, layer_window).map_err(|error| format!("the scratch: {error}"))
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .transpose()?
            // A layer whose stored expert outgrows its slot lends none.
            .filter(|slots: &Vec<Records>| stored.iter().zip(slots).all(|(stored, records)| stored.bytes() <= records.bytes()));
        let scratch_form = scratch_forms[index].iter().map(|(_, &form)| form).collect();
        // Every entry unmapped, its one writer the bank's records beside a
        // host tier (`experts::records`), where the host places the one
        // store's host records through their landing map
        // (`experts::host_records`), each the one writer of its tier map.
        let map = StampedMap::new(words(&slot_maps[index], 0)?);
        let view = map.view().clone();
        let landing = match plan.placer() {
            Placer::Tier => {
                bank_maps.push(map);
                let tier = layer.tier.as_ref().ok_or("a layer beside a host tier has no tier map")?;
                LandingMap::Tier(words(tier, -1)?)
            }
            Placer::Bank => LandingMap::Bank { map: Arc::new(map), steps: steps.clone() },
        };
        landing_maps.push(landing);
        // A round converts conveyor records, or the scratch slots the
        // layer's experts were read into where the host addresses the
        // scratch, into the scratch slot of each expert.
        let lent = if slots.is_some() { layer_window } else { 0 };
        let bounds = crate::experts::ListBounds { from: plan.binding(index)?.conveyor, to: layer_window, lent };
        let rounds = layer.rounds.iter().map(|list| Ok(LandingList::new(words(list, 0)?, bounds))).collect::<Result<Vec<_>, String>>()?;
        let windows = layer
            .windows
            .iter()
            .enumerate()
            .map(|(w, table)| words(table, if w == 0 { -2 } else { -1 }))
            .collect::<Result<Vec<_>, String>>()?;
        watched.push(Layer {
            group,
            experts: usize::try_from(experts).map_err(|_| "experts exceed the host")?,
            layer_window: usize::try_from(layer_window).map_err(|_| "layer window exceeds the host")?,
            map: view,
            file: ExchangeTable::new(layer.file.clone(), plan.binding(index)?.words.file(), steps.pulse())?,
            scratch: ExchangeTable::new(layer.scratch.clone(), plan.binding(index)?.words.scratch(), steps.pulse())?,
            rings: words(&layer.rings, 0)?,
            hints: plan.predicted(index).collect(),
            routing: words(&layer.routing, 0)?,
            keys: words(&layer.keys, 0)?,
            front: words(&layer.front, 0)?,
            policy: words(&layer.policy, 0)?,
            site: site_layers.iter().find(|site| site.layer() == index).map(|site| site.slots().clone()),
            stored: stored.into(),
            slots,
            scratch_form,
            landing: match plan.placer() {
                Placer::Tier => None,
                Placer::Bank => {
                    let binding = plan.binding(index)?;
                    let bounds = crate::experts::ListBounds { from: binding.staging, to: binding.records, lent: 0 };
                    Some(Arc::new(LandingList::new(words(&layer.landing, 0)?, bounds)))
                }
            },
            rounds,
            lists: {
                let binding = plan.binding(index)?;
                ListRecords { bank: binding.records, tier: binding.tier }
            },
            tiered: words(&layer.tiered, 0)?,
            seeds: words(&layer.seeds, 0)?,
            windows,
        });
    }
    // A staging or conveyor record is one contiguous range of its pool's,
    // one after another (a conveyor's in the lender's lent records), which
    // reads land in in place.
    let (staging_pools, _) = plan.staging_pools();
    let pool_records = |pool: &StagingPool, name: &str, tensor: &Tensor, records| {
        Records::rows(tensor, pool.record_bytes()?, records)
            .map_err(|error| format!("the {} {name}: {error}", pool.source.name()))
    };
    let pools = staging_pools
        .iter()
        .zip(&layout.staging)
        .map(|(pool, tensor)| Ok(Pool { staging: pool_records(pool, "staging", tensor, plan.staging_records())? }))
        .collect::<Result<Vec<_>, String>>()?;
    // What carries the conveyor's experts: the records its lender lends it,
    // each pool's conveyor where it lies in them, which reads land in in
    // place.
    let conveyor = plan.conveyor();
    let lending = |lender: &Lender| {
        let lent = &plan.groups()[lender.group].record;
        Ok::<_, String>(Lending {
            group: lender.group,
            pools: staging_pools
                .iter()
                .zip(&conveyor.conveyor.offsets)
                .map(|(pool, &offset)| {
                    Ok(LentPool {
                        record: lent
                            .get(plan.form().carrier(pool.kind))
                            .ok_or("a staging pool of a tensor the lender's records lack")?
                            .record_bytes()?,
                        offset,
                        conveyor: pool.record_bytes()?,
                    })
                })
                .collect::<Result<_, String>>()?,
        })
    };
    // What promotions copy with, beside a host tier: the device's transfer
    // queue, and the bank's records each copy writes, whose form is the
    // tier's (`BankPart::Tier`'s, the bank's resident one).
    let copier = match plan.placer() {
        Placer::Tier => {
            let copies = seismic::Transfer::new(&device)
                .map(|transfer| super::copies::Copies::new(transfer, steps.events().clone()))
                .map_err(|error| format!("the device's transfers: {error}"))?;
            Some(Copier { copies })
        }
        Placer::Bank => None,
    };
    let carrier = Carrier {
        records: usize::try_from(conveyor.conveyor.records).map_err(|_| "conveyor records exceed the host")?,
        slots: if addressed {
            usize::try_from(plan.window_slots()).map_err(|_| "scratch slots exceed the host")?
        } else {
            0
        },
        lending: lending(&conveyor.carrier.lender())?,
        pools: staging_pools
            .iter()
            .zip(conveyors)
            .map(|(pool, view)| pool_records(pool, "conveyor", &view.conveyor, conveyor.conveyor.records))
            .collect::<Result<Vec<_>, String>>()?,
    };
    // The records the host lands into: a host tier's, each group's at its
    // floor (backed to the plan's size beside the load), or where the host
    // places, the bank's own.
    let host = {
        let records = (0..tiers.len())
            .map(|group| u32::try_from(landed.present(group)).map_err(|_| "host records exceed u32".to_owned()))
            .collect::<Result<Vec<_>, String>>()?;
        let lender = conveyor.carrier.lender();
        let lent = (0..records.len())
            .map(|group| match group == lender.group {
                true => u32::try_from(lender.records).map_err(|_| "lent records exceed u32".to_owned()),
                false => Ok(0),
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Beside a host tier the bank's records write its maps, and the
        // scratch lies in the lender's bank records, lent to a prompt group
        // beside the conveyor's tier records; on the one store its lent
        // records are both's.
        let placement = match plan.placer() {
            Placer::Tier => {
                let lender = plan.scratch().lender;
                let lent = usize::try_from(lender.records).map_err(|_| "lent records exceed the host")?;
                Placement::Tier { bank_maps, tier_maps: landing_maps, lent: (lender.group, lent) }
            }
            Placer::Bank => Placement::Bank { landing_maps },
        };
        let reserve = usize::try_from(plan.tier_reserve()).map_err(|_| "the tier's reserve exceeds the host")?;
        TierStart { records, lent, reserve, placement }
    };
    // A group's commits span its ceiling, as its entries bind them, so a
    // resize keeps them. Initial bank record views are delivered into
    // `Group::bank` for `Placer::Tier` so promotions never touch `Stores`.
    let groups = layout
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let mut watched = Group {
                commits: Words::new(group.commits.clone(), plan.records(index), 0)?,
                tier: Some(tiers[index].clone()),
                bank: Vec::new(),
                // The bank records the directory places into: none where
                // landings land into the bank itself (their records are the
                // landings').
                records: match plan.placer() {
                    Placer::Tier => usize::try_from(plan.committed_records(index))
                        .map_err(|_| "committed records exceed the host")?,
                    Placer::Bank => 0,
                },
            };
            if plan.placer() == Placer::Tier {
                let backed = bank.backed(index);
                watched.join_bank(&backed, bank.record_views(index, &backed));
            }
            Ok(watched)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let compute = super::HostCompute::serving(site_layers, &threads)?;
    let tables = watched.iter().map(|layer| layer.scratch.clone()).collect();
    Watcher::start(WatcherParts {
        layers: watched,
        group_reads,
        groups,
        load: LoadWords { flags },
        pools,
        extents: Extents {
            widest: usize::try_from(plan.widest().get()).map_err(|_| "experts exceed the host")?,
            windows: plan.windows(),
            window_rounds: plan.window_rounds(),
            pass_launches: plan.pass.get(),
            round_records: usize::try_from(binding.round_records)
                .map_err(|_| "round records exceed the host")?,
            words: binding.words,
            selected: usize::try_from(binding.selected).map_err(|_| "choices exceed the host")?,
            demand: usize::try_from(plan.demand_records()).map_err(|_| "the demand reserve exceeds the host")?,
            placer: plan.placer(),
            recording: plan.recording,
        },
        slots,
        whole: plan.whole(),
        staging: u32::try_from(binding.staging).map_err(|_| "staging records exceed u32")?,
        host,
        carrier,
        source: ExpertSource::open(&files, &urgent)?,
        compute,
        copier,
        steps,
        spans: device.backend().times_spans() && device.backend().parks_in_kernels(),
        trip: crate::experts::exchange::abort(),
        line: plan.paths.line(),
    })
    .map(|watcher| (watcher, lines, tables))
}
