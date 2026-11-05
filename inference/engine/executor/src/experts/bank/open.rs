use super::{ExpertBank, PoolConveyor, allocate_placed, host_views, landed_tensors, placed_words, slot_maps, write_blocks};
use crate::experts::conveyor::{Carrier, LentPool, Lending};
use crate::experts::source::{ExpertSource, Records};
use crate::experts::watcher::{Copier, CopyTensor, Extents, Group, Layer, ListRecords, LoadWords, PlacedWords, Pool, Steps, StoredTensor, Watcher, WatcherParts};
use crate::experts::directory::{HostTier as TierStart, Placement};
use crate::experts::landing::TierTensor;
use crate::experts::map::{LandingMap, StampedMap};
use crate::experts::{ExchangeTable, Words};
use crate::planning::{BankAllocation, BankLayout, ConveyorCarrier, ExpertBankPlan, Lender, PlacedAllocations, Placer, RecordTensors, StagingPool};
use magnitude_artifacts::gguf::GgufArtifact;
use seismic::{Device, Tensor};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use super::resize::{Commitment, HostResize, HostTier, Resize};
use crate::storage_thread::{Releases, StorageThread};
use crate::experts::site::site_lines;

impl ExpertBank {
    /// Allocate `plan`'s bank on `device`, empty, and start its watcher over
    /// the files `artifact` stores the expert stacks in; each streamed
    /// layer's admitted CPU site (`sites`) served by its host
    /// compute executor, its lines those the job stored (`cpu`); its device
    /// storage released on the load's `releases` thread.
    pub(crate) fn open(
        device: &Device,
        plan: &ExpertBankPlan,
        artifact: &GgufArtifact,
        sites: &[Option<crate::native::BankSite>],
        lines: std::sync::Arc<[crate::assessment::BankLines]>,
        cpu: &crate::assessment::StoredCpuSite,
        releases: Releases,
    ) -> Result<Self, String> {
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
        // Every allocation of the plan's layout: words and staging
        // host-shared; each group's record tensors
        // reserved for every expert of its layers, its slots committed, so a
        // resize keeps every binding of them, and a host tier's again in
        // pinned upload storage; the stacks' per-expert scales imported.
        let allocated = plan.layout()?.try_map(|allocation| {
            let failed = |error: &dyn std::fmt::Display| format!("expert bank allocation failed: {error}");
            let held = |tensor: Result<Tensor, _>| tensor.map(|tensor| (tensor, None)).map_err(|error| failed(&error));
            match allocation {
                BankAllocation::Shared { element, extents } => held(Tensor::shared(device, *element, extents)),
                BankAllocation::Words { element, extents } => held(Tensor::words(device, *element, extents)),
                BankAllocation::Protocol { element, extents } => held(Tensor::protocol_words(device, *element, extents)),
                BankAllocation::Device { element, extents } => held(Tensor::zeros(device, *element, extents)),
                BankAllocation::Reserved {
                    contract,
                    element,
                    records,
                    matrix,
                    committed,
                    carries,
                    ..
                } => {
                    let extents = [*records, matrix[0], matrix[1]];
                    BankAllocation::reserve(device, *contract, *element, &extents, *committed, *carries)
                        .map(|(tensor, resizer)| (tensor, Some(resizer)))
                    .map_err(|error| failed(&error))
                }
                BankAllocation::Imported(scale) => crate::residency::import_scale(device, artifact, scale)
                    .map(|tensor| (tensor, None))
                    .map_err(|error| failed(&error)),
            }
        })?;
        let layout = allocated.try_map(|(tensor, _)| Ok::<_, String>(tensor.clone()))?;
        // Each list copy's blocks, its grid's first word (`routed_list_grid`
        // writes the rest): a list's whole grid, until a tier copy's over the
        // link are measured (`Self::tier_copy_blocks`).
        for grid in [&layout.load.tier_grid, &layout.load.seed_grid] {
            write_blocks(grid, magnitude_kernels::LIST_BLOCKS)?;
        }
        // A host tier's record tensors' resizers, which its resizes take.
        let host = allocated
            .host
            .map(|tiers| {
                let resizers = tiers
                    .into_iter()
                    .flat_map(|tier| tier.into_iter().map(|(_, resizer)| resizer))
                    .collect::<Option<Vec<_>>>()
                    .ok_or("a host tier's record tensor is not reserved")?;
                let ceiling = plan.host_ceiling().ok_or("a host tier's records beside a plan without one")?;
                Ok::<_, String>(Mutex::new(HostTier {
                    slots: Arc::new(AtomicU64::new(0)),
                    ceiling,
                    resize: HostResize::Idle(resizers),
                    last_resize: None,
                    thread: StorageThread::start("expert-bank-host-tier")?,
                }))
            })
            .transpose()?;
        // Each group's record tensors' resizers, which the bank's resizes
        // take and return.
        let resizers = allocated
            .groups
            .into_iter()
            .map(|group| {
                group
                    .records
                    .into_iter()
                    .map(|(_, resizer)| resizer.ok_or("a bank's record tensor is not reserved"))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (pools, pool_of) = plan.staging_pools();
        let conveyor = plan.conveyor();
        // Each pool's conveyor: its view of the lender's records as the
        // conveyor's layout lays it there, its records in its stored form:
        // the one store's on unified memory, the host tier's beside a
        // discrete device (`ConveyorCarrier`).
        let carriers = match conveyor.carrier {
            ConveyorCarrier::Bank(lender) => &layout.groups[lender.group].records,
            ConveyorCarrier::HostTier(lender) => {
                &layout.host.as_ref().ok_or("a host tier's conveyor beside no tier records")?[lender.group]
            }
        };
        let conveyors = (0..pools.len())
            .map(|index| {
                let pool: &StagingPool = &pools[index];
                let view = carriers
                    .get(plan.form().carrier(pool.kind))
                    .ok_or("a staging pool of a tensor the records lack")?
                    .region(pool.source, &[conveyor.conveyor.records, pool.matrix[0], pool.matrix[1]], conveyor.conveyor.offsets[index])
                    .map_err(|error| format!("conveyor view failed: {error}"))?;
                Ok(PoolConveyor { conveyor: view })
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Each scratch form's view of the lender's bank records as the
        // scratch's layout lays it there, and each layer's view of its form
        // (`ExpertBankPlan::scratch`).
        let lent = plan.scratch();
        let lender = &layout.groups[lent.lender.group].records;
        let forms = plan.layer_scratch_forms();
        let scratch = plan
            .layers
            .iter()
            .zip(&forms)
            .enumerate()
            .map(|(index, (layer, forms))| {
                let binding = plan.binding(index)?;
                binding.record.try_map(|kind, tensor| {
                    let form = *forms.get(kind).ok_or("a scratch form of a tensor the record lacks")?;
                    lender
                        .get(kind)
                        .ok_or("scratch of a tensor the lender lacks")?
                        .region(tensor.resident, &[layer.experts, tensor.matrix[0], tensor.matrix[1]], lent.layout.offsets[form])
                        .map_err(|error| format!("scratch view failed: {error}"))
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let slot_maps = slot_maps(plan, &layout)?;
        // Derived once here, so no resize derives them on the step's thread.
        let deriving = std::time::Instant::now();
        let counts = Arc::new(plan.counts()?);
        tracing::info!("derived the bank's counts up to {} slots in {:.1} ms", plan.ceiling, deriving.elapsed().as_secs_f64() * 1e3);
        let floor = allocate_placed(device, plan, std::num::NonZeroU64::MIN)
            .map_err(|error| format!("expert bank allocation failed: {error}"))?;
        // The slots the bank commits, which its watcher's telemetry reads.
        let slots = Arc::new(AtomicU64::new(plan.slots));
        let (watcher, lines) =
            watch(plan, &counts, artifact, &layout, &slot_maps, &floor, &pool_of, &conveyors, sites, lines, cpu, slots.clone())?;
        let move_lists = layout
            .groups
            .iter()
            .map(|group| Words::new(group.moves.clone(), group.moves.byte_len() / 4, 0))
            .collect::<Result<Vec<_>, String>>()?;
        let landed = landed_tensors(plan, &layout).map(Arc::from);
        let resize_thread = StorageThread::start("expert-bank-resize")?;
        let bank = Self {
            layout,
            floor,
            held: Mutex::new(None),
            device: device.clone(),
            pool_of,
            conveyors,
            scratch,
            host,
            slot_maps,
            watcher,
            lines,
            move_lists,
            commitment: Mutex::new(Commitment {
                slots,
                resize: Resize::Idle(resizers.into_iter().flatten().collect()),
                moves: None,
            }),
            last_resize: Mutex::new(None),
            resize_thread,
            release_thread: releases,
            rises: Arc::new(Mutex::new(())),
            landed,
            plan: plan.clone(),
            counts,
        };
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
/// telemetry reads
/// the bank's committed `slots` (`Commitment::slots`).
#[allow(clippy::too_many_arguments)]
fn watch(
    plan: &ExpertBankPlan,
    counts: &crate::planning::BankCounts,
    artifact: &GgufArtifact,
    layout: &BankLayout<Tensor>,
    slot_maps: &[Tensor],
    floor: &PlacedAllocations<Tensor>,
    pool_of: &[RecordTensors<usize>],
    conveyors: &[PoolConveyor],
    sites: &[Option<crate::native::BankSite>],
    lines: std::sync::Arc<[crate::assessment::BankLines]>,
    cpu: &crate::assessment::StoredCpuSite,
    slots: Arc<AtomicU64>,
) -> Result<(Watcher, std::sync::Arc<[crate::assessment::BankLines]>), String> {
    let words = |tensor: &Tensor, initial| Words::new(tensor.clone(), tensor.byte_len() / 4, initial);
    let binding = plan.binding(0)?;
    // The steps the watcher serves, under whose lock the host writes every
    // map entry (`experts::map`).
    let steps = Arc::new(Steps::new(plan.flights, plan.places, crate::experts::exchange::Pulse::of(layout.load.resident.device().backend())));
    // The floor's placed words, which the bank holds until a group holds
    // more places.
    let PlacedWords { flags, layers: group_reads } = placed_words(plan, floor)?;
    // Each group's tier tensors, which landings convert its stored experts
    // into: each routed tensor's committed records at the load, pinned
    // backed while the watcher holds them (a growth's views replace them,
    // `watcher::Join`; a shrink cuts them, `Shared::shrink_store`).
    let start = match plan.placer() {
        Placer::Tier => 0,
        Placer::Bank => plan.slots,
    };
    let tiers = landed_tensors(plan, layout)
        .map(|tensors| host_views(plan, &tensors, |group| counts.landed_records(group, start)))
        .transpose()?
        .map(|views| {
            plan.groups
                .iter()
                .zip(views)
                .map(|(planned, views)| {
                    planned
                        .record
                        .iter()
                        .zip(views)
                        .map(|((_, record), records)| {
                            TierTensor::new(records, record.source, record.resident, record.matrix, record.stored_bytes()?)
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?;
    // The host compute executor's threads, sized once for the load: the
    // executor lands every file miss into the tier and serves every sited
    // layer's CPU misses.
    let device = layout.load.resident.device();
    // A decode request reads at most a row's selected experts of one layer,
    // each a record stored as its routed tensors' ranges: the read engine
    // reads the files for it.
    let urgent = plan.urgent()?;
    let read_cores = magnitude_artifacts::read_cores(artifact.sources().map(|source| &**source), &urgent)
        .map_err(|error| format!("the read engine's cores: {error}"))?;
    let machine = super::Machine::discover(&device, read_cores)?;
    let (site_layers, lines, threads) = site_lines(plan, layout, tiers.as_deref(), sites, lines, &machine, cpu, steps.pulse())?;
    let mut files = Vec::<std::path::PathBuf>::new();
    let mut watched = Vec::with_capacity(layout.layers.len());
    let mut bank_maps = Vec::new();
    let mut landing_maps = Vec::new();
    let stacks = plan.stored_stacks(artifact)?;
    for (index, layer) in layout.layers.iter().enumerate() {
        let group = plan.layers[index].group;
        let stored = plan.groups[group]
            .record
            .iter()
            .zip(stacks[index].iter())
            .zip(pool_of[index].iter())
            .map(|(((_, tensor), (_, stack)), (_, &pool))| {
                let file = match files.iter().position(|path| *path == stack.path) {
                    Some(file) => file,
                    None => {
                        files.push(stack.path.clone());
                        files.len() - 1
                    }
                };
                Ok::<_, String>(StoredTensor {
                    file,
                    base: stack.offset + stack.first,
                    pitch: stack.pitch,
                    bytes: tensor.stored_bytes()?,
                    pool,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        watched.push(Layer {
            group,
            experts: usize::try_from(plan.layers[index].experts).map_err(|_| "experts exceed the host")?,
            map: view,
            file: ExchangeTable::new(layer.file.clone(), plan.binding(index)?.words.file(), steps.pulse())?,
            scratch: ExchangeTable::new(layer.scratch.clone(), plan.binding(index)?.words.scratch(), steps.pulse())?,
            rings: words(&layer.rings, 0)?,
            hints: plan.predicted(index).collect(),
            routing: words(&layer.routing, 0)?,
            policy: words(&layer.policy, 0)?,
            site: site_layers.iter().find(|site| site.layer() == index).map(|site| site.slots().clone()),
            stored,
            landing: layer
                .landing
                .grids
                .as_ref()
                .map(|grids| {
                    let binding = plan.binding(index)?;
                    let groups = |rows: u64| crate::operators::routed::streamed::convert_groups(rows).map_err(str::to_owned);
                    Ok::<_, String>(Arc::new(crate::experts::landing::DeviceWords {
                        list: words(&layer.landing.list, 0)?,
                        grids: words(grids, 0)?,
                        rows_groups: groups(binding.record.up.matrix[0])?,
                        columns_groups: groups(binding.record.down.matrix[0])?,
                        bounds: crate::experts::ListBounds { from: binding.staging, to: binding.records },
                    }))
                })
                .transpose()?,
            lists: {
                let binding = plan.binding(index)?;
                ListRecords { bank: binding.records, tier: binding.tier, conveyor: binding.conveyor }
            },
        });
    }
    // A staging or conveyor record is one contiguous range of its pool's,
    // where its view lays it (a lent conveyor's packed in the lender's
    // chunks), which reads land in in place.
    let (staging_pools, _) = plan.staging_pools();
    let pool_records = |pool: &StagingPool, name: &str, tensor: &Tensor, records| {
        Records::new(tensor.clone(), pool.record_bytes()?, records)
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
        let lent = &plan.groups[lender.group].record;
        Ok::<_, String>(Lending {
            group: lender.group,
            records: usize::try_from(lender.records).map_err(|_| "lent records exceed the host")?,
            pools: staging_pools
                .iter()
                .zip(&conveyor.conveyor.offsets)
                .zip(conveyors)
                .map(|((pool, &offset), view)| {
                    Ok(LentPool {
                        record: lent
                            .get(plan.form().carrier(pool.kind))
                            .ok_or("a staging pool of a tensor the lender's records lack")?
                            .record_bytes()?,
                        offset,
                        conveyor: view
                            .conveyor
                            .record_placement()
                            .ok_or("a conveyor's records lie whole in no chunk of their lender")?,
                    })
                })
                .collect::<Result<_, String>>()?,
        })
    };
    // What promotions copy with, beside a host tier: the device's transfer
    // queue, and each group's tier and bank record tensors of each routed
    // tensor, whose records are one form (`BankPart::Tier`'s, the bank's
    // resident one).
    let copier = layout
        .host
        .as_ref()
        .map(|tiers| {
            let copies = seismic::Transfer::new(&device)
                .map(|transfer| super::copies::Copies::new(transfer, steps.events().clone()))
                .map_err(|error| format!("the device's transfers: {error}"))?;
            let groups = plan
                .groups
                .iter()
                .enumerate()
                .map(|(group, planned)| {
                    planned
                        .record
                        .iter()
                        .map(|(kind, _)| {
                            let tier = tiers[group].get(kind).ok_or("a tier holds its record's tensors")?;
                            let bank = layout.groups[group].records.get(kind).ok_or("a bank group holds its record's tensors")?;
                            let tier_at = tier.record_placement().ok_or("a host tier's records lie whole in no chunk")?;
                            let bank_at = bank.record_placement().ok_or("a bank's records lie whole in no chunk")?;
                            Ok::<_, String>(CopyTensor { tier: tier.clone(), tier_at, bank: bank.clone(), bank_at })
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok::<_, String>(Copier { copies, groups })
        })
        .transpose()?;
    let carrier = Carrier {
        records: usize::try_from(conveyor.conveyor.records).map_err(|_| "conveyor records exceed the host")?,
        lending: lending(&conveyor.carrier.lender())?,
        pools: staging_pools
            .iter()
            .zip(conveyors)
            .map(|(pool, view)| pool_records(pool, "conveyor", &view.conveyor, conveyor.conveyor.records))
            .collect::<Result<Vec<_>, String>>()?,
    };
    // The records the host lands into: a host tier's, each group's at its
    // floor (backed to the plan's size beside the fill), or where the host
    // places, the bank's own.
    let host = tiers
        .as_ref()
        .map(|tiers| {
            let records = tiers
                .iter()
                .map(|tensors| u32::try_from(tensors[0].records.count()).map_err(|_| "host records exceed u32".to_owned()))
                .collect::<Result<Vec<_>, String>>()?;
            let lender = conveyor.carrier.lender();
            let lent = (0..records.len())
                .map(|group| match group == lender.group {
                    true => u32::try_from(lender.records).map_err(|_| "lent records exceed u32".to_owned()),
                    false => Ok(0),
                })
                .collect::<Result<Vec<_>, String>>()?;
            // Beside a host tier the bank's records write its maps, and the
            // scratch lies in the lender's bank records, lent to a prompt
            // group beside the conveyor's tier records; on the one store its
            // lent records are both's.
            let placement = match plan.placer() {
                Placer::Tier => {
                    let lender = plan.scratch().lender;
                    let lent = usize::try_from(lender.records).map_err(|_| "lent records exceed the host")?;
                    Placement::Tier { bank_maps, tier_maps: landing_maps, lent: (lender.group, lent) }
                }
                Placer::Bank => Placement::Bank { landing_maps },
            };
            Ok::<_, String>(TierStart { records, lent, placement })
        })
        .transpose()?
        .ok_or("a streamed bank's misses land in host records")?;
    // A group's commits span its ceiling, as its entries bind them, so a
    // resize keeps them.
    let groups = layout
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            Ok(Group {
                commits: Words::new(group.commits.clone(), plan.records(index), 0)?,
                tier: tiers.as_ref().map(|tiers| tiers[index].clone()),
                // The bank records the directory places into: none where
                // landings land into the bank itself (their records are the
                // landings').
                records: match plan.placer() {
                    Placer::Tier => usize::try_from(plan.committed_records(index))
                        .map_err(|_| "committed records exceed the host")?,
                    Placer::Bank => 0,
                },
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let compute = super::HostCompute::serving(site_layers, &threads)?;
    Watcher::start(WatcherParts {
        layers: watched,
        group_reads,
        groups,
        load: LoadWords {
            flags,
            lists: layout
                .load
                .lists
                .iter()
                .map(|list| words(list, 0))
                .collect::<Result<Vec<_>, String>>()?,
            tiered: words(&layout.load.tiered, 0)?,
            seeds: words(&layout.load.seeds, 0)?,
        },
        pools,
        extents: Extents {
            widest: usize::try_from(plan.widest().get()).map_err(|_| "experts exceed the host")?,
            round_records: usize::try_from(binding.round_records)
                .map_err(|_| "round records exceed the host")?,
            words: binding.words,
            selected: usize::try_from(binding.selected).map_err(|_| "choices exceed the host")?,
            demand: usize::try_from(plan.demand_records()).map_err(|_| "the demand reserve exceeds the host")?,
            reserve: usize::try_from(plan.tier_reserve()).map_err(|_| "the tier's reserve exceeds the host")?,
            placer: plan.placer(),
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
    })
    .map(|watcher| (watcher, lines))
}
