//! A streamed layer's bank entries (`ExpertBankKernels`), each once at its
//! prepared specialization over a fixture bank, then its routed form's
//! shared entries as the bank serves them.
//!
//! The fixture holds every word and record in the storage the load's bank
//! holds it in (`ExpertBankPlan::layout`): words and conveyor
//! host-shared, the bank's records reserved device storage, a host tier's
//! reserved upload storage, and the scratch tier the device's own. No host runs: each word a park waits for is raised before
//! the park, as serving's host raises it, so every entry runs to completion.
//! Each entry's effect is checked: the words it publishes, and each record a
//! conversion, copy or move writes, against the host's conversion of what
//! was staged as the entries' repack binding converts it
//! (`Element::repack_host`), over bytes no entry writes
//! (`RecordFixture::marker`).

use super::super::*;
use super::target::{ExpertReads, qualify_general_routed, qualify_routed};
use super::*;
use crate::operators::routed::streamed::{convert_groups, record_grid};
use crate::planning::{
    BankAllocation, BankBinding, BankedFeedForward, RecordTensor, RecordTensors,
};
use magnitude_kernels::constants::{
    DECODE_GRIDS, FILE_CHOICE, FILE_COUNT, FILE_RECORD, LIST_COUNT, LIST_ENTRIES, LIST_FROM, LIST_SEQUENCE,
    LIST_TO, MAP_FROM, MAP_RECORD, MAP_UNTIL, MAP_WORDS, POLICY_HOME, POLICY_PRICED, POLICY_RESOLVED, POLICY_WHOLE, POLICY_WORDS, RING_COUNT,
    RING_SEQUENCE, ROUND_GRIDS, SCRATCH_ASK, SCRATCH_ROUND, SCRATCH_STAGED, SLOT_ANSWER, SLOT_REQUEST,
    SOURCE_RECORD, SOURCE_ROWS, SOURCE_SITE, STEP_HIGH, STEP_LOW, STEP_WORDS, WAIT_GRID,
};
use magnitude_kernels::{
    SlotTable, list_extent, routed_convert, routed_copy, routed_list_grid, routed_list_stage, routed_move, routed_pass_map, routed_route_streamed, routed_seed_commit,
    routed_scratch_grids, routed_scratch_request, routed_scratch_wait, routed_select_streamed, routed_step,
    routed_tier_wait,
};

/// The fixture step's sequence.
const SEQUENCE: i32 = 1;

/// The fixture bank's records: the one a shrink's move lands in (0), and the
/// one a prompt group's seed lands in (1).
const RECORDS: u64 = 2;

/// Qualify `binding`'s bank entries `kernels` of the streamed feed-forward
/// `banked`, whose resident entries are `feed_forward` and whose rows are
/// `rows`: one decode row's route (each lead, each whole word), the tier
/// wait of its absent expert 0 landed in tier record 0, a scratch-served
/// step's request, its round 0 (wait, grids, every tensor's convert), the
/// seed of expert 0 into record 1 and its commit, which the host then maps,
/// and a shrink's
/// move of record 1 into record 0; then the shared routed entries over the
/// map the seed left (expert 0 held in record 1 for the step), the decode
/// entries' sources naming it.
///
/// Where the layer is sited (`ExpertBankKernels::site`), its site's verdict
/// under `requirement` as the measurement job stored it in `cache`
/// (`site::site_verdict`), which runs nothing; else none.
#[allow(clippy::too_many_arguments)]
pub(super) fn qualify_bank(
    device: &Device,
    banked: BankedFeedForward,
    binding: BankBinding,
    kernels: &ExpertBankKernels,
    feed_forward: &AttestedFeedForward,
    rows: SublayerRows,
    requirement: &seismic::NumericalRequirement,
    cache: Option<&crate::kernel_cache::KernelCache>,
) -> Result<Option<crate::assessment::Priced<()>>, CatalogFailure> {
    let label = format!("{banked:?} {binding:?}");
    let label = label.as_str();
    let ExpertBankKernels {
        route,
        tier_wait,
        step: step_kernel,
        scratch_request,
        scratch_wait,
        scratch_grids,
        list_stage,
        pass_map,
        convert,
        placement,
        site,
    } = kernels;
    let served = Served::of(banked, route, feed_forward, label)?;
    let mut records = binding
        .record
        .try_map(|_, tensor| RecordFixture::new(device, tensor, binding.experts, label))?;
    let selected = word(binding.selected, "routed_bank", label)?;
    let words = |extents: &[u64], set: &[(u64, i32)], entry| shared_words(device, extents, set, entry, label);

    // The step's sequence.
    let mut step = words(&[STEP_WORDS], &[], "routed_step")?;
    step_kernel
        .call(routed_step::Args {
            step: &mut step,
            sequence: &words(&[STEP_WORDS], &[(STEP_LOW, SEQUENCE)], "routed_step")?,
        })
        .map_err(|error| qualification_dynamic("routed_step", label, error))?;
    require_words(&step, &[(STEP_LOW, SEQUENCE), (STEP_HIGH, 0)], "routed_step", label)?;

    served.qualify_routes(device, binding, rows, &step, label)?;

    // The decode tier wait of one absent choice, expert 0 (one row's first
    // choice of its `selected`, its sources naming no record), which the
    // host landed into tier record 0 and answered before the call, so the
    // wait finds its answer at once (`call_answered`: it runs standalone on
    // every backend, where kernels wait in place and where a park is a
    // submission boundary alike).
    let entry = "routed_tier_wait";
    let map = words(&[binding.experts, MAP_WORDS], &unmapped(binding.experts), entry)?;
    // The request, in the step's slot of the file table: choice 0 expert 0,
    // absent; every other choice held (as the route requests a held one,
    // -1 - expert).
    let table = binding.words.file();
    let slot = table.lane(SEQUENCE as u64, 0);
    let others = (1..binding.selected).map(|choice| (slot + SlotTable::file_entry(choice) + FILE_CHOICE, -1));
    // Its answer names record 0 by its record word: tier record 0 (-2)
    // across a link, bank record 0 on unified memory.
    let landed = match binding.placer {
        crate::planning::Placer::Tier => -2,
        crate::planning::Placer::Bank => 0,
    };
    let request = [
        (slot + SLOT_REQUEST, SEQUENCE),
        (slot + SLOT_ANSWER, SEQUENCE),
        (slot + FILE_COUNT, selected),
        (slot + SlotTable::file_entry(0) + FILE_CHOICE, 0),
        (slot + SlotTable::file_entry(0) + FILE_RECORD, landed),
    ]
    .into_iter()
    .chain(others)
    .collect::<Vec<_>>();
    let file = words(&table.extents(), &request, entry)?;
    let count = binding.selected;
    // Its sources: none for the GPU, none the CPU's, so the landed choice
    // is the GPU's, served in place.
    let mut sources =
        semantic_i32(device, &[SOURCE_ROWS, 1, count], &vec![-1; (SOURCE_ROWS * count) as usize], entry, label)?;
    let wait_grid = semantic_i32(device, &[3], &[1, 1, 1], entry, label)?;
    let policy = words(&[POLICY_WORDS], &[], entry)?;
    let mut landing = words(&[magnitude_kernels::list_extent(1)], &[], entry)?;
    tier_wait
        .call_answered(routed_tier_wait::Args {
            step: &step,
            file: &file,
            sources: &mut sources,
            grid: &wait_grid,
            policy: &policy,
            landing: &mut landing,
        })
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    require_words(&sources, &[(SOURCE_RECORD * count, landed), (SOURCE_SITE * count, -1)], entry, label)?;

    // A scratch-served step's request, each choice its own expert, and its
    // round 0: the host answered the request with the count of records its
    // rounds convert (one), wrote round 0's list (staging record 0 into
    // scratch slot 0), then answered every round.
    let table = binding.words.scratch();
    let ask = table.lane(SEQUENCE as u64, SCRATCH_ASK);
    let first = table.lane(SEQUENCE as u64, SCRATCH_ROUND);
    let answered = (0..table.lanes()).map(|lane| (table.lane(SEQUENCE as u64, lane) + SLOT_ANSWER, SEQUENCE));
    let mut scratch_table = words(
        &table.extents(),
        &[(ask + SCRATCH_STAGED, 1)].into_iter().chain(answered).collect::<Vec<_>>(),
        "routed_scratch_request",
    )?;
    let mut flags = words(&[binding.widest], &[], "routed_scratch_request")?;
    scratch_request
        .call(routed_scratch_request::Args {
            routes: &semantic_i32(device, &[1, binding.selected], &(0..selected).collect::<Vec<_>>(), "routed_scratch_request", label)?,
            step: &step,
            flags: &mut flags,
            scratch: &mut scratch_table,
            resident: &words(&[1], &[], "routed_scratch_request")?,
        })
        .map_err(|error| qualification_dynamic("routed_scratch_request", label, error))?;
    require_words(&scratch_table, &[(ask + SLOT_REQUEST, SEQUENCE), (ask + SLOT_REQUEST + 1, 0)], "routed_scratch_request", label)?;
    let requested = (0..binding.selected).map(|expert| (expert, 1)).collect::<Vec<_>>();
    require_words(&flags, &requested, "routed_scratch_request", label)?;
    let list = |from, to| [(LIST_SEQUENCE, SEQUENCE), (LIST_COUNT, 1), (LIST_ENTRIES + LIST_FROM, from), (LIST_ENTRIES + LIST_TO, to)];
    // Conveyor record 0 into scratch slot 0.
    let convert_list = words(&[list_extent(1)], &list(0, 0), "routed_convert")?;
    // The seeds: scratch slot 0, which holds expert 0, into record 1.
    let seeds = words(&[list_extent(binding.widest)], &list(0, 1), "routed_copy")?;
    let gate_conveyor = gate_or_empty(&records, |fixture| &fixture.conveyor, label)?;
    let entry = "routed_scratch_wait";
    scratch_wait
        .call_answered(routed_scratch_wait::Args {
            step: &step,
            scratch: &mut scratch_table,
            round: 0,
            list: &convert_list,
            tiered: &words(&[list_extent(binding.widest)], &[], entry)?,
            conveyor_gate: &gate_conveyor,
            conveyor_up: &records.up.conveyor,
            conveyor_down: &records.down.conveyor,
            seeds: &seeds,
            map: &words(&[binding.experts, MAP_WORDS], &[], entry)?,
            // Round 0 always parks.
            grid: &semantic_i32(device, &[3], &[1, 1, 1], entry, label)?,
        })
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    require_words(&scratch_table, &[(first + SLOT_REQUEST, SEQUENCE)], entry, label)?;
    let entry = "routed_scratch_grids";
    let rounds = binding.words.rounds;
    let mut grids = semantic_i32(device, &[rounds, ROUND_GRIDS, 3], &vec![0; (rounds * ROUND_GRIDS * 3) as usize], entry, label)?;
    let groups = |rows| convert_groups(rows).map_err(|error| qualification_dynamic(entry, label, error));
    scratch_grids
        .call(routed_scratch_grids::Args {
            step: &step,
            scratch: &scratch_table,
            grids: &mut grids,
            round_records: word(binding.round_records, entry, label)?,
            rows_groups: groups(binding.record.up.matrix[0])?,
            columns_groups: groups(binding.record.down.matrix[0])?,
        })
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    // The round's convert list staged into device memory, which its
    // converts read.
    let entry = "routed_list_stage";
    let mut staged_list = words(&[list_extent(1)], &[], entry)?;
    list_stage
        .call(routed_list_stage::Args { list: &convert_list, step: &step, staged: &mut staged_list })
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    require_words(&staged_list, &list(0, 0), entry, label)?;
    // Each tensor's convert on round 0's grid of its records, which the
    // grids wrote.
    let entry = "routed_convert";
    let round_grid = |grid: u64| {
        grids
            .slice_leading(0, 1)
            .and_then(|round| round.reshape(&[ROUND_GRIDS, 3]))
            .and_then(|round| round.slice_leading(grid, grid + 1))
            .and_then(|grid| grid.reshape(&[3]))
            .map_err(|error| qualification_dynamic(entry, label, error))
    };
    for (kind, (kernel, fixture)) in convert.as_ref().zip(records.as_mut()).into_tensors() {
        kernel
            .call(routed_convert::Args {
                conveyor: &fixture.conveyor,
                list: &staged_list,
                step: &step,
                destination: &mut fixture.scratch,
                grid: &round_grid(record_grid(kind))?,
            })
            .map_err(|error| qualification_dynamic(entry, label, error))?;
        require_bytes(&fixture.scratch, &fixture.converted, entry, label)?;
    }

    // Expert 0 into record 1: where the device copies into the bank, the
    // prompt group's seed and its commit, then a shrink's move of record 1 into record 0
    // over bytes no entry wrote; where the host lands, its landing's bytes.
    // Then the host maps it, held from the step on.
    match placement {
        Some(crate::native::DevicePlacement { seed, seed_commit, list_grid, moves }) => {
            // A list's launch grid for its one entry: one row of a copy
            // within device memory's blocks, which the host writes and the
            // kernel keeps; and the list staged beside it.
            let grid_of = |listed: &Tensor, entry| {
                let blocks = magnitude_kernels::LIST_BLOCKS as i32;
                let mut grid = semantic_i32(device, &[3], &[blocks, 0, 0], entry, label)?;
                let mut staged = words(&[list_extent(binding.widest)], &[], entry)?;
                list_grid
                    .call(routed_list_grid::Args { list: listed, step: &step, grid: &mut grid, staged: &mut staged })
                    .map_err(|error| qualification_dynamic("routed_list_grid", label, error))?;
                require_words(&grid, &[(0, blocks), (1, 1), (2, 1)], "routed_list_grid", label)?;
                require_words(&staged, &list(0, 1), "routed_list_grid", label)?;
                Ok::<_, CatalogFailure>((grid, staged))
            };
            // The prompt group's seed of expert 0 into record 1, over bytes
            // no entry wrote; record 0 is left as it was.
            let entry = "routed_copy";
            let (seed_grid, staged) = grid_of(&seeds, entry)?;
            for (kernel, fixture) in seed.as_ref().zip(records.as_mut()) {
                fixture
                    .records
                    .write_from_host(&fixture.marker.repeat(2))
                    .map_err(|error| qualification_dynamic(entry, label, error))?;
                kernel
                    .call(routed_copy::Args {
                        source: &fixture.scratch,
                        list: &staged,
                        step: &step,
                        destination: &mut fixture.records,
                        grid: &seed_grid,
                    })
                    .map_err(|error| qualification_dynamic(entry, label, error))?;
                require_bytes(&fixture.records, &[&fixture.marker[..], &fixture.converted[..]].concat(), entry, label)?;
            }
            // Its commit of record 1.
            let entry = "routed_seed_commit";
            let mut commits = words(&[RECORDS], &[], entry)?;
            seed_commit
                .call(routed_seed_commit::Args { list: &seeds, step: &step, scratch: &scratch_table, commits: &mut commits })
                .map_err(|error| qualification_dynamic(entry, label, error))?;
            require_words(&commits, &[(0, 0), (1, SEQUENCE)], entry, label)?;
            // A shrink's move of record 1 into record 0, over bytes no entry wrote.
            let entry = "routed_move";
            let move_list = words(&[list_extent(1)], &list(1, 0), entry)?;
            for (kernel, fixture) in moves.records.as_ref().zip(records.as_mut()) {
                fixture
                    .records
                    .write_from_host(&[&fixture.marker[..], &fixture.converted[..]].concat())
                    .map_err(|error| qualification_dynamic(entry, label, error))?;
                kernel
                    .call(routed_move::Args {
                        records: &mut fixture.records,
                        list: &move_list,
                    })
                    .map_err(|error| qualification_dynamic(entry, label, error))?;
                require_bytes(&fixture.records, &fixture.converted.repeat(2), entry, label)?;
            }
        }
        None => {
            let failed = |error: &dyn fmt::Display| qualification_dynamic("host placement", label, error);
            for fixture in records.as_mut() {
                fixture
                    .records
                    .write_from_host(&[&fixture.marker[..], &fixture.converted[..]].concat())
                    .map_err(|error| failed(&error))?;
            }
        }
    }
    set_words(&map, &held(0, 1), "host placement", label)?;

    // A merge set's pass's map resolved into device memory: expert 0 held
    // in record 1 from the step on, every other expert in none, and the
    // step.
    let entry = "routed_pass_map";
    let mut resolved = words(&[binding.experts, MAP_WORDS], &[], entry)?;
    let mut pass_step = words(&[STEP_WORDS], &[], entry)?;
    pass_map
        .call(routed_pass_map::Args { map: &map, step: &step, resolved: &mut resolved, pass_step: &mut pass_step })
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    require_words(&pass_step, &[(STEP_LOW, SEQUENCE), (STEP_HIGH, 0)], entry, label)?;
    let from = |expert: u64, record: i32| {
        let row = expert * MAP_WORDS;
        [
            (row + MAP_FROM, SEQUENCE),
            (row + MAP_FROM + 1, 0),
            (row + MAP_UNTIL, -1),
            (row + MAP_UNTIL + 1, -1),
            (row + MAP_RECORD, record),
        ]
    };
    let wanted = (0..binding.experts).flat_map(|expert| from(expert, if expert == 0 { 1 } else { -1 })).collect::<Vec<_>>();
    require_words(&resolved, &wanted, entry, label)?;

    // The shared entries read the bank's record 1; beside a host tier they
    // take its records as their entries are prepared for (`TR`), reserved
    // as the tier's, none of which a source names.
    let tier = (binding.tier > 0)
        .then(|| {
            // A tier is reserved `Shared`, on a link or in unified memory.
            let kind = crate::planning::BankTier::Host.contract_on(device);
            binding.record.try_map(|_, tensor| {
                let [n, k] = tensor.matrix;
                BankAllocation::reserve(device, kind, tensor.resident, &[binding.tier, n, k], 1, None)
                    .map(|(tier, _)| tier)
                    .map_err(|error| qualification_dynamic("routed_bank", label, error))
            })
        })
        .transpose()?;
    let reads = ExpertReads {
        map,
        step: step.clone(),
        decode_record: Some(1),
        records: records.as_ref().map(|_, fixture| fixture.records.clone()),
        stacks: records.as_ref().map(|_, fixture| fixture.stacks.clone()),
        tier,
    };
    match &served {
        Served::Fused { binding, kernels, .. } => qualify_routed(device, *binding, kernels, rows, &reads)?,
        Served::General { binding, kernels, .. } => qualify_general_routed(device, *binding, kernels, &reads)?,
    }

    Ok(site.as_ref().map(|site| super::site::site_verdict(device, &binding, site.form, requirement, cache)))
}

/// A streamed layer's routed form: its binding, its resident entries and its
/// bank's decode routes, one by `AHEAD`.
pub(super) enum Served<'a> {
    Fused {
        binding: RoutedBinding,
        kernels: &'a RoutedKernels,
        routes: &'a [Option<NativeKernel<routed_route_streamed::Entry>>; LEADS],
    },
    General {
        binding: crate::GeneralRoutedBinding,
        kernels: &'a GeneralRoutedKernels,
        selects: &'a [Option<NativeKernel<routed_select_streamed::Entry>>; LEADS],
    },
}

/// A streamed binding's route at one lead, in its routed form.
#[derive(Clone, Copy)]
enum Lead<'a> {
    Fused {
        binding: RoutedBinding,
        kernel: &'a NativeKernel<routed_route_streamed::Entry>,
    },
    General {
        binding: crate::GeneralRoutedBinding,
        kernel: &'a NativeKernel<routed_select_streamed::Entry>,
    },
}

impl<'a> Served<'a> {
    fn of(
        banked: BankedFeedForward,
        route: &'a BankRouteKernels,
        feed_forward: &'a AttestedFeedForward,
        label: &str,
    ) -> Result<Self, CatalogFailure> {
        match (banked, route, feed_forward) {
            (BankedFeedForward::Routed(binding), BankRouteKernels::Fused(routes), AttestedFeedForward::Routed(kernels)) => {
                Ok(Self::Fused { binding, kernels, routes })
            }
            (
                BankedFeedForward::GeneralRouted(binding),
                BankRouteKernels::General(selects),
                AttestedFeedForward::GeneralRouted(kernels),
            ) => Ok(Self::General { binding, kernels, selects }),
            // A parallel block's routed branch is general.
            (BankedFeedForward::Parallel(parallel), BankRouteKernels::General(selects), AttestedFeedForward::Parallel(kernels)) => {
                Ok(Self::General {
                    binding: parallel.routed,
                    kernels: &kernels.routed,
                    selects,
                })
            }
            _ => Err(qualification_dynamic(
                "routed_bank",
                label,
                "the bank's route differs from its feed-forward's form",
            )),
        }
    }

    /// Its route at `ahead`, where a layer of the binding launches one.
    fn lead(&self, ahead: usize) -> Option<Lead<'a>> {
        match *self {
            Self::Fused { binding, routes, .. } => Some(Lead::Fused { binding, kernel: routes[ahead].as_ref()? }),
            Self::General { binding, selects, .. } => Some(Lead::General { binding, kernel: selects[ahead].as_ref()? }),
        }
    }

    /// One decode row's route by each lead the binding launches and each
    /// pair of this layer's whole word and the next layer's resolved word, every expert absent from the layer's map
    /// and its tier map (where the bank has a host tier), its policy lines
    /// zero and its hint ring of no step, over zero weights. A layer not
    /// whole writes its sources and routing record, every choice's source
    /// none for the GPU (across a link every miss the CPU's, pending its
    /// landing: SOURCE_SITE -2), and its grid, and publishes its request of
    /// every choice; a whole layer sources each choice from its home record,
    /// the expert past `WHOLE_HOME`, reading no map;
    /// and a hinting route whose next layer is not resolved publishes every
    /// lead's hints.
    fn qualify_routes(
        &self,
        device: &Device,
        binding: BankBinding,
        rows: SublayerRows,
        step: &Tensor,
        label: &str,
    ) -> Result<(), CatalogFailure> {
        let entry = match self {
            Self::Fused { .. } => "routed_route_streamed",
            Self::General { .. } => "routed_select_streamed",
        };
        let words = |extents: &[u64], set: &[(u64, i32)]| shared_words(device, extents, set, entry, label);
        let zeros = |element, extents: &[u64]| semantic_zeros(device, element, extents, entry, label);
        let failed = |error: seismic::CallError| qualification_dynamic(entry, label, error);
        let (experts, next, selected) = (binding.experts, binding.next_experts, binding.selected);
        let (ring, flight) = (binding.words.ring_words(), binding.words.flight);
        let choices = word(selected, entry, label)?;
        let routing_words = binding.words.routing_words();
        let record = u64::from(SEQUENCE.unsigned_abs()) % flight * routing_words;
        // Every choice's source none for the GPU, in the sources and the
        // routing record's sources; across a link a layer not whole, sited
        // (its answer slots bound), priced, gives the CPU every miss (zero
        // lines: no link time is ever less).
        let tier_experts = crate::native::tuning::routed::tier_experts(binding);
        let sited = i32::from(tier_experts > 0);
        let site = if tier_experts > 0 { -2 } else { -1 };
        let unsourced = (0..selected)
            .flat_map(|choice| [(SOURCE_RECORD * selected + choice, -1), (SOURCE_SITE * selected + choice, site)])
            .collect::<Vec<_>>();
        let unrouted = (0..selected)
            .map(|choice| (record + binding.words.routing_source(choice), -1))
            .collect::<Vec<_>>();
        // The tier wait's grid of one group.
        let grids_written = (0..3).map(|axis| (WAIT_GRID * 3 + axis, 1)).collect::<Vec<_>>();
        for (ahead, lead) in (0..LEADS).filter_map(|ahead| Some((ahead as u64, self.lead(ahead)?))) {
            for (whole, next_resolved) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let map = words(&[experts, MAP_WORDS], &unmapped(experts))?;
                let tier = words(&[tier_experts], &(0..tier_experts).map(|expert| (expert, -1)).collect::<Vec<_>>())?;
                let policy = words(&[POLICY_WORDS], &[(POLICY_WHOLE, whole), (POLICY_HOME, whole * WHOLE_HOME), (POLICY_PRICED, 1)])?;
                let next_policy = words(&[POLICY_WORDS], &[(POLICY_RESOLVED, next_resolved)])?;
                let mut file = words(&binding.words.file().extents(), &[])?;
                let mut grids = semantic_i32(device, &[DECODE_GRIDS, 3], &vec![-1; (DECODE_GRIDS * 3) as usize], entry, label)?;
                let mut sources = zeros(Element::i32(), &[SOURCE_ROWS, 1, selected])?;
                let mut rings = words(&[ahead, ring], &[])?;
                let mut routing = words(&[flight, routing_words], &[])?;
                let mut routes = zeros(Element::i32(), &[1, selected])?;
                match lead {
                    Lead::Fused { binding: b, kernel } => {
                        let (h, e) = (b.hidden, b.experts);
                        let norm = rows.norm(device, b.norm, h, entry, label)?;
                        kernel
                            .call(routed_route_streamed::Args {
                                residual: &rows.input(device, 1, h, entry, label)?,
                                norm: &norm,
                                router: &zeros(b.router, &[e, h])?,
                                shared_router: &zeros(Element::f32(), &[h])?,
                                shared_gate: &zeros(b.shared_gate, &[b.shared, h])?,
                                shared_up: &zeros(b.shared_up, &[b.shared, h])?,
                                routes: &mut routes,
                                scores: &mut zeros(Element::f32(), &[1, selected])?,
                                eps: 1.0e-5,
                                normalize: 1,
                                expert_scale: &unit_scale(device, kernel, "ES", entry, label)?,
                                map: &map,
                                tier: &tier,
                                policy: &policy,
                                sited,
                                step,
                                file: &mut file,
                                grids: &mut grids,
                                sources: &mut sources,
                                next_norm: &norm,
                                next_router: &zeros(b.router, &[next, h])?,
                                after_norm: &norm,
                                after_router: &zeros(b.router, &[next, h])?,
                                next_policy: &next_policy,
                                rings: &mut rings,
                                routing: &mut routing,
                            })
                            .map_err(failed)?;
                    }
                    Lead::General { binding: b, kernel } => {
                        let shape = b.shape;
                        let (h, e) = (shape.hidden, shape.experts);
                        let router_norm = zeros(b.router_norm, &[h])?;
                        kernel
                            .call(routed_select_streamed::Args {
                                residual: &semantic_ones(device, Element::f32(), &[1, h], entry, label)?,
                                norm: &zeros(b.norm, &[h])?,
                                router_norm: &router_norm,
                                router: &zeros(b.router, &[e, h])?,
                                bias: &zeros(Element::f32(), &[e])?,
                                expert_scale: &unit_scale(device, kernel, "ES", entry, label)?,
                                routes: &mut routes,
                                weights: &mut zeros(Element::f32(), &[1, selected])?,
                                epsilon: 1.0e-5,
                                score: shape.score,
                                normalization: shape.normalization,
                                normalization_epsilon: f32::from_bits(shape.normalization_epsilon),
                                scale: f32::from_bits(shape.scale),
                                map: &map,
                                tier: &tier,
                                policy: &policy,
                                sited,
                                step,
                                file: &mut file,
                                grids: &mut grids,
                                sources: &mut sources,
                                next_router_norm: &router_norm,
                                next_router: &zeros(b.router, &[next, h])?,
                                next_bias: &zeros(Element::f32(), &[next])?,
                                after_router_norm: &router_norm,
                                after_router: &zeros(b.router, &[next, h])?,
                                after_bias: &zeros(Element::f32(), &[next])?,
                                next_policy: &next_policy,
                                rings: &mut rings,
                                routing: &mut routing,
                            })
                            .map_err(failed)?;
                    }
                }
                if whole == 0 {
                    require_words(&routing, &[(record + RING_SEQUENCE, SEQUENCE), (record + RING_COUNT, choices)], entry, label)?;
                    require_words(&routing, &unrouted, entry, label)?;
                    require_words(&sources, &unsourced, entry, label)?;
                    require_words(&grids, &grids_written, entry, label)?;
                    let slot = binding.words.file().lane(SEQUENCE as u64, 0);
                    require_words(&file, &[(slot + SLOT_REQUEST, SEQUENCE), (slot + FILE_COUNT, choices)], entry, label)?;
                }
                if whole == 1 {
                    let routed = routes.read_to_host().map_err(|error| qualification_dynamic(entry, label, error))?;
                    let sourced = routed
                        .chunks_exact(4)
                        .zip(0..)
                        .flat_map(|(route, choice)| {
                            let expert = i32::from_ne_bytes(route.try_into().expect("a word"));
                            [(SOURCE_RECORD * selected + choice, WHOLE_HOME + expert), (SOURCE_SITE * selected + choice, -1)]
                        })
                        .collect::<Vec<_>>();
                    require_words(&sources, &sourced, entry, label)?;
                }
                // Every lead publishes its ring row.
                if next_resolved == 0 {
                    for lead in 0..ahead {
                        require_words(&rings, &[(lead * ring + RING_SEQUENCE, SEQUENCE)], entry, label)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The home of a whole layer a route qualifies at, past no expert's map.
const WHOLE_HOME: i32 = 3;

/// One routed tensor's fixture: one conveyor record as the
/// file stores it (zero bytes), one scratch tier record (slot 0, expert 0),
/// the stacks the grouped entries read an unmapped expert from by expert
/// (`X = E`: every expert's, of which only record 0 is backed and none is
/// read, expert 0 being mapped) and the bank's records, with
/// `converted`, the host's conversion of a zero stored record into its
/// resident form (`Element::repack_host`, the entries' binding), and
/// `marker`, its complement, which every resident record holds until an
/// entry writes it.
struct RecordFixture {
    converted: Vec<u8>,
    marker: Vec<u8>,
    conveyor: Tensor,
    scratch: Tensor,
    stacks: Tensor,
    records: Tensor,
}

impl RecordFixture {
    fn new(device: &Device, tensor: RecordTensor, experts: u64, label: &str) -> Result<Self, CatalogFailure> {
        let entry = "routed_bank";
        let failed = |error: &dyn fmt::Display| qualification_dynamic(entry, label, error);
        let [n, k] = tensor.matrix;
        let record = [1, n, k];
        let stored = tensor.stored_bytes().map_err(|error| failed(&error))?;
        let stored = usize::try_from(stored).map_err(|error| failed(&error))?;
        let converted = tensor
            .resident
            .repack_host(tensor.source, &record, &vec![0; stored])
            .ok_or_else(|| failed(&"the stored record has no conversion into its resident form"))?;
        let marker = converted.iter().map(|byte| !byte).collect::<Vec<_>>();
        let shared = |extents: &[u64]| Tensor::shared(device, tensor.source, extents).map_err(|error| failed(&error));
        let mut scratch = Tensor::zeros(device, tensor.resident, &record).map_err(|error| failed(&error))?;
        scratch.write_from_host(&marker).map_err(|error| failed(&error))?;
        let reserve = |kind, extents: &[u64], committed| {
            BankAllocation::reserve(device, kind, tensor.resident, extents, committed, None).map_err(|error| failed(&error))
        };
        let bank = crate::planning::BankTier::Bank.contract_on(device);
        let (stacks, _) = reserve(bank, &[experts, n, k], 1)?;
        let (mut records, _) = reserve(bank, &[RECORDS, n, k], RECORDS)?;
        records.write_from_host(&marker.repeat(2)).map_err(|error| failed(&error))?;
        Ok(Self {
            conveyor: shared(&record)?,
            converted,
            marker,
            scratch,
            stacks,
            records,
        })
    }
}

/// The gate's tensor of `records` a gate host port binds (`of` picks it):
/// the gate's, or of an up-only record none of its up's (`GT`, `GV` 0).
fn gate_or_empty(
    records: &RecordTensors<RecordFixture>,
    of: impl Fn(&RecordFixture) -> &Tensor,
    label: &str,
) -> Result<Tensor, CatalogFailure> {
    match &records.gate {
        Some(gate) => Ok(of(gate).clone()),
        None => of(&records.up)
            .slice_leading(0, 0)
            .map_err(|error| qualification_dynamic("routed_bank", label, error)),
    }
}

/// The words of a map of `experts` experts (`[E, MAP_WORDS]`) that holds
/// none (each record -1).
fn unmapped(experts: u64) -> Vec<(u64, i32)> {
    (0..experts).map(|expert| (expert * MAP_WORDS + MAP_RECORD, -1)).collect()
}

/// The words of a map row that holds `record` for expert `expert` from
/// step 0 on (MAP_UNTIL all ones).
fn held(expert: u64, record: i32) -> [(u64, i32); 5] {
    let row = expert * MAP_WORDS;
    [
        (row + MAP_FROM, 0),
        (row + MAP_FROM + 1, 0),
        (row + MAP_UNTIL, -1),
        (row + MAP_UNTIL + 1, -1),
        (row + MAP_RECORD, record),
    ]
}

/// `value` as an i32 word.
fn word(value: u64, entry: &'static str, label: &str) -> Result<i32, CatalogFailure> {
    i32::try_from(value).map_err(|error| qualification_dynamic(entry, label, error))
}

/// A host-shared i32 tensor of `extents` (`Tensor::shared`, as the bank
/// holds its words), zero but for each `(index, value)` of `set`.
fn shared_words(
    device: &Device,
    extents: &[u64],
    set: &[(u64, i32)],
    entry: &'static str,
    label: &str,
) -> Result<Tensor, CatalogFailure> {
    let tensor = Tensor::shared(device, Element::i32(), extents)
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    set_words(&tensor, set, entry, label)?;
    Ok(tensor)
}

/// Each `(index, value)` of `set` stored into the host-shared i32 `tensor`,
/// as the host stores a word.
fn set_words(tensor: &Tensor, set: &[(u64, i32)], entry: &'static str, label: &str) -> Result<(), CatalogFailure> {
    for &(index, value) in set {
        tensor
            .store_release(4 * index, u32::from_ne_bytes(value.to_ne_bytes()))
            .map_err(|error| qualification_dynamic(entry, label, error))?;
    }
    Ok(())
}

/// Fails unless each `(index, value)` of `wanted` holds in the i32 `tensor`.
fn require_words(
    tensor: &Tensor,
    wanted: &[(u64, i32)],
    entry: &'static str,
    label: &str,
) -> Result<(), CatalogFailure> {
    let bytes = tensor
        .read_to_host()
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    for &(index, value) in wanted {
        let at = usize::try_from(index * 4).map_err(|error| qualification_dynamic(entry, label, error))?;
        let held = bytes
            .get(at..at + 4)
            .map(|word| i32::from_le_bytes(word.try_into().expect("four-byte word")));
        if held != Some(value) {
            return Err(qualification_dynamic(
                entry,
                label,
                format!("word {index} holds {held:?}, not {value}"),
            ));
        }
    }
    Ok(())
}

/// Fails unless `tensor` holds `wanted`, byte for byte.
fn require_bytes(tensor: &Tensor, wanted: &[u8], entry: &'static str, label: &str) -> Result<(), CatalogFailure> {
    let bytes = tensor
        .read_to_host()
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    if bytes != wanted {
        return Err(qualification_dynamic(entry, label, "the records written differ from the conversion of what was staged"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::routed::fused_graph::DECODE_ROWS;
    use crate::planning::RegionWords;
    use crate::{Expansion, ExecutionPath, GeneralRoutedShape};
    use magnitude_kernels::{
        routed_combine, routed_down_project, routed_down_sited, routed_experts, routed_experts_up, routed_gate_up, routed_group,
        routed_output, routed_route, routed_route_shared, routed_scatter, routed_select, routed_site_request, routed_up,
    };
    use crate::planning::Placer;

    const HIDDEN: u64 = 256;
    const FEATURES: u64 = 128;
    const EXPERTS: u64 = 16;
    const SELECTED: u64 = 4;

    /// `E`'s specialization on `device` at the statics its implementation
    /// declares, each the first of `values` of its name (as preparation
    /// fixes them), its parameters the declared defaults.
    fn specialization<E: seismic::Entry>(device: &Device, values: &[(&str, u64)]) -> seismic::NativeSpecialization {
        let implementation = seismic::generated::native_implementation::<E>(device).unwrap().unwrap();
        let statics = implementation.statics.iter().fold(seismic::NativeSpecialization::new(), |statics, name| {
            let (_, value) = values
                .iter()
                .find(|(candidate, _)| *candidate == name.as_str())
                .unwrap_or_else(|| panic!("{} declares {name}", E::NAME));
            statics.with_static(name.clone(), *value)
        });
        implementation.default_specialization(&statics).unwrap()
    }

    /// How the test bank's down records are stored and held.
    #[derive(Clone, Copy)]
    enum Down {
        /// Q8_0, as its gate and up: a registered conversion.
        Q8,
        /// BF16 stored and resident, as a mixed-quantization file keeps
        /// them: a dense pair, which no registry entry converts.
        Bf16,
    }

    /// A bank of Q8_0 gate and up records of the test geometry, gated or
    /// up-only, and `down` records, as unified memory holds it: no host
    /// tier.
    fn bank(gated: bool, down: Down) -> BankBinding {
        let layout = crate::resident_layout(ExecutionPath::Native, BackendName::Metal);
        let q8 = |matrix| RecordTensor {
            source: Element::named("gguf_q8_0").unwrap(),
            resident: Element::stored("q8g32s", layout).unwrap(),
            matrix,
        };
        let down = match down {
            Down::Q8 => q8([HIDDEN, FEATURES]),
            Down::Bf16 => RecordTensor {
                source: Element::bf16(),
                resident: Element::bf16(),
                matrix: [HIDDEN, FEATURES],
            },
        };
        BankBinding {
            record: RecordTensors {
                gate: gated.then(|| q8([FEATURES, HIDDEN])),
                up: q8([FEATURES, HIDDEN]),
                down,
            },
            records: 4,
            experts: EXPERTS,
            next_experts: EXPERTS,
            selected: SELECTED,
            staging: 2,
            round_records: 2,
            widest: EXPERTS,
            conveyor: EXPERTS,
            moves: 2,
            tier: 0,
            placer: crate::planning::Placer::Tier,
            words: RegionWords {
                choices: DECODE_ROWS * SELECTED,
                rounds: 2,
                flight: 2,
                sites: 2,
            },
        }
    }

    /// The statics of `bank`'s decode route at `ahead` leads, beside
    /// `resident`'s.
    fn route_statics<'v>(bank: BankBinding, ahead: usize, resident: &[(&'v str, u64)]) -> Vec<(&'v str, u64)> {
        [
            ("AHEAD", ahead as u64),
            ("EN", bank.next_experts),
            ("TM", crate::native::tuning::routed::tier_experts(bank)),
        ]
        .into_iter()
        .chain(resident.iter().copied())
        .collect()
    }

    /// `bank`'s entries on `device` beside its `route`, each at the statics
    /// preparation fixes (`Preparation::bank`).
    fn bank_kernels(device: &Device, bank: BankBinding, route: BankRouteKernels) -> ExpertBankKernels {
        let RecordTensors { gate, up, down } = bank.record;
        let gate_source = gate.map_or(up.source, |gate| gate.source);
        let table = bank.words.scratch();
        let scratch = [("FLIGHT", table.slots()), ("LANES", table.lanes())];
        let tier_wait = routed_tier_wait::native_for_device(
            device,
            &specialization::<routed_tier_wait::Entry>(
                device,
                &[("K", bank.selected)],
            ),
        )
        .unwrap();
        let record_statics = |tensor: RecordTensor, list: u64| {
            let [n, k] = tensor.matrix;
            [
                ("R", bank.staging),
                ("V", bank.conveyor),
                ("S", bank.experts),
                ("D", bank.records),
                ("L", list),
                ("N", n),
                ("K", k),
            ]
        };
        ExpertBankKernels {
            route,
            tier_wait,
            step: routed_step::native_for_device(device, &specialization::<routed_step::Entry>(device, &[])).unwrap(),
            scratch_request: routed_scratch_request::native_for_device(
                device,
                &specialization::<routed_scratch_request::Entry>(
                    device,
                    &[&[("K", bank.selected), ("E", bank.widest)], scratch.as_slice()].concat(),
                ),
            )
            .unwrap(),
            scratch_wait: routed_scratch_wait::native_for_device_with(
                device,
                routed_scratch_wait::Elements { SG: gate_source, SU: up.source, SD: down.source },
                &specialization::<routed_scratch_wait::Entry>(device, &scratch),
            )
            .unwrap(),
            scratch_grids: routed_scratch_grids::native_for_device(
                device,
                &specialization::<routed_scratch_grids::Entry>(device, &[scratch.as_slice(), &[("R", bank.words.rounds)]].concat()),
            )
            .unwrap(),
            list_stage: routed_list_stage::native_for_device(device, &specialization::<routed_list_stage::Entry>(device, &[])).unwrap(),
            pass_map: routed_pass_map::native_for_device(device, &specialization::<routed_pass_map::Entry>(device, &[])).unwrap(),
            convert: bank.record.map(|_, tensor| {
                routed_convert::native_for_device_with(
                    device,
                    routed_convert::Elements { ST: tensor.source, U: tensor.resident },
                    &specialization::<routed_convert::Entry>(device, &record_statics(tensor, bank.round_records)),
                )
                .unwrap()
            }),
            placement: (bank.placer == crate::planning::Placer::Tier).then(|| DevicePlacement {
                seed: bank.record.map(|_, tensor| {
                    routed_copy::native_for_device_with(
                        device,
                        routed_copy::Elements { U: tensor.resident },
                        &specialization::<routed_copy::Entry>(device, &record_statics(tensor, bank.widest)),
                    )
                    .unwrap()
                }),
                seed_commit: routed_seed_commit::native_for_device(
                    device,
                    &specialization::<routed_seed_commit::Entry>(
                        device,
                        &[&[("X", bank.widest), ("BANK", bank.records)], scratch.as_slice()].concat(),
                    ),
                )
                .unwrap(),
                list_grid: routed_list_grid::native_for_device(device, &specialization::<routed_list_grid::Entry>(device, &[]))
                    .unwrap(),
                moves: BankMoves {
                    records: bank.record.map(|_, tensor| {
                        routed_move::native_for_device_with(
                            device,
                            routed_move::Elements { U: tensor.resident },
                            &specialization::<routed_move::Entry>(device, &record_statics(tensor, bank.moves)),
                        )
                        .unwrap()
                    }),
                },
            }),
            site: None,
        }
    }

    fn metal() -> Device {
        seismic::Devices::discover(&[])
            .unwrap()
            .open_backend(BackendName::Metal)
            .unwrap()
    }

    /// A fused routed layer's gated bank qualifies on Metal: every bank
    /// entry runs once and publishes or writes what it must, and the shared
    /// entries read expert 0 where the bank's slot map places it.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "the bank fixture here is Metal's")]
    fn a_fused_layers_bank_qualifies() {
        fused_bank_qualifies(bank(true, Down::Q8));
    }

    /// A bank the host places into (unified memory's one store) qualifies
    /// with no device placement: its tier wait sources the bank record its
    /// answer names, and the host's landing and map write place expert 0.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "the bank fixture here is Metal's")]
    fn a_host_placed_bank_qualifies() {
        fused_bank_qualifies(BankBinding { placer: crate::planning::Placer::Bank, ..bank(true, Down::Q8) });
    }

    /// A bank whose down records are a dense pair, which no registry entry
    /// converts, qualifies as every launched entry converts it: by
    /// elements, the dense records' conversion its convert and fill run.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "the bank fixture here is Metal's")]
    fn a_bank_of_dense_records_qualifies() {
        fused_bank_qualifies(bank(true, Down::Bf16));
    }

    /// The fused routed layer of `bank`'s geometry qualifies on Metal.
    fn fused_bank_qualifies(bank: BankBinding) {
        let device = metal();
        let (bf16, q8, down) = (Element::bf16(), bank.record.up.resident, bank.record.down.resident);
        let binding = RoutedBinding {
            hidden: HIDDEN,
            experts: EXPERTS,
            selected: SELECTED,
            features: FEATURES,
            shared: FEATURES,
            normalize_selected: true,
            norm: bf16,
            router: bf16,
            expert_gate: q8,
            expert_up: q8,
            expert_down: down,
            shared_gate: bf16,
            shared_up: bf16,
            shared_down: bf16,
            activation: bf16,
            scales: Default::default(),
        };
        let statics = [
            ("H", HIDDEN),
            ("PT", 1),
            ("NM", HIDDEN),
            ("E", EXPERTS),
            ("K", SELECTED),
            ("S", FEATURES),
            ("ES", 0),
            ("F", FEATURES),
            ("GS", 0),
            ("US", 0),
            ("SN", 1),
            ("IN", 0),
        ];
        let resident = RoutedKernels {
            route: routed_route::native_for_device_with(
                &device,
                routed_route::Elements { NW: bf16, RW: bf16, A: bf16 },
                &specialization::<routed_route::Entry>(&device, &statics),
            )
            .unwrap(),
            decode: RoutedDecodeKernels::SharedRoute {
                route: routed_route_shared::native_for_device_with(
                    &device,
                    routed_route_shared::Elements { NW: bf16, RW: bf16, SGW: bf16, SUW: bf16, A: bf16 },
                    &specialization::<routed_route_shared::Entry>(&device, &statics),
                )
                .unwrap(),
                choices: routed_gate_up::native_for_device_with(
                    &device,
                    routed_gate_up::Elements { EGW: q8, EUW: q8, A: bf16 },
                    &specialization::<routed_gate_up::Entry>(&device, &statics),
                )
                .unwrap(),
            },
            output: routed_output::native_for_device_with(
                &device,
                routed_output::Elements { EDW: down, SDW: bf16, A: bf16 },
                &specialization::<routed_output::Entry>(&device, &statics),
            )
            .unwrap(),
            group: routed_group::native_for_device(&device, &specialization::<routed_group::Entry>(&device, &statics))
                .unwrap(),
            experts: routed_experts::native_for_device_with(
                &device,
                routed_experts::Elements { A: bf16, EGW: q8, EUW: q8, EDW: down },
                &specialization::<routed_experts::Entry>(&device, &statics),
            )
            .unwrap(),
            combine: routed_combine::native_for_device_with(
                &device,
                routed_combine::Elements { A: bf16, SGW: bf16, SUW: bf16, SDW: bf16 },
                &specialization::<routed_combine::Entry>(&device, &statics),
            )
            .unwrap(),
        };
        let routes = std::array::from_fn(|ahead| {
            routed_route_streamed::native_for_device_with(
                &device,
                routed_route_streamed::Elements {
                    NW: bf16,
                    RW: bf16,
                    SGW: bf16,
                    SUW: bf16,
                    A: bf16,
                },
                &specialization::<routed_route_streamed::Entry>(&device, &route_statics(bank, ahead, &statics)),
            )
            .map(Some)
            .unwrap()
        });
        let kernels = bank_kernels(&device, bank, BankRouteKernels::Fused(routes));
        qualify_bank(
            &device,
            BankedFeedForward::Routed(binding),
            bank,
            &kernels,
            &AttestedFeedForward::Routed(resident),
            SublayerRows::SINGLE,
            &ResolvedNumericalProfile::standard().requirement_of(None),
            None,
        )
        .unwrap();
    }

    /// A general routed layer's up-only bank qualifies on Metal, as the
    /// fused layer's gated bank does.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "the bank fixture here is Metal's")]
    fn a_general_layers_up_only_bank_qualifies() {
        general_up_only_bank(&metal(), None);
    }

    /// A sited general layer's CPU site is admitted against the reference:
    /// its q8 arithmetic over representative Q8_0 records and rows is within
    /// the standard α of f32 math over the stored weights, and not within a
    /// hundredth of a rounding step.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "the bank fixture here is Metal's")]
    fn a_sited_layers_cpu_site_is_admitted_against_the_reference() {
        // Its admission sizes a pool from the CPUs the process may run on.
        let _placement = crate::test_support::process_placement();
        let standard = ResolvedNumericalProfile::standard().requirement_of(Some(DType::BF16));
        assert!(general_up_only_bank(&metal(), Some(&standard)));
        let exact = seismic::NumericalRequirement::RoundingSteps { alpha: 0.01, activation: Some(DType::BF16) };
        assert!(!general_up_only_bank(&metal(), Some(&exact)));
    }

    /// The general up-only bank of the test geometry qualified on `device`,
    /// sited where a requirement admits its site, over a tier of six
    /// records: whether its site was admitted.
    fn general_up_only_bank(device: &Device, requirement: Option<&seismic::NumericalRequirement>) -> bool {
        let sited = requirement.is_some();
        let device = device.clone();
        let tier = if sited { 6 } else { 0 };
        let bank = BankBinding { tier, placer: Placer::Tier, ..bank(false, Down::Q8) };
        let (bf16, f32, q8) = (Element::bf16(), Element::f32(), bank.record.up.resident);
        let binding = crate::GeneralRoutedBinding {
            shape: GeneralRoutedShape {
                hidden: HIDDEN,
                experts: EXPERTS,
                selected: SELECTED,
                intermediate: FEATURES,
                expert_hidden: HIDDEN,
                latent: false,
                // ReLU², up-only.
                experts_expansion: Expansion { gated: false, activation: 2 },
                shared: None,
                score: 1,
                normalization: 1,
                normalization_epsilon: 0,
                scale: 1.0_f32.to_bits(),
                bias: false,
                expert_scale: false,
                router_norm: false,
            },
            norm: bf16,
            router_norm: bf16,
            router: bf16,
            expert_gate: None,
            expert_up: q8,
            expert_down: q8,
            latent: None,
            shared: None,
            activation: bf16,
            scales: Default::default(),
        };
        let statics = [
            ("H", HIDDEN),
            ("E", EXPERTS),
            ("K", SELECTED),
            ("ES", 0),
            ("F", FEATURES),
            ("US", 0),
        ];
        let resident = GeneralRoutedKernels {
            select: routed_select::native_for_device_with(
                &device,
                routed_select::Elements { NW: bf16, RNW: bf16, RW: bf16, A: bf16 },
                &specialization::<routed_select::Entry>(&device, &statics),
            )
            .unwrap(),
            experts: ExpertKernels::Plain {
                decode: routed_up::native_for_device_with(
                    &device,
                    routed_up::Elements { EUW: q8, A: bf16 },
                    &specialization::<routed_up::Entry>(&device, &statics),
                )
                .unwrap(),
                grouped: routed_experts_up::native_for_device_with(
                    &device,
                    routed_experts_up::Elements { A: bf16, EUW: q8, EDW: q8 },
                    &specialization::<routed_experts_up::Entry>(&device, &statics),
                )
                .unwrap(),
            },
            down: routed_down::native_for_device_with(
                &device,
                routed_down::Elements { EDW: q8, A: bf16, R: f32 },
                &specialization::<routed_down::Entry>(&device, &statics),
            )
            .unwrap(),
            group: routed_group::native_for_device(&device, &specialization::<routed_group::Entry>(&device, &statics))
                .unwrap(),
            scatter: routed_scatter::native_for_device_with(
                &device,
                routed_scatter::Elements { A: bf16, R: f32 },
                &specialization::<routed_scatter::Entry>(&device, &statics),
            )
            .unwrap(),
            shared: None,
            latent: None,
        };
        let selects = std::array::from_fn(|ahead| {
            routed_select_streamed::native_for_device_with(
                &device,
                routed_select_streamed::Elements {
                    NW: bf16,
                    RNW: bf16,
                    RW: bf16,
                    A: bf16,
                },
                &specialization::<routed_select_streamed::Entry>(&device, &route_statics(bank, ahead, &statics)),
            )
            .map(Some)
            .unwrap()
        });
        let mut kernels = bank_kernels(&device, bank, BankRouteKernels::General(selects));
        if sited {
            let form = SiteForm { activation: bf16, function: binding.shape.experts_expansion.activation };
            kernels.site = Some(BankSite {
                request: routed_site_request::native_for_device_with(
                    &device,
                    routed_site_request::Elements { A: bf16 },
                    &specialization::<routed_site_request::Entry>(&device, &statics),
                )
                .unwrap(),
                combine: BankCombine::General {
                    project: routed_down_project::native_for_device_with(
                        &device,
                        routed_down_project::Elements { EDW: q8, A: bf16 },
                        &specialization::<routed_down_project::Entry>(&device, &statics),
                    )
                    .unwrap(),
                    sited: routed_down_sited::native_for_device_with(
                        &device,
                        routed_down_sited::Elements { A: bf16, R: f32 },
                        &specialization::<routed_down_sited::Entry>(&device, &statics),
                    )
                    .unwrap(),
                },
                form,
                admitted: None,
            });
        }
        let site = kernels.site.as_ref().map(|site| site.form);
        let verdict = qualify_bank(
            &device,
            BankedFeedForward::GeneralRouted(binding),
            bank,
            &kernels,
            &AttestedFeedForward::GeneralRouted(resident),
            SublayerRows::SINGLE,
            requirement.unwrap_or(&ResolvedNumericalProfile::standard().requirement_of(None)),
            None,
        )
        .unwrap();
        // A load runs no site: its verdict waits on the job, which runs it.
        assert!(matches!(verdict, None | Some(crate::assessment::Priced::Pending(_))));
        site.zip(requirement).is_some_and(|(form, requirement)| {
            let request = crate::assessment::SiteRequest::of(&bank, form, requirement.clone());
            super::super::site::admit_site(&device, &request).unwrap().is_ok()
        })
    }
}
