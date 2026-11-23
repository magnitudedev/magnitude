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
use crate::experts::{PooledRecords, pooled_records};
use crate::planning::{
    BankBinding, BankedFeedForward, HINT_LEADS, RecordTensor, RecordTensors,
};
use magnitude_kernels::constants::{
    FILE_CHOICE, FILE_COUNT, FILE_RECORD, LANDING_DONE, LANDING_FROM, LANDING_HEAD, LANDING_TO, LIST_COUNT,
    LIST_FROM, LIST_SEQUENCE, LIST_TO, MAP_FROM, MAP_RECORD, MAP_UNTIL, MAP_WORDS, POLICY_HOME, POLICY_PRICED,
    POLICY_RESOLVED, POLICY_WHOLE, POLICY_WORDS, RING_COUNT, RING_SEQUENCE, ROUND_LANE, SCRATCH_ASK, SCRATCH_STAGED,
    SLOT_ANSWER, SLOT_REQUEST, SOURCE_RECORD, SOURCE_ROWS, SOURCE_SITE, STEP_HIGH, STEP_LOW, STEP_GUESSES, STEP_WORDS, GUESS_CHAINED, GUESS_HEAD,
};
use magnitude_kernels::{
    constants, routed_copy, routed_gate_up, routed_gate_up_part,
    routed_hint_route, routed_hint_select, routed_land, routed_list_grid, routed_pass_map, routed_round,
    routed_route_streamed, routed_scratch_grids, routed_scratch_request, routed_seed_commit,
    routed_select_streamed, routed_step, step_words,
};

/// The fixture step's sequence.
const SEQUENCE: i32 = 1;
/// Rows of a step that are front rows when qualification passes a front: row 1.
const FRONT_ROWS: u32 = 0b10;

/// The fixture bank's records: one no entry writes (0), and the one a prompt
/// group's seed lands in (1).
const RECORDS: u64 = 2;

/// Qualify `binding`'s bank entries `kernels` of the streamed feed-forward
/// `banked`, whose resident entries are `feed_forward` and whose rows are
/// `rows`: one decode row's route (each lead, each whole word), the landing
/// launch of its absent expert 0 (converted into bank record 0 where the
/// bank places, landed in tier record 0 where the tier does), a
/// scratch-served step's request, its round 0's landing launch and grids,
/// the
/// seed of expert 0 into record 1 and its commit, which the host then maps;
/// then the shared routed entries over the
/// map the seed left (expert 0 held in record 1 for the step), the decode
/// entries' sources naming it.
///
/// Where the layer is sited (`ExpertBankKernels::site`), its site's verdict
/// under `requirement` as the measurement job stored it in `cache`
/// (`site::site_verdict`), which runs nothing, or its refusal where the
/// profile approves no CPU site (no `requirement`), which reads no verdict;
/// else none.
#[allow(clippy::too_many_arguments)]
pub(super) fn qualify_bank(
    charges: &seismic::Charges,
    banked: BankedFeedForward,
    binding: BankBinding,
    kernels: &ExpertBankKernels,
    feed_forward: &AttestedFeedForward,
    rows: SublayerRows,
    requirement: Option<&seismic::NumericalRequirement>,
    cache: Option<&crate::kernel_cache::KernelCache>,
) -> Result<Option<crate::assessment::Priced<()>>, CatalogFailure> {
    let label = format!("{banked:?} {binding:?}");
    let label = label.as_str();
    let ExpertBankKernels {
        route,
        land,
        step: step_kernel,
        scratch_request,
        round,
        scratch_grids,
        pass_map,
        placement,
        site,
        gate_up,
    } = kernels;
    let served = Served::of(banked, route, feed_forward, label)?;
    let mut records = binding
        .record
        .try_map(|_, tensor| RecordFixture::new(charges, tensor, binding.experts, label))?;
    let selected = word(binding.selected, "routed_bank", label)?;
    let words = |extents: &[u64], set: &[(u64, i32)], entry| shared_words(charges, extents, set, entry, label);

    // The step's sequence, of a graph without token rows.
    let none = words(&[0, 2], &[], "routed_step")?;
    let mut step = words(&[STEP_WORDS], &[], "routed_step")?;
    step_kernel
        .call(charges, routed_step::Args {
            step: &mut step,
            sequence: &words(&[STEP_WORDS], &sparse(&step_words(SEQUENCE as u64, 0)), "routed_step")?,
            guesses: &none,
        })
        .map_err(|error| qualification_dynamic("routed_step", label, error))?;
    require_words(&step, &sparse(&step_words(SEQUENCE as u64, 0)), "routed_step", label)?;
    // A step's front rows pass through with its sequence, each front row's
    // field of STEP_GUESSES the kind of the guess it runs on, its row
    // before's (`guesses[row - 1]`, (token, kind)); a row that is no front
    // row's is clear.
    let front = step_words(SEQUENCE as u64, FRONT_ROWS);
    for kind in [GUESS_CHAINED, GUESS_HEAD].map(|kind| kind as i32) {
        let mut fronted = words(&[STEP_WORDS], &[], "routed_step")?;
        step_kernel
            .call(charges, routed_step::Args {
                step: &mut fronted,
                sequence: &words(&[STEP_WORDS], &sparse(&front), "routed_step")?,
                guesses: &words(&[2, 2], &[(0, 7), (1, kind), (2, 9), (3, GUESS_HEAD as i32)], "routed_step")?,
            })
            .map_err(|error| qualification_dynamic("routed_step", label, error))?;
        let mut expected = front;
        expected[STEP_GUESSES as usize] = magnitude_kernels::step_guesses([(FRONT_ROWS.trailing_zeros() as usize, kind)]);
        require_words(&fronted, &sparse(&expected), "routed_step", label)?;
    }

    served.qualify_routes(charges, binding, rows, &step, label)?;

    // The decode landing launch of one absent choice, expert 0 (one row's
    // first choice of its `selected`, its sources naming no record), which
    // the host read and answered before the call, so the launch finds its
    // answer at once (`call_answered`: it runs standalone on every backend,
    // where kernels wait in place and where a park is a submission boundary
    // alike). Where the bank places, the host appended its landing, staging
    // record 0 into bank record 0, which the launch converts and acknowledges
    // (DONE); where the host tier places, it landed into tier record 0 and
    // the launch converts nothing.
    let entry = "routed_land";
    let map = words(&[binding.experts, MAP_WORDS], &unmapped(binding.experts), entry)?;
    // The request, in the step's slot of the file table: choice 0 expert 0,
    // absent; every other choice held (as the route requests a held one,
    // -1 - expert).
    let table = binding.words.file();
    let slot = table.lane(SEQUENCE as u64, 0);
    let others = (1..binding.selected).map(|choice| (slot + constants::FILE_ENTRY(choice) + FILE_CHOICE, -1));
    // Its answer names record 0 by its record word: tier record 0 (-2)
    // across a link, bank record 0 on unified memory.
    let (landed, appended) = match binding.placer {
        crate::planning::Placer::Tier => (-2, 0),
        crate::planning::Placer::Bank => (0, 1),
    };
    // It launches on the grid the route wrote of a step that requested.
    let groups = binding.placer.land_groups();
    let request = [
        (slot + SLOT_REQUEST, SEQUENCE),
        (slot + SLOT_ANSWER, SEQUENCE),
        (slot + FILE_COUNT, selected),
        (slot + constants::FILE_ENTRY(0) + FILE_CHOICE, 0),
        (slot + constants::FILE_ENTRY(0) + FILE_RECORD, landed),
    ]
    .into_iter()
    .chain(others)
    .collect::<Vec<_>>();
    let mut file = words(&table.extents(), &request, entry)?;
    let count = binding.selected;
    // Its sources: none for the GPU, none the CPU's, so the landed choice
    // is the GPU's, served in place.
    let mut sources =
        semantic_i32(charges, &[SOURCE_ROWS, 1, count], &vec![-1; (SOURCE_ROWS * count) as usize], entry, label)?;
    let policy = words(&[POLICY_WORDS], &[], entry)?;
    let mut landing = words(&[constants::LANDING_ENTRY(binding.selected)], &landing_list(appended, 0), entry)?;
    // Where the tier places, staging and records views of none.
    let none = |tensor: &Tensor| tensor.slice_leading(0, 0).map_err(|error| qualification_dynamic("routed_bank", label, error));
    let views = |tensor: &Tensor| match binding.placer {
        crate::planning::Placer::Tier => none(tensor),
        crate::planning::Placer::Bank => Ok(tensor.clone()),
    };
    let staging = records.as_ref().try_map(|_, fixture| views(&fixture.conveyor))?;
    let mut bank = records.as_ref().try_map(|_, fixture| views(&fixture.records.records))?;
    let staging_gate = staging.gate.clone().map_or_else(|| none(&staging.up), Ok)?;
    let mut gate = bank.gate.take().map_or_else(|| none(&bank.up), Ok)?;
    let hidden = binding.record.up.matrix[1];
    let zero_tensor = |element: Element, extents: &[u64]| {
        Tensor::zeros(charges, element, extents).map_err(|error| qualification_dynamic(entry, label, error))
    };
    let mut empty_site = zero_tensor(Element::i32(), &[0, 0])?;
    let mut empty_site_rows = zero_tensor(Element::f32(), &[0, constants::VERIFY_ROWS, hidden])?;
    land.call_answered(charges, routed_land::Args {
        step: &step,
        file: &mut file,
        sources: &mut sources,
        policy: &policy,
        landing: &mut landing,
        staging_gate: &staging_gate,
        staging_up: &staging.up,
        staging_down: &staging.down,
        gate: &mut gate,
        up: &mut bank.up,
        down: &mut bank.down,
        grid: &words(&[3], &[(0, groups), (1, 1), (2, 1)], entry)?,
        routes: &zero_tensor(Element::i32(), &[0, count])?,
        weights: &zero_tensor(Element::f32(), &[0, count])?,
        normalized: &zero_tensor(Element::bf16(), &[0, hidden])?,
        site: &mut empty_site,
        site_rows: &mut empty_site_rows,
    })
    .map_err(|error| qualification_dynamic(entry, label, error))?;
    require_words(&sources, &[(SOURCE_RECORD * count, landed), (SOURCE_SITE * count, -1)], entry, label)?;
    if appended > 0 {
        require_words(&landing, &[(constants::LANDING_ENTRY(0) + LANDING_DONE, SEQUENCE)], entry, label)?;
        for fixture in records.as_ref() {
            fixture.require(&[&fixture.converted, &fixture.marker], entry, label)?;
        }
    }

    // A scratch-served step's request, each choice its own expert, and its
    // round 0: the host answered the request with the count of records its
    // rounds convert (one), appended round 0's landing (conveyor record 0
    // into scratch slot 0), then answered every round.
    let table = binding.words.scratch();
    let ask = table.lane(SEQUENCE as u64, SCRATCH_ASK);
    let first = table.lane(SEQUENCE as u64, ROUND_LANE(0));
    let answered = (0..table.lanes()).map(|lane| (table.lane(SEQUENCE as u64, lane) + SLOT_ANSWER, SEQUENCE));
    let mut scratch_table = words(
        &table.extents(),
        &[(ask + SCRATCH_STAGED, 1)].into_iter().chain(answered).collect::<Vec<_>>(),
        "routed_scratch_request",
    )?;
    let mut flags = words(&[binding.widest], &[], "routed_scratch_request")?;
    scratch_request
        .call(charges, routed_scratch_request::Args {
            routes: &semantic_i32(charges, &[1, binding.selected], &(0..selected).collect::<Vec<_>>(), "routed_scratch_request", label)?,
            step: &step,
            flags: &mut flags,
            scratch: &mut scratch_table,
            resident: &words(&[1], &[], "routed_scratch_request")?,
        })
        .map_err(|error| qualification_dynamic("routed_scratch_request", label, error))?;
    require_words(&scratch_table, &[(ask + SLOT_REQUEST, SEQUENCE), (ask + SLOT_REQUEST + 1, 0)], "routed_scratch_request", label)?;
    let requested = (0..binding.selected).map(|expert| (expert, 1)).collect::<Vec<_>>();
    require_words(&flags, &requested, "routed_scratch_request", label)?;
    let list = |from, to| [(LIST_SEQUENCE, SEQUENCE), (LIST_COUNT, 1), (constants::LIST_ENTRY(0) + LIST_FROM, from), (constants::LIST_ENTRY(0) + LIST_TO, to)];
    // The seeds: scratch slot 0, which holds expert 0, into record 1.
    let seeds = words(&[constants::LIST_ENTRY(binding.window_slots)], &list(0, 1), "routed_copy")?;
    let entry = "routed_round";
    let window = semantic_i32(charges, &[1], &[0], entry, label)?;
    let window_rounds = word(binding.window_rounds, entry, label)?;
    let mut round_list = words(&[constants::LANDING_ENTRY(binding.round_records)], &landing_list(1, 0), entry)?;
    let gate_conveyor = gate_or_empty(&records, |fixture| &fixture.conveyor, label)?;
    let mut scratch = records.as_ref().map(|_, fixture| fixture.scratch.clone());
    let mut gate = scratch.gate.take().map_or_else(|| none(&scratch.up), Ok)?;
    round
        .call_answered(charges, routed_round::Args {
            step: &step,
            scratch: &mut scratch_table,
            window: &window,
            rounds: window_rounds,
            round: 0,
            landing: &mut round_list,
            tiered: &words(&[constants::LIST_ENTRY(binding.window_slots)], &[], entry)?,
            conveyor_gate: &gate_conveyor,
            conveyor_up: &records.up.conveyor,
            conveyor_down: &records.down.conveyor,
            gate: &mut gate,
            up: &mut scratch.up,
            down: &mut scratch.down,
            seeds: &seeds,
            map: &words(&[binding.experts, MAP_WORDS], &[], entry)?,
            // Round 0 always launches.
            grid: &semantic_i32(charges, &[3], &[magnitude_kernels::constants::LAND_GROUPS as i32, 1, 1], entry, label)?,
        })
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    require_words(&scratch_table, &[(first + SLOT_REQUEST, SEQUENCE)], entry, label)?;
    require_words(&round_list, &[(constants::LANDING_ENTRY(0) + LANDING_DONE, SEQUENCE)], entry, label)?;
    for fixture in records.as_ref() {
        require_bytes(&fixture.scratch, &fixture.converted, entry, label)?;
    }
    let entry = "routed_scratch_grids";
    let rounds = binding.window_rounds;
    let mut grids = semantic_i32(charges, &[rounds, 3], &vec![0; (rounds * 3) as usize], entry, label)?;
    scratch_grids
        .call(charges, routed_scratch_grids::Args {
            step: &step,
            scratch: &scratch_table,
            window: &window,
            grids: &mut grids,
            round_records: word(binding.round_records, entry, label)?,
        })
        .map_err(|error| qualification_dynamic(entry, label, error))?;

    // Expert 0 into record 1: the prompt group's seed and its commit,
    // whoever places, and the host maps it, held from the step on.
    {
        let crate::native::DevicePlacement { seed, seed_commit, list_grid } = placement;
        {
            // A list's launch grid for its one entry: one row of a copy
            // within device memory's blocks, which the host writes and the
            // kernel keeps; and the list staged beside it.
            let grid_of = |listed: &Tensor, entry| {
                let blocks = magnitude_kernels::LIST_BLOCKS as i32;
                let mut grid = semantic_i32(charges, &[3], &[blocks, 0, 0], entry, label)?;
                let mut staged = words(&[constants::LIST_ENTRY(binding.window_slots)], &[], entry)?;
                list_grid
                    .call(charges, routed_list_grid::Args { list: listed, step: &step, grid: &mut grid, staged: &mut staged })
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
                fixture.mark().map_err(|error| qualification_dynamic(entry, label, error))?;
                kernel
                    .call(charges, routed_copy::Args {
                        source: &fixture.scratch,
                        list: &staged,
                        step: &step,
                        destination: &mut fixture.records.records,
                        grid: &seed_grid,
                    })
                    .map_err(|error| qualification_dynamic(entry, label, error))?;
                fixture.require(&[&fixture.marker, &fixture.converted], entry, label)?;
            }
            // Its commit of record 1.
            let entry = "routed_seed_commit";
            let mut commits = words(&[RECORDS], &[], entry)?;
            seed_commit
                .call(charges, routed_seed_commit::Args {
                    list: &seeds,
                    step: &step,
                    scratch: &scratch_table,
                    window: &window,
                    rounds: window_rounds,
                    commits: &mut commits,
                })
                .map_err(|error| qualification_dynamic(entry, label, error))?;
            require_words(&commits, &[(0, 0), (1, SEQUENCE)], entry, label)?;
        }
    }
    set_words(&map, &held(0, 1), "host placement", label)?;

    // A merge set's pass's map resolved into device memory: expert 0 held
    // in record 1 from the step on, expert 1 in scratch slot 0 when more
    // than two experts exist, every other expert in none, and the step.
    let entry = "routed_pass_map";
    let window_codes = (0..binding.experts)
        .map(|expert| match expert {
            0 | 2 => -2,
            1 if binding.experts > 2 => 0,
            _ => -1,
        })
        .collect::<Vec<_>>();
    let pass_window = semantic_i32(charges, &[binding.experts], &window_codes, entry, label)?;
    let mut resolved = words(&[binding.experts, MAP_WORDS], &[], entry)?;
    let mut pass_step = words(&[STEP_WORDS], &[], entry)?;
    pass_map
        .call(charges, routed_pass_map::Args { map: &map, window: &pass_window, step: &step, resolved: &mut resolved, pass_step: &mut pass_step })
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
    let wanted = (0..binding.experts)
        .flat_map(|expert| {
            let record = match expert {
                0 => 1,
                1 if binding.experts > 2 => -2,
                _ => -1,
            };
            from(expert, record)
        })
        .collect::<Vec<_>>();
    require_words(&resolved, &wanted, entry, label)?;

    // The shared entries read the bank's record 1; beside a host tier they
    // take its records as their entries are prepared for (`TR`), pooled as
    // the tier's, none of which a source names.
    let tier = (binding.tier > 0)
        .then(|| {
            // A tier is pooled `Shared`, on a link or in unified memory.
            let kind = crate::planning::BankTier::Host.contract_on(&charges.device());
            binding.record.try_map(|_, tensor| {
                pooled_records(charges, kind, tensor.resident, &tensor.matrix, binding.tier, 1)
                    .map_err(|error| qualification_dynamic("routed_bank", label, error))
            })
        })
        .transpose()?;
    let reads = ExpertReads {
        map,
        step: step.clone(),
        decode_record: Some(1),
        records: records.as_ref().map(|_, fixture| fixture.records.records.clone()),
        stacks: records.as_ref().map(|_, fixture| fixture.stacks.records.clone()),
        tier: tier.as_ref().map(|tier| tier.as_ref().map(|_, tier| tier.records.clone())),
    };
    // Each routed form's resident gate/up, which a gated layer's passes
    // follow: a streamed fused layer's decode is the shared-route form
    // (planning admits no other beside a bank).
    let resident = match &served {
        Served::Fused { binding, kernels, .. } => {
            qualify_routed(charges, *binding, kernels, rows, &reads)?;
            match &kernels.decode {
                crate::native::RoutedDecodeKernels::SharedRoute { choices, .. } => Some(ResidentKernel {
                    kernel: choices.first(),
                    hidden: binding.hidden,
                    features: binding.features,
                    element: binding.activation,
                    activation: crate::operators::routed::FUSED_ACTIVATION,
                    limit: None,
                }),
                crate::native::RoutedDecodeKernels::Expand(_) => {
                    panic!("a streamed fused layer's decode is the shared-route form")
                }
            }
        }
        Served::General { binding, kernels, .. } => {
            qualify_general_routed(charges, *binding, kernels, &reads)?;
            match &kernels.experts {
                crate::native::ExpertKernels::Gated { decode, .. } => Some(ResidentKernel {
                    kernel: decode.first(),
                    hidden: binding.shape.expert_hidden,
                    features: binding.shape.intermediate,
                    element: binding.activation,
                    activation: binding.shape.experts_expansion.activation,
                    limit: binding.shape.experts_expansion.limit,
                }),
                crate::native::ExpertKernels::Plain { .. } => None,
            }
        }
    };
    if let Some(part) = gate_up {
        let resident = resident.expect("a bank's gate/up passes are a gated layer's (`bank_gate_up`)");
        qualify_gate_up(charges, binding, part.first(), resident, &reads, &step, label)?;
    }

    Ok(site.as_ref().map(|site| match requirement {
        Some(requirement) => super::site::site_verdict(charges, &binding, site.form, requirement, cache),
        None => crate::assessment::Priced::Unavailable("the numerical profile approves no CPU site".into()),
    }))
}

/// A gated layer's resident gate/up (`routed_gate_up`) at its dimensions,
/// which its streamed passes follow.
struct ResidentKernel<'a> {
    kernel: &'a NativeKernel<routed_gate_up::Entry>,
    hidden: u64,
    features: u64,
    element: Element,
    activation: i32,
    limit: Option<magnitude_family_contracts::GluLimit>,
}

/// A gated layer's gate/up passes (`routed_gate_up_part`) of one decode
/// row whose choices route to distinct experts, over one product: choice
/// 0's expert absent at the route (its source none, and its routing
/// record's of the step), landed by the wait into record 0; choice 1's
/// held in record 1 (where the layer selects two or more); every other
/// choice the CPU's, sourced on the GPU by neither. The early pass, then the
/// late pass once the wait has sourced choice 0, form the resident gate/up's one pass over the
/// sourced choices bit for bit, and the late pass alone writes the
/// absent choice's row and no other.
fn qualify_gate_up(
    charges: &seismic::Charges,
    binding: BankBinding,
    part: &NativeKernel<routed_gate_up_part::Entry>,
    resident: ResidentKernel<'_>,
    reads: &ExpertReads,
    step: &Tensor,
    label: &str,
) -> Result<(), CatalogFailure> {
    let entry = "routed_gate_up_part";
    let failed = |error: &dyn fmt::Display| qualification_dynamic(entry, label, error);
    let ResidentKernel { kernel: resident, hidden, features, element, activation, limit } = resident;
    let selected = binding.selected;
    let choices = usize::try_from(selected).map_err(|error| failed(&error))?;
    // Each choice's record: the route's (choice 0 absent, choice 1 held
    // in record 1, the rest the CPU's), then the wait's (choice 0 landed
    // in record 0).
    let sources = |first: i32| {
        let records = (0..choices).map(|choice| [first, 1, -1][choice.min(2)]).collect::<Vec<_>>();
        semantic_i32(
            charges,
            &[SOURCE_ROWS, 1, selected],
            &magnitude_kernels::source_words(&records, &vec![-1; choices]),
            entry,
            label,
        )
    };
    let (absent, sourced) = (sources(-1)?, sources(0)?);
    // The step's routing record: every choice but the held one absent at
    // its route.
    let (flight, routing_words) = (binding.words.flight, binding.words.routing_words());
    let slot = u64::from(SEQUENCE.unsigned_abs()) % flight * routing_words;
    let unrouted = (0..selected)
        .map(|choice| (slot + binding.words.routing_source(choice), if choice == 1 { 1 } else { -1 }))
        .collect::<Vec<_>>();
    let routing = shared_words(charges, &[flight, routing_words], &unrouted, entry, label)?;
    let normalized = semantic_ones(charges, element, &[1, hidden], entry, label)?;
    // Distinct experts, as a row's choices are.
    let experts = (0..i32::try_from(selected).map_err(|error| failed(&error))?).collect::<Vec<_>>();
    let routes = semantic_i32(charges, &[1, selected], &experts, entry, label)?;
    let gate = |records: &RecordTensors<Tensor>| records.gate.clone().ok_or_else(|| failed(&"a gated layer's records have no gate"));
    let tier = reads.tier_records(label)?;
    let (records_gate, tier_gate) = (gate(&reads.records)?, gate(&tier)?);
    let (gate_scale, up_scale) = (unit_scale(charges, part, "GS", entry, label)?, unit_scale(charges, part, "US", entry, label)?);
    let glu_limit = glu_limit(charges, limit, entry, label)?;
    let mut product = semantic_zeros(charges, element, &[1, selected, features], entry, label)?;
    let pass = |sources: &Tensor, late: i32, product: &mut Tensor| {
        part.call(charges, routed_gate_up_part::Args {
            normalized: &normalized,
            routes: &routes,
            sources,
            expert_gate: &records_gate,
            expert_up: &reads.records.up,
            tier_gate: &tier_gate,
            tier_up: &tier.up,
            activation,
            gate_scale: &gate_scale,
            up_scale: &up_scale,
            glu_limit: &glu_limit,
            routing: &routing,
            step,
            late,
            product,
        })
        .map(drop)
        .map_err(|error| failed(&error))
    };
    pass(&absent, 0, &mut product)?;
    pass(&sourced, 1, &mut product)?;
    let one = resident
        .call(charges, routed_gate_up::Args {
            normalized: &normalized,
            routes: &routes,
            sources: &sourced,
            expert_gate: &records_gate,
            expert_up: &reads.records.up,
            tier_gate: &tier_gate,
            tier_up: &tier.up,
            activation,
            gate_scale: &gate_scale,
            up_scale: &up_scale,
            glu_limit: &glu_limit,
        })
        .map_err(|error| failed(&error))?
        .value;
    // The late pass alone, over a product of ones: it writes the absent
    // choice's row, leaving every other as it found it.
    let mut marked = semantic_ones(charges, element, &[1, selected, features], entry, label)?;
    let ones = marked.read_to_host().map_err(|error| failed(&error))?;
    pass(&sourced, 1, &mut marked)?;
    let [passes, one, late] =
        [&product, &one, &marked].map(|tensor| tensor.read_to_host().map_err(|error| failed(&error)));
    let (one, late) = (one?, late?);
    if passes? != one {
        return Err(failed(&"the early and late passes differ from the resident gate/up's one pass"));
    }
    let row = one.len() / choices;
    let wanted = [&one[..row], &ones[row..]].concat();
    if late != wanted {
        return Err(failed(&"the late pass wrote other than the absent choice's row"));
    }
    Ok(())
}

/// A streamed layer's routed form: its binding, its resident entries and its
/// bank's decode route and ahead-of-step hint launches.
pub(super) enum Served<'a> {
    Fused {
        binding: RoutedBinding,
        kernels: &'a RoutedKernels,
        route: &'a RangedKernels<routed_route_streamed::Entry>,
        hints: &'a [Option<NativeKernel<routed_hint_route::Entry>>; HINT_LEADS as usize],
    },
    General {
        binding: crate::GeneralRoutedBinding,
        kernels: &'a GeneralRoutedKernels,
        select: &'a NativeKernel<routed_select_streamed::Entry>,
        hints: &'a [Option<NativeKernel<routed_hint_select::Entry>>; HINT_LEADS as usize],
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
            (
                BankedFeedForward::Routed(binding),
                BankRouteKernels::Fused { route, hints, .. },
                AttestedFeedForward::Routed(kernels),
            ) => Ok(Self::Fused { binding, kernels, route, hints }),
            (
                BankedFeedForward::GeneralRouted(binding),
                BankRouteKernels::General { select, hints, .. },
                AttestedFeedForward::GeneralRouted(kernels),
            ) => Ok(Self::General { binding, kernels, select, hints }),
            // A parallel block's routed branch is general.
            (
                BankedFeedForward::Parallel(parallel),
                BankRouteKernels::General { select, hints, .. },
                AttestedFeedForward::Parallel(kernels),
            ) => Ok(Self::General {
                binding: parallel.routed,
                kernels: &kernels.routed,
                select,
                hints,
            }),
            _ => Err(qualification_dynamic(
                "routed_bank",
                label,
                "the bank's route differs from its feed-forward's form",
            )),
        }
    }

    /// One decode row's route at each whole word, then each prepared hint
    /// launch at each resolved word of the next layer.
    fn qualify_routes(
        &self,
        charges: &seismic::Charges,
        binding: BankBinding,
        rows: SublayerRows,
        step: &Tensor,
        label: &str,
    ) -> Result<(), CatalogFailure> {
        let entry = match self {
            Self::Fused { .. } => "routed_route_streamed",
            Self::General { .. } => "routed_select_streamed",
        };
        let words = |extents: &[u64], set: &[(u64, i32)], entry| shared_words(charges, extents, set, entry, label);
        let zeros = |element, extents: &[u64], entry| semantic_zeros(charges, element, extents, entry, label);
        let (experts, next, selected) = (binding.experts, binding.next_experts, binding.selected);
        let (ring, flight) = (binding.words.ring_words(), binding.words.flight);
        let choices = word(selected, entry, label)?;
        let routing_words = binding.words.routing_words();
        let record = u64::from(SEQUENCE.unsigned_abs()) % flight * routing_words;
        let tier_experts = crate::native::tuning::routed::tier_experts(binding);
        let sited = i32::from(tier_experts > 0);
        let site = if tier_experts > 0 { -2 } else { -1 };
        let unsourced = (0..selected)
            .flat_map(|choice| [(SOURCE_RECORD * selected + choice, -1), (SOURCE_SITE * selected + choice, site)])
            .collect::<Vec<_>>();
        let unrouted = (0..selected)
            .map(|choice| (record + binding.words.routing_source(choice), -1))
            .collect::<Vec<_>>();
        let land_groups = binding.placer.land_groups();
        for whole in [0, 1] {
            let map = words(&[experts, MAP_WORDS], &unmapped(experts), entry)?;
            let tier = words(&[tier_experts], &(0..tier_experts).map(|expert| (expert, -1)).collect::<Vec<_>>(), entry)?;
            let policy = words(&[POLICY_WORDS], &[(POLICY_WHOLE, whole), (POLICY_HOME, whole * WHOLE_HOME), (POLICY_PRICED, 1)], entry)?;
            let mut file = words(&binding.words.file().extents(), &[], entry)?;
            let mut sources = zeros(Element::i32(), &[SOURCE_ROWS, 1, selected], entry)?;
            let mut routing = words(&[flight, routing_words], &[], entry)?;
            let mut keys = words(&[flight, constants::VERIFY_ROWS, binding.experts], &[], entry)?;
            let mut front = words(&[ring], &[], entry)?;
            let mut routes = zeros(Element::i32(), &[1, selected], entry)?;
            let mut land_grid = words(&[3], &[], entry)?;
            match *self {
                Self::Fused { binding: b, route, .. } => {
                    let kernel = route.first();
                    let (h, e) = (b.hidden, b.experts);
                    let norm = rows.norm(charges, b.norm, h, entry, label)?;
                    let mut empty_site = zeros(Element::i32(), &[0, 0], entry)?;
                    let mut empty_site_rows = zeros(Element::f32(), &[0, constants::VERIFY_ROWS, h], entry)?;
                    kernel
                        .call(charges, routed_route_streamed::Args {
                            residual: &rows.input(charges, 1, h, entry, label)?,
                            norm: &norm,
                            router: &zeros(b.router, &[e, h], entry)?,
                            shared_router: &zeros(Element::f32(), &[h], entry)?,
                            shared_gate: &zeros(b.shared_gate, &[b.shared, h], entry)?,
                            shared_up: &zeros(b.shared_up, &[b.shared, h], entry)?,
                            routes: &mut routes,
                            scores: &mut zeros(Element::f32(), &[1, selected], entry)?,
                            eps: 1.0e-5,
                            normalize: 1,
                            expert_scale: &unit_scale(charges, kernel, "ES", entry, label)?,
                            map: &map,
                            tier: &tier,
                            policy: &policy,
                            sited,
                            step,
                            file: &mut file,
                            sources: &mut sources,
                            routing: &mut routing,
                            keys: &mut keys,
                            front: &mut front,
                            land_grid: &mut land_grid,
                            land_groups,
                            site: &mut empty_site,
                            site_rows: &mut empty_site_rows,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?;
                }
                Self::General { binding: b, select: kernel, .. } => {
                    let shape = b.shape;
                    let (h, e) = (shape.hidden, shape.experts);
                    let router_norm = zeros(b.router_norm, &[h], entry)?;
                    kernel
                        .call(charges, routed_select_streamed::Args {
                            residual: &semantic_ones(charges, Element::f32(), &[1, h], entry, label)?,
                            norm: &zeros(b.norm, &[h], entry)?,
                            router_norm: &router_norm,
                            router: &zeros(b.router, &[e, h], entry)?,
                            bias: &zeros(Element::f32(), &[e], entry)?,
                            expert_scale: &unit_scale(charges, kernel, "ES", entry, label)?,
                            routes: &mut routes,
                            weights: &mut zeros(Element::f32(), &[1, selected], entry)?,
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
                            sources: &mut sources,
                            routing: &mut routing,
                            keys: &mut keys,
                            front: &mut front,
                            land_grid: &mut land_grid,
                            land_groups,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?;
                }
            }
            if whole == 0 {
                require_words(&routing, &[(record + RING_SEQUENCE, SEQUENCE), (record + RING_COUNT, choices)], entry, label)?;
                require_words(&routing, &unrouted, entry, label)?;
                require_words(&sources, &unsourced, entry, label)?;
                let slot = binding.words.file().lane(SEQUENCE as u64, 0);
                require_words(&file, &[(slot + SLOT_REQUEST, SEQUENCE), (slot + FILE_COUNT, choices)], entry, label)?;
            }
            // Every choice absent requests the step's slot, so the landing
            // launch lands; a whole group's lands nothing.
            let groups = if whole == 0 { land_groups } else { 0 };
            require_words(&land_grid, &[(0, groups), (1, 1), (2, 1)], entry, label)?;
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
        }
        match *self {
            Self::Fused { binding: b, hints, .. } => {
                let hint_entry = "routed_hint_route";
                let h = b.hidden;
                let norm = rows.norm(charges, b.norm, h, hint_entry, label)?;
                let activation = zeros(b.activation, &[0, b.selected], hint_entry)?;
                for (index, kernel) in hints.iter().enumerate() {
                    let Some(kernel) = kernel.as_ref() else { continue };
                    let ahead = index as u64 + 1;
                    for next_resolved in [0, 1] {
                        let next_policy = words(&[POLICY_WORDS], &[(POLICY_WHOLE, next_resolved)], hint_entry)?;
                        let mut rings = words(&[ahead, ring], &[], hint_entry)?;
                        kernel
                            .call(charges, routed_hint_route::Args {
                                residual: &rows.input(charges, 1, h, hint_entry, label)?,
                                activation: &activation,
                                next_norm: &norm,
                                next_router: &zeros(b.router, &[next, h], hint_entry)?,
                                after_norm: &norm,
                                after_router: &zeros(b.router, &[next, h], hint_entry)?,
                                eps: 1.0e-5,
                                step,
                                next_policy: &next_policy,
                                rings: &mut rings,
                            })
                            .map_err(|error| qualification_dynamic(hint_entry, label, error))?;
                        if next_resolved == 0 {
                            for lead in 0..ahead {
                                require_words(&rings, &[(lead * ring + RING_SEQUENCE, SEQUENCE)], hint_entry, label)?;
                            }
                        }
                    }
                }
            }
            Self::General { binding: b, hints, .. } => {
                let hint_entry = "routed_hint_select";
                let shape = b.shape;
                let h = shape.hidden;
                let router_norm = zeros(b.router_norm, &[h], hint_entry)?;
                let activation = zeros(b.activation, &[0, shape.selected], hint_entry)?;
                for (index, kernel) in hints.iter().enumerate() {
                    let Some(kernel) = kernel.as_ref() else { continue };
                    let ahead = index as u64 + 1;
                    for next_resolved in [0, 1] {
                        let next_policy = words(&[POLICY_WORDS], &[(POLICY_WHOLE, next_resolved)], hint_entry)?;
                        let mut rings = words(&[ahead, ring], &[], hint_entry)?;
                        kernel
                            .call(charges, routed_hint_select::Args {
                                residual: &semantic_ones(charges, Element::f32(), &[1, h], hint_entry, label)?,
                                activation: &activation,
                                next_router_norm: &router_norm,
                                next_router: &zeros(b.router, &[next, h], hint_entry)?,
                                next_bias: &zeros(Element::f32(), &[next], hint_entry)?,
                                after_router_norm: &router_norm,
                                after_router: &zeros(b.router, &[next, h], hint_entry)?,
                                after_bias: &zeros(Element::f32(), &[next], hint_entry)?,
                                score: shape.score,
                                epsilon: 1.0e-5,
                                step,
                                next_policy: &next_policy,
                                rings: &mut rings,
                            })
                            .map_err(|error| qualification_dynamic(hint_entry, label, error))?;
                        if next_resolved == 0 {
                            for lead in 0..ahead {
                                require_words(&rings, &[(lead * ring + RING_SEQUENCE, SEQUENCE)], hint_entry, label)?;
                            }
                        }
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
    stacks: PooledRecords,
    /// `RECORDS` records, each held.
    records: PooledRecords,
}

impl RecordFixture {
    fn new(charges: &seismic::Charges, tensor: RecordTensor, experts: u64, label: &str) -> Result<Self, CatalogFailure> {
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
        // Distinct at every byte from `converted` while keeping dense floats finite (`0x0000` -> `0x5555`).
        let marker = converted.iter().map(|byte| byte ^ 0x55).collect::<Vec<_>>();
        let shared = |extents: &[u64]| Tensor::shared(charges, tensor.source, extents).map_err(|error| failed(&error));
        let mut scratch = Tensor::zeros(charges, tensor.resident, &record).map_err(|error| failed(&error))?;
        scratch.write_from_host(&marker).map_err(|error| failed(&error))?;
        let bank = crate::planning::BankTier::Bank.contract_on(&charges.device());
        let pooled = |records, held| pooled_records(charges, bank, tensor.resident, &tensor.matrix, records, held).map_err(|error| failed(&error));
        let stacks = pooled(experts, 1)?;
        let records = pooled(RECORDS, RECORDS)?;
        let mut fixture = Self {
            conveyor: shared(&record)?,
            converted,
            marker,
            scratch,
            stacks,
            records,
        };
        fixture.mark().map_err(|error| failed(&error))?;
        Ok(fixture)
    }

    /// Write the marker into every record, through its view.
    fn mark(&mut self) -> Result<(), seismic::TensorError> {
        self.records.views.iter_mut().try_for_each(|view| view.write_from_host(&self.marker))
    }

    /// Require each record to hold `wanted`'s bytes, in order.
    fn require(&self, wanted: &[&[u8]], entry: &'static str, label: &str) -> Result<(), CatalogFailure> {
        self.records.views.iter().zip(wanted).try_for_each(|(view, wanted)| require_bytes(view, wanted, entry, label))
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

/// The words of a landing list holding `count` entries of step SEQUENCE,
/// entry 0 converting record `from` into record 0 (LANDING_ENTRY).
fn landing_list(count: i32, from: i32) -> Vec<(u64, i32)> {
    let mut words = vec![(LANDING_HEAD, SEQUENCE), (LANDING_HEAD + 1, count)];
    if count > 0 {
        words.extend([(constants::LANDING_ENTRY(0) + LANDING_FROM, from), (constants::LANDING_ENTRY(0) + LANDING_TO, 0)]);
    }
    words
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
    charges: &seismic::Charges,
    extents: &[u64],
    set: &[(u64, i32)],
    entry: &'static str,
    label: &str,
) -> Result<Tensor, CatalogFailure> {
    let tensor = Tensor::shared(charges, Element::i32(), extents)
        .map_err(|error| qualification_dynamic(entry, label, error))?;
    set_words(&tensor, set, entry, label)?;
    Ok(tensor)
}

/// The `(index, value)` pairs of `words`' nonzero words, the form `words`
/// and `require_words` take.
fn sparse(words: &[i32]) -> Vec<(u64, i32)> {
    words.iter().enumerate().filter(|(_, &word)| word != 0).map(|(index, &word)| (index as u64, word)).collect()
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
    use crate::operators::routed::fused_graph::TILE_ROWS;
    use magnitude_kernels::constants::VERIFY_ROWS;
    use crate::planning::RegionWords;
    use crate::{Expansion, ExecutionPath, GeneralRoutedShape};
    use magnitude_kernels::{
        routed_combine, routed_down_project, routed_down_sited, routed_experts, routed_experts_up, routed_gate_up, routed_group,
        routed_output, routed_route, routed_route_shared, routed_scatter, routed_select, routed_site_request, routed_up,
    };
    use crate::planning::Placer;

    /// `kernel` as the one range of its entry's rows.
    fn every_row<E: seismic::Entry>(kernel: seismic::NativeKernel<E>) -> RangedKernels<E> {
        RangedKernels::new(vec![(crate::native::tuning::ranged::EVERY_ROW, kernel)])
    }

    const HIDDEN: u64 = 256;
    const FEATURES: u64 = 128;
    const EXPERTS: u64 = 16;
    const SELECTED: u64 = 4;

    /// Prepares `entry` on `device` at the statics its implementation
    /// declares, each the first of `values` of its name (as preparation
    /// fixes them), its parameters at the declared defaults for `entry`'s
    /// bound elements.
    fn prepare<E: seismic::Entry>(
        device: &Device,
        entry: seismic::BoundEntry<E>,
        values: &[(&str, u64)],
    ) -> seismic::NativeKernel<E> {
        let implementation = entry.offered(device).unwrap().unwrap();
        let statics = implementation.statics.iter().fold(seismic::NativeSpecialization::new(), |statics, name| {
            let (_, value) = values
                .iter()
                .find(|(candidate, _)| *candidate == name.as_str())
                .unwrap_or_else(|| panic!("{} declares {name}", E::NAME));
            statics.with_static(name.clone(), *value)
        });
        entry
            .prepare(device, &implementation.default_specialization(&statics).unwrap())
            .unwrap()
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
            window_slots: EXPERTS,
            window_rounds: 2,
            conveyor: EXPERTS,
            tier: 0,
            placer: crate::planning::Placer::Tier,
            words: RegionWords {
                choices: VERIFY_ROWS * SELECTED,
                rounds: 2,
                flight: 2,
                sites: 2,
            },
        }
    }

    fn route_statics<'v>(bank: BankBinding, resident: &[(&'v str, u64)]) -> Vec<(&'v str, u64)> {
        crate::native::tuning::routed::bank_statics(bank)
            .into_iter()
            .chain(resident.iter().copied())
            .collect()
    }

    fn hint_statics<'v>(bank: BankBinding, ahead: u64, resident: &[(&'v str, u64)]) -> Vec<(&'v str, u64)> {
        [("AHEAD", ahead), ("EN", bank.next_experts)]
            .into_iter()
            .chain(resident.iter().copied())
            .collect()
    }

    /// `bank`'s entries on `device` beside its `route`, each at the statics
    /// preparation fixes (`Preparation::bank`).
    fn bank_kernels(device: &Device, bank: BankBinding, route: BankRouteKernels) -> ExpertBankKernels {
        let RecordTensors { gate: gated, up, down } = bank.record;
        let gate = gated.unwrap_or(up);
        let table = bank.words.scratch();
        let scratch = [("FLIGHT", table.slots()), ("LANES", table.lanes())];
        let activation = match &route {
            BankRouteKernels::Fused { activation, .. } | BankRouteKernels::General { activation, .. } => *activation,
        };
        let land = prepare(device, routed_land::native_entry_with(routed_land::Elements {
                SG: gate.source,
                SU: up.source,
                SD: down.source,
                UG: gate.resident,
                UU: up.resident,
                UD: down.resident,
                A: activation,
            }), &[("K", bank.selected)]);
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
            land,
            step: prepare(device, routed_step::native_entry(), &[]),
            scratch_request: prepare(device, routed_scratch_request::native_entry(), &[&[("K", bank.selected), ("E", bank.widest)], scratch.as_slice()].concat()),
            round: prepare(device, routed_round::native_entry_with(routed_round::Elements {
                    SG: gate.source,
                    SU: up.source,
                    SD: down.source,
                    UG: gate.resident,
                    UU: up.resident,
                    UD: down.resident,
                }), &scratch),
            scratch_grids: prepare(device, routed_scratch_grids::native_entry(), &[scratch.as_slice(), &[("RW", bank.window_rounds)]].concat()),
            pass_map: prepare(device, routed_pass_map::native_entry(), &[]),
            placement: DevicePlacement {
                seed: bank.record.map(|_, tensor| {
                    prepare(device, routed_copy::native_entry_with(routed_copy::Elements { U: tensor.resident }), &record_statics(tensor, bank.window_slots))
                }),
                seed_commit: prepare(device, routed_seed_commit::native_entry(), &[&[("X", bank.window_slots), ("BANK", bank.records)], scratch.as_slice()].concat()),
                list_grid: prepare(device, routed_list_grid::native_entry(), &[]),
            },
            site: None,
            // A gated record's passes, at the activations the fixtures
            // route in.
            gate_up: gated.map(|gate| {
                every_row(prepare(
                    device,
                    routed_gate_up_part::native_entry_with(routed_gate_up_part::Elements {
                        EGW: gate.resident,
                        EUW: up.resident,
                        A: Element::bf16(),
                    }),
                    &[("H", HIDDEN), ("K", bank.selected), ("F", FEATURES), ("GS", 0), ("US", 0), ("LB", 0)],
                ))
            }),
        }
    }

    fn metal() -> Device {
        crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), BackendName::Metal)
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

    /// A bank the host places into (unified memory's one store) qualifies:
    /// its landing launch converts expert 0 from staging into the bank
    /// record its answer names and sources it, and the prompt group's seed
    /// copy and commit place expert 0, as beside a tier.
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
            ("T", TILE_ROWS),
            ("GS", 0),
            ("US", 0),
            ("LB", 0),
            ("SN", 1),
            ("IN", 0),
        ];
        let f32 = Element::f32();
        let resident = RoutedKernels {
            route: prepare(&device, routed_route::native_entry_with(routed_route::Elements { NW: bf16, RW: bf16, SRW: f32, A: bf16 }), &statics),
            decode: RoutedDecodeKernels::SharedRoute {
                route: every_row(
                    prepare(&device, routed_route_shared::native_entry_with(routed_route_shared::Elements { NW: bf16, RW: bf16, SRW: f32, SGW: bf16, SUW: bf16, A: bf16 }), &statics),
                ),
                choices: every_row(
                    prepare(&device, routed_gate_up::native_entry_with(routed_gate_up::Elements { EGW: q8, EUW: q8, A: bf16 }), &statics),
                ),
            },
            output: every_row(
                prepare(&device, routed_output::native_entry_with(routed_output::Elements { EDW: down, SDW: bf16, A: bf16, NW: binding.norm }), &statics),
            ),
            operand_norm: binding.norm,
            group: prepare(&device, routed_group::native_entry(), &statics),
            experts: every_row(
                prepare(&device, routed_experts::native_entry_with(routed_experts::Elements { A: bf16, EGW: q8, EUW: q8, EDW: down }), &statics),
            ),
            combine: every_row(
                prepare(&device, routed_combine::native_entry_with(routed_combine::Elements { A: bf16, SGW: bf16, SUW: bf16, SDW: bf16 }), &statics),
            ),
        };
        let route = every_row(
            prepare(&device, routed_route_streamed::native_entry_with(routed_route_streamed::Elements {
                    NW: bf16,
                    RW: bf16,
                    SRW: f32,
                    SGW: bf16,
                    SUW: bf16,
                    A: bf16,
                }), &route_statics(bank, &statics)),
        );
        let hints = std::array::from_fn(|index| {
            let ahead = index as u64 + 1;
            Some(prepare(&device, routed_hint_route::native_entry_with(routed_hint_route::Elements { NW: bf16, RW: bf16, A: bf16 }), &hint_statics(bank, ahead, &statics)))
        });
        let kernels = bank_kernels(&device, bank, BankRouteKernels::Fused { activation: bf16, route, hints });
        qualify_bank(
            &seismic::Charges::tally(&device),
            BankedFeedForward::Routed(binding),
            bank,
            &kernels,
            &AttestedFeedForward::Routed(resident),
            SublayerRows::SINGLE,
            Some(&ResolvedNumericalProfile::standard().requirement_of(None)),
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
        let charges = seismic::Charges::tally(&device);
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
                experts_expansion: Expansion { gated: false, activation: 2, limit: None },
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
            ("T", TILE_ROWS),
            ("US", 0),
        ];
        let resident = GeneralRoutedKernels {
            select: prepare(&device, routed_select::native_entry_with(routed_select::Elements { NW: bf16, RNW: bf16, RW: bf16, A: bf16 }), &statics),
            experts: ExpertKernels::Plain {
                decode: every_row(
                    prepare(&device, routed_up::native_entry_with(routed_up::Elements { EUW: q8, A: bf16 }), &statics),
                ),
                grouped: every_row(
                    prepare(&device, routed_experts_up::native_entry_with(routed_experts_up::Elements { A: bf16, EUW: q8, EDW: q8 }), &statics),
                ),
            },
            down: every_row(
                prepare(&device, routed_down::native_entry_with(routed_down::Elements { EDW: q8, A: bf16, R: f32 }), &statics),
            ),
            group: prepare(&device, routed_group::native_entry(), &statics),
            scatter: prepare(&device, routed_scatter::native_entry_with(routed_scatter::Elements { A: bf16, R: f32 }), &statics),
            shared: None,
            latent: None,
        };
        let select = prepare(&device, routed_select_streamed::native_entry_with(routed_select_streamed::Elements {
                NW: bf16,
                RNW: bf16,
                RW: bf16,
                A: bf16,
            }), &route_statics(bank, &statics));
        let hints = std::array::from_fn(|index| {
            let ahead = index as u64 + 1;
            Some(prepare(&device, routed_hint_select::native_entry_with(routed_hint_select::Elements { RNW: bf16, RW: bf16, A: bf16 }), &hint_statics(bank, ahead, &statics)))
        });
        let mut kernels = bank_kernels(&device, bank, BankRouteKernels::General { activation: bf16, select, hints });
        if sited {
            let form = SiteForm {
                activation: bf16,
                function: binding.shape.experts_expansion.activation,
                limit: binding.shape.experts_expansion.limit,
            };
            kernels.site = Some(BankSite {
                request: prepare(&device, routed_site_request::native_entry_with(routed_site_request::Elements { A: bf16 }), &statics),
                combine: BankCombine::General {
                    project: every_row(
                        prepare(&device, routed_down_project::native_entry_with(routed_down_project::Elements { EDW: q8, A: bf16 }), &statics),
                    ),
                    sited: prepare(&device, routed_down_sited::native_entry_with(routed_down_sited::Elements { A: bf16, R: f32 }), &statics),
                },
                form,
                admitted: None,
            });
        }
        let site = kernels.site.as_ref().map(|site| site.form);
        let verdict = qualify_bank(
            &charges,
            BankedFeedForward::GeneralRouted(binding),
            bank,
            &kernels,
            &AttestedFeedForward::GeneralRouted(resident),
            SublayerRows::SINGLE,
            Some(requirement.unwrap_or(&ResolvedNumericalProfile::standard().requirement_of(None))),
            None,
        )
        .unwrap();
        // A load runs no site: its verdict waits on the job, which runs it.
        assert!(matches!(verdict, None | Some(crate::assessment::Priced::Pending(_))));
        site.zip(requirement).is_some_and(|(form, requirement)| {
            let request = crate::assessment::SiteRequest::of(&bank, form, requirement.clone());
            super::super::site::admit_site(&charges, &request).unwrap().is_ok()
        })
    }
}
