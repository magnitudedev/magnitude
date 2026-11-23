//! The cost model's interface: launch prices as main priced them call by
//! call, in each form, and the step model's two evaluations agreeing.

use crate::bank::{routed_padded_choices, routed_prompt_seconds};
use crate::compute::launch_terms;
use crate::*;
use seismic_measure::pricing::{self, Bytes, Experts, Figures, Flops, Format, HistoryFigure, HistoryLadder, HistoryRead, Layout, PriceError, Pricing, Rates, Storage, Work};

const BF16: Format = Format("bf16");
const Q4K: Format = Format("q4k");
const DENSE: Layout = Layout("dense");
const ROTATED: Layout = Layout("rotated_k4_uniform_v4");

/// `actual` is `expected` to a relative 1e-12: the same sums in another
/// order.
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(f64::MIN_POSITIVE), "{actual} against {expected}");
}

/// Rates of a GPU's order: bf16 at 300 GB/s, q4k at 200, upload at 20 and
/// host at 25; 20 and 10 TFLOP/s; a dense history ladder and a layout no
/// entry reads.
fn figures() -> Figures {
    let rates = Rates {
        activation: BF16,
        weights: Storage::ALL
            .iter()
            .flat_map(|&storage| {
                let scale = match storage {
                    Storage::Device => 1.,
                    Storage::Upload => 20. / 300.,
                    Storage::Host => 25. / 300.,
                };
                [(storage, BF16, 300e9 * scale), (storage, Q4K, 200e9 * scale)]
            })
            .collect(),
        flops: vec![(BF16, 20e12), (Q4K, 10e12)],
        history: vec![
            (DENSE, Ok(HistoryFigure { ladder: HistoryLadder::new(1024, [(1024, 10e-6), (4096, 30e-6), (16384, 90e-6)]), along: 300e9, slowdown: 1.5 })),
            (ROTATED, Err(PriceError::History("no native attention entry reads rotated history".into()))),
        ],
        launch: 5e-6,
        decode: None,
    };
    Figures { rates, composition: Ok(1.2), step_pipelined: 50e-6, step_round_trip: 150e-6 }
}

/// Launch counts of one launch below 4 rows and two from 4, of 8 decode
/// rows.
fn counts() -> LaunchCounts {
    LaunchCounts::new((1..=8).map(|rows| Ok(if rows < 4 { 1 } else { 2 })).collect())
}

fn projection() -> Work {
    Work { weights: vec![Bytes { format: Q4K, bytes: 30_000_000 }], row_flops: vec![Flops { format: Q4K, flops: 60_000_000 }], row_bytes: 8192, ..Work::default() }
}

fn attention() -> Work {
    Work {
        history: Some(HistoryRead { layout: Some(DENSE), bytes_per_token: 2048, flops_per_token: 8192, window: None }),
        sequence_bytes: 4096,
        ..Work::default()
    }
}

fn routed() -> Work {
    Work {
        experts: Some(Experts {
            experts: 128,
            selected: 8,
            slab_bytes: vec![Bytes { format: Q4K, bytes: 3_000_000 }],
            slab_flops: vec![Flops { format: Q4K, flops: 6_000_000 }],
        }),
        upload_row_bytes: 64,
        ..Work::default()
    }
}

/// A model's calls in decoder order: twelve layers of attention, a
/// projection and a routed call, the head's projection last.
fn calls() -> Vec<Work> {
    let mut calls = Vec::new();
    for _ in 0..12 {
        calls.extend([attention(), projection(), routed()]);
    }
    calls.push(projection());
    calls
}

/// `calls`' census.
fn census(calls: &[Work]) -> Census {
    Census::new(calls.iter().map(|work| (Kind { work: work.clone(), launches: counts() }, None)), 3, 8)
}

/// Main's launch price: every call in order, its own launches, summed and
/// composed.
fn main_launch(calls: &[Work], figures: &Figures, lookahead: bool, sequences: u64, rows: u64, depth: u32) -> f64 {
    let rates = &figures.rates;
    let sum = calls
        .iter()
        .map(|work| {
            let launches = if rows <= 8 { if rows < 4 { 1 } else { 2 } } else { 1 };
            pricing::call_seconds(work, rates.history_of(work).unwrap(), rates, rows, sequences, depth, launches).total()
        })
        .sum::<f64>();
    figures.composition.clone().unwrap() * sum + figures.step_seconds(lookahead) + 3. * figures.step_round_trip
}

/// A census's launches are main's call-by-call sums, kind by kind: at
/// decode rows, a verification's rows over fewer sequences, and a prefill
/// launch's beyond them.
#[test]
fn a_census_prices_its_launches_as_main_priced_them_call_by_call() {
    let (calls, figures) = (calls(), figures());
    let census = census(&calls);
    assert_eq!(census.kinds().len(), 3, "a layer's calls are three kinds and the head's projection one of them");
    let pricing = Pricing { figures: &figures, lookahead: true };
    for (sequences, rows, depth) in [(1, 1, 1), (1, 1, 3000), (4, 4, 9000), (2, 6, 100), (1, 512, 2000)] {
        let launch = launch_terms(&census, &pricing, sequences, rows, depth).unwrap();
        close(launch.total(), main_launch(&calls, &figures, true, sequences, rows, depth));
    }
    close(step_seconds(&census, &pricing, 4, 500).unwrap().total(), main_launch(&calls, &figures, true, 4, 4, 500));
    close(LaunchPrices::new(TargetCensus::Resident(census.clone()), figures.clone(), true).launch(300, 500).unwrap().resident, main_launch(&calls, &figures, true, 1, 300, 500));
    assert!(matches!(step_seconds(&census, &pricing, 9, 1), Err(PriceError::DecodeRows { rows: 9, decode_rows: 8 })));
}

/// A load's launches are main's in each form, a launch of rows as it runs
/// them (a decode step of a row a sequence, a prefill launch of one
/// sequence beyond the decode rows) at the exact depth, and none of no
/// rows; a load holding its experts whole decodes one form.
#[test]
fn a_loads_launches_are_mains_in_each_form() {
    let calls = calls();
    let streamed_calls = [calls.clone(), vec![projection(); 4]].concat();
    let figures = figures();
    let target = TargetCensus::of(census(&calls), Some(census(&streamed_calls)));
    assert_eq!(target.censuses().len(), 2);
    let banked = LaunchPrices::new(target, figures.clone(), true);
    let main = |calls: &[Work], sequences, rows, depth| main_launch(calls, &figures, true, sequences, rows, depth);
    for (rows, sequences, depth) in [(1, 1, 100), (4, 4, 500), (300, 1, 500)] {
        let launch = banked.launch(rows, depth).unwrap();
        close(launch.resident, main(&calls, sequences, rows.into(), depth));
        close(launch.streamed, main(&streamed_calls, sequences, rows.into(), depth));
        assert_eq!(banked.launch(rows, depth), Ok(launch), "a launch priced again is what it priced");
    }
    assert_eq!(banked.launch(0, 100), Err(PriceError::DecodeRows { rows: 0, decode_rows: 8 }));
    let resident = LaunchPrices::new(TargetCensus::of(census(&calls), None), figures.clone(), true).launch(3, 700).unwrap();
    assert_eq!(resident.resident, resident.streamed);
    close(resident.resident, main(&calls, 3, 3, 700));
}

/// Main's routed prompt seconds, layer by layer, are the per-kind sum's.
#[test]
fn a_prompts_routed_seconds_are_main_per_layer_sum() {
    let rates = figures().rates;
    let expert = |bytes| (vec![Bytes { format: Q4K, bytes }], vec![Flops { format: Q4K, flops: 2 * bytes }]);
    let experts = [128u64, 128, 64, 128, 64];
    let kinds = RoutedKind::number(experts);
    let layers = kinds.iter().map(|&kind| {
        let (bytes, flops) = expert(if kind == RoutedKind(0) { 3_000_000 } else { 2_000_000 });
        (kind, bytes, flops)
    });
    let pass = ExpertPass::new(rates, layers.collect(), HostReads::Read);
    let routed = PassLayers::of(&pass, experts, 8);
    assert_eq!(routed.len(), 2);
    let reads = |rows: u64| rows as f64 * 1e-6;
    let main = |pass_rows: u64, prompt: u64| {
        let one = |rows: u64| {
            experts
                .iter()
                .enumerate()
                .map(|(layer, &experts)| {
                    let active = pricing::expert_union(experts, 8, rows) / 2.;
                    2. * pass.seconds(layer, active, routed_padded_choices(active, (rows * 8) as f64 / 2.))
                })
                .sum::<f64>()
                + reads(rows)
        };
        (prompt / pass_rows) as f64 * one(pass_rows) + if !prompt.is_multiple_of(pass_rows) { one(prompt % pass_rows) } else { 0. }
    };
    for launches in [1, 2, 4] {
        close(routed_prompt_seconds(&pass, &routed, 2, 256, 10_000, launches, &reads), main(256 * launches, 10_000));
    }
}

/// The step model's run-length evaluation is its positional one over the
/// same layers, the queued case among them: host-tier layers whose hits
/// outlast their compute share, so each layer's file reads queue behind
/// the last's.
#[test]
fn a_run_by_run_step_is_the_layer_by_layer_step() {
    let read = ReadLine::new(4e-4, 1. / 2e9).unwrap();
    let tiered = ExpertLines::reference(ExpertLines::REFERENCE_RECORD);
    let unified = ExpertLines { host: HostLines { cpu: None, memory: None, landing: None, record_bytes: 2_000_000 }, ..tiered };
    let load = |tier, file| LayerLoad { routed: 64., rows: 1., tier, file, recall: 0. };
    let laid = [(&tiered, load(1, 2), 7), (&unified, load(0, 0), 3), (&unified, load(0, 3), 5), (&tiered, load(2, 0), 4), (&tiered, load(0, 1), 6)];
    let runs = laid.map(|(_, load, layers)| LayerRun { load, layers });
    let lines = laid.iter().flat_map(|&(lines, _, layers)| std::iter::repeat_n(*lines, layers)).collect::<Vec<_>>();
    let loads = runs.iter().flat_map(|run| std::iter::repeat_n(run.load, run.layers)).collect::<Vec<_>>();
    // 25 layers in 0.5 ms: 20 µs a layer against 62 hits of 5 µs each.
    for compute in [5e-4, 0.25] {
        let inputs = StepInputs { compute, head: 0., overlap: 1., read, sustained: read.seconds_per_byte(), ahead: ReadAhead::NONE, lines: &lines };
        let (positional, served, bytes, link) = inputs.one(&loads, &mut vec![0.; loads.len()]);
        let run_length = inputs.run([(&runs[..], 1)]);
        close(run_length.scheduled, positional);
        close(run_length.served, served);
        close(run_length.bytes, bytes);
        close(run_length.link, link);
    }
}

/// Qwen3.8-Flash-Next UD-Q4_K_XL's streamed decode pass on an M4 Pro, as
/// multi-front-cost.md derives it (§2, §3): its dense projections (q8,
/// 1.0625 bytes a weight, at 253.6 GB/s: their measured one-row time) and
/// readout (a 4-bit plane, 0.5625, at 240 GB/s), each at the decode row
/// line of `dense_output`'s measured slope; 48 routed layers of 512
/// experts choosing 10, each expert 3.09 MB at 249.2 GB/s (routed_output's
/// measured 4.0 µs a MB); every call one launch of 12.3 µs; and the rest of
/// the pass, whose rows cost nothing more, at what brings one row to its
/// measured 33.4 ms. The rows' other per-row work (≈ 0.28 ms a row,
/// unmeasured) is not priced.
mod front_pass {
    use super::*;

    const Q8: Format = Format("q8g32s");
    const Q4: Format = Format("q4_plane");
    const EXPERT: Format = Format("ud_q4_k_xl");

    fn figures() -> Figures {
        let formats = [(BF16, 250e9), (Q8, 253.6e9), (Q4, 240e9), (EXPERT, 249.2e9)];
        let rates = Rates {
            activation: BF16,
            weights: Storage::ALL.iter().flat_map(|&storage| formats.map(|(format, rate)| (storage, format, rate))).collect(),
            flops: formats.map(|(format, _)| (format, 20e12)).to_vec(),
            history: Vec::new(),
            launch: 12.3e-6,
            decode: Some(pricing::DecodeLine { fixed: 0.25e-12, row: 0.31e-12 }),
        };
        Figures { rates, composition: Ok(1.), step_pipelined: 0., step_round_trip: 0. }
    }

    /// A projection of `millions` million weights in `format` of
    /// `bytes` bytes a weight.
    fn projection(format: Format, millions: f64, bytes: f64) -> Work {
        let weights = (millions * 1e6) as u64;
        Work { weights: vec![Bytes { format, bytes: (weights as f64 * bytes) as u64 }], row_flops: vec![Flops { format, flops: 2 * weights }], ..Work::default() }
    }

    fn census() -> Census {
        let dense = [(36, 43.), (48, 15.8), (97, 3.35), (24, 17.), (97, 3.4)];
        let routed = Work {
            experts: Some(Experts {
                experts: 512,
                selected: 10,
                slab_bytes: vec![Bytes { format: EXPERT, bytes: 3_090_000 }],
                slab_flops: vec![Flops { format: EXPERT, flops: 11_000_000 }],
            }),
            ..Work::default()
        };
        let rest = Work { weights: vec![Bytes { format: BF16, bytes: 1_881_150_000 }], ..Work::default() };
        let one = |work: Work| (Kind { work, launches: LaunchCounts::new(vec![Ok(1); 8]) }, None);
        let calls = dense
            .into_iter()
            .flat_map(|(calls, millions)| std::iter::repeat_n(projection(Q8, millions, 1.0625), calls))
            .chain([projection(Q4, 636., 0.5625), rest])
            .chain(std::iter::repeat_n(routed, 48));
        Census::new(calls.map(one), 0, 8)
    }

    fn prices() -> LaunchPrices {
        LaunchPrices::new(TargetCensus::of(census(), Some(census())), figures(), true)
    }

    /// Each pass of one request's exact row and 0..=7 front rows, in ms,
    /// in the streamed form, against `table`'s C(1..8) within 10%.
    fn within(routes: &FrontRoutes, table: [f64; 8]) {
        let prices = prices();
        for (ahead, expected) in table.into_iter().enumerate() {
            let fronted = u32::from(ahead > 0);
            let priced = prices.front(1, fronted, ahead as u32, 0, routes).unwrap().streamed * 1e3;
            assert!((priced / expected - 1.).abs() <= 0.10, "{ahead} front rows: {priced:.2} ms against {expected} ms");
        }
    }

    /// Code (§3: measured overlap by lag, the bank holding a third of the
    /// fresh picks) prices C(1..8) within 10% of the derived table, a pass
    /// of seven front rows about 15 ms over one of none.
    #[test]
    fn code_prices_its_derived_pass_at_every_depth() {
        let code = FrontRoutes { fresh: vec![8.18, 7.16, 6.56, 6.09, 5.71, 5.37, 5.10, 4.85], held: Some(0.33) };
        within(&code, [33.4, 35.6, 37.5, 39.6, 41.9, 44.5, 47.5, 50.8]);
    }

    /// Prose at prose-continue's overlap and every pick held, and any text
    /// before its routes are measured (independent routing, every pick
    /// held), price the derived table's two ends.
    #[test]
    fn prose_and_unmeasured_routes_price_the_derived_ends() {
        let prose = FrontRoutes { fresh: vec![3.10, 1.73, 1.33, 1.07, 0.91, 0.80, 0.71, 0.65], held: Some(1.) };
        within(&prose, [33.4, 35.5, 37.0, 38.5, 40.3, 42.5, 45.0, 47.9]);
        within(&FrontRoutes::default(), [33.4, 39.5, 45.7, 52.0, 58.6, 65.6, 73.0, 80.6]);
    }

    /// Held whole, a front row reads every fresh expert: the resident form
    /// prices code's pass as the unheld bank does.
    #[test]
    fn a_whole_bank_reads_every_fresh_expert() {
        let code = FrontRoutes { fresh: vec![8.18, 7.16], held: Some(0.33) };
        let whole = FrontRoutes { held: Some(1.), ..code.clone() };
        let prices = prices();
        let (held, all) = (prices.front(1, 1, 2, 0, &code).unwrap(), prices.front(1, 1, 2, 0, &whole).unwrap());
        close(held.resident, all.streamed);
        close(held.resident, all.resident);
        assert!(held.streamed < held.resident);
    }

    /// At the code and prose operating points, on these prices: code's
    /// measured run (245.3 misses of 1 ms a step plain, 214.6 at seven
    /// rows, 115.5 unused a pass, about 16 a depth) runs none, its rows
    /// wasting more than they keep placed; with its waste cut to a tenth it
    /// runs seven; prose (1.3 misses) runs none.
    #[test]
    fn the_regime_prices_code_and_prose_at_their_operating_points() {
        let prices = prices();
        let passes = |routes: &FrontRoutes| (0..8).map(|ahead| prices.front(1, u32::from(ahead > 0), ahead, 0, routes).unwrap().streamed).collect::<Vec<_>>();
        let code = FrontRoutes { fresh: vec![8.18, 7.16, 6.56, 6.09, 5.71, 5.37, 5.10, 4.85], held: Some(0.33) };
        let prose = FrontRoutes { fresh: vec![3.10, 1.73, 1.33, 1.07, 0.91, 0.80, 0.71, 0.65], held: Some(1.) };
        let regime = |routes: &FrontRoutes, plain: f64, deepest: f64, unused: f64| {
            let wasted = (0..8).map(|depth| Some(if depth == 0 { 0. } else { unused * 1e-3 })).collect();
            let needed = crate::NeededByForm::both((0..8).map(|k| crate::Needed::Executed(plain - (plain - deepest) * k as f64 / 7.)).collect());
            Regime { resident: passes(routes), needed, stand: Some(0.997), accepts: vec![None; 7], stall: None, overhead: None, paths: vec![PathTerm { seconds: 1e-3, wasted: crate::Waste { front: wasted, verify: vec![None; 8] } }], routes: crate::VerifyRoutes::default() }
        };
        assert_eq!(regime(&code, 245.3, 214.6, 16.).depth(), 0);
        assert_eq!(regime(&code, 245.3, 214.6, 1.6).depth(), 7);
        assert_eq!(regime(&prose, 1.3, 1.3, 0.).depth(), 0);
    }
}

/// A path's records are its rate times a move's read plus demand queue wait
/// and its victim's release: 1 MiB moves at 100 µs latency and 2 GB/s
/// (524.3 µs a transfer) hold ⌈624.3/524.3⌉ = 2 in flight with no queue,
/// ⌈5624.3/524.3⌉ = 11 with a 5 ms hint-window queue, and a release of
/// 10 ms beside them makes ⌈10624.3/524.3⌉ = 21 free (never below `ahead`).
#[test]
fn a_paths_records_are_its_rate_times_a_moves_read_and_release() {
    let line = ReadLine::new(100e-6, 1. / 2e9).unwrap();
    assert_eq!(PathRecords::of(line, 1 << 20, 0., 10e-3), Some(PathRecords { ahead: 2, reserve: 21 }));
    assert_eq!(PathRecords::of(line, 1 << 20, 5e-3, 10e-3), Some(PathRecords { ahead: 11, reserve: 21 }));
    // When queue exceeds release, reserve is at least ahead so every in-flight
    // read ahead finds a free bank/tier record to land into.
    assert_eq!(PathRecords::of(line, 1 << 20, 10e-3, 2e-3), Some(PathRecords { ahead: 21, reserve: 21 }));
    assert_eq!(PathRecords::of(line, 1 << 20, 0., 20e-3).map(|paths| paths.reserve), Some(40));
    // Latency alone in flight: a free path's move holds its record once.
    assert_eq!(PathRecords::of(ReadLine::new(0., 1. / 2e9).unwrap(), 1 << 20, 0., 0.), Some(PathRecords { ahead: 1, reserve: 1 }));
    // A line without a bandwidth term gives no rate.
    assert_eq!(PathRecords::of(ReadLine::new(100e-6, 0.).unwrap(), 1 << 20, 0., 10e-3), None);
}

/// A victim waits, on the mean, half the step its moves' reads leave: a
/// 100 ms step reading 60 ms waits 20 ms; a step its reads fill, none; a
/// step reading nothing (every expert held), half of it.
#[test]
fn a_victims_release_is_half_the_step_its_reads_leave() {
    assert!((PathRecords::release(0.1, 0.06) - 0.02).abs() < 1e-15);
    assert_eq!(PathRecords::release(0.1, 0.1), 0.);
    assert_eq!(PathRecords::release(0.1, 0.12), 0.);
    assert_eq!(PathRecords::release(0.1, 0.), 0.05);
}

/// `BankPrices::step_seconds` credits the host tier when `HeldSlots { slots, host }`
/// is passed (unlike `HeldSlots::bank(slots)` which drops the host tier), and
/// `BankPrices::prompt_seconds` prices cold misses via `Pass::Group`
/// (`ExpertLines::group_stage`) without double-counting compute.
#[test]
fn bank_prices_step_accounts_for_host_tier_and_prompt_uses_group_stage() {
    let read = ReadLine::new(4e-4, 1. / 2e9).unwrap();
    let lines = ExpertLines::reference(ExpertLines::REFERENCE_RECORD);
    let experts = vec![32u64, 32];
    let layers = StreamedLayers::new(experts.clone(), 4, std::sync::Arc::from([lines, lines]));
    let pass = ExpertPass::new(
        figures().rates,
        vec![
            (RoutedKind(0), vec![Bytes { format: Q4K, bytes: ExpertLines::REFERENCE_RECORD }], vec![Flops { format: Q4K, flops: 2 * ExpertLines::REFERENCE_RECORD }]),
            (RoutedKind(0), vec![Bytes { format: Q4K, bytes: ExpertLines::REFERENCE_RECORD }], vec![Flops { format: Q4K, flops: 2 * ExpertLines::REFERENCE_RECORD }]),
        ],
        HostReads::Read,
    );
    let routed = PassLayers::of(&pass, experts, 4);
    let compute = FormSeconds { resident: 0.010, streamed: 0.012 };
    let rates = WorkloadRates::prior(0.012, 100., 512., 0.1);
    let prices = BankPrices::new(layers, pass, routed, 1, read, compute, rates, 256, 512, 8);

    // Host tier holds all 32 experts while VRAM holds 8: decode step is strictly
    // faster with host tier credited than with `HeldSlots::bank(8)` (file misses).
    let with_host = HeldSlots { slots: 8, host: 32 };
    let without_host = HeldSlots::bank(8);
    assert!(
        prices.step_seconds(with_host) < prices.step_seconds(without_host),
        "host tier must reduce decode step time: {} vs {}",
        prices.step_seconds(with_host),
        prices.step_seconds(without_host)
    );

    // Prompt seconds at 2 launches (512 rows, 1 pass) equals routed compute
    // (the whole-bank prompt seconds) plus `Pass::Group` staging of the 24
    // host-tier misses per layer.
    let whole_prompt = prices.prompt_seconds(2, HeldSlots::bank(32));
    let tiered_prompt = prices.prompt_seconds(2, with_host);
    let expected_stage = 2. * lines.group_stage(&read, 24, 0);
    close(tiered_prompt - whole_prompt, expected_stage);
}

/// `StreamedLayers::path_records` scales `ahead` staging records with
/// `decode_rows` so multi-row verify bursts do not starve cross-layer hints,
/// while preserving `reserve` from the single-row release wait when it already
/// dominates `ahead`.
#[test]
fn path_records_scales_ahead_for_multi_row_verify_without_inflating_reserve() {
    let read = ReadLine::new(3e-4, 1. / 4.6e9).unwrap();
    let record_bytes = 1_800_000u64;
    let lines = ExpertLines::reference(record_bytes);
    let num_layers = 48usize;
    let layers = StreamedLayers::new(
        vec![512u64; num_layers],
        10,
        std::sync::Arc::from(vec![lines; num_layers]),
    );
    let compute = FormSeconds { resident: 0.032, streamed: 0.035 };
    let held = HeldSlots::bank(355);

    let single = layers.path_records(read, compute, record_bytes, held, 2, num_layers as u64, 1).unwrap();
    let verify = layers.path_records(read, compute, record_bytes, held, 2, num_layers as u64, 8).unwrap();

    assert!(
        verify.ahead > single.ahead,
        "multi-row verify must size more ahead staging records than M=1: verify={verify:?} single={single:?}"
    );
    assert!(
        verify.reserve >= verify.ahead,
        "reserve must always cover ahead: verify={verify:?}"
    );
    assert_eq!(
        verify.reserve,
        single.reserve.max(verify.ahead),
        "reserve should not inflate beyond max(single-row release, verify.ahead): verify={verify:?} single={single:?}"
    );
}
