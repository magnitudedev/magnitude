//! The controller over synthetic token streams, against every fixed width.
//!
//! A stream gives each position the true number of proposals a block drafted
//! there gets right, drawn from a per-depth curve that changes by segment.
//! Launch costs are a table. The controller and each fixed width decode the
//! same stream, so their total times compare directly.

use super::{DraftController, DraftLoad, RoundChoice, TimedLaunch};
use magnitude_executor::TokenId;
use std::{sync::Arc, time::Duration};

const WIDEST: usize = 15;

/// How often a block is right through each depth, repeating a passage and
/// continuing one (Qwen 3.5 9B with its drafter, measured).
const PREDICTABLE: [f64; WIDEST] = [
    1.0, 1.0, 1.0, 0.99, 0.98, 0.97, 0.95, 0.90, 0.75, 0.65, 0.60, 0.50, 0.50, 0.30, 0.25,
];
const UNPREDICTABLE: [f64; WIDEST] = [
    0.58, 0.25, 0.07, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];
const PERFECT: [f64; WIDEST] = [1.0; WIDEST];

/// Launch milliseconds.
#[derive(Clone, Copy)]
struct Costs {
    draft: fn(usize) -> f64,
    verify: fn(usize) -> f64,
}

/// At 64k context (measured round times, split).
const LONG: Costs = Costs {
    draft: |generated| match generated {
        0 => 0.0,
        1 => 21.0,
        2..=3 => 21.7,
        4..=7 => 22.0,
        _ => 32.0,
    },
    verify: |rows| match rows {
        1 => 29.6,
        2 => 29.9,
        3..=4 => 39.3,
        5..=8 => 57.3,
        _ => 117.8,
    },
};

/// At 4k context (measured).
const SHORT: Costs = Costs {
    draft: |generated| match generated {
        0 => 0.0,
        1..=7 => 14.0,
        _ => 27.0,
    },
    verify: |rows| match rows {
        1 => 24.8,
        2 => 25.5,
        3..=4 => 31.7,
        5..=8 => 34.8,
        _ => 62.0,
    },
};

/// The 64k launches with a 16-row verification on the trend of the others.
const EVEN: Costs = Costs {
    draft: LONG.draft,
    verify: |rows| match rows {
        9.. => 93.0,
        rows => (LONG.verify)(rows),
    },
};

#[derive(Clone)]
struct Stream {
    /// Segment lengths in tokens with their curves.
    segments: Vec<(usize, [f64; WIDEST])>,
    seed: u64,
}

impl Stream {
    fn of(segments: &[(usize, [f64; WIDEST])]) -> Self {
        Self {
            segments: segments.to_vec(),
            seed: 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn len(&self) -> usize {
        self.segments.iter().map(|(tokens, _)| tokens).sum()
    }

    fn token(position: usize) -> TokenId {
        TokenId(position as u32)
    }

    /// Proposals a block anchored at `position` gets right.
    fn right(&self, position: usize) -> usize {
        let mut start = 0;
        let curve = self
            .segments
            .iter()
            .find_map(|(tokens, curve)| {
                start += tokens;
                (position < start).then_some(curve)
            })
            .unwrap_or(&self.segments.last().unwrap().1);
        let mut state = self.seed ^ (position as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 30;
        state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
        state ^= state >> 31;
        let draw = (state >> 11) as f64 / (1u64 << 53) as f64;
        curve.iter().take_while(|right| draw < **right).count()
    }

    fn block(&self, position: usize, generated: usize) -> Vec<TokenId> {
        let right = self.right(position);
        (1..=generated)
            .map(|depth| {
                if depth <= right {
                    Self::token(position + depth)
                } else {
                    TokenId(u32::MAX - depth as u32)
                }
            })
            .collect()
    }
}

#[derive(Default)]
struct Run {
    milliseconds: f64,
    /// Each round's choice and the tokens it committed.
    rounds: Vec<(RoundChoice, usize)>,
}

impl Run {
    /// Rounds starting in `tokens`.
    fn within(
        &self,
        tokens: std::ops::Range<usize>,
    ) -> impl Iterator<Item = &(RoundChoice, usize)> + '_ {
        let mut position = 0;
        self.rounds.iter().filter(move |(_, committed)| {
            let inside = tokens.contains(&position);
            position += committed;
            inside
        })
    }

    /// The share of the rounds starting in `tokens` that verify `verified`.
    fn share(&self, verified: usize, tokens: std::ops::Range<usize>) -> f64 {
        let rounds = self.within(tokens.clone()).count();
        let verifying = self
            .within(tokens)
            .filter(|(choice, _)| choice.verified == verified)
            .count();
        verifying as f64 / rounds as f64
    }

    fn looks(&self) -> usize {
        self.rounds.iter().filter(|(choice, _)| choice.look).count()
    }

    /// Every round as `generated/verified/committed`, a look marked `*`.
    fn rounds(&self) -> String {
        self.rounds
            .iter()
            .map(|(choice, committed)| {
                format!(
                    "{}/{}/{}{}",
                    choice.generated,
                    choice.verified,
                    committed,
                    if choice.look { "*" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

const CONTEXT: usize = 60_000;

/// One request over `stream` under the controller, after a prompt of
/// `context` tokens.
fn adaptive_at(stream: &Stream, costs: Costs, load: &Arc<DraftLoad>, context: usize) -> Run {
    adaptive_on(stream, costs, load, context, |_| 1.0)
}

/// One request over `stream` on a machine whose launches in a round anchored
/// at context position `p` take `slowdown(p)` times their cost. The run's
/// time counts the launches' own costs, so it compares the controller's
/// choices alone.
fn adaptive_on(
    stream: &Stream,
    costs: Costs,
    load: &Arc<DraftLoad>,
    context: usize,
    slowdown: impl Fn(usize) -> f64,
) -> Run {
    let mut controller = DraftController::new(WIDEST, load.clone());
    let mut run = Run::default();
    let mut position = 0;
    while position < stream.len() {
        let slowdown = slowdown(context + position);
        let limit = (stream.len() - position - 1).min(WIDEST);
        let generated = controller.generate(limit, context + position);
        let mut verified = 0;
        if generated > 0 {
            let seconds = (costs.draft)(generated) / 1000.0;
            run.milliseconds += seconds * 1000.0;
            controller.timed(
                TimedLaunch::Draft,
                Some(Duration::from_secs_f64(seconds * slowdown)),
                1,
                generated + 1,
            );
            let block = stream.block(position, generated);
            controller.drafted(generated, &block);
            verified = controller.verify(block.len(), limit);
        }
        let seconds = (costs.verify)(verified + 1) / 1000.0;
        run.milliseconds += seconds * 1000.0;
        controller.timed(
            TimedLaunch::Target,
            Some(Duration::from_secs_f64(seconds * slowdown)),
            1,
            verified + 1,
        );
        let accepted = stream.right(position).min(verified);
        let committed = (position + 1..=position + accepted + 1)
            .map(Stream::token)
            .collect::<Vec<_>>();
        controller.committed(&committed);
        run.rounds.push((controller.choice(), accepted + 1));
        position += accepted + 1;
    }
    run
}

fn adaptive(stream: &Stream, costs: Costs, load: &Arc<DraftLoad>) -> Run {
    adaptive_at(stream, costs, load, CONTEXT)
}

/// One request over `stream` at a fixed width, drafting its whole width or
/// nothing as a fixed budget does.
fn fixed(stream: &Stream, costs: Costs, width: usize) -> f64 {
    let mut milliseconds = 0.0;
    let mut position = 0;
    while position < stream.len() {
        let limit = stream.len() - position - 1;
        let width = if width <= limit { width } else { 0 };
        milliseconds += (costs.draft)(width) + (costs.verify)(width + 1);
        position += stream.right(position).min(width) + 1;
    }
    milliseconds
}

const FIXED: [usize; 5] = [0, 1, 3, 7, 15];

fn fastest_fixed(stream: &Stream, costs: Costs) -> (usize, f64) {
    FIXED
        .into_iter()
        .map(|width| (width, fixed(stream, costs, width)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// A load whose launches a request over `stream` has measured.
fn warmed(stream: &Stream, costs: Costs) -> Arc<DraftLoad> {
    let load = Arc::new(DraftLoad::default());
    adaptive(stream, costs, &load);
    load
}

/// `run` took at most `share` longer than the fastest fixed width over
/// `stream`, which is `width`.
#[track_caller]
fn within_of_fixed(run: &Run, stream: &Stream, costs: Costs, width: usize, share: f64) {
    let (fastest, milliseconds) = fastest_fixed(stream, costs);
    assert_eq!(fastest, width, "the fastest fixed width");
    assert!(
        run.milliseconds <= milliseconds * (1.0 + share),
        "adaptive {:.0} ms, fixed {width} {milliseconds:.0} ms ({:+.1}%), {} looks: {}",
        run.milliseconds,
        (run.milliseconds / milliseconds - 1.0) * 100.0,
        run.looks(),
        run.rounds()
    );
}

#[test]
fn steady_text_decodes_as_fast_as_its_best_fixed_width() {
    // Everything the controller spends on looks is its share of the time.
    for (curve, costs, width) in [
        (PREDICTABLE, LONG, 7),
        (UNPREDICTABLE, LONG, 0),
        (PREDICTABLE, SHORT, 7),
        (PERFECT, EVEN, 15),
        (PERFECT, LONG, 15),
    ] {
        let stream = Stream::of(&[(3_000, curve)]);
        let run = adaptive(&stream, costs, &warmed(&stream, costs));
        within_of_fixed(&run, &stream, costs, width, 0.02);
    }
}

#[test]
fn a_loads_first_request_pays_only_for_measuring_its_launches() {
    let stream = Stream::of(&[(2_000, PREDICTABLE)]);
    let run = adaptive(&stream, LONG, &Arc::new(DraftLoad::default()));
    assert_eq!(run.rounds[0].0.verified, WIDEST);
    within_of_fixed(&run, &stream, LONG, 7, 0.03);
}

/// The benchmark's shape: a short request at a short context, then one
/// 256-token request at a long one.
#[test]
fn a_short_request_at_a_new_context_stays_close_to_its_best_fixed_width() {
    let load = Arc::new(DraftLoad::default());
    adaptive_at(&Stream::of(&[(256, PREDICTABLE)]), SHORT, &load, 758);
    let stream = Stream::of(&[(256, PREDICTABLE)]);
    let run = adaptive(&stream, LONG, &load);
    within_of_fixed(&run, &stream, LONG, 7, 0.05);
}

#[test]
fn a_change_of_text_beats_every_fixed_width() {
    let stream = Stream::of(&[(700, PREDICTABLE), (500, UNPREDICTABLE), (700, PREDICTABLE)]);
    let load = warmed(&Stream::of(&[(500, PREDICTABLE)]), LONG);
    let run = adaptive(&stream, LONG, &load);
    for width in FIXED {
        let fixed = fixed(&stream, LONG, width);
        assert!(
            run.milliseconds < fixed,
            "adaptive {:.0} ms, fixed {width} {fixed:.0} ms",
            run.milliseconds
        );
    }
    // Each segment at its own best fixed width.
    let oracle = fixed(&Stream::of(&[(700, PREDICTABLE)]), LONG, 7) * 2.0
        + fixed(&Stream::of(&[(500, UNPREDICTABLE)]), LONG, 0);
    assert!(
        run.milliseconds <= oracle * 1.05,
        "adaptive {:.0} ms, oracle {oracle:.0} ms ({:+.1}%): {}",
        run.milliseconds,
        (run.milliseconds / oracle - 1.0) * 100.0,
        run.rounds()
    );
    // It follows the text: 7 while predictable, nothing while not, and 7
    // again, each within a few rounds of the change.
    assert!(run.share(7, 0..700) >= 0.9, "{}", run.rounds());
    assert!(run.share(7, 760..1_200) <= 0.05, "{}", run.rounds());
    assert!(run.share(7, 1_300..1_900) >= 0.9, "{}", run.rounds());
}

/// Predictable text broken by short stretches that are not.
#[test]
fn short_unpredictable_stretches_cost_little_more_than_they_must() {
    let segments = (0..10)
        .flat_map(|_| [(200, PREDICTABLE), (30, UNPREDICTABLE)])
        .collect::<Vec<_>>();
    let stream = Stream::of(&segments);
    let run = adaptive(
        &stream,
        LONG,
        &warmed(&Stream::of(&[(500, PREDICTABLE)]), LONG),
    );
    for width in FIXED {
        let fixed = fixed(&stream, LONG, width);
        assert!(
            run.milliseconds < fixed * 1.02,
            "adaptive {:.0} ms, fixed {width} {fixed:.0} ms ({:+.1}%): {}",
            run.milliseconds,
            (run.milliseconds / fixed - 1.0) * 100.0,
            run.rounds()
        );
    }
}

#[test]
fn text_that_becomes_perfect_is_verified_wider() {
    let stream = Stream::of(&[(1_000, PREDICTABLE), (2_000, PERFECT)]);
    let load = warmed(&Stream::of(&[(500, PREDICTABLE)]), EVEN);
    let run = adaptive(&stream, EVEN, &load);
    assert!(run.share(15, 1_500..2_980) >= 0.9, "{}", run.rounds());
}

#[test]
fn the_last_tokens_of_a_request_are_still_drafted() {
    let stream = Stream::of(&[(256, PERFECT)]);
    let run = adaptive(&stream, LONG, &warmed(&stream, LONG));
    let undrafted = run
        .rounds
        .iter()
        .filter(|(choice, _)| choice.generated == 0)
        .count();
    assert!(undrafted <= 2, "{}", run.rounds());
}

#[test]
fn a_launch_that_proves_slow_is_not_taken_again() {
    // A 16-row verification ten times the others.
    let costs = Costs {
        draft: LONG.draft,
        verify: |rows| if rows > 8 { 600.0 } else { (LONG.verify)(rows) },
    };
    let stream = Stream::of(&[(3_000, PERFECT)]);
    let run = adaptive(&stream, costs, &Arc::new(DraftLoad::default()));
    let wide = run
        .rounds
        .iter()
        .filter(|(choice, _)| choice.verified == 15)
        .count();
    assert_eq!(wide, 1, "{}", run.rounds());
}

#[test]
fn a_launch_not_measured_here_costs_no_more_than_a_larger_one_proven_here() {
    use super::{Durations, LaunchCosts, RoundCosts};
    let load = DraftLoad::default();
    {
        let mut costs = load.costs.lock().unwrap();
        // A held-up first duration of width 3 at a short context.
        let mut short = LaunchCosts::default();
        short.draft.insert(3, Durations(vec![(1, 0.0795)]));
        costs.insert((512, 1), short);
        // Width 7 proven at the request's context.
        let mut long = LaunchCosts::default();
        long.draft
            .insert(7, Durations(vec![(2, 0.0099), (3, 0.0099), (4, 0.0099)]));
        costs.insert((4096, 1), long);
    }
    let costs = RoundCosts::of(&load, 4096, 1, None);
    assert_eq!(costs.draft(3), 0.0099);
}

#[test]
fn a_launch_that_recorded_its_graph_prices_nothing() {
    let drafted = |duration: Option<Duration>| {
        let mut controller = DraftController::new(7, Arc::new(DraftLoad::default()));
        let generated = controller.generate(7, CONTEXT);
        assert!(generated > 0);
        controller.timed(TimedLaunch::Draft, duration, 1, generated + 1);
        controller.measured().draft.contains_key(&generated)
    };
    // A drafter launch that recorded its graph measured that recording; one
    // that replayed it measured itself.
    assert!(!drafted(None));
    assert!(drafted(Some(Duration::from_millis(9))));
}

#[test]
fn requests_sharing_a_launch_give_its_rows_to_the_proposals_most_often_right() {
    let load = Arc::new(DraftLoad::default());
    let stream = Stream::of(&[(64, PERFECT), (64, UNPREDICTABLE)]);
    // The launches two requests share, as rounds measured them.
    let mut measuring = DraftController::new(7, load.clone());
    for rows in [2, 4, 8, 16] {
        assert_eq!(measuring.generate(0, CONTEXT), 0);
        let seconds = (LONG.verify)(rows) / 1000.0;
        measuring.timed(
            TimedLaunch::Target,
            Some(Duration::from_secs_f64(seconds)),
            2,
            rows,
        );
    }
    // One request whose blocks have been right, one whose blocks have not.
    let outlook = |position: usize| {
        let mut controller = DraftController::new(7, Arc::new(DraftLoad::default()));
        for round in 0..4 {
            let anchor = position + 8 * round;
            controller.drafted(7, &stream.block(anchor, 7));
            let committed = (anchor + 1..=anchor + 8)
                .map(Stream::token)
                .collect::<Vec<_>>();
            controller.committed(&committed);
        }
        let mut controller = DraftController {
            load: load.clone(),
            ..controller
        };
        controller.generate(0, CONTEXT);
        controller.drafted(7, &stream.block(position + 32, 7));
        controller.outlook().unwrap()
    };
    let (right, wrong) = (outlook(0), outlook(64));
    // Eight rows: the two anchors and six of the proposals that are right.
    assert_eq!(
        super::share_rows(&[right.clone(), wrong.clone()], 16),
        Some(vec![7, 1])
    );
    assert_eq!(
        super::share_rows(&[wrong, right.clone()], 16),
        Some(vec![1, 7])
    );
    // Two requests alike share alike, in the class committing the most per
    // second: eight rows here, a 16-row launch costing over twice as much.
    assert_eq!(
        super::share_rows(&[right.clone(), right.clone()], 16),
        Some(vec![4, 4])
    );
    // One request decides for itself.
    assert_eq!(super::share_rows(&[right], 16), None);
}

#[test]
fn scoring_counts_what_a_full_verification_would_accept() {
    let stream = Stream::of(&[(400, PREDICTABLE), (400, UNPREDICTABLE)]);
    for position in (0..780).step_by(7) {
        let mut controller = DraftController::new(WIDEST, Arc::new(DraftLoad::default()));
        controller.drafted(WIDEST, &stream.block(position, WIDEST));
        let committed = (position + 1..=position + WIDEST + 1)
            .map(Stream::token)
            .collect::<Vec<_>>();
        // Committed a token at a time, as plain rounds would.
        for token in &committed {
            controller.committed(std::slice::from_ref(token));
        }
        let right = stream.right(position);
        let expected = (1..=WIDEST)
            .map(|depth| f64::from(u8::from(depth <= right)))
            .collect::<Vec<_>>();
        assert_eq!(controller.right(), expected);
    }
}

/// A chained head at 64k context (Gemma 4 E2B with its assistant, measured):
/// every proposal is another head pass, and every verified row another pass
/// of attention over the history.
const CHAINED: Costs = Costs {
    draft: |generated| 1.15 * generated as f64,
    verify: |rows| match rows {
        1 => 15.2,
        2 => 19.2,
        3..=4 => 27.1,
        5..=8 => 43.0,
        _ => 75.0,
    },
};

#[test]
fn a_chained_head_widens_again_when_the_text_turns_predictable() {
    // New prose, then a passage repeated: as one 256-token request ends.
    let stream = Stream::of(&[(400, UNPREDICTABLE), (400, PERFECT)]);
    let run = adaptive(&stream, CHAINED, &warmed(&stream, CHAINED));
    let narrow = run
        .within(400..800)
        .take_while(|(choice, _)| choice.verified < 3)
        .count();
    let rounds = run
        .within(400..800)
        .map(|(choice, committed)| {
            format!("{}/{}/{}", choice.generated, choice.verified, committed)
        })
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        narrow <= 16,
        "{narrow} rounds of perfect text before three proposals are verified: {rounds}"
    );
}

/// A CPU: the drafter's launch costs little next to the target's, whose
/// rows each cost a share of a plain decode, so drafting pays only on
/// predictable text.
const CPU: Costs = Costs {
    draft: |generated| if generated == 0 { 0.0 } else { 20.0 },
    verify: |rows| 50.0 + 10.0 * (rows - 1) as f64,
};

/// Drafting that measured slower stops, but never for good: the time set
/// aside for looks carries from request to request, so a load of short
/// requests, none long enough to earn a look by itself, still looks, and
/// drafts again once the text turns predictable.
#[test]
fn looks_carry_across_short_requests_so_drafting_that_stopped_can_resume() {
    let load = warmed(&Stream::of(&[(400, UNPREDICTABLE)]), CPU);
    let off = adaptive(&Stream::of(&[(30, UNPREDICTABLE)]), CPU, &load);
    assert!(
        off.rounds
            .iter()
            .filter(|(choice, _)| choice.generated > 0)
            .count()
            * 4
            <= off.rounds.len(),
        "drafting that doesn't pay is mostly off: {}",
        off.rounds()
    );
    let predictable = Stream::of(&[(30, PREDICTABLE)]);
    let resumed = (0..20)
        .map(|_| adaptive(&predictable, CPU, &load))
        .position(|run| {
            run.rounds
                .iter()
                .any(|(choice, _)| !choice.look && choice.verified > 0)
        });
    assert!(
        resumed.is_some(),
        "short requests over predictable text draft again"
    );
}

/// The CPU while another process holds most of it: every launch takes
/// longer, the drafter's most.
const CONTENDED_CPU: Costs = Costs {
    draft: |generated| if generated == 0 { 0.0 } else { 90.0 },
    verify: |rows| 110.0 + 40.0 * (rows - 1) as f64,
};

/// QA (linux-arch, round 3c): on a CPU shared with another long run,
/// drafting stopped after two requests and never came back. Launches that
/// stop being taken keep the cost they measured, so once the machine is free
/// again the plain decode got faster while the drafted rounds stayed priced
/// as on the busy machine, and no look was ever worth taking. Drafting that
/// stopped resumes when the text turns predictable on the freed machine.
#[test]
fn drafting_stopped_on_a_busy_cpu_resumes_once_it_is_free() {
    let load = warmed(&Stream::of(&[(400, UNPREDICTABLE)]), CONTENDED_CPU);
    let off = adaptive(&Stream::of(&[(60, UNPREDICTABLE)]), CONTENDED_CPU, &load);
    assert!(
        off.rounds
            .iter()
            .filter(|(choice, _)| choice.generated > 0)
            .count()
            * 4
            <= off.rounds.len(),
        "drafting that doesn't pay is mostly off: {}",
        off.rounds()
    );
    let predictable = Stream::of(&[(30, PREDICTABLE)]);
    let resumed = (0..20)
        .map(|_| adaptive(&predictable, CPU, &load))
        .position(|run| {
            run.rounds
                .iter()
                .any(|(choice, _)| !choice.look && choice.verified > 0)
        });
    assert!(
        resumed.is_some(),
        "short requests over predictable text on the freed CPU draft again"
    );
}

/// A mixture-of-experts target with about a billion active parameters on a
/// GPU (LFM2.5 8B-A1B with its DSpark drafter on an M4 Max, measured): every
/// verified row reaches more experts, so verification costs much more than
/// a plain decode, and a drafter launch costs half of one.
const SPARSE: Costs = Costs {
    draft: |generated| match generated {
        0 => 0.0,
        1 => 2.3,
        2..=3 => 2.6,
        4..=7 => 4.4,
        _ => 6.5,
    },
    verify: |rows| match rows {
        1 => 4.7,
        2 => 6.3,
        3..=4 => 9.0,
        5..=8 => 16.5,
        _ => 23.5,
    },
};

/// Its acceptance on reasoning prose (measured).
const PROSE: [f64; WIDEST] = [
    0.55, 0.30, 0.17, 0.13, 0.10, 0.07, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// QA (mac-r4-12): on a loaded M4 Max the drafter made decoding 25 % slower
/// than plain at 45 % acceptance. In a context class where only plain
/// decodes had been measured, one was held up by another process (traced:
/// 12.3 ms against 4.8 ms; 15 ms plain costs stuck for whole requests in
/// another run, here five times its cost); the verification
/// launches, first measured there afterwards, were not, so drafting looked
/// faster, the plain launch was no longer taken, and its cost stayed raised
/// for the rest of the request. A launch held up once moves no launch's
/// cost relative to another's, so drafting that does not pay stays off.
#[test]
fn a_launch_held_up_once_does_not_make_drafting_that_loses_look_faster() {
    let stream = Stream::of(&[(3_000, PROSE)]);
    let (width, plain) = fastest_fixed(&stream, SPARSE);
    assert_eq!(width, 0, "plain decoding is the fastest");
    // A load that measured its launches at a shorter context, then decodes
    // at a new class where a plain decode is held up once.
    let load = warmed(&Stream::of(&[(500, PROSE)]), SPARSE);
    let class = (4 * CONTEXT).next_power_of_two();
    let context = class - 50;
    let held_up = std::cell::Cell::new(false);
    let run = adaptive_on(&stream, SPARSE, &load, context, |position| {
        if position >= class + 20 && !held_up.replace(true) {
            5.0
        } else {
            1.0
        }
    });
    assert!(
        run.milliseconds <= plain * 1.02,
        "adaptive {:.0} ms, plain {plain:.0} ms ({:+.1}%): {}",
        run.milliseconds,
        (run.milliseconds / plain - 1.0) * 100.0,
        run.rounds()
    );
}

/// QA (mac-r4, Qwen3.5 4B with its DFlash drafter on a loaded M4 Max): a
/// width's drafter launch in a context class was held up by another process
/// (100 ms against 7 ms) on the only time it was measured there, and that
/// priced it for good: a width is measured again only when it is taken, and
/// a price that high is never taken, so drafting that pays stopped for the
/// rest of the load. A launch measured fewer times than its price needs costs
/// no more than the same launch proven over a shorter history, scaled by the
/// longer history, so it is taken, and measured, again.
#[test]
fn a_width_held_up_on_its_first_launch_is_taken_again() {
    let stream = Stream::of(&[(3_000, PREDICTABLE)]);
    assert_eq!(fastest_fixed(&stream, SHORT).0, 7, "drafting pays");
    let warm = warmed(&Stream::of(&[(500, PREDICTABLE)]), SHORT);
    let class = (4 * CONTEXT).next_power_of_two();
    for held in 0..12 {
        let load = Arc::new(DraftLoad {
            costs: std::sync::Mutex::new(warm.costs.lock().unwrap().clone()),
            rounds: std::sync::Mutex::new(*warm.rounds.lock().unwrap()),
            depths: std::sync::Mutex::new(warm.depths.lock().unwrap().clone()),
            look_run: std::sync::Mutex::new(*warm.look_run.lock().unwrap()),
            look_seconds: std::sync::Mutex::new(*warm.look_seconds.lock().unwrap()),
        });
        let round = std::cell::Cell::new(0usize);
        let run = adaptive_on(&stream, SHORT, &load, class - 50, |position| {
            if position < class {
                return 1.0;
            }
            round.set(round.get() + 1);
            if round.get() == held + 1 { 15.0 } else { 1.0 }
        });
        within_of_fixed(&run, &stream, SHORT, 7, 0.05);
    }
}

/// A GPU on which a round that drafts nothing is a decode pipelined behind
/// its predecessor, so changing between drafting and not costs time beyond
/// either round's launches.
#[derive(Clone, Copy)]
struct Pipelined {
    costs: Costs,
    resume: f64,
    discard: f64,
    catch_up: fn(usize) -> f64,
}

/// How the controller sees a round that changes between drafting and not:
/// whole (the decode it discarded in its drafter's launch, the drafter's
/// catch-up as an entry of the round, the restarted decode in its target's
/// launch), or as round 4e's controller saw it, with no switch in any
/// launch (its executor charged the discarded decode and the catch-up to no
/// launch, and the restarted decode to a plain decode the median dropped).
#[derive(Clone, Copy, PartialEq)]
enum Reported {
    Whole,
    WithoutSwitch,
}

/// One request over `stream` on `machine`, drafting at most `widest`, under
/// the controller. The run's time is what the rounds took: a one-row target
/// launch is a decode pipelined behind its predecessor, a plain round after
/// a drafting one adds `resume` (nothing was queued behind it), and a
/// drafting round after a plain one adds `discard` (the decode queued behind
/// that round finishes before its drafter runs) and the drafter's catch-up
/// of the pairs held since.
fn pipelined(
    stream: &Stream,
    machine: Pipelined,
    widest: usize,
    load: &Arc<DraftLoad>,
    reported: Reported,
) -> Run {
    let mut controller = DraftController::new(widest, load.clone());
    let mut run = Run::default();
    let mut position = 0;
    let mut drafted_before = None;
    let mut held = 0;
    let ms = |milliseconds: f64| Duration::from_secs_f64(milliseconds / 1000.0);
    while position < stream.len() {
        let limit = (stream.len() - position - 1).min(widest);
        let generated = controller.generate(limit, CONTEXT + position);
        let mut verified = 0;
        if generated > 0 {
            let mut draft = (machine.costs.draft)(generated);
            let mut entry = 0.0;
            if drafted_before == Some(false) {
                draft += machine.discard;
                entry = (machine.catch_up)(held);
            }
            run.milliseconds += draft + entry;
            match reported {
                Reported::Whole => {
                    if entry > 0.0 {
                        controller.timed(TimedLaunch::Entry, Some(ms(entry)), 1, held);
                    }
                    controller.timed(TimedLaunch::Draft, Some(ms(draft)), 1, generated + 1);
                }
                Reported::WithoutSwitch => {
                    let launch = (machine.costs.draft)(generated);
                    controller.timed(TimedLaunch::Draft, Some(ms(launch)), 1, generated + 1);
                }
            }
            held = 0;
            let block = stream.block(position, generated);
            controller.drafted(generated, &block);
            verified = controller.verify(block.len(), limit);
        } else {
            held = (held + 1).min(64);
        }
        let mut target = (machine.costs.verify)(verified + 1);
        if generated == 0 && drafted_before == Some(true) {
            target += machine.resume;
        }
        run.milliseconds += target;
        if reported == Reported::WithoutSwitch {
            target = (machine.costs.verify)(verified + 1);
        }
        controller.timed(TimedLaunch::Target, Some(ms(target)), 1, verified + 1);
        drafted_before = Some(generated > 0);
        let accepted = stream.right(position).min(verified);
        let committed = (position + 1..=position + accepted + 1)
            .map(Stream::token)
            .collect::<Vec<_>>();
        controller.committed(&committed);
        run.rounds.push((controller.choice(), accepted + 1));
        position += accepted + 1;
    }
    run
}

/// One request over `stream` on `machine` at a fixed width, drafting its
/// whole width or nothing as a fixed budget does. It switches only at the
/// request's last tokens, which this leaves out.
fn pipelined_fixed(stream: &Stream, machine: Pipelined, width: usize) -> f64 {
    let mut milliseconds = 0.0;
    let mut position = 0;
    while position < stream.len() {
        let limit = stream.len() - position - 1;
        let width = if width <= limit { width } else { 0 };
        milliseconds += (machine.costs.draft)(width) + (machine.costs.verify)(width + 1);
        position += stream.right(position).min(width) + 1;
    }
    milliseconds
}

/// Qwen3.5 2B's chained heads on a Radeon PRO V710 under Vulkan (round 4e,
/// traced): each proposal is another head pass, a verification costs a
/// third more than the plain decode it replaces, and that decode is
/// pipelined. Switching costs as an M4 Max traced them.
const V710_HEADS: Pipelined = Pipelined {
    costs: Costs {
        draft: |generated| match generated {
            0 => 0.0,
            1 => 1.8,
            2..=3 => 4.7,
            4..=7 => 10.9,
            _ => 12.4,
        },
        verify: |rows| match rows {
            1 => 6.1,
            2 => 8.4,
            3..=4 => 10.9,
            5..=8 => 11.2,
            _ => 16.0,
        },
    },
    resume: 1.7,
    discard: 4.5,
    catch_up: |held| {
        if held > 1 {
            0.6 + 0.02 * held as f64
        } else {
            0.0
        }
    },
};

/// The heads' acceptance on prose and on code (measured at width 1 on an M4
/// Max, deeper depths from the controller's scores there).
const HEADS_PROSE: [f64; WIDEST] = [
    0.64, 0.40, 0.24, 0.15, 0.09, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];
const HEADS_CODE: [f64; WIDEST] = [
    0.82, 0.64, 0.48, 0.36, 0.27, 0.20, 0.15, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];
const HEADS: usize = 8;

/// Many texts of `segments`, as a benchmark sends them.
fn texts(segments: &[(usize, [f64; WIDEST])]) -> Vec<Stream> {
    (0..24u64)
        .map(|seed| Stream {
            segments: segments.to_vec(),
            seed: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x51,
        })
        .collect()
}

/// The milliseconds of `streams` on `machine`: each a request on a load of
/// its own after a warming request over the same kind of text under the
/// controller (`Some`), or at a fixed `width` (`None`).
fn served(streams: &[Stream], machine: Pipelined, reported: Option<Reported>, width: usize) -> f64 {
    streams
        .iter()
        .map(|stream| match reported {
            None => pipelined_fixed(stream, machine, width),
            Some(reported) => {
                let load = Arc::new(DraftLoad::default());
                let warming = Stream {
                    seed: !stream.seed,
                    ..stream.clone()
                };
                pipelined(&warming, machine, HEADS, &load, reported);
                pipelined(stream, machine, HEADS, &load, reported).milliseconds
            }
        })
        .sum::<f64>()
}

/// The fastest fixed width over `streams` on `machine`, with its time.
fn fastest_served(streams: &[Stream], machine: Pipelined) -> (usize, f64) {
    [0, 1, 3, 7, 8]
        .into_iter()
        .map(|width| (width, served(streams, machine, None, width)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// QA (linux-vulkan-13): with Qwen3.5 2B's heads on Vulkan, decoding with
/// them came out 3-5 % slower than plain on prose. The heads barely pay
/// there (a verification costs a third more than the pipelined decode it
/// replaces), and every change between drafting and not cost a discarded
/// decode and the drafter's catch-up, which no launch was charged with: a
/// round entering drafting looked as cheap as one continuing it, so the
/// controller kept changing on noise in its acceptance and paid for every
/// change. Charged to the round that switches, and priced into a round that
/// would, a switch is taken only where it pays.
#[test]
fn switching_between_drafting_and_plain_decoding_is_priced() {
    // Many texts, each a short request after a warming one, as a benchmark
    // sends them.
    let total = |streams: &[Stream], reported: Option<Reported>, width: usize| {
        served(streams, V710_HEADS, reported, width)
    };
    let texts = |segments: Vec<(usize, [f64; WIDEST])>| texts(&segments);
    let mixed = vec![
        (100, HEADS_CODE),
        (100, HEADS_PROSE),
        (100, HEADS_CODE),
        (100, HEADS_PROSE),
    ];
    for (segments, name, width) in [
        (vec![(400, HEADS_PROSE)], "prose", 0),
        (vec![(400, HEADS_CODE)], "code", 3),
        (mixed, "mixed", 1),
    ] {
        let streams = texts(segments);
        let (fastest, best) = fastest_served(&streams, V710_HEADS);
        assert_eq!(fastest, width, "the fastest fixed width on {name}");
        let plain = total(&streams, None, 0);
        let priced = total(&streams, Some(Reported::Whole), HEADS);
        let unpriced = total(&streams, Some(Reported::WithoutSwitch), HEADS);
        let report = format!(
            "{name}: fixed {width} {best:.0} ms, plain {plain:.0} ms, priced {priced:.0} ms ({:+.1}%), as round 4e saw it {unpriced:.0} ms ({:+.1}%)",
            (priced / best - 1.0) * 100.0,
            (unpriced / best - 1.0) * 100.0,
        );
        eprintln!("{report}");
        assert!(priced < unpriced, "{report}");
        assert!(
            name != "code" || priced < plain,
            "drafting that pays goes on: {report}"
        );
        assert!(
            width != 0 || priced <= best * 1.015,
            "within the looks of plain decoding: {report}"
        );
    }
}

/// A switch is measured apart from the launches of the round that makes it:
/// what entering drafting costs (the discarded decode and the catch-up) and
/// what leaving it costs (the restarted decode), while the launches keep
/// the prices of rounds that do not switch.
#[test]
fn a_switch_is_measured_apart_from_its_rounds_launches() {
    let load = Arc::new(DraftLoad::default());
    let segments = (0..6)
        .flat_map(|_| [(150, HEADS_CODE), (150, HEADS_PROSE)])
        .collect::<Vec<_>>();
    let run = pipelined(
        &Stream::of(&segments),
        V710_HEADS,
        HEADS,
        &load,
        Reported::Whole,
    );
    let switches = run
        .rounds
        .windows(2)
        .filter(|pair| (pair[0].0.generated > 0) != (pair[1].0.generated > 0))
        .count();
    assert!(switches >= 4, "the run switches: {}", run.rounds());
    let costs = load.costs.lock().unwrap();
    let measured = &costs[&((CONTEXT + 1).next_power_of_two(), 1)];
    let ms = |durations: &super::Durations| durations.cost() * 1000.0;
    let enter = ms(&measured.enter);
    let catch_up = (V710_HEADS.catch_up)(64);
    assert!(
        enter >= V710_HEADS.discard * 0.99 && enter <= (V710_HEADS.discard + catch_up) * 1.01,
        "entering costs the discarded decode and the catch-up: {enter} ms"
    );
    assert!((ms(&measured.leave) - V710_HEADS.resume).abs() < 0.01);
    assert!((ms(&measured.verify[&1]) - (V710_HEADS.costs.verify)(1)).abs() < 0.01);
    // A catch-up's length varies with the pairs held, so a drafter launch
    // that pays for one is off by at most that variation. A launch first
    // measured in a switching round, before the switch had a price, keeps
    // that duration until it is measured again, so only launches measured
    // their full share of times are checked.
    for (generated, durations) in measured
        .draft
        .iter()
        .filter(|(_, durations)| durations.proven())
    {
        assert!(
            (ms(durations) - (V710_HEADS.costs.draft)(*generated)).abs() <= catch_up,
            "{generated} proposals: {} ms",
            ms(durations)
        );
    }
}

/// Qwen3.5 2B's chained heads on an M4 under Metal at 4k context (QA r4,
/// traced): each proposal is another head pass of 4.5 ms, a verification
/// costs little more than the plain decode it replaces, and entering
/// drafting waits for the queued decode it discards, about a plain decode.
const M4_HEADS: Pipelined = Pipelined {
    costs: Costs {
        draft: |generated| 4.5 * generated as f64,
        verify: |rows| match rows {
            1 => 14.6,
            2 => 15.5,
            3..=4 => 17.8,
            5..=8 => 18.0,
            _ => 29.7,
        },
    },
    resume: 0.8,
    discard: 13.5,
    catch_up: V710_HEADS.catch_up,
};

/// A stretch the heads get wrong: a quoted table of numbers, say.
const HEADS_WRONG: [f64; WIDEST] = [
    0.15, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// QA r4 (Qwen3.5 2B with its heads on an M4 under Metal): decoding under
/// the controller was 5-15 % slower than at a fixed width of 1. Acceptance is
/// observed only while drafting, so after a stretch of wrong blocks stopped
/// drafting its estimates stayed where the stretch left them (0.14-0.7, the
/// first depth's real acceptance 0.82: rounds right after the plain stretches
/// accepted 0.90), and the load stayed plain until a look, which on this
/// machine costs a discarded decode, came. Every drafting round measured
/// only as it entered drafting also kept the discarded decode in its price.
/// An estimate not scored returns to the load's usual acceptance, so drafting
/// comes back once the stretch has passed.
#[test]
fn drafting_that_pays_comes_back_after_a_stretch_of_wrong_blocks() {
    let streams = texts(&[(100, HEADS_CODE), (30, HEADS_WRONG), (126, HEADS_CODE)]);
    let (width, best) = fastest_served(&streams, M4_HEADS);
    assert!(width > 0, "drafting pays");
    let plain = served(&streams, M4_HEADS, None, 0);
    let adaptive = served(&streams, M4_HEADS, Some(Reported::Whole), HEADS);
    let report = format!(
        "fixed {width} {best:.0} ms, plain {plain:.0} ms, adaptive {adaptive:.0} ms ({:+.1}%)",
        (adaptive / best - 1.0) * 100.0
    );
    eprintln!("{report}");
    assert!(adaptive <= best * 1.05, "{report}");
}

/// Text whose acceptance does not change still brings runs of wrong blocks:
/// two in a row at 0.82 come every thirty rounds. Over a few blocks the
/// current text's mean falls below what drafting needs, but those departures
/// are what the binomial noise of so few blocks gives, so the estimate stays
/// with the load's usual acceptance and drafting that pays goes on.
#[test]
fn runs_of_wrong_blocks_on_steady_text_do_not_stop_drafting() {
    for machine in [V710_HEADS, M4_HEADS] {
        let streams = texts(&[(400, HEADS_CODE)]);
        let (width, best) = fastest_served(&streams, machine);
        let adaptive = served(&streams, machine, Some(Reported::Whole), HEADS);
        let report = format!(
            "fixed {width} {best:.0} ms, adaptive {adaptive:.0} ms ({:+.1}%)",
            (adaptive / best - 1.0) * 100.0
        );
        eprintln!("{report}");
        assert!(adaptive <= best * 1.03, "{report}");
    }
}
