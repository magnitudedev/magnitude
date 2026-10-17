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
    let mut controller = DraftController::new(WIDEST, load.clone());
    let mut run = Run::default();
    let mut position = 0;
    while position < stream.len() {
        let limit = (stream.len() - position - 1).min(WIDEST);
        let generated = controller.generate(limit, context + position);
        let mut verified = 0;
        if generated > 0 {
            let seconds = (costs.draft)(generated) / 1000.0;
            run.milliseconds += seconds * 1000.0;
            controller.timed(
                TimedLaunch::Draft,
                Duration::from_secs_f64(seconds),
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
            Duration::from_secs_f64(seconds),
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
            Duration::from_secs_f64(seconds),
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
        .map(|(choice, committed)| format!("{}/{}/{}", choice.generated, choice.verified, committed))
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
