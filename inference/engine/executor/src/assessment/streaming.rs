//! A streamed prompt group's stages ([`Stages`]), which its size and merge
//! sets are formed by, and the places a group's chunks are worth against the
//! bank's slots ([`GroupReads::grant`]). A layer's stage, when its cold
//! experts are in the GPU's memory, is priced by the one cost function
//! ([`ExpertLines::staged`], [`super::cost`]), as decode is.

use super::cost::{ExpertLines, Miss};
use super::estimate::ExpertPass;
use super::streaming_costs::ReadLine;
use crate::experts::MissCurve;
use std::sync::Arc;

/// One streamed layer of a prompt group, as [`Stages`] takes it: the
/// compute of one of the group's steps there, measured on the run, and the
/// cold experts the group moves there, each once: those read from the file
/// (`filed`) and those the host tier holds (`tiered`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupLayerTerms {
    pub compute_seconds: f64,
    pub filed: u64,
    pub tiered: u64,
}

/// A streamed prompt group's stages: its steps run layer-major. At each
/// layer every step's front publishes its request, so the layer's file
/// experts are read and landed as the layer before's backs run; the fill
/// then moves each cold expert into the GPU's memory behind the layer's
/// wait. The layer's stage, `stage_l`, is when its cold experts are there,
/// from its requests ([`ExpertLines::staged`], the file's along `line`). A
/// back computes `c_l` a step; a merge set's backs run as one pass, which
/// shortens that compute. The stage of layer `l + 1` is covered by whatever
/// runs between its first request and the end of layer `l`'s compute, which
/// a back run alone brings forward at the price of its own pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stages<'a> {
    pub layers: &'a [GroupLayerTerms],
    pub lines: &'a [ExpertLines],
    pub line: ReadLine,
    pub notice: Notice,
}

/// When the host reads a step's request, which starts its reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Notice {
    /// As the device writes it: kernels wait in place.
    Written,
    /// At the next park, a submission boundary: a merged group's first
    /// falls after all its fronts, so they cover none of the stage.
    Parked,
}

impl Stages<'_> {
    /// `stage_l`: its tier experts whole at once, its file experts after
    /// their one read and landing, each moved.
    fn stage(&self, layer: usize) -> f64 {
        let (terms, lines) = (&self.layers[layer], &self.lines[layer]);
        let filed = match terms.filed {
            0 => 0.,
            count => lines.filed(&self.line, count),
        };
        // A stage is its moves', which no row count enters.
        let misses = std::iter::repeat_n(0., terms.tiered as usize)
            .chain(std::iter::repeat_n(filed, terms.filed as usize))
            .map(|ready_seconds| Miss { rows: 1., ready_seconds })
            .collect::<Vec<_>>();
        lines.staged(&misses)
    }

    /// What a group of `steps` steps leaves of its stages uncovered: at each
    /// layer what its steps' compute does not cover of its stage, `max(0,
    /// stage_l − g · c_l)` (what [`Self::size`] makes none).
    pub fn exposed(&self, steps: usize) -> f64 {
        (0..self.layers.len())
            .map(|layer| (self.stage(layer) - steps as f64 * self.layers[layer].compute_seconds).max(0.))
            .sum()
    }

    /// The steps a group runs while decode waits, at most `capacity`: the
    /// fewest whose compute covers each layer's stage, `g · c_l ≥ stage_l`.
    pub fn size(&self, capacity: usize) -> usize {
        (0..self.layers.len())
            .map(|layer| (self.stage(layer) / self.layers[layer].compute_seconds).ceil() as usize)
            .fold(1, usize::max)
            .clamp(1, capacity)
    }

    /// Each layer's merge set in a group of `steps` steps, the steps it
    /// ends with: the whole group, or all but a lone first back, whichever
    /// takes fewer seconds, whole on a tie. Of `n` steps at a layer, each
    /// `f` of compute past its pass (`c_l` less the last group's passes,
    /// which `unions` price by `pass`, scaled to fit it as
    /// [`GroupReads::at`] scales them), a set of `s` computes `C(s)` and
    /// has the next layer's first request read after its first pass and
    /// front, `first(s)`, or after all of `C(n)` where a whole group's
    /// requests wait for a park ([`Notice`]), so it takes `max(C(s),
    /// first(s) + stage_{l+1})`: all the compute after that covers the
    /// stage. A lone first
    /// back brings that request forward by the merged pass, at the price
    /// of a pass more. A second lone back would add compute and bring
    /// nothing forward, as passes are subadditive. So fronts that cover the
    /// stage leave a group merged, as does the last layer, followed by no
    /// stage, and a group with nothing cold.
    pub fn merge_sets(&self, steps: usize, unions: &[&GroupUnion], pass: &ExpertPass) -> Vec<usize> {
        (0..self.layers.len())
            .map(|layer| {
                let stage = if layer + 1 < self.layers.len() { self.stage(layer + 1) } else { 0. };
                let union = unions[layer];
                let group = steps as f64 * self.layers[layer].compute_seconds;
                let measured = union.passes(pass, layer, union.steps, union.merged) * steps as f64 / union.steps as f64;
                let fit = (group / measured).min(1.);
                let front = (group - fit * measured) / steps as f64;
                let compute = |set: usize| steps as f64 * front + fit * union.passes(pass, layer, steps, set);
                let alone = fit * union.pass(pass, layer, 1) + front;
                let whole = match self.notice {
                    Notice::Written => fit * union.pass(pass, layer, steps) + front,
                    Notice::Parked => compute(steps),
                };
                match steps > 1 && compute(steps - 1).max(alone + stage) < compute(steps).max(whole + stage) {
                    true => steps - 1,
                    false => steps,
                }
            })
            .collect()
    }
}

/// A streamed layer's union in the last prompt group the ranking's stack
/// was fed (`Stack::read_group`): the group's steps, the steps its pass
/// merged there, and each expert it read, with its depth in slots as it was
/// read and the choices the group's rows routed to it.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupUnion {
    pub steps: usize,
    pub merged: usize,
    pub experts: Vec<(usize, u32)>,
}

impl GroupUnion {
    /// The chance a group of `steps` steps reads an expert the group's rows
    /// routed `choices` choices to: its rows a share `r = steps /
    /// self.steps` of the group's, each of those choices among them with
    /// that chance, `1 − (1 − r)^c`. A group at least as long reads every
    /// expert the group read: the counts name no other.
    fn chance(&self, steps: usize, choices: u32) -> f64 {
        let share = steps as f64 / self.steps as f64;
        match share < 1. {
            true => 1. - (1. - share).powi(choices as i32),
            false => 1.,
        }
    }

    /// The experts a group of `steps` steps is expected to read of those the
    /// group read at least `depth` slots deep, and how many of those there
    /// were.
    fn deeper(&self, steps: usize, depth: u64) -> (f64, usize) {
        self.experts
            .iter()
            .filter(|&&(at, _)| at as u64 >= depth)
            .fold((0., 0), |(expected, read), &(_, choices)| (expected + self.chance(steps, choices), read + 1))
    }

    /// The choices a group of `steps` steps routes: the group's a step.
    fn choices(&self, steps: usize) -> f64 {
        let routed = self.experts.iter().map(|&(_, choices)| u64::from(choices)).sum::<u64>();
        routed as f64 * steps as f64 / self.steps as f64
    }

    /// One pass over `steps` steps' rows at streamed layer `layer`.
    fn pass(&self, pass: &ExpertPass, layer: usize, steps: usize) -> f64 {
        pass.seconds(layer, self.deeper(steps, 0).0, self.choices(steps))
    }

    /// What a group of `steps` steps merging `merged` of them costs at
    /// streamed layer `layer` in passes (`pass`): its steps alone a pass of
    /// one step's experts each, its merge set one pass of theirs.
    fn passes(&self, pass: &ExpertPass, layer: usize, steps: usize, merged: usize) -> f64 {
        (steps - merged) as f64 * self.pass(pass, layer, 1) + self.pass(pass, layer, merged)
    }
}

/// What the run measured of its prompt groups, which weighs their places
/// against the bank's slots ([`Self::grant`]): each streamed layer's terms
/// at the bank's slots now (`c_l`, and the cold experts the next group moves
/// there) and its last group's union, its lines, the group-read curve (each
/// ended group's union by its depth in the ranking's stack, `steps` the
/// groups, at least one), the run's read line, and whether groups pull the
/// experts the host tier holds (`pulls`): a slot the bank
/// gives up then leaves its expert in the tier, to be pulled, not read. A
/// group's moves and its passes' compute follow its size by its last
/// group's union ([`GroupUnion`]).
#[derive(Clone, Debug, PartialEq)]
pub struct GroupReads {
    layers: Vec<GroupLayerTerms>,
    unions: Vec<GroupUnion>,
    lines: Arc<[ExpertLines]>,
    curve: MissCurve,
    line: ReadLine,
    pulls: bool,
    notice: Notice,
}

/// The bank's slots now, its floor and one slot's bytes, and its host
/// tier's slots, which a grant leaves as they are (inclusive of the bank's:
/// it holds the stack's top `host`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankSlots {
    pub now: u64,
    pub floor: u64,
    pub bytes: u64,
    pub host: u64,
}

/// A prompt group's places as its grant weighs them: the chunks in hand,
/// the places held, the bytes a class of each count of places from those
/// held to the chunks holds (`bytes[p − held]`, increasing), what a grant's
/// bytes stay below now, and the seconds a group may run (the wait bound,
/// `MemoryPace::wait_bound`; none before the host was seen taking memory).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Places<'a> {
    pub chunks: usize,
    pub held: usize,
    pub bytes: &'a [u64],
    pub room: u64,
    pub wait: Option<f64>,
}

impl Places<'_> {
    /// The most places whose class stays below `room` now, those held
    /// always, and the room that class leaves (all of it at those held):
    /// the one growth-room test of a group's places. A count `outlasts`
    /// says runs longer than a group may is not admitted, nor any after it.
    pub fn admitted(&self, outlasts: impl Fn(usize) -> bool) -> (usize, u64) {
        let mut admitted = (self.held, self.room);
        for (places, &bytes) in (self.held..).zip(self.bytes).skip(1) {
            match self.room.checked_sub(bytes).filter(|&left| left > 0 && !outlasts(places)) {
                Some(left) => admitted = (places, left),
                None => break,
            }
        }
        admitted
    }
}

/// Decode while the prompt runs: the decode seconds owed each second the
/// prompt runs (the scheduler's share), and the decode step on a bank of a
/// count of slots.
pub struct DecodeWaiting<'a, E> {
    pub owed: f64,
    pub step: &'a mut dyn FnMut(u64) -> Result<f64, E>,
}

/// What a grant decided: the places the chunks in hand are worth, and the
/// slots the bank keeps for them (fewer than now when places win memory
/// from slots).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceGrant {
    pub places: usize,
    pub slots: u64,
}

impl GroupReads {
    /// The measured terms and each layer's last group union; none before a
    /// group's reads were counted.
    pub fn new(
        layers: Vec<GroupLayerTerms>,
        unions: Vec<GroupUnion>,
        lines: Arc<[ExpertLines]>,
        curve: MissCurve,
        line: ReadLine,
        pulls: bool,
        notice: Notice,
    ) -> Option<Self> {
        (curve.steps > 0).then_some(Self { layers, unions, lines, curve, line, pulls, notice })
    }

    /// The stages of `layers` on this load's lines.
    fn stages<'a>(&'a self, layers: &'a [GroupLayerTerms]) -> Stages<'a> {
        Stages { layers, lines: &self.lines, line: self.line, notice: self.notice }
    }

    /// Each layer's terms for a group of `steps` steps on a bank of `slots`
    /// slots beside a host tier of `host`, the bank holding `now`. Where
    /// groups pull from the tier, it holds the union's experts from `slots`
    /// to `max(slots, host)` deep, and the file those deeper. Its file
    /// experts: the last group's moved by what the curve says the depths
    /// between hold of a group's union, per group counted, then scaled to
    /// the group's size by what it reads of the experts the last group read
    /// at least that deep there (unscaled where it read none that deep); its
    /// tier experts alike, the last group's pulls moved by the curve's
    /// experts between the bank and the file. Its step's compute: the
    /// measured step's, less its share of the last group's passes and plus
    /// its share of this group's (`pass`), the group merging what the merge
    /// rule merges ([`Stages::merge_sets`]) at the measured compute, both
    /// shares scaled down where the last group's exceeds the measured step
    /// that holds it, so it is never negative; the measured step's alone
    /// without the executor's costs.
    fn at(&self, now: u64, slots: u64, host: u64, steps: usize, pass: Option<&ExpertPass>) -> Vec<GroupLayerTerms> {
        let groups = self.curve.steps as f64;
        // The depth from which the file holds an expert.
        let filed = |slots: u64| match self.pulls {
            true => slots.max(host),
            false => slots,
        };
        let layers = self.layers.iter().zip(&self.curve.layers).zip(&self.unions);
        let read = layers
            .map(|((terms, curve), union)| {
                let misses = |slots: u64| curve.at(usize::try_from(slots).expect("a bank's slots fit the host")).1 as f64;
                let scale = |depth: u64| match union.deeper(steps, depth) {
                    (_, 0) => 1.,
                    (expected, read) => expected / read as f64,
                };
                let moved = (misses(filed(slots)) - misses(filed(now))) / groups;
                // The experts between the bank and the file, which the tier
                // holds, per group.
                let tiered = |slots: u64| (misses(slots) - misses(filed(slots))) / groups;
                let pulled = match self.pulls {
                    true => (terms.tiered as f64 + tiered(slots) - tiered(now)).max(0.) * scale(slots),
                    false => 0.,
                };
                GroupLayerTerms {
                    filed: ((terms.filed as f64 + moved).max(0.) * scale(filed(slots))).round() as u64,
                    tiered: pulled.round() as u64,
                    ..*terms
                }
            })
            .collect::<Vec<_>>();
        let Some(pass) = pass else {
            return read;
        };
        let merges = self.stages(&read).merge_sets(steps, &self.unions.iter().collect::<Vec<_>>(), pass);
        read.iter()
            .zip(&self.unions)
            .zip(merges)
            .enumerate()
            .map(|(layer, ((terms, union), merged))| {
                let measured = union.passes(pass, layer, union.steps, union.merged) / union.steps as f64;
                let passes = union.passes(pass, layer, steps, merged) / steps as f64;
                // The measured step holds the last group's passes: priced
                // above it, both pricings shrink by the share that fits it.
                let fit = (terms.compute_seconds / measured).min(1.);
                GroupLayerTerms {
                    compute_seconds: terms.compute_seconds + fit * (passes - measured),
                    ..*terms
                }
            })
            .collect()
    }

    /// The seconds `chunks` chunks take in groups of `places` on a bank of
    /// `slots` slots holding `now` beside a host tier of `host`: each
    /// group's compute and what it leaves of its stages uncovered
    /// ([`Stages::exposed`]), at its size.
    fn seconds(&self, now: u64, slots: u64, host: u64, chunks: usize, places: usize, pass: Option<&ExpertPass>) -> f64 {
        let group = |steps: usize| {
            let layers = self.at(now, slots, host, steps, pass);
            let compute = layers.iter().map(|layer| layer.compute_seconds).sum::<f64>() * steps as f64;
            compute + self.stages(&layers).exposed(steps)
        };
        let rest = match chunks % places {
            0 => 0.,
            rest => group(rest),
        };
        (chunks / places) as f64 * group(places) + rest
    }

    /// The places of the chunks in hand, by value per byte: the first group
    /// holds the most places `room` admits now: more places never cost it
    /// more, and free memory is what the bank's own growth left as not
    /// paying (the bank's growth takes the rest). Beyond them, each
    /// further count of places runs the rest of the chunks once the bank has
    /// given up the slots their bytes need past what that first grant leaves:
    /// worth it when the reads its fewer groups save exceed what the slots
    /// given up cost the prompt's groups (the group-read curve: an expert a
    /// slot gave up is read from the file, or pulled from the host tier
    /// where groups pull from it and it holds the expert) and the
    /// decode running while they run (the miss record's step at fewer
    /// slots, for the decode seconds owed). Each count's groups are priced
    /// at their size, their passes by `pass` where the executor's costs
    /// are measured ([`Self::at`]). The count of least seconds wins, fewer
    /// places on a tie; never below the bank's floor.
    pub fn grant<E>(
        &self,
        slots: BankSlots,
        places: Places<'_>,
        pass: Option<&ExpertPass>,
        mut decode: Option<DecodeWaiting<'_, E>>,
    ) -> Result<PlaceGrant, E> {
        let Places { chunks, held, bytes, wait, .. } = places;
        // The wait bound: nothing is released while a group runs, so no
        // group runs longer than pressure may wait for a release, on the
        // slots it runs on; the counts from the first that would are never
        // weighed.
        let outlasts = |kept: u64, places: usize| {
            wait.is_some_and(|wait| self.seconds(slots.now, kept, slots.host, places, places, pass) > wait)
        };
        let (now, left) = places.admitted(|places| outlasts(slots.now, places));
        let first = self.seconds(slots.now, slots.now, slots.host, now.min(chunks), now, pass);
        let alone = PlaceGrant { places: now, slots: slots.now };
        let mut least = (self.seconds(slots.now, slots.now, slots.host, chunks, now, pass), alone);
        let mut base = None;
        for (places, &bytes) in (held..).zip(bytes).skip(now + 1 - held) {
            let cut = bytes.saturating_add(1).saturating_sub(left).div_ceil(slots.bytes);
            let Some(fewer) = slots
                .now
                .checked_sub(cut)
                .filter(|&fewer| fewer >= slots.floor && !outlasts(fewer, places))
            else {
                break;
            };
            let rest = self.seconds(slots.now, fewer, slots.host, chunks.saturating_sub(now), places, pass);
            let slower = match &mut decode {
                Some(DecodeWaiting { owed, step }) => {
                    let base = match base {
                        Some(base) => base,
                        None => *base.insert(step(slots.now)?),
                    };
                    *owed * rest * (step(fewer)? / base - 1.)
                }
                None => 0.,
            };
            let seconds = first + rest + slower;
            if seconds < least.0 {
                least = (seconds, PlaceGrant { places, slots: fewer });
            }
        }
        Ok(least.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference RTX 3090 host's lines over experts of `bytes` stored bytes, one a layer.
    fn lines(bytes: u64, layers: usize) -> Arc<[ExpertLines]> {
        vec![ExpertLines::reference(bytes); layers].into()
    }

    /// A group's size covers every layer's stage. Four file experts of
    /// 1.93 MB: a 4.26 ms read at 2 GB/s and 0.4 ms, then four moves of
    /// 86 µs, 4.604 ms, at 1 ms of compute a step.
    #[test]
    fn a_groups_size_covers_every_stage() {
        let layers = [
            GroupLayerTerms { compute_seconds: 1e-3, filed: 0, tiered: 0 },
            GroupLayerTerms { compute_seconds: 1e-3, filed: 4, tiered: 0 },
            GroupLayerTerms { compute_seconds: 2e-3, filed: 0, tiered: 0 },
        ];
        let lines = lines(ExpertLines::REFERENCE_RECORD, 3);
        let stages = Stages { layers: &layers, lines: &lines, line: READ, notice: Notice::Written };
        assert!((stages.stage(1) - 4.604e-3).abs() < 1e-12, "{}", stages.stage(1));
        assert_eq!(stages.size(8), 5);
    }

    /// A lone first back is taken only where the request it brings forward
    /// saves more than its pass adds. 64 experts priced at 10 µs each and
    /// 1 ms of compute a step of 4, measured merged whole: a whole pass
    /// 640 µs, a lone back's 280 µs, three steps' 600 µs, so 0.84 ms of
    /// each step is its front. Whole takes `max(4, 1.48 + stage)` ms, a
    /// lone first back `max(4.24, 1.12 + stage)`, so it wins past a stage
    /// of 2.76 ms. Nothing cold, the group merges.
    #[test]
    fn a_lone_first_back_is_taken_only_where_it_saves_more_than_its_pass() {
        let bf16 = seismic::Element::bf16();
        let pass = ExpertPass::new(
            crate::assessment::fixtures::costs(),
            vec![(
                vec![crate::assessment::Bytes { format: bf16, bytes: 3_000_000 }],
                vec![crate::assessment::Flops { format: bf16, flops: 1 }],
            ); 2],
        );
        let union = GroupUnion { steps: 4, merged: 4, experts: (0..64).map(|expert| (expert, 2)).collect() };
        let unions = [&union, &union];
        let lines = lines(3_000_000, 2);
        let sets = |filed: u64| {
            let terms = [
                GroupLayerTerms { compute_seconds: 1e-3, filed: 0, tiered: 0 },
                GroupLayerTerms { compute_seconds: 1e-3, filed, tiered: 0 },
            ];
            let stages = Stages { layers: &terms, lines: &lines, line: READ, notice: Notice::Written };
            (stages.stage(1), stages.merge_sets(4, &unions, &pass))
        };
        assert_eq!(sets(0).1, [4, 4]);
        let (small, covered) = sets(1);
        assert!(small < 2.76e-3, "{small}");
        assert_eq!(covered, [4, 4], "fronts that cover the stage leave the group merged");
        let (large, exposed) = sets(8);
        assert!(large > 2.76e-3, "{large}");
        assert_eq!(exposed, [3, 4], "the last layer is followed by no stage");
        // Where requests wait for a park, fronts cover none of the stage:
        // whole takes 4 ms + stage, so a lone back wins from 0.24 ms.
        let terms = [
            GroupLayerTerms { compute_seconds: 1e-3, filed: 0, tiered: 0 },
            GroupLayerTerms { compute_seconds: 1e-3, filed: 1, tiered: 0 },
        ];
        let parked = Stages { layers: &terms, lines: &lines, line: READ, notice: Notice::Parked };
        assert!(parked.stage(1) > 0.24e-3, "{}", parked.stage(1));
        assert_eq!(parked.merge_sets(4, &unions, &pass), [3, 4]);
        let one = GroupLayerTerms { compute_seconds: 1e-3, filed: 8, tiered: 0 };
        let single = [one, one];
        assert_eq!(Stages { layers: &single, lines: &lines, line: READ, notice: Notice::Written }.merge_sets(1, &unions, &pass), [1, 1]);
    }

    /// A layer's tier experts are moved while its file experts are read, and
    /// its file experts once landed: two tier and four file experts are in
    /// the GPU's memory at the read's 4.26 ms and four moves; three tier
    /// experts alone at three moves.
    #[test]
    fn a_stage_moves_tier_experts_while_the_file_is_read() {
        let layers = [
            GroupLayerTerms { compute_seconds: 1e-3, filed: 4, tiered: 2 },
            GroupLayerTerms { compute_seconds: 1e-3, filed: 0, tiered: 3 },
        ];
        let lines = lines(ExpertLines::REFERENCE_RECORD, 2);
        let stages = Stages { layers: &layers, lines: &lines, line: READ, notice: Notice::Written };
        assert!((stages.stage(0) - (4.26e-3 + 4. * 86e-6)).abs() < 1e-12, "{}", stages.stage(0));
        assert!((stages.stage(1) - 3. * 86e-6).abs() < 1e-12, "{}", stages.stage(1));
        assert!((stages.exposed(1) - (4.26e-3 + 4. * 86e-6 - 1e-3)).abs() < 1e-12);
    }

    /// Two layers of 8 experts of 10 MB, 1 ms of compute a step; one group
    /// of `steps` steps counted, whose union at each layer lay at `depths`
    /// (8 a first read), `choices` routed to each.
    fn group(steps: usize, depths: &[usize], choices: u32) -> GroupReads {
        let mut layer = crate::experts::LayerCurve { misses: vec![0; 9], requests: vec![0; 9], choices: 0 };
        depths.iter().for_each(|&depth| layer.misses[depth] += 1);
        layer.requests[*depths.iter().max().unwrap()] += 1;
        let cold = depths.iter().filter(|&&depth| depth >= 8).count() as u64;
        let terms = GroupLayerTerms { compute_seconds: 1e-3, filed: cold, tiered: 0 };
        let union = GroupUnion {
            steps,
            merged: steps,
            experts: depths.iter().map(|&depth| (depth, choices)).collect(),
        };
        let curve = MissCurve { steps: 1, layers: vec![layer; 2] };
        GroupReads::new(vec![terms; 2], vec![union; 2], lines(10_000_000, 2), curve, READ, false, Notice::Written).unwrap()
    }

    /// [`group`] of one step: every group of the grant at least as long.
    fn reads(depths: &[usize]) -> GroupReads {
        group(1, depths, 1)
    }

    /// 8 chunks from one place held; a class of `p` places holds 13 MB a
    /// place past the first; 30 MB of room admits 3 now. A bank of 8 slots
    /// of 20 MB, floor 2; no wait bound.
    fn grant(reads: &GroupReads, decode: Option<(f64, fn(u64) -> f64)>) -> PlaceGrant {
        grant_within(reads, decode, None)
    }

    fn grant_within(
        reads: &GroupReads,
        decode: Option<(f64, fn(u64) -> f64)>,
        wait: Option<f64>,
    ) -> PlaceGrant {
        let bytes = (1..=8).map(|places| (places - 1) * 13_000_000).collect::<Vec<u64>>();
        let slots = BankSlots { now: 8, floor: 2, bytes: 20_000_000, host: 0 };
        let places = Places { chunks: 8, held: 1, bytes: &bytes, room: 30_000_000, wait };
        let mut step = |slots: u64| Ok::<_, ()>(decode.unwrap().1(slots));
        let decode = decode.map(|(owed, _)| DecodeWaiting { owed, step: &mut step });
        reads.grant(slots, places, None, decode).unwrap()
    }

    /// A group reads, of the experts the last group read, each with the
    /// chance one of the choices routed to it falls in its rows: a
    /// group of half the steps reads half an expert one choice chose and
    /// fifteen sixteenths of one four chose; one at least as long reads
    /// them all. Its file experts scale by what it reads of the experts the
    /// last group read cold at the slots, to the nearest expert.
    #[test]
    fn a_groups_reads_follow_its_size() {
        let union = GroupUnion { steps: 4, merged: 4, experts: vec![(8, 1), (8, 4), (0, 2)] };
        assert_eq!(union.deeper(2, 0), (0.5 + 0.9375 + 0.75, 3));
        assert_eq!(union.deeper(2, 8), (0.5 + 0.9375, 2));
        assert_eq!(union.deeper(4, 0), (3., 3));
        assert_eq!(union.deeper(8, 8), (2., 2));
        let cold = group(4, &[8; 7], 1);
        let at = |steps| cold.at(8, 8, 0, steps, None)[0].filed;
        assert_eq!((at(1), at(2), at(4), at(8)), (2, 4, 7, 7));
        // Nothing the last group read lies deeper than a bank of every
        // expert: its moved experts do not scale.
        assert_eq!(group(4, &[0; 7], 1).at(8, 8, 0, 1, None)[0].filed, 0);
    }

    /// Where groups pull from the host tier, a slot the bank gives up
    /// leaves its expert there: the group pulls it rather than read it. Of
    /// a union read at depths 5, 6, 7, 8, 8 (the last two cold), a bank of
    /// 4 slots beside a tier of 8 reads the same two from the file and
    /// pulls the three between; without pulls all five are read.
    #[test]
    fn a_slot_given_up_leaves_its_expert_to_the_tiers_pulls() {
        let reads = group(1, &[5, 6, 7, 8, 8], 1);
        let terms = reads.at(8, 4, 8, 1, None)[0];
        assert_eq!((terms.filed, terms.tiered), (5, 0), "no pulls: the host is the file's");
        let pulling = GroupReads { pulls: true, ..reads };
        let terms = pulling.at(8, 4, 8, 1, None)[0];
        assert_eq!((terms.filed, terms.tiered), (2, 3));
    }

    /// A merged pass reads its steps' union once where alone each step's
    /// pass reads its own: a layer of 64 experts of 3 MB, two choices
    /// each in a measured group of 4 steps merged whole, priced at 300 GB/s
    /// (10 µs an expert). A step's pass alone reads 28 of them (280 µs),
    /// the measured group's 64 over 4 steps (160 µs a step), a group of 8
    /// the same 64 over 8 (80 µs a step): the measured 1 ms a step becomes
    /// 1.12 ms alone and 0.92 ms in a group of 8.
    #[test]
    fn a_groups_compute_follows_its_merged_pass() {
        let bf16 = seismic::Element::bf16();
        let pass = ExpertPass::new(
            crate::assessment::fixtures::costs(),
            vec![(
                vec![crate::assessment::Bytes { format: bf16, bytes: 3_000_000 }],
                vec![crate::assessment::Flops { format: bf16, flops: 1 }],
            )],
        );
        let union = GroupUnion { steps: 4, merged: 4, experts: (0..64).map(|expert| (expert, 2)).collect() };
        let curve = MissCurve {
            steps: 1,
            layers: vec![crate::experts::LayerCurve { misses: vec![0; 65], requests: vec![0; 65], choices: 0 }],
        };
        let terms = GroupLayerTerms { compute_seconds: 1e-3, filed: 0, tiered: 0 };
        let reads = GroupReads::new(vec![terms], vec![union], lines(3_000_000, 1), curve, READ, false, Notice::Written).unwrap();
        let compute = |steps| reads.at(64, 64, 0, steps, Some(&pass))[0].compute_seconds;
        let close = |actual: f64, expected: f64| assert!((actual - expected).abs() < 1e-9, "{actual} ≠ {expected}");
        close(compute(1), 1.12e-3);
        close(compute(4), 1e-3);
        close(compute(8), 0.92e-3);
        assert_eq!(reads.at(64, 64, 0, 8, None)[0].compute_seconds, 1e-3, "unpriced without the executor's costs");
        // Measured at 50 µs a step, below the 160 µs the measured group's
        // passes are priced at: both pricings shrink to fit it (by 5/16),
        // so a group of 8 takes 25 µs a step, not -30 µs.
        let terms = GroupLayerTerms { compute_seconds: 50e-6, ..terms };
        let union = reads.unions[0].clone();
        let over = GroupReads::new(vec![terms], vec![union], lines(3_000_000, 1), reads.curve.clone(), READ, false, Notice::Written).unwrap();
        let compute = |steps| over.at(64, 64, 0, steps, Some(&pass))[0].compute_seconds;
        close(compute(1), 87.5e-6);
        close(compute(4), 50e-6);
        close(compute(8), 25e-6);
    }

    /// The wait bound clips a grant: a group whose union the bank holds
    /// runs its compute alone, 2 ms a place over two layers, so a bound of
    /// 5 ms admits 2 of the 3 places room admits. A group of cold experts
    /// runs its union's read and moves whatever its places (72.0 ms): a
    /// bound above
    /// it leaves the grant as unbounded, one below it holds the place held.
    #[test]
    fn a_grant_is_clipped_by_the_wait_bound() {
        let held = reads(&[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(grant_within(&held, None, Some(5e-3)), PlaceGrant { places: 2, slots: 8 });
        let cold = reads(&[8; 7]);
        assert_eq!(grant_within(&cold, None, Some(0.08)), grant(&cold, None));
        assert_eq!(grant_within(&cold, None, Some(0.07)), PlaceGrant { places: 1, slots: 8 });
    }

    /// Places win memory from slots where a group's experts are all cold:
    /// one slot holds none of its union, and a group fewer saves a whole
    /// union's read. The first group holds the 3 places room admits; the 5
    /// chunks left run as one group once the bank gives up 3 slots (the
    /// 52 MB of 5 places past the 4 MB the first leaves), and no more places
    /// save more. Slots win where decode runs beside the prompt and each
    /// slot given up slows its step by two thirds of a step, or where the
    /// slots given up hold the group's union.
    #[test]
    fn places_and_slots_are_weighed_by_what_they_save() {
        let cold = reads(&[8; 7]);
        assert_eq!(grant(&cold, None), PlaceGrant { places: 5, slots: 5 });
        let heavy = |slots: u64| 0.03 + 0.02 * (8 - slots) as f64;
        assert_eq!(grant(&cold, Some((1., heavy))), PlaceGrant { places: 3, slots: 8 });
        let light = |slots: u64| 0.03 + 0.001 * (8 - slots) as f64;
        assert_eq!(grant(&cold, Some((1., light))), PlaceGrant { places: 5, slots: 5 });
        let held = reads(&[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(grant(&held, None), PlaceGrant { places: 3, slots: 8 });
        // No group counted, no grant priced.
        let curve = MissCurve { steps: 0, layers: Vec::new() };
        assert_eq!(GroupReads::new(Vec::new(), Vec::new(), lines(1, 0), curve, READ, false, Notice::Written), None);
    }

    /// The read line of the tests.
    const READ: ReadLine = match ReadLine::new(4e-4, 1. / 2e9) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };
}
