//! A streamed prompt group's stages ([`Stages`]), which its size and merge
//! sets are formed by. A layer's stage, when its cold
//! experts are in the GPU's memory, is priced by the one cost function
//! ([`ExpertLines::staged`], [`super::cost`]), as decode is.

use super::cost::{ExpertLines, Miss};
use super::estimate::ExpertPass;
use super::streaming_costs::ReadLine;

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

    /// The steps a group runs while decode waits, at most `capacity`: the
    /// fewest whose compute covers each layer's stage, `g · c_l ≥ stage_l`,
    /// leaving none of it uncovered.
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
    /// which `unions` price by `pass`, both scaled down where the last
    /// group's exceed `c_l`, so `f` is never negative), a set of `s`
    /// computes `C(s)` and
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
    steps: usize,
    merged: usize,
    experts: Vec<(usize, u32)>,
    /// Each count of choices the group's rows routed to an expert, and how
    /// many experts it routed that many: what a pass over a group of any
    /// size reads ([`Self::read`]), taken from `experts` once, not at each
    /// of the many passes a group's pricing weighs.
    routed: Vec<(u32, usize)>,
}

impl GroupUnion {
    pub fn new(steps: usize, merged: usize, experts: Vec<(usize, u32)>) -> Self {
        let mut routed = std::collections::BTreeMap::<u32, usize>::new();
        for &(_, choices) in &experts {
            *routed.entry(choices).or_default() += 1;
        }
        Self { steps, merged, experts, routed: routed.into_iter().collect() }
    }

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

    /// The experts a group of `steps` steps is expected to read of all the
    /// group read ([`Self::deeper`] at no depth), by each count of choices.
    fn read(&self, steps: usize) -> f64 {
        self.routed.iter().map(|&(choices, experts)| experts as f64 * self.chance(steps, choices)).sum()
    }

    /// The choices a group of `steps` steps routes: the group's a step.
    fn choices(&self, steps: usize) -> f64 {
        let routed = self.routed.iter().map(|&(choices, experts)| u64::from(choices) * experts as u64).sum::<u64>();
        routed as f64 * steps as f64 / self.steps as f64
    }

    /// One pass over `steps` steps' rows at streamed layer `layer`.
    fn pass(&self, pass: &ExpertPass, layer: usize, steps: usize) -> f64 {
        pass.seconds(layer, self.read(steps), self.choices(steps))
    }

    /// What a group of `steps` steps merging `merged` of them costs at
    /// streamed layer `layer` in passes (`pass`): its steps alone a pass of
    /// one step's experts each, its merge set one pass of theirs.
    fn passes(&self, pass: &ExpertPass, layer: usize, steps: usize, merged: usize) -> f64 {
        (steps - merged) as f64 * self.pass(pass, layer, 1) + self.pass(pass, layer, merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
        let union = GroupUnion::new(4, 4, (0..64).map(|expert| (expert, 2)).collect());
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
    }

    /// The read line of the tests.
    const READ: ReadLine = match ReadLine::new(4e-4, 1. / 2e9) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };
}
