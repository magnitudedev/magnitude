//! Each expert's reuse as the routes measured it: the one
//! ranking of what the bank holds and of how much it holds. It decides every
//! residency choice (the bank's promotions and their victims, a prompt
//! group's seeds' and a shrink's; the host tier's evictions and the
//! order its fill lands in; the staging record a landing takes), and its
//! stack prices every bank size (`Stack`'s miss curve and returns).
//!
//! What it measures is the routes' reads ([`RoutedRead`]): each completed
//! decode step's routing record, every layer's chosen experts wherever
//! they were served from, and each prompt group's union of a layer's
//! routed experts, in the device's order (`Stack`, its one feed). An
//! expert's reuse is a decayed count of its reads: a read at step `t`
//! weighs `2^(−(now − t)/h)` at step `now`, `h` the half-life in steps.
//! Every expert decays alike, so the order between experts never changes as
//! steps pass: the score is the sum's *equivalent step* ([`Score`]),
//! `h · log2 Σ 2^(t/h)`, which ranks without being rescaled. At a half-life
//! of zero it is the last read: recency, the measure's degenerate case.
//!
//! Where the reads leave experts tied, their order is the one the reads
//! gave them before ([`Rank`]): each read hands its experts the next places
//! in one order of logging, among themselves as they ranked before it. No
//! choice of the bank's reads the experts' indices (the host tier's still
//! tie by them), so a prompt group's union read of
//! every expert once leaves a layer's decode share above the experts only
//! the group read, at recency as at every half-life. A read's experts enter
//! the stack and the pool least ranked first ([`Measure::entering`]), so the
//! most ranked end on top, and stay where a share or a pool cannot take
//! them all.
//!
//! The same score prices a promotion ([`Reuse::weighed`]): the reads it
//! expects over the bank's stay are its intensity at the live candidate's
//! half-life, decaying at that half-life through the stay, so what earns a
//! copy and what the ranking holds are judged by one value, and a want is
//! never weighed against a victim by a measure other than the one that
//! chose both. An expert is weighed by its own score at the live
//! half-life, never a score taken before: a score is in one half-life's
//! units, and weighed at another it would mean nothing (a long half-life's
//! score of a much-read expert lies far past the latest step, which a short
//! one weighs as unbounded). A choice that ranks or weighs several experts
//! reads them all from one hold of the ranking ([`Reuse::of`]), so the live
//! candidate never changes within it. A short half-life vouches for few steps: a just-read expert
//! is worth the reads the measure counts, never its latest rate held over
//! a whole stay.
//!
//! Each half-life of [`HALF_LIVES`] keeps its scores and, per layer, its
//! priority stack (Mattson's): a read moves its expert to the top, and each
//! expert it passes over settles below the more reused of those above it,
//! so a layer's share of `s` records under the ranking holds the stack's top
//! `s`, and one pass gives the misses at every size: each miss, as the
//! read's experts enter, promoted into its least reused record. Since the
//! scores of experts not read keep their order, the stack property holds.
//!
//! Each half-life also keeps, per bank group, the group's records as one
//! pool: every miss of a member layer promoted into the group's least
//! reused record wherever its layer (scores are steps, so they compare
//! across layers). A pool lets a layer whose
//! reads churn hold more than its share while a quieter one holds less;
//! Bélády gains 10–20% from that freedom, and recorded routes reward it at
//! some half-lives and sizes and punish it at others.
//!
//! The bank holds the candidate, a half-life's per-layer shares or its
//! group pools, that missed the fewest reads so far, the first on a tie
//! (per-layer recency until the reads tell otherwise). Every candidate is
//! kept up to date at every read, so a change of the one held is a choice
//! of index: nothing is reranked, and no rank is stored anywhere else to go
//! stale (whatever ranks asks [`Reuse::rank`]). Sizes are priced by the
//! per-layer stacks alone (`Stack`): a group pool is a cache at the group's
//! records, not a stack, since one ordered over a group's every expert
//! would cost a scan of thousands a miss.
//!
//! The reads reach the host only once a step completed. A rank may take a
//! read the log has yet to bring beside them ([`Reuse::rank`]: a prompt
//! group's routed step at its seeding, a landing's step): one past the
//! expert's last logged read counts as a read.

use super::directory::RoutedRead;
use super::watcher::Step;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard};

/// The half-lives, in steps, the ranking measures its reads by, ascending.
const HALF_LIVES: [f64; 10] = [0., 1., 2., 4., 8., 16., 32., 64., 128., 256.];

/// An expert's reuse as its equivalent step: greater is more reused. An
/// expert never read scores below every other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score(f64);

impl Score {
    pub const NEVER: Self = Self(f64::NEG_INFINITY);

    /// The score of one read at step `step`, at any half-life.
    pub fn at(step: Step) -> Self {
        Self(step as f64)
    }

    /// The score of the reads of `self` and of `other` together at
    /// half-life `half_life`: `h · log2(2^(a/h) + 2^(b/h))`, their later at
    /// a half-life of zero (the limit).
    fn sum(self, other: Self, half_life: f64) -> Self {
        let (low, high) = match self <= other {
            true => (self.0, other.0),
            false => (other.0, self.0),
        };
        if half_life == 0. || low == f64::NEG_INFINITY {
            return Self(high);
        }
        Self(high + half_life * (1. + ((low - high) / half_life).exp2()).log2())
    }

    /// The reads the score at half-life `half_life` expects over the
    /// `stay` steps from step `latest`: its count decayed to `latest`,
    /// `n = 2^((s − latest)/h)`, is a read intensity of `n · (1 − 2^(−1/h))`
    /// a step that goes on decaying at the half-life, which over the stay
    /// sums to `n · (1 − 2^(−stay/h))`. Never more than the reads the
    /// score counts, nor than the intensity held over the whole stay. At a
    /// half-life of zero (the limit) a read at `latest` expects one read,
    /// an earlier one none.
    fn expected(self, latest: Step, half_life: f64, stay: f64) -> f64 {
        let since = Self::at(latest).0 - self.0;
        if half_life == 0. {
            return if since <= 0. { 1. } else { 0. };
        }
        (-since / half_life).exp2() * (1. - (-stay / half_life).exp2())
    }
}

// A score is never NaN: each is a step, a sum of steps, or none's negative
// infinity.
impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// One expert's logged reads: their score, the step of the last, the
/// choices that read routed to it, and its place in the order of logging
/// (0 for none).
#[derive(Clone, Copy, Debug)]
struct Logged {
    score: Score,
    last: Option<Step>,
    routed: u32,
    order: u64,
}

impl Logged {
    const NONE: Self = Self { score: Score::NEVER, last: None, routed: 0, order: 0 };
}

/// An expert's rank, least first: its score, its last read, the choices
/// that read routed to it (of a prompt group's union, how much of the
/// prompt used it), then its place in the order of logging, which no two
/// experts read share: a total order over the experts read, which ranks
/// experts the reads tie as the reads before ranked them. The one order
/// every choice of what to keep compares, the ranking's and the
/// directory's alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank {
    score: Score,
    last: Option<Step>,
    routed: u32,
    order: u64,
}

impl Rank {
    /// The expert's reuse.
    pub fn score(self) -> Score {
        self.score
    }

    /// The step that last read the expert, if one has.
    pub fn last(self) -> Option<Step> {
        self.last
    }
}

/// A held expert's place in its group's pool, least reused first: its
/// [`Rank`] (its own, as every expert held was read), then the (layer,
/// expert) it holds.
type Pooled = (Rank, usize, usize);

/// One bank group's records as a pool at one half-life: the experts it
/// holds, by their rank, and how many it holds at most.
struct Pool {
    records: usize,
    held: BTreeSet<Pooled>,
}

/// The reads at one half-life: every layer's experts' scores, its priority
/// stack (each expert read, the top first), and the reads it missed at each
/// layer's share; and each group's pool, which experts it holds, and the
/// reads the pools missed.
struct Measure {
    half_life: f64,
    layers: Vec<Vec<Logged>>,
    /// The places in the order of logging handed out ([`Logged::order`]).
    logged: u64,
    stacks: Vec<Vec<usize>>,
    misses: u64,
    /// Each layer's group.
    groups: Vec<usize>,
    pools: Vec<Pool>,
    pooled: Vec<Vec<bool>>,
    pooled_misses: u64,
}

impl Measure {
    fn new(half_life: f64, experts: &[usize], groups: &[usize], records: &[usize]) -> Self {
        Self {
            half_life,
            layers: experts.iter().map(|&experts| vec![Logged::NONE; experts]).collect(),
            logged: 0,
            stacks: experts.iter().map(|&experts| Vec::with_capacity(experts)).collect(),
            misses: 0,
            groups: groups.to_vec(),
            pools: records.iter().map(|&records| Pool { records, held: BTreeSet::new() }).collect(),
            pooled: experts.iter().map(|&experts| vec![false; experts]).collect(),
            pooled_misses: 0,
        }
    }

    /// `layer`'s `expert`'s place in its group's pool.
    fn pooled_rank(&self, layer: usize, expert: usize) -> Pooled {
        (self.rank(layer, expert), layer, expert)
    }

    /// Group `group`'s pool holds `records` records from now: its least
    /// reused experts leave it until it fits.
    fn fit(&mut self, group: usize, records: usize) {
        let pool = &mut self.pools[group];
        pool.records = records;
        while pool.held.len() > records {
            let (.., layer, expert) = pool.held.pop_first().expect("a pool past its records holds one");
            self.pooled[layer][expert] = false;
        }
    }

    /// `layer`'s `expert`'s depth in its stack: the experts above it, its
    /// layer's experts where it was never read (deeper than any share).
    fn depth(&self, layer: usize, expert: usize) -> usize {
        let stack = &self.stacks[layer];
        stack.iter().position(|&above| above == expert).unwrap_or(self.layers[layer].len())
    }

    /// `layer`'s `expert`'s rank ([`Rank`]).
    fn rank(&self, layer: usize, expert: usize) -> Rank {
        let Logged { score, last, routed, order } = self.layers[layer][expert];
        Rank { score, last, routed, order }
    }

    /// `layer`'s `experts` in the order a read of them enters the stack and
    /// the pool, and takes its places in the order of logging: least ranked
    /// first. Experts nothing read before tie on every key; they keep the
    /// read's order, the only one there is.
    fn entering(&self, layer: usize, experts: &[usize]) -> Vec<usize> {
        let mut entering = experts.to_vec();
        entering.sort_by_key(|&expert| self.rank(layer, expert));
        entering
    }

    /// Log `read`: each expert's depth before any of them moves, counted a
    /// miss where at least `slots` deep, and whether its group's pool holds
    /// it, counted a pool miss where not; then its places in the order of
    /// logging, as its experts ranked before it, and its scores; then each
    /// expert to its stack's top, and each the pool lacked into the pool's
    /// least reused record (the read's own experts among them, as the stack
    /// ranks). `routed` is each expert's choices in the read. The read's
    /// experts move to the top, and into the pool, as they enter
    /// ([`Self::entering`], by their ranks after it): the most chosen of a
    /// prompt group's union end above the rest where their scores tie, and
    /// experts it leaves tied keep the order the reads before it gave them.
    /// The depths, and the layers whose experts left the pool.
    fn read(&mut self, read: &RoutedRead, routed: &[u32], slots: usize) -> (Vec<usize>, BTreeSet<usize>) {
        assert_eq!(routed.len(), read.experts.len(), "each expert read has its choices");
        let layer = read.layer;
        let depths = read.experts.iter().map(|&expert| self.depth(layer, expert)).collect::<Vec<_>>();
        self.misses += depths.iter().filter(|&&depth| depth >= slots).count() as u64;
        let (held, lacked): (Vec<usize>, Vec<usize>) = read.experts.iter().partition(|&&expert| self.pooled[layer][expert]);
        self.pooled_misses += lacked.len() as u64;
        let group = self.groups[layer];
        // A held expert's place moves with its score.
        for &expert in &held {
            let place = self.pooled_rank(layer, expert);
            self.pools[group].held.remove(&place);
        }
        for expert in self.entering(layer, &read.experts) {
            self.logged += 1;
            self.layers[layer][expert].order = self.logged;
        }
        let now = Score::at(read.sequence);
        for (&expert, &choices) in read.experts.iter().zip(routed) {
            let logged = &mut self.layers[layer][expert];
            logged.score = logged.score.sum(now, self.half_life);
            logged.last = logged.last.max(Some(read.sequence));
            logged.routed = choices;
        }
        for &expert in &held {
            let place = self.pooled_rank(layer, expert);
            self.pools[group].held.insert(place);
        }
        let entering = self.entering(layer, &read.experts);
        for &expert in &entering {
            self.promote(layer, expert);
        }
        let mut left = BTreeSet::new();
        for expert in entering.into_iter().filter(|expert| lacked.contains(expert)) {
            let pool = &mut self.pools[group];
            if pool.held.len() >= pool.records {
                let Some((.., of, victim)) = pool.held.pop_first() else {
                    continue;
                };
                self.pooled[of][victim] = false;
                left.insert(of);
            }
            if pool.held.len() < pool.records {
                let place = self.pooled_rank(layer, expert);
                self.pools[group].held.insert(place);
                self.pooled[layer][expert] = true;
            }
        }
        (depths, left)
    }

    /// Move `layer`'s `expert` to its stack's top: each expert above its
    /// place settles below the more reused of those above it, the less
    /// reused carried down into the place it left (Mattson's priority
    /// update).
    fn promote(&mut self, layer: usize, expert: usize) {
        let at = self.stacks[layer].iter().position(|&above| above == expert);
        let to = at.unwrap_or(self.stacks[layer].len());
        if to == 0 {
            // At the top already, or the first read of an empty stack.
            if at.is_none() {
                self.stacks[layer].push(expert);
            }
            return;
        }
        let mut carried = std::mem::replace(&mut self.stacks[layer][0], expert);
        for depth in 1..to {
            let held = self.stacks[layer][depth];
            if self.rank(layer, carried) > self.rank(layer, held) {
                self.stacks[layer][depth] = carried;
                carried = held;
            }
        }
        match at {
            Some(at) => self.stacks[layer][at] = carried,
            None => self.stacks[layer].push(carried),
        }
    }
}

/// Every streamed layer's experts' reuse, measured at every half-life of
/// [`HALF_LIVES`], each as per-layer shares and as group pools: the bank
/// holds the candidate that missed least.
pub struct Reuse {
    measures: Vec<Measure>,
    /// Each layer's group, and its bank share, in records.
    groups: Vec<usize>,
    slots: Vec<usize>,
    /// The candidate the bank holds: measure `live % HALF_LIVES.len()`,
    /// its per-layer shares below that count and its group pools from it.
    live: usize,
    /// Per layer, each expert's membership of what the live candidate
    /// holds of it.
    tops: Vec<Vec<bool>>,
    /// The (layer, expert)s whose membership of their layer's top changed
    /// since they were last taken ([`Self::take_changes`]).
    changed: BTreeSet<(usize, usize)>,
    /// The reads logged ([`Self::reads`]), and the latest step among them.
    reads: u64,
    latest: Option<Step>,
}

/// The one ranking, shared by its feed (`Stack`, which reads it in the
/// device's order) and everything that ranks by it (the directory).
pub type Ranking = Arc<Mutex<Reuse>>;

/// Each group's records: its member layers' shares.
fn records(groups: &[usize], slots: &[usize]) -> Vec<usize> {
    let mut records = vec![0; groups.iter().max().map_or(0, |&last| last + 1)];
    groups.iter().zip(slots).for_each(|(&group, &slots)| records[group] += slots);
    records
}

impl Reuse {
    /// No read logged of layers of `experts` experts each, in bank groups
    /// `groups`, whose bank shares are `slots` records each.
    pub fn new(experts: &[usize], groups: &[usize], slots: Vec<usize>) -> Self {
        assert_eq!(experts.len(), slots.len(), "each layer has its share");
        assert_eq!(experts.len(), groups.len(), "each layer has its group");
        let records = records(groups, &slots);
        let measures = HALF_LIVES.iter().map(|&half_life| Measure::new(half_life, experts, groups, &records)).collect();
        Self {
            measures,
            groups: groups.to_vec(),
            slots,
            live: 0,
            tops: experts.iter().map(|&experts| vec![false; experts]).collect(),
            changed: BTreeSet::new(),
            reads: 0,
            latest: None,
        }
    }

    /// [`Self::new`], shared.
    pub fn ranking(experts: &[usize], groups: &[usize], slots: Vec<usize>) -> Ranking {
        Arc::new(Mutex::new(Self::new(experts, groups, slots)))
    }

    /// The ranking `ranking` holds now.
    pub fn of(ranking: &Ranking) -> MutexGuard<'_, Self> {
        ranking.lock().expect("the reuse ranking")
    }

    /// The measure the live candidate is of.
    fn measure(&self) -> &Measure {
        &self.measures[self.live % HALF_LIVES.len()]
    }

    /// Whether the live candidate is its measure's group pools.
    fn pooled(&self) -> bool {
        self.live >= HALF_LIVES.len()
    }

    /// The reads candidate `candidate` missed so far.
    fn missed(&self, candidate: usize) -> u64 {
        let measure = &self.measures[candidate % HALF_LIVES.len()];
        match candidate >= HALF_LIVES.len() {
            true => measure.pooled_misses,
            false => measure.misses,
        }
    }

    /// Log the route's `read`, `routed` each expert's choices in it: each
    /// expert's depth in slots, by the live half-life's stack before the
    /// read (its layer's experts where never read).
    pub fn read(&mut self, read: &RoutedRead, routed: &[u32]) -> Vec<usize> {
        self.reads += 1;
        self.latest = self.latest.max(Some(read.sequence));
        let slots = self.slots[read.layer];
        let at = self.live % HALF_LIVES.len();
        let mut reads = self.measures.iter_mut().map(|measure| measure.read(read, routed, slots)).collect::<Vec<_>>();
        let (depths, left) = reads.swap_remove(at);
        let live = (0..2 * HALF_LIVES.len()).min_by_key(|&candidate| self.missed(candidate)).expect("candidates");
        // Another candidate reorders every layer; this read moves its own,
        // and a pool's misses the layers they evicted from.
        match live == self.live {
            true => {
                self.retop(read.layer);
                if self.pooled() {
                    left.into_iter().for_each(|layer| self.retop(layer));
                }
            }
            false => {
                self.live = live;
                (0..self.tops.len()).for_each(|layer| self.retop(layer));
            }
        }
        depths
    }

    /// `layer`'s bank share is `slots` records from now, and its group's
    /// pools hold its members' shares.
    pub fn resize(&mut self, layer: usize, slots: usize) {
        self.slots[layer] = slots;
        let group = self.groups[layer];
        let records = records(&self.groups, &self.slots)[group];
        self.measures.iter_mut().for_each(|measure| measure.fit(group, records));
        let members = (0..self.tops.len()).filter(|&member| self.groups[member] == group).collect::<Vec<_>>();
        members.into_iter().for_each(|member| self.retop(member));
    }

    /// Whether `layer`'s `expert` is among what the live candidate holds:
    /// its layer's share's top, or its group's pool.
    pub fn in_top(&self, layer: usize, expert: usize) -> bool {
        self.tops[layer][expert]
    }

    /// The reads logged so far: a want withheld at one count may earn its
    /// copy once it moves (`Directory::promotions`).
    pub fn reads(&self) -> u64 {
        self.reads
    }

    /// Take the (layer, expert)s whose membership of their share's experts
    /// ([`Self::in_top`]) changed since last taken.
    pub fn take_changes(&mut self) -> BTreeSet<(usize, usize)> {
        std::mem::take(&mut self.changed)
    }

    /// Take what the live candidate holds of `layer` again, noting each
    /// expert that entered or left it: in time proportional to the layer's
    /// experts.
    fn retop(&mut self, layer: usize) {
        let measure = self.measure();
        let held = match self.pooled() {
            true => measure.pooled[layer].clone(),
            false => {
                let stack = &measure.stacks[layer];
                let mut held = vec![false; self.tops[layer].len()];
                stack[..self.slots[layer].min(stack.len())].iter().for_each(|&expert| held[expert] = true);
                held
            }
        };
        let tops = &mut self.tops[layer];
        for (expert, (top, held)) in tops.iter_mut().zip(held).enumerate() {
            if *top != held {
                *top = held;
                self.changed.insert((layer, expert));
            }
        }
    }

    /// The half-life ranked by, in steps.
    #[cfg(test)]
    pub fn half_life(&self) -> f64 {
        self.measure().half_life
    }

    /// `layer`'s `expert`'s depth in slots now, by the ranking's stack.
    pub fn depth(&self, layer: usize, expert: usize) -> usize {
        self.measure().depth(layer, expert)
    }

    /// The step that last read `layer`'s `expert`, if one has.
    pub fn last(&self, layer: usize, expert: usize) -> Option<Step> {
        self.measure().layers[layer][expert].last
    }

    /// The reads `layer`'s `expert`, ranked beside a read at step `read`
    /// ([`Self::rank`]), is expected to have over a stay in the bank of
    /// `stay` steps, as the half-life the ranking ranks by expects them
    /// from the latest step logged ([`Score::expected`]). One value, the
    /// ranking's, so it orders experts as their ranks do; and it weighs the
    /// expert's score at the half-life that scored it. A stay not yet
    /// measured (no promotion has left one) is the longest half-life;
    /// before any read nothing is expected.
    pub fn weighed(&self, layer: usize, expert: usize, read: Option<Step>, stay: Option<f64>) -> f64 {
        let Some(latest) = self.latest else {
            return 0.;
        };
        let stay = stay.unwrap_or(HALF_LIVES[HALF_LIVES.len() - 1]);
        self.rank(layer, expert, read).score.expected(latest, self.measure().half_life, stay)
    }

    /// `layer`'s `expert`'s rank ([`Rank`]), beside a read at step `read`
    /// the log may not have brought yet (a step still running among them):
    /// counted as a read where it is past the last logged one, and then its
    /// last, the choices it routed unknown (none).
    pub fn rank(&self, layer: usize, expert: usize, read: Option<Step>) -> Rank {
        let measure = self.measure();
        let logged = measure.rank(layer, expert);
        match read.filter(|&read| Some(read) > logged.last) {
            Some(read) => Rank { score: logged.score.sum(Score::at(read), measure.half_life), last: Some(read), routed: 0, ..logged },
            None => logged,
        }
    }

    /// Rank by the stack of `half_life` from now, as though its candidate
    /// had missed least: what each layer's share holds is taken again.
    #[cfg(test)]
    pub fn go_live(&mut self, half_life: f64) {
        self.live = HALF_LIVES.iter().position(|&at| at == half_life).expect("a measured half-life");
        (0..self.tops.len()).for_each(|layer| self.retop(layer));
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    fn read(sequence: Step, experts: &[usize]) -> RoutedRead {
        RoutedRead { sequence, layer: 0, experts: experts.to_vec(), choices: experts.len() as u64 }
    }

    /// Each of `read`'s experts chosen once.
    fn once(read: &RoutedRead) -> Vec<u32> {
        vec![1; read.experts.len()]
    }

    /// Log `read` into `reuse`, each of its experts chosen once.
    fn log(reuse: &mut Reuse, read: RoutedRead) -> Vec<usize> {
        let routed = once(&read);
        reuse.read(&read, &routed)
    }

    /// Step `step`'s experts of a pattern that rewards frequency: expert 0
    /// every third step, beside a fresh one, `step`, at every step.
    pub(in crate::experts) fn every_third(step: Step) -> Vec<usize> {
        match step % 3 {
            0 => vec![0, step as usize],
            _ => vec![step as usize],
        }
    }

    /// Promote each of `layer`'s `experts` a read of `measure` lacked, as
    /// they enter ([`Measure::entering`]), into `held`'s least ranked
    /// record of `records`, by the measure's ranks, then (layer, expert): a
    /// cache that takes every miss, a layer's share or a group's pool.
    fn take_misses(measure: &Measure, layer: usize, experts: &[usize], held: &mut Vec<(usize, usize)>, records: usize) {
        for expert in measure.entering(layer, experts) {
            if held.contains(&(layer, expert)) {
                continue;
            }
            if held.len() < records {
                held.push((layer, expert));
                continue;
            }
            let victim = (0..held.len()).min_by_key(|&at| (measure.rank(held[at].0, held[at].1), held[at])).expect("a cache of records");
            held[victim] = (layer, expert);
        }
    }

    /// A recorded decode's routing (`fixtures/reuse`) misses at each read,
    /// by what the live candidate holds before it, exactly what the offline
    /// replay misses: every half-life's per-layer shares and its one pool
    /// of every layer's, each a cache built here taking every miss into its
    /// least ranked record (by the half-life's scores, never by the
    /// ranking's own stacks or pools), the least missed so far (the first on
    /// a tie) holding: read for read.
    fn replays_as_offline(fixture: &[u8]) {
        let word = |at: usize| u16::from_le_bytes([fixture[2 * at], fixture[2 * at + 1]]) as usize;
        let (layers, experts, width, steps, slots) = (word(0), word(1), word(2), word(3), word(4));
        let routes = &fixture[10..];
        assert_eq!(routes.len(), layers * steps * width, "a route a read");
        let mut reuse = Reuse::new(&vec![experts; layers], &vec![0; layers], vec![slots; layers]);
        let mut scored = HALF_LIVES.map(|half_life| Measure::new(half_life, &vec![experts; layers], &vec![0; layers], &[slots * layers]));
        let mut shares = vec![vec![Vec::new(); layers]; HALF_LIVES.len()];
        let mut pools = vec![Vec::new(); HALF_LIVES.len()];
        let mut missed = [0u64; 2 * HALF_LIVES.len()];
        for (position, route) in routes.chunks(width).enumerate() {
            let mut chosen = route.iter().map(|&expert| expert as usize).collect::<Vec<_>>();
            chosen.sort_unstable();
            let layer = position % layers;
            let lacked = |held: &dyn Fn(usize) -> bool| chosen.iter().filter(|&&expert| !held(expert)).count();
            let holds = |candidate: usize, expert: usize| match candidate < HALF_LIVES.len() {
                true => shares[candidate][layer].contains(&(layer, expert)),
                false => pools[candidate - HALF_LIVES.len()].contains(&(layer, expert)),
            };
            let live = (0..2 * HALF_LIVES.len()).min_by_key(|&at| missed[at]).expect("candidates");
            let offline = lacked(&|expert| holds(live, expert));
            assert_eq!(lacked(&|expert| reuse.in_top(layer, expert)), offline, "read {position} (step {}, layer {layer})", position / layers);
            let read = RoutedRead { sequence: (position / layers) as Step, layer, experts: chosen.clone(), choices: width as u64 };
            for at in 0..2 * HALF_LIVES.len() {
                missed[at] += lacked(&|expert| holds(at, expert)) as u64;
            }
            for (at, measure) in scored.iter_mut().enumerate() {
                measure.read(&read, &once(&read), 0);
                take_misses(measure, layer, &read.experts, &mut shares[at][layer], slots);
                take_misses(measure, layer, &read.experts, &mut pools[at], slots * layers);
            }
            reuse.read(&read, &once(&read));
        }
    }

    #[test]
    fn a_recorded_decode_replays_as_offline() {
        replays_as_offline(include_bytes!("../../../fixtures/reuse/qwen-gguf-47.bin"));
        replays_as_offline(include_bytes!("../../../fixtures/reuse/agent-26.bin"));
    }

    /// Before the reads tell otherwise the ranking is recency: a score is
    /// the last read, the log's or a later stamp's, and a read's depth is
    /// the experts read since.
    #[test]
    fn an_unmeasured_ranking_is_recency() {
        let mut reuse = Reuse::new(&[3], &[0], vec![2]);
        assert_eq!(log(&mut reuse, read(4, &[0])), [3], "a first read lies past every expert");
        assert_eq!(log(&mut reuse, read(5, &[1])), [3]);
        assert_eq!(log(&mut reuse, read(6, &[0])), [1], "one expert read since");
        assert_eq!(reuse.half_life(), 0.);
        assert_eq!(reuse.rank(0, 0, None).score(), Score(6.));
        assert_eq!(reuse.rank(0, 0, Some(3)).score(), Score(6.), "a stamp the log has passed adds nothing");
        assert_eq!(reuse.rank(0, 2, Some(7)).score(), Score(7.), "a stamp past the log is a later read");
        assert_eq!(reuse.rank(0, 2, None).score(), Score::NEVER);
        assert_eq!(reuse.last(0, 1), Some(5));
    }

    /// A score expects the reads its intensity gives as it decays over the
    /// stay: for an expert read every step, one over a stay of one step and
    /// `1 + 2^(−1/h) + …` over longer ones, all it counts (`1/(1 − 2^(−1/h))`)
    /// over a stay far past the half-life; `2^(−4/4) · (1 − 2^(−4/4))` over
    /// a stay of 4, four steps after one read at half-life 4; and at
    /// recency one read for an expert read at the latest step, none for
    /// one read before it, whatever the stay.
    #[test]
    fn a_score_expects_its_decaying_reads_over_the_stay() {
        let close = |got: f64, want: f64| assert!((got - want).abs() < 1e-9, "{got} against {want}");
        for half_life in [1., 4., 32.] {
            let mut measure = Measure::new(half_life, &[2], &[0], &[2]);
            for step in 1..=2000 {
                measure.read(&read(step, &[0]), &[1], 2);
            }
            let score = measure.rank(0, 0).score;
            close(score.expected(2000, half_life, 1.), 1.);
            close(score.expected(2000, half_life, 3.), (0..3).map(|k| (-k as f64 / half_life).exp2()).sum());
            close(score.expected(2000, half_life, 1e6), 1. / (1. - (-1. / half_life).exp2()));
        }
        close(Score::at(6).expected(10, 4., 4.), 0.25);
        close(Score::at(10).expected(10, 0., 100.), 1.);
        close(Score::at(9).expected(10, 0., 100.), 0.);
        close(Score::NEVER.expected(10, 4., 4.), 0.);
    }

    /// An expert's reads weigh over a stay what the live half-life expects
    /// of them from the latest read, the longest half-life before a stay
    /// is measured: at recency, an expert read at the latest step weighs
    /// one read over any stay, one read before it nothing.
    #[test]
    fn reads_weigh_what_the_live_half_life_expects_over_the_stay() {
        let mut reuse = Reuse::new(&[3], &[0], vec![2]);
        assert_eq!(reuse.weighed(0, 0, Some(6), Some(3.)), 0., "nothing read, nothing expected");
        log(&mut reuse, read(6, &[0]));
        log(&mut reuse, read(7, &[1]));
        assert_eq!(reuse.half_life(), 0.);
        assert_eq!(reuse.weighed(0, 1, None, Some(3.)), 1.);
        assert_eq!(reuse.weighed(0, 1, None, None), 1.);
        assert_eq!(reuse.weighed(0, 0, None, Some(3.)), 0.);
        assert_eq!(reuse.weighed(0, 2, None, Some(3.)), 0.);
    }

    /// Over a half-life an expert read often outranks one read once more
    /// recently, and two reads at one step weigh one half-life later.
    #[test]
    fn frequency_outranks_one_later_read_over_a_half_life() {
        let mut measure = Measure::new(4., &[2], &[0], &[2]);
        for step in [1, 2, 3, 4] {
            measure.read(&read(step, &[0]), &[1], 2);
        }
        measure.read(&read(5, &[1]), &[1], 2);
        assert!(measure.rank(0, 0) > measure.rank(0, 1));
        assert!((Score::at(0).sum(Score::at(0), 4.).0 - 4.).abs() < 1e-12);
    }

    /// A stack's misses at every share are those of a share of that many
    /// records whose every miss takes its least reused record: one pass
    /// counts what each size's own replay misses.
    #[test]
    fn a_stack_misses_what_each_share_misses() {
        let mut state = 0x2545_f491_u64;
        let reads = (1..400)
            .map(|step| {
                let mut experts = Vec::new();
                while experts.len() < 3 {
                    state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let uniform = (state >> 33) as f64 / (1u64 << 31) as f64;
                    let expert = (uniform * uniform * 16.) as usize;
                    if !experts.contains(&expert) {
                        experts.push(expert);
                    }
                }
                read(step, &experts)
            })
            .collect::<Vec<_>>();
        for half_life in [0., 4., 32.] {
            let mut stacked = Measure::new(half_life, &[16], &[0], &[0]);
            let depths = reads.iter().map(|read| stacked.read(read, &once(read), 0).0).collect::<Vec<_>>();
            for share in [3, 5, 8, 13] {
                // Each size replayed alone: every miss, as the read's
                // experts enter, promoted into the least reused record.
                let mut scored = Measure::new(half_life, &[16], &[0], &[0]);
                let mut held = Vec::new();
                let mut missed = 0;
                for read in &reads {
                    missed += read.experts.iter().filter(|&&expert| !held.contains(&(0, expert))).count();
                    scored.read(read, &once(read), share);
                    take_misses(&scored, 0, &read.experts, &mut held, share);
                }
                let counted = depths.iter().flatten().filter(|&&depth| depth >= share).count();
                assert_eq!(counted, missed, "half-life {half_life}, {share} records");
            }
        }
    }

    /// A prompt group's union read ranks its experts by what the reads tell
    /// apart, never by index: after decode read expert 9, a union of every
    /// expert, 12, 7, 3 and 9 the prompt's most chosen, leaves a share of
    /// four holding them at every half-life, as the stack and as a pool
    /// alike, the rest (each chosen once) below.
    #[test]
    fn a_unions_ties_break_by_its_choices_never_by_index() {
        let mut reuse = Reuse::new(&[16], &[0], vec![4]);
        log(&mut reuse, read(1, &[9]));
        let union = read(2, &(0..16).collect::<Vec<_>>());
        let choices = (0..16).map(|expert| match expert {
            12 => 50,
            7 => 40,
            3 => 30,
            9 => 20,
            _ => 1,
        });
        reuse.read(&union, &choices.collect::<Vec<_>>());
        let top = (0..16).filter(|&expert| reuse.in_top(0, expert)).collect::<Vec<_>>();
        assert_eq!(top, [3, 7, 9, 12]);
        for measure in &reuse.measures {
            let held = (0..16).filter(|&expert| measure.pooled[0][expert]).collect::<Vec<_>>();
            assert_eq!(held, [3, 7, 9, 12], "half-life {}'s pool", measure.half_life);
            let mut top = measure.stacks[0][..4].to_vec();
            top.sort_unstable();
            assert_eq!(top, [3, 7, 9, 12], "half-life {}'s stack", measure.half_life);
        }
    }

    /// A share of 2 records, an expert read every third step and a fresh
    /// one each step: recency evicts the hot expert for the fresh ones, a
    /// half-life keeps it, so the reads move the ranking off recency. A
    /// step reading both enters them as they rank, the hot one on top.
    #[test]
    fn the_half_life_ranked_by_is_the_one_that_missed_least() {
        let mut reuse = Reuse::new(&[128], &[0], vec![2]);
        for step in 1..=60 {
            log(&mut reuse, read(step, &every_third(step)));
        }
        assert!(reuse.half_life() > 0., "a half-life kept the hot expert");
        let (recency, live) = (&reuse.measures[0], &reuse.measures[reuse.live % HALF_LIVES.len()]);
        assert!(live.misses < recency.misses, "{} of {}", live.misses, recency.misses);
        assert_eq!(reuse.depth(0, 0), 0, "the hot expert above the step's fresh one");
    }
}
