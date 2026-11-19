use crate::experts::reuse::{Evidence, Ranking, Reuse, Rows, StepKeys};
use crate::experts::watcher::Step;

/// One step's routing at one streamed layer, as the device read it: the
/// step's sequence, the layer, its experts and the choices
/// its rows routed to them (`K` a row; several may choose one expert).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedRead {
    pub sequence: u64,
    pub layer: usize,
    pub experts: Vec<usize>,
    pub choices: u64,
}

/// A counted decode step's launch: its rows and the context tokens a row
/// reads, the mean over its rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CountedLaunch {
    pub rows: u64,
    pub tokens: u64,
}

/// Each streamed layer's decode reads by the depth at which they found their
/// experts in their group's order by the ranking ([`Stack`]), so the misses
/// of a bank of every size at once: a bank of `s` slots a layer (`s · L`
/// records of a group of `L` layers) misses a read whose depth bucket is `s`
/// or more ([`LayerCurve::at`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MissCurve {
    /// The decode steps counted.
    pub steps: u64,
    pub layers: Vec<LayerCurve>,
}

/// One layer's reads, `experts + 1` buckets each: an expert read at depth
/// `d` (the group's experts that outrank it) lies in bucket `d / L`, `L` the
/// group's layers, since `s` slots are `s · L` records; a first read in the
/// last.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LayerCurve {
    /// Each expert read, by its depth.
    pub misses: Vec<u64>,
    /// Each step's read, by its deepest expert's depth.
    pub requests: Vec<u64>,
    /// The choices its reads routed: over its expert reads, the rows an
    /// expert's use computes.
    pub choices: u64,
}

impl LayerCurve {
    /// Its expert reads, `U`: every read misses a bank of no slots.
    pub fn reads(&self) -> u64 {
        self.misses.iter().sum()
    }

    /// What a bank of `slots` slots misses at the layer over the steps
    /// counted: its requests (steps that missed there) and misses. A bank
    /// deeper than the stack (more slots than the layer has experts) misses
    /// nothing.
    pub fn at(&self, slots: usize) -> (u64, u64) {
        let depth = slots.min(self.misses.len());
        (self.requests[depth..].iter().sum(), self.misses[depth..].iter().sum())
    }
}

/// The reads of the bank's layers by their depth in the one ranking's group
/// order ([`Reuse`]): its feed, in the device's order, and the miss curves
/// that price every bank size by the ranking that decides what the bank
/// holds. A group's pool of `s · L` records holds its `s · L` most ranked
/// experts where it admits only what outranks its least, so a read misses it
/// when its expert lies `s` or more slots deep (`Reuse::depth`).
pub struct Stack {
    ranking: Ranking,
    experts: usize,
    curve: MissCurve,
    /// The last decode step counted.
    counted: Option<Step>,
    /// The rows and context tokens of the decode steps `curve` counts,
    /// summed: each step's launch's, counted with it.
    launches: CountedLaunch,
    /// Every decode step read, counted or not, by depth
    /// ([`Self::decoded`]), and the last of them.
    decoded: MissCurve,
    last_decoded: Option<Step>,
}

/// Keys by their last read, the reuse-distance stack (Mattson's) of a
/// stored table's rows (`stored_table::RowStack`): per step stamp, its keys
/// ascending, and a
/// Fenwick tree over the steps' ranks counting their keys, so a key's depth
/// is found in time logarithmic in the steps held.
pub(crate) struct Ranks<S, K> {
    ranks: std::collections::BTreeMap<S, usize>,
    keys: std::collections::HashMap<S, Vec<K>>,
    counts: Vec<u32>,
}

impl<S, K> Default for Ranks<S, K> {
    fn default() -> Self {
        Self {
            ranks: std::collections::BTreeMap::new(),
            keys: std::collections::HashMap::new(),
            counts: Vec::new(),
        }
    }
}

impl<S: Copy + Ord + std::hash::Hash, K: Copy + Ord> Ranks<S, K> {
    fn add(&mut self, rank: usize, value: i32) {
        let mut index = rank + 1;
        while index < self.counts.len() {
            self.counts[index] = self.counts[index].wrapping_add_signed(value);
            index += index & index.wrapping_neg();
        }
    }

    /// Keys of steps ranked below `rank`.
    fn below(&self, rank: usize) -> u32 {
        let (mut index, mut total) = (rank, 0);
        while index > 0 {
            total += self.counts[index];
            index -= index & index.wrapping_neg();
        }
        total
    }

    /// The keys ranked above `key`, last read at `stamp`.
    pub(crate) fn above(&self, stamp: S, key: K) -> usize {
        let rank = self.ranks[&stamp];
        let keys = &self.keys[&stamp];
        let later = self.below(self.counts.len() - 1) - self.below(rank + 1);
        later as usize + keys.len() - keys.partition_point(|&other| other <= key)
    }

    /// The earliest step held and its keys, the deepest ranked.
    pub(crate) fn oldest(&self) -> Option<(S, &[K])> {
        let (&stamp, _) = self.ranks.first_key_value()?;
        Some((stamp, &self.keys[&stamp]))
    }

    pub(crate) fn remove(&mut self, stamp: S, key: K) {
        let keys = self.keys.get_mut(&stamp).expect("a stamp of a ranked key");
        keys.remove(keys.binary_search(&key).expect("a ranked key"));
        let rank = self.ranks[&stamp];
        if keys.is_empty() {
            self.keys.remove(&stamp);
            self.ranks.remove(&stamp);
        }
        self.add(rank, -1);
    }

    /// Rank `key` at `stamp`, of a group of `capacity` keys. A step after
    /// every ranked one takes the next rank; any other (a chain's earlier
    /// step reappearing), or one past the tree, ranks every step again.
    pub(crate) fn insert(&mut self, stamp: S, key: K, capacity: usize) {
        if !self.ranks.contains_key(&stamp) {
            let next = self.ranks.last_key_value().map_or(0, |(_, &rank)| rank + 1);
            let last = self.ranks.last_key_value().map(|(&last, _)| last);
            match next + 1 < self.counts.len() && last.is_none_or(|last| last < stamp) {
                true => {
                    self.ranks.insert(stamp, next);
                }
                false => {
                    self.ranks.insert(stamp, 0);
                    // Live steps are at most the keys: twice that leaves as
                    // many ranks again before the next renumbering.
                    self.counts = vec![0; 2 * capacity + 2];
                    let ranks = std::mem::take(&mut self.ranks);
                    for (rank, step) in ranks.into_keys().enumerate() {
                        self.ranks.insert(step, rank);
                        let count = self.keys.get(&step).map_or(0, Vec::len);
                        self.add(rank, count as i32);
                    }
                }
            }
        }
        let keys = self.keys.entry(stamp).or_default();
        keys.insert(keys.partition_point(|&other| other < key), key);
        self.add(self.ranks[&stamp], 1);
    }
}

impl Stack {
    /// No read yet of `layers` layers of at most `experts` experts each,
    /// ranked by `ranking`, which it feeds.
    pub fn new(ranking: Ranking, layers: usize, experts: usize) -> Self {
        let empty = LayerCurve {
            misses: vec![0; experts + 1],
            requests: vec![0; experts + 1],
            choices: 0,
        };
        Self {
            ranking,
            experts,
            curve: MissCurve {
                steps: 0,
                layers: vec![empty.clone(); layers],
            },
            counted: None,
            launches: CountedLaunch::default(),
            decoded: MissCurve {
                steps: 0,
                layers: vec![empty; layers],
            },
            last_decoded: None,
        }
    }

    /// The ranking it feeds.
    #[cfg(test)]
    pub(crate) fn ranking(&self) -> Ranking {
        self.ranking.clone()
    }

    /// An empty stack over a ranking of its own of `layers` layers of
    /// `experts` experts each, each layer a group of its own of `slots`
    /// records.
    #[cfg(test)]
    pub fn alone(layers: usize, experts: usize, slots: usize) -> Self {
        let time = crate::experts::horizon::Time::new(&vec![vec![slots.max(1) as u64]; layers], 32);
        Self::new(Reuse::ranking(&vec![experts; layers], &(0..layers).collect::<Vec<_>>(), &vec![slots; layers], time), layers, experts)
    }

    /// Step `read.sequence` read `read.experts` at `read.layer`, in the
    /// device's order of reads, its rows' selection keys `keys` where its
    /// launch wrote them: together, each expert at its depth before any of
    /// them is ranked again. A decode read is counted where its step's
    /// launch is given (`counted`), and so is its step, with that launch,
    /// the first time one of its reads is (a group's steps first read in
    /// sequence order, and groups in theirs).
    pub fn read(&mut self, read: &RoutedRead, counted: Option<CountedLaunch>, keys: Option<&StepKeys>) {
        self.read_at(read, counted, keys, None);
    }

    /// [`Self::read`] with a pre-captured token clock position `clock` for
    /// `read.sequence` (`Reuse::read_at`).
    pub(crate) fn read_at(
        &mut self,
        read: &RoutedRead,
        counted: Option<CountedLaunch>,
        keys: Option<&StepKeys>,
        clock: Option<crate::experts::horizon::Tokens>,
    ) {
        let mut experts = read.experts.clone();
        experts.sort_unstable();
        experts.dedup();
        let rows = keys.map(|keys| std::sync::Arc::new(keys.rows(read.layer)));
        let depths = Reuse::feeding(&self.ranking).read_at(&RoutedRead { experts, ..read.clone() }, rows, Evidence::Decode, clock);
        count(&mut self.decoded, &mut self.last_decoded, read, &depths);
        if let Some(launch) = counted {
            if count(&mut self.curve, &mut self.counted, read, &depths) {
                self.launches = CountedLaunch { rows: self.launches.rows + launch.rows, tokens: self.launches.tokens + launch.tokens };
            }
        }
    }

    /// A prompt group's union `read`, read at its last step `read.sequence`
    /// ([`Self::read`]) into the ranking, its rows the union's choices
    /// (`Rows::union`): a bank of `s` slots holds the union's experts less
    /// than `s` deep. Each expert's depth, in slots, in `read.experts`'
    /// ascending order.
    pub fn read_group(&mut self, read: &RoutedRead, rows: Rows) -> Vec<usize> {
        Reuse::feeding(&self.ranking).read(read, Some(std::sync::Arc::new(rows)), Evidence::Prompt)
    }

    /// What the load's counted decode read and ran ([`LiveDecode`]); none
    /// before it counted a step.
    pub fn live(&self) -> Option<magnitude_cost_model::LiveDecode> {
        let steps = self.curve.steps;
        if steps == 0 {
            return None;
        }
        let rows = self.launches.rows as f64 / steps as f64;
        let depth = u32::try_from(self.launches.tokens / steps).expect("a decode's mean context fits a launch's");
        let layers = self.curve.layers.iter().map(|layer| magnitude_cost_model::LayerReads::of_depths(&layer.misses, &layer.requests)).collect();
        Some(magnitude_cost_model::LiveDecode { layers, steps, rows, depth })
    }

    /// Every decode step read since the load, counted into the record or
    /// not, by depth: the misses of a bank of every size over them, so the
    /// promotions the ranking asks of a layer a step at any share.
    pub fn decoded(&self) -> &MissCurve {
        &self.decoded
    }
}

/// Count `read`, its experts found at `depths`, into `curve`, and its step
/// the first time one of its reads is, `last` the step counted last: whether
/// its step was newly counted.
fn count(curve: &mut MissCurve, last: &mut Option<Step>, read: &RoutedRead, depths: &[usize]) -> bool {
    let new = last.is_none_or(|last| last < read.sequence);
    if new {
        *last = Some(read.sequence);
        curve.steps += 1;
    }
    if let Some(&deepest) = depths.iter().max() {
        let layer = &mut curve.layers[read.layer];
        depths.iter().for_each(|&depth| layer.misses[depth] += 1);
        layer.requests[deepest] += 1;
        layer.choices += read.choices;
    }
    new
}
