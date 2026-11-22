//! The bank's resizes. A shrink evicts records, beside running steps or
//! not, and moves nothing: the bank's records allow holes, so the experts it
//! keeps stay where their bytes are, and those it evicts leave the map for
//! the steps not yet begun (beside a running scratch-served chain, those
//! its steps have yet to decide too) while the steps before still read
//! them where they are, until the bank releases their storage behind them.
//! Residency answers it at its next turn, waiting on no flight. A growth
//! asks nothing: each of its slots joins as soon as every record tensor
//! backs it ([`Shared::join`]), beside running steps, which reach a record
//! only once the host maps it. Each carries the bank's slots, which the
//! watcher's telemetry reads.

use super::Shared;
use super::steps::{ScratchGroup, UnmapStep};
use super::core::PlacedWords;
use super::residency::Residency;
use super::telemetry::GroupReadWords;
use crate::experts::reuse::Reuse;
use crate::experts::source::Records;
use std::sync::{Condvar, Mutex};
use crate::domain::Trip;
use std::time::Instant;

/// Which records a shrink evicts: the bank's, which the device places into
/// beside a host tier, or those the host lands into (a host tier's, or the
/// one store's where the host places).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Evicting {
    Bank,
    Landed,
}

/// A shrink answered ([`Shared::shrink`]): the slots it commits the records
/// to, each group's records it evicted (holes from now, which the owner
/// releases the storage of behind the steps that may still read them), the
/// highest streamed layer whose hits a running chain's steps had decided at
/// it (none where they had decided none, or beside no chain), and its trip
/// through the watcher. No run a chain submits after every pass of that
/// layer resolves an evicted record.
pub(crate) struct Evicted {
    pub slots: usize,
    pub records: Vec<Vec<u32>>,
    pub decided: Option<usize>,
    pub trip: Trip,
}

/// A shrink's candidate commitments: from `shallowest` slots down to
/// `deepest` (the shrink's target), each group's records at a candidate's
/// slots formed by `records` only for the candidates the watcher weighs.
pub(crate) struct Candidates {
    pub shallowest: usize,
    pub deepest: usize,
    pub records: Box<dyn Fn(usize) -> Vec<usize> + Send>,
}

impl Candidates {
    /// One candidate: `records` at `slots`, the shrink's target.
    #[cfg(test)]
    pub(crate) fn one(slots: usize, records: Vec<usize>) -> Self {
        Self {
            shallowest: slots,
            deepest: slots,
            records: Box::new(move |_| records.clone()),
        }
    }

    fn count(&self) -> usize {
        self.shallowest - self.deepest + 1
    }

    /// The `index`th candidate, shallowest first: its slots.
    fn slots(&self, index: usize) -> usize {
        self.shallowest - index
    }
}

/// What the owner asks of the watcher to resize the bank.
pub(super) enum Ask {
    /// Evict `evicting`'s records down to the deepest of `candidates` whose
    /// records hold every one in flight, beside the running scratch-served
    /// chain, prompt group `chain`'s steps, where given.
    Evict { evicting: Evicting, candidates: Candidates, chain: Option<ScratchGroup> },
    /// Serve the places the bank holds from now by their words.
    Place(PlacedWords),
}

/// A growth's slot, which the watcher's directory applies at its next pass
/// in the order posted ([`Shared::join`]), backed by every record tensor of
/// its holding.
pub(crate) enum Join {
    /// Each group of the bank holds `records` more, beside a host tier
    /// (holes or past every record): the directory takes them, free, and
    /// promotions resolve their views on demand via `Group::bank`.
    Bank { records: Vec<Vec<usize>> },
    /// Each group's records the host lands into (its host tier's, or the
    /// one store's where the host places) hold `records` more, addressed by
    /// `views` (each group's record tensors' records the store holds now):
    /// the directory takes them, ready.
    Landed { records: Vec<Vec<u32>>, views: Vec<Vec<Records>> },
}

/// The slots a running growth has joined, in order, until the watcher
/// takes them, and how many it took and has not applied yet.
pub(super) struct Joins(Mutex<JoinQueue>);

#[derive(Default)]
struct JoinQueue {
    posted: Vec<Join>,
    applying: usize,
}

impl Joins {
    pub(super) fn new() -> Self {
        Self(Mutex::new(JoinQueue::default()))
    }

    fn queue(&self) -> std::sync::MutexGuard<'_, JoinQueue> {
        self.0.lock().expect("watcher joins")
    }

    fn post(&self, join: Join) {
        self.queue().posted.push(join);
    }

    /// The joins posted since the last take, in order, applying until
    /// [`Self::applied`].
    pub(super) fn take(&self) -> Vec<Join> {
        let mut queue = self.queue();
        let taken = std::mem::take(&mut queue.posted);
        queue.applying += taken.len();
        taken
    }

    /// `count` joins taken are applied: the directory and telemetry hold
    /// what they joined.
    fn applied(&self, count: usize) {
        self.queue().applying -= count;
    }

    /// Wait until serving applied every join posted: any request published
    /// from then on is served beside what they joined (the request
    /// service's pass applies the joins before it reads a request).
    #[cfg(test)]
    pub(super) fn wait_taken(&self) {
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while {
            let queue = self.queue();
            !queue.posted.is_empty() || queue.applying > 0
        } {
            assert!(Instant::now() < deadline, "the watcher applies a join within its rest");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

impl Ask {
    /// Whether it is served only between flights, once nothing moves a
    /// record: the places' words, which every step binds. A shrink waits on
    /// nothing: it evicts only records none moves.
    fn waits(&self) -> bool {
        matches!(self, Self::Place(_))
    }
}

enum Answer {
    Done(Trip),
    /// A shrink: evicted, none where even its shallowest candidate keeps
    /// fewer records than those in flight (or beside a chain it may not
    /// evict, or while a victim's write-back holds its record), or
    /// refused.
    Evicted(Result<Option<Evicted>, String>),
}

enum Resize {
    None,
    /// Asked, at that instant.
    Asked(Ask, Instant),
    Answered(Answer),
}

/// The resize the owner and the watcher exchange.
pub(super) struct ResizeHandoff {
    /// Serializes callers of [`Shared::ask`] so concurrent `shrink`/`place`
    /// requests never overwrite each other's `Resize::Asked` or steal each
    /// other's `Answer`. Never acquired by the watcher thread.
    ask_lock: Mutex<()>,
    resize: Mutex<Resize>,
    resized: Condvar,
}

impl ResizeHandoff {
    pub(super) fn new() -> Self {
        Self { ask_lock: Mutex::new(()), resize: Mutex::new(Resize::None), resized: Condvar::new() }
    }

    /// What the owner asks of a resize, and when it asked, taken.
    pub(super) fn asked(&self) -> Option<(Ask, Instant)> {
        let mut resize = self.resize.lock().expect("watcher resize");
        match std::mem::replace(&mut *resize, Resize::None) {
            Resize::Asked(asked, at) => Some((asked, at)),
            other => {
                *resize = other;
                None
            }
        }
    }

    /// Answer the owner, once what the resize changed is published
    /// ([`Residency::publish_resized`]) and what each layer holds ([`Owed`]).
    fn answer(&self, answer: Answer, _: Published) {
        let mut resize = self.resize.lock().expect("watcher resize");
        *resize = Resize::Answered(answer);
        self.resized.notify_all();
    }

    /// Wake an owner waiting for an answer: serving ended. Taken under the
    /// resize's lock, so an owner that found serving going on is already
    /// waiting.
    pub(super) fn wake(&self) {
        let _resize = self.resize.lock().expect("watcher resize");
        self.resized.notify_all();
    }
}

impl Shared {
    /// Evict `evicting`'s records down to the deepest of `candidates` whose
    /// every group keeps its records in flight, beside running steps or
    /// not, the running scratch-served chain, prompt group `chain`'s steps,
    /// where given: residency answers at its next
    /// turn, waiting on no flight ([`Residency::serve_evict`]). Nothing
    /// moves: each group keeps its most valuable experts where their bytes
    /// are, and evicts the rest, free records first. None where even the
    /// shallowest candidate keeps fewer records than those in flight,
    /// beside a chain it may not evict, or where a bank victim's exit
    /// writes it back, which holds its record until the copy ends.
    pub fn shrink(&self, evicting: Evicting, candidates: Candidates, chain: Option<ScratchGroup>) -> Result<Option<Evicted>, String> {
        match self.ask(Ask::Evict { evicting, candidates, chain })? {
            Answer::Evicted(evicted) => evicted,
            Answer::Done(_) => unreachable!("a shrink is answered by its evictions"),
        }
    }

    /// A growth's next slot is backed by every record tensor of its
    /// holding, beside running steps or not: residency takes it at its next
    /// turn, or the request service at its next pass, whichever is first
    /// (`Residency::join`), before either promotes, maps, lands or hints,
    /// and before any shrink asked after it is served. Nothing reaches its
    /// records until then: the device reaches a record only once the host
    /// maps it.
    pub(in crate::experts) fn join(&self, join: Join) {
        self.joins.post(join);
        self.events().notify();
    }

    /// The bank holds `words` for the places a group holds from now,
    /// between flights (`ExpertBank::hold_places`): every step retired, the
    /// watcher serves each place's scratch requests and the telemetry reads
    /// each record's group reads from them.
    pub(in crate::experts) fn place(&self, words: PlacedWords) -> Result<(), String> {
        self.ask(Ask::Place(words)).map(drop)
    }

    fn ask(&self, ask: Ask) -> Result<Answer, String> {
        let resizing = &self.resizing;
        let _ask = resizing.ask_lock.lock().expect("watcher resize caller");
        let post = |ask| {
            let mut resize = resizing.resize.lock().expect("watcher resize");
            assert!(matches!(*resize, Resize::None), "invariant violated: a resize is already pending in ResizeHandoff");
            *resize = Resize::Asked(ask, Instant::now());
        };
        match ask.waits() {
            true => self.steps.between_flights(|| post(ask))?,
            false => {
                post(ask);
                self.events().notify();
            }
        }
        // The watcher answers, or ends serving (`Shared::end_serving`), and
        // either notifies under this lock.
        let mut resize = resizing.resize.lock().expect("watcher resize");
        loop {
            if let Some(reason) = self.life.ended() {
                *resize = Resize::None;
                return Err(format!("the expert watcher stopped serving: {reason}"));
            }
            match std::mem::replace(&mut *resize, Resize::None) {
                Resize::Answered(answer) => return Ok(answer),
                asked => *resize = asked,
            }
            resize = resizing.resized.wait(resize).expect("watcher resize");
        }
    }
}

/// A resize the owner asked at `at`, taken by residency at `taken`, served
/// once nothing moves a record where it waits ([`Residency::resize`]).
pub(super) struct Asked {
    ask: Ask,
    at: Instant,
    taken: Instant,
}

impl Asked {
    /// `asked`, taken now.
    pub(super) fn taken((ask, at): (Ask, Instant)) -> Self {
        Self { ask, at, taken: Instant::now() }
    }
}

/// That a resize's changes are published ([`Residency::publish_resized`]),
/// which answering its owner takes: none answers with a map change the
/// device has not been given.
struct Published(());

/// What waits on what each layer holds being published, answered once
/// residency has published it as it is released
/// (`super::residency::Held`): a resize's answer, whose owner reads the
/// holdings it left, the joins applied, whose holdings follow them, and any
/// replaced `Records` views deferred to drop outside `residency` and
/// `NoBlockingAllocationGuard` before answering.
#[derive(Default)]
pub(super) struct Owed {
    answer: Option<(Answer, Published)>,
    joined: usize,
    pub(super) deferred_records: Vec<crate::experts::source::Records>,
}

impl Owed {
    /// Answer what waits, what each layer holds published and deferred views
    /// dropped outside `residency`.
    pub(super) fn settle(mut self, shared: &Shared) {
        if let Some((answer, published)) = self.answer.take() {
            shared.resizing.answer(answer, published);
        }
        if self.joined > 0 {
            shared.joins.applied(std::mem::take(&mut self.joined));
        }
    }
}

impl Residency {
    /// Take each slot a growth joined since ([`Shared::join`]), in order:
    /// the directory takes its new records (free bank records it then
    /// promotes into, or ready landed records, which file misses and the
    /// hints' reads land into), and the landings and the CPU site address
    /// landed records by their views. Whether it took any.
    pub(super) fn join(&mut self) -> bool {
        let joins = self.shared.joins.take();
        let count = joins.len();
        let mut coalesced: Vec<Join> = Vec::new();
        for join in joins {
            match (coalesced.last_mut(), join) {
                (Some(Join::Bank { records: acc }), Join::Bank { records }) => {
                    for (dst, src) in acc.iter_mut().zip(records) {
                        dst.extend(src);
                    }
                }
                (Some(Join::Landed { records: acc, views: acc_views }), Join::Landed { records, views }) => {
                    for (dst, src) in acc.iter_mut().zip(records) {
                        dst.extend(src);
                    }
                    let displaced = std::mem::replace(acc_views, views);
                    self.owed.deferred_records.extend(displaced.into_iter().flatten());
                }
                (_, join) => coalesced.push(join),
            }
        }
        for join in coalesced {
            match join {
                Join::Bank { records } => {
                    for (group, records) in records.into_iter().enumerate() {
                        self.core.directory.grow(group, records);
                    }
                }
                Join::Landed { records, views } => {
                    for (group, records) in records.into_iter().enumerate() {
                        self.core.directory.grow_host(group, records);
                    }
                    self.address_tier(views);
                }
            }
        }
        if count > 0 {
            self.owed.joined += count;
        }
        count > 0
    }

    /// Serve `asked`. One that waits ([`Ask::waits`]) is served between
    /// flights, every step ended, so the frontier is past the last flight
    /// and its steps retire, and only once nothing moves a record: the
    /// promotions in flight end (each copy's completion an event), then the
    /// reads in flight land (each landing's end an event), and a prompt
    /// group the device is past clears, returning the records it claimed;
    /// meanwhile nothing new is promoted or read ahead (`Residency::tend`).
    /// A shrink waits on nothing. The ask, while it waits; none once
    /// answered, or once serving ended (whose end wakes the owner).
    pub(super) fn resize(&mut self, asked: Asked, copier: Option<&super::link::Copier>) -> Option<Asked> {
        // Every slot a growth joined before the ask is the directory's.
        self.join();
        self.retire();
        if let Some(copier) = copier {
            self.link.settle(&mut self.core, &mut self.ledger, &self.shared, copier, self.prefetch.reached());
        }
        let _ = self.prefetch.settle(&mut self.core.directory, &mut self.ledger);
        if self.shared.life.ended().is_some() {
            return None;
        }
        // Whatever is asked, the cover's queued reads yield: a resize waits
        // for the reads that run, and a cut of the tier is deferred by the
        // landings in flight, never by a queue of reads ahead of use.
        self.prefetch.cancel_cover();
        if asked.ask.waits() {
            if self.core.directory.moving() || self.prefetch.outstanding() {
                return Some(asked);
            }
            // A prompt group the device is past holds its records claimed
            // until its conveyor clears (its landings from the conveyor end,
            // each an event).
            if self.scratch.clearing() && !self.scratch.clear_past(&mut self.core) {
                return Some(asked);
            }
        }
        let Asked { ask, at, taken } = asked;
        let trip = || Trip {
            picked: taken - at,
            served: taken.elapsed(),
        };
        let answer = match ask {
            Ask::Evict { evicting, candidates, chain } => Answer::Evicted(self.serve_evict(evicting, &candidates, chain, trip)),
            Ask::Place(PlacedWords { flags, layers }) => {
                let places = flags.len();
                self.scratch.place(flags);
                self.shared.telemetry.place(GroupReadWords { layers, places });
                Answer::Done(trip())
            }
        };
        let published = self.publish_resized();
        assert!(self.owed.answer.replace((answer, published)).is_none(), "a resize is answered before residency is released");
        None
    }

    /// Before the owner reads its answer, publish what a resize changed: the
    /// tier map of every record it evicted, so no step not yet begun
    /// resolves an expert at a hole; then, where no step runs, the policy
    /// words, the whole words learning which groups it left whole. The one
    /// way to the proof an answer takes.
    fn publish_resized(&mut self) -> Published {
        self.refill_tier();
        self.publish_policy();
        self.shared.write_staged();
        self.publish_lines(None);
        self.shared.telemetry.note_resized();
        Published(())
    }

    /// Evict `evicting`'s records at the deepest of `candidates` whose every
    /// group holds enough records no step writes or reads in place
    /// (`Directory::evictions`, `Directory::host_evictions`), moving
    /// nothing, beside running steps or not. Each evicted expert is unmapped
    /// from the first step not yet begun; beside the running scratch-served
    /// chain, prompt group `chain`'s steps, from the chain's first step
    /// ([`UnmapStep`]) where none of its requests taken so far routed it at
    /// its layer, so every step of the chain still to decide the layer's
    /// hits resolves it a miss and its rounds read it, as any cold expert.
    /// The steps before still read the
    /// rest where they are: the owner releases their storage once every
    /// pass that may read them is submitted (`Evicted::decided`).
    ///
    /// On every placement, beside running steps or not. While steps run, a
    /// group with a whole layer keeps its bank records: its runs read its
    /// experts at their homes without the map (`POLICY_WHOLE`), a word only
    /// a gap publishes anew, so none of its homes may leave under them; every
    /// other group, and a host tier's landed records (which no whole layer
    /// reads), evict as between flights. None beside a chain once a group
    /// began after it, whose steps decided their hits by the map before the
    /// eviction; nor where even the shallowest candidate keeps more records
    /// of some group than it may evict now; nor where a bank victim's exit
    /// writes it back (`Directory::evict`), whose record the next boundary
    /// evicts once the copy ends.
    fn serve_evict(&mut self, evicting: Evicting, candidates: &Candidates, chain: Option<ScratchGroup>, trip: impl Fn() -> Trip) -> Result<Option<Evicted>, String> {
        let running = !self.shared.steps.idle();
        let kept = (0..self.core.groups.len())
            .map(|group| {
                evicting == Evicting::Bank
                    && running
                    && self.core.layers.iter().enumerate().any(|(index, layer)| layer.group == group && self.shared.telemetry.is_whole(index))
            })
            .collect::<Vec<_>>();
        if chain.is_some_and(|chain| !self.shared.steps.latest(chain.last())) {
            return Ok(None);
        }
        // Ranked by every step that ended before it, fed by `Residency::keep`
        // outside the residency lock immediately before this turn.
        let ranking = self.core.directory.ranking();
        let last_read = |layer: usize, expert: usize| Reuse::of(&ranking).last(layer, expert);
        let groups = self.core.groups.len();
        let directory = &self.core.directory;
        let held = |group: usize| match evicting {
            Evicting::Bank => directory.records()[group],
            Evicting::Landed => directory.host(group).records() as usize,
        };
        let evictable = |group: usize, target: usize| match evicting {
            Evicting::Bank if kept[group] => Vec::new(),
            Evicting::Bank => directory.evictions(group, target, last_read).into_iter().map(|record| u32::try_from(record).expect("bank records fit u32")).collect::<Vec<_>>(),
            Evicting::Landed => directory.host_evictions(group, u32::try_from(target).expect("host records fit u32")),
        };
        let most = (0..groups).map(|group| evictable(group, 0).len()).collect::<Vec<_>>();
        let Some((slots, targets)) = deepest(candidates, |group, records| held(group).saturating_sub(records) <= most[group], groups) else {
            return Ok(None);
        };
        let evicted = targets.iter().enumerate().map(|(group, &target)| evictable(group, target)).collect::<Vec<_>>();
        // An expert the chain's steps routed at its layer stays mapped for
        // them, unmapped from the first step not yet begun; any other leaves
        // from the chain's first step.
        let scratch = &self.scratch;
        let leaves_first = |(layer, expert): (usize, usize)| chain.filter(|chain| !scratch.routed(chain.first(), layer, expert)).map(UnmapStep::from);
        match evicting {
            Evicting::Bank => {
                let (directory, steps) = (&mut self.core.directory, &self.core.steps);
                let records = evicted.iter().map(|records| records.iter().map(|&record| record as usize).collect::<Vec<_>>()).collect::<Vec<_>>();
                // A victim whose exit writes it back holds its record until
                // the copy ends: the shrink waits a boundary for it.
                if !steps.unbegun(|at| directory.evict(&records, last_read, |staged| leaves_first(staged).map_or(at.into(), |first| first.min(at.into())), at)) {
                    return Ok(None);
                }
            }
            Evicting::Landed => {
                // The views without the evicted records are formed first, so a
                // refusal changes nothing.
                let views = self.core.tier_views(&evicted)?;
                let directory = &mut self.core.directory;
                for (group, records) in evicted.iter().enumerate() {
                    directory.evict_host(group, records, leaves_first);
                }
                self.address_tier(views);
            }
        }
        let decided = chain.and_then(|chain| self.scratch.decided(chain.first()));
        Ok(Some(Evicted { slots, records: evicted, decided, trip: trip() }))
    }
}

/// The deepest of `candidates` whose records each of the `groups` groups
/// may evict down to (`fits(group, records)`): its slots and each group's
/// records.
fn deepest(candidates: &Candidates, fits: impl Fn(usize, usize) -> bool, groups: usize) -> Option<(usize, Vec<usize>)> {
    (0..candidates.count()).rev().map(|index| candidates.slots(index)).find_map(|slots| {
        let records = (candidates.records)(slots);
        assert_eq!(records.len(), groups, "a candidate names each group's records");
        records.iter().enumerate().all(|(group, &records)| fits(group, records)).then_some((slots, records))
    })
}
