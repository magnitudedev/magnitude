//! Residency: the one owner of where experts live and what moves them. It
//! holds the directory (every map entry's write, through the steps' lock,
//! `experts::map`), the host tier and its mirror in each layer's tier map,
//! staging and the landings into the tier, the conveyor and its scratch
//! tiers, the promotions into the bank and the policy words, behind one
//! lock whose every critical section is short: no method waits for
//! anything.
//!
//! Two threads act through it. The request service (`requests`) takes the
//! device's requests and answers them from it. Residency's own thread
//! ([`Residency::keep`]) does what no request asks: it feeds the ended
//! steps' reads to the ranking (outside the lock), retires the steps the
//! frontier passed, learns the landings that ended, keeps the tier's
//! map and reserve, carries the promotions, clears a prompt group's
//! conveyor once nothing of it runs, fills the tier ahead of every step,
//! joins a growth's slots, serves a resize and writes the policy words in
//! a gap. Each turn of it runs on the load's events (`events`): a step's
//! end (the frontier), a landing's or a read's end, a copy's completion, a
//! growth joined, a resize asked, the request service's progress. It never
//! waits on a time, and no job it does waits for another.

use super::core::Core;
use super::decode;
use super::prefetch::Prefetch;
use super::promotions::{Copier, Promotions};
use super::resize::Asked;
use super::scratch::Scratch;
use super::{Shared, panic_message};
use crate::assessment::{ExpertLines, SiteLines};
use crate::experts::directory::Point;
use magnitude_kernels::constants;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// Where experts live and what moves them, behind one lock
/// ([`lock`]).
pub(super) struct Residency {
    /// Per layer, whether its group was whole when the words were last
    /// published (`Residency::publish_policy`).
    pub(super) whole: Vec<bool>,
    /// Per layer, its home as its policy names it (`policy[POLICY_HOME]`),
    /// the directory's ([`Directory::home`](crate::experts::directory::Directory::home)).
    pub(super) homes: Vec<i32>,
    pub(super) shared: Arc<Shared>,
    pub(super) core: Core,
    pub(super) prefetch: Prefetch,
    pub(super) scratch: Scratch,
    /// What sizes the promotions' reserves.
    pub(super) promotions: Promotions,
    /// A resize asked, waiting until nothing moves a record.
    pub(super) asked: Option<Asked>,
    /// When residency opened, and whether the host tier keeps every expert
    /// (`Directory::tier_complete`), as last published.
    pub(super) opened: Instant,
    pub(super) complete: bool,
    /// The CPU site's answered requests and their path time when residency
    /// opened, and the site's share of the time over the load as its last
    /// answered request left it ([`Self::site_share`]).
    pub(super) site_opened: crate::experts::SiteServed,
    pub(super) site_share: SiteShare,
    /// Each layer's line words as last published ([`Self::publish_lines`]);
    /// none before the first.
    pub(super) lines: Option<Vec<Vec<(usize, i32)>>>,
}

/// Residency's lock. A panic while it is held is caught inside it, and
/// serving ends before it is released (`Residency::keep`,
/// `Requests::serve`): no thread serves from the state it left, which only
/// the refusals read (`liveness::refuse`), so a poisoned lock is never
/// observed. Every holder finds a shrink plan its owner dropped uncommitted
/// undone already ([`Residency::unshrink`]): no holder, whichever thread,
/// meets the records it held.
pub(super) fn lock(residency: &Mutex<Residency>) -> MutexGuard<'_, Residency> {
    let mut held = residency.lock().expect("residency is never left by a panic");
    held.unshrink();
    held
}

impl Residency {
    /// Residency's thread: a turn ([`Self::tend`]) on every event of the
    /// load, until it stops. Once serving ended (a panic, here or in the
    /// request service), it does nothing more: the request service refuses
    /// what the device waits on, and the owner waiting for a resize was
    /// woken with why.
    /// Promotions copy by `copier` (none where the bank has no host tier),
    /// which this thread alone holds.
    pub(super) fn keep(residency: Arc<Mutex<Self>>, shared: Arc<Shared>, copier: Option<Copier>) {
        let events = shared.events().clone();
        // The decode steps the reserve was last sized at.
        let mut decoded_steps = 0;
        loop {
            let seen = events.seen();
            if shared.life.stopping() {
                return;
            }
            if shared.life.ended().is_none() {
                // The ended groups' reads, fed to the ranking before the turn
                // ranks by it, and the decode steps' depth curve where a step
                // was read since (which sizes the promotions' reserve),
                // outside residency's lock so no request waits on the stack.
                let fed = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    shared.telemetry.feed();
                    copier.as_ref().and_then(|_| shared.telemetry.decoded_since(&mut decoded_steps))
                }));
                let mut held = lock(&residency);
                if let Err(payload) =
                    fed.and_then(|decoded| std::panic::catch_unwind(AssertUnwindSafe(|| held.tend(copier.as_ref(), decoded.as_ref()))))
                {
                    shared.end_serving(format!("it panicked: {}", panic_message(payload.as_ref())));
                }
            }
            events.wait(seen);
        }
    }

    /// One turn, none of whose jobs waits: a growth's slots joined; the
    /// steps the frontier passed retired; the landings that ended learned
    /// and the tier map published; then a resize asked, served once the
    /// promotions and landings in flight end (until then nothing new is
    /// promoted or read ahead); else the promotions asked and carried, a
    /// prompt group past cleared, what each layer holds published, the tier
    /// filled ahead of the steps, the line words written where they
    /// changed, and in a gap the policy words a run's encoding reads. A
    /// turn that changed what a request is served by (a retirement, a
    /// landing learned, a promotion, a group cleared, a growth joined) is
    /// an event: the request service looks again.
    fn tend(&mut self, copier: Option<&Copier>, decoded: Option<&crate::experts::directory::MissCurve>) {
        // A panic of the executor's work, or the device lost under the
        // parks, ends serving, as residency's own panic does.
        if let Some(reason) = self.shared.fault() {
            self.shared.end_serving(reason);
            return;
        }
        let joined = self.join();
        let retired = self.retire();
        let landed = self.prefetch.settle(&mut self.core.directory);
        self.refill_tier();
        if let Some(asked) = self.asked.take().or_else(|| self.shared.resizing.asked().map(Asked::taken)) {
            self.asked = self.resize(asked, copier);
            return;
        }
        // A shrink planned and not yet committed holds its records: nothing
        // is promoted or released until it commits or is dropped.
        // Held for a measurement beside the idle load, the copies running
        // end and none is asked; a step running ends the hold at once, so
        // no request waits on a measurement (the service preempts the
        // measurement as a request arrives, and serving outranks it here
        // until it has).
        let idle = self.shared.steps.idle();
        let held = self.shared.traffic_held() && idle;
        let promoted = !self.core.directory.shrink_pending()
            && copier.is_some_and(|copier| self.promotions.advance(&mut self.core, &self.shared, copier, decoded, held));
        let cleared = self.scratch.clear_past(&mut self.core, &mut self.prefetch);
        if retired || landed || promoted {
            self.publish_held();
        }
        if !held {
            self.read_ahead();
        }
        let copying = copier.is_some_and(|copier| !copier.copies.idle());
        // Its turning quiet is an event: a hold waits on it.
        let quiet = held && !copying && !self.prefetch.outstanding();
        if self.shared.traffic_quiet.swap(quiet, std::sync::atomic::Ordering::AcqRel) != quiet {
            self.shared.events().notify();
        }
        let steps = self.shared.steps.clone();
        // The words a run's encoding reads only while no step runs; the
        // lines at every turn.
        steps.while_idle(|| self.publish_policy());
        self.publish_lines();
        if joined || retired || landed || promoted || cleared {
            self.shared.events().notify();
        }
    }

    /// Retire through the frontier (`Steps::frontier`): the device is past
    /// every group whose every step is below it; and publish what each layer
    /// holds then. Whether the frontier moved.
    pub(super) fn retire(&mut self) -> bool {
        let frontier = self.shared.steps.frontier();
        if frontier <= self.core.retired {
            return false;
        }
        self.core.retired = frontier;
        let (group, reserving) = self.shared.steps.retire_groups(frontier);
        if let Some(group) = group {
            self.advance(Point::after(group));
        }
        for _ in 0..reserving {
            self.shared.lent.release();
        }
        self.publish_held();
        true
    }

    /// The device is at `progress`: learn the landings that ended (a
    /// landing ends before it answers, so the device past an answer proves
    /// its landing ended), retire the seeds and tier graces before it, free
    /// the records taken before it, and the conveyor's (staging what a
    /// group past held whole); then publish the tier map's changes. What a
    /// request at `progress` sees follows from this alone, so every request
    /// is taken or served only after it: none acts on what its own position
    /// proves past, however late residency learned of it.
    pub(super) fn advance(&mut self, progress: Point) {
        self.prefetch.past(progress);
        self.prefetch.settle(&mut self.core.directory);
        for listed in self.core.take_past(progress) {
            listed.release(&mut self.scratch.conveyor);
        }
        // The tiers it is past place their seeds, which it is past too.
        self.scratch.advance(progress, &mut self.core, &mut self.prefetch);
        self.core.retire(progress);
        self.refill_tier();
    }

    /// Keep each group's reserve of tier records ready or graced: the
    /// victims chosen from the ranking with no lock of the steps' held
    /// (`Directory::refill_victims`), then, where no step begins, each
    /// unmapped, unchanged since (`Directory::refill`), graced through the
    /// latest step begun; a lent record only while no prompt group holds or
    /// awaits the lent records (which a group reserves as it begins,
    /// `lent`), so none is graced through a group's own step. Under the
    /// steps' lock: a check, a map store and a state change a victim.
    pub(super) fn refill_tier(&mut self) {
        let (steps, lending) = (&self.shared.steps, &self.shared.lent);
        let reserve = self.core.extents.reserve;
        let directory = &mut self.core.directory;
        if !directory.refill_short(reserve) {
            return;
        }
        let victims = directory.refill_victims(reserve, !lending.lock().reserved());
        if victims.iter().all(Vec::is_empty) {
            return;
        }
        steps.while_none_begins(|latest, at| {
            let lent = !lending.lock().reserved();
            directory.refill(victims, lent, at, latest);
        });
        // A grace through a step the device is already past frees now: no
        // frontier's move would come for it until the next step.
        directory.retire_graces(self.prefetch.reached());
    }

    /// Take `layer`'s scratch requests published since, in step order,
    /// reading what each needs as its records free; none while the conveyor
    /// still clears a group past (`Scratch::clearing`), whose clearing is an
    /// event. True when it took any.
    pub(super) fn publish(&mut self, layer: usize) -> bool {
        // The device's requests are in place only while it is parked, where
        // parks are submission boundaries (`Parks::parked`).
        if !self.shared.parks.parked() {
            return false;
        }
        let layers = self.core.layers.len();
        let mut took = false;
        while let Some((place, at)) = self.scratch.published(&self.core, &self.shared, layer, layers) {
            // A request is its step's front's, which follows its step's back
            // at the layer before (the group's steps before it there too):
            // the device is past them, or past the groups before at the
            // first layer.
            let Point { group, sequence, .. } = at.point;
            self.advance(match layer {
                0 => Point { sequence: group, ..at.point },
                _ => Point {
                    group,
                    layer: layer - 1,
                    sequence: sequence + 1,
                },
            });
            if self.scratch.clearing() {
                break;
            }
            self.scratch.publish(at, place, &mut self.core, &self.shared);
            took = true;
        }
        took
    }

    /// Publish the experts each layer holds to the size rule, and those its
    /// group's host tier holds alone.
    pub(super) fn publish_held(&mut self) {
        let held = (0..self.core.layers.len())
            .map(|layer| self.core.directory.held_count(layer))
            .collect::<Vec<_>>();
        let alone = (0..self.core.layers.len())
            .map(|layer| self.core.directory.alone(layer))
            .collect::<Vec<_>>();
        self.shared.telemetry.publish_held(&held, &alone);
        // The moment the records misses land into first keep every expert:
        // from then no decode miss reads the file (`Prefetch::fill`). They
        // are a host tier's where the device places, the bank's own (the one
        // store) where the host does.
        let complete = self.core.directory.tier_complete();
        if complete && !self.complete {
            tracing::info!(
                "{} keeps every expert, {:.1} s after the bank opened",
                self.store(),
                self.opened.elapsed().as_secs_f64()
            );
        }
        self.complete = complete;
    }

    /// The store misses land into: a host tier's where the device places,
    /// the bank's own (the one store) where the host does.
    fn store(&self) -> &'static str {
        match self.core.extents.placer {
            crate::planning::Placer::Tier => "the host tier",
            crate::planning::Placer::Bank => "the bank",
        }
    }

    /// The CPU site's share of the time over the load: its answered
    /// requests' path time since residency opened over the time from then
    /// to the turn that reads them, at most all of it (requests of several
    /// layers overlap); none of it before any is answered. It moves only as
    /// requests are answered, so a turn with none answered since prices the
    /// copies as the turn before did and asks no promotions again. No time
    /// since residency opened holds no share: the share stands as it was,
    /// and the first turn after reads the requests.
    fn site_share(&mut self) -> f64 {
        let served = self.shared.compute.site_served().since(&self.site_opened);
        let elapsed = self.opened.elapsed();
        if served.requests != self.site_share.requests && !elapsed.is_zero() {
            let path = served.terms.iter().sum::<std::time::Duration>();
            self.site_share = SiteShare { requests: served.requests, share: (path.as_secs_f64() / elapsed.as_secs_f64()).min(1.) };
        }
        self.site_share.share
    }

    /// Between flights, write each layer's policy words a run's encoding
    /// reads (`BankPart::Policy`, this their one writer; its line words are
    /// [`Self::publish_lines`]'): its home (`Directory::home`, the same every
    /// flight, which a graph shared by blocks reads through each block's
    /// own policy port), its whole word, set
    /// while its group is whole and at home, no expert of it able to miss
    /// (`Directory::group_whole`: on the one store, its records the bank's,
    /// while it keeps every expert, as a bank imported whole there does) and
    /// each at its home record (`Directory::group_at_home`): its route then
    /// writes no routing record, so the ranking is fed nothing of it and
    /// starts blank after a shrink,
    /// so a prompt group of whole layers is resident and lends nothing; and
    /// the group that lends the conveyor its records only while every group
    /// is (a partial group's prompt demand claims the lender's records
    /// mid-flight,
    /// `Directory::claim_lent`). Nothing misses in a whole group, so no
    /// placement, seed or claim takes its records before the next flight
    /// ends, and every run encoded and launched until then reads the word
    /// alike. Called only while no step runs, under the lock every step
    /// begins under (`Steps::while_idle`), and a step begins before its
    /// runs are encoded, so no encoding overlaps a publication. A placement
    /// counts once it holds its record: a group's word turns on a flight
    /// late, never early. Its resolved word is set while it is whole or its
    /// group's host tier keeps every expert outside the conveyor's lent
    /// records (`HostRecords::complete`): no route of its can find an
    /// expert absent, so its tier wait is not encoded. The tier gives up a
    /// kept expert mid-flight only to land one it lacks, which a complete
    /// tier never does, or to a prompt group's claim of a lent record,
    /// which a complete tier keeps none in; otherwise only in a shrink
    /// between flights, after which this publishes.
    pub(super) fn publish_policy(&mut self) {
        let directory = &self.core.directory;
        let full = (0..self.core.groups.len()).map(|group| directory.group_whole(group) && directory.group_at_home(group)).collect::<Vec<_>>();
        let every = full.iter().all(|&full| full);
        let lender = self.scratch.conveyor.lending().group;
        let served = self.core.layers.iter().any(|layer| layer.site.is_some() && !(full[layer.group] && (every || lender != layer.group)));
        self.shared.sites_served.store(served, std::sync::atomic::Ordering::Release);
        for (index, layer) in self.core.layers.iter().enumerate() {
            let whole = full[layer.group] && (every || lender != layer.group);
            self.whole[index] = whole;
            self.shared.telemetry.whole(index, whole);
            layer.policy.store(constants::POLICY_WHOLE as usize, i32::from(whole));
            let resolved = whole || self.core.directory.group_tier_complete(layer.group);
            layer.policy.store(constants::POLICY_RESOLVED as usize, i32::from(resolved));
            layer.policy.store(constants::POLICY_HOME as usize, self.homes[index]);
        }
    }

    /// Write each layer's line words (`ExpertLines::policy`, this the one
    /// writer) from its lines as the load's own work refines them (its
    /// landings, what its CPU site's requests take and what its sampled
    /// steps' blocks take on the device, `Telemetry::refined`), and price
    /// the promotions' copies on the same lines at the site's share of the
    /// time now (`Directory::price_copies` asks them again only where a
    /// price moved). The words are stored only where they differ from those
    /// last published, so a turn that moved none of them takes no steps'
    /// lock. Unlike the words a run's encoding reads (whole, resolved, home:
    /// [`Self::publish_policy`]), the lines decide
    /// only the route's choice of a miss's site, and any value of each word
    /// is a valid one for any step: so they are stored under the lock every
    /// step begins under (`Steps::unbegun`), as a map entry is
    /// (`experts::map`), at any turn, the lock held for the stores alone.
    /// Every step from the first not yet begun reads them, and one begun
    /// that has not routed the layer yet may, as a mapping may take effect
    /// from such a step: word by word, some of a layer's words old and
    /// some new. A decode that never idles is repriced so.
    pub(super) fn publish_lines(&mut self) {
        let priced = self.shared.priced().map(|lines| self.shared.telemetry.refined(&lines));
        let copy = self.shared.copy.lock().expect("a copy's price").clone();
        if let (Some(lines), Some(copy)) = (&priced, &copy) {
            let site_share = self.site_share();
            for (layer, lines) in lines.iter().enumerate() {
                self.core.directory.price_copies(layer, lines.copy_hits(copy, site_share));
            }
        }
        let words = self
            .core
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                // A priced line holds a site's CPU lines once its verdict
                // admits it (`ExpertMeasures::lines`).
                let cpu = priced.as_ref().and_then(|lines| lines[index].host.cpu);
                let site = match (&layer.site, &cpu) {
                    (None, _) => SiteLines::Unsited,
                    (Some(_), None) => SiteLines::Pending,
                    (Some(_), Some(cpu)) => SiteLines::Measured(cpu),
                };
                ExpertLines::policy(site, priced.as_ref().map(|lines| &lines[index]))
            })
            .collect::<Vec<_>>();
        if self.lines.as_ref() == Some(&words) {
            return;
        }
        let layers = &self.core.layers;
        self.shared.steps.unbegun(|_| {
            for (layer, words) in layers.iter().zip(&words) {
                for &(word, value) in words {
                    layer.policy.store(word, value);
                }
            }
        });
        self.lines = Some(words);
    }

    /// What residency holds and moves, for the exchange's watchdog:
    /// staging and the reads and landings in flight, the promotions, the
    /// scratch tiers and conveyor, and per layer the experts its bank holds
    /// and those its group's host tier holds alone.
    pub(super) fn describe(&self) -> String {
        let layers = (0..self.core.layers.len())
            .map(|layer| (self.core.directory.held_count(layer), self.core.directory.alone(layer)))
            .collect::<Vec<_>>();
        format!(
            "{}; promotions {}; {}; tier keeps every expert {}; retired through {}; records taken {}; resize asked {}; \
             per layer (bank holds, tier holds alone) {layers:?}",
            self.prefetch.describe(),
            self.core.directory.describe_promotions(),
            self.scratch.describe(),
            self.complete,
            self.core.retired,
            self.core.taken.len(),
            self.asked.is_some(),
        )
    }

    /// Whether every layer is whole as last published: nothing a decode
    /// step runs then asks anything (no miss, no hint).
    pub(super) fn whole(&self) -> bool {
        self.whole.iter().all(|&whole| whole)
    }

    /// Read ahead of the steps' own reads and hints: the tier's fill, every
    /// expert the conveyor carries left to it (`Prefetch::fill`), and each
    /// fill's read a prompt group's step waits for promoted to demand
    /// (`Scratch::awaited`); then publish whether the fill still lands the
    /// tier (`Prefetch::filling`).
    pub(super) fn read_ahead(&mut self) {
        let scratch = &self.scratch;
        self.prefetch.fill(&mut self.core, &self.shared, |staged| scratch.carries(staged));
        for staged in self.scratch.awaited(&self.core) {
            self.prefetch.hasten(staged);
        }
        let fill = match self.prefetch.filling(&self.core) {
            true => crate::TierFill::Filling,
            false => crate::TierFill::Filled,
        };
        if self.shared.publish_fill(fill) && fill == crate::TierFill::Filled {
            tracing::info!(
                "{} is filled, {:.1} s after the bank opened",
                self.store(),
                self.opened.elapsed().as_secs_f64()
            );
        }
    }

    /// Each group's host tier records `views` address from now (a shrink's,
    /// between flights, or a growth's joined slot): the landings' tier
    /// tensors and the CPU site's, which read the same records.
    pub(super) fn address_tier(&mut self, views: Vec<Vec<crate::experts::source::Records>>) {
        self.shared.compute.address(&views);
        self.core.address_tier(views);
    }

    /// What decode serving reads and writes of it (`decode::Serving`).
    pub(super) fn serving(&mut self) -> decode::Serving<'_> {
        decode::Serving { core: &mut self.core, prefetch: &mut self.prefetch, shared: &self.shared }
    }
}

/// The CPU site's share of the time over the load ([`Residency::site_share`]),
/// as of its answered requests since residency opened.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SiteShare {
    requests: u64,
    share: f64,
}
