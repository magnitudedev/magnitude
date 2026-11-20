//! The host's promotions into the bank:
//! background work, off every step's path, on the device's copy engine.
//! None is triggered by a miss: the one ranking decides which of the host
//! tier's experts enter the bank and which records they take
//! (`Directory::promotions`: the experts its pool of each group holds and
//! the bank lacks, into free records), as the completed steps feed it
//! reads, and a copy competes with the step for the link and host memory,
//! so an expert enters only once its reads pay for its copy. Each promotion
//!
//! 1. holds the tier record keeping its expert (`HostRecords::hold`) and
//!    copies it into its free record on the copy engine
//!    (`Copies::post_within`), each its own batch, in the order they were
//!    queued (`Directory::queued`), while fewer than the transfer queue's
//!    measured depth are in flight (`TierCopyCosts::depth`): the link stays
//!    full, and a copy a route waits for waits behind that depth alone;
//! 2. once the copy has completed, maps its expert to the record from the
//!    map rule's step for its layer, so a step in flight that has yet to
//!    resolve the layer reads it from the bank, and releases the tier
//!    record.
//!
//! A copy earns by its expert's reuse alone (`Directory::earning`): a
//! later route's read of it in place over the link meanwhile is the
//! request service's, not a promotion's.
//!
//! The tiers are disjoint: a promoted expert's tier record is given up
//! once its copy has completed (`HostRecords::surrender`), ready for
//! another expert.
//!
//! A record is freed apart from any promotion, by a release
//! (`Directory::promotions`, `Release`): its expert, outside the group's
//! pool, is unmapped for every step from the first not yet begun, so the
//! steps already queued may still read it, and the record is free once
//! every one it was mapped for has passed its layer in the device's order
//! or ended, by whichever exit (`Steps::served`), and once its write-back
//! has ended: where keeping it in the host tier pays, the copy engine
//! writes it from the record into a tier record its landing took, the
//! reverse of a promotion, queued beside the promotions and posted before
//! them (each frees a record a promotion waits for), and the tier keeps
//! it alone once written (`Directory::written_back`). Each group keeps a reserve of free
//! or releasing records ([`reserve`]): the promotions the ranking asks
//! while a release drains, so a promotion's copy never waits for one.
//!
//! Each promotion's and release's stage is its record's
//! (`experts::records`), whose transitions are the only way it moves.
//!
//! So a promotion takes effect from the first step not yet begun once its
//! copy has completed. Where the bank has no host tier (unified memory)
//! there is nothing to promote: landings land into the bank itself.

use super::Shared;
use super::core::Core;
use super::prefetch::Prefetch;
use crate::assessment::TierCopyCosts;
use crate::experts::directory::{BankRecord, MissCurve, Point};
use crate::experts::horizon::{Hold, Tokens};
use crate::experts::staging::Staged;

/// What promotions copy with: the device's copy engine. Destination bank
/// record views are held in `Core::groups[group].bank` (delivered at open and
/// via `Join::Bank`), so promotions never touch `Stores` or `Pool`.
pub(in crate::experts) struct Copier {
    pub copies: crate::experts::copies::Copies,
}

/// What a reserve is sized within ([`reserve`]): each layer's group and
/// the most experts a decode step reads there (its experts, at most a
/// step's choices), each group's capacity (`Directory::capacity`),
/// and the steps in flight.
struct Sizing {
    groups: Vec<usize>,
    reads: Vec<usize>,
    records: Vec<usize>,
    flight: usize,
}

/// A candidate reserve of a group ([`reserve`]): its records, and what it
/// saves a step less what it costs, in misses.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Reserve {
    records: usize,
    net: f64,
}

/// Each group's reserve, where what it saves exceeds what it costs most,
/// from the decode steps' depth curve `decoded` (every decode step read
/// since the load) and `drain`, the mean steps a decode release waited.
///
/// Taking `ρ` slots off each of a group's `M` layers' whole `w` (its
/// records over its layers, the pool's depth in slots) reserves
/// `r = left + ρM` (`left`, the records `w·M` leaves over, at no depth
/// anyway). It costs the hits those records held: `Σ_l (m_l(w−ρ) −
/// m_l(w))` a step, `m_l(s)` layer `l`'s misses a step at `s` slots. It
/// saves each promotion it covers the drain's wait: the ranking asks `a =
/// Σ_l m_l(w)` promotions a step, a drain's worth `a·drain` of them find a
/// free record where `r` does, and each would have been read at the rate
/// of the record atop its layer's stack, `p = mean_l m_l(0) − m_l(1)`, for
/// each of the `drain` steps it waited: `min(a·drain, r)·p` a step. `ρ`
/// stops a slot short of the whole `w`, so the pool keeps one a layer.
/// None before a decode release or step is measured. A measure no load can
/// produce panics: more misses than reads, or a drain past the flight.
fn reserve(sizing: &Sizing, decoded: &MissCurve, drain: Option<f64>) -> Vec<Vec<Reserve>> {
    let groups = sizing.records.len();
    let (Some(drain), true) = (drain, decoded.steps > 0) else {
        return (0..groups).map(|_| vec![Reserve { records: 0, net: 0. }]).collect();
    };
    assert!(drain <= sizing.flight as f64, "a release waited {drain} steps, beyond the {} in flight", sizing.flight);
    let steps = decoded.steps as f64;
    (0..groups)
        .map(|group| {
            let members = (0..sizing.groups.len()).filter(|&layer| sizing.groups[layer] == group).collect::<Vec<_>>();
            let whole = sizing.records[group] / members.len().max(1);
            let left = sizing.records[group] - whole * members.len();
            let misses = |layer: usize, slots: usize| decoded.layers[layer].at(slots).1 as f64 / steps;
            for &layer in &members {
                let most = sizing.reads[layer] as f64;
                assert!(misses(layer, 0) <= most, "{} reads a step of a layer reading at most {most}", misses(layer, 0));
            }
            let asked = members.iter().map(|&layer| misses(layer, whole)).sum::<f64>() * drain;
            let rate = members.iter().map(|&layer| misses(layer, 0) - misses(layer, 1)).sum::<f64>() / members.len().max(1) as f64;
            (0..whole.max(1))
                .map(|levels| {
                    let records = left + levels * members.len();
                    let cost = members.iter().map(|&layer| misses(layer, whole - levels) - misses(layer, whole)).sum::<f64>();
                    Reserve { records, net: asked.min(records as f64) * rate - cost }
                })
                .collect()
        })
        .collect()
}

/// Of a group's reserves ([`reserve`]), the one to hold now, `current`
/// held: the best, where its gain over `current` a token (per-step net over
/// `tokens_per_step`), over `hold` at clock `now` ([`Hold::pays`], design
/// §3.6), repays moving the records between them (a promotion or a release
/// each, a miss each); else `current`. A `current` no longer among the
/// candidates (its group's records changed) is left for the best.
fn decide(candidates: &[Reserve], current: usize, hold: &Hold, now: Tokens, tokens_per_step: f64) -> usize {
    // The best, the fewer records on a tie.
    let best = candidates
        .iter()
        .copied()
        .max_by(|a, b| a.net.total_cmp(&b.net).then(b.records.cmp(&a.records)))
        .expect("a reserve's candidates");
    let Some(candidate) = candidates.iter().find(|candidate| candidate.records == current) else {
        return best.records;
    };
    let gain = (best.net - candidate.net) / tokens_per_step;
    let moved = best.records.abs_diff(current) as f64;
    match hold.pays(now, gain, moved) {
        true => best.records,
        false => current,
    }
}

/// What sizes the reserves: the steps the releases so far waited for, and
/// their count, and each group's reserve changes, in clock tokens
/// ([`Hold`]).
/// What promotions copy with is residency's thread's alone
/// (`Residency::keep`), passed to each turn: the request service, which
/// shares the records under residency's lock, reaches no transfer.
pub(super) struct Promotions {
    drained: (u64, u64),
    holds: Vec<Hold>,
}

impl Promotions {
    pub(super) fn new() -> Self {
        Self { drained: (0, 0), holds: Vec::new() }
    }

    /// One turn: size each group's reserve by `decoded`, the decode steps'
    /// depth curve where a step was read since the last turn ([`reserve`]),
    /// take what the ranking wants now (`Directory::promotions`, in time
    /// proportional to what changed), then carry every promotion and release
    /// in flight as far as it goes ([`carry`]), those just asked among
    /// them: no event follows an ask, so what it lets move moves now.
    /// Whether any moved. Where the load's traffic is `held` nothing is
    /// asked: only what is in flight is carried.
    /// None is asked before the transfer queue's line is measured
    /// ([`line`]): a copy is priced and paced by it alone.
    pub(super) fn advance(&mut self, core: &mut Core, shared: &Shared, copier: &Copier, decoded: Option<&MissCurve>, held: bool, prefetch: &Prefetch) -> bool {
        if let Some(decoded) = decoded {
            self.size(core, decoded);
        }
        let Some(copy) = line(core, shared) else {
            return false;
        };
        let asked = !held && self.ask(core, shared, copier, &copy);
        asked | carry(core, shared, copier, prefetch.reached())
    }

    /// Size each group's reserve by `decoded` ([`reserve`], [`decide`]).
    fn size(&mut self, core: &mut Core, decoded: &MissCurve) {
        let choices = magnitude_kernels::constants::VERIFY_ROWS as usize * core.extents.selected;
        let sizing = Sizing {
            groups: core.directory.layer_groups(),
            reads: core.directory.layer_experts().into_iter().map(|experts| experts.min(choices)).collect(),
            records: core.directory.capacities(),
            flight: core.extents.words.flight as usize,
        };
        let drain = (self.drained.1 > 0).then(|| self.drained.0 as f64 / self.drained.1 as f64);
        let (now, spacing) = {
            let times = core.directory.time().lock();
            let now = times.clock.through_latest();
            let spacing = (0..sizing.records.len()).map(|group| times.tokens_per_step(group)).collect::<Vec<_>>();
            (now, spacing)
        };
        // A group's reserve is first chosen at the first sizing that sees it.
        self.holds.resize(sizing.records.len(), Hold::started(now));
        for (group, candidates) in reserve(&sizing, decoded, drain).into_iter().enumerate() {
            let Some(tokens_per_step) = spacing[group] else {
                continue;
            };
            let current = core.directory.reserved(group);
            let records = decide(&candidates, current, &self.holds[group], now, tokens_per_step);
            if records != current {
                core.directory.reserve(group, records);
                self.holds[group].changed(now);
            }
        }
    }

    /// What the ranking asks now (`Directory::promotions`), capped to the
    /// copy engine's remaining in-flight headroom: each promotion
    /// to copy, and each release's expert unmapped from the first step not
    /// yet begun, its wait in steps measured. Whether it asked any. The
    /// ranking's changes settle first, outside the steps' lock, and a turn
    /// that stirred nothing (no want, record or lease changed) takes no
    /// steps' lock.
    fn ask(&mut self, core: &mut Core, shared: &Shared, copier: &Copier, copy: &TierCopyCosts) -> bool {
        let max_depth = record_bytes(core)
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| copy.depth(bytes))
            .max()
            .unwrap_or(0);
        let pending = copier.copies.flight() + core.directory.queued().len() + core.directory.backs_queued().len();
        let budget = max_depth.saturating_sub(pending);
        if budget == 0 && !core.directory.has_surrenders() {
            return false;
        }
        if !core.directory.settle_changes() {
            return false;
        }
        let decoding = !core.steps.prompt_running();
        // Asked under the steps' lock: each release's expert unmapped from
        // the first step not yet begun.
        let (directory, drained) = (&mut core.directory, &mut self.drained);
        let asked = core.steps.unbegun_beside_frontier(|frontier, at| {
            let asked = directory.promotions(at, budget);
            if decoding {
                let releases = asked.releases.len() as u64;
                *drained = (drained.0 + releases * (at.next() - frontier), drained.1 + releases);
            }
            asked
        });
        shared.promoted.store(directory.asked(), std::sync::atomic::Ordering::Relaxed);
        !asked.promotions.is_empty() || !asked.releases.is_empty()
    }
}

/// Carry each promotion and release as far as it goes (each one's stage
/// is its record's): one whose copy has completed is mapped from the map
/// rule's step for its layer; a release whose steps are each past its layer
/// at `progress`, where the device is known to be, or ended frees its
/// record; the queued promotions copy, their tier records held, in the
/// order they are due (`Directory::queued`), each its own batch while
/// fewer than the transfer queue's depth are in flight
/// (`TierCopyCosts::depth`; one whose tier record is gone, unmapped for
/// the tier's reserve meanwhile, copies nothing and frees its record).
/// Before the queue's line is measured nothing was asked, so nothing
/// moves ([`line`]). A copy the copy engine refuses or faults ends
/// serving: the device is lost. Whether any moved. Nothing here waits:
/// residency carries them again on the load's events, the frontier's
/// advance (a step's end) and a copy's completion (`Copies::start`)
/// among them.
pub(super) fn carry(core: &mut Core, shared: &Shared, copier: &Copier, progress: Point) -> bool {
    let Some(copy) = line(core, shared) else {
        return false;
    };
    let mut moved = false;
    // The write-backs' ends: each one written lands its expert in the tier.
    let bytes = record_bytes(core);
    for (group, record, batch) in core.directory.backing() {
        match copier.copies.completed(batch) {
            Ok(false) => {}
            Ok(true) => {
                core.directory.written_back(group, record, true);
                // It held the link its record's transfer out of the device.
                shared.telemetry.moved(copy.back.transfer_seconds(bytes[group] as f64), 1);
                moved = true;
            }
            Err(error) => {
                core.directory.written_back(group, record, false);
                shared.end_serving(format!("a release's write-back into the host tier: {error}"));
                moved = true;
            }
        }
    }
    // The copies' ends.
    let mut completed = Vec::new();
    for (group, record, batch) in core.directory.copying() {
        match copier.copies.completed(batch) {
            Ok(false) => {}
            Ok(true) => completed.push((group, record)),
            Err(error) => {
                core.directory.unpromoted(group, record);
                shared.end_serving(format!("a promotion's copy into the bank: {error}"));
                moved = true;
            }
        }
    }
    if !completed.is_empty() {
        // Each held the link its record's transfer: the link's term the
        // decode passes count beside their own reads (`Telemetry::moved`).
        let link = completed.iter().map(|&(group, _)| copy.transfer.transfer_seconds(bytes[group] as f64)).sum::<f64>();
        shared.telemetry.moved(link, completed.len());
        // Mapped by the one rule a read ahead's landing maps by: from the
        // first step in flight yet to resolve each layer as of `progress`
        // (`Steps::unresolved`).
        core.directory.promoted(completed, progress, &core.steps);
        moved = true;
    }
    // The write-backs queued, posted before the promotions: each frees a
    // record a promotion may wait for.
    for back @ (group, record, ..) in core.directory.backs_queued() {
        match post_back(core, copier, copy.depth(bytes[group]), back) {
            Ok(true) => moved = true,
            Ok(false) => break,
            Err(error) => {
                core.directory.written_back(group, record, false);
                shared.end_serving(format!("a release's write-back into the host tier: {error}"));
                return true;
            }
        }
    }
    // The releases no step still reads.
    moved |= core.directory.drain(progress, &core.steps);
    let mut queued = core.directory.queued();
    // A queued promotion whose tier record is gone, or whose expert dropped
    // out of the ranking's pool meanwhile, copies nothing.
    queued.retain(|&(group, record, (layer, expert))| {
        let kept = core.directory.tier_record(layer, expert).is_some() && core.directory.reuse_now().in_top(layer, expert);
        if !kept {
            core.directory.unpromoted(group, record);
            moved = true;
        }
        kept
    });
    if queued.is_empty() {
        return moved;
    }
    for promotion @ (group, record, _) in queued {
        match post(core, copier, copy.depth(bytes[group]), promotion) {
            Ok(true) => moved = true,
            Ok(false) => break,
            // The queue faulted: nothing was formed or held.
            Err(error) => {
                core.directory.unpromoted(group, record);
                shared.end_serving(format!("a promotion's copy into the bank: {error}"));
                return true;
            }
        }
    }
    moved
}

/// Post `promotion`'s copy as its own batch, while fewer than `depth` are
/// in flight (`Copies::post_within`), its tier record held: whether it was
/// posted; why the queue faulted.
fn post(core: &mut Core, copier: &Copier, depth: usize, (group, record, (layer, expert)): (usize, BankRecord, Staged)) -> Result<bool, String> {
    let (directory, groups) = (&mut core.directory, &core.groups);
    let posted = copier.copies.post_within(depth, || {
        let tier = directory.hold_tier(layer, expert).expect("a queued promotion's tier record is kept");
        let tensors = groups[group].tier.as_ref().expect("a promotion copies from its group's host tier");
        let runs = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| crate::experts::copies::Run {
                source: tensor.records.tensor(u64::from(tier.0)).clone(),
                source_offset: 0,
                destination: groups[group].bank_view(index, record.index()),
                destination_offset: 0,
                bytes: tensor.records.bytes(),
            })
            .collect();
        (runs, tier)
    })?;
    let Some((tier, ticket)) = posted else {
        return Ok(false);
    };
    directory.copy(group, record, tier, ticket);
    Ok(true)
}

/// Post a release's write-back `back` (group, record, expert, the tier
/// record its landing took) as its own batch, while fewer than `depth`
/// copies are in flight (`Copies::post_within`): the bank record, which
/// nothing overwrites until the write-back ends, routed tensor by routed
/// tensor into the tier record. Whether it was posted; why the queue
/// faulted.
fn post_back(core: &mut Core, copier: &Copier, depth: usize, (group, record, _, tier): (usize, BankRecord, Staged, crate::experts::directory::HostRecord)) -> Result<bool, String> {
    let (directory, groups) = (&mut core.directory, &core.groups);
    let posted = copier.copies.post_within(depth, || {
        let tensors = groups[group].tier.as_ref().expect("a write-back copies into its group's host tier");
        let runs = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| crate::experts::copies::Run {
                source: groups[group].bank_view(index, record.index()),
                source_offset: 0,
                destination: tensor.records.tensor(u64::from(tier.0)).clone(),
                destination_offset: 0,
                bytes: tensor.records.bytes(),
            })
            .collect();
        (runs, ())
    })?;
    let Some(((), ticket)) = posted else {
        return Ok(false);
    };
    directory.back_copy(group, record, ticket);
    Ok(true)
}

/// The transfer queue's line a promotion is priced and paced by
/// (`Shared::price_copy`), none before the measurement job has measured it
/// (`PointRequest::TierCopy`): until then no promotion is asked, so none is
/// in flight and no release drains, and one that is panics.
fn line(core: &Core, shared: &Shared) -> Option<TierCopyCosts> {
    let copy = shared.copy.lock().expect("a copy's price").clone();
    assert!(copy.is_some() || !core.directory.moving(), "a promotion moves before the transfer queue's line is measured");
    copy
}

/// Each group's record's bytes, every routed tensor of it: what one
/// promotion's copy moves (none where the group has no host tier, which
/// promotes nothing).
fn record_bytes(core: &Core) -> Vec<u64> {
    core.groups.iter().map(|group| group.tier.as_ref().map_or(0, |tensors| tensors.iter().map(|tensor| tensor.records.bytes()).sum())).collect()
}

#[cfg(test)]
mod tests {
    use super::{MissCurve, Reserve, Sizing, decide, reserve};
    use crate::experts::directory::LayerCurve;
    use crate::experts::horizon::{Hold, Tokens};

    /// One layer of 8 experts in a group of `records` records, 3 steps in
    /// flight, each step reading 8 experts.
    fn sizing(records: usize) -> Sizing {
        Sizing { groups: vec![0], reads: vec![8], records: vec![records], flight: 3 }
    }

    /// A curve of 100 decode steps of one layer: `misses[d]` reads at
    /// depth `d` (the last bucket a first read).
    fn curve(misses: &[u64]) -> MissCurve {
        MissCurve { steps: 100, layers: vec![LayerCurve { misses: misses.to_vec(), requests: vec![0; misses.len()], choices: 0 }] }
    }

    /// A hold with one finished interval of `tokens` tokens, queried at its
    /// change so its expectation is `tokens`.
    fn held(tokens: f64) -> (Hold, Tokens) {
        let at = Tokens::new(tokens);
        let mut hold = Hold::started(Tokens::ZERO);
        hold.changed(at);
        (hold, at)
    }

    /// A reserve's records, a share's level at a time, each with what it
    /// saves less what it costs: none before a drain or step is measured.
    #[test]
    fn a_reserve_is_priced_at_each_level_of_the_share() {
        let decoded = curve(&[100, 0, 0, 0, 100, 100, 0, 0, 100]);
        assert_eq!(reserve(&sizing(4), &decoded, None)[0], [Reserve { records: 0, net: 0. }]);
        let candidates = &reserve(&sizing(4), &decoded, Some(2.))[0];
        // A share of 4: 3 misses a step (depths 4, 5 and new), so 6 a drain;
        // the top record is read once a step; each level off the share costs
        // the reads at its depth: none at 3, 2 or 1.
        let records = candidates.iter().map(|candidate| candidate.records).collect::<Vec<_>>();
        assert_eq!(records, [0, 1, 2, 3]);
        let net = candidates.iter().map(|candidate| candidate.net).collect::<Vec<_>>();
        assert_eq!(net, [0., 1., 2., 3.]);
        let (hold, now) = held(10.);
        assert_eq!(decide(candidates, 0, &hold, now, 1.), 3, "repays moving 3 records within the 10 tokens expected held");
    }

    /// A short bank missing often keeps its records: each held record's
    /// hits outweigh the one-step waits a reserve saves.
    #[test]
    fn a_short_bank_missing_often_reserves_nothing() {
        // Every read at depths 0..2 or new: the share's second record holds
        // 1.5 reads a step, more than the top record's one a step that a
        // reserve record would save a covered promotion.
        let decoded = curve(&[100, 150, 350, 0, 0, 0, 0, 0, 200]);
        let candidates = &reserve(&sizing(2), &decoded, Some(1.))[0];
        let unbounded = Hold::started(Tokens::ZERO);
        assert_eq!(decide(candidates, 0, &unbounded, Tokens::ZERO, 1.), 0, "{candidates:?}");
    }

    /// The most a load can measure, every read a miss at every share and
    /// every release waiting out the flight, never takes a layer's last
    /// record.
    #[test]
    fn the_most_a_load_can_measure_leaves_every_layer_a_record() {
        let decoded = curve(&[0, 0, 0, 0, 0, 0, 0, 0, 800]);
        let candidates = &reserve(&sizing(4), &decoded, Some(3.))[0];
        assert!(candidates.iter().all(|candidate| candidate.records <= 3), "{candidates:?}");
        assert_eq!(reserve(&sizing(1), &decoded, Some(3.))[0].iter().map(|candidate| candidate.records).collect::<Vec<_>>(), [0]);
    }

    /// A reserve changes only where its gain, over the hold expected,
    /// repays the records it moves, before any change on any positive gain,
    /// and never on no gain.
    #[test]
    fn a_reserve_changes_only_where_its_gain_repays_the_move() {
        let candidates = [Reserve { records: 0, net: 0. }, Reserve { records: 37, net: 0.5 }];
        let (ten, at_ten) = held(10.);
        let (seventy_four, at_seventy_four) = held(74.);
        let unbounded = Hold::started(Tokens::ZERO);
        assert_eq!(decide(&candidates, 0, &ten, at_ten, 1.), 0, "5 misses saved do not repay moving 37 records");
        assert_eq!(decide(&candidates, 0, &seventy_four, at_seventy_four, 1.), 37);
        assert_eq!(decide(&candidates, 0, &unbounded, Tokens::ZERO, 1.), 37, "before the first change, any positive gain pays");
        assert_eq!(decide(&candidates, 37, &unbounded, Tokens::ZERO, 1.), 37, "already the best");
        let even = [Reserve { records: 0, net: 0. }, Reserve { records: 4, net: 0. }];
        assert_eq!(decide(&even, 4, &unbounded, Tokens::ZERO, 1.), 4, "no gain moves nothing");
    }

    /// A measure no load can produce panics, never sizes a reserve: more
    /// reads than a step makes, or a drain past the steps in flight.
    #[test]
    #[should_panic(expected = "of a layer reading at most 8")]
    fn a_rate_beyond_the_reads_panics() {
        reserve(&sizing(4), &curve(&[0, 0, 0, 0, 0, 0, 0, 0, 900]), Some(1.));
    }

    #[test]
    #[should_panic(expected = "beyond the 3 in flight")]
    fn a_drain_beyond_the_flight_panics() {
        reserve(&sizing(4), &curve(&[0, 0, 0, 0, 0, 0, 0, 0, 100]), Some(3.5));
    }
}
