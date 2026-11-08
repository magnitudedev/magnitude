//! The host's promotions into the bank:
//! background work, off every step's path, on the device's copy engine.
//! None is triggered by a miss: the one ranking decides which of the host
//! tier's experts enter the bank and which records they take
//! (`Directory::promotions`: the experts its share of each layer holds and
//! the bank lacks, into free records), as the completed steps feed it
//! reads, and a copy competes with the step for the link and host memory,
//! so an expert enters only once its reads pay for its copy. Each promotion
//!
//! 1. holds the tier record keeping its expert (`HostRecords::hold`) and
//!    copies it into its free record on the copy engine
//!    (`Copies::background`): every promotion asked by then in one batch,
//!    only while no batch runs, so the bank takes the ranking's share at
//!    the link's rate, not at one batch's round trip a record;
//! 2. once the copy has completed, maps its expert to the record for every
//!    step from the first not yet begun, and releases the tier record.
//!
//! A record is freed apart from any promotion, by a release
//! (`Directory::promotions`, `Release`): its expert, outside its layer's
//! share, is unmapped for every step from the first not yet begun, so the
//! steps already queued may still read it, and the record is free once
//! every one it was mapped for has passed its layer in the device's order
//! or ended, by whichever exit (`Steps::served`). Each group keeps a reserve of free
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
use crate::experts::directory::{MissCurve, Point};

/// One routed tensor of a bank group as a promotion copies it: its host
/// tier tensor (every record it reserves, which the copy engine reads, so
/// no host view of it) and its bank record tensor, and where the records of
/// each lie, whose record a promotion copies whole.
pub(in crate::experts) struct CopyTensor {
    pub tier: seismic::Tensor,
    pub tier_at: seismic::RecordPlacement,
    pub bank: seismic::Tensor,
    pub bank_at: seismic::RecordPlacement,
}

/// What promotions copy with: the device's copy engine, and each group's
/// routed tensors in its record's order.
pub(in crate::experts) struct Copier {
    pub copies: crate::experts::copies::Copies,
    pub groups: Vec<Vec<CopyTensor>>,
}

/// What a reserve is sized within ([`reserve`]): each layer's group and
/// the most experts a decode step reads there (its experts, at most a
/// step's choices), each group's records, and the steps in flight.
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
/// Taking `ρ` records off each of a group's `M` layers' whole share `w`
/// reserves `r = left + ρM` (`left`, the records `w` leaves over, held by
/// no share anyway). It costs the hits those records held: `Σ_l (m_l(w−ρ)
/// − m_l(w))` a step, `m_l(s)` layer `l`'s misses a step at `s` slots. It
/// saves each promotion it covers the drain's wait: the ranking asks `a =
/// Σ_l m_l(w)` promotions a step, a drain's worth `a·drain` of them find a
/// free record where `r` does, and each would have been read at the rate
/// of the record atop its layer's stack, `p = mean_l m_l(0) − m_l(1)`, for
/// each of the `drain` steps it waited: `min(a·drain, r)·p` a step. `ρ`
/// stops a record short of each whole share, so every layer keeps one.
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
/// held since `held` steps: the best, where its gain over `current`, held
/// as long again, repays moving the records between them (a promotion or
/// a release each, a miss each); else `current`.
fn decide(candidates: &[Reserve], current: usize, held: u64) -> usize {
    // The best, the fewer records on a tie.
    let best = candidates
        .iter()
        .copied()
        .max_by(|a, b| a.net.total_cmp(&b.net).then(b.records.cmp(&a.records)))
        .expect("a reserve's candidates");
    let now = candidates.iter().find(|candidate| candidate.records == current).map_or(f64::NEG_INFINITY, |candidate| candidate.net);
    let moved = best.records.abs_diff(current) as f64;
    match (best.net - now) * held as f64 >= moved {
        true => best.records,
        false => current,
    }
}

/// What sizes the reserves: the steps the releases so far waited for, and
/// their count, and the frontier each group's reserve last changed at.
/// What promotions copy with is residency's thread's alone
/// (`Residency::keep`), passed to each turn: the request service, which
/// shares the records under residency's lock, reaches no transfer.
pub(super) struct Promotions {
    drained: (u64, u64),
    changed: Vec<u64>,
}

impl Promotions {
    pub(super) fn new() -> Self {
        Self { drained: (0, 0), changed: Vec::new() }
    }

    /// One turn: size each group's reserve by `decoded`, the decode steps'
    /// depth curve where a step was read since the last turn ([`reserve`]),
    /// take what the ranking wants now (`Directory::promotions`, in time
    /// proportional to what changed), then carry every promotion and release
    /// in flight as far as it goes ([`carry`]), those just asked among
    /// them: no event follows an ask, so what it lets move moves now.
    /// Whether any moved. Where the load's traffic is `held` nothing is
    /// asked: only what is in flight is carried.
    pub(super) fn advance(&mut self, core: &mut Core, shared: &Shared, copier: &Copier, decoded: Option<&MissCurve>, held: bool, progress: Point) -> bool {
        if let Some(decoded) = decoded {
            self.size(core, decoded);
        }
        let asked = !held && self.ask(core, shared);
        asked | carry(core, shared, copier, progress)
    }

    /// Size each group's reserve by `decoded` ([`reserve`], [`decide`]).
    fn size(&mut self, core: &mut Core, decoded: &MissCurve) {
        let frontier = core.steps.frontier();
        let choices = crate::operators::routed::fused_graph::DECODE_ROWS as usize * core.extents.selected;
        let sizing = Sizing {
            groups: core.directory.layer_groups(),
            reads: core.directory.layer_experts().into_iter().map(|experts| experts.min(choices)).collect(),
            records: core.directory.records(),
            flight: core.extents.words.flight as usize,
        };
        let drain = (self.drained.1 > 0).then(|| self.drained.0 as f64 / self.drained.1 as f64);
        self.changed.resize(sizing.records.len(), frontier);
        for (group, candidates) in reserve(&sizing, decoded, drain).into_iter().enumerate() {
            let current = core.directory.reserved(group);
            let records = decide(&candidates, current, frontier - self.changed[group]);
            if records != current {
                core.directory.reserve(group, records);
                self.changed[group] = frontier;
            }
        }
    }

    /// What the ranking asks now (`Directory::promotions`): each promotion
    /// to copy, and each release's expert unmapped from the first step not
    /// yet begun, its wait in steps measured. Whether it asked any. The
    /// ranking's changes settle first, outside the steps' lock, and a turn
    /// that stirred nothing (no want, record or reservation changed) takes
    /// no steps' lock.
    fn ask(&mut self, core: &mut Core, shared: &Shared) -> bool {
        if !core.directory.settle_changes() && !shared.lent.lock().freed() {
            return false;
        }
        let decoding = !core.steps.prompt_running();
        // Asked under the steps' lock, under which a prompt group reserves
        // before its steps begin: it begins either after every release of
        // a lent record unmapped its expert from a step at or before its
        // first, or reserved before any is asked of one (`lent`).
        let (directory, drained) = (&mut core.directory, &mut self.drained);
        let asked = core.steps.unbegun_beside_frontier(|frontier, at| {
            let asked = directory.promotions(at, &mut shared.lent.lock());
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
/// is its record's): one whose copy has completed is mapped from the first
/// step not yet begun; a release whose steps are each past its layer at
/// `progress`, where the device is known to be, or ended frees its record; every promotion queued copies, its tier record held, all of
/// them in one batch (one whose tier record is gone, unmapped for the
/// tier's reserve meanwhile, copies nothing and frees its record). A copy
/// the copy engine refuses or faults ends serving: the device is lost.
/// Whether any moved. Nothing here waits: residency carries them again on
/// the load's events, the frontier's advance (a step's end) and a copy's
/// completion (`Copies::start`) among them.
pub(super) fn carry(core: &mut Core, shared: &Shared, copier: &Copier, progress: Point) -> bool {
    let mut moved = false;
    // The copies' ends.
    let mut completed = Vec::new();
    for (group, record, batch) in core.directory.copying() {
        match copier.copies.completed(batch) {
            Ok(false) => {}
            Ok(true) => completed.push((group, record)),
            Err(error) => {
                core.directory.unpromoted(group, record, &shared.lent);
                shared.end_serving(format!("a promotion's copy into the bank: {error}"));
                moved = true;
            }
        }
    }
    if !completed.is_empty() {
        core.directory.promoted(completed, &core.steps, &shared.lent);
        moved = true;
    }
    // The releases no step still reads.
    moved |= core.directory.drain(progress, &core.steps);
    // Under the reservation's lock to the copies' queueing: a prompt group
    // reserves the lent records only once no copy into one runs, and while
    // it holds them or waits to, a promotion queued into one copies nothing
    // (`Directory::queued`), so the wait covers at most the batch running.
    let mut lending = shared.lent.lock();
    let mut queued = core.directory.queued(&lending);
    // A queued promotion whose tier record is gone copies nothing.
    queued.retain(|&(group, record, (layer, expert))| {
        let kept = core.directory.tier_record(layer, expert).is_some();
        if !kept {
            core.directory.unpromoted(group, record, &shared.lent);
            moved = true;
        }
        kept
    });
    if queued.is_empty() {
        return moved;
    }
    // Every queued promotion's copy, one batch, queued only while no batch
    // runs (`Copies`).
    let directory = &mut core.directory;
    let posted = copier.copies.background(|| {
        let mut runs = Vec::new();
        let tiers = queued
            .iter()
            .map(|&(group, record, (layer, expert))| {
                let tier = directory.hold_tier(layer, expert).expect("a queued promotion's tier record is kept");
                runs.extend(copier.groups[group].iter().map(|tensor| crate::experts::copies::Run {
                    source: tensor.tier.clone(),
                    source_offset: tensor.tier_at.offset(u64::from(tier.0)),
                    destination: tensor.bank.clone(),
                    destination_offset: tensor.bank_at.offset(u64::from(record.0)),
                    bytes: tensor.tier_at.record(),
                }));
                tier
            })
            .collect::<Vec<_>>();
        (runs, tiers)
    });
    match posted {
        Ok(None) => moved,
        Ok(Some((tiers, batch))) => {
            for (&(group, record, _), tier) in queued.iter().zip(tiers) {
                directory.copy(group, record, tier, batch, &mut lending);
            }
            true
        }
        // The queue faulted: nothing was formed or held.
        Err(error) => {
            drop(lending);
            for (group, record, _) in queued {
                directory.unpromoted(group, record, &shared.lent);
            }
            shared.end_serving(format!("a promotion's copy into the bank: {error}"));
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MissCurve, Reserve, Sizing, decide, reserve};
    use crate::experts::directory::LayerCurve;

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
        assert_eq!(decide(candidates, 0, 10), 3, "repays moving 3 records within the 10 steps held");
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
        assert_eq!(decide(candidates, 0, u64::MAX / 4), 0, "{candidates:?}");
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

    /// A reserve changes only where its gain, held as long as the reserve
    /// it replaces was, repays the records it moves: a gain near zero never
    /// flips it.
    #[test]
    fn a_reserve_changes_only_where_its_gain_repays_the_move() {
        let candidates = [Reserve { records: 0, net: 0. }, Reserve { records: 37, net: 0.5 }];
        assert_eq!(decide(&candidates, 0, 10), 0, "5 misses saved do not repay moving 37 records");
        assert_eq!(decide(&candidates, 0, 74), 37);
        assert_eq!(decide(&candidates, 37, 1), 37, "already the best");
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
