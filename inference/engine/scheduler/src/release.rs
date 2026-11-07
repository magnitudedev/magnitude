//! The one order in which the owner gives memory back: the reclaim
//! band, a new request's residency and a group's provisioning all release
//! through it. Every holding but live requests is priced on the memory
//! heap's one scale (`ExecutorDomain::settle_memory`) and released the
//! lowest rate per byte first: idle state and graph slots at nothing, the
//! bank's slots (its borrowed tail the least of them), cached prefixes and
//! the optional components by what each saves. Preemption follows only
//! once nothing priced is left, and unloading the model, past it, only
//! once every rung is exhausted (`MemoryCondition::should_unload`).

/// What the owner can release.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rung {
    /// State backing beyond what the stores need, one spare slab kept per
    /// store, and idle graph slots: nothing is lost.
    Surplus,
    /// Every slab no sequence or checkpoint owns, the spares too: the next
    /// growth allocates again.
    Idle,
    /// One shrink of the expert bank toward its floor, to the size the
    /// heap settles it at: decode reads more of its experts from storage,
    /// and no work is lost. Priced.
    Bank,
    /// The retained prefix worth least: a later hit computes it again. Its
    /// rows stay charged until the cheaper rungs, which run next, release
    /// them. Priced.
    Prefix,
    /// The idle optional component worth least per byte (drafter or
    /// projector), once no request is resident: its next use loads it
    /// again. Priced; one whose worth is not priced yet (a head before its
    /// drafting is) is released only once no priced one is left, and the
    /// rung then holds nothing priced.
    Optional,
    /// Eligible live requests: each replays its accepted tokens.
    Preemption,
}

impl Rung {
    /// The rungs the heap prices, in the order a tie keeps.
    const PRICED: [Self; 5] = [Self::Surplus, Self::Idle, Self::Bank, Self::Prefix, Self::Optional];
}

/// What one release at a rung came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Release {
    /// Memory was released.
    Released,
    /// A release runs that a later flight boundary finishes, its own
    /// ([`Pending`]): nothing costlier runs before it.
    Pending(Pending),
    /// A release must wait for a gap between flights (a bank resize with a
    /// step still running), or for the records in flight to land (a shrink
    /// of the records the host lands into, every cut of which falls below
    /// one): nothing began, nothing costlier runs, and the next boundary
    /// releases again.
    Deferred,
    /// What is short is released already and the host has yet to show it:
    /// nothing began, nothing costlier runs, and a later reading decides
    /// again.
    Covered,
    /// The rung has nothing left.
    Nothing,
}

/// A release a later flight boundary finishes, by what it releases: each
/// finishes on its own, so the ladder waits for each apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pending {
    /// Idle graph slots, released off the submitting thread (the surplus
    /// rung's).
    Slots,
    /// A resize of the expert bank (the bank rung's).
    Bank,
}

/// Releases at one rung of the ladder at a time.
pub(crate) trait Releases {
    type Error;

    /// Release once at `rung`.
    fn release(&mut self, rung: Rung) -> Result<Release, Self::Error>;

    /// The lowest rate per byte a priced rung ([`Rung::PRICED`]) holds, in
    /// serving seconds saved a second per byte; none where it holds nothing
    /// priced to release.
    fn rate(&mut self, rung: Rung) -> Result<Option<f64>, Self::Error>;
}

/// The ladder now: the priced rungs by the lowest rate per byte each holds
/// (one holding nothing priced goes last of them, a tie in their own
/// order), then preemption.
fn ladder<R: Releases>(releases: &mut R) -> Result<Vec<Rung>, R::Error> {
    let mut priced = Vec::with_capacity(Rung::PRICED.len());
    for rung in Rung::PRICED {
        priced.push((releases.rate(rung)?.unwrap_or(f64::INFINITY), rung));
    }
    priced.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(priced.into_iter().map(|(_, rung)| rung).chain([Rung::Preemption]).collect())
}

/// Release once, at the cheapest rung of the ladder ([`ladder`]) that
/// releases or begins releasing anything, preemption only where `preempt`
/// (a group's provisioning splits the group before it preempts). A caller
/// repeats
/// it while its need is unmet and it releases, so a costlier rung runs only
/// once every cheaper one has nothing left, including what the costlier
/// one's last release gave back to them (a preempted request's slabs, a
/// dormant drafter); a pending release stops it until the release
/// finishes. The ladder is read afresh each time: a release changes what
/// the next byte of each holding is worth.
pub(crate) fn release_once<R: Releases>(releases: &mut R, preempt: bool) -> Result<Release, R::Error> {
    for rung in ladder(releases)? {
        if rung == Rung::Preemption && !preempt {
            continue;
        }
        match releases.release(rung)? {
            Release::Nothing => {}
            begun => return Ok(begun),
        }
    }
    Ok(Release::Nothing)
}

#[cfg(test)]
mod tests {
    use super::{Release, Releases, Rung, release_once};
    use std::collections::BTreeMap;

    /// A fake owner: how many releases each rung still has, what a release
    /// at a rung gives back to cheaper ones, and the rungs whose releases
    /// finish only at a later boundary.
    #[derive(Default)]
    struct Fake {
        left: BTreeMap<Rung, u32>,
        gives: BTreeMap<Rung, Vec<Rung>>,
        pending: Vec<Rung>,
        released: Vec<Rung>,
        /// Each priced rung's lowest rate per byte while it holds any.
        rates: BTreeMap<Rung, f64>,
    }

    impl Releases for Fake {
        type Error = ();

        fn release(&mut self, rung: Rung) -> Result<Release, ()> {
            let left = self.left.entry(rung).or_default();
            if *left == 0 {
                return Ok(Release::Nothing);
            }
            *left -= 1;
            self.released.push(rung);
            for given in self.gives.get(&rung).cloned().unwrap_or_default() {
                *self.left.entry(given).or_default() += 1;
            }
            Ok(match self.pending.contains(&rung) {
                true => Release::Pending(super::Pending::Bank),
                false => Release::Released,
            })
        }

        fn rate(&mut self, rung: Rung) -> Result<Option<f64>, ()> {
            let holds = self.left.get(&rung).is_some_and(|&left| left > 0);
            Ok(self.rates.get(&rung).copied().filter(|_| holds))
        }
    }

    /// Release while releases are immediate: what was released, a pending
    /// one last.
    fn drain(fake: &mut Fake, preempt: bool) -> Vec<Rung> {
        while release_once(fake, preempt).unwrap() == Release::Released {}
        std::mem::take(&mut fake.released)
    }

    /// A pending bank step stops the ladder: no prefix is evicted while it
    /// runs, and each boundary that finishes it resumes the ladder from the
    /// cheapest rung, so the bank reaches its floor before any prefix.
    #[test]
    fn a_pending_bank_step_stops_the_ladder_until_it_finishes() {
        let mut fake = Fake {
            left: BTreeMap::from([(Rung::Prefix, 1), (Rung::Bank, 2), (Rung::Surplus, 1)]),
            gives: BTreeMap::from([(Rung::Bank, vec![Rung::Surplus])]),
            pending: vec![Rung::Bank],
            ..Fake::default()
        };
        assert_eq!(drain(&mut fake, true), [Rung::Surplus, Rung::Bank]);
        assert_eq!(drain(&mut fake, true), [Rung::Surplus, Rung::Bank]);
        assert_eq!(drain(&mut fake, true), [Rung::Surplus, Rung::Prefix]);
    }

    /// An ask releases what loses nothing first, then the priced holding
    /// worth least per byte: prefixes that save less a byte than the bank's
    /// slots go before them, the bank's before prefixes worth more; and
    /// preemption comes only once nothing priced is left.
    #[test]
    fn an_ask_releases_the_lowest_rate_per_byte_first_and_preempts_last() {
        let left = || BTreeMap::from([
            (Rung::Preemption, 1),
            (Rung::Prefix, 2),
            (Rung::Bank, 3),
            (Rung::Idle, 1),
            (Rung::Surplus, 1),
        ]);
        let mut cheap_prefixes = Fake {
            left: left(),
            rates: BTreeMap::from([(Rung::Surplus, 0.), (Rung::Idle, 0.), (Rung::Bank, 0.5), (Rung::Prefix, 0.1)]),
            ..Fake::default()
        };
        assert_eq!(
            drain(&mut cheap_prefixes, true),
            [Rung::Surplus, Rung::Idle, Rung::Prefix, Rung::Prefix, Rung::Bank, Rung::Bank, Rung::Bank, Rung::Preemption]
        );
        let mut cheap_bank = Fake {
            left: left(),
            rates: BTreeMap::from([(Rung::Surplus, 0.), (Rung::Idle, 0.), (Rung::Bank, 0.1), (Rung::Prefix, 0.5)]),
            ..Fake::default()
        };
        assert_eq!(
            drain(&mut cheap_bank, true),
            [Rung::Surplus, Rung::Idle, Rung::Bank, Rung::Bank, Rung::Bank, Rung::Prefix, Rung::Prefix, Rung::Preemption]
        );
    }

    /// A bound drafter worth more per byte than the bank's slots is kept
    /// over them, and the bank's slots worth more than a prefix over it: no
    /// fixed order among priced rungs.
    #[test]
    fn priced_rungs_release_by_rate_alone() {
        let mut fake = Fake {
            left: BTreeMap::from([(Rung::Optional, 1), (Rung::Bank, 2), (Rung::Prefix, 1), (Rung::Preemption, 1)]),
            rates: BTreeMap::from([(Rung::Optional, 0.9), (Rung::Bank, 0.3), (Rung::Prefix, 0.1)]),
            ..Fake::default()
        };
        assert_eq!(
            drain(&mut fake, true),
            [Rung::Prefix, Rung::Bank, Rung::Bank, Rung::Optional, Rung::Preemption]
        );
    }

    /// What a costlier release gives back to cheaper rungs is released before
    /// the next costly one: an evicted prefix's slabs before another
    /// prefix, a preempted request's slabs and a dormant drafter before
    /// another preemption.
    #[test]
    fn what_a_costlier_release_frees_goes_before_the_next_costly_one() {
        let mut fake = Fake {
            left: BTreeMap::from([(Rung::Prefix, 2), (Rung::Preemption, 2)]),
            gives: BTreeMap::from([
                (Rung::Prefix, vec![Rung::Surplus]),
                (Rung::Preemption, vec![Rung::Idle, Rung::Optional]),
            ]),
            ..Fake::default()
        };
        assert_eq!(
            drain(&mut fake, true),
            [
                Rung::Prefix,
                Rung::Surplus,
                Rung::Prefix,
                Rung::Surplus,
                Rung::Preemption,
                Rung::Idle,
                Rung::Optional,
                Rung::Preemption,
                Rung::Idle,
                Rung::Optional,
            ]
        );
    }

    /// A caller that stops short of a rung never reaches it.
    #[test]
    fn releasing_stops_at_the_callers_rung() {
        let mut fake = Fake {
            left: BTreeMap::from([(Rung::Bank, 2), (Rung::Preemption, 1)]),
            ..Fake::default()
        };
        assert_eq!(drain(&mut fake, false), [Rung::Bank, Rung::Bank]);
        assert_eq!(release_once(&mut fake, true).unwrap(), Release::Released);
        assert_eq!(fake.released, [Rung::Preemption]);
    }
}
