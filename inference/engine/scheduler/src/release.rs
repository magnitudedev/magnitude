//! The one order in which the owner gives memory back: the reclaim
//! band, a new request's residency and a group's provisioning all release
//! through it. Each rung loses more than the one before it, and unloading the
//! model, past the last rung, loses everything: it follows only once every
//! rung is exhausted (`MemoryCondition::should_unload`).

/// What the owner can release, cheapest loss first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rung {
    /// State backing beyond what the stores need, one spare slab kept per
    /// store, and idle graph slots: nothing is lost.
    Surplus,
    /// Every slab no sequence or checkpoint owns, the spares too: the next
    /// growth allocates again.
    Idle,
    /// One step of the expert bank toward its floor: decode reads more of
    /// its experts from storage, and no work is lost.
    Bank,
    /// The least recently used retained prefix: a later hit computes it
    /// again. Its rows stay charged until the cheaper rungs, which run next,
    /// release them.
    Prefix,
    /// The optional components (drafter, projector) once no request is
    /// resident: their next use loads them again.
    Optional,
    /// Eligible live requests: each replays its accepted tokens.
    Preemption,
}

impl Rung {
    const LADDER: [Self; 6] = [
        Self::Surplus,
        Self::Idle,
        Self::Bank,
        Self::Prefix,
        Self::Optional,
        Self::Preemption,
    ];
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
}

/// Release once, at the cheapest rung up to `through` that releases or
/// begins releasing anything. A caller repeats it while its need is unmet and
/// it releases, so a costlier rung runs only once every cheaper one has
/// nothing left, including what the costlier one's last release gave back to
/// them (a preempted request's slabs, a dormant drafter); a pending release
/// stops it until the release finishes.
pub(crate) fn release_once<R: Releases>(releases: &mut R, through: Rung) -> Result<Release, R::Error> {
    for rung in Rung::LADDER {
        if rung > through {
            break;
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
    }

    /// Release while releases are immediate: what was released, a pending
    /// one last.
    fn drain(fake: &mut Fake, through: Rung) -> Vec<Rung> {
        while release_once(fake, through).unwrap() == Release::Released {}
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
        assert_eq!(drain(&mut fake, Rung::Preemption), [Rung::Surplus, Rung::Bank]);
        assert_eq!(drain(&mut fake, Rung::Preemption), [Rung::Surplus, Rung::Bank]);
        assert_eq!(drain(&mut fake, Rung::Preemption), [Rung::Surplus, Rung::Prefix]);
    }

    /// The bank shrinks to its floor after the free releases and before any
    /// retained prefix is evicted; preemption comes last.
    #[test]
    fn the_bank_shrinks_to_its_floor_before_any_prefix_is_evicted() {
        let mut fake = Fake {
            left: BTreeMap::from([
                (Rung::Preemption, 1),
                (Rung::Prefix, 2),
                (Rung::Bank, 3),
                (Rung::Idle, 1),
                (Rung::Surplus, 1),
            ]),
            ..Fake::default()
        };
        assert_eq!(
            drain(&mut fake, Rung::Preemption),
            [
                Rung::Surplus,
                Rung::Idle,
                Rung::Bank,
                Rung::Bank,
                Rung::Bank,
                Rung::Prefix,
                Rung::Prefix,
                Rung::Preemption,
            ]
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
            drain(&mut fake, Rung::Preemption),
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
        assert_eq!(drain(&mut fake, Rung::Optional), [Rung::Bank, Rung::Bank]);
        assert_eq!(release_once(&mut fake, Rung::Preemption).unwrap(), Release::Released);
        assert_eq!(fake.released, [Rung::Preemption]);
    }
}
