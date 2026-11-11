//! The one order in which the owner gives memory back for a live claim (a
//! new request's residency, a group's provisioning) that its holders'
//! state cannot meet: what loses nothing first, then what a later use
//! computes or loads again, and preemption last. The expert bank is not on
//! it: it is cache on the ledger, which yields to a claim at once, and it
//! follows its derived target at every boundary
//! (`ExecutorDomain::follow_cache_targets`). Unloading the model is no rung:
//! it happens only at a reading where the guarantee cannot be kept.

/// What the owner can release, in the order it releases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rung {
    /// State backing beyond what the stores need, one spare slab kept per
    /// store, and idle graph slots: nothing is lost.
    Surplus,
    /// Every slab no sequence or checkpoint owns, the spares too: the next
    /// growth allocates again.
    Idle,
    /// The least recently used retained prefix: a later hit computes it
    /// again.
    Prefix,
    /// The idle optional component worth least (drafter or projector), once
    /// no request is resident: its next use loads it again.
    Optional,
    /// Eligible live requests: each replays its accepted tokens.
    Preemption,
}

impl Rung {
    const ORDER: [Self; 5] = [Self::Surplus, Self::Idle, Self::Prefix, Self::Optional, Self::Preemption];
}

/// What one release at a rung came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Release {
    /// Memory was released.
    Released,
    /// Idle graph slots release off the submitting thread, and a later
    /// flight boundary finishes them: nothing costlier runs before it.
    Pending,
    /// The rung has nothing left.
    Nothing,
}

/// Releases at one rung at a time.
pub(crate) trait Releases {
    type Error;

    /// Release once at `rung`.
    fn release(&mut self, rung: Rung) -> Result<Release, Self::Error>;
}

/// Release once, at the first rung in order that releases or begins
/// releasing anything, preemption only where `preempt` (a group's
/// provisioning splits the group before it preempts). A caller repeats it
/// while its need is unmet and it releases, so a costlier rung runs only
/// once every cheaper one has nothing left, including what the costlier
/// one's last release gave back to them (a preempted request's slabs, a
/// dormant drafter); a pending release stops it until it finishes.
pub(crate) fn release_once<R: Releases>(releases: &mut R, preempt: bool) -> Result<Release, R::Error> {
    for rung in Rung::ORDER {
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
                true => Release::Pending,
                false => Release::Released,
            })
        }
    }

    /// Release while releases are immediate: what was released, a pending
    /// one last.
    fn drain(fake: &mut Fake, preempt: bool) -> Vec<Rung> {
        while release_once(fake, preempt).unwrap() == Release::Released {}
        std::mem::take(&mut fake.released)
    }

    /// A pending slot release stops the walk: nothing costlier runs until
    /// a later boundary finishes it.
    #[test]
    fn a_pending_slot_release_stops_the_walk_until_it_finishes() {
        let mut fake = Fake {
            left: BTreeMap::from([(Rung::Prefix, 1), (Rung::Surplus, 2)]),
            pending: vec![Rung::Surplus],
            ..Fake::default()
        };
        assert_eq!(drain(&mut fake, true), [Rung::Surplus]);
        assert_eq!(drain(&mut fake, true), [Rung::Surplus]);
        assert_eq!(drain(&mut fake, true), [Rung::Prefix]);
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

    /// A caller that stops short of preemption never reaches it.
    #[test]
    fn releasing_stops_at_the_callers_rung() {
        let mut fake = Fake {
            left: BTreeMap::from([(Rung::Prefix, 2), (Rung::Preemption, 1)]),
            ..Fake::default()
        };
        assert_eq!(drain(&mut fake, false), [Rung::Prefix, Rung::Prefix]);
        assert_eq!(release_once(&mut fake, true).unwrap(), Release::Released);
        assert_eq!(fake.released, [Rung::Preemption]);
    }
}
