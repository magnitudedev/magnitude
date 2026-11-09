//! The expert bank regrows after a press on its device store, over the
//! service owner and the controlled domain's fake expert bank
//! (`magnitude_executor::fixtures`): another program's hold takes the store
//! below its planning reserve, the owner's release ladder shrinks the bank
//! for Reclaim, the hold lets go, and the bank grows back at a later
//! reading's settle, whose supply holds the reserve derived from the host's
//! measured motion, with no wait beside it. The fixture's heap reads
//! the host's memory as first sampled, moved only by what the process takes
//! and gives back and by the test's press (`StillHost`). Each runs on every
//! backend discovery reports available (`bank_backends`).

use magnitude_executor::{
    fixtures::{bank_backends, bank_row, banked, misses_at, press, price_growth, shrink_settled, TestFamily},
    BankResize, ExpertResidency, ShrinkCause,
};
use magnitude_scheduler::{owner::Owner, ServiceLimits};
use magnitude_state::ShrinkPolicy;
use owner_host::Host;

mod owner_host;

/// The memory observations, one every observation interval, that a shrink
/// and the growth after it finish within.
const OBSERVATIONS: usize = 200;

fn limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 2,
        decode_tokens: 2,
        decode_share: 0.5,
        locality_seconds: 1.0,
        conversations: std::num::NonZeroUsize::MIN,
    }
}

fn experts(host: &Host<TestFamily>) -> ExpertResidency {
    host.owner().serving_census().unwrap().experts.expect("the load has an expert bank")
}

/// Observe until the bank's last finished resize is one `finished` takes:
/// the service time it was first seen at. What each observation read of
/// `store` and left of the bank is what a failure reports.
fn observe_until(host: &mut Host<TestFamily>, store: seismic::StoreId, finished: impl Fn(&BankResize) -> bool, what: &str) -> u64 {
    let mut seen = Vec::new();
    for _ in 0..OBSERVATIONS {
        host.observe();
        let census = experts(host);
        if census.last_resize.as_ref().is_some_and(&finished) {
            return host.now();
        }
        let reading = read(host, store);
        seen.push((host.now() / 1_000_000, reading.band, reading.headroom_bytes as i128 - reading.thresholds.planning_bytes as i128, census.slots, census.last_resize));
    }
    panic!("no {what} within {OBSERVATIONS} observations: (ms, band, headroom above planning, slots, last resize) {seen:?}")
}

/// The owner's domain's reading of `store` now.
fn read(host: &Host<TestFamily>, store: seismic::StoreId) -> magnitude_executor::platform::DomainReading {
    host.owner()
        .inspect_domain(|domain| domain.observe_memory().map_err(|error| error.to_string()))
        .unwrap()
        .into_iter()
        .find(|reading| reading.domain == store)
        .expect("the bank's store is read")
}

/// A press half a slot below the planning reserve of the bank's store puts
/// it in Reclaim, and the ladder's bank rung gives back a slot for it. The
/// press lets go at once, and a later reading's settle grows the bank back
/// to the four slots the load's misses stop falling at.
#[test]
fn a_bank_shrunk_for_a_press_regrows_once_the_press_lets_go() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        // Misses at 3 slots, none at 4: a fourth slot saves the decode time.
        price_growth(&mut domain, &mut bindings, misses_at(3, 100));
        // The state's surplus is released already: the bank's is the rung
        // that answers the press.
        shrink_settled(&mut domain, &mut bindings, ShrinkPolicy::Reclaim);
        let press = press(&domain);
        let mut host = Host::new(|wakes| Owner::new(domain, bindings, limits(), wakes));
        host.observe();
        assert_eq!(experts(&host).slots, 4, "{backend:?}: nothing shrinks while memory is free");
        let reading = read(&host, press.store());
        let above = reading.headroom_bytes - reading.thresholds.planning_bytes;
        press.take(above + row / 2);
        let reclaimed = |resize: &BankResize| matches!(resize, BankResize::Shrank { cause: ShrinkCause::Reclaim, .. });
        let shrunk = observe_until(&mut host, press.store(), reclaimed, &format!("{backend:?} Reclaim shrink"));
        assert!(experts(&host).slots < 4, "{backend:?}: the press shrank the bank");
        press.release();
        let grew = |resize: &BankResize| matches!(resize, BankResize::Grew { .. });
        let grown = observe_until(&mut host, press.store(), grew, &format!("{backend:?} growth"));
        assert!(grown > shrunk, "{backend:?}: grown before the shrink");
        assert_eq!(experts(&host).slots, 4, "{backend:?}: grown to where the misses stop falling");
    }
}
