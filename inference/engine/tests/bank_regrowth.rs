//! The laptop's failure at the service owner: a bank shrunk while another
//! hold took its room, which nothing grew back once the room was free
//! again. Over the controlled domain's fake expert bank
//! (`magnitude_executor::fixtures`), on every backend discovery reports
//! available (`bank_backends`).

use magnitude_executor::fixtures::{bank_backends, banked, press, TestFamily};
use magnitude_scheduler::{
    owner::Owner,
    protocol::{RoomAnswer, RoomRefusal},
    ServiceLimits,
};
use owner_host::Host;
use std::task::{Context, Poll, Waker};

mod owner_host;

/// Runs the owner takes at most to carry a resize to its end: each run is
/// a boundary, and a resize finishes within a few.
const RUNS: usize = 1_000;

fn limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 2,
        decode_tokens: 2,
        decode_share: 0.5,
        locality_seconds: 1.0,
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// The bank's slots now.
fn slots(host: &Host<TestFamily>) -> u64 {
    host.owner().serving_census().unwrap().experts.expect("a banked load").slots
}

/// Run the owner until the bank holds `want` slots, at most [`RUNS`] runs:
/// whether it got there.
fn runs_to(host: &mut Host<TestFamily>, want: u64) -> bool {
    (0..RUNS).any(|_| {
        host.run();
        slots(host) == want
    })
}

/// The room the model's committed memory leaves on the bank's store: a
/// claim past it is refused by exactly how short it is, and a refused
/// claim holds nothing.
fn room(host: &mut Host<TestFamily>) -> u64 {
    const PAST: u64 = u64::MAX / 4;
    let mut probe = host.owner_mut().claim_measurement_room(PAST);
    match probe.poll_answer(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(RoomAnswer::Refused(RoomRefusal::Short { bytes })) => PAST - bytes,
        answer => panic!("a claim past the budget is refused, not {answer:?}"),
    }
}

/// Item 9: a bank of four slots at its full target; another program taking
/// all the room the model's committed memory leaves, so the bank's target
/// falls to its floor and the owner shrinks it there; that program letting
/// go, and at the boundaries after it the owner grows the bank back to all
/// four slots. On main a settle released the bank and nothing grew it back.
#[test]
fn the_owner_regrows_the_bank_once_the_room_that_shrank_it_is_free() {
    for backend in bank_backends() {
        let (domain, bindings) = banked(backend, 4, 4);
        let other = press(&domain);
        let mut host = Host::new(|wakes| Owner::new(domain, bindings, limits(), wakes));
        host.run();
        assert_eq!(slots(&host), 4, "{backend:?}: the bank at its full target");
        other.take(room(&mut host));
        assert!(runs_to(&mut host, 1), "{backend:?}: the bank shrinks to its floor, at {}", slots(&host));
        other.release();
        assert!(runs_to(&mut host, 4), "{backend:?}: the bank grows back to its full target, at {}", slots(&host));
    }
}
