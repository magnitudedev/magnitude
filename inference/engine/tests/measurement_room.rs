//! The measurement job's room claimed of the service owner, over the
//! controlled domain's fake expert bank (`magnitude_executor::fixtures`):
//! the claim is committed memory on the device's ledger, which cache yields
//! to at once, so it is answered as it is made: made where the model's
//! committed memory leaves the room, refused with how short it is where it
//! does not. Each runs on every backend discovery reports available
//! (`bank_backends`).

use magnitude_executor::fixtures::{bank_backends, banked, TestFamily};
use magnitude_scheduler::{
    owner::Owner,
    protocol::{RoomAnswer, RoomGrant, RoomRefusal},
    ServiceLimits,
};
use owner_host::Host;
use std::task::{Context, Poll, Waker};

mod owner_host;

fn limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 2,
        decode_tokens: 2,
        decode_share: 0.5,
        locality_seconds: 1.0,
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// An owner over a bank of four slots on `backend`'s device.
fn owner(backend: seismic::BackendName) -> Host<TestFamily> {
    let (domain, bindings) = banked(backend, 4, 4);
    Host::new(|wakes| Owner::new(domain, bindings, limits(), wakes))
}

fn answer(grant: &mut RoomGrant) -> Option<RoomAnswer> {
    match grant.poll_answer(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(answer) => Some(answer),
        Poll::Pending => None,
    }
}

/// A claim the budget holds beside the model's committed memory is answered
/// made as it is claimed, with no reading or resize waited for; released,
/// a later claim stands on its own.
#[test]
fn a_claim_within_the_budget_is_made_at_once() {
    for backend in bank_backends() {
        let mut host = owner(backend);
        let mut grant = host.owner_mut().claim_measurement_room(1);
        assert_eq!(answer(&mut grant), Some(RoomAnswer::Made));
        host.owner_mut().release_measurement_room();
        let mut again = host.owner_mut().claim_measurement_room(1);
        assert_eq!(answer(&mut again), Some(RoomAnswer::Made));
    }
}

/// A claim past what the process may hold is refused as it is claimed,
/// with how many bytes it is short: cache never refuses it, so only the
/// model's committed memory can.
#[test]
fn a_claim_past_the_budget_is_refused_at_once_with_its_shortfall() {
    for backend in bank_backends() {
        let mut host = owner(backend);
        let mut grant = host.owner_mut().claim_measurement_room(u64::MAX / 4);
        assert!(matches!(answer(&mut grant), Some(RoomAnswer::Refused(RoomRefusal::Short { bytes })) if bytes > 0));
    }
}
