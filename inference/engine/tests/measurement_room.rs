//! The measurement job's room claimed of the service owner, over the
//! controlled domain's fake expert bank (`magnitude_executor::fixtures`): a
//! claim the room beside the model holds is answered made at the first
//! probe; one beyond it shrinks the bank a row at a time, each shrink decided
//! at a probe no resize ran across, and is made by the rows given back; one
//! beyond what the bank holds above its floor is refused before anything
//! shrinks. The fixture's heap reads the host's memory as first sampled,
//! moved only by what the process itself takes and gives back
//! (`StillHost`). Skipped on a device that reserves no storage, where the
//! bank cannot exist.

use magnitude_executor::{
    fixtures::{bank_row, banked, TestFamily},
    BankResize, ExecutorDomain, ExpertResidency, MeasurementRoom, ShrinkCause,
};
use magnitude_scheduler::{
    owner::Owner,
    protocol::{RoomAnswer, RoomGrant, RoomRefusal},
    ServiceLimits,
};
use owner_host::Host;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

mod owner_host;

/// The memory observations a claim gets to be answered within: a shrink
/// begins at one, installs at the next and releases at the one after.
const OBSERVATIONS: usize = 24;

fn limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 2,
        decode_tokens: 2,
        decode_share: 0.5,
        locality_seconds: 1.0,
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// The room beside the model at a probe of `domain`, and what its bank
/// holds above its floor, read through a claim no device holds.
fn room(domain: &mut ExecutorDomain<TestFamily>) -> (u64, u64) {
    domain
        .probe_memory(Duration::from_millis(100))
        .expect("the host's memory is in the Normal band, where no reclaim release runs");
    let beyond = u64::MAX / 4;
    domain.claim_measurement_room(beyond);
    let Some(MeasurementRoom::Beyond { short, releasable }) = domain.measurement_room().unwrap() else {
        panic!("a claim beyond the device is beyond the bank's floor")
    };
    domain.release_measurement_room();
    (beyond - short, releasable)
}

/// An owner over a bank of four slots, the room beside it, what the bank
/// holds above its floor, and the bytes of a slot.
fn owner() -> Option<(Host<TestFamily>, u64, u64, u64)> {
    let (mut domain, bindings) = banked(4, 4)?;
    let row = bank_row(&domain);
    let (room, releasable) = room(&mut domain);
    Some((Host::new(|wakes| Owner::new(domain, bindings, limits(), wakes)), room, releasable, row))
}

fn answer(grant: &mut RoomGrant) -> Option<RoomAnswer> {
    match grant.poll_answer(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(answer) => Some(answer),
        Poll::Pending => None,
    }
}

fn experts(host: &Host<TestFamily>) -> ExpertResidency {
    host.owner().serving_census().unwrap().experts.expect("the load has an expert bank")
}

/// The slots a shrink of one row for the measurement left.
fn shrunk_to(resize: &BankResize) -> u64 {
    match *resize {
        BankResize::Shrank { cause: ShrinkCause::Measurement, to, by, .. } if by.get() == 1 => to,
        other => panic!("a one-row shrink for the measurement, not {other:?}"),
    }
}

/// A claim the room holds is answered made at the first probe, and takes
/// nothing of the bank.
#[test]
fn a_claim_the_room_holds_is_answered_at_the_first_probe() {
    let Some((mut host, room, _, _)) = owner() else {
        return;
    };
    let mut grant = host.owner_mut().claim_measurement_room(room);
    host.observe();
    assert_eq!(answer(&mut grant), Some(RoomAnswer::Made));
    let census = experts(&host);
    assert_eq!((census.slots, census.last_resize), (4, None));
}

/// A claim two rows beyond the room shrinks the bank for the measurement
/// one row at a time, each shrink decided only once a probe read the one
/// before it finished, and is answered made at a probe after the second
/// finished: the rows the bank gave back are the room it reads.
#[test]
fn a_claim_beyond_the_room_is_made_by_the_banks_shrinks() {
    let Some((mut host, room, _, row)) = owner() else {
        return;
    };
    let mut grant = host.owner_mut().claim_measurement_room(room + 2 * row);
    let mut resizes = Vec::new();
    for observation in 0..OBSERVATIONS {
        host.observe();
        let census = experts(&host);
        if let Some(resize) = census.last_resize.filter(|resize| resizes.last().map(|(_, last)| last) != Some(resize)) {
            resizes.push((observation, resize));
        }
        let Some(answer) = answer(&mut grant) else {
            continue;
        };
        assert_eq!(answer, RoomAnswer::Made);
        let steps = resizes.iter().map(|(_, resize)| shrunk_to(resize)).collect::<Vec<_>>();
        assert_eq!(steps, [3, 2], "one row a shrink, as many as the claim was short");
        for pair in resizes.windows(2) {
            assert!(pair[1].0 - pair[0].0 >= 3, "a shrink begins, installs and releases at boundaries of its own: {resizes:?}");
        }
        let (made, _) = resizes.last().unwrap();
        assert!(observation > *made, "answered at a probe after the last shrink finished, not across it");
        assert_eq!(census.slots, 2);
        return;
    }
    panic!("the claim was not answered within {OBSERVATIONS} observations");
}

/// A claim short of the room by more than the bank holds above its floor
/// is refused with why at the first probe, and nothing shrinks for it.
#[test]
fn a_claim_beyond_what_the_bank_can_free_is_refused_before_any_shrink() {
    let Some((mut host, room, releasable, _)) = owner() else {
        return;
    };
    let mut grant = host.owner_mut().claim_measurement_room(room + releasable + 1);
    host.observe();
    let short = releasable + 1;
    assert_eq!(
        answer(&mut grant),
        Some(RoomAnswer::Refused(RoomRefusal::BeyondFloor { short, releasable }))
    );
    let census = experts(&host);
    assert_eq!((census.slots, census.last_resize), (4, None));
}
