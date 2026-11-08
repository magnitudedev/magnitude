//! A prompt group's places under memory pressure, over the controlled domain
//! (`magnitude_executor::fixtures`): its places are the settled target's,
//! and only real overcommit refuses them, read by the heap's claim of their
//! bytes. Another program's hold on the device's store (`press`) bounds the
//! store's ledger, which every domain of the process reads, so these run in
//! their own test binary, one test, one press at a time.

use magnitude_executor::{
    fixtures::{chained, group_opening, measure_load, press},
    platform::DeviceRequest,
};

/// A load whose plan holds its groups' class of three places opens a group
/// of four chunks with another program holding its device's store below
/// the planning line: in the Reclaim band, the group takes no place beyond
/// the one it holds.
#[test]
fn a_group_takes_no_places_in_reclaim() {
    let (mut domain, mut bindings) = chained(DeviceRequest::Automatic);
    measure_load(&mut domain, &mut bindings);
    let operations = group_opening(&mut domain, &mut bindings);
    let press = press(&domain);
    let reading = domain
        .observe_memory()
        .unwrap()
        .into_iter()
        .find(|reading| reading.domain == press.store())
        .expect("the device's store is read");
    press.take(reading.headroom_bytes - reading.thresholds.planning_bytes + 1);
    // What else the group's provision grows is refused as well.
    let _ = domain.provision(&mut bindings, &operations, None);
    assert_eq!(domain.group_places().get(), 1, "the group takes no place beyond the one it holds");
    press.release();
}
