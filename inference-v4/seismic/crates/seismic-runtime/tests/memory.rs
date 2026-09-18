use seismic_runtime::{Buffer, Device, Error};

#[test]
fn device_clones_views_and_completion_pins_share_one_charge() {
    let device = Device::cpu();
    device.set_memory_limit(Some(32)).unwrap();
    let peer = device.clone();
    let buffer = device.buffer(24).unwrap();
    let view = buffer.view(4..12).unwrap();
    let pin = view.clone();
    assert!(pin.belongs_to(&peer));
    assert!(!pin.belongs_to(&Device::cpu()));
    assert_eq!(peer.memory_usage().charged, 24);
    assert!(matches!(
        peer.buffer(9),
        Err(Error::Capacity {
            required: 9,
            available: 8
        })
    ));
    assert_eq!(Buffer::reclaimable_bytes([&buffer, &view]).unwrap(), 0);
    drop(buffer);
    drop(view);
    assert_eq!(device.memory_usage().charged, 24);
    assert_eq!(Buffer::reclaimable_bytes([&pin]).unwrap(), 24);
    drop(pin);
    assert_eq!(device.memory_usage().charged, 0);
    let whole = peer.buffer(32).unwrap();
    assert_eq!(device.memory_usage().charged, 32);
    assert!(device.set_memory_limit(Some(31)).is_err());
    assert_eq!(device.memory_usage().limit, Some(32));
    drop(whole);
    assert_eq!(peer.memory_usage().charged, 0);
}

#[test]
fn failed_allocation_and_partial_preparation_release_charges() {
    let device = Device::cpu();
    device.set_memory_limit(Some(24)).unwrap();
    let prepare = || -> Result<Vec<Buffer>, Error> {
        let first = device.buffer(16)?;
        let second = device.buffer(16)?;
        Ok(vec![first, second])
    };
    assert!(matches!(
        prepare(),
        Err(Error::Capacity {
            required: 16,
            available: 8
        })
    ));
    assert_eq!(device.memory_usage().charged, 0);
    device.set_memory_limit(None).unwrap();
    // CPU backing storage cannot represent this extent. Charging is unwound
    // even when the backend rejects an allocation admitted by the domain.
    assert!(matches!(device.buffer(usize::MAX), Err(Error::Failure(_))));
    assert_eq!(device.memory_usage().charged, 0);
    let buffer = device.buffer_from(&[1, 2, 3, 4]).unwrap();
    assert_eq!(device.memory_usage().charged, 4);
    let other = Device::cpu();
    assert_eq!(other.memory_usage().charged, 0);
    drop(buffer);
    assert_eq!(device.memory_usage().charged, 0);
}
