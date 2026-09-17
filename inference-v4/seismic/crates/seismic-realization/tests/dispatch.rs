use seismic_lang::types::DType;
use seismic_realization::dispatch::{GroupDispatch, TileDeclaration, TilePlacement};
#[test]
fn declaration_units_and_padding_are_explicit() {
    let dispatch = GroupDispatch::new(7, 32, 4).unwrap();
    assert_eq!(dispatch.dispatched_lanes(), 256);
    assert_eq!(dispatch.participating_lanes(), 224);
    assert_eq!(dispatch.padding_lanes(), 32);
    let tile = |placement| TileDeclaration {
        symbol: "tile".into(),
        dtype: DType::BF16,
        capacity: 65,
        placement,
    };
    assert_eq!(
        tile(TilePlacement::Replicated).bytes(&dispatch).unwrap(),
        (130, 0)
    );
    assert_eq!(
        tile(TilePlacement::Distributed).bytes(&dispatch).unwrap(),
        (6, 0)
    );
    assert_eq!(
        tile(TilePlacement::GroupShared).bytes(&dispatch).unwrap(),
        (0, 520)
    );
    assert!(GroupDispatch::new(u64::MAX, 32, 4).is_err());
    assert!(GroupDispatch::new(1, 0, 4).is_err());
}
