//! Nested declarations preserve writes to an enclosing tile binding.
use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::{execution::Config, msl::emit_storage_selected};
use seismic_realization::dispatch::TilePlacement;

fn emitted(control: &str, placement: TilePlacement) -> seismic_metal::msl::Emitted {
    let program = compile(
        &[SourceFile {
            path: "scope_bindings.seismic.portable".into(),
            scope: Scope::Portable,
            text: format!(
                "fn evaluate(flag:i32,out:tensor[3] i32):\n  t = tile[3] i32\n  for i in owned(t): t[i] = 10 + i\n  {control}\n    t = tile[3] i32\n    for i in owned(t): t[i] = 20 + i\n  store(t,out)\n"
            ),
        }],
        &[],
    )
    .unwrap();
    let function = lower(&program, "evaluate", "metal", &Default::default()).unwrap();
    emit_storage_selected(&function, Config::default(), &mut |decision| {
        assert!(decision.alternatives.contains(&placement));
        Ok(placement.clone())
    })
    .unwrap()
}

#[test]
fn nested_tile_rebinding_retains_selected_allocations() {
    for control in ["for outer in range(2):", "if flag > 0:"] {
        for placement in [
            TilePlacement::Replicated,
            TilePlacement::Distributed,
            TilePlacement::GroupShared,
        ] {
            let emitted = emitted(control, placement.clone());
            let tiles = emitted
                .launches
                .iter()
                .flat_map(|launch| &launch.tiles)
                .collect::<Vec<_>>();
            assert_eq!(tiles.len(), 2, "{control}, {placement:?}");
            assert!(tiles.iter().all(|tile| tile.placement == placement));
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn native_nested_tile_rebinding_preserves_enclosing_values() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    for control in ["for outer in range(2):", "if flag > 0:"] {
        for placement in [
            TilePlacement::Replicated,
            TilePlacement::Distributed,
            TilePlacement::GroupShared,
        ] {
            let emitted = emitted(control, placement.clone());
            let layout = emitted.scalar_layout().unwrap();
            let pipeline = device.compile(emitted).unwrap();
            let output = device.buffer(12).unwrap();
            for flag in [0.0, 1.0, 0.0] {
                let scalars = layout.encode(&[flag]).unwrap();
                device.run(&pipeline, &[&output], &scalars, 1).unwrap();
                let base = if control.starts_with("for") || flag > 0.0 {
                    20
                } else {
                    10
                };
                let actual = output
                    .read(12)
                    .chunks_exact(4)
                    .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual,
                    [base, base + 1, base + 2],
                    "{control}, {placement:?}, flag={flag}"
                );
            }
        }
    }
}
