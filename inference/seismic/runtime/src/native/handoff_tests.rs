use super::*;
use seismic_lang::checked::{NativeImplementation, SourceFile, SourceSet, check_source, check_source_with};
use seismic_lang::registry::{self, Layout};
use std::collections::HashMap;

/// The `project` module, its Metal source `asset`.
fn module_with(asset: &'static str) -> CheckedModule {
    check_source_with(
        SourceSet::new(vec![SourceFile {
            path: "handoff.seismic".into(),
            text: "fn project[N](weight: &tensor[N] W) -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    for i in 0..N:\n        result[i] = f32(weight[i])\n    return result\n\nnative project for metal from \"project.metal\":\n    static (N)\n    launch project:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".into(),
        }]),
        &mut |_entry: &str, _native: &NativeImplementation| {
            Ok::<_, std::convert::Infallible>(Some(asset.into()))
        },
    )
    .expect("test source checks")
}

fn module() -> CheckedModule {
    module_with("kernel void project() {}")
}

fn key(module: &CheckedModule, representation: RepresentationId, rows: u64) -> NativeHandoffKey {
    let entry = module.entry_named("project").unwrap();
    let logical = module
        .entry(entry, &ElementBindings::new().bind("W", representation))
        .unwrap();
    NativeHandoffKey::new(
        module,
        entry,
        &logical,
        &NativeSpecialization::new().with_static("N", rows),
        BackendName::Metal,
        None,
    )
}

#[test]
fn native_handoff_keeps_different_weight_representations_separate() {
    let module = module();
    let entry = module.entry_named("project").unwrap();
    let q4 = registry::storage("q4k", Layout::Rows32).unwrap();
    let q5 = registry::storage("q5k", Layout::Rows32).unwrap();
    let logical = |representation| {
        module
            .entry(entry, &ElementBindings::new().bind("W", representation))
            .unwrap()
    };
    // The old handoff key used this equal label and the same specialization.
    assert_eq!(logical(q4).identity(), logical(q5).identity());
    let mut chosen = HashMap::new();
    chosen.insert(key(&module, q4, 256), "q4 artifact");
    chosen.insert(key(&module, q5, 256), "q5 artifact");
    assert_eq!(chosen.len(), 2);
    assert_eq!(chosen.remove(&key(&module, q4, 256)), Some("q4 artifact"));
    assert_eq!(chosen.remove(&key(&module, q5, 256)), Some("q5 artifact"));
}

#[test]
fn native_handoff_reuses_exact_subjects_but_not_changed_assets_or_statics() {
    let original = module();
    let rebuilt = module();
    let q4 = registry::storage("q4k", Layout::Rows32).unwrap();
    assert_eq!(key(&original, q4, 256), key(&rebuilt, q4, 256));
    assert_ne!(key(&original, q4, 256), key(&rebuilt, q4, 512));
    let changed = module_with("kernel void project() { /* changed */ }");
    assert_ne!(key(&original, q4, 256), key(&changed, q4, 256));
}

#[test]
fn abandoned_native_handoff_does_not_keep_the_device_alive() {
    // Only formation and ownership are exercised; the function is never submitted.
    static KERNELS: CpuNativeKernels = CpuNativeKernels {
        digest: "handoff-lifetime-test",
        elements: &[],
        launches: &[CpuLaunchVariants {
            kernel: "project",
            variants: &crate::native::cpu::UNRUN_VARIANTS,
        }],
    };
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "handoff-lifetime.seismic".into(),
        text: "fn project[N](x: &tensor[N] f32) -> f32:\n    return x[0]\n\nnative project for cpu from \"project.rs\":\n    static (N)\n    launch project:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".into(),
    }])).unwrap();
    let devices = crate::devices::Devices::discover(&[]).unwrap();
    let device = devices.open_backend(BackendName::Cpu).unwrap();
    let weak = Arc::downgrade(&device);
    let formed = NativePrepared::prepare(
        &device,
        &module,
        module.entry_named("project").unwrap(),
        ElementBindings::new(),
        NativeSpecialization::new().with_static("N", 8),
        Some(&KERNELS),
    )
    .unwrap();
    formed.hand_off().unwrap();
    assert_eq!(device.chosen.lock().unwrap().len(), 1);
    let handed_scalars = device
        .chosen
        .lock()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .scalars
        .as_ref()
        .expect("a scalar result has slots")
        .slots
        .clone();
    let served = NativePrepared::prepare(
        &device,
        &module,
        module.entry_named("project").unwrap(),
        ElementBindings::new(),
        NativeSpecialization::new().with_static("N", 8),
        Some(&KERNELS),
    )
    .unwrap();
    assert!(device.chosen.lock().unwrap().is_empty());
    let slots = |kernel: &NativePrepared| kernel.scalar_slots().expect("a scalar result has slots").clone();
    assert!(Arc::ptr_eq(&handed_scalars, &slots(&served)));
    assert!(!Arc::ptr_eq(&slots(&formed), &slots(&served)));
    assert_eq!(formed.artifact, served.artifact);
    served.hand_off().unwrap();
    drop(handed_scalars);
    drop(served);
    drop(formed);
    drop(device);
    assert!(
        weak.upgrade().is_none(),
        "unused handoff retained its owning device"
    );
}

/// A tuning unit's handoff is its one current choice: each promotion
/// withdraws the choice it replaces with its device allocation, and the
/// serving preparation of the final choice takes the last one, so nothing
/// tuning handed off outlives the load on the device.
#[test]
fn a_promotion_withdraws_the_handoff_it_replaces_and_serving_takes_the_last() {
    static KERNELS: CpuNativeKernels = CpuNativeKernels {
        digest: "handoff-promotion-test",
        elements: &[],
        launches: &[CpuLaunchVariants {
            kernel: "project",
            variants: &crate::native::cpu::UNRUN_VARIANTS,
        }],
    };
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "handoff-promotion.seismic".into(),
        text: "fn project[N](x: &tensor[N] f32) -> f32:\n    return x[0]\n\nnative project for cpu from \"project.rs\":\n    static (N)\n    launch project:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".into(),
    }])).unwrap();
    let devices = crate::devices::Devices::discover(&[]).unwrap();
    let device = devices.open_backend(BackendName::Cpu).unwrap();
    // The handoff is keyed by specialization; the static stands for
    // whatever distinguishes one choice of a unit from the next.
    let form = |rows| {
        NativePrepared::prepare(
            &device,
            &module,
            module.entry_named("project").unwrap(),
            ElementBindings::new(),
            NativeSpecialization::new().with_static("N", rows),
            Some(&KERNELS),
        )
        .unwrap()
    };
    let held = |kernel: &NativePrepared| {
        Arc::downgrade(&device.chosen.lock().unwrap()[&kernel.handoff].scalars.as_ref().expect("a scalar result has slots").slots)
    };
    let choices = [form(8), form(16), form(32), form(64)];
    choices[0].hand_off().unwrap();
    let mut withdrawn = Vec::new();
    for pair in choices.windows(2) {
        withdrawn.push(held(&pair[0]));
        pair[1].hand_off_replacing(Some(&pair[0])).unwrap();
        let chosen = device.chosen.lock().unwrap();
        assert_eq!(chosen.len(), 1, "a unit holds one handoff");
        assert!(chosen.contains_key(&pair[1].handoff));
    }
    assert!(
        withdrawn.iter().all(|scalars| scalars.upgrade().is_none()),
        "a replaced choice kept its device allocation"
    );
    let last = held(&choices[3]);
    let served = form(64);
    assert!(device.chosen.lock().unwrap().is_empty());
    assert!(Arc::ptr_eq(&last.upgrade().unwrap(), served.scalar_slots().unwrap()));
    drop(served);
    assert!(last.upgrade().is_none());
}
