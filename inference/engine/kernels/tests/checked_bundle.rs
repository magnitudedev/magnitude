//! The kernel bundle the build embeds carries the checked module the build's
//! checker produced: decoding it rebuilds exactly that module without
//! checking. Checking its source section again yields the same module up to
//! the order in which the checker interns arena nodes, which is not stable
//! across checks; `seismic-runtime`'s native bundle identity test compares
//! the two by the native sources they form.

use seismic_lang::bundle::{check_bundle_sources, decode_checked_bundle, encode_checked_bundle};
use seismic_lang::checked::check_source;
use std::time::Instant;

mod declared;

static BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.seismicbundle"));

#[test]
fn embedded_kernel_bundle_round_trips_byte_identically() {
    let started = Instant::now();
    let decoded = decode_checked_bundle(BUNDLE).expect("the embedded kernel bundle decodes");
    eprintln!(
        "kernel bundle: {} bytes, {} entries, decoded in {:?}",
        BUNDLE.len(),
        decoded.entries().len(),
        started.elapsed()
    );
    assert!(
        encode_checked_bundle(&decoded) == BUNDLE,
        "decode, encode is not the identity"
    );
}

/// Slow: runs the checker over the whole kernel library.
#[test]
#[ignore]
fn embedded_kernel_bundle_matches_its_checked_sources() {
    let decoded = decode_checked_bundle(BUNDLE).expect("the embedded kernel bundle decodes");
    let started = Instant::now();
    let checked = check_bundle_sources(BUNDLE).expect("the kernel bundle's sources check");
    eprintln!("kernel bundle sources checked in {:?}", started.elapsed());
    assert_eq!(decoded.semantic_hash(), checked.semantic_hash());
    assert_eq!(decoded.entries().len(), checked.entries().len());
    for (decoded, checked) in decoded.entries().iter().zip(checked.entries()) {
        assert_eq!(
            (
                &decoded.name,
                &decoded.stable,
                &decoded.parameter_types,
                &decoded.element_domain
            ),
            (
                &checked.name,
                &checked.stable,
                &checked.parameter_types,
                &checked.element_domain
            )
        );
    }
}

#[test]
fn standard_library_round_trips_byte_identically() {
    let checked = check_source(seismic_std::sources()).expect("the standard library checks");
    // Held for the process, as a build embeds a bundle.
    let encoded: &'static [u8] = encode_checked_bundle(&checked).leak();
    let decoded = decode_checked_bundle(encoded).expect("the standard library bundle decodes");
    assert!(encode_checked_bundle(&decoded) == encoded);
}

/// A draft verification's rows compute the bits plain decode computes for
/// them: on every backend, a launch whose activity depends on the row count
/// at some count up to the rows a verification carries
/// ([`magnitude_kernels::constants::VERIFY_ROWS`]) is a row class (`alike`
/// a launch, or the launch it names: `derive_codes` holds each to the same
/// reduction orders) or a staging launch. That catches a batched form, a
/// GEMM or a router that starts at or below the verification's rows, whatever
/// bound declares it, and no native is exempt: one whose decode forms end
/// below `VERIFY_ROWS` fails. Staging launches, which only prepare operands
/// and compute no row, are recognised by their kernel names ending in
/// `_stage` or `_normalize`; that is the one name rule here. Launches are
/// tested at the smallest and the largest value of every parameter the
/// declaration's `where` admits, so no value of a domain breaks it.
#[test]
fn launches_chosen_by_rows_up_to_a_draft_verification_are_row_classes() {
    use seismic_lang::checked::NativeSpecialization;
    use seismic_lang::registry::BackendName;
    let verify = magnitude_kernels::constants::VERIFY_ROWS;
    // The widest verify body on every backend is 8 rows (Metal's
    // PROJECTION_FOR_SEVERAL_ROWS, Vulkan's PROJECTION_GEMV_MAX_M): a wider
    // verification has no class there. The bodies assert the same bound.
    assert!(verify <= 8, "VERIFY_ROWS {verify} exceeds the 8 rows the verify bodies serve");
    let module = decode_checked_bundle(BUNDLE).expect("the embedded kernel bundle decodes");
    let (mut classes, mut offenders) = (0, Vec::new());
    for entry in module.entries() {
        for backend in [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan, BackendName::Cpu] {
            let Some(native) = module.native_implementation(entry.id, backend) else {
                continue;
            };
            let row_launches = declared::row_launches(native);
            classes += declared::row_classes(native).len();
            for largest in [false, true] {
                let mut base = NativeSpecialization::new();
                for parameter in &native.params {
                    // The values the declaration's `where` admits for it alone.
                    let admitted = parameter.values.iter().filter(|&&value| {
                        native.constraint.as_ref().is_none_or(|constraint| {
                            constraint.may_hold(&|_: &str| None, &|name: &str| {
                                (name == parameter.name).then_some((value, value))
                            })
                        })
                    });
                    let value = if largest { admitted.max() } else { admitted.min() };
                    base = base.with_param(parameter.name.clone(), *value.expect("a parameter has an admitted value"));
                }
                let at = |rows: u64| {
                    let mut specialization = base.clone();
                    for dimension in ["M", "O"] {
                        if !native.statics.iter().any(|name| name == dimension) {
                            specialization = specialization.with_range(dimension, rows, rows);
                        }
                    }
                    (0..native.launches.len())
                        .map(|launch| native.reachable(&specialization, launch))
                        .collect::<Vec<_>>()
                };
                let active = (1..=verify).map(at).collect::<Vec<_>>();
                for (launch, declared) in native.launches.iter().enumerate() {
                    let some = active.iter().any(|rows| rows[launch]);
                    let every = active.iter().all(|rows| rows[launch]);
                    if some && !every && !declared.staging && !row_launches.contains(&launch) {
                        offenders.push(format!(
                            "{} on {backend:?}: `{}` is chosen by the row count up to {verify} and is no row class",
                            entry.name, declared.kernel
                        ));
                    }
                }
            }
        }
    }
    offenders.sort();
    offenders.dedup();
    assert!(classes > 0, "the bundle has row-class launches");
    assert!(offenders.is_empty(), "launches chosen by rows below a draft verification's:\n{}", offenders.join("\n"));
}
