//! The kernel bundle the build embeds carries the checked module the build's
//! checker produced: decoding it rebuilds exactly that module without
//! checking. Checking its source section again yields the same module up to
//! the order in which the checker interns arena nodes, which is not stable
//! across checks; `seismic-runtime`'s native bundle identity test compares
//! the two by the native sources they form.

use seismic_lang::bundle::{check_bundle_sources, decode_checked_bundle, encode_checked_bundle};
use seismic_lang::checked::check_source;
use std::time::Instant;

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
    assert!(encode_checked_bundle(&decoded) == BUNDLE, "decode, encode is not the identity");
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
            (&decoded.name, &decoded.stable, &decoded.parameter_types, &decoded.element_domain),
            (&checked.name, &checked.stable, &checked.parameter_types, &checked.element_domain)
        );
    }
}

#[test]
fn standard_library_round_trips_byte_identically() {
    let checked = check_source(seismic_std::sources()).expect("the standard library checks");
    let encoded = encode_checked_bundle(&checked);
    let decoded = decode_checked_bundle(&encoded).expect("the standard library bundle decodes");
    assert!(encode_checked_bundle(&decoded) == encoded);
}
