//! Every historical defect H-1..H-53 is guarded (design A10 §2.6).
use crate::common::corpus_path;
use seismic_corpus::library;
use seismic_corpus::scenario::{load_all, Origin};

/// Origins guarded by something other than a scenario, each for the reason of §2.6.
const EXEMPT: &[(u8, Option<char>)] = &[
    // Intrinsic row selection: A1's `IntrinsicOverload` checker test and the library matmul lines.
    (25, None),
    // The envelope box-subtraction mechanism no longer exists: forbidden symbols guard it.
    (26, Some('b')),
    // CUDA vector bitcast emission: no CUDA device; the PTX unit test guards it.
    (31, None),
    // A type-level "retained" unresolved candidate: the `member_index_literal` compile-fail fixture.
    (33, None),
];

#[test]
fn every_historical_defect_has_a_scenario() {
    let mut guarded: Vec<Origin> = load_all(&corpus_path("scenarios"))
        .into_iter()
        .flat_map(|scenario| scenario.origins)
        .collect();
    for file in ["library/std.invocations", "library/engine.invocations"] {
        let library = library::load(&corpus_path(file));
        guarded.extend(library.origins);
        guarded.extend(library.invocations.into_iter().flat_map(|i| i.origins));
    }
    let required = (1..=53u8)
        .map(|number| (number, None))
        .chain([(26, Some('b')), (26, Some('c'))])
        .filter(|origin| !EXEMPT.contains(origin));
    let missing: Vec<String> = required
        .map(|(number, suffix)| Origin::History { number, suffix })
        .filter(|origin| !guarded.contains(origin))
        .map(|origin| origin.to_string())
        .collect();
    assert!(
        missing.is_empty(),
        "{} historical defect(s) have no scenario or library line: {}",
        missing.len(),
        missing.join(" ")
    );
}
