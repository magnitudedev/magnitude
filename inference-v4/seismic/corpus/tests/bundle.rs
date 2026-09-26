//! Every corpus scenario that checks round-trips through a checked bundle:
//! decoding the encoded module and encoding it again yields the same bytes
//! (spec 26-09-24 §5 test 1). The scenarios exercise the checked IR's
//! variants, so a field or variant the wire format drops fails here.
use crate::common::corpus_path;
use seismic_corpus::scenario::{self, ScenarioName};
use seismic_lang::bundle::{decode_checked_bundle, encode_checked_bundle};
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use std::path::{Path, PathBuf};

fn sources(directory: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).expect("the scenario directory reads") {
        let path = entry.expect("a scenario directory entry").path();
        if path.is_dir() {
            sources(&path, found);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "seismic")
        {
            found.push(path);
        }
    }
}

#[test]
fn checked_scenarios_round_trip_through_bundles() {
    let root = corpus_path("scenarios");
    let mut paths = Vec::new();
    sources(&root, &mut paths);
    paths.sort();
    let mut round_tripped = 0;
    let mut failures = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(&path).expect("the scenario reads");
        let name = path
            .strip_prefix(&root)
            .expect("under the root")
            .display()
            .to_string();
        // The header decides whether the standard library is linked; a
        // header the corpus parser rejects is the matrix suite's failure.
        let Ok(parsed) =
            std::panic::catch_unwind(|| scenario::parse(ScenarioName::new(name.clone()), &text))
        else {
            continue;
        };
        let mut sources = if parsed.include_std {
            seismic_std::sources()
        } else {
            SourceSet::default()
        };
        sources.push(SourceFile {
            path: name.clone(),
            text: parsed.source.clone(),
        });
        // Scenarios that expect a source error have no checked module.
        let Ok(checked) = check_source(sources) else {
            continue;
        };
        let encoded = encode_checked_bundle(&checked);
        match decode_checked_bundle(&encoded) {
            Ok(decoded) if encode_checked_bundle(&decoded) == encoded => round_tripped += 1,
            Ok(_) => failures.push(format!("{name}: re-encoding differs")),
            Err(error) => failures.push(format!("{name}: {error}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} scenario(s) fail:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(round_tripped > 0);
    eprintln!("{round_tripped} checked scenarios round-trip");
}
