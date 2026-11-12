//! A header read from its skeleton in the service's header cache is assessed
//! as the whole header is: the cache stores each catalog target's header as
//! it stores any other, and the skeleton it gives back is fingerprinted,
//! admitted and inspected exactly as the whole header.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use magnitude_artifacts::catalog_headers::ComponentRole;
use magnitude_artifacts::gguf::{
    FileHeader, HeaderBytes, HeaderSource, Value, read_header, read_skeleton,
};
use magnitude_artifacts::{ArtifactIdentity, Error, PackageHeaders, PackageIdentity};
use magnitude_engine::assessment::{HeaderExtent, header_refusal};
use magnitude_engine::families::recognize;
use magnitude_engine::fingerprint::ModelFingerprint;
use magnitude_family_common::headers;
use magnitude_service_models::ModelCache;

const PHASES: [&str; 6] = [
    "skeleton read",
    "skeleton parse",
    "fingerprint",
    "admission",
    "skeleton total",
    "whole read+parse",
];

/// The headers of the files a package names, each stored in the header cache and read back as a
/// skeleton, with every payload but the token list's moved away while the skeleton is parsed:
/// parsing reads none of them. They are back in place for what is derived from the skeleton.
struct Skeletons {
    cache: ModelCache,
    /// The time spent reading skeletons from the cache, and parsing them.
    timing: Cell<[Duration; 2]>,
}

impl HeaderSource for Skeletons {
    fn header(&self, path: &Path) -> Result<FileHeader, Error> {
        let mut read = HeaderBytes::new();
        let data_offset = read.read(&mut std::fs::File::open(path)?)?.data_offset;
        let mut bytes = read.into_bytes();
        bytes.truncate(data_offset);
        let digest = self.cache.write_header(&bytes, BTreeSet::new);
        let tokens = match read_header(bytes)?.value("tokenizer.ggml.tokens") {
            Some(Value::Array(tokens)) => Some(tokens.digest()?),
            _ => None,
        };

        let started = Instant::now();
        let skeleton = self
            .cache
            .read_header_skeleton(&digest)
            .expect("the cache holds the header it stored");
        let read_time = started.elapsed();
        if let Some(tokens) = &tokens {
            assert!(
                skeleton
                    .absent
                    .iter()
                    .any(|absence| &absence.digest == tokens),
                "{path:?}: the token list is apart"
            );
        }
        let others = skeleton
            .absent
            .iter()
            .filter(|absence| Some(&absence.digest) != tokens.as_ref())
            .map(|absence| absence.file.clone())
            .collect::<Vec<_>>();
        let moved = |file: &PathBuf| file.with_extension("moved");
        for file in &others {
            std::fs::rename(file, moved(file))?;
        }
        let started = Instant::now();
        let directory = read_skeleton(skeleton);
        let parse_time = started.elapsed();
        for file in &others {
            std::fs::rename(moved(file), file)?;
        }
        let [read, parse] = self.timing.get();
        self.timing.set([read + read_time, parse + parse_time]);
        Ok(FileHeader {
            directory: directory?,
            size: std::fs::metadata(path)?.len(),
        })
    }
}

/// Every catalog target's skeleton, written to and read back from the
/// header cache, parses without its payloads apart other than the token
/// list's, and has the whole header's tensor table, fingerprint, admission
/// verdict and family definition (those derived with every payload in
/// place). Prints each phase's median, mean and maximum per model; the
/// skeleton total is what an assessment of a cached header spends before its
/// device work.
#[test]
#[ignore = "the catalog's headers: run on a rig host"]
fn every_catalog_targets_skeleton_assesses_as_its_whole_header() {
    let identity = PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    };
    let mut rows = Vec::<(String, [Duration; 6])>::new();
    for component in headers::index() {
        if component.role != ComponentRole::Target {
            continue;
        }
        let name = component.model.as_str();
        let file = component.file(&headers::root());

        // The whole header is the package's, every shard merged as production builds it.
        let started = Instant::now();
        let whole = PackageHeaders::open(&file, None)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .target()
            .clone();
        let whole_read = started.elapsed();

        let root = tempfile::tempdir().unwrap();
        let source = Skeletons {
            cache: ModelCache::new(root.path()),
            timing: Cell::new([Duration::ZERO; 2]),
        };
        let skeletal = PackageHeaders::open_from(&file, None, &source)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .target()
            .clone();
        let [skeleton_read, skeleton_parse] = source.timing.get();

        let started = Instant::now();
        let fingerprint = ModelFingerprint::from_directory(&skeletal);
        let fingerprinting = started.elapsed();
        let started = Instant::now();
        let refusal = header_refusal(&skeletal, HeaderExtent::Whole);
        let admission = started.elapsed();

        // The skeleton keeps exactly the whole header's directory.
        assert!(
            whole
                .tensors
                .iter()
                .any(|tensor| tensor.name == "token_embd.weight"),
            "{name}: the merged whole header holds the embedding"
        );
        assert_eq!(skeletal.tensors, whole.tensors, "{name}");
        assert_eq!(
            fingerprint,
            ModelFingerprint::from_directory(&whole),
            "{name}"
        );
        assert_eq!(
            refusal,
            header_refusal(&whole, HeaderExtent::Whole),
            "{name}"
        );
        let (family, whole_family) = (recognize(&skeletal), recognize(&whole));
        let Ok(family) = family else {
            // A target no family recognizes is refused the same way from either form.
            assert_eq!(
                format!("{:?}", family.err()),
                format!("{:?}", whole_family.err()),
                "{name}"
            );
            continue;
        };
        let whole_family = whole_family.unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(family.name(), whole_family.name(), "{name}");
        assert_eq!(
            format!("{:?}", family.inspect(&skeletal, None, identity)),
            format!("{:?}", family.inspect(&whole, None, identity)),
            "{name}"
        );
        assert_eq!(
            (skeletal.version, skeletal.alignment, skeletal.data_offset),
            (whole.version, whole.alignment, whole.data_offset),
            "{name}"
        );
        let total = skeleton_read + skeleton_parse + fingerprinting + admission;
        rows.push((
            name.to_owned(),
            [
                skeleton_read,
                skeleton_parse,
                fingerprinting,
                admission,
                total,
                whole_read,
            ],
        ));
    }
    assert!(!rows.is_empty());
    for (row, timings) in &rows {
        println!(
            "model {row}: {}",
            PHASES
                .iter()
                .zip(timings)
                .map(|(phase, time)| format!("{phase} {time:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    for (phase, label) in PHASES.iter().enumerate() {
        let mut times = rows.iter().map(|row| row.1[phase]).collect::<Vec<_>>();
        times.sort();
        let slowest = rows.iter().max_by_key(|row| row.1[phase]).unwrap();
        println!(
            "{label}: median {:?}, mean {:?}, max {:?} ({}) over {} models",
            times[times.len() / 2],
            times.iter().sum::<Duration>() / times.len() as u32,
            slowest.1[phase],
            slowest.0,
            rows.len()
        );
    }
}
