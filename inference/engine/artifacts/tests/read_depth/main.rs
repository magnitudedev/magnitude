//! The read engine against the device it reads, uncached, on model files:
//! the read depth measurements ([`measure`]) run once on a small synthetic
//! file of a model's layout, their results checked for shape; and, timed,
//! on a real model file, the target of the acquisition object
//! `MAGNITUDE_ARTIFACTS` names, its experts' layout from its GGUF tensor
//! table, with a prompt's reads against the volume's rate. The timed runs
//! are alone on the device (`-p magnitude-artifacts --release --test
//! read_depth -- --ignored --nocapture`).

#![cfg(unix)]

mod measure;

use measure::{lone, rates, uncached, Model, Rates, Rng, Sinks, ALIGN, SPLITS};
use seismic_io::{CachePolicy, Range, Reader, Want};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// A small file of two layers, each three stacks of four experts of
/// unequal unaligned bytes, as a GGUF lays them, removed when it drops.
fn synthetic(name: &str) -> Model {
    let (experts, parts) = (4u64, [36_864u64 + 32, 36_864 + 64, 45_056 + 96]);
    let mut layers = Vec::new();
    let mut at = 4096 + 32;
    for _ in 0..2 {
        let mut stacks = Vec::new();
        for per in parts {
            stacks.push((at, per));
            at += per * experts;
        }
        layers.push(stacks);
    }
    let size = (at + (4 << 20)).next_multiple_of(ALIGN);
    let path = std::env::temp_dir().join(format!("magnitude-read-depth-{name}-{}", std::process::id()));
    std::fs::write(&path, vec![7u8; size as usize]).unwrap();
    Model { path: path.to_str().unwrap().to_owned(), size, layers, experts, temporary: true }
}

/// Each measurement once on a small synthetic file of a model's layout:
/// every line it reports, each a finite positive time or rate. Its values
/// are not judged; its shape is, so a change to what it calls breaks it in
/// the suite.
#[test]
fn the_measurements_run_once_on_a_small_file() {
    let model = synthetic("smoke");
    let lines = rates(
        &model,
        &Rates { alone_steps: 1, batches: &[(4, 2)], batch_steps: 1, raw_bytes: &[128 << 10], raw_depths: &[2], raw_seconds: 0.01 },
    );
    assert_eq!(lines.len(), 3, "the record alone, one batch, one raw line");
    let lone = lone(&model, 1, 4);
    assert_eq!(lone.len(), 2 * 3 * SPLITS.len() + 3 + 3 * (SPLITS.len() - 1), "each issue, kind and split, the engine's each kind, and at each split's demand read size");
    for (label, value) in lines.iter().chain(&lone) {
        assert!(value.is_finite() && *value > 0., "{label}: {value}");
    }
}

/// The model file `MAGNITUDE_ARTIFACTS` names, its stacks from its
/// GGUF tensor table (`blk.N.` layers' `_exps` tensors), only those
/// that lie in this file (a sharded model's others address later
/// shards).
fn acquired() -> Model {
    let artifacts = std::env::var_os("MAGNITUDE_ARTIFACTS").expect("set MAGNITUDE_ARTIFACTS");
    let acquisition: serde_json::Value = serde_json::from_slice(&std::fs::read(&artifacts).unwrap()).unwrap();
    let path = acquisition["target"].as_str().unwrap().to_owned();
    let artifact = magnitude_artifacts::gguf::GgufArtifact::open(&path).unwrap();
    let mut layers = std::collections::BTreeMap::<u64, std::collections::BTreeMap<String, (u64, u64)>>::new();
    let mut experts = 0;
    for name in artifact.directory().tensors.iter().map(|tensor| tensor.name.clone()).collect::<Vec<_>>() {
        let Some((layer, tensor)) = name.strip_prefix("blk.").and_then(|rest| rest.split_once('.')) else { continue };
        if !tensor.ends_with("_exps.weight") {
            continue;
        }
        let slice = artifact.tensor(&name).unwrap();
        experts = *slice.shape.last().unwrap();
        layers.entry(layer.parse().unwrap()).or_default().insert(tensor.to_owned(), (slice.offset, slice.nbytes / experts));
    }
    let size = std::fs::metadata(&path).unwrap().len();
    let layers = layers
        .into_values()
        .map(|stacks| stacks.into_values().collect::<Vec<_>>())
        .filter(|stacks| stacks.iter().all(|&(offset, per)| offset + per * experts <= size))
        .collect::<Vec<_>>();
    assert!(!layers.is_empty(), "{path} has no expert stacks");
    Model { path, size, layers, experts, temporary: false }
}

#[test]
#[ignore = "times the read engine against the device on a real model file (`MAGNITUDE_ARTIFACTS`)"]
fn the_engine_reads_at_the_devices_rate() {
    let model = acquired();
    rates(
        &model,
        &Rates {
            alone_steps: 200,
            batches: &[(16, 16), (175, 5), (175, 175)],
            batch_steps: 20,
            raw_bytes: &[128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20],
            raw_depths: &[4, 8, 16, 32, 64, 128],
            raw_seconds: 0.75,
        },
    );
}

/// A prompt's cold experts: every layer's `experts` distinct random
/// experts, each its stored ranges.
fn prompt_experts(model: &Model, rng: &mut Rng, experts: usize) -> Vec<Vec<Vec<(u64, usize)>>> {
    model
        .layers
        .iter()
        .map(|layer| {
            let mut order = (0..model.experts).collect::<Vec<_>>();
            for at in (1..order.len()).rev() {
                order.swap(at, (rng.next() % (at as u64 + 1)) as usize);
            }
            order[..experts].iter().map(|&expert| layer.iter().map(|&(offset, per)| (offset + expert * per, per as usize)).collect()).collect()
        })
        .collect()
}

/// The engine reading a prompt's experts as a prompt's conveyor does: one
/// due request per expert, two layers in flight. Bytes a second.
fn engine_prompt(reader: &Reader, layers: &[Vec<Vec<(u64, usize)>>], record: usize) -> f64 {
    let experts = layers[0].len();
    let sinks = [Sinks::new(experts, record), Sinks::new(experts, record)];
    let bytes: usize = layers.iter().flatten().flatten().map(|&(_, length)| length).sum();
    let began = Instant::now();
    let mut behind: Vec<seismic_io::Handle> = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let jobs = layer
            .iter()
            .enumerate()
            .map(|(expert, ranges)| {
                let mut at = 0;
                let ranges = ranges
                    .iter()
                    .map(|&(offset, length)| {
                        let into = &sinks[index % 2];
                        let range = Range { offset, length, into: into.destination(), at: into.at(expert) + at };
                        at += length;
                        range
                    })
                    .collect();
                unsafe { reader.request(ranges, Want::now()) }.unwrap()
            })
            .collect::<Vec<_>>();
        for job in std::mem::replace(&mut behind, jobs) {
            assert!(matches!(job.wait(), seismic_io::Ended::Read { failed } if failed.is_empty()));
        }
    }
    for job in behind {
        assert!(matches!(job.wait(), seismic_io::Ended::Read { failed } if failed.is_empty()));
    }
    bytes as f64 / began.elapsed().as_secs_f64()
}

/// The experts' ranges aligned and cut at `piece`, as the engine cuts them.
fn prompt_pieces(layers: &[Vec<Vec<(u64, usize)>>], piece: u64) -> Vec<(u64, usize)> {
    layers
        .iter()
        .flatten()
        .flatten()
        .flat_map(|&(offset, length)| {
            let (start, end) = (offset / ALIGN * ALIGN, (offset + length as u64).next_multiple_of(ALIGN));
            (start..end).step_by(piece as usize).map(move |at| (at, (end - at).min(piece) as usize))
        })
        .collect()
}

/// Raw threads reading the same experts' pieces ([`prompt_pieces`]),
/// `depth` threads each taking the next. Bytes a second.
fn raw_prompt(model: &Model, layers: &[Vec<Vec<(u64, usize)>>], piece: u64, depth: usize) -> f64 {
    let pieces = prompt_pieces(layers, piece);
    let bytes: usize = layers.iter().flatten().flatten().map(|&(_, length)| length).sum();
    let (pieces, next) = (Arc::new(pieces), Arc::new(AtomicU64::new(0)));
    let began = Instant::now();
    let threads = (0..depth)
        .map(|_| {
            let (pieces, next, path) = (pieces.clone(), next.clone(), model.path.clone());
            std::thread::spawn(move || {
                let file = uncached(&path);
                let mut buffer = vec![0u8; (piece + ALIGN) as usize];
                let at = buffer.as_ptr().align_offset(ALIGN as usize);
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed) as usize;
                    let Some(&(offset, length)) = pieces.get(index) else { break };
                    file.read_exact_at(&mut buffer[at..at + length], offset).unwrap();
                }
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
    bytes as f64 / began.elapsed().as_secs_f64()
}

/// A prompt's reads, the engine against raw threads over the same experts
/// and against the bar's pattern (whole records at a depth): whether the
/// engine's shape of a prompt's reads or the window it holds them in is
/// what keeps it below the volume's sustained rate.
#[test]
#[ignore = "times a prompt's reads against the device on a real model file (`MAGNITUDE_ARTIFACTS`)"]
fn a_prompt_reads_at_the_volumes_rate() {
    let model = acquired();
    let urgent = model.urgent();
    let source = seismic_io::OpenFile::open(&model.path).unwrap();
    seismic_io::probe_volume(&source, Some(&urgent)).unwrap();
    let probed = seismic_io::link_shape(&source, Some(&urgent)).unwrap();
    eprintln!("link: {probed}");
    // Layers store their experts in unequal quantizations: each sink holds the widest.
    let record = model.layers.iter().map(|layer| layer.iter().map(|&(_, per)| per as usize).sum::<usize>()).max().unwrap();
    let experts = (model.experts as usize * 3 / 4).min(190);
    let mut rng = Rng(0x9a0b);
    let gb = |rate: f64| rate / 1e9;
    let reader = Reader::of_urgent(&source, CachePolicy::Uncached, &urgent).unwrap();
    let pieces = prompt_pieces(&prompt_experts(&model, &mut rng, experts), probed.due_read);
    let mean = pieces.iter().map(|&(_, length)| length as f64).sum::<f64>() / pieces.len() as f64;
    let depth = ((probed.due as f64 / mean).round() as usize).max(1);
    reader.evict_cached().unwrap();
    let layers = prompt_experts(&model, &mut rng, experts);
    eprintln!("engine: {:.2} GB/s", gb(engine_prompt(&reader, &layers, record)));
    reader.evict_cached().unwrap();
    let layers = prompt_experts(&model, &mut rng, experts);
    eprintln!("raw, the same pieces (mean {:.0} KiB) at depth {depth}: {:.2} GB/s", mean / 1024., gb(raw_prompt(&model, &layers, probed.due_read, depth)));
}

#[test]
#[ignore = "times lone requests on a real model file (`MAGNITUDE_ARTIFACTS`)"]
fn a_lone_request() {
    lone(&acquired(), 200, 64);
}
