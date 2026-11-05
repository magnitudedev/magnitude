//! What assessing models leaves live in the process. Its own test binary,
//! so the counting allocator sees no other test's allocations.

use flate2::read::GzDecoder;
use magnitude_engine::{
    assessment::{ModelPackagePaths, assess_model},
    options::{ModelMethod, ModelPolicy, PROFILE_SEQUENCES, standard_service_limits},
};
use magnitude_executor::platform::MemoryReserves;
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};

mod recorded;

/// The system allocator, counting the bytes live through it.
struct Counting;

static LIVE: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size() as i64, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size() as i64, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, size) };
        if !moved.is_null() {
            LIVE.fetch_add(size as i64 - layout.size() as i64, Ordering::Relaxed);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> i64 {
    LIVE.load(Ordering::Relaxed)
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/assessment")
}

/// Every fixture's package (`fixtures/assessment/manifest.json`), its
/// headers decompressed under `directory`.
fn packages(directory: &Path) -> Vec<(String, ModelPackagePaths)> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("manifest.json")).unwrap()).unwrap();
    let materialize = |component: &serde_json::Value| {
        let mut decoder = GzDecoder::new(
            std::fs::File::open(fixtures().join(component["header"].as_str().unwrap())).unwrap(),
        );
        let path = directory.join(
            Path::new(component["path"].as_str().unwrap())
                .file_name()
                .unwrap(),
        );
        std::io::copy(&mut decoder, &mut std::fs::File::create(&path).unwrap()).unwrap();
        path
    };
    manifest["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|fixture| {
            let name = fixture["name"].as_str().unwrap().to_owned();
            let directory = directory.join(&name);
            std::fs::create_dir_all(&directory).unwrap();
            let target = fixture["target"]
                .as_array()
                .unwrap()
                .iter()
                .map(materialize)
                .collect::<Vec<_>>();
            let package = ModelPackagePaths {
                target: target[0].clone(),
                projector: None,
                draft: fixture.get("draft").map(materialize),
                method: match fixture.get("method").and_then(|method| method.as_str()) {
                    None => ModelMethod::Auto,
                    Some("plain") => ModelMethod::Plain,
                    Some("mtp") => ModelMethod::Mtp,
                    Some("dflash") => ModelMethod::DFlash,
                    Some("dspark") => ModelMethod::DSpark,
                    Some("dflash2") => ModelMethod::DFlash2,
                    Some(method) => panic!("{name}: method {method}"),
                },
            };
            (name, package)
        })
        .collect()
}

/// What an assessment keeps is the checked module and what it forms once
/// per entry binding and geometry: assessing every fixture again keeps
/// nothing more.
#[test]
fn a_repeated_assessment_keeps_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let packages = packages(directory.path());
    let setup =
        recorded::m4_pro_setup(MemoryReserves::standard(), ModelPolicy::default(), standard_service_limits());
    let mut passes = Vec::new();
    for pass in 0..2 {
        let started = std::time::Instant::now();
        let mut kept = 0;
        for (name, package) in &packages {
            let before = live();
            let assessment = assess_model(package, &setup, &[], PROFILE_SEQUENCES);
            assert!(assessment.is_ok(), "{name}: {assessment:?}");
            drop(assessment);
            let added = live() - before;
            kept += added;
            println!("pass {pass} {name}: {added:+} bytes live");
        }
        println!(
            "pass {pass}: kept {kept} bytes, {:.2} s",
            started.elapsed().as_secs_f64()
        );
        passes.push(kept);
    }
    // The first pass formed everything an assessment keeps.
    assert_eq!(passes[1], 0);
}
