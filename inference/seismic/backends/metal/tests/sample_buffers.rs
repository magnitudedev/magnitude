//! How many timestamp sample buffers Metal lets exist, which bounds the
//! launch-timestamp page pool (`direct.rs`, `LaunchTimestamps`). Run
//! explicitly:
//! `cargo test -p seismic-metal --release --test sample_buffers -- --ignored --nocapture`.
//!
//! M4 Pro, macOS 15.7.9, 2026-10-24:
//! - a buffer holds at most 32768 bytes (4096 samples); 4097 is refused;
//! - the 33rd buffer is refused ("Cannot allocate sample buffer"), at 2
//!   samples as at 4096: the bound is a count;
//! - freed buffers can be formed again (32 again);
//! - a second process holding 16 leaves this one its 32: the bound is per
//!   process.
#![cfg(target_os = "macos")]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice, MTLStorageMode,
};

type Buffer = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

fn sample_buffer(device: &ProtocolObject<dyn MTLDevice>, samples: usize) -> Result<Buffer, String> {
    let set = device
        .counterSets()
        .and_then(|sets| {
            // SAFETY: a static Metal string constant.
            let wanted = unsafe { MTLCommonCounterSetTimestamp };
            sets.iter().find(|set| &*set.name() == wanted)
        })
        .expect("a timestamp counter set");
    let descriptor = MTLCounterSampleBufferDescriptor::new();
    descriptor.setCounterSet(Some(&set));
    descriptor.setStorageMode(MTLStorageMode::Shared);
    // SAFETY: Metal validates the count when forming the buffer.
    unsafe { descriptor.setSampleCount(samples) };
    device
        .newCounterSampleBufferWithDescriptor_error(&descriptor)
        .map_err(|error| error.localizedDescription().to_string())
}

/// Buffers of `samples` samples formed until Metal refuses one (at most
/// `limit`), and the refusal.
fn fill(device: &ProtocolObject<dyn MTLDevice>, samples: usize, limit: usize) -> (Vec<Buffer>, Option<String>) {
    let mut buffers = Vec::new();
    while buffers.len() < limit {
        match sample_buffer(device, samples) {
            Ok(buffer) => buffers.push(buffer),
            Err(error) => return (buffers, Some(error)),
        }
    }
    (buffers, None)
}

const HOLD: &str = "SEISMIC_SAMPLE_BUFFERS_HOLD";

#[test]
#[ignore = "device probe; run explicitly with --ignored --nocapture"]
fn sample_buffer_bound() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
    if let Ok(count) = std::env::var(HOLD) {
        // The child of the cross-process probe: hold `count` buffers until
        // the parent closes stdin.
        let (held, refused) = fill(&device, 4_096, count.parse().unwrap());
        println!("held {} refused {refused:?}", held.len());
        let mut line = String::new();
        let _ = std::io::stdin().read_line(&mut line);
        return;
    }
    for samples in [4_096, 4_097, 8_192] {
        println!("one buffer of {samples} samples: {:?}", sample_buffer(&device, samples).map(|_| ()));
    }
    for samples in [2, 4_096] {
        let (buffers, refused) = fill(&device, samples, 100_000);
        println!("{samples}-sample buffers formed: {}, then {refused:?}", buffers.len());
    }
    // Freed buffers return their share.
    let (first, _) = fill(&device, 4_096, 100_000);
    drop(first);
    let (again, _) = fill(&device, 4_096, 100_000);
    println!("after freeing, 4096-sample buffers formed again: {}", again.len());
    drop(again);
    // Whether another process's buffers count against this one's.
    let half = {
        let (buffers, _) = fill(&device, 4_096, 100_000);
        buffers.len() / 2
    };
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "sample_buffer_bound", "--nocapture"])
        .env(HOLD, half.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut reader = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        std::io::BufRead::read_line(&mut reader, &mut line).unwrap();
        if line.starts_with("held") || line.is_empty() {
            break;
        }
    }
    println!("another process: {}", line.trim());
    let (beside, refused) = fill(&device, 4_096, 100_000);
    println!("beside it, 4096-sample buffers formed: {}, then {refused:?}", beside.len());
    drop(child.stdin.take());
    child.wait().unwrap();
}
