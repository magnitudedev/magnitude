//! The live page probe (`platform::PageProbe::run`), run on this machine,
//! timed: the room it read, each fill sample's
//! bytes (the largest the probe's peak), whether the bound cut the fill
//! short, both rates and the wall time. It takes host memory, so it runs
//! with the machine otherwise idle.
//!
//! `cargo run --release -p magnitude-executor --example page_probe`

use magnitude_executor::platform::PageProbe;

fn main() {
    let started = std::time::Instant::now();
    let probe = PageProbe::run();
    let wall = started.elapsed();
    let peak = probe.fill_samples.iter().max().copied().unwrap_or(0);
    println!(
        "room {:.2} GB; fill samples {:?} B, peak {:.2} GB; fill {:.2} GB/s (censored: {}); release {:.1} GB/s; wall {:.3} s",
        probe.room as f64 / 1e9,
        probe.fill_samples,
        peak as f64 / 1e9,
        probe.fill / 1e9,
        probe.censored,
        probe.release / 1e9,
        wall.as_secs_f64()
    );
}
