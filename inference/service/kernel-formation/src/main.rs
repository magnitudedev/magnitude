//! Forms every native kernel the engine ships with one GPU backend's real
//! toolchain (every kernel, every preprocessor group of its sources, every
//! form value), then every kernel every supported catalog model requests.
//!
//!   magnitude-kernel-formation <vulkan|cuda|metal> --ledger <path> --catalog <bundle>
//!
//! Formation goes through a ledger of what earlier runs established
//! ([`forming`], [`ledger`]). Coverage whose sources and toolchain match a
//! complete coverage run reports that run and searches nothing; catalog
//! requests are always enumerated, and those the ledger holds are recalled.
//! Any failure, unreached group, unformed form value, configuration this host
//! cannot form, or catalog request that does not form fails the run.

mod forming;
mod ledger;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use forming::Ledgered;
use ledger::{Key, Ledger};
use magnitude_service_server::assessment::catalog_kernels::catalog_kernel_requests;
use seismic::BackendName;
use seismic::coverage::{Progress, Toolchain};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Arguments {
    #[arg(value_enum)]
    backend: Backend,
    /// The ledger of what earlier runs established.
    #[arg(long)]
    ledger: PathBuf,
    /// The release header bundle the catalog is read from.
    #[arg(long)]
    catalog: PathBuf,
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Vulkan,
    Cuda,
    Metal,
}

impl From<Backend> for BackendName {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Vulkan => Self::Vulkan,
            Backend::Cuda => Self::Cuda,
            Backend::Metal => Self::Metal,
        }
    }
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match form(
        arguments.backend.into(),
        &arguments.ledger,
        &arguments.catalog,
    ) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Form for `backend`; whether everything formed.
fn form(backend: BackendName, ledger: &Path, catalog: &Path) -> anyhow::Result<bool> {
    let name = backend.as_str();
    let toolchain = Toolchain::open(backend).map_err(|error| anyhow::anyhow!("{name}: {error}"))?;
    let ledger = Ledger::open(ledger).with_context(|| format!("opening {}", ledger.display()))?;
    // Coverage's result is a function of the sources the build script
    // digests and of the toolchain: an unchanged key is an unchanged result.
    let coverage_run = Key::of(&[
        b"kernel formation coverage",
        toolchain.identity().as_bytes(),
        env!("KERNEL_FORMATION_SEARCH_SOURCES").as_bytes(),
    ]);
    let forming = Ledgered::new(ledger);
    let module = magnitude_kernels::module().context("decoding the kernel bundle")?;
    let started = Instant::now();
    let progress = Progress::default();

    let covered = match forming.ledger().completion(&coverage_run) {
        Some(summary) => {
            println!("{name}: coverage unchanged since a complete run: {summary}");
            true
        }
        None => {
            let coverage = reporting(
                &started,
                &format!("{name} coverage"),
                "entries searched",
                &progress,
                &forming,
                || seismic::coverage::cover(module, &toolchain, &forming, &progress),
            )
            .map_err(|error| anyhow::anyhow!("{name} coverage did not run: {error}"))?;
            println!("{name}: {coverage}");
            let covered = coverage.complete() && coverage.unformable.is_empty();
            if covered {
                forming
                    .ledger()
                    .complete(&coverage_run, coverage.to_string().trim());
            }
            covered
        }
    };

    let catalog = catalog_kernel_requests(catalog, backend)?;
    for (model, reason) in &catalog.unsupported {
        println!("{name}: `{model}` is unsupported: {reason}");
    }
    let requests = catalog
        .requests
        .iter()
        .map(|request| request.request.clone())
        .collect::<Vec<_>>();
    let results = reporting(
        &started,
        &format!("{name} catalog"),
        "requests formed",
        &progress,
        &forming,
        || seismic::coverage::form_requests(module, &toolchain, &forming, &requests, &progress),
    );
    let mut failures = 0;
    for (request, result) in catalog.requests.iter().zip(results) {
        if let Err(failure) = result {
            failures += 1;
            println!(
                "catalog request failure: {}: {failure} (first requested by `{}`)",
                request.request, request.model
            );
        }
    }
    println!(
        "{name}: formed {} of {} catalog kernel requests; {}",
        requests.len() - failures,
        requests.len(),
        forming.activity()
    );

    forming
        .ledger()
        .compact()
        .context("compacting the formation ledger")?;
    Ok(covered && failures == 0)
}

/// How often a running phase reports its progress.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(30);

/// Run `work`, reporting its `progress` in `unit`s and the formations done
/// every [`PROGRESS_INTERVAL`] until it returns.
fn reporting<T>(
    started: &Instant,
    phase: &str,
    unit: &str,
    progress: &Progress,
    forming: &Ledgered,
    work: impl FnOnce() -> T,
) -> T {
    std::thread::scope(|scope| {
        let (finished, waiting) = std::sync::mpsc::channel::<()>();
        scope.spawn(move || {
            while let Err(RecvTimeoutError::Timeout) = waiting.recv_timeout(PROGRESS_INTERVAL) {
                // glibc keeps memory freed by the search's short-lived worker
                // threads in their arenas; on a long run that grows into
                // gigabytes the process does not use. Return it.
                #[cfg(all(target_os = "linux", target_env = "gnu"))]
                unsafe {
                    libc::malloc_trim(0);
                }
                let (done, total) = (progress.done(), progress.total());
                let percent = if total == 0 { 0 } else { done * 100 / total };
                let elapsed = started.elapsed().as_secs();
                println!(
                    "[{:02}:{:02}] {phase}: {done} of {total} {unit} ({percent}%); {}",
                    elapsed / 60,
                    elapsed % 60,
                    forming.activity()
                );
            }
        });
        let result = work();
        drop(finished);
        result
    })
}
