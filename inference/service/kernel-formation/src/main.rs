//! Forms every native kernel the engine ships with one GPU backend's real
//! toolchain (every kernel, every preprocessor group of its sources, every
//! form value), then every kernel every supported catalog model requests.
//!
//!   magnitude-kernel-formation <vulkan|cuda|metal> --ledger <path> --catalog <bundle>
//!
//! Formation runs in worker processes admitted by measured memory, through a
//! ledger of what earlier runs established ([`pool`], [`ledger`]). Coverage
//! whose sources and toolchain match a complete coverage run reports that run
//! and searches nothing; catalog requests are always enumerated, and those
//! the ledger holds are recalled. Any failure, unreached group, unformed form
//! value, configuration this host cannot form, or catalog request that does
//! not form fails the run.

mod ledger;
mod pool;
mod worker;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use ledger::{Key, Ledger};
use magnitude_service_server::assessment::catalog_kernels::catalog_kernel_requests;
use pool::{Pool, WorkerCommand};
use seismic::BackendName;
use seismic::coverage::{Progress, Toolchain};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Arguments {
    /// Serve one formation request on standard input: how the pool starts its workers.
    #[arg(long, exclusive = true)]
    worker: bool,
    #[arg(value_enum, required_unless_present = "worker")]
    backend: Option<Backend>,
    /// The ledger of what earlier runs established.
    #[arg(long, required_unless_present = "worker")]
    ledger: Option<PathBuf>,
    /// The release header bundle the catalog is read from.
    #[arg(long, required_unless_present = "worker")]
    catalog: Option<PathBuf>,
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
    let result = if arguments.worker {
        worker::serve().map(|()| true)
    } else {
        form(
            arguments.backend.expect("clap requires a backend").into(),
            &arguments.ledger.expect("clap requires a ledger"),
            &arguments.catalog.expect("clap requires a catalog"),
        )
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Form for `backend`; whether everything formed.
fn form(
    backend: BackendName,
    ledger: &std::path::Path,
    catalog: &std::path::Path,
) -> anyhow::Result<bool> {
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
    let pool = Pool::new(
        ledger,
        WorkerCommand {
            program: std::env::current_exe().context("locating this executable")?,
            arguments: vec!["--worker".into()],
        },
    );
    let module = magnitude_kernels::module().context("decoding the kernel bundle")?;
    let started = Instant::now();
    let progress = Progress::default();

    let covered = match pool.ledger().completion(&coverage_run) {
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
                &pool,
                || seismic::coverage::cover(module, &toolchain, &pool, &progress),
            )
            .map_err(|error| anyhow::anyhow!("{name} coverage did not run: {error}"))?;
            println!("{name}: {coverage}");
            let covered = coverage.complete() && coverage.unformable.is_empty();
            if covered {
                pool.ledger()
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
        &pool,
        || seismic::coverage::form_requests(module, &toolchain, &pool, &requests, &progress),
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
    let requested = format!(
        "formed {} of {} catalog kernel requests",
        requests.len() - failures,
        requests.len()
    );
    println!("{name}: {requested}");
    println!("{name}: {}", pool.report());

    pool.ledger()
        .compact()
        .context("compacting the formation ledger")?;
    Ok(covered && failures == 0)
}

/// How often a running phase reports its progress.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(30);

/// Run `work`, reporting its `progress` in `unit`s and the pool's activity
/// every [`PROGRESS_INTERVAL`] until it returns.
fn reporting<T>(
    started: &Instant,
    phase: &str,
    unit: &str,
    progress: &Progress,
    pool: &Pool,
    work: impl FnOnce() -> T,
) -> T {
    std::thread::scope(|scope| {
        let (finished, waiting) = std::sync::mpsc::channel::<()>();
        scope.spawn(move || {
            while let Err(RecvTimeoutError::Timeout) = waiting.recv_timeout(PROGRESS_INTERVAL) {
                let (done, total) = (progress.done(), progress.total());
                let percent = if total == 0 { 0 } else { done * 100 / total };
                let elapsed = started.elapsed().as_secs();
                println!(
                    "[{:02}:{:02}] {phase}: {done} of {total} {unit} ({percent}%); {}",
                    elapsed / 60,
                    elapsed % 60,
                    pool.activity()
                );
            }
        });
        let result = work();
        drop(finished);
        result
    })
}
