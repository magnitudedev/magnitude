//! Gate G-B: the same greedy request served through `run_worker` in-process
//! (channel transport) and in a worker process (length-prefixed frames over
//! the child's stdio) must produce byte-identical output.
//!
//! `MAGNITUDE_TEST_GGUF=M.gguf [MAGNITUDE_TEST_KERNEL_CACHE=DIR]
//! [MAGNITUDE_TEST_TOKENS=N] [MAGNITUDE_TEST_DEVICE=auto|metal|cpu|...]
//! cargo test --release -p magnitude-engine --test worker_parity`. Without
//! `MAGNITUDE_TEST_GGUF` the test is skipped.
//!
//! The test runs without the libtest harness: it re-executes itself with
//! `--worker` for the process transport, whose stdout carries the frames.

use magnitude_engine::{
    composition::{start_in_process, EngineConfiguration, ReadyEngine},
    options::{
        standard_service_limits, ModelPolicy, PackageOptions, ProjectorSelection, ResolvedMethod,
    },
    worker::{
        connect_worker,
        protocol::{HostMessage, WorkerMessage},
        serve_worker,
        transport::{stdio_worker_transport, FramedTransport},
        RequestEvent, RequestOptions,
    },
    LoadAttempt,
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath, UnpreparedUnits,
};
use magnitude_generation::{
    EndOfGeneration, FinishReason, MethodChoice, Options, Sampling, Shaping, TokenId,
};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use std::{
    future::Future,
    path::PathBuf,
    pin::pin,
    process::{Command, Stdio},
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};

const PROMPT: &str = "The following is a short, factual account of how the first transatlantic \
telegraph cable was laid in the nineteenth century, what went wrong, and what was learned:";

struct Arguments {
    model: PathBuf,
    cache_dir: Option<PathBuf>,
    tokens: usize,
    device: DeviceRequest,
}

/// The test's model and options; `None` when no model is named.
fn arguments() -> Result<Option<Arguments>, String> {
    let Some(model) = std::env::var_os("MAGNITUDE_TEST_GGUF") else {
        return Ok(None);
    };
    let tokens = match std::env::var("MAGNITUDE_TEST_TOKENS") {
        Ok(tokens) => tokens
            .parse()
            .map_err(|error| format!("MAGNITUDE_TEST_TOKENS: {error}"))?,
        Err(_) => 64,
    };
    let device = match std::env::var("MAGNITUDE_TEST_DEVICE") {
        Ok(device) => device
            .parse()
            .map_err(|error| format!("MAGNITUDE_TEST_DEVICE: {error}"))?,
        Err(_) => DeviceRequest::Automatic,
    };
    Ok(Some(Arguments {
        model: PathBuf::from(model),
        cache_dir: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        tokens,
        device,
    }))
}

fn configuration(arguments: &Arguments) -> EngineConfiguration {
    EngineConfiguration {
        package: PackageOptions {
            target: arguments.model.clone(),
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy::default(),
        context_tokens: Some(4096),
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device: arguments.device,
        kernel_cache: arguments.cache_dir.clone(),
        reserves: MemoryReserves::standard(),
    }
}

/// Drive a future on this thread; the engine client is executor-agnostic.
fn block_on<T>(future: impl Future<Output = T>) -> T {
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// One greedy request of `tokens` output tokens on `engine`.
fn generate(engine: &ReadyEngine, tokens: usize) -> Result<(Vec<TokenId>, FinishReason), String> {
    let host = engine.host();
    let prompt = host
        .tokenizer()
        .encode(PROMPT, magnitude_engine::inputs::SpecialTokens::Recognize)?;
    let input = host
        .prepare_input(prompt, &[])
        .map_err(|error| error.to_string())?;
    let method = match engine.ready_info().model.method {
        ResolvedMethod::Plain => MethodChoice::Plain,
        ResolvedMethod::Mtp {
            greedy_proposals, ..
        } => MethodChoice::Mtp {
            proposals: greedy_proposals,
        },
        ResolvedMethod::DFlash { proposals } => MethodChoice::DFlash { proposals },
    };
    let definition = host.definition();
    let options = Options {
        max_tokens: tokens,
        output_capacity: 16,
        context_limit: usize::try_from(definition.decoder.context_limit)
            .map_err(|_| "context exceeds the host domain")?,
        vocabulary: usize::try_from(definition.decoder.vocabulary)
            .map_err(|_| "vocabulary exceeds the host domain")?,
        stop_tokens: host.tokenizer().stop_tokens().clone(),
        suppressed_tokens: host.tokenizer().suppressed_tokens().clone(),
        sampling: Sampling::Greedy,
        shaping: Shaping {
            temperature: 0.0,
            ..Shaping::default()
        },
        seed: 0,
        forced_quantum: 0,
        method,
        end_of_generation: EndOfGeneration::Suppress,
        reasoning_budget: None,
    };
    block_on(async {
        let mut request = engine
            .client()
            .admit(RequestOptions {
                input,
                options,
                constraint: None,
                retention: PrefixRetention::Transient,
                output_capacity: 16,
            })
            .await
            .map_err(|error| error.to_string())?;
        let mut output = Vec::new();
        loop {
            match request.receive().await {
                Some(RequestEvent::Output(batch)) => {
                    output.extend(batch.into_iter().map(|token| token.token))
                }
                Some(RequestEvent::Completed { finish, usage, .. }) => {
                    if usage.completion_tokens != output.len() {
                        return Err(format!(
                            "usage reports {} completion tokens; {} were received",
                            usage.completion_tokens,
                            output.len()
                        ));
                    }
                    return Ok((output, finish));
                }
                Some(RequestEvent::Failed(error)) => return Err(error.to_string()),
                None => return Err("the request ended without a terminal event".into()),
            }
        }
    })
}

fn in_process(arguments: &Arguments) -> Result<(Vec<TokenId>, FinishReason), String> {
    let resolved = configuration(arguments)
        .resolve()
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    // This executable's `--worker` role runs the model's Prepare when the
    // kernel cache lacks records; the process side then reads them too.
    let mut worker =
        std::process::Command::new(std::env::current_exe().map_err(|error| error.to_string())?);
    worker.arg("--worker");
    let engine = start_in_process(resolved, worker, |progress| {
        eprintln!("in-process: {progress:?}")
    })
    .map_err(|error| error.to_string())?;
    eprintln!(
        "in-process: ready in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    let result = generate(&engine, arguments.tokens);
    // Dropping the last client closes the transport; the worker exits on
    // host loss and releases the device.
    drop(engine);
    result
}

fn worker_process(arguments: &Arguments) -> Result<(Vec<TokenId>, FinishReason), String> {
    let resolved = configuration(arguments)
        .resolve()
        .map_err(|error| error.to_string())?;
    let catalog = seismic::DeviceCatalog::discover().map_err(|error| error.to_string())?;
    let preview = resolved
        .manifest
        .preview(&catalog)
        .map_err(|error| error.to_string())?;
    eprintln!("worker process: preview {preview:?}");
    let started = Instant::now();
    let mut child = Command::new(std::env::current_exe().map_err(|error| error.to_string())?)
        .arg("--worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| error.to_string())?;
    let transport = FramedTransport::<_, _, WorkerMessage, HostMessage>::new(
        child.stdout.take().ok_or("worker stdout")?,
        child.stdin.take().ok_or("worker stdin")?,
    );
    // The gate compares serving, not preparation: both workers serve what
    // the kernel cache holds, and canonical defaults for the rest.
    let LoadAttempt::Loaded(connection) = connect_worker(
        transport,
        Some((resolved.manifest, UnpreparedUnits::ServeDefaults)),
        |progress| eprintln!("worker process: {progress:?}"),
    )
    .map_err(|error| error.to_string())?
    else {
        return Err("a load serving canonical defaults stopped unprepared".into());
    };
    let engine =
        ReadyEngine::new(Arc::new(resolved.host), connection).map_err(|error| error.to_string())?;
    eprintln!(
        "worker process: ready in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    let result = generate(&engine, arguments.tokens);
    let standing = block_on(engine.client().observe()).map_err(|error| error.to_string())?;
    eprintln!(
        "worker process: census at ready {:?}; observed after the request {:?}",
        engine.ready_info().census,
        standing
    );
    engine
        .client()
        .shutdown()
        .map_err(|error| error.to_string())?;
    let status = child.wait().map_err(|error| error.to_string())?;
    if !status.success() {
        return Err(format!("worker process exited with {status}"));
    }
    result
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--worker") {
        let exit = serve_worker(stdio_worker_transport());
        eprintln!("worker process: exited: {exit:?}");
        std::process::exit(match exit {
            magnitude_engine::worker::WorkerExit::Shutdown => 0,
            _ => 1,
        });
    }
    let arguments = match arguments() {
        Ok(Some(arguments)) => arguments,
        Ok(None) => {
            println!("worker_parity: skipped (set MAGNITUDE_TEST_GGUF)");
            return;
        }
        Err(error) => {
            eprintln!("worker_parity: {error}");
            std::process::exit(1);
        }
    };
    let result = in_process(&arguments).and_then(|local| Ok((local, worker_process(&arguments)?)));
    match result {
        Ok((local, remote)) => {
            println!("in-process     ({:?}): {:?}", local.1, local.0);
            println!("worker process ({:?}): {:?}", remote.1, remote.0);
            if local == remote {
                println!("identical: {} tokens", local.0.len());
            } else {
                println!("DIFFERENT");
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("worker_parity: {error}");
            std::process::exit(1);
        }
    }
}
