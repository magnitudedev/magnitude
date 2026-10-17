//! A drafter beside a projector, on a real device. A load with built-in MTP
//! heads (which embed every prompt token) and its projector drafts text
//! requests, generates image requests without drafting instead of failing
//! them, and never lets a request resume a cached prefix retained under the
//! other method. A separate DFlash or DSpark draft reads the target's
//! features, media rows included, and drafts image requests too.
//!
//! `MAGNITUDE_TEST_GGUF=T.gguf MAGNITUDE_TEST_PROJECTOR=mmproj.gguf
//! MAGNITUDE_TEST_IMAGE=image.jpg [MAGNITUDE_TEST_DRAFT=D.gguf]
//! [MAGNITUDE_TEST_DEVICE=auto|metal|cpu|...]
//! [MAGNITUDE_TEST_KERNEL_CACHE=DIR] cargo test --release -p magnitude-engine
//! --test media_drafting`. Without the three paths the test is skipped.
//!
//! The test runs without the libtest harness: it re-executes itself with
//! `--worker` for the model's Prepare.

use magnitude_chat::request::ImageInput;
use magnitude_engine::{
    composition::{start_in_process, EngineConfiguration, ReadyEngine},
    options::{
        standard_service_limits, ModelPolicy, PackageOptions, ProjectorSelection, ResolvedMethod,
    },
    worker::{serve_worker, transport::stdio_worker_transport, RequestEvent, RequestOptions},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath,
};
use magnitude_generation::{
    DetailedUsage, EndOfGeneration, MethodChoice, Options, Sampling, Shaping, TokenId,
};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use std::{
    future::Future,
    path::PathBuf,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

const PREAMBLE: &str = "You are a careful assistant. Answer in plain sentences, name what you \
are unsure of, and keep each answer short. The user is studying how the first transatlantic \
telegraph cable was laid in the nineteenth century, what went wrong, and what was learned. ";

struct Arguments {
    target: PathBuf,
    projector: PathBuf,
    image: PathBuf,
    draft: Option<PathBuf>,
    device: DeviceRequest,
    cache_dir: Option<PathBuf>,
}

fn arguments() -> Result<Option<Arguments>, String> {
    let path = |name: &str| std::env::var_os(name).map(PathBuf::from);
    let (Some(target), Some(projector), Some(image)) = (
        path("MAGNITUDE_TEST_GGUF"),
        path("MAGNITUDE_TEST_PROJECTOR"),
        path("MAGNITUDE_TEST_IMAGE"),
    ) else {
        return Ok(None);
    };
    let device = match std::env::var("MAGNITUDE_TEST_DEVICE") {
        Ok(device) => device
            .parse()
            .map_err(|error| format!("MAGNITUDE_TEST_DEVICE: {error}"))?,
        Err(_) => DeviceRequest::Automatic,
    };
    Ok(Some(Arguments {
        target,
        projector,
        image,
        draft: path("MAGNITUDE_TEST_DRAFT"),
        device,
        cache_dir: path("MAGNITUDE_TEST_KERNEL_CACHE"),
    }))
}

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

/// One greedy request of the preamble followed by `rest`, retaining the
/// preamble's state for later requests; its output and usage.
fn generate(
    engine: &ReadyEngine,
    rest: &str,
    images: &[ImageInput],
) -> Result<(Vec<TokenId>, DetailedUsage), String> {
    let host = engine.host();
    let encode = |text: &str| {
        host.tokenizer()
            .encode(text, magnitude_engine::inputs::SpecialTokens::Recognize)
    };
    let preamble = encode(PREAMBLE)?.len();
    let input = host
        .prepare_input(encode(&format!("{PREAMBLE}{rest}"))?, images)
        .map_err(|error| error.to_string())?;
    let method = match engine.ready_info().model.method {
        ResolvedMethod::Mtp {
            greedy_proposals, ..
        } => MethodChoice::Mtp {
            proposals: greedy_proposals,
        },
        ResolvedMethod::DFlash { proposals } => MethodChoice::DFlash { proposals },
        ResolvedMethod::Plain => return Err("the load drafts with nothing".into()),
    };
    let definition = host.definition();
    let options = Options {
        max_tokens: 32,
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
                retention: PrefixRetention::Retain {
                    cache_points: vec![preamble],
                },
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
                Some(RequestEvent::Completed { usage, .. }) => return Ok((output, usage)),
                Some(RequestEvent::Failed(error)) => return Err(error.to_string()),
                None => return Err("the request ended without a terminal event".into()),
            }
        }
    })
}

fn run(arguments: &Arguments) -> Result<(), String> {
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: arguments.target.clone(),
            projector: ProjectorSelection::Explicit(arguments.projector.clone()),
            draft: arguments.draft.clone(),
        },
        model: ModelPolicy::default(),
        context_tokens: Some(8192),
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device: arguments.device,
        kernel_cache: arguments.cache_dir.clone(),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .map_err(|error| error.to_string())?;
    if resolved.host.media_placeholder().is_none() {
        return Err("the load has no vision".into());
    }
    let mut worker = std::process::Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    worker.arg("--worker");
    let engine = start_in_process(resolved, worker, |progress| eprintln!("{progress:?}"))
        .map_err(|error| error.to_string())?;
    let placeholder = engine.host().media_placeholder().expect("vision").to_owned();
    let image = ImageInput {
        media_type: "image/jpeg".into(),
        bytes: std::fs::read(&arguments.image)
            .map_err(|error| error.to_string())?
            .into(),
    };
    let images = std::slice::from_ref(&image);
    let describe = format!("Here is an image:{placeholder}Describe it in one sentence.");
    let ask = "In one sentence, what went wrong with the 1858 cable?";

    if arguments.draft.is_some() {
        // A separate draft reads the target's features of the media rows.
        let (seen, usage) = generate(&engine, &describe, images)?;
        println!("image: {usage:?}");
        check(!seen.is_empty() && usage.draft_n > 0, "the image request drafts")?;
        let (_, usage) = generate(&engine, ask, &[])?;
        println!("text: {usage:?}");
        check(usage.draft_n > 0, "the text request drafts")?;
        let (seen_again, usage) = generate(&engine, &describe, images)?;
        println!("image again: {usage:?}");
        check(seen_again == seen, "the resumed image request repeats its greedy output")?;
        engine.client().shutdown().map_err(|error| error.to_string())?;
        return Ok(());
    }
    // The image request generates without drafting and retains the
    // preamble without drafter history.
    let (seen, usage) = generate(&engine, &describe, images)?;
    println!("image: {usage:?}");
    check(!seen.is_empty() && usage.draft_n == 0, "the image request drafts nothing")?;
    // A text request drafts, and does not resume the image request's
    // preamble state, which lacks its heads' history.
    let (_, usage) = generate(&engine, ask, &[])?;
    println!("text: {usage:?}");
    check(usage.draft_n > 0, "the text request drafts")?;
    check(usage.cached_tokens == 0, "the text request resumes no plain state")?;
    // Each method resumes only its own retained preamble.
    let (again, usage) = generate(&engine, ask, &[])?;
    println!("text again: {usage:?}");
    check(usage.draft_n > 0 && usage.cached_tokens > 0, "a later text request resumes the text state")?;
    let (seen_again, usage) = generate(&engine, &describe, images)?;
    println!("image again: {usage:?}");
    check(usage.draft_n == 0 && usage.cached_tokens > 0, "a later image request resumes the plain state")?;
    check(seen_again == seen, "the resumed image request repeats its greedy output")?;
    let (text, _) = generate(&engine, ask, &[])?;
    check(text == again, "text output is unchanged after image turns")?;
    engine.client().shutdown().map_err(|error| error.to_string())?;
    Ok(())
}

fn check(holds: bool, claim: &str) -> Result<(), String> {
    if holds {
        Ok(())
    } else {
        Err(format!("expected: {claim}"))
    }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--worker") {
        let exit = serve_worker(stdio_worker_transport());
        std::process::exit(match exit {
            magnitude_engine::worker::WorkerExit::Shutdown => 0,
            _ => 1,
        });
    }
    match arguments() {
        Ok(Some(arguments)) => {
            if let Err(error) = run(&arguments) {
                eprintln!("media_drafting: {error}");
                std::process::exit(1);
            }
            println!("media_drafting: ok");
        }
        Ok(None) => println!(
            "media_drafting: skipped (set MAGNITUDE_TEST_GGUF, MAGNITUDE_TEST_PROJECTOR and \
             MAGNITUDE_TEST_IMAGE)"
        ),
        Err(error) => {
            eprintln!("media_drafting: {error}");
            std::process::exit(1);
        }
    }
}
