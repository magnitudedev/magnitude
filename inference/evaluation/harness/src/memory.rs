//! `memory`: one load's host memory against its charge, the load `mrig
//! memory --backend metal|cpu` measures (`magnitude_rig.host.memory`), from
//! which the worker's own host costs (`HostProcessCosts::of`) are fitted.
//!
//! Loads a target plain, without its drafter or projector, as the service
//! would: standard service limits and reserves, the artifact's declared
//! context or `--context`, its weights under `--weight-cap` when given (its
//! routed experts then streamed), in a worker process of this executable.
//! The worker prints its memory moments on standard error (the engine's
//! `load memory` events), the last of them at load end (`loaded`),
//! with what the load holds of each storage term. This process prints one
//! JSON line on standard output: the charge of the manifest the worker
//! loads, planned on the same device, split into what the plan holds at
//! load end (its storage by term) and the worker's own costs, and where a
//! streamed load's experts are stored.
//!
//! With `--decode`, the load decodes a rendered fixture greedily and prints
//! its tokens; with `--pressure BYTES` too, a sibling process (`occupy`)
//! presses the load BYTES below its planning line from a quarter of the
//! decode to three quarters of it, and the worker prints each event of the
//! owner's memory ([`print_events`]): each release (its rung, time,
//! the expert bank's slots, the charge and footprint on each side) and each
//! change of its memory condition.
//!
//! `recommit` loads nothing: it times what an in-place recommit of a
//! reserved tensor (`Tensor::recommit`, an elastic expert bank's resize)
//! costs on the device at each commitment, printing one JSON line per
//! commitment.

mod store;
pub use store::PressureStore;

use magnitude_engine::{
    LoadAttempt,
    assessment::AssessmentStructure,
    composition::{EngineConfiguration, ReadyEngine},
    error::{RequestError, UnloadCause},
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection, standard_service_limits},
    planning::plan_execution,
    worker::{RequestEvent, RequestOptions, WorkerProcess, WorkerReport, connect_worker, start_worker_process},
};
use magnitude_executor::{
    ExecutionPath, UnpreparedUnits,
    assessment::prepare_execution_assessment,
    platform::{
        DeviceRequest, MemoryReserves, OpenedPlatform, PlatformConfig, observe_domains, open_selected,
        select_device,
    },
};
use magnitude_generation::{EndOfGeneration, MethodChoice, Options, Sampling, Shaping, TokenId};
use magnitude_scheduler::prefix_cache::PrefixRetention;
use seismic::{BackendName, Devices, Element, ReservedContract, Tensor};
use serde_json::{Value, json};
use std::future::Future;
use std::path::PathBuf;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Instant;

/// A unified-memory backend, by name.
pub fn backend(name: &str) -> Result<BackendName, String> {
    BackendName::parse(name).ok_or_else(|| format!("no backend {name}"))
}

/// What one load is.
pub struct Load {
    pub backend: BackendName,
    pub target: PathBuf,
    /// The kernel cache the load prepares from, as a served load's.
    pub kernel_cache: Option<PathBuf>,
    pub context_tokens: Option<usize>,
    pub weight_cap: Option<u64>,
    /// The rendered fixture decoded once loaded, and its tokens.
    pub decode: Option<Decode>,
    /// Another process's pressure on the load over the decode.
    pub pressure: Option<Pressing>,
    /// Keep the loaded worker until standard input closes, for a heap
    /// reader to attach.
    pub hold: bool,
}

/// Another process pressing the load `deficit` bytes below its planning
/// line over a window of the decode ([`Window`]): signalled once a quarter
/// of the decode's tokens are out, or, `from_prompt`, once the request is
/// admitted, so its prompt's groups run under it.
pub struct Pressing {
    pub store: PressureStore,
    pub deficit: u64,
    pub from_prompt: bool,
}

/// A decode the load runs before its experts' residency is read again: the
/// first fixture of a rendered set (`magnitude-harness render`), its first
/// `prompt_tokens` when given, `tokens` tokens greedily.
pub struct Decode {
    pub rendered: PathBuf,
    pub tokens: usize,
    pub prompt_tokens: Option<usize>,
}

/// Load as `load` says and print its memory against its charge.
pub fn load(load: Load) -> Result<(), String> {
    let Load {
        backend,
        target,
        kernel_cache,
        context_tokens,
        weight_cap,
        decode,
        pressure,
        hold,
    } = load;
    let mut pressure = pressure.map(|pressing| Pressure::new(backend, pressing)).transpose()?;
    let Planned { host, manifest, devices, structure, draft } = plan(backend, &target, kernel_cache.clone(), context_tokens, weight_cap)?;
    let definition = &manifest.definition;
    let context = u32::try_from(definition.decoder.context_limit)
        .map_err(|_| "the context exceeds the assessment domain")?;
    let execution =
        prepare_execution_assessment(definition, &draft, context).map_err(|error| error.to_string())?;
    // Where a streamed load's experts are stored, none of whose pages the
    // file cache may hold.
    let expert_ranges = crate::engine::expert_ranges(draft.load().expert_bank(), host.package().target())?;
    // The streamed load's bank: its slots at load, its floor and one slot.
    let bank = draft
        .load()
        .expert_bank()
        .map(|bank| {
            // Physical charge can plateau above the logical floor (request
            // records and allocation granules). Reuse the plan's monotone
            // search and charge owners to name where no further backing frees.
            let metadata = |holding: magnitude_executor::BankHolding, floor, initial, ceiling,
                            charge: &dyn Fn(u64) -> Result<u64, String>| {
                let minimum = charge(floor)?;
                let physical_floor = magnitude_executor::growth_target(floor, ceiling, minimum, charge)?.unwrap_or(floor);
                let kind = bank.holding_storage(holding).ok_or_else(|| format!("{holding:?} of a bank without it"))?.storage();
                Ok::<_, String>((format!("{holding:?}"), json!({
                    "floor": floor, "physical_floor": physical_floor, "initial": initial,
                    "storage_kind": format!("{kind:?}"),
                    "stores": store::holding_stores(&devices, &structure.selected.info, kind),
                })))
            };
            let mut holdings = serde_json::Map::new();
            let (name, value) = metadata(magnitude_executor::BankHolding::Experts,
                bank.floor(), bank.slots, bank.ceiling, &|at| bank.resident_charge(at))?;
            holdings.insert(name, value);
            if let Some(ceiling) = bank.host_ceiling() {
                let (name, value) = metadata(magnitude_executor::BankHolding::Host,
                    0, 0, ceiling.get(), &|at| bank.host_charge(at))?;
                holdings.insert(name, value);
            }
            for (table, plan) in bank.tables.iter().enumerate() {
                let (name, value) = metadata(magnitude_executor::BankHolding::Table { table },
                    plan.floor, plan.floor, plan.records.get(), &|at| plan.charge(bank.reservation, at))?;
                holdings.insert(name, value);
            }
            Ok::<_, String>(json!({
                "slots": bank.slots,
                "floor": bank.floor(),
                "slot_bytes": bank.slot_bytes()?,
                "holdings": holdings,
            }))
        })
        .transpose()?;
    let charge = execution.memory_charge();
    let bounds = &charge.bounds;
    let process = &bounds.host_process;
    // What the plan holds at load end: in device and upload storage the
    // weights (the expert bank's, under a cap), the loaded state and the
    // prepared resources; in host RAM the exact host terms. Each storage
    // term is named as the worker's `loaded` record names the holders'
    // charge in it, so the two read side by side.
    let sum = |terms: &[u64], what: &str| {
        terms
            .iter()
            .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
            .ok_or_else(|| format!("the plan's {what} overflow"))
    };
    let weights = charge
        .exact_resident_bytes
        .checked_sub(charge.state_bytes)
        .ok_or("the plan's state exceeds its exact resident bytes")?;
    let beyond = sum(&[bounds.recorded_graph_bytes, bounds.device_process.bytes()?], "bytes beyond allocations")?;
    let storage_terms = json!({
        "weights": weights,
        "state": charge.loaded_state_bytes,
        "graphs": bounds.graph_bytes,
        "programs": magnitude_executor::AttestedPrograms::DEVICE_BYTES,
        "beyond_allocations": beyond,
    });
    let storage = sum(&[weights, charge.loaded_state_bytes, bounds.prepared_resource_bytes()?], "storage bytes")?;
    let host_terms = sum(
        &[
            charge.host_table_bytes,
            bounds.sealed_graph_host_bytes.at_most,
            bounds.bound_graph_host_bytes,
            bounds.drafter_graph_host_bytes,
        ],
        "host bytes",
    )?;
    let vocabulary = definition.decoder.vocabulary;
    // What a shrink's release costs device work beside it, as the executor's
    // costs measured it in the kernel cache: the ratio a release may slow a
    // decode step it overlaps by. None unread.
    let release_step_ratio = match kernel_cache.as_deref().map(|cache| structure.stored_measurements(cache)) {
        Some(magnitude_engine::assessment::Measurements::Measured { costs, .. }) => {
            costs.release_step_ratio.map(|ratio| ratio.value)
        }
        _ => None,
    };

    // The load, in a worker process as the service loads.
    let WorkerProcess { transport, retire } =
        start_worker_process(crate::engine::worker(&[])?).map_err(|error| error.to_string())?;
    let loaded = connect_worker(transport, Some((manifest, UnpreparedUnits::ServeDefaults)), |_| {})
        .map_err(|error| error.to_string())?;
    let LoadAttempt::Loaded(connection) = loaded else {
        return Err("a load serving canonical defaults stopped unprepared".into());
    };
    let engine = ReadyEngine::new(host, connection).map_err(|error| error.to_string())?;
    // The experts' pages the file cache holds once loaded, then once the
    // load has decoded a rendered fixture.
    let (residency, decoded) = match decode {
        None => (None, None),
        Some(decode) => {
            let loaded = crate::engine::cached_pages(&expert_ranges)?;
            let tokens = match (generate(&engine, &decode, pressure.as_mut()), pressure.as_mut()) {
                (Ok(tokens), _) => Some(tokens),
                (Err(Failure::Unloaded(failure)), Some(pressure)) => {
                    pressure.failed(failure)?;
                    None
                }
                (Err(Failure::Unloaded(failure)), None) => return Err(failure),
                (Err(Failure::Cell(failure)), pressure) => {
                    return Err(match pressure.map(Pressure::free).transpose() {
                        Ok(_) => failure,
                        Err(freeing) => format!("{failure}; giving the memory back: {freeing}"),
                    });
                }
            };
            let decoded = crate::engine::cached_pages(&expert_ranges)?;
            let residency = json!({
                "pages": loaded.1,
                "loaded": loaded.0,
                "decoded": decoded.0,
                "decode_tokens": tokens.as_ref().map(|(_, tokens)| tokens.len()),
            });
            (Some(residency), tokens)
        }
    };
    println!(
        "{}",
        json!({
            "model": target.display().to_string(),
            "backend": backend.as_str(),
            "weight_cap": weight_cap,
            "context": context,
            "plan_storage_bytes": storage,
            "plan_storage_terms": storage_terms,
            "expert_residency": residency,
            "bank": bank,
            "decode_prompt_tokens": decoded.as_ref().map(|(prompt, _)| prompt),
            "decoded_tokens": decoded.as_ref().map(|(_, tokens)| tokens),
            "pressure": pressure.map(Pressure::into_record),
            "plan_host_bytes": host_terms,
            "weight_bytes": weights,
            "vocabulary": vocabulary,
            "release_step_ratio": release_step_ratio,
            "kernels": process.kernels,
            "charged_runtime_bytes": process.runtime,
            "charged_vocabulary_bytes": process.vocabulary,
            "charged_program_bytes": process.programs,
        })
    );
    // A heap reader attaches to the loaded worker while standard input is
    // open.
    if hold {
        std::io::copy(&mut std::io::stdin(), &mut std::io::sink()).map_err(|error| error.to_string())?;
    }
    // The worker ends once its host is gone.
    drop(engine);
    retire(WorkerReport::Reported)
}

/// A target planned as the service plans it on `backend`'s device: the
/// package the worker loads and its manifest, the device catalog and the
/// assessment structure the plan is drawn on, and the plan.
struct Planned {
    host: Arc<magnitude_engine::host::HostArtifacts>,
    manifest: magnitude_engine::options::ExecutionManifest,
    devices: Devices,
    structure: AssessmentStructure,
    draft: magnitude_executor::ExecutionPlanDraft,
}

/// Plan `target` on `backend`'s device ([`Planned`]): plain, without its
/// drafter or projector, at `context_tokens` or its declared context, its
/// weights under `weight_cap` when given, its kernels from `kernel_cache`.
fn plan(
    backend: BackendName,
    target: &std::path::Path,
    kernel_cache: Option<PathBuf>,
    context_tokens: Option<usize>,
    weight_cap: Option<u64>,
) -> Result<Planned, String> {
    let device = DeviceRequest::Backend(backend);
    let policy = ModelPolicy {
        method: ModelMethod::Plain,
        weight_cap,
        ..ModelPolicy::default()
    };
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: target.to_path_buf(),
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: policy.clone(),
        context_tokens,
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device,
        kernel_cache,
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .map_err(|error| error.to_string())?;
    let (host, manifest) = (Arc::new(resolved.host), resolved.manifest);
    // The charge: the manifest the worker loads, planned on the same device.
    let devices = Devices::discover(&[]).map_err(|error| error.to_string())?;
    let structure = AssessmentStructure::discover(
        &devices,
        device,
        MemoryReserves::standard(),
        policy,
        standard_service_limits(),
    )
    .map_err(|error| error.to_string())?;
    let draft = plan_execution(&manifest, &structure.selected).map_err(|error| error.to_string())?;
    Ok(Planned { host, manifest, devices, structure, draft })
}

/// What an in-place recommit costs on `backend`'s device at each of
/// `commitments` bytes, as an elastic bank's resize pays it between flights:
/// a reserved tensor of rows of the device's reservation granule
/// (`Limits::reservation`, a whole number of host pages) committed to the
/// rows that hold the commitment, with one row of room, shrinks one row and
/// regrows it `RESIZES` times, enough to show each way's spread beside its
/// median. Nothing uses the tensor, so each recommit times the backend's
/// resize alone. One JSON line per commitment: the reservation's time and
/// every resize's.
pub fn recommit(backend: BackendName, commitments: &[u64]) -> Result<(), String> {
    const RESIZES: usize = 5;
    let text = |error: &dyn std::fmt::Display| error.to_string();
    let platform = open_platform(backend)?;
    let row = platform
        .device()
        .info()
        .reserving(ReservedContract::DeviceOnly)
        .map_err(|why| format!("{} cannot reserve device storage: {why}", backend.as_str()))?
        .ok_or_else(|| format!("{} reserves no device storage", backend.as_str()))?
        .granule;
    let milliseconds = |since: Instant| since.elapsed().as_secs_f64() * 1e3;
    for &bytes in commitments {
        let rows = bytes.div_ceil(row);
        let shrunk = rows
            .checked_sub(1)
            .ok_or_else(|| format!("a commitment of {bytes} bytes has no row to shrink"))?;
        let committed_bytes = rows.checked_mul(row).ok_or_else(|| format!("a commitment of {bytes} bytes overflows"))?;
        let reserving = Instant::now();
        let (_tensor, mut resizer) =
            Tensor::reserved(platform.device(), Element::bool(), &[rows + 1, row], rows)
                .map_err(|error| text(&error))?;
        let reserve = milliseconds(reserving);
        let (mut shrink, mut grow) = (Vec::new(), Vec::new());
        for _ in 0..RESIZES {
            for (committed, times) in [(shrunk, &mut shrink), (rows, &mut grow)] {
                let resizing = Instant::now();
                resizer = resizer
                    .recommit(committed)
                    .map_err(|refusal| text(&refusal.error))?;
                times.push(milliseconds(resizing));
            }
        }
        println!(
            "{}",
            json!({
                "backend": backend.as_str(),
                "committed_bytes": committed_bytes,
                "row_bytes": row,
                "reserve_ms": reserve,
                "shrink_ms": shrink,
                "grow_ms": grow,
            })
        );
    }
    Ok(())
}

/// `backend`'s device, opened as the service opens it.
fn open_platform(backend: BackendName) -> Result<OpenedPlatform, String> {
    let text = |error: &dyn std::fmt::Display| error.to_string();
    let reserves = MemoryReserves::standard();
    let catalog = Devices::discover(&[]).map_err(|error| text(&error))?;
    let selected = select_device(&catalog, ExecutionPath::Native, DeviceRequest::Backend(backend), &reserves)
        .map_err(|error| text(&error))?;
    open_selected(
        &catalog,
        selected.info.selector(),
        PlatformConfig {
            graph_cost: None,
            path: ExecutionPath::Native,
            artifacts: None,
            reserves,
        },
    )
    .map_err(|error| text(&error))
}

/// Another process's memory, pressing the loaded process `deficit` bytes
/// below its planning line in the selected physical store, read through
/// the engine's observation owner, and `deficit` more. It takes them in two moves, so the time
/// the pressure takes to land is the deficit's, whatever the host's size:
/// first it comes to `deficit` above the line, where the engine still has
/// room, and prints `approached`; at the first line on standard input it
/// takes the rest, `deficit` below the line, and prints `occupied`; it holds
/// everything until standard input closes. The bytes are pseudo-random
/// words, as an application's data is: zeroed pages the OS compresses to
/// nothing would press nothing. They are written on every core the host
/// has, as fast as its memory takes them.
pub fn occupy(backend: BackendName, selected: PressureStore, deficit: u64) -> Result<(), String> {
    let target = store::Target::open(backend, selected)?;
    let before = target.snapshot()?;
    let ceiling = target.reading()?.ceiling_bytes;
    let approach = ceiling.saturating_sub(deficit);
    let approached = target.take(approach, 0)?;
    println!("{}", json!({ "approached": approach, "ceiling": ceiling, "at": unix_seconds(),
        "before": before, "observation": target.snapshot()? }));
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).map_err(|error| error.to_string())? == 0 { return Ok(()); }
    let bytes = ceiling.checked_add(deficit).ok_or("pressure allocation size overflows")?;
    let pressed = target.take(bytes - approach, 1)?;
    println!("{}", json!({ "occupied": bytes, "ceiling": ceiling, "at": unix_seconds(), "observation": target.snapshot()? }));
    std::io::copy(&mut std::io::stdin(), &mut std::io::sink()).map_err(|error| error.to_string())?;
    drop((approached, pressed));
    println!("{}", json!({ "released": true, "at": unix_seconds(), "observation": target.snapshot()? }));
    Ok(())
}

/// Time the same selected physical allocation owner before loading the model.
pub fn time_occupying(backend: BackendName, selected: PressureStore) -> Result<(), String> {
    let target = store::Target::open(backend, selected)?;
    let bytes = target.reading()?.ceiling_bytes;
    let started = Instant::now();
    let taken = target.take(bytes, 0)?;
    let seconds = started.elapsed().as_secs_f64();
    drop(taken);
    println!("{}", json!({ "bytes": bytes, "seconds": seconds, "observation": target.snapshot()? }));
    Ok(())
}

/// `bytes` of pseudo-random words, written by `writers` threads at once,
/// the first of them seeded as writer `first` of the process.
fn written(bytes: u64, writers: usize, first: usize) -> Result<Vec<u64>, String> {
    let mut held = vec![0u64; usize::try_from(bytes.div_ceil(8)).map_err(|error| error.to_string())?];
    // One writer's share of the words, at least one word for a chunk size.
    let share = held.len().div_ceil(writers).max(1);
    std::thread::scope(|scope| {
        for (writer, words) in held.chunks_mut(share).enumerate() {
            scope.spawn(move || {
                // xorshift64 from a seed of its own (an odd multiple, never
                // zero): every word differs, so no page compresses.
                let mut word = 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(2 * (first + writer) as u64 + 1);
                for slot in words {
                    word ^= word << 13;
                    word ^= word >> 7;
                    word ^= word << 17;
                    *slot = word;
                }
            });
        }
    });
    Ok(held)
}

/// Seconds since the Unix epoch, as the worker's diagnostic lines stamp
/// their moments.
fn unix_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is past the Unix epoch")
        .as_secs_f64()
}

/// Why a decode ended without its tokens.
#[derive(Debug, PartialEq, Eq)]
enum Failure {
    /// The engine unloaded the model under memory pressure
    /// (`ModelUnloaded { cause: MemoryPressure }`): under pressure, the
    /// ladder's last step, which the report reads beside the releases.
    Unloaded(String),
    /// Anything else, a pressure that missed its window among it: nothing
    /// was tested.
    Cell(String),
}

impl From<String> for Failure {
    fn from(error: String) -> Self {
        Self::Cell(error)
    }
}

impl From<RequestError> for Failure {
    fn from(error: RequestError) -> Self {
        match error {
            RequestError::ModelUnloaded {
                cause: UnloadCause::MemoryPressure,
            } => Self::Unloaded(error.to_string()),
            error => Self::Cell(error.to_string()),
        }
    }
}

/// Decode `decode`'s fixture greedily: `decode.tokens` tokens from its first
/// `decode.prompt_tokens`, or, under `pressure`, as long as its window needs
/// ([`Pressure::decode`]). The prompt's tokens decoded from, and the tokens.
fn generate(engine: &ReadyEngine, decode: &Decode, pressure: Option<&mut Pressure>) -> Result<(usize, Vec<u32>), Failure> {
    let prompt = rendered_prompt(&decode.rendered)?;
    match pressure {
        Some(pressure) => pressure.decode(engine, &prompt, decode.tokens),
        None => {
            let length = decode.prompt_tokens.unwrap_or(prompt.len());
            let prompt = prompt
                .get(..length)
                .ok_or_else(|| format!("the fixture has {} tokens, not {length}", prompt.len()))?;
            let (tokens, _) = decoded(engine, prompt, decode.tokens, |_| Ok(false))?;
            Ok((prompt.len(), tokens))
        }
    }
}

/// The first fixture of a rendered set: its tokens.
fn rendered_prompt(rendered: &std::path::Path) -> Result<Vec<TokenId>, String> {
    let text = std::fs::read_to_string(rendered).map_err(|error| format!("{}: {error}", rendered.display()))?;
    let line = text
        .lines()
        .next()
        .ok_or_else(|| format!("{} renders no fixture", rendered.display()))?;
    let rendered: Value = serde_json::from_str(line).map_err(|error| error.to_string())?;
    rendered["tokens"]
        .as_array()
        .ok_or("a rendered fixture has no tokens")?
        .iter()
        .map(|token| {
            token
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .map(TokenId)
                .ok_or_else(|| format!("{token} is no token"))
        })
        .collect()
}

/// The tokens a request's prompt and output may hold together.
fn context_limit(engine: &ReadyEngine) -> Result<usize, String> {
    usize::try_from(engine.host().definition().decoder.context_limit)
        .map_err(|_| "the context exceeds the host domain".into())
}

/// A moment of a decode ([`decoded`]): its request admitted, or this many
/// tokens out.
#[derive(Clone, Copy)]
enum Moment {
    Admitted,
    Out(usize),
}

/// Admit `prompt` for at most `max_tokens` greedy tokens and decode, telling
/// `at` each [`Moment`]; once `at` says to stop, the request is dropped,
/// which cancels it. The tokens, and whether `at` stopped it before it
/// completed.
fn decoded(
    engine: &ReadyEngine,
    prompt: &[TokenId],
    max_tokens: usize,
    mut at: impl FnMut(Moment) -> Result<bool, String>,
) -> Result<(Vec<u32>, bool), Failure> {
    let host = engine.host();
    let input = host.prepare_input(prompt.to_vec(), &[]).map_err(|error| error.to_string())?;
    let definition = host.definition();
    let options = Options {
        max_tokens,
        output_capacity: 16,
        context_limit: context_limit(engine)?,
        vocabulary: usize::try_from(definition.decoder.vocabulary)
            .map_err(|_| "the vocabulary exceeds the host domain".to_owned())?,
        stop_tokens: host.tokenizer().stop_tokens().clone(),
        suppressed_tokens: host.tokenizer().suppressed_tokens().clone(),
        sampling: Sampling::Greedy,
        shaping: Shaping {
            temperature: 0.0,
            ..Shaping::default()
        },
        seed: 0,
        forced_quantum: 0,
        method: MethodChoice::Plain,
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
            .await?;
        at(Moment::Admitted)?;
        let mut tokens = Vec::with_capacity(max_tokens);
        loop {
            match request.receive().await {
                Some(RequestEvent::Output(output)) => {
                    tokens.extend(output.iter().map(|output| output.token.0));
                    if at(Moment::Out(tokens.len()))? {
                        drop(request);
                        return Ok((tokens, true));
                    }
                }
                Some(RequestEvent::Completed { .. }) => return Ok((tokens, false)),
                Some(RequestEvent::Failed(error)) => return Err(error.into()),
                None => return Err(Failure::Cell("the request ended without a terminal event".into())),
            }
        }
    })
}

/// What a pressured decode does at an output ([`Window::next`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Next {
    Decode,
    /// Signal the occupying process to press.
    Press,
    /// Give the memory back: the window has its tokens.
    Free,
    /// Stop the decode: the tokens after the window are out too.
    Stop,
}

/// Where the occupying process's pressure is: not yet landing (signalled or
/// not, it is still approaching the line), landing (it takes the rest past
/// the line), or held (it holds everything).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pressed {
    Unlanded,
    Landing,
    Held,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Before,
    Pressing,
    Landing { from: usize },
    Held { from: usize, held: usize },
    Freed { from: usize, to: usize },
}

/// The window a pressured decode is measured in. Pressing is signalled once
/// `press_at` tokens are out; the window opens at the first output after the
/// pressure starts landing, as the engine answers it, and closes `span`
/// tokens after the first output that finds it held; the memory is then
/// given back, and the decode stops `after` tokens later. However long the
/// pressure takes to land, the window waits for it, so a decode that ends
/// before it is held tested nothing ([`Window::ended`]).
#[derive(Debug)]
struct Window {
    press_at: usize,
    span: usize,
    after: usize,
    stage: Stage,
}

impl Window {
    /// The window of a decode of `tokens`: a quarter of them before pressing
    /// (none, pressing from the prompt), half under the held pressure, the
    /// rest after it.
    fn of(tokens: usize, from_prompt: bool) -> Self {
        let span = tokens / 2;
        Self {
            press_at: if from_prompt { 0 } else { tokens / 4 },
            span,
            after: tokens - tokens / 4 - span,
            stage: Stage::Before,
        }
    }

    /// The tokens a decode needs for the window, the pressure taking
    /// `occupying` seconds to be held and a step `step` seconds: before
    /// pressing, the steps the pressure takes, the window and after it.
    fn room(&self, occupying: f64, step: f64) -> usize {
        self.press_at + (occupying / step).ceil() as usize + self.span + self.after
    }

    /// `decoded` tokens are out, the pressure where `pressed` says: what to
    /// do.
    fn next(&mut self, decoded: usize, pressed: Pressed) -> Next {
        let (stage, next) = match (self.stage, pressed) {
            (Stage::Before, _) if decoded >= self.press_at => (Stage::Pressing, Next::Press),
            (Stage::Pressing, Pressed::Landing) => (Stage::Landing { from: decoded }, Next::Decode),
            (Stage::Pressing, Pressed::Held) => (Stage::Held { from: decoded, held: decoded }, Next::Decode),
            (Stage::Landing { from }, Pressed::Held) => (Stage::Held { from, held: decoded }, Next::Decode),
            (Stage::Held { from, held }, _) if decoded >= held + self.span => {
                (Stage::Freed { from, to: decoded }, Next::Free)
            }
            (Stage::Freed { to, .. }, _) if decoded >= to + self.after => (self.stage, Next::Stop),
            (stage, _) => (stage, Next::Decode),
        };
        self.stage = stage;
        next
    }

    /// The window's tokens, once it opened: from, and to once it closed.
    fn tokens(&self) -> (Option<usize>, Option<usize>) {
        match self.stage {
            Stage::Landing { from } | Stage::Held { from, .. } => (Some(from), None),
            Stage::Freed { from, to } => (Some(from), Some(to)),
            Stage::Before | Stage::Pressing => (None, None),
        }
    }

    /// The decode completed at `decoded` tokens before the rule stopped it:
    /// why the window is not whole.
    fn ended(&self, decoded: usize) -> String {
        match self.stage {
            Stage::Before | Stage::Pressing | Stage::Landing { .. } => {
                format!("pressure missed: the decode ended at token {decoded}, before the pressure was held")
            }
            Stage::Held { held, .. } => format!(
                "the decode ended at token {decoded}, {} of the {} tokens its window holds under the held pressure",
                decoded - held,
                self.span
            ),
            Stage::Freed { to, .. } => format!(
                "the decode ended at token {decoded}, {} of the {} tokens after its window",
                decoded - to,
                self.after
            ),
        }
    }
}

/// Another process's memory over a decode (`occupy`, in a sibling process
/// of this executable), pressing the load `deficit` bytes below its planning
/// line over the decode's [`Window`].
struct Pressure {
    backend: BackendName,
    pressing: Pressing,
    /// How long occupying took here before the load ([`time_occupying`]):
    /// its bytes and seconds.
    occupying: (u64, f64),
    /// The occupying process, from the request's admission until it gives
    /// its memory back.
    sibling: Option<Sibling>,
    record: serde_json::Map<String, Value>,
    /// Each output's token count and time.
    steps: Vec<(usize, f64)>,
}

/// The occupying process: a line on `input` starts the pressure, closing it
/// gives the memory back; a thread passes on each report it prints.
struct Sibling {
    child: std::process::Child,
    input: std::process::ChildStdin,
    reports: std::sync::mpsc::Receiver<Result<Value, String>>,
    reader: std::thread::JoinHandle<()>,
}

impl Pressure {
    /// Before the load: how long occupying takes here is timed in a sibling
    /// process ([`time_occupying`]), the measurement the decode is sized by.
    fn new(backend: BackendName, pressing: Pressing) -> Result<Self, String> {
        let text = |error: std::io::Error| error.to_string();
        let timed = std::process::Command::new(std::env::current_exe().map_err(text)?)
            .args(["occupy-time", "--device", backend.as_str(), "--pressure-store", pressing.store.name()])
            .output()
            .map_err(text)?;
        if !timed.status.success() {
            return Err(format!("timing the occupation failed: {}", String::from_utf8_lossy(&timed.stderr)));
        }
        let timed: Value = serde_json::from_slice(&timed.stdout).map_err(|error| error.to_string())?;
        let bytes = timed["bytes"].as_u64().ok_or("the occupation's timing has no bytes")?;
        let seconds = timed["seconds"].as_f64().ok_or("the occupation's timing has no seconds")?;
        if bytes == 0 {
            return Err("the host leaves an engine nothing to claim: no pressure can be sized".into());
        }
        Ok(Self {
            backend,
            record: serde_json::Map::from_iter([
                ("deficit".to_owned(), json!(pressing.deficit)),
                ("occupying".to_owned(), timed),
            ]),
            pressing,
            occupying: (bytes, seconds),
            sibling: None,
            steps: Vec::new(),
        })
    }

    /// Decode `prompt` under the pressure, as its [`Window`] needs: first
    /// `span` tokens from the whole prompt, unpressed, for the step's time;
    /// then the decode sized to hold the window after the pressure lands, from
    /// that and how long occupying took here (more bytes than the pressure
    /// takes, with no load beside them), its prompt cut to leave the room.
    /// The prompt's tokens decoded from, and the tokens.
    fn decode(&mut self, engine: &ReadyEngine, prompt: &[TokenId], tokens: usize) -> Result<(usize, Vec<u32>), Failure> {
        let mut window = Window::of(tokens, self.pressing.from_prompt);
        let mut times = Vec::new();
        decoded(engine, prompt, window.span, |moment| {
            if let Moment::Out(decoded) = moment {
                times.push((decoded, Instant::now()));
            }
            Ok(false)
        })?;
        let step = match (times.first(), times.last()) {
            (Some(&(first, from)), Some(&(last, to))) if last > first => {
                to.duration_since(from).as_secs_f64() / (last - first) as f64
            }
            _ => return Err(Failure::Cell("the step's timing decode gave fewer than two outputs".into())),
        };
        let (bytes, seconds) = self.occupying;
        let occupying = seconds * (bytes + self.pressing.deficit) as f64 / bytes as f64;
        let room = window.room(occupying, step);
        let context = context_limit(engine)?;
        let length = context
            .checked_sub(room)
            .filter(|&length| length > 0)
            .ok_or_else(|| format!("a {context}-token context holds no prompt beside the window's {room} tokens"))?
            .min(prompt.len());
        self.record.insert(
            "sizing".into(),
            json!({ "step_seconds": step, "occupying_seconds": occupying, "room": room, "prompt_tokens": length }),
        );
        let prompt = &prompt[..length];
        let (decoded, stopped) = decoded(engine, prompt, context - length, |moment| {
            let (decoded, pressed) = match moment {
                Moment::Admitted => {
                    self.admitted()?;
                    (0, Pressed::Unlanded)
                }
                Moment::Out(decoded) => {
                    self.steps.push((decoded, unix_seconds()));
                    (decoded, self.pressed()?)
                }
            };
            match window.next(decoded, pressed) {
                Next::Decode => Ok(false),
                Next::Press => self.press(decoded).map(|()| false),
                Next::Free => {
                    self.record.insert("freeing".into(), json!({ "at": unix_seconds(), "token": decoded }));
                    self.free().map(|()| false)
                }
                Next::Stop => Ok(true),
            }
        })?;
        let (from, to) = window.tokens();
        self.record.insert("window".into(), json!({ "from": from, "to": to }));
        match stopped {
            true => Ok((length, decoded)),
            false => Err(Failure::Cell(window.ended(decoded.len()))),
        }
    }

    /// The request was admitted against the host as it is: the occupying
    /// process starts taking everything but `deficit` of what the engine
    /// could still claim, beside the request, so pressing later takes only
    /// twice the deficit, however much memory the host has. A thread passes
    /// on its reports, so the request's outputs are read meanwhile.
    fn admitted(&mut self) -> Result<(), String> {
        let text = |error: std::io::Error| error.to_string();
        let mut child = std::process::Command::new(std::env::current_exe().map_err(text)?)
            .args([
                "occupy",
                "--device",
                self.backend.as_str(),
                "--pressure-store",
                self.pressing.store.name(),
                "--deficit",
                &self.pressing.deficit.to_string(),
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(text)?;
        let output = std::io::BufReader::new(child.stdout.take().expect("piped"));
        let (sender, reports) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            for line in std::io::BufRead::lines(output) {
                let report = line.map_err(|error| error.to_string()).and_then(|line| {
                    serde_json::from_str(&line)
                        .map_err(|error| format!("the occupying process reported {line:?}: {error}"))
                });
                if sender.send(report).is_err() {
                    return;
                }
            }
        });
        self.sibling = Some(Sibling {
            input: child.stdin.take().expect("piped"),
            child,
            reports,
            reader,
        });
        Ok(())
    }

    /// Signal pressing, `decoded` tokens out: the occupying process takes the
    /// rest once it has approached.
    fn press(&mut self, decoded: usize) -> Result<(), String> {
        let sibling = self.sibling.as_mut().ok_or("pressing starts once the request is admitted")?;
        std::io::Write::write_all(&mut sibling.input, b"\n").map_err(|error| error.to_string())?;
        self.record.insert("started".into(), json!({ "at": unix_seconds(), "token": decoded }));
        Ok(())
    }

    /// Record the occupying process's reports so far: where its pressure is
    /// now. It lands once pressing is signalled and the process has
    /// approached the line, which it reads the signal after.
    fn pressed(&mut self) -> Result<Pressed, String> {
        let reports = self
            .sibling
            .as_ref()
            .map(|sibling| sibling.reports.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        self.recorded(reports)?;
        Ok(self.landed())
    }

    /// Where the pressure is by the moments recorded: held once the
    /// occupying process occupied everything, landing once it approached
    /// and pressing was signalled.
    fn landed(&self) -> Pressed {
        let has = |moment: &str| self.record.contains_key(moment);
        match (has("occupied"), has("approached") && has("started")) {
            (true, _) => Pressed::Held,
            (false, true) => Pressed::Landing,
            (false, false) => Pressed::Unlanded,
        }
    }

    /// Record each of the occupying process's `reports` by its moment.
    fn recorded(&mut self, reports: Vec<Result<Value, String>>) -> Result<(), String> {
        for report in reports {
            let report = report?;
            let moment = ["approached", "occupied", "released"]
                .into_iter()
                .find(|moment| report.get(moment).is_some())
                .ok_or_else(|| format!("the occupying process reported {report}"))?;
            self.record.insert(moment.into(), report);
        }
        Ok(())
    }

    /// What the decode under it recorded, its outputs' times (`steps`)
    /// among it.
    fn into_record(mut self) -> serde_json::Map<String, Value> {
        self.record.insert("steps".into(), json!(self.steps));
        self.record
    }

    /// The decode ended with `failure`: under pressure the ladder's last
    /// step, the model unloaded, which the report reads beside the releases.
    /// The memory is given back.
    fn failed(&mut self, failure: String) -> Result<(), String> {
        self.record.insert("ended".into(), json!({ "at": unix_seconds(), "failure": failure }));
        self.free()
    }

    /// Give the memory back, recording the reports the occupying process
    /// made and when the memory is back.
    fn free(&mut self) -> Result<(), String> {
        if let Some(Sibling { mut child, input, reports, reader }) = self.sibling.take() {
            drop(input);
            let status = child.wait().map_err(|error| error.to_string())?;
            if !status.success() {
                return Err(format!("the occupying process failed: {status}"));
            }
            reader.join().map_err(|_| "the occupying process's reader panicked")?;
            self.recorded(reports.try_iter().collect())?;
            if self.record.contains_key("started") {
                self.record.insert("freed".into(), json!({ "at": unix_seconds() }));
            }
        }
        Ok(())
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

/// Print the engine's and the scheduler's events on standard error as this
/// process runs, one JSON line each (`magnitude-harness: event {...}`): its
/// message, its time in Unix seconds (`at`) and its fields by name, an
/// unread value absent. The load's report and memory moments (`load memory`)
/// are among them, as are the owner's releases (`memory released`), memory
/// conditions (`memory condition changed`), which the pressure measurement
/// reads, and its expert bank's growth decisions; with
/// [`Scope::Steps`] also each streamed step's reads (`expert step
/// completed`), which name a slow step's cause, and each of the owner's
/// memory readings (`memory read`), the load's charge against its room.
pub fn print_events(scope: Scope) -> Result<(), String> {
    use tracing_subscriber::layer::SubscriberExt;
    let engine = "magnitude_engine=info,magnitude_executor=info,magnitude_scheduler=info";
    let filter = match scope {
        Scope::Decisions => engine.to_owned(),
        Scope::Steps => format!(
            "{engine},{}=debug,{}=debug",
            magnitude_executor::EXPERT_STEP_TRACE,
            magnitude_executor::MEMORY_TRACE,
        ),
    };
    let subscriber = tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(filter))
        .with(Events);
    tracing::subscriber::set_global_default(subscriber).map_err(|error| error.to_string())
}

/// Which of the events [`print_events`] prints: the engine's decisions
/// alone, or each streamed step's reads beside them, one event a
/// step.
pub enum Scope {
    Decisions,
    Steps,
}

/// The events [`print_events`] prints.
struct Events;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Events {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let mut fields = Fields(serde_json::Map::from_iter([("at".to_owned(), json!(unix_seconds()))]));
        event.record(&mut fields);
        eprintln!("magnitude-harness: event {}", Value::Object(fields.0));
    }
}

/// An event's fields, by name.
struct Fields(serde_json::Map<String, Value>);

impl tracing::field::Visit for Fields {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().into(), json!(value));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), json!(format!("{value:?}")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive `window` over outputs of one token each, the pressure landing
    /// from output `landing` and held from output `held` on: each output's
    /// action but decoding, until it stops or `outputs` are out.
    fn drive(window: &mut Window, landing: usize, held: usize, outputs: usize) -> Vec<(usize, Next)> {
        let pressed = |decoded| match decoded {
            decoded if decoded >= held => Pressed::Held,
            decoded if decoded >= landing => Pressed::Landing,
            _ => Pressed::Unlanded,
        };
        let mut actions = Vec::new();
        for decoded in 1..=outputs {
            match window.next(decoded, pressed(decoded)) {
                Next::Decode => {}
                Next::Stop => {
                    actions.push((decoded, Next::Stop));
                    break;
                }
                next => actions.push((decoded, next)),
            }
        }
        actions
    }

    /// The window opens as the pressure starts landing, however late, and
    /// closes its span after the pressure is held; a decode that ends before
    /// the pressure is held tested nothing.
    #[test]
    fn the_window_spans_the_landing_and_its_tokens_under_the_held_pressure() {
        // 128 tokens: pressed at 32, landing from 250, held at 300.
        let mut window = Window::of(128, false);
        assert_eq!(
            drive(&mut window, 250, 300, 1000),
            [(32, Next::Press), (364, Next::Free), (396, Next::Stop)]
        );
        assert_eq!(window.tokens(), (Some(250), Some(364)));

        // From the prompt: pressed at admission, held at the first output.
        let mut prompt = Window::of(128, true);
        assert_eq!(prompt.next(0, Pressed::Unlanded), Next::Press);
        assert_eq!(prompt.next(1, Pressed::Held), Next::Decode);
        assert_eq!(prompt.next(65, Pressed::Held), Next::Free);
        assert_eq!(prompt.tokens(), (Some(1), Some(65)));

        // Ended while the pressure landed: missed, never passed.
        let mut missed = Window::of(128, true);
        missed.next(0, Pressed::Unlanded);
        missed.next(90, Pressed::Landing);
        assert!(missed.ended(91).starts_with("pressure missed"));
        assert_eq!(missed.tokens(), (Some(90), None));

        // Ended under the held pressure, short of its span.
        let mut short = Window::of(128, true);
        short.next(0, Pressed::Unlanded);
        short.next(10, Pressed::Held);
        assert_eq!(
            short.ended(40),
            "the decode ended at token 40, 30 of the 64 tokens its window holds under the held pressure"
        );
    }

    /// A pressure lands once the occupying process has approached and
    /// pressing was signalled, in either order, and is held once it has
    /// occupied; a report of no moment fails. Freed or ended with no
    /// occupying process, it records no freeing, and an ending records why.
    #[test]
    fn the_pressure_lands_on_its_signal_and_approach_and_is_held_once_occupied() {
        let mut pressure = Pressure {
            backend: BackendName::Cpu,
            pressing: Pressing { store: PressureStore::Host, deficit: 1, from_prompt: false },
            occupying: (1, 1.0),
            sibling: None,
            record: serde_json::Map::new(),
            steps: Vec::new(),
        };
        assert_eq!(pressure.landed(), Pressed::Unlanded);
        pressure.recorded(vec![Ok(json!({ "approached": 1, "ceiling": 2, "at": 0.0 }))]).unwrap();
        assert_eq!(pressure.landed(), Pressed::Unlanded);
        pressure.record.insert("started".into(), json!({ "at": 0.0, "token": 3 }));
        assert_eq!(pressure.landed(), Pressed::Landing);
        pressure.recorded(vec![Ok(json!({ "occupied": 3, "ceiling": 2, "at": 1.0 }))]).unwrap();
        assert_eq!(pressure.landed(), Pressed::Held);
        assert!(pressure.recorded(vec![Ok(json!({ "other": 1 }))]).is_err());
        assert!(pressure.recorded(vec![Err("unreadable".into())]).is_err());

        pressure.free().unwrap();
        assert!(!pressure.record.contains_key("freed"));
        pressure.failed("model unloaded: memory pressure".into()).unwrap();
        assert_eq!(pressure.record["ended"]["failure"], "model unloaded: memory pressure");
    }

    /// Only the engine's unload under memory pressure is the pressure's
    /// outcome; every other request error fails the measurement.
    #[test]
    fn only_an_unload_under_memory_pressure_ends_a_pressured_decode() {
        let unloaded = RequestError::ModelUnloaded {
            cause: UnloadCause::MemoryPressure,
        };
        assert_eq!(Failure::from(unloaded.clone()), Failure::Unloaded(unloaded.to_string()));
        for error in [
            RequestError::ModelUnloaded {
                cause: UnloadCause::Shutdown,
            },
            RequestError::MemoryReclaim,
            RequestError::InvalidRequest { reason: "a bad prompt".into() },
        ] {
            assert_eq!(Failure::from(error.clone()), Failure::Cell(error.to_string()));
        }
    }

    /// The decode holds the window after the pressure lands: the steps the
    /// occupation's measured time takes, rounded up, beside the window.
    #[test]
    fn the_decode_is_sized_by_the_occupation_and_the_step() {
        let window = Window::of(128, false);
        assert_eq!(window.room(6.0, 0.025), 32 + 240 + 64 + 32);
        assert_eq!(window.room(6.01, 0.025), 32 + 241 + 64 + 32);
        assert_eq!(Window::of(128, true).room(0.0, 0.025), 64 + 32);
    }
}
