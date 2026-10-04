//! Token logits: the logits rows of one sequence fed the given tokens, as
//! raw little-endian F32 `[tokens - prefill + 1, vocabulary]`, for
//! position-by-position comparison with a family reference's `logits`
//! output.
//!
//! The first `--prefill` tokens are prefilled in chunks of at most 512 rows,
//! and the last chunk's last row's logits are the first written; the rest are
//! one-row decodes, as a served request runs them.
//!
//! ```text
//! token_logits --model M.gguf --tokens 1,2,3 --output logits.f32 [--prefill N]
//!     [--cache-dir DIR] [--kv-codec dense|affine-k8v4] [--device auto|cuda|...]
//!     [--tuning-record FILE | --tuning-replay FILE]   (with `--features pinned-tuning`)
//! ```

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, DomainError, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision,
    RequestId, ReservedResources, StateBindings, TokenId, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;
use std::io::Write;
use std::path::PathBuf;

/// Rows of one prefill forward.
const PREFILL_TOKENS: usize = 512;

struct Options {
    model: PathBuf,
    tokens: Vec<TokenId>,
    output: PathBuf,
    prefill: usize,
    cache: Option<PathBuf>,
    codec: KvCodec,
    device: DeviceRequest,
}

fn options(args: Vec<String>) -> Result<Options, String> {
    let mut args = args.into_iter();
    let (mut model, mut output, mut tokens) = (None, None, None);
    let mut options = Options {
        model: PathBuf::new(),
        tokens: Vec::new(),
        output: PathBuf::new(),
        prefill: 1,
        cache: None,
        codec: KvCodec::Dense,
        device: DeviceRequest::Automatic,
    };
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} requires a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--tokens" => {
                tokens = Some(
                    value()?
                        .split(',')
                        .map(|token| token.trim().parse().map(TokenId))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| "--tokens takes comma-separated token ids")?,
                )
            }
            "--prefill" => {
                options.prefill = value()?.parse().map_err(|_| "--prefill takes a count")?
            }
            "--cache-dir" => options.cache = Some(PathBuf::from(value()?)),
            "--kv-codec" => options.codec = value()?.parse()?,
            "--device" => {
                options.device = value()?
                    .parse()
                    .map_err(|error| format!("--device: {error}"))?
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    options.model = model.ok_or("--model is required")?;
    options.output = output.ok_or("--output is required")?;
    options.tokens = tokens.ok_or("--tokens is required")?;
    if options.prefill == 0 || options.prefill > options.tokens.len() {
        return Err("--prefill must be between 1 and the token count".into());
    }
    Ok(options)
}

fn text(error: DomainError) -> String {
    error.to_string()
}

/// One forward of `tokens` at `position`, the logits of every row that
/// returns them (a prefill returns its last row's) appended to `logits`. The
/// flight carries the bindings and returns them.
fn forward(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    kind: WorkKind,
    tokens: Vec<TokenId>,
    position: usize,
    logits: &mut Vec<f32>,
) -> Result<StateBindings, String> {
    let rows = tokens.len();
    let operation = Operation::Forward {
        request,
        kind,
        tokens,
        position,
        conditioning: None,
        demand: Demand::LOGITS,
        select: Vec::new(),
        committed: rows,
        prime: None,
    };
    let operations = [operation];
    let ReservedResources::Target(reservation) = domain
        .reserve(&mut bindings, &operations)
        .map_err(text)?
        .into_resources()
    else {
        return Err("a forward runs on the target lane".into());
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .map_err(|failure| failure.error().to_string())?;
    let (pending, bindings) = domain.finish_target(flight).map_err(text)?;
    for pending in pending {
        let Outcome::Forward { rows: results } = pending.outcome().clone() else {
            return Err("a forward returns forward rows".into());
        };
        for logits_row in results.iter().filter_map(|row| row.logits.as_ref()) {
            logits.extend(
                logits_row
                    .read_to_host()
                    .map_err(|error| error.to_string())?,
            );
        }
        domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: rows,
                },
            )
            .map_err(text)?;
    }
    Ok(bindings)
}

/// `--tuning-record FILE` / `--tuning-replay FILE` (`support/tuning_pin.rs`).
#[cfg(feature = "pinned-tuning")]
#[path = "support/tuning_pin.rs"]
mod tuning_pin;

fn main() -> Result<(), String> {
    #[cfg_attr(not(feature = "pinned-tuning"), allow(unused_mut))]
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    #[cfg(feature = "pinned-tuning")]
    let pin = tuning_pin::Pin::extract(&mut args)?;
    let options = options(args)?;
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: options.model.clone(),
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: options.codec,
            lookahead: false,
            exported_logits_rows: 512,
            error_classes: Vec::new(),
        },
        context_tokens: Some(options.tokens.len().next_power_of_two().max(256)),
        service: ServiceLimits {
            prefill_tokens: PREFILL_TOKENS,
            decode_tokens: 16,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: options.device,
        kernel_cache: options.cache.clone(),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .map_err(|error| error.to_string())?;
    let package = resolved.host.shared_package();
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, package).map_err(|error| error.to_string())?;
    let request = RequestId(1);
    domain.install_input(request, PreparedModelInput::continuation_only())?;
    domain
        .open_state(&mut bindings, request, None)
        .map_err(text)?;
    let mut logits = Vec::new();
    let (prefill, decode) = options.tokens.split_at(options.prefill);
    // A prompt longer than one prefill runs as consecutive chunks, as the
    // service runs it; only the last chunk's row is written.
    for (index, chunk) in prefill.chunks(PREFILL_TOKENS).enumerate() {
        logits.clear();
        bindings = forward(
            &mut domain,
            bindings,
            request,
            WorkKind::Prefill,
            chunk.to_vec(),
            index * PREFILL_TOKENS,
            &mut logits,
        )?;
    }
    for (offset, token) in decode.iter().enumerate() {
        bindings = forward(
            &mut domain,
            bindings,
            request,
            WorkKind::Decode,
            vec![*token],
            options.prefill + offset,
            &mut logits,
        )?;
    }
    let mut file = std::fs::File::create(&options.output).map_err(|error| error.to_string())?;
    for value in logits {
        file.write_all(&value.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }
    #[cfg(feature = "pinned-tuning")]
    pin.finish()?;
    Ok(())
}
