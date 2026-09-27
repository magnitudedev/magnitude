//! Token logits: the logits rows of one sequence fed the given tokens, as
//! raw little-endian F32 `[tokens - prefill + 1, vocabulary]`, for
//! position-by-position comparison with a family reference's `logits`
//! output.
//!
//! The first `--prefill` tokens are one prefill, whose last row's logits are
//! the first written; the rest are one-row decodes, as a served request runs
//! them.
//!
//! ```text
//! token_logits --model M.gguf --tokens 1,2,3 --output logits.f32 [--prefill N]
//!     [--cache-dir DIR] [--kv-codec dense|affine-k8v4] [--device auto|cuda|...]
//! ```

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, DomainError, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision,
    RequestId, TokenId, WorkKind,
};
use magnitude_scheduler::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use magnitude_state::KvCodec;
use std::io::Write;
use std::path::PathBuf;

struct Options {
    model: PathBuf,
    tokens: Vec<TokenId>,
    output: PathBuf,
    prefill: usize,
    cache: Option<PathBuf>,
    codec: KvCodec,
    device: DeviceRequest,
}

fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
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
/// returns them (a prefill returns its last row's) appended to `logits`.
fn forward(
    domain: &mut ExecutorDomain,
    request: RequestId,
    kind: WorkKind,
    tokens: Vec<TokenId>,
    position: usize,
    logits: &mut Vec<f32>,
) -> Result<(), String> {
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
    };
    let groups = service_domain::group(domain, vec![operation]);
    let [group] = groups.as_slice() else {
        return Err("one forward forms one group".into());
    };
    let DomainFlight::Target(flight) =
        service_domain::submit_group(domain, group).map_err(|error| error.to_string())?
    else {
        return Err("a forward runs on the target lane".into());
    };
    for pending in domain.finish_target(flight).map_err(text)? {
        let Outcome::Forward { rows: results } = pending.outcome().clone() else {
            return Err("a forward returns forward rows".into());
        };
        for logits_row in results.iter().filter_map(|row| row.logits.as_ref()) {
            logits.extend(logits_row.read_to_host().map_err(|error| error.to_string())?);
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
    Ok(())
}

fn main() -> Result<(), String> {
    let options = options()?;
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
        },
        context_tokens: Some(options.tokens.len().next_power_of_two().max(256)),
        service: ServiceLimits {
            prefill_tokens: 512,
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
    let (mut domain, _) =
        build_native_domain(&resolved.manifest, package).map_err(|error| error.to_string())?;
    let request = RequestId(1);
    domain.open(request).map_err(text)?;
    let mut logits = Vec::new();
    let (prefill, decode) = options.tokens.split_at(options.prefill);
    forward(
        &mut domain,
        request,
        WorkKind::Prefill,
        prefill.to_vec(),
        0,
        &mut logits,
    )?;
    for (offset, token) in decode.iter().enumerate() {
        forward(
            &mut domain,
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
    Ok(())
}
