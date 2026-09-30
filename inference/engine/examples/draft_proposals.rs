//! Draft proposals: one separate-draft (DFlash, DSpark) transaction after the
//! target has run the given tokens, for comparison with
//! `validation/dflash_reference.py propose` and the draft fixtures' passes.
//!
//! The target prefills `tokens[..n]` (its features condition the draft), then
//! the draft injects them and drafts its greedy proposals from the anchor
//! `tokens[n]`. Prints the proposals, the token at position `n + 1 + k`
//! first.
//!
//! ```text
//! draft_proposals --model T.gguf --draft D.gguf --tokens 1,2,3 [--proposals P]
//!     [--cache-dir DIR] [--device auto|cuda|...]
//! ```

use magnitude_engine::{
    build_native_domain,
    host::package_definition,
    options::{ExecutionManifest, ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, DomainError, DraftForm, ExecutionPath, FeatureReader, FeatureSpan, Operation, Outcome,
    PhysicalDecision, RequestId, Sampling, SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use magnitude_state::KvCodec;
use std::path::PathBuf;

struct Options {
    model: PathBuf,
    draft: PathBuf,
    tokens: Vec<TokenId>,
    proposals: Option<u8>,
    cache: Option<PathBuf>,
    device: DeviceRequest,
}

fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut draft, mut tokens) = (None, None, None);
    let mut proposals = None;
    let mut cache = None;
    let mut device = DeviceRequest::Automatic;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} requires a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--draft" => draft = Some(PathBuf::from(value()?)),
            "--tokens" => {
                tokens = Some(
                    value()?
                        .split(',')
                        .map(|token| token.trim().parse().map(TokenId))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| "--tokens takes comma-separated token ids")?,
                )
            }
            "--proposals" => {
                proposals = Some(value()?.parse().map_err(|_| "--proposals takes a count")?)
            }
            "--cache-dir" => cache = Some(PathBuf::from(value()?)),
            "--device" => {
                device = value()?
                    .parse()
                    .map_err(|error| format!("--device: {error}"))?
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let tokens: Vec<TokenId> = tokens.ok_or("--tokens is required")?;
    if tokens.len() < 2 {
        return Err("--tokens needs committed tokens and the anchor".into());
    }
    Ok(Options {
        model: model.ok_or("--model is required")?,
        draft: draft.ok_or("--draft is required")?,
        tokens,
        proposals,
        cache,
        device,
    })
}

fn text(error: DomainError) -> String {
    error.to_string()
}

fn greedy(position: usize) -> SelectSpec {
    SelectSpec {
        sampling: Sampling::Greedy,
        seed: 0,
        position,
        domain: 0,
        mask: None,
        shaping: Shaping::default(),
        history: None,
    }
}

fn main() -> Result<(), String> {
    let options = options()?;
    let committed = options.tokens.len() - 1;
    // No host artifacts: synthetic fixtures carry tokenizers the chat layer
    // does not admit, and proposals need none.
    let package = PackageOptions {
        target: options.model.clone(),
        projector: ProjectorSelection::Disabled,
        draft: Some(options.draft.clone()),
    }
    .open()
    .map_err(|error| error.to_string())?;
    let (_, mut definition) = package_definition(&package).map_err(|error| error.to_string())?;
    definition.decoder.context_limit = definition
        .decoder
        .context_limit
        .min(options.tokens.len().next_power_of_two().max(256) as u64);
    let model = ModelPolicy {
        // The draft's own variant (DFlash, DSpark or DFlash2).
        method: ModelMethod::Auto,
        mtp_proposals: options.proposals,
        kv_codec: KvCodec::Dense,
        lookahead: false,
    }
    .resolve(&definition)?;
    let proposals = model.method.proposals();
    let manifest = ExecutionManifest::new(
        package.manifest(),
        definition,
        model,
        ServiceLimits {
            prefill_tokens: 512,
            decode_tokens: 16,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        ExecutionPath::Native,
        options.device,
        options.cache.clone(),
        MemoryReserves::standard(),
    )?;
    let (mut domain, _) = build_native_domain(&manifest, std::sync::Arc::new(package))
        .map_err(|error| error.to_string())?;
    let request = RequestId(1);
    domain.install_input(request, PreparedModelInput::continuation_only())?;
    domain.open_state(request, None).map_err(text)?;

    // The target runs the committed tokens; every row's features condition
    // the draft.
    let prefill = Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: options.tokens[..committed].to_vec(),
        position: 0,
        conditioning: None,
        demand: Demand::FEATURES,
        select: Vec::new(),
        committed,
        prime: None,
    };
    let groups = service_domain::group(&domain, vec![prefill]);
    let [group] = groups.as_slice() else {
        return Err("one prefill forms one group".into());
    };
    let DomainFlight::Target(flight) =
        service_domain::submit_group(&mut domain, group).map_err(|error| error.to_string())?
    else {
        return Err("a prefill runs on the target lane".into());
    };
    let mut features = None;
    for pending in domain.finish_target(flight).map_err(text)? {
        let Outcome::Forward { rows } = pending.outcome().clone() else {
            return Err("a prefill returns forward rows".into());
        };
        features = rows.iter().find_map(|row| row.features.clone());
        domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: committed,
                },
            )
            .map_err(text)?;
    }
    let features = features.ok_or("the prefill returned no features")?;
    let conditioning = domain
        .read(&FeatureSpan::new(features, 0, committed).map_err(|error| error.to_string())?)?;

    // Entry row p pairs the token after target row p with row p's feature;
    // the last entry token is the anchor.
    let head = Operation::Head {
        request,
        phase: magnitude_executor::HeadPhase::Generation,
        tokens: options.tokens[1..].to_vec(),
        conditioning,
        position: 0,
        proposals: (0..proposals)
            .map(|proposal| greedy(committed + 1 + proposal))
            .collect(),
        form: DraftForm::Block,
    };
    let groups = service_domain::group(&domain, vec![head]);
    let [group] = groups.as_slice() else {
        return Err("one draft transaction forms one group".into());
    };
    let DomainFlight::Head(flight) =
        service_domain::submit_group(&mut domain, group).map_err(|error| error.to_string())?
    else {
        return Err("a draft transaction runs on the head lane".into());
    };
    for pending in domain.finish_head(flight).map_err(text)? {
        let Outcome::Head { proposals } = pending.outcome().clone() else {
            return Err("a draft transaction returns proposals".into());
        };
        println!(
            "committed={committed} anchor={} positions={:?} proposals={:?} statuses={:?}",
            options.tokens[committed].0,
            (0..proposals.len())
                .map(|proposal| committed + 1 + proposal)
                .collect::<Vec<_>>(),
            proposals
                .iter()
                .map(|selected| selected.token.0)
                .collect::<Vec<_>>(),
            proposals
                .iter()
                .map(|selected| selected.status)
                .collect::<Vec<_>>(),
        );
        domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: committed,
                },
            )
            .map_err(text)?;
    }
    Ok(())
}
