//! Forward fingerprint: the greedy tokens and a SHA-256 of every exported
//! logits row's bits for a fixed forced workload.
//!
//! Two runs on the same device with the same kernel cache print identical
//! lines when the forward is bit-identical. The workload covers each target
//! launch form (a long prefill, a multi-row replay, single-row greedy decode
//! and two sequences decoding together) and, with `--head`, the draft head's
//! greedy proposal chain.
//!
//! ```text
//! forward_fingerprint --model M.gguf [--cache-dir DIR] [--kv-codec dense|affine-k8v4]
//!     [--steps 24] [--head] [--device auto|metal|cuda|vulkan|cpu|SELECTOR]
//! ```

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, DomainError, ExecutionPath, ExecutorDomain, FeatureRows, Operation, Outcome, PhysicalDecision,
    RequestId, RowResult, Sampling, SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use magnitude_state::KvCodec;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const PROMPT: usize = 40;
const REPLAY: usize = 5;
const PROPOSALS: usize = 3;

struct Options {
    model: PathBuf,
    cache: Option<PathBuf>,
    codec: KvCodec,
    steps: usize,
    head: bool,
    device: DeviceRequest,
}

fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        model: PathBuf::new(),
        cache: None,
        codec: KvCodec::AffineK8V4,
        steps: 24,
        head: false,
        device: DeviceRequest::Automatic,
    };
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} requires a value"));
        match flag.as_str() {
            "--model" => options.model = PathBuf::from(value()?),
            "--cache-dir" => options.cache = Some(PathBuf::from(value()?)),
            "--kv-codec" => options.codec = value()?.parse()?,
            "--steps" => options.steps = value()?.parse().map_err(|_| "--steps takes a count")?,
            "--head" => options.head = true,
            "--device" => {
                options.device = value()?
                    .parse()
                    .map_err(|error| format!("--device: {error}"))?
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if options.model.as_os_str().is_empty() {
        return Err("--model is required".into());
    }
    Ok(options)
}

struct Fingerprint {
    domain: ExecutorDomain,
    vocabulary: usize,
    hidden: usize,
}

struct Sequence {
    request: RequestId,
    position: usize,
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

impl Fingerprint {
    /// A fixed spread over ordinary vocabulary rows.
    fn forced(&self, start: usize, rows: usize) -> Vec<TokenId> {
        let span = (self.vocabulary - 1024) as u64;
        (start..start + rows)
            .map(|position| {
                TokenId(((position as u64 * 2_654_435_761 + 12_345) % span + 512) as u32)
            })
            .collect()
    }

    fn open(&mut self, id: u64) -> Result<Sequence, String> {
        let request = RequestId(id);
        self.domain
            .install_input(request, PreparedModelInput::continuation_only())?;
        self.domain.open_state(request, None).map_err(text)?;
        Ok(Sequence {
            request,
            position: 0,
        })
    }

    /// One target group of forwards; every row commits and is accepted.
    fn step(
        &mut self,
        forwards: Vec<(&mut Sequence, WorkKind, Vec<TokenId>, Demand)>,
    ) -> Result<Vec<Vec<RowResult>>, String> {
        let mut operations = Vec::new();
        let mut sequences = Vec::new();
        for (sequence, kind, tokens, demand) in forwards {
            let rows = tokens.len();
            let select = if demand.contains(Demand::SELECT) {
                vec![greedy(sequence.position + rows - 1)]
            } else {
                Vec::new()
            };
            operations.push(Operation::Forward {
                request: sequence.request,
                kind,
                tokens,
                position: sequence.position,
                conditioning: None,
                demand,
                select,
                committed: rows,
            });
            sequences.push((sequence, rows));
        }
        let groups = service_domain::group(&self.domain, operations);
        let [group] = groups.as_slice() else {
            return Err("a fingerprint step must form one group".into());
        };
        let DomainFlight::Target(flight) = service_domain::submit_group(&mut self.domain, group)
            .map_err(|error| error.to_string())?
        else {
            return Err("a forward runs on the target lane".into());
        };
        let pending = self.domain.finish_target(flight).map_err(text)?;
        let mut results = Vec::new();
        for (pending, (sequence, rows)) in pending.into_iter().zip(sequences) {
            let Outcome::Forward { rows: outcome } = pending.outcome().clone() else {
                return Err("a forward returns forward rows".into());
            };
            self.domain
                .reconcile(
                    pending,
                    PhysicalDecision {
                        accepted_rows: rows,
                    },
                )
                .map_err(text)?;
            sequence.position += rows;
            results.push(outcome);
        }
        Ok(results)
    }

    /// A draft-head transaction: one committed entry row, then a greedy
    /// proposal chain.
    fn head(&mut self, sequence: &Sequence, token: TokenId) -> Result<Vec<TokenId>, String> {
        // A fixed bf16 conditioning row with values in [-1, 1).
        let row = (0..self.hidden)
            .flat_map(|index| {
                let value = ((index * 37 + sequence.position * 11) % 256) as f32 / 128.0 - 1.0;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect::<Vec<u8>>();
        let operation = Operation::Head {
            request: sequence.request,
            tokens: vec![token],
            conditioning: FeatureRows::new(row.into(), 1).map_err(|error| error.to_string())?,
            position: sequence.position,
            proposals: (1..=PROPOSALS)
                .map(|offset| greedy(sequence.position + offset))
                .collect(),
            form: magnitude_executor::DraftForm::Chained,
        };
        let groups = service_domain::group(&self.domain, vec![operation]);
        let [group] = groups.as_slice() else {
            return Err("one head operation forms one group".into());
        };
        let DomainFlight::Head(flight) = service_domain::submit_group(&mut self.domain, group)
            .map_err(|error| error.to_string())?
        else {
            return Err("a head operation runs on the head lane".into());
        };
        let mut proposals = Vec::new();
        for pending in self.domain.finish_head(flight).map_err(text)? {
            let Outcome::Head { proposals: chain } = pending.outcome().clone() else {
                return Err("a head operation returns proposals".into());
            };
            proposals.extend(chain.into_iter().map(|selected| selected.token));
            self.domain
                .reconcile(pending, PhysicalDecision { accepted_rows: 1 })
                .map_err(text)?;
        }
        Ok(proposals)
    }
}

/// Hash every exported logits row, and collect every selected token.
fn digest(rows: &[RowResult], hasher: &mut Sha256, tokens: &mut Vec<u32>) -> Result<(), String> {
    for row in rows {
        if let Some(logits) = &row.logits {
            for value in logits.read_to_host().map_err(|error| error.to_string())? {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        if let Some(selected) = &row.selected {
            tokens.push(selected.token.0);
        }
    }
    Ok(())
}

fn hex(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
            method: if options.head {
                ModelMethod::Mtp
            } else {
                ModelMethod::Plain
            },
            mtp_proposals: options.head.then_some(PROPOSALS as u8),
            kv_codec: options.codec,
            lookahead: false,
        },
        context_tokens: Some(2048),
        service: ServiceLimits {
            prefill_tokens: 64,
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
    let decoder = &resolved.manifest.definition.decoder;
    let vocabulary = usize::try_from(decoder.vocabulary).map_err(|_| "vocabulary exceeds host")?;
    let hidden = usize::try_from(decoder.hidden).map_err(|_| "hidden exceeds host")?;
    let package = resolved.host.shared_package();
    let (domain, _) =
        build_native_domain(&resolved.manifest, package).map_err(|error| error.to_string())?;
    let mut run = Fingerprint {
        domain,
        vocabulary,
        hidden,
    };

    let mut first = run.open(1)?;
    let prompt = run.forced(0, PROMPT);
    let mut hasher = Sha256::new();
    let mut tokens = Vec::new();
    for rows in run.step(vec![(&mut first, WorkKind::Prefill, prompt, Demand::LOGITS)])? {
        digest(&rows, &mut hasher, &mut tokens)?;
    }
    println!("prefill rows={PROMPT} logits={}", hex(hasher));

    let replay = run.forced(first.position, REPLAY);
    let mut hasher = Sha256::new();
    for rows in run.step(vec![(&mut first, WorkKind::Replay, replay, Demand::LOGITS)])? {
        digest(&rows, &mut hasher, &mut tokens)?;
    }
    println!("replay rows={REPLAY} logits={}", hex(hasher));

    let mut hasher = Sha256::new();
    let mut greedy_tokens = Vec::new();
    let mut next = run.forced(first.position, 1)[0];
    for _ in 0..options.steps {
        let results = run.step(vec![(
            &mut first,
            WorkKind::Decode,
            vec![next],
            Demand::SELECT | Demand::LOGITS,
        )])?;
        for rows in results {
            digest(&rows, &mut hasher, &mut greedy_tokens)?;
        }
        next = TokenId(*greedy_tokens.last().ok_or("decode selected no token")?);
    }
    println!("decode steps={} tokens={greedy_tokens:?} logits={}", options.steps, hex(hasher));

    let mut second = run.open(2)?;
    let prompt = run.forced(1000, PROMPT + 13);
    run.step(vec![(&mut second, WorkKind::Prefill, prompt, Demand::NONE)])?;
    let mut hasher = Sha256::new();
    let mut pair_tokens = Vec::new();
    let mut heads = [next, run.forced(second.position, 1)[0]];
    for _ in 0..8 {
        let results = run.step(vec![
            (
                &mut first,
                WorkKind::Decode,
                vec![heads[0]],
                Demand::SELECT | Demand::LOGITS,
            ),
            (
                &mut second,
                WorkKind::Decode,
                vec![heads[1]],
                Demand::SELECT | Demand::LOGITS,
            ),
        ])?;
        for (lane, rows) in results.iter().enumerate() {
            let mut selected = Vec::new();
            digest(rows, &mut hasher, &mut selected)?;
            heads[lane] = TokenId(*selected.last().ok_or("decode selected no token")?);
            pair_tokens.extend(selected);
        }
    }
    println!("concurrent steps=8 tokens={pair_tokens:?} logits={}", hex(hasher));

    if options.head {
        let mut proposals = Vec::new();
        let mut sequence = run.open(3)?;
        for round in 0..6 {
            let token = run.forced(sequence.position + round * 7, 1)[0];
            proposals.extend(run.head(&sequence, token)?.into_iter().map(|token| token.0));
            sequence.position += 1;
        }
        println!("head rounds=6 proposals={proposals:?}");
    }
    Ok(())
}
