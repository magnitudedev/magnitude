//! `draft`: the catalog drafter's behaviour on fixtures. At anchors spread
//! over each fixture's scored spans, the target prefills the tokens before
//! the anchor (its features condition the drafter), a drafter keeping its own
//! history is primed on them as the served path primes it (one reading the
//! target's history, such as an MTP assistant, enters none), drafts greedy
//! proposals from the
//! anchor token, and the target verifies them: the accepted prefix is the
//! proposals equal to the target's own greedy choices. Proposals are also
//! compared with the fixture's own next tokens.

use crate::artifacts::Artifacts;
use crate::engine::{self, Engine, EngineOptions, Load, PREFILL_ROWS, text};
use crate::render::Rendered;
use magnitude_engine::options::ResolvedMethod;
use magnitude_executor::{
    Demand, DraftForm, FeatureReader, FeatureSpan, HeadPhase, Operation, Outcome, PhysicalDecision,
    ReservedResources, TokenId, WorkKind,
};
use magnitude_family_contracts::HeadHistory;
use seismic::{TraceDetail, host_seconds};
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;

pub struct DraftOptions {
    /// Anchors per fixture, spread evenly over its scored positions.
    pub anchors: usize,
    /// Proposals per draft; the catalog drafter's own width when absent.
    pub proposals: Option<u8>,
    /// Time each anchor's draft, its verification and one plain decode step
    /// on the device, per submission (a diagnostic).
    pub attribute: bool,
}

/// Draft at every fixture's anchors, writing one JSON line per anchor to
/// `records`. Returns the drafting measurement and the device's name.
pub fn run(
    artifacts: &Artifacts,
    options: &EngineOptions,
    draft: &DraftOptions,
    rendered: &[Rendered],
    records: &Path,
) -> Result<(Value, String), String> {
    let longest = rendered
        .iter()
        .map(|fixture| fixture.tokens.len())
        .max()
        .expect("a fixture set is nonempty");
    let mut engine = Engine::open(
        artifacts,
        options,
        &Load {
            drafter: true,
            proposals: draft.proposals,
            context_tokens: longest + 2 * usize::from(u8::MAX),
            decode_rows: 16,
            exported_logits_rows: 0,
            lookahead: false,
        },
    )?;
    let drafter = Drafter::of(engine.method)
        .ok_or_else(|| format!("{}: the drafter loaded no drafting method", artifacts.model))?;
    let mut file = std::fs::File::create(records)
        .map_err(|error| format!("{}: {error}", records.display()))?;
    let (mut proposed, mut accepted, mut matched, mut verifications) = (0, 0, 0, 0);
    for fixture in rendered {
        let tokens = engine::tokens(&fixture.tokens);
        for anchor in anchors(fixture, draft.anchors) {
            let outcome = draft_at(&mut engine, drafter, &tokens, anchor, draft.attribute)?;
            proposed += outcome.proposals.len();
            accepted += outcome.accepted;
            matched += outcome.matched;
            verifications += 1;
            let record = json!({
                "fixture_id": fixture.fixture_id,
                "anchor": anchor,
                "proposals": outcome.proposals.iter().map(|token| token.0).collect::<Vec<_>>(),
                "target": outcome.target.iter().map(|token| token.0).collect::<Vec<_>>(),
                "accepted": outcome.accepted,
                "matched_fixture": outcome.matched,
                "attribution": outcome.attribution,
            });
            serde_json::to_writer(&mut file, &record)
                .map_err(|error| error.to_string())
                .and_then(|()| file.write_all(b"\n").map_err(|error| error.to_string()))?;
        }
    }
    let device = engine.device().info().name.clone();
    Ok((
        json!({
            "proposed": proposed,
            "accepted": accepted,
            "tokens_per_verify": (accepted + verifications) as f64 / verifications.max(1) as f64,
            "matched_fixture": matched,
            "anchors": verifications,
        }),
        device,
    ))
}

/// Up to `count` anchors spread evenly over the fixture's scored positions,
/// each with a committed token before it and a next token after it.
fn anchors(fixture: &Rendered, count: usize) -> Vec<usize> {
    let positions = fixture
        .scored_spans
        .iter()
        .flat_map(|span| span.clone())
        .filter(|&position| position >= 1 && position + 1 < fixture.tokens.len())
        .collect::<Vec<_>>();
    if positions.is_empty() || count == 0 {
        return Vec::new();
    }
    let count = count.min(positions.len());
    (0..count)
        .map(|index| positions[index * positions.len() / count])
        .collect()
}

struct Drafted {
    proposals: Vec<TokenId>,
    /// The target's greedy tokens after the anchor and each proposal.
    target: Vec<TokenId>,
    accepted: usize,
    matched: usize,
    /// Device time by entry of the draft, the verification and a plain
    /// decode step after it, when attributed.
    attribution: Value,
}

/// Submit one drafter operation and reconcile `accepted` rows of it.
fn head(engine: &mut Engine, operation: Operation, accepted: usize) -> Result<Outcome, String> {
    let operations = [operation];
    let mut bindings = engine.bindings();
    let ReservedResources::Head(workspace, output, advances) = engine
        .domain
        .reserve(&mut bindings, &operations)
        .map_err(text)?
        .into_resources()
    else {
        return Err("a drafter operation runs on the head lane".into());
    };
    let flight = engine
        .domain
        .submit_head(bindings, &operations, workspace, output, advances)
        .map_err(|failure| failure.error().to_string())?;
    let (pending, bindings) = engine.domain.finish_head(flight).map_err(text)?;
    engine.bindings = Some(bindings);
    let mut outcome = None;
    for pending in pending {
        outcome = Some(pending.outcome().clone());
        engine
            .domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: accepted,
                },
            )
            .map_err(text)?;
    }
    outcome.ok_or_else(|| "a drafter operation returned no outcome".into())
}

/// How the loaded drafter enters history and drafts, as the served method
/// drives it.
#[derive(Clone, Copy)]
struct Drafter {
    form: DraftForm,
    /// The drafter enters every committed row into its own history; a head
    /// reading the target's history enters none and drafts at the anchor's
    /// row.
    keeps_history: bool,
    width: usize,
}

impl Drafter {
    fn of(method: ResolvedMethod) -> Option<Self> {
        match method {
            ResolvedMethod::Plain => None,
            ResolvedMethod::DFlash { proposals } => Some(Self {
                form: DraftForm::Block,
                keeps_history: true,
                width: usize::from(proposals),
            }),
            ResolvedMethod::Mtp {
                greedy_proposals,
                history,
                ..
            } => Some(Self {
                form: DraftForm::Chained,
                keeps_history: history == HeadHistory::Own,
                width: usize::from(greedy_proposals),
            }),
        }
    }
}

fn draft_at(
    engine: &mut Engine,
    drafter: Drafter,
    tokens: &[TokenId],
    anchor: usize,
    attribute: bool,
) -> Result<Drafted, String> {
    let mut sequence = engine.open_sequence()?;
    // The target takes the committed tokens; every row's features condition
    // the drafter.
    let mut conditioning = Vec::new();
    for chunk in tokens[..anchor].chunks(PREFILL_ROWS) {
        let start = sequence.position;
        let operation = Engine::forward(
            &sequence,
            WorkKind::Prefill,
            chunk.to_vec(),
            Demand::FEATURES,
        );
        let (_, outcomes) = engine.step(vec![operation], &mut [&mut sequence], None)?;
        let [Outcome::Forward { rows }] = outcomes.as_slice() else {
            return Err("a prefill returns forward rows".into());
        };
        let features = rows
            .iter()
            .find_map(|row| row.features.clone())
            .ok_or("the prefill returned no features")?;
        let rows = engine.domain.read(
            &FeatureSpan::new(features, 0, chunk.len()).map_err(|error| error.to_string())?,
        )?;
        conditioning.push((start, rows));
    }
    // Entry row p pairs the token after target row p with row p's feature;
    // rows before the anchor's are primed, as the served path primes them.
    for (start, rows) in conditioning.iter().filter(|_| drafter.keeps_history) {
        let end = (start + PREFILL_ROWS).min(anchor - 1);
        if end <= *start {
            continue;
        }
        let operation = Operation::Head {
            request: sequence.request,
            phase: HeadPhase::Generation,
            tokens: tokens[start + 1..end + 1].to_vec(),
            conditioning: rows
                .slice(0, end - start)
                .map_err(|error| error.to_string())?,
            position: *start,
            proposals: Vec::new(),
            form: drafter.form,
        };
        head(engine, operation, end - start)?;
    }
    let (last_start, last_rows) = conditioning
        .last()
        .expect("an anchor follows a committed token");
    let operation = Operation::Head {
        request: sequence.request,
        phase: HeadPhase::Generation,
        tokens: vec![tokens[anchor]],
        conditioning: last_rows
            .slice(anchor - 1 - last_start, 1)
            .map_err(|error| error.to_string())?,
        position: if drafter.keeps_history {
            anchor - 1
        } else {
            anchor
        },
        proposals: (0..drafter.width)
            .map(|proposal| Engine::greedy(anchor + 1 + proposal))
            .collect(),
        form: drafter.form,
    };
    let trace = attribute
        .then(|| engine.device().trace_submissions(TraceDetail::Submissions))
        .transpose()
        .map_err(|error| error.to_string())?;
    let phase = |start: f64| -> Result<Value, String> {
        let Some(trace) = &trace else {
            return Ok(Value::Null);
        };
        let end = host_seconds();
        let submissions = trace.collect().map_err(|error| error.to_string())?;
        let device = submissions
            .iter()
            .map(|submission| submission.device.1 - submission.device.0)
            .sum::<f64>();
        Ok(json!({
            "wall_ms": (end - start) * 1e3,
            "device_ms": device * 1e3,
            "submissions": submissions.len(),
        }))
    };
    let draft_start = host_seconds();
    let Outcome::Head { proposals } = head(engine, operation, 1)? else {
        return Err("a draft returns proposals".into());
    };
    let proposals = proposals
        .iter()
        .map(|selected| selected.token)
        .collect::<Vec<_>>();
    let draft_phase = phase(draft_start)?;
    // The target verifies the anchor and the proposals greedily.
    let mut verified = vec![tokens[anchor]];
    verified.extend_from_slice(&proposals);
    let operations = [Operation::Forward {
        request: sequence.request,
        kind: WorkKind::Verify,
        select: (0..verified.len())
            .map(|row| Engine::greedy(anchor + row + 1))
            .collect(),
        tokens: verified,
        position: anchor,
        conditioning: None,
        demand: Demand::SELECT,
        committed: 1,
        prime: None,
    }];
    let verify_start = host_seconds();
    let mut bindings = engine.bindings();
    let ReservedResources::Target(reservation) = engine
        .domain
        .reserve(&mut bindings, &operations)
        .map_err(text)?
        .into_resources()
    else {
        return Err("verification runs on the target lane".into());
    };
    let flight = engine
        .domain
        .submit_target(bindings, &operations, reservation)
        .map_err(|failure| failure.error().to_string())?;
    let (pending, bindings) = engine.domain.finish_target(flight).map_err(text)?;
    engine.bindings = Some(bindings);
    let mut target = Vec::new();
    let mut accepted = 0;
    for pending in pending {
        let Outcome::Forward { rows } = pending.outcome() else {
            return Err("verification returns forward rows".into());
        };
        target = rows
            .iter()
            .map(|row| row.selected.as_ref().map(|selected| selected.token))
            .collect::<Option<Vec<_>>>()
            .ok_or("a verification row has no selection")?;
        accepted = proposals
            .iter()
            .zip(&target)
            .take_while(|(proposal, sampled)| proposal == sampled)
            .count();
        engine
            .domain
            .reconcile(
                pending,
                PhysicalDecision {
                    accepted_rows: accepted + 1,
                },
            )
            .map_err(text)?;
    }
    let verify_phase = phase(verify_start)?;
    // One plain decode step from the accepted position, for comparison.
    let plain_phase = if attribute {
        sequence.position = anchor + accepted + 1;
        let token = target[accepted];
        let plain_start = host_seconds();
        let operation = Engine::forward(&sequence, WorkKind::Decode, vec![token], Demand::SELECT);
        engine.step(vec![operation], &mut [&mut sequence], None)?;
        phase(plain_start)?
    } else {
        Value::Null
    };
    let matched = proposals
        .iter()
        .zip(&tokens[anchor + 1..])
        .take_while(|(proposal, next)| proposal == next)
        .count();
    engine.close(sequence)?;
    Ok(Drafted {
        proposals,
        target,
        accepted,
        matched,
        attribution: json!({ "draft": draft_phase, "verify": verify_phase, "plain": plain_phase }),
    })
}
