//! `logits`: the engine's next-token distribution predicting every token of
//! every scored span, teacher-forced over the full rendered conversation
//! (contract "Logits output"). The row at position `p` is the distribution
//! the forward of `tokens[p]` returns, the prediction of `tokens[p + 1]`, so
//! a span `[start, end)` yields rows at positions `start - 1 ..= end - 2`.
//!
//! Each fixture's tokens before those positions are prefilled in chunks of
//! at most `PREFILL_ROWS` rows, packed across the fixtures decoded together;
//! each predicting position is then fed as a decode (or, with a verify width
//! above one, as rows of one verification forward).

use crate::artifacts::Artifacts;
use crate::engine::{self, Engine, EngineOptions, Load, PREFILL_ROWS, Sequence};
use crate::render::Rendered;
use magnitude_executor::{Demand, Operation, Outcome, TokenId, WorkKind};
use serde_json::json;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::Path;

pub struct LogitsOptions {
    /// Fixtures decoded together.
    pub batch: usize,
    /// Scored rows per teacher-forced forward; above one they run as one
    /// verification forward.
    pub verify_width: usize,
}

/// One fixture's walk through its tokens: its sequence, its tokens, its
/// scored spans and where its rows go in the output.
struct Walk<'a> {
    sequence: Sequence,
    rendered: &'a Rendered,
    tokens: Vec<TokenId>,
    /// The first output row of this fixture.
    base: usize,
}

/// The positions whose rows predict the tokens of `span`.
fn predicting(span: &Range<usize>) -> Range<usize> {
    span.start - 1..span.end - 1
}

impl Walk<'_> {
    /// The predicting positions holding the next position, or the next ones:
    /// `None` when no row remains.
    fn next_rows(&self) -> Option<Range<usize>> {
        self.rendered
            .scored_spans
            .iter()
            .map(predicting)
            .find(|rows| rows.end > self.sequence.position && !rows.is_empty())
    }

    /// Output row of position `position`.
    fn row(&self, position: usize) -> usize {
        let mut row = self.base;
        for rows in self.rendered.scored_spans.iter().map(predicting) {
            if rows.contains(&position) {
                return row + position - rows.start;
            }
            row += rows.len();
        }
        unreachable!("a predicting position lies before a span")
    }
}

/// Write `<stem>.f32` and `<stem>.json` for `rendered` (whose file digest is
/// `rendered_digest`). Returns the execution device's name.
pub fn run(
    artifacts: &Artifacts,
    options: &EngineOptions,
    logits: &LogitsOptions,
    rendered: &[Rendered],
    rendered_digest: &str,
    stem: &Path,
) -> Result<String, String> {
    let longest = rendered
        .iter()
        .map(|fixture| fixture.tokens.len())
        .max()
        .expect("a fixture set is nonempty");
    let batch = logits.batch.min(rendered.len());
    let rows_per_step = batch * logits.verify_width;
    let mut engine = Engine::open(
        artifacts,
        options,
        &Load {
            drafter: false,
            proposals: None,
            context_tokens: longest + logits.verify_width,
            decode_rows: rows_per_step,
            exported_logits_rows: rows_per_step.max(PREFILL_ROWS),
            lookahead: false,
        },
    )?;
    let vocabulary = engine.vocabulary;
    let mut rows = Vec::new();
    let mut bases = Vec::with_capacity(rendered.len());
    for fixture in rendered {
        bases.push(rows.len());
        for span in &fixture.scored_spans {
            if span.start == 0 || span.end > fixture.tokens.len() {
                return Err(format!(
                    "fixture {}: a scored span must follow its first token and lie within its tokens",
                    fixture.fixture_id
                ));
            }
            rows.extend(
                predicting(span).map(
                    |position| json!({ "fixture_id": fixture.fixture_id, "position": position }),
                ),
            );
        }
    }
    let data = stem.with_extension("f32");
    let mut file = File::create(&data).map_err(|error| format!("{}: {error}", data.display()))?;
    file.set_len((rows.len() * vocabulary * 4) as u64)
        .map_err(|error| format!("{}: {error}", data.display()))?;
    let indices = (0..rendered.len()).collect::<Vec<_>>();
    for (group, chunk) in indices.chunks(batch).enumerate() {
        let mut walks = Vec::with_capacity(chunk.len());
        for &index in chunk {
            walks.push(Walk {
                sequence: engine.open_sequence()?,
                rendered: &rendered[index],
                tokens: engine::tokens(&rendered[index].tokens),
                base: bases[index],
            });
        }
        score(
            &mut engine,
            &mut walks,
            logits.verify_width,
            vocabulary,
            &mut file,
        )?;
        for walk in walks {
            engine.close(walk.sequence)?;
        }
        eprintln!(
            "magnitude-harness: scored {} of {} fixtures (batch {group})",
            (group * batch + chunk.len()).min(rendered.len()),
            rendered.len()
        );
    }
    file.flush()
        .map_err(|error| format!("{}: {error}", data.display()))?;
    let metadata = json!({
        "producer": "harness",
        "model": artifacts.model,
        "rendered_digest": rendered_digest,
        "vocabulary": vocabulary,
        "rows": rows,
    });
    let path = stem.with_extension("json");
    std::fs::write(
        &path,
        serde_json::to_vec(&metadata).expect("logits metadata serializes"),
    )
    .map_err(|error| format!("writing {}: {error}", path.display()))?;
    Ok(engine.device().info().name.clone())
}

/// Advance every walk through its predicting positions: a packed prefill
/// step while any walk has tokens to take before its next ones, otherwise
/// one teacher-forced step of every walk among them.
fn score(
    engine: &mut Engine,
    walks: &mut [Walk<'_>],
    width: usize,
    vocabulary: usize,
    file: &mut File,
) -> Result<(), String> {
    loop {
        let prefilling = walks.iter().any(|walk| {
            walk.next_rows()
                .is_some_and(|scored| walk.sequence.position < scored.start)
        });
        let mut operations = Vec::new();
        let mut advancing = Vec::new();
        if prefilling {
            let mut budget = PREFILL_ROWS;
            for walk in walks.iter_mut() {
                let Some(scored) = walk.next_rows() else {
                    continue;
                };
                let rows = (scored.start - walk.sequence.position).min(PREFILL_ROWS);
                if rows == 0 || rows > budget {
                    continue;
                }
                budget -= rows;
                let position = walk.sequence.position;
                let piece = walk.tokens[position..position + rows].to_vec();
                operations.push(Engine::forward(
                    &walk.sequence,
                    WorkKind::Prefill,
                    piece,
                    Demand::NONE,
                ));
                advancing.push(walk);
            }
        } else {
            for walk in walks.iter_mut() {
                let Some(predicting) = walk.next_rows() else {
                    continue;
                };
                let position = walk.sequence.position;
                let rows = width.min(predicting.end - position);
                operations.push(scored(
                    &walk.sequence,
                    walk.tokens[position..position + rows].to_vec(),
                ));
                advancing.push(walk);
            }
        }
        if operations.is_empty() {
            return Ok(());
        }
        let starts = advancing
            .iter()
            .map(|walk| walk.sequence.position)
            .collect::<Vec<_>>();
        let mut sequences = advancing
            .iter_mut()
            .map(|walk| &mut walk.sequence)
            .collect::<Vec<_>>();
        let (_, outcomes) = engine.step(operations, &mut sequences, None)?;
        if prefilling {
            continue;
        }
        for ((walk, start), outcome) in advancing.iter().zip(starts).zip(outcomes) {
            let Outcome::Forward { rows } = outcome else {
                return Err("a scored step returned a non-forward outcome".into());
            };
            for (offset, result) in rows.iter().enumerate() {
                let logits = result
                    .logits
                    .as_ref()
                    .ok_or("a scored step returned no logits")?
                    .read_to_host()
                    .map_err(|error| error.to_string())?;
                if logits.len() != vocabulary {
                    return Err("a logits row's width differs from the vocabulary".into());
                }
                let bytes = logits
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                let row = walk.row(start + offset);
                file.seek(SeekFrom::Start((row * vocabulary * 4) as u64))
                    .and_then(|_| file.write_all(&bytes))
                    .map_err(|error| error.to_string())?;
            }
        }
    }
}

/// The teacher-forced step of `forced` rows: a single-row decode, or above
/// one row a verification forward (the drafting verify shapes).
fn scored(sequence: &Sequence, forced: Vec<TokenId>) -> Operation {
    if forced.len() == 1 {
        return Engine::forward(sequence, WorkKind::Decode, forced, Demand::LOGITS);
    }
    Operation::Forward {
        request: sequence.request,
        kind: WorkKind::Verify,
        position: sequence.position,
        conditioning: None,
        demand: Demand::LOGITS | Demand::SELECT,
        select: (0..forced.len())
            .map(|offset| Engine::greedy(sequence.position + offset))
            .collect(),
        committed: forced.len(),
        tokens: forced,
        prime: None,
    }
}
