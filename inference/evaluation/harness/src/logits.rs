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
//!
//! With a sample, only runs of a few predicting positions spread evenly over
//! each fixture's are scored, the tokens between them prefilled: decode steps
//! from many depths of every fixture at a fraction of the rows.

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
    /// Passes over the fixtures on the one load.
    pub passes: usize,
    /// Score only a sample of each fixture's predicting positions.
    pub sample: Option<Sample>,
}

/// `windows` runs of `steps` predicting positions, spread evenly over a
/// fixture's predicting positions.
#[derive(Clone, Copy)]
pub struct Sample {
    pub windows: usize,
    pub steps: usize,
}

/// One fixture's walk through its tokens: its sequence, its tokens, the
/// positions it scores and where its rows go in the output.
struct Walk {
    sequence: Sequence,
    tokens: Vec<TokenId>,
    /// The ascending, disjoint, nonempty runs of predicting positions scored.
    scored: Vec<Range<usize>>,
    /// The first output row of this fixture.
    base: usize,
}

/// The positions whose rows predict the tokens of `span`.
fn predicting(span: &Range<usize>) -> Range<usize> {
    span.start - 1..span.end - 1
}

/// The runs of predicting positions scored for scored spans `spans`: all of
/// them, or `sample`'s windows. The `i`th window starts at the `i`th of
/// `windows` evenly spaced indices into the positions and ends after `steps`
/// positions or at its span's end; with no more positions than the windows
/// would take, all are scored.
fn scored_rows(spans: &[Range<usize>], sample: Option<Sample>) -> Vec<Range<usize>> {
    let all = spans
        .iter()
        .map(predicting)
        .filter(|rows| !rows.is_empty())
        .collect::<Vec<_>>();
    let Some(Sample { windows, steps }) = sample else {
        return all;
    };
    let total = all.iter().map(ExactSizeIterator::len).sum::<usize>();
    if total <= windows * steps {
        return all;
    }
    // Consecutive starts lie at least `total / windows >= steps` positions apart.
    (0..windows)
        .map(|window| {
            let mut index = window * total / windows;
            let rows = all
                .iter()
                .find(|rows| {
                    let holds = index < rows.len();
                    if !holds {
                        index -= rows.len();
                    }
                    holds
                })
                .expect("an index below the total lies in a span");
            let start = rows.start + index;
            start..(start + steps).min(rows.end)
        })
        .collect()
}

impl Walk {
    /// The scored positions holding the next position, or the next ones:
    /// `None` when no row remains.
    fn next_rows(&self) -> Option<Range<usize>> {
        self.scored
            .iter()
            .find(|rows| rows.end > self.sequence.position)
            .cloned()
    }

    /// Output row of position `position`.
    fn row(&self, position: usize) -> usize {
        let mut row = self.base;
        for rows in &self.scored {
            if rows.contains(&position) {
                return row + position - rows.start;
            }
            row += rows.len();
        }
        unreachable!("a predicting position lies before a scored run")
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
            method: magnitude_engine::options::ModelMethod::Plain,
            proposals: None,
            context_tokens: longest + logits.verify_width,
            // Every fixture of a batch holds its sequence until it is scored.
            sequences: std::num::NonZeroUsize::new(batch)
                .expect("a positive batch of a nonempty fixture set"),
            decode_rows: rows_per_step,
            exported_logits_rows: rows_per_step.max(PREFILL_ROWS),
            lookahead: false,
        },
    )?;
    let vocabulary = engine.vocabulary;
    let mut rows = Vec::new();
    let mut scored = Vec::with_capacity(rendered.len());
    for fixture in rendered {
        if fixture
            .scored_spans
            .iter()
            .any(|span| span.start == 0 || span.end > fixture.tokens.len())
        {
            return Err(format!(
                "fixture {}: a scored span must follow its first token and lie within its tokens",
                fixture.fixture_id
            ));
        }
        let runs = scored_rows(&fixture.scored_spans, logits.sample);
        let base = rows.len();
        for window in &runs {
            rows.extend(
                window.clone().map(
                    |position| json!({ "fixture_id": fixture.fixture_id, "position": position }),
                ),
            );
        }
        scored.push((base, runs));
    }
    let indices = (0..rendered.len()).collect::<Vec<_>>();
    let scoring = std::time::Instant::now();
    // Each pass over the fixtures runs on the one load, in the state the last left its bank.
    for pass in 0..logits.passes {
        let data = match pass {
            0 => stem.with_extension("f32"),
            pass => stem.with_file_name(format!("{}-{pass}.f32", stem.file_name().expect("a logits stem names a file").to_string_lossy())),
        };
        let mut file = File::create(&data).map_err(|error| format!("{}: {error}", data.display()))?;
        file.set_len((rows.len() * vocabulary * 4) as u64)
            .map_err(|error| format!("{}: {error}", data.display()))?;
        for (group, chunk) in indices.chunks(batch).enumerate() {
            let mut walks = Vec::with_capacity(chunk.len());
            for &index in chunk {
                let sequence = engine.open_sequence()?;
                let tokens = engine::tokens(&rendered[index].tokens);
                // Every row is teacher-forced: the whole sequence is known,
                // as a prompt is when its request is admitted.
                engine.prefetch(&sequence, &tokens)?;
                let (base, runs) = &scored[index];
                walks.push(Walk {
                    sequence,
                    tokens,
                    scored: runs.clone(),
                    base: *base,
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
    }
    // What the rig's estimate reads: scored rows per second of the whole walk (its prefill between windows included).
    let throughput = scoring_record(rows.len() * logits.passes, scoring.elapsed().as_secs_f64());
    eprintln!("magnitude-harness: scored {throughput}");
    // A streamed load's decode misses: the fewest any streamed layer took
    // (none means some layer never read past its bank), all of them, and
    // those its CPU site served.
    let experts = engine.domain.expert_costs().map(|costs| {
        json!({
            "least_layer_misses": costs.layers.iter().map(|layer| layer.misses).min(),
            "misses": costs.layers.iter().map(|layer| layer.misses).sum::<u64>(),
            "sited": costs.layers.iter().map(|layer| layer.sited).sum::<u64>(),
        })
    });
    // What each stored table's steps read after they were queued: none of
    // host-known rows once their prefetch held them.
    let tables = engine
        .domain
        .stored_table_waits()?
        .into_iter()
        .map(|(block, waits)| {
            json!({
                "block": block,
                "known_steps": waits.known_steps,
                "known_waited": waits.known_waited,
                "known_rows": waits.known_rows,
                "selected_steps": waits.selected_steps,
                "selected_rows": waits.selected_rows,
            })
        })
        .collect::<Vec<_>>();
    if !tables.is_empty() {
        eprintln!("magnitude-harness: stored table reads after their steps were queued: {}", json!(tables));
    }
    let metadata = json!({
        "producer": "harness",
        "model": artifacts.model,
        "rendered_digest": rendered_digest,
        "vocabulary": vocabulary,
        "rows": rows,
        // The sample that chose `rows`; none scores every predicting position.
        "sample": logits.sample.map(|Sample { windows, steps }| json!({ "windows": windows, "steps": steps })),
        "experts": experts,
        "scoring": throughput,
        "stored_tables": tables,
    });
    let path = stem.with_extension("json");
    std::fs::write(
        &path,
        serde_json::to_vec(&metadata).expect("logits metadata serializes"),
    )
    .map_err(|error| format!("writing {}: {error}", path.display()))?;
    Ok(engine.device().info().name().to_owned())
}

/// The scoring walk's throughput: the rows it scored (every pass's), its seconds and their quotient.
fn scoring_record(rows: usize, seconds: f64) -> serde_json::Value {
    json!({ "rows": rows, "seconds": seconds, "rows_per_second": rows as f64 / seconds })
}

/// Advance every walk through its predicting positions: a packed prefill
/// step while any walk has tokens to take before its next ones, otherwise
/// one teacher-forced step of every walk among them.
fn score(
    engine: &mut Engine,
    walks: &mut [Walk],
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
                if rows == 0 || rows > budget || !engine.domain.claims(walk.sequence.request) {
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
        // A lone prompt chunk runs with the chunks after it as one prompt
        // group, as the service runs a prompt.
        if let (true, [walk]) = (prefilling, &advancing[..]) {
            let scored = walk.next_rows().expect("a prefilling walk scores rows next");
            engine.plan_prompt(&walk.sequence, &walk.tokens[walk.sequence.position..scored.start], false)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_spreads_disjoint_windows_within_the_spans() {
        // Predicting positions 9..29 and 39..59: forty in all.
        let spans = [10..30, 40..60];
        let sample = |windows, steps| scored_rows(&spans, Some(Sample { windows, steps }));
        assert_eq!(scored_rows(&spans, None), [9..29, 39..59]);
        assert_eq!(sample(4, 3), [9..12, 19..22, 39..42, 49..52]);
        // A window starting near a span's end ends with it.
        assert_eq!(sample(3, 8), [9..17, 22..29, 45..53]);
        // No more positions than the windows would take: all of them.
        assert_eq!(sample(5, 8), [9..29, 39..59]);
    }

    #[test]
    fn a_scoring_walk_reports_its_rows_per_second() {
        let record = scoring_record(128, 64.0);
        assert_eq!(record["rows"], 128);
        assert_eq!(record["rows_per_second"], 2.0);
    }
}
