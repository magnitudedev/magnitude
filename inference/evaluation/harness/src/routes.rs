//! A decode cell's recorded routes (`time --record-routes`), written as
//! NumPy arrays a replay reads: every decode step its load recorded of the
//! cell's last attempt, each row's choices and selection key of every
//! expert at every streamed layer, its front rows' verdicts, and the
//! sequence's prompt and selected tokens.

use magnitude_executor::{RecordedRoutes, RoutedStep};
use std::path::Path;

/// Write `routes`, recorded over one sequence's decode cell, into
/// `directory`: `steps.npy` i64 `[S, 3]` (each step's sequence, rows and
/// front bits; the cell's last attempt, in order), `rows.npy` i64 `[R, 2]`
/// (its steps' rows in order: each one's request and position),
/// `choices.npy` i32 `[R, L, K]` and `keys.npy` f32 `[R, L, E]` (each row's
/// choices, in its route's order, and key of every expert, at every
/// streamed layer),
/// `stood.npy` i64 `[V, 3]` (each judged front row of those steps: the
/// step's sequence, its row, whether it stood), `prompt.npy` and
/// `selected.npy` i64 (the prompt's tokens and each decode step's
/// selection). Returns the steps written.
pub fn write(directory: &Path, routes: RecordedRoutes, prompt: &[u32], selected: &[u32]) -> Result<usize, String> {
    let RecordedRoutes { selected: choices, steps, stood } = routes;
    // A discarded attempt's decode begins again from its histories: a step
    // whose exact row is at or before a kept step's supersedes it and every
    // one after.
    let exact = |step: &RoutedStep| {
        let row = (0..step.rows.len()).find(|&row| step.front >> row & 1 == 0).expect("a decode step runs an exact row");
        step.rows[row].1
    };
    let mut kept: Vec<RoutedStep> = Vec::with_capacity(steps.len());
    for step in steps {
        let superseded = kept.iter().position(|kept| exact(kept) >= exact(&step)).unwrap_or(kept.len());
        kept.truncate(superseded);
        kept.push(step);
    }
    let first = kept.first().ok_or("the cell's load recorded no decode step")?;
    let layers = first.choices.len();
    let experts = first.keys[0].len() / first.rows.len();
    let rows = kept.iter().map(|step| step.rows.len()).sum::<usize>();
    let (mut step_bytes, mut row_bytes, mut choice_bytes, mut key_bytes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for step in &kept {
        let count = step.rows.len();
        assert!(
            step.choices.len() == layers
                && step.choices.iter().all(|choices_of| choices_of.len() == count * choices)
                && step.keys.iter().all(|keys| keys.len() == count * experts),
            "step {} recorded {} layers of {count} rows, not {layers} of {choices} choices and {experts} experts' keys",
            step.sequence,
            step.choices.len()
        );
        for word in [step.sequence as i64, count as i64, i64::from(step.front)] {
            step_bytes.extend(word.to_le_bytes());
        }
        for &(request, position) in &step.rows {
            row_bytes.extend((request as i64).to_le_bytes());
            row_bytes.extend((position as i64).to_le_bytes());
        }
        for row in 0..count {
            for layer in 0..layers {
                for &expert in &step.choices[layer][row * choices..][..choices] {
                    choice_bytes.extend((expert as i32).to_le_bytes());
                }
                for &key in &step.keys[layer][row * experts..][..experts] {
                    key_bytes.extend(key.to_le_bytes());
                }
            }
        }
    }
    let sequences = kept.iter().map(|step| step.sequence).collect::<std::collections::HashSet<_>>();
    let verdicts = stood.into_iter().filter(|(sequence, ..)| sequences.contains(sequence)).collect::<Vec<_>>();
    let verdict_bytes = verdicts
        .iter()
        .flat_map(|&(sequence, row, stood)| [sequence as i64, row as i64, i64::from(stood)])
        .flat_map(i64::to_le_bytes)
        .collect();
    let tokens = |tokens: &[u32]| tokens.iter().flat_map(|&token| i64::from(token).to_le_bytes()).collect();
    std::fs::create_dir_all(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    npy(&directory.join("steps.npy"), "<i8", &[kept.len(), 3], step_bytes)?;
    npy(&directory.join("rows.npy"), "<i8", &[rows, 2], row_bytes)?;
    npy(&directory.join("choices.npy"), "<i4", &[rows, layers, choices], choice_bytes)?;
    npy(&directory.join("keys.npy"), "<f4", &[rows, layers, experts], key_bytes)?;
    npy(&directory.join("stood.npy"), "<i8", &[verdicts.len(), 3], verdict_bytes)?;
    npy(&directory.join("prompt.npy"), "<i8", &[prompt.len()], tokens(prompt))?;
    npy(&directory.join("selected.npy"), "<i8", &[selected.len()], tokens(selected))?;
    Ok(kept.len())
}

/// A NumPy array file (format 1.0) of `shape`, its elements as `descr`
/// names them (little-endian, 8 bytes or 4) in `data`.
fn npy(path: &Path, descr: &str, shape: &[usize], data: Vec<u8>) -> Result<(), String> {
    let width = if descr.ends_with('8') { 8 } else { 4 };
    assert_eq!(shape.iter().product::<usize>() * width, data.len(), "{}: shape {shape:?} of {} bytes", path.display(), data.len());
    let dims = shape.iter().map(|dim| format!("{dim},")).collect::<String>();
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}");
    // The magic, version and length (10 bytes) and the header, padded with
    // spaces to a newline that ends it at a multiple of 64.
    let padded = (10 + header.len() + 1).div_ceil(64) * 64;
    header.extend(std::iter::repeat_n(' ', padded - 10 - header.len() - 1));
    header.push('\n');
    let mut bytes = Vec::with_capacity(padded + data.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&u16::try_from(header.len()).expect("a short header").to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(data);
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}
