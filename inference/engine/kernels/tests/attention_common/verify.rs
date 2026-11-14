// The verify block's rows, shared by the test files that check a decode row
// at every verify width; each defines the `Row` it fills (spans, fresh,
// destination, position) before including this.

/// The widest verify block: a verification pass's rows, drafting's
/// (`VERIFY_ROWS`).
const VERIFY_ROWS: usize = magnitude_kernels::constants::VERIFY_ROWS as usize;

/// Rows `rows` of a VERIFY_ROWS-row verify block after the history `spans`
/// (the last ending at the block's first row), launched alone: block row i
/// sees the history before the block, the last span continued over the
/// block's rows before `rows`, which the launch has in history, and the
/// launch's fresh rows up to its own, appending at the history's end + i. A
/// row's inputs are thus the same whatever block it launches in.
fn verify_rows(spans: &[(i32, i32)], rows: std::ops::Range<usize>) -> Vec<Row> {
    assert!(rows.end <= VERIFY_ROWS);
    let history = spans.last().unwrap().1;
    let mut seen = spans.to_vec();
    seen.last_mut().unwrap().1 += rows.start as i32;
    let first = rows.start;
    rows.map(|i| Row {
        spans: seen.clone(),
        fresh: (0, (i - first) as i32 + 1),
        destination: history + i as i32,
        position: history + i as i32,
    })
    .collect()
}

/// The sub-blocks each verify-width test launches beside the whole block:
/// every row alone (plain decode), then the last `width` rows at each width
/// between.
fn verify_blocks() -> Vec<std::ops::Range<usize>> {
    (0..VERIFY_ROWS)
        .map(|i| i..i + 1)
        .chain((2..VERIFY_ROWS).map(|width| VERIFY_ROWS - width..VERIFY_ROWS))
        .collect()
}
