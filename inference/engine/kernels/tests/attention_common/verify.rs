// The verify block's rows, shared by the test files that check a decode row
// at every verify width; each defines the `Row` it fills (spans, fresh,
// destination, position) before including this.

/// The widest verify block: a verification pass's rows.
const VERIFY_ROWS: usize = 16;

/// The rows of a verify block of `width` rows ending where a VERIFY_ROWS-row
/// block after the history `spans` (the last ending at the block's first
/// row) ends: block row j, the widest block's row i = VERIFY_ROWS - width +
/// j, sees the history before the block, the last span continued over the
/// block's earlier rows, which narrower blocks have in history, and the
/// block's fresh rows 0..=j, appending at the history's end + i.
fn verify_rows(spans: &[(i32, i32)], width: usize) -> Vec<Row> {
    let first = VERIFY_ROWS - width;
    let history = spans.last().unwrap().1;
    let mut seen = spans.to_vec();
    seen.last_mut().unwrap().1 += first as i32;
    (0..width)
        .map(|j| {
            let i = (first + j) as i32;
            Row {
                spans: seen.clone(),
                fresh: (0, j as i32 + 1),
                destination: history + i,
                position: history + i,
            }
        })
        .collect()
}
