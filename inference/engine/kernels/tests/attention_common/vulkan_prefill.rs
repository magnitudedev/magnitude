// Vulkan `attention_prefill`'s configurations, shared by the attention entry
// tests (included textually). Every tuning parameter is bound: the vector
// form's (SPAN, PARTS, SIMDS) at their defaults in the matrix form, the matrix
// form's (ROWS, SPLIT_GROUPS) at theirs in the vector form. SLICES is left to
// its declared default, which the declaration holds it to in both forms.

/// The vector form (MATRIX = 0) over SPAN-key partitions, PARTS of them, of
/// SIMDS subgroups.
fn vulkan_prefill_vector(span: u64, parts: u64, simds: u64) -> Vec<(&'static str, u64)> {
    vulkan_prefill_params(0, 64, 256, span, parts, simds)
}

/// The matrix form (MATRIX = 1) over tiles of ROWS matrix rows, split over
/// SPLIT_GROUPS.
fn vulkan_prefill_matrix(rows: u64, split_groups: u64) -> Vec<(&'static str, u64)> {
    vulkan_prefill_params(1, rows, split_groups, 32, 16, 4)
}

fn vulkan_prefill_params(
    matrix: u64,
    rows: u64,
    split_groups: u64,
    span: u64,
    parts: u64,
    simds: u64,
) -> Vec<(&'static str, u64)> {
    vec![
        ("MATRIX", matrix),
        ("ROWS", rows),
        ("SPLIT_GROUPS", split_groups),
        ("SPAN", span),
        ("PARTS", parts),
        ("SIMDS", simds),
    ]
}
