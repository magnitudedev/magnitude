/// The model row widths with a partial last 256-value block: Gemma's 704
/// (expert down) and 2112 (dense down) and Nemotron's 2688.
const PARTIAL_BLOCK_WIDTHS: [usize; 3] = [704, 2112, 2688];

/// The `PARTIAL_BLOCK_WIDTHS` that whole blocks of `block` values fill: a
/// format's partial-block tests run at these, and the build script generates
/// them for a format where there is one.
fn partial_block_widths(block: usize) -> impl Iterator<Item = usize> {
    PARTIAL_BLOCK_WIDTHS.into_iter().filter(move |width| width % block == 0)
}
