//! Two-row decode's chain (`guess_chain` in guess.seismic) on every backend
//! the host has: the guesses each slot's front rows and the head make, and
//! the token rows and kinds of the pass queued behind.

mod declared;

use magnitude_kernels::{constants, guess_chain};

/// The host's GPU (Metal on macOS, else Vulkan when present), then the CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let gpu = if cfg!(target_os = "macos") {
        Some(catalog.open_backend(seismic::BackendName::Metal).unwrap())
    } else {
        catalog.open_backend(seismic::BackendName::Vulkan).ok()
    };
    gpu.into_iter().chain(std::iter::once(catalog.open_backend(seismic::BackendName::Cpu).unwrap())).collect()
}

fn i32_tensor(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    seismic::Tensor::from_host(&seismic::Charges::tally(device), seismic::Element::i32(), shape, &bytes).unwrap()
}

fn read_i32(tensor: &seismic::Tensor) -> Vec<i32> {
    tensor.read_to_host().unwrap().chunks_exact(4).map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap())).collect()
}

const CHAINED: i32 = constants::GUESS_CHAINED as i32;
const HEAD: i32 = constants::GUESS_HEAD as i32;

/// A verified token or selection.
fn verified(value: i32) -> [i32; 2] {
    [value, 0]
}

/// The chaining words of a slot's exact row at token row `row` with `front`
/// front rows behind it.
fn chains(row: i32, front: i32) -> [i32; 2] {
    [row, front]
}

/// The chaining words of a row that does not chain (a front row's).
const FRONT: [i32; 2] = [-1, 0];

/// The pass's outputs: its guesses, and the queued pass's token rows and
/// kinds.
struct Chain {
    guesses: Vec<i32>,
    next_selected: Vec<i32>,
    next_guesses: Vec<i32>,
}

/// Chain the pass of token rows `tokens`, and of selected rows `chaining`,
/// `selected` and `drafted` (one each per selected row), for the queued pass
/// of `continuing` rows. Every output starts as words no clear leaves.
fn chain(
    device: &seismic::Device,
    tokens: &[[i32; 2]],
    chaining: &[[i32; 2]],
    selected: &[[i32; 2]],
    drafted: &[[i32; 2]],
    continuing: &[[i32; 2]],
) -> Chain {
    let (rows, continued) = (selected.len() as u64, continuing.len() as u64);
    assert_eq!((chaining.len(), drafted.len()), (selected.len(), selected.len()), "one chaining and drafted row per selected row");
    let stale = |rows: u64| i32_tensor(device, &[rows, 2], &vec![-7; 2 * rows as usize]);
    let (mut guesses, mut next_selected, mut next_guesses) = (stale(rows), stale(continued), stale(continued));
    declared::prepared(device, &guess_chain::native_entry(), &seismic::NativeSpecialization::new())
        .unwrap()
        .call(&seismic::Charges::tally(device), guess_chain::Args {
            tokens: &i32_tensor(device, &[tokens.len() as u64, 2], tokens.as_flattened()),
            chaining: &i32_tensor(device, &[rows, constants::CHAINING_WORDS], chaining.as_flattened()),
            selected: &i32_tensor(device, &[rows, 2], selected.as_flattened()),
            drafted: &i32_tensor(device, &[rows, 2], drafted.as_flattened()),
            continuing: &i32_tensor(device, &[continued, constants::CONTINUING_WORDS], continuing.as_flattened()),
            guesses: &mut guesses,
            next_selected: &mut next_selected,
            next_guesses: &mut next_guesses,
        })
        .unwrap();
    Chain { guesses: read_i32(&guesses), next_selected: read_i32(&next_selected), next_guesses: read_i32(&next_guesses) }
}

/// A slot's consistent front rows chain their selections, then the head's
/// selection on the deepest of them follows: k chained and the head's where
/// every front row stands; j - 1 chained and the head's on row j - 1 where
/// front row j falls (its guess not the selection under it, or its own
/// selection failed); nothing of an unverified exact selection; the head's
/// alone of the prompt's closing chunk (no front rows); and the chain stops
/// where the head selected nothing. Each guess lands at guesses row
/// s + depth - 1, every other row cleared.
#[test]
fn a_pass_chains_its_consistent_front_rows_then_the_heads_selection() {
    for device in devices() {
        let backend = device.backend();
        // (token row, chaining, selected, drafted) per row; token rows are
        // 1:1 with the selected rows.
        type Row = ([i32; 2], [i32; 2], [i32; 2], [i32; 2]);
        let rows: [Row; 16] = [
            // Rows 0-2: both front rows stand (6 = y, 7 = f_1).
            (verified(5), chains(0, 2), verified(6), verified(20)),
            (verified(6), FRONT, verified(7), verified(21)),
            (verified(7), FRONT, verified(8), verified(22)),
            // Rows 3-5: front row 2 falls (9 guessed for f_1 = 7).
            (verified(5), chains(3, 2), verified(6), verified(30)),
            (verified(6), FRONT, verified(7), verified(31)),
            (verified(9), FRONT, verified(8), verified(32)),
            // Rows 6-7: front row 1 falls (4 guessed for y = 6).
            (verified(5), chains(6, 1), verified(6), verified(40)),
            (verified(4), FRONT, verified(9), verified(41)),
            // Rows 8-9: y unverified.
            (verified(5), chains(8, 1), [6, 1], verified(45)),
            (verified(6), FRONT, verified(7), verified(46)),
            // Row 10: the closing chunk's one selecting row.
            (verified(5), chains(10, 0), verified(3), verified(50)),
            // Rows 11-12: front row 1 stands, the head selected nothing on it.
            (verified(5), chains(11, 1), verified(6), verified(55)),
            (verified(6), FRONT, verified(7), [0, 1]),
            // Rows 13-14: front row 1's own selection failed.
            (verified(5), chains(13, 1), verified(6), verified(60)),
            (verified(6), FRONT, [7, 2], verified(61)),
            // Row 15: a row that chains nothing.
            (verified(5), FRONT, verified(6), verified(65)),
        ];
        let column = |pick: fn(&Row) -> [i32; 2]| rows.iter().map(pick).collect::<Vec<_>>();
        let pass = chain(&device, &column(|row| row.0), &column(|row| row.1), &column(|row| row.2), &column(|row| row.3), &[]);
        #[rustfmt::skip]
        let expected = [
            7, CHAINED, 8, CHAINED, 22, HEAD,
            7, CHAINED, 31, HEAD, 0, 0,
            40, HEAD, 0, 0,
            0, 0, 0, 0,
            50, HEAD,
            7, CHAINED, 0, 0,
            60, HEAD, 0, 0,
            0, 0,
        ];
        assert_eq!(pass.guesses, expected, "{backend:?}: guesses");
    }
}

/// The queued pass's rows continue the chain: its exact row runs on the
/// slot's selection, status and all; its front row j on n_j where the chain
/// holds it, one deeper than the pass's own front rows included, else on
/// y as an idle row of no guess; and front row r's kind lands at
/// next_guesses row r - 1, the row `routed_step` reads it from. A row that
/// continues no slot stays cleared.
#[test]
fn a_queued_pass_runs_on_the_chain_and_idles_past_it() {
    for device in devices() {
        let backend = device.backend();
        // Slot 0 (rows 0-1): front row 1 stands, chain 7, then the head's 21
        // (two deep, one past the pass's front row). Slot 2 (rows 2-3):
        // front row 1 falls, chain the head's 30 alone. Slot 4: y
        // unverified, no chain.
        let tokens = [verified(5), verified(6), verified(5), verified(4), verified(5)];
        let chaining = [chains(0, 1), FRONT, chains(2, 1), FRONT, chains(4, 0)];
        let selected = [verified(6), verified(7), verified(8), verified(9), [6, 1]];
        let drafted = [verified(20), verified(21), verified(30), verified(31), verified(40)];
        let continuing = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1], [2, 2], [4, 0], [4, 1], [-1, 0]];
        let pass = chain(&device, &tokens, &chaining, &selected, &drafted, &continuing);
        assert_eq!(pass.guesses, [7, CHAINED, 21, HEAD, 30, HEAD, 0, 0, 0, 0], "{backend:?}: guesses");
        #[rustfmt::skip]
        let next_selected = [
            6, 0, 7, 0, 21, 0,
            8, 0, 30, 0, 8, 0,
            6, 1, 6, 0,
            0, 0,
        ];
        assert_eq!(pass.next_selected, next_selected, "{backend:?}: the queued pass's token rows");
        #[rustfmt::skip]
        let next_guesses = [
            7, CHAINED, 21, HEAD, 0, 0,
            30, HEAD, 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0,
        ];
        assert_eq!(pass.next_guesses, next_guesses, "{backend:?}: the queued pass's front rows' kinds, each at row r - 1");
    }
}
