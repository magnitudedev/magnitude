// Two-row decode's chain on CPU (`guess_chain` in guess.seismic; the GPU
// backends' `two_row/guess.hpp`).

use crate::constants::{GUESS_CHAINED, GUESS_HEAD};

/// A chaining row's chain of next guesses (guess.seismic's `guess_chain`):
/// from its pending token, its exact selection (`exact`, its status), its
/// `fronts` front rows' guesses and selections (`front(j)`, front row j's
/// (token, status) pairs, 1-based), and the head's selection on entry row
/// c (`drafted(c)`, its (token, status) pair), each depth's guess and kind,
/// the consistent front rows' selections then the head's on the deepest of
/// them, handed to `emit(depth, guess, kind)` in depth order. An unverified
/// exact selection chains nothing.
#[inline(always)]
pub fn chain(
    pending: [i32; 2],
    exact: [i32; 2],
    fronts: usize,
    front: impl Fn(usize) -> ([i32; 2], [i32; 2]),
    drafted: impl Fn(usize) -> [i32; 2],
    mut emit: impl FnMut(usize, i32, i32),
) {
    if exact[1] != 0 || pending[1] != 0 {
        return;
    }
    let mut last = exact[0];
    let mut deepest = 0;
    for depth in 1..=fronts {
        let (guessed, selected) = front(depth);
        if guessed != [last, 0] || selected[1] != 0 {
            break;
        }
        last = selected[0];
        emit(depth, last, GUESS_CHAINED as i32);
        deepest = depth;
    }
    if let [guess, 0] = drafted(deepest) {
        emit(deepest + 1, guess, GUESS_HEAD as i32);
    }
}
