// guess_chain on CPU (contract and portable body in guess.seismic): one work
// item chains each chaining row (`lib::two_row::guess`), then gives each
// continuing row its token row and kind, every output row cleared first,
// since a continuing row's kind lands on the row before it.

use crate::constants::{CHAINING_FRONT, CHAINING_ROW, CONTINUING_DEPTH, CONTINUING_SLOT};
use lib::two_row::guess;

#[inline(always)]
fn guess_chain<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let (tokens, chaining, selected, drafted) = (cx.arg_tokens(), cx.arg_chaining(), cx.arg_selected(), cx.arg_drafted());
    let continuing = cx.arg_continuing();
    let (guesses, next_selected, next_guesses) = (cx.arg_guesses(), cx.arg_next_selected(), cx.arg_next_guesses());
    let pair = |row: usize, tensor: &dyn Fn([usize; 2]) -> i32| [tensor([row, 0]), tensor([row, 1])];
    let (rows, token_rows, continued) = (cx.dim_s() as usize, cx.dim_m() as usize, cx.dim_n() as usize);
    for s in 0..rows {
        // SAFETY: the single work item writes every element.
        unsafe {
            guesses.set([s, 0], 0);
            guesses.set([s, 1], 0);
        }
    }
    for r in 0..continued {
        // SAFETY: the single work item writes every element.
        unsafe {
            next_selected.set([r, 0], 0);
            next_selected.set([r, 1], 0);
            next_guesses.set([r, 0], 0);
            next_guesses.set([r, 1], 0);
        }
    }
    let token = |index| tokens.get(index);
    let selection = |index| selected.get(index);
    let head = |index| drafted.get(index);
    for s in 0..rows {
        let t = chaining.get([s, CHAINING_ROW as usize]);
        if t < 0 || t as usize >= token_rows {
            continue;
        }
        let t = t as usize;
        let fronts = (chaining.get([s, CHAINING_FRONT as usize]).max(0) as usize).min(rows - 1 - s).min(token_rows - 1 - t);
        guess::chain(
            pair(t, &token),
            pair(s, &selection),
            fronts,
            |j| (pair(t + j, &token), pair(s + j, &selection)),
            |c| pair(s + c, &head),
            |depth, guess, kind| {
                // SAFETY: the single work item writes every element.
                unsafe {
                    guesses.set([s + depth - 1, 0], guess);
                    guesses.set([s + depth - 1, 1], kind);
                }
            },
        );
    }
    for r in 0..continued {
        let (s, depth) = (continuing.get([r, CONTINUING_SLOT as usize]), continuing.get([r, CONTINUING_DEPTH as usize]));
        if s < 0 || s as usize >= rows {
            continue;
        }
        let s = s as usize;
        if depth == 0 {
            // SAFETY: the single work item writes every element.
            unsafe {
                next_selected.set([r, 0], selected.get([s, 0]));
                next_selected.set([r, 1], selected.get([s, 1]));
            }
            continue;
        }
        let held = chaining.get([s, CHAINING_ROW as usize]) >= 0
            && depth >= 1
            && depth <= chaining.get([s, CHAINING_FRONT as usize]) + 1
            && s + (depth as usize) - 1 < rows;
        let guess = held
            .then(|| [guesses.get([s + depth as usize - 1, 0]), guesses.get([s + depth as usize - 1, 1])])
            .filter(|[_, kind]| *kind != 0);
        // SAFETY: the single work item writes every element.
        unsafe {
            next_selected.set([r, 0], guess.map_or(selected.get([s, 0]), |[guess, _]| guess));
            next_selected.set([r, 1], 0);
            if r >= 1 {
                let [guess, kind] = guess.unwrap_or([0, 0]);
                next_guesses.set([r - 1, 0], guess);
                next_guesses.set([r - 1, 1], kind);
            }
        }
    }
}
