//! Front-row decode: every pass carries the rows of one sequence through
//! the same launches, the exact row of the pending token and a front row on
//! each guess of the sequence's chain, the tokens after it.
//!
//! - A **front row** never waits: at each streamed layer it computes with
//!   the experts its step holds, skips the rest, and the host reads what it
//!   skipped ahead (`resolve`, the front ring). Front row `d` runs on the
//!   chain's `d`-th guess, exact `d` passes later where every guess up to it
//!   stands, so its names are that pass's.
//! - The **exact row** computes with every expert its token routes to,
//!   waiting where one is absent, as single-row decode does. Its reads were
//!   named passes ahead by the front rows of the same token, so where the
//!   guesses held it finds them landed or landing.
//! - **Every pass emits, and only from its exact row.** A pass is
//!   [`Rows::Front`]: a forward's outcome holds no front row's output, and
//!   only the exact row's state is accepted, so the tokens are the exact
//!   rows' own.
//! - **The chain is the device's, and one.** The MTP head runs on the
//!   device right behind each decode pass, and behind a prompt's closing
//!   chunk, and at its tail makes the next pass's chain (`guess_chain`),
//!   depth by depth: the pass's front row's selection at each depth where
//!   the tokens under it are the chain's, the verified one and the
//!   shallower guesses it keeps ([`crate::GuessKind::Chained`]),
//!   then the head's selection on the deepest such row, one past them
//!   ([`crate::GuessKind::Head`]). So the head lengthens the
//!   chain by one per pass, and a pass the host builds can run one row
//!   deeper than the pass before. The chain is the one authority for the
//!   next front tokens: the pass queued behind reads them on the device,
//!   and the host reads them from the outcome
//!   ([`crate::Outcome::Forward`]'s `guesses`) for a pass it builds, so
//!   nothing on the host computes a guess.
//! - **The next pass is queued behind the one running.** Its tokens are
//!   read on the device from the chain the head launch behind the running
//!   pass made (`domain::queue`), so no pass waits for the host to read the
//!   chain before it runs.
//! - **The depth is the regime's** (`magnitude_cost_model::Regime::depth`):
//!   a pass runs as many front rows as the launch grants
//!   ([`TwoRowDecode::pass`]): a pass the host builds as deep as the chain
//!   holds, and one queued behind its predecessor as deep as it was queued
//!   (at most one past its predecessor's rows, as the head lengthens the
//!   chain by one per pass), idle past the chain.

use crate::operation::{Guess, Rows};
use crate::RequestId;
use std::collections::BTreeMap;
use magnitude_kernels::constants::VERIFY_ROWS;

/// The most entry rows a front head launch takes, behind a launch of
/// chained forwards (`crate::Operation::entry_rows`): the decode form's
/// rows (`VERIFY_ROWS`), so a launch of chained decode passes, whose every
/// row the head launch enters, stays in the decode row classes.
pub const CHAINED_ROWS: usize = VERIFY_ROWS as usize;

/// Of a decode launch's `requests`, each decoding one exact row, how many
/// run `depth` front rows beside it: as many as the launch's `rows`
/// (`crate::ResourceLimits::chained_rows`) hold beside every exact row;
/// none of depth 0.
pub fn fronted_requests(rows: usize, requests: usize, depth: usize) -> usize {
    match depth {
        0 => 0,
        depth => requests.min(rows.saturating_sub(requests) / depth),
    }
}

/// The front rows of a launch of chained decode passes
/// (`crate::ExecutorDomain::front_pass`), per request that runs any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontPass {
    /// The claim of the pass queued behind the last flight, whose requests
    /// are exactly the launch's: each at the front rows it was queued with,
    /// the chain's guesses then idle rows past them
    /// ([`FrontGrant::Exactly`]).
    Claim(BTreeMap<RequestId, usize>),
    /// A pass the host builds, every request's in the one `form` the regime
    /// prices least per emitted token (`magnitude_cost_model::Regime::pass`):
    /// each request that runs guessed rows at the most it runs (`depths`),
    /// every other its exact row alone, in that form all the same.
    Built { form: BuiltForm, depths: BTreeMap<RequestId, usize> },
}

/// The form of a pass the host builds (§7.1): front rows, which only name
/// experts and are recomputed exact by later passes, or verify rows, whose
/// outputs are accepted wherever their guesses stand. A verify pass waits
/// on its own verdict, so it is never queued: the form changes only at
/// passes the host builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltForm {
    Front,
    Verify,
}

impl BuiltForm {
    /// The grant a sequence's round runs a pass of this form by, of at most
    /// `depth` guessed rows.
    pub fn grant(self, depth: usize) -> FrontGrant {
        match self {
            Self::Front => FrontGrant::UpTo(depth),
            Self::Verify => FrontGrant::Verify(depth),
        }
    }
}

/// How many front rows a sequence's next pass runs ([`TwoRowDecode::pass`]):
/// up to a bound where the host builds it, or exactly the depth of the pass
/// it claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontGrant {
    /// As many front rows as the chain holds, at most this many: a pass
    /// the host builds for its own reasons runs no idle row.
    UpTo(usize),
    /// Exactly this many front rows, the chain's guesses and idle rows past
    /// them: the pass that claims one queued behind its predecessor, which
    /// ran this deep before the chain was known (`domain::queue`).
    Exactly(usize),
    /// A verify pass of as many acceptable rows as the chain holds, at most
    /// this many: the exact row, then a row on each of the chain's first
    /// guesses, whose outputs are accepted where the guesses stand.
    Verify(usize),
}

impl FrontGrant {
    /// The guessed rows it grants at most.
    pub fn depth(self) -> usize {
        match self {
            Self::UpTo(depth) | Self::Exactly(depth) | Self::Verify(depth) => depth,
        }
    }
}

/// A sequence's front-row decode: the chain the head launch behind its last
/// round made past the pending token, by depth (empty where it made none):
/// its consistent front rows' selections, then the head's guess one past
/// them. The pending token is always the sequence's own, so a pass's exact
/// row can name no other.
#[derive(Clone, Debug, Default)]
pub struct TwoRowDecode {
    next: Vec<Guess>,
}

impl TwoRowDecode {
    /// The next pass as `grant` holds it: its exact row, of the sequence's
    /// pending token, a front row on each of the chain's first guesses up
    /// to the grant's depth, and, of an exact grant, an idle row for each
    /// depth past the chain; one row of a grant of depth 0, or of one up to
    /// a depth where the chain holds none.
    pub fn pass(&self, grant: FrontGrant) -> Rows {
        let guessed = grant.depth().min(self.next.len());
        let guesses = self.next[..guessed].to_vec();
        match grant {
            FrontGrant::UpTo(_) => Rows::front(guesses, 0),
            FrontGrant::Exactly(depth) => Rows::front(guesses, depth - guessed),
            FrontGrant::Verify(_) => Rows::Exact { acceptable: guesses.into_iter().map(|guess| guess.token).collect() },
        }
    }

    /// A round of the sequence completed, guessing the chain `next`
    /// ([`crate::Outcome::Forward`]'s `guesses`): empty of a round that made
    /// no guess.
    pub fn guessed(&mut self, next: Vec<Guess>) {
        self.next = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation::{GuessKind, TokenId};

    /// A launch fronts as many of its requests as its rows hold beside
    /// every exact row at the depth, none at depth 0, and none from that
    /// many requests on.
    #[test]
    fn requests_front_within_the_launchs_rows() {
        let rows = CHAINED_ROWS;
        for depth in 0..rows {
            for requests in 1..=rows + 2 {
                let fronted = fronted_requests(rows, requests, depth);
                assert!(fronted <= requests, "{requests} requests at depth {depth}");
                assert!(fronted == 0 || requests + fronted * depth <= rows, "{requests} requests front {fronted} at depth {depth}");
                assert!(
                    depth == 0 || fronted == requests || requests + (fronted + 1) * depth > rows,
                    "{requests} requests leave a front pass unrun at depth {depth}"
                );
            }
        }
        assert_eq!(fronted_requests(rows, 1, rows - 1), 1, "one sequence runs every row past its exact one");
    }

    /// A sequence no round of which guessed runs one row up to any depth,
    /// and idle rows exactly to one; every pass after a round that guessed
    /// runs on the chain's guesses, as deep as the grant and the chain
    /// allow, whatever their kinds, an exact grant past the chain idle
    /// there; a round that guessed nothing leaves the next pass one row
    /// again up to any depth.
    #[test]
    fn a_pass_runs_on_the_last_rounds_chain_as_deep_as_granted() {
        let mut decode = TwoRowDecode::default();
        let exact = Rows::Exact { acceptable: Vec::new() };
        assert_eq!(decode.pass(FrontGrant::UpTo(3)), exact);
        assert_eq!(decode.pass(FrontGrant::UpTo(3)).tokens(TokenId(3)), [TokenId(3)]);
        assert_eq!(decode.pass(FrontGrant::Exactly(2)), Rows::front(Vec::new(), 2));
        assert_eq!(decode.pass(FrontGrant::Exactly(2)).tokens(TokenId(3)), [TokenId(3); 3]);
        for kind in GuessKind::ALL {
            let chain = vec![Guess { token: TokenId(8), kind }, Guess { token: TokenId(9), kind: GuessKind::Head }];
            decode.guessed(chain.clone());
            assert_eq!(decode.pass(FrontGrant::UpTo(0)), exact);
            assert_eq!(decode.pass(FrontGrant::Exactly(0)), exact);
            assert_eq!(decode.pass(FrontGrant::UpTo(1)), Rows::front(chain[..1].to_vec(), 0));
            assert_eq!(decode.pass(FrontGrant::UpTo(3)), Rows::front(chain.clone(), 0), "no deeper than the chain");
            let deeper = decode.pass(FrontGrant::Exactly(3));
            assert_eq!(deeper, Rows::front(chain.clone(), 1), "idle past the chain");
            assert_eq!((deeper.front_rows(), deeper.guesses().len()), (3, 2));
            assert_eq!(deeper.tokens(TokenId(5)), [TokenId(5), TokenId(8), TokenId(9), TokenId(5)]);
            assert_eq!(decode.pass(FrontGrant::UpTo(2)).tokens(TokenId(5)), [TokenId(5), TokenId(8), TokenId(9)]);
        }
        decode.guessed(Vec::new());
        assert_eq!(decode.pass(FrontGrant::UpTo(2)), exact);
    }
}
