//! The bank's derived target on a reservation-backed bank, read from its
//! device's ledger at each boundary.

use super::Followed;
use crate::domain::fixtures::{bank_backends, banked};
use crate::domain::{BankResized, BankStep};
use crate::experts::derived::bank_room;
use crate::memory::ledger::{Available, Ledger, Need};
use std::time::{Duration, Instant};

/// The store's capacity.
const TOTAL: u64 = 1 << 40;

/// What other programs hold, unchanged at every reading: their rise
/// between readings is no part of the guarantee here.
const OTHERS: u64 = 1 << 38;

fn wake() -> crate::CompletionWake {
    crate::CompletionWake::new(|| {})
}

/// A reading at `at` showing free what neither other programs nor the
/// process's allocations hold.
fn read(ledger: &Ledger, at: Instant) {
    let available = TOTAL - OTHERS - ledger.snapshot().allocated();
    ledger.read(|| Ok(Available::new(TOTAL, available, None, at).expect("a reading")));
}

/// Item 9, the laptop's failure, on the ledger: a bank at its full target,
/// a prompt's claim taking room so the target falls and the bank shrinks,
/// then the claim freed. At the next boundary the target is the full bank
/// again and the bank grows back to it. On main a settle released the bank
/// for the prompt and nothing grew it back.
#[test]
fn the_bank_regrows_to_its_target_once_a_prompts_claim_is_freed() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = domain.family.bank.as_ref().unwrap().tensor.byte_len() / 4;
        let ledger = domain.resources.ledgers().device().clone();
        let start = Instant::now();
        let at = |reading: u32| start + Duration::from_millis(100) * reading;
        // Two readings of the same others: no rise between them, so the
        // guarantee holds only the load's own growth.
        read(&ledger, at(1));
        read(&ledger, at(2));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: the bank holds its full target");
        // The prompt's claim leaves the bank's cache a slot and a half of
        // room above its floor: its target falls to two slots.
        let room = bank_room(&ledger.snapshot(), domain.in_process_growth());
        let prompt = ledger.claim(Need::Measurement { bytes: room - row * 3 / 2 }).expect("the prompt's claim fits");
        read(&ledger, at(3));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        let shrunk = loop {
            if let Some((resized, _)) = domain.advance_expert_bank(wake()).unwrap() {
                break resized;
            }
        };
        assert!(matches!(shrunk, BankResized::Shrunk { from: 4, to: 2, .. }), "{backend:?}: {shrunk:?}");
        // The prompt ends: its claim is freed, and the next boundary grows
        // the bank back to its full target.
        drop(prompt);
        read(&ledger, at(4));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Grew(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        let grown = loop {
            if let Some((resized, _)) = domain.advance_expert_bank(wake()).unwrap() {
                break resized;
            }
        };
        assert!(matches!(grown, BankResized::Grown { from: 2, to: 4, .. }), "{backend:?}: {grown:?}");
        read(&ledger, at(5));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: the full bank again");
    }
}
