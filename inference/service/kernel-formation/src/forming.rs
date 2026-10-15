//! Formation through the ledger: a formation whose key the ledger holds is
//! recalled; any other is formed in this process and its verdict recorded.

use crate::ledger::{Established, Key, Ledger};
use seismic::coverage::{Formation, Forming, ProgramSource, Verdict};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Ledgered {
    ledger: Ledger,
    /// Formations formed in this run.
    formed: AtomicUsize,
    /// Formations whose verdict the ledger already held.
    recalled: AtomicUsize,
}

impl Ledgered {
    pub fn new(ledger: Ledger) -> Self {
        Self {
            ledger,
            formed: AtomicUsize::new(0),
            recalled: AtomicUsize::new(0),
        }
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// How many formations this run formed and recalled so far.
    pub fn activity(&self) -> String {
        format!(
            "{} formed, {} known from the ledger",
            self.formed.load(Ordering::Relaxed),
            self.recalled.load(Ordering::Relaxed),
        )
    }
}

impl Forming for Ledgered {
    fn form(&self, formation: Formation<'_>) -> Verdict {
        let key = formation_key(formation.namespace, &formation.material, formation.program);
        if let Some(established) = self.ledger.verdict(&key) {
            self.recalled.fetch_add(1, Ordering::Relaxed);
            return match established {
                Established::Formed => Verdict::Formed,
                Established::Rejected => Verdict::Rejected,
            };
        }
        let verdict = seismic::coverage::form(formation.target, formation.program);
        self.formed.fetch_add(1, Ordering::Relaxed);
        match verdict {
            Verdict::Formed => self.ledger.record(&key, Established::Formed),
            Verdict::Rejected => self.ledger.record(&key, Established::Rejected),
            Verdict::Failed(_) => {}
        }
        verdict
    }
}

/// The ledger key of forming `program` with the toolchain `namespace`
/// identified by `material`: everything the formation reads.
fn formation_key(namespace: &str, material: &str, program: &ProgramSource) -> Key {
    let entries = program
        .entries
        .iter()
        .map(|entry| {
            format!(
                "{}@{:?}#{:?}",
                entry.symbol, entry.group_size, entry.constants
            )
        })
        .collect::<Vec<_>>()
        .join("\t");
    Key::of(&[
        b"formation",
        namespace.as_bytes(),
        material.as_bytes(),
        program.text.as_bytes(),
        entries.as_bytes(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::coverage::ProgramEntry;

    #[test]
    fn formation_keys_cover_every_input() {
        let program = ProgramSource {
            text: "kernel".into(),
            entries: vec![ProgramEntry::named("k")],
        };
        let key = formation_key("cuda", "m", &program);
        assert_eq!(key, formation_key("cuda", "m", &program.clone()));
        assert_ne!(key, formation_key("metal", "m", &program));
        assert_ne!(key, formation_key("cuda", "n", &program));
        let mut changed = program.clone();
        changed.text.push(' ');
        assert_ne!(key, formation_key("cuda", "m", &changed));
        let mut changed = program.clone();
        changed.entries[0].symbol = "j".into();
        assert_ne!(key, formation_key("cuda", "m", &changed));
        let mut changed = program.clone();
        changed.entries[0].group_size = Some([1, 1, 1]);
        assert_ne!(key, formation_key("cuda", "m", &changed));
        let mut changed = program;
        changed.entries[0].constants = vec![1];
        assert_ne!(key, formation_key("cuda", "m", &changed));
    }
}
