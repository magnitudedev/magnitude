//! What formation has established, kept between runs.
//!
//! A verdict is a fact about its inputs: the program and the toolchain
//! identity its key addresses. Only the toolchain's own verdicts are kept,
//! formed or rejected by a `static_assert`; a compile error is never kept. A
//! completion records a coverage run that established everything, by a key
//! derived from every input of its result.
//!
//! Records are appended as they are established, so a run that dies keeps
//! everything it finished. A line that does not parse is ignored and an
//! absent file is an empty ledger: the ledger only skips work whose outcome
//! is already known, and never decides an outcome.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A content address: the SHA-256 of length-prefixed parts, in order.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key(String);

impl Key {
    pub fn of(parts: &[&[u8]]) -> Self {
        let mut digest = Sha256::new();
        for part in parts {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
        Self(
            digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        )
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the toolchain established for one formation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Established {
    Formed,
    /// A `static_assert` fired: the kernel declines the configuration.
    Rejected,
}

#[derive(Default)]
struct Records {
    verdicts: BTreeMap<String, Established>,
    completions: BTreeMap<String, String>,
}

/// The verdicts and completions of one ledger file.
pub struct Ledger {
    path: PathBuf,
    records: Mutex<Records>,
    appending: Mutex<std::fs::File>,
}

impl Ledger {
    /// The ledger at `path`, empty if there is none, opened for appending.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let records = match std::fs::read_to_string(path) {
            Ok(text) => parse(&text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Records::default(),
            Err(error) => return Err(error),
        };
        if let Some(directory) = path.parent() {
            std::fs::create_dir_all(directory)?;
        }
        let appending = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            path: path.to_owned(),
            records: Mutex::new(records),
            appending: Mutex::new(appending),
        })
    }

    pub fn verdict(&self, key: &Key) -> Option<Established> {
        self.lock().verdicts.get(&key.0).copied()
    }

    pub fn record(&self, key: &Key, verdict: Established) {
        self.lock().verdicts.insert(key.0.clone(), verdict);
        self.append(&verdict_line(&key.0, verdict));
    }

    /// The summary of the completed run keyed `run`, if there is one.
    pub fn completion(&self, run: &Key) -> Option<String> {
        self.lock().completions.get(&run.0).cloned()
    }

    /// Record that the run keyed `run` established everything, with its
    /// `summary` kept on one line.
    pub fn complete(&self, run: &Key, summary: &str) {
        let summary = summary.trim().replace('\n', "; ");
        self.lock()
            .completions
            .insert(run.0.clone(), summary.clone());
        self.append(&format!("c {run} {summary}\n"));
    }

    /// Rewrite the file as its records, sorted, each once.
    pub fn compact(&self) -> std::io::Result<()> {
        let text = {
            let records = self.lock();
            let mut text = String::new();
            for (key, verdict) in &records.verdicts {
                text.push_str(&verdict_line(key, *verdict));
            }
            for (run, summary) in &records.completions {
                text.push_str(&format!("c {run} {summary}\n"));
            }
            text
        };
        let mut appending = self
            .appending
            .lock()
            .expect("no thread panics holding the ledger file");
        let staged = self.path.with_extension("compacting");
        std::fs::write(&staged, text)?;
        std::fs::rename(&staged, &self.path)?;
        *appending = std::fs::OpenOptions::new().append(true).open(&self.path)?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Records> {
        self.records
            .lock()
            .expect("no thread panics holding the ledger")
    }

    /// Append one line. A failed write loses only that record: a later run
    /// establishes it again.
    fn append(&self, line: &str) {
        let mut file = self
            .appending
            .lock()
            .expect("no thread panics holding the ledger file");
        if let Err(error) = file.write_all(line.as_bytes()) {
            eprintln!("formation ledger {}: {error}", self.path.display());
        }
    }
}

fn verdict_line(key: &str, verdict: Established) -> String {
    let established = match verdict {
        Established::Formed => "formed",
        Established::Rejected => "rejected",
    };
    format!("v {key} {established}\n")
}

fn parse(text: &str) -> Records {
    let mut records = Records::default();
    for line in text.lines() {
        let mut fields = line.splitn(3, ' ');
        match (fields.next(), fields.next(), fields.next()) {
            (Some("v"), Some(key), Some(established)) => {
                let established = match established {
                    "formed" => Established::Formed,
                    "rejected" => Established::Rejected,
                    _ => continue,
                };
                records.verdicts.insert(key.to_owned(), established);
            }
            (Some("c"), Some(run), Some(summary)) => {
                records
                    .completions
                    .insert(run.to_owned(), summary.to_owned());
            }
            _ => {}
        }
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("kernel-formation-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&directory).ok();
        directory.join("cuda.ledger")
    }

    #[test]
    fn records_survive_reopening_and_compaction() {
        let path = path("reopen");
        let (a, b, run) = (Key::of(&[b"a"]), Key::of(&[b"b"]), Key::of(&[b"run"]));
        {
            let ledger = Ledger::open(&path).unwrap();
            assert_eq!(ledger.verdict(&a), None);
            ledger.record(&a, Established::Formed);
            ledger.record(&b, Established::Rejected);
            ledger.record(&a, Established::Formed);
            ledger.complete(&run, "formed 2 programs\nformed 1 request\n");
        }
        let ledger = Ledger::open(&path).unwrap();
        assert_eq!(ledger.verdict(&a), Some(Established::Formed));
        assert_eq!(ledger.verdict(&b), Some(Established::Rejected));
        assert_eq!(
            ledger.completion(&run).as_deref(),
            Some("formed 2 programs; formed 1 request")
        );
        ledger.compact().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 3);
        // Appending continues into the compacted file.
        let c = Key::of(&[b"c"]);
        ledger.record(&c, Established::Formed);
        assert_eq!(
            Ledger::open(&path).unwrap().verdict(&c),
            Some(Established::Formed)
        );
    }

    #[test]
    fn lines_that_do_not_parse_are_ignored() {
        let records = parse("v k1 formed\nv k2\nv k3 maybe\nv k4 formed 5\nx y z\nc r done\n");
        assert_eq!(records.verdicts.keys().collect::<Vec<_>>(), ["k1"]);
        assert_eq!(
            records.completions.get("r").map(String::as_str),
            Some("done")
        );
    }

    #[test]
    fn keys_separate_their_parts() {
        assert_ne!(Key::of(&[b"ab", b"c"]), Key::of(&[b"a", b"bc"]));
        assert_eq!(Key::of(&[b"ab", b"c"]), Key::of(&[b"ab", b"c"]));
    }
}
