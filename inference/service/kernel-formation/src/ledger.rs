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
///
/// Every process with the ledger open appends to the same file, and any of
/// them may compact it. Appending and compacting hold one lock beside the
/// file (`<ledger>.lock`) exclusively, across processes: an append writes
/// to the file the path names when it holds the lock, never to one a
/// compaction has since replaced, and a compaction reads every record
/// appended before it, by any process, into the file it publishes. So no
/// record is lost to a compaction. A record enters the ledger's memory and
/// its file under the one lock hold ([`Ledger::establish`]), so no
/// compaction falls between them to fold an older record of its key over
/// it: what the ledger holds is what its file says.
pub struct Ledger {
    path: PathBuf,
    lock: PathBuf,
    records: Mutex<Records>,
}

impl Ledger {
    /// The ledger at `path`, empty if there is none.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let records = read(path)?;
        if let Some(directory) = path.parent() {
            std::fs::create_dir_all(directory)?;
        }
        let name = path.file_name().expect("a ledger is a file").to_string_lossy();
        Ok(Self {
            path: path.to_owned(),
            lock: path.with_file_name(format!("{name}.lock")),
            records: Mutex::new(records),
        })
    }

    pub fn verdict(&self, key: &Key) -> Option<Established> {
        self.records().verdicts.get(&key.0).copied()
    }

    pub fn record(&self, key: &Key, verdict: Established) {
        self.establish(&verdict_line(&key.0, verdict), |records| {
            records.verdicts.insert(key.0.clone(), verdict);
        });
    }

    /// The summary of the completed run keyed `run`, if there is one.
    pub fn completion(&self, run: &Key) -> Option<String> {
        self.records().completions.get(&run.0).cloned()
    }

    /// Record that the run keyed `run` established everything, with its
    /// `summary` kept on one line.
    pub fn complete(&self, run: &Key, summary: &str) {
        let summary = summary.trim().replace('\n', "; ");
        self.establish(&format!("c {run} {summary}\n"), move |records| {
            records.completions.insert(run.0.clone(), summary);
        });
    }

    /// Rewrite the file as its records, sorted, each once, the ledger's lock
    /// held: this ledger's and every one any process appended to the file,
    /// which this ledger then holds too. The file is replaced in one atomic
    /// step ([`seismic_io::publish`]), so a reader finds it as it was or as
    /// this compaction wrote it, never part written.
    pub fn compact(&self) -> std::io::Result<()> {
        let _held = self.held()?;
        let appended = read(&self.path)?;
        let text = {
            let mut records = self.records();
            records.verdicts.extend(appended.verdicts);
            records.completions.extend(appended.completions);
            let mut text = String::new();
            for (key, verdict) in &records.verdicts {
                text.push_str(&verdict_line(key, *verdict));
            }
            for (run, summary) in &records.completions {
                text.push_str(&format!("c {run} {summary}\n"));
            }
            text
        };
        seismic_io::publish(&self.path, seismic_io::Durability::Volatile, |file| file.write_all(text.as_bytes()))
    }

    fn records(&self) -> std::sync::MutexGuard<'_, Records> {
        self.records
            .lock()
            .expect("no thread panics holding the ledger")
    }

    /// Hold the ledger's lock exclusively, of every thread and process,
    /// until the returned file is dropped.
    fn held(&self) -> std::io::Result<std::fs::File> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&self.lock)?;
        file.lock()?;
        Ok(file)
    }

    /// Establish a record: append its `line` to the file and, once it is
    /// there, `insert` it into memory, both under one hold of the ledger's
    /// lock, so memory never holds a record the file lacks (which a
    /// compaction, folding the file over memory, would replace with an
    /// older line). A failed write is reported and the record is not
    /// established: a later formation records it again.
    fn establish(&self, line: &str, insert: impl FnOnce(&mut Records)) {
        let appended = self.held().and_then(|held| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?
                .write_all(line.as_bytes())?;
            Ok(held)
        });
        match appended {
            Ok(_held) => insert(&mut self.records()),
            Err(error) => eprintln!("formation ledger {}: {error}", self.path.display()),
        }
    }
}

/// The records of the ledger file at `path`, none if there is no file.
fn read(path: &Path) -> std::io::Result<Records> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(parse(&text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Records::default()),
        Err(error) => Err(error),
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

    /// Two processes' ledgers on one file, one appending while the other
    /// compacts again and again: every line of the file parses, and holds
    /// every record either appended, none lost to a compaction.
    #[test]
    fn appends_and_compactions_at_once_lose_no_record() {
        let path = path("compacting");
        let [appending, compacting] = ["a", "b"].map(|name| {
            let ledger = Ledger::open(&path).unwrap();
            ledger.record(&Key::of(&[name.as_bytes()]), Established::Formed);
            ledger
        });
        std::thread::scope(|scope| {
            let appending = &appending;
            seismic::threads::spawn_scoped(scope, "appending", move || {
                for record in 0..200u8 {
                    appending.record(&Key::of(&[b"appended", &[record]]), Established::Rejected);
                }
            })
            .unwrap();
            for _ in 0..50 {
                compacting.compact().unwrap();
            }
        });
        compacting.compact().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let found = parse(&text);
        assert_eq!(found.verdicts.len(), text.lines().count(), "every line parses, each record once");
        assert!(found.verdicts.contains_key(&Key::of(&[b"a"]).0) && found.verdicts.contains_key(&Key::of(&[b"b"]).0));
        for record in 0..200u8 {
            let key = Key::of(&[b"appended", &[record]]);
            assert_eq!(found.verdicts.get(&key.0), Some(&Established::Rejected), "no append is lost");
        }
    }

    /// One ledger re-recording a key on one thread while another thread
    /// compacts it again and again: no compaction folds an older record of
    /// the key over a newer one between its memory and its file, so once
    /// the recording ends the ledger and its file both hold the last.
    #[test]
    fn a_record_and_a_compaction_at_once_leave_memory_as_the_file() {
        let path = path("recompacting");
        let ledger = Ledger::open(&path).unwrap();
        let key = Key::of(&[b"rerecorded"]);
        let verdict = |record: u16| if record % 2 == 0 { Established::Formed } else { Established::Rejected };
        const RECORDS: u16 = 401;
        std::thread::scope(|scope| {
            let (ledger, key) = (&ledger, &key);
            seismic::threads::spawn_scoped(scope, "recording", move || {
                for record in 0..RECORDS {
                    ledger.record(key, verdict(record));
                }
            })
            .unwrap();
            for _ in 0..100 {
                ledger.compact().unwrap();
            }
        });
        let last = verdict(RECORDS - 1);
        assert_eq!(ledger.verdict(&key), Some(last), "the ledger holds the last record");
        assert_eq!(Ledger::open(&path).unwrap().verdict(&key), Some(last), "its file says the last record");
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
