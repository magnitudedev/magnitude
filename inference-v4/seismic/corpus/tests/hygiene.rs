//! Textual hygiene of the in-scope source trees (design A10 §2.11).
//!
//! The patterns are assembled with `concat!` so this file does not match itself.
use crate::common::corpus_path;
use std::path::{Path, PathBuf};

/// Scanned trees, relative to the workspace root `inference-v4/`.
const TREES: &[&str] = &[
    "seismic/lang",
    "seismic/ir",
    "seismic/native-target",
    "seismic/compiler",
    "seismic/runtime",
    "seismic/api",
    "seismic/build",
    "seismic/cli",
    "seismic/backends",
    "seismic/corpus",
    "seismic-std",
    "engine/src",
    "engine/tests",
    "engine/model-kernels",
];

/// Every `.rs` file of the scanned trees, excluding `target/` directories.
fn sources() -> Vec<(PathBuf, String)> {
    fn walk(path: &Path, files: &mut Vec<(PathBuf, String)>) {
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "target") {
                return;
            }
            let entries =
                std::fs::read_dir(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            for entry in entries {
                walk(
                    &entry
                        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
                        .path(),
                    files,
                );
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let text =
                std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            files.push((path.to_owned(), text));
        }
    }
    let root = corpus_path("../..");
    let mut files = Vec::new();
    for tree in TREES {
        let path = root.join(tree);
        assert!(path.exists(), "scanned tree {tree} is missing");
        walk(&path, &mut files);
    }
    files.sort();
    files
}

/// `path:line: text<note>` for every line that `violation` describes with a note.
fn scan(violation: impl Fn(&str) -> Option<String>) -> Vec<String> {
    let root = corpus_path("../..");
    let mut hits = Vec::new();
    for (path, text) in sources() {
        let path = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        for (index, line) in text.lines().enumerate() {
            if let Some(note) = violation(line) {
                hits.push(format!("{path}:{}: {}{note}", index + 1, line.trim()));
            }
        }
    }
    hits
}

fn assert_none(what: &str, hits: Vec<String>) {
    assert!(
        hits.is_empty(),
        "{} {what}:\n{}",
        hits.len(),
        hits.join("\n")
    );
}

#[test]
fn no_ignored_tests() {
    assert_none(
        "ignored test(s)",
        scan(|line| line.contains(concat!("#[", "ignore")).then(String::new)),
    );
}

#[test]
fn no_stack_override() {
    assert_none(
        "stack override(s)",
        scan(|line| {
            line.contains(concat!("RUST_MIN", "_STACK"))
                .then(String::new)
        }),
    );
}

#[test]
fn no_dead_code_allowances() {
    assert_none(
        "dead-code allowance(s)",
        scan(|line| {
            let line: String = line.split_whitespace().collect();
            (line.contains(concat!("allow(", "dead_code"))
                || line.contains(concat!("allow(", "unused")))
            .then(String::new)
        }),
    );
}

/// Every identifier listed in `hygiene/forbidden-A*.txt`.
fn forbidden_symbols() -> Vec<(String, String)> {
    let directory = corpus_path("hygiene");
    let mut lists: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap_or_else(|e| panic!("{}: {e}", directory.display()))
        .map(|entry| entry.expect("hygiene directory entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("forbidden-A") && name.ends_with(".txt"))
        })
        .collect();
    lists.sort();
    let mut symbols = Vec::new();
    for list in lists {
        let text =
            std::fs::read_to_string(&list).unwrap_or_else(|e| panic!("{}: {e}", list.display()));
        let owner = list
            .file_name()
            .expect("listed file")
            .to_string_lossy()
            .into_owned();
        for (index, line) in text.lines().enumerate() {
            let symbol = line
                .split('#')
                .next()
                .expect("split yields one part")
                .trim();
            if symbol.is_empty() {
                continue;
            }
            assert!(
                symbol
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "{owner}:{}: `{symbol}` is not one identifier",
                index + 1
            );
            symbols.push((symbol.to_owned(), owner.clone()));
        }
    }
    symbols
}

fn contains_word(line: &str, word: &str) -> bool {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    line.match_indices(word).any(|(start, _)| {
        !line[..start].ends_with(is_word) && !line[start + word.len()..].starts_with(is_word)
    })
}

#[test]
fn superseded_symbols_are_absent() {
    let symbols = forbidden_symbols();
    let hits = scan(|line| {
        let named: Vec<String> = symbols
            .iter()
            .filter(|(symbol, _)| contains_word(line, symbol))
            .map(|(symbol, owner)| format!("{symbol} ({owner})"))
            .collect();
        (!named.is_empty()).then(|| format!("  <- {}", named.join(", ")))
    });
    assert_none("use(s) of superseded symbols", hits);
}
