//! Every file in the standard library parses, and printing is idempotent.

use seismic_lang::{parse, print};
use std::path::{Path, PathBuf};

fn std_files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib");
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.to_string_lossy().contains(".seismic.") {
                out.push(path);
            }
        }
    }
    walk(&root, &mut out);
    out.sort();
    out
}

#[test]
fn standard_library_round_trips() {
    let files = std_files();
    assert!(!files.is_empty(), "no std files found");
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let file = parse(&text).unwrap_or_else(|d| panic!("{}", d.render(&path.display().to_string(), &text)));
        let printed = print(&file);
        let reparsed = parse(&printed).unwrap_or_else(|d| panic!("reparse of {}:\n{}\n{}", path.display(), d.render("<printed>", &printed), printed));
        assert_eq!(print(&reparsed), printed, "printing is not idempotent for {}", path.display());
        assert_eq!(reparsed.decls.len(), file.decls.len());
    }
}

#[test]
fn precedence_and_parenthesization() {
    let cases = [
        ("x = (a + b) * c\n", "x = (a + b) * c\n"),
        ("x = a + b * c\n", "x = a + b * c\n"),
        ("x = -(a + b)\n", "x = -(a + b)\n"),
        ("x = a - (b - c)\n", "x = a - (b - c)\n"),
        ("x = (a - b) - c\n", "x = a - b - c\n"),
        ("x = not a and b\n", "x = not a and b\n"),
        ("x = not (a and b)\n", "x = not (a and b)\n"),
        ("x = (w >> (4 * (k % 8))) & 0xF\n", "x = w >> 4 * (k % 8) & 15\n"),
        ("x = w >> (4 & k)\n", "x = w >> (4 & k)\n"),
        ("x = f((a, b) -> a * b, t)\n", "x = f((a, b) -> a * b, t)\n"),
        ("x = t[1:, :k, 2]\n", "x = t[1:, :k, 2]\n"),
    ];
    for (input, expected) in cases {
        let src = format!("fn f():\n  {input}");
        let file = parse(&src).unwrap_or_else(|d| panic!("{}", d.render("<case>", &src)));
        let printed = print(&file);
        assert_eq!(printed, format!("fn f():\n  {expected}"), "input: {input}");
    }
}
