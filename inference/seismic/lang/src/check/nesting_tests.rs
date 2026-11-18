use crate::bundle::{decode_checked_bundle, encode_checked_bundle};
use crate::checked::{check_source, CheckedModule, DiagnosticRule, SourceError, SourceFile, SourceSet};
use crate::entry::ElementBindings;
use super::ir::MAX_NESTING;

fn source(name: &str, text: String) -> SourceSet {
    SourceSet::new(vec![SourceFile { path: format!("{name}.seismic"), text }])
}

/// `fn chain() -> f32` returning a sum of `terms` literals, an expression
/// nested about `terms` deep.
fn chain(terms: usize) -> SourceSet {
    let sum = vec!["1.0"; terms].join(" + ");
    source("chain", format!("fn chain() -> f32:\n    return {sum}\n"))
}

/// `fn deep` whose body nests `levels` `if` statements, the innermost
/// assigning a sum: a body nested `levels + 3` deep in statements.
fn statements(levels: usize) -> SourceSet {
    let mut text = "fn deep(x: i32) -> i32:\n    let mut y = 0\n".to_owned();
    for level in 0..levels {
        text.push_str(&format!("{}if x > {level}:\n", "    ".repeat(level + 1)));
    }
    text.push_str(&format!("{}y = y + x\n    return y\n", "    ".repeat(levels + 1)));
    source("statements", text)
}

/// `fn deep` binding a sum of `terms` operands, left to right: an
/// expression the parser builds without recursing, nested `terms` deep.
fn sums(terms: usize) -> SourceSet {
    let sum = vec!["x"; terms].join(" + ");
    source("sums", format!("fn deep(x: f32) -> f32:\n    let z = {sum}\n    return z\n"))
}

/// `fn deep` binding a sum nested right to left in parentheses: an
/// expression the parser recurses into, nested `terms` deep.
fn parenthesized(terms: usize) -> SourceSet {
    let sum = format!("{}x + x{}", "x + (".repeat(terms - 2), ")".repeat(terms - 2));
    source("parenthesized", format!("fn deep(x: f32) -> f32:\n    let z = {sum}\n    return z\n"))
}

fn refused_for_nesting(refused: &SourceError) -> bool {
    refused.diagnostics().items().iter().any(|item| item.rule == DiagnosticRule::Nesting)
}

/// A body nested within the bound checks; one nested past it is refused,
/// so no checked program's decode recurses deeper than the stack it is
/// sized for.
#[test]
fn a_body_nested_past_the_bound_is_refused() {
    seismic_threads::run("nesting", || {
        check_source(chain(MAX_NESTING / 2)).unwrap();
        let refused = check_source(chain(2 * MAX_NESTING)).unwrap_err();
        assert!(refused_for_nesting(&refused), "{:?}", refused.diagnostics());
    });
}

/// Source nested well past the bound, in every way the parser recurses or
/// builds without recursing, is refused as it is parsed: no walk recurses
/// with it, so none overflows however deep it nests.
#[test]
fn source_nested_past_the_bound_is_refused_as_it_parses() {
    seismic_threads::run("nesting", || {
        let deep = 4 * MAX_NESTING;
        for refused in [statements(deep), sums(deep), parenthesized(deep)].map(|source| check_source(source).unwrap_err()) {
            assert!(refused_for_nesting(&refused), "{:?}", refused.diagnostics());
        }
    });
}

/// `walk` on a thread of exactly `seismic_threads::PROGRAM_BYTES`: the
/// stack the process's threads are sized to give one walk of a program.
fn on_the_program_budget(walk: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("program-budget".into())
        .stack_size(seismic_threads::PROGRAM_BYTES)
        .spawn(walk)
        .unwrap()
        .join()
        .unwrap();
}

/// The deepest program of `shape` that checks, nested exactly
/// [`MAX_NESTING`] deep, with the next deeper refused for its nesting.
fn deepest(shape: fn(usize) -> SourceSet) -> CheckedModule {
    let mut deepest = None;
    for size in MAX_NESTING - 12..=MAX_NESTING + 1 {
        match check_source(shape(size)) {
            Ok(module) => deepest = Some(module),
            Err(refused) => {
                assert!(refused_for_nesting(&refused), "{size}: {:?}", refused.diagnostics());
                break;
            }
        }
    }
    let module = deepest.expect("a program a dozen levels short of the bound checks");
    let nesting = module.internal().definitions.iter().find(|definition| definition.name == "deep").expect("the program's definition").body.nesting();
    assert_eq!(nesting, MAX_NESTING, "the deepest program the parser accepts is nested as deep as the checked bound");
    module
}

/// Every walk of a program nested as deep as the bound allows (parsing,
/// checking, monomorphizing, encoding and decoding it, and monomorphizing
/// the decoded program as a load does) runs within the stack the process
/// gives one walk of a program, in an unoptimized build: the frames those
/// walks take at `MAX_NESTING` levels are within `PROGRAM_BYTES`, which
/// `STACK_BYTES` is derived from. A walk that outgrew it overflows here.
#[test]
fn every_walk_of_the_deepest_program_runs_within_its_stack() {
    on_the_program_budget(|| {
        for shape in [statements as fn(usize) -> SourceSet, sums, parenthesized] {
            let module = deepest(shape);
            let entry = module.entry_named("deep").expect("the program's entry");
            module.entry(entry, &ElementBindings::default()).expect("the deepest program monomorphizes");
            let bytes: &'static [u8] = encode_checked_bundle(&module).leak();
            let decoded = decode_checked_bundle(bytes).expect("the deepest program decodes");
            let entry = decoded.entry_named("deep").expect("the decoded program's entry");
            decoded.entry(entry, &ElementBindings::default()).expect("the decoded program monomorphizes");
        }
    });
}
