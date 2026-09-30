use super::*;
use std::{collections::BTreeSet, sync::Arc};

/// Tokens 0..=255 spell their byte; 256 is the stop token.
struct ByteTable {
    bytes: Vec<Vec<u8>>,
    stops: BTreeSet<TokenId>,
}

impl ByteTable {
    fn new() -> Arc<Self> {
        let mut bytes = (0..=255u8).map(|byte| vec![byte]).collect::<Vec<_>>();
        bytes.push(b"<eos>".to_vec());
        Arc::new(Self {
            bytes,
            stops: BTreeSet::from([TokenId(256)]),
        })
    }
}

impl TokenTable for ByteTable {
    fn len(&self) -> usize {
        257
    }
    fn bytes(&self, token: TokenId) -> Option<&[u8]> {
        self.bytes.get(token.0 as usize).map(Vec::as_slice)
    }
    fn stop_tokens(&self) -> &BTreeSet<TokenId> {
        &self.stops
    }
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
        Ok(text.bytes().map(|byte| TokenId(byte.into())).collect())
    }
}

fn vocabulary() -> Vocabulary {
    Vocabulary::new(
        ByteTable::new(),
        264,
        CacheLimits {
            entries: 0,
            bytes: 0,
        },
    )
    .unwrap()
}

fn tokens(text: &str) -> Vec<TokenId> {
    text.bytes().map(|byte| TokenId(byte.into())).collect()
}

/// Whether the grammar accepts the complete text followed by the stop token.
fn accepts(vocabulary: &mut Vocabulary, grammar: &Grammar, text: &str) -> bool {
    let state = vocabulary.bind(grammar, &[]).unwrap();
    let mut input = tokens(text);
    input.push(TokenId(256));
    state.advance(&input).is_ok()
}

/// Masks along every prefix of `text`, until the grammar rejects it.
fn masks(vocabulary: &mut Vocabulary, grammar: &Grammar, text: &str) -> Vec<Vec<u32>> {
    let mut state = vocabulary.bind(grammar, &[]).unwrap();
    let mut masks = Vec::new();
    for token in tokens(text) {
        masks.push(Constraint::mask(&state).unwrap().to_vec());
        match state.advance(&[token]) {
            Ok(next) => state = next,
            Err(_) => return masks,
        }
    }
    masks.push(Constraint::mask(&state).unwrap().to_vec());
    masks
}

use magnitude_generation::Constraint;

/// The compiled grammar and the character-level oracle agree on every text
/// and every mask along it, and both match the expectation.
fn check_language(gbnf: &str, accepted: &[&str], rejected: &[&str]) -> CompileReport {
    let compiled = compile(gbnf).unwrap();
    let oracle = compile_characters(gbnf).unwrap();
    let mut vocabulary = vocabulary();
    for (texts, expected) in [(accepted, true), (rejected, false)] {
        for text in texts {
            assert_eq!(
                accepts(&mut vocabulary, &compiled.grammar, text),
                expected,
                "compiled {text:?}\n{}",
                compiled.grammar.lark()
            );
            assert_eq!(
                accepts(&mut vocabulary, &oracle, text),
                expected,
                "oracle {text:?}"
            );
            assert_eq!(
                masks(&mut vocabulary, &compiled.grammar, text),
                masks(&mut vocabulary, &oracle, text),
                "masks along {text:?}\n{}",
                compiled.grammar.lark()
            );
        }
    }
    compiled.report
}

/// Reasoning closed by a delimiter, then recursive JSON-like values.
const REASONING_JSON: &str = r#"root ::= "<t>" reasoning space value
reasoning ::= [^<] reasoning | "<" open
open ::= "/" slash | "<" open | [^/<] reasoning
slash ::= "t" tee | "<" open | [^t<] reasoning
tee ::= ">" | "<" open | [^><] reasoning
value ::= object | array | string | "null"
object ::= "{" space (string ":" space value ("," space string ":" space value)*)? space "}"
array ::= "[" space (value ("," space value)*)? space "]"
string ::= "\"" [^"]* "\""
space ::= | " " | "\n"
"#;

/// Free text the first byte of the following recursion can continue:
/// greedy lexing would lose "abx".
const OVERRUN: &str = r#"root ::= [a-z]* nest
nest ::= "(" nest ")" | "x"
"#;

/// Two parses stay alive through the recursion, so after it "b" and "bce"
/// are allowed together.
const AMBIGUOUS_CONTEXTS: &str = r#"root ::= "a" nest "b" tail | "a" nest "bce"
nest ::= "(" nest ")" | "k"
tail ::= "cd" | "[" tail "]"
"#;

#[test]
fn rejects_inadmissible_sources() {
    for invalid in [
        "root ::= missing",
        "root ::= \"\\uD800\"",
        "root ::= [z-a]",
        "root ::= \"x\"{3,2}",
        "root ::= (\"x\"",
        "root ::= \"x\"\nroot ::= \"y\"",
        "other ::= \"x\"",
    ] {
        assert!(
            matches!(compile(invalid), Err(GrammarError::Syntax(_))),
            "{invalid}"
        );
    }
    assert!(matches!(
        compile("root ::= loop\nloop ::= \"a\" loop"),
        Err(GrammarError::Language(_))
    ));
}

#[test]
fn preserves_unicode_classes_repetitions_and_names() {
    let cases: [(&str, &[&str], &[&str]); 5] = [
        (
            "root ::= start A a\nstart ::= \"\\u000a\"\nA ::= [\\x61-\\u0062]{1,2}\na ::= \"\\U0001f999\"\n",
            &["\na🦙", "\nbb🦙"],
            &["na🦙", "\nccc🦙", "\na"],
        ),
        (
            "root ::= (\"a\" |\n \"b\")+ [^x-z]? # comment\n",
            &["a", "abba!", "b\n"],
            &["", "abz", "xx"],
        ),
        ("root ::= .{2,3}\n", &["é🦙", "\n\n", "abc"], &["a", "abcd"]),
        ("root ::= \"a\"{2,}\n", &["aa", "aaaa"], &["", "a", "aab"]),
        (
            "root ::= x | dead\nx ::= \"x\"\ndead ::= \"d\" dead\n",
            &["x"],
            &["d", "dd"],
        ),
    ];
    for (gbnf, accepted, rejected) in cases {
        let report = check_language(gbnf, accepted, rejected);
        assert_eq!(report.character_lexemes, 0, "{gbnf}");
    }
}

#[test]
fn reasoning_scanner_and_recursive_values_are_word_level() {
    let report = check_language(
        REASONING_JSON,
        &[
            "<t>a<b</t>{\"k\": [null, {\"x\":\"y\"}]}",
            "<t></t> [ ]",
            "<t>x</t>\n{ }",
            "<t>{\"not\": \"json yet\"}</t>\"s\"",
            "<t><</t>{\"a\":{\"b\":{\"c\":[[[]]]}}}",
        ],
        &[
            "<t>a</t>{\"k\" null}",
            "<t>a</t>[",
            "<t>a{}",
            "<t>a</t>  {}",
            "<t>a</t>{} ",
        ],
    );
    assert_eq!(report.scanners, 1);
    assert_eq!(report.character_lexemes, 0);
    // The reasoning scanner, its delimiter and the separating space are one
    // lexeme.
    let lark = compile(REASONING_JSON).unwrap().grammar;
    assert!(lark.lark().contains("\"<t>\""), "{}", lark.lark());
}

#[test]
fn lexically_ambiguous_grammars_are_character_level_only_where_needed() {
    let report = check_language(
        OVERRUN,
        &["abx", "x", "ab((x))", "(x)"],
        &["ab(x", "abx)", "ab", "ABx"],
    );
    assert!(report.character_lexemes > 0);
    let report = check_language(
        AMBIGUOUS_CONTEXTS,
        &["akbcd", "akbce", "a((k))b[[cd]]", "a(k)bce"],
        &["akbc", "akb", "akbcde", "a(k))bcd"],
    );
    assert!(report.character_lexemes > 0);
}

#[test]
fn recursive_scanners_and_nested_structure() {
    check_language(
        "root ::= scanner object\nscanner ::= [^<] scanner | \"<\"\nobject ::= \"(\" object \" )\" | \"value\"\n",
        &["arbitrary scanner prefix <((value ) )", "<value"],
        &["prefix <(value", "prefix"],
    );
    check_language(
        "root ::= scan \"b\"\nscan ::= \"a\" scan | \"b\" scan | \"\"\n",
        &["b", "ab", "bb", "aabb"],
        &["", "a", "ba"],
    );
}

/// Text up to a delimiter, then any number of calls each opened by that
/// delimiter and holding recursion: the tool-call section of a whole-completion
/// grammar whose arguments are free-form objects.
const SCANNED_CALLS: &str = r#"root ::= text ("<c>" nest ">")*
text ::= | "<" open | [^<] text
open ::= | "<" open | "c" tag | [^<c] text
tag ::= | "<" open | [^<>] text
nest ::= "(" nest ")" | "x"
"#;

#[test]
fn a_scanner_and_its_delimiter_share_a_lexeme_across_a_call_loop() {
    // The scanner continues over the delimiter's first byte, so it is exact
    // under greedy lexing only in one lexeme with the delimiter: the loop is
    // peeled rather than its head left between them, which would render the
    // free text one character at a time.
    let report = check_language(
        SCANNED_CALLS,
        &[
            "",
            "free <text> <c",
            "a<c>(x)><c>((x))>",
            "<<c<c x<c>x>",
            "<c>x>",
        ],
        &["a<c>", "a<c>(x)> b", "<c>(x)", "a<c>(x)>><c", "<c>>"],
    );
    assert_eq!(report.peeled, 1, "{report:?}");
    assert_eq!(report.character_lexemes, 0, "{report:?}");
}

/// llama.cpp's any-order argument lattice: one rule per subset of the
/// remaining required arguments.
fn lattice(arguments: usize, recursive: bool) -> String {
    // Reasoning is any text up to and including the first "</t>".
    let mut gbnf = String::from("root ::= \"<t>\" reasoning \"\\n\" call\n");
    gbnf.push_str("reasoning ::= [^<] reasoning | \"<\" open\n");
    gbnf.push_str("open ::= \"/\" slash | \"<\" open | [^/<] reasoning\n");
    gbnf.push_str("slash ::= \"t\" tee | \"<\" open | [^t<] reasoning\n");
    gbnf.push_str("tee ::= \">\" | \"<\" open | [^><] reasoning\n");
    gbnf.push_str("call ::= \"<function=f>\\n\" args-full \"</function>\"\n");
    gbnf.push_str("value ::= [^\\n]+\n");
    if recursive {
        gbnf.push_str("json ::= \"{\" (json (\",\" json)*)? \"}\" | \"1\"\n");
    }
    for index in 0..arguments {
        let value = if recursive && index == 0 {
            "json"
        } else {
            "value"
        };
        gbnf.push_str(&format!(
            "arg{index} ::= \"<parameter=p{index}>\\n\" {value} \"\\n</parameter>\\n\"\n"
        ));
    }
    let full = (1u32 << arguments) - 1;
    for subset in 1..=full {
        let name = if subset == full {
            "args-full".to_string()
        } else {
            format!("args-{subset}")
        };
        let alternatives = (0..arguments)
            .filter(|index| subset & (1 << index) != 0)
            .map(|index| {
                let rest = subset & !(1 << index);
                if rest == 0 {
                    format!("arg{index}")
                } else if rest == full {
                    format!("arg{index} args-full")
                } else {
                    format!("arg{index} args-{rest}")
                }
            })
            .collect::<Vec<_>>();
        gbnf.push_str(&format!("{name} ::= {}\n", alternatives.join(" | ")));
    }
    gbnf
}

#[test]
fn regular_argument_lattices_compile_to_one_lexeme_of_linear_size() {
    let mut previous = 0;
    for arguments in 1..=6 {
        let gbnf = lattice(arguments, false);
        let report = compile(&gbnf).unwrap().report;
        assert_eq!(report.earley_rules, 1, "{arguments}");
        assert_eq!(report.lexemes, 1, "{arguments}");
        assert_eq!(report.character_lexemes, 0);
        assert!(report.lark_bytes <= 8 * report.gbnf_bytes, "{report:?}");
        assert!(report.lark_bytes > previous);
        previous = report.lark_bytes;
    }
    check_language(
        &lattice(3, false),
        &[
            "<t>a<b</t>\n<function=f>\n<parameter=p1>\nx\n</parameter>\n<parameter=p0>\ny\n</parameter>\n<parameter=p2>\nz\n</parameter>\n</function>",
        ],
        &[
            "<t></t>\n<function=f>\n<parameter=p1>\nx\n</parameter>\n<parameter=p1>\ny\n</parameter>\n<parameter=p2>\nz\n</parameter>\n</function>",
            "<t></t>\n<function=f>\n<parameter=p1>\nx\n</parameter>\n</function>",
        ],
    );
}

#[test]
fn recursive_argument_lattices_stay_word_level_and_bounded() {
    for arguments in 1..=6 {
        let gbnf = lattice(arguments, true);
        let report = compile(&gbnf).unwrap().report;
        assert_eq!(report.character_lexemes, 0, "{arguments}: {report:?}");
        assert_eq!(report.exhausted_questions, 0, "{arguments}: {report:?}");
        assert!(
            report.lark_bytes <= ALLOWANCE_BOUND * report.gbnf_bytes,
            "{report:?}"
        );
    }
    check_language(
        &lattice(3, true),
        &[
            "<t>r</t>\n<function=f>\n<parameter=p1>\nx\n</parameter>\n<parameter=p0>\n{{1},1}\n</parameter>\n<parameter=p2>\nz\n</parameter>\n</function>",
            "<t></t>\n<function=f>\n<parameter=p0>\n1\n</parameter>\n<parameter=p2>\nz\n</parameter>\n<parameter=p1>\nx\n</parameter>\n</function>",
        ],
        &[
            "<t></t>\n<function=f>\n<parameter=p0>\n{1\n</parameter>\n<parameter=p2>\nz\n</parameter>\n<parameter=p1>\nx\n</parameter>\n</function>",
            "<t></t>\n<function=f>\n<parameter=p0>\n1\n</parameter>\n</function>",
        ],
    );
}

const ALLOWANCE_BOUND: usize = 2 * terminal::ALLOWANCE_FACTOR;

#[test]
fn compiled_grammars_are_cached_by_source() {
    let cache = CompileCache::new(1, 1 << 20);
    let first = cache.compile("root ::= \"a\"").unwrap();
    assert!(Arc::ptr_eq(
        &first,
        &cache.compile("root ::= \"a\"").unwrap()
    ));
    let other = cache.compile("root ::= \"b\"").unwrap();
    assert!(!Arc::ptr_eq(
        &first,
        &cache.compile("root ::= \"a\"").unwrap()
    ));
    assert!(!Arc::ptr_eq(
        &other,
        &cache.compile("root ::= \"a\"").unwrap()
    ));
    assert!(matches!(
        cache.compile("root ::= missing"),
        Err(GrammarError::Syntax(_))
    ));
}

#[test]
fn binding_consumes_the_prefix_and_caches_by_grammar() {
    let mut vocabulary = Vocabulary::new(
        ByteTable::new(),
        264,
        CacheLimits {
            entries: 2,
            bytes: 1 << 20,
        },
    )
    .unwrap();
    let grammar = compile("root ::= \"prefix:ab\"").unwrap().grammar;
    let initial = vocabulary.bind(&grammar, &tokens("prefix:")).unwrap();
    assert_eq!(initial.position(), 0);
    assert_eq!(initial.forced(1).unwrap(), tokens("a"));
    let advanced = initial.advance(&tokens("a")).unwrap();
    assert_eq!(advanced.forced(5).unwrap(), tokens("b"));
    vocabulary.bind(&grammar, &tokens("prefix:")).unwrap();
    assert_eq!(vocabulary.cache_hits(), 1);
    assert!(vocabulary.bind(&grammar, &tokens("wrong")).is_err());
    assert!(vocabulary.bind(&grammar, &[TokenId(256)]).is_err());
}

/// Deterministic pseudo-random strings of a grammar and near misses: the
/// compiled grammar and the oracle must agree on all of them.
#[test]
fn compiled_grammars_agree_with_the_oracle_on_generated_texts() {
    struct Random(u64);
    impl Random {
        fn below(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % bound.max(1)
        }
    }
    /// Whether a complete derivation was written within the depth bound.
    fn generate(
        rules: &gbnf::Rules,
        expr: &gbnf::Expr,
        random: &mut Random,
        depth: usize,
        out: &mut String,
    ) -> bool {
        if depth > 60 {
            return false;
        }
        match expr {
            gbnf::Expr::Literal(text) => {
                out.push_str(text);
                true
            }
            gbnf::Expr::Class { negated, ranges } => {
                let candidates = [
                    'a', 'b', 'k', 'x', '<', '/', 't', '>', '"', '{', '}', '(', ')', '[', ']', ' ',
                    '\n', '1', ',', 'c', 'd', 'e',
                ];
                let fits = |c: char| ranges.iter().any(|(s, e)| (*s..=*e).contains(&c)) != *negated;
                let options = candidates
                    .iter()
                    .copied()
                    .filter(|&c| fits(c))
                    .collect::<Vec<_>>();
                match options.get(random.below(options.len())) {
                    Some(&c) => out.push(c),
                    None if !*negated => out.push(ranges[0].0),
                    None => return false,
                }
                true
            }
            gbnf::Expr::Any => {
                out.push('a');
                true
            }
            gbnf::Expr::Reference(name) => generate(rules, &rules[name], random, depth + 1, out),
            gbnf::Expr::Sequence(parts) => parts
                .iter()
                .all(|part| generate(rules, part, random, depth, out)),
            gbnf::Expr::Alternative(parts) => {
                // Deep in a derivation, prefer alternatives that end it.
                let closing = parts
                    .iter()
                    .filter(|part| {
                        let mut names = BTreeSet::new();
                        part.references(&mut names);
                        names.is_empty()
                    })
                    .collect::<Vec<_>>();
                let part = if depth > 8 && !closing.is_empty() {
                    closing[random.below(closing.len())]
                } else {
                    &parts[random.below(parts.len())]
                };
                generate(rules, part, random, depth, out)
            }
            gbnf::Expr::Repeat(part, min, max) => {
                let limit = max.unwrap_or(min + 3).min(min + 3);
                let count = if depth > 8 {
                    *min as usize
                } else {
                    *min as usize + random.below((limit - min + 1) as usize)
                };
                (0..count).all(|_| generate(rules, part, random, depth + 1, out))
            }
        }
    }
    let mut vocabulary = vocabulary();
    let mut random = Random(7);
    for gbnf in [
        REASONING_JSON,
        OVERRUN,
        AMBIGUOUS_CONTEXTS,
        &lattice(3, true),
    ] {
        let rules = gbnf::parse(gbnf).unwrap();
        let compiled = compile(gbnf).unwrap().grammar;
        let oracle = compile_characters(gbnf).unwrap();
        let mut generated = 0;
        while generated < 40 {
            let mut text = String::new();
            if !generate(&rules, &rules["root"], &mut random, 0, &mut text) {
                continue;
            }
            generated += 1;
            assert!(accepts(&mut vocabulary, &oracle, &text), "{gbnf}\n{text:?}");
            let mut variants = vec![text.clone()];
            if !text.is_empty() {
                let at = text
                    .char_indices()
                    .map(|(index, _)| index)
                    .nth(random.below(text.chars().count()))
                    .unwrap();
                let mut removed = text.clone();
                removed.remove(at);
                variants.push(removed);
                let mut inserted = text.clone();
                inserted.insert(at, ['<', '"', '}', 'x', '\n'][random.below(5)]);
                variants.push(inserted);
            }
            for variant in variants {
                assert_eq!(
                    accepts(&mut vocabulary, &compiled, &variant),
                    accepts(&mut vocabulary, &oracle, &variant),
                    "{gbnf}\n{variant:?}\n{}",
                    compiled.lark()
                );
            }
        }
    }
}
