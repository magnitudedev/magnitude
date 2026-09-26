//! Lexical compilation of GBNF: recursive grammars scan regular stretches as
//! single lexemes, and every compiled form accepts exactly the grammar's
//! language, including when greedy lexing cannot be proven safe.
use magnitude_chat::{
    BpeConfig, ByteBpeTokenizer, CacheLimits, ConstraintPlan, PieceKind, SpecialTokens, TokenId,
    Vocabulary,
};
use magnitude_generation::grammar::{to_lark, CONVERTER_IDENTITY};
use std::{collections::BTreeSet, sync::Arc};

fn tokenizer() -> Arc<ByteBpeTokenizer> {
    let mut bytes: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut alphabet: Vec<u32> = bytes.iter().map(|&byte| u32::from(byte)).collect();
    let mut next = 256;
    for byte in 0..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            alphabet.push(next);
            next += 1;
        }
    }
    let mut pieces = vec![String::new(); 256];
    for (byte, code) in bytes.into_iter().zip(alphabet) {
        pieces[byte as usize] = char::from_u32(code).unwrap().to_string();
    }
    pieces.push("<eos>".into());
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.push(PieceKind::Control);
    Arc::new(
        ByteBpeTokenizer::new(BpeConfig {
            artifact_identity: "fixture".into(),
            pieces,
            kinds,
            merges: vec![],
            pattern: r".+|\s".into(),
            normalize_nfc: false,
            stop_tokens: BTreeSet::from([TokenId(256)]),
        })
        .unwrap(),
    )
}

fn accepts(vocabulary: &mut Vocabulary, tokenizer: &ByteBpeTokenizer, gbnf: &str, text: &str) -> bool {
    let initial = vocabulary
        .bind(&ConstraintPlan {
            artifact_identity: tokenizer.artifact_identity().into(),
            tokenizer_identity: tokenizer.identity().into(),
            template_identity: "fixture".into(),
            converter_identity: CONVERTER_IDENTITY.into(),
            gbnf: gbnf.into(),
            initial_prefix: String::new(),
        })
        .unwrap();
    let mut tokens = tokenizer.encode(text, SpecialTokens::Recognize).unwrap();
    tokens.push(TokenId(256));
    initial.advance(&tokens).is_ok()
}

/// Reasoning closed by a delimiter, then recursive JSON-like values: the
/// shape of structured output after a reasoning section.
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

/// A free-text lexeme that the first byte of the following recursion can
/// continue: greedy lexing would lose "abx", so it must not be compiled so.
const OVERRUN: &str = r#"root ::= [a-z]* nest
nest ::= "(" nest ")" | "x"
"#;

/// Two parses stay alive through the recursion, so after it "b" and "bce"
/// are allowed together: greedy lexing of "bcd" would follow "bce" and fail.
const AMBIGUOUS_CONTEXTS: &str = r#"root ::= "a" nest "b" tail | "a" nest "bce"
nest ::= "(" nest ")" | "k"
tail ::= "cd" | "[" tail "]"
"#;

#[test]
fn recursive_grammars_scan_regular_stretches_as_lexemes() {
    let compiled = to_lark(REASONING_JSON).unwrap();
    let lexemes = compiled
        .lines()
        .filter(|line| line.starts_with('L'))
        .collect::<Vec<_>>();
    assert!(!lexemes.is_empty(), "{compiled}");
    // The whole reasoning scanner is inside the lexeme that opens the value.
    assert!(
        lexemes
            .iter()
            .any(|line| line.contains("\"<t>\"") && line.contains(r#""/" "t" ">""#)),
        "{compiled}"
    );
    for unproven in [OVERRUN, AMBIGUOUS_CONTEXTS] {
        let fallback = to_lark(unproven).unwrap();
        assert!(!fallback.lines().any(|line| line.starts_with('L')), "{fallback}");
    }
}

#[test]
fn lexical_compilation_accepts_exactly_the_grammar_language() {
    let tokenizer = tokenizer();
    let mut vocabulary = Vocabulary::new(
        tokenizer.clone(),
        tokenizer.vocabulary(),
        CacheLimits {
            entries: 0,
            bytes: 0,
        },
    )
    .unwrap();
    let cases: [(&str, &[&str], &[&str]); 3] = [
        (
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
        ),
        (
            OVERRUN,
            &["abx", "x", "ab((x))", "(x)"],
            &["ab(x", "abx)", "ab", "ABx"],
        ),
        (
            AMBIGUOUS_CONTEXTS,
            &["akbcd", "akbce", "a((k))b[[cd]]", "a(k)bce"],
            &["akbc", "akb", "akbcde", "a(k))bcd"],
        ),
    ];
    for (grammar, accepted, rejected) in cases {
        for text in accepted {
            assert!(accepts(&mut vocabulary, &tokenizer, grammar, text), "{text:?}");
        }
        for text in rejected {
            assert!(!accepts(&mut vocabulary, &tokenizer, grammar, text), "{text:?}");
        }
    }
}
