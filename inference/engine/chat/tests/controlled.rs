//! Controlled encoding: control tokens are recognized only inside the
//! template-authored ranges, and the text is encoded in one pass.
mod support;

use magnitude_chat::{
    AddedTokenEnd, BpeConfig, ByteBpeTokenizer, Normalization, PieceEncoding, PieceKind,
    SpecialTokens, Split, SplitBehavior, TokenId,
};
use std::collections::BTreeSet;

const CONTROL: &str = "<|c|>";
const LONGER: &str = "<|c|>x";

/// Byte pieces, the merged piece `ab` (256), a control token (257), a
/// longer control token sharing its prefix (258) and a user-defined piece
/// (259). Every run of non-space characters is one split piece.
fn tokenizer() -> ByteBpeTokenizer {
    tokenizer_starting(None)
}

/// [`tokenizer`] with `implicit_bos` as its implicit sequence start.
fn tokenizer_starting(implicit_bos: Option<TokenId>) -> ByteBpeTokenizer {
    let mut pieces = support::byte_pieces();
    let mut kinds = vec![PieceKind::Normal; 256];
    for (piece, kind) in [
        ("ab", PieceKind::Normal),
        (CONTROL, PieceKind::Control),
        (LONGER, PieceKind::Control),
        ("<u>", PieceKind::UserDefined),
    ] {
        pieces.push(piece.into());
        kinds.push(kind);
    }
    ByteBpeTokenizer::new(BpeConfig {
        artifact_identity: "fixture".into(),
        pieces,
        kinds,
        merges: vec![("a".into(), "b".into())],
        normalization: Normalization::None,
        splits: vec![Split {
            pattern: r"\S+|\s+".into(),
            behavior: SplitBehavior::Isolated,
        }],
        encoding: PieceEncoding::ByteLevel,
        ignore_merges: false,
        implicit_bos,
        stop_tokens: BTreeSet::new(),
        suppressed_tokens: BTreeSet::new(),
    })
    .unwrap()
}

/// The tokens of `text` as a sequence (the fixture inserts no BOS).
fn controlled(
    tokenizer: &ByteBpeTokenizer,
    text: &str,
    controls: &[std::ops::Range<usize>],
) -> Result<Vec<TokenId>, String> {
    Ok(tokenizer.encode_sequence_controlled(text, controls)?.tokens)
}

fn byte(c: char) -> TokenId {
    TokenId(c as u32)
}

#[test]
fn controlled_equals_recognizing_when_request_text_holds_no_control_spelling() {
    let tokenizer = tokenizer();
    for (text, controls) in [
        ("<|c|>ab cd<|c|>", vec![0..5, 10..15]),
        ("<|c|>xab<|c|>", vec![0..13]),
        ("ab <u> ab", vec![]),
        ("", vec![]),
    ] {
        assert_eq!(
            controlled(&tokenizer, text, &controls).unwrap(),
            tokenizer.encode(text, SpecialTokens::Recognize).unwrap(),
            "{text:?}"
        );
    }
}

#[test]
fn control_spellings_outside_the_ranges_stay_text() {
    let tokenizer = tokenizer();
    let text = "<|c|>a<|c|>";
    assert_eq!(
        controlled(&tokenizer, text, &[0..5]).unwrap(),
        [
            vec![TokenId(257), byte('a')],
            tokenizer.encode(CONTROL, SpecialTokens::Literal).unwrap(),
        ]
        .concat()
    );
    // A spelling straddling a range's edge is literal too.
    assert_eq!(
        controlled(&tokenizer, text, &[0..8]).unwrap(),
        [
            vec![TokenId(257), byte('a')],
            tokenizer.encode(CONTROL, SpecialTokens::Literal).unwrap(),
        ]
        .concat()
    );
}

#[test]
fn ranges_do_not_split_the_text_they_border() {
    let tokenizer = tokenizer();
    // `a` is template text and `b` request text: one piece, one merge.
    assert_eq!(
        controlled(&tokenizer, "<|c|>ab", &[0..6]).unwrap(),
        vec![TokenId(257), TokenId(256)]
    );
    // The longest control spelling inside a range wins.
    assert_eq!(
        controlled(&tokenizer, "<|c|>x", &[0..6]).unwrap(),
        vec![TokenId(258)]
    );
    assert_eq!(
        controlled(&tokenizer, "<|c|>x", &[0..5]).unwrap(),
        vec![TokenId(257), byte('x')]
    );
}

#[test]
fn ranges_must_be_ascending_disjoint_character_ranges() {
    let tokenizer = tokenizer();
    for controls in [vec![2..4, 0..1], vec![0..3, 2..4], vec![3..2]] {
        assert!(controlled(&tokenizer, "abcdef", &controls).is_err());
    }
    assert!(controlled(&tokenizer, "é", &[0..1]).is_err());
}

#[test]
fn a_controlled_sequence_starts_once_and_ends_its_added_tokens() {
    let tokenizer = tokenizer_starting(Some(TokenId(257)));
    let end = |offset, position| AddedTokenEnd { offset, position };
    // A sequence whose template text opens with the implicit BOS has no
    // second one; any other is given one, which moves every position.
    for (text, tokens, ends) in [
        (
            "<|c|>ab <u>",
            vec![TokenId(257), TokenId(256), byte(' '), TokenId(259)],
            vec![end(5, 1), end(11, 4)],
        ),
        (
            "ab <u> <|c|>x",
            vec![
                TokenId(257),
                TokenId(256),
                byte(' '),
                TokenId(259),
                byte(' '),
                TokenId(258),
            ],
            vec![end(6, 4), end(13, 6)],
        ),
    ] {
        let sequence = tokenizer
            .encode_sequence_controlled(text, &[0..text.len()])
            .unwrap();
        assert_eq!(sequence.tokens, tokens, "{text:?}");
        assert_eq!(sequence.added_token_ends, ends, "{text:?}");
    }
    // Request text spelling a control token stays text; the sequence still
    // starts with the implicit BOS, and the user-defined piece ends at 6.
    let sequence = tokenizer
        .encode_sequence_controlled("<|c|> <u>", &[])
        .unwrap();
    assert_eq!(sequence.tokens.first(), Some(&TokenId(257)));
    assert_eq!(sequence.tokens.last(), Some(&TokenId(259)));
    assert_eq!(
        sequence.added_token_ends,
        vec![AddedTokenEnd {
            offset: 9,
            position: sequence.tokens.len(),
        }]
    );
}
