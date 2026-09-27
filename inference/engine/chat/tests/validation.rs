//! Tokenizer validation reaches the verdict construction reaches: assessment
//! validates a model's tokenizer without building it, so a configuration it
//! accepts builds, and one it rejects does not.
use magnitude_chat::{
    BpeConfig, ByteBpeTokenizer, Normalization, PieceEncoding, PieceKind, Split, SplitBehavior,
    TokenId,
};
use std::collections::BTreeSet;

/// A byte-level vocabulary spelling every byte, a merged piece and a
/// control piece, with the merge that forms the merged piece.
fn config() -> BpeConfig {
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
    pieces.push("ab".into());
    pieces.push("<eos>".into());
    let mut kinds = vec![PieceKind::Normal; 257];
    kinds.push(PieceKind::Control);
    BpeConfig {
        artifact_identity: "fixture".into(),
        pieces,
        kinds,
        merges: vec![("a".into(), "b".into())],
        normalization: Normalization::None,
        splits: vec![Split {
            pattern: r"\w+|\s".into(),
            behavior: SplitBehavior::Isolated,
        }],
        encoding: PieceEncoding::ByteLevel,
        ignore_merges: false,
        implicit_bos: None,
        stop_tokens: BTreeSet::from([TokenId(257)]),
        suppressed_tokens: BTreeSet::new(),
    }
}

fn verdicts(config: BpeConfig) -> (Option<usize>, Option<usize>) {
    (
        config.validate().ok(),
        ByteBpeTokenizer::new(config)
            .ok()
            .map(|tokenizer| tokenizer.vocabulary()),
    )
}

#[test]
fn a_valid_configuration_validates_to_its_built_vocabulary() {
    assert_eq!(verdicts(config()), (Some(258), Some(258)));
}

#[test]
fn every_rejection_of_construction_is_a_rejection_of_validation() {
    let cases: [(&str, fn(&mut BpeConfig)); 7] = [
        ("a merge joins a piece outside the vocabulary", |config| {
            config.merges.push(("a".into(), "zz".into()))
        }),
        ("a merge forms a piece outside the vocabulary", |config| {
            config.merges.push(("b".into(), "c".into()))
        }),
        ("a split pattern does not compile", |config| {
            config.splits[0].pattern = "(".into()
        }),
        ("a piece repeats", |config| {
            config.pieces[256] = config.pieces[0].clone()
        }),
        ("a byte is not spelled", |config| {
            config.pieces[0] = "zz".into()
        }),
        ("a stop token is outside the vocabulary", |config| {
            config.stop_tokens.insert(TokenId(9_999));
        }),
        ("the artifact is unnamed", |config| {
            config.artifact_identity.clear()
        }),
    ];
    for (case, change) in cases {
        let mut config = config();
        change(&mut config);
        assert_eq!(verdicts(config), (None, None), "{case}");
    }
}
