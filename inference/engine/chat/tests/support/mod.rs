//! Shared fixtures of the chat integration tests: a byte-level tokenizer with
//! one token per byte, template bundles over one template source, and the
//! uncached grammar vocabulary of a fixture tokenizer.
#![allow(dead_code)]

use magnitude_chat::{
    BpeConfig, ByteBpeTokenizer, Normalization, PieceEncoding, PieceKind, Split, SplitBehavior,
    TemplateBundle, TemplateVariant, TokenId,
};
use magnitude_grammar::{CacheLimits, Vocabulary};
use std::{collections::BTreeSet, sync::Arc};

/// The byte-level piece of every byte 0..=255, indexed by byte.
pub fn byte_pieces() -> Vec<String> {
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
    pieces
}

/// Byte pieces 0..=255 and the control stop token `stop` 256: one token per
/// byte.
pub fn byte_tokenizer(stop: &str) -> Arc<ByteBpeTokenizer> {
    let mut pieces = byte_pieces();
    pieces.push(stop.into());
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.push(PieceKind::Control);
    Arc::new(
        ByteBpeTokenizer::new(BpeConfig {
            artifact_identity: "fixture".into(),
            pieces,
            kinds,
            merges: vec![],
            normalization: Normalization::None,
            splits: vec![Split {
                pattern: r".+|\s".into(),
                behavior: SplitBehavior::Isolated,
            }],
            encoding: PieceEncoding::ByteLevel,
            ignore_merges: false,
            implicit_bos: None,
            stop_tokens: BTreeSet::from([TokenId(256)]),
            suppressed_tokens: BTreeSet::new(),
        })
        .unwrap(),
    )
}

/// The single default template `source`, rendered with `bos` and `eos` as
/// its BOS and EOS token texts.
pub fn bundle(source: &str, bos: &str, eos: &str) -> TemplateBundle {
    TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: source.into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        [
            ("bos_token".to_string(), bos.to_string()),
            ("eos_token".to_string(), eos.to_string()),
        ]
        .into(),
    )
    .unwrap()
}

/// The grammar vocabulary of `tokenizer`, without a mask cache.
pub fn vocabulary(tokenizer: &Arc<ByteBpeTokenizer>) -> Vocabulary {
    Vocabulary::new(
        tokenizer.clone(),
        tokenizer.vocabulary(),
        CacheLimits {
            entries: 0,
            bytes: 0,
        },
    )
    .unwrap()
}
