use aho_corasick::{AhoCorasick, MatchKind};
use magnitude_artifacts::UnsupportedReason;
use magnitude_generation::TokenId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Range;
use tokenizers::{
    models::bpe::BPE,
    normalizers::{replace::Replace, unicode::NFC, NormalizerWrapper},
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split as SplitStage, SplitPattern},
        PreTokenizerWrapper,
    },
    AddedToken, Encoding, SplitDelimiterBehavior, Tokenizer,
};

/// A model input sequence's tokens, and where its added tokens end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedSequence {
    pub tokens: Vec<TokenId>,
    /// Every added token (a control or user-defined piece), in order. The
    /// text is split at added tokens before anything else is encoded, so
    /// the tokens before an added token's end are the same for every text
    /// sharing the bytes before it.
    pub added_token_ends: Vec<AddedTokenEnd>,
}

/// Where an added token ends: its text's end offset in bytes, and the
/// sequence position after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddedTokenEnd {
    pub offset: usize,
    pub position: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialTokens {
    Recognize,
    Literal,
}

/// Why a container's tokenizer cannot be adapted or constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    /// The container declares a tokenizer scheme no implemented profile
    /// defines, or sets metadata whose semantics the engine does not
    /// implement: every such cause at once.
    Unsupported(Vec<UnsupportedReason>),
    /// The tokenizer payload or configuration is malformed or inconsistent.
    Invalid(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reasons) => {
                formatter.write_str("unsupported GGUF tokenizer: ")?;
                for (index, reason) in reasons.iter().enumerate() {
                    if index > 0 {
                        formatter.write_str(", ")?;
                    }
                    reason.fmt(formatter)?;
                }
                Ok(())
            }
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<&str> for TokenizerError {
    fn from(message: &str) -> Self {
        Self::Invalid(message.into())
    }
}

impl From<String> for TokenizerError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

/// Normalization of text outside special tokens, before splitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Normalization {
    None,
    /// Unicode canonical composition.
    Nfc,
    /// Every space becomes U+2581, the SentencePiece word marker.
    Metaspace,
}

/// How a split pattern's matches become pieces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitBehavior {
    /// Each match and each run of text between matches is its own piece.
    Isolated,
    /// Each match joins the text that follows it.
    MergedWithNext,
}

/// One regular-expression split stage. Stages apply in order, each to every
/// piece the previous stage produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub pattern: String,
    pub behavior: SplitBehavior,
}

/// How normal vocabulary pieces spell text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PieceEncoding {
    /// The GPT-2 byte-level alphabet: every byte is one printable character,
    /// and every byte is a base piece.
    ByteLevel,
    /// UTF-8 text in which U+2581 spells a space. Text outside the vocabulary
    /// is spelled by the `<0xXX>` byte pieces, which cover every byte.
    MetaspaceByteFallback,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BpeConfig {
    pub artifact_identity: String,
    pub pieces: Vec<String>,
    pub kinds: Vec<PieceKind>,
    pub merges: Vec<(String, String)>,
    pub normalization: Normalization,
    pub splits: Vec<Split>,
    pub encoding: PieceEncoding,
    /// A split piece that is itself a vocabulary piece is one token, without
    /// applying merges.
    pub ignore_merges: bool,
    /// The token every model input sequence begins with. It is part of the
    /// sequence exactly once: inserted unless the encoded text already begins
    /// with it.
    pub implicit_bos: Option<TokenId>,
    pub stop_tokens: BTreeSet<TokenId>,
    /// Tokens the model must never generate.
    pub suppressed_tokens: BTreeSet<TokenId>,
}

pub struct ByteBpeTokenizer {
    encoders: [Tokenizer; 2],
    /// The control pieces' spellings, matched leftmost-longest as the
    /// recognizing encoder splits them, and their tokens.
    controls: AhoCorasick,
    control_tokens: Vec<TokenId>,
    pieces: Vec<Vec<u8>>,
    kinds: Vec<PieceKind>,
    implicit_bos: Option<TokenId>,
    stop_tokens: BTreeSet<TokenId>,
    suppressed_tokens: BTreeSet<TokenId>,
    identity: String,
    artifact_identity: String,
}

/// The GPT-2 byte-level alphabet: each byte's printable stand-in character.
fn byte_level_alphabet() -> HashMap<char, u8> {
    let mut values: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut codes: Vec<u32> = values.iter().map(|&b| u32::from(b)).collect();
    let mut next = 256;
    for byte in 0..=255 {
        if !values.contains(&byte) {
            values.push(byte);
            codes.push(next);
            next += 1;
        }
    }
    codes
        .into_iter()
        .zip(values)
        .map(|(c, b)| (char::from_u32(c).unwrap(), b))
        .collect()
}

/// The byte a byte-fallback piece spells. Only the canonical `<0xXX>` form
/// (uppercase hex) is one, since byte fallback looks pieces up by it.
fn fallback_byte(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    let byte = u8::from_str_radix(hex, 16).ok()?;
    (piece == format!("<0x{byte:02X}>")).then_some(byte)
}

impl ByteBpeTokenizer {
    pub fn new(config: BpeConfig) -> Result<Self, TokenizerError> {
        let pieces = config.checked_pieces()?;
        let model = BPE::builder()
            .vocab_and_merges(
                config
                    .pieces
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.clone(), i as u32))
                    .collect::<tokenizers::models::bpe::Vocab>(),
                config.merges.clone(),
            )
            .fuse_unk(false)
            .byte_fallback(config.encoding == PieceEncoding::MetaspaceByteFallback)
            .ignore_merges(config.ignore_merges)
            .build()
            .map_err(|e| TokenizerError::Invalid(e.to_string()))?;
        let normalizer: Option<NormalizerWrapper> = match config.normalization {
            Normalization::None => None,
            Normalization::Nfc => Some(NFC.into()),
            Normalization::Metaspace => Some(
                Replace::new(" ", "\u{2581}")
                    .map_err(|e| TokenizerError::Invalid(e.to_string()))?
                    .into(),
            ),
        };
        let mut stages = config.split_stages()?;
        if config.encoding == PieceEncoding::ByteLevel {
            stages.push(ByteLevel::new(false, false, false).into());
        }
        let make = |recognize| -> Result<Tokenizer, TokenizerError> {
            let mut encoder = Tokenizer::new(model.clone());
            encoder.with_normalizer(normalizer.clone());
            if !stages.is_empty() {
                encoder.with_pre_tokenizer(Some(Sequence::new(stages.clone())));
            }
            for (i, (piece, kind)) in config.pieces.iter().zip(&config.kinds).enumerate() {
                if *kind == PieceKind::UserDefined || (recognize && *kind == PieceKind::Control) {
                    encoder.add_tokens(&[AddedToken::from(
                        piece.clone(),
                        *kind == PieceKind::Control,
                    )
                    .normalized(false)]);
                    if encoder.token_to_id(piece) != Some(i as u32) {
                        return Err("tokenizer construction changed a token ID".into());
                    }
                }
            }
            Ok(encoder)
        };
        let encoders = [make(true)?, make(false)?];
        let (control_pieces, control_tokens): (Vec<&str>, Vec<TokenId>) = config
            .pieces
            .iter()
            .zip(&config.kinds)
            .enumerate()
            .filter(|(_, (_, kind))| **kind == PieceKind::Control)
            .map(|(id, (piece, _))| (piece.as_str(), TokenId(id as u32)))
            .unzip();
        let controls = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&control_pieces)
            .map_err(|e| TokenizerError::Invalid(e.to_string()))?;
        let identity = Sha256::digest(
            serde_json::to_vec(&config).map_err(|e| TokenizerError::Invalid(e.to_string()))?,
        )
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
        Ok(Self {
            encoders,
            controls,
            control_tokens,
            pieces,
            kinds: config.kinds,
            implicit_bos: config.implicit_bos,
            stop_tokens: config.stop_tokens,
            suppressed_tokens: config.suppressed_tokens,
            identity,
            artifact_identity: config.artifact_identity,
        })
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }
    pub fn vocabulary(&self) -> usize {
        self.pieces.len()
    }
}

impl BpeConfig {
    /// The split stages the pre-tokenizer applies before any byte-level
    /// stage.
    fn split_stages(&self) -> Result<Vec<PreTokenizerWrapper>, TokenizerError> {
        self.splits
            .iter()
            .map(|split| {
                SplitStage::new(
                    SplitPattern::Regex(split.pattern.clone()),
                    match split.behavior {
                        SplitBehavior::Isolated => SplitDelimiterBehavior::Isolated,
                        SplitBehavior::MergedWithNext => SplitDelimiterBehavior::MergedWithNext,
                    },
                    false,
                )
                .map(PreTokenizerWrapper::from)
                .map_err(|e| TokenizerError::Invalid(e.to_string()))
            })
            .collect()
    }

    /// Every piece's bytes, after checking the vocabulary and merges: the
    /// artifact identity is named, pieces are nonempty and unique, special
    /// tokens are in the vocabulary, every piece decodes in its encoding,
    /// every byte is spelled, and every merge joins two vocabulary pieces
    /// into a vocabulary piece.
    fn checked_pieces(&self) -> Result<Vec<Vec<u8>>, TokenizerError> {
        let config = self;
        let in_vocabulary = |id: &TokenId| (id.0 as usize) < config.pieces.len();
        let vocabulary = config.pieces.iter().map(String::as_str).collect::<HashSet<_>>();
        if config.artifact_identity.is_empty()
            || config.pieces.is_empty()
            || config.pieces.len() > i32::MAX as usize
            || config.pieces.len() != config.kinds.len()
            || config.pieces.iter().any(String::is_empty)
            || vocabulary.len() != config.pieces.len()
            || !config.stop_tokens.iter().all(in_vocabulary)
            || !config.suppressed_tokens.iter().all(in_vocabulary)
            || !config.implicit_bos.iter().all(in_vocabulary)
        {
            return Err("invalid BPE vocabulary or artifact identity".into());
        }
        let mut merged = String::new();
        for (left, right) in &config.merges {
            merged.clear();
            merged.push_str(left);
            merged.push_str(right);
            if !vocabulary.contains(left.as_str())
                || !vocabulary.contains(right.as_str())
                || !vocabulary.contains(merged.as_str())
            {
                return Err(TokenizerError::Invalid(format!(
                    "BPE merge ({left}, {right}) is outside the vocabulary"
                )));
            }
        }
        let alphabet = byte_level_alphabet();
        let pieces = config
            .pieces
            .iter()
            .zip(&config.kinds)
            .map(|(piece, kind)| match (kind, config.encoding) {
                (PieceKind::Normal, PieceEncoding::ByteLevel) => piece
                    .chars()
                    .map(|c| {
                        alphabet
                            .get(&c)
                            .copied()
                            .ok_or("normal BPE piece is outside the byte alphabet")
                    })
                    .collect(),
                (PieceKind::Normal, PieceEncoding::MetaspaceByteFallback) => {
                    Ok(piece.replace('\u{2581}', " ").into_bytes())
                }
                (PieceKind::Byte, PieceEncoding::MetaspaceByteFallback) => fallback_byte(piece)
                    .map(|byte| vec![byte])
                    .ok_or("byte-fallback piece is not <0xXX>"),
                (PieceKind::Control | PieceKind::UserDefined | PieceKind::Unused, _) => {
                    Ok(piece.as_bytes().to_vec())
                }
                (PieceKind::Byte, PieceEncoding::ByteLevel) => {
                    Err("byte-level BPE does not have byte-fallback pieces")
                }
                (PieceKind::Unknown, _) => Err("BPE does not implement unknown pieces"),
            })
            .collect::<Result<Vec<Vec<u8>>, &str>>()?;
        // BPE without an unknown token silently omits text its vocabulary
        // cannot spell. Require complete byte coverage before accepting input.
        let spelled_bytes: HashSet<u8> = pieces
            .iter()
            .zip(&config.kinds)
            .filter(|(piece, kind)| {
                piece.len() == 1
                    && **kind
                        == match config.encoding {
                            PieceEncoding::ByteLevel => PieceKind::Normal,
                            PieceEncoding::MetaspaceByteFallback => PieceKind::Byte,
                        }
            })
            .map(|(piece, _)| piece[0])
            .collect();
        if spelled_bytes.len() != 256 {
            return Err("BPE vocabulary must spell every byte".into());
        }
        Ok(pieces)
    }
}

impl ByteBpeTokenizer {
    pub fn kind(&self, token: TokenId) -> Result<PieceKind, String> {
        self.kinds
            .get(token.0 as usize)
            .copied()
            .ok_or_else(|| "token is outside the vocabulary".into())
    }
    pub fn stop_tokens(&self) -> &BTreeSet<TokenId> {
        &self.stop_tokens
    }
    /// Tokens the model must never generate.
    pub fn suppressed_tokens(&self) -> &BTreeSet<TokenId> {
        &self.suppressed_tokens
    }
    /// The token every model input sequence begins with, when the tokenizer
    /// inserts one.
    pub fn implicit_bos(&self) -> Option<TokenId> {
        self.implicit_bos
    }
    /// Encode a text fragment. No sequence-start token is inserted.
    pub fn encode(&self, text: &str, special: SpecialTokens) -> Result<Vec<TokenId>, String> {
        Ok(self
            .encoding(text, special)?
            .get_ids()
            .iter()
            .map(|&id| TokenId(id))
            .collect())
    }

    /// Encode a text fragment whose template-authored byte ranges are
    /// `controls` (ascending, disjoint, on character boundaries), with each
    /// token's byte range of `text`: a control token is recognized only
    /// wholly inside one of them, and every other byte is literal text. The text is split at the recognized control
    /// tokens and nothing else, exactly as the recognizing encoder splits it,
    /// so the tokens equal [`Self::encode`] with
    /// [`SpecialTokens::Recognize`] whenever no control token's spelling
    /// occurs outside `controls` (nor straddles one's edge). User-defined
    /// pieces are recognized everywhere, as in both modes. No sequence-start
    /// token is inserted.
    pub fn encode_controlled_offsets(
        &self,
        text: &str,
        controls: &[Range<usize>],
    ) -> Result<Vec<(TokenId, Range<usize>)>, String> {
        let mut end = 0;
        for range in controls {
            if range.start < end
                || range.start > range.end
                || !text.is_char_boundary(range.start)
                || !text.is_char_boundary(range.end)
            {
                return Err("control ranges must be ascending, disjoint character ranges of the text".into());
            }
            end = range.end;
        }
        let mut tokens = Vec::new();
        let mut literal = 0;
        let push_literal = |tokens: &mut Vec<_>, range: Range<usize>| -> Result<(), String> {
            let encoding = self.encoding(&text[range.clone()], SpecialTokens::Literal)?;
            tokens.extend(encoding.get_ids().iter().zip(encoding.get_offsets()).map(
                |(&id, &(start, end))| (TokenId(id), range.start + start..range.start + end),
            ));
            Ok(())
        };
        for range in controls {
            for found in self.controls.find_iter(&text[range.clone()]) {
                let start = range.start + found.start();
                push_literal(&mut tokens, literal..start)?;
                literal = range.start + found.end();
                tokens.push((self.control_tokens[found.pattern().as_usize()], start..literal));
            }
        }
        push_literal(&mut tokens, literal..text.len())?;
        Ok(tokens)
    }

    fn encoding(&self, text: &str, special: SpecialTokens) -> Result<Encoding, String> {
        let index = usize::from(special == SpecialTokens::Literal);
        self.encoders[index]
            .encode(text, false)
            .map_err(|e| e.to_string())
    }
    /// Encode the complete text of a model input sequence whose
    /// template-authored byte ranges are `controls`, as
    /// [`Self::encode_controlled_offsets`] encodes them. The implicit BOS is
    /// part of the sequence exactly once: a sequence whose text already
    /// begins with it (a chat template that renders the BOS text) is not
    /// given a second one.
    /// Every added token's end is reported.
    pub fn encode_sequence_controlled(
        &self,
        text: &str,
        controls: &[Range<usize>],
    ) -> Result<EncodedSequence, String> {
        let encoded = self.encode_controlled_offsets(text, controls)?;
        let bos = self
            .implicit_bos
            .filter(|bos| encoded.first().map(|(token, _)| token) != Some(bos));
        let inserted = usize::from(bos.is_some());
        let added_token_ends = encoded
            .iter()
            .enumerate()
            .filter(|(_, (token, _))| {
                matches!(
                    self.kinds[token.0 as usize],
                    PieceKind::Control | PieceKind::UserDefined
                )
            })
            .map(|(index, (_, range))| AddedTokenEnd {
                offset: range.end,
                position: inserted + index + 1,
            })
            .collect();
        Ok(EncodedSequence {
            tokens: bos
                .into_iter()
                .chain(encoded.into_iter().map(|(token, _)| token))
                .collect(),
            added_token_ends,
        })
    }

    pub fn piece(&self, token: TokenId, skip_control: bool) -> Result<&[u8], String> {
        let id = token.0 as usize;
        let piece = self
            .pieces
            .get(id)
            .ok_or("token is outside the vocabulary")?;
        Ok(if skip_control && self.kinds[id] == PieceKind::Control {
            &[]
        } else {
            piece
        })
    }
    pub fn decoder(&self, skip_control: bool) -> TokenDecoder<'_> {
        TokenDecoder {
            tokenizer: self,
            skip_control,
            pending: Vec::new(),
            finished: false,
        }
    }
}

impl magnitude_grammar::TokenTable for ByteBpeTokenizer {
    fn len(&self) -> usize {
        self.vocabulary()
    }
    /// Unused pieces, and byte-fallback pieces outside UTF-8, never spell
    /// grammar text.
    fn bytes(&self, token: TokenId) -> Option<&[u8]> {
        let id = token.0 as usize;
        let piece = self.pieces.get(id)?;
        (self.kinds[id] != PieceKind::Unused && !piece.starts_with(&[0xff])).then_some(piece)
    }
    fn stop_tokens(&self) -> &BTreeSet<TokenId> {
        &self.stop_tokens
    }
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
        ByteBpeTokenizer::encode(self, text, SpecialTokens::Recognize)
    }
}

/// Replacement decoding matches V3: incomplete UTF-8 waits for subsequent token
/// bytes; invalid sequences and an incomplete final suffix publish U+FFFD.
pub struct TokenDecoder<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    skip_control: bool,
    pending: Vec<u8>,
    finished: bool,
}
impl TokenDecoder<'_> {
    pub fn push(&mut self, token: TokenId) -> Result<String, String> {
        if self.finished {
            return Err("token decoder is finished".into());
        }
        self.pending
            .extend_from_slice(self.tokenizer.piece(token, self.skip_control)?);
        let mut output = String::new();
        let mut offset = 0;
        while offset < self.pending.len() {
            match std::str::from_utf8(&self.pending[offset..]) {
                Ok(text) => {
                    output.push_str(text);
                    offset = self.pending.len();
                }
                Err(error) => {
                    let valid = offset + error.valid_up_to();
                    output.push_str(std::str::from_utf8(&self.pending[offset..valid]).unwrap());
                    offset = valid;
                    if let Some(length) = error.error_len() {
                        output.push('\u{fffd}');
                        offset += length;
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending = self.pending[offset..].to_vec();
        Ok(output)
    }
    pub fn finish(&mut self) -> Result<String, String> {
        if self.finished {
            return Err("token decoder is finished".into());
        }
        self.finished = true;
        Ok(String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned())
    }
}
