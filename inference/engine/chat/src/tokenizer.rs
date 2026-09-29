use magnitude_generation::TokenId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use tokenizers::{
    models::bpe::BPE,
    normalizers::{replace::Replace, unicode::NFC, NormalizerWrapper},
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split as SplitStage, SplitPattern},
        PreTokenizerWrapper,
    },
    AddedToken, SplitDelimiterBehavior, Tokenizer,
};

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
    /// The container declares a tokenizer scheme (`tokenizer.ggml.model`
    /// and `tokenizer.ggml.pre`) that no implemented profile defines.
    UnsupportedProfile { model: String, pre: Option<String> },
    /// The container sets tokenizer metadata whose semantics the engine does
    /// not implement.
    UnsupportedMetadata { key: String },
    /// The tokenizer payload or configuration is malformed or inconsistent.
    Invalid(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedProfile { model, pre } => write!(
                formatter,
                "unsupported GGUF tokenizer profile: model {model:?}, pre-tokenizer {pre:?}"
            ),
            Self::UnsupportedMetadata { key } => {
                write!(formatter, "unsupported GGUF tokenizer metadata: {key}")
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
        let identity = Sha256::digest(
            serde_json::to_vec(&config).map_err(|e| TokenizerError::Invalid(e.to_string()))?,
        )
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
        Ok(Self {
            encoders,
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
        let index = usize::from(special == SpecialTokens::Literal);
        Ok(self.encoders[index]
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .iter()
            .map(|&id| TokenId(id))
            .collect())
    }
    /// Encode the complete text of a model input sequence, recognizing special
    /// tokens. The implicit BOS is part of the sequence exactly once: a
    /// sequence whose text already begins with it (a chat template that
    /// renders the BOS text) is not given a second one.
    pub fn encode_sequence(&self, text: &str) -> Result<Vec<TokenId>, String> {
        let mut tokens = self.encode(text, SpecialTokens::Recognize)?;
        if let Some(bos) = self.implicit_bos {
            if tokens.first() != Some(&bos) {
                tokens.insert(0, bos);
            }
        }
        Ok(tokens)
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
