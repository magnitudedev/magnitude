use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use tokenizers::{
    models::bpe::BPE,
    normalizers::unicode::NFC,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
    AddedToken, SplitDelimiterBehavior, Tokenizer,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenId(pub u32);

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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BpeConfig {
    pub artifact_identity: String,
    pub pieces: Vec<String>,
    pub kinds: Vec<PieceKind>,
    pub merges: Vec<(String, String)>,
    pub pattern: String,
    pub normalize_nfc: bool,
    pub stop_tokens: BTreeSet<TokenId>,
}

pub struct ByteBpeTokenizer {
    encoders: [Tokenizer; 2],
    pieces: Vec<Vec<u8>>,
    kinds: Vec<PieceKind>,
    stop_tokens: BTreeSet<TokenId>,
    identity: String,
    artifact_identity: String,
}
impl ByteBpeTokenizer {
    pub fn new(config: BpeConfig) -> Result<Self, String> {
        if config.artifact_identity.is_empty()
            || config.pieces.is_empty()
            || config.pieces.len() > u32::MAX as usize
            || config.pieces.len() != config.kinds.len()
            || config.pieces.iter().any(String::is_empty)
            || config.pieces.iter().collect::<HashSet<_>>().len() != config.pieces.len()
            || config
                .stop_tokens
                .iter()
                .any(|id| id.0 as usize >= config.pieces.len())
        {
            return Err("invalid byte BPE vocabulary or artifact identity".into());
        }
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
        let alphabet: HashMap<char, u8> = codes
            .into_iter()
            .zip(values)
            .map(|(c, b)| (char::from_u32(c).unwrap(), b))
            .collect();
        let pieces = config
            .pieces
            .iter()
            .zip(&config.kinds)
            .map(|(piece, kind)| match kind {
                PieceKind::Normal => piece
                    .chars()
                    .map(|c| {
                        alphabet
                            .get(&c)
                            .copied()
                            .ok_or_else(|| "normal BPE piece is outside the byte alphabet".into())
                    })
                    .collect(),
                PieceKind::Control | PieceKind::UserDefined | PieceKind::Unused => {
                    Ok(piece.as_bytes().to_vec())
                }
                _ => Err("byte BPE does not implement unknown or byte-fallback pieces".into()),
            })
            .collect::<Result<Vec<Vec<u8>>, String>>()?;
        // BPE without an unknown token silently omits bytes missing from its
        // alphabet. Require complete base-byte coverage before accepting input.
        let base_bytes: HashSet<u8> = pieces
            .iter()
            .zip(&config.kinds)
            .filter(|(piece, kind)| **kind == PieceKind::Normal && piece.len() == 1)
            .map(|(piece, _)| piece[0])
            .collect();
        if base_bytes.len() != 256 {
            return Err("byte BPE vocabulary must cover every base byte".into());
        }
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
            .byte_fallback(false)
            .build()
            .map_err(|e| e.to_string())?;
        let make = |recognize| -> Result<Tokenizer, String> {
            let mut encoder = Tokenizer::new(model.clone());
            if config.normalize_nfc {
                encoder.with_normalizer(Some(NFC));
            }
            encoder.with_pre_tokenizer(Some(Sequence::new(vec![
                Split::new(
                    SplitPattern::Regex(config.pattern.clone()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .map_err(|e| e.to_string())?
                .into(),
                ByteLevel::new(false, false, false).into(),
            ])));
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
        let identity = Sha256::digest(serde_json::to_vec(&config).map_err(|e| e.to_string())?)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        Ok(Self {
            encoders,
            pieces,
            kinds: config.kinds,
            stop_tokens: config.stop_tokens,
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
    pub fn kind(&self, token: TokenId) -> Result<PieceKind, String> {
        self.kinds
            .get(token.0 as usize)
            .copied()
            .ok_or_else(|| "token is outside the vocabulary".into())
    }
    pub fn stop_tokens(&self) -> &BTreeSet<TokenId> {
        &self.stop_tokens
    }
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
