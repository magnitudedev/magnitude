//! Rendered prompts: text with the ranges the template authored (control
//! ranges) and the images it places, its tokenization, and the escape
//! request text takes before it is rendered.

use crate::description::{EscapeMethod, Spellings};
use crate::json::python_strip;
use magnitude_artifacts::TokenId;
use magnitude_chat::{ByteBpeTokenizer, PieceKind};
use std::ops::Range;

/// How request text spelling one of the tokenizer's control tokens is
/// changed before it is rendered, as the model was trained to read it.
/// Literal tokenization already keeps request text from becoming a control
/// token; an escape changes the text itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Escape {
    None,
    /// Each spelling, longest first, has its `<` and `>` replaced by the
    /// full-width `＜` and `＞` (pplx).
    FullWidth(Vec<String>),
    /// A zero-width space after the first character of each spelling,
    /// longest first, repeated until none is left (d1).
    ZeroWidth(Vec<String>),
}

const ZERO_WIDTH_SPACE: char = '\u{200b}';

impl Escape {
    /// The escape `method` names over its spellings, longest first: the
    /// listed ones, or the tokenizer's control-token spellings at least
    /// `min_length` characters long. The zero-width escape leaves a spelling
    /// already holding the zero-width space, which inserting one would
    /// never clear.
    pub fn of(method: &EscapeMethod, tokenizer: &ByteBpeTokenizer) -> Result<Self, String> {
        let spellings = |spellings: &Spellings| -> Result<Vec<String>, String> {
            let min_length = match spellings {
                Spellings::Listed(listed) => return Ok(listed.clone()),
                Spellings::Control { min_length } => *min_length,
            };
            let mut spellings = Vec::new();
            for id in 0..tokenizer.vocabulary() {
                let token = TokenId(id as u32);
                if tokenizer.kind(token)? == PieceKind::Control {
                    let spelling = String::from_utf8(tokenizer.piece(token, false)?.to_vec())
                        .map_err(|error| error.to_string())?;
                    if spelling.chars().count() >= min_length {
                        spellings.push(spelling);
                    }
                }
            }
            Ok(spellings)
        };
        Ok(match method {
            EscapeMethod::None => Self::None,
            EscapeMethod::FullWidth(listed) => Self::FullWidth(longest_first(spellings(listed)?)),
            EscapeMethod::ZeroWidth(listed) => Self::ZeroWidth(longest_first(
                spellings(listed)?
                    .into_iter()
                    .filter(|spelling| !spelling.contains(ZERO_WIDTH_SPACE)),
            )),
        })
    }

    pub fn apply(&self, text: &str) -> String {
        match self {
            Self::None => text.into(),
            Self::FullWidth(spellings) => {
                spellings.iter().fold(text.to_owned(), |text, spelling| {
                    text.replace(spelling, &spelling.replace('<', "＜").replace('>', "＞"))
                })
            }
            Self::ZeroWidth(_) => self.apply_with_boundaries(text).0,
        }
    }

    /// The escaped text, and for each of its character boundaries the
    /// boundary of `text` it stands at: an inserted character takes the
    /// position of the one after it.
    pub fn apply_with_boundaries(&self, text: &str) -> (String, Vec<usize>) {
        let identity = |text: &str| (0..=text.chars().count()).collect();
        match self {
            // Each replaces characters one for one.
            Self::None | Self::FullWidth(_) => (self.apply(text), identity(text)),
            Self::ZeroWidth(spellings) => {
                let mut text = text.to_owned();
                let mut boundaries: Vec<usize> = identity(&text);
                while let Some((escaped, inserted)) = break_once(&text, spellings) {
                    for at in inserted.into_iter().rev() {
                        boundaries.insert(at, boundaries[at]);
                    }
                    text = escaped;
                }
                (text, boundaries)
            }
        }
    }
}

fn longest_first(spellings: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut spellings: Vec<String> = spellings.into_iter().collect();
    spellings.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()).then(a.cmp(b)));
    spellings.dedup();
    spellings
}

/// One pass of Python's `re.subn` over the alternation of `spellings` (the
/// first alternative matching at the leftmost position wins), breaking each
/// match after its first character, and the character positions of the
/// text before the pass where it inserted; `None` when nothing matched.
fn break_once(text: &str, spellings: &[String]) -> Option<(String, Vec<usize>)> {
    let mut out = String::with_capacity(text.len());
    let mut inserted = Vec::new();
    let (mut at, mut characters) = (0, 0);
    while at < text.len() {
        let rest = &text[at..];
        match spellings
            .iter()
            .find(|spelling| rest.starts_with(spelling.as_str()))
        {
            Some(spelling) => {
                let first = spelling.chars().next().map_or(0, char::len_utf8);
                out.push_str(&spelling[..first]);
                out.push(ZERO_WIDTH_SPACE);
                out.push_str(&spelling[first..]);
                inserted.push(characters + 1);
                at += spelling.len();
                characters += spelling.chars().count();
            }
            None => {
                let next = rest.chars().next().map_or(1, char::len_utf8);
                out.push_str(&rest[..next]);
                at += next;
                characters += 1;
            }
        }
    }
    (!inserted.is_empty()).then_some((out, inserted))
}

/// A rendered prompt: its text, the template-authored ranges, and the
/// request images its placeholders stand for, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    pub controls: Vec<Range<usize>>,
    pub images: Vec<usize>,
}

/// A rendered prompt's tokens and each token's byte range of its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    pub tokens: Vec<TokenId>,
    pub ranges: Vec<Range<usize>>,
}

impl Rendered {
    pub fn encode(&self, tokenizer: &ByteBpeTokenizer) -> Result<Encoded, String> {
        let (tokens, ranges) = tokenizer
            .encode_controlled_offsets(&self.text, &self.controls)?
            .into_iter()
            .unzip();
        Ok(Encoded { tokens, ranges })
    }
}

impl Rendered {
    /// The tokens of the rendering's text in `range` alone, with the
    /// control ranges inside it, each token's bytes from the range's start.
    pub fn encode_range(
        &self,
        tokenizer: &ByteBpeTokenizer,
        range: Range<usize>,
    ) -> Result<Encoded, String> {
        let controls: Vec<Range<usize>> = self
            .controls
            .iter()
            .filter(|control| control.start < range.end && range.start < control.end)
            .map(|control| {
                control.start.max(range.start) - range.start
                    ..control.end.min(range.end) - range.start
            })
            .collect();
        let (tokens, ranges) = tokenizer
            .encode_controlled_offsets(&self.text[range], &controls)?
            .into_iter()
            .unzip();
        Ok(Encoded { tokens, ranges })
    }
}

impl Encoded {
    /// Leading tokens whose bytes all lie before `offset`: a token
    /// straddling it belongs after it.
    pub fn tokens_before(&self, offset: usize) -> usize {
        self.ranges
            .iter()
            .take_while(|range| range.end <= offset)
            .count()
    }

    /// Where a slot anchor at `offset` of `text` reads: the token holding
    /// the byte before it, when the rest of that token is whitespace.
    pub fn slot_token(&self, text: &str, offset: usize) -> Option<usize> {
        self.ranges
            .iter()
            .position(|range| range.start < offset && offset <= range.end)
            .filter(|token| python_strip(&text[offset..self.ranges[*token].end]).is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_follow_their_references() {
        let full = Escape::FullWidth(longest_first(["<|im".into(), "<|im_end|>".into()]));
        assert_eq!(full.apply("a<|im_end|>b<|im"), "a＜|im_end|＞b＜|im");
        let zero = Escape::ZeroWidth(longest_first(["<|im_end|>".into(), "<|im_start|>".into()]));
        assert_eq!(zero.apply("<|im_end|>"), "<\u{200b}|im_end|>");
        assert_eq!(zero.apply("plain"), "plain");
        let (escaped, boundaries) = zero.apply_with_boundaries("x<|im_end|>");
        assert_eq!(escaped, "x<\u{200b}|im_end|>");
        // `x` `<` then the inserted space at the position of `|`.
        assert_eq!(boundaries, vec![0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }
}
