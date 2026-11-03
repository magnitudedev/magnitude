//! Answer codes: what a slot reads for each option, its code (the string
//! the template lists it under) and the tokens it is read at, its spellings
//! (`KEYS.md`, "Codes"). An option's value is the best of its spellings'.

use crate::json::python_strip;
use crate::render::Rendered;
use magnitude_artifacts::TokenId;
use magnitude_chat::{ByteBpeTokenizer, SpecialTokens};

/// `A`..`Z`, then `AA`, `AB`, .. `ZZ`: every candidate in preference order.
pub(crate) fn letter_candidates() -> impl Iterator<Item = String> {
    let letters = || (b'A'..=b'Z').map(char::from);
    letters().map(String::from).chain(
        letters().flat_map(move |first| letters().map(move |second| format!("{first}{second}"))),
    )
}

/// The fallback pool an alias is taken from: `A`..`Z`, `00`..`99`,
/// `a`..`z`, `#0`..`#199`, then `AA`..`ZZ`.
fn fallback_pool() -> impl Iterator<Item = String> {
    let upper = || (b'A'..=b'Z').map(char::from);
    upper()
        .map(String::from)
        .chain((0..100).map(|number| format!("{number:02}")))
        .chain((b'a'..=b'z').map(|letter| char::from(letter).to_string()))
        .chain((0..200).map(|number| format!("#{number}")))
        .chain(upper().flat_map(move |first| upper().map(move |second| format!("{first}{second}"))))
}

/// One slot's codes: each option's code and the tokens of its spellings, in
/// option order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CodeSet {
    codes: Vec<String>,
    tokens: Vec<TokenId>,
    /// How many of `tokens` each option reads, in order.
    groups: Vec<usize>,
}

/// The single token `text` encodes to as literal text, when it is one.
fn single_token(tokenizer: &ByteBpeTokenizer, text: &str) -> Result<Option<TokenId>, String> {
    Ok(match tokenizer.encode(text, SpecialTokens::Literal)?.as_slice() {
        [token] => Some(*token),
        _ => None,
    })
}

/// The tokens of the `texts` that are one token each, each once.
fn ids<'a>(
    tokenizer: &ByteBpeTokenizer,
    texts: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<TokenId>, String> {
    let mut out = Vec::new();
    for text in texts {
        if let Some(token) = single_token(tokenizer, text)? {
            if !out.contains(&token) {
                out.push(token);
            }
        }
    }
    Ok(out)
}

/// Python's `math.isclose` with `abs_tol`.
pub(crate) fn isclose(a: f64, b: f64, absolute: f64) -> bool {
    (a - b).abs() <= (1e-9 * a.abs().max(b.abs())).max(absolute)
}

impl CodeSet {
    fn push(&mut self, code: String, spellings: Vec<TokenId>) {
        self.codes.push(code);
        self.groups.push(spellings.len());
        self.tokens.extend(spellings);
    }

    /// Codes each read at one value by position, as a test reads them.
    #[cfg(test)]
    pub fn positional(codes: &[&str]) -> Self {
        Self {
            codes: codes.iter().map(|code| (*code).to_owned()).collect(),
            tokens: (0..codes.len() as u32).map(TokenId).collect(),
            groups: vec![1; codes.len()],
        }
    }

    /// How many options.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn codes(&self) -> &[String] {
        &self.codes
    }

    /// Every option's spellings' tokens, in option order.
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn groups(&self) -> &[usize] {
        &self.groups
    }

    /// The first `count` options' codes.
    pub fn prefix(&self, count: usize) -> Self {
        let tokens = self.groups[..count].iter().sum();
        Self {
            codes: self.codes[..count].to_vec(),
            tokens: self.tokens[..tokens].to_vec(),
            groups: self.groups[..count].to_vec(),
        }
    }

    /// `code`'s token, when it is one token no option of these reads: what
    /// every code selection keeps.
    fn fresh(&self, tokenizer: &ByteBpeTokenizer, code: &str) -> Result<Option<TokenId>, String> {
        Ok(single_token(tokenizer, code)?.filter(|token| !self.tokens.contains(token)))
    }

    /// `codes`, each one token distinct from the others', or the refusal
    /// `why` gives the first that is not.
    fn each_fresh(
        tokenizer: &ByteBpeTokenizer,
        codes: impl IntoIterator<Item = String>,
        why: impl Fn(&str) -> String,
    ) -> Result<Self, String> {
        let mut set = Self::default();
        for code in codes {
            let token = set.fresh(tokenizer, &code)?.ok_or_else(|| why(&code))?;
            set.push(code, vec![token]);
        }
        Ok(set)
    }

    /// `codes.sequence = letters`: the first `count` letter codes that are
    /// a single token not taken by an earlier code and, after `probe` (the
    /// template's probe), still that one token.
    pub fn letters(
        tokenizer: &ByteBpeTokenizer,
        count: usize,
        probe: Option<&Rendered>,
    ) -> Result<Self, String> {
        let probe = probe
            .map(|rendered| {
                rendered
                    .encode(tokenizer)
                    .map(|encoded| (rendered, encoded.tokens))
            })
            .transpose()?;
        let mut codes = Self::default();
        for candidate in letter_candidates() {
            if codes.len() == count {
                break;
            }
            let Some(token) = codes.fresh(tokenizer, &candidate)? else {
                continue;
            };
            if let Some((rendered, tokens)) = &probe {
                let mut after = (*rendered).clone();
                after.text.push_str(&candidate);
                let mut expected = tokens.clone();
                expected.push(token);
                if after.encode(tokenizer)?.tokens != expected {
                    continue;
                }
            }
            codes.push(candidate, vec![token]);
        }
        if codes.len() < count {
            return Err(format!(
                "the tokenizer yields only {} letter codes of one token each; the description reads {count}",
                codes.len()
            ));
        }
        Ok(codes)
    }

    /// `labels`, each a single token distinct from the others'.
    pub fn labels(tokenizer: &ByteBpeTokenizer, labels: &[String]) -> Result<Self, String> {
        Self::each_fresh(tokenizer, labels.iter().cloned(), |label| {
            format!("decision metadata `labels` lists `{label}`, which is not one token of its own")
        })
    }

    /// `codes.sequence = aliases` over `wanted`, each option's code in
    /// order: kept when it is one token no earlier option took, else the
    /// first such code of the fallback pool; each also read after a space
    /// when that is another single token.
    pub fn aliases(
        tokenizer: &ByteBpeTokenizer,
        wanted: impl IntoIterator<Item = String>,
    ) -> Result<Self, String> {
        let mut taken = Self::default();
        for code in wanted {
            let mut candidates = std::iter::once(code).chain(fallback_pool());
            let (code, token) = loop {
                let Some(code) = candidates.next() else {
                    return Err(format!(
                        "no single-token code is left for option {}",
                        taken.len() + 1
                    ));
                };
                if let Some(token) = taken.fresh(tokenizer, &code)? {
                    break (code, token);
                }
            };
            taken.push(code, vec![token]);
        }
        let mut codes = Self::default();
        for (code, token) in taken.codes.into_iter().zip(taken.tokens) {
            let mut spellings = vec![token];
            spellings.extend(
                single_token(tokenizer, &format!(" {code}"))?.filter(|spaced| *spaced != token),
            );
            codes.push(code, spellings);
        }
        Ok(codes)
    }

    /// `noul.codes = yes_no`: `yes`, then `no`, each in its spellings that
    /// are one token.
    pub fn yes_no(tokenizer: &ByteBpeTokenizer) -> Result<Self, String> {
        let mut codes = Self::default();
        for word in [["yes", "Yes", "YES"], ["no", "No", "NO"]] {
            let spellings = ids(tokenizer, word)?;
            if spellings.is_empty() {
                return Err(format!(
                    "`noul.codes = yes_no`: the tokenizer writes no spelling of `{}` as one token",
                    word[0]
                ));
            }
            codes.push(word[0].to_owned(), spellings);
        }
        Ok(codes)
    }

    /// `score.codes = digits`: `0` to `count - 1`, each its own token.
    pub fn digits(tokenizer: &ByteBpeTokenizer, count: usize) -> Result<Self, String> {
        Self::each_fresh(tokenizer, (0..count).map(|digit| digit.to_string()), |digit| {
            format!("`score.codes = digits`: the tokenizer does not write `{digit}` as a token of its own")
        })
    }

    /// `<kind>.numbers`: each value as the integer it is, plain and
    /// after a space, when every value is an integer that is one token.
    pub fn numbers(
        tokenizer: &ByteBpeTokenizer,
        values: impl IntoIterator<Item = f64>,
    ) -> Result<Option<Self>, String> {
        let mut codes = Self::default();
        for value in values {
            let number = value.round_ties_even();
            if !isclose(number, value, 1e-6) {
                return Ok(None);
            }
            let written = format!("{}", number as i64);
            let spellings = ids(tokenizer, [written.as_str(), &format!(" {written}")])?;
            if spellings.is_empty() || single_token(tokenizer, &written)?.is_none() {
                return Ok(None);
            }
            codes.push(written, spellings);
        }
        Ok(Some(codes))
    }
}

/// A question's own keys as its codes under `codes.keys_if_letters`, when
/// every key, stripped, is one letter.
pub(crate) fn letter_keys<'k>(keys: impl IntoIterator<Item = &'k str>) -> Option<Vec<String>> {
    let stripped: Vec<&str> = keys.into_iter().map(python_strip).collect();
    stripped
        .iter()
        .all(|key| {
            let mut characters = key.chars();
            characters.next().is_some_and(char::is_alphabetic) && characters.next().is_none()
        })
        .then(|| stripped.into_iter().map(str::to_owned).collect())
}
