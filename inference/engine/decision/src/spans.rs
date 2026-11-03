//! Spans: a read points at state positions, a start and an end over the
//! tokens that write the state's text, with position 0 (the sequence start)
//! for nothing (d1-2B's `pointer.py` and `json_spans.py`).

use crate::answers::ReadSpan;
use crate::json::{python_space, python_strip};
use std::ops::Range;

/// `StatePositions`: each sequence position's character range in the
/// state's rendered text, for the positions that write it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StatePositions {
    /// `state_text`: what a span's offsets index, as characters.
    text: Vec<char>,
    /// Through the last position that writes the state (`pointer_mask`);
    /// position 0 is the nothing readout.
    offsets: Vec<Option<(usize, usize)>>,
    /// Each JSON string value's character range and pointer, for a JSON
    /// state.
    paths: Vec<(usize, usize, String)>,
}

impl StatePositions {
    /// `state_positions`: `full` is the rendered prompt and `tokens` its
    /// tokens' byte ranges; the escaped state starts at byte `start`, and
    /// `boundaries` map its characters to those of `text` (the state
    /// unescaped, without the blank line after it). A token writes the
    /// state when it overlaps the state's text, what it holds outside it is
    /// whitespace, and the text it covers is not.
    pub fn new(
        full: &str,
        tokens: &[Range<usize>],
        start: usize,
        text: &str,
        boundaries: &[usize],
        json: bool,
    ) -> Self {
        let characters: Vec<char> = full.chars().collect();
        // Each byte offset's character, a byte inside a character rounding
        // down for a start and up for an end, as the tokenizer's offsets
        // do.
        let mut floor = vec![0; full.len() + 1];
        let mut ceil = vec![0; full.len() + 1];
        let mut index = 0;
        for (at, character) in full.char_indices() {
            for byte in at..at + character.len_utf8() {
                floor[byte] = index;
                ceil[byte] = if byte == at { index } else { index + 1 };
            }
            index += 1;
        }
        floor[full.len()] = index;
        ceil[full.len()] = index;
        let start = floor[start];
        // `bisect_right(boundaries, len(text)) - 1`: where the state's text
        // ends in the escaped state.
        let state: Vec<char> = text.chars().collect();
        let stop = start + boundaries.partition_point(|at| *at <= state.len()) - 1;
        let blank = |range: Range<usize>| {
            python_strip(&characters[range].iter().collect::<String>()).is_empty()
        };
        let mut offsets: Vec<Option<(usize, usize)>> = tokens
            .iter()
            .map(|range| {
                let (a, b) = (floor[range.start], ceil[range.end]);
                let (low, high) = (a.max(start), b.min(stop));
                if low >= high || !blank(a..low) || !blank(high..b) {
                    return None;
                }
                let (left, right) = (boundaries[low - start], boundaries[high - start]);
                (left < right
                    && right <= state.len()
                    && !python_strip(&state[left..right].iter().collect::<String>()).is_empty())
                .then_some((left, right))
            })
            .collect();
        let last = offsets.iter().rposition(Option::is_some).unwrap_or(0);
        offsets.truncate(last + 1);
        Self {
            paths: if json {
                string_paths(&state)
            } else {
                Vec::new()
            },
            text: state,
            offsets,
        }
    }

    /// The state's text, which a span's offsets index and its value is
    /// parsed in.
    pub fn context(&self) -> String {
        self.text.iter().collect()
    }

    /// Positions the pointer scores: 0 and those writing the state.
    pub fn width(&self) -> usize {
        self.offsets.len()
    }

    pub fn valid(&self, position: usize) -> bool {
        position == 0 || self.offsets[position].is_some()
    }

    /// The positions after 0 that the pointer scores, when there are any:
    /// the first that writes the state through the last.
    pub fn state_rows(&self) -> Option<Range<usize>> {
        let first = self.offsets.iter().position(Option::is_some)?;
        Some(first..self.offsets.len())
    }

    /// `StatePositions.read`: the span from the start position's first
    /// character to the end position's last, its boundary whitespace
    /// trimmed and clipped to the one JSON string it lies in, if one.
    pub fn read(&self, first: usize, last: usize) -> ReadSpan {
        let ((start, _), (_, end)) = (
            self.offsets[first].expect("the pointer reads positions that write the state"),
            self.offsets[last].expect("the pointer reads positions that write the state"),
        );
        let mut span = [start, end];
        while span[0] < span[1] && python_space(self.text[span[0]]) {
            span[0] += 1;
        }
        while span[1] > span[0] && python_space(self.text[span[1] - 1]) {
            span[1] -= 1;
        }
        let held: Vec<&(usize, usize, String)> = self
            .paths
            .iter()
            .filter(|(start, end, _)| *start < span[1] && span[0] < *end)
            .collect();
        if let [(start, end, path)] = held.as_slice() {
            let span = [span[0].max(*start), span[1].min(*end)];
            return ReadSpan {
                text: json_text(&self.text[span[0]..span[1]].iter().collect::<String>()),
                span,
                path: Some(path.clone()),
            };
        }
        ReadSpan {
            text: self.text[span[0]..span[1]].iter().collect(),
            span,
            path: None,
        }
    }
}

/// `pointer.spans` for one read: P(found) is one less the start
/// distribution's mass at 0; when found at one half or more, the likeliest
/// start and end at most `cap` tokens apart, else `(0, 0)`. `start` and
/// `end` score every position through `width`; invalid ones are `-inf`.
pub(crate) fn span(start: &[f64], end: &[f64], cap: usize) -> ((usize, usize), f64) {
    let log_softmax = |scores: &[f64]| {
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = scores.iter().map(|score| (score - max).exp()).sum();
        let shift = max + total.ln();
        scores
            .iter()
            .map(|score| score - shift)
            .collect::<Vec<f64>>()
    };
    let (start, end) = (log_softmax(start), log_softmax(end));
    let found = -start[0].exp_m1();
    let width = start.len();
    if width == 1 || found < 0.5 {
        return ((0, 0), found);
    }
    let cap = cap.min(width - 1);
    // The first maximum, as `argmax` and `max_pool1d` take it.
    let first_max = |values: &mut dyn Iterator<Item = (usize, f64)>| {
        values.fold((usize::MAX, f64::NEG_INFINITY), |best, (at, value)| {
            if best.0 == usize::MAX || value > best.1 {
                (at, value)
            } else {
                best
            }
        })
    };
    let (mut best, mut chosen) = (f64::NEG_INFINITY, (0, 0));
    for first in 1..width {
        let (last, window) = first_max(
            &mut (first..first + cap)
                .map(|at| (at, end.get(at).copied().unwrap_or(f64::NEG_INFINITY))),
        );
        let score = start[first] + window;
        if first == 1 || score > best {
            best = score;
            chosen = (first, last);
        }
    }
    (chosen, found)
}

/// `string_paths`: each string value's character range inside its quotes
/// and its RFC 6901 pointer, in a JSON text.
fn string_paths(text: &[char]) -> Vec<(usize, usize, String)> {
    struct Scan<'a> {
        text: &'a [char],
        at: usize,
        out: Vec<(usize, usize, String)>,
    }
    impl Scan<'_> {
        fn skip(&mut self) {
            while self.at < self.text.len() && self.text[self.at].is_whitespace() {
                self.at += 1;
            }
        }
        /// A string from its opening quote; its decoded characters.
        fn string(&mut self) -> String {
            let begin = self.at + 1;
            self.at = begin;
            while self.text[self.at] != '"' {
                self.at += if self.text[self.at] == '\\' { 2 } else { 1 };
            }
            self.at += 1;
            json_text(&self.text[begin..self.at - 1].iter().collect::<String>())
        }
        fn value(&mut self, path: &str) {
            self.skip();
            match self.text[self.at] {
                '{' => {
                    self.at += 1;
                    self.skip();
                    while self.text[self.at] != '}' {
                        let key = self.string();
                        self.skip();
                        self.at += 1; // the colon
                        let name = key.replace('~', "~0").replace('/', "~1");
                        self.value(&format!("{path}/{name}"));
                        self.skip();
                        if self.text[self.at] == ',' {
                            self.at += 1;
                            self.skip();
                        }
                    }
                    self.at += 1;
                }
                '[' => {
                    self.at += 1;
                    self.skip();
                    let mut index = 0;
                    while self.text[self.at] != ']' {
                        self.value(&format!("{path}/{index}"));
                        index += 1;
                        self.skip();
                        if self.text[self.at] == ',' {
                            self.at += 1;
                            self.skip();
                        }
                    }
                    self.at += 1;
                }
                '"' => {
                    let begin = self.at + 1;
                    self.string();
                    self.out.push((begin, self.at - 1, path.to_owned()));
                }
                _ => {
                    while self.at < self.text.len()
                        && !matches!(self.text[self.at], ',' | '}' | ']')
                        && !self.text[self.at].is_whitespace()
                    {
                        self.at += 1;
                    }
                }
            }
        }
    }
    let mut scan = Scan {
        text,
        at: 0,
        out: Vec::new(),
    };
    scan.value("");
    scan.out
}

/// `json_text`: a slice of a rendered JSON string as the characters it
/// encodes, unchanged when it cuts an escape.
fn json_text(escaped: &str) -> String {
    let mut out = String::with_capacity(escaped.len());
    let mut characters = escaped.chars();
    while let Some(character) = characters.next() {
        if character == '"' || (character as u32) < 0x20 {
            return escaped.to_owned();
        }
        if character != '\\' {
            out.push(character);
            continue;
        }
        let decoded = match characters.next() {
            Some('"') => '"',
            Some('\\') => '\\',
            Some('/') => '/',
            Some('b') => '\u{8}',
            Some('f') => '\u{c}',
            Some('n') => '\n',
            Some('r') => '\r',
            Some('t') => '\t',
            Some('u') => {
                let mut unit = || {
                    let hex: String = characters.by_ref().take(4).collect();
                    (hex.len() == 4)
                        .then(|| u32::from_str_radix(&hex, 16).ok())
                        .flatten()
                };
                let Some(high) = unit() else {
                    return escaped.to_owned();
                };
                // Python decodes a lone surrogate as itself; the writer
                // never escapes one, so a slice holds none.
                match char::from_u32(high) {
                    Some(character) => character,
                    None => return escaped.to_owned(),
                }
            }
            _ => return escaped.to_owned(),
        };
        out.push(decoded);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_strings_are_found_with_their_pointers() {
        let text: Vec<char> = "{\n  \"a/b\": \"x\\\"y\",\n  \"c\": [\n    1,\n    \"z\"\n  ]\n}"
            .chars()
            .collect();
        let paths = string_paths(&text);
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].2, "/a~1b");
        assert_eq!(
            text[paths[0].0..paths[0].1].iter().collect::<String>(),
            "x\\\"y"
        );
        assert_eq!(paths[1].2, "/c/1");
        assert_eq!(json_text("x\\\"y"), "x\"y");
        assert_eq!(json_text("x\\"), "x\\");
    }

    #[test]
    fn spans_take_the_likeliest_start_and_end_within_the_cap() {
        let none = f64::NEG_INFINITY;
        // Nothing at 0 unlikely; start 2, end 3.
        let start = [-5.0, 0.0, 3.0, 0.0, none];
        let end = [-5.0, 0.0, 0.0, 4.0, none];
        let (at, found) = span(&start, &end, 32);
        assert_eq!(at, (2, 3));
        assert!(found > 0.99);
        // A cap of one token keeps start and end together.
        assert_eq!(span(&start, &end, 1).0, (3, 3));
        // Nothing likely.
        assert_eq!(span(&[5.0, 0.0], &[5.0, 0.0], 32).0, (0, 0));
    }
}
