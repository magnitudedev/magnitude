//! Streaming output correctness helpers shared by native executors.

/// Incrementally converts token-piece bytes into valid UTF-8 without exposing an incomplete code
/// point to a downstream parser or transport.
#[derive(Debug, Default)]
pub struct Utf8Buffer {
    pending: Vec<u8>,
}

impl Utf8Buffer {
    #[must_use]
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        self.drain(false)
    }

    #[must_use]
    pub fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn drain(&mut self, final_chunk: bool) -> String {
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    output.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        // SAFETY: `valid_up_to` is supplied by `Utf8Error`.
                        output.push_str(unsafe {
                            std::str::from_utf8_unchecked(&self.pending[..valid_up_to])
                        });
                        self.pending.drain(..valid_up_to);
                    }
                    match error.error_len() {
                        Some(invalid_len) => {
                            output.push('\u{fffd}');
                            self.pending.drain(..invalid_len);
                        }
                        None if final_chunk => {
                            output.push('\u{fffd}');
                            self.pending.clear();
                            break;
                        }
                        None => break,
                    }
                }
            }
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopOutput {
    pub text: String,
    pub matched: Option<String>,
}

/// Holds possible stop-string prefixes so streamed text never has to be retracted.
#[derive(Debug)]
pub struct StopBuffer {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopBuffer {
    #[must_use]
    pub fn new(stops: impl IntoIterator<Item = String>) -> Self {
        let mut stops = stops
            .into_iter()
            .filter(|stop| !stop.is_empty())
            .collect::<Vec<_>>();
        stops.sort();
        stops.dedup();
        Self {
            stops,
            pending: String::new(),
            stopped: false,
        }
    }

    #[must_use]
    pub fn push(&mut self, text: &str) -> StopOutput {
        if self.stopped {
            return StopOutput {
                text: String::new(),
                matched: None,
            };
        }
        self.pending.push_str(text);

        let matched = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop).map(|position| (position, stop)))
            .min_by_key(|(position, _)| *position);
        if let Some((position, stop)) = matched {
            let emitted = self.pending[..position].to_owned();
            let stop = stop.clone();
            self.pending.clear();
            self.stopped = true;
            return StopOutput {
                text: emitted,
                matched: Some(stop),
            };
        }

        let held_bytes = self
            .pending
            .char_indices()
            .map(|(index, _)| &self.pending[index..])
            .filter(|suffix| self.stops.iter().any(|stop| stop.starts_with(suffix)))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let emit_bytes = self.pending.len() - held_bytes;
        let emitted = self.pending[..emit_bytes].to_owned();
        self.pending.drain(..emit_bytes);
        StopOutput {
            text: emitted,
            matched: None,
        }
    }

    #[must_use]
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        self.stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_is_invariant_under_every_byte_split() {
        let expected = "aλ🦀z";
        for split in 0..=expected.len() {
            let mut decoder = Utf8Buffer::default();
            let mut actual = decoder.push(&expected.as_bytes()[..split]);
            actual.push_str(&decoder.push(&expected.as_bytes()[split..]));
            actual.push_str(&decoder.finish());
            assert_eq!(actual, expected, "split {split}");
        }
    }

    #[test]
    fn invalid_utf8_uses_replacement_character() {
        let mut decoder = Utf8Buffer::default();
        assert_eq!(decoder.push(b"ok\xffdone"), "ok\u{fffd}done");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn stop_is_never_emitted_across_chunk_boundaries() {
        let mut buffer = StopBuffer::new(["<stop>".to_owned()]);
        assert_eq!(
            buffer.push("hello<st"),
            StopOutput {
                text: "hello".into(),
                matched: None,
            }
        );
        assert_eq!(
            buffer.push("op>ignored"),
            StopOutput {
                text: String::new(),
                matched: Some("<stop>".into()),
            }
        );
        assert!(buffer.is_stopped());
    }

    #[test]
    fn unicode_and_overlapping_stops_choose_the_earliest_match() {
        let mut buffer = StopBuffer::new(["🦀x".to_owned(), "END".to_owned()]);
        assert_eq!(buffer.push("a🦀").text, "a");
        let stopped = buffer.push("xEND");
        assert_eq!(stopped.text, "");
        assert_eq!(stopped.matched.as_deref(), Some("🦀x"));
    }

    #[test]
    fn finish_releases_a_partial_prefix() {
        let mut buffer = StopBuffer::new(["END".to_owned()]);
        assert_eq!(buffer.push("valueEN").text, "value");
        assert_eq!(buffer.finish(), "EN");
    }
}
