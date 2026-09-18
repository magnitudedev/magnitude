use super::PreparedChat;
use crate::{
    generation::{FinishReason, OutputToken},
    inputs::{ByteBpeTokenizer, TokenDecoder},
};
use magnitude_templates::{Event, OutputStream, PreparedRequest, TerminalCause};

/// Host-owned composition of accepted token publication and native parsing.
/// A matched stop is terminal here; the caller must cancel further generation.
pub struct TokenChatStream<'a> {
    decoder: TokenDecoder<'a>,
    chat: ChatStream,
    next_index: usize,
    terminal: bool,
}
impl<'a> TokenChatStream<'a> {
    pub fn new(
        prepared: &PreparedChat,
        tokenizer: &'a ByteBpeTokenizer,
        stops: Vec<String>,
        max_output_bytes: usize,
    ) -> Result<Self, String> {
        if prepared.input().tokenizer_identity != tokenizer.identity()
            || prepared.input().artifact_identity != tokenizer.artifact_identity()
        {
            return Err("stream tokenizer differs from prepared chat".into());
        }
        Ok(Self {
            // Parser delimiters can themselves be control tokens. Generation
            // removes EOS before publication; preserve every other token byte.
            decoder: tokenizer.decoder(false),
            chat: ChatStream::new(prepared.native(), stops, max_output_bytes)?,
            next_index: 0,
            terminal: false,
        })
    }
    pub fn stopped(&self) -> bool {
        self.chat.stopped()
    }
    pub fn feed(&mut self, output: &OutputToken) -> Result<Vec<Event>, String> {
        if self.terminal {
            return Err("token chat stream is terminal".into());
        }
        let result = (|| {
            if output.index != self.next_index {
                return Err("token publication is out of order".into());
            }
            let next = self
                .next_index
                .checked_add(1)
                .ok_or("token publication index overflow")?;
            let text = self.decoder.push(output.token)?;
            let events = self.chat.feed(&text)?;
            self.next_index = next;
            Ok(events)
        })();
        self.terminal = self.chat.stopped() || result.is_err();
        result
    }
    pub fn finish(&mut self, reason: FinishReason) -> Result<Vec<Event>, String> {
        if self.terminal {
            return Err("token chat stream is terminal".into());
        }
        self.terminal = true;
        let tail = self.decoder.finish()?;
        let mut events = self.chat.feed(&tail)?;
        if !self.chat.stopped() {
            let cause = match reason {
                FinishReason::Stop => TerminalCause::Natural,
                FinishReason::Length | FinishReason::Context => TerminalCause::Length,
                FinishReason::Cancelled => TerminalCause::Cancelled,
                FinishReason::Failed => TerminalCause::Failed,
            };
            events.extend(self.chat.finish(cause)?);
        }
        Ok(events)
    }
}

/// Retains only a suffix that could become a stop string. The winner is the first
/// completed match, with configured order breaking ties, independent of chunks.
pub struct StopText {
    stops: Vec<String>,
    pending: String,
    matched: Option<usize>,
    closed: bool,
}
impl StopText {
    pub fn new(stops: Vec<String>) -> Result<Self, String> {
        if stops.iter().any(String::is_empty) {
            return Err("stop strings must be nonempty".into());
        }
        Ok(Self {
            stops,
            pending: String::new(),
            matched: None,
            closed: false,
        })
    }
    pub fn matched(&self) -> Option<&str> {
        self.matched.map(|i| self.stops[i].as_str())
    }
    pub fn feed(&mut self, text: &str) -> Result<String, String> {
        if self.closed {
            return Err("stop filter is closed".into());
        }
        if self.matched.is_some() {
            return Ok(String::new());
        }
        self.pending.push_str(text);
        let matched = self
            .stops
            .iter()
            .enumerate()
            .filter_map(|(i, stop)| {
                self.pending
                    .find(stop)
                    .map(|start| (start + stop.len(), i, start))
            })
            .min();
        if let Some((_, index, start)) = matched {
            let output = self.pending[..start].to_owned();
            self.pending.clear();
            self.matched = Some(index);
            return Ok(output);
        }
        let keep = self
            .stops
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .map(|(i, _)| i)
                    .filter(|&i| i > 0)
                    .filter(|&i| self.pending.ends_with(&stop[..i]))
            })
            .max()
            .unwrap_or(0);
        let end = self.pending.len() - keep;
        let output = self.pending[..end].to_owned();
        // Replace rather than drain so one large transport chunk is not retained.
        self.pending = self.pending[end..].to_owned();
        Ok(output)
    }
    pub fn finish(&mut self) -> Result<String, String> {
        if self.closed {
            return Err("stop filter is closed".into());
        }
        self.closed = true;
        Ok(std::mem::take(&mut self.pending))
    }
}

/// Consumes incrementally decoded text. Token decoding stays with the tokenizer;
/// stop filtering precedes semantic parsing so truncated calls remain incomplete.
pub struct ChatStream {
    parser: OutputStream,
    stops: StopText,
    user_stops: usize,
    terminal: bool,
}
impl ChatStream {
    pub fn new(
        plan: &PreparedRequest,
        mut stops: Vec<String>,
        max_output_bytes: usize,
    ) -> Result<Self, String> {
        let user_stops = stops.len();
        stops.extend(plan.description().additional_stops.iter().cloned());
        Ok(Self {
            parser: plan.stream(max_output_bytes).map_err(|e| e.to_string())?,
            stops: StopText::new(stops)?,
            user_stops,
            terminal: false,
        })
    }
    pub fn stopped(&self) -> bool {
        self.stops.matched().is_some()
    }
    pub fn feed(&mut self, text: &str) -> Result<Vec<Event>, String> {
        if self.terminal {
            return Err("chat stream is terminal".into());
        }
        let text = self.stops.feed(text)?;
        let result = (|| {
            let mut events = self
                .parser
                .feed(text.as_bytes())
                .map_err(|e| e.to_string())?;
            if self.stopped() {
                events.extend(
                    self.parser
                        .finish(if self.stops.matched.unwrap() < self.user_stops {
                            TerminalCause::UserStop
                        } else {
                            TerminalCause::Natural
                        })
                        .map_err(|e| e.to_string())?,
                );
            }
            Ok(events)
        })();
        self.terminal = self.stopped() || result.is_err();
        result
    }
    pub fn finish(&mut self, cause: TerminalCause) -> Result<Vec<Event>, String> {
        if self.terminal {
            return Err("chat stream is terminal".into());
        }
        self.terminal = true;
        let tail = self.stops.finish()?;
        let mut events = self
            .parser
            .feed(tail.as_bytes())
            .map_err(|e| e.to_string())?;
        events.extend(self.parser.finish(cause).map_err(|e| e.to_string())?);
        Ok(events)
    }
}
