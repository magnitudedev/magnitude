use super::PreparedChat;
use super::{ByteBpeTokenizer, TokenDecoder};
use magnitude_generation::{FinishReason, OutputToken};
use magnitude_templates::{Event, OutputStream, PreparedRequest, TerminalCause};

enum MatchedStop {
    User(String),
    Template,
}

enum TerminalState {
    Stopped(MatchedStop),
    Ended,
}

impl TerminalState {
    fn stopped(&self) -> bool {
        matches!(self, Self::Stopped(_))
    }

    fn matched_stop(&self) -> Option<&str> {
        match self {
            Self::Stopped(MatchedStop::User(stop)) => Some(stop.as_str()),
            Self::Stopped(MatchedStop::Template) | Self::Ended => None,
        }
    }
}

enum TokenChatStreamState<'a> {
    Active {
        decoder: TokenDecoder<'a>,
        chat: ChatStream,
        next_index: usize,
    },
    Terminal(TerminalState),
}

/// Host-owned composition of accepted token publication and native parsing.
/// A matched stop is terminal here; the caller must cancel further generation.
pub struct TokenChatStream<'a> {
    state: TokenChatStreamState<'a>,
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
            state: TokenChatStreamState::Active {
                // Parser delimiters can themselves be control tokens. Generation
                // removes EOS before publication; preserve every other token byte.
                decoder: tokenizer.decoder(false),
                chat: ChatStream::new(prepared.native(), stops, max_output_bytes)?,
                next_index: 0,
            },
        })
    }
    pub fn stopped(&self) -> bool {
        match &self.state {
            TokenChatStreamState::Active { .. } => false,
            TokenChatStreamState::Terminal(terminal) => terminal.stopped(),
        }
    }
    /// The caller stop sequence that ended output, if one did.
    pub fn matched_stop(&self) -> Option<&str> {
        match &self.state {
            TokenChatStreamState::Active { .. } => None,
            TokenChatStreamState::Terminal(terminal) => terminal.matched_stop(),
        }
    }
    pub fn feed(&mut self, output: &OutputToken) -> Result<Vec<Event>, String> {
        let TokenChatStreamState::Active {
            decoder,
            chat,
            next_index,
        } = &mut self.state
        else {
            return Err("token chat stream is terminal".into());
        };
        let result = (|| {
            if output.index != *next_index {
                return Err("token publication is out of order".into());
            }
            let next = next_index
                .checked_add(1)
                .ok_or("token publication index overflow")?;
            let text = decoder.push(output.token)?;
            let events = chat.feed(&text)?;
            *next_index = next;
            Ok(events)
        })();
        let should_terminate = chat.is_terminal() || result.is_err();
        if should_terminate {
            self.finish_to_terminal();
        }
        result
    }
    pub fn finish(&mut self, reason: FinishReason) -> Result<Vec<Event>, String> {
        let TokenChatStreamState::Active { decoder, chat, .. } = &mut self.state else {
            return Err("token chat stream is terminal".into());
        };
        let result = (|| {
            let tail = decoder.finish()?;
            let mut events = chat.feed(&tail)?;
            if !chat.stopped() {
                let cause = match reason {
                    FinishReason::Stop => TerminalCause::Natural,
                    FinishReason::Length | FinishReason::Context => TerminalCause::Length,
                    FinishReason::Cancelled => TerminalCause::Cancelled,
                    FinishReason::Failed => TerminalCause::Failed,
                };
                events.extend(chat.finish(cause)?);
            }
            Ok(events)
        })();
        self.finish_to_terminal();
        result
    }
    fn finish_to_terminal(&mut self) {
        if matches!(self.state, TokenChatStreamState::Active { .. }) {
            let prev = std::mem::replace(
                &mut self.state,
                TokenChatStreamState::Terminal(TerminalState::Ended),
            );
            if let TokenChatStreamState::Active { chat, .. } = prev {
                self.state = TokenChatStreamState::Terminal(chat.into_terminal());
            }
        }
    }
}

enum StopState {
    Buffering { pending: String },
    Matched(usize),
    Closed,
}

enum StopFeed {
    Continue(String),
    Matched { output: String, index: usize },
}

impl StopFeed {
    fn into_output(self) -> String {
        match self {
            Self::Continue(output) | Self::Matched { output, .. } => output,
        }
    }
}

/// Retains only a suffix that could become a stop string. The winner is the first
/// completed match, with configured order breaking ties, independent of chunks.
pub struct StopText {
    stops: Vec<String>,
    state: StopState,
}
impl StopText {
    pub fn new(stops: Vec<String>) -> Result<Self, String> {
        if stops.iter().any(String::is_empty) {
            return Err("stop strings must be nonempty".into());
        }
        Ok(Self {
            stops,
            state: StopState::Buffering {
                pending: String::new(),
            },
        })
    }
    pub fn matched(&self) -> Option<&str> {
        match self.state {
            StopState::Matched(index) => Some(self.stops[index].as_str()),
            StopState::Buffering { .. } | StopState::Closed => None,
        }
    }
    pub fn feed(&mut self, text: &str) -> Result<String, String> {
        Ok(self.feed_outcome(text)?.into_output())
    }
    pub fn finish(&mut self) -> Result<String, String> {
        match std::mem::replace(&mut self.state, StopState::Closed) {
            StopState::Buffering { pending } => Ok(pending),
            StopState::Matched(_) => Ok(String::new()),
            StopState::Closed => Err("stop filter is closed".into()),
        }
    }
    fn stop(&self, index: usize) -> &str {
        self.stops[index].as_str()
    }
    fn feed_outcome(&mut self, text: &str) -> Result<StopFeed, String> {
        let pending = match &mut self.state {
            StopState::Closed => return Err("stop filter is closed".into()),
            StopState::Matched(index) => {
                return Ok(StopFeed::Matched {
                    output: String::new(),
                    index: *index,
                });
            }
            StopState::Buffering { pending } => pending,
        };
        pending.push_str(text);
        let matched = self
            .stops
            .iter()
            .enumerate()
            .filter_map(|(i, stop)| {
                pending
                    .find(stop)
                    .map(|start| (start + stop.len(), i, start))
            })
            .min();
        if let Some((_, index, start)) = matched {
            let output = pending[..start].to_owned();
            self.state = StopState::Matched(index);
            return Ok(StopFeed::Matched { output, index });
        }
        let keep = {
            let pending = pending.as_str();
            self.stops
                .iter()
                .flat_map(|stop| {
                    stop.char_indices()
                        .map(|(i, _)| i)
                        .filter(|&i| i > 0)
                        .filter(move |&i| pending.ends_with(&stop[..i]))
                })
                .max()
                .unwrap_or(0)
        };
        let end = pending.len() - keep;
        let output = pending[..end].to_owned();
        pending.drain(..end);
        // Shrink only if an unusually large chunk inflated the retained buffer.
        if pending.capacity() > 4096 && pending.len() <= 1024 {
            pending.shrink_to(1024);
        }
        Ok(StopFeed::Continue(output))
    }
}

enum ChatStreamState {
    Active {
        parser: OutputStream,
        stops: StopText,
        user_stops: usize,
    },
    Terminal(TerminalState),
}

/// Consumes incrementally decoded text. Token decoding stays with the tokenizer;
/// stop filtering precedes semantic parsing so truncated calls remain incomplete.
pub struct ChatStream {
    state: ChatStreamState,
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
            state: ChatStreamState::Active {
                parser: plan.stream(max_output_bytes).map_err(|e| e.to_string())?,
                stops: StopText::new(stops)?,
                user_stops,
            },
        })
    }
    pub fn stopped(&self) -> bool {
        match &self.state {
            ChatStreamState::Active { .. } => false,
            ChatStreamState::Terminal(terminal) => terminal.stopped(),
        }
    }
    /// The caller stop sequence that ended output; template stops are natural
    /// completion.
    pub fn matched_stop(&self) -> Option<&str> {
        match &self.state {
            ChatStreamState::Active { .. } => None,
            ChatStreamState::Terminal(terminal) => terminal.matched_stop(),
        }
    }
    pub fn feed(&mut self, text: &str) -> Result<Vec<Event>, String> {
        let ChatStreamState::Active {
            parser,
            stops,
            user_stops,
        } = &mut self.state
        else {
            return Err("chat stream is terminal".into());
        };
        match stops.feed_outcome(text)? {
            StopFeed::Continue(output) => match parser.feed(output.as_bytes()) {
                Ok(events) => Ok(events),
                Err(err) => {
                    self.state = ChatStreamState::Terminal(TerminalState::Ended);
                    Err(err.to_string())
                }
            },
            StopFeed::Matched { output, index } => {
                let (matched, cause) = if index < *user_stops {
                    (
                        MatchedStop::User(stops.stop(index).to_owned()),
                        TerminalCause::UserStop,
                    )
                } else {
                    (MatchedStop::Template, TerminalCause::Natural)
                };
                let result = (|| {
                    let mut events = parser.feed(output.as_bytes()).map_err(|e| e.to_string())?;
                    events.extend(parser.finish(cause).map_err(|e| e.to_string())?);
                    Ok(events)
                })();
                self.state = ChatStreamState::Terminal(TerminalState::Stopped(matched));
                result
            }
        }
    }
    pub fn finish(&mut self, cause: TerminalCause) -> Result<Vec<Event>, String> {
        let ChatStreamState::Active { parser, stops, .. } = &mut self.state else {
            return Err("chat stream is terminal".into());
        };
        let result = (|| {
            let tail = stops.finish()?;
            let mut events = parser.feed(tail.as_bytes()).map_err(|e| e.to_string())?;
            events.extend(parser.finish(cause).map_err(|e| e.to_string())?);
            Ok(events)
        })();
        self.state = ChatStreamState::Terminal(TerminalState::Ended);
        result
    }
    fn is_terminal(&self) -> bool {
        matches!(self.state, ChatStreamState::Terminal(_))
    }
    fn into_terminal(self) -> TerminalState {
        match self.state {
            ChatStreamState::Active { .. } => TerminalState::Ended,
            ChatStreamState::Terminal(terminal) => terminal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending_buf(stop: &StopText) -> Option<&String> {
        match &stop.state {
            StopState::Buffering { pending } => Some(pending),
            StopState::Matched(_) | StopState::Closed => None,
        }
    }

    #[test]
    fn stop_text_drains_prefix_and_shrinks_oversized_chunks() {
        let mut stop = StopText::new(vec!["STOP".into()]).unwrap();
        // Holds partial stop prefix, then releases it when continuation does not match.
        assert_eq!(stop.feed("hello ST").unwrap(), "hello ");
        assert_eq!(pending_buf(&stop).map(String::as_str), Some("ST"));
        assert_eq!(stop.feed("ART!").unwrap(), "START!");
        assert_eq!(pending_buf(&stop).map(String::as_str), Some(""));

        // Oversized chunk does not stay pinned in `pending`.
        let large = "x".repeat(16 * 1024);
        assert_eq!(stop.feed(&large).unwrap(), large);
        let pending = pending_buf(&stop).unwrap();
        assert!(pending.is_empty());
        assert!(pending.capacity() <= 4096);

        // Split stop match across chunks emits only text before the stop.
        assert_eq!(stop.feed("tail ST").unwrap(), "tail ");
        assert_eq!(stop.feed("OPignored").unwrap(), "");
        assert_eq!(stop.matched(), Some("STOP"));
        assert!(pending_buf(&stop).is_none());
        assert_eq!(stop.feed("more").unwrap(), "");
        assert_eq!(stop.matched(), Some("STOP"));
    }

    #[test]
    fn stop_text_finish_flushes_pending_and_closes() {
        let mut stop = StopText::new(vec!["STOP".into()]).unwrap();
        assert_eq!(stop.feed("hello ST").unwrap(), "hello ");
        assert_eq!(stop.finish().unwrap(), "ST");
        assert!(matches!(stop.state, StopState::Closed));
        assert_eq!(stop.matched(), None);
        assert!(stop.feed("more").is_err());
        assert!(stop.finish().is_err());
    }
}
