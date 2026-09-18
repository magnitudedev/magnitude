//! OpenAI-compatible response framing of native semantic publications. Usage is an
//! explicit execution snapshot; token counts and device timings are never inferred.
use super::{ChatPublication, Event, TerminalCause};
use serde_json::{json, Value};

pub use crate::generation::Usage;
impl Usage {
    fn json(self) -> Result<Value, String> {
        let total = self
            .prompt_tokens
            .checked_add(self.completion_tokens)
            .ok_or("token usage overflow")?;
        Ok(
            json!({"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "total_tokens": total, "prompt_tokens_details": {"cached_tokens": 0}}),
        )
    }
}
#[derive(Default)]
struct Semantics {
    tools: Vec<bool>,
    terminal: bool,
}
impl Semantics {
    fn observe(&mut self, event: &Event) -> Result<Option<&'static str>, String> {
        if self.terminal {
            return Err("semantic event follows terminal event".into());
        }
        match event {
            Event::ToolStart { index, name, id } => {
                if *index as usize != self.tools.len() || name.is_empty() || id.is_empty() {
                    return Err("invalid tool start or noncontiguous tool index".into());
                }
                self.tools.push(false);
            }
            Event::ToolArguments { index, .. } => {
                if self.tools.get(*index as usize) != Some(&false) {
                    return Err("arguments require an open tool call".into());
                }
            }
            Event::ToolComplete { index } => {
                let complete = self
                    .tools
                    .get_mut(*index as usize)
                    .ok_or("completion requires an open tool call")?;
                if *complete {
                    return Err("tool call already completed".into());
                }
                *complete = true;
            }
            Event::Finish { cause } => {
                if *cause == TerminalCause::Natural && self.tools.iter().any(|&complete| !complete)
                {
                    return Err("natural completion contains an incomplete tool call".into());
                }
                self.terminal = true;
                return Ok(Some(match cause {
                    TerminalCause::Length => "length",
                    TerminalCause::Natural if !self.tools.is_empty() => "tool_calls",
                    _ => "stop",
                }));
            }
            _ => {}
        }
        Ok(None)
    }
}
fn validate_identity(id: &str, model: &str, limit: usize) -> Result<(), String> {
    if id.is_empty() || id.len() > 128 || model.is_empty() || model.len() > 1024 || limit == 0 {
        return Err("invalid response identity or byte limit".into());
    }
    Ok(())
}
pub struct SseResponse {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    started: bool,
    finished: bool,
    semantics: Semantics,
    bytes: usize,
    limit: usize,
}
impl SseResponse {
    pub fn new(
        id: String,
        model: String,
        created: u64,
        include_usage: bool,
        max_bytes: usize,
    ) -> Result<Self, String> {
        validate_identity(&id, &model, max_bytes)?;
        Ok(Self {
            id,
            model,
            created,
            include_usage,
            started: false,
            finished: false,
            semantics: Semantics::default(),
            bytes: 0,
            limit: max_bytes,
        })
    }
    fn envelope(&self, choices: Value) -> Value {
        json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":choices})
    }
    fn chunk(&self, delta: Value, reason: Option<&str>) -> Value {
        self.envelope(json!([{"index":0,"delta":delta,"finish_reason":reason}]))
    }
    fn frame(&mut self, payload: Value) -> Result<Vec<u8>, String> {
        let encoded = serde_json::to_string(&payload).map_err(|e| e.to_string())?;
        self.charge(
            encoded
                .len()
                .checked_add(8)
                .ok_or("response byte overflow")?,
        )?;
        Ok(format!("data: {encoded}\n\n").into_bytes())
    }
    fn charge(&mut self, bytes: usize) -> Result<(), String> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .filter(|&n| n <= self.limit)
            .ok_or("response byte limit exceeded")?;
        Ok(())
    }
    /// A terminal publication must carry usage when include_usage was requested.
    /// Once framing fails the response is closed; already returned frames remain
    /// caller-owned and must never be replayed.
    pub fn feed(&mut self, publication: &ChatPublication) -> Result<Vec<Vec<u8>>, String> {
        if self.finished {
            return Err("response is terminal".into());
        }
        let result = self.feed_impl(publication);
        if result.is_err() {
            self.finished = true;
        }
        result
    }
    fn feed_impl(&mut self, publication: &ChatPublication) -> Result<Vec<Vec<u8>>, String> {
        let mut frames = Vec::new();
        if !self.started {
            frames.push(self.frame(self.chunk(json!({"role":"assistant"}), None))?);
            self.started = true;
        }
        for event in &publication.events {
            let reason = self.semantics.observe(event)?;
            let delta = match event {
                Event::Content { text } => Some(json!({"content":text})),
                Event::Reasoning { text } => Some(json!({"reasoning_content":text})),
                Event::ToolStart { index, name, id } => Some(
                    json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":""}}]}),
                ),
                Event::ToolArguments { index, text } => {
                    Some(json!({"tool_calls":[{"index":index,"function":{"arguments":text}}]}))
                }
                Event::ToolComplete { .. } => None,
                Event::Finish { cause } => {
                    if publication.error.is_some()
                        || matches!(cause, TerminalCause::Failed | TerminalCause::Cancelled)
                    {
                        let kind = if *cause == TerminalCause::Cancelled {
                            "request_cancelled"
                        } else {
                            "server_error"
                        };
                        let message = publication.error.as_deref().unwrap_or(
                            if *cause == TerminalCause::Cancelled {
                                "request cancelled"
                            } else {
                                "model execution failed"
                            },
                        );
                        frames.push(
                            self.frame(
                                json!({"error":{"message":message,"type":kind,"code":kind}}),
                            )?,
                        );
                    } else {
                        frames.push(self.frame(self.chunk(json!({}), reason))?);
                        if self.include_usage {
                            let usage = publication
                                .usage
                                .ok_or("terminal response requires authoritative token usage")?
                                .json()?;
                            let mut envelope = self.envelope(json!([]));
                            envelope["usage"] = usage;
                            frames.push(self.frame(envelope)?);
                        }
                    }
                    self.charge(14)?;
                    frames.push(b"data: [DONE]\n\n".to_vec());
                    self.finished = true;
                    None
                }
            };
            if let Some(delta) = delta {
                frames.push(self.frame(self.chunk(delta, None))?);
            }
        }
        if publication.error.is_some() && !self.finished {
            return Err("execution error lacks a terminal semantic event".into());
        }
        Ok(frames)
    }
}

#[derive(serde::Serialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}
#[derive(serde::Serialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionCall,
}
/// Bounded nonstream assembly over exactly the same semantic publication stream.
/// Failed/cancelled requests return an error, never a partial successful response.
pub struct CompleteResponse {
    id: String,
    model: String,
    created: u64,
    semantics: Semantics,
    content: String,
    reasoning: String,
    saw_reasoning: bool,
    calls: Vec<ToolCall>,
    retained: usize,
    limit: usize,
    finished: bool,
}
impl CompleteResponse {
    pub fn new(id: String, model: String, created: u64, max_bytes: usize) -> Result<Self, String> {
        validate_identity(&id, &model, max_bytes)?;
        Ok(Self {
            id,
            model,
            created,
            semantics: Semantics::default(),
            content: String::new(),
            reasoning: String::new(),
            saw_reasoning: false,
            calls: Vec::new(),
            retained: 0,
            limit: max_bytes,
            finished: false,
        })
    }
    fn charge(&mut self, bytes: usize) -> Result<(), String> {
        self.retained = self
            .retained
            .checked_add(bytes)
            .filter(|&n| n <= self.limit)
            .ok_or("response byte limit exceeded")?;
        Ok(())
    }
    pub fn feed(&mut self, publication: &ChatPublication) -> Result<Option<Vec<u8>>, String> {
        if self.finished {
            return Err("response is terminal".into());
        }
        let result = self.feed_impl(publication);
        if result.is_err() || matches!(&result, Ok(Some(_))) {
            self.finished = true;
        }
        result
    }
    fn feed_impl(&mut self, publication: &ChatPublication) -> Result<Option<Vec<u8>>, String> {
        let mut terminal = None;
        for event in &publication.events {
            let reason = self.semantics.observe(event)?;
            match event {
                Event::Content { text } => {
                    self.charge(text.len())?;
                    self.content.push_str(text);
                }
                Event::Reasoning { text } => {
                    self.charge(text.len())?;
                    self.reasoning.push_str(text);
                    self.saw_reasoning = true;
                }
                Event::ToolStart { id, name, .. } => {
                    self.charge(
                        id.len()
                            .checked_add(name.len())
                            .ok_or("response byte overflow")?,
                    )?;
                    self.calls.push(ToolCall {
                        id: id.clone(),
                        kind: "function",
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: String::new(),
                        },
                    });
                }
                Event::ToolArguments { index, text } => {
                    self.charge(text.len())?;
                    self.calls[*index as usize]
                        .function
                        .arguments
                        .push_str(text);
                }
                Event::ToolComplete { .. } => {}
                Event::Finish { cause } => {
                    terminal = Some((*cause, reason.expect("terminal reason")))
                }
            }
        }
        if let Some(error) = &publication.error {
            return Err(error.clone());
        }
        let Some((cause, reason)) = terminal else {
            return Ok(None);
        };
        match cause {
            TerminalCause::Failed => return Err("model execution failed".into()),
            TerminalCause::Cancelled => return Err("request cancelled".into()),
            _ => {}
        }
        let usage = publication
            .usage
            .ok_or("terminal response requires authoritative token usage")?
            .json()?;
        let mut message = json!({"role":"assistant","content":if self.content.is_empty() { Value::Null } else { json!(self.content) }});
        if self.saw_reasoning {
            message["reasoning_content"] = json!(self.reasoning);
        }
        if !self.calls.is_empty() {
            message["tool_calls"] = json!(self.calls);
        }
        let result = json!({"id":self.id,"object":"chat.completion","created":self.created,
            "model":self.model,"choices":[{"index":0,"message":message,"finish_reason":reason}],"usage":usage});
        let bytes = serde_json::to_vec(&result).map_err(|error| error.to_string())?;
        if bytes.len() > self.limit {
            return Err("response byte limit exceeded".into());
        }
        Ok(Some(bytes))
    }
}
