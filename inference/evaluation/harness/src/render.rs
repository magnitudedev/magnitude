//! Rendering template lanes exactly as the engine serves them (contract
//! "Rendered lane"): every conversation passes through the serving chat adapter, the
//! model's chat template and its tokenizer, the same host path that counting
//! and generation use.

use crate::digest::sha256_hex;
use crate::template::{Session, TemplateFile};
use magnitude_chat::{ChatRequest, TemplateSelection, ToolChoice};
use magnitude_engine::host::HostArtifacts;
use magnitude_serving::chat::{ChatCompletionRequest, chat_input};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::ops::Range;

/// The time templates that print the date render with
/// (2026-01-01T00:00:00Z), so a rendering is a function of the session.
const RENDER_TIME: i64 = 1_767_225_600;

/// Stands in for what a scored message generated (its reasoning, its content)
/// to locate it in the prompt: it is where the two prompts differ.
const CONTENT_PROBE: &str = "\u{E000}";

/// Chat-completions request fields besides `messages` and `tools` that every
/// rendered request carries, as the serving benchmark sends them (for
/// example `chat_template_kwargs`).
pub type RequestOptions = Map<String, Value>;

/// The options evaluation serves with: reasoning disabled.
pub fn standard_request_options() -> RequestOptions {
    let Value::Object(options) = json!({ "chat_template_kwargs": { "enable_thinking": false } })
    else {
        unreachable!("an object literal")
    };
    options
}

/// Parse request options: a JSON object of request fields that leave the
/// conversation itself to the session.
pub fn request_options(text: &str) -> Result<RequestOptions, String> {
    let Value::Object(options) =
        serde_json::from_str(text).map_err(|error| format!("request options: {error}"))?
    else {
        return Err("request options must be a JSON object".into());
    };
    if ["model", "messages", "tools"]
        .iter()
        .any(|key| options.contains_key(*key))
    {
        return Err("request options cannot carry the model, messages or tools".into());
    }
    Ok(options)
}

/// One lane of a template rendered for one model, sized to the template's
/// context.
pub struct Rendered {
    /// The template's name.
    pub template: String,
    pub lane: usize,
    /// SHA-256 of the chat template variant the conversation was prepared
    /// with.
    pub chat_template_sha256: String,
    /// The serving request's options the conversation was rendered with.
    pub request_options: RequestOptions,
    pub context: usize,
    /// The history units sizing kept before the probe.
    pub history: usize,
    pub tokens: Vec<u32>,
    pub scored_spans: Vec<Range<usize>>,
    /// The length of the serving request's prompt (the messages before the
    /// generation prefix).
    pub generation_prefix_tokens: usize,
}

impl Rendered {
    /// SHA-256 of the rendered tokens as `u32` little-endian bytes.
    pub fn prompt_sha256(&self) -> String {
        let bytes = self
            .tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        sha256_hex(&bytes)
    }

    /// The lane's label in messages: `<template>/<lane>`.
    pub fn label(&self) -> String {
        format!("{}/{}", self.template, self.lane)
    }

    pub fn line<'a>(&'a self, model: &'a str, target_sha256: &'a str) -> RenderedLine<'a> {
        RenderedLine {
            template: &self.template,
            lane: self.lane,
            model,
            target_sha256,
            chat_template_sha256: &self.chat_template_sha256,
            request_options: &self.request_options,
            context: self.context,
            history: self.history,
            tokens: &self.tokens,
            prompt_sha256: self.prompt_sha256(),
            scored_spans: self
                .scored_spans
                .iter()
                .map(|span| [span.start, span.end])
                .collect(),
            generation_prefix_tokens: self.generation_prefix_tokens,
        }
    }
}

/// The contract's rendered line.
#[derive(Serialize)]
pub struct RenderedLine<'a> {
    pub template: &'a str,
    pub lane: usize,
    pub model: &'a str,
    pub target_sha256: &'a str,
    pub chat_template_sha256: &'a str,
    pub request_options: &'a RequestOptions,
    pub context: usize,
    pub history: usize,
    pub tokens: &'a [u32],
    pub prompt_sha256: String,
    pub scored_spans: Vec<[usize; 2]>,
    pub generation_prefix_tokens: usize,
}

/// Writes rendered lanes as the contract's JSON Lines and returns the
/// file's digest.
pub fn write_rendered(
    path: &std::path::Path,
    rendered: &[Rendered],
    model: &str,
    target_sha256: &str,
) -> Result<String, String> {
    let mut bytes = Vec::new();
    for lane in rendered {
        serde_json::to_writer(&mut bytes, &lane.line(model, target_sha256))
            .map_err(|error| error.to_string())?;
        bytes.push(b'\n');
    }
    std::fs::write(path, &bytes).map_err(|error| format!("writing {}: {error}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

/// The conversation a session is rendered as after sizing: its messages
/// without the session's own fields, which of them are scored, the offered
/// tools and the serving request's length.
struct Conversation {
    messages: Vec<Map<String, Value>>,
    scored: Vec<bool>,
    tools: Vec<Value>,
    generation_prefix: usize,
}

impl Conversation {
    /// The session's messages at `kept` (ascending indices). The serving
    /// request shrinks by the messages dropped before its end.
    fn of(session: &Session, kept: &[usize]) -> Result<Self, String> {
        let mut tools = session.tools.clone();
        let mut messages = Vec::with_capacity(kept.len());
        let mut scored = Vec::with_capacity(kept.len());
        for &index in kept {
            let mut message = session.messages[index].clone();
            scored.push(message.remove("scored").is_some());
            if let Some(unit_tools) = message.remove("tools") {
                let Value::Array(unit_tools) = unit_tools else {
                    return Err("a unit's tools must be an array".into());
                };
                tools.extend(unit_tools);
            }
            messages.push(message);
        }
        // Offered tools are deduplicated by function name, first kept.
        let mut names = HashSet::new();
        let mut offered = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = tool["function"]["name"]
                .as_str()
                .ok_or("a tool has no function name")?
                .to_owned();
            if names.insert(name) {
                offered.push(tool);
            }
        }
        let generation_prefix = kept
            .iter()
            .take_while(|&&index| index < session.generation.prefix)
            .count();
        Ok(Self {
            messages,
            scored,
            tools: offered,
            generation_prefix,
        })
    }
}

/// A served prompt: the template's text and the tokens it encodes to.
struct Prompt {
    text: String,
    tokens: Vec<magnitude_chat::TokenId>,
}

/// A conversation, its template-level request, its served prompt, and each
/// scored message's byte span in that prompt.
struct Candidate {
    conversation: Conversation,
    request: ChatRequest,
    prompt: Prompt,
    scored_bytes: Vec<Range<usize>>,
}

pub struct Renderer<'a> {
    host: &'a HostArtifacts,
    /// The served model's name, which a request must carry; it does not
    /// change the prompt.
    model: &'a str,
    options: &'a RequestOptions,
}

impl<'a> Renderer<'a> {
    /// A renderer of every request for `model` with `options`, as the
    /// serving benchmark sends them.
    pub fn new(host: &'a HostArtifacts, model: &'a str, options: &'a RequestOptions) -> Self {
        Self {
            host,
            model,
            options,
        }
    }

    /// Every lane of `file`, in lane order, each sized to its context.
    pub fn render(&self, file: &TemplateFile) -> Result<Vec<Rendered>, String> {
        file.sessions
            .iter()
            .enumerate()
            .map(|(lane, session)| {
                self.render_lane(&file.name, lane, session, file.context)
                    .map_err(|error| format!("{} lane {lane}: {error}", file.name))
            })
            .collect()
    }

    /// `session`, lane `lane` of `template`, sized to `context` tokens.
    pub fn render_lane(
        &self,
        template: &str,
        lane: usize,
        session: &Session,
        context: usize,
    ) -> Result<Rendered, String> {
        let (candidate, history) = self.sized(session, context)?;
        let Candidate {
            conversation,
            request,
            prompt: prepared,
            scored_bytes,
        } = candidate;
        let tokens = prepared
            .tokens
            .iter()
            .map(|token| token.0)
            .collect::<Vec<_>>();
        let offsets = self.offsets(&prepared)?;
        let scored_spans = scored_bytes
            .into_iter()
            .map(|bytes| token_range(&offsets, bytes))
            .collect();
        let asked = self.request(
            &conversation.messages[..conversation.generation_prefix],
            &conversation.tools,
        )?;
        let generation_prefix_tokens = self.prepare(&asked)?.tokens.len();
        Ok(Rendered {
            template: template.to_owned(),
            lane,
            chat_template_sha256: self.chat_template_sha256(&request)?,
            request_options: self.options.clone(),
            context,
            history,
            tokens,
            scored_spans,
            generation_prefix_tokens,
        })
    }

    /// Keep the head and the probe, then the newest history units that fit
    /// `target` tokens, in conversation order, and how many history units
    /// it kept. A conversation never exceeds its target.
    fn sized(&self, session: &Session, target: usize) -> Result<(Candidate, usize), String> {
        let units = session.units()?;
        let (probe, history) = units.split_last().expect("a session has a probe");
        let render = |kept_units: usize| {
            let mut kept = (0..session.keep_head).collect::<Vec<_>>();
            for unit in &history[history.len() - kept_units..] {
                kept.extend(unit.clone());
            }
            kept.extend(probe.clone());
            self.candidate(Conversation::of(session, &kept)?)
        };
        let base = render(0)?;
        let base_length = base.prompt.tokens.len();
        if base_length > target {
            return Err(format!(
                "its head and probe exceed the context of {target} tokens"
            ));
        }
        // The rendered length grows with every unit kept: keep the most that
        // fit. Every count tried is rendered exactly, so the choice is the
        // one a bisection makes; each try renders the count the measured
        // lengths predict (units of one session are of similar length),
        // alternating with bisection so the interval at least halves every
        // other try. A session's long history is never rendered whole.
        let (mut fitting, mut fitting_length, mut chosen) = (0, base_length, base);
        let mut exceeding: Option<(usize, usize)> = None;
        let mut tries = 0usize;
        loop {
            let high = exceeding.map_or(history.len() + 1, |(count, _)| count);
            if high - fitting <= 1 {
                return Ok((chosen, fitting));
            }
            let per_unit = match exceeding {
                Some((count, length)) => {
                    (length - fitting_length) as f64 / (count - fitting) as f64
                }
                None if fitting > 0 => (fitting_length - base_length) as f64 / fitting as f64,
                None => 0.0,
            };
            tries += 1;
            let predicted = if tries % 2 == 0 && exceeding.is_some() {
                (fitting + high) / 2
            } else if per_unit > 0.0 {
                fitting + ((target - fitting_length) as f64 / per_unit) as usize
            } else {
                fitting + 1
            };
            let trying = predicted.clamp(fitting + 1, high - 1);
            let candidate = render(trying)?;
            let length = candidate.prompt.tokens.len();
            if length <= target {
                (fitting, fitting_length, chosen) = (trying, length, candidate);
            } else {
                exceeding = Some((trying, length));
            }
        }
    }

    /// `conversation` as the engine serves it, with any generation-prompt
    /// prefix that the template omitted from a historical scored assistant turn
    /// spliced back before it so teacher-forcing sees the exact prompt the
    /// model answered.
    fn candidate(&self, conversation: Conversation) -> Result<Candidate, String> {
        let request = self.request(&conversation.messages, &conversation.tools)?;
        let mut prompt = self.prepare(&request)?;
        let mut scored_bytes = Vec::new();
        let mut splices = Vec::new();
        for (index, _) in conversation
            .scored
            .iter()
            .enumerate()
            .filter(|(_, scored)| **scored)
        {
            let (raw_bytes, reasoned) =
                self.raw_scored_bytes(&conversation, &request, &prompt, index)?;
            let asked = self.prepare(
                &self.request(&conversation.messages[..index], &conversation.tools)?,
            )?;
            let (bytes, splice) =
                reconcile_scored_prefix(&prompt.text, &asked.text, raw_bytes, reasoned)?;
            if let Some(suffix) = splice {
                splices.push((bytes.start, suffix));
            }
            scored_bytes.push(bytes);
        }
        if !splices.is_empty() {
            let text = splice_generation_prefixes(&prompt.text, &splices, &mut scored_bytes);
            prompt = self.encode(text)?;
        }
        Ok(Candidate {
            conversation,
            request,
            prompt,
            scored_bytes,
        })
    }

    /// The template-level request the serving chat adapter makes of these
    /// messages and tools.
    fn request(
        &self,
        messages: &[Map<String, Value>],
        tools: &[Value],
    ) -> Result<ChatRequest, String> {
        let mut body = self.options.clone();
        body.insert("model".into(), json!(self.model));
        body.insert("messages".into(), json!(messages));
        if !tools.is_empty() {
            body.insert("tools".into(), Value::from(tools.to_vec()));
        }
        let request: ChatCompletionRequest = serde_json::from_value(Value::Object(body))
            .map_err(|error| format!("chat request: {error}"))?;
        let input = chat_input(request).map_err(|error| error.body.message)?;
        let rendered = input
            .render(RENDER_TIME, self.host.media_placeholder())
            .map_err(|error| error.to_string())?;
        if !rendered.images.is_empty() {
            return Err("sessions carry no images".into());
        }
        Ok(rendered.request)
    }

    /// The prompt the engine serves for `request`, as `PreparedChat::prepare`
    /// makes it: the template's text and its encoding. The prompt needs
    /// neither the request's output grammar compiled (which, for a
    /// conversation offering many tools, costs far more than rendering) nor
    /// the engine's choice between parallel tool calls and one call per turn
    /// that consults it: that choice reaches only the output parser and
    /// grammar, never the template's input.
    fn prepare(&self, request: &ChatRequest) -> Result<Prompt, String> {
        let prompt = self
            .host
            .templates()
            .prepare(request, &TemplateSelection::default())
            .map(|(native, _)| native.description().prompt.clone())
            .map_err(|error| error.to_string())?;
        self.encode(prompt)
    }

    fn encode(&self, text: String) -> Result<Prompt, String> {
        let tokens = self.host.tokenizer().encode_sequence(&text)?.tokens;
        if tokens.is_empty() {
            return Err("the chat template produced no input tokens".into());
        }
        Ok(Prompt { text, tokens })
    }

    /// SHA-256 of the template variant the request was prepared with.
    fn chat_template_sha256(&self, request: &ChatRequest) -> Result<String, String> {
        let tools_offered = !request.tools.is_empty() && request.tool_choice != ToolChoice::None;
        let variant = self
            .host
            .templates()
            .select(tools_offered, &TemplateSelection::default())?;
        Ok(sha256_hex(variant.source.as_bytes()))
    }

    /// Each prompt token's byte range in the prompt text. A sequence-start
    /// token the tokenizer inserted covers no text.
    fn offsets(&self, prepared: &Prompt) -> Result<Vec<Range<usize>>, String> {
        let tokenizer = self.host.tokenizer();
        let tokens = &prepared.tokens;
        let prompt = prepared.text.as_bytes();
        let inserted = usize::from(
            tokenizer.implicit_bos() == tokens.first().copied()
                && !prompt.starts_with(tokenizer.piece(tokens[0], false)?),
        );
        let mut offsets = vec![0..0; inserted];
        let mut offset = 0;
        for &token in &tokens[inserted..] {
            let piece = tokenizer.piece(token, false)?;
            if prompt.get(offset..offset + piece.len()) != Some(piece) {
                return Err(format!(
                    "token {} does not reproduce the prompt at byte {offset}",
                    token.0
                ));
            }
            offsets.push(offset..offset + piece.len());
            offset += piece.len();
        }
        if offset != prompt.len() {
            return Err("the prompt's tokens do not cover its text".into());
        }
        Ok(offsets)
    }

    /// The raw prompt bytes a scored message renders as, and whether it
    /// carried `reasoning_content`: where the prompt differs from the prompt
    /// with its reasoning (where it has any) and its content (where it has any,
    /// or tool calls, which are removed) each replaced by a probe.
    fn raw_scored_bytes(
        &self,
        conversation: &Conversation,
        request: &ChatRequest,
        prepared: &Prompt,
        scored: usize,
    ) -> Result<(Range<usize>, bool), String> {
        // The chat adapter drops assistant turns without content, reasoning
        // or tool calls, and nothing else between assistant turns.
        let nonempty = |message: &Map<String, Value>, key: &str| match message.get(key) {
            Some(Value::String(text)) => !text.is_empty(),
            Some(Value::Array(items)) => !items.is_empty(),
            _ => false,
        };
        let carries = |message: &Map<String, Value>| {
            nonempty(message, "content") || nonempty(message, "reasoning_content") || nonempty(message, "tool_calls")
        };
        if !carries(&conversation.messages[scored]) {
            return Err("a scored assistant message has no content".into());
        }
        let ordinal = conversation.messages[..scored]
            .iter()
            .filter(|message| {
                message.get("role").and_then(Value::as_str) == Some("assistant") && carries(message)
            })
            .count();
        let position = request
            .messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message["role"] == "assistant")
            .nth(ordinal)
            .map(|(position, _)| position)
            .ok_or("the scored message is missing from the rendered request")?;
        let mut probe = request.clone();
        let message = probe.messages[position]
            .as_object_mut()
            .ok_or("a rendered message is not an object")?;
        let reasoned = nonempty(&*message, "reasoning_content");
        if reasoned {
            message.insert("reasoning_content".into(), CONTENT_PROBE.into());
        }
        // Content left empty stays empty, so the span of an answer that only
        // reasoned ends where its reasoning does, not past the template's
        // close of the block, which the model never generated.
        let said = nonempty(&*message, "content") || nonempty(&*message, "tool_calls") || !reasoned;
        if said {
            message.insert("content".into(), CONTENT_PROBE.into());
        }
        message.remove("tool_calls");
        let probed = self
            .prepare(&probe)
            .map_err(|error| format!("the template rejects the scored-span probe: {error}"))?;
        let (prompt, probed) = (prepared.text.as_bytes(), probed.text.as_bytes());
        let start = prompt
            .iter()
            .zip(probed)
            .take_while(|(left, right)| left == right)
            .count();
        let limit = prompt.len().min(probed.len()) - start;
        let suffix = prompt
            .iter()
            .rev()
            .zip(probed.iter().rev())
            .take(limit)
            .take_while(|(left, right)| left == right)
            .count();
        let end = prompt.len() - suffix;
        if end <= start {
            return Err("a scored message renders no text".into());
        }
        Ok((start..end, reasoned))
    }
}

/// Reconciles a scored reply's byte span in the rendered conversation against
/// the prompt the model actually saw when generating it (`asked`).
///
/// - When the reply reasoned, `asked` must prefix `prompt[..bytes.start]` and
///   the span widens back to `asked.len()` so any reasoning opener the model
///   generated is included in the scored span.
/// - When the reply did not reason and `asked` strictly extends
///   `prompt[..bytes.start]` (as with templates whose `add_generation_prompt`
///   under `enable_thinking: false` prefills an empty thought block such as
///   `<|channel>thought\n<channel|>` or `<think></think>`, while their
///   historical assistant rendering omits it), returns the missing suffix to
///   splice immediately before `bytes.start`.
fn reconcile_scored_prefix(
    prompt: &str,
    asked: &str,
    bytes: Range<usize>,
    reasoned: bool,
) -> Result<(Range<usize>, Option<String>), String> {
    if reasoned {
        if asked.len() > bytes.start || !prompt.starts_with(asked) {
            return Err(
                "a reasoned reply's rendering does not continue the prompt it answered".into(),
            );
        }
        return Ok((asked.len()..bytes.end, None));
    }
    if asked.len() > bytes.start
        && prompt.is_char_boundary(bytes.start)
        && asked.is_char_boundary(bytes.start)
        && asked.starts_with(&prompt[..bytes.start])
    {
        return Ok((bytes.clone(), Some(asked[bytes.start..].to_owned())));
    }
    Ok((bytes, None))
}

/// Splices each `(at, suffix)` (in ascending `at` order) into `text`, shifting
/// every span in `spans` whose start is at or past `at` by `suffix.len()`.
fn splice_generation_prefixes(
    text: &str,
    splices: &[(usize, String)],
    spans: &mut [Range<usize>],
) -> String {
    let extra: usize = splices.iter().map(|(_, suffix)| suffix.len()).sum();
    let mut out = String::with_capacity(text.len() + extra);
    let mut cursor = 0;
    for (at, suffix) in splices {
        assert!(*at >= cursor, "splices must be in ascending byte order");
        out.push_str(&text[cursor..*at]);
        out.push_str(suffix);
        cursor = *at;
    }
    out.push_str(&text[cursor..]);
    for span in spans.iter_mut() {
        let shift: usize = splices
            .iter()
            .take_while(|(at, _)| *at <= span.start)
            .map(|(_, suffix)| suffix.len())
            .sum();
        span.start += shift;
        span.end += shift;
    }
    out
}

/// The tokens whose text overlaps `bytes`.
fn token_range(offsets: &[Range<usize>], bytes: Range<usize>) -> Range<usize> {
    let start = offsets
        .iter()
        .position(|token| token.end > bytes.start && !token.is_empty())
        .unwrap_or(offsets.len());
    let end = offsets
        .iter()
        .position(|token| token.start >= bytes.end && !token.is_empty())
        .unwrap_or(offsets.len());
    start..end
}

/// Real-model checks of rendering:
/// `MAGNITUDE_TEST_GGUF=M.gguf MAGNITUDE_TEST_TEMPLATE=FILE.json cargo test
/// --release -p magnitude-harness -- --ignored render`.
#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_engine::options::{PackageOptions, ProjectorSelection};

    fn setup() -> (HostArtifacts, TemplateFile) {
        let variable = |name: &str| {
            std::path::PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
        };
        let host = HostArtifacts::open(&PackageOptions::single(
            variable("MAGNITUDE_TEST_GGUF"),
            ProjectorSelection::Disabled,
            None,
        ))
        .unwrap();
        (
            host,
            TemplateFile::read(&variable("MAGNITUDE_TEST_TEMPLATE")).unwrap(),
        )
    }

    /// The text of `tokens`, pieces concatenated.
    fn text(host: &HostArtifacts, tokens: &[u32]) -> String {
        let bytes = tokens
            .iter()
            .flat_map(|&token| {
                host.tokenizer()
                    .piece(magnitude_chat::TokenId(token), false)
                    .unwrap()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[test]
    fn reconcile_splices_missing_generation_prefix_suffix() {
        // Gemma 4: add_generation_prompt with enable_thinking=false emits
        // `<|channel>thought\n<channel|>`, which historical assistant turns omit.
        let prompt = "<|turn>user\nGo on.<turn|>\n<|turn>model\n```python\npass<turn|>\n";
        let start = "<|turn>user\nGo on.<turn|>\n<|turn>model\n".len();
        let end = start + "```python\npass".len();
        let asked = "<|turn>user\nGo on.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";

        let (span, splice) =
            reconcile_scored_prefix(prompt, asked, start..end, false).unwrap();
        assert_eq!(span, start..end);
        assert_eq!(splice.as_deref(), Some("<|channel>thought\n<channel|>"));

        let mut spans = vec![span];
        let spliced = splice_generation_prefixes(
            prompt,
            &[(start, splice.unwrap())],
            &mut spans,
        );
        assert!(spliced.starts_with(asked));
        assert_eq!(&spliced[spans[0].clone()], "```python\npass");
        assert_eq!(spans[0].start, asked.len());
    }

    #[test]
    fn reconcile_leaves_matching_prefix_untouched() {
        let prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n";
        let asked = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let start = asked.len();
        let end = start + "answer".len();

        let (span, splice) =
            reconcile_scored_prefix(prompt, asked, start..end, false).unwrap();
        assert_eq!(span, start..end);
        assert_eq!(splice, None);
    }

    #[test]
    fn reconcile_widens_reasoned_span_and_rejects_mismatched_asked() {
        let prompt = "<|turn>model\n<|channel>thought\nhmm\n<channel|>ok<turn|>\n";
        let asked = "<|turn>model\n";
        let start = "<|turn>model\n<|channel>thought\n".len();
        let end = "<|turn>model\n<|channel>thought\nhmm\n<channel|>ok".len();

        let (span, splice) =
            reconcile_scored_prefix(prompt, asked, start..end, true).unwrap();
        assert_eq!(span, asked.len()..end);
        assert_eq!(splice, None);
        assert_eq!(&prompt[span], "<|channel>thought\nhmm\n<channel|>ok");

        let mismatched = "<|turn>model\n<|channel>thought\n<channel|>";
        assert!(reconcile_scored_prefix(prompt, mismatched, start..end, true).is_err());
    }

    #[test]
    fn splice_generation_prefixes_handles_multiple_scored_spans() {
        let text = "U1 A1 T1 A2 end";
        let mut spans = vec![3..5, 9..11];
        let spliced = splice_generation_prefixes(
            text,
            &[(3, "<P1>".into()), (9, "<P2>".into())],
            &mut spans,
        );
        assert_eq!(spliced, "U1 <P1>A1 T1 <P2>A2 end");
        assert_eq!(&spliced[spans[0].clone()], "A1");
        assert_eq!(&spliced[spans[1].clone()], "A2");

        // Only first span spliced.
        let mut spans_first = vec![3..5, 9..11];
        let spliced_first =
            splice_generation_prefixes(text, &[(3, "<P1>".into())], &mut spans_first);
        assert_eq!(spliced_first, "U1 <P1>A1 T1 A2 end");
        assert_eq!(&spliced_first[spans_first[0].clone()], "A1");
        assert_eq!(&spliced_first[spans_first[1].clone()], "A2");
    }

    /// Every scored span is exactly its message's content (its tool calls'
    /// markup when it has them) at the template's context, and sizing never
    /// exceeds it.
    #[test]
    #[ignore = "requires MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_TEMPLATE"]
    fn render_spans_cover_exactly_the_scored_content() {
        let (host, file) = setup();
        let options = standard_request_options();
        let renderer = Renderer::new(&host, "evaluation", &options);
        for (session, rendered) in file.sessions.iter().zip(renderer.render(&file).unwrap()) {
            let label = rendered.label();
            assert!(rendered.tokens.len() <= file.context, "{label}");
            let scored = session
                .messages
                .iter()
                .filter(|message| message.contains_key("scored"))
                .collect::<Vec<_>>();
            assert_eq!(scored.len(), rendered.scored_spans.len(), "{label}");
            for (message, span) in scored.iter().zip(&rendered.scored_spans) {
                let spanned = text(&host, &rendered.tokens[span.clone()]);
                match message.get("content").and_then(Value::as_str) {
                    Some(content) if !message.contains_key("tool_calls") => {
                        assert_eq!(spanned.trim(), content.trim(), "{label}")
                    }
                    _ => {
                        for call in message["tool_calls"].as_array().unwrap() {
                            let name = call["function"]["name"].as_str().unwrap();
                            assert!(spanned.contains(name), "{label}: {spanned}");
                        }
                    }
                }
            }
            eprintln!(
                "{label}: {} tokens, {} history units, spans {:?}",
                rendered.tokens.len(),
                rendered.history,
                rendered.scored_spans
            );
        }
    }
}
