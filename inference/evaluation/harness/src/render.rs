//! Rendering fixtures exactly as the engine serves them (contract "Rendered
//! fixture"): every conversation passes through the serving chat adapter, the
//! model's chat template and its tokenizer, the same host path that counting
//! and generation use.

use crate::digest::sha256_hex;
use crate::fixture::{Fixture, Sizing};
use magnitude_chat::{ChatRequest, TemplateSelection, ToolChoice};
use magnitude_engine::host::HostArtifacts;
use magnitude_serving::chat::{ChatCompletionRequest, chat_input};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::ops::Range;

/// The time templates that print the date render with
/// (2026-01-01T00:00:00Z), so a rendering is a function of the fixture.
const RENDER_TIME: i64 = 1_767_225_600;

/// Stands in for a scored message's content to locate it in the prompt: the
/// content is where the two prompts differ.
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
/// conversation itself to the fixture.
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

/// One fixture rendered for one model.
pub struct Rendered {
    pub fixture_id: String,
    pub template_sha256: String,
    /// The serving request's options the conversation was rendered with.
    pub request_options: RequestOptions,
    pub context_target: Option<usize>,
    pub tokens: Vec<u32>,
    pub scored_spans: Vec<Range<usize>>,
    /// The length of `generation`'s serving prompt, when the fixture has one.
    pub generation_prefix_tokens: Option<usize>,
    /// That serving prompt, when rendered in this process: a rendered file
    /// holds only its length.
    pub generation_prompt: Option<Vec<u32>>,
}

/// The contract's rendered-fixture line, as read from a rendered file.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedLine {
    fixture_id: String,
    fixture_set_digest: String,
    model: String,
    target_sha256: String,
    template_sha256: String,
    request_options: RequestOptions,
    context_target: Option<usize>,
    tokens: Vec<u32>,
    scored_spans: Vec<[usize; 2]>,
    generation_prefix_tokens: Option<usize>,
}

/// A rendered-fixture file read back: its fixtures and its digest (SHA-256
/// of its bytes).
pub struct RenderedFile {
    pub bytes: Vec<u8>,
    pub digest: String,
    pub rendered: Vec<Rendered>,
}

impl RenderedFile {
    /// Read `path`, which must have been rendered from the fixture set of
    /// `fixture_set_digest` for this model and target.
    pub fn read(
        path: &std::path::Path,
        fixture_set_digest: &str,
        model: &str,
        target_sha256: &str,
    ) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|error| format!("rendered {}: {error}", path.display()))?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|error| format!("rendered {}: {error}", path.display()))?;
        let mut rendered = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let failed = |reason: String| {
                format!("rendered {} line {}: {reason}", path.display(), index + 1)
            };
            let line: OwnedLine =
                serde_json::from_str(line).map_err(|error| failed(error.to_string()))?;
            if line.model != model || line.target_sha256 != target_sha256 {
                return Err(failed(format!(
                    "rendered for {} ({}), not {model} ({target_sha256})",
                    line.model, line.target_sha256
                )));
            }
            if line.fixture_set_digest != fixture_set_digest {
                return Err(failed("rendered from another fixture set".into()));
            }
            rendered.push(Rendered {
                fixture_id: line.fixture_id,
                template_sha256: line.template_sha256,
                request_options: line.request_options,
                context_target: line.context_target,
                tokens: line.tokens,
                scored_spans: line
                    .scored_spans
                    .into_iter()
                    .map(|[start, end]| start..end)
                    .collect(),
                generation_prefix_tokens: line.generation_prefix_tokens,
                generation_prompt: None,
            });
        }
        if rendered.is_empty() {
            return Err(format!("rendered {} holds no fixture", path.display()));
        }
        Ok(Self {
            digest: sha256_hex(&bytes),
            bytes,
            rendered,
        })
    }
}

/// The contract's rendered-fixture line.
#[derive(Serialize)]
pub struct RenderedLine<'a> {
    pub fixture_id: &'a str,
    pub fixture_set_digest: &'a str,
    pub model: &'a str,
    pub target_sha256: &'a str,
    pub template_sha256: &'a str,
    pub request_options: &'a RequestOptions,
    pub context_target: Option<usize>,
    pub tokens: &'a [u32],
    pub scored_spans: Vec<[usize; 2]>,
    pub generation_prefix_tokens: Option<usize>,
}

impl Rendered {
    pub fn line<'a>(
        &'a self,
        fixture_set_digest: &'a str,
        model: &'a str,
        target_sha256: &'a str,
    ) -> RenderedLine<'a> {
        RenderedLine {
            fixture_id: &self.fixture_id,
            fixture_set_digest,
            model,
            target_sha256,
            template_sha256: &self.template_sha256,
            request_options: &self.request_options,
            context_target: self.context_target,
            tokens: &self.tokens,
            scored_spans: self
                .scored_spans
                .iter()
                .map(|span| [span.start, span.end])
                .collect(),
            generation_prefix_tokens: self.generation_prefix_tokens,
        }
    }
}

/// Writes rendered fixtures as the contract's JSON Lines and returns the
/// file's digest.
pub fn write_rendered(
    path: &std::path::Path,
    rendered: &[Rendered],
    fixture_set_digest: &str,
    model: &str,
    target_sha256: &str,
) -> Result<String, String> {
    let mut bytes = Vec::new();
    for fixture in rendered {
        serde_json::to_writer(
            &mut bytes,
            &fixture.line(fixture_set_digest, model, target_sha256),
        )
        .map_err(|error| error.to_string())?;
        bytes.push(b'\n');
    }
    std::fs::write(path, &bytes).map_err(|error| format!("writing {}: {error}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

/// The conversation a fixture is rendered as after sizing: its messages
/// without the fixture's own fields, which of them are scored, the offered
/// tools and the serving request's length.
struct Conversation {
    messages: Vec<Map<String, Value>>,
    scored: Vec<bool>,
    tools: Vec<Value>,
    generation_prefix: Option<usize>,
}

impl Conversation {
    /// The fixture's messages at `kept` (ascending indices). The serving
    /// request shrinks by the messages dropped before its end.
    fn of(fixture: &Fixture, kept: &[usize]) -> Result<Self, String> {
        let mut tools = fixture.tools.clone();
        let mut messages = Vec::with_capacity(kept.len());
        let mut scored = Vec::with_capacity(kept.len());
        for &index in kept {
            let mut message = fixture.messages[index].clone();
            scored.push(message.remove("scored").is_some());
            if let Some(unit_tools) = message.remove("tools") {
                let Value::Array(unit_tools) = unit_tools else {
                    return Err(format!(
                        "fixture {}: a unit's tools must be an array",
                        fixture.id
                    ));
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
                .ok_or_else(|| format!("fixture {}: a tool has no function name", fixture.id))?
                .to_owned();
            if names.insert(name) {
                offered.push(tool);
            }
        }
        let generation_prefix = fixture.generation.as_ref().map(|generation| {
            kept.iter()
                .take_while(|&&index| index < generation.prefix)
                .count()
        });
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

/// A conversation, its template-level request and its served prompt.
struct Candidate {
    conversation: Conversation,
    request: ChatRequest,
    prompt: Prompt,
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

    /// Render `fixture`, sized to `context` tokens when given.
    pub fn render(&self, fixture: &Fixture, context: Option<usize>) -> Result<Rendered, String> {
        let failed = |error: String| format!("fixture {}: {error}", fixture.id);
        let all = (0..fixture.messages.len()).collect::<Vec<_>>();
        let (candidate, context_target) = match (context, fixture.sizing) {
            (None, _) => (
                self.candidate(Conversation::of(fixture, &all)?)
                    .map_err(failed)?,
                None,
            ),
            (Some(target), Sizing::Fixed) => {
                let candidate = self
                    .candidate(Conversation::of(fixture, &all)?)
                    .map_err(failed)?;
                let length = candidate.prompt.tokens.len();
                if length > target {
                    return Err(failed(format!(
                        "a fixed fixture of {length} tokens exceeds the context of {target}"
                    )));
                }
                (candidate, None)
            }
            (Some(target), Sizing::Turn { keep_head }) => {
                (self.sized(fixture, keep_head, target)?, Some(target))
            }
        };
        let Candidate {
            conversation,
            request,
            prompt: prepared,
        } = candidate;
        let tokens = prepared
            .tokens
            .iter()
            .map(|token| token.0)
            .collect::<Vec<_>>();
        let offsets = self.offsets(&prepared).map_err(failed)?;
        let mut scored_spans = Vec::new();
        for (index, _) in conversation
            .scored
            .iter()
            .enumerate()
            .filter(|(_, scored)| **scored)
        {
            let bytes = self
                .scored_bytes(&conversation, &request, &prepared, index)
                .map_err(failed)?;
            scored_spans.push(token_range(&offsets, bytes));
        }
        let generation_prompt = conversation
            .generation_prefix
            .map(|prefix| {
                let request =
                    self.request(&conversation.messages[..prefix], &conversation.tools)?;
                Ok::<_, String>(
                    self.prepare(&request)?
                        .tokens
                        .iter()
                        .map(|token| token.0)
                        .collect(),
                )
            })
            .transpose()
            .map_err(failed)?;
        Ok(Rendered {
            fixture_id: fixture.id.clone(),
            template_sha256: self.template_sha256(&request).map_err(failed)?,
            request_options: self.options.clone(),
            context_target,
            tokens,
            scored_spans,
            generation_prefix_tokens: generation_prompt.as_ref().map(Vec::len),
            generation_prompt,
        })
    }

    /// Keep the head and the probe, then the newest history units that fit
    /// `target` tokens, in conversation order. A conversation never exceeds
    /// its target.
    fn sized(
        &self,
        fixture: &Fixture,
        keep_head: usize,
        target: usize,
    ) -> Result<Candidate, String> {
        let failed = |error: String| format!("fixture {}: {error}", fixture.id);
        let units = fixture.units()?;
        let (probe, history) = units.split_last().expect("a fixture has a probe");
        let render = |kept_units: usize| {
            let mut kept = (0..keep_head).collect::<Vec<_>>();
            for unit in &history[history.len() - kept_units..] {
                kept.extend(unit.clone());
            }
            kept.extend(probe.clone());
            self.candidate(Conversation::of(fixture, &kept)?)
                .map_err(failed)
        };
        let base = render(0)?;
        let base_length = base.prompt.tokens.len();
        if base_length > target {
            return Err(failed(format!(
                "its head and probe exceed the context of {target} tokens"
            )));
        }
        // The rendered length grows with every unit kept: keep the most that
        // fit. Every count tried is rendered exactly, so the choice is the
        // one a bisection makes; each try renders the count the measured
        // lengths predict (units of one fixture are of similar length),
        // alternating with bisection so the interval at least halves every
        // other try. A fixture's long history is never rendered whole.
        let (mut fitting, mut fitting_length, mut chosen) = (0, base_length, base);
        let mut exceeding: Option<(usize, usize)> = None;
        let mut tries = 0usize;
        loop {
            let high = exceeding.map_or(history.len() + 1, |(count, _)| count);
            if high - fitting <= 1 {
                return Ok(chosen);
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

    /// `conversation` as the engine serves it.
    fn candidate(&self, conversation: Conversation) -> Result<Candidate, String> {
        let request = self.request(&conversation.messages, &conversation.tools)?;
        let prompt = self.prepare(&request)?;
        Ok(Candidate {
            conversation,
            request,
            prompt,
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
            return Err("fixtures carry no images".into());
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
        let tokens = self.host.tokenizer().encode_sequence(&prompt)?.tokens;
        if tokens.is_empty() {
            return Err("the chat template produced no input tokens".into());
        }
        Ok(Prompt {
            text: prompt,
            tokens,
        })
    }

    /// SHA-256 of the template variant the request was prepared with.
    fn template_sha256(&self, request: &ChatRequest) -> Result<String, String> {
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

    /// The prompt bytes a scored message renders as: where the prompt differs
    /// from the prompt with that message's content replaced by a probe and
    /// its tool calls removed. Its reasoning is not scored.
    fn scored_bytes(
        &self,
        conversation: &Conversation,
        request: &ChatRequest,
        prepared: &Prompt,
        scored: usize,
    ) -> Result<Range<usize>, String> {
        // The chat adapter drops assistant turns without content, reasoning
        // or tool calls, and nothing else between assistant turns.
        let carries = |message: &Map<String, Value>| {
            let nonempty = |key: &str| match message.get(key) {
                Some(Value::String(text)) => !text.is_empty(),
                Some(Value::Array(items)) => !items.is_empty(),
                _ => false,
            };
            nonempty("content") || nonempty("reasoning_content") || nonempty("tool_calls")
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
        message.insert("content".into(), CONTENT_PROBE.into());
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
        Ok(start..end)
    }
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
/// `MAGNITUDE_TEST_GGUF=M.gguf MAGNITUDE_TEST_FIXTURES=SET.jsonl cargo test
/// --release -p magnitude-harness -- --ignored render`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::FixtureSet;
    use magnitude_engine::options::{PackageOptions, ProjectorSelection};

    fn setup() -> (HostArtifacts, FixtureSet) {
        let variable = |name: &str| {
            std::path::PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
        };
        let host = HostArtifacts::open(&PackageOptions {
            target: variable("MAGNITUDE_TEST_GGUF"),
            projector: ProjectorSelection::Disabled,
            draft: None,
        })
        .unwrap();
        (
            host,
            FixtureSet::read(&variable("MAGNITUDE_TEST_FIXTURES")).unwrap(),
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

    /// Every scored span is exactly its message's content (its tool calls'
    /// markup when it has them) at served context lengths, and sizing never
    /// exceeds the target.
    #[test]
    #[ignore = "requires MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_FIXTURES"]
    fn render_spans_cover_exactly_the_scored_content() {
        let (host, set) = setup();
        let options = standard_request_options();
        let renderer = Renderer::new(&host, "evaluation", &options);
        for context in [Some(4096), Some(16384), Some(65536)] {
            for fixture in &set.fixtures {
                let rendered = match renderer.render(fixture, context) {
                    Ok(rendered) => rendered,
                    Err(error) if error.contains("exceed the context") => continue,
                    Err(error) => panic!("{error}"),
                };
                if let Some(target) = context {
                    assert!(rendered.tokens.len() <= target, "{}", fixture.id);
                }
                let scored = fixture
                    .messages
                    .iter()
                    .filter(|message| message.contains_key("scored"))
                    .collect::<Vec<_>>();
                assert_eq!(scored.len(), rendered.scored_spans.len(), "{}", fixture.id);
                for (message, span) in scored.iter().zip(&rendered.scored_spans) {
                    let spanned = text(&host, &rendered.tokens[span.clone()]);
                    match message.get("content").and_then(Value::as_str) {
                        Some(content) if !message.contains_key("tool_calls") => assert_eq!(
                            spanned.trim(),
                            content.trim(),
                            "{} at {context:?}",
                            fixture.id
                        ),
                        _ => {
                            for call in message["tool_calls"].as_array().unwrap() {
                                let name = call["function"]["name"].as_str().unwrap();
                                assert!(spanned.contains(name), "{}: {spanned}", fixture.id);
                            }
                        }
                    }
                }
                eprintln!(
                    "{} at {context:?}: {} tokens, spans {:?}",
                    fixture.id,
                    rendered.tokens.len(),
                    rendered.scored_spans
                );
            }
        }
    }
}
