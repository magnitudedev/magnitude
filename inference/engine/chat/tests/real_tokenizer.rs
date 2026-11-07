//! Token-exact agreement with a model's own tokenizer and chat template.
//!
//! A reference holds a corpus encoded by the model's original `tokenizer.json`
//! and chat requests rendered by the model's own chat format, with their ids.
//! DeepSeek V4.1's also holds assistant replies with tool calls, as its
//! encoding writes them, which the engine must read back.
//!
//! - DeepSeek V4.1's reference is committed and always runs:
//!   `tests/assets/deepseek41.json` beside the published GGUF's vocabulary,
//!   with the chat template the converter writes
//!   (`inference/conversion/src/magnitude_conversion/deepseek41.jinja`). It is
//!   written by `magnitude_references.deepseek41_chat` from the checkpoint's
//!   `tokenizer.json` and its encoding (`encoding/encoding.py`).
//! - Any other model's comes from `transformers` `apply_chat_template` over the
//!   GGUF's template, written by
//!   `inference/engine/chat/tests/support/tokenizer_reference.py`. Run with
//!   `MAGNITUDE_TEST_GGUF=<name>.gguf MAGNITUDE_TEST_REFERENCE=<name>.json
//!   cargo test -p magnitude-chat --test real_tokenizer -- --ignored`. The GGUF
//!   may be a complete file or the generator's vocabulary-only header.
use flate2::read::GzDecoder;
use magnitude_artifacts::{
    gguf::{inspect_header, read_header},
    TemplatePayload, TemplateSource, TokenizerPayload,
};
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer, ChatRequest, Event, FinishReason, OutputToken, PreparedChat, SpecialTokens,
    TemplateBundle, TemplateSelection, TerminalCause, TokenChatStream, TokenId,
};
use magnitude_grammar::{CacheLimits, Vocabulary};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::{io::Read, sync::Arc};

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    /// The reference decoding of `ids`: the text after normalization.
    decoded: String,
}

#[derive(Deserialize)]
struct Render {
    name: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
    /// The request's `chat_template_kwargs`.
    #[serde(default)]
    template_arguments: Map<String, Value>,
    text: String,
    ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct Call {
    name: String,
    arguments: Map<String, Value>,
}

/// An assistant reply, as the text the model writes after its request's
/// prompt (without the EOS generation drops), and what it was written from.
#[derive(Deserialize)]
struct Reply {
    name: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
    template_arguments: Map<String, Value>,
    text: String,
    content: String,
    reasoning: String,
    calls: Vec<Call>,
}

#[derive(Deserialize)]
struct Reference {
    now: i64,
    corpus: Vec<Case>,
    renders: Vec<Render>,
    /// Only DeepSeek V4.1's reference has replies.
    #[serde(default)]
    replies: Vec<Reply>,
}

fn load() -> (ByteBpeTokenizer, TemplateBundle, Reference) {
    let path = std::env::var("MAGNITUDE_TEST_GGUF").expect("MAGNITUDE_TEST_GGUF names a GGUF");
    let directory = inspect_header(&path).unwrap();
    let tokenizer_payload = TokenizerPayload::from_directory(&directory);
    let tokenizer =
        ByteBpeTokenizer::new(gguf_byte_bpe(&tokenizer_payload, "test".into()).unwrap()).unwrap();
    let templates = gguf_templates(
        &TemplatePayload::from_directory(&directory, &path).unwrap(),
        &tokenizer_payload,
    )
    .unwrap();
    let reference = std::env::var("MAGNITUDE_TEST_REFERENCE")
        .expect("MAGNITUDE_TEST_REFERENCE names the reference JSON");
    let reference = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    (tokenizer, templates, reference)
}

/// The template DeepSeek V4.1 GGUFs carry, as the converter writes it.
const DEEPSEEK41_TEMPLATE: &str =
    include_str!("../../../conversion/src/magnitude_conversion/deepseek41.jinja");

/// The published GGUF's vocabulary and the reference, as
/// `magnitude_references.deepseek41_chat` writes them.
const DEEPSEEK41_VOCABULARY: &[u8] =
    include_bytes!("assets/DeepSeek-V4.1-Flash-1017-vocabulary.gguf.gz");
const DEEPSEEK41_REFERENCE: &str = include_str!("assets/deepseek41.json");

fn deepseek41() -> (ByteBpeTokenizer, TemplateBundle, Reference) {
    let mut header = Vec::new();
    GzDecoder::new(DEEPSEEK41_VOCABULARY)
        .read_to_end(&mut header)
        .unwrap();
    let tokenizer_payload = TokenizerPayload::from_directory(&read_header(&header).unwrap());
    let tokenizer =
        ByteBpeTokenizer::new(gguf_byte_bpe(&tokenizer_payload, "test".into()).unwrap()).unwrap();
    let templates = gguf_templates(
        &TemplatePayload {
            sources: vec![TemplateSource {
                name: "default".into(),
                source: DEEPSEEK41_TEMPLATE.into(),
                provenance: "magnitude-conversion deepseek41.jinja".into(),
            }],
        },
        &tokenizer_payload,
    )
    .unwrap();
    let reference = serde_json::from_str(DEEPSEEK41_REFERENCE).unwrap();
    (tokenizer, templates, reference)
}

fn ids(tokens: &[TokenId]) -> Vec<u32> {
    tokens.iter().map(|token| token.0).collect()
}

fn decode(tokenizer: &ByteBpeTokenizer, tokens: &[u32]) -> String {
    let mut decoder = tokenizer.decoder(false);
    let mut text = String::new();
    for token in tokens {
        text.push_str(&decoder.push(TokenId(*token)).unwrap());
    }
    text.push_str(&decoder.finish().unwrap());
    text
}

/// Every corpus text encodes to the reference ids, which decode back to the
/// reference text.
fn corpus_agrees(tokenizer: &ByteBpeTokenizer, reference: &Reference) {
    let mut failures = Vec::new();
    for case in &reference.corpus {
        let encoded = ids(&tokenizer
            .encode(&case.text, SpecialTokens::Recognize)
            .unwrap());
        if encoded != case.ids {
            failures.push(format!(
                "{:?}\n  engine    {encoded:?}\n  reference {:?}",
                case.text, case.ids
            ));
        }
        let decoded = decode(tokenizer, &case.ids);
        if decoded != case.decoded {
            failures.push(format!(
                "{:?} decodes as {decoded:?}, reference {:?}",
                case.text, case.decoded
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("{} texts agree", reference.corpus.len());
}

/// Every request renders the reference text, and its model input is the
/// reference tokenization of that text with the implicit BOS present exactly
/// once.
fn renders_agree(tokenizer: &ByteBpeTokenizer, templates: &TemplateBundle, reference: &Reference) {
    let mut failures = Vec::new();
    for render in &reference.renders {
        let mut request = ChatRequest::new(render.messages.clone(), reference.now);
        request.tools = render.tools.clone();
        request.template_arguments = render.template_arguments.clone();
        let prepared = match PreparedChat::prepare(
            templates,
            tokenizer,
            &request,
            &TemplateSelection::default(),
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                failures.push(format!("{}: {error}", render.name));
                continue;
            }
        };
        if prepared.prompt() != render.text {
            failures.push(format!(
                "{}: prompt differs\n--- engine\n{}\n--- reference\n{}",
                render.name,
                prepared.prompt(),
                render.text
            ));
            continue;
        }
        let mut expected = render.ids.clone();
        if let Some(bos) = tokenizer.implicit_bos() {
            if expected.first() != Some(&bos.0) {
                expected.insert(0, bos.0);
            }
            assert_ne!(expected.get(1), Some(&bos.0), "{}: double BOS", render.name);
        }
        let tokens = ids(&prepared.input().tokens);
        if tokens != expected {
            failures.push(format!(
                "{}: tokens differ\n  engine    {tokens:?}\n  reference {expected:?}",
                render.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("{} renders agree", reference.renders.len());
}

#[test]
#[ignore = "needs MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_REFERENCE"]
fn corpus_encodes_token_exactly_and_decodes_back() {
    let (tokenizer, _, reference) = load();
    corpus_agrees(&tokenizer, &reference);
}

#[test]
#[ignore = "needs MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_REFERENCE"]
fn chat_prompts_render_as_the_reference_and_begin_with_one_bos() {
    let (tokenizer, templates, reference) = load();
    renders_agree(&tokenizer, &templates, &reference);
}

#[test]
fn deepseek41_corpus_encodes_as_its_tokenizer_json() {
    let (tokenizer, _, reference) = deepseek41();
    assert_eq!(tokenizer.implicit_bos(), None);
    corpus_agrees(&tokenizer, &reference);
}

#[test]
fn deepseek41_requests_render_as_its_encoding() {
    let (tokenizer, templates, reference) = deepseek41();
    renders_agree(&tokenizer, &templates, &reference);
}

/// Reasoning is on by default at the encoding's default effort, turns off
/// with `enable_thinking`, and takes exactly the encoding's named efforts.
#[test]
fn deepseek41_reasoning_is_the_encodings() {
    let (_, templates, _) = deepseek41();
    let inspection = templates.inspect().unwrap();
    let reasoning = &inspection.reasoning;
    let efforts = reasoning
        .mappings
        .iter()
        .map(|mapping| mapping.effort.as_str())
        .collect::<Vec<_>>();
    assert_eq!(efforts, ["none", "low", "high", "max"]);
    assert_eq!(reasoning.default_effort.as_deref(), Some("high"));
}

/// What the output parser reads from a reply.
#[derive(Debug, PartialEq)]
struct Parsed {
    content: String,
    reasoning: String,
    calls: Vec<Call>,
}

impl Reply {
    fn request(&self, now: i64) -> ChatRequest {
        let mut request = ChatRequest::new(self.messages.clone(), now);
        request.tools = self.tools.clone();
        request.template_arguments = self.template_arguments.clone();
        request
    }
}

/// `text` generated after `request`'s prompt: its tokens, then EOS.
struct Generated {
    tokenizer: Arc<ByteBpeTokenizer>,
    prepared: PreparedChat,
    tokens: Vec<TokenId>,
}

impl Generated {
    fn new(
        tokenizer: &Arc<ByteBpeTokenizer>,
        templates: &TemplateBundle,
        request: &ChatRequest,
        text: &str,
    ) -> Self {
        let prepared =
            PreparedChat::prepare(templates, tokenizer, request, &TemplateSelection::default())
                .unwrap();
        let mut tokens = tokenizer.encode(text, SpecialTokens::Recognize).unwrap();
        let eos = tokenizer
            .encode("<｜end▁of▁sentence｜>", SpecialTokens::Recognize)
            .unwrap();
        assert!(eos.len() == 1 && tokenizer.stop_tokens().contains(&eos[0]));
        tokens.extend(eos);
        Self {
            tokenizer: tokenizer.clone(),
            prepared,
            tokens,
        }
    }

    /// Whether the request's grammar admits the tokens and ends with them.
    fn admitted(&self) -> Result<(), String> {
        let plan = self
            .prepared
            .input()
            .constraint
            .clone()
            .expect("a request with tools or JSON output is constrained");
        let mut vocabulary = Vocabulary::new(
            self.tokenizer.clone(),
            self.tokenizer.vocabulary(),
            CacheLimits {
                entries: 0,
                bytes: 0,
            },
        )
        .unwrap();
        vocabulary
            .bind(&plan.grammar, &plan.prefix)
            .unwrap()
            .advance(&self.tokens)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    /// The output parser's reading, each token published in turn as
    /// generation publishes it, the stop token ending the output.
    fn parsed(&self) -> Result<Parsed, String> {
        let mut stream = TokenChatStream::new(&self.prepared, &self.tokenizer, Vec::new(), 1 << 20)?;
        let mut events = Vec::new();
        for (index, token) in self.tokens[..self.tokens.len() - 1].iter().enumerate() {
            events.extend(stream.feed(&OutputToken {
                index,
                token: *token,
            })?);
        }
        events.extend(stream.finish(FinishReason::Stop)?);
        let mut parsed = Parsed {
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
        };
        // A call is read once it completes: its name and its argument text.
        let mut started = Vec::<(String, String)>::new();
        for event in events {
            match event {
                Event::Content { text } => parsed.content.push_str(&text),
                Event::Reasoning { text } => parsed.reasoning.push_str(&text),
                Event::ToolStart { index, name, .. } => {
                    assert_eq!(index as usize, started.len());
                    started.push((name, String::new()));
                }
                Event::ToolArguments { index, text } => started[index as usize].1.push_str(&text),
                Event::ToolComplete { index } => {
                    assert_eq!(index as usize, parsed.calls.len());
                    let (name, text) = &started[index as usize];
                    parsed.calls.push(Call {
                        name: name.clone(),
                        arguments: serde_json::from_str(text)
                            .map_err(|error| format!("call {index} arguments {text:?}: {error}"))?,
                    });
                }
                Event::Finish { cause } => assert_eq!(cause, TerminalCause::Natural),
            }
        }
        if parsed.calls.len() != started.len() {
            return Err(format!("{} of {} calls completed", parsed.calls.len(), started.len()));
        }
        Ok(parsed)
    }
}

/// Every reply the encoding writes is admitted by its request's grammar and
/// reads back as the reasoning, content and calls (names, and arguments as
/// JSON) it was written from.
#[test]
fn deepseek41_replies_read_back_as_their_calls() {
    let (tokenizer, templates, reference) = deepseek41();
    let tokenizer = Arc::new(tokenizer);
    assert!(!reference.replies.is_empty());
    let mut failures = Vec::new();
    for reply in &reference.replies {
        let generated = Generated::new(
            &tokenizer,
            &templates,
            &reply.request(reference.now),
            &reply.text,
        );
        if let Err(error) = generated.admitted() {
            failures.push(format!("{}: {error}", reply.name));
        }
        let expected = Parsed {
            content: reply.content.clone(),
            reasoning: reply.reasoning.clone(),
            calls: reply.calls.clone(),
        };
        match generated.parsed() {
            Ok(parsed) if parsed == expected => {}
            Ok(parsed) => failures.push(format!(
                "{}: parsed\n  engine    {parsed:?}\n  reference {expected:?}",
                reply.name
            )),
            Err(error) => failures.push(format!("{}: {error}", reply.name)),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("{} replies read back", reference.replies.len());
}

/// A parameter without its `string` attribute is not V4.1's format (the
/// encoding's own reader refuses it, encoding.py:927-929): the grammar never
/// writes it, and the parser refuses it.
#[test]
fn deepseek41_refuses_a_parameter_without_its_string_attribute() {
    let (tokenizer, templates, reference) = deepseek41();
    let tokenizer = Arc::new(tokenizer);
    let reply = reference
        .replies
        .iter()
        .find(|reply| reply.name == "single_call")
        .unwrap();
    let malformed = reply.text.replacen(" string=\"true\"", "", 1);
    assert_ne!(malformed, reply.text);
    let generated = Generated::new(
        &tokenizer,
        &templates,
        &reply.request(reference.now),
        &malformed,
    );
    assert_eq!(
        generated.admitted(),
        Err("grammar violation: the tokens violate the grammar".into())
    );
    assert_eq!(
        generated.parsed(),
        Err("Output does not match its prepared parser".into())
    );
}

/// JSON output follows the convention of llama.cpp's DeepSeek handler, which
/// V4.1 shares from v0.9.0: the value in a ```json fence. The encoding has no
/// reply form for a response format (encoding.py:604-629 only states the
/// schema), so the fence is the engine's, and the parser reads the value
/// inside it as the content.
#[test]
fn deepseek41_reads_fenced_json_output() {
    let (tokenizer, templates, reference) = deepseek41();
    let tokenizer = Arc::new(tokenizer);
    let user = reference
        .renders
        .iter()
        .find(|render| render.name == "user")
        .unwrap();
    let mut request = ChatRequest::new(user.messages.clone(), reference.now);
    request.json_schema = Some(serde_json::json!({
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"]
    }));
    let generated = Generated::new(
        &tokenizer,
        &templates,
        &request,
        "Two plus two.</think>```json\n{\"answer\": 4}\n```",
    );
    generated.admitted().unwrap();
    let parsed = generated.parsed().unwrap();
    assert_eq!(parsed.reasoning, "Two plus two.");
    assert!(parsed.calls.is_empty());
    assert_eq!(
        serde_json::from_str::<Value>(&parsed.content).unwrap(),
        serde_json::json!({"answer": 4})
    );
}
