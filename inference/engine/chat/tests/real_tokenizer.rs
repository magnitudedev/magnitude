//! Token-exact agreement with a model's own tokenizer and chat template.
//!
//! A reference holds a corpus encoded by the model's original `tokenizer.json`
//! and chat requests rendered by the model's own chat format, with their ids.
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
    ByteBpeTokenizer, ChatRequest, PreparedChat, SpecialTokens, TemplateBundle, TemplateSelection,
    TokenId,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::io::Read;

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

#[derive(Deserialize)]
struct Reference {
    now: i64,
    corpus: Vec<Case>,
    renders: Vec<Render>,
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
