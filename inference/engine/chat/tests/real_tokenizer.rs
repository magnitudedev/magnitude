//! Token-exact agreement with a model's own tokenizer and chat template.
//!
//! The reference comes from the model's original `tokenizer.json` and
//! `transformers` `apply_chat_template` over the GGUF's template, written by
//! `inference/validation/tokenizer_reference.py`. Run with
//! `MAGNITUDE_TEST_GGUF=<name>.gguf MAGNITUDE_TEST_REFERENCE=<name>.json
//! cargo test -p magnitude-chat --test real_tokenizer -- --ignored`.
//! The GGUF may be a complete file or the generator's vocabulary-only header.
use magnitude_artifacts::{gguf::inspect_header, TemplatePayload, TokenizerPayload};
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer, ChatRequest, PreparedChat, SpecialTokens, TemplateBundle, TemplateSelection,
    TokenId,
};
use serde::Deserialize;
use serde_json::Value;

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

#[test]
#[ignore = "needs MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_REFERENCE"]
fn corpus_encodes_token_exactly_and_decodes_back() {
    let (tokenizer, _, reference) = load();
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
        let decoded = decode(&tokenizer, &case.ids);
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

#[test]
#[ignore = "needs MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_REFERENCE"]
fn chat_prompts_render_as_the_reference_and_begin_with_one_bos() {
    let (tokenizer, templates, reference) = load();
    let mut failures = Vec::new();
    for render in &reference.renders {
        let mut request = ChatRequest::new(render.messages.clone(), reference.now);
        request.tools = render.tools.clone();
        let prepared = match PreparedChat::prepare(
            &templates,
            &tokenizer,
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
        // The model input is the reference tokenization of the rendered text
        // with the implicit BOS present exactly once.
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
