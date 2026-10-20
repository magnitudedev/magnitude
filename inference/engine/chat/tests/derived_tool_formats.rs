//! Every template's tool format yields a constraint the grammar compiler
//! accepts, or no tool support at all. DavidAU's LFM2.5 8B-A1B finetune spells
//! its tool list `"List of tools: " + tools | tojson` rather than the stock
//! `"List of tools: ["`, so it missed the LFM2.5 handler; the generic analysis
//! could not separate its Python-style calls from the probe calls and emitted
//! GBNF the compiler refused ("unexpected production character"), failing
//! every tool request with a server error.
mod support;

use magnitude_chat::{
    ChatRequest, Event, PreparedChat, SpecialTokens, TemplateSelection, TerminalCause, TokenId,
    ToolChoice,
};
use serde_json::{json, Value};
use support::{bundle, byte_tokenizer, vocabulary};

const TBRILLIANCE: &str =
    include_str!("../../templates/tests/assets/LFM2.5-8B-A1B-Q3.8-TBrilliance.jinja");

fn request(choice: ToolChoice) -> ChatRequest {
    let mut request = ChatRequest::new(
        vec![json!({"role":"user","content":"What's the weather in Paris right now, in celsius? Use the tool."})],
        946684800,
    );
    request.tools = vec![json!({"type":"function","function":{
        "name":"get_weather","description":"Get the current weather",
        "parameters":{"type":"object","properties":{
            "city":{"type":"string"},
            "unit":{"type":"string","enum":["celsius","fahrenheit"]}
        },"required":["city"]}
    }})];
    request.tool_choice = choice;
    request
}

#[derive(Debug, Default)]
struct Transcript {
    content: String,
    reasoning: String,
    calls: Vec<(String, String)>,
}

/// Generates `output` under the prepared constraint, as constrained decoding
/// would, and reads it with the prepared parser.
fn generate(prepared: &PreparedChat, output: &str) -> Transcript {
    let tokenizer = byte_tokenizer("<|im_end|>");
    let plan = prepared.input().constraint.clone().unwrap();
    let mut grammar = vocabulary(&tokenizer)
        .bind(&plan.grammar, &plan.prefix)
        .unwrap();
    let mut tokens = tokenizer.encode(output, SpecialTokens::Recognize).unwrap();
    tokens.push(TokenId(256));
    for (index, token) in tokens.iter().enumerate() {
        grammar = grammar
            .advance(&[*token])
            .unwrap_or_else(|_| panic!("grammar refuses {:?}", &output[..index]));
    }
    let mut stream = prepared.native().stream(1 << 20).unwrap();
    let mut events = stream.feed(output.as_bytes()).unwrap();
    events.extend(stream.finish(TerminalCause::Natural).unwrap());
    let mut transcript = Transcript::default();
    for event in events {
        match event {
            Event::Content { text } => transcript.content.push_str(&text),
            Event::Reasoning { text } => transcript.reasoning.push_str(&text),
            Event::ToolStart { name, .. } => transcript.calls.push((name, String::new())),
            Event::ToolArguments { index, text } => {
                transcript.calls[index as usize].1.push_str(&text)
            }
            Event::ToolComplete { .. } | Event::Finish { .. } => {}
        }
    }
    transcript
}

/// The finetune's template takes the LFM2.5 format: its constraint compiles,
/// admits a call after the reasoning the generation prompt opens, and the call
/// and reasoning read back as a call and reasoning rather than content.
#[test]
fn lfm25_finetune_template_constrains_and_reads_its_calls() {
    let tokenizer = byte_tokenizer("<|im_end|>");
    let bundle = bundle(TBRILLIANCE, "<|startoftext|>", "<|im_end|>");
    assert!(bundle.inspect().unwrap().tools);
    for choice in [ToolChoice::Auto, ToolChoice::Required] {
        let prepared = PreparedChat::prepare(
            &bundle,
            &tokenizer,
            &request(choice),
            &TemplateSelection::default(),
        )
        .unwrap();
        assert!(prepared
            .native()
            .description()
            .prompt
            .ends_with("<|im_start|>assistant\n<think>\n"));
        let transcript = generate(
            &prepared,
            "The user wants the weather.</think><|tool_call_start|>[get_weather(city='Paris', unit='celsius')]<|tool_call_end|>",
        );
        assert_eq!(transcript.reasoning.trim(), "The user wants the weather.");
        assert!(transcript.content.trim().is_empty(), "{transcript:?}");
        assert_eq!(transcript.calls.len(), 1, "{transcript:?}");
        assert_eq!(transcript.calls[0].0, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&transcript.calls[0].1).unwrap(),
            json!({"city": "Paris", "unit": "celsius"})
        );
    }
}

/// The same Python-style calls under markers no handler knows: the generic
/// analysis cannot separate the call syntax from its probe data, so the
/// template has no tool support. Tool requests render the tools and read the
/// reply as content; a required call is refused as unsupported.
#[test]
fn underivable_tool_format_is_unsupported_rather_than_invalid() {
    let source = TBRILLIANCE
        .replace("<|tool_call_start|>", "<|calls|>")
        .replace("<|tool_call_end|>", "<|/calls|>");
    let tokenizer = byte_tokenizer("<|im_end|>");
    let bundle = bundle(&source, "<|startoftext|>", "<|im_end|>");
    assert!(!bundle.inspect().unwrap().tools);
    let prepared = PreparedChat::prepare(
        &bundle,
        &tokenizer,
        &request(ToolChoice::Auto),
        &TemplateSelection::default(),
    )
    .unwrap();
    assert!(prepared.input().constraint.is_none());
    assert!(prepared.native().description().prompt.contains("get_weather"));
    assert!(PreparedChat::prepare(
        &bundle,
        &tokenizer,
        &request(ToolChoice::Required),
        &TemplateSelection::default(),
    )
    .is_err());
}

/// The finetune's generation prompt always opens `<think>`, so a plain reply's
/// reasoning reads as reasoning and only the text after `</think>` as content.
#[test]
fn lfm25_finetune_reasoning_reads_as_reasoning() {
    let tokenizer = byte_tokenizer("<|im_end|>");
    let bundle = bundle(TBRILLIANCE, "<|startoftext|>", "<|im_end|>");
    let mut request = request(ToolChoice::Auto);
    request.tools.clear();
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    assert!(prepared.native().description().supports_thinking);
    let mut stream = prepared.native().stream(1 << 20).unwrap();
    let mut events = stream
        .feed(b"The user asks.\n</think>\n\nIt is 18 C in Paris.")
        .unwrap();
    events.extend(stream.finish(TerminalCause::Natural).unwrap());
    let (mut reasoning, mut content) = (String::new(), String::new());
    for event in events {
        match event {
            Event::Reasoning { text } => reasoning.push_str(&text),
            Event::Content { text } => content.push_str(&text),
            _ => {}
        }
    }
    assert_eq!(reasoning.trim(), "The user asks.");
    assert_eq!(content.trim(), "It is 18 C in Paris.");
}
