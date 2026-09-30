//! A published tool call is never retracted. The output stream announces a
//! call once its opener parses, before its arguments arrive; generation runs
//! under the template's grammar, so that is safe exactly when the grammar
//! refuses every continuation the output parser would not read as that call.
//! Cline's `read_files` against Gemma 4 once broke this: the grammar admitted
//! a call body the parser rejects, and the stream failed with "Output parser
//! would retract a published tool call" mid-response.
mod support;

use magnitude_chat::{
    ChatRequest, Event, PreparedChat, SpecialTokens, TemplateSelection, TerminalCause, TokenId,
};
use serde_json::json;
use support::{bundle, byte_tokenizer, vocabulary};

const GEMMA4: &str = include_str!("../../templates/tests/assets/gemma-4-12B-it.jinja");
const QWEN35: &str = include_str!("../../templates/tests/assets/Qwen3.5-4B.jinja");

/// Cline's `read_files`: an array of objects with nullable integer bounds.
fn read_files() -> serde_json::Value {
    let bound = json!({"anyOf":[{"type":"integer","exclusiveMinimum":0},{"type":"null"}]});
    json!({"type":"function","function":{
        "name":"read_files","description":"Read files",
        "parameters":{"type":"object","properties":{"files":{"type":"array","items":{
            "type":"object",
            "properties":{"path":{"type":"string"},"start_line":bound,"end_line":bound},
            "required":["path"],"additionalProperties":false
        }}},"required":["files"],"additionalProperties":false}
    }})
}

/// Writes `output` one byte at a time as constrained generation would: a byte
/// the grammar refuses ends the output there. Every admitted byte must parse
/// without retracting a published call. Returns whether the whole output was
/// admitted, and the calls published.
fn generate(source: &str, output: &str) -> (bool, usize) {
    let tokenizer = byte_tokenizer("<eos>");
    let mut request = ChatRequest::new(
        vec![json!({"role":"user","content":"Read /a"})],
        946684800,
    );
    request.tools = vec![read_files()];
    let prepared = PreparedChat::prepare(
        &bundle(source, "<bos>", "<eos>"),
        &tokenizer,
        &request,
        &TemplateSelection::default(),
    )
    .unwrap();
    let plan = prepared.input().constraint.clone().unwrap();
    let mut vocabulary = vocabulary(&tokenizer);
    let mut grammar = vocabulary.bind(&plan.grammar, &plan.prefix).unwrap();
    let mut stream = prepared.native().stream(1 << 20).unwrap();
    let tokens = tokenizer.encode(output, SpecialTokens::Recognize).unwrap();
    assert_eq!(tokens.len(), output.len());
    let mut published = 0;
    for (index, token) in tokens.into_iter().enumerate() {
        let Ok(next) = grammar.advance(&[token]) else {
            return (false, published);
        };
        grammar = next;
        let events = stream
            .feed(&output.as_bytes()[index..=index])
            .unwrap_or_else(|error| panic!("{error} after {:?}", &output[..=index]));
        published += events
            .iter()
            .filter(|event| matches!(event, Event::ToolStart { .. }))
            .count();
    }
    assert!(grammar.advance(&[TokenId(256)]).is_ok(), "{output}");
    stream.finish(TerminalCause::Natural).unwrap();
    (true, published)
}

#[test]
fn gemma4_grammar_refuses_call_bodies_the_parser_would_retract() {
    assert_eq!(
        generate(
            GEMMA4,
            "<|tool_call>call:read_files{files:[{path:<|\"|>/a<|\"|>,start_line:3}]}<tool_call|>"
        ),
        (true, 1)
    );
    for slip in [
        // JSON quotes instead of Gemma's string delimiter.
        "<|tool_call>call:read_files{files:[{path:\"/a\"}]}<tool_call|>",
        // A bare value where the schema wants an array of objects.
        "<|tool_call>call:read_files{files:<|\"|>/a<|\"|>}<tool_call|>",
        // Text after a call opener that no call reads.
        "Reading.<|tool_call>call:read_files ls -la",
    ] {
        assert!(!generate(GEMMA4, slip).0, "{slip}");
    }
}

#[test]
fn qwen35_grammar_refuses_call_bodies_the_parser_would_retract() {
    let call = "<tool_call>\n<function=read_files>\n<parameter=files>\n[{\"path\": \"/a\"}]\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(generate(QWEN35, &format!("Read it.\n</think>\n\n{call}")), (true, 1));
    for slip in [
        "Read it.\n</think>\n\n<tool_call>\n<function=read_files>\n<parameter=files>\n/a\n</parameter>\n</function>\n</tool_call>",
        "Read it.\n</think>\n\n<tool_call>\n<function=read_files>\nls -la",
    ] {
        assert!(!generate(QWEN35, slip).0, "{slip}");
    }
}
