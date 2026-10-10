//! Tool-call text reads one way in MiniCPM5 and LFM2.5 when a free-form
//! argument makes the tool grammar recursive, so text before a call, the
//! arguments and the call itself are separate lexemes. Where two lexemes read
//! the same text, certification renders them one character at a time, and over
//! a real vocabulary the first mask of Claude Code's or Oh My Pi's tool set then
//! exceeded the matcher's item limit, refusing every request (the captured
//! sets bind in `real_vocabulary`).
mod support;

use magnitude_chat::{
    ChatRequest, PreparedChat, SpecialTokens, TemplateSelection, TokenId, ToolChoice,
};
use serde_json::{json, Value};
use support::{bundle, byte_tokenizer, vocabulary};

const MINICPM5: &str = include_str!("../../templates/tests/assets/MiniCPM5-2B.jinja");
const LFM25: &str = include_str!("../../templates/tests/assets/LFM2.5-2.6B.jinja");

fn request(tools: Vec<Value>, choice: ToolChoice) -> ChatRequest {
    let mut request = ChatRequest::new(
        vec![json!({"role": "user", "content": "Read notes.txt"})],
        946684800,
    );
    request.tools = tools;
    request.tool_choice = choice;
    request.parallel_tool_calls = true;
    request
}

/// Whether MiniCPM5's constraint under `choice`, with a tool taking a string
/// and a free-form argument, admits `output` and then the end of the turn.
fn minicpm5_admits(choice: ToolChoice, output: &str) -> bool {
    let tokenizer = byte_tokenizer("<|im_end|>");
    let bundle = bundle(MINICPM5, "<s>", "<|im_end|>");
    let tools = vec![json!({"type": "function", "function": {
        "name": "write", "description": "",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string"},
            "args": {}
        }}
    }})];
    let prepared = PreparedChat::prepare(
        &bundle,
        &tokenizer,
        &request(tools, choice),
        &TemplateSelection::default(),
    )
    .unwrap();
    assert_eq!(prepared.constraint().unwrap().report.character_lexemes, 0);
    let plan = prepared.input().constraint.clone().unwrap();
    let state = vocabulary(&tokenizer)
        .bind(&plan.grammar, &plan.prefix)
        .unwrap();
    let mut tokens = tokenizer.encode(output, SpecialTokens::Recognize).unwrap();
    tokens.push(TokenId(256));
    state.advance(&tokens).is_ok()
}

/// A MiniCPM5 string argument is CDATA exactly when it opens with
/// `<![CDATA[`, as the parser reads it; otherwise it ends at the first
/// `</param>`. A value that opens CDATA and never closes it is refused rather
/// than read as raw text.
#[test]
fn minicpm5_string_arguments_read_raw_or_cdata_one_way() {
    let call = |content: &str| {
        format!(
            "<function name=\"write\"><param name=\"content\">{content}</param>\n\
             <param name=\"args\">{{\"a\": [1, {{\"b\": null}}]}}</param>\n</function>"
        )
    };
    for content in [
        "plain <b>text</b>",
        "<![CDATA[x</param>y]]>",
        "<![CDATA",
        "<!-- note -->",
        "",
    ] {
        assert!(minicpm5_admits(ToolChoice::Auto, &call(content)), "{content}");
    }
    assert!(!minicpm5_admits(ToolChoice::Auto, &call("<![CDATA[unclosed")));
}

/// Text that opens with `<think>` is reasoning, so content no reasoning
/// precedes never opens with it; content after reasoning, or text that only
/// resembles the tag, is unaffected.
#[test]
fn minicpm5_content_never_opens_reasoning_it_does_not_close() {
    for output in [
        "Hello!",
        "<thinker> said hi",
        "<think>\nNo tool needed.\n</think>\n\nHello!",
        "<think>\nShow it.\n</think>\n\n<think> is a tag.",
    ] {
        assert!(minicpm5_admits(ToolChoice::Auto, output), "{output}");
    }
    assert!(!minicpm5_admits(ToolChoice::Auto, "<think>never closed"));
}

/// LFM2.5's Python-style values own no whitespace after them: the separators
/// around them do, so a dictionary or list followed by a separator reads one
/// way.
#[test]
fn lfm2_python_values_leave_whitespace_to_their_separators() {
    let tokenizer = byte_tokenizer("<|im_end|>");
    let bundle = bundle(LFM25, "<|startoftext|>", "<|im_end|>");
    let tools = vec![json!({"type": "function", "function": {
        "name": "run", "description": "",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "options": {}
        }}
    }})];
    let prepared = PreparedChat::prepare(
        &bundle,
        &tokenizer,
        &request(tools, ToolChoice::Auto),
        &TemplateSelection::default(),
    )
    .unwrap();
    assert_eq!(prepared.constraint().unwrap().report.character_lexemes, 0);
    let plan = prepared.input().constraint.clone().unwrap();
    let mut vocabulary = vocabulary(&tokenizer);
    let mut admits = |call: &str| {
        let state = vocabulary.bind(&plan.grammar, &plan.prefix).unwrap();
        let mut tokens = tokenizer
            .encode(
                &format!("Run it.</think><|tool_call_start|>[{call}]<|tool_call_end|>"),
                SpecialTokens::Recognize,
            )
            .unwrap();
        tokens.push(TokenId(256));
        state.advance(&tokens).is_ok()
    };
    for call in [
        "run(command=\"ls\", options={\"a\": [1, 2], \"b\": {}})",
        "run(options=[{'x': True} , None], command='ls')",
        "run( options={ \"a\" : [ 1 , 2 ] } )",
    ] {
        assert!(admits(call), "{call}");
    }
    assert!(!admits("run(options={\"a\": 1} , command=\"ls\")"));
}
