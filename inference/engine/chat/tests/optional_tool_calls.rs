//! Offered tools under the default `auto` choice constrain the whole
//! completion, which must still admit a turn that answers in text and ends:
//! a constraint that demands a call leaves the model no end but degenerate
//! text. A required choice still demands one.
mod support;

use magnitude_chat::{
    ChatRequest, PreparedChat, SpecialTokens, TemplateSelection, TokenId, ToolChoice,
};
use serde_json::json;
use support::{bundle, byte_tokenizer, vocabulary};

const LFM25: &str = include_str!("../../templates/tests/assets/LFM2.5-2.6B.jinja");
const MINICPM5: &str = include_str!("../../templates/tests/assets/MiniCPM5-2B.jinja");
const QWEN35: &str = include_str!("../../templates/tests/assets/Qwen3.5-4B.jinja");

/// Whether the constraint of `source` under `choice` admits `output` and
/// then the end of the turn.
fn admits(source: &str, bos: &str, choice: ToolChoice, output: &str) -> bool {
    advances(source, bos, choice, output, true)
}

/// Whether the constraint of `source` under `choice` refuses `output` itself,
/// before any end of the turn is asked for.
fn rejects_before_the_end(source: &str, bos: &str, choice: ToolChoice, output: &str) -> bool {
    !advances(source, bos, choice, output, false)
}

fn advances(source: &str, bos: &str, choice: ToolChoice, output: &str, end: bool) -> bool {
    let tokenizer = byte_tokenizer("<|im_end|>");
    let bundle = bundle(source, bos, "<|im_end|>");
    let mut request = ChatRequest::new(
        vec![json!({"role":"user","content":"List the files."})],
        946684800,
    );
    request.tools = vec![json!({"type":"function","function":{
        "name":"bash","description":"Run a shell command",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
    }})];
    request.tool_choice = choice;
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    let plan = prepared.input().constraint.clone().unwrap();
    let mut vocabulary = vocabulary(&tokenizer);
    let state = vocabulary.bind(&plan.grammar, &plan.prefix).unwrap();
    let mut tokens = tokenizer.encode(output, SpecialTokens::Recognize).unwrap();
    if end {
        tokens.push(TokenId(256));
    }
    state.advance(&tokens).is_ok()
}

#[test]
fn minicpm5_answers_in_text_or_calls_under_auto() {
    let (text, call) = (
        "<think>\nNo tool needed.\n</think>\n\nHello!",
        "<think>\nList them.\n</think>\n\n<function name=\"bash\"><param name=\"command\">ls</param></function>",
    );
    assert!(admits(MINICPM5, "<s>", ToolChoice::Auto, text));
    assert!(admits(MINICPM5, "<s>", ToolChoice::Auto, "Hello!"));
    assert!(admits(MINICPM5, "<s>", ToolChoice::Auto, call));
    assert!(!admits(MINICPM5, "<s>", ToolChoice::Required, text));
    assert!(admits(MINICPM5, "<s>", ToolChoice::Required, call));
}

#[test]
fn lfm2_answers_in_text_or_calls_under_auto() {
    // The generation prompt opens reasoning, which closes before the turn
    // ends: the parser reads text that opens with it as reasoning.
    let (text, call) = (
        "No tool needed.</think>Hello!",
        "<|tool_call_start|>[bash(command=\"ls\")]<|tool_call_end|>",
    );
    let reasoned = format!("List them.</think>{call}");
    assert!(admits(LFM25, "<|startoftext|>", ToolChoice::Auto, text));
    assert!(admits(LFM25, "<|startoftext|>", ToolChoice::Auto, &reasoned));
    assert!(!admits(LFM25, "<|startoftext|>", ToolChoice::Auto, "Hello!"));
    assert!(!admits(LFM25, "<|startoftext|>", ToolChoice::Auto, call));
    assert!(!admits(LFM25, "<|startoftext|>", ToolChoice::Required, text));
    assert!(admits(LFM25, "<|startoftext|>", ToolChoice::Required, &reasoned));
}

/// Reasoning the generation prompt opens closes before the turn ends: the
/// parser streams it as reasoning, and a constraint that read it as content
/// instead would let the turn end with that reasoning never closed.
#[test]
fn reasoning_opened_by_the_prompt_closes_before_the_turn_ends() {
    let call =
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
    for choice in [ToolChoice::Auto, ToolChoice::Required] {
        assert!(!admits(QWEN35, "", choice.clone(), "Still thinking <here>."));
        assert!(admits(QWEN35, "", choice.clone(), &format!("List them.\n{call}")));
        assert!(admits(
            QWEN35,
            "",
            choice,
            &format!("List them.\n</think>\n\n{call}")
        ));
    }
    assert!(admits(
        QWEN35,
        "",
        ToolChoice::Auto,
        "List them.\n</think>\n\nHello!"
    ));
}

/// A required call follows reasoning directly. Text before it is refused as
/// it is written: a constraint that admitted it could never end, and a model
/// that answers in text would repeat itself until it ran out of tokens.
#[test]
fn required_calls_admit_no_text_before_the_call() {
    let lfm2_call = "<|tool_call_start|>[bash(command=\"ls\")]<|tool_call_end|>";
    let minicpm5_call = "<function name=\"bash\"><param name=\"command\">ls</param></function>";
    let qwen_call =
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
    for (source, bos, reasoning, call) in [
        (LFM25, "<|startoftext|>", "List them.</think>", lfm2_call),
        (MINICPM5, "<s>", "<think>\nList them.\n</think>\n\n", minicpm5_call),
        (QWEN35, "", "List them.\n</think>\n\n", qwen_call),
    ] {
        assert!(admits(source, bos, ToolChoice::Required, &format!("{reasoning}{call}")));
        assert!(admits(source, bos, ToolChoice::Auto, &format!("{reasoning}Sure.{call}")));
        assert!(
            rejects_before_the_end(source, bos, ToolChoice::Required, &format!("{reasoning}Sure.")),
            "{call}"
        );
        assert!(!admits(
            source,
            bos,
            ToolChoice::Required,
            &format!("{reasoning}Sure.{call}")
        ));
    }
}
