//! Offered tools under the default `auto` choice constrain the whole
//! completion, which must still admit a turn that answers in text and ends:
//! a constraint that demands a call leaves the model no end but degenerate
//! text. A required choice still demands one.
use magnitude_chat::{
    BpeConfig, ByteBpeTokenizer, CacheLimits, ChatRequest, Normalization, PieceEncoding, PieceKind,
    PreparedChat, SpecialTokens, Split, SplitBehavior, TemplateBundle, TemplateSelection,
    TemplateVariant, TokenId, ToolChoice, Vocabulary,
};
use serde_json::json;
use std::{collections::BTreeSet, sync::Arc};

const LFM25: &str = include_str!("../../templates/tests/assets/LFM2.5-2.6B.jinja");
const MINICPM5: &str = include_str!("../../templates/tests/assets/MiniCPM5-2B.jinja");

/// Byte pieces 0..=255 and the stop token 256.
fn tokenizer() -> Arc<ByteBpeTokenizer> {
    let mut bytes: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut alphabet: Vec<u32> = bytes.iter().map(|&byte| u32::from(byte)).collect();
    let mut next = 256;
    for byte in 0..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            alphabet.push(next);
            next += 1;
        }
    }
    let mut pieces = vec![String::new(); 256];
    for (byte, code) in bytes.into_iter().zip(alphabet) {
        pieces[byte as usize] = char::from_u32(code).unwrap().to_string();
    }
    pieces.push("<|im_end|>".into());
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.push(PieceKind::Control);
    Arc::new(
        ByteBpeTokenizer::new(BpeConfig {
            artifact_identity: "fixture".into(),
            pieces,
            kinds,
            merges: vec![],
            normalization: Normalization::None,
            splits: vec![Split {
                pattern: r".+|\s".into(),
                behavior: SplitBehavior::Isolated,
            }],
            encoding: PieceEncoding::ByteLevel,
            ignore_merges: false,
            implicit_bos: None,
            stop_tokens: BTreeSet::from([TokenId(256)]),
            suppressed_tokens: BTreeSet::new(),
        })
        .unwrap(),
    )
}

/// Whether the constraint of `source` under `choice` admits `output` and
/// then the end of the turn.
fn admits(source: &str, bos: &str, choice: ToolChoice, output: &str) -> bool {
    let tokenizer = tokenizer();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: source.into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        [
            ("bos_token".to_string(), bos.to_string()),
            ("eos_token".to_string(), "<|im_end|>".to_string()),
        ]
        .into(),
    )
    .unwrap();
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
    let mut vocabulary = Vocabulary::new(
        tokenizer.clone(),
        tokenizer.vocabulary(),
        CacheLimits {
            entries: 0,
            bytes: 0,
        },
    )
    .unwrap();
    let state = vocabulary.bind(&plan).unwrap();
    let mut tokens = tokenizer.encode(output, SpecialTokens::Recognize).unwrap();
    tokens.push(TokenId(256));
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
    let (text, call) = (
        "Hello!",
        "<|tool_call_start|>[bash(command=\"ls\")]<|tool_call_end|>",
    );
    assert!(admits(LFM25, "<|startoftext|>", ToolChoice::Auto, text));
    assert!(admits(LFM25, "<|startoftext|>", ToolChoice::Auto, call));
    assert!(!admits(LFM25, "<|startoftext|>", ToolChoice::Required, text));
    assert!(admits(LFM25, "<|startoftext|>", ToolChoice::Required, call));
}
