//! Gemma 4 tool calls are generated under a grammar derived from each
//! function's schema: the constraint admits exactly the argument dictionaries
//! the schema allows, in the template's key order.
use magnitude_chat::{
    BpeConfig, ByteBpeTokenizer, CacheLimits, ChatRequest, Normalization, PieceEncoding, PieceKind,
    PreparedChat, SpecialTokens, Split, SplitBehavior, TemplateBundle, TemplateSelection,
    TemplateVariant, TokenId, ToolChoice, Vocabulary,
};
use serde_json::json;
use std::{collections::BTreeSet, sync::Arc};

const GEMMA4: &str = include_str!("../../templates/tests/assets/gemma-4-12B-it.jinja");

/// Byte pieces 0..=255 and the stop token `<eos>` 256.
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
    pieces.push("<eos>".into());
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

#[test]
fn gemma4_tool_constraint_admits_exactly_the_schema_arguments() {
    let tokenizer = tokenizer();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: GEMMA4.into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        [
            ("bos_token".to_string(), "<bos>".to_string()),
            ("eos_token".to_string(), "<eos>".to_string()),
        ]
        .into(),
    )
    .unwrap();
    let mut request = ChatRequest::new(
        vec![json!({"role":"user","content":"Forecast for Oslo?"})],
        946684800,
    );
    request.tools = vec![json!({"type":"function","function":{
        "name":"forecast","description":"Forecast",
        "parameters":{"type":"object","properties":{
            "units":{"type":"string","enum":["metric","imperial"]},
            "city":{"type":"string"},
            "Days":{"type":"integer"},
            "tags":{"type":"array","items":{"type":"string"}}
        },"required":["city","units"]}
    }})];
    request.tool_choice = ToolChoice::Required;
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
    let mut accepts = |arguments: &str| {
        let state = vocabulary.bind(&plan).unwrap();
        let mut tokens = tokenizer
            .encode(
                &format!("<|tool_call>call:forecast{arguments}<tool_call|>"),
                SpecialTokens::Recognize,
            )
            .unwrap();
        tokens.push(TokenId(256));
        state.advance(&tokens).is_ok()
    };
    for arguments in [
        "{city:<|\"|>Oslo<|\"|>,units:<|\"|>metric<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>,Days:3,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],units:<|\"|>imperial<|\"|>}",
        "{city:<|\"|>line\nbreak<|\"|>,Days:-12,tags:[],units:<|\"|>metric<|\"|>}",
    ] {
        assert!(accepts(arguments), "{arguments}");
    }
    for arguments in [
        "{units:<|\"|>metric<|\"|>,city:<|\"|>Oslo<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>,units:<|\"|>kelvin<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>,Days:2.5,units:<|\"|>metric<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>,units:<|\"|>metric<|\"|>,zone:1}",
        "{city:3,units:<|\"|>metric<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>,tags:[1],units:<|\"|>metric<|\"|>}",
    ] {
        assert!(!accepts(arguments), "{arguments}");
    }
}
