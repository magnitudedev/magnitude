//! Gemma 4 tool calls are generated under a grammar derived from each
//! function's schema: the constraint admits exactly the argument dictionaries
//! the schema allows, in the template's key order.
mod support;

use magnitude_chat::{
    ChatRequest, PreparedChat, SpecialTokens, TemplateBundle, TemplateSelection, TokenId,
    ToolChoice,
};
use serde_json::json;
use support::{byte_tokenizer, vocabulary};

const GEMMA4: &str = include_str!("../../templates/tests/assets/gemma-4-12B-it.jinja");

fn gemma4() -> TemplateBundle {
    support::bundle(GEMMA4, "<bos>", "<eos>")
}

#[test]
fn gemma4_tool_constraint_admits_exactly_the_schema_arguments() {
    let tokenizer = byte_tokenizer("<eos>");
    let bundle = gemma4();
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
    let mut vocabulary = vocabulary(&tokenizer);
    let mut accepts = |arguments: &str| {
        let state = vocabulary.bind(&plan.grammar, &plan.prefix).unwrap();
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

/// Free-form object arguments make the tool grammar recursive, so the text
/// before a call and the call itself are separate lexemes. The text must still
/// lex greedily at word level: a character-level free-text lexeme makes every
/// byte a parser step, and a real vocabulary's mask then exceeds llguidance's
/// item budget at binding.
#[test]
fn gemma4_free_form_object_arguments_bind_at_word_level() {
    let tokenizer = byte_tokenizer("<eos>");
    let bundle = gemma4();
    let mut request = ChatRequest::new(
        vec![json!({"role":"user","content":"Configure it."})],
        946684800,
    );
    request.tools = vec![
        json!({"type":"function","function":{
            "name":"configure","description":"Configure",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string"},
                "options":{"type":"object"},
                "extra":{}
            },"required":["path"]}
        }}),
        json!({"type":"function","function":{
            "name":"anything","description":"Anything",
            "parameters":{"type":"object"}
        }}),
    ];
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    let report = &prepared.constraint().unwrap().report;
    assert!(report.earley_rules > 1, "{report:?}");
    assert_eq!(report.character_lexemes, 0, "{report:?}");
    let plan = prepared.input().constraint.clone().unwrap();
    let mut vocabulary = vocabulary(&tokenizer);
    let mut accepts = |output: &str| {
        let state = vocabulary.bind(&plan.grammar, &plan.prefix).unwrap();
        let mut tokens = tokenizer.encode(output, SpecialTokens::Recognize).unwrap();
        tokens.push(TokenId(256));
        state.advance(&tokens).is_ok()
    };
    let nested = "{mode:<|\"|>fast<|\"|>,limits:{depth:3,tags:[<|\"|>a<|\"|>,{deep:[1,2]}]},on:true}";
    for output in [
        "Let me configure it.".to_owned(),
        // Declared keys in the template's (sorted) order.
        format!("<|tool_call>call:configure{{options:{nested},path:<|\"|>/a<|\"|>}}<tool_call|>"),
        format!("Sure.<|tool_call>call:anything{nested}<tool_call|>"),
        "<|tool_call>call:anything{}<tool_call|><|tool_call>call:configure{path:<|\"|>/b<|\"|>}<tool_call|>"
            .to_owned(),
    ] {
        assert!(accepts(&output), "{output}");
    }
}

/// A required call may follow only a thought: text before it could never
/// end, since the turn cannot end without the call.
#[test]
fn gemma4_required_call_admits_no_text_before_it() {
    let tokenizer = byte_tokenizer("<eos>");
    let bundle = gemma4();
    let call = "<|tool_call>call:lookup{key:<|\"|>a<|\"|>}<tool_call|>";
    let mut accepts = |choice: ToolChoice, output: &str| {
        let mut request = ChatRequest::new(
            vec![json!({"role":"user","content":"Look it up."})],
            946684800,
        );
        request.tools = vec![json!({"type":"function","function":{
            "name":"lookup","description":"Lookup",
            "parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}
        }})];
        request.tool_choice = choice;
        let prepared =
            PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
                .unwrap();
        let plan = prepared.input().constraint.clone().unwrap();
        let state = vocabulary(&tokenizer).bind(&plan.grammar, &plan.prefix).unwrap();
        let mut tokens = tokenizer.encode(output, SpecialTokens::Recognize).unwrap();
        tokens.push(TokenId(256));
        state.advance(&tokens).is_ok()
    };
    let named = ToolChoice::Named("lookup".into());
    for choice in [ToolChoice::Required, named] {
        assert!(accepts(choice.clone(), call));
        assert!(!accepts(choice.clone(), &format!("Sure.{call}")));
        assert!(!accepts(choice, "Sure."));
    }
    assert!(accepts(ToolChoice::Auto, &format!("Sure.{call}")));
}
