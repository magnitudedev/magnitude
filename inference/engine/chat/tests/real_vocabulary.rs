//! Structured output and tool grammars bound to a real model vocabulary.
//!
//! Every mask is computed over the full vocabulary trie, so these cases catch
//! grammars whose lexical structure makes mask computation exceed llguidance's
//! per-mask budgets — something byte-alphabet fixtures cannot observe.
//! Run with `MAGNITUDE_TEST_GGUF=<model.gguf> cargo test -p magnitude-chat
//! --test real_vocabulary -- --ignored`.
use magnitude_artifacts::Package;
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer, ChatRequest, PreparedChat, SpecialTokens, TemplateBundle, TemplateSelection,
    ToolChoice,
};
use magnitude_generation::Constraint;
use magnitude_grammar::{CacheLimits, GrammarConstraint, Vocabulary};
use serde_json::{json, Value};
use std::{sync::Arc, time::Instant};

struct Model {
    tokenizer: Arc<ByteBpeTokenizer>,
    templates: TemplateBundle,
    vocabulary: Vocabulary,
}

fn model() -> Model {
    let path = std::env::var("MAGNITUDE_TEST_GGUF").expect("MAGNITUDE_TEST_GGUF names a GGUF");
    let package = Package::open_without_projector(path).unwrap();
    let tokenizer = Arc::new(
        ByteBpeTokenizer::new(gguf_byte_bpe(package.tokenizer(), "test".into()).unwrap()).unwrap(),
    );
    let templates = gguf_templates(package.templates(), package.tokenizer()).unwrap();
    let vocabulary = Vocabulary::new(
        tokenizer.clone(),
        tokenizer.vocabulary(),
        CacheLimits {
            entries: 0,
            bytes: 0,
        },
    )
    .unwrap();
    Model {
        tokenizer,
        templates,
        vocabulary,
    }
}

fn allowed(mask: &[u32], token: u32) -> bool {
    mask[token as usize / 32] & (1 << (token % 32)) != 0
}

/// Every mask admits the stop tokens exactly where the grammar accepts: a
/// stop token is never spelled as text inside the grammar's free text.
fn ends_exactly_when_accepting(
    model: &Model,
    state: &GrammarConstraint,
    mask: &[u32],
) -> Result<(), String> {
    let accepting = state.accepting().map_err(|e| e.to_string())?;
    for stop in model.tokenizer.stop_tokens() {
        if allowed(mask, stop.0) != accepting {
            return Err(format!(
                "mask admits stop token {stop:?}: {}, grammar accepting: {accepting}",
                allowed(mask, stop.0)
            ));
        }
    }
    Ok(())
}

/// Bind the request's constraint, then walk `completion` token by token:
/// every step's mask must compute and admit the next token, and the grammar
/// must accept the complete text with an end of generation.
fn walk(model: &mut Model, request: &ChatRequest, completion: &str) -> Result<(), String> {
    let prepared = PreparedChat::prepare(
        &model.templates,
        &model.tokenizer,
        request,
        &TemplateSelection::default(),
    )
    .map_err(|error| format!("prepare: {error:?}"))?;
    let plan = prepared
        .input()
        .constraint
        .as_ref()
        .ok_or("request prepared without a constraint")?;
    // Template grammars compile with every lexeme at word level: a
    // character-level lexeme makes each of its bytes a parser step.
    let report = &prepared.constraint().unwrap().report;
    if report.character_lexemes != 0 {
        return Err(format!("grammar has character-level lexemes: {report:?}"));
    }
    let started = Instant::now();
    let mut state = model
        .vocabulary
        .bind(&plan.grammar, &plan.prefix)
        .map_err(|e| e.to_string())?;
    let bound = started.elapsed();
    let tokens = model
        .tokenizer
        .encode(completion, SpecialTokens::Recognize)?;
    let mut slowest = std::time::Duration::ZERO;
    for token in &tokens {
        let started = Instant::now();
        let mask = Constraint::mask(&state)?;
        slowest = slowest.max(started.elapsed());
        ends_exactly_when_accepting(model, &state, &mask)?;
        if !allowed(&mask, token.0) {
            let piece = model.tokenizer.piece(*token, false)?;
            return Err(format!(
                "mask rejects {:?} after {:?}",
                String::from_utf8_lossy(piece),
                state.position()
            ));
        }
        state = state.advance(&[*token]).map_err(|e| e.to_string())?;
    }
    let mask = Constraint::mask(&state)?;
    ends_exactly_when_accepting(model, &state, &mask)?;
    if !state.accepting().map_err(|e| e.to_string())? {
        return Err("grammar does not accept the complete text".into());
    }
    eprintln!(
        "bound in {bound:?}; slowest mask {slowest:?} over {} tokens",
        tokens.len()
    );
    Ok(())
}

/// The grammar must refuse `completion` at some token or at its end.
fn refuse(model: &mut Model, request: &ChatRequest, completion: &str) -> Result<(), String> {
    match walk(model, request, completion) {
        Ok(()) => Err(format!("grammar accepts {completion:?}")),
        Err(error) if error.starts_with("mask rejects") || error.contains("does not accept") => {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn ask(content: &str) -> ChatRequest {
    ChatRequest::new(vec![json!({"role": "user", "content": content})], 946684800)
}

fn schema(value: Value) -> ChatRequest {
    let mut request = ask("Describe a person.");
    request.json_schema = Some(value);
    request
}

fn tool(name: &str, parameters: Value) -> Value {
    json!({"type": "function", "function": {"name": name, "parameters": parameters}})
}

fn tools(tools: Vec<Value>, choice: ToolChoice, parallel: bool) -> ChatRequest {
    let mut request = ask("Use the tools.");
    request.tools = tools;
    request.tool_choice = choice;
    request.parallel_tool_calls = parallel;
    request
}

const THINK: &str = "Let me think about <this> {carefully}.\n</think>\n\n";

fn call(name: &str, arguments: &[(&str, &str)]) -> String {
    let mut text = format!("<tool_call>\n<function={name}>\n");
    for (key, value) in arguments {
        text.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
    }
    text.push_str("</function>\n</tool_call>");
    text
}

#[test]
#[ignore = "requires MAGNITUDE_TEST_GGUF"]
fn structured_output_and_tool_grammars_compute_masks_over_the_real_vocabulary() {
    let mut model = model();
    let person = json!({
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"]
    });
    let mut open_person = person.clone();
    open_person["additionalProperties"] = json!(true);
    let nested = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"type": "string", "enum": ["admin", "user"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "zip": {"type": "integer"}},
                "required": ["city"],
                "additionalProperties": false
            },
            "score": {"type": "number"},
            "active": {"type": "boolean"},
            "kind": {"const": "person"}
        },
        "required": ["name", "role", "address", "kind"],
        "additionalProperties": false
    });
    let edit = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "enum": ["src/main.rs"]},
            "mode": {"type": "string", "enum": ["append", "replace"]},
            "count": {"type": "integer", "enum": [1, 2, 3]},
            "text": {"type": "string"}
        },
        "required": ["path", "mode", "text"],
        "additionalProperties": false
    });
    let options = json!({
        "type": "object",
        "properties": {"query": {"type": "string"}, "filters": {"type": "object"}},
        "required": ["query"]
    });
    let cases: Vec<(&str, ChatRequest, String)> = vec![
        (
            "json_schema listing properties is closed by default",
            schema(person.clone()),
            format!("{THINK}{{\"name\": \"Ada {{x}}\", \"age\": 36}}"),
        ),
        (
            "json_schema explicitly allowing additional properties",
            schema(open_person.clone()),
            format!("{THINK}{{\"name\": \"Ada {{x}}\", \"age\": 36, \"extra\": {{\"a\": [1, \"b\", null]}}}}"),
        ),
        (
            "nested json_schema with enums, const, arrays and objects",
            schema(nested),
            format!(
                "{THINK}{{\"name\": \"Ada\", \"role\": \"admin\", \
                 \"address\": {{\"city\": \"Paris\", \"zip\": 75001}}, \"kind\": \"person\", \
                 \"tags\": [\"x\", \"y\"], \"score\": -1.5e3, \"active\": true}}"
            ),
        ),
        (
            "json_object",
            schema(json!({"type": "object"})),
            format!("{THINK}{{\"a\": {{\"b\": [1, 2, {{\"c\": \"d\"}}]}}}}"),
        ),
        (
            "auto tools answered with content",
            tools(vec![tool("edit", edit.clone())], ToolChoice::Auto, true),
            format!("{THINK}No tool is needed <here>."),
        ),
        (
            "required tool with string enums and a numeric enum",
            tools(vec![tool("edit", edit.clone())], ToolChoice::Required, false),
            format!(
                "{THINK}{}",
                call(
                    "edit",
                    &[("path", "src/main.rs"), ("mode", "replace"), ("text", "fn main() {}\n<x>"), ("count", "2")]
                )
            ),
        ),
        (
            "parallel calls with a recursive object parameter",
            tools(
                vec![tool("edit", edit.clone()), tool("search", options.clone())],
                ToolChoice::Auto,
                true,
            ),
            format!(
                "{THINK}{}\n{}",
                call("search", &[("query", "rust"), ("filters", "{\"lang\": {\"any\": [1, 2]}}")]),
                call("edit", &[("path", "src/main.rs"), ("mode", "append"), ("text", "x")])
            ),
        ),
        (
            "named tool",
            tools(
                vec![tool("edit", edit.clone()), tool("search", options.clone())],
                ToolChoice::Named("search".into()),
                true,
            ),
            format!("{THINK}{}", call("search", &[("query", "q")])),
        ),
        (
            "allowed tools",
            tools(
                vec![tool("edit", edit), tool("search", options)],
                ToolChoice::Allowed {
                    names: vec!["search".into()],
                    required: true,
                },
                false,
            ),
            format!("{THINK}{}", call("search", &[("query", "q")])),
        ),
    ];
    let many = (0..12)
        .map(|index| {
            tool(
                &format!("tool_{index}"),
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "options": {"type": "object"},
                        "items": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer"}
                    },
                    "required": ["path"]
                }),
            )
        })
        .collect::<Vec<_>>();
    let mut cases = cases;
    cases.push((
        "twelve tools with free-form object parameters",
        tools(many, ToolChoice::Auto, true),
        format!(
            "{THINK}{}",
            call(
                "tool_7",
                &[
                    ("path", "a"),
                    ("options", "{\"x\": {\"y\": [1]}}"),
                    ("limit", "3")
                ]
            )
        ),
    ));
    let enums = tools(
        vec![tool(
            "edit",
            json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["append", "replace"]},
                    "count": {"type": "integer", "enum": [1, 2]}
                },
                "required": ["mode"],
                "additionalProperties": false
            }),
        )],
        ToolChoice::Required,
        false,
    );
    let refusals: Vec<(&str, ChatRequest, String)> = vec![
        (
            "a value outside a string enum",
            enums.clone(),
            format!("{THINK}{}", call("edit", &[("mode", "delete")])),
        ),
        (
            "a value outside a numeric enum",
            enums.clone(),
            format!(
                "{THINK}{}",
                call("edit", &[("mode", "append"), ("count", "3")])
            ),
        ),
        (
            "content instead of a required call",
            enums,
            format!("{THINK}I will not call it."),
        ),
        (
            "an undeclared property of an object listing properties",
            schema(person.clone()),
            format!("{THINK}{{\"name\": \"Ada\", \"age\": 36, \"extra\": 1}}"),
        ),
        (
            "an object missing a required property",
            schema(person.clone()),
            format!("{THINK}{{\"name\": \"Ada\"}}"),
        ),
        (
            "structured output inside unclosed reasoning",
            schema(person),
            "Thinking {\"name\": \"Ada\", \"age\": 36}".into(),
        ),
    ];
    let mut failures = Vec::new();
    for (name, request, completion) in cases {
        eprintln!("case: {name}");
        if let Err(error) = walk(&mut model, &request, &completion) {
            failures.push(format!("{name}: {error}"));
        }
    }
    for (name, request, completion) in refusals {
        eprintln!("refusal: {name}");
        if let Err(error) = refuse(&mut model, &request, &completion) {
            failures.push(format!("{name}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Oh My Pi's `task` tool (issue #165, minimized): an array of objects with no
/// required properties, closed, holding free-form JSON in an `anyOf`. With
/// parallel calls the grammar must still compile fully lexical and bind.
#[test]
#[ignore = "requires MAGNITUDE_TEST_GGUF"]
fn harness_task_schema_binds_with_parallel_calls() {
    let mut model = model();
    let task = json!({
        "type": "object",
        "properties": {
            "i": {"type": "string"},
            "context": {"type": "string"},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string"},
                        "model": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                        "outputSchema": {"anyOf": [{"type": "object"}, {"type": "boolean"}, {"type": "string"}, {"type": "null"}]}
                    },
                    "required": [],
                    "additionalProperties": false
                }
            }
        },
        "required": ["context", "tasks", "i"],
        "additionalProperties": false
    });
    let request = tools(vec![tool("task", task)], ToolChoice::Auto, true);
    let completion = format!(
        "{THINK}{}",
        call(
            "task",
            &[
                ("i", "one"),
                ("context", "ctx"),
                ("tasks", "[{\"agent\": \"a\", \"outputSchema\": {\"type\": \"object\", \"x\": [1, 2]}}]"),
            ]
        )
    );
    walk(&mut model, &request, &completion).unwrap();
}

/// Claude Code's and Oh My Pi's tool sets (captured requests, schemas only)
/// once failed every request on MiniCPM5 and LFM2.5: their tool grammars had
/// character-level lexemes, and the first mask over the real vocabulary
/// exceeded the matcher's item limit.
#[test]
#[ignore = "requires MAGNITUDE_TEST_GGUF"]
fn harness_tool_sets_bind_over_the_real_vocabulary() {
    let mut model = model();
    let sets: Value = serde_json::from_str(include_str!("assets/harness-tools.json")).unwrap();
    let mut failures = Vec::new();
    for harness in ["claude-code", "oh-my-pi"] {
        let offered = sets[harness]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| tool(entry["name"].as_str().unwrap(), entry["parameters"].clone()))
            .collect::<Vec<_>>();
        for choice in [ToolChoice::Auto, ToolChoice::Required] {
            let case = format!("{harness} / {choice:?}");
            let request = tools(offered.clone(), choice, true);
            let prepared = PreparedChat::prepare(
                &model.templates,
                &model.tokenizer,
                &request,
                &TemplateSelection::default(),
            )
            .unwrap();
            let report = &prepared.constraint().unwrap().report;
            if report.character_lexemes != 0 {
                failures.push(format!("{case}: character-level lexemes: {report:?}"));
            }
            let plan = prepared.input().constraint.as_ref().unwrap();
            let bound = model
                .vocabulary
                .bind(&plan.grammar, &plan.prefix)
                .map_err(|error| error.to_string())
                .and_then(|state| Constraint::mask(&state).map(drop));
            if let Err(error) = bound {
                failures.push(format!("{case}: {error}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
