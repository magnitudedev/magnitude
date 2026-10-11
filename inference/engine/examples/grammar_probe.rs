//! Constrained-decoding cost on a real vocabulary: compiles each BFCL
//! request's template grammar, binds it, and times the mask at every token
//! of a plausible completion (reasoning, then the reference tool calls).
//!
//! GGUF=<model.gguf> BFCL=<dir with BFCL_v4_*.json and possible_answer/>
//! cargo run --release --example grammar_probe [-- scale]
use magnitude_artifacts::Package;
use magnitude_chat::{
    artifacts::{gguf_byte_bpe, gguf_templates},
    ByteBpeTokenizer, ChatRequest, PreparedChat, SpecialTokens, TemplateBundle, TemplateSelection,
    ToolChoice,
};
use magnitude_engine::generation::Constraint;
use magnitude_grammar::{CacheLimits, Vocabulary};
use serde_json::{json, Map, Value};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

const THINK: &str = "The user wants me to call the tools. Let me figure out the right \
arguments, being careful with units {like \"km\"} and <tags>.\n</think>\n\n";

fn normalize(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
        Value::Object(object) => {
            let mut result = Map::new();
            for (key, item) in object {
                if key == "optional" || key == "format" || (key == "type" && item == "any") {
                    continue;
                }
                if key == "required" {
                    let Some(properties) = object.get("properties").and_then(Value::as_object)
                    else {
                        continue;
                    };
                    let kept = item
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|name| properties.contains_key(name.as_str().unwrap()))
                        .cloned()
                        .collect::<Vec<_>>();
                    if !kept.is_empty() {
                        result.insert(key.clone(), Value::Array(kept));
                    }
                    continue;
                }
                let value = match (key.as_str(), item.as_str()) {
                    ("type", Some("dict")) => json!("object"),
                    ("type", Some("list" | "tuple")) => json!("array"),
                    ("type", Some("float")) => json!("number"),
                    _ => normalize(item),
                };
                result.insert(key.clone(), value);
            }
            Value::Object(result)
        }
        other => other.clone(),
    }
}

fn tool(function: &Value) -> Value {
    json!({"type": "function", "function": {
        "name": function["name"],
        "description": function.get("description").cloned().unwrap_or(json!("BFCL function")),
        "parameters": normalize(&function["parameters"]),
    }})
}

fn completion(answer: &Value) -> String {
    let mut text = THINK.to_string();
    for (index, entry) in answer["ground_truth"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        for (name, arguments) in entry.as_object().unwrap() {
            if index > 0 {
                text.push('\n');
            }
            text.push_str(&format!("<tool_call>\n<function={name}>\n"));
            for (key, values) in arguments.as_object().unwrap() {
                let value = match values {
                    Value::Array(values) => values[0].clone(),
                    value => value.clone(),
                };
                if value == json!("") {
                    continue;
                }
                let rendered = match &value {
                    Value::String(text) => text.clone(),
                    value => value.to_string(),
                };
                text.push_str(&format!("<parameter={key}>\n{rendered}\n</parameter>\n"));
            }
            text.push_str("</function>\n</tool_call>");
        }
    }
    text
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn summary(mut times: Vec<Duration>) -> String {
    if times.is_empty() {
        return "-".into();
    }
    let mean = times.iter().map(|t| millis(*t)).sum::<f64>() / times.len() as f64;
    times.sort();
    let p95 = millis(times[(times.len() - 1) * 95 / 100]);
    let max = millis(*times.last().unwrap());
    format!("mean={mean:.2} p95={p95:.2} max={max:.2}")
}

fn probe(
    label: &str,
    tokenizer: &Arc<ByteBpeTokenizer>,
    templates: &TemplateBundle,
    vocabulary: &mut Vocabulary,
    request: &ChatRequest,
    text: &str,
) {
    let prepared =
        match PreparedChat::prepare(templates, tokenizer, request, &TemplateSelection::default()) {
            Ok(prepared) => prepared,
            Err(error) => return println!("{label} prepare failed: {error}"),
        };
    let plan = prepared.input().constraint.clone().unwrap();
    let report = &prepared.constraint().unwrap().report;
    let started = Instant::now();
    let mut state = match vocabulary.bind(&plan.grammar, &plan.prefix) {
        Ok(state) => state,
        Err(error) => return println!("{label} bind failed: {error}"),
    };
    let bind = started.elapsed();
    let tokens = tokenizer.encode(text, SpecialTokens::Recognize).unwrap();
    let thinking = tokenizer
        .encode(THINK, SpecialTokens::Recognize)
        .unwrap()
        .len();
    let (mut first, mut reasoning, mut calls) = (Duration::ZERO, Vec::new(), Vec::new());
    let mut failure = String::new();
    for (index, token) in tokens.iter().enumerate() {
        let started = Instant::now();
        let mask = match Constraint::mask(&state) {
            Ok(mask) => mask,
            Err(error) => {
                failure = format!("mask@{index}: {error}");
                break;
            }
        };
        let elapsed = started.elapsed();
        match index {
            0 => first = elapsed,
            _ if index < thinking => reasoning.push(elapsed),
            _ => calls.push(elapsed),
        }
        if mask[token.0 as usize / 32] & (1 << (token.0 % 32)) == 0 {
            failure = format!(
                "rejected {:?} at {index}",
                String::from_utf8_lossy(tokenizer.piece(*token, false).unwrap())
            );
            break;
        }
        state = state.advance(&[*token]).unwrap();
    }
    println!(
        "{label:<24} gbnf={:>6} lark={:>7} earley={:>3} lexemes={:>4} chars={:>2} unflat={:>2} compile={:>7.2}ms bind={:>7.2}ms mask0={:>5.2} | reasoning {} | calls {} | n={} {failure}",
        report.gbnf_bytes,
        report.lark_bytes,
        report.earley_rules,
        report.lexemes,
        report.character_lexemes,
        report.unflattened,
        report.compile_us as f64 / 1e3,
        millis(bind),
        millis(first),
        summary(reasoning),
        summary(calls),
        tokens.len(),
    );
}

fn read_lines(path: &str) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn main() {
    let gguf = std::env::var("GGUF").expect("GGUF names a model");
    let bfcl = std::env::var("BFCL").expect("BFCL names the BFCL v4 data directory");
    let scale = std::env::args().any(|argument| argument == "scale");
    let package = Package::open_without_projector(gguf).unwrap();
    let tokenizer = Arc::new(
        ByteBpeTokenizer::new(gguf_byte_bpe(package.tokenizer(), "probe".into()).unwrap()).unwrap(),
    );
    let templates = gguf_templates(package.templates(), package.tokenizer()).unwrap();
    let mut vocabulary = Vocabulary::new(
        tokenizer.clone(),
        tokenizer.vocabulary(),
        CacheLimits {
            entries: 0,
            bytes: 0,
        },
    )
    .unwrap();
    let request = |messages: &Value, tools: Vec<Value>| {
        let mut request = ChatRequest::new(messages.as_array().unwrap().clone(), 946684800);
        request.tools = tools;
        request.tool_choice = ToolChoice::Auto;
        request.parallel_tool_calls = true;
        request
    };
    let mut functions = Vec::new();
    for category in ["simple_python", "parallel", "parallel_multiple"] {
        let questions = read_lines(&format!("{bfcl}/BFCL_v4_{category}.json"));
        let answers = read_lines(&format!("{bfcl}/possible_answer/BFCL_v4_{category}.json"));
        for (question, answer) in questions.iter().zip(&answers) {
            functions.extend(question["function"].as_array().unwrap().iter().cloned());
            if scale {
                continue;
            }
            let turns = &question["question"];
            let messages = if turns[0].is_array() {
                &turns[0]
            } else {
                turns
            };
            let tools = question["function"]
                .as_array()
                .unwrap()
                .iter()
                .map(tool)
                .collect();
            probe(
                question["id"].as_str().unwrap(),
                &tokenizer,
                &templates,
                &mut vocabulary,
                &request(messages, tools),
                &completion(answer),
            );
        }
    }
    if scale {
        let questions = read_lines(&format!("{bfcl}/BFCL_v4_simple_python.json"));
        let answers = read_lines(&format!(
            "{bfcl}/possible_answer/BFCL_v4_simple_python.json"
        ));
        let mut seen = std::collections::BTreeSet::new();
        let unique = std::iter::once(questions[0]["function"][0].clone())
            .chain(functions)
            .filter(|function| seen.insert(function["name"].as_str().unwrap().to_string()))
            .collect::<Vec<_>>();
        for count in [1, 2, 4, 8, 16, 32, 64, 128] {
            let tools = unique.iter().take(count).map(tool).collect();
            probe(
                &format!("scale-{count}"),
                &tokenizer,
                &templates,
                &mut vocabulary,
                &request(&questions[0]["question"][0], tools),
                &completion(&answers[0]),
            );
        }
    }
}
