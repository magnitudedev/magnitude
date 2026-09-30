//! Every valid JSON Schema lowers to a grammar that admits values within the
//! schema, and exactly those values wherever no relaxation is recorded.
mod support;

use magnitude_chat::{
    ByteBpeTokenizer, ChatRequest, PreparedChat, SpecialTokens, TemplateBundle, TemplateSelection,
    TokenId, ToolChoice,
};
use magnitude_grammar::Vocabulary;
use magnitude_templates::{Relaxation, RelaxationReason, RelaxationSubject};
use serde_json::{json, Value};
use std::sync::Arc;

const ASSETS: &[(&str, &str)] = &[
    (
        "Qwen3",
        include_str!("../../templates/tests/assets/Qwen-Qwen3-0.6B.jinja"),
    ),
    (
        "Qwen3.5",
        include_str!("../../templates/tests/assets/Qwen3.5-4B.jinja"),
    ),
    (
        "Gemma 4",
        include_str!("../../templates/tests/assets/gemma-4-12B-it.jinja"),
    ),
    (
        "LFM2.5",
        include_str!("../../templates/tests/assets/LFM2.5-2.6B.jinja"),
    ),
    (
        "LFM2.5 MoE",
        include_str!("../../templates/tests/assets/LFM2.5-8B-A1B.jinja"),
    ),
    (
        "Laguna",
        include_str!("../../templates/tests/assets/Laguna-S-2.1.jinja"),
    ),
    (
        "MiniCPM5",
        include_str!("../../templates/tests/assets/MiniCPM5-2B.jinja"),
    ),
    (
        "Muse",
        include_str!("../../templates/tests/assets/Muse-Glimmer-30B.jinja"),
    ),
    (
        "Nemotron",
        include_str!("../../templates/tests/assets/NVIDIA-Nemotron-3.5-Lightning-30B-A3B.jinja"),
    ),
];

fn tokenizer() -> Arc<ByteBpeTokenizer> {
    support::byte_tokenizer("<|stop|>")
}

fn bundle(source: &str) -> TemplateBundle {
    support::bundle(source, "<s>", "</s>")
}

/// JSON output constrained by one schema under the Qwen3 template.
struct JsonOutput {
    tokenizer: Arc<ByteBpeTokenizer>,
    bundle: TemplateBundle,
    vocabulary: Vocabulary,
}

impl JsonOutput {
    fn new() -> Self {
        let tokenizer = tokenizer();
        Self {
            vocabulary: support::vocabulary(&tokenizer),
            tokenizer,
            bundle: bundle(ASSETS[0].1),
        }
    }

    fn prepare(&self, schema: Value) -> PreparedChat {
        let mut request = ChatRequest::new(vec![json!({"role":"user","content":"Hi"})], 946684800);
        request.json_schema = Some(schema);
        PreparedChat::prepare(
            &self.bundle,
            &self.tokenizer,
            &request,
            &TemplateSelection::default(),
        )
        .unwrap()
    }

    /// Whether the constraint admits `text` as the complete JSON output.
    fn accepts(&mut self, prepared: &PreparedChat, text: &str) -> bool {
        let plan = prepared.input().constraint.as_ref().unwrap();
        let state = self.vocabulary.bind(&plan.grammar, &plan.prefix).unwrap();
        let mut tokens = self
            .tokenizer
            .encode(text, SpecialTokens::Recognize)
            .unwrap();
        tokens.push(TokenId(256));
        state.advance(&tokens).is_ok()
    }
}

fn relaxations(prepared: &PreparedChat) -> &[Relaxation] {
    &prepared.native().description().relaxations
}

fn output_relaxation(path: &str, keyword: &str, reason: RelaxationReason) -> Relaxation {
    Relaxation {
        subject: RelaxationSubject::Output,
        path: path.into(),
        keyword: keyword.into(),
        reason,
    }
}

/// A plain decimal JSON number and its value.
fn plain_decimal(text: &str) -> Option<f64> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    let (integer, fraction) = digits
        .split_once('.')
        .map_or((digits, None), |(i, f)| (i, Some(f)));
    let canonical_integer = integer == "0" || (!integer.is_empty() && !integer.starts_with('0'));
    let valid_fraction = fraction.map_or(true, |f| !f.is_empty());
    let all_digits = integer
        .chars()
        .chain(fraction.unwrap_or("").chars())
        .all(|c| c.is_ascii_digit());
    (canonical_integer && valid_fraction && all_digits).then(|| text.parse().unwrap())
}

#[test]
fn bounded_numbers_admit_exactly_the_values_within_their_bounds() {
    let mut output = JsonOutput::new();
    let bounds: Vec<Option<f64>> = vec![None, Some(-1.5), Some(0.0), Some(0.25), Some(2.0)];
    let mut candidates = Vec::new();
    for sign in ["", "-"] {
        for integer in 0..=3 {
            candidates.push(format!("{sign}{integer}"));
            // Around each bound: below, equal (also with trailing zeros), above.
            for fraction in ["0", "2", "25", "250", "251", "5", "50", "9", "001"] {
                candidates.push(format!("{sign}{integer}.{fraction}"));
            }
        }
    }
    candidates.extend(["1e3", "2E-1", "01", "1.", ".5", "--1"].map(String::from));
    for integral in [false, true] {
        for minimum in &bounds {
            for maximum in &bounds {
                for exclusive_minimum in [false, true] {
                    for exclusive_maximum in [false, true] {
                        if (minimum.is_none() && exclusive_minimum)
                            || (maximum.is_none() && exclusive_maximum)
                            || (minimum.is_none() && maximum.is_none())
                        {
                            continue;
                        }
                        let mut schema =
                            json!({"type": if integral { "integer" } else { "number" }});
                        if let Some(minimum) = minimum {
                            schema[if exclusive_minimum {
                                "exclusiveMinimum"
                            } else {
                                "minimum"
                            }] = json!(minimum);
                        }
                        if let Some(maximum) = maximum {
                            schema[if exclusive_maximum {
                                "exclusiveMaximum"
                            } else {
                                "maximum"
                            }] = json!(maximum);
                        }
                        let within = |value: f64| {
                            minimum.map_or(true, |m| {
                                if exclusive_minimum {
                                    value > m
                                } else {
                                    value >= m
                                }
                            }) && maximum.map_or(true, |m| {
                                if exclusive_maximum {
                                    value < m
                                } else {
                                    value <= m
                                }
                            })
                        };
                        let empty = match (minimum, maximum) {
                            (Some(low), Some(high)) if integral => {
                                let lowest = if exclusive_minimum {
                                    low.floor() + 1.0
                                } else {
                                    low.ceil()
                                };
                                let highest = if exclusive_maximum {
                                    high.ceil() - 1.0
                                } else {
                                    high.floor()
                                };
                                lowest > highest
                            }
                            (Some(low), Some(high)) => {
                                low > high
                                    || (low == high && (exclusive_minimum || exclusive_maximum))
                            }
                            _ => false,
                        };
                        let prepared = output.prepare(schema.clone());
                        if empty {
                            // No value in range: loosened and recorded, never an error.
                            assert!(!relaxations(&prepared).is_empty(), "{schema}");
                            continue;
                        }
                        assert!(relaxations(&prepared).is_empty(), "{schema}");
                        for text in &candidates {
                            let accepted = output.accepts(&prepared, text);
                            let value = plain_decimal(text);
                            let valid =
                                value.is_some_and(|v| within(v) && (!integral || v.fract() == 0.0));
                            // Negative zero and integers written with a fraction
                            // are valid spellings the grammar need not admit.
                            let canonical = !text.starts_with("-0") || value != Some(0.0);
                            let canonical = canonical && !(integral && text.contains('.'));
                            assert!(!accepted || valid, "{schema} admits {text}");
                            assert!(!(valid && canonical) || accepted, "{schema} rejects {text}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn the_opencode_bash_tool_lowers_exactly() {
    let parameters = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {"type": "number", "exclusiveMinimum": 0, "description": "Optional timeout in milliseconds"},
            "workdir": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["command", "description"],
        "additionalProperties": false
    });
    let tokenizer = tokenizer();
    for (family, source) in ASSETS {
        let bundle = bundle(source);
        let mut request = ChatRequest::new(vec![json!({"role":"user","content":"ls"})], 946684800);
        request.tools =
            vec![json!({"type":"function","function":{"name":"bash","parameters":parameters}})];
        let prepared =
            PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
                .unwrap_or_else(|error| panic!("{family}: {error}"));
        // Families writing arguments as JSON enforce the bound; the rest record it.
        for relaxation in relaxations(&prepared) {
            assert_eq!(
                relaxation.path, "#/properties/timeout",
                "{family}: {relaxation:?}"
            );
            assert_eq!(
                relaxation.keyword, "exclusiveMinimum",
                "{family}: {relaxation:?}"
            );
        }
    }
    let mut output = JsonOutput::new();
    let prepared = output.prepare(parameters);
    assert!(relaxations(&prepared).is_empty());
    for (text, expected) in [
        (r#"{"command": "ls", "description": "List"}"#, true),
        (
            r#"{"command": "ls", "description": "List", "timeout": 120000}"#,
            true,
        ),
        (
            r#"{"command": "ls", "description": "List", "timeout": 0.5}"#,
            true,
        ),
        (
            r#"{"command": "ls", "description": "List", "timeout": 0}"#,
            false,
        ),
        (
            r#"{"command": "ls", "description": "List", "timeout": -5}"#,
            false,
        ),
        (
            r#"{"command": "ls", "description": "List", "shell": "zsh"}"#,
            false,
        ),
        (r#"{"command": "ls"}"#, false),
    ] {
        assert_eq!(output.accepts(&prepared, text), expected, "{text}");
    }
}

#[test]
fn patterns_constrain_the_value_and_keep_the_json_valid() {
    let mut output = JsonOutput::new();
    for (pattern, accepted, rejected) in [
        (
            r"^[^x]*$",
            vec![r#""a\"b""#, r#""\\""#, r#""tab\there""#, r#""""#],
            vec![r#""x""#, r#""a"b""#, r#""a\qb""#],
        ),
        (r#"^a"b$"#, vec![r#""a\"b""#], vec![r#""a"b""#, r#""ab""#]),
        (
            r"\d{3}",
            vec![r#""x123y""#, r#""123""#],
            vec![r#""12""#, r#""1x2x3""#],
        ),
        (
            r"^\w+$",
            vec![r#""snake_case1""#],
            vec![r#""kebab-case""#, r#""""#],
        ),
        (
            r"^[a-z]+\\[0-9]$",
            vec![r#""ab\\1""#],
            vec![r#""ab\1""#, r#""ab1""#],
        ),
        (
            r"^.+$",
            vec![r#""any \" thing""#],
            vec![r#""line\nbreak""#, r#""""#],
        ),
        (
            r"^(?<year>\d{4})-\d{2}$",
            vec![r#""2026-09""#],
            vec![r#""26-09""#],
        ),
        (r"^é+$", vec!["\"éé\""], vec![r#""e""#]),
    ] {
        let prepared = output.prepare(json!({"type": "string", "pattern": pattern}));
        assert!(relaxations(&prepared).is_empty(), "{pattern}");
        for text in accepted {
            assert!(output.accepts(&prepared, text), "{pattern} rejects {text}");
        }
        for text in rejected {
            assert!(!output.accepts(&prepared, text), "{pattern} admits {text}");
        }
    }
    // A pattern no grammar expresses admits any string, and is recorded.
    let prepared = output.prepare(json!({"type": "string", "pattern": r"^(?!x).*$"}));
    assert_eq!(
        relaxations(&prepared),
        [output_relaxation(
            "#",
            "pattern",
            RelaxationReason::Unenforced
        )]
    );
    assert!(output.accepts(&prepared, r#""xyz""#));
}

/// A bounded decimal is written with at most 15 digits, so its value as a
/// double lies on the same side of every bound as its decimal value.
#[test]
fn bounded_numbers_stay_within_bounds_as_doubles() {
    let mut output = JsonOutput::new();
    let prepared = output.prepare(json!({"type": "number", "exclusiveMinimum": 0.25}));
    assert!(output.accepts(&prepared, "0.25000000000001"));
    // 0.25000000000000001 reads back as the double 0.25.
    assert!(!output.accepts(&prepared, "0.25000000000000001"));
    let prepared = output.prepare(json!({"type": "integer", "maximum": 1e18}));
    assert!(output.accepts(&prepared, "999999999999999999"));
}

#[test]
fn strings_are_always_valid_json() {
    let mut output = JsonOutput::new();
    for (schema, accepted, rejected) in [
        (
            json!({"type": "string"}),
            vec![r#""é""#, r#""😀""#, r#""a\nb""#],
            vec![r#""\ud800""#, r#""\ude00""#, "\"a\nb\"", "\"a\tb\""],
        ),
        // A surrogate pair is one character.
        (
            json!({"type": "string", "minLength": 2}),
            vec![r#""😀x""#],
            vec![r#""😀""#],
        ),
    ] {
        let prepared = output.prepare(schema.clone());
        for text in accepted {
            assert!(output.accepts(&prepared, text), "{schema} rejects {text}");
        }
        for text in rejected {
            assert!(!output.accepts(&prepared, text), "{schema} admits {text:?}");
        }
    }
}

#[test]
fn compositions_lower_exactly() {
    let mut output = JsonOutput::new();
    let cases = [
        // allOf of a referenced base and an extension merges both.
        (
            json!({
                "$defs": {"base": {"type": "object", "properties": {"id": {"type": "integer", "minimum": 1}}, "required": ["id"]}},
                "allOf": [{"$ref": "#/$defs/base"}, {"properties": {"name": {"type": "string"}}, "required": ["name"]}]
            }),
            vec![r#"{"id": 1, "name": "a"}"#],
            vec![
                r#"{"id": 0, "name": "a"}"#,
                r#"{"id": 1}"#,
                r#"{"name": "a"}"#,
            ],
        ),
        // Siblings of anyOf apply to every alternative.
        (
            json!({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                   "anyOf": [{"required": ["a"]}, {"required": ["b"]}]}),
            vec![r#"{"a": "x"}"#, r#"{"b": "y"}"#, r#"{"a": "x", "b": "y"}"#],
            vec![r#"{}"#, r#"{"c": 1}"#],
        ),
        // A type array applies each type's keywords to its own values.
        (
            json!({"type": ["integer", "null"], "minimum": 3}),
            vec!["3", "null"],
            vec!["2", "\"3\""],
        ),
        // An enum takes only the values of its schema's type.
        (
            json!({"type": "string", "enum": ["a", 1, "b"]}),
            vec!["\"a\"", "\"b\""],
            vec!["1"],
        ),
        // Disjoint oneOf alternatives are exact.
        (
            json!({"oneOf": [
                {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}}, "required": ["kind"]},
                {"type": "object", "properties": {"kind": {"const": "b"}, "y": {"type": "string"}}, "required": ["kind"]}
            ]}),
            vec![r#"{"kind": "a", "x": 1}"#, r#"{"kind": "b", "y": "z"}"#],
            vec![r#"{"kind": "a", "y": "z"}"#, r#"{"kind": "c"}"#],
        ),
        // A false subschema forbids its property.
        (
            json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": false}}),
            vec![r#"{"a": 1}"#, r#"{}"#],
            vec![r#"{"b": 1}"#],
        ),
        // An object listing properties stays closed when none can appear.
        (
            json!({"type": "object", "properties": {"b": false}}),
            vec![r#"{}"#],
            vec![r#"{"b": 1}"#, r#"{"c": 1}"#],
        ),
        // An object listing no properties is open.
        (
            json!({"required": ["a"]}),
            vec![r#"{"a": 1, "z": 2}"#],
            vec![r#"{"z": 2}"#],
        ),
        // Draft 4 spells exclusive bounds as booleans.
        (
            json!({"$schema": "http://json-schema.org/draft-04/schema#", "type": "integer", "minimum": 0, "exclusiveMinimum": true}),
            vec!["1"],
            vec!["0"],
        ),
        // Tuples.
        (
            json!({"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": false}),
            vec![r#"[1, "a"]"#],
            vec![r#"["a", 1]"#, r#"[1, "a", 2]"#],
        ),
        // Recursive references.
        (
            json!({"$defs": {"node": {"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/node"}}}}},
                   "$ref": "#/$defs/node"}),
            vec![r#"{"children": [{"children": []}, {}]}"#],
            vec![r#"{"children": [1]}"#],
        ),
        // Integer multiples within bounds are listed.
        (
            json!({"type": "integer", "minimum": 0, "maximum": 20, "multipleOf": 5}),
            vec!["0", "15"],
            vec!["3", "25"],
        ),
    ];
    for (schema, accepted, rejected) in cases {
        let prepared = output.prepare(schema.clone());
        assert!(
            relaxations(&prepared).is_empty(),
            "{schema}: {:?}",
            relaxations(&prepared)
        );
        for text in accepted {
            assert!(output.accepts(&prepared, text), "{schema} rejects {text}");
        }
        for text in rejected {
            assert!(!output.accepts(&prepared, text), "{schema} admits {text}");
        }
    }
}

#[test]
fn unenforceable_keywords_are_loosened_and_recorded() {
    let mut output = JsonOutput::new();
    let unenforced = RelaxationReason::Unenforced;
    for (schema, expected, admitted) in [
        (
            json!({"type": "array", "items": {"type": "integer"}, "uniqueItems": true}),
            vec![output_relaxation("#", "uniqueItems", unenforced)],
            "[1, 1]",
        ),
        (
            json!({"type": "string", "not": {"const": "x"}}),
            vec![output_relaxation("#", "not", unenforced)],
            "\"x\"",
        ),
        (
            json!({"oneOf": [{"type": "integer"}, {"type": "number"}]}),
            vec![output_relaxation("#", "oneOf", unenforced)],
            "1",
        ),
        // 1.0 is an integer, so these alternatives overlap.
        (
            json!({"oneOf": [{"const": 1.0}, {"type": "integer"}]}),
            vec![output_relaxation("#", "oneOf", unenforced)],
            "1.0",
        ),
        (
            json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"],
                   "if": {"properties": {"a": {"const": 1}}}, "then": {"required": ["b"]}}),
            vec![output_relaxation("#", "if", unenforced)],
            r#"{"a": 1}"#,
        ),
        (
            json!({"type": "integer", "minimum": 5, "maximum": 1}),
            vec![output_relaxation(
                "#",
                "minimum",
                RelaxationReason::Unsatisfiable,
            )],
            "7",
        ),
        (
            json!({"type": "object", "properties": {"x": {"$ref": "#foo"}}, "$defs": {"a": {"$anchor": "foo", "type": "string"}}}),
            vec![output_relaxation("#/properties/x", "$ref", unenforced)],
            r#"{"x": 1}"#,
        ),
    ] {
        let prepared = output.prepare(schema.clone());
        assert_eq!(relaxations(&prepared), expected, "{schema}");
        assert!(
            output.accepts(&prepared, admitted),
            "{schema} rejects {admitted}"
        );
    }
}

/// Valid schemas of every kind, as argument schemas.
fn corpus() -> Vec<Value> {
    let values = [
        json!({}),
        json!({"type": "string", "format": "email"}),
        json!({"type": "string", "format": "date-time"}),
        json!({"type": "string", "minLength": 2, "maxLength": 100000}),
        json!({"type": "string", "pattern": "^(?=a)"}),
        json!({"type": "string", "pattern": "^[\"\\\\]+$", "maxLength": 3}),
        json!({"type": ["integer", "null"], "minimum": 0, "exclusiveMaximum": 1e20}),
        json!({"type": "number", "multipleOf": 0.5, "minimum": -1e-7}),
        json!({"enum": ["a", 1, null, {"b": [2]}]}),
        json!({"const": {"deep": [1, "x<|\"|>y"]}}),
        json!({"type": "array", "minItems": 2, "prefixItems": [{"type": "boolean"}], "items": {"type": "integer"}}),
        json!({"type": "array", "contains": {"type": "string"}, "maxItems": 100000}),
        json!({"allOf": [{"type": "string"}, {"type": "integer"}]}),
        json!({"anyOf": [false, {"type": "null"}]}),
        json!(false),
        json!({"type": "object", "patternProperties": {"^x-": {"type": "string"}}, "propertyNames": {"maxLength": 4}}),
        json!({"type": "object", "dependentRequired": {"a": ["b"]}, "minProperties": 2, "unevaluatedProperties": false}),
        json!({"properties": {"a/b~c": {"type": "integer"}}, "required": ["a/b~c", "undeclared"], "additionalProperties": {"type": "string"}}),
        json!({"$defs": {"a/b": {"type": "integer"}}, "$ref": "#/$defs/a~1b"}),
        json!({"$dynamicRef": "#meta", "not": {}, "if": true, "then": false}),
    ];
    let mut schemas = Vec::new();
    for value in values {
        schemas.push(json!({"type": "object", "properties": {"v": value.clone(), "w": {"type": "string"}}, "required": ["v"]}));
        if value.is_object() && value.get("type") == Some(&json!("object")) {
            schemas.push(value);
        }
    }
    schemas
}

#[test]
fn every_valid_schema_prepares_in_every_family() {
    let tokenizer = tokenizer();
    let mut failures = Vec::new();
    for (family, source) in ASSETS {
        let bundle = bundle(source);
        for parameters in corpus() {
            for choice in [ToolChoice::Auto, ToolChoice::Required] {
                let mut request =
                    ChatRequest::new(vec![json!({"role":"user","content":"Go"})], 946684800);
                request.tools = vec![
                    json!({"type":"function","function":{"name":"f","parameters":parameters}}),
                ];
                request.tool_choice = choice;
                if let Err(error) = PreparedChat::prepare(
                    &bundle,
                    &tokenizer,
                    &request,
                    &TemplateSelection::default(),
                ) {
                    failures.push(format!("{family} {parameters}: {error}"));
                }
            }
        }
    }
    let output = JsonOutput::new();
    for schema in corpus() {
        let mut request = ChatRequest::new(vec![json!({"role":"user","content":"Go"})], 946684800);
        request.json_schema = Some(schema.clone());
        if let Err(error) = PreparedChat::prepare(
            &output.bundle,
            &output.tokenizer,
            &request,
            &TemplateSelection::default(),
        ) {
            failures.push(format!("output {schema}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn violations_under_exact_grammars_are_defects() {
    use magnitude_chat::{
        conformance::{Conformance, Enforcement, OutputSchemas},
        request::{ChatInput, Conversation, Entry, OutputFormat, Tools, UserPart},
        schema::JsonSchema,
        ReasoningIntent,
    };
    let tokenizer = tokenizer();
    let bundle = bundle(ASSETS[0].1);
    let schemas = |schema: Value| {
        let Value::Object(schema) = schema else {
            panic!("schema fixtures are objects")
        };
        let input = ChatInput {
            conversation: Conversation::new(
                None,
                vec![Entry::User(vec![UserPart::Text("Hi".into())])],
            )
            .unwrap(),
            tools: Tools::none(),
            reasoning: ReasoningIntent::ModelDefault,
            output: OutputFormat::JsonSchema {
                name: "answer".into(),
                schema: JsonSchema::new(schema).unwrap(),
            },
            template_arguments: Default::default(),
        };
        let rendered = input.render(946684800, None).unwrap();
        let prepared = PreparedChat::prepare(
            &bundle,
            &tokenizer,
            &rendered.request,
            &TemplateSelection::default(),
        )
        .unwrap();
        OutputSchemas::new(&input, &prepared)
    };
    let exact = schemas(json!({"type": "integer", "minimum": 1}));
    assert_eq!(exact.output(" 3 "), Some(Conformance::Conforms));
    assert!(matches!(
        exact.output("0"),
        Some(Conformance::GrammarDefect { .. })
    ));
    assert!(matches!(
        exact.output("three"),
        Some(Conformance::GrammarDefect { .. })
    ));
    let loosened = schemas(json!({"type": "array", "uniqueItems": true}));
    assert!(matches!(
        loosened.output("[1, 1]"),
        Some(Conformance::Unenforced {
            enforcement: Enforcement::Loosened(_),
            ..
        })
    ));
}
