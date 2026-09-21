use magnitude_engine::{
    chat::{
        wire::{ModelLimits, Request},
        TemplateBundle, TemplateSelection, TemplateVariant,
    },
    generation::Sampling,
    inputs::{BpeConfig, ByteBpeTokenizer, PieceKind, TokenId},
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn body() -> Value {
    json!({"model":"fixture","messages":[{"role":"user","content":"hello"}]})
}
fn parse(value: &Value) -> Result<Request, String> {
    Request::parse(&serde_json::to_vec(value).unwrap(), 1 << 20).map_err(Into::into)
}
fn preparation() -> (TemplateBundle, ByteBpeTokenizer) {
    let included: BTreeSet<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let missing: Vec<_> = (0..=255).filter(|b| !included.contains(b)).collect();
    let mut pieces: Vec<_> = (0..=255)
        .map(|b| {
            char::from_u32(if included.contains(&b) {
                u32::from(b)
            } else {
                256 + missing.iter().position(|x| *x == b).unwrap() as u32
            })
            .unwrap()
            .to_string()
        })
        .collect();
    pieces.push("<eos>".into());
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.push(PieceKind::Control);
    let tokenizer = ByteBpeTokenizer::new(BpeConfig {
        artifact_identity: "wire".into(),
        pieces,
        kinds,
        merges: vec![],
        pattern: r".+|\s".into(),
        normalize_nfc: false,
        stop_tokens: BTreeSet::from([TokenId(256)]),
    })
    .unwrap();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    (bundle, tokenizer)
}
#[test]
fn unsupported_and_malformed_wire_policies_fail_before_preparation() {
    for (field, value) in [
        ("temperature", json!(0.5)),
        ("temperature", json!("1")),
        ("top_p", json!(0.9)),
        ("top_k", json!(1)),
        ("min_p", json!(0.1)),
        ("repetition_penalty", json!(1.1)),
        ("presence_penalty", json!(-1)),
        ("frequency_penalty", json!(1)),
        ("seed", json!(-1)),
        ("seed", json!(1.5)),
        ("n", json!(2)),
        ("n", json!(true)),
        ("stream", json!(1)),
        ("max_tokens", json!(-1)),
        ("max_tokens", json!(2147483648u64)),
        ("reasoning_effort", json!("unknown")),
        ("unknown", json!(null)),
        ("stop", json!("")),
        ("stop", json!(["a", "b", "c", "d", "e"])),
        ("messages", json!([])),
        ("model", json!("")),
        ("stream_options", json!({"unknown":true})),
        ("response_format", json!({"type":"text","unknown":1})),
        (
            "response_format",
            json!({"type":"json_schema","json_schema":{"name":"x","schema":{},"strict":null}}),
        ),
    ] {
        let mut request = body();
        request[field] = value;
        assert!(parse(&request).is_err(), "accepted invalid {field}");
    }
    for message in [
        json!({"role":"invalid","content":"x"}),
        json!({"role":"user","content":42}),
        json!({"role":"user","content":[{"type":"text","text":"x","extra":1}]}),
        json!({"role":"assistant","tool_calls":[{"id":"c","function":{"name":"f","arguments":[]}}]}),
        json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA==","detail":"high"}}]}),
    ] {
        let mut request = body();
        request["messages"] = json!([message]);
        assert!(parse(&request).is_err());
    }
    assert!(Request::parse(&serde_json::to_vec(&body()).unwrap(), 4).is_err());
}
#[test]
fn aliases_defaults_unicode_stops_and_context_limits_follow_v3() {
    let (bundle, tokenizer) = preparation();
    let limits = ModelLimits {
        model: "fixture",
        context_tokens: 8,
        vocabulary: 257,
        output_capacity: 4,
        forced_quantum: 2,
    };
    let default = parse(&body()).unwrap();
    assert_eq!(default.output_limit(), 512);
    assert!(!default.stream());
    assert!(!default.include_usage());
    let prepared = default
        .prepare(
            &bundle,
            &tokenizer,
            &TemplateSelection::default(),
            0,
            &limits,
        )
        .unwrap();
    assert_eq!(prepared.chat.prompt_tokens(), 5);
    assert_eq!(prepared.options.max_tokens, 4);
    assert_eq!(prepared.options.sampling, Sampling::Categorical);
    let mut raw = body();
    raw["max_tokens"] = json!(0);
    raw["max_completion_tokens"] = json!(0);
    raw["temperature"] = json!(0);
    raw["seed"] = json!(u64::MAX);
    raw["stop"] = json!("世".repeat(1024));
    raw["stream"] = json!(true);
    raw["stream_options"] = json!({"include_usage":true});
    let request = parse(&raw).unwrap();
    let prepared = request
        .prepare(
            &bundle,
            &tokenizer,
            &TemplateSelection::default(),
            0,
            &limits,
        )
        .unwrap();
    assert_eq!(prepared.options.max_tokens, 0);
    assert_eq!(prepared.options.seed, u64::MAX);
    assert_eq!(prepared.options.sampling, Sampling::Greedy);
    assert!(request.stream() && request.include_usage());
    assert_eq!(request.stops()[0].chars().count(), 1024);
    raw["stop"] = json!("世".repeat(1025));
    assert!(parse(&raw).is_err());
    raw["stop"] = Value::Null;
    raw["max_completion_tokens"] = json!(1);
    assert!(parse(&raw).is_err());
    let wrong = ModelLimits {
        model: "other",
        ..limits
    };
    assert!(default
        .prepare(
            &bundle,
            &tokenizer,
            &TemplateSelection::default(),
            0,
            &wrong
        )
        .is_err());
    let short = ModelLimits {
        model: "fixture",
        context_tokens: 4,
        ..wrong
    };
    assert!(default
        .prepare(
            &bundle,
            &tokenizer,
            &TemplateSelection::default(),
            0,
            &short
        )
        .is_err());
}
#[test]
fn tool_schema_policy_and_unconnected_media_are_explicit() {
    let mut raw = body();
    raw["tools"] = json!([{"function":{"name":"lookup"}}]);
    raw["tool_choice"] = json!({"function":{"name":"lookup"}});
    assert!(parse(&raw).is_ok());
    raw["response_format"] = json!({"type":"json_object"});
    assert!(parse(&raw).is_err());
    raw["tool_choice"] = json!("none");
    assert!(parse(&raw).is_ok());
    raw["tools"][0]["function"]["strict"] = Value::Null;
    assert!(parse(&raw).is_err());
    raw["tools"][0]["function"]
        .as_object_mut()
        .unwrap()
        .remove("strict");
    raw["tools"][0]["function"]["parameters"] = Value::Null;
    assert!(parse(&raw).is_err());
    raw["tools"] = json!([]);
    raw["tool_choice"] = json!("required");
    assert!(parse(&raw).is_err());
    let (bundle, tokenizer) = preparation();
    let mut schema = body();
    schema["response_format"] = json!({"type":"json_schema","json_schema":{
        "name":"answer","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}
    }});
    let limits = ModelLimits {
        model: "fixture",
        context_tokens: 32,
        vocabulary: 257,
        output_capacity: 4,
        forced_quantum: 0,
    };
    let structured = parse(&schema)
        .unwrap()
        .prepare(
            &bundle,
            &tokenizer,
            &TemplateSelection::default(),
            0,
            &limits,
        )
        .unwrap();
    assert!(structured.chat.input().constraint.is_some());
    let mut image = body();
    image["messages"][0]["content"] =
        json!([{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]);
    let request = parse(&image).unwrap();
    let limits = ModelLimits {
        model: "fixture",
        context_tokens: 32,
        vocabulary: 257,
        output_capacity: 4,
        forced_quantum: 0,
    };
    assert!(request
        .prepare(
            &bundle,
            &tokenizer,
            &TemplateSelection::default(),
            0,
            &limits
        )
        .err()
        .unwrap()
        .contains("media preparation"));
}
