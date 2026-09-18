use seismic_engine::chat::{
    ChatRequest, ChatStream, Event, StopText, TemplateBundle, TemplateSelection, TemplateVariant,
    TerminalCause, ToolChoice,
};
use serde_json::json;

fn variant(name: &str, source: &str) -> TemplateVariant {
    TemplateVariant {
        name: name.into(),
        source: source.into(),
        provenance: "fixture".into(),
    }
}
fn bundle() -> TemplateBundle {
    TemplateBundle::new(
        vec![
            variant(
                "default",
                "{% for message in messages %}{{ message.content }}{% endfor %}",
            ),
            variant("tool_use", "tool variant"),
        ],
        "default".into(),
        Default::default(),
    )
    .unwrap()
}

#[test]
fn template_precedence_uses_effective_tools_and_explicit_default() {
    let bundle = bundle();
    assert_eq!(
        bundle
            .select(false, &TemplateSelection::default())
            .unwrap()
            .name,
        "default"
    );
    assert_eq!(
        bundle
            .select(true, &TemplateSelection::default())
            .unwrap()
            .name,
        "tool_use"
    );
    assert_eq!(
        bundle
            .select(
                true,
                &TemplateSelection {
                    variant: Some("default"),
                    source_override: None
                }
            )
            .unwrap()
            .name,
        "default"
    );
    let source = variant("override", "operator source");
    assert_eq!(
        bundle
            .select(
                true,
                &TemplateSelection {
                    variant: None,
                    source_override: Some(&source)
                }
            )
            .unwrap()
            .name,
        "override"
    );
    assert!(bundle
        .select(
            true,
            &TemplateSelection {
                variant: Some("default"),
                source_override: Some(&source)
            }
        )
        .is_err());
    assert!(TemplateBundle::new(
        vec![variant("other", "source")],
        "default".into(),
        Default::default()
    )
    .is_err());
    let mut request = ChatRequest::new(vec![json!({"role":"user","content":"hello"})], 0);
    request.tools =
        vec![json!({"type":"function","function":{"name":"test","parameters":{"type":"object"}}})];
    request.tool_choice = ToolChoice::None;
    let plan = bundle
        .prepare(&request, &TemplateSelection::default())
        .unwrap();
    assert_eq!(plan.description().prompt, "hello");
    assert_eq!(request.tools.len(), 1);
}

#[test]
fn named_tool_selection_and_history_normalization_preserve_caller_data() {
    let source = include_str!(
        "../../../inference-v3/native/templates/upstream/models/templates/Qwen-Qwen3-0.6B.jinja"
    );
    let bundle = TemplateBundle::new(
        vec![variant("default", source)],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let mut request = ChatRequest::new(
        vec![
            json!({"role":"user","content":"hello"}),
            json!({"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"query\":\"hello\"}"}}]}),
            json!({"role":"tool","name":"search","tool_call_id":"call_1","content":"found"}),
        ],
        0,
    );
    request.tools = ["search","other"].into_iter().map(|name| json!({"type":"function","function":{"name":name,"parameters":{"type":"object","properties":{"query":{"type":"string"}}}}})).collect();
    request.tool_choice = ToolChoice::Named("search".into());
    let original = serde_json::to_value(&request).unwrap();
    let plan = bundle
        .prepare(&request, &TemplateSelection::default())
        .unwrap();
    assert!(!plan.description().grammar.is_empty());
    assert!(!plan.description().prompt.contains("\"name\": \"other\""));
    assert_eq!(serde_json::to_value(&request).unwrap(), original);
    request.tool_choice = ToolChoice::Named("missing".into());
    assert!(bundle
        .prepare(&request, &TemplateSelection::default())
        .is_err());
}

#[test]
fn overlapping_unicode_stop_patterns_are_chunk_invariant() {
    let input = "prefix 世界! rest";
    let stops = vec!["世界!".into(), "界".into(), "!".into()];
    for split in input
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(input.len()))
    {
        let mut filter = StopText::new(stops.clone()).unwrap();
        let mut output = filter.feed(&input[..split]).unwrap();
        output.push_str(&filter.feed(&input[split..]).unwrap());
        output.push_str(&filter.finish().unwrap());
        assert_eq!(output, "prefix 世");
        assert_eq!(filter.matched(), Some("界"));
    }
    let mut filter = StopText::new(vec!["END".into()]).unwrap();
    assert_eq!(filter.feed("text EN").unwrap(), "text ");
    assert_eq!(filter.finish().unwrap(), "EN");
}

#[test]
fn stop_filter_finishes_parser_once_and_flushes_unmatched_suffixes() {
    let plan = bundle()
        .prepare(
            &ChatRequest::new(vec![json!({"role":"user","content":"hi"})], 0),
            &TemplateSelection::default(),
        )
        .unwrap();
    let mut stream = ChatStream::new(&plan, vec!["STOP".into()], 100).unwrap();
    let mut events = stream.feed("hello ST").unwrap();
    events.extend(stream.feed("OP hidden").unwrap());
    assert!(stream.stopped());
    assert!(stream.finish(TerminalCause::Natural).is_err());
    assert_eq!(
        events,
        vec![
            Event::Content {
                text: "hello ".into()
            },
            Event::Finish {
                cause: TerminalCause::UserStop
            }
        ]
    );
    let mut stream = ChatStream::new(&plan, vec!["STOP".into()], 100).unwrap();
    let mut events = stream.feed("hello ST").unwrap();
    events.extend(stream.finish(TerminalCause::Length).unwrap());
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            Event::Content { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hello ST");
    assert_eq!(
        events.last(),
        Some(&Event::Finish {
            cause: TerminalCause::Length
        })
    );
}

#[test]
fn template_stops_finish_naturally_and_explicit_user_stops_take_precedence() {
    let source=include_str!("../../../inference-v3/native/templates/upstream/models/templates/poolside-Laguna-XS.2.jinja");
    let bundle = TemplateBundle::new(
        vec![variant("default", source)],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let request = ChatRequest::new(vec![json!({"role":"user","content":"hello"})], 0);
    let prepared = bundle
        .prepare(&request, &TemplateSelection::default())
        .unwrap();
    assert!(prepared
        .description()
        .additional_stops
        .iter()
        .any(|s| s == "</assistant>"));
    for (stops, cause) in [
        (vec![], TerminalCause::Natural),
        (vec!["</assistant>".into()], TerminalCause::UserStop),
    ] {
        let mut stream = ChatStream::new(&prepared, stops, 4096).unwrap();
        let mut events = stream.feed("hello</assis").unwrap();
        events.extend(stream.feed("tant>ignored").unwrap());
        assert!(stream.stopped());
        assert_eq!(events.last(), Some(&Event::Finish { cause }));
    }
}
