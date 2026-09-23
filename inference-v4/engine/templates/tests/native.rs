use magnitude_templates::{build_info, Event, Request, Template, TerminalCause, ToolChoice};
use serde_json::json;

const QWEN: &str = include_str!("assets/Qwen-Qwen3-0.6B.jinja");
const QWEN35: &str = include_str!("assets/Qwen3.5-4B.jinja");

fn request() -> Request {
    let mut request = Request::new(vec![json!({"role":"user","content":"Hello"})], 946684800);
    request.tools = vec![json!({"type":"function","function":{
        "name":"search","description":"Search","parameters":{
            "type":"object","properties":{"query":{"type":"string"}},
            "required":["query"],"additionalProperties":false
        }
    }})];
    request
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Transcript {
    content: String,
    reasoning: String,
    calls: Vec<(String, String, String, bool)>,
    finish: Option<TerminalCause>,
}
impl Transcript {
    fn accept(&mut self, events: Vec<Event>) {
        for event in events {
            match event {
                Event::Content { text } => self.content.push_str(&text),
                Event::Reasoning { text } => self.reasoning.push_str(&text),
                Event::ToolStart { index, name, id } => {
                    assert_eq!(index as usize, self.calls.len());
                    self.calls.push((name, id, String::new(), false));
                }
                Event::ToolArguments { index, text } => {
                    let call = &mut self.calls[index as usize];
                    assert!(!call.3);
                    call.2.push_str(&text);
                }
                Event::ToolComplete { index } => {
                    let call = &mut self.calls[index as usize];
                    assert!(!call.3);
                    call.3 = true;
                }
                Event::Finish { cause } => assert!(self.finish.replace(cause).is_none()),
            }
        }
    }
}

#[test]
fn rendering_preserves_absent_controls_special_tokens_and_explicit_time() {
    let build = build_info().unwrap();
    assert_eq!(build.abi, 1);
    assert_eq!(build.upstream, "930e2fa5995789efbf249a8bf61325bb626e417b");
    assert_eq!(build.build.len(), 64);
    assert!(build.build.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let template = Template::new(
        "{{ bos_token }}{{ enable_thinking is defined }} {{ strftime_now('%Y-%m-%d') }}",
        &[("bos_token".into(), "<s>".into())].into(),
    )
    .unwrap();
    let rendered = template.render(&Default::default(), 0).unwrap();
    assert_eq!(rendered, "<s>False 1970-01-01");
    assert!(template
        .render(json!({"bos_token":"other"}).as_object().unwrap(), 0)
        .is_err());
    let a = Template::new("hello", &Default::default()).unwrap();
    let b = Template::new("hello", &Default::default()).unwrap();
    assert_eq!(a.identity(), b.identity());
    assert!(!a.capabilities().unwrap().is_empty());
}

#[test]
fn prepared_and_stream_owners_outlive_their_sources() {
    let plan = Template::new(QWEN, &Default::default())
        .unwrap()
        .prepare(&request())
        .unwrap();
    assert_eq!(plan.description().grammar_dialect, "gbnf");
    let mut stream = plan.stream(4096).unwrap();
    drop(plan);
    let retained = stream.feed(b"first ").unwrap();
    stream.feed(b"second").unwrap();
    let mut result = Transcript::default();
    result.accept(retained);
    assert_eq!(result.content, "first ");
    stream.finish(TerminalCause::Natural).unwrap();
    assert!(stream.feed(b"late").is_err());
}

#[test]
fn real_templates_preserve_tools_reasoning_and_unicode_at_every_byte_split() {
    for (source,output,reasoning) in [
        (QWEN,"<think>réflexion 世界</think><tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"héllo 世界\"}}\n</tool_call>","réflexion 世界"),
        (QWEN35,"consider</think>\n\n<tool_call>\n<function=search>\n<parameter=query>\nhéllo 世界\n</parameter>\n</function>\n</tool_call>","consider"),
    ] {
        let plan = Template::new(source,&Default::default()).unwrap().prepare(&request()).unwrap();
        let bytes = output.as_bytes();
        let parse = |chunks: Vec<&[u8]>| {
            let mut stream = plan.stream(4096).unwrap();
            let mut result = Transcript::default();
            for chunk in chunks { result.accept(stream.feed(chunk).unwrap()); }
            result.accept(stream.finish(TerminalCause::Natural).unwrap());
            result
        };
        let expected = parse(vec![bytes]);
        assert_eq!(expected.reasoning,reasoning);
        assert_eq!(expected.calls.len(),1);
        assert_eq!(expected.calls[0].0,"search");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&expected.calls[0].2).unwrap(),json!({"query":"héllo 世界"}));
        assert!(expected.calls[0].3);
        for split in 0..=bytes.len() {
            assert_eq!(parse(vec![&bytes[..split],&bytes[split..]]),expected,"split {split}");
        }
        assert_eq!(parse(bytes.chunks(1).collect()),expected);
    }
}

#[test]
fn truncation_never_completes_partial_tool_calls() {
    let plan = Template::new(QWEN, &Default::default())
        .unwrap()
        .prepare(&request())
        .unwrap();
    let partial = b"<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"hel";
    for cause in [
        TerminalCause::Length,
        TerminalCause::UserStop,
        TerminalCause::Cancelled,
        TerminalCause::Failed,
    ] {
        let mut stream = plan.stream(4096).unwrap();
        let mut result = Transcript::default();
        result.accept(stream.feed(partial).unwrap());
        assert_eq!(result.calls.len(), 1);
        assert!(!result.calls[0].2.is_empty());
        result.accept(stream.finish(cause).unwrap());
        assert!(!result.calls[0].3);
        assert_eq!(result.finish, Some(cause));
    }
    let mut stream = plan.stream(4096).unwrap();
    stream.feed(partial).unwrap();
    assert!(stream.finish(TerminalCause::Natural).is_err());
}

#[test]
fn failures_are_local_and_output_limits_include_pending_utf8() {
    let plan = Template::new(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        &Default::default(),
    )
    .unwrap()
    .prepare(&Request::new(
        vec![json!({"role":"user","content":"hi"})],
        0,
    ))
    .unwrap();
    assert!(plan.stream(0).is_err());
    let mut bounded = plan.stream(2).unwrap();
    bounded.feed(&[0xe4]).unwrap();
    assert!(bounded.feed(&[0xb8, 0x96]).is_err());
    assert!(bounded.finish(TerminalCause::Length).is_err());
    let mut invalid = plan.stream(100).unwrap();
    assert!(invalid.feed(&[0xff]).is_err());
    let mut fresh = plan.stream(100).unwrap();
    let mut result = Transcript::default();
    result.accept(fresh.feed("世界".as_bytes()).unwrap());
    result.accept(fresh.finish(TerminalCause::Natural).unwrap());
    assert_eq!(result.content, "世界");
}

#[test]
fn required_tools_and_unsupported_schemas_fail_explicitly() {
    let template = Template::new(QWEN, &Default::default()).unwrap();
    let mut request = request();
    request.tool_choice = ToolChoice::Required;
    let plan = template.prepare(&request).unwrap();
    assert!(!plan.description().grammar.is_empty());
    let mut stream = plan.stream(4096).unwrap();
    // A native parser may reject when feeding or when finalizing.
    assert!(stream
        .feed(b"ordinary prose")
        .and_then(|_| stream.finish(TerminalCause::Natural))
        .is_err());
    request.tools[0]["function"]["parameters"]["properties"]["query"]["minLength"] = json!(3);
    assert!(template.prepare(&request).is_err());
}
