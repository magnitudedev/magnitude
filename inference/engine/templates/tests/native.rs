use magnitude_templates::{
    build_info, Event, Relaxation, RelaxationReason, RelaxationSubject, Request, SpecialTokens,
    Template, TerminalCause, ToolChoice,
};
use serde_json::json;

const QWEN: &str = include_str!("assets/Qwen-Qwen3-0.6B.jinja");
const QWEN35: &str = include_str!("assets/Qwen3.5-4B.jinja");
const QWEN36: &str = include_str!("assets/Qwen3.6-35B-A3B.jinja");
const DEEPSEEK: &str = include_str!("assets/DeepSeek-V3.2.jinja");
const GEMMA4: &str = include_str!("assets/gemma-4-12B-it.jinja");
const LLAMA32: &str = include_str!("assets/Llama-3.2-3B-Instruct.jinja");

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
    calls: Vec<(String, Option<String>, String, bool)>,
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
        // Neither format writes call IDs; the stream invents none (the host
        // assigns identities unique across responses).
        assert_eq!(expected.calls[0].1,None);
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
fn required_tools_fail_explicitly_and_every_schema_prepares() {
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
    // A length bound is enforced by the JSON argument grammar, and an array
    // keyword on a string constrains nothing.
    let query = &mut request.tools[0]["function"]["parameters"]["properties"]["query"];
    query["minLength"] = json!(3);
    query["uniqueItems"] = json!(true);
    let plan = template.prepare(&request).unwrap();
    assert!(plan.description().relaxations.is_empty());
    // A keyword no grammar enforces is loosened and recorded.
    request.tools[0]["function"]["parameters"]["properties"]["query"]["not"] =
        json!({"const": "x"});
    let plan = template.prepare(&request).unwrap();
    assert_eq!(
        plan.description().relaxations,
        vec![Relaxation {
            subject: RelaxationSubject::Tool {
                name: "search".into()
            },
            path: "#/properties/query".into(),
            keyword: "not".into(),
            reason: RelaxationReason::Unenforced,
        }]
    );
}

/// Qwen3.5's prompt opens reasoning and the model ends without closing it. The
/// stream has already published that text as reasoning; ending the output
/// closes the reasoning instead of reading it again as content.
#[test]
fn natural_end_inside_published_reasoning_keeps_it() {
    let output = "He is the capital city of Australia.";
    for request in [
        Request::new(
            vec![json!({"role":"user","content":"What is the capital city of Australia? Reply with one word."})],
            0,
        ),
        request(),
    ] {
        let plan = Template::new(QWEN35, &Default::default())
            .unwrap()
            .prepare(&request)
            .unwrap();
        assert!(plan.description().prompt.ends_with("<think>\n"));
        let bytes = output.as_bytes();
        for split in 0..=bytes.len() {
            let mut stream = plan.stream(4096).unwrap();
            let mut result = Transcript::default();
            result.accept(stream.feed(&bytes[..split]).unwrap());
            result.accept(stream.feed(&bytes[split..]).unwrap());
            result.accept(stream.finish(TerminalCause::Natural).unwrap());
            assert_eq!(
                result,
                Transcript {
                    reasoning: output.into(),
                    finish: Some(TerminalCause::Natural),
                    ..Default::default()
                },
                "split {split}"
            );
        }
    }
    // A call the model leaves open is still not a natural completion.
    let plan = Template::new(QWEN35, &Default::default())
        .unwrap()
        .prepare(&request())
        .unwrap();
    let mut stream = plan.stream(4096).unwrap();
    stream
        .feed(b"consider</think>\n\n<tool_call>\n<function=search>\n<parameter=query>\nhel")
        .unwrap();
    assert!(stream.finish(TerminalCause::Natural).is_err());
}

/// The prompts of two consecutive tasks: the first mid-task, after a tool
/// result, and the second at the next user message, after the first task's
/// final reply. `arguments` are the caller's template arguments.
fn two_task_prompts(source: &str, arguments: serde_json::Value) -> (String, String) {
    let template = Template::new(source, &Default::default()).unwrap();
    let first_task = vec![
        json!({"role":"user","content":"Find the release notes."}),
        json!({"role":"assistant","content":"","reasoning_content":"I should search first.",
            "tool_calls":[{"id":"call_0","type":"function",
                "function":{"name":"search","arguments":{"query":"release notes"}}}]}),
        json!({"role":"tool","tool_call_id":"call_0","content":"Release 1.2 fixes the cache."}),
    ];
    let mut second_task = first_task.clone();
    second_task.extend([
        json!({"role":"assistant","content":"Release 1.2 fixes the cache.",
            "reasoning_content":"The result answers the question."}),
        json!({"role":"user","content":"Now summarize it."}),
    ]);
    let prompt = |messages: Vec<serde_json::Value>| {
        let mut request = request();
        request.messages = messages;
        request.template_arguments = arguments.as_object().unwrap().clone();
        template
            .prepare(&request)
            .unwrap()
            .description()
            .prompt
            .clone()
    };
    (prompt(first_task), prompt(second_task))
}

#[test]
fn templates_reading_preserve_thinking_keep_prompts_append_only_across_user_messages() {
    let (first, second) = two_task_prompts(QWEN36, json!({}));
    assert!(first.contains("I should search first."));
    assert!(second.starts_with(&first), "{first:?}\n{second:?}");
    assert!(second.contains("The result answers the question."));

    let (first, second) = two_task_prompts(QWEN36, json!({"preserve_thinking": false}));
    assert!(first.contains("I should search first."));
    assert!(!second.contains("I should search first."));
    assert!(!second.contains("The result answers the question."));
}

#[test]
fn templates_without_preserve_thinking_render_unchanged() {
    assert!(!QWEN35.contains("preserve_thinking"));
    let defaults = two_task_prompts(QWEN35, json!({}));
    assert_eq!(
        defaults,
        two_task_prompts(QWEN35, json!({"preserve_thinking": false}))
    );
    assert!(!defaults.1.contains("I should search first."));
}

#[test]
fn peg_parser_save_and_grammar_match_expected_across_families() {
    struct OrderedRuleEntries(Vec<(String, usize)>);

    impl<'de> serde::Deserialize<'de> for OrderedRuleEntries {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct EntriesVisitor;
            impl<'de> serde::de::Visitor<'de> for EntriesVisitor {
                type Value = OrderedRuleEntries;
                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("a JSON object mapping rule names to parser IDs")
                }
                fn visit_map<M: serde::de::MapAccess<'de>>(
                    self,
                    mut map: M,
                ) -> Result<Self::Value, M::Error> {
                    let mut entries = Vec::new();
                    while let Some((key, value)) = map.next_entry::<String, usize>()? {
                        entries.push((key, value));
                    }
                    Ok(OrderedRuleEntries(entries))
                }
            }
            deserializer.deserialize_map(EntriesVisitor)
        }
    }

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SavedPegArena {
        parsers: Vec<serde_json::Value>,
        root: usize,
        rules: OrderedRuleEntries,
    }

    let special = |bos: &str, eos: &str| -> SpecialTokens {
        [
            ("bos_token".to_string(), bos.to_string()),
            ("eos_token".to_string(), eos.to_string()),
        ]
        .into()
    };

    let mut req = request();
    req.template_arguments
        .insert("enable_thinking".into(), json!(true));

    let cases = [
        (
            "deepseek",
            DEEPSEEK,
            special("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
            true,
        ),
        ("qwen3_coder", QWEN35, special("", "<|im_end|>"), true),
        ("gemma4", GEMMA4, special("<bos>", "<eos>"), true),
        (
            "llama",
            LLAMA32,
            special("<|begin_of_text|>", "<|eot_id|>"),
            false,
        ),
    ];

    for (family, source, tokens, expect_grammar) in cases {
        let plan = Template::new(source, &tokens)
            .unwrap_or_else(|e| panic!("{family}: template init failed: {e}"))
            .prepare(&req)
            .unwrap_or_else(|e| panic!("{family}: prepare failed: {e}"));
        let desc = plan.description();
        assert_eq!(desc.grammar_dialect, "gbnf", "{family}");

        let saved: SavedPegArena = serde_json::from_str(&desc.parser)
            .unwrap_or_else(|e| panic!("{family}: invalid saved PEG parser JSON: {e}"));
        assert!(
            !saved.parsers.is_empty(),
            "{family}: saved PEG parser has no parsers"
        );
        assert!(
            saved.parsers.iter().all(|p| p.get("type").and_then(|v| v.as_str()).is_some()),
            "{family}: every parser variant must have a string type"
        );
        assert!(
            saved.root < saved.parsers.len(),
            "{family}: root {} out of bounds for {} parsers",
            saved.root,
            saved.parsers.len()
        );
        assert!(
            saved.rules.0.windows(2).all(|w| w[0].0 < w[1].0),
            "{family}: rule keys are not strictly sorted in serialized JSON"
        );
        assert!(
            saved.rules.0.iter().all(|(_, id)| *id < saved.parsers.len()),
            "{family}: rule parser ID out of bounds"
        );

        let mut stream = plan
            .stream(4096)
            .unwrap_or_else(|e| panic!("{family}: open stream failed: {e}"));
        let mut transcript = Transcript::default();
        transcript.accept(
            stream
                .feed(b"hello")
                .unwrap_or_else(|e| panic!("{family}: stream feed failed: {e}")),
        );
        transcript.accept(
            stream
                .finish(TerminalCause::Length)
                .unwrap_or_else(|e| panic!("{family}: stream finish failed: {e}")),
        );
        assert_eq!(transcript.finish, Some(TerminalCause::Length), "{family}");

        if expect_grammar {
            let grammar = &desc.grammar;
            assert!(!grammar.is_empty(), "{family}: expected non-empty grammar");
            assert!(
                grammar.lines().any(|line| {
                    line.strip_prefix("root ::= ")
                        .is_some_and(|rhs| !rhs.is_empty())
                }),
                "{family}: grammar missing non-empty root rule"
            );
            assert!(
                grammar.lines().all(|line| {
                    line.split_once(" ::= ")
                        .is_some_and(|(name, _)| !name.is_empty())
                }),
                "{family}: malformed GBNF rule line in grammar"
            );
        }
    }
}

#[test]
fn preserve_policy_covers_all_template_toggle_names_and_thinking_transitions() {
    const NEMOTRON35: &str = include_str!("assets/NVIDIA-Nemotron-3.5-Lightning-30B-A3B.jinja");
    const LFM25_THINKING: &str = include_str!("assets/LFM2.5-1.2B-Thinking.jinja");
    const LAGUNA: &str = include_str!("assets/Laguna-S-2.1.jinja");

    let render = |source: &str, messages: Vec<serde_json::Value>, args: serde_json::Value| {
        let template = Template::new(source, &Default::default()).unwrap();
        let mut req = Request::new(messages, 946684800);
        req.template_arguments = args.as_object().unwrap().clone();
        template.prepare(&req).unwrap().description().prompt.clone()
    };

    // Nemotron-3.5: defaults `truncate_history_thinking=false` (preserves reasoning), caller override wins.
    let (n1, n2) = two_task_prompts(NEMOTRON35, json!({}));
    assert!(n2.contains("I should search first."), "Nemotron default must preserve reasoning:\n{n2}");
    assert!(n2.starts_with(&n1), "Nemotron default must be append-only:\n{n1:?}\n{n2:?}");
    let (_, n2_override) = two_task_prompts(NEMOTRON35, json!({"truncate_history_thinking": true}));
    assert!(!n2_override.contains("I should search first."), "Nemotron explicit override must win");

    // LFM2.5-1.2B-Thinking: defaults `keep_past_thinking=true`, caller override wins.
    let lfm_msgs = vec![
        json!({"role":"user","content":"First?"}),
        json!({"role":"assistant","content":"<think>Plan A</think>Ans A"}),
        json!({"role":"user","content":"Second?"}),
        json!({"role":"assistant","content":"<think>Plan B</think>Ans B"}),
        json!({"role":"user","content":"Third?"}),
    ];
    let lfm_default = render(LFM25_THINKING, lfm_msgs.clone(), json!({}));
    assert!(lfm_default.contains("Plan A"), "LFM2.5-1.2B must default keep_past_thinking=true:\n{lfm_default}");
    let lfm_override = render(LFM25_THINKING, lfm_msgs, json!({"keep_past_thinking": false}));
    assert!(!lfm_override.contains("Plan A"), "LFM2.5-1.2B explicit override must win");

    // Laguna-S-2.1: OFF->OFF must NOT inject `preserve_thinking=true` (which would emit `<think></think>` vs `</think>`),
    // while ON->ON and ON->OFF MUST preserve historical reasoning.
    let u1 = vec![json!({"role":"user","content":"Hi"})];
    let u1_off = render(LAGUNA, u1.clone(), json!({"enable_thinking": false}));
    let u2_off_off = render(
        LAGUNA,
        vec![
            json!({"role":"user","content":"Hi"}),
            json!({"role":"assistant","content":"Hello"}),
            json!({"role":"user","content":"Again"}),
        ],
        json!({"enable_thinking": false}),
    );
    assert!(
        u2_off_off.starts_with(&u1_off),
        "Laguna OFF->OFF must stay append-only:\n{u1_off:?}\n{u2_off_off:?}"
    );
    let u2_on_off = render(
        LAGUNA,
        vec![
            json!({"role":"user","content":"Hi"}),
            json!({"role":"assistant","content":"Hello","reasoning_content":"Think 1"}),
            json!({"role":"user","content":"Again"}),
        ],
        json!({"enable_thinking": false}),
    );
    assert!(
        u2_on_off.contains("<think>Think 1</think>"),
        "Laguna ON->OFF must preserve prior turn reasoning:\n{u2_on_off}"
    );

    // Qwen3.6: OFF->OFF keeps `<think>\n\n</think>\n\n` via `preserve_thinking=true` so prefix is preserved.
    let q1_off = render(QWEN36, u1.clone(), json!({"enable_thinking": false}));
    let q2_off_off = render(
        QWEN36,
        vec![
            json!({"role":"user","content":"Hi"}),
            json!({"role":"assistant","content":"Hello"}),
            json!({"role":"user","content":"Again"}),
        ],
        json!({"enable_thinking": false}),
    );
    assert!(
        q2_off_off.starts_with(&q1_off),
        "Qwen3.6 OFF->OFF must preserve empty think block:\n{q1_off:?}\n{q2_off_off:?}"
    );

    // Synthetic `clear_thinking` (GLM-4.7-Flash) and `drop_thinking` (DeepSeek-V4) polarity checks.
    let clear_tmpl = "{%- set clear_thinking = clear_thinking | default(true) -%}\
        {%- for m in messages -%}\
        {%- if m.role == 'assistant' -%}<|assistant|>{% if not clear_thinking and m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}<|end|>\
        {%- else -%}<|{{ m.role }}|>{{ m.content }}<|end|>{%- endif -%}\
        {%- endfor -%}{% if add_generation_prompt %}<|assistant|><think>{% endif %}";
    let drop_tmpl = clear_tmpl.replace("clear_thinking", "drop_thinking");
    let two_turns = vec![
        json!({"role":"user","content":"Q1"}),
        json!({"role":"assistant","content":"A1","reasoning_content":"R1"}),
        json!({"role":"user","content":"Q2"}),
    ];
    assert!(render(clear_tmpl, two_turns.clone(), json!({})).contains("<think>R1</think>"));
    assert!(!render(clear_tmpl, two_turns.clone(), json!({"clear_thinking": true})).contains("<think>R1</think>"));
    assert!(render(&drop_tmpl, two_turns.clone(), json!({})).contains("<think>R1</think>"));
    assert!(!render(&drop_tmpl, two_turns, json!({"drop_thinking": true})).contains("<think>R1</think>"));
}
