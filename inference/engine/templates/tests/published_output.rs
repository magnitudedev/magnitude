//! Published output is monotonic: every reading of a prefix of the output is a
//! prefix of every later reading. Each family's own template prepares a
//! request, and output written in its format, with text that only resembles
//! its markers (a partial or differently spelled delimiter, a special token
//! named in prose), is fed one byte at a time and at every split. A parser
//! that read such text as one thing and then as another would fail the stream
//! with "Output parser would change published output".
use magnitude_templates::{Event, Request, SpecialTokens, Template, TerminalCause};
use serde_json::{json, Value};

const QWEN3: &str = include_str!("assets/Qwen-Qwen3-0.6B.jinja");
const QWEN35: &str = include_str!("assets/Qwen3.5-4B.jinja");
const NEMOTRON: &str = include_str!("assets/NVIDIA-Nemotron-3.5-Lightning-30B-A3B.jinja");
const GEMMA4: &str = include_str!("assets/gemma-4-12B-it.jinja");
const LFM25: &str = include_str!("assets/LFM2.5-2.6B.jinja");
const MINICPM5: &str = include_str!("assets/MiniCPM5-2B.jinja");
const LAGUNA: &str = include_str!("assets/Laguna-S-2.1.jinja");
const KIMI_K2: &str = include_str!("assets/Kimi-K2-Thinking.jinja");
const DEEPSEEK: &str = include_str!("assets/DeepSeek-V3.2.jinja");
const LLAMA32: &str = include_str!("assets/Llama-3.2-3B-Instruct.jinja");

fn special(bos: &str, eos: &str) -> SpecialTokens {
    [
        ("bos_token".to_string(), bos.to_string()),
        ("eos_token".to_string(), eos.to_string()),
    ]
    .into()
}

/// A user turn, with `get_weather` offered when `tools`.
fn request(tools: bool) -> Request {
    let mut request = Request::new(
        vec![json!({"role":"user","content":"Weather in Paris, and a note?"})],
        946684800,
    );
    if tools {
        request.tools = vec![json!({"type":"function","function":{
            "name":"get_weather","description":"Get the weather and file a note",
            "parameters":{"type":"object","properties":{
                "city":{"type":"string"},
                "note":{"type":"string"}
            },"required":["city","note"]}
        }})];
    }
    request
}

#[derive(Debug, Default, PartialEq)]
struct Transcript {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
}

#[derive(Default)]
struct Reader {
    reasoning: String,
    content: String,
    calls: Vec<(String, String, bool)>,
}

impl Reader {
    fn accept(&mut self, events: Vec<Event>) {
        for event in events {
            match event {
                Event::Reasoning { text } => self.reasoning.push_str(&text),
                Event::Content { text } => self.content.push_str(&text),
                Event::ToolStart { index, name, .. } => {
                    assert_eq!(index as usize, self.calls.len());
                    self.calls.push((name, String::new(), false));
                }
                Event::ToolArguments { index, text } => {
                    let call = &mut self.calls[index as usize];
                    assert!(!call.2, "arguments after the call completed");
                    call.1.push_str(&text);
                }
                Event::ToolComplete { index } => {
                    let call = &mut self.calls[index as usize];
                    assert!(!call.2, "a call completed twice");
                    call.2 = true;
                }
                Event::Finish { .. } => {}
            }
        }
    }

    fn transcript(self) -> Transcript {
        Transcript {
            reasoning: self.reasoning,
            content: self.content,
            calls: self
                .calls
                .into_iter()
                .map(|(name, arguments, complete)| {
                    assert!(complete, "call {name} never completed");
                    (name, serde_json::from_str(&arguments).unwrap())
                })
                .collect(),
        }
    }
}

/// Feeds `output` in `chunks` and ends it naturally.
fn read(template: &Template, request: &Request, chunks: &[&[u8]]) -> Transcript {
    let plan = template.prepare(request).unwrap();
    let mut stream = plan.stream(1 << 16).unwrap();
    let mut reader = Reader::default();
    let mut fed = Vec::new();
    for chunk in chunks {
        fed.extend_from_slice(chunk);
        let events = stream.feed(chunk).unwrap_or_else(|error| {
            panic!("{error} after {:?}", String::from_utf8_lossy(&fed))
        });
        reader.accept(events);
    }
    reader.accept(stream.finish(TerminalCause::Natural).unwrap());
    reader.transcript()
}

struct Case {
    source: &'static str,
    special: (&'static str, &'static str),
    tools: bool,
    output: String,
    expected: Transcript,
}

fn transcript(reasoning: &str, content: &str, calls: Vec<(&str, Value)>) -> Transcript {
    Transcript {
        reasoning: reasoning.into(),
        content: content.into(),
        calls: calls
            .into_iter()
            .map(|(name, arguments)| (name.into(), arguments))
            .collect(),
    }
}

fn check(cases: Vec<Case>) {
    for case in cases {
        let template = Template::new(case.source, &special(case.special.0, case.special.1)).unwrap();
        let request = request(case.tools);
        let bytes = case.output.as_bytes();
        let whole = read(&template, &request, &[bytes]);
        assert_eq!(whole, case.expected, "{:?}", case.output);
        let single: Vec<&[u8]> = bytes.chunks(1).collect();
        assert_eq!(read(&template, &request, &single), whole, "{:?}", case.output);
        for split in 1..bytes.len() {
            assert_eq!(
                read(&template, &request, &[&bytes[..split], &bytes[split..]]),
                whole,
                "{:?} split at {split}",
                case.output
            );
        }
    }
}

/// Kimi K2 reasoning ends at `</think>` or at a call opener. Reasoning that
/// names a special token holds `<|` back until it is resolved, rather than
/// reading it as content the next byte takes back.
#[test]
fn kimi_k2_reasoning_holds_back_partial_call_openers() {
    check(vec![
        Case {
            source: KIMI_K2,
            special: ("[BOS]", "[EOS]"),
            tools: false,
            output: "<think>Turns end with <|im_end|>, and <|tool_calls|> is not a marker.</think>The special tokens are shown above.".into(),
            expected: transcript(
                "Turns end with <|im_end|>, and <|tool_calls|> is not a marker.",
                "The special tokens are shown above.",
                vec![],
            ),
        },
        Case {
            source: KIMI_K2,
            special: ("[BOS]", "[EOS]"),
            tools: true,
            output: "<think>Check <|im_end|> handling, then call.</think>Calling it.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\", \"note\": \"a <|tool_call_end b\"}<|tool_call_end|><|tool_calls_section_end|>".into(),
            expected: transcript(
                "Check <|im_end|> handling, then call.",
                "Calling it.",
                vec![("get_weather", json!({"city": "Paris", "note": "a <|tool_call_end b"}))],
            ),
        },
    ]);
}

/// Qwen3.5 and Nemotron (the Qwen3-Coder format): reasoning ends at
/// `</think>` or `<tool_call>`, content at `<tool_call>`, a string argument at
/// `\n</parameter>\n`. Text spelling a longer or different tag is read once.
#[test]
fn qwen3_coder_format_reads_resembling_tags_once() {
    let mut cases = Vec::new();
    for (source, special) in [(QWEN35, ("", "<|im_end|>")), (NEMOTRON, ("<s>", "<|im_end|>"))] {
        cases.push(Case {
            source,
            special,
            tools: false,
            output: "Tags such as </thinking> and <tool_calls> are not mine.\n</think>\n\nThe answer uses `#include <functional>` and </think.".into(),
            expected: transcript(
                "Tags such as </thinking> and <tool_calls> are not mine.\n",
                "The answer uses `#include <functional>` and </think.",
                vec![],
            ),
        });
        cases.push(Case {
            source,
            special,
            tools: true,
            output: "I'll check, not with <tool_calls>.\n</think>\n\nChecking the <toolbox>.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=note>\nfirst\n</parameter> stays\n</parameter>\n</function>\n</tool_call>".into(),
            expected: transcript(
                "I'll check, not with <tool_calls>.\n",
                "Checking the <toolbox>.\n\n",
                vec![("get_weather", json!({"city": "Paris", "note": "first\n</parameter> stays"}))],
            ),
        });
    }
    check(cases);
}

/// Qwen3 (Hermes-style JSON calls in `<tool_call>`, read by the generic
/// autoparser).
#[test]
fn hermes_json_calls_read_resembling_tags_once() {
    check(vec![
        Case {
            source: QWEN3,
            special: ("", "<|im_end|>"),
            tools: false,
            output: "<think>\nThe </thinking> tag and </think > are not my closer.\n</think>\n\nIt is <b>sunny</b>; <tool_call> is just text here.".into(),
            expected: transcript(
                "The </thinking> tag and </think > are not my closer.\n",
                "It is <b>sunny</b>; <tool_call> is just text here.",
                vec![],
            ),
        },
        Case {
            source: QWEN3,
            special: ("", "<|im_end|>"),
            tools: true,
            output: "<think>\nLook it up.\n</think>\n\nUsing the <tool_calls> API.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"note\": \"a </tool_call> b\"}}\n</tool_call>".into(),
            expected: transcript(
                "Look it up.\n",
                "Using the <tool_calls> API.\n",
                vec![("get_weather", json!({"city": "Paris", "note": "a </tool_call> b"}))],
            ),
        },
    ]);
}

/// LFM2.5: reasoning ends at `</think>`, content at `<|tool_call_start|>`.
#[test]
fn lfm2_5_reads_resembling_markers_once() {
    check(vec![
        Case {
            source: LFM25,
            special: ("<|startoftext|>", "<|im_end|>"),
            tools: false,
            output: "Compare </thin with </thinking>.</think>Plain answer; <|tool_call_start is prose.".into(),
            expected: transcript(
                "Compare </thin with </thinking>.",
                "Plain answer; <|tool_call_start is prose.",
                vec![],
            ),
        },
        Case {
            source: LFM25,
            special: ("<|startoftext|>", "<|im_end|>"),
            tools: true,
            output: "Weather.</think>Writing <|tool_call_starts in prose is odd.<|tool_call_start|>[get_weather(city='Paris', note='a ] b')]<|tool_call_end|>".into(),
            expected: transcript(
                "Weather.",
                "Writing <|tool_call_starts in prose is odd.",
                vec![("get_weather", json!({"city": "Paris", "note": "a ] b"}))],
            ),
        },
    ]);
}

/// Gemma 4: a thought is `<|channel>thought…<channel|>`, and content ends at
/// either channel marker or a call opener.
#[test]
fn gemma4_reads_resembling_channel_markers_once() {
    check(vec![
        Case {
            source: GEMMA4,
            special: ("<bos>", "<eos>"),
            tools: false,
            output: "<|channel>thought\nThe <channel> tag and <|chan prefix are text.<channel|>Write <channel> or <|chann literally.".into(),
            expected: transcript(
                "The <channel> tag and <|chan prefix are text.",
                "Write <channel> or <|chann literally.",
                vec![],
            ),
        },
        Case {
            source: GEMMA4,
            special: ("<bos>", "<eos>"),
            tools: true,
            output: "<|channel>thought\nNeed weather.<channel|>Checking; <|tool_calls are text.<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,note:<|\"|>a <|\" b<|\"|>}<tool_call|>".into(),
            expected: transcript(
                "Need weather.",
                "Checking; <|tool_calls are text.",
                vec![("get_weather", json!({"city": "Paris", "note": "a <|\" b"}))],
            ),
        },
    ]);
}

/// MiniCPM5: reasoning ends at `</think>`, content at `<function`, a raw
/// argument at `</param>`.
#[test]
fn minicpm5_reads_resembling_tags_once() {
    check(vec![
        Case {
            source: MINICPM5,
            special: ("<s>", "</s>"),
            tools: false,
            output: "Notes on </thinking>.\n</think>\n\nInclude <functional> here.".into(),
            expected: transcript("Notes on </thinking>.\n", "Include <functional> here.", vec![]),
        },
        Case {
            source: MINICPM5,
            special: ("<s>", "</s>"),
            tools: true,
            output: "Weather.\n</think>\n\nUse the <functor>.\n<function name=\"get_weather\"><param name=\"city\">Paris</param><param name=\"note\">a </para b</param></function>".into(),
            expected: transcript(
                "Weather.\n",
                "Use the <functor>.\n",
                vec![("get_weather", json!({"city": "Paris", "note": "a </para b"}))],
            ),
        },
    ]);
}

/// Laguna (tagged arguments, read by the generic autoparser).
#[test]
fn laguna_reads_resembling_tags_once() {
    check(vec![Case {
        source: LAGUNA,
        special: ("〈|EOS|〉", "〈|EOS|〉"),
        tools: true,
        output: "Weather.</think>See <tool_calls>.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>note</arg_key><arg_value>a </arg_val b</arg_value></tool_call>".into(),
        expected: transcript(
            "Weather.",
            "See <tool_calls>.",
            vec![("get_weather", json!({"city": "Paris", "note": "a </arg_val b"}))],
        ),
    }]);
}

/// DeepSeek V3.2: with tools, reasoning ends at `</think>` or at the calls
/// block, which it may open directly; either way it is read once.
#[test]
fn deepseek_reads_resembling_markers_once() {
    let call = "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name=\"note\" string=\"true\">a </｜DSML｜param b</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>";
    let closed = format!("Mention <｜DSML｜func and </thinking>.</think>Calling.\n\n{call}");
    let direct = format!("Straight to the call.\n\n{call}");
    let arguments = json!({"city": "Paris", "note": "a </｜DSML｜param b"});
    check(vec![
        Case {
            source: DEEPSEEK,
            special: ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
            tools: false,
            output: "The </thinking> tag is not mine.</think>Answer with <｜DSML｜ text.".into(),
            expected: transcript("The </thinking> tag is not mine.", "Answer with <｜DSML｜ text.", vec![]),
        },
        Case {
            source: DEEPSEEK,
            special: ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
            tools: true,
            output: closed,
            expected: transcript(
                "Mention <｜DSML｜func and </thinking>.",
                "Calling.",
                vec![("get_weather", arguments.clone())],
            ),
        },
        Case {
            source: DEEPSEEK,
            special: ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
            tools: true,
            output: direct,
            expected: transcript("Straight to the call.", "", vec![("get_weather", arguments)]),
        },
    ]);
}

/// Llama 3.x writes a call as a bare JSON object, read by the generic
/// autoparser with no grammar under an automatic tool choice.
#[test]
fn bare_json_calls_read_once() {
    check(vec![
        Case {
            source: LLAMA32,
            special: ("<|begin_of_text|>", "<|eot_id|>"),
            tools: true,
            output: "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"note\": \"a } b\"}}".into(),
            expected: transcript("", "", vec![("get_weather", json!({"city": "Paris", "note": "a } b"}))]),
        },
        Case {
            source: LLAMA32,
            special: ("<|begin_of_text|>", "<|eot_id|>"),
            tools: true,
            output: "It is sunny in Paris.".into(),
            expected: transcript("", "It is sunny in Paris.", vec![]),
        },
    ]);
}
