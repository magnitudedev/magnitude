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
const QWEN36: &str = include_str!("assets/Qwen3.6-35B-A3B.jinja");
const NEMOTRON: &str = include_str!("assets/NVIDIA-Nemotron-3.5-Lightning-30B-A3B.jinja");
const GEMMA4: &str = include_str!("assets/gemma-4-12B-it.jinja");
const LFM25: &str = include_str!("assets/LFM2.5-2.6B.jinja");
const LFM25_1B_INSTRUCT: &str = include_str!("assets/LFM2.5-1.2B-Instruct.jinja");
const LFM25_1B_THINKING: &str = include_str!("assets/LFM2.5-1.2B-Thinking.jinja");
const LFM25_8B: &str = include_str!("assets/LFM2.5-8B-A1B.jinja");
const LFM25_8B_TBRILLIANCE: &str = include_str!("assets/LFM2.5-8B-A1B-Q3.8-TBrilliance.jinja");
const MINICPM5: &str = include_str!("assets/MiniCPM5-2B.jinja");
const LAGUNA: &str = include_str!("assets/Laguna-S-2.1.jinja");
const MUSE_GLIMMER: &str = include_str!("assets/Muse-Glimmer-30B.jinja");
const KIMI_K2: &str = include_str!("assets/Kimi-K2-Thinking.jinja");
const DEEPSEEK: &str = include_str!("assets/DeepSeek-V3.2.jinja");
const DEEPSEEK41: &str = include_str!("../../../conversion/src/magnitude_conversion/deepseek41.jinja");
const LLAMA32: &str = include_str!("assets/Llama-3.2-3B-Instruct.jinja");

fn special(bos: &str, eos: &str) -> SpecialTokens {
    [
        ("bos_token".to_string(), bos.to_string()),
        ("eos_token".to_string(), eos.to_string()),
    ]
    .into()
}

#[derive(Clone, Copy)]
enum Turn {
    /// A user turn.
    Chat,
    /// A user turn with `get_weather` offered.
    Tools,
    /// The model's reply to a `get_weather` result, `get_weather` offered.
    AfterCall,
}

fn request(turn: Turn) -> Request {
    let mut messages = vec![json!({"role":"user","content":"Weather in Paris, and a note?"})];
    if let Turn::AfterCall = turn {
        messages.push(json!({"role":"assistant","content":"","tool_calls":[{"id":"c0","type":"function",
            "function":{"name":"get_weather","arguments":{"city":"Paris","note":"first"}}}]}));
        messages.push(json!({"role":"tool","tool_call_id":"c0","content":"Sunny"}));
    }
    let mut request = Request::new(messages, 946684800);
    if !matches!(turn, Turn::Chat) {
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
    turn: Turn,
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
        let request = request(case.turn);
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

/// Kimi K2 reasoning ends at `</think>` or at a call opener, which then opens
/// calls. Reasoning that names a special token holds `<|` back until it is
/// resolved, rather than reading it as content the next byte takes back.
#[test]
fn kimi_k2_reasoning_holds_back_partial_call_openers() {
    check(vec![
        Case {
            source: KIMI_K2,
            special: ("[BOS]", "[EOS]"),
            turn: Turn::Chat,
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
            turn: Turn::Tools,
            output: "<think>Check <|im_end|> handling, then call.</think>Calling it.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\", \"note\": \"a <|tool_call_end b\"}<|tool_call_end|><|tool_calls_section_end|>".into(),
            expected: transcript(
                "Check <|im_end|> handling, then call.",
                "Calling it.",
                vec![("get_weather", json!({"city": "Paris", "note": "a <|tool_call_end b"}))],
            ),
        },
        Case {
            source: KIMI_K2,
            special: ("[BOS]", "[EOS]"),
            turn: Turn::AfterCall,
            output: "<think>Sunny; file the note <|tool_call straight away<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Paris\", \"note\": \"sunny\"}<|tool_call_end|>".into(),
            expected: transcript(
                "Sunny; file the note <|tool_call straight away",
                "",
                vec![("get_weather", json!({"city": "Paris", "note": "sunny"}))],
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
            turn: Turn::Chat,
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
            turn: Turn::Tools,
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
            turn: Turn::Chat,
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
            turn: Turn::Tools,
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
            turn: Turn::Chat,
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
            turn: Turn::Tools,
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
/// either channel marker or a call opener. With tools, a call opener also ends
/// a thought, as the tool grammar ends the text before the calls there: after
/// a tool result the prompt opens the thought, and a call the model writes
/// before closing it is a call, whose arguments may then hold `<channel|>`.
#[test]
fn gemma4_reads_resembling_channel_markers_once() {
    check(vec![
        Case {
            source: GEMMA4,
            special: ("<bos>", "<eos>"),
            turn: Turn::Chat,
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
            turn: Turn::Tools,
            output: "<|channel>thought\nNeed weather.<channel|>Checking; <|tool_calls are text.<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,note:<|\"|>a <|\" b<|\"|>}<tool_call|>".into(),
            expected: transcript(
                "Need weather.",
                "Checking; <|tool_calls are text.",
                vec![("get_weather", json!({"city": "Paris", "note": "a <|\" b"}))],
            ),
        },
        Case {
            source: GEMMA4,
            special: ("<bos>", "<eos>"),
            turn: Turn::AfterCall,
            output: "Sunny; file it.<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,note:<|\"|>ends <channel|> then <|tool_call>call:get_weather{x<|\"|>}<tool_call|>".into(),
            expected: transcript(
                "Sunny; file it.",
                "",
                vec![("get_weather", json!({"city": "Paris", "note": "ends <channel|> then <|tool_call>call:get_weather{x"}))],
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
            turn: Turn::Chat,
            output: "Notes on </thinking>.\n</think>\n\nInclude <functional> here.".into(),
            expected: transcript("Notes on </thinking>.\n", "Include <functional> here.", vec![]),
        },
        Case {
            source: MINICPM5,
            special: ("<s>", "</s>"),
            turn: Turn::Tools,
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
        turn: Turn::Tools,
        output: "Weather.</think>See <tool_calls>.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>note</arg_key><arg_value>a </arg_val b</arg_value></tool_call>".into(),
        expected: transcript(
            "Weather.",
            "See <tool_calls>.",
            vec![("get_weather", json!({"city": "Paris", "note": "a </arg_val b"}))],
        ),
    }]);
}

/// DeepSeek V3.2: with tools, reasoning ends at `</think>` or at the calls
/// block, which it may open directly; either way it is read once. A reply to
/// tool results opens with `\n\n` where a reply to the user opens with
/// `<｜Assistant｜>`. DeepSeek V4.1 renders tool results as a user turn, so its
/// reply to them opens with `<｜Assistant｜>`.
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
            turn: Turn::Chat,
            output: "The </thinking> tag is not mine.</think>Answer with <｜DSML｜ text.".into(),
            expected: transcript("The </thinking> tag is not mine.", "Answer with <｜DSML｜ text.", vec![]),
        },
        Case {
            source: DEEPSEEK,
            special: ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
            turn: Turn::Tools,
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
            turn: Turn::Tools,
            output: direct,
            expected: transcript("Straight to the call.", "", vec![("get_weather", arguments.clone())]),
        },
        Case {
            source: DEEPSEEK,
            special: ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
            turn: Turn::AfterCall,
            output: format!("Sunny; file the note.</think>Filing it.\n\n{call}"),
            expected: transcript("Sunny; file the note.", "Filing it.", vec![("get_weather", arguments)]),
        },
        Case {
            source: DEEPSEEK41,
            special: ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
            turn: Turn::AfterCall,
            output: "Sunny; file the note.</think>Filing it.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"note\" string=\"true\">a </｜DSML｜ param b</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>".into(),
            expected: transcript(
                "Sunny; file the note.",
                "Filing it.",
                vec![("get_weather", json!({"city": "Paris", "note": "a </｜DSML｜ param b"}))],
            ),
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
            turn: Turn::Tools,
            output: "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"note\": \"a } b\"}}".into(),
            expected: transcript("", "", vec![("get_weather", json!({"city": "Paris", "note": "a } b"}))]),
        },
        Case {
            source: LLAMA32,
            special: ("<|begin_of_text|>", "<|eot_id|>"),
            turn: Turn::Tools,
            output: "It is sunny in Paris.".into(),
            expected: transcript("", "It is sunny in Paris.", vec![]),
        },
    ]);
}

/// Every text a family's tool grammar admits is read monotonically. Each
/// family's grammar generates its own text from its literals (markers, tool
/// names, keys), their prefixes, pieces of its earlier output and single
/// characters, taking only what the grammar admits, so markers and whole calls
/// land in strings, arguments and reasoning wherever generation could put them.
/// The admitted text is fed one byte at a time and every feed must succeed:
/// published output is never revised. The seeds reach Gemma 4 calls written
/// inside the thought a tool result's prompt opens. Llama 3.2 is absent: under
/// an automatic tool choice it has no grammar.
#[test]
fn grammar_admitted_output_reads_monotonically() {
    const BUDGET: usize = 160;
    const OUTPUTS: usize = 12;
    const SEEDS: [u64; 2] = [38, 234];
    let families: &[(&str, &str, (&str, &str))] = &[
        ("Qwen3", QWEN3, ("", "<|im_end|>")),
        ("Qwen3.5", QWEN35, ("", "<|im_end|>")),
        ("Qwen3.6", QWEN36, ("", "<|im_end|>")),
        ("Nemotron", NEMOTRON, ("<s>", "<|im_end|>")),
        ("Gemma 4", GEMMA4, ("<bos>", "<eos>")),
        ("LFM2.5", LFM25, ("<|startoftext|>", "<|im_end|>")),
        ("LFM2.5 1.2B Instruct", LFM25_1B_INSTRUCT, ("<|startoftext|>", "<|im_end|>")),
        ("LFM2.5 1.2B Thinking", LFM25_1B_THINKING, ("<|startoftext|>", "<|im_end|>")),
        ("LFM2.5 8B", LFM25_8B, ("<|startoftext|>", "<|im_end|>")),
        ("LFM2.5 8B TBrilliance", LFM25_8B_TBRILLIANCE, ("<|startoftext|>", "<|im_end|>")),
        ("MiniCPM5", MINICPM5, ("<s>", "</s>")),
        ("Laguna", LAGUNA, ("〈|EOS|〉", "〈|EOS|〉")),
        ("Muse Glimmer", MUSE_GLIMMER, ("<|begin_of_text|>", "<|end_of_text|>")),
        ("Kimi K2", KIMI_K2, ("[BOS]", "[EOS]")),
        ("DeepSeek V3.2", DEEPSEEK, ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>")),
        ("DeepSeek V4.1", DEEPSEEK41, ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>")),
    ];
    let mut failures = Vec::new();
    for &(family, source, (bos, eos)) in families {
        let template = Template::new(source, &special(bos, eos)).unwrap();
        for turn in [Turn::Tools, Turn::AfterCall] {
            let plan = template.prepare(&request(turn)).unwrap();
            let description = plan.description();
            let grammar = gbnf::Grammar::parse(&description.grammar);
            let mut start = grammar.start();
            for c in description.grammar_initial_prefix.chars() {
                start = grammar.advance(&start, c).unwrap_or_else(|| panic!("{family}'s grammar refuses its initial prefix"));
            }
            let atoms = grammar.atoms();
            for seed in SEEDS {
                let mut random = gbnf::Random(seed);
                let mut echoes = Vec::new();
                for _ in 0..OUTPUTS {
                    let output = grammar.generate(&start, &atoms, &echoes, &mut random, BUDGET);
                    echoes.extend(gbnf::pieces(&output, &mut random));
                    let mut stream = plan.stream(1 << 16).unwrap();
                    for (i, byte) in output.bytes().enumerate() {
                        if let Err(error) = stream.feed(&[byte]) {
                            let fed = String::from_utf8_lossy(&output.as_bytes()[..=i]);
                            failures.push(format!("{family} (seed {seed}): {error} after {fed:?}"));
                            break;
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// GBNF, the language a prepared grammar constrains generation to, recognized
/// one character at a time as sets of rule stacks.
mod gbnf {
    use std::collections::{HashMap, HashSet};

    enum Element {
        Class { negated: bool, ranges: Vec<(char, char)> },
        Rule(usize),
    }

    /// (rule, alternative, next element) frames, the innermost last.
    type Stack = Vec<(usize, usize, usize)>;

    /// The stacks waiting on a character; an empty stack has completed the root.
    pub struct State {
        stacks: Vec<Stack>,
    }

    /// Pieces of `output` for later outputs to repeat: model text quoting
    /// model text, such as a call written into an argument.
    pub fn pieces(output: &str, random: &mut Random) -> Vec<String> {
        let boundaries: Vec<usize> = output.char_indices().map(|(i, _)| i).chain([output.len()]).collect();
        (0..4)
            .filter_map(|_| {
                let start = boundaries[random.below(boundaries.len())];
                let end = boundaries[random.below(boundaries.len())];
                (end > start).then(|| output[start..end].to_string())
            })
            .collect()
    }

    pub struct Grammar {
        rules: Vec<Vec<Vec<Element>>>,
        names: HashMap<String, usize>,
        literals: Vec<String>,
    }

    pub struct Random(pub u64);
    impl Random {
        pub fn below(&mut self, bound: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % bound as u64) as usize
        }
    }

    type Chars<'a> = std::iter::Peekable<std::str::Chars<'a>>;

    fn escaped(chars: &mut Chars) -> char {
        let hex = |chars: &mut Chars, n| {
            let digits: String = chars.take(n).collect();
            char::from_u32(u32::from_str_radix(&digits, 16).unwrap()).unwrap()
        };
        match chars.next().unwrap() {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'x' => hex(chars, 2),
            'u' => hex(chars, 4),
            'U' => hex(chars, 8),
            other => other,
        }
    }

    impl Grammar {
        pub fn parse(text: &str) -> Self {
            let mut grammar = Self { rules: Vec::new(), names: HashMap::new(), literals: Vec::new() };
            for line in text.lines() {
                let Some((name, body)) = line.split_once(" ::= ") else { continue };
                let id = grammar.name(name.trim());
                let alternatives = grammar.alternatives(&mut body.chars().peekable(), None);
                grammar.rules[id] = alternatives;
            }
            grammar
        }

        fn name(&mut self, name: &str) -> usize {
            if let Some(&id) = self.names.get(name) {
                return id;
            }
            self.rules.push(Vec::new());
            self.names.insert(name.into(), self.rules.len() - 1);
            self.rules.len() - 1
        }

        fn synthetic(&mut self, alternatives: Vec<Vec<Element>>) -> usize {
            self.rules.push(alternatives);
            self.rules.len() - 1
        }

        fn alternatives(&mut self, chars: &mut Chars, close: Option<char>) -> Vec<Vec<Element>> {
            let mut alternatives = vec![Vec::new()];
            loop {
                while chars.peek() == Some(&' ') {
                    chars.next();
                }
                let Some(&next) = chars.peek() else { break };
                if Some(next) == close {
                    chars.next();
                    break;
                }
                let item = match next {
                    '|' => {
                        chars.next();
                        alternatives.push(Vec::new());
                        continue;
                    }
                    '"' => {
                        chars.next();
                        let mut literal = String::new();
                        loop {
                            match chars.next().unwrap() {
                                '"' => break,
                                '\\' => literal.push(escaped(chars)),
                                other => literal.push(other),
                            }
                        }
                        let item = literal.chars().map(|c| Element::Class { negated: false, ranges: vec![(c, c)] }).collect();
                        if literal.chars().count() > 1 {
                            self.literals.push(literal);
                        }
                        item
                    }
                    '[' => {
                        chars.next();
                        let negated = chars.next_if_eq(&'^').is_some();
                        let mut ranges = Vec::new();
                        loop {
                            let low = match chars.next().unwrap() {
                                ']' => break,
                                '\\' => escaped(chars),
                                other => other,
                            };
                            let mut ahead = chars.clone();
                            if ahead.next() == Some('-') && ahead.peek() != Some(&']') {
                                chars.next();
                                let high = match chars.next().unwrap() {
                                    '\\' => escaped(chars),
                                    other => other,
                                };
                                ranges.push((low, high));
                            } else {
                                ranges.push((low, low));
                            }
                        }
                        vec![Element::Class { negated, ranges }]
                    }
                    '.' => {
                        chars.next();
                        vec![Element::Class { negated: true, ranges: Vec::new() }]
                    }
                    '(' => {
                        chars.next();
                        let group = self.alternatives(chars, Some(')'));
                        vec![Element::Rule(self.synthetic(group))]
                    }
                    _ => {
                        let mut name = String::new();
                        while let Some(c) = chars.next_if(|c| c.is_alphanumeric() || *c == '-' || *c == '_') {
                            name.push(c);
                        }
                        assert!(!name.is_empty(), "unexpected {next:?} in GBNF");
                        vec![Element::Rule(self.name(&name))]
                    }
                };
                let (low, high) = match chars.next_if(|c| matches!(c, '?' | '*' | '+' | '{')) {
                    Some('?') => (0, Some(1)),
                    Some('*') => (0, None),
                    Some('+') => (1, None),
                    Some(_) => {
                        let bounds: String = chars.by_ref().take_while(|c| *c != '}').collect();
                        let (low, high) = bounds.split_once(',').unwrap_or((&bounds, &bounds));
                        let high = high.trim();
                        (low.trim().parse().unwrap(), (!high.is_empty()).then(|| high.parse().unwrap()))
                    }
                    None => (1, Some(1)),
                };
                let repeated = self.repeat(item, low, high);
                alternatives.last_mut().unwrap().extend(repeated);
            }
            alternatives
        }

        /// `item` between `low` and `high` (unbounded when `None`) times.
        fn repeat(&mut self, item: Vec<Element>, low: usize, high: Option<usize>) -> Vec<Element> {
            if (low, high) == (1, Some(1)) {
                return item;
            }
            let item = self.synthetic(vec![item]);
            let mut sequence: Vec<Element> = (0..low).map(|_| Element::Rule(item)).collect();
            match high {
                None => {
                    let star = self.rules.len();
                    self.rules.push(vec![vec![Element::Rule(item), Element::Rule(star)], Vec::new()]);
                    sequence.push(Element::Rule(star));
                }
                Some(high) => {
                    let mut optional = None;
                    for _ in low..high {
                        let mut more = vec![Element::Rule(item)];
                        more.extend(optional.map(Element::Rule));
                        optional = Some(self.synthetic(vec![more, Vec::new()]));
                    }
                    sequence.extend(optional.map(Element::Rule));
                }
            }
            sequence
        }

        pub fn start(&self) -> State {
            let root = self.names["root"];
            self.expand((0..self.rules[root].len()).map(|a| vec![(root, a, 0)]).collect())
        }

        /// Steps every stack to a character element, or to its end.
        fn expand(&self, mut pending: Vec<Stack>) -> State {
            let mut seen = HashSet::new();
            let mut stacks = Vec::new();
            while let Some(mut stack) = pending.pop() {
                if !seen.insert(stack.clone()) {
                    continue;
                }
                assert!(seen.len() < 100_000, "the GBNF recognizer exceeded its stack bound");
                let Some(&(rule, alternative, next)) = stack.last() else {
                    stacks.push(stack);
                    continue;
                };
                let elements = &self.rules[rule][alternative];
                match elements.get(next) {
                    None => {
                        stack.pop();
                        pending.push(stack);
                    }
                    Some(Element::Class { .. }) => stacks.push(stack),
                    Some(&Element::Rule(called)) => {
                        // The caller resumes after the call; a call in tail
                        // position returns nowhere, so loops stay finite.
                        stack.last_mut().unwrap().2 += 1;
                        if next + 1 == elements.len() {
                            stack.pop();
                        }
                        for a in 0..self.rules[called].len() {
                            let mut branch = stack.clone();
                            branch.push((called, a, 0));
                            pending.push(branch);
                        }
                    }
                }
            }
            State { stacks }
        }

        fn class(&self, stack: &Stack) -> Option<(bool, &[(char, char)])> {
            let &(rule, alternative, next) = stack.last()?;
            match &self.rules[rule][alternative][next] {
                Element::Class { negated, ranges } => Some((*negated, ranges)),
                Element::Rule(_) => unreachable!("expanded stacks wait on characters"),
            }
        }

        pub fn advance(&self, state: &State, c: char) -> Option<State> {
            let next: Vec<Stack> = state
                .stacks
                .iter()
                .filter(|stack| {
                    self.class(stack).is_some_and(|(negated, ranges)| {
                        ranges.iter().any(|&(low, high)| low <= c && c <= high) != negated
                    })
                })
                .map(|stack| {
                    let mut stack = stack.clone();
                    stack.last_mut().unwrap().2 += 1;
                    stack
                })
                .collect();
            (!next.is_empty()).then(|| self.expand(next))
        }

        /// The grammar's literals, then their proper prefixes: its markers,
        /// names and keys, whole or partial, which single characters rarely
        /// spell.
        pub fn atoms(&self) -> (Vec<String>, Vec<String>) {
            let mut whole = self.literals.clone();
            whole.sort();
            whole.dedup();
            let mut partial: Vec<String> = whole
                .iter()
                .flat_map(|literal| literal.char_indices().skip(1).map(|(i, _)| literal[..i].to_string()))
                .collect();
            partial.sort();
            partial.dedup();
            (whole, partial)
        }

        /// Text from `state` the grammar admits, up to about `budget` bytes:
        /// whole literals, pieces of earlier output, literal prefixes or single
        /// characters, each only where the grammar admits it.
        pub fn generate(&self, state: &State, atoms: &(Vec<String>, Vec<String>), echoes: &[String], random: &mut Random, budget: usize) -> String {
            const CHARACTERS: &[char] = &['a', 'x', ' ', '\n', '<', '>', '|', '"', '\\', '{', '}', '[', ']', ',', ':', '_', '1', '-', 'é'];
            let mut output = String::new();
            let mut state = State { stacks: state.stacks.clone() };
            while output.len() < budget {
                let pick = random.below(5);
                if pick < 4 {
                    let pool = match pick {
                        0 | 1 => &atoms.0,
                        2 if !echoes.is_empty() => echoes,
                        _ if !atoms.1.is_empty() => &atoms.1,
                        _ => &atoms.0,
                    };
                    let atom = &pool[random.below(pool.len())];
                    let mut next = Some(State { stacks: state.stacks.clone() });
                    for c in atom.chars() {
                        next = next.and_then(|next| self.advance(&next, c));
                    }
                    if let Some(next) = next {
                        output.push_str(atom);
                        state = next;
                        continue;
                    }
                }
                let waiting: Vec<_> = state.stacks.iter().filter_map(|stack| self.class(stack)).collect();
                if waiting.is_empty() {
                    break;
                }
                let (negated, ranges) = waiting[random.below(waiting.len())];
                let admits = |c: char| ranges.iter().any(|&(low, high)| low <= c && c <= high) != negated;
                let c = (0..16)
                    .map(|_| CHARACTERS[random.below(CHARACTERS.len())])
                    .find(|&c| admits(c))
                    .or_else(|| ranges.iter().map(|range| range.0).find(|&c| admits(c)))
                    .or_else(|| (' '..='~').find(|&c| admits(c)))
                    .expect("a character class admits a character");
                output.push(c);
                state = self.advance(&state, c).unwrap();
            }
            output
        }
    }
}
