//! Each in-scope family's own chat template (the catalog GGUF's
//! `tokenizer.chat_template`) renders a tool request, and its parser recovers
//! reasoning and tool calls from output written in the model's format.
use magnitude_templates::{Event, Request, SpecialTokens, Template, TerminalCause};
use serde_json::{json, Value};

const GEMMA4: &str = include_str!("assets/gemma-4-12B-it.jinja");
const LFM25: &str = include_str!("assets/LFM2.5-2.6B.jinja");
const LFM25_MOE: &str = include_str!("assets/LFM2.5-8B-A1B.jinja");
const LAGUNA: &str = include_str!("assets/Laguna-S-2.1.jinja");
const MINICPM5: &str = include_str!("assets/MiniCPM5-2B.jinja");
const MUSE: &str = include_str!("assets/Muse-Glimmer-30B.jinja");
const NEMOTRON: &str = include_str!("assets/NVIDIA-Nemotron-3.5-Lightning-30B-A3B.jinja");

fn special(bos: &str, eos: &str) -> SpecialTokens {
    [
        ("bos_token".to_string(), bos.to_string()),
        ("eos_token".to_string(), eos.to_string()),
    ]
    .into()
}

fn request(enable_thinking: bool) -> Request {
    let mut request = Request::new(
        vec![json!({"role":"user","content":"Weather in Paris, and a note?"})],
        946684800,
    );
    if enable_thinking {
        request
            .template_arguments
            .insert("enable_thinking".into(), json!(true));
    }
    request.tools = vec![json!({"type":"function","function":{
        "name":"get_weather","description":"Get the weather and file a note",
        "parameters":{"type":"object","properties":{
            "city":{"type":"string"},
            "note":{"type":"string"}
        },"required":["city","note"]}
    }})];
    request
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Transcript {
    content: String,
    reasoning: String,
    calls: Vec<(String, String)>,
}

fn parse(
    prepared: &magnitude_templates::PreparedRequest,
    output: &str,
) -> Result<Transcript, magnitude_templates::Error> {
    let mut stream = prepared.stream(1 << 16)?;
    let mut events = Vec::new();
    // Byte-at-a-time feeding exercises every incremental split.
    for byte in output.as_bytes().chunks(1) {
        events.extend(stream.feed(byte)?);
    }
    events.extend(stream.finish(TerminalCause::Natural)?);
    let mut transcript = Transcript::default();
    for event in events {
        match event {
            Event::Content { text } => transcript.content.push_str(&text),
            Event::Reasoning { text } => transcript.reasoning.push_str(&text),
            Event::ToolStart { name, .. } => transcript.calls.push((name, String::new())),
            Event::ToolArguments { index, text } => {
                transcript.calls[index as usize].1.push_str(&text)
            }
            Event::ToolComplete { .. } | Event::Finish { .. } => {}
        }
    }
    Ok(transcript)
}

struct Family {
    name: &'static str,
    source: &'static str,
    special: SpecialTokens,
    /// Reasoning is off by default and on with `enable_thinking`.
    enable_thinking: bool,
    /// Text the rendered generation prompt ends with.
    generation_prompt: &'static str,
    /// A completion in the model's own format: reasoning, then one call of
    /// `get_weather` with a multi-line `note`.
    output: &'static str,
    reasoning: &'static str,
}

fn families() -> Vec<Family> {
    vec![
        Family {
            name: "gemma4",
            source: GEMMA4,
            special: special("<bos>", "<eos>"),
            enable_thinking: true,
            generation_prompt: "<|turn>model\n",
            output: "<|channel>thought\nThe user wants weather.<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,note:<|\"|>first line\nsecond <line> & more<|\"|>}<tool_call|>",
            reasoning: "The user wants weather.",
        },
        Family {
            name: "lfm2.5",
            source: LFM25,
            special: special("<|startoftext|>", "<|im_end|>"),
            enable_thinking: false,
            generation_prompt: "<|im_start|>assistant\n<think>",
            output: "The user wants weather.</think><|tool_call_start|>[get_weather(city='Paris', note='first line\\nsecond <line> & more')]<|tool_call_end|>",
            reasoning: "The user wants weather.",
        },
        Family {
            name: "lfm2.5-moe",
            source: LFM25_MOE,
            special: special("<|startoftext|>", "<|im_end|>"),
            enable_thinking: false,
            generation_prompt: "<|im_start|>assistant\n",
            output: "<think>The user wants weather.</think><|tool_call_start|>[get_weather(city='Paris', note='first line\\nsecond <line> & more')]<|tool_call_end|>",
            reasoning: "The user wants weather.",
        },
        Family {
            name: "laguna",
            source: LAGUNA,
            special: special("〈|EOS|〉", "〈|EOS|〉"),
            enable_thinking: false,
            generation_prompt: "<assistant><think>",
            output: "The user wants weather.</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>note</arg_key><arg_value>first line\nsecond <line> & more</arg_value></tool_call>",
            reasoning: "The user wants weather.",
        },
        Family {
            name: "minicpm5",
            source: MINICPM5,
            special: special("<s>", "</s>"),
            enable_thinking: false,
            generation_prompt: "<|im_start|>assistant\n",
            output: "<think>\nThe user wants weather.\n</think>\n\n<function name=\"get_weather\"><param name=\"city\">Paris</param><param name=\"note\"><![CDATA[first line\nsecond <line> & more]]></param></function>",
            reasoning: "The user wants weather.",
        },
        Family {
            name: "muse-glimmer",
            source: MUSE,
            special: special("<|begin_of_text|>", "<|end_of_text|>"),
            enable_thinking: false,
            generation_prompt: "<|start|>assistant",
            output: " to=self<|message|>The user wants weather.<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n<atem:parameter name=\"note\">first line\nsecond <line> & more</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
            reasoning: "The user wants weather.",
        },
        Family {
            name: "nemotron-h",
            source: NEMOTRON,
            special: special("<s>", "<|im_end|>"),
            enable_thinking: false,
            generation_prompt: "<|im_start|>assistant\n<think>\n",
            output: "The user wants weather.\n</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=note>\nfirst line\nsecond <line> & more\n</parameter>\n</function>\n</tool_call>\n",
            reasoning: "The user wants weather.",
        },
    ]
}

/// Gemma 4 argument dictionaries follow the function's schema: keys in the
/// template's case-insensitive `dictsort` order, nested objects, arrays,
/// enums, integers and nullable values.
const GEMMA4_SCHEMA_TOOL: &str = r#"{"type":"function","function":{
    "name":"forecast","description":"Forecast",
    "parameters":{"type":"object","properties":{
        "units":{"type":"string","enum":["metric","imperial"]},
        "city":{"type":"string"},
        "Days":{"type":"integer"},
        "tags":{"type":"array","items":{"type":"string"}},
        "options":{"type":"object","properties":{
            "verbose":{"type":"boolean"},
            "limit":{"anyOf":[{"type":"integer"},{"type":"null"}]}
        },"required":["verbose"]}
    },"required":["city","units"]}
}}"#;

#[test]
fn gemma4_arguments_follow_the_function_schema() {
    let template = Template::new(GEMMA4, &special("<bos>", "<eos>")).unwrap();
    let mut request = request(false);
    request.tools = vec![serde_json::from_str(GEMMA4_SCHEMA_TOOL).unwrap()];
    let prepared = template.prepare(&request).unwrap();
    assert!(!prepared.description().grammar.is_empty());
    for (arguments, expected) in [
        (
            "{city:<|\"|>Paris<|\"|>,Days:3,options:{limit:null,verbose:true},tags:[<|\"|>a<|\"|>,<|\"|>b\nc<|\"|>],units:<|\"|>metric<|\"|>}",
            json!({"city":"Paris","Days":3,"options":{"limit":null,"verbose":true},"tags":["a","b\nc"],"units":"metric"}),
        ),
        (
            "{city:<|\"|>Oslo<|\"|>,options:{limit:12,verbose:false},units:<|\"|>imperial<|\"|>}",
            json!({"city":"Oslo","options":{"limit":12,"verbose":false},"units":"imperial"}),
        ),
        (
            "{city:<|\"|>Oslo<|\"|>,units:<|\"|>metric<|\"|>}",
            json!({"city":"Oslo","units":"metric"}),
        ),
    ] {
        let output = format!("<|tool_call>call:forecast{arguments}<tool_call|>");
        let transcript = parse(&prepared, &output).unwrap();
        assert_eq!(transcript.calls.len(), 1, "{output}");
        assert_eq!(transcript.calls[0].0, "forecast");
        assert_eq!(
            serde_json::from_str::<Value>(&transcript.calls[0].1).unwrap(),
            expected,
            "{output}"
        );
    }
    // Keys out of order, a missing required key, an undeclared key and values
    // outside their schema do not parse.
    for arguments in [
        "{units:<|\"|>metric<|\"|>,city:<|\"|>Oslo<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>,units:<|\"|>metric<|\"|>,zone:1}",
        "{city:<|\"|>Oslo<|\"|>,units:<|\"|>kelvin<|\"|>}",
        "{city:<|\"|>Oslo<|\"|>,Days:2.5,units:<|\"|>metric<|\"|>}",
        "{city:3,units:<|\"|>metric<|\"|>}",
    ] {
        let output = format!("<|tool_call>call:forecast{arguments}<tool_call|>");
        let parsed = parse(&prepared, &output);
        assert!(
            parsed
                .as_ref()
                .map_or(true, |transcript| transcript.calls.is_empty()),
            "{output} parsed as {parsed:?}"
        );
    }
    // Constraints the dictionary grammar cannot enforce are rejected.
    for parameters in [
        json!({"type":"object","properties":{"city":{"type":"string","pattern":"^[A-Z]"}}}),
        json!({"type":"object","properties":{"days":{"type":"integer","minimum":1}}}),
        json!({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true}),
    ] {
        request.tools =
            vec![json!({"type":"function","function":{"name":"f","parameters":parameters}})];
        assert!(template.prepare(&request).is_err(), "{parameters}");
    }
}

#[test]
fn family_templates_render_tools_and_parse_reasoning_and_calls() {
    let mut failures = Vec::new();
    for family in families() {
        let template = Template::new(family.source, &family.special).unwrap();
        let prepared = match template.prepare(&request(family.enable_thinking)) {
            Ok(prepared) => prepared,
            Err(error) => {
                failures.push(format!("{}: {error}", family.name));
                continue;
            }
        };
        let description = prepared.description();
        if !description.prompt.ends_with(family.generation_prompt)
            || !description.prompt.contains("get_weather")
        {
            failures.push(format!(
                "{}: prompt {:?} (format {})",
                family.name, description.prompt, description.format
            ));
            continue;
        }
        let transcript = match parse(&prepared, family.output) {
            Ok(transcript) => transcript,
            Err(error) => {
                failures.push(format!(
                    "{} (format {}): {error}",
                    family.name, description.format
                ));
                continue;
            }
        };
        let arguments = transcript
            .calls
            .first()
            .and_then(|(_, arguments)| serde_json::from_str::<Value>(arguments).ok());
        if transcript.reasoning.trim() != family.reasoning
            || transcript.calls.len() != 1
            || transcript.calls[0].0 != "get_weather"
            || arguments
                != Some(json!({"city": "Paris", "note": "first line\nsecond <line> & more"}))
            || !transcript.content.trim().is_empty()
        {
            failures.push(format!(
                "{} (format {}): {transcript:?}",
                family.name, description.format
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
