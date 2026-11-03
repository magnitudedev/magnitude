//! Render and answer parity: every described decision model tokenizes, cuts
//! its trunk and places its readouts exactly as its publisher's pinned
//! reference does, and answers as it does.
//!
//! The fixtures (`fixtures/<model>.json`) are written by
//! `support/references.py` from the reference code and the model's own
//! tokenizer, for the requests of `fixtures/requests.json`. Each model's
//! description (`fixtures/<model>-description.json`) is the one its
//! conversion writes into its GGUF, emitted by
//! `catalog/conversions/decision.py describe` from the checkpoint's
//! configuration files. The tokenizers
//! are vocabulary-only GGUFs (llama.cpp's token types, written by
//! `support/vocab_gguf.py`) in the directory
//! `MAGNITUDE_DECISION_VOCABULARIES` names: `d3u.gguf`, `mam.gguf`,
//! `pplx.gguf`, `d1.gguf`. Run with
//! `MAGNITUDE_DECISION_VOCABULARIES=<dir> cargo test -p magnitude-decision --test parity -- --ignored`.
//!
//! Every fixture also carries synthetic readout values and the answers the
//! reference's own code computes from them, a score's level and choice and
//! score confidences TypeSafe's (`references.py` `typesafe`); the tests feed
//! the same values to the prepared decision and compare the answers. A noul's,
//! a choice's and a score's numbers compare within the bounds their goldens
//! carry, derived from the reference's floating-point arithmetic and ours
//! (`references.py` `softmax_bounds`); the numbers of other kinds' answers,
//! whose pipelines those bounds do not cover, within a millionth.
//!
//! One departure is deliberate: request text never tokenizes to a control
//! token. A reference that tokenizes its whole prompt lets it (d3 and qev
//! escape nothing; pplx escapes its tokenizer's `all_special_tokens`, which
//! leave out `<|im_start|>` and other added tokens), so the fixtures spell
//! in request text only what each reference itself neutralizes.
use magnitude_artifacts::{
    gguf::inspect_header, DecisionMetadata, DecisionValue, TokenId, TokenizerPayload,
};
use magnitude_chat::{artifacts::gguf_byte_bpe, request::ImageInput, ByteBpeTokenizer};
use magnitude_decision::answers::{argmax, ReadValue as Parsed, WrittenNumber};
use magnitude_decision::json::{Segment, Text, Value};
use magnitude_decision::plan::{canonical, TokenPlan, TokenReadout};
use magnitude_decision::request::{ChoiceOption, Question, QuestionKind, ReadValue};
use magnitude_decision::{Answer, Decider, DecisionRequest};
use magnitude_readout::{ReadoutUsage, ReadoutValues};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;

const QWEN_IMAGE: &str = "<|vision_start|><|image_pad|><|vision_end|>";

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    /// Logits token lists the readouts name by index (d1).
    #[serde(default)]
    token_lists: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    shared: usize,
    sequences: Vec<Sequence>,
    /// The reference's answers to the sequences' synthetic values.
    answers: Vec<serde_json::Value>,
    /// Each answer's bounds, mirroring its numbers (null where it has none).
    bounds: Vec<serde_json::Value>,
}

/// The tokens a logits readout reads: listed, or an index into the
/// fixture's token lists.
#[derive(Deserialize)]
#[serde(untagged)]
enum Tokens {
    Listed(Vec<u32>),
    Table(usize),
}

/// A readout's synthetic values: listed (a pointer's scores), or a code
/// slot's logits as the generator describes them (`references.py`
/// `synthetic`).
#[derive(Deserialize)]
#[serde(untagged)]
enum Synthetic {
    Listed(Vec<f32>),
    Peak {
        peak: [f64; 2],
        seed: u64,
    },
    Presence {
        presence: f64,
        yes: usize,
        seed: u64,
    },
    Spread {
        seed: u64,
    },
}

/// `references.py` `noise`.
fn noise(seed: u64, index: u64) -> f64 {
    (((seed * 2654435761) ^ (index * 40503)) % 1000) as f64 / 1000.0 * 0.6 - 0.3
}

impl Synthetic {
    fn values(&self, count: usize) -> Vec<f32> {
        let each = |value: &dyn Fn(u64) -> f64| {
            (0..count as u64).map(|index| value(index) as f32).collect()
        };
        match self {
            Self::Listed(values) => values.clone(),
            Self::Peak {
                peak: [center, width],
                seed,
            } => each(&|index| {
                let at = (index as f64 - center) / width;
                -0.5 * at * at + noise(*seed, index)
            }),
            Self::Presence {
                presence,
                yes,
                seed,
            } => each(&|index| {
                (if (index as usize) < *yes {
                    *presence
                } else {
                    0.0
                }) + noise(*seed, index)
            }),
            Self::Spread { seed } => each(&|index| 10.0 * noise(*seed, index)),
        }
    }
}

#[derive(Deserialize)]
struct Sequence {
    tokens: Vec<u32>,
    images: usize,
    trunk: usize,
    /// The question groups after the trunk, where a type packs several.
    #[serde(default)]
    groups: Option<Vec<usize>>,
    readouts: Vec<usize>,
    /// The tokens each logits readout reads; `null` for a score readout.
    #[serde(default)]
    logits: Option<Vec<Option<Tokens>>>,
    /// The row range each score readout reads.
    #[serde(default)]
    scores: Option<Vec<Option<[usize; 2]>>>,
    values: Vec<Synthetic>,
}

#[derive(Deserialize)]
struct Request {
    name: String,
    #[serde(default)]
    types: Option<Vec<String>>,
    state: Value,
    questions: Value,
    #[serde(default)]
    images: usize,
    #[serde(default)]
    assets: Vec<String>,
}

fn tokenizer(name: &str) -> Arc<ByteBpeTokenizer> {
    let directory = std::env::var("MAGNITUDE_DECISION_VOCABULARIES")
        .expect("MAGNITUDE_DECISION_VOCABULARIES names the vocabulary GGUFs");
    let header = inspect_header(format!("{directory}/{name}.gguf")).unwrap();
    let payload = TokenizerPayload::from_directory(&header);
    Arc::new(ByteBpeTokenizer::new(gguf_byte_bpe(&payload, name.into()).unwrap()).unwrap())
}

/// TypeSafe's reference replacement, as the d3 reference implements it: a
/// whole value first, else left to right, longest reference first.
fn place(text: &str, references: &[(String, usize)]) -> Text {
    if let Some((_, image)) = references.iter().find(|(reference, _)| reference == text) {
        return Text::new([Segment::Image(*image)]);
    }
    let mut segments = Vec::new();
    let (mut at, mut start) = (0, 0);
    while at < text.len() {
        match references
            .iter()
            .find(|(reference, _)| text[at..].starts_with(reference.as_str()))
        {
            Some((reference, image)) => {
                segments.push(Segment::Literal(text[start..at].into()));
                segments.push(Segment::Image(*image));
                at += reference.len();
                start = at;
            }
            None => at += text[at..].chars().next().unwrap().len_utf8(),
        }
    }
    segments.push(Segment::Literal(text[start..].into()));
    Text::new(segments)
}

fn placed(value: &Value, references: &[(String, usize)]) -> Value {
    match value {
        Value::String(text) => Value::String(place(text.as_plain().unwrap(), references)),
        Value::Array(items) => {
            Value::Array(items.iter().map(|item| placed(item, references)).collect())
        }
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .map(|(key, item)| (key.clone(), placed(item, references)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn field<'a>(question: &'a Value, name: &str) -> Option<&'a Value> {
    let Value::Object(entries) = question else {
        panic!("a question is an object")
    };
    entries
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

fn number(value: &Value) -> f64 {
    match value {
        Value::Number(number) => number.to_string().parse().unwrap(),
        _ => panic!("expected a number"),
    }
}

fn text(value: &Value) -> String {
    let Value::String(text) = value else {
        panic!("expected a string")
    };
    text.as_plain().unwrap().into()
}

/// The fixture's System One request as the wire adapter maps it.
fn request(request: &Request) -> DecisionRequest {
    let mut sorted: Vec<(String, usize)> = request
        .assets
        .iter()
        .enumerate()
        .map(|(index, reference)| (reference.clone(), request.images + index))
        .collect();
    sorted.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then(a.cmp(b)));
    let Value::Object(questions) = &request.questions else {
        panic!("questions are an object")
    };
    let questions = questions
        .iter()
        .map(|(_, question)| {
            let get = |name| field(question, name).cloned().unwrap_or(Value::Null);
            let criteria = get("criteria");
            let kind = match text(&get("type")).as_str() {
                "noul" => {
                    let of = |name| match &criteria {
                        Value::Object(_) => field(&criteria, name).cloned().unwrap_or(Value::Null),
                        _ => Value::Null,
                    };
                    QuestionKind::Predicate {
                        truthy: placed(&of("true"), &sorted),
                        falsy: placed(&of("false"), &sorted),
                    }
                }
                "choice" => {
                    let Value::Object(options) = &criteria else {
                        panic!("choice criteria")
                    };
                    QuestionKind::Choice {
                        options: options
                            .iter()
                            .map(|(key, description)| ChoiceOption {
                                key: key.clone(),
                                description: placed(description, &sorted),
                            })
                            .collect(),
                        multiple: matches!(get("multiple"), Value::Bool(true)),
                    }
                }
                "score" => {
                    let Value::Array(levels) = &criteria else {
                        panic!("score criteria")
                    };
                    QuestionKind::Score {
                        levels: levels.iter().map(|level| placed(level, &sorted)).collect(),
                    }
                }
                "box" => QuestionKind::Box,
                "boxes" => QuestionKind::Boxes {
                    max: match get("max") {
                        Value::Null => None,
                        max => Some(number(&max) as u32),
                    },
                },
                "text" => QuestionKind::Read {
                    value: ReadValue::Text,
                },
                "date" => QuestionKind::Read {
                    value: ReadValue::Date,
                },
                "number" => {
                    let unit = match get("unit") {
                        Value::Null => None,
                        unit => Some(text(&unit)),
                    };
                    match get("range") {
                        Value::Array(range) => QuestionKind::Estimate {
                            low: number(&range[0]),
                            high: number(&range[1]),
                            step: match get("step") {
                                Value::Null => None,
                                step => Some(number(&step)),
                            },
                            unit,
                        },
                        _ => QuestionKind::Read {
                            value: ReadValue::Number { unit },
                        },
                    }
                }
                other => panic!("unknown kind {other}"),
            };
            Question {
                instructions: placed(&get("instructions"), &sorted),
                kind,
            }
        })
        .collect();
    let images = request.images + request.assets.len();
    DecisionRequest {
        state: placed(&request.state, &sorted),
        images: (0..images)
            .map(|_| ImageInput {
                media_type: "image/png".into(),
                bytes: Arc::from(&[][..]),
            })
            .collect(),
        front: (0..request.images).collect(),
        questions,
    }
}

/// The description `decision.py describe` writes for `model`: its template
/// and keys, each read as its GGUF type reads back, and the rows of the
/// readout head its conversion stores, if any.
fn description(model: &str) -> (DecisionMetadata, Option<usize>) {
    #[derive(Deserialize)]
    struct Written {
        template: String,
        keys: BTreeMap<String, Key>,
        head: Option<usize>,
    }
    #[derive(Deserialize)]
    struct Key {
        r#type: String,
        value: serde_json::Value,
    }
    let path = format!(
        "{}/tests/fixtures/{model}-description.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let written: Written = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}")),
    )
    .unwrap();
    let values = written
        .keys
        .into_iter()
        .map(|(name, Key { r#type, value })| {
            let value = match r#type.as_str() {
                "str" => DecisionValue::String(serde_json::from_value(value).unwrap()),
                "bool" => DecisionValue::Bool(serde_json::from_value(value).unwrap()),
                "i32" => DecisionValue::Integer(serde_json::from_value(value).unwrap()),
                "f32" => DecisionValue::Float(f64::from(
                    serde_json::from_value::<f64>(value).unwrap() as f32,
                )),
                "f64" => DecisionValue::Float(serde_json::from_value(value).unwrap()),
                "[str]" => DecisionValue::Strings(serde_json::from_value(value).unwrap()),
                "[f64]" => DecisionValue::Floats(serde_json::from_value(value).unwrap()),
                other => panic!("{name}: no GGUF type {other}"),
            };
            (name, value)
        })
        .collect();
    (
        DecisionMetadata::new(written.template, false, values),
        written.head,
    )
}

fn compare(type_key: &str, plan: &TokenPlan, expected: &Case, token_lists: &[Vec<u32>]) {
    let name = &expected.name;
    assert_eq!(
        plan.sequences.len(),
        expected.sequences.len(),
        "{type_key} {name}: sequences"
    );
    assert_eq!(plan.shared, expected.shared, "{type_key} {name}: shared");
    for (index, (ours, theirs)) in plan.sequences.iter().zip(&expected.sequences).enumerate() {
        let tokens: Vec<u32> = ours.tokens.iter().map(|token| token.0).collect();
        assert_eq!(
            tokens, theirs.tokens,
            "{type_key} {name} sequence {index}: tokens"
        );
        assert_eq!(
            ours.images.len(),
            theirs.images,
            "{type_key} {name} sequence {index}: images"
        );
        // The reference's attention as it states it: a packing reference's
        // trunk reads every group, each group the trunk and itself; any
        // other's trunk reads itself, and what follows it the trunk and
        // itself.
        let stated: Vec<(std::ops::Range<usize>, Vec<usize>)> = match &theirs.groups {
            Some(groups) => {
                let mut stated = vec![(0..theirs.trunk, (0..=groups.len()).collect())];
                let mut at = theirs.trunk;
                for (group, length) in groups.iter().enumerate() {
                    stated.push((at..at + length, vec![0, group + 1]));
                    at += length;
                }
                stated
            }
            None => std::iter::once((0..theirs.trunk, vec![0]))
                .chain(
                    (theirs.trunk < theirs.tokens.len())
                        .then(|| (theirs.trunk..theirs.tokens.len(), vec![0, 1])),
                )
                .collect(),
        };
        assert_eq!(
            ours.blocks,
            canonical(&stated),
            "{type_key} {name} sequence {index}: blocks"
        );
        let positions: Vec<usize> = ours
            .readouts
            .iter()
            .map(|(position, _)| *position)
            .collect();
        assert_eq!(
            positions, theirs.readouts,
            "{type_key} {name} sequence {index}: readouts"
        );
        if let Some(logits) = &theirs.logits {
            let scores = theirs.scores.as_ref();
            for (slot, ((_, readout), tokens)) in ours.readouts.iter().zip(logits).enumerate() {
                let expected = match tokens {
                    Some(Tokens::Listed(tokens)) => {
                        TokenReadout::Logits(tokens.iter().map(|token| TokenId(*token)).collect())
                    }
                    Some(Tokens::Table(table)) => TokenReadout::Logits(
                        token_lists[*table]
                            .iter()
                            .map(|token| TokenId(*token))
                            .collect(),
                    ),
                    None => {
                        let [start, end] = scores.and_then(|scores| scores[slot]).unwrap();
                        TokenReadout::Scores(start..end)
                    }
                };
                assert_eq!(
                    readout, &expected,
                    "{type_key} {name} sequence {index} readout {slot}"
                );
            }
        }
    }
}

fn parity(type_key: &str, vocabulary: &str, placeholder: Option<&str>) {
    let (metadata, head) = description(type_key);
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(format!(
            "{}/tests/fixtures/{type_key}.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap();
    let requests: Vec<Request> = serde_json::from_slice(
        &std::fs::read(format!(
            "{}/tests/fixtures/requests.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap();
    let decider = Decider::new(&metadata, head, tokenizer(vocabulary), placeholder).unwrap();
    let requests: Vec<&Request> = requests
        .iter()
        .filter(|request| {
            request
                .types
                .as_ref()
                .is_none_or(|types| types.iter().any(|t| t == type_key))
        })
        .collect();
    assert_eq!(requests.len(), fixture.cases.len());
    // A model's plans read logits of chosen tokens and scores only when its
    // description says so: a load of it keeps its output projection packed
    // and prepares scores only then.
    let facts = magnitude_decision::load_facts(&metadata, head).unwrap();
    for (request_case, expected) in requests.into_iter().zip(&fixture.cases) {
        assert_eq!(request_case.name, expected.name);
        let prepared = decider
            .prepare(&request(request_case))
            .unwrap_or_else(|error| panic!("{type_key} {}: {error}", expected.name));
        let reads = |read: fn(&TokenReadout) -> bool| {
            prepared.token_plan().sequences.iter().any(|sequence| {
                sequence.readouts.iter().any(|(_, readout)| read(readout))
            })
        };
        assert!(
            facts.reads_logits || !reads(|readout| matches!(readout, TokenReadout::Logits(_))),
            "{type_key} {} reads logits its description does not declare",
            expected.name
        );
        assert!(
            facts.reads_scores || !reads(|readout| matches!(readout, TokenReadout::Scores(_))),
            "{type_key} {} reads scores its description does not declare",
            expected.name
        );
        compare(
            type_key,
            prepared.token_plan(),
            expected,
            &fixture.token_lists,
        );
        // The synthetic values, one list per readout, as many as it reads.
        let plan = prepared.token_plan();
        let sequences = plan
            .sequences
            .iter()
            .zip(&expected.sequences)
            .map(|(ours, theirs)| {
                ours.readouts
                    .iter()
                    .enumerate()
                    .map(|(slot, (_, readout))| {
                        let count = match readout {
                            TokenReadout::Logits(tokens) => tokens.len(),
                            TokenReadout::Scores(rows) => rows.len(),
                            TokenReadout::Head => head.expect("a head readout reads a stored head"),
                        };
                        theirs.values[slot].values(count)
                    })
                    .collect()
            })
            .collect();
        let decided = prepared
            .answers(&ReadoutValues {
                sequences,
                usage: ReadoutUsage::default(),
            })
            .unwrap_or_else(|error| panic!("{type_key} {}: {error}", expected.name));
        let Value::Object(questions) = &request_case.questions else {
            panic!("questions are an object")
        };
        assert_eq!(decided.answers.len(), expected.answers.len());
        for (((answer, (id, question)), theirs), bounds) in decided
            .answers
            .iter()
            .zip(questions)
            .zip(&expected.answers)
            .zip(&expected.bounds)
        {
            close(
                &reference_shape(answer, question),
                theirs,
                (!bounds.is_null()).then_some(bounds),
                &format!("{type_key} {} {id}", expected.name),
            );
        }
    }
}

/// An answer in `api.answer`'s shape, less a score's legend.
fn reference_shape(answer: &Answer, question: &Value) -> serde_json::Value {
    use serde_json::{json, Map};
    let names: Vec<String> = match field(question, "criteria") {
        Some(Value::Object(options)) => options.iter().map(|(key, _)| key.clone()).collect(),
        _ => Vec::new(),
    };
    let keyed = |probabilities: &[f64]| {
        names
            .iter()
            .cloned()
            .zip(probabilities.iter().map(|p| json!(p)))
            .collect::<Map<_, _>>()
    };
    let rect = |rect: &magnitude_decision::answers::Rect| json!({"top": rect.top, "bottom": rect.bottom, "left": rect.left, "right": rect.right});
    match answer {
        Answer::Predicate { probability } => json!({"type": "noul", "noul": probability}),
        Answer::Choice {
            probabilities,
            confidence,
        } => {
            json!({"type": "choice", "choice": names[argmax(probabilities)], "confidence": confidence,
                    "probabilities": keyed(probabilities)})
        }
        Answer::Multiple { probabilities } => json!({"type": "choice",
            "choices": names.iter().zip(probabilities).filter(|(_, p)| **p >= 0.5).map(|(name, _)| name).collect::<Vec<_>>(),
            "probabilities": keyed(probabilities)}),
        Answer::Score {
            probabilities,
            score,
            confidence,
        } => json!({"type": "score", "score": score, "confidence": confidence,
                    "probabilities": probabilities.iter().enumerate().map(|(level, p)| (level.to_string(), json!(p))).collect::<Map<_, _>>()}),
        Answer::Read { found, read, value } => {
            let mut out = Map::new();
            let number = |number: &Option<WrittenNumber>| match number {
                Some(WrittenNumber::Integer(digits)) => json!(digits.parse::<i64>().unwrap()),
                Some(WrittenNumber::Float(value)) => json!(value),
                None => serde_json::Value::Null,
            };
            let (kind, written) = match value {
                Parsed::Text => ("text", json!(read.as_ref().map(|read| &read.text))),
                Parsed::Number { number: parsed, .. } => ("number", number(parsed)),
                Parsed::Date { date } => ("date", json!(date)),
            };
            out.insert("type".into(), json!(kind));
            out.insert(kind.into(), written);
            out.insert("span".into(), json!(read.as_ref().map(|read| read.span)));
            out.insert("found".into(), json!(found));
            if kind != "text" {
                out.insert("text".into(), json!(read.as_ref().map(|read| &read.text)));
            }
            if let Some(path) = read.as_ref().and_then(|read| read.path.as_ref()) {
                out.insert("path".into(), json!(path));
            }
            if let Parsed::Number { unit, .. } = value {
                out.insert("unit".into(), json!(unit));
            }
            serde_json::Value::Object(out)
        }
        Answer::Estimate { median, p10, p90 } => {
            let unit = match field(question, "unit") {
                Some(unit) if !unit.is_null() => json!(text(unit)),
                _ => serde_json::Value::Null,
            };
            json!({"type": "number", "number": median, "interval": [p10, p90], "unit": unit})
        }
        Answer::Box {
            found,
            rect: drawn,
            confidence,
        } => {
            let mut out =
                json!({"type": "box", "box": drawn.as_ref().map(rect), "confidence": confidence});
            // P(present), for a model that reads presence.
            if let Some(found) = found {
                out["found"] = json!(found);
            }
            out
        }
        Answer::Boxes {
            boxes,
            at_least,
            presence,
        } => json!({"type": "boxes",
            "boxes": boxes.iter().map(|found| json!({"box": rect(&found.rect), "found": found.found, "confidence": found.confidence})).collect::<Vec<_>>(),
            "count": boxes.len(), "at_least": at_least, "presence": presence}),
    }
}

/// Equal, and numbers within their derived `bounds` (the answer's shape),
/// every number of a bounded answer having one; an answer the references
/// derive no bounds for (`None`: its kind's are not derived) within a
/// millionth.
fn close(
    ours: &serde_json::Value,
    theirs: &serde_json::Value,
    bounds: Option<&serde_json::Value>,
    at: &str,
) {
    use serde_json::Value as Json;
    let within = |key: &str| bounds.map(|bounds| bounds.get(key).unwrap_or(&Json::Null));
    match (ours, theirs) {
        (Json::Number(a), Json::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            let bound = match bounds {
                None => 1e-6,
                Some(bound) => bound
                    .as_f64()
                    .unwrap_or_else(|| panic!("{at}: a bounded answer's number has no bound")),
            };
            assert!((a - b).abs() <= bound, "{at}: {a} != {b} (within {bound})");
        }
        (Json::Array(a), Json::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{at}: lengths");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                close(
                    a,
                    b,
                    bounds.map(|bounds| bounds.get(index).unwrap_or(&Json::Null)),
                    &format!("{at}[{index}]"),
                );
            }
        }
        (Json::Object(a), Json::Object(b)) => {
            let keys = |map: &serde_json::Map<String, Json>| {
                map.keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>()
            };
            assert_eq!(keys(a), keys(b), "{at}: keys");
            for (key, value) in a {
                close(value, &b[key], within(key), &format!("{at}.{key}"));
            }
        }
        (a, b) => assert_eq!(a, b, "{at}"),
    }
}

#[test]
#[ignore = "needs MAGNITUDE_DECISION_VOCABULARIES"]
fn mammoth_and_elephant_render_and_answer_as_onejev_qev() {
    parity("qev", "mam", None);
}

#[test]
#[ignore = "needs MAGNITUDE_DECISION_VOCABULARIES"]
fn d3_ultra_renders_and_answers_as_its_reference() {
    parity("d3", "d3u", Some(QWEN_IMAGE));
}

#[test]
#[ignore = "needs MAGNITUDE_DECISION_VOCABULARIES"]
fn pplx_flash_renders_and_answers_as_pplx_protocol_3() {
    parity("pplx", "pplx", Some(QWEN_IMAGE));
}

#[test]
#[ignore = "needs MAGNITUDE_DECISION_VOCABULARIES"]
fn d1_2b_renders_and_answers_as_its_reference() {
    parity("d1", "d1", Some("<|vision_start|><|image|><|vision_end|>"));
}

/// d1-2B described from a configuration with `d1_numbers` off: grids and
/// box edges read as codes.
#[test]
#[ignore = "needs MAGNITUDE_DECISION_VOCABULARIES"]
fn d1_2b_reads_grids_as_codes_without_number_readouts() {
    parity(
        "d1-codes",
        "d1",
        Some("<|vision_start|><|image|><|vision_end|>"),
    );
}
