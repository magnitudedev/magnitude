//! OpenAI Decisions, `POST /v1/decisions`: OpenAI's guide and API reference
//! (GA 2026-10-22), adapted to and from a [`DecisionRequest`].
//!
//! `input` becomes the state: a string, or a lone user message whose
//! content is one text or one `input_json` part, is that text or JSON; any
//! other item list is a JSON state holding the items in order. Images in
//! user messages are read before the state, in order, and images in a
//! question's instructions in place; both are base64 data URLs, since
//! network URLs are never fetched. Questions are `predicate`, `choice`
//! (`choices: [{value, description?}]`, values strings or booleans) and
//! `score` (`levels: [{label, description?}]`; a level with a description
//! is the object of both). `prompt_cache_key` and the cache options are
//! accepted: the trunk cache is automatic.
//!
//! Answers are in question order with `name` echoed; `score` and
//! `confidence` are the model's own, as its description computes them.
//! Errors use OpenAI's envelope.

use crate::error::ApiError;
use crate::media;
use crate::{Serving, ServingError};
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use magnitude_chat::request::ImageInput;
use magnitude_decision::answers::{Answer, argmax};
use magnitude_decision::json::{Segment, Text, Value};
use magnitude_decision::request::{ChoiceOption, Question, QuestionKind};
use magnitude_decision::{DecisionAnswers, DecisionError, DecisionRequest};
use serde::Serialize;

/// OpenAI's question count limit.
const MAX_QUESTIONS: usize = 400;

/// A question as the response echoes it.
struct Asked {
    name: Option<String>,
    /// A choice's values as the request typed them; a score's labels.
    values: Vec<Value>,
}

struct Adapted {
    model: String,
    request: DecisionRequest,
    asked: Vec<Asked>,
}

fn refuse(param: impl Into<String>, message: impl Into<String>) -> ApiError {
    let mut error = ApiError::invalid(message);
    error.body.param = Some(param.into());
    error
}

fn field<'v>(value: &'v Value, name: &str, param: &str) -> Result<&'v Value, ApiError> {
    value
        .get(name)
        .ok_or_else(|| refuse(param, format!("`{param}.{name}` is required")))
}

fn image(url: &Value, param: &str) -> Result<ImageInput, ApiError> {
    let url = url
        .as_str()
        .ok_or_else(|| refuse(param, "`image_url` must be a string"))?;
    media::image(url).map_err(|error| refuse(param, error.to_string()))
}

fn kind<'v>(part: &'v Value) -> Option<&'v str> {
    part.get("type").and_then(Value::as_str)
}

/// A user message's content: its images taken to `images` (their indices
/// to `front`), the rest as given.
fn user_content(
    content: &Value,
    param: &str,
    images: &mut Vec<ImageInput>,
    front: &mut Vec<usize>,
) -> Result<Value, ApiError> {
    let Value::Array(parts) = content else {
        return Ok(content.clone());
    };
    let mut kept = Vec::with_capacity(parts.len());
    for (index, part) in parts.iter().enumerate() {
        let param = format!("{param}[{index}]");
        match kind(part) {
            Some("input_image") => {
                front.push(images.len());
                images.push(image(field(part, "image_url", &param)?, &param)?);
            }
            Some("input_text" | "input_json") => kept.push(part.clone()),
            _ => {
                return Err(refuse(
                    param,
                    "a user content part is input_text, input_image or input_json",
                ));
            }
        }
    }
    Ok(Value::Array(kept))
}

/// The state `input` gives, and the images read before it.
fn state(
    input: &Value,
    images: &mut Vec<ImageInput>,
    front: &mut Vec<usize>,
) -> Result<Value, ApiError> {
    let items = match input {
        Value::String(_) => return Ok(input.clone()),
        Value::Array(items) => items,
        _ => return Err(refuse("input", "`input` is a string or an array of items")),
    };
    let mut kept = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let param = format!("input[{index}]");
        let Value::Object(entries) = item else {
            return Err(refuse(param, "an input item is an object"));
        };
        if item.get("role").and_then(Value::as_str) == Some("user") {
            let content = user_content(
                field(item, "content", &param)?,
                &format!("{param}.content"),
                images,
                front,
            )?;
            kept.push(Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            if key == "content" {
                                content.clone()
                            } else {
                                value.clone()
                            },
                        )
                    })
                    .collect(),
            ));
        } else {
            kept.push(item.clone());
        }
    }
    // A lone user message of one text or one JSON part is that state.
    if let [message] = kept.as_slice() {
        if message.get("role").and_then(Value::as_str) == Some("user") {
            match message.get("content") {
                Some(text @ Value::String(_)) => return Ok(text.clone()),
                Some(Value::Array(parts)) => match parts.as_slice() {
                    [] => return Ok(Value::Null),
                    [part] if kind(part) == Some("input_text") => {
                        return Ok(field(part, "text", "input[0].content[0]")?.clone());
                    }
                    [part] if kind(part) == Some("input_json") => {
                        return Ok(field(part, "data", "input[0].content[0]")?.clone());
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    Ok(Value::Array(kept))
}

/// Instructions as text with images in place, or one JSON part.
fn instructions(
    value: &Value,
    param: &str,
    images: &mut Vec<ImageInput>,
) -> Result<Value, ApiError> {
    let parts = match value {
        Value::String(_) | Value::Null => return Ok(value.clone()),
        Value::Array(parts) => parts,
        _ => return Err(refuse(param, "instructions are text or an array of parts")),
    };
    if let [part] = parts.as_slice() {
        if kind(part) == Some("input_json") {
            return Ok(field(part, "data", param)?.clone());
        }
    }
    let mut segments = Vec::with_capacity(parts.len());
    for (index, part) in parts.iter().enumerate() {
        let param = format!("{param}[{index}]");
        match kind(part) {
            Some("input_text") => segments.push(Segment::Literal(
                field(part, "text", &param)?
                    .as_str()
                    .ok_or_else(|| refuse(&param, "`text` must be a string"))?
                    .to_owned(),
            )),
            Some("input_image") => {
                segments.push(Segment::Image(images.len()));
                images.push(image(field(part, "image_url", &param)?, &param)?);
            }
            _ => {
                return Err(refuse(
                    param,
                    "instruction parts are input_text and input_image, or one input_json",
                ));
            }
        }
    }
    Ok(Value::String(Text::new(segments)))
}

fn adapt(body: &Value) -> Result<Adapted, ApiError> {
    if !matches!(body, Value::Object(_)) {
        return Err(ApiError::invalid("the body is a JSON object"));
    }
    let model = field(body, "model", "body")?
        .as_str()
        .ok_or_else(|| refuse("model", "`model` must be a string"))?
        .to_owned();
    let mut images = Vec::new();
    let mut front = Vec::new();
    let state = state(field(body, "input", "body")?, &mut images, &mut front)?;
    let Value::Array(listed) = field(body, "questions", "body")? else {
        return Err(refuse("questions", "`questions` is an array"));
    };
    if listed.is_empty() || listed.len() > MAX_QUESTIONS {
        return Err(refuse(
            "questions",
            format!("a request has 1 to {MAX_QUESTIONS} questions"),
        ));
    }
    let mut questions = Vec::with_capacity(listed.len());
    let mut asked = Vec::with_capacity(listed.len());
    for (index, question) in listed.iter().enumerate() {
        let param = format!("questions[{index}]");
        let name = match question.get("name") {
            None | Some(Value::Null) => None,
            Some(name) => Some(
                name.as_str()
                    .ok_or_else(|| refuse(&param, "`name` must be a string"))?
                    .to_owned(),
            ),
        };
        let described = |entry: &Value, param: &str| -> Result<Value, ApiError> {
            match entry.get("description") {
                None | Some(Value::Null) => Ok(Value::Null),
                Some(text @ Value::String(_)) => Ok(text.clone()),
                Some(_) => Err(refuse(param, "`description` must be a string")),
            }
        };
        let mut values = Vec::new();
        let kind = match field(question, "type", &param)?.as_str() {
            Some("predicate") => QuestionKind::Predicate {
                truthy: Value::Null,
                falsy: Value::Null,
            },
            Some("choice") => {
                let Value::Array(choices) = field(question, "choices", &param)? else {
                    return Err(refuse(&param, "`choices` is an array"));
                };
                let mut options = Vec::with_capacity(choices.len());
                for (choice, entry) in choices.iter().enumerate() {
                    let param = format!("{param}.choices[{choice}]");
                    let value = field(entry, "value", &param)?;
                    let key = match value {
                        Value::Bool(flag) => flag.to_string(),
                        _ => value
                            .as_str()
                            .ok_or_else(|| {
                                refuse(&param, "a choice value is a string or a boolean")
                            })?
                            .to_owned(),
                    };
                    values.push(value.clone());
                    options.push(ChoiceOption {
                        key,
                        description: described(entry, &param)?,
                    });
                }
                QuestionKind::Choice {
                    options,
                    multiple: false,
                }
            }
            Some("score") => {
                let Value::Array(levels) = field(question, "levels", &param)? else {
                    return Err(refuse(&param, "`levels` is an array"));
                };
                let mut read = Vec::with_capacity(levels.len());
                for (level, entry) in levels.iter().enumerate() {
                    let param = format!("{param}.levels[{level}]");
                    let label = field(entry, "label", &param)?;
                    if label.as_str().is_none() {
                        return Err(refuse(&param, "`label` must be a string"));
                    }
                    values.push(label.clone());
                    read.push(match described(entry, &param)? {
                        Value::Null => label.clone(),
                        description => Value::Object(vec![
                            ("label".into(), label.clone()),
                            ("description".into(), description),
                        ]),
                    });
                }
                QuestionKind::Score { levels: read }
            }
            _ => {
                return Err(refuse(
                    &param,
                    "a question's type is predicate, choice or score",
                ));
            }
        };
        let instructions = instructions(
            question.get("instructions").unwrap_or(&Value::Null),
            &format!("{param}.instructions"),
            &mut images,
        )?;
        questions.push(Question {
            instructions,
            kind,
        });
        asked.push(Asked { name, values });
    }
    Ok(Adapted {
        model,
        request: DecisionRequest {
            state,
            images,
            front,
            questions,
        },
        asked,
    })
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Answered<'a> {
    Predicate {
        name: Option<&'a str>,
        probability: f64,
    },
    Choice {
        name: Option<&'a str>,
        choice: &'a Value,
        confidence: f64,
        probabilities: Vec<Choiced<'a>>,
    },
    Score {
        name: Option<&'a str>,
        score: f64,
        confidence: f64,
        probabilities: Vec<Leveled<'a>>,
    },
}

#[derive(Serialize)]
struct Choiced<'a> {
    value: &'a Value,
    probability: f64,
}

#[derive(Serialize)]
struct Leveled<'a> {
    value: usize,
    label: &'a Value,
    probability: f64,
}

#[derive(Serialize)]
struct Decision<'a> {
    model: String,
    answers: Vec<Answered<'a>>,
    usage: Usage,
}

#[derive(Serialize)]
struct Usage {
    input_tokens: u64,
    input_tokens_details: InputDetails,
    output_tokens: u64,
    output_tokens_details: OutputDetails,
    total_tokens: u64,
}

#[derive(Serialize)]
struct InputDetails {
    cached_tokens: u64,
    cache_write_tokens: u64,
}

#[derive(Serialize)]
struct OutputDetails {
    reasoning_tokens: u64,
}

fn answer<'a>(asked: &'a Asked, answer: &Answer) -> Answered<'a> {
    let name = asked.name.as_deref();
    match answer {
        Answer::Predicate { probability } => Answered::Predicate {
            name,
            probability: *probability,
        },
        Answer::Choice {
            probabilities,
            confidence,
        } => Answered::Choice {
            name,
            choice: &asked.values[argmax(probabilities)],
            confidence: *confidence,
            probabilities: asked
                .values
                .iter()
                .zip(probabilities)
                .map(|(value, probability)| Choiced {
                    value,
                    probability: *probability,
                })
                .collect(),
        },
        Answer::Score {
            probabilities,
            score,
            confidence,
        } => Answered::Score {
            name,
            score: *score,
            confidence: *confidence,
            probabilities: asked
                .values
                .iter()
                .zip(probabilities)
                .enumerate()
                .map(|(value, (label, probability))| Leveled {
                    value,
                    label,
                    probability: *probability,
                })
                .collect(),
        },
        _ => unreachable!("`adapt` asks only predicate, choice and score questions"),
    }
}

fn decision<'a>(
    model: String,
    asked: &'a [Asked],
    decided: &DecisionAnswers,
) -> Decision<'a> {
    assert_eq!(asked.len(), decided.answers.len(), "one answer per question");
    Decision {
        model,
        answers: asked
            .iter()
            .zip(&decided.answers)
            .map(|(asked, decided)| answer(asked, decided))
            .collect(),
        usage: Usage {
            input_tokens: decided.usage.input_tokens,
            input_tokens_details: InputDetails {
                cached_tokens: decided.usage.cached_tokens,
                cache_write_tokens: 0,
            },
            output_tokens: 0,
            output_tokens_details: OutputDetails {
                reasoning_tokens: 0,
            },
            total_tokens: decided.usage.input_tokens,
        },
    }
}

/// `POST /v1/decisions`.
pub async fn decisions(State(state): State<Serving>, body: Bytes) -> Result<Response, ApiError> {
    let body: Value = serde_json::from_slice(&body)
        .map_err(|error| ApiError::invalid(format!("the body is not valid JSON: {error}")))?;
    let Adapted {
        model,
        request,
        asked,
    } = adapt(&body)?;
    let decided = state.decide(&model, request).await.map_err(|error| {
        let param = match &error {
            ServingError::Decision(DecisionError::Question { question, .. }) => {
                Some(format!("questions[{question}]"))
            }
            ServingError::Model(_) => Some("model".into()),
            _ => None,
        };
        let mut error = ApiError::from(error);
        error.body.param = param;
        error
    })?;
    let id = state.next_id("decision-icn-");
    let response = decision(model, &asked, &decided);
    Ok(crate::with_request_id(Json(response).into_response(), &id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn input_maps_to_the_state_its_items_give() {
        let text = adapt(&parse(
            r#"{"model": "m", "input": "hello", "questions": [{"type": "predicate", "instructions": "Hi?"}]}"#,
        ))
        .unwrap();
        assert_eq!(text.request.state, Value::string("hello"));
        assert_eq!(text.asked[0].name, None);
        let json = adapt(&parse(
            r#"{"model": "m", "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="},
                    {"type": "input_json", "data": {"b": 1, "a": 2}}]}],
                "questions": [{"type": "score", "name": "s", "levels": [{"label": "low"}, {"label": "high", "description": "very"}]}]}"#,
        ))
        .unwrap();
        assert_eq!(json.request.front, vec![0]);
        assert_eq!(
            serde_json::to_string(&json.request.state).unwrap(),
            r#"{"b":1,"a":2}"#
        );
        let QuestionKind::Score { levels } = &json.request.questions[0].kind else {
            panic!()
        };
        assert_eq!(
            serde_json::to_string(&levels[1]).unwrap(),
            r#"{"label":"high","description":"very"}"#
        );
        let conversation = adapt(&parse(
            r#"{"model": "m", "input": [{"role": "developer", "content": "Be strict."}, {"role": "user", "content": "x"}],
                "questions": [{"type": "predicate"}]}"#,
        ))
        .unwrap();
        assert!(matches!(conversation.request.state, Value::Array(ref items) if items.len() == 2));
    }

    #[test]
    fn network_images_and_unknown_kinds_are_refused() {
        let hosted = parse(
            r#"{"model": "m", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}],
                "questions": [{"type": "predicate"}]}"#,
        );
        assert!(adapt(&hosted).is_err());
        let boxed = parse(r#"{"model": "m", "input": "x", "questions": [{"type": "box"}]}"#);
        assert!(adapt(&boxed).is_err());
    }

    #[test]
    fn choices_echo_their_typed_values() {
        let asked = Asked {
            name: Some("flag".into()),
            values: vec![Value::Bool(true), Value::string("no")],
        };
        let answered = answer(
            &asked,
            &Answer::Choice {
                probabilities: vec![0.75, 0.25],
                confidence: 0.75,
            },
        );
        assert_eq!(
            serde_json::to_string(&answered).unwrap(),
            r#"{"type":"choice","name":"flag","choice":true,"confidence":0.75,"probabilities":[{"value":true,"probability":0.75},{"value":"no","probability":0.25}]}"#
        );
    }
}
