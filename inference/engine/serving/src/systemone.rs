//! System One, `POST /v1/systemone`: TypeSafe's contract (OpenAPI
//! `info.version` 0.2.0 and its documentation), adapted to and from a
//! [`DecisionRequest`].
//!
//! A request names a model, a state, questions keyed by id (`noul`,
//! `choice`, `score`, `box`) and the images its strings reference
//! (`assets`, by TypeSafe's reference replacement), as a JSON body or as
//! `multipart/form-data` whose first part, `request`, is the JSON body and
//! whose file parts are the images assets name by `part`. As llama.cpp,
//! SGLang and Ollama take them, images are also read before the state from
//! top-level `images` (or its alias `files`, never both), data URLs or raw
//! base64, and then from the `image_url` parts of a `state` that is a list
//! of chat messages (or an object holding one as `messages`), which leave
//! the state. d1-2B's extended kinds are its request shapes: a `multiple`
//! choice, `boxes` with an optional `max`, `text`, `number` (an optional
//! `unit`, or a `range` with an optional `step` for an estimate) and
//! `date`. The opt-in
//! `TypeSafe-Danger-Zone-*` headers are accepted and not required: what a
//! model serves is its decision description's.
//!
//! Answers follow TypeSafe's shapes and carry the model's own values: a
//! choice's and a score's `confidence` and a score's `score` are what its
//! description computes. A box answer adds `found` where the model reads
//! presence; the extended kinds answer in d1-2B's shapes (`api.answer`).
//! Validation failures are 422 `{detail: [{loc, msg, type}]}`, body rule
//! failures and an over-long request (`max_tokens_exceeded`) 400
//! `{detail}`, and overload 529. Every response carries
//! `x-typesafe-request-id`.

use crate::Serving;
use crate::error::ApiError;
use crate::media;
use axum::Json;
use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use magnitude_chat::request::ImageInput;
use magnitude_decision::answers::{Answer, ReadValue as Parsed, Rect, WrittenNumber, argmax};
use magnitude_decision::json::{Number, Segment, Text, Value};
use magnitude_decision::request::{ChoiceOption, Question, QuestionKind, ReadValue};
use magnitude_decision::{DecisionAnswers, DecisionError, DecisionRequest};
use magnitude_engine::error::RequestError;
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};
use serde_json::json;

/// TypeSafe's limits.
const MAX_OPTIONS: usize = 255;
const MAX_LEVELS: usize = 10;
const MAX_NESTING: usize = 64;
const MAX_REFERENCE_BYTES: usize = 1024;
const MAX_REFERENCES_BYTES: usize = 65_536;

/// A value's place in the request, as `loc` writes it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Loc {
    Field(String),
    Index(usize),
}

impl Serialize for Loc {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Field(name) => serializer.serialize_str(name),
            Self::Index(index) => serializer.serialize_u64(*index as u64),
        }
    }
}

fn loc(path: &[&str]) -> Vec<Loc> {
    std::iter::once("body")
        .chain(path.iter().copied())
        .map(|part| Loc::Field(part.to_owned()))
        .collect()
}

/// Why a request is refused.
#[derive(Debug)]
enum Refusal {
    /// 422: a field is missing or invalid.
    Invalid {
        loc: Vec<Loc>,
        msg: String,
        kind: &'static str,
    },
    /// 400: the body breaks a request rule.
    Rule(String),
    /// The body could not be read: its extractor's status (413 past the
    /// body limit) and reason.
    Unread(StatusCode, String),
}

fn invalid(loc: Vec<Loc>, kind: &'static str, msg: impl Into<String>) -> Refusal {
    Refusal::Invalid {
        loc,
        msg: msg.into(),
        kind,
    }
}

/// A question as the response needs it: its id, a score's levels as the
/// request wrote them, a choice's option names, and an estimate's unit.
struct Asked {
    id: String,
    levels: Vec<Value>,
    options: Vec<String>,
    unit: Option<String>,
}

/// A multipart request's file part: its name, declared content type and
/// bytes, and whether an asset named it.
struct Part {
    name: String,
    content_type: Option<String>,
    bytes: Bytes,
    named: bool,
}

/// A request's file parts, none for a JSON body.
#[derive(Default)]
struct Parts(Vec<Part>);

impl Parts {
    /// The part `name`, claimed by one asset.
    fn claim(&mut self, name: &str) -> Result<(Option<&str>, &Bytes), Refusal> {
        let part = self
            .0
            .iter_mut()
            .find(|part| part.name == name)
            .ok_or_else(|| Refusal::Rule(format!("no file part is named `{name}`")))?;
        if std::mem::replace(&mut part.named, true) {
            return Err(Refusal::Rule(format!(
                "two assets name the file part `{name}`"
            )));
        }
        Ok((part.content_type.as_deref(), &part.bytes))
    }
}

/// The adapted request.
struct Adapted {
    model: String,
    request: DecisionRequest,
    asked: Vec<Asked>,
}

/// An object's entries, or a refusal at `path`.
fn object<'v>(value: &'v Value, path: &[&str]) -> Result<&'v [(String, Value)], Refusal> {
    match value {
        Value::Object(entries) => Ok(entries),
        _ => Err(invalid(
            loc(path),
            "dict_type",
            "Input should be a valid dictionary",
        )),
    }
}

fn nesting(value: &Value, what: &str) -> Result<(), Refusal> {
    if value.depth() > MAX_NESTING {
        return Err(Refusal::Rule(format!(
            "{what} is nested more than {MAX_NESTING} levels deep"
        )));
    }
    Ok(())
}

/// TypeSafe's reference replacement: a string that is exactly a reference
/// is that image; in a longer string references are replaced left to right
/// without overlapping, the longest first where several match. Object keys
/// stay text.
struct References {
    /// Longest first, then in byte order, each with its image.
    sorted: Vec<(String, usize)>,
    used: Vec<bool>,
}

impl References {
    fn place_text(&mut self, text: &str) -> Text {
        if let Some(index) = self
            .sorted
            .iter()
            .position(|(reference, _)| reference == text)
        {
            self.used[index] = true;
            return Text::new([Segment::Image(self.sorted[index].1)]);
        }
        let mut segments = Vec::new();
        let (mut at, mut start) = (0, 0);
        while at < text.len() {
            match self
                .sorted
                .iter()
                .position(|(reference, _)| text[at..].starts_with(reference.as_str()))
            {
                Some(index) => {
                    self.used[index] = true;
                    segments.push(Segment::Literal(text[start..at].into()));
                    segments.push(Segment::Image(self.sorted[index].1));
                    at += self.sorted[index].0.len();
                    start = at;
                }
                None => at += text[at..].chars().next().map_or(1, char::len_utf8),
            }
        }
        segments.push(Segment::Literal(text[start..].into()));
        Text::new(segments)
    }

    fn place(&mut self, value: &Value) -> Value {
        match value {
            Value::String(text) => Value::String(
                self.place_text(text.as_plain().expect("parsed strings place no images")),
            ),
            Value::Array(items) => {
                Value::Array(items.iter().map(|item| self.place(item)).collect())
            }
            Value::Object(entries) => Value::Object(
                entries
                    .iter()
                    .map(|(key, item)| (key.clone(), self.place(item)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }
}

/// An image as the compatibility lists send it: a data URL or raw base64.
fn listed_image(value: &Value, path: Vec<Loc>) -> Result<ImageInput, Refusal> {
    let text = value.as_str().ok_or_else(|| {
        invalid(
            path.clone(),
            "string_type",
            "Input should be a valid string",
        )
    })?;
    let url = if text.starts_with("data:") {
        text.to_owned()
    } else {
        format!("data:{};base64,{text}", sniffed_media_type(text))
    };
    media::image(&url).map_err(|error| invalid(path, "value_error", error.to_string()))
}

/// The media type of raw base64 image bytes, from their signature; PNG
/// otherwise, for the decoder to refuse.
fn sniffed_media_type(base64: &str) -> &'static str {
    if base64.starts_with("/9j/") {
        "image/jpeg"
    } else {
        "image/png"
    }
}

/// The media type of image bytes, from their signature (JPEG's
/// `FF D8 FF`); PNG otherwise, for the decoder to refuse.
fn sniffed_bytes_type(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        "image/jpeg"
    } else {
        "image/png"
    }
}

/// llama.cpp's chat-message state: a list of messages, or an object
/// holding one as `messages`. Each `image_url` part of a message's content
/// list is an image read before the state (after the listed ones) and
/// leaves the content; every other part stays.
fn chat_images(
    state: &Value,
    images: &mut Vec<ImageInput>,
    front: &mut Vec<usize>,
) -> Result<Value, Refusal> {
    let (messages, wrapped) = match state {
        Value::Object(entries) => match entries.iter().position(|(key, _)| key == "messages") {
            Some(index) => (&entries[index].1, Some(index)),
            None => return Ok(state.clone()),
        },
        other => (other, None),
    };
    let Value::Array(messages) = messages else {
        return Ok(state.clone());
    };
    let mut path = loc(&["state"]);
    if wrapped.is_some() {
        path.push(Loc::Field("messages".into()));
    }
    let mut out = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let Value::Object(fields) = message else {
            out.push(message.clone());
            continue;
        };
        let Some(Value::Array(content)) = message.get("content") else {
            out.push(message.clone());
            continue;
        };
        let mut kept = Vec::with_capacity(content.len());
        for (part_index, part) in content.iter().enumerate() {
            let mut at = path.clone();
            at.extend([
                Loc::Index(index),
                Loc::Field("content".into()),
                Loc::Index(part_index),
            ]);
            match (
                part.get("type").and_then(Value::as_str),
                part.get("image_url"),
            ) {
                (Some("image_url"), Some(image_url)) => {
                    let url = image_url.get("url").unwrap_or(image_url);
                    let url = url.as_str().ok_or_else(|| {
                        invalid(
                            at.clone(),
                            "string_type",
                            "an image_url is a data URL string",
                        )
                    })?;
                    let image = media::image(url)
                        .map_err(|error| invalid(at, "value_error", error.to_string()))?;
                    front.push(images.len());
                    images.push(image);
                }
                (Some("input_audio"), _) if part.get("input_audio").is_some() => {
                    return Err(invalid(at, "value_error", "audio input is not supported"));
                }
                _ => kept.push(part.clone()),
            }
        }
        out.push(Value::Object(
            fields
                .iter()
                .map(|(key, value)| {
                    let value = if key == "content" {
                        Value::Array(kept.clone())
                    } else {
                        value.clone()
                    };
                    (key.clone(), value)
                })
                .collect(),
        ));
    }
    Ok(match (state, wrapped) {
        (Value::Object(entries), Some(at)) => {
            let mut entries = entries.clone();
            entries[at].1 = Value::Array(out);
            Value::Object(entries)
        }
        _ => Value::Array(out),
    })
}

/// A JSON number as a float.
fn float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(Number::Integer(value)) => Some(*value as f64),
        Value::Number(Number::Unsigned(value)) => Some(*value as f64),
        Value::Number(Number::Float(value)) => Some(*value),
        _ => None,
    }
}

/// An image's media type from its format, as a multipart part or a data
/// URL declares it.
fn format_media_type(format: &str) -> Option<&'static str> {
    match format {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        _ => None,
    }
}

fn asset_image(reference: &str, asset: &Value, parts: &mut Parts) -> Result<ImageInput, Refusal> {
    let at = |field: &str| loc(&["assets", reference, field]);
    object(asset, &["assets", reference])?;
    match asset.get("type") {
        None => {}
        Some(kind) if kind.as_str() == Some("image") => {}
        Some(_) => {
            return Err(invalid(
                at("type"),
                "literal_error",
                "Input should be 'image'",
            ));
        }
    }
    let format = match asset.get("format") {
        None => None,
        Some(format) => Some(format.as_str().and_then(format_media_type).ok_or_else(|| {
            invalid(
                at("format"),
                "enum",
                "Input should be 'png', 'jpg' or 'jpeg'",
            )
        })?),
    };
    // An asset with both uses its part.
    if let Some(part) = asset.get("part") {
        let name = part
            .as_str()
            .ok_or_else(|| invalid(at("part"), "string_type", "Input should be a valid string"))?;
        let (content_type, bytes) = parts.claim(name)?;
        let declared = match content_type {
            None | Some("application/octet-stream") => None,
            Some("image/png") => Some("image/png"),
            Some("image/jpeg" | "image/jpg") => Some("image/jpeg"),
            Some(other) => {
                return Err(Refusal::Rule(format!(
                    "file part `{name}` has content type `{other}`, not image/png, image/jpeg, image/jpg or application/octet-stream"
                )));
            }
        };
        if bytes.is_empty() || bytes.len() > media::MAX_HTTP_IMAGE_BYTES {
            return Err(invalid(
                at("part"),
                "value_error",
                format!(
                    "file part `{name}` must hold 1 to {} bytes",
                    media::MAX_HTTP_IMAGE_BYTES
                ),
            ));
        }
        let media_type = format
            .or(declared)
            .unwrap_or_else(|| sniffed_bytes_type(bytes));
        return Ok(ImageInput {
            media_type: media_type.into(),
            bytes: bytes.to_vec().into(),
        });
    }
    let data = asset
        .get("data_base64")
        .ok_or_else(|| invalid(at("data_base64"), "missing", "Field required"))?
        .as_str()
        .ok_or_else(|| {
            invalid(
                at("data_base64"),
                "string_type",
                "Input should be a valid string",
            )
        })?;
    let media_type = format.ok_or_else(|| invalid(at("format"), "missing", "Field required"))?;
    media::image(&format!("data:{media_type};base64,{data}"))
        .map_err(|error| invalid(at("data_base64"), "value_error", error.to_string()))
}

fn adapt(body: &Value, mut parts: Parts) -> Result<Adapted, Refusal> {
    object(body, &[])?;
    let model = body
        .get("model")
        .ok_or_else(|| invalid(loc(&["model"]), "missing", "Field required"))?
        .as_str()
        .ok_or_else(|| {
            invalid(
                loc(&["model"]),
                "string_type",
                "Input should be a valid string",
            )
        })?
        .to_owned();
    let state = body
        .get("state")
        .ok_or_else(|| invalid(loc(&["state"]), "missing", "Field required"))?;
    if matches!(state, Value::Bool(_) | Value::Number(_)) {
        return Err(invalid(
            loc(&["state"]),
            "value_error",
            "state must be a string, an object or an array",
        ));
    }
    nesting(state, "state")?;
    let questions = body
        .get("questions")
        .ok_or_else(|| invalid(loc(&["questions"]), "missing", "Field required"))?;
    let questions = object(questions, &["questions"])?;
    if questions.is_empty() {
        return Err(invalid(
            loc(&["questions"]),
            "too_short",
            "questions should have at least 1 item",
        ));
    }

    let mut images = Vec::new();
    let mut front = Vec::new();
    let listed = match (body.get("images"), body.get("files")) {
        (Some(_), Some(_)) => {
            return Err(invalid(
                loc(&["files"]),
                "value_error",
                "`files` is an alias of `images`: send one of them",
            ))
        }
        (Some(listed), None) => Some(("images", listed)),
        (None, Some(listed)) => Some(("files", listed)),
        (None, None) => None,
    };
    if let Some((field, listed)) = listed {
        let Value::Array(listed) = listed else {
            return Err(invalid(
                loc(&[field]),
                "list_type",
                "Input should be a valid list",
            ));
        };
        for (index, image) in listed.iter().enumerate() {
            let mut path = loc(&[field]);
            path.push(Loc::Index(index));
            front.push(images.len());
            images.push(listed_image(image, path)?);
        }
    }
    let mut references = References {
        sorted: Vec::new(),
        used: Vec::new(),
    };
    if let Some(assets) = body.get("assets") {
        let assets = object(assets, &["assets"])?;
        let total: usize = assets.iter().map(|(reference, _)| reference.len()).sum();
        if total > MAX_REFERENCES_BYTES {
            return Err(Refusal::Rule(format!(
                "asset references exceed {MAX_REFERENCES_BYTES} bytes together"
            )));
        }
        for (reference, asset) in assets {
            if reference.trim().is_empty() || reference.len() > MAX_REFERENCE_BYTES {
                return Err(invalid(
                    loc(&["assets", reference]),
                    "value_error",
                    format!("a reference is 1 to {MAX_REFERENCE_BYTES} bytes, not only whitespace"),
                ));
            }
            references.sorted.push((reference.clone(), images.len()));
            images.push(asset_image(reference, asset, &mut parts)?);
        }
        references
            .sorted
            .sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then(a.cmp(b)));
        references.used = vec![false; references.sorted.len()];
    }

    if let Some(part) = parts.0.iter().find(|part| !part.named) {
        return Err(Refusal::Rule(format!(
            "file part `{}` is not named by any asset",
            part.name
        )));
    }
    let state = references.place(&chat_images(state, &mut images, &mut front)?);
    let mut adapted = Vec::with_capacity(questions.len());
    let mut asked = Vec::with_capacity(questions.len());
    for (id, question) in questions {
        let at = |field: &str| loc(&["questions", id, field]);
        object(question, &["questions", id])?;
        let instructions = question.get("instructions").unwrap_or(&Value::Null);
        nesting(instructions, "instructions")?;
        let criteria = question.get("criteria").unwrap_or(&Value::Null);
        nesting(criteria, "criteria")?;
        let kind = question
            .get("type")
            .ok_or_else(|| invalid(at("type"), "missing", "Field required"))?
            .as_str()
            .ok_or_else(|| invalid(at("type"), "string_type", "Input should be a valid string"))?;
        // d1-2B's `check_question`: each extended field belongs to its kind.
        for (field, owner) in [
            ("multiple", "choice"),
            ("max", "boxes"),
            ("range", "number"),
            ("step", "number"),
        ] {
            if kind != owner && question.get(field).is_some() {
                return Err(invalid(
                    at(field),
                    "extra_forbidden",
                    format!("{field} belongs to {owner}"),
                ));
            }
        }
        let mut levels = Vec::new();
        let mut options = Vec::new();
        let mut echoed = None;
        let kind = match kind {
            "noul" => match criteria {
                Value::Null => QuestionKind::Predicate {
                    truthy: Value::Null,
                    falsy: Value::Null,
                },
                Value::Object(_) => QuestionKind::Predicate {
                    truthy: references.place(criteria.get("true").unwrap_or(&Value::Null)),
                    falsy: references.place(criteria.get("false").unwrap_or(&Value::Null)),
                },
                _ => {
                    return Err(invalid(
                        at("criteria"),
                        "dict_type",
                        "Input should be a valid dictionary",
                    ));
                }
            },
            "choice" => {
                let Value::Object(entries) = criteria else {
                    return Err(invalid(
                        at("criteria"),
                        "dict_type",
                        "Input should be a valid dictionary",
                    ));
                };
                if entries.is_empty() || entries.len() > MAX_OPTIONS {
                    return Err(invalid(
                        at("criteria"),
                        "value_error",
                        format!("a choice has 1 to {MAX_OPTIONS} criteria"),
                    ));
                }
                options = entries.iter().map(|(key, _)| key.clone()).collect();
                let multiple = match question.get("multiple") {
                    None => false,
                    Some(Value::Bool(multiple)) => *multiple,
                    Some(_) => {
                        return Err(invalid(
                            at("multiple"),
                            "bool_type",
                            "Input should be a valid boolean",
                        ));
                    }
                };
                QuestionKind::Choice {
                    options: entries
                        .iter()
                        .map(|(key, description)| ChoiceOption {
                            key: key.clone(),
                            description: references.place(description),
                        })
                        .collect(),
                    multiple,
                }
            }
            "score" => {
                let Value::Array(items) = criteria else {
                    return Err(invalid(
                        at("criteria"),
                        "list_type",
                        "Input should be a valid list",
                    ));
                };
                if items.is_empty() || items.len() > MAX_LEVELS || items.iter().any(Value::is_null)
                {
                    return Err(invalid(
                        at("criteria"),
                        "value_error",
                        format!("a score has 1 to {MAX_LEVELS} levels, none null"),
                    ));
                }
                levels = items.clone();
                QuestionKind::Score {
                    levels: items.iter().map(|level| references.place(level)).collect(),
                }
            }
            "box" => QuestionKind::Box,
            "boxes" => QuestionKind::Boxes {
                max: match question.get("max") {
                    None => None,
                    Some(max) => Some(
                        match max {
                            Value::Number(Number::Integer(max)) => u32::try_from(*max).ok(),
                            Value::Number(Number::Unsigned(max)) => u32::try_from(*max).ok(),
                            _ => None,
                        }
                        .filter(|max| *max >= 1)
                        .ok_or_else(|| {
                            invalid(
                                at("max"),
                                "value_error",
                                "max is a whole number of boxes, at least 1",
                            )
                        })?,
                    ),
                },
            },
            "text" => QuestionKind::Read {
                value: ReadValue::Text,
            },
            "date" => QuestionKind::Read {
                value: ReadValue::Date,
            },
            "number" => {
                let unit = match question.get("unit") {
                    None | Some(Value::Null) => None,
                    Some(unit) => Some(
                        unit.as_str()
                            .ok_or_else(|| {
                                invalid(at("unit"), "string_type", "Input should be a valid string")
                            })?
                            .to_owned(),
                    ),
                };
                match question.get("range") {
                    None if question.get("step").is_some() => {
                        return Err(invalid(at("step"), "value_error", "step needs a range"));
                    }
                    None => QuestionKind::Read {
                        value: ReadValue::Number { unit },
                    },
                    Some(range) => {
                        let bounds = match range {
                            Value::Array(bounds) => {
                                bounds.iter().map(float).collect::<Option<Vec<_>>>()
                            }
                            _ => None,
                        };
                        let Some(&[low, high]) = bounds.as_deref() else {
                            return Err(invalid(
                                at("range"),
                                "value_error",
                                "an estimate's range is [low, high]",
                            ));
                        };
                        let step = match question.get("step") {
                            None => None,
                            Some(step) => Some(float(step).ok_or_else(|| {
                                invalid(at("step"), "float_type", "Input should be a valid number")
                            })?),
                        };
                        echoed = unit.clone();
                        QuestionKind::Estimate {
                            low,
                            high,
                            step,
                            unit,
                        }
                    }
                }
            }
            other => {
                return Err(invalid(
                    at("type"),
                    "union_tag_invalid",
                    format!(
                        "question type `{other}` is not served; expected noul, choice, score, box, boxes, text, number or date"
                    ),
                ));
            }
        };
        adapted.push(Question {
            instructions: references.place(instructions),
            kind,
        });
        asked.push(Asked {
            id: id.clone(),
            levels,
            options,
            unit: echoed,
        });
    }
    if let Some(index) = references.used.iter().position(|used| !used) {
        return Err(Refusal::Rule(format!(
            "asset `{}` is not referenced in the state, instructions or criteria",
            references.sorted[index].0
        )));
    }
    Ok(Adapted {
        model,
        request: DecisionRequest {
            state,
            images,
            front,
            questions: adapted,
        },
        asked,
    })
}

/// Entries serialized as a JSON object in their order.
struct Ordered<V>(Vec<(String, V)>);

impl<V: Serialize> Serialize for Ordered<V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// `values` keyed by their level, `"0"` first.
fn leveled<V>(values: Vec<V>) -> Ordered<V> {
    Ordered(
        (0..values.len())
            .map(|level| level.to_string())
            .zip(values)
            .collect(),
    )
}

/// An answer in TypeSafe's shape, or for the extended kinds d1-2B's
/// (`api.answer`), fields in their order.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Answered<'a> {
    Noul {
        noul: f64,
    },
    Choice {
        choice: &'a str,
        confidence: f64,
        probabilities: Ordered<f64>,
    },
    /// Every option chosen at one half.
    #[serde(rename = "choice")]
    Multiple {
        choices: Vec<&'a str>,
        probabilities: Ordered<f64>,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: Ordered<&'a Value>,
        probabilities: Ordered<f64>,
    },
    Box {
        r#box: Option<Edges>,
        #[serde(skip_serializing_if = "Option::is_none")]
        found: Option<f64>,
        confidence: f64,
    },
    Boxes {
        boxes: Vec<Found>,
        count: usize,
        at_least: bool,
        presence: &'a [f64],
    },
    Text {
        text: Option<&'a str>,
        span: Option<[usize; 2]>,
        found: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<&'a str>,
    },
    #[serde(rename = "number")]
    ReadNumber {
        number: Option<Written<'a>>,
        span: Option<[usize; 2]>,
        found: f64,
        text: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<&'a str>,
        unit: Option<&'a str>,
    },
    Date {
        date: Option<&'a str>,
        span: Option<[usize; 2]>,
        found: f64,
        text: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<&'a str>,
    },
    #[serde(rename = "number")]
    Estimate {
        number: f64,
        interval: [f64; 2],
        unit: Option<&'a str>,
    },
}

#[derive(Serialize)]
struct Edges {
    top: f64,
    bottom: f64,
    left: f64,
    right: f64,
}

impl From<&Rect> for Edges {
    fn from(rect: &Rect) -> Self {
        Self {
            top: rect.top,
            bottom: rect.bottom,
            left: rect.left,
            right: rect.right,
        }
    }
}

/// One of several boxes.
#[derive(Serialize)]
struct Found {
    r#box: Edges,
    found: f64,
    confidence: f64,
}

/// A number as written: an integer exactly, its digits as a JSON number.
struct Written<'a>(&'a WrittenNumber);

impl Serialize for Written<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            WrittenNumber::Integer(digits) => {
                serde_json::value::RawValue::from_string(digits.clone())
                    .map_err(serde::ser::Error::custom)?
                    .serialize(serializer)
            }
            WrittenNumber::Float(value) => serializer.serialize_f64(*value),
        }
    }
}

#[derive(Serialize)]
struct Answers<'a> {
    model: String,
    answers: Ordered<Answered<'a>>,
    usage: Usage,
}

#[derive(Serialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

fn answer<'a>(asked: &'a Asked, answer: &'a Answer) -> Answered<'a> {
    let keyed = |probabilities: &[f64]| {
        Ordered(
            asked
                .options
                .iter()
                .cloned()
                .zip(probabilities.iter().copied())
                .collect(),
        )
    };
    match answer {
        Answer::Predicate { probability } => Answered::Noul { noul: *probability },
        Answer::Choice {
            probabilities,
            confidence,
        } => Answered::Choice {
            choice: &asked.options[argmax(probabilities)],
            confidence: *confidence,
            probabilities: keyed(probabilities),
        },
        Answer::Multiple { probabilities } => Answered::Multiple {
            choices: asked
                .options
                .iter()
                .zip(probabilities)
                .filter(|(_, probability)| **probability >= 0.5)
                .map(|(option, _)| option.as_str())
                .collect(),
            probabilities: keyed(probabilities),
        },
        Answer::Score {
            probabilities,
            score,
            confidence,
        } => Answered::Score {
            score: *score,
            confidence: *confidence,
            legend: leveled(asked.levels.iter().collect()),
            probabilities: leveled(probabilities.clone()),
        },
        Answer::Box {
            found,
            rect,
            confidence,
        } => Answered::Box {
            r#box: rect.as_ref().map(Edges::from),
            found: *found,
            confidence: *confidence,
        },
        Answer::Boxes {
            boxes,
            at_least,
            presence,
        } => Answered::Boxes {
            boxes: boxes
                .iter()
                .map(|found| Found {
                    r#box: Edges::from(&found.rect),
                    found: found.found,
                    confidence: found.confidence,
                })
                .collect(),
            count: boxes.len(),
            at_least: *at_least,
            presence,
        },
        Answer::Read { found, read, value } => {
            let text = read.as_ref().map(|read| read.text.as_str());
            let span = read.as_ref().map(|read| read.span);
            let path = read.as_ref().and_then(|read| read.path.as_deref());
            let found = *found;
            match value {
                Parsed::Text => Answered::Text {
                    text,
                    span,
                    found,
                    path,
                },
                Parsed::Number { number, unit } => Answered::ReadNumber {
                    number: number.as_ref().map(Written),
                    span,
                    found,
                    text,
                    path,
                    unit: unit.as_deref(),
                },
                Parsed::Date { date } => Answered::Date {
                    date: date.as_deref(),
                    span,
                    found,
                    text,
                    path,
                },
            }
        }
        Answer::Estimate { median, p10, p90 } => Answered::Estimate {
            number: *median,
            interval: [*p10, *p90],
            unit: asked.unit.as_deref(),
        },
    }
}

fn response<'a>(model: String, asked: &'a [Asked], decided: &'a DecisionAnswers) -> Answers<'a> {
    assert_eq!(asked.len(), decided.answers.len(), "one answer per question");
    Answers {
        model,
        answers: Ordered(
            asked
                .iter()
                .zip(&decided.answers)
                .map(|(asked, decided)| (asked.id.clone(), answer(asked, decided)))
                .collect(),
        ),
        usage: Usage {
            input_tokens: decided.usage.input_tokens,
            output_tokens: 0,
        },
    }
}

fn refusal_response(refusal: Refusal) -> Response {
    match refusal {
        Refusal::Invalid { loc, msg, kind } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"detail": [{"loc": loc, "msg": msg, "type": kind}]})),
        )
            .into_response(),
        Refusal::Rule(detail) => {
            (StatusCode::BAD_REQUEST, Json(json!({"detail": detail}))).into_response()
        }
        Refusal::Unread(status, detail) => {
            (status, Json(json!({"detail": detail}))).into_response()
        }
    }
}

fn failure_response(error: crate::ServingError, asked: &[Asked]) -> Response {
    match error {
        crate::ServingError::Decision(DecisionError::Question { question, reason }) => {
            let id = asked.get(question).map_or("", |asked| asked.id.as_str());
            refusal_response(invalid(loc(&["questions", id]), "value_error", reason))
        }
        crate::ServingError::Decision(DecisionError::Request { reason }) => {
            refusal_response(invalid(loc(&[]), "value_error", reason))
        }
        crate::ServingError::Request(RequestError::ContextLengthExceeded { required, limit }) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": {
                "type": "max_tokens_exceeded",
                "message": format!("the request needs {required} input tokens; the model reads at most {limit}"),
            }})),
        )
            .into_response(),
        other => {
            let error = ApiError::from(other);
            let status = if error.is_transient_unavailable() {
                // TypeSafe's overload status.
                StatusCode::from_u16(529).expect("529 is a valid status")
            } else {
                error.status
            };
            (status, Json(json!({"detail": error.body.message}))).into_response()
        }
    }
}

/// A request's JSON body and its file parts: the body itself, or a
/// multipart request's first part `request` and the parts after it.
async fn body(state: &Serving, request: Request) -> Result<(Value, Parts), Refusal> {
    let json = |bytes: &[u8]| {
        serde_json::from_slice::<Value>(bytes)
            .map_err(|error| Refusal::Rule(format!("the body is not valid JSON: {error}")))
    };
    let multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));
    if !multipart {
        let bytes = Bytes::from_request(request, state)
            .await
            .map_err(|error| Refusal::Unread(error.status(), error.body_text()))?;
        return Ok((json(&bytes)?, Parts::default()));
    }
    let mut form = Multipart::from_request(request, state)
        .await
        .map_err(|error| Refusal::Unread(error.status(), error.body_text()))?;
    let unread = |error: axum::extract::multipart::MultipartError| {
        Refusal::Unread(error.status(), error.body_text())
    };
    let first = form
        .next_field()
        .await
        .map_err(unread)?
        .filter(|field| field.name() == Some("request"))
        .ok_or_else(|| {
            Refusal::Rule("the first part is named `request` and holds the JSON body".into())
        })?;
    if first
        .content_type()
        .is_some_and(|content_type| content_type != "application/json")
    {
        return Err(Refusal::Rule(
            "the `request` part is application/json".into(),
        ));
    }
    let body = json(&first.bytes().await.map_err(unread)?)?;
    let mut parts = Parts::default();
    while let Some(field) = form.next_field().await.map_err(unread)? {
        let name = field.name().unwrap_or_default().to_owned();
        if name.is_empty() || name == "request" || parts.0.iter().any(|part| part.name == name) {
            return Err(Refusal::Rule(format!(
                "part names are unique, not empty and not `request`: `{name}`"
            )));
        }
        let content_type = field.content_type().map(str::to_owned);
        let bytes = field.bytes().await.map_err(unread)?;
        parts.0.push(Part {
            name,
            content_type,
            bytes,
            named: false,
        });
    }
    Ok((body, parts))
}

/// `POST /v1/systemone`.
pub async fn systemone(State(state): State<Serving>, request: Request) -> Response {
    let id = state.next_id("req_icn_");
    let mut response = match body(&state, request)
        .await
        .and_then(|(body, parts)| adapt(&body, parts))
    {
        Err(refusal) => refusal_response(refusal),
        Ok(Adapted {
            model,
            request,
            asked,
        }) => match state.decide(&model, request).await {
            Err(error) => failure_response(error, &asked),
            Ok(decided) => Json(response(model, &asked, &decided)).into_response(),
        },
    };
    if let Ok(value) = id.parse() {
        response
            .headers_mut()
            .insert("x-typesafe-request-id", value);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn references_replace_whole_values_first_then_left_to_right_longest_first() {
        let body = parse(
            r#"{"model": "m", "state": {"camera": "ref-door", "note": "see ref-door and ref-doorbell"},
                "assets": {"ref-door": {"format": "png", "data_base64": "iVBORw=="},
                           "ref-doorbell": {"format": "jpg", "data_base64": "/9j/AA=="}},
                "questions": {"q": {"type": "noul", "instructions": "Is ref-doorbell lit?"}}}"#,
        );
        let adapted = adapt(&body, Parts::default()).unwrap();
        let request = adapted.request;
        assert_eq!(request.images.len(), 2);
        // Sorted longest first: ref-doorbell is image 1, ref-door image 0.
        assert_eq!(request.state.get("camera").unwrap().images(), vec![0]);
        assert_eq!(request.state.get("note").unwrap().images(), vec![0, 1]);
        assert_eq!(request.questions[0].instructions.images(), vec![1]);
        assert_eq!(request.images[1].media_type, "image/jpeg");
    }

    #[test]
    fn unreferenced_assets_and_malformed_questions_are_refused() {
        let unreferenced = parse(
            r#"{"model": "m", "state": "s", "assets": {"ref-x": {"format": "png", "data_base64": "iVBORw=="}},
                "questions": {"q": {"type": "noul"}}}"#,
        );
        assert!(matches!(
            adapt(&unreferenced, Parts::default()),
            Err(Refusal::Rule(_))
        ));
        let no_state = parse(r#"{"model": "m", "questions": {"q": {"type": "noul"}}}"#);
        assert!(matches!(
            adapt(&no_state, Parts::default()),
            Err(Refusal::Invalid {
                kind: "missing",
                ..
            })
        ));
        let levels = parse(
            r#"{"model": "m", "state": "s", "questions": {"q": {"type": "score", "criteria": []}}}"#,
        );
        assert!(matches!(
            adapt(&levels, Parts::default()),
            Err(Refusal::Invalid { .. })
        ));
    }

    #[test]
    fn answers_take_typesafe_shapes_with_the_models_values() {
        let asked = Asked {
            id: "q".into(),
            levels: vec![Value::string("low"), Value::string("high")],
            options: Vec::new(),
            unit: None,
        };
        let score = serde_json::to_value(answer(
            &asked,
            &Answer::Score {
                probabilities: vec![0.25, 0.75],
                score: 1.0,
                confidence: 0.0,
            },
        ))
        .unwrap();
        // The model's level and confidence, as its description computes them.
        assert_eq!(score["score"], json!(1.0));
        assert_eq!(score["confidence"], json!(0.0));
        assert_eq!(score["legend"], json!({"0": "low", "1": "high"}));
        assert_eq!(score["probabilities"], json!({"0": 0.25, "1": 0.75}));
        let asked = Asked {
            id: "c".into(),
            levels: Vec::new(),
            options: vec!["b".into(), "a".into()],
            unit: None,
        };
        let decided = Answer::Choice {
            probabilities: vec![0.25, 0.75],
            confidence: 0.75,
        };
        let choice = answer(&asked, &decided);
        // Options keep their request order.
        assert_eq!(
            serde_json::to_string(&choice).unwrap(),
            r#"{"type":"choice","choice":"a","confidence":0.75,"probabilities":{"b":0.25,"a":0.75}}"#
        );
    }
}
