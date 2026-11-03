//! The decision prompt: the model's decision template rendered natively
//! over a fixed context, and the anchors it marks (both specified in
//! `KEYS.md`, "Template").
//!
//! The context is request data, so every string in it renders as literal
//! text; the template's own text is control text (the native renderer's
//! provenance, patches 0030 and 0031). Anchors are written with the
//! noncharacters Unicode reserves for a process's internal use, and only
//! control text is searched for them, so request text never forms one.

use crate::json::{python_general, python_space, write_json, Segment, Sink, Style, Text, Value};
use crate::kinds::{Asked, Kind};
use crate::render::{Escape, Rendered};
use crate::request::ReadValue;
use magnitude_templates::{RenderedPrompt, SpecialTokens, Template};
use serde_json::{json, Map, Value as Json};
use std::ops::Range;
use std::sync::Mutex;

const OPEN: char = '\u{FDD0}';
const CLOSE: char = '\u{FDD1}';

/// The model's decision template, compiled once. The native renderer
/// serializes every call, so one lock costs nothing.
pub(crate) struct PromptTemplate {
    template: Mutex<Template>,
}

/// One rendering with its anchors removed: where each of its questions
/// starts (the first where the state ends), where its slots end, and the
/// state's text a read indexes, when the template places it.
pub(crate) struct Anchored {
    pub rendered: Rendered,
    pub starts: Vec<usize>,
    pub slots: Vec<Vec<usize>>,
    pub read: Option<StateText>,
}

/// The state's text as a `read:<form>` anchor places it: where it starts
/// in the rendering, the form's own text, the boundary of that text each
/// character boundary of its escaped rendering stands at, and whether the
/// form is JSON.
pub(crate) struct StateText {
    pub start: usize,
    pub text: String,
    pub boundaries: Vec<usize>,
    pub json: bool,
}

/// The state written in `form` (`text`, `line`, `indented`), when it places
/// no image.
fn form_text(state: &Value, form: &str) -> Result<(String, bool), String> {
    #[derive(Default)]
    struct Plain {
        text: String,
        image: bool,
    }
    impl Sink for Plain {
        fn text(&mut self, text: &str) {
            self.text.push_str(text);
        }
        fn image(&mut self, _: usize, _: bool) {
            self.image = true;
        }
    }
    let mut plain = Plain::default();
    let json = match (form, state) {
        ("text", Value::String(text)) => {
            for segment in text.segments() {
                match segment {
                    Segment::Literal(literal) => plain.text(literal),
                    Segment::Image(image) => plain.image(*image, false),
                }
            }
            false
        }
        ("text", _) => return Err("the decision template reads a state that is no string as text".into()),
        ("line", _) => {
            write_json(state, Style::Line, &mut plain);
            true
        }
        ("indented", _) => {
            write_json(state, Style::Indented, &mut plain);
            true
        }
        _ => {
            return Err(format!(
                "the decision template writes the anchor `read:{form}`, which names no form of the state"
            ))
        }
    };
    if plain.image {
        return Err("a read indexes the state's text, but the state places an image".into());
    }
    Ok((plain.text, json))
}

impl PromptTemplate {
    pub fn new(source: &str) -> Result<Self, String> {
        Template::new(source, &SpecialTokens::new())
            .map(|template| Self {
                template: Mutex::new(template),
            })
            .map_err(|error| {
                format!("`tokenizer.chat_template.decision` does not compile: {error}")
            })
    }

    /// `codes.probe`: the text the template renders for the context
    /// `{"probe": true}`, after which a letter code must stay the one token
    /// it is alone (the reference's own probe, as its template writes it).
    /// It is template text throughout and marks no anchor.
    pub fn probe(&self) -> Result<Rendered, String> {
        let mut context = Map::new();
        context.insert("probe".into(), json!(true));
        let RenderedPrompt { prompt, controls } = self
            .template
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .render(&context, 0)
            .map_err(|error| format!("`codes.probe`: the decision template does not render its probe: {error}"))?;
        if prompt.contains([OPEN, CLOSE]) {
            return Err("`codes.probe`: the decision template's probe marks an anchor".into());
        }
        Ok(Rendered {
            text: prompt,
            controls,
            images: Vec::new(),
        })
    }

    /// Render `context` for `questions` questions over `images` request
    /// images, each image anchor replaced by `placeholder` and a read
    /// anchor by `state`'s form, escaped by `escape`.
    pub fn render(
        &self,
        context: &Map<String, Json>,
        questions: usize,
        images: usize,
        placeholder: Option<&str>,
        state: &Value,
        escape: &Escape,
    ) -> Result<Anchored, String> {
        let rendered = self
            .template
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .render(context, 0)
            .map_err(|error| format!("the decision template does not render: {error}"))?;
        anchored(rendered, questions, images, placeholder, state, escape)
    }
}

/// A rendering's text with its anchors removed, as it reads them.
fn anchored(
    prompt: RenderedPrompt,
    questions: usize,
    images: usize,
    placeholder: Option<&str>,
    state: &Value,
    escape: &Escape,
) -> Result<Anchored, String> {
    let RenderedPrompt {
        prompt: text,
        controls,
    } = prompt;
    let mut out = Builder::default();
    let mut starts = vec![None; questions];
    let mut slots = vec![Vec::new(); questions];
    let mut read = None;
    let mut at = 0;
    for range in controls {
        out.literal(&text[at..range.start]);
        let mut piece = &text[range.clone()];
        while let Some(open) = piece.find(OPEN) {
            let (before, rest) = (&piece[..open], &piece[open + OPEN.len_utf8()..]);
            if before.contains(CLOSE) {
                return Err("the decision template closes an anchor it never opened".into());
            }
            out.control(before);
            let close = rest.find(CLOSE).ok_or(
                "a decision template anchor does not close within the template's own text",
            )?;
            let name = &rest[..close];
            let index = |prefix: &str, bound: usize| -> Option<usize> {
                let written = name.strip_prefix(prefix)?;
                written
                    .parse::<usize>()
                    .ok()
                    .filter(|index| index.to_string() == written && *index < bound)
            };
            let start = match name {
                "state" => Some(0),
                _ => index("question:", questions).filter(|question| *question > 0),
            };
            if let Some(question) = start {
                if starts[question].replace(out.offset()).is_some() {
                    return Err(format!(
                        "the decision template writes the anchor `{name}` twice"
                    ));
                }
            } else if let Some(question) = index("slot:", questions) {
                slots[question].push(out.offset());
            } else if let Some(form) = name.strip_prefix("read:") {
                let (text, json) = form_text(state, form)?;
                let (escaped, boundaries) = escape.apply_with_boundaries(&text);
                let placed = StateText {
                    start: out.offset(),
                    text,
                    boundaries,
                    json,
                };
                if read.replace(placed).is_some() {
                    return Err("the decision template places the state's text twice".into());
                }
                out.literal(&escaped);
            } else if let Some(image) = index("image:", images) {
                let placeholder = placeholder
                    .ok_or("the decision template places an image, but the model reads none")?;
                out.control(placeholder);
                out.images.push(image);
            } else {
                return Err(format!(
                    "the decision template writes the anchor `{name}`, which names no state, question or slot of its questions or image of the request"
                ));
            }
            piece = &rest[close + CLOSE.len_utf8()..];
        }
        if piece.contains(CLOSE) {
            return Err("the decision template closes an anchor it never opened".into());
        }
        out.control(piece);
        at = range.end;
    }
    out.literal(&text[at..]);
    let starts = starts
        .into_iter()
        .enumerate()
        .map(|(question, start)| {
            start.ok_or_else(|| match question {
                0 => "the decision template does not mark the state's end".to_owned(),
                _ => {
                    format!("the decision template does not mark where question {question} starts")
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if starts.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("the decision template's questions do not start in order".into());
    }
    Ok(Anchored {
        starts,
        slots,
        read,
        rendered: Rendered {
            text: out.text,
            controls: out.controls,
            images: out.images,
        },
    })
}

#[derive(Default)]
struct Builder {
    text: String,
    controls: Vec<Range<usize>>,
    images: Vec<usize>,
}

impl Builder {
    fn offset(&self) -> usize {
        self.text.len()
    }

    fn literal(&mut self, text: &str) {
        self.text.push_str(text);
    }

    fn control(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let start = self.text.len();
        self.text.push_str(text);
        match self.controls.last_mut() {
            Some(last) if last.end == start => last.end = self.text.len(),
            _ => self.controls.push(start..self.text.len()),
        }
    }
}

/// The context of one rendering: the request's state and front images,
/// every question of the request (with its instructions), and the questions
/// of this rendering's `set`, each listing the `codes` its slots read;
/// `probe` is false (see [`PromptTemplate::probe`]).
pub(crate) fn context(
    state: &Value,
    front: &[usize],
    questions: &[(&Asked<'_, '_>, &Value)],
    set: &[usize],
    escape: &Escape,
) -> Map<String, Json> {
    let all: Vec<Json> = questions
        .iter()
        .map(|(asked, instructions)| question(asked, instructions, escape))
        .collect();
    let mut context = Map::new();
    context.insert("probe".into(), json!(false));
    context.insert("state".into(), content(state, escape));
    context.insert("images".into(), json!(front));
    context.insert(
        "questions".into(),
        Json::Array(set.iter().map(|question| all[*question].clone()).collect()),
    );
    context.insert("request_questions".into(), Json::Array(all));
    context
}

fn question(asked: &Asked<'_, '_>, instructions: &Value, escape: &Escape) -> Json {
    let mut object = Map::new();
    let unit = |unit: Option<&str>| content(&unit.map_or(Value::Null, Value::string), escape);
    match &asked.kind {
        Kind::Noul { truthy, falsy, .. } => {
            object.insert("truthy".into(), content(truthy, escape));
            object.insert("falsy".into(), content(falsy, escape));
        }
        Kind::Choice { options, .. } | Kind::Multiple { options, .. } => {
            let listed = options
                .iter()
                .map(|option| {
                    json!({
                        "key": escape.apply(&option.key),
                        "label": content(&Value::string(option.key.as_str()), escape),
                        "description": content(&option.description, escape),
                    })
                })
                .collect();
            object.insert("options".into(), Json::Array(listed));
        }
        Kind::Score { levels, .. } => {
            let listed = levels.iter().map(|level| content(level, escape)).collect();
            object.insert("levels".into(), Json::Array(listed));
        }
        Kind::Read { value, .. } => {
            let (name, written) = match value {
                ReadValue::Text => ("text", None),
                ReadValue::Number { unit } => ("number", unit.as_deref()),
                ReadValue::Date => ("date", None),
            };
            object.insert("value".into(), json!(name));
            object.insert("unit".into(), unit(written));
        }
        Kind::Estimate {
            grid,
            unit: written,
            numbers,
            ..
        } => {
            object.insert("unit".into(), unit(*written));
            object.insert(
                "grid".into(),
                json!({
                    "low": python_general(grid.low()),
                    "high": python_general(grid.high()),
                    "step": python_general(grid.step()),
                    "size": grid.size(),
                    "numbers": numbers,
                }),
            );
        }
        Kind::Box { kind } => {
            object.insert(
                "grid".into(),
                json!({"size": asked.codes.len(), "numbers": kind.numbers}),
            );
        }
        Kind::Boxes { blocks, kind } => {
            object.insert(
                "grid".into(),
                json!({"size": asked.codes.len(), "numbers": kind.numbers}),
            );
            object.insert("max".into(), json!(blocks));
            object.insert("most".into(), json!(kind.max));
        }
    }
    object.insert("kind".into(), json!(asked.name()));
    object.insert("instructions".into(), content(instructions, escape));
    object.insert("codes".into(), json!(asked.codes.codes()));
    Json::Object(object)
}

/// A request value's forms (see the module documentation).
fn content(value: &Value, escape: &Escape) -> Json {
    let form = |style: Style| {
        let mut parts = Parts::new(escape);
        write_json(value, style, &mut parts);
        parts.finish()
    };
    let mut object = Map::new();
    object.insert(
        "kind".into(),
        json!(match value {
            Value::Null => "none",
            Value::String(_) => "string",
            _ => "json",
        }),
    );
    object.insert("truthy".into(), json!(value.is_truthy()));
    object.insert("line".into(), form(Style::Line));
    object.insert("indented".into(), form(Style::Indented));
    if let Value::String(text) = value {
        object.insert("text".into(), segments(text, escape));
        object.insert("stripped".into(), segments(&stripped(text), escape));
    }
    Json::Object(object)
}

fn segments(text: &Text, escape: &Escape) -> Json {
    let mut parts = Parts::new(escape);
    for segment in text.segments() {
        match segment {
            Segment::Literal(literal) => parts.text(literal),
            Segment::Image(image) => parts.image(*image, false),
        }
    }
    parts.finish()
}

/// Python's `str.strip()`; an image is no whitespace.
fn stripped(text: &Text) -> Text {
    let mut segments = text.segments().to_vec();
    if let Some(Segment::Literal(first)) = segments.first_mut() {
        *first = first.trim_start_matches(python_space).to_owned();
    }
    if let Some(Segment::Literal(last)) = segments.last_mut() {
        *last = last.trim_end_matches(python_space).to_owned();
    }
    Text::new(segments)
}

/// A form's parts: each run of request text between images escaped whole,
/// as the references escape a value's rendered text.
struct Parts<'e> {
    escape: &'e Escape,
    parts: Vec<Json>,
    pending: String,
}

impl<'e> Parts<'e> {
    fn new(escape: &'e Escape) -> Self {
        Self {
            escape,
            parts: Vec::new(),
            pending: String::new(),
        }
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.parts.push(Json::String(self.escape.apply(&pending)));
        }
    }

    fn finish(mut self) -> Json {
        self.flush();
        Json::Array(self.parts)
    }
}

impl Sink for Parts<'_> {
    fn text(&mut self, text: &str) {
        self.pending.push_str(text);
    }
    fn image(&mut self, image: usize, whole: bool) {
        self.flush();
        self.parts.push(json!({"image": image, "whole": whole}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(text: &str, controls: Vec<Range<usize>>) -> RenderedPrompt {
        RenderedPrompt {
            prompt: text.into(),
            controls,
        }
    }

    #[test]
    fn anchors_in_template_text_are_removed_and_read() {
        let text = "<a>\u{FDD0}image:1\u{FDD1}x\u{FDD0}state\u{FDD1}<q>y\u{FDD0}slot:0\u{FDD1}";
        // `x` and `y` are request text.
        let x = text.find('x').unwrap();
        let y = text.find('y').unwrap();
        let anchored = anchored(
            prompt(text, vec![0..x, x + 1..y, y + 1..text.len()]),
            1,
            2,
            Some("<img>"),
            &Value::Null,
            &Escape::None,
        )
        .unwrap();
        assert_eq!(anchored.rendered.text, "<a><img>x<q>y");
        assert_eq!(anchored.rendered.controls, vec![0..8, 9..12]);
        assert_eq!(anchored.rendered.images, vec![1]);
        assert_eq!(anchored.starts, vec![9]);
        assert_eq!(anchored.slots, vec![vec![13]]);
    }

    #[test]
    fn questions_start_at_the_state_and_their_anchors() {
        let text = "s\u{FDD0}state\u{FDD1}a\u{FDD0}question:1\u{FDD1}b";
        let read = anchored(
            prompt(text, vec![0..text.len()]),
            2,
            0,
            None,
            &Value::Null,
            &Escape::None,
        )
        .unwrap();
        assert_eq!(read.rendered.text, "sab");
        assert_eq!(read.starts, vec![1, 2]);
        for text in [
            "\u{FDD0}state\u{FDD1}",
            "a\u{FDD0}question:1\u{FDD1}b\u{FDD0}state\u{FDD1}",
            "\u{FDD0}state\u{FDD1}\u{FDD0}question:0\u{FDD1}\u{FDD0}question:1\u{FDD1}",
        ] {
            assert!(
                anchored(
                    prompt(text, vec![0..text.len()]),
                    2,
                    0,
                    None,
                    &Value::Null,
                    &Escape::None
                )
                .is_err(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn anchor_spellings_in_request_text_stay_text() {
        let text = "\u{FDD0}state\u{FDD1}\u{FDD0}slot:0\u{FDD1}";
        let state = "\u{FDD0}state\u{FDD1}".len();
        let anchored = anchored(
            prompt(text, vec![0..state]),
            1,
            0,
            None,
            &Value::Null,
            &Escape::None,
        )
        .unwrap();
        assert_eq!(anchored.rendered.text, "\u{FDD0}slot:0\u{FDD1}");
        assert!(anchored.rendered.controls.is_empty());
        assert_eq!(anchored.slots, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn malformed_anchors_are_refused() {
        let all = |text: &str| prompt(text, vec![0..text.len()]);
        for text in [
            "no state",
            "\u{FDD0}state\u{FDD1}\u{FDD0}state\u{FDD1}",
            "\u{FDD0}state\u{FDD1}\u{FDD0}slot:1\u{FDD1}",
            "\u{FDD0}state\u{FDD1}\u{FDD0}slot:00\u{FDD1}",
            "\u{FDD0}state\u{FDD1}\u{FDD0}image:0\u{FDD1}",
            "\u{FDD0}state",
            "state\u{FDD1}",
        ] {
            assert!(
                anchored(all(text), 1, 1, None, &Value::Null, &Escape::None).is_err(),
                "{text:?}"
            );
        }
        // An anchor that holds request text is not the template's.
        let split = "\u{FDD0}st\u{FDD1}";
        assert!(anchored(
            prompt(split, vec![0..3]),
            1,
            0,
            None,
            &Value::Null,
            &Escape::None
        )
        .is_err());
    }

    #[test]
    fn content_forms_follow_python() {
        let escape = Escape::None;
        let value = Value::String(Text::new([
            Segment::Literal(" \u{1c}a ".into()),
            Segment::Image(0),
            Segment::Literal("b \n".into()),
        ]));
        let forms = content(&value, &escape);
        assert_eq!(forms["kind"], "string");
        assert_eq!(
            forms["text"],
            json!([" \u{1c}a ", {"image": 0, "whole": false}, "b \n"])
        );
        assert_eq!(
            forms["stripped"],
            json!(["a ", {"image": 0, "whole": false}, "b"])
        );
        assert_eq!(
            forms["line"],
            json!(["\" \\u001ca ", {"image": 0, "whole": false}, "b \\n\""])
        );
        let object = Value::Object(vec![("k".into(), Value::Array(vec![Value::Null]))]);
        let forms = content(&object, &escape);
        assert_eq!(forms["kind"], "json");
        assert_eq!(forms["line"], json!(["{\"k\": [null]}"]));
        assert_eq!(
            forms["indented"],
            json!(["{\n  \"k\": [\n    null\n  ]\n}"])
        );
        assert!(forms.get("text").is_none());
        assert_eq!(forms["truthy"], true);
        assert_eq!(content(&Value::Null, &escape)["kind"], "none");
        assert_eq!(content(&Value::Array(Vec::new()), &escape)["truthy"], false);
    }
}
