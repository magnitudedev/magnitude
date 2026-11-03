//! Request values as the references read them: JSON whose objects keep
//! their key order (a Python `dict`), whose strings may place images, and
//! whose serialization is Python's `json.dumps` with `ensure_ascii=False`
//! byte for byte, since every reference renders states and descriptions
//! with it.

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{self, Serialize, SerializeMap, SerializeSeq, Serializer};
use std::collections::hash_map::{Entry, HashMap};
use std::fmt::{self, Write as _};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(Number),
    String(Text),
    Array(Vec<Value>),
    /// Entries in order, keys distinct.
    Object(Vec<(String, Value)>),
}

/// A JSON number as Python's `json.loads` reads it: an integer literal is an
/// integer, anything else a float. An integer literal outside 64 bits is
/// read as a float, as `serde_json` reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    Integer(i64),
    Unsigned(u64),
    Float(f64),
}

/// A string: literal text with images placed in it.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Text {
    segments: Vec<Segment>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Segment {
    Literal(String),
    /// An image of [`crate::DecisionRequest::images`], by index.
    Image(usize),
}

impl Text {
    /// Adjacent literals are joined and empty ones dropped, so equal texts
    /// have equal segments.
    pub fn new(segments: impl IntoIterator<Item = Segment>) -> Self {
        let mut joined: Vec<Segment> = Vec::new();
        for segment in segments {
            match (joined.last_mut(), segment) {
                (_, Segment::Literal(text)) if text.is_empty() => {}
                (Some(Segment::Literal(last)), Segment::Literal(text)) => last.push_str(&text),
                (_, segment) => joined.push(segment),
            }
        }
        Self { segments: joined }
    }

    pub fn plain(text: impl Into<String>) -> Self {
        Self::new([Segment::Literal(text.into())])
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The text when it places no image.
    pub fn as_plain(&self) -> Option<&str> {
        match self.segments.as_slice() {
            [] => Some(""),
            [Segment::Literal(text)] => Some(text),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn images(&self) -> impl Iterator<Item = usize> + '_ {
        self.segments.iter().filter_map(|segment| match segment {
            Segment::Image(image) => Some(*image),
            Segment::Literal(_) => None,
        })
    }
}

impl Value {
    pub fn string(text: impl Into<String>) -> Self {
        Self::String(Text::plain(text))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// An object's value at `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// A string that places no image.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => text.as_plain(),
            _ => None,
        }
    }

    /// Levels of nested arrays and objects: 0 for a scalar.
    pub fn depth(&self) -> usize {
        match self {
            Self::Array(items) => 1 + items.iter().map(Value::depth).max().unwrap_or(0),
            Self::Object(entries) => {
                1 + entries
                    .iter()
                    .map(|(_, value)| value.depth())
                    .max()
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    /// Python truthiness, which a reference's `value or default` reads:
    /// null, false, zero and empty strings, arrays and objects are falsy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(value) => *value,
            Self::Number(Number::Integer(value)) => *value != 0,
            Self::Number(Number::Unsigned(value)) => *value != 0,
            Self::Number(Number::Float(value)) => *value != 0.0,
            Self::String(text) => !text.is_empty(),
            Self::Array(items) => !items.is_empty(),
            Self::Object(entries) => !entries.is_empty(),
        }
    }

    /// Every image the value places, in reading order.
    pub fn images(&self) -> Vec<usize> {
        let mut images = Vec::new();
        self.visit_images(&mut images);
        images
    }

    fn visit_images(&self, images: &mut Vec<usize>) {
        match self {
            Self::String(text) => images.extend(text.images()),
            Self::Array(items) => items.iter().for_each(|item| item.visit_images(images)),
            Self::Object(entries) => entries
                .iter()
                .for_each(|(_, value)| value.visit_images(images)),
            Self::Null | Self::Bool(_) | Self::Number(_) => {}
        }
    }
}

/// Where rendered text goes: literal text, and images in place.
pub trait Sink {
    fn text(&mut self, text: &str);
    /// An image in place; `whole` when it is a whole JSON string, written
    /// without its quotes.
    fn image(&mut self, image: usize, whole: bool);
}

/// `json.dumps` layouts the references use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// `json.dumps(value, ensure_ascii=False)`: `", "` and `": "`.
    Line,
    /// `json.dumps(value, ensure_ascii=False, indent=2)`.
    Indented,
}

/// Write `value` as `json.dumps` writes it. A string that is exactly one
/// image is that image, quotes included; an image inside a longer string
/// sits between its escaped text.
pub fn write_json(value: &Value, style: Style, sink: &mut dyn Sink) {
    write_at(value, style, 0, sink);
}

fn write_at(value: &Value, style: Style, depth: usize, sink: &mut dyn Sink) {
    match value {
        Value::Null => sink.text("null"),
        Value::Bool(true) => sink.text("true"),
        Value::Bool(false) => sink.text("false"),
        Value::Number(number) => sink.text(&number.to_string()),
        Value::String(text) => write_string(text, sink),
        Value::Array(items) if items.is_empty() => sink.text("[]"),
        Value::Object(entries) if entries.is_empty() => sink.text("{}"),
        Value::Array(items) => {
            sink.text("[");
            for (index, item) in items.iter().enumerate() {
                separator(style, depth + 1, index == 0, sink);
                write_at(item, style, depth + 1, sink);
            }
            close(style, depth, sink);
            sink.text("]");
        }
        Value::Object(entries) => {
            sink.text("{");
            for (index, (key, item)) in entries.iter().enumerate() {
                separator(style, depth + 1, index == 0, sink);
                sink.text(&quoted(key));
                sink.text(": ");
                write_at(item, style, depth + 1, sink);
            }
            close(style, depth, sink);
            sink.text("}");
        }
    }
}

fn separator(style: Style, depth: usize, first: bool, sink: &mut dyn Sink) {
    match (style, first) {
        (Style::Line, true) => {}
        (Style::Line, false) => sink.text(", "),
        (Style::Indented, first) => {
            sink.text(if first { "\n" } else { ",\n" });
            sink.text(&"  ".repeat(depth));
        }
    }
}

fn close(style: Style, depth: usize, sink: &mut dyn Sink) {
    if style == Style::Indented {
        sink.text("\n");
        sink.text(&"  ".repeat(depth));
    }
}

fn write_string(text: &Text, sink: &mut dyn Sink) {
    if let [Segment::Image(image)] = text.segments() {
        return sink.image(*image, true);
    }
    sink.text("\"");
    for segment in text.segments() {
        match segment {
            Segment::Literal(literal) => sink.text(&escaped(literal)),
            Segment::Image(image) => sink.image(*image, false),
        }
    }
    sink.text("\"");
}

/// A string as `json.dumps` quotes it with `ensure_ascii=False`.
pub fn quoted(text: &str) -> String {
    format!("\"{}\"", escaped(text))
}

fn escaped(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", control as u32);
            }
            other => out.push(other),
        }
    }
    out
}

impl fmt::Display for Number {
    /// Python's `str` of an `int` and `repr` of a `float`: the shortest
    /// round-trip digits, positional when the decimal exponent is in
    /// `[-4, 16)` (with a `.0` when integral), else scientific with a signed
    /// two-digit exponent.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Integer(value) => write!(formatter, "{value}"),
            Self::Unsigned(value) => write!(formatter, "{value}"),
            Self::Float(value) => formatter.write_str(&python_float(value)),
        }
    }
}

/// Python's `format(value, "g")` of a finite float: six significant
/// digits, positional when the exponent is in `[-4, 6)`, trailing zeros
/// and a bare point dropped.
pub fn python_general(value: f64) -> String {
    const PRECISION: i32 = 6;
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    // `{:.5e}` rounds to six significant digits as Python does (the exact
    // binary value, ties to even).
    let scientific = format!("{:.*e}", (PRECISION - 1) as usize, value);
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes an integer exponent");
    let trim = |digits: String| -> String {
        if digits.contains('.') {
            digits
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_owned()
        } else {
            digits
        }
    };
    if (-4..PRECISION).contains(&exponent) {
        trim(format!("{:.*}", (PRECISION - 1 - exponent) as usize, value))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{}e{sign}{:02}",
            trim(mantissa.to_owned()),
            exponent.unsigned_abs()
        )
    }
}

/// Python's `str.isspace()` for one character: Unicode whitespace and the
/// four information separators `\x1c`..`\x1f`.
pub(crate) fn python_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// Python's `str.strip()`.
pub(crate) fn python_strip(text: &str) -> &str {
    text.trim_matches(python_space)
}

/// `repr` of a finite Python float.
pub fn python_float(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into();
    }
    // `{:e}` writes the shortest round-trip digits as `d.ddde<exp>`.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes an integer exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if value < 0.0 { "-" } else { "" };
    let point = exponent + 1;
    if (-3..=16).contains(&point) {
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{sign}{first}{fraction}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        )
    }
}

impl<'de> Deserialize<'de> for Value {
    /// An object's repeated key keeps its first position and its last value,
    /// as `json.loads` builds a `dict`.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }
    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        Value::deserialize(deserializer)
    }
    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }
    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(Number::Integer(value)))
    }
    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(Number::Unsigned(value)))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        if !value.is_finite() {
            return Err(E::custom("JSON numbers are finite"));
        }
        Ok(Value::Number(Number::Float(value)))
    }
    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::string(value))
    }
    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::string(value))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = access.next_element()? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Value, A::Error> {
        let mut entries: Vec<(String, Value)> = Vec::new();
        let mut positions: HashMap<String, usize> = HashMap::new();
        while let Some((key, value)) = access.next_entry::<String, Value>()? {
            match positions.entry(key) {
                Entry::Occupied(position) => entries[*position.get()].1 = value,
                Entry::Vacant(position) => {
                    entries.push((position.key().clone(), value));
                    position.insert(entries.len() - 1);
                }
            }
        }
        Ok(Value::Object(entries))
    }
}

impl Serialize for Value {
    /// Objects in their order. A string that places an image has no JSON
    /// form: the request's wire form names it.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(Number::Integer(value)) => serializer.serialize_i64(*value),
            Self::Number(Number::Unsigned(value)) => serializer.serialize_u64(*value),
            Self::Number(Number::Float(value)) => serializer.serialize_f64(*value),
            Self::String(text) => match text.as_plain() {
                Some(text) => serializer.serialize_str(text),
                None => Err(ser::Error::custom(
                    "a string placing an image has no JSON form",
                )),
            },
            Self::Array(items) => {
                let mut sequence = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    sequence.serialize_element(item)?;
                }
                sequence.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    /// Text with each image written `<k>`.
    struct Record(String);
    impl Sink for Record {
        fn text(&mut self, text: &str) {
            self.0.push_str(text);
        }
        fn image(&mut self, image: usize, _: bool) {
            let _ = write!(self.0, "<{image}>");
        }
    }

    fn dumps(value: &Value, style: Style) -> String {
        let mut record = Record(String::new());
        write_json(value, style, &mut record);
        record.0
    }

    #[test]
    fn floats_print_as_python_repr() {
        for (value, repr) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789012345.67, "123456789012345.67"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1.2345e100, "1.2345e+100"),
            (-0.0, "-0.0"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(python_float(value), repr, "{value}");
        }
    }

    #[test]
    fn objects_keep_their_order_and_dumps_matches_python() {
        let value = parse(
            r#"{"b": [1, 2.0, {}], "a": "é\n\"\u0001", "c": {"d": null, "e": true}, "b": []}"#,
        );
        let line = dumps(&value, Style::Line);
        assert_eq!(
            line,
            r#"{"b": [], "a": "é\n\"\u0001", "c": {"d": null, "e": true}}"#
        );
        let value = parse(r#"{"k": [1, {"x": 2.5}], "e": []}"#);
        assert_eq!(
            dumps(&value, Style::Indented),
            "{\n  \"k\": [\n    1,\n    {\n      \"x\": 2.5\n    }\n  ],\n  \"e\": []\n}"
        );
    }

    #[test]
    fn a_whole_string_image_replaces_its_quotes() {
        let value = Value::Object(vec![
            ("a".into(), Value::String(Text::new([Segment::Image(0)]))),
            (
                "b".into(),
                Value::String(Text::new([
                    Segment::Literal("x".into()),
                    Segment::Image(1),
                    Segment::Literal("\"".into()),
                ])),
            ),
        ]);
        assert_eq!(dumps(&value, Style::Line), r#"{"a": <0>, "b": "x<1>\""}"#);
    }
}
