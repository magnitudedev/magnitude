//! Written values: numbers, dates and units read strictly from the text a
//! read's span writes, in the context of the state's text, as LiquidAI
//! d1-2B's `parsing.py` reads them; no arithmetic, unit conversion or
//! invented date components. Patterns are the reference's,
//! with Python's `re` semantics (`fullmatch` anchored at both ends). A
//! pattern that passes the matcher's backtrack limit (Python's has none)
//! fails the reading rather than reading nothing.

use crate::answers::WrittenNumber;
use crate::json::{python_space, python_strip};
use fancy_regex::{Captures, Regex};
use std::collections::BTreeSet;
use std::sync::LazyLock;

fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("the reference's patterns compile")
}

/// `re.fullmatch`.
fn full(pattern: &str) -> Regex {
    regex(&format!(r"\A(?:{pattern})\z"))
}

/// A reading's result, or why its matcher stopped.
type Read<T> = Result<T, String>;

fn stopped(error: fancy_regex::Error) -> String {
    format!("reading a written value stopped: {error}")
}

fn matches(pattern: &Regex, text: &str) -> Read<bool> {
    pattern.is_match(text).map_err(stopped)
}

fn captures<'t>(pattern: &Regex, text: &'t str) -> Read<Option<Captures<'t, str>>> {
    pattern.captures(text).map_err(stopped)
}

/// Every match's groups, left to right without overlapping.
fn every<'t>(pattern: &Regex, text: &'t str) -> Read<Vec<Captures<'t, str>>> {
    pattern
        .captures_iter(text)
        .map(|found| found.map_err(stopped))
        .collect()
}

/// Every match's text (`re.findall` of a pattern without groups).
fn found<'t>(pattern: &Regex, text: &'t str) -> Read<Vec<&'t str>> {
    pattern
        .find_iter(text)
        .map(|found| found.map(|matched| matched.as_str()).map_err(stopped))
        .collect()
}

/// `re.sub`.
fn replace(pattern: &Regex, text: &str, with: &str) -> Read<String> {
    pattern
        .try_replacen(text, 0, with)
        .map(|replaced| replaced.into_owned())
        .map_err(stopped)
}

fn group<'t>(found: &Captures<'t, str>, index: usize) -> Option<&'t str> {
    found.get(index).map(|matched| matched.as_str())
}

/// Characters, as Python's `len` counts them.
fn length(text: &str) -> usize {
    text.chars().count()
}

/// The decimal digits of `text` (Unicode `Nd`) as ASCII, as `int` and
/// `Decimal` read them: each `Nd` digit's value is its distance from the
/// zero that starts its run of ten.
fn ascii_digits(text: &str) -> Read<String> {
    static DIGIT: LazyLock<Regex> = LazyLock::new(|| full(r"\d"));
    let digit = |character: char| matches(&DIGIT, character.encode_utf8(&mut [0; 4]));
    text.chars()
        .map(|character| {
            if character.is_ascii_digit() || !digit(character)? {
                return Ok(character);
            }
            let mut value = 0;
            let mut below = character as u32;
            while let Some(previous) = below.checked_sub(1).and_then(char::from_u32) {
                if !digit(previous)? {
                    break;
                }
                value += 1;
                below -= 1;
            }
            Ok(char::from(b'0' + (value % 10) as u8))
        })
        .collect()
}

/// `int` of a run of decimal digits, saturated: a value past `i64` is no
/// date component.
fn integer(text: &str) -> Read<i64> {
    Ok(ascii_digits(text)?.parse().unwrap_or(i64::MAX))
}

/// Python's `str.casefold`.
fn casefold(text: &str) -> String {
    caseless::default_case_fold_str(text)
}

/// `_fold`: case folded, compatibility decomposed, combining marks dropped.
fn fold(text: &str) -> String {
    use unicode_normalization::char::canonical_combining_class;
    use unicode_normalization::UnicodeNormalization;
    casefold(text)
        .nfkd()
        .filter(|character| canonical_combining_class(*character) == 0)
        .collect()
}

/// Month names a date is written with (accents folded; Polish and Russian
/// in the genitive a date uses).
const MONTH_NAMES: [&str; 12] = [
    "january janvier januar enero gennaio janeiro januari stycznia styczen января",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
];

/// `MONTHS`: every name, its unambiguous three-letter abbreviations, and
/// `janv`, `fevr`, `juil`, `sept`.
static MONTHS: LazyLock<Vec<(String, u32)>> = LazyLock::new(|| {
    let mut months: Vec<(String, u32)> = Vec::new();
    let set = |months: &mut Vec<(String, u32)>, name: String, month: u32| match months
        .iter_mut()
        .find(|(known, _)| *known == name)
    {
        Some(entry) => entry.1 = month,
        None => months.push((name, month)),
    };
    for (index, names) in MONTH_NAMES.iter().enumerate() {
        for name in names.split_whitespace() {
            set(&mut months, name.to_owned(), index as u32 + 1);
        }
    }
    let mut abbreviations: Vec<(String, BTreeSet<u32>)> = Vec::new();
    for (name, month) in months.clone() {
        let short: String = name.chars().take(3).collect();
        match abbreviations.iter_mut().find(|(known, _)| *known == short) {
            Some((_, of)) => {
                of.insert(month);
            }
            None => abbreviations.push((short, BTreeSet::from([month]))),
        }
    }
    for (short, of) in abbreviations {
        if of.len() == 1 {
            set(&mut months, short, *of.first().expect("one month"));
        }
    }
    for (name, month) in [("janv", 1), ("fevr", 2), ("juil", 7), ("sept", 9)] {
        set(&mut months, name.to_owned(), month);
    }
    months
});

fn month(name: &str) -> Option<u32> {
    MONTHS
        .iter()
        .find(|(known, _)| known == name)
        .map(|(_, month)| *month)
}

const WEEKDAYS: &str = "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun \
lundi mardi mercredi jeudi vendredi samedi dimanche montag dienstag mittwoch donnerstag freitag samstag sonntag lunes \
martes miercoles jueves viernes sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica segunda terca \
quarta quinta sexta";

/// `sorted(names, key=len, reverse=True)`, escaped and joined.
fn alternation<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let mut names: Vec<&str> = names.into_iter().collect();
    names.sort_by_key(|name| std::cmp::Reverse(length(name)));
    names
        .iter()
        .map(|name| fancy_regex::escape(name).into_owned())
        .collect::<Vec<_>>()
        .join("|")
}

static CJK_DATE: LazyLock<Regex> =
    LazyLock::new(|| full(r"(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]"));
const NUMERIC_DATE: &str = r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})(?!\d)";
static NUMERIC: LazyLock<Regex> = LazyLock::new(|| regex(NUMERIC_DATE));
static NUMERIC_FULL: LazyLock<Regex> = LazyLock::new(|| full(NUMERIC_DATE));
static WORD_DATE: LazyLock<Regex> = LazyLock::new(|| {
    let names = alternation(MONTHS.iter().map(|(name, _)| name.as_str()));
    regex(&format!(
        r"\b(?:\d{{1,2}}\w*[ .]+(?:de\s+)?(?:{names})[ ,.]+(?:de\s+)?\d{{4}}|(?:{names})[ .]+\d{{1,2}}\w*[ ,.]+\d{{4}})\b"
    ))
});

/// The decimal mark a currency's amounts are written with: `.`, `,`, none
/// (`""`, whole units), or both in use (`None`).
const CURRENCIES: [(Option<&str>, &[&str]); 4] = [
    (
        Some("."),
        &[
            "$", "£", "₹", "USD", "GBP", "CAD", "AUD", "NZD", "INR", "Rs.", "Rs", "MXN", "CNY",
            "RMB", "元", "HKD", "SGD",
        ],
    ),
    (
        Some(","),
        &[
            "R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY",
        ],
    ),
    (
        Some(""),
        &[
            "¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP",
        ],
    ),
    (None, &["€", "EUR", "CHF"]),
];

/// A matched currency's spelling and decimal mark, by its case fold
/// (`_NAMES`, `_MARK`).
fn currency(matched: &str) -> Option<(&'static str, Option<&'static str>)> {
    let folded = casefold(matched);
    CURRENCIES
        .iter()
        .rev()
        .flat_map(|(mark, names)| names.iter().map(move |name| (*name, *mark)))
        .find(|(name, _)| casefold(name) == folded)
}

/// `_CURRENCY`: an alternation, letters bounded by words.
fn currency_pattern() -> String {
    let mut names: Vec<&str> = CURRENCIES
        .iter()
        .flat_map(|(_, names)| names.iter().copied())
        .collect();
    names.sort_by_key(|name| std::cmp::Reverse(length(name)));
    let alternatives: Vec<String> = names
        .iter()
        .map(|name| {
            let first = name.chars().next().expect("a currency is spelled");
            let escaped = fancy_regex::escape(name);
            if first.is_ascii_alphabetic() {
                format!(r"\b{escaped}(?![a-z])")
            } else {
                escaped.into_owned()
            }
        })
        .collect();
    format!("(?i:{})", alternatives.join("|"))
}

static CURRENCY: LazyLock<Regex> = LazyLock::new(|| regex(&currency_pattern()));
static CURRENCY_FULL: LazyLock<Regex> = LazyLock::new(|| full(&currency_pattern()));
static CURRENCY_AROUND: LazyLock<Regex> = LazyLock::new(|| {
    let currency = currency_pattern();
    full(&format!(r"({currency})\s*(.*)|(.*?)\s*({currency})"))
});

/// `SCALES`: scale words, written apart or attached.
const SCALES: [(&str, u32); 35] = [
    ("thousand", 3),
    ("mil", 3),
    ("tausend", 3),
    ("тыс", 3),
    ("千", 3),
    ("万", 4),
    ("萬", 4),
    ("lakh", 5),
    ("lakhs", 5),
    ("crore", 7),
    ("crores", 7),
    ("million", 6),
    ("millions", 6),
    ("mio", 6),
    ("mln", 6),
    ("millón", 6),
    ("millones", 6),
    ("milhão", 6),
    ("milhões", 6),
    ("milione", 6),
    ("milioni", 6),
    ("млн", 6),
    ("亿", 8),
    ("億", 8),
    ("billion", 9),
    ("billions", 9),
    ("milliard", 9),
    ("milliards", 9),
    ("milliarde", 9),
    ("milliarden", 9),
    ("mrd", 9),
    ("miliardo", 9),
    ("miliardi", 9),
    ("млрд", 9),
    ("trillion", 12),
];

static SCALE: LazyLock<Regex> = LazyLock::new(|| {
    let names = alternation(SCALES.iter().map(|(name, _)| *name));
    regex(&format!(r"(?i)\s*({names})\.?\z"))
});
static SUFFIX: LazyLock<Regex> = LazyLock::new(|| regex(r"(?i)(?<=\d)(k|m|bn|b)\z"));
static COUNTED: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?i)\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|users|downloads)\b",
    )
});
static GROUPED: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"(?<![\d.,])\d{1,3}(?:([,. \x{a0}\x{202f}'’])\d{3})+(?!\1)([.,])\d+(?![\d.,])")
});
static SHORT: LazyLock<Regex> = LazyLock::new(|| regex(r"(?<![\d.,])\d+([.,])\d{1,2}(?![\d.,])"));
static REPEATED: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?<![\d.,])\d{1,3}(?:([.,])\d{3}){2,}(?![\d.,])"));
static LOCALE: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r#"\b(?:locale|language|lang)["']?\s*[=:]\s*["']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\b"#,
    )
});
static DECLARATION: LazyLock<Regex> = LazyLock::new(|| regex(r"<\?[^>]*\?>"));

/// `_one`: the one value evidence settles on.
fn one<T: Ord + Clone>(values: &BTreeSet<T>) -> Option<T> {
    (values.len() == 1).then(|| values.first().expect("one value").clone())
}

/// `written_unit`: only an explicitly written percent or currency marker.
pub(crate) fn written_unit(text: &str) -> Read<Option<String>> {
    if text.trim_end_matches(python_space).ends_with('%') {
        return Ok(Some("%".into()));
    }
    Ok(match found(&CURRENCY, text)?.as_slice() {
        [marker] => currency(marker).map(|(name, _)| name.to_owned()),
        _ => None,
    })
}

/// `_currency_marks`: the decimal marks of the currencies `text` writes; a
/// currency of both marks says nothing.
fn currency_marks(text: &str) -> Read<BTreeSet<&'static str>> {
    Ok(found(&CURRENCY, text)?
        .into_iter()
        .filter_map(|matched| currency(matched).and_then(|(_, mark)| mark))
        .collect())
}

/// `_decimal_mark`: from the first evidence that speaks: the span's own
/// currency, the document's other numbers, its currencies, a declared
/// locale.
fn decimal_mark(written: &str, context: &str) -> Read<Option<&'static str>> {
    let context = replace(&DECLARATION, context, "")?;
    let mark = |text: &str| -> &'static str {
        if text == "." {
            "."
        } else {
            ","
        }
    };
    let mut numbers: BTreeSet<&'static str> = every(&GROUPED, &context)?
        .iter()
        .map(|found| mark(group(found, 2).expect("group 2 takes part")))
        .collect();
    numbers.extend(
        every(&SHORT, &context)?
            .iter()
            .map(|found| mark(group(found, 1).expect("group 1 takes part"))),
    );
    numbers.extend(every(&REPEATED, &context)?.iter().map(|found| {
        if group(found, 1) == Some(".") {
            ","
        } else {
            "."
        }
    }));
    let folded = casefold(&context);
    let locales: BTreeSet<&'static str> = every(&LOCALE, &folded)?
        .iter()
        .map(|found| {
            if matches!(group(found, 1), Some("en" | "ja" | "zh" | "ko")) {
                "."
            } else {
                ","
            }
        })
        .collect();
    Ok([
        currency_marks(written)?,
        numbers,
        currency_marks(&context)?,
        locales,
    ]
    .into_iter()
    .find(|marks| !marks.is_empty())
    .and_then(|marks| one(&marks)))
}

/// `_digits`: the number as plain digits and a point, the decimal mark from
/// the evidence, the other mark a complete grouping.
fn digits(text: &str, written: &str, context: &str) -> Read<Option<String>> {
    static SHAPE: LazyLock<Regex> = LazyLock::new(|| full(r"\d[\d., '’]*"));
    static SPACED: LazyLock<Regex> = LazyLock::new(|| regex(r"[ '’]"));
    static GROUPING: LazyLock<Regex> =
        LazyLock::new(|| full(r"(\d{1,3}(?:[ '’]\d{3})+)([.,]\d+)?"));
    static LAKHS: LazyLock<Regex> = LazyLock::new(|| full(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?"));
    static RUPEES: LazyLock<Regex> = LazyLock::new(|| regex(r"(?i)₹|\bINR\b|\bRs\b|lakh|crore"));
    // Groups of the other mark, then the point.
    static POINT: LazyLock<Regex> = LazyLock::new(|| full(r"\d{1,3}(?:,\d{3})+\.\d+"));
    static COMMA: LazyLock<Regex> = LazyLock::new(|| full(r"\d{1,3}(?:\.\d{3})+,\d+"));
    if !matches(&SHAPE, text)? {
        return Ok(None);
    }
    if matches(&SPACED, text)? {
        let Some(grouping) = captures(&GROUPING, text)? else {
            return Ok(None);
        };
        let whole = replace(
            &SPACED,
            group(&grouping, 1).expect("group 1 takes part"),
            "",
        )?;
        return digits(
            &format!("{whole}{}", group(&grouping, 2).unwrap_or("")),
            written,
            context,
        );
    }
    if matches(&LAKHS, text)? && matches(&RUPEES, &format!("{written}\n{context}"))? {
        return Ok(Some(text.replace(',', "")));
    }
    if text.contains('.') && text.contains(',') {
        let (point, separator, grouped) = if text.rfind('.') > text.rfind(',') {
            (".", ",", &POINT)
        } else {
            (",", ".", &COMMA)
        };
        if !matches(grouped, text)? {
            return Ok(None);
        }
        return Ok(Some(text.replace(separator, "").replace(point, ".")));
    }
    if !text.contains('.') && !text.contains(',') {
        return Ok(Some(text.to_owned()));
    }
    let mark = if text.contains('.') { "." } else { "," };
    let parts: Vec<&str> = text.split(mark).collect();
    if parts.iter().any(|part| part.is_empty()) {
        return Ok(None);
    }
    let point = decimal_mark(written, context)?;
    if parts.len() > 2 || point.is_some_and(|point| point != mark) {
        let whole =
            (1..=3).contains(&length(parts[0])) && parts[1..].iter().all(|part| length(part) == 3);
        return Ok(whole.then(|| parts.concat()));
    }
    if point.is_none() && length(parts[1]) == 3 && length(parts[0]) <= 3 {
        return Ok(None);
    }
    Ok(Some(parts.join(".")))
}

/// `Decimal(text) * 10**shift` in Python's default context (28 significant
/// digits, half even): an integral value exactly, else the nearest float.
fn decimal(negative: bool, digits: &str, shift: u32) -> Read<Option<WrittenNumber>> {
    const PRECISION: usize = 28;
    let digits = ascii_digits(digits)?;
    let (whole, fraction) = digits.split_once('.').unwrap_or((&digits, ""));
    let mut coefficient: Vec<u8> = format!("{whole}{fraction}").into_bytes();
    let mut exponent = shift as i64 - fraction.len() as i64;
    let leading = coefficient
        .iter()
        .take_while(|digit| **digit == b'0')
        .count()
        .min(coefficient.len().saturating_sub(1));
    coefficient.drain(..leading);
    if coefficient.len() > PRECISION {
        let dropped = coefficient.split_off(PRECISION);
        exponent += dropped.len() as i64;
        let half = dropped[0].cmp(&b'5').then_with(|| {
            if dropped[1..].iter().any(|digit| *digit != b'0') {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let odd = coefficient[PRECISION - 1] % 2 == 1;
        if half.is_gt() || (half.is_eq() && odd) {
            let mut at = PRECISION;
            loop {
                if at == 0 {
                    coefficient.insert(0, b'1');
                    break;
                }
                at -= 1;
                if coefficient[at] == b'9' {
                    coefficient[at] = b'0';
                } else {
                    coefficient[at] += 1;
                    break;
                }
            }
            if coefficient.len() > PRECISION {
                coefficient.pop();
                exponent += 1;
            }
        }
    }
    let coefficient = String::from_utf8(coefficient).expect("ASCII digits");
    let zero = coefficient.bytes().all(|digit| digit == b'0');
    let sign = if negative && !zero { "-" } else { "" };
    if exponent >= 0 {
        let value = if zero {
            "0".to_owned()
        } else {
            format!("{coefficient}{}", "0".repeat(exponent as usize))
        };
        return Ok(Some(WrittenNumber::Integer(format!("{sign}{value}"))));
    }
    let point = coefficient.len() as i64 + exponent;
    let (integral, fractional) = if point > 0 {
        coefficient.split_at(point as usize)
    } else {
        ("", coefficient.as_str())
    };
    let padding = "0".repeat((-point).max(0) as usize);
    if fractional.bytes().all(|digit| digit == b'0') {
        let integral = integral.trim_start_matches('0');
        let integral = if integral.is_empty() { "0" } else { integral };
        let sign = if integral == "0" { "" } else { sign };
        return Ok(Some(WrittenNumber::Integer(format!("{sign}{integral}"))));
    }
    let value: f64 = format!(
        "{}{}.{padding}{fractional}",
        if negative { "-" } else { "" },
        if integral.is_empty() { "0" } else { integral }
    )
    .parse()
    .expect("a decimal parses");
    Ok(value.is_finite().then_some(WrittenNumber::Float(value)))
}

/// `parse_number`: the written magnitude, percent points included, its
/// currency and percent sign set apart, its scale word applied. `None` when
/// the writing does not settle one value.
pub(crate) fn parse_number(
    text: &str,
    context: &str,
    unit: Option<&str>,
) -> Read<Option<WrittenNumber>> {
    let mut context = context.to_owned();
    if let Some(unit) = unit.filter(|unit| !unit.is_empty()) {
        if matches(&CURRENCY_FULL, python_strip(unit))? {
            context.push('\n');
            context.push_str(unit);
        }
    }
    let written = text;
    let cleaned = python_strip(text)
        .replace('\u{2212}', "-")
        .replace('\u{a0}', " ")
        .replace('\u{202f}', " ");
    let cleaned = cleaned.strip_suffix('%').unwrap_or(&cleaned);
    let mut text = cleaned.trim_end_matches(python_space).to_owned();
    let negative = text.starts_with('(') && text.ends_with(')');
    if negative {
        text = python_strip(&text[1..text.len() - 1]).to_owned();
    }
    let (mut sign, mut currency): (Option<char>, Option<()>) = (None, None);
    for _ in 0..2 {
        if currency.is_none() {
            if let Some(found) = captures(&CURRENCY_AROUND, &text)? {
                currency = Some(());
                let rest = if group(&found, 1).is_some() {
                    group(&found, 2)
                } else {
                    group(&found, 3)
                };
                text = python_strip(rest.unwrap_or("")).to_owned();
            }
        }
        if sign.is_none() {
            if let Some(first @ ('-' | '+')) = text.chars().next() {
                sign = Some(first);
                text = python_strip(&text[1..]).to_owned();
            }
        }
    }
    if negative && sign.is_some() {
        return Ok(None);
    }
    let mut shift = Some(0);
    if let Some(word) = captures(&SCALE, &text)? {
        let name = group(&word, 1).expect("group 1 takes part");
        // German `Billion` is 10^12: no value. A case variant whose lower
        // case is no scale word (the reference's `KeyError`) is none either.
        let lower = name.to_lowercase();
        shift = SCALES
            .iter()
            .find(|(scale, _)| name != "Billion" && *scale == lower)
            .map(|(_, shift)| *shift);
        text.truncate(word.get(0).expect("the whole match").start());
    } else if let Some(suffix) = captures(&SUFFIX, &text)? {
        let money = matches(&CURRENCY, &format!("{written}\n{context}"))?;
        if !(money || matches(&COUNTED, &context)?) {
            return Ok(None);
        }
        shift = match group(&suffix, 1)
            .expect("group 1 takes part")
            .to_lowercase()
            .as_str()
        {
            "k" => Some(3),
            "m" => Some(6),
            "b" | "bn" => Some(9),
            _ => None,
        };
        text.truncate(suffix.get(0).expect("the whole match").start());
    }
    match (digits(&text, written, &context)?, shift) {
        (Some(digits), Some(shift)) => decimal(negative || sign == Some('-'), &digits, shift),
        _ => Ok(None),
    }
}

/// A declared numeric date format, in English, French, German, Spanish or
/// Italian.
static DATE_FORMATS: LazyLock<[(Regex, &'static str); 2]> = LazyLock::new(|| {
    let day = r"(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)";
    let month = r"(?:mm?|month|mois|monat|mes|mese)";
    let year = r"(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)";
    [
        (
            regex(&format!(r"\b{day}\s*[/.-]\s*{month}\s*[/.-]\s*{year}\b")),
            "dmy",
        ),
        (
            regex(&format!(r"\b{month}\s*[/.-]\s*{day}\s*[/.-]\s*{year}\b")),
            "mdy",
        ),
    ]
});

/// `date_order`: a declared format or locale first, then the state's other
/// unambiguous numeric dates.
fn date_order(context: &str, folded: &str) -> Read<Option<&'static str>> {
    static LOCALES: LazyLock<Regex> = LazyLock::new(|| {
        regex(
            r#"\b(?:locale|language|lang|langue|sprache|idioma|lingua)["']?\s*[=:]\s*["']?(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b"#,
        )
    });
    let mut explicit = BTreeSet::new();
    for (pattern, order) in DATE_FORMATS.iter() {
        if matches(pattern, folded)? {
            explicit.insert(*order);
        }
    }
    if !explicit.is_empty() {
        return Ok(one(&explicit));
    }
    let locales: BTreeSet<&'static str> = every(&LOCALES, folded)?
        .iter()
        .map(|found| {
            let locale = group(found, 1)
                .expect("group 1 takes part")
                .replace('_', "-");
            if locale == "en" || locale == "en-us" {
                "mdy"
            } else {
                "dmy"
            }
        })
        .collect();
    if !locales.is_empty() {
        return Ok(one(&locales));
    }
    let mut orders = BTreeSet::new();
    for found in every(&NUMERIC, context)? {
        let fourth = group(&found, 4).expect("group 4 takes part");
        let (a, b, year) = (
            integer(group(&found, 1).expect("group 1 takes part"))?,
            integer(group(&found, 3).expect("group 3 takes part"))?,
            integer(fourth)?,
        );
        if length(fourth) != 4 || a > 31 || b > 31 || a < 1 || b < 1 {
            continue;
        }
        for (day, month, order) in [(a, b, "dmy"), (b, a, "mdy")] {
            if iso_date(year, month, day).is_some() && (a > 12) != (b > 12) {
                orders.insert(order);
            }
        }
    }
    Ok(one(&orders))
}

/// `date(year, month, day).isoformat()`, when the date exists.
fn iso_date(year: i64, month: i64, day: i64) -> Option<String> {
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    ((1..=9999).contains(&year) && (1..=days).contains(&day))
        .then(|| format!("{year:04}-{month:02}-{day:02}"))
}

/// `datetime.fromisoformat(date + "T" + clock).isoformat()` (Python 3.11)
/// for an ISO date and a clock `HH:MM[:SS[.f…]][Z|±HH:MM]` of ASCII digits.
fn iso_datetime(date: &str, clock: &str) -> Read<Option<String>> {
    static CLOCK: LazyLock<Regex> = LazyLock::new(|| {
        full(r"([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]+))?)?(Z|[+-][0-9]{2}:[0-9]{2})?")
    });
    static DATE: LazyLock<Regex> = LazyLock::new(|| full(r"[0-9]{4}-[0-9]{2}-[0-9]{2}"));
    if !matches(&DATE, date)? {
        return Ok(None);
    }
    let Some(found) = captures(&CLOCK, clock)? else {
        return Ok(None);
    };
    Ok(clocked(date, &found))
}

/// The ISO date and time of a matched clock, when it is a valid one.
fn clocked(date: &str, found: &Captures<'_, str>) -> Option<String> {
    let number =
        |index: usize| group(found, index).map_or(Some(0), |text| text.parse::<u32>().ok());
    let (hour, minute, second) = (number(1)?, number(2)?, number(3)?);
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let micro = group(found, 4).map(|fraction| {
        let kept: String = fraction.chars().take(6).collect();
        format!("{kept:0<6}")
    });
    let zone = match group(found, 5) {
        None => String::new(),
        Some("Z") => "+00:00".into(),
        Some(offset) => {
            let (hours, minutes): (u32, u32) =
                (offset[1..3].parse().ok()?, offset[4..6].parse().ok()?);
            if hours > 23 || minutes > 59 {
                return None;
            }
            if hours == 0 && minutes == 0 {
                "+00:00".into()
            } else {
                offset.to_owned()
            }
        }
    };
    let fraction = micro
        .filter(|micro| micro != "000000")
        .map_or(String::new(), |micro| format!(".{micro}"));
    Some(format!(
        "{date}T{hour:02}:{minute:02}:{second:02}{fraction}{zone}"
    ))
}

/// `parse_date`: the ISO value, `None` for an invalid or unrecognised span
/// or an unresolved ambiguity.
pub(crate) fn parse_date(text: &str, context: &str) -> Read<Option<String>> {
    let folded = fold(context);
    let order = std::cell::OnceCell::new();
    Ok(Dates {
        context,
        folded: &folded,
        order: &order,
    }
    .parse(text)?
    .0)
}

struct Dates<'a> {
    context: &'a str,
    folded: &'a str,
    order: &'a std::cell::OnceCell<Option<&'static str>>,
}

impl Dates<'_> {
    /// The context's date order, read once when a date needs it.
    fn order(&self) -> Read<Option<&'static str>> {
        if let Some(order) = self.order.get() {
            return Ok(*order);
        }
        let order = date_order(self.context, self.folded)?;
        Ok(*self.order.get_or_init(|| order))
    }

    /// `_candidates`: day first, month first, or the one the order says.
    fn candidates(&self, year: i64, a: i64, b: i64) -> Read<Vec<(i64, i64, i64)>> {
        let (dmy, mdy) = ((year, b, a), (year, a, b));
        Ok(match self.order()? {
            None => vec![dmy, mdy],
            Some("dmy") => vec![dmy],
            Some(_) => vec![mdy],
        })
    }

    /// `_parse_date`: the ISO value and whether an ambiguity was left
    /// unresolved.
    fn parse(&self, text: &str) -> Read<(Option<String>, bool)> {
        static TIMED: LazyLock<Regex> = LazyLock::new(|| {
            full(r"(.+?)[T\s]+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)")
        });
        static ORDINAL: LazyLock<Regex> = LazyLock::new(|| regex(r"(?<=\d)(?:st|nd|rd|th|er)\b"));
        static INDICATOR: LazyLock<Regex> = LazyLock::new(|| regex(r"(?<=\d)\.?(?:o|a|°)\b"));
        static ARTICLE: LazyLock<Regex> =
            LazyLock::new(|| regex(r"\b(?:de|of|the|le|el|il|den)\b"));
        static PUNCTUATION: LazyLock<Regex> = LazyLock::new(|| regex(r"[,.]"));
        static WHOLE: LazyLock<Regex> = LazyLock::new(|| full(r"\d+"));
        let text = python_strip(text);
        if text.is_empty() {
            return Ok((None, false));
        }
        if let Some(timed) = captures(&TIMED, text)? {
            let (date, ambiguous) = self.parse(group(&timed, 1).expect("group 1 takes part"))?;
            let Some(date) = date else {
                return Ok((None, ambiguous));
            };
            let mut clock = group(&timed, 2).expect("group 2 takes part").to_owned();
            if clock.chars().nth(1) == Some(':') {
                clock.insert(0, '0');
            }
            return Ok((iso_datetime(&date, &clock)?, false));
        }
        if let Some(numeric) = captures(&NUMERIC_FULL, text)? {
            let parts =
                [1, 3, 4].map(|index| group(&numeric, index).expect("the date's groups take part"));
            let [a, b, c] = [integer(parts[0])?, integer(parts[1])?, integer(parts[2])?];
            let candidates = if length(parts[0]) == 4 {
                vec![(a, b, c)]
            } else if length(parts[2]) == 4 {
                self.candidates(c, a, b)?
            } else {
                // A century is never invented: another unambiguous complete
                // date must establish it.
                if length(parts[2]) != 2 {
                    return Ok((None, false));
                }
                let mut complete: Vec<String> = every(&NUMERIC, self.context)?
                    .into_iter()
                    .filter(|found| {
                        length(group(found, 1).expect("group 1 takes part")) == 4
                            || length(group(found, 4).expect("group 4 takes part")) == 4
                    })
                    .map(|found| group(&found, 0).expect("the whole match").to_owned())
                    .collect();
                complete.extend(
                    every(&WORD_DATE, self.folded)?
                        .into_iter()
                        .map(|found| group(&found, 0).expect("the whole match").to_owned()),
                );
                let mut centuries = BTreeSet::new();
                for candidate in complete {
                    if let (Some(value), false) = self.parse(candidate.as_str())? {
                        let year: i64 = value[..4].parse().expect("an ISO year");
                        centuries.insert(year / 100 * 100);
                    }
                }
                let Some(century) = one(&centuries) else {
                    return Ok((None, false));
                };
                self.candidates(century + c, a, b)?
            };
            let values: BTreeSet<String> = candidates
                .into_iter()
                .filter_map(|(year, month, day)| iso_date(year, month, day))
                .collect();
            if values.len() > 1 {
                return Ok((None, true));
            }
            return Ok((values.into_iter().next(), false));
        }
        if let Some(cjk) = captures(&CJK_DATE, text)? {
            let part = |index| integer(group(&cjk, index).expect("the date's groups take part"));
            return Ok((iso_date(part(1)?, part(2)?, part(3)?), false));
        }
        let words = fold(text);
        let words = replace(&ORDINAL, &words, "")?;
        let words = replace(&INDICATOR, &words, "")?;
        let words = replace(&ARTICLE, &words, " ")?;
        let words = replace(&PUNCTUATION, &words, " ")?;
        let mut words: Vec<&str> = words
            .split(python_space)
            .filter(|word| !word.is_empty())
            .collect();
        if words.first().is_some_and(|first| {
            WEEKDAYS
                .split(' ')
                .any(|day| day == first.strip_suffix("-feira").unwrap_or(first))
        }) {
            words.remove(0);
        }
        let [first, second, third] = words.as_slice() else {
            return Ok((None, false));
        };
        let (day, month, year) = if let Some(month) = month(first) {
            (*second, month, *third)
        } else if let Some(month) = month(second) {
            (*first, month, *third)
        } else {
            return Ok((None, false));
        };
        if !matches(&WHOLE, day)? || !matches(&WHOLE, year)? || length(year) != 4 {
            return Ok((None, false));
        }
        Ok((
            iso_date(integer(year)?, i64::from(month), integer(day)?),
            false,
        ))
    }
}
