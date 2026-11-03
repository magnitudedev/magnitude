//! A decision model's description, typed: everything the engine does for a
//! decision model is selected and parameterized here, from its GGUF (see
//! `KEYS.md`). A description that parses is one the engine implements;
//! every missing key, key nothing reads and mechanism the engine lacks is
//! refused naming its key.

use crate::answers::through;
use crate::boxes::{BoxRead, Crossed, Drawn, Edge, Points, Positions};
use crate::metadata::{wrong, Keys};
use crate::plan::{Layout, QuestionAttends, StateAttends};
use crate::request::Placement;
use magnitude_artifacts::DecisionMetadata;
use std::num::NonZeroUsize;
use std::ops::RangeInclusive;
use std::sync::Arc;

/// The kinds a description may name, as its keys spell them.
const KINDS: [&str; 8] = [
    "noul", "choice", "multiple", "score", "read", "estimate", "box", "boxes",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Description {
    /// `tokenizer.chat_template.decision`: the prompt template.
    pub(crate) template: String,
    pub(crate) escape: EscapeMethod,
    pub(crate) segments: Segments,
    pub(crate) layout: Layout,
    /// Whose rows an option's value is read from: derived from whether the
    /// GGUF stores the readout head `decision.readout.weight`.
    pub(crate) readout: Matrix,
    pub(crate) codes: CodeSequence,
    pub(crate) kinds: Kinds,
    pub(crate) images: Images,
}

/// `escape.method`: how request text spelling a control token is changed
/// before it is rendered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EscapeMethod {
    None,
    /// `fullwidth`: `<` and `>` become `＜` and `＞` in each spelling.
    FullWidth(Spellings),
    /// `zero_width`: a zero-width space after each spelling's first
    /// character.
    ZeroWidth(Spellings),
}

/// The spellings an escape neutralizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Spellings {
    /// `escape.min_length`: the tokenizer's control tokens at least this
    /// many characters long.
    Control { min_length: usize },
    /// `escape.spellings`: these, as the reference lists them.
    Listed(Vec<String>),
}

/// `segments`: how a sequence's text is tokenized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segments {
    /// `whole`: as one text.
    Whole,
    /// `per_question`: the first question's whole prompt (the state through
    /// it) as one text, each further question's text alone; the state is
    /// the first prompt less the tokens of the first question's text alone.
    PerQuestion,
}

/// Whose rows an option's value is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    /// The LM head, an option's row its code's token: a GGUF without a
    /// readout head.
    Output,
    /// The readout head `decision.readout.weight` of `rows` rows, an
    /// option's row its code's index.
    Head { rows: usize },
}

/// The option codes the engine computes for the template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodeSequence {
    /// `codes.sequence = letters`: the first `count` of `A`..`Z`, `AA`.. that
    /// are each one token, distinct, and with `probe` (`codes.probe`) still
    /// that one token after the template's probe. `count` is the readout
    /// head's rows, or `codes.count` over the LM head.
    Letters { count: usize, probe: bool },
    /// `labels`: these codes, each one token, distinct.
    Labels(Vec<String>),
    /// `codes.sequence = aliases`: each option's positional letter code (or,
    /// with `keys_if_letters`, its key when every key is one letter), kept
    /// when it is one token not taken by an earlier option, else the first
    /// such code of the fallback pool; each code also read after a space
    /// when that is another single token.
    Aliases { keys_if_letters: bool },
}

/// `noul.codes`: what a noul's two options are read at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoulCodes {
    /// `sequence`: the model's first two option codes.
    Sequence,
    /// `yes_no`: `yes` and `no`, each in its lower, capitalized and upper
    /// spellings that are one token.
    YesNo,
}

/// `score.codes`: what a score's levels are read at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreCodes {
    /// `sequence`: the model's option codes.
    Sequence,
    /// `digits`: `0`, `1`, .. each one token.
    Digits,
}

/// `<kind>.grid` and `<kind>.numbers`: what a grid's values are read at.
#[derive(Clone, Debug, PartialEq)]
pub struct GridCodes {
    pub posed: Posed,
    /// `<kind>.numbers`: each value as the number it is, plain and after a
    /// space, when every value is an integer one token writes (a box's bin
    /// `k` as the number `k`); a grid that is not is read as `posed`.
    pub numbers: bool,
}

/// `<kind>.grid`: how a grid's values are posed when they are not read as
/// numbers.
#[derive(Clone, Debug, PartialEq)]
pub enum Posed {
    /// `codes`: as a choice's options, the model's codes by position.
    Codes,
    /// `levels`: as a score's levels when the model scores that many, else
    /// as `codes`.
    Levels(Arc<Score>),
}

/// `<kind>.slots`: where a question's values are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slots {
    /// `last`: the last token of the question's block.
    Last,
    /// `anchors`: the tokens the template ends with its slot anchors.
    Anchors,
}

/// `temperature.<kind>` and its buckets `temperature.<kind>.<bound>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Temperature {
    base: f64,
    /// `(bound, temperature)`, bounds ascending: a request of at most
    /// `bound` bucketed questions takes the first such temperature, any
    /// other `base`.
    buckets: Vec<(usize, f64)>,
}

impl Temperature {
    /// Whether a count picks it.
    pub fn bucketed(&self) -> bool {
        !self.buckets.is_empty()
    }

    /// The temperature at `questions` bucketed questions in the request:
    /// those whose kinds' temperatures are bucketed.
    pub fn at(&self, questions: usize) -> f64 {
        self.buckets
            .iter()
            .find(|(bound, _)| questions <= *bound)
            .map_or(self.base, |(_, temperature)| *temperature)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Noul {
    pub codes: NoulCodes,
    pub slots: Slots,
    /// `noul.true_option`: which of the template's two options is true.
    pub true_option: usize,
    pub temperature: Temperature,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// `choice.max_options`: the most options the reference reads.
    pub max_options: usize,
    pub slots: Slots,
    pub temperature: Temperature,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    /// `score.min_levels` and `score.max_levels`: the levels the reference
    /// reads.
    pub levels: RangeInclusive<usize>,
    pub codes: ScoreCodes,
    pub slots: Slots,
    pub temperature: Temperature,
}

/// `multiple.*`: each option of a multiple choice read as its `noul` (its
/// codes and true option), its P(true) through `calibration.multiple`. A
/// multiple choice is a `choice`: `choice.*` bounds its options.
#[derive(Clone, Debug, PartialEq)]
pub struct Multiple {
    pub noul: Arc<Noul>,
    pub choice: Arc<Choice>,
    pub table: Arc<Table>,
    pub temperature: Temperature,
}

/// `read.*`: a span of the state, its start and end slots scoring the
/// state's positions and position 0 (nothing).
#[derive(Clone, Debug, PartialEq)]
pub struct ReadKind {
    /// `read.<value>.max_tokens` and `calibration.found_<value>`, for
    /// text, number and date.
    pub values: [ReadOf; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReadOf {
    /// The longest span, in tokens.
    pub max_tokens: usize,
    /// P(found) through it.
    pub found: Arc<Table>,
}

/// `estimate.*`: a number over a grid of the request's range.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    /// `estimate.max_values`: the most values a grid reads; a finer step
    /// is read on this many.
    pub max_values: usize,
    pub grid: GridCodes,
    pub slots: Slots,
    /// `calibration.estimate`: the level each quantile is read at.
    pub levels: Arc<Table>,
    pub temperature: Temperature,
}

/// `box.*`: a box read from four edge slots, after a presence slot read as
/// a noul with `box.presence`.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxKind {
    pub edges: Edges,
    pub presence: Option<Presence>,
    /// `calibration.box`: the drawn boxes' share of hits, calibrated.
    pub confidence: Arc<Table>,
    pub temperature: Temperature,
}

/// A box's four edges: the pipeline's parameters and what their bins are
/// read at.
#[derive(Clone, Debug, PartialEq)]
pub struct Edges {
    pub read: BoxRead,
    pub grid: GridCodes,
}

/// A presence slot, read as its `noul`, P(present) through `found`
/// (`calibration.found_box` or `calibration.found_boxes`).
#[derive(Clone, Debug, PartialEq)]
pub struct Presence {
    pub noul: Arc<Noul>,
    pub found: Arc<Table>,
}

/// `boxes.*`: every instance, at most `max` blocks of a presence and four
/// edges, each read as `box.*` reads a box with its presence.
#[derive(Clone, Debug, PartialEq)]
pub struct Boxes {
    pub edges: Edges,
    /// `boxes.max`: the blocks read when a question names no `max`, and the
    /// most it may name.
    pub max: usize,
    /// Each block's presence, P(count > k) through `calibration.found_boxes`.
    pub presence: Presence,
    /// `calibration.boxes`: each box's confidence.
    pub confidence: Arc<Table>,
    pub temperature: Temperature,
}

/// `calibration.<name>.raw`, `.calibrated` and `.inverted`: a monotone
/// table, the calibrated value of a raw one by linear interpolation
/// through its knots; inverted, the raw value of a calibrated one.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    knots: Vec<[f64; 2]>,
}

impl Table {
    pub fn through(&self, value: f64) -> f64 {
        through(value, &self.knots)
    }

    /// A table of these `(raw, calibrated)` knots, read as a test writes
    /// them.
    #[cfg(test)]
    pub fn of(knots: Vec<[f64; 2]>) -> Self {
        Self { knots }
    }

    fn read(keys: &Keys<'_>, name: &str) -> Result<Self, String> {
        let side = |side: &str| -> Result<Vec<f64>, String> {
            let key = format!("calibration.{name}.{side}");
            let values = keys.floats(&key)?;
            Ok(values.to_vec())
        };
        let (mut raw, mut calibrated) = (side("raw")?, side("calibrated")?);
        let inverted = keys.flag(&format!("calibration.{name}.inverted"))?;
        if inverted {
            std::mem::swap(&mut raw, &mut calibrated);
        }
        if raw.is_empty()
            || raw.len() != calibrated.len()
            || raw.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(format!(
                "decision metadata `calibration.{name}` needs paired nonempty `.raw` and `.calibrated` knots, the side it is read from nondecreasing"
            ));
        }
        Ok(Self {
            knots: raw
                .into_iter()
                .zip(calibrated)
                .map(|(x, y)| [x, y])
                .collect(),
        })
    }
}

/// The kinds the model answers: those `kinds` lists.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Kinds {
    pub noul: Option<Arc<Noul>>,
    pub choice: Option<Arc<Choice>>,
    pub multiple: Option<Multiple>,
    pub score: Option<Arc<Score>>,
    pub read: Option<ReadKind>,
    pub estimate: Option<Estimate>,
    pub r#box: Option<BoxKind>,
    pub boxes: Option<Boxes>,
}

/// Where the model reads images and at most how many pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Images {
    /// `image_placements`; none when absent.
    pub placements: Vec<Placement>,
    /// `image_max_pixels`: a larger picture is downscaled to it before its
    /// processor; absent where the projector's own bound holds.
    pub max_pixels: Option<usize>,
}

impl Description {
    /// The description `metadata` gives, for a GGUF that stores a readout
    /// head of `head` rows, or none.
    pub fn parse(metadata: &DecisionMetadata, head: Option<usize>) -> Result<Self, String> {
        // Every implemented layout reads blocks both ways on global
        // attention layers, as the model was trained to.
        if metadata.causal() {
            return Err(
                "a decision model's global attention reads both ways, but `attention.causal` is true"
                    .into(),
            );
        }
        let keys = Keys::new(metadata);
        let spellings = || -> Result<Spellings, String> {
            Ok(match keys.optional("escape.spellings") {
                Some(_) => Spellings::Listed(keys.strings("escape.spellings")?.to_vec()),
                None => Spellings::Control {
                    min_length: keys.count("escape.min_length")?,
                },
            })
        };
        let escape = match keys.choose("escape.method", &["none", "fullwidth", "zero_width"])? {
            0 => EscapeMethod::None,
            1 => EscapeMethod::FullWidth(spellings()?),
            _ => EscapeMethod::ZeroWidth(spellings()?),
        };
        let segments = [Segments::Whole, Segments::PerQuestion]
            [keys.choose("segments", &["whole", "per_question"])?];
        let layout = layout(&keys)?;
        let readout = match head {
            Some(rows) => Matrix::Head { rows },
            None => Matrix::Output,
        };
        let codes = codes(&keys, readout)?;
        let kinds = kinds(&keys, readout)?;
        let images = Images {
            placements: match keys.optional("image_placements") {
                None => Vec::new(),
                Some(_) => keys
                    .strings("image_placements")?
                    .iter()
                    .map(|placement| placement_named(placement))
                    .collect::<Result<_, _>>()?,
            },
            max_pixels: match keys.optional("image_max_pixels") {
                None => None,
                Some(_) => Some(keys.count("image_max_pixels")?),
            },
        };
        keys.finish()?;
        Ok(Self {
            template: metadata.template().to_owned(),
            escape,
            segments,
            layout,
            readout,
            codes,
            kinds,
            images,
        })
    }

    /// Whether its plans read scores (`ReadoutKind::Scores`), which only a
    /// `read` kind's slots do.
    pub fn reads_scores(&self) -> bool {
        self.kinds.read.is_some()
    }

    /// Whether its plans read logits of chosen tokens (`ReadoutKind::Logits`):
    /// whether it reads codes over the LM head, which every kind's slots but
    /// a `read` kind's do.
    pub fn reads_logits(&self) -> bool {
        let Kinds {
            noul,
            choice,
            multiple,
            score,
            read: _,
            estimate,
            r#box,
            boxes,
        } = &self.kinds;
        self.readout == Matrix::Output
            && (noul.is_some()
                || choice.is_some()
                || multiple.is_some()
                || score.is_some()
                || estimate.is_some()
                || r#box.is_some()
                || boxes.is_some())
    }
}

fn placement_named(name: &str) -> Result<Placement, String> {
    [
        Placement::Front,
        Placement::State,
        Placement::Instructions,
        Placement::Options,
    ]
    .into_iter()
    .find(|placement| placement.as_str() == name)
    .ok_or_else(|| {
        wrong(
            "image_placements",
            "placements among `front`, `state`, `instructions` and `options`",
        )
    })
}

/// `set_size` (0: every question), `state_attends` and `question_attends`.
fn layout(keys: &Keys<'_>) -> Result<Layout, String> {
    Ok(Layout {
        set_size: NonZeroUsize::new(keys.integer("set_size")?),
        state_attends: [StateAttends::State, StateAttends::Set]
            [keys.choose("state_attends", &["state", "set"])?],
        question_attends: [QuestionAttends::Question, QuestionAttends::Set]
            [keys.choose("question_attends", &["question", "set"])?],
    })
}

fn codes(keys: &Keys<'_>, readout: Matrix) -> Result<CodeSequence, String> {
    if keys.optional("labels").is_some() {
        if keys.optional("codes.sequence").is_some() {
            return Err(
                "decision metadata `labels` lists the codes, so `codes.sequence` must be absent"
                    .into(),
            );
        }
        return Ok(CodeSequence::Labels(keys.strings("labels")?.to_vec()));
    }
    Ok(match keys.choose("codes.sequence", &["letters", "aliases"])? {
        0 => CodeSequence::Letters {
            count: match readout {
                Matrix::Head { rows } => rows,
                Matrix::Output => keys.count("codes.count")?,
            },
            probe: keys.flag("codes.probe")?,
        },
        _ => {
            // A question's codes follow its keys and read their spellings,
            // so they are tokens, not head rows.
            if readout != Matrix::Output {
                return Err(wrong(
                    "codes.sequence",
                    "`letters` for a GGUF with a readout head",
                ));
            }
            CodeSequence::Aliases {
                keys_if_letters: keys.flag("codes.keys_if_letters")?,
            }
        }
    })
}

fn kinds(keys: &Keys<'_>, readout: Matrix) -> Result<Kinds, String> {
    // `kinds`: the kinds the model was trained to answer, each once.
    let listed = keys.strings("kinds")?;
    if listed.is_empty()
        || listed.iter().any(|kind| !KINDS.contains(&kind.as_str()))
        || (1..listed.len()).any(|index| listed[..index].contains(&listed[index]))
    {
        return Err(wrong(
            "kinds",
            &format!("distinct kinds among {}, at least one", KINDS.join(", ")),
        ));
    }
    let described = |kind: &str| listed.iter().any(|listed| listed == kind);
    let slots = |kind: &str| -> Result<Slots, String> {
        Ok([Slots::Last, Slots::Anchors]
            [keys.choose(&format!("{kind}.slots"), &["last", "anchors"])?])
    };
    let temperature = |kind: &str| temperature(keys, kind);
    let table = |name: &str| Table::read(keys, name).map(Arc::new);
    // Codes other than the model's sequence are tokens, read at the LM head.
    let tokens = |key: &str, sequence: bool| {
        if sequence || readout == Matrix::Output {
            Ok(())
        } else {
            Err(wrong(key, "`sequence` for a GGUF with a readout head"))
        }
    };
    let noul = match described("noul") {
        false => None,
        true => {
            let true_option = keys.integer("noul.true_option")?;
            if true_option > 1 {
                return Err(wrong("noul.true_option", "0 or 1"));
            }
            let codes = [NoulCodes::Sequence, NoulCodes::YesNo]
                [keys.choose("noul.codes", &["sequence", "yes_no"])?];
            tokens("noul.codes", codes == NoulCodes::Sequence)?;
            Some(Arc::new(Noul {
                codes,
                slots: slots("noul")?,
                true_option,
                temperature: temperature("noul")?,
            }))
        }
    };
    let choice = match described("choice") {
        false => None,
        true => {
            Some(Arc::new(Choice {
                max_options: keys.count("choice.max_options")?,
                slots: slots("choice")?,
                temperature: temperature("choice")?,
            }))
        }
    };
    let score = match described("score") {
        false => None,
        true => {
            Some(Arc::new(Score {
                levels: {
                    let levels = keys.count("score.min_levels")?..=keys.count("score.max_levels")?;
                    if levels.is_empty() {
                        return Err(wrong("score.max_levels", "at least `score.min_levels`"));
                    }
                    levels
                },
                codes: {
                    let codes = [ScoreCodes::Sequence, ScoreCodes::Digits]
                        [keys.choose("score.codes", &["sequence", "digits"])?];
                    tokens("score.codes", codes == ScoreCodes::Sequence)?;
                    codes
                },
                slots: slots("score")?,
                temperature: temperature("score")?,
            }))
        }
    };
    let grid = |kind: &str| -> Result<GridCodes, String> {
        let key = format!("{kind}.grid");
        let posed = match keys.choose(&key, &["codes", "levels"])? {
            0 => Posed::Codes,
            _ => Posed::Levels(score.clone().ok_or_else(|| {
                wrong(&key, "`levels` only with the kind `score`, whose levels it reads")
            })?),
        };
        let key = format!("{kind}.numbers");
        let numbers = keys.flag(&key)?;
        tokens(&key, !numbers)?;
        Ok(GridCodes { posed, numbers })
    };
    let multiple = match described("multiple") {
        false => None,
        true => {
            let (Some(noul), Some(choice)) = (&noul, &choice) else {
                return Err(wrong("kinds", "`multiple` only with `noul` and `choice`, which read and bound its options"));
            };
            Some(Multiple {
                noul: noul.clone(),
                choice: choice.clone(),
                table: table("multiple")?,
                temperature: temperature("multiple")?,
            })
        }
    };
    let read = match described("read") {
        false => None,
        true => {
            let of = |value: &str| -> Result<ReadOf, String> {
                Ok(ReadOf {
                    max_tokens: keys.count(&format!("read.{value}.max_tokens"))?,
                    found: table(&format!("found_{value}"))?,
                })
            };
            Some(ReadKind {
                values: [of("text")?, of("number")?, of("date")?],
            })
        }
    };
    let estimate = match described("estimate") {
        false => None,
        true => {
            let max_values = keys.count("estimate.max_values")?;
            if max_values < 2 {
                return Err(wrong("estimate.max_values", "at least 2"));
            }
            Some(Estimate {
                max_values,
                grid: grid("estimate")?,
                slots: slots("estimate")?,
                levels: table("estimate")?,
                temperature: temperature("estimate")?,
            })
        }
    };
    let r#box = match described("box") {
        false => None,
        true => {
            let bins = keys.count("box.bins")?;
            if bins < 2 {
                return Err(wrong("box.bins", "at least 2"));
            }
            let iou = keys.positive("box.iou")?;
            if iou > 1.0 {
                return Err(wrong("box.iou", "at most 1"));
            }
            Some(BoxKind {
                edges: Edges {
                read: BoxRead {
                    bins,
                    positions: [Positions::Centres, Positions::Ends, Positions::Starts]
                        [keys.choose("box.positions", &["centres", "ends", "starts"])?],
                    edge: [Edge::Mean, Edge::Median]
                        [keys.choose("box.edge", &["mean", "median"])?],
                    points: [Points::R4, Points::Halton]
                        [keys.choose("box.points", &["r4", "halton"])?],
                    draws: keys.count("box.draws")?,
                    iou,
                    crossed: [Crossed::Clamp, Crossed::Shift]
                        [keys.choose("box.crossed", &["clamp", "shift"])?],
                    drawn: [Drawn::Valid, Drawn::Ordered]
                        [keys.choose("box.drawn", &["valid", "ordered"])?],
                },
                grid: grid("box")?,
                },
                presence: match (keys.flag("box.presence")?, &noul) {
                    (false, _) => None,
                    (true, Some(noul)) => Some(Presence {
                        noul: noul.clone(),
                        found: table("found_box")?,
                    }),
                    (true, None) => {
                        return Err(wrong(
                            "box.presence",
                            "true only with the kind `noul`, which reads it",
                        ))
                    }
                },
                confidence: table("box")?,
                temperature: temperature("box")?,
            })
        }
    };
    let boxes = match described("boxes") {
        false => None,
        true => {
            let Some((edges, Some(presence))) =
                r#box.as_ref().map(|kind| (&kind.edges, &kind.presence))
            else {
                return Err(wrong(
                    "kinds",
                    "`boxes` only with `box` and `box.presence`, which read its blocks",
                ));
            };
            Some(Boxes {
                edges: edges.clone(),
                max: keys.count("boxes.max")?,
                presence: Presence {
                    noul: presence.noul.clone(),
                    found: table("found_boxes")?,
                },
                confidence: table("boxes")?,
                temperature: temperature("boxes")?,
            })
        }
    };
    Ok(Kinds {
        noul,
        choice,
        multiple,
        score,
        read,
        estimate,
        r#box,
        boxes,
    })
}

/// `temperature.<kind>` and its buckets `temperature.<kind>.<bound>`.
fn temperature(keys: &Keys<'_>, kind: &str) -> Result<Temperature, String> {
    let base_key = format!("temperature.{kind}");
    let base = keys.positive(&base_key)?;
    let mut buckets = Vec::new();
    for key in keys.under(&format!("{base_key}.")) {
        let written = &key[base_key.len() + 1..];
        let bound = written
            .parse::<usize>()
            .ok()
            .filter(|bound| bound.to_string() == written)
            .ok_or_else(|| wrong(key, &format!("named `{base_key}.<count>`")))?;
        buckets.push((bound, keys.positive(key)?));
    }
    buckets.sort_by_key(|(bound, _)| *bound);
    Ok(Temperature { base, buckets })
}
