//! Typed decisions over prompt-only readouts.
//!
//! A decision model is described entirely by its GGUF: a decision template
//! (`tokenizer.chat_template.decision`) and the keys under
//! `decision.` (`KEYS.md`), read into a [`description`] that selects
//! and parameterizes each mechanism the engine implements once. The engine
//! has no per-model code: how a [`DecisionRequest`] renders ([`prompt`]),
//! which attention blocks its sequences use, which readouts it reads
//! ([`plan`]) and how readout values become [`Answer`]s ([`answers`]) all
//! follow from the description. How it runs (forks, trunk retention,
//! batching) is the engine's readout plan, which knows nothing of
//! decisions.

pub mod answers;
mod boxes;
mod codes;
mod description;
mod grid;
pub mod json;
mod kinds;
mod metadata;
pub mod plan;
mod prompt;
mod render;
pub mod request;
mod spans;
mod values;

pub use answers::{Answer, DecisionAnswers, DecisionUsage};
pub use request::{DecisionRequest, KindName, Placement, Question, QuestionKind};

use codes::{isclose, letter_candidates, letter_keys, CodeSet};
use description::{
    CodeSequence, Description, Edges, Kinds, Matrix, Noul, NoulCodes, Posed, Presence, Score,
    ScoreCodes, Segments, Slots,
};
use grid::Grid;
use json::{python_general, Value};
use kinds::{
    Asked, Coded, GridRead, Kind, PosedCodes, Reading, Served, ServedBox, ServedBoxes,
    ServedEstimate, ServedMultiple, SlotRead,
};
use magnitude_artifacts::{DecisionMetadata, TokenId};
use magnitude_chat::request::ImageInput;
use magnitude_chat::ByteBpeTokenizer;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_readout::{ReadoutPlan, ReadoutValues};
use plan::{LaidSequence, Layout, ReadoutPlanError, TokenPlan, TokenReadout};
use prompt::{Anchored, PromptTemplate};
use render::Escape;
use request::ChoiceOption;
use spans::StatePositions;
use std::cell::OnceCell;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

/// Why a request cannot be decided by this model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionError {
    /// The question at this index of the request.
    Question { question: usize, reason: String },
    /// The request as a whole.
    Request { reason: String },
}

impl fmt::Display for DecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Question { question, reason } => {
                write!(formatter, "question {question}: {reason}")
            }
            Self::Request { reason } => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for DecisionError {}

fn question_error(question: usize, reason: impl Into<String>) -> DecisionError {
    DecisionError::Question {
        question,
        reason: reason.into(),
    }
}

fn request_error(reason: impl Into<String>) -> DecisionError {
    DecisionError::Request {
        reason: reason.into(),
    }
}

/// What a load of a decision model needs from its description before its
/// tokenizer binds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadFacts {
    /// Whether its plans read logits of chosen tokens (`ReadoutKind::Logits`),
    /// which a load reads from its output projection's packed form.
    pub reads_logits: bool,
    /// Whether its plans read scores (`ReadoutKind::Scores`), which a load
    /// prepares only for a model whose plans read them.
    pub reads_scores: bool,
    /// `image_max_pixels`: the most pixels it reads a picture at, when its
    /// description bounds them below its projector's own bound.
    pub image_pixels: Option<usize>,
}

/// The load facts of the model `metadata` describes, for a GGUF that
/// stores a readout head of `head` rows, or none. A description the engine
/// does not implement is refused.
pub fn load_facts(metadata: &DecisionMetadata, head: Option<usize>) -> Result<LoadFacts, String> {
    let description = Description::parse(metadata, head)?;
    Ok(LoadFacts {
        reads_logits: description.reads_logits(),
        reads_scores: description.reads_scores(),
        image_pixels: description.images.max_pixels,
    })
}

/// A loaded decision model: its description, bound to its tokenizer and
/// template, each kind with the codes its slots read.
pub struct Decider {
    template: PromptTemplate,
    escape: Escape,
    tokenizer: Arc<ByteBpeTokenizer>,
    placeholder: Option<String>,
    segments: Segments,
    layout: Layout,
    readout: Matrix,
    codes: CodeSequence,
    /// `image_placements`.
    placements: Vec<Placement>,
    /// The model's option codes in position order, as many as a slot lists.
    sequence: CodeSet,
    kinds: Served,
}

impl Decider {
    /// The description `metadata` gives, for a GGUF whose readout head has
    /// `head` rows (none: it stores no head), its codes checked against the
    /// tokenizer and the head and its template compiled. `placeholder` is
    /// the text one image renders as when the model reads images.
    pub fn new(
        metadata: &DecisionMetadata,
        head: Option<usize>,
        tokenizer: Arc<ByteBpeTokenizer>,
        placeholder: Option<&str>,
    ) -> Result<Self, String> {
        let Description {
            template,
            escape,
            segments,
            layout,
            readout,
            codes,
            kinds,
            images,
        } = Description::parse(metadata, head)?;
        let template = PromptTemplate::new(&template)?;
        let escape = Escape::of(&escape, &tokenizer)?;
        // The most codes any slot lists by position.
        let posed = |grid: &description::GridCodes, size: usize| match &grid.posed {
            Posed::Codes => Some(size),
            Posed::Levels(score) => (!score.levels.contains(&size)).then_some(size),
        };
        let most = [
            kinds
                .noul
                .as_ref()
                .filter(|noul| noul.codes == NoulCodes::Sequence)
                .map(|_| 2),
            kinds.choice.as_ref().map(|choice| choice.max_options),
            kinds
                .score
                .as_ref()
                .filter(|score| score.codes == ScoreCodes::Sequence)
                .map(|score| *score.levels.end()),
            kinds
                .estimate
                .as_ref()
                .and_then(|estimate| posed(&estimate.grid, estimate.max_values)),
            kinds
                .r#box
                .as_ref()
                .filter(|r#box| !r#box.edges.grid.numbers)
                .and_then(|r#box| posed(&r#box.edges.grid, r#box.edges.read.bins)),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(0);
        let sequence = match &codes {
            CodeSequence::Letters { count, probe } => {
                let probe = match probe {
                    true => Some(template.probe()?),
                    false => None,
                };
                CodeSet::letters(&tokenizer, *count, probe.as_ref())?
            }
            CodeSequence::Labels(labels) => CodeSet::labels(&tokenizer, labels)?,
            CodeSequence::Aliases { .. } => {
                CodeSet::aliases(&tokenizer, letter_candidates().take(most))?
            }
        };
        if most > sequence.len() {
            return Err(format!(
                "the description lists up to {most} codes by position, but it has {}",
                sequence.len()
            ));
        }
        if let Matrix::Head { rows } = readout {
            if rows < sequence.len() {
                return Err(format!(
                    "the readout head `decision.readout.weight` has {rows} rows, fewer than the description's {} codes",
                    sequence.len()
                ));
            }
        }
        let kinds = Self::serve(kinds, &sequence, &tokenizer)?;
        Ok(Self {
            template,
            escape,
            tokenizer,
            placeholder: placeholder.map(str::to_owned),
            segments,
            layout,
            readout,
            codes,
            placements: images.placements,
            sequence,
            kinds,
        })
    }

    /// Each kind bound to the codes its slots read.
    fn serve(
        kinds: Kinds,
        sequence: &CodeSet,
        tokenizer: &ByteBpeTokenizer,
    ) -> Result<Served, String> {
        let noul = |noul: &Arc<Noul>| -> Result<Coded<Arc<Noul>>, String> {
            Ok(Coded {
                codes: Arc::new(match noul.codes {
                    NoulCodes::Sequence => sequence.prefix(2),
                    NoulCodes::YesNo => CodeSet::yes_no(tokenizer)?,
                }),
                kind: noul.clone(),
            })
        };
        let levels = |score: &Score| match score.codes {
            ScoreCodes::Sequence => Ok(sequence.prefix(*score.levels.end())),
            ScoreCodes::Digits => CodeSet::digits(tokenizer, *score.levels.end()),
        };
        let grid = |grid: &description::GridCodes| -> Result<GridRead, String> {
            Ok(GridRead {
                posed: match &grid.posed {
                    Posed::Codes => PosedCodes::Codes,
                    Posed::Levels(score) => PosedCodes::Levels {
                        levels: score.levels.clone(),
                        codes: levels(score)?,
                    },
                },
                numbers: grid.numbers,
            })
        };
        let presence = |presence: &Presence| -> Result<Coded<Presence>, String> {
            Ok(Coded {
                codes: noul(&presence.noul)?.codes,
                kind: presence.clone(),
            })
        };
        // A box's bins' codes: its bin `k` as the number `k`, or as posed.
        let edges = |edges: &Edges| -> Result<Coded<boxes::BoxRead>, String> {
            let bins = edges.read.bins;
            let codes = match edges.grid.numbers {
                true => CodeSet::numbers(tokenizer, (0..bins).map(|bin| bin as f64))?
                    .ok_or_else(|| {
                        format!(
                            "`box.numbers`: the tokenizer does not write every bin 0 to {} as one token",
                            bins - 1
                        )
                    })?,
                false => grid(&edges.grid)?.codes(sequence, bins),
            };
            Ok(Coded {
                kind: edges.read,
                codes: Arc::new(codes),
            })
        };
        Ok(Served {
            noul: kinds.noul.as_ref().map(noul).transpose()?,
            multiple: kinds
                .multiple
                .map(|kind| -> Result<_, String> {
                    Ok(ServedMultiple {
                        noul: noul(&kind.noul)?,
                        choice: kind.choice,
                        table: kind.table,
                        temperature: kind.temperature,
                    })
                })
                .transpose()?,
            choice: kinds.choice,
            score: kinds
                .score
                .map(|score| -> Result<_, String> {
                    Ok(Coded {
                        codes: Arc::new(levels(&score)?),
                        kind: score,
                    })
                })
                .transpose()?,
            read: kinds.read,
            estimate: kinds
                .estimate
                .map(|kind| -> Result<_, String> {
                    Ok(ServedEstimate {
                        grid: grid(&kind.grid)?,
                        kind,
                    })
                })
                .transpose()?,
            r#box: kinds
                .r#box
                .map(|kind| -> Result<_, String> {
                    Ok(ServedBox {
                        edges: edges(&kind.edges)?,
                        numbers: kind.edges.grid.numbers,
                        presence: kind.presence.as_ref().map(presence).transpose()?,
                        confidence: kind.confidence,
                        temperature: kind.temperature,
                    })
                })
                .transpose()?,
            boxes: kinds
                .boxes
                .map(|kind| -> Result<_, String> {
                    Ok(ServedBoxes {
                        edges: edges(&kind.edges)?,
                        numbers: kind.edges.grid.numbers,
                        presence: presence(&kind.presence)?,
                        max: kind.max,
                        confidence: kind.confidence,
                        temperature: kind.temperature,
                    })
                })
                .transpose()?,
        })
    }

    /// A choice's codes: its keys under `codes.keys_if_letters` when each is
    /// one letter, else its options' positions.
    fn choice_codes(&self, options: &[ChoiceOption]) -> Result<CodeSet, String> {
        if let CodeSequence::Aliases {
            keys_if_letters: true,
        } = self.codes
        {
            if let Some(keys) = letter_keys(options.iter().map(|option| option.key.as_str())) {
                return CodeSet::aliases(&self.tokenizer, keys);
            }
        }
        Ok(self.sequence.prefix(options.len()))
    }

    /// Check the request against the description, then render, tokenize
    /// and lay it out. Host-only: no device is touched.
    pub fn prepare(&self, request: &DecisionRequest) -> Result<PreparedDecision, DecisionError> {
        let asked = self.admit(request)?;
        let bucketed = asked
            .iter()
            .filter(|item| {
                item.temperature()
                    .is_some_and(|temperature| temperature.bucketed())
            })
            .count();
        let questions: Vec<_> = asked
            .iter()
            .zip(&request.questions)
            .map(|(item, question)| (item, &question.instructions))
            .collect();
        let layout = self.layout;
        let mut sequences = Vec::new();
        let mut readings = Vec::with_capacity(asked.len());
        for set in layout.sets(asked.len()) {
            let set: Vec<usize> = set.collect();
            let context = prompt::context(
                &request.state,
                &request.front,
                &questions,
                &set,
                &self.escape,
            );
            let anchored = self
                .template
                .render(
                    &context,
                    set.len(),
                    request.images.len(),
                    self.placeholder.as_deref(),
                    &request.state,
                    &self.escape,
                )
                .map_err(request_error)?;
            let questions: Vec<(usize, &Asked)> =
                set.iter().map(|question| (*question, &asked[*question])).collect();
            let (sequence, read) =
                self.sequence(anchored, &questions, bucketed, &request.state)?;
            sequences.push(sequence);
            // Sets are consecutive, so their questions' readings are in
            // request order.
            readings.extend(read);
        }
        let plan = layout.plan(sequences).map_err(request_error)?;
        Ok(PreparedDecision { plan, readings })
    }

    /// The questions of `request` its description answers, refusing a
    /// request whose kinds, option or level counts, ranges or image
    /// placements it does not serve; a model without an image processor
    /// reads images nowhere.
    fn admit<'r>(&self, request: &'r DecisionRequest) -> Result<Vec<Asked<'r, '_>>, DecisionError> {
        if request.questions.is_empty() {
            return Err(request_error("a request needs at least one question"));
        }
        let placements: &[Placement] = match self.placeholder {
            Some(_) => &self.placements,
            None => &[],
        };
        let placed = |placement: Placement| placements.contains(&placement);
        if request
            .front
            .iter()
            .chain(request.state.images().iter())
            .any(|image| *image >= request.images.len())
        {
            return Err(request_error(
                "an image placement names no image of the request",
            ));
        }
        if !request.front.is_empty() && !placed(Placement::Front) {
            return Err(request_error(
                "this model does not read images before the state",
            ));
        }
        if !request.state.images().is_empty() && !placed(Placement::State) {
            return Err(request_error(
                "this model does not read images inside the state",
            ));
        }
        let mut asked = Vec::with_capacity(request.questions.len());
        for (index, question) in request.questions.iter().enumerate() {
            let mut option_images = Vec::new();
            let item = self
                .ask(&question.kind, &mut option_images)
                .map_err(|reason| question_error(index, reason))?;
            let instruction_images = question.instructions.images();
            if instruction_images
                .iter()
                .chain(&option_images)
                .any(|image| *image >= request.images.len())
            {
                return Err(question_error(
                    index,
                    "an image placement names no image of the request",
                ));
            }
            if !instruction_images.is_empty() && !placed(Placement::Instructions) {
                return Err(question_error(
                    index,
                    "this model does not read images in instructions",
                ));
            }
            if !option_images.is_empty() && !placed(Placement::Options) {
                return Err(question_error(
                    index,
                    "this model does not read images in options",
                ));
            }
            asked.push(item);
        }
        Ok(asked)
    }

    /// One question as its kind's description reads it, and the images its
    /// options place.
    fn ask<'r>(
        &self,
        question: &'r QuestionKind,
        option_images: &mut Vec<usize>,
    ) -> Result<Asked<'r, '_>, String> {
        let kinds = &self.kinds;
        let refused = || {
            format!(
                "this model does not answer {} questions",
                question.name().as_str()
            )
        };
        let options = |options: &'r [ChoiceOption], most: usize, option_images: &mut Vec<usize>| {
            if options.is_empty() || options.len() > most {
                return Err(format!("a choice needs 1 to {most} options"));
            }
            let mut keys: Vec<&str> = options.iter().map(|option| option.key.as_str()).collect();
            keys.sort_unstable();
            if keys.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err("choice option keys must be distinct".to_owned());
            }
            option_images.extend(
                options
                    .iter()
                    .flat_map(|option| option.description.images()),
            );
            Ok(())
        };
        Ok(match question {
            QuestionKind::Predicate { truthy, falsy } => {
                let kind = kinds.noul.as_ref().ok_or_else(refused)?;
                option_images.extend(truthy.images());
                option_images.extend(falsy.images());
                Asked {
                    codes: kind.codes.clone(),
                    kind: Kind::Noul {
                        truthy,
                        falsy,
                        kind,
                    },
                }
            }
            QuestionKind::Choice {
                options: listed,
                multiple: false,
            } => {
                let kind = kinds.choice.as_ref().ok_or_else(refused)?;
                options(listed, kind.max_options, option_images)?;
                Asked {
                    codes: Arc::new(self.choice_codes(listed)?),
                    kind: Kind::Choice {
                        options: listed,
                        kind,
                    },
                }
            }
            QuestionKind::Choice {
                options: listed,
                multiple: true,
            } => {
                let kind = kinds.multiple.as_ref().ok_or_else(refused)?;
                options(listed, kind.choice.max_options, option_images)?;
                Asked {
                    codes: kind.noul.codes.clone(),
                    kind: Kind::Multiple {
                        options: listed,
                        kind,
                    },
                }
            }
            QuestionKind::Score { levels } => {
                let kind = kinds.score.as_ref().ok_or_else(refused)?;
                if !kind.kind.levels.contains(&levels.len()) {
                    return Err(format!(
                        "a score needs {} to {} levels",
                        kind.kind.levels.start(),
                        kind.kind.levels.end()
                    ));
                }
                option_images.extend(levels.iter().flat_map(Value::images));
                Asked {
                    codes: Arc::new(kind.codes.prefix(levels.len())),
                    kind: Kind::Score {
                        levels,
                        kind: &kind.kind,
                    },
                }
            }
            QuestionKind::Read { value } => {
                let kind = kinds.read.as_ref().ok_or_else(refused)?;
                Asked {
                    codes: Arc::new(CodeSet::default()),
                    kind: Kind::Read { value, kind },
                }
            }
            QuestionKind::Estimate {
                low,
                high,
                step,
                unit,
            } => {
                let ServedEstimate { kind, grid: posed } =
                    kinds.estimate.as_ref().ok_or_else(refused)?;
                // The range runs low to high and a step, above 0, divides it.
                let values = step.map(|step| (high - low) / step);
                if !(low < high)
                    || step.is_some_and(|step| !(step > 0.0))
                    || values.is_some_and(|values| !isclose(values, values.round_ties_even(), 0.0))
                {
                    return Err(format!(
                        "an estimate's range runs low to high and its step, above 0, divides it: not [{}, {}] in steps of {}",
                        python_general(*low),
                        python_general(*high),
                        step.map_or("None".into(), python_general)
                    ));
                }
                let size = values.map_or(kind.max_values, |values| {
                    kind.max_values.min(values.round_ties_even() as usize + 1)
                });
                let grid = Grid::new(*low, *high, size)?;
                let numbers = match posed.numbers {
                    true => CodeSet::numbers(&self.tokenizer, grid.values())?,
                    false => None,
                };
                Asked {
                    kind: Kind::Estimate {
                        grid,
                        unit: unit.as_deref(),
                        numbers: numbers.is_some(),
                        kind,
                    },
                    codes: Arc::new(numbers.unwrap_or_else(|| posed.codes(&self.sequence, size))),
                }
            }
            QuestionKind::Box => {
                let kind = kinds.r#box.as_ref().ok_or_else(refused)?;
                Asked {
                    codes: kind.edges.codes.clone(),
                    kind: Kind::Box { kind },
                }
            }
            QuestionKind::Boxes { max } => {
                let kind = kinds.boxes.as_ref().ok_or_else(refused)?;
                let blocks = match max {
                    None => kind.max,
                    Some(max) if (1..=kind.max).contains(&(*max as usize)) => *max as usize,
                    Some(_) => {
                        return Err(format!(
                            "\"max\" belongs to boxes, a whole number from 1 to {}",
                            kind.max
                        ))
                    }
                };
                Asked {
                    codes: kind.edges.codes.clone(),
                    kind: Kind::Boxes { blocks, kind },
                }
            }
        })
    }

    /// One rendering's sequence, before its layout's blocks: its state's
    /// tokens, then each of its `questions`' (request index and question),
    /// as its segments tokenize them, and its readouts; and how each
    /// question is read, a read over the positions of its sequence's state.
    fn sequence(
        &self,
        anchored: Anchored,
        questions: &[(usize, &Asked)],
        bucketed: usize,
        state: &Value,
    ) -> Result<(LaidSequence, Vec<Reading>), DecisionError> {
        let Anchored {
            rendered,
            starts,
            slots: anchors,
            read: state_text,
        } = anchored;
        let mut ends: Vec<usize> = starts[1..].to_vec();
        ends.push(rendered.text.len());
        let encode = |range: Range<usize>| {
            rendered
                .encode_range(&self.tokenizer, range)
                .map_err(request_error)
        };
        let per_question = self.segments == Segments::PerQuestion;
        // The text the state lies in: the whole rendering, or the first
        // question's whole prompt.
        let first = &rendered.text[..if per_question {
            ends[0]
        } else {
            rendered.text.len()
        }];
        let first_encoded = encode(0..first.len())?;
        // Each question's text tokenized alone, under `per_question`.
        let alone = match per_question {
            false => Vec::new(),
            true => starts
                .iter()
                .zip(&ends)
                .map(|(start, end)| encode(*start..*end))
                .collect::<Result<Vec<_>, _>>()?,
        };
        let mut tokens = first_encoded.tokens.clone();
        let rows: Vec<Range<usize>> = match per_question {
            false => {
                let mut bounds: Vec<usize> = starts
                    .iter()
                    .map(|start| first_encoded.tokens_before(*start))
                    .collect();
                bounds.push(first_encoded.tokens.len());
                bounds.windows(2).map(|pair| pair[0]..pair[1]).collect()
            }
            true => {
                // The state is the first prompt less its question's own
                // tokens.
                let mut at = first_encoded
                    .tokens
                    .len()
                    .checked_sub(alone[0].tokens.len())
                    .ok_or_else(|| {
                        request_error(
                            "the first question tokenizes longer alone than in its prompt",
                        )
                    })?;
                for encoded in &alone[1..] {
                    tokens.extend(&encoded.tokens);
                }
                alone
                    .iter()
                    .map(|encoded| {
                        at += encoded.tokens.len();
                        at - encoded.tokens.len()..at
                    })
                    .collect()
            }
        };
        // The position of the token a question's slot anchor reads.
        let anchored_at = |index: usize, anchor: usize| match per_question {
            false => first_encoded.slot_token(&rendered.text, anchor),
            true => alone[index]
                .slot_token(
                    &rendered.text[starts[index]..ends[index]],
                    anchor - starts[index],
                )
                .map(|token| rows[index].start + token),
        };
        // The positions of the state, computed once for the set's reads: the
        // text the template places, or none for a request without a state.
        let positions = OnceCell::new();
        let mut state_positions = || -> Result<Arc<StatePositions>, String> {
            if let Some(positions) = positions.get() {
                return Ok(Arc::clone(positions));
            }
            let placed = match (&state_text, state) {
                (Some(placed), _) => StatePositions::new(
                    first,
                    &first_encoded.ranges,
                    placed.start,
                    &placed.text,
                    &placed.boundaries,
                    placed.json,
                ),
                (None, Value::Null) => {
                    StatePositions::new(first, &first_encoded.ranges, 0, "", &[0], false)
                }
                (None, _) => {
                    return Err(
                        "a read indexes the state's text, but the decision template places none"
                            .into(),
                    )
                }
            };
            Ok(Arc::clone(positions.get_or_init(|| Arc::new(placed))))
        };
        let mut readouts = Vec::new();
        let mut readings = Vec::with_capacity(questions.len());
        let mut lists: Vec<(Arc<CodeSet>, Arc<[TokenId]>)> = Vec::new();
        for (index, (question, asked)) in questions.iter().enumerate() {
            let rows = rows[index].clone();
            if rows.is_empty() {
                return Err(question_error(*question, "the question renders no token"));
            }
            let reading = asked
                .reading(bucketed, &mut state_positions)
                .map_err(|reason| question_error(*question, reason))?;
            let slots = reading.slots();
            let at: Vec<usize> = match asked.at() {
                Slots::Last if anchors[index].is_empty() && slots.len() == 1 => vec![rows.end - 1],
                Slots::Anchors if anchors[index].len() == slots.len() => anchors[index]
                    .iter()
                    .map(|anchor| {
                        anchored_at(index, *anchor)
                            .filter(|position| rows.contains(position))
                            .ok_or_else(|| {
                                question_error(
                                    *question,
                                    "a slot anchor ends no token of its question, but for whitespace",
                                )
                            })
                    })
                    .collect::<Result<_, _>>()?,
                at => {
                    return Err(question_error(
                        *question,
                        format!(
                            "the decision template marks {} slots for this question, which reads {}",
                            anchors[index].len(),
                            match at {
                                Slots::Last => 0,
                                Slots::Anchors => slots.len(),
                            }
                        ),
                    ))
                }
            };
            for (position, slot) in at.into_iter().zip(slots) {
                match slot {
                    SlotRead::Codes(codes) => readouts.push((
                        position,
                        match self.readout {
                            Matrix::Head { .. } => TokenReadout::Head,
                            Matrix::Output => {
                                let list =
                                    match lists.iter().find(|(set, _)| Arc::ptr_eq(set, codes)) {
                                        Some((_, list)) => list.clone(),
                                        None => {
                                            let list: Arc<[TokenId]> = Arc::from(codes.tokens());
                                            lists.push((codes.clone(), list.clone()));
                                            list
                                        }
                                    };
                                TokenReadout::Logits(list)
                            }
                        },
                    )),
                    SlotRead::Span(positions) => {
                        readouts.push((position, TokenReadout::Scores(0..1)));
                        if let Some(state) = positions.state_rows() {
                            readouts.push((position, TokenReadout::Scores(state)));
                        }
                    }
                }
            }
            readings.push(reading);
        }
        Ok((
            LaidSequence {
                images: rendered.images,
                readouts,
                state: rows[0].start,
                questions: rows.iter().map(|rows| rows.len()).collect(),
                tokens,
            },
            readings,
        ))
    }
}

/// A request rendered, tokenized and laid out, waiting for its readout
/// values.
pub struct PreparedDecision {
    plan: TokenPlan,
    /// Per question, in request order, which is plan order.
    readings: Vec<Reading>,
}

impl PreparedDecision {
    pub fn token_plan(&self) -> &TokenPlan {
        &self.plan
    }

    /// The engine plan over the request's images, each sequence's input
    /// prepared by `prepare` (see [`TokenPlan::readout_plan`]).
    pub fn readout_plan<E>(
        &self,
        request: &DecisionRequest,
        prepare: impl FnMut(Vec<TokenId>, &[ImageInput]) -> Result<PreparedModelInput, E>,
    ) -> Result<ReadoutPlan, ReadoutPlanError<E>> {
        self.plan.readout_plan(&request.images, prepare)
    }

    pub fn answers(&self, values: &ReadoutValues) -> Result<DecisionAnswers, String> {
        let shape: Vec<usize> = self
            .plan
            .sequences
            .iter()
            .map(|sequence| sequence.readouts.len())
            .collect();
        let read: Vec<usize> = values.sequences.iter().map(Vec::len).collect();
        if shape != read {
            return Err("readout values do not match the plan".into());
        }
        // The readouts in plan order are the questions' slots in request
        // order.
        let mut slots = values.sequences.iter().flatten().map(Vec::as_slice);
        let answers = self
            .readings
            .iter()
            .map(|reading| reading.answer(&mut slots))
            .collect::<Result<_, _>>()?;
        if slots.next().is_some() {
            return Err("readout values outnumber the questions' slots".into());
        }
        Ok(DecisionAnswers {
            answers,
            usage: DecisionUsage {
                input_tokens: values.usage.computed_tokens + values.usage.cached_tokens,
                cached_tokens: values.usage.cached_tokens,
            },
        })
    }
}

#[cfg(test)]
mod arithmetic {
    //! The answer arithmetic against d1-2B's own functions
    //! (`tests/support/d1_arithmetic.py`): written values, correctly rounded
    //! sums, and boxes and estimates through its tables, its edges read as
    //! its 128 codes.
    use crate::answers::{fsum, Answer, Rect, WrittenNumber};
    use crate::boxes::{self, BoxRead, Crossed, Drawn, Edge, Points, Positions};
    use crate::description::Table;
    use crate::grid::Grid;
    use crate::values::{parse_date, parse_number, written_unit};
    use serde_json::Value;

    const EDGES: BoxRead = BoxRead {
        bins: 128,
        positions: Positions::Ends,
        edge: Edge::Median,
        points: Points::Halton,
        draws: 512,
        iou: 0.5,
        crossed: Crossed::Shift,
        drawn: Drawn::Ordered,
    };

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/d1_arithmetic.json"))
            .expect("the fixture is JSON")
    }

    fn floats(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .expect("an array")
            .iter()
            .map(|value| value.as_f64().expect("a number"))
            .collect()
    }

    fn knots(fixture: &Value, name: &str) -> Vec<[f64; 2]> {
        fixture["tables"][name]
            .as_array()
            .expect("a table")
            .iter()
            .map(|knot| {
                let knot = floats(knot);
                [knot[0], knot[1]]
            })
            .collect()
    }

    fn table(fixture: &Value, name: &str) -> Table {
        Table::of(knots(fixture, name))
    }

    fn close(ours: f64, theirs: &Value, at: &str) {
        let theirs = theirs.as_f64().expect("a number");
        assert!((ours - theirs).abs() <= 1e-12, "{at}: {ours} != {theirs}");
    }

    #[test]
    fn written_values_parse_as_the_reference_parses_them() {
        let fixture = fixture();
        for case in fixture["numbers"].as_array().expect("numbers") {
            let at = format!("number {case}");
            let parsed = parse_number(
                case["text"].as_str().expect("text"),
                case["context"].as_str().expect("context"),
                case["unit"].as_str(),
            )
            .expect("read within the backtrack limit");
            match (&parsed, &case["number"]) {
                (None, Value::Null) => {}
                (Some(WrittenNumber::Integer(digits)), expected) => {
                    assert_eq!(Some(digits.as_str()), expected["integer"].as_str(), "{at}")
                }
                (Some(WrittenNumber::Float(value)), expected) => {
                    assert_eq!(Some(*value), expected["float"].as_f64(), "{at}")
                }
                (None, expected) => panic!("{at}: none, not {expected}"),
            }
        }
        for case in fixture["dates"].as_array().expect("dates") {
            assert_eq!(
                parse_date(
                    case["text"].as_str().expect("text"),
                    case["context"].as_str().expect("context")
                )
                .expect("read within the backtrack limit")
                .as_deref(),
                case["date"].as_str(),
                "date {case}"
            );
        }
        for case in fixture["units"].as_array().expect("units") {
            assert_eq!(
                written_unit(case["text"].as_str().expect("text"))
                    .expect("read within the backtrack limit")
                    .as_deref(),
                case["unit"].as_str(),
                "unit {case}"
            );
        }
    }

    #[test]
    fn sums_are_correctly_rounded_as_fsum_rounds_them() {
        for case in fixture()["sums"].as_array().expect("sums") {
            assert_eq!(
                fsum(floats(&case["values"])),
                case["sum"].as_f64().expect("a sum"),
                "{case}"
            );
        }
    }

    #[test]
    fn boxes_and_estimates_read_as_the_reference_reads_them() {
        let fixture = fixture();
        let rect = |rect: &Rect, theirs: &Value, at: &str| {
            close(rect.top, &theirs["top"], at);
            close(rect.bottom, &theirs["bottom"], at);
            close(rect.left, &theirs["left"], at);
            close(rect.right, &theirs["right"], at);
        };
        for case in fixture["boxes"].as_array().expect("boxes") {
            let rows: Vec<Vec<f64>> = case["probabilities"]
                .as_array()
                .expect("rows")
                .iter()
                .map(floats)
                .collect();
            let expected = &case["answer"];
            match case["blocks"].as_u64() {
                None => {
                    let Answer::Box {
                        found,
                        rect: drawn,
                        confidence,
                    } = boxes::present(
                        &EDGES,
                        rows[0][0],
                        &rows[1..],
                        &table(&fixture, "found:box"),
                        &table(&fixture, "box"),
                    )
                    else {
                        panic!("a box answer")
                    };
                    close(
                        found.expect("d1 reads presence"),
                        &expected["found"],
                        "found",
                    );
                    close(confidence, &expected["confidence"], "confidence");
                    match drawn {
                        Some(drawn) => rect(&drawn, &expected["box"], "box"),
                        None => assert!(expected["box"].is_null()),
                    }
                }
                Some(_) => {
                    let blocks: Vec<(f64, Vec<Vec<f64>>)> = rows
                        .chunks(5)
                        .map(|block| (block[0][0], block[1..].to_vec()))
                        .collect();
                    let Answer::Boxes {
                        boxes,
                        at_least,
                        presence,
                    } = boxes::every(
                        &EDGES,
                        &blocks,
                        &table(&fixture, "found:boxes"),
                        &table(&fixture, "boxes"),
                    )
                    else {
                        panic!("a boxes answer")
                    };
                    assert_eq!(Some(at_least), expected["at_least"].as_bool());
                    let theirs = expected["presence"].as_array().expect("presence");
                    assert_eq!(presence.len(), theirs.len());
                    for (ours, theirs) in presence.iter().zip(theirs) {
                        close(*ours, theirs, "presence");
                    }
                    let listed = expected["boxes"].as_array().expect("boxes");
                    assert_eq!(boxes.len(), listed.len());
                    for (ours, theirs) in boxes.iter().zip(listed) {
                        rect(&ours.rect, &theirs["box"], "boxes");
                        close(ours.found, &theirs["found"], "found");
                        close(ours.confidence, &theirs["confidence"], "confidence");
                    }
                }
            }
        }
        // `calibration.estimate` is read backwards.
        let levels = Table::of(
            knots(&fixture, "estimate")
                .into_iter()
                .map(|[raw, calibrated]| [calibrated, raw])
                .collect(),
        );
        for case in fixture["estimates"].as_array().expect("estimates") {
            let bounds = floats(&case["grid"]);
            let grid = Grid::new(bounds[0], bounds[1], bounds[2] as usize).expect("a grid");
            let read = grid.read(&floats(&case["probabilities"]), |share| {
                levels.through(share)
            });
            let theirs = &case["read"];
            close(read.p10, &theirs[0], "p10");
            close(read.median, &theirs[1], "median");
            close(read.p90, &theirs[2], "p90");
        }
    }
}
