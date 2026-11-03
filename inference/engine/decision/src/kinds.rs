//! The question kinds the engine answers: what each question lists, where
//! it is read, and how its slots' values become its answer, as its kind's
//! description says.

use crate::answers::{
    choice_confidence, expected_level, score_confidence, softmax, Answer, ReadValue as Parsed,
};
use crate::boxes::{self, BoxRead};
use crate::codes::CodeSet;
use crate::description::{
    Choice, Estimate, Noul, Presence, ReadKind, ReadOf, Score,
    Slots, Table, Temperature,
};
use crate::grid::Grid;
use crate::json::Value;
use crate::request::{ChoiceOption, ReadValue};
use crate::spans::{self, StatePositions};
use crate::values;
use std::ops::RangeInclusive;
use std::sync::Arc;

/// A part of a description bound to the codes its slots read.
#[derive(Clone, Debug)]
pub(crate) struct Coded<K> {
    pub kind: K,
    pub codes: Arc<CodeSet>,
}

/// How a grid's values are posed when not read as numbers, bound to the
/// codes: by position, or as a score's levels when the model scores that
/// many (`description::Posed`).
#[derive(Clone, Debug)]
pub(crate) enum PosedCodes {
    Codes,
    Levels {
        levels: RangeInclusive<usize>,
        codes: CodeSet,
    },
}

/// A grid bound to the codes it is posed at.
#[derive(Clone, Debug)]
pub(crate) struct GridRead {
    pub posed: PosedCodes,
    pub numbers: bool,
}

impl GridRead {
    /// The codes of a grid of `size` values, posed as it says (numbers aside).
    pub fn codes(&self, sequence: &CodeSet, size: usize) -> CodeSet {
        match &self.posed {
            PosedCodes::Levels { levels, codes } if levels.contains(&size) => codes.prefix(size),
            _ => sequence.prefix(size),
        }
    }
}

/// A multiple choice: each option read as `noul`, bounded by `choice`.
pub(crate) struct ServedMultiple {
    pub noul: Coded<Arc<Noul>>,
    pub choice: Arc<Choice>,
    pub table: Arc<Table>,
    pub temperature: Temperature,
}

pub(crate) struct ServedEstimate {
    pub kind: Estimate,
    pub grid: GridRead,
}

/// A box: its edges and their bins' codes, and its presence slot's.
pub(crate) struct ServedBox {
    pub edges: Coded<BoxRead>,
    pub numbers: bool,
    pub presence: Option<Coded<Presence>>,
    pub confidence: Arc<Table>,
    pub temperature: Temperature,
}

/// Every instance: `max` blocks, each a presence and four edges.
pub(crate) struct ServedBoxes {
    pub edges: Coded<BoxRead>,
    pub numbers: bool,
    pub presence: Coded<Presence>,
    pub max: usize,
    pub confidence: Arc<Table>,
    pub temperature: Temperature,
}

/// The kinds a model answers, each bound to the codes it reads.
pub(crate) struct Served {
    pub noul: Option<Coded<Arc<Noul>>>,
    pub choice: Option<Arc<Choice>>,
    pub multiple: Option<ServedMultiple>,
    /// Its codes: the longest score's levels.
    pub score: Option<Coded<Arc<Score>>>,
    pub read: Option<ReadKind>,
    pub estimate: Option<ServedEstimate>,
    pub r#box: Option<ServedBox>,
    pub boxes: Option<ServedBoxes>,
}

/// A question admitted by the description: what it renders, and its kind as
/// the model serves it.
pub(crate) enum Kind<'r, 'd> {
    Noul {
        truthy: &'r Value,
        falsy: &'r Value,
        kind: &'d Coded<Arc<Noul>>,
    },
    Choice {
        options: &'r [ChoiceOption],
        kind: &'d Choice,
    },
    /// Each option read as a noul.
    Multiple {
        options: &'r [ChoiceOption],
        kind: &'d ServedMultiple,
    },
    Score {
        levels: &'r [Value],
        kind: &'d Score,
    },
    Read {
        value: &'r ReadValue,
        kind: &'d ReadKind,
    },
    /// `numbers`: whether its codes are its values as numbers.
    Estimate {
        grid: Grid,
        unit: Option<&'r str>,
        numbers: bool,
        kind: &'d Estimate,
    },
    Box {
        kind: &'d ServedBox,
    },
    Boxes {
        blocks: usize,
        kind: &'d ServedBoxes,
    },
}

/// A question as the template lists it: its kind, and the codes its answer
/// slots read (`codes`).
pub(crate) struct Asked<'r, 'd> {
    pub kind: Kind<'r, 'd>,
    pub codes: Arc<CodeSet>,
}

/// One code slot: its codes, through a softmax temperature.
#[derive(Clone, Debug)]
pub(crate) struct CodeSlot {
    codes: Arc<CodeSet>,
    temperature: f64,
}

impl CodeSlot {
    /// Its option probabilities: each option's best spelling, through the
    /// temperature.
    fn shares(&self, values: &[f32]) -> Result<Vec<f64>, String> {
        let spelled: usize = self.codes.groups().iter().sum();
        let values = values
            .get(..spelled)
            .ok_or("the readout reads fewer values than the question's codes")?;
        let mut at = 0;
        let best: Vec<f64> = self
            .codes
            .groups()
            .iter()
            .map(|count| {
                let best = values[at..at + count]
                    .iter()
                    .map(|value| f64::from(*value))
                    .fold(f64::NEG_INFINITY, f64::max);
                at += count;
                best
            })
            .collect();
        Ok(softmax(&best, self.temperature))
    }
}

/// What one slot reads: codes, or (a read's start or end) the state's
/// positions and position 0.
#[derive(Clone)]
pub(crate) enum SlotRead<'a> {
    Codes(&'a Arc<CodeSet>),
    Span(&'a StatePositions),
}

/// A presence slot and its reading: the noul's true option, P(present)
/// through `found`.
#[derive(Clone, Debug)]
pub(crate) struct PresenceSlot {
    slot: CodeSlot,
    true_option: usize,
    found: Arc<Table>,
}

/// One question as the plan reads it: its slots, and how their values
/// become its answer.
#[derive(Clone, Debug)]
pub(crate) enum Reading {
    Noul {
        slot: CodeSlot,
        true_option: usize,
    },
    Choice {
        slot: CodeSlot,
    },
    /// A slot per option, each read as a noul.
    Multiple {
        slot: CodeSlot,
        options: usize,
        true_option: usize,
        table: Arc<Table>,
    },
    Score {
        slot: CodeSlot,
    },
    /// A start and an end slot over the positions of its sequence's state.
    Span {
        positions: Arc<StatePositions>,
        value: ReadValue,
        of: ReadOf,
    },
    Estimate {
        slot: CodeSlot,
        grid: Grid,
        levels: Arc<Table>,
    },
    /// Its presence, when it reads one, then four edges (top, bottom, left,
    /// right).
    Box {
        presence: Option<PresenceSlot>,
        edges: CodeSlot,
        read: BoxRead,
        confidence: Arc<Table>,
    },
    /// `blocks` blocks of a presence and four edges.
    Boxes {
        blocks: usize,
        presence: PresenceSlot,
        edges: CodeSlot,
        read: BoxRead,
        confidence: Arc<Table>,
    },
}

impl Asked<'_, '_> {
    /// The kind as the template's context names it.
    pub fn name(&self) -> &'static str {
        match self.kind {
            Kind::Noul { .. } => "noul",
            Kind::Choice { .. } => "choice",
            Kind::Multiple { .. } => "multiple",
            Kind::Score { .. } => "score",
            Kind::Read { .. } => "read",
            Kind::Estimate { .. } => "estimate",
            Kind::Box { .. } => "box",
            Kind::Boxes { .. } => "boxes",
        }
    }

    /// Where its slots are read: a kind of one slot as its description
    /// says, every other at the template's slot anchors.
    pub fn at(&self) -> Slots {
        match &self.kind {
            Kind::Noul { kind, .. } => kind.kind.slots,
            Kind::Choice { kind, .. } => kind.slots,
            Kind::Score { kind, .. } => kind.slots,
            Kind::Estimate { kind, .. } => kind.slots,
            Kind::Multiple { .. } | Kind::Read { .. } | Kind::Box { .. } | Kind::Boxes { .. } => {
                Slots::Anchors
            }
        }
    }

    /// Its slots' softmax temperature; a read scores positions as they
    /// are.
    pub fn temperature(&self) -> Option<&Temperature> {
        Some(match &self.kind {
            Kind::Noul { kind, .. } => &kind.kind.temperature,
            Kind::Choice { kind, .. } => &kind.temperature,
            Kind::Multiple { kind, .. } => &kind.temperature,
            Kind::Score { kind, .. } => &kind.temperature,
            Kind::Read { .. } => return None,
            Kind::Estimate { kind, .. } => &kind.temperature,
            Kind::Box { kind } => &kind.temperature,
            Kind::Boxes { kind, .. } => &kind.temperature,
        })
    }

    /// How the plan reads it, its temperature picked by the request's
    /// `bucketed` questions; a read over the positions `state` gives its
    /// sequence's state.
    pub fn reading(
        &self,
        bucketed: usize,
        state: &mut dyn FnMut() -> Result<Arc<StatePositions>, String>,
    ) -> Result<Reading, String> {
        let slot = |temperature: &Temperature| CodeSlot {
            codes: self.codes.clone(),
            temperature: temperature.at(bucketed),
        };
        let presence = |presence: &Coded<Presence>, temperature: f64| PresenceSlot {
            slot: CodeSlot {
                codes: presence.codes.clone(),
                temperature,
            },
            true_option: presence.kind.noul.true_option,
            found: presence.kind.found.clone(),
        };
        Ok(match &self.kind {
            Kind::Noul { kind, .. } => Reading::Noul {
                slot: slot(&kind.kind.temperature),
                true_option: kind.kind.true_option,
            },
            Kind::Choice { kind, .. } => Reading::Choice {
                slot: slot(&kind.temperature),
            },
            Kind::Multiple { options, kind } => Reading::Multiple {
                slot: slot(&kind.temperature),
                options: options.len(),
                true_option: kind.noul.kind.true_option,
                table: kind.table.clone(),
            },
            Kind::Score { kind, .. } => Reading::Score {
                slot: slot(&kind.temperature),
            },
            Kind::Read { value, kind } => Reading::Span {
                positions: state()?,
                of: kind.values[match value {
                    ReadValue::Text => 0,
                    ReadValue::Number { .. } => 1,
                    ReadValue::Date => 2,
                }]
                .clone(),
                value: (*value).clone(),
            },
            Kind::Estimate { grid, kind, .. } => Reading::Estimate {
                slot: slot(&kind.temperature),
                grid: *grid,
                levels: kind.levels.clone(),
            },
            Kind::Box { kind } => {
                let edges = slot(&kind.temperature);
                Reading::Box {
                    presence: kind
                        .presence
                        .as_ref()
                        .map(|coded| presence(coded, edges.temperature)),
                    edges,
                    read: kind.edges.kind,
                    confidence: kind.confidence.clone(),
                }
            }
            Kind::Boxes { blocks, kind } => {
                let edges = slot(&kind.temperature);
                Reading::Boxes {
                    blocks: *blocks,
                    presence: presence(&kind.presence, edges.temperature),
                    edges,
                    read: kind.edges.kind,
                    confidence: kind.confidence.clone(),
                }
            }
        })
    }
}

impl Reading {
    /// What each of its slots reads, in slot order.
    pub fn slots(&self) -> Vec<SlotRead<'_>> {
        fn codes(slot: &CodeSlot, count: usize) -> Vec<SlotRead<'_>> {
            vec![SlotRead::Codes(&slot.codes); count]
        }
        match self {
            Self::Noul { slot, .. }
            | Self::Choice { slot, .. }
            | Self::Score { slot, .. }
            | Self::Estimate { slot, .. } => codes(slot, 1),
            Self::Multiple { slot, options, .. } => codes(slot, *options),
            Self::Span { positions, .. } => vec![SlotRead::Span(positions); 2],
            Self::Box {
                presence, edges, ..
            } => presence
                .iter()
                .map(|presence| SlotRead::Codes(&presence.slot.codes))
                .chain(codes(edges, 4))
                .collect(),
            Self::Boxes {
                blocks,
                presence,
                edges,
                ..
            } => (0..*blocks)
                .flat_map(|_| {
                    std::iter::once(SlotRead::Codes(&presence.slot.codes)).chain(codes(edges, 4))
                })
                .collect(),
        }
    }

    /// The question's answer from its readouts' values, in plan order.
    pub fn answer<'v>(
        &self,
        values: &mut dyn Iterator<Item = &'v [f32]>,
    ) -> Result<Answer, String> {
        let mut take = || {
            values.next().ok_or_else(|| {
                "the readout reads fewer values than the question's slots".to_owned()
            })
        };
        fn edges<'v>(
            slot: &CodeSlot,
            take: &mut dyn FnMut() -> Result<&'v [f32], String>,
        ) -> Result<Vec<Vec<f64>>, String> {
            (0..4).map(|_| slot.shares(take()?)).collect()
        }
        let present = |presence: &PresenceSlot, values: &[f32]| -> Result<f64, String> {
            Ok(presence.slot.shares(values)?[presence.true_option])
        };
        Ok(match self {
            Self::Noul { slot, true_option } => Answer::Predicate {
                probability: slot.shares(take()?)?[*true_option],
            },
            Self::Choice { slot } => {
                let probabilities = slot.shares(take()?)?;
                Answer::Choice {
                    confidence: choice_confidence(&probabilities),
                    probabilities,
                }
            }
            Self::Multiple {
                slot,
                options,
                true_option,
                table,
            } => Answer::Multiple {
                probabilities: (0..*options)
                    .map(|_| Ok(table.through(slot.shares(take()?)?[*true_option])))
                    .collect::<Result<_, String>>()?,
            },
            Self::Score { slot } => {
                let probabilities = slot.shares(take()?)?;
                Answer::Score {
                    score: expected_level(&probabilities),
                    confidence: score_confidence(&probabilities),
                    probabilities,
                }
            }
            Self::Span {
                positions,
                value,
                of,
            } => {
                let start = span_scores(positions, &mut take)?;
                let end = span_scores(positions, &mut take)?;
                read(value, of, positions, &start, &end)?
            }
            Self::Estimate {
                slot,
                grid,
                levels,
            } => {
                let read = grid.read(&slot.shares(take()?)?, |share| levels.through(share));
                Answer::Estimate {
                    median: read.median,
                    p10: read.p10,
                    p90: read.p90,
                }
            }
            Self::Box {
                presence: None,
                edges: slot,
                read,
                confidence,
            } => {
                let shares = edges(slot, &mut take)?;
                let rect = read.rect(&shares);
                Answer::Box {
                    found: None,
                    confidence: confidence.through(read.confidence(&shares, &rect)),
                    rect: Some(rect),
                }
            }
            Self::Box {
                presence: Some(presence),
                edges: slot,
                read,
                confidence,
            } => {
                let present = present(presence, take()?)?;
                let shares = edges(slot, &mut take)?;
                boxes::present(read, present, &shares, &presence.found, confidence)
            }
            Self::Boxes {
                blocks,
                presence,
                edges: slot,
                read,
                confidence,
            } => {
                let blocks: Vec<(f64, Vec<Vec<f64>>)> = (0..*blocks)
                    .map(|_| Ok((present(presence, take()?)?, edges(slot, &mut take)?)))
                    .collect::<Result<_, String>>()?;
                boxes::every(read, &blocks, &presence.found, confidence)
            }
        })
    }
}

/// A span slot's scores over its positions: position 0, then the state's;
/// a position that writes no state is `-inf`.
fn span_scores<'a>(
    positions: &StatePositions,
    take: &mut dyn FnMut() -> Result<&'a [f32], String>,
) -> Result<Vec<f64>, String> {
    let mut scores = vec![f64::NEG_INFINITY; positions.width()];
    let [nothing] = take()? else {
        return Err("a span slot reads one score at position 0".into());
    };
    scores[0] = f64::from(*nothing);
    if let Some(rows) = positions.state_rows() {
        let read = take()?;
        if read.len() != rows.len() {
            return Err("a span slot reads a score per state position".into());
        }
        for (position, value) in rows.zip(read) {
            if positions.valid(position) {
                scores[position] = f64::from(*value);
            }
        }
    }
    Ok(scores)
}

/// A read's answer: P(found) through its table, and the span its start and
/// end scores point at when that is at least one half, its value parsed in
/// the state's text.
fn read(
    value: &ReadValue,
    of: &ReadOf,
    positions: &StatePositions,
    start: &[f64],
    end: &[f64],
) -> Result<Answer, String> {
    let ((first, last), found) = spans::span(start, end, of.max_tokens);
    let found = of.found.through(found);
    let read = (found >= 0.5 && (first, last) != (0, 0)).then(|| positions.read(first, last));
    let context = positions.context();
    let value = match value {
        ReadValue::Text => Parsed::Text,
        ReadValue::Number { unit } => match &read {
            Some(read) => Parsed::Number {
                number: values::parse_number(&read.text, &context, unit.as_deref())?,
                unit: match unit.as_deref().filter(|unit| !unit.is_empty()) {
                    Some(unit) => Some(unit.to_owned()),
                    None => values::written_unit(&read.text)?,
                },
            },
            None => Parsed::Number {
                number: None,
                unit: unit.clone(),
            },
        },
        ReadValue::Date => Parsed::Date {
            date: match &read {
                Some(read) => values::parse_date(&read.text, &context)?,
                None => None,
            },
        },
    };
    Ok(Answer::Read { found, read, value })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(codes: CodeSet) -> CodeSlot {
        CodeSlot {
            codes: Arc::new(codes),
            temperature: 1.0,
        }
    }

    fn answer(reading: &Reading, values: &[f32]) -> Result<Answer, String> {
        reading.answer(&mut std::iter::once(values))
    }

    /// Each answer reads only its codes' values: a score its expected level,
    /// a choice and a score TypeSafe's confidences.
    #[test]
    fn answers_follow_their_codes() {
        let two = || slot(CodeSet::positional(&["a", "b"]));
        let even: &[f32] = &[0.0, 0.0, 7.0];
        let Answer::Predicate { probability } = answer(
            &Reading::Noul {
                slot: two(),
                true_option: 1,
            },
            &[0.0, 3f32.ln()],
        )
        .unwrap() else {
            panic!("a noul answer")
        };
        assert!((probability - 0.75).abs() < 1e-6);
        assert_eq!(
            answer(&Reading::Score { slot: two() }, even).unwrap(),
            Answer::Score {
                probabilities: vec![0.5, 0.5],
                score: 0.5,
                confidence: 0.0,
            }
        );
        let Answer::Choice { confidence, .. } =
            answer(&Reading::Choice { slot: two() }, even).unwrap()
        else {
            panic!("a choice answer")
        };
        assert_eq!(confidence, 0.0);
        assert!(answer(
            &Reading::Noul {
                slot: two(),
                true_option: 0
            },
            &[0.0]
        )
        .is_err());
    }
}
