//! Answers: the model's probabilities per question, and the values and
//! confidences derived from them: a score's expected level and TypeSafe's
//! choice and score confidences, which the wire defines for every model, and
//! what a read, an estimate or a box derives as its description says. The
//! wire adapters only format them; each formula exists once, here.

/// Every question's answer, in request order, and what the plan read.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionAnswers {
    pub answers: Vec<Answer>,
    pub usage: DecisionUsage,
}

/// Input rows the evaluation read: the trunk once, each branch, and every
/// packed or repeated trunk as it ran; `cached_tokens` of them were restored
/// from a retained trunk rather than computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecisionUsage {
    pub input_tokens: u64,
    pub cached_tokens: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// P(true).
    Predicate { probability: f64 },
    /// One probability per option, in option order, summing to one, and
    /// TypeSafe's confidence; the choice is the likeliest option.
    Choice {
        probabilities: Vec<f64>,
        confidence: f64,
    },
    /// Each option's own P(yes), in option order; an option is chosen at
    /// one half or more.
    Multiple { probabilities: Vec<f64> },
    /// One probability per level, lowest first, the expected level and
    /// TypeSafe's confidence.
    Score {
        probabilities: Vec<f64>,
        score: f64,
        confidence: f64,
    },
    /// The box, when one is drawn: `found` is P(present) for a type that
    /// reads presence (absent for one that always draws).
    Box {
        found: Option<f64>,
        rect: Option<Rect>,
        confidence: f64,
    },
    /// Every instance drawn, in reading order; `at_least` when every block
    /// read is present (the true count may be higher); `presence[k]` is
    /// P(count > k).
    Boxes {
        boxes: Vec<FoundBox>,
        at_least: bool,
        presence: Vec<f64>,
    },
    /// A value read from the state: P(found) through the type's table, the
    /// span when the read is found and points at one, and the value the
    /// span writes.
    Read {
        found: f64,
        read: Option<ReadSpan>,
        value: ReadValue,
    },
    /// The median and central 80 percent interval over the grid.
    Estimate { median: f64, p10: f64, p90: f64 },
}

/// Edges as fractions of the picture: top and bottom from its top edge,
/// left and right from its left edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundBox {
    pub rect: Rect,
    pub found: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReadSpan {
    /// The characters read: the state's own, or the characters a JSON
    /// string's escapes encode when the span lies in one string.
    pub text: String,
    /// Its character range in the state's rendered text.
    pub span: [usize; 2],
    /// The JSON pointer (RFC 6901) of the one string the span lies in.
    pub path: Option<String>,
}

/// What a read's span writes, by the read's value type.
#[derive(Clone, Debug, PartialEq)]
pub enum ReadValue {
    /// The text itself.
    Text,
    /// The number the span writes, when its writing settles one, and its
    /// unit: the question's, else a percent or currency sign the span
    /// writes.
    Number {
        number: Option<WrittenNumber>,
        unit: Option<String>,
    },
    /// The ISO 8601 date (or date and time) the span writes, when it
    /// settles one.
    Date { date: Option<String> },
}

/// A number as written: integral values exactly, as decimal digits with an
/// optional leading `-`; others as the nearest float.
#[derive(Clone, Debug, PartialEq)]
pub enum WrittenNumber {
    Integer(String),
    Float(f64),
}

/// `softmax(scores / temperature)` in f64.
pub fn softmax(scores: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = scores.iter().map(|score| score / temperature).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scaled.iter().map(|score| (score - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

/// Python's `math.fsum`: the correctly rounded sum (Shewchuk's partials,
/// rounded half-even across them).
pub fn fsum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for value in values {
        let mut x = value;
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0.0 {
                partials[kept] = low;
                kept += 1;
            }
            x = high;
        }
        partials.truncate(kept);
        partials.push(x);
    }
    let Some(mut high) = partials.pop() else {
        return 0.0;
    };
    let mut low = 0.0;
    while let Some(y) = partials.pop() {
        let x = high;
        high = x + y;
        low = y - (high - x);
        if low != 0.0 {
            break;
        }
    }
    if let Some(&next) = partials.last() {
        if (low < 0.0 && next < 0.0) || (low > 0.0 && next > 0.0) {
            let y = low * 2.0;
            let x = high + y;
            if y == x - high {
                high = x;
            }
        }
    }
    high
}

/// The index of the largest probability; the first on ties.
pub fn argmax(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .fold(0, |best, (index, value)| {
            if *value > probabilities[best] {
                index
            } else {
                best
            }
        })
}

/// The probability-weighted mean level index.
pub fn expected_level(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(level, probability)| level as f64 * probability)
        .sum()
}

/// TypeSafe's choice confidence: the peak probability rescaled from uniform
/// (0) to certainty (1).
pub fn choice_confidence(probabilities: &[f64]) -> f64 {
    let count = probabilities.len() as f64;
    if probabilities.len() == 1 {
        return 1.0;
    }
    let uniform = 1.0 / count;
    let peak = probabilities[argmax(probabilities)];
    ((peak - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
}

/// TypeSafe's score confidence: one minus the probability-weighted mean
/// absolute distance from the modal level, over that of the uniform
/// distribution.
pub fn score_confidence(probabilities: &[f64]) -> f64 {
    let levels = probabilities.len();
    if levels == 1 {
        return 1.0;
    }
    let mode = argmax(probabilities) as f64;
    let distance: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(level, probability)| probability * (level as f64 - mode).abs())
        .sum();
    let center = (levels - 1) as f64 / 2.0;
    let uniform = (0..levels)
        .map(|level| (level as f64 - center).abs())
        .sum::<f64>()
        / levels as f64;
    (1.0 - distance / uniform).max(0.0)
}

/// Piecewise-linear interpolation through `knots` (`(x, y)`, nonempty, `x`
/// nondecreasing), held flat outside them; a vertical step takes its lower
/// value.
pub fn through(value: f64, knots: &[[f64; 2]]) -> f64 {
    let [x, y] = knots[0];
    if value <= x {
        return y;
    }
    for pair in knots.windows(2) {
        let ([x0, y0], [x1, y1]) = (pair[0], pair[1]);
        if value <= x1 {
            return if x1 == x0 {
                y0
            } else {
                y0 + (y1 - y0) * (value - x0) / (x1 - x0)
            };
        }
    }
    knots[knots.len() - 1][1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn formulas_match_their_references() {
        // qev `answers.py` and pplx `protocol.py`.
        close(
            choice_confidence(&[0.7, 0.2, 0.1]),
            (0.7 - 1.0 / 3.0) / (2.0 / 3.0),
        );
        close(choice_confidence(&[1.0]), 1.0);
        let levels = [0.1, 0.6, 0.3];
        close(expected_level(&levels), 1.2);
        // mode 1, distance 0.4, uniform MAD 2/3.
        close(score_confidence(&levels), 1.0 - 0.4 / (2.0 / 3.0));
        close(through(0.5, &[[0.0, 0.0], [1.0, 2.0]]), 1.0);
        close(through(-1.0, &[[0.0, 0.25], [1.0, 2.0]]), 0.25);
        close(through(3.0, &[[0.0, 0.25], [1.0, 2.0]]), 2.0);
        close(fsum([1e16, 1.0, -1e16]), 1.0);
        let probabilities = softmax(&[1.0, 2.0], 0.5);
        close(probabilities[1], 1.0 / (1.0 + (-2.0f64).exp()));
    }
}
