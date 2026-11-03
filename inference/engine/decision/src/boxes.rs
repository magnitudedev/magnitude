//! Boxes read from four edge distributions (top, bottom, left, right), each
//! over equally spaced bins across the picture: the box itself and the
//! share of boxes drawn from the distributions that overlap it. One
//! pipeline serves every model; its description gives the parameters
//! (`box.*`, `KEYS.md`).

use crate::answers::{fsum, Answer, FoundBox, Rect};
use crate::description::Table;
use crate::grid::{Cumulative, Grid};

/// Where bin `k` of `n` lies across the picture, from 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Positions {
    /// `centres`: `(k + 1/2) / n`.
    Centres,
    /// `ends`: the grid of `n` values from 0 to 1, from one side to the
    /// other.
    Ends,
    /// `starts`: the grid of `n` values from 0 to `(n - 1) / n`.
    Starts,
}

/// `box.edge`: an edge's value from its distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// `mean`: the probability-weighted mean position.
    Mean,
    /// `median`: the first position whose cumulative share reaches one
    /// half (the last when rounding leaves every sum short of it).
    Median,
}

/// `box.points`: the low-discrepancy points boxes are drawn at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Points {
    /// `r4`: the R4 sequence from 1/2 (`(1/2 + i α) mod 1`, `α` the powers
    /// of the inverse plastic number of dimension four), from index 1.
    R4,
    /// `halton`: the Halton sequence in bases 2, 3, 5 and 7, from index 1.
    Halton,
}

/// `box.crossed`: two edges of one axis read out of order become one bin
/// about their middle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crossed {
    /// `clamp`: the bin about the middle, cut at the picture's sides and
    /// regrown inside them when that leaves it empty.
    Clamp,
    /// `shift`: the bin about the middle, moved inside the picture.
    Shift,
}

/// `box.drawn`: how a drawn box whose edges cross counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawn {
    /// `valid`: as a miss.
    Valid,
    /// `ordered`: with its edges ordered as the box's own are.
    Ordered,
}

/// The pipeline's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxRead {
    /// `box.bins`: the bins of each edge distribution.
    pub bins: usize,
    pub positions: Positions,
    pub edge: Edge,
    pub points: Points,
    /// `box.draws`: boxes drawn for the confidence.
    pub draws: usize,
    /// `box.iou`: the overlap a drawn box needs to count.
    pub iou: f64,
    pub crossed: Crossed,
    pub drawn: Drawn,
}

impl BoxRead {
    /// The grid an `ends` or `starts` edge's bins lie on.
    fn grid(&self) -> Option<Grid> {
        let bins = self.bins as f64;
        let high = match self.positions {
            Positions::Centres => return None,
            Positions::Ends => 1.0,
            Positions::Starts => (bins - 1.0) / bins,
        };
        Some(Grid::new(0.0, high, self.bins).expect("`box.bins` is at least 2"))
    }

    fn position(&self, bin: usize) -> f64 {
        match self.grid() {
            Some(grid) => grid.value(bin),
            None => (bin as f64 + 0.5) / self.bins as f64,
        }
    }

    /// The distance between neighbouring positions.
    fn step(&self) -> f64 {
        self.grid()
            .map_or(1.0 / self.bins as f64, |grid| grid.step())
    }

    fn edge(&self, probabilities: &[f64]) -> f64 {
        match self.edge {
            Edge::Mean => {
                let total: f64 = probabilities.iter().sum();
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(bin, probability)| probability * self.position(bin))
                    .sum::<f64>()
                    / total
            }
            Edge::Median => self.position(
                Cumulative::new(probabilities, fsum(probabilities.iter().copied())).reaching(0.5),
            ),
        }
    }

    /// Two edges of one axis, the second after the first.
    fn ordered(&self, a: f64, b: f64) -> (f64, f64) {
        if a < b {
            return (a, b);
        }
        let half = self.step() / 2.0;
        let middle = (a + b) / 2.0;
        match self.crossed {
            Crossed::Shift => {
                let middle = (1.0 - half).min(half.max(middle));
                (middle - half, middle + half)
            }
            Crossed::Clamp => {
                let (low, high) = ((middle - half).max(0.0), (middle + half).min(1.0));
                if low < high {
                    (low, high)
                } else if high >= 1.0 {
                    ((high - 2.0 * half).max(0.0), high)
                } else {
                    (low, low + 2.0 * half)
                }
            }
        }
    }

    fn rectangle(&self, [top, bottom, left, right]: [f64; 4]) -> Rect {
        let (top, bottom) = self.ordered(top, bottom);
        let (left, right) = self.ordered(left, right);
        Rect {
            top,
            bottom,
            left,
            right,
        }
    }

    /// The box the edge distributions (top, bottom, left, right) read.
    pub fn rect(&self, edges: &[Vec<f64>]) -> Rect {
        self.rectangle(std::array::from_fn(|edge| self.edge(&edges[edge])))
    }

    fn points(&self) -> Vec<[f64; 4]> {
        match self.points {
            Points::R4 => {
                let mut phi = 1.0f64;
                for _ in 0..64 {
                    phi = (1.0 + phi).powf(1.0 / 5.0);
                }
                let alphas: [f64; 4] =
                    std::array::from_fn(|d| (1.0 / phi).powi(d as i32 + 1).rem_euclid(1.0));
                (0..self.draws)
                    .map(|i| {
                        std::array::from_fn(|d| (0.5 + alphas[d] * (i + 1) as f64).rem_euclid(1.0))
                    })
                    .collect()
            }
            Points::Halton => {
                let inverse = |mut index: usize, base: usize| {
                    let (mut value, mut scale) = (0.0f64, 1.0f64);
                    while index > 0 {
                        let digit = index % base;
                        index /= base;
                        scale /= base as f64;
                        value += digit as f64 * scale;
                    }
                    value
                };
                (1..=self.draws)
                    .map(|index| [2, 3, 5, 7].map(|base| inverse(index, base)))
                    .collect()
            }
        }
    }

    /// The share of boxes drawn through the edges' cumulative
    /// distributions whose overlap with `rect` reaches `iou`.
    pub fn confidence(&self, edges: &[Vec<f64>], rect: &Rect) -> f64 {
        let cdfs: Vec<Cumulative> = edges
            .iter()
            .map(|row| Cumulative::new(row, fsum(row.iter().copied())))
            .collect();
        let points = self.points();
        let hits = points
            .iter()
            .filter(|point| {
                let drawn: [f64; 4] = std::array::from_fn(|edge| {
                    // The first bin whose cumulative share reaches the point.
                    self.position(cdfs[edge].reaching(point[edge]))
                });
                let (drawn, valid) = match self.drawn {
                    Drawn::Ordered => (self.rectangle(drawn), true),
                    Drawn::Valid => {
                        let [top, bottom, left, right] = drawn;
                        (
                            Rect {
                                top,
                                bottom,
                                left,
                                right,
                            },
                            top < bottom && left < right,
                        )
                    }
                };
                valid && iou(&drawn, rect) >= self.iou
            })
            .count();
        hits as f64 / points.len() as f64
    }
}

/// A box with its presence: P(present) through `found`, and the box drawn,
/// its confidence through `confidence`, when that is at least one half.
pub fn present(
    read: &BoxRead,
    presence: f64,
    edges: &[Vec<f64>],
    found: &Table,
    confidence: &Table,
) -> Answer {
    let found = found.through(presence);
    if found < 0.5 {
        return Answer::Box {
            found: Some(found),
            rect: None,
            confidence: 0.0,
        };
    }
    let rect = read.rect(edges);
    Answer::Box {
        found: Some(found),
        confidence: confidence.through(read.confidence(edges, &rect)),
        rect: Some(rect),
    }
}

/// Every instance from its blocks, each a P(present) and four edges:
/// presence through `found` is kept falling block by block (P(count > k)),
/// the boxes are the blocks before the first below one half, in reading
/// order (left edge, then top), each confident through `confidence`.
pub fn every(
    read: &BoxRead,
    blocks: &[(f64, Vec<Vec<f64>>)],
    found: &Table,
    confidence: &Table,
) -> Answer {
    let mut below = 1.0f64;
    let presence: Vec<f64> = blocks
        .iter()
        .map(|(present, _)| {
            below = below.min(found.through(*present));
            below
        })
        .collect();
    let mut boxes: Vec<FoundBox> = blocks
        .iter()
        .zip(&presence)
        .take_while(|(_, found)| **found >= 0.5)
        .map(|((_, edges), found)| {
            let rect = read.rect(edges);
            FoundBox {
                rect,
                found: *found,
                confidence: confidence.through(read.confidence(edges, &rect)),
            }
        })
        .collect();
    boxes.sort_by(|a, b| {
        (a.rect.left, a.rect.top)
            .partial_cmp(&(b.rect.left, b.rect.top))
            .expect("edges are finite")
    });
    Answer::Boxes {
        at_least: !boxes.is_empty() && boxes.len() == presence.len(),
        boxes,
        presence,
    }
}

fn area(rect: &Rect) -> f64 {
    (rect.bottom - rect.top).max(0.0) * (rect.right - rect.left).max(0.0)
}

fn iou(a: &Rect, b: &Rect) -> f64 {
    let overlap = (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0.0)
        * (a.right.min(b.right) - a.left.max(b.left)).max(0.0);
    let union = area(a) + area(b) - overlap;
    if union > 0.0 {
        overlap / union
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(positions: Positions, crossed: Crossed) -> BoxRead {
        BoxRead {
            bins: 4,
            positions,
            edge: Edge::Mean,
            points: Points::R4,
            draws: 64,
            iou: 0.5,
            crossed,
            drawn: Drawn::Valid,
        }
    }

    #[test]
    fn positions_and_crossed_edges_follow_their_rules() {
        let centres = read(Positions::Centres, Crossed::Clamp);
        assert_eq!(
            (0..4).map(|bin| centres.position(bin)).collect::<Vec<_>>(),
            [0.125, 0.375, 0.625, 0.875]
        );
        let ends = read(Positions::Ends, Crossed::Shift);
        assert_eq!(ends.position(3), 1.0);
        assert_eq!(read(Positions::Starts, Crossed::Shift).position(2), 0.5);
        // d3 cuts a bin at the side (regrowing it only when that empties
        // it); d1 shifts it inside.
        assert_eq!(centres.ordered(0.0, 0.0), (0.0, 0.125));
        assert_eq!(centres.ordered(1.0, 1.0), (0.875, 1.0));
        assert_eq!(centres.ordered(0.5, 0.25), (0.25, 0.5));
        let (low, high) = ends.ordered(0.0, 0.0);
        assert!((low - 0.0).abs() < 1e-12 && (high - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_sharp_box_reads_its_bins_with_full_confidence() {
        let one_hot =
            |bin: usize| -> Vec<f64> { (0..4).map(|k| f64::from(u8::from(k == bin))).collect() };
        for drawn in [Drawn::Valid, Drawn::Ordered] {
            for (edge, points) in [(Edge::Mean, Points::R4), (Edge::Median, Points::Halton)] {
                let read = BoxRead {
                    edge,
                    points,
                    drawn,
                    ..read(Positions::Centres, Crossed::Clamp)
                };
                let edges = [one_hot(0), one_hot(2), one_hot(1), one_hot(3)];
                let rect = read.rect(&edges);
                assert_eq!(
                    rect,
                    Rect {
                        top: 0.125,
                        bottom: 0.625,
                        left: 0.375,
                        right: 0.875
                    }
                );
                assert_eq!(read.confidence(&edges, &rect), 1.0);
            }
        }
    }
}
