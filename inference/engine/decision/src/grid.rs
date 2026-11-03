//! Grids: ordered values equally spaced from a low to a high one, read as a
//! distribution over them (an estimate's values, a box edge's bins).

use crate::answers::fsum;

/// `size` equally spaced values from `low` to `high`, two or more, `low`
/// below `high`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    low: f64,
    high: f64,
    size: usize,
}

/// The central 80 percent interval and the median a grid reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantiles {
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
}

impl Grid {
    pub fn new(low: f64, high: f64, size: usize) -> Result<Self, String> {
        if size < 2 || !(low < high) {
            return Err("a grid needs two values or more, low to high".into());
        }
        Ok(Self { low, high, size })
    }

    pub fn low(&self) -> f64 {
        self.low
    }

    pub fn high(&self) -> f64 {
        self.high
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn step(&self) -> f64 {
        (self.high - self.low) / (self.size - 1) as f64
    }

    /// Value `index`: `low + index * step`, the last `high` itself.
    pub fn value(&self, index: usize) -> f64 {
        if index + 1 == self.size {
            self.high
        } else {
            self.low + index as f64 * self.step()
        }
    }

    pub fn values(&self) -> Vec<f64> {
        (0..self.size).map(|index| self.value(index)).collect()
    }

    /// The values where the summed share first reaches each quantile's
    /// `level` (`Cumulative::reaching`).
    pub fn read(&self, probabilities: &[f64], level: impl Fn(f64) -> f64) -> Quantiles {
        let shares = Cumulative::new(probabilities, fsum(probabilities.iter().copied()));
        let [p10, median, p90] = [0.1, 0.5, 0.9].map(|q| self.value(shares.reaching(level(q))));
        Quantiles { p10, median, p90 }
    }
}

/// A distribution's running shares of its `total`: entry `i` is the share of
/// its first `i + 1` probabilities, summed in order.
pub struct Cumulative(Vec<f64>);

impl Cumulative {
    /// The running shares of `probabilities` (nonempty, none negative) of
    /// `total`, which a reference sums its own way.
    pub fn new(probabilities: &[f64], total: f64) -> Self {
        let mut mass = 0.0;
        Self(
            probabilities
                .iter()
                .map(|probability| {
                    mass += probability / total;
                    mass
                })
                .collect(),
        )
    }

    /// The first index whose share reaches `q`; the last when rounding
    /// leaves every share short of it. The shares never fall, so the first
    /// is found by bisection.
    pub fn reaching(&self, q: f64) -> usize {
        self.0
            .partition_point(|mass| *mass < q)
            .min(self.0.len() - 1)
    }
}
