//! What the operations of a load have taken, priced from their own
//! observations: the classes tuning and formation admit work by before it
//! starts.

use std::time::Duration;

/// What one class of operation has taken in this load: its wall times, in
/// seconds, or what it added to the bytes held, in bytes. A class is priced
/// only from its own observations: their mean. A class never observed has
/// no price.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Observed {
    seconds: f64,
    count: usize,
    longest: f64,
}

impl Observed {
    /// One more operation of the class, which took `seconds` (or added
    /// that many bytes).
    pub fn observe(&mut self, seconds: f64) {
        self.seconds += seconds;
        self.count += 1;
        self.longest = self.longest.max(seconds);
    }

    /// The class's price: the mean of its observations.
    pub fn price(&self) -> Option<f64> {
        (self.count > 0).then(|| self.seconds / self.count as f64)
    }

    /// The longest single operation of the class.
    pub fn longest(&self) -> f64 {
        self.longest
    }

    /// Everything the class's operations have taken.
    pub fn total(&self) -> f64 {
        self.seconds
    }
}

/// What calls of one kind have taken in this load, each with how many units
/// of work it did (a compile call and the entries it formed): a cost of the
/// call itself and a cost of each unit, the least-squares line through the
/// observations. A call of the toolchain pays for its start, its parse of
/// the source and its link whatever it forms, and for each entry's code
/// besides: one mean over calls of different sizes prices a small call as a
/// large one (NVRTC, 2026-10-10: compiles of 3 to 48 entries took 0.3 to
/// 4.3 s in one load).
///
/// Until calls of two sizes are observed the line cannot be told from a
/// constant, and the price is the mean; neither term is ever negative.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Linear {
    count: f64,
    units: f64,
    seconds: f64,
    units_squared: f64,
    units_seconds: f64,
    longest: f64,
}

impl Linear {
    /// One more call, which did `units` units of work in `seconds`.
    pub fn observe(&mut self, units: usize, seconds: f64) {
        let units = units as f64;
        self.count += 1.;
        self.units += units;
        self.seconds += seconds;
        self.units_squared += units * units;
        self.units_seconds += units * seconds;
        self.longest = self.longest.max(seconds);
    }

    /// The cost of a call and of each unit; none before an observation.
    fn terms(&self) -> Option<(f64, f64)> {
        if self.count == 0. {
            return None;
        }
        let mean = self.seconds / self.count;
        let spread = self.count * self.units_squared - self.units * self.units;
        // Sizes that differ by less than rounding are one size.
        if spread <= self.units_squared * f64::EPSILON * self.count {
            return Some((mean, 0.));
        }
        let each = (self.count * self.units_seconds - self.units * self.seconds) / spread;
        let call = (self.seconds - each * self.units) / self.count;
        Some(if each < 0. {
            (mean, 0.)
        } else if call < 0. {
            (0., self.units_seconds / self.units_squared)
        } else {
            (call, each)
        })
    }

    /// The price of a call that does `units` units.
    pub fn price(&self, units: usize) -> Option<f64> {
        self.terms().map(|(call, each)| call + each * units as f64)
    }

    /// The cost of each unit, once calls of two sizes were observed.
    pub fn each(&self) -> Option<f64> {
        self.terms().map(|(_, each)| each).filter(|each| *each > 0.)
    }

    /// The mean call, and the mean of its units.
    pub fn mean(&self) -> Option<(f64, f64)> {
        (self.count > 0.).then(|| (self.seconds / self.count, self.units / self.count))
    }

    /// The longest single call.
    pub fn longest(&self) -> f64 {
        self.longest
    }

    /// Everything the calls have taken.
    pub fn total(&self) -> f64 {
        self.seconds
    }
}

/// The bytes a whole piece of work adds, at its most, to what the load
/// holds when it starts: priced like its seconds ([`Quote`]), as the sum of
/// its operations' class prices and how many of its operations belong to a
/// class this load has not observed. A piece that needs nothing more than
/// is held already has a footprint of nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Footprint {
    pub priced: u64,
    pub unpriced: usize,
}

impl Footprint {
    /// Add `times` operations of a class priced `price` bytes.
    pub fn add(&mut self, price: Option<f64>, times: usize) {
        match price {
            Some(bytes) => {
                self.priced = self
                    .priced
                    .saturating_add((bytes * times as f64).ceil() as u64)
            }
            None => self.unpriced += times,
        }
    }

    /// The footprint of this work and `other` together.
    pub fn and(mut self, other: Footprint) -> Footprint {
        self.priced = self.priced.saturating_add(other.priced);
        self.unpriced += other.unpriced;
        self
    }

    /// What the work is taken to add: its priced bytes, with each unpriced
    /// operation at `largest`, the most a single piece of work of this load
    /// has added.
    pub fn forecast(&self, largest: u64) -> u64 {
        u64::try_from(self.unpriced)
            .ok()
            .and_then(|unpriced| largest.checked_mul(unpriced))
            .map_or(u64::MAX, |unpriced| self.priced.saturating_add(unpriced))
    }

    /// The admission rule for bytes, the one for seconds ([`Quote::fits`])
    /// over again. Work starts only when what it is taken to add fits
    /// `free`, the bytes its device's memory plan leaves for it. A device
    /// without a plan (`None`) bounds nothing: what is known of it is that
    /// no limit is enforced, and no number is put in its place.
    pub fn fits(&self, free: Option<u64>, largest: u64) -> bool {
        free.is_none_or(|free| self.forecast(largest) <= free)
    }
}

/// The operation classes of one point, each observed as wall time: placing
/// a kernel's calls there and restoring its state, one observed invocation
/// of one argument set (a numerical witness), one timed submission (a
/// calibrating pass or a sample), and warming the device with its calls.
/// Device medians remain the performance objective, never a price.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PointPrices {
    pub placement: Observed,
    pub witness: Observed,
    pub sample: Observed,
    pub warm: Observed,
}

impl PointPrices {
    /// The longest single operation of any of the point's classes.
    pub fn longest(&self) -> f64 {
        [self.placement, self.witness, self.sample, self.warm]
            .iter()
            .map(Observed::longest)
            .fold(0., f64::max)
    }
}

/// The price of a whole piece of work before it starts: the sum of its
/// operations' class prices, and how many of its operations belong to a
/// class this load has not observed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quote {
    pub priced: Duration,
    pub unpriced: usize,
}

impl Quote {
    /// Add `times` operations of one class, priced `price`. A class with
    /// no price is one thing not known, however many of its operations the
    /// work has: what is not known is what an operation of the class takes,
    /// once, and an unknown for each operation held a piece of seven
    /// challengers to seven times the longest operation of any kind the
    /// load had seen (a 4.3 s compile) and shut units that had not started
    /// out of the last 23 s of a budget.
    pub fn add(&mut self, price: Option<f64>, times: usize) {
        match price {
            Some(seconds) => self.priced += Duration::from_secs_f64(seconds * times as f64),
            None => self.unpriced += usize::from(times > 0),
        }
    }

    /// Add operations of a class priced `price` whose number is not known
    /// before they run, at the number `expected` of them.
    pub fn expect(&mut self, price: Option<f64>, expected: f64) {
        match price {
            Some(seconds) => self.priced += Duration::from_secs_f64(seconds * expected),
            None => self.unpriced += usize::from(expected > 0.),
        }
    }

    /// The price of this work and `other` together.
    pub fn and(mut self, other: Quote) -> Quote {
        self.priced += other.priced;
        self.unpriced += other.unpriced;
        self
    }

    /// The forecast of the work's wall time; none while any of it is
    /// unpriced.
    pub fn forecast(&self) -> Option<Duration> {
        (self.unpriced == 0).then_some(self.priced)
    }

    /// The one admission rule. Work starts only when its priced operations
    /// fit the remaining budget; an unpriced operation is late, and its
    /// work is not started, once what remains after the priced operations
    /// is less than `longest`, the longest single operation this load has
    /// completed, for each unpriced operation: the load then holds direct
    /// evidence that an operation can outlast what is left, and none that
    /// this one will not.
    pub fn fits(&self, remaining: Duration, longest: Duration) -> bool {
        u32::try_from(self.unpriced)
            .ok()
            .and_then(|unpriced| longest.checked_mul(unpriced))
            .and_then(|unpriced| self.priced.checked_add(unpriced))
            .is_some_and(|needed| needed <= remaining)
    }
}
