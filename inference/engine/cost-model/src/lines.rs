//! The lines a streamed expert's reads and landings take: the model store's
//! read line ([`ReadLine`]), a per-record line ([`RecordLine`]) and the
//! host's landing line ([`FileLine`]). Each is a line through measured
//! points (`seismic::measure::Sums::fit`), of no negative term by
//! construction ([`Terms`]); measuring them is the executor's.

use seismic_measure::{InvalidLine, Terms};

/// A read line: a read of `b` bytes takes `λ + b / D`, `λ` (its latency)
/// not negative and `1/D` (its seconds a byte) not negative, by
/// construction ([`Self::new`]): a per-byte cost of zero is a path whose
/// reads grow no slower with their bytes, as reads served from memory do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ReadTerms", into = "ReadTerms")]
pub struct ReadLine(Terms<2>);

/// A read line as it is recorded.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct ReadTerms {
    latency_seconds: f64,
    seconds_per_byte: f64,
}

impl TryFrom<ReadTerms> for ReadLine {
    type Error = InvalidLine;
    fn try_from(terms: ReadTerms) -> Result<Self, InvalidLine> {
        Self::new(terms.latency_seconds, terms.seconds_per_byte)
    }
}

impl From<ReadLine> for ReadTerms {
    fn from(line: ReadLine) -> Self {
        Self { latency_seconds: line.latency_seconds(), seconds_per_byte: line.seconds_per_byte() }
    }
}

impl ReadLine {
    /// The line of latency `latency_seconds` and `seconds_per_byte`, or why
    /// they make none.
    pub const fn new(latency_seconds: f64, seconds_per_byte: f64) -> Result<Self, InvalidLine> {
        match Terms::new([latency_seconds, seconds_per_byte]) {
            Ok(terms) => Ok(Self(terms)),
            Err(invalid) => Err(invalid),
        }
    }

    /// `λ`.
    pub const fn latency_seconds(&self) -> f64 {
        self.0.fixed()
    }

    /// `1/D`.
    pub const fn seconds_per_byte(&self) -> f64 {
        self.0.slope()
    }

    /// The transfer term of a read of `bytes` bytes, `b/D`, beside its
    /// latency `λ`; fractional bytes are an expectation's.
    pub fn transfer_seconds(&self, bytes: f64) -> f64 {
        bytes * self.seconds_per_byte()
    }

    /// What a read of `bytes` stored bytes costs, `λ + b/D`: the one price
    /// of reading experts from the file, which a decode layer's and a prompt
    /// group's file misses and a stored table's misses take.
    pub fn seconds(&self, bytes: u64) -> f64 {
        self.latency_seconds() + self.transfer_seconds(bytes as f64)
    }
}

/// A per-record line: `n` records take `fixed_seconds + n · record_seconds`,
/// neither term negative, by construction ([`Self::new`]).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "RecordTerms", into = "RecordTerms")]
pub struct RecordLine(Terms<2>);

/// A per-record line as it is recorded.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct RecordTerms {
    fixed_seconds: f64,
    record_seconds: f64,
}

impl TryFrom<RecordTerms> for RecordLine {
    type Error = InvalidLine;
    fn try_from(terms: RecordTerms) -> Result<Self, InvalidLine> {
        Self::new(terms.fixed_seconds, terms.record_seconds)
    }
}

impl From<RecordLine> for RecordTerms {
    fn from(line: RecordLine) -> Self {
        Self { fixed_seconds: line.0.fixed(), record_seconds: line.0.slope() }
    }
}

impl RecordLine {
    /// The line of `fixed_seconds` and `record_seconds`, or why they make
    /// none.
    pub const fn new(fixed_seconds: f64, record_seconds: f64) -> Result<Self, InvalidLine> {
        match Terms::new([fixed_seconds, record_seconds]) {
            Ok(terms) => Ok(Self(terms)),
            Err(invalid) => Err(invalid),
        }
    }

    /// The line of `terms`, a fixed part and a record's, as a fit gives
    /// them.
    pub fn of(terms: Terms<2>) -> Self {
        Self(terms)
    }

    /// The fixed part and a record's, as recorded.
    pub fn terms(&self) -> Terms<2> {
        self.0
    }

    /// The seconds of `records` records.
    pub fn seconds(&self, records: u64) -> f64 {
        self.0.at([1., records as f64])
    }
}

/// What a layer's file misses cost beside their read (`λ + b/D`, the read
/// line's) as the host lands them into a host tier: each miss is a read of
/// its own, whose end submits its landing to the host compute executor,
/// whose workers convert its pieces (`RowConversion`) beside the other
/// landings', then map and answer it. A request's `n` landings end `fixed +
/// n · record` after its last read's end: observed from a load's requests'
/// own landings (`Telemetry::landings_ended`) on the one generic line. Not
/// in it: the park's notice of the answer, an in-kernel poll of a host word
/// where the park spins (one link read), unmeasured where a park is a
/// submission boundary.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FileLine {
    pub landing: RecordLine,
}

impl FileLine {
    /// The line of `terms`, a fixed part and a record's.
    pub fn of(terms: Terms<2>) -> Self {
        Self { landing: RecordLine::of(terms) }
    }

    /// What `misses` file misses of a layer add past their reads' end.
    pub fn seconds(&self, misses: u64) -> f64 {
        self.landing.seconds(misses)
    }
}
