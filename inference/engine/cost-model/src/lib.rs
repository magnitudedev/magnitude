//! The cost model: what every launch of a loaded model costs on this
//! machine, from plain data alone, so pricing has one owner.
//!
//! - **The census** ([`census`]): one launch's calls by kind, each kind's
//!   work and launches, and its multiplicity: the cost model's input
//!   contract.
//! - **Launches** ([`compute`]): a census's launches priced kind by kind by
//!   Seismic's per-call formula ([`seismic_measure::pricing`]), and a
//!   routed pass over a streamed layer's experts ([`ExpertPass`]).
//! - **Launch prices** ([`launch`]): a load's launches in each form it
//!   decodes in ([`LaunchPrices`]).
//! - **Misses** ([`expert_lines`], `miss_model`, [`step`], [`streamed`]):
//!   where a missed expert is computed ([`ExpertLines::serve`], also the
//!   route's live assignment rule), the generic miss model, the one step
//!   model, and a streamed plan's decode step over them
//!   ([`StreamedLayers`]).
//! - **Bank prices** ([`bank`]): what a plan prices its bank's slots and
//!   its prompt passes with under the workload prior ([`BankPrices`]).
//! - **Lines** ([`lines`]): the read, record and landing lines the misses
//!   take.
//! - **Terms** ([`terms`]): a streamed decode token's busy seconds on each
//!   resource, the GPU's pass, every path and every in-place tier
//!   ([`Terms`]), which every streamed price reads.
//! - **The regime** ([`regime`]): the form and depth of a streamed decode
//!   pass, front rows or verify rows ([`Form`]), whichever takes least a
//!   token emitted, from its resident pass at each depth, the misses its
//!   ranking makes at each depth of foresight, every other term's seconds a
//!   miss and moves nothing used, the stall of misses nothing named in
//!   time, the verify rows' stand rates and the host turn a verify pass
//!   leaves the GPU idle ([`Regime`]).

pub mod bank;
pub mod census;
pub mod compute;
pub mod expert_lines;
pub mod launch;
pub mod lines;
mod miss_model;
pub mod regime;
pub mod step;
pub mod streamed;
pub mod terms;

pub use bank::{BankPrices, PassLayers, PathRecords, Prior, PriorOverflow, WorkloadRates};
pub use census::{Census, Kind, LayerCall, LaunchCounts, RoutedCall, RoutedKind};
pub use compute::{step_seconds, ExpertPass, HostReads};
pub use expert_lines::{
    CpuLines, CpuLinesError, Exposed, ExpertLines, GpuLines, Hits, HostLines, LandingLine, LineError, Miss, Moves, Nanoseconds, Pass, PassLine,
    PolicyLayout, RowCost, RowLine, Sample, ServedMisses, Site, SiteLines,
};
pub use launch::{FrontRoutes, LaunchPrices, TargetCensus};
pub use lines::{FileLine, ReadLine, RecordLine};
pub use regime::{Form, Needed, NeededByForm, PathTerm, ReadAdmission, Regime, VerifyRoutes, Waste};
pub use step::{LayerLoad, LayerRun, Lead, ReadAhead, StepInputs, StepTime};
pub use terms::Terms;
pub use streamed::{step_inputs, DecodePrice, FormSeconds, HeldSlots, LayerReads, LiveDecode, StreamedLayers, StreamedStep};

#[cfg(test)]
mod tests;
