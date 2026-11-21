//! A streamed plan's layers as the cost model prices them
//! ([`magnitude_cost_model::StreamedLayers`]), from the plan's structure and
//! the machine's stored figures ([`StreamedFigures`]): no model's own
//! measurement enters them, so a model never loaded is estimated as one
//! loaded many times.

use super::cost::{host_lines, BankLines, HostLinesError};
use super::executor_costs::ExecutorCosts;
use super::priced::Priced;
use super::requests::{Outstanding, PendingPrices};
use super::streaming_costs::{ConvertCosts, ConvertForm, CpuCosts, CpuForm, MemoryCosts, StorageCosts};
use crate::planning::ExpertBankPlan;
use magnitude_cost_model::{CpuLines, ExpertPass, HostLines, LandingLine, ReadLine, RecordLine, StreamedLayers};
use seismic::measure::{InvalidLine, Record, RecordStore, StoreError};
use seismic::{DeviceTopology, Executor, PointKey};
use std::cell::RefCell;
use std::fmt;
use std::path::Path;

/// What a streamed estimate reads beside the executor's costs, each a
/// standing request of the measurement job (`requests::standing`): the
/// model store's read line, where a bank can have a host tier, the host
/// memory's seconds a byte with every CPU reading and a CPU site's lines
/// at its reference form (`CpuForm::reference`), and a stored byte's
/// seconds through the device's convert (`ConvertForm::reference`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamedFigures {
    pub read: ReadLine,
    pub memory: Option<f64>,
    pub site: Option<(CpuForm, CpuLines)>,
    pub convert: f64,
}

impl StreamedFigures {
    /// The figures `store`'s records give `executor` of the model store at
    /// `storage`, measuring nothing, with the records they were read from:
    /// pending while the job has not measured them, unavailable where they
    /// cannot be had.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        storage: &Path,
        store: &dyn RecordStore<PointKey>,
    ) -> (Priced<Self>, Vec<(PointKey, Record)>) {
        let reading = Reading { store, read: RefCell::default() };
        let read = StorageCosts::priced(topology, executor, storage, &reading);
        let readers = MemoryCosts::readers(topology);
        let memory = match topology.shared_ranges_lie_in_device_memory(executor) {
            true => Priced::Measured(None),
            false => MemoryCosts::priced(topology, executor, readers, &reading).and_then(|costs| Priced::Measured(Some(costs.byte_seconds))),
        };
        let site = match topology.shared_ranges_lie_in_device_memory(executor) {
            true => Priced::Measured(None),
            false => match CpuForm::reference(executor.backend()) {
                Err(error) => Priced::Unavailable(error.to_string()),
                Ok(None) => Priced::Unavailable(format!("{:?} streams nothing", executor.backend())),
                Ok(Some(form)) => CpuCosts::priced(topology, executor, &form, &reading).and_then(|costs| Priced::Measured(Some((form, costs.lines)))),
            },
        };
        let convert = match ConvertForm::reference(executor.backend()) {
            Err(error) => Priced::Unavailable(error.to_string()),
            Ok(None) => Priced::Unavailable(format!("{:?} streams nothing", executor.backend())),
            Ok(Some(form)) => ConvertCosts::priced(topology, executor, form, &reading).derive(|costs| form.stored_byte_seconds(costs.record_seconds)),
        };
        let figures = match (read, memory, site, convert) {
            (Priced::Measured(read), Priced::Measured(memory), Priced::Measured(site), Priced::Measured(convert)) => {
                Priced::Measured(Self { read, memory, site, convert })
            }
            (read, memory, site, convert) => {
                let mut standing = Vec::new();
                let mut unavailable = None;
                for state in [
                    read.and_then(|_| Priced::Measured(())),
                    memory.and_then(|_| Priced::Measured(())),
                    site.and_then(|_| Priced::Measured(())),
                    convert.and_then(|_| Priced::Measured(())),
                ] {
                    match state {
                        Priced::Measured(()) => {}
                        Priced::Pending(pending) => standing.extend(pending.outstanding().standing.iter().cloned()),
                        Priced::Unavailable(reason) => {
                            unavailable.get_or_insert(reason);
                        }
                    }
                }
                match unavailable {
                    Some(reason) => Priced::Unavailable(reason),
                    None => Priced::Pending(
                        PendingPrices::new(Outstanding { standing, loads: Vec::new() })
                            .expect("non-measured, non-unavailable streamed figures must have pending requests"),
                    ),
                }
            }
        };
        (figures, reading.read.into_inner())
    }
}

/// A store whose loads are kept: what a figure was read from.
struct Reading<'a> {
    store: &'a dyn RecordStore<PointKey>,
    read: RefCell<Vec<(PointKey, Record)>>,
}

impl RecordStore<PointKey> for Reading<'_> {
    fn load(&self, key: &PointKey) -> Result<Option<Record>, StoreError> {
        let record = self.store.load(key)?;
        if let Some(record) = &record {
            self.read.borrow_mut().push((key.clone(), record.clone()));
        }
        Ok(record)
    }

    fn store(&self, _: &PointKey, _: &Record) -> Result<(), StoreError> {
        unreachable!("a figure's reading stores nothing")
    }
}

/// Why a streamed plan's layers have no prices: their host lines' or their
/// passes' figures, which no measurement brings.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamedError {
    Host(HostLinesError),
    Costs(String),
    /// A landing's line from the convert's figure.
    Landing(InvalidLine),
}

impl fmt::Display for StreamedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host(error) => error.fmt(formatter),
            Self::Costs(error) => write!(formatter, "the streamed layers' passes: {error}"),
            Self::Landing(error) => write!(formatter, "a landing's line: {error}"),
        }
    }
}

impl std::error::Error for StreamedError {}

/// `bank`'s layers on the executor `costs` price, a host tier's record read
/// at `figures`' memory line: pending while the costs have not read host
/// storage's routes, the link line's (`cost::expert_lines`).
pub fn streamed_layers(bank: &ExpertBankPlan, costs: &ExecutorCosts, figures: &StreamedFigures) -> Priced<StreamedLayers> {
    let parts = || -> Result<(ExpertPass, Vec<HostLines>), StreamedError> {
        let pass = super::estimate::expert_pass(bank, costs).map_err(|error| StreamedError::Costs(error.to_string()))?;
        let byte_seconds = figures.memory.filter(|_| bank.tiered());
        let host = host_lines(bank, &BankLines::of_bank(bank).map_err(StreamedError::Host)?, byte_seconds).map_err(StreamedError::Host)?;
        let host = host
            .iter()
            .enumerate()
            .map(|(layer, &host)| {
                let cpu = match figures.site.filter(|_| bank.sited(layer)) {
                    Some((form, lines)) => Some(form.lines_of(lines, host.record_bytes).map_err(|error| StreamedError::Costs(error.to_string()))?),
                    None => None,
                };
                let landing = landing(&host, figures.convert).map_err(StreamedError::Landing)?;
                Ok(HostLines { cpu, landing, ..host })
            })
            .collect::<Result<Vec<_>, StreamedError>>()?;
        Ok((pass, host))
    };
    match parts() {
        Ok((pass, host)) => super::cost::expert_lines(&host, &pass)
            .and_then(|lines| Priced::Measured(StreamedLayers::new(bank.layers().iter().map(|layer| layer.experts).collect(), bank.selected(), lines))),
        Err(error) => Priced::Unavailable(error.to_string()),
    }
}

/// What a request's file misses of a layer of `host`'s lines take past
/// their reads' end: with no host tier the device's landing launch converts
/// each into its bank record as its read lands, so past the last read only
/// the last record's convert remains, its stored bytes at `byte_seconds`;
/// beside a tier the host lands them, at a line only a load observes.
fn landing(host: &HostLines, byte_seconds: f64) -> Result<Option<LandingLine>, InvalidLine> {
    match host.memory {
        Some(_) => Ok(None),
        None => RecordLine::new(0., host.record_bytes as f64 * byte_seconds).map(|convert| Some(LandingLine::Device(convert))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_cost_model::Nanoseconds;

    /// With no host tier a request's file misses land through the device's
    /// convert, past their last read one record's, however many they are
    /// while each read takes longer than a convert (here 1 s against
    /// 0.2 ms); beside a tier the host lands them, unpriced before a load observes it.
    #[test]
    fn a_file_miss_lands_through_one_records_convert_with_no_host_tier() {
        let host = HostLines { cpu: None, memory: None, landing: None, record_bytes: 2_000_000 };
        let line = landing(&host, 1e-10).unwrap().expect("no tier: the device lands it");
        for count in [1, 3] {
            assert!((line.seconds(count, 1.) - 2e-4).abs() < 1e-12, "{count} misses");
        }
        let tier = HostLines { memory: Some(Nanoseconds::saturating(1.)), ..host };
        assert_eq!(landing(&tier, 1e-10).unwrap(), None);
    }

}
