//! The executor's side of the cost function of where a missed expert is
//! computed ([`magnitude_cost_model::ExpertLines`]): a bank's lines from its
//! plan ([`BankLines::of_bank`], [`host_lines`]), the lines a load's
//! measured host lines and its kernels' costs price ([`expert_lines`]), the
//! CPU sites a load read from the records ([`StoredCpuSite`]), and what a
//! host tier's copy costs against the reads it saves ([`copy_hits`]).

use super::priced::Priced;
use crate::planning::ExpertBankPlan;
use magnitude_cost_model::{CpuLines, ExpertLines, ExpertPass, HostLines, LineError, Nanoseconds, PolicyLayout};
use magnitude_kernels::constants;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Where the route's policy words lie, as the kernels' sources declare them
/// (`streaming_words.seismic`): what the cost function writes the lines at
/// ([`ExpertLines::policy`]).
pub const POLICY_LAYOUT: PolicyLayout = PolicyLayout {
    words: constants::POLICY_WORDS as usize,
    cpu: constants::POLICY_CPU_NS as usize,
    cpu_row: constants::POLICY_CPU_ROW_NS as usize,
    handshake: constants::POLICY_HANDSHAKE_NS as usize,
    link: constants::POLICY_LINK_NS as usize,
    link_row: constants::POLICY_LINK_ROW_NS as usize,
    link_busy: constants::POLICY_LINK_BUSY_NS as usize,
    hit: constants::POLICY_HIT_NS as usize,
    hit_row: constants::POLICY_HIT_ROW_NS as usize,
    priced: constants::POLICY_PRICED as usize,
};

/// What a load read of its CPU sites from the records (`CpuCosts`): the
/// workers the job sized its pool to, and the lines of each streamed layer
/// that has them stored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoredCpuSite {
    pub workers: Option<usize>,
    pub lines: std::collections::BTreeMap<usize, CpuLines>,
}

/// What the bank holds of a streamed layer's lines on the host's side: its
/// CPU pool's lines, none without a CPU site (unified memory, where the GPU
/// reads host memory at full speed); and one expert's stored bytes, what a
/// file miss reads.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BankLines {
    pub cpu: Option<CpuLines>,
    pub record_bytes: u64,
}

impl BankLines {
    /// Each streamed layer's lines as the bank opens, by its group
    /// (`plan`'s): one expert's stored bytes. The CPU's lines are measured
    /// where their pool is built.
    pub fn of_bank(plan: &ExpertBankPlan) -> Result<Arc<[Self]>, HostLinesError> {
        let groups = plan
            .groups()
            .iter()
            .enumerate()
            .map(|(group, members)| {
                let record_bytes = members
                    .record
                    .iter()
                    .map(|(_, tensor)| tensor.stored_bytes())
                    .sum::<Result<u64, _>>()
                    .map_err(|reason| HostLinesError::Record { group, reason })?;
                Ok(Self { cpu: None, record_bytes })
            })
            .collect::<Result<Vec<_>, HostLinesError>>()?;
        Ok(plan.layers().iter().map(|layer| groups[layer.group]).collect())
    }
}

/// Why a bank group's host lines could not be had.
#[derive(Clone, Debug, PartialEq)]
pub enum HostLinesError {
    /// A routed tensor's stored or resident record bytes of group `group`.
    Record { group: usize, reason: String },
    /// Group `group`'s record read from host memory past a policy word.
    Line { group: usize, cause: LineError },
}

impl fmt::Display for HostLinesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Record { group, reason } => write!(formatter, "bank group {group}'s record: {reason}"),
            Self::Line { group, cause } => write!(formatter, "bank group {group}'s record read from host memory: {cause}"),
        }
    }
}

impl std::error::Error for HostLinesError {}

/// Each streamed layer's host lines: the bank's (`bank`, in decoder order)
/// beside, where `plan`'s bank has a host tier, one tier record's read from
/// host memory at `byte_seconds` (`MemoryCosts`), which only a bank with a
/// tier has.
pub fn host_lines(plan: &ExpertBankPlan, bank: &[BankLines], byte_seconds: Option<f64>) -> Result<Arc<[HostLines]>, HostLinesError> {
    assert_eq!(
        byte_seconds.is_some(),
        plan.tiered(),
        "a host memory line prices exactly a bank with a host tier"
    );
    let memory = plan
        .groups()
        .iter()
        .enumerate()
        .map(|(group, members)| {
            byte_seconds
                .map(|byte_seconds| {
                    let bytes = members
                        .record
                        .iter()
                        .map(|(_, tensor)| tensor.record_bytes())
                        .sum::<Result<u64, _>>()
                        .map_err(|reason| HostLinesError::Record { group, reason })?;
                    Nanoseconds::of(bytes as f64 * byte_seconds).map_err(|cause| HostLinesError::Line { group, cause })
                })
                .transpose()
        })
        .collect::<Result<Vec<_>, HostLinesError>>()?;
    Ok(plan
        .layers()
        .iter()
        .zip(bank)
        .map(|(layer, bank)| HostLines { cpu: bank.cpu, memory: memory[layer.group], landing: None, record_bytes: bank.record_bytes })
        .collect())
}

/// Each streamed layer's lines: the host's the load measured (`host`, in
/// decoder order) and the GPU's its kernels' costs price (`pass`). A host
/// tier's link line is host storage's routes of the costs, which nothing
/// else stands in for: pending until the job reads them (a standing
/// request, `CostWalk::Whole`), and unavailable where a GPU line is past a
/// policy word.
pub fn expert_lines(host: &[HostLines], pass: &ExpertPass) -> Priced<Arc<[ExpertLines]>> {
    let lines = host
        .iter()
        .enumerate()
        .map(|(layer, &host)| pass.lines(layer).map(|gpu| gpu.map(|gpu| ExpertLines { host, gpu })))
        .collect::<Option<Result<Arc<[ExpertLines]>, LineError>>>();
    match lines {
        None => match pass.host_unobtainable() {
            Some(unread) => Priced::Unavailable(format!("the executor's costs cannot read host storage, the link line's: {unread}")),
            None => Priced::Pending(super::PendingPrices::standing(super::PointRequest::Costs(super::CostWalk::Whole))),
        },
        Some(Ok(lines)) => Priced::Measured(lines),
        Some(Err(error)) => Priced::Unavailable(format!("a GPU line: {error}")),
    }
}

/// The shares of the decode's time a promotion's copy runs beside, each the
/// busy seconds of a resource the copy shares over the decode passes'
/// seconds (`watcher::passes::Passes::beside`): the CPU site's (`cpu`),
/// which reads host memory as the copy does, and the link's reads in place
/// (`link`), which cross the link as the copy does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Beside {
    pub cpu: f64,
    pub link: f64,
}

/// The reads a promotion of a layer of `lines` must save to earn its copy:
/// what its copy costs ([`copy_nanoseconds`]) over what a read of it then
/// saves, a miss's one expert at the site it ends soonest on, less a hit.
/// None where a hit saves nothing (the lines the load fits can price a miss
/// at or below a hit): no read earns the copy.
pub fn copy_hits(lines: &ExpertLines, copy: &super::TierCopyCosts, beside: Beside) -> Option<f64> {
    let cpu = lines.host.cpu.map_or(f64::INFINITY, |cpu| cpu.expert.nanoseconds(1.));
    let saved = cpu.min(lines.gpu.link.nanoseconds(1.)) - lines.gpu.resident.nanoseconds(1.);
    (saved > 0.).then(|| copy_nanoseconds(lines, copy, beside) / saved)
}

/// What one record's copy of a layer of `lines` on the transfer queue costs
/// the steps beside it, on every resource it shares: its bytes at the
/// transfer queue's rate (`copy.transfer`, its latency hidden by the
/// batches kept in flight); the device's time it takes from the work beside
/// it meanwhile (`copy`'s stall share); the link's time it takes from the
/// reads in place, its own transfer for the share of the time they cross
/// the link (`beside.link`), as a read in place runs no faster than the link
/// the copy holds too; and, where a CPU site serves, the host memory's time
/// its read takes from the site (one record's read at the memory's rate,
/// [`HostLines::memory`]) for the share of the time the site reads
/// (`beside.cpu`), as a memory-bound site runs no faster than the memory
/// the copy reads too.
pub fn copy_nanoseconds(lines: &ExpertLines, copy: &super::TierCopyCosts, beside: Beside) -> f64 {
    let link = copy.transfer.transfer_seconds(lines.host.record_bytes as f64) * 1e9;
    for share in [beside.cpu, beside.link] {
        assert!((0. ..=1.).contains(&share), "a share of the time: {share}");
    }
    let memory = match (lines.host.cpu, lines.host.memory) {
        (Some(_), Some(memory)) => memory.get() * beside.cpu,
        (None, _) => 0.,
        (Some(_), None) => unreachable!("a CPU site reads a host tier, whose record read a memory line prices"),
    };
    link + copy.stall(link) + link * beside.link + memory
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_cost_model::{GpuLines, PassLine, ReadLine};

    fn lines() -> ExpertLines {
        ExpertLines::reference(ExpertLines::REFERENCE_RECORD)
    }

    /// A copy is priced in the reads it must save: its record over the
    /// link, the device time it stalls beside it, the link's time it takes
    /// from the reads in place for the share of the time they cross it,
    /// and, where a site serves, the host memory's time its read takes for
    /// the share of the time the site reads, over what a read then saves,
    /// the cheaper of the miss's sites less a hit.
    #[test]
    fn a_copy_is_worth_the_hits_its_costs_buy() {
        let lines = lines();
        let (records, bytes) = super::super::streaming_costs::TIER_COPY_PROBE;
        // The link's in-place rate is not what a copy takes: the transfer
        // queue's is.
        let transfer = ReadLine::new(5e-6, 1e-10).unwrap();
        let copy = super::super::TierCopyCosts { blocks: 1, whole_seconds: (records * bytes) as f64 * 3e-10, stall_share: 0.5, transfer, back: transfer };
        let saved = lines.host.cpu.unwrap().expert.nanoseconds(1.).min(lines.gpu.link.nanoseconds(1.)) - lines.gpu.resident.nanoseconds(1.);
        let link = lines.host.record_bytes as f64 * 1e-10 * 1e9;
        let memory = lines.host.memory.unwrap().get();
        let close = |got: f64, want: f64| assert!((got - want).abs() <= want * 1e-12, "{got} against {want}");
        let idle = Beside::default();
        close(copy_hits(&lines, &copy, idle).unwrap(), link * 1.5 / saved);
        close(copy_hits(&lines, &copy, Beside { cpu: 0.25, link: 0. }).unwrap(), (link * 1.5 + memory * 0.25) / saved);
        close(copy_hits(&lines, &copy, Beside { cpu: 0.25, link: 0.4 }).unwrap(), (link * 1.9 + memory * 0.25) / saved);
        let unsited = ExpertLines { host: HostLines { cpu: None, ..lines.host }, ..lines };
        close(copy_nanoseconds(&unsited, &copy, Beside { cpu: 1., link: 0. }), link * 1.5);
        // A miss that costs no more than a hit saves nothing: no read
        // earns a copy, at a link at or below the resident line.
        let flat = ExpertLines { host: HostLines { cpu: None, ..lines.host }, gpu: GpuLines { link: lines.gpu.resident, ..lines.gpu } };
        assert_eq!(copy_hits(&flat, &copy, idle), None);
        let cheaper = PassLine { weights: Nanoseconds::saturating(lines.gpu.resident.nanoseconds(1.) * 1e-9 / 2.), ..lines.gpu.resident };
        let under = ExpertLines { gpu: GpuLines { link: cheaper, ..flat.gpu }, ..flat };
        assert_eq!(copy_hits(&under, &copy, idle), None);
    }


    mod route {
        //! The route's assignment (`assign`, routed.seismic's reference
        //! body, which every native is admitted against) computes
        //! [`ExpertLines::serve`]: on the inputs the route has (one step's
        //! choices, each distinct expert's rows its choices, every record
        //! whole at the router, the policy words of the lines), the CPU's
        //! experts are exactly the function's.
        use super::super::*;
        use magnitude_cost_model::{GpuLines, Hits, Miss, Moves, Pass, PassLine, RowCost, RowLine, Site, SiteLines};
        use seismic_lang::checked::CheckedModule;
        use seismic_lang::entry::ElementBindings;
        use seismic_lang::failure::SourceTermination;
        use seismic_lang::interp::{Arg, Interpreter, TensorData};
        use seismic_lang::types::DType;

        fn module() -> CheckedModule {
            crate::test_support::check_kernels()
        }

        /// The sources `assign` leaves of `routes` (`M` rows of `K`, none a
        /// front row) resolved to `records` under `policy`.
        fn assign(module: &CheckedModule, k: usize, routes: &[i32], records: &[i32], policy: &[i32]) -> Vec<i32> {
            let m = routes.len() / k;
            let entry = module.entry(module.entry_named("assign").unwrap(), &ElementBindings::new()).unwrap();
            let mut interpreter = Interpreter::new(&entry);
            let mut tensor = |shape: Vec<usize>, values: &[i32]| {
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::I32, shape, values.iter().map(|&v| f64::from(v)).collect())))
            };
            let sources = magnitude_kernels::source_words(records, &vec![-1; records.len()]);
            let arguments = [
                tensor(vec![m, k], routes),
                tensor(vec![constants::POLICY_WORDS as usize], policy),
                Arg::Scalar(seismic_lang::reference_math::ReferenceScalar::I32(0)),
                tensor(vec![constants::SOURCE_ROWS as usize, m, k], &sources),
            ];
            let outcome = interpreter.run(&arguments).unwrap();
            assert!(!matches!(outcome.termination(), SourceTermination::Failed(_)), "assign failed");
            let input = outcome.inputs().find(|input| input.ordinal() == 3).unwrap();
            let sources = input.tensor();
            (0..sources.element_count()).map(|at| sources.read(at).unwrap() as i32).collect()
        }

        /// A small deterministic generator (xorshift).
        struct Draw(u64);

        impl Draw {
            fn below(&mut self, bound: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % bound
            }
        }

        #[test]
        fn the_routes_assignment_is_the_cost_functions() {
            let module = module();
            let (k, experts) = (4usize, 12i32);
            let mut draw = Draw(0x9e37_79b9_7f4a_7c15);
            for case in 0..200 {
                let m = 1 + draw.below(4) as usize;
                // Each row's K distinct experts.
                let mut routes = Vec::with_capacity(m * k);
                for _ in 0..m {
                    let mut row = Vec::with_capacity(k);
                    while row.len() < k {
                        let expert = draw.below(experts as u64) as i32;
                        if !row.contains(&expert) {
                            row.push(expert);
                        }
                    }
                    routes.extend(row);
                }
                // Each expert a hit, a tier miss or absent.
                let state = (0..experts).map(|_| draw.below(3)).collect::<Vec<_>>();
                let records = routes
                    .iter()
                    .map(|&expert| match state[expert as usize] {
                        0 => 100 + expert,
                        1 => -2 - (50 + expert),
                        _ => -1,
                    })
                    .collect::<Vec<_>>();
                let ns = |low: u64, high: u64| Nanoseconds::try_from((low + draw.below(high - low)) as u32).unwrap();
                let mut ns = ns;
                let lines = ExpertLines {
                    host: HostLines {
                        cpu: Some(CpuLines {
                            expert: RowLine { one: ns(20_000, 90_000), row: RowCost::Measured(ns(0, 20_000)) },
                            handshake: ns(0, 10_000),
                        }),
                        memory: None,
                        landing: None,
                        record_bytes: 1,
                    },
                    gpu: GpuLines {
                        link: PassLine { weights: ns(20_000, 150_000), row: ns(0, 60_000) },
                        resident: PassLine { weights: ns(1_000, 10_000), row: ns(0, 5_000) },
                        moves: Moves { link: ns(0, 300_000), records: 0. },
                    },
                };
                let mut policy = vec![0; constants::POLICY_WORDS as usize];
                for (word, value) in ExpertLines::policy(&POLICY_LAYOUT, SiteLines::Measured(lines.host.cpu.as_ref().unwrap()), Some(&lines)) {
                    policy[word] = value;
                }
                let sources = assign(&module, k, &routes, &records, &policy);
                // The host function over the same step: each distinct
                // expert in first-choice order, its rows its choices.
                let mut distinct = Vec::<i32>::new();
                for &expert in &routes {
                    if !distinct.contains(&expert) {
                        distinct.push(expert);
                    }
                }
                let rows = |expert: i32| routes.iter().filter(|&&chosen| chosen == expert).count() as f64;
                let hit = distinct.iter().filter(|&&expert| state[expert as usize] == 0).map(|&expert| Hits { experts: 1., rows: rows(expert) }).collect::<Vec<_>>();
                let missed = distinct.iter().copied().filter(|&expert| state[expert as usize] != 0).collect::<Vec<_>>();
                let misses = missed.iter().map(|&expert| Miss { rows: rows(expert), ready_seconds: 0. }).collect::<Vec<_>>();
                let served = lines.serve(Pass::Decode, &hit, &misses);
                let count = routes.len();
                for (choice, &expert) in routes.iter().enumerate() {
                    let (record, site) = (sources[constants::SOURCE_RECORD as usize * count + choice], sources[constants::SOURCE_SITE as usize * count + choice]);
                    let expected = match missed.iter().position(|&miss| miss == expert).map(|at| served.sites[at]) {
                        Some(Site::Cpu) => (-1, match state[expert as usize] { 1 => 50 + expert, _ => -2 }),
                        _ => (records[choice], -1),
                    };
                    assert_eq!((record, site), expected, "case {case}: choice {choice} (expert {expert}) of {routes:?}, {served:?}");
                }
            }
        }
    }
}
