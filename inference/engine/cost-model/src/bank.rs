//! What a plan prices its bank's holdings with: the generic prior
//! assessment estimates with, from the model's headers and the machine's
//! standing figures alone, never live data or recorded data of a model. A
//! bank slot is worth the decode seconds it saves (the step `C` of the form
//! its slots decode in with the misses its slots leave,
//! [`StreamedLayers::step_seconds`]) times decode steps a second; a prompt
//! pass size the routed prefill seconds its fewer passes save
//! ([`routed_prompt_seconds`]) times requests a second; both rates the
//! workload prior's ([`WorkloadRates::prior`]).

use crate::census::RoutedKind;
use crate::compute::ExpertPass;
use crate::lines::ReadLine;
use crate::streamed::{FormSeconds, HeldSlots, StreamedLayers};
use seismic_measure::pricing::expert_union;
use std::fmt;

/// The load's serving rates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkloadRates {
    /// Decode steps a second of serving.
    pub decode_steps_per_s: f64,
    /// The prefill's speed while it runs: tokens a second of prefill.
    pub prefill_tokens_per_s: f64,
    /// Requests a second of serving: each prefills its prompt once, in
    /// whole or from a resumed prefix, so the prompts prefilled a second
    /// too.
    pub requests_per_s: f64,
    /// Requests a second whose input the vision projector encodes.
    pub vision_uses_per_s: f64,
}

impl WorkloadRates {
    /// The rates before the load serves, with no request history to read:
    /// one stream that prefills `prefill_tokens` (the planned context) in
    /// `prefill_seconds`, then decodes `decode_steps` steps of
    /// `step_seconds`, then repeats.
    ///
    /// # Panics
    /// Where a figure is not positive and finite: a load that cannot
    /// prefill or step has no rates.
    pub fn prior(step_seconds: f64, decode_steps: f64, prefill_tokens: f64, prefill_seconds: f64) -> Self {
        for figure in [step_seconds, decode_steps, prefill_tokens, prefill_seconds] {
            assert!(figure.is_finite() && figure > 0.0, "a workload prior needs positive figures, not {figure}");
        }
        let span = prefill_seconds + decode_steps * step_seconds;
        Self {
            decode_steps_per_s: decode_steps / span,
            prefill_tokens_per_s: prefill_tokens / prefill_seconds,
            requests_per_s: 1.0 / span,
            vision_uses_per_s: 0.0,
        }
    }
}

/// A prior's figure past what a launch's coordinates hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorOverflow;

impl fmt::Display for PriorOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a prior's figure exceeds u32")
    }
}

impl std::error::Error for PriorOverflow {}

/// The workload prior before a load serves, with no request history to
/// read: one stream that prefills half the planned context, then decodes
/// the other half, a row at a time. The plan prices its holdings with it,
/// and serving starts every load's meter from it, banked or not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prior {
    prefill_tokens: u64,
    decode_steps: u64,
    /// The depth a decode step is priced at: the middle of the decode.
    pub decode_depth: u32,
    /// The prefill's launches of `launch_rows` rows each, priced at the
    /// middle of the prompt.
    launch_rows: u32,
    prefill_launches: u64,
    prefill_depth: u32,
    /// The rows of one launch's region of a prompt pass: the widest row
    /// class of a launch.
    pub region_rows: u64,
}

impl Prior {
    /// The prior of a load planning `context` rows of context in launches
    /// of at most `launch_rows` rows, the widest row class of which is
    /// `region_rows`.
    pub fn of(context: u64, launch_rows: u64, region_rows: u64) -> Result<Self, PriorOverflow> {
        let narrow = |figure: u64| u32::try_from(figure).map_err(|_| PriorOverflow);
        let context = context.max(2);
        let prefill_tokens = context / 2;
        let decode_steps = context - prefill_tokens;
        Ok(Self {
            prefill_tokens,
            decode_steps,
            decode_depth: narrow(prefill_tokens + decode_steps / 2)?,
            launch_rows: narrow(launch_rows)?,
            prefill_launches: prefill_tokens.div_ceil(launch_rows),
            prefill_depth: narrow(prefill_tokens / 2)?,
            region_rows,
        })
    }

    /// The prior's rates, its decode steps `step_seconds` each and its
    /// prefill's launches priced on the resident step `resident(rows,
    /// depth)`.
    pub fn rates<E>(&self, step_seconds: f64, resident: impl Fn(u32, u32) -> Result<f64, E>) -> Result<WorkloadRates, E> {
        let launch = resident(self.launch_rows, self.prefill_depth)?;
        let prefill_seconds = self.prefill_launches as f64 * launch;
        Ok(WorkloadRates::prior(step_seconds, self.decode_steps as f64, self.prefill_tokens as f64, prefill_seconds))
    }
}

/// Consecutive or not, the streamed layers of one routed kind: their kind
/// ([`ExpertPass::kind`]), its experts, the experts a row selects, and how
/// many layers are of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassLayers {
    pub kind: RoutedKind,
    pub experts: u64,
    pub selected: u64,
    pub layers: u64,
}

impl PassLayers {
    /// The streamed layers of `passes` grouped by their routed kind, each
    /// layer's `experts` in decoder order, `selected` a row, in order of
    /// first appearance.
    ///
    /// # Panics
    /// Where two layers of one kind have other experts.
    pub fn of(passes: &ExpertPass, experts: impl IntoIterator<Item = u64>, selected: u64) -> Vec<Self> {
        let mut kinds = Vec::<Self>::new();
        for (layer, experts) in experts.into_iter().enumerate() {
            let kind = passes.kind(layer);
            match kinds.iter_mut().find(|known| known.kind == kind) {
                Some(known) => {
                    assert_eq!(known.experts, experts, "invariant violated: layers of {kind:?} have other experts");
                    known.layers += 1;
                }
                None => kinds.push(Self { kind, experts, selected, layers: 1 }),
            }
        }
        kinds
    }
}

/// The routed seconds a prompt of `prompt_rows` rows takes in passes over
/// `launches` launches of `region_rows` rows each, on the executor's costs
/// (`passes`): each pass reads, at every routed layer, the union its rows
/// choose of the layer's experts ([`expert_union`]), and computes their
/// choices, each routed kind once at its layers' count; and it reads from
/// storage the misses of that union a partial bank does not hold (`reads`,
/// a pass's seconds by its rows; none where the bank holds every expert).
/// Fewer, longer passes read each expert fewer times, from the device's
/// memory and from storage alike.
pub(crate) fn routed_prompt_seconds(
    passes: &ExpertPass,
    routed: &[PassLayers],
    windows: u64,
    region_rows: u64,
    prompt_rows: u64,
    launches: u64,
    reads: &dyn Fn(u64) -> f64,
) -> f64 {
    let pass_rows = region_rows * launches;
    let windows = windows.max(1) as f64;
    let one = |rows: u64| -> f64 {
        routed
            .iter()
            .map(|kind| {
                let union = expert_union(kind.experts, kind.selected, rows);
                kind.layers as f64 * windows * passes.kind_seconds(kind.kind, union / windows, (rows * kind.selected) as f64 / windows)
            })
            .sum::<f64>()
            + reads(rows)
    };
    let (whole, rest) = (prompt_rows / pass_rows, prompt_rows % pass_rows);
    whole as f64 * one(pass_rows) + if rest > 0 { one(rest) } else { 0.0 }
}

/// The prices of one bank's holdings under the workload prior: its layers
/// and their passes, its routed kinds, its windows, the model store's read
/// line, `C` the step at the prior's row and depth in each form, the
/// prior's rates, and the rows of a pass's region and of a prompt.
pub struct BankPrices {
    layers: StreamedLayers,
    pass: ExpertPass,
    routed: Vec<PassLayers>,
    windows: u64,
    read: ReadLine,
    compute: FormSeconds,
    rates: WorkloadRates,
    region_rows: u64,
    prompt_rows: u64,
}

impl BankPrices {
    /// The prices of a bank of `layers`, its passes `pass` over `routed`, in
    /// `windows` windows, its file reads along `read`, its step `compute`
    /// in each form, under `rates`, its passes' regions `region_rows` rows
    /// and its prompts `prompt_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: StreamedLayers,
        pass: ExpertPass,
        routed: Vec<PassLayers>,
        windows: u64,
        read: ReadLine,
        compute: FormSeconds,
        rates: WorkloadRates,
        region_rows: u64,
        prompt_rows: u64,
    ) -> Self {
        Self { layers, pass, routed, windows, read, compute, rates, region_rows, prompt_rows }
    }

    /// A decode step's seconds holding `held` of each layer's experts.
    pub fn step_seconds(&self, held: HeldSlots) -> f64 {
        self.layers.step_seconds(None, self.read, held, 1.0, self.compute).total_seconds
    }

    /// A whole-context prompt's routed seconds in passes of `launches`
    /// launches holding `held`: each pass's compute and the reads of its
    /// misses ([`StreamedLayers::pass_reads`]).
    pub fn prompt_seconds(&self, launches: u64, held: HeldSlots) -> f64 {
        let reads = |rows: u64| self.layers.pass_reads(self.read, held, rows as f64);
        routed_prompt_seconds(&self.pass, &self.routed, self.windows, self.region_rows, self.prompt_rows, launches, &reads)
    }

    /// The prior's rates.
    pub fn rates(&self) -> WorkloadRates {
        self.rates
    }

    /// The model store's read line.
    pub fn read(&self) -> ReadLine {
        self.read
    }

    /// The free records the model store's path needs for moves of `bytes`
    /// stored bytes, the bank holding `held` ([`PathRecords::of`]), with
    /// lookahead hints spanning `hint_leads` layers ahead
    /// (`designs/streamed-decode.md` §5, §6.4): each read ahead holds its
    /// staging record across its own read `λ + b·D⁻¹` plus queueing behind
    /// the `hint_leads` layers of demand reads (`hint_leads · reads / L`),
    /// and each move's victim is released after the mean deadline-ordered
    /// wait ([`PathRecords::release`]) of a decode step at `held` and its
    /// reads; none where the line has no bandwidth term.
    ///
    /// Plan-time pricing: the size that minimises the step weighs a record
    /// more of reserve, a slot less (its misses' rise on the placement
    /// shadows' miss curve, times their read), against the reads it unblocks
    /// (the path's rate times the chance a move finds no free record).
    /// Neither marginal is priced before the load runs, so the plan holds
    /// the mean release's records and leaves that trade to the running
    /// load.
    pub fn paths(&self, bytes: u64, held: HeldSlots, hint_leads: u64) -> Option<PathRecords> {
        let step = self.step_seconds(held);
        let reads = self.layers.pass_reads(self.read, held, 1.0);
        let layers: u64 = self.routed.iter().map(|kind| kind.layers).sum();
        let queue = if layers == 0 { 0.0 } else { hint_leads as f64 * (reads / layers as f64) };
        PathRecords::of(self.read, bytes, queue, PathRecords::release(step, reads))
    }
}

/// The free records a path needs so that no move it starts is ever blocked
/// by a missing one: the records its moves hold from issue to landing at
/// the rate it reads them (`ahead`, staging records), and those beside them
/// whose victims the device still reads (`reserve`, the tier records kept
/// free). Each is the path's rate times a measured time: a move's read
/// plus its demand queue wait across the hint window, and its read with
/// its victim's release after it (`designs/streamed-decode.md` §5, §6.4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PathRecords {
    pub ahead: u64,
    pub reserve: u64,
}

impl PathRecords {
    /// No records: a bank that reads nothing ahead.
    pub const NONE: Self = Self { ahead: 0, reserve: 0 };

    /// The records of a path reading moves of `bytes` stored bytes along
    /// `line`, each read ahead queued `queue` seconds behind demand reads
    /// across the hint window (`hint_leads · reads / L`) and each move's
    /// victim released `release` seconds after the move lands
    /// (`designs/streamed-decode.md` §5, §6.4). By Little's Law at the path's
    /// rate `1 / (b·D⁻¹)`: a read ahead holds its staging record for
    /// `read + queue` (`ahead = ⌈(λ + b·D⁻¹ + queue)/(b·D⁻¹)⌉`), and its
    /// victim is held `release` beyond the read
    /// (`reserve = max(ahead, ⌈(λ + b·D⁻¹ + release)/(b·D⁻¹)⌉)`).
    /// None where a move costs no transfer time (a line without a bandwidth
    /// term, as memory's): its rate is no line's to give.
    pub fn of(line: ReadLine, bytes: u64, queue: f64, release: f64) -> Option<Self> {
        let transfer = line.transfer_seconds(bytes as f64);
        if transfer <= 0. {
            return None;
        }
        assert!(queue >= 0., "invariant violated: a move queued {queue} s behind demand");
        assert!(release >= 0., "invariant violated: a victim released {release} s after its move");
        let read = line.seconds(bytes);
        let records = |seconds: f64| (seconds / transfer).ceil() as u64;
        let ahead = records(read + queue);
        let reserve = records(read + release).max(ahead);
        Some(Self { ahead, reserve })
    }

    /// The mean time from a move's issue to its victim's release, for moves
    /// a decode step of `step` seconds issues earliest deadline first while
    /// its reads take `reads` of them. A move for layer fraction `x` of the
    /// next step evicts a victim of layer `x` of this one, free once the
    /// device passes `x`, at `x·step`; issued in layer order at the path's
    /// pace, it starts at `x·reads`. So it waits `x·(step − reads)`, whose
    /// mean over the layers is `(step − reads)/2`. A step its reads fill
    /// releases each victim as its move is issued.
    pub fn release(step: f64, reads: f64) -> f64 {
        (step - reads).max(0.) / 2.
    }
}
