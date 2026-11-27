//! Per-layer router calibration (design §3.4): two position-indexed orders
//! (route order and history order) with uniform tie-class spreading, `route`,
//! `fell`, `history` and `far` tables rolling on the group's tier-0 epoch via
//! [`horizon::Window`], a fixed-size sample ring of length `DEPTHS + 1`, and
//! O(1) reads via per-table prefix sums.
//!
//! Every allocation is fixed at construction from `(experts, DEPTHS)` and never
//! grows with run length or τ.

use std::mem::size_of;

use super::super::horizon::{self, Tally};
use super::super::watcher::Step;
use super::super::DEPTHS;
use super::history::History;
use super::Rows;

/// The three route-table forms (design §3.4): front-or-plain decode steps,
/// verify decode steps, and prompt unions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Form {
    Front = 0,
    Verify = 1,
    Prompt = 2,
}

const FORMS: usize = 3;

impl Form {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

/// One cell's tally over a window epoch: 8 bytes (`2 × f32`), rolling on the
/// group's tier-0 epoch via [`horizon::Window`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CellTally {
    seen: f32,
    used: f32,
}

impl Tally for CellTally {
    #[inline]
    fn is_empty(&self) -> bool {
        self.seen == 0.0
    }
}

/// Cumulative prefix sum over cells `0..i` in `f64` so subtracting across any
/// tie class `[lo, hi)` stays exact even after millions of steps.
#[derive(Clone, Copy, Debug, Default)]
struct Prefix {
    seen: f64,
    used: f64,
}

/// One E-cell calibration table with O(1) pooled reads for both the epoch it
/// was last rebuilt at (`prefix_now`) and any later epoch without intervening
/// counts (`prefix_next`), matching [`horizon::Window::read`] without interior
/// mutability.
#[derive(Clone, Debug)]
struct CellTable {
    cells: Vec<horizon::Window<CellTally>>,
    epoch: u64,
    prefix_now: Vec<Prefix>,
    prefix_next: Vec<Prefix>,
}

impl CellTable {
    fn new(experts: usize, epoch: u64) -> Self {
        Self {
            cells: vec![horizon::Window::new(epoch); experts],
            epoch,
            prefix_now: vec![Prefix::default(); experts + 1],
            prefix_next: vec![Prefix::default(); experts + 1],
        }
    }

    /// Count one tie class covering cells `[lo, hi)` (`lo < hi <= E`) with
    /// total `class_used` across its `n = hi - lo` members. Each cell in
    /// `[lo, hi)` receives `seen += 1.0`, `used += class_used / n` (exact
    /// expectation of a uniformly random tie-break, design §3.4).
    fn count_class(&mut self, lo: usize, hi: usize, class_used: f64, epoch: u64) {
        debug_assert!(lo < hi && hi <= self.cells.len());
        let n = (hi - lo) as f64;
        let d_used = (class_used / n) as f32;
        for cell in &mut self.cells[lo..hi] {
            let tally = cell.count(epoch);
            tally.seen += 1.0;
            tally.used += d_used;
        }
    }

    /// Rebuild `prefix_now` (at `epoch`) and `prefix_next` (for any later epoch
    /// with no new counts) in O(E).
    fn rebuild_prefix(&mut self, epoch: u64) {
        assert!(epoch >= self.epoch, "calibration table at epoch {} rolled to {epoch}", self.epoch);
        self.epoch = epoch;
        let mut now = Prefix::default();
        let mut next = Prefix::default();
        self.prefix_now[0] = now;
        self.prefix_next[0] = next;
        for (i, cell) in self.cells.iter_mut().enumerate() {
            if let Some((prev, curr)) = cell.read(epoch) {
                now.seen += f64::from(prev.seen) + f64::from(curr.seen);
                now.used += f64::from(prev.used) + f64::from(curr.used);
                let rolled = if !curr.is_empty() { curr } else { prev };
                next.seen += f64::from(rolled.seen);
                next.used += f64::from(rolled.used);
            }
            self.prefix_now[i + 1] = now;
            self.prefix_next[i + 1] = next;
        }
    }

    /// Pooled `(seen, used)` over cells `[lo, hi)` at `epoch`, in O(1).
    /// Panics if `epoch` regresses before `self.epoch`.
    #[inline]
    fn pooled(&self, lo: usize, hi: usize, epoch: u64) -> (f64, f64) {
        assert!(epoch >= self.epoch, "calibration table at epoch {} read at {epoch}", self.epoch);
        debug_assert!(lo < hi && hi <= self.cells.len());
        let prefix = if epoch == self.epoch { &self.prefix_now } else { &self.prefix_next };
        let a = prefix[lo];
        let b = prefix[hi];
        (b.seen - a.seen, b.used - a.used)
    }

    /// Pooled calibrated probability `clamp((Σ used + prior) / (Σ seen + 1.0), 0.0, 1.0)`
    /// over cells `[lo, hi)` at `epoch`, in O(1).
    #[inline]
    fn prob(&self, lo: usize, hi: usize, prior: f64, epoch: u64) -> f64 {
        let (seen, used) = self.pooled(lo, hi, epoch);
        ((used + prior) / (seen + 1.0)).clamp(0.0, 1.0)
    }

    fn bytes(&self) -> u64 {
        (self.cells.capacity() * size_of::<horizon::Window<CellTally>>()
            + self.prefix_now.capacity() * size_of::<Prefix>()
            + self.prefix_next.capacity() * size_of::<Prefix>()) as u64
    }
}

/// Per-expert tie class `[lo, hi)` and prior in one sample.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SampleEntry {
    lo: u16,
    hi: u16,
    prior: f32,
}

/// One pre-allocated slot in the fixed-length (`DEPTHS + 1`) decode sample ring.
#[derive(Clone, Debug)]
struct RingSlot {
    occupied: bool,
    decode_idx: u64,
    step: Step,
    form: Form,
    front_depth: usize,
    fell_recorded: bool,
    route: Vec<SampleEntry>,
    /// Front rows `1..DEPTHS - 1` (row `j` at index `j - 1`), kept for `fell`
    /// on live calibration only; empty on shadow calibration.
    front_rows: Vec<Vec<SampleEntry>>,
    history: Vec<SampleEntry>,
    picked: Vec<bool>,
}

impl RingSlot {
    fn new(experts: usize) -> Self {
        Self {
            occupied: false,
            decode_idx: 0,
            step: 0,
            form: Form::Front,
            front_depth: 0,
            fell_recorded: false,
            route: vec![SampleEntry::default(); experts],
            front_rows: Vec::new(),
            history: vec![SampleEntry::default(); experts],
            picked: vec![false; experts],
        }
    }

    fn ensure_front_rows(&mut self, needed: usize, experts: usize) {
        while self.front_rows.len() < needed {
            self.front_rows.push(vec![SampleEntry::default(); experts]);
        }
    }

    fn bytes(&self) -> u64 {
        let front_bytes: usize = self.front_rows.capacity() * size_of::<Vec<SampleEntry>>()
            + self.front_rows.iter().map(|r| r.capacity() * size_of::<SampleEntry>()).sum::<usize>();
        (self.route.capacity() * size_of::<SampleEntry>()
            + self.history.capacity() * size_of::<SampleEntry>()
            + self.picked.capacity() * size_of::<bool>()
            + front_bytes) as u64
    }
}

/// One pre-allocated slot for a pending `fell` sample (at most `DEPTHS` active
/// at once).
#[derive(Clone, Debug)]
struct FellSlot {
    /// Decode index of row `f - 1` whose successor row `f` fell; `None` when
    /// inactive.
    origin_idx: Option<u64>,
    sample: Vec<SampleEntry>,
}

impl FellSlot {
    fn new(experts: usize) -> Self {
        Self {
            origin_idx: None,
            sample: vec![SampleEntry::default(); experts],
        }
    }

    fn bytes(&self) -> u64 {
        (self.sample.capacity() * size_of::<SampleEntry>()) as u64
    }
}

/// Metadata of the latest logged step at this layer.
#[derive(Clone, Copy, Debug)]
struct LatestMeta {
    step: Step,
    form: Form,
    front_depth: usize,
}

/// One decile (of 10) of history-order cells comparing `history` (lead `L + 1`)
/// against `far` (lead `⌈τ_0 / ā⌉`) for the §8 A1 diagnostic printout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct FarDecile {
    pub history_seen: f64,
    pub history_used: f64,
    pub far_seen: f64,
    pub far_used: f64,
}

impl FarDecile {
    pub(super) fn history_rate(self) -> Option<f64> {
        (self.history_seen > 0.0).then(|| self.history_used / self.history_seen)
    }

    pub(super) fn history_se(self) -> Option<f64> {
        let p = self.history_rate()?;
        Some((p * (1.0 - p) / self.history_seen).max(0.0).sqrt())
    }

    pub(super) fn far_rate(self) -> Option<f64> {
        (self.far_seen > 0.0).then(|| self.far_used / self.far_seen)
    }

    pub(super) fn far_se(self) -> Option<f64> {
        let p = self.far_rate()?;
        Some((p * (1.0 - p) / self.far_seen).max(0.0).sqrt())
    }
}

/// Per-layer router calibration (design §3.4).
#[derive(Clone, Debug)]
pub(super) struct Calibration {
    experts: usize,
    active: usize,
    /// Active maximum decode depth (`1..=DEPTHS`), sizing the sample ring and
    /// carried lead tables to the load's actual decode width.
    max_depths: usize,
    /// Whether this calibration tracks `fell` and `far` tables (live = true,
    /// shadow = false).
    is_live: bool,
    /// Initial epoch for lazily created [`CellTable`]s.
    initial_epoch: u64,
    /// `route[form][lead - 1]` for `form ∈ {Front, Verify, Prompt}`,
    /// `lead ∈ 1..=max_depths`; lazily allocated on first count.
    route: [[Option<Box<CellTable>>; DEPTHS]; FORMS],
    /// `fell[lead - 1]` for `lead ∈ 1..=max_depths`; lazily allocated on first
    /// count on live calibration only.
    fell: [Option<Box<CellTable>>; DEPTHS],
    /// History-order table counted at lead `max_depths + 1`, rolling on the
    /// group's tier-0 epoch; lazily allocated on first count.
    history: Option<Box<CellTable>>,
    /// Instrument table for §8 A1, counted at lead `⌈τ_0 / ā⌉`; lazily
    /// allocated on first count on live calibration only.
    far: Option<Box<CellTable>>,
    /// Decode sample ring of length at most `max_depths + 1`, grown lazily as
    /// decode steps arrive.
    ring: Vec<RingSlot>,
    /// Next slot index in `ring` (`decode_count % (max_depths + 1)`).
    decode_count: u64,
    /// Pending `fell` samples (grown lazily up to `max_depths` on live, empty
    /// on shadow).
    pending_fell: Vec<FellSlot>,
    /// Pending prompt sample (`prompt_base = Some(decode_count_at_log)`) for
    /// settling `route[Form::Prompt]` across leads `1..=max_depths`.
    prompt_base: Option<u64>,
    /// Decode step count when `prompt_sample` was last logged (decoupled from
    /// `prompt_base` so `history_prob` blends each new request's prompt union
    /// smoothly over `⌈E / k⌉` decode steps after prefill).
    prompt_logged_at: Option<u64>,
    prompt_sample: Vec<SampleEntry>,
    /// Cumulative row-weighted per-expert prompt priors summed across all
    /// prefill chunks of the current request (`prompt_logged_at == Some(decode_count)`),
    /// divided by `prompt_rows` so multi-chunk prefills weight chunks by token count.
    prompt_accum: Vec<f64>,
    prompt_rows: f64,
    /// Single pending `far` slot (bounded, constant in τ).
    far_target: Option<u64>,
    far_sample: Vec<SampleEntry>,
    /// Each expert's last decode read step.
    last_read: Vec<Option<Step>>,
    /// Latest decode step logged (for monotonicity and leaving prompt evidence).
    latest_decode_step: Option<Step>,
    /// Latest step metadata (may be `Form::Prompt` before the first decode step).
    latest_meta: Option<LatestMeta>,
    /// Cached route samples for `latest_meta` (`0..=front_depth`, starts with
    /// row 0 and grows lazily up to `front_depth + 1`).
    latest_rows: Vec<Vec<SampleEntry>>,
    /// Whether each row `0..=front_depth` of `latest_meta` chose each expert
    /// (starts with row 0 and grows lazily up to `front_depth + 1`).
    latest_chose: Vec<Vec<bool>>,
    /// Cached history sample for §8 A1 diagnostics (`history` and `far`).
    latest_history: Vec<SampleEntry>,
    /// Reusable O(E) scratch buffers (zero allocations per step).
    scratch_order: Vec<usize>,
    scratch_keys: Vec<f32>,
    scratch_picked: Vec<bool>,
    scratch_class: Vec<(f64, u16)>,
}

impl Calibration {
    /// Live layer calibration (with `fell` and `far` tables) at full `DEPTHS`.
    pub(super) fn new(experts: usize, active: usize, epoch: u64) -> Self {
        Self::with_depth(experts, active, epoch, DEPTHS)
    }

    /// Live layer calibration sized to `max_depths` (`1..=DEPTHS`).
    pub(super) fn with_depth(experts: usize, active: usize, epoch: u64, max_depths: usize) -> Self {
        Self::build(experts, active, epoch, max_depths, false)
    }

    /// Shadow layer calibration (omits `fell` and `far`, since shadows run on
    /// true routes and never print A1; keeps `history` so shadows rank with the
    /// exact same calibrated `p_∞` as live).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn new_shadow(experts: usize, active: usize, epoch: u64) -> Self {
        Self::new_shadow_with_depth(experts, active, epoch, DEPTHS)
    }

    /// Shadow layer calibration sized to `max_depths` (`1..=DEPTHS`).
    pub(super) fn new_shadow_with_depth(
        experts: usize,
        active: usize,
        epoch: u64,
        max_depths: usize,
    ) -> Self {
        Self::build(experts, active, epoch, max_depths, true)
    }

    fn build(
        experts: usize,
        active: usize,
        epoch: u64,
        max_depths: usize,
        is_shadow: bool,
    ) -> Self {
        assert!(experts > 0 && experts <= usize::from(u16::MAX), "experts {experts} must fit u16 (1..={})", u16::MAX);
        assert!(active > 0 && active <= experts, "active {active} must be in 1..={experts}");
        let max_depths = max_depths.clamp(1, DEPTHS);
        let cold_prior = (active as f64 / experts as f64) as f32;
        let cold_entry = SampleEntry { lo: 0, hi: experts as u16, prior: cold_prior };
        Self {
            experts,
            active,
            max_depths,
            is_live: !is_shadow,
            initial_epoch: epoch,
            route: std::array::from_fn(|_| std::array::from_fn(|_| None)),
            fell: std::array::from_fn(|_| None),
            history: None,
            far: None,
            ring: Vec::new(),
            decode_count: 0,
            pending_fell: Vec::new(),
            prompt_base: None,
            prompt_logged_at: None,
            prompt_sample: vec![SampleEntry::default(); experts],
            prompt_accum: vec![0.0; experts],
            prompt_rows: 0.0,
            far_target: None,
            far_sample: if is_shadow { Vec::new() } else { vec![SampleEntry::default(); experts] },
            last_read: vec![None; experts],
            latest_decode_step: None,
            latest_meta: None,
            latest_rows: vec![vec![SampleEntry::default(); experts]],
            latest_chose: vec![vec![false; experts]],
            latest_history: vec![cold_entry; experts],
            scratch_order: (0..experts).collect(),
            scratch_keys: vec![0.0; experts],
            scratch_picked: vec![false; experts],
            scratch_class: vec![(0.0, 0); experts],
        }
    }

    /// Maximum carried route lead for `form`: `Form::Verify` steps span a
    /// multi-token verify burst (`ā ≈ 5..6`), so only lead 1 has 1-step Markov
    /// autocorrelation; leads `g >= 2` lie `10..90` tokens ahead where
    /// calibrated long-term history (`p_∞`) governs reuse.
    #[inline]
    fn route_horizon(&self, form: Form) -> usize {
        match form {
            Form::Verify => 1,
            Form::Front | Form::Prompt => self.max_depths,
        }
    }

    #[inline]
    fn table_prob(table: &Option<Box<CellTable>>, lo: usize, hi: usize, prior: f64, epoch: u64) -> f64 {
        match table {
            Some(table) => table.prob(lo, hi, prior, epoch),
            None => prior.clamp(0.0, 1.0),
        }
    }

    /// Log a decode step: settle pending samples at leads `1..=DEPTHS` (`route`,
    /// `fell`, `prompt`), lead `DEPTHS + 1` (`history`), and `far`; update
    /// `last_read`; build and push this step's samples; refresh cached prefix
    /// sums and latest-step state.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn log_decode(
        &mut self,
        step: Step,
        form: Form,
        rows: Option<&Rows>,
        chosen: &[usize],
        epoch: u64,
        tau_steps: Option<usize>,
        history: &History,
        layer: usize,
    ) {
        assert!(form != Form::Prompt, "log_decode called with Form::Prompt");
        // Looped architectures (e.g. Inkling 2 Mini) visit the same physical
        // expert stack multiple times within a single decode step, so equal
        // step numbers are valid; only strictly decreasing steps are out of order.
        assert!(
            self.latest_decode_step.is_none_or(|prev| step >= prev),
            "decode step {step} logged out of order after {:?}",
            self.latest_decode_step
        );
        if let Some(rows) = rows {
            assert_eq!(rows.experts, self.experts, "rows experts {} != layer experts {}", rows.experts, self.experts);
        }

        // 1. Mark chosen experts for outcome settling.
        self.scratch_picked.fill(false);
        for &expert in chosen {
            assert!(expert < self.experts, "chosen expert {expert} out of range 0..{}", self.experts);
            assert!(!self.scratch_picked[expert], "duplicate expert {expert} in decode step {step}");
            self.scratch_picked[expert] = true;
        }

        let decode_idx = self.decode_count;
        self.decode_count += 1;

        // 2. Settle pending samples against this step's outcome.
        let max_depths = self.max_depths;
        let experts = self.experts;
        let initial_epoch = self.initial_epoch;
        let mut route_dirty = [[false; DEPTHS]; FORMS];
        let mut fell_dirty = [false; DEPTHS];
        let mut history_dirty = false;

        for slot in &self.ring {
            if !slot.occupied {
                continue;
            }
            let lead = (decode_idx - slot.decode_idx) as usize;
            let horizon = self.route_horizon(slot.form);
            if (1..=horizon).contains(&lead) {
                let fi = slot.form.index();
                let table = self.route[fi][lead - 1]
                    .get_or_insert_with(|| Box::new(CellTable::new(experts, initial_epoch)));
                count_sample(
                    table,
                    &slot.route,
                    &self.scratch_picked,
                    &mut self.scratch_class,
                    epoch,
                );
                route_dirty[fi][lead - 1] = true;
            } else if lead == horizon + 1 {
                let table = self
                    .history
                    .get_or_insert_with(|| Box::new(CellTable::new(experts, initial_epoch)));
                count_sample(
                    table,
                    &slot.history,
                    &self.scratch_picked,
                    &mut self.scratch_class,
                    epoch,
                );
                history_dirty = true;
            }
        }

        if self.is_live {
            for slot in &mut self.pending_fell {
                if let Some(origin) = slot.origin_idx {
                    if decode_idx > origin {
                        let lead = (decode_idx - origin) as usize;
                        if lead <= max_depths {
                            let table = self.fell[lead - 1]
                                .get_or_insert_with(|| Box::new(CellTable::new(experts, initial_epoch)));
                            count_sample(
                                table,
                                &slot.sample,
                                &self.scratch_picked,
                                &mut self.scratch_class,
                                epoch,
                            );
                            fell_dirty[lead - 1] = true;
                        }
                        if lead >= max_depths {
                            slot.origin_idx = None;
                        }
                    }
                }
            }
        }

        if let Some(base) = self.prompt_base {
            let lead = (decode_idx + 1 - base) as usize;
            if (1..=max_depths).contains(&lead) {
                let fi = Form::Prompt.index();
                let table = self.route[fi][lead - 1]
                    .get_or_insert_with(|| Box::new(CellTable::new(experts, initial_epoch)));
                count_sample(
                    table,
                    &self.prompt_sample,
                    &self.scratch_picked,
                    &mut self.scratch_class,
                    epoch,
                );
                route_dirty[fi][lead - 1] = true;
            }
            if lead >= max_depths {
                self.prompt_base = None;
            }
        }

        if self.is_live {
            if let Some(target) = self.far_target {
                if decode_idx >= target {
                    let far = self
                        .far
                        .get_or_insert_with(|| Box::new(CellTable::new(experts, initial_epoch)));
                    count_sample(far, &self.far_sample, &self.scratch_picked, &mut self.scratch_class, epoch);
                    far.rebuild_prefix(epoch);
                    self.far_target = None;
                }
            }
        }

        // Rebuild prefix sums for every table that received a count this step.
        for fi in 0..FORMS {
            for g in 0..max_depths {
                if route_dirty[fi][g] {
                    self.route[fi][g].as_mut().expect("dirty route table exists").rebuild_prefix(epoch);
                }
            }
        }
        if self.is_live {
            for g in 0..max_depths {
                if fell_dirty[g] {
                    self.fell[g].as_mut().expect("dirty fell table exists").rebuild_prefix(epoch);
                }
            }
        }
        if history_dirty {
            self.history.as_mut().expect("dirty history table exists").rebuild_prefix(epoch);
        }

        // 3. Update last_read.
        for &expert in chosen {
            self.last_read[expert] = self.last_read[expert].max(Some(step));
        }

        // 4. Build history sample (and capture into far slot if idle).
        let prompt = self.prompt_logged_at.map(|logged_at| (self.prompt_sample.as_slice(), logged_at));
        build_history_sample(
            self.experts,
            self.active,
            self.decode_count,
            history,
            layer,
            prompt,
            &mut self.scratch_order,
            &mut self.latest_history,
        );
        if self.is_live && self.far_target.is_none() {
            if let Some(lead) = tau_steps {
                if lead >= 1 {
                    self.far_sample.copy_from_slice(&self.latest_history);
                    self.far_target = Some(decode_idx + lead as u64);
                }
            }
        }

        // 5. Build route samples for row 0 (and front rows 1..=front_depth).
        let front_depth = match (form, rows) {
            (Form::Front, Some(rows)) => {
                let d = usize::from(rows.deepest());
                assert!(d < DEPTHS, "front depth {d} must be < DEPTHS ({DEPTHS})");
                d
            }
            _ => 0,
        };
        while self.latest_rows.len() <= front_depth {
            self.latest_rows.push(vec![SampleEntry::default(); self.experts]);
            self.latest_chose.push(vec![false; self.experts]);
        }

        match (form, rows) {
            (Form::Verify, Some(rows)) => {
                // Verify reading rows are all rows in the pass.
                extract_rows_keys(
                    rows,
                    None,
                    Some(&self.scratch_picked),
                    &mut self.scratch_keys,
                    Some(&mut self.latest_chose[0]),
                );
                build_route_sample(
                    self.experts,
                    &self.scratch_keys,
                    history,
                    layer,
                    &mut self.scratch_order,
                    &mut self.latest_rows[0],
                );
            }
            (Form::Front, Some(rows)) => {
                for depth in 0..=front_depth {
                    let matched = extract_rows_keys(
                        rows,
                        Some(depth as u8),
                        Some(&self.scratch_picked),
                        &mut self.scratch_keys,
                        Some(&mut self.latest_chose[depth]),
                    );
                    assert!(
                        depth == 0 || matched,
                        "front rows must cover every depth 1..={front_depth}, missing {depth}"
                    );
                    build_route_sample(
                        self.experts,
                        &self.scratch_keys,
                        history,
                        layer,
                        &mut self.scratch_order,
                        &mut self.latest_rows[depth],
                    );
                }
            }
            (_, None) => {
                for e in 0..self.experts {
                    let p = self.scratch_picked[e];
                    self.scratch_keys[e] = if p { 1.0 } else { 0.0 };
                    self.latest_chose[0][e] = p;
                }
                build_route_sample(
                    self.experts,
                    &self.scratch_keys,
                    history,
                    layer,
                    &mut self.scratch_order,
                    &mut self.latest_rows[0],
                );
            }
            (Form::Prompt, _) => unreachable!(),
        }

        // 6. Write this step into the lazily-grown sample ring (at most `max_depths + 1` slots).
        let ring_cap = self.max_depths + 1;
        let slot_idx = (decode_idx as usize) % ring_cap;
        while self.ring.len() <= slot_idx {
            self.ring.push(RingSlot::new(self.experts));
        }
        let slot = &mut self.ring[slot_idx];
        slot.occupied = true;
        slot.decode_idx = decode_idx;
        slot.step = step;
        slot.form = form;
        slot.front_depth = front_depth;
        slot.fell_recorded = false;
        slot.route.copy_from_slice(&self.latest_rows[0]);
        slot.history.copy_from_slice(&self.latest_history);
        slot.picked.copy_from_slice(&self.scratch_picked);
        if self.is_live && front_depth > 1 {
            // Rows 1..front_depth may become row `f - 1` if row `f` falls.
            slot.ensure_front_rows(front_depth - 1, self.experts);
            for j in 1..front_depth {
                slot.front_rows[j - 1].copy_from_slice(&self.latest_rows[j]);
            }
        }

        self.latest_decode_step = Some(step);
        self.latest_meta = Some(LatestMeta { step, form, front_depth });
    }

    /// Log a prompt union sample: accumulated across all prefill chunks of a
    /// request (reset cleanly when `decode_count` advances for a new request),
    /// counted against the next `DEPTHS` decode steps in `route[Form::Prompt]`
    /// and blended into `history_prob` for the next `⌈E / k⌉` decode steps.
    /// Never an outcome and never enters decode history.
    pub(super) fn log_prompt(&mut self, step: Step, union: &Rows, history: &History, layer: usize) {
        assert_eq!(union.experts, self.experts, "prompt union experts {} != layer experts {}", union.experts, self.experts);
        self.set_active(union.choices);
        if self.prompt_logged_at != Some(self.decode_count) {
            self.prompt_accum.fill(0.0);
            self.prompt_rows = 0.0;
            self.latest_chose[0].fill(false);
        }
        extract_rows_keys(union, None, None, &mut self.scratch_keys, Some(self.scratch_picked.as_mut_slice()));
        let chunk_rows = union.depths.len().max(1) as f64;
        self.prompt_rows += chunk_rows;
        let denom = self.prompt_rows;
        for e in 0..self.experts {
            self.prompt_accum[e] += chunk_rows * f64::from(self.scratch_keys[e]);
            self.latest_chose[0][e] |= self.scratch_picked[e];
            let avg = (self.prompt_accum[e] / denom).clamp(0.0, 1.0) as f32;
            self.scratch_keys[e] = if avg == 0.0 { 0.0 } else { avg };
        }
        build_route_sample(
            self.experts,
            &self.scratch_keys,
            history,
            layer,
            &mut self.scratch_order,
            &mut self.prompt_sample,
        );
        self.prompt_base = Some(self.decode_count);
        self.prompt_logged_at = Some(self.decode_count);
        self.latest_rows[0].copy_from_slice(&self.prompt_sample);
        build_history_sample(
            self.experts,
            self.active,
            self.decode_count,
            history,
            layer,
            Some((self.prompt_sample.as_slice(), self.decode_count)),
            &mut self.scratch_order,
            &mut self.latest_history,
        );
        self.latest_meta = Some(LatestMeta { step, form: Form::Prompt, front_depth: 0 });
    }

    /// Record that on front step `step`, guessed row `first_fell_row` (`1 <= f <= d`)
    /// was the first row that fell. Counts row `f - 1`'s route sample into `fell[g]`
    /// for every decode step at lead `g ∈ 1..=DEPTHS` after row `f - 1` (both
    /// already-logged steps in the ring and future decode steps).
    pub(super) fn record_fell(&mut self, step: Step, first_fell_row: usize, epoch: u64) {
        assert!(
            (1..DEPTHS).contains(&first_fell_row),
            "first_fell_row {first_fell_row} out of range 1..{DEPTHS}"
        );
        if !self.is_live {
            return;
        }
        let Some(ring_pos) = self.ring.iter().position(|s| s.occupied && s.step == step) else {
            return;
        };
        if self.ring[ring_pos].fell_recorded || self.ring[ring_pos].form != Form::Front {
            return;
        }
        assert!(
            first_fell_row <= self.ring[ring_pos].front_depth,
            "first_fell_row {first_fell_row} exceeds step {step}'s front_depth {}",
            self.ring[ring_pos].front_depth
        );
        self.ring[ring_pos].fell_recorded = true;

        let origin_idx = self.ring[ring_pos].decode_idx + (first_fell_row - 1) as u64;
        let max_depths = self.max_depths;
        let experts = self.experts;
        let initial_epoch = self.initial_epoch;

        // Place row `f - 1`'s sample into a free slot of `pending_fell` (grow
        // lazily up to `max_depths`, or reuse the oldest slot once full).
        let fell_slot_idx = if let Some(idx) = self.pending_fell.iter().position(|s| s.origin_idx.is_none()) {
            idx
        } else if self.pending_fell.len() < max_depths {
            self.pending_fell.push(FellSlot::new(experts));
            self.pending_fell.len() - 1
        } else {
            self.pending_fell
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.origin_idx.unwrap_or(0))
                .map(|(i, _)| i)
                .expect("pending_fell has at least 1 slot")
        };

        if first_fell_row == 1 {
            self.pending_fell[fell_slot_idx].sample.copy_from_slice(&self.ring[ring_pos].route);
        } else {
            self.pending_fell[fell_slot_idx]
                .sample
                .copy_from_slice(&self.ring[ring_pos].front_rows[first_fell_row - 2]);
        }

        // Retroactively settle any outcomes already logged in `ring` after `origin_idx`.
        let mut fell_dirty = [false; DEPTHS];
        {
            let sample = &self.pending_fell[fell_slot_idx].sample;
            for other in &self.ring {
                if !other.occupied || other.decode_idx <= origin_idx {
                    continue;
                }
                let lead = (other.decode_idx - origin_idx) as usize;
                if lead <= max_depths {
                    let table = self.fell[lead - 1]
                        .get_or_insert_with(|| Box::new(CellTable::new(experts, initial_epoch)));
                    count_sample(table, sample, &other.picked, &mut self.scratch_class, epoch);
                    fell_dirty[lead - 1] = true;
                }
            }
            for g in 0..max_depths {
                if fell_dirty[g] {
                    self.fell[g].as_mut().expect("dirty fell table exists").rebuild_prefix(epoch);
                }
            }
        }

        // Keep active if more leads `<= max_depths` remain in the future.
        self.pending_fell[fell_slot_idx].origin_idx = if self.decode_count <= origin_idx + max_depths as u64 {
            Some(origin_idx)
        } else {
            None
        };
    }

    /// Calibrated route probability for `expert` at `lead ∈ 1..=DEPTHS` from
    /// `row` (`0..=latest_front_depth()`) of the latest logged step, in O(1).
    pub(super) fn route_prob(&self, form: Form, lead: usize, row: usize, expert: usize, epoch: u64) -> f64 {
        assert!((1..=DEPTHS).contains(&lead), "route lead {lead} out of range 1..={DEPTHS}");
        assert!(expert < self.experts, "expert {expert} out of range 0..{}", self.experts);
        if lead > self.route_horizon(form) {
            return self.history_prob(expert, epoch);
        }
        match self.latest_meta {
            Some(meta) => {
                assert!(row <= meta.front_depth, "row {row} exceeds latest front_depth {}", meta.front_depth);
                let entry = self.latest_rows[row][expert];
                let prior = match form {
                    Form::Verify => self.history_prob(expert, epoch),
                    Form::Front | Form::Prompt => f64::from(entry.prior),
                };
                Self::table_prob(
                    &self.route[form.index()][lead - 1],
                    usize::from(entry.lo),
                    usize::from(entry.hi),
                    prior,
                    epoch,
                )
            }
            None => self.history_prob(expert, epoch),
        }
    }

    /// Calibrated `fell` probability for `expert` at `lead ∈ 1..=DEPTHS` from
    /// standing row `row` (`0..=latest_front_depth()`), in O(1). Panics on
    /// shadow calibration.
    pub(super) fn fell_prob(&self, lead: usize, row: usize, expert: usize, epoch: u64) -> f64 {
        assert!((1..=DEPTHS).contains(&lead), "fell lead {lead} out of range 1..={DEPTHS}");
        assert!(expert < self.experts, "expert {expert} out of range 0..{}", self.experts);
        assert!(self.is_live, "fell_prob called on shadow calibration");
        if lead > self.max_depths {
            return self.history_prob(expert, epoch);
        }
        match self.latest_meta {
            Some(meta) => {
                assert!(row <= meta.front_depth, "row {row} exceeds latest front_depth {}", meta.front_depth);
                let entry = self.latest_rows[row][expert];
                Self::table_prob(
                    &self.fell[lead - 1],
                    usize::from(entry.lo),
                    usize::from(entry.hi),
                    f64::from(entry.prior),
                    epoch,
                )
            }
            None => self.history_prob(expert, epoch),
        }
    }

    /// Calibrated history probability `p_∞` for `expert` (past lead `max_depths`),
    /// in O(1): routes through the epoch-rolling `history` [`CellTable`]
    /// (settled at lead `max_depths + 1`) with the live measured half-life's
    /// decayed rate (blended with any recent prompt union) as its Laplace prior.
    /// Cold start returns exact `k / E`; short-`h` 1-shot spikes calibrate down
    /// to their true far-horizon reuse rate while stationary Zipf experts retain
    /// their true rate.
    pub(super) fn history_prob(&self, expert: usize, epoch: u64) -> f64 {
        assert!(expert < self.experts, "expert {expert} out of range 0..{}", self.experts);
        let entry = self.latest_history[expert];
        Self::table_prob(
            &self.history,
            usize::from(entry.lo),
            usize::from(entry.hi),
            f64::from(entry.prior),
            epoch,
        )
    }

    /// Calibrated prompt-table probabilities at leads `1..=DEPTHS` for each
    /// seed `(expert, count, last)` in `routed`, ordered by `(count_share, history_rank)`
    /// against `route[Form::Prompt]`. Used by `Reuse::rank_union` at seed time.
    pub(super) fn prompt_probs(
        &self,
        routed: &[(usize, u32, Step)],
        epoch: u64,
        history: &History,
        layer: usize,
    ) -> Vec<[f64; DEPTHS]> {
        if routed.is_empty() {
            return Vec::new();
        }
        let total: f64 = routed.iter().map(|&(_, count, _)| f64::from(count)).sum();
        let mut keys = vec![0.0f32; self.experts];
        if total > 0.0 {
            for &(expert, count, _) in routed {
                assert!(expert < self.experts, "prompt expert {expert} out of range 0..{}", self.experts);
                let p = (self.active as f64 * f64::from(count) / total).min(1.0) as f32;
                keys[expert] = if p == 0.0 { 0.0 } else { p };
            }
        }
        let mut order: Vec<usize> = (0..self.experts).collect();
        let mut entries = vec![SampleEntry::default(); self.experts];
        build_route_sample(self.experts, &keys, history, layer, &mut order, &mut entries);
        let prompt_tables = &self.route[Form::Prompt.index()];
        routed
            .iter()
            .map(|&(expert, _, _)| {
                let entry = entries[expert];
                let lo = usize::from(entry.lo);
                let hi = usize::from(entry.hi);
                let prior = f64::from(entry.prior);
                std::array::from_fn(|g| Self::table_prob(&prompt_tables[g], lo, hi, prior, epoch))
            })
            .collect()
    }

    #[inline]
    pub(super) fn last_read(&self, expert: usize) -> Option<Step> {
        self.last_read[expert]
    }

    #[inline]
    pub(super) fn latest_step(&self) -> Option<Step> {
        self.latest_meta.map(|m| m.step)
    }

    /// Whether at least one decode step has been logged at this layer (once
    /// true, prompt evidence has left the ranking, §4).
    #[inline]
    pub(super) fn has_decode(&self) -> bool {
        self.latest_decode_step.is_some()
    }

    /// Whether this layer's `in_top` pool reflects a *previous* request's
    /// decode (decode steps have run since the last prompt was logged), rather
    /// than the current request's prefill.
    pub(super) fn has_stale_decode_pool(&self) -> bool {
        self.decode_count > 0 && self.prompt_logged_at != Some(self.decode_count)
    }

    #[inline]
    pub(super) fn latest_form(&self) -> Option<Form> {
        self.latest_meta.map(|m| m.form)
    }

    #[inline]
    pub(super) fn latest_front_depth(&self) -> usize {
        self.latest_meta.map_or(0, |m| m.front_depth)
    }

    #[inline]
    pub(super) fn row_chose(&self, row: usize, expert: usize) -> bool {
        let meta = self.latest_meta.expect("row_chose called before any step logged");
        assert!(row <= meta.front_depth, "row {row} exceeds latest front_depth {}", meta.front_depth);
        assert!(expert < self.experts, "expert {expert} out of range 0..{}", self.experts);
        self.latest_chose[row][expert]
    }

    #[inline]
    pub(super) fn experts(&self) -> usize {
        self.experts
    }

    #[inline]
    pub(super) fn active(&self) -> usize {
        self.active
    }

    /// Update the layer's `active` (choices per row) before the first decode
    /// step is logged, so cold-start prior `k / E` uses the observed `choices`.
    pub(super) fn set_active(&mut self, active: usize) {
        if self.decode_count == 0 && active > 0 && active <= self.experts {
            self.active = active;
            let cold_prior = (active as f64 / self.experts as f64) as f32;
            for entry in &mut self.latest_history {
                entry.prior = cold_prior;
            }
        }
    }

    /// Feed one shadow read (decode or prompt) into this shadow calibration.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_shadow(
        &mut self,
        step: Step,
        experts: &[usize],
        rows: Option<&Rows>,
        evidence: super::Evidence,
        epoch: u64,
        history: &History,
        layer: usize,
    ) {
        match evidence {
            super::Evidence::Decode => {
                if let Some(rows) = rows {
                    self.set_active(rows.choices);
                } else if !experts.is_empty() {
                    self.set_active(experts.len());
                }
                let form = if rows.is_some_and(Rows::verifies) { Form::Verify } else { Form::Front };
                self.log_decode(step, form, rows, experts, epoch, None, history, layer);
            }
            super::Evidence::Prompt => {
                if let Some(rows) = rows {
                    self.log_prompt(step, rows, history, layer);
                } else {
                    let counts: Vec<(usize, u32)> = experts.iter().map(|&e| (e, 1)).collect();
                    let union = Rows::union(self.experts, self.active.max(1), &counts);
                    self.log_prompt(step, &union, history, layer);
                }
            }
        }
    }

    /// Precompute true-route shadow inputs into `scratch` (zero allocations)
    /// for one step at this layer. Depth `d` knows the true routes of tokens
    /// `1..=min(d, truths.len())` ahead (standing with rate 1), and carries
    /// forward past `d` from the deepest known true route's `route[Form::Front]`
    /// calibration table. Per-expert probabilities are computed on the stack
    /// only for the experts queried by [`ShadowPrepared::rank`].
    pub(super) fn prepare_shadow<'a>(
        &'a mut self,
        scratch: &'a mut ShadowScratch,
        truths: &[super::shadows::TrueRoute<'_>],
        epoch: u64,
        history: &History,
        layer: usize,
    ) -> ShadowPrepared<'a> {
        let experts = self.experts;
        let known = truths.len().min(self.max_depths.saturating_sub(1));
        for (idx, truth) in truths.iter().take(known).enumerate() {
            let chose = &mut scratch.truth_chosen[idx][..experts];
            chose.fill(false);
            match truth {
                super::shadows::TrueRoute::Row { rows, depth } => {
                    extract_rows_keys(rows, Some(*depth), None, &mut self.scratch_keys, Some(chose));
                }
                super::shadows::TrueRoute::Unkeyed { experts: picked } => {
                    self.scratch_keys.fill(0.0);
                    for &e in *picked {
                        self.scratch_keys[e] = 1.0;
                        chose[e] = true;
                    }
                }
            }
            let entries = &mut scratch.truth_samples[idx][..experts];
            build_route_sample(
                experts,
                &self.scratch_keys,
                history,
                layer,
                &mut self.scratch_order,
                entries,
            );
        }

        let form = self.latest_form().unwrap_or(Form::Front);
        for e in 0..experts {
            scratch.history_probs[e] = self.history_prob(e, epoch);
        }
        ShadowPrepared {
            cal: self,
            scratch,
            known,
            epoch,
            form,
            has_latest: self.latest_meta.is_some(),
        }
    }

    /// Exact memory footprint in bytes (bounded in `(experts, max_depths)`,
    /// constant in run length and τ after warmup).
    pub(super) fn bytes(&self) -> u64 {
        let mut total = size_of::<Self>() as u64;
        for form in &self.route {
            for table in form.iter().flatten() {
                total += size_of::<CellTable>() as u64 + table.bytes();
            }
        }
        for table in self.fell.iter().flatten() {
            total += size_of::<CellTable>() as u64 + table.bytes();
        }
        if let Some(history) = &self.history {
            total += size_of::<CellTable>() as u64 + history.bytes();
        }
        if let Some(far) = &self.far {
            total += size_of::<CellTable>() as u64 + far.bytes();
        }
        total += (self.ring.capacity() * size_of::<RingSlot>()) as u64;
        for slot in &self.ring {
            total += slot.bytes();
        }
        total += (self.pending_fell.capacity() * size_of::<FellSlot>()) as u64;
        for slot in &self.pending_fell {
            total += slot.bytes();
        }
        total += (self.prompt_sample.capacity() * size_of::<SampleEntry>()) as u64;
        total += (self.prompt_accum.capacity() * size_of::<f64>()) as u64;
        total += (self.far_sample.capacity() * size_of::<SampleEntry>()) as u64;
        total += (self.last_read.capacity() * size_of::<Option<Step>>()) as u64;
        total += (self.latest_rows.capacity() * size_of::<Vec<SampleEntry>>()) as u64;
        for row in &self.latest_rows {
            total += (row.capacity() * size_of::<SampleEntry>()) as u64;
        }
        total += (self.latest_chose.capacity() * size_of::<Vec<bool>>()) as u64;
        for row in &self.latest_chose {
            total += (row.capacity() * size_of::<bool>()) as u64;
        }
        total += (self.latest_history.capacity() * size_of::<SampleEntry>()) as u64;
        total += (self.scratch_order.capacity() * size_of::<usize>()) as u64;
        total += (self.scratch_keys.capacity() * size_of::<f32>()) as u64;
        total += (self.scratch_picked.capacity() * size_of::<bool>()) as u64;
        total += (self.scratch_class.capacity() * size_of::<(f64, u16)>()) as u64;
        total
    }

    /// Per-decile comparison of `history` (lead `L + 1`) against `far` (lead
    /// `⌈τ_0 / ā⌉`) for the §8 A1 diagnostic printout; `None` on shadow
    /// calibration.
    pub(super) fn far_deciles(&self, epoch: u64) -> Option<[FarDecile; 10]> {
        if !self.is_live {
            return None;
        }
        let mut out = [FarDecile::default(); 10];
        for (d, slot) in out.iter_mut().enumerate() {
            let lo = d * self.experts / 10;
            let hi = (d + 1) * self.experts / 10;
            if lo < hi {
                let (h_seen, h_used) = self
                    .history
                    .as_deref()
                    .map_or((0.0, 0.0), |table| table.pooled(lo, hi, epoch));
                let (f_seen, f_used) = self
                    .far
                    .as_deref()
                    .map_or((0.0, 0.0), |table| table.pooled(lo, hi, epoch));
                *slot = FarDecile {
                    history_seen: h_seen,
                    history_used: h_used,
                    far_seen: f_seen,
                    far_used: f_used,
                };
            }
        }
        Some(out)
    }
}

/// Reusable fixed-size scratch buffers for [`Calibration::prepare_shadow`]
/// (allocated once in [`super::shadows::Shadows::new`], constant in run length).
pub(super) struct ShadowScratch {
    truth_chosen: Vec<Vec<bool>>,
    truth_samples: Vec<Vec<SampleEntry>>,
    history_probs: Vec<f64>,
}

impl ShadowScratch {
    pub(super) fn new(max_experts: usize, max_depths: usize) -> Self {
        let slots = max_depths.saturating_sub(1);
        Self {
            truth_chosen: (0..slots).map(|_| vec![false; max_experts]).collect(),
            truth_samples: (0..slots).map(|_| vec![SampleEntry::default(); max_experts]).collect(),
            history_probs: vec![0.0; max_experts],
        }
    }

    pub(super) fn bytes(&self) -> u64 {
        let chosen: usize = self.truth_chosen.capacity() * size_of::<Vec<bool>>()
            + self.truth_chosen.iter().map(|v| v.capacity() * size_of::<bool>()).sum::<usize>();
        let samples: usize = self.truth_samples.capacity() * size_of::<Vec<SampleEntry>>()
            + self.truth_samples.iter().map(|v| v.capacity() * size_of::<SampleEntry>()).sum::<usize>();
        let hist = self.history_probs.capacity() * size_of::<f64>();
        (size_of::<Self>() + chosen + samples + hist) as u64
    }
}

/// Borrowed view of one layer's prepared true-route shadow inputs.
pub(super) struct ShadowPrepared<'a> {
    cal: &'a Calibration,
    scratch: &'a ShadowScratch,
    known: usize,
    epoch: u64,
    form: Form,
    has_latest: bool,
}

impl ShadowPrepared<'_> {
    #[inline]
    pub(super) fn rank(
        &self,
        expert: usize,
        depth: usize,
        horizon: super::super::horizon::Horizon,
        a_bar: f64,
    ) -> super::Rank {
        let eff = depth.min(self.known);
        let p_inf = self.scratch.history_probs[expert];
        let probs = std::array::from_fn(|index| {
            let g = index + 1;
            if g <= eff {
                f64::from(u8::from(self.scratch.truth_chosen[g - 1][expert]))
            } else if eff == 0 {
                if self.has_latest && g <= self.cal.route_horizon(self.form) {
                    let entry = self.cal.latest_rows[0][expert];
                    let prior = match self.form {
                        Form::Verify => p_inf,
                        Form::Front | Form::Prompt => f64::from(entry.prior),
                    };
                    Calibration::table_prob(
                        &self.cal.route[self.form.index()][g - 1],
                        usize::from(entry.lo),
                        usize::from(entry.hi),
                        prior,
                        self.epoch,
                    )
                } else {
                    p_inf
                }
            } else {
                let lead = g - eff;
                if lead <= self.cal.max_depths {
                    let entry = self.scratch.truth_samples[eff - 1][expert];
                    Calibration::table_prob(
                        &self.cal.route[Form::Front.index()][lead - 1],
                        usize::from(entry.lo),
                        usize::from(entry.hi),
                        f64::from(entry.prior),
                        self.epoch,
                    )
                } else {
                    p_inf
                }
            }
        });
        let expected = super::expected_from(probs, p_inf, horizon, a_bar, None);
        super::Rank::new(expected, self.cal.last_read[expert])
    }
}

/// Extract per-expert keys `1 - Π_r (1 - π_{r,e})` and optional `chose` flags
/// from `rows` for all rows matching `depth_filter` (or all rows if `None`).
/// Returns `true` if at least one row matched; if none matched, falls back to
/// `fallback_picked` (or all zeros when `None`).
fn extract_rows_keys(
    rows: &Rows,
    depth_filter: Option<u8>,
    fallback_picked: Option<&[bool]>,
    out_keys: &mut [f32],
    mut out_chose: Option<&mut [bool]>,
) -> bool {
    let experts = rows.experts;
    if let Some(chose) = out_chose.as_deref_mut() {
        chose.fill(false);
    }
    let mut matched = false;
    let mut first = true;
    for (r, &depth) in rows.depths.iter().enumerate() {
        if let Some(target) = depth_filter {
            if depth != target {
                continue;
            }
        }
        matched = true;
        let priors = &rows.priors[r * experts..][..experts];
        let chosen = &rows.chosen[r * experts..][..experts];
        if first {
            for e in 0..experts {
                let p = priors[e];
                assert!(p.is_finite(), "non-finite router prior {p} at row {r} expert {e}");
                let p = p.clamp(0.0, 1.0);
                out_keys[e] = if p == 0.0 { 0.0 } else { p };
                if let Some(chose) = out_chose.as_deref_mut() {
                    chose[e] = chosen[e];
                }
            }
            first = false;
        } else {
            for e in 0..experts {
                let p = priors[e];
                assert!(p.is_finite(), "non-finite router prior {p} at row {r} expert {e}");
                let p = f64::from(p.clamp(0.0, 1.0));
                let prev = f64::from(out_keys[e]);
                let combined = (1.0 - (1.0 - prev) * (1.0 - p)).clamp(0.0, 1.0) as f32;
                out_keys[e] = if combined == 0.0 { 0.0 } else { combined };
                if let Some(chose) = out_chose.as_deref_mut() {
                    if chosen[e] {
                        chose[e] = true;
                    }
                }
            }
        }
    }
    if !matched {
        for e in 0..experts {
            let p = fallback_picked.is_some_and(|picked| picked[e]);
            out_keys[e] = if p { 1.0 } else { 0.0 };
            if let Some(chose) = out_chose.as_deref_mut() {
                chose[e] = p;
            }
        }
    }
    matched
}

/// Build a route-order sample descending by `(key, history_rank)` with exact
/// tie classes `[lo, hi)`.
fn build_route_sample(
    experts: usize,
    keys: &[f32],
    history: &History,
    layer: usize,
    order: &mut [usize],
    out: &mut [SampleEntry],
) {
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    order.sort_unstable_by(|&a, &b| {
        keys[b]
            .total_cmp(&keys[a])
            .then_with(|| history.rank(layer, b).cmp(&history.rank(layer, a)))
    });
    let mut lo = 0;
    while lo < experts {
        let rep = order[lo];
        let rep_key = keys[rep].to_bits();
        let rep_rank = history.rank(layer, rep);
        let mut hi = lo + 1;
        while hi < experts {
            let e = order[hi];
            if keys[e].to_bits() == rep_key && history.rank(layer, e) == rep_rank {
                hi += 1;
            } else {
                break;
            }
        }
        for &e in &order[lo..hi] {
            out[e] = SampleEntry { lo: lo as u16, hi: hi as u16, prior: keys[e] };
        }
        lo = hi;
    }
}

/// History probability `p_∞` for `expert`:
/// - Cold start before the first decode step with no prompt logged:
///   exact `k / E`.
/// - Warmup after a logged prompt union (`decode_count - logged_at < ⌈E / k⌉`,
///   including before decode step 1 when `history.rate` is `None`):
///   smooth linear blend from `prompt_prior` into `history.rate` (or `k / E`
///   before decode step 1) over one expert-space pass `⌈E / k⌉` with zero
///   tuned constants.
/// - Steady state (`decode_count - logged_at >= ⌈E / k⌉`, or no prompt logged):
///   `history.rate(layer, expert)` directly from the live measured half-life.
#[inline]
fn history_prior(
    history: &History,
    layer: usize,
    decode_count: u64,
    prompt_sample: Option<(&[SampleEntry], u64)>,
    expert: usize,
    active: usize,
    experts: usize,
) -> f64 {
    let uniform = active as f64 / experts as f64;
    let rate = history.rate(layer, expert).unwrap_or(uniform);
    if let Some((prompt, logged_at)) = prompt_sample {
        let min_steps = experts.div_ceil(active) as u64;
        let steps_since_prompt = decode_count.saturating_sub(logged_at);
        if steps_since_prompt < min_steps {
            let w = (min_steps - steps_since_prompt) as f64 / min_steps as f64;
            let p = f64::from(prompt[expert].prior);
            // When a prior request already populated `history` (`logged_at > 0`),
            // discount its stale rate toward uniform `k / E` during the fresh
            // prompt's `⌈E / k⌉` warmup so a disjoint request trusts its own
            // `prompt_prior` cleanly just like request 1.
            let effective_rate = if logged_at > 0 {
                let s = steps_since_prompt as f64 / min_steps as f64;
                s * rate + (1.0 - s) * uniform
            } else {
                rate
            };
            return ((1.0 - w) * effective_rate + w * p).clamp(0.0, 1.0);
        }
    }
    rate
}

/// Build a history-order sample descending by `(history_prior, history_rank)`
/// with exact tie classes `[lo, hi)` for §8 A1 diagnostics.
#[allow(clippy::too_many_arguments)]
fn build_history_sample(
    experts: usize,
    active: usize,
    decode_count: u64,
    history: &History,
    layer: usize,
    prompt_sample: Option<(&[SampleEntry], u64)>,
    order: &mut [usize],
    out: &mut [SampleEntry],
) {
    let prior_of = |e: usize| history_prior(history, layer, decode_count, prompt_sample, e, active, experts) as f32;
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    order.sort_unstable_by(|&a, &b| {
        prior_of(b)
            .total_cmp(&prior_of(a))
            .then_with(|| history.rank(layer, b).cmp(&history.rank(layer, a)))
    });
    let mut lo = 0;
    while lo < experts {
        let rep = order[lo];
        let rep_prior = prior_of(rep).to_bits();
        let rep_rank = history.rank(layer, rep);
        let mut hi = lo + 1;
        while hi < experts {
            let e = order[hi];
            if prior_of(e).to_bits() == rep_prior && history.rank(layer, e) == rep_rank {
                hi += 1;
            } else {
                break;
            }
        }
        for &e in &order[lo..hi] {
            out[e] = SampleEntry { lo: lo as u16, hi: hi as u16, prior: prior_of(e) };
        }
        lo = hi;
    }
}

/// Count every tie class of `sample` against outcome `picked` into `table` in
/// O(E) using `scratch` (`(class_used, class_hi)` indexed by `lo`).
fn count_sample(
    table: &mut CellTable,
    sample: &[SampleEntry],
    picked: &[bool],
    scratch: &mut [(f64, u16)],
    epoch: u64,
) {
    let experts = sample.len();
    for (e, &entry) in sample.iter().enumerate() {
        let lo = usize::from(entry.lo);
        let acc = &mut scratch[lo];
        if picked[e] {
            acc.0 += 1.0;
        }
        acc.1 = entry.hi;
    }
    let mut lo = 0;
    while lo < experts {
        let (class_used, hi) = scratch[lo];
        let hi = usize::from(hi);
        debug_assert!(hi > lo && hi <= experts, "tie class partition broken at {lo}..{hi}");
        table.count_class(lo, hi, class_used, epoch);
        scratch[lo] = (0.0, 0);
        lo = hi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "expected {a} ≈ {b} (diff {}, eps {eps})", (a - b).abs());
    }

    fn decode(
        cal: &mut Calibration,
        history: &mut History,
        step: Step,
        form: Form,
        rows: Option<&Rows>,
        chosen: &[usize],
        epoch: u64,
        tau_steps: Option<usize>,
    ) {
        history.read_decode(0, step, chosen, 4);
        cal.log_decode(step, form, rows, chosen, epoch, tau_steps, history, 0);
    }

    #[test]
    fn a_tie_class_spreads_over_its_positions() {
        // 4 experts, k = 1 active.
        let mut cal = Calibration::new(4, 1, 0);
        let mut history = History::new(&[4]);

        // Cold start: all 4 experts tied in [0, 4) with prior k/E = 0.25.
        for e in 0..4 {
            approx(cal.history_prob(e, 0), 0.25, 1e-12);
            approx(cal.route_prob(Form::Front, 1, 0, e, 0), 0.25, 1e-12);
        }

        // Step 1: keyed row where experts 0 and 1 tie at key 0.4, experts 2 and 3 tie at key 0.1.
        let rows1 = Rows::of(1, &[0], &[0.4, 0.4, 0.1, 0.1]);
        decode(&mut cal, &mut history, 1, Form::Front, Some(&rows1), &[0], 0, None);
        // Expert 0 was read so history ranks 0 > 1; 2 and 3 both have key=0.1 and unread -> tie class [2, 4).
        assert_eq!(cal.latest_rows[0][0].lo, 0);
        assert_eq!(cal.latest_rows[0][0].hi, 1);
        assert_eq!(cal.latest_rows[0][1].lo, 1);
        assert_eq!(cal.latest_rows[0][1].hi, 2);
        assert_eq!(cal.latest_rows[0][2].lo, 2);
        assert_eq!(cal.latest_rows[0][2].hi, 4);
        assert_eq!(cal.latest_rows[0][3].lo, 2);
        assert_eq!(cal.latest_rows[0][3].hi, 4);

        // Step 2: outcome chooses expert 2 (one of the two tied experts in [2, 4)).
        let rows2 = Rows::of(1, &[0], &[0.25, 0.25, 0.25, 0.25]);
        decode(&mut cal, &mut history, 2, Form::Front, Some(&rows2), &[2], 0, None);

        // Every position 0..4 received seen = 1.0 on route[Front][0].
        for cell in 0..4 {
            let (seen, _) = cal.route[Form::Front.index()][0]
                .as_deref()
                .expect("route[Front][0] allocated")
                .pooled(cell, cell + 1, 0);
            approx(seen, 1.0, 1e-6);
        }
        // In tie class [2, 4), 1 of 2 members was chosen -> each cell in 2..4 gets used = 0.5.
        for cell in 2..4 {
            let (_, used) = cal.route[Form::Front.index()][0]
                .as_deref()
                .expect("route[Front][0] allocated")
                .pooled(cell, cell + 1, 0);
            approx(used, 0.5, 1e-6);
        }

        // All E tied: fresh calibration with equal keys and no reads.
        let mut tied = Calibration::new(4, 1, 0);
        let mut tied_hist = History::new(&[4]);
        let flat = Rows::of(1, &[0], &[1.0, 1.0, 1.0, 1.0]);
        // Step 1 with unkeyed empty chosen so all 4 stay tied at unread.
        decode(&mut tied, &mut tied_hist, 1, Form::Front, None, &[], 0, None);
        assert_eq!(tied.latest_rows[0][0].lo, 0);
        assert_eq!(tied.latest_rows[0][0].hi, 4);
        // Step 2 chooses expert 3: every cell 0..4 gets seen=1, used=0.25.
        decode(&mut tied, &mut tied_hist, 2, Form::Front, None, &[3], 0, None);
        for cell in 0..4 {
            let (seen, used) = tied.route[Form::Front.index()][0]
                .as_deref()
                .expect("route[Front][0] allocated")
                .pooled(cell, cell + 1, 0);
            approx(seen, 1.0, 1e-6);
            approx(used, 0.25, 1e-6);
        }
        // Unkeyed reads order by history rank: expert 3 was read > others unread.
        assert_eq!(tied.latest_rows[0][3].lo, 0);
        assert_eq!(tied.latest_rows[0][3].hi, 1);
        // Equal single-use history, later last_read wins: step 3 chooses expert 1 -> 1 outranks 3.
        decode(&mut tied, &mut tied_hist, 3, Form::Front, Some(&flat), &[1], 0, None);
        assert_eq!(tied.latest_history[1].lo, 0);
        assert_eq!(tied.latest_history[1].hi, 1);
        assert_eq!(tied.latest_history[3].lo, 1);
        assert_eq!(tied.latest_history[3].hi, 2);
    }

    #[test]
    fn the_carry_is_counted_at_each_lead_against_the_steps_that_followed() {
        let mut cal = Calibration::new(4, 1, 0);
        let mut history = History::new(&[4]);
        // Alternating pattern: step 1 -> [0], step 2 -> [1], step 3 -> [0], step 4 -> [1]...
        // Unkeyed reads: chosen expert has prior 1.0 at cell [0, 1), unchosen have prior 0.0.
        // At lead 1: 5 settled samples, chosen expert never chosen next (used = 0 -> (0 + 1) / (5 + 1) = 1/6).
        // At lead 2: 4 settled samples, chosen expert always chosen 2 steps later (used = 4 -> (4 + 1) / (4 + 1) = 1).
        for step in 1..=6u64 {
            let chosen = if step % 2 == 1 { [0] } else { [1] };
            decode(&mut cal, &mut history, step, Form::Front, None, &chosen, 0, None);
        }
        // Latest step is 6 (chose [1]): expert 1 is at cell [0, 1) with prior 1.0.
        approx(cal.route_prob(Form::Front, 1, 0, 1, 0), 1.0 / 6.0, 1e-6);
        approx(cal.route_prob(Form::Front, 2, 0, 1, 0), 1.0, 1e-6);

        // Verify form has its own separate route table at lead 1 (uncounted so far -> reads p_inf prior)
        // and falls back directly to history_prob at leads >= 2.
        approx(cal.route_prob(Form::Verify, 1, 0, 1, 0), cal.history_prob(1, 0), 1e-6);
        approx(cal.route_prob(Form::Verify, 2, 0, 1, 0), cal.history_prob(1, 0), 1e-6);

        // Prompt union is counted into route[Prompt] against the next L decode steps,
        // and never enters decode history or counts as an outcome.
        let union = Rows::union(4, 1, &[(2, 3), (3, 1)]);
        cal.log_prompt(7, &union, &history, 0);
        assert_eq!(cal.last_read(2), None, "prompt never updates decode last_read");
        assert_eq!(history.rate(0, 2), Some(0.0), "prompt never enters decode history");
        // Next decode step chooses expert 2 -> settles prompt lead 1.
        decode(&mut cal, &mut history, 8, Form::Front, None, &[2], 0, None);
        let (seen, _) = cal.route[Form::Prompt.index()][0]
            .as_deref()
            .expect("route[Prompt][0] allocated")
            .pooled(0, 4, 0);
        approx(seen, 4.0, 1e-6);

        // `fell` is counted only on front steps whose row fell.
        let front_rows = Rows::of(1, &[0, 1], &[0.9, 0.05, 0.03, 0.02, 0.05, 0.9, 0.03, 0.02]);
        decode(&mut cal, &mut history, 9, Form::Front, Some(&front_rows), &[0], 0, None);
        assert_eq!(cal.latest_front_depth(), 1);
        assert!(cal.row_chose(0, 0));
        assert!(cal.row_chose(1, 1));
        cal.record_fell(9, 1, 0);
        // Step 10 settles fell lead 1 (row 0 of step 9, where expert 0 was top-ranked).
        decode(&mut cal, &mut history, 10, Form::Front, None, &[3], 0, None);
        let (fell_seen, _) = cal.fell[0]
            .as_deref()
            .expect("fell[0] allocated")
            .pooled(0, 4, 0);
        approx(fell_seen, 4.0, 1e-6);
    }

    #[test]
    fn past_the_carried_leads_the_history_rate_prices_a_lead() {
        let mut cal = Calibration::new(8, 2, 0);
        let mut history = History::new(&[8]);
        let bytes_initial = cal.bytes() + history.bytes();

        // Cold start: history prior is k / E = 2 / 8 = 0.25.
        for e in 0..8 {
            approx(cal.history_prob(e, 0), 0.25, 1e-12);
        }

        // Expert 0 is read every step; expert 1 is read only once at step 20 (same recency as 0).
        // Experts 2..7 take turns as the second read on earlier steps.
        for step in 1..=20u64 {
            let other = if step == 20 { 1 } else { 2 + (step as usize % 6) };
            let mut chosen = [0, other];
            chosen.sort_unstable();
            decode(&mut cal, &mut history, step, Form::Front, None, &chosen, 0, Some(4));
        }

        // Expert 0 was read on every step (most recent & most frequent); expert 2 was read earlier.
        let p0 = cal.history_prob(0, 0);
        let p_cold = cal.history_prob(2, 0);
        assert!(p0 > 0.0 && p0 <= 1.0, "expert 0 rate {p0} out of (0, 1]");
        assert!(p0 > p_cold, "frequent expert 0 ({p0}) must outrank less frequent expert 2 ({p_cold})");

        // Memory footprint starts minimal before warmup and stays strictly
        // constant across 10,000 steps once exercised tables are allocated.
        let bytes_after_20 = cal.bytes() + history.bytes();
        assert!(bytes_initial <= bytes_after_20);
        for step in 21..=10_000u64 {
            decode(&mut cal, &mut history, step, Form::Front, None, &[0, 2], 0, Some(4));
        }
        assert_eq!(bytes_after_20, cal.bytes() + history.bytes(), "calibration size must be constant over 10,000 steps");

        // `far` table accumulated samples at lead tau_steps = 4.
        let deciles = cal.far_deciles(0).expect("live calibration has far_deciles");
        let total_far_seen: f64 = deciles.iter().map(|d| d.far_seen).sum();
        assert!(total_far_seen > 0.0, "far table must record samples at tau_steps");
    }

    #[test]
    fn history_prob_never_inverts_hottest_below_cold_expert() {
        let mut cal = Calibration::new(16, 2, 0);
        let mut history = History::new(&[16]);
        for step in 1..=5u64 {
            decode(&mut cal, &mut history, step, Form::Front, None, &[0, step as usize], 0, None);
        }
        decode(&mut cal, &mut history, 6, Form::Front, None, &[6, 7], 0, None);
        decode(&mut cal, &mut history, 7, Form::Front, None, &[8, 9], 0, None);
        decode(&mut cal, &mut history, 8, Form::Front, None, &[10, 11], 0, None);
        decode(&mut cal, &mut history, 9, Form::Front, None, &[12, 13], 0, None);
        for i in 1..=5u64 {
            decode(&mut cal, &mut history, 9 + i, Form::Front, None, &[i as usize, 5 + i as usize], 0, None);
        }

        let p_hot = cal.history_prob(0, 0);
        let p_cold = cal.history_prob(15, 0);
        assert!(
            p_hot > p_cold && p_hot > 0.0,
            "hottest expert 0 ({p_hot}) must strictly outrank never-used cold expert 15 ({p_cold})"
        );
    }

    #[test]
    fn prompt_prior_blends_past_depths_until_min_steps() {
        // Choose E = 4 * DEPTHS, k = 2 -> min_steps = ⌈E / k⌉ = 2 * DEPTHS > DEPTHS.
        let experts = 4 * DEPTHS;
        let min_steps = experts.div_ceil(2) as u64;
        let mut cal = Calibration::new(experts, 2, 0);
        let mut history = History::new(&[experts]);
        let union = Rows::union(experts, 2, &[(0, 10), (1, 1)]);
        cal.log_prompt(1, &union, &history, 0);

        // Log DEPTHS + 1 decode steps choosing only the last two experts, so
        // prompt_base is already cleared after DEPTHS steps while
        // decode_count = DEPTHS + 1 < min_steps.
        let tail = [experts - 2, experts - 1];
        let past_depths = DEPTHS as u64 + 1;
        for step in 2..=(1 + past_depths) {
            decode(&mut cal, &mut history, step, Form::Front, None, &tail, 0, None);
        }
        assert!(cal.prompt_base.is_none(), "prompt_base clears after DEPTHS steps");
        let p_prompt = cal.history_prob(0, 0);
        let p_unseen = cal.history_prob(2, 0);
        assert!(
            p_prompt > p_unseen,
            "prompt-seeded expert 0 ({p_prompt}) must still blend above unseen expert 2 ({p_unseen}) at steps < min_steps"
        );

        // Reaching decode_count = min_steps ends the prompt blend cleanly.
        for step in (2 + past_depths)..=(1 + min_steps) {
            decode(&mut cal, &mut history, step, Form::Front, None, &tail, 0, None);
        }
        approx(cal.history_prob(0, 0), cal.history_prob(2, 0), 1e-12);
    }

    #[test]
    fn multi_chunk_log_prompt_accumulates_and_history_prior_uses_prompt_before_decode() {
        let experts = 8;
        let mut cal = Calibration::new(experts, 2, 0);
        let mut history = History::new(&[experts]);

        // Log 3 prefill chunks before any decode step (`decode_count == 0`).
        // Chunk 1 routes expert 0, chunk 2 routes expert 1, chunk 3 routes expert 2.
        cal.log_prompt(1, &Rows::union(experts, 2, &[(0, 10)]), &history, 0);
        cal.log_prompt(2, &Rows::union(experts, 2, &[(1, 10)]), &history, 0);
        cal.log_prompt(3, &Rows::union(experts, 2, &[(2, 10)]), &history, 0);

        // Before decode step 1 (`history.rate` is None), `history_prob` must reflect
        // ALL 3 accumulated prompt chunks rather than returning flat `k / E` or
        // dropping chunks 1 and 2.
        let p_unseen = cal.history_prob(7, 0);
        for &e in &[0usize, 1, 2] {
            let p = cal.history_prob(e, 0);
            assert!(
                p > p_unseen,
                "accumulated prompt expert {e} ({p}) must strictly exceed unseen expert 7 ({p_unseen}) before decode step 1"
            );
        }
        approx(cal.history_prob(0, 0), cal.history_prob(1, 0), 1e-6);
        approx(cal.history_prob(1, 0), cal.history_prob(2, 0), 1e-6);

        // After a decode step advances `decode_count`, a new request's prompt resets
        // accumulation cleanly so the new prompt's expert 5 outranks old prompt expert 0.
        decode(&mut cal, &mut history, 4, Form::Front, None, &[6, 7], 0, None);
        cal.log_prompt(5, &Rows::union(experts, 2, &[(5, 10)]), &history, 0);
        assert!(
            cal.history_prob(5, 0) > cal.history_prob(0, 0),
            "new request's prompt expert 5 ({}) must outrank previous request's expert 0 ({})",
            cal.history_prob(5, 0),
            cal.history_prob(0, 0),
        );
    }

    #[test]
    fn disjoint_request_discounts_stale_prior_history_during_prompt_warmup() {
        let experts = 8;
        let active = 2;
        let mut cal = Calibration::new(experts, active, 0);
        let mut history = History::new(&[experts]);

        // Request 1: 20 decode steps heavily routing experts 0 and 1.
        for step in 1..=20u64 {
            decode(&mut cal, &mut history, step, Form::Front, None, &[0, 1], 0, None);
        }
        assert!(history.rate(0, 0).unwrap() > 0.8);

        // Request 2: disjoint prompt routing experts 4 and 5 (`logged_at = 20 > 0`).
        cal.log_prompt(21, &Rows::union(experts, active, &[(4, 10), (5, 10)]), &history, 0);
        // Step 1 of req 2 (`steps_since_prompt = 1 < min_steps = 4`): stale req 1 expert 0
        // is discounted toward uniform `k/E = 0.25`, so req 2's prompt expert 4 strongly leads.
        decode(&mut cal, &mut history, 22, Form::Front, None, &[4, 5], 0, None);
        let p_req2_prompt = cal.history_prob(4, 0);
        let p_req1_stale = cal.history_prob(0, 0);
        assert!(
            p_req2_prompt > p_req1_stale * 2.0,
            "req 2 prompt expert 4 ({p_req2_prompt}) must strongly outrank stale req 1 expert 0 ({p_req1_stale}) during warmup"
        );
    }
}
