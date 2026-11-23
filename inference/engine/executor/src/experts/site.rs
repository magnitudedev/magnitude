//! The CPU site:
//! a streamed decode layer's misses the route assigned to the CPU, each
//! computed whole on the host's cores from its tier record, in the device's
//! resident form where the host addresses it.
//!
//! - **Request.** The device writes a layer's request into the step's slot
//!   `step mod FLIGHT` of the layer's site table, a slot of the exchange
//!   (`streaming_words.seismic`): the rows (f32), the count and each
//!   entry's row, tier record, routing weight and expert, then the step as
//!   the slot's request (SLOT_REQUEST, one 64-bit release store). Nothing
//!   relays it: one of the host compute executor's workers
//!   (`compute::HostCompute`), the one noticing ([`Site::notice`]), polls
//!   the slots while a flight is issued.
//! - **Order.** The noticing worker, seeing a step newer than the slot's
//!   last taken, parses it into a [`Work`] and issues it into the slot's
//!   [`Cell`], announced by one word the other workers poll: its entries
//!   grouped by expert (tier record and expert), each an [`Expert`] with
//!   every row that names it. A request of no entries has no stage work: it
//!   is answered at once, every row zero, so the handshake it measures
//!   ([`measure`]) is the notice and the answer alone. A request's work is
//!   four stages, each of units cut into the pool's shares, which any
//!   worker claims once, its own share first ([`Share`]); a stage's units
//!   are claimed only once the stage before is finished, so a worker waits
//!   only on units of the request's own work in progress, never on a worker
//!   busy elsewhere (in a job's piece): its units are taken by the others.
//!   No claim, completion or barrier writes a line another worker writes,
//!   but where it steals a unit.
//! - **Stages.** Every share of a record is fixed by the record and the
//!   pool's size alone, never by the request around it ([`expands`],
//!   [`share`]), and a worker claims its own share first. Each stage
//!   computes an expert once over all its rows, in one projection call:
//!   each weight block decoded once for every group of up to four of its
//!   rows and dotted with each row's q8 activations (`projection::project_arithmetic` of several rows), each
//!   row's sums the ones it has alone.
//!   1. each expert's rows, a row a unit: copied from the slot's rows and
//!      quantized (`Work::gather`), so the noticing worker's issue reads no
//!      row;
//!   2. gate/up and the activation by units of a 16-row tile's storage
//!      group of columns, dealt contiguously, every share's count within one
//!      of every other's;
//!   3. each expert's products, a q8 block a unit: the tiles split between
//!      shares joined and activated, and the block of each of its rows
//!      quantized, once (`Work::join`);
//!   4. down by tiles of `H` rows, each share's tiles one call an expert,
//!      summing each of its rows' weighted outputs into the row's output.
//!
//!   The arithmetic is the standard one: activations quantized to q8
//!   per block (`quant::quantize`), i32 dots within a block, f32 across
//!   blocks and in the combine (`projection::project_arithmetic`). The
//!   products stay f32: they never leave the CPU.
//! - **Answer.** The output `[VERIFY_ROWS, H]` f32, every row written (zero
//!   where no entry names it), then the slot's answer, the step with
//!   release, or the step with the failure flag (SLOT_FAILED). The publish
//!   only moves a slot's answer forward ([`SiteLayer::publish`]), so a late answer
//!   never overwrites a later step's. Every taken request ends in a publish:
//!   a computed one by the worker that completes its last unit, a panicking
//!   unit recorded and still completed, one taken once the executor stopped
//!   issuing ([`Site::close`]) at once. A failed answer fails every row the
//!   request names ([`SiteFailure`]). A malformed request is the device's
//!   invariant violation, never a failure: reading it panics.
//! - **Admission.** The arithmetic is admitted at the bank's qualification
//!   against the reference, f32 math over the stored weights
//!   (`native::qualification::site`).

use super::exchange::ExchangeTable;
pub(in crate::experts) use super::exchange::FAILED;
use super::source::Records;
use magnitude_family_contracts::GluLimit;
use magnitude_kernels::constants;
use magnitude_kernels::cpu_library::cpu::lib::core::functions;
use magnitude_kernels::cpu_library::cpu::lib::projection::projection;
use seismic::cpu::quant::{self, Q8Block};
use seismic::cpu::{F32, Weights};
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

mod bank;
mod measure;

use super::{ComputeThreads, Machine, exchange, measure_site};
pub(crate) use bank::site_lines;
pub(crate) use measure::measure_cpu_site;

/// The tier's records of one routed tensor of a group, read by the site in
/// place: its records (`source::Records`), each `[rows, columns]` in the
/// bank's resident form, as the CPU's components read it on this host's
/// instruction tier (`Element::cpu_operand`, resolved once: a registry
/// lookup, which no request repeats).
#[derive(Clone)]
pub(crate) struct SiteRecords {
    pub records: Records,
    pub rows: usize,
    pub columns: usize,
    operand: seismic::cpu::WeightOperand,
}

// SAFETY: the records are host memory the bank holds for the load's life,
// which the site only reads while a request that names them is served.
unsafe impl Sync for SiteRecords {}

impl SiteRecords {
    /// `records`, each `[rows, columns]` of `element`; an error where this
    /// host's CPU reads no rows of it.
    pub(crate) fn new(records: Records, rows: usize, columns: usize, element: seismic::Element) -> Result<Self, String> {
        let tier = seismic::cpu::Tier::detected().ok_or("no CPU instruction tier serves the site")?;
        let operand = element.cpu_operand(tier).ok_or_else(|| format!("the CPU reads no {} rows", element.name()))?;
        Ok(Self { records, rows, columns, operand })
    }

    pub(crate) fn count(&self) -> usize {
        self.records.count() as usize
    }

    /// Record `record` as a weight operand of the CPU's components, read
    /// where it lies in the device's row layout, holding its slab view for
    /// the operand's life.
    fn weights(&self, record: usize) -> HeldWeights {
        assert!(record < self.count(), "a site entry names tier record {record} of {}", self.count());
        let hold = self.records.hold(record as u64);
        // SAFETY: the device named the record for this step, so its bytes are
        // whole and nothing writes or moves them until the step completes;
        // `hold` keeps the slab view alive for the life of `HeldWeights`.
        let weights = unsafe {
            Weights::from_tensor(
                hold.as_ptr(),
                &[self.rows as u64, self.columns as u64],
                &[self.columns as u64, 1],
                self.operand,
            )
        };
        HeldWeights { _hold: hold, weights }
    }

    /// Record `record`'s values, `[rows, columns]` in row order, as the
    /// CPU's components decode them: exactly the registry's decode of every
    /// element a row conversion writes (`seismic-native-cpu`'s
    /// `every_row_conversion_decodes_as_the_registry`). Read while nothing
    /// writes the record.
    pub(crate) fn values(&self, record: usize) -> Vec<f32> {
        let weights = self.weights(record);
        let mut values = vec![0.0; self.rows * self.columns];
        for (row, out) in values.chunks_exact_mut(self.columns).enumerate() {
            weights.decode_row(row, out);
        }
        values
    }
}

/// A layer's tier tensors as the site reads them: its group's records of
/// gate (where gated), up and down, which a resize of the tier re-addresses
/// between flights ([`Site::address`]).
#[derive(Clone)]
pub(crate) struct SiteTensors {
    pub gate: Option<SiteRecords>,
    pub up: SiteRecords,
    pub down: SiteRecords,
}

impl SiteTensors {
    fn address(&mut self, views: &[Records], displaced: &mut Vec<Records>) {
        let tensors = self.gate.iter_mut().chain([&mut self.up, &mut self.down]);
        for (tensor, records) in tensors.zip(views) {
            displaced.push(std::mem::replace(&mut tensor.records, records.clone()));
        }
    }
}

/// A layer's per-expert second-level scales (NVFP4 `.scale`) of its gate
/// and its up, each where its stack has one (none: the stack is unscaled,
/// as the device's `gate_scale`/`up_scale` of extent 0), indexed by an
/// entry's expert (`REQUEST_EXPERT`): one a layer's expert, which
/// [`SiteLayer::over`] admits.
#[derive(Clone)]
pub(crate) struct SiteScales {
    pub gate: Option<Arc<[f32]>>,
    pub up: Option<Arc<[f32]>>,
}

impl SiteScales {
    /// The gate's and the up's scale of `expert`, a layer's expert.
    fn of(&self, expert: usize) -> (f32, f32) {
        let scale = |scales: &Option<Arc<[f32]>>| scales.as_ref().map_or(1.0, |scales| scales[expert]);
        (scale(&self.gate), scale(&self.up))
    }
}

/// A layer's request and answer storage, host-shared: its site table of the
/// exchange (`site[FLIGHT, R]`, [`SiteShape::table`]), the rows
/// `rows[FLIGHT, VERIFY_ROWS, H]` f32 and the output
/// `output[FLIGHT, VERIFY_ROWS, H]` f32.
pub(crate) struct SiteStorage {
    pub table: seismic::Tensor,
    pub rows: seismic::Tensor,
    pub output: seismic::Tensor,
}

/// A layer's site extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SiteShape {
    /// Its site table's layout (`RegionWords::site`).
    pub table: magnitude_kernels::SlotTable,
    /// Rows a slot, `VERIFY_ROWS`.
    pub rows: usize,
    /// Choices a row, `K`: a request holds at most `rows * choices` entries.
    pub choices: usize,
    /// The layer's experts, `E`: what an entry's expert names.
    pub experts: usize,
    pub hidden: usize,
    pub features: usize,
    /// The expansion's function code (`functions::activate`).
    pub function: i32,
    /// The expansion's clamp (`functions::clamped_glu`'s `limit`).
    pub limit: Option<GluLimit>,
}

impl SiteShape {
    /// The extents of a site of form `form`.
    pub(crate) fn of(form: &crate::assessment::CpuForm) -> Result<Self, String> {
        let to = |value: u64| usize::try_from(value).map_err(|_| "site extents exceed the host".to_owned());
        Ok(Self {
            table: magnitude_kernels::SlotTable::site(form.flight, form.widest),
            rows: to(constants::VERIFY_ROWS)?,
            choices: to(form.choices)?,
            experts: to(form.experts)?,
            hidden: to(form.record.down.matrix[0])?,
            features: to(form.record.up.matrix[0])?,
            function: form.function,
            limit: form.limit,
        })
    }

    /// Its slots, `FLIGHT`.
    pub(crate) fn flight(&self) -> usize {
        self.table.slots() as usize
    }

    /// The slot step `step` requests in: `step` mod the flight.
    pub(crate) fn slot(&self, step: u64) -> usize {
        (step % self.table.slots()) as usize
    }
}

/// A gated expansion's product of its scaled F32 gate and up sums under
/// function code `function` (`functions::activate`), as published F32: the
/// `glu` of functions.seismic, [`functions::clamped_glu`] where the
/// expansion has a limit and [`functions::glu`] otherwise. The one owner of
/// that choice for the site and its references.
#[inline(always)]
pub(crate) fn gated(function: i32, limit: Option<GluLimit>) -> impl Fn(f32, f32) -> f32 + Copy {
    let limit = functions::glu_limit(GluLimit::word(limit));
    move |gate, up| match limit {
        None => functions::glu::<F32>(function, gate, up),
        Some(limit) => functions::clamped_glu::<F32>(function, limit, gate, up),
    }
}

/// F32 values of a host-shared tensor the site reads or writes in place.
struct Floats(seismic::Tensor);

impl Floats {
    fn new(tensor: seismic::Tensor, len: usize) -> Result<Self, String> {
        tensor.shared_address().ok_or("site storage the host does not address")?;
        if tensor.byte_len() < (len * 4) as u64 {
            return Err(format!("site storage of {} bytes for {len} values", tensor.byte_len()));
        }
        Ok(Self(tensor))
    }

    fn as_ptr(&self) -> *mut f32 {
        self.0.shared_address().expect("host-addressed since it was made").as_ptr().cast()
    }

    /// `len` values from `index`, read out to the host once nothing writes
    /// them: every NaN is the registry's one NaN, as [`seismic::Tensor::read_to_host`] gives.
    unsafe fn read_out(&self, index: usize, len: usize) -> Vec<f32> {
        let mut values = unsafe { std::slice::from_raw_parts(self.as_ptr().add(index), len) }.to_vec();
        // SAFETY: `f32` values are initialized 4-byte elements.
        let bytes = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), values.len() * 4) };
        seismic::Element::f32().canonicalize_in_place(bytes);
        values
    }
}

/// One streamed layer as the site serves it.
pub(crate) struct SiteLayer {
    /// Its site table of the exchange: each slot's request and answer.
    slots: ExchangeTable,
    rows: Floats,
    output: Floats,
    /// Its streamed layer (the bank's index), by which its gate knows it.
    layer: usize,
    /// Its group (the bank's index), whose tier records it reads.
    group: usize,
    /// Its tier tensors, re-addressed only between flights.
    tensors: RwLock<SiteTensors>,
    scales: SiteScales,
    pub shape: SiteShape,
}

impl SiteLayer {
    /// Streamed layer `layer` of the site over group `group`'s `tensors`,
    /// `scales` (each one a layer's expert) and `storage`, its table beating
    /// with `pulse`.
    pub(crate) fn over(
        storage: SiteStorage,
        layer: usize,
        group: usize,
        tensors: SiteTensors,
        scales: SiteScales,
        shape: SiteShape,
        pulse: &crate::experts::exchange::Pulse,
    ) -> Result<Self, String> {
        let SiteStorage { table, rows, output } = storage;
        for (stack, scales) in [("gate", &scales.gate), ("up", &scales.up)] {
            if let Some(scales) = scales.as_ref().filter(|scales| scales.len() != shape.experts) {
                return Err(format!("{} {stack} scales of a layer of {} experts", scales.len(), shape.experts));
            }
        }
        let floats = shape.flight() * shape.rows * shape.hidden;
        Ok(Self {
            slots: ExchangeTable::new(table, shape.table, pulse)?,
            rows: Floats::new(rows, floats)?,
            output: Floats::new(output, floats)?,
            layer,
            group,
            tensors: RwLock::new(tensors),
            scales,
            shape,
        })
    }

    /// A layer over `tensors` and `scales` with storage of its own through
    /// `charges`, host-shared: the layer a measurement or an
    /// admission serves ([`measure`], `native::qualification::site`), of an
    /// executor of its own whose requests the host writes (no park answers
    /// them), as layer 0 of group 0.
    pub(crate) fn owned(charges: &seismic::Charges, tensors: SiteTensors, scales: SiteScales, shape: SiteShape) -> Result<Self, String> {
        let shared = |element: seismic::Element, count: usize| {
            seismic::Tensor::shared(charges, element, &[count as u64]).map_err(|error| format!("site storage: {error}"))
        };
        let storage = SiteStorage {
            table: shared(seismic::Element::i32(), shape.table.words() as usize)?,
            rows: shared(seismic::Element::f32(), shape.flight() * shape.rows * shape.hidden)?,
            output: shared(seismic::Element::f32(), shape.flight() * shape.rows * shape.hidden)?,
        };
        // Its table beats with a pulse of its own: no load's steps run here.
        Self::over(storage, 0, 0, tensors, scales, shape, &crate::experts::exchange::Pulse::new())
    }

    /// Slot `slot`'s request step.
    fn sequence(&self, slot: usize) -> u64 {
        self.slots.held(slot as u64, 0).0
    }

    /// Write `step`'s request in its slot as the device does: each entry
    /// `(row, tier record, weight, expert)` and the rows, then the count,
    /// then the request (release). For a request the host itself issues: a
    /// measurement's, a qualification's, a test's.
    pub(crate) fn request(&self, step: u64, entries: &[(usize, usize, f32, usize)], rows: &[f32]) {
        self.payload(step, entries, rows);
        self.slots.request(step, 0);
    }

    /// `step`'s request's payload in its slot, all but its request word
    /// ([`Self::request`]).
    fn payload(&self, step: u64, entries: &[(usize, usize, f32, usize)], rows: &[f32]) {
        let (shape, slot) = (self.shape, self.shape.slot(step));
        assert!(entries.len() <= shape.rows * shape.choices && rows.len() == shape.rows * shape.hidden, "a request of the layer's extents");
        for (index, &(row, record, weight, expert)) in entries.iter().enumerate() {
            let at = constants::SITE_ENTRY(index as u64);
            let words = [
                (constants::SITE_ROW, row as i32),
                (constants::SITE_TIER_RECORD, record as i32),
                (constants::SITE_WEIGHT_BITS, weight.to_bits() as i32),
                (constants::SITE_EXPERT, expert as i32),
            ];
            for (word, value) in words {
                self.slots.store(step, 0, at + word, value);
            }
        }
        // SAFETY: no request of this slot is being served (the caller issues
        // it after the slot's last answer), so nothing reads its rows.
        unsafe { std::ptr::copy_nonoverlapping(rows.as_ptr(), self.rows.as_ptr().add(slot * shape.rows * shape.hidden), rows.len()) };
        self.slots.store(step, 0, constants::SITE_COUNT, entries.len() as i32);
    }

    /// Its streamed layer (the bank's index).
    pub(super) fn layer(&self) -> usize {
        self.layer
    }

    /// Slot `slot`'s output, once answered.
    pub(crate) fn output(&self, slot: usize) -> Vec<f32> {
        let len = self.shape.rows * self.shape.hidden;
        // SAFETY: the slot was answered, so no worker writes it.
        unsafe { self.output.read_out(slot * len, len) }
    }

    /// Slot `slot`'s output, every row zero: a request of no entries' answer.
    fn clear(&self, slot: usize) {
        let len = self.shape.rows * self.shape.hidden;
        // SAFETY: the slot's request is taken and not issued, so no worker
        // writes its output, and the device reads it only once answered.
        unsafe { std::ptr::write_bytes(self.output.as_ptr().add(slot * len), 0, len) };
    }

    /// Slot `slot`'s answer.
    pub(crate) fn ready(&self, slot: usize) -> u64 {
        self.slots.held(slot as u64, 0).1
    }

    /// Its site table: what the device's wait for its answers reads, by the
    /// exchange's one rule (`watcher::parks`).
    pub(crate) fn slots(&self) -> &ExchangeTable {
        &self.slots
    }

    /// Publish `step`'s answer in its slot (`failed`: with the failure
    /// flag), after every output write: the answer only moves forward
    /// (`ExchangeTable::answer`).
    fn publish(&self, step: u64, failed: bool) {
        self.slots.answer(step, 0, failed);
    }

    /// `step`'s request's entries as the device wrote them in its slot, read
    /// after its sequence was acquired. The device writes only well-formed
    /// requests: an entry count, row, tier record or expert beyond the
    /// layer's extents is an invariant violation, and panics.
    fn entries(&self, step: u64) -> impl Iterator<Item = Requested> + '_ {
        let shape = self.shape;
        let count = self.slots.load(step, 0, constants::SITE_COUNT);
        let count = usize::try_from(count)
            .ok()
            .filter(|&count| count <= shape.rows * shape.choices)
            .unwrap_or_else(|| panic!("a site request of {count} entries"));
        (0..count).map(move |entry| {
            let at = constants::SITE_ENTRY(entry as u64);
            let field = |word| self.slots.load(step, 0, at + word);
            let (row, record, expert) = (field(constants::SITE_ROW), field(constants::SITE_TIER_RECORD), field(constants::SITE_EXPERT));
            Requested {
                row: usize::try_from(row).ok().filter(|&row| row < shape.rows).unwrap_or_else(|| panic!("a site entry of row {row}")),
                record: usize::try_from(record).unwrap_or_else(|_| panic!("a site entry of tier record {record}")),
                weight: f32::from_bits(field(constants::SITE_WEIGHT_BITS) as u32),
                expert: usize::try_from(expert)
                    .ok()
                    .filter(|&expert| expert < shape.experts)
                    .unwrap_or_else(|| panic!("a site entry of expert {expert} of {}", shape.experts)),
            }
        })
    }

    /// `step`'s request and its answer, where its slot holds them answered
    /// and not failed (no later step has reused the slot): what a test
    /// checks against [`reference`].
    #[cfg(test)]
    pub(crate) fn answered(&self, step: u64) -> Option<Answered> {
        let slot = self.shape.slot(step);
        if self.sequence(slot) != step || self.ready(slot) != step {
            return None;
        }
        let entries = self
            .entries(step)
            .map(|Requested { row, weight, expert, .. }| Entry { row, weight, expert, scales: self.scales.of(expert) })
            .collect();
        let len = self.shape.rows * self.shape.hidden;
        // SAFETY: the slot was answered, and the device writes its rows only
        // for a later step's request.
        let rows = unsafe { self.rows.read_out(slot * len, len) };
        Some(Answered { entries, rows, output: self.output(slot) })
    }

    /// Populate `work` in place with slot `slot`'s request for `step`, read
    /// after its sequence was acquired ([`Self::entries`]), its entries
    /// grouped by (tier record, expert) into experts (in the order the request
    /// first names them), reusing all buffer capacities held by `work`.
    fn prepare(&self, slot: usize, step: u64, work: &mut Work) {
        let shape = self.shape;
        work.slot = slot;
        work.step = step;
        // SAFETY: within the rows (`Floats::new`); the device wrote the
        // slot's rows before its sequence, which was acquired, and writes
        // them no more while this step runs.
        work.input = unsafe { self.rows.as_ptr().add(slot * shape.rows * shape.hidden) };
        work.experts.clear();
        work.entry_rows.clear();
        work.requested.clear();
        work.requested.extend(self.entries(step));
        if work.requested.is_empty() {
            return;
        }
        let tensors = self.tensors.read().expect("site tensors");
        for &Requested { record, expert, .. } in &work.requested {
            match work.experts.iter_mut().find(|named| (named.record, named.expert) == (record, expert)) {
                Some(named) => named.rows += 1,
                None => work.experts.push(Expert {
                    record,
                    expert,
                    scales: self.scales.of(expert),
                    gate: tensors.gate.as_ref().map(|records| records.weights(record)),
                    up: tensors.up.weights(record),
                    down: tensors.down.weights(record),
                    at: 0,
                    rows: 1,
                }),
            }
        }
        drop(tensors);
        let mut named = 0;
        for expert in &mut work.experts {
            expert.at = named;
            named += expert.rows;
            expert.rows = 0;
        }
        work.entry_rows.resize(named, (0, 0.0));
        for &Requested { row, record, weight, expert } in &work.requested {
            let entry = work
                .experts
                .iter_mut()
                .find(|named| (named.record, named.expert) == (record, expert))
                .expect("expert added in first pass");
            work.entry_rows[entry.at + entry.rows] = (row, weight);
            entry.rows += 1;
        }
        work.ensure_buffers();
    }
}

/// One entry of a site request as the device wrote it: a row's choice of
/// the expert its tier record holds, at its routing weight.
#[derive(Clone, Copy)]
struct Requested {
    row: usize,
    record: usize,
    weight: f32,
    expert: usize,
}

/// What a site layer answered of one step ([`SiteLayer::answered`]): its
/// request's entries, its rows `[rows, H]` and its output `[rows, H]`.
#[cfg(test)]
pub(crate) struct Answered {
    pub entries: Vec<Entry>,
    pub rows: Vec<f32>,
    pub output: Vec<f32>,
}

/// One entry of a request: a row's choice of the layer's expert `expert`
/// at its routing weight, with the expert's gate and up scales.
#[cfg(test)]
pub(crate) struct Entry {
    pub row: usize,
    pub weight: f32,
    pub expert: usize,
    pub scales: (f32, f32),
}

/// One expert's stored values, as the reference reads them: its gate's
/// (where gated) and its up's rows `[F, H]`, its down's `[H, F]`, and its
/// gate's and up's scales.
#[cfg(test)]
pub(crate) struct ExpertValues<'v> {
    pub gate: Option<&'v [f32]>,
    pub up: &'v [f32],
    pub down: &'v [f32],
    pub scales: (f32, f32),
}

/// Half a q8 step, in steps: a value's rounding to its code, and the f32
/// inverse of the step it is multiplied by (at most 127 codes, each off by
/// under 2^-22 of one).
#[cfg(test)]
const HALF_STEP: f32 = 0.5 + 1.0 / 32768.0;

/// The bound of f32 rounding over a computation of `terms` terms, relative
/// to the sum of their magnitudes: every accumulation order, every scale
/// and conversion, and the activation's evaluation, with room to spare.
#[cfg(test)]
fn rounding(terms: usize) -> f32 {
    (terms as f32 + 16.0) * f32::EPSILON
}

/// Each of `values`' departure bound under the site's q8 quantization
/// (`quant::quantize`): half the step of its sub-block of `quant::Q8_GROUP`
/// values, the sub-block's largest magnitude over 127. `magnitudes` bound the
/// quantized values' magnitudes (the values themselves, or what may depart
/// from them before quantization).
#[cfg(test)]
fn q8_departures(magnitudes: &[f32]) -> Vec<f32> {
    magnitudes
        .chunks(quant::Q8_GROUP)
        .flat_map(|group| {
            let step = group.iter().fold(0.0f32, |most, value| most.max(value.abs())) / 127.0;
            std::iter::repeat_n(step * HALF_STEP, group.len())
        })
        .collect()
}

/// A bound of `function`'s slope over `[z - radius, z + radius]`.
#[cfg(test)]
fn slope(function: i32, z: f32, radius: f32) -> f32 {
    match function {
        // SiLU's slope is at most 1.0998 anywhere, GELU-tanh's 1.1289.
        functions::SILU => 1.1,
        functions::GELU_TANH => 1.13,
        functions::RELU_SQUARED => 2.0 * (z.abs() + radius),
        other => panic!("the site's activation {other} has no slope bound"),
    }
}

/// The site's answer to one row `x` (`[H]`) of `experts`, each at its
/// routing weight, as the reference computes it (f64 dots of the stored
/// values, the activation in f32, as admission's reference does), and, for
/// each channel, the bound of the site's departure from it that its
/// arithmetic admits, the one owner of that bound:
///
/// - the row's activations are q8 per block (`q8_departures`): a gate or up
///   sum departs by at most `Σ |w_i| u_i` (`u_i` the activation's half
///   step), plus f32 rounding;
/// - its products depart through the activation by at most
///   `L e_g (|up| + e_u) + |act(gate)| e_u` (`L` the activation's slope,
///   `e` the sums' departures), or `L e_u` up-only, with `gate` and `up`
///   clamped at `limit` (`functions::clamped_glu`), which moves neither departure
///   further;
/// - the products are q8 per block in turn, their blocks' steps from the
///   products' magnitudes plus their departures, and the down sum departs
///   by at most `Σ |d_f| (e_f + u_f)`, plus f32 rounding;
/// - the row's output is each expert's weighted down sum.
///
/// An expert of another layer's values departs by its whole size, far past
/// the bound.
#[cfg(test)]
pub(crate) fn reference(function: i32, limit: Option<GluLimit>, x: &[f32], experts: &[(f32, ExpertValues<'_>)]) -> (Vec<f32>, Vec<f32>) {
    let glu = gated(function, limit);
    let clamp = functions::glu_limit(GluLimit::word(limit)).unwrap_or(f32::INFINITY);
    let hidden = x.len();
    let activations = q8_departures(x);
    // A dot over `x`: its value, and its departure under the site's q8 rows.
    let sum = |weights: &[f32], scale: f32| {
        let value = weights.iter().zip(x).map(|(w, x)| f64::from(*w) * f64::from(*x)).sum::<f64>() as f32 * scale;
        let quantized = weights.iter().zip(&activations).map(|(w, u)| w.abs() * u).sum::<f32>();
        let magnitude = weights.iter().zip(x).zip(&activations).map(|((w, x), u)| w.abs() * (x.abs() + u)).sum::<f32>();
        (value, scale.abs() * (quantized + rounding(hidden) * magnitude))
    };
    let (mut value, mut bound) = (vec![0.0f32; hidden], vec![0.0f32; hidden]);
    for (weight, expert) in experts {
        let features = expert.up.len() / hidden;
        let (gate_scale, up_scale) = expert.scales;
        let (products, departures): (Vec<f32>, Vec<f32>) = (0..features)
            .map(|feature| {
                let at = feature * hidden..(feature + 1) * hidden;
                let (up, e_up) = sum(&expert.up[at.clone()], up_scale);
                match expert.gate {
                    Some(gate) => {
                        let (gate, e_gate) = sum(&gate[at], gate_scale);
                        let product = glu(gate, up);
                        let (gate, up) = (gate.min(clamp), up.clamp(-clamp, clamp));
                        let activated = functions::activate(function, gate);
                        let moved = slope(function, gate, e_gate) * e_gate;
                        let departure = moved * (up.abs() + e_up) + activated.abs() * e_up;
                        (product, departure + rounding(0) * (activated.abs() + moved) * (up.abs() + e_up))
                    }
                    None => {
                        let product = functions::activate(function, up);
                        let moved = slope(function, up, e_up) * e_up;
                        (product, moved + rounding(0) * (product.abs() + moved))
                    }
                }
            })
            .unzip();
        let magnitudes = products.iter().zip(&departures).map(|(p, e)| p.abs() + e).collect::<Vec<_>>();
        let quantized = q8_departures(&magnitudes);
        for channel in 0..hidden {
            let down = &expert.down[channel * features..(channel + 1) * features];
            let y = down.iter().zip(&products).map(|(d, p)| f64::from(*d) * f64::from(*p)).sum::<f64>() as f32;
            let departs = down.iter().zip(departures.iter().zip(&quantized)).map(|(d, (e, u))| d.abs() * (e + u)).sum::<f32>();
            let magnitude = down.iter().zip(magnitudes.iter().zip(&quantized)).map(|(d, (m, u))| d.abs() * (m + u)).sum::<f32>();
            value[channel] += weight * y;
            bound[channel] += weight.abs() * (departs + rounding(features) * magnitude + rounding(experts.len()) * (y.abs() + departs));
        }
    }
    (value, bound)
}

/// One expert of a request: the layer's expert a tier record holds, which
/// the CPU computes whole, once over every row that names it.
/// A weight operand paired with its [`HeldRecord`] so the slab view stays
/// pinned while the CPU site reads it and drops as soon as the work ends.
struct HeldWeights {
    _hold: crate::experts::source::HeldRecord,
    weights: Weights<'static>,
}

impl std::ops::Deref for HeldWeights {
    type Target = Weights<'static>;
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

struct Expert {
    /// Its tier record, which fixes each worker's share of it.
    record: usize,
    /// The layer's expert it is, whose scales it applies.
    expert: usize,
    /// Its gate's and up's second-level scales.
    scales: (f32, f32),
    gate: Option<HeldWeights>,
    up: HeldWeights,
    down: HeldWeights,
    /// Its first row among the request's experts' rows, expert after expert:
    /// where its rows' entries, activations and products, and their q8 blocks,
    /// lie in the request's ([`Work`]).
    at: usize,
    /// How many rows of the request name it.
    rows: usize,
}

/// A request's stages, in order (module docs): gather, expand, join, then
/// project.
const GATHER: usize = 0;
const EXPAND: usize = 1;
const JOIN: usize = 2;
const STAGES: usize = 4;

/// The low bits of a cell's `progress` that count its stages finished.
const STAGE_BITS: u32 = 3;

const _: () = assert!(STAGES < 1 << STAGE_BITS, "a request's stages and its finish fit their bits");

/// Issue `issue` with `stages` stages finished, as a cell's `progress`.
fn progress(issue: u64, stages: usize) -> u64 {
    issue << STAGE_BITS | stages as u64
}

/// A cell's `progress` as its issue and its stages finished ([`progress`]).
fn progressed(word: u64) -> (u64, usize) {
    (word >> STAGE_BITS, (word & ((1 << STAGE_BITS) - 1)) as usize)
}

/// The low bits of a share's word that count its units ([`Share`]); the
/// rest are the issue the count is of.
const COUNT_BITS: u32 = 20;

/// `count` units of issue `issue`, as a share's word holds them.
fn tagged(issue: u64, count: usize) -> u64 {
    issue << COUNT_BITS | count as u64
}

/// One worker's share of each stage of a cell's request: the units of it
/// claimed and completed, each word tagged with the issue it counts
/// (`tagged`). A word of an earlier issue counts none of this one's, so a
/// request's issue resets nothing; a word of a later one means the request
/// is answered. Its own pair of lines (an x86 core prefetches a line's
/// pair), written by its worker alone while no other steals from it: a
/// claim or a completion moves no line another worker writes.
#[repr(align(128))]
struct Share {
    claimed: [AtomicU64; STAGES],
    done: [AtomicU64; STAGES],
}

impl Share {
    fn new() -> Self {
        Self { claimed: std::array::from_fn(|_| AtomicU64::new(0)), done: std::array::from_fn(|_| AtomicU64::new(0)) }
    }

    /// The units of issue `issue` share word `word` counts: none counted of
    /// an earlier issue's word, no count at all (the request is answered) of
    /// a later one's.
    fn count(word: u64, issue: u64) -> Option<usize> {
        match (word >> COUNT_BITS).cmp(&issue) {
            std::cmp::Ordering::Less => Some(0),
            std::cmp::Ordering::Equal => Some((word & ((1 << COUNT_BITS) - 1)) as usize),
            std::cmp::Ordering::Greater => None,
        }
    }

    /// Claim the next of its `len` units of `stage` of issue `issue`: its
    /// index within the share, none where every one is claimed.
    fn claim(&self, issue: u64, stage: usize, len: usize) -> Option<usize> {
        let word = &self.claimed[stage];
        let mut held = word.load(Ordering::Relaxed);
        loop {
            let next = Self::count(held, issue).filter(|&next| next < len)?;
            match word.compare_exchange_weak(held, tagged(issue, next + 1), Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return Some(next),
                Err(now) => held = now,
            }
        }
    }

    /// Complete a claimed unit of `stage` of issue `issue`, releasing its
    /// writes.
    fn complete(&self, issue: u64, stage: usize) {
        let word = &self.done[stage];
        let mut held = word.load(Ordering::Relaxed);
        loop {
            let done = Self::count(held, issue).expect("a claimed unit's request is not answered");
            match word.compare_exchange_weak(held, tagged(issue, done + 1), Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return,
                Err(now) => held = now,
            }
        }
    }

    /// Whether its `len` units of `stage` of issue `issue` are complete
    /// (acquiring their writes); a word of a later issue is complete.
    fn finished(&self, issue: u64, stage: usize, len: usize) -> bool {
        len == 0 || Self::count(self.done[stage].load(Ordering::Acquire), issue).is_none_or(|done| done == len)
    }
}

/// The request of one slot of one site layer the pool serves: the cell's
/// state, which the noticing worker ([`Site::take`]) writes as it issues a
/// request and every worker reads. The request's stages are cut into the
/// pool's shares ([`share`]), each claimed by its worker first ([`Share`]);
/// `progress` counts the stages finished, which the worker that first sees
/// a stage's every share complete advances, and the one that advances it
/// past the last publishes.
struct Cell {
    /// Its site layer.
    layer: usize,
    /// The latest step taken from the slot (by the noticing worker alone).
    taken: AtomicU64,
    /// The latest step taken whose request is out of the cell, as its
    /// answer is written: the cell takes the slot's next request only once
    /// this is `taken`. The device requests a slot again only after its answer,
    /// but a wait that ended on its pulse (`exchange::Pulse`) let its step
    /// run on, and a later step may request the slot while the pool still
    /// serves the earlier one; that request waits here for the cell.
    answered: AtomicU64,
    /// The issue its request is and its stages finished ([`progress`]; all
    /// of them: answered or being answered), as a sequence lock over `units`
    /// and `work`: zero while they are rewritten.
    progress: AtomicU64,
    /// Each stage's units of the request.
    units: [AtomicUsize; STAGES],
    /// The request's step, for the watchdog.
    step: AtomicU64,
    /// The request and its reusable buffers, which only a worker holding a
    /// claimed unit of it (or its publisher) reads, and only the noticing
    /// worker (after `answered == taken`) or publisher (before `answered`)
    /// mutates.
    work: UnsafeCell<Work>,
    shares: Box<[Share]>,
    /// When its current request reached each point of its path
    /// (`NOTICED`..`MARKS`), in nanoseconds since the site began: each
    /// written once per issue, before the next point's.
    marks: [AtomicU64; MARKS],
    /// The experts its latest request served, their rows (each row that
    /// names one) and its path's time (ns, taken to answered): none until
    /// it is answered whole, its answer's store after these.
    served: AtomicU64,
    rows: AtomicU64,
    took: AtomicU64,
}

// SAFETY: `work` is mutated only by the noticing worker as it prepares a
// request into a cell whose last request was answered, or by its publisher
// before storing `answered`, and read only under a claimed unit (`Cell::work`).
unsafe impl Sync for Cell {}

impl Cell {
    fn new(layer: usize, shape: SiteShape, workers: usize) -> Self {
        Self {
            layer,
            taken: AtomicU64::new(0),
            answered: AtomicU64::new(0),
            progress: AtomicU64::new(0),
            units: std::array::from_fn(|_| AtomicUsize::new(0)),
            step: AtomicU64::new(0),
            work: UnsafeCell::new(Work::empty(shape, workers)),
            shares: (0..workers).map(|_| Share::new()).collect(),
            marks: std::array::from_fn(|_| AtomicU64::new(0)),
            served: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            took: AtomicU64::new(0),
        }
    }

    /// Its request, issue `issue`.
    ///
    /// # Safety
    /// The caller holds a claimed unit of issue `issue` it has not
    /// completed, or is its issuer/publisher while no unit is in flight.
    unsafe fn work(&self) -> &Work {
        // SAFETY: the caller's contract.
        unsafe { &*self.work.get() }
    }

    /// The stage of issue `issue` not yet finished, and its units, read
    /// consistently (`progress` as a sequence lock); none where the cell
    /// holds another issue or every stage is finished.
    fn stage(&self, issue: u64) -> Option<(usize, usize)> {
        let (held, stage) = progressed(self.progress.load(Ordering::Acquire));
        if held != issue || stage == STAGES {
            return None;
        }
        let units = self.units[stage].load(Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Acquire);
        (progressed(self.progress.load(Ordering::Relaxed)).0 == issue).then_some((stage, units))
    }
}

/// A request's path at the site, in order: the noticing worker took it
/// (`NOTICED`), issued it to the pool (`ISSUED`), and each stage finished
/// (`STAGE_ENDS + stage`), the last stage's end being its answer's
/// publication's start.
const NOTICED: usize = 0;
const ISSUED: usize = 1;
const STAGE_ENDS: usize = 2;
const MARKS: usize = STAGE_ENDS + STAGES;

/// What the site's answered requests took, summed since the load
/// ([`Site::served`]): the requests, and the time of each term of their
/// path: issue (taken to issued), each stage
/// (gather, expand, join, project: from the end of the term before), and
/// publish (the last stage's end to the answer written). What the device
/// waited beyond these is the request's notice and the answer's return.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SiteServed {
    pub requests: u64,
    pub experts: u64,
    pub terms: [std::time::Duration; SITE_TERMS],
}

/// The terms of [`SiteServed`]: issue, gather, expand, join, project,
/// publish.
pub const SITE_TERMS: usize = MARKS;

/// Each term's name, as [`SiteServed::terms`] orders them.
pub const SITE_TERM_NAMES: [&str; SITE_TERMS] = ["issue", "gather", "expand", "join", "project", "publish"];

impl SiteServed {
    /// What was served since `before`, an earlier reading of the same site.
    pub fn since(&self, before: &Self) -> Self {
        Self {
            requests: self.requests - before.requests,
            experts: self.experts - before.experts,
            terms: std::array::from_fn(|term| self.terms[term] - before.terms[term]),
        }
    }
}

/// The site's running sums of [`SiteServed`], in nanoseconds.
struct ServedSums {
    requests: AtomicU64,
    experts: AtomicU64,
    terms: [AtomicU64; SITE_TERMS],
}

/// What a worker did for the site's requests ([`Site::serve`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WorkerTurn {
    /// It ran a unit, or saw a stage finished.
    Ran,
    /// A request it may serve has units in progress elsewhere and none to
    /// claim.
    Waiting,
    /// No request is issued with units to claim or in progress.
    Idle,
}

/// A request served by the pool, its units claimed by any worker (each
/// worker its own shares first), stage by stage. All buffer capacities are
/// retained on the cell across requests so steady-state decode allocates
/// nothing on the hot path.
struct Work {
    slot: usize,
    step: u64,
    /// The slot's rows `[VERIFY_ROWS, H]`, which the device wrote.
    input: *const f32,
    /// Scratch buffer for reading a request's entries in [`SiteLayer::prepare`].
    requested: Vec<Requested>,
    experts: Vec<Expert>,
    /// Every expert's `(row, routing_weight)` entries laid out contiguously
    /// (`Expert::at..Expert::at + Expert::rows`).
    entry_rows: Vec<(usize, f32)>,
    /// Every expert's rows' activations `[rows, H]`, expert after expert
    /// (`Expert::at`), each gathered from the input by its gather unit, and
    /// their q8 blocks (`[rows, blocks(H)]`), as a projection of several
    /// rows reads them (`projection::project_arithmetic`).
    x: Vec<UnsafeCell<f32>>,
    x8: Vec<UnsafeCell<Q8Block>>,
    /// Every expert's rows' products `[rows, F]`, expert after expert
    /// (`Expert::at`): each whole tile's written by its expand unit, each
    /// split tile's by the join unit of its block.
    products: Vec<UnsafeCell<f32>>,
    /// Each expert's tiles an expand share computed part of (`[experts,
    /// workers, 2]`, its first and its last): their partial gate and up
    /// sums, which the join joins.
    partials: Vec<UnsafeCell<Partial>>,
    /// Every expert's rows' products quantized (`[rows, blocks(F)]`), each
    /// block of an expert's rows by its join unit.
    quantized: Vec<UnsafeCell<Q8Block>>,
    /// Reusable per-worker scratch buffers for the `expand` and `project`
    /// stages (each worker index is claimed by at most one thread per stage,
    /// and stages never overlap).
    scratch: Box<[UnsafeCell<WorkerScratch>]>,
    workers: usize,
    shape: SiteShape,
}

/// Reusable per-worker scratch buffers for `expand` and `project`.
#[derive(Default)]
struct WorkerScratch {
    /// Expand: whole-tile gate sums; Project: output channel sums.
    a: Vec<f32>,
    /// Expand: whole-tile up sums; Project: expert down dots.
    b: Vec<f32>,
    /// Expand partial tile: packed input activations over the column window.
    values: Vec<f32>,
    /// Expand partial tile: packed q8 blocks over the column window.
    q8: Vec<Q8Block>,
}

/// A tile's partial sums over some of its columns' storage groups, for
/// each of an expert's rows (activation-major, `[rows, tile rows]`): of the
/// gate (zero where ungated) and the up of each of the tile's rows.
#[derive(Default)]
struct Partial {
    /// The tile, or none.
    tile: Option<usize>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

/// Rows of a unit: an `mma16` tile's (`Weights::row_block`), the largest
/// component block.
const TILE: usize = projection::MAX_BLOCK;
const TILES_PER_BLOCK: usize = quant::Q8_BLOCK / TILE;

const _: () = assert!(quant::Q8_BLOCK % TILE == 0, "a q8 block of products is whole tiles");

// SAFETY: each unit writes cells no other unit writes (a gather unit its
// row's activations, an expand share its whole tiles, its own scratch and
// its own partials, a join unit its block's split tiles and quantized block,
// a project share its own scratch and its channels of the output), and a
// stage reads the cells of the stages before it only once they are finished,
// which their completions release.
unsafe impl Sync for Work {}
unsafe impl Send for Work {}

/// `cells` as the values they hold.
///
/// # Safety
/// No write to them is in flight.
unsafe fn values<T>(cells: &[UnsafeCell<T>]) -> &[T] {
    // SAFETY: `UnsafeCell<T>` has `T`'s layout; the caller's contract.
    unsafe { std::slice::from_raw_parts(UnsafeCell::raw_get(cells.as_ptr()), cells.len()) }
}

/// `cells` as values the caller alone writes.
///
/// # Safety
/// Nothing else reads or writes them while the slice lives.
#[allow(clippy::mut_from_ref)]
unsafe fn writable<T>(cells: &[UnsafeCell<T>]) -> &mut [T] {
    // SAFETY: `UnsafeCell<T>` has `T`'s layout; the caller's contract.
    unsafe { std::slice::from_raw_parts_mut(UnsafeCell::raw_get(cells.as_ptr()), cells.len()) }
}

impl Work {
    fn empty(shape: SiteShape, workers: usize) -> Self {
        Self {
            slot: 0,
            step: 0,
            input: std::ptr::null(),
            requested: Vec::new(),
            experts: Vec::new(),
            entry_rows: Vec::new(),
            x: Vec::new(),
            x8: Vec::new(),
            products: Vec::new(),
            partials: Vec::new(),
            quantized: Vec::new(),
            scratch: (0..workers).map(|_| UnsafeCell::default()).collect(),
            workers,
            shape,
        }
    }

    /// Grow reusable buffers to fit the current request and reset active
    /// partial markers without freeing any buffer capacities.
    fn ensure_buffers(&mut self) {
        let rows = self.entry_rows.len();
        let shape = self.shape;
        let x_len = rows * shape.hidden;
        if self.x.len() < x_len {
            self.x.resize_with(x_len, || UnsafeCell::new(0.0));
        }
        let x8_len = rows * quant::blocks(shape.hidden);
        if self.x8.len() < x8_len {
            self.x8.resize_with(x8_len, || UnsafeCell::new(Q8Block::ZERO));
        }
        let products_len = rows * shape.features;
        if self.products.len() < products_len {
            self.products.resize_with(products_len, || UnsafeCell::new(0.0));
        }
        let partials_len = self.experts.len() * self.workers * 2;
        if self.partials.len() < partials_len {
            self.partials.resize_with(partials_len, UnsafeCell::default);
        }
        for partial in &mut self.partials[..partials_len] {
            partial.get_mut().tile = None;
        }
        let quantized_len = rows * quant::blocks(shape.features);
        if self.quantized.len() < quantized_len {
            self.quantized.resize_with(quantized_len, || UnsafeCell::new(Q8Block::ZERO));
        }
    }

    /// Drop pinned [`HeldWeights`] handles while retaining all buffer
    /// capacities for the next request.
    fn release_weights(&mut self) {
        self.experts.clear();
    }

    /// Each stage's units: an expert's row of gather, a share a worker of
    /// expand and project, a q8 block of an expert's products of join.
    fn units(&self) -> [usize; STAGES] {
        [self.entry_rows.len(), self.workers, self.experts.len() * quant::blocks(self.shape.features), self.workers]
    }

    /// Run unit `unit` of stage `stage`, claimed, writing the output at
    /// `output`.
    ///
    /// # Safety
    /// Every stage before `stage` is finished.
    unsafe fn run(&self, stage: usize, unit: usize, output: *mut f32) {
        let blocks = quant::blocks(self.shape.features);
        match stage {
            GATHER => self.gather(unit),
            // SAFETY: the caller's contract.
            EXPAND => unsafe { self.expand(unit) },
            // SAFETY: the caller's contract.
            JOIN => unsafe { self.join(unit / blocks, unit % blocks) },
            // SAFETY: the caller's contract.
            _ => unsafe { self.project(unit, output) },
        }
    }

    /// Stage one, the request's row `named` of its experts' rows: the input
    /// row it is copied and quantized (each row on its own).
    fn gather(&self, named: usize) {
        let hidden = self.shape.hidden;
        let blocks = quant::blocks(hidden);
        let (row, _) = self.entry_rows[named];
        // SAFETY: the input's rows are the device's, whole for the step.
        let input = unsafe { std::slice::from_raw_parts(self.input.add(row * hidden), hidden) };
        // SAFETY: this row's activations are this unit's alone.
        let (x, x8) = unsafe { (writable(&self.x[named * hidden..][..hidden]), writable(&self.x8[named * blocks..][..blocks])) };
        x.copy_from_slice(input);
        quant::quantize(x, x8);
    }

    /// Stage two, share `worker`'s units of every expert ([`expands`]): its
    /// whole tiles' gate/up of all the expert's rows and their activation in
    /// one call each, f32, and the partial sums of a tile it computes part
    /// of.
    ///
    /// # Safety
    /// The gather stage is finished.
    unsafe fn expand(&self, worker: usize) {
        let (features, hidden) = (self.shape.features, self.shape.hidden);
        let blocks = quant::blocks(hidden);
        // SAFETY: unit `worker` of EXPAND is claimed by this thread alone.
        let scratch = unsafe { &mut *self.scratch[worker].get() };
        for (index, expert) in self.experts.iter().enumerate() {
            let count = expert.rows;
            let named = expert.at..expert.at + count;
            // SAFETY: no gather write is in flight (the caller's contract).
            let (values, q8) = unsafe { (values(&self.x[named.start * hidden..named.end * hidden]), values(&self.x8[named.start * blocks..named.end * blocks])) };
            let groups = expert.up.k().div_ceil(quant::Q8_BLOCK);
            let units = expands(worker, self.workers, expert.record, features.div_ceil(TILE) * groups);
            if units.is_empty() {
                continue;
            }
            let (first, last) = (units.start / groups, (units.end - 1) / groups);
            let whole = |tile: usize| units.start <= tile * groups && (tile + 1) * groups <= units.end;
            // Its whole tiles are a contiguous subrange of `first..=last`, one call a matrix.
            let from = if whole(first) { first } else { first + 1 };
            let to_excl = if whole(last) { last + 1 } else { last };
            if from < to_excl {
                let range = from * TILE..(to_excl * TILE).min(features);
                let width = range.len();
                let len = count * width;
                if let Some(weights) = &expert.gate {
                    scratch.a.resize(len, 0.0);
                    projection::project_arithmetic(weights, range.start, values, Some(q8), &mut scratch.a[..len]);
                }
                scratch.b.resize(len, 0.0);
                projection::project_arithmetic(&expert.up, range.start, values, Some(q8), &mut scratch.b[..len]);
                for (row, up) in scratch.b[..len].chunks_exact(width).enumerate() {
                    // SAFETY: these whole tiles of the expert's rows are this
                    // worker's alone.
                    let out = unsafe { writable(&self.products[(expert.at + row) * features + range.start..][..width]) };
                    if expert.gate.is_some() {
                        out.copy_from_slice(&scratch.a[row * width..(row + 1) * width]);
                    }
                    self.activate(expert, out, up);
                }
            }
            // Its partial tiles, at most its first and its last.
            for (side, tile) in [first, last].into_iter().enumerate().filter(|&(_, tile)| !whole(tile)) {
                if side == 1 && last == first {
                    break;
                }
                let groups_in = units.start.max(tile * groups) - tile * groups..units.end.min((tile + 1) * groups) - tile * groups;
                let columns = groups_in.start * quant::Q8_BLOCK..(groups_in.end * quant::Q8_BLOCK).min(expert.up.k());
                let rows = tile * TILE..((tile + 1) * TILE).min(features);
                // The expert's rows over these columns, row after row.
                scratch.values.clear();
                for row in 0..count {
                    scratch.values.extend_from_slice(&values[row * hidden..][columns.clone()]);
                }
                scratch.q8.clear();
                for row in 0..count {
                    scratch.q8.extend_from_slice(&q8[row * blocks..][groups_in.clone()]);
                }
                // SAFETY: this worker's own partial of the expert.
                let partial = unsafe { &mut *self.partials[(index * self.workers + worker) * 2 + side].get() };
                partial.tile = Some(tile);
                let partial_len = count * rows.len();
                if let Some(gate) = &expert.gate {
                    partial.gate.resize(partial_len, 0.0);
                    let (_, window) = gate.columns(columns.clone());
                    projection::project_arithmetic(&window, rows.start, &scratch.values, Some(&scratch.q8), &mut partial.gate[..partial_len]);
                }
                partial.up.resize(partial_len, 0.0);
                let (_, window) = expert.up.columns(columns);
                projection::project_arithmetic(&window, rows.start, &scratch.values, Some(&scratch.q8), &mut partial.up[..partial_len]);
            }
        }
    }

    /// `out` (the gate's sums, where gated) and `up` activated in place: the
    /// gated `glu` of the scaled sums, or the up-only activation.
    fn activate(&self, expert: &Expert, out: &mut [f32], up: &[f32]) {
        let (gate_scale, up_scale) = expert.scales;
        match &expert.gate {
            Some(_) => {
                let glu = gated(self.shape.function, self.shape.limit);
                for (value, up) in out.iter_mut().zip(up) {
                    *value = glu(gate_scale * *value, up_scale * up);
                }
            }
            None => {
                for (value, up) in out.iter_mut().zip(up) {
                    *value = functions::activate(self.shape.function, up_scale * up);
                }
            }
        }
    }

    /// Stage three, expert `index`'s products of q8 block `block`, for each of
    /// its rows: each tile of it expand shares computed part of joined from
    /// their sums in share order and activated, then the block quantized.
    ///
    /// # Safety
    /// The expand stage is finished.
    unsafe fn join(&self, index: usize, block: usize) {
        let features = self.shape.features;
        let expert = &self.experts[index];
        let range = block * quant::Q8_BLOCK..((block + 1) * quant::Q8_BLOCK).min(features);
        let tiles = range.start / TILE..range.end.div_ceil(TILE);
        // SAFETY: no expand write is in flight (the caller's contract).
        let partials = unsafe { values(&self.partials[index * self.workers * 2..(index + 1) * self.workers * 2]) };
        let mut has_partial = [false; TILES_PER_BLOCK];
        let mut any_partial = false;
        for partial in partials {
            if let Some(tile) = partial.tile.filter(|tile| tiles.contains(tile)) {
                has_partial[tile - tiles.start] = true;
                any_partial = true;
            }
        }
        for row in 0..expert.rows {
            let at = expert.at + row;
            // SAFETY: the block's split tiles and quantized block of each of
            // the expert's rows are this unit's alone.
            let (products, quantized) = unsafe {
                (
                    writable(&self.products[at * features + range.start..at * features + range.end]),
                    writable(&self.quantized[at * quant::blocks(features) + block..][..1]),
                )
            };
            if any_partial {
                for (offset, tile) in tiles.clone().enumerate() {
                    if !has_partial[offset] {
                        continue;
                    }
                    let rows = tile * TILE..((tile + 1) * TILE).min(features);
                    let width = rows.len();
                    let sums = row * width..(row + 1) * width;
                    let mut gate = [0.0f32; TILE];
                    let mut up = [0.0f32; TILE];
                    for partial in partials {
                        if partial.tile == Some(tile) {
                            if expert.gate.is_some() {
                                for (dst, &src) in gate[..width].iter_mut().zip(&partial.gate[sums.clone()]) {
                                    *dst += src;
                                }
                            }
                            for (dst, &src) in up[..width].iter_mut().zip(&partial.up[sums.clone()]) {
                                *dst += src;
                            }
                        }
                    }
                    let out = &mut products[rows.start - range.start..rows.end - range.start];
                    if expert.gate.is_some() {
                        out.copy_from_slice(&gate[..width]);
                    }
                    self.activate(expert, out, &up[..width]);
                }
            }
            quant::quantize(products, quantized);
        }
    }

    /// Stage four, share `worker`'s channels ([`share`], whole tiles): the
    /// weighted sum of every expert's down rows over each of its rows'
    /// quantized products, one call an expert; a row no entry names is zero.
    ///
    /// # Safety
    /// The join stage is finished.
    unsafe fn project(&self, worker: usize, output: *mut f32) {
        let (hidden, rows, features) = (self.shape.hidden, self.shape.rows, self.shape.features);
        let tiles = share(worker, self.workers, hidden.div_ceil(TILE));
        if tiles.is_empty() {
            return;
        }
        let channels = tiles.start * TILE..(tiles.end * TILE).min(hidden);
        let width = channels.len();
        // SAFETY: unit `worker` of PROJECT is claimed by this thread alone.
        let scratch = unsafe { &mut *self.scratch[worker].get() };
        scratch.a.clear();
        scratch.a.resize(rows * width, 0.0);
        let most = self.experts.iter().map(|expert| expert.rows).max().unwrap_or(0);
        scratch.b.resize(most * width, 0.0);
        let blocks = quant::blocks(features);
        for expert in &self.experts {
            let named = expert.at..expert.at + expert.rows;
            // SAFETY: no join write is in flight (the caller's contract).
            let (products, quantized) = unsafe {
                (
                    values(&self.products[named.start * features..named.end * features]),
                    values(&self.quantized[named.start * blocks..named.end * blocks]),
                )
            };
            let dots = &mut scratch.b[..named.len() * width];
            projection::project_arithmetic(&expert.down, channels.start, products, Some(quantized), dots);
            for (&(row, weight), dots) in self.entry_rows[named].iter().zip(dots.chunks_exact(width)) {
                for (sum, dot) in scratch.a[row * width..(row + 1) * width].iter_mut().zip(dots) {
                    *sum += weight * dot;
                }
            }
        }
        for row in 0..rows {
            // SAFETY: this worker alone writes these channels of the slot's
            // output, which no device work reads before the answer.
            let out = unsafe { std::slice::from_raw_parts_mut(output.add((self.slot * rows + row) * hidden + channels.start), width) };
            out.copy_from_slice(&scratch.a[row * width..(row + 1) * width]);
        }
    }

    /// The rows its entries name.
    fn rows_named(&self) -> BTreeSet<usize> {
        self.entry_rows.iter().map(|&(row, _)| row).collect()
    }
}

/// Worker `worker` of `workers`' expand units of record `record`'s
/// `units` (tiles by their columns' storage groups): contiguous, dealt from
/// a rotation of the record's own. Every worker's count is within one of
/// every other's whatever the request.
fn expands(worker: usize, workers: usize, record: usize, units: usize) -> std::ops::Range<usize> {
    share((worker + workers - record % workers) % workers, workers, units)
}

/// Worker `worker`'s units of `units`, shared by `workers` in contiguous
/// runs: the project stage's channel tiles, the same for every request.
fn share(worker: usize, workers: usize, units: usize) -> std::ops::Range<usize> {
    worker * units / workers..(worker + 1) * units / workers
}

/// What the site failed of one step: by each layer a request failed at,
/// the rows it names (its request's own rows, which the step's window at
/// that layer places in the batch, `Steps::batch_rows`), and why each
/// failed.
#[derive(Debug)]
pub(crate) struct SiteFailure {
    pub rows: BTreeMap<usize, BTreeSet<usize>>,
    pub reasons: Vec<String>,
}

/// The site's state on the executor (`compute::HostCompute`): its layers,
/// a cell of each slot of each, and the requests issued to the pool.
///
/// One worker at a time notices requests ([`Self::notice`]): it alone reads
/// the slots, takes a request and issues it into its cell, then announces
/// it by `issued`, the one word every other worker polls; none of them
/// reads a slot or writes a shared word to learn of it.
pub(super) struct Site {
    layers: Vec<SiteLayer>,
    /// `cells[layer offset + slot]`.
    cells: Box<[Cell]>,
    /// The worker noticing requests, or none ([`Self::notice`]).
    noticer: AtomicUsize,
    /// The requests issued so far, each numbered by its issue from one:
    /// written by the noticing worker alone.
    issued: AtomicU64,
    /// `order[issue mod cells]`: the cell of each recent issue. An issue
    /// older than the cells' count is answered (each cell holds one
    /// unanswered request at most), so its entry is never needed.
    order: Box<[AtomicUsize]>,
    /// Whether issuing has closed ([`Self::close`]).
    closed: AtomicBool,
    /// The requests issued and not yet answered: what keeps a worker
    /// serving.
    open: AtomicUsize,
    /// The pool's workers, by which each record's shares are cut.
    workers: usize,
    /// Each worker's order over the shares of a stage: its own, then its
    /// memory node's, then the others' (a stolen unit's lines move within
    /// the node where they can).
    orders: Box<[Box<[usize]>]>,
    /// Each step the site failed a request of, until the step takes it.
    failures: Mutex<BTreeMap<u64, SiteFailure>>,
    /// What gates its requests and observes its answers ([`SiteGate`]):
    /// none for a measurement's site, whose requests the host writes.
    gate: OnceLock<Arc<dyn SiteGate>>,
    /// When the site began, which its requests' marks count from.
    began: std::time::Instant,
    /// What its answered requests took ([`Self::served`]).
    served: ServedSums,
}

/// What the site's requests are gated by and its answers announced to,
/// where the device's waits are not the site's own spins (the load's parks
/// where they are submission boundaries, `watcher::Parks`): the site names
/// no owner of it.
pub(crate) trait SiteGate: Send + Sync {
    /// Whether `layer`'s request of `step` is the host's to read.
    fn published(&self, layer: usize, step: u64) -> bool;
    /// An answer word was written: whatever waits on one learns it from the
    /// word.
    fn answered(&self);
}

impl Site {
    /// The site over `layers`, served by the workers of `threads`, in their
    /// order.
    pub(super) fn new(layers: Vec<SiteLayer>, threads: &super::ComputeThreads) -> Result<Self, String> {
        for layer in &layers {
            let units = layer.shape.rows * layer.shape.choices * quant::blocks(layer.shape.features);
            if units >> COUNT_BITS != 0 {
                return Err(format!("a site layer of {units} join units, beyond a share's count"));
            }
        }
        let workers = threads.count();
        let nodes = threads.nodes.iter().enumerate().flat_map(|(node, cpus)| cpus.iter().map(move |_| node)).collect::<Vec<_>>();
        let orders = (0..workers)
            .map(|worker| {
                let mut order = (0..workers).map(|offset| (worker + offset) % workers).collect::<Vec<_>>();
                order.sort_by_key(|&share| (share != worker, nodes[share] != nodes[worker]));
                order.into()
            })
            .collect();
        let cells = layers
            .iter()
            .enumerate()
            .flat_map(|(index, layer)| (0..layer.shape.flight()).map(move |_| (index, layer.shape)))
            .collect::<Vec<_>>();
        Ok(Self {
            cells: cells.iter().map(|&(layer, shape)| Cell::new(layer, shape, workers)).collect(),
            order: cells.iter().map(|_| AtomicUsize::new(0)).collect(),
            layers,
            noticer: AtomicUsize::new(usize::MAX),
            issued: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            open: AtomicUsize::new(0),
            workers,
            orders,
            failures: Mutex::new(BTreeMap::new()),
            gate: OnceLock::new(),
            began: std::time::Instant::now(),
            served: ServedSums {
                requests: AtomicU64::new(0),
                experts: AtomicU64::new(0),
                terms: std::array::from_fn(|_| AtomicU64::new(0)),
            },
        })
    }

    /// The gate of each request and the observer of each answer from now
    /// on.
    pub(super) fn gated(&self, gate: Arc<dyn SiteGate>) {
        assert!(self.gate.set(gate).is_ok(), "a site has one gate");
    }

    /// Publish `layer`'s answer of `step` ([`SiteLayer::publish`]), then
    /// announce it.
    fn answer(&self, layer: &SiteLayer, step: u64, failed: bool) {
        layer.publish(step, failed);
        if let Some(gate) = self.gate.get() {
            gate.answered();
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub(super) fn layers(&self) -> &[SiteLayer] {
        &self.layers
    }

    /// Whether a request is issued and not yet answered.
    pub(super) fn busy(&self) -> bool {
        self.open.load(Ordering::Acquire) > 0
    }

    /// Issue no request from now: one taken after is failed at once (one
    /// taken as it closes may still be issued, and its noticing worker then
    /// serves it before it exits). The executor closes it before it stops
    /// its workers, which serve every request issued before they exit
    /// (`compute::work`).
    pub(super) fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    /// Whether worker `worker` notices requests ([`Self::take`]): it does
    /// already, or no worker does and it now will. One worker notices at a
    /// time, so no slot is read by more than one, and no take races
    /// another.
    pub(super) fn notice(&self, worker: usize) -> bool {
        match self.noticer.load(Ordering::Relaxed) {
            noticer if noticer == worker => true,
            usize::MAX => self.noticer.compare_exchange(usize::MAX, worker, Ordering::Acquire, Ordering::Relaxed).is_ok(),
            _ => false,
        }
    }

    /// Worker `worker` notices requests no more, where it did: before it
    /// enters a job's piece or parks, so a request is noticed by whichever
    /// worker is free.
    pub(super) fn release(&self, worker: usize) {
        if self.noticer.load(Ordering::Relaxed) == worker {
            self.noticer.store(usize::MAX, Ordering::Release);
        }
    }

    /// Whether some worker currently holds the noticing role.
    pub(super) fn has_noticer(&self) -> bool {
        self.noticer.load(Ordering::Acquire) != usize::MAX
    }

    /// Record that the site failed step `step`'s request at streamed layer
    /// `layer` for `why`, naming its request's `rows`: the one record of a
    /// failure, which [`Self::take_failure`] takes.
    pub(super) fn failed(&self, step: u64, layer: usize, rows: BTreeSet<usize>, why: &str) {
        let mut failures = self.failures.lock().expect("site failures");
        let failure = failures.entry(step).or_insert_with(|| SiteFailure { rows: BTreeMap::new(), reasons: Vec::new() });
        failure.rows.entry(layer).or_default().extend(rows);
        failure.reasons.push(format!("layer {layer}: {why}"));
    }

    /// What the site failed of step `step`, taken: none where it failed none
    /// of its requests.
    pub(super) fn take_failure(&self, step: u64) -> Option<SiteFailure> {
        self.failures.lock().expect("site failures").remove(&step)
    }

    /// Each group's tier records `views` address from now (between flights,
    /// or a growth's joined slot, which leaves every record a request names
    /// where it is), for every layer of the group. Returns the displaced
    /// `Records` so the caller can drop them outside `layer.tensors.write()`
    /// and any no-blocking-allocation scope.
    pub(super) fn address(&self, views: &[Vec<Records>]) -> Vec<Records> {
        let mut displaced = Vec::new();
        for layer in &self.layers {
            layer.tensors.write().expect("site tensors").address(&views[layer.group], &mut displaced);
        }
        displaced
    }

    /// Record `step`'s failure at site layer `index` of `rows` for `why`,
    /// then publish it, which releases the device.
    fn fail(&self, index: usize, step: u64, rows: BTreeSet<usize>, why: String) {
        let layer = &self.layers[index];
        self.failed(step, layer.layer, rows, &why);
        self.answer(layer, step, true);
    }

    /// Take every slot whose sequence is newer than its last taken, once
    /// the request is the host's to read (`SiteGate::published`): one of no
    /// entries answered at once with every row zero (it has no stage work),
    /// any other issued to the pool.
    /// The noticing worker's alone ([`Self::notice`]). True when one was
    /// taken.
    pub(super) fn take(&self) -> bool {
        let mut took = false;
        for (cell, index, slot) in self.slots() {
            let Some(step) = self.requested(cell, index, slot) else { continue };
            let layer = &self.layers[index];
            self.mark(&self.cells[cell], NOTICED);
            // It serves none until answered whole ([`Self::publish`]).
            self.cells[cell].served.store(0, Ordering::Relaxed);
            self.cells[cell].rows.store(0, Ordering::Relaxed);
            self.cells[cell].took.store(0, Ordering::Relaxed);
            self.cells[cell].taken.store(step, Ordering::Release);
            took = true;
            // SAFETY: `self.requested` verified `cell.answered == taken`, so no
            // worker holds a unit of this cell, and only the noticing worker
            // calls `take`.
            let work = unsafe { &mut *self.cells[cell].work.get() };
            layer.prepare(slot, step, work);
            if work.experts.is_empty() {
                layer.clear(slot);
                self.cells[cell].answered.store(step, Ordering::Release);
                self.answer(layer, step, false);
            } else if self.closed.load(Ordering::Acquire) {
                let rows = work.rows_named();
                work.release_weights();
                self.cells[cell].answered.store(step, Ordering::Release);
                self.fail(index, step, rows, "the executor stopped serving".into());
            } else {
                self.issue(cell);
            }
        }
        took
    }

    /// Each slot as (its cell, its site layer, its slot).
    fn slots(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let slots = self.layers.iter().enumerate().flat_map(|(index, layer)| (0..layer.shape.flight()).map(move |slot| (index, slot)));
        slots.enumerate().map(|(cell, (index, slot))| (cell, index, slot))
    }

    /// The step slot `slot` of site layer `index` (cell `cell`) requests,
    /// where it is newer than its last taken, the cell has answered that
    /// one ([`Cell::answered`]), and it is the host's to read
    /// (`SiteGate::published`).
    fn requested(&self, cell: usize, index: usize, slot: usize) -> Option<u64> {
        let layer = &self.layers[index];
        let step = layer.sequence(slot);
        let cell = &self.cells[cell];
        let taken = cell.taken.load(Ordering::Relaxed);
        (step > taken
            && cell.answered.load(Ordering::Acquire) == taken
            && self.gate.get().is_none_or(|gate| gate.published(layer.layer, step)))
        .then_some(step)
    }

    /// Issue the request already prepared in cell `index`, whose last request
    /// was answered (its slot was requested again), then announce it to every
    /// worker.
    fn issue(&self, index: usize) {
        let cell = &self.cells[index];
        let issue = self.issued.load(Ordering::Relaxed) + 1;
        // SAFETY: the cell's last request was answered (`Cell::answered`), and
        // `take` prepared `cell.work` before calling `issue`; no worker reads
        // it before claiming a unit of this issue, published below.
        let work = unsafe { cell.work() };
        // A sequence lock's write: no worker reads the cell's units as of
        // another issue.
        cell.progress.store(0, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        for (units, count) in cell.units.iter().zip(work.units()) {
            units.store(count, Ordering::Relaxed);
        }
        cell.step.store(work.step, Ordering::Relaxed);
        cell.progress.store(progress(issue, 0), Ordering::Release);
        self.mark(cell, ISSUED);
        self.order[(issue % self.cells.len() as u64) as usize].store(index, Ordering::Relaxed);
        self.open.fetch_add(1, Ordering::AcqRel);
        self.issued.store(issue, Ordering::Release);
    }

    /// Serve as worker `worker`, which has finished with every issue up to
    /// `seen`: for each issue after it in order, claim a unit of its stage
    /// in progress ([`Self::advance`]) and run it. With no request open it
    /// reads two words, and while every request it may serve waits on units
    /// elsewhere it reads only the lines of their shares.
    pub(super) fn serve(&self, worker: usize, seen: &mut u64) -> WorkerTurn {
        let issued = self.issued.load(Ordering::Acquire);
        // Every issue up to `issued` was counted open before it was
        // announced: none open, all of them are answered.
        if self.open.load(Ordering::Acquire) == 0 {
            *seen = issued;
            return WorkerTurn::Idle;
        }
        let mut served = WorkerTurn::Idle;
        for issue in *seen + 1..=issued {
            let cell = self.order[(issue % self.cells.len() as u64) as usize].load(Ordering::Relaxed);
            match self.advance(cell, issue, worker) {
                None if issue == *seen + 1 => *seen = issue,
                None => {}
                Some(WorkerTurn::Ran) => return WorkerTurn::Ran,
                Some(_) => served = WorkerTurn::Waiting,
            }
        }
        served
    }

    /// Worker `worker`'s part in issue `issue` of cell `cell`, none once it
    /// has none (the cell holds a later issue, or every stage is finished):
    /// a unit of its stage in progress claimed and run, its own share's
    /// first ([`Self::orders`]); else, once every share of the stage is
    /// complete, the stage finished, and past the last, the request
    /// answered; else waiting on units in progress elsewhere.
    fn advance(&self, cell: usize, issue: u64, worker: usize) -> Option<WorkerTurn> {
        let cell = &self.cells[cell];
        let (stage, units) = cell.stage(issue)?;
        let lens = |share: usize| self::share(share, self.workers, units);
        for &share in &self.orders[worker] {
            let Some(at) = cell.shares[share].claim(issue, stage, lens(share).len()) else { continue };
            // SAFETY: a unit of the issue is claimed, so it is not answered
            // before the unit completes.
            let work = unsafe { cell.work() };
            let output = self.layers[cell.layer].output.as_ptr();
            // SAFETY: a stage's units are claimed only once the stages
            // before it are finished (`Cell::stage`).
            unsafe { work.run(stage, lens(share).start + at, output) };
            cell.shares[share].complete(issue, stage);
            return Some(WorkerTurn::Ran);
        }
        if !self.orders[worker].iter().all(|&share| cell.shares[share].finished(issue, stage, lens(share).len())) {
            return Some(WorkerTurn::Waiting);
        }
        // Every worker that sees the stage finished may advance it; only one
        // whose read finds it not yet advanced writes the word.
        let at = progress(issue, stage);
        if cell.progress.load(Ordering::Relaxed) == at
            && cell.progress.compare_exchange(at, progress(issue, stage + 1), Ordering::AcqRel, Ordering::Relaxed).is_ok()
        {
            self.mark(cell, STAGE_ENDS + stage);
            if stage + 1 == STAGES {
                self.publish(cell);
            }
        }
        Some(WorkerTurn::Ran)
    }

    /// Answer `cell`'s request, every stage of it finished, as its one
    /// publisher: every output is written.
    fn publish(&self, cell: &Cell) {
        // SAFETY: its publisher, before the answer: no worker holds a unit of
        // it, and the noticing worker will not rewrite the cell before
        // `cell.answered` is stored below.
        let work = unsafe { &mut *cell.work.get() };
        let step = work.step;
        let experts = work.experts.len() as u64;
        let rows = work.entry_rows.len() as u64;
        // Drop `HeldWeights` so slab views are unpinned before the cell is
        // released, while retaining all heap buffer capacities for reuse.
        work.release_weights();
        let layer = &self.layers[cell.layer];
        // Its path's terms, read before the answer lets the slot be
        // requested again (which rewrites the marks), counted as the answer
        // is written.
        let marks = cell.marks.each_ref().map(|mark| mark.load(Ordering::Relaxed));
        // The experts it served, which its step's completion counts: the
        // slot is not requested again before then.
        cell.served.store(experts, Ordering::Relaxed);
        cell.rows.store(rows, Ordering::Relaxed);
        cell.took.store((self.began.elapsed().as_nanos() as u64).saturating_sub(marks[NOTICED]), Ordering::Relaxed);
        // Nothing of the request is read once it is answered: the device may
        // then request the slot again, which rewrites the cell.
        // Out of the cell, the request frees it for the slot's next, which
        // the device makes only after the answer below.
        cell.answered.store(step, Ordering::Release);
        self.answer(layer, step, false);
        self.count(marks, experts);
        self.open.fetch_sub(1, Ordering::AcqRel);
    }

    /// Mark `cell`'s request at point `point` of its path, now.
    fn mark(&self, cell: &Cell, point: usize) {
        cell.marks[point].store(self.began.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Count an answered request whose path reached its points at `marks`,
    /// its answer written now.
    fn count(&self, marks: [u64; MARKS], experts: u64) {
        let now = self.began.elapsed().as_nanos() as u64;
        let ends = marks.iter().skip(1).copied().chain([now]);
        for (term, (start, end)) in marks.iter().zip(ends).enumerate() {
            self.served.terms[term].fetch_add(end.saturating_sub(*start), Ordering::Relaxed);
        }
        self.served.experts.fetch_add(experts, Ordering::Relaxed);
        self.served.requests.fetch_add(1, Ordering::Relaxed);
    }

    /// The experts site layer `index`'s request of step `step` served, their
    /// rows and its path's time, once the step has completed (the slot is
    /// not requested again before); none where the step requested none
    /// there.
    pub(super) fn sited(&self, index: usize, step: u64) -> (u64, u64, std::time::Duration) {
        let base = self.layers[..index].iter().map(|layer| layer.shape.flight()).sum::<usize>();
        let cell = &self.cells[base + (step % self.layers[index].shape.flight() as u64) as usize];
        match cell.taken.load(Ordering::Acquire) == step {
            true => (
                cell.served.load(Ordering::Acquire),
                cell.rows.load(Ordering::Relaxed),
                std::time::Duration::from_nanos(cell.took.load(Ordering::Relaxed)),
            ),
            false => (0, 0, std::time::Duration::ZERO),
        }
    }

    /// What its answered requests took since the load ([`SiteServed`]): a
    /// reading taken while requests are answered may count a request's
    /// terms before its count, which the next reading settles.
    pub(super) fn served(&self) -> SiteServed {
        SiteServed {
            requests: self.served.requests.load(Ordering::Relaxed),
            experts: self.served.experts.load(Ordering::Relaxed),
            terms: std::array::from_fn(|term| std::time::Duration::from_nanos(self.served.terms[term].load(Ordering::Relaxed))),
        }
    }

    /// Each request issued and not yet answered, as (site layer, step, each
    /// stage's units completed of its units): what the exchange's watchdog
    /// reports of the site (`Compute::report`).
    pub(super) fn progress(&self) -> Vec<(usize, u64, [(usize, usize); STAGES])> {
        let open = self.cells.iter().filter_map(|cell| {
            let (issue, stages) = progressed(cell.progress.load(Ordering::Acquire));
            let units = |stage: usize| cell.units[stage].load(Ordering::Relaxed);
            let done = |stage: usize| cell.shares.iter().map(|share| Share::count(share.done[stage].load(Ordering::Acquire), issue).unwrap_or(0)).sum();
            (issue > 0 && stages < STAGES).then(|| (cell.layer, cell.step.load(Ordering::Relaxed), std::array::from_fn(|stage| (done(stage), units(stage)))))
        });
        open.collect()
    }
}

/// The seconds the executor on `threads` takes to answer each of
/// `requests`, `(experts, rows each)`: `experts` distinct tier records of
/// `layer`, each named by `rows each` rows, from its sequence's store to its
/// answer, sampled ([`Sample::of`]) over every set of records no sample of
/// the request read before, as many as the tier's records hold (a request
/// of no entries as many times as one expert's), after one request whose
/// time is discarded (the pool's first pays its warm-up). A request of no entries (no experts,
/// or no rows) is the handshake alone: a request noticed and answered,
/// which has no stage work ([`Site::take`]). It runs on an executor of its
/// own over the layer alone (`SiteLayer::owned`), sized as the load's: the
/// cost function's CPU lines.
pub(crate) fn measure(
    layer: SiteLayer,
    threads: &super::ComputeThreads,
    requests: &[(usize, usize)],
) -> Result<Vec<magnitude_cost_model::Sample>, String> {
    let shape = layer.shape;
    let records = layer.tensors.read().expect("site tensors").up.count();
    for &(experts, rows) in requests {
        if experts > records.min(shape.choices) || rows > shape.rows {
            return Err(format!("a site measurement of {experts} experts by {rows} rows beyond the layer's"));
        }
    }
    let compute = super::HostCompute::start(vec![layer], threads)?;
    let handle = compute.handle();
    let _flight = handle.flight().expect("a site of one layer");
    let layer = &handle.layers()[0];
    let values = vec![0.5f32; shape.rows * shape.hidden];
    let mut step = 0u64;
    let warm_up = requests.first().map(|&request| (request, true));
    let measured = requests.iter().map(|&request| (request, false));
    warm_up
        .into_iter()
        .chain(measured)
        .map(|((experts, rows), discarded)| {
            let samples = match discarded {
                true => 1,
                false => records / experts.max(1),
            };
            let mut times = Vec::with_capacity(samples);
            for repeat in 0..samples {
                step += 1;
                let slot = shape.slot(step);
                let records = (0..experts).map(|expert| (repeat * experts + expert) % records).collect::<Vec<_>>();
                let entries = (0..experts * rows)
                    .map(|entry| (entry % rows, records[entry / rows], 1.0, records[entry / rows] % shape.experts))
                    .collect::<Vec<_>>();
                layer.payload(step, &entries, &values);
                let began = std::time::Instant::now();
                layer.slots.request(step, 0);
                loop {
                    match layer.ready(slot) {
                        ready if ready == step => break,
                        ready if ready == step | FAILED => {
                            return Err(format!("the site failed its measurement of {experts} experts by {rows} rows"));
                        }
                        _ => std::hint::spin_loop(),
                    }
                }
                times.push(began.elapsed().as_secs_f64());
            }
            Ok((discarded, times))
        })
        .filter(|measured| !matches!(measured, Ok((true, _))))
        .map(|measured| {
            let (_, times) = measured?;
            magnitude_cost_model::Sample::of(&times).ok_or_else(|| format!("a site measurement of {} samples has no resolution", times.len()))
        })
        .collect()
}

#[cfg(test)]
mod tests;
