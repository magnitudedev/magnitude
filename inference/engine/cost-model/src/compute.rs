//! A census's launches priced call kind by call kind (Seismic's per-call
//! formula, [`seismic_measure::pricing`]), and what a routed pass over a
//! streamed layer's experts costs ([`ExpertPass`]).
//!
//! A launch of up to the census's decode row classes is a decode step of
//! its sequences' rows, each kind's calls in their decode forms at its
//! launches at those rows, their arithmetic at the decode row line and its
//! routed calls reading the experts their caller states ([`read_terms`]);
//! beyond them, a prefill launch, its calls each one
//! launch of their prefill form over the same work. What the prefill
//! forms' kernels achieve beyond that work is not priced.

use crate::census::{Census, RoutedKind};
use crate::expert_lines::{GpuLines, LineError, Moves, PassLine};
use seismic_measure::pricing::{self, Bytes, CallSeconds, CallTerm, ExpertRead, Experts, Flops, HistoryFigure, LaunchSeconds, PriceError, Pricing, Rates, Storage};
use std::collections::BTreeMap;

/// A launch of `census`'s calls over `rows` rows of `sequences` sequences
/// at `depth` tokens of context, priced at `pricing`: each kind once, at
/// its multiplicity, a routed call's rows routing independently.
pub(crate) fn launch_terms(census: &Census, pricing: &Pricing<'_>, sequences: u64, rows: u64, depth: u32) -> Result<LaunchSeconds, PriceError> {
    read_terms(census, pricing, sequences, rows, depth, &|experts| ExpertRead::independent(experts, rows))
}

/// [`launch_terms`], a routed call's rows reading and computing what
/// `read` says of its experts. Within the decode row classes each call is
/// priced in its decode form ([`pricing::decode_call_seconds`]); beyond
/// them, in its prefill form, its rows route independently.
pub(crate) fn read_terms(census: &Census, pricing: &Pricing<'_>, sequences: u64, rows: u64, depth: u32, read: &dyn Fn(&Experts) -> ExpertRead) -> Result<LaunchSeconds, PriceError> {
    let rates = &pricing.figures.rates;
    let calls = (0..census.kinds().len())
        .map(|kind| {
            let seconds = call(census, kind, rates, sequences, rows, depth, census.kinds()[kind].work.experts.as_ref().map(read))?;
            Ok(CallTerm { seconds, multiplicity: census.multiplicity(kind) })
        })
        .collect::<Result<Vec<_>, PriceError>>()?;
    pricing::compose(calls, census.boundaries(), pricing)
}

/// The routed calls of a streamed layer of `kind` in a launch of `rows`
/// rows of `sequences` sequences, their experts computing `picks` picks
/// routed independently, priced at `pricing` at no context: the sum of the
/// calls' seconds, composed, with none of the launch's own terms beside
/// them.
///
/// # Panics
/// Where a routed call reads history: it reads none at any depth.
pub(crate) fn routed_seconds(census: &Census, kind: RoutedKind, pricing: &Pricing<'_>, sequences: u64, rows: u64, picks: u64) -> Result<f64, PriceError> {
    let rates = &pricing.figures.rates;
    let calls = census.routed(kind).iter().try_fold(0.0, |seconds, &at| {
        let work = &census.kinds()[at].work;
        assert!(work.history.is_none(), "invariant violated: a routed call of {kind:?} reads history");
        let read = work.experts.as_ref().map(|experts| ExpertRead::picked(experts, picks));
        Ok::<_, PriceError>(seconds + call(census, at, rates, sequences, rows, 0, read)?.total())
    })?;
    Ok(pricing.figures.composition.clone()? * calls)
}

/// Streamed layer `layer`'s hint launch in a launch of `rows` rows of
/// `sequences` sequences, priced at `pricing` at no context, composed, with
/// none of the launch's own terms beside it: none where the layer hints
/// none.
///
/// # Panics
/// Where a hint launch reads history: it reads none at any depth.
pub(crate) fn hint_seconds(census: &Census, layer: usize, pricing: &Pricing<'_>, sequences: u64, rows: u64) -> Result<Option<f64>, PriceError> {
    let Some(kind) = census.hint(layer) else {
        return Ok(None);
    };
    let rates = &pricing.figures.rates;
    let work = &census.kinds()[kind].work;
    assert!(work.history.is_none(), "invariant violated: streamed layer {layer}'s hint launch reads history");
    let seconds = call(census, kind, rates, sequences, rows, 0, work.experts.as_ref().map(|experts| ExpertRead::independent(experts, rows)))?.total();
    Ok(Some(pricing.figures.composition.clone()? * seconds))
}

/// A call of `census`'s kind `kind` in a launch of `rows` rows of
/// `sequences` sequences at `depth`, priced at `rates`: within the decode
/// row classes in its decode form, its routed experts read as `read` says
/// ([`pricing::decode_call_seconds`]); beyond them in its prefill form, its
/// rows routing independently ([`pricing::call_seconds`]).
fn call(census: &Census, kind: usize, rates: &Rates, sequences: u64, rows: u64, depth: u32, read: Option<ExpertRead>) -> Result<CallSeconds, PriceError> {
    let (launches, history) = call_at(census, kind, rates, sequences, rows)?;
    let work = &census.kinds()[kind].work;
    Ok(match census.decodes(rows) {
        true => pricing::decode_call_seconds(work, history, rates, rows, sequences, depth, launches, read),
        false => pricing::call_seconds(work, history, rates, rows, sequences, depth, launches),
    })
}

/// What a call of `census`'s kind `kind` in a launch of `rows` rows of
/// `sequences` sequences is priced with: its launches (at the rows within
/// the decode row classes; beyond them one, its prefill form) and its
/// history's figure at `rates`.
fn call_at<'r>(census: &Census, kind: usize, rates: &'r Rates, sequences: u64, rows: u64) -> Result<(u64, Option<&'r HistoryFigure>), PriceError> {
    assert!(sequences >= 1 && sequences <= rows, "invariant violated: {sequences} sequences over {rows} rows");
    let call = &census.kinds()[kind];
    let launches = match census.decodes(rows) {
        true => call.launches.at(rows)?,
        false => 1,
    };
    Ok((launches, rates.history_of(&call.work)?))
}

/// One decode step's time at `depth` over `sequences` sequences, a row
/// each: only where their rows run the decode forms
/// ([`Census::decode_step`]).
pub fn step_seconds(census: &Census, pricing: &Pricing<'_>, sequences: u64, depth: u32) -> Result<LaunchSeconds, PriceError> {
    census.decode_step(sequences)?;
    launch_terms(census, pricing, sequences, sequences, depth)
}

/// Whether the rates read host storage's routes, which a streamed layer's
/// link line takes and no other figure stands in for: read; not yet read;
/// or never to be read, and why.
#[derive(Clone, Debug, PartialEq)]
pub enum HostReads {
    Read,
    Unread,
    Unobtainable(String),
}

/// What a routed pass over a streamed layer's experts costs on this
/// executor: a pass reads each distinct expert it routes to once and
/// computes every choice, so it costs, as a call does, the slower of its
/// experts' weights at their format's rate (each expert one record of its
/// bank group's tensors, as resident) and its choices' arithmetic over
/// their format's throughput. A routed kind's layers are alike
/// ([`RoutedKind`]), so each kind is priced once.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertPass {
    rates: Rates,
    /// Each kind's expert: its bytes and its FLOPs a choice, by format.
    kinds: BTreeMap<RoutedKind, (Vec<Bytes>, Vec<Flops>)>,
    /// Each streamed layer's kind, in decoder order.
    layers: Vec<RoutedKind>,
    host: HostReads,
}

impl ExpertPass {
    /// The passes of streamed layers each of its kind and whose expert is
    /// its bytes and FLOPs a choice (`layers`, in decoder order), at
    /// `rates`, which read host storage as `host` says.
    ///
    /// # Panics
    /// Where two layers of one kind have other experts.
    pub fn new(rates: Rates, layers: Vec<(RoutedKind, Vec<Bytes>, Vec<Flops>)>, host: HostReads) -> Self {
        let mut kinds = BTreeMap::<RoutedKind, (Vec<Bytes>, Vec<Flops>)>::new();
        let layers = layers
            .into_iter()
            .map(|(kind, bytes, flops)| {
                let known = kinds.entry(kind).or_insert_with(|| (bytes.clone(), flops.clone()));
                assert_eq!(*known, (bytes, flops), "invariant violated: layers of {kind:?} have other experts");
                kind
            })
            .collect();
        Self { rates, kinds, layers, host }
    }

    /// Streamed layer `layer`'s kind.
    pub fn kind(&self, layer: usize) -> RoutedKind {
        self.layers[layer]
    }

    /// The kind's expert.
    fn of(&self, kind: RoutedKind) -> &(Vec<Bytes>, Vec<Flops>) {
        self.kinds.get(&kind).unwrap_or_else(|| panic!("invariant violated: {kind:?} is none of the pass's"))
    }

    /// The seconds of a pass at a layer of kind `kind` reading `experts`
    /// experts for `choices` choices.
    pub(crate) fn kind_seconds(&self, kind: RoutedKind, experts: f64, choices: f64) -> f64 {
        let (bytes, flops) = self.of(kind);
        self.rates.weights(bytes, experts).max(self.rates.compute(flops, choices))
    }

    /// The seconds of a pass at streamed layer `layer` reading `experts`
    /// experts for `choices` choices.
    pub fn seconds(&self, layer: usize, experts: f64, choices: f64) -> f64 {
        self.kind_seconds(self.kind(layer), experts, choices)
    }

    /// Streamed layer `layer`'s expert: its resident record's bytes and its
    /// FLOPs a choice, every format's together.
    pub fn expert(&self, layer: usize) -> (u64, u64) {
        let (bytes, flops) = self.of(self.kind(layer));
        (bytes.iter().map(|bytes| bytes.bytes).sum(), flops.iter().map(|flops| flops.flops).sum())
    }

    /// Why [`Self::lines`] has none for good: the host route's rate is one
    /// no job will measure.
    pub fn host_unobtainable(&self) -> Option<&str> {
        match &self.host {
            HostReads::Unobtainable(reason) => Some(reason),
            HostReads::Read | HostReads::Unread => None,
        }
    }

    /// Streamed layer `layer`'s expert on the GPU ([`GpuLines`]): its record
    /// read in place where the host tier lies (host storage: across a
    /// discrete GPU's link) and from the GPU's memory, each against its
    /// arithmetic a row; none until the rates read host storage's routes.
    pub fn lines(&self, layer: usize) -> Option<Result<GpuLines, LineError>> {
        let (bytes, flops) = self.of(self.kind(layer));
        let row = self.rates.compute(flops, 1.);
        let line = |storage| PassLine::of(self.rates.weights_in(storage, bytes, 1.), row);
        matches!(self.host, HostReads::Read).then(|| Ok(GpuLines { link: line(Storage::Host)?, resident: line(Storage::Device)?, moves: Moves::NONE }))
    }
}
