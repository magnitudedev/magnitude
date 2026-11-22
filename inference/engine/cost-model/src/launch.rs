//! The price of a loaded model's launches ([`LaunchPrices`]): a launch of
//! some rows at some depth in each form the load decodes in, its experts
//! held whole and as the load plans them, from the load's censuses and the
//! machine's figures.
//!
//! A price is the sum of its terms ([`LaunchSeconds`]): each call kind's
//! seconds times its multiplicity, composed, plus the launch's fixed step
//! and submission terms. In the streamed form the price is device time;
//! what its misses add is the step model's apart
//! (`StreamedLayers::step_seconds`), over a miss rate (the generic model's,
//! or the residency's live reads) and the lines its links read at.
//!
//! A load's prices are one immutable snapshot: cloning shares it.

use crate::census::{Census, RoutedKind};
use crate::compute::{hint_seconds, launch_terms, read_terms, routed_seconds};
use crate::speculative::{DraftCensus, DraftMethod, SpeculativePrice};
use crate::streamed::FormSeconds;
use seismic_measure::pricing::{expert_union, ExpertRead, Experts, Figures, LaunchSeconds, PriceError, Pricing};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A load's target censuses: of a load holding its experts whole, its
/// launch; of one that banks them, its launch with its experts held whole
/// (`resident`) and in its streamed form (`streamed`).
#[derive(Clone, Debug, PartialEq)]
pub enum TargetCensus {
    Resident(Census),
    Banked { resident: Census, streamed: Census },
}

impl TargetCensus {
    /// A load's censuses as they pair: with a streamed census a banked
    /// load's, without one a load holding its experts whole.
    pub fn of(resident: Census, streamed: Option<Census>) -> Self {
        match streamed {
            Some(streamed) => Self::Banked { resident, streamed },
            None => Self::Resident(resident),
        }
    }

    /// Its censuses, each once: those the figures it is priced at cover.
    pub fn censuses(&self) -> Vec<&Census> {
        match self {
            Self::Resident(census) => vec![census],
            Self::Banked { resident, streamed } => vec![resident, streamed],
        }
    }

    /// `launch` in each form: its experts held whole, and as the load plans
    /// them, which a load holding them whole decodes alike.
    pub fn each(&self, launch: impl Fn(&Census) -> Result<LaunchSeconds, PriceError>) -> Result<FormSeconds, PriceError> {
        match self {
            Self::Resident(census) => {
                let seconds = launch(census)?.total();
                Ok(FormSeconds { resident: seconds, streamed: seconds })
            }
            Self::Banked { resident, streamed } => Ok(FormSeconds { resident: launch(resident)?.total(), streamed: launch(streamed)?.total() }),
        }
    }

    /// The census whose decode rows each of its censuses has
    /// ([`LaunchPrices::new`]).
    fn decoding(&self) -> &Census {
        match self {
            Self::Resident(census) | Self::Banked { resident: census, .. } => census,
        }
    }
}

/// How a sequence's rows route, as the decode measured it: what a front
/// pass's rows read of a streamed layer's experts.
///
/// - `fresh`: the experts a sequence's row at each lag (lag 1 first) adds
///   to those of the rows before it, a layer's mean. Rows of a sequence
///   are its next tokens, which share experts, so each lag adds fewer than
///   a row's picks; beyond the lags measured (none before a pass is
///   measured), what independent routing adds ([`expert_union`]).
/// - `held`: the share of picks the bank holds, none where unmeasured (then
///   every pick). A front row computes only the picks the step holds
///   besides those its exact row chose (`routed.seismic`'s `computes`), so
///   in the streamed form its fresh experts are read only as held; held
///   whole, every one is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrontRoutes {
    pub fresh: Vec<f64>,
    pub held: Option<f64>,
}

impl FrontRoutes {
    /// The experts a sequence's exact row and its `ahead` front rows read
    /// of a layer of `experts`, a layer's mean, the front rows' fresh
    /// experts read where held (`streamed`) or all of them.
    fn sequence(&self, experts: &Experts, ahead: u64, streamed: bool) -> f64 {
        let independent = |lag: u64| expert_union(experts.experts, experts.selected, lag + 1) - expert_union(experts.experts, experts.selected, lag);
        let fresh = (1..=ahead).map(|lag| self.fresh.get(lag as usize - 1).copied().unwrap_or_else(|| independent(lag))).sum::<f64>();
        let held = match streamed {
            true => self.held.unwrap_or(1.),
            false => 1.,
        };
        experts.selected as f64 + held * fresh
    }

    /// What a launch of `requests` sequences' exact rows, `fronted` of them
    /// with `ahead` front rows each, reads of a layer of `experts`: the
    /// sequences route independently of one another, the union of their
    /// experts; it computes every pick of its rows, the most its front rows
    /// can compute.
    fn read(&self, experts: &Experts, requests: u64, fronted: u64, ahead: u64, streamed: bool) -> ExpertRead {
        let total = experts.experts as f64;
        let missed = |read: f64, sequences: u64| (1. - read / total).clamp(0., 1.).powf(sequences as f64);
        let unread = missed(experts.selected as f64, requests - fronted) * missed(self.sequence(experts, ahead, streamed), fronted);
        ExpertRead { experts: total * (1. - unread), picks: ((requests + fronted * ahead) * experts.selected) as f64 }
    }
}

/// What a load's prices are priced from: its censuses at its figures, each
/// launch priced kept (`priced`), a function of its sequences, rows and
/// exact depth alone, so a launch priced again reads what it priced the
/// first time.
#[derive(Debug)]
struct Source {
    target: TargetCensus,
    draft: Option<DraftCensus>,
    figures: Figures,
    lookahead: bool,
    priced: Mutex<HashMap<(u32, u32, u32), FormSeconds>>,
    priced_verify: Mutex<HashMap<(u32, u32, u32), FormSeconds>>,
    priced_draft: Mutex<HashMap<(u32, u8, u32), f64>>,
}

/// The price of a loaded model's launches: one immutable snapshot of its
/// censuses priced at the machine's figures.
#[derive(Clone, Debug)]
pub struct LaunchPrices {
    source: Arc<Source>,
}

impl LaunchPrices {
    /// The prices of a load of `target`'s censuses at `figures`, its decode
    /// under `lookahead`, with no drafter census.
    ///
    /// # Panics
    /// Where `figures` lack a rate of a census's format or layout, or its
    /// censuses decode up to other rows.
    pub fn new(target: TargetCensus, figures: Figures, lookahead: bool) -> Self {
        Self::with_draft(target, None, figures, lookahead)
    }

    /// The prices of a load of `target`'s and optional `draft`'s censuses at
    /// `figures`, its plain decode under `lookahead`.
    ///
    /// # Panics
    /// Where `figures` lack a rate of any target or draft census's format or
    /// layout, or target censuses decode up to other rows.
    pub fn with_draft(target: TargetCensus, draft: Option<DraftCensus>, figures: Figures, lookahead: bool) -> Self {
        let decode_rows = target.decoding().decode_rows();
        for census in target.censuses() {
            assert!(figures.rates.covers(&census.formats(), &census.layouts()), "invariant violated: the figures price every format and layout of the load's censuses");
            assert_eq!(census.decode_rows(), decode_rows, "invariant violated: a load's censuses decode up to the same rows");
        }
        if let Some(draft) = &draft {
            for census in draft.censuses() {
                assert!(
                    figures.rates.covers(&census.formats(), &census.layouts()),
                    "invariant violated: the figures price every format and layout of the load's draft censuses"
                );
            }
        }
        Self {
            source: Arc::new(Source {
                target,
                draft,
                figures,
                lookahead,
                priced: Mutex::default(),
                priced_verify: Mutex::default(),
                priced_draft: Mutex::default(),
            }),
        }
    }

    /// The structural class of the load's drafter, if censused.
    pub fn draft_method(&self) -> Option<DraftMethod> {
        self.source.draft.as_ref().map(|draft| draft.method)
    }

    /// Prior wall-clock seconds of a drafter launch generating `generated`
    /// proposals for `sequences` sequences at `depth` tokens, unpipelined
    /// (`lookahead: false`). Returns `Ok(None)` when no draft census is held
    /// or `generated == 0`.
    pub fn draft_seconds(&self, sequences: u32, generated: u8, depth: u32) -> Result<Option<f64>, PriceError> {
        let Source { draft, figures, priced_draft, .. } = &*self.source;
        let Some(draft) = draft else {
            return Ok(None);
        };
        let Some(width) = draft.width(generated) else {
            return Ok(None);
        };
        let key = (sequences, width.proposals, depth);
        {
            let cache = priced_draft.lock().expect("a load's priced draft launches");
            if let Some(&seconds) = cache.get(&key) {
                return Ok(Some(seconds));
            }
        }
        let seconds = SpeculativePrice::draft_launch(width, figures, u64::from(sequences.max(1)), depth)?;
        priced_draft.lock().expect("a load's priced draft launches").insert(key, seconds);
        Ok(Some(seconds))
    }

    /// Whether the target model uses a banked/streamed MoE plan (whose live step
    /// time includes streaming miss service, so GPU-only `.resident` launch priors
    /// must not be used in place of `StreamedVerifyPrior`).
    pub fn is_banked(&self) -> bool {
        matches!(self.source.target, TargetCensus::Banked { .. })
    }

    /// Prior wall-clock seconds of an unpipelined speculative verify launch
    /// (`lookahead: false`) of `rows` rows for `sequences` sequences at
    /// `depth` tokens, in each form (`resident` and `streamed`).
    pub fn verify_seconds(&self, sequences: u32, rows: u32, depth: u32) -> Result<FormSeconds, PriceError> {
        let Source { target, figures, priced_verify, .. } = &*self.source;
        let key = (sequences, rows, depth);
        {
            let cache = priced_verify.lock().expect("a load's priced verify launches");
            if let Some(&seconds) = cache.get(&key) {
                return Ok(seconds);
            }
        }
        let pricing = Pricing { figures, lookahead: false };
        let seconds = target.each(|census| launch_terms(census, &pricing, sequences.into(), rows.into(), depth))?;
        priced_verify.lock().expect("a load's priced verify launches").insert(key, seconds);
        Ok(seconds)
    }

    /// Prior wall-clock seconds of a plain 1-row-per-sequence decode launch of
    /// `sequences` sequences at `depth` tokens under the load's `lookahead`
    /// policy, in each form.
    pub fn plain_seconds(&self, sequences: u32, depth: u32) -> Result<FormSeconds, PriceError> {
        self.target(sequences, sequences, depth)
    }

    /// The sequences a launch of `rows` rows runs: beyond the decode rows
    /// the prefill of one, else the decode step of a row each
    /// ([`Census::decode_step`]).
    fn sequences(&self, rows: u32) -> Result<u32, PriceError> {
        let census = self.source.target.decoding();
        match rows > 0 && !census.decodes(rows.into()) {
            true => Ok(1),
            false => {
                census.decode_step(rows.into())?;
                Ok(rows)
            }
        }
    }

    /// The load's launch of `rows` rows at `depth` tokens in each form, as
    /// a launch of that many rows runs ([`Self::sequences`]).
    pub fn launch(&self, rows: u32, depth: u32) -> Result<FormSeconds, PriceError> {
        self.target(self.sequences(rows)?, rows, depth)
    }

    /// The seconds of the routed calls of a streamed layer of `kind` in a
    /// launch of `rows` rows as it runs ([`Self::sequences`]), their experts
    /// computing `picks` picks (the exact rows' and those a two-row pass's
    /// front rows compute): the front gate's one price of a front value,
    /// at no context, as a routed call reads none.
    ///
    /// # Panics
    /// Where the load banks no experts, or `kind` is none of its streamed
    /// layers'.
    pub fn routed(&self, kind: RoutedKind, rows: u32, picks: u32) -> Result<f64, PriceError> {
        let Source { target, figures, lookahead, .. } = &*self.source;
        let TargetCensus::Banked { streamed, .. } = target else {
            panic!("invariant violated: a load holding its experts whole has no streamed layer's routed calls");
        };
        let sequences = self.sequences(rows)?;
        routed_seconds(streamed, kind, &Pricing { figures, lookahead: *lookahead }, sequences.into(), rows.into(), picks.into())
    }

    /// The seconds of streamed layer `layer`'s hint launch in a launch of
    /// `rows` rows as it runs ([`Self::sequences`]): the hint gate's one
    /// price of running a layer's hints, at no context, as a hint reads
    /// none. None where the layer hints none.
    ///
    /// # Panics
    /// Where the load banks no experts.
    pub fn hint(&self, layer: usize, rows: u32) -> Result<Option<f64>, PriceError> {
        let Source { target, figures, lookahead, .. } = &*self.source;
        let TargetCensus::Banked { streamed, .. } = target else {
            panic!("invariant violated: a load holding its experts whole has no streamed layer's hints");
        };
        let sequences = self.sequences(rows)?;
        hint_seconds(streamed, layer, &Pricing { figures, lookahead: *lookahead }, sequences.into(), rows.into())
    }

    /// A front pass of `requests` requests' exact rows, `fronted` of them
    /// running `ahead` front rows each, at `depth` tokens in each form, its
    /// routed calls reading their experts as `routes` says
    /// ([`FrontRoutes`]): what the regime prices a depth it has not
    /// measured by. Not kept: its routes are the decode's live measure.
    ///
    /// # Panics
    /// Where `fronted` exceeds `requests`.
    pub fn front(&self, requests: u32, fronted: u32, ahead: u32, depth: u32, routes: &FrontRoutes) -> Result<FormSeconds, PriceError> {
        assert!(fronted <= requests, "invariant violated: {fronted} of {requests} requests front");
        let Source { target, figures, lookahead, .. } = &*self.source;
        let (requests, fronted, ahead) = (u64::from(requests), u64::from(fronted), u64::from(ahead));
        let rows = requests + fronted * ahead;
        target.decoding().decode_step(rows)?;
        let pricing = Pricing { figures, lookahead: *lookahead };
        let launch = |census: &Census, streamed: bool| read_terms(census, &pricing, requests, rows, depth, &|experts| routes.read(experts, requests, fronted, ahead, streamed)).map(|terms| terms.total());
        Ok(match target {
            TargetCensus::Resident(census) => {
                let seconds = launch(census, false)?;
                FormSeconds { resident: seconds, streamed: seconds }
            }
            TargetCensus::Banked { resident, streamed } => FormSeconds { resident: launch(resident, false)?, streamed: launch(streamed, true)? },
        })
    }

    /// The load's launch of `rows` rows of `sequences` sequences at `depth`
    /// tokens in each form ([`launch_terms`]), kept once priced; a refusal
    /// is not kept. A two-row decode launch's sequences are its requests,
    /// fewer than its rows.
    pub fn target(&self, sequences: u32, rows: u32, depth: u32) -> Result<FormSeconds, PriceError> {
        let Source { target, figures, lookahead, priced, .. } = &*self.source;
        let mut priced = priced.lock().expect("a load's priced launches");
        if let Some(&seconds) = priced.get(&(sequences, rows, depth)) {
            return Ok(seconds);
        }
        let pricing = Pricing { figures, lookahead: *lookahead };
        let seconds = target.each(|census| launch_terms(census, &pricing, sequences.into(), rows.into(), depth))?;
        priced.insert((sequences, rows, depth), seconds);
        Ok(seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Measured `fresh` sums exceeding `total` experts saturate at `total`
    /// rather than raising a negative base to `sequences` and yielding
    /// negative or out-of-range `experts`.
    #[test]
    fn front_routes_read_saturates_when_fresh_exceeds_total() {
        let experts = Experts { experts: 16, selected: 4, slab_bytes: Vec::new(), slab_flops: Vec::new() };
        // selected + fresh = 4 + 40 = 44 > 2 * 16: unclamped `1 - 44/16 = -1.75`
        // would give `(-1.75)^2 > 1` and negative `experts` for `fronted = 2`.
        let oversaturated = FrontRoutes { fresh: vec![20., 20.], held: Some(1.) };
        for (requests, fronted) in [(1, 1), (2, 2), (4, 2), (3, 3)] {
            let read = oversaturated.read(&experts, requests, fronted, 2, true);
            assert!(read.experts.is_finite());
            assert_eq!(read.experts, 16.);
            assert_eq!(read.picks, ((requests + fronted * 2) * 4) as f64);
        }
        // Zero fronted sequences ignore the oversaturated front factor.
        let unfronted = oversaturated.read(&experts, 2, 0, 2, true);
        let expected = 16. * (1. - (1. - 4. / 16f64).powf(2.));
        assert!((unfronted.experts - expected).abs() < 1e-12);
    }
}
