//! The census: what one launch's calls read and compute, kind by kind
//! ([`Census`]). A kind is one call's work ([`Work`]) and its native
//! launches at each decode row count ([`LaunchCounts`]); a launch makes its
//! kinds' calls, each kind as many times as its
//! multiplicity. The census is the cost model's input contract: producing
//! one (the executor's decode demand, from a load's plans and its
//! backend's declarations) is not the cost model's.

use seismic_measure::pricing::{Format, Layout, PriceError, Work};
use std::collections::BTreeMap;

/// The native launches one call of a kind dispatches at each decode row
/// count, one to the decode row classes' most ([`Self::decode_rows`]): a
/// launch of up to that many rows runs its decode forms, beyond them its
/// prefill forms, each call one launch. At a count its entry's prepared
/// specializations dispatch no one count of, why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchCounts(Vec<Result<u64, String>>);

impl LaunchCounts {
    /// The counts at one row up to the decode row classes' most, in order.
    ///
    /// # Panics
    /// Where `counts` is empty: a decode launch has at least one row.
    pub fn new(counts: Vec<Result<u64, String>>) -> Self {
        assert!(!counts.is_empty(), "a kind's launches are counted at one row at least");
        Self(counts)
    }

    /// The most rows a decode launch has.
    pub fn decode_rows(&self) -> u64 {
        self.0.len() as u64
    }

    /// The launches at `rows` rows, a decode row count.
    pub fn at(&self, rows: u64) -> Result<u64, PriceError> {
        assert!((1..=self.decode_rows()).contains(&rows), "invariant violated: {rows} rows is no decode row count");
        self.0[rows as usize - 1].clone().map_err(PriceError::Launches)
    }
}

/// One kind of call: what it reads and computes, and its launches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kind {
    pub work: Work,
    pub launches: LaunchCounts,
}

/// A streamed routed layer's routed kind: one numbering a load's
/// ([`Self::number`]), by which its routed passes reach their price
/// (`ExpertPass`) and its prompts their grouping (`PassLayers`). Layers of a
/// kind have alike routed calls in every form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoutedKind(pub u32);

impl RoutedKind {
    /// Each layer's kind, of layers keyed by `keys` (what makes two layers'
    /// routed calls alike), numbered in order of first appearance.
    pub fn number<K: PartialEq>(keys: impl IntoIterator<Item = K>) -> Vec<Self> {
        let mut seen = Vec::<K>::new();
        keys.into_iter()
            .map(|key| {
                let at = seen.iter().position(|known| *known == key).unwrap_or_else(|| {
                    seen.push(key);
                    seen.len() - 1
                });
                Self(u32::try_from(at).expect("a load's routed kinds fit u32"))
            })
            .collect()
    }
}

/// A call that computes a streamed layer's routed picks (its expansion or
/// its down projection): the layer's routed kind, and the layer, in decoder
/// order among the streamed layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedCall {
    pub kind: RoutedKind,
    pub layer: usize,
}

/// A call of a streamed layer that the layer's gates price apart from the
/// launch: one computing its routed picks ([`RoutedCall`]), or its hint
/// launch (the routers of the layers it hints read ahead of their routes),
/// of the layer, in decoder order among the streamed layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerCall {
    Routed(RoutedCall),
    Hint(usize),
}

/// What one launch's calls read and compute: its distinct kinds, each with
/// its multiplicity (the calls of it the launch makes), and the submissions
/// its calls end beyond its own (a streamed layer's parks and publications
/// where the backend's parks are submission boundaries), each a submit and
/// a wait beside device time. Every kind counts its launches at the same
/// decode row classes. Each routed kind of its streamed layers maps to the
/// kinds of a layer's routed calls, in call order (`routed`), and each
/// streamed layer that hints to the kind of its hint launch (`hints`).
#[derive(Clone, Debug, PartialEq)]
pub struct Census {
    kinds: Vec<Kind>,
    multiplicity: Vec<u64>,
    boundaries: u64,
    decode_rows: u64,
    routed: BTreeMap<RoutedKind, Vec<usize>>,
    hints: BTreeMap<usize, usize>,
}

impl Census {
    /// The census of a launch making `calls`, each where it is a streamed
    /// layer's call its gates price that layer's ([`LayerCall`]), and
    /// ending `boundaries` submissions beyond its own, counting launches at
    /// `decode_rows` decode row classes: equal calls are one kind.
    ///
    /// # Panics
    /// Where a call counts its launches at other decode row classes, two
    /// layers of one routed kind make other routed calls, or a layer makes
    /// two hint launches.
    pub fn new(calls: impl IntoIterator<Item = (Kind, Option<LayerCall>)>, boundaries: u64, decode_rows: u64) -> Self {
        let (mut kinds, mut multiplicity) = (Vec::<Kind>::new(), Vec::<u64>::new());
        let mut layers = BTreeMap::<usize, (RoutedKind, Vec<usize>)>::new();
        let mut hints = BTreeMap::<usize, usize>::new();
        for (call, of) in calls {
            assert_eq!(call.launches.decode_rows(), decode_rows, "invariant violated: a launch's calls count their launches at one set of decode rows");
            let kind = match kinds.iter().position(|kind| *kind == call) {
                Some(kind) => kind,
                None => {
                    kinds.push(call);
                    multiplicity.push(0);
                    kinds.len() - 1
                }
            };
            multiplicity[kind] += 1;
            match of {
                Some(LayerCall::Routed(RoutedCall { kind: routed, layer })) => {
                    let (of, calls) = layers.entry(layer).or_insert_with(|| (routed, Vec::new()));
                    assert_eq!(*of, routed, "invariant violated: streamed layer {layer}'s routed calls are of one kind");
                    calls.push(kind);
                }
                Some(LayerCall::Hint(layer)) => {
                    assert!(hints.insert(layer, kind).is_none(), "invariant violated: streamed layer {layer} makes one hint launch");
                }
                None => {}
            }
        }
        let mut routed = BTreeMap::<RoutedKind, Vec<usize>>::new();
        for (layer, (kind, calls)) in layers {
            let known = routed.entry(kind).or_insert_with(|| calls.clone());
            assert_eq!(*known, calls, "invariant violated: streamed layer {layer} of {kind:?} makes other routed calls than its kind's");
        }
        Self { kinds, multiplicity, boundaries, decode_rows, routed, hints }
    }

    /// The kind of streamed layer `layer`'s hint launch, none where it
    /// hints no layer.
    pub(crate) fn hint(&self, layer: usize) -> Option<usize> {
        self.hints.get(&layer).copied()
    }

    /// The kinds of the routed calls of a streamed layer of `kind`, in call
    /// order.
    ///
    /// # Panics
    /// Where `kind` is none of the census's streamed layers'.
    pub(crate) fn routed(&self, kind: RoutedKind) -> &[usize] {
        self.routed.get(&kind).unwrap_or_else(|| panic!("invariant violated: {kind:?} is none of the census's routed kinds"))
    }

    /// The launch's distinct kinds.
    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }

    /// The most rows a decode launch of the census has: beyond, its prefill
    /// forms run.
    pub fn decode_rows(&self) -> u64 {
        self.decode_rows
    }

    /// Whether `rows` rows run the decode forms: one to the decode rows. The
    /// one comparison against them.
    pub(crate) fn decodes(&self, rows: u64) -> bool {
        (1..=self.decode_rows).contains(&rows)
    }

    /// Whether `sequences` sequences, a row each, make one decode step: only
    /// where their rows run the decode forms ([`Self::decodes`]).
    pub(crate) fn decode_step(&self, sequences: u64) -> Result<(), PriceError> {
        match self.decodes(sequences) {
            true => Ok(()),
            false => Err(PriceError::DecodeRows { rows: sequences, decode_rows: self.decode_rows }),
        }
    }

    /// The calls of kind `kind` the launch makes.
    pub fn multiplicity(&self, kind: usize) -> u64 {
        self.multiplicity[kind]
    }

    /// The submissions the launch's calls end beyond its own.
    pub fn boundaries(&self) -> u64 {
        self.boundaries
    }

    /// The weight formats its kinds read and compute in: the formats its
    /// rates price beside the activation format.
    pub fn formats(&self) -> Vec<Format> {
        distinct(self.kinds.iter().flat_map(|kind| kind.work.formats()))
    }

    /// The history layouts its kinds' attention entries read.
    pub fn layouts(&self) -> Vec<Layout> {
        distinct(self.kinds.iter().filter_map(|kind| kind.work.layout()))
    }
}

/// `items` each once, in first-seen order.
fn distinct<T: PartialEq>(items: impl Iterator<Item = T>) -> Vec<T> {
    let mut seen = Vec::new();
    for item in items {
        if !seen.contains(&item) {
            seen.push(item);
        }
    }
    seen
}
