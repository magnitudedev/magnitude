//! A family's model numbers as the facts of its graph traces
//! (`seismic::trace`).
//!
//! A recorder (`seismic::GraphRecorder`) reads only its topology key and its
//! facts. Every builder input that comes from the model is therefore split
//! once, outside the recorder, into
//! - its **numbers**: a struct generic over its number type `X<D>` (widths,
//!   heads, rows, extents...), which implements [`Numbers`]; and
//! - its **topology**: everything else it branches on (variants, flags,
//!   elements, kernel forms), a `Clone + Eq + Hash` value with no model
//!   number in it. A number's presence (an `Option`, a `Vec`'s length) is
//!   topology.
//!
//! The family's key holds each topology and each numbers struct with its
//! numbers replaced by their [`FactIndex`]es ([`FactSink::index`]), so the
//! key compares exactly what decides the graph and no model number. The
//! recorder rebuilds each numbers struct over its facts ([`Facts::dims`]) and
//! the builder takes `&X<Dim>`, never the model's own structs: a builder
//! cannot read a number its trace does not hold as a fact.
//!
//! A real model number (an epsilon, a scale, a rotary base) is neither: it
//! only ever reaches a kernel's scalar argument (`seismic::Argument`), which
//! no size depends on. It is a **real** ([`FactSink::real`]), which the key
//! holds as its [`RealIndex`] and the recorder reads as an argument
//! ([`Facts::real`]). A table of them (a rotary pair's frequencies) only
//! ever reaches a graph constant's contents: it is a **table**
//! ([`FactSink::table`]), held as its [`TableIndex`] and read as a
//! `seismic::Table` ([`Facts::table`]). No trace compares reals or tables,
//! so one recording serves every model whatever they are.
//!
//! The load builds through the same recorder at constant facts
//! (`Certified::traced_each`), so the split is the one route from a model to
//! its graphs.

use seismic::{Argument, Dim, Reals, Table};
use std::sync::Arc;

/// A model number's place among a recording's facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct FactIndex(usize);

impl FactIndex {
    /// Its position among its recording's facts: what a recorder's facts
    /// slice is indexed by.
    pub(crate) fn position(self) -> usize {
        self.0
    }
}

/// A real model number's place among a recording's reals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RealIndex(usize);

/// A table's place among a recording's tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TableIndex(usize);

/// A struct of model numbers generic over its number type: `u64` as the
/// model gives them, [`FactIndex`] in a topology key, [`Dim`] in a recorder.
/// `map` visits every number in one fixed order, the struct's own, which is
/// the only place that order is written.
pub(crate) trait Numbers<D> {
    type Of<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> Self::Of<E>;
}

/// The facts a model's numbers are split into, in the order they are
/// indexed.
#[derive(Clone, Debug, Default)]
pub(crate) struct FactSink {
    values: Vec<u64>,
    reals: Reals,
}

impl FactSink {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// `numbers` as the key holds them: each number a fact, in `map`'s order.
    pub(crate) fn index<N: Numbers<u64>>(&mut self, numbers: N) -> N::Of<FactIndex> {
        numbers.map(&mut |value| self.fact(value))
    }

    /// One number as a fact.
    pub(crate) fn fact(&mut self, value: u64) -> FactIndex {
        self.values.push(value);
        FactIndex(self.values.len() - 1)
    }

    /// `reals` (a struct of real model numbers, [`Numbers`] over `f64`) as
    /// the key holds them: each a real, in `map`'s order.
    pub(crate) fn index_reals<N: Numbers<f64>>(&mut self, reals: N) -> N::Of<RealIndex> {
        reals.map(&mut |value| self.real(value))
    }

    /// One real model number, the source of a kernel's scalar argument.
    pub(crate) fn real(&mut self, value: f64) -> RealIndex {
        RealIndex(self.reals.number(value))
    }

    /// One table, as the bytes of the constant it fills.
    pub(crate) fn table(&mut self, bytes: impl Into<Arc<[u8]>>) -> TableIndex {
        TableIndex(self.reals.table(bytes.into()))
    }

    /// The facts, in index order: what the family's traces are asked at.
    pub(crate) fn values(&self) -> &[u64] {
        &self.values
    }

    /// The reals and tables, in index order: what the family's recordings
    /// pass to their kernels and constants, never compared.
    pub(crate) fn reals(&self) -> &Reals {
        &self.reals
    }
}

/// A recorder's facts, which rebuild the numbers its key indexes.
#[derive(Clone, Copy)]
pub(crate) struct Facts<'f> {
    facts: &'f [Dim],
    reals: &'f Reals,
}

impl<'f> Facts<'f> {
    pub(crate) fn new(facts: &'f [Dim], reals: &'f Reals) -> Self {
        Self { facts, reals }
    }

    /// The reals `indexed` (a key's) as kernels' arguments take them.
    pub(crate) fn arguments<N: Numbers<RealIndex>>(&self, indexed: N) -> N::Of<Argument<f64>> {
        indexed.map(&mut |index| self.real(index))
    }

    /// One real, as a kernel's argument takes it.
    pub(crate) fn real(&self, RealIndex(index): RealIndex) -> Argument<f64> {
        self.reals.argument(index)
    }

    /// One table, as a constant takes it (`GraphConstant::table`).
    pub(crate) fn table(&self, TableIndex(index): TableIndex) -> Table {
        self.reals.table_fact(index)
    }

    /// The numbers `indexed` (a key's) at these facts.
    pub(crate) fn dims<N: Numbers<FactIndex>>(&self, indexed: N) -> N::Of<Dim> {
        indexed.map(&mut |index| self.dim(index))
    }

    /// One fact.
    ///
    /// # Panics
    ///
    /// Where `index` is not one of these facts': a key is evaluated only at
    /// the facts its own split gave.
    pub(crate) fn dim(&self, FactIndex(index): FactIndex) -> Dim {
        self.facts
            .get(index)
            .unwrap_or_else(|| panic!("fact {index} of a key outside its {} facts", self.facts.len()))
            .clone()
    }
}

/// Constant numbers: the model's own, as the load's builders outside any
/// recording take them.
pub(crate) fn constant<N: Numbers<u64>>(numbers: N) -> N::Of<Dim> {
    numbers.map(&mut Dim::from)
}

/// Constant reals: the model's own, as the load's builders outside any
/// recording pass them.
pub(crate) fn constant_reals<N: Numbers<f64>>(reals: N) -> N::Of<Argument<f64>> {
    reals.map(&mut Argument::from)
}

impl<D> Numbers<D> for Vec<D> {
    type Of<E> = Vec<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> Vec<E> {
        self.into_iter().map(f).collect()
    }
}

impl<D> Numbers<D> for Option<D> {
    type Of<E> = Option<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> Option<E> {
        self.map(f)
    }
}

impl<D, const N: usize> Numbers<D> for [D; N] {
    type Of<E> = [E; N];
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> [E; N] {
        self.map(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_rebuilds_its_numbers_in_order() {
        let mut sink = FactSink::new();
        let indexed = sink.index(vec![7u64, 3, 9]);
        let pair = sink.index([1u64, 2]);
        assert_eq!(sink.values(), &[7, 3, 9, 1, 2]);
        let facts = sink.values().iter().map(|&value| Dim::from(value)).collect::<Vec<_>>();
        let reals = Reals::new();
        let facts = Facts::new(&facts, &reals);
        let rebuilt = facts.dims(indexed).into_iter().map(|dim| dim.fixed()).collect::<Vec<_>>();
        assert_eq!(rebuilt, [7, 3, 9]);
        assert_eq!(facts.dims(pair).map(|dim| dim.fixed()), [1, 2]);
    }
}
