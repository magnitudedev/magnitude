//! Graph traces: a graph builder run once per topology and evaluated at every
//! model's dimensions.
//!
//! What a family's graph template holds ([`NativeGraphOutline`]) is a function
//! of its topology (nodes, ports, edges, liveness) and of its buffers' sizes.
//! The sizes are the kernel contracts' shape expressions at the nodes'
//! dimensions, and the dimensions are arithmetic of a few numbers the family
//! reads from the model (widths, heads, rows, ...). A trace records the
//! builder once with those numbers as symbols ([`Dim`]) and answers every
//! later model whose numbers take the builder down the same path, by
//! evaluation alone. The builder stays the only owner of what a graph holds:
//! nothing here restates a size.
//!
//! # Converting a family
//!
//! A family's template is recorded by a recorder ([`GraphRecorder`]): a plain
//! `fn`, not a closure, so the only inputs it can read are the three it is
//! given.
//! - The **topology key** `K` (any `Clone + Eq + Hash`): everything that is
//!   not a model number and decides the graph's structure (backend, elements,
//!   block kinds, forms, flags). A trace answers only its own key, compared
//!   exactly, so a number may be part of the key too, at the cost of a trace
//!   per value.
//! - The **facts**, one [`Dim`] per model number, in an order the family
//!   fixes. A class variable the template's slices vary (rows, segments) is a
//!   fact as well, at the class the family records at, and is named a
//!   [`ClassFact`]: the slices then replace it per class exactly as
//!   [`NativeGraphResourceTemplate::outline`] does, and every branch the
//!   builder takes on it must go the same way at every class of the slices
//!   (Soundness).
//! - The **reals** ([`Reals`]): the model's real numbers (epsilons, scales,
//!   bases), the sources of its kernels' scalar arguments (Scalar
//!   arguments), and its tables (a rotary pair's frequencies, say), the
//!   contents of its graphs' constants ([`Table`]), each in an order the
//!   family fixes. A trace never compares them, so a real number or a
//!   table belongs here, never in the key.
//!
//! It returns the template and the builder's **output** beside it (its
//! binding constants, say): a [`TraceOutput`], kept as expressions over the
//! facts ([`Dim::expression`]) and the tables ([`Table::expression`]) and
//! evaluated at each model's, or `()`.
//!
//! The family asks its own `static` [`NativeGraphTraces`] for the outlines
//! ([`NativeGraphTraces::outline`]), and a load certifies the same recorder
//! through [`NativeGraphTraces::certify`]: both take a list of slice sets, so
//! one recording serves every class group of a template (one trace lookup
//! per set).
//!
//! In the recorder, dimensions are [`Dim`]s end to end: [`NativeGraphMetadata`]
//! takes them wherever it takes a `u64` (extents and named dimensions). A
//! `Dim` combines with `+ - * /`, [`Dim::ceil_div`], [`Dim::rem`],
//! [`Dim::min`], [`Dim::max`] and the `checked_*` forms (which answer `None`
//! exactly where the `u64` operation would). It has no `PartialEq` or
//! `PartialOrd` and no value accessor: a branch on a dimension is
//! [`Dim::less_than`], [`Dim::at_most`], [`Dim::greater_than`],
//! [`Dim::at_least`], [`Dim::equals`] or [`Dim::is_zero`], each of which
//! records its outcome as a guard, and [`Dim::pin`] reads the value itself
//! while recording that it is exactly this value (a count the topology
//! repeats, such as layers, makes the trace that value's). [`Dim::fixed`]
//! reads a dimension no recording made symbolic, for a prepared graph's
//! builder, and panics on a symbolic one.
//!
//! # Soundness
//!
//! A trace answers a model only where:
//! - its key equals the model's key;
//! - every guard holds at the model's facts and at every class point of the
//!   slices it answers (each comparison the builder made comes out as it did
//!   when recorded, at every class the template is certified for);
//! - every dimension evaluates (no arithmetic overflows where the recording's
//!   did not);
//! - every check the recording made of a shape holds again at the model's
//!   shapes: each node's call is admitted by its contract and its
//!   implementation's domain, each argument matches its parameter, each
//!   stored and shared result its port and primary
//!   (`NativeGraphResourceTemplate::admitted_at`).
//!
//! The checks split by what they depend on, and each runs where its inputs
//! change, no more often:
//! - the guards depend on the class facts, so they run per slice set, at its
//!   class points: arithmetic over the recorded expressions;
//! - the shape checks depend on the model's facts alone (a class varies only
//!   a capacity, never a recorded shape), so they run once per model and
//!   trace ([`Answer`]), whatever its slice sets; and each node's check is a
//!   pure function of its entry, backend and evaluated dimensions, so a
//!   process runs it once per distinct call (`checked_call`): a node whose
//!   dimensions no fact reaches is never checked again after its recording,
//!   and the layers and models that repeat a call share its one check;
//! - per slice set, only the capacities and the storage plan that follow
//!   from them are evaluated.
//!
//! Where any of these fails, the family's recorder runs again at the model's
//! facts and its trace joins the key's: the model gets what recording it
//! alone gives, error included. Where all hold, the evaluation is the
//! recording's at the model's shapes, so its outline equals the one a
//! recording at the model's own facts gives, exactly.
//!
//! A recording whose guards fail at a class point of the slices it is
//! certified for panics, for the load and for assessment alike: the family
//! certifies one template over classes its builder builds differently, and
//! must split those classes by its key or its slices.
//!
//! What a builder must not do: read a model number by any route other than
//! the facts (the `fn` recorder makes this structural), or keep a symbolic
//! `Dim` past its recording (comparing it then panics).
//!
//! Views (`WorkflowTensor::reshape`, `slice_leading`) take [`Dim`]s, and a
//! trace checks them again at each model's facts.
//!
//! # Scalar arguments
//!
//! A kernel's scalar arguments (an epsilon, a scale, a slab's row count) take
//! an [`Argument`]: a value the graph passes to its kernel and that nothing
//! else reads. No size, check or storage plan depends on one, so a trace
//! neither keys on its arguments nor guards them, and answers every model
//! whatever their values. A recorder makes them from
//! - its **reals** ([`Reals::argument`]);
//! - its dimensions ([`Dim::real`], [`Dim::word`]);
//! - constants (`Argument::from`);
//!
//! combined by `+ - * /` and narrowed to the kernel's word
//! ([`Argument::f32`]). An `Argument` has no accessor, comparison or
//! formatting of its value: a builder cannot branch on one. A graph a load
//! runs is built at its own model's reals, so each argument there is the
//! model's own.
//!
//! [`NativeGraphMetadata`]: crate::NativeGraphMetadata
//! [`NativeGraphResourceTemplate::outline`]: crate::NativeGraphResourceTemplate::outline

use crate::{
    NativeGraphClassSlice, NativeGraphLayout, NativeGraphMetadataError, NativeGraphOutline, NativeGraphResourceTemplate,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock};

pub use seismic_lang::size::DimensionExpr;
pub(crate) use seismic_lang::size::{DimensionExpr as Expr, Operation};

/// A comparison a recorded builder made and what it found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relation {
    Less,
    LessOrEqual,
    Equal,
}

impl Relation {
    fn holds(self, left: u64, right: u64) -> bool {
        match self {
            Self::Less => left < right,
            Self::LessOrEqual => left <= right,
            Self::Equal => left == right,
        }
    }
}

/// One branch a recorded builder took on its dimensions: `left relation
/// right` was `held` at the recording's facts.
#[derive(Clone, Debug)]
pub(crate) struct Guard {
    left: Expr,
    relation: Relation,
    right: Expr,
    held: bool,
    /// The facts its sides read, ascending: a class it is checked at varies
    /// only these.
    reads: Vec<usize>,
}

impl Guard {
    fn new(left: Expr, relation: Relation, right: Expr, held: bool) -> Self {
        let mut reads = Vec::new();
        fact_reads(&left, &mut reads);
        fact_reads(&right, &mut reads);
        reads.sort_unstable();
        reads.dedup();
        Self { left, relation, right, held, reads }
    }

    /// Whether the branch goes the same way at `facts`: both sides evaluate
    /// and compare as they did.
    fn holds(&self, facts: &[u64]) -> bool {
        match (self.left.evaluate(facts), self.right.evaluate(facts)) {
            (Some(left), Some(right)) => self.relation.holds(left, right) == self.held,
            _ => false,
        }
    }
}

/// The facts `expr` reads, appended to `reads`.
fn fact_reads(expr: &Expr, reads: &mut Vec<usize>) {
    match expr {
        Expr::Constant(_) => {}
        Expr::Fact(index) => reads.push(*index),
        Expr::Apply(application) => {
            let (_, left, right) = &**application;
            fact_reads(left, reads);
            fact_reads(right, reads);
        }
    }
}

/// The guard log of one recording, shared by every symbolic dimension made
/// from its facts. It takes guards only while the recording runs.
#[derive(Default)]
struct Recording {
    guards: RefCell<Vec<Guard>>,
    finished: Cell<bool>,
}

impl Recording {
    fn guard(&self, guard: Guard) {
        assert!(
            !self.finished.get(),
            "a dimension of a finished recording is compared: {:?} {:?} {:?}",
            guard.left,
            guard.relation,
            guard.right
        );
        self.guards.borrow_mut().push(guard);
    }
}

/// A symbolic dimension's expression and the recording it belongs to.
#[derive(Clone)]
struct Symbolic {
    expr: Expr,
    recording: Rc<Recording>,
}

/// A graph dimension: a number a builder computes with and passes to the
/// graph, which a recording ([`NativeGraphTraces`]) keeps as an expression
/// over its facts. Outside a recording every dimension is a constant
/// (`Dim::from`).
///
/// Its value is read only by a recorded comparison ([`Self::less_than`] and
/// its kin, [`Self::pin`]), by [`Self::fixed`] where no recording made it
/// symbolic, or by the graph that takes it, so a recorded builder cannot
/// branch on a value its trace does not check.
#[derive(Clone)]
pub struct Dim {
    /// Its value at the recording's facts (or its constant).
    value: u64,
    /// Its expression where a recording's facts reach it; a constant has
    /// none.
    symbolic: Option<Symbolic>,
}

impl From<u64> for Dim {
    fn from(value: u64) -> Self {
        Self { value, symbolic: None }
    }
}

impl fmt::Debug for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.symbolic {
            None => write!(f, "{}", self.value),
            Some(symbolic) => write!(f, "{} = {:?}", self.value, symbolic.expr),
        }
    }
}

impl Dim {
    /// Fact `index` of `recording`, at `value`.
    fn fact(index: usize, value: u64, recording: &Rc<Recording>) -> Self {
        Self {
            value,
            symbolic: Some(Symbolic { expr: Expr::Fact(index), recording: Rc::clone(recording) }),
        }
    }

    /// Its value at the recording's facts: what the graph it is passed to
    /// records at, never a builder's branch.
    pub(crate) fn recorded(&self) -> u64 {
        self.value
    }

    /// Its expression over the recording's facts: what a builder's output
    /// beside its template keeps of it ([`TraceOutput`]), evaluated only at
    /// a model's facts.
    pub fn expression(&self) -> DimensionExpr {
        match &self.symbolic {
            None => Expr::Constant(self.value),
            Some(symbolic) => symbolic.expr.clone(),
        }
    }

    /// The value of a dimension no recording made symbolic, as a prepared
    /// graph's builder passes it on.
    ///
    /// # Panics
    ///
    /// On a symbolic dimension: a recorded builder reads values only through
    /// guarded comparisons.
    pub fn fixed(&self) -> u64 {
        assert!(
            self.symbolic.is_none(),
            "a recorded builder reads the symbolic dimension {self:?} unguarded"
        );
        self.value
    }

    /// Its value, recording that the trace holds only where it is exactly
    /// this value: for a number the topology itself repeats or selects by.
    pub fn pin(&self) -> u64 {
        self.equals(self.value);
        self.value
    }

    /// It as a real argument ([`Argument`]): never compared, so a trace
    /// holds for it at any value.
    pub fn real(&self) -> Argument<f64> {
        Argument(self.value as f64)
    }

    /// It as an integer argument of word `T`, `None` where it exceeds `T`:
    /// that comparison is its only one, which the trace guards, so a model
    /// whose dimension fits records nothing new.
    pub fn word<T: ArgumentWord>(&self) -> Option<Argument<T>> {
        self.at_most(<T as sealed::Word>::MAX)
            .then(|| Argument(<T as sealed::Word>::of(self.value)))
    }

    /// The recording both operands belong to, if either is symbolic.
    fn recording(left: &Self, right: &Self) -> Option<Rc<Recording>> {
        match (&left.symbolic, &right.symbolic) {
            (None, None) => None,
            (Some(symbolic), None) | (None, Some(symbolic)) => Some(Rc::clone(&symbolic.recording)),
            (Some(left), Some(right)) => {
                assert!(
                    Rc::ptr_eq(&left.recording, &right.recording),
                    "dimensions of two recordings are combined"
                );
                Some(Rc::clone(&left.recording))
            }
        }
    }

    fn apply(&self, operation: Operation, right: &Self) -> Option<Self> {
        let value = operation.apply(self.value, right.value)?;
        let symbolic = Self::recording(self, right).map(|recording| Symbolic {
            expr: Expr::Apply(Arc::new((operation, self.expression(), right.expression()))),
            recording,
        });
        Some(Self { value, symbolic })
    }

    fn applied(&self, operation: Operation, right: &Self) -> Self {
        self.apply(operation, right).unwrap_or_else(|| {
            panic!(
                "dimension arithmetic overflows: {self:?} {} {right:?}",
                operation.symbol()
            )
        })
    }

    fn compare(&self, relation: Relation, right: &Self) -> bool {
        let held = relation.holds(self.value, right.value);
        if let Some(recording) = Self::recording(self, right) {
            recording.guard(Guard::new(self.expression(), relation, right.expression(), held));
        }
        held
    }

    pub fn checked_add<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.apply(Operation::Add, &Dim::from(right))
    }
    pub fn checked_sub<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.apply(Operation::Sub, &Dim::from(right))
    }
    pub fn checked_mul<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.apply(Operation::Mul, &Dim::from(right))
    }
    pub fn checked_div<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.apply(Operation::Div, &Dim::from(right))
    }
    /// `⌈self / right⌉`; panics where `right` is zero.
    pub fn ceil_div<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        self.applied(Operation::CeilDiv, &Dim::from(right))
    }
    /// `self % right`; panics where `right` is zero.
    pub fn rem<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        self.applied(Operation::Rem, &Dim::from(right))
    }
    pub fn min<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        self.applied(Operation::Min, &Dim::from(right))
    }
    pub fn max<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        self.applied(Operation::Max, &Dim::from(right))
    }

    /// `self < right`, recorded.
    pub fn less_than<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.compare(Relation::Less, &Dim::from(right))
    }
    /// `self <= right`, recorded.
    pub fn at_most<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.compare(Relation::LessOrEqual, &Dim::from(right))
    }
    /// `self > right`, recorded.
    pub fn greater_than<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        Dim::from(right).compare(Relation::Less, self)
    }
    /// `self >= right`, recorded.
    pub fn at_least<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        Dim::from(right).compare(Relation::LessOrEqual, self)
    }
    /// `self == right`, recorded.
    pub fn equals<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.compare(Relation::Equal, &Dim::from(right))
    }
    /// `self == 0`, recorded.
    pub fn is_zero(&self) -> bool {
        self.equals(0)
    }
}

macro_rules! dim_operator {
    ($trait:ident, $method:ident, $operation:ident) => {
        impl<R> std::ops::$trait<R> for Dim
        where
            Dim: From<R>,
        {
            type Output = Dim;
            fn $method(self, right: R) -> Dim {
                self.applied(Operation::$operation, &Dim::from(right))
            }
        }
        impl<R> std::ops::$trait<R> for &Dim
        where
            Dim: From<R>,
        {
            type Output = Dim;
            fn $method(self, right: R) -> Dim {
                self.applied(Operation::$operation, &Dim::from(right))
            }
        }
    };
}

dim_operator!(Add, add, Add);
dim_operator!(Sub, sub, Sub);
dim_operator!(Mul, mul, Mul);
dim_operator!(Div, div, Div);

impl From<&Dim> for Dim {
    fn from(dim: &Dim) -> Self {
        dim.clone()
    }
}

/// A family's template recorder: the template of topology `K` at `facts`
/// and `reals` (module docs), and what else its builder gives beside it
/// ([`TraceOutput`]; `()` where nothing). A `fn`, so its only inputs are its
/// key, its facts and its reals.
pub type GraphRecorder<K, X, E> = fn(&K, &[Dim], &Reals) -> Result<(NativeGraphResourceTemplate, X), E>;

/// A model's reals, which no trace compares (module docs): its real numbers,
/// each a kernel's scalar argument's source, and its tables, each a graph
/// constant's contents. A family fills them in the order its key indexes
/// them; a recorder reads them only as [`Argument`]s and [`Table`]s.
#[derive(Clone, Debug, Default)]
pub struct Reals {
    numbers: Vec<f64>,
    tables: Vec<Arc<[u8]>>,
}

impl Reals {
    pub fn new() -> Self {
        Self::default()
    }

    /// One real number; its index is what a key holds of it.
    pub fn number(&mut self, value: f64) -> usize {
        self.numbers.push(value);
        self.numbers.len() - 1
    }

    /// One table, as its constant's bytes; its index is what a key holds of
    /// it.
    pub fn table(&mut self, bytes: Arc<[u8]>) -> usize {
        self.tables.push(bytes);
        self.tables.len() - 1
    }

    /// Real number `index`, as a kernel's argument takes it.
    ///
    /// # Panics
    ///
    /// Where `index` is not one of these numbers': a key is read only at
    /// the reals its own split gave.
    pub fn argument(&self, index: usize) -> Argument<f64> {
        Argument(*self.numbers.get(index).unwrap_or_else(|| {
            panic!("real {index} of a key outside its {} reals", self.numbers.len())
        }))
    }

    /// Table `index`, as a recorder passes it to a constant: the trace keeps
    /// which table it is ([`Table::expression`]) and reads its bytes at
    /// each model's reals.
    ///
    /// # Panics
    ///
    /// Where `index` is not one of these tables'.
    pub fn table_fact(&self, index: usize) -> Table {
        let bytes = self.tables.get(index).unwrap_or_else(|| {
            panic!("table {index} of a key outside its {} tables", self.tables.len())
        });
        Table { bytes: Arc::clone(bytes), fact: Some(index) }
    }
}

/// A graph constant's contents from the model ([`Reals::table_fact`]) or a
/// constant (`From`): which a recording keeps as the table it is, never as
/// its bytes, so a trace answers a model with that model's own.
#[derive(Clone)]
pub struct Table {
    /// Its bytes at the recording's reals (or its constant's).
    bytes: Arc<[u8]>,
    /// Its index among the reals' tables, where a recording's reals gave it.
    fact: Option<usize>,
}

impl From<Arc<[u8]>> for Table {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self { bytes, fact: None }
    }
}

impl fmt::Debug for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.fact {
            Some(index) => write!(f, "Table({index})"),
            None => write!(f, "Table({} bytes)", self.bytes.len()),
        }
    }
}

impl Table {
    /// What a builder's output keeps of it ([`TraceOutput`]), evaluated
    /// only at a model's reals.
    pub fn expression(&self) -> TableExpr {
        match self.fact {
            Some(index) => TableExpr::Fact(index),
            None => TableExpr::Constant(Arc::clone(&self.bytes)),
        }
    }
}

/// A [`Table`] as a trace keeps it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableExpr {
    Fact(usize),
    Constant(Arc<[u8]>),
}

impl TableExpr {
    /// Its bytes at `reals`, `None` where they hold no such table.
    pub fn at(&self, reals: &Reals) -> Option<Arc<[u8]>> {
        match self {
            Self::Fact(index) => reals.tables.get(*index).cloned(),
            Self::Constant(bytes) => Some(Arc::clone(bytes)),
        }
    }
}

/// A kernel's scalar argument as a builder passes it to the graph: its value
/// at the model the graph is built for, which nothing but the graph reads
/// (module docs, Scalar arguments). Made from a constant (`From`), a
/// recorder's reals ([`Reals::argument`]), or a dimension ([`Dim::real`], [`Dim::word`]).
#[derive(Clone, Copy)]
pub struct Argument<T>(T);

impl<T> From<T> for Argument<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> fmt::Debug for Argument<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Argument")
    }
}

impl<T> Argument<T> {
    /// Its value, as the graph's argument encoding takes it.
    pub(crate) fn value(self) -> T {
        self.0
    }
}

impl Argument<f64> {
    /// It rounded to an `f32` argument.
    pub fn f32(self) -> Argument<f32> {
        Argument(self.0 as f32)
    }
}

macro_rules! argument_operator {
    ($trait:ident, $method:ident) => {
        impl std::ops::$trait for Argument<f64> {
            type Output = Self;
            fn $method(self, right: Self) -> Self {
                Self(std::ops::$trait::$method(self.0, right.0))
            }
        }
    };
}

argument_operator!(Add, add);
argument_operator!(Sub, sub);
argument_operator!(Mul, mul);
argument_operator!(Div, div);

/// An integer word a kernel's argument takes ([`Dim::word`]).
pub trait ArgumentWord: Copy + sealed::Word {}

mod sealed {
    pub trait Word {
        const MAX: u64;
        fn of(value: u64) -> Self;
    }
}

macro_rules! argument_word {
    ($word:ty) => {
        impl sealed::Word for $word {
            const MAX: u64 = <$word>::MAX as u64;
            fn of(value: u64) -> Self {
                <$word>::try_from(value).expect("a word argument is guarded within its word")
            }
        }
        impl ArgumentWord for $word {}
    };
}

argument_word!(i32);
argument_word!(u32);
argument_word!(u64);

/// What a builder gives beside its template (its graphs' binding constants,
/// say), kept as expressions over the recording's facts and evaluated at a
/// model's ([`NativeGraphTraces::outline`]). A family's type implements it
/// from [`DimensionExpr`]s ([`Dim::expression`]) and values no fact reaches.
pub trait TraceOutput: Send + Sync + 'static {
    /// Its value at a model, which every slice set answered there shares.
    type Value: Clone;
    /// It at `facts` and `reals`, asked only where the trace's guards hold
    /// and its template admits `facts`; `None` where a number does not
    /// evaluate there (the trace then does not answer them). Every other
    /// limit of building its value at `facts` is one the recording guarded
    /// or a port of its template admits, so failing one is an invariant
    /// violation, never a `None`.
    fn at(&self, facts: &[u64], reals: &Reals) -> Option<Self::Value>;
}

impl TraceOutput for () {
    type Value = ();
    fn at(&self, _facts: &[u64], _reals: &Reals) -> Option<()> {
        Some(())
    }
}

impl<T: TraceOutput> TraceOutput for Vec<T> {
    type Value = Vec<T::Value>;
    fn at(&self, facts: &[u64], reals: &Reals) -> Option<Self::Value> {
        self.iter().map(|output| output.at(facts, reals)).collect()
    }
}

/// One recorded run of a family's builder: its template and output, every
/// dimension an expression over the facts it was recorded at, and the
/// branches it took.
struct Trace<X> {
    template: NativeGraphResourceTemplate,
    output: X,
    guards: Vec<Guard>,
}

impl<X: TraceOutput> Trace<X> {
    fn record<K, E>(
        key: &K,
        facts: &[u64],
        reals: &Reals,
        record: GraphRecorder<K, X, E>,
    ) -> Result<Self, E> {
        let recording = Rc::new(Recording::default());
        let template = {
            let facts = facts
                .iter()
                .enumerate()
                .map(|(index, &value)| Dim::fact(index, value, &recording))
                .collect::<Vec<_>>();
            record(key, &facts, reals)
        };
        recording.finished.set(true);
        let (template, output) = template?;
        Ok(Self { template, output, guards: recording.guards.take() })
    }

    /// What it holds at `facts`, where its template admits them: every
    /// dimension evaluated and every shape check rerun there, once per model
    /// whatever its slice sets (module docs, Soundness). Its branches are
    /// checked per slice set, at the set's class points.
    /// Its output is evaluated only once the template admits `facts`
    /// ([`TraceOutput::at`]).
    fn at(&self, facts: &[u64], reals: &Reals) -> Option<Answer<X::Value>> {
        let points = self.template.points(facts)?;
        if !self.template.admitted_at(facts, &points) {
            return None;
        }
        let output = self.output.at(facts, reals)?;
        Some(Answer { points, output, programs: Default::default() })
    }
}

/// A trace at one model's facts ([`Trace::at`]): what every slice set it
/// answers there reads, and its template's buffers compiled at the model's
/// points, once per naming of the slices ([`crate::CapacityProgram`]).
struct Answer<V> {
    points: crate::TemplatePoints,
    output: V,
    /// The compiled programs, each with the class point (the model's facts
    /// with the class facts at one point's values) it was compiled at, or
    /// `None` at the model's own points.
    programs: std::cell::RefCell<Vec<(Option<Vec<u64>>, std::rc::Rc<crate::CapacityProgram>)>>,
}

impl<V> Answer<V> {
    /// `template`'s buffers compiled for slices named like `slice`, at the
    /// class `point` (its facts and their points, which fix the dimensions
    /// of `classes`) or, where the family names no class facts, at the
    /// model's own points, which fix none; compiled the first time a naming
    /// asks at a point.
    fn program(
        &self,
        template: &NativeGraphResourceTemplate,
        slice: &NativeGraphClassSlice,
        classes: &[ClassFact],
        point: Option<(&[u64], &crate::TemplatePoints)>,
    ) -> Result<std::rc::Rc<crate::CapacityProgram>, NativeGraphMetadataError> {
        let (names, facts) = (slice.names(), point.map(|(facts, _)| facts));
        if let Some((_, program)) = self
            .programs
            .borrow()
            .iter()
            .find(|(at, program)| at.as_deref() == facts && program.names() == names)
        {
            return Ok(std::rc::Rc::clone(program));
        }
        let at = point.map_or(&self.points, |(_, at)| at);
        let program = std::rc::Rc::new(template.capacity_program(at, slice, classes)?);
        self.programs.borrow_mut().push((facts.map(<[u64]>::to_vec), std::rc::Rc::clone(&program)));
        Ok(program)
    }

    /// Every buffer of `template` at its capacity over the classes of
    /// `slices`, whose class points are `points`. Where the family names class
    /// facts, each class point's buffers are its own: every dimension evaluated at the model's
    /// facts with the class facts at the point's values, so a buffer follows
    /// each dimension a class fact reaches whether or not the slices name it,
    /// and the capacity is the widest over the points. Without class facts,
    /// the slices' named dimensions vary it.
    fn capacities(
        &self,
        template: &NativeGraphResourceTemplate,
        points: &ClassPoints,
        slices: &[NativeGraphClassSlice],
    ) -> Result<crate::Capacities, NativeGraphMetadataError> {
        if points.slices.iter().all(Vec::is_empty) {
            let Some(first) = slices.first() else {
                return Err(NativeGraphMetadataError::Unsupported("graph regime has no admitted classes".into()));
            };
            return self.program(template, first, &[], None)?.at(slices);
        }
        let mut widest: Option<crate::Capacities> = None;
        let mut first = None;
        for (slice, facts) in points.points() {
            let at = template.points(&facts).ok_or_else(|| {
                NativeGraphMetadataError::Unsupported(format!("a dimension does not evaluate at class point {facts:?}"))
            })?;
            if let Some((entry, name)) = first.as_ref().and_then(|first| template.varied_static(first, &at)) {
                return Err(NativeGraphMetadataError::Unsupported(format!(
                    "graph regime classes vary `{entry}`'s static dimension `{name}`"
                )));
            }
            let slice = std::slice::from_ref(&slices[slice]);
            let capacities = self.program(template, &slice[0], &points.classes, Some((&facts, &at)))?.at(slice)?;
            widest = Some(match widest {
                None => capacities,
                Some(widest) => widest.cover(capacities),
            });
            first.get_or_insert(at);
        }
        widest.ok_or_else(|| NativeGraphMetadataError::Unsupported("graph regime has no admitted classes".into()))
    }

    /// `template`'s outline over `slices` at the model: the buffers' class
    /// points evaluated, and the storage planned from them.
    fn outline(
        &self,
        template: &NativeGraphResourceTemplate,
        points: &ClassPoints,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphOutline, NativeGraphMetadataError> {
        template.outline_capacities(self.capacities(template, points, slices)?)
    }

    /// The layout `template` seals into over `slices` at the model.
    fn certify(
        &self,
        template: &NativeGraphResourceTemplate,
        points: &ClassPoints,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphLayout, NativeGraphMetadataError> {
        template.certify_capacities(self.capacities(template, points, slices)?)
    }
}

/// A family's traces, by topology key, each with its builder's output `X`:
/// one per process, a `static` of the family's.
pub struct NativeGraphTraces<K, X = ()> {
    /// Each key's traces, behind the key's own lock: a key records under it,
    /// so concurrent misses of one key record once while other keys and
    /// every hit proceed in parallel ([`Self::outline`]).
    /// A trace is pushed only once its recording succeeded, so the traces
    /// are whole whatever panics while a lock is held: a poisoned lock is
    /// taken as it is, and a later model of the key records again.
    traces: OnceLock<RwLock<HashMap<K, Arc<Mutex<Vec<Arc<Trace<X>>>>>>>>,
}

impl<K, X> NativeGraphTraces<K, X> {
    pub const fn new() -> Self {
        Self { traces: OnceLock::new() }
    }
}

impl<K, X> Default for NativeGraphTraces<K, X> {
    fn default() -> Self {
        Self::new()
    }
}

/// A fact that is a class variable: the value the family's slices give the
/// class dimension `dimension` (of class scope or entry name `scope`, or
/// unscoped) replaces it per class, as `NativeGraphResourceTemplate`'s
/// capacities replace a node's dimension of that name. A builder's branch on
/// it must go the same way at every class a template is certified for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassFact {
    pub fact: usize,
    pub scope: Option<&'static str>,
    pub dimension: &'static str,
}

/// The classes of a slice set at a model: its `facts`, and for each slice
/// the values each class fact takes there (a slice fixes
/// correlated dimensions together; of class facts naming one fact, the last
/// gives its value, as it is the last written). A class point is `facts`
/// with each class fact at one of its slice's values, over every
/// combination within a slice; a guard is checked only over the class facts
/// it reads ([`Guard::reads`]), so a branch that reads one class varies only
/// that class, and the combinations of the classes it does not read, which
/// cannot change its outcome, are never formed.
struct ClassPoints {
    facts: Vec<u64>,
    /// The facts the slices replace per class.
    classes: Vec<ClassFact>,
    slices: Vec<Vec<(usize, Vec<u64>)>>,
}

impl ClassPoints {
    /// # Panics
    ///
    /// Where a slice gives a class fact no values: the family names a class
    /// dimension its slices do not vary.
    fn of(facts: &[u64], classes: &[ClassFact], slices: &[NativeGraphClassSlice]) -> Self {
        let slices = slices
            .iter()
            .map(|slice| {
                let mut varying: Vec<(usize, Vec<u64>)> = Vec::with_capacity(classes.len());
                for class in classes {
                    let values = slice.class_values(class.scope, class.dimension).unwrap_or_else(|| {
                        panic!("class fact {class:?} names a dimension its slices do not vary")
                    });
                    assert!(!values.is_empty(), "class fact {class:?} takes no values in a slice");
                    let values = values.to_vec();
                    match varying.iter_mut().find(|(fact, _)| *fact == class.fact) {
                        Some(entry) => entry.1 = values,
                        None => varying.push((class.fact, values)),
                    }
                }
                varying
            })
            .collect();
        Self { facts: facts.to_vec(), classes: classes.to_vec(), slices }
    }

    /// Every class point: for each slice (its index), the model's facts with
    /// the slice's class facts at each combination of their values.
    fn points(&self) -> Vec<(usize, Vec<u64>)> {
        let mut points = Vec::new();
        for (index, slice) in self.slices.iter().enumerate() {
            each_combination(&self.facts, &slice.iter().collect::<Vec<_>>(), |point| {
                points.push((index, point.to_vec()));
                None::<()>
            });
        }
        points
    }

    /// A point at which `guard` does not go the way it went, if any: the
    /// model's own facts, then each slice's combinations of the class facts
    /// the guard reads.
    fn failure(&self, guard: &Guard) -> Option<Vec<u64>> {
        if !guard.holds(&self.facts) {
            return Some(self.facts.clone());
        }
        self.slices.iter().find_map(|slice| {
            let read = slice
                .iter()
                .filter(|(fact, _)| guard.reads.binary_search(fact).is_ok())
                .collect::<Vec<_>>();
            if read.is_empty() {
                return None;
            }
            each_combination(&self.facts, &read, |point| (!guard.holds(point)).then(|| point.to_vec()))
        })
    }
}

/// `facts` with each of `varying`'s facts at every combination of its
/// values, passed to `each` until it finds something.
fn each_combination<T>(
    facts: &[u64],
    varying: &[&(usize, Vec<u64>)],
    mut each: impl FnMut(&[u64]) -> Option<T>,
) -> Option<T> {
    let mut point = facts.to_vec();
    let mut digits = vec![0; varying.len()];
    loop {
        for ((fact, values), &digit) in varying.iter().map(|entry| (entry.0, &entry.1)).zip(&digits) {
            point[fact] = values[digit];
        }
        if let Some(found) = each(&point) {
            return Some(found);
        }
        let advanced = digits.iter_mut().zip(varying).any(|(digit, (_, values))| {
            *digit += 1;
            if *digit < values.len() {
                return true;
            }
            *digit = 0;
            false
        });
        if !advanced {
            return None;
        }
    }
}

impl<X: TraceOutput> Trace<X> {
    /// Whether every branch its builder took goes the same way at every
    /// point of `points`.
    fn holds_at(&self, points: &ClassPoints) -> bool {
        first_failure(&self.guards, points).is_none()
    }

    /// A recording certified at the classes of `points`, each of whose
    /// branches must go the same way at all of them.
    ///
    /// # Panics
    ///
    /// Where one does not: the family certifies one template over classes
    /// its builder builds differently, so it must split them (by its key or
    /// its slices).
    fn assert_holds_at(&self, points: &ClassPoints) {
        assert_branches_hold(&self.guards, points);
    }
}

/// The outline over `slices` and output of the first of `known` whose
/// branches hold at `points` and whose template admits `facts`, if any.
#[allow(clippy::type_complexity)]
fn answer_from<X: TraceOutput>(
    known: &[(Arc<Trace<X>>, std::cell::OnceCell<Option<Answer<X::Value>>>)],
    facts: &[u64],
    reals: &Reals,
    points: &ClassPoints,
    slices: &[NativeGraphClassSlice],
) -> Option<Result<(NativeGraphOutline, X::Value), NativeGraphMetadataError>> {
    known.iter().find_map(|(trace, answer)| {
        if !trace.holds_at(points) {
            return None;
        }
        let answer = answer.get_or_init(|| trace.at(facts, reals)).as_ref()?;
        Some(answer.outline(&trace.template, points, slices).map(|outline| (outline, answer.output.clone())))
    })
}

/// The first of `guards` that fails at a class point of `points`, and the
/// point.
fn first_failure<'g>(guards: &'g [Guard], points: &ClassPoints) -> Option<(&'g Guard, Vec<u64>)> {
    guards.iter().find_map(|guard| points.failure(guard).map(|point| (guard, point)))
}

/// # Panics
///
/// Where a guard fails at a point of `points`: a template certified over
/// classes its builder branches between.
fn assert_branches_hold(guards: &[Guard], points: &ClassPoints) {
    if let Some((guard, point)) = first_failure(guards, points) {
        panic!("a template is certified over classes its builder branches between: {guard:?} fails at {point:?}");
    }
}

impl<K: Clone + Eq + Hash, X: TraceOutput> NativeGraphTraces<K, X> {
    /// What `record`'s template of `key` at `facts` (and `reals`, which no
    /// trace compares) holds over each slice set of `slice_sets`
    /// ([`NativeGraphResourceTemplate::outline`]), and its output at `facts`
    /// and `reals`: for each set, from a trace of `key` that answers `facts`
    /// at every class of the set, or from a new recording at `facts`, which
    /// then answers every model it holds for. `classes`
    /// names the facts the slices replace per class ([`ClassFact`]).
    #[allow(clippy::type_complexity)]
    pub fn outline<E: From<NativeGraphMetadataError>>(
        &self,
        key: &K,
        facts: &[u64],
        reals: &Reals,
        classes: &[ClassFact],
        slice_sets: &[Vec<NativeGraphClassSlice>],
        record: GraphRecorder<K, X, E>,
    ) -> Result<Vec<(NativeGraphOutline, X::Value)>, E> {
        let slot = self.slot(key);
        // Each trace of `key` at `facts`, evaluated the first time a slice
        // set asks it: a model's facts are checked once, its slice sets
        // only at their class points. `known` is always a prefix of the
        // key's traces, in their order.
        let mut known: Vec<(Arc<Trace<X>>, std::cell::OnceCell<Option<Answer<X::Value>>>)> = slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|trace| (Arc::clone(trace), std::cell::OnceCell::new()))
            .collect();
        let mut outlines = Vec::with_capacity(slice_sets.len());
        for slices in slice_sets {
            let points = ClassPoints::of(facts, classes, slices);
            if let Some(outline) = answer_from(&known, facts, reals, &points, slices) {
                outlines.push(outline.map_err(E::from)?);
                continue;
            }
            // A miss records under the key's lock, after the traces another
            // recording of the key pushed meanwhile are asked too: a key
            // records once however many models miss it at once.
            let mut traces = slot.lock().unwrap_or_else(PoisonError::into_inner);
            let asked = known.len();
            known.extend(traces[asked..].iter().map(|trace| (Arc::clone(trace), std::cell::OnceCell::new())));
            if let Some(outline) = answer_from(&known[asked..], facts, reals, &points, slices) {
                outlines.push(outline.map_err(E::from)?);
                continue;
            }
            let trace = Arc::new(Trace::record(key, facts, reals, record)?);
            trace.assert_holds_at(&points);
            let answer = trace.at(facts, reals).expect("a recording answers its own facts");
            outlines.push((answer.outline(&trace.template, &points, slices)?, answer.output.clone()));
            traces.push(Arc::clone(&trace));
            known.push((trace, std::cell::OnceCell::from(Some(answer))));
        }
        Ok(outlines)
    }

    /// The layouts a load seals `record`'s template of `key` at `facts`
    /// into, one for each slice set, and its output at `facts`: one
    /// recording, its branches checked at every class of every set as
    /// [`Self::outline`] checks them.
    pub fn certify<E: From<NativeGraphMetadataError>>(
        key: &K,
        facts: &[u64],
        reals: &Reals,
        classes: &[ClassFact],
        slice_sets: &[Vec<NativeGraphClassSlice>],
        record: GraphRecorder<K, X, E>,
    ) -> Result<(Vec<NativeGraphLayout>, X::Value), E> {
        let trace = Trace::record(key, facts, reals, record)?;
        for slices in slice_sets {
            trace.assert_holds_at(&ClassPoints::of(facts, classes, slices));
        }
        let points = trace
            .template
            .points(facts)
            .expect("a recording's dimensions evaluate at its own facts");
        let answer = Answer { points, output: (), programs: Default::default() };
        let layouts = slice_sets
            .iter()
            .map(|slices| answer.certify(&trace.template, &ClassPoints::of(facts, classes, slices), slices))
            .collect::<Result<Vec<_>, _>>()?;
        let output = trace.output.at(facts, reals).expect("a recording's output evaluates at its own facts");
        Ok((layouts, output))
    }

    /// How many traces each key holds: what recorded beside what answered.
    pub fn recorded(&self) -> usize {
        self.traces.get().map_or(0, |traces| {
            traces
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .values()
                .map(|slot| slot.lock().unwrap_or_else(PoisonError::into_inner).len())
                .sum()
        })
    }

    /// `key`'s traces and the lock it records under.
    fn slot(&self, key: &K) -> Arc<Mutex<Vec<Arc<Trace<X>>>>> {
        let traces = self.traces.get_or_init(Default::default);
        if let Some(slot) = traces.read().unwrap_or_else(PoisonError::into_inner).get(key) {
            return Arc::clone(slot);
        }
        Arc::clone(traces.write().unwrap_or_else(PoisonError::into_inner).entry(key.clone()).or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guard_holds_where_the_comparison_goes_the_same_way() {
        let recording = Rc::new(Recording::default());
        let width = Dim::fact(0, 64, &recording);
        let heads = Dim::fact(1, 4, &recording);
        let total = &width * &heads;
        assert!(total.at_most(1024));
        assert!(!width.less_than(&heads));
        recording.finished.set(true);
        let guards = recording.guards.take();
        assert_eq!(guards.len(), 2);
        assert_eq!(total.expression().evaluate(&[128, 8]), Some(1024));
        assert!(guards.iter().all(|guard| guard.holds(&[128, 8])));
        assert!(!guards.iter().all(|guard| guard.holds(&[128, 16])));
        assert!(!guards.iter().all(|guard| guard.holds(&[2, 4])));
    }

    #[test]
    fn constants_record_no_guard_and_overflow_does_not_evaluate() {
        let recording = Rc::new(Recording::default());
        let rows = Dim::fact(0, 3, &recording);
        assert!(Dim::from(2).less_than(3));
        assert_eq!(rows.checked_sub(4).map(|dim| dim.recorded()), None);
        let left = rows.checked_sub(1).unwrap();
        recording.finished.set(true);
        assert!(recording.guards.borrow().is_empty());
        assert_eq!(left.expression().evaluate(&[0]), None);
        assert_eq!(left.expression().evaluate(&[9]), Some(8));
    }

    /// A builder's branch on a class variable (chunked rows from 17 on)
    /// holds a trace only for slices whose every class takes it alike.
    fn chunked(rows: u64) -> Vec<Guard> {
        let recording = Rc::new(Recording::default());
        let width = Dim::fact(0, 64, &recording);
        let rows = Dim::fact(1, rows, &recording);
        let _ = rows.at_least(17);
        let _ = width.at_most(128);
        recording.finished.set(true);
        recording.guards.take()
    }

    fn rows(values: &[u64]) -> Vec<NativeGraphClassSlice> {
        vec![NativeGraphClassSlice::new().dimension("rows", values.to_vec())]
    }

    const ROWS: [ClassFact; 1] = [ClassFact { fact: 1, scope: None, dimension: "rows" }];

    #[test]
    fn class_guards_hold_at_every_class_of_the_slices() {
        let step = chunked(1);
        let points = ClassPoints::of(&[64, 1], &ROWS, &rows(&[1, 2, 4, 8, 16]));
        assert_eq!(points.slices, vec![vec![(1, vec![1, 2, 4, 8, 16])]]);
        assert!(first_failure(&step, &points).is_none());
        // Every class of a later model's slices on the same side answers.
        assert!(first_failure(&step, &ClassPoints::of(&[96, 2], &ROWS, &rows(&[2, 16]))).is_none());
        // A slice set reaching 17 rows does not, nor a model past the width.
        assert!(first_failure(&step, &ClassPoints::of(&[64, 1], &ROWS, &rows(&[1, 16, 32]))).is_some());
        assert!(first_failure(&step, &ClassPoints::of(&[256, 1], &ROWS, &rows(&[1]))).is_some());
        let chunks = chunked(32);
        assert!(first_failure(&chunks, &ClassPoints::of(&[64, 32], &ROWS, &rows(&[17, 512]))).is_none());
    }

    /// A guard varies only the classes it reads: a slice's other classes,
    /// however many values they take, are never combined with them, and a
    /// class value listed twice is one class.
    #[test]
    fn a_guard_is_checked_over_the_classes_it_reads_alone() {
        let recording = Rc::new(Recording::default());
        let width = Dim::fact(0, 64, &recording);
        let segments = Dim::fact(2, 4, &recording);
        let slots = Dim::fact(3, 2, &recording);
        let _ = segments.at_most(width.clone());
        let _ = (segments.clone() * slots).at_most(width);
        recording.finished.set(true);
        let guards = recording.guards.take();
        assert_eq!([guards[0].reads.as_slice(), guards[1].reads.as_slice()], [&[0, 2][..], &[0, 2, 3][..]]);
        let classes = [
            ClassFact { fact: 1, scope: None, dimension: "rows" },
            ClassFact { fact: 2, scope: None, dimension: "segments" },
            ClassFact { fact: 3, scope: None, dimension: "slots" },
        ];
        let many = (1..=4096).collect::<Vec<_>>();
        let slice = |segments: &[u64], slots: &[u64]| {
            NativeGraphClassSlice::new()
                .dimension("rows", many.clone())
                .dimension("segments", segments.to_vec())
                .dimension("slots", slots.to_vec())
        };
        // Segments listed once per slot count: each distinct value once.
        let points = ClassPoints::of(&[64, 1, 4, 2], &classes, &[slice(&[1, 4, 1, 4], &[1, 2, 1, 2])]);
        assert_eq!(points.slices[0][1..], [(2, vec![1, 4]), (3, vec![1, 2])]);
        assert!(first_failure(&guards, &points).is_none());
        // Within a slice, the classes a guard reads are combined: 64
        // segments with 2 slots exceed the width.
        let points = ClassPoints::of(&[64, 1, 4, 2], &classes, &[slice(&[1, 64], &[1, 2])]);
        let (failed, point) = first_failure(&guards, &points).expect("64 segments at 2 slots exceed 64");
        assert_eq!((failed.reads.as_slice(), &point[2..]), (&[0, 2, 3][..], &[64, 2][..]));
    }

    #[test]
    #[should_panic(expected = "branches between")]
    fn a_template_over_classes_its_builder_branches_between_is_refused() {
        assert_branches_hold(&chunked(1), &ClassPoints::of(&[64, 1], &ROWS, &rows(&[1, 16, 32])));
    }

    #[test]
    #[should_panic(expected = "unguarded")]
    fn a_symbolic_dimension_has_no_fixed_value() {
        let recording = Rc::new(Recording::default());
        Dim::fact(0, 3, &recording).fixed();
    }

    #[test]
    #[should_panic(expected = "finished recording")]
    fn a_dimension_outliving_its_recording_cannot_branch() {
        let recording = Rc::new(Recording::default());
        let rows = Dim::fact(0, 3, &recording);
        recording.finished.set(true);
        rows.is_zero();
    }
}
