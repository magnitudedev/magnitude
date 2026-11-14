//! Host constants of sealed graphs. A constant is declared as an external
//! port while its graph is built. Binding uploads each distinct value once and
//! attaches it with the weights as a static binding, so no run writes it.

use super::graph::{GraphError, draft::GraphDraft};
use seismic::{
    Device, Dim, DimensionExpr, Element, GraphCensus, HostHeap, LaunchGraphs, LaunchedGraph, NativeGraphStorageBytes,
    NativePort, Reals, Table, TableExpr, Tensor, TraceOutput,
};
use std::sync::{Arc, Mutex};

/// A constant of one graph: the port it binds, its value, and how that
/// value follows the graph's class rows. The value is shared: the graphs of
/// a family that bind one value refer to the family's one copy of it
/// (`FamilyConstants`).
#[derive(Clone)]
pub(crate) struct GraphConstant {
    port: Option<NativePort>,
    source: ConstantSource,
}

/// What a constant's value is: known, or, inside a graph trace's recording
/// where a fact reaches its numbers, expressions over the facts that the
/// trace evaluates at each model's ([`ConstantExpr`]).
#[derive(Clone)]
enum ConstantSource {
    Known {
        value: Arc<ConstantStorage>,
        /// The rule its value is made from at its graph's class rows, of a
        /// constant whose extents or values follow them; none of one whose
        /// value is the same at every class.
        rows: Option<RowRule>,
    },
    Recorded(ConstantExpr),
}

/// A constant as a graph trace keeps it beside its template: what a family
/// recorder returns of its builder's constants
/// ([`GraphConstant::recorded`]), each the constant of a model's facts there
/// (`seismic::TraceOutput`).
#[derive(Clone)]
pub(crate) struct RecordedConstant(ConstantSource);

/// A constant whose numbers a recording's facts reach, as expressions over
/// them.
#[derive(Clone)]
enum ConstantExpr {
    /// `rule` at `count`, following the class rows where `follows`.
    Rule { rule: RowRule<DimensionExpr>, count: DimensionExpr, follows: bool },
    /// An empty tensor of `element` and `extents`, one of them zero.
    Empty { element: Element, extents: Vec<DimensionExpr> },
    /// The `f32` tensor of `extents` holding `value` everywhere.
    Filled { extents: Vec<DimensionExpr>, value: f32 },
    /// The tensor of `element` and `extents` holding a model's table.
    Table { element: Element, extents: Vec<DimensionExpr>, table: TableExpr },
}

impl ConstantExpr {
    /// Its value at `facts` and `reals`, `None` where a number does not
    /// evaluate there.
    ///
    /// # Panics
    ///
    /// Where its value cannot be built at numbers that evaluate: every such
    /// limit is a guard of its recording ([`RowRule::count_limit`]) or of
    /// its port's extents, which the trace's template admits before asking
    /// (`seismic::TraceOutput::at`).
    fn at(&self, facts: &[u64], reals: &Reals) -> Option<ConstantSource> {
        let evaluated = |extents: &[DimensionExpr]| extents.iter().map(|extent| extent.evaluate(facts)).collect::<Option<Vec<_>>>();
        let admitted = "a trace admits its constants' values (`TraceOutput::at`)";
        Some(match self {
            Self::Rule { rule, count, follows } => {
                let rule = rule.try_map(|width| width.evaluate(facts))?;
                let value = rule.storage(count.evaluate(facts)?).unwrap_or_else(|error| panic!("{admitted}: {error}"));
                ConstantSource::Known { value: Arc::new(value), rows: follows.then_some(rule) }
            }
            Self::Empty { element, extents } => ConstantSource::Known {
                value: Arc::new(ConstantStorage {
                    element: *element,
                    extents: evaluated(extents)?.into(),
                    value: ConstantValue::Bytes(Arc::from([])),
                }),
                rows: None,
            },
            Self::Filled { extents, value } => {
                let value = filled_f32(evaluated(extents)?, *value).unwrap_or_else(|error| panic!("{admitted}: {error}"));
                ConstantSource::Known { value: Arc::new(value), rows: None }
            }
            Self::Table { element, extents, table } => {
                let bytes = table.at(reals).expect("a trace is evaluated at reals of its own split");
                ConstantSource::Known { value: Arc::new(table_storage(*element, evaluated(extents)?, bytes)), rows: None }
            }
        })
    }
}

/// The `f32` tensor of `extents` holding `value` everywhere.
fn filled_f32(extents: Vec<u64>, value: f32) -> Result<ConstantStorage, String> {
    let count = extents
        .iter()
        .try_fold(1u64, |count, &extent| count.checked_mul(extent))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or("filled constant length overflows")?;
    Ok(ConstantStorage {
        element: Element::f32(),
        extents: extents.into(),
        value: ConstantValue::Bytes(std::iter::repeat_n(value.to_le_bytes(), count).flatten().collect()),
    })
}

/// The tensor of `element` and `extents` holding `bytes`.
///
/// # Panics
///
/// Where `bytes` do not fill exactly those extents: a family's table and
/// the extents it states for it come from one split of one model.
fn table_storage(element: Element, extents: Vec<u64>, bytes: Arc<[u8]>) -> ConstantStorage {
    let length = element
        .canonical_byte_len(&extents)
        .unwrap_or_else(|error| panic!("a table constant's extents {extents:?} have no length: {error}"));
    assert_eq!(
        length,
        bytes.len() as u64,
        "a table of {} bytes fills a constant of {extents:?}",
        bytes.len()
    );
    ConstantStorage { element, extents: extents.into(), value: ConstantValue::Bytes(bytes) }
}

/// The value of a constant expression: a builder's numbers that no fact
/// reached.
fn constant_of(expression: &DimensionExpr) -> Option<u64> {
    match expression {
        DimensionExpr::Constant(value) => Some(*value),
        _ => None,
    }
}

impl TraceOutput for RecordedConstant {
    type Value = GraphConstant;
    fn at(&self, facts: &[u64], reals: &Reals) -> Option<GraphConstant> {
        let source = match &self.0 {
            known @ ConstantSource::Known { .. } => known.clone(),
            ConstantSource::Recorded(expression) => expression.at(facts, reals)?,
        };
        Some(GraphConstant { port: None, source })
    }
}

/// How a constant's extents and values follow its graph's class rows: the
/// one description its builder declares it from and a plan counts it from
/// at each class a template certifies (`GraphConstant::at_rows`). Its
/// numbers `N` are `u64`s, a builder's [`Dim`]s, or a trace's
/// [`DimensionExpr`]s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowRule<N = u64> {
    /// The rank-1 `i32` row table `0, 1, …, rows - 1`.
    Identity,
    /// The zero matrix `[rows, width]`.
    Zeros { element: Element, width: N },
    /// The empty tensor `[0, rows, width]`.
    Empty { element: Element, width: N },
    /// The rank-1 `i32` table of `rows` copies of one value.
    Filled(i32),
    /// The F32 matrix `[rows, width]` of ones in column 0 and zeros
    /// elsewhere: each row one-hot on its first column.
    OneHot { width: N },
    /// The one `i32` `rows` itself.
    Count,
}

impl<N> RowRule<N> {
    fn element(&self) -> Element {
        match self {
            Self::Identity | Self::Filled(_) | Self::Count => Element::i32(),
            Self::Zeros { element, .. } | Self::Empty { element, .. } => *element,
            Self::OneHot { .. } => Element::f32(),
        }
    }

    /// It with each number `f` of it, `None` where one is `None`.
    fn try_map<M>(&self, f: impl Fn(&N) -> Option<M>) -> Option<RowRule<M>> {
        Some(match self {
            Self::Identity => RowRule::Identity,
            Self::Zeros { element, width } => RowRule::Zeros { element: *element, width: f(width)? },
            Self::Empty { element, width } => RowRule::Empty { element: *element, width: f(width)? },
            Self::Filled(value) => RowRule::Filled(*value),
            Self::OneHot { width } => RowRule::OneHot { width: f(width)? },
            Self::Count => RowRule::Count,
        })
    }

    /// The largest row count its value holds, where its port's extents do
    /// not already bound it: an `i32` row index or row count.
    fn count_limit(&self) -> Option<(u64, &'static str)> {
        match self {
            Self::Identity => Some((1 << 31, "identity index exceeds i32")),
            Self::Count => Some((i32::MAX as u64, "row count exceeds i32")),
            Self::Zeros { .. } | Self::Empty { .. } | Self::Filled(_) | Self::OneHot { .. } => None,
        }
    }
}

impl RowRule<Dim> {
    /// The extents of its value at `count` rows.
    fn extents(&self, count: &Dim) -> Vec<Dim> {
        match self {
            Self::Identity | Self::Filled(_) => vec![count.clone()],
            Self::Count => vec![Dim::from(1)],
            Self::Zeros { width, .. } | Self::OneHot { width } => {
                vec![count.clone(), width.clone()]
            }
            Self::Empty { width, .. } => vec![Dim::from(0), count.clone(), width.clone()],
        }
    }
}

impl RowRule {
    fn storage(self, rows: u64) -> Result<ConstantStorage, String> {
        let table = |values: Vec<i32>| ConstantStorage {
            element: Element::i32(),
            extents: Box::new([rows]),
            value: ConstantValue::Bytes(values.iter().flat_map(|value| value.to_le_bytes()).collect()),
        };
        Ok(match self {
            Self::Identity => table(
                (0..rows)
                    .map(|index| i32::try_from(index).map_err(|_| "identity index exceeds i32".to_owned()))
                    .collect::<Result<_, _>>()?,
            ),
            Self::Zeros { element, width } => {
                element.canonical_byte_len(&[rows, width]).map_err(|error| error.to_string())?;
                ConstantStorage { element, extents: Box::new([rows, width]), value: ConstantValue::Zero }
            }
            Self::Empty { element, width } => ConstantStorage {
                element,
                extents: Box::new([0, rows, width]),
                value: ConstantValue::Bytes(Arc::from([])),
            },
            Self::Filled(value) => table(vec![value; usize::try_from(rows).map_err(|_| "row count exceeds usize")?]),
            Self::OneHot { width } => f32_rows(rows, width, |column| if column == 0 { 1.0 } else { 0.0 })?,
            Self::Count => {
                let rows = i32::try_from(rows).map_err(|_| "row count exceeds i32")?;
                ConstantStorage {
                    element: Element::i32(),
                    extents: Box::new([1]),
                    value: ConstantValue::Bytes(rows.to_le_bytes().into()),
                }
            }
        })
    }
}

/// The F32 matrix `[rows, width]` whose every row holds `column`'s value at
/// each column.
fn f32_rows(rows: u64, width: u64, column: impl Fn(u64) -> f32) -> Result<ConstantStorage, String> {
    let count = rows.checked_mul(width).ok_or("row constant extent exceeds u64")?;
    usize::try_from(count).map_err(|_| "row constant extent exceeds usize")?;
    let values = (0..count).flat_map(|at| column(at % width).to_le_bytes());
    Ok(ConstantStorage {
        element: Element::f32(),
        extents: Box::new([rows, width]),
        value: ConstantValue::Bytes(values.collect()),
    })
}

/// A constant's value: what the binder uploads for it.
pub(crate) struct ConstantStorage {
    element: Element,
    /// Its extents, allocated exactly: what its host heap counts
    /// (`host_heap`).
    extents: Box<[u64]>,
    value: ConstantValue,
}

#[derive(Clone)]
enum ConstantValue {
    Bytes(Arc<[u8]>),
    Zero,
}

impl ConstantStorage {
    fn same_storage_as(&self, other: &Self) -> bool {
        self.element == other.element
            && self.extents == other.extents
            && match (&self.value, &other.value) {
                (ConstantValue::Zero, ConstantValue::Zero) => true,
                (ConstantValue::Bytes(left), ConstantValue::Bytes(right)) => left == right,
                (ConstantValue::Zero, ConstantValue::Bytes(bytes))
                | (ConstantValue::Bytes(bytes), ConstantValue::Zero) => {
                    bytes.iter().all(|&byte| byte == 0)
                }
            }
    }

    /// Whether the value is held as bytes in a family's one copy of it:
    /// empty bytes, or bytes that are not all zero. A nonempty value of
    /// zeros is held as `ConstantValue::Zero` there, whichever form its
    /// graph declared it in, so what a family holds of a value does not
    /// follow which of its graphs came first.
    fn held_bytes(&self) -> Option<&Arc<[u8]>> {
        match &self.value {
            ConstantValue::Bytes(bytes)
                if bytes.is_empty() || bytes.iter().any(|&byte| byte != 0) =>
            {
                Some(bytes)
            }
            _ => None,
        }
    }

    /// The host heap a family's one copy of this value holds: the value in
    /// its shared allocation, its extents, and its bytes in theirs.
    fn host_heap(&self) -> HostHeap {
        use std::mem::size_of;
        let counts = 2 * size_of::<usize>();
        let mut heap = HostHeap::allocation(1, size_of::<Self>() + counts);
        heap.hold(self.extents.len(), size_of::<u64>());
        if let Some(bytes) = self.held_bytes() {
            // A shared byte slice is allocated with its two counts, to the
            // counts' alignment.
            heap.hold(
                1,
                (counts + bytes.len()).next_multiple_of(size_of::<usize>()),
            );
        }
        heap
    }
}

impl GraphConstant {
    fn of(
        port: Option<NativePort>,
        element: Element,
        extents: Vec<u64>,
        value: ConstantValue,
    ) -> Self {
        Self {
            port,
            source: ConstantSource::Known {
                value: Arc::new(ConstantStorage {
                    element,
                    extents: extents.into(),
                    value,
                }),
                rows: None,
            },
        }
    }

    /// Its value and the rule it follows class rows by.
    ///
    /// # Panics
    ///
    /// On a constant of a recording a fact reaches: a trace evaluates it
    /// ([`Self::recorded`]) before anything reads its value.
    fn known(&self) -> (&Arc<ConstantStorage>, Option<RowRule>) {
        match &self.source {
            ConstantSource::Known { value, rows } => (value, *rows),
            ConstantSource::Recorded(_) => {
                panic!("a recorded graph constant's value is read only at a model's facts (`GraphConstant::recorded`)")
            }
        }
    }

    fn value(&self) -> &Arc<ConstantStorage> {
        self.known().0
    }

    /// What a graph trace keeps of it beside its template: a family
    /// recorder's output, one for each of its builder's constants.
    pub(crate) fn recorded(&self) -> RecordedConstant {
        RecordedConstant(self.source.clone())
    }

    /// The constant `rule` makes at `count`, following the class rows where
    /// `follows`, its port of its extents following the class dimension
    /// `class`, or fixed without one.
    fn of_rule<G: GraphDraft>(
        graph: &mut G,
        rule: RowRule<Dim>,
        count: Dim,
        follows: bool,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError> {
        // Its value's one limit its port's extents do not bound, guarded: a
        // model past it records again, and is refused there.
        if let Some((limit, error)) = rule.count_limit() {
            if !count.at_most(limit) {
                return Err(error.into());
            }
        }
        let extents = rule.extents(&count);
        let element = rule.element();
        let port = match class {
            Some(class) => graph.port_with_class_extent(element, &extents, 0, class)?,
            None => graph.port(element, &extents)?,
        };
        let source = match (rule.try_map(|width| constant_of(&width.expression())), constant_of(&count.expression())) {
            (Some(known), Some(count)) => ConstantSource::Known {
                value: Arc::new(known.storage(count)?),
                rows: follows.then_some(known),
            },
            _ => ConstantSource::Recorded(ConstantExpr::Rule {
                rule: rule.try_map(|width| Some(width.expression())).expect("every dimension has an expression"),
                count: count.expression(),
                follows,
            }),
        };
        Ok(Self { port: Some(port), source })
    }

    /// The constant `rule` makes at `rows` class rows ([`Self::of_rule`]).
    fn of_rows<G: GraphDraft>(
        graph: &mut G,
        rule: RowRule<Dim>,
        rows: Dim,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError> {
        Self::of_rule(graph, rule, rows, true, class)
    }

    /// This constant's value at `rows` class rows, as its graph built at
    /// that class binds it: its own where it does not follow them.
    pub(crate) fn at_rows(&self, rows: u64) -> Result<Self, String> {
        let (_, rule) = self.known();
        Ok(match rule {
            None => self.clone(),
            Some(rule) => Self {
                port: None,
                source: ConstantSource::Known { value: Arc::new(rule.storage(rows)?), rows: Some(rule) },
            },
        })
    }

    fn same_storage_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.value(), other.value()) || self.value().same_storage_as(other.value())
    }

    pub(crate) fn storage_bytes(&self) -> Result<u64, String> {
        self.value()
            .element
            .canonical_byte_len(&self.value().extents)
            .map_err(|error| error.to_string())
    }

    /// The constant of `element` and `extents` holding `table`: a model's
    /// (`Facts::table`), which a trace keeps as the table it is and fills
    /// with each model's own, or a constant one.
    pub(crate) fn table<G: GraphDraft>(
        graph: &mut G,
        element: Element,
        extents: &[Dim],
        table: &Table,
    ) -> Result<Self, GraphError> {
        let port = graph.port(element, extents)?;
        let expressions = extents.iter().map(Dim::expression).collect::<Vec<_>>();
        let known = expressions.iter().map(constant_of).collect::<Option<Vec<_>>>();
        let source = match (known, table.expression()) {
            (Some(extents), TableExpr::Constant(bytes)) => {
                ConstantSource::Known { value: Arc::new(table_storage(element, extents, bytes)), rows: None }
            }
            (_, table) => ConstantSource::Recorded(ConstantExpr::Table { element, extents: expressions, table }),
        };
        Ok(Self { port: Some(port), source })
    }

    /// A rank-1 `i32` constant.
    pub(crate) fn i32<G: GraphDraft>(graph: &mut G, values: &[i32]) -> Result<Self, GraphError> {
        Self::rank1(
            graph,
            Element::i32(),
            values.len(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }


    /// A rank-1 `f32` constant.
    pub(crate) fn f32<G: GraphDraft>(graph: &mut G, values: &[f32]) -> Result<Self, GraphError> {
        Self::rank1(
            graph,
            Element::f32(),
            values.len(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }

    /// A typed empty vector, bound like every other graph constant.
    pub(crate) fn empty<G: GraphDraft>(
        graph: &mut G,
        element: Element,
        constants: &mut Vec<Self>,
    ) -> Result<seismic::WorkflowTensor, GraphError> {
        Self::empty_of(graph, element, &[0u64], constants)
    }

    /// A typed empty tensor of `extents`, one of them zero (a port its
    /// entry binds and reads nothing of), bound like every other graph
    /// constant.
    pub(crate) fn empty_of<G: GraphDraft, D: Clone>(
        graph: &mut G,
        element: Element,
        extents: &[D],
        constants: &mut Vec<Self>,
    ) -> Result<seismic::WorkflowTensor, GraphError>
    where
        Dim: From<D>,
    {
        Ok(Self::empty_port(graph, element, extents, constants)?.tensor().clone())
    }

    /// [`Self::empty_of`]'s port, for an entry argument it binds mutably.
    pub(crate) fn empty_port<G: GraphDraft, D: Clone>(
        graph: &mut G,
        element: Element,
        extents: &[D],
        constants: &mut Vec<Self>,
    ) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        let extents = extents.iter().cloned().map(Dim::from).collect::<Vec<_>>();
        assert!(
            extents.iter().any(|extent| constant_of(&extent.expression()) == Some(0)),
            "an empty tensor has a zero extent"
        );
        let port = graph.port::<Dim>(element, &extents)?;
        let expressions = extents.iter().map(Dim::expression).collect::<Vec<_>>();
        let constant = match expressions.iter().map(constant_of).collect::<Option<Vec<_>>>() {
            Some(extents) => Self::of(Some(port), element, extents, ConstantValue::Bytes(Arc::from([]))),
            None => Self {
                port: Some(port),
                source: ConstantSource::Recorded(ConstantExpr::Empty { element, extents: expressions }),
            },
        };
        let port = constant.port().clone();
        constants.push(constant);
        Ok(port)
    }

    /// The graph's absent operand: a zero-extent F32 constant, which every
    /// port of a weight's second-level scale or packing bound binds where the
    /// weight has none. Its first reader declares it in `constants`, the
    /// graph's (an unread port is unbound), and every later one shares it.
    pub(crate) fn absent<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<Self>,
    ) -> Result<NativePort, GraphError> {
        // The absent port is always a known constant; a recorded one (a
        // fact reaches it) is never it, and is not read here.
        let declared = constants.iter().find_map(|constant| match &constant.source {
            ConstantSource::Known { value, .. } if value.element == Element::f32() && *value.extents == [0] => {
                constant.port.clone()
            }
            _ => None,
        });
        if let Some(port) = declared {
            return Ok(port);
        }
        let absent = Self::f32(graph, &[])?;
        let port = absent.port().clone();
        constants.push(absent);
        Ok(port)
    }

    /// A zero-extent F32 port `[rows, 0]` for a row argument its entry reads
    /// nothing of (the inject values of an output that joins a single
    /// residual stream, the previous pre of a Sinkhorn site mixing by its
    /// own), its rows following the class dimension `class`, or fixed
    /// without one.
    pub(crate) fn absent_rows<G: GraphDraft, R>(
        graph: &mut G,
        rows: R,
        class: Option<&'static str>,
        constants: &mut Vec<Self>,
    ) -> Result<seismic::WorkflowTensor, GraphError>
    where
        Dim: From<R>,
    {
        let inject = Self::row_zeros::<G, Dim, u64>(graph, Element::f32(), Dim::from(rows), 0, class)?;
        let tensor = inject.port().tensor().clone();
        constants.push(inject);
        Ok(tensor)
    }

    /// An `f32` constant of the given extents, possibly empty.
    pub(crate) fn f32_shaped<G: GraphDraft>(
        graph: &mut G,
        extents: &[u64],
        values: &[f32],
    ) -> Result<Self, GraphError> {
        if extents.iter().product::<u64>() != values.len() as u64 {
            return Err("constant values disagree with its extents".into());
        }
        let port = graph.port(Element::f32(), extents)?;
        Ok(Self::of(
            Some(port),
            Element::f32(),
            extents.to_vec(),
            ConstantValue::Bytes(
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        ))
    }

    /// An `i32` constant of the given extents, possibly empty.
    pub(crate) fn i32_shaped<G: GraphDraft>(
        graph: &mut G,
        extents: &[u64],
        values: &[i32],
    ) -> Result<Self, GraphError> {
        if extents.iter().product::<u64>() != values.len() as u64 {
            return Err("constant values disagree with its extents".into());
        }
        let port = graph.port(Element::i32(), extents)?;
        Ok(Self::of(
            Some(port),
            Element::i32(),
            extents.to_vec(),
            ConstantValue::Bytes(
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        ))
    }

    /// The `f32` tensor of `extents` holding `value` everywhere.
    pub(crate) fn f32_filled<G: GraphDraft, D: Clone>(
        graph: &mut G,
        extents: &[D],
        value: f32,
    ) -> Result<Self, GraphError>
    where
        Dim: From<D>,
    {
        let extents = extents.iter().cloned().map(Dim::from).collect::<Vec<_>>();
        let port = graph.port::<Dim>(Element::f32(), &extents)?;
        let expressions = extents.iter().map(Dim::expression).collect::<Vec<_>>();
        let source = match expressions.iter().map(constant_of).collect::<Option<Vec<_>>>() {
            Some(extents) => ConstantSource::Known { value: Arc::new(filled_f32(extents, value)?), rows: None },
            None => ConstantSource::Recorded(ConstantExpr::Filled { extents: expressions, value }),
        };
        Ok(Self { port: Some(port), source })
    }

    /// The `u32` zero matrix `[count, width]`, which does not follow the
    /// class rows.
    pub(crate) fn u32_zeros<G: GraphDraft, C, W>(
        graph: &mut G,
        count: C,
        width: W,
    ) -> Result<Self, GraphError>
    where
        Dim: From<C> + From<W>,
    {
        Self::of_rule(graph, RowRule::Zeros { element: Element::u32(), width: Dim::from(width) }, Dim::from(count), false, None)
    }

    /// The rank-1 `i32` constant `0, 1, …, count - 1` of a count that does
    /// not follow the class rows.
    pub(crate) fn identity<G: GraphDraft, C>(graph: &mut G, count: C) -> Result<Self, GraphError>
    where
        Dim: From<C>,
    {
        Self::of_rule(graph, RowRule::Identity, Dim::from(count), false, None)
    }

    /// Identity constant value for resource accounting without a graph port.
    pub(crate) fn identity_value(count: u64) -> Result<Self, String> {
        Ok(Self {
            port: None,
            source: ConstantSource::Known { value: Arc::new(RowRule::<u64>::Identity.storage(count)?), rows: None },
        })
    }

    /// The row table `0, 1, …, rows - 1` of the class's `rows` rows, its
    /// port following the class dimension `class`, or fixed without one.
    pub(crate) fn row_identity<G: GraphDraft, R>(
        graph: &mut G,
        rows: R,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError>
    where
        Dim: From<R>,
    {
        Self::of_rows(graph, RowRule::Identity, Dim::from(rows), class)
    }

    /// The rank-1 `i32` constant holding the class's `rows` itself.
    pub(crate) fn row_count<G: GraphDraft, R>(graph: &mut G, rows: R) -> Result<Self, GraphError>
    where
        Dim: From<R>,
    {
        Self::of_rows(graph, RowRule::Count, Dim::from(rows), None)
    }

    /// The `f32` zero matrix `[count, width]`, for resource accounting
    /// without a graph port.
    #[cfg(test)]
    fn zeros_value(count: u64, width: u64) -> Result<Self, String> {
        Ok(Self {
            port: None,
            source: ConstantSource::Known {
                value: Arc::new(RowRule::<u64>::Zeros { element: Element::f32(), width }.storage(count)?),
                rows: None,
            },
        })
    }

    /// The zero matrix `[rows, width]` of `element` at the class's `rows`
    /// rows (a zero base for a sum, an absent bitmap or inject), its port
    /// following the class dimension `class`, or fixed without one.
    pub(crate) fn row_zeros<G: GraphDraft, R, W>(
        graph: &mut G,
        element: Element,
        rows: R,
        width: W,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError>
    where
        Dim: From<R> + From<W>,
    {
        Self::of_rows(graph, RowRule::Zeros { element, width: Dim::from(width) }, Dim::from(rows), class)
    }

    /// The empty tensor `[0, rows, width]` of `element` at the class's
    /// `rows` rows (a port its entry binds and reads nothing of).
    pub(crate) fn row_empty<G: GraphDraft, R, W>(
        graph: &mut G,
        element: Element,
        rows: R,
        width: W,
        constants: &mut Vec<Self>,
    ) -> Result<seismic::WorkflowTensor, GraphError>
    where
        Dim: From<R> + From<W>,
    {
        let constant = Self::of_rows(graph, RowRule::Empty { element, width: Dim::from(width) }, Dim::from(rows), None)?;
        let tensor = constant.port().tensor().clone();
        constants.push(constant);
        Ok(tensor)
    }

    /// The F32 matrix `[rows, width]` at the class's `rows` rows, each row
    /// one-hot on its first column.
    pub(crate) fn row_one_hot<G: GraphDraft, R, W>(
        graph: &mut G,
        rows: R,
        width: W,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError>
    where
        Dim: From<R> + From<W>,
    {
        Self::of_rows(graph, RowRule::OneHot { width: Dim::from(width) }, Dim::from(rows), class)
    }

    /// The rank-1 `i32` table of `value` for each of the class's `rows`
    /// rows.
    pub(crate) fn row_filled<G: GraphDraft, R>(graph: &mut G, rows: R, value: i32) -> Result<Self, GraphError>
    where
        Dim: From<R>,
    {
        Self::of_rows(graph, RowRule::Filled(value), Dim::from(rows), None)
    }

    fn rank1<G: GraphDraft>(
        graph: &mut G,
        element: Element,
        length: usize,
        bytes: Arc<[u8]>,
    ) -> Result<Self, GraphError> {
        let extents = vec![u64::try_from(length).map_err(|_| "constant length exceeds u64")?];
        let port = graph.port(element, &extents)?;
        Ok(Self::of(
            Some(port),
            element,
            extents,
            ConstantValue::Bytes(bytes),
        ))
    }

    pub(crate) fn port(&self) -> &NativePort {
        self.port
            .as_ref()
            .expect("a bound graph constant has a port")
    }
}

/// The binder uploads one tensor for each distinct constant value. Count the
/// same unique set before binding so the capacity grant covers its allocation.
pub(crate) fn distinct_storage_bytes<'a>(
    constants: impl IntoIterator<Item = &'a GraphConstant>,
) -> Result<u64, String> {
    let mut unique = Vec::<&GraphConstant>::new();
    let mut total = 0u64;
    for constant in constants {
        if unique.iter().any(|other| other.same_storage_as(constant)) {
            continue;
        }
        total = total
            .checked_add(constant.storage_bytes()?)
            .ok_or("graph constant charge overflows")?;
        unique.push(constant);
    }
    Ok(total)
}

/// Metadata-only family holdings. Each graph contributes independently to
/// the three arena maxima, while constants are uploaded once per distinct
/// value across the family by the production binder.
pub(crate) struct CheckedGraphFamilyResources {
    storage: Option<NativeGraphStorageBytes>,
    constants: Vec<GraphConstant>,
    plans: HostHeap,
    /// Each graph its launches replay ([`Self::include_graph`]).
    graphs: Vec<LaunchedGraph>,
}

impl CheckedGraphFamilyResources {
    pub(crate) fn new() -> Self {
        Self {
            storage: None,
            constants: Vec::new(),
            plans: HostHeap::default(),
            graphs: Vec::new(),
        }
    }

    /// Count one graph of a family whose launch is one run of one of its
    /// graphs (`seismic::NativeGraphFamily::new`): a layout's, or a plan's
    /// sealed alone, recording `recorded`, which a decode step's launch
    /// replays where `decodes`, kept live for each launch in flight.
    pub(crate) fn include_graph(&mut self, recorded: GraphCensus, decodes: bool) {
        self.graphs.push(LaunchedGraph { graph: recorded, pinned_runs: u64::from(decodes), runs: 1 });
    }

    /// Count the host heap of one sealed plan of the family
    /// (`NativeGraphMetadata::plan_heap` or `NativeGraphLayout::plan_heap`).
    pub(crate) fn include_plan(&mut self, heap: HostHeap) -> Result<(), String> {
        self.plans = self
            .plans
            .plus(heap)
            .ok_or("checked graph plan heap overflows")?;
        Ok(())
    }

    pub(crate) fn include(
        &mut self,
        storage: NativeGraphStorageBytes,
        constants: impl IntoIterator<Item = GraphConstant>,
    ) {
        self.storage = Some(match self.storage {
            None => storage,
            Some(previous) => previous.most(storage),
        });
        for constant in constants {
            if !self
                .constants
                .iter()
                .any(|prior| prior.same_storage_as(&constant))
            {
                self.constants.push(constant);
            }
        }
    }

    /// Include a template's graph certified at each of `rows` class rows:
    /// its storage, and each of its constants as a graph of that class binds
    /// it (`GraphConstant::at_rows`).
    pub(crate) fn include_classes(
        &mut self,
        storage: NativeGraphStorageBytes,
        constants: &[GraphConstant],
        rows: &[u64],
    ) -> Result<(), String> {
        let classes = rows
            .iter()
            .flat_map(|&rows| constants.iter().map(move |constant| constant.at_rows(rows)))
            .collect::<Result<Vec<_>, _>>()?;
        self.include(storage, classes);
        Ok(())
    }

    /// The host heap of the family's one copy of each distinct constant
    /// value: what `FamilyConstants` holds once every graph of the family
    /// is prepared.
    pub(crate) fn constant_heap(&self) -> Result<HostHeap, String> {
        self.constants
            .iter()
            .try_fold(HostHeap::default(), |heap, constant| {
                heap.plus(constant.value().host_heap())
            })
            .ok_or_else(|| "graph constant heap overflows".to_owned())
    }

    pub(crate) fn finish(self) -> Result<CheckedGraphResources, String> {
        Ok(CheckedGraphResources {
            host_heap: self
                .plans
                .plus(self.constant_heap()?)
                .ok_or("checked graph host heap overflows")?,
            storage: self.storage.ok_or("checked graph family has no class")?,
            binding_constant_bytes: distinct_storage_bytes(self.constants.iter())?,
            graphs: LaunchGraphs::of(self.graphs, Vec::new()).ok_or("checked graph family census overflows")?,
        })
    }
}

/// A checked family's device storage and constants, and the host heap of
/// its sealed plans and its one copy of each constant value.
#[derive(Debug, PartialEq)]
pub(crate) struct CheckedGraphResources {
    pub(crate) storage: NativeGraphStorageBytes,
    /// What its graphs hold with launches in flight, which its sealed
    /// family's are within ([`CheckedGraphFamilyResources::include_graph`]).
    pub(crate) graphs: LaunchGraphs,
    pub(crate) binding_constant_bytes: u64,
    pub(crate) host_heap: HostHeap,
}

/// The constant values of one graph family, each held once: the family's
/// graphs are prepared on several threads, and every graph that binds a
/// value refers to the copy here. The distinct values are few, so a linear
/// search suffices.
#[derive(Default)]
pub(crate) struct FamilyConstants {
    values: Mutex<Vec<Arc<ConstantStorage>>>,
}

impl FamilyConstants {
    /// Make `constant` refer to the family's copy of its value, which is
    /// its own when no graph before it bound that value.
    pub(crate) fn share(&self, constant: &mut GraphConstant) {
        let mut values = self.values.lock().expect("family constants poisoned");
        let ConstantSource::Known { value, .. } = &mut constant.source else {
            panic!("a prepared graph's constants are known")
        };
        if let Some(held) = values.iter().find(|held| held.same_storage_as(value)) {
            *value = held.clone();
            return;
        }
        // A nonempty value of zeros is held without its bytes
        // (`ConstantStorage::held_bytes`).
        if value.held_bytes().is_none() && matches!(value.value, ConstantValue::Bytes(_)) {
            *value = Arc::new(ConstantStorage {
                element: value.element,
                extents: value.extents.clone(),
                value: ConstantValue::Zero,
            });
        }
        values.push(value.clone());
    }

    /// The host heap the family's values hold
    /// (`CheckedGraphFamilyResources::constant_heap` is what a plan counts
    /// for them).
    pub(crate) fn host_heap(&self) -> Result<HostHeap, String> {
        self.values
            .lock()
            .expect("family constants poisoned")
            .iter()
            .try_fold(HostHeap::default(), |heap, value| {
                heap.plus(value.host_heap())
            })
            .ok_or_else(|| "a family's constants' host heap overflows".to_owned())
    }
}

/// Device tensors of bound constants, one per distinct (element, extents,
/// bytes). Constants are small and few, so a linear search suffices.
pub(crate) struct ConstantTensors {
    device: Device,
    uploaded: Vec<(GraphConstant, Tensor)>,
}

impl ConstantTensors {
    pub(crate) fn new(device: Device) -> Self {
        Self {
            device,
            uploaded: Vec::new(),
        }
    }

    pub(crate) fn tensor(&mut self, constant: &GraphConstant) -> Result<Tensor, String> {
        if let Some((_, tensor)) = self
            .uploaded
            .iter()
            .find(|(uploaded, _)| uploaded.same_storage_as(constant))
        {
            return Ok(tensor.clone());
        }
        let value = constant.value();
        let tensor = match &value.value {
            ConstantValue::Zero => Tensor::zeros(&self.device, value.element, &value.extents),
            ConstantValue::Bytes(bytes) => {
                Tensor::from_host(&self.device, value.element, &value.extents, bytes)
            }
        }
        .map_err(|error| format!("graph constant upload failed: {error}"))?;
        self.uploaded.push((constant.clone(), tensor.clone()));
        Ok(tensor)
    }

    /// Keep the unique uploaded allocations available to the owner census.
    pub(crate) fn into_tensors(self) -> Vec<Tensor> {
        self.uploaded
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::BackendName;

    #[test]
    fn zero_constants_keep_metadata_symbolic_and_bind_exact_bytes() {
        let large = GraphConstant::zeros_value(1 << 20, 1 << 16).unwrap();
        assert!(matches!(large.value().value, ConstantValue::Zero));
        assert_eq!(large.storage_bytes().unwrap(), 1 << 38);

        let zero = GraphConstant::zeros_value(2, 3).unwrap();
        let filled = |value| {
            let storage = filled_f32(vec![2, 3], value).unwrap();
            GraphConstant::of(None, storage.element, storage.extents.into_vec(), storage.value)
        };
        let (explicit, negative_zero) = (filled(0.0), filled(-0.0));
        assert!(zero.same_storage_as(&explicit));
        assert!(!zero.same_storage_as(&negative_zero));
        assert_eq!(distinct_storage_bytes([&zero, &explicit]).unwrap(), 24);

        let mut backends = vec![BackendName::Cpu];
        #[cfg(target_os = "macos")]
        backends.push(BackendName::Metal);
        for backend in backends {
            let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), backend)
                .unwrap();
            let charged = || device.device_ledger().charged;
            let before = charged();
            let mut uploaded = ConstantTensors::new(device.clone());
            let tensor = uploaded.tensor(&zero).unwrap();
            assert_eq!(tensor.read_to_host().unwrap(), vec![0; 24]);
            let tied = uploaded.tensor(&explicit).unwrap();
            assert_eq!(tied.read_to_host().unwrap(), vec![0; 24]);
            assert_eq!(charged() - before, 24);
            drop(tensor);
            drop(tied);
            drop(uploaded);
            assert_eq!(charged(), before);
        }
    }

    #[test]
    fn binding_claim_matches_distinct_physical_constant_charge() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap();
        let mut graph = device.native_graph();
        let first = GraphConstant::i32(&mut graph, &[1, 2, 3]).unwrap();
        let tied = GraphConstant::i32(&mut graph, &[1, 2, 3]).unwrap();
        let other = GraphConstant::f32(&mut graph, &[1.0, 2.0]).unwrap();
        let claim = distinct_storage_bytes([&first, &tied, &other]).unwrap();
        let charged = || device.device_ledger().charged;
        let before = charged();
        let mut uploaded = ConstantTensors::new(device.clone());
        uploaded.tensor(&first).unwrap();
        uploaded.tensor(&tied).unwrap();
        uploaded.tensor(&other).unwrap();
        assert_eq!(charged() - before, claim);
        drop(uploaded);
        assert_eq!(charged(), before);
    }
}
