//! Host constants of sealed graphs. A constant is declared as an external
//! port while its graph is built. Binding uploads each distinct value once and
//! attaches it with the weights as a static binding, so no run writes it.

use super::graph::{GraphError, draft::GraphDraft};
use seismic::{
    Dim, Element, GraphCensus, HostHeap, LaunchGraphs, LaunchedGraph, NativeGraphStorageBytes,
    NativePort, Table, Tensor,
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

#[derive(Clone)]
enum ConstantSource {
    Known {
        value: Arc<ConstantStorage>,
        /// The rule its value is made from at its graph's class rows, of a
        /// constant whose extents or values follow them; none of one whose
        /// value is the same at every class.
        rows: Option<RowRule>,
    },
}

/// The `f32` tensor of `extents` holding `value` everywhere.
fn filled_f32(extents: Vec<u64>, value: f32) -> Result<ConstantStorage, String> {
    let count = extents
        .iter()
        .try_fold(1u64, |count, &extent| count.checked_mul(extent))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or("filled constant length overflows")?;
    Element::f32()
        .canonical_byte_len(&extents)
        .ok()
        .and_then(|len| usize::try_from(len).ok())
        .ok_or("filled constant length overflows")?;
    let bits = value.to_bits();
    let value = if count > 0 && bits == 0 {
        ConstantValue::Zero
    } else {
        ConstantValue::FilledF32(bits)
    };
    Ok(ConstantStorage {
        element: Element::f32(),
        extents: extents.into(),
        value,
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
        Ok(match self {
            Self::Identity => {
                if rows > (1u64 << 31) {
                    return Err("identity index exceeds i32".to_owned());
                }
                let byte_len = Element::i32()
                    .canonical_byte_len(&[rows])
                    .map_err(|error| error.to_string())?;
                usize::try_from(byte_len).map_err(|_| "identity byte length exceeds usize")?;
                ConstantStorage {
                    element: Element::i32(),
                    extents: Box::new([rows]),
                    value: ConstantValue::Identity,
                }
            }
            Self::Zeros { element, width } => {
                element.canonical_byte_len(&[rows, width]).map_err(|error| error.to_string())?;
                ConstantStorage { element, extents: Box::new([rows, width]), value: ConstantValue::Zero }
            }
            Self::Empty { element, width } => ConstantStorage {
                element,
                extents: Box::new([0, rows, width]),
                value: ConstantValue::Bytes(Arc::from([])),
            },
            Self::Filled(value) => {
                usize::try_from(rows).map_err(|_| "row count exceeds usize")?;
                let byte_len = Element::i32()
                    .canonical_byte_len(&[rows])
                    .map_err(|error| error.to_string())?;
                usize::try_from(byte_len).map_err(|_| "row constant byte length exceeds usize")?;
                ConstantStorage {
                    element: Element::i32(),
                    extents: Box::new([rows]),
                    value: ConstantValue::FilledI32(value),
                }
            }
            Self::OneHot { width } => {
                let count = rows.checked_mul(width).ok_or("row constant extent exceeds u64")?;
                usize::try_from(count).map_err(|_| "row constant extent exceeds usize")?;
                let byte_len = Element::f32()
                    .canonical_byte_len(&[rows, width])
                    .map_err(|error| error.to_string())?;
                usize::try_from(byte_len).map_err(|_| "row constant byte length exceeds usize")?;
                ConstantStorage {
                    element: Element::f32(),
                    extents: Box::new([rows, width]),
                    value: ConstantValue::OneHot,
                }
            }
            Self::Count => {
                let rows = i32::try_from(rows).map_err(|_| "row count exceeds i32")?;
                ConstantStorage {
                    element: Element::i32(),
                    extents: Box::new([1]),
                    value: ConstantValue::FilledI32(rows),
                }
            }
        })
    }
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
    /// Rank-1 `i32` `0, 1, …, rows - 1`.
    Identity,
    /// Rank-1 `i32` repeated `value`.
    FilledI32(i32),
    /// Rank-2 `f32` `[rows, width]` with `1.0` in column 0 and `0.0` elsewhere.
    OneHot,
    /// Tensor of `f32` filled with `f32::from_bits(bits)`.
    FilledF32(u32),
}

impl ConstantStorage {
    fn canonical_usize_len(&self) -> usize {
        let len = self
            .element
            .canonical_byte_len(&self.extents)
            .expect("constant extents have a canonical byte length");
        usize::try_from(len).expect("constant byte length fits usize")
    }

    fn element_count(&self) -> usize {
        let count = self
            .extents
            .iter()
            .try_fold(1u64, |acc, &extent| acc.checked_mul(extent))
            .expect("constant element count fits u64");
        usize::try_from(count).expect("constant element count fits usize")
    }

    fn matches_bytes(&self, bytes: &[u8]) -> bool {
        match &self.value {
            ConstantValue::Zero => bytes.iter().all(|&byte| byte == 0),
            ConstantValue::Bytes(own) => own.as_ref() == bytes,
            ConstantValue::Identity => {
                bytes.len() == self.canonical_usize_len()
                    && bytes
                        .chunks_exact(4)
                        .enumerate()
                        .all(|(index, chunk)| chunk == (index as i32).to_le_bytes())
            }
            ConstantValue::FilledI32(value) => {
                let pattern = value.to_le_bytes();
                bytes.len() == self.canonical_usize_len()
                    && bytes.chunks_exact(4).all(|chunk| chunk == pattern)
            }
            ConstantValue::OneHot => {
                let width = usize::try_from(self.extents[1]).expect("one-hot width fits usize");
                let one = 1.0f32.to_le_bytes();
                let zero = 0.0f32.to_le_bytes();
                bytes.len() == self.canonical_usize_len()
                    && bytes
                        .chunks_exact(4)
                        .enumerate()
                        .all(|(index, chunk)| chunk == if index % width == 0 { one } else { zero })
            }
            ConstantValue::FilledF32(bits) => {
                let pattern = bits.to_le_bytes();
                bytes.len() == self.canonical_usize_len()
                    && bytes.chunks_exact(4).all(|chunk| chunk == pattern)
            }
        }
    }

    fn materialize_bytes(&self) -> Vec<u8> {
        match &self.value {
            ConstantValue::Zero => vec![0u8; self.canonical_usize_len()],
            ConstantValue::Bytes(bytes) => bytes.to_vec(),
            ConstantValue::Identity => {
                let rows = usize::try_from(self.extents[0]).expect("identity rows fit usize");
                let mut bytes = Vec::with_capacity(rows * 4);
                for index in 0..rows {
                    bytes.extend_from_slice(&(index as i32).to_le_bytes());
                }
                bytes
            }
            ConstantValue::FilledI32(value) => {
                let count = self.element_count();
                let pattern = value.to_le_bytes();
                let mut bytes = Vec::with_capacity(count * 4);
                for _ in 0..count {
                    bytes.extend_from_slice(&pattern);
                }
                bytes
            }
            ConstantValue::OneHot => {
                let rows = usize::try_from(self.extents[0]).expect("one-hot rows fit usize");
                let width = usize::try_from(self.extents[1]).expect("one-hot width fits usize");
                let mut bytes = vec![0u8; rows * width * 4];
                if width > 0 {
                    let one = 1.0f32.to_le_bytes();
                    for row in 0..rows {
                        let offset = row * width * 4;
                        bytes[offset..offset + 4].copy_from_slice(&one);
                    }
                }
                bytes
            }
            ConstantValue::FilledF32(bits) => {
                let count = self.element_count();
                let pattern = bits.to_le_bytes();
                let mut bytes = Vec::with_capacity(count * 4);
                for _ in 0..count {
                    bytes.extend_from_slice(&pattern);
                }
                bytes
            }
        }
    }

    fn same_storage_as(&self, other: &Self) -> bool {
        if self.element != other.element || self.extents != other.extents {
            return false;
        }
        if self.extents.contains(&0) {
            return true;
        }
        match (&self.value, &other.value) {
            (ConstantValue::Zero, ConstantValue::Zero)
            | (ConstantValue::Identity, ConstantValue::Identity)
            | (ConstantValue::OneHot, ConstantValue::OneHot) => true,
            (ConstantValue::FilledI32(left), ConstantValue::FilledI32(right)) => left == right,
            (ConstantValue::FilledF32(left), ConstantValue::FilledF32(right)) => left == right,
            (ConstantValue::Bytes(left), ConstantValue::Bytes(right)) => left == right,
            (ConstantValue::Zero, ConstantValue::Identity)
            | (ConstantValue::Identity, ConstantValue::Zero) => *self.extents == [1],
            (ConstantValue::Zero, ConstantValue::FilledI32(value))
            | (ConstantValue::FilledI32(value), ConstantValue::Zero) => *value == 0,
            (ConstantValue::Zero, ConstantValue::FilledF32(bits))
            | (ConstantValue::FilledF32(bits), ConstantValue::Zero) => *bits == 0,
            (ConstantValue::Zero, ConstantValue::OneHot)
            | (ConstantValue::OneHot, ConstantValue::Zero) => false,
            (ConstantValue::Identity, ConstantValue::FilledI32(value))
            | (ConstantValue::FilledI32(value), ConstantValue::Identity) => {
                *self.extents == [1] && *value == 0
            }
            (ConstantValue::OneHot, ConstantValue::FilledF32(bits))
            | (ConstantValue::FilledF32(bits), ConstantValue::OneHot) => {
                self.extents.len() == 2
                    && self.extents[1] == 1
                    && *bits == 1.0f32.to_bits()
            }
            (_, ConstantValue::Bytes(bytes)) => self.matches_bytes(bytes),
            (ConstantValue::Bytes(bytes), _) => other.matches_bytes(bytes),
            _ => false,
        }
    }

    /// The byte length held for this value in a family's one copy of it:
    /// empty bytes, or bytes that are not all zero. A nonempty value of
    /// zeros is held as `ConstantValue::Zero` there, whichever form its
    /// graph declared it in, so what a family holds of a value does not
    /// follow which of its graphs came first.
    fn held_byte_len(&self) -> Option<usize> {
        match &self.value {
            ConstantValue::Zero => None,
            ConstantValue::Bytes(bytes) => {
                (bytes.is_empty() || bytes.iter().any(|&byte| byte != 0)).then_some(bytes.len())
            }
            ConstantValue::Identity => {
                (self.extents[0] != 1).then(|| self.canonical_usize_len())
            }
            ConstantValue::FilledI32(value) => {
                let len = self.canonical_usize_len();
                (len == 0 || *value != 0).then_some(len)
            }
            ConstantValue::OneHot => Some(self.canonical_usize_len()),
            ConstantValue::FilledF32(bits) => {
                let len = self.canonical_usize_len();
                (len == 0 || *bits != 0).then_some(len)
            }
        }
    }

    /// The host heap a family's one copy of this value holds: the value in
    /// its shared allocation, its extents, and its bytes in theirs.
    fn host_heap(&self) -> HostHeap {
        use std::mem::size_of;
        let counts = 2 * size_of::<usize>();
        let mut heap = HostHeap::allocation(1, size_of::<Self>() + counts);
        heap.hold(self.extents.len(), size_of::<u64>());
        if let Some(len) = self.held_byte_len() {
            // A shared byte slice is allocated with its two counts, to the
            // counts' alignment.
            heap.hold(
                1,
                (counts + len).next_multiple_of(size_of::<usize>()),
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
    fn known(&self) -> (&Arc<ConstantStorage>, Option<RowRule>) {
        match &self.source {
            ConstantSource::Known { value, rows } => (value, *rows),
        }
    }

    fn value(&self) -> &Arc<ConstantStorage> {
        self.known().0
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
        let known = rule
            .try_map(|width| Some(width.fixed()))
            .expect("every dimension is concrete");
        let source = ConstantSource::Known {
            value: Arc::new(known.storage(count.fixed())?),
            rows: follows.then_some(known),
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
        let values = extents.iter().map(Dim::fixed).collect::<Vec<_>>();
        let source = ConstantSource::Known {
            value: Arc::new(table_storage(element, values, Arc::clone(table.bytes()))),
            rows: None,
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
            extents.iter().any(|extent| extent.fixed() == 0),
            "an empty tensor has a zero extent"
        );
        let port = graph.port::<Dim>(element, &extents)?;
        let values = extents.iter().map(Dim::fixed).collect::<Vec<_>>();
        let constant = Self::of(Some(port), element, values, ConstantValue::Bytes(Arc::from([])));
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
        let values = extents.iter().map(Dim::fixed).collect::<Vec<_>>();
        let source = ConstantSource::Known {
            value: Arc::new(filled_f32(values, value)?),
            rows: None,
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

    /// Rank-1 `i32` range constant `[start, end)` for resource accounting
    /// without a graph port.
    pub(crate) fn i32_range_value(start: u64, end: u64) -> Result<Self, String> {
        let len = end.saturating_sub(start);
        let bytes = (start..end)
            .map(|row| i32::try_from(row).map_err(|_| "draft output row exceeds i32".to_owned()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        Ok(Self::of(None, Element::i32(), vec![len], ConstantValue::Bytes(bytes)))
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
        let ConstantSource::Known { value, .. } = &mut constant.source;
        if let Some(held) = values.iter().find(|held| held.same_storage_as(value)) {
            *value = held.clone();
            return;
        }
        // A nonempty value of zeros is held without its bytes
        // (`ConstantStorage::held_byte_len`).
        if value.held_byte_len().is_none() && matches!(value.value, ConstantValue::Bytes(_)) {
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
pub(crate) struct ConstantTensors<'c> {
    /// What the constants are uploaded through: the binding's.
    charges: &'c seismic::Charges,
    uploaded: Vec<(GraphConstant, Tensor)>,
}

impl<'c> ConstantTensors<'c> {
    pub(crate) fn new(charges: &'c seismic::Charges) -> Self {
        Self {
            charges,
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
            ConstantValue::Zero => Tensor::zeros(self.charges, value.element, &value.extents),
            ConstantValue::Bytes(bytes) => {
                Tensor::from_host(self.charges, value.element, &value.extents, bytes)
            }
            ConstantValue::Identity
            | ConstantValue::FilledI32(_)
            | ConstantValue::OneHot
            | ConstantValue::FilledF32(_) => {
                let bytes = value.materialize_bytes();
                Tensor::from_host(self.charges, value.element, &value.extents, &bytes)
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
    fn zero_constants_keep_metadata_symbolic_and_bind_one_allocation() {
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
            let before = device.allocated();
            let lone = charge_of(&device, &[&zero]);
            let charges = seismic::Charges::tally(&device);
            let mut uploaded = ConstantTensors::new(&charges);
            let tensor = uploaded.tensor(&zero).unwrap();
            assert_eq!(tensor.read_to_host().unwrap(), vec![0; 24]);
            let tied = uploaded.tensor(&explicit).unwrap();
            assert_eq!(tied.read_to_host().unwrap(), vec![0; 24]);
            assert_eq!(uploaded.uploaded.len(), 1, "{backend:?}: tied constants share one allocation");
            assert_eq!(device.allocated() - before, lone, "{backend:?}: tied constants charge one allocation's bytes");
            drop(tensor);
            drop(tied);
            drop(uploaded);
            assert_eq!(device.allocated(), before);
        }
    }

    /// What `device` charges for `constants` uploaded alone, each once,
    /// returned once they drop: the backend's charge for those
    /// allocations, which a CPU's carved block rounds past their bytes
    /// (`seismic_cpu::buffer::Carver`).
    fn charge_of(device: &seismic::Device, constants: &[&GraphConstant]) -> u64 {
        let before = device.allocated();
        let charges = seismic::Charges::tally(&device);
        let mut uploaded = ConstantTensors::new(&charges);
        for constant in constants {
            uploaded.tensor(constant).unwrap();
        }
        let charged = device.allocated() - before;
        drop(uploaded);
        assert_eq!(device.allocated(), before, "uploaded constants return their charge as they drop");
        charged
    }

    #[test]
    fn binding_claim_counts_each_distinct_constant_once() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap();
        let mut graph = device.native_graph();
        let first = GraphConstant::i32(&mut graph, &[1, 2, 3]).unwrap();
        let tied = GraphConstant::i32(&mut graph, &[1, 2, 3]).unwrap();
        let other = GraphConstant::f32(&mut graph, &[1.0, 2.0]).unwrap();
        assert_eq!(distinct_storage_bytes([&first, &tied, &other]).unwrap(), 20, "the claim counts each distinct value once");
        let distinct = charge_of(&device, &[&first, &other]);
        let before = device.allocated();
        let charges = seismic::Charges::tally(&device);
        let mut uploaded = ConstantTensors::new(&charges);
        uploaded.tensor(&first).unwrap();
        uploaded.tensor(&tied).unwrap();
        uploaded.tensor(&other).unwrap();
        assert_eq!(uploaded.uploaded.len(), 2, "one allocation a distinct value");
        assert_eq!(device.allocated() - before, distinct, "a tied constant charges nothing past its value's");
        drop(uploaded);
        assert_eq!(device.allocated(), before);
    }

    #[test]
    fn symbolic_constants_match_raw_bytes_in_metadata_heap_and_upload() {
        let from_storage = |storage: ConstantStorage| {
            GraphConstant::of(
                None,
                storage.element,
                storage.extents.into_vec(),
                storage.value,
            )
        };
        let raw_i32 = |extents: Vec<u64>, values: &[i32]| {
            let bytes: Arc<[u8]> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            GraphConstant::of(None, Element::i32(), extents, ConstantValue::Bytes(bytes))
        };
        let raw_f32 = |extents: Vec<u64>, values: &[f32]| {
            let bytes: Arc<[u8]> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            GraphConstant::of(None, Element::f32(), extents, ConstantValue::Bytes(bytes))
        };

        let cases: Vec<(&str, GraphConstant, GraphConstant, Vec<u8>)> = vec![
            (
                "identity_5",
                from_storage(RowRule::<u64>::Identity.storage(5).unwrap()),
                raw_i32(vec![5], &[0, 1, 2, 3, 4]),
                [0i32, 1, 2, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            (
                "identity_1",
                from_storage(RowRule::<u64>::Identity.storage(1).unwrap()),
                raw_i32(vec![1], &[0]),
                0i32.to_le_bytes().to_vec(),
            ),
            (
                "identity_0",
                from_storage(RowRule::<u64>::Identity.storage(0).unwrap()),
                raw_i32(vec![0], &[]),
                Vec::new(),
            ),
            (
                "filled_i32_nonzero",
                from_storage(RowRule::<u64>::Filled(7).storage(4).unwrap()),
                raw_i32(vec![4], &[7, 7, 7, 7]),
                [7i32; 4].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            (
                "filled_i32_zero",
                from_storage(RowRule::<u64>::Filled(0).storage(4).unwrap()),
                raw_i32(vec![4], &[0, 0, 0, 0]),
                vec![0u8; 16],
            ),
            (
                "filled_i32_empty",
                from_storage(RowRule::<u64>::Filled(7).storage(0).unwrap()),
                raw_i32(vec![0], &[]),
                Vec::new(),
            ),
            (
                "count_42",
                from_storage(RowRule::<u64>::Count.storage(42).unwrap()),
                raw_i32(vec![1], &[42]),
                42i32.to_le_bytes().to_vec(),
            ),
            (
                "count_0",
                from_storage(RowRule::<u64>::Count.storage(0).unwrap()),
                raw_i32(vec![1], &[0]),
                0i32.to_le_bytes().to_vec(),
            ),
            (
                "one_hot_3x4",
                from_storage(RowRule::<u64>::OneHot { width: 4 }.storage(3).unwrap()),
                raw_f32(
                    vec![3, 4],
                    &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ),
                [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            (
                "one_hot_3x1",
                from_storage(RowRule::<u64>::OneHot { width: 1 }.storage(3).unwrap()),
                raw_f32(vec![3, 1], &[1.0, 1.0, 1.0]),
                [1.0f32; 3].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            (
                "one_hot_empty",
                from_storage(RowRule::<u64>::OneHot { width: 0 }.storage(3).unwrap()),
                raw_f32(vec![3, 0], &[]),
                Vec::new(),
            ),
            (
                "filled_f32_nonzero",
                from_storage(filled_f32(vec![2, 3], 1.5).unwrap()),
                raw_f32(vec![2, 3], &[1.5; 6]),
                [1.5f32; 6].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            (
                "filled_f32_neg_zero",
                from_storage(filled_f32(vec![2, 3], -0.0).unwrap()),
                raw_f32(vec![2, 3], &[-0.0; 6]),
                [-0.0f32; 6].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            (
                "filled_f32_pos_zero",
                from_storage(filled_f32(vec![2, 3], 0.0).unwrap()),
                raw_f32(vec![2, 3], &[0.0; 6]),
                vec![0u8; 24],
            ),
            (
                "filled_f32_empty",
                from_storage(filled_f32(vec![0, 3], 0.0).unwrap()),
                raw_f32(vec![0, 3], &[]),
                Vec::new(),
            ),
        ];

        // Verify symbolic variants did not fall back to Bytes.
        assert!(matches!(cases[0].1.value().value, ConstantValue::Identity));
        assert!(matches!(cases[3].1.value().value, ConstantValue::FilledI32(7)));
        assert!(matches!(cases[6].1.value().value, ConstantValue::FilledI32(42)));
        assert!(matches!(cases[8].1.value().value, ConstantValue::OneHot));
        assert!(matches!(cases[11].1.value().value, ConstantValue::FilledF32(_)));
        assert!(matches!(cases[13].1.value().value, ConstantValue::Zero));

        // Cross-symbolic equivalence and non-equivalence checks.
        let one_hot_3x1 = &cases[9].1;
        let filled_ones_3x1 = from_storage(filled_f32(vec![3, 1], 1.0).unwrap());
        assert!(one_hot_3x1.same_storage_as(&filled_ones_3x1));
        assert!(filled_ones_3x1.same_storage_as(one_hot_3x1));
        let identity_1 = &cases[1].1;
        let count_0 = &cases[7].1;
        assert!(identity_1.same_storage_as(count_0));
        assert!(count_0.same_storage_as(identity_1));
        assert!(!cases[0].1.same_storage_as(&from_storage(RowRule::<u64>::Filled(0).storage(5).unwrap())));
        assert!(!cases[8].1.same_storage_as(&from_storage(filled_f32(vec![3, 4], 1.0).unwrap())));
        assert!(!cases[12].1.same_storage_as(&cases[13].1));

        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = crate::test_support::open_backend(&catalog, BackendName::Cpu).unwrap();
        for (name, symbolic, raw, expected_bytes) in &cases {
            assert!(symbolic.same_storage_as(raw), "{name}: symbolic == raw");
            assert!(raw.same_storage_as(symbolic), "{name}: raw == symbolic");
            assert_eq!(
                symbolic.storage_bytes().unwrap(),
                raw.storage_bytes().unwrap(),
                "{name}: storage_bytes"
            );
            assert_eq!(
                symbolic.value().host_heap(),
                raw.value().host_heap(),
                "{name}: host_heap"
            );
            assert_eq!(
                distinct_storage_bytes([symbolic, raw]).unwrap(),
                raw.storage_bytes().unwrap(),
                "{name}: distinct_storage_bytes"
            );

            let charges = seismic::Charges::tally(&device);
            let mut uploaded = ConstantTensors::new(&charges);
            let tensor = uploaded.tensor(symbolic).unwrap();
            assert_eq!(tensor.read_to_host().unwrap(), *expected_bytes, "{name}: upload bytes");
            let tied = uploaded.tensor(raw).unwrap();
            assert_eq!(tied.read_to_host().unwrap(), *expected_bytes, "{name}: tied upload bytes");
            assert_eq!(uploaded.uploaded.len(), 1, "{name}: symbolic and raw share one allocation");
        }
    }
}
