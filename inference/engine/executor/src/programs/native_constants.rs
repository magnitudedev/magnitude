//! Host constants of sealed graphs. A constant is declared as an external
//! port while its graph is built. Binding uploads each distinct value once and
//! attaches it with the weights as a static binding, so no run writes it.

use super::graph::{GraphError, draft::GraphDraft};
use seismic::{Device, Element, GraphCensus, HostHeap, LaunchGraphs, LaunchedGraph, NativeGraphStorageBytes, NativePort, Tensor};
use std::sync::{Arc, Mutex};

/// A constant of one graph: the port it binds, its value, and how that
/// value follows the graph's class rows. The value is shared: the graphs of
/// a family that bind one value refer to the family's one copy of it
/// (`FamilyConstants`).
#[derive(Clone)]
pub(crate) struct GraphConstant {
    port: Option<NativePort>,
    value: Arc<ConstantStorage>,
    /// The rule its value is made from at its graph's class rows, of a
    /// constant whose extents or values follow them; none of one whose
    /// value is the same at every class.
    rows: Option<RowRule>,
}

/// How a constant's extents and values follow its graph's class rows: the
/// one description its builder declares it from and a plan counts it from
/// at each class a template certifies (`GraphConstant::at_rows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowRule {
    /// The rank-1 `i32` row table `0, 1, …, rows - 1`.
    Identity,
    /// The zero matrix `[rows, width]`.
    Zeros { element: Element, width: u64 },
    /// The empty tensor `[0, rows, width]`.
    Empty { element: Element, width: u64 },
    /// The rank-1 `i32` table of `rows` copies of one value.
    Filled(i32),
    /// The one `i32` `rows` itself.
    Count,
}

impl RowRule {
    fn storage(self, rows: u64) -> Result<ConstantStorage, String> {
        let table = |values: Vec<i32>| ConstantStorage {
            element: Element::i32(),
            extents: vec![rows],
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
                ConstantStorage { element, extents: vec![rows, width], value: ConstantValue::Zero }
            }
            Self::Empty { element, width } => ConstantStorage {
                element,
                extents: vec![0, rows, width],
                value: ConstantValue::Bytes(Arc::from([])),
            },
            Self::Filled(value) => table(vec![value; usize::try_from(rows).map_err(|_| "row count exceeds usize")?]),
            Self::Count => {
                let rows = i32::try_from(rows).map_err(|_| "row count exceeds i32")?;
                ConstantStorage {
                    element: Element::i32(),
                    extents: vec![1],
                    value: ConstantValue::Bytes(rows.to_le_bytes().into()),
                }
            }
        })
    }
}

/// A constant's value: what the binder uploads for it.
struct ConstantStorage {
    element: Element,
    extents: Vec<u64>,
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
            value: Arc::new(ConstantStorage {
                element,
                extents,
                value,
            }),
            rows: None,
        }
    }

    /// The constant `rule` makes at `rows` rows, its port of those extents
    /// following the class dimension `class`, or fixed without one.
    fn of_rows<G: GraphDraft>(
        graph: &mut G,
        rule: RowRule,
        rows: u64,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError> {
        let value = rule.storage(rows)?;
        let port = match class {
            Some(class) => graph.port_with_class_extent(value.element, &value.extents, 0, class)?,
            None => graph.port(value.element, &value.extents)?,
        };
        Ok(Self { port: Some(port), value: Arc::new(value), rows: Some(rule) })
    }

    /// This constant's value at `rows` class rows, as its graph built at
    /// that class binds it: its own where it does not follow them.
    pub(crate) fn at_rows(&self, rows: u64) -> Result<Self, String> {
        Ok(match self.rows {
            None => self.clone(),
            Some(rule) => Self { port: None, value: Arc::new(rule.storage(rows)?), rows: Some(rule) },
        })
    }

    fn same_storage_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value) || self.value.same_storage_as(&other.value)
    }

    pub(crate) fn storage_bytes(&self) -> Result<u64, String> {
        self.value
            .element
            .canonical_byte_len(&self.value.extents)
            .map_err(|error| error.to_string())
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

    /// A rank-1 `u64` constant.
    pub(crate) fn u64<G: GraphDraft>(graph: &mut G, values: &[u64]) -> Result<Self, GraphError> {
        Self::rank1(
            graph,
            Element::u64(),
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
        Self::empty_of(graph, element, &[0], constants)
    }

    /// A typed empty tensor of `extents`, one of them zero (a port its
    /// entry binds and reads nothing of), bound like every other graph
    /// constant.
    pub(crate) fn empty_of<G: GraphDraft>(
        graph: &mut G,
        element: Element,
        extents: &[u64],
        constants: &mut Vec<Self>,
    ) -> Result<seismic::WorkflowTensor, GraphError> {
        assert!(extents.contains(&0), "an empty tensor has a zero extent");
        let port = graph.port(element, extents)?;
        let constant = Self::of(Some(port), element, extents.to_vec(), ConstantValue::Bytes(Arc::from([])));
        let tensor = constant.port().tensor().clone();
        constants.push(constant);
        Ok(tensor)
    }

    /// The graph's absent operand: a zero-extent F32 constant, which every
    /// port of a weight's second-level scale or packing bound binds where the
    /// weight has none. Its first reader declares it in `constants`, the
    /// graph's (an unread port is unbound), and every later one shares it.
    pub(crate) fn absent<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<Self>,
    ) -> Result<NativePort, GraphError> {
        let declared = constants.iter().find_map(|constant| {
            (constant.value.element == Element::f32() && constant.value.extents == [0])
                .then(|| constant.port.clone())
                .flatten()
        });
        if let Some(port) = declared {
            return Ok(port);
        }
        let absent = Self::f32(graph, &[])?;
        let port = absent.port().clone();
        constants.push(absent);
        Ok(port)
    }

    /// A zero-extent inject port `[rows, 0]` for an output that joins a
    /// single residual stream, its rows following the class dimension
    /// `class`, or fixed without one.
    pub(crate) fn absent_inject<G: GraphDraft>(
        graph: &mut G,
        rows: u64,
        class: Option<&'static str>,
        constants: &mut Vec<Self>,
    ) -> Result<seismic::WorkflowTensor, GraphError> {
        let inject = Self::row_zeros(graph, Element::f32(), rows, 0, class)?;
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

    /// The `u32` zero matrix `[count, width]` of fixed extents.
    pub(crate) fn u32_zeros<G: GraphDraft>(
        graph: &mut G,
        count: u64,
        width: u64,
    ) -> Result<Self, GraphError> {
        let value = RowRule::Zeros { element: Element::u32(), width }.storage(count)?;
        let port = graph.port(value.element, &value.extents)?;
        Ok(Self { port: Some(port), value: Arc::new(value), rows: None })
    }

    /// An `f32` constant value of the given extents, for resource accounting
    /// without a graph port.
    pub(crate) fn f32_value(extents: &[u64], values: &[f32]) -> Result<Self, String> {
        if extents.iter().product::<u64>() != values.len() as u64 {
            return Err("constant values disagree with its extents".into());
        }
        Ok(Self::of(
            None,
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

    /// A rank-1 `i32` constant value, for resource accounting without a
    /// graph port.
    pub(crate) fn i32_value(values: &[i32]) -> Self {
        Self::of(
            None,
            Element::i32(),
            vec![values.len() as u64],
            ConstantValue::Bytes(
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        )
    }

    /// The rank-1 `i32` constant `0, 1, …, count - 1` of a count that does
    /// not follow the class rows.
    pub(crate) fn identity<G: GraphDraft>(graph: &mut G, count: u64) -> Result<Self, GraphError> {
        let mut constant = Self::identity_value(count)?;
        constant.port = Some(graph.port(Element::i32(), &constant.value.extents)?);
        Ok(constant)
    }

    /// Identity constant value for resource accounting without a graph port.
    pub(crate) fn identity_value(count: u64) -> Result<Self, String> {
        Ok(Self {
            port: None,
            value: Arc::new(RowRule::Identity.storage(count)?),
            rows: None,
        })
    }

    /// The row table `0, 1, …, rows - 1` of the class's `rows` rows, its
    /// port following the class dimension `class`, or fixed without one.
    pub(crate) fn row_identity<G: GraphDraft>(
        graph: &mut G,
        rows: u64,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError> {
        Self::of_rows(graph, RowRule::Identity, rows, class)
    }

    /// The rank-1 `i32` constant holding the class's `rows` itself.
    pub(crate) fn row_count<G: GraphDraft>(graph: &mut G, rows: u64) -> Result<Self, GraphError> {
        Self::of_rows(graph, RowRule::Count, rows, None)
    }

    /// The `f32` zero matrix `[count, width]`, for resource accounting
    /// without a graph port.
    #[cfg(test)]
    fn zeros_value(count: u64, width: u64) -> Result<Self, String> {
        Ok(Self {
            port: None,
            value: Arc::new(RowRule::Zeros { element: Element::f32(), width }.storage(count)?),
            rows: None,
        })
    }

    /// The zero matrix `[rows, width]` of `element` at the class's `rows`
    /// rows (a zero base for a sum, an absent bitmap or inject), its port
    /// following the class dimension `class`, or fixed without one.
    pub(crate) fn row_zeros<G: GraphDraft>(
        graph: &mut G,
        element: Element,
        rows: u64,
        width: u64,
        class: Option<&'static str>,
    ) -> Result<Self, GraphError> {
        Self::of_rows(graph, RowRule::Zeros { element, width }, rows, class)
    }

    /// The empty tensor `[0, rows, width]` of `element` at the class's
    /// `rows` rows (a port its entry binds and reads nothing of).
    pub(crate) fn row_empty<G: GraphDraft>(
        graph: &mut G,
        element: Element,
        rows: u64,
        width: u64,
        constants: &mut Vec<Self>,
    ) -> Result<seismic::WorkflowTensor, GraphError> {
        let constant = Self::of_rows(graph, RowRule::Empty { element, width }, rows, None)?;
        let tensor = constant.port().tensor().clone();
        constants.push(constant);
        Ok(tensor)
    }

    /// The rank-1 `i32` table of `value` for each of the class's `rows`
    /// rows.
    pub(crate) fn row_filled<G: GraphDraft>(graph: &mut G, rows: u64, value: i32) -> Result<Self, GraphError> {
        Self::of_rows(graph, RowRule::Filled(value), rows, None)
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
                heap.plus(constant.value.host_heap())
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
        if let Some(held) = values
            .iter()
            .find(|held| held.same_storage_as(&constant.value))
        {
            constant.value = held.clone();
            return;
        }
        // A nonempty value of zeros is held without its bytes
        // (`ConstantStorage::held_bytes`).
        if constant.value.held_bytes().is_none()
            && matches!(constant.value.value, ConstantValue::Bytes(_))
        {
            constant.value = Arc::new(ConstantStorage {
                element: constant.value.element,
                extents: constant.value.extents.clone(),
                value: ConstantValue::Zero,
            });
        }
        values.push(constant.value.clone());
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
                if value.extents.capacity() != value.extents.len() {
                    return None;
                }
                heap.plus(value.host_heap())
            })
            .ok_or_else(|| "a family constant holds more than its extents".to_owned())
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
        let value = &constant.value;
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
        assert!(matches!(large.value.value, ConstantValue::Zero));
        assert_eq!(large.storage_bytes().unwrap(), 1 << 38);

        let zero = GraphConstant::zeros_value(2, 3).unwrap();
        let explicit = GraphConstant::f32_value(&[2, 3], &[0.0; 6]).unwrap();
        let negative_zero = GraphConstant::f32_value(&[2, 3], &[-0.0; 6]).unwrap();
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
