use crate::{
    ModelLoadPlan,
    planning::BankPart,
    programs::{
        graph::{
            GraphError,
            draft::GraphDraft,
            facts::{Numbers, constant},
        },
        native_constants::GraphConstant,
    },
};
use magnitude_family_contracts::{SublayerIndex, Decoder, ProgressiveCodeDepth, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{import_dense, repack_weight, selection_notice, stored_rows};
use seismic::{
    Dim, Element, NativeGraphMetadata, NativePort, NativeTimelinePort, WorkflowTensor,
};

/// How a load holds a planned weight's values, as its graphs read them
/// (`crate::planning::Holding`): topology, never a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ViewHolding {
    /// Every value, imported at load.
    Whole,
    /// A streamed gathered table, its slots filled as `StreamedFill` says.
    Streamed(StreamedFill),
    /// A routed expert stack in the load's expert bank.
    Banked,
}

/// How a streamed table's slots are filled (`crate::TableFill`): by the
/// host as stored records, or imported, with landing slots for a read of
/// selected rows (`lands`) or without.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StreamedFill {
    Stored,
    Imported { lands: bool },
}

/// What a graph reads of one planned weight (`WeightPlan`): its
/// representation and holding as topology, its extents as numbers. The one
/// route from a load's weights to its graphs ([`PlannedWeights`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct WeightView<D = Dim> {
    pub role: WeightRole,
    /// The representation its import reads (`WeightPlan::upload`).
    pub upload: Element,
    pub resident: Element,
    pub holding: ViewHolding,
    /// The source code depth a progressive placement retains.
    pub progressive: Option<ProgressiveCodeDepth>,
    /// Its resident shape: a streamed table's slots', a banked stack's
    /// group's records'.
    pub shape: Vec<D>,
    /// Its second-level scale's resident extent `S` (`WeightScalePlan`), of
    /// a weight with one: 1 for a matrix, its matrices for a stack.
    pub scale: Option<D>,
    /// Its packing bound's extent `B`, of a weight with one.
    pub bound: Option<D>,
    /// The shape `[R, F]` of a streamed table's directory; `[0, 1]` of a
    /// weight held otherwise (`WeightPlan::directory_shape`).
    pub directory: [D; 2],
    /// A banked stack's layer's experts: its home records' `E`.
    pub home: Option<D>,
}

impl WeightView<u64> {
    /// The view of `load`'s planned weight of `role`.
    pub(crate) fn planned(load: &ModelLoadPlan, role: WeightRole) -> Result<Self, GraphError> {
        let plan = load
            .weights()
            .find(|plan| plan.role == role)
            .ok_or_else(|| format!("missing planned weight {role:?}"))?;
        Ok(Self::of(load, plan))
    }

    /// The view of `plan`, a weight of `load`.
    pub(crate) fn of(load: &ModelLoadPlan, plan: &crate::WeightPlan) -> Self {
        let holding = match plan.holding {
            crate::Holding::Whole => ViewHolding::Whole,
            crate::Holding::Streamed(streamed) => ViewHolding::Streamed(match streamed.fill {
                crate::planning::TableFill::Stored => StreamedFill::Stored,
                crate::planning::TableFill::Imported(_) => StreamedFill::Imported { lands: plan.lands_selected_rows() },
            }),
            crate::Holding::Banked { .. } => ViewHolding::Banked,
        };
        let home = plan
            .banked()
            .and(load.expert_bank())
            .and_then(|bank| bank.layer(plan.role.scope))
            .map(|(_, layer)| layer.experts);
        Self {
            role: plan.role,
            upload: plan.upload,
            resident: plan.resident,
            holding,
            progressive: plan.progressive_depth(),
            shape: plan.shape.clone(),
            scale: plan.scale.as_ref().map(|scale| scale.extent),
            bound: (plan.bound != 0).then_some(plan.bound),
            directory: plan.directory_shape(),
            home,
        }
    }
}

impl<D> Numbers<D> for WeightView<D> {
    type Of<E> = WeightView<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> WeightView<E> {
        WeightView {
            role: self.role,
            upload: self.upload,
            resident: self.resident,
            holding: self.holding,
            progressive: self.progressive,
            shape: self.shape.map(&mut *f),
            scale: self.scale.map(&mut *f),
            bound: self.bound.map(&mut *f),
            directory: self.directory.map(&mut *f),
            home: self.home.map(&mut *f),
        }
    }
}

impl<D> WeightView<D> {
    /// Whether a read of rows the device selected from its table is landed:
    /// a table held whole is read as it is, an imported streamed table only
    /// through its landing slots.
    pub(crate) fn reads_landed(&self) -> Result<bool, GraphError> {
        match self.holding {
            ViewHolding::Whole | ViewHolding::Banked => Ok(false),
            ViewHolding::Streamed(StreamedFill::Imported { lands: true }) => Ok(true),
            ViewHolding::Streamed(StreamedFill::Stored | StreamedFill::Imported { lands: false }) => Err(format!(
                "planned weight {:?} has no landing slots for a read of selected rows",
                self.role
            )
            .into()),
        }
    }
}

impl WeightView {
    /// A gathered table's rows as its readers address them (`V`): a
    /// streamed table's slots (`WeightPlan::table_slots`), else its first
    /// extent.
    fn table_rows(&self) -> Result<Dim, GraphError> {
        let rows = match self.holding {
            ViewHolding::Streamed(_) => self.shape.split_last().map(|(_, slots)| {
                slots.iter().fold(Dim::from(1), |product, extent| &product * extent)
            }),
            ViewHolding::Whole | ViewHolding::Banked => self.shape.first().cloned(),
        };
        Ok(rows.ok_or_else(|| format!("planned weight {:?} is no table of rows", self.role))?)
    }
}

/// The element and extents of one part of a streamed routed layer's bank
/// (`ExpertBankPlan::part_shape`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BankPartView<D = Dim> {
    pub scope: WeightScope,
    pub part: BankPart,
    pub element: Element,
    pub extents: Vec<D>,
}

impl<D> Numbers<D> for BankPartView<D> {
    type Of<E> = BankPartView<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> BankPartView<E> {
        BankPartView { scope: self.scope, part: self.part, element: self.element, extents: self.extents.map(f) }
    }
}

/// A load's weights as a graph builder reads them: the load itself, its
/// numbers constants, or a recording's [`GraphWeights`], its numbers facts.
/// Every weight helper of the graphs reads weights through it and nothing
/// else of the load.
pub(crate) trait PlannedWeights: PlannedBank {
    /// The view of the planned weight of `role`.
    fn weight(&self, role: WeightRole) -> Result<WeightView, GraphError>;
}

/// A load's expert bank as a graph builder reads its parts: the bank itself,
/// the load's, or a recording's [`GraphWeights`].
pub(crate) trait PlannedBank {
    /// Part `part` of the bank of the streamed routed layer of `scope`.
    fn bank_part(&self, scope: WeightScope, part: BankPart) -> Result<BankPartView, GraphError>;
}

impl PlannedBank for crate::ExpertBankPlan {
    fn bank_part(&self, scope: WeightScope, part: BankPart) -> Result<BankPartView, GraphError> {
        Ok(constant(bank_part_view(self, scope, part)?))
    }
}

impl PlannedWeights for ModelLoadPlan {
    fn weight(&self, role: WeightRole) -> Result<WeightView, GraphError> {
        Ok(constant(WeightView::planned(self, role)?))
    }
}

impl PlannedBank for ModelLoadPlan {
    fn bank_part(&self, scope: WeightScope, part: BankPart) -> Result<BankPartView, GraphError> {
        let bank = self.expert_bank().ok_or_else(|| format!("{scope:?} has no expert bank"))?;
        bank.bank_part(scope, part)
    }
}

fn bank_part_view(
    bank: &crate::ExpertBankPlan,
    scope: WeightScope,
    part: BankPart,
) -> Result<BankPartView<u64>, GraphError> {
    let (layer, _) = bank
        .layer(scope)
        .ok_or_else(|| format!("{scope:?} is not a streamed routed layer"))?;
    let (element, extents) = bank.part_shape(layer, part)?;
    Ok(BankPartView { scope, part, element, extents })
}

/// The weights a recorded graph reads, split from a load
/// ([`GraphWeights::of`], [`GraphWeights::of_block`]): what its key holds (with [`FactIndex`]
/// numbers) and what its recorder reads ([`Dim`] numbers).
///
/// [`FactIndex`]: crate::programs::graph::facts::FactIndex
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GraphWeights<D = Dim> {
    weights: Vec<WeightView<D>>,
    banks: Vec<BankPartView<D>>,
}

impl GraphWeights<u64> {
    /// The weights of `load` whose roles `reads` selects: every weight a
    /// graph's builder may read, and no bank part. A weight the builder
    /// reads beyond them is missing from its recording, which fails as the
    /// load would without it.
    pub(crate) fn of(load: &ModelLoadPlan, reads: impl Fn(WeightRole) -> bool) -> Self {
        Self {
            weights: load.weights().filter(|plan| reads(plan.role)).map(|plan| WeightView::of(load, plan)).collect(),
            banks: Vec::new(),
        }
    }

    /// What a graph of target block `block` may read: its sublayers'
    /// weights, with the target's own of the exit block (`exit`), and every
    /// part of its streamed layers' banks, with the weights and policy of
    /// each layer they lead to (`ExpertBankPlan::lead_layer`).
    pub(crate) fn of_block(load: &ModelLoadPlan, block: u32, exit: bool) -> Result<Self, GraphError> {
        let own = |scope: WeightScope| match scope {
            WeightScope::TargetSublayer(index) | WeightScope::TargetBranch { sublayer: index, .. } => {
                index.block == block
            }
            WeightScope::Target => exit,
            _ => false,
        };
        // Its streamed layers, and the layers their hint leads predict,
        // whose hinted weights and policy they bind.
        let mut banks = Vec::new();
        let mut leads = Vec::new();
        if let Some(bank) = load.expert_bank() {
            for index in 0..bank.layers().len() {
                let scope = bank.layers()[index].scope;
                if !own(scope) {
                    continue;
                }
                banks.extend(
                    bank.held_parts(index)?
                        .into_iter()
                        .map(|(part, element, extents)| BankPartView { scope, part, element, extents }),
                );
                for lead in 0..crate::planning::HINT_LEADS as usize {
                    let next = bank.layers()[bank.lead_layer(index, lead)].scope;
                    if !leads.contains(&next) {
                        leads.push(next);
                        if !own(next) {
                            banks.push(bank_part_view(bank, next, BankPart::Policy)?);
                        }
                    }
                }
            }
        }
        Ok(Self { banks, ..Self::of(load, |role| own(role.scope) || leads.contains(&role.scope)) })
    }
}

/// The weights of `load` as a graph of target block `block` reads them:
/// every scope relative to the block (`relative_scope`), as its key holds
/// them, resolved to the block's own.
pub(crate) struct BlockWeights<'l> {
    pub load: &'l ModelLoadPlan,
    pub block: u32,
}

impl PlannedWeights for BlockWeights<'_> {
    fn weight(&self, role: WeightRole) -> Result<WeightView, GraphError> {
        self.load.weight(WeightRole { scope: absolute_scope(role.scope, self.block), ..role })
    }
}

impl PlannedBank for BlockWeights<'_> {
    fn bank_part(&self, scope: WeightScope, part: BankPart) -> Result<BankPartView, GraphError> {
        self.load.bank_part(absolute_scope(scope, self.block), part)
    }
}

/// The scope a scope relative to target block `block` names
/// (`relative_scope`'s inverse).
pub(crate) fn absolute_scope(scope: WeightScope, block: u32) -> WeightScope {
    relative_scope(scope, block.wrapping_neg())
}

/// `scope` relative to the target block of `base` (`relative_scope`): what
/// a streamed layer of `base` names of another layer.
pub(crate) fn relative_scope_of(scope: WeightScope, base: WeightScope) -> WeightScope {
    match base {
        WeightScope::TargetSublayer(index) | WeightScope::TargetBranch { sublayer: index, .. } => {
            relative_scope(scope, index.block)
        }
        _ => scope,
    }
}

/// `scope` relative to target block `block`: block `block + k`'s as block
/// `k` (wrapping), any other as it is. A block graph names what it reads
/// so, the block itself block 0, so blocks of one topology at any index
/// share its traces and its prepared graph (`block_scope`).
pub(crate) fn relative_scope(scope: WeightScope, block: u32) -> WeightScope {
    let relative = |index: SublayerIndex| SublayerIndex { block: index.block.wrapping_sub(block), ..index };
    match scope {
        WeightScope::TargetSublayer(index) => WeightScope::TargetSublayer(relative(index)),
        WeightScope::TargetBranch { sublayer, branch } => WeightScope::TargetBranch { sublayer: relative(sublayer), branch },
        scope => scope,
    }
}

impl<D> GraphWeights<D> {
    /// Its scopes relative to target block `block` (`relative_scope`).
    pub(crate) fn relative_to(self, block: u32) -> Self {
        Self {
            weights: self
                .weights
                .into_iter()
                .map(|view| WeightView {
                    role: WeightRole { scope: relative_scope(view.role.scope, block), ..view.role },
                    ..view
                })
                .collect(),
            banks: self
                .banks
                .into_iter()
                .map(|part| BankPartView { scope: relative_scope(part.scope, block), ..part })
                .collect(),
        }
    }
}

impl<D> Numbers<D> for GraphWeights<D> {
    type Of<E> = GraphWeights<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> GraphWeights<E> {
        GraphWeights {
            weights: self.weights.into_iter().map(|view| view.map(&mut *f)).collect(),
            banks: self.banks.into_iter().map(|part| part.map(&mut *f)).collect(),
        }
    }
}

impl PlannedWeights for GraphWeights {
    fn weight(&self, role: WeightRole) -> Result<WeightView, GraphError> {
        self.weights
            .iter()
            .find(|view| view.role == role)
            .cloned()
            .ok_or_else(|| format!("missing planned weight {role:?}").into())
    }
}

impl PlannedBank for GraphWeights {
    fn bank_part(&self, scope: WeightScope, part: BankPart) -> Result<BankPartView, GraphError> {
        self.banks
            .iter()
            .find(|view| view.scope == scope && view.part == part)
            .cloned()
            .ok_or_else(|| format!("bank part {part:?} of {scope:?} is not among the recorded weights").into())
    }
}

/// What a graph's weight port binds of the resident weight of `role`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct WeightPort {
    pub role: WeightRole,
    pub part: WeightPart,
}

/// A resident weight's bound parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WeightPart {
    /// The weight in its resident representation.
    Values,
    /// Its resident second-level scale (`WeightPlan::scale`).
    Scale,
    /// Its resident packing bound (`WeightPlan::bound`).
    Bound,
    /// A streamed gathered table's directory of every row's slot
    /// (`WeightPlan::streamed`).
    Directory,
    /// A banked expert stack's home records, `[E, N, K]`: its layer's
    /// experts at their home records in its group's (`ExpertBankPlan::home`),
    /// where a whole bank holds each, which the layer's resident form reads
    /// in place (`RunStage::Resident`).
    Home,
}

/// A graph's park: one of `kind`, before a call of the streamed routed
/// layer of `scope` in the block the graph was prepared for. A graph shared
/// by equal-shape blocks parks each run at its own block's layer
/// (`block_scope`), as a bank port binds its own block's part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GraphPark {
    pub scope: WeightScope,
    pub kind: crate::experts::ParkKind,
}

/// A timeline node of a graph that each run binds to the bank's park
/// timeline (`BankRun::bind`): a park, or the publication
/// of the site request of the streamed layer of a scope
/// (`seismic::timeline::Publish`), each registered with the value it
/// takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphTimeline {
    Park(GraphPark),
    Publish(WeightScope),
}

/// What a graph's bank port binds: `part` of the expert bank of the
/// streamed routed layer of `scope` (`ExpertBank::part`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BankPort {
    pub scope: WeightScope,
    pub part: crate::planning::BankPart,
}

/// The port of `part` of the bank of the streamed routed layer of `scope`,
/// in the shape the bank allocates it.
pub(crate) fn bank_port<G: GraphDraft>(
    graph: &mut G,
    bank: &(impl PlannedBank + ?Sized),
    scope: WeightScope,
    part: BankPart,
    ports: &mut Vec<(BankPort, NativePort)>,
) -> Result<NativePort, GraphError> {
    let view = bank.bank_part(scope, part)?;
    let port = graph.port(view.element, &view.extents)?;
    ports.push((BankPort { scope, part }, port.clone()));
    Ok(port)
}

impl WeightPart {
    /// This part of `weight`.
    pub(crate) fn of(self, weight: &crate::ResidentWeight) -> Result<&seismic::Tensor, String> {
        match self {
            Self::Values => Ok(weight.tensor()),
            Self::Directory => weight.table().directory().ok_or_else(|| {
                format!(
                    "resident weight {:?} is held whole and has no directory",
                    weight.descriptor().name
                )
            }),
            Self::Scale => weight.scale().ok_or_else(|| {
                format!(
                    "resident weight {:?} has no second-level scale",
                    weight.descriptor().name
                )
            }),
            Self::Bound => weight.bound().ok_or_else(|| {
                format!(
                    "resident weight {:?} has no packing bound",
                    weight.descriptor().name
                )
            }),
            // Only a banked stack has home records, and the bank binds them
            // (`ResidentTarget::bound`).
            Self::Home => panic!("resident weight {:?} is held whole and has no home records", weight.descriptor().name),
        }
    }
}

/// The port of the planned weight of `role`'s `part`.
pub(crate) fn weight_port<G: GraphDraft>(
    graph: &mut G,
    weights: &(impl PlannedWeights + ?Sized),
    role: WeightRole,
    part: WeightPart,
    ports: &mut Vec<(WeightPort, NativePort)>,
) -> Result<NativePort, GraphError> {
    let view = weights.weight(role)?;
    let port = match part {
        WeightPart::Values => graph.port(view.resident, &view.shape),
        WeightPart::Scale => match view.scale {
            Some(extent) => graph.port(Element::f32(), &[extent]),
            None => return Err(format!("planned weight {role:?} has no second-level scale").into()),
        },
        WeightPart::Bound => match view.bound {
            Some(extent) => graph.port(Element::f32(), &[extent]),
            None => return Err(format!("planned weight {role:?} has no packing bound").into()),
        },
        WeightPart::Directory => match view.holding {
            ViewHolding::Streamed(_) => graph.port(Element::i32(), &view.directory),
            ViewHolding::Whole | ViewHolding::Banked => {
                return Err(format!("planned weight {role:?} is held whole and has no directory").into());
            }
        },
        // The group's records' shape at its layer's experts.
        WeightPart::Home => match view.home {
            Some(experts) => graph.port(view.resident, &[&[experts][..], &view.shape[1..]].concat()),
            None => {
                return Err(format!("planned weight {role:?} is no banked stack and has no home records").into());
            }
        },
    }?;
    ports.push((WeightPort { role, part }, port.clone()));
    Ok(port)
}

/// What a reader of a gathered table addresses its rows through
/// (`gathered_row`): a directory of every row's slot, or an empty one under
/// which the reader reads each row it is given.
#[derive(Clone)]
pub(crate) struct RowDirectory {
    pub tensor: WorkflowTensor,
    /// Its entries and their fields: the readers' `R` and `F`.
    pub shape: [Dim; 2],
}

impl RowDirectory {
    /// The empty directory, a constant of the graph.
    pub(crate) fn direct<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let directory = GraphConstant::i32_shaped(graph, &[0, 1], &[])?;
        let tensor = directory.port().tensor().clone();
        constants.push(directory);
        Ok(Self {
            tensor,
            shape: [Dim::from(0), Dim::from(1)],
        })
    }
}

/// A gathered table as a graph's reads find it: the tensor they gather from
/// (the table's rows, or the slots of a streamed one) and the directory they
/// address it through. A read of rows the host knows gathers by the indices
/// the host writes (`ResidentTable::indices`): `by_rows`. A read of rows the
/// device selected finds their slots in the table's directory: `selected`.
pub(crate) struct GatheredTable {
    pub table: WorkflowTensor,
    /// The rows of `table`: the readers' `V`.
    pub rows: Dim,
    role: WeightRole,
    /// The shape of a streamed table's directory; `[0, 1]` of a table held
    /// whole.
    directory_shape: [Dim; 2],
    /// The empty directory, declared by the graph's first read by rows.
    direct: Option<RowDirectory>,
    /// The directory's port, declared by the graph's first read of selected
    /// rows: a graph that reads none has no such port.
    directory: Option<RowDirectory>,
}

impl GatheredTable {
    /// The planned table of `scope` and `kind`.
    pub(crate) fn of<G: GraphDraft>(
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        scope: WeightScope,
        kind: WeightKind,
        ports: &mut Vec<(WeightPort, NativePort)>,
    ) -> Result<Self, GraphError> {
        let role = WeightRole { scope, kind };
        let view = weights.weight(role)?;
        let table = weight_port(graph, weights, role, WeightPart::Values, ports)?
            .tensor()
            .clone();
        Ok(Self {
            table,
            rows: view.table_rows()?,
            role,
            directory_shape: view.directory,
            direct: None,
            directory: None,
        })
    }

    /// The directory of a read of the table's own rows, as the host gives
    /// them: empty.
    pub(crate) fn by_rows<G: GraphDraft>(
        &mut self,
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<RowDirectory, GraphError> {
        if let Some(direct) = &self.direct {
            return Ok(direct.clone());
        }
        Ok(self
            .direct
            .insert(RowDirectory::direct(graph, constants)?)
            .clone())
    }

    /// The table's directory of every row's slot, declared by the graph's
    /// first read of it.
    fn directory<G: GraphDraft>(
        &mut self,
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        ports: &mut Vec<(WeightPort, NativePort)>,
    ) -> Result<RowDirectory, GraphError> {
        if let Some(directory) = &self.directory {
            return Ok(directory.clone());
        }
        let tensor = weight_port(graph, weights, self.role, WeightPart::Directory, ports)?
            .tensor()
            .clone();
        Ok(self
            .directory
            .insert(RowDirectory {
                tensor,
                shape: self.directory_shape.clone(),
            })
            .clone())
    }

    /// A read of rows the device selected: `reader`, given the directory it
    /// addresses the table through, enqueues it. A table held whole is read
    /// under the empty directory. An imported streamed table is read through
    /// its directory with the read landed (`DeviceTable::read_selected`):
    /// ahead of the reader the step copies `selection` (`[rows, 2]`, the
    /// selected rows) for the host and waits at the table's gate, then the
    /// table weight's own import converts the landing staging into the
    /// landing slots; after it the read signals. The landed read's ports,
    /// which each run binds, are returned with the reader's value. `class`
    /// names the graph-class dimension `rows` is, when the graph's classes
    /// vary it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_selected<'a, G: GraphDraft + 'a, T>(
        &mut self,
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        landing: Option<LandingEntries<'a, G>>,
        selection: seismic::WorkflowTensorRef<'_>,
        rows: &Dim,
        class: Option<&'static str>,
        reader: impl FnOnce(&mut G, &RowDirectory) -> Result<T, GraphError>,
    ) -> Result<(T, Option<LandedReadPorts>), GraphError> {
        let view = weights.weight(self.role)?;
        if !view.reads_landed()? {
            let directory = self.by_rows(graph, constants)?;
            return Ok((reader(graph, &directory)?, None));
        }
        let landing =
            landing.ok_or_else(|| format!("a landed read of {:?} without its entries", self.role))?;
        let directory = self.directory(graph, weights, ports)?;
        let (value, ports) = landed_read(graph, &view, landing, selection, rows, class, |graph| {
            reader(graph, &directory)
        })?;
        Ok((value, Some(ports)))
    }
}

/// The read `reader` enqueues of `rows` rows the device selected
/// (`selection`, `[rows, 2]`) from the imported table `plan`, landed
/// (`DeviceTable::read_selected`): ahead of it the selection's copy for the
/// host, its signal and the table's gate, then the table weight's own
/// import of the landing staging into the landing slots; after it the
/// read's signal. `class` names the graph-class dimension `rows` is, when
/// the graph's classes vary it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn landed_read<'a, G: GraphDraft + 'a, T>(
    graph: &mut G,
    plan: &WeightView,
    landing: LandingEntries<'a, G>,
    selection: seismic::WorkflowTensorRef<'_>,
    rows: &Dim,
    class: Option<&'static str>,
    reader: impl FnOnce(&mut G) -> Result<T, GraphError>,
) -> Result<(T, LandedReadPorts), GraphError> {
    let port = |graph: &mut G, element, extents: &[&Dim], axis| match class {
        Some(dimension) => graph.port_with_class_extent(element, extents, axis, dimension),
        None => graph.port(element, extents),
    };
    let two = Dim::from(2);
    let mut notice = port(graph, Element::i32(), &[rows, &two], 0)?;
    graph.enqueue::<selection_notice::Entry>(
        landing.notice,
        &[("M", rows)],
        selection_notice::WorkflowArgs {
            selection,
            notice: notice.tensor_mut().into(),
        },
    )?;
    let noticed = graph.timeline(seismic::timeline::Signal);
    let gate = graph.timeline(seismic::timeline::Wait);
    // The import's `[B, N, K]` view of the landing rows.
    let width = plan.shape.get(1).ok_or_else(|| format!("landed table {:?} is not a matrix", plan.role))?;
    let one = Dim::from(1);
    let staging = port(graph, plan.upload, &[&one, rows, width], 1)?;
    let landed = port(graph, plan.resident, &[&one, rows, width], 1)?;
    let dimensions = [("B", &one), ("N", rows), ("K", width)];
    let converted = match landing.import {
        LandingImport::Dense(entry) => {
            graph
                .enqueue::<import_dense::Entry>(
                    entry,
                    &dimensions,
                    import_dense::WorkflowArgs {
                        source: staging.tensor().into(),
                    },
                )?
                .value
        }
        LandingImport::Repack(entry) => {
            graph
                .enqueue::<repack_weight::Entry>(
                    entry,
                    &dimensions,
                    repack_weight::WorkflowArgs {
                        source: staging.tensor().into(),
                    },
                )?
                .value
        }
    };
    graph.store(&converted, &landed)?;
    let value = reader(graph)?;
    let read = graph.timeline(seismic::timeline::Signal);
    Ok((
        value,
        LandedReadPorts {
            rows: rows.clone(),
            notice,
            noticed,
            gate,
            staging,
            landing: landed,
            read,
        },
    ))
}

/// The entries of a landed read (`GatheredTable::read_selected`): the copy
/// of its selection for the host, and its table weight's own import, which
/// converts the rows the host landed as stored.
pub(crate) struct LandingEntries<'a, G: GraphDraft + 'a> {
    pub notice: G::Binding<'a, selection_notice::Entry>,
    pub import: LandingImport<'a, G>,
}

pub(crate) enum LandingImport<'a, G: GraphDraft + 'a> {
    Dense(G::Binding<'a, import_dense::Entry>),
    Repack(G::Binding<'a, repack_weight::Entry>),
}

impl<'a, G: GraphDraft + 'a> Clone for LandingEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, G: GraphDraft + 'a> Copy for LandingEntries<'a, G> {}

impl<'a, G: GraphDraft + 'a> Clone for LandingImport<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, G: GraphDraft + 'a> Copy for LandingImport<'a, G> {}

impl<'a> LandingEntries<'a, seismic::NativeGraph> {
    /// The attested entries of a table's landed reads.
    pub(crate) fn native(landing: &'a crate::native::AttestedLanding) -> Self {
        Self {
            notice: &landing.notice,
            import: match &landing.import {
                crate::native::AttestedImport::Dense(entry) => LandingImport::Dense(entry),
                crate::native::AttestedImport::Repack { repack, .. } => {
                    LandingImport::Repack(repack)
                }
            },
        }
    }
}

/// The checked entries of the landed reads of the table `role`, for the
/// metadata graphs that certify a family's storage: `None` of a table
/// without landing slots.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CheckedLandingEntries {
    import: crate::ImportProgramSlot,
    elements: [(&'static str, Element); 2],
}

impl CheckedLandingEntries {
    pub(crate) fn of(weights: &(impl PlannedWeights + ?Sized), role: WeightRole) -> Result<Option<Self>, GraphError> {
        let view = weights.weight(role)?;
        Ok(matches!(view.holding, ViewHolding::Streamed(StreamedFill::Imported { lands: true })).then(|| Self {
            import: crate::ImportProgramSlot::between(view.upload, view.resident),
            elements: [("E", view.upload), ("U", view.resident)],
        }))
    }

    pub(crate) fn entries(&self) -> LandingEntries<'_, NativeGraphMetadata> {
        LandingEntries {
            notice: &[],
            import: match self.import {
                crate::ImportProgramSlot::Dense { .. } => LandingImport::Dense(&self.elements[..]),
                crate::ImportProgramSlot::Repack { .. } => {
                    LandingImport::Repack(&self.elements[..])
                }
            },
        }
    }
}

/// The ports a landed read binds each run (`DeviceTable::read_selected`):
/// the notice its step copies its selection into and the signal after it,
/// the gate it waits at, the landing staging and landing slots its
/// conversion reads and writes (`[1, rows, ..]`), and its read signal.
#[derive(Clone)]
pub(crate) struct LandedReadPorts {
    /// The rows the read names (`DeviceTable::read_selected`): fixed in a
    /// prepared graph, which alone is queued.
    pub rows: Dim,
    pub notice: NativePort,
    pub noticed: NativeTimelinePort<seismic::timeline::Reached>,
    pub gate: NativeTimelinePort<seismic::timeline::Gate>,
    pub staging: NativePort,
    pub landing: NativePort,
    pub read: NativeTimelinePort<seismic::timeline::Reached>,
}

impl LandedReadPorts {
    /// Queue the read a run makes of `table` and bind its ports.
    pub(crate) fn queue(
        &self,
        table: &crate::device_table::ResidentTable,
        bindings: &mut seismic::NativeGraphBindings,
    ) -> Result<crate::device_table::SelectedRead, String> {
        let read = table
            .device()
            .ok_or("a landed read of a table held whole")?
            .read_selected(self.rows.fixed())?;
        self.bind(bindings, &read)?;
        Ok(read)
    }

    fn bind(
        &self,
        bindings: &mut seismic::NativeGraphBindings,
        read: &crate::device_table::SelectedRead,
    ) -> Result<(), String> {
        let text = |error: &dyn std::fmt::Display| format!("landed read: {error}");
        // The import's view of a run of whole rows: one matrix of them.
        let view = |tensor: &seismic::Tensor| {
            let extents = std::iter::once(1)
                .chain(tensor.extents().iter().copied())
                .collect::<Vec<_>>();
            tensor.reshape(&extents).map_err(|error| text(&error))
        };
        bindings.set(&self.notice, &read.notice).map_err(|error| text(&error))?;
        bindings
            .set(&self.staging, &view(&read.staging)?)
            .map_err(|error| text(&error))?;
        bindings
            .set(&self.landing, &view(&read.landing)?)
            .map_err(|error| text(&error))?;
        bindings
            .set_timeline(self.noticed, &read.noticed, read.value)
            .and_then(|()| bindings.set_timeline(self.gate, &read.gate, read.value))
            .and_then(|()| bindings.set_timeline(self.read, &read.read, read.value))
            .map_err(|error| text(&error))
    }
}

/// The ports a read of a stored table binds each run (`stored_table_rows`):
/// the gate it waits at until the host has written every row it names, and
/// the signal after it, which says the read is done with the rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoredReadPorts {
    pub gate: NativeTimelinePort<seismic::timeline::Gate>,
    pub read: NativeTimelinePort<seismic::timeline::Reached>,
}

/// The rows `rows` (`[count, 2]`, (row, status) pairs) of the stored table
/// `table` (`TableFill::Stored`), decoded to activations `[count, width]`:
/// behind a gate the step's rows open (`StoredTable`), read through the
/// table's directory (`stored_rows`), then a signal that the read is done.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stored_table_rows<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entry: G::Binding<'a, stored_rows::Entry>,
    table: &mut GatheredTable,
    weights: &(impl PlannedWeights + ?Sized),
    ports: &mut Vec<(WeightPort, NativePort)>,
    rows: &WorkflowTensor,
    count: &Dim,
) -> Result<(WorkflowTensor, StoredReadPorts), GraphError> {
    let view = weights.weight(table.role)?;
    if !matches!(view.holding, ViewHolding::Streamed(StreamedFill::Stored)) {
        return Err(format!("planned weight {:?} is not a stored table", table.role).into());
    }
    let [records, record_rows, width] = &view.shape[..] else {
        return Err(format!("stored table {:?} is not records of rows", table.role).into());
    };
    let directory = table.directory(graph, weights, ports)?;
    let gate = graph.timeline(seismic::timeline::Wait);
    let decoded = graph
        .enqueue(
            entry,
            &[
                ("M", count),
                ("C", records),
                ("P", record_rows),
                ("W", width),
                ("R", &directory.shape[0]),
                ("F", &directory.shape[1]),
            ],
            stored_rows::WorkflowArgs {
                slots: (&table.table).into(),
                tokens: rows.into(),
                directory: (&directory.tensor).into(),
            },
        )?
        .value;
    let read = graph.timeline(seismic::timeline::Signal);
    Ok((decoded, StoredReadPorts { gate, read }))
}

/// The port of a weight an entry binds without an accumulator-scale port
/// (plan admission refused a scaled one: `PlanError::UnportedScale`).
pub(crate) fn weight<G: GraphDraft>(
    graph: &mut G,
    weights: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
) -> Result<seismic::WorkflowTensor, GraphError> {
    weight_port(
        graph,
        weights,
        WeightRole { scope, kind },
        WeightPart::Values,
        ports,
    )
    .map(|port| port.tensor().clone())
}

/// A projection weight and its second-level scale.
pub(crate) struct ScaledWeight {
    pub weight: WorkflowTensor,
    pub scale: WeightScale,
}

/// A weight's second-level scale as an entry's accumulator-scale port binds
/// it (`[S] f32`): its resident scale and the S values it holds (1 per
/// tensor), or the graph's absent operand at S = 0 where the weight has
/// none (`GraphConstant::absent`: never read).
#[derive(Clone)]
pub(crate) struct WeightScale {
    pub tensor: WorkflowTensor,
    pub extent: Dim,
}

impl WeightScale {
    /// The second-level scale of the planned weight of `role`.
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let (port, extent) = weight_scale_port(graph, weights, role, ports, constants)?;
        Ok(Self {
            tensor: port.tensor().clone(),
            extent,
        })
    }

    /// The per-expert second-level scales of the stacked expert weight of
    /// `role` (a routed entry's per-expert scale port: E values), or the
    /// absent operand where it has none.
    pub(crate) fn per_expert<G: GraphDraft>(
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        match weights.weight(role)?.scale {
            None => Self::absent(graph, constants),
            Some(extent) => Ok(Self {
                tensor: weight_port(graph, weights, role, WeightPart::Scale, ports)?
                    .tensor()
                    .clone(),
                extent,
            }),
        }
    }

    /// No scale: the graph's absent operand.
    pub(crate) fn absent<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            tensor: GraphConstant::absent(graph, constants)?.tensor().clone(),
            extent: Dim::from(0),
        })
    }
}

/// The operands a projection entry binds beside a weight's values: its
/// second-level scale and its packing bound.
#[derive(Clone)]
pub(crate) struct WeightOperands {
    pub scale: WeightScale,
    pub bound: PackingBound,
}

impl WeightOperands {
    /// The operands of the planned weight of `role`.
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            scale: WeightScale::planned(graph, weights, role, ports, constants)?,
            bound: PackingBound::planned(graph, weights, role, ports, constants)?,
        })
    }

    /// Neither: an absent segment's.
    pub(crate) fn absent<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            scale: WeightScale::absent(graph, constants)?,
            bound: PackingBound::absent(graph, constants)?,
        })
    }
}

/// A weight's packing bound as the bound port of an entry with a packed
/// form binds it (`WeightPlan::bound`, `[B] f32`): its resident bound and
/// the B runs it holds, or the graph's absent operand at B = 0 where the
/// weight has none.
#[derive(Clone)]
pub(crate) struct PackingBound {
    pub tensor: WorkflowTensor,
    pub extent: Dim,
}

impl PackingBound {
    /// The packing bound of the planned weight of `role`.
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        weights: &(impl PlannedWeights + ?Sized),
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        match weights.weight(role)?.bound {
            None => Self::absent(graph, constants),
            Some(extent) => Ok(Self {
                tensor: weight_port(graph, weights, role, WeightPart::Bound, ports)?
                    .tensor()
                    .clone(),
                extent,
            }),
        }
    }

    /// No bound: the graph's absent operand.
    pub(crate) fn absent<G: GraphDraft>(
        graph: &mut G,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            tensor: GraphConstant::absent(graph, constants)?.tensor().clone(),
            extent: Dim::from(0),
        })
    }
}

/// A projection weight an entry with a packed form binds: its values and
/// their operands (second-level scale and packing bound).
#[derive(Clone)]
pub(crate) struct PackedWeight {
    pub weight: WorkflowTensor,
    pub operands: WeightOperands,
}

/// A head projection's ports registered as the weight parts they bind
/// (`StoredHead::ports`), an absent scale as the absent constant.
pub(crate) struct RegisteredPorts<'p, W: PlannedWeights + ?Sized> {
    pub weights: &'p W,
    pub ports: &'p mut Vec<(WeightPort, NativePort)>,
    pub constants: &'p mut Vec<GraphConstant>,
}

impl<G: GraphDraft, W: PlannedWeights + ?Sized> crate::programs::graph::readout::ProjectionBinding<G>
    for RegisteredPorts<'_, W>
{
    fn weight(&mut self, graph: &mut G, role: WeightRole) -> Result<NativePort, GraphError> {
        weight_port(graph, self.weights, role, WeightPart::Values, self.ports)
    }

    fn scale(&mut self, graph: &mut G, role: WeightRole) -> Result<(NativePort, Dim), GraphError> {
        weight_scale_port(graph, self.weights, role, self.ports, self.constants)
    }
}

/// The port of a weight an entry binds with a per-tensor accumulator-scale
/// port (`[S] f32`, S ∈ {0, 1}).
pub(crate) fn scaled_weight<G: GraphDraft>(
    graph: &mut G,
    weights: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<ScaledWeight, GraphError> {
    let role = WeightRole { scope, kind };
    Ok(ScaledWeight {
        weight: weight_port(graph, weights, role, WeightPart::Values, ports)?
            .tensor()
            .clone(),
        scale: WeightScale::planned(graph, weights, role, ports, constants)?,
    })
}

/// The weight of an entry with a packed form: its values, second-level scale
/// and packing bound.
pub(crate) fn packed_weight<G: GraphDraft>(
    graph: &mut G,
    weights: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<PackedWeight, GraphError> {
    let role = WeightRole { scope, kind };
    Ok(PackedWeight {
        weight: weight_port(graph, weights, role, WeightPart::Values, ports)?
            .tensor()
            .clone(),
        operands: WeightOperands::planned(graph, weights, role, ports, constants)?,
    })
}

/// The same scope-owned scaled operand as graph ports, for common projection
/// builders. Registration and absent-scale lifetime have one authority.
pub(crate) fn scaled_weight_ports<G: GraphDraft>(
    graph: &mut G,
    weights: &(impl PlannedWeights + ?Sized),
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<(NativePort, NativePort, Dim), GraphError> {
    let role = WeightRole { scope, kind };
    let weight = weight_port(graph, weights, role, WeightPart::Values, ports)?;
    let (scale, extent) = weight_scale_port(graph, weights, role, ports, constants)?;
    Ok((weight, scale, extent))
}

/// Register the accumulator-scale operand, including its absent lifetime.
pub(crate) fn weight_scale_port<G: GraphDraft>(
    graph: &mut G,
    weights: &(impl PlannedWeights + ?Sized),
    role: WeightRole,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<(NativePort, Dim), GraphError> {
    match weights.weight(role)?.scale {
        None => Ok((GraphConstant::absent(graph, constants)?, Dim::from(0))),
        Some(extent) if extent.equals(1) => Ok((weight_port(graph, weights, role, WeightPart::Scale, ports)?, extent)),
        Some(_) => Err(format!("{role:?} has a per-matrix scale at a per-tensor port").into()),
    }
}

pub(crate) fn activation(geometry: &Decoder) -> Element {
    match geometry.activation_dtype {
        magnitude_family_contracts::ActivationDType::F16 => Element::f16(),
        magnitude_family_contracts::ActivationDType::BF16 => Element::bf16(),
    }
}
