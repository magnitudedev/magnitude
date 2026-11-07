use crate::{
    ModelLoadPlan,
    programs::{
        graph::{GraphError, draft::GraphDraft},
        native_constants::GraphConstant,
    },
};
use magnitude_family_contracts::{Decoder, WeightKind, WeightRole, WeightScope};
use magnitude_kernels::{import_dense, repack_weight, selection_notice, stored_rows};
use seismic::{
    Element, NativeGraphMetadata, NativePort, NativeTimelinePort, WorkflowTensor,
};

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
    bank: &crate::ExpertBankPlan,
    scope: WeightScope,
    part: crate::planning::BankPart,
    ports: &mut Vec<(BankPort, NativePort)>,
) -> Result<NativePort, GraphError> {
    let (layer, _) = bank
        .layer(scope)
        .ok_or_else(|| format!("{scope:?} is not a streamed routed layer"))?;
    let (element, extents) = bank.part_shape(layer, part)?;
    let port = graph.port(element, &extents)?;
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
    load: &ModelLoadPlan,
    role: WeightRole,
    part: WeightPart,
    ports: &mut Vec<(WeightPort, NativePort)>,
) -> Result<NativePort, GraphError> {
    let plan = load
        .weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| format!("missing planned weight {role:?}"))?;
    let port = match (part, &plan.scale) {
        (WeightPart::Values, _) => graph.port(plan.resident, &plan.shape),
        (WeightPart::Scale, Some(scale)) => graph.port(Element::f32(), &[scale.extent]),
        (WeightPart::Scale, None) => {
            return Err(format!("planned weight {role:?} has no second-level scale").into());
        }
        (WeightPart::Bound, _) if plan.bound != 0 => graph.port(Element::f32(), &[plan.bound]),
        (WeightPart::Bound, _) => {
            return Err(format!("planned weight {role:?} has no packing bound").into());
        }
        (WeightPart::Directory, _) => match plan.streamed() {
            Some(_) => graph.port(Element::i32(), &plan.directory_shape()),
            None => {
                return Err(
                    format!("planned weight {role:?} is held whole and has no directory").into(),
                );
            }
        },
        // The group's records' shape at its layer's experts.
        (WeightPart::Home, _) => match (plan.banked(), load.expert_bank().and_then(|bank| bank.layer(role.scope))) {
            (Some(_), Some((_, layer))) => {
                graph.port(plan.resident, &[&[layer.experts][..], &plan.shape[1..]].concat())
            }
            _ => return Err(format!("planned weight {role:?} is no banked stack and has no home records").into()),
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
    pub shape: [u64; 2],
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
            shape: [0, 1],
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
    pub rows: u64,
    role: WeightRole,
    /// The shape of a streamed table's directory; `[0, 1]` of a table held
    /// whole.
    directory_shape: [u64; 2],
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
        load: &ModelLoadPlan,
        scope: WeightScope,
        kind: WeightKind,
        ports: &mut Vec<(WeightPort, NativePort)>,
    ) -> Result<Self, GraphError> {
        let role = WeightRole { scope, kind };
        let plan = load
            .weights()
            .find(|plan| plan.role == role)
            .ok_or_else(|| format!("missing planned weight {role:?}"))?;
        let rows = match plan.streamed() {
            Some(_) => plan.table_slots(),
            None => plan.shape[0],
        };
        let table = weight_port(graph, load, role, WeightPart::Values, ports)?
            .tensor()
            .clone();
        Ok(Self {
            table,
            rows,
            role,
            directory_shape: plan.directory_shape(),
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
        load: &ModelLoadPlan,
        ports: &mut Vec<(WeightPort, NativePort)>,
    ) -> Result<RowDirectory, GraphError> {
        if let Some(directory) = &self.directory {
            return Ok(directory.clone());
        }
        let tensor = weight_port(graph, load, self.role, WeightPart::Directory, ports)?
            .tensor()
            .clone();
        Ok(self
            .directory
            .insert(RowDirectory {
                tensor,
                shape: self.directory_shape,
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
        load: &ModelLoadPlan,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
        landing: Option<LandingEntries<'a, G>>,
        selection: seismic::WorkflowTensorRef<'_>,
        rows: u64,
        class: Option<&'static str>,
        reader: impl FnOnce(&mut G, &RowDirectory) -> Result<T, GraphError>,
    ) -> Result<(T, Option<LandedReadPorts>), GraphError> {
        let plan = load
            .weights()
            .find(|plan| plan.role == self.role)
            .ok_or_else(|| format!("missing planned weight {:?}", self.role))?;
        if !reads_landed(plan)? {
            let directory = self.by_rows(graph, constants)?;
            return Ok((reader(graph, &directory)?, None));
        }
        let landing =
            landing.ok_or_else(|| format!("a landed read of {:?} without its entries", self.role))?;
        let directory = self.directory(graph, load, ports)?;
        let (value, ports) = landed_read(graph, plan, landing, selection, rows, class, |graph| {
            reader(graph, &directory)
        })?;
        Ok((value, Some(ports)))
    }
}

/// Whether a read of rows the device selected from `plan`'s table is
/// landed: a table held whole is read as it is, and an imported streamed
/// table only through its landing slots.
pub(super) fn reads_landed(plan: &crate::WeightPlan) -> Result<bool, GraphError> {
    match (plan.streamed().is_some(), plan.lands_selected_rows()) {
        (false, _) => Ok(false),
        (true, true) => Ok(true),
        (true, false) => Err(format!(
            "planned weight {:?} has no landing slots for a read of selected rows",
            plan.role
        )
        .into()),
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
    plan: &crate::WeightPlan,
    landing: LandingEntries<'a, G>,
    selection: seismic::WorkflowTensorRef<'_>,
    rows: u64,
    class: Option<&'static str>,
    reader: impl FnOnce(&mut G) -> Result<T, GraphError>,
) -> Result<(T, LandedReadPorts), GraphError> {
    let port = |graph: &mut G, element, extents: &[u64], axis| match class {
        Some(dimension) => graph.port_with_class_extent(element, extents, axis, dimension),
        None => graph.port(element, extents),
    };
    let mut notice = port(graph, Element::i32(), &[rows, 2], 0)?;
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
    let width = plan.shape[1];
    let staging = port(graph, plan.upload, &[1, rows, width], 1)?;
    let landed = port(graph, plan.resident, &[1, rows, width], 1)?;
    let dimensions = [("B", 1), ("N", rows), ("K", width)];
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
            rows,
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
pub(crate) struct CheckedLandingEntries {
    import: crate::ImportProgramSlot,
    elements: [(&'static str, Element); 2],
}

impl CheckedLandingEntries {
    pub(crate) fn of(load: &ModelLoadPlan, role: WeightRole) -> Result<Option<Self>, String> {
        let plan = load
            .weights()
            .find(|plan| plan.role == role)
            .ok_or_else(|| format!("missing planned weight {role:?}"))?;
        Ok(plan.lands_selected_rows().then(|| Self {
            import: crate::ImportProgramSlot::of(plan),
            elements: [("E", plan.upload), ("U", plan.resident)],
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
    /// The rows the read names (`DeviceTable::read_selected`).
    pub rows: u64,
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
            .read_selected(self.rows)?;
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
    load: &ModelLoadPlan,
    ports: &mut Vec<(WeightPort, NativePort)>,
    rows: &WorkflowTensor,
    count: u64,
) -> Result<(WorkflowTensor, StoredReadPorts), GraphError> {
    let plan = load
        .weights()
        .find(|plan| plan.role == table.role)
        .ok_or_else(|| format!("missing planned weight {:?}", table.role))?;
    if !plan.stored_in_slots() {
        return Err(format!("planned weight {:?} is not a stored table", table.role).into());
    }
    let [records, record_rows, width] = plan.shape[..] else {
        return Err(format!("stored table {:?} is not records of rows", table.role).into());
    };
    let directory = table.directory(graph, load, ports)?;
    let gate = graph.timeline(seismic::timeline::Wait);
    let decoded = graph
        .enqueue(
            entry,
            &[
                ("M", count),
                ("C", records),
                ("P", record_rows),
                ("W", width),
                ("R", directory.shape[0]),
                ("F", directory.shape[1]),
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
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
) -> Result<seismic::WorkflowTensor, GraphError> {
    weight_port(
        graph,
        load,
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
    pub extent: u64,
}

impl WeightScale {
    /// The second-level scale of the planned weight of `role`.
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        load: &ModelLoadPlan,
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let (port, extent) = weight_scale_port(graph, load, role, ports, constants)?;
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
        load: &ModelLoadPlan,
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        match planned_scale_extent(load, role)? {
            0 => Self::absent(graph, constants),
            extent => Ok(Self {
                tensor: weight_port(graph, load, role, WeightPart::Scale, ports)?
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
            extent: 0,
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
        load: &ModelLoadPlan,
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            scale: WeightScale::planned(graph, load, role, ports, constants)?,
            bound: PackingBound::planned(graph, load, role, ports, constants)?,
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
    pub extent: u64,
}

impl PackingBound {
    /// The packing bound of the planned weight of `role`.
    pub(crate) fn planned<G: GraphDraft>(
        graph: &mut G,
        load: &ModelLoadPlan,
        role: WeightRole,
        ports: &mut Vec<(WeightPort, NativePort)>,
        constants: &mut Vec<GraphConstant>,
    ) -> Result<Self, GraphError> {
        let extent = load
            .weights()
            .find(|plan| plan.role == role)
            .map(|plan| plan.bound)
            .ok_or_else(|| format!("missing planned weight {role:?}"))?;
        match extent {
            0 => Self::absent(graph, constants),
            _ => Ok(Self {
                tensor: weight_port(graph, load, role, WeightPart::Bound, ports)?
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
            extent: 0,
        })
    }
}

/// A projection weight an entry with a packed form binds: its values and
/// their operands (second-level scale and packing bound).
pub(crate) struct PackedWeight {
    pub weight: WorkflowTensor,
    pub operands: WeightOperands,
}

/// Register one scope-owned readout representation and its scale lifetime.
pub(crate) fn projection_ports<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    certified: bool,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<crate::programs::graph::readout::ProjectionPorts, GraphError> {
    use crate::programs::graph::readout::{HeadPlans, ProjectionPorts};
    Ok(match HeadPlans::of(load, scope)?.selected() {
        HeadPlans::Packed(_) => {
            let (weight, scale, scale_extent) =
                scaled_weight_ports(graph, load, scope, WeightKind::Output, ports, constants)?;
            ProjectionPorts::Packed {
                weight,
                scale,
                scale_extent,
            }
        }
        plans @ HeadPlans::Progressive(_) => {
            let depth = plans.depth().ok_or("progressive projection has no depth")?;
            let planes = magnitude_family_contracts::ProgressivePlane::ALL
                .into_iter()
                .filter(|plane| {
                    certified || *plane != magnitude_family_contracts::ProgressivePlane::Radius
                })
                .map(|plane| {
                    Ok((
                        plane,
                        weight_port(
                            graph,
                            load,
                            WeightRole {
                                scope,
                                kind: WeightKind::OutputPlane(plane),
                            },
                            WeightPart::Values,
                            ports,
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>, GraphError>>()?;
            ProjectionPorts::Progressive { depth, planes }
        }
        HeadPlans::Qualified { .. } => unreachable!("a recipe was selected above"),
    })
}

/// The port of a weight an entry binds with a per-tensor accumulator-scale
/// port (`[S] f32`, S ∈ {0, 1}).
pub(crate) fn scaled_weight<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<ScaledWeight, GraphError> {
    let role = WeightRole { scope, kind };
    Ok(ScaledWeight {
        weight: weight_port(graph, load, role, WeightPart::Values, ports)?
            .tensor()
            .clone(),
        scale: WeightScale::planned(graph, load, role, ports, constants)?,
    })
}

/// The weight of an entry with a packed form: its values, second-level scale
/// and packing bound.
pub(crate) fn packed_weight<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<PackedWeight, GraphError> {
    let role = WeightRole { scope, kind };
    Ok(PackedWeight {
        weight: weight_port(graph, load, role, WeightPart::Values, ports)?
            .tensor()
            .clone(),
        operands: WeightOperands::planned(graph, load, role, ports, constants)?,
    })
}

/// The same scope-owned scaled operand as graph ports, for common projection
/// builders. Registration and absent-scale lifetime have one authority.
pub(crate) fn scaled_weight_ports<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<(NativePort, NativePort, u64), GraphError> {
    let role = WeightRole { scope, kind };
    let weight = weight_port(graph, load, role, WeightPart::Values, ports)?;
    let (scale, extent) = weight_scale_port(graph, load, role, ports, constants)?;
    Ok((weight, scale, extent))
}

/// Register the accumulator-scale operand, including its absent lifetime.
pub(crate) fn weight_scale_port<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    role: WeightRole,
    ports: &mut Vec<(WeightPort, NativePort)>,
    constants: &mut Vec<GraphConstant>,
) -> Result<(NativePort, u64), GraphError> {
    let extent = planned_scale_extent(load, role)?;
    let scale = match extent {
        0 => GraphConstant::absent(graph, constants)?,
        1 => weight_port(graph, load, role, WeightPart::Scale, ports)?,
        _ => return Err(format!("{role:?} has a per-matrix scale at a per-tensor port").into()),
    };
    Ok((scale, extent))
}

fn planned_scale_extent(load: &ModelLoadPlan, role: WeightRole) -> Result<u64, String> {
    load.weights()
        .find(|plan| plan.role == role)
        .map(crate::WeightPlan::scale_extent)
        .ok_or_else(|| format!("missing planned weight {role:?}"))
}

pub(crate) fn activation(geometry: &Decoder) -> Element {
    match geometry.activation_dtype {
        magnitude_family_contracts::ActivationDType::F16 => Element::f16(),
        magnitude_family_contracts::ActivationDType::BF16 => Element::bf16(),
    }
}
