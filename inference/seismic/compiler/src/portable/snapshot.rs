//! Preserve the captured contents of computed tensor values before a conflicting
//! source write. The source events choose the point; actual storage chooses the
//! dependency. Selected products are realized only in their selected arm.
use super::*;
use seismic_lang::entry::SemanticEventKind;

type Roots = BTreeMap<SemanticValueId, Vec<PortableTensor>>;

pub(super) fn stored_roots(value: &SegmentBound, roots: &mut Vec<PortableTensor>) {
    match value {
        SegmentBound::Tensor(tensor) => match &tensor.value {
            TensorDefinitionValue::Physical(place) => roots.push(place.tensor.clone()),
            TensorDefinitionValue::View { base, .. } => {
                stored_roots(&SegmentBound::Tensor((**base).clone()), roots)
            }
            TensorDefinitionValue::Selected {
                then, otherwise, ..
            } => {
                stored_roots(&SegmentBound::Tensor((**then).clone()), roots);
                stored_roots(&SegmentBound::Tensor((**otherwise).clone()), roots);
            }
            // A computed result denotes its own contents, never a writable
            // alias of the inputs used to produce those contents.
            TensorDefinitionValue::Elementwise { .. } | TensorDefinitionValue::Reduce { .. } => {}
        },
        SegmentBound::Tuple(values) => values.iter().for_each(|value| stored_roots(value, roots)),
        _ => {}
    }
}

/// Follow checked view/call/region products to the actual products already in
/// the current environment. Fresh allocations cannot alias a pre-existing
/// captured value. This is a query over the source graph, not another effect IR.
fn roots_of<'f>(
    function: &'f SemanticFunction,
    value: SemanticValueId,
    environment: &Roots,
    helpers: &BTreeMap<FamilyId, &'f SemanticFunction>,
    seen: &mut BTreeSet<SemanticValueId>,
) -> Vec<PortableTensor> {
    if let Some(roots) = environment.get(&value) {
        return roots.clone();
    }
    if !seen.insert(value) {
        return Vec::new();
    }
    let ValueOrigin::Node(node) = function.value(value).origin else {
        return Vec::new();
    };
    let resolve = |id, seen: &mut BTreeSet<_>| roots_of(function, id, environment, helpers, seen);
    match function.node(node).view() {
        SemanticNodeView::View { base, .. } => resolve(base, seen),
        SemanticNodeView::ElementWrite { place, .. } | SemanticNodeView::Atomic { place, .. } => {
            resolve(place, seen)
        }
        SemanticNodeView::Store { destination, .. } => resolve(destination, seen),
        SemanticNodeView::TuplePack { inputs, .. } => {
            inputs.iter().flat_map(|id| resolve(*id, seen)).collect()
        }
        SemanticNodeView::TupleGet { tuple, .. } => resolve(tuple, seen),
        SemanticNodeView::Call {
            family,
            inputs,
            outputs,
        } => {
            let helper = helpers[&family];
            let mut nested = environment.clone();
            for (parameter, input) in helper.parameters().iter().zip(inputs) {
                nested.insert(parameter.value, resolve(*input, &mut seen.clone()));
            }
            let ordinal = outputs
                .iter()
                .position(|id| *id == value)
                .expect("call result belongs to its node");
            roots_of(helper, helper.results()[ordinal], &nested, helpers, seen)
        }
        SemanticNodeView::If {
            captures,
            outputs,
            then,
            otherwise,
            ..
        } => {
            let ordinal = outputs
                .iter()
                .position(|id| *id == value)
                .expect("branch result belongs to its node");
            [then, otherwise]
                .into_iter()
                .flat_map(|region| {
                    let nested = region_roots(function, region, captures, environment, helpers);
                    roots_of(
                        function,
                        function.region(region).results()[ordinal],
                        &nested,
                        helpers,
                        &mut seen.clone(),
                    )
                })
                .collect()
        }
        SemanticNodeView::Loop {
            captures,
            body,
            carries,
            ..
        } => {
            let mut nested = region_roots(function, body, captures, environment, helpers);
            for carry in carries {
                nested.insert(carry.parameter, resolve(carry.initial, &mut seen.clone()));
            }
            let carry = carries
                .iter()
                .find(|carry| carry.result == value)
                .expect("loop result owns its carry");
            let mut roots = resolve(carry.initial, &mut seen.clone());
            roots.extend(roots_of(function, carry.yielded, &nested, helpers, seen));
            roots
        }
        _ => Vec::new(),
    }
}

fn region_roots<'f>(
    function: &'f SemanticFunction,
    region: RegionId,
    captures: &[SemanticValueId],
    environment: &Roots,
    helpers: &BTreeMap<FamilyId, &'f SemanticFunction>,
) -> Roots {
    let mut nested = environment.clone();
    let parameters = function.region(region).parameters();
    let parameters = if matches!(function.region(region).kind(), RegionKind::LoopBody { .. }) {
        &parameters[1..]
    } else {
        parameters
    };
    for (parameter, input) in parameters.iter().zip(captures) {
        nested.insert(
            *parameter,
            roots_of(function, *input, environment, helpers, &mut BTreeSet::new()),
        );
    }
    nested
}

fn writes<'f>(
    function: &'f SemanticFunction,
    node: NodeId,
    environment: &Roots,
    helpers: &BTreeMap<FamilyId, &'f SemanticFunction>,
    output: &mut Vec<PortableTensor>,
) {
    let node = function.node(node);
    for event in node.events() {
        if matches!(
            event.kind(),
            SemanticEventKind::Write(_) | SemanticEventKind::AtomicRmw { .. }
        ) {
            if let Some(place) = event.place() {
                output.extend(roots_of(
                    function,
                    place,
                    environment,
                    helpers,
                    &mut BTreeSet::new(),
                ));
            }
        }
    }
    match node.view() {
        SemanticNodeView::Call { family, inputs, .. } => {
            let helper = helpers[&family];
            let mut nested = environment.clone();
            for (parameter, input) in helper.parameters().iter().zip(inputs) {
                nested.insert(
                    parameter.value,
                    roots_of(function, *input, environment, helpers, &mut BTreeSet::new()),
                );
            }
            for (id, _) in helper.nodes(helper.root()) {
                writes(helper, id, &nested, helpers, output);
            }
        }
        SemanticNodeView::If {
            captures,
            then,
            otherwise,
            ..
        } => {
            for region in [then, otherwise] {
                let nested = region_roots(function, region, captures, environment, helpers);
                for (id, _) in function.nodes(region) {
                    writes(function, id, &nested, helpers, output);
                }
            }
        }
        SemanticNodeView::Loop {
            captures,
            body,
            carries,
            ..
        } => {
            let mut nested = region_roots(function, body, captures, environment, helpers);
            // Every initial and yielded storage alternative is an actual source
            // carry dependency. Close their finite root union across backedges.
            for carry in carries {
                nested.insert(
                    carry.parameter,
                    roots_of(
                        function,
                        carry.initial,
                        environment,
                        helpers,
                        &mut BTreeSet::new(),
                    ),
                );
            }
            // A simple source path can cross at most one edge per carried
            // product before repeating a root; roots_of itself cuts cycles.
            for _ in 0..carries.len() {
                let next = carries
                    .iter()
                    .map(|carry| {
                        (
                            carry.parameter,
                            roots_of(
                                function,
                                carry.yielded,
                                &nested,
                                helpers,
                                &mut BTreeSet::new(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                for (parameter, roots) in next {
                    nested.entry(parameter).or_default().extend(roots);
                }
            }
            for (id, _) in function.nodes(body) {
                writes(function, id, &nested, helpers, output);
            }
        }
        _ => {}
    }
}

impl SegmentTensor {
    pub(super) fn reads_overlap<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &PortableBuilder<'_, B>,
        writes: &[PortableTensor],
    ) -> bool {
        match &self.value {
            TensorDefinitionValue::Physical(tensor) => writes
                .iter()
                .any(|write| kernel.tensors_may_overlap(&tensor.tensor, write)),
            TensorDefinitionValue::Elementwise { inputs, .. } => {
                inputs.iter().any(|value| match value {
                    SegmentBound::Tensor(tensor) => tensor.reads_overlap(kernel, writes),
                    _ => false,
                })
            }
            TensorDefinitionValue::Reduce { input, .. } => input.reads_overlap(kernel, writes),
            TensorDefinitionValue::View { base, .. } => base.reads_overlap(kernel, writes),
            TensorDefinitionValue::Selected {
                then, otherwise, ..
            } => then.reads_overlap(kernel, writes) || otherwise.reads_overlap(kernel, writes),
        }
    }

    fn needs_snapshot<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &PortableBuilder<'_, B>,
        writes: &[PortableTensor],
    ) -> bool {
        match &self.value {
            TensorDefinitionValue::Physical(_) => false,
            TensorDefinitionValue::View { base, .. } => base.needs_snapshot(kernel, writes),
            TensorDefinitionValue::Selected {
                then, otherwise, ..
            } => then.needs_snapshot(kernel, writes) || otherwise.needs_snapshot(kernel, writes),
            _ => self.reads_overlap(kernel, writes),
        }
    }

    fn require_snapshot_capacity<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &PortableBuilder<'_, B>,
        writes: &[PortableTensor],
    ) -> Result<(), capacity::CapacityPending> {
        if !self.needs_snapshot(kernel, writes) {
            return Ok(());
        }
        match &self.value {
            TensorDefinitionValue::Elementwise { result, .. }
            | TensorDefinitionValue::Reduce { result, .. } => result.require_capacity(),
            TensorDefinitionValue::View { base, .. } => {
                base.require_snapshot_capacity(kernel, writes)
            }
            TensorDefinitionValue::Selected {
                then, otherwise, ..
            } => {
                then.require_snapshot_capacity(kernel, writes)?;
                otherwise.require_snapshot_capacity(kernel, writes)
            }
            TensorDefinitionValue::Physical(_) => Ok(()),
        }
    }

    fn snapshot<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        writes: &[PortableTensor],
    ) -> Self {
        if !self.needs_snapshot(kernel, writes) {
            return self.clone();
        }
        match &self.value {
            TensorDefinitionValue::Elementwise { result, .. }
            | TensorDefinitionValue::Reduce { result, .. } => {
                let destination = result.snapshot_storage(kernel, &self.axes);
                let dtype = element_dtype(result.tensor.representation);
                segment_copy_tensor(
                    kernel,
                    self,
                    &destination,
                    dtype,
                    &self.axes,
                    0,
                    &mut Vec::new(),
                );
                Self::physical(kernel, SegmentStorage::source(result.clone(), destination))
            }
            TensorDefinitionValue::View { base, transform } => Self {
                axes: self.axes.clone(),
                value: TensorDefinitionValue::View {
                    base: Arc::new(base.snapshot(kernel, writes)),
                    transform: transform.clone(),
                },
            },
            TensorDefinitionValue::Selected {
                condition,
                then,
                otherwise,
            } => {
                let branch = kernel.begin_branch(*condition);
                let then = SegmentBound::Tensor(then.snapshot(kernel, writes));
                kernel.begin_otherwise(&branch);
                let otherwise = SegmentBound::Tensor(otherwise.snapshot(kernel, writes));
                let mut fields = Vec::new();
                products::join_fields(&then, &otherwise, &mut fields);
                let mut fields = kernel.finish_branch_products(branch, fields).into_iter();
                let SegmentBound::Tensor(value) =
                    products::joined(kernel, *condition, &then, &otherwise, &mut fields)
                else {
                    unreachable!()
                };
                assert!(fields.next().is_none());
                value
            }
            TensorDefinitionValue::Physical(_) => {
                unreachable!("stored aliases do not snapshot themselves")
            }
        }
    }
}

impl SegmentBound {
    fn needs_snapshot<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &PortableBuilder<'_, B>,
        writes: &[PortableTensor],
    ) -> bool {
        match self {
            Self::Tensor(tensor) => tensor.needs_snapshot(kernel, writes),
            Self::Tuple(values) => values
                .iter()
                .any(|value| value.needs_snapshot(kernel, writes)),
            _ => false,
        }
    }
    fn require_snapshot_capacity<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &PortableBuilder<'_, B>,
        writes: &[PortableTensor],
    ) -> Result<(), capacity::CapacityPending> {
        match self {
            Self::Tensor(tensor) => tensor.require_snapshot_capacity(kernel, writes),
            Self::Tuple(values) => {
                for value in values {
                    value.require_snapshot_capacity(kernel, writes)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    fn snapshot<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        writes: &[PortableTensor],
    ) -> Self {
        match self {
            Self::Tensor(tensor) => Self::Tensor(tensor.snapshot(kernel, writes)),
            Self::Tuple(values) => Self::Tuple(
                values
                    .iter()
                    .map(|value| value.snapshot(kernel, writes))
                    .collect(),
            ),
            _ => self.clone(),
        }
    }
}

impl<'s, 'k, 'f, 'r, B: seismic_native_target::TargetFamily> SegmentLowerer<'s, 'k, 'f, 'r, B> {
    pub(super) fn preserve_captures_before(
        &mut self,
        node: NodeId,
    ) -> Result<(), capacity::CapacityPending> {
        let environment = self
            .values
            .iter()
            .map(|(id, value)| {
                let mut roots = Vec::new();
                stored_roots(value, &mut roots);
                (*id, roots)
            })
            .collect();
        let mut destinations = Vec::new();
        writes(
            self.function,
            node,
            &environment,
            self.helpers,
            &mut destinations,
        );
        if destinations.is_empty() {
            return Ok(());
        }
        let captures = self
            .values
            .iter()
            .filter(|(_, value)| value.needs_snapshot(self.kernel, &destinations))
            .map(|(id, value)| (*id, value.clone()))
            .collect::<Vec<_>>();
        if captures.is_empty() {
            return Ok(());
        }
        for (_, value) in &captures {
            value.require_snapshot_capacity(self.kernel, &destinations)?;
        }
        let branch = self.kernel.begin_branch(self.alive);
        let snapshots = captures
            .iter()
            .map(|(id, value)| (*id, value.snapshot(self.kernel, &destinations)))
            .collect::<Vec<_>>();
        let fields = snapshots
            .iter()
            .flat_map(|(_, value)| value.scalar_fields())
            .map(|value| (Some(value), None))
            .collect();
        self.kernel.begin_otherwise(&branch);
        let mut fields = self
            .kernel
            .finish_branch_products(branch, fields)
            .into_iter();
        for (id, value) in snapshots {
            self.values
                .insert(id, value.with_scalar_fields(self.kernel, &mut fields));
        }
        assert!(fields.next().is_none());
        Ok(())
    }
}
