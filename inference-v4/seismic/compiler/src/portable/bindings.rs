//! Construction bindings: the lowered value of each checked source value and
//! the arena that owns their physical realizations.
use super::*;

#[derive(Clone, Debug)]
pub(super) enum Bound {
    Tensor(TensorRealization),
    Scalar(ScalarBinding),
    Range { start: Box<Bound>, end: Box<Bound> },
    Tuple(Vec<Bound>),
    Unit,
}

pub(super) struct LoopConstruction {
    pub(super) schedule: crate::implementation::ScheduleRepeat,
    pub(super) parent: SemanticBindings,
    pub(super) initial: Bound,
    pub(super) start: NatExpr,
    pub(super) end: NatExpr,
    pub(super) carries: Vec<seismic_lang::entry::Carry>,
}

/// The value produced by tensor lowering, independent of whether it has storage.
#[derive(Clone, Debug)]
pub(super) enum TensorRealization {
    Stored(StoredTensor),
    Computed(Arc<StreamTensorPlan>),
}

impl TensorRealization {
    pub(super) fn stored(&self) -> StoredView {
        match self {
            Self::Stored(value) => value.view.clone(),
            Self::Computed(_) => panic!("addressable use was not materialized by lowering"),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum SegmentBound {
    Tensor(SegmentTensor),
    Scalar(PortableValue),
    Opaque(seismic_ir::kernel::ops::SemanticOpaque),
    Range {
        start: Box<SegmentBound>,
        end: Box<SegmentBound>,
    },
    Tuple(Vec<SegmentBound>),
    Unit,
}

/// One tensor producer definition, instantiated either with construction
/// bindings or with a kernel's physical operands. Instantiation changes the
/// operands, never the operation or its rounding/indexing meaning.
#[derive(Clone, Debug)]
pub(super) struct TensorDefinition<P, V, I, T, C> {
    pub(super) axes: Vec<I>,
    pub(super) value: TensorDefinitionValue<P, V, I, T, C>,
}

#[derive(Clone, Debug)]
pub(super) enum TensorDefinitionValue<P, V, I, T, C> {
    Physical(P),
    Elementwise {
        primitive: PrimitiveId,
        inputs: Vec<V>,
        result: producer::TensorResult<C>,
    },
    Reduce {
        op: ReduceOp,
        axis: usize,
        input_dtype: DType,
        input: Arc<TensorDefinition<P, V, I, T, C>>,
        result: producer::TensorResult<C>,
    },
    View {
        base: Arc<TensorDefinition<P, V, I, T, C>>,
        transform: T,
    },
    Selected {
        condition: C,
        then: Arc<TensorDefinition<P, V, I, T, C>>,
        otherwise: Arc<TensorDefinition<P, V, I, T, C>>,
    },
}

pub(super) type SegmentTensor =
    TensorDefinition<SegmentStorage, SegmentBound, PortableValue, SegmentViewTransform, PortableValue>;

/// Total construction environment for one checked function. Owner and
/// ordinal checks are concentrated here; lowering sites never join raw IDs
/// against an unqualified map.
#[derive(Clone, Debug)]
pub(super) struct SemanticBindings {
    pub(super) function: FunctionId,
    pub(super) values: Vec<Option<BindingId>>,
    pub(super) selections: BindingSelections,
    pub(super) contents: StorageContents,
    pub(super) binders: Vec<seismic_lang::expr::SymbolId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingId {
    pub(super) owner: u64,
    pub(super) index: usize,
}

/// Physical realizations belong to construction, while lexical environments
/// retain only their actual bound handles. Cloning an environment cannot clone
/// a producer or manufacture a new storage identity.
#[derive(Debug)]
pub(crate) struct BindingArena {
    pub(super) owner: u64,
    pub(super) nodes: Vec<BindingValue>,
    pub(super) parameters: Vec<BindingId>,
    pub(super) results: Vec<BindingId>,
    pub(super) parameter_selections: BindingSelections,
    pub(super) entry_contents: StorageContents,
    pub(super) entry_binders: Vec<seismic_lang::expr::SymbolId>,
    pub(super) result_contents: StorageContents,
}

pub(crate) type BindingSelector = seismic_lang::expr::BoolExpr;

pub(crate) type BindingSelections = std::collections::HashMap<BindingSelector, i64>;

pub(crate) fn initialization_context<'a>(
    arena: &'a mut ExprArena,
    selections: &BindingSelections,
    binders: &[seismic_lang::expr::SymbolId],
) -> InitializationContext<'a> {
    let mut context = InitializationContext::new(arena);
    for (selection, value) in selections {
        context.assume(*selection, *value != 0, binders);
    }
    context
}


#[derive(Clone, Debug)]
pub(super) enum BindingValue {
    Value(Bound),
    Selected {
        selector: BindingSelector,
        options: Vec<(i64, BindingId)>,
    },
}

impl Default for BindingArena {
    fn default() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("binding construction owner space exhausted");
        Self {
            owner,
            nodes: Vec::new(),
            parameters: Vec::new(),
            results: Vec::new(),
            parameter_selections: BindingSelections::new(),
            entry_contents: StorageContents::new(),
            entry_binders: Vec::new(),
            result_contents: StorageContents::new(),
        }
    }
}

impl BindingArena {
    pub(super) fn insert(&mut self, bound: Bound) -> BindingId {
        let id = BindingId {
            owner: self.owner,
            index: self.nodes.len(),
        };
        self.nodes.push(BindingValue::Value(bound));
        id
    }

    pub(super) fn get(&self, id: BindingId) -> Bound {
        assert_eq!(
            id.owner, self.owner,
            "binding belongs to another construction"
        );
        match &self.nodes[id.index] {
            BindingValue::Value(value) => value.clone(),
            BindingValue::Selected { .. } => {
                panic!("selected binding was not resolved in its construction region")
            }
        }
    }

    pub(super) fn selected(&self, mut id: BindingId, selections: &BindingSelections) -> BindingId {
        loop {
            assert_eq!(
                id.owner, self.owner,
                "binding belongs to another construction"
            );
            match &self.nodes[id.index] {
                BindingValue::Value(_) => return id,
                BindingValue::Selected { selector, options } => {
                    let value = selections
                        .get(selector)
                        .expect("selected binding has no dominating construction arm");
                    id = options
                        .iter()
                        .find(|(option, _)| option == value)
                        .expect("selected binding option is complete")
                        .1;
                }
            }
        }
    }

    pub(super) fn unresolved(&self, id: BindingId, selections: &BindingSelections) -> Option<BindingSelector> {
        fn tensor(
            arena: &BindingArena,
            plan: &StreamTensorPlan,
            selections: &BindingSelections,
        ) -> Option<BindingSelector> {
            match &plan.value {
                TensorDefinitionValue::Physical(id) => arena.unresolved(*id, selections),
                TensorDefinitionValue::Elementwise { inputs, .. } => {
                    inputs.iter().find_map(|input| match input {
                        StreamBoundPlan::Tensor(plan) => tensor(arena, plan, selections),
                        StreamBoundPlan::Scalar(_) => None,
                    })
                }
                TensorDefinitionValue::Reduce { input, .. } => tensor(arena, input, selections),
                TensorDefinitionValue::View { base, .. } => tensor(arena, base, selections),
                TensorDefinitionValue::Selected { then, otherwise, .. } => tensor(arena, then, selections).or_else(|| tensor(arena, otherwise, selections)),
            }
        }
        fn bound(
            arena: &BindingArena,
            value: &Bound,
            selections: &BindingSelections,
        ) -> Option<BindingSelector> {
            match value {
                Bound::Tensor(TensorRealization::Computed(plan)) => tensor(arena, plan, selections),
                Bound::Range { start, end } => {
                    bound(arena, start, selections).or_else(|| bound(arena, end, selections))
                }
                Bound::Tuple(values) => values
                    .iter()
                    .find_map(|value| bound(arena, value, selections)),
                _ => None,
            }
        }
        assert_eq!(
            id.owner, self.owner,
            "binding belongs to another construction"
        );
        match &self.nodes[id.index] {
            BindingValue::Value(value) => bound(self, value, selections),
            BindingValue::Selected { selector, options } => match selections.get(selector) {
                Some(value) => self.unresolved(
                    options
                        .iter()
                        .find(|(option, _)| option == value)
                        .expect("selected binding option is complete")
                        .1,
                    selections,
                ),
                None => Some(*selector),
            },
        }
    }

    pub(crate) fn selected_value(
        &mut self,
        selector: BindingSelector,
        options: Vec<(i64, BindingId)>,
    ) -> BindingId {
        let first = options.first().expect("selected value has an option").1;
        for (_, value) in &options {
            assert_eq!(value.owner, self.owner);
        }
        if options.iter().all(|(_, value)| *value == first) {
            return first;
        }
        let id = BindingId {
            owner: self.owner,
            index: self.nodes.len(),
        };
        self.nodes.push(BindingValue::Selected { selector, options });
        id
    }
}

/// Immutable construction products. External parameters remain explicit owner
/// edges so consuming import substitutes the caller's binding, including aliases.
#[derive(Debug, Default)]
pub(crate) struct FrozenBindings {
    pub(super) arena: BindingArena,
}

impl BindingArena {
    pub(crate) fn physical_binding(
        &self,
        id: BindingId,
        layout: impl FnOnce(AnyBufferView) -> seismic_ir::storage::BufferViewLayout,
    ) -> Option<PhysicalBinding> {
        Some(
            match self.get(self.selected(id, &self.parameter_selections)) {
                Bound::Tensor(TensorRealization::Computed(_)) => return None,
                Bound::Tensor(value) => {
                    let logical = value.stored();
                    let view = *logical.direct_backing()?;
                    PhysicalBinding::View {
                        view,
                        layout: layout(view),
                    }
                }
                Bound::Scalar(value) => PhysicalBinding::Scalar(value),
                Bound::Range { start, end } => PhysicalBinding::Range {
                    start: start.scalar(),
                    end: end.scalar(),
                },
                Bound::Tuple(_) | Bound::Unit => {
                    panic!("call arguments must be normalized value leaves")
                }
            },
        )
    }
    pub(crate) fn parameter_handles(&self) -> &[BindingId] {
        &self.parameters
    }
    pub(crate) fn import_parameters(
        &mut self,
        source: &BindingArena,
        parameters: &[BindingId],
        selections: &BindingSelections,
        contents: &StorageContents,
        binders: &[seismic_lang::expr::SymbolId],
        physical: &mut impl BindingPhysicalImport,
    ) {
        self.parameters = self.import_bindings(source, parameters, &[], physical);
        self.parameter_selections = selections.clone();
        self.entry_contents = contents.clone();
        self.entry_binders = binders.to_vec();
    }
    pub(crate) fn freeze(self) -> FrozenBindings {
        FrozenBindings { arena: self }
    }
}

/// Rebind physical leaves through the existing owning storage/schedule import.
/// The graph importer alone owns semantic node memoization and substitution.
pub(crate) type BindingPath = Vec<(BindingSelector, i64)>;

pub(crate) trait BindingPhysicalImport {
    fn remap_tensor(&mut self, value: &StoredTensor, path: &BindingPath) -> StoredTensor;
    fn remap_slot(&mut self, slot: AnyScalarSlot, path: &BindingPath) -> AnyScalarSlot;
    fn remap_quantity_slot(&mut self, slot: seismic_ir::schedule::HostQuantitySlot, path: &BindingPath) -> seismic_ir::schedule::HostQuantitySlot;
    fn remap_prepared(&mut self, value: PreparedArg, path: &BindingPath) -> PreparedArg;
    fn remap_axis(&mut self, value: NatExpr, path: &BindingPath) -> NatExpr;
    fn remap_selector(&mut self, value: BindingSelector, path: &BindingPath) -> BindingSelector;
}

impl BindingArena {
    /// Visit the storage retained by complete products, including the captured
    /// operands of deferred producers. Historical source associations are not
    /// live products and do not acquire final initialized contents.
    pub(super) fn visit_stored(&self, products: &[BindingId], visit: &mut impl FnMut(&StoredTensor)) {
        fn tensor(
            value: &StreamTensorPlan,
            arena: &BindingArena,
            seen: &mut std::collections::HashSet<BindingId>,
            visit: &mut impl FnMut(&StoredTensor),
        ) {
            match &value.value {
                TensorDefinitionValue::Physical(id) => node(*id, arena, seen, visit),
                TensorDefinitionValue::Elementwise { inputs, .. } => {
                    for input in inputs {
                        if let StreamBoundPlan::Tensor(input) = input {
                            tensor(input, arena, seen, visit);
                        }
                    }
                }
                TensorDefinitionValue::Reduce { input, .. } => tensor(input, arena, seen, visit),
                TensorDefinitionValue::View { base, .. } => tensor(base, arena, seen, visit),
                TensorDefinitionValue::Selected { then, otherwise, .. } => {
                    tensor(then, arena, seen, visit);
                    tensor(otherwise, arena, seen, visit);
                }
            }
        }
        fn bound(
            value: &Bound,
            arena: &BindingArena,
            seen: &mut std::collections::HashSet<BindingId>,
            visit: &mut impl FnMut(&StoredTensor),
        ) {
            match value {
                Bound::Tensor(TensorRealization::Stored(value)) => visit(value),
                Bound::Tensor(TensorRealization::Computed(value)) => tensor(value, arena, seen, visit),
                Bound::Tuple(values) => {
                    for value in values { bound(value, arena, seen, visit); }
                }
                Bound::Range { start, end } => {
                    bound(start, arena, seen, visit);
                    bound(end, arena, seen, visit);
                }
                Bound::Scalar(_) | Bound::Unit => {}
            }
        }
        fn node(
            id: BindingId,
            arena: &BindingArena,
            seen: &mut std::collections::HashSet<BindingId>,
            visit: &mut impl FnMut(&StoredTensor),
        ) {
            assert_eq!(id.owner, arena.owner);
            if !seen.insert(id) { return; }
            match &arena.nodes[id.index] {
                BindingValue::Value(value) => bound(value, arena, seen, visit),
                BindingValue::Selected { options, .. } => {
                    for (_, id) in options { node(*id, arena, seen, visit); }
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        for id in products { node(*id, self, &mut seen, visit); }
    }

    pub(crate) fn import_bindings(
        &mut self,
        source: &BindingArena,
        results: &[BindingId],
        parameters: &[(BindingId, BindingId)],
        physical: &mut impl BindingPhysicalImport,
    ) -> Vec<BindingId> {
        let parent = self;
        let mut remap = std::collections::HashMap::new();
        for (source, target) in parameters {
            // Verify the actual caller owns each substituted parameter.
            assert_eq!(
                target.owner, parent.owner,
                "call parameter belongs to another construction"
            );
            remap.insert((*source, Vec::new()), *target);
        }
        fn scalar(
            value: ScalarBinding,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> ScalarBinding {
            match value {
                ScalarBinding::Published(slot) => {
                    ScalarBinding::Published(physical.remap_slot(slot,path))
                }
                ScalarBinding::Quantity(slot) => {
                    ScalarBinding::Quantity(physical.remap_quantity_slot(slot,path))
                }
                ScalarBinding::Value { symbol,dtype } => match physical.remap_prepared(PreparedArg::Scalar(symbol,dtype),path) {
                    PreparedArg::Scalar(symbol,dtype)=>ScalarBinding::Value {symbol,dtype},
                    _=>unreachable!("physical remap preserves scalar category"),
                },
                ScalarBinding::Integer(value) => match physical.remap_prepared(PreparedArg::Integer(value),path) {
                    PreparedArg::Integer(value)=>ScalarBinding::Integer(value),
                    _=>unreachable!("physical remap preserves quantity category"),
                },
                ScalarBinding::Index(value)=>ScalarBinding::Index(physical.remap_axis(value,path)),
            }
        }
        fn tensor(
            source: &StreamTensorPlan,
            child: &BindingArena,
            parent: &mut BindingArena,
            remap: &mut std::collections::HashMap<(BindingId, BindingPath), BindingId>,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> Arc<StreamTensorPlan> {
            let value = match &source.value {
                TensorDefinitionValue::Physical(id) => {
                    TensorDefinitionValue::Physical(node(*id, child, parent, remap, physical, path))
                }
                TensorDefinitionValue::Elementwise {
                    primitive,
                    inputs,
                    result,
                } => TensorDefinitionValue::Elementwise {
                    primitive: primitive.clone(),
                    inputs: inputs
                        .iter()
                        .map(|value| match value {
                            StreamBoundPlan::Tensor(value) => StreamBoundPlan::Tensor(tensor(
                                value, child, parent, remap, physical, path,
                            )),
                            StreamBoundPlan::Scalar(value) => StreamBoundPlan::Scalar(physical.remap_prepared(*value,path)),
                        })
                        .collect(),
                    result: result.map_captures(&mut |value|physical.remap_prepared(value,path)),
                },
                TensorDefinitionValue::Reduce {
                    op,
                    axis,
                    input_dtype,
                    input,
                    result,
                } => TensorDefinitionValue::Reduce {
                    op: *op,
                    axis: *axis,
                    input_dtype: *input_dtype,
                    input: tensor(input, child, parent, remap, physical, path),
                    result: result.map_captures(&mut |value|physical.remap_prepared(value,path)),
                },
                TensorDefinitionValue::View { base, transform } => TensorDefinitionValue::View {
                    base: tensor(base, child, parent, remap, physical, path),
                    transform: match transform {
                        StreamViewPlan::Transpose(permutation)=>StreamViewPlan::Transpose(permutation.clone()),
                        StreamViewPlan::Reshape=>StreamViewPlan::Reshape,
                        StreamViewPlan::Slice(axes)=>StreamViewPlan::Slice(axes.iter().map(|axis|match axis {
                            StreamSliceAxisPlan::Full=>StreamSliceAxisPlan::Full,
                            StreamSliceAxisPlan::Point(value)=>StreamSliceAxisPlan::Point(physical.remap_prepared(*value,path)),
                            StreamSliceAxisPlan::Range {start}=>StreamSliceAxisPlan::Range {start:physical.remap_prepared(*start,path)},
                        }).collect()),
                    },
                },
                TensorDefinitionValue::Selected { condition, then, otherwise } => TensorDefinitionValue::Selected {
                    condition: physical.remap_prepared(*condition,path),
                    then: tensor(then, child, parent, remap, physical, path),
                    otherwise: tensor(otherwise, child, parent, remap, physical, path),
                },
            };
            Arc::new(StreamTensorPlan {
                axes: source.axes.iter().map(|axis|physical.remap_axis(*axis,path)).collect(),
                value,
            })
        }
        fn bound(
            value: Bound,
            child: &BindingArena,
            parent: &mut BindingArena,
            remap: &mut std::collections::HashMap<(BindingId, BindingPath), BindingId>,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> Bound {
            match value {
                Bound::Tensor(TensorRealization::Stored(view)) => {
                    Bound::stored(physical.remap_tensor(&view,path))
                }
                Bound::Tensor(TensorRealization::Computed(plan)) => Bound::Tensor(
                    TensorRealization::Computed(tensor(&plan, child, parent, remap, physical, path)),
                ),
                Bound::Scalar(value) => Bound::Scalar(scalar(value, physical, path)),
                Bound::Range { start, end } => Bound::Range {
                    start: Box::new(bound(*start, child, parent, remap, physical, path)),
                    end: Box::new(bound(*end, child, parent, remap, physical, path)),
                },
                Bound::Tuple(values) => Bound::Tuple(
                    values
                        .into_iter()
                        .map(|value| bound(value, child, parent, remap, physical, path))
                        .collect(),
                ),
                Bound::Unit => Bound::Unit,
            }
        }
        fn node(
            id: BindingId,
            child: &BindingArena,
            parent: &mut BindingArena,
            remap: &mut std::collections::HashMap<(BindingId, BindingPath), BindingId>,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> BindingId {
            if let Some(mapped) = remap.get(&(id, Vec::new())).or_else(||remap.get(&(id,path.clone()))) {
                return *mapped;
            }
            assert_eq!(
                id.owner, child.owner,
                "frozen binding belongs to another construction"
            );
            let mapped = match &child.nodes[id.index] {
                BindingValue::Value(value) => {
                    let value = bound(value.clone(), child, parent, remap, physical, path);
                    parent.insert(value)
                }
                BindingValue::Selected { selector, options } => {
                    let options = options
                        .iter()
                        .map(|(value, binding)| {
                            let mut selected=path.clone(); selected.push((*selector,*value));
                            (*value, node(*binding, child, parent, remap, physical, &selected))
                        })
                        .collect();
                    parent.selected_value(physical.remap_selector(*selector,path), options)
                }
            };
            remap.insert((id,path.clone()), mapped);
            mapped
        }
        results
            .iter()
            .map(|result| node(*result, source, parent, &mut remap, physical, &Vec::new()))
            .collect()
    }
}

impl FrozenBindings {
    pub(crate) fn import_into(
        self,
        parent: &mut BindingArena,
        parameters: &[BindingId],
        physical: &seismic_ir::schedule::ImportedBindings,
        storage: &seismic_ir::storage::TopologyBuilder,
        contents: &mut StorageContents,
    ) -> Vec<BindingId> {
        assert_eq!(
            self.arena.parameters.len(),
            parameters.len(),
            "call binding arity differs"
        );
        struct ScheduleImport<'a> {
            physical: &'a seismic_ir::schedule::ImportedBindings,
            storage: &'a seismic_ir::storage::TopologyBuilder,
        }
        impl BindingPhysicalImport for ScheduleImport<'_> {
            fn remap_tensor(&mut self, value: &StoredTensor, _: &BindingPath) -> StoredTensor {
                let view = value.view.map(|view| self.physical.remap_view(*view), |value| *value);
                let root = self.storage.view_layout(*view.backing()).base;
                StoredTensor { view, root, initialized_view: value.initialized_view.clone() }
            }
            fn remap_slot(&mut self, slot: AnyScalarSlot, _: &BindingPath) -> AnyScalarSlot {
                self.physical.remap_slot(slot)
            }
            fn remap_quantity_slot(&mut self, slot: seismic_ir::schedule::HostQuantitySlot, _: &BindingPath) -> seismic_ir::schedule::HostQuantitySlot {
                self.physical.remap_quantity_slot(slot)
            }
            fn remap_prepared(&mut self, value: PreparedArg, _: &BindingPath) -> PreparedArg { value }
            fn remap_axis(&mut self, value: NatExpr, _: &BindingPath) -> NatExpr { value }
            fn remap_selector(&mut self, value: BindingSelector, _: &BindingPath) -> BindingSelector { value }
        }
        // Only storage retained by escaping products crosses the initialized
        // contents boundary. Parameter effects have already advanced the
        // caller's existing roots through the checked call transfer.
        self.arena.visit_stored(&self.arena.results, &mut |value| {
            let view = physical.remap_view(*value.view.backing());
            let root = storage.view_layout(view).base;
            contents.import_allocation(&self.arena.result_contents, value, root);
        });
        let parameters = self
            .arena
            .parameters
            .iter()
            .copied()
            .zip(parameters.iter().copied())
            .collect::<Vec<_>>();
        // The schedule import already inserts the complete child body,
        // including Unit effects. Only escaping result values need physical
        // remapping into the parent's binding arena.
        parent.import_bindings(
            &self.arena, &self.arena.results, &parameters,
            &mut ScheduleImport { physical, storage },
        )
    }
}

impl SemanticBindings {
    pub(super) fn new(function: &SemanticFunction) -> Self {
        Self {
            function: function.id(),
            values: vec![None; function.values().count()],
            selections: std::collections::HashMap::new(),
            contents: StorageContents::new(),
            binders: Vec::new(),
        }
    }
    pub(super) fn slot(&self, value: SemanticValueId) -> usize {
        assert_eq!(
            value.function(),
            self.function,
            "semantic binding belongs to another function"
        );
        let slot = value.index();
        assert!(
            slot < self.values.len(),
            "semantic binding is outside the function arena"
        );
        slot
    }
    pub(super) fn handle(&self, value: SemanticValueId) -> BindingId {
        self.values[self.slot(value)]
            .expect("checked semantic value was used before its dominating definition")
    }
    pub(super) fn get(&self, arena: &BindingArena, value: SemanticValueId) -> Bound {
        arena.get(arena.selected(self.handle(value), &self.selections))
    }
    pub(super) fn bind(&mut self, arena: &mut BindingArena, value: SemanticValueId, bound: Bound) {
        let slot = self.slot(value);
        let binding = arena.insert(bound);
        assert!(self.values[slot].replace(binding).is_none(), "semantic value was bound twice");
    }
    pub(super) fn rebind(&mut self, arena: &mut BindingArena, value: SemanticValueId, bound: Bound) {
        let slot = self.slot(value);
        // A new residence is scoped to this lexical environment. In particular
        // constructing one branch cannot change its sibling's inherited value.
        let binding = arena.insert(bound);
        self.values[slot] = Some(binding);
    }
    pub(super) fn contains(&self, value: SemanticValueId) -> bool {
        self.values[self.slot(value)].is_some()
    }
}

impl Bound {
    pub(super) fn stored(value: StoredTensor) -> Self {
        Self::Tensor(TensorRealization::Stored(value))
    }

    pub(super) fn scalar(&self) -> ScalarBinding {
        match self {
            Self::Scalar(value) => *value,
            _ => panic!("checked scalar value has no scalar portable binding"),
        }
    }

    pub(super) fn from_binding(binding: PhysicalBinding, tensor: impl FnOnce(AnyBufferView) -> StoredTensor) -> Self {
        match binding {
            PhysicalBinding::View { view, .. } => Self::stored(tensor(view)),
            PhysicalBinding::Scalar(value) => Self::Scalar(value),
            PhysicalBinding::Range { start, end } => Self::Range {
                start: Box::new(Self::Scalar(start)),
                end: Box::new(Self::Scalar(end)),
            },
        }
    }
    pub(super) fn tensor(&self) -> StoredView {
        match self {
            Self::Tensor(value) => value.stored(),
            _ => panic!("checked tensor value has no tensor binding"),
        }
    }
}
