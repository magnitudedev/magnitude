//! Transport of complete values across structured regions: branch and loop
//! products, region parameters and root result publication.
use super::*;

impl<'f, 'b, B: seismic_native_target::TargetFamily> Lowerer<'f, 'b, B> {
    pub(super) fn materialize_tensor(&mut self, value: SemanticValueId) {
        let Bound::Tensor(TensorRealization::Computed(plan)) =
            self.values.get(self.builder.bindings(), value)
        else {
            return;
        };
        let destination = self
            .builder
            .portable_allocate_tensor_axes(value, plan.axes.clone());
        self.realize_tensor_at(&plan, destination);
        let bound = self.stored_allocation(value, destination, InitializationState::full());
        self.values
            .rebind(self.builder.bindings_mut(), value, bound);
    }

    /// Realize an owned product at an affine transport boundary. Affine maps
    /// preserve storage; nonaffine maps relocate bytes and initialized regions.
    pub(super) fn realize_region_product(&mut self, bound: Bound, ty: &SemanticType) -> Bound {
        self.realize_region_product_with(bound, ty, &mut BTreeMap::new())
    }

    pub(super) fn realize_region_product_with(
        &mut self,
        bound: Bound,
        ty: &SemanticType,
        realized: &mut BTreeMap<*const StreamTensorPlan, StoredTensor>,
    ) -> Bound {
        match (bound, ty) {
            (Bound::Tensor(TensorRealization::Computed(plan)), SemanticType::Tensor(tensor)) => {
                let identity = Arc::as_ptr(&plan);
                if let Some(stored) = realized.get(&identity) {
                    return Bound::stored(stored.clone());
                }
                let mut tensor = tensor.clone();
                tensor.axes = plan.axes.clone();
                let destination = self.builder.allocate_tensor_product(&tensor);
                self.realize_tensor_at(&plan, destination);
                let root = self.builder.portable_layout(destination).base;
                let axes = tensor
                    .axes
                    .iter()
                    .map(|axis| self.builder.arena().int_from_nat(*axis))
                    .collect::<Vec<_>>();
                let stored = self.values.contents.root(
                    &mut InitializationContext::new(self.builder.arena()),
                    root,
                    StoredView::new(destination, tensor.axes.clone()),
                    &axes,
                    InitializationState::full(),
                );
                realized.insert(identity, stored.clone());
                Bound::stored(stored)
            }
            (Bound::Tensor(TensorRealization::Stored(source)), SemanticType::Tensor(tensor))
                if !source.view.steps().is_empty() =>
            {
                if let Some(view) = self.builder.portable_affine_view(&source.view) {
                    let mapped = StoredView::new(view, source.view.extents().to_vec());
                    return Bound::stored(self.values.contents.alias(
                        &source,
                        mapped,
                        source.initialized_view.clone(),
                    ));
                }
                // This owned transport boundary may include unspecified
                // elements. Relocate storage bits, then transfer the exact
                // initialized region through the same logical map.
                let mut tensor = tensor.clone();
                tensor.axes = source.view.extents().to_vec();
                let destination = self.builder.allocate_tensor_product(&tensor);
                let mapped = StoredView::new(destination, tensor.axes.clone());
                self.builder
                    .portable_relocate_tensor(&source.view, destination);
                let root = self.builder.portable_layout(destination).base;
                let axes = tensor
                    .axes
                    .iter()
                    .map(|axis| self.builder.arena().int_from_nat(*axis))
                    .collect::<Vec<_>>();
                let stored = self.values.contents.root(
                    &mut InitializationContext::new(self.builder.arena()),
                    root,
                    mapped,
                    &axes,
                    InitializationState::empty(),
                );
                self.values.contents.copy_allocation(
                    &mut InitializationContext::new(self.builder.arena()),
                    &source,
                    &stored,
                );
                Bound::stored(stored)
            }
            (Bound::Tuple(values), SemanticType::Tuple(types)) => {
                assert_eq!(values.len(), types.len());
                Bound::Tuple(
                    values
                        .into_iter()
                        .zip(types)
                        .map(|(value, ty)| self.realize_region_product_with(value, ty, realized))
                        .collect(),
                )
            }
            (bound, _) => bound,
        }
    }

    pub(super) fn region_operand_product(
        &mut self,
        bound: &Bound,
    ) -> seismic_ir::region::Product<seismic_ir::region::ValueOperand> {
        use seismic_ir::region::{Product, ScalarOperand, ValueOperand};
        match bound {
            Bound::Unit => Product::Unit,
            Bound::Tensor(value) => Product::Leaf(ValueOperand::Tensor(
                *value.stored().direct_backing().expect(
                    "mapped carry requires owned realization before affine region transport",
                ),
            )),
            Bound::Scalar(ScalarBinding::Quantity(slot)) => {
                use seismic_ir::region::QuantityOperand;
                let operand = match slot.kind() {
                    seismic_ir::schedule::HostQuantityKind::Integer => {
                        QuantityOperand::Integer(self.builder.arena().int_symbol(slot.symbol()))
                    }
                    seismic_ir::schedule::HostQuantityKind::Natural => {
                        QuantityOperand::Natural(self.builder.arena().nat_symbol(slot.symbol()))
                    }
                };
                Product::Leaf(ValueOperand::Quantity(operand))
            }
            Bound::Scalar(value) => match prepare_scalar(self.builder.arena(), *value) {
                PreparedArg::Index(value) => Product::Leaf(ValueOperand::Quantity(
                    seismic_ir::region::QuantityOperand::Natural(value),
                )),
                PreparedArg::Integer(value) => Product::Leaf(ValueOperand::Quantity(
                    seismic_ir::region::QuantityOperand::Integer(value),
                )),
                PreparedArg::Scalar(symbol, dtype) => {
                    Product::Leaf(ValueOperand::Scalar(ScalarOperand::Word { symbol, dtype }))
                }
            },
            Bound::Range { start, end } => Product::Range(
                Box::new(self.region_operand_product(start)),
                Box::new(self.region_operand_product(end)),
            ),
            Bound::Tuple(values) => Product::Tuple(
                values
                    .iter()
                    .map(|value| self.region_operand_product(value))
                    .collect(),
            ),
        }
    }

    pub(super) fn region_destination_product(
        &mut self,
        destination: &seismic_ir::region::Product<seismic_ir::region::ValueDestination>,
        source: &Bound,
        ty: &SemanticType,
        contents: &StorageContents,
        alternate: Option<(&Bound, &StorageContents)>,
    ) -> Bound {
        use seismic_ir::region::{Product, ValueDestination};
        match (destination, source) {
            (Product::Unit, Bound::Unit) => Bound::Unit,
            (Product::Leaf(ValueDestination::Scalar(slot)), Bound::Scalar(_)) => {
                Bound::Scalar(ScalarBinding::Published(*slot))
            }
            (Product::Leaf(ValueDestination::Quantity(slot)), Bound::Scalar(_)) => {
                Bound::Scalar(ScalarBinding::Quantity(*slot))
            }
            (
                Product::Leaf(ValueDestination::Tensor(view)),
                Bound::Tensor(TensorRealization::Stored(source)),
            ) => {
                let layout = self.builder.portable_layout(*view).clone();
                // Every value a carry takes has its type's axes. Where the
                // source's extents are exactly those axes, the header and the
                // result have them too; so does a result whose entry and
                // backedge share extents. Otherwise the destination's own
                // extents name them.
                let typed = matches!(ty, SemanticType::Tensor(tensor)
                    if tensor.axes.as_slice() == source.view.extents());
                let shared = matches!(alternate,
                    Some((Bound::Tensor(TensorRealization::Stored(other)), _))
                        if other.view.extents() == source.view.extents());
                let extents = if typed || shared {
                    source.view.extents().to_vec()
                } else {
                    layout.extents.clone()
                };
                let axes = extents
                    .iter()
                    .map(|axis| self.builder.arena().int_from_nat(*axis))
                    .collect::<Vec<_>>();
                let mut context = InitializationContext::new(self.builder.arena());
                let mut state = context.project(contents.state(source), &source.initialized_view);
                if let Some((Bound::Tensor(TensorRealization::Stored(other)), other_contents)) =
                    alternate
                {
                    state = state.intersection(
                        &context.project(other_contents.state(other), &other.initialized_view),
                    );
                }
                Bound::stored(self.values.contents.root(
                    &mut context,
                    layout.base,
                    StoredView::new(*view, extents),
                    &axes,
                    state,
                ))
            }
            (Product::Range(a, b), Bound::Range { start, end }) => {
                let alternatives = alternate.map(|(other, contents)| {
                    let Bound::Range { start, end } = other else {
                        panic!("range carry changed shape")
                    };
                    ((start.as_ref(), contents), (end.as_ref(), contents))
                });
                Bound::Range {
                    start: Box::new(self.region_destination_product(
                        a,
                        start,
                        ty,
                        contents,
                        alternatives.map(|a| a.0),
                    )),
                    end: Box::new(self.region_destination_product(
                        b,
                        end,
                        ty,
                        contents,
                        alternatives.map(|a| a.1),
                    )),
                }
            }
            (Product::Tuple(destinations), Bound::Tuple(values)) => {
                assert_eq!(destinations.len(), values.len());
                let SemanticType::Tuple(types) = ty else {
                    panic!("tuple carry has non-tuple semantics")
                };
                let other = alternate.map(|(other, contents)| {
                    let Bound::Tuple(values) = other else {
                        panic!("tuple carry changed shape")
                    };
                    (values, contents)
                });
                Bound::Tuple(
                    destinations
                        .iter()
                        .zip(values)
                        .enumerate()
                        .map(|(index, (destination, value))| {
                            self.region_destination_product(
                                destination,
                                value,
                                &types[index],
                                contents,
                                other.map(|(values, contents)| (&values[index], contents)),
                            )
                        })
                        .collect(),
                )
            }
            _ => panic!("region product transport changed shape"),
        }
    }

    pub(super) fn begin_if(
        &mut self,
        condition_value: SemanticValueId,
        capture_values: &[SemanticValueId],
        outputs: &[SemanticValueId],
        then_region: RegionId,
        else_region: RegionId,
    ) -> construction::IfConstruction {
        let condition_bound = self.bound(condition_value);
        let condition = condition_expr(
            self.builder.arena(),
            &self.values.host_conditions,
            &condition_bound,
        );
        let captures = capture_values
            .iter()
            .map(|value| self.values.handle(*value))
            .collect::<Vec<_>>();
        let parent = self.values.clone();
        let schedule = self.builder.begin_source_branch(condition);
        self.values.selections.insert(condition, 1);
        self.bind_if_parameters(then_region, &captures);
        construction::IfConstruction {
            schedule,
            condition,
            captures,
            outputs: outputs.to_vec(),
            then_region,
            else_region,
            parent,
            then_products: Vec::new(),
            then_contents: None,
        }
    }

    pub(super) fn bind_if_parameters(&mut self, region: RegionId, captures: &[BindingId]) {
        let parameters = self.function.region(region).parameters();
        assert_eq!(parameters.len(), captures.len());
        for (parameter, capture) in parameters.iter().zip(captures) {
            let slot = self.values.slot(*parameter);
            assert!(self.values.values[slot].replace(*capture).is_none());
        }
    }

    /// A branch arm's results are transported into its branch's destination
    /// exactly as a loop backedge is: a result still selected by a branch
    /// enclosed in the arm is that branch's one joined destination.
    pub(super) fn realized_region_results(&mut self, region: RegionId) -> Vec<Bound> {
        let ids = self.function.region(region).results().to_vec();
        let values = Bound::Tuple(ids.iter().map(|value| self.transported(*value)).collect());
        let types = SemanticType::Tuple(
            ids.iter()
                .map(|value| self.function.value(*value).ty.clone())
                .collect(),
        );
        let Bound::Tuple(values) = self.realize_region_product(values, &types) else {
            unreachable!()
        };
        values
    }

    /// Descriptor forwarding changes physical location only. Each arm keeps
    /// its actual storage origin and initialized view. A leaf both arms give
    /// the same meaning keeps that meaning in the join: identical extents or
    /// scalars can mention only what both arms see, which is what precedes
    /// the branch, so the join needs no fresh name for them. Only where the
    /// arms differ does the join's value become its destination's own slot.
    pub(super) fn forward_branch_products(
        &mut self,
        destination: &seismic_ir::region::Product<seismic_ir::region::ValueDestination>,
        then: &Bound,
        otherwise: &Bound,
    ) -> (Bound, Bound) {
        use seismic_ir::region::{Product, ValueDestination};
        match (destination, then, otherwise) {
            (Product::Unit, Bound::Unit, Bound::Unit) => (Bound::Unit, Bound::Unit),
            (Product::Leaf(_), Bound::Scalar(a), Bound::Scalar(b)) if a == b => {
                (Bound::Scalar(*a), Bound::Scalar(*b))
            }
            (Product::Leaf(ValueDestination::Scalar(slot)), Bound::Scalar(_), Bound::Scalar(_)) => {
                (
                    Bound::Scalar(ScalarBinding::Published(*slot)),
                    Bound::Scalar(ScalarBinding::Published(*slot)),
                )
            }
            (
                Product::Leaf(ValueDestination::Quantity(slot)),
                Bound::Scalar(_),
                Bound::Scalar(_),
            ) => (
                Bound::Scalar(ScalarBinding::Quantity(*slot)),
                Bound::Scalar(ScalarBinding::Quantity(*slot)),
            ),
            (
                Product::Leaf(ValueDestination::Tensor(view)),
                Bound::Tensor(TensorRealization::Stored(a)),
                Bound::Tensor(TensorRealization::Stored(b)),
            ) => {
                let extents = if a.view.extents() == b.view.extents() {
                    a.view.extents().to_vec()
                } else {
                    self.builder.portable_layout(*view).extents
                };
                let forward = |source: &StoredTensor| {
                    let mut value = source.clone();
                    value.view = StoredView::new(*view, extents.clone());
                    Bound::stored(value)
                };
                (forward(a), forward(b))
            }
            (
                Product::Range(a, b),
                Bound::Range {
                    start: then_start,
                    end: then_end,
                },
                Bound::Range {
                    start: else_start,
                    end: else_end,
                },
            ) => {
                let (then_start, else_start) =
                    self.forward_branch_products(a, then_start, else_start);
                let (then_end, else_end) = self.forward_branch_products(b, then_end, else_end);
                (
                    Bound::Range {
                        start: Box::new(then_start),
                        end: Box::new(then_end),
                    },
                    Bound::Range {
                        start: Box::new(else_start),
                        end: Box::new(else_end),
                    },
                )
            }
            (Product::Tuple(destinations), Bound::Tuple(a), Bound::Tuple(b)) => {
                assert_eq!(destinations.len(), a.len());
                assert_eq!(destinations.len(), b.len());
                let (a, b) = destinations
                    .iter()
                    .zip(a)
                    .zip(b)
                    .map(|((destination, a), b)| self.forward_branch_products(destination, a, b))
                    .unzip();
                (Bound::Tuple(a), Bound::Tuple(b))
            }
            _ => panic!("branch product changed shape"),
        }
    }

    pub(super) fn next_if(&mut self, branch: &mut construction::IfConstruction) {
        branch.then_products = self.realized_region_results(branch.then_region);
        branch.then_contents = Some(self.values.contents.clone());
        assert_eq!(
            self.builder.next_source_branch(&mut branch.schedule),
            Some(0)
        );
        self.values = branch.parent.clone();
        self.values.selections.insert(branch.condition, 0);
        self.bind_if_parameters(branch.else_region, &branch.captures);
    }

    pub(super) fn finish_if(&mut self, branch: construction::IfConstruction) {
        let otherwise = self.realized_region_results(branch.else_region);
        let else_contents = self.values.contents.clone();
        let then_values = Bound::Tuple(branch.then_products.clone());
        let else_values = Bound::Tuple(otherwise.clone());
        let then_operands = self.region_operand_product(&then_values);
        let else_operands = self.region_operand_product(&else_values);
        let destination =
            self.builder
                .finish_value_branch(branch.schedule, then_operands, else_operands);
        let (Bound::Tuple(then_values), Bound::Tuple(else_values)) =
            self.forward_branch_products(&destination, &then_values, &else_values)
        else {
            unreachable!()
        };
        self.values = branch.parent;
        self.values.contents.branch(
            &mut InitializationContext::new(self.builder.arena()),
            branch.condition,
            &self.values.binders,
            &branch
                .then_contents
                .expect("then arm completed before else arm"),
            &else_contents,
        );
        assert_eq!(branch.outputs.len(), branch.then_products.len());
        assert_eq!(branch.outputs.len(), otherwise.len());
        for ((output, then), otherwise) in
            branch.outputs.into_iter().zip(then_values).zip(else_values)
        {
            let then = self.builder.bindings_mut().insert(then);
            let otherwise = self.builder.bindings_mut().insert(otherwise);
            let selected = self
                .builder
                .bindings_mut()
                .selected_value(branch.condition, vec![(1, then), (0, otherwise)]);
            let slot = self.values.slot(output);
            assert!(self.values.values[slot].replace(selected).is_none());
        }
    }

    pub(super) fn begin_loop(
        &mut self,
        start_value: SemanticValueId,
        end_value: SemanticValueId,
        capture_values: &[SemanticValueId],
        body: RegionId,
        carries: &[seismic_lang::entry::Carry],
        visits: seismic_ir::schedule::RepeatVisits,
    ) -> LoopConstruction {
        let start_bound = self.bound(start_value);
        let end_bound = self.bound(end_value);
        let start = index_expr(self.builder.arena(), &start_bound);
        let end = index_expr(self.builder.arena(), &end_bound);
        let captures = capture_values
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let initial = Bound::Tuple(
            carries
                .iter()
                .map(|carry| self.bound(carry.initial))
                .collect(),
        );
        let initial_type = SemanticType::Tuple(
            carries
                .iter()
                .map(|carry| self.function.value(carry.initial).ty.clone())
                .collect(),
        );
        let initial = self.realize_region_product(initial, &initial_type);
        let operands = self.region_operand_product(&initial);
        let parent = self.values.clone();
        let schedule = self
            .builder
            .begin_value_repeat(start, end, visits, operands);
        let binding = schedule.binding();
        self.values.binders.push(binding.symbol);
        self.bind_region_parameters(body, &captures);
        let RegionKind::LoopBody { binder_value, .. } = self.function.region(body).kind() else {
            panic!("loop body kind")
        };
        let index = self.builder.arena().nat_symbol(binding.symbol);
        self.values.bind(
            self.builder.bindings_mut(),
            *binder_value,
            Bound::Scalar(ScalarBinding::Index(index)),
        );
        let Bound::Tuple(headers) = self.region_destination_product(
            schedule.header(),
            &initial,
            &initial_type,
            &parent.contents,
            None,
        ) else {
            unreachable!()
        };
        for (carry, header) in carries.iter().zip(headers) {
            self.values
                .rebind(self.builder.bindings_mut(), carry.parameter, header);
        }
        let loop_values = self
            .function
            .region(body)
            .parameters()
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let arguments = self.initialization_arguments(&loop_values);
        let start_integer = self.builder.arena().int_from_nat(start);
        let current_integer = self.builder.arena().int_from_nat(index);
        let mut context = initialization_context(
            self.builder.arena(),
            &self.values.selections,
            &self.values.binders,
        );
        self.values.contents.loop_header(
            &mut context,
            self.function
                .region(body)
                .loop_initialization()
                .expect("loop owns checked initialization"),
            &arguments,
            start_integer,
            current_integer,
        );
        LoopConstruction {
            schedule,
            parent,
            initial,
            start,
            end,
            carries: carries.to_vec(),
        }
    }

    /// A value transported by a region product (a branch arm's result or a
    /// loop backedge) is one product, never a selection the region's
    /// destination cannot represent. A value still selected by an enclosed
    /// branch joins its arms: when every arm was already forwarded into that
    /// branch's one destination, the product is that destination with the
    /// contents both arms guarantee. Arms a selected construction left in
    /// their own storage are forwarded now, by a branch on the same selector
    /// whose arms only transport their value into one destination.
    fn transported(&mut self, value: SemanticValueId) -> Bound {
        let ty = self.function.value(value).ty.clone();
        self.transported_binding(self.values.handle(value), &ty)
    }

    fn transported_binding(&mut self, id: BindingId, ty: &SemanticType) -> Bound {
        let bindings = self.builder.bindings();
        let mut values = bindings.possible_values(id, &self.values.selections);
        if values.len() == 1 {
            return values.pop().expect("one possible value");
        }
        if one_destination(&values) {
            let mut selections = self.values.selections.clone();
            let joined = self.joined_tree(id, &mut selections);
            return self.published_join(joined);
        }
        let Resolution::Undecided { selector, options } =
            bindings.resolve(id, &self.values.selections)
        else {
            unreachable!("a binding with several possible values is undecided")
        };
        self.forwarded_selection(selector, &options, ty)
    }

    fn forwarded_selection(
        &mut self,
        selector: BindingSelector,
        options: &[(i64, BindingId)],
        ty: &SemanticType,
    ) -> Bound {
        let option = |selected: i64| {
            options
                .iter()
                .find(|(option, _)| *option == selected)
                .expect("selected binding option is complete")
                .1
        };
        let parent = self.values.clone();
        let (selected, mut schedule) = self.builder.begin_source_selection(selector);
        assert_eq!(selected, 1, "a selection begins with its taken arm");
        self.values.selections.insert(selector, 1);
        let then = self.transported_binding(option(1), ty);
        let then = self.realize_region_product(then, ty);
        let then_contents = self.values.contents.clone();
        assert_eq!(self.builder.next_source_branch(&mut schedule), Some(0));
        self.values = parent.clone();
        self.values.selections.insert(selector, 0);
        let otherwise = self.transported_binding(option(0), ty);
        let otherwise = self.realize_region_product(otherwise, ty);
        let else_contents = self.values.contents.clone();
        let then_operands = self.region_operand_product(&then);
        let else_operands = self.region_operand_product(&otherwise);
        let destination = self
            .builder
            .finish_value_branch(schedule, then_operands, else_operands);
        let (then, otherwise) = self.forward_branch_products(&destination, &then, &otherwise);
        self.values = parent;
        self.values.contents.branch(
            &mut InitializationContext::new(self.builder.arena()),
            selector,
            &self.values.binders,
            &then_contents,
            &else_contents,
        );
        let (then, otherwise) = (self.joined_leaf(then), self.joined_leaf(otherwise));
        let joined = self.joined_branch(selector, then, otherwise);
        self.published_join(joined)
    }

    /// The join of a binding's arms that share one destination, each arm's
    /// contents holding exactly where its selection does.
    fn joined_tree(&mut self, id: BindingId, selections: &mut BindingSelections) -> Joined {
        match self.builder.bindings().resolve(id, selections) {
            Resolution::Value(bound) => self.joined_leaf(bound),
            Resolution::Undecided { selector, options } => {
                assert_eq!(
                    options.len(),
                    2,
                    "a selection has a taken and an untaken arm"
                );
                let option = |selected: i64| {
                    options
                        .iter()
                        .find(|(option, _)| *option == selected)
                        .expect("selected binding option is complete")
                        .1
                };
                selections.insert(selector, 1);
                let then = self.joined_tree(option(1), selections);
                selections.insert(selector, 0);
                let otherwise = self.joined_tree(option(0), selections);
                selections.remove(&selector);
                self.joined_branch(selector, then, otherwise)
            }
        }
    }

    fn joined_leaf(&mut self, bound: Bound) -> Joined {
        match bound {
            Bound::Tensor(TensorRealization::Stored(arm)) => {
                let state = InitializationContext::new(self.builder.arena())
                    .project(self.values.contents.state(&arm), &arm.initialized_view);
                Joined::Tensor {
                    arm,
                    shared_extents: true,
                    state,
                }
            }
            Bound::Tensor(TensorRealization::Computed(_)) => {
                panic!("a branch product is realized before its join")
            }
            Bound::Scalar(scalar) => Joined::Scalar(scalar),
            Bound::Range { start, end } => Joined::Range(
                Box::new(self.joined_leaf(*start)),
                Box::new(self.joined_leaf(*end)),
            ),
            Bound::Tuple(items) => Joined::Tuple(
                items
                    .into_iter()
                    .map(|item| self.joined_leaf(item))
                    .collect(),
            ),
            Bound::Unit => Joined::Unit,
        }
    }

    /// The join of two arms of `selector` forwarded into one destination.
    /// The destination holds the taken arm's contents where the selector
    /// holds and the other arm's where it does not; an arm's contents never
    /// stand for the other world.
    fn joined_branch(
        &mut self,
        selector: BindingSelector,
        then: Joined,
        otherwise: Joined,
    ) -> Joined {
        match (then, otherwise) {
            (
                Joined::Tensor {
                    arm,
                    shared_extents: then_shared,
                    state: then_state,
                },
                Joined::Tensor {
                    arm: other,
                    shared_extents: else_shared,
                    state: else_state,
                },
            ) => {
                assert!(
                    arm.view.direct_backing() == other.view.direct_backing(),
                    "joined tensor arms have one destination"
                );
                let state = InitializationContext::new(self.builder.arena()).branch(
                    selector,
                    &self.values.binders,
                    &then_state,
                    &else_state,
                );
                Joined::Tensor {
                    shared_extents: then_shared
                        && else_shared
                        && arm.view.extents() == other.view.extents(),
                    arm,
                    state,
                }
            }
            (Joined::Scalar(a), Joined::Scalar(b)) => {
                assert_eq!(
                    a, b,
                    "joined scalar arms were forwarded into one destination"
                );
                Joined::Scalar(a)
            }
            (Joined::Range(a_start, a_end), Joined::Range(b_start, b_end)) => Joined::Range(
                Box::new(self.joined_branch(selector, *a_start, *b_start)),
                Box::new(self.joined_branch(selector, *a_end, *b_end)),
            ),
            (Joined::Tuple(a), Joined::Tuple(b)) => {
                assert_eq!(a.len(), b.len(), "joined tuple arms differ in arity");
                Joined::Tuple(
                    a.into_iter()
                        .zip(b)
                        .map(|(a, b)| self.joined_branch(selector, a, b))
                        .collect(),
                )
            }
            (Joined::Unit, Joined::Unit) => Joined::Unit,
            _ => panic!("joined arms differ in kind"),
        }
    }

    /// Publish a join as the destination's one root.
    fn published_join(&mut self, joined: Joined) -> Bound {
        match joined {
            Joined::Tensor {
                arm,
                shared_extents,
                state,
            } => {
                let view = *arm
                    .view
                    .direct_backing()
                    .expect("a branch product is its destination view");
                let layout = self.builder.portable_layout(view).clone();
                // Arms forwarded with one shared meaning of their extents
                // keep it; otherwise the destination's own extents name them.
                let extents = if shared_extents {
                    arm.view.extents().to_vec()
                } else {
                    layout.extents.clone()
                };
                let axes = extents
                    .iter()
                    .map(|axis| self.builder.arena().int_from_nat(*axis))
                    .collect::<Vec<_>>();
                let mut context = InitializationContext::new(self.builder.arena());
                Bound::stored(self.values.contents.root(
                    &mut context,
                    layout.base,
                    StoredView::new(view, extents),
                    &axes,
                    state,
                ))
            }
            Joined::Scalar(scalar) => Bound::Scalar(scalar),
            Joined::Range(start, end) => Bound::Range {
                start: Box::new(self.published_join(*start)),
                end: Box::new(self.published_join(*end)),
            },
            Joined::Tuple(items) => Bound::Tuple(
                items
                    .into_iter()
                    .map(|item| self.published_join(item))
                    .collect(),
            ),
            Joined::Unit => Bound::Unit,
        }
    }

    pub(super) fn finish_loop(&mut self, ticket: LoopConstruction) {
        let yielded = Bound::Tuple(
            ticket
                .carries
                .iter()
                .map(|carry| self.transported(carry.yielded))
                .collect(),
        );
        let yielded_type = SemanticType::Tuple(
            ticket
                .carries
                .iter()
                .map(|carry| self.function.value(carry.yielded).ty.clone())
                .collect(),
        );
        let yielded = self.realize_region_product(yielded, &yielded_type);
        let backedge = self.region_operand_product(&yielded);
        let iteration = self.values.contents.clone();
        let symbol = ticket.schedule.binding().symbol;
        let destinations = self.builder.finish_value_repeat(ticket.schedule, backedge);
        self.values = ticket.parent;
        let before = self.values.contents.clone();
        let start = self.builder.arena().int_from_nat(ticket.start);
        let end = self.builder.arena().int_from_nat(ticket.end);
        self.values.contents.completed_loop(
            &mut InitializationContext::new(self.builder.arena()),
            &before,
            &iteration,
            symbol,
            start,
            end,
        );
        let Bound::Tuple(results) = self.region_destination_product(
            &destinations,
            &yielded,
            &yielded_type,
            &iteration,
            Some((&ticket.initial, &before)),
        ) else {
            unreachable!()
        };
        for (carry, result) in ticket.carries.iter().zip(results) {
            self.values
                .bind(self.builder.bindings_mut(), carry.result, result);
        }
    }

    pub(super) fn bind_region_parameters(&mut self, region: RegionId, captures: &[Bound]) {
        let region = self.function.region(region);
        let parameters = match region.kind() {
            RegionKind::LoopBody { binder_value, .. } => {
                let (binder, captures) = region
                    .parameters()
                    .split_first()
                    .expect("loop region is missing its binder parameter");
                assert_eq!(
                    binder, binder_value,
                    "loop binder is not the first region parameter"
                );
                captures
            }
            RegionKind::Root | RegionKind::Then | RegionKind::Else => region.parameters(),
        };
        assert_eq!(
            parameters.len(),
            captures.len(),
            "region capture arity differs from semantic inputs"
        );
        for (parameter, capture) in parameters.iter().zip(captures) {
            self.values
                .bind(self.builder.bindings_mut(), *parameter, capture.clone());
        }
    }

    pub(super) fn publication_selection(&mut self) -> Option<BindingSelector> {
        if !self.builder.portable_is_root() {
            return None;
        }
        // Scalar publication slots are stable across selected arms. Tensor
        // publication consumes each arm's actual completed descriptor instead
        // of allocating a nominal destination before the arm executes.
        for result in self.function.results() {
            if !matches!(self.function.value(*result).ty, SemanticType::Tensor(_)) {
                let _ = self.output_target(*result);
            }
        }
        self.function.results().iter().find_map(|value| {
            self.builder
                .bindings()
                .unresolved(self.values.handle(*value), &self.values.selections)
        })
    }

    pub(super) fn publish_results(&mut self) {
        if !self.builder.portable_is_root() {
            self.builder.bindings_mut().result_contents = self.values.contents.clone();
            self.builder.bindings_mut().results = self
                .function
                .results()
                .iter()
                .map(|result| self.values.handle(*result))
                .collect();
            return;
        }
        assert!(
            self.publication_selection().is_none(),
            "publication requires its selected construction arm"
        );
        for result in self.function.results() {
            self.materialize_tensor(*result);
            let source = self.bound(*result);
            let target = if matches!(source, Bound::Tensor(_)) {
                let ty = self.function.value(*result).ty.clone();
                let realized = self.realize_region_product(source, &ty);
                let view = *realized
                    .tensor()
                    .direct_backing()
                    .expect("publication realization must have an affine descriptor");
                self.builder.portable_publish_tensor(*result, view);
                realized
            } else {
                let target = self.output_target(*result);
                self.assign(&source, &target);
                target
            };
            self.values
                .rebind(self.builder.bindings_mut(), *result, target);
        }
        self.builder.bindings_mut().result_contents = self.values.contents.clone();
        self.builder.bindings_mut().results = self
            .function
            .results()
            .iter()
            .map(|result| self.values.handle(*result))
            .collect();
    }
}

/// Whether the arms of a selection already share their branch's one
/// destination: the same storage view for every tensor and the same
/// realization for every scalar.
fn one_destination(values: &[Bound]) -> bool {
    let first = &values[0];
    match first {
        Bound::Unit => values.iter().all(|value| matches!(value, Bound::Unit)),
        Bound::Scalar(scalar) => values
            .iter()
            .all(|value| matches!(value, Bound::Scalar(arm) if arm == scalar)),
        Bound::Tensor(TensorRealization::Stored(stored)) => {
            let Some(view) = stored.view.direct_backing() else {
                return false;
            };
            values.iter().all(|value| {
                matches!(
                    value,
                    Bound::Tensor(TensorRealization::Stored(arm))
                        if arm.view.direct_backing() == Some(view)
                )
            })
        }
        Bound::Tensor(TensorRealization::Computed(_)) => false,
        Bound::Range { .. } => {
            let mut starts = Vec::with_capacity(values.len());
            let mut ends = Vec::with_capacity(values.len());
            for value in values {
                let Bound::Range { start, end } = value else {
                    panic!("selected range arms differ in kind")
                };
                starts.push((**start).clone());
                ends.push((**end).clone());
            }
            one_destination(&starts) && one_destination(&ends)
        }
        Bound::Tuple(items) => (0..items.len()).all(|index| {
            let column = values
                .iter()
                .map(|value| {
                    let Bound::Tuple(items) = value else {
                        panic!("selected tuple arms differ in kind")
                    };
                    items[index].clone()
                })
                .collect::<Vec<_>>();
            one_destination(&column)
        }),
    }
}

/// A join of branch arms before it is published as one destination root:
/// each tensor leaf carries the contents of every world it may come from.
enum Joined {
    Unit,
    Scalar(ScalarBinding),
    Tensor {
        arm: StoredTensor,
        shared_extents: bool,
        state: InitializationState,
    },
    Range(Box<Joined>, Box<Joined>),
    Tuple(Vec<Joined>),
}
