//! Transport of complete values across structured regions: branch and loop
//! products, region parameters and root result publication.
use super::*;

impl<'f, 'b, B: seismic_target::TargetFamily> Lowerer<'f, 'b, B> {
    pub(super) fn materialize_tensor(&mut self, value: SemanticValueId) {
        let Bound::Tensor(TensorRealization::Computed(plan)) = self.values.get(self.builder.bindings(), value) else { return; };
        let destination = self.builder.portable_allocate_tensor_axes(value, plan.axes.clone());
        self.realize_tensor_at(&plan,destination);
        let bound=self.stored_allocation(value,destination,InitializationState::full());
        self.values.rebind(self.builder.bindings_mut(),value,bound);
    }

    /// Realize an owned product at an affine transport boundary. Affine maps
    /// preserve storage; nonaffine maps relocate bytes and initialized regions.
    pub(super) fn realize_region_product(&mut self, bound: Bound, ty: &SemanticType) -> Bound {
        self.realize_region_product_with(bound,ty,&mut BTreeMap::new())
    }

    pub(super) fn realize_region_product_with(&mut self, bound: Bound, ty: &SemanticType, realized: &mut BTreeMap<*const StreamTensorPlan,StoredTensor>) -> Bound {
        match (bound,ty) {
            (Bound::Tensor(TensorRealization::Computed(plan)),SemanticType::Tensor(tensor)) => {
                let identity=Arc::as_ptr(&plan);
                if let Some(stored)=realized.get(&identity) { return Bound::stored(stored.clone()); }
                let mut tensor = tensor.clone();
                tensor.axes = plan.axes.clone();
                let destination=self.builder.allocate_tensor_product(&tensor);
                self.realize_tensor_at(&plan,destination);
                let root=self.builder.portable_layout(destination).base;
                let axes=tensor.axes.iter().map(|axis|self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                let stored=self.values.contents.root(&mut InitializationContext::new(self.builder.arena()),root,StoredView::new(destination,tensor.axes.clone()),&axes,InitializationState::full());
                realized.insert(identity,stored.clone());
                Bound::stored(stored)
            }
            (Bound::Tensor(TensorRealization::Stored(source)),SemanticType::Tensor(tensor)) if !source.view.steps().is_empty() => {
                if let Some(view)=self.builder.portable_affine_view(&source.view) {
                    let mapped=StoredView::new(view,source.view.extents().to_vec());
                    return Bound::stored(self.values.contents.alias(&source,mapped,source.initialized_view.clone()));
                }
                // This owned transport boundary may include unspecified
                // elements. Relocate storage bits, then transfer the exact
                // initialized region through the same logical map.
                let mut tensor=tensor.clone();
                tensor.axes=source.view.extents().to_vec();
                let destination=self.builder.allocate_tensor_product(&tensor);
                let mapped=StoredView::new(destination,tensor.axes.clone());
                self.builder.portable_relocate_tensor(&source.view,destination);
                let root=self.builder.portable_layout(destination).base;
                let axes=tensor.axes.iter().map(|axis|self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                let stored=self.values.contents.root(&mut InitializationContext::new(self.builder.arena()),root,mapped,&axes,InitializationState::empty());
                self.values.contents.copy_allocation(&mut InitializationContext::new(self.builder.arena()),&source,&stored);
                Bound::stored(stored)
            }
            (Bound::Tuple(values),SemanticType::Tuple(types)) => {
                assert_eq!(values.len(),types.len());
                Bound::Tuple(values.into_iter().zip(types).map(|(value,ty)|self.realize_region_product_with(value,ty,realized)).collect())
            }
            (bound,_)=>bound,
        }
    }

    pub(super) fn region_operand_product(&mut self, bound: &Bound) -> seismic_ir::region::Product<seismic_ir::region::ValueOperand> {
        use seismic_ir::region::{Product,ValueOperand,ScalarOperand};
        match bound {
            Bound::Unit=>Product::Unit,
            Bound::Tensor(value)=>Product::Leaf(ValueOperand::Tensor(*value.stored().direct_backing().expect("mapped carry requires owned realization before affine region transport"))),
            Bound::Scalar(ScalarBinding::Quantity(slot)) => {
                use seismic_ir::region::QuantityOperand;
                let operand = match slot.kind() {
                    seismic_ir::schedule::HostQuantityKind::Integer => QuantityOperand::Integer(self.builder.arena().int_symbol(slot.symbol())),
                    seismic_ir::schedule::HostQuantityKind::Natural => QuantityOperand::Natural(self.builder.arena().nat_symbol(slot.symbol())),
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
                PreparedArg::Scalar(symbol, dtype) => Product::Leaf(ValueOperand::Scalar(
                    ScalarOperand::Word { symbol, dtype },
                )),
            },
            Bound::Range{start,end}=>Product::Range(Box::new(self.region_operand_product(start)),Box::new(self.region_operand_product(end))),
            Bound::Tuple(values)=>Product::Tuple(values.iter().map(|value|self.region_operand_product(value)).collect()),
        }
    }

    pub(super) fn region_destination_product(
        &mut self,
        destination: &seismic_ir::region::Product<seismic_ir::region::ValueDestination>,
        source: &Bound,
        contents: &StorageContents,
        alternate: Option<(&Bound,&StorageContents)>,
    ) -> Bound {
        use seismic_ir::region::{Product,ValueDestination};
        match (destination,source) {
            (Product::Unit,Bound::Unit)=>Bound::Unit,
            (Product::Leaf(ValueDestination::Scalar(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Published(*slot)),
            (Product::Leaf(ValueDestination::Quantity(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Quantity(*slot)),
            (Product::Leaf(ValueDestination::Tensor(view)),Bound::Tensor(TensorRealization::Stored(source)))=> {
                let layout=self.builder.portable_layout(*view).clone();
                let axes=layout.extents.iter().map(|axis|self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                let mut context=InitializationContext::new(self.builder.arena());
                let mut state=context.project(contents.state(source),&source.initialized_view);
                if let Some((Bound::Tensor(TensorRealization::Stored(other)),other_contents))=alternate {
                    state=state.intersection(&context.project(other_contents.state(other),&other.initialized_view));
                }
                Bound::stored(self.values.contents.root(&mut context,layout.base,StoredView::new(*view,layout.extents.clone()),&axes,state))
            }
            (Product::Range(a,b),Bound::Range{start,end})=> {
                let alternatives=alternate.map(|(other,contents)| {
                    let Bound::Range{start,end}=other else { panic!("range carry changed shape") };
                    ((start.as_ref(),contents),(end.as_ref(),contents))
                });
                Bound::Range{start:Box::new(self.region_destination_product(a,start,contents,alternatives.map(|a|a.0))),end:Box::new(self.region_destination_product(b,end,contents,alternatives.map(|a|a.1)))}
            }
            (Product::Tuple(destinations),Bound::Tuple(values))=> {
                assert_eq!(destinations.len(),values.len());
                let other=alternate.map(|(other,contents)| {
                    let Bound::Tuple(values)=other else { panic!("tuple carry changed shape") };
                    (values,contents)
                });
                Bound::Tuple(destinations.iter().zip(values).enumerate().map(|(index,(destination,value))|self.region_destination_product(destination,value,contents,other.map(|(values,contents)|(&values[index],contents)))).collect())
            }
            _=>panic!("region product transport changed shape"),
        }
    }

    pub(super) fn begin_if(
        &mut self, condition_value: SemanticValueId, capture_values: &[SemanticValueId],
        outputs: &[SemanticValueId], then_region: RegionId, else_region: RegionId,
    ) -> construction::IfConstruction {
        let condition_bound = self.bound(condition_value);
        let condition = condition_expr(self.builder.arena(), &condition_bound);
        let captures = capture_values.iter().map(|value| self.values.handle(*value)).collect::<Vec<_>>();
        let parent = self.values.clone();
        let schedule = self.builder.begin_source_branch(condition);
        self.values.selections.insert(condition, 1);
        self.bind_if_parameters(then_region, &captures);
        construction::IfConstruction { schedule, condition, captures, outputs: outputs.to_vec(), then_region, else_region, parent, then_products: Vec::new(), then_contents: None }
    }

    pub(super) fn bind_if_parameters(&mut self, region: RegionId, captures: &[BindingId]) {
        let parameters = self.function.region(region).parameters();
        assert_eq!(parameters.len(), captures.len());
        for (parameter, capture) in parameters.iter().zip(captures) {
            let slot = self.values.slot(*parameter);
            assert!(self.values.values[slot].replace(*capture).is_none());
        }
    }

    pub(super) fn realized_region_results(&mut self, region: RegionId) -> Vec<Bound> {
        let ids = self.function.region(region).results().to_vec();
        let values = Bound::Tuple(ids.iter().map(|value|self.bound(*value)).collect());
        let types = SemanticType::Tuple(ids.iter().map(|value|self.function.value(*value).ty.clone()).collect());
        let Bound::Tuple(values) = self.realize_region_product(values, &types) else { unreachable!() };
        values
    }

    /// Descriptor forwarding changes physical location only. The selected
    /// source arm keeps its actual storage origin and initialized view.
    pub(super) fn forward_region_product(&mut self, destination: &seismic_ir::region::Product<seismic_ir::region::ValueDestination>, source: &Bound) -> Bound {
        use seismic_ir::region::{Product,ValueDestination};
        match (destination,source) {
            (Product::Unit,Bound::Unit)=>Bound::Unit,
            (Product::Leaf(ValueDestination::Scalar(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Published(*slot)),
            (Product::Leaf(ValueDestination::Quantity(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Quantity(*slot)),
            (Product::Leaf(ValueDestination::Tensor(view)),Bound::Tensor(TensorRealization::Stored(source)))=>{
                let layout=self.builder.portable_layout(*view);
                let mut value=source.clone();
                value.view=StoredView::new(*view,layout.extents);
                Bound::stored(value)
            }
            (Product::Range(a,b),Bound::Range{start,end})=>Bound::Range {start:Box::new(self.forward_region_product(a,start)),end:Box::new(self.forward_region_product(b,end))},
            (Product::Tuple(a),Bound::Tuple(b))=>{
                assert_eq!(a.len(),b.len());
                Bound::Tuple(a.iter().zip(b).map(|(a,b)|self.forward_region_product(a,b)).collect())
            }
            _=>panic!("branch product changed shape"),
        }
    }

    pub(super) fn next_if(&mut self, branch: &mut construction::IfConstruction) {
        branch.then_products = self.realized_region_results(branch.then_region);
        branch.then_contents = Some(self.values.contents.clone());
        assert_eq!(self.builder.next_source_branch(&mut branch.schedule), Some(0));
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
        let destination = self.builder.finish_value_branch(branch.schedule, then_operands, else_operands);
        let Bound::Tuple(then_values) = self.forward_region_product(&destination, &then_values) else { unreachable!() };
        let Bound::Tuple(else_values) = self.forward_region_product(&destination, &else_values) else { unreachable!() };
        self.values = branch.parent;
        self.values.contents.branch(&mut InitializationContext::new(self.builder.arena()), branch.condition, &self.values.binders, &branch.then_contents.expect("then arm completed before else arm"), &else_contents);
        assert_eq!(branch.outputs.len(), branch.then_products.len());
        assert_eq!(branch.outputs.len(), otherwise.len());
        for ((output, then), otherwise) in branch.outputs.into_iter().zip(then_values).zip(else_values) {
            let then = self.builder.bindings_mut().insert(then);
            let otherwise = self.builder.bindings_mut().insert(otherwise);
            let selected = self.builder.bindings_mut().selected_value(branch.condition, vec![(1, then), (0, otherwise)]);
            let slot = self.values.slot(output);
            assert!(self.values.values[slot].replace(selected).is_none());
        }
    }

    pub(super) fn begin_loop(&mut self,start_value:SemanticValueId,end_value:SemanticValueId,capture_values:&[SemanticValueId],body:RegionId,carries:&[seismic_lang::entry::Carry])->LoopConstruction {
        let start_bound=self.bound(start_value);
        let end_bound=self.bound(end_value);
        let start=index_expr(self.builder.arena(),&start_bound);
        let end=index_expr(self.builder.arena(),&end_bound);
        let captures=capture_values.iter().map(|value|self.bound(*value)).collect::<Vec<_>>();
        let initial=Bound::Tuple(carries.iter().map(|carry|self.bound(carry.initial)).collect());
        let initial_type=SemanticType::Tuple(carries.iter().map(|carry|self.function.value(carry.initial).ty.clone()).collect());
        let initial=self.realize_region_product(initial,&initial_type);
        let operands=self.region_operand_product(&initial);
        let parent=self.values.clone();
        let schedule=self.builder.begin_value_repeat(start,end,operands);
        let binding=schedule.binding();
        self.values.binders.push(binding.symbol);
        self.bind_region_parameters(body,&captures);
        let RegionKind::LoopBody{binder_value,..}=self.function.region(body).kind() else {panic!("loop body kind")};
        let index=self.builder.arena().nat_symbol(binding.symbol);
        self.values.bind(self.builder.bindings_mut(),*binder_value,Bound::Scalar(ScalarBinding::Index(index)));
        let Bound::Tuple(headers)=self.region_destination_product(schedule.header(),&initial,&parent.contents,None) else {unreachable!()};
        for (carry,header) in carries.iter().zip(headers) {
            self.values.rebind(self.builder.bindings_mut(),carry.parameter,header);
        }
        let loop_values=self.function.region(body).parameters().iter().map(|value|self.bound(*value)).collect::<Vec<_>>();
        let arguments=self.initialization_arguments(&loop_values);
        let start_integer=self.builder.arena().int_from_nat(start);
        let current_integer=self.builder.arena().int_from_nat(index);
        let mut context=initialization_context(self.builder.arena(),&self.values.selections,&self.values.binders);
        self.values.contents.loop_header(&mut context,self.function.region(body).loop_initialization().expect("loop owns checked initialization"),&arguments,start_integer,current_integer);
        LoopConstruction{schedule,parent,initial,start,end,carries:carries.to_vec()}
    }

    pub(super) fn finish_loop(&mut self,ticket:LoopConstruction) {
        let yielded=Bound::Tuple(ticket.carries.iter().map(|carry|self.bound(carry.yielded)).collect());
        let yielded_type=SemanticType::Tuple(ticket.carries.iter().map(|carry|self.function.value(carry.yielded).ty.clone()).collect());
        let yielded=self.realize_region_product(yielded,&yielded_type);
        let backedge=self.region_operand_product(&yielded);
        let iteration=self.values.contents.clone();
        let symbol=ticket.schedule.binding().symbol;
        let destinations=self.builder.finish_value_repeat(ticket.schedule,backedge);
        self.values=ticket.parent;
        let before=self.values.contents.clone();
        let start=self.builder.arena().int_from_nat(ticket.start);
        let end=self.builder.arena().int_from_nat(ticket.end);
        self.values.contents.completed_loop(&mut InitializationContext::new(self.builder.arena()),&before,&iteration,symbol,start,end);
        let Bound::Tuple(results)=self.region_destination_product(&destinations,&yielded,&iteration,Some((&ticket.initial,&before))) else {unreachable!()};
        for (carry,result) in ticket.carries.iter().zip(results) {
            self.values.bind(self.builder.bindings_mut(),carry.result,result);
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
        if !self.builder.portable_is_root() { return None; }
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
        assert!(self.publication_selection().is_none(), "publication requires its selected construction arm");
        for result in self.function.results() {
            self.materialize_tensor(*result);
            let source = self.bound(*result);
            let target = if matches!(source, Bound::Tensor(_)) {
                let ty = self.function.value(*result).ty.clone();
                let realized = self.realize_region_product(source, &ty);
                let view = *realized.tensor().direct_backing()
                    .expect("publication realization must have an affine descriptor");
                self.builder.portable_publish_tensor(*result, view);
                realized
            } else {
                let target = self.output_target(*result);
                self.assign(&source, &target);
                target
            };
            self.values.rebind(self.builder.bindings_mut(), *result, target);
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
