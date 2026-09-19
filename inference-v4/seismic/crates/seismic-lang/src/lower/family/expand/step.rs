use super::*;

#[derive(Clone)]
struct Prepared {
    decision: Option<DecisionId>,
    alternatives: Vec<(usize, Expr, Sym)>,
}

impl Expander {
    pub(super) fn step_allocations(&mut self, reduction: &ReductionFamily, site: &Site) -> Result<RegionId, String> {
        let implementation = reduction.operation.step.as_ref().and_then(|step| step.implementation.as_ref()).ok_or("fold has no step parameter bindings")?.clone();
        let mut setup = vec![self.allocations(&implementation.left, site)];
        if let Some(decision) = &reduction.state {
            let separate = self.allocations(&implementation.output, &self.active(site, decision, 0));
            let retained = self.sequence(Vec::new(), &self.active(site, decision, 1));
            setup.push(self.branch(decision, vec![separate, retained], site));
        } else { setup.push(self.allocations(&implementation.output, site)); }
        Ok(self.sequence(setup, site))
    }
    pub(super) fn step_visit(&mut self, reduction: &ReductionFamily, state: &[Expr], inputs: &[Expr], site: &Site) -> Result<(RegionId, RegionId), String> {
        let prepared = inputs.iter().map(|input| Prepared { decision: None, alternatives: vec![(0, input.clone(), Sym::constant(0))] }).collect::<Vec<_>>();
        self.prepared_visit(reduction, state, &prepared, None, site)
    }
    fn prepared_visit(&mut self, reduction: &ReductionFamily, state: &[Expr], inputs: &[Prepared], at: Option<&Expr>, site: &Site) -> Result<(RegionId, RegionId), String> {
        let implementation = reduction.operation.step.as_ref().and_then(|step| step.implementation.as_ref()).ok_or("fold has no step parameter bindings")?.clone();
        let mut setup = Vec::new();
        let mut visit = vec![self.copies(&implementation.left, state, site)];
        // A visit owns its operand bindings. Whole, full, tail and prepared
        // visits may have different backing storage, so sharing a callback
        // parameter identity would merge otherwise disjoint borrow lifetimes.
        let mut rename = HashMap::new();
        let targets = implementation.right.iter().map(|parameter| {
            let ExprKind::Var(original) = parameter.kind else { return Err("step input has no stable parameter binding"); };
            let target = self.ordinary(site).local(parameter.ty.clone());
            let ExprKind::Var(fresh) = target.kind else { unreachable!() };
            rename.insert(original, fresh); Ok(target)
        }).collect::<Result<Vec<_>, _>>()?;
        for (input, (target, prepared)) in targets.iter().zip(inputs).enumerate() {
            let operand = reduction.operands.iter().find(|(candidate, _)| *candidate == input).map(|(_, decision)| decision);
            if let Some(decision) = operand {
                let private = self.allocations(std::slice::from_ref(target), &self.active(site, decision, 0));
                let view = self.sequence(Vec::new(), &self.active(site, decision, 1));
                setup.push(self.branch(decision, vec![private, view], site));
            } else { setup.push(self.allocations(std::slice::from_ref(target), site)); }
            let mut ownership = Vec::new();
            for mode in 0..(if operand.is_some() { 2 } else { 1 }) {
                let ownership_site = operand.map_or_else(|| site.clone(), |decision| self.active(site, decision, mode));
                let mut preparations = Vec::new();
                for (ordinal, source, offset) in &prepared.alternatives {
                    let active = prepared.decision.as_ref().map_or_else(|| ownership_site.clone(), |decision| self.active(&ownership_site, decision, *ordinal));
                    let value = at.map_or_else(|| source.clone(), |index| structured::slice(source, reduction.operation.axis,
                        &symbol(index.sym.as_ref().unwrap().sub(offset), site.span), site.span));
                    let prepared = if mode == 1 {
                        let borrowed = Expr { kind: ExprKind::Load { view: Box::new(value), mode: LoadMode::Borrow }, ty: target.ty.clone(), sym: None, span: site.span };
                        self.statement(stmt(StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value: borrowed }, site.span), &active)
                    } else { self.copies(std::slice::from_ref(target), std::slice::from_ref(&value), &active) };
                    preparations.push(prepared);
                }
                ownership.push(if let Some(decision) = &prepared.decision { self.branch(decision, preparations, &ownership_site) } else { preparations[0] });
            }
            // Keep the ownership branch outside preparation choices. Its
            // immutable selector is the same one guarding private allocation,
            // so repeated visits retain that source lifetime fact directly.
            visit.push(if let Some(decision) = operand { self.branch(decision, ownership, site) } else { ownership[0] });
        }
        let callback = self.callback(reduction, CallbackRole::Step, site)?;
        let callback = self.remap_region(callback, &rename, &[], site)?;
        if let Some(decision) = &reduction.state {
            let separate_site = self.active(site, decision, 0);
            let result = self.copies(state, &implementation.output, &separate_site);
            let separate = self.sequence(vec![callback, result], &separate_site);
            let retained_site = self.active(site, decision, 1);
            let rename = implementation.output.iter().zip(&implementation.left).map(|(output, left)| {
                let (ExprKind::Var(output), ExprKind::Var(left)) = (&output.kind, &left.kind) else { return Err("retained step parameters need stable bindings"); };
                Ok((*output, *left))
            }).collect::<Result<HashMap<_, _>, _>>()?;
            let retained = self.remap_region(callback, &rename, &[], &retained_site)?;
            let result = self.copies(state, &implementation.left, &retained_site);
            let retained = self.sequence(vec![retained, result], &retained_site);
            visit.push(self.branch(decision, vec![separate, retained], site));
        } else {
            visit.push(callback); visit.push(self.copies(state, &implementation.output, site));
        }
        Ok((self.sequence(setup, site), self.sequence(visit, site)))
    }
    pub(super) fn segmented(&mut self, reduction: &ReductionFamily, tree: Tree, site: &Site) -> Result<RegionId, String> {
        let (_, _, geometry) = reduction.segments.iter().find(|(guard, _, _)| guard.choices.iter().all(|choice| site.guard.choices.contains(choice)))
            .ok_or("segmented tree has no original capacity geometry")?;
        let preparation = reduction.preparation.iter().find(|preparation| preparation.guard.choices.iter().all(|choice| site.guard.choices.contains(choice)))
            .ok_or("segmented fold has no preparation family")?;
        let step = reduction.operation.step.as_ref().ok_or("segmented operation has no step")?;
        let implementation = step.implementation.as_ref().ok_or("segmented step has no bindings")?;
        let partial = implementation.left.clone();
        let mut setup = vec![self.step_allocations(reduction, site)?];
        let mut identity = Vec::new(); let mut leaves = Vec::new(); let mut ordinary = Vec::new();
        for (state, initial) in reduction.operation.state.iter().zip(&step.identity) {
            let value = self.ordinary(site).alloc(&initial.ty, &mut ordinary);
            ordinary.push(self.ordinary(site).copy(&value, initial)); identity.push(value);
            let mut shape = state.ty.shaped().ok_or("fold state has no shape")?.clone(); shape.shape.insert(0, geometry.visits.clone());
            leaves.push(self.ordinary(site).alloc(&Ty::Tile(shape), &mut ordinary));
        }
        setup.push(self.statements(ordinary, site));
        let group = self.ordinary(site).index(); let start = group.sym.as_ref().unwrap().mul(&geometry.capacity);
        let mut group_body = vec![self.copies(&partial, &identity, site)];
        let mut prepared = reduction.operation.inputs.iter().map(|source| Prepared { decision: None, alternatives: vec![(0, source.clone(), Sym::constant(0))] }).collect::<Vec<_>>();
        for (input, decision) in &preparation.inputs {
            let source = &reduction.operation.inputs[*input];
            let domain = self.family.decisions.iter().find(|entry| entry.id == *decision).ok_or("fold input decision is absent")?.domain.alternatives.clone();
            let mut alternatives = Vec::new();
            for ordinal in 0..domain.len() {
                let active = self.active(site, decision, ordinal);
                match domain.get(ordinal).ok_or("invalid fold input ordinal")? {
                    Alternative::InputSnapshot(structured::PreparationScope::Segment) => {
                        let (cache, body) = self.snapshot(source, reduction.operation.axis, start.clone(), geometry.capacity.clone(), geometry.extent.clone(), &active)?;
                        group_body.push(body); alternatives.push((ordinal, cache, start.clone()));
                    },
                    Alternative::SegmentSnapshot => {
                        let (cache, body) = self.encoded_snapshot(source, reduction.operation.axis, start.clone(), geometry.capacity.clone(), &active)?;
                        group_body.push(body); alternatives.push((ordinal, cache, start.clone()));
                    },
                    Alternative::DecodedPackets => {
                        let packet = preparation.packets.iter().find(|packet| packet.input == *input
                            && packet.guard.choices.iter().all(|choice| active.guard.choices.contains(choice)))
                            .ok_or("decoded packet input has no retained preparation parameters")?;
                        let width = self.numeric(&packet.width)?;
                        let decoder_domain = self.family.decisions.iter().find(|entry| entry.id == packet.decoder)
                            .ok_or("decoded packet input has no decoder domain")?.domain.alternatives.clone();
                        let coefficients_domain = self.family.decisions.iter().find(|entry| entry.id == packet.coefficients)
                            .ok_or("decoded packet input has no coefficient lifetime domain")?.domain.alternatives.clone();
                        let words_domain = self.family.decisions.iter().find(|entry| entry.id == packet.words)
                            .ok_or("decoded packet input has no word lifetime domain")?.domain.alternatives.clone();
                        let mut dense_shape = source.ty.shaped().ok_or("decoded packet input has no shape")?.clone();
                        let dtype = dense_shape.elem.read_dtype().ok_or("decoded packet input has no decoded dtype")?;
                        dense_shape.elem = Elem::Dtype(dtype);
                        dense_shape.shape[reduction.operation.axis] = geometry.capacity.clone();
                        dense_shape.packed_axis = None;
                        let mut cache_setup = Vec::new();
                        let cache = self.ordinary(&active).alloc(&Ty::Tile(dense_shape), &mut cache_setup);
                        group_body.push(self.statements(cache_setup, &active));
                        let mut decoder_arms = Vec::new();
                        for decoder_ordinal in 0..decoder_domain.len() {
                            let Alternative::PacketDecoder(decoder) = decoder_domain.get(decoder_ordinal).ok_or("invalid packet decoder ordinal")? else {
                                return Err("packet decoder domain contains a non-decoder arm".into());
                            };
                            let mut coefficient_arms = Vec::new();
                            for _coefficient_ordinal in 0..coefficients_domain.len() {
                                let mut word_arms = Vec::new();
                                for _word_ordinal in 0..words_domain.len() {
                                    // Coefficient and word lifetime choices are
                                    // retained as source arms. Direct packet
                                    // access remains valid for every arm while
                                    // symbolic widths are unresolved; later
                                    // materialization may replace these reads
                                    // with the selected cache.
                                    let (decoded_cache, producer) = crate::composition::decode_segment_parameterized(
                                        source,
                                        start.clone(),
                                        geometry.capacity.clone(),
                                        width.clone(),
                                        decoder,
                                        None,
                                        None,
                                        &mut self.family.template.vars,
                                    )?;
                                    let produced = self.statements(producer, &active);
                                    let copied = self.copies(std::slice::from_ref(&cache), std::slice::from_ref(&decoded_cache), &active);
                                    word_arms.push(self.sequence(vec![produced, copied], &active));
                                }
                                coefficient_arms.push(self.branch(&packet.words, word_arms, &active));
                            }
                            decoder_arms.push(self.branch(&packet.coefficients, coefficient_arms, &active));
                        }
                        let decoded = self.branch(&packet.decoder, decoder_arms, &active);
                        group_body.push(decoded);
                        alternatives.push((ordinal, cache, start.clone()));
                    },
                    _ => alternatives.push((ordinal, source.clone(), Sym::constant(0))),
                }
            }
            prepared[*input] = Prepared { decision: Some(decision.clone()), alternatives };
        }
        for (guard, traversal) in &preparation.traversals {
            let active = Site { guard: guard.clone(), ..site.clone() };
            let requested = preparation.window.as_ref().and_then(|window| self.family.decisions.iter().find(|entry| entry.id == *window))
                .is_some_and(|entry| entry.guard == *guard);
            let window = if requested { self.numeric(preparation.window.as_ref().unwrap())? } else { geometry.capacity.clone() };
            let width = self.numeric(traversal)?;
            let window_index = self.ordinary(&active).index();
            let offset = start.add(&window_index.sym.as_ref().unwrap().mul(&window));
            let body = self.fold_window(reduction, &partial, &prepared, offset, window.clone(), width.clone(), geometry.extent.clone(), &active)?;
            group_body.push(self.range(&window_index, Sym::constant(0), geometry.capacity.quot(&window), body, &active));
            let tail = geometry.capacity.rem(&window);
            let tail_site = self.predicate(&active, tail.sub(&Sym::constant(1)))?;
            group_body.push(self.fold_window(reduction, &partial, &prepared, start.add(&geometry.capacity.sub(&tail)), tail, width, geometry.extent.clone(), &tail_site)?);
        }
        let targets = leaves.iter().map(|leaf| structured::slice(leaf, 0, &group, site.span)).collect::<Vec<_>>();
        group_body.push(self.copies(&targets, &partial, site));
        let group_body = self.sequence(group_body, site);
        setup.push(self.range(&group, Sym::constant(0), geometry.visits.clone(), group_body, site));
        let merge = if tree == Tree::Explicit {
            let explicit = reduction.explicit.as_ref().ok_or("explicit segmented tree metadata is absent")?;
            self.explicit_values(reduction, explicit, &leaves, 0, site)?
        } else {
            self.pairwise(reduction, &leaves, 0, geometry.visits.clone(), tree == Tree::SeedThenPairwise, site)?
        };
        setup.push(merge);
        Ok(self.sequence(setup, site))
    }
    fn fold_window(&mut self, reduction: &ReductionFamily, partial: &[Expr], prepared: &[Prepared], start: Sym, length: Sym, width: Sym, extent: Sym, site: &Site) -> Result<RegionId, String> {
        let mut prepared = prepared.to_vec(); let mut setup = Vec::new();
        for (input, prepared) in prepared.iter_mut().enumerate() {
            let Some(decision) = &prepared.decision else { continue; };
            let domain = self.family.decisions.iter().find(|entry| entry.id == *decision).ok_or("window input decision is absent")?.domain.alternatives.clone();
            for (ordinal, value, offset) in &mut prepared.alternatives {
                if domain.get(*ordinal) != Some(Alternative::InputSnapshot(structured::PreparationScope::Window)) { continue; }
                let active = self.active(site, decision, *ordinal);
                let (cache, body) = self.snapshot(&reduction.operation.inputs[input], reduction.operation.axis, start.clone(), length.clone(), extent.clone(), &active)?;
                setup.push(body); *value = cache; *offset = start.clone();
            }
        }
        let chunk = self.ordinary(site).index(); let copy = self.ordinary(site).index();
        let at = start.add(&chunk.sym.as_ref().unwrap().mul(&width)).add(copy.sym.as_ref().unwrap());
        let (inputs, visit) = self.prepared_visit(reduction, partial, &prepared, Some(&symbol(at.clone(), site.span)), site)?;
        setup.push(inputs);
        let empty = self.sequence(Vec::new(), site);
        let within_extent = self.conditional(condition(at.clone(), BinaryOp::Lt, extent, site.span), visit, empty, site);
        let empty = self.sequence(Vec::new(), site);
        let within_window = self.conditional(condition(at, BinaryOp::Lt, start.add(&length), site.span), within_extent, empty, site);
        let ExprKind::Var(index) = copy.kind else { unreachable!() };
        let effects = self.family.regions[within_window.0].effects.clone();
        let replicated = self.push(site, RegionKind::Replicated { index, count: width.clone(), body: within_window }, effects);
        setup.push(self.range(&chunk, Sym::constant(0), length.add(&width).sub(&Sym::constant(1)).quot(&width), replicated, site));
        Ok(self.sequence(setup, site))
    }
    fn snapshot(&mut self, source: &Expr, axis: usize, start: Sym, length: Sym, extent: Sym, site: &Site) -> Result<(Expr, RegionId), String> {
        let mut shape = source.ty.shaped().ok_or("snapshot input has no shape")?.clone();
        let dtype = shape.elem.read_dtype().ok_or("snapshot element type is unresolved")?;
        shape.shape[axis] = length; shape.elem = Elem::Dtype(dtype); shape.packed_axis = None;
        let mut setup = Vec::new(); let cache = self.ordinary(site).alloc(&Ty::Tile(shape.clone()), &mut setup);
        let indices = (0..shape.shape.len()).map(|_| self.ordinary(site).index()).collect::<Vec<_>>();
        let at = start.add(indices[axis].sym.as_ref().unwrap());
        let mut source_indices = indices.iter().cloned().map(Index::Point).collect::<Vec<_>>(); source_indices[axis] = Index::Point(symbol(at.clone(), site.span));
        let value = Expr { kind: ExprKind::Index { base: Box::new(source.clone()), indices: source_indices }, ty: Ty::Scalar(dtype), sym: None, span: site.span };
        let target = Expr { kind: ExprKind::Index { base: Box::new(cache.clone()), indices: indices.iter().cloned().map(Index::Point).collect() }, ty: Ty::Scalar(dtype), sym: None, span: site.span };
        let copy = stmt(StmtKind::Assign { target, op: AssignOp::Assign, value }, site.span);
        setup.push(stmt(StmtKind::Owned { vars: indices.iter().map(|index| { let ExprKind::Var(id) = index.kind else { unreachable!() }; id }).collect(), tile: cache.clone(),
            body: vec![stmt(StmtKind::If { cond: condition(at, BinaryOp::Lt, extent, site.span), then: vec![copy], els: Vec::new() }, site.span)] }, site.span));
        Ok((cache, self.statements(setup, site)))
    }
    fn encoded_snapshot(&mut self, source: &Expr, axis: usize, start: Sym, length: Sym, site: &Site) -> Result<(Expr, RegionId), String> {
        let mut shape = source.ty.shaped().ok_or("encoded snapshot input has no shape")?.clone(); shape.shape[axis] = length.clone();
        let mut indices = vec![Index::Slice { start: None, end: None }; shape.shape.len()];
        indices[axis] = Index::Slice { start: Some(symbol(start.clone(), site.span)), end: Some(symbol(start.add(&length), site.span)) };
        let ty = Ty::Tile(shape); let target = self.ordinary(site).local(ty.clone());
        let view = Expr { kind: ExprKind::Index { base: Box::new(source.clone()), indices }, ty: ty.clone(), sym: None, span: site.span };
        let value = Expr { kind: ExprKind::Load { view: Box::new(view), mode: LoadMode::Materialize }, ty, sym: None, span: site.span };
        let body = self.statement(stmt(StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value }, site.span), site);
        Ok((target, body))
    }
}
