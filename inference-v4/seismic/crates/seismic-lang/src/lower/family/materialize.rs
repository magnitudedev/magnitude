use super::*;

pub(super) fn instantiate(family: &ExecutionFamily, assignment: &Assignment) -> Result<LoweredIr, String> {
    for id in assignment.keys() {
        if !family.decisions.iter().any(|decision| &decision.id == id) {
            return Err(format!("assignment contains an unknown source decision {id:?}"));
        }
    }
    let mut selected = BTreeMap::new();
    let mut symbols = family.template.shapes.iter().map(|(name, &value)| (name.clone(), value)).collect::<BTreeMap<_, _>>();
    for decision in &family.decisions {
        if !decision.guard.active(assignment)? {
            if assignment.contains_key(&decision.id) {
                return Err(format!("assignment contains inactive source decision {:?}", decision.id));
            }
            continue;
        }
        let ordinal = assignment.get(&decision.id).ok_or_else(|| format!("missing source decision {:?}", decision.id))?;
        let alternative = decision.domain.alternatives.get(*ordinal)
            .ok_or_else(|| format!("source decision {:?} is outside its original domain", decision.id))?;
        if let Some(parameter) = &decision.numeric {
            let value = decision.domain.alternatives.numeric().and_then(|numeric| numeric.value(*ordinal))
                .or_else(|| match alternative { Alternative::StreamCapacity(value) => Some(value), _ => None })
                .ok_or("numeric source decision has no numeric alternative")?;
            let Atom::Param(name) = &parameter.atom else { return Err("family parameter must be a named atom".into()); };
            symbols.insert(name.clone(), value);
        }
        selected.insert(decision.id.clone(), alternative);
    }
    for obligation in &family.obligations {
        if obligation.guard.active(assignment)? {
            return Err(format!("incomplete source family {:?} at {:?}: {}", obligation.class, obligation.occurrence, obligation.reason));
        }
    }
    for requirement in &family.requirements {
        if requirement.guard.active(assignment)? {
            match requirement.nonnegative.eval(&|name| symbols.get(name).copied()) {
                Some(value) if value >= 0 => {},
                Some(_) => return Err("selected body violates a retained applicability requirement".into()),
                None => return Err("body requirement still depends on unproved runtime values".into()),
            }
        }
    }
    let mut function = family.template.clone();
    function.body = region(family, family.root, assignment, &selected, &symbols)?;
    let environment = symbols.iter().map(|(name, &value)| (name.clone(), Sym::constant(value))).collect::<std::collections::HashMap<_, _>>();
    let atoms = environment.iter().map(|(name, value)| (Atom::Param(name.clone()), value.clone())).collect::<Vec<_>>();
    for statement in &mut function.body { crate::composition::remap(statement, &std::collections::HashMap::new(), &atoms); }
    for variable in &mut function.vars { variable.ty = crate::lower::subst_ty(&variable.ty, &environment); }
    for decision in &family.decisions {
        let Some(alternative) = selected.get(&decision.id) else { continue; };
        function.decisions.push(DecisionRecord { domain: decision.domain.clone(), selected: alternative.clone() });
        if let (DecisionKind::Construct { name, shape_args, .. }, Alternative::Body(choice)) = (&decision.domain.kind, alternative) {
            function.selections.push(Selection { construct: name.clone(),
                shape_args: shape_args.iter().map(|shape| shape.eval(&|name| symbols.get(name).copied()).unwrap_or(-1)).collect(),
                choice: choice.clone() });
        }
    }
    crate::normalize::identify(&mut function.body, &mut 0);
    crate::verify::lowered(&function, crate::verify::Stage::Expanded)?;
    Ok(function)
}

fn region(family: &ExecutionFamily, id: RegionId, assignment: &Assignment, selected: &BTreeMap<DecisionId, Alternative>, symbols: &BTreeMap<String, i64>) -> Result<Vec<Stmt>, String> {
    let retained = family.regions.get(id.0).ok_or("invalid retained region identity")?;
    if !retained.guard.active(assignment)? { return Ok(Vec::new()); }
    match &retained.kind {
        RegionKind::Sequence(children) => {
            let mut body = Vec::new();
            for &child in children { body.extend(region(family, child, assignment, selected, symbols)?); }
            Ok(body)
        },
        RegionKind::Statement(statement) => Ok(vec![statement.clone()]),
        RegionKind::Choice { decision, arms } => {
            let ordinal = assignment.get(decision).ok_or("missing body decision")?;
            let arm = arms.get(*ordinal).ok_or("body assignment is outside retained arms")?;
            region(family, *arm, assignment, selected, symbols)
        },
        RegionKind::Repeated { header, body, .. } => {
            let mut statement = header.clone();
            let retained = region(family, *body, assignment, selected, symbols)?;
            match &mut statement.kind {
                StmtKind::Parallel { body, .. } | StmtKind::Range { body, .. }
                | StmtKind::Owned { body, .. } | StmtKind::Lanes { body, .. } => *body = retained,
                _ => return Err("repeat has an invalid typed header".into()),
            }
            Ok(vec![statement])
        },
        RegionKind::Replicated { index, count, body } => {
            let count = count.eval(&|name| symbols.get(name).copied()).ok_or("replication count is unresolved")?;
            if count < 0 { return Err("replication count is negative".into()); }
            let VarKind::Index(atom) = &family.template.vars[*index].kind else { return Err("replication index has no symbolic identity".into()); };
            let original = region(family, *body, assignment, selected, symbols)?;
            let mut result = Vec::new();
            for ordinal in 0..count {
                let mut copy = original.clone();
                let value = crate::reduction::structured::integer(ordinal, family.template.vars[*index].span);
                crate::widen::replace_index(&mut copy, *index, atom, &value);
                result.extend(copy);
            }
            Ok(result)
        },
        RegionKind::Conditional { header, then, els } => {
            let then_active = family.regions[then.0].guard.active(assignment)?;
            let else_active = family.regions[els.0].guard.active(assignment)?;
            if !then_active { return if else_active { region(family, *els, assignment, selected, symbols) } else { Ok(Vec::new()) }; }
            if !else_active { return region(family, *then, assignment, selected, symbols); }
            let mut statement = header.clone();
            let then_body = region(family, *then, assignment, selected, symbols)?;
            let else_body = region(family, *els, assignment, selected, symbols)?;
            let StmtKind::If { then, els, .. } = &mut statement.kind else { return Err("conditional has an invalid typed header".into()); };
            *then = then_body; *els = else_body;
            Ok(vec![statement])
        },
        RegionKind::Stream { header, body, capacity, .. } => {
            let Some(Alternative::StreamCapacity(value)) = selected.get(capacity) else { return Err("stream has no assigned capacity".into()); };
            if *value <= 0 { return Err("stream capacity must be positive".into()); }
            let mut statement = header.clone();
            let selected_body = region(family, *body, assignment, selected, symbols)?;
            let StmtKind::LoadLoop { body, capacity, .. } = &mut statement.kind else { return Err("stream has an invalid typed header".into()); };
            *capacity = Some(*value); *body = selected_body;
            Ok(vec![statement])
        },
        RegionKind::Reduction(reduction) => {
            let mut operation = reduction.operation.clone();
            for callback in &reduction.callbacks {
                let body = region(family, callback.body, assignment, selected, symbols)?;
                let implementation = match callback.role {
                    CallbackRole::Merge => operation.implementation.as_mut(),
                    CallbackRole::Step => operation.step.as_mut().and_then(|step| step.implementation.as_mut()),
                }.ok_or("retained callback has no parameter bindings")?;
                implementation.body = body;
            }
            let Some(Alternative::ReductionTree(tree)) = selected.get(&reduction.tree) else { return Err("missing retained reduction tree".into()); };
            operation.tree = Some(*tree);
            if *tree == crate::reduction::structured::Tree::Explicit {
                let explicit = reduction.explicit.as_ref().ok_or("explicit reduction tree metadata is absent")?;
                let leaves = explicit.leaves.eval(&|name| symbols.get(name).copied())
                    .ok_or("explicit reduction leaf count is unresolved")?;
                if leaves < 2 { return Err("explicit reduction tree needs at least two leaves".into()); }
                let mut stack = vec![(0i64, 1i64)];
                let mut next_leaf = 1i64;
                let mut branches = Vec::new();
                for merge in &explicit.merges {
                    if !merge.guard.active(assignment)? { continue; }
                    let Some(Alternative::ReductionFrontier(frontier)) = selected.get(&merge.frontier) else {
                        return Err("missing explicit reduction frontier".into());
                    };
                    if *frontier < next_leaf || *frontier > leaves {
                        return Err("explicit reduction frontier is outside its monotone leaf range".into());
                    }
                    while next_leaf < *frontier {
                        stack.push((next_leaf, next_leaf + 1));
                        next_leaf += 1;
                    }
                    let right = stack.pop().ok_or("explicit reduction stack underflow")?;
                    let left = stack.pop().ok_or("explicit reduction stack underflow")?;
                    if left.1 != right.0 { return Err("explicit reduction children are not contiguous".into()); }
                    branches.push(crate::reduction::structured::Branch { start: left.0, cut: right.0, end: right.1 });
                    stack.push((left.0, right.1));
                }
                if next_leaf != leaves || stack.len() != 1 || stack[0] != (0, leaves) {
                    return Err("explicit reduction frontiers do not cover the root".into());
                }
                // Frontiers construct children before parents. Reverse that
                // order for the retained branch contract in linear time.
                branches.reverse();
                operation.branches = branches;
            }
            for (guard, segment, _) in &reduction.segments {
                if guard.active(assignment)? {
                    let Some(Alternative::ReductionSegment(value)) = selected.get(segment) else { return Err("missing retained reduction segment".into()); };
                    operation.segment = Some(*value);
                }
            }
            for preparation in &reduction.preparation {
                if !preparation.guard.active(assignment)? { continue; }
                for (input, decision) in &preparation.inputs {
                    use crate::reduction::structured::InputPreparation;
                    operation.preparation[*input] = match selected.get(decision).ok_or("missing fold input preparation")? {
                        Alternative::Direct | Alternative::Encoded => InputPreparation::Direct,
                        Alternative::InputSnapshot(scope) => InputPreparation::DecodedSnapshot { scope: *scope },
                        Alternative::SegmentSnapshot => InputPreparation::EncodedSnapshot,
                        Alternative::DecodedPackets => InputPreparation::Direct,
                        _ => return Err("invalid retained fold preparation alternative".into()),
                    };
                }
                if let Some(window) = preparation.window.as_ref().and_then(|decision| selected.get(decision)) {
                    let Alternative::PreparationWindow(width) = window else { return Err("invalid fold preparation window".into()); };
                    operation.preparation_window = Some(*width);
                }
                for (guard, decision) in &preparation.traversals {
                    if guard.active(assignment)? {
                        let Some(Alternative::UnrollWidth(width)) = selected.get(decision) else { return Err("missing fold traversal width".into()); };
                        operation.unroll = *width;
                    }
                }
                for packet in &preparation.packets {
                    if !packet.guard.active(assignment)? { continue; }
                    let Some(Alternative::PacketWidth(width)) = selected.get(&packet.width) else { return Err("missing packet preparation width".into()); };
                    let Some(Alternative::PacketDecoder(decoder)) = selected.get(&packet.decoder) else { return Err("missing packet preparation decoder".into()); };
                    let Some(Alternative::CoefficientScope(coefficients)) = selected.get(&packet.coefficients) else { return Err("missing packet coefficient lifetime".into()); };
                    let Some(Alternative::WordScope(words)) = selected.get(&packet.words) else { return Err("missing packet word lifetime".into()); };
                    operation.preparation[packet.input] = crate::reduction::structured::InputPreparation::Packets {
                        width: u32::try_from(*width).map_err(|_| "packet width exceeds u32")?, decoder: *decoder, coefficients: *coefficients, words: *words,
                    };
                }
            }
            if let Some(step) = &mut operation.step {
                for (input, decision) in &reduction.operands {
                    let Some(Alternative::StepOperand(operand)) = selected.get(decision) else { return Err("missing fold operand ownership".into()); };
                    *step.operands.get_mut(*input).ok_or("invalid retained fold operand")? = *operand;
                }
                if let Some(decision) = &reduction.state {
                    let Some(Alternative::StepState(state)) = selected.get(decision) else { return Err("missing fold state ownership".into()); };
                    step.state = *state;
                }
            }
            if let Some(decomposition) = &reduction.decomposition {
                if decomposition.guard.active(assignment)? {
                    let Some(Alternative::StreamCapacity(capacity)) = selected.get(&decomposition.decision) else { return Err("missing ordered reduction decomposition capacity".into()); };
                    let extent = decomposition.geometry.extent.eval(&|name| symbols.get(name).copied()).ok_or("selected decomposition extent is unresolved")?;
                    if *capacity < extent {
                        let instantiate_piece = |piece: &ReductionPiece| {
                            let mut body = piece.setup.clone();
                            let mut piece_operation = piece.operation.clone();
                            piece_operation.tree = operation.tree;
                            piece_operation.implementation = operation.implementation.clone();
                            piece_operation.step = operation.step.clone();
                            body.push(Stmt { id: None, span: operation.span, kind: StmtKind::Reduction(Box::new(piece_operation)) });
                            body
                        };
                        let mut body = vec![Stmt { id: None, span: operation.span, kind: StmtKind::Range {
                            var: decomposition.index, lo: Sym::constant(0), hi: Sym::constant(extent / *capacity),
                            body: instantiate_piece(&decomposition.full),
                        } }];
                        if extent % *capacity != 0 { body.extend(instantiate_piece(&decomposition.tail)); }
                        return Ok(body);
                    }
                }
            }
            if let Some(decomposition) = &reduction.dynamic_decomposition {
                if decomposition.guard.active(assignment)? {
                    let Some(Alternative::StreamCapacity(value)) = selected.get(&decomposition.decision) else { return Err("missing dynamic reduction decomposition capacity".into()); };
                    let mut header = decomposition.header.clone();
                    let mut piece = decomposition.operation.clone();
                    piece.tree = operation.tree; piece.step = operation.step.clone();
                    piece.implementation = operation.implementation.clone();
                    let StmtKind::LoadLoop { body, capacity, .. } = &mut header.kind else { return Err("dynamic decomposition has an invalid header".into()); };
                    *capacity = Some(*value);
                    *body = vec![Stmt { id: None, span: operation.span, kind: StmtKind::Reduction(Box::new(piece)) }];
                    return Ok(vec![header]);
                }
            }
            Ok(vec![Stmt { id: None, span: operation.span, kind: StmtKind::Reduction(Box::new(operation)) }])
        },
    }
}
