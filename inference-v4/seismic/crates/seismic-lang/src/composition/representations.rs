//! Storage choices for actual packed load values after logical decomposition.
//! A decoded cache is an ordinary F32 producer; packet access retains the original
//! binding. Both storage and decode arithmetic therefore survive into emission.
use super::*;
mod packets;
pub(crate) use packets::{decode_segment as decode_packet_segment, prepare_coefficients as prepare_packet_coefficients, prepare_words as prepare_packet_words};

type Values = HashMap<VarId, Expr>;

pub(crate) fn select(
    function: &mut LoweredIr,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    block(
        &mut function.body,
        &mut function.vars,
        &Values::new(),
        &HashSet::new(),
        select,
    )
}

fn block(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    inherited: &Values,
    inherited_packets: &HashSet<VarId>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    let mut values = inherited.clone();
    let mut aligned = inherited_packets.clone();
    let mut result = Vec::new();
    for mut statement in std::mem::take(body) {
        let binding = match &statement.kind {
            StmtKind::Assign {
                target:
                    Expr {
                        kind: ExprKind::Var(id),
                        ty: Ty::Tile(shape),
                        ..
                    },
                op: AssignOp::Assign,
                value,
            } if matches!(shape.elem, Elem::Repr(_)) => Some((*id, value.clone())),
            _ => None,
        };
        // A tile alias captures a value at this point, including its original
        // logical offsets. Its hidden decoded producer is immutable thereafter.
        let alias = binding
            .as_ref()
            .and_then(|(_, value)| logical_view(value, &values));
        let packet_binding = binding
            .as_ref()
            .is_some_and(|(_, value)| packets::aligned(value, vars, &aligned));
        match &mut statement.kind {
            StmtKind::Assign { target, value, .. } => {
                read(value, &values);
                // Assignment addresses retain their original physical owner.
                if let ExprKind::Index { indices, .. } = &mut target.kind {
                    for index in indices {
                        match index {
                            Index::Point(e) => read(e, &values),
                            Index::Slice { start, end } => {
                                for e in start.iter_mut().chain(end) {
                                    read(e, &values);
                                }
                            }
                        }
                    }
                }
            }
            StmtKind::Expr(e) => read(e, &values),
            StmtKind::Reduction(r) => {
                for input in &mut r.inputs {
                    // Reducing away the packed axis produces a logical dense
                    // leaf. Other axes may expose packets to the step helper.
                    if input
                        .ty
                        .shaped()
                        .is_some_and(|s| s.packed_axis == Some(r.axis))
                    {
                        if let Some(decoded) = logical_view(input, &values) {
                            *input = decoded;
                        }
                    }
                    read(input, &values);
                }
                select_fold_inputs(r, vars, &aligned, select)?;
                if let Some(step) = &mut r.step {
                    use crate::reduction::structured::{StepState, StepOperand};
                    if let Some(m) = &step.implementation {
                        step.operands = vec![StepOperand::Private; m.right.len()];
                        for (input, parameter) in m.right.iter().enumerate() {
                            if m.can_view_operand(input) {
                                let decision = Decision {
                                    kind: DecisionKind::FoldOperand { input, ty: parameter.ty.clone() },
                                    alternatives: crate::lowered_ir::Alternatives::Explicit(vec![
                                        Alternative::StepOperand(StepOperand::Private),
                                        Alternative::StepOperand(StepOperand::View),
                                    ]),
                                };
                                step.operands[input] = match select(&decision)? {
                                    Alternative::StepOperand(placement) => placement,
                                    _ => return Err("invalid fold operand storage".into()),
                                };
                            }
                        }
                    }
                    if step.implementation.as_ref().is_some_and(|m| m.can_retain_state()) {
                        let decision = Decision {
                            kind: DecisionKind::FoldState { fields: r.state.iter().map(|e| e.ty.clone()).collect() },
                            alternatives: crate::lowered_ir::Alternatives::Explicit(vec![
                                Alternative::StepState(StepState::Separate),
                                Alternative::StepState(StepState::Retained),
                            ]),
                        };
                        step.state = match select(&decision)? {
                            Alternative::StepState(state) => state,
                            _ => return Err("invalid fold state storage".into()),
                        };
                    }
                }
                if let Some(segment) = r.segment.filter(|_| r.step.is_some()) {
                    let decision = Decision {
                        kind: DecisionKind::FoldTraversal { segment, window: r.preparation_window.unwrap_or(segment) },
                        alternatives: crate::lowered_ir::Alternatives::UnrollWidths { maximum: r.preparation_window.unwrap_or(segment) },
                    };
                    r.unroll = match select(&decision)? {
                        Alternative::UnrollWidth(width) if (1..=r.preparation_window.unwrap_or(segment)).contains(&width) => width,
                        _ => return Err("invalid fold traversal width".into()),
                    };
                }
                for step in r.step.iter_mut() {
                    for e in &mut step.identity {
                        read(e, &values);
                    }
                }
            }
            StmtKind::If { cond, .. } => read(cond, &values),
            StmtKind::LoadLoop { domain, views, .. } => {
                read(&mut domain.view, &values);
                for e in views {
                    read(e, &values);
                }
            }
            _ => {}
        }
        let mut nested = values.clone();
        let mut nested_packets = aligned.clone();
        if matches!(
            statement.kind,
            StmtKind::Range { .. }
                | StmtKind::Parallel { .. }
                | StmtKind::Owned { .. }
                | StmtKind::Lanes { .. }
                | StmtKind::LoadLoop { .. }
        ) {
            // A loop body cannot reuse a pre-loop cache for a value carried and
            // mutated by another iteration.
            nested.retain(|&id, _| !crate::effects::tile_mutated(&statement, id));
            nested_packets.retain(|&id| !crate::effects::tile_mutated(&statement, id));
        }
        if let StmtKind::LoadLoop {
            vars: bindings,
            views,
            axes,
            capacity,
            body,
            ..
        } = &mut statement.kind
        {
            // These are actual load owners as well: each invocation creates its
            // own bounded snapshot, so decoding belongs inside the same loop.
            let mut prefix = Vec::new();
            for ((&variable, view), &axis) in bindings.iter().zip(views.iter()).zip(axes.iter()) {
                let packet_aligned = packets::stream_aligned(view, axis, *capacity, vars, &aligned);
                if packet_aligned {
                    nested_packets.insert(variable);
                }
                if matches!(
                    vars[variable].ty.shaped().map(|s| &s.elem),
                    Some(Elem::Repr(_))
                ) {
                    if let Some((cache, producer)) = choose(variable, packet_aligned, vars, select)?
                    {
                        prefix.extend(producer);
                        nested.insert(variable, cache);
                    }
                }
            }
            block(body, vars, &nested, &nested_packets, select)?;
            prefix.append(body);
            *body = prefix;
        } else {
            let mut error = None;
            nested_mut(&mut statement, &mut |body| {
                if error.is_none() {
                    error = block(body, vars, &nested, &nested_packets, select).err();
                }
            });
            if let Some(error) = error {
                return Err(error);
            }
        }
        values.retain(|&id, _| !crate::effects::tile_mutated(&statement, id));
        aligned.retain(|&id| !crate::effects::tile_mutated(&statement, id));
        if let Some((id, _)) = &binding {
            if packet_binding {
                aligned.insert(*id);
            }
        }
        if let Some((id, _)) = &binding {
            if let Some(alias) = alias {
                values.insert(*id, alias);
            }
        }
        let packed_load = binding.as_ref().filter(|(_, value)| {
            matches!(
                value.kind,
                ExprKind::Builtin {
                    name: Builtin::Load,
                    ..
                } | ExprKind::Load { .. }
            )
        });
        result.push(statement);
        if let Some((variable, _)) = packed_load {
            if let Some((cache, producer)) = choose(*variable, packet_binding, vars, select)? {
                result.extend(producer);
                values.insert(*variable, cache);
            }
        }
    }
    *body = result;
    Ok(())
}
fn choose(
    variable: VarId,
    packet_aligned: bool,
    vars: &mut Vec<Var>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<Option<(Expr, Vec<Stmt>)>, String> {
    let mut alternatives = vec![Alternative::Encoded, Alternative::Decoded];
    if packet_aligned && packets::supported(&vars[variable].ty) {
        alternatives.push(Alternative::DecodedPackets);
    }
    let decision = Decision {
        kind: DecisionKind::Representation { variable },
        alternatives: alternatives.into(),
    };
    match select(&decision)? {
        Alternative::Encoded => Ok(None),
        Alternative::Decoded => decode(&super::variable(variable, vars), vars).map(Some),
        Alternative::DecodedPackets if packet_aligned && packets::supported(&vars[variable].ty) => {
            let source = super::variable(variable, vars);
            let Elem::Repr(name) = &source.ty.shaped().unwrap().elem else {
                unreachable!()
            };
            let group = i64::from(crate::repr::lookup(name).unwrap().group);
            let domain = Decision {
                kind: DecisionKind::PacketDecode { variable, group },
                alternatives: crate::lowered_ir::Alternatives::PacketWidths { maximum: group },
            };
            let Alternative::PacketWidth(width) = select(&domain)? else {
                return Err("packet decoding requires a code width".into());
            };
            if !domain
                .alternatives
                .contains(&Alternative::PacketWidth(width))
            {
                return Err("packet width exceeds coefficient group".into());
            }
            let decoder = choose_decoder(variable, width, select)?;
            packets::decode(&source, width as u32, decoder, vars).map(Some)
        }
        _ => Err("invalid packed value storage representation".into()),
    }
}

fn dense_type(ty: &Ty) -> Ty {
    let mut ty = ty.clone();
    if let Ty::Tile(shape) = &mut ty {
        shape.elem = Elem::Dtype(DType::F32);
        shape.packed_axis = None;
    }
    ty
}
fn logical_view(e: &Expr, values: &Values) -> Option<Expr> {
    let mut result = e.clone();
    result.kind = match &e.kind {
        ExprKind::Var(id) => return values.get(id).cloned(),
        ExprKind::Index { base, indices } => ExprKind::Index {
            base: Box::new(logical_view(base, values)?),
            indices: indices.clone(),
        },
        ExprKind::Transpose(base) => ExprKind::Transpose(Box::new(logical_view(base, values)?)),
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => {
            let mut args = args.clone();
            args[0] = logical_view(&args[0], values)?;
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            }
        }
        _ => return None,
    };
    result.ty = dense_type(&result.ty);
    Some(result)
}
fn read(e: &mut Expr, values: &Values) {
    if let ExprKind::Index { base, .. } = &e.kind {
        if matches!(e.ty, Ty::Scalar(DType::F32))
            && matches!(base.ty.shaped().map(|s| &s.elem), Some(Elem::Repr(_)))
        {
            if let Some(decoded) = logical_view(e, values) {
                *e = decoded;
            }
        }
    }
    if let ExprKind::Builtin {
        name: Builtin::Store,
        args,
    } = &mut e.kind
    {
        if let Some(decoded) = logical_view(&args[0], values) {
            args[0] = decoded;
        }
    }
    // Accessors expose packets of the original snapshot, not values of the
    // logical decoded cache. Their indices outside this node are still visited.
    if matches!(e.kind, ExprKind::Accessor { .. }) {
        return;
    }
    children_mut(e, &mut |child| read(child, values));
}
fn decode(source: &Expr, vars: &mut Vec<Var>) -> Result<(Expr, Vec<Stmt>), String> {
    let span = source.span;
    let Ty::Tile(shape) = dense_type(&source.ty) else {
        return Err("decoded cache requires a packed tile load owner".into());
    };
    let id = vars.len();
    vars.push(Var {
        name: format!("decoded_{id}"),
        ty: Ty::Tile(shape.clone()),
        kind: VarKind::Local,
        span,
    });
    let cache = super::variable(id, vars);
    let mut indices = Vec::new();
    for _ in &shape.shape {
        let id = vars.len();
        vars.push(Var {
            name: format!("decode_index_{id}"),
            ty: Ty::Scalar(DType::I32),
            kind: VarKind::Index(Atom::Param(format!("$decode_{id}"))),
            span,
        });
        indices.push(id);
    }
    let coordinates = indices
        .iter()
        .map(|&id| Index::Point(super::variable(id, vars)))
        .collect::<Vec<_>>();
    let element = |base: Expr| Expr {
        kind: ExprKind::Index {
            base: Box::new(base),
            indices: coordinates.clone(),
        },
        ty: Ty::Scalar(DType::F32),
        sym: None,
        span,
    };
    let assignment = Stmt {
        id: None,
        span,
        kind: StmtKind::Assign {
            target: element(cache.clone()),
            op: AssignOp::Assign,
            value: element(source.clone()),
        },
    };
    Ok((
        cache.clone(),
        vec![
            Stmt {
                id: None,
                span,
                kind: StmtKind::Assign {
                    target: cache.clone(),
                    op: AssignOp::Assign,
                    value: Expr {
                        kind: ExprKind::TileAlloc {
                            shape: shape.shape,
                            dtype: Elem::Dtype(DType::F32),
                        },
                        ty: cache.ty.clone(),
                        sym: None,
                        span,
                    },
                },
            },
            Stmt {
                id: None,
                span,
                kind: StmtKind::Owned {
                    vars: indices,
                    tile: cache,
                    body: vec![assignment],
                },
            },
        ],
    ))
}

/// Segment-local preparation is independent of full-value cache selection. It
/// applies to dense or encoded snapshots. Packet preparation additionally needs
/// aligned complete groups; ordinary decoded snapshots cover partial segments.
fn select_fold_inputs(
    reduction: &mut crate::reduction::structured::Reduction,
    vars: &[Var],
    aligned: &HashSet<VarId>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    use crate::reduction::structured::{PreparationScope, InputPreparation, Tree};
    if reduction.step.is_none() || !matches!(reduction.tree, Some(Tree::Pairwise | Tree::Explicit | Tree::SeedThenPairwise))
    {
        return Ok(());
    }
    let Some(segment) = reduction.segment else {
        return Ok(());
    };
    let Some(extent) = reduction.extent().as_constant() else {
        return Ok(());
    };
    if segment <= 0 {
        return Ok(());
    }
    for (input, source) in reduction.inputs.iter().enumerate() {
        let ExprKind::Var(variable) = source.kind else {
            continue;
        };
        let Some(shape) = source.ty.shaped() else {
            continue;
        };
        let encoded = matches!(shape.elem, Elem::Repr(_));
        let mut alternatives = vec![if encoded { Alternative::Encoded } else { Alternative::Direct }];
        if let Elem::Repr(name) = &shape.elem {
            if let Some(repr) = crate::repr::lookup(name) {
                if extent % segment == 0 && shape.packed_axis == Some(reduction.axis)
                    && (segment % i64::from(repr.group) == 0 || i64::from(repr.group) % segment == 0)
                    && packets::supported(&source.ty) && packets::aligned(source, vars, aligned) {
                    alternatives.push(Alternative::DecodedPackets);
                    alternatives.push(Alternative::SegmentSnapshot);
                }
            }
        }
        alternatives.extend([
            Alternative::InputSnapshot(PreparationScope::Segment),
            Alternative::InputSnapshot(PreparationScope::Window),
        ]);
        let decision = Decision {
            kind: DecisionKind::ReductionInput { input, variable, segment },
            alternatives: alternatives.into(),
        };
        let selected = select(&decision)?;
        if !decision.alternatives.contains(&selected) { return Err("fold input preparation is outside its domain".into()); }
        match selected {
            Alternative::Direct | Alternative::Encoded => {}
            Alternative::SegmentSnapshot => reduction.preparation[input] = InputPreparation::EncodedSnapshot,
            Alternative::InputSnapshot(scope) => reduction.preparation[input] = InputPreparation::DecodedSnapshot { scope },
            Alternative::DecodedPackets => {
                reduction.preparation[input] = InputPreparation::Packets { width: 1, decoder: crate::repr::PacketDecoder::Specialized, coefficients: PreparationScope::Window, words: PreparationScope::Window }
            }
            _ => return Err("invalid fold input preparation".into()),
        }
    }
    let groups = reduction
        .inputs
        .iter()
        .zip(&reduction.preparation)
        .filter_map(|(source, preparation)| {
            if !matches!(preparation, InputPreparation::Packets { .. }) {
                return None;
            }
            let Elem::Repr(name) = &source.ty.shaped()?.elem else {
                return None;
            };
            Some(crate::repr::lookup(name)?.group)
        })
        .collect::<Vec<_>>();
    if !groups.is_empty() || reduction.preparation.iter().any(|p| *p == (InputPreparation::DecodedSnapshot { scope: PreparationScope::Window })) {
        let windows = crate::lowered_ir::FoldWindows::new(segment, &groups)?;
        let decision = Decision {
            kind: DecisionKind::FoldPreparation { segment },
            alternatives: crate::lowered_ir::Alternatives::FoldWindows(windows),
        };
        let window = match select(&decision)? {
            Alternative::PreparationWindow(width)
                if decision
                    .alternatives
                    .contains(&Alternative::PreparationWindow(width)) =>
            {
                width
            }
            _ => return Err("invalid fold preparation window".into()),
        };
        reduction.preparation_window = Some(window);
        for (input, (source, preparation)) in reduction.inputs.iter().zip(&mut reduction.preparation).enumerate() {
            if !matches!(preparation, InputPreparation::Packets { .. }) {
                continue;
            }
            let ExprKind::Var(variable) = source.kind else {
                unreachable!()
            };
            let Elem::Repr(name) = &source.ty.shaped().unwrap().elem else {
                unreachable!()
            };
            let group = i64::from(crate::repr::lookup(name).unwrap().group);
            let decision = Decision {
                kind: DecisionKind::PacketDecode { variable, group },
                alternatives: crate::lowered_ir::Alternatives::PacketWidths {
                    maximum: group.min(window),
                },
            };
            let width = match select(&decision)? {
                Alternative::PacketWidth(width) if (1..=group.min(window)).contains(&width) => {
                    width as u32
                }
                _ => return Err("invalid segment packet decode width".into()),
            };
            let decision = Decision {
                kind: DecisionKind::FoldCoefficients { input, segment, window },
                alternatives: vec![
                    Alternative::CoefficientScope(PreparationScope::Window),
                    Alternative::CoefficientScope(PreparationScope::Segment),
                ].into(),
            };
            let coefficients = match select(&decision)? {
                Alternative::CoefficientScope(scope) => scope,
                _ => return Err("invalid fold coefficient scope".into()),
            };
            let mut alternatives = vec![Alternative::WordScope(PreparationScope::Window)];
            if packets::retain_words(&source.ty, segment) { alternatives.push(Alternative::WordScope(PreparationScope::Segment)); }
            let decision = Decision {
                kind: DecisionKind::FoldWords { input, segment, window },
                alternatives: alternatives.into(),
            };
            let words = match select(&decision)? {
                Alternative::WordScope(scope) if decision.alternatives.contains(&Alternative::WordScope(scope)) => scope,
                _ => return Err("invalid fold word scope".into()),
            };
            let decoder = choose_decoder(variable, i64::from(width), select)?;
            *preparation = InputPreparation::Packets { width, decoder, coefficients, words };
        }
    }
    Ok(())
}

fn choose_decoder(variable: VarId, width: i64, select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>) -> Result<crate::repr::PacketDecoder, String> {
    let decision = Decision {
        kind: DecisionKind::PacketDecoder { variable, width },
        alternatives: vec![Alternative::PacketDecoder(crate::repr::PacketDecoder::Specialized), Alternative::PacketDecoder(crate::repr::PacketDecoder::Indexed)].into(),
    };
    match select(&decision)? {
        Alternative::PacketDecoder(decoder) => Ok(decoder),
        _ => Err("invalid packet decoder cover".into()),
    }
}
