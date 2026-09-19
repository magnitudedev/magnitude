use super::*;
use crate::reduction::structured::{PreparationScope, Reduction};

impl Builder<'_> {
    pub(super) fn fold_preparation(&mut self, operation: &Reduction, geometry: &StreamGeometry, tree: usize,
        occurrence: &OccurrenceId, guard: &Guard) -> Result<FoldPreparationFamily, String> {
        let maximum = self.numeric_bound(&geometry.capacity).ok_or("segment has no finite original domain")?;
        let origin = child_origin(occurrence, "fold.preparation", tree, operation.merge_name().into());
        let mut inputs = Vec::new(); let mut requesting = Vec::new(); let mut no_window = guard.clone();
        let mut packet_inputs = Vec::new();
        for (input, source) in operation.inputs.iter().enumerate() {
            let ExprKind::Var(variable) = source.kind else { continue; };
            let Some(shape) = source.ty.shaped() else { continue; };
            let encoded = matches!(shape.elem, Elem::Repr(_));
            let packet = match &shape.elem {
                Elem::Repr(name) if shape.packed_axis == Some(operation.axis)
                    && crate::composition::packet_supported(&source.ty)
                    && crate::composition::packet_aligned(source, &self.family.template.vars, &self.packet_aligned) => crate::repr::lookup(name),
                _ => None,
            };
            let mut alternatives = vec![if encoded { Alternative::Encoded } else { Alternative::Direct }];
            if packet.is_some() { alternatives.extend([Alternative::DecodedPackets, Alternative::SegmentSnapshot]); }
            let segment_ordinal = alternatives.len();
            alternatives.push(Alternative::InputSnapshot(PreparationScope::Segment));
            let window_ordinal = alternatives.len();
            alternatives.push(Alternative::InputSnapshot(PreparationScope::Window));
            let id = self.decision(&origin, guard, Decision { kind: DecisionKind::ReductionInput { input, variable, segment: maximum }, alternatives: alternatives.into() }, input)?;
            requesting.push((id.clone(), window_ordinal));
            let mut without_window = vec![(id.clone(), 0), (id.clone(), segment_ordinal)];
            if let Some(packet) = packet {
                requesting.push((id.clone(), 1));
                without_window.push((id.clone(), 2));
                let group = Sym::constant(i64::from(packet.group));
                let aligned = geometry.capacity.rem(&group).mul(&group.rem(&geometry.capacity));
                // Both packed preparations capture exact complete segments.
                for ordinal in [1, 2] {
                    let active = guard.with(id.clone(), ordinal);
                    self.family.requirements.push(Requirement { guard: active.clone(), nonnegative: Sym::constant(0).sub(&geometry.extent.rem(&geometry.capacity)) });
                    self.family.requirements.push(Requirement { guard: active, nonnegative: Sym::constant(0).sub(&aligned) });
                }
                packet_inputs.push((input, variable, packet, guard.with(id.clone(), 1)));
            } else if encoded && shape.shape.iter().any(|extent| extent.as_constant().is_none()) {
                self.obligation(&origin, guard, DecisionClass::PacketDecode,
                    "packet alternatives need alignment provenance across a parameterized source slice");
            }
            no_window.one_of.push(without_window);
            inputs.push((input, id));
        }
        let mut traversals = Vec::new(); let mut packets = Vec::new();
        let window = if requesting.is_empty() { None } else {
            let mut active = guard.clone(); active.one_of.push(requesting);
            let id = self.decision(&origin, &active, Decision { kind: DecisionKind::FoldPreparation { segment: maximum },
                alternatives: Alternatives::FoldWindows(crate::lowered_ir::FoldWindows::new(maximum, &[])?) }, 0)?;
            let parameter = self.family.decisions.last().unwrap().numeric.clone().unwrap();
            let window = Sym::atom(parameter.atom);
            self.family.requirements.push(Requirement { guard: active.clone(), nonnegative: geometry.capacity.sub(&window) });
            for (input, variable, packet, packet_guard) in packet_inputs {
                let group = i64::from(packet.group);
                let group_sym = Sym::constant(group);
                // The union window domain remains compact. Each selected
                // packet input contributes its own exact alignment relation,
                // so independent inputs share one window without a product of
                // completed preparation programs.
                let whole_groups = window.rem(&group_sym).add(&geometry.capacity.rem(&group_sym));
                let subgroups = group_sym.rem(&window).add(&geometry.capacity.rem(&window));
                self.family.requirements.push(Requirement { guard: packet_guard.clone(),
                    nonnegative: Sym::constant(0).sub(&whole_groups.mul(&subgroups)) });
                let width = self.decision(&origin, &packet_guard, Decision { kind: DecisionKind::PacketDecode { variable, group },
                    alternatives: Alternatives::PacketWidths { maximum: group.min(maximum) } }, input)?;
                let width_sym = Sym::atom(self.family.decisions.last().unwrap().numeric.clone().unwrap().atom);
                self.family.requirements.push(Requirement { guard: packet_guard.clone(), nonnegative: window.sub(&width_sym) });
                let coefficients = self.decision(&origin, &packet_guard, Decision { kind: DecisionKind::FoldCoefficients { input, segment: maximum, window: maximum },
                    alternatives: vec![Alternative::CoefficientScope(PreparationScope::Window), Alternative::CoefficientScope(PreparationScope::Segment)].into() }, input)?;
                let words = self.decision(&origin, &packet_guard, Decision { kind: DecisionKind::FoldWords { input, segment: maximum, window: maximum },
                    alternatives: vec![Alternative::WordScope(PreparationScope::Window), Alternative::WordScope(PreparationScope::Segment)].into() }, input)?;
                self.family.requirements.push(Requirement { guard: packet_guard.with(words.clone(), 1),
                    nonnegative: Sym::constant(0).sub(&geometry.capacity.mul(&Sym::constant(i64::from(packet.bits))).rem(&Sym::constant(32))) });
                let decoder = self.decision(&origin, &packet_guard, Decision { kind: DecisionKind::PacketDecoder { variable, width: group.min(maximum) },
                    alternatives: vec![Alternative::PacketDecoder(crate::repr::PacketDecoder::Specialized), Alternative::PacketDecoder(crate::repr::PacketDecoder::Indexed)].into() }, input)?;
                packets.push(PacketPreparationFamily { guard: packet_guard, input, width, decoder, coefficients, words });
            }
            let traversal = self.decision(&origin, &active, Decision { kind: DecisionKind::FoldTraversal { segment: maximum, window: maximum },
                alternatives: Alternatives::UnrollWidths { maximum } }, 0)?;
            let width = Sym::atom(self.family.decisions.last().unwrap().numeric.clone().unwrap().atom);
            self.family.requirements.push(Requirement { guard: active.clone(), nonnegative: window.sub(&width) });
            traversals.push((active, traversal));
            Some(id)
        };
        let traversal = self.decision(&origin, &no_window, Decision { kind: DecisionKind::FoldTraversal { segment: maximum, window: maximum },
            alternatives: Alternatives::UnrollWidths { maximum } }, 1)?;
        let width = Sym::atom(self.family.decisions.last().unwrap().numeric.clone().unwrap().atom);
        self.family.requirements.push(Requirement { guard: no_window.clone(), nonnegative: geometry.capacity.sub(&width) });
        traversals.push((no_window, traversal));
        Ok(FoldPreparationFamily { guard: guard.clone(), segment: geometry.capacity.clone(), inputs, window, traversals, packets })
    }
}
