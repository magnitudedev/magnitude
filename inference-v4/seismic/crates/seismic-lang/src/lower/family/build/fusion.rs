use super::*;

impl Builder<'_> {
    pub(super) fn compose(&mut self, children: Vec<RegionId>, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let mut concrete = Vec::new();
        for &child in &children {
            let Some(body) = self.closed_region(child) else {
                self.open_fusion_obligations(&children, occurrence, guard);
                return self.retained_producer_family(children, occurrence, guard);
            };
            concrete.extend(body);
        }
        let parallel = concrete.iter().enumerate().filter_map(|(index, statement)|
            (matches!(statement.kind, StmtKind::Parallel { .. })
                && !crate::composition::family::forbids_parallel_fusion(&self.family.template, std::slice::from_ref(statement)))
                .then_some(index)).collect::<Vec<_>>();
        let serial = concrete.iter().enumerate().filter_map(|(index, statement)| matches!(statement.kind, StmtKind::Range { .. } | StmtKind::LoadLoop { .. }).then_some(index)).collect::<Vec<_>>();
        if parallel.len() > 2 || serial.len() > 2 || (parallel.len() == 2 && serial.len() == 2) {
            self.obligation(occurrence, guard, DecisionClass::ParallelFusion,
                "overlapping fusion intervals require retained compatibility and publication constraints");
            return self.retained_producer_family(children, occurrence, guard);
        }
        let mut function = self.family.template.clone(); function.body = concrete.clone();
        let pair = if let [first, second] = parallel.as_slice() {
            crate::composition::family::parallel(&function, &mut self.family.template.vars, *first, *second).map(|fusion| (*first, *second, fusion))
        } else if let [first, second] = serial.as_slice() {
            crate::composition::family::stream(&concrete, &mut self.family.template.vars, *first, *second)
                .or_else(|| crate::composition::family::range(&concrete, &mut self.family.template.vars, *first, *second))
                .map(|fusion| (*first, *second, fusion))
        } else { None };
        let Some((first, second, fusion)) = pair else {
            return self.retained_producer_family(children, occurrence, guard);
        };
        let origin = child_origin(occurrence, "fusion", first, format!("{}:{}", operation(&concrete[first]), operation(&concrete[second])));
        let decision = self.decision(&origin, guard, fusion.domain, 0)?;
        for (ordinal, reason) in fusion.coverage {
            self.obligation(&origin, &guard.with(decision.clone(), ordinal), decision.class, reason);
        }
        let mut arms = Vec::new(); let mut summary = Effects::default();
        for (ordinal, replacement) in fusion.arms.into_iter().enumerate() {
            let active = guard.with(decision.clone(), ordinal);
            let arm_origin = child_origin(&origin, "fusion.arm", ordinal, format!("{ordinal}"));
            // Producer definitions and their consumers can straddle the fused
            // interval. Retain their original common scope so recomputation
            // choices still govern every use of the same reaching value.
            let mut body = concrete[..first].to_vec();
            body.extend(replacement);
            body.extend_from_slice(&concrete[second + 1..]);
            if let Some(producers) = self.producer_family(&body, &arm_origin, &active)? {
                merge_effects(&mut summary, &self.family.regions[producers.0].effects);
                arms.push(producers);
                continue;
            }
            let mut leaves = Vec::new();
            for (index, statement) in body.into_iter().enumerate() {
                let site = child_origin(&origin, "fused.operation", index, operation(&statement));
                let effects = effects(&statement, self.family.template.vars.len());
                merge_effects(&mut summary, &effects);
                leaves.push(self.push(&site, &active, RegionKind::Statement(statement), effects));
            }
            arms.push(self.sequence(leaves, &origin, &active));
        }
        Ok(self.push(&origin, guard, RegionKind::Choice { decision, arms }, summary))
    }

    fn open_fusion_obligations(&mut self, children: &[RegionId], occurrence: &OccurrenceId, guard: &Guard) {
        let mut parallel = 0; let mut serial = 0; let mut reductions = 0;
        for &id in children {
            match &self.family.regions[id.0].kind {
                RegionKind::Repeated { header: Stmt { kind: StmtKind::Parallel { .. }, .. }, .. } => {
                    if !self.forbids_parallel_fusion(id, guard) { parallel += 1; }
                },
                RegionKind::Repeated { header: Stmt { kind: StmtKind::Range { .. }, .. }, .. } | RegionKind::Stream { .. } => serial += 1,
                RegionKind::Reduction(_) => reductions += 1,
                _ => {},
            }
        }
        for (count, class) in [(parallel, DecisionClass::ParallelFusion), (serial, DecisionClass::StreamFusion), (reductions, DecisionClass::ReductionFusion)] {
            if count > 1 { self.obligation(occurrence, guard, class,
                "fusion between parameterized regions requires compatibility over their retained original decisions"); }
        }
    }

    /// A definite publication in every original local alternative rules out
    /// parallel fusion under the same ownership rule as concrete composition.
    /// Walk local arms independently; never choose or multiply their assignments.
    fn forbids_parallel_fusion(&self, id: RegionId, known: &Guard) -> bool {
        let region = &self.family.regions[id.0];
        if !region.guard.choices.iter().all(|choice| known.choices.contains(choice))
            || !region.guard.predicates.iter().all(|predicate| known.predicates.contains(predicate))
            || !region.guard.one_of.iter().all(|clause| known.one_of.contains(clause)) {
            return false;
        }
        let statement = |statement| crate::composition::family::forbids_parallel_fusion(
            &self.family.template, std::slice::from_ref(statement));
        match &region.kind {
            RegionKind::Statement(value) => statement(value),
            RegionKind::Sequence(children) => children.iter().any(|&child| self.forbids_parallel_fusion(child, known)),
            RegionKind::Choice { decision, arms } => {
                if let Some((_, ordinal)) = known.choices.iter().find(|(id, _)| id == decision) {
                    return arms.get(*ordinal).is_some_and(|&arm| self.forbids_parallel_fusion(arm, known));
                }
                !arms.is_empty() && arms.iter().enumerate().all(|(ordinal, &arm)|
                    self.forbids_parallel_fusion(arm, &known.with(decision.clone(), ordinal)))
            },
            RegionKind::Repeated { header, body, .. } | RegionKind::Stream { header, body, .. } =>
                statement(header) || self.forbids_parallel_fusion(*body, known),
            RegionKind::Replicated { body, .. } => self.forbids_parallel_fusion(*body, known),
            RegionKind::Conditional { header, then, els } => statement(header)
                || self.forbids_parallel_fusion(*then, known) || self.forbids_parallel_fusion(*els, known),
            // Reduction expansion can change which callback statements are
            // present. Leave that proof to its retained applicability relation.
            RegionKind::Reduction(_) => false,
        }
    }
}
