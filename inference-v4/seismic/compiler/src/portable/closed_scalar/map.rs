//! Independent parallel maps. A checked `parallel for` has independent
//! visits: no visit reads what another writes, so the visit order is
//! unobservable. Both projections interpret one symbolic visit `i` (an
//! iteration term) over `[0, extent)` and relate the per-visit writes.
use super::*;

impl Analysis<'_> {
    /// Interpret one visit and record it as a map effect. `visit` runs the
    /// body on `lane_state`, a copy of `state` whose path knows `i < extent`.
    pub(in crate::portable) fn map_visit(
        &mut self,
        state: &mut State,
        extent: Term,
        visit: &mut dyn FnMut(&mut Self, Term, &mut State) -> Result<()>,
    ) -> Result<()> {
        let depth = self.loop_depth;
        let iteration = self.terms.node(Node::Iteration(depth));
        let mut lane = state.clone();
        let active = self.terms.compare(reference::CmpOp::Lt, iteration, extent);
        lane.path.enter_arm(active, true);
        let (effects, writes) = (lane.effects.len(), lane.writes.len());
        self.loop_depth += 1;
        let result = visit(self, iteration, &mut lane);
        self.loop_depth = depth;
        result?;
        self.close_map(state, lane, effects, writes, extent)
    }

    fn close_map(
        &mut self,
        state: &mut State,
        lane: State,
        effects: usize,
        writes: usize,
        extent: Term,
    ) -> Result<()> {
        // Values a visit defines are local to it (a checked parallel loop
        // carries nothing out), so its slots end with the visit.
        let mut observed = Vec::new();
        for effect in &lane.effects[effects..] {
            match effect {
                Effect::Write(place, value) => observed.push((place.clone(), *value)),
                Effect::Failure(..) | Effect::Map { .. } => {
                    return Err("parallel map failure or nested map relation is unfinished")
                }
            }
        }
        let mut roots = Vec::new();
        for write in &lane.writes[writes..] {
            if !roots.contains(&write.root()) {
                roots.push(write.root());
            }
        }
        state.writes.extend(roots.into_iter().map(|root| MemoryWrite::Map { root }));
        if !observed.is_empty() {
            state.effects.push(Effect::Map { extent, writes: observed });
        }
        Ok(())
    }

    /// A compiler-owned logical launch: participant `p` executes logical
    /// index `min(p, extent)`, and only indices below the extent are active.
    /// The active lanes are the map's visits; every other lane must be
    /// provably inert.
    pub(in crate::portable) fn physical_map<B: seismic_native_target::TargetFamily>(
        &mut self,
        launch: &seismic_ir::schedule::Launch<B>,
        kernel: &seismic_ir::kernel::Kernel<B>,
        state: &mut State,
    ) -> Result<()> {
        let (Some(base), Some(extent)) = (launch.logical_base, launch.parallel_extent) else {
            return Err("participant-map relation of an authored launch is unfinished");
        };
        if base.is_chunked()
            || launch.grid[1..].iter().chain(&launch.workgroup[1..]).any(|axis| {
                !matches!(self.expressions.view((*axis).into()), seismic_lang::expr::NodeView::NatConst(1))
            })
        {
            return Err("chunked or multi-axis participant-map relation is unfinished");
        }
        if !matches!(self.expressions.view(base.value().into()), seismic_lang::expr::NodeView::NatConst(0)) {
            return Err("offset participant-map relation is unfinished");
        }
        let extent = self.expression(extent.into(), &state.slots)?;
        // Inactive lanes: the participant is at or beyond the extent.
        let mut inert = state.clone();
        let (effects, writes) = (inert.effects.len(), inert.writes.len());
        let beyond = self.terms.opaque();
        self.lane = Some(Lane { participant: beyond, bound: extent, active: false });
        let result = self.physical_block(kernel, kernel.root(), &mut HashMap::new(), &mut inert);
        self.lane = None;
        result?;
        if inert.effects.len() != effects || inert.writes.len() != writes
            || inert.slots.iter().any(|(symbol, value)| state.slots.get(symbol) != Some(value))
        {
            return Err("an inactive participant's inertness is not established");
        }
        self.map_visit(state, extent, &mut |analysis, participant, lane| {
            analysis.lane = Some(Lane { participant, bound: extent, active: true });
            let result = analysis.physical_block(kernel, kernel.root(), &mut HashMap::new(), lane);
            analysis.lane = None;
            result.map(|_| ())
        })
    }
}
