//! Symbolic scalar recurrence. A body is interpreted once under its typed
//! header; the closed Fold denotes all visits, including the zero-trip case.
use super::*;
use seismic_ir::region::{Product, RepeatCarry, ScalarOperand, ValueDestination, ValueOperand};

impl Terms {
    pub(in crate::portable) fn fold(
        &mut self,
        start: Term,
        end: Term,
        initial: Vec<Term>,
        next: Vec<Term>,
    ) -> Vec<Term> {
        assert_eq!(initial.len(), next.len());
        if matches!((&self.nodes[start.0],&self.nodes[end.0]),(Node::Natural(a),Node::Natural(b)) if a>=b)
        {
            return initial;
        }
        (0..initial.len())
            .map(|output| {
                self.node(Node::Fold {
                    start,
                    end,
                    initial: initial.clone(),
                    next: next.clone(),
                    output: output as u32,
                })
            })
            .collect()
    }
}
impl Analysis<'_> {
    pub(in crate::portable) fn schedule_scalar_operand(
        &mut self,
        operand: ScalarOperand,
        state: &State,
    ) -> Result<Term> {
        match operand {
            ScalarOperand::Natural(value) => self.expression(value.into(), &state.slots),
            ScalarOperand::Word { symbol, .. } => state
                .slots
                .get(&symbol)
                .or_else(|| self.inputs.get(&symbol))
                .copied()
                .ok_or("loop operand is unavailable"),
        }
    }
    pub(in crate::portable) fn scalar_schedule_repeat<B: seismic_native_target::TargetFamily>(
        &mut self,
        schedule: &ParametricSchedule<B>,
        kernels: &seismic_ir::kernel::KernelArena<B>,
        symbol: seismic_lang::expr::SymbolId,
        start: Term,
        end: Term,
        body: &[ScheduleStep],
        carries: &Product<RepeatCarry>,
        state: &mut State,
    ) -> Result<()> {
        let mut leaves = Vec::new();
        carries.visit(&mut |carry| leaves.push(carry.clone()));
        let mut initial = Vec::new();
        let mut iteration = state.clone();
        let depth = self.loop_depth;
        iteration
            .slots
            .insert(symbol, self.terms.node(Node::Iteration(depth)));
        for (ordinal, carry) in leaves.iter().enumerate() {
            let (ValueOperand::Scalar(input), ValueDestination::Scalar(header)) =
                (carry.initial(), carry.header())
            else {
                return Err("tensor recurrence relation is unfinished");
            };
            initial.push(self.schedule_scalar_operand(input, state)?);
            iteration.slots.insert(
                header.symbol(),
                self.terms.node(Node::Header {
                    depth,
                    ordinal: ordinal as u32,
                    kind: header.kind(),
                }),
            );
        }
        self.loop_depth += 1;
        let result = self.schedule(schedule, kernels, body, &mut iteration);
        self.loop_depth = depth;
        result?;
        if iteration.writes.len() != state.writes.len()
            || iteration.effects.len() != state.effects.len()
        {
            return Err("stateful recurrence relation is unfinished");
        }
        let mut next = Vec::new();
        for carry in &leaves {
            let ValueOperand::Scalar(value) = carry.backedge() else {
                return Err("tensor recurrence backedge relation is unfinished");
            };
            next.push(self.schedule_scalar_operand(value, &iteration)?);
        }
        for (carry, value) in leaves
            .iter()
            .zip(self.terms.fold(start, end, initial, next))
        {
            let ValueDestination::Scalar(result) = carry.result() else {
                return Err("tensor recurrence result relation is unfinished");
            };
            state.slots.insert(result.symbol(), value);
        }
        Ok(())
    }
    pub(in crate::portable) fn scalar_kernel_repeat<B: seismic_native_target::TargetFamily>(
        &mut self,
        kernel: &seismic_ir::kernel::Kernel<B>,
        start: Term,
        end: Term,
        binder: ops::ErasedValue,
        initial: Vec<Term>,
        headers: &[ops::ErasedValue],
        body: seismic_ir::kernel::BlockId,
        values: &HashMap<ops::ErasedValue, Term>,
        state: &State,
    ) -> Result<Vec<Term>> {
        if headers.len() != initial.len() {
            return Err("repeat scalar product arity differs");
        }
        let depth = self.loop_depth;
        let mut values = values.clone();
        let mut iteration = state.clone();
        values.insert(binder, self.terms.node(Node::Iteration(depth)));
        for (ordinal, header) in headers.iter().enumerate() {
            let kind = match kernel.value_type(*header) {
                ValueType::Index => ScalarKind::Nat64,
                ty => ScalarKind::Scalar(dtype(ty)?),
            };
            values.insert(
                *header,
                self.terms.node(Node::Header {
                    depth,
                    ordinal: ordinal as u32,
                    kind,
                }),
            );
        }
        self.loop_depth += 1;
        let next = self.physical_block(kernel, body, &mut values, &mut iteration);
        self.loop_depth = depth;
        let next = next?;
        if iteration.writes.len() != state.writes.len()
            || iteration.effects.len() != state.effects.len()
        {
            return Err("stateful kernel recurrence relation is unfinished");
        }
        Ok(self.terms.fold(start, end, initial, next))
    }
}
