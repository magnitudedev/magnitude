//! Source call lowering: argument bindings and initialization facts handed to
//! the implementation builder's call construction.
use super::*;

impl<'f, 'b, B: seismic_native_target::TargetFamily> Lowerer<'f, 'b, B> {
    pub(super) fn initialization_scalar(&mut self, bound: &Bound) -> InitializationArgument {
        match bound {
            Bound::Scalar(scalar) => match prepare_scalar(self.builder.arena(), *scalar) {
                PreparedArg::Index(value) => InitializationArgument::Integer(self.builder.arena().int_from_nat(value)),
                PreparedArg::Integer(value) => InitializationArgument::Integer(value),
                PreparedArg::Scalar(symbol, DType::I32) => {
                    let value = self.builder.arena().scalar_symbol::<seismic_lang::expr::I32>(symbol);
                    InitializationArgument::Integer(self.builder.arena().int_from_scalar(value))
                }
                PreparedArg::Scalar(symbol, DType::U32) => {
                    let value = self.builder.arena().scalar_symbol::<seismic_lang::expr::U32>(symbol);
                    InitializationArgument::Integer(self.builder.arena().int_from_scalar(value))
                }
                PreparedArg::Scalar(_, DType::Bool) => InitializationArgument::Predicate { value: condition_expr(self.builder.arena(), &self.values.host_conditions, bound), binders: self.values.binders.clone() },
                _ => InitializationArgument::Unknown,
            },
            Bound::Range { start, end } => {
                match (self.initialization_scalar(start), self.initialization_scalar(end)) {
                    (InitializationArgument::Integer(start), InitializationArgument::Integer(end)) => InitializationArgument::Range { start, end },
                    _ => InitializationArgument::Unknown,
                }
            }
            _ => InitializationArgument::Unknown,
        }
    }

    pub(super) fn initialization_arguments<'v>(&mut self, bounds: &'v [Bound]) -> Vec<CallArgument<'v>> {
        bounds.iter().map(|bound| match bound {
            Bound::Tensor(TensorRealization::Stored(stored)) => CallArgument::Tensor(stored),
            Bound::Tensor(TensorRealization::Computed(plan)) => {
                let axes = plan.axes.iter().map(|axis| self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                CallArgument::Computed(InitializationContext::new(self.builder.arena()).root(&axes))
            }
            other => CallArgument::Scalar(self.initialization_scalar(other)),
        }).collect()
    }

    pub(super) fn prepare_call(&mut self, id: NodeId, inputs: &[SemanticValueId]) -> (crate::implementation::CallConstruction<B>, Vec<Bound>) {
        let arguments = inputs.iter().map(|value| self.builder.bindings().selected(self.values.handle(*value), &self.values.selections)).collect::<Vec<_>>();
        let bounds = inputs.iter().map(|value| self.bound(*value)).collect::<Vec<_>>();
        let initialized_arguments = self.initialization_arguments(&bounds);
        let progress = self.builder.begin_call(id, &arguments, &self.values.selections, &self.values.contents, &initialized_arguments, &self.values.binders);
        (progress, bounds)
    }

    pub(super) fn finish_call(&mut self, progress: crate::implementation::CallConstruction<B>, bounds: &[Bound]) {
        let initialized_arguments = self.initialization_arguments(bounds);
        let call = self.builder.finish_call(progress, &self.values.selections, &mut self.values.contents, &initialized_arguments, &self.values.binders);
        for (output, binding) in &call {
            let slot = self.values.slot(*output);
            assert!(self.values.values[slot].replace(*binding).is_none(), "call result was bound twice");
        }
    }
}
