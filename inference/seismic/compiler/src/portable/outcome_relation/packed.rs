//! Checked packed reads use the registry's same sealed decode sequence as IR.
use super::*;
use seismic_lang::intrinsics::MathOp;
use seismic_lang::reference_math::ScalarOp;
use seismic_lang::registry::DecodeStep;
use seismic_lang::syntax::ast::{BinaryOp, UnaryOp};

impl Analysis<'_> {
    pub(super) fn source_tensor_read(
        &mut self,
        view: AnyBufferView,
        indices: &[Term],
        state: &State,
    ) -> Result<Term> {
        self.source_tensor_component(view, indices, state, None)
    }
    pub(super) fn source_tensor_component(
        &mut self,
        view: AnyBufferView,
        indices: &[Term],
        state: &State,
        component: Option<registry::QuantizedComponent>,
    ) -> Result<Term> {
        if matches!(
            registry::representation_info(view.representation()).kind,
            RepresentationKind::Dense(_)
        ) {
            if component.is_some() {
                return Err("dense representation has no quantized component");
            }
            let place = self.place(view, indices, state)?;
            return self.read(state, place);
        }
        let info = registry::representation_info(view.representation());
        let recipe = registry::decode_recipe(info.id, info.decoded)
            .ok_or("source packet has no registered decode")?;
        if let Some(registry::DecodeComponent::Constant(value)) =
            component.and_then(|role| recipe.component(role))
        {
            return Ok(self.terms.scalar(value));
        }
        let mut values = Vec::with_capacity(recipe.temporary_count());
        for step in recipe.steps() {
            let get = |temp| values[recipe.ordinal(temp)];
            let value = match step {
                DecodeStep::ReadPlaneField { plane, field, .. } => {
                    self.plane_field(view, indices, *plane, *field, state)?
                }
                DecodeStep::InterpretCode {
                    raw,
                    bits,
                    interpretation,
                    ..
                } => {
                    self.total_recipe(&reference::code_recipe(interpretation, *bits), &[get(*raw)])?
                }
                DecodeStep::DecodeFloatCode { raw, format, .. } => {
                    self.total_recipe(&reference::float_code_recipe(*format), &[get(*raw)])?
                }
                DecodeStep::ConvertToF32 { from, .. } => self.total_recipe(
                    &reference::scalar_recipe(ScalarOp::Cast(DType::F32), &[recipe.dtype(*from)]),
                    &[get(*from)],
                )?,
                DecodeStep::Multiply { left, right, .. } => self.total_recipe(
                    &reference::scalar_recipe(
                        ScalarOp::Binary(BinaryOp::Mul),
                        &[DType::F32, DType::F32],
                    ),
                    &[get(*left), get(*right)],
                )?,
                DecodeStep::Negate { from, .. } => self.total_recipe(
                    &reference::scalar_recipe(ScalarOp::Unary(UnaryOp::Neg), &[DType::F32]),
                    &[get(*from)],
                )?,
                DecodeStep::MultiplyAdd {
                    factor,
                    multiplicand,
                    addend,
                    ..
                } => self.total_recipe(
                    &reference::scalar_recipe(
                        ScalarOp::Math(MathOp::Fma),
                        &[DType::F32, DType::F32, DType::F32],
                    ),
                    &[get(*factor), get(*multiplicand), get(*addend)],
                )?,
                DecodeStep::Cast { from, to, .. } => self.total_recipe(
                    &reference::scalar_recipe(ScalarOp::Cast(*to), &[recipe.dtype(*from)]),
                    &[get(*from)],
                )?,
            };
            values.push(value);
        }
        Ok(match component {
            None => values[recipe.ordinal(recipe.output())],
            Some(role) => match recipe
                .component(role)
                .ok_or("representation lacks checked numerical component")?
            {
                registry::DecodeComponent::Temporary(temp) => values[recipe.ordinal(temp)],
                registry::DecodeComponent::Constant(value) => self.terms.scalar(value),
            },
        })
    }
    fn total_recipe(
        &mut self,
        recipe: &reference::ReferenceRecipe,
        inputs: &[Term],
    ) -> Result<Term> {
        let (value, failures) = self.terms.recipe(recipe, inputs);
        if !failures.is_empty() {
            return Err("registered packet decode is not total");
        }
        Ok(value)
    }
}
