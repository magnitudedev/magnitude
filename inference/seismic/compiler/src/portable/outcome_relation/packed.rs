//! Checked packed reads use the registry's same sealed decode sequence as IR.
use super::*;
use seismic_lang::intrinsics::MathOp;
use seismic_lang::reference_math::ScalarOp;
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
        let target = recipe
            .target(component)
            .ok_or("representation lacks checked numerical component")?;
        recipe.decode(
            &mut Decode {
                analysis: self,
                view,
                indices,
                state,
            },
            target,
        )
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

/// A checked read's decode: plane reads project through the view; every
/// numerical step is its scalar reference recipe, which must be total.
struct Decode<'d, 'a, 'i> {
    analysis: &'d mut Analysis<'a>,
    view: AnyBufferView,
    indices: &'i [Term],
    state: &'i State,
}

impl Decode<'_, '_, '_> {
    fn scalar(&mut self, op: ScalarOp, dtypes: &[DType], inputs: &[Term]) -> Result<Term> {
        self.analysis
            .total_recipe(&reference::scalar_recipe(op, dtypes), inputs)
    }
}

impl registry::DecodeDomain for Decode<'_, '_, '_> {
    type Value = Term;
    type Error = &'static str;

    fn read_plane_field(&mut self, plane: &registry::PlaneSchema, field: u32) -> Result<Term> {
        self.analysis
            .plane_field(self.view, self.indices, plane.ordinal, field, self.state)
    }

    fn read_plane_lane(&mut self, plane: &registry::PlaneSchema) -> Result<Term> {
        self.analysis
            .plane_lane(self.view, self.indices, plane.ordinal, self.state)
    }

    fn interpret_code(
        &mut self,
        raw: &Term,
        _: u32,
        bits: u32,
        interpretation: &registry::CodeInterpretation,
    ) -> Result<Term> {
        self.analysis
            .total_recipe(&reference::code_recipe(interpretation, bits), &[*raw])
    }

    fn signed_code(&mut self, code: &Term, raw: &Term, bit: u32) -> Result<Term> {
        self.analysis
            .total_recipe(&reference::signed_code_recipe(bit), &[*code, *raw])
    }

    fn interpret_grid_code(
        &mut self,
        raw: &Term,
        lane: &Term,
        grid: &'static registry::GridCodes,
    ) -> Result<Term> {
        self.analysis
            .total_recipe(&reference::grid_code_recipe(grid), &[*raw, *lane])
    }

    fn decode_float_code(&mut self, raw: &Term, format: registry::FloatCodeFormat) -> Result<Term> {
        self.analysis
            .total_recipe(&reference::float_code_recipe(format), &[*raw])
    }

    fn convert_to_f32(&mut self, from: &Term, from_dtype: DType) -> Result<Term> {
        self.scalar(ScalarOp::Cast(DType::F32), &[from_dtype], &[*from])
    }

    fn constant(&mut self, value: reference::ReferenceScalar) -> Result<Term> {
        Ok(self.analysis.terms.scalar(value))
    }

    fn multiply(&mut self, left: &Term, right: &Term) -> Result<Term> {
        self.scalar(
            ScalarOp::Binary(BinaryOp::Mul),
            &[DType::F32, DType::F32],
            &[*left, *right],
        )
    }

    fn negate(&mut self, from: &Term) -> Result<Term> {
        self.scalar(ScalarOp::Unary(UnaryOp::Neg), &[DType::F32], &[*from])
    }

    fn multiply_add(&mut self, factor: &Term, multiplicand: &Term, addend: &Term) -> Result<Term> {
        self.scalar(
            ScalarOp::Math(MathOp::Fma),
            &[DType::F32, DType::F32, DType::F32],
            &[*factor, *multiplicand, *addend],
        )
    }

    fn cast(&mut self, from: &Term, from_dtype: DType, to: DType) -> Result<Term> {
        self.scalar(ScalarOp::Cast(to), &[from_dtype], &[*from])
    }
}
