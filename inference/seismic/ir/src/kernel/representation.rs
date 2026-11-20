//! Instantiation of the registry's existing decode sequence before kernel closure.
//! Field extraction is a physical read; all numerical steps use scalar recipes.
use super::internals::{PlaceEntry, PortableBuilder, PortableValue};
use super::ops::{BinaryOp, UnaryOp, ValueType};
use crate::physical_target::PhysicalDialect;
use seismic_lang::reference_math;
use seismic_lang::registry;
use seismic_lang::types::DType;

pub(super) fn read<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    place: PlaceEntry,
    indices: &[PortableValue],
) -> PortableValue {
    read_component(builder, place, indices, None)
}

pub(super) fn read_component<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    place: PlaceEntry,
    indices: &[PortableValue],
    component: Option<registry::QuantizedComponent>,
) -> PortableValue {
    let info = registry::representation_info(place.representation);
    let recipe = registry::decode_recipe(place.representation, info.decoded)
        .expect("registered packed representation has a decode sequence");
    let target = recipe
        .target(component)
        .expect("checked representation supports role");
    let Ok(value) = recipe.decode(
        &mut Lowering {
            builder,
            place,
            indices,
        },
        target,
    );
    value
}

/// A decode's IR: plane reads are physical reads of `place` at `indices`;
/// every numerical step is a scalar recipe or a primitive op.
struct Lowering<'b, 'p, 'i, B: PhysicalDialect> {
    builder: &'b mut PortableBuilder<'p, B>,
    place: PlaceEntry,
    indices: &'i [PortableValue],
}

impl<B: PhysicalDialect> registry::DecodeDomain for Lowering<'_, '_, '_, B> {
    type Value = PortableValue;
    type Error = std::convert::Infallible;

    fn dtype(value: &PortableValue) -> Option<DType> {
        match value.ty {
            ValueType::Scalar(dtype) => Some(dtype),
            other => panic!("a decode step lowered to a value of type {other:?}"),
        }
    }

    fn read_plane_field(
        &mut self,
        plane: &registry::PlaneSchema,
        field: u32,
    ) -> Result<PortableValue, Self::Error> {
        Ok(self
            .builder
            .read_plane_field(self.place, self.indices, plane.ordinal, field))
    }

    fn read_plane_lane(
        &mut self,
        plane: &registry::PlaneSchema,
    ) -> Result<PortableValue, Self::Error> {
        Ok(self
            .builder
            .read_plane_lane(self.place, self.indices, plane.ordinal))
    }

    fn interpret_code(
        &mut self,
        raw: &PortableValue,
        _: u32,
        bits: u32,
        interpretation: &registry::CodeInterpretation,
    ) -> Result<PortableValue, Self::Error> {
        let scalar = reference_math::code_recipe(interpretation, bits);
        Ok(super::reference_math::expand_total_recipe(
            self.builder,
            &scalar,
            &[*raw],
        ))
    }

    fn signed_code(
        &mut self,
        code: &PortableValue,
        raw: &PortableValue,
        bit: u32,
    ) -> Result<PortableValue, Self::Error> {
        let scalar = reference_math::signed_code_recipe(bit);
        Ok(super::reference_math::expand_total_recipe(
            self.builder,
            &scalar,
            &[*code, *raw],
        ))
    }

    fn interpret_grid_code(
        &mut self,
        raw: &PortableValue,
        lane: &PortableValue,
        grid: &'static registry::GridCodes,
    ) -> Result<PortableValue, Self::Error> {
        let scalar = reference_math::grid_code_recipe(grid);
        Ok(super::reference_math::expand_total_recipe(
            self.builder,
            &scalar,
            &[*raw, *lane],
        ))
    }

    fn decode_float_code(
        &mut self,
        raw: &PortableValue,
        format: registry::FloatCodeFormat,
    ) -> Result<PortableValue, Self::Error> {
        let scalar = reference_math::float_code_recipe(format);
        Ok(super::reference_math::expand_total_recipe(
            self.builder,
            &scalar,
            &[*raw],
        ))
    }

    fn convert_to_f32(
        &mut self,
        from: &PortableValue,
        _: DType,
    ) -> Result<PortableValue, Self::Error> {
        Ok(self.builder.cast(*from, ValueType::Scalar(DType::F32)))
    }

    fn constant(
        &mut self,
        value: reference_math::ReferenceScalar,
    ) -> Result<PortableValue, Self::Error> {
        Ok(self.builder.constant(
            super::ops::ConstantValue::from_scalar(value),
            ValueType::Scalar(value.dtype()),
        ))
    }

    fn multiply(
        &mut self,
        left: &PortableValue,
        right: &PortableValue,
    ) -> Result<PortableValue, Self::Error> {
        Ok(self.builder.binary(BinaryOp::Mul, *left, *right))
    }

    fn negate(&mut self, from: &PortableValue) -> Result<PortableValue, Self::Error> {
        Ok(self.builder.unary(UnaryOp::Neg, *from))
    }

    fn multiply_add(
        &mut self,
        factor: &PortableValue,
        multiplicand: &PortableValue,
        addend: &PortableValue,
    ) -> Result<PortableValue, Self::Error> {
        Ok(self.builder.fma(*factor, *multiplicand, *addend))
    }

    fn cast(
        &mut self,
        from: &PortableValue,
        _: DType,
        to: DType,
    ) -> Result<PortableValue, Self::Error> {
        Ok(self.builder.cast(*from, ValueType::Scalar(to)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::Construction;
    use crate::kernel::ops::Op;
    use crate::physical_target::{
        IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, VectorSupport,
    };
    use crate::storage::GlobalBufferKind;
    use seismic_lang::expr::ExprArena;

    #[derive(Debug)]
    struct Dialect;
    #[derive(Clone, Debug)]
    enum NoIntrinsic {}
    impl PhysicalDialect for Dialect {
        type LaunchDescriptor = ();
        fn ordinary_launch() {}
        const NAME: registry::BackendName = registry::BackendName::Cpu;
        type DeviceInfo = ();
        type Intrinsic = NoIntrinsic;
        fn write_intrinsic_identity(op: &NoIntrinsic, _: &mut IntrinsicIdentityBuilder) {
            match *op {}
        }
        fn intrinsic_numerics(
            _: &(),
            _: &registry::IntrinsicSignature,
            op: &NoIntrinsic,
        ) -> IntrinsicNumericalSemantics {
            match *op {}
        }
        fn intrinsic_addressable_resources(
            op: &NoIntrinsic,
        ) -> Vec<crate::kernel::ops::AddressableResourceHandle> {
            match *op {}
        }
    }

    #[test]
    fn every_packed_read_closes_with_typed_fields_and_terminal_scalar_work() {
        for info in registry::representations() {
            let registry::RepresentationKind::Packed(layout) = &info.kind else {
                continue;
            };
            let mut arena = ExprArena::default();
            let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            let n = arena.nat(u64::from(layout.group));
            let (_, view) = construction.storage_mut().tensor(
                &mut arena,
                GlobalBufferKind::Arena,
                info.id,
                vec![n],
            );
            let view = construction.view(view, info.id);
            let vectors = VectorSupport::default();
            let mut builder = construction.portable_kernel(&mut arena, &(), &[], &vectors);
            let place = builder.arg_view(view, false);
            let index = builder.index_constant(7);
            builder.read(place, &[index]);
            builder.close();
            let kernel = &construction.kernels()[0];
            let mut fields = 0;
            let mut word_work = 0;
            for op in kernel.blocks().iter().flat_map(|block| &block.ops) {
                match op {
                    Op::ReadPlaneField {
                        out, plane, field, ..
                    } => {
                        fields += 1;
                        let schema = &layout.planes[*plane as usize];
                        assert!(*field < schema.fields);
                        assert_eq!(kernel.value_type(*out), ValueType::Scalar(schema.dtype()));
                    }
                    Op::Read { .. } | Op::VectorRead { .. } => {
                        panic!("{} retained an implicit decoder", info.name)
                    }
                    Op::Binary { out, .. } | Op::Bit { out, .. } => {
                        assert_eq!(kernel.value_type(*out), ValueType::Scalar(DType::U32));
                        word_work += 1;
                    }
                    Op::Fma { .. } | Op::Math { .. } | Op::Cast { .. } => {
                        panic!("{} retained unexpanded source arithmetic", info.name)
                    }
                    _ => {}
                }
            }
            assert!(fields >= 2, "{} has no typed field reads", info.name);
            assert!(word_work > 0, "{} has no visible decode work", info.name);
        }
    }
}
