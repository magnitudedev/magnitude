//! Instantiation of the registry's existing decode sequence before kernel closure.
//! Field extraction is a physical read; all numerical steps use scalar recipes.
use super::internals::{PlaceEntry, PortableBuilder, PortableValue};
use super::ops::{BinaryOp, UnaryOp, ValueType};
use crate::physical_target::PhysicalDialect;
use seismic_lang::reference_math;
use seismic_lang::registry::{self, DecodeStep};
use seismic_lang::types::DType;

pub(super) fn read<B: PhysicalDialect>(
    builder: &mut PortableBuilder<'_, B>,
    place: PlaceEntry,
    indices: &[PortableValue],
) -> PortableValue {
    let info = registry::representation_info(place.representation);
    let recipe = registry::decode_recipe(place.representation, info.decoded)
        .expect("registered packed representation has a decode sequence");
    let mut values = Vec::with_capacity(recipe.temporary_count());
    for step in recipe.steps() {
        let get = |temp| values[recipe.ordinal(temp)];
        let value = match step {
            DecodeStep::ReadPlaneField { plane, field, .. } => {
                builder.read_plane_field(place, indices, *plane, *field)
            }
            DecodeStep::InterpretCode {
                raw,
                bits,
                interpretation,
                ..
            } => {
                let scalar = reference_math::code_recipe(interpretation, *bits);
                super::reference_math::expand_total_recipe(builder, &scalar, &[get(*raw)])
            }
            DecodeStep::DecodeFloatCode { raw, format, .. } => {
                let scalar = reference_math::float_code_recipe(*format);
                super::reference_math::expand_total_recipe(builder, &scalar, &[get(*raw)])
            }
            DecodeStep::ConvertToF32 { from, .. } => {
                builder.cast(get(*from), ValueType::Scalar(DType::F32))
            }
            DecodeStep::Multiply { left, right, .. } => {
                builder.binary(BinaryOp::Mul, get(*left), get(*right))
            }
            DecodeStep::Negate { from, .. } => builder.unary(UnaryOp::Neg, get(*from)),
            DecodeStep::MultiplyAdd {
                factor,
                multiplicand,
                addend,
                ..
            } => builder.fma(get(*factor), get(*multiplicand), get(*addend)),
            DecodeStep::Cast { from, to, .. } => builder.cast(get(*from), ValueType::Scalar(*to)),
        };
        assert_eq!(
            recipe.ordinal(step.defines()),
            values.len(),
            "decode sequence has ordered definitions"
        );
        assert_eq!(value.ty, ValueType::Scalar(recipe.dtype(step.defines())));
        values.push(value);
    }
    values[recipe.ordinal(recipe.output())]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::Construction;
    use crate::kernel::ops::Op;
    use crate::storage::GlobalBufferKind;
    use crate::physical_target::{IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, VectorSupport};
    use seismic_lang::expr::ExprArena;

    #[derive(Debug)]
    struct Dialect;
    #[derive(Clone, Debug)]
    enum NoIntrinsic {}
    impl PhysicalDialect for Dialect {
        type LaunchDescriptor = ();
        fn ordinary_launch() {}
        const NAME: registry::BackendName = registry::BackendName::Cpu;
        type Facts = ();
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
                        let dtype = match schema.encoding {
                            registry::PlaneEncoding::Dense(dtype) => dtype,
                            _ => DType::U32,
                        };
                        assert_eq!(kernel.value_type(*out), ValueType::Scalar(dtype));
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
