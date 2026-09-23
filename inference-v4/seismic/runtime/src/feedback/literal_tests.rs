use super::*;
use seismic_compiler::feedback::FeedbackOptions;
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;
use seismic_lang::reference_math::ReferenceScalar;

fn literal_publications(backend: registry::BackendName) {
    let catalog = crate::api::catalog::Catalog::discover().unwrap();
    let info = catalog
        .devices()
        .iter()
        .find(|device| device.backend == backend)
        .unwrap();
    let device = catalog.open(info.id).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "literal-publication.seismic".into(),
        text: "fn probe() -> (f16, bf16, f32, f32, f16, bf16, f32, f16):\n    return (1.0004882812500002, 1.0039062500000002, 9007199791611905, 18446744073709551615, -0.0, -0.0, -0.0, -inf)\n".into(),
    }])).unwrap();
    let kernel = Arc::new(
        crate::api::kernel::prepare(
            &module,
            module.entry_named("probe").unwrap(),
            ElementBindings::default(),
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    search_time: Duration::ZERO,
                    ..Default::default()
                },
            ),
        )
        .unwrap(),
    );
    let mut results = crate::api::kernel::call(&kernel, EncodedArgs::new()).unwrap();
    for expected in [
        ReferenceScalar::F16(0x3c01),
        ReferenceScalar::BF16(0x3f81),
        ReferenceScalar::F32(0x5a00_0001),
        ReferenceScalar::F32(0x5f80_0000),
        ReferenceScalar::F16(0x8000),
        ReferenceScalar::BF16(0x8000),
        ReferenceScalar::F32(0x8000_0000),
        ReferenceScalar::F16(0xfc00),
    ] {
        let actual = match results.take_scalar() {
            ArgumentValue::F16(bits) => ReferenceScalar::F16(bits),
            ArgumentValue::BF16(bits) => ReferenceScalar::BF16(bits),
            ArgumentValue::F32(value) => ReferenceScalar::F32(value.to_bits()),
            other => panic!("unexpected constant result: {other:?}"),
        };
        assert_eq!(actual, expected, "{backend:?} literal publication");
    }
}

#[test]
fn cpu_literal_publications_preserve_checked_bits() {
    literal_publications(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_literal_publications_preserve_checked_bits() {
    literal_publications(registry::BackendName::Metal);
}
