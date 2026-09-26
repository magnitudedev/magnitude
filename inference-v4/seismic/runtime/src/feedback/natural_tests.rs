use super::*;
use crate::api::kernel::EncodedScalar;
use seismic_compiler::feedback::FeedbackOptions;
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;

fn wide_scalar_publications(backend: registry::BackendName) {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(backend).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "wide-scalar-publication.seismic".into(),
        text: r#"fn identity(i: index[1152921504606846976]) -> index[1152921504606846976]:
    return i

fn helper(i: index[1152921504606846976]) -> index[1152921504606846976]:
    return identity(i)

fn branch(i: index[1152921504606846976], j: index[1152921504606846976], choose: bool) -> index[1152921504606846976]:
    let mut result = i
    if choose:
        result = j
    return identity(result)

fn carry(i: index[1152921504606846976], j: index[1152921504606846976], times: range[3]) -> index[1152921504606846976]:
    let mut result = i
    for iteration in times:
        result = identity(j)
    return result

fn range_identity(r: range[1152921504606846976]) -> range[1152921504606846976]:
    return r

fn range_helper(r: range[1152921504606846976]) -> range[1152921504606846976]:
    return range_identity(r)

fn source_words(x: u32, y: i32) -> (u32, i32):
    return (x, y)

fn source_arithmetic(x: u32, y: i32) -> (u32, i32):
    return (x + u32(1), y + i32(1))
"#.into(),
    }])).unwrap();
    let high = (1u64 << 54) + 3;
    let other = (1u64 << 40) + 7;
    for (name, arguments, expected) in [
        (
            "identity",
            vec![EncodedScalar::Index(high.into())],
            vec![ArgumentValue::Index(high.into())],
        ),
        (
            "helper",
            vec![EncodedScalar::Index(high.into())],
            vec![ArgumentValue::Index(high.into())],
        ),
        (
            "branch",
            vec![
                EncodedScalar::Index(high.into()),
                EncodedScalar::Index(other.into()),
                EncodedScalar::Bool(false),
            ],
            vec![ArgumentValue::Index(high.into())],
        ),
        (
            "branch",
            vec![
                EncodedScalar::Index(high.into()),
                EncodedScalar::Index(other.into()),
                EncodedScalar::Bool(true),
            ],
            vec![ArgumentValue::Index(other.into())],
        ),
        (
            "carry",
            vec![
                EncodedScalar::Index(high.into()),
                EncodedScalar::Index(other.into()),
                EncodedScalar::Range {
                    start: 0_u64.into(),
                    end: 0_u64.into(),
                },
            ],
            vec![ArgumentValue::Index(high.into())],
        ),
        (
            "carry",
            vec![
                EncodedScalar::Index(high.into()),
                EncodedScalar::Index(other.into()),
                EncodedScalar::Range {
                    start: 1_u64.into(),
                    end: 3_u64.into(),
                },
            ],
            vec![ArgumentValue::Index(other.into())],
        ),
        (
            "range_identity",
            vec![EncodedScalar::Range {
                start: other.into(),
                end: high.into(),
            }],
            vec![ArgumentValue::Range {
                start: other.into(),
                end: high.into(),
            }],
        ),
        (
            "range_helper",
            vec![EncodedScalar::Range {
                start: high.into(),
                end: high.into(),
            }],
            vec![ArgumentValue::Range {
                start: high.into(),
                end: high.into(),
            }],
        ),
        (
            "source_words",
            vec![EncodedScalar::U32(u32::MAX), EncodedScalar::I32(i32::MIN)],
            vec![ArgumentValue::U32(u32::MAX), ArgumentValue::I32(i32::MIN)],
        ),
        (
            "source_arithmetic",
            vec![EncodedScalar::U32(u32::MAX), EncodedScalar::I32(-3)],
            vec![ArgumentValue::U32(0), ArgumentValue::I32(-2)],
        ),
    ] {
        let kernel = Arc::new(
            crate::api::kernel::prepare(
                &module,
                module.entry_named(name).unwrap(),
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
            .unwrap_or_else(|error| panic!("{backend:?} {name}: {error:?}")),
        );
        let mut args = EncodedArgs::new();
        for argument in arguments {
            args.push_scalar(argument);
        }
        let mut results = crate::api::kernel::call(&kernel, args)
            .unwrap_or_else(|error| panic!("{backend:?} {name}: {error:?}"));
        for expected in expected {
            match (results.take_scalar(), expected) {
                (ArgumentValue::Index(actual), ArgumentValue::Index(expected)) => {
                    assert_eq!(actual, expected, "{backend:?} {name}")
                }
                (
                    ArgumentValue::Range { start, end },
                    ArgumentValue::Range {
                        start: expected_start,
                        end: expected_end,
                    },
                ) => assert_eq!(
                    (start, end),
                    (expected_start, expected_end),
                    "{backend:?} {name}"
                ),
                (ArgumentValue::U32(actual), ArgumentValue::U32(expected)) => {
                    assert_eq!(actual, expected)
                }
                (ArgumentValue::I32(actual), ArgumentValue::I32(expected)) => {
                    assert_eq!(actual, expected)
                }
                (actual, expected) => {
                    panic!("{backend:?} {name}: expected {expected:?}, got {actual:?}")
                }
            }
        }
    }
}

#[test]
fn cpu_wide_scalar_publications_preserve_naturals_and_source_dtypes() {
    wide_scalar_publications(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_wide_scalar_publications_preserve_naturals_and_source_dtypes() {
    wide_scalar_publications(registry::BackendName::Metal);
}
