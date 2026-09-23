//! End-to-end composition through ordinary compiler construction and execution.
use super::*;
use seismic_compiler::feedback::FeedbackOptions;
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;

#[test]
fn cpu_helper_products_preserve_captured_contents_and_unit_effects() {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "binding-composition.seismic".into(),
        text: r#"fn twice(x: &tensor[4] f32) -> tensor[4] f32:
    return x + x

fn total(x: &tensor[4] f32) -> f32:
    return reduce(x, 0, sum)

fn clear(x: &mut tensor[4] f32):
    for i in 0..4:
        x[i] = 0.0

fn chained(x: &tensor[4] f32) -> f32:
    return total(twice(x))

fn captured(x: &mut tensor[4] f32) -> f32:
    let snapshot = twice(x)
    clear(x)
    return total(snapshot)

fn selected_product(x: &tensor[4] f32) -> f32:
    let mut y = twice(x)
    if x[0] > 0.0:
        y = x + x + x
    return total(y)

fn unselected_product(x: &tensor[4] f32) -> f32:
    let mut y = twice(x)
    if x[0] < 0.0:
        y = x + x + x
    return total(y)

fn first(x: &tensor[4] f32) -> f32:
    return x[0]

fn seed_first(x: &mut tensor[4] f32):
    x[0] = 7.0

fn partial_unit(x: &tensor[4] f32) -> f32:
    let mut y = tensor[4] f32
    seed_first(y)
    return first(y)

fn loop_prefix(x: &tensor[4] f32) -> f32:
    let mut y = tensor[4] f32
    let mut result = 0.0
    for i in 0..4:
        if i > 0:
            result = result + first(y)
        seed_first(y)
    return result

fn selected_stored(x: &tensor[4] f32) -> f32:
    let mut y = zeros_like(x)
    if x[0] > 0.0:
        y = to_owned(x)
    return total(y)

fn unit_effect(x: &mut tensor[4] f32) -> f32:
    clear(x)
    return total(x)
"#
        .into(),
    }]))
    .unwrap();
    for (name, expected) in [
        ("chained", 20.0f32),
        ("captured", 20.0),
        ("unit_effect", 0.0),
        ("selected_product", 30.0),
        ("unselected_product", 20.0),
        ("partial_unit", 7.0),
        ("loop_prefix", 21.0),
        ("selected_stored", 10.0),
    ] {
        eprintln!("binding composition: {name}");
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
            .unwrap_or_else(|error| panic!("{name}: {error:?}")),
        );
        let bytes = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let input = Arc::new(
            TensorInner::from_host(&device, registry::dense(DType::F32), &[4], &bytes).unwrap(),
        );
        let mut args = EncodedArgs::new();
        args.push_tensor(input);
        let mut output = crate::api::kernel::call(&kernel, args)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let ArgumentValue::F32(actual) = output.take_scalar() else {
            panic!("{name}: expected scalar")
        };
        assert_eq!(actual.to_bits(), expected.to_bits(), "{name}");
    }
}

#[test]
fn cpu_repeat_transports_actual_tensor_views_and_fresh_instances() {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "repeat-products.seismic".into(),
        text: r#"fn zero(x: &tensor[2,2] f32) -> f32:
    let mut y = to_owned(x)
    for i in 0..0:
        y = y.T
    return y[0,1]

fn strided(x: &tensor[2,2] f32) -> f32:
    let mut y = to_owned(x)
    for i in 0..3:
        y = y.T
    return y[0,1]

fn fresh(x: &tensor[2,2] f32) -> f32:
    let mut y = to_owned(x)
    for i in 0..3:
        y = y + y
    return y[0,1]

fn fresh_partial(x: &tensor[2,2] f32) -> f32:
    let mut y = to_owned(x)
    for i in 0..3:
        let mut next = tensor[2,2] f32
        next[0,1] = y[0,1] + 1.0
        y = next
    return y[0,1]
"#
        .into(),
    }]))
    .unwrap();
    for (name, expected) in [("zero", 2.0f32), ("strided", 3.0), ("fresh", 16.0), ("fresh_partial",5.0)] {
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
            .unwrap_or_else(|error| panic!("{name}: {error:?}")),
        );
        let bytes = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let input = Arc::new(
            TensorInner::from_host(&device, registry::dense(DType::F32), &[2, 2], &bytes).unwrap(),
        );
        let mut args = EncodedArgs::new();
        args.push_tensor(input);
        let mut output = crate::api::kernel::call(&kernel, args)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let ArgumentValue::F32(actual) = output.take_scalar() else {
            panic!("expected scalar")
        };
        assert_eq!(actual.to_bits(), expected.to_bits(), "{name}");
    }
}

#[test]
fn exclusive_carried_borrow_rejects_simultaneous_access_to_its_root() {
    let source = r#"fn first_one(x: &tensor[2] f32) -> f32:
    return x[1]

fn main() -> f32:
    let mut x = tensor[2] f32
    x[0] = 1.0
    let mut y = x[0:2]
    let mut observed = 0.0
    for i in 0..2:
        y[1] = 7.0
        observed = first_one(x)
        if i == 0:
            let mut fresh = tensor[2] f32
            fresh[0] = 0.0
            fresh[1] = 0.0
            y = fresh
    return observed + first_one(y)
"#;
    let error = match check_source(SourceSet::new(vec![SourceFile {
        path: "carried-place-alias.seismic".into(),
        text: source.into(),
    }])) {
        Ok(_) => panic!("exclusive borrowed storage must not permit access through its root"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("exclusive tensor borrow"), "{error}");
}

#[test]
fn cpu_repeat_uses_nonzero_offset_and_rectangular_transpose_composition() {
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "repeat-offset-view.seismic".into(),
        text: r#"fn main(x: &tensor[3,4] f32) -> f32:
    let y = x[1:3,0:3]
    let mut total = 0.0
    for i in 0..3:
        let view = y.T.T
        total += view[1,i]
    return total
"#
        .into(),
    }]))
    .unwrap();
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    let kernel = Arc::new(
        crate::api::kernel::prepare(
            &module,
            module.entry_named("main").unwrap(),
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
    let bytes = (1..=12)
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let input = Arc::new(
        TensorInner::from_host(&device, registry::dense(DType::F32), &[3, 4], &bytes).unwrap(),
    );
    let mut args = EncodedArgs::new();
    args.push_tensor(input);
    let mut output = crate::api::kernel::call(&kernel, args).unwrap();
    let ArgumentValue::F32(actual) = output.take_scalar() else {
        panic!("expected scalar")
    };
    assert_eq!(actual, 30.0);
}

#[test]
fn cpu_overlapping_store_preserves_rhs_and_root_descriptor() {
    let module=check_source(SourceSet::new(vec![SourceFile{
        path:"overlapping-store.seismic".into(),
        text:"fn main(x: &mut tensor[3] f32) -> f32:\n    x[1:3] = x[0:2]\n    return x[2]\n".into(),
    }])).unwrap();
    let catalog=crate::devices::Catalog::discover().unwrap();
    let device=catalog.open_backend(registry::BackendName::Cpu).unwrap();
    let kernel=Arc::new(crate::api::kernel::prepare(
        &module,module.entry_named("main").unwrap(),ElementBindings::default(),&device,
        PreparationOptions::feedback(PrecisionPolicy::Exact,FeedbackOptions{search_time:Duration::ZERO,..Default::default()}),
    ).unwrap());
    let bytes=[1.0f32,2.,3.].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
    let input=Arc::new(TensorInner::from_host(&device,registry::dense(DType::F32),&[3],&bytes).unwrap());
    let mut args=EncodedArgs::new();args.push_tensor(input);
    let mut output=crate::api::kernel::call(&kernel,args).unwrap();
    let ArgumentValue::F32(actual)=output.take_scalar() else{panic!("expected scalar")};
    assert_eq!(actual,2.);
}

#[test]
fn cpu_stored_logical_views_preserve_reshape_aliases_calls_and_publication() {
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "stored-logical-views.seismic".into(),
        text: r#"fn edit(x: &mut tensor[6] f32):
    x[1] = 9.0

fn copy_view(x: &tensor[6] f32) -> tensor[6] f32:
    return to_owned(x)

fn transposed(x: &tensor[2,3] f32) -> tensor[3,2] f32:
    let y = to_owned(x)
    return y.T

fn owned(x: &tensor[2,3] f32) -> tensor[6] f32:
    let y = to_owned(x)
    return reshape(y.T, (6,))

fn write(x: &mut tensor[2,3] f32) -> tensor[6] f32:
    let mut y = reshape(x.T, (6,))
    edit(y)
    return to_owned(y)

fn empty(x: &tensor[2,3] f32) -> tensor[0] f32:
    let y = tensor[0,2] f32
    return reshape(y.T, (0,))

fn failure_prefix(x: &mut tensor[2,3] f32, index: i32) -> tensor[6] f32:
    let y = to_owned(x)
    x[0,0] = 7.0
    let checked = x[index,0]
    return reshape(y.T, (6,))

fn captured(x: &tensor[3,4] f32) -> tensor[6] f32:
    let y = x[1:3,0:3]
    return copy_view(reshape(y.T, (6,)))
"#.into(),
    }])).unwrap();
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    for (name, shape, expected, expected_input) in [
        ("transposed", vec![2,3], vec![1.,4.,2.,5.,3.,6.], vec![1.,2.,3.,4.,5.,6.]),
        ("owned", vec![2,3], vec![1.,4.,2.,5.,3.,6.], vec![1.,2.,3.,4.,5.,6.]),
        ("write", vec![2,3], vec![1.,9.,2.,5.,3.,6.], vec![1.,2.,3.,9.,5.,6.]),
        ("empty", vec![2,3], vec![], vec![1.,2.,3.,4.,5.,6.]),
        ("captured", vec![3,4], vec![5.,9.,6.,10.,7.,11.], (1..=12).map(|x| x as f32).collect()),
    ] {
        let kernel = Arc::new(crate::api::kernel::prepare(
            &module, module.entry_named(name).unwrap(), ElementBindings::default(), &device,
            PreparationOptions::feedback(PrecisionPolicy::Exact, FeedbackOptions { search_time: Duration::ZERO, ..Default::default() }),
        ).unwrap_or_else(|error| panic!("{name}: {error:?}")));
        let count = shape.iter().product::<u64>();
        let bytes = (1..=count).flat_map(|value| (value as f32).to_le_bytes()).collect::<Vec<_>>();
        let input = Arc::new(TensorInner::from_host(&device, registry::dense(DType::F32), &shape, &bytes).unwrap());
        let mut args = EncodedArgs::new();
        args.push_tensor(input.clone());
        let mut result = crate::api::kernel::call(&kernel, args).unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let expected = expected.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        let output = result.take_tensor();
        assert_eq!(output.read_to_host().unwrap(), expected, "{name} result");
        if name == "transposed" {
            let replacement = [11_f32,14.,12.,15.,13.,16.].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
            output.write_from_host(&replacement).unwrap();
            assert_eq!(output.read_to_host().unwrap(), replacement);
        }
        let expected_input = expected_input.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(input.read_to_host().unwrap(), expected_input, "{name} backing");
    }
    let kernel = Arc::new(crate::api::kernel::prepare(
        &module, module.entry_named("failure_prefix").unwrap(), ElementBindings::default(), &device,
        PreparationOptions::feedback(PrecisionPolicy::Exact, FeedbackOptions { search_time: Duration::ZERO, ..Default::default() }),
    ).unwrap());
    let bytes = (1..=6).flat_map(|value| (value as f32).to_le_bytes()).collect::<Vec<_>>();
    let input = Arc::new(TensorInner::from_host(&device, registry::dense(DType::F32), &[2,3], &bytes).unwrap());
    let mut args = EncodedArgs::new();
    args.push_tensor(input.clone());
    args.push_scalar(crate::api::kernel::EncodedScalar::I32(2));
    assert!(crate::api::kernel::call(&kernel,args).is_err(), "failed source prefix must not publish a mapped result");
    let expected = [7_f32,2.,3.,4.,5.,6.].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
    assert_eq!(input.read_to_host().unwrap(),expected);
}

#[test]
fn cpu_nonaffine_owned_carries_relocate_initialized_storage() {
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "nonaffine-owned-carries.seismic".into(),
        text: r#"fn carried(x: &tensor[2,3] f32) -> tensor[2,3] f32:
    let mut y = to_owned(x)
    for i in 0..2:
        y = reshape(y.T, (2,3))
    y[0,1] = 17.0
    return y

fn partial(x: &tensor[2,3] f32) -> tensor[1] f32:
    let mut y = tensor[2,3] f32
    y[0,0] = x[0,0]
    for i in 0..2:
        y = reshape(y.T, (2,3))
    let mut result = tensor[1] f32
    result[0] = y[0,0]
    return result

"#.into(),
    }])).unwrap();
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    for (name, expected) in [("carried",vec![1_f32,17.,4.,3.,2.,6.]),("partial",vec![1.])] {
        let kernel = Arc::new(crate::api::kernel::prepare(
            &module,module.entry_named(name).unwrap(),ElementBindings::default(),&device,
            PreparationOptions::feedback(PrecisionPolicy::Exact,FeedbackOptions { search_time: Duration::ZERO,..Default::default() }),
        ).unwrap());
        let bytes=(1..=6).flat_map(|value|(value as f32).to_le_bytes()).collect::<Vec<_>>();
        let input=Arc::new(TensorInner::from_host(&device,registry::dense(DType::F32),&[2,3],&bytes).unwrap());
        let mut args=EncodedArgs::new();
        args.push_tensor(input);
        let mut result=crate::api::kernel::call(&kernel,args).unwrap();
        assert_eq!(result.take_tensor().read_to_host().unwrap(),expected.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>(),"{name}");
    }
}

#[test]
fn cpu_return_boundary_rounds_scalar_and_tensor_elements() {
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "return-rounding.seismic".into(),
        text: "fn scalar(x: f32) -> f16:\n    return x\n\nfn tensor(x: &tensor[2] f32) -> tensor[2] f16:\n    return to_owned(x)\n\nfn stored(x: &tensor[2] f32) -> tensor[2] f16:\n    let mut y = tensor[2] f16\n    y[:] = to_owned(x)\n    return y\n".into(),
    }])).unwrap();
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    let options = || PreparationOptions::feedback(
        PrecisionPolicy::Exact,
        FeedbackOptions { search_time: Duration::ZERO, ..Default::default() },
    );

    let scalar = Arc::new(crate::api::kernel::prepare(&module, module.entry_named("scalar").unwrap(), ElementBindings::default(), &device, options()).unwrap());
    let mut args = EncodedArgs::new();
    args.push_scalar(crate::api::kernel::EncodedScalar::F32Bits(1.0006f32.to_bits()));
    let mut result = crate::api::kernel::call(&scalar, args).unwrap();
    let ArgumentValue::F16(bits) = result.take_scalar() else { panic!("expected f16 scalar") };
    assert_eq!(bits, 0x3c01);

    let bytes = [1.0006f32, 2.0].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
    for name in ["stored", "tensor"] {
        let tensor = Arc::new(crate::api::kernel::prepare(&module, module.entry_named(name).unwrap(), ElementBindings::default(), &device, options()).unwrap());
        let input = Arc::new(TensorInner::from_host(&device, registry::dense(DType::F32), &[2], &bytes).unwrap());
        let mut args = EncodedArgs::new();
        args.push_tensor(input);
        let mut result = crate::api::kernel::call(&tensor, args).unwrap();
        assert_eq!(result.take_tensor().read_to_host().unwrap(), [0x3c01u16, 0x4000].into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>(), "{name}");
    }
}
