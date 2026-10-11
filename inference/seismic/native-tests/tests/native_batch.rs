use seismic::{
    BackendName, DeviceCatalog, Element, NativeSpecialization, NativeTensorBatch, Tensor,
};
use seismic_native_tests::{scale_rows, scoped_scale};

#[test]
fn different_native_entries_complete_in_one_metal_batch() {
    let catalog = DeviceCatalog::discover().unwrap();
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let rows =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let implementation = scoped_scale::native_implementation(&device)
        .unwrap()
        .unwrap();
    let defaults = implementation
        .default_specialization(&NativeSpecialization::new())
        .unwrap();
    let scoped = scoped_scale::native_for_device(&device, &defaults).unwrap();
    let first_values = [1.0f32, -2.0, 3.0, 4.0];
    let second_values = [2.0f32, 0.5, -3.0, 8.0, 1.0, -1.0, 0.0];
    let bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let first = Tensor::from_host(&device, Element::f32(), &[2, 2], &bytes(&first_values)).unwrap();
    let second = Tensor::from_host(&device, Element::f32(), &[7], &bytes(&second_values)).unwrap();
    // SAFETY: both authored entries write every f32 result element.
    let mut first_result =
        unsafe { Tensor::uninitialized(&device, Element::f32(), &[2, 2]) }.unwrap();
    let mut second_result =
        unsafe { Tensor::uninitialized(&device, Element::f32(), &[7]) }.unwrap();
    let mut batch = NativeTensorBatch::new(&device);
    batch
        .push(
            &rows,
            scale_rows::Args {
                x: &first,
                factor: 3.0,
            },
            scale_rows::OutputArgs {
                value: &mut first_result,
            },
        )
        .unwrap();
    batch
        .push(
            &scoped,
            scoped_scale::Args { x: &second },
            scoped_scale::OutputArgs {
                value: &mut second_result,
            },
        )
        .unwrap();
    batch.submit().unwrap().wait().unwrap();
    assert_eq!(
        first_result.read_to_host().unwrap(),
        bytes(&first_values.map(|value| value * 3.0))
    );
    assert_eq!(
        second_result.read_to_host().unwrap(),
        bytes(&second_values.map(|value| value * 2.0))
    );
}

#[test]
fn canonical_formation_alone_serves_a_legal_choice_without_building_observations() {
    use seismic::{
        NumericalRequirement, PointInputs, PointSource, PointSpec,
    };

    struct UnbuiltPoints {
        weight: f64,
    }
    impl PointSource<'static, scoped_scale::Entry> for UnbuiltPoints {
        fn points(&self) -> Vec<PointSpec> {
            vec![PointSpec {
                label: "required witness".into(),
                weight: self.weight,
                class: None,
                cost: 1.,
                census: true,
            }]
        }
        fn build(&mut self, _: usize) -> Result<PointInputs<'static>, String> {
            panic!("the mandatory work and its record must not build observations")
        }
    }

    let catalog = DeviceCatalog::discover().unwrap();
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    // Both a demanded case and a zero-demand required witness have no evidence
    // merely because their canonical program can be formed.
    for weight in [1., 0.] {
        let mut points = UnbuiltPoints { weight };
        let unprepared = scoped_scale::native_search(
            &device,
            &NativeSpecialization::new(),
            NumericalRequirement::BitExact,
        )
        .unwrap();
        let formed = device.programs_formed();
        assert!(unprepared.terminal().is_err());
        assert_eq!(
            device.programs_formed(),
            formed,
            "a record cannot compile a missing default"
        );
        let mut search = scoped_scale::native_search(
            &device,
            &NativeSpecialization::new(),
            NumericalRequirement::BitExact,
        )
        .unwrap();
        search.prepare_defaults(&mut points).unwrap();
        let formed = device.programs_formed();
        // From its mandatory work on the unit holds a legal served choice:
        // the canonical default, with its unmeasured scope explicit.
        let result = search.terminal().unwrap();
        assert!(result.time.forming_seconds > 0.0);
        assert!(result.observations.is_empty());
        assert!(
            result
                .configurations
                .iter()
                .all(|record| record.observations.is_empty())
        );
        assert_eq!(result.served, [result.overall.clone()]);
        assert!(matches!(
            result.disposition,
            seismic::TuningDisposition::CanonicalFallback { .. }
        ));
        assert_eq!(result, search.terminal().unwrap());
        assert_eq!(
            device.programs_formed(),
            formed,
            "the record reuses the prepared default"
        );
    }
}
