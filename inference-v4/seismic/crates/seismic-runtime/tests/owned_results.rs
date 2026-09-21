use seismic_lang::{
    logical::specialization::{ShapeBinding, SpecializationDomain},
    program::{compile, SourceFile},
    types::Elem,
};
use seismic_runtime::{
    invocation::Bindings,
    plan::{PlanCompiler, Settings},
    submission::Submission,
    Buffer, Device,
};
use std::collections::BTreeMap;

#[test]
fn cpu_allocates_and_returns_hidden_tuple_destinations() {
    let program = compile(&[SourceFile {
        path: "owned-results.seismic".into(),
        text: "fn pair[N](x: tensor[N] f32, y: tensor[N] f32) -> (tensor[N] f32, tensor[N] f32):\n    return x, y\n".into(),
    }])
    .unwrap_or_else(|diagnostics| {
        panic!(
            "{}",
            diagnostics
                .iter()
                .map(|diagnostic| diagnostic.render())
                .collect::<Vec<_>>()
                .join("\n")
        )
    });
    let device = Device::cpu().unwrap();
    let shapes = BTreeMap::from([("N".to_string(), 4_i64)]);
    let elements = BTreeMap::<String, Elem>::new();
    let domain = SpecializationDomain::new(
        &program,
        "pair",
        shapes
            .iter()
            .map(|(n, v)| (n.clone(), ShapeBinding::Exact(*v as u64)))
            .collect(),
        elements,
    )
    .unwrap();
    let mut compiler = PlanCompiler::new(&device, &program, Settings::default());
    let plan = compiler.compile_entry(&domain).unwrap();
    assert_eq!(
        compiler.kernel_count(),
        1,
        "compile_entry must eagerly native-compile"
    );
    let values = [[1.0_f32, 2.0, 3.0, 4.0], [5.0_f32, 6.0, 7.0, 8.0]];
    let inputs = values
        .iter()
        .map(|values| {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            device.buffer_from(&bytes).unwrap()
        })
        .collect::<Vec<_>>();

    struct Inputs<'a> {
        buffers: &'a [Buffer],
        shapes: &'a BTreeMap<String, i64>,
    }
    impl Bindings for Inputs<'_> {
        fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
            match (root, plane) {
                ("x", "") => self.buffers.first(),
                ("y", "") => self.buffers.get(1),
                _ => None,
            }
        }
        fn scalar(&self, _name: &str) -> Option<f64> {
            None
        }
        fn shape(&self, name: &str) -> Option<u64> {
            self.shapes.get(name).and_then(|v| u64::try_from(*v).ok())
        }
    }

    let bound = Inputs {
        buffers: &inputs,
        shapes: &shapes,
    };
    let invocation = plan.prepare(&bound).unwrap();
    let results = Submission::single(invocation).execute().unwrap().remove(0);

    let paths = results
        .planes
        .iter()
        .map(|plane| plane.path.clone())
        .collect::<Vec<_>>();
    assert_eq!(paths, [vec![0], vec![1]]);
    for (plane, expected) in results.planes.iter().zip(values.iter()) {
        let mut actual = vec![0_u8; expected.len() * size_of::<f32>()];
        plane.buffer.read(&mut actual).unwrap();
        let actual = actual
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(&actual, expected);
    }
}
