use seismic_lang::{
    program::{compile, SourceFile},
    types::Elem,
};
use seismic_runtime::{
    plan::{PlanCompiler, Settings},
    Device,
};
use std::collections::HashMap;

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
    let shapes = HashMap::from([("N".into(), 4)]);
    let elements = HashMap::<String, Elem>::new();
    let mut compiler = PlanCompiler::new(&device, &program, Settings::default());
    let mut plan = compiler.compile_entry("pair", &shapes, &elements).unwrap();
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

    let results = plan.execute_buffers_with_results(&inputs, &[]).unwrap();

    assert_eq!(
        results
            .iter()
            .map(|result| result.path.clone())
            .collect::<Vec<_>>(),
        [vec![0], vec![1]]
    );
    for (result, expected) in results.iter().zip(values) {
        let mut actual = vec![0_u8; expected.len() * size_of::<f32>()];
        result.buffer.read(&mut actual).unwrap();
        let actual = actual
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
