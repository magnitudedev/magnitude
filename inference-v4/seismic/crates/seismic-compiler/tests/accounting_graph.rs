use cranelift_codegen::ir::{Value, ValueDef};
use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_realization::{
    execution::MemoryObject,
    graph::{Graph, MemoryAccess},
    CallConv, Dispatch,
};
use std::collections::HashMap;

#[test]
fn memory_order_does_not_assume_distinct_parameter_names_are_disjoint() {
    let access = |object, address, offset, write| MemoryAccess {
        object,
        address: Value::from_u32(address),
        offset,
        bytes: 4,
        write,
    };
    let x = Some(MemoryObject::Buffer {
        parameter: "x".into(),
        plane: String::new(),
    });
    let y = Some(MemoryObject::Buffer {
        parameter: "y".into(),
        plane: String::new(),
    });
    let a = access(x.clone(), 0, 0, false);
    assert!(a.may_overlap(&access(y, 1, 0, true)));
    assert!(a.may_overlap(&access(None, 1, 0, true)));
    assert!(!a.may_overlap(&access(Some(MemoryObject::PrivateScratch), 1, 0, true)));
    assert!(!a.may_overlap(&access(x.clone(), 0, 4, true)));
    assert!(a.may_overlap(&access(x.clone(), 0, 3, true)));
    assert!(!a.may_overlap(&access(x, 0, -4, true)));
}

#[test]
fn compiler_graph_preserves_loop_carried_values_and_typed_memory_dependencies() {
    let source = "fn accumulate[N](x: tensor[N] f32, out: tensor[1] f32):\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  t = load(x)\n  acc[0] = reduce(t,0,sum,ordered=true)\n  store(acc,out)\n";
    let program = compile(
        &[SourceFile {
            path: "graph.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = lower(
        &program,
        "accumulate",
        "cpu",
        &HashMap::from([("N".into(), 73)]),
    )
    .unwrap();
    let emitted =
        seismic_compiler::scalar_with(&lowered, CallConv::SystemV, Dispatch::Sequential).unwrap();
    let graph = Graph::scalar(&emitted);
    assert!(graph.unavailable.is_empty(), "{:?}", graph.unavailable);
    assert_eq!(
        graph.instructions.len(),
        emitted
            .function
            .layout
            .blocks()
            .map(|b| emitted.function.layout.block_insts(b).count())
            .sum::<usize>()
    );
    assert!(graph.edges.iter().any(|edge| {
        edge.arguments.iter().any(|(source, target)| {
            matches!(
                emitted.function.dfg.value_def(*source),
                ValueDef::Result(_, _)
            ) && matches!(
                emitted.function.dfg.value_def(*target),
                ValueDef::Param(_, _)
            )
        })
    }));
    for edge in &graph.edges {
        let target = graph
            .blocks
            .iter()
            .find(|b| b.id == edge.destination)
            .unwrap();
        assert_eq!(edge.arguments.len(), target.parameters.len());
        for ((source, destination), (parameter, ty)) in
            edge.arguments.iter().zip(&target.parameters)
        {
            assert_eq!(destination, parameter);
            assert_eq!(emitted.function.dfg.value_type(*source), *ty);
        }
    }
    assert!(graph
        .instructions
        .iter()
        .filter_map(|i| i.memory.as_ref())
        .any(|a| a.object == Some(MemoryObject::PrivateScratch) && a.write));
    assert!(!graph.memory_order.is_empty());
    for &(before, after) in &graph.memory_order {
        let a = graph.instructions.iter().find(|i| i.id == before).unwrap();
        let b = graph.instructions.iter().find(|i| i.id == after).unwrap();
        assert_eq!(a.block, b.block);
        assert!(
            a.opaque_effect
                || b.opaque_effect
                || a.memory.as_ref().unwrap().write
                || b.memory.as_ref().unwrap().write
        );
    }
}
