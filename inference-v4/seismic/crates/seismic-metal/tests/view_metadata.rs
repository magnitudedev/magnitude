//! Shape queries retain view checks without requiring addressable tile storage.
use seismic_lang::{
    Scope,
    lower::lower,
    lowered_ir::LoweredIr,
    program::{SourceFile, compile},
};
use seismic_metal::{
    execution::{Config, Execution, prepare_storage_selected},
    model,
    msl::emit_execution,
};
use seismic_realization::{LoadStrategy, dispatch::TilePlacement};

const QUERIES: &[(&str, i32)] = &[
    ("extent(t,0)", 3),
    ("extent(t.T,0)", 5),
    ("extent(t[1,:],0)", 5),
    ("extent(t[:,1:4],1)", 3),
    ("extent(t.T[1:4,:],0)", 3),
    ("extent(reshape(x,(5,3)),0)", 5),
];

fn source(query: &str) -> LoweredIr {
    lower_source(&format!(
        "fn evaluate(x:tensor[15] i32,copy:tensor[3,5] i32,out:tensor[1] i32):\n  t = load(reshape(x,(3,5)))\n  for i,j in owned(t): t[i,j] = t[i,j] + 1\n  result = tile[1] i32\n  for i in owned(result): result[i] = {query}\n  store(t,copy)\n  store(result,out)\n"
    ))
}

fn lower_source(text: &str) -> LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "view_metadata.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    lower(&program, "evaluate", "metal", &Default::default()).unwrap()
}

fn prepare(function: &LoweredIr, placement: TilePlacement) -> Execution {
    let execution = prepare_storage_selected(
        function,
        Config {
            loads: LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |decision| {
            assert!(
                decision.alternatives.contains(&placement),
                "metadata required a data-access placement: {decision:?}"
            );
            Ok(placement.clone())
        },
    )
    .unwrap();
    let tile = execution
        .function()
        .vars
        .iter()
        .position(|v| v.name == "t")
        .unwrap();
    assert_eq!(
        execution.storage().declaration(tile).unwrap().placement,
        placement
    );
    execution
}

#[test]
fn composed_shape_queries_admit_distributed_storage_and_typed_emission() {
    for (query, _) in QUERIES {
        let execution = prepare(&source(query), TilePlacement::Distributed);
        let requirements = model::requirements(&execution).unwrap();
        assert!(
            requirements.unmapped.is_empty(),
            "{query}: {:?}",
            requirements.unmapped
        );
        let emitted = emit_execution(&execution).unwrap();
        assert!(!emitted.terminal.launches().is_empty(), "{query}");
    }
}

#[test]
fn shape_only_reshape_preserves_contiguous_layout_requirement() {
    for source in [
        "fn evaluate(x:tensor[2,3] i32,out:tensor[1] i32):\n  result = tile[1] i32\n  for i in owned(result): result[i] = extent(reshape(x.T,(6,)),0)\n  store(result,out)\n",
        "fn evaluate(x:tensor[2,3] i32,out:tensor[6] i32):\n  result = load(reshape(x.T,(6,)))\n  store(result,out)\n",
    ] {
        let function = lower_source(source);
        let error = seismic_metal::msl::emit_with(&function, Config::default()).unwrap_err();
        assert!(
            error.contains("reshape requires contiguous row-major storage"),
            "{error}"
        );
    }
}

const POINT_GUARD: &str = "fn evaluate(x:tensor[3,5] i32,index:tensor[1] i32,copy:tensor[3,5] i32,out:tensor[1] i32):\n  t = load(x)\n  for i,j in owned(t): t[i,j] = t[i,j] + 1\n  result = tile[1] i32\n  for i in owned(result): result[i] = extent(t[index[0],:],0)\n  store(t,copy)\n  store(result,out)\n";

#[test]
fn distributed_shape_query_retains_its_dynamic_point_guard() {
    let execution = prepare(&lower_source(POINT_GUARD), TilePlacement::Distributed);
    let requirements = model::requirements(&execution).unwrap();
    assert!(
        requirements.unmapped.is_empty(),
        "{:?}",
        requirements.unmapped
    );
    let emitted = emit_execution(&execution).unwrap();
    assert!(emitted.status_slot.is_some());
}

#[test]
fn metadata_endpoints_retain_cross_lane_data_dependencies() {
    for query in [
        "extent(t[positions[0],:],0)",
        "extent(t[:,positions[0]:positions[1]],1)",
        "extent(t.T[positions[0]:positions[1],:],0)",
    ] {
        let function = lower_source(&format!(
            "fn evaluate(x:tensor[3,5] i32,index:tensor[2] i32,copy:tensor[3,5] i32,out:tensor[1] i32):\n  t = load(x)\n  positions = load(index)\n  for i,j in owned(t): t[i,j] = t[i,j] + 1\n  result = tile[1] i32\n  for i in owned(result): result[i] = {query}\n  store(t,copy)\n  store(result,out)\n"
        ));
        let mut checked_endpoints = false;
        let execution = prepare_storage_selected(
            &function,
            Config {
                loads: LoadStrategy::Materialize,
                ..Default::default()
            },
            &mut |decision| {
                if decision.name == "positions" {
                    checked_endpoints = true;
                    assert!(decision.cross_lane_read, "{query}");
                    assert!(!decision.alternatives.contains(&TilePlacement::Distributed));
                    assert!(decision.select(TilePlacement::Distributed).is_err());
                    Ok(TilePlacement::Replicated)
                } else {
                    assert!(
                        decision.alternatives.contains(&TilePlacement::Distributed),
                        "{decision:?}"
                    );
                    Ok(TilePlacement::Distributed)
                }
            },
        )
        .unwrap();
        assert!(checked_endpoints, "{query}");
        assert!(model::requirements(&execution).unwrap().unmapped.is_empty());
    }
}

const OWNER_COORDINATES: &[(&str, bool)] = &[
    ("i + 0", false),
    ("i * 1", false),
    ("i32(i)", false),
    ("i + 0 * extent(guard[index[0],:],0)", true),
];

fn owner_coordinate_source(coordinate: &str) -> LoweredIr {
    lower_source(&format!(
        "fn evaluate(x:tensor[65] i32,guard:tensor[2,4] i32,index:tensor[1] i32,out:tensor[65] i32):\n  t = load(x)\n  for i in owned(t): t[i] = t[{coordinate}] + 1\n  store(t,out)\n"
    ))
}

#[test]
fn normalized_owner_coordinates_preserve_distributed_admission_and_emission() {
    for (coordinate, _) in OWNER_COORDINATES {
        let execution = prepare(
            &owner_coordinate_source(coordinate),
            TilePlacement::Distributed,
        );
        let requirements = model::requirements(&execution).unwrap();
        assert!(
            requirements.unmapped.is_empty(),
            "{coordinate}: {:?}",
            requirements.unmapped
        );
        emit_execution(&execution).unwrap();
    }
}

#[cfg(target_os = "macos")]
fn encode(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|n| n.to_le_bytes()).collect()
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn native_composed_shape_queries_preserve_each_selected_storage_form() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let values = (0..15).collect::<Vec<i32>>();
    let input = device.buffer_from(&encode(&values)).unwrap();
    for (query, expected) in QUERIES {
        for placement in [
            TilePlacement::Replicated,
            TilePlacement::Distributed,
            TilePlacement::GroupShared,
        ] {
            let execution = prepare(&source(query), placement.clone());
            let pipeline = device.compile(emit_execution(&execution).unwrap()).unwrap();
            let copy = device.buffer(60).unwrap();
            let out = device.buffer(4).unwrap();
            device
                .run(&pipeline, &[&input, &copy, &out], &[], 1)
                .unwrap();
            assert_eq!(
                i32::from_le_bytes(out.read(4).try_into().unwrap()),
                *expected,
                "{query}, {placement:?}"
            );
            let copied = copy.read(60);
            for (value, bytes) in values.iter().zip(copied.chunks_exact(4)) {
                assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), value + 1);
            }
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn native_distributed_shape_queries_preserve_point_failures_and_recovery() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let execution = prepare(&lower_source(POINT_GUARD), TilePlacement::Distributed);
    let pipeline = device.compile(emit_execution(&execution).unwrap()).unwrap();
    let input = device.buffer_from(&encode(&[1; 15])).unwrap();
    let index = device.buffer(4).unwrap();
    let copy = device.buffer(60).unwrap();
    let out = device.buffer(4).unwrap();
    for coordinate in [0, -1, 2, 3, 0] {
        index.write(&encode(&[coordinate]));
        let result = device.run(&pipeline, &[&input, &index, &copy, &out], &[], 1);
        assert_eq!(
            result.is_ok(),
            (0..3).contains(&coordinate),
            "{coordinate}: {result:?}"
        );
        if result.is_ok() {
            assert_eq!(i32::from_le_bytes(out.read(4).try_into().unwrap()), 5);
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal hardware"]
fn native_normalized_owner_coordinates_preserve_values_and_extent_failures() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let values = (0..65).map(|i| i * 3 - 60).collect::<Vec<i32>>();
    let input = device.buffer_from(&encode(&values)).unwrap();
    let guard = device.buffer_from(&encode(&[1; 8])).unwrap();
    let index = device.buffer(4).unwrap();
    let out = device.buffer(65 * 4).unwrap();
    for (coordinate, checks_extent) in OWNER_COORDINATES {
        let execution = prepare(
            &owner_coordinate_source(coordinate),
            TilePlacement::Distributed,
        );
        let pipeline = device.compile(emit_execution(&execution).unwrap()).unwrap();
        for point in [0, -1, 1, 2, 0] {
            index.write(&encode(&[point]));
            let result = device.run(&pipeline, &[&input, &guard, &index, &out], &[], 1);
            assert_eq!(
                result.is_ok(),
                !checks_extent || (0..2).contains(&point),
                "{coordinate}, point={point}: {result:?}"
            );
            if result.is_ok() {
                for (bytes, value) in out.read(65 * 4).chunks_exact(4).zip(&values) {
                    assert_eq!(
                        i32::from_le_bytes(bytes.try_into().unwrap()),
                        value + 1,
                        "{coordinate}, point={point}"
                    );
                }
            }
        }
    }
}
