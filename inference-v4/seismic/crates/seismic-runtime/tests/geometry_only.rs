//! Logical view geometry does not require storage for unused element data.
use seismic_lang::{
    lower::lower,
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
    Scope,
};
use seismic_realization::CallConv;
use seismic_runtime::Device;
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;

const NO_STORAGE: &str = "
fn evaluate():
  geometry = tile[1024,4] i32
  for i,j in owned(geometry): geometry[i,j] = 0
  length = extent(geometry,0)
";

const GEOMETRY: &str = "
fn evaluate(out:tensor[1] i32):
  geometry = tile[1024,4] i32
  for i,j in owned(geometry): geometry[i,j] = 0
  transposed = geometry.T
  window = transposed[1:3,0:4]
  result = tile[1] i32
  for i in owned(result): result[i] = extent(geometry,0) + extent(window,0)
  store(result,out)
";

const SNAPSHOT: &str = "
fn evaluate(x:tensor[8] i32,bounds:tensor[2] i32,out:tensor[1] i32):
  controls = load(bounds)
  snapshot = load(x[controls[0]:controls[1]])
  for i in owned(controls): controls[i] = 0
  result = tile[1] i32
  for i in owned(result): result[i] = extent(snapshot,0)
  store(result,out)
";

fn program(source: &str) -> seismic_lang::program::Program {
    compile(
        &[SourceFile {
            path: "geometry_only.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap()
}
fn function(source: &str, backend: &str) -> LoweredIr {
    lower(&program(source), "evaluate", backend, &Default::default()).unwrap()
}

fn bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn execute(device: &Device) {
    for (source, inputs, expected) in [
        (GEOMETRY, vec![], 1026),
        (SNAPSHOT, vec![vec![1; 8], vec![2, 7]], 5),
    ] {
        let program = program(source);
        let input = seismic_runtime::tuner::Input::Portable {
            program: &program,
            entry: "evaluate",
            shapes: &Default::default(),
            elements: &Default::default(),
            options: &Default::default(),
        };
        let mut buffers = inputs
            .iter()
            .map(|input| device.buffer_from(&bytes(input)).unwrap())
            .collect::<Vec<_>>();
        let output = device.buffer(4).unwrap();
        buffers.push(output.clone());
        let controls: &[usize] = if source == SNAPSHOT { &[1] } else { &[] };
        let mut kernel =
            automatic_hardware::compile_with_controls(device, input, &buffers, &[], controls)
                .unwrap();
        kernel.execute(&buffers, &[]).unwrap();
        let mut actual = [0; 4];
        output.read(&mut actual).unwrap();
        assert_eq!(i32::from_le_bytes(actual), expected);
    }
}

#[test]
fn cpu_geometry_only_tiles_omit_element_storage_and_preserve_snapshots() {
    for (source, scratch) in [(NO_STORAGE, 0), (GEOMETRY, 8), (SNAPSHOT, 16)] {
        let function = function(source, "cpu");
        let scalar = seismic_compiler::scalar(&function, CallConv::SystemV).unwrap();
        assert_eq!(scalar.scratch_bytes, scratch);
    }
    execute(&Device::cpu());
}

const INVALID_RESHAPE: &str = "
fn evaluate(geometry:tensor[2,4] i32,out:tensor[1] i32):
  result = tile[1] i32
  for i in owned(result): result[i] = extent(reshape(geometry[:,1:3],(4,)),0)
  store(result,out)
";

// Element production is dead, but its layout validation remains observable.
const DEAD_PRODUCER_INVALID_RESHAPE: &str = "
fn evaluate(x:tensor[2,4] i32,out:tensor[1] i32):
  geometry = tile[2] i32
  for i in owned(geometry): geometry[i] = extent(reshape(x[:,1:3],(4,)),0)
  result = tile[1] i32
  for i in owned(result): result[i] = extent(geometry,0)
  store(result,out)
";

#[test]
fn cpu_geometry_only_views_retain_reshape_layout_checks() {
    for source in [INVALID_RESHAPE, DEAD_PRODUCER_INVALID_RESHAPE] {
        let function = function(source, "cpu");
        let Err(error) = seismic_compiler::scalar(&function, CallConv::SystemV) else {
            panic!("geometry query erased reshape layout validity: {source}");
        };
        assert!(error.contains("reshape"), "{error}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_geometry_only_tiles_have_no_allocation_or_placement() {
    for source in [NO_STORAGE, GEOMETRY, SNAPSHOT] {
        let function = function(source, "metal");
        let execution = seismic_metal::execution::prepare_storage_selected(
            &function,
            Default::default(),
            &mut |decision| {
                assert!(matches!(
                    function.vars[decision.variable].name.as_str(),
                    "result" | "controls"
                ));
                Ok(decision.alternatives[0].clone())
            },
        )
        .unwrap();
        let emitted = seismic_metal::msl::emit_execution(&execution).unwrap();
        assert!(emitted
            .launches
            .iter()
            .flat_map(|launch| &launch.tiles)
            .all(|tile| tile.capacity <= 2));
    }
    for source in [INVALID_RESHAPE, DEAD_PRODUCER_INVALID_RESHAPE] {
        let function = function(source, "metal");
        let execution = seismic_metal::execution::prepare_storage_selected(
            &function,
            Default::default(),
            &mut |decision| Ok(decision.alternatives[0].clone()),
        );
        let error = match execution {
            Ok(execution) => seismic_metal::msl::emit_execution(&execution).unwrap_err(),
            Err(error) => error,
        };
        assert!(error.contains("reshape"), "{error}");
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_geometry_only_snapshots_execute() {
    execute(&Device::metal().unwrap());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_geometry_only_snapshots_execute() {
    execute(&Device::cuda(0).unwrap());
}
