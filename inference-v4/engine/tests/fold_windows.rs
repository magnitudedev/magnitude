//! Decode windows retain one segment's FMA chain across repeated preparation and
//! a shorter final window. These explicit assignments qualify the shared scalar
//! execution; they do not represent an automatically selected optimum.
use seismic_lang::{
    interp::{Rng, TensorData},
    lower::lower_selected,
    lowered_ir::{Alternative, DecisionKind},
    program::{compile, SourceFile},
    reduction::structured::{PreparationScope, StepOperand, StepState, Tree},
    types::Elem,
    Scope,
};
use seismic_runtime::{Candidate, Device};

fn exercise(device: Device, candidate: Candidate) {
    exercise_preparations(device, candidate, &[0, 1, 2]);
}
fn exercise_preparations(device: Device, candidate: Candidate, preparations: &[usize]) {
    let p = compile(
        &[SourceFile {
            path: "window.seismic.portable".into(),
            scope: Scope::Portable,
            text: r#"
fn merge(left:tile[3] f32,right:tile[3] f32,out:tile[3] f32):
  for i in owned(out): out[i] = left[i] + right[i]
fn step(state:tile[3] f32,a:tile[1] f32,w:tile[3] W,out:tile[3] f32):
  for i in owned(out): out[i] = fma(a[0],w[i],state[i])
fn evaluate[K](x:tensor[1,K] f32,gate:tensor[3,K] W,up:tensor[3,K] W,out:tensor[3] f32):
  a=load(x)
  g=load(gate)
  u=load(up)
  left=tile[3] f32
  right=tile[3] f32
  zero=tile[3] f32
  for i in owned(left): left[i]=1.25
  for i in owned(right): right[i]=-0.5
  for i in owned(zero): zero[i]=0.0
  reduce((a,g),1,merge,into=(left,),step=step,identity=(zero,),ordered=false)
  reduce((a,u),1,merge,into=(right,),step=step,identity=(zero,),ordered=false)
  for i in owned(left): left[i]=left[i]+right[i]
  store(left,out)
"#
            .into(),
        }],
        &[],
    )
    .unwrap();
    for name in ["q4g64", "q5k", "q6k", "iq4g32"] {
        let r = seismic_lang::repr::lookup(name).unwrap();
        let group = i64::from(r.group);
        let extent = i64::from(r.storage_group()) * 6;
        let x = device
            .buffer_from(
                &(0..extent)
                    .map(|i| ((i % 17) as f32 - 8.0) / 31.0)
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let weights = [123, 471].map(|seed| {
            TensorData::random_packed(&mut Rng(seed), r, vec![3, extent as usize])
                .device_bytes()
                .iter()
                .map(|bytes| device.buffer_from(bytes).unwrap())
                .collect::<Vec<_>>()
        });
        let mut expected = None;
        for &preparation in preparations {
            let packets = preparation != 0;
            let mut windows = 0;
            let ir = lower_selected(
                &p,
                "evaluate",
                device.backend(),
                &[("K".into(), extent)].into(),
                &[("W".into(), Elem::Repr(name.into()))].into(),
                &Default::default(),
                &mut |d| {
                    Ok(match d.kind {
                        DecisionKind::Reduction { .. } => {
                            Alternative::ReductionTree(Tree::Pairwise)
                        }
                        DecisionKind::ReductionSegments { .. } => {
                            Alternative::ReductionSegment(group * 3)
                        }
                        DecisionKind::ReductionFusion { .. } => Alternative::Fuse,
                        DecisionKind::ReductionInput { .. } if packets => {
                            if preparation != 3 && d.alternatives.contains(&Alternative::DecodedPackets) { Alternative::DecodedPackets }
                            else { Alternative::InputSnapshot(if preparation == 1 { PreparationScope::Segment } else { PreparationScope::Window }) }
                        }
                        DecisionKind::FoldPreparation { .. } => {
                            windows += 1;
                            Alternative::PreparationWindow(group * 2)
                        }
                        DecisionKind::PacketDecode { .. } => Alternative::PacketWidth(group.min(7)),
                        DecisionKind::FoldCoefficients { .. } if preparation == 2 => {
                            Alternative::CoefficientScope(PreparationScope::Segment)
                        }
                        DecisionKind::PacketDecoder { .. } if preparation == 2 => Alternative::PacketDecoder(seismic_lang::repr::PacketDecoder::Indexed),
                        DecisionKind::FoldWords { .. } if preparation == 2 => Alternative::WordScope(PreparationScope::Segment),
                        DecisionKind::FoldOperand { .. } if preparation == 2 => Alternative::StepOperand(StepOperand::View),
                        DecisionKind::FoldState { .. } if preparation == 2 => Alternative::StepState(StepState::Retained),
                        DecisionKind::FoldTraversal { .. } => Alternative::UnrollWidth(7),
                        _ => d.alternatives.get(0).unwrap(),
                    })
                },
            )
            .unwrap();
            assert_eq!(windows, usize::from(packets));
            let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
            let out = device.buffer(12).unwrap();
            let buffers = kernel
                .buffers()
                .iter()
                .map(|slot| match slot.parameter.as_str() {
                    "x" => x.clone(),
                    "gate" => weights[0][r.plane_index(&slot.plane).unwrap()].clone(),
                    "up" => weights[1][r.plane_index(&slot.plane).unwrap()].clone(),
                    "out" => out.clone(),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            kernel.execute(&buffers, &[]).unwrap();
            let mut actual = vec![0; 12];
            out.read(&mut actual).unwrap();
            if let Some(expected) = &expected {
                assert_eq!(
                    &actual, expected,
                    "{name}: preparation changed the segment's arithmetic"
                );
            } else {
                expected = Some(actual);
            }
        }
    }
}
#[test]
fn cpu_windows_preserve_product_state_across_full_and_final_windows() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
        },
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_windows_preserve_product_state_across_full_and_final_windows() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
            },
            threads_per_block: 32,
        },
    );
}

#[test]
fn cpu_decoded_snapshots_preserve_dense_and_packed_input_precision() {
    exercise_preparations(Device::cpu(), Candidate::Cpu { loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly }, &[0, 3]);
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_decoded_snapshots_preserve_dense_and_packed_input_precision() {
    exercise_preparations(Device::cuda(0).unwrap(), Candidate::Cuda {
        options: seismic_realization::ScalarOptions { dispatch: seismic_realization::Dispatch::ParallelRoot, loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly }, threads_per_block: 32,
    }, &[0, 3]);
}
