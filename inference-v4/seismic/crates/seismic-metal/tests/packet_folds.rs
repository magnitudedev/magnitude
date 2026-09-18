#![cfg(target_os = "macos")]
use seismic_lang::{
    interp::{Rng, TensorData},
    lower::lower_selected,
    lowered_ir::{Alternative, DecisionKind},
    program::{compile, SourceFile},
    reduction::structured::{PreparationScope, StepOperand, StepState, Tree},
    types::Elem,
    Scope,
};
use seismic_metal::execution::{prepare_with_transfers, Config, FoldOwnership};

#[test]
#[ignore = "requires Metal hardware"]
fn private_segment_packets_preserve_product_folds_and_bound_decoded_storage() {
    let program = compile(
        &[SourceFile {
            path: "packets.seismic.portable".into(),
            scope: Scope::Portable,
            text: r#"
fn merge(left:tile[3] f32,right:tile[3] f32,out:tile[3] f32):
  for i in owned(out): out[i] = left[i] + right[i]
fn accumulate(state:tile[3] f32,a:tile[1] f32,w:tile[3] W,out:tile[3] f32):
  for i in owned(out): out[i] = fma(a[0],w[i],state[i])
fn evaluate(x:tensor[1,256] f32,gate:tensor[3,256] W,up:tensor[3,256] W,out:tensor[3] f32):
  a = load(x)
  g = load(gate)
  u = load(up)
  left = tile[3] f32
  right = tile[3] f32
  zero = tile[3] f32
  for i in owned(left): left[i] = 1.25
  for i in owned(right): right[i] = -0.5
  for i in owned(zero): zero[i] = 0.0
  reduce((a,g),1,merge,into=(left,),step=accumulate,identity=(zero,),ordered=false)
  reduce((a,u),1,merge,into=(right,),step=accumulate,identity=(zero,),ordered=false)
  for i in owned(left): left[i] = left[i] + right[i]
  store(left,out)
"#
            .into(),
        }],
        &[],
    )
    .unwrap();
    let device = seismic_metal::runtime::Device::open().unwrap();
    let x = device
        .buffer_from(
            &(0..256)
                .map(|i| (i as f32 - 64.0) / 97.0)
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut bundled_reads = 0;
    for (name, segment, width) in [
        ("q4g64", 16, 7),
        ("q5k", 16, 7),
        ("q6k", 32, 11),
        ("iq4g32", 8, 3),
        ("q4g64", 256, 16),
    ] {
        let representation = seismic_lang::repr::lookup(name).unwrap();
        let gate = TensorData::random_packed(&mut Rng(123), representation, vec![3, 256])
            .device_bytes()
            .iter()
            .map(|b| device.buffer_from(b).unwrap())
            .collect::<Vec<_>>();
        let up = TensorData::random_packed(&mut Rng(471), representation, vec![3, 256])
            .device_bytes()
            .iter()
            .map(|b| device.buffer_from(b).unwrap())
            .collect::<Vec<_>>();
        let mut expected = None;
        for (preparation, unroll) in [
            (0, 1),
            (1, segment.min(32)),
            (1, 7),
            (2, segment.min(32)),
            (3, 7),
            (4, 7),
        ] {
            let window = if preparation >= 3 {
                if segment == 256 {
                    192
                } else {
                    segment / 2
                }
            } else {
                segment
            };
            let unroll = unroll.min(window);
            let mut preparations = 0;
            let ir = lower_selected(
                &program,
                "evaluate",
                "metal",
                &Default::default(),
                &[("W".into(), Elem::Repr(name.into()))].into(),
                &Default::default(),
                &mut |d| {
                    Ok(match d.kind {
                        DecisionKind::Reduction { .. } => {
                            Alternative::ReductionTree(Tree::Pairwise)
                        }
                        DecisionKind::ReductionSegments { .. } => {
                            Alternative::ReductionSegment(segment)
                        }
                        DecisionKind::ReductionFusion { .. } => Alternative::Fuse,
                        DecisionKind::ReductionInput { .. } if preparation != 0 && d.alternatives.contains(&Alternative::Encoded) => {
                            preparations += 1;
                            if preparation != 2 {
                                Alternative::DecodedPackets
                            } else {
                                Alternative::SegmentSnapshot
                            }
                        }
                        DecisionKind::FoldOperand { .. } if preparation >= 3 => Alternative::StepOperand(StepOperand::View),
                        DecisionKind::FoldState { .. } if preparation >= 3 => Alternative::StepState(StepState::Retained),
                        DecisionKind::FoldTraversal { .. } => Alternative::UnrollWidth(unroll),
                        DecisionKind::PacketDecoder { .. } if preparation == 4 => Alternative::PacketDecoder(seismic_lang::repr::PacketDecoder::Indexed),
                        DecisionKind::FoldWords { .. } if preparation == 4 && d.alternatives.contains(&Alternative::WordScope(PreparationScope::Segment)) => Alternative::WordScope(PreparationScope::Segment),
                        DecisionKind::FoldCoefficients { .. } if preparation == 4 => {
                            Alternative::CoefficientScope(PreparationScope::Segment)
                        }
                        DecisionKind::FoldPreparation { .. } => {
                            Alternative::PreparationWindow(window)
                        }
                        DecisionKind::PacketDecode { .. } => {
                            Alternative::PacketWidth(width.min(window))
                        }
                        _ => d.alternatives.get(0).unwrap(),
                    })
                },
            )
            .unwrap();
            assert_eq!(preparations, if preparation != 0 { 2 } else { 0 });
            let execution = prepare_with_transfers(
                &ir,
                Config::default(),
                None,
                &mut |_| Ok(FoldOwnership::Participants),
                &mut |site, s| {
                    Ok(if s.can_borrow && !(preparation == 2 && site >= 3) {
                        seismic_lang::ir::LoadMode::Borrow
                    } else {
                        seismic_lang::ir::LoadMode::Materialize
                    })
                },
                &mut |s| Ok(s.diagnostic()),
                &mut |r| Ok(r.diagnostic()),
                &mut |a| Ok(a.alternatives[0]),
                &mut |choice| {
                    if preparation == 4 && choice.kind == seismic_metal::terminal::transfer::Kind::CopyLoop { Ok(choice.maximum) }
                    else if preparation != 0 && choice.kind == seismic_metal::terminal::transfer::Kind::ReadBundle {
                        bundled_reads += 1;
                        Ok(choice.maximum)
                    } else { Ok(1) }
                },
                &mut |choice| Ok(if preparation == 4 { choice.iterations } else { 1 }),
            )
            .unwrap();
            if preparation >= 3 {
                let caches = execution
                    .function()
                    .vars
                    .iter()
                    .filter(|v| v.name.starts_with("decoded_packets_"))
                    .map(|v| {
                        v.ty.shaped()
                            .unwrap()
                            .shape
                            .last()
                            .unwrap()
                            .as_constant()
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert!(!caches.is_empty());
                assert!(caches.iter().all(|&n| n <= window));
                assert!(caches.contains(&window));
                if segment % window != 0 {
                    assert!(caches.contains(&(segment % window)));
                }
            }
            if preparation == 2 {
                let snapshots = execution
                    .function()
                    .vars
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.name.starts_with("segment_snapshot_"))
                    .collect::<Vec<_>>();
                assert_eq!(snapshots.len(), 2);
                for (id, snapshot) in snapshots {
                    assert_eq!(
                        snapshot
                            .ty
                            .shaped()
                            .unwrap()
                            .shape
                            .last()
                            .unwrap()
                            .as_constant(),
                        Some(segment)
                    );
                    assert!(
                        execution.storage().packets(id).is_some(),
                        "the selected segment snapshot must own packed planes"
                    );
                }
            }
            let requirements = seismic_metal::model::requirements(&execution).unwrap();
            assert!(
                requirements.unmapped.is_empty(),
                "{name}: {:?}",
                requirements.unmapped
            );
            let emitted = seismic_metal::msl::emit_execution(&execution).unwrap();
            assert_eq!(
                emitted
                    .launches
                    .iter()
                    .map(|l| l.declared_threadgroup_bytes)
                    .sum::<u64>(),
                0,
                "participant preparations must remain private"
            );
            assert!(execution
                .memory()
                .launches()
                .iter()
                .flat_map(|l| &l.arrays)
                .all(|a| a.declaration.capacity
                    <= 3 * if preparation == 2 {
                        256
                    } else {
                        segment as u64
                    }));
            let out = device.buffer(12).unwrap();
            let buffers = emitted
                .buffers
                .iter()
                .map(|b| match b.parameter.as_str() {
                    "x" => &x,
                    "gate" => &gate[representation.plane_index(&b.plane).unwrap()],
                    "up" => &up[representation.plane_index(&b.plane).unwrap()],
                    "out" => &out,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            let pipeline = device.compile(emitted).unwrap();
            device.run(&pipeline, &buffers, &[], 1).unwrap();
            let actual = out.read(12);
            if let Some(expected) = &expected {
                assert_eq!(
                    &actual, expected,
                    "{name}/{segment}/{width} preparation={preparation} traversal={unroll}"
                );
            } else {
                expected = Some(actual);
            }
        }
    }
    assert!(bundled_reads > 0, "packed decoding must expose ordinary adjacent-word transfers");
}
