//! Packet reuse must be reachable from ordinary grouped, bounded contractions.
//! Compare the exact decoded-element execution with the packet producer under
//! the same contraction choices; this does not claim completed model selection.
use seismic_lang::{
    Scope,
    interp::{Rng, TensorData},
    lower::{Options, lower_selected},
    lowered_ir::{Alternative, DecisionKind, LoweredIr},
    program::{SourceFile, compile},
    repr,
    types::Elem,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;

fn lower(backend: &str, k: i64, representation: Alternative, decoder: repr::PacketDecoder) -> LoweredIr {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "packet_projection.seismic.portable".into(),
        scope: Scope::Portable,
        text: format!(
            r#"
fn paired(x:tensor[1,{k}] f32,gate:tensor[5,768] W,up:tensor[5,768] W,out:tensor[1,5] f32):
  for col in parallel:
    a = load(x)
    g = load(gate[col:col+1,0:{k}])
    u = load(up[col:col+1,0:{k}])
    left = tile[1,1] f32
    right = tile[1,1] f32
    for i,j in owned(left): left[i,j] = 0.0
    for i,j in owned(right): right[i,j] = 0.0
    matmul(a,g,left)
    matmul(a,u,right)
    for i,j in owned(left): left[i,j] = left[i,j] / (1.0 + exp(-left[i,j])) * right[i,j]
    store(left,out[:,col:col+1])
"#
        ),
    });
    let p = compile(&sources, &["cpu".into(), "metal".into(), "cuda".into()]).unwrap();
    let mut packets = 0;
    let f = lower_selected(
        &p,
        "paired",
        backend,
        &HashMap::new(),
        &HashMap::from([("W".into(), Elem::Repr("q5k".into()))]),
        &Options::default(),
        &mut |d| {
            Ok(match &d.kind {
                DecisionKind::OutputGroup { .. } => Alternative::OutputWidth(3),
                DecisionKind::Stream { maximum, .. } => {
                    Alternative::StreamCapacity(128.min(*maximum))
                }
                DecisionKind::Producer { ty, .. }
                    if ty.shaped().is_some_and(|s| {
                        s.shape.last().and_then(|n| n.as_constant()) == Some(k)
                    }) && d.alternatives.contains(&Alternative::Recompute) =>
                {
                    Alternative::Recompute
                }
                DecisionKind::StreamFusion { .. } => Alternative::Fuse,
                DecisionKind::Intermediate { .. }
                    if d.alternatives.contains(&Alternative::RetainLocal) =>
                {
                    Alternative::RetainLocal
                }
                DecisionKind::PacketDecode { .. } => Alternative::PacketWidth(7),
                DecisionKind::PacketDecoder { .. } => Alternative::PacketDecoder(decoder),
                DecisionKind::Representation { .. } => {
                    if d.alternatives.contains(&representation) {
                        if representation == Alternative::DecodedPackets {
                            packets += 1;
                        }
                        representation.clone()
                    } else {
                        Alternative::Decoded
                    }
                }
                _ => d.alternatives.get(0).unwrap(),
            })
        },
    )
    .unwrap();
    if representation == Alternative::DecodedPackets {
        assert!(
            packets > 0,
            "ordinary paired projection lost packet decoding"
        );
    }
    // Bounded projection must not secretly decode the whole logical weight row.
    for v in &f.vars {
        if v.name.starts_with("decoded_packets_") {
            assert!(
                v.ty.shaped()
                    .unwrap()
                    .shape
                    .last()
                    .unwrap()
                    .as_constant()
                    .unwrap()
                    <= 128
            );
        }
    }
    f
}

fn exercise(device: Device, candidate: Candidate) {
    let r = repr::lookup("q5k").unwrap();
    let gate = TensorData::random_packed(&mut Rng(0x145622), r, vec![5, 768]);
    let up = TensorData::random_packed(&mut Rng(0x592133), r, vec![5, 768]);
    for k in [512, 544] {
        let x = (0..k)
            .map(|i| ((i % 13) as f32 - 6.0) / 256.0)
            .collect::<Vec<_>>();
        let mut baseline = None;
        for (representation, decoder) in [(Alternative::Decoded, repr::PacketDecoder::Specialized), (Alternative::DecodedPackets, repr::PacketDecoder::Specialized), (Alternative::DecodedPackets, repr::PacketDecoder::Indexed)] {
            let ir = lower(device.backend(), k, representation, decoder);
            let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
            let xb = device
                .buffer_from(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let gp = gate
                .device_bytes()
                .iter()
                .map(|p| device.buffer_from(p).unwrap())
                .collect::<Vec<_>>();
            let up = up
                .device_bytes()
                .iter()
                .map(|p| device.buffer_from(p).unwrap())
                .collect::<Vec<_>>();
            let output = device.buffer(20).unwrap();
            let buffers = kernel
                .buffers()
                .iter()
                .map(|slot| match slot.parameter.as_str() {
                    "x" => xb.clone(),
                    "gate" => gp[r.plane_index(&slot.plane).unwrap()].clone(),
                    "up" => up[r.plane_index(&slot.plane).unwrap()].clone(),
                    "out" => output.clone(),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            kernel.execute(&buffers, &[]).unwrap();
            let mut bytes = vec![0; 20];
            output.read(&mut bytes).unwrap();
            if let Some(expected) = &baseline {
                assert_eq!(
                    &bytes, expected,
                    "K={k}: packet decode changed fused projection values"
                );
            } else {
                baseline = Some(bytes);
            }
        }
    }
}

#[test]
fn cpu_packet_decode_composes_with_grouped_paired_projection_and_tails() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_packet_decode_composes_with_grouped_paired_projection_and_tails() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_packet_decode_composes_with_grouped_paired_projection_and_tails() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}

fn dynamic_decode(device: Device, candidate: Candidate) {
    for name in ["q4g64", "q5k", "q6k", "iq4g32"] {
        let p = compile(&[SourceFile {
            path: "dynamic_packets.seismic.portable".into(), scope: Scope::Portable,
            text: format!("fn decode(x:tensor[2,256] {name},end:i32,out:tensor[2,256] f32):\n  values = load(x[:,0:end])\n  result = tile[2,256] f32\n  for row,col in owned(result):\n    if col < extent(values,1): result[row,col] = values[row,col]\n    else: result[row,col] = -123.0\n  store(result,out)\n"),
        }], &[]).unwrap();
        let mut selected = false;
        let ir = lower_selected(
            &p,
            "decode",
            device.backend(),
            &HashMap::new(),
            &HashMap::new(),
            &Options::default(),
            &mut |d| {
                Ok(if let DecisionKind::PacketDecode { group, .. } = d.kind {
                    assert_eq!(d.alternatives.len(), group as usize);
                    for width in 1..=group {
                        assert!(d.alternatives.contains(&Alternative::PacketWidth(width)));
                    }
                    Alternative::PacketWidth(7.min(group))
                } else if matches!(d.kind, DecisionKind::Representation { .. }) {
                    assert!(d.alternatives.contains(&Alternative::DecodedPackets));
                    selected = true;
                    Alternative::DecodedPackets
                } else {
                    d.alternatives.get(0).unwrap()
                })
            },
        )
        .unwrap();
        assert!(selected);
        let mut kernel = device.compile(&ir, candidate.clone()).unwrap();
        let input = TensorData::random_packed(
            &mut Rng(0x472678),
            repr::lookup(name).unwrap(),
            vec![2, 256],
        );
        let planes = input
            .device_bytes()
            .iter()
            .map(|bytes| device.buffer_from(bytes).unwrap())
            .collect::<Vec<_>>();
        for end in [
            -1, 0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 999,
        ] {
            let output = device
                .buffer_from(&(-123.0f32).to_le_bytes().repeat(512))
                .unwrap();
            let mut buffers = planes.clone();
            buffers.push(output.clone());
            kernel.execute(&buffers, &[f64::from(end)]).unwrap();
            let mut bytes = vec![0; 2048];
            output.read(&mut bytes).unwrap();
            for (i, bytes) in bytes.chunks_exact(4).enumerate() {
                let expected = if i % 256 < end.clamp(0, 256) as usize {
                    input.get(i) as f32
                } else {
                    -123.0
                };
                assert_eq!(
                    f32::from_le_bytes(bytes.try_into().unwrap()).to_bits(),
                    expected.to_bits(),
                    "{name} end={end} element={i}"
                );
            }
        }
    }
}

#[test]
fn cpu_packet_decode_captures_dynamic_prefixes_and_preserves_empty_and_partial_groups() {
    dynamic_decode(
        Device::cpu(),
        Candidate::Cpu {
            loads: LoadStrategy::Materialize,
        },
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_packet_decode_captures_dynamic_prefixes_and_preserves_empty_and_partial_groups() {
    dynamic_decode(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_packet_decode_composes_with_reused_activation_matrix_staging() {
    use seismic_lang::lowered_ir::Choice;
    let device = Device::metal().unwrap();
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "packet_matrix.seismic.portable".into(), scope: Scope::Portable,
        text: "fn matrix(x:tensor[9,65] f32,w:tensor[17,128] q4g64,out:tensor[9,17] f32):\n  a = load(x)\n  b = load(w[:,0:65])\n  c = tile[9,17] f32\n  for i,j in owned(c): c[i,j] = 0.125\n  matmul(a,b,c)\n  store(c,out)\n".into(),
    });
    let p = compile(&sources, &["cpu".into(), "metal".into(), "cuda".into()]).unwrap();
    let input = (0..9 * 65)
        .map(|i| ((i % 19) as f32 - 9.0) / 256.0)
        .collect::<Vec<_>>();
    let weight = TensorData::random_packed(
        &mut Rng(0x529016),
        repr::lookup("q4g64").unwrap(),
        vec![17, 128],
    );
    let xb = device
        .buffer_from(
            &input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let wp = weight
        .device_bytes()
        .iter()
        .map(|p| device.buffer_from(p).unwrap())
        .collect::<Vec<_>>();
    let mut baseline = None;
    for block in [2, 3] {
        let mut packet_selected = false;
        let ir = lower_selected(
            &p,
            "matrix",
            "metal",
            &HashMap::new(),
            &HashMap::new(),
            &Options::default(),
            &mut |d| {
                Ok(match &d.kind {
                    DecisionKind::Construct { name, .. } if name == "matmul" => {
                        Alternative::Body(Choice::Block(block))
                    }
                    DecisionKind::Stream { maximum, .. } => Alternative::StreamCapacity(*maximum),
                    DecisionKind::Representation { .. } => {
                        assert!(d.alternatives.contains(&Alternative::DecodedPackets));
                        packet_selected = true;
                        Alternative::DecodedPackets
                    }
                    _ => d.alternatives.get(0).unwrap(),
                })
            },
        )
        .unwrap();
        assert!(packet_selected);
        assert!(
            ir.selections
                .iter()
                .any(|s| s.construct == "matmul" && s.choice == Choice::Block(block))
        );
        let mut config = match Candidate::Metal(Default::default()) {
            Candidate::Metal(config) => config,
            _ => unreachable!(),
        };
        config.sg_per_tg = 1;
        let mut kernel = device.compile(&ir, Candidate::Metal(config)).unwrap();
        let out = device.buffer(9 * 17 * 4).unwrap();
        let mut buffers = vec![xb.clone()];
        buffers.extend(wp.iter().cloned());
        buffers.push(out.clone());
        kernel.execute(&buffers, &[]).unwrap();
        let mut bytes = vec![0; 9 * 17 * 4];
        out.read(&mut bytes).unwrap();
        if let Some(expected) = &baseline {
            assert_eq!(
                &bytes, expected,
                "activation reuse changed matrix/tail arithmetic"
            );
        } else {
            baseline = Some(bytes);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_paired_matrix_calls_share_prepared_activation_without_changing_tails() {
    use seismic_lang::lowered_ir::Choice;
    let device = Device::metal().unwrap();
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "paired_matrix.seismic.portable".into(),
        scope: Scope::Portable,
        text: r#"
fn paired(x:tensor[9,65] f32,g:tensor[17,128] q4g64,u:tensor[17,128] q4g64,out:tensor[9,17] f32):
  a = load(x)
  gate = load(g[:,0:65])
  up = load(u[:,0:65])
  left = tile[9,17] f32
  for i,j in owned(left): left[i,j] = 0.125
  matmul(a,gate,left)
  right = tile[9,17] f32
  for i,j in owned(right): right[i,j] = -0.25
  matmul(a,up,right)
  for i,j in owned(left): left[i,j] = left[i,j] * right[i,j]
  store(left,out)
"#
        .into(),
    });
    let p = compile(&sources, &["cpu".into(), "metal".into(), "cuda".into()]).unwrap();
    let input = (0..9 * 65)
        .flat_map(|i| (((i % 19) as f32 - 9.0) / 256.0).to_le_bytes())
        .collect::<Vec<_>>();
    let mut buffers = vec![device.buffer_from(&input).unwrap()];
    for seed in [0x58192, 0x991472] {
        let w = TensorData::random_packed(
            &mut Rng(seed),
            repr::lookup("q4g64").unwrap(),
            vec![17, 128],
        );
        buffers.extend(
            w.device_bytes()
                .iter()
                .map(|p| device.buffer_from(p).unwrap()),
        );
    }
    let out = device.buffer(9 * 17 * 4).unwrap();
    buffers.push(out.clone());
    let mut baseline = None;
    for (block, fused) in [3, 4, 7, 8, 9].into_iter().flat_map(|block| [(block, false), (block, true)]) {
        if !fused { baseline = None; }
        let mut fusion = 0;
        let mut shared = 0;
        let ir = lower_selected(
            &p,
            "paired",
            "metal",
            &HashMap::new(),
            &HashMap::new(),
            &Options::default(),
            &mut |d| {
                Ok(match &d.kind {
                    DecisionKind::Construct { name, .. } if name == "matmul" => {
                        Alternative::Body(Choice::Block(block))
                    }
                    DecisionKind::Stream { maximum, .. } => Alternative::StreamCapacity(*maximum),
                    DecisionKind::RangeFusion { .. } if fused => {
                        fusion += 1;
                        Alternative::Fuse
                    }
                    DecisionKind::Intermediate { .. }
                        if fused && d.alternatives.contains(&Alternative::RetainLocal) =>
                    {
                        shared += 1;
                        Alternative::RetainLocal
                    }
                    DecisionKind::Representation { .. } => Alternative::DecodedPackets,
                    _ => d.alternatives.get(0).unwrap(),
                })
            },
        )
        .unwrap();
        if fused {
            assert!(fusion > 0, "paired row blocks must admit serial fusion");
            assert!(
                shared > 0,
                "fusion must expose the common activation preparation"
            );
        }
        let Candidate::Metal(mut config) = Candidate::Metal(Default::default()) else {
            unreachable!()
        };
        config.sg_per_tg = 1;
        let execution = seismic_runtime::execution::Execution::prepare(
            &ir,
            Candidate::Metal(config),
            &device.facts(),
        )
        .unwrap();
        let seismic_runtime::execution::Account::MetalStorage {
            implementation,
            account,
        } = execution.account().unwrap()
        else {
            unreachable!()
        };
        let staged_operands = implementation
            .memory()
            .launches()
            .iter()
            .flat_map(|launch| &launch.arrays)
            .filter(|array| if block == 9 {
                array.declaration.capacity == 64 && ["A0", "A1", "B0", "B1"].iter().any(|name| array.declaration.symbol.contains(name))
            } else { array.declaration.capacity == 8 * 65 })
            .count();
        assert_eq!(
            staged_operands,
            if block == 9 { if fused { 6 } else { 8 } } else if fused { 3 } else { 4 },
            "matrix preparation must be shared in the actual memory plan"
        );
        let packet_caches = implementation
            .memory()
            .launches()
            .iter()
            .flat_map(|launch| &launch.arrays)
            .filter(|array| array.declaration.symbol.starts_with("decoded_packets"))
            .collect::<Vec<_>>();
        assert_eq!(packet_caches.len(), 2);
        assert!(packet_caches.iter().all(|a| a.declaration.placement
            == seismic_realization::dispatch::TilePlacement::GroupShared));
        assert!(account.launches.iter().all(|launch| {
            launch
                .declared_shared_array_bytes_per_group
                .bounds()
                .is_some()
        }));
        let mut kernel = device.compile_execution(execution).unwrap();
        kernel.execute(&buffers, &[]).unwrap();
        let mut bytes = vec![0; 9 * 17 * 4];
        out.read(&mut bytes).unwrap();
        if let Some(expected) = &baseline {
            assert_eq!(
                &bytes, expected,
                "paired fusion changed the selected cover’s matrix accumulations or scalar tails"
            );
        } else {
            baseline = Some(bytes);
        }
    }
}
