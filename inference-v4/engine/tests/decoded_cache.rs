use seismic_lang::{
    interp::{round_to, Rng, TensorData},
    lower::{lower_selected, Options},
    lowered_ir::{Alternative, DecisionKind},
    program::{compile, SourceFile},
    repr,
    types::DType,
    Scope,
};
use seismic_runtime::{Candidate, Device};

fn exercise(device: Device) {
    for rep in repr::REPRS {
        let name = rep.name;
        let rep = repr::lookup(name).unwrap();
        let word_count = rep.planes()[0].storage_elements(256).unwrap() as usize;
        let input = TensorData::random_packed(&mut Rng(0x743292), rep, vec![2, 512]);
        let backend = device.backend().to_string();
        let text = format!(
            r#"
fn cache(x:tensor[2,512] {name},out:tensor[4,2,255] f32,raw:tensor[2,{word_count}] u32):
  values = load(x[:,0:256])
  saved = values
  sliced = saved[:,1:256]
  result = tile[4,2,255] f32
  for channel,row,i in owned(result):
    if channel == 0: result[channel,row,i] = saved[row,i+1]
    else:
      if channel == 1: result[channel,row,i] = sliced[row,i]
      else:
        if channel == 2: result[channel,row,i] = values[row,i+1]
        else: result[channel,row,i] = f32(f16(saved[row,i+1])) + f32(bf16(saved[row,i+1]))
  store(result,out)
  store(values.words,raw)
"#
        );
        let p = compile(
            &[SourceFile {
                path: format!("cache.seismic.{backend}").into(),
                scope: Scope::Backend(backend.clone()),
                text,
            }],
            &[backend.clone()],
        )
        .unwrap();
        for representation in [Alternative::Encoded, Alternative::Decoded, Alternative::DecodedPackets] {
            let mut decisions = 0;
            let f = lower_selected(
                &p,
                "cache",
                &backend,
                &Default::default(),
                &Default::default(),
                &Options::default(),
                &mut |d| {
                    Ok(if matches!(d.kind, DecisionKind::Representation { .. }) {
                        decisions += 1;
                        assert!(d.alternatives.contains(&representation), "{name}: {representation:?} is unavailable");
                        representation.clone()
                    } else {
                        d.alternatives.get(0).unwrap()
                    })
                },
            )
            .unwrap();
            assert!(decisions > 0);
            let loads = seismic_realization::LoadStrategy::Materialize;
            let candidate = match backend.as_str() {
                "cpu" => Candidate::Cpu { loads },
                "cuda" => Candidate::Cuda {
                    options: seismic_realization::ScalarOptions {
                        dispatch: seismic_realization::Dispatch::Sequential,
                        loads,
                    },
                    threads_per_block: 32,
                },
                #[cfg(target_os = "macos")]
                "metal" => {
                    let mut c = Candidate::Metal(Default::default());
                    if let Candidate::Metal(config) = &mut c {
                        config.loads = loads;
                    }
                    c
                }
                _ => unreachable!(),
            };
            let mut kernel = device.compile(&f, candidate).unwrap();
            let mut buffers = input
                .device_bytes()
                .iter()
                .map(|b| device.buffer_from(b).unwrap())
                .collect::<Vec<_>>();
            let output = device.buffer(4 * 2 * 255 * 4).unwrap();
            let raw = device.buffer(2 * word_count * 4).unwrap();
            buffers.extend([output.clone(), raw.clone()]);
            kernel.execute(&buffers, &[]).unwrap();
            let mut bytes = vec![0; 4 * 2 * 255 * 4];
            output.read(&mut bytes).unwrap();
            for (i, b) in bytes.chunks_exact(4).enumerate() {
                let channel = i / (2 * 255);
                let row = i / 255 % 2;
                let column = i % 255 + 1;
                let original = input.get(row * 512 + column) as f32;
                let expected = match channel {
                    0 | 1 => original,
                    2 => input.get(row * 512 + column) as f32,
                    3 => {
                        round_to(DType::F16, original as f64) as f32
                            + round_to(DType::BF16, original as f64) as f32
                    }
                    _ => unreachable!(),
                };
                assert_eq!(
                    f32::from_le_bytes(b.try_into().unwrap()),
                    expected,
                    "{name} representation={representation:?}, element{i}"
                );
            }
            let mut bytes = vec![0; 2 * word_count * 4];
            raw.read(&mut bytes).unwrap();
            let raw_expected = &input.device_bytes()[0];
            for row in 0..2 {
                assert_eq!(
                    &bytes[row * word_count * 4..(row + 1) * word_count * 4],
                    &raw_expected[(row * 2) * word_count * 4..(row * 2 + 1) * word_count * 4]
                );
            }
        }
    }
}
#[test]
fn cpu_decoded_cache_preserves_snapshot_aliases_packets_and_casts() {
    exercise(Device::cpu())
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_decoded_cache_preserves_snapshot_aliases_packets_and_casts() {
    exercise(Device::cuda(0).unwrap())
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_decoded_cache_preserves_snapshot_aliases_packets_and_casts() {
    exercise(Device::metal().unwrap())
}

fn exercise_rebinding(device: Device) {
    for name in ["q4k", "q5k", "q6k"] {
        let input = TensorData::random_packed(
            &mut Rng(0x632487),
            repr::lookup(name).unwrap(),
            vec![2, 512],
        );
        for trips in 0..=2 {
            let text = format!(
                r#"
fn rebind(x:tensor[2,512] {name},out:tensor[2,2,253] f32):
  values = load(x[:,1:254])
  saved = values
  for iteration in range({trips}):
    if iteration == 0: values = load(x[:,2:255])
    else: values = load(x[:,258:511])
  result = tile[2,2,253] f32
  for version,row,i in owned(result):
    if version == 0: result[version,row,i] = saved[row,i]
    else: result[version,row,i] = values[row,i]
  store(result,out)
"#
            );
            let p = compile(
                &[SourceFile {
                    path: "rebind.seismic.portable".into(),
                    scope: Scope::Portable,
                    text,
                }],
                &[],
            )
            .unwrap();
            for decoded in [false, true] {
                let f = lower_selected(
                    &p,
                    "rebind",
                    device.backend(),
                    &Default::default(),
                    &Default::default(),
                    &Options::default(),
                    &mut |d| {
                        Ok(
                            if matches!(d.kind, DecisionKind::Representation { .. }) && decoded {
                                Alternative::Decoded
                            } else {
                                d.alternatives.get(0).unwrap()
                            },
                        )
                    },
                )
                .unwrap();
                let loads = seismic_realization::LoadStrategy::Materialize;
                let candidate = match device.backend() {
                    "cpu" => Candidate::Cpu { loads },
                    "cuda" => Candidate::Cuda {
                        options: seismic_realization::ScalarOptions {
                            dispatch: seismic_realization::Dispatch::Sequential,
                            loads,
                        },
                        threads_per_block: 32,
                    },
                    #[cfg(target_os = "macos")]
                    "metal" => {
                        let mut c = Candidate::Metal(Default::default());
                        if let Candidate::Metal(config) = &mut c {
                            config.loads = loads;
                        }
                        c
                    }
                    _ => unreachable!(),
                };
                let mut kernel = device.compile(&f, candidate).unwrap();
                let mut buffers = input
                    .device_bytes()
                    .iter()
                    .map(|b| device.buffer_from(b).unwrap())
                    .collect::<Vec<_>>();
                let out = device.buffer(2 * 2 * 253 * 4).unwrap();
                buffers.push(out.clone());
                kernel.execute(&buffers, &[]).unwrap();
                let mut bytes = vec![0; 2 * 2 * 253 * 4];
                out.read(&mut bytes).unwrap();
                for (i, b) in bytes.chunks_exact(4).enumerate() {
                    let version = i / (2 * 253);
                    let row = i / 253 % 2;
                    let column = i % 253;
                    let start = if version == 0 || trips == 0 {
                        1
                    } else if trips == 1 {
                        2
                    } else {
                        258
                    };
                    assert_eq!(
                        f32::from_le_bytes(b.try_into().unwrap()),
                        input.get(row * 512 + start + column) as f32,
                        "{name} decoded{decoded} trips{trips} value{i}"
                    );
                }
            }
        }
    }
}
#[test]
fn cpu_packed_rebinding_carries_packet_prefix_through_empty_loops_and_branches() {
    exercise_rebinding(Device::cpu())
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_packed_rebinding_carries_packet_prefix_through_empty_loops_and_branches() {
    exercise_rebinding(Device::cuda(0).unwrap())
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_packed_rebinding_carries_packet_prefix_through_empty_loops_and_branches() {
    exercise_rebinding(Device::metal().unwrap())
}

fn exercise_dynamic(device: Device) {
    for name in ["q4k", "q5k", "q6k"] {
        let mut storage_by_representation = std::collections::HashMap::new();
        for capacity in [512, 1024] {
            let input = TensorData::random_packed(
                &mut Rng(0x156374),
                repr::lookup(name).unwrap(),
                vec![capacity],
            );
            let text = format!(
                r#"
fn consume[N](raw:tile[N] {name},out:tensor[1] f32):
  values = tile[N] f16
  for k in owned(values): values[k] = f16(raw[k] + f32(k) / 8.0)
  result = tile[1] f32
  for i in owned(result): result[i] = 0.0
  result[0] = reduce(values,0,sum,ordered=true)
  store(result,out)
fn dynamic(x:tensor[{capacity}] {name},visible:tensor[2] i32,out:tensor[1] f32):
  raw = load(x[visible[0]:visible[1]])
  consume(raw,out)
"#
            );
            let p = compile(
                &[SourceFile {
                    path: "dynamic-cache.seismic.portable".into(),
                    scope: Scope::Portable,
                    text,
                }],
                &[],
            )
            .unwrap();
            for decoded in [false, true] {
                let f = lower_selected(
                    &p,
                    "dynamic",
                    device.backend(),
                    &Default::default(),
                    &Default::default(),
                    &Options {
                        piece: Some(7),
                        ..Default::default()
                    },
                    &mut |d| {
                        Ok(match d.kind {
                            DecisionKind::Representation { .. } => {
                                if decoded {
                                    Alternative::Decoded
                                } else {
                                    Alternative::Encoded
                                }
                            }
                            DecisionKind::Producer { .. }
                                if d.alternatives.contains(&Alternative::Recompute) =>
                            {
                                Alternative::Recompute
                            }
                            _ => d.alternatives.get(0).unwrap(),
                        })
                    },
                )
                .unwrap();
                let owner = f
                    .body
                    .iter()
                    .find_map(|s| match &s.kind {
                        seismic_lang::ir::StmtKind::LoadLoop { vars, capacity, .. } => {
                            assert_eq!(*capacity, Some(7));
                            Some(vars)
                        }
                        _ => None,
                    })
                    .expect("bounded actual input snapshots");
                assert!(f.decisions.iter().any(|d|matches!(&d.domain.kind,DecisionKind::Representation {variable} if owner.contains(variable))),"cache must belong to bounded stream input");
                let loads = seismic_realization::LoadStrategy::Materialize;
                let candidate = match device.backend() {
                    "cpu" => Candidate::Cpu { loads },
                    "cuda" => Candidate::Cuda {
                        options: seismic_realization::ScalarOptions {
                            dispatch: seismic_realization::Dispatch::Sequential,
                            loads,
                        },
                        threads_per_block: 32,
                    },
                    #[cfg(target_os = "macos")]
                    "metal" => {
                        let mut c = Candidate::Metal(Default::default());
                        if let Candidate::Metal(config) = &mut c {
                            config.loads = loads;
                        }
                        c
                    }
                    _ => unreachable!(),
                };
                let data = seismic_lang::demand::data_variables(&f.body);
                for (variable, definition) in f.vars.iter().enumerate() {
                    if matches!(definition.name.as_str(), "raw" | "values") {
                        assert!(
                            !data.contains(&variable),
                            "whole producer retains data: {name} decoded{decoded} {}",
                            definition.name
                        );
                    }
                }
                let execution =
                    seismic_runtime::execution::Execution::prepare(&f, candidate, &device.facts())
                        .unwrap();
                use seismic_accounting::quantity::Count;
                use seismic_runtime::execution::Account;
                let storage = match execution.account().unwrap() {
                    Account::CpuScalarIr { account, .. } => {
                        vec![Count::Exact(account.scratch_bytes_per_invocation)]
                    }
                    Account::Cuda(phases) => phases
                        .iter()
                        .map(|phase| Count::Exact(phase.scalar_ir().scratch_bytes_per_invocation))
                        .collect(),
                    #[cfg(target_os = "macos")]
                    Account::MetalStorage { account, .. } => {
                        let mut storage = vec![account.retained_scratch_bytes];
                        for launch in account.launches {
                            storage.extend([
                                launch.declared_private_array_bytes_per_lane,
                                launch.declared_shared_array_bytes_per_group,
                                launch.declared_fragment_payload_bytes_per_subgroup,
                            ]);
                        }
                        storage
                    }
                };
                assert!(storage
                    .iter()
                    .all(|quantity| matches!(quantity, Count::Exact(_))));
                if let Some(previous) = storage_by_representation.insert(decoded, storage.clone()) {
                    assert_eq!(
                        storage, previous,
                        "packed stream storage grows with source capacity: {name} decoded{decoded}"
                    );
                }
                let mut kernel = device.compile_execution(execution).unwrap();
                for (start, end) in [(0i32, 0i32), (1, 13), (3, 258), (257, capacity as i32)] {
                    let mut buffers = input
                        .device_bytes()
                        .iter()
                        .map(|b| device.buffer_from(b).unwrap())
                        .collect::<Vec<_>>();
                    buffers.push(
                        device
                            .buffer_from(
                                &[start, end]
                                    .iter()
                                    .flat_map(|n| n.to_le_bytes())
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap(),
                    );
                    let out = device.buffer(4).unwrap();
                    buffers.push(out.clone());
                    kernel.execute(&buffers, &[]).unwrap();
                    let mut bytes = [0; 4];
                    out.read(&mut bytes).unwrap();
                    let expected = (start..end).enumerate().fold(0f32, |sum, (k, i)| {
                        let leaf = round_to(
                            DType::F16,
                            (input.get(i as usize) as f32 + k as f32 / 8.0) as f64,
                        ) as f32;
                        round_to(DType::F16, (sum + leaf) as f64) as f32
                    });
                    assert_eq!(
                        f32::from_le_bytes(bytes),
                        expected,
                        "{name} decoded{decoded} {start}:{end}"
                    );
                }
            }
        }
    }
}
#[test]
fn cpu_dynamic_packed_producer_cache_has_bounded_ownership_and_exact_casts() {
    exercise_dynamic(Device::cpu())
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_dynamic_packed_producer_cache_has_bounded_ownership_and_exact_casts() {
    exercise_dynamic(Device::cuda(0).unwrap())
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_dynamic_packed_producer_cache_has_bounded_ownership_and_exact_casts() {
    exercise_dynamic(Device::metal().unwrap())
}
