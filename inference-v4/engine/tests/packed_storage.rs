use seismic_lang::{interp::{Rng, TensorData}, program::{compile,SourceFile}, repr, Scope};
use seismic_runtime::{Candidate,Device};

fn exercise(device: Device, loads: seismic_realization::LoadStrategy) {
    for decoded in [false, true] {
    let candidate = match device.backend() {
        "cpu" => Candidate::Cpu { loads },
        "cuda" => Candidate::Cuda { options: seismic_realization::ScalarOptions { dispatch: seismic_realization::Dispatch::ParallelRoot, loads }, threads_per_block:32 },
        #[cfg(target_os="macos")]
        "metal" => { let mut c = Candidate::Metal(Default::default()); if let Candidate::Metal(config) = &mut c { config.loads = loads; } c },
        _ => unreachable!(),
    };
    for name in ["q4k", "q5k", "q6k"] {
        let representation=repr::lookup(name).unwrap();
        let input=TensorData::random_packed(&mut Rng(0x1477_abcd),representation,vec![2,512]);
        for text in [
            format!("fn decode(x: tensor[2,512] {name}, out: tensor[2,255] f32):\n  for row in parallel:\n    values = load(x[row,1:256])\n    result = tile[255] f32\n    for i in owned(result): result[i] = values[i]\n    store(result,out[row])\n"),
            format!("fn decode(x: tensor[2,512] {name}, out: tensor[2,255] f32):\n  values = load(x[:,1:256])\n  result = tile[2,255] f32\n  for row,i in owned(result): result[row,i] = values[row,i]\n  store(result,out)\n"),
        ] {
        let program=compile(&[SourceFile{path:"packed.seismic.portable".into(),scope:Scope::Portable,text}],&[]).unwrap();
        let lowered=lower_storage(&program,"decode",device.backend(),decoded);
        let mut kernel=device.compile(&lowered,candidate.clone()).unwrap();
        let mut buffers=input.device_bytes().iter().map(|bytes|device.buffer_from(bytes).unwrap()).collect::<Vec<_>>();
        let out=device.buffer(2*255*4).unwrap(); buffers.push(out.clone());
        kernel.execute(&buffers,&[]).unwrap();
        let mut bytes=vec![0;2*255*4];out.read(&mut bytes).unwrap();
        for (i,bytes) in bytes.chunks_exact(4).enumerate() {
            assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()),input.get(i/255*512+i%255+1)as f32,"{name} {loads:?} element {i}");
        }
        }
        // Packet accessors of a nonzero aligned slice must address the same
        // physical planes as ordinary element decoding, including factor groups.
        for plane in representation.planes() {
            let length=plane.storage_elements(256).unwrap();
            let dtype=plane.dtype();
            let type_name=dtype.name();
            let text=format!("fn access(x: tensor[2,512] {name}, out: tensor[2,{length}] {type_name}):\n  for row in parallel:\n    values = load(x[row,256:512])\n    store(values.{},out[row])\n",plane.name);
            let backend=device.backend().to_string();
            let program=compile(&[SourceFile{path:format!("packet.seismic.{backend}").into(),scope:Scope::Backend(backend.clone()),text}],&[backend]).unwrap();
            let lowered=lower_storage(&program,"access",device.backend(),decoded);
            let mut kernel=device.compile(&lowered,candidate.clone()).unwrap();
            let out=device.buffer((2*length*u64::from(dtype.bytes()))as usize).unwrap();
            let mut buffers=input.device_bytes().iter().map(|bytes|device.buffer_from(bytes).unwrap()).collect::<Vec<_>>();buffers.push(out.clone());
            kernel.execute(&buffers,&[]).unwrap();
            let bytes_per_slice=plane.bytes(256).unwrap()as usize;
            let mut bytes=vec![0;2*bytes_per_slice];out.read(&mut bytes).unwrap();
            let source=&input.device_bytes()[representation.plane_index(plane.name).unwrap()];
            for row in 0..2 { assert_eq!(&bytes[row*bytes_per_slice..(row+1)*bytes_per_slice],&source[(row*2+1)*bytes_per_slice..(row*2+2)*bytes_per_slice],"{name} {} {loads:?}",plane.name); }
        }
    }
    }
}
fn lower_storage(program: &seismic_lang::program::Program, name: &str, backend: &str, decoded: bool) -> seismic_lang::lowered_ir::LoweredIr {
    use seismic_lang::{lower::Options,lowered_ir::{Alternative,DecisionKind}};
    let mut count=0;
    let f=seismic_lang::lower::lower_selected(program,name,backend,&Default::default(),&Default::default(),&Options::default(),&mut |d| {
        Ok(if matches!(d.kind,DecisionKind::Representation {..}) { count+=1; if decoded {Alternative::Decoded}else{Alternative::Encoded} }else {d.alternatives.get(0).unwrap()})
    }).unwrap();
    assert!(count>0,"packed load must own its representation choice");
    f
}
#[test] fn cpu_compact_tail_and_packet_accessors(){for loads in [seismic_realization::LoadStrategy::BorrowProvenReadOnly,seismic_realization::LoadStrategy::Materialize]{exercise(Device::cpu(),loads)}}
#[test] #[ignore="requires CUDA hardware"] fn cuda_compact_tail_and_packet_accessors(){for loads in [seismic_realization::LoadStrategy::BorrowProvenReadOnly,seismic_realization::LoadStrategy::Materialize]{exercise(Device::cuda(0).unwrap(),loads)}}
#[cfg(target_os="macos")] #[test] #[ignore="requires Metal hardware"] fn metal_compact_tail_and_packet_accessors(){for loads in [seismic_realization::LoadStrategy::BorrowProvenReadOnly,seismic_realization::LoadStrategy::Materialize]{exercise(Device::metal().unwrap(),loads)}}
