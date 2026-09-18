use seismic_cuda::{model, tuning};
use seismic_accounting::{schedule,workload};
use seismic_lang::{program::{compile,SourceFile},Scope};
use seismic_realization::{Dispatch,LoadStrategy,ScalarOptions};
const SOURCE:&str="fn kernel(x: tensor[3,35] f32, out: tensor[3] f32):\n  for row in parallel:\n    partial = 0.0\n    for k in lanes(35,1):\n      partial = fma(x[row,k],1.0,partial)\n    total = simd_sum(partial)\n    y = tile[1] f32\n    for i in owned(y): y[i] = total\n    store(y,out[row:row+1])\n";
fn lowered(source:&str)->seismic_lang::lowered_ir::LoweredIr{
 let program=compile(&[SourceFile{path:"subgroup.seismic.cuda".into(),scope:Scope::Backend("cuda".into()),text:source.into()}],&["cuda".into()]).unwrap();
 seismic_lang::lower::lower(&program,"kernel","cuda",&Default::default()).unwrap()
}
fn device()->seismic_cuda::DeviceInfo{seismic_cuda::DeviceInfo{name:"synthetic full-warp device".into(),compute_capability:(8,0),driver_version:0,max_threads_per_block:64,max_grid_x:8,warp_size:32,multiprocessors:1,global_memory_bytes:4096,l2_cache_bytes:0,max_threads_per_multiprocessor:128,registers_32bit_per_multiprocessor:4096,shared_bytes_per_multiprocessor:4096}}
fn options()->ScalarOptions{ScalarOptions{dispatch:Dispatch::ParallelRoot,loads:LoadStrategy::Materialize}}
#[test]
fn source_collective_retains_communication_dispatch_and_single_publication(){
 let function=lowered(SOURCE);let selected=tuning::prepare_fixed(&function,&device(),options(),64).unwrap();let execution=&selected[0];
 assert_eq!(execution.dispatch().lanes_per_item,32);assert_eq!(execution.dispatch().items_per_group,2);assert_eq!(execution.dispatch().groups,2);
 assert_eq!(execution.storage().status_bytes,3*32*4);
 let ptx=seismic_cuda::ptx::print(execution.target_plan());assert_eq!(ptx.matches("shfl.sync.bfly").count(),5);assert_eq!(ptx.matches("shfl.sync.idx").count(),1);
 let mut primitives=Vec::new();for requirement in model::requirements(execution){if let model::Requirement::Instruction(p)=requirement{if !primitives.contains(&p){primitives.push(p);}}}
 let hardware=model::CudaHardware{identity:"synthetic unit service; no native timing claim".into(),scope:model::Scope::HypotheticalInstructionPreservingPtxV1,timebase:schedule::Timebase{seconds_numerator:1,seconds_denominator:1_000_000_000},execution_units:1,warp_width:32,cohorts:model::CohortPolicy::LowestPosition,internal_alignment:256,resources:vec![model::Resource{name:"issue".into(),scope:model::ResourceScope::Device,capacity:1,unit:schedule::CapacityUnit::Slots}],timings:primitives.into_iter().map(|primitive|model::PrimitiveTiming{primitive,latency:model::Ticks::Fixed(1),reservations:vec![model::Reservation{resource:0,offset:0,duration:model::Ticks::Fixed(1),units:model::Amount::fixed(1)}]}).collect(),block_residency:vec![],per_unit_residency:vec![model::UnitResidency{name:"resident blocks".into(),capacity:2,units_per_block:model::Amount::fixed(1)}]};
 let bindings=workload::ScalarWorkload{identity:"independent inputs".into(),allocations:execution.program().buffers.iter().enumerate().map(|(i,b)|workload::Allocation{id:i as u64,bytes:b.bytes as u64,alignment:256,known_bytes:Default::default()}).collect(),buffers:execution.program().buffers.iter().enumerate().map(|(i,b)|workload::BufferBinding{allocation:i as u64,offset:0,bytes:b.bytes as u64}).collect(),scalars:vec![]};
 let derived=model::derive_sequence(&selected,&hardware,&bindings,workload::DerivationLimits{instructions:1_000_000,operations:100_000}).unwrap();
 assert!(derived.operations.len()>18);assert!(derived.lower_bound().unwrap()>18);
 assert!(tuning::prepare_fixed(&function,&device(),options(),33).is_err());
}
#[test]
fn varying_publication_is_unsupported_in_replicated_warp_form(){
 let source=SOURCE.replace("y[i] = total","y[i] = partial");let result=tuning::prepare_fixed(&lowered(&source),&device(),options(),32);
 assert!(result.err().unwrap().contains("uniform"));
}
#[test]
#[ignore="requires CUDA hardware"]
fn native_warp_sum_handles_partial_lane_work_and_padding_warps(){
 let device=seismic_cuda::Device::open(0).unwrap();let function=lowered(SOURCE);
 for threads in [32,64,96]{
  let selected=tuning::prepare_fixed(&function,&device.info,options(),threads).unwrap();
  let mut kernel=device.compile_executions(selected).unwrap();
  let data=(0..105).map(|i|(i as f32-17.0)/8.0).collect::<Vec<_>>();
  let input=device.buffer_from(&data.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).unwrap();let output=device.buffer(12).unwrap();
  kernel.execute(&[input,output.clone()],&[],false).unwrap();let mut bytes=vec![0;12];output.read(&mut bytes).unwrap();
  for(row,bytes)in bytes.chunks_exact(4).enumerate(){let expected=data[row*35..row*35+35].iter().sum::<f32>();assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()),expected);}
 }
}
