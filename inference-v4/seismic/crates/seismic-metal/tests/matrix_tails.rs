//! Partial output fragments preserve seed/publication bounds and keep the K
//! remainder scalar. This qualifies a library cover, not automatic selection.
use seismic_lang::{Scope, program::{compile, SourceFile}, lower::{lower_selected, Options}, lowered_ir::{Alternative, Choice, DecisionKind}};
use seismic_metal::{execution::{self, Config}, model, terminal::Primitive};
use seismic_accounting::workload::{Allocation, BufferBinding, DerivationLimits, ScalarWorkload};

fn selected(m: i64, n: i64, k: i64, block: usize) -> execution::Execution { selected_panel(m,n,k,block,1) }
fn selected_panel(m: i64, n: i64, k: i64, block: usize, width: i64) -> execution::Execution {
    let program = compile(&[
        SourceFile { path: "matmul.seismic.portable".into(), scope: Scope::Portable, text: include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.portable").into() },
        SourceFile { path: "matmul.seismic.metal".into(), scope: Scope::Backend("metal".into()), text: include_str!("../../../../seismic-std/lib/constructs/matmul.seismic.metal").into() },
        SourceFile { path: "partial_matrix.seismic.portable".into(), scope: Scope::Portable,
            text: format!("fn evaluate(a:tensor[{m},80] f32,b:tensor[{n},80] f32,seed:tensor[{m},{n}] f32,out:tensor[{m},{n}] f32):\n  x=load(a[:,0:{k}])\n  y=load(b[:,0:{k}])\n  c=load(seed)\n  matmul(x,y,c)\n  store(c,out)\n") },
    ], &["metal".into()]).unwrap();
    let mut panels = 0;
    let ir = lower_selected(&program, "evaluate", "metal", &Default::default(), &Default::default(), &Options::default(), &mut |decision| Ok(match decision.kind {
        DecisionKind::MatrixPanel { iterations, .. } => { panels += 1; assert!(width <= iterations); Alternative::MatrixPanelWidth(width) },
        DecisionKind::Construct { .. } => Alternative::Body(Choice::Block(block)),
        DecisionKind::Stream { maximum, .. } => Alternative::StreamCapacity(maximum),
        _ => decision.alternatives.get(0).unwrap(),
    })).unwrap();
    if width > 1 { assert!(panels > 0, "the ordinary fragment loop must expose operand panels"); }
    execution::prepare_storage_selected(&ir, Config { sg_per_tg: 1, loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly, ..Default::default() }, &mut |d| {
        Ok(if d.alternatives.contains(&seismic_realization::dispatch::TilePlacement::Replicated) { seismic_realization::dispatch::TilePlacement::Replicated } else { d.diagnostic() })
    }).unwrap()
}

#[test]
fn partial_output_fragments_account_complete_k_tiles_and_exact_publications() {
    for (m, n, k, block) in (6..=9).flat_map(|block| [(1, 3, 17, block), (9, 17, 65, block), (1, 3, 0, block)]) {
        let execution = selected(m, n, k, block);
        let sizes = [m * 80 * 4, n * 80 * 4, m * n * 4, m * n * 4];
        let workload = ScalarWorkload { integer_domains: Vec::new(), identity: format!("partial fragment {m}/{n}/{k}"),
            allocations: sizes.iter().enumerate().map(|(id, bytes)| Allocation { id: id as u64, bytes: *bytes as u64, alignment: 16, known_bytes: Default::default() }).collect(),
            buffers: sizes.iter().enumerate().map(|(allocation, bytes)| BufferBinding { allocation: allocation as u64, offset: 0, bytes: *bytes as u64 }).collect(), scalars: vec![],
        };
        let account = model::invocation_account(&execution, &workload, DerivationLimits { instructions: 1_000_000, operations: 1_000_000 }).unwrap();
        assert!(account.is_complete(), "{:?}", account.unmapped);
        let matrices: u64 = account.operations.iter().filter(|op| matches!(op.primitive, Primitive::MatrixMultiplyAccumulate { .. })).map(|op| op.instances).sum();
        let fragments = if block == 9 { 4 * ((m + 15) / 16) * ((n + 15) / 16) } else { ((m + 7) / 8) * ((n + 7) / 8) };
        assert_eq!(matrices, (fragments * (k / 8)) as u64);
        let stores: u64 = account.operations.iter().filter(|op| matches!(op.primitive, Primitive::Write { space: seismic_metal::terminal::Space::Device, .. })).map(|op| op.instances * op.lanes).sum();
        assert_eq!(stores, (m * n) as u64, "padded fragment elements cannot become publications");
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_partial_output_fragments_preserve_seeds_and_scalar_k_tails() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    for (m, n, k, block) in (6..=9).flat_map(|block| [(1, 3, 17, block), (9, 17, 65, block), (1, 3, 0, block)]) {
        let a: Vec<f32> = (0..m * 80).map(|i| (i % 13 - 6) as f32 / 16.0).collect();
        let b: Vec<f32> = (0..n * 80).map(|i| (i % 11 - 5) as f32 / 32.0).collect();
        let seed: Vec<f32> = (0..m * n).map(|i| if k == 0 { [f32::from_bits(0x80000000), f32::INFINITY, f32::from_bits(0x7fc12345)][i as usize % 3] } else { (i % 7 - 3) as f32 / 8.0 }).collect();
        let bytes = |values: &[f32]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let a_buffer = device.buffer_from(&bytes(&a)).unwrap();
        let b_buffer = device.buffer_from(&bytes(&b)).unwrap();
        let seed_buffer = device.buffer_from(&bytes(&seed)).unwrap();
        let output = device.buffer((m * n * 4) as usize).unwrap();
        let execution = selected(m, n, k, block);
        let emitted = seismic_metal::msl::prepare_execution(&execution).unwrap();
        let bindings = emitted.buffers.iter().map(|binding| match binding.parameter.as_str() {
            "a" => &a_buffer, "b" => &b_buffer, "seed" => &seed_buffer, "out" => &output, _ => unreachable!(),
        }).collect::<Vec<_>>();
        let kernel = device.compile(emitted.clone()).unwrap();
        device.run(&kernel, &bindings, &[], 1).unwrap();
        let mut expected = seed.clone();
        for row in 0..m { for column in 0..n { for inner in 0..k {
            let at = (row * n + column) as usize;
            expected[at] = a[(row * 80 + inner) as usize].mul_add(b[(column * 80 + inner) as usize], expected[at]);
        } } }
        // Dyadic inputs make this finite fixture exact, independently of the
        // admitted matrix reduction grouping. K=0 must retain seed bits alone.
        assert_eq!(output.read((m * n * 4) as usize), bytes(&expected));
        assert_eq!(seed_buffer.read((m * n * 4) as usize), bytes(&seed));
    }
}

#[cfg(target_os="macos")]
#[test]
#[ignore="requires Metal hardware"]
fn native_matrix_panels_preserve_fragment_updates_and_partial_panels() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let (m,n,k) = (9,17,73);
    let a: Vec<f32> = (0..m*80).map(|i| (i%13-6) as f32/16.0).collect();
    let b: Vec<f32> = (0..n*80).map(|i| (i%11-5) as f32/32.0).collect();
    let seed: Vec<f32> = (0..m*n).map(|i| (i%7-3) as f32/8.0).collect();
    let bytes = |v: &[f32]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let buffers = [device.buffer_from(&bytes(&a)).unwrap(),device.buffer_from(&bytes(&b)).unwrap(),device.buffer_from(&bytes(&seed)).unwrap(),device.buffer((m*n*4) as usize).unwrap()];
    let mut expected=seed.clone();
    for row in 0..m { for col in 0..n { for q in 0..k { let at=(row*n+col) as usize; expected[at]=a[(row*80+q) as usize].mul_add(b[(col*80+q) as usize],expected[at]); } } }
    for width in [1,2,3,9] {
        let execution=selected_panel(m,n,k,9,width);
        let emitted=seismic_metal::msl::prepare_execution(&execution).unwrap();
        let bindings=emitted.buffers.iter().map(|binding| &buffers[match binding.parameter.as_str() {"a"=>0,"b"=>1,"seed"=>2,"out"=>3,_=>unreachable!()}]).collect::<Vec<_>>();
        let kernel=device.compile(emitted.clone()).unwrap(); device.run(&kernel,&bindings,&[],1).unwrap();
        assert_eq!(buffers[3].read((m*n*4) as usize),bytes(&expected),"panel {width}");
        let workload=ScalarWorkload { integer_domains: Vec::new(), identity: format!("panel {width}"), allocations: [m*80*4,n*80*4,m*n*4,m*n*4].into_iter().enumerate().map(|(id,bytes)| Allocation {id:id as u64,bytes:bytes as u64,alignment:16,known_bytes:Default::default()}).collect(), buffers: [m*80*4,n*80*4,m*n*4,m*n*4].into_iter().enumerate().map(|(id,bytes)| BufferBinding {allocation:id as u64,offset:0,bytes:bytes as u64}).collect(), scalars:vec![] };
        let account=model::invocation_account(&execution,&workload,DerivationLimits{instructions:1_000_000,operations:1_000_000}).unwrap();
        assert!(account.is_complete(),"{:?}",account.unmapped);
        assert_eq!(account.operations.iter().filter(|op|matches!(op.primitive,Primitive::MatrixMultiplyAccumulate{..})).map(|op|op.instances).sum::<u64>(),72);
    }
}
