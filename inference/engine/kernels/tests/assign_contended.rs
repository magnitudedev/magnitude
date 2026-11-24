//! Focused tests verifying that `routed.seismic::assign` (portable Seismic
//! reference) and `resolve.hpp::streaming_assign` (native Metal kernel) price
//! CPU-vs-GPU offload using `POLICY_LINK_CONTENDED_NS` vs uncontended
//! `POLICY_LINK_NS`, offloading `C` of `N` misses iff
//! `max(T_gpu_contended(N - C), T_cpu(C)) < T_gpu_uncontended(N)`.

mod common;

use magnitude_kernels::constants;
use seismic_lang::entry::ElementBindings;
use seismic_lang::failure::SourceTermination;
use seismic_lang::interp::{Arg, Interpreter, TensorData};
use seismic_lang::types::DType;

fn policy_words(
    cpu: i32,
    cpu_row: i32,
    handshake: i32,
    link: i32,
    link_contended: i32,
    hit: i32,
) -> Vec<i32> {
    let mut words = vec![0; constants::POLICY_WORDS as usize];
    words[constants::POLICY_PRICED as usize] = 1;
    words[constants::POLICY_CPU_NS as usize] = cpu;
    words[constants::POLICY_CPU_ROW_NS as usize] = cpu_row;
    words[constants::POLICY_HANDSHAKE_NS as usize] = handshake;
    words[constants::POLICY_LINK_NS as usize] = link;
    words[constants::POLICY_LINK_CONTENDED_NS as usize] = link_contended;
    words[constants::POLICY_HIT_NS as usize] = hit;
    words
}

fn run_assign(
    module: &seismic_lang::checked::CheckedModule,
    m: usize,
    routes: &[i32],
    records: &[i32],
    policy: &[i32],
) -> (Vec<i32>, Vec<i32>) {
    let k = routes.len() / m;
    let entry = module
        .entry(module.entry_named("assign").unwrap(), &ElementBindings::new())
        .unwrap();
    let mut interpreter = Interpreter::new(&entry);
    let mut tensor = |shape: Vec<usize>, values: &[i32]| {
        Arg::Tensor(interpreter.add_tensor(TensorData::dense(
            DType::I32,
            shape,
            values.iter().map(|&v| f64::from(v)).collect(),
        )))
    };
    let sources = magnitude_kernels::source_words(records, &vec![-1; records.len()]);
    let arguments = [
        tensor(vec![m, k], routes),
        tensor(vec![constants::POLICY_WORDS as usize], policy),
        Arg::Scalar(seismic_lang::reference_math::ReferenceScalar::I32(0)),
        tensor(vec![constants::SOURCE_ROWS as usize, m, k], &sources),
    ];
    let outcome = interpreter.run(&arguments).unwrap();
    assert!(!matches!(outcome.termination(), SourceTermination::Failed(_)), "assign failed");
    let out_input = outcome.inputs().find(|input| input.ordinal() == 3).unwrap();
    let out_tensor = out_input.tensor();
    let out: Vec<i32> = (0..out_tensor.element_count())
        .map(|at| out_tensor.read(at).unwrap() as i32)
        .collect();
    let n = routes.len();
    (out[..n].to_vec(), out[n..2 * n].to_vec())
}

#[test]
fn portable_assign_contended_dma_physics_and_edge_cases() {
    let module = common::kernels();

    // 1. C = 0: contended DMA (180 µs vs 100 µs uncontended), CPU 180 µs, handshake 20 µs, N = 3.
    //    GPU-only = 300 µs; C=1: max(360,200)=360; C=2: max(180,380)=380; C=3: 560 -> C = 0.
    let (recs, sites) = run_assign(
        &module,
        1,
        &[10, 11, 12, 13],
        &[5, -2, -3, -4],
        &policy_words(180, 0, 20, 100, 180, 0),
    );
    assert_eq!(recs, vec![5, -2, -3, -4]);
    assert_eq!(sites, vec![-1, -1, -1, -1]);

    // 2. Exact tie: uncontended=100, contended=160, cpu=250, handshake=50, N=8.
    //    GPU-only=800; C=3 gives max(5*160=800, 50+750=800) = 800 == 800 -> C = 0.
    let (recs, sites) = run_assign(
        &module,
        1,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[-2, -3, -4, -5, -6, -7, -8, -9],
        &policy_words(250, 0, 50, 100, 160, 0),
    );
    assert_eq!(recs, vec![-2, -3, -4, -5, -6, -7, -8, -9]);
    assert_eq!(sites, vec![-1; 8]);

    // 3. Strictly faster split under contention: uncontended=100, contended=120, cpu=150, handshake=10, N=8.
    //    GPU-only=800; C=3: max(5*120=600, 10+450=460)=600 < 800; C=4: max(480, 610)=610 -> C = 3.
    let (recs, sites) = run_assign(
        &module,
        1,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[-2, -3, -4, -5, -6, -7, -8, -9],
        &policy_words(150, 0, 10, 100, 120, 0),
    );
    assert_eq!(recs, vec![-1, -1, -1, -5, -6, -7, -8, -9]);
    assert_eq!(sites, vec![0, 1, 2, -1, -1, -1, -1, -1]);

    // 4. C = N: all misses offloaded to CPU -> 0 GPU misses -> no DMA contention.
    //    uncontended=100, contended=500, cpu=40, handshake=10, N=1 -> C=1 gives 50 < 100.
    let (recs, sites) = run_assign(
        &module,
        1,
        &[10, 11],
        &[5, -2],
        &policy_words(40, 0, 10, 100, 500, 0),
    );
    assert_eq!(recs, vec![5, -1]);
    assert_eq!(sites, vec![-1, 0]);

    // 5. Zero contention (`POLICY_LINK_CONTENDED_NS == POLICY_LINK_NS` or `0`):
    //    produces identical assignment.
    let a = run_assign(&module, 1, &[0, 1, 2, 3], &[-2, -3, -4, -5], &policy_words(150, 0, 10, 100, 100, 0));
    let b = run_assign(&module, 1, &[0, 1, 2, 3], &[-2, -3, -4, -5], &policy_words(150, 0, 10, 100, 0, 0));
    assert_eq!(a, b);
}

#[test]
fn native_resolve_hpp_matches_contended_dma_physics() {
    let Some((device, _tally)) = seismic_native_tests::tallied_device(seismic::BackendName::Metal) else {
        return;
    };
    let entry = magnitude_kernels::routed_select_streamed::native_entry_with(
        magnitude_kernels::routed_select_streamed::Elements {
            NW: seismic::Element::f32(),
            RNW: seismic::Element::f32(),
            RW: seismic::Element::f32(),
            A: seismic::Element::bf16(),
        },
    );
    let Some(implementation) = entry.offered(&device).unwrap() else {
        return;
    };
    let charges = seismic::Charges::tally(&device);
    let (h, e, k, flight) = (64u64, 32u64, 4u64, 4u64);
    let statics = [("H", h), ("E", e), ("K", k), ("ES", 0), ("TM", e)];
    let spec = implementation
        .statics
        .iter()
        .fold(seismic::NativeSpecialization::new(), |s, name| {
            let (_, value) = statics.iter().find(|(c, _)| *c == name.as_str()).unwrap();
            s.with_static(name.clone(), *value)
        });
    let kernel = entry.prepare(&device, &implementation.default_specialization(&spec).unwrap()).unwrap();

    let shared_i32 = |shape: &[u64], values: &[i32]| {
        let t = seismic::Tensor::shared(&charges, seismic::Element::i32(), shape).unwrap();
        for (idx, &v) in values.iter().enumerate() {
            t.store_release((idx * 4) as u64, v as u32).unwrap();
        }
        t
    };
    let f32_tensor = |shape: &[u64], values: &[f32]| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        seismic::Tensor::from_host(&charges, seismic::Element::f32(), shape, &bytes).unwrap()
    };
    let i32_tensor = |shape: &[u64], values: &[i32]| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        seismic::Tensor::from_host(&charges, seismic::Element::i32(), shape, &bytes).unwrap()
    };
    let zeros = |el, shape: &[u64]| seismic::Tensor::zeros(&charges, el, shape).unwrap();
    let f32 = seismic::Element::f32();

    // All experts in host tier (`tier[t] = t`), none in VRAM (`map` unmapped).
    let tier = shared_i32(&[e], &(0..e as i32).collect::<Vec<_>>());
    let map = shared_i32(&[e, constants::MAP_WORDS], &vec![-1; (e * constants::MAP_WORDS) as usize]);
    let mut router_vals = vec![0.0f32; (e * h) as usize];
    for expert in 0..e as usize {
        router_vals[expert * h as usize] = (expert + 1) as f32;
    }
    let router = f32_tensor(&[e, h], &router_vals);
    let mut res_vals = vec![0.0f32; h as usize];
    res_vals[0] = 1.0;
    let residual = f32_tensor(&[1, h], &res_vals);
    let ones = f32_tensor(&[h], &vec![1.0f32; h as usize]);
    let bias = zeros(f32, &[e]);
    let expert_scale = zeros(f32, &[0]);
    let step = i32_tensor(&[constants::STEP_WORDS], &[1, 0, 0, 0]);

    let run_with_policy = |pol: &[i32]| -> (Vec<i32>, Vec<i32>) {
        let mut routes = zeros(seismic::Element::i32(), &[1, k]);
        let mut weights = zeros(f32, &[1, k]);
        let policy = shared_i32(&[constants::POLICY_WORDS], pol);
        let fw = ((constants::FILE_ENTRY(k) + 1) / 2) * 2;
        let mut file = shared_i32(&[flight, fw], &vec![0; (flight * fw) as usize]);
        let mut sources = zeros(seismic::Element::i32(), &[constants::SOURCE_ROWS, 1, k]);
        let mut routing = shared_i32(&[flight, constants::ROUTING_WORDS(k)], &vec![0; (flight * constants::ROUTING_WORDS(k)) as usize]);
        let mut keys = shared_i32(&[flight, constants::VERIFY_ROWS, e], &vec![0; (flight * constants::VERIFY_ROWS * e) as usize]);
        let mut front = shared_i32(&[constants::HINT_ENTRY(k)], &vec![0; constants::HINT_ENTRY(k) as usize]);
        let mut land_grid = zeros(seismic::Element::i32(), &[3]);
        kernel
            .call(
                &charges,
                magnitude_kernels::routed_select_streamed::Args {
                    residual: &residual,
                    norm: &ones,
                    router_norm: &ones,
                    router: &router,
                    bias: &bias,
                    expert_scale: &expert_scale,
                    routes: &mut routes,
                    weights: &mut weights,
                    epsilon: 1e-6,
                    score: 0,
                    normalization: 0,
                    normalization_epsilon: 1e-6,
                    scale: 1.0,
                    map: &map,
                    tier: &tier,
                    policy: &policy,
                    sited: 1,
                    step: &step,
                    file: &mut file,
                    sources: &mut sources,
                    routing: &mut routing,
                    keys: &mut keys,
                    front: &mut front,
                    land_grid: &mut land_grid,
                    land_groups: 1,
                },
            )
            .unwrap();
        let bytes = sources.read_to_host().unwrap();
        let words: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let n = k as usize;
        (words[..n].to_vec(), words[n..2 * n].to_vec())
    };

    // Case 1: Contended DMA (180 µs vs 100 µs uncontended), CPU 180 µs, handshake 20 µs, N = 4 misses.
    // GPU-only = 400 µs; C=1: max(540,200)=540; C=2: max(360,380)=380 < 400 -> C = 2!
    let (_, sites) = run_with_policy(&policy_words(180, 0, 20, 100, 180, 0));
    assert_eq!(sites.iter().filter(|&&s| s >= 0).count(), 2, "expected C=2 on native resolve.hpp: {sites:?}");

    // Case 2: Contended DMA (220 µs vs 100 µs uncontended), CPU 220 µs, handshake 20 µs, N = 4 misses.
    // GPU-only = 400 µs; C=1: max(660,240)=660; C=2: max(440,460)=460; C=3: max(220,680)=680; C=4: 900 -> C = 0!
    let (_, sites) = run_with_policy(&policy_words(220, 0, 20, 100, 220, 0));
    assert_eq!(sites.iter().filter(|&&s| s >= 0).count(), 0, "expected C=0 on native resolve.hpp: {sites:?}");

    // Case 3: C = N = 4 (all misses on CPU -> 0 GPU misses -> no DMA contention):
    // uncontended=100, contended=500, cpu=50, handshake=10 -> C=4 gives 210 < 400 -> C = 4!
    let (_, sites) = run_with_policy(&policy_words(50, 0, 10, 100, 500, 0));
    assert_eq!(sites.iter().filter(|&&s| s >= 0).count(), 4, "expected C=4 on native resolve.hpp: {sites:?}");
}
