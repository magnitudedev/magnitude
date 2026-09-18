use seismic_accounting::{
    execution_model::*, schedule::*, selection::Budget, workload::DerivationLimits,
};
use seismic_lang::{
    Scope,
    composition::Ownership,
    ir::StmtKind,
    lower::{self, Options},
    lowered_ir::{Alternative, DecisionKind},
    program::{SourceFile, compile},
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{
    Buffer, Candidate, Device,
    plan::{Bindings, PlanCompiler, Settings},
    tuner::{Form, Hardware},
};
use std::collections::{BTreeSet, HashMap};
fn source() -> seismic_lang::program::Program {
    compile(
        &[SourceFile {
            path: "composition.seismic.portable".into(),
            scope: Scope::Portable,
            text: r#"
fn project[N](x: tensor[N] f32, tmp: tensor[N] f16):
  for row in parallel:
    a = load(x[row:row+1])
    y = tile[1] f32
    for i in owned(y): y[i] = a[i] * 1.0003
    store(y,tmp[row:row+1])
fn scale[N](x: tensor[N] f16, tmp: tensor[N] f32):
  for row in parallel:
    a = load(x[row:row+1])
    y = tile[1] f32
    for i in owned(y): y[i] = f32(a[i]) * 1.25
    store(y,tmp[row:row+1])
fn publish[N](x: tensor[N] f32, out: tensor[N] f32):
  for row in parallel:
    a = load(x[row:row+1])
    store(a,out[row:row+1])
fn chain(x: tensor[2] f32, tmp: tensor[2] f16, work: tensor[2] f32, out: tensor[2] f32):
  project(x,tmp)
  scale(tmp,work)
  publish(work,out)
"#
            .into(),
        }],
        &[],
    )
    .unwrap()
}
fn ownership() -> Ownership {
    Ownership {
        intermediates: BTreeSet::from(["tmp".into(), "work".into()]),
    }
}
fn options() -> Options {
    Options {
        ownership: ownership(),
        ..Default::default()
    }
}
fn assignment(d: &seismic_lang::lowered_ir::Decision, fuse: bool) -> Alternative {
    if fuse {
        if let Some(a) = d
            .alternatives
            .iter()
            .filter(|a| matches!(a, Alternative::ParallelFusion { .. }))
            .max_by_key(|a| match a {
                Alternative::ParallelFusion {
                    shared_axes,
                    refine_consumer,
                } => (*shared_axes, *refine_consumer),
                _ => unreachable!(),
            })
        {
            return a;
        }
        if d.alternatives.contains(&Alternative::Fuse) {
            return Alternative::Fuse;
        }
        if d.alternatives.contains(&Alternative::RetainLocal) {
            return Alternative::RetainLocal;
        }
    }
    d.alternatives.get(0).unwrap()
}
fn selected(fuse: bool) -> seismic_lang::lowered_ir::LoweredIr {
    lower::lower_selected(
        &source(),
        "chain",
        "cpu",
        &HashMap::new(),
        &HashMap::new(),
        &options(),
        &mut |d| Ok(assignment(d, fuse)),
    )
    .unwrap()
}
#[test]
fn composed_native_preserves_private_publication_rounding_and_alias_requirements() {
    let separate = selected(false);
    let fused = selected(true);
    assert_eq!(
        separate
            .body
            .iter()
            .filter(|s| matches!(s.kind, StmtKind::Parallel { .. }))
            .count(),
        3
    );
    assert_eq!(
        fused
            .body
            .iter()
            .filter(|s| matches!(s.kind, StmtKind::Parallel { .. }))
            .count(),
        2
    );
    assert!(fused.decisions.iter().any(|d| matches!(
        d.domain.kind,
        DecisionKind::Intermediate { .. }
    ) && d.selected == Alternative::RetainLocal));
    let device = Device::cpu();
    let mut outputs = Vec::new();
    for lowered in [separate, fused] {
        let mut kernel = device
            .compile(
                &lowered,
                Candidate::Cpu {
                    loads: LoadStrategy::Materialize,
                },
            )
            .unwrap();
        let input = device
            .buffer_from(
                &[1.0001f32, 1000.125]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let tmp = device.buffer(4).unwrap();
        let work = device.buffer(8).unwrap();
        let out = device.buffer(8).unwrap();
        kernel
            .execute(
                &[input.clone(), tmp.clone(), work.clone(), out.clone()],
                &[],
            )
            .unwrap();
        let mut bytes = [0u8; 8];
        out.read(&mut bytes).unwrap();
        outputs.push(bytes);
        assert!(
            kernel
                .execute(&[input, work.view(0..4).unwrap(), work, out], &[])
                .unwrap_err()
                .contains("private intermediate")
        );
    }
    assert_eq!(outputs[0], outputs[1]);
    let expected = [1.0001f32, 1000.125].map(|x| {
        seismic_lang::numeric::f16_to_f32(seismic_lang::numeric::f16_bits(x * 1.0003)) * 1.25
    });
    assert_eq!(
        outputs[0].to_vec(),
        expected
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}
struct Bound(HashMap<String, Buffer>);
impl Bindings for Bound {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        assert!(plane.is_empty());
        self.0.get(root)
    }
    fn scalar(&self, _: &str) -> Option<f64> {
        None
    }
}
#[test]
fn production_plan_submits_enclosing_source_and_caches_only_applicable_native_artifacts() {
    let program = source();
    let shapes = HashMap::new();
    let elements = HashMap::new();
    let opts = options();
    let mut patterns = Vec::new();
    for attempt in lower::alternatives::Space::new(lower::alternatives::Specialization {
        program: &program,
        entry: "chain",
        backend: "cpu",
        shapes: &shapes,
        elements: &elements,
        options: &opts,
    }) {
        let lowered = attempt.result.unwrap();
        for loads in [
            LoadStrategy::Materialize,
            LoadStrategy::BorrowProvenReadOnly,
        ] {
            for primitive in requirements(&seismic_cpu::prepare(&lowered, loads).unwrap()).unwrap()
            {
                let p = primitive.signature();
                if !patterns.contains(&p) {
                    patterns.push(p)
                }
            }
        }
    }
    let hardware = ScalarHardware {
        identity: "hypothetical composition fixture".into(),
        scope: seismic_accounting::execution_model::Scope::HypotheticalDirectScalarV1,
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![Resource {
            name: "issue".into(),
            capacity: 1,
            unit: CapacityUnit::Slots,
        }],
        timings: patterns
            .into_iter()
            .map(|primitive| PrimitiveTiming {
                primitive,
                latency: 1,
                services: vec![Reservation {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: 1,
                }],
            })
            .collect(),
    };
    let device = Device::cpu();
    let settings = Settings {
        hardware: Hardware::Cpu(hardware),
        form: Form::CpuScalar,
        derivation_limits: DerivationLimits {
            instructions: 100000,
            operations: 100000,
        },
        search: Budget {
            nodes: 20000,
            schedule_assignments: 100000,
        },
    };
    let mut compiler = PlanCompiler::new(&device, &program, settings);
    let mut plan = compiler
        .compile_entry("chain", &shapes, &elements, &ownership())
        .unwrap();
    assert_eq!(plan.kernel_count(), 0);
    assert_eq!(plan.step_count(), 1);
    let out = device.buffer(8).unwrap();
    let input = [2f32, 4.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let bindings = Bound(HashMap::from([
        ("x".into(), device.buffer_from(&input).unwrap()),
        ("tmp".into(), device.buffer(4).unwrap()),
        ("work".into(), device.buffer(8).unwrap()),
        ("out".into(), out.clone()),
    ]));
    plan.execute(&bindings).unwrap();
    assert_eq!(plan.kernel_count(), 1);
    plan.execute(&bindings).unwrap();
    assert_eq!(plan.kernel_count(), 1);
    let mut bytes = [0u8; 8];
    out.read(&mut bytes).unwrap();
    assert_eq!(
        bytes.to_vec(),
        [2.5f32, 5.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[test]
fn compatible_projections_share_their_stream_without_changing_accumulation_order() {
    let program=compile(&[SourceFile{path:"streams.seismic.portable".into(),scope:Scope::Portable,text:r#"
fn add[S](a:tile[S] f32,b:tile[S] f32,y:tile[S] f32):
  for i in owned(y): y[i] = a[i] + b[i]
fn product[S](s:tile[S] f32,a:tile[S] f32,b:tile[S] f32,y:tile[S] f32):
  for i in owned(y): y[i] = fma(a[i], b[i], s[i])
construct contraction[M,N,K](a:tile[M,K] f32,b:tile[N,K] f32,out:tile[M,N] f32):
  for i,j in owned(out):
    left = a[i:i+1,:]
    right = b[j:j+1,:]
    state = tile[1] f32
    zero = tile[1] f32
    for t in owned(state): state[t] = out[i,j]
    for t in owned(zero): zero[t] = 0.0
    reduce((left,right),1,add,into=(state,),step=product,identity=(zero,),ordered=true)
    out[i,j] = state[0]

fn projection[M,N,K](x: tensor[M,K] f32, w: tensor[N,K] f32, out: tensor[M,N] f16):
  for row,col in parallel:
    acc = tile[1,1] f32
    for i,j in owned(acc): acc[i,j] = 0.0
    xt = load(x[row:row+1])
    wt = load(w[col:col+1])
    contraction(xt,wt,acc)
    store(acc,out[row:row+1,col:col+1])
fn combine[M,N](p: tensor[M,N] f16,q: tensor[M,N] f16,work: tensor[M,N] f32):
  for row in parallel:
    a = load(p[row])
    b = load(q[row])
    y = tile[N] f32
    for i in owned(y): y[i] = f32(a[i]) * f32(b[i])
    store(y,work[row])
fn publish[M,N](work: tensor[M,N] f32,out: tensor[M,N] f32):
  for row in parallel:
    t = load(work[row])
    store(t,out[row])
fn chain(x: tensor[1,4] f32,w: tensor[2,4] f32,v: tensor[2,4] f32,p: tensor[1,2] f16,q: tensor[1,2] f16,work: tensor[1,2] f32,out: tensor[1,2] f32):
  projection(x,w,p)
  projection(x,v,q)
  combine(p,q,work)
  publish(work,out)
"#.into()},SourceFile{path:"streams.seismic.cpu".into(),scope:Scope::Backend("cpu".into()),text:"lower contraction: portable\n".into()}],&[]).unwrap();
    let options = Options {
        piece: Some(2),
        ownership: Ownership {
            intermediates: BTreeSet::from(["p".into(), "q".into(), "work".into()]),
        },
    };
    let lower = |fuse| {
        lower::lower_selected(
            &program,
            "chain",
            "cpu",
            &HashMap::new(),
            &HashMap::new(),
            &options,
            &mut |d| Ok(if matches!(d.kind,DecisionKind::Producer{..}) && d.alternatives.contains(&Alternative::Recompute) {Alternative::Recompute} else {assignment(d, fuse)}),
        )
        .unwrap()
    };
    let separate = lower(false);
    let fused = lower(true);
    assert!(
        fused.decisions.iter().any(
            |d| matches!(d.domain.kind, DecisionKind::StreamFusion { .. })
                && d.selected == Alternative::Fuse
        ),
        "{:?}",
        fused.decisions
    );
    fn streams(body: &[seismic_lang::ir::Stmt]) -> Vec<usize> {
        body.iter()
            .flat_map(|s| match &s.kind {
                StmtKind::Range { body, .. } => {
                    let loads=body.iter().filter(|s|matches!(&s.kind,StmtKind::Assign{value:seismic_lang::ir::Expr{kind:seismic_lang::ir::ExprKind::Builtin{name:seismic_lang::ir::Builtin::Load,..},..},..})).count();
                    let mut n=if loads>0{vec![loads]}else{vec![]};n.extend(streams(body));n
                }
                StmtKind::Parallel { body, .. } | StmtKind::Owned { body, .. } => streams(body),
                _ => Vec::new(),
            })
            .collect()
    }
    assert_eq!(streams(&separate.body), vec![2, 2]);
    assert_eq!(streams(&fused.body), vec![3]);
    let device = Device::cpu();
    let mut outputs = Vec::new();
    for lowered in [separate, fused] {
        let mut kernel = device
            .compile(
                &lowered,
                Candidate::Cpu {
                    loads: LoadStrategy::Materialize,
                },
            )
            .unwrap();
        let floats = |xs: &[f32]| {
            device
                .buffer_from(&xs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let out = device.buffer(8).unwrap();
        let buffers = vec![
            floats(&[1., 2., 3., 4.]),
            floats(&[1., 2., 1., 2., 3., 4., 3., 4.]),
            floats(&[0.5, 1., 0.5, 1., 1., 2., 1., 2.]),
            device.buffer(4).unwrap(),
            device.buffer(4).unwrap(),
            device.buffer(8).unwrap(),
            out.clone(),
        ];
        kernel.execute(&buffers, &[]).unwrap();
        let mut bytes = [0u8; 8];
        out.read(&mut bytes).unwrap();
        outputs.push(bytes);
    }
    assert_eq!(outputs[0], outputs[1]);
}
