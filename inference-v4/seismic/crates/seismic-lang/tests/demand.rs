use seismic_lang::{
    demand::data_variables,
    ir::{ExprKind, Stmt, StmtKind},
    lower::lower,
    lowered_ir::LoweredIr,
    program::{compile, SourceFile},
    Scope,
};

fn lowered(expression: &str, query: &str) -> LoweredIr {
    let program = compile(&[SourceFile {
        path: "demand.seismic.portable".into(),
        scope: Scope::Portable,
        text: format!("fn evaluate(x:tensor[8] i32,n:i32,out:tensor[1] i32):\n  raw = load(x)\n  s = tile[8] i32\n  for i in owned(s): s[i] = {expression}\n  window = s[n:]\n  alias = window[:]\n  result = tile[1] i32\n  for i in owned(result): result[i] = {query}\n  store(result,out)\n"),
    }], &[]).unwrap();
    lower(&program, "evaluate", "cpu", &Default::default()).unwrap()
}

fn named(function: &LoweredIr, name: &str) -> usize {
    function
        .vars
        .iter()
        .position(|v| v.name == name)
        .unwrap_or_else(|| {
            panic!(
                "missing {name}: {:?}",
                function.vars.iter().map(|v| &v.name).collect::<Vec<_>>()
            )
        })
}

fn produces(body: &[Stmt], variable: usize) -> bool {
    body.iter().any(|s| match &s.kind {
        StmtKind::Owned { tile, body, .. } => {
            matches!(tile.kind, ExprKind::Var(v) if v == variable) || produces(body, variable)
        }
        StmtKind::Parallel { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::LoadLoop { body, .. }
        | StmtKind::Lanes { body, .. } => produces(body, variable),
        StmtKind::If { then, els, .. } => produces(then, variable) || produces(els, variable),
        StmtKind::Reduction(r) => r.bodies().any(|b| produces(b, variable)),
        _ => false,
    })
}

#[test]
fn geometry_survives_without_computed_elements_or_transitive_input_data() {
    let function = lowered("raw[i] + i", "extent(alias,0)");
    let data = data_variables(&function.body);
    for name in ["raw", "s", "window", "alias"] {
        let variable = named(&function, name);
        assert!(!data.contains(&variable), "{name} retains element storage");
    }
    assert!(!produces(&function.body, named(&function, "s")));
    assert!(data.contains(&named(&function, "result")));
}

#[test]
fn geometry_endpoint_element_reads_retain_their_producer() {
    let function = lowered("raw[i] + i", "extent(s[s[0]:],0)");
    let data = data_variables(&function.body);
    assert!(data.contains(&named(&function, "s")));
    assert!(data.contains(&named(&function, "raw")));
    assert!(produces(&function.body, named(&function, "s")));
}

#[test]
fn unused_elements_do_not_authorize_erasing_numerical_failures() {
    let function = lowered("raw[i] / n", "extent(alias,0)");
    assert!(produces(&function.body, named(&function, "s")));
    assert!(data_variables(&function.body).contains(&named(&function, "s")));
}
