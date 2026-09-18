//! Read-only bounded inspection of the real embedding lowering family.
//! No native implementation is compiled, selected, or executed.
use seismic_lang::lower::{alternatives::{self, Expansion, Specialization}, Options};
use seismic_lang::types::{DType, Elem};
use std::collections::{BTreeMap, HashMap, VecDeque};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 { return Err("usage: qwen_frontier ROWS NODE_LIMIT".into()); }
    let rows: i64 = args[0].parse()?;
    let limit: usize = args[1].parse()?;
    if rows <= 0 || limit == 0 { return Err("rows and node limit must be positive".into()); }
    let program = seismic_engine::models::qwen35::program::program()?;
    let shapes = HashMap::from([("M".into(), rows), ("V".into(), 248320), ("D".into(), 2560)]);
    let elements = HashMap::from([("EW".into(), Elem::Repr("q4g64".into())), ("A".into(), Elem::Dtype(DType::BF16))]);
    let options = Options::default();
    let request = || Specialization { program: &program, entry: "qwen_embedding_rows", backend: "metal", shapes: &shapes, elements: &elements, options: &options };
    let start = std::time::Instant::now();
    let root = alternatives::expand(request(), &[])?;
    // Each retained decision tracks its unvisited suffix, including symbolic
    // domains too large to enumerate. Breadth-first visitation is diagnostic,
    // not an optimization policy or an executable choice.
    let mut pending = VecDeque::from([(Vec::<usize>::new(), root, 0usize)]);
    let mut decisions = BTreeMap::<String, usize>::new();
    let mut failures = BTreeMap::<String, Vec<Vec<usize>>>::new();
    let mut visited = 1usize;
    let mut lowered = 0usize;
    while let Some((path, node, index)) = pending.pop_front() {
        let decision = match &node {
            Expansion::Choice(d) => d,
            Expansion::RetainedChoice(c) => c.decision(),
            Expansion::Lowered { .. } => { lowered += 1; continue; }
        };
        if index == 0 { *decisions.entry(format!("{:?}", decision.kind)).or_default() += 1; }
        if visited == limit { pending.push_front((path, node, index)); break; }
        let count = decision.alternatives.len();
        let mut child_path = path.clone();
        child_path.push(index);
        let result = match &node {
            Expansion::RetainedChoice(c) => c.refine(index),
            Expansion::Choice(_) => alternatives::expand(request(), &child_path),
            _ => unreachable!(),
        };
        visited += 1;
        if index + 1 < count { pending.push_back((path, node, index + 1)); }
        match result {
            Ok(child) => pending.push_back((child_path, child, 0)),
            Err(error) => { failures.entry(error).or_default().push(child_path); }
        }
    }
    let unvisited_alternatives: u128 = pending.iter().map(|(_, node, index)| match node {
        Expansion::Choice(d) => (d.alternatives.len() - index) as u128,
        Expansion::RetainedChoice(c) => (c.decision().alternatives.len() - index) as u128,
        Expansion::Lowered { .. } => 0,
    }).sum();
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "entry": "qwen_embedding_rows", "rows": rows, "nodes_visited": visited,
        "elapsed_seconds": start.elapsed().as_secs_f64(), "lowered_leaves_observed": lowered,
        "unvisited_alternatives": unvisited_alternatives.to_string(), "pending_regions": pending.len(),
        "decisions": decisions, "unresolved_failures": failures,
        "status": "frontend inspection only; no execution selection or native compilation"
    }))?);
    Ok(())
}
