//! Coverage of the construct x context matrix (design A10 §2.5).
use crate::common::corpus_path;
use seismic_corpus::matrix::{CellMark, Matrix, MatrixCell};
use seismic_corpus::scenario::{load_all, ScenarioClass};

fn matrix() -> Matrix {
    Matrix::load(&corpus_path("scenarios/matrix/cells.txt"))
}

#[test]
fn every_required_cell_has_a_scenario() {
    let scenarios = load_all(&corpus_path("scenarios"));
    let tagged: Vec<&MatrixCell> = scenarios.iter().flat_map(|s| &s.cells).collect();
    let uncovered: Vec<String> = matrix()
        .required()
        .filter(|(cell, _)| !tagged.contains(&cell))
        .map(|(cell, mark)| format!("{cell} ({mark:?})"))
        .collect();
    assert!(
        uncovered.is_empty(),
        "{} required cell(s) have no scenario:\n{}",
        uncovered.len(),
        uncovered.join("\n")
    );
}

#[test]
fn cell_tags_name_known_cells() {
    let matrix = matrix();
    let mut failures = Vec::new();
    for scenario in load_all(&corpus_path("scenarios")) {
        for cell in &scenario.cells {
            let executes = matches!(scenario.class, ScenarioClass::Executes(_));
            match matrix.mark(cell) {
                None => failures.push(format!("{}: unknown row in `{cell}`", scenario.name)),
                Some(CellMark::Unrequired) => {
                    failures.push(format!("{}: `{cell}` is a `.` cell", scenario.name))
                }
                Some(CellMark::Executes) if !executes => failures.push(format!(
                    "{}: `{cell}` requires an executing scenario",
                    scenario.name
                )),
                Some(CellMark::Rejected) if executes => failures.push(format!(
                    "{}: `{cell}` requires a rejected scenario",
                    scenario.name
                )),
                Some(_) => {}
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
