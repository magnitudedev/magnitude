//! The corpus test target: one generated public-API test per scenario,
//! backend and selection (`build.rs`), and the corpus-wide suites. Module
//! paths are the filters of the interim checkpoints
//! (`cargo test -p seismic-corpus areas__A2`, `... library`, `... hygiene`).
mod bundle;
mod common;
mod history;
mod hygiene;
mod library;
mod matrix;
mod resources;
mod source_tracking;
mod workflows;

include!(concat!(env!("OUT_DIR"), "/generated_public.rs"));
