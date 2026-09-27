//! Replay the production tuning search against a directory of survey records
//! (`forward_bench --tuning-survey DIR`) and print the report (tuning spec
//! §E2/§E3). Development only.
//!
//! `cargo run --release -p magnitude-model-executor --features tuning-survey
//! --example tuning_replay -- DIR`

fn main() -> std::process::ExitCode {
    let Some(directory) = std::env::args().nth(1) else {
        eprintln!("usage: tuning_replay SURVEY_DIR");
        return std::process::ExitCode::FAILURE;
    };
    match magnitude_model_executor::tuning_survey::replay_report(std::path::Path::new(&directory)) {
        Ok(report) => {
            print!("{report}");
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("tuning_replay: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
