use std::io;
use std::process::ExitCode;

fn main() -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();

    match icn_parity_probe::run_jsonl(stdin.lock(), stdout.lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("icn-probe transport failure: {error}");
            ExitCode::FAILURE
        }
    }
}
