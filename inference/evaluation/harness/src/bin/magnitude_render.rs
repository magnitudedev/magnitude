//! `magnitude-render`: host-only template rendering without development engine
//! features, exposing the same `render` subcommand as `magnitude-harness`.

use clap::{Parser, Subcommand};
use magnitude_harness::{RunArgs, run_render};

#[derive(Parser)]
#[command(
    name = "magnitude-render",
    about = "Render benchmark templates through the engine's serving chat path"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Render every template's lanes, sized to its context, to the exact
    /// tokens the engine serves.
    Render {
        #[command(flatten)]
        run: RunArgs,
    },
}

fn main() {
    let Command::Render { run } = Cli::parse().command;
    if let Err(error) = run_render(&run) {
        eprintln!("magnitude-render: {error}");
        std::process::exit(1);
    }
}
