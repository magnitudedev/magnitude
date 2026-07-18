use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use icn_api::{AppState, FakeBackend, app};
use icn_core::ModelConfig;
use icn_llamacpp::{LlamaCompletionBackend, LlamaCppEngine};

#[derive(Debug, Parser)]
#[command(
    name = "magnitude-icn",
    version,
    about = "Magnitude inference control node"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        #[arg(long, conflicts_with = "model")]
        fake: bool,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        model_alias: Option<String>,
        #[arg(long, default_value_t = 4096)]
        context_size: u32,
        #[arg(long, default_value_t = 999)]
        gpu_layers: u32,
    },
    Doctor,
    Version {
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Serve {
            bind,
            fake,
            model,
            model_alias,
            context_size,
            gpu_layers,
        } => {
            let state = if fake || model.is_none() {
                AppState::new(FakeBackend::new(
                    model_alias.unwrap_or_else(|| "icn-fake".into()),
                    "Hello from ICN.",
                ))
            } else {
                let path = model.expect("model is present");
                let alias = model_alias.unwrap_or_else(|| {
                    path.file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("local-model")
                        .to_owned()
                });
                let engine = LlamaCppEngine::load(ModelConfig {
                    model_path: path,
                    context_size,
                    gpu_layers,
                })
                .context("failed to load llama.cpp backend")?;
                AppState::new(LlamaCompletionBackend::new(alias, engine))
            };
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .with_context(|| format!("failed to bind {bind}"))?;
            println!("magnitude-icn listening on http://{bind}");
            axum::serve(listener, app(state))
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        Command::Doctor => println!("ICN runtime and native backend loaded successfully"),
        Command::Version { json } => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "version": env!("CARGO_PKG_VERSION") })
                );
            } else {
                println!("{}", env!("CARGO_PKG_VERSION"));
            }
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
