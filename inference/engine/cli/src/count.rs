//! `magnitude-count`: prompt token counts from a GGUF's own tokenizer and chat
//! templates, without loading the model. It reads the file's metadata only:
//! no worker starts, no device is opened and no weights are read.
//!
//! `magnitude-count chat --model TARGET.gguf`: each stdin line is an OpenAI chat
//! completion request; each stdout line is the prompt tokens the engine's
//! `/v1/count` reports for it, from the same host code (chat template,
//! tokenizer, media expansion). The projector is not opened, as
//! `magnitude-engine --no-projector` serves.
//!
//! `magnitude-count text --model FILE.gguf`: each stdin line is a JSON string;
//! each stdout line is the number of tokens the file's tokenizer encodes it to,
//! special-token text recognized and no sequence-start token added.
use magnitude_artifacts::Package;
use magnitude_chat::artifacts::gguf_byte_bpe;
use magnitude_chat::{ByteBpeTokenizer, SpecialTokens};
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::options::{PackageOptions, ProjectorSelection};
use magnitude_serving::chat::{ChatCompletionRequest, chat_input};
use std::io::{BufRead, Write};
use std::path::PathBuf;

const USAGE: &str = "magnitude-count chat|text --model FILE.gguf  (one JSON request or string per stdin line)";

enum Mode {
    Chat,
    Text,
}

fn parse() -> Result<(Mode, PathBuf), String> {
    let mut args = std::env::args().skip(1);
    let mode = match args.next().as_deref() {
        Some("chat") => Mode::Chat,
        Some("text") => Mode::Text,
        _ => return Err(USAGE.into()),
    };
    match (args.next().as_deref(), args.next(), args.next()) {
        (Some("--model"), Some(model), None) => Ok((mode, PathBuf::from(model))),
        _ => Err(USAGE.into()),
    }
}

/// Counts each stdin line with `count`, one result line per input line.
fn serve(mut count: impl FnMut(&str) -> Result<u64, String>) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for (index, line) in std::io::stdin().lock().lines().enumerate() {
        let line = line.map_err(|error| format!("reading input line {}: {error}", index + 1))?;
        let tokens = count(&line).map_err(|error| format!("input line {}: {error}", index + 1))?;
        writeln!(out, "{tokens}")
            .and_then(|()| out.flush())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn chat(model: PathBuf) -> Result<(), String> {
    let host = HostArtifacts::open(&PackageOptions {
        target: model,
        projector: ProjectorSelection::Disabled,
        draft: None,
    })
    .map_err(|error| error.to_string())?;
    serve(|line| {
        let request: ChatCompletionRequest =
            serde_json::from_str(line).map_err(|error| format!("chat request: {error}"))?;
        let input = chat_input(request).map_err(|error| error.body.message)?;
        magnitude_engine::chat::count_tokens(&host, &input).map_err(|error| error.to_string())
    })
}

fn text(model: PathBuf) -> Result<(), String> {
    let package = Package::open_without_projector(&model).map_err(|error| error.to_string())?;
    let tokenizer = gguf_byte_bpe(package.tokenizer(), model.display().to_string())
        .and_then(ByteBpeTokenizer::new)
        .map_err(|error| error.to_string())?;
    serve(|line| {
        let text: String =
            serde_json::from_str(line).map_err(|error| format!("text: {error}"))?;
        let tokens = tokenizer.encode(&text, SpecialTokens::Recognize)?;
        Ok(tokens.len() as u64)
    })
}

fn main() {
    let result = parse().and_then(|(mode, model)| match mode {
        Mode::Chat => chat(model),
        Mode::Text => text(model),
    });
    if let Err(error) = result {
        eprintln!("magnitude-count: {error}");
        std::process::exit(1);
    }
}
