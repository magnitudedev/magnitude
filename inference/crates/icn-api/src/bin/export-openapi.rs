use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let output = std::env::args_os().nth(1).map(PathBuf::from);
    let json = serde_json::to_string_pretty(&icn_api::openapi()?)?;
    match output {
        Some(path) => std::fs::write(path, format!("{json}\n"))?,
        None => println!("{json}"),
    }
    Ok(())
}
