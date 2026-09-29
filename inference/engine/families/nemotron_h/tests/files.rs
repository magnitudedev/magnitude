//! Inspection of real downloaded GGUF files (every shard of a split file),
//! read through the engine's own header reader. The files live on the test
//! hosts, so the test runs only when `NEMOTRON_H_GGUF` lists them
//! (colon-separated first-shard paths).

use magnitude_artifacts::PackageHeaders;
use magnitude_family_nemotron_h::inspect_components;

#[test]
#[ignore = "needs NEMOTRON_H_GGUF on a host with the catalog files"]
fn downloaded_files_inspect() {
    let paths = std::env::var("NEMOTRON_H_GGUF").expect("NEMOTRON_H_GGUF lists GGUF paths");
    for path in paths.split(':') {
        let headers =
            PackageHeaders::open(path, None).unwrap_or_else(|error| panic!("{path}: {error}"));
        let definition = inspect_components(headers.target(), None, headers.identity())
            .unwrap_or_else(|error| panic!("{path}: {error}"));
        println!(
            "{path}: {} blocks, {} sublayers",
            definition.decoder.blocks.len(),
            definition.decoder.sublayers().count()
        );
    }
}
