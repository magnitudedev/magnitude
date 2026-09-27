use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const SOURCES: &[&str] = &[
    "native/src/abi.cpp",
    "native/src/output-stream.cpp",
    "native/src/schema-validation.cpp",
    "native/src/templates-support.cpp",
    "native/source/common/chat-auto-parser-generator.cpp",
    "native/source/common/chat-auto-parser-helpers.cpp",
    "native/source/common/chat-diff-analyzer.cpp",
    "native/source/common/chat-peg-parser.cpp",
    "native/source/common/chat.cpp",
    "native/source/common/jinja/caps.cpp",
    "native/source/common/jinja/lexer.cpp",
    "native/source/common/jinja/parser.cpp",
    "native/source/common/jinja/runtime.cpp",
    "native/source/common/jinja/string.cpp",
    "native/source/common/jinja/value.cpp",
    "native/source/common/json-schema-to-grammar.cpp",
    "native/source/common/json-schema.cpp",
    "native/source/common/json.cpp",
    "native/source/common/parsers/cohere2moe.cpp",
    "native/source/common/parsers/deepseek.cpp",
    "native/source/common/parsers/functionary-v3-2.cpp",
    "native/source/common/parsers/gemma4.cpp",
    "native/source/common/parsers/gigachat-v3.cpp",
    "native/source/common/parsers/gpt-oss.cpp",
    "native/source/common/parsers/kimi-k2.cpp",
    "native/source/common/parsers/kimi-k3.cpp",
    "native/source/common/parsers/lfm2.cpp",
    "native/source/common/parsers/minicpm5.cpp",
    "native/source/common/parsers/minimax-m3.cpp",
    "native/source/common/parsers/ministral3.cpp",
    "native/source/common/parsers/muse-glimmer.cpp",
    "native/source/common/parsers/parsers.cpp",
    "native/source/common/parsers/qwen3-coder.cpp",
    "native/source/common/peg-parser.cpp",
    "native/source/common/trie.cpp",
    "native/source/common/unicode.cpp",
];

fn files_below(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("read vendored template directory") {
        let path = entry.expect("read vendored template entry").path();
        if path.is_dir() {
            files_below(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn build_id(root: &Path, compiler: &cc::Tool) -> String {
    let mut inputs = vec![root.join("build.rs")];
    files_below(&root.join("native/include"), &mut inputs);
    files_below(&root.join("native/provenance"), &mut inputs);
    files_below(&root.join("native/source"), &mut inputs);
    files_below(&root.join("native/src"), &mut inputs);
    inputs.sort();

    let mut digest = Sha256::new();
    digest.update(env::var("TARGET").expect("Cargo TARGET"));
    digest.update(env::var("PROFILE").expect("Cargo PROFILE"));
    digest.update(compiler.path().as_os_str().as_encoded_bytes());
    for argument in compiler.args() {
        digest.update(argument.as_encoded_bytes());
    }
    for (key, value) in compiler.env() {
        digest.update(key.as_encoded_bytes());
        digest.update(value.as_encoded_bytes());
    }
    let mut version = compiler.to_command();
    version.arg(if compiler.is_like_msvc() {
        "/Bv"
    } else {
        "--version"
    });
    if let Ok(output) = version.output() {
        digest.update(output.stdout);
        digest.update(output.stderr);
    }
    for path in inputs {
        println!("cargo:rerun-if-changed={}", path.display());
        digest.update(
            path.strip_prefix(root)
                .expect("owned build input")
                .as_os_str()
                .as_encoded_bytes(),
        );
        digest.update(fs::read(path).expect("read owned build input"));
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest directory"));
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .pic(true)
        .warnings(false)
        .include(root.join("native/include"))
        .include(root.join("native/src"))
        .include(root.join("native/source/common"))
        .include(root.join("native/source/vendor"))
        .flag_if_supported("-fvisibility=hidden")
        .flag_if_supported("-fvisibility-inlines-hidden");
    for source in SOURCES {
        build.file(root.join(source));
    }

    let compiler = build.get_compiler();
    let definition = format!("\"{}\"", build_id(&root, &compiler));
    build.define("TEMPLATES_BUILD_ID", definition.as_str());
    build.compile("templates");
}
