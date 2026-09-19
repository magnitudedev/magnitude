//! Authored numerical compositions following V3's publication boundaries.
use seismic_lang::{
    program::{compile, SourceFile},
    sir::Program,
};
/// Contains no hardware policy. Candidate selection belongs to the compiler and
/// accounting system; these equations describe the model's numerical stages.
pub fn program() -> Result<Program, String> {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "qwen35/vision.seismic".into(),
        text: include_str!("../../../lib/vision.seismic").into(),
    });
    sources.push(SourceFile {
        path: "qwen35/dense_suffix.seismic".into(),
        text: include_str!("../../../lib/dense_suffix.seismic").into(),
    });
    sources.push(SourceFile {
        path: "qwen35/recurrent_step.seismic".into(),
        text: include_str!("../../../lib/recurrent_step.seismic").into(),
    });
    sources.push(SourceFile {
        path: "qwen35/attention_step.seismic".into(),
        text: include_str!("../../../lib/attention_step.seismic").into(),
    });
    sources.push(SourceFile {
        path: "qwen35/token_io.seismic".into(),
        text: include_str!("../../../lib/token_io.seismic").into(),
    });
    sources.push(SourceFile {
        path: "qwen35/routed_suffix.seismic".into(),
        text: include_str!("../../../lib/routed_suffix.seismic").into(),
    });
    sources.push(SourceFile {
        path: "qwen35/sequence.seismic".into(),
        text: include_str!("../../../lib/sequence.seismic").into(),
    });
    compile(&sources).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.render())
            .collect::<Vec<_>>()
            .join("\n")
    })
}
