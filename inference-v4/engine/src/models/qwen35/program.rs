//! Authored numerical compositions following V3's publication boundaries.
use seismic_lang::{
    program::{compile, SourceFile},
    sir::Program,
    Scope,
};
/// Contains no hardware policy. Candidate selection belongs to the compiler and
/// accounting system; these equations describe the model's numerical stages.
pub fn program() -> Result<Program, String> {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "qwen35/vision.seismic.portable".into(),
        text: include_str!("../../../lib/vision.seismic.portable").into(),
        scope: Scope::Portable,
    });
    sources.push(SourceFile {
        path: "qwen35/dense_suffix.seismic.portable".into(),
        text: include_str!("../../../lib/dense_suffix.seismic.portable").into(),
        scope: Scope::Portable,
    });
    sources.push(SourceFile {
        path: "qwen35/recurrent_step.seismic.portable".into(),
        text: include_str!("../../../lib/recurrent_step.seismic.portable").into(),
        scope: Scope::Portable,
    });
    sources.push(SourceFile {
        path: "qwen35/attention_step.seismic.portable".into(),
        text: include_str!("../../../lib/attention_step.seismic.portable").into(),
        scope: Scope::Portable,
    });
    sources.push(SourceFile {
        path: "qwen35/token_io.seismic.portable".into(),
        text: include_str!("../../../lib/token_io.seismic.portable").into(),
        scope: Scope::Portable,
    });
    sources.push(SourceFile {
        path: "qwen35/routed_suffix.seismic.portable".into(),
        text: include_str!("../../../lib/routed_suffix.seismic.portable").into(),
        scope: Scope::Portable,
    });
    sources.push(SourceFile {
        path: "qwen35/sequence.seismic.portable".into(),
        text: include_str!("../../../lib/sequence.seismic.portable").into(),
        scope: Scope::Portable,
    });
    compile(&sources, &["cpu".into(), "cuda".into(), "metal".into()]).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.render())
            .collect::<Vec<_>>()
            .join("\n")
    })
}
