//! Embedded standard constructs and kernels. The same authored sources are available
//! to portable inspection and native compilation without a source-tree dependency.
use seismic_lang::{
    program::{compile, SourceFile},
    sir::Program,
    Scope,
};
pub fn sources() -> Vec<SourceFile> {
    let embedded: &[(&str, &str, Scope)] = &[
        ("lib/kernels/gelu.seismic.portable", include_str!("../lib/kernels/gelu.seismic.portable"), Scope::Portable),
        ("lib/kernels/layer_norm.seismic.portable", include_str!("../lib/kernels/layer_norm.seismic.portable"), Scope::Portable),
        ("lib/kernels/linear_bias.seismic.portable", include_str!("../lib/kernels/linear_bias.seismic.portable"), Scope::Portable),
        ("lib/kernels/sampling.seismic.portable", include_str!("../lib/kernels/sampling.seismic.portable"), Scope::Portable),
        ("lib/kernels/gguf_import.seismic.portable",include_str!("../lib/kernels/gguf_import.seismic.portable"),Scope::Portable),
        (
            "lib/kernels/embedding_row.seismic.portable",
            include_str!("../lib/kernels/embedding_row.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/kv_append.seismic.portable",
            include_str!("../lib/kernels/kv_append.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/rotary_prepare.seismic.portable",
            include_str!("../lib/kernels/rotary_prepare.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/constructs/matmul.seismic.cpu",
            include_str!("../lib/constructs/matmul.seismic.cpu"),
            Scope::Backend("cpu".into()),
        ),
        (
            "lib/constructs/matmul.seismic.cuda",
            include_str!("../lib/constructs/matmul.seismic.cuda"),
            Scope::Backend("cuda".into()),
        ),
        (
            "lib/constructs/matmul.seismic.metal",
            include_str!("../lib/constructs/matmul.seismic.metal"),
            Scope::Backend("metal".into()),
        ),
        (
            "lib/constructs/matmul.seismic.portable",
            include_str!("../lib/constructs/matmul.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/argmax.seismic.portable",
            include_str!("../lib/kernels/argmax.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/attention.seismic.portable",
            include_str!("../lib/kernels/attention.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/attention_gate.seismic.portable",
            include_str!("../lib/kernels/attention_gate.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/attention_prepare.seismic.portable",
            include_str!("../lib/kernels/attention_prepare.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/delta_step.seismic.portable",
            include_str!("../lib/kernels/delta_step.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/elementwise.seismic.portable",
            include_str!("../lib/kernels/elementwise.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/embedding.seismic.portable",
            include_str!("../lib/kernels/embedding.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/gate_projection_add.seismic.portable",
            include_str!("../lib/kernels/gate_projection_add.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/gated_norm.seismic.portable",
            include_str!("../lib/kernels/gated_norm.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/gated_projection.seismic.portable",
            include_str!("../lib/kernels/gated_projection.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/linear.seismic.portable",
            include_str!("../lib/kernels/linear.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/logits.seismic.portable",
            include_str!("../lib/kernels/logits.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/norm_gated_projection.seismic.portable",
            include_str!("../lib/kernels/norm_gated_projection.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/norm_logits.seismic.portable",
            include_str!("../lib/kernels/norm_logits.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/norm_projection.seismic.portable",
            include_str!("../lib/kernels/norm_projection.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/projection.seismic.portable",
            include_str!("../lib/kernels/projection.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/projection_add.seismic.portable",
            include_str!("../lib/kernels/projection_add.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/recurrent_prepare.seismic.portable",
            include_str!("../lib/kernels/recurrent_prepare.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/rms_norm.seismic.portable",
            include_str!("../lib/kernels/rms_norm.seismic.portable"),
            Scope::Portable,
        ),
        (
            "lib/kernels/weight_import.seismic.portable",
            include_str!("../lib/kernels/weight_import.seismic.portable"),
            Scope::Portable,
        ),
    ];
    embedded
        .iter()
        .map(|(path, text, scope)| SourceFile {
            path: (*path).into(),
            text: (*text).into(),
            scope: scope.clone(),
        })
        .collect()
}
pub fn program() -> Result<Program, String> {
    compile(&sources(), &["cpu".into(), "cuda".into(), "metal".into()]).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.render())
            .collect::<Vec<_>>()
            .join("\n")
    })
}
#[cfg(test)]
mod tests {
    #[test]
    fn embedded_library_checks() {
        super::program().unwrap();
    }
}
