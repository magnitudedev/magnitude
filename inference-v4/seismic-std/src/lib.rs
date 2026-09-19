//! Embedded standard constructs and kernels. The same authored sources are available
//! to portable inspection and native compilation without a source-tree dependency.
use seismic_lang::{
    program::{compile, SourceFile},
    sir::Program,
};

pub fn sources() -> Vec<SourceFile> {
    let embedded: &[(&str, &str)] = &[
        (
            "lib/kernels/gelu.seismic",
            include_str!("../lib/kernels/gelu.seismic"),
        ),
        (
            "lib/kernels/layer_norm.seismic",
            include_str!("../lib/kernels/layer_norm.seismic"),
        ),
        (
            "lib/kernels/linear_bias.seismic",
            include_str!("../lib/kernels/linear_bias.seismic"),
        ),
        (
            "lib/kernels/sampling.seismic",
            include_str!("../lib/kernels/sampling.seismic"),
        ),
        (
            "lib/kernels/gguf_import.seismic",
            include_str!("../lib/kernels/gguf_import.seismic"),
        ),
        (
            "lib/kernels/embedding_row.seismic",
            include_str!("../lib/kernels/embedding_row.seismic"),
        ),
        (
            "lib/kernels/kv_append.seismic",
            include_str!("../lib/kernels/kv_append.seismic"),
        ),
        (
            "lib/kernels/rotary_prepare.seismic",
            include_str!("../lib/kernels/rotary_prepare.seismic"),
        ),
        (
            "lib/constructs/matmul-cpu.seismic",
            include_str!("../lib/constructs/matmul-cpu.seismic"),
        ),
        (
            "lib/constructs/matmul-cuda.seismic",
            include_str!("../lib/constructs/matmul-cuda.seismic"),
        ),
        (
            "lib/constructs/matmul-metal.seismic",
            include_str!("../lib/constructs/matmul-metal.seismic"),
        ),
        (
            "lib/constructs/matmul.seismic",
            include_str!("../lib/constructs/matmul.seismic"),
        ),
        (
            "lib/kernels/argmax.seismic",
            include_str!("../lib/kernels/argmax.seismic"),
        ),
        (
            "lib/kernels/attention.seismic",
            include_str!("../lib/kernels/attention.seismic"),
        ),
        (
            "lib/kernels/attention_gate.seismic",
            include_str!("../lib/kernels/attention_gate.seismic"),
        ),
        (
            "lib/kernels/attention_prepare.seismic",
            include_str!("../lib/kernels/attention_prepare.seismic"),
        ),
        (
            "lib/kernels/delta_step.seismic",
            include_str!("../lib/kernels/delta_step.seismic"),
        ),
        (
            "lib/kernels/elementwise.seismic",
            include_str!("../lib/kernels/elementwise.seismic"),
        ),
        (
            "lib/kernels/embedding.seismic",
            include_str!("../lib/kernels/embedding.seismic"),
        ),
        (
            "lib/kernels/gate_projection_add.seismic",
            include_str!("../lib/kernels/gate_projection_add.seismic"),
        ),
        (
            "lib/kernels/gated_norm.seismic",
            include_str!("../lib/kernels/gated_norm.seismic"),
        ),
        (
            "lib/kernels/gated_projection.seismic",
            include_str!("../lib/kernels/gated_projection.seismic"),
        ),
        (
            "lib/kernels/linear.seismic",
            include_str!("../lib/kernels/linear.seismic"),
        ),
        (
            "lib/kernels/logits.seismic",
            include_str!("../lib/kernels/logits.seismic"),
        ),
        (
            "lib/kernels/norm_gated_projection.seismic",
            include_str!("../lib/kernels/norm_gated_projection.seismic"),
        ),
        (
            "lib/kernels/norm_logits.seismic",
            include_str!("../lib/kernels/norm_logits.seismic"),
        ),
        (
            "lib/kernels/norm_projection.seismic",
            include_str!("../lib/kernels/norm_projection.seismic"),
        ),
        (
            "lib/kernels/projection.seismic",
            include_str!("../lib/kernels/projection.seismic"),
        ),
        (
            "lib/kernels/projection_add.seismic",
            include_str!("../lib/kernels/projection_add.seismic"),
        ),
        (
            "lib/kernels/recurrent_prepare.seismic",
            include_str!("../lib/kernels/recurrent_prepare.seismic"),
        ),
        (
            "lib/kernels/rms_norm.seismic",
            include_str!("../lib/kernels/rms_norm.seismic"),
        ),
        (
            "lib/kernels/weight_import.seismic",
            include_str!("../lib/kernels/weight_import.seismic"),
        ),
    ];
    embedded
        .iter()
        .map(|(path, text)| SourceFile {
            path: (*path).into(),
            text: (*text).into(),
        })
        .collect()
}

pub fn program() -> Result<Program, String> {
    compile(&sources()).map_err(|errors| {
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
