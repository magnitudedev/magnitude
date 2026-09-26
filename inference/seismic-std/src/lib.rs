//! The embedded standard library sources. `seismic-build` links them into a
//! consumer module (`Build::std(true)`); tooling checks them through
//! `seismic_lang::checked::check_source`. Nothing here is checked or compiled
//! at crate build time: the crate is source text only.
use seismic_lang::checked::{SourceFile, SourceSet};

const EMBEDDED: &[(&str, &str)] = &[
    (
        "std/constructs/matmul.seismic",
        include_str!("../lib/constructs/matmul.seismic"),
    ),
    (
        "std/constructs/matmul-metal.seismic",
        include_str!("../lib/constructs/matmul-metal.seismic"),
    ),
    (
        "std/kernels/argmax.seismic",
        include_str!("../lib/kernels/argmax.seismic"),
    ),
    (
        "std/kernels/attention.seismic",
        include_str!("../lib/kernels/attention.seismic"),
    ),
    (
        "std/kernels/attention_gate.seismic",
        include_str!("../lib/kernels/attention_gate.seismic"),
    ),
    (
        "std/kernels/delta_step.seismic",
        include_str!("../lib/kernels/delta_step.seismic"),
    ),
    (
        "std/kernels/elementwise.seismic",
        include_str!("../lib/kernels/elementwise.seismic"),
    ),
    (
        "std/kernels/embedding.seismic",
        include_str!("../lib/kernels/embedding.seismic"),
    ),
    (
        "std/kernels/embedding_row.seismic",
        include_str!("../lib/kernels/embedding_row.seismic"),
    ),
    (
        "std/kernels/gate_projection_add.seismic",
        include_str!("../lib/kernels/gate_projection_add.seismic"),
    ),
    (
        "std/kernels/gated_norm.seismic",
        include_str!("../lib/kernels/gated_norm.seismic"),
    ),
    (
        "std/kernels/gated_projection.seismic",
        include_str!("../lib/kernels/gated_projection.seismic"),
    ),
    (
        "std/kernels/gelu.seismic",
        include_str!("../lib/kernels/gelu.seismic"),
    ),
    (
        "std/kernels/kv_append.seismic",
        include_str!("../lib/kernels/kv_append.seismic"),
    ),
    (
        "std/kernels/layer_norm.seismic",
        include_str!("../lib/kernels/layer_norm.seismic"),
    ),
    (
        "std/kernels/linear.seismic",
        include_str!("../lib/kernels/linear.seismic"),
    ),
    (
        "std/kernels/linear_bias.seismic",
        include_str!("../lib/kernels/linear_bias.seismic"),
    ),
    (
        "std/kernels/logits.seismic",
        include_str!("../lib/kernels/logits.seismic"),
    ),
    (
        "std/kernels/norm_gated_projection.seismic",
        include_str!("../lib/kernels/norm_gated_projection.seismic"),
    ),
    (
        "std/kernels/norm_logits.seismic",
        include_str!("../lib/kernels/norm_logits.seismic"),
    ),
    (
        "std/kernels/norm_projection.seismic",
        include_str!("../lib/kernels/norm_projection.seismic"),
    ),
    (
        "std/kernels/projection.seismic",
        include_str!("../lib/kernels/projection.seismic"),
    ),
    (
        "std/kernels/projection_add.seismic",
        include_str!("../lib/kernels/projection_add.seismic"),
    ),
    (
        "std/kernels/recurrent_prepare.seismic",
        include_str!("../lib/kernels/recurrent_prepare.seismic"),
    ),
    (
        "std/kernels/rms_norm.seismic",
        include_str!("../lib/kernels/rms_norm.seismic"),
    ),
    (
        "std/kernels/rotary_prepare.seismic",
        include_str!("../lib/kernels/rotary_prepare.seismic"),
    ),
    (
        "std/kernels/sampling.seismic",
        include_str!("../lib/kernels/sampling.seismic"),
    ),
    (
        "std/kernels/weight_import.seismic",
        include_str!("../lib/kernels/weight_import.seismic"),
    ),
];

/// Every standard library source file, as one closed source set. Paths are
/// diagnostic labels.
pub fn sources() -> SourceSet {
    SourceSet::new(
        EMBEDDED
            .iter()
            .map(|(path, text)| SourceFile {
                path: (*path).to_owned(),
                text: (*text).to_owned(),
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn embedded_library_checks() {
        seismic_lang::checked::check_source(super::sources()).unwrap();
    }
}
