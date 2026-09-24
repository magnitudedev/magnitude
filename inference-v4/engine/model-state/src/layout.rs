use crate::{ComponentDescriptor, ComponentSpec, KvCodec, LayerRef};
use magnitude_model_contracts::{
    ActivationDType, DecoderGeometry, MixerGeometry, RecurrentGeometry,
};
use seismic::DType;

/// Device-free state allocation plan derived from the family-neutral model
/// geometry. It is shared by every execution path; only the physical stage
/// that binds the resulting planes differs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelStateLayout {
    pub target_history: Vec<ComponentDescriptor>,
    pub target_recurrent: Vec<ComponentSpec>,
    pub head_history: Vec<ComponentDescriptor>,
}

impl ModelStateLayout {
    /// `tape_rows` is the most speculative rows a recurrent bank records after
    /// its published state (the planned draft width; 0 without drafting).
    pub fn derive(
        geometry: &DecoderGeometry,
        head_depth: usize,
        target_codec: KvCodec,
        tape_rows: usize,
    ) -> Result<Self, String> {
        geometry.validate().map_err(|error| error.to_string())?;
        let activation = activation_dtype(geometry.activation_dtype);
        let mut target_history = Vec::new();
        let mut target_recurrent = Vec::new();
        let mut head_geometry = None;

        for (index, block) in geometry.blocks.iter().enumerate() {
            match &block.mixer {
                MixerGeometry::Attention(attention) => {
                    let kv_heads = host(attention.kv_heads, "attention KV heads")?;
                    let width = host(attention.width, "attention head width")?;
                    let component = ComponentDescriptor::new(
                        LayerRef::Target(layer(index)?),
                        target_codec.spec(activation, width, width),
                        kv_heads,
                    )
                    .map_err(|error| error.to_string())?;
                    target_history.push(component);
                    head_geometry = Some((kv_heads, width));
                }
                MixerGeometry::Recurrent(recurrent) => {
                    target_recurrent.extend(recurrent_components(recurrent, activation, tape_rows)?);
                }
            }
        }

        let mut head_history = Vec::with_capacity(head_depth);
        if head_depth != 0 {
            let (kv_heads, width) = head_geometry
                .ok_or("draft head requires at least one target attention geometry")?;
            for index in 0..head_depth {
                // Head history stays dense whatever the target codec: the
                // draft head's attention reads dense planes only.
                head_history.push(
                    ComponentDescriptor::new(
                        LayerRef::Head(layer(index)?),
                        KvCodec::Dense.spec(activation, width, width),
                        kv_heads,
                    )
                    .map_err(|error| error.to_string())?,
                );
            }
        }

        Ok(Self {
            target_history,
            target_recurrent,
            head_history,
        })
    }
}

fn activation_dtype(dtype: ActivationDType) -> DType {
    match dtype {
        ActivationDType::F16 => DType::F16,
        ActivationDType::BF16 => DType::BF16,
    }
}

/// A recurrent layer's bank components, in the order the state entries take
/// them (`recurrent.seismic`): the window (C - 1 raw rows before the state,
/// then the raw rows of the tape), the delta state, and the tape of the rows
/// after the state (innovations, keys, decays). A bank holds at least one tape
/// row, so the tape is a real component; plain advances never write it.
fn recurrent_components(
    recurrent: &RecurrentGeometry,
    activation: DType,
    tape_rows: usize,
) -> Result<[ComponentSpec; 3], String> {
    let convolution = host(
        recurrent
            .convolution_width
            .checked_sub(1)
            .ok_or("recurrent convolution width underflow")?,
        "recurrent convolution history",
    )?;
    let channels = host(
        recurrent.channels().map_err(|error| error.to_string())?,
        "recurrent channels",
    )?;
    let key_heads = host(recurrent.key_heads, "recurrent key heads")?;
    let value_heads = host(recurrent.value_heads, "recurrent value heads")?;
    let width = host(recurrent.width, "recurrent head width")?;
    let tape = tape_rows.max(1);
    Ok([
        ComponentSpec {
            shape: vec![convolution + tape, channels],
            dtype: activation,
        },
        ComponentSpec {
            shape: vec![value_heads, width, width],
            dtype: DType::F32,
        },
        ComponentSpec {
            shape: vec![tape, (value_heads + key_heads) * width + value_heads],
            dtype: DType::F32,
        },
    ])
}

fn host(value: u64, field: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{field} exceeds the host domain"))
}

fn layer(index: usize) -> Result<u32, String> {
    u32::try_from(index).map_err(|_| "model layer index exceeds u32".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_model_contracts::{
        AttentionGeometry, BlockGeometry, FeedForwardGeometry, RecurrentHeadMapping,
        RotarySemantics,
    };

    fn attention() -> MixerGeometry {
        MixerGeometry::Attention(AttentionGeometry {
            heads: 4,
            kv_heads: 2,
            width: 8,
            rotary: RotarySemantics::Interleaved {
                width: 8,
                base: 10_000.0,
                sections: vec![2, 1, 1],
                axis_pattern: vec![0, 1, 2],
            },
        })
    }

    #[test]
    fn derives_target_recurrent_and_dense_head_layouts() {
        let geometry = DecoderGeometry {
            activation_dtype: ActivationDType::BF16,
            hidden: 32,
            vocabulary: 64,
            context_limit: 128,
            epsilon: 1e-6,
            blocks: vec![
                BlockGeometry {
                    mixer: attention(),
                    feedforward: FeedForwardGeometry::Dense { intermediate: 64 },
                },
                BlockGeometry {
                    mixer: MixerGeometry::Recurrent(RecurrentGeometry {
                        convolution_width: 4,
                        key_heads: 2,
                        value_heads: 4,
                        width: 8,
                        head_mapping: RecurrentHeadMapping::Grouped,
                    }),
                    feedforward: FeedForwardGeometry::Dense { intermediate: 64 },
                },
            ],
        };
        let layout = ModelStateLayout::derive(&geometry, 2, KvCodec::AffineK8V4, 3).unwrap();
        assert_eq!(layout.target_history.len(), 1);
        assert_eq!(layout.target_history[0].layer, LayerRef::Target(0));
        // One codec group per (row, kv head) vector: 2 heads of width 8, each
        // code row padded to 16 bytes.
        assert_eq!(layout.target_history[0].codec.key_width, 8);
        assert_eq!(layout.target_history[0].heads, 2);
        assert_eq!(layout.target_history[0].planes()[0].row_extents, [2, 4]);
        assert_eq!(layout.target_history[0].planes()[1].row_extents, [2, 2]);
        // Window: 3 history rows + 3 tape rows of 64 channels; delta; tape rows
        // of u [4, 8] | k [2, 8] | d [4].
        assert_eq!(layout.target_recurrent.len(), 3);
        assert_eq!(layout.target_recurrent[0].shape, [6, 64]);
        assert_eq!(layout.target_recurrent[0].dtype, DType::BF16);
        assert_eq!(layout.target_recurrent[1].shape, [4, 8, 8]);
        assert_eq!(layout.target_recurrent[1].dtype, DType::F32);
        assert_eq!(layout.target_recurrent[2].shape, [3, 52]);
        assert_eq!(layout.target_recurrent[2].dtype, DType::F32);
        assert_eq!(layout.head_history.len(), 2);
        assert_eq!(layout.head_history[1].layer, LayerRef::Head(1));
        assert_eq!(layout.head_history[1].codec.key_width, 8);
        assert_eq!(layout.head_history[1].planes()[0].row_extents, [2, 8]);
        assert!(matches!(
            layout.head_history[1].codec.key,
            crate::Codec::Dense { dtype: DType::BF16 }
        ));
    }

    #[test]
    fn dense_target_history_preserves_head_and_width_axes() {
        let geometry = DecoderGeometry {
            activation_dtype: ActivationDType::F16,
            hidden: 32,
            vocabulary: 64,
            context_limit: 128,
            epsilon: 1e-6,
            blocks: vec![BlockGeometry {
                mixer: attention(),
                feedforward: FeedForwardGeometry::Dense { intermediate: 64 },
            }],
        };
        let layout = ModelStateLayout::derive(&geometry, 0, KvCodec::Dense, 0).unwrap();
        assert_eq!(layout.target_history[0].planes()[0].row_extents, [2, 8]);
        assert_eq!(layout.target_history[0].planes()[1].row_extents, [2, 8]);
    }
}
