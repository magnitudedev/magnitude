//! The official LFM3 1.7B and LFM3-VL 2B GGUF inventories at the catalog's
//! pinned revision: the hybrid decoder of both and the LFM3-VL tower.

use magnitude_artifacts::{ArtifactIdentity, PackageIdentity};
use magnitude_family_common::headers;
use magnitude_family_contracts::{
    AttentionGate, CellReduction, GateFunction, GateGranularity, HistoryDomain, KeyValue,
    ModelFamily, Operator, OutputForm, Rotary, VisionActivation, VisionNorm, VisionOperator,
    VisionResize, VisionStem,
};
use magnitude_family_lfm2::{inspect_components, Lfm2Family};

const VARIANTS: [&str; 3] = ["q4-qat", "q6", "q8"];

/// Layers 2, 6, …, 22 attend over the whole context, layers 4, 8, …, 20
/// over the last 256 rows; the rest convolve.
const FULL: [usize; 6] = [2, 6, 10, 14, 18, 22];
const WINDOW: [usize; 5] = [4, 8, 12, 16, 20];

fn identity(projector: bool) -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([0; 32]),
        projector: projector.then_some(ArtifactIdentity([1; 32])),
    }
}

/// The pairs of a rotary table that turn.
fn rotated_pairs(rotary: &Rotary) -> usize {
    match rotary {
        Rotary::Table { pairs, .. } => pairs.iter().filter(|pair| pair.frequency != 0.0).count(),
        other => panic!("unexpected rotary {other:?}"),
    }
}

#[test]
fn released_quantizations_bind_the_hybrid_decoder() {
    for (model, context) in [("lfm3-1.7b", 262_144), ("lfm3-vl-2b", 128_000)] {
        for variant in VARIANTS {
            let directory = headers::target(&format!("{model}:gguf:{variant}"));
            let definition = inspect_components(&directory, None, identity(false)).unwrap();
            definition.validate().unwrap();
            let decoder = &definition.decoder;
            assert_eq!(decoder.hidden, 2048);
            assert_eq!(decoder.vocabulary, 128_000);
            assert_eq!(decoder.context_limit, context);
            assert_eq!(decoder.blocks.len(), 24);
            assert_eq!(decoder.entry.embedding, decoder.exit.output);
            assert!(definition.vision.is_none());
            for (index, block) in decoder.blocks.iter().enumerate() {
                for sublayer in &block.sublayers {
                    assert!(matches!(sublayer.output, OutputForm::PostNorm(_)));
                }
                match &block.sublayers[0].op {
                    Operator::Attention(a) => {
                        assert_eq!((a.heads, a.kv_heads, a.width), (16, 4, 128));
                        let AttentionGate::Separate {
                            weight,
                            function: GateFunction::Sigmoid,
                            granularity: GateGranularity::Head,
                        } = &a.gate
                        else {
                            panic!("layer {index}: an LFM3 head gate");
                        };
                        assert_eq!(weight.shape, [16, 2048]);
                        let KeyValue::Owned { domain, .. } = &a.key_value else {
                            panic!("layer {index}: an owned key/value history");
                        };
                        if WINDOW.contains(&index) {
                            assert_eq!(*domain, HistoryDomain::Window { tokens: 256 });
                            assert_eq!(rotated_pairs(&a.rotary), 64);
                        } else {
                            assert!(FULL.contains(&index));
                            assert_eq!(*domain, HistoryDomain::Token);
                            assert_eq!(rotated_pairs(&a.rotary), 32);
                            assert!(matches!(
                                &a.rotary,
                                Rotary::Table {
                                    divisors: Some(_),
                                    ..
                                }
                            ));
                        }
                    }
                    Operator::ShortConv(c) => {
                        assert!(!FULL.contains(&index) && !WINDOW.contains(&index));
                        assert_eq!((c.channels, c.width), (2048, 7));
                    }
                    other => panic!("unexpected mixer {}", other.name()),
                }
            }
        }
    }
}

#[test]
fn released_projector_binds_the_lfm3_vl_tower() {
    let projector = headers::projector("lfm3-vl-2b:gguf:q4-qat");
    for variant in VARIANTS {
        let target = headers::target(&format!("lfm3-vl-2b:gguf:{variant}"));
        let definition = inspect_components(&target, Some(&projector), identity(true)).unwrap();
        definition.validate().unwrap();
        let vision = definition.vision.as_ref().expect("an LFM3-VL tower");
        assert_eq!((vision.hidden, vision.output_hidden), (1024, 2048));
        assert_eq!(
            vision.preprocessing.resize,
            VisionResize::PixelBounds {
                min_pixels: 65_536,
                max_pixels: 16_777_216,
            }
        );
        assert_eq!(
            (vision.preprocessing.patch, vision.preprocessing.merge),
            (16, 2)
        );
        let VisionStem::Patch {
            frames,
            bias,
            positions,
            norm,
            registers,
        } = &vision.stem
        else {
            panic!("a patch stem");
        };
        assert_eq!(frames.len(), 2);
        assert!(bias.is_none() && positions.is_none() && registers.is_none());
        assert!(matches!(norm, Some(VisionNorm::Rms { .. })));
        assert_eq!(vision.blocks.len(), 14);
        for (index, block) in vision.blocks.iter().enumerate() {
            assert_eq!(block.feedforward.activation, VisionActivation::Silu);
            match &block.operator {
                VisionOperator::Attention(attention) => {
                    assert!(index % 2 == 0 && (2..=12).contains(&index));
                    assert_eq!(
                        (attention.heads, attention.kv_heads, attention.width),
                        (16, 8, 64)
                    );
                    assert!(attention.query_norm.is_some() && attention.key_norm.is_some());
                }
                VisionOperator::GatedConvolution(convolution) => {
                    assert!(index % 2 == 1 || index == 0 || index == 13);
                    assert_eq!(convolution.side(), 3);
                }
            }
        }
        let merger = &vision.merger;
        assert_eq!(merger.reduction, CellReduction::Concatenate);
        assert!(matches!(
            merger.norm,
            Some(VisionNorm::Layer { epsilon, .. }) if epsilon == 1e-6
        ));
        assert_eq!(merger.stages.len(), 2);
        assert_eq!(
            Lfm2Family.media_placeholder(&definition),
            Some("<|vision_start|><|image|><|vision_end|>")
        );
    }
}

#[test]
fn lfm2_refuses_a_projector() {
    let target = headers::target("lfm2.5-2.6b:gguf:q4");
    let projector = headers::projector("lfm3-vl-2b:gguf:q4-qat");
    assert!(inspect_components(&target, Some(&projector), identity(true)).is_err());
}
