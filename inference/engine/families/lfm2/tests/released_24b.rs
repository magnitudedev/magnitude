//! The four official LFM2.5 24B-A2B GGUF inventories at the catalog's pinned revision.

use magnitude_artifacts::{ArtifactIdentity, PackageIdentity};
use magnitude_family_common::headers;
use magnitude_family_contracts::{ExpertSelection, Operator, RouteNormalization, ScoreFunction};
use magnitude_family_lfm2::inspect_components;

#[test]
fn released_quantizations_bind_the_complete_24b_decoder() {
    for variant in ["q4", "q5", "q6", "q8"] {
        let directory = headers::target(&format!("lfm2.5-24b-a2b:gguf:{variant}"));
        let definition = inspect_components(
            &directory,
            None,
            PackageIdentity {
                target: ArtifactIdentity([0; 32]),
                projector: None,
            },
        )
        .unwrap();
        definition.validate().unwrap();
        let decoder = &definition.decoder;
        assert_eq!(decoder.hidden, 2048);
        assert_eq!(decoder.vocabulary, 128_000);
        assert_eq!(decoder.context_limit, 131_072);
        assert_eq!(decoder.blocks.len(), 40);
        assert_eq!(decoder.entry.embedding, decoder.exit.output);
        // Neither the original checkpoint nor the GGUFs contain a prediction head.
        assert!(definition.head.is_none() && definition.draft.is_none());
        assert!(definition.vision.is_none());
        let attention = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38];
        for (index, block) in decoder.blocks.iter().enumerate() {
            assert_eq!(block.sublayers.len(), 2);
            match &block.sublayers[0].op {
                Operator::Attention(a) => {
                    assert!(attention.contains(&index));
                    assert_eq!((a.heads, a.kv_heads, a.width), (32, 8, 64));
                }
                Operator::ShortConv(c) => {
                    assert!(!attention.contains(&index));
                    assert_eq!((c.channels, c.width), (2048, 3));
                }
                other => panic!("unexpected mixer {}", other.name()),
            }
            match &block.sublayers[1].op {
                Operator::DenseFfn(f) => {
                    assert!(index < 2);
                    assert_eq!(f.intermediate, 11_776);
                }
                Operator::RoutedFfn(f) => {
                    assert!(index >= 2);
                    assert_eq!((f.experts, f.selected, f.intermediate), (64, 4, 1536));
                    assert_eq!(f.router.score, ScoreFunction::Sigmoid);
                    assert_eq!(
                        f.router.normalization,
                        RouteNormalization::SumPlusEpsilon(1e-6)
                    );
                    assert!(matches!(
                        f.router.selection,
                        ExpertSelection::TopK { bias: Some(_) }
                    ));
                    assert!(f.shared.is_none());
                }
                other => panic!("unexpected feed-forward {}", other.name()),
            }
        }
    }
}
