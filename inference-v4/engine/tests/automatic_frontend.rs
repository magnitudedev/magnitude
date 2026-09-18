//! Full-size frontend admission, before hardware accounting or native compilation.
//! An unresolved choice is retained: this test never chooses an implementation.
use seismic_lang::{
    lower::{
        alternatives::{expand, Expansion, Specialization},
        Options,
    },
    types::{DType, Elem},
};
use std::collections::HashMap;

#[test]
fn full_size_entries_reach_automatic_frontier() {
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    let cases: &[(&str, &[(&str, i64)], &[&str], &[&str])] = &[
        (
            "qwen_vision_merger",
            &[("G", 4), ("H", 1152), ("D", 2560)],
            &["UW", "DW"],
            &["NW", "NB", "UB", "DB"],
        ),
        (
            "qwen_vision_merger_up",
            &[("G", 4), ("H", 1152)],
            &["UW"],
            &["NW", "NB", "UB"],
        ),
        (
            "qwen_vision_block",
            &[("H", 16), ("P", 18), ("F", 4608)],
            &["QW", "PW", "UW", "DW"],
            &["NW", "NB", "QB", "PB", "UB", "DB"],
        ),
        ("qwen_vision_attention", &[("H", 16), ("P", 18)], &[], &[]),
        (
            "qwen_vision_stem",
            &[("C", 3), ("T", 2), ("P", 16), ("H", 1152), ("L", 2304)],
            &["W"],
            &["B"],
        ),
        (
            "qwen_embedding_rows",
            &[("V", 248320), ("D", 2560)],
            &["EW"],
            &[],
        ),
        (
            "qwen_readout_rows",
            &[("V", 248320), ("D", 2560)],
            &["OW"],
            &["NW"],
        ),
        (
            "qwen_readout_selected",
            &[("V", 248320), ("D", 2560), ("S", 7)],
            &["OW"],
            &["NW"],
        ),
        (
            "qwen_dense_suffix",
            &[("H", 2560), ("F", 9216)],
            &["GW", "UW", "DW"],
            &["NW"],
        ),
        (
            "qwen_recurrent_sequence",
            &[("H", 2560), ("NK", 16), ("GV", 2), ("W", 128), ("C", 4)],
            &["QW", "GW", "AW", "BW", "OW"],
            &["NW", "RN"],
        ),
        (
            "qwen_attention_sequence",
            &[
                ("D", 2560),
                ("T", 256),
                ("R", 3),
                ("H", 16),
                ("KV", 4),
                ("P", 32),
                ("S", 192),
                ("SH", 11),
                ("SW", 10),
            ],
            &["QW", "KW", "VW", "OW"],
            &["NW"],
        ),
    ];
    for &(entry, dimensions, packed, dense) in cases {
        let mut elements = HashMap::from([("A".into(), Elem::Dtype(DType::BF16))]);
        elements.extend(
            packed
                .iter()
                .map(|name| (name.to_string(), Elem::Repr("q4g64".into()))),
        );
        elements.extend(
            dense
                .iter()
                .map(|name| (name.to_string(), Elem::Dtype(DType::BF16))),
        );
        for rows in [1, 32, 128] {
            let mut shapes: HashMap<_, _> = dimensions
                .iter()
                .map(|(name, n)| (name.to_string(), *n))
                .collect();
            shapes.insert("M".into(), rows);
            let options = Options::default();
            let frontier = expand(
                Specialization {
                    program: &program,
                    entry,
                    backend: "metal",
                    shapes: &shapes,
                    elements: &elements,
                    options: &options,
                },
                &[],
            )
            .unwrap_or_else(|error| panic!("{entry}, {rows} rows: {error}"));
            match frontier {
                Expansion::Choice(decision) => {
                    assert!(!decision.alternatives.is_empty());
                    eprintln!(
                        "{entry}, {rows} rows: {:?}, {} alternatives",
                        decision.kind,
                        decision.alternatives.len()
                    );
                }
                Expansion::RetainedChoice(choice) => {
                    assert!(!choice.decision().alternatives.is_empty());
                    eprintln!(
                        "{entry}, {rows} rows: {:?}, {} alternatives",
                        choice.decision().kind,
                        choice.decision().alternatives.len()
                    );
                }
                Expansion::Lowered { function, consumed } => {
                    assert_eq!(consumed, 0);
                    function.ownership.validate(&function).unwrap();
                    eprintln!("{entry}, {rows} rows: lowering has no unresolved frontend choices");
                }
            }
        }
    }
}
