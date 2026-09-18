use seismic_lang::{
    Scope,
    lower::{
        Options,
        alternatives::{Expansion, Specialization, expand},
        lower_selected,
    },
    lowered_ir::{Alternative, Decision, DecisionKind},
    program::{SourceFile, compile},
};
use std::collections::HashMap;

#[test]
fn composition_refinement_preserves_full_path_ir_and_shares_its_prepared_boundary() {
    let p = compile(
        &[
            SourceFile {
                path: "refinement.seismic.portable".into(),
                scope: Scope::Portable,
                text: r#"
construct multiply[N](left:tile[N] W,right:tile[N] W,out:tile[N] f32):
  for i in owned(out): out[i] = left[i] * right[i]
fn decode(x:tensor[64] q4g64,out:tensor[64] f32):
  values = load(x)
  duplicate = load(x)
  result = tile[64] f32
  multiply(values,duplicate,result)
  store(result,out)
"#
                .into(),
            },
            SourceFile {
                path: "refinement.seismic.cpu".into(),
                scope: Scope::Backend("cpu".into()),
                text: "lower multiply: portable\n".into(),
            },
        ],
        &["cpu".into()],
    )
    .unwrap();
    let shapes = HashMap::new();
    let elements = HashMap::new();
    let options = Options::default();
    let request = || Specialization {
        program: &p,
        entry: "decode",
        backend: "cpu",
        shapes: &shapes,
        elements: &elements,
        options: &options,
    };
    for representation in [
        Alternative::Encoded,
        Alternative::Decoded,
        Alternative::DecodedPackets,
    ] {
        for width in [1, 7, 64] {
            let choose = |d: &Decision| match d.kind {
                DecisionKind::Representation { .. } => d
                    .alternatives
                    .iter()
                    .position(|a| a == representation)
                    .unwrap(),
                DecisionKind::PacketDecode { .. } => d
                    .alternatives
                    .iter()
                    .position(|a| a == Alternative::PacketWidth(width))
                    .unwrap(),
                _ => 0,
            };
            let mut path = Vec::new();
            let mut expansion = expand(request(), &path).unwrap();
            let mut retained = None;
            let mut retained_body = false;
            let function = loop {
                expansion = match expansion {
                    Expansion::Choice(d) => {
                        path.push(choose(&d));
                        expand(request(), &path).unwrap()
                    }
                    Expansion::RetainedChoice(owner) => {
                        retained_body |=
                            matches!(owner.decision().kind, DecisionKind::Construct { .. });
                        let base = owner.prepared().decisions.len();
                        if let Some((previous_base, previous)) = retained {
                            if base == previous_base {
                                assert!(std::ptr::eq(owner.prepared(), previous));
                            } else {
                                assert!(base > previous_base);
                            }
                        }
                        retained = Some((base, owner.prepared() as *const _));
                        // Structural owner identity agrees with rebuilding the same
                        // domain from the original program and global ordinals.
                        let Expansion::RetainedChoice(rebuilt) = expand(request(), &path).unwrap()
                        else {
                            panic!("lost retained stage")
                        };
                        assert_eq!(owner, rebuilt);
                        assert!(owner.refine(owner.decision().alternatives.len()).is_err());
                        let index = choose(owner.decision());
                        path.push(index);
                        owner.refine(index).unwrap()
                    }
                    Expansion::Lowered { function, consumed } => {
                        assert_eq!(consumed, path.len());
                        break function;
                    }
                };
            };
            assert!(retained.is_some());
            assert!(
                retained_body,
                "backend bodies must reuse the partitioned computation"
            );
            let mut consumed = 0;
            let replay = lower_selected(
                &p,
                "decode",
                "cpu",
                &shapes,
                &elements,
                &options,
                &mut |d| {
                    let selected = d.alternatives.get(path[consumed]).unwrap();
                    consumed += 1;
                    Ok(selected)
                },
            )
            .unwrap();
            assert_eq!(consumed, path.len());
            assert_eq!(function, replay);
        }
    }
}
