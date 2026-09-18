#![cfg(target_os = "macos")]
use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::{execution::Config, msl::emit_with};
#[test]
fn explicit_candidates_are_never_silently_shrunk_or_ignored() {
    let program=compile(&[SourceFile{path:"candidate.seismic.portable".into(),scope:Scope::Portable,text:"fn copy(x: tensor[3] f32, out: tensor[3] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    store(t,out[row:row+1])\n".into()}],&[]).unwrap();
    let lowered = lower(&program, "copy", "metal", &Default::default()).unwrap();
    assert!(emit_with(&lowered, Config::default()).is_ok());
    for candidate in [
        Config {
            per_item: 2,
            ..Default::default()
        },
        Config {
            split: 2,
            ..Default::default()
        },
        Config {
            per_item: 0,
            ..Default::default()
        },
        Config {
            sg_per_tg: 0,
            ..Default::default()
        },
        Config {
            sg_per_tg: 33,
            ..Default::default()
        },
        Config {
            per_item: 3,
            split: 2,
            ..Default::default()
        },
    ] {
        assert!(emit_with(&lowered, candidate).is_err());
    }
}
