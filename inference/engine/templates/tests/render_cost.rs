//! A loop's cost per item is bounded by its body, whatever the scope holds and however
//! many names the body looks up. Templates that scan the conversation once per message
//! (Gemma 4) render a number of items quadratic in the conversation's length, so any
//! cost per item beyond the body's own multiplies with it.
use magnitude_templates::{SpecialTokens, Template};
use serde_json::{json, Map};
use std::time::{Duration, Instant};

const ITEMS: usize = 3000;
const ROUNDS: usize = 5;

fn fastest_render(source: &str) -> Duration {
    let template = Template::new(source, &SpecialTokens::new()).unwrap();
    let mut context = Map::new();
    context.insert("messages".into(), json!([]));
    context.insert("x".into(), json!(1));
    (0..ROUNDS)
        .map(|_| {
            let start = Instant::now();
            template.render(&context, 946684800).unwrap();
            start.elapsed()
        })
        .min()
        .unwrap()
}

fn looped(prelude: &str, body: &str) -> String {
    format!("{prelude}{{%- for i in range({ITEMS}) -%}}{body}{{%- endfor -%}}")
}

#[test]
fn a_loop_item_costs_the_same_however_many_variables_the_scope_holds() {
    let many: String = (0..200)
        .map(|index| format!("{{%- set v{index} = {index} -%}}"))
        .collect();
    let few = fastest_render(&looped("{%- set v0 = 0 -%}", ""));
    let crowded = fastest_render(&looped(&many, ""));
    assert!(
        crowded < few * 4,
        "a loop under 200 variables took {crowded:?}, against {few:?} under one"
    );
}

#[test]
fn looking_up_a_variable_costs_about_as_much_as_a_literal() {
    let literal = fastest_render(&looped("", &"{{- 1 -}}".repeat(32)));
    let lookup = fastest_render(&looped("", &"{{- x -}}".repeat(32)));
    assert!(
        lookup < literal * 2,
        "32 lookups per item took {lookup:?}, against {literal:?} for 32 literals"
    );
}
