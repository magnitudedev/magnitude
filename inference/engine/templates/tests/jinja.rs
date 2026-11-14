//! Jinja features renders equal to `transformers`' chat-template environment
//! (Jinja2 with loop controls and its own `tojson`), one feature per test.
//! The expected text is what that environment renders for each source.
use magnitude_templates::Template;

fn render(source: &str) -> String {
    Template::new(source, &Default::default())
        .unwrap()
        .render(&Default::default(), 0)
        .unwrap()
}

#[test]
fn loop_controls_leave_the_innermost_loop() {
    assert_eq!(
        render(
            "{% for x in [1, 2, 3, 4] %}{% if x == 2 %}{% continue %}{% endif %}\
             {% if x == 4 %}{% break %}{% endif %}{{ x }}{% endfor %}"
        ),
        "13"
    );
    assert_eq!(
        render(
            "{% for x in [1, 2] %}{% for y in [1, 2, 3] %}{% if y > x %}{% break %}{% endif %}\
             {{ x }}{{ y }} {% endfor %}{% endfor %}"
        ),
        "11 21 22 "
    );
}

/// A loop's `else` runs only when the loop has no iteration, never after
/// iterations that all `continue` or one that `break`s.
#[test]
fn a_loops_else_runs_only_without_an_iteration() {
    assert_eq!(render("{% for x in [] %}{{ x }}{% else %}none{% endfor %}"), "none");
    assert_eq!(render("{% for x in [1, 2] if x > 2 %}{{ x }}{% else %}none{% endfor %}"), "none");
    assert_eq!(render("{% for x in [1, 2] %}{% continue %}{% else %}none{% endfor %}"), "");
    assert_eq!(render("{% for x in [1, 2] %}{% break %}{% else %}none{% endfor %}"), "");
}

/// An assignment in a loop body lasts until its iteration ends: the next
/// iteration and the code after the loop see the value from before the loop.
#[test]
fn each_loop_iteration_has_its_own_scope() {
    assert_eq!(
        render(
            "{% set kept = 'outer' %}{% for x in [1, 2, 3] %}\
             {% if x == 1 %}{% set fresh = 'one' %}{% set kept = 'one' %}{% endif %}\
             {{ x }}:{{ fresh is defined }},{{ kept }} {% endfor %}{{ fresh is defined }},{{ kept }}"
        ),
        "1:True,one 2:False,outer 3:False,outer False,outer"
    );
    assert_eq!(
        render(
            "{% for x in [1, 2] %}{% for y in [1, 2] %}{% if y == 1 %}{% set inner = x %}{% endif %}\
             {{ inner is defined }} {% endfor %}{% set outer = x %}{% endfor %}"
        ),
        "True False True False "
    );
}

#[test]
fn namespaces_carry_assignments_out_of_loops_and_macros() {
    assert_eq!(
        render(
            "{% set ns = namespace(n=0, xs=[]) %}{% set y = 1 %}\
             {% for x in [1, 2, 3] %}{% set ns.n = ns.n + x %}{% set ns.xs = ns.xs + [x] %}\
             {% set y = x %}{% endfor %}{{ ns.n }} {{ ns.xs | join(',') }} {{ y }}"
        ),
        "6 1,2,3 1"
    );
    assert_eq!(
        render(
            "{% macro first_even(xs) %}{% set ns = namespace(found=none) %}\
             {% for x in xs %}{% if x % 2 == 0 %}{% set ns.found = x %}{% break %}{% endif %}\
             {% endfor %}{{ ns.found }}{% endmacro %}{{ first_even([3, 4, 6]) }}"
        ),
        "4"
    );
}

#[test]
fn capitalize_uppercases_the_first_character_and_lowercases_the_rest() {
    assert_eq!(
        render("{{ 'max' | capitalize }} {{ 'hIGH' | capitalize }} {{ '' | capitalize }}."),
        "Max High ."
    );
}

#[test]
fn tojson_escapes_non_ascii_only_when_asked() {
    assert_eq!(
        render(
            "{{ {'a': 'é', 'b': [1, '東京']} | tojson(ensure_ascii=False) }} \
             {{ 'é' | tojson(ensure_ascii=True) }} {{ 'é' | tojson }}"
        ),
        r#"{"a": "é", "b": [1, "東京"]} "\u00e9" "é""#
    );
}

/// `unique` keeps each item whose key is new, in order: by default the key of
/// a string is its lower case.
#[test]
fn unique_keeps_first_occurrences_by_case_folded_or_exact_key() {
    assert_eq!(
        render(
            "{{ ['b', 'A', 'a', 'B', 'c', 1, 1.0] | unique | join(',') }} \
             {{ ['b', 'A', 'a', 'B', 'b'] | unique(case_sensitive=true) | join(',') }} \
             {{ [{'k': 'x', 'v': 1}, {'k': 'X', 'v': 2}, {'k': 'y', 'v': 3}] \
                | unique(attribute='k') | map(attribute='v') | join(',') }} \
             {{ [{'k': 'x', 'v': 1}, {'k': 'X', 'v': 2}] \
                | unique(case_sensitive=true, attribute='k') | map(attribute='v') | join(',') }} \
             {{ ['a', 'b'] | unique | list | length }}"
        ),
        "b,A,c,1 b,A,a,B 1,3 1,2 2"
    );
}

/// Sorting is stable in either direction and, by default, compares strings
/// by their lower case.
#[test]
fn sort_is_stable_and_case_folded_unless_case_sensitive() {
    assert_eq!(
        render(
            "{{ ['b', 'A', 'a', 'B'] | sort | join(',') }} \
             {{ ['b', 'A', 'a', 'B'] | sort(case_sensitive=true) | join(',') }} \
             {{ ['b', 'A', 'a', 'B'] | sort(reverse=true) | join(',') }} \
             {{ [{'id': 'b', 'i': 0}, {'id': 'A', 'i': 1}, {'id': 'a', 'i': 2}] \
                | sort(attribute='id', case_sensitive=true) | map(attribute='i') | join(',') }} \
             {{ [{'id': 'b', 'i': 0}, {'id': 'A', 'i': 1}, {'id': 'a', 'i': 2}] \
                | sort(attribute='id') | map(attribute='i') | join(',') }} \
             {% for pair in [[2, 'x'], [1, 'y'], [2, 'z']] | sort(attribute=0) %}{{ pair[1] }}{% endfor %}"
        ),
        "A,a,b,B A,B,a,b b,B,A,a 1,2,0 1,2,0 yxz"
    );
}

#[test]
fn dictsort_min_and_max_fold_case_unless_case_sensitive() {
    assert_eq!(
        render(
            "{% for k, v in {'b': 1, 'B': 2, 'a': 3} | dictsort %}{{ k }}{% endfor %} \
             {% for k, v in {'b': 1, 'B': 2, 'a': 3} | dictsort(case_sensitive=true) %}{{ k }}{% endfor %} \
             {% for k, v in {'x': 'b', 'y': 'A', 'z': 'a'} | dictsort(by='value') %}{{ k }}{% endfor %} \
             {{ ['b', 'a', 'C'] | min }}{{ ['b', 'a', 'C'] | max }} \
             {{ ['b', 'a', 'C'] | min(case_sensitive=true) }}{{ ['b', 'a', 'C'] | max(case_sensitive=true) }} \
             {{ ([{'n': 2}, {'n': 5}, {'n': 1}] | max(attribute='n')).n }}"
        ),
        "abB Bab yzx aC Cb 5"
    );
}
