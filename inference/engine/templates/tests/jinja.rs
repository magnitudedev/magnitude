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

#[test]
fn global_builtins_cover_dict_and_standard_python_helpers() {
    assert_eq!(
        render(
            "{{ dict() | tojson }} \
             {{ dict(a=1, b='x') | tojson }} \
             {{ dict({'a': 1, 'b': 2}, b=3, c=4) | tojson }} \
             {{ dict([('k', 1), ('m', 2)]) | tojson }}"
        ),
        r#"{} {"a": 1, "b": "x"} {"a": 1, "b": 3, "c": 4} {"k": 1, "m": 2}"#
    );
    assert_eq!(
        render(
            "{% for a, b in zip([1, 2, 3], ['x', 'y']) %}{{ a }}{{ b }}{% endfor %} \
             {% for i, v in enumerate(['a', 'b'], start=1) %}{{ i }}:{{ v }},{% endfor %} \
             {{ len([1, 2, 3]) }} {{ len('abcd') }} {{ len({'a': 1, 'b': 2}) }} \
             {{ str(42) }} {{ int('ff', 16) }} {{ int(3.9) }} {{ float('1.5') }} \
             {{ bool(0) }} {{ bool('hi') }} {{ list('ab') | join('-') }} {{ list({'x': 1, 'y': 2}) | join(',') }}"
        ),
        "1x2y 1:a,2:b, 3 4 2 42 255 3 1.5 False True a-b x,y"
    );
    assert_eq!(
        render(
            "{{ min(3, 1, 2) }} {{ max([3, 9, 2]) }} {{ min([], default=7) }} \
             {{ sum([1, 2, 3]) }} {{ sum([1, 2.5], 10) }} {{ sum([[1], [2]], []) | join(',') }} \
             {{ abs(-5) }} {{ abs(-2.5) }} {{ round(3.6) }} {{ round(3.14159, 2) }} \
             {{ sorted([3, 1, 2]) | join(',') }} {{ sorted([3, 1, 2], reverse=true) | join(',') }} \
             {{ reversed([1, 2, 3]) | join(',') }} \
             {{ any([0, '', 1]) }} {{ any([0, '']) }} {{ all([1, 'a']) }} {{ all([1, 0]) }}"
        ),
        "1 9 7 6 13.5 1,2 5 2.5 4 3.14 1,2,3 3,2,1 3,2,1 True False True False"
    );
}

#[test]
fn object_filters_do_not_shadow_object_keys() {
    assert_eq!(
        render(
            "{% set schema = {'default': 'fallback', 'length': 8, 'string': 'ok'} %}\
             {{ schema.default }} {{ schema.length }} {{ schema.string }} \
             {{ (schema | default({})).default }} {{ (missing | default({'k': 1})).k }} \
             {{ ({} | d({'k': 2}, boolean=true)).k }} {{ schema | length }} {{ schema | list | join(',') }}"
        ),
        "fallback 8 ok fallback 1 2 3 default,length,string"
    );
}

#[test]
fn tojson_sort_keys_and_from_json_roundtrip() {
    assert_eq!(
        render(
            "{{ {'b': 1, 'a': {'d': 2, 'c': 3}} | tojson(sort_keys=true) }} \
             {% set parsed = '{\"x\": 42, \"y\": [1, 2]}' | from_json %}\
             {{ parsed.x }}:{{ parsed.y | sum }}"
        ),
        r#"{"a": {"c": 3, "d": 2}, "b": 1} 42:3"#
    );
}

#[test]
fn select_reject_tests_and_array_filters() {
    assert_eq!(
        render(
            "{{ [0, 1, '', 'a'] | select | join(',') }} \
             {{ [0, 1, '', 'a'] | reject | length }} \
             {{ [1, 2, 3, 4] | select('odd') | join(',') }} \
             {{ [1, 2, 3, 4] | select('le', 2) | join(',') }} \
             {{ [1, 2, 3, 4] | reject('equalto', 3) | join(',') }} \
             {{ [{'a': 1}, {'b': 2}] | selectattr('a', 'defined') | map(attribute='a') | join(',') }} \
             {{ [{'k': 'x'}, {'k': 'y'}] | rejectattr('k', 'equalto', 'x') | map(attribute='k') | join(',') }} \
             {{ 2 is le(2) }} {{ 3 is le(2) }} {{ {} is iterable }} \
             {{ [{'n': 2}, {'n': 5}] | sum(attribute='n', start=1) }}"
        ),
        "1,a 2 1,3 1,2 1,2,4 1 y True False True 8"
    );
}

#[test]
fn do_statement_and_mutating_dict_and_list_methods() {
    assert_eq!(
        render(
            "{% set d = {'a': 1} %}\
             {% do d.update({'b': 2}, c=3) %}\
             {% set popped = d.pop('b') %}\
             {% set missing = d.pop('z', 9) %}\
             {% set s1 = d.setdefault('a', 99) %}\
             {% set s2 = d.setdefault('d', 4) %}\
             {% set d2 = d.copy() %}\
             {% do d.clear() %}\
             {{ popped }},{{ missing }},{{ s1 }},{{ s2 }},{{ d | length }},{{ d2 | tojson(sort_keys=true) }}"
        ),
        r#"2,9,1,4,0,{"a": 1, "c": 3, "d": 4}"#
    );
    assert_eq!(
        render(
            "{% set xs = [1, 2] %}\
             {% do xs.extend([3, 2]) %}\
             {% do xs.insert(1, 9) %}\
             {% do xs.remove(2) %}\
             {% set ys = xs.copy() %}\
             {% set idx = xs.index(2) %}\
             {% set cnt = xs.count(2) %}\
             {% set total = xs | count %}\
             {% do xs.clear() %}\
             {{ idx }},{{ cnt }},{{ total }},{{ xs | length }},{{ ys | join(',') }}"
        ),
        "3,1,4,0,1,9,3,2"
    );
}

#[test]
fn string_methods_negative_indexing_and_char_loops() {
    assert_eq!(
        render(
            "{{ 'hello'[-1] }}{{ 'hello'[-5] }} \
             {{ 'banana'.find('na') }},{{ 'banana'.rfind('na') }},{{ 'banana'.count('na') }} \
             {{ 'a\\nb\\r\\nc\\n'.splitlines() | join('|') }} \
             {{ 'a\\nb'.splitlines(true) | length }} \
             {{ 'prefix_val'.removeprefix('prefix_') }},{{ 'val_suffix'.removesuffix('_suffix') }} \
             {{ 'foo.py'.endswith(('.rs', '.py')) }},{{ 'foo.py'.startswith(['bar', 'foo']) }} \
             {{ '  a  b\\tc  '.split() | join('|') }} \
             [{{ '  a  b  c  '.split(none, 1) | join('|') }}] \
             [{{ '  a  b  c  '.rsplit(none, 1) | join('|') }}] \
             {{ 'aaa'.replace('a', 'b', 2) }},{{ 'aaa'.replace('a', 'b', 0) }} \
             {{ ','.join(['x', 'y', 'z']) }} \
             {{ 'abc' | list | join('-') }} \
             {% for ch in 'xyz' %}[{{ ch }}]{% endfor %}"
        ),
        "oh 2,4,2 a|b|c 2 val,val True,True a|b|c [a|b  c  ] [  a  b|c] bba,aaa x,y,z a-b-c [x][y][z]"
    );
}
