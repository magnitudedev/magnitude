use crate::checked::{check_source, SourceFile, SourceSet};

fn check(source: &str) -> Result<(), String> {
    check_source(SourceSet::new(vec![SourceFile {
        path: "initialization.seismic".into(),
        text: source.into(),
    }]))
    .map(|_| ())
    .map_err(|error| error.to_string())
}
const FILL: &str =
    "fn fill[N](a: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        a[i] = 0.0\n\n";

#[test]
fn deferred_reads_resolve_before_private_paths_or_input_requirements_escape() {
    let nested = "fn read(a: &tensor[4] f32) -> f32:\n    let mut result = 0.0\n    for i in 0..2:\n        for j in 0..2:\n            result += a[i*2+j]\n    return result\n\n";
    check(&format!("{FILL}{nested}fn probe() -> f32:\n    let mut a = tensor[4] f32\n    fill(a)\n    return read(a)\n")).unwrap();
    let error = check(&format!(
        "{nested}fn probe() -> f32:\n    let a = tensor[4] f32\n    return read(a)\n"
    ))
    .unwrap_err();
    assert!(error.contains("initialization"), "{error}");

    // The contradictory guard belongs to the inner loop. It must be resolved
    // before exporting a requirement with those private symbols removed.
    check("fn read(a: &tensor[4] f32) -> f32:\n    let mut result = 0.0\n    for i in 0..2:\n        for j in 0..2:\n            if j < i and j >= i:\n                result += a[j]\n    return result\n\nfn probe() -> f32:\n    let a = tensor[4] f32\n    return read(a)\n").unwrap();
    check("fn probe() -> f32:\n    let a = tensor[4] f32\n    let mut result = 0.0\n    for i in 0..2:\n        for j in 0..2:\n            if j < i and j >= i:\n                result += a[j]\n    return result\n").unwrap();
}

#[test]
fn deferred_reads_preserve_source_writes_and_trial_incoming_requirements() {
    check("fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    a[0] = 1.0\n    for i in 1..4:\n        a[i] = a[i-1]\n    return a\n").unwrap();
    let error = check("fn probe() -> f32:\n    let mut a = tensor[4] f32\n    let mut result = 0.0\n    for i in 0..4:\n        result += a[i]\n        a[i] = 1.0\n    return result\n").unwrap_err();
    assert!(error.contains("initialization"), "{error}");

    let carry = "fn carry(x: tensor[2,3] f32) -> f32:\n    let mut y = x\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n        let current = y[0,0]\n    return y[0,0]\n\n";
    check(&format!("{carry}fn probe() -> f32:\n    let mut x = tensor[2,3] f32\n    x[:] = zeros_like(x)\n    return carry(x)\n")).unwrap();
    let error = check(&format!(
        "{carry}fn probe() -> f32:\n    let mut x = tensor[2,3] f32\n    return carry(x)\n"
    ))
    .unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn rejects_the_seven_whole_root_promotion_counterexamples() {
    let cases = [
        (
            "read before later fill",
            "fn bad[N](a: &mut tensor[N] f32):\n    let prior = a[0]\n    fill(a)\n\nfn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    bad(a)\n    return a\n",
        ),
        (
            "compound assignment reads old state",
            "fn bad[N](a: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        a[i] += 1.0\n\nfn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    bad(a)\n    return a\n",
        ),
        (
            "slice fill is not whole root",
            "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    fill(a[:2])\n    return a\n",
        ),
        (
            "conditional forwarding keeps its path",
            "fn maybe[N](a: &mut tensor[N] f32, choose: bool):\n    if choose:\n        fill(a)\n\nfn probe(choose: bool) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    maybe(a, choose)\n    return a\n",
        ),
        (
            "zero trip writes nothing",
            "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    for i in 0..0:\n        a[:] = zeros_like(a)\n    return a\n",
        ),
        (
            "first point is not whole loop image",
            "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    parallel for i in 0..4:\n        a[i] = 0.0\n        let future = a[3]\n    return a\n",
        ),
        (
            "diagonal is not cartesian coverage",
            "fn diagonal[N](a: &mut tensor[N, N] f32):\n    parallel for i in 0..N:\n        a[i, i] = 0.0\n\nfn probe() -> tensor[4, 4] f32:\n    let mut a = tensor[4, 4] f32\n    diagonal(a)\n    return a\n",
        ),
    ];
    for (name, source) in cases {
        let error = check(&format!("{FILL}{source}")).expect_err(name);
        assert!(
            error.contains("initialization"),
            "{name}: wrong diagnostic: {error}"
        );
    }
}

#[test]
fn accepts_partial_writes_views_and_helper_sequences() {
    for source in [
        "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    a[0] = 1.0\n    let current = a[0]\n    a[1:] = zeros_like(a[1:])\n    return a\n",
        "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    fill(a[:2])\n    fill(a[2:])\n    return a\n",
        "fn probe() -> tensor[4] f32:\n    let a = tensor[4] f32\n    let view = a[:]\n    return zeros_like(view)\n",
        "fn probe(choose: bool) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    if choose:\n        fill(a)\n    if choose:\n        let initialized = a[0]\n    return zeros_like(a)\n",
        "fn probe(choose: bool) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    if choose:\n        a = zeros_like(a)\n    else:\n        a = ones_like(a)\n    return a\n",
        "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    a[0] = 1.0\n    for i in 1..4:\n        a[i] = a[i-1]\n    return a\n",
    ] {
        check(&format!("{FILL}{source}")).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn complete_owned_carry_survives_rectangular_transpose_reshape() {
    check("fn probe(x: &tensor[2,3] f32) -> tensor[2,3] f32:\n    let mut y = to_owned(x)\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n        y[0,1] = 17.0\n    return y\n").unwrap();
}

#[test]
fn initialized_point_survives_rectangular_transpose_reshape_carry() {
    check("fn probe(x: &tensor[2,3] f32) -> tensor[1] f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n    let mut result = tensor[1] f32\n    result[0] = y[0,0]\n    return result\n").unwrap();
    check("fn probe(x: &tensor[2,3] f32, choose: bool) -> f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..2:\n        if choose:\n            y = reshape(y.T, (2,3))\n    return y[0,0]\n").unwrap();
    check("fn probe(x: &tensor[2,3] f32) -> f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..0:\n        y = reshape(y.T, (2,3))\n    return y[0,0]\n").unwrap();
}

#[test]
fn partial_owned_carry_is_not_promoted_through_transpose_reshape() {
    let error = check("fn probe(x: &tensor[2,3] f32) -> tensor[2,3] f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n        let missing = y[0,1]\n    return to_owned(x)\n").unwrap_err();
    assert!(error.contains("initialization"), "{error}");
    let error = check("fn probe(x: &tensor[2,3] f32, choose: bool) -> f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..2:\n        if choose:\n            y = reshape(y.T, (2,3))\n        else:\n            y = tensor[2,3] f32\n    return y[0,0]\n").unwrap_err();
    assert!(error.contains("initializ"), "{error}");
}

#[test]
fn preserves_joint_rectangular_flattened_and_partition_maps() {
    for source in [
        "fn probe[N, M](shape: &tensor[N, M] f32) -> tensor[N, M] f32:\n    let mut a = tensor[N, M] f32\n    parallel for i in 0..N:\n        parallel for j in 0..M:\n            a[i,j] = 0.0\n    return a\n",
        "fn probe[N, M](shape: &tensor[N, M] f32) -> tensor[N*M] f32:\n    let mut a = tensor[N*M] f32\n    parallel for i in 0..N:\n        parallel for j in 0..M:\n            a[i*M+j] = 0.0\n    return a\n",
        "fn probe[N, M](shape: &tensor[N, M] f32) -> tensor[N, M] f32:\n    let mut a = tensor[N, M] f32\n    parallel for i in 0..N*M:\n        a[i/M,i%M] = 0.0\n    return a\n",
        "fn probe[N](shape: &tensor[N] f32) -> tensor[N*2] f32:\n    let mut a = tensor[N*2] f32\n    parallel for i in 0..N:\n        a[i*2:(i+1)*2] = zeros_like(a[i*2:(i+1)*2])\n    return a\n",
        "fn probe() -> tensor[0] f32:\n    let a = tensor[0] f32\n    return a\n",
    ] {
        check(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn parallel_iteration_cannot_borrow_predecessor_initialization() {
    let error=check("fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    a[0] = 1.0\n    parallel for i in 1..4:\n        a[i] = a[i-1]\n    return a\n").unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn parallel_accesses_require_pairwise_cross_visit_independence() {
    let unsafe_bodies = [
        "        a[i] = a[i + 1]\n",
        "        a[i] = 1.0\n        a[i + 1] = 2.0\n",
        "        let current = a[i + 1]\n        a[i] = current\n",
        "        a[:] = zeros_like(a)\n",
    ];
    for body in unsafe_bodies {
        let source =
            format!("fn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..3:\n{body}");
        let error = check(&source).expect_err(&source);
        assert!(
            error.contains("overlap across distinct visits"),
            "{source}\n{error}"
        );
    }
    for body in [
        "        a[i] = a[i] + 1.0\n",
        "        a[i * 2] = 1.0\n        a[i * 2 + 1] = 2.0\n",
    ] {
        let source =
            format!("fn probe(a: &mut tensor[6] f32):\n    parallel for i in 0..3:\n{body}");
        check(&source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
    check("fn probe(a: &mut tensor[4] f32):\n    for i in 0..3:\n        a[i] = a[i + 1]\n")
        .unwrap();
    check("fn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..1:\n        a[:] = zeros_like(a)\n")
        .unwrap();
    check("fn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..4:\n        if i == 0:\n            a[0] = 1.0\n        else:\n            a[i] = a[i] + 1.0\n")
        .unwrap();
    let whole_rebind = "fn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..4:\n        a = zeros_like(a)\n";
    assert!(check(whole_rebind).is_err());
}

#[test]
fn parallel_accesses_follow_views_calls_and_nested_loop_footprints() {
    check("fn increment_one(dst: &mut tensor[1] f32):\n    dst[0] = dst[0] + 1.0\n\nfn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..4:\n        increment_one(a[i:i+1])\n").unwrap();
    let unsafe_call = "fn set_one(dst: &mut tensor[1] f32):\n    dst[0] = 1.0\n\nfn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..3:\n        set_one(a[i:i+1])\n        a[i+1] = 2.0\n";
    assert!(check(unsafe_call)
        .unwrap_err()
        .contains("overlap across distinct visits"));
    let unsafe_call_read = "fn read_one(src: &tensor[1] f32) -> f32:\n    return src[0]\n\nfn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..3:\n        a[i] = read_one(a[i+1:i+2])\n";
    assert!(check(unsafe_call_read)
        .unwrap_err()
        .contains("overlap across distinct visits"));
    check("fn probe(a: &mut tensor[3, 3] f32):\n    let mut view = a.T\n    parallel for i in 0..3:\n        view[i,0] = view[i,0] + 1.0\n").unwrap();
    let unsafe_transpose = "fn probe(a: &mut tensor[3, 3] f32):\n    let mut view = a.T\n    parallel for i in 0..2:\n        view[i,0] = view[i+1,0]\n";
    assert!(check(unsafe_transpose)
        .unwrap_err()
        .contains("overlap across distinct visits"));
    check("fn probe(a: &mut tensor[4, 4] f32):\n    parallel for i in 0..4:\n        parallel for j in 0..4:\n            a[i,j] = a[i,j] + 1.0\n").unwrap();
    check("fn probe(a: &mut tensor[4, 4] f32):\n    parallel for i in 0..4:\n        for j in 0..4:\n            a[i,j] = a[i,j] + 1.0\n").unwrap();
    let unsafe_nested = "fn probe(a: &mut tensor[4, 4] f32):\n    parallel for i in 0..3:\n        parallel for j in 0..4:\n            a[i,j] = a[i+1,j]\n";
    assert!(check(unsafe_nested)
        .unwrap_err()
        .contains("overlap across distinct visits"));
}

#[test]
fn parallel_branch_on_per_visit_read_does_not_exclude_cross_visit_accesses() {
    let source = "fn probe(choice: &tensor[2] u32, out: &mut tensor[3] f32):\n    parallel for i in 0..2:\n        if choice[i] == 0:\n            out[i] = 1.0\n        else:\n            out[i + 1] = 2.0\n";
    let error =
        check(source).expect_err("different visits can take opposite arms and write out[1]");
    assert!(error.contains("overlap across distinct visits"), "{error}");
}

#[test]
fn parallel_row_store_and_captured_condition_keep_independent_footprints() {
    check("fn probe(a: &mut tensor[2, 4] f32):\n    parallel for row in 0..2:\n        a[row, 1:4] = a[row, 0:3]\n").unwrap();
    check("fn probe(choose: bool, a: &mut tensor[4] f32):\n    parallel for i in 0..2:\n        if choose:\n            a[i] = 1.0\n        else:\n            a[i + 2] = 2.0\n").unwrap();
}

#[test]
fn initialized_branch_uses_actual_wrapped_word_value() {
    for expression in ["2147483647 + 1", "i32(u32(4294967295))"] {
        let source = format!(
            "fn probe() -> tensor[1] f32:\n    let mut out = tensor[1] f32\n    let n = {expression}\n    if n < 0:\n        out[0] = 1.0\n    return out\n"
        );
        check(&source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn tensor_data_can_choose_a_wide_quantity_carry_path() {
    check("fn probe(flag: &tensor[1] bool, start: index[1000000000], times: range[1000000000]) -> i32:\n    let mut q = start + 0\n    if flag[0]:\n        for i in times:\n            q = q * (i + 1)\n    return i32(q)\n")
        .unwrap();
}

#[test]
fn negative_mutable_quantity_cannot_be_assumed_to_be_a_shape() {
    let source = "fn probe(i: index[4]):\n    let mut q = i - 1\n    let a = tensor[q] f32\n";
    let error = check(source).expect_err("signed quantity has no implicit nonnegative fact");
    assert!(error.contains("tensor extent may be negative"), "{error}");
    check("fn probe(i: index[4]):\n    let mut q = i - 1\n    if q >= 0:\n        let a = tensor[q] f32\n")
        .unwrap();
}

#[test]
fn participant_local_quantities_join_and_divergent_cohorts_are_rejected() {
    // L13/L14: a cohort intrinsic under participant-divergent control.
    let error = check("fn probe(flag: &tensor[32] bool, x: &tensor[32] f32, out: &mut tensor[32] f32):\n    parallel for i in 0..32:\n        out[i] = x[i]\n\nlower probe(flag: &tensor[32] bool, x: &tensor[32] f32, out: &mut tensor[32] f32)\n    for metal requires metal.subgroup:\n    parallel for i in 0..32:\n        let mut q = i - 1\n        if flag[i]:\n            q = i + 1\n        if q >= 0:\n            out[i] = metal.subgroup.simd_sum(x[i])\n        else:\n            out[i] = x[i]\n").unwrap_err();
    assert!(error.contains("Placement"), "{error}");
    assert!(error.contains("differs between participants"), "{error}");

    check("fn probe(flag: &tensor[32] bool, x: &tensor[32] f32, out: &mut tensor[32] f32):\n    parallel for i in 0..32:\n        out[i] = x[i]\n\nlower probe(flag: &tensor[32] bool, x: &tensor[32] f32, out: &mut tensor[32] f32)\n    for metal requires metal.subgroup:\n    parallel for i in 0..32:\n        let mut selector = i - i\n        if flag[i]:\n            selector = i + 0\n        out[i] = metal.subgroup.shuffle(x[i], i32(selector))\n").unwrap();
}

#[test]
fn parallel_loop_does_not_implicitly_reduce_captured_scalar_state() {
    let source = "fn probe() -> i32:\n    let mut x = 0\n    parallel for i in 0..4:\n        x += i\n    return x\n";
    let error = check(source).unwrap_err();
    assert!(error.contains("cannot reassign captured state"), "{error}");

    let owned_tensor = "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    parallel for i in 0..4:\n        a = zeros_like(tensor[4] f32)\n    return a\n";
    let error = check(owned_tensor).unwrap_err();
    assert!(error.contains("cannot reassign captured state"), "{error}");

    check(
        "fn probe() -> i32:\n    let mut x = 0\n    for i in 0..4:\n        x += i\n    return x\n",
    )
    .unwrap();
    check("fn probe(a: &mut tensor[4] f32):\n    parallel for i in 0..4:\n        let mut x = 0\n        x += i\n        a[i] = 1.0\n").unwrap();
}

#[test]
fn reassigned_predicate_cannot_reuse_its_old_path_fact() {
    let source = "fn probe(input: bool) -> tensor[4] f32:\n    let mut choose = input\n    let mut a = tensor[4] f32\n    if choose:\n        fill(a)\n    choose = not choose\n    if choose:\n        let missing = a[0]\n    return zeros_like(a)\n";
    assert!(check(&format!("{FILL}{source}"))
        .unwrap_err()
        .contains("initialization"));
}

#[test]
fn loop_carried_predicate_does_not_reuse_its_initial_value() {
    let source = "fn probe(input: bool) -> tensor[4] f32:\n    let mut choose = input\n    let mut a = tensor[4] f32\n    for i in 0..4:\n        if choose:\n            a[i] = 0.0\n        choose = not choose\n    if input:\n        return a\n    return zeros_like(a)\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}
#[test]
fn mutable_view_reads_its_current_initialized_region() {
    check("fn probe() -> tensor[2] f32:\n    let mut a = tensor[2] f32\n    let mut view = a[:]\n    view[0] = 1.0\n    let current = view[0]\n    view[1] = current\n    return to_owned(view)\n").unwrap();
}

#[test]
fn optional_contract_uses_actual_input_and_preserves_reference_output() {
    use crate::entry::ElementBindings;
    use crate::initialization::{
        InitializationArgument as Argument, InitializationContext, InitializationPhase,
        InitializationState,
    };
    let source = "fn reference(a: &mut tensor[4] f32):\n    a[:] = zeros_like(a)\n\nfn stronger(a: &mut tensor[4] f32):\n    let old = a[0]\n    a[:] = zeros_like(a)\n\nfn weaker(a: &mut tensor[4] f32):\n    a[0] = 0.0\n\nfn probe(a: &mut tensor[4] f32):\n    reference(a)\n    stronger(a)\n    weaker(a)\n";
    let checked = check_source(SourceSet::new(vec![SourceFile {
        path: "contracts.seismic".into(),
        text: source.into(),
    }]))
    .unwrap();
    let mut entry = checked
        .entry(
            checked.entry_named("probe").unwrap(),
            &ElementBindings::default(),
        )
        .unwrap()
        .into_parts();
    let contract = |name: &str| {
        entry
            .program
            .functions()
            .find(|(_, function)| function.name() == name)
            .unwrap()
            .1
            .initialization()
    };
    let reference = contract("reference");
    let stronger = contract("stronger");
    let weaker = contract("weaker");
    let four = entry.arena.int(4);
    let zero = entry.arena.int(0);
    let mut context = InitializationContext::new(&mut entry.arena);
    let view = context.root(&[four]);
    let fresh = Argument::Tensor {
        state: InitializationState::empty(),
        view: view.clone(),
    };
    assert_eq!(
        context
            .applicable(stronger, reference, &[fresh.clone()])
            .unwrap_err()
            .phase,
        InitializationPhase::Input
    );
    assert_eq!(
        context
            .applicable(weaker, reference, &[fresh.clone()])
            .unwrap_err()
            .phase,
        InitializationPhase::Output
    );
    let initialized = Argument::Tensor {
        state: InitializationState::full(),
        view: view.clone(),
    };
    assert!(context
        .applicable(stronger, reference, &[initialized])
        .is_ok());
    let first = context.slice(&view, &[(Some(zero), None, true)]);
    let partial = context.write(&InitializationState::empty(), &first);
    let partial = Argument::Tensor {
        state: partial,
        view: view.clone(),
    };
    let outputs = context.applicable(stronger, reference, &[partial]).unwrap();
    assert!(context.readable(outputs[0].as_ref().unwrap(), &view));
}

#[test]
fn helper_private_predicates_cannot_manufacture_public_initialization() {
    let source = "fn maybe(a: &mut tensor[4] f32, source: f32):\n    if source > 0.0:\n        a[:] = zeros_like(a)\n\nfn probe(source: f32) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    maybe(a, source)\n    return a\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}

#[test]
fn nested_successful_return_does_not_gain_later_writes() {
    let source = "fn maybe(a: &mut tensor[4] f32, outer: bool, inner: bool):\n    if outer:\n        if inner:\n            return\n    a[:] = zeros_like(a)\n\nfn probe(outer: bool, inner: bool) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    maybe(a, outer, inner)\n    return a\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}

#[test]
fn tensor_carries_require_a_preserved_initialized_region() {
    let unsafe_source = "fn probe() -> tensor[2] f32:\n    let mut a = zeros_like(tensor[2] f32)\n    for i in 0..2:\n        let missing_on_second_visit = a[1]\n        let mut next = tensor[2] f32\n        next[0] = 1.0\n        a = next\n    return zeros_like(a)\n";
    assert!(check(unsafe_source).unwrap_err().contains("initializ"));
    let partial = "fn probe() -> f32:\n    let mut a = tensor[2] f32\n    a[0] = 1.0\n    for i in 0..4:\n        let current = a[0]\n        let mut next = tensor[2] f32\n        next[0] = current + 1.0\n        a = next\n    return a[0]\n";
    check(partial).unwrap();
    let one_visit = unsafe_source.replace("0..2:", "0..1:");
    check(&one_visit).unwrap();
    let zero_visits = unsafe_source.replace("0..2:", "0..0:");
    check(&zero_visits).unwrap();
    check("fn replace(a: &tensor[2] f32) -> tensor[2] f32:\n    let mut carry = tensor[2] f32\n    for i in 0..2:\n        carry = zeros_like(a)\n    return carry\n").unwrap();
}

#[test]
fn prior_iterations_do_not_inherit_current_dynamic_predicate() {
    let source = "fn maybe(a: &mut tensor[1] f32, choose: bool):\n    if choose:\n        a[0] = 1.0\n\nfn probe(flags: &tensor[2] bool) -> tensor[2] f32:\n    let mut a = tensor[2] f32\n    for i in 0..2:\n        let choose = flags[i]\n        maybe(a[i:i+1], choose)\n        if choose and i > 0:\n            let missing = a[i-1]\n    return zeros_like(a)\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}

#[test]
fn parallel_writes_are_separated_on_any_axis() {
    for source in [
        "fn probe[M, N](x: &tensor[M, N] f32, y: &mut tensor[M, N] f32):\n    parallel for c in 0..N:\n        for r in 0..M:\n            y[r, c] = x[r, c]\n",
        "fn probe[M, T, KV](key: &tensor[M, KV] f32, hist: &mut tensor[T, KV] f32, dest: &tensor[M] i32):\n    parallel for head in 0..KV:\n        for row in 0..M:\n            hist[dest[row], head] = key[row, head]\n",
    ] {
        check(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
    for body in ["        out[0, 0] = 1.0\n", "        out[i % 2, 0] = 1.0\n"] {
        let source =
            format!("fn probe(out: &mut tensor[4, 4] f32):\n    parallel for i in 0..4:\n{body}");
        let error = check(&source).expect_err(&source);
        assert!(
            error.contains("overlap across distinct visits"),
            "{source}\n{error}"
        );
    }
}

#[test]
fn atomic_accesses_commute_only_with_the_same_operation() {
    check("fn probe[N](x: &tensor[N] i32, out: &mut tensor[1] i32):\n    parallel for i in 0..N:\n        atomic(add, out[0], x[i])\n        atomic(add, out[0], x[i])\n")
        .unwrap();
    let mixed = "fn probe[N](x: &tensor[N] i32, out: &mut tensor[1] i32):\n    parallel for i in 0..N:\n        atomic(add, out[0], x[i])\n        atomic(max, out[0], x[i])\n";
    assert!(check(mixed)
        .unwrap_err()
        .contains("overlap across distinct visits"));
    let ordinary = "fn probe[N](x: &tensor[N] i32, out: &mut tensor[1] i32):\n    parallel for i in 0..N:\n        atomic(add, out[0], x[i])\n        out[0] = x[i]\n";
    assert!(check(ordinary)
        .unwrap_err()
        .contains("overlap across distinct visits"));
}

#[test]
fn branch_writes_under_a_non_leading_loop_cover_the_root() {
    for source in [
        "fn w[C, N](t: &mut tensor[C, N] f32):\n    for column in 0..N:\n        for j in 0..C:\n            if j < 1:\n                t[j, column] = 1.0\n            else:\n                t[j, column] = 2.0\n\nfn probe[C, N](x: &tensor[C, N] f32) -> tensor[C, N] f32:\n    let mut t = tensor[C, N] f32\n    w(t)\n    return t\n",
        "fn w[M, C, N](projected: &tensor[M, N] f32, window: &tensor[C - 1, N] f32, next: &mut tensor[C - 1, N] f32):\n    for column in 0..N:\n        for j in 0..C - 1:\n            if M + j < C - 1:\n                next[j, column] = window[M + j, column]\n            else:\n                next[j, column] = projected[M + j - (C - 1), column]\n\nfn probe[M, C, N](projected: &tensor[M, N] f32, window: &tensor[C - 1, N] f32) -> tensor[C - 1, N] f32:\n    let mut t = tensor[C - 1, N] f32\n    w(projected, window, t)\n    return t\n",
        "fn probe[M, N](x: &tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut t = tensor[M, N] f32\n    for c in 0..N:\n        t[0:M, c] = x[0:M, c]\n    return t\n",
    ] {
        check(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
    let partial = "fn probe[M, N](x: &tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut t = tensor[M, N] f32\n    for c in 1..N:\n        t[0:M, c] = x[0:M, c]\n    return t\n";
    assert!(check(partial).unwrap_err().contains("initialization"));
}

#[test]
fn a_failed_callee_reports_only_its_own_diagnostics() {
    let source = "fn broken[N](x: &tensor[N] f32, y: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        y[i] = x[i] + undeclared\n\nfn probe[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut t = tensor[N] f32\n    let mut u = tensor[N] f32\n    broken(x, t)\n    broken(t, u)\n    return u\n";
    let error = check(source).unwrap_err();
    assert!(error.contains("undeclared"), "{error}");
    assert!(!error.contains("before this read"), "{error}");
}

#[test]
fn calls_keep_only_alternatives_whose_contract_applies() {
    use crate::entry::ElementBindings;
    let candidates = |alternative: &str| {
        let source = format!(
            "fn fill[N](a: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        a[i] = 0.0\n\nfn fill[N](a: &mut tensor[N] f32):\n{alternative}\nfn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    fill(a)\n    return a\n"
        );
        let checked = check_source(SourceSet::new(vec![SourceFile {
            path: "alternatives.seismic".into(),
            text: source.clone(),
        }]))
        .unwrap_or_else(|error| panic!("{source}\n{error}"));
        let entry = checked
            .entry(
                checked.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        let program = entry.program();
        let (_, family) = program
            .families()
            .find(|(_, family)| family.name() == "fill")
            .unwrap();
        family.candidates().len()
    };
    assert_eq!(
        candidates("    a[:] = zeros_like(a)\n"),
        2,
        "equal guarantees"
    );
    assert_eq!(candidates("    a[0] = 0.0\n"), 1, "weaker guarantees");
    assert_eq!(
        candidates("    let old = a[0]\n    a[:] = zeros_like(a)\n"),
        1,
        "a requirement the reference does not have"
    );
}

#[test]
fn borrowed_whole_assignment_updates_the_actual_place() {
    use crate::entry::ElementBindings;
    use crate::interp::{Arg, Interpreter, TensorData};
    use crate::types::DType;
    for body in [
        "    dst = zeros_like(dst)\n",
        "    if true:\n        let mut view = dst[:]\n        view = zeros_like(view)\n",
    ] {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "borrowed-assignment.seismic".into(),
            text: format!("fn probe(dst: &mut tensor[2] f32):\n{body}"),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        let mut oracle = Interpreter::new(&entry);
        let input = oracle.add_tensor(TensorData::dense(DType::F32, vec![2], vec![3., 4.]));
        let outcome = oracle.run(&[Arg::Tensor(input)]).unwrap();
        let actual = outcome.inputs().next().unwrap();
        assert_eq!(actual.tensor().read(0).unwrap(), 0., "{body}");
        assert_eq!(actual.tensor().read(1).unwrap(), 0., "{body}");
    }
}

#[test]
fn allocation_extent_with_a_signed_cofactor_is_rejected() {
    // `i*k - k = (i - 1)*k` is negative for `i >= 2`, `k <= -1` (G-C24-1).
    let error = check("fn probe[N](x: &tensor[N] f32, out: &mut tensor[N] f32):\n    let mut k = N - N\n    let m = N - N - 1\n    for i in 1..N:\n        k = k - 1\n        if k <= m:\n            let mut t = tensor[i * k - k] f32\n            for q in 0..i * k - k:\n                t[q] = 1.0\n            out[i] = 1.0\n").unwrap_err();
    assert!(error.contains("tensor extent may be negative"), "{error}");
}

fn sequential_ifs(header: &str, count: usize) -> String {
    let mut source = format!(
        "fn probe[N](x: &tensor[N] f32, y: &mut tensor[N] f32):\n{header}        let mut v = x[i]\n"
    );
    for k in 1..=count {
        source.push_str(&format!("        if v > {k}.0:\n            v = v - 1.0\n"));
    }
    source.push_str("        y[i] = v\n");
    source
}

#[test]
fn seqif_ser_20_checks() {
    // One world per program point (L30): twenty joins check on the default
    // test thread, in the debug profile, without path splitting.
    check(&sequential_ifs("    for i in 0..N:\n", 20)).unwrap();
    check(&sequential_ifs("    parallel for i in 0..N:\n", 20)).unwrap();
}

#[test]
fn sequential_loop_exits_join() {
    let mut source = "fn probe[N](x: &tensor[N] f32, y: &mut tensor[N] f32):\n".to_string();
    for _ in 0..20 {
        source.push_str("    for i in 0..N:\n        y[i] = x[i]\n");
    }
    check(&source).unwrap();
}

#[test]
fn joined_integer_keeps_the_bounds_both_arms_prove() {
    // After the join `j` is 1 or 2, so `t[j]` reads the initialized `t[0:3]`.
    let read = |other: u32| {
        format!(
            "fn probe(c: bool) -> f32:\n    let mut t = tensor[4] f32\n    t[0:3] = zeros_like(t[0:3])\n    let mut j = 1\n    if c:\n        j = {other}\n    return t[j]\n"
        )
    };
    check(&read(2)).unwrap();
    let error = check(&read(3)).unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn owned_helper_results_initialize_rebound_state_on_every_branch() {
    let helper = "fn make[N](seed: &tensor[N] f32) -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    parallel for i in 0..N:\n        result[i] = seed[i]\n    return result\n\n";
    for body in [
        "    result = make(seed)\n",
        "    if first:\n        result = make(seed)\n    else:\n        if second:\n            result = make(seed)\n        else:\n            result = make(seed)\n",
    ] {
        check(&format!("{helper}fn probe[N](seed: &tensor[N] f32, first: bool, second: bool) -> tensor[N] f32:\n    let mut result = tensor[N] f32\n{body}    return result\n")).unwrap();
    }
    let error = check(&format!("{helper}fn probe[N](seed: &tensor[N] f32, first: bool) -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    if first:\n        result = make(seed)\n    return result\n")).unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn boolean_domains_refine_only_after_alternatives_are_excluded() {
    let source = "fn aligned[N](x: &tensor[N] f32) where N % 32 == 0:\n    let unused = N\n\nfn legal_lanes[LANES](x: &tensor[LANES] f32) where LANES >= 0 and (LANES == 8 or LANES == 16 or LANES == 32):\n    let unused = LANES\n\nfn optional[N, MODE](x: &tensor[N] f32, mode: &tensor[MODE + 1] f32) where MODE >= 0 and (MODE != 1 or N % 32 == 0):\n    if MODE == 1:\n        aligned(x)\n\nfn narrower[LANES](x: &tensor[LANES] f32) where LANES >= 0 and (LANES == 16 or LANES == 32):\n    legal_lanes(x)\n\nfn optional_axis[I](x: &tensor[I] f32) -> f32 where I >= 0 and I <= 1:\n    let mut value = 0.0\n    if I == 0:\n        value = 0.0\n    else:\n        value = x[0]\n    return value\n";
    check(source).unwrap();
    let invalid = "fn aligned[N](x: &tensor[N] f32) where N % 32 == 0:\n    let unused = N\n\nfn unguarded[N, MODE](x: &tensor[N] f32, mode: &tensor[MODE + 1] f32) where MODE >= 0 and (MODE != 1 or N % 32 == 0):\n    aligned(x)\n";
    assert!(check(invalid).unwrap_err().contains("requires"));
}

#[test]
fn integer_interval_implies_only_an_exhaustive_boolean_domain() {
    let legal = "fn flag[MODE](x: &tensor[MODE + 1] f32) where MODE >= 0 and (MODE == 0 or MODE == 1):\n    let unused = MODE\n\nfn probe[MODE](x: &tensor[MODE + 1] f32) where MODE >= 0 and MODE <= 1:\n    flag(x)\n";
    check(legal).unwrap();
    let gap = "fn choices[MODE](x: &tensor[MODE + 1] f32) where MODE >= 0 and (MODE == 0 or MODE == 2):\n    let unused = MODE\n\nfn probe[MODE](x: &tensor[MODE + 1] f32) where MODE >= 0 and MODE <= 2:\n    choices(x)\n";
    assert!(check(gap).unwrap_err().contains("requires"));
    let lane_gap = "fn choices[LANES](x: &tensor[LANES] f32) where LANES >= 0 and (LANES == 16 or LANES == 32):\n    let unused = LANES\n\nfn probe[LANES](x: &tensor[LANES] f32) where LANES >= 16 and LANES <= 32:\n    choices(x)\n";
    assert!(check(lane_gap).unwrap_err().contains("requires"));
}

#[test]
fn scalar_live_count_keeps_actual_prefix_initialization_across_calls() {
    let helpers = "fn publish(x: &mut tensor[4] f32, count: i32):\n    if count >= 0 and count <= 4:\n        for i in 0..count:\n            x[i] = 1.0\n\nfn consume(x: &tensor[4] f32, count: i32) -> f32:\n    let mut total = 0.0\n    if count >= 0 and count <= 4:\n        for i in 0..count:\n            total = total + x[i]\n    return total\n\n";
    check(&format!("{helpers}fn probe(requested: i32) -> f32:\n    let count = max(requested, 0)\n    let mut x = tensor[4] f32\n    let mut total = 0.0\n    if count <= 4:\n        publish(x, count)\n        total = consume(x, count)\n    return total\n")).unwrap();
    let invalid = check(&format!("{helpers}fn probe() -> f32:\n    let mut x = tensor[4] f32\n    publish(x, 2)\n    return consume(x, 3)\n")).unwrap_err();
    assert!(invalid.contains("initialization"), "{invalid}");
}

#[test]
fn bounded_unsigned_word_prefix_cast_keeps_call_initialization() {
    let consume = "fn consume(x: &tensor[4] f32, count: u32) -> f32:\n    let live = i32(count)\n    let mut total = 0.0\n    if live >= 0 and live <= 4:\n        for i in 0..live:\n            total = total + x[i]\n    return total\n\n";
    check(&format!("{consume}fn probe(requested: i32) -> f32:\n    let mut x = tensor[4] f32\n    let mut total = 0.0\n    if requested >= 0 and requested <= 4:\n        for i in 0..requested:\n            x[i] = 1.0\n        total = consume(x, u32(requested))\n    return total\n")).unwrap();
    check(&format!("{consume}fn probe(requested: i32) -> f32:\n    let count = min(requested, 4)\n    let mut x = tensor[4] f32\n    let mut total = 0.0\n    if count >= 0:\n        for i in 0..count:\n            x[i] = 1.0\n        total = consume(x, u32(count))\n    return total\n")).unwrap();
    // Signed/unsigned casts preserve bits even for negative signed counts;
    // casting back cannot invent a positive prefix that reads unwritten data.
    check(&format!("{consume}fn probe(requested: i32) -> f32:\n    let mut x = tensor[4] f32\n    let mut total = 0.0\n    if requested < 0:\n        total = consume(x, u32(requested))\n    return total\n")).unwrap();
    let invalid = check(&format!("{consume}fn probe() -> f32:\n    let mut x = tensor[4] f32\n    for i in 0..2:\n        x[i] = 1.0\n    return consume(x, u32(3))\n")).unwrap_err();
    assert!(invalid.contains("initialization"), "{invalid}");
}

#[test]
fn conditional_overwrites_preserve_prior_written_regions() {
    let complete = "fn probe(flags: &tensor[2] i32) -> tensor[2, 3] f32:\n    let mut result = tensor[2, 3] f32\n    for row in 0..2:\n        for column in 0..3:\n            result[row, column] = 0.0\n        if flags[row] >= 0:\n            for column in 0..3:\n                if flags[row] != 0:\n                    result[row, column] = 1.0\n    return result\n";
    check(complete).unwrap();
    let incomplete = "fn probe(flags: &tensor[2] i32) -> tensor[2, 3] f32:\n    let mut result = tensor[2, 3] f32\n    for row in 0..2:\n        result[row, 0] = 0.0\n        if flags[row] >= 0:\n            for column in 0..3:\n                result[row, column] = 1.0\n    return result\n";
    assert!(check(incomplete).unwrap_err().contains("initialization"));
}

#[test]
fn initialization_region_union_is_structurally_idempotent() {
    use crate::initialization::Region;
    let mut arena = crate::expr::ExprArena::default();
    let a = Region::Linear(arena.int(0), arena.int(1));
    let b = Region::Linear(arena.int(2), arena.int(3));
    let combined = a.clone().union(b.clone());
    assert_eq!(combined.clone().union(a), combined);
    assert_eq!(combined.clone().union(b), combined);
    assert_eq!(combined.clone().union(combined.clone()), combined);
    // Deduplication retains a real gap; it never promotes an interval to Full.
    assert!(matches!(combined, Region::Union(ref parts) if parts.len() == 2));
}

#[test]
fn unconditional_coverage_preserves_guarded_read_requirements() {
    let consume = "fn consume(x: &tensor[4] f32, choose: bool) -> f32:\n    let mut result = 0.0\n    if choose:\n        result = x[0]\n    return result\n\n";
    // A genuinely fully initialized root covers a read on either branch.
    check(&format!("{consume}fn probe(choose: bool) -> f32:\n    let mut x = tensor[4] f32\n    for i in 0..4:\n        x[i] = 1.0\n    return consume(x, choose)\n")).unwrap();
    // The same guarded publication/read is valid without choosing that guard.
    check(&format!("{consume}fn probe(choose: bool) -> f32:\n    let mut x = tensor[4] f32\n    if choose:\n        x[0] = 1.0\n    return consume(x, choose)\n")).unwrap();
    // A conditional publication does not cover an unconditional read.
    let invalid = check("fn probe(choose: bool) -> f32:\n    let mut x = tensor[4] f32\n    if choose:\n        x[0] = 1.0\n    return x[0]\n").unwrap_err();
    assert!(invalid.contains("initialization"), "{invalid}");
}

#[test]
fn visit_geometry_precedes_guards_without_losing_overlap_checks() {
    // Each row is independent even when its data chooses different writes.
    check("fn probe(flags: &tensor[4] i32, out: &mut tensor[4, 2] f32):\n    parallel for row in 0..4:\n        if flags[row] > 0:\n            out[row, 0] = 1.0\n        else:\n            out[row, 1] = 2.0\n").unwrap();
    // Identical cells can still be legal only through the path-aware proof:
    // this captured choice permits exactly one of the two visits to write.
    check("fn probe(choose: bool, out: &mut tensor[1] f32):\n    parallel for i in 0..2:\n        if choose:\n            if i == 0:\n                out[0] = 1.0\n        else:\n            if i == 1:\n                out[0] = 2.0\n").unwrap();
    // Per-visit data cannot establish mutually exclusive cross-visit guards.
    let invalid = check("fn probe(flags: &tensor[4] i32, out: &mut tensor[1] f32):\n    parallel for row in 0..4:\n        if flags[row] > 0:\n            out[0] = 1.0\n").unwrap_err();
    assert!(invalid.contains("overlap"), "{invalid}");
}

#[test]
fn optional_zero_inference_preserves_empty_quotient_extents() {
    // An explicit equality still transports an exact nonlinear zero fact.
    check("fn probe[N]() -> tensor[N * N] f32 where N >= 0:\n    let mut result = tensor[N * N] f32\n    if N * N != 0:\n        for i in 0..N * N:\n            result[i] = 1.0\n    return result\n").unwrap();
    check("fn probe[H, KV]() -> tensor[H / KV] f32 where H >= 0:\n    let mut result = tensor[H / KV] f32\n    if H / KV > 0:\n        for i in 0..H / KV:\n            result[i] = 1.0\n    return result\n").unwrap();
    let invalid = check("fn probe[H, KV]() -> tensor[H / KV] f32 where H >= 0:\n    let mut result = tensor[H / KV] f32\n    if H / KV > 1:\n        for i in 0..H / KV:\n            result[i] = 1.0\n    return result\n").unwrap_err();
    assert!(invalid.contains("initialization"), "{invalid}");
}

#[test]
fn vacuity_keeps_incoming_requirements_only_on_possible_paths() {
    let impossible = "fn consume(x: &tensor[1] f32, value: i32):\n    if value >= 1:\n        if value <= 0:\n            let unused = x[0]\n\nfn probe(value: i32):\n    let mut x = tensor[1] f32\n    consume(x, value)\n";
    check(impossible).unwrap();
    let possible = impossible.replace("value <= 0", "value <= 1");
    let error = check(&possible).unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn finite_recipes_keep_symbolic_positive_coefficients_and_coupled_domains() {
    let source = r#"fn shared[SHARED_F, SHARED_C, QP, GP, KP, VP](input: f32) -> f32
    where SHARED_F >= 0 and SHARED_C >= 0 and QP >= 0 and GP >= 0 and KP >= 0 and VP >= 0 and (SHARED_F == 0 or SHARED_F == 1 or SHARED_F == 2) and (SHARED_F == 0 or SHARED_C > 0) and (QP == 0 or QP == 1) and (GP == 0 or GP == 1) and (KP == 0 or KP == 1) and (VP == 0 or VP == 1):
    return input

fn caller[HAS_TENSOR_OPS, QF, QC, GF, GC, KF, KC, VF, VC](input: f32) -> f32
    where HAS_TENSOR_OPS >= 0 and QF >= 0 and GF >= 0 and KF >= 0 and VF >= 0 and (HAS_TENSOR_OPS == 0 or HAS_TENSOR_OPS == 1) and (QF == 0 or QF == 1 or QF == 2) and (GF == 0 or GF == 1 or GF == 2) and (KF == 0 or KF == 1 or KF == 2) and (VF == 0 or VF == 1 or VF == 2) and QC > 0 and GC > 0 and KC > 0 and VC > 0:
    return shared[
        SHARED_F = ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF)))),
        SHARED_C = max(max(QC * (min(1, QF) * (1 - min(1, max(QF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(QF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF)))))))), GC * (min(1, GF) * (1 - min(1, max(GF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(GF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))))))), max(KC * (min(1, KF) * (1 - min(1, max(KF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(KF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF)))))))), VC * (min(1, VF) * (1 - min(1, max(VF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(VF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF)))))))))),
        QP = (min(1, QF) * (1 - min(1, max(QF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(QF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF)))))))),
        GP = (min(1, GF) * (1 - min(1, max(GF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(GF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF)))))))),
        KP = (min(1, KF) * (1 - min(1, max(KF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(KF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF)))))))),
        VP = (min(1, VF) * (1 - min(1, max(VF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))) - min(VF, ((1 - HAS_TENSOR_OPS) * (QF + (1 - min(1, QF)) * (GF + (1 - min(1, GF)) * (KF + (1 - min(1, KF)) * VF))))))))](input)
"#;
    check(source).unwrap();
    // A selected coefficient may actually be zero when its positivity is absent.
    let missing_positive = source.replace("QC > 0", "QC >= 0");
    assert!(check(&missing_positive)
        .unwrap_err()
        .contains("CallContract"));
    // Finite cases retain the exact bound, rather than inventing a larger one.
    let excessive_requirement = source.replace("SHARED_C > 0", "SHARED_C > 1");
    assert!(check(&excessive_requirement)
        .unwrap_err()
        .contains("CallContract"));
}

#[test]
fn unconditional_prefix_covers_guarded_overwrites_without_hiding_earlier_reads() {
    let source = "fn probe[N](flag: i32) -> tensor[N] u32:\n    let mut codes = tensor[N] u32\n    for word in 0..N:\n        codes[word] = u32(0)\n    for word in 0..N:\n        if flag != 0:\n            codes[word] = codes[word] | u32(1)\n    return codes\n";
    check(source).unwrap();
    let partial = source.replacen("for word in 0..N:", "for word in 0..N - 1:", 1);
    assert!(check(&partial).unwrap_err().contains("initialization"));
    let conditional = source.replace(
        "    for word in 0..N:\n        codes[word] = u32(0)",
        "    if flag != 0:\n        for word in 0..N:\n            codes[word] = u32(0)",
    );
    assert!(check(&conditional).unwrap_err().contains("initialization"));
    let earlier_read = source.replace(
        "        codes[word] = u32(0)",
        "        codes[word] = codes[word] | u32(0)",
    );
    assert!(check(&earlier_read).unwrap_err().contains("initialization"));
}

#[test]
fn optional_separation_keeps_structural_maps_and_rejects_dependencies() {
    use super::ir;
    use crate::initialization::VisitSeparation;

    fn outer(source: &str) -> VisitSeparation {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "optional-separation.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let definition = module
            .internal()
            .definitions
            .iter()
            .find(|d| d.name == "probe")
            .unwrap();
        definition
            .body
            .root
            .statements
            .iter()
            .find_map(|statement| match statement {
                ir::Stmt::Loop { separation, .. } => Some(*separation),
                _ => None,
            })
            .unwrap()
    }

    for source in [
        "fn probe[N,M](out: &mut tensor[N,M] f32):\n    for row in 0..N:\n        for col in 0..M:\n            out[row,col] = 1.0\n",
        "fn probe[N,M](out: &mut tensor[N,M] f32):\n    for col in 0..M:\n        for row in 0..N:\n            out[row,col] = 1.0\n",
        "fn probe[N](out: &mut tensor[N*2] f32):\n    for row in 0..N:\n        for sub in 0..2:\n            out[2*row+sub] = 1.0\n",
        "fn probe[N,M](out: &mut tensor[N*M] f32):\n    for row in 0..N:\n        for col in 0..M:\n            out[row*M+col] = 1.0\n",
    ] {
        assert_eq!(outer(source), VisitSeparation::Separated, "{source}");
    }
    for source in [
        "fn probe(out: &mut tensor[2] u32):\n    for i in 0..8:\n        out[i/4] = out[i/4] + u32(1)\n",
        "fn probe(out: &mut tensor[4] f32):\n    for i in 0..3:\n        out[i] = out[i+1]\n",
        "fn probe(flags: &tensor[4] bool, out: &mut tensor[2] f32):\n    for i in 0..4:\n        if flags[i]:\n            out[i%2] = 1.0\n",
        "fn probe(out: &mut tensor[4] f32):\n    for i in 1..4:\n        out[i] = out[i-1]\n",
    ] {
        assert_eq!(outer(source), VisitSeparation::Unproven, "{source}");
    }
}

#[test]
fn guarded_loop_prefix_survives_parameter_contract_export() {
    let consume = "fn consume(x: &tensor[32,2] f32, count: u32) -> f32:\n    let mut total = 0.0\n    for row in 0..32:\n        if i32(row) < i32(count):\n            for column in 0..2:\n                total += x[row,column]\n    return total\n\n";
    check(&format!("{consume}fn probe(requested: i32) -> f32:\n    let count = min(requested,32)\n    let mut x = tensor[32,2] f32\n    let mut total = 0.0\n    if count >= 0:\n        for row in 0..count:\n            for column in 0..2:\n                x[row,column] = 1.0\n        total = consume(x,u32(count))\n    return total\n")).unwrap();
    let invalid = check(&format!("{consume}fn probe() -> f32:\n    let mut x = tensor[32,2] f32\n    for row in 0..2:\n        for column in 0..2:\n            x[row,column] = 1.0\n    return consume(x,u32(3))\n")).unwrap_err();
    assert!(invalid.contains("initialization"), "{invalid}");
    // Two private coordinates must keep their individual guards through both
    // loop boundaries. Filling one column never supplies the other column.
    let nested = "fn consume(x: &tensor[4,4] f32, rows: i32, columns: i32) -> f32:\n    let mut total = 0.0\n    for row in 0..4:\n        for column in 0..4:\n            if row < rows and column < columns:\n                total += x[row,column]\n    return total\n\n";
    check(&format!("{nested}fn probe() -> f32:\n    let mut x = tensor[4,4] f32\n    for row in 0..2:\n        for column in 0..3:\n            x[row,column] = 1.0\n    return consume(x,2,3)\n")).unwrap();
    assert!(check(&format!("{nested}fn probe() -> f32:\n    let mut x = tensor[4,4] f32\n    for row in 0..2:\n        x[row,0] = 1.0\n    return consume(x,2,3)\n")).unwrap_err().contains("initialization"));
}

#[test]
fn guarded_loop_domains_preserve_direction_and_reject_missing_cells() {
    let consume = "fn consume(x: &tensor[4] f32, count: i32) -> f32:\n    let mut result = 0.0\n    for row in 0..4:\n        if not (count <= row):\n            result += x[row]\n    return result\n\n";
    check(&format!("{consume}fn probe() -> f32:\n    let mut x = tensor[4] f32\n    for row in 0..2:\n        x[row] = 1.0\n    return consume(x,2)\n")).unwrap();
    assert!(check(&format!("{consume}fn probe() -> f32:\n    let mut x = tensor[4] f32\n    x[0] = 1.0\n    return consume(x,2)\n")).unwrap_err().contains("initialization"));
    let exact = "fn consume(x: &tensor[4] f32, selected: i32) -> f32:\n    let mut result = 0.0\n    for row in 0..4:\n        if selected == row:\n            result += x[row]\n    return result\n\n";
    check(&format!("{exact}fn probe() -> f32:\n    let mut x = tensor[4] f32\n    x[2] = 1.0\n    return consume(x,2)\n")).unwrap();
    assert!(check(&format!("{exact}fn probe() -> f32:\n    let mut x = tensor[4] f32\n    x[1] = 1.0\n    return consume(x,2)\n")).unwrap_err().contains("initialization"));
    // A modular cast outside its fundamental interval cannot be replaced by
    // the mathematical coordinate. This visits only row zero, not a prefix.
    let wrapped = "fn consume(x: &tensor[4] f32) -> f32:\n    let mut result = 0.0\n    for row in 0..4:\n        if i32(u32(row) + u32(4294967295)) < 0:\n            result += x[row]\n    return result\n\n";
    assert!(check(&format!(
        "{wrapped}fn probe() -> f32:\n    let x = tensor[4] f32\n    return consume(x)\n"
    ))
    .unwrap_err()
    .contains("initialization"));
    // Unsupported coupled predicates retain their guards. They cannot confer
    // permission to read an uninitialized tensor.
    assert!(check("fn consume(x: &tensor[4] f32) -> f32:\n    let mut result = 0.0\n    for row in 0..4:\n        if row * row < 3:\n            result += x[row]\n    return result\n\nfn probe() -> f32:\n    let x = tensor[4] f32\n    return consume(x)\n").unwrap_err().contains("initialization"));
}

/// Reassigned and captured scalar producers keep distinct source-position
/// quantities: a smaller written prefix never supplies a larger read, and a
/// guard's facts never reach a branch where no write occurred.
#[test]
fn scalar_producer_prefixes_never_supply_unwritten_rows() {
    let consume = "fn consume(x: &tensor[32,2] f32, count: u32) -> f32:\n    let mut total = 0.0\n    for row in 0..32:\n        if i32(row) < i32(count):\n            for column in 0..2:\n                total += x[row,column]\n    return total\n\n";
    assert!(check(&format!("{consume}fn probe(requested: i32) -> f32:\n    let mut count = min(requested,32)\n    let old = count\n    count = min(count,16)\n    let mut x = tensor[32,2] f32\n    let mut total = 0.0\n    if count >= 0 and old > count:\n        for row in 0..count:\n            for column in 0..2:\n                x[row,column] = 1.0\n        total = consume(x,u32(old))\n    return total\n")).unwrap_err().contains("initialization"));
    for source in [
        format!("{consume}fn probe(requested: i32) -> f32:\n    let old = min(requested,32)\n    let count = min(old,16)\n    let mut x = tensor[32,2] f32\n    let mut total = 0.0\n    if count >= 0 and old > count:\n        for row in 0..count:\n            for column in 0..2:\n                x[row,column] = 1.0\n        total = consume(x,u32(old))\n    return total\n"),
        format!("{consume}fn probe(requested: i32) -> f32:\n    let old = min(requested,32)\n    let mut x = tensor[32,2] f32\n    if old == 0:\n        for row in 0..old:\n            for column in 0..2:\n                x[row,column] = 1.0\n    return consume(x,u32(1))\n"),
    ] { assert!(check(&source).unwrap_err().contains("initialization")); }
}

#[test]
fn unknown_scalar_producers_are_stable_but_distinct_and_source_ordered() {
    let source = "fn unknown(seed: f32) -> i32:\n    return i32(seed)\n\nfn consume(x: &tensor[4] f32, count: i32) -> f32:\n    let mut total = 0.0\n    for row in 0..4:\n        if row < count:\n            total += x[row]\n    return total\n\n";
    check(&format!("{source}fn probe(seed: f32) -> f32:\n    let count = unknown(seed)\n    let mut x = tensor[4] f32\n    let mut total = 0.0\n    if count >= 0 and count <= 4:\n        for row in 0..count:\n            x[row] = 1.0\n        total = consume(x,count)\n    return total\n")).unwrap();
    assert!(check(&format!("{source}fn probe(seed: f32, other: f32) -> f32:\n    let written = unknown(seed)\n    let read = unknown(other)\n    let mut x = tensor[4] f32\n    let mut total = 0.0\n    if written >= 0 and written <= 4 and read > written and read <= 4:\n        for row in 0..written:\n            x[row] = 1.0\n        total = consume(x,read)\n    return total\n")).unwrap_err().contains("initialization"));
    // A new loop visit and an assignment cannot reuse the prior scalar value;
    // later writes cannot discharge this iteration's earlier read either.
    assert!(check(&format!("{source}fn probe() -> f32:\n    let mut count = 1\n    let mut total = 0.0\n    for round in 0..2:\n        let mut x = tensor[4] f32\n        if count >= 0 and count < 4:\n            for row in 0..count:\n                x[row] = 1.0\n            count = count + 1\n            total += consume(x,count)\n    return total\n")).unwrap_err().contains("initialization"));
    assert!(check(&format!("{source}fn probe(seed: f32) -> f32:\n    let count = unknown(seed)\n    let mut x = tensor[4] f32\n    let mut total = 0.0\n    if count > 0 and count <= 4:\n        total = consume(x,count)\n        for row in 0..count:\n            x[row] = 1.0\n    return total\n")).unwrap_err().contains("initialization"));
}

#[test]
fn actual_scalar_values_reach_dynamic_allocation_axes() {
    check("fn unknown(seed: f32) -> i32:\n    return i32(seed)\n\nfn probe(seed: f32) -> f32:\n    let count = unknown(seed)\n    let mut total = 0.0\n    if count > 0 and count <= 32:\n        let mut x = tensor[count] f32\n        for row in 0..count:\n            x[row] = 1.0\n        for row in 0..count:\n            total += x[row]\n    return total\n").unwrap();
    assert!(check("fn unknown(seed: f32) -> i32:\n    return i32(seed)\n\nfn probe(seed: f32) -> f32:\n    let count = unknown(seed)\n    let mut total = 0.0\n    if count > 1 and count <= 32:\n        let mut x = tensor[count] f32\n        for row in 0..count - 1:\n            x[row] = 1.0\n        total = x[count - 1]\n    return total\n").unwrap_err().contains("initialization"));
}

#[test]
fn private_coordinate_projection_preserves_known_prefix_and_actual_view_axes() {
    // The joined column is private to the callee. Its may-read footprint can
    // cover both columns without widening the independently known row prefix.
    let consume = "fn consume(x: &tensor[4,2] f32, count: u32, flip: bool) -> f32:\n    let live = i32(count)\n    let mut total = 0.0\n    if live > 0:\n        for row in 0..live:\n            for column in 0..2:\n                let mut at = column\n                if flip:\n                    at = 1 - column\n                total += x[row,at]\n    return total\n\n";
    check(&format!("{consume}fn probe(flip: bool) -> f32:\n    let mut x = tensor[4,2] f32\n    for row in 0..2:\n        for column in 0..2:\n            x[row,column] = 1.0\n    return consume(x,u32(2),flip)\n")).unwrap();
    for producer in [
        "for column in 0..2:\n        x[0,column] = 1.0",
        "for row in 0..2:\n        x[row,0] = 1.0",
    ] {
        assert!(check(&format!("{consume}fn probe(flip: bool) -> f32:\n    let mut x = tensor[4,2] f32\n    {producer}\n    return consume(x,u32(2),flip)\n")).unwrap_err().contains("initialization"));
    }
    // A view's logical two-column geometry is distinct from its four-column
    // backing root. Projection must use the former before the call maps it.
    check(&format!("{consume}fn probe(flip: bool) -> f32:\n    let mut x = tensor[2,4] f32\n    for column in 0..4:\n        x[0,column] = 1.0\n    let view = reshape(x,(4,2))\n    return consume(view,u32(2),flip)\n")).unwrap();
    assert!(check(&format!("{consume}fn probe(flip: bool) -> f32:\n    let mut x = tensor[2,4] f32\n    for column in 0..3:\n        x[0,column] = 1.0\n    let view = reshape(x,(4,2))\n    return consume(view,u32(2),flip)\n")).unwrap_err().contains("initialization"));
}

#[test]
fn incremental_writes_preserve_scoped_exports_and_required_reads() {
    // Each row accumulates independent stores. Both branch arms finish it;
    // loop and helper export must retain their source-position write meaning.
    let publish = "fn publish(a: &mut tensor[2,4] f32, flags: &tensor[2] i32):\n    for row in 0..2:\n        for column in 0..2:\n            a[row,column] = 1.0\n        let prior = a[row,0]\n        if flags[row] >= 0:\n            a[row,2] = prior\n            a[row,3] = prior\n        else:\n            for column in 2..4:\n                a[row,column] = prior\n\n";
    check(&format!("{publish}fn probe(flags: &tensor[2] i32) -> tensor[2,4] f32:\n    let mut a = tensor[2,4] f32\n    publish(a, flags)\n    return a\n")).unwrap();

    // Optional publication remains guarded across a helper exit and two
    // branch joins; its matching guarded read is sufficient, not global Full.
    let guarded = "fn publish(a: &mut tensor[2] f32, choose: bool):\n    if choose:\n        a[0] = 1.0\n        a[1] = 2.0\n\nfn consume(a: &tensor[2] f32, choose: bool) -> f32:\n    let mut result = 0.0\n    if choose:\n        result = a[0] + a[1]\n    return result\n\n";
    check(&format!("{guarded}fn probe(choose: bool) -> f32:\n    let mut a = tensor[2] f32\n    publish(a, choose)\n    return consume(a, choose)\n")).unwrap();
}

#[test]
fn incremental_writes_do_not_promote_gaps_or_reorder_reads() {
    for source in [
        "fn publish(a: &mut tensor[4] f32, choose: bool):\n    a[0] = 1.0\n    a[1] = 1.0\n    if choose:\n        a[2] = 1.0\n        a[3] = 1.0\n    else:\n        a[3] = 1.0\n\nfn probe(choose: bool) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    publish(a, choose)\n    return a\n",
        "fn probe() -> f32:\n    let mut a = tensor[4] f32\n    a[0] = 1.0\n    for i in 1..4:\n        let prior = a[i]\n        a[i] = a[i-1]\n    return a[3]\n",
        "fn publish(a: &mut tensor[2] f32, choose: bool):\n    if choose:\n        a[0] = 1.0\n        a[1] = 1.0\n\nfn probe(choose: bool) -> f32:\n    let mut a = tensor[2] f32\n    publish(a, choose)\n    return a[1]\n",
    ] {
        let error = check(source).unwrap_err();
        assert!(error.contains("initialization"), "{error}");
    }
}

#[test]
fn typed_float_cast_effect_parameters_retain_actual_copied_leaf() {
    let fill = "fn fill(a: &mut tensor[4] f32, source: f32):\n    let at = i32(source)\n    if at >= 0 and at < 4:\n        a[at] = 1.0\n\n";
    let probe = "fn probe(source: f32) -> f32:\n    let mut current = source\n    let old = current\n    let mut a = tensor[4] f32\n    fill(a, old)\n    current = current + 1.0\n    let at = i32(old)\n    let mut result = 0.0\n    if at >= 0 and at < 4:\n        result = a[at]\n    return result\n";
    check(&format!("{fill}{probe}")).unwrap();
    let error = check(&format!(
        "{fill}{}",
        probe.replace("let at = i32(old)", "let at = i32(current)")
    ))
    .unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn typed_boolean_word_effect_parameters_keep_leaf_and_guard() {
    let fill = "fn fill(a: &mut tensor[2] f32, choice: bool):\n    let at = i32(choice)\n    if at >= 0 and at < 2:\n        a[at] = 1.0\n\n";
    let probe = "fn probe(choice: bool) -> f32:\n    let mut a = tensor[2] f32\n    fill(a, choice)\n    let at = i32(choice)\n    let mut result = 0.0\n    if at >= 0 and at < 2:\n        result = a[at]\n    return result\n";
    check(&format!("{fill}{probe}")).unwrap();
    let error = check(&format!(
        "{fill}{}",
        probe.replace("let at = i32(choice)", "let at = i32(not choice)")
    ))
    .unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn partial_helper_successful_exit_domain_preserves_input_obligations() {
    // F32-to-I32 publication has a partial runtime domain. Both successful
    // branches export writes over the same actual cast, not an invented leaf.
    let fill = "fn fill(a: &mut tensor[4] f32, source: f32, flip: bool):\n    let count = i32(source)\n    if count >= 0 and count <= 4:\n        if flip:\n            for row in 0..count:\n                a[row] = 1.0\n        else:\n            for row in 0..count:\n                a[row] = 2.0\n\n";
    let probe = "fn probe(source: f32, flip: bool) -> f32:\n    let mut a = tensor[4] f32\n    fill(a, source, flip)\n    let count = i32(source)\n    let mut total = 0.0\n    if count >= 0 and count <= 4:\n        for row in 0..count:\n            total += a[row]\n    return total\n";
    check(&format!("{fill}{probe}")).unwrap();

    // The other successful exit may leave the entire destination untouched.
    let conditional = "fn fill(a: &mut tensor[4] f32, source: f32, flip: bool):\n    if flip:\n        let count = i32(source)\n        if count >= 0 and count <= 4:\n            for row in 0..count:\n                a[row] = 1.0\n\n";
    let error = check(&format!("{conditional}{probe}")).unwrap_err();
    assert!(error.contains("initialization"), "{error}");

    // Input reads precede both exported writes and successful-return facts.
    let reads_first = fill
        .replace("a[row] = 1.0", "a[row] = a[row] + 1.0")
        .replace("a[row] = 2.0", "a[row] = a[row] + 2.0");
    let error = check(&format!("{reads_first}{probe}")).unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn admitted_dynamic_point_bounds_preserve_initialization_authority() {
    check(&format!(
        "{FILL}fn probe(at: i32) -> f32:\n    let mut a = tensor[4] f32\n    fill(a)\n    return a[at]\n"
    ))
    .unwrap();

    // A successful in-bounds point read narrows its coordinate to the root,
    // not to the initialized prefix. It cannot manufacture the missing cells.
    let error = check(&format!(
        "{FILL}fn probe(at: i32) -> f32:\n    let mut a = tensor[4] f32\n    fill(a[:2])\n    return a[at]\n"
    ))
    .unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn admitted_variable_distance_reads_support_reduction() {
    check(&format!(
        "{FILL}fn probe(distance: i32) -> f32:\n    let mut a = tensor[4] f32\n    fill(a)\n    let mut total = 0.0\n    for row in 0..4:\n        total += a[row + distance]\n    return total\n"
    ))
    .unwrap();

    // The actual reduction shape changes distance between visits. Successful
    // source/target checks belong to each reached iteration, not a fixed bound.
    check("fn probe() -> f32:\n    let mut a = tensor[4] f32\n    for lane in 0..4:\n        a[lane] = 1.0\n    let mut distance = 2\n    for step in 0..4:\n        if distance > 0:\n            for lane in 0..distance:\n                a[lane] = a[lane] + a[lane + distance]\n            distance = distance / 2\n    return a[0]\n").unwrap();
}

#[test]
fn checked_target_bounds_reach_rhs_without_initializing_prior_value() {
    // The target's checked point is reached before evaluating its RHS. That
    // successful access also establishes the reflected source is in bounds.
    check(&format!(
        "{FILL}fn probe(at: i32) -> f32:\n    let mut source = tensor[4] f32\n    fill(source)\n    let mut target = tensor[4] f32\n    target[at] = source[3 - at]\n    return target[at]\n"
    ))
    .unwrap();

    // Compound assignment still reads the old destination. Target bounds
    // are not a produced write or permission to consume uninitialized bytes.
    let error = check(
        "fn probe(at: i32) -> f32:\n    let mut target = tensor[4] f32\n    target[at] += 1.0\n    return target[at]\n",
    )
    .unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

#[test]
fn checked_dynamic_view_bounds_preserve_full_and_partial_sources() {
    check(&format!(
        "{FILL}fn probe(first: i32, end: i32) -> f32:\n    let mut a = tensor[4] f32\n    fill(a)\n    let view = a[first:end]\n    let owned = to_owned(view)\n    return owned[0]\n"
    ))
    .unwrap();

    let error = check(&format!(
        "{FILL}fn probe(first: i32, end: i32) -> f32:\n    let mut a = tensor[4] f32\n    fill(a[:2])\n    let view = a[first:end]\n    let owned = to_owned(view)\n    return owned[0]\n"
    ))
    .unwrap_err();
    assert!(error.contains("initialization"), "{error}");
}

/// Entry instantiation keeps each contract symbol's sort: a typed scalar
/// parameter's value stays a scalar symbol, so applying the contract at a
/// call substitutes the scalar argument for it.
#[test]
fn instantiated_contract_symbols_keep_their_parameter_sorts() {
    use crate::expr::SymbolSort;
    use crate::initialization::ParameterPart;
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "initialization.seismic".into(),
        text: "fn scaled(a: &mut tensor[4] f32, s: f32, n: i32):\n    if s > 0.0 and n > 0:\n        for i in 0..4:\n            a[i] = s\n\nfn probe(s: f32) -> f32:\n    let mut a = tensor[4] f32\n    for i in 0..4:\n        a[i] = 0.0\n    scaled(a, s, 2)\n    return a[0]\n".into(),
    }]))
    .unwrap();
    let logical = module
        .entry(module.entry_named("probe").unwrap(), &Default::default())
        .unwrap();
    let mut scalars = 0;
    for (_, function) in logical.program().functions() {
        for (symbol, part) in &function.initialization().symbols {
            let sort = logical.arena().symbol_sort(*symbol);
            match part {
                ParameterPart::Scalar(_) => {
                    scalars += 1;
                    assert!(matches!(sort, SymbolSort::Scalar(_)), "{sort:?}");
                }
                _ => assert!(!matches!(sort, SymbolSort::Scalar(_)), "{sort:?}"),
            }
        }
    }
    assert!(scalars > 0, "the contract binds the scalar parameter");
}
