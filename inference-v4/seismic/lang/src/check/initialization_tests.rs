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
fn rejects_the_seven_whole_root_promotion_counterexamples() {
    let cases = [
        ("read before later fill", "fn bad[N](a: &mut tensor[N] f32):\n    let prior = a[0]\n    fill(a)\n\nfn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    bad(a)\n    return a\n"),
        ("compound assignment reads old state", "fn bad[N](a: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        a[i] += 1.0\n\nfn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    bad(a)\n    return a\n"),
        ("slice fill is not whole root", "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    fill(a[:2])\n    return a\n"),
        ("conditional forwarding keeps its path", "fn maybe[N](a: &mut tensor[N] f32, choose: bool):\n    if choose:\n        fill(a)\n\nfn probe(choose: bool) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    maybe(a, choose)\n    return a\n"),
        ("zero trip writes nothing", "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    for i in 0..0:\n        a[:] = zeros_like(a)\n    return a\n"),
        ("first point is not whole loop image", "fn probe() -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    parallel for i in 0..4:\n        a[i] = 0.0\n        let future = a[3]\n    return a\n"),
        ("diagonal is not cartesian coverage", "fn diagonal[N](a: &mut tensor[N, N] f32):\n    parallel for i in 0..N:\n        a[i, i] = 0.0\n\nfn probe() -> tensor[4, 4] f32:\n    let mut a = tensor[4, 4] f32\n    diagonal(a)\n    return a\n"),
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
    ] { check(&format!("{FILL}{source}")).unwrap_or_else(|error|panic!("{source}\n{error}")); }
}

#[test]
fn complete_owned_carry_survives_rectangular_transpose_reshape() {
    check("fn probe(x: &tensor[2,3] f32) -> tensor[2,3] f32:\n    let mut y = load(x)\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n        y[0,1] = 17.0\n    return y\n").unwrap();
}

#[test]
fn initialized_point_survives_rectangular_transpose_reshape_carry() {
    check("fn probe(x: &tensor[2,3] f32) -> tensor[1] f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n    let mut result = tensor[1] f32\n    result[0] = y[0,0]\n    return result\n").unwrap();
    check("fn probe(x: &tensor[2,3] f32, choose: bool) -> f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..2:\n        if choose:\n            y = reshape(y.T, (2,3))\n    return y[0,0]\n").unwrap();
    check("fn probe(x: &tensor[2,3] f32) -> f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..0:\n        y = reshape(y.T, (2,3))\n    return y[0,0]\n").unwrap();
}

#[test]
fn partial_owned_carry_is_not_promoted_through_transpose_reshape() {
    let error = check("fn probe(x: &tensor[2,3] f32) -> tensor[2,3] f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = x[0,0]\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n        let missing = y[0,1]\n    return load(x)\n").unwrap_err();
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
    ] { check(source).unwrap_or_else(|error|panic!("{source}\n{error}")); }
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
    let error = check(source).expect_err("different visits can take opposite arms and write out[1]");
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
fn target_collective_control_can_depend_on_participant_local_quantity() {
    check("fn probe(flag: &tensor[32] bool, x: &tensor[32] f32, out: &mut tensor[32] f32):\n    parallel for i in 0..32:\n        out[i] = x[i]\n\nlower probe(flag: &tensor[32] bool, x: &tensor[32] f32, out: &mut tensor[32] f32)\n    for metal requires metal.subgroup:\n    parallel for i in 0..32:\n        let mut q = i - 1\n        if flag[i]:\n            q = i + 1\n        if q >= 0:\n            out[i] = metal.subgroup.simd_sum(x[i])\n        else:\n            out[i] = x[i]\n").unwrap();

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
    let source="fn probe(input: bool) -> tensor[4] f32:\n    let mut choose = input\n    let mut a = tensor[4] f32\n    if choose:\n        fill(a)\n    choose = not choose\n    if choose:\n        let missing = a[0]\n    return zeros_like(a)\n";
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
    check("fn probe() -> tensor[2] f32:\n    let mut a = tensor[2] f32\n    let mut view = a[:]\n    view[0] = 1.0\n    let current = view[0]\n    view[1] = current\n    return load(view)\n").unwrap();
}

#[test]
fn optional_contract_uses_actual_input_and_preserves_reference_output() {
    use crate::entry::ElementBindings;
    use crate::initialization::{
        InitializationArgument as Argument, InitializationContext, InitializationPhase,
        InitializationState,
    };
    let source="fn reference(a: &mut tensor[4] f32):\n    a[:] = zeros_like(a)\n\nfn stronger(a: &mut tensor[4] f32):\n    let old = a[0]\n    a[:] = zeros_like(a)\n\nfn weaker(a: &mut tensor[4] f32):\n    a[0] = 0.0\n\nfn probe(a: &mut tensor[4] f32):\n    reference(a)\n    stronger(a)\n    weaker(a)\n";
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
    let source="fn maybe(a: &mut tensor[4] f32, source: f32):\n    if source > 0.0:\n        a[:] = zeros_like(a)\n\nfn probe(source: f32) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    maybe(a, source)\n    return a\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}

#[test]
fn nested_successful_return_does_not_gain_later_writes() {
    let source="fn maybe(a: &mut tensor[4] f32, outer: bool, inner: bool):\n    if outer:\n        if inner:\n            return\n    a[:] = zeros_like(a)\n\nfn probe(outer: bool, inner: bool) -> tensor[4] f32:\n    let mut a = tensor[4] f32\n    maybe(a, outer, inner)\n    return a\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}

#[test]
fn tensor_carries_require_a_preserved_initialized_region() {
    let unsafe_source="fn probe() -> tensor[2] f32:\n    let mut a = zeros_like(tensor[2] f32)\n    for i in 0..2:\n        let missing_on_second_visit = a[1]\n        let mut next = tensor[2] f32\n        next[0] = 1.0\n        a = next\n    return zeros_like(a)\n";
    assert!(check(unsafe_source).unwrap_err().contains("initializ"));
    let partial="fn probe() -> f32:\n    let mut a = tensor[2] f32\n    a[0] = 1.0\n    for i in 0..4:\n        let current = a[0]\n        let mut next = tensor[2] f32\n        next[0] = current + 1.0\n        a = next\n    return a[0]\n";
    check(partial).unwrap();
    let one_visit = unsafe_source.replace("0..2:", "0..1:");
    check(&one_visit).unwrap();
    let zero_visits = unsafe_source.replace("0..2:", "0..0:");
    check(&zero_visits).unwrap();
    check("fn replace(a: &tensor[2] f32) -> tensor[2] f32:\n    let mut carry = tensor[2] f32\n    for i in 0..2:\n        carry = zeros_like(a)\n    return carry\n").unwrap();
}

#[test]
fn prior_iterations_do_not_inherit_current_dynamic_predicate() {
    let source="fn maybe(a: &mut tensor[1] f32, choose: bool):\n    if choose:\n        a[0] = 1.0\n\nfn probe(flags: &tensor[2] bool) -> tensor[2] f32:\n    let mut a = tensor[2] f32\n    for i in 0..2:\n        let choose = flags[i]\n        maybe(a[i:i+1], choose)\n        if choose and i > 0:\n            let missing = a[i-1]\n    return zeros_like(a)\n";
    assert!(check(source).unwrap_err().contains("initialization"));
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
