use crate::checked::{check_source, SourceFile, SourceSet};
fn check(source: &str) -> Result<(), String> {
    check_source(SourceSet::new(vec![SourceFile {
        path: "ownership.seismic".into(),
        text: source.into(),
    }]))
    .map(|_| ())
    .map_err(|e| e.to_string())
}
#[test]
fn owned_results_cannot_launder_borrowed_tensor_leaves() {
    for source in [
        "fn escape(x: &tensor[2] f32) -> tensor[2] f32:\n    return x[:]\n",
        "fn escape(x: &tensor[2] f32) -> (tensor[2] f32, i32):\n    return (x[:], 1)\n",
        "fn escape(x: &tensor[2] f32) -> (tensor[2] f32, i32):\n    let pair = (x[:], 1)\n    return pair\n",
        "fn escape(x: &tensor[2] f32) -> tensor[2] f32:\n    let pair = (x[:], 1)\n    let (view, number) = pair\n    return view\n",
    ] {
        assert!(check(source).unwrap_err().contains("borrowed"), "{source}");
    }
}
#[test]
fn tuple_moves_consume_owned_tensor_leaves() {
    for source in [
        "fn probe() -> f32:\n    let mut a = tensor[2] f32\n    a[:] = zeros_like(a)\n    let pair = (a, 1)\n    return a[0]\n",
        "fn probe() -> f32:\n    let mut a = tensor[2] f32\n    a[:] = zeros_like(a)\n    let pair = (a, 1)\n    let copy = pair\n    let (again, number) = pair\n    return again[0]\n",
    ] {
        assert!(check(source).unwrap_err().contains("moved"), "{source}");
    }
}

#[test]
fn tuple_signatures_keep_each_tensor_access_mode() {
    use crate::checked::{SignatureType, TensorAccess};
    let module=check_source(SourceSet::new(vec![SourceFile{path:"tuple-parameters.seismic".into(),text:"fn probe(pair: (&tensor[2] f32, &mut tensor[2] f32)):\n    let (source, destination) = pair\n    destination[:] = to_owned(source)\n".into()}])).unwrap();
    let info = module
        .entries()
        .iter()
        .find(|entry| entry.name == "probe")
        .unwrap();
    let SignatureType::Tuple(parts) = &info.parameter_types[0].1 else {
        panic!("tuple parameter")
    };
    assert!(matches!(
        parts[0],
        SignatureType::Tensor {
            access: TensorAccess::Shared,
            ..
        }
    ));
    assert!(matches!(
        parts[1],
        SignatureType::Tensor {
            access: TensorAccess::Mutable,
            ..
        }
    ));
}

#[test]
fn owned_view_transfer_and_branch_loop_moves_preserve_authority() {
    check("fn probe(x: tensor[2,2] f32) -> tensor[2,2] f32:\n    return x.T\n").unwrap();
    check("fn probe(x: &tensor[2,2] f32) -> f32:\n    let mut y = to_owned(x)\n    for i in 0..3:\n        y = y.T\n    return y[0,1]\n").unwrap();
    for source in [
        "fn probe(choose: bool) -> f32:\n    let mut a = tensor[2] f32\n    a[:] = zeros_like(a)\n    if choose:\n        let pair = (a, 1)\n    return a[0]\n",
        "fn probe() -> f32:\n    let mut a = tensor[2] f32\n    a[:] = zeros_like(a)\n    for i in 0..2:\n        let pair = (a, 1)\n    return 0.0\n",
    ] { assert!(check(source).is_err(),"{source}"); }
}

#[test]
fn tuple_owned_call_consumes_leaves_and_rejects_borrowed_laundering() {
    let helper="fn take(pair: (tensor[2] f32, i32)) -> f32:\n    let (value, number) = pair\n    return value[0]\n\n";
    assert!(check(&format!(
        "{helper}fn probe(x: &tensor[2] f32) -> f32:\n    return take((x[:], 1))\n"
    ))
    .unwrap_err()
    .contains("borrowed"));
    assert!(check(&format!("{helper}fn probe() -> f32:\n    let mut a = tensor[2] f32\n    a[:] = zeros_like(a)\n    let first = take((a, 1))\n    return a[0]\n")).unwrap_err().contains("moved"));
}

#[test]
fn mixed_tuple_assignment_writes_borrowed_leaf_and_replaces_owned_leaf() {
    use crate::entry::ElementBindings;
    use crate::interp::{Arg, Interpreter, TensorData};
    use crate::types::DType;
    let module=check_source(SourceSet::new(vec![SourceFile{path:"mixed-tuple.seismic".into(),text:"fn probe(dst: &mut tensor[2] f32):\n    let mut first = tensor[2] f32\n    first[:] = zeros_like(first)\n    let mut pair = (dst[:], first)\n    let mut next = tensor[2] f32\n    next[:] = ones_like(next)\n    pair = (zeros_like(next), next)\n".into()}])).unwrap();
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
    assert_eq!(actual.tensor().read(0).unwrap(), 0.);
    assert_eq!(actual.tensor().read(1).unwrap(), 0.);
}

#[test]
fn tuple_shared_leaf_does_not_inherit_sibling_write_permission() {
    for source in [
        "fn probe(pair: (&tensor[2] f32, &mut tensor[2] f32)):\n    let (source, destination) = pair\n    source[0] = 1.0\n",
        "fn fill(dst: &mut tensor[2] f32):\n    dst[:] = zeros_like(dst)\n\nfn probe(pair: (&tensor[2] f32, &mut tensor[2] f32)):\n    let (source, destination) = pair\n    fill(source)\n",
    ] {assert!(check(source).is_err(),"{source}");}
}

#[test]
fn live_shared_view_prevents_moving_its_owned_backing() {
    for source in [
        "fn probe() -> f32:\n    let mut owner = tensor[2] f32\n    owner[:] = zeros_like(owner)\n    let view = owner[:]\n    let moved = owner\n    return view[0]\n",
        "fn probe() -> tensor[2] f32:\n    let mut owner = tensor[2] f32\n    owner[:] = zeros_like(owner)\n    let view = owner[:]\n    return owner\n",
    ] {assert!(check(source).unwrap_err().contains("borrow"),"{source}");}
}

#[test]
fn an_owned_tensor_cannot_be_duplicated_in_a_product() {
    for statement in ["let pair = (owner, owner)\n    return zeros_like(tensor[2] f32), zeros_like(tensor[2] f32)","return owner, owner"] {
        let source=format!("fn probe() -> (tensor[2] f32, tensor[2] f32):\n    let mut owner = tensor[2] f32\n    owner[:] = zeros_like(owner)\n    {statement}\n");
        assert!(check(&source).is_err(),"{source}");
    }
}

#[test]
fn tuple_tensor_carry_requires_the_same_inductive_initialized_region() {
    let source="fn probe() -> f32:\n    let mut initial = tensor[2] f32\n    initial[:] = zeros_like(initial)\n    let mut pair = (initial, 0)\n    for i in 0..2:\n        let (current, number) = pair\n        let missing_on_second_visit = current[1]\n        let mut next = tensor[2] f32\n        next[0] = 1.0\n        pair = (next, number)\n    return 0.0\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}

#[test]
fn overlapping_store_preserves_root_descriptor_and_snapshots_rhs() {
    use crate::entry::ElementBindings;
    use crate::interp::{Arg, Interpreter, OutcomeValue, TensorData};
    use crate::reference_math::ReferenceScalar;
    use crate::types::DType;
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "overlapping-store.seismic".into(),
        text: "fn probe(x: &mut tensor[3] f32) -> f32:\n    x[1:3] = x[0:2]\n    return x[2]\n"
            .into(),
    }]))
    .unwrap();
    let entry = module
        .entry(
            module.entry_named("probe").unwrap(),
            &ElementBindings::default(),
        )
        .unwrap();
    let mut oracle = Interpreter::new(&entry);
    let input = oracle.add_tensor(TensorData::dense(DType::F32, vec![3], vec![1., 2., 3.]));
    let outcome = oracle.run(&[Arg::Tensor(input)]).unwrap();
    assert!(
        matches!(outcome.results().next().unwrap().value(),OutcomeValue::Scalar(ReferenceScalar::F32(value)) if value==2.0f32.to_bits())
    );
    let actual = outcome.inputs().next().unwrap();
    assert_eq!(actual.tensor().shape(), &[3]);
    assert_eq!(actual.tensor().read(1).unwrap(), 1.);
    assert_eq!(actual.tensor().read(2).unwrap(), 2.);
}

#[test]
fn repeated_allocation_definition_does_not_inherit_a_prior_instances_writes() {
    let source="fn probe() -> f32:\n    let mut carry = tensor[2] f32\n    carry[:] = zeros_like(carry)\n    for i in 0..3:\n        let previous = carry[1]\n        let mut next = tensor[2] f32\n        next[0] = previous\n        if i == 0:\n            next[1] = 2.0\n        carry = next\n    return 0.0\n";
    assert!(check(source).unwrap_err().contains("initialization"));
}

/// A host port is part of the entry's signature only: its access is the
/// host's, the body never names it, and it may be a stored form the device
/// cannot write.
#[test]
fn host_ports_carry_the_hosts_access_and_no_body_names_them() {
    use crate::checked::{SignatureType, TensorAccess};
    let source = "fn request(served: &host tensor[4] i32, staged: &host mut tensor[2, 32] gguf_q8_0, out: &mut tensor[1] i32):\n    out[0] = 1\n";
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "host-ports.seismic".into(),
        text: source.into(),
    }]))
    .unwrap();
    let info = module.entries().iter().find(|entry| entry.name == "request").unwrap();
    let access = |index: usize| match &info.parameter_types[index].1 {
        SignatureType::Tensor { access, .. } => *access,
        other => panic!("{other:?}"),
    };
    assert_eq!(access(0), TensorAccess::Host { write: false });
    assert_eq!(access(1), TensorAccess::Host { write: true });
    assert!(check("fn request(staged: &mut tensor[2, 32] gguf_q8_0):\n    return\n")
        .unwrap_err()
        .contains("decode-only"));
    for body in ["    out[0] = served[0]\n", "    served[0] = 1\n"] {
        let source = format!(
            "fn request(served: &host mut tensor[4] i32, out: &mut tensor[1] i32):\n{body}"
        );
        assert!(check(&source).unwrap_err().contains("host port"), "{body}");
    }
    assert!(check("fn request(pair: (&host tensor[4] i32, &tensor[4] i32)):\n    return\n")
        .unwrap_err()
        .contains("host port"));
    assert!(check("fn request(served: &host tensor[4] i32):\n    return\n\nfn caller(served: &tensor[4] i32):\n    request(served)\n")
        .unwrap_err()
        .contains("host ports"));
}

/// An awaited port is a device borrow (shared, or mutable with `mut`) the
/// body reads like any other; it is a parameter of its own.
#[test]
fn awaited_ports_are_borrows_of_their_own() {
    use crate::checked::{SignatureType, TensorAccess};
    let source = "fn parked(answer: &await tensor[1] i32, words: &await mut tensor[2] i32, out: &mut tensor[1] i32):\n    words[1] = answer[0]\n    out[0] = words[0]\n";
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "awaited-ports.seismic".into(),
        text: source.into(),
    }]))
    .unwrap();
    let info = module.entries().iter().find(|entry| entry.name == "parked").unwrap();
    let access = |index: usize| match &info.parameter_types[index].1 {
        SignatureType::Tensor { access, .. } => *access,
        other => panic!("{other:?}"),
    };
    assert_eq!(access(0), TensorAccess::Shared);
    assert_eq!(access(1), TensorAccess::Mutable);
    assert!(check("fn parked(pair: (&await tensor[4] i32, &tensor[4] i32)):\n    return\n")
        .unwrap_err()
        .contains("awaited port"));
}

#[test]
fn awaited_entries_cannot_be_called_by_definitions() {
    for mutable in ["", "mut "] {
        let helper = format!("fn inner(words: &await {mutable}tensor[1] i32):\n    return\n\n");
        let caller = format!("fn outer(words: &{mutable}tensor[1] i32):\n    inner(words)\n");
        let error = check(&(helper.clone() + &caller)).unwrap_err();
        assert!(error.contains("awaited ports") && error.contains("no definition calls"), "{error}");
        // The same ordinary borrow remains a callable helper.
        check(&(helper.replace("&await ", "&") + &caller)).unwrap();
    }
}

#[test]
fn overlapping_definitions_agree_on_awaited_ports() {
    for mutable in ["", "mut "] {
        let awaited = format!("fn same(words: &await {mutable}tensor[1] i32):\n    return\n\n");
        let ordinary = awaited.replace("&await ", "&");
        for source in [awaited.clone() + &ordinary, ordinary + &awaited] {
            let error = check(&source).unwrap_err();
            assert!(error.contains("different binding"), "{error}");
        }
        check(&(awaited.clone() + &awaited)).unwrap();
    }
}
