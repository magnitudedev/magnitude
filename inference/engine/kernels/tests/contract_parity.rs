//! T1 parity test: compares compile-time generated `EntryContract` methods
//! (`contract_admit`, `contract_default_params`, `contract_sizing`) bit-for-bit
//! against `CheckedModule` / `native_implementation_at` /
//! `NativeGraphCheckedEntry` across representative entries and all 4 backends.

use seismic::{
    BackendName, ContractError, Element, EntryContract, NativeSpecialization,
    NativeSpecializationError, OperandFormSpec, OperandRole, ScratchSpec, TensorResultSpec,
};
use seismic_lang::checked::NativeOperand;

const ALL_BACKENDS: [BackendName; 4] = [
    BackendName::Cpu,
    BackendName::Metal,
    BackendName::Cuda,
    BackendName::Vulkan,
];

fn assert_contract_parity<E: EntryContract>(
    label: &str,
    elements: &[(&str, Element)],
    dims: &[(&str, u64)],
) {
    for backend in ALL_BACKENDS {
        let impl_opt = seismic::generated::native_implementation_at::<E>(backend, elements)
            .expect("bundle decodes cleanly");

        let checked_bind = seismic::generated::NativeGraphCheckedEntry::bind::<E>(backend, elements)
            .expect("bundle decodes cleanly");

        let Some(implementation) = impl_opt else {
            let err = E::contract_default_params(backend, None, elements, dims).unwrap_err();
            assert!(
                matches!(
                    err,
                    ContractError::NoImplementation(_) | ContractError::UnsupportedElements
                ),
                "{label} ({backend:?}): expected NoImplementation/UnsupportedElements, got {err:?}"
            );
            let err = E::contract_admit(backend, None, elements, dims, &[]).unwrap_err();
            assert!(
                matches!(
                    err,
                    ContractError::NoImplementation(_) | ContractError::UnsupportedElements
                ),
                "{label} ({backend:?}): expected NoImplementation/UnsupportedElements, got {err:?}"
            );
            continue;
        };

        let Ok(checked) = checked_bind else {
            let err = E::contract_default_params(backend, None, elements, dims).unwrap_err();
            assert_eq!(
                err,
                ContractError::UnsupportedElements,
                "{label} ({backend:?}): element domain rejected by CheckedModule"
            );
            continue;
        };

        let dim_lookup = |name: &str| {
            dims.iter()
                .find(|(candidate, _)| *candidate == name)
                .map(|(_, value)| *value)
        };

        let mut statics = NativeSpecialization::new();
        for name in &implementation.statics {
            let value = dim_lookup(name).unwrap_or_else(|| {
                panic!("{label} ({backend:?}): test omitted static dimension `{name}`")
            });
            statics = statics.with_static(name.clone(), value);
        }

        // 1. Compare contract_default_params against default_specialization_without_device.
        let expected_default = implementation.default_specialization_without_device(&statics);
        let actual_default = E::contract_default_params(backend, None, elements, dims);

        let (spec, device_dependent) = match expected_default {
            None => {
                // Implementation's defaults or constraint read `device.NAME`.
                assert!(
                    matches!(actual_default, Err(ContractError::DeviceDependent(_))),
                    "{label} ({backend:?}): expected DeviceDependent, got {actual_default:?}"
                );
                let constraint_reads_device =
                    implementation.constraint.as_ref().is_some_and(|constraint| {
                        let mut reads = Vec::new();
                        constraint.device_reads(&mut reads);
                        !reads.is_empty()
                    });
                if constraint_reads_device {
                    continue;
                }
                let mut first = None;
                let walked = implementation
                    .at_device(&seismic::device::BackendFloor(backend))
                    .walk_admissible(&statics, |configuration| {
                        first = Some(configuration.clone());
                        false
                    });
                match (walked, first) {
                    (Ok(()), Some(spec)) => (spec, true),
                    _ => continue,
                }
            }
            Some(Err(NativeSpecializationError::Inadmissible)) => {
                assert_eq!(
                    actual_default,
                    Err(ContractError::DomainViolation),
                    "{label} ({backend:?}): expected DomainViolation"
                );
                continue;
            }
            Some(Err(NativeSpecializationError::Default { .. })) => {
                assert!(
                    matches!(actual_default, Err(ContractError::NoDefault(_))),
                    "{label} ({backend:?}): expected NoDefault, got {actual_default:?}"
                );
                continue;
            }
            Some(Err(other)) => {
                panic!("{label} ({backend:?}): unexpected default_specialization error: {other}");
            }
            Some(Ok(spec)) => (spec, false),
        };

        let spec_ref = &spec;
        let expected_params: Vec<(&str, u64)> = implementation
            .params
            .iter()
            .map(|p| {
                (
                    p.name.as_str(),
                    spec_ref
                        .param(&p.name)
                        .expect("default specialization binds every entry param"),
                )
            })
            .chain(
                implementation
                    .launches
                    .iter()
                    .enumerate()
                    .flat_map(|(launch_idx, launch)| {
                        launch.params.iter().map(move |p| {
                            (
                                p.name.as_str(),
                                spec_ref
                                    .launch_param(launch_idx, &p.name)
                                    .expect("default specialization binds every launch param"),
                            )
                        })
                    }),
            )
            .collect();

        let params: Vec<(&str, u64)> = if device_dependent {
            expected_params
        } else {
            let params = actual_default.unwrap_or_else(|err| {
                panic!("{label} ({backend:?}): contract_default_params failed: {err:?}")
            });
            assert_eq!(
                params, expected_params,
                "{label} ({backend:?}): default params mismatch"
            );
            params
        };

        // 2. Compare contract_admit against implementation.validate(&spec).
        assert_eq!(
            implementation
                .at_device(&seismic::device::BackendFloor(backend))
                .validate(&spec),
            Ok(()),
            "{label} ({backend:?}): runtime validate should accept default spec"
        );
        assert_eq!(
            E::contract_admit(backend, None, elements, dims, &params),
            Ok(()),
            "{label} ({backend:?}): contract_admit should accept default params"
        );

        // Negative checks on contract_admit: unknown parameter & out-of-domain value.
        let mut with_unknown = params.clone();
        with_unknown.push(("__NO_SUCH_PARAM__", 1));
        assert!(
            matches!(
                E::contract_admit(backend, None, elements, dims, &with_unknown),
                Err(ContractError::InvalidParameter(_))
            ),
            "{label} ({backend:?}): unknown param must fail"
        );

        if !params.is_empty() {
            let mut out_of_domain = params.clone();
            out_of_domain[0].1 = u64::MAX;
            assert!(
                matches!(
                    E::contract_admit(backend, None, elements, dims, &out_of_domain),
                    Err(ContractError::ParameterOutsideDomain { .. })
                ),
                "{label} ({backend:?}): out-of-domain param must fail"
            );
        }

        if let Some(first_static) = implementation.statics.first() {
            let missing_dims: Vec<(&str, u64)> = dims
                .iter()
                .copied()
                .filter(|(name, _)| *name != first_static.as_str())
                .collect();
            assert_eq!(
                E::contract_admit(backend, None, elements, &missing_dims, &params),
                Err(ContractError::MissingDimension(first_static.as_str())),
                "{label} ({backend:?}): missing static must fail"
            );
        }

        // 3. Compare contract_sizing against CheckedModule / NativeGraphCheckedEntry.
        let sizing = E::contract_sizing(backend, None, elements, dims, &params).unwrap_or_else(|err| {
            panic!("{label} ({backend:?}): contract_sizing failed: {err:?}")
        });

        // 3a. Results parity.
        let (_, expected_results_meta) = checked
            .shapes(dims)
            .unwrap_or_else(|err| panic!("{label} ({backend:?}): checked.shapes failed: {err}"));
        let expected_results: Vec<TensorResultSpec> = expected_results_meta
            .into_iter()
            .flatten()
            .map(|meta| TensorResultSpec {
                element: meta.element,
                bytes: meta.canonical_bytes,
            })
            .collect();
        assert_eq!(
            sizing.results, expected_results,
            "{label} ({backend:?}): result specs mismatch"
        );

        // 3b. Scratch parity.
        let param_lookup = |name: &str| spec.param(name);
        let expected_scratch: Vec<ScratchSpec> = implementation
            .scratch
            .iter()
            .map(|buffer| {
                let bytes = buffer
                    .charge(&dim_lookup, &param_lookup)
                    .unwrap_or_else(|err| {
                        panic!(
                            "{label} ({backend:?}): scratch `{}` charge failed: {err}",
                            buffer.name
                        )
                    });
                ScratchSpec {
                    bytes,
                    sync: buffer.sync,
                    emission: buffer.emission,
                }
            })
            .collect();
        assert_eq!(
            sizing.scratch, expected_scratch,
            "{label} ({backend:?}): scratch specs mismatch"
        );

        // 3c. Operand forms parity.
        let expected_operands: Vec<OperandFormSpec> = implementation
            .scratch
            .iter()
            .enumerate()
            .filter_map(|(scratch_index, buffer)| {
                buffer.operand.as_ref().map(|operand| match operand {
                    NativeOperand::Formed {
                        parameter,
                        formation,
                        pairs,
                        rows,
                        norm,
                        formed,
                        nvfp4,
                    } => OperandFormSpec {
                        scratch_index,
                        role: OperandRole::Formed {
                            parameter: *parameter,
                            formation: *formation,
                            pairs: pairs.holds(&dim_lookup, &param_lookup).unwrap(),
                            formed: formed.holds(&dim_lookup, &param_lookup).unwrap(),
                            nvfp4: nvfp4.holds(&dim_lookup, &param_lookup).unwrap(),
                        },
                        rows: *rows,
                        norm: *norm,
                    },
                    NativeOperand::Emitted { result, rows, norm } => OperandFormSpec {
                        scratch_index,
                        role: OperandRole::Emitted { result: *result },
                        rows: *rows,
                        norm: *norm,
                    },
                })
            })
            .collect();
        assert_eq!(
            sizing.operand_forms, expected_operands,
            "{label} ({backend:?}): operand forms mismatch"
        );

        // 3d. Active launches parity (including repeat rounds).
        let active: Vec<bool> = implementation
            .launches
            .iter()
            .enumerate()
            .map(|(launch_idx, launch)| {
                let launch_param_lookup = |name: &str| {
                    spec.launch_param(launch_idx, name)
                        .or_else(|| spec.param(name))
                };
                match &launch.when {
                    None => true,
                    Some(cond) => cond.holds(&dim_lookup, &launch_param_lookup).unwrap(),
                }
            })
            .collect();
        let rounds = match &implementation.repeat {
            None => 1u64,
            Some(repeat) => repeat.count.evaluate(&dim_lookup, &param_lookup).unwrap(),
        };
        let expected_active_launches = implementation
            .dispatch_order(rounds)
            .filter(|&ordinal| active[ordinal])
            .count() as u64;
        assert_eq!(
            sizing.active_launches, expected_active_launches,
            "{label} ({backend:?}): active launches mismatch"
        );
    }
}

#[test]
fn dense_expand_parity_all_backends() {
    let bf16 = Element::bf16();
    let f32 = Element::f32();
    let q4k = Element::named("q4k").expect("q4k registered");

    let dims_decode = [
        ("M", 1),
        ("O", 1),
        ("H", 2048),
        ("F", 5632),
        ("GS", 0),
        ("US", 0),
        ("GB", 0),
        ("UB", 0),
        ("LB", 0),
    ];
    let dims_prefill = [
        ("M", 128),
        ("O", 128),
        ("H", 2048),
        ("F", 5632),
        ("GS", 0),
        ("US", 0),
        ("GB", 0),
        ("UB", 0),
        ("LB", 0),
    ];

    for (tag, dims) in [("decode", &dims_decode[..]), ("prefill", &dims_prefill[..])] {
        assert_contract_parity::<magnitude_kernels::dense_expand::Entry>(
            &format!("dense_expand/{tag}/bf16"),
            &[("A", bf16), ("NW", f32), ("GW", bf16), ("UW", bf16)],
            dims,
        );
        assert_contract_parity::<magnitude_kernels::dense_expand::Entry>(
            &format!("dense_expand/{tag}/q4k"),
            &[("A", bf16), ("NW", f32), ("GW", q4k), ("UW", q4k)],
            dims,
        );
    }
}

#[test]
fn dense_output_parity_all_backends() {
    let bf16 = Element::bf16();
    let q4k = Element::named("q4k").expect("q4k registered");

    let dims = [
        ("M", 32),
        ("O", 32),
        ("H", 2048),
        ("F", 5632),
        ("DS", 0),
        ("DB", 0),
    ];
    assert_contract_parity::<magnitude_kernels::dense_output::Entry>(
        "dense_output/bf16",
        &[("A", bf16), ("DW", bf16)],
        &dims,
    );
    assert_contract_parity::<magnitude_kernels::dense_output::Entry>(
        "dense_output/q4k",
        &[("A", bf16), ("DW", q4k)],
        &dims,
    );
}

#[test]
fn readout_project_rows_parity_all_backends() {
    let bf16 = Element::bf16();
    let q4k = Element::named("q4k").expect("q4k registered");
    let dims = [
        ("O", 4),
        ("V", 32000),
        ("D", 2048),
        ("S", 32000),
        ("I", 0),
        ("WS", 0),
    ];
    assert_contract_parity::<magnitude_kernels::readout_project_rows::Entry>(
        "readout_project_rows/bf16",
        &[("A", bf16), ("OW", bf16)],
        &dims,
    );
    assert_contract_parity::<magnitude_kernels::readout_project_rows::Entry>(
        "readout_project_rows/q4k",
        &[("A", bf16), ("OW", q4k)],
        &dims,
    );
}

#[test]
fn attention_decode_parity_all_backends() {
    let bf16 = Element::bf16();
    let dims = [
        ("M", 1),
        ("T", 256),
        ("KV", 8),
        ("G", 4),
        ("P", 64),
        ("S", 0),
        ("I", 0),
        ("U", 0),
        ("F", 1),
        ("NQ", 0),
        ("N", 0),
        ("NV", 0),
        ("R", 1),
        ("SK", 0),
        ("RA", 0),
        ("RT", 0),
        ("RO", 0),
        ("VK", 0),
        ("KR", 0),
        ("AB", 0),
        ("KS", 0),
        ("TS", 0),
    ];
    assert_contract_parity::<magnitude_kernels::attention_decode::Entry>(
        "attention_decode/bf16",
        &[("A", bf16)],
        &dims,
    );
}

#[test]
fn repack_and_elementwise_parity_all_backends() {
    let f32 = Element::f32();
    let bf16 = Element::bf16();
    let q4k_src = Element::named("gguf_q4_k").expect("gguf_q4_k registered");
    let q4k_dst = Element::named("q4k").expect("q4k registered");

    assert_contract_parity::<magnitude_kernels::repack_weight::Entry>(
        "repack_weight/q4k",
        &[("E", q4k_src), ("U", q4k_dst)],
        &[("B", 1), ("N", 1024), ("K", 2048)],
    );
    // Dense-to-dense repack exercises `unit_columns(U)` returning None on backends that read it.
    assert_contract_parity::<magnitude_kernels::repack_weight::Entry>(
        "repack_weight/f32_to_bf16",
        &[("E", f32), ("U", bf16)],
        &[("B", 1), ("N", 1024), ("K", 2048)],
    );
    assert_contract_parity::<magnitude_kernels::widen_rows::Entry>(
        "widen_rows/bf16",
        &[("A", bf16)],
        &[("M", 16), ("D", 2048)],
    );
    assert_contract_parity::<magnitude_kernels::corrector_features::Entry>(
        "corrector_features",
        &[],
        &[("M", 4), ("V", 1024), ("K", 64), ("H", 128)],
    );
}

#[test]
fn packed_and_non_float_element_admission_parity() {
    let gguf_q2_k = Element::named("gguf_q2_k").expect("gguf_q2_k registered");
    let q2k_rows16 = Element::named("q2k@rows16").expect("q2k@rows16 registered");

    fn assert_admits<E: EntryContract>(
        label: &str,
        backends: &[BackendName],
        elements: &[(&str, Element)],
        dims: &[(&str, u64)],
    ) {
        assert_contract_parity::<E>(label, elements, dims);
        for &backend in backends {
            let params = E::contract_default_params(backend, None, elements, dims).unwrap_or_else(|err| {
                panic!("{label} ({backend:?}): contract_default_params failed: {err:?}")
            });
            assert_eq!(
                E::contract_admit(backend, None, elements, dims, &params),
                Ok(()),
                "{label} ({backend:?}): contract_admit should succeed"
            );
        }
    }

    let routed_land_elems = [
        ("SG", gguf_q2_k),
        ("SU", gguf_q2_k),
        ("SD", gguf_q2_k),
        ("UG", q2k_rows16),
        ("UU", q2k_rows16),
        ("UD", q2k_rows16),
        ("A", Element::bf16()),
    ];
    let routed_round_elems = [
        ("SG", gguf_q2_k),
        ("SU", gguf_q2_k),
        ("SD", gguf_q2_k),
        ("UG", q2k_rows16),
        ("UU", q2k_rows16),
        ("UD", q2k_rows16),
    ];
    let routed_land_dims = [
        ("M", 1),
        ("K", 8),
        ("FW", 64),
        ("FLIGHT", 4),
        ("L", 8),
        ("V", 4),
        ("GV", 4),
        ("S", 8),
        ("GS", 8),
        ("F", 512),
        ("H", 256),
        ("SM", 0),
        ("SF", 0),
        ("SR", 0),
    ];
    assert_admits::<magnitude_kernels::routed_land::Entry>(
        "routed_land/q2k",
        &[BackendName::Cuda, BackendName::Metal],
        &routed_land_elems,
        &routed_land_dims,
    );

    let routed_round_dims = [
        ("FLIGHT", 4),
        ("LANES", 16),
        ("L", 8),
        ("V", 4),
        ("GV", 4),
        ("S", 8),
        ("GS", 8),
        ("F", 512),
        ("H", 256),
        ("E", 8),
        ("EL", 64),
    ];
    assert_admits::<magnitude_kernels::routed_round::Entry>(
        "routed_round/q2k",
        &[BackendName::Cuda, BackendName::Metal],
        &routed_round_elems,
        &routed_round_dims,
    );

    let routed_copy_dims = [("S", 8), ("D", 8), ("L", 4), ("N", 512), ("K", 256)];
    assert_admits::<magnitude_kernels::routed_copy::Entry>(
        "routed_copy/bool",
        &[BackendName::Cuda, BackendName::Metal],
        &[("U", Element::bool())],
        &routed_copy_dims,
    );
    assert_admits::<magnitude_kernels::routed_copy::Entry>(
        "routed_copy/q2k_rows16",
        &[BackendName::Cuda, BackendName::Metal],
        &[("U", q2k_rows16)],
        &routed_copy_dims,
    );

    let copy_rows_dims = [("N", 4), ("T", 64), ("KV", 8), ("W", 128)];
    assert_admits::<magnitude_kernels::copy_rows::Entry>(
        "copy_rows/u32",
        &[BackendName::Cuda, BackendName::Metal, BackendName::Cpu],
        &[("A", Element::u32())],
        &copy_rows_dims,
    );
    assert_admits::<magnitude_kernels::copy_rows::Entry>(
        "copy_rows/i32",
        &[BackendName::Cuda, BackendName::Metal, BackendName::Cpu],
        &[("A", Element::i32())],
        &copy_rows_dims,
    );

    assert_admits::<magnitude_kernels::repack_weight::Entry>(
        "repack_weight/q2k_rows16",
        &[BackendName::Cuda, BackendName::Metal, BackendName::Cpu],
        &[("E", gguf_q2_k), ("U", q2k_rows16)],
        &[("B", 2), ("N", 512), ("K", 256)],
    );
}

#[test]
fn contract_node_shape_presence_matches_native_implementation_all_entries() {
    use std::collections::BTreeSet;

    fn check_entry<E: EntryContract>(name: &'static str, checked: &mut BTreeSet<&'static str>) {
        assert!(checked.insert(name), "duplicate entry in test list: {name}");
        for backend in ALL_BACKENDS {
            let has_shape = E::contract_node_shape(backend).is_ok();
            let has_native = seismic::generated::native_implementation_for_backend::<E>(backend)
                .unwrap_or_else(|err| panic!("{name} ({backend:?}): bundle decode failed: {err}"))
                .is_some();
            assert_eq!(
                has_shape, has_native,
                "{name} ({backend:?}): contract_node_shape.is_ok()={has_shape} != native_implementation_for_backend.is_some()={has_native}"
            );
        }
    }

    macro_rules! check_all_entries {
        ($checked:ident; $($entry:ident),+ $(,)?) => {
            $(
                check_entry::<magnitude_kernels::$entry::Entry>(stringify!($entry), &mut $checked);
            )+
        };
    }

    let mut checked = BTreeSet::new();
    check_all_entries!(
        checked;
        affine_encode,
        affine_decode,
        attention_absorb,
        attention_output,
        attention_project,
        held_in_activation,
        head_norm,
        rotate_pairs,
        head_rotary,
        rounded_head_rotary,
        output_rotary,
        gated_column,
        gate_row,
        attention_rows,
        block_read,
        attention_decode,
        attention_prefill,
        attention_rows_k8v4,
        attention_append_dense,
        attention_append_k8v4,
        append_rows_k8v4,
        attention_decode_k8v4,
        attention_prefill_k8v4,
        cluster_token_rows,
        shortlist_token_rows,
        place_logits_rows,
        select_compact_rows,
        corrector_features,
        corrector_rows,
        corrector_nodes,
        corrector_attention,
        corrector_factors,
        corrector_path,
        project_rows,
        tap_rows,
        feature_rows,
        widen_rows,
        draft_confidence,
        token_confidence,
        draft_convolve_input,
        draft_convolve_residual,
        draft_gated_rows,
        draft_top_k,
        draft_path_step,
        draft_rows,
        readout_project_rows,
        dense_expand,
        dense_output,
        dense_up,
        activate,
        glu,
        form_input,
        sum_parts,
        scatter,
        output_gate,
        scale_rows,
        cap_logit,
        rank_experts,
        rank_choices,
        select_experts,
        route_shared,
        import_dense,
        packing_bound,
        repack_weight,
        indexer_keys,
        indexer_score,
        rows_topk,
        unit_reads,
        candidate_blocks,
        candidate_rows,
        ngram_rows,
        ngram_gate,
        ngram_add,
        ngram_inject,
        per_layer_gate,
        pool_rows,
        token_norm,
        readout_features_rows,
        readout_head_rows,
        readout_selected_rows,
        gumbel_rows,
        progressive_view,
        certify_level,
        readout_top_rows,
        readout_refine_rows,
        readout_exact_rows,
        readout_planes_rows,
        gated_delta_project,
        gated_delta_convolve,
        delta_advance,
        gated_delta_advance,
        gated_delta_rows,
        gated_delta_step,
        gated_delta_project_convolved,
        gated_delta_step_convolved,
        gated_delta_chunk,
        kimi_delta_advance,
        kimi_delta_rows,
        kimi_delta_step,
        kimi_delta_chunk,
        post_norm_residual,
        moe_tail,
        per_layer_inputs,
        round_codes,
        power_of_two_ceil,
        element_mantissa,
        element_min_exponent,
        element_max,
        block_scale,
        round_element,
        block_round,
        routed_route,
        routed_route_shared,
        routed_expand,
        routed_output,
        decode_output_published,
        decode_output_join,
        routed_group,
        routed_experts,
        routed_combine,
        routed_select,
        decode_source,
        routed_gate_up,
        routed_up,
        decode_down,
        routed_down,
        decode_join,
        decode_selected,
        routed_experts_up,
        routed_scatter,
        step_sequence,
        pair_at,
        held,
        flight_slot,
        slot_answered,
        slot_succeeded,
        store_pair,
        request_slot,
        resolve,
        first_of,
        assign,
        routed_route_streamed,
        routed_select_streamed,
        routed_site_request,
        site_publish,
        any_sited,
        routed_down_project,
        routed_down_sited,
        routed_output_project,
        routed_output_sited,
        routed_output_streamed,
        landed_entries,
        land_tensor,
        land,
        source_landed,
        routed_land,
        routed_step,
        scratch_lane,
        routed_scratch_request,
        routed_round,
        routed_scratch_grids,
        stage_list,
        routed_list_grid,
        routed_pass_map,
        routed_copy,
        routed_seed_commit,
        ordered_key,
        count_boundary,
        mass_boundary,
        tie_cutoff,
        shape_rows,
        short_conv_rows,
        short_conv_project,
        copy_rows,
        state_space_rows,
        state_space_step,
        state_space_chunk,
        state_space_gate,
        group_rms,
        stream_mix_down,
        stream_mix_up,
        stream_mix_sinkhorn,
        collapse,
        stream_collapse,
        stream_join,
        repeat_rows,
        gathered_bucket,
        gathered_row,
        embedding_rows,
        conditioning_overlay,
        stored_rows,
        selection_notice,
        feature_copy,
        vision_patch_stem,
        vision_activate,
        vision_norm,
        vision_linear,
        vision_clamp,
        vision_rotate,
        vision_attention,
        vision_gated_convolution,
        vision_pool,
        vision_position,
        matmul,
        argmax_row,
        attention_update,
        attention,
        attention_append,
        attention_gate,
        delta_steps,
        delta_step,
        silu,
        multiply,
        add,
        sigmoid,
        gelu_tanh,
        embedding,
        embedding_row,
        gate_projection_add,
        gated_norm,
        gated_projection,
        erf_values,
        gelu,
        kv_append,
        layer_norm,
        linear,
        linear_bias,
        logits,
        norm_gated_projection,
        norm_logits,
        norm_projection,
        operand_rows,
        operand_row,
        nvfp4_row,
        nvfp4_global,
        ue4m3,
        e2m1,
        projection,
        projection_add,
        recurrent_prepare_rows,
        recurrent_prepare,
        rms_norm,
        sum_any_order,
        norm_rotary,
        rotary_prepare,
        sample_rows,
        import_weight,
        import_weight_negative_exp,
        computes,
        exact_chooses,
        exact_rows,
        gate_up_choice,
        guess_chain,
        name_entry,
        publish_front,
        routed_gate_up_part,
        select_route,
        stands,
        weight_word,
    );

    let module = magnitude_kernels::module().expect("bundle decodes cleanly");
    let catalog: BTreeSet<&str> = module
        .checked()
        .entries()
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(
        checked, catalog,
        "contract_node_shape parity test must cover every entry in magnitude_kernels"
    );
}

struct OverrideDevice {
    backend: BackendName,
    overrides: &'static [(&'static str, u64)],
}

impl seismic::device::DeviceValues for OverrideDevice {
    fn value(&self, name: &str) -> u64 {
        self.overrides
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| seismic::device::BackendFloor(self.backend).value(name))
    }

    fn measured(&self, name: &str) -> Option<u64> {
        seismic::device::BackendFloor(self.backend).measured(name)
    }
}

fn param_value(params: &[(&str, u64)], name: &str) -> u64 {
    params
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| *v)
        .unwrap_or_else(|| panic!("missing parameter `{name}` in {params:?}"))
}

#[test]
fn canonical_default_gaps_are_closed() {
    let bf16 = Element::bf16();

    // 1. CUDA routed choice|summed: GATHER_TPW defaults to 1 even when N is wide.
    let cuda_dev = OverrideDevice {
        backend: BackendName::Cuda,
        overrides: &[("units", 48)],
    };
    let routed_params = magnitude_kernels::routed_output::Entry::contract_default_params(
        BackendName::Cuda,
        Some(&cuda_dev),
        &[("A", bf16), ("EDW", bf16), ("SDW", bf16), ("NW", bf16)],
        &[
            ("M", 4),
            ("H", 7168),
            ("SN", 1),
            ("IN", 0),
            ("E", 256),
            ("K", 8),
            ("F", 2048),
            ("S", 2048),
            ("XS", 0),
            ("TR", 0),
            ("NO", 0),
        ],
    )
    .expect("CUDA routed_output defaults resolve");
    assert_eq!(param_value(&routed_params, "GATHER_TPW"), 1);

    // 2. Metal dense single: TALL defaults to 0 when G > 1 (e.g. project_rows G=64),
    //    and stays 1 when G == 1 and K % 64 == 0.
    let metal_m5 = OverrideDevice {
        backend: BackendName::Metal,
        overrides: &[
            ("units", 20),
            ("shared_bytes_per_group", 32768),
            ("max_group_threads", 1024),
            ("subgroup_width", 32),
            ("tensor_ops", 1),
        ],
    };
    let proj_g64 = magnitude_kernels::project_rows::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m5),
        &[("A", bf16), ("W", bf16), ("Y", bf16)],
        &[("M", 64), ("G", 64), ("K", 128), ("N", 512), ("WS", 0), ("WB", 0)],
    )
    .expect("Metal project_rows G=64 defaults resolve");
    assert_eq!(param_value(&proj_g64, "TALL"), 0);

    let proj_g1 = magnitude_kernels::project_rows::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m5),
        &[("A", bf16), ("W", bf16), ("Y", bf16)],
        &[("M", 64), ("G", 1), ("K", 128), ("N", 512), ("WS", 0), ("WB", 0)],
    )
    .expect("Metal project_rows G=1 defaults resolve");
    assert_eq!(param_value(&proj_g1, "TALL"), 1);

    // 3. Vulkan attention_decode_k8v4: SLICES follows register-pressure rule bounded by SIMDS=2.
    let vulkan_v710 = OverrideDevice {
        backend: BackendName::Vulkan,
        overrides: &[
            ("units", 28),
            ("shared_bytes_per_group", 65536),
            ("max_group_threads", 1024),
        ],
    };
    let vk_g8_w256 = magnitude_kernels::attention_decode_k8v4::Entry::contract_default_params(
        BackendName::Vulkan,
        Some(&vulkan_v710),
        &[("A", bf16)],
        &[
            ("M", 1),
            ("T", 256),
            ("KV", 2),
            ("G", 8),
            ("P", 128),
            ("S", 0),
            ("I", 0),
            ("U", 0),
            ("F", 1),
            ("NQ", 0),
            ("N", 0),
            ("NV", 0),
            ("KC", 64),
            ("KP", 16),
            ("VC", 32),
            ("VP", 16),
            ("R", 1),
            ("SK", 0),
            ("RA", 0),
            ("RT", 0),
            ("RO", 0),
            ("VK", 0),
        ],
    )
    .expect("Vulkan attention_decode_k8v4 G=8 W=256 defaults resolve");
    assert_eq!(param_value(&vk_g8_w256, "SIMDS"), 2);
    assert_eq!(param_value(&vk_g8_w256, "SLICES"), 2);

    let vk_g4_w128 = magnitude_kernels::attention_decode_k8v4::Entry::contract_default_params(
        BackendName::Vulkan,
        Some(&vulkan_v710),
        &[("A", bf16)],
        &[
            ("M", 1),
            ("T", 256),
            ("KV", 4),
            ("G", 4),
            ("P", 64),
            ("S", 0),
            ("I", 0),
            ("U", 0),
            ("F", 1),
            ("NQ", 0),
            ("N", 0),
            ("NV", 0),
            ("KC", 32),
            ("KP", 8),
            ("VC", 16),
            ("VP", 8),
            ("R", 1),
            ("SK", 0),
            ("RA", 0),
            ("RT", 0),
            ("RO", 0),
            ("VK", 0),
        ],
    )
    .expect("Vulkan attention_decode_k8v4 G=4 W=128 defaults resolve");
    assert_eq!(param_value(&vk_g4_w128, "SLICES"), 1);

    // 4. Metal attention_decode: MATRIX==2 scales PARTS occupancy by active threadgroup width
    //    so KV=1, G=64, S=512, units=20 picks PARTS=16 instead of 64.
    let metal_mla = magnitude_kernels::attention_decode::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m5),
        &[("A", bf16)],
        &[
            ("M", 1),
            ("T", 256),
            ("KV", 1),
            ("G", 64),
            ("P", 0),
            ("S", 512),
            ("I", 0),
            ("U", 0),
            ("F", 1),
            ("NQ", 0),
            ("N", 0),
            ("NV", 0),
            ("R", 1),
            ("SK", 0),
            ("RA", 0),
            ("RT", 0),
            ("RO", 0),
            ("VK", 0),
            ("KR", 0),
            ("AB", 0),
            ("KS", 0),
            ("TS", 0),
        ],
    )
    .expect("Metal attention_decode MLA defaults resolve");
    assert_eq!(param_value(&metal_mla, "MATRIX"), 2);
    assert_eq!(param_value(&metal_mla, "PARTS"), 16);

    // 5. Routed decode GATHERED defaults to 1 across CUDA and Metal, and Metal
    //    summed keeps LANES=8 whenever SUMMED_STAGE <= 32 (falling back to 16
    //    only when LANES=8 with ROWS=2 exceeds the 8-packet register budget).
    let cuda_gate_up = magnitude_kernels::routed_gate_up::Entry::contract_default_params(
        BackendName::Cuda,
        Some(&cuda_dev),
        &[("A", bf16), ("EGW", bf16), ("EUW", bf16)],
        &[
            ("M", 2),
            ("H", 7168),
            ("E", 256),
            ("K", 8),
            ("F", 2048),
            ("GS", 0),
            ("US", 0),
            ("XS", 0),
            ("TR", 0),
            ("LB", 0),
        ],
    )
    .expect("CUDA routed_gate_up defaults resolve");
    assert_eq!(param_value(&cuda_gate_up, "GATHERED"), 1);
    assert_eq!(param_value(&cuda_gate_up, "GATHER_TPW"), 1);

    let metal_down_k8 = magnitude_kernels::routed_down::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m5),
        &[("A", bf16), ("EDW", bf16), ("R", bf16)],
        &[
            ("M", 4),
            ("H", 4096),
            ("E", 256),
            ("K", 8),
            ("F", 1536),
            ("XS", 0),
            ("TR", 0),
        ],
    )
    .expect("Metal routed_down K=8 F=1536 defaults resolve");
    assert_eq!(param_value(&metal_down_k8, "GATHERED"), 1);
    assert_eq!(param_value(&metal_down_k8, "LANES"), 8);
    assert_eq!(param_value(&metal_down_k8, "ROWS"), 2);

    let metal_down_k4 = magnitude_kernels::routed_down::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m5),
        &[("A", bf16), ("EDW", bf16), ("R", bf16)],
        &[
            ("M", 4),
            ("H", 4096),
            ("E", 256),
            ("K", 4),
            ("F", 1536),
            ("XS", 0),
            ("TR", 0),
        ],
    )
    .expect("Metal routed_down K=4 F=1536 defaults resolve");
    assert_eq!(param_value(&metal_down_k4, "GATHERED"), 1);
    assert_eq!(param_value(&metal_down_k4, "LANES"), 16);
    assert_eq!(param_value(&metal_down_k4, "ROWS"), 2);

    let metal_output = magnitude_kernels::routed_output::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m5),
        &[("A", bf16), ("EDW", bf16), ("SDW", bf16), ("NW", bf16)],
        &[
            ("M", 4),
            ("H", 4096),
            ("SN", 1),
            ("IN", 0),
            ("E", 256),
            ("K", 8),
            ("F", 1536),
            ("S", 1536),
            ("XS", 0),
            ("TR", 0),
            ("NO", 0),
        ],
    )
    .expect("Metal routed_output defaults resolve");
    assert_eq!(param_value(&metal_output, "GATHERED"), 1);
    assert_eq!(param_value(&metal_output, "LANES"), 8);
    assert_eq!(param_value(&metal_output, "ROWS"), 2);

    let metal_gate_up = magnitude_kernels::routed_gate_up::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m5),
        &[("A", bf16), ("EGW", bf16), ("EUW", bf16)],
        &[
            ("M", 4),
            ("H", 4096),
            ("E", 256),
            ("K", 8),
            ("F", 1536),
            ("GS", 0),
            ("US", 0),
            ("XS", 0),
            ("TR", 0),
            ("LB", 0),
        ],
    )
    .expect("Metal routed_gate_up defaults resolve");
    assert_eq!(param_value(&metal_gate_up, "GATHERED"), 1);
    assert_eq!(param_value(&metal_gate_up, "ROWS"), 1);

    let metal_m2 = OverrideDevice {
        backend: BackendName::Metal,
        overrides: &[
            ("units", 19),
            ("shared_bytes_per_group", 32768),
            ("tensor_ops", 0),
        ],
    };
    let metal_gate_up_m2 = magnitude_kernels::routed_gate_up::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m2),
        &[("A", bf16), ("EGW", bf16), ("EUW", bf16)],
        &[
            ("M", 4),
            ("H", 2560),
            ("E", 512),
            ("K", 10),
            ("F", 640),
            ("GS", 0),
            ("US", 0),
            ("XS", 0),
            ("TR", 0),
            ("LB", 0),
        ],
    )
    .expect("Metal M2 routed_gate_up defaults resolve");
    assert_eq!(param_value(&metal_gate_up_m2, "GATHERED"), 0);
    assert_eq!(param_value(&metal_gate_up_m2, "ROWS"), 1);

    let metal_output_m2 = magnitude_kernels::routed_output::Entry::contract_default_params(
        BackendName::Metal,
        Some(&metal_m2),
        &[("A", bf16), ("EDW", bf16), ("SDW", bf16), ("NW", bf16)],
        &[
            ("M", 4),
            ("H", 2560),
            ("SN", 1),
            ("IN", 0),
            ("E", 512),
            ("K", 10),
            ("F", 640),
            ("S", 640),
            ("XS", 0),
            ("TR", 0),
            ("NO", 0),
        ],
    )
    .expect("Metal M2 routed_output defaults resolve");
    assert_eq!(param_value(&metal_output_m2, "GATHERED"), 1);
    assert_eq!(param_value(&metal_output_m2, "LANES"), 8);
    assert_eq!(param_value(&metal_output_m2, "ROWS"), 2);
}
