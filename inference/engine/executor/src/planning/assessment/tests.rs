use super::*;
use magnitude_kernels::dense_output;
use seismic::fixtures::recorded;
use seismic::{BackendName, Element, Layout, NativeGraphMetadata};

#[test]
fn checked_native_declarations_are_inspectable_without_a_device() {
    let implementation = seismic::generated::native_implementation_for_backend::<
        magnitude_kernels::dense_output::Entry,
    >(BackendName::Cpu)
    .unwrap();
    assert!(implementation.is_some());
    let invalid = seismic::generated::checked_native_binding::<dense_output::Entry>(
        BackendName::Cpu,
        &[
            ("DW", Element::f32()),
            ("A", Element::f32()),
            ("BOGUS", Element::f32()),
        ],
    )
    .unwrap();
    assert!(matches!(
        invalid,
        seismic::NativeBindingCheck::Unsupported(_)
    ));

    let weight = seismic::generated::checked_native_tensor_parameter::<dense_output::Entry>(
        BackendName::Cpu,
        &[("DW", Element::f32()), ("A", Element::f32())],
        "down_weight",
        &[
            ("M", 4),
            ("O", 2),
            ("H", 16),
            ("F", 32),
            ("DS", 0),
            ("DB", 0),
        ],
    )
    .unwrap();
    assert_eq!(
        weight,
        seismic::NativeTensorParameterCheck::Checked(seismic::NativeTensorMetadata {
            element: Element::f32(),
            extents: vec![16, 32],
            canonical_bytes: 16 * 32 * 4,
        })
    );
    let results = seismic::generated::checked_native_tensor_results::<dense_output::Entry>(
        BackendName::Cpu,
        &[("DW", Element::f32()), ("A", Element::f32())],
        &[
            ("M", 4),
            ("O", 2),
            ("H", 16),
            ("F", 32),
            ("DS", 0),
            ("DB", 0),
        ],
    )
    .unwrap();
    assert_eq!(
        results,
        seismic::NativeTensorResultsCheck::Checked(vec![Some(seismic::NativeTensorMetadata {
            element: Element::f32(),
            extents: vec![2, 16],
            canonical_bytes: 2 * 16 * 4,
        })])
    );
    let call = seismic::generated::checked_native_graph_call::<dense_output::Entry>(
        BackendName::Cpu,
        &[("DW", Element::f32()), ("A", Element::f32())],
        &[
            ("M", 4),
            ("O", 2),
            ("H", 16),
            ("F", 32),
            ("DS", 0),
            ("DB", 0),
        ],
    )
    .unwrap();
    let seismic::generated::CheckedNativeGraphCall::Checked {
        parameters,
        results: call_results,
        ..
    } = call
    else {
        panic!("checked dense output call is supported on CPU");
    };
    let seismic::NativeTensorParameterCheck::Checked(weight) = weight else {
        unreachable!()
    };
    assert!(parameters.contains(&Some(weight)));
    assert_eq!(
        seismic::NativeTensorResultsCheck::Checked(call_results),
        results
    );
}

#[test]
fn checked_embedding_graph_distinguishes_upload_from_selected_input() {
    let definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let uploaded = crate::programs::native_target_graph::checked_entry_graph_storage(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        2,
        true,
    )
    .unwrap();
    let selected = crate::programs::native_target_graph::checked_entry_graph_storage(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        2,
        false,
    )
    .unwrap();
    assert_eq!(uploaded.workspace, selected.workspace);
    assert_eq!(uploaded.output, selected.output);
    assert_eq!(uploaded.upload, 2 * 2 * 4);
    assert_eq!(selected.upload, 0);
    assert!(uploaded.workspace > 0);
    assert!(uploaded.output > 0);

    let elements = [("EW", Element::f32()), ("A", Element::f32())];
    let dimensions = [
        ("M", 2),
        ("V", definition.decoder.vocabulary),
        ("D", definition.decoder.hidden),
        ("R", 0),
        ("F", 1),
    ];
    let mut invalid = NativeGraphMetadata::new(BackendName::Cpu);
    let wrong_table = invalid.port(Element::f32(), &[1, 1]).unwrap();
    let tokens = invalid
        .input_for::<magnitude_kernels::embedding_rows::Entry>(&elements, "tokens", &dimensions)
        .unwrap();
    let direct = invalid.port(Element::i32(), &[0, 1]).unwrap();
    let mismatch = invalid.enqueue::<magnitude_kernels::embedding_rows::Entry>(
        &elements,
        &dimensions,
        magnitude_kernels::embedding_rows::WorkflowArgs {
            table: wrong_table.tensor().into(),
            tokens: tokens.tensor().into(),
            directory: direct.tensor().into(),
            scale: 1.0,
            normalize: 0,
            epsilon: 0.0,
        },
    );
    assert!(matches!(
        mismatch,
        Err(seismic::NativeGraphMetadataError::Call(
            seismic::CallError::Workflow(seismic::WorkflowError::NativeGraphArgumentMismatch {
                parameter: 0
            })
        ))
    ));
}

#[test]
fn checked_decoder_family_bounds_every_prepared_class_axis() {
    let definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let limits = crate::ResourceLimits {
        max_launch_rows: 2,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: 0,
        max_images_per_request: 1,
        max_image_cells: 0,
        lookahead: false,
        chain_capacity: std::num::NonZeroUsize::MIN,
        conversations: std::num::NonZeroUsize::MIN,
    };
    let state = crate::ResourcePlanner::state_plan(
        &definition,
        &load,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
        crate::ResourceCapacity {
            domain_bytes: 512 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::Absent,
        },
    )
    .unwrap();
    let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
    let family = crate::programs::native_target_graph::checked_target_family_storage(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        &state,
        plan.target(),
        limits,
    )
    .unwrap();
    let (block, _) = crate::programs::native_target_graph::checked_block_graph_resources(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        &state,
        plan.target().blocks()[0],
        0,
        (2, 4, 2, 0, 0),
    )
    .unwrap();
    assert!(family.storage.workspace >= block.workspace);
    assert!(family.storage.output >= block.output);
    assert!(family.storage.upload >= block.upload);
    assert!(family.storage.workspace > 0);
    assert!(family.storage.output > 0);
    let pools = AssessmentGraphResourceBounds::derive(
        &definition,
        &load,
        &state,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
        BackendName::Cpu,
    )
    .unwrap();
    assert_eq!(pools.target.workspace_bytes, family.storage.workspace);
    assert_eq!(pools.target.output_bytes, family.storage.output);
    assert_eq!(pools.binding_constant_bytes, family.binding_constant_bytes);
    assert!(pools.binding_constant_bytes > 0);
    // The entry's tokens and the launch's controls, which its blocks
    // bind: a block of a plain load writes no input of its own.
    assert_eq!(pools.target.upload_regions.len(), 2);
    assert_eq!(pools.target.upload_regions[0], family.storage.upload);
    assert!(pools.total_bytes >= pools.target.committed_bytes);
}

#[test]
fn checked_head_family_includes_chained_and_shaped_classes() {
    use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
    use magnitude_family_contracts::{
        ActivationFunction, ExitNorm, ExpertSelection, FeedForwardUp, Head, HeadBlock,
        HeadChain, HeadComponent, HeadHistory, HeadInput, HeadReadout, InputNorm, Operator,
        RmsNorm, RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction,
        SharedExpert, SharedExpertGate, VocabularyProjection, WeightDescriptor,
    };

    // At the width every backend's mix sites admit: the test later puts
    // the hyper-connection fixture's decoder under the same head.
    let hidden = crate::planning::fixtures::mix_site_hidden();
    let mut definition = crate::planning::fixtures::fixture_definition_of(hidden);
    let descriptor =
        |name: &str, shape: &[u64]| WeightDescriptor::stored(format!("head_{name}"), shape);
    let rms = |name: &str| RmsNorm {
        weight: descriptor(name, &[hidden]),
        epsilon: 1e-6,
    };
    // The head block is the target block (attention, dense) under head
    // names, between its own norms and combine.
    let mut head_layer = definition.decoder.blocks[0].clone();
    let rename = |weight: &mut WeightDescriptor| weight.name = format!("head_{}", weight.name);
    for sublayer in &mut head_layer.sublayers {
        if let InputNorm::Rms(norm) = &mut sublayer.input {
            rename(&mut norm.weight);
        }
        match &mut sublayer.op {
            Operator::Attention(attention) => {
                rename(&mut attention.query);
                rename(&mut attention.output);
                if let magnitude_family_contracts::HeadNorm::Rms(norm) =
                    &mut attention.query_norm
                {
                    rename(&mut norm.weight);
                }
                if let magnitude_family_contracts::KeyValue::Owned {
                    key,
                    value: magnitude_family_contracts::ValueSource::Projected(value),
                    key_norm: magnitude_family_contracts::HeadNorm::Rms(key_norm),
                    ..
                } = &mut attention.key_value
                {
                    rename(key);
                    rename(value);
                    rename(&mut key_norm.weight);
                }
            }
            Operator::DenseFfn(dense) => {
                if let FeedForwardUp::Gated { gate, up, .. } = &mut dense.up {
                    rename(gate);
                    rename(up);
                }
                rename(&mut dense.down);
            }
            _ => unreachable!(),
        }
    }
    definition.head = Some(Head {
        blocks: vec![HeadBlock {
            residual: magnitude_family_contracts::ResidualForm::Single,
            input: HeadInput::Combine {
                embedding_norm: rms("embedding_norm"),
                hidden_norm: rms("hidden_norm"),
                combine: descriptor("combine", &[hidden, 2 * hidden]),
            },
            blocks: vec![head_layer],
            output_norm: ExitNorm::Rms(rms("output_norm")),
            readout: HeadReadout::full(VocabularyProjection::Target),
            chain: HeadChain::Output,
        }],
        history: HeadHistory::Own,
        component: HeadComponent::Target,
    });
    let mut manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let mut offset = manifest.target.files[0].size;
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: true,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let method = PlannedMethod::Mtp {
        greedy_proposals: 1,
        sampled_proposals: 1,
        widths: crate::DraftWidths::Adaptive,
    };
    let limits = crate::ResourceLimits {
        max_launch_rows: 256,
        max_launch_slots: 16,
        max_selected_rows: 16,
        max_drafting_slots: 16,
        exported_logits_rows: 0,
        max_images_per_request: 1,
        max_image_cells: 0,
        lookahead: false,
        chain_capacity: std::num::NonZeroUsize::MIN,
        conversations: std::num::NonZeroUsize::MIN,
    };
    let state = crate::ResourcePlanner::state_plan(
        &definition,
        &load,
        method,
        KvCodec::Dense,
        limits,
        crate::ResourceCapacity {
            domain_bytes: 512 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::Absent,
        },
    )
    .unwrap();
    let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
    let head = definition.head.clone().unwrap();
    let head_block = crate::operators::CausalHeadBlock::of(&head, &head.blocks[0]).unwrap();
    let head_state = state.head_state().unwrap();
    let head_history = head_state.sole_history().unwrap();
    let classes = crate::programs::native_head::head_graph_classes(
        limits,
        head_history.rows as u64,
        head_history.slab_rows,
        head_state.span_limit(),
        method.draft_rows(),
    )
    .unwrap();
    assert!(classes.iter().any(|class| class.steps == 1 && class.shaped));
    assert!(
        classes
            .iter()
            .any(|class| class.entry_rows > crate::operators::routed::fused_graph::DECODE_ROWS)
    );
    let family = crate::programs::native_head::checked_head_family_storage(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        head_block,
        plan.head().unwrap().blocks()[0],
        classes.clone(),
    )
    .unwrap();
    crate::programs::native_head::verify_head_family_certificates(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        head_block,
        plan.head().unwrap().blocks()[0],
        &classes,
    )
    .unwrap();
    assert!(family.storage.workspace > 0);
    assert!(family.storage.output > 0);
    let pools = AssessmentGraphResourceBounds::derive(
        &definition,
        &load,
        &state,
        method,
        KvCodec::Dense,
        limits,
        BackendName::Cpu,
    )
    .unwrap();
    assert_eq!(
        pools.head.unwrap().workspace_bytes,
        family.storage.workspace
    );
    let target = crate::programs::native_target_graph::checked_target_family_storage(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        &state,
        plan.target(),
        limits,
    )
    .unwrap();
    assert_eq!(
        pools.binding_constant_bytes,
        target.binding_constant_bytes + family.binding_constant_bytes
    );

    // Candidate views do not change resource metadata. This fixture
    // explicitly finalizes a selected-row class before forming both recipes.
    let output = manifest
        .target
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == definition.decoder.exit.output.name)
        .unwrap();
    let packed = (output.encoding, output.nbytes);
    output.encoding = Encoding::Q8_0;
    output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
    for backend in [BackendName::Metal, BackendName::Cuda] {
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            crate::planning::resident_layout(crate::ExecutionPath::Native, backend),
            crate::PlannedMethod::Plain,
            crate::planning::fixtures::LIMITS,
        )
        .unwrap()
        .with_readout_candidates(&definition)
        .unwrap();
        let candidate_plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        assert!(matches!(
            candidate_plan.head().unwrap().blocks()[0].projection,
            crate::ReadoutHead::Packed { .. }
        ));
        let load = load
            .finalize_readout(&[crate::ReadoutClassChoice {
                scope: magnitude_family_contracts::WeightScope::Target,
                norm_scope: magnitude_family_contracts::WeightScope::HeadBlock(0),
                prepared: true,
                rows: vec![1],
            }])
            .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let binding = plan.head().unwrap().blocks()[0];
        assert!(matches!(
            binding.projection,
            crate::ReadoutHead::Qualified { .. }
        ));
        // The head's packed readers bind the unplaced target
        // projection, and the assessment lists their kernels over it.
        HostProcessBytes::derive(&definition, &load, &plan, limits, backend).unwrap();
        let classes = crate::programs::native_head::head_graph_classes(
            limits,
            head_history.rows as u64,
            head_history.slab_rows,
            head_state.span_limit(),
            method.draft_rows(),
        )
        .unwrap();
        let family = crate::programs::native_head::checked_head_family_storage(
            backend,
            &load,
            &definition.decoder,
            head_block,
            binding,
            classes.clone(),
        )
        .unwrap();
        assert!(family.storage.workspace > 0, "{backend:?}");
        crate::programs::native_head::verify_head_family_certificates(
            backend,
            &load,
            &definition.decoder,
            head_block,
            binding,
            &classes,
        )
        .unwrap();
    }
    let output = manifest
        .target
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == definition.decoder.exit.output.name)
        .unwrap();
    (output.encoding, output.nbytes) = packed;

    let descriptor = |name: &str, shape: &[u64]| {
        WeightDescriptor::stored(format!("routed_head_{name}"), shape)
    };
    let silu = |gate, up| FeedForwardUp::Gated {
        activation: ActivationFunction::Silu,
        gate,
        up,
    };
    let routed = RoutedFfn {
        experts: 4,
        selected: 2,
        intermediate: 64,
        router: Router {
            weight: descriptor("router", &[4, hidden]),
            input: RouterInput::Operator,
            score: ScoreFunction::Softmax,
            selection: ExpertSelection::TopK { bias: None },
            normalization: RouteNormalization::Sum,
            scale: 1.0,
        },
        expert_up: silu(
            descriptor("expert_gate", &[4, 64, hidden]),
            descriptor("expert_up", &[4, 64, hidden]),
        ),
        expert_down: descriptor("expert_down", &[4, hidden, 64]),
        expert_scale: None,
        latent: None,
        shared: Some(SharedExpert {
            intermediate: 64,
            up: silu(
                descriptor("shared_gate", &[64, hidden]),
                descriptor("shared_up", &[64, hidden]),
            ),
            down: descriptor("shared_down", &[hidden, 64]),
            gate: SharedExpertGate::Sigmoid(descriptor("shared_router", &[hidden])),
        }),
    };
    for weight in [
        &routed.router.weight,
        routed.expert_up.gate().unwrap(),
        routed.expert_up.up(),
        &routed.expert_down,
        routed.shared.as_ref().unwrap().up.gate().unwrap(),
        routed.shared.as_ref().unwrap().up.up(),
        &routed.shared.as_ref().unwrap().down,
        match &routed.shared.as_ref().unwrap().gate {
            SharedExpertGate::Sigmoid(gate) => gate,
            SharedExpertGate::None => unreachable!(),
        },
    ] {
        let nbytes = weight.shape.iter().product::<u64>() * 2;
        manifest.target.tensors.push(TensorDescriptor {
            name: weight.name.clone(),
            shape: weight.shape.clone(),
            encoding: Encoding::F16,
            offset,
            nbytes,
        });
        offset += nbytes;
    }
    manifest.target.files[0].size = offset;
    definition.head.as_mut().unwrap().blocks[0].blocks[0].sublayers[1].op =
        Operator::RoutedFfn(Box::new(routed));
    let head = definition.head.clone().unwrap();
    let head_block = crate::operators::CausalHeadBlock::of(&head, &head.blocks[0]).unwrap();
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: true,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
    assert!(matches!(
        plan.head().unwrap().blocks()[0].feed_forward,
        crate::FeedForwardProgramSlot::Routed(_)
    ));
    let routed_family = crate::programs::native_head::checked_head_family_storage(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        head_block,
        plan.head().unwrap().blocks()[0],
        crate::programs::native_head::head_graph_classes(
            limits,
            head_history.rows as u64,
            head_history.slab_rows,
            head_state.span_limit(),
            method.draft_rows(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(routed_family.storage.workspace > 0);
    let routed_pools = AssessmentGraphResourceBounds::derive(
        &definition,
        &load,
        &state,
        method,
        KvCodec::Dense,
        limits,
        BackendName::Cpu,
    )
    .unwrap();
    assert_eq!(
        routed_pools.head.unwrap().workspace_bytes,
        routed_family.storage.workspace
    );
    // Raw stream conditioning has independent target and head readout
    // features. Certify priming, prefix, chained and shaped classes with
    // the same topology that execution uses.
    definition.decoder = crate::planning::fixtures::hyper_connection_fixture_definition().decoder;
    let mut site = match &definition.decoder.exit.norm {
        ExitNorm::HyperConnectionHead(site) => site.clone(),
        _ => unreachable!(),
    };
    let head = &mut definition.head.as_mut().unwrap().blocks[0];
    head.residual = definition.decoder.residual.clone();
    head.chain = HeadChain::Residual;
    let HeadInput::Combine { hidden_norm, .. } = &mut head.input else {
        unreachable!()
    };
    hidden_norm.weight.shape = vec![4 * hidden];
    for (index, sublayer) in head.blocks[0].sublayers.iter_mut().enumerate() {
        site.norm.weight.name = format!("head_mix_{index}_norm");
        site.down.name = format!("head_mix_{index}_down");
        site.up.name = format!("head_mix_{index}_up");
        site.inject = Some(descriptor(&format!("mix_{index}_inject"), &[4, 4 * hidden]));
        sublayer.input = InputNorm::HyperConnectionMix(site.clone());
        sublayer.output = magnitude_family_contracts::OutputForm::Scatter;
    }
    site.norm.weight.name = "head_exit_norm".into();
    site.down.name = "head_exit_down".into();
    site.up.name = "head_exit_up".into();
    site.inject = None;
    head.output_norm = ExitNorm::HyperConnectionHead(site);
    let Operator::RoutedFfn(routed) = &mut head.blocks[0].sublayers[1].op else {
        unreachable!()
    };
    // CUDA routes whole groups of 32 experts.
    routed.experts = 32;
    routed.router.weight.shape[0] = 32;
    if let FeedForwardUp::Gated { gate, up, .. } = &mut routed.expert_up {
        gate.shape[0] = 32;
        up.shape[0] = 32;
    }
    routed.expert_down.shape[0] = 32;
    definition.validate().unwrap();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: true,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    assert_eq!(load.residual_conditioning(), Some(4 * hidden));
    let method = PlannedMethod::Mtp {
        greedy_proposals: 3,
        sampled_proposals: 3,
        widths: crate::DraftWidths::Adaptive,
    };
    let state = crate::ResourcePlanner::state_plan(
        &definition,
        &load,
        method,
        KvCodec::Dense,
        limits,
        crate::ResourceCapacity {
            domain_bytes: 512 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::Absent,
        },
    )
    .unwrap();
    let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
    let binding = plan.head().unwrap().blocks()[0];
    assert_eq!(binding.conditioning, Element::f32());
    assert_eq!(binding.streams.unwrap().geometry.streams, 4);
    let head = definition.head.as_ref().unwrap();
    let block = crate::operators::CausalHeadBlock::of(head, &head.blocks[0]).unwrap();
    let head_state = state.head_state().unwrap();
    let history = head_state.sole_history().unwrap();
    let classes = crate::programs::native_head::head_graph_classes(
        limits,
        history.rows as u64,
        history.slab_rows,
        head_state.span_limit(),
        method.draft_rows(),
    )
    .unwrap();
    for backend in [BackendName::Cpu, BackendName::Cuda] {
        crate::programs::native_head::verify_head_family_certificates(
            backend,
            &load,
            &definition.decoder,
            block,
            binding,
            &classes,
        )
        .unwrap();
        let pools = AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            method,
            KvCodec::Dense,
            limits,
            backend,
        )
        .unwrap();
        assert!(pools.head.unwrap().workspace_bytes > 0);
    }
}

/// A draft head of its own width that reads the target's history and is
/// stored in the draft component (the Gemma 4 assistant's form): the
/// fixture target's layer under head names at width 96, its attention
/// Shared with the target's only attention layer.
fn target_reading_head_fixture() -> (ModelDefinition, magnitude_artifacts::PackageManifest) {
    use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
    use magnitude_artifacts::{ArtifactIdentity, ComponentFile, ComponentManifest};
    use magnitude_family_contracts::{
        ExitNorm, FeedForwardUp, Head, HeadBlock, HeadChain, HeadComponent, HeadHistory,
        HeadInput, HeadNorm, HeadReadout, InputNorm, KeyValue, Operator, RmsNorm,
        SublayerIndex, VocabularyProjection, WeightDescriptor,
    };
    const WIDTH: u64 = 96;
    let mut definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let (hidden, vocabulary) = (definition.decoder.hidden, definition.decoder.vocabulary);
    let descriptor =
        |name: &str, shape: &[u64]| WeightDescriptor::stored(format!("head_{name}"), shape);
    let rms = |name: &str, width: u64| RmsNorm {
        weight: descriptor(name, &[width]),
        epsilon: 1e-6,
    };
    let mut block = definition.decoder.blocks[0].clone();
    for (index, sublayer) in block.sublayers.iter_mut().enumerate() {
        sublayer.input = InputNorm::Rms(rms(&format!("input_norm_{index}"), WIDTH));
        match &mut sublayer.op {
            Operator::Attention(attention) => {
                let [rows, _] = attention.query.shape[..] else {
                    unreachable!()
                };
                attention.query = descriptor("query", &[rows, WIDTH]);
                let [_, columns] = attention.output.shape[..] else {
                    unreachable!()
                };
                attention.output = descriptor("attention_output", &[WIDTH, columns]);
                if let HeadNorm::Rms(norm) = &mut attention.query_norm {
                    norm.weight = descriptor("query_norm", &norm.weight.shape);
                }
                attention.key_value = KeyValue::Shared {
                    source: SublayerIndex {
                        block: 0,
                        sublayer: 0,
                    },
                };
            }
            Operator::DenseFfn(dense) => {
                let FeedForwardUp::Gated { gate, up, .. } = &mut dense.up else {
                    unreachable!()
                };
                *gate = descriptor("gate", &[dense.intermediate, WIDTH]);
                *up = descriptor("up", &[dense.intermediate, WIDTH]);
                dense.down = descriptor("down", &[WIDTH, dense.intermediate]);
            }
            _ => unreachable!(),
        }
    }
    definition.head = Some(Head {
        blocks: vec![HeadBlock {
            residual: magnitude_family_contracts::ResidualForm::Single,
            input: HeadInput::Projected {
                pre: descriptor("pre_projection", &[WIDTH, 2 * hidden]),
                embedding_scale: definition.decoder.entry.scale,
            },
            blocks: vec![block],
            output_norm: ExitNorm::Rms(rms("output_norm", WIDTH)),
            readout: HeadReadout::full(VocabularyProjection::Own {
                table: descriptor("table", &[vocabulary, WIDTH]),
            }),
            chain: HeadChain::Projected {
                post: descriptor("post_projection", &[hidden, WIDTH]),
            },
        }],
        history: HeadHistory::Target,
        component: HeadComponent::Draft,
    });
    definition.validate().unwrap();
    let mut offset = 0;
    let tensors = crate::operators::head_weights(definition.head.as_ref().unwrap())
        .unwrap()
        .into_iter()
        .map(|(_, descriptor)| {
            let nbytes = descriptor.shape.iter().product::<u64>() * 2;
            let tensor = TensorDescriptor {
                name: descriptor.name.clone(),
                shape: descriptor.shape.clone(),
                encoding: Encoding::F16,
                offset,
                nbytes,
            };
            offset += nbytes;
            tensor
        })
        .collect();
    let manifest = magnitude_artifacts::PackageManifest {
        draft: Some(ComponentManifest {
            files: vec![ComponentFile {
                path: "assistant-fixture.gguf".into(),
                size: offset,
            }],
            identity: ArtifactIdentity([9; 32]),
            tensors,
        }),
        ..manifest
    };
    (definition, manifest)
}

/// A head reading target history plans its own program over the draft
/// component's weights, keeps no state store, and its checked graph
/// family is the head charge of header assessment.
#[test]
fn target_reading_head_plans_its_program_weights_and_charge() {
    let (definition, manifest) = target_reading_head_fixture();
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: true,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    // Every head weight, its own vocabulary table and both projections
    // included, is the draft component's.
    let head_weights = load.head.as_ref().unwrap();
    assert_eq!(
        head_weights.len(),
        crate::operators::head_weights(definition.head.as_ref().unwrap())
            .unwrap()
            .len()
    );
    assert!(
        head_weights
            .iter()
            .all(|weight| weight.component.kind == crate::ArtifactComponentKind::Draft)
    );
    let method = PlannedMethod::Mtp {
        greedy_proposals: 3,
        sampled_proposals: 3,
        widths: crate::DraftWidths::Adaptive,
    };
    let limits = fixture_limits(false);
    let state = crate::ResourcePlanner::state_plan(
        &definition,
        &load,
        method,
        KvCodec::Dense,
        limits,
        crate::ResourceCapacity {
            domain_bytes: 512 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::Absent,
        },
    )
    .unwrap();
    assert!(state.head_state().is_none());
    let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
    assert!(plan.head().is_none() && plan.draft().is_none());
    assert!(plan.drafts_with_head());
    let shared = plan.shared_head().unwrap();
    assert_eq!((shared.width, shared.blocks.len()), (96, 1));
    let geometry = crate::programs::native_shared_head::SharedHeadGeometry::new(
        &definition,
        &state,
        shared,
    )
    .unwrap();
    let classes =
        crate::programs::native_shared_head::shared_head_graph_classes(limits, 3).unwrap();
    assert!(classes.iter().any(|class| class.steps == 3 && class.shaped));
    let family = crate::programs::native_shared_head::checked_shared_head_family_storage(
        BackendName::Cpu,
        &load,
        &geometry,
        classes,
    )
    .unwrap();
    assert!(family.storage.workspace > 0);
    assert!(family.storage.output > 0);
    let pools = AssessmentGraphResourceBounds::derive(
        &definition,
        &load,
        &state,
        method,
        KvCodec::Dense,
        limits,
        BackendName::Cpu,
    )
    .unwrap();
    assert_eq!(
        pools.head.unwrap().workspace_bytes,
        family.storage.workspace
    );
}

/// A separate draft keeps its full packed logits: the projection it
/// reads, the target's or its own, is stored packed and never as views.
#[test]
fn a_projection_a_separate_draft_reads_is_stored_packed() {
    use magnitude_artifacts::gguf::{Encoding, TensorDescriptor};
    use magnitude_artifacts::{ArtifactIdentity, ComponentFile, ComponentManifest};
    use magnitude_family_contracts::{
        BlockAttention, BlockLayout, DraftDefinition, DraftEmbedding, DraftMethod, RmsNorm,
        TapPoint, TokenId, WeightDescriptor,
    };
    for owns_output in [false, true] {
        let mut definition = crate::planning::fixtures::fixture_definition();
        let mut manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let (hidden, vocabulary) = (definition.decoder.hidden, definition.decoder.vocabulary);
        let norm = |name: &str| RmsNorm {
            weight: WeightDescriptor::stored(name, &[hidden]),
            epsilon: 1e-6,
        };
        definition.draft = Some(DraftDefinition {
            method: DraftMethod::DFlash,
            convolutions: None,
            taps: vec![TapPoint::Exit],
            fusion: WeightDescriptor::stored("draft_fusion", &[hidden, hidden]),
            fusion_norm: norm("draft_fusion_norm"),
            embedding: DraftEmbedding::Target,
            output: owns_output
                .then(|| WeightDescriptor::stored("draft_output", &[vocabulary, hidden])),
            embedding_scale: 1.0,
            blocks: definition.decoder.blocks.clone(),
            block_attention: vec![BlockAttention::Bidirectional],
            output_norm: norm("draft_output_norm"),
            logit_softcap: Some(0.1),
            block_size: 2,
            mask_token: TokenId(0),
            layout: BlockLayout::MaskSlots,
        });
        definition.validate().unwrap();
        let draft = definition.draft.as_ref().unwrap();
        let mut offset = 0;
        let tensors = crate::operators::draft::fusion_weights(draft)
            .into_iter()
            .chain(crate::operators::draft::draft_weights(draft).unwrap())
            .map(|(_, descriptor)| {
                let nbytes = descriptor.shape.iter().product::<u64>() * 2;
                let tensor = TensorDescriptor {
                    name: descriptor.name.clone(),
                    shape: descriptor.shape.clone(),
                    encoding: Encoding::F16,
                    offset,
                    nbytes,
                };
                offset += nbytes;
                tensor
            })
            .collect();
        manifest.draft = Some(ComponentManifest {
            files: vec![ComponentFile {
                path: "draft-fixture.gguf".into(),
                size: offset,
            }],
            identity: ArtifactIdentity([9; 32]),
            tensors,
        });
        let (component, descriptor) = match &draft.output {
            Some(output) => (manifest.draft.as_mut().unwrap(), output),
            None => (&mut manifest.target, &definition.decoder.exit.output),
        };
        let output = component
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == descriptor.name)
            .unwrap();
        output.encoding = Encoding::Q8_0;
        output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: true,
                vision: false,
            },
            Layout::Rows16,
            crate::PlannedMethod::Plain,
            crate::planning::fixtures::LIMITS,
        )
        .unwrap()
        .with_readout_candidates(&definition)
        .unwrap();
        let candidate_plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        assert!(matches!(
            candidate_plan.draft().unwrap().projection(),
            crate::ReadoutHead::Packed { .. }
        ));
        // A weight has one stored form, and the draft has no kernel for
        // the views: a projection it reads is not offered as them,
        // whether it is the target's or the draft's own, so no reader
        // of it can select a class.
        let read_by_draft = if owns_output {
            magnitude_family_contracts::WeightScope::Draft
        } else {
            magnitude_family_contracts::WeightScope::Target
        };
        assert!(
            !load
                .readout_candidates()
                .iter()
                .any(|candidate| candidate.role.scope == read_by_draft),
            "a projection the draft reads is not offered as views"
        );
        assert!(
            load.clone()
                .finalize_readout(&[crate::ReadoutClassChoice {
                    scope: magnitude_family_contracts::WeightScope::Target,
                    norm_scope: magnitude_family_contracts::WeightScope::Target,
                    prepared: false,
                    rows: vec![1],
                }])
                .is_err()
        );
        let load = load.finalize_readout(&[]).unwrap();
        let limits = fixture_limits(false);
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::DFlash {
                proposals: 1,
                widths: crate::DraftWidths::Adaptive,
            },
            KvCodec::Dense,
            limits,
            crate::ResourceCapacity {
                domain_bytes: 512 << 20,
                tensor_operations: crate::TensorOperations::Absent,
            },
        )
        .unwrap();
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let draft_plan = plan.draft().unwrap();
        // The draft's block pass projects every proposing row through
        // the packed weight, whichever component owns it, and the
        // target's readout reads the same one form.
        assert!(matches!(
            plan.target().readout().head,
            crate::ReadoutHead::Packed { .. }
        ));
        assert!(matches!(
            draft_plan.projection(),
            crate::ReadoutHead::Packed { .. }
        ));
        assert_eq!(
            draft_plan.projection_scope(),
            if owns_output {
                magnitude_family_contracts::WeightScope::Draft
            } else {
                magnitude_family_contracts::WeightScope::Target
            }
        );
        let geometry = crate::programs::native_draft::DraftGeometry::new(
            &definition,
            state.head_state().unwrap(),
        )
        .unwrap();
        let class = crate::programs::native_draft::DraftGraphClass {
            entry_rows: 1,
            slots: 1,
            proposals: 1,
            shaped: false,
            windowed: true,
            device: false,
        };
        let family = crate::programs::native_draft::checked_draft_family_storage(
            BackendName::Metal,
            &load,
            draft_plan,
            &geometry,
            [class],
        )
        .unwrap();
        assert!(family.storage.workspace > 0 && family.storage.output > 0);
    }
}

/// A head reading target history whose embedding scale differs from
/// the target's entry transform is refused: its graphs gather rows
/// through the target's embedding entry.
#[test]
fn target_reading_head_must_scale_the_embedding_as_its_target() {
    use magnitude_family_contracts::{EmbeddingScale, HeadInput};
    let (mut definition, _) = target_reading_head_fixture();
    let head = definition.head.as_mut().unwrap();
    let HeadInput::Projected {
        embedding_scale, ..
    } = &mut head.blocks[0].input
    else {
        unreachable!()
    };
    assert_eq!(*embedding_scale, EmbeddingScale::Unit);
    *embedding_scale = EmbeddingScale::SqrtHidden;
    assert!(matches!(
        crate::operators::admit(&definition, true),
        Err(crate::PlanError::Unsupported(
            "a draft head scaling the target's embedding differently"
        ))
    ));
    // An unselected head does not bear on the model.
    crate::operators::admit(&definition, false).unwrap();
}

#[test]
fn checked_feature_readout_charges_only_its_export_and_upload() {
    use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

    let definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let norm = load
        .weights()
        .find(|weight| {
            weight.role
                == WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::OutputNorm,
                }
        })
        .unwrap();
    let storage = crate::programs::graph::readout::checked_features_graph_storage(
        BackendName::Cpu,
        &definition.decoder,
        Some(norm),
        2,
        1,
    )
    .unwrap();
    assert_eq!(storage.workspace, 0);
    assert_eq!(storage.upload, 4);
    assert!(storage.output > 0);
}

#[test]
fn checked_projected_readout_includes_native_scratch() {
    use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

    let definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let weight = |kind| {
        load.weights()
            .find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind,
                    }
            })
            .unwrap()
    };
    let features = crate::programs::graph::readout::checked_features_graph_storage(
        BackendName::Cpu,
        &definition.decoder,
        Some(weight(WeightKind::OutputNorm)),
        2,
        1,
    )
    .unwrap();
    let projected = crate::programs::graph::readout::checked_projected_graph_storage(
        BackendName::Cpu,
        &definition.decoder,
        Some(weight(WeightKind::OutputNorm)),
        weight(WeightKind::Output),
        2,
        1,
        1,
    )
    .unwrap();
    assert!(projected.workspace > features.workspace);
    assert!(projected.output > features.output);
    assert!(projected.upload > features.upload);
}

#[test]
fn checked_selection_readout_charges_shaping_controls() {
    use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};

    let definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let weight = |kind| {
        load.weights()
            .find(|weight| {
                weight.role
                    == WeightRole {
                        scope: WeightScope::Target,
                        kind,
                    }
            })
            .unwrap()
    };
    let selection = |shaped| {
        crate::programs::graph::readout::checked_selection_graph_storage(
            BackendName::Cpu,
            &definition.decoder,
            Some(weight(WeightKind::OutputNorm)),
            weight(WeightKind::Output),
            2,
            1,
            1,
            shaped,
        )
        .unwrap()
    };
    let plain = selection(false);
    let shaped = selection(true);
    assert!(shaped.workspace > plain.workspace);
    assert_eq!(shaped.output, plain.output);
    assert!(shaped.upload > plain.upload);
}

#[test]
fn checked_readout_family_bounds_its_selection_classes() {
    let definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        Layout::Rows16,
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap();
    let limits = crate::ResourceLimits {
        max_launch_rows: 2,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: 0,
        max_images_per_request: 0,
        max_image_cells: 0,
        lookahead: false,
        chain_capacity: std::num::NonZeroUsize::MIN,
        conversations: std::num::NonZeroUsize::MIN,
    };
    let family = crate::programs::graph::readout::checked_readout_family_storage(
        BackendName::Cpu,
        &load,
        &definition.decoder,
        limits,
    )
    .unwrap();
    assert!(family.workspace > 0);
    assert!(family.output > 0);
    assert!(family.upload > 0);
}

#[test]
fn checked_vision_family_bounds_exact_patch_classes() {
    use crate::{ArtifactComponent, ArtifactComponentKind, VisionProgramPlan, WeightPlan};
    use magnitude_artifacts::ArtifactIdentity;
    use magnitude_family_contracts::{
        ActivationDType, CellReduction, MergerStage, PositionSampling, VisionActivation,
        VisionAttention, VisionOperator, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
        VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm,
        VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem,
        VisionUp, WeightDescriptor,
    };

    let f32 = Element::f32();
    let stored = |name: &str, shape: &[u64]| WeightDescriptor::stored(name, shape);
    let linear = |name: &str, outputs: u64, inputs: u64| VisionLinear {
        weight: stored(&format!("{name}.weight"), &[outputs, inputs]),
        bias: Some(stored(&format!("{name}.bias"), &[outputs])),
        clamp: None,
    };
    let layer_norm = |name: &str, width: u64| VisionNorm::Layer {
        weight: stored(&format!("{name}.weight"), &[width]),
        bias: stored(&format!("{name}.bias"), &[width]),
        epsilon: 1e-6,
    };
    // A Qwen3-VL-shaped tower at width 128: one block of one 128-wide
    // head, 2 x 2 cells.
    let description = VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden: 128,
        output_hidden: 128,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PixelBounds {
                min_pixels: 16,
                max_pixels: 64,
            },
            resampling: VisionResampling::Bicubic,
            mean: [0.5; 3],
            std: [0.5; 3],
            channels: 3,
            patch: 2,
            merge: 2,
        },
        stem: VisionStem::Patch {
            frames: vec![
                stored("patch.0", &[128, 3, 2, 2]),
                stored("patch.1", &[128, 3, 2, 2]),
            ],
            bias: Some(stored("patch.bias", &[128])),
            positions: Some(VisionPositions {
                table: stored("position", &[16, 128]),
                sampling: PositionSampling::AlignedCorners { side: 4 },
            }),
            norm: None,
            registers: None,
        },
        window: None,
        blocks: vec![VisionBlock {
            operator_norm: layer_norm("ln1", 128),
            operator: VisionOperator::Attention(VisionAttention {
                heads: 1,
                kv_heads: 1,
                width: 128,
                query: linear("q", 128, 128),
                key: linear("k", 128, 128),
                value: linear("v", 128, 128),
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: Some(10000.0),
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: VisionAttentionSpan::Full,
                output: linear("o", 128, 128),
            }),
            operator_post_norm: None,
            feedforward_norm: layer_norm("ln2", 128),
            feedforward: VisionFeedForward {
                up: VisionUp::Plain(linear("up", 128, 128)),
                activation: VisionActivation::GeluTanh,
                down: linear("down", 128, 128),
            },
            feedforward_post_norm: None,
        }],
        merger: VisionMerger {
            norm: Some(layer_norm("post", 128)),
            reduction: CellReduction::Concatenate,
            standardize: None,
            projection_norm: None,
            stages: vec![
                MergerStage {
                    linear: linear("mm.0", 512, 512),
                    activation: Some(VisionActivation::GeluErf),
                },
                MergerStage {
                    linear: linear("mm.2", 128, 512),
                    activation: None,
                },
            ],
            output_norm: None,
        },
    };
    // A C-RADIO-shaped tower: three register rows ahead of the patches
    // (no patch bias), no rotary, and the projection's RMS norm over
    // each concatenated cell before a squared-ReLU stage.
    let mut radio = description.clone();
    radio.stem = VisionStem::Patch {
        frames: vec![stored("patch.0", &[128, 3, 2, 2])],
        bias: None,
        positions: Some(VisionPositions {
            table: stored("position", &[16, 128]),
            sampling: PositionSampling::CroppedSquare { side: 4 },
        }),
        norm: None,
        registers: Some(stored("registers", &[3, 128])),
    };
    let VisionOperator::Attention(attention) = &mut radio.blocks[0].operator else {
        unreachable!("the fixture block attends")
    };
    attention.rotary_base = None;
    radio.merger = VisionMerger {
        norm: None,
        reduction: CellReduction::Concatenate,
        standardize: None,
        projection_norm: Some(VisionNorm::Rms {
            weight: Some(stored("mm.norm.weight", &[512])),
            epsilon: 1e-5,
        }),
        stages: vec![
            MergerStage {
                linear: VisionLinear {
                    bias: None,
                    ..linear("mm.1", 256, 512)
                },
                activation: Some(VisionActivation::ReluSquared),
            },
            MergerStage {
                linear: VisionLinear {
                    bias: None,
                    ..linear("mm.3", 128, 256)
                },
                activation: None,
            },
        ],
        output_norm: None,
    };
    radio.validate().unwrap();
    for description in [description, radio] {
        let component = ArtifactComponent {
            kind: ArtifactComponentKind::Projector,
            identity: ArtifactIdentity([9; 32]),
        };
        let weights = description
            .weights()
            .into_iter()
            .map(|(role, descriptor)| {
                let bytes = descriptor.shape.iter().product::<u64>() * 4;
                WeightPlan {
                    role,
                    component,
                    source: f32,
                    upload: f32,
                    resident: f32,
                    shape: descriptor.shape.clone(),
                    descriptor: descriptor.clone(),
                    source_bytes: bytes,
                    resident_bytes: bytes,
                    scale: None,
                    bound: 0,
                    holding: crate::Holding::Whole,
                }
            })
            .collect();
        let load = ModelLoadPlan {
            residual_conditioning: None,
            target: Vec::new(),
            head: None,
            vision: Some(weights),
            host_tables: Vec::new(),
            readout_candidates: Vec::new(),
            readout_choices: Vec::new(),
            unplaced_output: Vec::new(),
            bank: None,
            record_ports: None,
        };
        let program =
            crate::operators::vision::vision_program(&description, &|_| Ok(f32)).unwrap();
        assert_eq!(program.registers, description.stem.registers());
        let plan = VisionProgramPlan::new(program.kernels().into_iter().cloned().collect());
        let resources = |classes: &[u64]| {
            crate::programs::native_vision::checked_vision_family_resources(
                BackendName::Cpu,
                &load,
                &description,
                &plan,
                classes.iter().copied(),
            )
            .unwrap()
            .storage
        };
        let single = resources(&[4]);
        let double = resources(&[8]);
        let family = resources(&[4, 8]);
        crate::programs::native_vision::verify_vision_family_certificates(
            BackendName::Cpu,
            &load,
            &description,
            &plan,
            &[4, 8],
        )
        .unwrap();
        assert_eq!(family.workspace, single.workspace.max(double.workspace));
        assert_eq!(family.output, single.output.max(double.output));
        assert_eq!(family.upload, single.upload.max(double.upload));
        assert!(family.workspace > 0 && family.output > 0 && family.upload > 0);
    }
}

/// A load whose readout selected planes stores its packed output
/// projection unplaced, and its packed readers still bind it: the
/// assessment's kernel listing over the finalized plan resolves it.
#[test]
fn kernels_of_a_finalized_target_readout_resolve_the_unplaced_projection() {
    let definition = crate::planning::fixtures::fixture_definition();
    let mut manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let output = manifest
        .target
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == definition.decoder.exit.output.name)
        .unwrap();
    output.encoding = magnitude_artifacts::gguf::Encoding::Q8_0;
    output.nbytes = output.shape.iter().product::<u64>() / 32 * 34;
    let backend = BackendName::Metal;
    let load = ModelLoadPlan::derive(
        &manifest,
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        crate::planning::resident_layout(crate::ExecutionPath::Native, backend),
        crate::PlannedMethod::Plain,
        crate::planning::fixtures::LIMITS,
    )
    .unwrap()
    .with_readout_candidates(&definition)
    .unwrap()
    .finalize_readout(&[crate::ReadoutClassChoice {
        scope: magnitude_family_contracts::WeightScope::Target,
        norm_scope: magnitude_family_contracts::WeightScope::Target,
        prepared: false,
        rows: vec![1],
    }])
    .unwrap();
    assert!(
        !load.weights().any(|placed| placed.role
            == magnitude_family_contracts::WeightRole {
                scope: magnitude_family_contracts::WeightScope::Target,
                kind: magnitude_family_contracts::WeightKind::Output,
            }),
        "the selected scope's packed projection is not placed"
    );
    let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
    HostProcessBytes::derive(&definition, &load, &plan, fixture_limits(false), backend)
        .unwrap();
}

fn fixture_limits(lookahead: bool) -> crate::ResourceLimits {
    crate::ResourceLimits {
        max_launch_rows: 2,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: 0,
        max_images_per_request: 1,
        max_image_cells: 0,
        lookahead,
        chain_capacity: std::num::NonZeroUsize::MIN,
        conversations: std::num::NonZeroUsize::MIN,
    }
}

#[test]
fn header_terms_include_exact_selected_weights_history_and_bounds() {
    let mut definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let selection = ComponentSelection {
        head: false,
        vision: false,
    };
    let load =
        ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows16, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    let limits = fixture_limits(false);
    let terms = AssessmentMemoryTerms::derive(
        &definition,
        &load,
        selection,
        KvCodec::Dense,
        PlannedMethod::Plain,
        limits,
    )
    .unwrap();

    assert_eq!(
        terms.target_weights,
        crate::planning::weight_bytes_by_component(&load).unwrap()[0]
    );
    assert_eq!(terms.head_weights, 0);
    assert_eq!(terms.vision_weights, 0);
    assert_eq!(terms.history_per_token, 2 * 64 * 2);
    assert_eq!(terms.recurrent_per_bank, 0);
    // Accepted bank, one in-flight successor, the prompt-boundary,
    // terminal and turn-start checkpoints, pristine seed.
    assert_eq!(terms.recurrent_banks, 6);
    assert_eq!(terms.fit_depth, 128);
    assert_eq!(terms.history_at_fit_depth().unwrap(), 128 * 2 * 64 * 2);
    let pipelined = AssessmentMemoryTerms::derive(
        &definition,
        &load,
        selection,
        KvCodec::Dense,
        PlannedMethod::Plain,
        fixture_limits(true),
    )
    .unwrap();
    assert_eq!(pipelined.recurrent_banks, 7);
    assert!(
        AssessmentMemoryTerms::derive(
            &definition,
            &load,
            selection,
            KvCodec::Dense,
            PlannedMethod::Mtp {
                greedy_proposals: 1,
                sampled_proposals: 1,
                widths: crate::DraftWidths::Adaptive,
            },
            limits,
        )
        .is_err()
    );

    let header = AssessmentHeaderBounds::derive(&load).unwrap();
    assert_eq!(
        header.startup_additional_bytes,
        AttestedPrograms::qualification_peak_bytes(&load)
            .unwrap()
            .max(load.target_upload_peak_bytes().unwrap())
    );
    assert_eq!(
        header.staging_upload_bytes,
        load.target_upload_peak_bytes().unwrap()
    );
    assert_eq!(
        header.component_upload_bytes,
        load.head_upload_peak_bytes()
            .unwrap()
            .max(load.vision_upload_peak_bytes().unwrap())
    );
    let state = crate::ResourcePlanner::state_plan(
        &definition,
        &load,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
        crate::ResourceCapacity {
            domain_bytes: 512 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::Absent,
        },
    )
    .unwrap();
    let graph = AssessmentGraphResourceBounds::derive(
        &definition,
        &load,
        &state,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
        BackendName::Cpu,
    )
    .unwrap();
    let host_process = HostProcessBytes::derive(
        &definition,
        &load,
        &load.program_plan(&definition, KvCodec::Dense).unwrap(),
        limits,
        BackendName::Cpu,
    )
    .unwrap();
    let device_process =
        DeviceProcessBytes::derive(BackendName::Cpu, host_process.kernels).unwrap();
    // The CPU records no graphs: its startup graphs hold nothing.
    let bounds = header
        .with_graph_resource_bound(&graph, seismic::GraphCost::None, host_process, device_process)
        .unwrap();
    assert_eq!(bounds.host_process, host_process);
    assert_eq!(bounds.device_process, device_process);
    assert!(graph.recorded_graphs.launches > 0);
    assert_eq!(bounds.recorded_graph_bytes, 0);
    assert_eq!(
        bounds.prepared_resource_bytes().unwrap(),
        AttestedPrograms::DEVICE_BYTES + graph.total_bytes + device_process.bytes().unwrap()
    );
    assert_eq!(
        bounds.startup_additional_bytes,
        header.startup_additional_bytes
    );
    let state_bytes = state.capacity().workload_state_bytes;
    let history = state.target_state().sole_history().unwrap();
    let history_layout = history.slab_layout().unwrap();
    let history_slabs = terms.fit_depth.div_ceil(u64::from(history.slab_rows));
    assert_eq!(
        state_bytes,
        history_layout.address_table_bytes + history_slabs * history_layout.slab_bytes
    );
    assert!(state_bytes > terms.history_at_fit_depth().unwrap());
    let startup_state = state.startup_state_bytes().unwrap();
    // A warm-up's successor bank, held once loaded: no recurrent state
    // here, so no more than the startup slabs.
    assert_eq!(state.loaded_state_bytes().unwrap(), startup_state);
    let charge = terms
        .charge(bounds, state_bytes, startup_state, startup_state)
        .unwrap();
    assert_eq!(
        charge.allocation_bytes,
        terms.exact_resident_bytes(0).unwrap()
            + bounds.prepared_resource_bytes().unwrap()
            + (state_bytes + bounds.component_upload_bytes)
                .max(startup_state + bounds.startup_additional_bytes)
    );
    assert_eq!(charge.staging_bytes, header.staging_upload_bytes);

    // The fit depth is the context limit capped at 100,000 tokens.
    definition.decoder.context_limit = 262_144;
    let long = AssessmentMemoryTerms::derive(
        &definition,
        &load,
        selection,
        KvCodec::Dense,
        PlannedMethod::Plain,
        limits,
    )
    .unwrap();
    assert_eq!(long.fit_depth, 100_000);
    assert_eq!(
        long.history_at_fit_depth().unwrap(),
        100_000 * terms.history_per_token
    );
}

const NO_HOST_PROCESS: HostProcessBytes = HostProcessBytes {
    runtime: 0,
    vocabulary: 0,
    programs: 0,
    vision: 0,
    kernels: 0,
};

const NO_DEVICE_PROCESS: DeviceProcessBytes = DeviceProcessBytes {
    context: 0,
    programs: 0,
    slack: 0,
};

fn fixture_charge(allocation_bytes: u64, staging_bytes: u64) -> AssessmentMemoryCharge {
    AssessmentMemoryCharge {
        exact_resident_bytes: allocation_bytes,
        state_bytes: 0,
        startup_state_bytes: 0,
        loaded_state_bytes: 0,
        bounds: AssessmentMemoryBounds {
            graph_bytes: 0,
            recorded_graph_bytes: 0,
            startup_additional_bytes: 0,
            component_upload_bytes: 0,
            staging_upload_bytes: staging_bytes,
            sealed_graph_host_bytes: SealedGraphHostBytes {
                at_most: 0,
                at_least: 0,
            },
            bound_graph_host_bytes: 0,
            drafter_graph_host_bytes: 0,
            host_process: NO_HOST_PROCESS,
            device_process: NO_DEVICE_PROCESS,
        },
        serving_bytes: allocation_bytes,
        allocation_bytes,
        staging_bytes,
        host_table_bytes: 0,
        weight_demand: ClaimDemand::default(),
    }
}

#[test]
fn workload_charge_composes_exact_terms_and_upper_bounds() {
    let terms = AssessmentMemoryTerms {
        target_weights: 100,
        head_weights: 20,
        vision_weights: 5,
        history_per_token: 2,
        recurrent_per_bank: 10,
        recurrent_banks: 3,
        fit_depth: 10,
        conversations: 1,
        host_table_bytes: 30,
        weight_demand: ClaimDemand::default(),
    };
    assert_eq!(terms.exact_resident_bytes(50).unwrap(), 100 + 20 + 5 + 50);
    let bounds = AssessmentMemoryBounds {
        // 40 prepared resource bytes, the programs' owner among them.
        graph_bytes: 40 - AttestedPrograms::DEVICE_BYTES,
        recorded_graph_bytes: 0,
        startup_additional_bytes: 60,
        component_upload_bytes: 0,
        staging_upload_bytes: 7,
        sealed_graph_host_bytes: SealedGraphHostBytes {
            at_most: 0,
            at_least: 0,
        },
        bound_graph_host_bytes: 0,
        drafter_graph_host_bytes: 0,
        host_process: NO_HOST_PROCESS,
        device_process: NO_DEVICE_PROCESS,
    };
    // The workload's state outgrows the load's startup state and peak:
    // the peak is not charged on top of it.
    let charge = terms.charge(bounds, 90, 10, 15).unwrap();
    assert_eq!(charge.exact_resident_bytes, 100 + 20 + 5 + 90);
    assert_eq!(charge.allocation_bytes, 125 + 40 + 90);
    assert_eq!(charge.serving_bytes, 125 + 40 + 90);
    assert_eq!(charge.staging_bytes, 7);
    // The load's startup state and peak exceed the workload's state:
    // serving holds the workload's state, not the load's peak.
    let loading = terms.charge(bounds, 50, 10, 15).unwrap();
    assert_eq!(loading.allocation_bytes, 125 + 40 + (10 + 60));
    assert_eq!(loading.serving_bytes, 125 + 40 + 50);
    // The loaded state follows the startup peak, so it bounds the load
    // only where it is the larger.
    let warmed = terms.charge(bounds, 50, 10, 80).unwrap();
    assert_eq!(warmed.allocation_bytes, 125 + 40 + 80);
    assert!(terms.charge(bounds, u64::MAX, 10, 15).is_err());
    assert!(terms.charge(bounds, 50, u64::MAX, 15).is_err());
    // An optional component a request loads at the fit depth claims its
    // import window beside the workload's state: serving's peak, and a
    // dedicated device stages the larger of the two windows.
    let component = AssessmentMemoryBounds {
        component_upload_bytes: 30,
        ..bounds
    };
    let loaded = terms.charge(component, 90, 10, 15).unwrap();
    assert_eq!(loaded.allocation_bytes, 125 + 40 + 90 + 30);
    assert_eq!(loaded.serving_bytes, 125 + 40 + 90);
    assert_eq!(loaded.staging_bytes, 30);
    // The load's own peak still bounds it where it is the larger.
    let early = terms.charge(component, 20, 10, 15).unwrap();
    assert_eq!(early.allocation_bytes, 125 + 40 + (10 + 60));
    assert!(
        AssessmentMemoryTerms {
            history_per_token: u64::MAX,
            ..terms
        }
        .history_at_fit_depth()
        .is_err()
    );
}

const GIB: u64 = 1 << 30;

/// Status of an unopened machine whose host RAM `by` reports, sampled
/// for every executor of `topology`: assessment plans on stable fit,
/// which no headroom changes.
fn unopened(topology: &DeviceTopology, by: seismic::ReportedBy) -> Status {
    Status {
        sample: seismic::Sample {
            taken: std::time::SystemTime::now(),
            host_ram: Ok(seismic::HostStatus {
                allocatable: seismic::Reported::new(16 * GIB, by),
                obtainable: seismic::Reported::new(16 * GIB, by),
                limits: Vec::new(),
                limit_visibility: seismic::LimitVisibility::Complete,
                displacement: None,
                kernel_pressure: None,
                held: None,
                displaced_bytes: None,
            }),
            volumes: Vec::new(),
            executors: topology
                .executors()
                .iter()
                .map(|executor| executor.id())
                .collect(),
            grants: Vec::new(),
        },
        runtime: Default::default(),
    }
}

fn executor(topology: &DeviceTopology, backend: BackendName) -> &Executor {
    topology
        .executors()
        .iter()
        .find(|executor| executor.backend() == backend)
        .unwrap()
}

/// The machine's one device memory store.
fn device_memory(topology: &DeviceTopology) -> StoreId {
    topology
        .stores()
        .iter()
        .find(|store| matches!(store.kind(), seismic::StoreKind::DeviceMemory(_)))
        .unwrap()
        .id()
}

/// The fit of `store` in `fit`.
fn domain(fit: &AssessmentFit, store: StoreId) -> DomainFit {
    *fit.domains
        .iter()
        .find(|domain| domain.domain == store)
        .unwrap()
}

/// Each store a load charges is fitted on its own against its capacity
/// less its planning reserve, and the store with the largest deficit
/// limits the load. On the M4 Pro's CPU the load, its host tables and
/// its sealed graph plans are one domain, host RAM; the RTX 3090's CUDA
/// charges its allocation to the 3090's memory and its staged upload,
/// host tables and sealed graph plans to host RAM.
#[test]
fn fit_is_decided_per_domain_against_capacity_less_reserve() {
    let reserves = crate::platform::MemoryReserves::standard();
    let m4_pro = recorded::m4_pro();
    let status = unopened(&m4_pro, seismic::ReportedBy::MachMovablePages);
    let cpu = executor(&m4_pro, BackendName::Cpu);
    let host_ram = m4_pro.host_ram().id();
    let host = Fit::new(&m4_pro, &status, &reserves, HostRoom::Allocatable)
        .store(host_ram)
        .unwrap();
    let mut unified = AssessmentMemoryCharge {
        host_table_bytes: 30,
        ..fixture_charge(host.stable_bytes - 50, 0)
    };
    unified.bounds.sealed_graph_host_bytes = SealedGraphHostBytes {
        at_most: 20,
        at_least: 15,
    };
    let fits = unified
        .assess_fit(&m4_pro, cpu, &status, &reserves)
        .unwrap();
    assert_eq!(fits.verdict, AssessmentFitVerdict::Fits);
    assert_eq!(
        fits.domains,
        [DomainFit {
            role: DomainRole::Allocation,
            domain: host_ram,
            capacity_bytes: host.capacity_bytes,
            required_bytes: host.stable_bytes,
            reserve_bytes: host.thresholds.planning_bytes,
            remaining_bytes: 0,
        }]
    );
    let short = AssessmentMemoryCharge {
        host_table_bytes: 31,
        ..unified
    }
    .assess_fit(&m4_pro, cpu, &status, &reserves)
    .unwrap();
    assert_eq!(
        short.verdict,
        AssessmentFitVerdict::DoesNotFit {
            limiting: host_ram,
            deficit_bytes: NonZeroU64::MIN,
        }
    );
    assert_eq!(short.domains[0].remaining_bytes, -1);

    let rtx3090 = recorded::rtx3090();
    let status = unopened(&rtx3090, seismic::ReportedBy::LinuxMemAvailable);
    let cuda = executor(&rtx3090, BackendName::Cuda);
    let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
    let fit = Fit::new(&rtx3090, &status, &reserves, HostRoom::Allocatable);
    let (own, host) = (fit.store(gpu).unwrap(), fit.store(host_ram).unwrap());
    // CUDA's room is its usable memory less the store's planning
    // reserve, below the store's own fit.
    let room = fit.placement(cuda, seismic::Contract::DeviceOnly).unwrap().stable_bytes;
    assert!(room < own.stable_bytes);
    let dedicated = |allocation, staging| {
        AssessmentMemoryCharge {
            host_table_bytes: 30,
            ..fixture_charge(allocation, staging)
        }
        .assess_fit(&rtx3090, cuda, &status, &reserves)
        .unwrap()
    };
    let both = dedicated(room, host.stable_bytes - 30);
    assert_eq!(both.verdict, AssessmentFitVerdict::Fits);
    assert_eq!(
        domain(&both, gpu),
        DomainFit {
            role: DomainRole::Allocation,
            domain: gpu,
            capacity_bytes: room + own.thresholds.planning_bytes,
            required_bytes: room,
            reserve_bytes: own.thresholds.planning_bytes,
            remaining_bytes: 0,
        }
    );
    assert_eq!(domain(&both, host_ram).role, DomainRole::Staging);
    assert_eq!(domain(&both, host_ram).required_bytes, host.stable_bytes);
    assert_eq!(domain(&both, host_ram).remaining_bytes, 0);
    // Staging 50 bytes beyond host RAM's fit: host RAM limits the load.
    let staging = dedicated(room, host.stable_bytes - 30 + 50);
    assert_eq!(
        staging.verdict,
        AssessmentFitVerdict::DoesNotFit {
            limiting: host_ram,
            deficit_bytes: NonZeroU64::new(50).unwrap(),
        }
    );
    assert_eq!(domain(&staging, gpu).remaining_bytes, 0);
    assert_eq!(domain(&staging, host_ram).remaining_bytes, -50);
    // The largest deficit limits the load.
    let worst = dedicated(room + 100, host.stable_bytes - 30 + 50);
    assert_eq!(
        worst.verdict,
        AssessmentFitVerdict::DoesNotFit {
            limiting: gpu,
            deficit_bytes: NonZeroU64::new(100).unwrap(),
        }
    );
}

/// The graphs' static bindings, the drafter's graphs and the worker's
/// own host bytes are system RAM, held with the host tables and the
/// sealed plans: on the RTX 3090's CUDA they are host RAM's, beside the
/// staged upload, and the device's own memory holds none of them.
#[test]
fn worker_host_bytes_are_charged_to_host_ram() {
    let reserves = crate::platform::MemoryReserves::standard();
    let mut worker = AssessmentMemoryCharge {
        host_table_bytes: 50,
        ..fixture_charge(800, 300)
    };
    worker.bounds.sealed_graph_host_bytes = SealedGraphHostBytes {
        at_most: 20,
        at_least: 15,
    };
    worker.bounds.bound_graph_host_bytes = 3;
    worker.bounds.drafter_graph_host_bytes = 1;
    worker.bounds.host_process = HostProcessBytes {
        runtime: 4,
        vocabulary: 5,
        programs: 6,
        vision: 7,
        kernels: 2,
    };
    assert_eq!(
        worker.host_bytes().unwrap(),
        50 + 20 + 3 + 1 + 4 + 5 + 6 + 7
    );

    let rtx3090 = recorded::rtx3090();
    let status = unopened(&rtx3090, seismic::ReportedBy::LinuxMemAvailable);
    let cuda = executor(&rtx3090, BackendName::Cuda);
    let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
    let fit = worker
        .assess_fit(&rtx3090, cuda, &status, &reserves)
        .unwrap();
    assert_eq!(domain(&fit, gpu).required_bytes, 800);
    assert_eq!(
        domain(&fit, host_ram).required_bytes,
        300 + worker.host_bytes().unwrap()
    );
}

/// The worker's device bytes are the backend's context and a cost per
/// kernel the load prepares; a unified device's are in its host bytes,
/// and a CUDA context is already outside the device's usable capacity.
#[test]
fn worker_device_bytes_price_the_context_and_kernels() {
    for backend in [
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Vulkan,
        BackendName::Cuda,
    ] {
        let costs = DeviceProcessCosts::of(backend);
        let device = DeviceProcessBytes::derive(backend, 40).unwrap();
        assert_eq!(
            device,
            DeviceProcessBytes {
                context: costs.context,
                programs: 40 * costs.per_kernel,
                slack: costs.slack,
            }
        );
        assert_eq!(
            device.bytes().unwrap(),
            device.context + device.programs + device.slack
        );
    }
    for unified in [BackendName::Cpu, BackendName::Metal] {
        let device = DeviceProcessBytes::derive(unified, 40).unwrap();
        assert_eq!(device.bytes().unwrap(), 0);
    }
    // A CUDA context is outside the usable capacity; a dedicated
    // device's block allocator holds free space whatever the model.
    let cuda = DeviceProcessBytes::derive(BackendName::Cuda, 40).unwrap();
    assert_eq!(cuda.context, 0);
    assert!(cuda.slack > 0);
    let vulkan = DeviceProcessBytes::derive(BackendName::Vulkan, 40).unwrap();
    assert!(vulkan.context > 0 && vulkan.programs > 0 && vulkan.slack > 0);
    assert!(DeviceProcessBytes::derive(BackendName::Vulkan, u64::MAX).is_err());
}

/// The worker's host bytes are counts the plan fixes times the costs
/// measured on the backend: its runtime, a cost per vocabulary entry and
/// one per native kernel the load prepares, the kernels being the
/// load's own inventory (`kernel_requests`, which a load's preparation
/// is tested to prepare exactly).
#[test]
fn worker_host_bytes_price_the_plans_vocabulary_and_kernels() {
    let definition = crate::planning::fixtures::fixture_definition();
    let manifest = crate::planning::fixtures::fixture_manifest(&definition);
    let selection = ComponentSelection {
        head: false,
        vision: false,
    };
    let load =
        ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows16, crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS).unwrap();
    let programs = load.program_plan(&definition, KvCodec::Dense).unwrap();
    let limits = fixture_limits(false);
    for backend in [
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Vulkan,
        BackendName::Cuda,
    ] {
        let costs = HostProcessCosts::of(backend);
        let kernels = crate::native::kernel_requests(
            backend,
            &programs,
            &definition,
            &load,
            crate::TuningLimits::of(limits, &definition),
        )
        .unwrap()
        .len() as u64;
        assert!(kernels > 0);
        let host =
            HostProcessBytes::derive(&definition, &load, &programs, limits, backend).unwrap();
        // The fixture reads no images: no vision term.
        assert!(programs.vision().is_none());
        assert_eq!(
            host,
            HostProcessBytes {
                runtime: costs.runtime,
                vocabulary: costs.per_token * definition.decoder.vocabulary,
                programs: costs.per_kernel * kernels,
                vision: 0,
                kernels,
            }
        );
        assert_eq!(
            host.bytes().unwrap(),
            host.runtime + host.vocabulary + host.programs
        );
    }
}

/// The staging window is part of the allocation charge. An executor
/// whose device storage is placed on host RAM (Metal on the M4 Pro,
/// CUDA on GB10, Vulkan on Strix Halo and Lunar Lake) holds it there
/// once, inside the allocation: its staging adds nothing. A dedicated
/// device's staging window (CUDA on the RTX 3090) is host RAM's, beside
/// the allocation in its own memory.
#[test]
fn staging_is_charged_to_host_ram_once() {
    let reserves = crate::platform::MemoryReserves::standard();
    let (allocation, staged, tables) = (8 * GIB, 2 * GIB, 30);
    let assess = |topology: &DeviceTopology, executor: &Executor, status: &Status, staging| {
        AssessmentMemoryCharge {
            host_table_bytes: tables,
            ..fixture_charge(allocation, staging)
        }
        .assess_fit(topology, executor, status, &reserves)
        .unwrap()
    };
    use seismic::ReportedBy::{LinuxMemAvailable, MachMovablePages, WindowsPhysicalAndCommit};
    for (topology, backend, by) in [
        (recorded::m4_pro(), BackendName::Metal, MachMovablePages),
        (recorded::gb10(), BackendName::Cuda, LinuxMemAvailable),
        (
            recorded::strix_halo_linux(),
            BackendName::Vulkan,
            LinuxMemAvailable,
        ),
        (
            recorded::lunar_lake(),
            BackendName::Vulkan,
            WindowsPhysicalAndCommit,
        ),
    ] {
        let status = unopened(&topology, by);
        let executor = executor(&topology, backend);
        let host_ram = topology.host_ram().id();
        let staging = assess(&topology, executor, &status, staged);
        assert_eq!(staging.verdict, AssessmentFitVerdict::Fits);
        assert_eq!(
            domain(&staging, host_ram).required_bytes,
            allocation + tables,
            "{backend:?} on {}",
            executor.name()
        );
        assert_eq!(staging, assess(&topology, executor, &status, 0));
    }

    let rtx3090 = recorded::rtx3090();
    let status = unopened(&rtx3090, seismic::ReportedBy::LinuxMemAvailable);
    let cuda = executor(&rtx3090, BackendName::Cuda);
    let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
    let staging = assess(&rtx3090, cuda, &status, staged);
    assert_eq!(domain(&staging, gpu).required_bytes, allocation);
    assert_eq!(domain(&staging, host_ram).required_bytes, tables + staged);
}

/// Metal's working set bounds only its device storage: host-resident
/// tables are mapped pages, not Metal buffers, so host RAM holds them
/// beside it, and the store's room for the load is the tables with the
/// working set's room.
#[test]
fn host_demand_sits_beside_the_metal_working_set() {
    let reserves = crate::platform::MemoryReserves::standard();
    let m4_pro = recorded::m4_pro();
    let status = unopened(&m4_pro, seismic::ReportedBy::MachMovablePages);
    let metal = executor(&m4_pro, BackendName::Metal);
    let host_ram = m4_pro.host_ram().id();
    let fit = Fit::new(&m4_pro, &status, &reserves, HostRoom::Allocatable);
    let room = fit.placement(metal, seismic::Contract::DeviceOnly).unwrap().stable_bytes;
    let host = fit.store(host_ram).unwrap();
    assert!(room + GIB < host.stable_bytes);
    let assess = |device| {
        AssessmentMemoryCharge {
            host_table_bytes: GIB,
            ..fixture_charge(device, 0)
        }
        .assess_fit(&m4_pro, metal, &status, &reserves)
        .unwrap()
    };
    let full = assess(room);
    assert_eq!(full.verdict, AssessmentFitVerdict::Fits);
    assert_eq!(
        domain(&full, host_ram),
        DomainFit {
            role: DomainRole::Allocation,
            domain: host_ram,
            capacity_bytes: room + GIB + host.thresholds.planning_bytes,
            required_bytes: room + GIB,
            reserve_bytes: host.thresholds.planning_bytes,
            remaining_bytes: 0,
        }
    );
    assert_eq!(
        assess(room + 1).verdict,
        AssessmentFitVerdict::DoesNotFit {
            limiting: host_ram,
            deficit_bytes: NonZeroU64::MIN,
        }
    );
}

/// A stored table's storage (upload storage, behind `Tensor::shared`) is
/// charged where its executor's upload placement lands: on the RTX 3090,
/// CUDA's pinned host RAM, where a load whose host RAM is full is
/// refused though the 3090's memory holds the rest, and Vulkan's device
/// memory (ReBAR), where the same load fits.
#[test]
fn stored_tables_are_charged_in_the_domain_of_shared_storage() {
    let reserves = crate::platform::MemoryReserves::standard();
    let topology = recorded::rtx3090();
    let status = unopened(&topology, seismic::ReportedBy::LinuxMemAvailable);
    let (gpu, host_ram) = (device_memory(&topology), topology.host_ram().id());
    let fit = Fit::new(&topology, &status, &reserves, HostRoom::Allocatable);
    let (own, host) = (fit.store(gpu).unwrap(), fit.store(host_ram).unwrap());
    let cuda = executor(&topology, BackendName::Cuda);
    let vulkan = executor(&topology, BackendName::Vulkan);
    let charge = |staging, executor| {
        AssessmentMemoryCharge {
            weight_demand: ClaimDemand { upload: GIB, ..ClaimDemand::default() },
            ..fixture_charge(own.stable_bytes, staging)
        }
        .assess_fit(&topology, executor, &status, &reserves)
        .unwrap()
    };
    let pinned = charge(0, cuda);
    assert_eq!(pinned.verdict, AssessmentFitVerdict::Fits);
    assert_eq!(domain(&pinned, gpu).required_bytes, own.stable_bytes - GIB);
    // What remains of CUDA's room there: its usable memory less the
    // reserve, below the store's own fit.
    let room = fit.placement(cuda, seismic::Contract::DeviceOnly).unwrap().stable_bytes;
    assert_eq!(
        domain(&pinned, gpu).remaining_bytes,
        i64::try_from(room - (own.stable_bytes - GIB)).unwrap()
    );
    assert_eq!(domain(&pinned, host_ram).required_bytes, GIB);
    let local = charge(0, vulkan);
    assert_eq!(local.verdict, AssessmentFitVerdict::Fits);
    assert_eq!(domain(&local, gpu).required_bytes, own.stable_bytes);
    assert_eq!(domain(&local, host_ram).required_bytes, 0);
    // Staging takes host RAM's fit.
    let crowded = charge(host.stable_bytes, cuda);
    assert_eq!(
        crowded.verdict,
        AssessmentFitVerdict::DoesNotFit {
            limiting: host_ram,
            deficit_bytes: NonZeroU64::new(GIB).unwrap(),
        }
    );
    assert_eq!(
        domain(&crowded, gpu).remaining_bytes,
        domain(&pinned, gpu).remaining_bytes
    );
    assert_eq!(
        charge(host.stable_bytes, vulkan).verdict,
        AssessmentFitVerdict::Fits
    );
}

#[test]
fn reserved_host_weights_stay_in_host_ram_beside_rebar_upload() {
    let topology = recorded::rtx3090();
    let status = unopened(&topology, seismic::ReportedBy::LinuxMemAvailable);
    let reserves = crate::platform::MemoryReserves::standard();
    let vulkan = executor(&topology, BackendName::Vulkan);
    let charge = AssessmentMemoryCharge {
        weight_demand: ClaimDemand { upload: GIB, host: 2 * GIB, reserved: GIB,
            ..ClaimDemand::default() },
        ..fixture_charge(4 * GIB, 0)
    }.assess_fit(&topology, vulkan, &status, &reserves).unwrap();
    assert_eq!(charge.verdict, AssessmentFitVerdict::Fits);
    assert_eq!(domain(&charge, device_memory(&topology)).required_bytes, 2 * GIB);
    assert_eq!(domain(&charge, topology.host_ram().id()).required_bytes, 2 * GIB);
}
