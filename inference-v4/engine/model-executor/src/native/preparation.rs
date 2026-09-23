use super::*;

/// The phase-one catalog. Every handle is prepared before qualification and
/// retained for warm calls; this type has no API capable of preparing again.
#[derive(Debug)]
pub(super) struct NativePreparationCache {
    pub(super) owner: Tensor,
    pub(super) import: ImportKernels,
    pub(super) glue: GlueKernels,
    pub(super) target: TargetKernels,
    pub(super) head: Option<HeadKernels>,
    pub(super) vision: Option<VisionKernels>,
}

#[derive(Clone, Copy, Debug)]
enum PreparationBinding {
    TargetEmbedding(EmbeddingBinding),
    TargetAttention(AttentionBinding),
    TargetRecurrent(RecurrentBinding),
    TargetDense(DenseBinding),
    TargetRouted(RoutedBinding),
    Readout(ReadoutBinding),
    Features(FeaturesBinding),
    Head(crate::HeadBinding),
    VisionPatch(crate::VisionPatchBinding),
    VisionBlock(crate::VisionBlockBinding),
    VisionMerger(crate::VisionMergerBinding),
}

impl NativePreparationCache {
    pub(super) fn prepare_programs(
        device: &Device,
        plan: &ProgramPlan,
    ) -> Result<Self, CatalogError> {
        if device.backend() != BackendName::Metal {
            return Err(CatalogError::Backend {
                path: ExecutionPath::NativeMetal,
                backend: device.backend(),
                outcome: "phase one admits Metal only".into(),
            });
        }
        let mut import_handles = HashMap::new();
        let mut repack_handles = HashMap::new();
        for slot in plan.imports() {
            match *slot {
                ImportProgramSlot::Dense { source, resident } => {
                    if import_handles.contains_key(&(source, resident)) {
                        continue;
                    }
                    let bindings = dense_binding_name(source, resident);
                    import_handles.insert(
                        (source, resident),
                        import_dense::native_for_device_with(
                            device,
                            import_dense::Elements {
                                E: Element::dense(source),
                                U: Element::dense(resident),
                            },
                        )
                        .map_err(|error| preparation_dynamic("import_dense", &bindings, error))?,
                    );
                }
                ImportProgramSlot::Repack { source, resident } => {
                    if repack_handles.contains_key(&(source, resident)) {
                        continue;
                    }
                    let bindings = element_binding_name(source, resident);
                    repack_handles.insert(
                        (source, resident),
                        repack_weight::native_for_device_with(
                            device,
                            repack_weight::Elements {
                                E: source,
                                U: resident,
                            },
                        )
                        .map_err(|error| preparation_dynamic("repack_weight", &bindings, error))?,
                    );
                }
            }
        }
        let owner = Tensor::zeros(device, Element::u32(), &[1]).map_err(|error| {
            CatalogError::Preparation {
                path: ExecutionPath::NativeMetal,
                entry: "catalog_owner",
                bindings: "A=u32".into(),
                outcome: error.to_string(),
            }
        })?;
        let import = ImportKernels {
            import_dense: import_handles,
            repack_weight: repack_handles,
        };
        let mut prepared = Self {
            owner,
            import,
            glue: GlueKernels {
                shape_rows: Some(
                    shape_rows::native_for_device(device)
                        .map_err(|error| preparation("shape_rows", "fixed", error))?,
                ),
                sample_rows: Some(
                    sample_rows::native_for_device(device)
                        .map_err(|error| preparation("sample_rows", "fixed", error))?,
                ),
                conditioning_overlay: Some(
                    qwen_conditioning_overlay::native_for_device(device).map_err(|error| {
                        preparation("qwen_conditioning_overlay", "fixed", error)
                    })?,
                ),
                gather_rows: Some(
                    gather_rows::native_for_device(device)
                        .map_err(|error| preparation("gather_rows", "fixed", error))?,
                ),
                copy_rows_f32: prepare_optional_copy(
                    device,
                    plan.state().copies(),
                    Element::f32(),
                    "A=f32",
                )?,
                copy_rows_f16: prepare_optional_copy(
                    device,
                    plan.state().copies(),
                    Element::f16(),
                    "A=f16",
                )?,
                copy_rows_bf16: prepare_optional_copy(
                    device,
                    plan.state().copies(),
                    Element::bf16(),
                    "A=bf16",
                )?,
                copy_rows_u32: prepare_optional_copy(
                    device,
                    plan.state().copies(),
                    Element::u32(),
                    "A=u32",
                )?,
            },
            target: TargetKernels::default(),
            head: plan.head().map(|_| HeadKernels::default()),
            vision: plan.vision().map(|_| VisionKernels::default()),
        };
        prepared.prepare_binding(
            device,
            PreparationBinding::TargetEmbedding(plan.target().embedding()),
        )?;
        for block in plan.target().blocks() {
            prepared.prepare_binding(
                device,
                match block.mixer() {
                    MixerProgramSlot::Attention(binding) => {
                        PreparationBinding::TargetAttention(binding)
                    }
                    MixerProgramSlot::Recurrent(binding) => {
                        PreparationBinding::TargetRecurrent(binding)
                    }
                },
            )?;
            prepared.prepare_binding(
                device,
                match block.feed_forward() {
                    FeedForwardProgramSlot::Dense(binding) => {
                        PreparationBinding::TargetDense(binding)
                    }
                    FeedForwardProgramSlot::Routed(binding) => {
                        PreparationBinding::TargetRouted(binding)
                    }
                },
            )?;
        }
        prepared.prepare_binding(device, PreparationBinding::Readout(plan.target().readout()))?;
        if let Some(binding) = plan.target().features() {
            prepared.prepare_binding(device, PreparationBinding::Features(binding))?;
        }
        if let Some(head) = plan.head() {
            for &binding in head.blocks() {
                prepared.prepare_binding(device, PreparationBinding::Head(binding))?;
            }
        }
        if let Some(vision) = plan.vision() {
            prepared.prepare_binding(device, PreparationBinding::VisionPatch(vision.patch()))?;
            for &binding in vision.blocks() {
                prepared.prepare_binding(device, PreparationBinding::VisionBlock(binding))?;
            }
            prepared.prepare_binding(device, PreparationBinding::VisionMerger(vision.merger()))?;
        }
        Ok(prepared)
    }

    fn prepare_binding(
        &mut self,
        device: &Device,
        key: PreparationBinding,
    ) -> Result<(), CatalogError> {
        let prepared = self;
        match key {
            PreparationBinding::TargetEmbedding(b) => {
                if !prepared.target.embedding.contains_key(&b) {
                    let k = qwen_embedding_rows::native_for_device_with(
                        device,
                        qwen_embedding_rows::Elements {
                            EW: b.table,
                            A: b.activation,
                        },
                    )
                    .map_err(|e| target_preparation("qwen_embedding_rows", b, e))?;
                    prepared.target.embedding.insert(b, k);
                }
            }
            PreparationBinding::TargetAttention(b) => {
                if !prepared.target.attention.contains_key(&b) {
                    let kernels = AttentionKernels {
                        normalize: qwen_attention_normalize::native_for_device_with(
                            device,
                            qwen_attention_normalize::Elements {
                                NW: b.norm,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_attention_normalize", b, e))?,
                        project: qwen_attention_project::native_for_device_with(
                            device,
                            qwen_attention_project::Elements {
                                QW: b.query_gate,
                                KW: b.key,
                                VW: b.value,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_attention_project", b, e))?,
                        prepare: qwen_attention_prepare::native_for_device_with(
                            device,
                            qwen_attention_prepare::Elements { A: b.activation },
                        )
                        .map_err(|e| target_preparation("qwen_attention_prepare", b, e))?,
                        attend: qwen_attention_attend::native_for_device_with(
                            device,
                            qwen_attention_attend::Elements { A: b.activation },
                        )
                        .map_err(|e| target_preparation("qwen_attention_attend", b, e))?,
                        output: qwen_attention_output::native_for_device_with(
                            device,
                            qwen_attention_output::Elements {
                                OW: b.output,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_attention_output", b, e))?,
                    };
                    prepared.target.attention.insert(b, kernels);
                }
            }
            PreparationBinding::TargetRecurrent(b) => {
                check_native_recurrent_scan_width(b)?;
                if !prepared.target.recurrent.contains_key(&b) {
                    let k = RecurrentKernels {
                        normalize: qwen_recurrent_normalize::native_for_device_with(
                            device,
                            qwen_recurrent_normalize::Elements {
                                NW: b.norm,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_recurrent_normalize", b, e))?,
                        project: qwen_recurrent_project::native_for_device_with(
                            device,
                            qwen_recurrent_project::Elements {
                                QW: b.qkv,
                                GW: b.gate,
                                AW: b.alpha,
                                BW: b.beta,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_recurrent_project", b, e))?,
                        prepare: qwen_recurrent_prepare::native_for_device_with(
                            device,
                            qwen_recurrent_prepare::Elements {
                                A: b.activation,
                                RN: b.recurrent_norm,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_recurrent_prepare", b, e))?,
                        scan: qwen_recurrent_scan::native_for_device_with(
                            device,
                            qwen_recurrent_scan::Elements { A: b.activation },
                        )
                        .map_err(|e| target_preparation("qwen_recurrent_scan", b, e))?,
                        mix: qwen_recurrent_mix::native_for_device_with(
                            device,
                            qwen_recurrent_mix::Elements {
                                RN: b.recurrent_norm,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_recurrent_mix", b, e))?,
                        output: qwen_recurrent_output::native_for_device_with(
                            device,
                            qwen_recurrent_output::Elements {
                                OW: b.output,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_recurrent_output", b, e))?,
                    };
                    prepared.target.recurrent.insert(b, k);
                }
            }
            PreparationBinding::TargetDense(b) => {
                if !prepared.target.dense.contains_key(&b) {
                    let kernels = DenseKernels {
                        expand: qwen_dense_expand::native_for_device_with(
                            device,
                            qwen_dense_expand::Elements {
                                NW: b.norm,
                                GW: b.gate,
                                UW: b.up,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_dense_expand", b, e))?,
                        output: qwen_dense_output::native_for_device_with(
                            device,
                            qwen_dense_output::Elements {
                                DW: b.down,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_dense_output", b, e))?,
                        expand_demanded: qwen_dense_expand_demanded::native_for_device_with(
                            device,
                            qwen_dense_expand_demanded::Elements {
                                NW: b.norm,
                                GW: b.gate,
                                UW: b.up,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_dense_expand_demanded", b, e))?,
                        output_demanded: qwen_dense_output_demanded::native_for_device_with(
                            device,
                            qwen_dense_output_demanded::Elements {
                                DW: b.down,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_dense_output_demanded", b, e))?,
                    };
                    prepared.target.dense.insert(b, kernels);
                }
            }
            PreparationBinding::TargetRouted(b) => {
                if !prepared.target.routed.contains_key(&b) {
                    let kernels = RoutedKernels {
                        normalize: qwen_routed_normalize::native_for_device_with(
                            device,
                            qwen_routed_normalize::Elements {
                                NW: b.norm,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_routed_normalize", b, e))?,
                        logits: qwen_routed_logits::native_for_device_with(
                            device,
                            qwen_routed_logits::Elements {
                                RW: b.router,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_routed_logits", b, e))?,
                        select: qwen_routed_select::native_for_device(device)
                            .map_err(|e| target_preparation("qwen_routed_select", b, e))?,
                        expand: qwen_routed_expand::native_for_device_with(
                            device,
                            qwen_routed_expand::Elements {
                                EGW: b.expert_gate,
                                EUW: b.expert_up,
                                SGW: b.shared_gate,
                                SUW: b.shared_up,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_routed_expand", b, e))?,
                        output: qwen_routed_output::native_for_device_with(
                            device,
                            qwen_routed_output::Elements {
                                EDW: b.expert_down,
                                SDW: b.shared_down,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_routed_output", b, e))?,
                    };
                    prepared.target.routed.insert(b, kernels);
                }
            }
            PreparationBinding::Readout(b) => {
                if !prepared.target.readout.contains_key(&b) {
                    let kernels = ReadoutKernels {
                        features: qwen_features_rows::native_for_device_with(
                            device,
                            qwen_features_rows::Elements {
                                NW: b.norm,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("qwen_features_rows", b, e))?,
                        logits: head_logits_rows::native_for_device_with(
                            device,
                            head_logits_rows::Elements {
                                OW: b.weight,
                                A: b.activation,
                            },
                        )
                        .map_err(|e| target_preparation("head_logits_rows", b, e))?,
                    };
                    prepared.target.readout.insert(b, kernels);
                    let selected = qwen_selected_rows::native_for_device_with(
                        device,
                        qwen_selected_rows::Elements {
                            NW: b.norm,
                            OW: b.weight,
                            A: b.activation,
                        },
                    )
                    .map_err(|e| target_preparation("qwen_selected_rows", b, e))?;
                    prepared.target.selected.insert(b, selected);
                }
            }
            PreparationBinding::Features(b) => {
                if !prepared.target.features.contains_key(&b) {
                    let k = qwen_features_rows::native_for_device_with(
                        device,
                        qwen_features_rows::Elements {
                            NW: b.norm,
                            A: b.activation,
                        },
                    )
                    .map_err(|e| target_preparation("qwen_features_rows", b, e))?;
                    prepared.target.features.insert(b, k);
                }
            }
            PreparationBinding::Head(b) => {
                let head = prepared
                    .head
                    .as_mut()
                    .expect("head requirement creates head group");
                if head.input.contains_key(&b) {
                    return Ok(());
                }
                head.input.insert(
                    b,
                    qwen_head_rows::native_for_device_with(
                        device,
                        qwen_head_rows::Elements {
                            EW: b.embedding_table,
                            A: b.activation,
                            EN: b.embedding_norm,
                            HN: b.hidden_norm,
                            CW: b.combine,
                        },
                    )
                    .map_err(|error| target_preparation("qwen_head_rows", b, error))?,
                );
                head.attention.insert(
                    b,
                    AttentionKernels {
                        normalize: qwen_attention_normalize::native_for_device_with(
                            device,
                            qwen_attention_normalize::Elements {
                                NW: b.input_norm,
                                A: b.activation,
                            },
                        )
                        .map_err(|error| {
                            target_preparation("qwen_attention_normalize", b, error)
                        })?,
                        project: qwen_attention_project::native_for_device_with(
                            device,
                            qwen_attention_project::Elements {
                                QW: b.query_gate,
                                KW: b.key,
                                VW: b.value,
                                A: b.activation,
                            },
                        )
                        .map_err(|error| target_preparation("qwen_attention_project", b, error))?,
                        prepare: qwen_attention_prepare::native_for_device_with(
                            device,
                            qwen_attention_prepare::Elements { A: b.activation },
                        )
                        .map_err(|error| target_preparation("qwen_attention_prepare", b, error))?,
                        attend: qwen_attention_attend::native_for_device_with(
                            device,
                            qwen_attention_attend::Elements { A: b.activation },
                        )
                        .map_err(|error| target_preparation("qwen_attention_attend", b, error))?,
                        output: qwen_attention_output::native_for_device_with(
                            device,
                            qwen_attention_output::Elements {
                                OW: b.attention_output,
                                A: b.activation,
                            },
                        )
                        .map_err(|error| target_preparation("qwen_attention_output", b, error))?,
                    },
                );
                head.dense.insert(
                    b,
                    DenseKernels {
                        expand: qwen_dense_expand::native_for_device_with(
                            device,
                            qwen_dense_expand::Elements {
                                NW: b.feedforward_norm,
                                GW: b.gate,
                                UW: b.up,
                                A: b.activation,
                            },
                        )
                        .map_err(|error| target_preparation("qwen_dense_expand", b, error))?,
                        output: qwen_dense_output::native_for_device_with(
                            device,
                            qwen_dense_output::Elements {
                                DW: b.down,
                                A: b.activation,
                            },
                        )
                        .map_err(|error| target_preparation("qwen_dense_output", b, error))?,
                        expand_demanded: qwen_dense_expand_demanded::native_for_device_with(
                            device,
                            qwen_dense_expand_demanded::Elements {
                                NW: b.feedforward_norm,
                                GW: b.gate,
                                UW: b.up,
                                A: b.activation,
                            },
                        )
                        .map_err(|error| {
                            target_preparation("qwen_dense_expand_demanded", b, error)
                        })?,
                        output_demanded: qwen_dense_output_demanded::native_for_device_with(
                            device,
                            qwen_dense_output_demanded::Elements {
                                DW: b.down,
                                A: b.activation,
                            },
                        )
                        .map_err(|error| {
                            target_preparation("qwen_dense_output_demanded", b, error)
                        })?,
                    },
                );
                head.features.insert(
                    b,
                    qwen_features_rows::native_for_device_with(
                        device,
                        qwen_features_rows::Elements {
                            NW: b.output_norm,
                            A: b.activation,
                        },
                    )
                    .map_err(|error| target_preparation("qwen_features_rows", b, error))?,
                );
                head.logits.insert(
                    b,
                    head_logits_rows::native_for_device_with(
                        device,
                        head_logits_rows::Elements {
                            A: b.activation,
                            OW: b.projection,
                        },
                    )
                    .map_err(|error| target_preparation("head_logits_rows", b, error))?,
                );
            }
            PreparationBinding::VisionPatch(b) => {
                let vision = prepared
                    .vision
                    .as_mut()
                    .expect("vision requirement creates vision group");
                vision.stem.insert(
                    b,
                    qwen_vision_stem::native_for_device_with(
                        device,
                        qwen_vision_stem::Elements {
                            W0: b.temporal_weight_0,
                            W1: b.temporal_weight_1,
                            B: b.bias,
                            PE: b.position,
                            A: b.activation,
                        },
                    )
                    .map_err(|error| target_preparation("qwen_vision_stem", b, error))?,
                );
            }
            PreparationBinding::VisionBlock(b) => {
                let vision = prepared
                    .vision
                    .as_mut()
                    .expect("vision requirement creates vision group");
                if vision.blocks.contains_key(&b) {
                    return Ok(());
                }
                vision.blocks.insert(
                    b,
                    qwen_vision_block::native_for_device_with(
                        device,
                        qwen_vision_block::Elements {
                            A: b.activation,
                            N1W: b.input_norm_weight,
                            N1B: b.input_norm_bias,
                            QW: b.qkv_weight,
                            QB: b.qkv_bias,
                            PW: b.attention_output,
                            PB: b.attention_output_bias,
                            N2W: b.feedforward_norm_weight,
                            N2B: b.feedforward_norm_bias,
                            UW: b.up,
                            UB: b.up_bias,
                            DW: b.down,
                            DB: b.down_bias,
                        },
                    )
                    .map_err(|error| target_preparation("qwen_vision_block", b, error))?,
                );
            }
            PreparationBinding::VisionMerger(b) => {
                let vision = prepared
                    .vision
                    .as_mut()
                    .expect("vision requirement creates vision group");
                vision.merger.insert(
                    b,
                    qwen_vision_merger::native_for_device_with(
                        device,
                        qwen_vision_merger::Elements {
                            A: b.activation,
                            NW: b.output_norm_weight,
                            NB: b.output_norm_bias,
                            UW: b.hidden,
                            UB: b.hidden_bias,
                            DW: b.output,
                            DB: b.output_bias,
                        },
                    )
                    .map_err(|error| target_preparation("qwen_vision_merger", b, error))?,
                );
                vision.output.insert(
                    b,
                    qwen_vision_feature_output::native_for_device_with(
                        device,
                        qwen_vision_feature_output::Elements { A: b.activation },
                    )
                    .map_err(|error| target_preparation("qwen_vision_feature_output", b, error))?,
                );
            }
            _ => {}
        }
        Ok(())
    }
}

fn prepare_copy_rows(
    device: &Device,
    element: Element,
    bindings: &'static str,
) -> Result<NativeKernel<copy_rows::Entry>, CatalogError> {
    copy_rows::native_for_device_with(device, copy_rows::Elements { A: element })
        .map_err(|error| preparation("copy_rows", bindings, error))
}

fn prepare_optional_copy(
    device: &Device,
    copies: &[Element],
    element: Element,
    bindings: &'static str,
) -> Result<Option<NativeKernel<copy_rows::Entry>>, CatalogError> {
    copies
        .contains(&element)
        .then(|| prepare_copy_rows(device, element, bindings))
        .transpose()
}

fn preparation(
    entry: &'static str,
    bindings: &'static str,
    outcome: impl fmt::Display,
) -> CatalogError {
    CatalogError::Preparation {
        path: ExecutionPath::NativeMetal,
        entry,
        bindings: bindings.into(),
        outcome: outcome.to_string(),
    }
}

fn preparation_dynamic(
    entry: &'static str,
    bindings: &str,
    outcome: impl fmt::Display,
) -> CatalogError {
    CatalogError::Preparation {
        path: ExecutionPath::NativeMetal,
        entry,
        bindings: bindings.to_owned(),
        outcome: outcome.to_string(),
    }
}

fn target_preparation(
    entry: &'static str,
    binding: impl fmt::Debug,
    outcome: impl fmt::Display,
) -> CatalogError {
    CatalogError::Preparation {
        path: ExecutionPath::NativeMetal,
        entry,
        bindings: "model binding".into(),
        outcome: format!("{binding:?}: {outcome}"),
    }
}

// The direct Metal scan stores one W-vector per thread. Program planning may
// describe wider models; only this native kernel has the finite width bound.
const NATIVE_RECURRENT_SCAN_WIDTH_CAPACITY: u64 = 256;

pub(super) fn check_native_recurrent_scan_width(
    binding: RecurrentBinding,
) -> Result<(), CatalogError> {
    if binding.width > NATIVE_RECURRENT_SCAN_WIDTH_CAPACITY {
        return Err(target_preparation(
            "qwen_recurrent_scan",
            binding,
            "native recurrent scan supports width at most 256",
        ));
    }
    Ok(())
}
