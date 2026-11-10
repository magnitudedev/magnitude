use super::{
    GroupExtents,
    block::{BlockGraphInputs, checked_block_graph_draft, checked_block_graph_resources},
    bound::{TokenRow, TokenRows},
    certify::certify_target_family,
    classes::{
        BlockOutputScales, FamilyOutline, LaunchControl, LaunchControlLayout, Passes, StagedClass,
        UnreadScales, block_classes, block_listings, block_routed, block_served, block_stages,
        block_upload_regions, class_topology, enumerate_block_graphs, once_class, once_graphs,
        slot_scores,
    },
    entry::{
        EntryTokens, EntryWidth, TARGET_TABLE, embedding_transform, entry_graph_topology, keeps_token_rows,
        notifies_selection,
    },
    prepared::{block_graph_shapes, block_scopes, block_taps, chain_rows, trailing_run},
    weights::{CheckedLandingEntries, PlannedWeights, activation},
};
use crate::{
    ModelLoadPlan, ResourceLimits,
    operators::{
        BlockKeeps, paired_block,
        routed::split::{OnceStage, RunStage, Stage},
    },
    programs::native_constants::{CheckedGraphFamilyResources, FamilyConstants},
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, HostHeap, NativeGraphLayout, NativeGraphMetadata, NativeGraphStorageBytes,
};

#[cfg(test)]
mod token_rows_tests {
    use super::{TokenRow, TokenRows};

    #[test]
    fn a_token_row_is_held_from_the_step_that_writes_it_until_overwritten() {
        assert!(matches!(
            TokenRows::new(8, 0).find(3).unwrap(),
            TokenRow::None
        ));
        assert!(TokenRows::new(8, 2).find(8).is_err());

        let rows = TokenRows::new(8, 2);
        let TokenRow::Free(first) = rows.find(3).unwrap() else {
            panic!("a token no step wrote has a free slot")
        };
        // The slot is the token's only once the writing step is queued.
        assert!(matches!(rows.find(3).unwrap(), TokenRow::Free(_)));
        rows.hold(3, first);
        assert!(matches!(rows.find(3).unwrap(), TokenRow::Held(slot) if slot == first));
        let TokenRow::Free(second) = rows.find(4).unwrap() else {
            panic!("the other slot is free")
        };
        assert_ne!(second, first);
        rows.hold(4, second);
        assert!(matches!(rows.find(4).unwrap(), TokenRow::Held(slot) if slot == second));
        // A third token takes a slot from one of the two.
        let TokenRow::Free(third) = rows.find(5).unwrap() else {
            panic!("a full directory frees a slot")
        };
        rows.hold(5, third);
        assert!(matches!(rows.find(5).unwrap(), TokenRow::Held(slot) if slot == third));
        rows.overwrite(0..2);
        assert!(matches!(rows.find(5).unwrap(), TokenRow::Free(_)));
    }
}

#[cfg(test)]
mod upload_region_tests {
    use super::{RunStage, block_upload_regions};

    /// A decode class runs the block whole; a prompt class its front, which
    /// writes the mixer's controls, and a back that writes a word. Each
    /// region holds the most any class's run in its place writes, so every
    /// class's runs fit regions of their own.
    #[test]
    fn a_block_takes_a_region_per_run_holding_the_most_any_class_writes() {
        let stages = |rows: u64| -> &'static [RunStage] {
            match rows {
                1 => &[RunStage::Whole],
                _ => &[RunStage::Front, RunStage::Combine],
            }
        };
        let upload = |rows: u64, stage: RunStage| match stage {
            RunStage::Whole | RunStage::Resident => 40,
            RunStage::Front => 100 * rows,
            RunStage::Combine => 4,
        };
        assert_eq!(block_upload_regions([1, 16, 512], stages, upload), vec![51_200, 4]);
        // A stage that writes nothing takes no region.
        assert_eq!(block_upload_regions([1], stages, |_, _| 0), Vec::<u64>::new());
    }

    /// A launch uploads each control once, in order; a read whose listing
    /// blocks are sealed for the most segments alone has its spans padded
    /// only while a prefill launch has fewer, and the region holds the
    /// largest segment class's upload, each control on a buffer boundary.
    #[test]
    fn launch_controls_pad_a_read_only_below_the_most_segments() {
        use super::{LaunchControl, LaunchControlLayout};
        let reads = magnitude_state::HistoryReads::of(&[], 1, 1).unwrap();
        let layout = LaunchControlLayout {
            padded: vec![false, true],
            picked: vec![None, None],
            max_segments: 4,
            region_bytes: 0,
        };
        let controls = layout.controls(16, 2, 1);
        let at = |control| {
            controls
                .iter()
                .find(|(of, _)| *of == control)
                .map(|(_, extents)| extents.clone())
        };
        assert_eq!(at(LaunchControl::Coordinates), Some(vec![16, 4]));
        assert_eq!(at(LaunchControl::Visible { read: 0 }), Some(vec![16, 2, 2]));
        assert_eq!(at(LaunchControl::Padded { read: 0 }), None);
        assert_eq!(at(LaunchControl::Padded { read: 1 }), Some(vec![16, 4, 2]));
        assert_eq!(at(LaunchControl::Segments), Some(vec![2, 2]));
        // Each control once: coordinates, two reads' three and one padded,
        // five slot controls.
        assert_eq!(controls.len(), 1 + 3 * 2 + 1 + 5);
        for (control, _) in &controls {
            assert_eq!(controls.iter().filter(|(of, _)| of == control).count(), 1);
        }
        // At the most segments, or in a decode launch, nothing pads.
        for (rows, segments) in [(16, 4), (8, 2)] {
            assert!(
                layout
                    .controls(rows, segments, 1)
                    .iter()
                    .all(|(of, _)| !matches!(of, LaunchControl::Padded { .. }))
            );
        }
        // Without history reads: coordinates and the slot controls, each
        // at a 256-byte boundary; the last, one slot's word.
        let plain = LaunchControlLayout::of(&reads, &[], &[], 16, 4, 1).unwrap();
        assert_eq!(plain.region_bytes(), 5 * 256 + 4);
    }

    /// A picked Block read takes its rows' complete units and the picks
    /// table; one whose own indexer produces a candidate pool also the pool
    /// table, of the pool's blocks and the launch's segments a row.
    #[test]
    fn launch_controls_hold_a_pool_table_only_for_a_producing_read() {
        use crate::programs::native_target_graph::classes::PickedRead;
        use super::{LaunchControl, LaunchControlLayout};
        let layout = LaunchControlLayout {
            padded: vec![false, false],
            picked: vec![
                Some(PickedRead { top: 2, pool_blocks: Some(8) }),
                Some(PickedRead { top: 3, pool_blocks: None }),
            ],
            max_segments: 4,
            region_bytes: 0,
        };
        let controls = layout.controls(16, 2, 1);
        let at = |control| {
            controls
                .iter()
                .find(|(of, _)| *of == control)
                .map(|(_, extents)| extents.clone())
        };
        assert_eq!(at(LaunchControl::Complete { read: 0 }), Some(vec![16]));
        assert_eq!(at(LaunchControl::Picks { read: 0 }), Some(vec![16, 2]));
        assert_eq!(at(LaunchControl::Pool { read: 0 }), Some(vec![16, 8 + 2, 2]));
        assert_eq!(at(LaunchControl::Picks { read: 1 }), Some(vec![16, 3]));
        assert_eq!(at(LaunchControl::Pool { read: 1 }), None);
        assert!(LaunchControl::Pool { read: 0 }.per_row());
    }
}

#[cfg(test)]
mod stateless_tail_tests {
    use super::trailing_run;

    #[test]
    fn the_tail_is_the_trailing_run_of_stateless_blocks() {
        // Gemma 4 E2B: 35 blocks, the last 20 read shared history.
        assert_eq!(trailing_run(35, |index| index >= 15), 15);
        // A stateful block inside the run ends it.
        assert_eq!(trailing_run(35, |index| index >= 15 && index != 30), 31);
        // A stateful last block leaves no tail; nor do stateless leading blocks.
        assert_eq!(trailing_run(35, |index| index < 34), 35);
        assert_eq!(trailing_run(4, |_| false), 4);
        assert_eq!(trailing_run(4, |_| true), 0);
    }
}

#[cfg(test)]
mod shared_graph_tests {
    use super::*;
    use super::super::bound::block_layer;
    use magnitude_family_contracts::WeightScope;

    /// Streamed layers at different homes of one group share a sealed
    /// graph when their blocks are of one shape, as the attention blocks of
    /// `qwen4exp_fixture_definition_of(4, 32)` are: so that graph reads each
    /// layer's home through its own ports (`policy[POLICY_HOME]`), which
    /// `ngram_steps_tests::a_whole_banks_shared_graphs_read_each_layers_own_experts`
    /// runs on the device.
    #[test]
    fn streamed_layers_at_different_homes_share_a_graph() {
        let (definition, load) = streamed_fixture_load();
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 64,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
        };
        let codec = magnitude_state::KvCodec::Dense;
        let state = crate::ResourcePlanner::state_plan(
            &definition,
            &load,
            crate::PlannedMethod::Plain,
            codec,
            limits,
            crate::ResourceCapacity { domain_bytes: 1 << 30, tensor_operations: crate::TensorOperations::Absent },
        )
        .unwrap();
        let plan = load.program_plan(&definition, codec).unwrap();
        let (_, tapped) = block_taps(plan.target(), &definition.decoder).unwrap();
        let shapes = block_graph_shapes(&load, &definition.decoder, &state, &tapped, None).unwrap();
        let bank = load.expert_bank().unwrap();
        let home = |index: usize| block_layer(bank, index).ok().map(|layer| bank.home(layer));
        assert_eq!(shapes[1], shapes[2], "the first two attention blocks share a graph");
        assert_ne!(home(1), home(2), "at different homes of one group");
        assert!(home(1).is_some());
    }

    /// Two Full layers of one form (both pools softmax-gated) share one block
    /// plan, which therefore names neither one's pooled source: each block
    /// binds its own at run time (`PickedSource::pooled`). A mean pool beside
    /// a gated one is another form and another plan. These are
    /// `compressed_layers_decode_equals_prefill`'s shared-plan case and its
    /// control.
    #[test]
    fn two_full_layers_of_one_form_share_one_plan() {
        use crate::operators::attention::tests::selected::{CompressedForm, compressed};
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 64,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
        };
        for (second_gated, shared) in [(true, true), (false, false)] {
            let (definition, manifest) = compressed(CompressedForm::TwoFull { second_gated });
            let selection = crate::ComponentSelection { head: false, vision: false };
            let layout = crate::resident_layout(crate::ExecutionPath::Native, BackendName::Cpu);
            let load = ModelLoadPlan::derive(&manifest, &definition, selection, layout, crate::PlannedMethod::Plain, limits)
                .unwrap();
            let codec = magnitude_state::KvCodec::Dense;
            let state = crate::ResourcePlanner::state_plan(
                &definition,
                &load,
                crate::PlannedMethod::Plain,
                codec,
                limits,
                crate::ResourceCapacity { domain_bytes: 1 << 30, tensor_operations: crate::TensorOperations::Absent },
            )
            .unwrap();
            let plan = load.program_plan(&definition, codec).unwrap();
            let (_, tapped) = block_taps(plan.target(), &definition.decoder).unwrap();
            let shapes = block_graph_shapes(&load, &definition.decoder, &state, &tapped, None).unwrap();
            assert_eq!(shapes[0] == shapes[1], shared, "second pool gated {second_gated}: the blocks share a plan");
        }
    }

    /// The Qwen4Exp fixture with four attention blocks of 32 experts, its
    /// experts streamed at the bank's floor on Metal.
    fn streamed_fixture_load() -> (magnitude_family_contracts::ModelDefinition, ModelLoadPlan) {
        use crate::planning::fixtures::{qwen4exp_fixture_definition_of, streamed_experts_manifest};
        let backend = BackendName::Metal;
        let definition = qwen4exp_fixture_definition_of(4, 32);
        let manifest = streamed_experts_manifest(&definition);
        let selection = crate::ComponentSelection { head: false, vision: false };
        let load = ModelLoadPlan::derive(&manifest, &definition, selection, crate::resident_layout(crate::ExecutionPath::Native, backend), crate::PlannedMethod::Plain, crate::planning::fixtures::LIMITS)
            .unwrap()
            .with_expert_bank_at_floor(&definition, crate::planning::BankSubstrate {
                backend,
                reservation: seismic::Reservation { granule: 2 << 20, host_addressed: true, chunks: None },
                host: crate::planning::HostTier::Unified(crate::planning::Parking::InKernels),
            })
            .unwrap();
        (definition, load)
    }

    /// A block's streamed layer is its feed-forward's, the sublayer's own or
    /// a parallel sublayer's routed branch (Gemma 4's experts): each block's
    /// bank calls (`BankRun::fill_followed`, `BankRun::block_sampled`) find
    /// it where the bank holds the branch, which the sublayer's own scope
    /// names no layer of.
    #[test]
    fn a_blocks_streamed_layer_is_its_feed_forwards_own_or_branch() {
        let (_, load) = streamed_fixture_load();
        let bank = load.expert_bank().unwrap();
        let own = (0..bank.layers().len()).map(|layer| (block_layer(bank, layer_block(bank, layer)).unwrap(), layer)).collect::<Vec<_>>();
        let layers = bank
            .layers()
            .iter()
            .cloned()
            .map(|mut layer| {
                let WeightScope::TargetSublayer(sublayer) = layer.scope else {
                    panic!("the fixture's streamed layers are sublayers")
                };
                layer.scope = WeightScope::TargetBranch { sublayer, branch: 1 };
                layer
            })
            .collect();
        let branched = bank.clone().with_layers(crate::planning::BankLayers::new(layers).unwrap());
        for (found, layer) in own {
            assert_eq!(found, layer);
            let block = layer_block(&branched, layer);
            assert!(branched.layer(block_scopes(block).unwrap()[1]).is_none(), "the sublayer names no branch layer");
            assert_eq!(block_layer(&branched, block), Ok(layer), "block {block}");
        }
    }

    fn layer_block(bank: &crate::ExpertBankPlan, layer: usize) -> usize {
        match bank.layers()[layer].scope {
            WeightScope::TargetSublayer(sublayer) | WeightScope::TargetBranch { sublayer, .. } => sublayer.block as usize,
            scope => panic!("{scope:?} is no block's"),
        }
    }
}

#[cfg(test)]
mod resource_template_tests {
    use super::*;
    use crate::{
        ComponentSelection, ExecutionPath, PlannedMethod, ResourceCapacity, ResourcePlanner,
        assessment::fixtures::{DeclaredFeedForward, QWEN35_CONFIGURATIONS, declared_model},
        resident_layout,
    };
    use magnitude_state::KvCodec;

    /// A load whose blocks tap for a separate draft holds, of each tapped
    /// block, the graphs a load without the drafter runs, for each class a
    /// launch none of whose rows demands features takes: certified into
    /// the drafter-free family's layouts, and of no other class.
    #[test]
    fn a_tapped_block_holds_its_drafter_free_graphs_of_untapped_launches() {
        use super::super::{certify::certify_target_and_undrafted, classes::Holder};
        use crate::planning::fixtures::{LIMITS, separate_draft_fixture};
        let (definition, manifest) = separate_draft_fixture(false);
        let geometry = &definition.decoder;
        let method = PlannedMethod::DFlash { proposals: 3, widths: crate::DraftWidths::Adaptive };
        for backend in [BackendName::Cpu, BackendName::Metal, BackendName::Cuda, BackendName::Vulkan] {
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection { head: true, vision: false },
                resident_layout(ExecutionPath::Native, backend),
                method,
                LIMITS,
            )
            .unwrap();
            let state = ResourcePlanner::state_plan(
                &definition,
                &load,
                method,
                KvCodec::Dense,
                LIMITS,
                ResourceCapacity { domain_bytes: 1 << 30, tensor_operations: crate::TensorOperations::Absent },
            )
            .unwrap();
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let [tapped, undrafted] = certify_target_and_undrafted(
                backend,
                &load,
                geometry,
                &state,
                plan.target(),
                LIMITS,
                crate::PassLaunches::whole_chain(LIMITS),
            )
            .unwrap();
            let outline = &tapped.outline;
            let taps = (0..geometry.blocks.len()).filter(|&index| outline.untapped(index).is_some()).collect::<Vec<_>>();
            assert!(!taps.is_empty(), "{backend:?}: the draft taps a block");
            for index in 0..geometry.blocks.len() {
                let untapped = &tapped.untapped[index];
                if !taps.contains(&index) {
                    assert!(untapped.is_empty(), "{backend:?} block {index}");
                    continue;
                }
                assert_eq!(outline.untapped(index), Some(undrafted.outline.shapes[index].clone()));
                // Each run topology the drafter-free family certifies, into
                // its very layout.
                let runs = |graphs: &std::collections::BTreeMap<super::super::classes::Topology, super::super::certify::CertifiedBlockGraph>| {
                    graphs
                        .iter()
                        .filter(|(&(.., stage, _), _)| matches!(stage, Stage::Run(_)))
                        .map(|(topology, graph)| (*topology, graph.layout.storage_bytes()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(runs(untapped), runs(&undrafted.blocks[index]), "{backend:?} block {index}");
            }
            // An untapped graph for every run class of a tapped block of at
            // most `untapped_rows`, sealed or shared, and none other.
            let enumeration = enumerate_block_graphs(
                outline,
                &tapped.served,
                tapped.passes,
                &state,
                plan.target(),
                BlockOutputScales::Planned(UnreadScales::Distinct),
            )
            .unwrap();
            let mut expected = enumeration
                .sealing
                .iter()
                .copied()
                .chain(enumeration.sharing.iter().map(|&(index, class, _)| (index, class)))
                .filter(|&(index, ((rows, ..), stage))| {
                    taps.contains(&index) && matches!(stage, Stage::Run(_)) && rows <= outline.untapped_rows
                })
                .collect::<Vec<_>>();
            expected.sort_unstable();
            let mut untapped = enumeration
                .untapped_sealing
                .iter()
                .copied()
                .chain(enumeration.untapped_sharing.iter().map(|&(index, class, holder)| {
                    match holder {
                        Holder::Block(block) => assert!(!taps.contains(&block)),
                        Holder::Untapped(first) => assert!(first < index && taps.contains(&first)),
                    }
                    (index, class)
                }))
                .collect::<Vec<_>>();
            untapped.sort_unstable();
            assert!(!untapped.is_empty(), "{backend:?}");
            assert_eq!(untapped, expected, "{backend:?}");
            // A decode launch runs a tapped block's either form, as its rows
            // demand features or not, so the family keeps both live for each
            // launch in flight: its pins hold more graphs than the
            // drafter-free family's, one each of the same launches.
            assert_eq!(outline.untapped_rows, LIMITS.decode_rows(), "{backend:?}");
            assert!(
                tapped.graphs.pinned.graphs > undrafted.graphs.pinned.graphs,
                "{backend:?}: {:?} pins no more than the drafter-free {:?}",
                tapped.graphs.pinned,
                undrafted.graphs.pinned
            );
        }
    }

    /// Identical sublayers may share an interior graph, but neither the
    /// widening entry nor the exit mix may disappear through that sharing.
    #[test]
    fn stream_boundaries_keep_their_own_block_graphs() {
        use crate::planning::fixtures::{fixture_manifest, hyper_connection_fixture_definition};
        let mut definition = hyper_connection_fixture_definition();
        let manifest = fixture_manifest(&definition);
        // Tied weights make the four sublayers identical in every respect.
        definition.decoder.blocks = vec![definition.decoder.blocks[0].clone(); 4];
        let backend = BackendName::Cuda;
        let codec = KvCodec::Dense;
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 64,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
        };
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            resident_layout(ExecutionPath::Native, backend),
            PlannedMethod::Plain,
            limits,
        )
        .unwrap();
        let state = ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::Plain,
            codec,
            limits,
            ResourceCapacity {
                domain_bytes: 1 << 30,
                tensor_operations: crate::TensorOperations::Absent,
            },
        )
        .unwrap();
        let geometry = &definition.decoder;
        let plan = load.program_plan(&definition, codec).unwrap();
        let (_, tapped) = block_taps(plan.target(), geometry).unwrap();
        let shapes = block_graph_shapes(&load, geometry, &state, &tapped, None).unwrap();
        assert_ne!(shapes[0], shapes[1], "stream widening cannot be shared");
        assert_eq!(shapes[1], shapes[2], "interior graphs still share");
        assert_ne!(shapes[2], shapes[3], "the exit mix cannot be omitted");
        let passes = Passes { region: 64, launches: 1 };
        let outline = FamilyOutline {
            row_classes: vec![1, 64],
            largest_rows: 64,
            ..FamilyOutline::of(backend, &load, geometry, &state, plan.target(), limits).unwrap()
        };
        let enumeration = enumerate_block_graphs(
            &outline,
            &block_served(&load, geometry, limits, 64, passes).unwrap(),
            passes,
            &state,
            plan.target(),
            BlockOutputScales::Planned(UnreadScales::Equal),
        )
        .unwrap();
        assert!(!enumeration.sharing.is_empty());
        assert!(
            enumeration
                .sharing
                .iter()
                .all(|&(index, _, first)| index == 2 && first == 1)
        );
        for boundary in [0, 3] {
            assert!(
                enumeration
                    .sealing
                    .iter()
                    .any(|&(index, _)| index == boundary)
            );
        }
    }

    /// The 512-row attention class workspaces of one model on Metal over
    /// K8/V4 history: the family's, each listing class's and that of the class
    /// that lists none, with the store's reservation of history rows.
    struct Charged {
        reservation: u64,
        family: u64,
        /// The largest workspace of the block's exact 512-row classes.
        largest: u64,
        listed: Vec<u64>,
        unlisted: u64,
    }

    fn charged(
        (kv_heads, head_width): (u64, u64),
        context: u64,
        domain_bytes: u64,
        tensor_operations: bool,
    ) -> Charged {
        let limits = ResourceLimits {
            max_launch_rows: 512,
            max_launch_slots: 512,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
        };
        let mut configuration = QWEN35_CONFIGURATIONS[0];
        configuration.blocks = configuration.attention_interval;
        configuration.kv_heads = kv_heads;
        configuration.head_width = head_width;
        let (mut definition, manifest) = declared_model(&configuration);
        definition.decoder.context_limit = context;
        let backend = BackendName::Metal;
        let codec = KvCodec::AffineK8V4;
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            resident_layout(ExecutionPath::Native, backend),
            PlannedMethod::Plain,
            limits,
        )
        .unwrap();
        let plan = load.program_plan(&definition, codec).unwrap();
        let index = plan
            .target()
            .blocks()
            .iter()
            .position(|slot| matches!(slot.mixer(), crate::MixerProgramSlot::Attention(_)))
            .unwrap();
        let state = ResourcePlanner::state_plan(
            &definition,
            &load,
            PlannedMethod::Plain,
            codec,
            limits,
            ResourceCapacity {
                domain_bytes,
                tensor_operations: crate::TensorOperations::of(tensor_operations),
            },
        )
        .unwrap();
        let certificate = certify_target_family(
            backend,
            &load,
            &definition.decoder,
            &state,
            plan.target(),
            limits,
            crate::PassLaunches::whole_chain(limits),
        )
        .unwrap();
        let family = certificate.resources.storage.workspace;
        // The load's assessment charges the family's workspace, and every
        // exact 512-row class of the block seals into the layout the family
        // certified for it and is charged that layout.
        let assessed = crate::AssessmentGraphResourceBounds::derive(
            &definition,
            &load,
            &state,
            PlannedMethod::Plain,
            codec,
            limits,
            backend,
        )
        .unwrap();
        assert_eq!(assessed.target.charge.workspace_bytes, family);
        let max_segments = state.target_state().span_limit().next_power_of_two() as u64;
        let listings =
            block_listings(&state, backend, plan.target().blocks()[index], index, 512).unwrap();
        // Tensor operations read every head's history decoded; without
        // them the co-issue form reads 128-, 256- and 512-column heads'.
        assert_eq!(
            listings.tiles.is_empty(),
            !(tensor_operations || matches!(head_width, 128 | 256 | 512))
        );
        let mut most = 0;
        for class in block_classes(BlockKeeps::History, 512, 512, max_segments, &listings, false)
        {
            let (graph, _, _) = checked_block_graph_draft(
                NativeGraphMetadata::new(backend),
                plan.target().blocks()[index],
                None,
                BlockGraphInputs::of_class(
                    &load,
                    &definition.decoder,
                    &state,
                    index,
                    (class, Stage::Run(RunStage::Whole)),
                    class.0,
                ),
            )
            .unwrap();
            let layout = &certificate.blocks[index][&class_topology((class, Stage::Run(RunStage::Whole)))].layout;
            assert_eq!(
                graph.seal_with_layout(layout).unwrap(),
                layout.storage_bytes()
            );
            assert!(layout.storage_bytes().workspace <= family);
            most = most.max(layout.storage_bytes().workspace);
        }
        let class = |listed| {
            checked_block_graph_resources(
                backend,
                &load,
                &definition.decoder,
                &state,
                plan.target().blocks()[index],
                index,
                (512, 1, 1, listed, 0),
            )
            .unwrap()
            .0
            .workspace
        };
        // One request's worth of tiles is the largest listing: its context
        // and the rows appended within one advance (a causal reader of
        // compressed history reads them through their reserved
        // destinations), each in at most two partial pages.
        let history = state
            .target_state()
            .layer_history(LayerRef::Target(index as u32))
            .unwrap()
            .store;
        if let Some(&most) = listings.tiles.last() {
            assert!(
                most * 256
                    <= context + limits.max_launch_rows as u64 + 4 * u64::from(history.page_rows)
            );
        }
        Charged {
            reservation: history.rows as u64,
            family,
            largest: most,
            listed: listings.tiles.iter().map(|&tiles| class(tiles)).collect(),
            unlisted: class(0),
        }
    }

    /// An attention block's graph scratch follows neither the store's
    /// reservation of history rows (an address-space ceiling from the
    /// device's bytes) nor the context limit: on Metal over K8/V4 history
    /// every 512-row class that lists tiles is charged at least what the
    /// class that lists none is, each over the configurations it admits, and
    /// the family, its largest class's, holds the same bytes with and
    /// without tensor operations.
    #[test]
    fn attention_graph_scratch_follows_neither_the_reservation_nor_the_context() {
        for heads in [(4, 256), (1, 128), (2, 128), (1, 256), (2, 512), (4, 512)] {
            for context in [65_536, 262_144] {
                let plain = charged(heads, context, 8 << 30, false);
                let small = charged(heads, context, 8 << 30, true);
                let large = charged(heads, context, 256 << 30, true);
                assert!(large.reservation > small.reservation);
                assert!(!small.listed.is_empty());
                eprintln!(
                    "kv heads and width {heads:?}, context {context}, reservation {} / {} rows: family {} bytes, largest class {} bytes, 512-row attention class {} bytes listing none and {:?} listing",
                    small.reservation,
                    large.reservation,
                    small.family,
                    small.largest,
                    small.unlisted,
                    small.listed
                );
                assert_eq!(small.family, small.largest, "{heads:?} at {context}: the family's workspace is its largest class's");
                for &listed in small
                    .listed
                    .iter()
                    .chain(&large.listed)
                    .chain(&plain.listed)
                {
                    assert!(listed >= small.unlisted, "{heads:?} at {context}: a listing class {listed} below the unlisted {}", small.unlisted);
                }
                assert_eq!(
                    (small.family, small.unlisted),
                    (large.family, large.unlisted),
                    "{heads:?} at {context}: graph scratch follows the history reservation"
                );
                assert_eq!(
                    (plain.family, plain.unlisted),
                    (small.family, small.unlisted)
                );
                assert_eq!(
                    charged(heads, context, 256 << 30, false).family,
                    plain.family
                );
            }
        }
    }

    /// A chain's launches run in one submission, and at a split block a
    /// merge set of one run makes a pass of each run: a chain of 29
    /// launches of 512 rows runs 29 passes of 512 rows there, where a
    /// launch alone runs one. On CUDA, where one graph serves one run of
    /// its shape a submission, the family holds a graph for each, beyond
    /// one each its launches hold. For every chain length and merge set,
    /// what the group runs of its once graphs beyond one each, walked by
    /// the engine's own split (`Passes::sets`), is within what the census
    /// holds for the chain (`LaunchGraphs::chained`); where a graph updates
    /// in flight it holds none.
    #[test]
    fn a_chain_holds_a_graph_for_every_pass_it_runs() {
        let limits = ResourceLimits {
            max_launch_rows: 512,
            max_launch_slots: 64,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::new(29).unwrap(),
            conversations: std::num::NonZeroUsize::MIN,
        };
        let mut configuration = QWEN35_CONFIGURATIONS[3];
        configuration.blocks = configuration.attention_interval;
        let (definition, manifest) = declared_model(&configuration);
        for backend in [BackendName::Cuda, BackendName::Metal] {
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection { head: false, vision: false },
                resident_layout(ExecutionPath::Native, backend),
                PlannedMethod::Plain,
                limits,
            )
            .unwrap();
            let capacity = ResourceCapacity { domain_bytes: 64 << 30, tensor_operations: crate::TensorOperations::Absent };
            let state =
                ResourcePlanner::state_plan(&definition, &load, PlannedMethod::Plain, KvCodec::Dense, limits, capacity).unwrap();
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let certificate =
                certify_target_family::<seismic::NativeGraphLayout>(backend, &load, &definition.decoder, &state, plan.target(), limits, crate::PassLaunches::whole_chain(limits))
                    .unwrap();
            let chained = &certificate.graphs.chained;
            if backend.graphs_update_in_flight() {
                assert!(chained.is_empty(), "{backend:?}");
                continue;
            }
            assert_eq!(chained.len(), 28);
            let passes = certificate.passes;
            assert_eq!(passes.region, 512);
            let once = |index: usize, stage: OnceStage, rows: u64| {
                let first = certificate.outline.first_of_shape(index);
                certificate.blocks[first][&class_topology(once_class(rows, stage))].layout.storage_bytes().recorded
            };
            let split = certificate
                .served
                .iter()
                .enumerate()
                .filter(|(_, block)| block.split.is_some())
                .map(|(index, block)| (index, block.streams))
                .collect::<Vec<_>>();
            assert!(!split.is_empty());
            let further = |launches: usize, merged: usize| {
                let mut lengths = std::collections::BTreeMap::<usize, u64>::new();
                for set in passes.sets(launches, merged) {
                    *lengths.entry(set.len()).or_default() += 1;
                }
                split.iter().fold(seismic::GraphCensus::default(), |census, &(index, streams)| {
                    let census = lengths.iter().fold(census, |census, (&length, &count)| {
                        census.plus(once(index, OnceStage::Pass, passes.of_chain(length)).times(count - 1).unwrap()).unwrap()
                    });
                    match streams {
                        true => census
                            .plus(once(index, OnceStage::Fill, passes.rows()).times(launches as u64 - 1).unwrap())
                            .unwrap(),
                        false => census,
                    }
                })
            };
            for launches in 2..=29 {
                for merged in 1..=launches {
                    assert!(
                        further(launches, merged).within(chained[launches - 2]),
                        "{launches} launches, merge sets of {merged}: {:?} beyond {:?}",
                        further(launches, merged),
                        chained[launches - 2]
                    );
                }
            }
            // Merge sets of one run: 29 passes of 512 rows at each split
            // block, 28 of them beyond the launch's own.
            let single = split
                .iter()
                .fold(seismic::GraphCensus::default(), |census, &(index, _)| {
                    census.plus(once(index, OnceStage::Pass, 512).times(28).unwrap()).unwrap()
                });
            assert!(single.graphs >= 28 && single.within(chained[27]), "{single:?} beyond {:?}", chained[27]);
            assert!(
                certificate.graphs.held(29).unwrap().graphs
                    >= certificate.graphs.pinned.graphs * 28 + certificate.graphs.union.graphs + single.graphs
            );
        }
    }

    /// Every exact entry and block class seals into the layout certified for
    /// its structural regime and is charged that layout, and the family
    /// charges exactly the binding constants the exact classes bind.
    #[test]
    fn every_exact_target_class_seals_into_its_certified_layout() {
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 64,
            max_selected_rows: 64,
            max_drafting_slots: 64,
            exported_logits_rows: 0,
            max_images_per_request: 1,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
        };
        // Qwen3.6 with its odd blocks' experts pruned to half: routed layers
        // of two shapes, which every routed layer's views of one buffer of
        // each value serve.
        let mut pruned = QWEN35_CONFIGURATIONS[3];
        if let DeclaredFeedForward::Routed { odd_experts, .. } = &mut pruned.feed_forward {
            *odd_experts = 128;
        }
        for mut configuration in [QWEN35_CONFIGURATIONS[0], QWEN35_CONFIGURATIONS[3], pruned] {
            // One recurrent and one attention block shape.
            configuration.blocks = configuration.attention_interval;
            let (definition, manifest) = declared_model(&configuration);
            for (backend, codec, chain) in [
                BackendName::Cpu,
                BackendName::Metal,
                BackendName::Cuda,
                BackendName::Vulkan,
            ]
            .into_iter()
            .flat_map(|backend| [KvCodec::Dense, KvCodec::AffineK8V4].map(|codec| (backend, codec)))
            .flat_map(|(backend, codec)| [1, 2].map(|chain| (backend, codec, chain)))
            {
                let limits = ResourceLimits {
                    chain_capacity: std::num::NonZeroUsize::new(chain).unwrap(),
                    ..limits
                };
                let load = ModelLoadPlan::derive(
                    &manifest,
                    &definition,
                    ComponentSelection {
                        head: false,
                        vision: false,
                    },
                    resident_layout(ExecutionPath::Native, backend),
                    PlannedMethod::Plain,
                    limits,
                )
                .unwrap();
                let state = ResourcePlanner::state_plan(
                    &definition,
                    &load,
                    PlannedMethod::Plain,
                    codec,
                    limits,
                    ResourceCapacity {
                        domain_bytes: 64 * 1024 * 1024 * 1024,
                        tensor_operations: crate::TensorOperations::Absent,
                    },
                )
                .unwrap();
                let plan = load.program_plan(&definition, codec).unwrap();
                let certificate = certify_target_family(
                    backend,
                    &load,
                    &definition.decoder,
                    &state,
                    plan.target(),
                    limits,
                    crate::PassLaunches::whole_chain(limits),
                )
                .unwrap();
                let geometry = &definition.decoder;
                let max_segments = state.target_state().span_limit().next_power_of_two() as u64;
                let weight = load.weight(TARGET_TABLE).unwrap();
                let elements = [("EW", weight.resident), ("A", activation(geometry))];
                // The graphs a load seals and their plans' host heap, from
                // each exact class's own topology: blocks of one shape seal
                // one plan per class (no block here reads a layer scale).
                let (_, tapped) = block_taps(plan.target(), geometry).unwrap();
                let shapes = block_graph_shapes(&load, geometry, &state, &tapped, None).unwrap();
                let sealed = std::cell::Cell::new((0usize, HostHeap::default()));
                let seals = |heap: HostHeap| {
                    let (graphs, held) = sealed.get();
                    sealed.set((graphs + 1, held.plus(heap).unwrap()));
                };
                let owned = std::cell::Cell::new(HostHeap::default());
                let shared = FamilyConstants::default();
                let fits =
                    |graph: NativeGraphMetadata, layout: &NativeGraphLayout, class: String| {
                        // Every class of a regime seals a plan of the heap
                        // its layout plans.
                        assert_eq!(
                            graph.plan_heap(),
                            layout.plan_heap(),
                            "{} {backend:?} {class} plan heap",
                            configuration.model
                        );
                        let charged = graph.seal_with_layout(layout).unwrap_or_else(|error| {
                            panic!(
                            "{} {backend:?} {class} cannot seal into its certified layout: {error}",
                            configuration.model
                        )
                        });
                        assert_eq!(charged, layout.storage_bytes(), "{backend:?} {class}");
                    };
                let constants = std::cell::RefCell::new(CheckedGraphFamilyResources::new());
                let chain_rows = chain_rows(64, limits).unwrap();
                let passes = certificate.passes;
                let pass_rows = passes.rows();
                assert!(pass_rows <= chain_rows && passes.region == 64, "{passes:?}");
                let served = block_served(&load, geometry, limits, chain_rows, passes).unwrap();
                // Resident experts split only to merge a chain's launches.
                for (index, block) in served.iter().enumerate() {
                    let routed = block_routed(geometry, index).unwrap().is_some();
                    assert_eq!(block.split.is_some(), routed && chain > 1);
                }
                // Each merged experts buffer is as long as the longest layer's.
                let lengths = served
                    .iter()
                    .filter_map(|block| block.split.as_ref())
                    .map(|split| split.merged.experts.1.iter().product::<u64>())
                    .collect::<Vec<_>>();
                if let Some(split) = GroupExtents::of(None, None, None, &served, chain_rows).split {
                    assert_eq!(split.merged.experts.1, [*lengths.iter().max().unwrap()]);
                }
                // Each staged class's graph, drafted alone, fits its
                // certified layout and owns what it plans.
                let check = |index: usize, staged: StagedClass| {
                    let ((rows, ..), stage) = staged;
                    let (graph, mut block_constants, block_owned) = checked_block_graph_draft(
                        NativeGraphMetadata::new(backend),
                        plan.target().blocks()[index],
                        None,
                        BlockGraphInputs::of_class(
                            &load,
                            geometry,
                            &state,
                            index,
                            staged,
                            if stage == Stage::Once(OnceStage::Pass) { rows } else { pass_rows },
                        ),
                    )
                    .unwrap();
                    let certified = &certificate.blocks[index][&class_topology(staged)];
                    assert_eq!(
                        block_owned, certified.owned,
                        "{} {backend:?} block {index} {staged:?} owned vectors",
                        configuration.model
                    );
                    if !(0..index).any(|first| shapes[first] == shapes[index]) {
                        seals(graph.plan_heap());
                        owned.set(owned.get().plus(block_owned).unwrap());
                    }
                    for constant in &mut block_constants {
                        shared.share(constant);
                    }
                    fits(graph, &certified.layout, format!("block {index} {staged:?}"));
                    constants.borrow_mut().include(NativeGraphStorageBytes::default(), block_constants);
                };
                for rows in magnitude_batching::row_classes(limits.max_launch_rows)
                    .into_iter()
                    .map(|rows| rows as u64)
                {
                    for source in [EntryTokens::Uploaded, EntryTokens::Selected] {
                        let landing = CheckedLandingEntries::of(&load, TARGET_TABLE).unwrap();
                        let graph = entry_graph_topology(
                            NativeGraphMetadata::new(backend),
                            &elements,
                            notifies_selection(&load).then_some(&[][..]),
                            landing.as_ref().map(CheckedLandingEntries::entries),
                            keeps_token_rows(&load),
                            &weight,
                            EntryWidth {
                                hidden: &seismic::Dim::from(geometry.hidden),
                                transform: crate::programs::native_target_graph::entry::transform_arguments(embedding_transform(geometry)),
                            },
                            &seismic::Dim::from(rows),
                            source,
                        )
                        .unwrap()
                        .graph;
                        seals(graph.plan_heap());
                        fits(
                            graph,
                            &certificate.entries[&source],
                            format!("entry rows={rows}"),
                        );
                    }
                    for (index, block) in geometry.blocks.iter().enumerate() {
                        for class in block_classes(
                            paired_block(block).unwrap().keeps(),
                            rows,
                            limits.max_launch_slots as u64,
                            max_segments,
                            &block_listings(
                                &state,
                                backend,
                                plan.target().blocks()[index],
                                index,
                                rows,
                            )
                            .unwrap(),
                            slot_scores(plan.target().blocks()[index]),
                        ) {
                            for &stage in block_stages(&served, index, class.0) {
                                check(index, (class, Stage::Run(stage)));
                            }
                        }
                    }
                }
                for (index, block) in served.iter().enumerate() {
                    for (stage, rows) in once_graphs(block.streams, block.split.is_some(), passes) {
                        check(index, once_class(rows, stage));
                    }
                }
                let floor = GroupExtents::of(None, None, None, &served, 64);
                assert_eq!(
                    certificate.resources.binding_constant_bytes,
                    constants.into_inner().finish().unwrap().binding_constant_bytes
                        + floor.bytes().unwrap()
                        + floor.pass_bytes().unwrap(),
                    "{} {backend:?} {codec:?} constants",
                    configuration.model
                );
                assert!(certificate.sealed.exact());
                let planned = certificate.sealed.at_most;
                assert!(planned.plans.bytes > 0 && planned.owned.bytes > 0);
                assert_eq!(
                    (planned.graphs, planned.plans),
                    sealed.get(),
                    "{} {backend:?} {codec:?} sealed graphs",
                    configuration.model
                );
                assert_eq!(
                    planned.owned,
                    owned.get(),
                    "{} {backend:?} {codec:?} owned vectors",
                    configuration.model
                );
                // The values the exact classes bind, each held once, are
                // the ones the plan counted.
                assert_eq!(
                    certificate.sealed.constants,
                    shared.host_heap().unwrap(),
                    "{} {backend:?} {codec:?} constant values",
                    configuration.model
                );
            }
        }
    }

    /// A plan charges exactly the pass its load holds when its pass budget
    /// exceeds what the arena fits (memory-failures item 8): the Qwen4Exp
    /// fixture streamed on Metal in chains of 1024 launches, more than its arena fits, its bank built
    /// at every place (a budget of the whole chain), holds passes the arena
    /// caps (`group_pass_rows`), and the family the plan charges holds its
    /// prompt groups' class at those passes' places, never at a pass above
    /// the cap.
    #[test]
    fn a_budget_above_the_cap_charges_the_pass_the_load_holds() {
        use crate::planning::fixtures::{LIMITS, qwen4exp_fixture_definition_of, streamed_experts_manifest};
        let backend = BackendName::Metal;
        let chain = 1024;
        let limits = ResourceLimits { chain_capacity: std::num::NonZeroUsize::new(chain).unwrap(), ..LIMITS };
        let definition = qwen4exp_fixture_definition_of(4, 32);
        let manifest = streamed_experts_manifest(&definition);
        let selection = ComponentSelection { head: false, vision: false };
        let reservation = seismic::Reservation { granule: 2 << 20, host_addressed: true, chunks: None };
        let substrate = crate::planning::BankSubstrate { backend, reservation, host: crate::planning::HostTier::Unified(crate::planning::Parking::of(backend)) };
        let derived = ModelLoadPlan::derive(&manifest, &definition, selection, resident_layout(ExecutionPath::Native, backend), PlannedMethod::Plain, limits).unwrap();
        let floor = derived.bank_floor(&definition, chain as u64, crate::FlightDepth::PIPELINED, substrate).unwrap();
        let whole = derived.with_expert_bank(&definition, floor, chain as u64, crate::FlightDepth::PIPELINED, substrate).unwrap();
        let capacity = ResourceCapacity { domain_bytes: 64 << 30, tensor_operations: crate::TensorOperations::Absent };
        let state = ResourcePlanner::state_plan(&definition, &whole, PlannedMethod::Plain, KvCodec::Dense, limits, capacity).unwrap();
        let plan = whole.program_plan(&definition, KvCodec::Dense).unwrap();
        let budget = whole.pass(limits);
        let certificate =
            certify_target_family::<seismic::NativeGraphLayout>(backend, &whole, &definition.decoder, &state, plan.target(), limits, budget).unwrap();
        let passes = certificate.passes;
        let held = certificate.group.extents(&certificate.served, 1).pass_bytes().unwrap();
        let found = format!("budget {budget:?}, {passes:?}, {held} bytes held");
        assert!(budget.get() > passes.launches(), "setup: the budget is within the cap, so nothing here exceeds it: {found}");
        assert!(held > 0, "setup: no block of the fixture splits, so it holds no pass buffers: {found}");
        let checked = super::super::certify::checked_target_family(backend, &whole, &definition.decoder, &state, plan.target(), limits, budget).unwrap();
        assert_eq!(checked.group_bytes.most().get(), passes.launches(), "{found}");
    }
}

#[cfg(test)]
mod sinkhorn_graph_tests {
    use super::super::block::stage_input;
    use super::*;
    use magnitude_family_contracts::ModelDefinition;
    use crate::operators::streams::ResidualCarry;
    use crate::planning::fixtures::{fixture_manifest, qwen4exp_fixture_definition, sinkhorn_fixture_definition};
    use crate::{ComponentSelection, ExecutionPath, PlannedMethod, ResourceCapacity, ResourcePlanner, resident_layout};
    use magnitude_state::KvCodec;

    const STAGES: [RunStage; 3] = [RunStage::Whole, RunStage::Front, RunStage::Combine];

    /// The carry each block graph reads: a Sinkhorn decoder's carries the
    /// `[rows, S]` pre beside its streams into every block after the first
    /// and every back; a low-rank decoder's (qwen4exp's) reads its rows
    /// alone, the one port it read before. Each carries its own hidden
    /// width's rows, its `S` streams of them where it reads streams.
    #[test]
    fn only_sinkhorn_streams_carry_a_pre() {
        use crate::planning::fixtures::STREAMS;
        let (sinkhorn, low_rank) = (sinkhorn_fixture_definition(2), qwen4exp_fixture_definition());
        for (block, stage) in (0..2).flat_map(|block| STAGES.map(|stage| (block, stage))) {
            let streams = block > 0 || stage == RunStage::Combine;
            let rows = |definition: &ModelDefinition| match streams {
                true => STREAMS * definition.decoder.hidden,
                false => definition.decoder.hidden,
            };
            let input = |definition: &ModelDefinition| stage_input(&definition.decoder, block, Stage::Run(stage)).unwrap();
            assert_eq!(input(&low_rank), Some(ResidualCarry::rows(rows(&low_rank))), "block {block} {stage:?}");
            assert_eq!(
                input(&sinkhorn),
                Some(ResidualCarry { rows: rows(&sinkhorn), pre: streams.then_some(STREAMS) }),
                "block {block} {stage:?}"
            );
        }
    }

    /// The Sinkhorn fixture's block graphs check on every backend, whole at
    /// decode and prompt rows and as a split block's front and back.
    #[test]
    fn sinkhorn_block_graphs_check_on_every_backend() {
        let definition = sinkhorn_fixture_definition(2);
        let manifest = fixture_manifest(&definition);
        let codec = KvCodec::Dense;
        let selection = ComponentSelection { head: false, vision: false };
        let limits = crate::planning::fixtures::LIMITS;
        for backend in BackendName::ALL {
            let layout = resident_layout(ExecutionPath::Native, backend);
            let load = ModelLoadPlan::derive(&manifest, &definition, selection, layout, PlannedMethod::Plain, limits).unwrap();
            let capacity = ResourceCapacity { domain_bytes: 1 << 30, tensor_operations: crate::TensorOperations::Absent };
            let state = ResourcePlanner::state_plan(&definition, &load, PlannedMethod::Plain, codec, limits, capacity).unwrap();
            let plan = load.program_plan(&definition, codec).unwrap();
            let classes = [(1, RunStage::Whole), (512, RunStage::Whole), (512, RunStage::Front), (512, RunStage::Combine)];
            for (index, (rows, stage)) in (0..2).flat_map(|index| classes.map(|class| (index, class))) {
                let inputs = BlockGraphInputs::of_class(
                    &load,
                    &definition.decoder,
                    &state,
                    index,
                    ((rows, 1, 1, 0, 0), Stage::Run(stage)),
                    rows,
                );
                let checked = checked_block_graph_draft(NativeGraphMetadata::new(backend), plan.target().blocks()[index], None, inputs);
                let label = format!("{backend:?} block {index} {rows} rows {stage:?}");
                drop(checked.unwrap_or_else(|error| panic!("{label}: {error}")));
            }
        }
    }
}
