//! Zero-cost-when-off thread-local stage breakdown timer for model assessment.

use std::cell::RefCell;
use std::time::Instant;

#[derive(Default, Clone, Debug)]
pub struct StageTimings {
    pub header_open_ns: u64,
    pub chat_inspect_ns: u64,
    pub admit_other_ns: u64,
    pub plan_resident_ns: u64,
    pub target_ns: u64,
    pub readout_ns: u64,
    pub native_head_ns: u64,
    pub native_head_groups: u64,
    pub other_drafter_ns: u64,
    pub vision_state_ns: u64,
    pub host_process_ns: u64,
    pub decode_demand_ns: u64,

    // SizingDraft deep breakdown:
    pub sd_drafts_built: u64,
    pub sd_nodes_enqueued: u64,
    pub sd_slice_sets: u64,
    pub sd_port_param_spec_ns: u64,
    pub sd_enqueue_ns: u64,
    pub sd_operand_links_ns: u64,
    pub sd_storage_compile_ns: u64,
    pub sd_cover_ports_ns: u64,
    pub sd_scratch_configs_hit_ns: u64,
    pub sd_scratch_configs_miss_ns: u64,
    pub sd_scratch_configs_hits: u64,
    pub sd_scratch_configs_misses: u64,
    pub sd_sizing_point_hit_ns: u64,
    pub sd_sizing_point_miss_ns: u64,
    pub sd_sizing_point_hits: u64,
    pub sd_sizing_point_misses: u64,
    pub sd_storage_eval_ns: u64,
    pub sd_configs_by_entry: std::collections::BTreeMap<&'static str, (u64, u64)>,
    pub sd_point_by_entry: std::collections::BTreeMap<&'static str, (u64, u64)>,

    // admit_inner deep breakdown:
    pub admit_fingerprint_ns: u64,
    pub admit_unsupported_weights_ns: u64,
    pub admit_interpret_target_ns: u64,
    pub admit_tokenizer_vocab_ns: u64,
    pub admit_gguf_templates_ns: u64,
    pub admit_drafter_ns: u64,

    // plan_resident (ExecutionPlanner::backend_plan) deep breakdown:
    pub pr_load_derive_ns: u64,
    pub pr_record_ports_ns: u64,
    pub pr_readout_candidates_ns: u64,
    pub pr_program_plan_ns: u64,

    // floor_forms (execution_plan.rs) deep breakdown:
    pub ff_bank_ceiling_ns: u64,
    pub ff_rising_ns: u64,
    pub ff_whole_chains_ns: u64,
}

thread_local! {
    static ACTIVE: RefCell<Option<StageTimings>> = const { RefCell::new(None) };
}

#[inline]
pub fn is_recording() -> bool {
    ACTIVE.with(|cell| cell.borrow().is_some())
}

#[inline]
pub fn record_stage(f: impl FnOnce(&mut StageTimings)) {
    ACTIVE.with(|cell| {
        if let Some(timings) = cell.borrow_mut().as_mut() {
            f(timings);
        }
    });
}

pub fn measure_stages<T>(f: impl FnOnce() -> T) -> (T, StageTimings) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|cell| {
                *cell.borrow_mut() = None;
            });
        }
    }

    ACTIVE.with(|cell| {
        *cell.borrow_mut() = Some(StageTimings::default());
    });
    let _reset = Reset;
    let result = f();
    let timings = ACTIVE.with(|cell| cell.borrow_mut().take().unwrap_or_default());
    (result, timings)
}

#[inline]
pub fn time_stage<T>(add: impl FnOnce(&mut StageTimings, u64), f: impl FnOnce() -> T) -> T {
    if !is_recording() {
        return f();
    }
    let start = Instant::now();
    let result = f();
    let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    record_stage(|timings| add(timings, elapsed));
    result
}
