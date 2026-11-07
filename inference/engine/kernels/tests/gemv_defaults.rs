//! The GEMV rule (`kernels/dense_rows.seismic`) as Metal evaluates it, without a device,
//! at every shape the M4 Pro opens of Qwen3.6-35B-A3B (with its DFlash drafter) and a
//! 2560-wide model recorded: each declaration's BATCH_FROM and each GEMV launch's
//! SIMDGROUPS, ROWS and LANES at their defaults. A change to the rule, or to a
//! declaration's terms, changes a value here.
//!
//! Where the rule moved a default, the case names the old value: attention_project's
//! 2048-row K/V shape takes two weight rows a group (was one), stream_mix_down 16
//! simdgroups at one row (was 4), stream_mix_up at width 320 8-lane groups and
//! BATCH_FROM 3 (were 16 and 5), and readout_top_rows BATCH_FROM 3 (was 5). The indexer's
//! attention_project binds no gate or value weight (GR = V = 0), so GW and VW are bound to
//! its query weight's BF16.

use seismic::{BackendName, Element, NativeSpecialization};

struct Case {
    entry: &'static str,
    elements: &'static [(&'static str, &'static str)],
    statics: &'static [(&'static str, u64)],
    batch_from: Option<u64>,
    /// Each GEMV launch, by the suffix of its kernel name: SIMDGROUPS, ROWS, LANES.
    launches: &'static [(&'static str, [u64; 3])],
}

const CASES: &[Case] = &[
    Case { entry: "attention_output", elements: &[("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("IN", 0), ("OB", 128), ("OS", 0), ("Q", 16), ("SN", 1), ("W", 256)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "attention_output", elements: &[("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("IN", 0), ("OB", 128), ("OS", 0), ("Q", 32), ("SN", 1), ("W", 128)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "attention_output", elements: &[("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2560), ("IN", 4), ("OB", 192), ("OS", 0), ("Q", 24), ("SN", 4), ("W", 256)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "attention_output", elements: &[("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2560), ("IN", 4), ("OB", 192), ("OS", 0), ("Q", 48), ("SN", 4), ("W", 128)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "attention_project", elements: &[("NW", "bf16"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("KW", "q8g32s@rows32"), ("VW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("GB", 0), ("GR", 0), ("GS", 0), ("K", 512), ("KB", 64), ("KS", 0), ("NM", 2048), ("PT", 1), ("Q", 0), ("QB", 0), ("QS", 0), ("V", 512), ("VB", 64), ("VS", 0)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "attention_project", elements: &[("NW", "bf16"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("KW", "q8g32s@rows32"), ("VW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("GB", 0), ("GR", 0), ("GS", 0), ("K", 512), ("KB", 64), ("KS", 0), ("NM", 2048), ("PT", 1), ("Q", 8192), ("QB", 64), ("QS", 0), ("V", 512), ("VB", 64), ("VS", 0)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "attention_project", elements: &[("NW", "bf16"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("KW", "q8g32s@rows32"), ("VW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("GB", 0), ("GR", 0), ("GS", 0), ("K", 1024), ("KB", 64), ("KS", 0), ("NM", 2048), ("PT", 1), ("Q", 0), ("QB", 0), ("QS", 0), ("V", 1024), ("VB", 64), ("VS", 0)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "attention_project", elements: &[("NW", "bf16"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("KW", "q8g32s@rows32"), ("VW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("GB", 0), ("GR", 0), ("GS", 0), ("K", 1024), ("KB", 64), ("KS", 0), ("NM", 2048), ("PT", 1), ("Q", 4096), ("QB", 64), ("QS", 0), ("V", 1024), ("VB", 64), ("VS", 0)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "attention_project", elements: &[("NW", "f32"), ("QW", "bf16"), ("GW", "bf16"), ("KW", "bf16"), ("VW", "bf16"), ("A", "bf16")], statics: &[("D", 2560), ("GB", 0), ("GR", 0), ("GS", 0), ("K", 128), ("KB", 0), ("KS", 0), ("NM", 0), ("PT", 4), ("Q", 512), ("QB", 0), ("QS", 0), ("V", 0), ("VB", 0), ("VS", 0)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "attention_project", elements: &[("NW", "f32"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("KW", "q8g32s@rows32"), ("VW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2560), ("GB", 0), ("GR", 0), ("GS", 0), ("K", 512), ("KB", 80), ("KS", 0), ("NM", 0), ("PT", 4), ("Q", 0), ("QB", 0), ("QS", 0), ("V", 512), ("VB", 80), ("VS", 0)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "attention_project", elements: &[("NW", "f32"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("KW", "q8g32s@rows32"), ("VW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2560), ("GB", 0), ("GR", 0), ("GS", 0), ("K", 512), ("KB", 80), ("KS", 0), ("NM", 0), ("PT", 4), ("Q", 12288), ("QB", 80), ("QS", 0), ("V", 512), ("VB", 80), ("VS", 0)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "dense_expand", elements: &[("NW", "bf16"), ("GW", "q8g32s@rows32"), ("UW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("F", 6144), ("GB", 64), ("GS", 0), ("H", 2048), ("UB", 64), ("US", 0)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "dense_output", elements: &[("DW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("DB", 192), ("DS", 0), ("F", 6144), ("H", 2048)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "gated_delta_project", elements: &[("NW", "bf16"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("AW", "bf16"), ("BW", "bf16"), ("A", "bf16")], statics: &[("H", 2048), ("NK", 16), ("NM", 2048), ("NV", 32), ("PT", 1), ("QB", 64), ("W", 128), ("ZB", 64)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "gated_delta_project", elements: &[("NW", "f32"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("AW", "bf16"), ("BW", "bf16"), ("A", "bf16")], statics: &[("H", 2560), ("NK", 16), ("NM", 0), ("NV", 48), ("PT", 4), ("QB", 80), ("W", 128), ("ZB", 80)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "gated_delta_project_convolved", elements: &[("NW", "bf16"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("AW", "bf16"), ("BW", "bf16"), ("A", "bf16")], statics: &[("C", 4), ("H", 2048), ("NK", 16), ("NM", 2048), ("NV", 32), ("PT", 1), ("W", 128)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "gated_delta_project_convolved", elements: &[("NW", "f32"), ("QW", "q8g32s@rows32"), ("GW", "q8g32s@rows32"), ("AW", "bf16"), ("BW", "bf16"), ("A", "bf16")], statics: &[("C", 4), ("H", 2560), ("NK", 16), ("NM", 0), ("NV", 48), ("PT", 4), ("W", 128)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 1, 8]), ("gemv_pair", [8, 1, 8])] },
    Case { entry: "project_rows", elements: &[("A", "bf16"), ("W", "q8g32s@rows32"), ("Y", "f32")], statics: &[("K", 2560), ("N", 2560), ("WB", 80), ("WS", 0)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "project_rows", elements: &[("A", "bf16"), ("W", "q8g32s@rows32"), ("Y", "f32")], statics: &[("K", 2560), ("N", 10240), ("WB", 80), ("WS", 0)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "project_rows", elements: &[("A", "bf16"), ("W", "q8g32s@rows32"), ("Y", "f32")], statics: &[("K", 16384), ("N", 2048), ("WB", 512), ("WS", 0)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "readout_head_rows", elements: &[("NW", "bf16"), ("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("NM", 2048), ("PT", 1), ("V", 248320), ("WS", 0)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "readout_head_rows", elements: &[("NW", "f32"), ("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2560), ("NM", 0), ("PT", 4), ("V", 248320), ("WS", 0)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16]), ("gemv_rows", [8, 2, 8]), ("gemv_pair", [8, 2, 8])] },
    Case { entry: "readout_planes_rows", elements: &[("NW", "bf16"), ("A", "bf16")], statics: &[("D", 2048), ("NM", 2048), ("P", 3), ("PT", 1), ("V", 248320)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "readout_project_rows", elements: &[("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("I", 0), ("S", 248320), ("V", 248320), ("WS", 0)], batch_from: None, launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "readout_project_rows", elements: &[("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2560), ("I", 0), ("S", 248320), ("V", 248320), ("WS", 0)], batch_from: None, launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "readout_selected_rows", elements: &[("NW", "bf16"), ("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2048), ("NM", 2048), ("PT", 1), ("V", 248320)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "readout_selected_rows", elements: &[("NW", "f32"), ("OW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("D", 2560), ("NM", 0), ("PT", 4), ("V", 248320)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "readout_top_rows", elements: &[("NW", "bf16"), ("A", "bf16")], statics: &[("D", 2048), ("NM", 2048), ("P", 3), ("PT", 1), ("V", 248320)], batch_from: Some(3), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "stream_mix_down", elements: &[("NW", "bf16"), ("DW", "q8g32s@rows32"), ("IW", "bf16"), ("A", "bf16")], statics: &[("C", 1), ("H", 2560), ("LR", 320), ("S", 4)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "stream_mix_down", elements: &[("NW", "bf16"), ("DW", "q8g32s@rows32"), ("IW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("C", 0), ("H", 2560), ("LR", 320), ("S", 4)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "stream_mix_down", elements: &[("NW", "bf16"), ("DW", "q8g32s@rows32"), ("IW", "q8g32s@rows32"), ("A", "bf16")], statics: &[("C", 1), ("H", 2560), ("LR", 320), ("S", 4)], batch_from: Some(5), launches: &[("gemv", [16, 1, 16])] },
    Case { entry: "stream_mix_up", elements: &[("UW", "q8g32s@rows32"), ("NW", "bf16"), ("A", "bf16")], statics: &[("H", 2560), ("LR", 320), ("S", 4)], batch_from: Some(3), launches: &[("gemv", [16, 1, 8])] },
];

/// The launches' kernel names and the default specialization of `E`'s Metal
/// declaration at `case`'s elements and statics.
fn defaults<E: seismic::Entry>(case: &Case) -> (Vec<String>, NativeSpecialization) {
    let elements = case
        .elements
        .iter()
        .map(|(name, element)| (*name, Element::named(element).unwrap()))
        .collect::<Vec<_>>();
    let implementation = seismic::generated::native_implementation_at::<E>(BackendName::Metal, &elements)
        .unwrap()
        .expect("the declaration forms at these elements");
    let bound = implementation
        .device_independent()
        .expect("the declaration reads no device property");
    let statics = implementation.statics.iter().fold(NativeSpecialization::new(), |fixed, name| {
        let (_, value) = case
            .statics
            .iter()
            .find(|(recorded, _)| recorded == name)
            .unwrap_or_else(|| panic!("{}: static {name} is not recorded", case.entry));
        fixed.with_static(name.clone(), *value)
    });
    let specialization = bound.default_specialization(&statics).unwrap();
    (implementation.launches.iter().map(|launch| launch.kernel.clone()).collect(), specialization)
}

fn evaluate(case: &Case) -> (Vec<String>, NativeSpecialization) {
    use magnitude_kernels::*;
    match case.entry {
        "attention_output" => defaults::<attention_output::Entry>(case),
        "attention_project" => defaults::<attention_project::Entry>(case),
        "dense_expand" => defaults::<dense_expand::Entry>(case),
        "dense_output" => defaults::<dense_output::Entry>(case),
        "gated_delta_project" => defaults::<gated_delta_project::Entry>(case),
        "gated_delta_project_convolved" => defaults::<gated_delta_project_convolved::Entry>(case),
        "project_rows" => defaults::<project_rows::Entry>(case),
        "readout_head_rows" => defaults::<readout_head_rows::Entry>(case),
        "readout_planes_rows" => defaults::<readout_planes_rows::Entry>(case),
        "readout_project_rows" => defaults::<readout_project_rows::Entry>(case),
        "readout_selected_rows" => defaults::<readout_selected_rows::Entry>(case),
        "readout_top_rows" => defaults::<readout_top_rows::Entry>(case),
        "stream_mix_down" => defaults::<stream_mix_down::Entry>(case),
        "stream_mix_up" => defaults::<stream_mix_up::Entry>(case),
        other => panic!("no case dispatch for {other}"),
    }
}

#[test]
fn metal_gemv_defaults_follow_the_rule_at_every_recorded_shape() {
    for case in CASES {
        let (kernels, specialization) = evaluate(case);
        let at = format!("{} {:?}", case.entry, case.statics);
        assert_eq!(specialization.param("BATCH_FROM"), case.batch_from, "BATCH_FROM: {at}");
        let mut checked = 0;
        for (index, kernel) in kernels.iter().enumerate() {
            let Some((suffix, [simdgroups, rows, lanes])) = case
                .launches
                .iter()
                .find(|(suffix, _)| kernel.strip_prefix(case.entry).and_then(|rest| rest.strip_prefix('_')) == Some(*suffix))
            else {
                continue;
            };
            let found = ["SIMDGROUPS", "ROWS", "LANES"].map(|name| specialization.launch_param(index, name));
            assert_eq!(found, [Some(*simdgroups), Some(*rows), Some(*lanes)], "{suffix}: {at}");
            checked += 1;
        }
        assert_eq!(checked, case.launches.len(), "every GEMV launch is declared: {at}");
    }
}
