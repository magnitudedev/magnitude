//! Synthetic executor costs: one sample of every figure a plan prices
//! with, for tests that price a plan without a measurement job.

use magnitude_artifacts::gguf::Encoding;
use magnitude_family_contracts::ScaledActivations;

/// The encodings the synthetic costs measure: every one the fixtures and
/// the catalog's models hold.
const FORMATS: [Encoding; 16] = [
    Encoding::Q8_0,
    Encoding::Q5K,
    Encoding::Q6K,
    Encoding::Q4K,
    Encoding::Mxfp4,
    Encoding::Iq4Nl,
    Encoding::Nvfp4,
    Encoding::Q5_1,
    Encoding::Iq2Xs,
    Encoding::Q4_0,
    Encoding::Q5_0,
    Encoding::Iq2Xxs,
    Encoding::Iq3Xxs,
    Encoding::Q3K,
    Encoding::Iq4Xs,
    Encoding::Iq3S,
];

/// Synthetic executor costs, each from one sample: 300 GB/s weight
/// streaming of every format, both storage kinds on device storage's
/// route; 20 TFLOP/s over every format's representation on every backend
/// (with the dense f32 and bf16); every history layout at 10 µs per call
/// plus 200 GB/s; 5 µs launches; a 1.2 composition ratio; a 50 µs
/// pipelined and a 150 µs round-trip step; a 1.1 release step ratio.
pub fn costs() -> super::ExecutorCosts {
    use super::executor_costs::{
        FormatRates, HistoryCosts, MatrixCosts, MeasuredCost, Placement, Placements, RouteCosts,
        WeightRate, HISTORY_DEPTHS,
    };
    let cost = |value| MeasuredCost {
        value,
        interval: None,
        samples: 1,
    };
    let mut formats = seismic::BackendName::ALL
        .into_iter()
        .flat_map(|backend| {
            let layout = crate::resident_layout(crate::ExecutionPath::Native, backend);
            FORMATS.iter().flat_map(move |encoding| {
                [ScaledActivations::Rounded, ScaledActivations::Nvfp4]
                    .into_iter()
                    .filter_map(move |activations| {
                        crate::resident_element(
                            *encoding,
                            activations,
                            4096,
                            seismic::DType::BF16,
                            layout,
                        )
                    })
            })
        })
        .chain([seismic::Element::bf16(), seismic::Element::f32()])
        .collect::<Vec<_>>();
    formats.sort_unstable_by_key(|element| element.name());
    formats.dedup();
    formats.retain(|format| *format != seismic::Element::bf16());
    let rate = || WeightRate {
        bytes_per_second: cost(300e9),
        features: 16384,
    };
    let matrix = || MatrixCosts {
        flops: cost(20e12),
        features: 16384,
    };
    // 10 µs a call and its history at 200 GB/s, at every rung.
    let history = |layout| {
        let per_token = super::executor_costs::token_bytes(layout).unwrap() as f64;
        let at = |depth: u64| cost(10e-6 + depth as f64 * per_token / 200e9);
        HistoryCosts::new(layout, HISTORY_DEPTHS.iter().map(|&depth| at(depth)).collect())
            .unwrap()
    };
    super::ExecutorCosts {
        placements: Placements {
            device: Placement::along(kind(0)),
            upload: Placement::along(kind(0)),
            host: Placement::along(kind(0)),
        },
        first: RouteCosts {
            route: kind(0),
            activation: rate(),
            formats: formats
                .iter()
                .map(|&format| {
                    let rates = FormatRates {
                        pilot: rate(),
                        streamed: Some(rate()),
                    };
                    (format, rates)
                })
                .collect(),
        },
        routes: Vec::new(),
        activation_matrix: Some(matrix()),
        matrices: formats
            .into_iter()
            .map(|format| (format, matrix()))
            .collect(),
        dense: history(magnitude_state::KvCodec::Dense),
        affine: history(magnitude_state::KvCodec::AffineK8V4),
        // A chain of one launch-only call a layer, read at 1.2 times its
        // price: a composition of 1.2.
        chain: super::executor_costs::ChainCost {
            timing: seismic::measure::Timing { median: 1.2 * 8. * 5e-6, interval: None, samples: 8 },
            layers: 8,
            calls: vec![super::demand::Work::default()],
        },
        launch_seconds: cost(5e-6),
        step_pipelined_seconds: cost(50e-6),
        step_round_trip_seconds: cost(150e-6),
    }
}

/// The synthetic costs' storage kind `id`: one of a CPU's, standing for any
/// executor's.
pub fn kind(id: u32) -> seismic::NativeKind {
    seismic::NativeKind { backend: seismic::BackendName::Cpu, id }
}

/// Synthetic streamed figures: a model store read in 100 µs plus 5 GB/s,
/// host memory at 50 GB/s, no CPU site, a convert at 100 GB/s.
pub fn streamed_figures() -> super::StreamedFigures {
    super::StreamedFigures {
        read: super::ReadLine::new(100e-6, 1.0 / 5e9).expect("a synthetic read line is valid"),
        memory: Some(1.0 / 50e9),
        site: None,
        convert: 1.0 / 100e9,
    }
}

/// Synthetic plan prices: [`costs`], [`streamed_figures`], and a resident
/// step of 10 ms, 100 µs a row and 10 ns a token of depth, its streamed
/// form's what the banked plan's streamed form adds to its experts held
/// whole on [`costs`] (`assessment::decode_step`).
pub fn plan_prices() -> crate::PlanPrices {
    crate::PlanPrices::new(
        costs(),
        streamed_figures(),
        Box::new(|definition, banked| {
            let forms = super::decode_step(definition, &banked.held_whole(definition)?, banked, costs())?;
            Ok(Box::new(move |rows, depth| {
                let forms = forms(rows, depth)?;
                let resident = 10e-3 + f64::from(rows) * 100e-6 + f64::from(depth) * 10e-9;
                Ok(super::FormSeconds { resident, streamed: resident + forms.streamed - forms.resident })
            }))
        }),
    )
}
