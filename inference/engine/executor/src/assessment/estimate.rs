//! Analytical decode-speed estimation (spec §2.14): what a model's decode
//! achieves on this machine, from its census and what the engine's kernels
//! measurably achieve on its executor ([`ExecutorCosts`]'s figures), priced
//! by the cost model (`magnitude_cost_model::step_seconds`, over
//! Seismic's per-call formula, `seismic::measure::pricing`).
//!
//! A step decodes one token per conversation: plain decode; speculation is
//! not priced. At N sequences it runs N rows and costs its calls, scaled by
//! the measured composition ratio, plus the step cost of the model's
//! resolved lookahead policy. A call costs the slower of its bytes and its
//! arithmetic, plus its attention's fixed cost and its launches:
//!
//! - bytes: weights (a routed call's union of the rows' chosen experts,
//!   E · (1 − (1 − k/E)^N) slabs, as every row chooses its own), each
//!   format's at its own weight rate, the slowest of its storage's
//!   placement paths (the engine's projection over that format streaming
//!   from there, which prices the kernel and the path at once); other
//!   depth-independent bytes (N rows, N sequences' state, uploaded rows) at
//!   the activation dtype's (bf16) rate; history (N sequences' reads at
//!   depth d, in device storage) at its layout's attention rate, measured
//!   along device storage's first route and scaled down where another of
//!   the placement's routes is slower (an indexer's block keys, which no
//!   attention entry reads, at the activation dtype's rate);
//! - arithmetic: N rows' matmul FLOPs over their weight format's
//!   throughput, and attention's over the bf16 matrix throughput (its score
//!   and value products are bf16 matrix products);
//! - an attention call's layout's fixed cost per call;
//! - launches: the backend's declared launches at N rows times the launch
//!   cost.
//!
//! A step also pays each submission boundary its calls end beyond its own
//! (a streamed layer's parks and publications where the backend's parks
//! are boundaries, `DecodeDemand::boundaries`) at the executor's round trip
//! beyond device time (`ExecutorCosts::step_round_trip_seconds`): one
//! submit and wait. What a streamed layer's waits add beside it has no
//! figure yet: a spinning wait's device time (a landing launch's or sited
//! combine's kernel past its launch), the parks' keeper's latency in
//! answering a park, and the submitting thread's work beginning each
//! submission.
//!
//! Where it can be off, and in which direction: about ±15% typically, the
//! spread of the composition ratio across models once format, layout,
//! launch and composition are measured (spec §2.14, Expected error). The
//! remaining sources: a matrix's shape at one row (the launch-plus-bytes
//! rule's median 7–18%, worst for weights under 2 MB), MoE expert
//! projections at few rows, recurrent and state-space kernels (priced by
//! their bytes only, their arithmetic is not derived), attention's
//! arithmetic at many rows (priced at the bf16 matrix throughput, not
//! measured on the attention entry), contention between devices, compute
//! overlapping transfers, and sustained throttling. A figure the
//! executor's costs have not read yet is priced from those they have, by
//! the costs' own accessors (`ExecutorCosts::route_rate`,
//! `ExecutorCosts::matrix_flops`, `ExecutorCosts::rates`), until a
//! job reads it.

use super::demand::format_of;
use super::executor_costs::ExecutorCosts;
use super::AssessmentError;
use magnitude_cost_model::{Census, ExpertPass, RoutedKind};
use seismic::measure::pricing::{Bytes, Flops, Pricing};

/// The decode speed at one depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerformanceEstimate {
    /// The depth: tokens of history each conversation's step reads.
    pub context_tokens: u32,
    /// Tokens per second per conversation.
    pub tokens_per_second: f64,
}

/// The depths to estimate: requested depths above the context limit are
/// dropped, duplicates removed, the limit itself always included; ascending.
pub fn performance_depths(context_limit: u32, requested: &[u32]) -> Vec<u32> {
    let mut depths = requested
        .iter()
        .copied()
        .filter(|depth| *depth > 0 && *depth < context_limit)
        .collect::<Vec<_>>();
    depths.push(context_limit);
    depths.sort_unstable();
    depths.dedup();
    depths
}

/// The passes of `bank`'s streamed layers on the executor `costs` price
/// ([`ExpertPass`]): each streamed layer's routed kind
/// (`ExpertBankPlan::routed_kinds`) and expert, one record of its bank
/// group's tensors as resident, its bytes and its FLOPs a choice.
pub fn expert_pass(bank: &crate::planning::ExpertBankPlan, costs: &ExecutorCosts) -> Result<ExpertPass, AssessmentError> {
    let layers = bank
        .layers()
        .iter()
        .zip(bank.routed_kinds())
        .map(|(layer, kind)| {
            let records = &bank.groups()[layer.group].record;
            let bytes = records
                .iter()
                .map(|(_, record)| {
                    let bytes = record.record_bytes().map_err(AssessmentError::Plan)?;
                    Ok(Bytes { format: format_of(record.resident), bytes })
                })
                .collect::<Result<Vec<_>, AssessmentError>>()?;
            let flops = records
                .iter()
                .map(|(_, record)| Flops { format: format_of(record.resident), flops: 2 * record.matrix[0] * record.matrix[1] })
                .collect();
            Ok((kind, bytes, flops))
        })
        .collect::<Result<Vec<(RoutedKind, Vec<Bytes>, Vec<Flops>)>, AssessmentError>>()?;
    let formats = layers.iter().flat_map(|(_, bytes, _)| bytes.iter().map(|bytes| bytes.format)).collect::<Vec<_>>();
    Ok(ExpertPass::new(costs.rates(&formats, &[]), layers, costs.host_reads()))
}

/// One estimate per depth: tokens per second per conversation, of
/// `census`'s decode at `sequences` sequences.
pub fn estimate_decode(census: &Census, pricing: &Pricing<'_>, sequences: u64, depths: &[u32]) -> Result<Vec<PerformanceEstimate>, AssessmentError> {
    depths
        .iter()
        .map(|&depth| {
            Ok(PerformanceEstimate {
                context_tokens: depth,
                tokens_per_second: 1.0 / magnitude_cost_model::step_seconds(census, pricing, sequences, depth)?.total(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::demand::{layout_of, Call};
    use crate::assessment::executor_costs::{
        HistoryCosts, MeasuredCost, Placement, RouteCosts, WeightRate, DECODE_LINE,
    };
    use crate::assessment::launches::{Declarations, Kernel};
    use magnitude_state::KvCodec;
    use crate::assessment::fixtures::kind;
    use crate::assessment::DecodeDemand;
    use seismic::measure::pricing::{CallSeconds, Experts, HistoryRead, LaunchSeconds, PriceError, Work};
    use seismic::Element;

    /// `demand`'s decode step of `sequences` sequences at `depth` on
    /// `costs`' figures, under `lookahead`.
    fn step(demand: &DecodeDemand, costs: &ExecutorCosts, lookahead: bool, sequences: u64, depth: u32) -> Result<LaunchSeconds, PriceError> {
        let census = demand.census();
        let figures = costs.figures(&[&census]);
        magnitude_cost_model::step_seconds(&census, &Pricing { figures: &figures, lookahead }, sequences, depth)
    }

    /// `demand`'s launch of `rows` rows at `depth`, with lookahead, as a load
    /// holding its experts whole prices it
    /// (`magnitude_cost_model::LaunchPrices::launch`).
    fn launch(demand: &DecodeDemand, costs: &ExecutorCosts, rows: u32, depth: u32) -> Result<f64, PriceError> {
        let census = demand.census();
        let figures = costs.figures(&[&census]);
        let prices = magnitude_cost_model::LaunchPrices::new(magnitude_cost_model::TargetCensus::of(census, None), figures, true);
        prices.launch(rows, depth).map(|forms| forms.resident)
    }

    /// The one call of `demand`'s prefill launch of `rows` rows of one
    /// sequence at `depth`, one launch of it, which is what a load's launch
    /// of those rows prices, composed.
    fn prefill(demand: &DecodeDemand, costs: &ExecutorCosts, rows: u64, depth: u32) -> CallSeconds {
        let census = demand.census();
        let figures = costs.figures(&[&census]);
        let [kind] = census.kinds() else {
            panic!("one call");
        };
        let call = seismic::measure::pricing::call_seconds(&kind.work, figures.rates.history_of(&kind.work).unwrap(), &figures.rates, rows, 1, depth, 1);
        let composed = figures.composition.clone().unwrap() * call.total() + figures.step_seconds(true) + census.boundaries() as f64 * figures.step_round_trip;
        close(launch(demand, costs, rows.try_into().unwrap(), depth).unwrap(), composed);
        call
    }

    /// One call of the CPU backend's dense expansion doing `work`.
    fn call(work: Work) -> DecodeDemand {
        let definition = crate::planning::fixtures::fixture_definition();
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let load = crate::ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            seismic::Layout::Rows8,
            crate::PlannedMethod::Plain,
            crate::planning::fixtures::LIMITS,
        )
        .unwrap();
        let declarations: Declarations = crate::assessment::demand::tests::declarations(
            &definition,
            &load,
            KvCodec::Dense,
            seismic::BackendName::Cpu,
        )
        .unwrap();
        let launches = declarations
            .launches(
                Kernel::of::<magnitude_kernels::dense_expand::Entry>(),
                seismic::OperandRecipe::Rounded,
                &[],
                &[],
                &[],
            )
            .unwrap();
        DecodeDemand {
            calls: vec![Call { launches, work, layer: None }],
            boundaries: 0,
        }
    }

    /// The format the fixture's costs measure at 20 TFLOP/s.
    fn format() -> seismic::measure::pricing::Format {
        format_of(Element::bf16())
    }

    /// A 4 GB, 8 GFLOP-per-row projection: bandwidth-bound at one row.
    fn projection() -> Work {
        Work {
            weights: vec![Bytes {
                format: format(),
                bytes: 4_000_000_000,
            }],
            row_flops: vec![Flops {
                format: format(),
                flops: 8_000_000_000,
            }],
            ..Work::default()
        }
    }

    fn only(step: LaunchSeconds) -> CallSeconds {
        let [call] = step.calls[..] else {
            panic!("one call");
        };
        assert_eq!(call.multiplicity, 1);
        call.seconds
    }

    /// `actual` is `expected` to rounding: a history line is derived from
    /// its ladder's rungs.
    fn close(actual: f64, expected: f64) {
        assert!(
            (actual / expected - 1.).abs() < 1e-12,
            "{actual} is not {expected}"
        );
    }

    #[test]
    fn depths_keep_the_limit_and_drop_what_exceeds_it() {
        assert_eq!(
            performance_depths(60_000, &[25_000, 50_000, 75_000, 50_000]),
            vec![25_000, 50_000, 60_000]
        );
        assert_eq!(performance_depths(1_000, &[]), vec![1_000]);
    }

    #[test]
    fn a_decode_of_one_row_is_bandwidth_bound() {
        let costs = crate::assessment::fixtures::costs();
        let demand = call(projection());
        let step = step(&demand, &costs, true, 1, 1_000).unwrap();
        let call = only(step.clone());
        assert!(!call.compute_bound());
        assert_eq!(call.bytes, 4e9 / 300e9);
        assert_eq!(call.compute, 8e9 * (DECODE_LINE.fixed + DECODE_LINE.row));
        assert_eq!(
            step.total(),
            1.2 * (4e9 / 300e9 + call.launches as f64 * 5e-6) + 50e-6
        );
    }

    /// On an executor of 2 TFLOP/s (a CPU's order) the projection that one
    /// row reads in 13 ms computes eight rows in 32 ms.
    #[test]
    fn a_decode_of_many_rows_is_compute_bound() {
        let mut costs = crate::assessment::fixtures::costs();
        costs.activation_matrix.as_mut().unwrap().flops.value = 2e12;
        let demand = call(projection());
                assert!(!only(step(&demand, &costs, true, 1, 1_000).unwrap()).compute_bound());
        // Eight sequences, a row each.
        let call = only(step(&demand, &costs, true, 8, 1_000).unwrap());
        assert!(call.compute_bound());
        assert_eq!(call.compute, 8.0 * 8e9 / 2e12);
        assert_eq!(call.total(), call.compute + call.launch);
        assert!(step(&demand, &costs, true, 9, 1_000).is_err());
        assert!(step(&demand, &costs, true, 0, 1_000).is_err());
    }

    /// A launch beyond the decode row classes is a prefill launch of one
    /// sequence: its weights read once and its arithmetic per row, each call
    /// one launch, and its history read once for the sequence while every
    /// row computes over it.
    #[test]
    fn a_launch_beyond_the_decode_rows_is_one_sequences_prefill() {
        let costs = crate::assessment::fixtures::costs();
        let projection = call(projection());
        let projected = prefill(&projection, &costs, 512, 1_000);
        assert_eq!(projected.bytes, 4e9 / 300e9);
        assert_eq!(projected.compute, 512.0 * 8e9 / 20e12);
        assert_eq!(projected.launches, 1);
        assert_eq!(launch(&projection, &costs, 8, 1_000).unwrap(), step(&projection, &costs, true, 8, 1_000).unwrap().total());
        let attention = call(Work {
            history: Some(HistoryRead { layout: Some(layout_of(KvCodec::Dense)), bytes_per_token: 1024, flops_per_token: 4096, window: None }),
            ..Work::default()
        });
        let attended = prefill(&attention, &costs, 512, 1_000);
        close(attended.bytes, 1_000.0 * 1024.0 / 200e9);
        assert_eq!(attended.compute, 512.0 * 1_000.0 * 4096.0 / 20e12);
    }

    /// Each byte class is priced at its own rate: weights and rows at the
    /// slowest weight rate of their storage's placement paths (upload
    /// storage at device storage's where its placement is the same route),
    /// history at its layout's rate beyond its fixed cost, scaled down by a
    /// slower route of its placement; and the step at the policy's step
    /// cost.
    #[test]
    fn each_byte_class_takes_its_own_rate() {
        let rate = |value| MeasuredCost {
            value,
            interval: None,
            samples: 1,
        };
        let mut costs = crate::assessment::fixtures::costs();
        let work = Work {
            weights: vec![Bytes {
                format: format(),
                bytes: 3_000_000,
            }],
            upload_row_bytes: 2_000_000,
            history: Some(HistoryRead {
                layout: Some(layout_of(KvCodec::Dense)),
                bytes_per_token: 1_000_000,
                flops_per_token: 0,
                window: None,
            }),
            ..Work::default()
        };
        let demand = call(work);
        let bytes =
            |costs: &ExecutorCosts| only(step(&demand, costs, true, 1, 1_000).unwrap());
        // Upload storage on device storage's route.
        let same = bytes(&costs);
        close(same.bytes, 5e6 / 300e9 + 1e9 / 200e9);
        close(same.fixed, 10e-6);
        // Upload storage on a route of its own (a discrete GPU's pinned host
        // RAM).
        let route = |route, value| RouteCosts {
            route,
            activation: Some(WeightRate {
                bytes_per_second: rate(value),
                features: 16384,
            }),
            formats: Vec::new(),
        };
        costs.routes.push(route(kind(1), 20e9));
        costs.placements.upload = Placement::along(kind(1));
        close(bytes(&costs).bytes, 3e6 / 300e9 + 2e6 / 20e9 + 1e9 / 200e9);
        // Device storage spanning a second, slower route (an integrated
        // GPU's own memory, then a heap of host RAM): weights at the slower
        // rate, history at its layout's rate scaled by the same ratio.
        let mut spanning = costs.clone();
        spanning
            .routes
            .push(route(kind(2), 100e9));
        spanning
            .placements
            .device
            .rest
            .push(kind(2));
        close(
            bytes(&spanning).bytes,
            3e6 / 100e9 + 2e6 / 20e9 + 1e9 / (200e9 * (100e9 / 300e9)),
        );
        let cost = |lookahead| step(&demand, &costs, lookahead, 1, 1).unwrap().step;
        assert_eq!((cost(true), cost(false)), (50e-6, 150e-6));
    }

    /// Rows choosing experts independently read the expected union of
    /// their choices; arithmetic follows the chosen experts of every row.
    #[test]
    fn routed_calls_read_the_union_of_their_rows_choices() {
        let costs = crate::assessment::fixtures::costs();
        let demand = call(Work {
            experts: Some(Experts {
                experts: 256,
                selected: 8,
                slab_bytes: vec![Bytes {
                    format: format(),
                    bytes: 3_000_000,
                }],
                slab_flops: vec![Flops {
                    format: format(),
                    flops: 6_000_000,
                }],
            }),
            ..Work::default()
        });
        let at = |rows| only(step(&demand, &costs, true, rows, 1).unwrap());
        assert_eq!(at(1).bytes, 8.0 * 3e6 / 300e9);
        let union = 256.0 * (1.0 - (1.0 - 8.0 / 256.0f64).powf(4.0));
        assert_eq!(at(4).bytes, union * 3e6 / 300e9);
        assert!(union < 4.0 * 8.0 && union > 8.0);
        assert_eq!(at(4).compute, 4.0 * 8.0 * 6e6 / 20e12);
    }

    /// History is read per sequence up to its window, at its layout's rate
    /// beyond its fixed cost, and attention's arithmetic per row and token
    /// read.
    #[test]
    fn windowed_history_stops_growing_at_its_window() {
        let costs = crate::assessment::fixtures::costs();
        let demand = call(Work {
            history: Some(HistoryRead {
                layout: Some(layout_of(KvCodec::Dense)),
                bytes_per_token: 1024,
                flops_per_token: 4096,
                window: Some(4_096),
            }),
            ..Work::default()
        });
                let at =
            |sequences, depth| only(step(&demand, &costs, true, sequences, depth).unwrap());
        close(at(1, 1_000).bytes, 1_000.0 * 1024.0 / 200e9);
        close(at(1, 1_000).fixed, 10e-6);
        assert_eq!(at(1, 4_096).bytes, at(1, 100_000).bytes);
        close(at(2, 100_000).bytes, 2.0 * 4_096.0 * 1024.0 / 200e9);
        assert_eq!(at(2, 100_000).compute, 2.0 * 4_096.0 * 4096.0 / 20e12);
        let census = demand.census();
        let figures = costs.figures(&[&census]);
        let estimates = estimate_decode(&census, &Pricing { figures: &figures, lookahead: true }, 1, &[1_000, 4_096, 100_000]).unwrap();
        assert!(estimates[0].tokens_per_second > estimates[1].tokens_per_second);
        assert_eq!(
            estimates[1].tokens_per_second,
            estimates[2].tokens_per_second
        );
    }

    /// Each format's weight bytes are priced at its own weight rate.
    #[test]
    fn each_format_streams_at_its_own_rate() {
        let mut costs = crate::assessment::fixtures::costs();
        for (format, rates) in &mut costs.first.formats {
            if *format == Element::f32() {
                rates
                    .streamed
                    .as_mut()
                    .expect("the fixture streams every format")
                    .bytes_per_second
                    .value = 100e9;
            }
        }
        let demand = call(Work {
            weights: vec![
                Bytes {
                    format: format(),
                    bytes: 3_000_000,
                },
                Bytes {
                    format: format_of(Element::f32()),
                    bytes: 2_000_000,
                },
            ],
            ..Work::default()
        });
        let call = only(step(&demand, &costs, true, 1, 1).unwrap());
        assert_eq!(call.bytes, 3e6 / 300e9 + 2e6 / 100e9);
    }

    /// A recurrent call's state is per sequence: N sequences move N states.
    #[test]
    fn recurrent_state_is_moved_per_sequence() {
        let costs = crate::assessment::fixtures::costs();
        let demand = call(Work {
            sequence_bytes: 3_000_000,
            ..Work::default()
        });
        let bytes =
            |sequences| only(step(&demand, &costs, true, sequences, 1).unwrap()).bytes;
        assert_eq!(bytes(4), 4.0 * 3e6 / 300e9);
        assert_eq!(bytes(1), 3e6 / 300e9);
    }

    /// A format whose pilot could not be read is priced at the activation
    /// dtype's rate along the same route and its throughput: the estimate
    /// is made, from the figures that are.
    #[test]
    fn a_format_without_a_pilot_is_priced_at_the_activation_dtypes_figures() {
        let mut costs = crate::assessment::fixtures::costs();
        costs.first.activation.bytes_per_second.value = 150e9;
        // Slower than DECODE_LINE at one row (~1.79 TFLOP/s) so the decode
        // step prices arithmetic at the activation dtype's matrix throughput.
        costs.activation_matrix.as_mut().unwrap().flops.value = 1e12;
        costs.first
            .formats
            .retain(|(format, _)| *format != Element::f32());
        costs
            .matrices
            .retain(|(format, _)| *format != Element::f32());
        let demand = call(Work {
            weights: vec![Bytes {
                format: format_of(Element::f32()),
                bytes: 3_000_000,
            }],
            row_flops: vec![Flops {
                format: format_of(Element::f32()),
                flops: 4_000_000,
            }],
            ..Work::default()
        });
        let call = only(step(&demand, &costs, true, 1, 1).unwrap());
        assert_eq!(call.bytes, 3e6 / 150e9);
        assert_eq!(call.compute, 4e6 / 1e12);
    }

    /// Until a job reads a format's streaming rate and throughput, its
    /// pilot prices them: its bytes at the slower of its pilot's rate and
    /// the activation dtype's streaming one, its arithmetic at its pilot's
    /// one-row rate.
    #[test]
    fn a_format_read_only_by_its_pilot_is_priced_from_it() {
        let mut costs = crate::assessment::fixtures::costs();
        costs.first.activation.bytes_per_second.value = 150e9;
        costs
            .matrices
            .retain(|(format, _)| *format != Element::f32());
        let demand = call(Work {
            weights: vec![Bytes {
                format: format_of(Element::f32()),
                bytes: 3_000_000,
            }],
            row_flops: vec![Flops {
                format: format_of(Element::f32()),
                flops: 4_000_000,
            }],
            ..Work::default()
        });
        let priced = |pilot: f64, costs: &mut ExecutorCosts| {
            let (_, rates) = costs.first
                .formats
                .iter_mut()
                .find(|(format, _)| *format == Element::f32())
                .unwrap();
            rates.streamed = None;
            rates.pilot.bytes_per_second.value = pilot;
            only(step(&demand, costs, true, 1, 1).unwrap())
        };
        // A pilot faster than memory streams: memory bounds the rate.
        assert_eq!(priced(400e9, &mut costs).bytes, 3e6 / 150e9);
        // A pilot slower than memory: its arithmetic bounds it, and its
        // rate over f32's 4-byte weights is half a FLOP per byte.
        let slow = priced(100e9, &mut costs);
        assert_eq!(slow.bytes, 3e6 / 100e9);
        assert_eq!(slow.compute, 4e6 / 50e9);
    }

    /// Costs holding the probe alone of the figures others price from price
    /// every demand from it: weights and rows at its rate, arithmetic at its
    /// one-row FLOP rate, and history at its rate with no fixed cost.
    #[test]
    fn the_probe_alone_prices_every_demand() {
        let mut costs = crate::assessment::fixtures::costs();
        costs.first.activation.bytes_per_second.value = 150e9;
        costs.first.formats.clear();
        costs.matrices.clear();
        costs.activation_matrix = None;
        costs.dense = HistoryCosts::new(KvCodec::Dense, Vec::new()).unwrap();
        let demand = call(Work {
            weights: vec![Bytes {
                format: format_of(Element::f32()),
                bytes: 3_000_000,
            }],
            row_flops: vec![Flops {
                format: format_of(Element::f32()),
                flops: 4_000_000,
            }],
            history: Some(HistoryRead {
                layout: Some(layout_of(KvCodec::Dense)),
                bytes_per_token: 1_000,
                flops_per_token: 0,
                window: None,
            }),
            ..Work::default()
        });
        let step = step(&demand, &costs, true, 1, 1_000).unwrap();
        let call = only(step);
        assert_eq!(call.fixed, 0.);
        close(call.bytes, (3e6 + 1e6) / 150e9);
        // bf16's 2-byte weights: a FLOP per byte at one row, two per
        // weight.
        close(call.compute, 4e6 / 150e9);
    }

    /// History no native attention entry reads is no load's: its demand
    /// cannot be priced.
    #[test]
    fn history_no_entry_reads_is_a_demand_failure() {
        let costs = crate::assessment::fixtures::costs();
        let rotated = Work {
            history: Some(HistoryRead {
                layout: Some(layout_of(KvCodec::RotatedK4V4)),
                bytes_per_token: 1,
                flops_per_token: 0,
                window: None,
            }),
            ..Work::default()
        };
        assert!(matches!(
            step(&call(rotated), &costs, true, 1, 1),
            Err(PriceError::History(_))
        ));
    }
}
