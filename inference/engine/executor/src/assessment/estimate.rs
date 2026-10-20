//! Analytical decode-speed estimation (spec §2.14): what a model's decode
//! achieves on this machine, from its per-call demand and what the
//! engine's kernels measurably achieve on its executor
//! ([`ExecutorCosts`]).
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
//! Where it can be off, and in which direction: about ±15% typically, the
//! spread of the composition ratio across models once format, layout,
//! launch and composition are measured (spec §2.14, Expected error). The
//! remaining sources: a matrix's shape at one row (the launch-plus-bytes
//! rule's median 7–18%, worst for weights under 2 MB), MoE expert
//! projections at few rows, recurrent and state-space kernels (priced by
//! their bytes only, their arithmetic is not derived), attention's
//! arithmetic at many rows (priced at the bf16 matrix throughput, not
//! measured on the attention entry), contention between devices, compute
//! overlapping transfers, and sustained throttling. A format or layout the
//! executor's costs do not hold leaves the estimate unavailable
//! ([`AssessmentError::Unmeasured`]); no figure is borrowed from another.

use super::demand::{Bytes, Call, DecodeDemand, Flops};
use super::executor_costs::ExecutorCosts;
use super::AssessmentError;
use seismic::{Element, StorageKind};

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

/// What a step on one executor is priced with: its kernels' costs and the
/// policy its decode runs under, steps queued ahead (lookahead) or each
/// submitted and waited for.
#[derive(Clone, Copy, Debug)]
pub struct Pricing<'a> {
    pub costs: &'a ExecutorCosts,
    pub lookahead: bool,
}

impl Pricing<'_> {
    /// The seconds of `bytes`, each format's at its weight rate in device
    /// storage, `scale` times over.
    fn weights(&self, bytes: &[Bytes], scale: f64) -> Result<f64, AssessmentError> {
        bytes.iter().try_fold(0.0, |seconds, bytes| {
            Ok(seconds
                + scale * bytes.bytes as f64
                    / self.costs.weight_rate(StorageKind::Device, bytes.format)?)
        })
    }

    /// Activation-dtype bytes in `storage`: the slowest rate of its
    /// placement's paths.
    pub fn activations(&self, storage: StorageKind) -> Result<f64, AssessmentError> {
        self.costs.weight_rate(storage, Element::bf16())
    }

    /// The seconds of `flops` over their format's throughput.
    fn compute(&self, flops: &[Flops], scale: f64) -> Result<f64, AssessmentError> {
        flops.iter().try_fold(0.0, |seconds, flops| {
            Ok(seconds + scale * flops.flops as f64 / self.costs.matrix_flops(flops.format)?)
        })
    }
}

/// One call's time and its parts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallSeconds {
    /// Its bytes' time.
    pub bytes: f64,
    /// Its arithmetic's time.
    pub compute: f64,
    /// Its attention's fixed cost per call.
    pub fixed: f64,
    /// The native launches it dispatches.
    pub launches: u64,
    /// Its launches' time.
    pub launch: f64,
}

impl CallSeconds {
    /// The call's time: the slower of its bytes and its arithmetic, plus
    /// its fixed cost and its launches.
    pub fn total(&self) -> f64 {
        self.bytes.max(self.compute) + self.fixed + self.launch
    }

    /// Whether its arithmetic, not its bytes, bounds it.
    pub fn compute_bound(&self) -> bool {
        self.compute > self.bytes
    }
}

/// One decode step's time at one depth, call by call.
#[derive(Clone, Debug, PartialEq)]
pub struct StepSeconds {
    /// Each call's time, alone.
    pub calls: Vec<CallSeconds>,
    /// The ratio the calls' sum is scaled by, composed in a step.
    pub composition: f64,
    /// The host's step cost under the decode's lookahead policy.
    pub step: f64,
}

impl StepSeconds {
    /// The step's time: its calls composed, plus the step cost.
    pub fn total(&self) -> f64 {
        self.composition * self.calls.iter().map(CallSeconds::total).sum::<f64>() + self.step
    }
}

/// The expected slabs `rows` rows read of `experts` slabs, each row
/// choosing `selected` of them: every slab not chosen by any row is not
/// read, E · (1 − (1 − k/E)^rows).
fn expert_union(experts: u64, selected: u64, rows: u64) -> f64 {
    let experts = experts as f64;
    let missed = 1.0 - selected as f64 / experts;
    experts * (1.0 - missed.powf(rows as f64))
}

/// One call's time at `sequences` sequences, one row each.
fn call_seconds(
    call: &Call,
    pricing: &Pricing<'_>,
    sequences: u64,
    depth: u32,
) -> Result<CallSeconds, AssessmentError> {
    let work = &call.work;
    let rows = sequences;
    let (rows_f, sequences) = (rows as f64, sequences as f64);
    let mut bytes = pricing.weights(&work.weights, 1.0)?;
    if let Some(experts) = &work.experts {
        let union = expert_union(experts.experts, experts.selected, rows);
        bytes += pricing.weights(&experts.slab_bytes, union)?;
    }
    let activations = rows_f * work.row_bytes as f64 + sequences * work.sequence_bytes as f64;
    if activations > 0.0 {
        bytes += activations / pricing.activations(StorageKind::Device)?;
    }
    let upload = rows_f * work.upload_row_bytes as f64;
    if upload > 0.0 {
        bytes += upload / pricing.activations(StorageKind::Upload)?;
    }
    let mut compute = pricing.compute(&work.row_flops, rows_f)?;
    if let Some(experts) = &work.experts {
        compute += pricing.compute(&experts.slab_flops, rows_f * experts.selected as f64)?;
    }
    let mut fixed = 0.0;
    if let Some(history) = &work.history {
        let tokens = history.tokens_at(depth) as f64;
        let read = sequences * history.bytes_per_token as f64 * tokens;
        // History is in device storage: its layout's rate, over the
        // placement's paths; history no attention entry reads, at the
        // activation dtype's rate there.
        bytes += match history.layout {
            Some(layout) => {
                fixed = pricing.costs.history(layout)?.fixed_seconds.value;
                read / pricing.costs.history_rate(layout)?
            }
            None => read / pricing.activations(StorageKind::Device)?,
        };
        compute += rows_f * history.flops_per_token as f64 * tokens
            / pricing.costs.matrix_flops(Element::bf16())?;
    }
    let launches = call.launches.at(rows)?;
    Ok(CallSeconds {
        bytes,
        compute,
        fixed,
        launches,
        launch: launches as f64 * pricing.costs.launch_seconds.value,
    })
}

/// The most rows the demand's calls describe: the decode row classes,
/// beyond which attention and routed calls run their prefill forms
/// (`operators::attention::graph`, `operators::routed::fused_graph`).
fn decode_rows() -> u64 {
    crate::operators::attention::graph::DECODE_ROWS
        .min(crate::operators::routed::fused_graph::DECODE_ROWS)
}

/// One decode step's time at `depth` over `sequences` sequences.
pub fn step_seconds(
    demand: &DecodeDemand,
    pricing: &Pricing<'_>,
    sequences: u64,
    depth: u32,
) -> Result<StepSeconds, AssessmentError> {
    if sequences == 0 || sequences > decode_rows() {
        return Err(AssessmentError::Demand(format!(
            "a decode step of {sequences} rows is not one the decode demand describes \
             (1 to {} rows; beyond, prefill forms run)",
            decode_rows()
        )));
    }
    Ok(StepSeconds {
        calls: demand
            .calls
            .iter()
            .map(|call| call_seconds(call, pricing, sequences, depth))
            .collect::<Result<_, _>>()?,
        composition: pricing.costs.composition.value,
        step: pricing.costs.step_seconds(pricing.lookahead),
    })
}

/// One estimate per depth: tokens per second per conversation.
pub fn estimate_decode(
    demand: &DecodeDemand,
    pricing: &Pricing<'_>,
    sequences: u64,
    depths: &[u32],
) -> Result<Vec<PerformanceEstimate>, AssessmentError> {
    depths
        .iter()
        .map(|&depth| {
            Ok(PerformanceEstimate {
                context_tokens: depth,
                tokens_per_second: 1.0 / step_seconds(demand, pricing, sequences, depth)?.total(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::demand::{Experts, HistoryRead, Work};
    use crate::assessment::executor_costs::{MeasuredCost, Placement, WeightRate};
    use crate::assessment::launches::{Declarations, Kernel};
    use magnitude_state::KvCodec;
    use seismic::AccessRoute;

    /// The synthetic costs ([`crate::assessment::fixtures::costs`]), with
    /// lookahead.
    fn pricing(costs: &ExecutorCosts) -> Pricing<'_> {
        Pricing {
            costs,
            lookahead: true,
        }
    }

    /// One call of the CPU backend's dense expansion doing `work`.
    fn call(work: Work) -> DecodeDemand {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = crate::ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            seismic::Layout::Rows8,
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
            .launches(Kernel::of::<magnitude_kernels::dense_expand::Entry>(), &[])
            .unwrap();
        DecodeDemand {
            calls: vec![Call { launches, work }],
        }
    }

    /// The format the fixture's costs measure at 20 TFLOP/s.
    fn format() -> Element {
        Element::bf16()
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

    fn only(step: StepSeconds) -> CallSeconds {
        let [call] = step.calls[..] else {
            panic!("one call");
        };
        call
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
        let step = step_seconds(&demand, &pricing(&costs), 1, 1_000).unwrap();
        let call = only(step.clone());
        assert!(!call.compute_bound());
        assert_eq!(call.bytes, 4e9 / 300e9);
        assert_eq!(call.compute, 8e9 / 20e12);
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
        for matrix in &mut costs.matrices {
            matrix.flops.value = 2e12;
        }
        let demand = call(projection());
        let pricing = pricing(&costs);
        assert!(!only(step_seconds(&demand, &pricing, 1, 1_000).unwrap()).compute_bound());
        // Eight sequences, a row each.
        let call = only(step_seconds(&demand, &pricing, 8, 1_000).unwrap());
        assert!(call.compute_bound());
        assert_eq!(call.compute, 8.0 * 8e9 / 2e12);
        assert_eq!(call.total(), call.compute + call.launch);
        assert!(step_seconds(&demand, &pricing, 9, 1_000).is_err());
        assert!(step_seconds(&demand, &pricing, 0, 1_000).is_err());
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
                layout: Some(KvCodec::Dense),
                bytes_per_token: 1_000_000,
                flops_per_token: 0,
                window: None,
            }),
            ..Work::default()
        };
        let demand = call(work);
        let bytes =
            |costs: &ExecutorCosts| only(step_seconds(&demand, &pricing(costs), 1, 1_000).unwrap());
        // Upload storage on device storage's route.
        let same = bytes(&costs);
        assert_eq!(same.bytes, 5e6 / 300e9 + 1e9 / 200e9);
        assert_eq!(same.fixed, 10e-6);
        // Upload storage on a route of its own (a discrete GPU's pinned host
        // RAM).
        costs.weight_rates.push(WeightRate {
            route: AccessRoute::Upload,
            format: Element::bf16(),
            bytes_per_second: rate(20e9),
            features: 16384,
        });
        costs.placements.upload = Placement::along(AccessRoute::Upload);
        assert_eq!(bytes(&costs).bytes, 3e6 / 300e9 + 2e6 / 20e9 + 1e9 / 200e9);
        // An upload route left unmeasured leaves the estimate unavailable,
        // never priced at device storage's rate.
        let mut unmeasured = costs.clone();
        unmeasured
            .weight_rates
            .retain(|rate| rate.route != AccessRoute::Upload);
        assert!(matches!(
            step_seconds(&demand, &pricing(&unmeasured), 1, 1_000),
            Err(AssessmentError::Unmeasured(_))
        ));
        // Device storage spanning a second, slower route (an integrated
        // GPU's own memory, then a heap of host RAM): weights at the slower
        // rate, history at its layout's rate scaled by the same ratio.
        let mut spanning = costs.clone();
        spanning.weight_rates.push(WeightRate {
            route: AccessRoute::VulkanHeap(1),
            format: Element::bf16(),
            bytes_per_second: rate(100e9),
            features: 16384,
        });
        spanning
            .placements
            .device
            .rest
            .push(AccessRoute::VulkanHeap(1));
        assert_eq!(
            bytes(&spanning).bytes,
            3e6 / 100e9 + 2e6 / 20e9 + 1e9 / (200e9 * (100e9 / 300e9))
        );
        let round_trip = Pricing {
            costs: &costs,
            lookahead: false,
        };
        let step = |pricing| step_seconds(&demand, pricing, 1, 1).unwrap().step;
        assert_eq!((step(&pricing(&costs)), step(&round_trip)), (50e-6, 150e-6));
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
        let at = |rows| only(step_seconds(&demand, &pricing(&costs), rows, 1).unwrap());
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
                layout: Some(KvCodec::Dense),
                bytes_per_token: 1024,
                flops_per_token: 4096,
                window: Some(4_096),
            }),
            ..Work::default()
        });
        let pricing = pricing(&costs);
        let at =
            |sequences, depth| only(step_seconds(&demand, &pricing, sequences, depth).unwrap());
        assert_eq!(at(1, 1_000).bytes, 1_000.0 * 1024.0 / 200e9);
        assert_eq!(at(1, 1_000).fixed, 10e-6);
        assert_eq!(at(1, 4_096).bytes, at(1, 100_000).bytes);
        assert_eq!(at(2, 100_000).bytes, 2.0 * 4_096.0 * 1024.0 / 200e9);
        assert_eq!(at(2, 100_000).compute, 2.0 * 4_096.0 * 4096.0 / 20e12);
        let estimates = estimate_decode(&demand, &pricing, 1, &[1_000, 4_096, 100_000]).unwrap();
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
        for rate in &mut costs.weight_rates {
            if rate.format == Element::f32() {
                rate.bytes_per_second.value = 100e9;
            }
        }
        let demand = call(Work {
            weights: vec![
                Bytes {
                    format: format(),
                    bytes: 3_000_000,
                },
                Bytes {
                    format: Element::f32(),
                    bytes: 2_000_000,
                },
            ],
            ..Work::default()
        });
        let call = only(step_seconds(&demand, &pricing(&costs), 1, 1).unwrap());
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
            |sequences| only(step_seconds(&demand, &pricing(&costs), sequences, 1).unwrap()).bytes;
        assert_eq!(bytes(4), 4.0 * 3e6 / 300e9);
        assert_eq!(bytes(1), 3e6 / 300e9);
    }

    /// A format or layout the costs do not hold leaves the estimate
    /// unavailable, never priced at another's figure.
    #[test]
    fn an_unmeasured_format_or_layout_is_unavailable() {
        let costs = crate::assessment::fixtures::costs();
        let pricing = pricing(&costs);
        let unmeasured = |work| {
            matches!(
                step_seconds(&call(work), &pricing, 1, 1),
                Err(AssessmentError::Unmeasured(_))
            )
        };
        assert!(unmeasured(Work {
            row_flops: vec![Flops {
                format: Element::f16(),
                flops: 1,
            }],
            ..Work::default()
        }));
        assert!(unmeasured(Work {
            weights: vec![Bytes {
                format: Element::f16(),
                bytes: 1,
            }],
            ..Work::default()
        }));
        let mut dense_only = crate::assessment::fixtures::costs();
        dense_only
            .histories
            .retain(|history| history.layout == KvCodec::Dense);
        let affine = Work {
            history: Some(HistoryRead {
                layout: Some(KvCodec::AffineK8V4),
                bytes_per_token: 1,
                flops_per_token: 0,
                window: None,
            }),
            ..Work::default()
        };
        assert!(matches!(
            step_seconds(&call(affine), &self::pricing(&dense_only), 1, 1),
            Err(AssessmentError::Unmeasured(_))
        ));
    }
}
