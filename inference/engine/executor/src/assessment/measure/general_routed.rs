//! Measurement of the general routed feed-forward's own classes
//! (`operators::routed`): selection (`routed_select`, linear in the router it
//! reads), the decode expansions (`routed_gate_up`, `routed_up`) and down
//! projections (`routed_down`) as weight-streaming classes, and the up-only
//! dense expansion of a shared expert (`dense_up`). Its gated shared
//! expansion, shared down and latent projections are `dense_expand`,
//! `dense_output` and `project_rows`, measured as those classes.

use super::{
    bytes, failed, launch_point, multiple, size_point, Launch, MeasuredPoint, Runner, Step,
    Timed, ROUTED_SELECTED, ROW_ALIGNMENT,
};
use magnitude_kernels::{dense_up, routed_down, routed_gate_up, routed_select, routed_up};
use seismic::Element;

/// The decode expert entries' dimensions: 8 selected experts of `features`
/// rows over a `hidden`-wide input.
fn expert_dimensions(hidden: u64, features: u64) -> [(&'static str, u64); 5] {
    [
        ("M", 1),
        ("H", hidden),
        ("E", ROUTED_SELECTED),
        ("K", ROUTED_SELECTED),
        ("F", features),
    ]
}

impl Runner<'_, '_, '_> {
    /// Router logits, biased top-k and weights: 8 of `experts` over a
    /// `hidden`-wide row. Its bytes are the router's.
    pub(super) fn routed_select(
        &self,
        norm: Element,
        router: Element,
        activation: Element,
        hidden: u64,
        experts: u64,
    ) -> Step<MeasuredPoint> {
        let dimensions = [("M", 1), ("H", hidden), ("E", experts), ("K", ROUTED_SELECTED)];
        let device = self.device();
        let kernels = self.form::<routed_select::Entry>(
            &[norm, router, activation],
            &dimensions,
            move |specialization| {
                routed_select::native_for_device_with(
                    device,
                    routed_select::Elements {
                        NW: norm,
                        RNW: norm,
                        RW: router,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let router_bytes = bytes(router, &[experts, hidden])?;
        let launches = self.copies(router_bytes);
        let routers = self.views(router, hidden, experts, launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let norm = self.zeros(norm, &[hidden])?;
        let bias = self.zeros(Element::f32(), &[experts])?;
        let scales = self.zeros(Element::f32(), &[experts])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let norm = timed.bound(&norm)?;
            let bias = timed.bound(&bias)?;
            let scales = timed.bound(&scales)?;
            for router in &routers {
                let router = timed.bound(router)?;
                let mut routes = timed
                    .graph
                    .local_for(kernel, "routes", &dimensions)
                    .map_err(failed)?;
                let mut weights = timed
                    .graph
                    .local_for(kernel, "weights", &dimensions)
                    .map_err(failed)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        routed_select::WorkflowArgs {
                            residual: residual.tensor().into(),
                            norm: norm.tensor().into(),
                            router_norm: norm.tensor().into(),
                            router: router.tensor().into(),
                            bias: bias.tensor().into(),
                            expert_scale: scales.tensor().into(),
                            routes: routes.tensor_mut().into(),
                            weights: weights.tensor_mut().into(),
                            epsilon: 1e-6,
                            score: 1,
                            normalization: 1,
                            normalization_epsilon: 0.0,
                            scale: 1.0,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
                timed.export(weights.tensor())?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(router_bytes, samples))
    }

    /// Decode expansion of 8 selected gated experts, gate and up each: the
    /// launch's rows in 16 equal parts. Only the selected experts are
    /// allocated: routes name each once, as a decode row's choices do.
    pub(super) fn routed_gate_up(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let experts = ROUTED_SELECTED;
        let features = multiple(at.rows / (2 * experts), ROW_ALIGNMENT);
        let hidden = at.reduction;
        let point_bytes = bytes(weight, &[experts, features, hidden])? * 2;
        let device = self.device();
        let kernels = self.form::<routed_gate_up::Entry>(
            &[weight, activation],
            &expert_dimensions(hidden, features),
            move |specialization| {
                routed_gate_up::native_for_device_with(
                    device,
                    routed_gate_up::Elements {
                        A: activation,
                        EGW: weight,
                        EUW: weight,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let gates = self.shaped(weight, &[experts, features, hidden], launches)?;
        let ups = self.shaped(weight, &[experts, features, hidden], launches)?;
        let normalized = self.zeros(activation, &[1, hidden])?;
        let routes = self.i32s(
            &[1, ROUTED_SELECTED],
            &(0..ROUTED_SELECTED as i32).collect::<Vec<_>>(),
        )?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let normalized = timed.bound(&normalized)?;
            let routes = timed.bound(&routes)?;
            for (gate, up) in gates.iter().zip(&ups) {
                let gate = timed.bound(gate)?;
                let up = timed.bound(up)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        routed_gate_up::WorkflowArgs {
                            normalized: normalized.tensor().into(),
                            routes: routes.tensor().into(),
                            expert_gate: gate.tensor().into(),
                            expert_up: up.tensor().into(),
                            activation: 0,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, 2 * experts * features, weight, point_bytes, samples))
    }

    /// Decode expansion of 8 selected up-only experts: the launch's rows in
    /// 8 equal parts.
    pub(super) fn routed_up(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let experts = ROUTED_SELECTED;
        let features = multiple(at.rows / experts, ROW_ALIGNMENT);
        let hidden = at.reduction;
        let point_bytes = bytes(weight, &[experts, features, hidden])?;
        let device = self.device();
        let kernels = self.form::<routed_up::Entry>(
            &[weight, activation],
            &expert_dimensions(hidden, features),
            move |specialization| {
                routed_up::native_for_device_with(
                    device,
                    routed_up::Elements {
                        A: activation,
                        EUW: weight,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let ups = self.shaped(weight, &[experts, features, hidden], launches)?;
        let normalized = self.zeros(activation, &[1, hidden])?;
        let routes = self.i32s(
            &[1, ROUTED_SELECTED],
            &(0..ROUTED_SELECTED as i32).collect::<Vec<_>>(),
        )?;
        let up_scale = self.zeros(Element::f32(), &[experts])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let normalized = timed.bound(&normalized)?;
            let routes = timed.bound(&routes)?;
            let up_scale = timed.bound(&up_scale)?;
            for up in &ups {
                let up = timed.bound(up)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        routed_up::WorkflowArgs {
                            normalized: normalized.tensor().into(),
                            routes: routes.tensor().into(),
                            expert_up: up.tensor().into(),
                            up_scale: up_scale.tensor().into(),
                            activation: 2,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, experts * features, weight, point_bytes, samples))
    }

    /// Decode down projection of 8 selected experts over the launch's
    /// reduction onto an F32 base: the launch's rows in 8 equal parts.
    pub(super) fn routed_down(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let experts = ROUTED_SELECTED;
        let hidden = multiple(at.rows / experts, ROW_ALIGNMENT);
        let features = at.reduction;
        let point_bytes = bytes(weight, &[experts, hidden, features])?;
        let device = self.device();
        let kernels = self.form::<routed_down::Entry>(
            &[weight, activation],
            &[
                ("M", 1),
                ("H", hidden),
                ("E", experts),
                ("K", ROUTED_SELECTED),
                ("F", features),
            ],
            move |specialization| {
                routed_down::native_for_device_with(
                    device,
                    routed_down::Elements {
                        A: activation,
                        EDW: weight,
                        R: Element::f32(),
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let downs = self.shaped(weight, &[experts, hidden, features], launches)?;
        let base = self.zeros(Element::f32(), &[1, hidden])?;
        let product = self.zeros(activation, &[1, ROUTED_SELECTED, features])?;
        let routes = self.i32s(
            &[1, ROUTED_SELECTED],
            &(0..ROUTED_SELECTED as i32).collect::<Vec<_>>(),
        )?;
        let weights = self.zeros(Element::f32(), &[1, ROUTED_SELECTED])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let base = timed.bound(&base)?;
            let product = timed.bound(&product)?;
            let routes = timed.bound(&routes)?;
            let weights = timed.bound(&weights)?;
            for down in &downs {
                let down = timed.bound(down)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        routed_down::WorkflowArgs {
                            base: base.tensor().into(),
                            product: product.tensor().into(),
                            routes: routes.tensor().into(),
                            weights: weights.tensor().into(),
                            expert_down: down.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, experts * hidden, weight, point_bytes, samples))
    }

    /// Up-only projection into the launch's rows.
    pub(super) fn dense_up(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (features, hidden) = (at.rows, at.reduction);
        let point_bytes = bytes(weight, &[features, hidden])?;
        let device = self.device();
        let kernels = self.form::<dense_up::Entry>(
            &[norm, weight, activation],
            &[("M", 1), ("O", 1), ("H", hidden), ("F", features), ("US", 0)],
            move |specialization| {
                dense_up::native_for_device_with(
                    device,
                    dense_up::Elements {
                        NW: norm,
                        UW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let ups = self.views(weight, hidden, features, launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let norm = self.zeros(norm, &[hidden])?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let scale = self.zeros(Element::f32(), &[0])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let norm = timed.bound(&norm)?;
            let out_rows = timed.bound(&out_rows)?;
            let scale = timed.bound(&scale)?;
            for up in &ups {
                let up = timed.bound(up)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        dense_up::WorkflowArgs {
                            residual: residual.tensor().into(),
                            norm: norm.tensor().into(),
                            up_weight: up.tensor().into(),
                            out_rows: out_rows.tensor().into(),
                            eps: 1e-5,
                            activation: 2,
                            up_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, features, weight, point_bytes, samples))
    }
}
