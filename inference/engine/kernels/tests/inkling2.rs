//! Numerical parity tests for Inkling 2 kernels:
//! - `block_attn_res_mix` (Block Attention Residuals)
//! - `rms_concat` & `message_gate_mix` (Cross-Pass Message Controllers)
//! - `causal_short_conv` (static + per-head dynamic taps, decode/verify/prefill)
//! - `head_l2_clip` (optional scale, L2 clip, and global `tau` log-scaling)
//! - `attention_rel_bias` (in-kernel relative-position bias, static & dynamic
//!   per-token sinks, output gate, decode/verify/prefill + KV cache append)
//! - SITU softcapped activation in `dense_expand` (`activation = 3`)

mod common;
mod declared;
mod versions_common;

use magnitude_kernels::{
    attention_rel_bias, block_attn_res_mix, causal_short_conv, dense_expand, head_l2_clip,
    message_gate_mix, rms_concat, rnn_markov_gate, situ_limit_word,
};
use seismic::{BackendName, Charges, Device, Element, NativeSpecialization, Tensor};
use versions_common::{slab_arenas, Arena};

fn devices() -> Vec<Device> {
    use BackendName::{Cpu, Cuda, Metal, Vulkan};
    seismic_native_tests::first_and_cpu(&[Cuda, Metal], &[Cuda, Metal, Vulkan, Cpu])
}

fn pattern(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn f32_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let charges = Charges::tally(device);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_host(&charges, Element::f32(), shape, &bytes).unwrap()
}

fn i32_tensor(device: &Device, shape: &[u64], values: &[i32]) -> Tensor {
    let charges = Charges::tally(device);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_host(&charges, Element::i32(), shape, &bytes).unwrap()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let bound = atol + rtol * e.abs();
        assert!(
            diff <= bound,
            "{label}[{i}]: actual={a}, expected={e}, diff={diff}, bound={bound}"
        );
    }
}

#[test]
fn block_attn_res_mix_parity() {
    for device in devices() {
        let charges = Charges::tally(&device);
        let kernel = declared::prepared(
            &device,
            &block_attn_res_mix::native_entry(),
            &NativeSpecialization::new(),
        )
        .unwrap();
        for &(s, m, h, q, norm_v) in &[
            (5usize, 1usize, 128usize, 1usize, 1i32),
            (4, 3, 256, 3, 1),
            (3, 2, 128, 2, 0),
        ] {
            let eps = 1e-6f32;
            let scale = 1.0f32 / (h as f32).sqrt();
            let sources = pattern(s * m * h, 11, 2.0);
            let queries = pattern(q * h, 17, 1.5);

            // Reference in F64
            let mut expected = vec![0.0f32; q * m * h];
            for qi in 0..q {
                for row in 0..m {
                    let mut inv = vec![0.0f64; s];
                    let mut logits = vec![0.0f64; s];
                    let mut max_l = f64::NEG_INFINITY;
                    for si in 0..s {
                        let base = (si * m + row) * h;
                        let sq: f64 = sources[base..base + h].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
                        let dot: f64 = sources[base..base + h]
                            .iter()
                            .zip(&queries[qi * h..(qi + 1) * h])
                            .map(|(&v, &qv)| f64::from(v) * f64::from(qv))
                            .sum();
                        inv[si] = 1.0 / (sq / h as f64 + f64::from(eps)).sqrt();
                        logits[si] = dot * inv[si] * f64::from(scale);
                        if logits[si] > max_l {
                            max_l = logits[si];
                        }
                    }
                    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
                    let sum_e: f64 = exps.iter().sum();
                    for hi in 0..h {
                        let mut acc = 0.0f64;
                        for si in 0..s {
                            let w = exps[si] / sum_e;
                            let c = if norm_v != 0 { w * inv[si] } else { w };
                            acc += c * f64::from(sources[(si * m + row) * h + hi]);
                        }
                        expected[(qi * m + row) * h + hi] = acc as f32;
                    }
                }
            }

            let src_t = f32_tensor(&device, &[s as u64, m as u64, h as u64], &sources);
            let q_t = f32_tensor(&device, &[q as u64, h as u64], &queries);
            let out = kernel
                .call(&charges, block_attn_res_mix::Args {
                    sources: &src_t,
                    queries: &q_t,
                    epsilon: eps,
                    scale,
                    normalize_values: norm_v,
                })
                .unwrap()
                .value;
            assert_close(
                &format!("block_attn_res_mix({:?}, S={s}, M={m}, H={h}, Q={q}, norm_v={norm_v})", device.backend()),
                &read_f32(&out),
                &expected,
                1e-5,
                1e-5,
            );
        }
    }
}

#[test]
fn message_controller_ops_parity() {
    for device in devices() {
        let charges = Charges::tally(&device);
        let concat_k = declared::prepared(
            &device,
            &rms_concat::native_entry_with(rms_concat::Elements { A: Element::f32() }),
            &NativeSpecialization::new(),
        )
        .unwrap();
        let mix_k = declared::prepared(
            &device,
            &message_gate_mix::native_entry(),
            &NativeSpecialization::new(),
        )
        .unwrap();

        let (m, h) = (3usize, 128usize);
        let eps = 1e-6f32;
        let dest = pattern(m * h, 21, 3.0);
        let msg = pattern(m * h, 23, 2.5);
        let gain = pattern(m * h, 29, 1.5);
        let forget = pattern(m * h, 31, 1.5);

        let mut exp_concat = vec![0.0f32; m * 2 * h];
        let mut exp_mix = vec![0.0f32; m * h];
        for row in 0..m {
            let sq_a: f64 = dest[row * h..(row + 1) * h].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let sq_b: f64 = msg[row * h..(row + 1) * h].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let inv_a = 1.0 / (sq_a / h as f64 + f64::from(eps)).sqrt();
            let inv_b = 1.0 / (sq_b / h as f64 + f64::from(eps)).sqrt();
            for i in 0..h {
                exp_concat[row * 2 * h + i] = (f64::from(dest[row * h + i]) * inv_a) as f32;
                exp_concat[row * 2 * h + h + i] = (f64::from(msg[row * h + i]) * inv_b) as f32;
                let tf = f64::from(forget[row * h + i]).tanh();
                let tg = f64::from(gain[row * h + i]).tanh();
                exp_mix[row * h + i] =
                    ((1.0 - tf) * f64::from(dest[row * h + i]) + tg * f64::from(msg[row * h + i])) as f32;
            }
        }

        let dest_t = f32_tensor(&device, &[m as u64, h as u64], &dest);
        let msg_t = f32_tensor(&device, &[m as u64, h as u64], &msg);
        let gain_t = f32_tensor(&device, &[m as u64, h as u64], &gain);
        let forget_t = f32_tensor(&device, &[m as u64, h as u64], &forget);

        let cat_out = concat_k
            .call(&charges, rms_concat::Args { a: &dest_t, b: &msg_t, epsilon: eps })
            .unwrap()
            .value;
        assert_close("rms_concat", &read_f32(&cat_out), &exp_concat, 1e-5, 1e-5);

        let mix_out = mix_k
            .call(&charges, message_gate_mix::Args {
                dest: &dest_t,
                msg: &msg_t,
                gain: &gain_t,
                forget: &forget_t,
            })
            .unwrap()
            .value;
        assert_close("message_gate_mix", &read_f32(&mix_out), &exp_mix, 1e-5, 1e-5);
    }
}

#[test]
fn causal_short_conv_static_and_dynamic_parity() {
    for device in devices() {
        let charges = Charges::tally(&device);
        let kernel = declared::prepared(
            &device,
            &causal_short_conv::native_entry_with(causal_short_conv::Elements {
                A: Element::f32(),
                Y: Element::f32(),
            }),
            &NativeSpecialization::new(),
        )
        .unwrap();

        let (ch, c, banks, t) = (64usize, 4usize, 4usize, 2usize);
        let slab_banks = 2u64;
        let win_rows = c - 1 + t;

        for &(m, hd, residual_scale) in &[
            (1usize, 0usize, 1.0f32 / 5.0f32.sqrt()),
            (4usize, 4usize, 1.0f32 / 3.0f32),
            (6usize, 0usize, 1.0f32 / 5.0f32.sqrt()),
        ] {
            let input = pattern(m * ch, 41, 2.0);
            let conv = pattern(ch * c, 43, 0.5);
            let dyn_taps = pattern(m * hd * c, 47, 0.4);
            let mut window = vec![0.0f32; banks * win_rows * ch];
            // Populate bank 1 with past history frames
            for v in &mut window[win_rows * ch..2 * win_rows * ch] {
                *v = 0.25;
            }
            let init_hist = pattern((c - 1) * ch, 53, 1.2);
            window[win_rows * ch..win_rows * ch + (c - 1) * ch].copy_from_slice(&init_hist);

            let segments = vec![0i32, m as i32, m as i32, m as i32];
            let stop = vec![m as i32];
            let previous_bank = vec![1i32];
            let previous_tape = vec![0i32];
            let following_bank = vec![2i32];

            // Reference
            let mut exp_out = vec![0.0f32; m * ch];
            let mut exp_win = window.clone();
            let head_width = if hd > 0 { ch / hd } else { 1 };
            for row in 0..m {
                for channel in 0..ch {
                    let mut static_sum = 0.0f32;
                    let mut dyn_sum = 0.0f32;
                    for tap in 0..c {
                        let val = if row + tap < c - 1 {
                            window[win_rows * ch + (row + tap) * ch + channel]
                        } else {
                            input[(row + tap - (c - 1)) * ch + channel]
                        };
                        static_sum = conv[channel * c + tap].mul_add(val, static_sum);
                        if hd > 0 {
                            let head = channel / head_width;
                            dyn_sum = dyn_taps[(row * hd + head) * c + (c - 1 - tap)].mul_add(val, dyn_sum);
                        }
                    }
                    exp_out[row * ch + channel] = (input[row * ch + channel] + static_sum + dyn_sum) * residual_scale;
                }
            }
            for tap in 0..c - 1 {
                for channel in 0..ch {
                    let val = if m + tap < c - 1 {
                        window[win_rows * ch + (m + tap) * ch + channel]
                    } else {
                        input[(m + tap - (c - 1)) * ch + channel]
                    };
                    exp_win[2 * win_rows * ch + tap * ch + channel] = val;
                }
            }

            let in_t = f32_tensor(&device, &[m as u64, ch as u64], &input);
            let conv_t = f32_tensor(&device, &[ch as u64, c as u64], &conv);
            let dyn_t = f32_tensor(&device, &[m as u64, hd as u64, c as u64], &dyn_taps);
            let seg_t = i32_tensor(&device, &[2, 2], &segments);
            let stop_t = i32_tensor(&device, &[1], &stop);
            let pb_t = i32_tensor(&device, &[1], &previous_bank);
            let pt_t = i32_tensor(&device, &[1], &previous_tape);
            let fb_t = i32_tensor(&device, &[1], &following_bank);
            let (_slabs, mut arenas) = slab_arenas(
                &device,
                banks,
                slab_banks,
                &[Arena {
                    element: Element::f32(),
                    bank_shape: vec![win_rows as u64, ch as u64],
                    values: &window,
                }],
                false,
            );
            let mut win_slab = arenas.remove(0);

            let out = kernel
                .call(&charges, causal_short_conv::Args {
                    input: &in_t,
                    convolution: &conv_t,
                    dynamic_taps: &dyn_t,
                    segments: &seg_t,
                    stop: &stop_t,
                    previous_bank: &pb_t,
                    previous_tape: &pt_t,
                    following_bank: &fb_t,
                    window: &mut win_slab,
                    residual_scale,
                    slab_banks: slab_banks as u32,
                })
                .unwrap()
                .value;

            assert_close(
                &format!("causal_short_conv({:?}, M={m}, HD={hd})", device.backend()),
                &read_f32(&out),
                &exp_out,
                1e-5,
                1e-5,
            );
            let got_win = read_f32(&win_slab);
            assert_close(
                &format!("causal_short_conv_win({:?}, M={m}, HD={hd})", device.backend()),
                &got_win[2 * win_rows * ch..2 * win_rows * ch + (c - 1) * ch],
                &exp_win[2 * win_rows * ch..2 * win_rows * ch + (c - 1) * ch],
                1e-6,
                1e-6,
            );
        }
    }
}

#[test]
fn head_l2_clip_and_tau_parity() {
    for device in devices() {
        let charges = Charges::tally(&device);
        let kernel = declared::prepared(
            &device,
            &head_l2_clip::native_entry_with(head_l2_clip::Elements {
                A: Element::f32(),
                Y: Element::f32(),
            }),
            &NativeSpecialization::new(),
        )
        .unwrap();

        for &(m, h, w, sc, threshold, log_floor, log_alpha) in &[
            (2usize, 6usize, 128usize, 1usize, 22.627417f32, 8192.0f32, 0.1f32),
            (2, 4, 128, 0, 22.627417, 8192.0, 0.0),
            (2, 6, 16, 1, 8.0, 8192.0, 0.1),
        ] {
            let eps = 1e-6f32;
            // Make norm exceed threshold on some heads and stay below on others
            let x = pattern(m * h * w, 61, 4.0);
            let scale = pattern(sc * w, 67, 1.5);
            // Row 0 below log_floor (tau=1), row 1 well above log_floor (tau > 1)
            let coords = vec![100i32, 0, 0, 0, 32767, 0, 0, 0];

            let mut expected = vec![0.0f32; m * h * w];
            for row in 0..m {
                let tau = if log_alpha != 0.0 {
                    let ratio = (f64::from(coords[row * 4]) + 1.0) / f64::from(log_floor);
                    1.0 + f64::from(log_alpha) * ratio.max(1.0).ln()
                } else {
                    1.0
                };
                for head in 0..h {
                    let base = (row * h + head) * w;
                    let mut sq = 0.0f64;
                    for c in 0..w {
                        let mut v = f64::from(x[base + c]);
                        if sc == 1 {
                            v *= f64::from(scale[c]);
                        }
                        sq += v * v;
                    }
                    let clip = (f64::from(threshold) / (sq.sqrt() + f64::from(eps))).min(1.0);
                    for c in 0..w {
                        let mut v = f64::from(x[base + c]);
                        if sc == 1 {
                            v *= f64::from(scale[c]);
                        }
                        expected[base + c] = (v * clip * tau) as f32;
                    }
                }
            }

            let x_t = f32_tensor(&device, &[m as u64, h as u64, w as u64], &x);
            let s_t = f32_tensor(&device, &[sc as u64, w as u64], &scale);
            let c_t = i32_tensor(&device, &[m as u64, 4], &coords);
            let out = kernel
                .call(&charges, head_l2_clip::Args {
                    x: &x_t,
                    scale: &s_t,
                    coordinates: &c_t,
                    threshold,
                    epsilon: eps,
                    log_floor,
                    log_alpha,
                })
                .unwrap()
                .value;
            assert_close(
                &format!("head_l2_clip({:?}, W={w}, SC={sc}, alpha={log_alpha})", device.backend()),
                &read_f32(&out),
                &expected,
                1e-5,
                1e-5,
            );
        }
    }
}

#[test]
fn attention_rel_bias_and_dynamic_sink_parity() {
    for device in devices() {
        let charges = Charges::tally(&device);
        let kernel = declared::prepared(
            &device,
            &attention_rel_bias::native_entry_with(attention_rel_bias::Elements {
                A: Element::f32(),
            }),
            &NativeSpecialization::new(),
        )
        .unwrap();

        let (kv, g, w, ext) = (2usize, 3usize, 128usize, 8usize);
        let q_heads = kv * g;
        let hist_len = 12usize;
        let t_cap = 32usize;
        let slab_rows = 8u64;
        let scale = 1.0f32 / (w as f32).sqrt();

        // Test decode (M=1), verify (M=4) with dynamic sinks, native prefill (M=17 > VERIFY_ROWS=16) with static sinks, and ungated/no-sink (M=2)
        for &(m, sk, u) in &[(1usize, 1usize, w), (4, 4, w), (17, 1, w), (2, 0, 0)] {
            let query = pattern(m * q_heads * w, 71, 0.8);
            let gate = pattern(m * q_heads * u, 73, 1.2);
            let key = pattern(m * kv * w, 79, 0.8);
            let value = pattern(m * kv * w, 83, 0.8);
            let rel_bias = pattern(m * q_heads * ext, 89, 0.5);
            let sinks = pattern(sk * q_heads, 97, 1.0);
            let mut h_key = vec![0.0f32; t_cap * kv * w];
            let mut h_val = vec![0.0f32; t_cap * kv * w];
            h_key[..hist_len * kv * w].copy_from_slice(&pattern(hist_len * kv * w, 101, 0.8));
            h_val[..hist_len * kv * w].copy_from_slice(&pattern(hist_len * kv * w, 103, 0.8));

            let mut visible = vec![0i32; m * 1 * 2];
            let mut fresh = vec![0i32; m * 2];
            let mut destinations = vec![0i32; m];
            for row in 0..m {
                visible[row * 2] = 0;
                visible[row * 2 + 1] = hist_len as i32;
                fresh[row * 2] = 0;
                fresh[row * 2 + 1] = (row + 1) as i32;
                destinations[row] = (hist_len + row) as i32;
            }

            // Reference in F64
            let mut expected = vec![0.0f32; m * q_heads * w];
            for row in 0..m {
                let total_keys = hist_len + (row + 1);
                for kv_h in 0..kv {
                    for head in 0..g {
                        let qh = kv_h * g + head;
                        let mut scores = Vec::with_capacity(total_keys);
                        let mut max_s = f64::NEG_INFINITY;
                        for o in 0..total_keys {
                            let k_slice = if o < hist_len {
                                &h_key[(o * kv + kv_h) * w..(o * kv + kv_h + 1) * w]
                            } else {
                                let t_idx = o - hist_len;
                                &key[(t_idx * kv + kv_h) * w..(t_idx * kv + kv_h + 1) * w]
                            };
                            let dot: f64 = query[(row * q_heads + qh) * w..(row * q_heads + qh + 1) * w]
                                .iter()
                                .zip(k_slice)
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum();
                            let d = (total_keys - 1) - o;
                            let mut s = dot * f64::from(scale);
                            if d < ext {
                                s += f64::from(rel_bias[(row * q_heads + qh) * ext + d]);
                            }
                            if s > max_s {
                                max_s = s;
                            }
                            scores.push(s);
                        }
                        let exps: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                        let mut denom: f64 = exps.iter().sum();
                        if sk > 0 {
                            let sink = f64::from(sinks[(row % sk) * q_heads + qh]);
                            denom += (sink - max_s).exp();
                        }
                        for c in 0..w {
                            let mut acc = 0.0f64;
                            for o in 0..total_keys {
                                let v = if o < hist_len {
                                    h_val[(o * kv + kv_h) * w + c]
                                } else {
                                    value[((o - hist_len) * kv + kv_h) * w + c]
                                };
                                acc += exps[o] * f64::from(v);
                            }
                            let mut out_v = acc / denom.max(1e-30);
                            if u > 0 {
                                let gv = f64::from(gate[(row * q_heads + qh) * u + (c % u)]);
                                out_v /= 1.0 + (-gv).exp();
                            }
                            expected[(row * q_heads + qh) * w + c] = out_v as f32;
                        }
                    }
                }
            }

            let q_t = f32_tensor(&device, &[m as u64, q_heads as u64, w as u64], &query);
            let g_t = f32_tensor(&device, &[m as u64, q_heads as u64, u as u64], &gate);
            let k_t = f32_tensor(&device, &[m as u64, kv as u64, w as u64], &key);
            let v_t = f32_tensor(&device, &[m as u64, kv as u64, w as u64], &value);
            let rb_t = f32_tensor(&device, &[m as u64, q_heads as u64, ext as u64], &rel_bias);
            let sk_t = f32_tensor(&device, &[sk as u64, q_heads as u64], &sinks);
            let vis_t = i32_tensor(&device, &[m as u64, 1, 2], &visible);
            let fr_t = i32_tensor(&device, &[m as u64, 2], &fresh);
            let dst_t = i32_tensor(&device, &[m as u64], &destinations);
            let (_slabs, mut arenas) = slab_arenas(
                &device,
                t_cap,
                slab_rows,
                &[
                    Arena {
                        element: Element::f32(),
                        bank_shape: vec![kv as u64, w as u64],
                        values: &h_key,
                    },
                    Arena {
                        element: Element::f32(),
                        bank_shape: vec![kv as u64, w as u64],
                        values: &h_val,
                    },
                ],
                false,
            );
            let mut hk_slab = arenas.remove(0);
            let mut hv_slab = arenas.remove(0);

            let out = kernel
                .call(&charges, attention_rel_bias::Args {
                    query: &q_t,
                    gate: &g_t,
                    key: &k_t,
                    value: &v_t,
                    rel_bias: &rb_t,
                    sinks: &sk_t,
                    visible: &vis_t,
                    fresh: &fr_t,
                    destinations: &dst_t,
                    history_key: &mut hk_slab,
                    history_value: &mut hv_slab,
                    scale,
                    gate_function: 0,
                    slab_rows: slab_rows as u32,
                })
                .unwrap()
                .value;

            assert_close(
                &format!("attention_rel_bias({:?}, M={m}, SK={sk}, U={u})", device.backend()),
                &read_f32(&out),
                &expected,
                1e-5,
                1e-5,
            );
            // Verify KV append (both K and V)
            let got_hk = read_f32(&hk_slab);
            assert_close(
                &format!("attention_rel_bias_append_k({:?}, M={m})", device.backend()),
                &got_hk[hist_len * kv * w..(hist_len + m) * kv * w],
                &key,
                1e-6,
                1e-6,
            );
            let got_hv = read_f32(&hv_slab);
            assert_close(
                &format!("attention_rel_bias_append_v({:?}, M={m})", device.backend()),
                &got_hv[hist_len * kv * w..(hist_len + m) * kv * w],
                &value,
                1e-6,
                1e-6,
            );
        }
    }
}

fn bf16_round(v: f32) -> f32 {
    seismic_lang::registry::bf16_round(v)
}

fn bf16_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let charges = Charges::tally(device);
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| ((bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    Tensor::from_host(&charges, Element::bf16(), shape, &bytes).unwrap()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect()
}

#[test]
fn situ_activation_in_dense_expand_parity() {
    let gate_cap = 4.0f32;
    let up_cap = 25.0f32;
    let lb = situ_limit_word(gate_cap, up_cap);
    let (m, h, f) = (2usize, 256usize, 256usize);
    let eps = 1e-6f32;

    let residual = pattern(m * h, 107, 4.0);
    let norm = vec![1.0f32; h];
    let gate_w: Vec<f32> = pattern(f * h, 109, 1.5).into_iter().map(bf16_round).collect();
    let up_w: Vec<f32> = pattern(f * h, 113, 2.5).into_iter().map(bf16_round).collect();
    let out_rows: Vec<i32> = (0..m as i32).collect();

    // Reference in F64: SITU(g, u) = (gate_cap * tanh(g / gate_cap)) * sigmoid(g) * (up_cap * tanh(u / up_cap))
    let mut expected = vec![0.0f32; m * f];
    for row in 0..m {
        let sq: f64 = residual[row * h..(row + 1) * h].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let inv = 1.0 / (sq / h as f64 + f64::from(eps)).sqrt();
        let normed: Vec<f64> = residual[row * h..(row + 1) * h]
            .iter()
            .map(|&v| f64::from(bf16_round((f64::from(v) * inv) as f32)))
            .collect();
        for col in 0..f {
            let g_sum: f64 = normed.iter().zip(&gate_w[col * h..(col + 1) * h]).map(|(&x, &w)| x * f64::from(w)).sum();
            let u_sum: f64 = normed.iter().zip(&up_w[col * h..(col + 1) * h]).map(|(&x, &w)| x * f64::from(w)).sum();
            let g = f64::from(bf16_round(g_sum as f32));
            let u = f64::from(bf16_round(u_sum as f32));
            let gc = f64::from(gate_cap);
            let uc = f64::from(up_cap);
            let sig = 1.0 / (1.0 + (-g).exp());
            expected[row * f + col] = bf16_round(((gc * (g / gc).tanh()) * sig * (uc * (u / uc).tanh())) as f32);
        }
    }

    for device in devices() {
        let charges = Charges::tally(&device);
        let mut spec = NativeSpecialization::new();
        if device.backend() != BackendName::Cpu {
            for (name, val) in [
                ("H", h as u64),
                ("F", f as u64),
                ("GS", 0),
                ("US", 0),
                ("GB", 0),
                ("UB", 0),
                ("LB", lb),
            ] {
                spec = spec.with_static(name, val);
            }
        } else {
            spec = spec.with_static("GS", 0).with_static("US", 0);
        }
        let kernel = declared::prepared(
            &device,
            &dense_expand::native_entry_with(dense_expand::Elements {
                A: Element::bf16(),
                NW: Element::f32(),
                GW: Element::bf16(),
                UW: Element::bf16(),
            }),
            &spec,
        )
        .unwrap();

        let res_t = f32_tensor(&device, &[m as u64, h as u64], &residual);
        let norm_t = f32_tensor(&device, &[h as u64], &norm);
        let gw_t = bf16_tensor(&device, &[f as u64, h as u64], &gate_w);
        let uw_t = bf16_tensor(&device, &[f as u64, h as u64], &up_w);
        let or_t = i32_tensor(&device, &[m as u64], &out_rows);
        let empty_f32 = f32_tensor(&device, &[0], &[]);
        let glu_limit = i32_tensor(&device, &[0, lb], &[]);

        let out = kernel
            .call(&charges, dense_expand::Args {
                residual: &res_t,
                norm: &norm_t,
                gate_weight: &gw_t,
                up_weight: &uw_t,
                out_rows: &or_t,
                eps,
                activation: 3,
                gate_scale: &empty_f32,
                up_scale: &empty_f32,
                gate_bound: &empty_f32,
                up_bound: &empty_f32,
                glu_limit: &glu_limit,
            })
            .unwrap()
            .value;

        assert_close(
            &format!("situ_dense_expand({:?})", device.backend()),
            &read_bf16(&out),
            &expected,
            1e-1,
            2e-2,
        );
    }
}

#[test]
fn rnn_markov_gate_parity() {
    for device in devices() {
        let charges = Charges::tally(&device);
        let kernel_bf16 = declared::prepared(
            &device,
            &rnn_markov_gate::native_entry_with(rnn_markov_gate::Elements { A: Element::bf16() }),
            &NativeSpecialization::new(),
        )
        .unwrap();
        let kernel_f32 = declared::prepared(
            &device,
            &rnn_markov_gate::native_entry_with(rnn_markov_gate::Elements { A: Element::f32() }),
            &NativeSpecialization::new(),
        )
        .unwrap();

        for &(s, r) in &[(1usize, 64usize), (3, 128), (2, 512)] {
            let h_prev = pattern(s * r, 71, 1.5);
            let base_frnn = pattern(s * 3 * r, 73, 0.8);
            let base_x = pattern(s * 3 * r, 79, 0.8);
            let he_h = pattern(s * 3 * r, 83, 0.8);
            let he_e = pattern(s * 3 * r, 89, 0.8);

            let mut exp_h_next = vec![0.0f32; s * r];
            let mut exp_latent = vec![0.0f32; s * r];
            for si in 0..s {
                for ri in 0..r {
                    let i0 = si * 3 * r + ri;
                    let i1 = i0 + r;
                    let i2 = i0 + 2 * r;
                    let g0 = f64::from(base_frnn[i0])
                        + f64::from(base_x[i0])
                        + f64::from(he_h[i0])
                        + f64::from(he_e[i0]);
                    let g1 = f64::from(base_frnn[i1])
                        + f64::from(base_x[i1])
                        + f64::from(he_h[i1])
                        + f64::from(he_e[i1]);
                    let g2 = f64::from(base_frnn[i2])
                        + f64::from(base_x[i2])
                        + f64::from(he_h[i2])
                        + f64::from(he_e[i2]);
                    let gate = 1.0 / (1.0 + (-g0).exp());
                    let cand = g1.tanh();
                    exp_h_next[si * r + ri] =
                        (gate * f64::from(h_prev[si * r + ri]) + (1.0 - gate) * cand) as f32;
                    exp_latent[si * r + ri] = g2.tanh() as f32;
                }
            }

            let hp_t = f32_tensor(&device, &[s as u64, r as u64], &h_prev);
            let bf_t = f32_tensor(&device, &[s as u64, (3 * r) as u64], &base_frnn);
            let bx_t = f32_tensor(&device, &[s as u64, (3 * r) as u64], &base_x);
            let hh_t = f32_tensor(&device, &[s as u64, (3 * r) as u64], &he_h);
            let he_t = f32_tensor(&device, &[s as u64, (3 * r) as u64], &he_e);

            let out_bf16 = kernel_bf16
                .call(
                    &charges,
                    rnn_markov_gate::Args {
                        h_prev: &hp_t,
                        base_frnn: &bf_t,
                        base_x: &bx_t,
                        he_h: &hh_t,
                        he_e: &he_t,
                    },
                )
                .unwrap();
            assert_close(
                &format!("rnn_markov_gate.h_next({:?}, S={s}, R={r})", device.backend()),
                &read_f32(&out_bf16.r0),
                &exp_h_next,
                1e-5,
                1e-5,
            );
            assert_close(
                &format!("rnn_markov_gate.h_next_bf16({:?}, S={s}, R={r})", device.backend()),
                &read_bf16(&out_bf16.r1),
                &exp_h_next,
                1e-2,
                1e-2,
            );
            assert_close(
                &format!("rnn_markov_gate.latent_bf16({:?}, S={s}, R={r})", device.backend()),
                &read_bf16(&out_bf16.r2),
                &exp_latent,
                1e-2,
                1e-2,
            );

            let out_f32 = kernel_f32
                .call(
                    &charges,
                    rnn_markov_gate::Args {
                        h_prev: &hp_t,
                        base_frnn: &bf_t,
                        base_x: &bx_t,
                        he_h: &hh_t,
                        he_e: &he_t,
                    },
                )
                .unwrap();
            assert_close(
                &format!("rnn_markov_gate.h_next_f32({:?}, S={s}, R={r})", device.backend()),
                &read_f32(&out_f32.r1),
                &exp_h_next,
                1e-5,
                1e-5,
            );
            assert_close(
                &format!("rnn_markov_gate.latent_f32({:?}, S={s}, R={r})", device.backend()),
                &read_f32(&out_f32.r2),
                &exp_latent,
                1e-5,
                1e-5,
            );
        }
    }
}
