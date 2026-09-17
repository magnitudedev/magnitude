//! Qwen 3.5 decode harness: loads an MLX 4-bit checkpoint into device buffers, plans and
//! compiles the `qwen35_decode` composition once, then runs it per token.

mod bindings;
mod json;
mod safetensors;

use bindings::{Packed, Qwen35Decode, Qwen35DecodeShapes};
use json::Json;
use safetensors::SafeTensors;
use seismic_lang::program::{collect_files, compile};
use seismic_metal::msl::Config;
use seismic_metal::plan_exec::compile_plan;
use seismic_metal::runtime::{Buffer, Device};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const EOS: &[u32] = &[248044, 248046];

struct Args {
    model: PathBuf,
    libs: Vec<PathBuf>,
    prompt: String,
    ids: Option<Vec<u32>>,
    raw: bool,
    steps: usize,
    history: i64,
    partitions: i64,
    block: i64,
    sg_per_tg: i64,
    piece: Option<i64>,
    rows_per_item: i64,
    split: i64,
    dump_logits: Option<PathBuf>,
    /// Run the first prompt token step by step and dump named tensors after each layer here.
    dump_dir: Option<PathBuf>,
    /// Time every distinct kernel instantiation and print the per-token breakdown.
    profile: bool,
}

fn args() -> Result<Args, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut a = Args {
        model: PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"),
        libs: vec![root.join("../../seismic-std/lib"), root.join("../lib")],
        prompt: "Explain what a roofline model is in two sentences.".into(),
        ids: None,
        raw: false,
        steps: 64,
        history: 4096,
        partitions: 8,
        block: 4096,
        sg_per_tg: 4,
        piece: None,
        rows_per_item: 4,
        split: 1,
        dump_logits: None,
        dump_dir: None,
        profile: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let mut next = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => a.model = PathBuf::from(next(&mut i, "--model")?),
            "--lib" => a.libs.push(PathBuf::from(next(&mut i, "--lib")?)),
            "--prompt" => a.prompt = next(&mut i, "--prompt")?,
            "--ids" => a.ids = Some(next(&mut i, "--ids")?.split(',').map(|s| s.trim().parse::<u32>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?),
            "--raw" => a.raw = true,
            "--steps" => a.steps = next(&mut i, "--steps")?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--history" => a.history = next(&mut i, "--history")?.parse().map_err(|e| format!("--history: {e}"))?,
            "--partitions" => a.partitions = next(&mut i, "--partitions")?.parse().map_err(|e| format!("--partitions: {e}"))?,
            "--block" => a.block = next(&mut i, "--block")?.parse().map_err(|e| format!("--block: {e}"))?,
            "--sg-per-tg" => a.sg_per_tg = next(&mut i, "--sg-per-tg")?.parse().map_err(|e| format!("--sg-per-tg: {e}"))?,
            "--split" => a.split = next(&mut i, "--split")?.parse().map_err(|e| format!("--split: {e}"))?,
            "--rows-per-item" => a.rows_per_item = next(&mut i, "--rows-per-item")?.parse().map_err(|e| format!("--rows-per-item: {e}"))?,
            "--piece" => a.piece = Some(next(&mut i, "--piece")?.parse().map_err(|e| format!("--piece: {e}"))?),
            "--dump-logits" => a.dump_logits = Some(PathBuf::from(next(&mut i, "--dump-logits")?)),
            "--dump-dir" => a.dump_dir = Some(PathBuf::from(next(&mut i, "--dump-dir")?)),
            "--profile" => a.profile = true,
            other => return Err(format!("unknown argument `{other}`")),
        }
        i += 1;
    }
    Ok(a)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn shapes_from_config(config: &Json, a: &Args) -> Result<Qwen35DecodeShapes, String> {
    let t = config.get("text_config")?;
    let layers = t.get("num_hidden_layers")?.as_i64()?;
    let types = t.get("layer_types")?.as_array()?;
    if layers % 4 != 0 || types.len() as i64 != layers {
        return Err("layer count is not a multiple of four".into());
    }
    for (i, ty) in types.iter().enumerate() {
        let expect = if i % 4 == 3 { "full_attention" } else { "linear_attention" };
        if ty.as_str()? != expect {
            return Err(format!("layer {i} is {} but the composition expects {expect}", ty.as_str()?));
        }
    }
    let head_dim = t.get("head_dim")?.as_i64()?;
    let factor = t.get("rope_parameters")?.get("partial_rotary_factor")?.as_f64()?;
    let r = (head_dim as f64 * factor).round() as i64;
    let key_dim = t.get("linear_key_head_dim")?.as_i64()?;
    if key_dim != t.get("linear_value_head_dim")?.as_i64()? {
        return Err("the composition assumes equal recurrent key and value widths".into());
    }
    Ok(Qwen35DecodeShapes {
        D: t.get("hidden_size")?.as_i64()?,
        LA: layers / 4,
        H: t.get("num_attention_heads")?.as_i64()?,
        KV: t.get("num_key_value_heads")?.as_i64()?,
        R: r,
        S: head_dim - r,
        NK: t.get("linear_num_key_heads")?.as_i64()?,
        GV: t.get("linear_num_value_heads")?.as_i64()? / t.get("linear_num_key_heads")?.as_i64()?,
        RW: key_dim,
        C: t.get("linear_conv_kernel_dim")?.as_i64()?,
        F: t.get("intermediate_size")?.as_i64()?,
        V: t.get("vocab_size")?.as_i64()?,
        T: a.history,
        B: a.block,
    })
}

struct Store {
    dense: HashMap<&'static str, Buffer>,
    packed: HashMap<&'static str, Packed>,
}

impl Store {
    fn dense(&self, name: &str) -> &Buffer {
        self.dense.get(name).unwrap_or_else(|| panic!("no dense buffer `{name}`"))
    }
    fn packed(&self, name: &str) -> &Packed {
        self.packed.get(name).unwrap_or_else(|| panic!("no packed buffer `{name}`"))
    }
}

/// Concatenate the named tensors into one buffer of exactly `expected` bytes.
fn stack(device: &Device, st: &SafeTensors, names: &[String], expected: usize, transform: Option<&dyn Fn(&[u8]) -> Vec<u8>>) -> Result<Buffer, String> {
    let buffer = device.buffer(expected)?;
    let mut offset = 0usize;
    for name in names {
        let bytes = st.read(name)?;
        let bytes = match transform {
            Some(f) => f(&bytes),
            None => bytes,
        };
        if offset + bytes.len() > expected {
            return Err(format!("`{name}`: stacked size exceeds the expected {expected} bytes"));
        }
        buffer.write_at(offset, &bytes);
        offset += bytes.len();
    }
    if offset != expected {
        return Err(format!("stacked {} bytes for `{}`, expected {expected}", offset, names.first().map(String::as_str).unwrap_or("")));
    }
    Ok(buffer)
}

fn stack_packed(device: &Device, st: &SafeTensors, prefixes: &[String], sizes: &HashMap<(&str, &str), usize>, param: &'static str) -> Result<Packed, String> {
    let part = |suffix: &str, part: &str| -> Result<Buffer, String> {
        let names: Vec<String> = prefixes.iter().map(|p| format!("{p}.{suffix}")).collect();
        stack(device, st, &names, sizes[&(param, part)], None)
    };
    Ok(Packed { words: part("weight", "words")?, scale: part("scales", "scale")?, bias: part("biases", "bias")? })
}

fn run() -> Result<(), String> {
    let a = args()?;
    let t0 = Instant::now();
    let config = Json::parse(&std::fs::read_to_string(a.model.join("config.json")).map_err(|e| format!("config.json: {e}"))?)?;
    let shapes = shapes_from_config(&config, &a)?;
    eprintln!("shapes: {shapes:?}");

    // Program: standard library plus the model library.
    let files = collect_files(&a.libs)?;
    let program = compile(&files, &["metal".to_string(), "cpu".to_string()]).map_err(|errors| errors.iter().map(|e| e.render()).collect::<Vec<_>>().join("\n"))?;
    let plan = seismic_lang::plan::plan(&program, "qwen35_decode", &shapes.map())?;
    let device = Device::open()?;
    eprintln!("device: {}", device.info().name);
    let t1 = Instant::now();
    let compiled = compile_plan(&device, &program, &plan, Config { sg_per_tg: a.sg_per_tg, piece: a.piece, per_item: a.rows_per_item, split: a.split, cores: device.info().cores as i64, max_threads_per_threadgroup: device.info().max_threads_per_threadgroup as i64, max_threadgroup_bytes: device.info().max_threadgroup_bytes as i64 })?;
    eprintln!("plan: {} steps, {} pipelines, compiled in {:.2} s", plan.steps.len(), compiled.pipelines.len(), t1.elapsed().as_secs_f64());

    // Weights.
    let st = SafeTensors::open(&a.model.join("model.safetensors"))?;
    let sizes: HashMap<(&str, &str), usize> = Qwen35Decode::sizes(&shapes).into_iter().map(|(p, part, n)| ((p, part), n)).collect();
    let mut store = Store { dense: HashMap::new(), packed: HashMap::new() };
    let layer = |l: i64| format!("language_model.model.layers.{l}");
    let attn_layer = |a: i64| 4 * a + 3;
    let rec_layer = |r: i64| 4 * (r / 3) + r % 3;
    let all: Vec<i64> = (0..4 * shapes.LA).collect();
    let attn: Vec<i64> = (0..shapes.LA).map(attn_layer).collect();
    let rec: Vec<i64> = (0..3 * shapes.LA).map(rec_layer).collect();
    let names = |layers: &[i64], suffix: &str| -> Vec<String> { layers.iter().map(|l| format!("{}.{suffix}", layer(*l))).collect() };
    let t2 = Instant::now();
    store.packed.insert("embed", stack_packed(&device, &st, &["language_model.model.embed_tokens".to_string()], &sizes, "embed")?);
    store.dense.insert("final_norm", stack(&device, &st, &["language_model.model.norm.weight".to_string()], sizes[&("final_norm", "")], None)?);
    store.dense.insert("in_norm", stack(&device, &st, &names(&all, "input_layernorm.weight"), sizes[&("in_norm", "")], None)?);
    store.dense.insert("post_norm", stack(&device, &st, &names(&all, "post_attention_layernorm.weight"), sizes[&("post_norm", "")], None)?);
    for (param, suffix) in [("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"), ("v_proj", "self_attn.v_proj"), ("o_proj", "self_attn.o_proj")] {
        store.packed.insert(param, stack_packed(&device, &st, &names(&attn, suffix), &sizes, param)?);
    }
    store.dense.insert("q_norm", stack(&device, &st, &names(&attn, "self_attn.q_norm.weight"), sizes[&("q_norm", "")], None)?);
    store.dense.insert("k_norm", stack(&device, &st, &names(&attn, "self_attn.k_norm.weight"), sizes[&("k_norm", "")], None)?);
    for (param, suffix) in [("qkv_proj", "linear_attn.in_proj_qkv"), ("z_proj", "linear_attn.in_proj_z"), ("a_proj", "linear_attn.in_proj_a"), ("b_proj", "linear_attn.in_proj_b"), ("out_proj", "linear_attn.out_proj")] {
        store.packed.insert(param, stack_packed(&device, &st, &names(&rec, suffix), &sizes, param)?);
    }
    store.dense.insert("conv", stack(&device, &st, &names(&rec, "linear_attn.conv1d.weight"), sizes[&("conv", "")], None)?);
    let negative_exp = |bytes: &[u8]| -> Vec<u8> {
        bytes.chunks_exact(4).flat_map(|c| (-(f32::from_le_bytes([c[0], c[1], c[2], c[3]]).exp())).to_le_bytes()).collect()
    };
    store.dense.insert("rate", stack(&device, &st, &names(&rec, "linear_attn.A_log"), sizes[&("rate", "")], Some(&negative_exp))?);
    let bf16_to_f32 = |bytes: &[u8]| -> Vec<u8> { bytes.chunks_exact(2).flat_map(|c| [0u8, 0u8, c[0], c[1]]).collect() };
    let dt_bias_names = names(&rec, "linear_attn.dt_bias");
    let widen: Option<&dyn Fn(&[u8]) -> Vec<u8>> = if st.info(&dt_bias_names[0])?.dtype == "BF16" { Some(&bf16_to_f32) } else { None };
    store.dense.insert("dt_bias", stack(&device, &st, &dt_bias_names, sizes[&("dt_bias", "")], widen)?);
    store.dense.insert("rnorm", stack(&device, &st, &names(&rec, "linear_attn.norm.weight"), sizes[&("rnorm", "")], None)?);
    for (param, suffix) in [("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj"), ("down_proj", "mlp.down_proj")] {
        store.packed.insert(param, stack_packed(&device, &st, &names(&all, suffix), &sizes, param)?);
    }
    let weight_bytes: usize = store.dense.values().map(|b| b.len).sum::<usize>() + store.packed.values().map(|p| p.words.len + p.scale.len + p.bias.len).sum::<usize>();
    eprintln!("weights: {:.2} GB in {:.2} s", weight_bytes as f64 / 1e9, t2.elapsed().as_secs_f64());

    // States and scratch, zeroed.
    for name in ["kcache", "vcache", "window", "delta", "visible", "x", "qg", "kin", "vin", "q", "gate", "attended", "projected", "z", "a_in", "b_in", "qkv", "beta", "decay", "rout", "rgated", "h", "logits_out", "amax_vals", "amax_idx", "next"] {
        let n = sizes[&(name, "")];
        let b = device.buffer(n)?;
        b.write(&vec![0u8; n]);
        store.dense.insert(name, b);
    }

    let mut decode = Qwen35Decode {
        embed: store.packed("embed"),
        final_norm: store.dense("final_norm"),
        in_norm: store.dense("in_norm"),
        post_norm: store.dense("post_norm"),
        q_proj: store.packed("q_proj"),
        k_proj: store.packed("k_proj"),
        v_proj: store.packed("v_proj"),
        o_proj: store.packed("o_proj"),
        q_norm: store.dense("q_norm"),
        k_norm: store.dense("k_norm"),
        qkv_proj: store.packed("qkv_proj"),
        z_proj: store.packed("z_proj"),
        a_proj: store.packed("a_proj"),
        b_proj: store.packed("b_proj"),
        conv: store.dense("conv"),
        rate: store.dense("rate"),
        dt_bias: store.dense("dt_bias"),
        rnorm: store.dense("rnorm"),
        out_proj: store.packed("out_proj"),
        gate_proj: store.packed("gate_proj"),
        up_proj: store.packed("up_proj"),
        down_proj: store.packed("down_proj"),
        kcache: store.dense("kcache"),
        vcache: store.dense("vcache"),
        window: store.dense("window"),
        delta: store.dense("delta"),
        token: 0,
        pos: 0,
        visible: store.dense("visible"),
        x: store.dense("x"),
        qg: store.dense("qg"),
        kin: store.dense("kin"),
        vin: store.dense("vin"),
        q: store.dense("q"),
        gate: store.dense("gate"),
        attended: store.dense("attended"),
        // The same rows, seen flat by the output projection.
        attended_flat: store.dense("attended"),
        projected: store.dense("projected"),
        z: store.dense("z"),
        a_in: store.dense("a_in"),
        b_in: store.dense("b_in"),
        qkv: store.dense("qkv"),
        beta: store.dense("beta"),
        decay: store.dense("decay"),
        rout: store.dense("rout"),
        rgated: store.dense("rgated"),
        h: store.dense("h"),
        logits_out: store.dense("logits_out"),
        amax_vals: store.dense("amax_vals"),
        amax_idx: store.dense("amax_idx"),
        next: store.dense("next"),
    };

    // Prompt.
    let tokenizer = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json")).map_err(|e| format!("tokenizer: {e}"))?;
    let prompt_ids: Vec<u32> = match &a.ids {
        Some(ids) => ids.clone(),
        None => {
            let text = if a.raw { a.prompt.clone() } else { format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", a.prompt) };
            tokenizer.encode(text, false).map_err(|e| format!("encode: {e}"))?.get_ids().to_vec()
        }
    };
    if prompt_ids.len() + a.steps > a.history as usize {
        return Err(format!("prompt plus steps exceed the history of {}", a.history));
    }
    eprintln!("prompt: {} token(s); ready in {:.2} s", prompt_ids.len(), t0.elapsed().as_secs_f64());

    let mut step = |decode: &mut Qwen35Decode, token: u32, pos: usize| -> Result<(u32, f64), String> {
        decode.token = token as i64;
        decode.pos = pos as i64;
        let mut visible = Vec::with_capacity(8);
        visible.extend_from_slice(&0i32.to_le_bytes());
        visible.extend_from_slice(&((pos + 1) as i32).to_le_bytes());
        decode.visible.write(&visible);
        let gpu = device.run_plan(&compiled, decode)?;
        let next = decode.next.read(4);
        Ok((i32::from_le_bytes([next[0], next[1], next[2], next[3]]) as u32, gpu))
    };

    // Profile: each distinct kernel instantiation encoded 50 times in one command buffer,
    // then the plan's per-token time attributed by step.
    if a.profile {
        decode.token = prompt_ids[0] as i64;
        decode.pos = 0;
        let mut visible = Vec::new();
        visible.extend_from_slice(&0i32.to_le_bytes());
        visible.extend_from_slice(&1i32.to_le_bytes());
        decode.visible.write(&visible);
        let mut per_step: Vec<f64> = vec![0.0; plan.steps.len()];
        let mut seen: HashMap<usize, f64> = HashMap::new();
        for (i, step) in compiled.steps.iter().enumerate() {
            let t = match seen.get(&step.pipeline) {
                Some(t) => *t,
                None => {
                    device.run_plan_steps_repeated(&compiled, &decode, i..i + 1, 5)?;
                    let t = device.run_plan_steps_repeated(&compiled, &decode, i..i + 1, 50)? / 50.0;
                    seen.insert(step.pipeline, t);
                    t
                }
            };
            per_step[i] = t;
        }
        let mut by_kernel: std::collections::BTreeMap<String, (usize, f64)> = std::collections::BTreeMap::new();
        for (i, step) in plan.steps.iter().enumerate() {
            let mut shapes: Vec<String> = step.shapes.iter().map(|(k, v)| format!("{k}={v}")).collect();
            shapes.sort();
            let e = by_kernel.entry(format!("{}[{}]", step.kernel, shapes.join(","))).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += per_step[i];
        }
        let total: f64 = per_step.iter().sum();
        let whole = device.run_plan(&compiled, &decode)?;
        eprintln!("{:>8} {:>5} {:>9} {:>9}  kernel", "ms", "n", "ms/each", "share");
        let mut rows: Vec<(&String, &(usize, f64))> = by_kernel.iter().collect();
        rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        for (k, (n, t)) in rows {
            eprintln!("{:8.3} {n:5} {:9.4} {:8.1}%  {k}", t * 1e3, t * 1e3 / *n as f64, 100.0 * t / total);
        }
        eprintln!("sum of isolated kernel times: {:.3} ms; whole step in one command buffer: {:.3} ms", total * 1e3, whole * 1e3);
        // Dispatch overhead: the embedding step (5 KB of work) encoded as many times as the plan has steps.
        let n = plan.steps.len();
        let trivial = device.run_plan_steps_repeated(&compiled, &decode, 0..1, n)?;
        eprintln!("{n} dispatches of the embedding step: {:.3} ms ({:.2} us each)", trivial * 1e3, trivial * 1e6 / n as f64);
        return Ok(());
    }

    // Debug: the first prompt tokens step by step, dumping the residual stream after every
    // layer plus the recurrent and attention intermediates of the first layers.
    if let Some(dir) = &a.dump_dir {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let dump = |name: &str, buffer: &Buffer, pos: usize| -> Result<(), String> {
            let bytes = buffer.read(buffer.len);
            std::fs::write(dir.join(format!("p{pos}_{name}.bin")), bytes).map_err(|e| e.to_string())
        };
        let positions = prompt_ids.len().min(2);
        for pos in 0..positions {
            decode.token = prompt_ids[pos] as i64;
            decode.pos = pos as i64;
            let mut visible = Vec::new();
            visible.extend_from_slice(&0i32.to_le_bytes());
            visible.extend_from_slice(&((pos + 1) as i32).to_le_bytes());
            decode.visible.write(&visible);
            let mut layer = 0;
            for (i, step) in plan.steps.iter().enumerate() {
                device.run_plan_steps(&compiled, &decode, i..i + 1)?;
                if step.kernel == "embedding" {
                    dump("embed", decode.x, pos)?;
                }
                if step.kernel == "attention" && layer < 4 {
                    dump(&format!("l{layer}_attended"), decode.attended, pos)?;
                }
                if step.kernel == "recurrent_prepare" && layer < 4 {
                    dump(&format!("l{layer}_qkv"), decode.qkv, pos)?;
                    dump(&format!("l{layer}_beta"), decode.beta, pos)?;
                    dump(&format!("l{layer}_decay"), decode.decay, pos)?;
                    dump(&format!("l{layer}_projected"), decode.projected, pos)?;
                }
                if step.kernel == "delta_step" && layer < 4 {
                    dump(&format!("l{layer}_rout"), decode.rout, pos)?;
                }
                if step.kernel == "gated_norm" && layer < 4 {
                    dump(&format!("l{layer}_rgated"), decode.rgated, pos)?;
                }
                if step.kernel == "attention_prepare" && layer < 4 {
                    dump(&format!("l{layer}_q"), decode.q, pos)?;
                    dump(&format!("l{layer}_gate"), decode.gate, pos)?;
                }
                if step.kernel == "attention" && layer < 4 {
                    dump(&format!("l{layer}_attended"), decode.attended, pos)?;
                }
                if (step.kernel == "projection_add" || step.kernel == "gate_projection_add") && step.tensors.iter().any(|t| t.root == "out_proj" || t.root == "o_proj") && layer < 4 {
                    dump(&format!("l{layer}_mixed"), decode.x, pos)?;
                }
                if step.kernel == "projection_add" && step.tensors.iter().any(|t| t.root == "down_proj") {
                    dump(&format!("l{layer}"), decode.x, pos)?;
                    layer += 1;
                }
                if step.kernel == "norm_logits" {
                    dump("logits", decode.logits_out, pos)?;
                }
            }
        }
        eprintln!("dumped {} position(s) to {}", positions, dir.display());
        return Ok(());
    }

    let t3 = Instant::now();
    let mut next = 0u32;
    let mut prefill_gpu = 0.0;
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        let (n, gpu) = step(&mut decode, tok, pos)?;
        next = n;
        prefill_gpu += gpu;
    }
    eprintln!("prefill: {} token(s) in {:.3} s wall, {:.3} s GPU", prompt_ids.len(), t3.elapsed().as_secs_f64(), prefill_gpu);
    if let Some(path) = &a.dump_logits {
        let bytes = decode.logits_out.read(sizes[&("logits_out", "")]);
        std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        eprintln!("wrote logits after the prompt to {}", path.display());
    }

    let t4 = Instant::now();
    let mut generated = Vec::new();
    let mut gen_gpu = 0.0;
    let stdout = std::io::stdout();
    for i in 0..a.steps {
        generated.push(next);
        let piece = tokenizer.decode(&[next], true).map_err(|e| format!("decode: {e}"))?;
        let mut out = stdout.lock();
        write!(out, "{piece}").ok();
        out.flush().ok();
        if EOS.contains(&next) {
            break;
        }
        let pos = prompt_ids.len() + i;
        let (n, gpu) = step(&mut decode, next, pos)?;
        next = n;
        gen_gpu += gpu;
    }
    println!();
    let wall = t4.elapsed().as_secs_f64();
    eprintln!("generated {} token(s): {:.1} tok/s wall ({:.2} ms/token), {:.2} ms/token GPU", generated.len(), generated.len() as f64 / wall, 1e3 * wall / generated.len().max(1) as f64, 1e3 * gen_gpu / generated.len().max(1) as f64);
    eprintln!("ids: {:?}", generated);
    Ok(())
}
