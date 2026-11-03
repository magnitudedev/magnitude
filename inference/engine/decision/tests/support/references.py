"""Golden render and answer fixtures for the decision types, from each publisher's pinned reference code.

Runs on CPU with tokenizers only (no weights). For every request of `requests.json` the reference renders
and tokenizes its prompts; the fixture records each evaluated sequence's token ids, its trunk (state block)
length, its question groups after the trunk where the type packs several, its readout positions and the
tokens each logits readout reads, which the Rust parity tests (`tests/parity.rs`) compare exactly. Images are
recorded unexpanded: one placeholder per image. For d1 each readout is a logits readout (an index into the
fixture's `token_lists`) or a score readout over a row range (a read's pointer); `d1-codes` is the same with
`d1_numbers` off, grids and box edges read as codes.

Every case also carries synthetic readout values and the answers the reference's own code computes from them:
each noul, choice and score through the reference's probability code (qev's `calibrate.softmax` and
`answers._normalize`, pplx's `Decider._respond`, d3's float32 readout softmax and `product_answer`, d1's
`api.answer` through `calibration.json`), a d3 box through its runtime's box pipeline, d1's other kinds through
`api.answer`. A score's level and choice and score confidences are the wire's for every model: TypeSafe's
formulas (`typesafe`), applied to the reference's probabilities; a score's legend is the wire format's and is
left out. Each noul, choice and score answer comes with the bound of each of its numbers (`softmax_bounds`,
`typesafe`), the case's `bounds`; an answer of another kind has none (null).

d1 needs torch (CPU) for the reference's `pack_set` and `pointer.spans`, d3 and pplx for their readout softmax.

    python references.py <d3|qev|pplx|d1|d1-codes> <reference dir> <tokenizer dir> <requests.json> <out.json>

`<reference dir>` holds the publisher's code at its pinned commit: d3-ultra `52ba9e7` (`d3_format.py`,
`d3_compose.py`, `d3_runtime.py`), OneJev `qev` `81ce62f` with mammoth-1 `serve.patch` applied (`qev/`),
pplx-decider Flash v1.2 `9ff7cf9` (`source/src/pplx_decider/`), LiquidAI/d1-2B `97c5704` (the repository itself:
`prompt.py`, `media.py`). `<tokenizer dir>` the model repository's tokenizer files at the same commit.
"""

from __future__ import annotations

import bisect
import json
import math
import struct
import sys
import types


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)


def chat(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def f32(value):
    """`value` rounded to float32, as a readout value is."""
    return struct.unpack("f", struct.pack("f", value))[0]


def noise(seed, index):
    """A deterministic share in [-0.3, 0.3), computed alike by the Rust tests."""
    return ((seed * 2654435761) ^ (index * 40503)) % 1000 / 1000.0 * 0.6 - 0.3


def synthetic(described, count):
    """A code slot's logits from their description, one per token (`tests/parity.rs` computes the same)."""
    seed = described["seed"]
    if "peak" in described:
        center, width = described["peak"]
        return [f32(-0.5 * ((i - center) / width) * ((i - center) / width) + noise(seed, i)) for i in range(count)]
    if "presence" in described:
        return [f32((described["presence"] if i < described["yes"] else 0.0) + noise(seed, i)) for i in range(count)]
    return [f32(10.0 * noise(seed, i)) for i in range(count)]


# Unit roundoffs of float32 and float64, and their smallest subnormals.
F32, F64 = 2.0 ** -24, 2.0 ** -53
SUBNORMAL = {F32: 2.0 ** -149, F64: 2.0 ** -1074}


def softmax_bounds(values, temperature, described, unit):
    """Bounds on how far each of `softmax(values / temperature)` as a reference computes it, in floating point of
    unit roundoff `unit` and then renormalized in float64, can lie from the decision crate's float64 softmax of the
    same values through `described`, the temperature as its description stores it: each probability's bound, in
    option order.

    Every operation rounds with relative error at most its format's unit roundoff u (exp, one ulp: 2u). With
    s_i = x_i / T (a division, or a product with the rounded reciprocal: 2u) and the shift c = max s, the shifted
    d_i = s_i - c is off by at most 2u(|s_i| + |c|) + u|d_i|, and by e|d_i| more where the reference's temperature
    is the described one's times 1 + e, so e_i = exp(d_i) by the relative error
    a_i = 2u(|s_i| + |c|) + (u + e)|d_i| + 2u (the shift itself is exact: the softmax does not depend on it). Their sum
    of n terms is off by sigma = sum q_i a_i + (n - 1)u (q the exact probabilities, every summation order), and a
    quotient (one or two roundings) by a_i + sigma + 2u. A float64 renormalization by the probabilities' total
    adds abar + sigma + 2u + n u64, abar = sum q_i a_i, to give r_i = a_i + abar + 2 sigma + 4u + n u64. The crate
    computes the same softmax in float64: r_i at u64, a bound on its own error. The two compose by products of
    factors 1 + delta whose magnitudes sum to at most R_i = r_i(u) + r_i(u64), so the probability lies within
    expm1(R_i) q_i; a result that underflows into subnormals is off by at most the smallest subnormal per
    operation, 2n of them in all."""
    n = len(values)
    scaled = [value / described for value in values]
    top = max(scaled)
    weights = [math.exp(s - top) for s in scaled]
    total = math.fsum(weights)
    exact = [w / total for w in weights]

    def relative(u, e):
        a = [2 * u * (abs(s) + abs(top)) + (u + e) * (top - s) + 2 * u for s in scaled]
        mean = math.fsum(q * x for q, x in zip(exact, a))
        sigma = mean + (n - 1) * u
        return [x + mean + 2 * sigma + 4 * u + n * F64 for x in a]

    # The crate divides by the described temperature itself.
    theirs, ours = relative(unit, abs(described / temperature - 1)), relative(F64, 0.0)
    return [math.expm1(r + o) * q + 2 * n * SUBNORMAL[unit] for r, o, q in zip(theirs, ours, exact)]


def typesafe(reference, bounds):
    """`reference`'s noul, choice or score answer with the wire's derived fields (TypeSafe's published formulas,
    docs.typesafe.ai "Confidence"; a score's level the probability-weighted mean), the same for every model, and
    the bound of each of its numbers. `bounds` holds each probability's bound (`softmax_bounds`; a noul's one).

    A derived field's bound follows from the probabilities' through its formula: the peak moves by at most the
    largest bound, the mean level by sum i b_i, the mean distance from the modal level by sum b_i |i - m|; each
    side's float64 evaluation of a sum of n terms of at most M adds at most n M u64."""
    kind = reference["type"]
    if kind == "noul":
        return reference, {"noul": bounds[0]}
    keys, shares = list(reference["probabilities"]), list(reference["probabilities"].values())
    n = len(shares)
    keyed = dict(zip(keys, bounds))
    if kind == "choice":
        # (p_max - 1/n) / (1 - 1/n); one option is certain.
        confidence = 1.0 if n == 1 else (max(shares) - 1 / n) / (1 - 1 / n)
        bound = 0.0 if n == 1 else max(bounds) * n / (n - 1) + 4 * F64
        return ({"type": "choice", "choice": reference["choice"], "probabilities": reference["probabilities"],
                 "confidence": confidence}, {"probabilities": keyed, "confidence": bound})
    # max(0, 1 - sum p_i |i - m| / MAD_unif), m the first most likely level.
    mode = shares.index(max(shares))
    spread = sum(p * abs(i - mode) for i, p in enumerate(shares))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    score = sum(i * p for i, p in enumerate(shares))
    evaluated = 2 * n * (n - 1) * F64
    return ({"type": "score", "score": score, "probabilities": reference["probabilities"],
             "confidence": max(0.0, 1 - spread / even)},
            {"probabilities": keyed, "score": sum(i * b for i, b in enumerate(bounds)) + evaluated,
             "confidence": (sum(b * abs(i - mode) for i, b in enumerate(bounds)) + evaluated) / even + 2 * F64})


def slot_values(rng, count):
    """A code slot's synthetic logits: peaked about one of its `count` options, or spread."""
    seed = rng.randrange(1 << 20)
    if rng.random() < 0.5:
        return {"peak": [f32(rng.uniform(0, count - 1)), f32(rng.uniform(0.7, 3.0))], "seed": seed}
    return {"seed": seed}


def d3(reference, tokenizer, requests):
    sys.path.insert(0, reference)
    import random

    import d3_compose as dc
    import d3_format as df
    import d3_runtime as dr
    import torch

    codes, code_ids = df.answer_codes(tokenizer)
    out = {"codes": code_ids, "cases": []}
    # A box's edges as the runtime reads them (`D3Runtime._composed_row`, `_box`): the head's rows past the bins
    # masked, a float32 softmax through the box temperature, then `decode`, `confidence` and `calibrate` through
    # `box_calibration`.
    config = json.load(open(f"{sys.argv[3]}/decision_config.json"))
    temperature, calibration = config["temperature"], config["systemone"]["box_calibration"]
    by_questions = types.SimpleNamespace(temperature=temperature, temperature_by_questions=dr.temperature_table(
        config.get("temperature_by_questions"), "decision_config.json"))

    def edges(rng):
        described = {"peak": [f32(rng.uniform(0.1, 0.9) * dc.BINS), f32(rng.uniform(1.0, 12.0))],
                     "seed": rng.randrange(1 << 20)}
        return described, synthetic(described, len(code_ids))

    def decision(question, decisions, rng):
        """A noul, choice or score readout's synthetic logits, and the runtime's answer to them (`D3.probabilities`,
        `_composed_row`): the head's rows past the options masked, a float32 softmax through the request's
        temperature (`temperature_for` its `decisions`), then `product_answer`."""
        asked = dr.normalize_question(question)
        count, at = len(asked.keys), dr.D3.temperature_for(by_questions, decisions)
        values = slot_values(rng, count)
        logits = torch.tensor(synthetic(values, len(code_ids)), dtype=torch.float32)
        logits = logits.masked_fill(torch.arange(len(code_ids)) >= count, float("-inf"))
        answer = dr.product_answer(asked, (logits / at).softmax(-1).tolist()[:count])
        bounds = softmax_bounds(synthetic(values, count), at, f32(at), F32)
        # A noul's answer is its option 1, true.
        return values, typesafe(answer, bounds[1:] if asked.kind == "noul" else bounds)

    def normalized(question):
        # d3_runtime.normalize_question: a score renders as a choice over its level texts.
        if question["type"] == "score":
            return {"type": "choice", "instructions": question.get("instructions"),
                    "criteria": {df.describe(level): None for level in question["criteria"]}}
        return question

    for case in requests:
        state, questions, front = case["state"], case["questions"], case.get("images", 0)
        assets = case.get("assets", [])
        sequences = []
        composed = bool(assets) or any(q["type"] == "box" for q in questions.values())
        rng, answers, bounds = random.Random(case["name"]), [], []
        decisions = sum(q["type"] in dr.DECISION_KINDS for q in questions.values())
        if composed:
            refs = dc.check_references(assets)
            hoisted, uses = dc.hoist_state(state, refs, front)
            for question in questions.values():
                parts = dc.user_parts(hoisted, question, codes, front + len(uses), 0, refs)
                box = question["type"] == "box"
                content = [{"type": "text", "text": item} if kind == dc.TEXT else {"type": "image"} for kind, item in parts]
                text = chat(tokenizer, [{"role": "system", "content": df.SYSTEM_PROMPT}, {"role": "user", "content": content}])
                text += dc.suffix(False) if box else ""
                enc = encode(tokenizer, text)
                ids = enc["input_ids"]
                trunk = df.state_token_count(enc["offset_mapping"], df.state_end_char(text, hoisted))
                readouts = ([len(ids) - 1 - o for o in dc.readout_offsets(tokenizer, text, False)] if box else [len(ids) - 1])
                sequence = {"tokens": ids, "images": text.count("<|image_pad|>"), "trunk": trunk, "readouts": readouts}
                if box:
                    drawn = [edges(rng) for _ in readouts]
                    logits = torch.tensor([row for _, row in drawn], dtype=torch.float32)
                    logits = logits.masked_fill(torch.arange(len(code_ids))[None] >= dc.BINS, float("-inf"))
                    values = [row[:dc.BINS] for row in (logits / temperature).softmax(-1).tolist()]
                    decoded = dc.decode(values)
                    sequence["values"] = [described for described, _ in drawn]
                    answers.append({"type": "box", "box": decoded,
                                    "confidence": dc.calibrate(dc.confidence(values, decoded), calibration)})
                    bounds.append(None)
                else:
                    values, (answer, bound) = decision(question, decisions, rng)
                    sequence["values"] = [values]
                    answers.append(answer)
                    bounds.append(bound)
                sequences.append(sequence)
        else:
            header = [normalized(q) for q in questions.values()] if not front else None
            for question in questions.values():
                rendered = normalized(question)
                if front:
                    text = chat(tokenizer, [{"role": "system", "content": df.SYSTEM_PROMPT},
                                            {"role": "user", "content": [{"type": "image"}] * front
                                             + [{"type": "text", "text": df.user_prompt(state, rendered, codes)}]}])
                else:
                    text = df.render(tokenizer, state, rendered, codes, header)
                enc = encode(tokenizer, text)
                ids = enc["input_ids"]
                trunk = df.state_token_count(enc["offset_mapping"], df.state_end_char(text, state))
                values, (answer, bound) = decision(question, decisions, rng)
                sequences.append({"tokens": ids, "images": front, "trunk": trunk, "readouts": [len(ids) - 1],
                                  "values": [values]})
                answers.append(answer)
                bounds.append(bound)
        out["cases"].append({"name": case["name"], "shared": sequences[0]["trunk"], "sequences": sequences,
                             "answers": answers, "bounds": bounds})
    return out


def qev(reference, tokenizer, requests):
    sys.path.insert(0, reference)
    import random

    from qev import prompt as qp
    from qev.answers import _normalize
    from qev.calibrate import Calibration, softmax
    from qev.schema import SystemOneRequest

    def one(label):
        enc = tokenizer.encode(label, add_special_tokens=False)
        return enc[0] if len(enc) == 1 and tokenizer.decode(enc) == label else None

    slots = [one(label) for label in qp.SLOT_LABELS]
    assert all(slot is not None for slot in slots)
    out = {"slots": slots, "cases": []}
    # `DecisionEngine` without a `calibration.json`, which neither Base Labs checkpoint ships.
    calibration = Calibration()

    def decision(r, rng):
        """A question's synthetic slot logits and `DecisionEngine.decide`'s answer to them (one branch, `debias` 1):
        the reference's own float64 `softmax` through its temperature and `_normalize`; `make_noul`'s clamp of
        P(true) (option 0) and `make_choice`'s argmax are restated here, because those functions round their
        output to six places, which would hide the arithmetic the bounds measure."""
        values = slot_values(rng, r.n_slots)
        logits = synthetic(values, r.n_slots)
        temperature = calibration.temperature(r.kind, r.n_slots)
        probs = softmax(logits, temperature)
        bounds = softmax_bounds(logits, temperature, f32(temperature), F64)
        if r.kind == "noul":
            return values, typesafe({"type": "noul", "noul": min(1.0, max(0.0, probs[0]))}, bounds[:1])
        p = _normalize(probs)
        if r.kind == "choice":
            best = max(range(len(p)), key=p.__getitem__)
            reference = {"type": "choice", "choice": r.labels[best], "probabilities": dict(zip(r.labels, p))}
        else:
            reference = {"type": "score", "probabilities": {str(i): pi for i, pi in enumerate(p)}}
        return values, typesafe(reference, bounds)

    for case in requests:
        rng = random.Random(case["name"])
        request = SystemOneRequest(state=case["state"], questions=case["questions"])
        rendered = [qp.render_question(k, q) for k, q in request.questions.items()]

        def render(suffix):
            return chat(tokenizer, qp.build_messages(request.state, suffix))

        # DecisionEngine._prefix_and_suffix_ids and _split.
        full = [tokenizer.encode(render(r.suffix), add_special_tokens=False) for r in rendered]
        prefix_text = render("QEV_PREFIX_MARK")
        prefix_ids = tokenizer.encode(prefix_text[: prefix_text.index("QEV_PREFIX_MARK")], add_special_tokens=False)
        n = len(prefix_ids)
        while n > 0 and not all(ids[:n] == prefix_ids[:n] for ids in full):
            n -= 1
        sequences, answers, bounds = [], [], []
        for r, ids in zip(rendered, full):
            # readout.token_layout (which imports torch): tokens ending by the suffix start.
            prompt = render(r.suffix)
            start = prompt.rindex(r.suffix)
            enc = encode(tokenizer, prompt)
            assert enc["input_ids"] == ids and bisect.bisect_right([e for _, e in enc["offset_mapping"]], start) == n
            values, (answer, bound) = decision(r, rng)
            sequences.append({"tokens": ids, "images": 0, "trunk": n, "readouts": [len(ids) - 1],
                              "logits": [slots[: r.n_slots]], "values": [values]})
            answers.append(answer)
            bounds.append(bound)
        out["cases"].append({"name": case["name"], "shared": n, "sequences": sequences, "answers": answers,
                             "bounds": bounds})
    return out


def pplx(reference, tokenizer, requests):
    sys.path.insert(0, reference)
    import random

    from pplx_decider import protocol
    from pplx_decider.model import Decider

    config = json.load(open(f"{sys.argv[3]}/decision_config.json"))
    scale = float(config["temperature"])  # `Decider`: the saved temperature

    class Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def apply_chat_template(self, messages, **kwargs):
            return tokenizer.apply_chat_template(messages, **kwargs)

    encoder = protocol.Encoder(Processor(tokenizer), config["codes"])
    out = {"codes": config["token_ids"], "cases": []}
    for case in requests:
        request = {"state": case["state"], "questions": case["questions"], "images": [None] * case.get("images", 0)}
        encoder.validate(request)
        text = encoder.text(request)
        ids = tokenizer.encode(text, add_special_tokens=False)
        markers = [i for i, token in enumerate(ids) if token == encoder.marker_id]
        assert len(markers) == len(case["questions"])
        # Each marker's synthetic readout rows, the question's options read from the first ones, and
        # `Decider._respond`'s answers to them: a float32 softmax through the temperature, then `answer`.
        rng = random.Random(case["name"])
        values = {key: slot_values(rng, len(protocol.options(q)[0])) for key, q in case["questions"].items()}
        logits = {key: synthetic(values[key], len(protocol.options(q)[0])) for key, q in case["questions"].items()}
        response = Decider._respond(None, request, logits, [], scale)
        answers, bounds = [], []
        for key, question in case["questions"].items():
            bound = softmax_bounds(logits[key], scale, f32(scale), F32)
            # A noul's answer is its option 1, true.
            answer, bound = typesafe(response["answers"][key], bound[1:] if question["type"] == "noul" else bound)
            answers.append(answer)
            bounds.append(bound)
        out["cases"].append({"name": case["name"], "shared": 0, "sequences": [
            {"tokens": ids, "images": case.get("images", 0), "trunk": len(ids), "readouts": markers,
             "values": [values[key] for key in case["questions"]]}], "answers": answers, "bounds": bounds})
    return out


# The text each fixture read points at (by case and question); a read without one points at a third of the way
# into the state, or at nothing.
D1_TARGETS = {
    ("d1_multiple_and_reads", "sender"): "Acme GmbH",
    ("d1_multiple_and_reads", "total"): "1.234,50 €",
    ("d1_multiple_and_reads", "due"): "2026-05-01 9:30",
    ("d1_reads_json_state", "name"): 'Zoë \\"Z\\" Smith',
    ("d1_reads_json_state", "since"): "20 octobre 2019",
    ("d1_reads_json_state", "amount"): "$12.5k",
    ("d1_reads_across_packs", "when"): "12 March 2026",
    ("d1_reads_across_packs", "parcels"): "3",
}


def d1(reference, tokenizer, requests, numbers=None):
    # The repository's own package: `prompt.py`, `attention.py` (`pack_set`), `pointer.py`, `api.py` and
    # `calibration.py` (torch on CPU, no weights).
    package = types.ModuleType("d1")
    package.__path__ = [reference]
    sys.modules["d1"] = package
    import random

    import torch
    from d1 import prompt as dp
    from d1.api import answer
    from d1.attention import pack_set
    from d1.calibration import Calibration
    from d1.pointer import SPAN_CAPS, spans, state_positions

    # What d1-2B's config records (`runner._serve`): codes scheme 2, numbers, prompt 1, median box edges;
    # questions packed `d1_set_size` after their state under `set` attention. The LFM3 lead (`lfm3.LEADS`).
    config = json.load(open(f"{reference}/config.json"))
    # `numbers` False reads grids and box edges as codes, as a checkpoint without `d1_numbers` does.
    numbers = config["d1_numbers"] if numbers is None else numbers
    tokenizer.d1_codes, tokenizer.d1_numbers, tokenizer.d1_prompt = config["d1_codes"], numbers, 1
    tokenizer.d1_box_edges = "median"
    assert config["d1_attention"] == "set"
    set_size, lead, bos = config["d1_set_size"], "<think></think>", tokenizer.bos_token
    markup = "<|vision_start|><|image|><|vision_end|>"  # `media.template_markup`, the image unexpanded
    calibration = Calibration.load(f"{reference}/calibration.json")
    vocabulary = len(tokenizer)

    def describe(rng, slot, groups, k):
        """Peaked over a grid's values, yes falling block by block for a presence, else spread."""
        seed = rng.randrange(1 << 20)
        count = sum(map(len, groups))
        if isinstance(slot, dp.Number) or len(groups) > 26:
            return {"peak": [f32(rng.uniform(0.1, 0.9) * count), f32(rng.uniform(1.0, count / 6))], "seed": seed}
        if isinstance(slot, dp.Noul):
            return {"presence": f32(3.0 - 1.2 * k), "yes": len(groups[0]), "seed": seed}
        return {"seed": seed}

    tables = {}  # each distinct logits token list once (a box's edges read 2,000 tokens a slot)
    out = {"set_size": set_size, "numbers": numbers, "temperature": calibration.default,
           "tables": calibration.tables, "cases": []}
    for case in requests:
        rng = random.Random(case["name"])
        state = case["state"]
        for q in case["questions"].values():
            dp.check_question(q, tokenizer)
        questions = [dp.as_question(q) for q in case["questions"].values()]
        prefix = dp.prefix_text(tokenizer, state, bos, media=markup * case.get("images", 0))
        header = dp.question_header(tokenizer, state)
        layouts = [dp.suffix_layout(tokenizer, q, lead) for q in questions]
        suffixes, offsets = [text for text, _ in layouts], [slots for _, slots in layouts]
        sequences, probabilities, softmaxed = [], {}, {}
        for k in range(0, len(questions), set_size):
            # `runner._sequences` under `set`: the first question as its lone prompt, then `pack_set`.
            text = prefix + suffixes[k]
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            packed = questions[k:k + set_size]
            ids, state_length, _, ends = pack_set(tokenizer, prefix, suffixes[k:k + set_size], header,
                                                  offsets[k:k + set_size], first_ids=encoded["input_ids"])
            groups = [len(encoded["input_ids"]) - state_length]
            groups += [len(tokenizer.encode(header + suffix, add_special_tokens=False))
                       for suffix in suffixes[k + 1:k + set_size]]
            positions = (state_positions(tokenizer, state, prefix, text, encoded=encoded)
                         if any(isinstance(q, dp.Read) for q in packed) else None)
            readouts, logits, scores, values = [], [], [], []
            names = list(case["questions"])[k:k + set_size]
            for j, q in enumerate(packed):
                key = k + j
                if isinstance(q, dp.Read):
                    # The pointer's start and end scores over 0 and the state's positions, each slot read as two
                    # row ranges (0, then the state's positions); found unless position 0 leads the start.
                    mask = positions.pointer_mask
                    after = [i for i, valid in enumerate(mask) if valid and i > 0]
                    start, end = ([f32(rng.gauss(0, 1)) for _ in mask] for _ in range(2))
                    target = D1_TARGETS.get((case["name"], names[j]))
                    nothing = not after or target is None and rng.random() < 0.25
                    start[0] = f32(5.0 if nothing else -4.0)
                    if target is not None:
                        # The positions whose text overlaps the target's first and last characters.
                        at = positions.text.index(target)
                        spans_at = [(i, positions.offsets[i]) for i in after]
                        first = min(i for i, (a, b) in spans_at if b > at)
                        last = max(i for i, (a, b) in spans_at if a < at + len(target))
                        start[first] = end[last] = f32(6.0)
                    elif after:
                        first = after[len(after) // 3]
                        start[first] = end[[i for i in after if i >= first][:4][-1]] = f32(6.0)
                    for row, at in zip((start, end), ends[j]):
                        readouts.append(at)
                        logits.append(None)
                        scores.append([0, 1])
                        values.append(row[:1])
                        if after:
                            readouts.append(at)
                            logits.append(None)
                            scores.append([after[0], len(mask)])
                            values.append(row[after[0]:])
                    rows = torch.tensor([start, end], dtype=torch.float32).masked_fill(~torch.tensor(mask)[None], -torch.inf)
                    cap = SPAN_CAPS[q.type]
                    at, found = spans(rows[None], max_tokens=cap)
                    probabilities[key] = {**positions.read(*at[0].tolist(), max_tokens=cap), "found": found[0].item()}
                    continue
                typed = dp.typed(q)
                for index, (slot, at) in enumerate(zip(dp.question_slots(q, tokenizer), ends[j])):
                    groups_ids = dp.readout_ids(tokenizer, slot)
                    tokens = [t for group in groups_ids for t in group]
                    described = describe(rng, slot, groups_ids, index // 5)
                    row = synthetic(described, len(tokens))
                    readouts.append(at)
                    logits.append(tables.setdefault(tuple(tokens), len(tables)))
                    scores.append(None)
                    values.append(described)
                    logz = torch.full((vocabulary,), -1e4, dtype=torch.float32)
                    logz[torch.tensor(tokens)] = torch.tensor(row, dtype=torch.float32)
                    p = dp.readouts(tokenizer, [slot], logz[None], [None if typed else calibration])[0]
                    if isinstance(q, dp.Choice) and q.multiple:
                        probabilities.setdefault(key, []).append(p[0])
                    elif isinstance(q, (dp.Box, dp.Boxes)):
                        probabilities.setdefault(key, []).append(p)
                    else:
                        probabilities[key] = p
                    if isinstance(q, (dp.Noul, dp.Score)) or isinstance(q, dp.Choice) and not q.multiple:
                        # `readouts`' float64 softmax: each option's best form, through the temperature.
                        softmaxed[key] = ([logz[list(group)].max().item() for group in groups_ids],
                                          1.0 if typed else calibration.temperature(q))
            sequences.append({"tokens": ids, "images": case.get("images", 0), "trunk": state_length,
                              "groups": groups, "readouts": readouts, "logits": logits, "scores": scores,
                              "values": values})
        context = dp.state_text(state)  # `SystemOneApi.answer`'s, for a read
        answers, bounds = [], []
        for i, q in enumerate(questions):
            reference = answer(q, probabilities[i], "median", None, context if isinstance(q, dp.Read) else "",
                               calibration, "median")
            if i in softmaxed:
                logits, temperature = softmaxed[i]
                bound = softmax_bounds(logits, temperature, f32(temperature), F64)
                # A noul's answer is its option 0, yes.
                reference, bound = typesafe(reference, bound[:1] if isinstance(q, dp.Noul) else bound)
            else:
                bound = None
            answers.append(reference)
            bounds.append(bound)
        out["cases"].append({"name": case["name"], "shared": 0, "sequences": sequences, "answers": answers,
                             "bounds": bounds})
    out["token_lists"] = [list(tokens) for tokens in tables]
    return out


def main():
    kind, reference, tokenizer_dir, requests, target = sys.argv[1:6]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)
    cases = [case for case in json.load(open(requests)) if kind in case.get("types", [kind])]
    kinds = {"d3": d3, "qev": qev, "pplx": pplx, "d1": d1, "d1-codes": lambda *a: d1(*a, numbers=False)}
    fixture = kinds[kind](reference, tokenizer, cases)
    with open(target, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
