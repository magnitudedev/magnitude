# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy~=2.2.6",
#   "protobuf>=4.21.0,<5.0.0",
#   "safetensors>=0.5",
#   "sentencepiece>=0.1.98,<0.3.0",
#   "torch==2.11.0",
#   "transformers==5.19.0",
# ]
# ///
"""Convert a decision model's safetensors checkpoint into Magnitude's decision GGUF.

The base architecture is written by llama.cpp's own converter (`conversion/`) at the pinned
revision, as every `magnitudedev` conversion is; this adds only what makes the GGUF a decision
model: its decision description, which is everything the engine does for it. That is the decision
template `tokenizer.chat_template.decision` (from `decision-templates/`), the description's keys
under `decision.` and `<arch>.attention.causal = false`, and for a model that reads a
separate readout head, the head as its own tensor `decision.readout.weight` ([K, hidden], in the
reference dtype, never folded into the output projection), whose presence is what makes the
description read option values from it rather than from the LM head. The keys, their values and the template's context are the `decision` crate's
(`engine/decision/KEYS.md`); every key is the checkpoint's own fact.

The model is recognized from the checkpoint's own configuration files, never named on the command
line, and each description refuses a configuration it was not written for. Dependencies are those
of llama.cpp's `requirements-convert_hf_to_gguf.txt` at the pinned revision, except transformers:
llama.cpp's 4.57.6 cannot load the Base Labs tokenizers, which transformers 5 saved
(`Qwen3_5Tokenizer`); 5.19.0 is what Base Labs pin.

    uv run --script inference/catalog/conversions/decision.py convert \\
        --llama-cpp LLAMA_CPP_CHECKOUT --checkpoint CHECKPOINT_DIR \\
        --output OUTPUT_DIR --name elephant-1 [--outtype bf16]

writes OUTPUT_DIR/NAME-<OUTTYPE>.gguf and, for a model that reads images, its projector as llama.cpp
writes it, OUTPUT_DIR/mmproj-NAME-<OUTTYPE>.gguf. `describe --checkpoint CHECKPOINT_DIR --output FILE` writes
the description alone as JSON (`template`, `keys` each with its GGUF `type` and `value`, and `head`, the
readout head's rows the conversion stores, or null), the form the decision crate's parity tests read; it needs neither llama.cpp nor the weights, only the
standard library. Quantizing a GGUF with a readout head keeps the head in its reference dtype:
`llama-quantize --tensor-type decision.readout=<f32|bf16> IN OUT Q4_K_M`.

    uv run --script inference/catalog/conversions/decision.py rename-namespace \\
        --llama-cpp LLAMA_CPP_CHECKOUT --input IN.gguf --output OUT.gguf

rewrites a GGUF whose description keys are under `<arch>.decision.`, as conversions wrote them before
the description had a namespace of its own, with each under `decision.` in its place; every other key
and every tensor byte is copied unchanged. It refuses a GGUF with no such key or with a `decision.`
key already, and an output that exists.

A score's level (the expected one) and the choice and score confidences (TypeSafe's) are the wire's, the same
for every model, and so no key.

Described: Base Labs mammoth-1 and elephant-1 (OneJev `qev`), Perplexity pplx-decider Flash v1.2,
vLLM-SR d3-ultra and Liquid d1-2B.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable

LLAMA_CPP_REVISION = "8ca4060b7e30e76baaa5670ee1ec1b760b7f8552"
READOUT = "decision.readout.weight"
# The namespace of the description's keys: the description is the same for every architecture.
NAMESPACE = "decision."  # the engine's `magnitude_artifacts::DECISION_NAMESPACE` (KEYS.md)
TEMPLATE = "tokenizer.chat_template.decision"
TEMPLATES = Path(__file__).resolve().parent / "decision-templates"
OUTTYPES = ("f32", "f16", "bf16", "q8_0")
# A key's GGUF type: llama.cpp's keys (`temperature.*`, `labels`) as llama.cpp defines them, floats
# f32; every other float f64, the precision the references read their parameters in.
KEY_TYPES = ("str", "bool", "i32", "f32", "f64", "[str]", "[f64]")


def read_json(checkpoint: Path, name: str) -> dict[str, Any]:
    with open(checkpoint / name, encoding="utf-8") as file:
        return json.load(file)


def require(condition: bool, message: str, subject: str = "checkpoint") -> None:
    if not condition:
        raise SystemExit(f"unsupported {subject}: {message}")


@dataclass(frozen=True)
class Key:
    type: str
    value: Any


@dataclass(frozen=True)
class Description:
    """A decision model's description: its template, its keys under `decision.`, and the rows of the
    readout head `decision.readout.weight` its GGUF stores (one per code), or None for one read at the LM head."""

    template: str
    keys: dict[str, Key]
    head: int | None = None

    def __post_init__(self) -> None:
        for name, key in self.keys.items():
            require(key.type in KEY_TYPES, f"`{name}` has no GGUF type {key.type!r}")

    def as_json(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "keys": {name: {"type": key.type, "value": key.value} for name, key in self.keys.items()},
            "head": self.head,
        }


@dataclass(frozen=True)
class DecisionModel:
    """One described checkpoint as a conversion writes it."""

    name: str
    # The readout head's reference dtype (`F32` or `BF16`), for a model that reads one (`Description.head`).
    readout: str | None
    describe: Callable[[Path], Description]
    # llama.cpp's projector class (`module.Class` under `conversion/`) for a model that reads images.
    projector: str | None = None
    # llama.cpp's text model class (`module.Class` under `conversion/`), and whether its checkpoint's hparams
    # are read by the loader llama.cpp registers for it rather than as its config names them.
    text: str = "qwen.Qwen3_5TextModel"
    registered_hparams: bool = False


def template(name: str) -> str:
    return (TEMPLATES / f"{name}.jinja").read_text(encoding="utf-8")


# OneJev `qev` 81ce62f, qev/prompt.py: the slot labels prompt `qev-labels-v2` lists options under,
# A..Z then the two-letter labels the Qwen3.5 tokenizer holds as single tokens.
QEV_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
QEV_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK",
                     "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
QEV_SLOT_LABELS = list(QEV_LETTERS) + [
    a + b for a in QEV_LETTERS for b in QEV_LETTERS if a + b not in QEV_NOT_ONE_TOKEN
][: 256 - len(QEV_LETTERS)]


def qev_describe(checkpoint: Path) -> Description:
    """Base Labs mammoth-1 and elephant-1, as OneJev `qev` 81ce62f with Base Labs' `serve.patch` serves
    them: prompt `qev-labels-v2`, readout `slot_lm` (each slot label's LM-head logit at the last
    token), attention `noncausal` (the state once, each question a branch)."""
    readout = read_json(checkpoint, "readout.json")
    require(readout.get("readout") == "slot_lm", "qev reads slot labels from the LM head")
    slots = readout.get("n_slots")
    require(isinstance(slots, int) and 0 < slots <= len(QEV_SLOT_LABELS), f"n_slots is 1 to {len(QEV_SLOT_LABELS)}")
    # qev reads a temperature from `calibration.json`; neither Base Labs checkpoint ships one, so its
    # default of 1.0 applies, and a checkpoint that does is a format this does not read yet.
    require(not (checkpoint / "calibration.json").is_file(), "a calibration.json is not read yet")
    keys = {
        "escape.method": Key("str", "none"),
        "segments": Key("str", "whole"),
        "set_size": Key("i32", 1),
        "state_attends": Key("str", "state"),
        "question_attends": Key("str", "question"),
        "labels": Key("[str]", QEV_SLOT_LABELS[:slots]),
        # The options render yes (true) first.
        "noul.codes": Key("str", "sequence"),
        "noul.true_option": Key("i32", 0),
        # qev/schema.py: MAX_CHOICE_OPTIONS and MAX_SCORE_LEVELS.
        "choice.max_options": Key("i32", 255),
        "score.codes": Key("str", "sequence"),
        "score.min_levels": Key("i32", 1),
        "score.max_levels": Key("i32", 10),
    }
    for kind in ("noul", "choice", "score"):
        keys[f"{kind}.slots"] = Key("str", "last")
        keys[f"temperature.{kind}"] = Key("f32", 1.0)
    # The kinds the model was trained to answer.
    keys["kinds"] = Key("[str]", ["noul", "choice", "score"])
    return Description(template("qev-labels-v2"), keys)


def special_tokens(checkpoint: Path) -> list[str]:
    """The tokenizer's `all_special_tokens` as transformers 5 assembles them from `tokenizer_config.json`: its
    named special tokens, then the model's own (`model_specific_special_tokens`) and the additional ones, each
    once, in that order."""
    config = read_json(checkpoint, "tokenizer_config.json")
    named = [config.get(name) for name in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token",
                                           "cls_token", "mask_token")]
    named += list(config.get("model_specific_special_tokens", {}).values())
    named += config.get("additional_special_tokens", [])
    spellings: list[str] = []
    for token in named:
        token = token.get("content") if isinstance(token, dict) else token
        if token and token not in spellings:
            spellings.append(token)
    return spellings


PPLX_MARKER = "<|fim_suffix|>"


def pplx_describe(checkpoint: Path) -> Description:
    """Perplexity pplx-decider Flash v1.2, as its package 9ff7cf9 serves it: protocol 3
    (`joint-question-inline-readout-v1`), the state and every question in one sequence read both ways, the
    readout head applied at each question's `<|fim_suffix|>` marker, request text's special-token spellings
    in full-width brackets (`Encoder._text`)."""
    config = read_json(checkpoint, "decision_config.json")
    joint = read_json(checkpoint, "joint_config.json")
    require("format_id" not in config, "a pplx decision_config names no format_id")
    require(config.get("attention_mode") == "noncausal_full_attention", "pplx reads its sequence both ways")
    require(config.get("pooling") == "last", "pplx reads its readout rows' own states")
    require(
        joint.get("version") == 3
        and joint.get("template") == "joint-question-inline-readout-v1"
        and joint.get("marker") == PPLX_MARKER
        and joint.get("readout_offset") == 0,
        "pplx is joint protocol 3, read at each <|fim_suffix|>",
    )
    codes = config["codes"]
    # protocol.py: MAX_OPTIONS, a score of at least two levels; `Encoder` escapes the marker beside the
    # tokenizer's special tokens, longest first.
    options = min(255, len(codes))
    escapes = special_tokens(checkpoint)
    keys = {
        "escape.method": Key("str", "fullwidth"),
        "escape.spellings": Key("[str]", escapes + [PPLX_MARKER] * (PPLX_MARKER not in escapes)),
        "segments": Key("str", "whole"),
        "set_size": Key("i32", 0),
        "state_attends": Key("str", "set"),
        "question_attends": Key("str", "set"),
        "labels": Key("[str]", list(codes)),
        # The options render false then true.
        "noul.codes": Key("str", "sequence"),
        "noul.true_option": Key("i32", 1),
        "choice.max_options": Key("i32", options),
        "score.codes": Key("str", "sequence"),
        "score.min_levels": Key("i32", 2),
        "score.max_levels": Key("i32", options),
        # Images are read before the state.
        "image_placements": Key("[str]", ["front"]),
    }
    for kind in ("noul", "choice", "score"):
        keys[f"{kind}.slots"] = Key("str", "anchors")
        keys[f"temperature.{kind}"] = Key("f32", float(config["temperature"]))
    # The kinds the model was trained to answer.
    keys["kinds"] = Key("[str]", ["noul", "choice", "score"])
    return Description(template("joint-question-inline-readout-v1"), keys, head=len(codes))


# d3_format.py and d3_compose.py: the readout head's codes, a score's levels, a box's edge bins, and the
# boxes drawn for its confidence.
D3_CODES = 255
D3_LEVELS = (2, 10)
D3_BINS = 128
D3_DRAWS = 512
D3_IOU = 0.5
# `systemone.image_placements` as the engine names them: an image in the state may also come before it.
D3_PLACEMENTS = {"state": ["front", "state"], "instructions": ["instructions"], "criteria": ["options"]}


def d3_describe(checkpoint: Path) -> Description:
    """vLLM Semantic Router d3-ultra, as its runtime 52ba9e7 serves it: format `d3-code-readout-v1` in layout
    `header-v1`, the state once (`prefix_bidir`) and each question a branch read through the F32 readout head
    at its last token, a box at its four edge markers; codes `answer_codes` probes after the assistant turn
    opens, temperatures by question count (`temperature_for`),
    boxes from expected edge bins with R4-drawn IoU confidence through `box_calibration`."""
    config = read_json(checkpoint, "decision_config.json")
    require(config.get("format_id") == "d3-code-readout-v1", "d3 reads d3-code-readout-v1")
    require(config.get("attention_mode") == "prefix_bidir", "d3 reads its prefix both ways")
    require(config.get("pooling") == "last", "d3 reads the last token")
    require(config.get("readout_dtype") == "float32", "d3 reads an F32 head")
    require(config.get("prompt_layout") == "header-v1", "d3 renders the header-v1 layout")
    require(len(config["codes"]) == D3_CODES, f"d3 reads {D3_CODES} codes")
    by_questions = config["temperature_by_questions"]
    bounds, temperatures = by_questions["max_questions"], by_questions["temperature"]
    require(
        len(temperatures) == len(bounds) + 1 and bounds == sorted(set(bounds)),
        "one temperature per question-count bound and one beyond them",
    )
    systemone = config["systemone"]
    # `track` reads video, which no request carries.
    require(set(systemone["question_types"]) <= {"box", "track"}, "d3 knows box and track questions")
    knots = systemone["box_calibration"]
    require(all(len(knot) == 2 for knot in knots), "box calibration knots are (raw, calibrated) pairs")
    placements = [name for placement in systemone["image_placements"] for name in D3_PLACEMENTS[placement]]
    keys = {
        "escape.method": Key("str", "none"),
        "segments": Key("str", "whole"),
        "set_size": Key("i32", 1),
        "state_attends": Key("str", "state"),
        "question_attends": Key("str", "question"),
        # The head's rows are the codes, each kept while it stays one token after the template's probe.
        "codes.sequence": Key("str", "letters"),
        "codes.probe": Key("bool", True),
        # The options render false then true; a score is a choice over its levels.
        "noul.codes": Key("str", "sequence"),
        "noul.true_option": Key("i32", 1),
        "choice.max_options": Key("i32", D3_CODES),
        "score.codes": Key("str", "sequence"),
        "score.min_levels": Key("i32", D3_LEVELS[0]),
        "score.max_levels": Key("i32", D3_LEVELS[1]),
        "image_placements": Key("[str]", placements),
    }
    for kind in ("noul", "choice", "score"):
        keys[f"{kind}.slots"] = Key("str", "last")
        keys[f"temperature.{kind}"] = Key("f32", float(temperatures[-1]))
        for bound, temperature in zip(bounds, temperatures):
            keys[f"temperature.{kind}.{int(bound)}"] = Key("f32", float(temperature))
    if "box" in systemone["question_types"]:
        keys |= {
            "box.presence": Key("bool", False),
            "box.grid": Key("str", "codes"),
            "box.numbers": Key("bool", False),
            "box.bins": Key("i32", D3_BINS),
            "box.positions": Key("str", "centres"),
            "box.edge": Key("str", "mean"),
            "box.points": Key("str", "r4"),
            "box.draws": Key("i32", D3_DRAWS),
            "box.iou": Key("f64", D3_IOU),
            "box.crossed": Key("str", "clamp"),
            "box.drawn": Key("str", "valid"),
            "temperature.box": Key("f32", float(config["temperature"])),
            "calibration.box.raw": Key("[f64]", [float(raw) for raw, _ in knots]),
            "calibration.box.calibrated": Key("[f64]", [float(calibrated) for _, calibrated in knots]),
            "calibration.box.inverted": Key("bool", False),
        }
    # The kinds the model was trained to answer.
    keys["kinds"] = Key("[str]", ["noul", "choice", "score"] + ["box"] * ("box" in systemone["question_types"]))
    return Description(template("d3-header-v1"), keys, head=D3_CODES)


# d1-2B's calibration tables (calibration.py) by the names the description gives them, and whether each is read
# backwards (`Calibration.level`).
D1_TABLES = {
    "multiple": ("multiple", False),
    "found_text": ("found:text", False),
    "found_number": ("found:number", False),
    "found_date": ("found:date", False),
    "found_box": ("found:box", False),
    "box": ("box", False),
    "found_boxes": ("found:boxes", False),
    "boxes": ("boxes", False),
    "estimate": ("estimate", True),
}
# prompt.py and api.py: System One's option bound, a score's 2 to 10 levels, an estimate's 128 values at most;
# pointer.py `SPAN_CAPS`; box.py `INSTANCES` (prompt 1's boxes), `DRAWS`, the IoU a drawn box needs, and an edge's
# grid: thousandths written as numbers, else 128 values posed as a grid question (prompt.py `grid_question`).
D1_OPTIONS = 255
D1_LEVELS = (2, 10)
D1_VALUES = 128
D1_SPANS = {"text": 512, "number": 32, "date": 32}
D1_INSTANCES = 16
D1_DRAWS = 512
D1_IOU = 0.5
D1_EDGES = {True: (1000, "starts"), False: (128, "ends")}


def d1_describe(checkpoint: Path) -> Description:
    """Liquid d1-2B, as its repository 97c5704 serves it (`runner.py`, `prompt.py`, `attention.py`, `pointer.py`,
    `box.py`, `grid.py`, `api.py`): prompt 1 with codes scheme 2, the first question's whole prompt and each further
    one tokenized alone, packed `d1_set_size` after their state, which reads its set while each question reads it and
    itself (`set` attention); each option's best spelling at the LM head; noul, choice and score through the
    calibration's temperature, the typed answers raw through its tables; median box edges."""
    config = read_json(checkpoint, "config.json")
    calibration = read_json(checkpoint, "calibration.json")
    require(config.get("d1_codes") == 2, "d1 renders codes scheme 2")
    require(config.get("d1_attention") == "set", "d1 reads `set` attention")
    require(config.get("d1_prompt", 1) == 1, "d1 renders prompt 1")
    require(config.get("d1_box_edges", "median") == "median", "d1 reads box edges at their median")
    set_size = config.get("d1_set_size")
    require(isinstance(set_size, int) and 2 <= set_size <= 128, "a recorded d1_set_size is 2 to 128")
    numbers = config.get("d1_numbers")
    require(isinstance(numbers, bool), "the checkpoint records d1_numbers")
    max_pixels = config.get("d1_vision_max_pixels")
    require(isinstance(max_pixels, int) and max_pixels > 0, "the checkpoint records d1_vision_max_pixels")
    # `Calibration.temperature`: one temperature for every option count, as no bucket is recorded.
    require(not calibration.get("temperatures"), "per-bucket temperatures are not read")
    tables = calibration.get("tables", {})
    require(set(tables) == {name for name, _ in D1_TABLES.values()},
            f"calibration tables are {sorted(name for name, _ in D1_TABLES.values())}")
    bins, positions = D1_EDGES[numbers]
    keys = {
        # `plain`: a zero-width space after the first character of every special token's spelling longer than one.
        "escape.method": Key("str", "zero_width"),
        "escape.min_length": Key("i32", 2),
        "segments": Key("str", "per_question"),
        "set_size": Key("i32", set_size),
        "state_attends": Key("str", "set"),
        "question_attends": Key("str", "question"),
        "codes.sequence": Key("str", "aliases"),
        "codes.keys_if_letters": Key("bool", True),
        "noul.codes": Key("str", "yes_no"),
        "noul.true_option": Key("i32", 0),
        "choice.max_options": Key("i32", D1_OPTIONS),
        "score.codes": Key("str", "digits"),
        "score.min_levels": Key("i32", D1_LEVELS[0]),
        "score.max_levels": Key("i32", D1_LEVELS[1]),
        "estimate.slots": Key("str", "last"),
        "estimate.max_values": Key("i32", D1_VALUES),
        # `grid_question`: a grid is a score over its values when it has at most 10 (a score's most), else a
        # choice; `grid_slot`: the tokenizer's numbers instead where the checkpoint reads them so.
        "estimate.grid": Key("str", "levels"),
        "estimate.numbers": Key("bool", numbers),
        "box.presence": Key("bool", True),
        "box.grid": Key("str", "levels"),
        "box.numbers": Key("bool", numbers),
        "box.bins": Key("i32", bins),
        "box.positions": Key("str", positions),
        "box.edge": Key("str", "median"),
        "box.points": Key("str", "halton"),
        "box.draws": Key("i32", D1_DRAWS),
        "box.iou": Key("f64", D1_IOU),
        "box.crossed": Key("str", "shift"),
        "box.drawn": Key("str", "ordered"),
        "boxes.max": Key("i32", D1_INSTANCES),
        "image_placements": Key("[str]", ["front"]),
        # The runner's image cap; the projector carries the processor's larger one.
        "image_max_pixels": Key("i32", max_pixels),
    }
    for kind, slots in (("noul", "last"), ("choice", "last"), ("score", "last")):
        keys[f"{kind}.slots"] = Key("str", slots)
        keys[f"temperature.{kind}"] = Key("f32", float(calibration["default"]))
    # The typed answers read their slots raw.
    for kind in ("multiple", "estimate", "box", "boxes"):
        keys[f"temperature.{kind}"] = Key("f32", 1.0)
    for value, most in D1_SPANS.items():
        keys[f"read.{value}.max_tokens"] = Key("i32", most)
    for name, (table, inverted) in D1_TABLES.items():
        knots = tables[table]
        require(all(len(knot) == 2 for knot in knots), f"`{table}` knots are (raw, calibrated) pairs")
        keys[f"calibration.{name}.raw"] = Key("[f64]", [float(raw) for raw, _ in knots])
        keys[f"calibration.{name}.calibrated"] = Key("[f64]", [float(value) for _, value in knots])
        keys[f"calibration.{name}.inverted"] = Key("bool", inverted)
    # The kinds the model was trained to answer.
    keys["kinds"] = Key("[str]", ["noul", "choice", "multiple", "score", "read", "estimate", "box", "boxes"])
    return Description(template("d1-prompt-v1"), keys)


MODELS = (
    DecisionModel("qev", readout=None, describe=qev_describe),
    DecisionModel("pplx", readout="BF16", describe=pplx_describe, projector="pplx_decider.PplxDeciderVisionModel"),
    DecisionModel("d3", readout="F32", describe=d3_describe, projector="vllm_d3.VllmD3VisionModel"),
    # d1 decides on LFM3-VL's own weights and vocabulary, which llama.cpp's LFM3 class writes from the hparams its
    # LFM3 d1 loader reads; `LFM3D1Model` would add llama.cpp's own d1 type and template, which the description
    # replaces.
    DecisionModel("d1", readout=None, describe=d1_describe, projector="lfm3.LFM3VLMmprojModel",
                  text="lfm3.LFM3Model", registered_hparams=True),
)


def recognize(checkpoint: Path) -> DecisionModel:
    """The one described model whose configuration files the checkpoint carries."""
    def has(*names: str) -> bool:
        return all((checkpoint / name).is_file() for name in names)

    def format_id() -> str | None:
        return read_json(checkpoint, "decision_config.json").get("format_id")

    def d1() -> bool:
        config = read_json(checkpoint, "config.json")
        return config.get("architectures") == ["D1Model"] and config.get("model_type") == "lfm3_vl"

    matches = [
        model
        for model, present in (
            (MODELS[0], has("readout.json")),
            (MODELS[1], has("decision_config.json", "joint_config.json", "readout.safetensors") and format_id() is None),
            (MODELS[2], has("decision_config.json", "readout.safetensors") and format_id() is not None),
            (MODELS[3], has("config.json", "calibration.json") and d1()),
        )
        if present
    ]
    require(len(matches) == 1, f"recognized as {[model.name for model in matches]}")
    return matches[0]


def import_gguf(root: Path):
    """gguf-py from the llama.cpp checkout `root`, at the pinned revision."""
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    require(revision == LLAMA_CPP_REVISION, f"llama.cpp is at {revision}, not {LLAMA_CPP_REVISION}")
    sys.path[:0] = [str(root / "gguf-py"), str(root)]
    import gguf  # noqa: PLC0415

    return gguf


def import_llama_cpp(root: Path):
    gguf = import_gguf(root)
    import importlib  # noqa: PLC0415

    from conversion import ModelBase  # noqa: PLC0415

    def named(written: str):
        module, name = written.rsplit(".", 1)
        return getattr(importlib.import_module(f"conversion.{module}"), name)

    return gguf, ModelBase, named


def text_model(gguf, base, model: DecisionModel, description: Description, checkpoint: Path):
    class DecisionTextModel(base):
        model_arch = base.model_arch
        # None of the checkpoints has an MTP head, and a decision model generates nothing.
        no_mtp = True

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            writer = self.gguf_writer
            # Every described layout reads its global attention layers both ways.
            writer.add_causal_attention(False)
            writer.add_string(TEMPLATE, description.template)
            for key, typed in description.keys.items():
                name = f"{NAMESPACE}{key}"
                {
                    "str": writer.add_string,
                    "bool": writer.add_bool,
                    "i32": writer.add_int32,
                    "f32": writer.add_float32,
                    "f64": writer.add_float64,
                    "[str]": writer.add_array,
                    "[f64]": lambda name, value: writer.add_key_value(
                        name, value, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.FLOAT64
                    ),
                }[typed.type](name, typed.value)

        def generate_extra_tensors(self) -> Iterable:
            yield from super().generate_extra_tensors()
            require((model.readout is None) == (description.head is None),
                    "a readout head's dtype is known exactly when the description reads one")
            if model.readout is None:
                return
            from safetensors.torch import load_file  # noqa: PLC0415

            head = load_file(self.dir_model / "readout.safetensors")["weight"]
            hidden = self.find_hparam(["hidden_size"])
            require(str(head.dtype) == {"F32": "torch.float32", "BF16": "torch.bfloat16"}[model.readout],
                    f"the readout head is {head.dtype}, not {model.readout}")
            rows = description.head
            require(list(head.shape) == [rows, hidden],
                    f"the readout head is {list(head.shape)}, not one row of width {hidden} per code")
            yield READOUT, head

        def modify_tensors(self, data_torch, name: str, bid: int | None):
            if name == READOUT:
                return [(name, data_torch)]
            return super().modify_tensors(data_torch, name, bid)

        def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
            if new_name == READOUT:
                return getattr(gguf.GGMLQuantizationType, model.readout)
            return super().tensor_force_quant(name, new_name, bid, n_dims)

    return DecisionTextModel


def rename_namespace(gguf, source: Path, output: Path) -> int:
    """Write `source` to `output` with each description key under `<arch>.decision.` under `decision.` in its
    place, every other key and every tensor byte as `source` has them; the number of keys renamed. `source`
    must be this converter's output, which carries the decision template: llama.cpp's own decision GGUFs
    use `<arch>.decision.` for llama.cpp's keys and are refused."""
    require(not output.exists(), f"{output} exists", "output")
    reader = gguf.GGUFReader(source, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    old = f"{arch}.{NAMESPACE}"
    # The reader's `GGUF.*` fields are the header's counts and version, which the writer derives.
    fields = [field for field in reader.fields.values() if not field.name.startswith("GGUF.")]
    require(not any(field.name.startswith(NAMESPACE) for field in fields), f"it has `{NAMESPACE}` keys already", "GGUF")
    renamed = sum(field.name.startswith(old) for field in fields)
    require(renamed > 0, f"it has no `{old}` keys", "GGUF")
    require(reader.get_field(TEMPLATE) is not None, f"it has no `{TEMPLATE}`: not a decision conversion", "GGUF")
    writer = gguf.GGUFWriter(output, arch, endianess=reader.endianess)
    # Every key in its place: the writer adds the architecture first, and the source places it.
    writer.remove_key(gguf.Keys.General.ARCHITECTURE)
    for field in fields:
        name = NAMESPACE + field.name.removeprefix(old) if field.name.startswith(old) else field.name
        kind = field.types[0]
        writer.add_key_value(name, field.contents(), kind,
                             sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    writer.data_alignment = reader.alignment
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="write the decision GGUF")
    convert.add_argument("--llama-cpp", type=Path, required=True, help=f"a checkout at {LLAMA_CPP_REVISION}")
    convert.add_argument("--checkpoint", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--name", required=True, help="the output file's base name")
    convert.add_argument("--outtype", choices=OUTTYPES, default="bf16")
    describe = commands.add_parser("describe", help="write the decision description as JSON")
    describe.add_argument("--checkpoint", type=Path, required=True)
    describe.add_argument("--output", type=Path, required=True)
    rename = commands.add_parser("rename-namespace",
                                 help=f"move a GGUF's description keys from `<arch>.{NAMESPACE}` to `{NAMESPACE}`")
    rename.add_argument("--llama-cpp", type=Path, required=True, help=f"a checkout at {LLAMA_CPP_REVISION}")
    rename.add_argument("--input", type=Path, required=True)
    rename.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "rename-namespace":
        renamed = rename_namespace(import_gguf(args.llama_cpp.resolve()), args.input, args.output)
        print(f"renamed {renamed} keys: wrote {args.output}")
        return

    model = recognize(args.checkpoint)
    description = model.describe(args.checkpoint)
    if args.command == "describe":
        args.output.write_text(json.dumps(description.as_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{model.name}: wrote {args.output}")
        return
    gguf, model_base, named = import_llama_cpp(args.llama_cpp.resolve())
    ftype = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }[args.outtype]
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = args.outtype.upper()
    # The hparams as the config names them (Qwen3.5's), or as llama.cpp's loader for the checkpoint reads them.
    hparams = model_base.load_hparams(args.checkpoint, False, guess=model.registered_hparams)
    outputs = [(text_model(gguf, named(model.text), model, description, args.checkpoint),
                args.output / f"{args.name}-{suffix}.gguf")]
    if model.projector is not None:
        outputs.append((named(model.projector), args.output / f"mmproj-{args.name}-{suffix}.gguf"))
    for model_class, path in outputs:
        model_class(args.checkpoint, ftype, path, hparams=dict(hparams)).write()
        print(f"{model.name}: wrote {path}")


if __name__ == "__main__":
    main()
