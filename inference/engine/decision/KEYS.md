# Decision description

A decision model is described entirely by its GGUF, and the engine has no
per-model code. The description is the decision template
`tokenizer.chat_template.decision` and the keys under `decision.` (read
with that prefix removed), beside `<arch>.attention.causal = false`: every
implemented layout reads its global attention layers both ways. The
description is the same for every architecture, so its keys are a
namespace of their own beside `general.*` and `tokenizer.*`, which no
family reads; the causality flag is a fact of the architecture and stays in
its namespace. A GGUF with decision keys and no
template, or a non-causal one without a template, is refused as it is read
(`magnitude-artifacts` `DecisionMetadata::of`); a causal one with a
template, a missing key, a key nothing reads and a value naming a mechanism
the engine lacks are each refused naming the key (`src/description.rs`,
`src/metadata.rs`). Whether option values are read from the readout head
`decision.readout.weight` is whether the GGUF stores that tensor: a slot's
code is then the row of its index (the head has a row for every code, or the
model is refused), and otherwise the LM head's row of its token. Our
conversion (`catalog/conversions/decision.py`) is the one writer of
descriptions; its `describe` command emits one as JSON for the parity
tests, with the rows of the head the conversion stores.

**A value is a description key if and only if it is a property of the model: trained, fitted, or part of the reference computation that a fitted value of the model depends on (e.g. the box pipeline a box calibration table was fitted on). A value defined by the wire contract or the same for any model is engine code, never a key.**

A mechanism a key names is defined in full by its value (`r4` is the R4
sequence from 1/2 at indices 1.., `halton` the Halton sequence in bases 2,
3, 5 and 7, `letters` and `aliases` their candidate orders, `yes_no` its
spellings), so a model differing in any part of it names another value. A
model's kinds are its `kinds`, the list of those it was trained to
answer. Engine code that is a rule rather than a model's
property: one half as the line a calibrated probability (present, found, an
option of a multiple choice) is taken at, an estimate's median and 10th and
90th percentiles (the answer's shape; `calibration.estimate` gives the
levels they are read at), and the wire formats' request limits.

What is the same for every decision model (softmax, code selection, and
what the wire defines: a score's expected level and TypeSafe's choice and
score confidences) is engine code, and what is derivable from
the GGUF (vocabulary, token attributes, control-token spellings) is never a
key. Floats are f32 for llama.cpp's keys (`temperature.*`), f64 otherwise.

## Keys

| Key | Value | Meaning |
| --- | --- | --- |
| `escape.method` | `none`, `fullwidth`, `zero_width` | How request text spelling one of the tokenizer's control tokens is changed before rendering: `＜`/`＞` for `<`/`>`, or a zero-width space after the first character |
| `escape.min_length` | integer > 0, with `fullwidth` and `zero_width` | The tokenizer's control-token spellings at least this many characters long |
| `escape.spellings` | strings, in place of `escape.min_length` | These spellings, where the reference lists its own rather than the tokenizer's control tokens |
| `segments` | `whole`, `per_question` | The prompt is tokenized as one text; or the first question's whole prompt (through where question 1 starts) as one text and each further question's text alone, the state being the first prompt less the tokens of the first question's text alone |
| `set_size` | integer ≥ 0 | Questions laid out after one state in one sequence; 0 for all |
| `state_attends` | `state`, `set` | The state's rows read, both ways on global attention layers, themselves or also every question of their set. A state that reads only itself is computed once, shared by every set and retained across requests |
| `question_attends` | `question`, `set` | A question's rows read the state and themselves, or the state and every question of their set |
| `labels` | strings | The option codes, each one token of its own, in order |
| `codes.sequence` | `letters`, `aliases`, without `labels` | Codes `A`..`Z` then `AA`.., each kept when it is one token distinct from the codes before it; or (`aliases`, without a readout head) each option's code by position, kept when it is one token no earlier option took, else the first such code of the fallback pool (`A`..`Z`, `00`..`99`, `a`..`z`, `#0`..`#199`, `AA`..`ZZ`), each also read after a space when that is another single token |
| `codes.count` | integer > 0, with `letters` and no readout head | How many letter codes; with a head, its rows |
| `codes.probe` | bool, with `letters` | Whether a code must also stay that one token after the template's probe: what it renders for the context `{"probe": true}` (template text, no anchor) |
| `codes.keys_if_letters` | bool, with `aliases` | Whether a choice whose keys, stripped, are each one letter takes its keys as its codes in place of their positions |
| `noul.codes` | `sequence`, `yes_no` | A noul's two options read at the first two codes, or (no readout head) at `yes` and `no` in their lower, capitalized and upper spellings that are one token |
| `score.codes` | `sequence`, `digits` | A score's levels read at the codes, or (no readout head) at `0`, `1`, .. each one token |
| `<kind>.grid` | `codes`, `levels` (`estimate`, `box`) | How a grid's values are posed: as a choice's options, at the codes by position; or (with the kind `score`) as a score's levels when the model scores that many, else as `codes` |
| `<kind>.numbers` | bool (`estimate`, `box`), true only without a readout head | Whether each value is read as the integer it is, plain and after a space, when every value is an integer one token writes (else as `<kind>.grid` poses it); a box's bin `k` as `k` |
| `kinds` | strings among `noul`, `choice`, `multiple`, `score`, `read`, `estimate`, `box`, `boxes`, distinct, at least one | The kinds the model was trained to answer: a kind's other keys are required when it is listed and refused when it is not |
| `<kind>.slots` | `last`, `anchors` (`noul`, `choice`, `score`, `estimate`) | Where a kind of one slot is read: the last token of its rows, or the token its slot anchor ends; every other kind is read at its slot anchors |
| `noul.true_option` | `0` or `1` | Which of the template's two noul options is true |
| `choice.max_options` | integer > 0, at most the codes | The most options a choice reads |
| `score.min_levels`, `score.max_levels` | integers > 0, at most the codes | How many levels a score reads |
| `temperature.<kind>` | float > 0 | The softmax temperature of every kind but `read`, whose spans score as they are |
| `temperature.<kind>.<bound>` | float > 0 | The temperature of a request of at most `bound` questions whose kinds' temperatures are bucketed (the first such bucket); other requests take `temperature.<kind>` |
| `calibration.<name>.raw`, `.calibrated` | floats, paired, the side read from nondecreasing | A monotone table: a raw value's calibrated one by linear interpolation, held flat outside the knots |
| `calibration.<name>.inverted` | bool | Whether the table is read backwards: a calibrated value's raw one |
| `image_placements` | strings among `front`, `state`, `instructions`, `options` | Where the model reads images; none when absent |
| `image_max_pixels` | integer > 0 | A larger picture is downscaled to this many pixels before its processor |

| Kind | Answer |
| --- | --- |
| `noul` | P(true): the probability of option `noul.true_option` |
| `choice` | Every option's probability; the choice is the likeliest; TypeSafe's choice confidence (the peak rescaled from uniform to certainty) |
| `score` | Every level's probability; the level, their probability-weighted mean; TypeSafe's score confidence (one less the mean distance from the modal level over the uniform distribution's) |
| `multiple` | Each option's own P(true): a slot per option, at the slot anchors, read as a noul (`noul.*`) through `temperature.multiple` and `calibration.multiple`; a multiple choice is a choice, its options bounded by `choice.*` |
| `read` | A span of the state: a start and an end slot, at the slot anchors, scoring, by `h·h_j/√d`, the positions that write the state's text and position 0 (nothing); P(found), one less position 0's share of the start, through `calibration.found_<value>`; when that is at least one half, the likeliest start and end at most `read.<value>.max_tokens` apart, its text, and the number or date it writes |
| `estimate` | A number over the request's range: a grid of its values a step apart, or `estimate.max_values` of them (the most it reads), read at one slot through `temperature.estimate`; the median and the 10th and 90th percentiles at the levels `calibration.estimate` gives |
| `box` | The box from four edge slots (top, bottom, left, right) at the slot anchors, each reading `box.bins` codes (`box.grid`, `box.numbers`) through `temperature.box`; its confidence, the share of drawn boxes that overlap it, through `calibration.box`. With `box.presence`, a presence slot read as a noul comes first, P(present) through `calibration.found_box`, and a box is drawn only when that is at least one half |
| `boxes` | Every instance: `boxes.max` blocks (fewer when the question names a `max`, at most this) of a presence slot and four edges, at the slot anchors, each read as `box.*` reads a box with its presence, through `temperature.boxes`; presence through `calibration.found_boxes` kept falling block by block, the boxes the blocks before the first below one half, in reading order, each confident through `calibration.boxes` |

| Box key | Value | Meaning |
| --- | --- | --- |
| `box.presence` | bool | Whether a presence slot, read as a noul, precedes the edges (the kind `noul` listed) |
| `box.bins` | integer ≥ 2 | Bins of each edge distribution |
| `box.positions` | `centres`, `ends`, `starts` | Bin `k` of `n` lies at `(k + 1/2) / n`, `k / (n - 1)` or `k / n` across the picture |
| `box.edge` | `mean`, `median` | An edge's value: the probability-weighted mean position, or the first whose cumulative share reaches one half |
| `box.crossed` | `clamp`, `shift` | Two edges read out of order become one bin about their middle, cut at the picture's sides (regrown inside when that empties it) or moved inside them |
| `box.points` | `r4`, `halton` | The low-discrepancy points boxes are drawn at: R4 from 1/2, or Halton in bases 2, 3, 5, 7 |
| `box.draws` | integer > 0 | Boxes drawn |
| `box.drawn` | `valid`, `ordered` | A drawn box whose edges cross is a miss, or is ordered as the box's own edges are |
| `box.iou` | float in (0, 1] | The overlap a drawn box needs to count |

| Read key | Value | Meaning |
| --- | --- | --- |
| `read.text.max_tokens`, `read.number.max_tokens`, `read.date.max_tokens` | integer > 0 | The longest span of each value kind, in tokens |

| Estimate key | Value | Meaning |
| --- | --- | --- |
| `estimate.max_values` | integer ≥ 2 | The most values an estimate's grid reads; a finer step is read on this many |
| `estimate.slots` | `last`, `anchors` | As `<kind>.slots` |

A number or date read is parsed in the state's text by the engine's value
readings (`src/values.rs`): decimal marks and digit groups settled by the
writing and its context, currency and percent signs, number words, and
dates in the common written orders and languages.

## Template

The template renders one sequence's prompt: a state and the questions of
one set. Its text is control text and its
context is literal request text. The context (`src/prompt.rs`):

- `probe`: false (true only for the probe, `codes.probe`);
- `state`: content;
- `images`: the indices of the images read before the state;
- `questions`: the rendering's questions, each with `kind` (`noul`,
  `choice`, `multiple`, `score`, `read`, `estimate`, `box`, `boxes`),
  `instructions` (content), `codes` (what its answer slots read: one per
  option, level, grid value or bin; a noul's two, which a multiple choice's
  options and a box's presence also read); a `noul` has `truthy` and
  `falsy`, a `choice` and a `multiple` `options` (`key`, the key as
  `label` content, `description`), a `score` `levels`, a `read` `value`
  (`text`, `number`, `date`) and `unit` (content), an `estimate` `unit` and
  `grid` (`low`, `high` and `step` as Python's `format(value, "g")`,
  `size`, and `numbers`, whether its codes are its values), a `box` and
  `boxes` `grid` (`size`, `numbers`), `boxes` `max` (the blocks it reads) and `most` (`boxes.max`);
- `request_questions`: every question of the request, likewise.

A content value has `kind` (`none`, `string`, `json`) and `truthy` (Python's
truthiness, for a reference's `value or default`); a string has `text`
(as written) and `stripped` (Python's `str.strip()`); every value has `line`
and `indented` (Python's `json.dumps` with `ensure_ascii=False`, without and
with `indent=2`). Each form is a list of parts: request text (escaped), or
`{"image": k, "whole": w}`, `w` when the image is a whole JSON string written
without its quotes.

The template marks positions with anchors in its own text, `U+FDD0 name
U+FDD1`, which the engine removes: `state` once, where the state ends and
the first question starts; `question:<i>` where each further question `i`
starts (each question's rows run to the next one's start, the last to the
prompt's end); `slot:<i>` where a slot of the rendering's question `i`
ends, the slot read at the token holding the byte before it, the rest of
which may only be whitespace; `image:<k>` where image `k` goes, replaced by
the family's image placeholder; and `read:<form>` where the state's
`<form>` (`text`, `line`, `indented`) goes, placed by the engine as request
text, escaped, which a read's spans index and its values are parsed in. Request text never forms an anchor, since only template text
is read for them.

## Models

| Model | Template | Description |
| --- | --- | --- |
| Base Labs mammoth-1, elephant-1 (OneJev `qev` `81ce62f`, `serve.patch`) | `decision-templates/qev-labels-v2.jinja` | 1 / state / question; `output` with `labels` (`readout.json` `n_slots` of qev's slot labels); every kind `last`; noul true option 0; up to 255 options and 1 to 10 levels (qev's schema); temperature 1.0 |
| Perplexity pplx-decider Flash v1.2 (`9ff7cf9`, protocol 3) | `decision-templates/joint-question-inline-readout-v1.jinja` | 0 / set / set; `head` with `labels` (`decision_config.json` `codes`), every kind at its `<\|fim_suffix\|>` anchor; `fullwidth` over `escape.spellings` (the tokenizer's `all_special_tokens` and the marker); noul true option 1; up to 255 options, 2 to 255 levels; images before the state |
| vLLM-SR d3-ultra (`52ba9e7`, `d3-code-readout-v1`, `header-v1`) | `decision-templates/d3-header-v1.jinja` | 1 / state / question; `head` with 255 probed letter codes; noul, choice and score at their last token, box at four edge anchors; noul true option 1; up to 255 options, 2 to 10 levels; temperatures by question count (`temperature_by_questions`) and the box's base; box 128 `centres` bins, `mean` edges, `clamp`, 512 `r4` draws, `valid`, IoU 0.5, `calibration.box` (`box_calibration`); images before and inside the state, in instructions and options |
| Liquid d1-2B (`97c5704`, prompt 1, codes scheme 2) | `decision-templates/d1-prompt-v1.jinja` | `zero_width` over control tokens of 2 characters or more; `per_question`; 16 / set / question (`d1_set_size`, `d1_attention` `set`); `output` with `aliases` and `keys_if_letters`; noul `yes_no`, true option 0; choice up to 255 options; score `digits`, 2 to 10 levels; noul, choice and score at their last token through `calibration.json` `default`, every typed kind raw (1.0); `multiple` with `calibration.multiple`; `read` spans of 512, 32 and 32 tokens; `estimate` on at most 128 values at its last token, `calibration.estimate` inverted; box with presence, `median` edges, `shift`, 512 `halton` draws, `ordered`, IoU 0.5, as 1000 `starts` bins read as `numbers` (`d1_numbers`; 128 `ends` bins without it), its grids, an estimate's too, posed as `levels` (`grid_question`); `boxes` of 16; the nine `calibration.json` tables; images before the state, at most `d1_vision_max_pixels` |
