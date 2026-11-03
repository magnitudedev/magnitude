"""Golden answer arithmetic for d1-2B, from LiquidAI/d1-2B `97c5704`'s own functions, without a tokenizer.

Writes `fixtures/d1_arithmetic.json`, which the crate's arithmetic tests (`src/lib.rs`) read: `parsing.parse_number`,
`parsing.parse_date` and `parsing.written_unit` over written values and their states; `math.fsum`; and
`box.answer`, `box.answer_many` and `Grid.read` over synthetic distributions through d1-2B's
`calibration.json`. Python 3.11; numpy only (`prompt.py`).

    python d1_arithmetic.py <d1-2B repository> <out.json>
"""

from __future__ import annotations

import json
import math
import random
import sys
import types

NUMBERS = [
    ("1,234.50", "", None), ("1.234,50 €", "", None), ("1.234,50", "", "EUR"), ("$1.2M", "", None),
    ("(12)", "", None), ("-(12)", "", None), ("12k", "", None), ("12k", "the price is", None),
    ("2.5 million", "", None), ("3 Billion", "", None), ("3 billion", "", None), ("1,250", "", None),
    ("1,250", "total $3.50", None), ("1,250", "locale: fr", None), ("1 234,50", "", None), ("12'345.5", "", None),
    ("1,23,45,678", "₹ amounts", None), ("1,23,45,678", "", None), ("٣٤", "", None), ("1.234.567", "", None),
    ("0,5", "", None), ("12.5%", "", None), ("−7", "", None), ("+7", "", None), ("USD 1,000", "", None),
    ("1,000 kr", "", None), ("¥1,000", "", None), ("12,345.678.9", "", None), ("1.2.3", "", None),
    ("000123", "", None), ("1" * 40, "", None), ("1" * 30 + ".5", "", None), ("9" * 29, "", None),
    ("0.000", "", None), ("-0.0", "", None), ("12 mil", "", None), ("4 lakh", "", None), ("3 trillion", "", None),
    ("1.5bn", "revenue", None), ("7B", "users", None), ("hello", "", None), ("", "", None), ("()", "", None),
    ("1 000 000", "", None), ("1 000", "", None), ("1,5", "<?xml version=\"1.0\"?> 3,25", None),
    ("1.250", "lang=\"de\"", None), ("CHF 12.50", "", None), ("12.50 CHF", "", None), ("R$ 1.234,56", "", None),
]
DATES = [
    ("2026-10-20", ""), ("20 octobre 2026", ""), ("October 20th, 2026", ""), ("Tue, 20 Oct 2026", ""),
    ("03/04/2026", ""), ("03/04/2026", "on 25/12/2025"), ("03/04/2026", "date format: mm/dd/yyyy"),
    ("03/04/2026", "locale: en-US"), ("13/04/2026", ""), ("03/04/26", "since 2019-01-01"), ("03/04/26", ""),
    ("2026年10月20日", ""), ("2026년 10월 20일", ""), ("2026-10-20 9:30", ""), ("2026-10-20T09:30:15.5Z", ""),
    ("2026-10-20 10:00+02:00", ""), ("2026-10-20 25:00", ""), ("2026-02-30", ""), ("29 février 2024", ""),
    ("segunda-feira, 20 de outubro de 2026", ""), ("1 мая 2026", ""), ("Sept 3 2026", ""), ("3 mar 2026", ""),
    ("20.10.2026", "tt.mm.jjjj"), ("10-20-2026", ""), ("1st of May 2026", ""), ("May 2026", ""), ("", ""),
    ("2026-10-20 10:00 11:00", ""), ("12 März 2026", ""), ("12 de marzo de 2026", ""), ("0000-01-01", ""),
]
UNITS = ["12%", "12 €", "€12 and $3", "USD 5", "5 usd", "5", "kr 12", "12 % ", "R$ 5"]


def main():
    reference, target = sys.argv[1:3]
    package = types.ModuleType("d1")
    package.__path__ = [reference]
    sys.modules["d1"] = package
    from d1 import box
    from d1.calibration import Calibration
    from d1.grid import Grid
    from d1.parsing import parse_date, parse_number, written_unit

    calibration = Calibration.load(f"{reference}/calibration.json")
    rng = random.Random("d1 arithmetic")

    def number(value):
        return None if value is None else {"integer": str(value)} if isinstance(value, int) else {"float": value}

    def peaked(size):
        center, width = rng.uniform(0, size), rng.uniform(1, size / 4)
        scores = [-0.5 * ((i - center) / width) ** 2 + rng.gauss(0, 0.3) for i in range(size)]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        return [e / sum(exps) for e in exps]

    def block():
        present = rng.uniform(0.0, 1.0)
        return [[present, 1 - present], *(peaked(box.GRID.size) for _ in box.EDGES)]

    boxes = []
    for blocks in (None, 1, 3, 3):
        if blocks is None:
            read = block()
            boxes.append({"blocks": None, "probabilities": read, "answer": box.answer(read, calibration.through)})
            continue
        many = [row for _ in range(blocks) for row in block()]
        boxes.append({"blocks": blocks, "probabilities": many, "answer": box.answer_many(many, calibration.through)})

    estimates = []
    for low, high, size in ((0, 10, 11), (0, 1, 5), (-3.5, 2.25, 128), (0, 1000, 128)):
        grid = Grid(low, high, size)
        probabilities = peaked(size)
        read = grid.read(probabilities, lambda share: calibration.level("estimate", share))
        estimates.append({"grid": [low, high, size], "probabilities": probabilities,
                          "read": [read["p10"], read["median"], read["p90"]]})

    sums = []
    for _ in range(20):
        values = [rng.choice([rng.uniform(-1, 1), rng.uniform(-1e16, 1e16), rng.uniform(0, 1e-16)])
                  for _ in range(rng.randrange(1, 40))]
        sums.append({"values": values, "sum": math.fsum(values)})
    sums.append({"values": [1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50], "sum": math.fsum(
        [1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50])})

    fixture = {
        "numbers": [{"text": text, "context": context, "unit": unit, "number": number(parse_number(text, context, unit=unit))}
                    for text, context, unit in NUMBERS],
        "dates": [{"text": text, "context": context, "date": parse_date(text, context)[0]} for text, context in DATES],
        "units": [{"text": text, "unit": written_unit(text)} for text in UNITS],
        "sums": sums,
        "tables": calibration.tables,
        "boxes": boxes,
        "estimates": estimates,
    }
    with open(target, "w") as f:
        json.dump(fixture, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
