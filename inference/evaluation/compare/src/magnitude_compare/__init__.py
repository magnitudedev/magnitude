"""compare: did a change make it better or worse? A base build against a candidate build (and
each against an independent reference) on the same host, for quality and speed.

Controller (`cli`, `controller`, `builds`): sides, inputs and one remote job per model.
Host (`runner`, `steps`, `reference`): the harness runs, references and judgement.
Judgement (`quality`, `speed`, `rendering`, `report`): pure functions over result records and
logits outputs.
"""
